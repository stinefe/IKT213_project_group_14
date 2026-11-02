"""MQTT alerts for live fall detection (issue #11).

The detector PUBLISHES messages; any listener (alert_listener.py, a phone app,
...) SUBSCRIBES. The broker (EMQX Cloud or local Mosquitto) is the post office in between.

Topics (DEVICE_ID lets several cameras share one broker):
  falldet/<device>/alert   {"event": "fall" | "recovered", ...}   QoS 1, not retained
  falldet/<device>/status  {"state": ..., "fps": ..., ...}        retained heartbeat
                           {"state": "offline"}                   sent by the BROKER
                                                                  if the detector dies
  falldet/<device>/led     "fall" | "clear"                       retained: the CURRENT alarm
                                                                  state, for the phone's LED
                                                                  and Acknowledge button
  falldet/<device>/ack     anything                               sent BY the phone when the
                                                                  nurse presses Acknowledge

Alarm flow:
  fall confirmed      -> led = "fall"   (LED red, button green)
  nurse presses ack   -> led = "clear"  (LED off, button grey)  + acknowledged = True
  person gets up      -> led = "clear"

Settings come from environment variables, so laptop and Docker need no code changes:
  MQTT_HOST (default localhost), MQTT_PORT (1883), DEVICE_ID (computer name),
  MQTT_USER / MQTT_PASS (cloud broker: login + TLS)
"""
import json
import os
import socket
import time

import paho.mqtt.client as mqtt

MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
DEVICE_ID = os.environ.get("DEVICE_ID", socket.gethostname())

ALERT_TOPIC = f"falldet/{DEVICE_ID}/alert"
STATUS_TOPIC = f"falldet/{DEVICE_ID}/status"
LED_TOPIC = f"falldet/{DEVICE_ID}/led"
ACK_TOPIC = f"falldet/{DEVICE_ID}/ack"
HEARTBEAT_SECONDS = 10


def _new_client():
    """paho-mqtt 2.x needs a callback version; 1.x does not have that argument."""
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"falldet-{DEVICE_ID}")
    except AttributeError:
        return mqtt.Client(client_id=f"falldet-{DEVICE_ID}")


class AlertClient:
    """Small wrapper so live_detect.py only calls fall(), recovered() and heartbeat()."""

    def __init__(self):
        self.client = _new_client()
        self.last_heartbeat = 0.0
        self.alarm_active = False   # True between a confirmed fall and ack / getting up
        self.acknowledged = False   # True once the nurse pressed Acknowledge for this fall

        # Last Will: if this program crashes or the Jetson loses power, the BROKER
        # publishes "offline" for us. A silent camera must never look like "all OK".
        self.client.will_set(STATUS_TOPIC, json.dumps({"state": "offline"}), qos=1, retain=True)
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

        # Cloud broker (EMQX / HiveMQ): login + encryption.
        # Only used when MQTT_USER is set, so the local Mosquitto setup still works as before.
        user = os.environ.get("MQTT_USER")
        if user:
            self.client.username_pw_set(user, os.environ.get("MQTT_PASS"))
            self.client.tls_set()      # encrypt the connection (needs port 8883)

        # Listen for the phone's Acknowledge button
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    # ---- connection ---------------------------------------------------------
    def start(self):
        """Connect in the background. Detection keeps running even if the broker is down."""
        try:
            self.client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=30)
            self.client.loop_start()   # background thread: sends, receives, reconnects
            self._set_led("clear")     # new run: no alarm left over from an earlier test
            print(f"MQTT: publishing to {MQTT_HOST}:{MQTT_PORT} as '{DEVICE_ID}'")
        except Exception as error:     # bad host name etc. -> warn, never crash detection
            print(f"MQTT: could not start ({error}); alerts will only be printed")

    def _on_connect(self, client, *args):
        # Runs after every (re)connect, so the subscription survives network drops.
        client.subscribe(ACK_TOPIC, qos=1)

    def _on_message(self, client, userdata, msg):
        if msg.topic == ACK_TOPIC and not self.alarm_active:
            print("ACK received, but no active alarm -> ignored "
                  "(already acknowledged, or no new fall since)")
        if msg.topic == ACK_TOPIC and self.alarm_active:
            self.acknowledged = True
            self.alarm_active = False
            self._set_led("clear")     # LED off + button back to grey
            self._publish(ALERT_TOPIC, {"event": "acknowledged",
                                        "text": f"{DEVICE_ID}: nurse is on the way"})
            print("ACK -> nurse acknowledged the fall")

    # ---- sending ------------------------------------------------------------
    def _publish(self, topic, payload, retain=False):
        payload["device"] = DEVICE_ID
        payload["time"] = time.strftime("%d.%m.%Y %H:%M:%S")   # e.g. 03.11.2026 00:00:24
        message = json.dumps(payload)
        # QoS 1 = "deliver at least once": paho keeps the message and resends it
        # after a reconnect, so a short network drop does not lose the alarm.
        self.client.publish(topic, message, qos=1, retain=retain)
        return message

    def _set_led(self, value):
        # Plain text + retained: the broker REMEMBERS it, so a phone that opens the
        # app later still sees the current state (red LED / green button).
        self.client.publish(LED_TOPIC, value, qos=1, retain=True)

    def fall(self):
        self.alarm_active = True
        self.acknowledged = False
        msg = self._publish(ALERT_TOPIC, {"event": "fall",
                                          "text": f"FALL detected in {DEVICE_ID}"})
        self._set_led("fall")          # LED red + button green
        print("ALERT ->", msg)

    def recovered(self, seconds_down):
        self.alarm_active = False
        msg = self._publish(ALERT_TOPIC, {"event": "recovered",
                                          "seconds_down": round(seconds_down, 1),
                                          "text": f"{DEVICE_ID}: person is up again "
                                                  f"(down {round(seconds_down)} s)"})
        self._set_led("clear")
        print("ALERT ->", msg)

    def heartbeat(self, state, fps):
        """Call every frame; only actually sends every HEARTBEAT_SECONDS."""
        now = time.time()
        if now - self.last_heartbeat >= HEARTBEAT_SECONDS:
            self._publish(STATUS_TOPIC, {"state": state, "fps": round(fps, 1)}, retain=True)
            self.last_heartbeat = now

    def stop(self):
        """Clean shutdown: say 'stopped' (not 'offline'), then disconnect."""
        self._publish(STATUS_TOPIC, {"state": "stopped"}, retain=True)
        time.sleep(0.3)                # give the background thread time to send it
        self.client.loop_stop()
        self.client.disconnect()