"""MQTT alerts for live fall detection (issue #11).

The detector PUBLISHES messages; any listener (alert_listener.py, a phone app,
Home Assistant, ...) SUBSCRIBES. The broker (Mosquitto) is the post office in between.

Topics (DEVICE_ID lets several cameras share one broker):
  falldet/<device>/alert   {"event": "fall" | "recovered", ...}   QoS 1, not retained
  falldet/<device>/status  {"state": ..., "fps": ..., ...}        retained heartbeat
                           {"state": "offline"}                   sent by the BROKER
                                                                  if the detector dies

Settings come from environment variables, so laptop and Docker need no code changes:
  MQTT_HOST (default localhost), MQTT_PORT (1883), DEVICE_ID (computer name)
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
        # Last Will: if this program crashes or the Jetson loses power, the BROKER
        # publishes "offline" for us. A silent camera must never look like "all OK".
        self.client.will_set(STATUS_TOPIC, json.dumps({"state": "offline"}), qos=1, retain=True)
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

    def start(self):
        """Connect in the background. Detection keeps running even if the broker is down."""
        try:
            self.client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=30)
            self.client.loop_start()   # background thread: sends, receives, reconnects
            print(f"MQTT: publishing to {MQTT_HOST}:{MQTT_PORT} as '{DEVICE_ID}'")
        except Exception as error:     # bad host name etc. -> warn, never crash detection
            print(f"MQTT: could not start ({error}); alerts will only be printed")

    def _publish(self, topic, payload, retain=False):
        payload["device"] = DEVICE_ID
        payload["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        message = json.dumps(payload)
        # QoS 1 = "deliver at least once": paho keeps the message and resends it
        # after a reconnect, so a short network drop does not lose the alarm.
        self.client.publish(topic, message, qos=1, retain=retain)
        return message

    def fall(self, torso_angle=None):
        msg = self._publish(ALERT_TOPIC, {"event": "fall", "torso_angle": torso_angle})
        print("ALERT ->", msg)

    def recovered(self, seconds_down):
        msg = self._publish(ALERT_TOPIC, {"event": "recovered",
                                          "seconds_down": round(seconds_down, 1)})
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