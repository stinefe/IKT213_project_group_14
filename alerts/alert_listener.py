"""The "nurse station": listens for alerts from every fall-detection camera.

Run it on any computer that can reach the broker:
    python alert_listener.py                  # broker on this computer
    MQTT_HOST=192.168.1.20 python alert_listener.py

Prints falls and recoveries, and warns when a camera goes quiet or offline.
"""
import json
import os
import threading
import time

import paho.mqtt.client as mqtt

MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
SILENT_AFTER = 30   # seconds without a heartbeat -> "camera silent" warning

last_seen = {}      # device -> time of last status message
warned = set()      # devices we already warned about (warn once, not every second)


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"Connected to {MQTT_HOST}:{MQTT_PORT}, waiting for alerts...")
    client.subscribe("falldet/+/alert", qos=1)    # + = any device name
    client.subscribe("falldet/+/status", qos=1)


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload)
    except ValueError:
        return
    device = msg.topic.split("/")[1]

    if msg.topic.endswith("/alert"):
        if data.get("event") == "fall":
            print("\a")  # terminal beep
            print("=" * 50)
            print(f"  FALL DETECTED   device={device}   time={data.get('time')}")
            print("=" * 50)
        elif data.get("event") == "recovered":
            print(f"  recovered       device={device}   down for {data.get('seconds_down')} s")

    else:  # status / heartbeat
        state = data.get("state")
        if state in ("offline", "stopped"):
            print(f"  !! camera {device} is {state.upper()}")
            last_seen.pop(device, None)
        else:
            if device not in last_seen or device in warned:
                print(f"  camera {device} online (state={state}, fps={data.get('fps')})")
            last_seen[device] = time.time()
            warned.discard(device)


def watch_silence():
    """A camera that stops sending heartbeats is a problem, even without 'offline'."""
    while True:
        now = time.time()
        for device, seen in list(last_seen.items()):
            if now - seen > SILENT_AFTER and device not in warned:
                print(f"  !! camera {device} silent for {int(now - seen)} s")
                warned.add(device)
        time.sleep(5)


def main():
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:          # paho-mqtt 1.x
        client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    threading.Thread(target=watch_silence, daemon=True).start()
    client.loop_forever()


if __name__ == "__main__":
    main()