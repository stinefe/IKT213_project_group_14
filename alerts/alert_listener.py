"""Laptop side: listen for fall alerts from the Jetson (or the test sender)."""
import json
import paho.mqtt.client as mqtt
from falldet.config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC


def on_connect(client, userdata, flags, reason_code, properties):
    print("Connected:", reason_code)
    client.subscribe(MQTT_TOPIC)


def on_message(client, userdata, msg):
    try:
        alert = json.loads(msg.payload.decode())
        print(f"\a⚠️  FALL DETECTED in {alert['room']} at {alert['time']}")
    except (json.JSONDecodeError, KeyError):
        print("Ignored unexpected message:", msg.payload.decode(errors="replace"))


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever()