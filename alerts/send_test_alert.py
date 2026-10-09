"""Pretend to be the Jetson: send a fake fall alert over MQTT."""
import json
import time
import paho.mqtt.publish as publish
from falldet.config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, MQTT_STATUS_TOPIC

message = {"event": "fall", "room": "Room 1", "time": time.strftime("%H:%M:%S")}
publish.single(MQTT_TOPIC, json.dumps(message), hostname=MQTT_BROKER, port=MQTT_PORT)
print("Sent:", message)
publish.single(MQTT_STATUS_TOPIC, "fall", hostname=MQTT_BROKER, port=MQTT_PORT)