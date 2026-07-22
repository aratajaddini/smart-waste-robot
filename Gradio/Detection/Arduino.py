import serial
import time


def connect():
    try:
        arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)
        time.sleep(2)  
        print("✅ Connected Successfully.")
    except Exception as e:
        arduino = None
        print(f"⚠️ Port Not Found : {e}")

    SERIAL_COMMANDS = {
        "Plastic": "1",
        "Glass": "2",
        "Metal": "3",
        "Paper/Cardboard": "4",
        "Waste": "5",
    }
    return arduino , SERIAL_COMMANDS

def send_to_arduino(target_class,SERIAL_COMMANDS,arduino):
    if arduino and target_class in SERIAL_COMMANDS:
        cmd = SERIAL_COMMANDS[target_class]
        arduino.write(cmd.encode('utf-8'))  
        print(f"📡 Command Sent To Arduino: {cmd} For {target_class}")