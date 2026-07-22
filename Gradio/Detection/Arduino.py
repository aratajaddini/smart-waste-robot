import serial
import serial.tools.list_ports
import time

SERIAL_COMMANDS = {
    "Plastic": "1",
    "Glass": "2",
    "Metal": "3",
    "Paper/Cardboard": "4",
    "Waste": "5",
}

def connect(default_port='COM3', baudrate=9600):
   
    arduino = None
    status_msg = ""
  
    try:
        arduino = serial.Serial(port=default_port, baudrate=baudrate, timeout=1)
        time.sleep(2) 
        status_msg = f"🟢 Connected Successfully on {default_port}"
        print(f"✅ {status_msg}")
        return arduino, SERIAL_COMMANDS, status_msg
    except Exception as e:
        print(f"⚠️ {default_port} Not Found: {e}. Searching other ports...")

   
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        try:
            arduino = serial.Serial(port=p.device, baudrate=baudrate, timeout=1)
            time.sleep(2)
            status_msg = f"🟢 Auto-Connected on {p.device}"
            print(f"✅ {status_msg}")
            return arduino, SERIAL_COMMANDS, status_msg
        except Exception:
            continue

    status_msg = "🔴 Disconnected (No Arduino Port Found)"
    print(f"❌ {status_msg}")
    return None, SERIAL_COMMANDS, status_msg


def send_to_arduino(target_class, SERIAL_COMMANDS, arduino):
   
    if arduino and arduino.is_open and target_class in SERIAL_COMMANDS:
        try:
            cmd = SERIAL_COMMANDS[target_class]
            arduino.write(cmd.encode('utf-8'))  
            print(f"📡 Command Sent To Arduino: {cmd} For {target_class}")
            return True
        except Exception as e:
            print(f"❌ Failed to send command via Serial: {e}")
            return False
    else:
        print(f"⚠️ Arduino is not connected. Skipped command for {target_class}")
        return False