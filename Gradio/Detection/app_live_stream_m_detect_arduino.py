import gradio as gr
import cv2
import time
import pandas as pd
from ultralytics import YOLO
import numpy as np
import Arduino 


arduino, SERIAL_COMMANDS, status_msg = Arduino.connect()


MODEL_PATH = "model_path"
model = YOLO(MODEL_PATH)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


CLASS_MAPPING = {
    "Glass": "Glass",
    "Metal": "Metal",
    "Paper": "Paper/Cardboard",
    "Plastic": "Plastic",
    "Waste": "Waste",
}

ENVIRONMENTAL_PRIORITY = {
    "Metal": 1,
    "Plastic": 2,
    "Glass": 3,
    "Paper/Cardboard": 4,
    "Waste": 5,
}

WASTE_VALUES = {
    "Metal": 5.00,
    "Plastic": 2.50,
    "Glass": 1.50,
    "Paper/Cardboard": 1.00,
    "Waste": 0.50,
}

BIN_CAPACITIES = {
    "Plastic": 100,
    "Glass": 100,
    "Metal": 100,
    "Paper/Cardboard": 100,
    "Waste": 100,
}


TRIGGER_LINE_RATIO = 0.50
TRIGGER_TOLERANCE = 30
STATIONARY_PIXEL_THRESHOLD = 25

last_processed_object = {
    "center_x": -999,
    "center_y": -999,
    "class": None,
    "timestamp": 0
}

system_metrics = {
    "total_count": 0,
    "Plastic": 0,
    "Glass": 0,
    "Metal": 0,
    "Paper/Cardboard": 0,
    "Waste": 0,
    "confidence_sum": 0.0,
    "total_revenue": 0.0,
}

sorting_timestamps = []
log_history = []
time_series_data = {"Time": [], "Total Sorted": []}
start_time = time.time()



def handle_reconnect():

    global arduino, SERIAL_COMMANDS
    if arduino and hasattr(arduino, 'is_open') and arduino.is_open:
        try:
            arduino.close()
        except Exception:
            pass
            
    arduino, SERIAL_COMMANDS, new_status = Arduino.connect()
    return new_status


def reset_system_metrics():

    global system_metrics, sorting_timestamps, log_history, time_series_data, start_time, last_processed_object
    
    system_metrics = {
        "total_count": 0,
        "Plastic": 0,
        "Glass": 0,
        "Metal": 0,
        "Paper/Cardboard": 0,
        "Waste": 0,
        "confidence_sum": 0.0,
        "total_revenue": 0.0,
    }
    
    sorting_timestamps.clear()
    log_history.clear()
    time_series_data = {"Time": [], "Total Sorted": []}
    start_time = time.time()
    last_processed_object = {"center_x": -999, "center_y": -999, "class": None, "timestamp": 0}
    
    reset_log = "🔄 System Metrics, Spatial Counters & Bin Capacities Resetted!"
    log_history.append(reset_log)


    empty_rates_data = {
        "Waste Category": ["Plastic", "Glass", "Metal", "Paper/Cardboard", "Waste"],
        "Sorted Count": [0, 0, 0, 0, 0],
        "Share (%)": ["0%", "0%", "0%", "0%", "0%"],
        "Bin Capacity": [
            f"0/{BIN_CAPACITIES['Plastic']}",
            f"0/{BIN_CAPACITIES['Glass']}",
            f"0/{BIN_CAPACITIES['Metal']}",
            f"0/{BIN_CAPACITIES['Paper/Cardboard']}",
            f"0/{BIN_CAPACITIES['Waste']}",
        ],
    }
    df_rates = pd.DataFrame(empty_rates_data)
    df_chart = pd.DataFrame({"Time": [0], "Total Sorted": [0]})


    return (
        "0.0%",     
        "0 WPM",     
        "0.0%",      
        "$0.00",      
        df_rates,    
        reset_log,   
        df_chart,  
    )


def advanced_robot_logic(detected_objects, frame_height):
    global last_processed_object, log_history
    
    if not detected_objects:
        return None, "Conveyor belt is empty in this frame."

    trigger_y = int(frame_height * TRIGGER_LINE_RATIO)
    current_time = time.time()
    valid_objects = []

    for obj in detected_objects:
        obj_class = obj['class']
        center_x = obj['center_x']
        center_y = obj['center_y']

       
        current_fill = system_metrics[obj_class]
        max_cap = BIN_CAPACITIES[obj_class]
        if current_fill >= max_cap:
            timestamp = time.strftime('%H:%M:%S')
            log_msg = f"[{timestamp}] ⚠️ WARNING: Bin [{obj_class}] is FULL!"
            if log_msg not in log_history[:3]:
                log_history.insert(0, log_msg)
                log_history = log_history[:50]
            continue

        
        if abs(center_y - trigger_y) > TRIGGER_TOLERANCE:
            continue

   
        dist_to_last = np.sqrt((center_x - last_processed_object["center_x"])**2 + (center_y - last_processed_object["center_y"])**2)
        if dist_to_last < STATIONARY_PIXEL_THRESHOLD and obj_class == last_processed_object["class"]:
            continue

        valid_objects.append(obj)

    if not valid_objects:
        return None, "Monitoring conveyor belt (Conveyor Stalled / Cooldown Active)..."

   
    sorted_queue = sorted(
        valid_objects,
        key=lambda x: (
            ENVIRONMENTAL_PRIORITY.get(x['class'], 99),
            -x['confidence'],
        ),
    )
    target_object = sorted_queue[0]

    last_processed_object["center_x"] = target_object['center_x']
    last_processed_object["center_y"] = target_object['center_y']
    last_processed_object["class"] = target_object['class']
    last_processed_object["timestamp"] = current_time

    timestamp = time.strftime('%H:%M:%S')
    
    cx, cy = target_object['center_x'], target_object['center_y']
    bbox = target_object['bbox']
    
    log_msg = (
        f"[{timestamp}] 🤖 COMMAND: Sort [{target_object['class']}] ({target_object['confidence']:.1%}) "
        f"| Pos: Center({cx}, {cy}) BBox{bbox} -> Sent To Arduino"
    )

    if len(sorted_queue) > 1:
        suspended_items = [obj['class'] for obj in sorted_queue[1:]]
        log_msg += f" | ⚠️ Lower priority suspended: {suspended_items}"

    return target_object, log_msg


def process_stream_pipeline(video_frame):
    global log_history
    
    if video_frame is None:
        return None, "0.0%", "0 WPM", "0.0%", "$0.00", pd.DataFrame(), "", pd.DataFrame()

    frame = np.array(video_frame)
    frame_h, frame_w, _ = frame.shape
    trigger_y = int(frame_h * TRIGGER_LINE_RATIO)


    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    enhanced_frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

    current_time = time.time()
    results = model(enhanced_frame, verbose=False)[0]
    detected_batch = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        raw_class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if raw_class_name in CLASS_MAPPING:
            mapped_class = CLASS_MAPPING[raw_class_name]
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            detected_batch.append({
                'class': mapped_class,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
                'center_x': center_x,
                'center_y': center_y
            })

            cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(enhanced_frame, (center_x, center_y), 4, (255, 0, 0), -1)
            cv2.putText(
                enhanced_frame,
                f"{mapped_class}: {confidence:.1%}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    target, control_log = advanced_robot_logic(detected_batch, frame_h)


    line_color = (0, 255, 0) if target else (255, 0, 0)
    cv2.line(enhanced_frame, (0, trigger_y), (frame_w, trigger_y), line_color, 2)
    cv2.putText(enhanced_frame, "TRIGGER LINE", (10, trigger_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1)

    if target:
        sent_success = Arduino.send_to_arduino(str(target['class']), SERIAL_COMMANDS, arduino)
        if not sent_success:
            control_log += " | ⚠️ Hardware Not Connected"

        system_metrics[target['class']] += 1
        system_metrics["total_count"] += 1
        system_metrics["confidence_sum"] += target['confidence']
        system_metrics["total_revenue"] += WASTE_VALUES.get(target['class'], 0.0)

        sorting_timestamps.append(current_time)
        log_history.insert(0, control_log)
        log_history = log_history[:50]

    total = system_metrics["total_count"]


    rates_data = {
        "Waste Category": ["Plastic", "Glass", "Metal", "Paper/Cardboard", "Waste"],
        "Sorted Count": [
            system_metrics["Plastic"],
            system_metrics["Glass"],
            system_metrics["Metal"],
            system_metrics["Paper/Cardboard"],
            system_metrics["Waste"],
        ],
        "Share (%)": [
            f"{(system_metrics['Plastic']/total)*100:.1f}%" if total > 0 else "0%",
            f"{(system_metrics['Glass']/total)*100:.1f}%" if total > 0 else "0%",
            f"{(system_metrics['Metal']/total)*100:.1f}%" if total > 0 else "0%",
            f"{(system_metrics['Paper/Cardboard']/total)*100:.1f}%" if total > 0 else "0%",
            f"{(system_metrics['Waste']/total)*100:.1f}%" if total > 0 else "0%",
        ],
        "Bin Capacity": [
            f"{system_metrics['Plastic']}/{BIN_CAPACITIES['Plastic']}",
            f"{system_metrics['Glass']}/{BIN_CAPACITIES['Glass']}",
            f"{system_metrics['Metal']}/{BIN_CAPACITIES['Metal']}",
            f"{system_metrics['Paper/Cardboard']}/{BIN_CAPACITIES['Paper/Cardboard']}",
            f"{system_metrics['Waste']}/{BIN_CAPACITIES['Waste']}",
        ],
    }
    df_rates = pd.DataFrame(rates_data)


    sorting_timestamps[:] = [t for t in sorting_timestamps if current_time - t <= 60]
    wpm = len(sorting_timestamps)
    wpm_speed = f"{wpm} WPM"

    avg_conf_raw = (system_metrics["confidence_sum"] / total) if total > 0 else 0.0
    avg_conf = f"{avg_conf_raw * 100:.1f}%"

    max_nominal_speed = 45.0
    performance_rate = min(wpm / max_nominal_speed, 1.0) if wpm > 0 else 1.0
    oee_score = (avg_conf_raw * performance_rate * 100) if total > 0 else 0.0
    oee_display = f"{oee_score:.1f}%"

    revenue_display = f"${system_metrics['total_revenue']:.2f}"


    elapsed_time = int(current_time - start_time)
    if not time_series_data["Time"] or elapsed_time != time_series_data["Time"][-1]:
        time_series_data["Time"].append(elapsed_time)
        time_series_data["Total Sorted"].append(total)
        if len(time_series_data["Time"]) > 60:
            time_series_data["Time"].pop(0)
            time_series_data["Total Sorted"].pop(0)

    df_chart = pd.DataFrame(time_series_data)
    logs_display = "\n".join(log_history[:8])

    return (
        enhanced_frame,
        avg_conf,
        wpm_speed,
        oee_display,
        revenue_display,
        df_rates,
        logs_display,
        df_chart,
    )


with gr.Blocks(title="Industrial Smart Waste Sorter", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏭 Industrial Smart Waste Sorting Automation System (Detection Model)
    """)

   
    with gr.Row():
        with gr.Column(scale=3):
            arduino_status_box = gr.Textbox(
                label="🔌 Hardware Serial Port Status (Arduino Connection)",
                value=status_msg,
                interactive=False,
            )
        with gr.Column(scale=1):
            btn_reconnect = gr.Button("⚡ Reconnect Hardware", variant="primary")


    with gr.Row():
        with gr.Column():
            metric_conf = gr.Textbox(label="🎯 Model Accuracy (Quality)", value="0.0%")
        with gr.Column():
            metric_speed = gr.Textbox(label="⚡ Real-time Speed (Performance)", value="0 WPM")
        with gr.Column():
            metric_oee = gr.Textbox(label="📊 Overall Effectiveness (OEE)", value="0.0%")
        with gr.Column():
            metric_rev = gr.Textbox(label="💵 Economic Value Generated", value="$0.00")

  
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📷 Live Conveyor Belt Monitoring")
            input_cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Input Stream")
            output_cam = gr.Image(show_label=False, type="numpy", label="Processed Stream")

        with gr.Column(scale=1):
            gr.Markdown("### 📈 Bin Capacities & Distribution")
            rates_table = gr.Dataframe(interactive=False)

            gr.Markdown("### 🧠 Robot Logic & Controller Logs (with Priority & Coordinates)")
            control_logs = gr.Textbox(lines=6, interactive=False, label="Priority Event & Location Log")
            btn_reset = gr.Button("🔄 Reset Metrics & Counters", variant="secondary")


    with gr.Row():
        live_chart = gr.LinePlot(
            x="Time",
            y="Total Sorted",
            title="Total Sorted Waste vs. Elapsed Time (seconds)",
            x_title="Elapsed Time (Seconds)",
            y_title="Total Units Sorted",
        )


    input_cam.stream(
        fn=process_stream_pipeline,
        inputs=input_cam,
        outputs=[
            output_cam,
            metric_conf,
            metric_speed,
            metric_oee,
            metric_rev,
            rates_table,
            control_logs,
            live_chart,
        ],
        queue=True,
        show_progress="hidden",
    )

    
    btn_reset.click(
        fn=reset_system_metrics,
        inputs=None,
        outputs=[
            metric_conf,
            metric_speed,
            metric_oee,
            metric_rev,
            rates_table,
            control_logs,
            live_chart,
        ],
    )

    btn_reconnect.click(fn=handle_reconnect, inputs=None, outputs=arduino_status_box)

if __name__ == "__main__":
    demo.launch(share=True)