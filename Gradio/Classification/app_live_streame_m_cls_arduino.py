import gradio as gr
import cv2
import time
import pandas as pd
from ultralytics import YOLO
import numpy as np
import Arduino 


arduino, SERIAL_COMMANDS, status_msg = Arduino.connect()


MODEL_PATH = "best.pt"
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

CONFIDENCE_THRESHOLD = 0.60
CLASSIFICATION_COOLDOWN = 1.5
last_classification_time = 0

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
    
    global system_metrics, sorting_timestamps, log_history, time_series_data, start_time, last_classification_time
    
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
    last_classification_time = 0
    
    reset_log = "🔄 System Metrics, KPIs & Bin Capacities Resetted!"
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


def classification_robot_logic(predicted_class, confidence):
    global last_classification_time, log_history
    
    current_time = time.time()

    if confidence < CONFIDENCE_THRESHOLD:
        return False, f"Low Confidence ({confidence:.1%}) - Ignored."

    if (current_time - last_classification_time) < CLASSIFICATION_COOLDOWN:
        return False, "Conveyor Moving / Classification Cooldown Active..."

    current_fill = system_metrics[predicted_class]
    max_cap = BIN_CAPACITIES[predicted_class]
    if current_fill >= max_cap:
        timestamp = time.strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] ⚠️ WARNING: Bin [{predicted_class}] is FULL!"
        if log_msg not in log_history[:3]:
            log_history.insert(0, log_msg)
            log_history = log_history[:50]
        return False, log_msg

    last_classification_time = current_time
    timestamp = time.strftime('%H:%M:%S')
    
    log_msg = (
        f"[{timestamp}] 🤖 Sort: [{predicted_class}] ({confidence:.1%}) "
        f"| Priority Grade: {ENVIRONMENTAL_PRIORITY.get(predicted_class, 'N/A')} -> Sent To Arduino"
    )

    return True, log_msg


def process_classification_pipeline(video_frame):
    global log_history
    
    if video_frame is None:
        return None, "0.0%", "0 WPM", "0.0%", "$0.00", pd.DataFrame(), "", pd.DataFrame()

    frame = np.array(video_frame)
    
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    enhanced_frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

    current_time = time.time()
    results = model(enhanced_frame, verbose=False)[0]
    
    top1_index = int(results.probs.top1)
    raw_class_name = model.names[top1_index]
    confidence = float(results.probs.top1conf)

    mapped_class = CLASS_MAPPING.get(raw_class_name, "Waste")

    cv2.putText(
        enhanced_frame,
        f"Class: {mapped_class} ({confidence:.1%})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (255, 0, 0),
        2,
    )

    is_valid_sort, control_log = classification_robot_logic(mapped_class, confidence)

    if is_valid_sort:
        sent_success = Arduino.send_to_arduino(str(mapped_class), SERIAL_COMMANDS, arduino)
        if not sent_success:
            control_log += " | ⚠️ Hardware Not Connected"

        system_metrics[mapped_class] += 1
        system_metrics["total_count"] += 1
        system_metrics["confidence_sum"] += confidence
        system_metrics["total_revenue"] += WASTE_VALUES.get(mapped_class, 0.0)

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



with gr.Blocks(title="Classification Waste Sorter Dashboard", theme=gr.themes.Soft()) as demo_cls:
    gr.Markdown("""
    # 🏷️ Industrial Smart Waste Classification System
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
            metric_conf = gr.Textbox(label="🎯 Model Classification Accuracy", value="0.0%")
        with gr.Column():
            metric_speed = gr.Textbox(label="⚡ Real-time Speed (Performance)", value="0 WPM")
        with gr.Column():
            metric_oee = gr.Textbox(label="📊 Overall Effectiveness (OEE)", value="0.0%")
        with gr.Column():
            metric_rev = gr.Textbox(label="💵 Economic Value Generated", value="$0.00")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📷 Live Single-Item Conveyor Stream")
            input_cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Input Stream")
            output_cam = gr.Image(show_label=False, type="numpy", label="Classification Result")

        with gr.Column(scale=1):
            gr.Markdown("### 📈 Bin Capacities & Distribution")
            rates_table = gr.Dataframe(interactive=False)

            gr.Markdown("### 🧠 Classification Logic Logs")
            control_logs = gr.Textbox(lines=6, interactive=False, label="Classification Event Log")
            btn_reset = gr.Button("🔄 Reset Metrics & Counters", variant="secondary")

    with gr.Row():
        live_chart = gr.LinePlot(
            x="Time",
            y="Total Sorted",
            title="Total Classified Waste vs. Elapsed Time (seconds)",
            x_title="Elapsed Time (Seconds)",
            y_title="Total Units Sorted",
        )

   
    input_cam.stream(
        fn=process_classification_pipeline,
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
    demo_cls.launch(share=True)