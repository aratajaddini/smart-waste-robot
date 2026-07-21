import gradio as gr
import cv2
import time
import pandas as pd
from ultralytics import YOLO
import numpy as np


MODEL_PATH = "best.pt" 
model = YOLO(MODEL_PATH)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


CLASS_MAPPING = {
    "Glass": "Glass",
    "Metal": "Metal",
    "Paper": "Paper/Cardboard",
    "Plastic": "Plastic",
    "Waste": "Waste"
}

ENVIRONMENTAL_PRIORITY = {
    "Metal": 1, 
    "Plastic": 2, 
    "Glass": 3, 
    "Paper/Cardboard": 4,
    "Waste": 5 
}

WASTE_VALUES = {
    "Metal": 5.00, 
    "Plastic": 2.50,
    "Glass": 1.50,
    "Paper/Cardboard": 1.00,
    "Waste": 0.50
}


system_metrics = {
    "total_count": 0,
    "Plastic": 0, "Glass": 0, "Metal": 0, "Paper/Cardboard": 0, "Waste": 0,
    "confidence_sum": 0.0,
    "total_revenue": 0.0
}

BIN_CAPACITIES = {
    "Plastic": 100, 
    "Glass": 100, 
    "Metal": 100, 
    "Paper/Cardboard": 100,
    "Waste": 100
}

sorting_timestamps = []
log_history = []
time_series_data = {"Time": [], "Total Sorted": []}
start_time = time.time()


def advanced_robot_logic(detected_object):
    if not detected_object:
        return None, "Conveyor belt is empty in this frame."

    current_fill = system_metrics[detected_object['class']]
    max_cap = BIN_CAPACITIES[detected_object['class']]
    
    if current_fill >= max_cap:
        timestamp = time.strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] ⚠️ WARNING: [{detected_object['class']}] ignored. Bin is FULL!"
        log_history.insert(0, log_msg)
        return None, log_msg

    timestamp = time.strftime('%H:%M:%S')
    log_msg = f"[{timestamp}] 🤖 COMMAND: Sort [{detected_object['class']}] with confidence {detected_object['confidence']:.2%}"
    return detected_object, log_msg


def process_pipeline(video_frame):
    if video_frame is None:
        return None, "0%", "0 WPM", "0%", "$0.00", pd.DataFrame(), "", pd.DataFrame()

    frame = np.array(video_frame)
    
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    enhanced_frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    
    current_time = time.time()
    

    results = model(enhanced_frame, verbose=False)[0]
    probs = results.probs

    detected_object = None
    
    if probs is not None:
        top1_index = int(probs.top1)
        raw_class_name = model.names[top1_index]
        confidence = float(probs.top1conf)

        if confidence > 0.40 and raw_class_name in CLASS_MAPPING:
            mapped_class = CLASS_MAPPING[raw_class_name]
            detected_object = {'class': mapped_class, 'confidence': confidence}
            
            h, w, _ = enhanced_frame.shape

            cv2.rectangle(enhanced_frame,(10, h - 60), (400, h - 10), (0, 0, 0), -1)
            cv2.putText(
                enhanced_frame, 
                f"{mapped_class}: {confidence:.1%}", 
                (20, h - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 255, 0), 
                2
            )

    target, control_log = advanced_robot_logic(detected_object)
    
    if target:
        system_metrics[target['class']] += 1
        system_metrics["total_count"] += 1
        system_metrics["confidence_sum"] += target['confidence']
        system_metrics["total_revenue"] += WASTE_VALUES.get(target['class'], 0.0)
        
        sorting_timestamps.append(current_time)
        log_history.insert(0, control_log)

    total = system_metrics["total_count"]
    


    rates_data = {
        "Waste Category": ["Plastic", "Glass", "Metal", "Paper/Cardboard", "Waste"],
        "Sorted Count": [
            system_metrics["Plastic"], 
            system_metrics["Glass"], 
            system_metrics["Metal"], 
            system_metrics["Paper/Cardboard"],
            system_metrics["Waste"]
        ],
        "Share (%)": [
            f"{(system_metrics['Plastic']/total)*100:.1f}%" if total > 0 else "0%",
            f"{(system_metrics['Glass']/total)*100:.1f}%" if total > 0 else "0%",
            f"{(system_metrics['Metal']/total)*100:.1f}%" if total > 0 else "0%",
            f"{(system_metrics['Paper/Cardboard']/total)*100:.1f}%" if total > 0 else "0%",
            f"{(system_metrics['Waste']/total)*100:.1f}%" if total > 0 else "0%"
        ],
        "Bin Status": [
            f"{system_metrics['Plastic']}/{BIN_CAPACITIES['Plastic']} ({(system_metrics['Plastic']/BIN_CAPACITIES['Plastic'])*100:.0f}%)",
            f"{system_metrics['Glass']}/{BIN_CAPACITIES['Glass']} ({(system_metrics['Glass']/BIN_CAPACITIES['Glass'])*100:.0f}%)",
            f"{system_metrics['Metal']}/{BIN_CAPACITIES['Metal']} ({(system_metrics['Metal']/BIN_CAPACITIES['Metal'])*100:.0f}%)",
            f"{system_metrics['Paper/Cardboard']}/{BIN_CAPACITIES['Paper/Cardboard']} ({(system_metrics['Paper/Cardboard']/BIN_CAPACITIES['Paper/Cardboard'])*100:.0f}%)",
            f"{system_metrics['Waste']}/{BIN_CAPACITIES['Waste']} ({(system_metrics['Waste']/BIN_CAPACITIES['Waste'])*100:.0f}%)"
        ]
    }
    df_rates = pd.DataFrame(rates_data)



    sorting_timestamps[:] = [t for t in sorting_timestamps if current_time - t <= 60]
    wpm = len(sorting_timestamps)
    wpm_speed = f"{wpm} WPM"

    avg_conf_raw = (system_metrics["confidence_sum"] / total) if total > 0 else 0.0
    avg_conf = f"{avg_conf_raw * 100:.1f}%"
    
    max_nominal_speed = 45.0
    performance_rate = min(wpm / max_nominal_speed, 1.0)
    oee_score = avg_conf_raw * performance_rate * 100
    oee_display = f"{oee_score:.1f}%"

    revenue_display = f"${system_metrics['total_revenue']:.2f}"


    elapsed_time = int(current_time - start_time)
    if elapsed_time not in time_series_data["Time"]:
        time_series_data["Time"].append(elapsed_time)
        time_series_data["Total Sorted"].append(total)
    df_chart = pd.DataFrame(time_series_data)

    logs_display = "\n".join(log_history[:8])

    return enhanced_frame, avg_conf, wpm_speed, oee_display, revenue_display, df_rates, logs_display, df_chart




with gr.Blocks(title="Industrial Smart Waste Sorter", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏭 Industrial Smart Waste Sorting Automation System (Classification)
    """)
    
    with gr.Row():
        with gr.Column():
            metric_conf = gr.Textbox(label="🎯 Model Accuracy (Quality)", value="0.0%")
        with gr.Column():
            metric_speed = gr.Textbox(label="⚡ Real-time Speed (Performance)", value="0 WPM")
        with gr.Column():
            metric_oee = gr.Textbox(label="📊 Overall Equipment Effectiveness (OEE)", value="0.0%")
        with gr.Column():
            metric_rev = gr.Textbox(label="💵 Economic Value Generated", value="$0.00")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📷 Live Conveyor Belt Monitoring")
            input_cam = gr.Image(sources=["webcam"], streaming=True, label="Live Input Feed", type="numpy")
            
            output_cam = gr.Image(label="AI Result", type="numpy")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📈 Real-Time Analytics & Bin Management")
            rates_table = gr.Dataframe(interactive=False)
            
            gr.Markdown("### 🧠 Robot Logic & Controller Logs")
            control_logs = gr.Textbox(label="Environmental Priority Logs", lines=5, interactive=False)

    with gr.Row():
        gr.Markdown("### 📉 System Sorting Performance Over Time")
    with gr.Row():
        live_chart = gr.LinePlot(
            x="Time", y="Total Sorted", 
            title="Total Sorted Waste vs. Elapsed Time (seconds)",
            x_title="Elapsed Time (Seconds)", y_title="Total Units Sorted"
        )

    input_cam.stream(
        fn=process_pipeline,
        inputs=input_cam,
        outputs=[output_cam, metric_conf, metric_speed, metric_oee, metric_rev, rates_table, control_logs, live_chart],
        queue=True,
        show_progress="hidden"
    )

if __name__ == "__main__":
    demo.launch(share=True)