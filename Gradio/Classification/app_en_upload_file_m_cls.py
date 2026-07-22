import gradio as gr
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import time












# 1. Model Loading & Initial Configuration

MODEL_PATH = "best.pt" 
model = YOLO(MODEL_PATH)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Output class mapping
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

# Estimated economic value per unit (USD)
WASTE_VALUES = {
    "Metal": 5.00, 
    "Plastic": 2.50,
    "Glass": 1.50,
    "Paper/Cardboard": 1.00,
    "Waste": 0.50
}


# 2. System Metrics & State Tracking

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

log_history = []
history_records = [] 


# 3. Robot Control Logic

def advanced_robot_logic(detected_object):
    if not detected_object:
        return None, "No valid object detected in the uploaded image."

    current_fill = system_metrics[detected_object['class']]
    max_cap = BIN_CAPACITIES[detected_object['class']]
    
    if current_fill >= max_cap:
        log_msg = f"⚠️ WARNING: [{detected_object['class']}] ignored. Bin is FULL!"
        log_history.insert(0, log_msg)
        return None, log_msg

    log_msg = f"🤖 COMMAND: Sort [{detected_object['class']}] with confidence {detected_object['confidence']:.2%},Command Sent To Arduino"
    return detected_object, log_msg


# 4. Processing Pipeline 

def process_uploaded_image(uploaded_frame):
    if uploaded_frame is None:
        return None, "0.0%", "1 Unit", "0.0%", "$0.00", pd.DataFrame(), "No image uploaded.", pd.DataFrame()

    frame = np.array(uploaded_frame)
    
    # Pre-processing with CLAHE
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    enhanced_frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    
    # Model inference
    results = model(enhanced_frame, verbose=False)[0]
    probs = results.probs

    detected_object = None
    
    if probs is not None:
        top1_index = int(probs.top1)
        raw_class_name = model.names[top1_index]
        confidence = float(probs.top1conf)

        # Confidence threshold 
        if confidence > 0.40 and raw_class_name in CLASS_MAPPING:
            mapped_class = CLASS_MAPPING[raw_class_name]
            
            detected_object = {'class': mapped_class, 'confidence': confidence}
            
            # Display status overlay on image
            cv2.rectangle(enhanced_frame, (10, 10), (380, 60), (0, 0, 0), -1)
            cv2.putText(
                enhanced_frame, 
                f"{mapped_class}: {confidence:.1%}", 
                (20, 45), 
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
        log_history.insert(0, control_log)

    total = system_metrics["total_count"]
    
    # Dynamic Dataframe for Bin Capacities
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

    # Calculate KPIs
    avg_conf_raw = (system_metrics["confidence_sum"] / total) if total > 0 else 0.0
    avg_conf = f"{avg_conf_raw * 100:.1f}%"
    
    # Display OEE equivalent for upload mode
    current_conf_display = f"{detected_object['confidence']:.1%}" if detected_object else "N/A"
    revenue_display = f"${system_metrics['total_revenue']:.2f}"
    processed_units = f"{total} Units"

    # Update Chart History 
    history_records.append({"Upload ID": len(history_records) + 1, "Total Sorted": total})
    df_chart = pd.DataFrame(history_records)

    logs_display = "\n".join(log_history[:8])

    return enhanced_frame, avg_conf, processed_units, current_conf_display, revenue_display, df_rates, logs_display, df_chart


# 5. Gradio Dashboard 

with gr.Blocks(title="Industrial Smart Waste Sorter (Classification Model Upload Mode)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏭 Industrial Smart Waste Sorting System (Classification Model Image Upload Mode)
    """)
    
    with gr.Row():
        with gr.Column():
            metric_conf = gr.Textbox(label="🎯 Cumulative Accuracy", value="0.0%")
        with gr.Column():
            metric_speed = gr.Textbox(label="📦 Total Processed", value="0 Units")
        with gr.Column():
            metric_oee = gr.Textbox(label="📊 Last Detection Score", value="0.0%")
        with gr.Column():
            metric_rev = gr.Textbox(label="💵 Economic Value Generated", value="$0.00")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📷 Image Upload & Detection")
            input_img = gr.Image(sources=["upload", "clipboard"], label="Upload Waste Image", type="numpy")
            btn_analyze = gr.Button("Analyze Image", variant="primary")
            output_img = gr.Image(label="AI Result", type="numpy")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📈 Real-Time Analytics & Bin Management")
            rates_table = gr.Dataframe(interactive=False)
            
            gr.Markdown("### 🧠 Robot Logic & Controller Logs")
            control_logs = gr.Textbox(label="Environmental Priority Logs", lines=5, interactive=False)

    with gr.Row():
        gr.Markdown("### 📉 Sorting Accumulation History")
    with gr.Row():
        live_chart = gr.LinePlot(
            x="Upload ID", y="Total Sorted", 
            title="Accumulative Sorted Items vs. Image Upload Sequence",
            x_title="Upload Sequence", y_title="Total Units Sorted"
        )

    # Trigger action on Button click or Image Upload
    btn_analyze.click(
        fn=process_uploaded_image,
        inputs=input_img,
        outputs=[output_img, metric_conf, metric_speed, metric_oee, metric_rev, rates_table, control_logs, live_chart]
    )

if __name__ == "__main__":
    demo.launch(share=True)
