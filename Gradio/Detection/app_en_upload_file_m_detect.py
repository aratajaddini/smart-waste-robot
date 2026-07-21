import gradio as gr
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np



MODEL_PATH = "clone/YOLO-Waste-Detection/best_model.pt" 
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

log_history = []
history_records = [] 



def advanced_robot_logic(detected_objects):
    if not detected_objects:
        return None, "No valid waste detected in the uploaded image."

    valid_objects = []
    for obj in detected_objects:
        current_fill = system_metrics[obj['class']]
        max_cap = BIN_CAPACITIES[obj['class']]
        
        if current_fill < max_cap:
            valid_objects.append(obj)
        else:
            log_msg = f"⚠️ WARNING: Waste [{obj['class']}] ignored. Bin is FULL!"
            log_history.insert(0, log_msg)

    if not valid_objects:
        return None, "⚠️ All corresponding bins for detected waste in this image are full!"

    sorted_queue = sorted(valid_objects, key=lambda x: (ENVIRONMENTAL_PRIORITY.get(x['class'], 99), -x['confidence']))
    target_object = sorted_queue[0]
    log_msg = f"🤖 COMMAND: Sort [{target_object['class']}] with confidence {target_object['confidence']:.2%}"
    
    if len(sorted_queue) > 1:
        log_msg += f" | ⚠️ Lower priority suspended: {[obj['class'] for obj in sorted_queue[1:]]}"
        
    return target_object, log_msg



def process_uploaded_image(uploaded_frame):
    if uploaded_frame is None:
        return None, "0.0%", "0 Units", "0.0%", "$0.00", pd.DataFrame(), "No image uploaded.", pd.DataFrame()

    frame = np.array(uploaded_frame)
    
    
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    enhanced_frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    
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
            
            detected_batch.append({'class': mapped_class, 'confidence': confidence})
            
     
     
            cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(enhanced_frame, f"{mapped_class}: {confidence:.1%}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    target, control_log = advanced_robot_logic(detected_batch)
    
    if target:
        system_metrics[target['class']] += 1
        system_metrics["total_count"] += 1
        system_metrics["confidence_sum"] += target['confidence']
        system_metrics["total_revenue"] += WASTE_VALUES.get(target['class'], 0.0)
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



    avg_conf_raw = (system_metrics["confidence_sum"] / total) if total > 0 else 0.0
    avg_conf = f"{avg_conf_raw * 100:.1f}%"
    
    last_detection_conf = f"{target['confidence']:.1%}" if target else "N/A"
    revenue_display = f"${system_metrics['total_revenue']:.2f}"
    processed_units = f"{total} Units"


    history_records.append({"Upload ID": len(history_records) + 1, "Total Sorted": total})
    df_chart = pd.DataFrame(history_records)

    logs_display = "\n".join(log_history[:8])

    return enhanced_frame, avg_conf, processed_units, last_detection_conf, revenue_display, df_rates, logs_display, df_chart




with gr.Blocks(title="Industrial Smart Waste Sorter (Upload Mode)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏭 Industrial Smart Waste Sorting System (Image Upload Mode Model Detection)
    """)
    
    with gr.Row():
        with gr.Column():
            metric_conf = gr.Textbox(label="🎯 Cumulative Accuracy", value="0.0%")
        with gr.Column():
            metric_speed = gr.Textbox(label="📦 Total Processed Units", value="0 Units")
        with gr.Column():
            metric_oee = gr.Textbox(label="📊 Last Detection Score", value="0.0%")
        with gr.Column():
            metric_rev = gr.Textbox(label="💵 Economic Value Generated", value="$0.00")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📷 Image Upload & Detection")
            input_img = gr.Image(sources=["upload", "clipboard"], label="Upload Waste Image", type="numpy")
            btn_analyze = gr.Button("Analyze Image", variant="primary")
            output_img = gr.Image(label="AI Detection Result", type="numpy")
            
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



    btn_analyze.click(
        fn=process_uploaded_image,
        inputs=input_img,
        outputs=[output_img, metric_conf, metric_speed, metric_oee, metric_rev, rates_table, control_logs, live_chart]
    )

if __name__ == "__main__":
    demo.launch(share=True)