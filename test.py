
image_path = "img_path" 

with open(image_path, "rb") as f:
    image_bytes = f.read()  


run_inference(image_bytes)