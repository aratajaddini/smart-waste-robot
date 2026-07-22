import torch
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

model_path = Path('weights/best.pt')
print(f'1. File exists: {model_path.exists()}')
print(f'2. File size: {model_path.stat().st_size / (1024*1024):.2f} MB')

try:
    model = YOLO(str(model_path))
    names = model.names
    print(f'3. Number of classes: {len(names)}')
    print(f'   Class mapping: {names}')
    
    required = {'Glass', 'Metal', 'Paper', 'Plastic', 'Waste'}
    if set(names.values()) == required:
        print('   ✅ Contract PASS: classes match exactly.')
    else:
        print('   ❌ Contract FAIL: expected ' + str(required) + ', got ' + str(set(names.values())))
    
    # Test classification vs detection
    dummy = Image.new('RGB', (224, 224), color='red')
    results = model(dummy, verbose=False)
    result = results[0]
    if hasattr(result, 'probs') and result.probs is not None:
        print('   ✅ Model type PASS: classification model (has probs).')
    else:
        print('   ❌ Model type FAIL: detection model – ask Abbas for classification (-cls).')
        
except Exception as e:
    print(f'❌ Error loading model: {e}')