import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0

app = Flask(__name__)

# --- CONFIGURATION ---
WEIGHTS_PATH = "fer_model_weights.h5"
IMG_SIZE = 112
NUM_CLASSES = 7
EMOTIONS_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
FACE_MARGIN = 0.1

# Load Haar Cascade
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- MODEL ARCHITECTURE ---
def build_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model = EfficientNetB0(include_top=False, weights=None, input_tensor=inputs)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# --- INITIALIZATION ---
print(f"Building model")
model = build_model()

try:
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading weights from {WEIGHTS_PATH}...")
        model.load_weights(WEIGHTS_PATH)
        print("Model loaded successfully.")
    else:
        print(f"ERROR: {WEIGHTS_PATH} not found.")
except Exception as e:
    print(f"Error loading weights: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Decode image
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        height, width, _ = frame.shape
        
        # Face detection (requires grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        results = []
        
        for (x, y, w, h) in faces:
            # Calculate margin
            margin_x = int(w * FACE_MARGIN)
            margin_y = int(h * FACE_MARGIN)
            x_start = max(0, x - margin_x)
            y_start = max(0, y - margin_y)
            x_end = min(width, x + w + margin_x)
            y_end = min(height, y + h + margin_y)
            
            # Extract ROI in Color (RGB)
            roi_color = frame[y_start:y_end, x_start:x_end]
            
            try:
                resized = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
            except Exception:
                continue
            
            # Preprocessing
            img_pixels = resized.astype('float32')
            img_pixels = np.expand_dims(img_pixels, axis=0)
            
            # Prediction
            raw_predictions = model.predict(img_pixels)[0]
            all_scores = [float(score) for score in raw_predictions]
            
            maxindex = int(np.argmax(raw_predictions))
            emotion_label = EMOTIONS_LIST[maxindex]
            # Calculate top confidence percentage
            top_conf_percent = round(all_scores[maxindex] * 100, 1)
            
            results.append({
                "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "top_emotion": emotion_label,
                "top_confidence": top_conf_percent,
                "all_scores": all_scores
            })
            
        return jsonify(results)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)