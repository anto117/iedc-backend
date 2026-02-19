import os
from flask import Flask, request, jsonify
import joblib
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- 🟢 SETUP & CONFIG ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. LOAD AUDIO MODEL (Random Forest)
AUDIO_MODEL_FILE = "poultry_audio_model.pkl" 
audio_model = None

if os.path.exists(AUDIO_MODEL_FILE):
    try:
        audio_model = joblib.load(AUDIO_MODEL_FILE)
        print("✅ Audio Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading Audio Model: {e}")
else:
    print(f"⚠️ Warning: {AUDIO_MODEL_FILE} not found.")

# 2. LOAD IMAGE MODEL (TensorFlow Lite)
IMAGE_MODEL_FILE = "poultry_image_model.tflite" 
interpreter = None
input_details = None
output_details = None
IMAGE_CLASSES = {0: 'Healthy', 1: 'Sick'}

if os.path.exists(IMAGE_MODEL_FILE):
    try:
        interpreter = tf.lite.Interpreter(model_path=IMAGE_MODEL_FILE)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("✅ Image Model (TFLite) loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading Image Model: {e}")
else:
    print(f"⚠️ Warning: {IMAGE_MODEL_FILE} not found.")


# --- 🟢 HELPER FUNCTIONS ---

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"❌ Audio Processing Error: {e}")
        return None

def preprocess_image(file_path):
    try:
        img = Image.open(file_path).convert('RGB')
        img = img.resize((224, 224)) 
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) 
        img_array /= 255.0 
        # TFLite requires float32 specifically
        return img_array.astype(np.float32)
    except Exception as e:
        print(f"❌ Image Processing Error: {e}")
        return None


# --- 🟢 API ROUTES ---

@app.route('/', methods=['GET'])
def home():
    return "Poultry AI Backend is Running! 🐔"

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    if audio_model is None:
        return jsonify({"status": "Error", "confidence": "Audio Model Not Loaded"}), 500

    file = request.files.get('file') or request.files.get('audio')
    
    if file is None or file.filename == '':
        return jsonify({"status": "Error", "confidence": "No file selected"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        features = extract_audio_features(filepath)
        if features is None:
            return jsonify({"status": "Error", "confidence": "Bad Audio File"}), 400

        prediction = audio_model.predict([features])[0]
        try:
            probabilities = audio_model.predict_proba([features])[0]
            confidence = np.max(probabilities) * 100
        except:
            confidence = 100.0 

        result = "Healthy" if prediction == 0 else "Respiratory Infection"

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({"status": result, "confidence": f"{confidence:.1f}%"})

    except Exception as e:
        return jsonify({"status": "Error", "confidence": str(e)}), 500


@app.route('/predict-image', methods=['POST'])
def predict_image():
    if interpreter is None:
        return jsonify({"status": "Error", "confidence": "Image Model Not Loaded"}), 500

    if 'image' not in request.files:
        return jsonify({"status": "Error", "confidence": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "Error", "confidence": "No file selected"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        img_array = preprocess_image(filepath)
        if img_array is None:
            return jsonify({"status": "Error", "confidence": "Bad Image File"}), 400

        # Feed the image to the TFLite model
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        
        result = IMAGE_CLASSES.get(predicted_index, "Unknown")

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({"status": result, "confidence": f"{confidence:.1f}%"})

    except Exception as e:
        return jsonify({"status": "Error", "confidence": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)