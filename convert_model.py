import tensorflow as tf

print("⏳ Loading heavy .h5 model...")
model = tf.keras.models.load_model('poultry_image_model.h5')

print("🗜️ Compressing into TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('poultry_image_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Compression complete! Saved as 'poultry_image_model.tflite'")
