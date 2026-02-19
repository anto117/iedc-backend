import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
# 🟢 FIX 1: Added 'Input' to the imports
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import os

# 1. SETUP PATHS
BASE_DIR = 'dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALID_DIR = os.path.join(BASE_DIR, 'valid') 

TRAIN_CSV = os.path.join(TRAIN_DIR, '_annotations.csv')
VALID_CSV = os.path.join(VALID_DIR, '_annotations.csv')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_and_clean_csv(csv_path):
    """Loads CSV and automatically removes duplicate image rows to prevent Keras crashes"""
    try:
        df = pd.read_csv(csv_path)
        # Drop duplicates based on the 'filename' column so Keras only sees one label per image
        df = df.drop_duplicates(subset=['filename'])
        return df
    except Exception as e:
        print(f"❌ Error loading {csv_path}: {e}")
        return None

def train_image_ai():
    print("📸 Loading Data Maps...")

    train_df = load_and_clean_csv(TRAIN_CSV)
    valid_df = load_and_clean_csv(VALID_CSV)

    if train_df is None or valid_df is None:
        print("❌ Stopping training because CSV files could not be loaded.")
        return

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    print(f"   - Found {len(train_df)} unique training images")
    print(f"   - Found {len(valid_df)} unique validation images")

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=TRAIN_DIR,        
        x_col="filename",           
        y_col="class",              
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    valid_generator = val_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=VALID_DIR,
        x_col="filename",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    # 2. BUILD THE BRAIN (CNN)
    print("🧠 Building the Brain...")
    model = Sequential([
        # 🟢 FIX 2: Explicit Input layer to resolve the Keras UserWarning
        Input(shape=(224, 224, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        # 🟢 FIX 3: Safely count the class indices to resolve the AttributeError
        Dense(len(train_generator.class_indices), activation='softmax') 
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 3. TRAIN
    print("🚀 Training Started...")
    model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=10
    )

    # 4. SAVE
    model.save("poultry_image_model.h5")
    print("✅ Model Saved as 'poultry_image_model.h5'")
    
    print(f"🏷️ IMPORTANT Class Indices: {train_generator.class_indices}")
    print("⚠️ Make sure the IMAGE_CLASSES dictionary in app.py exactly matches these indices!")

if __name__ == "__main__":
    train_image_ai()