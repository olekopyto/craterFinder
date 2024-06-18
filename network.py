import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to load images and masks
def load_data(input_dir, output_dir):
    input_images = []
    output_masks = []
    expected_size = (512, 512)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            # Load input image
            input_image_path = os.path.join(input_dir, filename)
            if not os.path.exists(input_image_path):
                print(f"Input image file does not exist: {input_image_path}")
                continue
            input_image = Image.open(input_image_path).convert('L')  # Convert to grayscale
            input_image = np.array(input_image)
            
            if input_image.shape != expected_size:
                print(f"Skipping {filename} due to unexpected size: {input_image.shape}")
                continue

            input_image = input_image / 255.0  # Normalize to [0, 1]
            
            # Load corresponding output mask
            output_filename = f't_{filename}'
            output_mask_path = os.path.join(output_dir, output_filename)
            if not os.path.exists(output_mask_path):
                print(f"Output mask file does not exist: {output_mask_path}")
                continue
            output_mask = Image.open(output_mask_path).convert('L')  # Convert to grayscale
            output_mask = np.array(output_mask)

            if output_mask.shape != expected_size:
                print(f"Skipping {output_filename} due to unexpected size: {output_mask.shape}")
                continue

            output_mask = output_mask / 255.0  # Normalize to [0, 1]
            
            # Append to lists
            input_images.append(input_image)
            output_masks.append(output_mask)
    
    # Convert lists to numpy arrays
    if len(input_images) == 0 or len(output_masks) == 0:
        print("No images or masks found. Please check your directories.")
        return None, None
    
    input_images = np.expand_dims(np.array(input_images), axis=-1)  # Add channel dimension
    output_masks = np.expand_dims(np.array(output_masks), axis=-1)  # Add channel dimension
    
    return input_images, output_masks

# Load the data
input_dir = 'moonAlt'
output_dir = 'moonCraters'
input_images, output_masks = load_data(input_dir, output_dir)

if input_images is None or output_masks is None:
    print("Error loading data. Exiting.")
    exit()

# Define the U-Net model
def unet_model(input_size=(512, 512, 1)):
    inputs = tf.keras.Input(input_size)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Build and compile the model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(input_images, output_masks, epochs=10, batch_size=16, validation_split=0.1)

# Save the model
model.save('crater_detection_unet.h5')

print("Model training complete and saved.")
