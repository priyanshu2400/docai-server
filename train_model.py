import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Image data generator for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Normalizing the images

train_generator = train_datagen.flow_from_directory(
    'Training',  # Your folder path containing '0', '1', '2', '3' folders
    target_size=(150, 150),   # Resize images to 150x150
    batch_size=40,            # Batch size
    class_mode='categorical',  # For multi-class classification
    subset='training',        # Training data (80%)
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    'Test',  
    target_size=(150, 150),
    batch_size=40,
    class_mode='categorical',
    subset='validation',      # Validation data (20%)
    shuffle=True
)

# Building the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 output classes
])

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Save the model
model.save('lung_disease_model.h5')
