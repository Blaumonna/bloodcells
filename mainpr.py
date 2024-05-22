import os
import numpy as np
import matplotlib.pyplot as plt
import keras
import seaborn as sns
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import time
from sklearn.metrics import confusion_matrix, classification_report

# Set the path to your dataset
dataset_path = r"C:\Users\blaum\PycharmProjects\pr\mixed"
#r"C:\Users\blaum\PycharmProjects\tpr\input\blood-cells-image-dataset\bloodcells_dataset"
#new_dataset_path = r"C:\Users\blaum\PycharmProjects\tpr\input\blood-cells-image-dataset\mixed"
# Define the image size and batch size
img_size = (128, 128)
batch_size = 20

# Create data generators for training and validation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True  # Shuffle the training data
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)
# Combine labels from both training and validation sets
combined_labels = np.concatenate([train_generator.labels, validation_generator.labels])

# Display the number of images in each category
print("Number of training images per category:")
print(train_generator.class_indices)

# Display random images from each category
class_names = list(train_generator.class_indices.keys())
fig, axes = plt.subplots(1, len(class_names), figsize=(15, 15))

for i, class_name in enumerate(class_names):
    img_list = os.listdir(os.path.join(dataset_path, class_name))
    img_name = np.random.choice(img_list)
    img_path = os.path.join(dataset_path, class_name, img_name)

    img = keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = keras.preprocessing.image.img_to_array(img) / 255.0
    axes[i].imshow(img_array)
    axes[i].set_title(class_name)
    axes[i].axis("off")

plt.show()

# Plot a histogram for the distribution of data between classes in both training and validation sets
plt.figure(figsize=(10, 6))
plt.bar(class_names, np.bincount(combined_labels), color='skyblue')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0, ha='right')
plt.show()

# Check if the model file exists
model_file_path = "blood_cells_model7.h5"
if os.path.exists(model_file_path):
    # Load the pre-trained model
    model = keras.models.load_model(model_file_path)
else:
    # Define the CNN model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(len(class_names), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model and measure time
    start_time = time.time()
    history = model.fit(
        train_generator,
        epochs=5,
        validation_data=validation_generator
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time} seconds")

    # Save the trained model
    model.save(model_file_path)

    # Plot training history
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

# Evaluate the model
evaluation_result = model.evaluate(validation_generator)
print(f"Validation Accuracy: {evaluation_result[1]*100:.2f}%")

# Display predictions for random images in one batch
random_images = []
random_labels = []

for class_name in class_names:
    img_list = os.listdir(os.path.join(dataset_path, class_name))
    np.random.shuffle(img_list)  # Shuffle the list of image names
    img_name = img_list[0]  # Take the first randomly selected image
    img_path = os.path.join(dataset_path, class_name, img_name)

    img = keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = keras.preprocessing.image.img_to_array(img) / 255.0
    random_images.append(img_array)
    random_labels.append(class_name)

random_images = np.array(random_images)
random_labels = np.array(random_labels)

predictions = model.predict(random_images)
predicted_classes = [class_names[np.argmax(pred)] for pred in predictions]

# Display predictions for random images
plt.figure(figsize=(15, 7))
for i in range(len(random_images)):
    plt.subplot(2, len(random_images)//2, i+1)
    plt.imshow(random_images[i])
    plt.title(f"Actual: {random_labels[i]}\nPredicted: {predicted_classes[i]}")
    plt.axis("off")

plt.show()

# Plot confusion matrix
true_labels = validation_generator.labels
predictions = model.predict(validation_generator)
predicted_labels = np.argmax(predictions, axis=1)

conf_matrix = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
class_report = classification_report(true_labels, predicted_labels, target_names=class_names)
print("Classification Report:\n", class_report)