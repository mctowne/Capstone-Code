import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage import exposure

def load_dataset(dataset_dir):
    images = []
    labels = []
    class_names = sorted(os.listdir(dataset_dir))

    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)  # Load image using OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            image = cv2.resize(image, (32, 32))  # Resize to a consistent size
            images.append(image)
            labels.append(i)  # Assign a label to the image based on the class index

    return np.array(images), np.array(labels)


# Example usage:
dataset_dir = "data"
images, labels = load_dataset(dataset_dir)
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

# Step 2: Preprocess the images and extract features (Using HOG)
def extract_hog_features(images):
    hog_features = []
    for img in images:
        fd = hog(img, orientations=8, pixels_per_cell=(8, 8),
                 cells_per_block=(1, 1), visualize=False, multichannel=False)
        hog_features.append(fd)
    return np.array(hog_features)

# Extract HOG features from the images
hog_features = extract_hog_features(images)

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)

# Step 4: Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
