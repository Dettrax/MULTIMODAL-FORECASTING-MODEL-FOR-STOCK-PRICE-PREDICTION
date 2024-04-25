import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_image

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

# Normalize pixel values
X /= 255.0

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier (or any other model)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Select a specific instance to explain (e.g., an image of digit '7')
instance_idx = 0
image = X_test[instance_idx].reshape(28, 28)

# Define LIME explainer for images
explainer = lime.lime_image.LimeImageExplainer()

# Explain model prediction for the selected instance
# Convert image dtype to double
image = image.astype('double')


def predict_proba_wrapper(x):
    # Reshape the input from (num_samples, img_height, img_width, num_channels)
    # to (num_samples, img_height * img_width * num_channels)
    x_reshaped = x.reshape(x.shape[0], -1)

    # If the input has more than 784 features (e.g., because it's a color image with 3 channels),
    # then convert it to grayscale by averaging across the color channels
    if x_reshaped.shape[1] > 784:
        x_reshaped = np.mean(x_reshaped.reshape(x.shape[0], 28, 28, -1), axis=-1).reshape(x.shape[0], -1)

    return model.predict_proba(x_reshaped)

# Now call the explain_instance method with the wrapper function
explanation = explainer.explain_instance(image, predict_proba_wrapper, top_labels=1, hide_color=0, num_samples=1000)

# Show original image
plt.figure(figsize=(4,4))
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# Show LIME explanation

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
plt.figure(figsize=(4,4))
plt.imshow(mask, cmap='gray')
plt.title('LIME Explanation')
plt.axis('off')
plt.show()
