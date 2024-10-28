from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the previously saved model
model = load_model('saved_model/my_model.h5')

def predict_action(image_path):
    """Predicts the action (left, hold, or right) for a given image."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load the image in color
    image = cv2.resize(image, (100, 50))  # Resize to match model input
    image = image / 255.0  # Normalize pixel values
    image = image.reshape(1, 50, 100, 3)  # Add batch dimension and channel

    prediction = model.predict(image)
    action = np.argmax(prediction)

    if action == 0:
        return "left"
    elif action == 1:
        return "hold"
    else:
        return "right"

# Test the prediction function with a new game image
test_image_path = 'image.png'  # Replace with the path to your test image
action = predict_action(test_image_path)
print(f"The recommended action is: {action}")
