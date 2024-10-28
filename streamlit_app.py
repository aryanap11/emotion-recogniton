import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model("emotion_recognition_model.keras")

# Emotion labels based on your dataset (FER2013)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((48, 48))  # Resize to 48x48
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image /= 255.0  # Normalize pixel values
    return image

# Streamlit app layout
st.title("Facial Emotion Recognition")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)
    emotion = emotion_labels[predicted_class[0]]

    # Display the prediction
    st.write(f"Predicted Emotion: **{emotion}**")

# Add a footer
st.markdown("---")
st.write("Upload an image to see the predicted emotion.")
