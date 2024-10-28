# Emotion Recognition using Convolutional Neural Networks

This project is a facial emotion recognition app using a Convolutional Neural Network (CNN) model trained on the FER2013 dataset. Users can upload an image, and the app will identify the emotion displayed.

## Project Overview

This project uses a CNN model to classify facial expressions into one of seven categories:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The model was trained on the FER2013 dataset and achieves approximately 63% test accuracy.

How to Use
- Start the app by running the Streamlit command above.
- Upload an image with a visible face.
- The app will display the predicted emotion.

Model Information
- Model Architecture: CNN with multiple convolutional and pooling layers, trained on grayscale images of shape (48, 48, 1).
- Framework: TensorFlow and Keras.

Dataset
-The model was trained on the FER2013 dataset, which is widely used for facial expression recognition tasks.

Future Improvements
Potential enhancements include:

- Improving model accuracy with further hyperparameter tuning.
- Adding support for real-time emotion recognition using a webcam.
- Allowing users to draw bounding boxes around faces for multi-face image inputs.
