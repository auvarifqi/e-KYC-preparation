This repository serves as our experimentation ground for OCR (Optical Character Recognition) technology and passive liveness training. Through this project, we aim to explore and implement various cutting-edge methods and techniques in both domains, with the goal of enhancing our understanding and capabilities in text extraction from images as well as passive face detection using AI. This repository contains implementations for both face liveness detection using depth map prediction and OCR (Optical Character Recognition) integration with NLP (Natural Language Processing) techniques.

# Liveness Detection
The liveness_detection folder contains the implementation for passive liveness detection using AI. It utilizes a fusion of Convolutional Neural Networks (CNNs) and Computer Vision techniques to distinguish between genuine and fake faces in real-time scenarios. The webcam captures image frames, which are then processed through a pre-trained model. This model is specifically trained on depth maps derived from the dataset. The depth map generation process is accomplished using a separate CNN model.

Requirements
Python3
Tensorflow
dlib
Keras
numpy
sklearn
Imutils
OpenCV

For more detailed information, please refer to the liveness_detection/README.md file.

# OCR
The ocr folder contains the implementation for text recognition with TensorFlow and CTC (Connectionist Temporal Classification) network. This integration enhances text recognition from images by integrating Optical Character Recognition (OCR) with Natural Language Processing (NLP) techniques. By combining these two methodologies, we can improve the accuracy and efficiency of extracting text from images in various contexts.

Benefits of OCR and NLP Integration
Improved Accuracy
Efficient Processing
Enhanced User Experience

For more detailed information, please refer to the ocr/README.md file.
