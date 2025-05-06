# Deepfake Image Detection Web Application

This web application provides an interface for detecting deepfake images using deep learning. It uses EfficientNet and Vision Transformers to analyze uploaded images and determine if they are real or manipulated.

## Features

- Modern, responsive web interface
- Drag-and-drop image upload
- Real-time image analysis
- Confidence score display
- Support for common image formats (PNG, JPG, JPEG)

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)


1. Upload an image using the drag-and-drop interface or file browser

2. Click "Analyze Image" to process the upload

3. View the results showing whether the image is real or fake, along with a confidence score

## Project Structure

```
deepfake_detection_web/
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
├── templates/         
│   └── index.html     # Web interface
└── uploads/           # Temporary storage for uploaded images
```

## Model Integration

The current implementation includes a placeholder for the deepfake detection model. To integrate your trained model:

1. Place your model file in the project directory
2. Update the `load_model()` function in `app.py`
3. Modify the preprocessing steps in `preprocess_image()` if needed
4. Update the prediction code in the `analyze_image()` route

## Security Notes

- The application includes file type validation
- Uploaded files are automatically deleted after processing
- Maximum file size is limited to 16MB
- Only specific image formats are allowed

## Contributing

Feel free to submit issues and enhancement requests!


















# DeepFake-Image-Detection
A hybrid deep learning model combining EfficientNet and Vision Transformers for accurate deepfake image detection. Trained on FF++ and DFDC datasets, the model improves feature extraction, generalization, and precision across manipulated media.Aim
The aim of this project is to develop an improved deepfake detection model for images
by integrating EfficientNet in place of CNN and Vision Transformers (ViTs) instead
of RNN, enhancing accuracy in identifying manipulated content. By leveraging the
power of Vision Transformers, the model will achieve superior performance in detecting
deepfakes.
2.2 Objectives
• Replace the existing CNN with EfficientNet to enhance feature extraction and improve
deepfake manipulation detection.
• Improve the model’s accuracy and generalization capabilities by utilizing Vision
Transformers, focusing on critical spatial features within images for more precise
detection.


Methodology
 Problem Analysis and Requirement Gathering
• Analyze existing systems to highlight gaps in image manipulation detection.
6.2 Model Selection and Development
• Develop the core detection system using EfficientNet to improve feature extraction
and overall detection performance.
• Enhance detection capabilities through transfer learning by utilizing a pre-trained
model, such as EfficientNet-B0 or EfficientNet-B1.
• Replace the RNN layers with Vision Transformers to improve focus on critical
8
features, boosting detection accuracy.
6.3 Dataset Preparation
• Gather and preprocess dataset containing both real and manipulated
images.including FF++ (FaceForensics++), Celeb-DF, and DFDC (DeepFake
Detection Challenge).
• Perform data augmentation to ensure the model is trained on diverse variations of
images.
• Split the dataset into training, validation, and testing sets for effective evaluation.
6.4 Model Training and Validation
• Train the model on the prepared dataset.
• Validate the model’s performance using the validation set and assess its accuracy
on the testing set.
6.5 Evaluation and Comparison
• Compare the performance of the proposed model against baseline models, such
as standard CNN and RNN architectures.
• Use comprehensive evaluation metrics including accuracy, precision, recall,
F1-score, and AUC-ROC to assess model performance thoroughly and ensure
balanced evaluation across various performance dimensions.
![image](https://github.com/user-attachments/assets/1ed00a74-2a79-4880-81e1-75ee69929a4c)

