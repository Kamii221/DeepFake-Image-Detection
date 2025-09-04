# Deepfake Image Detection Flask App

This project is a Flask web application designed for detecting deepfake images using a hybrid model architecture. The application allows users to upload images, select different models for analysis, and view the classification results along with confidence scores.

## Project Structure

- **app.py**: Main application file that sets up the Flask web server and handles routes for image analysis and model comparison.
- **models.json**: Configuration file containing details about the different models used in the application, including their architectures and input sizes.
- **weights/**: Directory containing the trained model weights files (e.g., `[model_name].pth`) for the various models.
- **templates/**: 
  - **index.html**: HTML template for the upload page where users can select an image and model.
  - **result.html**: HTML template for displaying the results of the image analysis.
- **static/**: 
  - **style.css**: CSS styles for the application, defining the layout and appearance of the HTML pages.
- **uploads/**: Directory used to temporarily store uploaded image files before processing.
- **requirements.txt**: Lists the Python dependencies required to run the application.

## Deployment Guide

1. **Install dependencies**: 
   Run the following command to install all necessary packages:
   ```
   pip install -r requirements.txt
   ```

2. **Run Flask server**: 
   Execute the following command to start the Flask application:
   ```
   python app.py
   ```

3. **Open browser**: 
   Navigate to `http://127.0.0.1:5000/` to access the application.

## Hybrid Model Integration Explanation

The application integrates a hybrid model architecture that combines multiple deep learning models (e.g., EfficientNet, Vision Transformer, ResNet) for image classification. Each model is loaded based on the user's selection, and the input image is preprocessed according to the chosen model's requirements. The predictions from the models are then aggregated to provide a final classification label and confidence score, enhancing the robustness of the analysis.