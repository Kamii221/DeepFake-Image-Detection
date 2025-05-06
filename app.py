import os
import json
from flask import Flask, render_template, request, jsonify, make_response
from werkzeug.utils import secure_filename
import tensorflow as tf
import torch
import torch.nn as nn
import timm
from torchvision import models, transforms
from PIL import Image
import numpy as np
from datetime import datetime
import cv2
import logging
import torch.optim as optim

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load models configuration from JSON file
try:
    models_path = os.path.join(os.path.dirname(__file__), 'models.json')
    logger.debug(f"Loading models from: {models_path}")
    with open(models_path, 'r') as f:
        MODELS = json.load(f)
    logger.debug(f"Loaded models: {MODELS}")
except Exception as e:
    logger.error(f"Error loading models.json: {e}")
    MODELS = {}

# Create required directories if they don't exist
for directory in ['uploads', 'templates', 'static']:
    dir_path = os.path.join(os.path.dirname(__file__), directory)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.debug(f"Created directory: {dir_path}")

class EfficientNetModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        if 'b0' in model_config['architecture']['backbone'].lower():
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        else:
            self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        
        layers = []
        for layer in model_config['architecture']['classifier']['layers']:
            if layer['type'] == 'Linear':
                layers.append(nn.Linear(layer['in_features'], layer['out_features']))
            elif layer['type'] == 'BatchNorm1d':
                layers.append(nn.BatchNorm1d(layer['features']))
            elif layer['type'] == 'GELU':
                layers.append(nn.GELU())
            elif layer['type'] == 'LeakyReLU':
                negative_slope = layer.get('negative_slope', 0.01)
                layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            elif layer['type'] == 'Dropout':
                layers.append(nn.Dropout(layer['p']))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

class EnsembleModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.effnet = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.effnet.classifier = nn.Identity()
        
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit.head = nn.Identity()
        
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        
        # Calculate the output sizes
        effnet_output_size = 1280  # EfficientNet-B1 output size
        vit_output_size = 768  # ViT output size
        resnet_output_size = 2048  # ResNet50 output size
        
        # Enhanced classifier architecture
        self.classifier = nn.Sequential(
            nn.Linear(effnet_output_size + vit_output_size + resnet_output_size, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        eff_features = self.effnet(x)
        vit_features = self.vit(x)
        resnet_features = self.resnet(x)
        combined_features = torch.cat([eff_features, vit_features, resnet_features], dim=1)
        return self.classifier(combined_features)

def get_model_type(model_name):
    """Determine the model type from the model name"""
    if model_name.startswith('ensemble'):
        return 'ensemble'
    elif 'b0' in model_name:
        return 'efficientnet_b0'
    elif 'b1' in model_name:
        return 'efficientnet_b1'
    return None

def get_activation_type(model_name):
    """Get the activation function type from the model name"""
    if 'gelu' in model_name.lower():
        return 'GELU'
    elif 'leakyrelu' in model_name.lower():
        return 'LeakyReLU'
    return None

def load_model(model_name):
    """Load the specified model with its architecture"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config = MODELS[model_name]
        
        model_type = get_model_type(model_name)
        if model_type == 'ensemble':
            model = EnsembleModel(model_config)
        else:
            model = EfficientNetModel(model_config)
        
        # Load model weights if available
        weights_path = f'weights/{model_name}.pth'
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
            logger.debug(f"Loaded weights for {model_name} from {weights_path}")
        
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_image_info(image_path):
    """Extract basic information about the image"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        info = {
            'dimensions': f"{img.width}x{img.height}",
            'format': img.format,
            'mode': img.mode,
            'size_kb': os.path.getsize(image_path) / 1024,
            'channels': len(img.getbands()),
        }
        
        if img.mode == 'RGB':
            avg_color = np.mean(img_array, axis=(0, 1))
            info['avg_color'] = f"RGB({int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])})"
            brightness = np.mean(img_array)
            info['brightness'] = f"{int(brightness)}/255"
        
        return info
    except Exception as e:
        logger.error(f"Error in get_image_info: {e}")
        raise

def preprocess_image(image_path, model_name):
    """Preprocess image according to model requirements"""
    try:
        img = Image.open(image_path).convert('RGB')
        target_size = MODELS[model_name]['input_size']
        
        # Enhanced data augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Standard validation transform
        val_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Use validation transform for inference
        img_tensor = val_transform(img).unsqueeze(0)
        return img_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

@app.route('/')
def home():
    try:
        logger.debug("Rendering home page")
        # Group models by type and activation
        grouped_models = {
            'EfficientNet-B0': [m for m in MODELS if 'b0' in m],
            'EfficientNet-B1': [m for m in MODELS if 'b1' in m and not m.startswith('ensemble')],
            'Ensemble': [m for m in MODELS if m.startswith('ensemble')]
        }
        return render_template('index.html', models=MODELS, grouped_models=grouped_models)
    except Exception as e:
        logger.error(f"Error rendering home page: {e}")
        return f"Error: {str(e)}", 500

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify(MODELS)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        if 'file' not in request.files:
            logger.warning("No file in request")
            return make_response(jsonify({'error': 'No file uploaded'}), 400)
        
        file = request.files['file']
        model_name = request.form.get('model', 'efficientnet_b0_gelu')
        
        logger.debug(f"Received file: {file.filename}, model: {model_name}")
        
        if file.filename == '':
            logger.warning("No file selected")
            return make_response(jsonify({'error': 'No file selected'}), 400)
        
        if model_name not in MODELS:
            logger.warning(f"Invalid model selected: {model_name}")
            return make_response(jsonify({'error': 'Invalid model selected'}), 400)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.debug(f"Saving file to: {filepath}")
            file.save(filepath)
            
            try:
                image_info = get_image_info(filepath)
                logger.debug(f"Image info: {image_info}")
                
                model = load_model(model_name)
                if model is None:
                    return make_response(jsonify({'error': 'Failed to load model'}), 500)
                
                processed_image = preprocess_image(filepath, model_name)
                with torch.no_grad():
                    output = model(processed_image)
                    prediction = torch.sigmoid(output).item()
                
                analysis_result = {
                    'result': 'fake' if prediction > 0.5 else 'real',
                    'confidence': float(prediction) if prediction > 0.5 else float(1 - prediction),
                    'model_used': MODELS[model_name]['name'],
                    'activation_type': get_activation_type(model_name),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'image_info': image_info,
                    'technical_details': {
                        'model_name': model_name,
                        'model_architecture': MODELS[model_name]['architecture'],
                        'preprocessing': f"Resized to {MODELS[model_name]['input_size']}"
                    }
                }
                
                return jsonify(analysis_result)
            
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                return make_response(jsonify({'error': str(e)}), 500)
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug(f"Removed temporary file: {filepath}")
        
        logger.warning(f"Invalid file type: {file.filename}")
        return make_response(jsonify({'error': 'Invalid file type'}), 400)
    
    except Exception as e:
        logger.error(f"Unexpected error in analyze_image: {e}")
        return make_response(jsonify({'error': str(e)}), 500)

@app.route('/compare', methods=['POST'])
def compare_models():
    """Compare results from different model variants on the same image"""
    try:
        if 'file' not in request.files:
            return make_response(jsonify({'error': 'No file uploaded'}), 400)
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return make_response(jsonify({'error': 'Invalid file'}), 400)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            image_info = get_image_info(filepath)
            results = {}
            
            # Group models by type
            model_groups = {
                'EfficientNet-B0': ['efficientnet_b0_gelu', 'efficientnet_b0_leakyrelu'],
                'EfficientNet-B1': ['efficientnet_b1_gelu', 'efficientnet_b1_leakyrelu'],
                'Ensemble': ['ensemble_gelu', 'ensemble_leakyrelu']
            }
            
            for group, models in model_groups.items():
                group_results = {}
                for model_name in models:
                    model = load_model(model_name)
                    if model is not None:
                        processed_image = preprocess_image(filepath, model_name)
                        with torch.no_grad():
                            output = model(processed_image)
                            prediction = torch.sigmoid(output).item()
                        
                        group_results[model_name] = {
                            'result': 'fake' if prediction > 0.5 else 'real',
                            'confidence': float(prediction) if prediction > 0.5 else float(1 - prediction),
                            'activation_type': get_activation_type(model_name),
                            'raw_score': float(prediction)
                        }
                results[group] = group_results
            
            comparison_result = {
                'image_info': image_info,
                'results': results,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return jsonify(comparison_result)
            
        except Exception as e:
            logger.error(f"Error in model comparison: {e}")
            return make_response(jsonify({'error': str(e)}), 500)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        logger.error(f"Unexpected error in compare_models: {e}")
        return make_response(jsonify({'error': str(e)}), 500)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response

def get_training_config():
    """Get optimized training configuration"""
    return {
        'batch_size': 8,
        'learning_rate': 1e-5,
        'weight_decay': 5e-4,
        'epochs': 30,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'gradient_clip': 1.0,
        'early_stopping_patience': 5,
        'checkpoint_frequency': 1
    }

def get_optimizer(model, config):
    """Get optimizer with specified configuration"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    return optimizer

def get_scheduler(optimizer, config):
    """Get learning rate scheduler"""
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['learning_rate'] * 0.1
    )
    return scheduler

def train_model(model, train_loader, val_loader, config):
    """Train model with improved training loop"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            torch.cuda.empty_cache()
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': avg_val_loss
            }, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{config['epochs']} - "
                   f"Train Loss: {avg_train_loss:.4f} - "
                   f"Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['checkpoint_frequency'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': avg_val_loss
            }, f'checkpoint_epoch_{epoch}.pth')
    
    return model

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=True, host='127.0.0.1', port=5500) 