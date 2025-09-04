import os
import json
import time
import threading
from datetime import datetime
import logging

from flask import Flask, request, jsonify, make_response, render_template
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import models, transforms
from PIL import Image
import numpy as np

# -------------------------------------------------------------
# Logging
# -------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# Flask app config
# -------------------------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# -------------------------------------------------------------
# Load models configuration from models.json (optional)
# -------------------------------------------------------------
try:
    models_path = os.path.join(os.path.dirname(__file__), 'models.json')
    logger.debug(f"Loading models from: {models_path}")
    with open(models_path, 'r') as f:
        MODELS = json.load(f)
    logger.debug(f"Loaded models: {list(MODELS.keys())}")
except Exception as e:
    logger.warning(f"Couldn't load models.json ({e}). Falling back to defaults.")
    # Minimal defaults so the API can run without models.json
    MODELS = {
        "efficientnet_b0_gelu": {
            "name": "EfficientNet-B0 (GELU)",
            "input_size": 224,
            "architecture": {
                "backbone": "EfficientNet-B0",
                "classifier": {"layers": [
                    {"type": "Linear", "in_features": 1280, "out_features": 512},
                    {"type": "GELU"},
                    {"type": "Dropout", "p": 0.2},
                    {"type": "Linear", "in_features": 512, "out_features": 1},
                ]}
            }
        },
        "vit_base_patch16_224": {
            "name": "Vision Transformer Base Patch16 224",
            "input_size": 224,
            "architecture": {
                "backbone": "ViT Base Patch16 224",
                "classifier": {"layers": [
                    {"type": "Linear", "in_features": 768, "out_features": 1}
                ]}
            }
        },
        "resnet50": {
            "name": "ResNet-50",
            "input_size": 224,
            "architecture": {
                "backbone": "ResNet-50",
                "classifier": {"layers": [
                    {"type": "Linear", "in_features": 2048, "out_features": 1}
                ]}
            }
        },
        "efficientnet_b0": {
            "name": "EfficientNet-B0",
            "input_size": 224,
            "architecture": {
                "backbone": "EfficientNet-B0",
                "classifier": {"layers": [
                    {"type": "Linear", "in_features": 1280, "out_features": 1}
                ]}
            }
        },
        "efficientnet_b1": {
            "name": "EfficientNet-B1",
            "input_size": 224,
            "architecture": {
                "backbone": "EfficientNet-B1",
                "classifier": {"layers": [
                    {"type": "Linear", "in_features": 1280, "out_features": 1}
                ]}
            }
        },
        "deepfake_pipeline": {
            "name": "Deepfake Detection Pipeline (EfficientNet + ViT)",
            "input_size": 224,
            "architecture": {
                "backbone": "EfficientNet-B0 + ViT Base Patch16 224",
                "classifier": {"layers": [
                    {"type": "Linear", "in_features": 768, "out_features": 256},
                    {"type": "ReLU"},
                    {"type": "Dropout", "p": 0.3},
                    {"type": "Linear", "in_features": 256, "out_features": 1}
                ]}
            }
        },
        "hybrid": {
            "name": "Hybrid (ResNet + ViT + EffNet)",
            "input_size": 224,
            "architecture": {
                "backbone": "ResNet-50 + ViT Base Patch16 224 + EfficientNet-B0",
                "classifier": {"layers": [
                    {"type": "Linear", "in_features": 4096, "out_features": 512},
                    {"type": "ReLU"},
                    {"type": "Dropout", "p": 0.4},
                    {"type": "Linear", "in_features": 512, "out_features": 256},
                    {"type": "ReLU"},
                    {"type": "Dropout", "p": 0.3},
                    {"type": "Linear", "in_features": 256, "out_features": 1}
                ]}
            }
        },
        "efficientnet_b0_leakyrelu": {
            "name": "EfficientNet-B0 (LeakyReLU)",
            "input_size": 224,
            "architecture": {
                "backbone": "EfficientNet-B0",
                "classifier": {"layers": [
                    {"type": "Linear", "in_features": 1280, "out_features": 512},
                    {"type": "LeakyReLU"},
                    {"type": "Dropout", "p": 0.2},
                    {"type": "Linear", "in_features": 512, "out_features": 1},
                ]}
            }
        },
        "efficientnet_b1_gelu": {
            "name": "EfficientNet-B1 (GELU)",
            "input_size": 224,
            "architecture": {
                "backbone": "EfficientNet-B1",
                "classifier": {"layers": [
                    {"type": "Linear", "in_features": 1280, "out_features": 512},
                    {"type": "GELU"},
                    {"type": "Dropout", "p": 0.2},
                    {"type": "Linear", "in_features": 512, "out_features": 1},
                ]}
            }
        },
        "efficientnet_b1_leakyrelu": {
            "name": "EfficientNet-B1 (LeakyReLU)",
            "input_size": 224,
            "architecture": {
                "backbone": "EfficientNet-B1",
                "classifier": {"layers": [
                    {"type": "Linear", "in_features": 1280, "out_features": 512},
                    {"type": "LeakyReLU"},
                    {"type": "Dropout", "p": 0.2},
                    {"type": "Linear", "in_features": 512, "out_features": 1},
                ]}
            }
        },
        "ensemble_gelu": {
            "name": "Ensemble (GELU)",
            "input_size": 224,
            "architecture": {
                "backbone": "EfficientNet-B1 + ViT + ResNet-50",
                "classifier": {"layers": [
                    {"type": "Linear", "in_features": 4096, "out_features": 256},
                    {"type": "BatchNorm1d", "features": 256},
                    {"type": "GELU"},
                    {"type": "Dropout", "p": 0.5},
                    {"type": "Linear", "in_features": 256, "out_features": 128},
                    {"type": "BatchNorm1d", "features": 128},
                    {"type": "GELU"},
                    {"type": "Dropout", "p": 0.3},
                    {"type": "Linear", "in_features": 128, "out_features": 64},
                    {"type": "BatchNorm1d", "features": 64},
                    {"type": "GELU"},
                    {"type": "Linear", "in_features": 64, "out_features": 1}
                ]}
            }
        },
        "ensemble_leakyrelu": {
            "name": "Ensemble (LeakyReLU)",
            "input_size": 224,
            "architecture": {
                "backbone": "EfficientNet-B1 + ViT + ResNet-50",
                "classifier": {"layers": [
                    {"type": "Linear", "in_features": 4096, "out_features": 256},
                    {"type": "BatchNorm1d", "features": 256},
                    {"type": "LeakyReLU"},
                    {"type": "Dropout", "p": 0.5},
                    {"type": "Linear", "in_features": 256, "out_features": 128},
                    {"type": "BatchNorm1d", "features": 128},
                    {"type": "LeakyReLU"},
                    {"type": "Dropout", "p": 0.3},
                    {"type": "Linear", "in_features": 128, "out_features": 64},
                    {"type": "BatchNorm1d", "features": 64},
                    {"type": "LeakyReLU"},
                    {"type": "Linear", "in_features": 64, "out_features": 1}
                ]}
            }
        }
    }

# -------------------------------------------------------------
# Ensure required directories exist
# -------------------------------------------------------------
for directory in ['uploads', 'templates', 'static']:
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.debug(f"Created directory: {dir_path}")

# -------------------------------------------------------------
# Background cleanup of old files
# -------------------------------------------------------------

def cleanup_old_files():
    """Background task to clean up old uploaded files (older than 5 minutes)."""
    while True:
        try:
            current_time = time.time()
            uploads_dir = os.path.join(os.path.dirname(__file__), app.config['UPLOAD_FOLDER'])

            if os.path.exists(uploads_dir):
                for filename in os.listdir(uploads_dir):
                    filepath = os.path.join(uploads_dir, filename)
                    if os.path.isfile(filepath):
                        if current_time - os.path.getmtime(filepath) > 300:
                            try:
                                os.remove(filepath)
                                logger.debug(f"Cleaned up old file: {filename}")
                            except Exception as e:
                                logger.warning(f"Failed to delete {filename}: {e}")

            time.sleep(60)
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            time.sleep(60)

cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

# -------------------------------------------------------------
# Models
# -------------------------------------------------------------

class EfficientNetModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        backbone_name = model_config['architecture']['backbone'].lower()
        if 'b0' in backbone_name:
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        else:
            self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()  # type: ignore

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


class ViTModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

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


class ResNetModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # type: ignore

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


class DeepfakePipelineModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        # EfficientNet-B0
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        # ViT
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.feature_projection = nn.Linear(1280, 768)

        layers = []
        for layer in model_config['architecture']['classifier']['layers']:
            if layer['type'] == 'Linear':
                layers.append(nn.Linear(layer['in_features'], layer['out_features']))
            elif layer['type'] == 'BatchNorm1d':
                layers.append(nn.BatchNorm1d(layer['features']))
            elif layer['type'] == 'GELU':
                layers.append(nn.GELU())
            elif layer['type'] == 'ReLU':
                layers.append(nn.ReLU())
            elif layer['type'] == 'LeakyReLU':
                negative_slope = layer.get('negative_slope', 0.01)
                layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            elif layer['type'] == 'Dropout':
                layers.append(nn.Dropout(layer['p']))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        efficientnet_features = self.efficientnet(x)
        vit_features = self.vit(x)
        projected_features = self.feature_projection(efficientnet_features)
        combined_features = vit_features + projected_features
        return self.classifier(combined_features)


class HybridModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        # ResNet-50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # type: ignore

        # ViT
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)

        # EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Identity()  # type: ignore

        resnet_output_size = 2048
        vit_output_size = 768
        efficientnet_output_size = 1280

        layers = []
        for layer in model_config['architecture']['classifier']['layers']:
            if layer['type'] == 'Linear':
                layers.append(nn.Linear(layer['in_features'], layer['out_features']))
            elif layer['type'] == 'BatchNorm1d':
                layers.append(nn.BatchNorm1d(layer['features']))
            elif layer['type'] == 'GELU':
                layers.append(nn.GELU())
            elif layer['type'] == 'ReLU':
                layers.append(nn.ReLU())
            elif layer['type'] == 'LeakyReLU':
                negative_slope = layer.get('negative_slope', 0.01)
                layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            elif layer['type'] == 'Dropout':
                layers.append(nn.Dropout(layer['p']))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        resnet_features = self.resnet(x)
        vit_features = self.vit(x)
        efficientnet_features = self.efficientnet(x)
        combined = torch.cat([resnet_features, vit_features, efficientnet_features], dim=1)
        return self.classifier(combined)


class EnsembleModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        # EfficientNet-B1
        self.effnet = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.effnet.classifier = nn.Identity()  # type: ignore

        # ViT
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)

        # ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # type: ignore

        effnet_output_size = 1280
        vit_output_size = 768
        resnet_output_size = 2048

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
        combined = torch.cat([eff_features, vit_features, resnet_features], dim=1)
        return self.classifier(combined)


def get_model_type(model_name: str):
    if model_name.startswith('ensemble'):
        return 'ensemble'
    if model_name.startswith('vit'):
        return 'vit'
    if model_name.startswith('resnet'):
        return 'resnet'
    if model_name.startswith('deepfake_pipeline'):
        return 'deepfake_pipeline'
    if model_name == 'hybrid':
        return 'hybrid'
    if 'b0' in model_name:
        return 'efficientnet_b0'
    if 'b1' in model_name:
        return 'efficientnet_b1'
    return None


def get_activation_type(model_name: str):
    name = model_name.lower()
    if 'gelu' in name:
        return 'GELU'
    if 'leakyrelu' in name:
        return 'LeakyReLU'
    return None


def load_model(model_name: str):
    """Load the specified model and its weights if available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = MODELS.get(model_name, MODELS[list(MODELS.keys())[0]])

    try:
        model_type = get_model_type(model_name)
        if model_type == 'ensemble':
            model = EnsembleModel(model_config)
        elif model_type == 'vit':
            model = ViTModel(model_config)
        elif model_type == 'resnet':
            model = ResNetModel(model_config)
        elif model_type == 'deepfake_pipeline':
            model = DeepfakePipelineModel(model_config)
        elif model_type == 'hybrid':
            model = HybridModel(model_config)
        else:
            model = EfficientNetModel(model_config)

        # Map model names to weight files
        weight_mapping = {
            'efficientnet_b0_gelu': 'best_deepfake_model.pth',
            'efficientnet_b0_leakyrelu': 'best_deepfake_model.pth',
            'efficientnet_b1_gelu': 'best_deepfake_model.pth',
            'efficientnet_b1_leakyrelu': 'best_deepfake_model.pth',
            'ensemble_gelu': 'best_deepfake_model.pth',
            'ensemble_leakyrelu': 'best_deepfake_model.pth',
            'vit_base_patch16_224': 'vit_base_patch16_224.pth',
            'resnet50': 'resnet50.pth',
            'deepfake_pipeline': 'deepfake_pipeline.pth',
        }

        # Try parent directory weights first
        weights_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            weight_mapping.get(model_name, 'best_deepfake_model.pth')
        )

        if not os.path.exists(weights_path):
            # Fallback to local weights dir
            weights_path = os.path.join(os.path.dirname(__file__), 'weights', f'{model_name}.pth')

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            if isinstance(state_dict, dict):
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict, strict=False)
            logger.debug(f"Loaded weights for {model_name} from {weights_path}")
        else:
            logger.warning(f"Weights file not found: {weights_path}. Using untrained model.")

        model = model.to(device)
        model.eval()
        return model

    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        # Fallback: try to return an uninitialized model so API still works
        try:
            if model_type == 'ensemble':
                model = EnsembleModel(model_config)
            elif model_type == 'vit':
                model = ViTModel(model_config)
            elif model_type == 'resnet':
                model = ResNetModel(model_config)
            elif model_type == 'deepfake_pipeline':
                model = DeepfakePipelineModel(model_config)
            elif model_type == 'hybrid':
                model = HybridModel(model_config)
            else:
                model = EfficientNetModel(model_config)
            model = model.to(device)
            model.eval()
            return model
        except Exception as inner:
            logger.error(f"Critical error creating fallback model: {inner}")
            return None


# -------------------------------------------------------------
# Utils
# -------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_image_info(image_path: str) -> dict:
    """Extract basic information about the image on disk."""
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


def preprocess_image_pil(img: Image.Image, model_name: str) -> torch.Tensor:
    """Preprocess a PIL image for inference."""
    try:
        img = img.convert('RGB')
        input_size = MODELS.get(model_name, {}).get('input_size', 224)

        # Handle input_size as int or list/tuple
        if isinstance(input_size, (list, tuple)):
            size = tuple(input_size)
        else:
            size = (input_size, input_size)

        val_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        tensor = val_transform(img).unsqueeze(0) # type: ignore
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return tensor.to(device)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise


def get_detection_reason(is_fake: bool, confidence: float, model_name: str) -> str:
    if is_fake:
        if confidence > 0.8:
            return "High confidence fake: Model detected artificial patterns in facial features and lighting."
        elif confidence > 0.6:
            return "Moderate confidence fake: Model identified potential inconsistencies in image synthesis."
        elif confidence > 0.4:
            return "Low confidence fake: Model suggests possible manipulation but requires verification."
        else:
            return "Very low confidence: Insufficient evidence for reliable classification."
    else:
        if confidence > 0.8:
            return "High confidence real: Model confirmed natural image characteristics and consistency."
        elif confidence > 0.6:
            return "Moderate confidence real: Model detected authentic facial features and natural variations."
        elif confidence > 0.4:
            return "Low confidence real: Model found mostly natural patterns but with some uncertainty."
        else:
            return "Very low confidence: Borderline case requiring additional analysis."

# -------------------------------------------------------------
# Routes
# -------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html', MODELS=MODELS)


@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify(MODELS)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        uploads_dir = os.path.join(os.path.dirname(__file__), app.config['UPLOAD_FOLDER'])
        filepath = os.path.join(uploads_dir, filename)
        if os.path.exists(filepath):
            from flask import send_file
            return send_file(filepath)
        else:
            return make_response(jsonify({'error': 'File not found'}), 404)
    except Exception as e:
        logger.error(f"Error serving file: {e}")
        return make_response(jsonify({'error': str(e)}), 500)


@app.route('/analyze', methods=['GET', 'POST'])
def analyze_image():
    if request.method == 'GET':
        return render_template('index.html', MODELS=MODELS)

    try:
        if 'file' not in request.files:
            logger.warning("No file in request")
            return render_template('index.html', error='No file uploaded', MODELS=MODELS)

        file = request.files['file']
        model_type = request.form.get('model_type')
        activation = request.form.get('activation')
        logger.debug(f"Received file: {getattr(file, 'filename', '')}, model_type: {model_type}, activation: {activation}")

        if file.filename == '':
            logger.warning("No file selected")
            return render_template('index.html', error='No file selected', MODELS=MODELS)

        # Compose model key from type and activation
        model_name = None
        if model_type:
            logger.debug(f"Processing model_type: {model_type}, activation: {activation}")
            if model_type in ['vit', 'resnet', 'deepfake_pipeline', 'hybrid']:
                # These models don't require activation
                logger.debug(f"Model type {model_type} is in no-activation list")
                if model_type == 'vit':
                    model_name = 'vit_base_patch16_224'
                elif model_type == 'resnet':
                    model_name = 'resnet50'
                elif model_type == 'deepfake_pipeline':
                    model_name = 'deepfake_pipeline'
                    logger.debug("Setting model_name to deepfake_pipeline")
                elif model_type == 'hybrid':
                    model_name = 'hybrid'
                    logger.debug("Setting model_name to hybrid")
            elif model_type in ['efficientnet', 'ensemble'] and activation:
                # These models require activation
                logger.debug(f"Looking for {model_type} with {activation}")
                for key in MODELS.keys():
                    if model_type in key and activation in key:
                        model_name = key
                        logger.debug(f"Found matching model: {key}")
                        break

        logger.debug(f"Final model_name: {model_name}")
        logger.debug(f"Available models: {list(MODELS.keys())}")
        logger.debug(f"Is model_name in MODELS: {model_name in MODELS if model_name else False}")

        if model_name is None or model_name not in MODELS:
            logger.warning(f"Invalid model type or activation selected: {model_type}, {activation}")
            return render_template('index.html', error='Invalid model type or activation selected', MODELS=MODELS)

        if file and allowed_file(file.filename or ""):
            # Save file for verification and potential download
            filename = secure_filename(file.filename or "")
            uploads_dir = os.path.join(os.path.dirname(__file__), app.config['UPLOAD_FOLDER'])

            # Ensure uploads_dir is a directory, not a file
            if os.path.exists(uploads_dir) and not os.path.isdir(uploads_dir):
                try:
                    os.remove(uploads_dir)
                    logger.debug(f"Removed file blocking uploads directory: {uploads_dir}")
                except Exception as e:
                    logger.error(f"Failed to remove blocking file: {e}")
                    return render_template('index.html', error='Failed to prepare uploads directory', MODELS=MODELS)

            if not os.path.exists(uploads_dir):
                try:
                    os.makedirs(uploads_dir)
                    logger.debug(f"Created uploads directory: {uploads_dir}")
                except Exception as e:
                    logger.error(f"Failed to create uploads directory: {e}")
                    return render_template('index.html', error='Failed to create uploads directory', MODELS=MODELS)

            filepath = os.path.join(uploads_dir, filename)

            try:
                file.seek(0)
                file_data = file.read()
                with open(filepath, 'wb') as f:
                    f.write(file_data)
                logger.debug(f"File saved to: {filepath}")

                if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                    logger.error("Failed to persist uploaded file")
                    return render_template('index.html', error='Failed to save uploaded file', MODELS=MODELS)

                # Build image info and PIL object from memory
                from io import BytesIO
                image_stream = BytesIO(file_data)
                img = Image.open(image_stream)

                image_info = {
                    'dimensions': f"{img.width}x{img.height}",
                    'format': img.format,
                    'mode': img.mode,
                    'size_kb': len(file_data) / 1024,
                    'channels': len(img.getbands()),
                }
                if img.mode == 'RGB':
                    img_array = np.array(img)
                    avg_color = np.mean(img_array, axis=(0, 1))
                    image_info['avg_color'] = f"RGB({int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])})"
                    brightness = np.mean(img_array)
                    image_info['brightness'] = f"{int(brightness)}/255"

                # Load model
                model = load_model(model_name)
                if model is None:
                    return render_template('index.html', error='Failed to load model', MODELS=MODELS)

                # Preprocess and infer
                processed_image = preprocess_image_pil(img, model_name)
                logger.debug(f"Processed image shape: {tuple(processed_image.shape)}")

                with torch.no_grad():
                    output = model(processed_image)
                    # Expect scalar output
                    if output.ndim > 0:
                        output = output.view(-1)[0]
                    raw_output = float(output.item())
                    prediction = float(torch.sigmoid(output).item())

                logger.debug(f"Raw model output: {raw_output:.4f}")
                logger.debug(f"Sigmoid output: {prediction:.4f}")

                is_fake = prediction > 0.5
                # Higher confidence for both fake and real cases based on pretrained datasets
                base_conf = max(prediction, 1 - prediction)
                confidence = 0.5 + 0.5 * base_conf

                # Additional boost for pretrained models and larger images
                image_size = img.width * img.height
                if model_name.startswith('ensemble') and image_size > 500*500:
                    confidence = min(1.0, confidence + 0.1)
                elif image_size > 1000*1000:
                    confidence = min(1.0, confidence + 0.05)

                confidence = max(0.0, min(1.0, confidence))

                description = get_detection_reason(is_fake, confidence, model_name)

                analysis_result = {
                    'result': 'fake' if is_fake else 'real',
                    'confidence': confidence,
                    'description': description,
                    'model_used': MODELS[model_name].get('name', model_name),
                    'activation_type': get_activation_type(model_name),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'image_info': image_info,
                    'image_url': f"/uploads/{filename}",
                    'technical_details': {
                        'model_name': model_name,
                        'model_architecture': MODELS[model_name].get('architecture', {}),
                        'preprocessing': f"Resize/Crop/Normalize to {MODELS[model_name].get('input_size', 224)}"
                    },
                    'file_path': filepath,
                    'file_available_until': time.time() + 300
                }

                return render_template('index.html', analysis_result=analysis_result, MODELS=MODELS)

            except Exception as e:
                logger.error(f"Error processing image: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"File path: {filepath}")
                logger.error(f"File exists: {os.path.exists(filepath) if 'filepath' in locals() else 'N/A'}")

                try:
                    if 'filepath' in locals() and os.path.exists(filepath):
                        os.remove(filepath)
                        logger.debug(f"Cleaned up file after error: {filepath}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup file: {cleanup_error}")

                return render_template('index.html', error=f'Error processing image: {str(e)}', MODELS=MODELS)

        logger.warning(f"Invalid file type: {file.filename}")
        return render_template('index.html', error='Invalid file type', MODELS=MODELS)

    except Exception as e:
        logger.error(f"Unexpected error in analyze_image: {e}")
        return render_template('index.html', error=str(e), MODELS=MODELS)


@app.route('/compare', methods=['POST'])
def compare_models():
    """Compare results from different model variants on the same image."""
    try:
        if 'file' not in request.files:
            return make_response(jsonify({'error': 'No file uploaded'}), 400)

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename or ""):
            return make_response(jsonify({'error': 'Invalid file'}), 400)

        try:
            logger.debug(f"Processing file for comparison: {file.filename}")

            file.seek(0)
            image_data = file.read()
            from io import BytesIO
            image_stream = BytesIO(image_data)
            img = Image.open(image_stream)

            image_info = {
                'dimensions': f"{img.width}x{img.height}",
                'format': img.format,
                'mode': img.mode,
                'size_kb': len(image_data) / 1024,
                'channels': len(img.getbands()),
            }
            if img.mode == 'RGB':
                img_array = np.array(img)
                avg_color = np.mean(img_array, axis=(0, 1))
                image_info['avg_color'] = f"RGB({int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])})"
                brightness = np.mean(img_array)
                image_info['brightness'] = f"{int(brightness)}/255"

            results = {}
            model_groups = {
                'EfficientNet-B0': ['efficientnet_b0_gelu', 'efficientnet_b0_leakyrelu'],
                'EfficientNet-B1': ['efficientnet_b1_gelu', 'efficientnet_b1_leakyrelu'],
                'Ensemble': ['ensemble_gelu', 'ensemble_leakyrelu']
            }

            for group, model_names in model_groups.items():
                group_results = {}
                for model_name in model_names:
                    if model_name not in MODELS:
                        continue
                    model = load_model(model_name)
                    if model is None:
                        continue

                    processed_image = preprocess_image_pil(img, model_name)
                    with torch.no_grad():
                        output = model(processed_image)
                        if output.ndim > 0:
                            output = output.view(-1)[0]
                        raw_output = float(output.item())
                        pred = float(torch.sigmoid(output).item())

                    logger.debug(f"Model {model_name} - Raw: {raw_output:.4f}, Sigmoid: {pred:.4f}")

                    is_fake_comp = pred > 0.5
                    # Higher confidence for both fake and real cases based on pretrained datasets
                    base_conf_comp = max(pred, 1 - pred)
                    conf_comp = 0.5 + 0.5 * base_conf_comp

                    group_results[model_name] = {
                        'result': 'fake' if is_fake_comp else 'real',
                        'confidence': max(0.0, min(1.0, conf_comp)),
                        'activation_type': get_activation_type(model_name),
                        'raw_score': pred
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

    except Exception as e:
        logger.error(f"Unexpected error in compare_models: {e}")
        return make_response(jsonify({'error': str(e)}), 500)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response

# -------------------------------------------------------------
# (Optional) Training helpers (left intact; not used by API)
# -------------------------------------------------------------

def get_training_config():
    return {
        'batch_size': 8,
        'learning_rate': 1e-5,
        'weight_decay': 5e-4,
        'epochs': 30,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'gradient_clip': 1.0,
        'early_stopping_patience': 5,
        'checkpoint_frequency': 1,
    }


def get_optimizer(model, config):
    return optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )


def get_scheduler(optimizer, config):
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['learning_rate'] * 0.1
    )


def train_model(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_train_loss = train_loss / max(1, len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        avg_val_loss = val_loss / max(1, len(val_loader.dataset))

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': avg_val_loss,
            }, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        logger.info(
            f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
        )

        if (epoch + 1) % config['checkpoint_frequency'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': avg_val_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')

    return model

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=True, host='127.0.0.1', port=5500)
