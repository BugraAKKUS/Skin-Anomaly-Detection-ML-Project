"""
HAM10000 Skin Lesion Classification System
Classifies dermatoscopic images into 7 diagnostic categories
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ====================== CONFIGURATION ======================
class Config:
    # Paths - ADJUST THESE TO YOUR DATA LOCATION
    DATA_DIR = './HAM10000_images'  # Directory containing all images
    METADATA_PATH = './HAM10000_metadata.csv'  # Path to metadata CSV
    
    # Model settings
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.0001
    NUM_CLASSES = 7
    
    # Class labels
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    # Training settings
    EARLY_STOPPING_PATIENCE = 7
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data split ratios
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15


# ====================== DATASET CLASS ======================
class HAM10000Dataset(Dataset):
    """Custom Dataset for HAM10000 images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ====================== DATA LOADING ======================
def load_ham10000_data(data_dir, metadata_path):
    """Load HAM10000 dataset with metadata"""
    
    print("Loading metadata...")
    df = pd.read_csv(metadata_path)
    
    # Create full image paths
    image_paths = []
    labels = []
    missing_files = 0
    
    for idx, row in df.iterrows():
        image_id = row['image_id']
        diagnosis = row['dx']
        
        # Try different possible image extensions and subdirectories
        possible_paths = [
            os.path.join(data_dir, f"{image_id}.jpg"),
            os.path.join(data_dir, f"{image_id}.jpeg"),
            os.path.join(data_dir, 'HAM10000_images_part_1', f"{image_id}.jpg"),
            os.path.join(data_dir, 'HAM10000_images_part_2', f"{image_id}.jpg"),
        ]
        
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path:
            image_paths.append(img_path)
            labels.append(Config.CLASS_TO_IDX[diagnosis])
        else:
            missing_files += 1
    
    if missing_files > 0:
        print(f"Warning: {missing_files} images not found")
    
    print(f"Loaded {len(image_paths)} images")
    return image_paths, labels


# ====================== DATA PREPROCESSING ======================
def get_transforms(training=True):
    """Get image transforms for training and validation"""
    
    if training:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(image_paths, labels):
    """Split data and create data loaders with balanced sampling"""
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, 
        test_size=Config.TEST_SIZE, 
        stratify=labels,
        random_state=42
    )
    
    # Second split: separate train and validation
    val_size_adjusted = Config.VAL_SIZE / (1 - Config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"Training: {len(X_train)} images")
    print(f"Validation: {len(X_val)} images")
    print(f"Test: {len(X_test)} images")
    
    # Create datasets
    train_dataset = HAM10000Dataset(X_train, y_train, transform=get_transforms(training=True))
    val_dataset = HAM10000Dataset(X_val, y_val, transform=get_transforms(training=False))
    test_dataset = HAM10000Dataset(X_test, y_test, transform=get_transforms(training=False))
    
    # Handle class imbalance with weighted sampling
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ====================== MODEL ARCHITECTURE ======================
class SkinLesionClassifier(nn.Module):
    """CNN classifier using pre-trained EfficientNet-B0"""
    
    def __init__(self, num_classes=7):
        super(SkinLesionClassifier, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        self.backbone = models.efficientnet_b4(pretrained=True)
        
        # Get the number of input features for the classifier
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ====================== TRAINING ======================
class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/len(train_loader), 
                         'acc': 100.*correct/total})
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total


def train_model(model, train_loader, val_loader, num_epochs, device):
    """Full training loop with early stopping"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"\nTraining on {device}")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("\nEarly stopping triggered!")
            model.load_state_dict(early_stopping.best_model)
            break
    
    return model, history


# ====================== EVALUATION ======================
def evaluate_model(model, test_loader, device):
    """Comprehensive model evaluation"""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    print(f"Weighted F1-Score: {f1:.4f}\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=Config.CLASS_NAMES,
                                digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, Config.CLASS_NAMES)
    
    return accuracy, f1, cm


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")


def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history saved as 'training_history.png'")


# ====================== PREDICTION ======================
class SkinLesionPredictor:
    """Predictor class for new images"""
    
    def __init__(self, model_path, device):
        self.device = device
        self.model = SkinLesionClassifier(num_classes=Config.NUM_CLASSES).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.transform = get_transforms(training=False)
    
    def predict(self, image_path):
        """Predict the class of a single image"""
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        predicted_class = Config.CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item()
        
        # Get all class probabilities
        all_probs = {
            Config.CLASS_NAMES[i]: probabilities[0][i].item() 
            for i in range(Config.NUM_CLASSES)
        }
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'all_probabilities': all_probs
        }
    
    def predict_batch(self, image_paths):
        """Predict classes for multiple images"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            results.append(result)
        return results


# ====================== MAIN EXECUTION ======================
def main():
    """Main execution function"""
    
    print("="*60)
    print("HAM10000 Skin Lesion Classification")
    print("="*60)
    
    # Load data
    image_paths, labels = load_ham10000_data(Config.DATA_DIR, Config.METADATA_PATH)
    
    # Display class distribution
    print("\nClass Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"{Config.CLASS_NAMES[idx]}: {count} images ({count/len(labels)*100:.2f}%)")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(image_paths, labels)
    
    # Initialize model
    model = SkinLesionClassifier(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    model, history = train_model(
        model, train_loader, val_loader, 
        Config.NUM_EPOCHS, Config.DEVICE
    )
    
    # Save model
    model_save_path = 'skin_lesion_classifier.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to '{model_save_path}'")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    evaluate_model(model, test_loader, Config.DEVICE)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    
    # Example prediction
    print("\n" + "="*60)
    print("PREDICTION EXAMPLE")
    print("="*60)
    print("\nTo predict a new image:")
    print("```python")
    print("predictor = SkinLesionPredictor('skin_lesion_classifier.pth', Config.DEVICE)")
    print("result = predictor.predict('path/to/new/image.jpg')")
    print("print(f\"Predicted: {result['predicted_class']}\")")
    print("print(f\"Confidence: {result['confidence']:.2%}\")")
    print("```")


if __name__ == "__main__":
    main()