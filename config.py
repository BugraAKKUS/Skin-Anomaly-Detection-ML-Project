class Config:
    # Paths - ADJUST THESE TO YOUR DATA LOCATION
    DATA_DIR = './HAM10000_images'  # Directory containing all images
    METADATA_PATH = './HAM10000_metadata.csv'  # Path to metadata CSV
    
    # Model settings
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 3
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