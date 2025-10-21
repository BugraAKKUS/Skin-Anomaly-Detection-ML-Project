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