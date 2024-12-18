import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


# Function to normalize YOLO annotations
def normalize_yolo(x1, y1, x2, y2, img_width, img_height, x1_bar=None, y1_bar=None, x2_bar=None, y2_bar=None):
    
    x_lmk_center = (x1 + x2) / 2 / img_width
    y_lmk_center = (y1 + y2) / 2 / img_height
    width_lmk = abs(x2 - x1) / img_width
    height_lmk = abs(y2 - y1) / img_height
    
    if x1_bar is not None:
        x_bar_center = (x1_bar + x2_bar) / 2 / img_width
        y_bar_center = (y1_bar + y2_bar) / 2 / img_height
        width_bar = abs(x2_bar - x1_bar) / img_width
        height_bar = abs(y2_bar - y1_bar) / img_height
        return x_lmk_center, y_lmk_center, width_lmk, height_lmk, x_bar_center, y_bar_center, width_bar, height_bar
    return x_lmk_center, y_lmk_center, width_lmk, height_lmk

def normalize(x, dim):
    return x/dim
    
def create_thorax_dataset(csv_path, image_folder, image_dim):
    # Load the CSV file
    
    data = pd.read_csv(csv_path).drop_duplicates()
    
    # Create directories
    base_dir = os.getcwd()
    os.makedirs(os.path.join(base_dir,'thorax_dataset/images/train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'thorax_dataset/images/val'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'thorax_dataset/images/test'), exist_ok=True)
    
    os.makedirs(os.path.join(base_dir,'thorax_dataset/labels/train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'thorax_dataset/labels/val'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'thorax_dataset/labels/test'), exist_ok=True)


    # Resize images and prepare annotations
    image_paths = []
    annotations = []

    for i, row in data.iterrows():
        img_name = row['ant']
        x1, y1, x2, y2 = row['x1_lmk'], row['y1_lmk'], row['x2_lmk'], row['y2_lmk']
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            continue

        # Open the image
        img = Image.open(img_path)
        original_width, original_height = img.size

        # Resize the image
        resized_img = img.resize((image_dim, image_dim), Image.Resampling.LANCZOS)

        # Save the resized image
        resized_img_path = os.path.join(base_dir, 'thorax_dataset/images', img_name)
        resized_img.save(resized_img_path)
        
        
        # Normalize YOLO coordinates
        x_center, y_center, width, height = normalize_yolo(
            x1, y1, x2, y2, image_dim, image_dim
        )

        x1_normalized = normalize(x1, image_dim)
        y1_normalized = normalize(y1, image_dim)
        x2_normalized = normalize(x2, image_dim)
        y2_normalized = normalize(y2, image_dim)

        # Save annotation
        annotation = f"0 {x_center} {y_center} {width} {height} {x1_normalized} {y1_normalized} {x2_normalized} {y2_normalized}\n"  # Assuming class id 0
        annotations.append((img_name, annotation))

    # print(annotations)
    #create all splits train, val, test
    train_files, val_test_files = train_test_split(annotations, test_size=0.4, random_state=42)
    val_files, test_files = train_test_split(val_test_files, test_size=0.5, random_state=42)
  
    # Move images and create labels
    #train 
    for img_name, annotation in train_files:
        base_name = os.path.splitext(img_name)[0]
        original_img_path = os.path.join(base_dir, 'thorax_dataset/images', img_name)
        train_img_path = os.path.join(base_dir, 'thorax_dataset/images/train', img_name)
        if os.path.exists(original_img_path):
            os.rename(original_img_path, train_img_path)
            
        label_path = os.path.join(base_dir, 'thorax_dataset/labels/train', f'{base_name}.txt')
        with open(label_path, 'w') as f:
            f.write(annotation)
    #val
    for img_name, annotation in val_files:
        base_name = os.path.splitext(img_name)[0]
        original_img_path = os.path.join(base_dir, 'thorax_dataset/images', img_name)
        val_img_path = os.path.join(base_dir, 'thorax_dataset/images/val', img_name)
        if os.path.exists(original_img_path):
            os.rename(original_img_path, val_img_path)
        
        label_path = os.path.join(base_dir, 'thorax_dataset/labels/val', f'{base_name}.txt')
        with open(label_path, 'w') as f:
            f.write(annotation)
            
    #test
    for img_name, annotation in test_files:
        base_name = os.path.splitext(img_name)[0]
        original_img_path = os.path.join(base_dir, 'thorax_dataset/images', img_name)
        test_img_path = os.path.join(base_dir, 'thorax_dataset/images/test', img_name)
        if os.path.exists(original_img_path):
            os.rename(original_img_path, test_img_path)
        
        label_path = os.path.join(base_dir, 'thorax_dataset/labels/test', f'{base_name}.txt')
        with open(label_path, 'w') as f:
            f.write(annotation)
            
    #create the asscoiated .yaml file for YOLO with correct train/val/test directories
    yaml_path = os.path.join(base_dir, 'thorax_dataset/dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'train: {os.path.join(base_dir, "thorax_dataset/images/train")}\n')
        f.write(f'val: {os.path.join(base_dir, "thorax_dataset/images/val")}\n')
        f.write(f'test: {os.path.join(base_dir, "thorax_dataset/images/test")}\n')
        f.write('nc: 1\n')
        f.write('names: ["thorax"]\n')
        f.write('keypoint_names: ["thorax_start", "thorax_end"]\n')
        f.write('num_keypoints: 2\n')
        f.write('kpt_shape: [2, 2]\n')
    
    print(f"Created dataset.yaml at: {yaml_path}")
    print("Data preparation complete.")
    
    
    
    
def create_thorax_and_scale_dataset(csv_path, image_folder, image_dim):
    # Load the CSV file
    
    data = pd.read_csv(csv_path).drop_duplicates()
    
    # Create directories
    base_dir = os.getcwd()
    os.makedirs(os.path.join(base_dir,'thorax_and_scale_dataset/images/train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'thorax_and_scale_dataset/images/val'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'thorax_and_scale_dataset/images/test'), exist_ok=True)
    
    os.makedirs(os.path.join(base_dir,'thorax_and_scale_dataset/labels/train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'thorax_and_scale_dataset/labels/val'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'thorax_and_scale_dataset/labels/test'), exist_ok=True)


    # Resize images and prepare annotations
    annotations = []

    for i, row in data.iterrows():
        img_name = row['ant']
        
        x1, y1, x2, y2 = row['x1_lmk'], row['y1_lmk'], row['x2_lmk'], row['y2_lmk']
        x1_bar, y1_bar, x2_bar, y2_bar = row['x1_bar'], row['y1_bar'], row['x2_bar'], row['y2_bar']
        
        img_path = os.path.join(image_folder, img_name)

        if not os.path.exists(img_path):
            # print(f"Image {img_path} not found. Skipping...")
            continue

        # Open the image
        img = Image.open(img_path)

        # Resize the image
        resized_img = img.resize((image_dim, image_dim), Image.Resampling.LANCZOS)

        # Save the resized image
        resized_img_path = os.path.join(base_dir, 'thorax_and_scale_dataset/images', img_name)
        resized_img.save(resized_img_path)
        
        
        # Normalize YOLO coordinates
        x_lmk_center, y_lmk_center, width_lmk, height_lmk, x_bar_center, y_bar_center, width_bar, height_bar = normalize_yolo(
            x1, y1, x2, y2, image_dim, image_dim, x1_bar, y1_bar, x2_bar, y2_bar)

        x1_normalized = normalize(x1, image_dim)
        y1_normalized = normalize(y1, image_dim)
        x2_normalized = normalize(x2, image_dim)
        y2_normalized = normalize(y2, image_dim)
        
        x1_bar_normalized = normalize(x1_bar, image_dim)
        y1_bar_normalized = normalize(y1_bar, image_dim)
        x2_bar_normalized = normalize(x2_bar, image_dim)
        y2_bar_normalized = normalize(y2_bar, image_dim)
        
        # Save annotation
        #<class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
        annotation_lmk = f"0 {x_lmk_center} {y_lmk_center} {width_lmk} {height_lmk} {x1_normalized} {y1_normalized} {x2_normalized} {y2_normalized} 0 0 0 0\n"  # Assuming class id 0
        annotation_bar = f"1 {x_bar_center} {y_bar_center} {width_bar} {height_bar} {x1_bar_normalized} {y1_bar_normalized} {x2_bar_normalized} {y2_bar_normalized} 0 0 0 0\n" 
        annotation = [annotation_lmk, annotation_bar]

        annotations.append((img_name, annotation))
        
    #create all splits train, val, test
    train_files, val_test_files = train_test_split(annotations, test_size=0.4, random_state=42)
    val_files, test_files = train_test_split(val_test_files, test_size=0.5, random_state=42)
  
    # Move images and create labels
    #train 
    for img_name, annotation in train_files:
        base_name = os.path.splitext(img_name)[0]
        original_img_path = os.path.join(base_dir, 'thorax_and_scale_dataset/images', img_name)
        train_img_path = os.path.join(base_dir, 'thorax_and_scale_dataset/images/train', img_name)
        if os.path.exists(original_img_path):
            os.rename(original_img_path, train_img_path)
            
        label_path = os.path.join(base_dir, 'thorax_and_scale_dataset/labels/train', f'{base_name}.txt')
        with open(label_path, 'w') as f:
            f.write(annotation[0])
            f.write(annotation[1])
    #val
    for img_name, annotation in val_files:
        base_name = os.path.splitext(img_name)[0]
        original_img_path = os.path.join(base_dir, 'thorax_and_scale_dataset/images', img_name)
        val_img_path = os.path.join(base_dir, 'thorax_and_scale_dataset/images/val', img_name)
        if os.path.exists(original_img_path):
            os.rename(original_img_path, val_img_path)
        
        label_path = os.path.join(base_dir, 'thorax_and_scale_dataset/labels/val', f'{base_name}.txt')
        with open(label_path, 'w') as f:
            f.write(annotation[0])
            f.write(annotation[1])
            
    #test
    for img_name, annotation in test_files:
        base_name = os.path.splitext(img_name)[0]
        original_img_path = os.path.join(base_dir, 'thorax_and_scale_dataset/images', img_name)
        test_img_path = os.path.join(base_dir, 'thorax_and_scale_dataset/images/test', img_name)
        if os.path.exists(original_img_path):
            os.rename(original_img_path, test_img_path)
        
        label_path = os.path.join(base_dir, 'thorax_and_scale_dataset/labels/test', f'{base_name}.txt')
        with open(label_path, 'w') as f:
            f.write(annotation[0])
            f.write(annotation[1])
            
    #create the asscoiated .yaml file for YOLO with correct train/val/test directories
    yaml_path = os.path.join(base_dir, 'thorax_and_scale_dataset/dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'train: {os.path.join(base_dir, "thorax_and_scale_dataset/images/train")}\n')
        f.write(f'val: {os.path.join(base_dir, "thorax_and_scale_dataset/images/val")}\n')
        f.write(f'test: {os.path.join(base_dir, "thorax_and_scale_dataset/images/test")}\n')
        f.write('nc: 2\n')
        f.write('names: ["thorax", "bar"]\n')
        f.write('keypoint_names: ["thorax_p1", "thorax_p2", "bar_p1", "bar_p2"]\n')
        f.write('num_keypoints: 4\n')
        f.write('kpt_shape: [4, 2]\n')
    
    print(f"Created dataset.yaml at: {yaml_path}")
    print("Data preparation complete.")
    
    
def create_scale_dataset(csv_path, image_folder, image_dim):
    # Load the CSV file
    
    data = pd.read_csv(csv_path).drop_duplicates()
    
    # Create directories
    base_dir = os.getcwd()
    os.makedirs(os.path.join(base_dir,'scale_dataset/images/train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'scale_dataset/images/val'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'scale_dataset/images/test'), exist_ok=True)
    
    os.makedirs(os.path.join(base_dir,'scale_dataset/labels/train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'scale_dataset/labels/val'), exist_ok=True)
    os.makedirs(os.path.join(base_dir,'scale_dataset/labels/test'), exist_ok=True)


    # Resize images and prepare annotations
    image_paths = []
    annotations = []

    for i, row in data.iterrows():
        img_name = row['ant']
        x1, y1, x2, y2 = row['x1_bar'], row['y1_bar'], row['x2_bar'], row['y2_bar']
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            continue

        # Open the image
        img = Image.open(img_path)
        original_width, original_height = img.size

        # Resize the image
        resized_img = img.resize((image_dim, image_dim), Image.Resampling.LANCZOS)

        # Save the resized image
        resized_img_path = os.path.join(base_dir, 'scale_dataset/images', img_name)
        resized_img.save(resized_img_path)
        
        
        # Normalize YOLO coordinates
        x_center, y_center, width, height = normalize_yolo(
            x1, y1, x2, y2, image_dim, image_dim
        )

        x1_normalized = normalize(x1, image_dim)
        y1_normalized = normalize(y1, image_dim)
        x2_normalized = normalize(x2, image_dim)
        y2_normalized = normalize(y2, image_dim)

        # Save annotation
        annotation = f"0 {x_center} {y_center} {width} {height} {x1_normalized} {y1_normalized} {x2_normalized} {y2_normalized}\n"  # Assuming class id 0
        annotations.append((img_name, annotation))

    # print(annotations)
    #create all splits train, val, test
    train_files, val_test_files = train_test_split(annotations, test_size=0.4, random_state=42)
    val_files, test_files = train_test_split(val_test_files, test_size=0.5, random_state=42)
  
    # Move images and create labels
    #train 
    for img_name, annotation in train_files:
        base_name = os.path.splitext(img_name)[0]
        original_img_path = os.path.join(base_dir, 'scale_dataset/images', img_name)
        train_img_path = os.path.join(base_dir, 'scale_dataset/images/train', img_name)
        if os.path.exists(original_img_path):
            os.rename(original_img_path, train_img_path)
            
        label_path = os.path.join(base_dir, 'scale_dataset/labels/train', f'{base_name}.txt')
        with open(label_path, 'w') as f:
            f.write(annotation)
    #val
    for img_name, annotation in val_files:
        base_name = os.path.splitext(img_name)[0]
        original_img_path = os.path.join(base_dir, 'scale_dataset/images', img_name)
        val_img_path = os.path.join(base_dir, 'scale_dataset/images/val', img_name)
        if os.path.exists(original_img_path):
            os.rename(original_img_path, val_img_path)
        
        label_path = os.path.join(base_dir, 'scale_dataset/labels/val', f'{base_name}.txt')
        with open(label_path, 'w') as f:
            f.write(annotation)
            
    #test
    for img_name, annotation in test_files:
        base_name = os.path.splitext(img_name)[0]
        original_img_path = os.path.join(base_dir, 'scale_dataset/images', img_name)
        test_img_path = os.path.join(base_dir, 'scale_dataset/images/test', img_name)
        if os.path.exists(original_img_path):
            os.rename(original_img_path, test_img_path)
        
        label_path = os.path.join(base_dir, 'scale_dataset/labels/test', f'{base_name}.txt')
        with open(label_path, 'w') as f:
            f.write(annotation)
            
    #create the asscoiated .yaml file for YOLO with correct train/val/test directories
    yaml_path = os.path.join(base_dir, 'scale_dataset/dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'train: {os.path.join(base_dir, "scale_dataset/images/train")}\n')
        f.write(f'val: {os.path.join(base_dir, "scale_dataset/images/val")}\n')
        f.write(f'test: {os.path.join(base_dir, "scale_dataset/images/test")}\n')
        f.write('nc: 1\n')
        f.write('names: ["scale"]\n')
        f.write('keypoint_names: ["scale_start", "scale_end"]\n')
        f.write('num_keypoints: 2\n')
        f.write('kpt_shape: [2, 2]\n')
    
    print(f"Created dataset.yaml at: {yaml_path}")
    print("Data preparation complete.")
    

if __name__ == '__main__':
    csv_path = './csv/landmark_digitalization_3.csv' #'./csv/new_annotations.csv'
    image_folder = './original' # "../../original"
    image_dim = 640
    
    # create_thorax_dataset(csv_path, image_folder, image_dim)
    create_thorax_and_scale_dataset(csv_path, image_folder, image_dim)
    # create_scale_dataset(csv_path, image_folder, image_dim)
    