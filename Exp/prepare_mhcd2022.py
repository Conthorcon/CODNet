import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
import shutil

def parse_voc_annotation(ann_path):
    """Parse VOC format annotation"""
    tree = ET.parse(ann_path)
    root = tree.getroot()
    
    boxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        boxes.append({
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],  # [x, y, w, h]
            'category_id': 1,  # Assuming single class
            'iscrowd': 0
        })
    
    return boxes

def prepare_mhcd2022_coco(base_dir, output_dir):
    """Prepare MHCD2022 dataset in COCO format"""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    
    # Read train/val splits
    with open(os.path.join(base_dir, 'ImageSets/Main/train.txt'), 'r') as f:
        train_ids = [line.strip() for line in f.readlines()]
    with open(os.path.join(base_dir, 'ImageSets/Main/val.txt'), 'r') as f:
        val_ids = [line.strip() for line in f.readlines()]
    
    # Process train set
    train_images = []
    train_annotations = []
    ann_id = 1
    
    for img_id in train_ids:
        # Copy image
        src_img = os.path.join(base_dir, 'JPEGImages', f'{img_id}.jpg')
        dst_img = os.path.join(output_dir, 'train', f'{img_id}.jpg')
        shutil.copy2(src_img, dst_img)
        
        # Get image size
        img = Image.open(src_img)
        width, height = img.size
        
        # Add image entry
        train_images.append({
            'id': len(train_images) + 1,
            'file_name': f'{img_id}.jpg',
            'height': height,
            'width': width
        })
        
        # Parse annotations
        ann_path = os.path.join(base_dir, 'Annotations', f'{img_id}.xml')
        boxes = parse_voc_annotation(ann_path)
        
        # Add annotation entries
        for box in boxes:
            train_annotations.append({
                'id': ann_id,
                'image_id': len(train_images),
                'category_id': box['category_id'],
                'bbox': box['bbox'],
                'area': box['bbox'][2] * box['bbox'][3],
                'iscrowd': box['iscrowd']
            })
            ann_id += 1
    
    # Process validation set
    val_images = []
    val_annotations = []
    
    for img_id in val_ids:
        # Copy image
        src_img = os.path.join(base_dir, 'JPEGImages', f'{img_id}.jpg')
        dst_img = os.path.join(output_dir, 'val', f'{img_id}.jpg')
        shutil.copy2(src_img, dst_img)
        
        # Get image size
        img = Image.open(src_img)
        width, height = img.size
        
        # Add image entry
        val_images.append({
            'id': len(val_images) + 1,
            'file_name': f'{img_id}.jpg',
            'height': height,
            'width': width
        })
        
        # Parse annotations
        ann_path = os.path.join(base_dir, 'Annotations', f'{img_id}.xml')
        boxes = parse_voc_annotation(ann_path)
        
        # Add annotation entries
        for box in boxes:
            val_annotations.append({
                'id': ann_id,
                'image_id': len(val_images),
                'category_id': box['category_id'],
                'bbox': box['bbox'],
                'area': box['bbox'][2] * box['bbox'][3],
                'iscrowd': box['iscrowd']
            })
            ann_id += 1
    
    # Save COCO format annotations
    train_coco = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': [{'id': 1, 'name': 'object'}]
    }
    
    val_coco = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': [{'id': 1, 'name': 'object'}]
    }
    
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_coco, f)
    
    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_coco, f)

if __name__ == '__main__':
    base_dir = 'MHCD2022'
    output_dir = 'MHCD2022_COCO'
    prepare_mhcd2022_coco(base_dir, output_dir) 