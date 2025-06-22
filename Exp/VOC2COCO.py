import os
import shutil
from glob import glob
from tqdm import tqdm

voc_dir = "MHCD2022/Annotations"
coco_dir = os.path.join("MHCD2022", '/COCOAnnotations')
os.makedirs(coco_dir, exist_ok=True)

xml_files = glob(os.path.join(voc_dir, "*.xml"))
img_files = glob(os.path.join(voc_dir, "*.jpg"))
# copy annotations
for f in tqdm(xml_files):
  shutil.copy(f, coco_dir)
# copy images
for f in tqdm(img_files):
  shutil.copy(f, coco_dir)

from pylabel import importer
# load dataset
dataset = importer.ImportVOC(coco_dir, name="brain tumors")

# export
coco_file = os.path.join(coco_dir, "_annotations.coco.json")
# Detectron requires starting index from 1
dataset.export.ExportToCoco(coco_file, cat_id_index=1)

for f in xml_files:
  os.remove(f.replace(voc_dir, coco_dir))