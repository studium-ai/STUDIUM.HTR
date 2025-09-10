## Preparation
# curl --cookie zenodo-cookies.txt https://zenodo.org/records/11209325/files/VOC%20Ground%20truths%20of%20the%20trainingset%20in%20PAGE%20xml.7z?download=1 --output archive_voc_xml.7z
# curl --cookie zenodo-cookies.txt https://zenodo.org/records/11209325/files/VOC%20Images%20of%20the%20trainingset.7z?download=1 --output archive_voc_image.7z

import py7zr, os, shutil, random, yaml, requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

def split_data(path_source_images, path_source_xmls, path_train_images, path_val_images, path_train_xmls, path_val_xmls):
    os.makedirs(path_train_images, exist_ok=True)
    os.makedirs(path_val_images, exist_ok=True)
    os.makedirs(path_train_xmls, exist_ok=True)
    os.makedirs(path_val_xmls, exist_ok=True)
    images = sorted(os.listdir(path_source_images))
    xmls = sorted(os.listdir(path_source_xmls))
    division = random.sample(range(len(images)), int(len(images) * 0.05))
    for i in tqdm(range(len(images))):
        if i in division:
            shutil.copyfile(os.path.join(path_source_images, images[i]), os.path.join(path_val_images, images[i]))
            shutil.copyfile(os.path.join(path_source_xmls, xmls[i]), os.path.join(path_val_xmls, xmls[i]))
        else:
            shutil.copyfile(os.path.join(path_source_images, images[i]), os.path.join(path_train_images, images[i]))
            shutil.copyfile(os.path.join(path_source_xmls, xmls[i]), os.path.join(path_train_xmls, xmls[i]))


def extract_polygons(train_images_path, train_labels_path, train_xml_path):
    train_images = os.listdir(train_images_path)
    for i in tqdm(range(len(train_images))):
        im_path = os.path.join(train_images_path, train_images[i])
        if 'DIGI_0033_1010' in train_images[i]: continue
        im = Image.open(im_path)
        width, height = im.size
        label_path = os.path.join(train_labels_path, train_images[i][:-4] + '.txt')
        image_path = os.path.join(train_images_path, train_images[i])
        # re-save image to ensure consistency
        im.save(image_path, quality=95, subsampling=0)
        path_xml = os.path.join(train_xml_path, train_images[i][:-4] + '.xml')
        with open(path_xml, 'r', encoding="utf-8") as f:
            data = f.read()
        data = BeautifulSoup(data, "xml")
        textlines = data.find_all('TextLine')
        for textline in textlines:
            coords = textline.find('Coords')
            if coords and coords.has_attr('points'):
                polygon = coords['points'].split(' ')
                polygon = [k.split(',') for k in polygon]
                polygon = [(int(l[0]), int(l[1])) for l in polygon]
                poly = []
                for p in polygon:
                    poly.append(p[0] / width)
                    poly.append(p[1] / height)
                line = 0, *poly
                with open(label_path, "a") as f_out:
                    f_out.write(("%g " * len(line)).rstrip() % line + "\n")

def remove_unmatched_files(folder, valid_basenames):
    for f in os.listdir(folder):
        basename, ext = os.path.splitext(f)
        if basename not in valid_basenames:
            os.remove(os.path.join(folder, f))


# unarchive the data
xml_dir, image_dir = 'voc_xmls', 'voc_images'
if not os.path.exists(xml_dir):
    os.makedirs(xml_dir)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
with py7zr.SevenZipFile('archive_voc_xml.7z', mode='r') as z:
    z.extractall(xml_dir)
with py7zr.SevenZipFile('archive_voc_image.7z', mode='r') as z:
    z.extractall(image_dir)

# To train YOLO, you need to have 2 folders: train and val;
# each folder contains 3 folders: images (scanned manuscripts), xmls (xml files) and
# and labels (txt files that contain the polygons with the coordinates of each line)

# split the data into train and validation (according to the YOLO dataset format)
split_data('voc_images', 'voc_xmls', 'voc_data/train/images', 'voc_data/val/images', 'voc_data/train/xmls', 'voc_data/val/xmls')

# a = sorted([i[:-3] for i in os.listdir('voc_data/val/images')])
# b = sorted([i[:-3] for i in os.listdir('voc_data/val/xmls')])
# a == b

# generate the labels (txt files with line coordinates)
train_images_path = 'voc_data/train/images'
train_labels_path = 'voc_data/train/labels'
os.makedirs(train_labels_path, exist_ok=True)
train_xml_path = 'voc_data/train/xmls'
extract_polygons(train_images_path, train_labels_path, train_xml_path)
print(len(os.listdir(train_images_path)), len(os.listdir(train_xml_path)), len(os.listdir(train_labels_path)))

label_files = os.listdir(train_labels_path)
label_basenames = set(os.path.splitext(f)[0] for f in label_files)
remove_unmatched_files(train_images_path, label_basenames)
remove_unmatched_files(train_xml_path, label_basenames)
print(len(os.listdir(train_images_path)), len(os.listdir(train_xml_path)), len(os.listdir(train_labels_path)))

val_images_path = 'voc_data/val/images'
val_labels_path = 'voc_data/val/labels'
os.makedirs(val_labels_path, exist_ok=True)
val_xml_path = 'voc_data/val/xmls'
extract_polygons(val_images_path, val_labels_path, val_xml_path)
print(len(os.listdir(val_images_path)), len(os.listdir(val_xml_path)), len(os.listdir(val_labels_path)))

label_files = os.listdir(val_labels_path)
label_basenames = set(os.path.splitext(f)[0] for f in label_files)
remove_unmatched_files(val_images_path, label_basenames)
remove_unmatched_files(val_xml_path, label_basenames)
print(len(os.listdir(val_images_path)), len(os.listdir(val_xml_path)), len(os.listdir(val_labels_path)))

### prepare the yaml file
d = {
    "path": "voc_data",
    "train": "train",
    "val": "val",
    "nc": 1,
    "names": {0:'line'},
}
with open('yolo_voc_data.yaml', "w") as f:
    yaml.dump(d, f, sort_keys=False)

# train YOLOv11; more details here: https://github.com/ultralytics/ultralytics
model = YOLO("yolo11x-seg.pt")
results = model.train(data="yolo_voc_data.yaml", epochs=250)


