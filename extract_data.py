
from bs4 import Tag, NavigableString, BeautifulSoup
from PIL import Image, ImageDraw
from unidecode import unidecode
import numpy as np
import os, shutil
from tqdm import tqdm
import argparse, json

def crop_image(image, polygon):
    # Expected polygon (example) = "492,275 517,277 542,278 567,277 592,276 617,274 642,272 667,271 692,270 717,271 742,272 768,276 768,240 742,236 717,235 692,234 667,235 642,236 617,238 592,240 567,241 542,242 517,241 492,239"
    polygon = polygon.split(' ')
    polygon = [i.split(',') for i in polygon]
    polygon = [(int(i[0]), int(i[1])) for i in polygon]
    
    mask = Image.new("L", image.size, 0)  
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, fill=255)  
    result = Image.new("RGBA", image.size, (0, 0, 0, 0)) 
    result.paste(image, mask=mask)
    bounding_box = mask.getbbox()
    cropped_image = result.crop(bounding_box)
    cropped_image = cropped_image.convert("RGB") 
    return cropped_image


def extract_data(xml_paths, image_source_paths, images_path, text_path):
    if os.path.exists(images_path):
        shutil.rmtree(images_path)
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if os.path.exists(text_path):
        shutil.rmtree(text_path)
    if not os.path.exists(text_path):
        os.makedirs(text_path)
    xml_files = os.listdir(xml_paths)
    images_source = os.listdir(image_source_paths)
    images_source = [i for i in images_source if i[-4:] =='.jpg']

    for xml_file in tqdm(xml_files):
        document = xml_file[:-4]
        path_image_source = os.path.join(image_source_paths, document+'.jpg')
        image = Image.open(path_image_source).convert("RGBA") 
        
        with open(os.path.join(xml_paths, xml_file), 'r', encoding="utf8") as f:
            data = f.read()
        data = BeautifulSoup(data, "xml")
        lines =data.find_all('TextLine')
        for i, line in enumerate(lines):
            document_line = document + 'line_' + str(i)
            path_image_line = os.path.join(images_path, document_line + '.jpg')
            path_text_line = os.path.join(text_path, document_line + '.txt')
            
            for child in line.children: #print(child)
                if child.name == 'Baseline':
                    polygon = child['points']
                if child.name == 'TextEquiv':
                    text = unidecode(child.text).replace('\n', '')

            if '@' in text or '$' in text:
                text = text.replace('@', '')
                text = text.replace('$', '')
                text = text.lstrip().rstrip()
                cropped_image = crop_image(image, polygon)
                cropped_image.save(path_image_line) 
                text_file = open(path_text_line, "w")
                text_file.write(text)
                text_file.close()

def prepare_json(text_path, json_file):
    text_train = os.listdir(text_path)
    data_json = {}
    for file in text_train:
        path = os.path.join(text_path, file)
        f = open(path, "r")
        text = f.read()
        data_json[file[:-4]+'.jpg'] = text

    with open(json_file, 'w') as f:
        json.dump(data_json, f) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_paths", type=str) # The path to XML files.
    parser.add_argument("--image_source_paths", type=str) # The path to the images.
    parser.add_argument("--images_path", type=str) # The path where the image lines will be saved.
    parser.add_argument("--text_path", type=str) # The path where the transcriptions will be saved
    parser.add_argument("--json_file", type=str) # The path where the json file will be saved. The file maps the image lines to their transcriptions.

    args = parser.parse_args()
    extract_data(args.xml_paths, args.image_source_paths, args.images_path, args.text_path)
    prepare_json(args.text_path, args.json_file)
    