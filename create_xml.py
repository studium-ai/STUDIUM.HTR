from tqdm import tqdm
import json, os, shutil, argparse
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd

a4_size = (2479, 3508)

def read_json(path):
    f = open(path)
    return json.load(f)

def resize_to_a4(polygon, orig_size, a4_size=a4_size):
    orig_w, orig_h = orig_size
    a4_w, a4_h = a4_size
    scale_x = a4_w / orig_w
    scale_y = a4_h / orig_h
    return [(x * scale_x, y * scale_y) for x, y in polygon]


def generate_page_xml(image_filename, image_width, image_height, lines_data, output_filename="output.xml"):
    ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    ET.register_namespace("", ns)
    root = ET.Element(f"{{{ns}}}PcGts")
    metadata = ET.SubElement(root, f"{{{ns}}}Metadata")
    ET.SubElement(metadata, f"{{{ns}}}Creator").text = "Transkribus"
    ET.SubElement(metadata, f"{{{ns}}}Created").text = datetime.now().isoformat()
    page = ET.SubElement(root, f"{{{ns}}}Page", {
        "imageFilename": image_filename,
        "imageWidth": str(image_width),
        "imageHeight": str(image_height)
    })
    text_region = ET.SubElement(page, f"{{{ns}}}TextRegion", {"id": "r1"})
    for line_id, coords, transcription in lines_data:
        text_line = ET.SubElement(text_region, f"{{{ns}}}TextLine", {"id": line_id})
        ET.SubElement(text_line, f"{{{ns}}}Baseline", {"points": coords})
        text_equiv = ET.SubElement(text_line, f"{{{ns}}}TextEquiv")
        unicode_text = ET.SubElement(text_equiv, f"{{{ns}}}Unicode")
        unicode_text.text = transcription
    tree = ET.ElementTree(root)
    tree.write(output_filename, encoding="UTF-8", xml_declaration=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str, default='magister_dixit', help='Folder path where the scanned manuscript images are stored.')
    parser.add_argument("--save_path", type=str, default='magister_dixit_xml', help='Folder path where the xmls will be stored')
    parser.add_argument("--segment_path", type=str, default='magister_dixit_segment', help='folder where the output of the segmentation is stored')
    parser.add_argument("--id_dict", type=str, default="data/id_dict.json", help='json that stores the manuscript IDs (globus)')
    parser.add_argument("--output_path_polygons", type=str, default='lines_polygons.json', help='json file that stores the polygons of the lines')
    parser.add_argument("--output_path_texts", type=str, default='lines_text.json', help='json file that stores the transcriptions of the lines')
    parser.add_argument("--output_path_image_names", type=str, default='image_names.json', help='json file with all initial images that were cropped')
    parser.add_argument("--upload_transkirbus", type=int, default=0, help='if yes, polygons are resized to match the A4 format.')

    args = parser.parse_args()

    f = open(args.id_dict, "r", encoding="utf-8")
    ie_to_rep = json.load(f)

    for i, ie in enumerate(tqdm(ie_to_rep.keys(), desc="Processing IE PIDs")):
        for rep in ie_to_rep[ie]:
            print('ie + "_"+ rep', ie + "_"+ rep)
            candidate_output = os.path.join(args.segment_path, ie + "_"+ rep, args.output_path_texts)
            # if not os.path.exists(candidate_output):
            #     print(f"Skipping {candidate_output} (does not exists).")
            #     continue
            # else:
            #     texts = read_json(candidate_output)
            texts = read_json(os.path.join(args.segment_path,ie + "_"+ rep, args.output_path_texts))

            output_path = os.path.join(args.save_path, f"{ie}_{rep}")
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)

            image_names = read_json(os.path.join(args.segment_path, ie + "_"+ rep, args.output_path_image_names))
            polygons = read_json(os.path.join(args.segment_path, ie + "_"+ rep, args.output_path_polygons))

            all_lines_data = []
            all_lines_data = []
            for idx, image_name in tqdm(enumerate(image_names)):
                # if idx>0:break
                jpg_path = os.path.join(args.images_path, ie+'_'+rep, image_name)
                scanned_page = Image.open(jpg_path)
                orig_w, orig_h = scanned_page.size
                image_name_split = image_name[:-4]
                lines_data = []
                for line_name in texts.keys():
                    line_name_split = line_name.split('_line')[0]
                    if image_name_split == line_name_split:
                        polygon = polygons[line_name]
                        if args.upload_transkirbus==1:
                            polygon = resize_to_a4(polygon, (orig_w, orig_h))
                        polygon = [str(int(i[0]))+','+str(int(i[1])) for i in polygon]
                        polygon = ' '.join(polygon)
                        temp = (line_name, polygon, texts[line_name][0])
                        lines_data.append(temp)
                if len(lines_data)> 0:
                    XML_FILE_PATH = os.path.join(output_path, image_name_split+'.xml')
                    if args.upload_transkirbus == 1:
                        generate_page_xml(image_name, a4_size[0], a4_size[1], lines_data, XML_FILE_PATH)
                    else:
                        generate_page_xml(image_name, orig_w, orig_h, lines_data, XML_FILE_PATH)

