
# inference
from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2, math, os, shutil, random
import pandas as pd
import argparse, json
from tqdm import tqdm

def left_most_x(polygon): return min(x for x, _ in polygon)

def top_most_y(polygon): return min(y for _, y in polygon)

def left_most_x_box(bbox): return bbox[0]  # x1 is the first element (x1, y1, x2, y2)

def top_most_y_box(bbox): return bbox[1]

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str, default='magister_dixit', help="Folder path where the scanned manuscript images are stored.")
    parser.add_argument("--save_path", type=str, default='magister_dixit_segment', help="Folder path where all line images of the manuscripts will be saved.")
    parser.add_argument("--output_path_lines", type=str, default='lines', help="folder name that stores image lines of a manuscript")
    parser.add_argument("--output_path_images_with_lines", type=str, default=None, help="Folder path where the initial images with drawn lines will be saved.(for visualization)")
    parser.add_argument("--output_path_pdf", type=str, default=None, help="PDF with all scanned images of a manuscript  (for transkribus)")
    parser.add_argument("--output_path_polygons", type=str, default='lines_polygons.json', help=" json file that stores the polygons of the lines")
    parser.add_argument("--output_path_image_names", type=str, default='image_names.json', help=" json file that stores the names of all initial images that were cropped")
    parser.add_argument("--model_path", type=str, default='voc/yolo_voc_data_aug/runs/segment/train4/weights/best.pt', help=" path to the trained yolo model")
    parser.add_argument("--threshold", type=float, default=0.5, help=" Keeps only the lines for which YOLO assigns a confidence score higher than the threshold")
    parser.add_argument("--id_dict", type=str, default="data/ie_dict.json", help=" 'json that stores the manuscript IDs (globus)'")
    parser.add_argument("--start", type=int, default=0, help=" starting point for splitting the dictionary with manuscripts")
    parser.add_argument("--end", type=int, default=600, help=" ending point for splitting the dictionary with manuscripts")
    args = parser.parse_args()

    f = open(args.id_dict, "r", encoding="utf-8")
    ie_to_rep = json.load(f)

    all_keys = list(ie_to_rep.keys())
    valid_keys = all_keys[args.start:args.end]

    print('args.model_path', args.model_path)
    model = YOLO(args.model_path)

    for i, ie in enumerate(tqdm(valid_keys, desc="Processing IE PIDs")):
        for rep in ie_to_rep[ie]:
            output_path = os.path.join(args.save_path, f"{ie}_{rep}")
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)

            output_path_folder_lines = os.path.join(output_path, args.output_path_lines)
            if os.path.exists(output_path_folder_lines):
                shutil.rmtree(output_path_folder_lines)
            os.makedirs(output_path_folder_lines)

            if args.output_path_images_with_lines is not None:
                output_path_images_with_lines = args.output_path_images_with_lines
                if os.path.exists(output_path_images_with_lines):
                    shutil.rmtree(output_path_images_with_lines)
                os.makedirs(output_path_images_with_lines)

            images_path = os.path.join(args.images_path, ie + "_" + rep)
            images_names = os.listdir(images_path)

            index1, index2 = 0, 0
            initial_images = []
            final_images = []
            polygons_dict = {}
            for image_name in tqdm(images_names):
                if index2 == 1: break
                try:
                    image_path = os.path.join(images_path, image_name)
                    results = model.predict(image_path, verbose=False)
                    image = Image.open(image_path).convert("RGBA")

                    masks, x0s, y0s, confs = [], [], [], []
                    for i in range(len(results[0].masks)):
                        mask = results[0].masks[i].xy[0]
                        conf = results[0].boxes[i].conf.cpu().item()
                        masks.append(mask)
                        x0s.append(mask[0][0].item())
                        y0s.append(mask[0][1].item())
                        confs.append(conf)

                    df = pd.DataFrame.from_dict({'m':masks, 'x0':x0s, 'y0':y0s, 'conf':confs})
                    df = df[df.conf>args.threshold]
                    df = df.sort_values(by=["m"], key=lambda x: x.apply(left_most_x))  # Sort by X
                    df = df.sort_values(by=["m"], key=lambda x: x.apply(top_most_y), kind="stable")  # Sort by Y (stable sort)
                    masks = df.m.tolist()
                    confs = df.conf.tolist()

                    for i, polygon in enumerate(masks):
                        index1 += 1
                        # if index1>5000: index2=1
                        mask = Image.new("L", image.size, 0)
                        draw = ImageDraw.Draw(mask)
                        draw.polygon(polygon, fill=255)
                        result = Image.new("RGBA", image.size, (0, 0, 0, 0))
                        result.paste(image, mask=mask)
                        bounding_box = mask.getbbox()
                        cropped_image = result.crop(bounding_box)
                        conf = str(round(df.conf.iloc[i].item()*100, 2))
                        output_image_path = os.path.join(output_path_folder_lines, image_name[:-4]+'_'+'line'+str(i)+'_'+conf+'.jpg')
                        cropped_image.convert('RGB').save(output_image_path)
                        polygons_dict[image_name[:-4]+'_'+'line'+str(i)+'_'+conf+'.jpg'] = polygon.tolist()

                    if args.output_path_images_with_lines is not None:
                        image = Image.open(image_path).convert("RGBA")
                        draw = ImageDraw.Draw(image)
                        for i, polygon in enumerate(masks):
                            outline_color = random_color()
                            draw.polygon(polygon, outline=outline_color, width=3)
                            draw.text((polygon[0][0]+5, polygon[0][1]+5),'line'+str(i) +'_'+str(round(confs[i],2)) , fill='black')
                        image.convert('RGB').save(os.path.join(output_path_images_with_lines, image_name))

                    if args.output_path_pdf is not None:
                        image_rgb = Image.open(image_path).convert("RGB")
                        initial_images.append(image_rgb)

                    final_images.append(image_name)
                    ## uncomment if you want to merge all scanned images into a pdf; transkribus does not accept a pdf with more than 200 images
                    # if len(initial_images) > 199: index2=1
                except: continue
            with open(os.path.join(output_path, args.output_path_polygons), "w") as final:
                json.dump(polygons_dict, final)
            with open(os.path.join(output_path, args.output_path_image_names), "w") as final:
                json.dump(final_images, final)
            if args.output_path_pdf is not None:
                initial_images[0].save(os.path.join(output_path, args.output_path_pdf), save_all=True, append_images=initial_images[1:])


