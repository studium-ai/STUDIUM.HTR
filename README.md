# Studium.HTR

## Summary

**STUDIUM.HTR** is a Handwritten Text Recognition (HTR) pipeline that can automatically transcribe handwritten text. While the current project focuses on **Latin historical manuscripts**, the pipeline can be adapted to other languages as well. The development of STUDIUM.HTR is part of the STUDIUM.AI project. More details about the STUDIUM.AI project are available [**here**](https://www.kuleuven.be/lectio/studium.ai).


**STUDIUM.HTR** phases:

1. **Training:** The training phase requires two main models: DiffLine for Handwritten Text Generation (HTG) and AdapterTrOCR for HTR. While AdapterTrOCR is the core model of the pipeline, DiffLine is used to augment the training data.

2. **Inference:** During inference, we extract line images using a YOLOv11 model fine-tuned for line detection. The transcriptions are then generated from these line images using the trained AdapterTrOCR model. We apply the **STUDIUM.HTR** pipeline to produce transcriptions for the [**Magister Dixit collection**](https://www.kuleuven.be/lectio/research/MagisterDixit).


---
 ## Instalation
 Create and activate a virtual environment
 ```
python -m venv adapter_trocr
source adapter_trocr/bin/activate   
```
Install the required libraries
```
pip install -r requirements.txt
```
---
## Data

Magister Dixit is a large collection of 575 Latin manuscripts containing student notes from the 16th to the 18th centuries. To capture the wide range of writing styles, we are building a dataset that covers 25-year intervals within this period. For the interval 1600-1675 we do not have data collected and for the interval 1524-1549 we do not have manuscripts. In total, our data covers 8 25-year intervals.

| Dates      | Manuscript             | Image lines | AdapterTrOCR | Style Adapter|
|------------|-----------------------|-----------|-----------|-----------|
| 1500-1524  | BC Ms 237             | 380         | [model](https://drive.google.com/file/d/1UM9PTXUAvlBmR_RsQM1WDojFcBRWza17/view?usp=sharing)|[model](https://drive.google.com/file/d/1HXyoRsW3otB3IKwA-CMmRvo2S8cCKp1a/view?usp=sharing)|
| 1524-1549  | No Manuscript           | -         | -|-|
| 1550-1574  | Jarrick               | 3765         |[model](https://drive.google.com/file/d/1O3XhivYdR2cAEHAfuS8V2dVmywi7484I/view?usp=sharing)|[model](https://drive.google.com/file/d/1qt4R7ywwPeioOC3l7ojHxxU3OTgyqsgm/view?usp=sharing)|
| 1575-1599  | KBR Ms II 1034; M s. 22153         | 867; 738        | [model](https://drive.google.com/file/d/1lOOxN_vvcsCUqkLH34f-BqD9SJGLz5xQ/view?usp=sharing)|[model](https://drive.google.com/file/d/14PYML1UIEOtXRzrGSNAfF4ECrNsC4ktu/view?usp=sharing)|
| 1600-1624  | KBR ms. 20897-20898   | -         |-|-|
| 1625-1649  | BC Ms. 364            | -         |-|-|
| 1650-1674  | Berne Heeswijk Ms K14 | -         | -|-|
| 1675-1699  | KBR Ms. 19376         | 362         |[model](https://drive.google.com/file/d/1le0o0cRb8e__AeC9JnTw3JR3W7Ho-YFe/view?usp=sharing) |[model](https://drive.google.com/file/d/1bxc6SwFKoiernXJVWR3y0FPh_yOv41Ro/view?usp=sharing)|
| 1700-1724  | BC Ms. 379            | 684         | [model](https://drive.google.com/file/d/1EwTeOZAAgtqUviF1WIsuSow1RJOQA8d6/view?usp=sharing)|[model](https://drive.google.com/file/d/10aFlt6IFobugIyqyA_m3YC-jH1AaJSte/view?usp=sharing)|
| 1725-1749  | MSB PM0328            | 327         | [model](https://drive.google.com/file/d/1GT8nX_YAkjABOyTkTKRe4na575Rpr904/view?usp=sharing)|[model](https://drive.google.com/file/d/1kTYDW8i1C33_3rZ4ETjElcorTHs5NDY1/view?usp=sharing)|
| 1750-1774  | BC Ms. 290            | 354         | [model](https://drive.google.com/file/d/1HMjzLrHLuonmWAy-E3IfpKe3ZfGcLflo/view?usp=sharing)|[model](https://drive.google.com/file/d/1tfQsOppSb-H6c1Lv8Br7wqTB7JDsgFxk/view?usp=sharing)|
| 1775-1799  | KBR ms. 21909         | 234         |[model](https://drive.google.com/file/d/1jvc-_9Opop6vr5kd9qNU-9knwP27OAcs/view?usp=sharing)|[model](https://drive.google.com/file/d/1KjUhotNUJkOeLlHyHV8XUvpTOSZD0Myo/view?usp=sharing)|

Additionally, we include in the training data the Rudolph Gwalther dataset.

| Dates      | Manuscript             | Image lines |
|------------|-----------------------|-----------|
| 1540-1580  | Rudolph Gwalther      | 4041         |


* The folder `data/pagexml_magister_dixit` contains the data in the standard **PAGE-XML format**. 

* The folder `data/processed_magister_dixit` contains the line images with are required to train **AdapterTrOCR**. 

* To rebuild the initial archive from the the archive chunks store in `data/pagexml_magister_dixit` or `data/processed_magister_dixit`, run:

```
python merge_archive_chunks.py \
--path_chunks="data/pagexml_magister_dixit" \
--path_merge="data/pagexml_magister_dixit.zip" 

python merge_archive_chunks.py \
--path_chunks="data/processed_magister_dixit" \
--path_merge="data/processed_magister_dixit.zip" 
```

* New HTR datasets stored in the standard **PAGE-XML format** can be converted into the format required to train **AdapterTrOCR** with the following command:

```
python extract_data.py \
--xml_paths="source_data/page" \
--image_source_paths="source_data" \
--images_path="extracted_data/images" \
--text_path="extracted_data/text" \
--json_file="extracted_data/json_file.json"
```
---
## Training

**AdapterTrOCR** extends the TrOCR model for Handwritten Text Recognition (HTR) with two adapter modules:
- **Task-Language adapter** for historical Latin.
- **Style adapter** for diverse handwriting styles.

This modular design allows efficient transfer from a generic English-trained model to low-resource settings, such as 16th–18th century Latin manuscripts. 

To train **AdapterTrOCR** and its adapter structures we rely on both ground-truth data (see Section Data) and synthetic data generated. The artificial data is produced using our model **DiffLine**, a diffusion-based approach for Handwritten Text Generation (HTG). More details are provided [here](https://github.com/studium-ai/DiffLine).

### Adapters

**AdapterTrOCR** integrates the following adapters:

* **Task-Language adaptation** requires:
    
    * one adapter for *task ability* ([available here](https://drive.google.com/file/d/1-n_wTUEGgei4v0GyH_JsO83b75C0BF59/view?usp=sharing)). Store it in the `task_ability_adapter` folder.
    * two adapters for *language ability*: one for Dutch ([available here](https://drive.google.com/file/d/1pTaJU61dlceG6yiVGIZms_RjNN9c8HZg/view?usp=sharing)) and one for Latin ([available here](https://drive.google.com/file/d/1hTIiDlyruVu85hwQ65c5el5mmJ-R3-_U/view?usp=sharing)). Store the Dutch adapter in the `language_ability_dutch` folder and Latin adapter in the `language_ability_latin` folder.

* **Style adaptation**: The style adapters are trained on the data of each 25-year interval. the style adapters are available in Section Data. Store the style adapters in the `style_adapters` folder.
**TODO**: generate style adapters for the inteval 1600-1675.  

To train a new style adapter for a specific 25-year interval, you need to combine the ground-truth data of that interval with 2000 synthetic line images generated using **DiffLine** under the writing style condition of the same interval. Store the training data in the `train_data_style_adapter_model/model_interval` folder. The JSON files of the 8 style adapters are stored in `data/style_adapter_data`.

To train a style adapter run the below code:

```
mkdir style_adapters

python cli.py \
--model_save_path='style_adapters/model_interval' \
--train_dir='train_data_style_adapter_model/model_interval' \
--train_dir_ref='train_data_style_adapter/model_interval' \
--json_file='data/style_adapter_data/model_interval.json' \
--model_trocr='microsoft/trocr-large-handwritten' \
--phase='train' \
--batch_size=8 \
--train_epochs=30 \
--train_lora_component=1
```

### AdapterTrOCR

All 8 **AdapterTrOCR** models are available in Section Data. Store them in the `adaptar_trocr_models` folder. The models are trained with their own style adapter on the entire training dataset plus 2000 synthetic images generated with **DiffLine** to capture the writing style of the manuscript for the associated 25-year interval.  

To train a new **AdapterTrOCR** model for a certain 25-year interval, you need to combine the full ground-truth data with 2000 synthetic line images generated using **DiffLine** using the writing style of given interval. Store the training data in the `train_data_adapter_trocr_model/model_interval` folder. The json file that keeps track of the full training data plus 2000 synthetic images is `data/adapter_trocr_data.json`.

To train a new **AdapterTrOCR** run the following command:

```
mkdir adaptar_trocr_models

python cli.py \
--model_save_path='adaptar_trocr_models/model_interval' \
--train_dir='train_data_adapter_trocr_model/model_interval' \
--json_file='data/adapter_trocr_data.json' \
--train_dir_ref='train_data_adapter_trocr_model/model_interval' \
--model_trocr='microsoft/trocr-large-handwritten' \
--phase='train' \
--batch_size=8 \
--train_epochs=30 \
--use_lora_style_language=1 \
--task_ability_adapter='task_ability_adapter' \
--language_ability_dutch='language_ability_dutch' \
--language_ability_latin='language_ability_latin' \
--model_style_path='style_adapters/model_interval'
```

---
## Inference: Transcription Generation for Magister Dixit

### Step 1: Download the data

*  Download the dataset (171 GB) using the command below.  
   All manuscripts will be saved separately in the specified `save_path`.

```
python globus.py \
--client_id='Your_Globus_client_ID' \
--id_source='UUID_source_collection' \
--id_target='UUID_target_collection' \
--save_path='magister_dixit' \
--globus_data_path="/DATASET_1" 
```

#### Step 2: Segment the Scanned Images into Lines

* To do the line segmentation, you need to either:

    * Download the fine-tuned YOLOv11 model for line detection (available [here](https://drive.google.com/file/d/1CH43Vu37IHgGsZs_GkfOxXjjCaVjfnLq/view?usp=sharing)), or  
    * Fine-tune a new model by following the instructions in `train_segmentation.py`.

* `output_path_pdf` specifies the path where all scanned images containing writing lines are merged into a single PDF.  
  Use this argument only if you plan to upload the data to Transkribus.  
  *Note: Transkribus currently supports uploads of documents with no more than 200 images.*
* `images_path` specifies the directory where the scanned images are stored (organized separately for each manuscript).

* `save_path` indicates the directory where the extracted image lines will be saved (also organized per manuscript). In each manuscript subfolder inside `save_path`, a `lines_polygons.json` file will be created to map the line images to their extracted coordinates.

```
python segmentation.py \
--images_path='magister_dixit' \
--save_path='magister_dixit_segment' \
--model_path='YOLOv11_model_path' \
--output_path_pdf='pdf_file' \
--output_path_polygons='lines_polygons.json'
```

#### Step 3: Generate transcriptions for the extracted lines using **AdapterTrOCR**

* `image_paths` specifies the directory where the line images are saved (organized separately for each manuscript). In each manuscript subfolder inside `image_paths`, a `line_text.json` file will be created to map the line images to their generated transcriptions.

```
python cli.py \
--model_save_path='adaptar_trocr_models/model_interval' \
--model_trocr='microsoft/trocr-large-handwritten' \
--phase='generate' \
--image_paths='magister_dixit_segment' \
--use_lora_style_language=1 \
--output_file='line_text.json' \
--task_ability_adapter='task_ability_adapter' \
--language_ability_dutch='language_ability_dutch' \
--language_ability_latin='language_ability_latin' \
--model_style_path='style_adapters/model_interval'
```

#### Step 4: Store Line Coordinates and Transcriptions as XML

* The argument `upload_transkribus` resizes the polygons of the image lines to match the A4 format. Use this argument only when uploading files to Transkribus.

* `images_path` specifies the directory where the scanned images are stored (organized separately for each manuscript).

* `segmentation_path` specifies the directory where line coordinates and transcriptions are saved (organized separately for each manuscript).

* `save_path` specifies the directory where the XML files will be saved (organized separately for each manuscript).

```
python create_xml.py \
--images_path='magister_dixit' \
--save_path='magister_dixit_xml' \
--segmentation_path='magister_dixit_segment' 
```
#### Step 5 (Optional): Upload XML Files to Transkribus

1. First, upload to Transkribus the generated PDF with the scanned images.  
2. Then, run the command below to upload the corresponding XML files.

* To find the required `collection_id` and `document_id` arguments, check the URL of the created collection in Transkribus:  
  `https://app.transkribus.org/collection/collection_id/doc/document_id`

```
python upload_transkribus.py \
--xml_path='magister_dixit_xml/document_name'
--username='Transkribus_username' \
--password='Transkribus_password' \
--collection_id='collection_id' \
--document_id='document_id'
```





