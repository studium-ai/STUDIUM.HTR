# AdapterTrOCR

## Summary

**AdapterTrOCR** extends the TrOCR model for Handwritten Text Recognition (HTR) with two lightweight adapter modules:
- **Language adapter** for historical Latin.
- **Style adapter** for diverse handwriting styles.

This modular design enables efficient transfer from a generic English-trained model to low-resource settings, such as 16th–18th century Latin manuscripts. The current repository introduce **AdapterTrOCR** and its training procedure for **Latin Historical HTR**. The ultimate goal of this project is to generate transcriptions for the [**Magister Dixit collection**](https://www.kuleuven.be/lectio/research/MagisterDixit).

While **AdapterTrOCR** is designed to work for Latin, it can be trained on different HTR datasets, and its modular structure allows the removal or addition of adapter components based on the specifics of the HTR task.

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

| Dates      | Manuscript             | Image lines |
|------------|-----------------------|-----------|
| 1500-1524  | BC Ms 237             | 380         | 
| 1524-1549  | No Manuscript           | -         | 
| 1550-1574  | Jarrick               | 3765         |
| 1575-1599  | KBR Ms II 1034        | 867         | 
| 1575-1599  | Ms. 22153             | 738         | 
| 1600-1624  | KBR ms. 20897-20898   | -         |
| 1625-1649  | BC Ms. 364            | -         |
| 1650-1674  | Berne Heeswijk Ms K14 | -         | 
| 1675-1699  | KBR Ms. 19376         | 362         | 
| 1700-1724  | BC Ms. 379            | 684         | 
| 1725-1749  | MSB PM0328            | 327         | 
| 1750-1774  | BC Ms. 290            | 354         | 
| 1775-1799  | KBR ms. 21909         | 234         |

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
## Training procedure

To train **AdapterTrOCR** and its adapter structures we rely on both ground-truth data (see Section Data) and synthetic data generated. The artificial data is produced using our model **DiffWord**, a diffusion-based approach for Handwritten Text Generation (HTG). More details are provided [here](https://github.com/studium-ai/DiffWord).

### Adapters

**AdapterTrOCR** integrates the following adapters:

* **Historical Latin adaptation** requires:
    
    * one adapter for *task ability* (available here). Store it in the `task_ability_adapter` folder.
    * two adapters for *language ability*: one for Dutch (available here) and one for Latin (available here). Store the Dutch adapter in the `language_ability_dutch` folder and Latin adapter in the `language_ability_latin` folder.

* **Style adaptation**: The style adapters are trained on the data of each 25-year interval. All 8 style adapters are available here. Store the style adapters in the `style_adapters` folder.
**TODO**: generate style adapters for the inteval 1600-1675.  

To train a new style adapter for a specific 25-year interval, you need to combine the ground-truth data of that interval with 2000 synthetic line images generated using **DiffWord** under the writing style condition of the same interval. Store the training data in the `train_data_style_adapter_model/model_interval` folder. The JSON files of the 8 style adapters are stored in `data/style_adapter_data`.

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

All 11 **AdapterTrOCR** models are available here. Store them in the `adaptar_trocr_models` folder.  

* 8 models are trained with their own style adapter on the entire training dataset plus 2000 synthetic images generated with **DiffWord** to capture the writing style of the manuscript for the associated 25-year interval.  
* 3 models (covering the period between 1600–1675) are trained only on the training dataset, without synthetic images. These 3 models use the style adapter associated with the period 1575–1599.  

To train a new **AdapterTrOCR** model for a certain 25-year interval, you need to combine the full ground-truth data with 2000 synthetic line images generated using **DiffWord** using the writing style of given interval. Store the training data in the `train_data_adapter_trocr_model/model_interval` folder. The json file that keeps track of the full training data plus 2000 synthetic images is `data/adapter_trocr_data.json`.

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

## AdapterTrOCR: Inference

Store the line images in the `image_path` folder and run the command below to generate transcriptions. The transcriptions will be saved as a dictionary in the `output/results.json` file.

```
python cli.py \
--model_save_path='adaptar_trocr_models/model_interval' \
--model_trocr='microsoft/trocr-large-handwritten' \
--phase='generate' \
--image_path='image_lines' \
--output_file='output/results.json' \
--use_lora_style_language=1 \
--task_ability_adapter='task_ability_adapter' \
--language_ability_dutch='language_ability_dutch' \
--language_ability_latin='language_ability_latin' \
--model_style_path='style_adapters/model_interval'
```
---
## Magister Dixit: Transcription Generation

#### Step 1: Download the data

*  Download the dataset (171 GB) using the command below.  
   All manuscripts will be saved separately in the specified `save_path`.
* The file `dataset_xls` contains metadata about the manuscripts.  
   It is currently available via Globus.
```
python globus.py \
--client_id='Your_Globus_client_ID' \
--id_source='UUID_source_collection' \
--id_target='UUID_target_collection' \
--dataset_xls="dataset_1.xlsx" \
--save_path='magister_dixit' \
--globus_data_path="/DATASET_1" 
```

#### Step 2: Segment the Scanned Images into Lines

* To do this, you need to either:

    * Download the fine-tuned YOLOv11 model for line detection (available [here]), or  
    * Fine-tune a new model by following the instructions in `train_segmentation.py`.

* `output_path_pdf` specifies the path where all scanned images containing writing lines are merged into a single PDF.  
  Use this argument only if you plan to upload the data to Transkribus.  
  *Note: Transkribus currently supports uploads of documents with no more than 200 images.*
```
python segment.py \
--images_path='magister_dixit' \
--save_path='magister_dixit_segment' \
--model_path='YOLOv11_model_path' \
--output_path_pdf='pdf_file'
```

3. Generate transcriptions for the extracted lines

#### Step 4: Store Line Coordinates and Transcriptions as XML

* The coordinates of the line images (polygons) and their corresponding transcriptions are stored in XML format.  

* If you use the argument `upload_transkribus`, the XML files will be adjusted for upload to Transkribus. In this case, the line coordinates are resized to match the A4 format.

```
python create_xml.py \
--images_path='magister_dixit' \
--save_path='magister_dixit_xml' \
--segmentation_path='magister_dixit_segment' \
--dataset_name="dataset_1.xlsx" \
--upload_transkirbus=1
```
#### Step 5 (Optional): Upload XML Files to Transkribus

1. First, upload the generated PDF with the scanned images.  
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


