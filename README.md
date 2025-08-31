# AdapterTrOCR

## Summary

**AdapterTrOCR** extends the TrOCR model for Handwritten Text Recognition (HTR) with two lightweight adapter modules:
- **Language adapter** for historical Latin.
- **Style adapter** for diverse handwriting styles.

This modular design enables efficient transfer from a generic English-trained model to low-resource settings, such as 16th–18th century Latin manuscripts. To overcome limited training data, AdapterTrOCR also leverages **Handwritten Text Generation (HTG)**.

In addition, we introduce **DiffWord**, a diffusion-based model for HTG, which further improves the quality and diversity of synthetic training data.

## AdapterTrOCR: Training

Steps to train AdapterTrOCR:

### 1. Data Extraction

To train **AdapterTrOCR**, you need:

- A folder containing the **line images**.
- A **dictionary (JSON file)** that maps each image to its corresponding transcription.

Assuming your initial data is stored in the standard **PAGE-XML format**, run the following commands to extract data:

```
python extract_data.py \
--xml_paths="source_data/page" \
--image_source_paths="source_data" \
--images_path="extracted_data/images" \
--text_path="extracted_data/text" \
--json_file="extracted_data/json_file.json"
```


