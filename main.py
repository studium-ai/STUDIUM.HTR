from PIL import Image
from torch.utils.data import DataLoader

from context import Context
from dataset import HCRDataset, MemoryDataset
from scripts import predict, train, validate
from util import debug_print, init_model_for_training, load_model, load_processor
import torch, os, json
from transformers import VisionEncoderDecoderModel
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from dataset import load_csv_labels
from torchmetrics.functional.text import char_error_rate
import numpy as np
from tqdm import tqdm
import jiwer

def calculate_wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
    deletions = len(ref_words) - len(hyp_words)
    insertions = len(hyp_words) - len(ref_words)
    total_words = len(ref_words)
    wer = (substitutions + deletions + insertions) / total_words
    return wer

class TrocrPredictor:
    def __init__(self, args):
        is_latin = 0
        if args.is_mt5 == 1 or args.is_latinbert==1 or args.is_laberta==1 or args.is_lata==1 or args.is_mt5==1 or args.is_mbart==1:
            is_latin = 1
        self.processor = load_processor(args.model_trocr, args.model_decoder, args.use_latin_tokenizer, is_latin)

        if args.is_mt5 ==1 or args.is_mbart==1 or args.use_lora==1 or args.use_lora0==1 or args.is_sft==1 or args.use_lora_style==1 \
            or args.use_lora_style_language ==1 :
            self.model = load_model(args,self.processor)
            self.model.load_state_dict(torch.load(os.path.join(args.model_save_path, 'model.pth'), weights_only=True))
        else:
            self.model = VisionEncoderDecoderModel.from_pretrained(args.model_save_path).to(device)
            # self.model = load_model(model_save_path, use_latin_tokenizer, model_decoder, 0,
            #                         is_latinbert, is_laberta, is_lata, is_mt5, is_mbart, self.processor)
        init_model_for_training(self.model, self.processor, args.use_latin_tokenizer, args.is_latinbert, args.is_laberta, \
                                args.is_lata, args.is_mt5,
                                args.is_mbart)
        self.batch_size = args.batch_size

    def predict_for_image_paths(self, args, image_paths: list[str]) -> list[tuple[str, float]]:
        images = [Image.open(path) for path in image_paths]
        return self.predict_images(args, images)

    def predict_images(self, args, images: list[Image.Image]) -> list[tuple[str, float]]:
        dataset = MemoryDataset(images, self.processor)
        dataloader = DataLoader(dataset, self.batch_size)

        predictions, confidence_scores = predict(self.processor, self.model, dataloader, args)
        return zip([p[1] for p in sorted(predictions)], [p[1] for p in sorted(confidence_scores)])
    #
    def generate_transcriptions(self, args):
        images = os.listdir(args.image_paths)
        labels, predictions, wer_scores = [], [], []
        results = {}
        print('args.output_file', args.output_file)
        for image in tqdm(images):
            path_image = os.path.join(args.image_paths, image)
            prediction = self.predict_images(args, [Image.open(path_image)])
            prediction = [i[0] for i in prediction][0]
            predictions.append(prediction)
            results[image] = [prediction]

        with open(args.output_file, "w") as final:
            json.dump(results, final)
        return 0, 0, 0

    def compute_scores(self, args):
        label_dict = load_csv_labels(args.json_file)
        images = os.listdir(args.image_paths)
        correct_count = 0
        labels, predictions, wer_scores = [], [], []
        results = {}
        for image in tqdm(images):
            path_image = os.path.join(args.image_paths, image)
            label = label_dict[image]
            if label=='': continue
            labels.append(label)
            prediction = self.predict_images(args, [Image.open(path_image)])
            prediction = [i[0] for i in prediction][0]
            predictions.append(prediction)
            print('path_image', path_image)
            print('prediction', prediction)
            print('label', label)
            if prediction == label:
                correct_count += 1
            # try:
            #     wer_score = calculate_wer(label, prediction)
            # except: continue
            # wer_scores.append(wer_score)
            results[image] = [label, prediction]

        # with open(args.output_file, "w") as final:
        #     json.dump(results, final)
        acc = correct_count / (len(predictions))
        char_score = char_error_rate(preds=predictions, target=labels)
        wer_score = jiwer.wer(labels, predictions)
        return acc, char_score, wer_score

def main_train(args):
    is_latin = 0
    if args.is_mt5 == 1 or args.is_latinbert ==1 or args.is_laberta==1 or \
            args.is_lata==1 or args.is_mt5==1 or args.is_mbart==1:
        is_latin=1
    processor = load_processor(args.model_trocr, args.model_decoder, args.use_latin_tokenizer, is_latin)

    train_dataset = HCRDataset(args.json_file, args.train_dir, args.train_dir_ref, processor)

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)#, num_workers=1)
    ### commented
    # val_dataset = HCRDataset(args.json_file, args.val_dir, args.val_dir_ref, processor)
    # val_dataloader = DataLoader(val_dataset, args.batch_size)#, num_workers=8)

    model = load_model(args, processor)
    init_model_for_training(model, processor, args.use_latin_tokenizer, args.is_latinbert, args.is_laberta, \
                            args.is_lata, args.is_mt5, args.is_mbart)

    ### commented
    # context = Context(model, processor, train_dataset, train_dataloader, val_dataset, val_dataloader)
    context = Context(model, processor, train_dataset, train_dataloader, train_dataset, train_dataloader)
    train(args, context)
    debug_print(f"Saving model to {args.model_save_path}...")
    model.save_pretrained(args.model_save_path)
    torch.save(model.state_dict(), os.path.join(args.model_save_path, 'model.pth'))

def main_validate(use_local_model: bool = True):
    processor = load_processor()
    val_dataset = HCRDataset(paths.val_dir, processor)
    val_dataloader = DataLoader(val_dataset, constants.batch_size, shuffle=True, num_workers=constants.num_workers)

    model = load_model(use_local_model)

    context = Context(model, processor, None, None, val_dataset, val_dataloader)
    validate(context, True)
