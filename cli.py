from main import TrocrPredictor, main_train, main_validate
import argparse, shutil, os, json
from tqdm import tqdm

def run(args):
    if args.phase=='train':
        if os.path.exists(args.model_save_path) and os.path.isdir(args.model_save_path):
            shutil.rmtree(args.model_save_path)
        os.mkdir(args.model_save_path)
        main_train(args)
    if args.phase=='validation':
        main_validate()
    if args.phase=='test':
        predictions = TrocrPredictor(args).predict_for_image_paths(args, [args.image_paths])
        for path, (prediction, confidence) in zip(args.image_paths, predictions):
            print(f"Path:\t\t{path}\nPrediction:\t{prediction}\nConfidence:\t{confidence}\n")

    if args.phase=='evaluate':
        print('args.model_trocr', args.model_trocr)
        acc, char_error, wer_scores = TrocrPredictor(args).compute_scores(args)
        print(f"Accuracy:\t\t{acc}\nCER:\t{char_error}\nWER:\t{wer_scores}\n")

    if args.phase=='generate':
        f = open(args.id_dict, "r", encoding="utf-8")
        ie_to_rep = json.load(f)
        all_keys = list(ie_to_rep.keys())
        valid_keys = all_keys[args.start:args.end]
        for i, ie in enumerate(tqdm(valid_keys, desc="Processing IEs")):
            for rep in ie_to_rep[ie]:

                # candidate_output=os.path.join(args.output_file, ie + "_"+ rep,"lines_text.json")
                candidate_output=os.path.join(args.image_paths, ie + "_"+ rep,args.output_file)
                if os.path.exists(candidate_output):
                    print(f"Skipping {candidate_output} (already exists).")
                    continue
                else:
                    args.output_file = candidate_output
                args.model_save_path=os.path.join(args.model_save_path,
                                                  ie_to_rep[ie][rep][1])
                args.model_style_path=os.path.join(args.model_style_path,
                                                   ie_to_rep[ie][rep][1])
                args.image_paths=os.path.join(args.image_paths, ie + "_"+ rep, "lines")
                acc, char_error, wer_scores = TrocrPredictor(args).generate_transcriptions(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int , default=0, help='start for splitting the dictionary with manuscripts')
    parser.add_argument("--end", type=int, default=600, help='end for splitting the dictionary with manuscripts')
    parser.add_argument("--id_dict", type=str, default="data/id_dict.json", help='json that stores the manuscript IDs (globus)')
    parser.add_argument("--model_style_path", type=str, default=None, help='path to the style adapter')
    parser.add_argument("--language_ability_dutch", type=str, default = None, help='path to the Dutch language adaptor')
    parser.add_argument("--language_ability_latin", type=str, default = None, help='path to the Latin language adapter')
    parser.add_argument("--task_ability_adapter", type=str, default = None, help='path to the task adapter')
    parser.add_argument("--lora_language_weight", type=float, default=0.0, help='weight of the LORA adaptation (language)')
    parser.add_argument("--lora_language_task_weight", type=float, default=0.0, help='weight of the LORA adaptation (task+language)')

    parser.add_argument("--model_save_path", type=str, help='path where the model will saved')
    parser.add_argument("--batch_size", type=int, default=4, help='batch size')
    parser.add_argument("--train_epochs", type=int, default=36, help='training epochs')

    parser.add_argument("--model_trocr", type=str, default='microsoft/trocr-large-handwritten', help='TrOCR model for initialization')
    parser.add_argument("--train_dir", type=str, default=None, help='path to the training line images')
    parser.add_argument("--json_file", type=str, default=None, help='dictionary that maps image lines to their transcriptions')
    parser.add_argument("--val_dir", type=str, default=None, help='path to the validation line images')
    parser.add_argument("--train_dir_ref", type=str, default=None, help='path to the training line images used as reference')
    parser.add_argument("--val_dir_ref", type=str, default=None, help='path to the validation line images used as reference')
    parser.add_argument("--num_training_steps", type=int, default=None, help='training steps')
    parser.add_argument("--image_paths", type=str, default=None, help='path where all line images of the manuscripts are saved.')
    parser.add_argument("--output_file", type=str, default='lines_text.json"', help='dictionary that maps the line images with the generated transcriptions')
    parser.add_argument("--train_lora_component", type=int, default=0, help='trains task adapter/style adapters')
    parser.add_argument("--use_lora_style", type=int, default=0, help='trains adapter_trocr with style adapter')
    parser.add_argument("--use_lora_style_language", type=int, default=0, help='trains adapter_trocr with style adapters/task/language adapters')

    ### Sparse Fine-tuning (SFT)
    parser.add_argument("--is_sft", type=int, default=0, help='SFT: train adapter_trocr with SFT')
    parser.add_argument("--model_language_path", type=str, default=None, help='SFT: path to the language adapter')
    parser.add_argument("--running_layers_language_path", type=str, default=None, help="SFT: name of the layers where the language adapters learn the most during the training")
    parser.add_argument("--model_task_path", type=str, default=None, help='SFT: path to the task adapter')
    parser.add_argument("--running_layers_task_path", type=str, default=None, help='SFT: name of the layers where the task adapter learns the most during the training')
    parser.add_argument("--is_sft_train", type=int, default=0, help='SFT: trains task adapter')
    parser.add_argument("--is_langauge_adapter", type=int, default=0)
    parser.add_argument("--model_language_phase2", type=str, default=None)
    parser.add_argument("--model_task_phase1", type=str, default=None)
    parser.add_argument("--running_layers", type=str, default=None)

    ### Curriculum Learning
    parser.add_argument("--is_curriculum", type=int, default=0, help='trains adapter_trocr with curriculum learning')

    ### Other arguments
    parser.add_argument("--model_decoder", type=str, default=None, help='decoder model')
    parser.add_argument("--freeze", type=int, default=0, help='activates layer freezing')
    parser.add_argument("--phase", type=str)
    parser.add_argument("--use_latin_tokenizer", type=int, default=0, help='train with a latin tokenizer')
    parser.add_argument("--is_latinbert", type=int, default=0, help='train with Latin BERT as a decoder')
    parser.add_argument("--is_laberta", type=int, default=0, help='train with LaBERTa as a decoder')
    parser.add_argument("--is_lata", type=int, default=0, help='train with LaTA as a decoder')
    parser.add_argument("--is_mt5", type=int, default=0, help='train with the mT5 deccoder')
    parser.add_argument("--is_mbart", type=int, default=0, help='train with the mBART decoder')
    parser.add_argument("--is_ref", type=int, default=0, help='train using an extra reference loss that compares training data with reference training data')
    parser.add_argument("--weight_ref", type=float, default=0.3, help='weight of the reference loss')
    parser.add_argument("--filter_tokenizer", type=int, default=0)
    parser.add_argument("--is_ctc", type=int, default=0, help='train with a CTC loss function')
    parser.add_argument("--weight_ctc", type=float, default=0, help='weight of the CTC loss function')
    parser.add_argument("--is_cer", type=int, default=0, help='train with a CER loss function')
    parser.add_argument("--weight_cer", type=float, default=0, help='weight of the CER loss function')
    parser.add_argument("--use_lora", type=int, default=0, help='train adapter_trocr with task and Dutch and Latin adapters')
    parser.add_argument("--use_lora0", type=int, default=0, help='train adapter_trocr with task adapter')
    args = parser.parse_args()
    run(args)