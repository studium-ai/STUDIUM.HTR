import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler, AutoTokenizer

from context import Context
from util import debug_print
import torch
from tqdm import tqdm
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import torch.nn as nn
from torchmetrics.functional.text import char_error_rate
import Levenshtein


def predict(
    processor, model, dataloader, args) -> tuple[list[tuple[int, str]], list[float]]:
    output: list[tuple[int, str]] = []
    confidence_scores: list[tuple[int, float]] = []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            inputs: torch.Tensor = batch["input"].to(device)
            bad_words_ids = None
            begin_suppress_tokens = None
            if args.use_latin_tokenizer==1 or args.is_laberta==1 or args.is_latinbert==1 or args.is_mbart==1:
                bad_words_ids = [[model.config.decoder.decoder_start_token_id],[model.config.pad_token_id],
                [model.config.eos_token_id]]
                begin_suppress_tokens=[model.config.decoder.decoder_start_token_id, model.config.pad_token_id, model.config.eos_token_id]
            if args.is_mt5==1 or args.use_lora ==1 or args.use_lora0==1 or args.is_sft == 1 or \
                    args.train_lora_component==1 or args.use_lora_style==1 or args.use_lora_style_language==1:
                bad_words_ids = [[model.config.decoder.decoder_start_token_id],[model.config.pad_token_id]]
                begin_suppress_tokens=[model.config.decoder.decoder_start_token_id, model.config.pad_token_id]
            generated_ids = model.generate(inputs, return_dict_in_generate=True, output_scores = True,
                                           # num_beams=3, do_sample=True, early_stopping=False,
                                           bad_words_ids = bad_words_ids,
                                           begin_suppress_tokens = begin_suppress_tokens,
                                           max_new_tokens = 30
                                                                      )
            generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)
            ids = [t.item() for t in batch["idx"]]
            output.extend(zip(ids, generated_text))

            # Compute confidence scores
            if args.is_mbart==0:
                batch_confidence_scores = get_confidence_scores(generated_ids)
                confidence_scores.extend(zip(ids, batch_confidence_scores))

    return output, confidence_scores

def get_confidence_scores(generated_ids) -> list[float]:
    # Get raw logits, with shape (examples,tokens,token_vals)
    logits = generated_ids.scores
    logits = torch.stack(list(logits),dim=1)

    # Transform logits to softmax and keep only the highest (chosen) p for each token
    logit_probs = F.softmax(logits, dim=2)
    char_probs = logit_probs.max(dim=2)[0]

    # Only tokens of val>2 should influence the confidence. Thus, set probabilities to 1 for tokens 0-2
    mask = generated_ids.sequences[:,:-1] > 2
    char_probs[mask] = 1

    # Confidence of each example is cumulative product of token probs
    batch_confidence_scores = char_probs.cumprod(dim=1)[:, -1]
    return [v.item() for v in batch_confidence_scores]

# will return the accuracy but not print predictions
def validate(args, context: Context, print_wrong: bool = False) -> float:
    predictions, _ = predict(context.processor, context.model,
                             context.val_dataloader, args)
    assert len(predictions) > 0

    correct_count = 0
    wrong_count = 0
    for id, prediction in predictions:
        label = context.val_dataset.get_label(id)
        path = context.val_dataset.get_path(id)
        print('path', path)
        print('label', label)
        print('prediction', prediction)
        if prediction == label:
            correct_count += 1
        else:
            wrong_count += 1
            if print_wrong:
                print(f"Predicted: \t{prediction}\nLabel: \t\t{label}\nPath: \t\t{path}")

    if print_wrong:
        print(f"\nCorrect: {correct_count}\nWrong: {wrong_count}")
    return correct_count / (len(predictions))


def train(args, context: Context, learning_rate=5e-6):
    model = context.model
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = args.train_epochs * len(context.train_dataloader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    weights = None
    if args.filter_tokenizer==1:
        model_trocr = 'microsoft/trocr-base-handwritten'
        english_tokenizer = TrOCRProcessor.from_pretrained(model_trocr).tokenizer
        latin_tokenizer = AutoTokenizer.from_pretrained('bowphs/LaBerta')
        diff_tokens = list(set(english_tokenizer.vocab).intersection(latin_tokenizer.vocab))
        weights = [0.0] * english_tokenizer.vocab_size
        for word in tqdm(english_tokenizer.vocab):
            if word in diff_tokens:
                weights[english_tokenizer.vocab[word]] = 1.0
        weights = torch.tensor(weights).to(device)

    model.to(device)
    model.train()
    ctc_loss = nn.CTCLoss()

    if args.is_curriculum==0:
        reduction = "mean"
    if args.is_curriculum==1:
        reduction = "none"

    train_dataloader = context.train_dataloader
    train_dataset = context.train_dataset
    # new lines
    if args.is_curriculum==1:
        train_dataset = context.train_dataset
        train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=False)

    for epoch in range(args.train_epochs):
        print('epoch ', epoch)
        j = 0
        sample_losses = torch.zeros(len(train_dataset))
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            indices: torch.Tensor = batch["idx"].to(device)

            inputs: torch.Tensor = batch["input"].to(device)
            labels: torch.Tensor = batch["label"].to(device)

            outputs = model(pixel_values=inputs, labels=labels, reduction = reduction,
                            filter_tokenizer=args.filter_tokenizer, weights=weights)
            loss = outputs.loss

            # new lines
            if args.is_curriculum==1:
                loss_matrix = loss.view(len(inputs), int(len(loss)/len(inputs)))
                instance_loss = torch.mean(loss_matrix, dim=-1)
                loss = torch.mean(instance_loss)
                for k, idx in enumerate(indices):
                    sample_losses[idx] = instance_loss[k].item()

            if args.is_ref == 1:
                inputs_ref: torch.Tensor = batch["input_ref"].to(device)

                outputs_ref = model(pixel_values=inputs_ref, labels=labels, reduction=reduction,
                                filter_tokenizer=args.filter_tokenizer, weights=weights)
                loss_ref = outputs_ref.loss

                loss += args.weight_ref*loss_ref

            if args.is_ctc==1:
                logits = torch.permute(outputs.logits, (1, 0, 2))
                T, N, S_min = labels.shape[1], labels.shape[0], 10
                input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
                target_lengths = torch.randint(low=S_min, high=T, size=(N,), dtype=torch.long)
                loss_ctc = args.weight_ctc * ctc_loss(logits, labels, input_lengths, target_lengths)
                loss += loss_ctc

            if args.is_cer==1:
                logits = torch.argmax(outputs.logits, dim=-1)
                if args.is_curriculum==0:
                    loss_cer = args.weight_cer * char_error_rate(preds=logits, target=labels)
                    loss += loss_cer
                if args.is_curriculum==1:
                    losses_cer = []
                    for k, idx in enumerate(indices):
                        error = Levenshtein.distance(logits[k], labels[k])
                        instance_loss_cer = torch.tensor(args.weight_cer * error, dtype=torch.float32,)
                        losses_cer.append(instance_loss_cer)
                        sample_losses[idx] += instance_loss_cer
                    loss_cer = torch.mean(torch.stack(losses_cer))
                    loss += torch.tensor(loss_cer, device=loss.device)
                
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress_bar.set_postfix(loss=loss.item())
            # debug_print(f"11Epoch {1 + epoch}, Batch {1 + j}: {loss} loss")
            del loss, outputs

        ### commented
        # if len(context.val_dataloader) > 0:
        #     accuracy = validate(args, context)
        #     print(f"\n---- Epoch {1 + epoch} ----\nAccuracy: {accuracy}\n\n")

        if args.is_curriculum==1:
            train_dataset.update_losses(sample_losses)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)



