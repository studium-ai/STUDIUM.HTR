from transformers import TrOCRProcessor, VisionEncoderDecoderModel # fff
from transformers import AutoTokenizer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer, BertGenerationDecoder
from transformers import RobertaForCausalLM, AutoModelForSeq2SeqLM, BartForConditionalGeneration, BartModel, AutoModelForCausalLM
from transformers import T5ForConditionalGeneration, MT5ForConditionalGeneration
# from .configs import paths
# from .configs import constants
import torch.nn as nn
import torch, os, json
import numpy as np

from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from peft import PeftModel,LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict,prepare_model_for_kbit_training
from tqdm import tqdm


class BartTrOCRForCausalLM(nn.Module):
    _tied_weights_keys = ["output_projection.weight"]

    def __init__(self, decoder, lm_head, config):
        super(BartTrOCRForCausalLM, self).__init__()
        self.decoder = decoder
        self.output_projection = lm_head
        self.config = config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.output_projection(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

# T5ForConditionalGeneration
class T5TrOCRForCausalLM(nn.Module):
    _tied_weights_keys = ["output_projection.weight"]

    def __init__(self, decoder, lm_head, config):
        super(T5TrOCRForCausalLM, self).__init__()
        self.decoder = decoder
        self.output_projection = lm_head
        self.config = config
        self.is_decoder = True
        config.is_decoder = True

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            decoder_attention_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.output_projection

    def set_output_embeddings(self, new_embeddings):
        self.output_projection = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.output_projection(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

def load_processor(model_trocr, model_decoder, use_latin_tokenizer, is_latin) -> TrOCRProcessor:

    processor = TrOCRProcessor.from_pretrained(model_trocr)
    if use_latin_tokenizer == 1:
        processor.tokenizer = AutoTokenizer.from_pretrained('LuisAVasquez/simple-latin-bert-uncased')
    if is_latin == 1:
        processor.tokenizer = AutoTokenizer.from_pretrained(model_decoder)
    return processor

def get_decoder(model_trocr, tokenizer):
    base_model = VisionEncoderDecoderModel.from_pretrained(model_trocr)
    base_model = base_model.decoder
    base_model.config.decoder_start_token_id = tokenizer.cls_token_id
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.vocab_size = tokenizer.vocab_size
    return base_model

def get_peft_decoder(model_trocr):
    model = VisionEncoderDecoderModel.from_pretrained(model_trocr)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "k_proj",
            "q_proj",
            "v_proj",
            "out_proj",
            "fc1",
            "fc2",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    decoder = get_peft_model(model.decoder, peft_config)
    model.decoder = decoder
    return model

def get_peft_sft_decoder(model_trocr):
    model = VisionEncoderDecoderModel.from_pretrained(model_trocr)
    path = '/home/pricie/trusca/trocr2/trocr/src/data_adapter_synthetic/sft_htr_large.json'
    f = open(path)
    top_param_names = json.load(f)
    top_module_names = list({'.'.join(name.split('.')[:-1]) for name in top_param_names})
    top_module_names = [i[8:] for i in top_module_names if i.startswith('decoder.')==True]

    supported_types = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding, torch.nn.Conv3d)
    target_modules = []
    for name, module in model.decoder.named_modules():
        if isinstance(module, supported_types):
            if name in top_module_names:
                target_modules.append(name)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    decoder = get_peft_model(model.decoder, peft_config)
    model.decoder = decoder
    return model

def prepare_model(base_model1, base_model2, output_file):
    k = int(len(list(base_model1.named_parameters()))/2)
    deltas = []
    for name1, param1 in list(base_model1.named_parameters()):
        for name2, param2 in list(base_model2.named_parameters()):
            if name1 == name2:
                delta = torch.abs(param1-param2)
                delta = torch.mean(delta.view(-1)).item()
                deltas.append(delta)
    ind = np.argpartition(deltas, -k)[-k:] # max deltas (trainable layers)
    layers = [i[0] for i in list(base_model1.named_parameters())]
    running_layers = [layers[i] for i in ind]
    for name, param in base_model1.named_parameters():
        if name not in running_layers:
            param.requires_grad = False
    with open(output_file, "w") as final: json.dump(running_layers, final)
    return base_model1

def load_model(args, processor) -> VisionEncoderDecoderModel:
    model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(args.model_trocr)
    encoder = model.encoder

    debug_print(f"Loaded local model from {args.model_trocr}")
    debug_print(f"Using device {device}.")

    if args.is_sft ==1:
        params_model = dict(model.named_parameters())
        model_task = VisionEncoderDecoderModel.from_pretrained(args.model_task_path)
        params_model_task = dict(model_task.named_parameters())
        running_layers_task = open(args.running_layers_task_path)
        running_layers_task = json.load(running_layers_task)

        for name, param in params_model.items():
            if name in running_layers_task:
                diff_task = params_model_task[name].data - param.data
            else:
                diff_task = torch.zeros_like(param.data)
            param.data += diff_task

    if args.is_langauge_adapter==1:
        decoder = model.decoder
        decoder.load_state_dict(torch.load(os.path.join(args.model_language_phase2, 'model.pth'), weights_only=True))
        model.decoder = decoder

    if args.is_sft_train==1:
        model2: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(args.model_task_phase1)
        model = prepare_model(model, model2, args.running_layers)

    if args.train_lora_component == 1:
        model = VisionEncoderDecoderModel.from_pretrained(args.model_trocr)
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "k_proj",
                "q_proj",
                "v_proj",
                "out_proj",
                "fc1",
                "fc2",
            ],
            # target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        decoder = get_peft_model(model.decoder, peft_config)
        model.decoder = decoder

    if args.use_lora_style == 1:
        ## training with style adapters (rudolf) -lora
        model_english_htr = get_peft_decoder(args.model_trocr)
        # model_style_path
        model_english_htr.load_state_dict(torch.load(os.path.join(args.model_style_path, 'model.pth'), weights_only=True))
        for name, param in model_english_htr.named_parameters():
            param.requires_grad=True

        if args.freeze==1:
            for name, param in model_english_htr.named_parameters():
                if 'lora_A' in name or 'lora_B' in name:
                    param.requires_grad = False

        model = model_english_htr

    if args.use_lora_style_language == 1:
        model_english = get_decoder(args.model_trocr, processor.tokenizer)
        model_latin = get_decoder(args.model_trocr, processor.tokenizer)
        model_english_htr = get_peft_decoder(args.model_trocr)
        model_style = get_peft_decoder(args.model_trocr)

        model_english_htr.load_state_dict(torch.load(os.path.join(args.task_ability_adapter, 'model.pth'), weights_only=True))
        model_style.load_state_dict(torch.load(os.path.join(args.model_style_path, 'model.pth'), weights_only=True))

        model_english = PeftModel.from_pretrained(model_english, args.language_ability_dutch, adapter_name="English_CM")
        model_latin = PeftModel.from_pretrained(model_latin, args.language_ability_latin, adapter_name="Latin_CM")
        params_model_english = dict(model_english.named_parameters())
        params_model_latin = dict(model_latin.named_parameters())
        params_model_english_htr = dict(model_english_htr.named_parameters())
        params_model_style_htr = dict(model_style.named_parameters())

        a, b, c = [], [], []
        for name_latin, param_Latin in params_model_latin.items():
            if "Latin_CM" in name_latin:
                for name_english, param_english in params_model_english.items():
                    if 'English_CM' in name_english:
                        name_english = name_english.replace("English_CM", "Latin_CM")
                        if name_latin == name_english:
                            a.append(name_latin)
                            param_Latin.data = args.lora_language_weight*(param_Latin.data - param_english.data)

        for name_english_htr, param_english_htr in params_model_english_htr.items():
            if 'default' in name_english_htr:
                name_english_htr = name_english_htr.replace("default", "Latin_CM")
                name_english_htr = name_english_htr.replace("decoder.base_model", "base_model")
                for name_latin, param_Latin in params_model_latin.items():
                    if "Latin_CM" in name_latin:
                        if name_latin == name_english_htr:
                            b.append(name_latin)
                            param_english_htr.data = param_english_htr.data + param_Latin.data


        for name_style, param_style in params_model_style_htr.items():
            if 'default' in name_style:
                for name_english_htr, param_english_htr in params_model_english_htr.items():
                    if name_english_htr == name_style:
                        c.append(name_english_htr)
                        param_style.data = param_style.data + args.lora_language_task_weight * param_english_htr.data

        for name, param in model_style.named_parameters():
            param.requires_grad=True

        # model = model_english_htr
        model = model_style
    if args.use_lora == 1:
        model_english = get_decoder(args.model_trocr, processor.tokenizer)
        model_latin = get_decoder(args.model_trocr, processor.tokenizer)
        model_english_htr = get_peft_decoder(args.model_trocr)
        model_english_htr.load_state_dict(torch.load(os.path.join(args.task_ability_adapter, 'model.pth'), weights_only=True))

        model_english = PeftModel.from_pretrained(model_english, args.language_ability_dutch, adapter_name="English_CM")
        model_latin = PeftModel.from_pretrained(model_latin, args.language_ability_latin, adapter_name="Latin_CM")
        params_model_english = dict(model_english.named_parameters())
        params_model_latin = dict(model_latin.named_parameters())
        params_model_english_htr = dict(model_english_htr.named_parameters())

        for name_latin, param_Latin in params_model_latin.items():
            if "Latin_CM" in name_latin:
                for name_english, param_english in params_model_english.items():
                    if 'English_CM' in name_english:
                        name_english = name_english.replace("English_CM", "Latin_CM")
                        if name_latin == name_english:
                            param_Latin.data = args.lora_language_weight*(param_Latin.data - param_english.data)

        for name_english_htr, param_english_htr in params_model_english_htr.items():
            if 'default' in name_english_htr:
                name_english_htr = name_english_htr.replace("default", "Latin_CM")
                name_english_htr = name_english_htr.replace("decoder.base_model", "base_model")
                for name_latin, param_Latin in params_model_latin.items():
                    if "Latin_CM" in name_latin:
                        if name_latin == name_english_htr:
                            param_english_htr.data = param_english_htr.data + param_Latin.data

        for name, param in model_english_htr.named_parameters():
            param.requires_grad=True

        if args.freeze==1:
            for name, param in model_english_htr.named_parameters():
                if 'lora_A' in name or 'lora_B' in name:
                    param.requires_grad = False

        model = model_english_htr

    if args.use_lora0 == 1:
        ### training with english language adapter
        model_english = get_decoder(args.model_trocr, processor.tokenizer)
        model_latin = get_decoder(args.model_trocr, processor.tokenizer)
        model_english = PeftModel.from_pretrained(model_english, args.language_ability_dutch, adapter_name="English_CM")
        model_latin = PeftModel.from_pretrained(model_latin, args.language_ability_latin, adapter_name="Latin_CM")
        params_model_english = dict(model_english.named_parameters())
        params_model_latin = dict(model_latin.named_parameters())
        for name_latin, param_Latin in params_model_latin.items():
            if "Latin_CM" in name_latin:
                for name_english, param_english in params_model_english.items():
                    if 'English_CM' in name_english:
                        name_english = name_english.replace("English_CM", "Latin_CM")
                        if name_latin == name_english:
                            param_Latin.data = args.lora_language_weight*(param_Latin.data - param_english.data)
        for name, param in model_latin.named_parameters():
            param.requires_grad = True
        model_latin.base_model.model.output_projection.weight.requires_grad = True

        if args.freeze == 1:
            for name, param in model_latin.named_parameters():
                if 'layers' in name and 'Latin_CM' not in name:
                    param.requires_grad = False

        model.decoder = model_latin
    if args.is_latinbert == 1:
        decoder = BertGenerationDecoder.from_pretrained(args.model_decoder, is_decoder=True, add_cross_attention=True)
        model = VisionEncoderDecoderModel(encoder = encoder, decoder=decoder)
    if args.is_laberta == 1:
        decoder = RobertaForCausalLM.from_pretrained(args.model_decoder, is_decoder=True, add_cross_attention=True)
        model = VisionEncoderDecoderModel(encoder = encoder, decoder=decoder)
    if args.is_mt5 == 1:
        del model, encoder
        torch.set_default_tensor_type(torch.DoubleTensor)
        model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(args.model_trocr,
                                                                                     torch_dtype=torch.float64)
        encoder = model.encoder
        model_used_decoding = MT5ForConditionalGeneration.from_pretrained(args.model_decoder, add_cross_attention=True,
                                                                          torch_dtype=torch.float64)
        t5_decoder = model_used_decoding.decoder
        out_features = processor.tokenizer.vocab_size

        lm_head = model_used_decoding.lm_head
        in_features = lm_head.in_features
        bias = lm_head.bias
        lm_head = nn.Linear(in_features=in_features, out_features=out_features, bias=bias, dtype=torch.float64)

        config_decoder = model_used_decoding.config
        config_decoder.vocab_size = processor.tokenizer.vocab_size
        decoder = T5TrOCRForCausalLM(t5_decoder, lm_head, config_decoder)
        model = VisionEncoderDecoderModel(encoder = encoder, decoder=decoder)

    if args.is_mbart == 1:
        model_used_decoding = BartForConditionalGeneration.from_pretrained(args.model_decoder, add_cross_attention=True)
        bart_decoder = model_used_decoding.model.decoder
        lm_head = model_used_decoding.lm_head
        config_decoder = model_used_decoding.config
        decoder = T5TrOCRForCausalLM(bart_decoder, lm_head, config_decoder)
        model = VisionEncoderDecoderModel(encoder = encoder, decoder=decoder)

    model.to(device)
    return model


def init_model_for_training(model, processor, use_latin_tokenizer, is_latinbert, is_laberta, is_lata, is_mt5, is_mbart):
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    if use_latin_tokenizer ==1:
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.eos_token_id = processor.tokenizer.sep_token_id
        model.config.vocab_size = processor.tokenizer.vocab_size
        model.decoder.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.decoder.config.pad_token_id = processor.tokenizer.pad_token_id
        model.decoder.config.eos_token_id = processor.tokenizer.sep_token_id
        # model.decoder.config.vocab_size = processor.tokenizer.vocab_size
    if is_latinbert==1 or is_laberta==1 or is_mbart == 1:
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.eos_token_id = processor.tokenizer.sep_token_id
        model.config.vocab_size = processor.tokenizer.vocab_size
        model.decoder.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.decoder.config.pad_token_id = processor.tokenizer.pad_token_id
        model.decoder.config.eos_token_id = processor.tokenizer.sep_token_id
        model.decoder.config.vocab_size = processor.tokenizer.vocab_size
    if is_mt5 == 1:
        model.config.decoder_start_token_id = processor.tokenizer.pad_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.eos_token_id = processor.tokenizer.sep_token_id
        model.config.vocab_size = processor.tokenizer.vocab_size
        model.decoder.config.decoder_start_token_id = processor.tokenizer.pad_token_id
        model.decoder.config.pad_token_id = processor.tokenizer.pad_token_id
        model.decoder.config.eos_token_id = processor.tokenizer.sep_token_id
        model.decoder.config.vocab_size = processor.tokenizer.vocab_size

def debug_print(string: str):
    print(string)
