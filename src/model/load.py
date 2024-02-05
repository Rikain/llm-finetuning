import inspect


from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    AutoConfig,
    T5Config,
)
from peft import PeftModel
import torch
from accelerate import Accelerator


def load_finetuned(
        base_model_config,
        model_checkpoint_path,
        quantization_config=None,
        pad_token_id=None
        ):
    base_model = get_model(
        base_model_config=base_model_config,
        quantization_config=quantization_config,
        pad_token_id=pad_token_id,
    )
    print(base_model)
    ft_model = PeftModel.from_pretrained(
        base_model,
        model_checkpoint_path,
    )
    return ft_model


def recongnize_t5(base_model_name):
    config = AutoConfig.from_pretrained(base_model_name)
    return isinstance(config, T5Config)


def _update_quantization_config(base_model_config, quantization_config):
    if recongnize_t5(base_model_config['pretrained_model_name_or_path']):
        quantization_config['llm_int8_skip_modules'] = ['dense', 'out_proj']
    else:
        if base_model_config['problem_type'] == 'generative_multi_label_classification':
            tune_lm_head = base_model_config.pop('tune_lm_head', False)
            if tune_lm_head:
                quantization_config['llm_int8_skip_modules'] = ['embed_out', 'lm_head']
            else:
                quantization_config['llm_int8_skip_modules'] = []
        else:
            quantization_config['llm_int8_skip_modules'] = ['score']
    return quantization_config


def _bnb_quantization_config(quantization_config):
    if quantization_config is not None:
        quantization_config = BitsAndBytesConfig(
            **quantization_config
        )
    return quantization_config


def get_model(base_model_config, quantization_config=None, pad_token_id=None):
    """
    Multilabel classification labels need to be float values because of the loss function
    that is used in AutoModelForSequenceClassification for multilabel classification.
    """
    base_model_name = base_model_config['pretrained_model_name_or_path']
    problem_type = base_model_config['problem_type']
    quantization_config = _update_quantization_config(
        base_model_config=base_model_config,
        quantization_config=quantization_config
    )
    quantization_config = _bnb_quantization_config(quantization_config)
    device_index = Accelerator().process_index
    if problem_type == 'generative_multi_label_classification':
        if recongnize_t5(base_model_name):
            argnames = set(inspect.getargspec(AutoModelForSeq2SeqLM.from_pretrained)[0])
            kwargs = {k: v for k, v in base_model_config.items() if k in argnames}
            model = AutoModelForSeq2SeqLM.from_pretrained(
                **kwargs,
                device_map={"": device_index},
                quantization_config=quantization_config,
                attn_implementation=base_model_config['attn_implementation'],
            )
        else:
            argnames = set(inspect.getargspec(AutoModelForCausalLM.from_pretrained)[0])
            kwargs = {k: v for k, v in base_model_config.items() if k in argnames}
            model = AutoModelForCausalLM.from_pretrained(
                **kwargs,
                device_map={"": device_index},
                quantization_config=quantization_config,
                attn_implementation=base_model_config['attn_implementation'],
            )
    else:
        argnames = set(inspect.getargspec(AutoModelForSequenceClassification.from_pretrained)[0])
        kwargs = {k: v for k, v in base_model_config.items() if k in argnames}
        model = AutoModelForSequenceClassification.from_pretrained(
            **kwargs,
            device_map={"": device_index},
            quantization_config=quantization_config,
            num_labels=base_model_config['num_labels'],
            problem_type=base_model_config['problem_type'],
            attn_implementation=base_model_config['attn_implementation'],
        )
    if pad_token_id is not None:
        model.config.pad_token_id = pad_token_id
    return model
