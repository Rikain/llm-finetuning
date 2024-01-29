from transformers import (
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    AutoConfig,
    T5Config,
)
from peft import PeftModel


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
    ft_model = PeftModel.from_pretrained(
        base_model,
        model_checkpoint_path,
    )
    return ft_model


def _update_quantization_config(base_model_name, quantization_config):
    config = AutoConfig.from_pretrained(base_model_name)
    if isinstance(config, T5Config):
        quantization_config['llm_int8_skip_modules'] = ['dense', 'out_proj']
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
    num_labels = base_model_config['num_labels']
    problem_type = base_model_config['problem_type']
    # attn_implementation = base_model_config['attn_implementation']
    quantization_config = _update_quantization_config(
        base_model_name=base_model_name,
        quantization_config=quantization_config
    )
    quantization_config = _bnb_quantization_config(quantization_config)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=base_model_name,
        # ValueError: You can't train a model that has been loaded with `device_map='auto'` in any distributed mode.
        device_map="auto",
        quantization_config=quantization_config,
        num_labels=num_labels,
        problem_type=problem_type,
        # attn_implementation=attn_implementation,
    )
    if pad_token_id is not None:
        model.config.pad_token_id = pad_token_id
    return model
