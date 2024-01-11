from transformers import (
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import PeftModel


def load_finetuned(base_model, model_checkpoint_path):
    ft_model = PeftModel.from_pretrained(
        base_model,
        model_checkpoint_path,
    )
    return ft_model


def _bnb_quantization_config(quantization_config):
    if quantization_config is not None:
        quantization_config = BitsAndBytesConfig(
            **quantization_config
        )
    return quantization_config


def get_model(base_model_config, quantization_config=None, pad_token_id=None):
    """
    From AutoModelForSequenceClassification:
    if self.num_labels == 1:
        self.config.problem_type = "regression"
    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
        self.config.problem_type = "single_label_classification"
    else:
        self.config.problem_type = "multi_label_classification"
    """
    base_model_name = base_model_config['base_model_name']
    num_labels = base_model_config['num_labels']
    problem_type = base_model_config['problem_type']
    quantization_config = _bnb_quantization_config(quantization_config)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        device_map="auto",
        quantization_config=quantization_config,
        num_labels=num_labels,
    )
    model.config.problem_type = problem_type
    if pad_token_id is not None:
        model.config.pad_token_id = pad_token_id
    return model
