"""Utility functions to use across multiple independent files."""
import ast
from pathlib import Path
from configparser import ConfigParser

import torch
import evaluate
import numpy as np

from trl import DataCollatorForCompletionOnlyLM


from transformers import set_seed
from transformers import AutoTokenizer, DataCollatorWithPadding

from src.datasets import load
from src.datasets.prepare import response_template
from src.datasets import GoEmo, Unhealthy, Docanno



def get_tokenizer(base_model_config):
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=base_model_config['pretrained_model_name_or_path'],
        padding=True,
        truncation=True,
        max_seq_length=base_model_config['max_seq_length'],
    )
    if tokenizer.pad_token is None or not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.unk_token
        pad_token_id = tokenizer.unk_token_id
    else:
        pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = 'right'
    return tokenizer, pad_token_id


def formatting_prompts_func(example):
    output_texts = []
    print("dupa")
    for i in range(len(example['full_prompt'])):
        text = f"{example['full_prompt'][i]}{example['text_labels'][i]}"
        output_texts.append(text)
    print(output_texts)
    return output_texts


def get_data_collector(base_model_config):
    tokenizer, _ = get_tokenizer(base_model_config=base_model_config)
    if base_model_config['problem_type'] == 'generative_multi_label_classification':
        # Creates a problem because toknizer beigns with begginging of sentence token
        data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=tokenizer(response_template, add_special_tokens=False)['input_ids'][2:])
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return data_collator


def read_config(config_file='real-config.ini'):
    """Change config file into a dictionary.

    Args:
        config (str, optional): The config filename. Defaults to 'config.ini'.

    Returns:
        Dict: Dictionary with config values.
    """
    config = ConfigParser()
    config_dict = dict()
    if not Path(config_file).is_file():
        config_file = 'config.ini'
    config.read(config_file)
    for category in config.sections():
        config_dict[category] = dict()
        for option in config[category]:
            try:
                config_dict[category][option] = ast.literal_eval(
                    config[category][option]
                    )
            except Exception:
                if 'dtype' in option:
                    config_dict[category][option] = \
                        eval(config[category][option])
                else:
                    config_dict[category][option] = config[category][option]
    return config_dict


def seed_everything(seed=42):
    set_seed(seed)
    return


def prepare_configuration():
    config = read_config()
    base_model_config, lora_config, quantization_config, \
        training_config, data_config, seed = parse_config(
            config=config
        )

    try:
        # Try to check if data_class has been imported
        data_class = data_config['data_class']
        data_class = eval(data_class)
    except NameError:
        raise Exception(
            f"The dataset class `{data_class}` does not exist. "
            "Change your `data_class` property in `config.ini` to one of src.datasets classes."
        )

    if quantization_config is not None:
        assert lora_config is not None
    base_model = base_model_config['pretrained_model_name_or_path']

    max_seq_length = base_model_config['max_seq_length']
    tokenizer, pad_token_id = get_tokenizer(base_model_config=base_model_config)

    data_dict, num_labels, label_names = load(
        base_model_config,
        data_config,
        tokenizer,
        data_class
    )

    base_model_config['num_labels'] = num_labels
    base_model_config['label_names'] = label_names

    return seed, base_model_config, lora_config, quantization_config, \
        training_config, data_dict, pad_token_id


def get_metrics_evaluators(base_model_config):
    if base_model_config['problem_type'] == 'multi_label_classification':
        accuracy_metric = evaluate.load('accuracy', 'multilabel')
        f1_metric = evaluate.load('f1', 'multilabel')
    else:
        accuracy_metric = evaluate.load('accuracy')
        f1_metric = evaluate.load('f1')

    def compute_metrics(eval_pred):
        logits = eval_pred[0]
        if isinstance(logits, tuple):  # Idk why it is sometimes tuple.
            logits = logits[0]
        labels = eval_pred[1]
        predictions = logits > 0
        predictions = np.intc(predictions)
        labels = np.intc(labels)
        metrics = accuracy_metric.compute(
            predictions=predictions,
            references=labels,
        )
        metrics = metrics | f1_metric.compute(
            predictions=predictions,
            references=labels,
            average='micro'
        )
        metrics['f1_micro'] = metrics.pop('f1')
        metrics = metrics | f1_metric.compute(
            predictions=predictions,
            references=labels,
            average='macro',
        )
        metrics['f1_macro'] = metrics.pop('f1')
        if 'label_names' in base_model_config \
                and base_model_config['label_names'] is not None:
            per_label_f1 = f1_metric.compute(
                predictions=predictions,
                references=labels,
                average=None,
            )
            metrics = metrics | dict(
                zip([label_name + '_f1' for label_name in
                     base_model_config['label_names']], per_label_f1['f1'].tolist())
            )
        return metrics
    
    def compute_metrics_from_text(eval_pred):
        ids = eval_pred[0]
        if isinstance(ids, tuple):  # Idk why it is sometimes tuple.
            ids = ids[0]
        labels = eval_pred[1]
        #TODO

    
    if base_model_config['problem_type'] == 'multi_label_classification':
        return_metric_function = compute_metrics
    elif base_model_config['problem_type'] == 'generative_multi_label_classification':
        return_metric_function = compute_metrics_from_text
    else:
        return_metric_function = None
        
    return (accuracy_metric, f1_metric), return_metric_function


def parse_config(config):
    general_config = config['general']
    seed = general_config['seed']
    base_model_config = config['base_model_config']
    if general_config['use_lora']:
        lora_config = config['lora_config']
    else:
        lora_config = None
    if general_config['use_quantization']:
        quantization_config = config['quantization_config']
    else:
        quantization_config = None
    training_config = config['training_config']
    training_config['seed'] = seed
    data_config = config['data_config']
    if data_config['generative']:
        if base_model_config['problem_type'] == 'multi_label_classification':
            base_model_config['problem_type'] = 'generative_multi_label_classification'
    if base_model_config['problem_type'] == 'generative_multi_label_classification':
        assert data_config['generative']
        if lora_config is not None and lora_config['task_type'] == 'SEQ_CLS':
            lora_config['task_type'] = 'CASUAL_LM'
    return base_model_config, lora_config, quantization_config, \
        training_config, data_config, seed