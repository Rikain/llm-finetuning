"""Utility functions to use across multiple independent files."""
import ast
from configparser import ConfigParser


import torch
from transformers import AutoTokenizer, DataCollatorWithPadding


def get_tokenizer(base_model, max_seq_length):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=base_model,
        padding=True,
        truncation=True,
        max_seq_length=max_seq_length,
    )
    if tokenizer.pad_token is None or not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.unk_token
        pad_token_id = tokenizer.unk_token_id
    else:
        pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = 'right'
    return tokenizer, pad_token_id


def get_data_collector(base_model_config): 
    tokenizer, _ = get_tokenizer(
        base_model=base_model_config['base_model_name'],
        max_seq_length=base_model_config['max_seq_length']
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return data_collator


def read_config(config_file='config.ini'):
    """Change config file into a dictionary.

    Args:
        config (str, optional): The config filename. Defaults to 'config.ini'.

    Returns:
        Dict: Dictionary with config values.
    """
    config = ConfigParser()
    config_dict = dict()
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


def parse_config(config):
    general_config = config['general']
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
    return base_model_config, lora_config, quantization_config, \
        training_config
