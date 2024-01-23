from src.utils import read_config, parse_config
from src.train import train
from src.model.load import get_model
from src.utils import get_tokenizer


from importlib.machinery import SourceFileLoader
from pathlib import Path


def main():
    config = read_config()
    base_model_config, lora_config, quantization_config, \
        training_config, data_config = parse_config(
            config=config
            )
    if quantization_config is not None:
        assert lora_config is not None
    dataset_loader_folder = Path("src") / config['paths']['dataset_loader_folder']
    dataset_functions = SourceFileLoader(
        "dataset_module", (dataset_loader_folder / 'load.py').as_posix()
    ).load_module()
    base_model = base_model_config['base_model_name']
    max_seq_length = base_model_config['max_seq_length']
    tokenizer, pad_token_id = get_tokenizer(base_model, max_seq_length)
    data_dict, num_labels = dataset_functions.load(
        base_model_config,
        config['paths'],
        tokenizer
    )
    base_model_config['num_labels'] = num_labels
    model = get_model(base_model_config, quantization_config, pad_token_id)
    train(
        model=model,
        data_dict=data_dict,
        base_model_config=base_model_config,
        training_config=training_config,
        quantization_config=quantization_config,
        lora_config=lora_config
    )
    return


if __name__ == '__main__':
    main()
