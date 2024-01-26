from src.go_emo.prepare_goemotions import prepare
from src.go_emo.prepare_goemotions import EMOTIONS


from datasets import DatasetDict
from pathlib import Path


def load(base_model_config, data_config, tokenizer):
    base_model = base_model_config['pretrained_model_name_or_path']
    max_seq_length = base_model_config['max_seq_length']
    personalized = data_config['personalized']
    instruct = data_config['instruct']
    data_dir = Path(data_config['data_folder'])
    train, val, test = prepare(
        base_model=base_model,
        train_csv_path=data_dir/data_config['train_filename'],
        val_csv_path=data_dir/data_config['val_filename'],
        test_csv_path=data_dir/data_config['test_filename'],
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        personalized=personalized,
        instruct=instruct
    )
    data_dict = DatasetDict(
            {
                "train": train,
                "test": val,
                "validation": test,
            }
        )
    num_labels = len(EMOTIONS)
    return data_dict, num_labels
