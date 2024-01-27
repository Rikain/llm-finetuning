from src.datasets.prepare import prepare


from datasets import DatasetDict
from pathlib import Path


def load(base_model_config, data_config, tokenizer, data_class):
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
        instruct=instruct,
        data_class=data_class
    )

    data_dict = DatasetDict(
            {
                "train": train,
                "test": val,
                "validation": test,
            }
        )
    num_labels = len(data_class.labels)
    return data_dict, num_labels, data_class.labels
