from src.datasets.prepare import prepare


from datasets import DatasetDict, Dataset
from pathlib import Path


def load(base_model_config, data_config, tokenizer, data_class):
    base_model = base_model_config['pretrained_model_name_or_path']
    max_seq_length = base_model_config['max_seq_length']

    personalized = data_config['personalized']
    instruct = data_config['instruct']
    generative = data_config['generative']

    data_dir = Path(data_config['data_folder'])

    train, val, test, test_1_shot, test_2_shot = prepare(
        base_model=base_model,
        train_csv_path=data_dir/data_config['train_filename'],
        val_csv_path=data_dir/data_config['val_filename'],
        test_csv_path=data_dir/data_config['test_filename'],
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        personalized=personalized,
        instruct=instruct,
        generative=generative,
        data_class=data_class,
    )
    train = Dataset.from_list(train)
    val = Dataset.from_list(val)
    test = Dataset.from_list(test)
    few_shot = {}
    if test_1_shot is not None:
        test_1_shot = Dataset.from_list(test_1_shot)
        few_shot["test_1_shot"] = test_1_shot 
    if test_2_shot is not None:
        test_2_shot = Dataset.from_list(test_2_shot)
        few_shot["test_2_shot"] = test_2_shot

    data_dict = DatasetDict(
            {
                "train": train,
                "validation": val,
                "test": test,
                **few_shot
            }
        )
    num_labels = len(data_class.labels)
    return data_dict, num_labels, data_class.labels
