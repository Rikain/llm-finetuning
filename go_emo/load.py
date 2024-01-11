from go_emo.prepare_goemotions import prepare
from go_emo.prepare_goemotions import EMOTIONS


from datasets import DatasetDict


def load(base_model_config, tokenizer):
    base_model = base_model_config['base_model_name']
    max_seq_length = base_model_config['max_seq_length']
    personalized = True,
    instruct = True
    train, val, test = prepare(
        base_model=base_model,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        personalized=personalized,
        instruct=instruct)
    data_dict = DatasetDict(
            {
                "train": train,
                "test": val,
                "validation": test,
            }
        )
    num_labels = len(EMOTIONS)
    return data_dict, num_labels
