import abc
import sys
import json
import logging

from tqdm import tqdm
from pathlib import Path
from typing import Optional

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
logger = logging.getLogger(__name__)
sys.path.append(str(wd))


class MetaDataClass:

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @classmethod
    def generate_prompt(cls, example: dict, personalized: bool, instruct: bool) -> str:
        pass


class GoEmo(MetaDataClass):

    labels = [
        'admiration','amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
        'remorse', 'sadness', 'surprise', 'neutral'
    ]
    columns = ["rater_id", "text"] + list(labels)

    def __init__(self):
        super(MetaDataClass, self).__init__()

    @classmethod
    def generate_prompt(cls, example: dict, personalized: bool, instruct: bool) -> str:
        """Generates a standardized message to prompt the model with an instruction, optional input and a
        'response' field."""

        if instruct:
            if personalized:
                return (
                    "Categorize the following text for the specified user by selecting the most appropriate emotion from the provided list."
                    "Emotions can be subtle or overt, so analyze the text carefully to make an accurate classification.\n\n"
                    f"### User ID:\n{example['rater_id']}\n\n"
                    f"### Text:\n{example['text']}\n\n"
                    "### Emotions:\n" + "\n- ".join(cls.labels) + "\n\n"
                    "### Response:"
                )
            else:
                return (
                    "Categorize the following text by selecting the most appropriate emotion from the provided list."
                    "Emotions can be subtle or overt, so analyze the text carefully to make an accurate classification.\n\n"
                    f"### Text:\n{example['text']}\n\n"
                    "### Emotions:\n" + "\n- ".join(cls.labels) + "\n\n"
                    "### Response:"
                )
        else:
            if personalized:
                return (
                    f"{example['rater_id']}\n{example['text']}"
                )
            else:
                return example['text']


class Unhealthy(MetaDataClass):

    def __init__(self):
        super(MetaDataClass, self).__init__()

    @classmethod
    def generate_prompt(cls, example: dict, personalized: bool, instruct: bool) -> str:
        pass


class Docanno(MetaDataClass):

    def __init__(self):
        super(MetaDataClass, self).__init__()

    @classmethod
    def generate_prompt(cls, example: dict, personalized: bool, instruct: bool) -> str:
        pass


def prepare(
    base_model: str,
    max_seq_length: int,
    tokenizer,
    personalized: bool,
    instruct: bool,
    data_class: MetaDataClass,
    train_csv_path: Path = Path("data/personalized/train.csv"),
    val_csv_path: Path  = Path("data/personalized/val.csv"),
    test_csv_path: Path  = Path("data/personalized/test.csv"),
    mask_inputs: bool = False,
    ignore_index: int = -1,
    encode_labels: bool = False
) -> None:
    """Prepare a CSV dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    logger.info("Loading data files ...")
    import pandas as pd

    # loading train set
    df_train = pd.read_csv(train_csv_path, dtype=str).fillna("")[data_class.columns]
    if not (df_train.columns.values == data_class.columns).all():
        raise ValueError(f"Train CSV columns must be {data_class.columns}, found {df_train.columns.values}")
    train_data = json.loads(df_train.to_json(orient="records", indent=4))

    # loading validation set
    df_val = pd.read_csv(val_csv_path, dtype=str).fillna("")[data_class.columns]
    if not (df_val.columns.values == data_class.columns).all():
        raise ValueError(f"Val CSV columns must be {data_class.columns}, found {df_val.columns.values}")
    val_data = json.loads(df_val.to_json(orient="records", indent=4))
    
    # loading train set
    df_test = pd.read_csv(test_csv_path, dtype=str).fillna("")[data_class.columns]
    if not (df_test.columns.values == data_class.columns).all():
        raise ValueError(f"Test CSV columns must be {data_class.columns}, found {df_test.columns.values}")
    test_data = json.loads(df_test.to_json(orient="records", indent=4))
    
    
    print(f"train has {len(train_data):,} samples")
    print(f"val has {len(val_data):,} samples")
    print(f"test has {len(test_data):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
            personalized=personalized,
            instruct=instruct,
            encode_labels=encode_labels,
            data_class=data_class
        )
        for sample in tqdm(train_data)
    ]

    print("Processing val split ...")
    val_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
            personalized=personalized,
            instruct=instruct,
            encode_labels=encode_labels,
            data_class=data_class
        )
        for sample in tqdm(val_data)
    ]
    
    print("Processing test split ...")
    test_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
            personalized=personalized,
            instruct=instruct,
            encode_labels=encode_labels,
            data_class=data_class
        )
        for sample in tqdm(test_data)
    ]
    return train_set, val_set, test_set
    

def prepare_sample(example: dict, tokenizer, max_length: int, mask_inputs: bool, ignore_index: int,
                   personalized: bool, instruct: bool, encode_labels: bool, data_class: MetaDataset) -> dict:
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response vector

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = data_class.generate_prompt(example, personalized, instruct)
    # full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    # encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

    labels = {label: float(example[label]) for label in data_class.labels}
    # When `encode_labels` is True => the labels are the list of 0s and 1s of emotions
    # Otherwise => the labels are concatenated into a single string
    labels = list(labels.values()) if encode_labels else ", ".join(labels.keys())
    
    return {
        **example,
        # "input_ids": encoded_full_prompt_and_response,
        # "input_ids_no_response": encoded_full_prompt,
        "input_ids": encoded_full_prompt,
        "labels": labels,
    }