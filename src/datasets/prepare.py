import sys
import json
import logging

from tqdm import tqdm
from pathlib import Path
from typing import Optional

from src.datasets.metaclass import MetaDataClass

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
logger = logging.getLogger(__name__)
sys.path.append(str(wd))

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
    generative: bool = False
) -> None:
    """Prepare a CSV dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    logger.info("Loading data files ...")
    import pandas as pd

    # loading train set
    df_train = pd.read_csv(train_csv_path, dtype=str).fillna("")
    # if not (df_train.columns.values == data_class.columns).all():
    #     raise ValueError(f"Train CSV columns must be {data_class.columns}, found {df_train.columns.values}")
    train_data = json.loads(df_train[data_class.columns].to_json(orient="records", indent=4))

    # loading validation set
    df_val = pd.read_csv(val_csv_path, dtype=str).fillna("")
    # if not (df_val.columns.values == data_class.columns).all():
    #     raise ValueError(f"Val CSV columns must be {data_class.columns}, found {df_val.columns.values}")
    val_data = json.loads(df_val[data_class.columns].to_json(orient="records", indent=4))

    # loading train set
    df_test = pd.read_csv(test_csv_path, dtype=str)
    # if not (df_test.columns.values == data_class.columns).all():
    #     raise ValueError(f"Test CSV columns must be {data_class.columns}, found {df_test.columns.values}")
    test_data = json.loads(df_test[data_class.columns].to_json(orient="records", indent=4))

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
            generative=generative,
            data_class=data_class,
            n_shot=0
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
            generative=generative,
            data_class=data_class,
            n_shot=0
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
            generative=generative,
            data_class=data_class,
            n_shot=0
        )
        for sample in tqdm(test_data)
    ]
    test_set_1, test_set_2 = None, None
    if generative:
        print("Processing 1-shot test split ...")
        one_shot_columns = ["example1", "example1_response"]
        test_data_1 = json.loads(df_test.dropna(subset=["example1"])[data_class.columns + one_shot_columns].to_json(orient="records", indent=4))
        test_set_1 = [
            prepare_sample(
                example=sample,
                tokenizer=tokenizer,
                max_length=max_seq_length,
                mask_inputs=mask_inputs,
                ignore_index=ignore_index,
                personalized=personalized,
                instruct=instruct,
                generative=generative,
                data_class=data_class,
                n_shot=1
            )
            for sample in tqdm(test_data_1)
        ]
        
        print("Processing 2-shot test split ...")
        two_shot_columns = one_shot_columns + ["example2", "example2_response"]
        test_data_2 = json.loads(df_test.dropna(subset=["example1", "example2"])[data_class.columns + two_shot_columns].to_json(orient="records", indent=4))
        test_set_2 = [
            prepare_sample(
                example=sample,
                tokenizer=tokenizer,
                max_length=max_seq_length,
                mask_inputs=mask_inputs,
                ignore_index=ignore_index,
                personalized=personalized,
                instruct=instruct,
                generative=generative,
                data_class=data_class,
                n_shot=2
            )
            for sample in tqdm(test_data_2)
        ]
    
    return train_set, val_set, test_set, test_set_1, test_set_2


def prepare_sample(example: dict, tokenizer, max_length: int, mask_inputs: bool, ignore_index: int,
                   personalized: bool, instruct: bool, generative: bool, data_class: MetaDataClass, n_shot: int) -> dict:
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
    full_prompt = data_class.generate_prompt(example, personalized, instruct, generative, n_shot)
    # full_prompt_and_response = full_prompt + example["output"]
    # encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)
    labels = [float(example[_label]) for _label in data_class.labels]
    # When `generative` is True => the labels are the list of 0s and 1s of emotions
    # Otherwise => the labels are concatenated into a single string
    if generative:
        text_labels = [emotion for emotion, _label in zip(data_class.labels, labels) if _label]
        text_labels = ', '.join(text_labels)
        return {
            **example,
            # "input_ids": encoded_full_prompt_and_response,
            # "input_ids_no_response": encoded_full_prompt,
            "full_prompt": full_prompt,
            "text_labels": text_labels,
            "hot_labels": labels,
        }

    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)

    return {
        **example,
        # "input_ids": encoded_full_prompt_and_response,
        # "input_ids_no_response": encoded_full_prompt,
        "input_ids": encoded_full_prompt,
        "labels": labels,
    }
