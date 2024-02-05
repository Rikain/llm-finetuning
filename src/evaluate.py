from src.model.load import get_model, load_finetuned
from src.utils import prepare_configuration, seed_everything, get_tokenizer
from src.datasets import GoEmo, Unhealthy, Docanno

import torch
from torchmetrics import F1Score
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
from typing import Dict, Union, List, Any
from pathlib import Path
from datasets import Dataset
import pandas as pd
from pathlib import Path


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))



def get_responses(batch, tokenizer, model, max_tokens):
    model_input = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**model_input, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id).to("cuda")
    out = out[:, model_input["input_ids"].shape[1]:].to("cuda")

    responses = tokenizer.batch_decode(out, skip_special_tokens=True)
    return responses


def clean_word(word):
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    # punctuation
    word = word.replace(punc, "")
    # trailing spaces
    word = word.strip()
    return word
    

def map_texts_to_vectors(
    responses: List[str],
    labels_map: Dict[str, int],
    delimiter: str = ',',
):
    batch_matrix = [single_mapping(response, labels_map, delimiter) for response in responses]
    return torch.tensor(batch_matrix)


def single_mapping(
    response: str,
    labels_map: Dict[str, int],
    delimiter: str = ',',
):
    split_response = response.split(delimiter)
    vector = [0 for _ in range(len(labels_map))]
    for pred_lab in split_response:
        curr = clean_word(pred_lab)
        if curr in labels_map:
            vector[labels_map[curr]] = 1
    return vector


def process_one_sample(
    batch: List[str],
    tokenizer: Any,
    model: Any,
    max_tokens: int,
    labels_map: Dict[str, int],
    delimiter: str = ',',
):
    responses = get_responses(batch, tokenizer, model, max_tokens)
    prediction = map_texts_to_vectors(responses, labels_map, delimiter)
    return responses, prediction


def evaluate_dataset(
    dataset,
    model,
    tokenizer,
    data_class,
    out_file,
    delimiter=','
):
    
    labels_map =  {label: i for i, label in enumerate(data_class.labels)} 
    max_tokens = tokenizer([", ".join(data_class.labels)], return_tensors="pt")["input_ids"].shape[1]
    f1 = F1Score(task="multilabel", num_labels = len(labels_map), average="macro")
    dataloader = DataLoader(dataset, batch_size=32)
    texts_to_save = {"prompt": [], "response": [], "expected": []}

    for batch in tqdm(dataloader):
        text_batch = batch["full_prompt"]
        true_labels = torch.stack(batch["hot_labels"], dim=1)

        responses, preds = process_one_sample(
            text_batch,
            tokenizer,
            model,
            max_tokens,
            labels_map,
            delimiter
        )
        
        texts_to_save["prompt"].extend(text_batch)
        texts_to_save["response"].extend(responses)
        texts_to_save["expected"].extend(batch["text_labels"])
        
        f1.update(preds, true_labels)
        
    texts_to_save = pd.DataFrame(texts_to_save).to_csv(out_file, index=False)
    result = f1.compute()
    return result


def evaluate_all_tests(data_dict, model, tokenizer, data_class, out_filename, out_path):
    tests = ["test", "test_1_shot", "test_2_shot"]
    for test in tests:
        print("Evaluating", test, "...")
        curr_out_file = out_path / (out_filename + "_" + test + ".csv")

        score =  evaluate_dataset(
            data_dict[test],
            model,
            tokenizer,
            data_class,
            curr_out_file
        )
        print("F1 Macro:", score)

def main_test(load_tuned=False):
    seed, base_model_config, lora_config, quantization_config, \
        training_config, data_dict, pad_token_id, data_config = prepare_configuration()
    
    seed_everything(seed)
    if load_tuned:
        model = load_finetuned(base_model_config=base_model_config,
                            quantization_config=quantization_config,
                            pad_token_id=pad_token_id,
                            model_checkpoint_path=training_config['output_dir'],
                            )
    else:
        model = get_model(base_model_config, quantization_config, pad_token_id)
    tokenizer, _ = get_tokenizer(base_model_config, padding_side="left")

    model.eval()
    
    out_filename = "responses"
    if data_config["personalized"]:
        out_filename += "_pers"
    
    out_path = Path(data_config["responses_dirname"]) / data_config["data_folder"] / training_config['output_dir']
    out_path.mkdir(parents=True, exist_ok=True)
    evaluate_all_tests(data_dict, model, tokenizer, data_config["data_class"], out_filename, out_path)    


if __name__ == "__main__":
    main_test(load_tuned=True)