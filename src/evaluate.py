from src.train import prepare_model, prepare_trainer
from src.model.load import get_model
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
from sklearn.metrics import f1_score


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))



def get_responses(batch, tokenizer, model):
    model_input = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**model_input, max_new_tokens=16)
    out = out[:, model_input["input_ids"].shape[1]:]
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
    labels_map: Dict[str, int],
    delimiter: str = ',',
):
    responses = get_responses(batch, tokenizer, model)
    prediction = map_texts_to_vectors(responses, labels_map, delimiter)
    return prediction


def evaluate_dataset(
    dataset,
    model,
    tokenizer,
    data_class,
    delimiter=','
):
    
    labels_map =  {label: i for i, label in enumerate(data_class.labels)} 
    f1 = F1Score(task="multilabel", num_labels = len(labels_map), average="macro")
    dataloader = DataLoader(dataset, batch_size=8)
    for batch in tqdm(dataloader):
        text_batch = batch["full_prompt"]
        true_labels = torch.stack(batch["hot_labels"], dim=1)

        preds = process_one_sample(
            text_batch,
            tokenizer,
            model,
            labels_map,
            delimiter
        )
        
        f1.update(preds, true_labels)
        
    result = f1.compute()
    print(result)

    # f1 = f1_score(true, pred, average='macro')
    return result


def main_test():
    seed, base_model_config, lora_config, quantization_config, \
        training_config, data_dict, pad_token_id = prepare_configuration()
    
    seed_everything(seed)
    model = get_model(base_model_config, quantization_config, pad_token_id)
    model = prepare_model(
        model=model,
        quantization_config=quantization_config,
        lora_config=lora_config,
    )
    tokenizer, _ = get_tokenizer(base_model_config, padding_side="left")

    score =  evaluate_dataset(
        [data_dict["test"][i] for i in range(32)],
        model,
        tokenizer,
        GoEmo
    )
    print(score)
    


if __name__ == "__main__":
    main_test()