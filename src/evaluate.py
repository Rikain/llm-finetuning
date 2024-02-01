from src.train import prepare_model, prepare_trainer
from src.model.load import get_model
from src.utils import prepare_configuration, seed_everything, get_tokenizer
from src.datasets import GoEmo, Unhealthy, Docanno

import torch
import sys
from tqdm import tqdm
from typing import Dict, Union, List, Any
from pathlib import Path
from datasets import Dataset
from sklearn.metrics import f1_score


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))



def get_response(sample, tokenizer, model):
    model_input = tokenizer([sample], return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**model_input, max_new_tokens=64)
    response = decode_response(out, tokenizer)
    return response
   
def decode_response(out, tokenizer):
    return tokenizer.decode(out[0], skip_special_tokens=True)
    
def map_text_to_vector(
  response: str,
  labels_map: Dict[str, int],
  delimiter: str = ',',
):
    split_response = response.split(delimiter)
    vector = torch.zeros(len(labels_map))
    for pred_lab in split_response:
        curr = pred_lab.strip()
        if curr in labels_map:
            vector[labels_map[curr]] = 1
    return vector

def process_one_sample(
    sample_text: str,
    sample_label: str,
    tokenizer: Any,
    model: Any,
    labels_map: Dict[str, int],
    delimiter: str = ',',
):
    ground_truth = map_text_to_vector(sample_label, labels_map, delimiter)
    response = get_response(sample_text, tokenizer, model)
    prediction = map_text_to_vector(response, labels_map, delimiter)
    return ground_truth, prediction


def evaluate_dataset(
    dataset,
    model,
    tokenizer,
    data_class,
    delimiter=','
):
    
    labels_map =  {label: i for i, label in enumerate(data_class.labels)} 
    true = torch.empty((len(dataset),len(labels_map)))
    pred = torch.empty((len(dataset),len(labels_map)))
    
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        sample_text = sample["text"]
        sample_label = sample["text_labels"]
        sample_true, sample_pred = process_one_sample(
            sample_text,
            sample_label,
            tokenizer,
            model,
            labels_map,
            delimiter
        )
        true[i] = sample_true
        pred[i] = sample_pred

    f1 = f1_score(true, pred, average='macro')
    return f1


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
    tokenizer, _ = get_tokenizer(base_model_config)


    score =  evaluate_dataset(
        data_dict["test_1_shot"],
        model,
        tokenizer,
        GoEmo
    )
    print(score)
    


if __name__ == "__main__":
    main_test()