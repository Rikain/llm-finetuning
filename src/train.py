from src.utils import get_tokenizer, get_data_collector
from src.lora_utils import print_trainable_parameters, find_all_linear_names

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import TrainingArguments, Trainer
from datasets import load_metric


def train(
    model,
    data_dict,
    base_model_config,
    training_config,
    quantization_config=None,
    lora_config=None,
):
    model.config.use_cache = False
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    if lora_config is not None:
        if lora_config['target_modules'] == 'all':
            lora_config['target_modules'] = find_all_linear_names(
                model=model,
                quantization_config=quantization_config
            )
        lora_config = LoraConfig(
            **lora_config,
        )
        model = get_peft_model(model, lora_config)

        print_trainable_parameters(
            model=model,
            quantization_config=quantization_config
        )
    training_arguments = TrainingArguments(
        **training_config,
    )
    data_collator = get_data_collector(base_model_config=base_model_config)
    metric = load_metric('f1', 'accuracy')
    trainer = Trainer(
            model=model,
            train_dataset=data_dict["train"],
            eval_dataset=data_dict["validation"],
            args=training_arguments,
            compute_metrics=metric,
            data_collator=data_collator,
        )
    trainer.train()
    trainer.save_model()
    trainer.evaluate(eval_dataset=data_dict['test'])
    return
