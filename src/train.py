from src.utils import get_tokenizer, get_data_collector
from src.lora_utils import print_trainable_parameters, find_all_linear_names, \
    check_gradients, add_modules_to_save


import shutil


from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np


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

        lora_config = add_modules_to_save(model=model, lora_config=lora_config)

        lora_configuration = LoraConfig(
            **lora_config,
        )
        model = get_peft_model(model, lora_configuration)

        assert check_gradients(model=model, lora_config=lora_config)

        print_trainable_parameters(
            model=model,
            quantization_config=quantization_config
        )
    training_arguments = TrainingArguments(
        **training_config,
    )
    data_collator = get_data_collector(base_model_config=base_model_config)

    if base_model_config['problem_type'] == 'multi_label_classification':
        accuracy_metric = evaluate.load('accuracy', 'multilabel')
        f1_metric = evaluate.load('f1', 'multilabel')
    else:
        accuracy_metric = evaluate.load('accuracy')
        f1_metric = evaluate.load('f1')

    def compute_metrics(eval_pred):
        logits = eval_pred[0]
        if isinstance(logits, tuple):  # Idk why it is sometimes tuple.
            logits = logits[0]
        labels = eval_pred[1]
        predictions = logits > 0
        predictions = np.intc(predictions)
        labels = np.intc(labels)
        metrics = accuracy_metric.compute(predictions=predictions, references=labels)
        metrics = metrics | f1_metric.compute(predictions=predictions, references=labels, average='micro')
        metrics['f1_micro'] = metrics.pop('f1')
        metrics = metrics | f1_metric.compute(predictions=predictions, references=labels, average='macro')
        metrics['f1_macro'] = metrics.pop('f1')
        return metrics

    trainer = Trainer(
        model=model,
        train_dataset=data_dict['train'],
        eval_dataset=data_dict['validation'],
        args=training_arguments,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model()
    shutil.copy2('config.ini', training_config['output_dir'])
    scores = trainer.evaluate(eval_dataset=data_dict['test'])
    print('Test_scores', scores)
    return
