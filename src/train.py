from src.utils import get_data_collector, get_metrics_evaluators
from src.lora_utils import print_trainable_parameters, find_all_linear_names, \
    check_gradients, add_modules_to_save


import shutil
from pathlib import Path


from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import TrainingArguments, Trainer


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

    metrics, compute_metrics = get_metrics_evaluators(
        base_model_config=base_model_config
    )

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
    config_file = 'real-config.ini'
    if not Path(config_file).is_file():
        config_file = 'config.ini'
    shutil.copy2(config_file, training_config['output_dir'])
    if base_model_config['problem_type'] == 'generative_multi_label_classification':
        # Calcualte F1 using code that measures the performance of a trained model.
        scores = trainer.evaluate(eval_dataset=data_dict['test'])
        print('Test_scores', scores)
        return
    scores = trainer.evaluate(eval_dataset=data_dict['test'])
    print('Test_scores', scores)
    return
