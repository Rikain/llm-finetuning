from typing import Any
from src.utils import get_data_collector, get_metrics_evaluators, \
    formatting_prompts_func, get_tokenizer
from src.lora_utils import print_trainable_parameters, find_all_linear_names, \
    check_gradients, add_modules_to_save
from src.evaluate import main_test
from src.model.load import recongnize_t5


import shutil
from pathlib import Path


from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from trl import SFTTrainer


def prepare_model(
    model,
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

        # TODO
        # Add LoftQ parameter inicialization
        # https://huggingface.co/docs/peft/conceptual_guides/lora
        lora_configuration = LoraConfig(
            **lora_config,
        )
        model = get_peft_model(model, lora_configuration)

        assert check_gradients(model=model, lora_config=lora_config)

        print_trainable_parameters(
            model=model,
            quantization_config=quantization_config
        )
    return model


def prepare_trainer(
    model,
    data_dict,
    base_model_config,
    training_config
):
    if recongnize_t5(base_model_name=base_model_config['pretrained_model_name_or_path']):
        training_config.pop('group_by_length')
        training_arguments = Seq2SeqTrainingArguments(
            **training_config,
            predict_with_generate=True,
        )
    else:
        training_arguments = TrainingArguments(
            **training_config,
        )
    data_collator = get_data_collector(base_model_config=base_model_config)

    metrics, compute_metrics = get_metrics_evaluators(
        base_model_config=base_model_config
    )

    if base_model_config['problem_type'] == 'generative_classification':
        tokenizer, _ = get_tokenizer(
            base_model_config=base_model_config,
            padding_side='right',
        )
        
        if recongnize_t5(base_model_name=base_model_config['pretrained_model_name_or_path']):
            from transformers import DataCollatorForSeq2Seq
            data_collator = DataCollatorForSeq2Seq(tokenizer)
            data_dict['train'] = data_dict['train'].map(lambda x: tokenizer(x["full_prompt"], truncation=True, padding="max_length", max_length=1024))

            data_dict['train'] = data_dict['train'].map(lambda y: 
                {"labels": tokenizer(y["text_labels"], truncation=True, padding="max_length", max_length=1024)['input_ids']}
            )

            data_dict['validation'] = data_dict['validation'].map(lambda x: tokenizer(x["full_prompt"], 
                truncation=True, padding="max_length", max_length=1024))
            data_dict['validation'] = data_dict['validation'].map(lambda y: 
                {"labels": tokenizer(y["text_labels"], truncation=True, padding="max_length", max_length=1024)['input_ids']}
            )

            trainer = Seq2SeqTrainer(
                model=model,
                train_dataset=data_dict['train'],
                eval_dataset=data_dict['validation'],
                tokenizer=tokenizer,
                args=training_arguments,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
        else:
            trainer = SFTTrainer(
                model=model,
                train_dataset=data_dict['train'],
                eval_dataset=data_dict['validation'],
                formatting_func=formatting_prompts_func,
                tokenizer=tokenizer,
                max_seq_length=base_model_config['max_seq_length'],
                args=training_arguments,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
        # I don't know how else to force trainer to output
        # evaluation loss.
        trainer.can_return_loss = lambda **_: True
    else:
        trainer = Trainer(
            model=model,
            train_dataset=data_dict['train'],
            eval_dataset=data_dict['validation'],
            args=training_arguments,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

    return trainer


def train(
    model,
    data_dict,
    base_model_config,
    training_config,
    quantization_config=None,
    lora_config=None,
):

    model = prepare_model(
        model=model,
        quantization_config=quantization_config,
        lora_config=lora_config,
    )

    trainer = prepare_trainer(
        model=model,
        data_dict=data_dict,
        base_model_config=base_model_config,
        training_config=training_config
    )

    trainer.train()
    trainer.save_model()
    config_file = 'real-config.ini'
    if not Path(config_file).is_file():
        config_file = 'config.ini'
    shutil.copy2(config_file, training_config['output_dir'])
    if not base_model_config['problem_type'] == 'generative_classification':
        scores = trainer.evaluate(eval_dataset=data_dict['test'])
        print('Test_scores', scores)
    else:
        main_test(load_tuned=True)
    return
