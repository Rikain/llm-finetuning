from model.load import load_finetuned
from src.utils import prepare_configuration, seed_everything, \
      get_data_collector, get_metrics_evaluators


from transformers import TrainingArguments, Trainer


def main():
    seed, base_model_config, lora_config, quantization_config, \
        training_config, data_dict, pad_token_id = prepare_configuration()
    seed_everything(seed)
    model = load_finetuned(
        base_model_config=base_model_config,
        model_checkpoint_path=training_config['output_dir'],
        quantization_config=quantization_config,
        pad_token_id=pad_token_id
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
    scores = trainer.evaluate(eval_dataset=data_dict['test'])
    print('Test_scores', scores)
    return


if __name__ == '__main__':
    main()
