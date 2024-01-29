from src.train import train
from src.model.load import get_model
from src.utils import prepare_configuration, seed_everything


def main():
    seed, base_model_config, lora_config, quantization_config, \
        training_config, data_dict, pad_token_id = prepare_configuration()
    
    seed_everything(seed)
    model = get_model(base_model_config, quantization_config, pad_token_id)

    train(
        model=model,
        data_dict=data_dict,
        base_model_config=base_model_config,
        training_config=training_config,
        quantization_config=quantization_config,
        lora_config=lora_config
    )
    return


if __name__ == '__main__':
    main()
