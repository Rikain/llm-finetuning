import bitsandbytes as bnb
import torch


def print_trainable_parameters(model, quantization_config):
    """
    Prints the number of trainable parameters in the model.
    """
    use_4bit = 'load_in_4bit' in quantization_config \
        and quantization_config['load_in_4bit']
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    if use_4bit:
        trainable_params /= 2

    print(
        f"All Parameters: {int(all_param):,d} || Trainable Parameters: {int(trainable_params):,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(model, quantization_config):
    """
    Find modules to apply LoRA to.
    """
    if ('load_in_4bit' in quantization_config and
        quantization_config['load_in_4bit']) \
            or ('load_in_8bit' in quantization_config and
                quantization_config['load_in_8bit']):
        classes = [bnb.nn.Linear4bit, bnb.nn.Linear8bitLt]
    else:
        classes = [torch.nn.Linear]
    lora_module_names = set()
    for name, module in model.named_modules():
        if any([isinstance(module, c) for c in classes]):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    if 'score' in lora_module_names:
        lora_module_names.remove('score')
    print('Modules to apply LORA to:', list(lora_module_names))
    return list(lora_module_names)
