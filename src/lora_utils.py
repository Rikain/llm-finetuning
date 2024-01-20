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


def get_classification_head_name(model):
    classification_head = None
    if hasattr(model, 'classification_head'):
        classification_head = 'classification_head'
    elif hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'score'):
        classification_head = 'score'
    assert classification_head is not None
    return classification_head


def check_gradients(model, lora_config):
    classification_head_name = get_classification_head_name(model)
    if lora_config['task_type'] == 'SEQ_CLS':
        if classification_head_name == 'score':
            assert not model.base_model.model.score.original_module.weight.requires_grad
            assert model.base_model.model.score.modules_to_save["default"].weight.requires_grad
            assert model.base_model.model.score.active_adapter == 'default'
            assert (torch.allclose(
                model.base_model.model.score.modules_to_save["default"].weight,
                model.base_model.model.score.original_module.weight
            ))
        else:
            # TO DO FOR T5 model
            return True
    return True


def add_modules_to_save(model, lora_config):
    modules_to_save = None
    if lora_config['task_type'] == 'SEQ_CLS':
        modules_to_save = []
        classification_head = get_classification_head_name(model)
        if classification_head == 'classification_head':
            for module_name, module in model.classification_head.named_modules():
                if module_name not in lora_config['target_modules']:
                    if isinstance(module, torch.nn.Linear):
                        modules_to_save.append(module_name)
        else:
            modules_to_save.append(classification_head)
    lora_config['modules_to_save'] = modules_to_save
    print('modules_to_save', modules_to_save)
    return lora_config


def find_all_linear_names(model, quantization_config):
    """
    Find modules to apply LoRA to.
    """
    if hasattr(model, 'classification_head'):
        # (Expected for For T5ForSequenceClassification)
        possible_modules_to_save = ['dense', 'out_proj']
    elif hasattr(model, 'score'):
        # (Expected for LlamaForSequenceClassification)
        possible_modules_to_save = ['score']
    else:
        # Propably a class not accounted for.
        raise Exception('Propably a class not accounted for.')
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
    for lm_head in possible_modules_to_save:
        if lm_head in lora_module_names:
            lora_module_names.remove(lm_head)
    print('Modules to apply LORA to:', list(lora_module_names))
    return list(lora_module_names)
