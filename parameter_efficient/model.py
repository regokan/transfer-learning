"""This file contains code for creating a model using LoRA PEFT"""

from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model


def create_peft_model(args):
    """
    Create a model with LoRA configuration using PEFT.
    """
    # Loading the model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
    )

    # Freezing the base model's parameters
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Configuring LoRA parameters
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        # target_modules=['query_proj', 'key_proj', 'value_proj'],
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type=args.lora_task_type,
    )

    # Applying LoRA to the model
    model = get_peft_model(model, lora_config)

    # Training pooler and classifier layers as well
    for param in model.pooler.parameters():
        param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model
