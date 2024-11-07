"""This file contains code for training a model using PEFT"""

import os
import torch
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
import numpy as np

from data import load_and_prepare_dataset, preprocess_data
from model import create_peft_model
from utils import parse_flags, get_device


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for the model.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def train_model(model, tokenized_ds, tokenizer, output_dir, model_dir, device, args):
    """
    Train the model using the provided datasets and tokenizer.
    """
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,  # Handled by SageMaker
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=os.path.join(
            output_dir, "logs"
        ),  # Log directory inside the output directory
        logging_steps=args.logging_steps,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    # Save the final model to the output directory (SageMaker will store it in S3)
    trainer.save_model(output_dir)

    with open(os.path.join(model_dir, "model.pth"), "wb") as f:
        torch.save(model.state_dict(), f)

    model.save_pretrained(model_dir)


def train_peft():
    """
    Main function to execute the fine-tuning pipeline.
    """

    # Parse Hyperparrametrs from sagemaker or command line (local)
    args = parse_flags()

    # Get the best available device (CUDA, MPS, or CPU)
    device = get_device()

    # SageMaker specific: retrieve environment variables
    train_data_dir = os.environ.get("SM_CHANNEL_TRAIN")
    model_dir = os.environ.get(
        "SM_MODEL_DIR", "/opt/ml/model"
    )  # Where the model should be saved
    output_dir = os.environ.get(
        "SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"
    )  # Output artifacts

    # Load and prepare the dataset
    ds = load_and_prepare_dataset(data_dir=train_data_dir)
    # Ensure it can load from the train data directory

    # Initialize the tokenizer for DeBERTa-v3
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model_name)

    # Preprocess the dataset
    tokenized_ds = preprocess_data(tokenizer, ds, device)

    # Create the model with LoRA configuration
    model = create_peft_model(args)

    # Train the model
    train_model(model, tokenized_ds, tokenizer, output_dir, model_dir, device, args)


if __name__ == "__main__":
    # Detect whether running on SageMaker or locally, and run training
    train_peft()
