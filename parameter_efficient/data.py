"""This file contains code for loading and preprocessing dataset for fine-tuning"""

import os
import pandas as pd
from datasets import load_dataset, Dataset


def load_and_prepare_dataset(data_dir=None):
    """
    Load and preprocess the dataset.

    If data_dir is provided (SageMaker case), load the dataset from local CSV files.
    Otherwise, load the QQP dataset from Hugging Face Datasets.
    """
    if data_dir:
        # SageMaker case: load from CSV files located in the mounted S3 directory
        train_file = os.path.join(data_dir, "train.csv")
        validation_file = os.path.join(data_dir, "validation.csv")

        # Load the dataset using pandas and convert it to Hugging Face Dataset format
        train_df = pd.read_csv(train_file)
        validation_df = pd.read_csv(validation_file)

        # Convert to Hugging Face Dataset
        ds = Dataset.from_pandas(train_df), Dataset.from_pandas(validation_df)
        ds = {"train": ds[0], "validation": ds[1]}
    else:
        # Local case: load directly from Hugging Face datasets
        ds = load_dataset("glue", "qqp")

    # Filter out examples with missing labels
    ds["train"] = ds["train"].filter(lambda example: example["label"] != -1)
    ds["validation"] = ds["validation"].filter(lambda example: example["label"] != -1)

    return ds


def preprocess_data(tokenizer, ds, device):
    """
    Tokenize the dataset using the provided tokenizer and move the data to the specified device.
    This function skips empty splits and applies tokenization
    and device movement to non-empty splits.
    """

    def preprocess_function(examples):
        # Tokenize the input question pairs
        tokenized = tokenizer(
            examples["question1"],
            examples["question2"],
            truncation=True,
            padding=True,
            max_length=128,
        )
        # Include the labels in the output dictionary
        tokenized["labels"] = examples["label"]
        return tokenized

    # Apply the preprocessing function to the dataset
    # (this adds the tokenized columns like input_ids)
    tokenized_ds = {}
    for split in ds:
        if len(ds[split]) > 0:
            # Tokenize non-empty splits
            tokenized_ds[split] = ds[split].map(preprocess_function, batched=True)
        else:
            # Skip tokenization for empty splits (e.g., test set)
            tokenized_ds[split] = ds[split]

    # Now we can set the format to include the tokenized columns and labels
    for split, dataset in tokenized_ds.items():
        if len(dataset) > 0:  # Ensure the split is not empty before formatting
            dataset.set_format(
                type="torch", columns=["input_ids", "attention_mask", "labels"]
            )

    # Move each dataset (train/validation) to the device
    # (optional, but useful for GPU/CPU compatibility)
    tokenized_ds["train"] = tokenized_ds["train"].with_format("torch", device=device)
    tokenized_ds["validation"] = tokenized_ds["validation"].with_format(
        "torch", device=device
    )

    return tokenized_ds
