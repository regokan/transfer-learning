"""This file contains code for training a model using PEFT on SageMaker"""

import json
import sagemaker
from sagemaker.pytorch import PyTorch, PyTorchModel
from parameter_efficient.inference import model_fn, input_fn, predict_fn, output_fn


def sagemaker_train_peft(sagemaker_role, sagemaker_bucket, args=None):
    """
    Train the model using the provided datasets and tokenizer on SageMaker.
    """
    # Set up the SageMaker session
    _ = sagemaker.Session()

    # Define the PyTorch Estimator (or use TensorFlow, HuggingFace, etc.)
    if args.use_spot_instances:
        pytorch_estimator = PyTorch(
            source_dir="parameter_efficient",
            entry_point="train.py",
            role=sagemaker_role,
            instance_count=args.instance_count,
            instance_type=args.instance_type,
            framework_version="2.3.0",
            py_version="py311",
            hyperparameters={
                "epochs": args.epochs,
                "batch-size": args.batch_size,
                "learning-rate": args.learning_rate,
                "weight-decay": args.weight_decay,
                "logging-steps": args.logging_steps,
                "num-labels": args.num_labels,
                "tokenizer-model-name": args.tokenizer_model_name,
                "model-name": args.model_name,
                "lora-r": args.lora_r,
                "lora-alpha": args.lora_alpha,
                "lora-dropout": args.lora_dropout,
                "lora-task-type": args.lora_task_type,
            },
            output_path=f"s3://{sagemaker_bucket}/output",  # Output path for models
            checkpoint_s3_uri=f"s3://{sagemaker_bucket}/checkpoints",  # Optional checkpointing
            dependencies=["requirements.txt"],
            # Spot instance settings
            use_spot_instances=args.use_spot_instances,  # Enable spot instances
            max_wait=(
                86400
            ),  # Max wait time for Spot Instances (in seconds) - here 24 hours
            max_run=(
                43200
            ),  # Maximum runtime (in seconds) - here 12 hours for training
        )
    else:
        pytorch_estimator = PyTorch(
            source_dir="parameter_efficient",
            entry_point="train.py",
            role=sagemaker_role,
            instance_count=args.instance_count,
            instance_type=args.instance_type,
            framework_version="2.3.0",
            py_version="py311",
            hyperparameters={
                "epochs": args.epochs,
                "batch-size": args.batch_size,
                "learning-rate": args.learning_rate,
                "weight-decay": args.weight_decay,
                "logging-steps": args.logging_steps,
                "num-labels": args.num_labels,
                "tokenizer-model-name": args.tokenizer_model_name,
                "model-name": args.model_name,
                "lora-r": args.lora_r,
                "lora-alpha": args.lora_alpha,
                "lora-dropout": args.lora_dropout,
                "lora-task-type": args.lora_task_type,
            },
            dependencies=["requirements.txt"],
            output_path=f"s3://{sagemaker_bucket}/output",  # Output path for models
            checkpoint_s3_uri=f"s3://{sagemaker_bucket}/checkpoints",  # Optional checkpointing
        )

    # Start the training job
    if args.data_dir:
        pytorch_estimator.fit({"train": f"{args.data_dir}"}, wait=False)
    else:
        # Using HuggingFace Datasets
        pytorch_estimator.fit(wait=False)


# pylint: disable=line-too-long
def sagemaker_evaluate_local_peft(sagemaker_bucket, args=None):

    s3_output_path = f"s3://{sagemaker_bucket}/output/"  # S3 path for the output
    model_dir = (
        f"{s3_output_path}{args.sagemaker_pytorch_training_version}"
        "/output/model.tar.gz"
    )

    # Step 1: Load the model
    model, tokenizer = model_fn(model_dir)

    # Step 2: Simulate CSV input
    test_csv = '''"id","qid1","qid2","question1","question2","is_duplicate"
    "0","1","2","What is the step by step guide to invest in share market in india?","What is the step by step guide to invest in share market?","0"
    "1","3","4","What is the story of Kohinoor (Koh-i-Noor) Diamond?","What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?","0"
    "2","5","6","How can I increase the speed of my internet connection while using a VPN?","How can Internet speed be increased by hacking through DNS?","0"
    "3","7","8","Why am I mentally very lonely? How can I solve it?","Find the remainder when [math]23^{24}[/math] is divided by 24,23?","0"
    "4","9","10","Which one dissolve in water quikly sugar, salt, methane and carbon di oxide?","Which fish would survive in salt water?","0"
    "5","11","12","Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?","I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me?","1"
    "6","13","14","Should I buy tiago?","What keeps childern active and far from phone and video games?","0"
    "7","15","16","How can I be a good geologist?","What should I do to be a great geologist?","1"'''

    # Step 3: Use input_fn to parse the input
    input_data = input_fn(test_csv, "text/csv")

    # Step 4: Make predictions using the model
    predictions = predict_fn(input_data, (model, tokenizer))

    # Step 5: Format the output using output_fn
    output = output_fn(predictions)

    # Print the results
    print(output)

    # Save Json output to a file
    with open("data/output.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(output), f)


def sagemaker_evaluate_peft(sagemaker_role, sagemaker_bucket, args=None):
    """
    Evaluate the trained model using the provided datasets on SageMaker.
    """
    s3_input_path = (
        f"s3://{sagemaker_bucket}/input/evaluate.csv"  # S3 path to test dataset
    )
    s3_output_path = f"s3://{sagemaker_bucket}/output/"  # S3 path for the output
    s3_model_path = (
        f"{s3_output_path}{args.sagemaker_pytorch_training_version}"
        "/output/model.tar.gz"
    )  # S3 path to model

    # Define the model in SageMaker
    pytorch_model = PyTorchModel(
        source_dir="parameter_efficient",
        model_data=s3_model_path,
        role=sagemaker_role,
        entry_point="inference.py",
        framework_version="2.3.0",
        py_version="py311",
        dependencies=["requirements.txt"],
    )

    # Create a Transformer object for batch transform
    transformer = pytorch_model.transformer(
        instance_count=1,
        instance_type=args.evaluate_instance_type,
        output_path=s3_output_path,
        # How the output is assembled (default: None).
        # Valid values: 'Line' or 'None'.
        assemble_with="Line",
        # The strategy used to decide how to batch records in a single request (default: None).
        # # Valid values: 'MultiRecord' and 'SingleRecord'.
        strategy="MultiRecord",
        env={
            "EPOCHS": str(args.epochs),
            "BATCH_SIZE": str(args.batch_size),
            "LEARNING_RATE": str(args.learning_rate),
            "WEIGHT_DECAY": str(args.weight_decay),
            "LOGGING_STEPS": str(args.logging_steps),
            "NUM_LABELS": str(args.num_labels),
            "TOKENIZER_MODEL_NAME": args.tokenizer_model_name,
            "MODEL_NAME": args.model_name,
            "LORA_R": str(args.lora_r),
            "LORA_ALPHA": str(args.lora_alpha),
            "LORA_DROPOUT": str(args.lora_dropout),
            "LORA_TASK_TYPE": args.lora_task_type,
            "SAGEMAKER_PYTORCH_TRAINING_VERSION": args.sagemaker_pytorch_training_version,
            "SAGEMAKER_BUCKET": sagemaker_bucket,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )

    # Perform batch transform (inference) on the QQP test dataset
    transformer.transform(
        data=s3_input_path,  # Input path (S3 location for the Kaggle test data)
        content_type="text/csv",  # Adjust the content type based on the data format
        split_type="Line",  # Assuming the test data is line-delimited
        wait=False,
    )
