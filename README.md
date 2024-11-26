# Transfer Learning with Parameter-Efficient Fine-Tuning (PEFT) on AWS

This project demonstrates the use of transfer learning with a focus on Parameter-Efficient Fine-Tuning (PEFT) using models from Hugging Face (e.g., DeBERTa-v3). It is designed to facilitate efficient training and evaluation on AWS SageMaker, leveraging infrastructure as code for seamless setup and detailed cost-performance analysis for optimization.

---

## [Infrastructure](./infra/)

The infrastructure for this project is defined using Terraform, which automates the setup of AWS resources required for training and evaluation. Key components include:

- **S3 Buckets**: Provisioned using the `s3` module in the `infra/modules/s3/` directory. The `peft.tf` file defines the S3 bucket used for storing model artifacts and datasets. The outputs and variables for the S3 bucket are configured in `output.tf` and `variables.tf`, respectively.
- **IAM Roles**: Created using the `iam` module in the `infra/modules/iam/` directory. The `sagemaker_execution_role.tf` file defines the IAM role with the necessary permissions for SageMaker to access S3 and other AWS services. Outputs and variables for IAM roles are defined in `output.tf` and `variables.tf`.

- **Terraform Configuration**:
  - The main configuration is in `main.tf`, which orchestrates the setup of resources.
  - The `config.tf` file defines Terraform backend and provider configurations.
  - Supporting scripts like `apply.local`, `plan.local`, and `destroy.local` allow for quick deployment, planning, and teardown of resources locally.

This modular setup ensures reusability and scalability of resources while maintaining security and efficiency.

---

## [Analysis](./analysis/)

The `overview.ipynb` notebook in the `analysis` folder provides a comprehensive breakdown of the training process, leveraging Parameter-Efficient Fine-Tuning (PEFT) with the DeBERTa-v3 model and cost analysis on AWS SageMaker.

### **Model Analysis**

- **Pre-trained Model**: The `microsoft/deberta-v3-base` model is configured for binary sequence classification (e.g., QQP dataset).
- **LoRA Configuration**:
  - Uses Low-Rank Adaptation (LoRA) to inject trainable parameters into specific attention layers (`query_proj`, `key_proj`, and `value_proj`).
  - Key parameters: `r=4`, `lora_alpha=16`, `lora_dropout=0.1`.
  - Focused on minimizing computational overhead by freezing most of the model and only training select layers.
- **Trainable Parameters**:
  - After applying LoRA, the number of trainable parameters is reduced to approximately **741,124**, making training lightweight and efficient.

### **Dataset and Preprocessing**

- **Dataset**: The Quora Question Pairs (QQP) dataset from the GLUE benchmark is loaded and preprocessed.
- **Preprocessing**:
  - Tokenizes question pairs and converts them into `input_ids` and `attention_mask` tensors.
  - Moves the tokenized dataset to the appropriate device (GPU/CPU) for training.

### **Cost Analysis**

The notebook includes a detailed cost analysis for training on AWS SageMaker using two instance types, **`ml.p3.2xlarge`** and **`ml.g4dn.xlarge`**. Both **on-demand** and **spot instance** pricing are considered.

#### **Training Parameters**:

- **Trainable Parameters**: 741,124
- **Training Rows**: 363,846
- **Epochs**: 5
- **Batch Size**: 32
- **Total Steps**: 56,850

#### **Time and Cost Estimates**:

| **Instance Type** | **Total Time (5 epochs)** | **Cost (On-Demand)** | **Cost (Spot)** |
| ----------------- | ------------------------- | -------------------- | --------------- |
| `ml.p3.2xlarge`   | 3.16 hours                | \$12.08              | \$3.63          |
| `ml.g4dn.xlarge`  | 5.53 hours                | \$4.16               | \$1.25          |

- **Time per Step**:
  - `ml.p3.2xlarge`: 0.20 seconds/step.
  - `ml.g4dn.xlarge`: 0.35 seconds/step.

#### **Conclusion**:

- For **cost efficiency**, `ml.g4dn.xlarge` on **spot instances** is the best choice, with a total cost of **\$1.25**.
- For **speed**, `ml.p3.2xlarge` on **spot instances** completes training in **3.16 hours** at a cost of **\$3.63**.

---

## [main.py](./main.py): Main Execution Script

The `main.py` script serves as the entry point for executing various tasks in the project. It integrates with AWS SageMaker for training and evaluation while also supporting local testing.

### **Key Features**:

1. **Environment Variables**:

   - Requires the environment variables `PEFT_BUKET` (S3 bucket for storing artifacts) and `PEFT_SAGEMAKER_EXECUTION_ROLE` (IAM role for SageMaker) to be set.
   - These can be set using `.env` file for better configuration management in local development.

2. **Supported Execution Types**:

   - **`train`**: Starts a SageMaker training job.
   - **`evaluate`**: Runs evaluation of the trained model on SageMaker.
   - **`evaluate-local`**: Performs evaluation of the trained model locally.

3. **Command-Line Flags**:
   The script uses the following command-line flags for customization:

   - **General Parameters**:

     - `--type`: Task type (`train`, `evaluate`, or `evaluate-local`).
     - `--epochs`: Number of epochs for training (default: `1`).
     - `--batch-size`: Batch size for training (default: `32`).
     - `--learning-rate`: Learning rate for optimizer (default: `1e-3`).
     - `--num-labels`: Number of labels for classification (default: `2`).
     - `--weight-decay`: Weight decay for optimizer (default: `0.01`).
     - `--logging-steps`: Log every X updates steps (default: `50`).

   - **Model Parameters**:

     - `--tokenizer-model-name`: Hugging Face tokenizer model name (default: `microsoft/deberta-v3-base`).
     - `--model-name`: Hugging Face model name (default: `microsoft/deberta-v3-base`).
     - `--lora-r`: Rank of the LoRA adaptation matrix (default: `4`).
     - `--lora-alpha`: Scaling factor (alpha) for LoRA (default: `16`).
     - `--lora-dropout`: Dropout rate for LoRA layers (default: `0.1`).
     - `--lora-bias`: Bias type for LoRA layers (`none`, default: `none`).
     - `--lora-task-type`: Task type for LoRA configuration (`SEQ_CLS`, default: `SEQ_CLS`).

   - **Data Parameters**:

     - `--data-dir`: S3 directory containing the dataset (default: QQP from Hugging Face).

   - **SageMaker Parameters**:
     - `--use-spot-instances`: Whether to use spot instances (default: `False`).
     - `--instance-count`: Number of EC2 instances to use (default: `1`).
     - `--instance-type`: EC2 instance type for training (default: `ml.p3.2xlarge`).
     - `--evaluate-instance-type`: EC2 instance type for evaluation (default: `ml.g4dn.xlarge`).
     - `--sagemaker-pytorch-training-version`: SageMaker PyTorch training container version (default: `pytorch-training-2024-09-19-09-17-17-352`).
     - `--sagemaker-bucket`: S3 bucket for SageMaker (default: `peft-learning`).

### **How to Use**:

1. **Set Up Environment**:

   - Ensure the environment variables `PEFT_BUKET` and `PEFT_SAGEMAKER_EXECUTION_ROLE` are defined.

2. **Run the Script**:

   - Use the `--type` flag to specify the task type:
     - **Train**:
       ```bash
       python main.py --type train --epochs 3 --batch-size 16 --learning-rate 1e-4
       ```
     - **Evaluate**:
       ```bash
       python main.py --type evaluate --model-name microsoft/deberta-v3-base
       ```
     - **Local Evaluation**:
       ```bash
       python main.py --type evaluate-local --data-dir s3://my-bucket/data
       ```

3. **Example**:
   - Start a SageMaker training job with custom parameters:
     ```bash
     python main.py --type train --epochs 5 --batch-size 64 --lora-r 8 --instance-type ml.g4dn.xlarge
     ```

This script provides a seamless way to manage training and evaluation workflows, enabling efficient use of AWS SageMaker for large-scale experiments.

## [utils](./parameter_efficient/utils/): Utility Functions

The `parameter_efficient` folder contains utility functions and scripts that support the main operations of the project:

- **`parser.py`**: Handles command-line argument parsing, allowing users to specify hyperparameters and other configurations for the training job.
- **`torch.py`**: Provides helper functions related to PyTorch, such as determining the best available device (CPU, CUDA, or MPS).
- **`local_test.py`**: Includes functions for downloading and extracting models locally, useful for testing without SageMaker.

## Fine-Tuning with PEFT

The core of the project is the fine-tuning of the DeBERTa-v3 model using PEFT. This approach focuses on:

- **LoRA Configuration**: Utilizes Low-Rank Adaptation (LoRA) to reduce the number of trainable parameters, making the model more efficient for fine-tuning.
- **Model Training**: The `train.py` script orchestrates the training process, leveraging the utilities and configurations defined in the project.
- **Hyperparameter Tuning**: Users can adjust various hyperparameters through command-line flags or environment variables, as defined in `parser.py`.

---

## Getting Started

To begin using this project, follow these steps:

1. **Set Up Prerequisites**:

   - Install Terraform for provisioning infrastructure.
   - Ensure you have AWS CLI configured with appropriate credentials.

2. **Configure Environment**:

   - Create a `.env` file with the following variables:
     ```env
     PEFT_BUKET=<your-s3-bucket-name>
     PEFT_SAGEMAKER_EXECUTION_ROLE=<your-sagemaker-execution-role-arn>
     ```
   - Replace `<your-s3-bucket-name>` and `<your-sagemaker-execution-role-arn>` with your actual AWS resource details.

3. **Provision Infrastructure**:

   - Navigate to the `infra` directory and run:
     ```bash
     terraform init
     terraform apply
     ```
   - This sets up the S3 bucket and IAM role required for training and evaluation on SageMaker.

4. **Run Tasks**:

   - Use the `main.py` script to train or evaluate the model:

---

## Conclusion

This project provides a robust framework for implementing Parameter-Efficient Fine-Tuning (PEFT) with transfer learning on AWS SageMaker. The modular infrastructure, combined with flexible execution scripts, simplifies training and evaluation workflows while optimizing resource utilization.

### Key Highlights:

- **Scalable Infrastructure**: Automated setup using Terraform ensures easy provisioning of AWS resources.
- **Flexible Training**: LoRA-based PEFT enables efficient fine-tuning, minimizing computational costs without sacrificing performance.
- **Cost Optimization**: Detailed cost analysis in the `analysis` folder helps you make informed decisions about instance types and pricing models.
- **Ease of Use**: The `main.py` script and well-documented flags make it simple to execute various tasks, from training to evaluation.

By combining efficient fine-tuning techniques with cloud-based training capabilities, this project offers a comprehensive solution for building scalable machine learning models. It is designed to cater to both experimentation and production-level deployment.

For further assistance or contributions, please [create a GitHub issue](https://github.com/regokan/transfer-learning/issues/new/choose) or contact the maintainers.
