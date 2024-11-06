# iam.tf
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "sagemaker_execution_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "sagemaker.amazonaws.com"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name        = "peft"
    Project     = "transfer-learning"
    Environment = "Production"
  }
}

resource "aws_iam_role_policy_attachment" "sagemaker_execution_role_sagemaker_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_policy" "s3_access" {
    name = "s3_access"

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = [
                    "s3:*",
                ]
                Effect   = "Allow"
                Resource = [
                    var.aws_s3_bucket_peft_arn,
                    "${var.aws_s3_bucket_peft_arn}/*",
                ]
            }
        ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_execution_role_s3_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = aws_iam_policy.s3_access.arn
}
