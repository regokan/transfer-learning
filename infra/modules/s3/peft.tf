resource "aws_s3_bucket" "peft_learning" {
  bucket = "peft-learning"

  tags = {
    Name        = "peft_learning"
    Project     = "transfer-learning"
    Environment = "Production"
  }
}

resource "aws_s3_bucket_ownership_controls" "peft_learning_ownership_controls" {
  bucket = aws_s3_bucket.peft_learning.id
  rule {
    object_ownership = "BucketOwnerPreferred"
  }
}

resource "aws_s3_bucket_acl" "peft_learning_acl" {
  depends_on = [aws_s3_bucket_ownership_controls.peft_learning_ownership_controls]

  bucket = aws_s3_bucket.peft_learning.id
  acl    = "private"
}

resource "aws_s3_bucket_versioning" "peft_learning_versioning" {
  bucket = aws_s3_bucket.peft_learning.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "peft_learning_server_side_encryption_configuration" {
  bucket = aws_s3_bucket.peft_learning.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
