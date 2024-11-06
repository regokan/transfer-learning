output "aws_s3_bucket_peft_bucket" {
  value = aws_s3_bucket.peft_learning.bucket
}

output "aws_s3_bucket_peft_arn" {
  value = aws_s3_bucket.peft_learning.arn
}
