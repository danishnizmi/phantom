# Phantom Tech Influencer - Terraform Outputs

output "project_id" {
  description = "GCP Project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP Region"
  value       = var.region
}

output "timezone" {
  description = "Configured timezone"
  value       = var.timezone
}

output "service_account_email" {
  description = "Service account email"
  value       = google_service_account.phantom_sa.email
}

output "artifact_registry_url" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.phantom_repo.repository_id}"
}

output "container_image" {
  description = "Full container image URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.phantom_repo.repository_id}/${var.image_name}:latest"
}

output "job_name" {
  description = "Cloud Run Job name"
  value       = google_cloud_run_v2_job.phantom_job.name
}

output "scheduler_count" {
  description = "Number of scheduler triggers created"
  value       = length(google_cloud_scheduler_job.phantom_triggers)
}

output "build_command" {
  description = "Command to build and push container"
  value       = "gcloud builds submit --tag ${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.phantom_repo.repository_id}/${var.image_name} .."
}

output "execute_command" {
  description = "Command to manually run the job"
  value       = "gcloud run jobs execute ${var.job_name} --region ${var.region}"
}

output "force_post_command" {
  description = "Command to force a post (bypass scheduler)"
  value       = "gcloud run jobs execute ${var.job_name} --region ${var.region} --update-env-vars FORCE_POST=true"
}

output "logs_command" {
  description = "Command to view logs"
  value       = "gcloud logging read 'resource.type=cloud_run_job AND resource.labels.job_name=${var.job_name}' --limit=50"
}

output "secrets_to_configure" {
  description = "Secrets that need values added"
  value = [
    "TWITTER_CONSUMER_KEY",
    "TWITTER_CONSUMER_SECRET",
    "TWITTER_ACCESS_TOKEN",
    "TWITTER_ACCESS_TOKEN_SECRET",
    "TWITTER_BEARER_TOKEN (optional)",
  ]
}
