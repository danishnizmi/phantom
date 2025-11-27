# Phantom Tech Influencer - Terraform Outputs

output "project_id" {
  description = "GCP Project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP Region"
  value       = var.region
}

output "service_account_email" {
  description = "Service account email used by the Cloud Run Job"
  value       = local.service_account_email
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.phantom_repo.repository_id}"
}

output "container_image_url" {
  description = "Full container image URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.phantom_repo.repository_id}/${var.image_name}:latest"
}

output "cloud_run_job_name" {
  description = "Cloud Run Job name"
  value       = google_cloud_run_v2_job.phantom_job.name
}

output "cloud_run_job_uri" {
  description = "Cloud Run Job URI"
  value       = "https://console.cloud.google.com/run/jobs/details/${var.region}/${google_cloud_run_v2_job.phantom_job.name}?project=${var.project_id}"
}

output "scheduler_job_names" {
  description = "Cloud Scheduler job names"
  value       = [for job in google_cloud_scheduler_job.phantom_triggers : job.name]
}

output "scheduler_schedules" {
  description = "Cloud Scheduler cron expressions (AWST)"
  value       = var.scheduler_triggers
}

output "timezone" {
  description = "Configured timezone"
  value       = var.timezone
}

output "secrets_to_configure" {
  description = "Secrets that need to be configured manually"
  value = [
    "TWITTER_CONSUMER_KEY",
    "TWITTER_CONSUMER_SECRET",
    "TWITTER_ACCESS_TOKEN",
    "TWITTER_ACCESS_TOKEN_SECRET",
    "TWITTER_BEARER_TOKEN (optional)",
  ]
}

output "next_steps" {
  description = "Instructions for completing setup"
  value       = <<-EOT

    ╔══════════════════════════════════════════════════════════════════╗
    ║                    TERRAFORM APPLY COMPLETE                       ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Next Steps:                                                      ║
    ║                                                                   ║
    ║  1. Configure Twitter API secrets in Secret Manager:              ║
    ║     gcloud secrets versions add TWITTER_CONSUMER_KEY \            ║
    ║       --data-file=/path/to/key.txt                                ║
    ║                                                                   ║
    ║  2. Build and push the container image:                           ║
    ║     cd /home/user/phantom                                         ║
    ║     ./deploy.sh                                                   ║
    ║                                                                   ║
    ║  3. Test the job manually:                                        ║
    ║     gcloud run jobs execute ${google_cloud_run_v2_job.phantom_job.name} \                           ║
    ║       --region=${var.region}                                      ║
    ║                                                                   ║
    ║  4. Monitor logs:                                                 ║
    ║     gcloud logging read "resource.type=cloud_run_job" --limit=50  ║
    ╚══════════════════════════════════════════════════════════════════╝

  EOT
}
