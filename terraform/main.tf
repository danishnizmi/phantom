# Phantom Tech Influencer - Terraform Configuration
# Infrastructure as Code for GCP deployment

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }

  # Uncomment and configure for remote state (recommended for production)
  # backend "gcs" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "phantom-influencer"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# ============================================================================
# Enable Required APIs
# ============================================================================

resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "cloudscheduler.googleapis.com",
    "secretmanager.googleapis.com",
    "firestore.googleapis.com",
    "artifactregistry.googleapis.com",
    "aiplatform.googleapis.com",
    "cloudbuild.googleapis.com",
  ])

  service            = each.key
  disable_on_destroy = false
}

# ============================================================================
# Service Account
# ============================================================================

resource "google_service_account" "phantom_sa" {
  count        = var.service_account_email == "" ? 1 : 0
  account_id   = "phantom-influencer-sa"
  display_name = "Phantom Influencer Service Account"
  description  = "Service account for the Phantom Tech Influencer Cloud Run Job"
}

locals {
  service_account_email = var.service_account_email != "" ? var.service_account_email : google_service_account.phantom_sa[0].email
}

# IAM roles for the service account
resource "google_project_iam_member" "phantom_sa_roles" {
  for_each = toset([
    "roles/secretmanager.secretAccessor",
    "roles/datastore.user",
    "roles/aiplatform.user",
    "roles/storage.objectViewer",
    "roles/logging.logWriter",
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${local.service_account_email}"
}

# ============================================================================
# Artifact Registry (Container Repository)
# ============================================================================

resource "google_artifact_registry_repository" "phantom_repo" {
  location      = var.region
  repository_id = "phantom-influencer"
  description   = "Docker repository for Phantom Influencer"
  format        = "DOCKER"
  labels        = var.labels

  depends_on = [google_project_service.required_apis["artifactregistry.googleapis.com"]]
}

# ============================================================================
# Firestore Database
# ============================================================================

resource "google_firestore_database" "phantom_db" {
  project     = var.project_id
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"

  # Prevent accidental deletion
  deletion_policy = "DELETE"

  depends_on = [google_project_service.required_apis["firestore.googleapis.com"]]

  lifecycle {
    # Firestore database can only be created once per project
    ignore_changes = [name, location_id, type]
  }
}

# ============================================================================
# Cloud Run Job
# ============================================================================

resource "google_cloud_run_v2_job" "phantom_job" {
  name     = var.job_name
  location = var.region
  labels   = var.labels

  template {
    template {
      service_account = local.service_account_email
      timeout         = "${var.timeout_seconds}s"
      max_retries     = var.max_retries

      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/phantom-influencer/${var.image_name}:latest"

        resources {
          limits = {
            cpu    = var.cpu_limit
            memory = var.memory_limit
          }
        }

        env {
          name  = "PROJECT_ID"
          value = var.project_id
        }

        env {
          name  = "REGION"
          value = var.region
        }

        env {
          name  = "TIMEZONE"
          value = var.timezone
        }

        env {
          name  = "BUDGET_MODE"
          value = tostring(var.budget_mode)
        }

        env {
          name  = "CURRENCY"
          value = "AUD"
        }
      }
    }
  }

  depends_on = [
    google_project_service.required_apis["run.googleapis.com"],
    google_artifact_registry_repository.phantom_repo,
    google_project_iam_member.phantom_sa_roles,
  ]

  lifecycle {
    # Don't recreate the job if only the image tag changes
    ignore_changes = [
      template[0].template[0].containers[0].image,
    ]
  }
}

# ============================================================================
# Cloud Scheduler Triggers
# ============================================================================

resource "google_cloud_scheduler_job" "phantom_triggers" {
  count = length(var.scheduler_triggers)

  name        = "${var.job_name}-trigger-${count.index + 1}"
  description = "Trigger ${count.index + 1} for Phantom Influencer (AWST)"
  schedule    = var.scheduler_triggers[count.index]
  time_zone   = var.timezone

  http_target {
    http_method = "POST"
    uri         = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/${var.job_name}:run"

    oauth_token {
      service_account_email = local.service_account_email
      scope                 = "https://www.googleapis.com/auth/cloud-platform"
    }
  }

  retry_config {
    retry_count          = 1
    min_backoff_duration = "5s"
    max_backoff_duration = "60s"
  }

  depends_on = [
    google_project_service.required_apis["cloudscheduler.googleapis.com"],
    google_cloud_run_v2_job.phantom_job,
  ]
}

# ============================================================================
# Secret Manager Secrets (structure only - values set manually)
# ============================================================================

resource "google_secret_manager_secret" "twitter_secrets" {
  for_each = toset([
    "TWITTER_CONSUMER_KEY",
    "TWITTER_CONSUMER_SECRET",
    "TWITTER_ACCESS_TOKEN",
    "TWITTER_ACCESS_TOKEN_SECRET",
    "TWITTER_BEARER_TOKEN",
  ])

  secret_id = each.key
  labels    = var.labels

  replication {
    auto {}
  }

  depends_on = [google_project_service.required_apis["secretmanager.googleapis.com"]]
}

# Grant service account access to secrets
resource "google_secret_manager_secret_iam_member" "secret_access" {
  for_each = google_secret_manager_secret.twitter_secrets

  secret_id = each.value.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${local.service_account_email}"
}
