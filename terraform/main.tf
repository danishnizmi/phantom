# Phantom Tech Influencer - Terraform Configuration
# Infrastructure as Code for GCP deployment
#
# Usage:
#   export TF_VAR_project_id="your-project-id"
#   terraform init
#   terraform apply

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ============================================================================
# Enable Required APIs (must be first)
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
    "iam.googleapis.com",
  ])

  project            = var.project_id
  service            = each.key
  disable_on_destroy = false

  timeouts {
    create = "10m"
    update = "10m"
  }
}

# ============================================================================
# Service Account
# ============================================================================

resource "google_service_account" "phantom_sa" {
  account_id   = "phantom-influencer-sa"
  display_name = "Phantom Influencer Service Account"
  description  = "Service account for the Phantom Tech Influencer Cloud Run Job"
  project      = var.project_id

  depends_on = [google_project_service.required_apis["iam.googleapis.com"]]
}

# IAM roles for the service account
resource "google_project_iam_member" "phantom_sa_roles" {
  for_each = toset([
    "roles/secretmanager.secretAccessor",
    "roles/datastore.user",
    "roles/aiplatform.user",
    "roles/storage.objectViewer",
    "roles/logging.logWriter",
    "roles/run.invoker",
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.phantom_sa.email}"

  depends_on = [google_service_account.phantom_sa]
}

# ============================================================================
# Artifact Registry (Container Repository)
# ============================================================================

resource "google_artifact_registry_repository" "phantom_repo" {
  location      = var.region
  repository_id = "phantom-influencer"
  description   = "Docker repository for Phantom Influencer"
  format        = "DOCKER"
  project       = var.project_id

  depends_on = [google_project_service.required_apis["artifactregistry.googleapis.com"]]
}

# ============================================================================
# Firestore Database - Reference existing database
# ============================================================================

# Note: The default Firestore database already exists in the project
# Using a data source to reference it instead of creating a new one
data "google_firestore_database" "phantom_db" {
  project = var.project_id
  name    = "(default)"

  depends_on = [google_project_service.required_apis["firestore.googleapis.com"]]
}

# ============================================================================
# Secret Manager - Reference EXISTING secrets (don't create new ones)
# Secrets are expected to already exist in the project
# ============================================================================

# Data sources to reference existing secrets
data "google_secret_manager_secret" "twitter_consumer_key" {
  project   = var.project_id
  secret_id = "TWITTER_CONSUMER_KEY"
  depends_on = [google_project_service.required_apis["secretmanager.googleapis.com"]]
}

data "google_secret_manager_secret" "twitter_consumer_secret" {
  project   = var.project_id
  secret_id = "TWITTER_CONSUMER_SECRET"
  depends_on = [google_project_service.required_apis["secretmanager.googleapis.com"]]
}

data "google_secret_manager_secret" "twitter_access_token" {
  project   = var.project_id
  secret_id = "TWITTER_ACCESS_TOKEN"
  depends_on = [google_project_service.required_apis["secretmanager.googleapis.com"]]
}

data "google_secret_manager_secret" "twitter_access_token_secret" {
  project   = var.project_id
  secret_id = "TWITTER_ACCESS_TOKEN_SECRET"
  depends_on = [google_project_service.required_apis["secretmanager.googleapis.com"]]
}

data "google_secret_manager_secret" "twitter_bearer_token" {
  project   = var.project_id
  secret_id = "TWITTER_BEARER_TOKEN"
  depends_on = [google_project_service.required_apis["secretmanager.googleapis.com"]]
}

# Grant service account access to existing secrets
resource "google_secret_manager_secret_iam_member" "consumer_key_access" {
  project   = var.project_id
  secret_id = data.google_secret_manager_secret.twitter_consumer_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.phantom_sa.email}"
  depends_on = [google_service_account.phantom_sa]
}

resource "google_secret_manager_secret_iam_member" "consumer_secret_access" {
  project   = var.project_id
  secret_id = data.google_secret_manager_secret.twitter_consumer_secret.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.phantom_sa.email}"
  depends_on = [google_service_account.phantom_sa]
}

resource "google_secret_manager_secret_iam_member" "access_token_access" {
  project   = var.project_id
  secret_id = data.google_secret_manager_secret.twitter_access_token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.phantom_sa.email}"
  depends_on = [google_service_account.phantom_sa]
}

resource "google_secret_manager_secret_iam_member" "access_token_secret_access" {
  project   = var.project_id
  secret_id = data.google_secret_manager_secret.twitter_access_token_secret.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.phantom_sa.email}"
  depends_on = [google_service_account.phantom_sa]
}

resource "google_secret_manager_secret_iam_member" "bearer_token_access" {
  project   = var.project_id
  secret_id = data.google_secret_manager_secret.twitter_bearer_token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.phantom_sa.email}"
  depends_on = [google_service_account.phantom_sa]
}

# ============================================================================
# Cloud Run Job
# ============================================================================

resource "google_cloud_run_v2_job" "phantom_job" {
  name     = var.job_name
  location = var.region
  project  = var.project_id

  template {
    template {
      service_account = google_service_account.phantom_sa.email
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
    data.google_firestore_database.phantom_db,
  ]

  lifecycle {
    # Don't fail if image doesn't exist yet (will be built after)
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
  project     = var.project_id
  region      = var.region

  http_target {
    http_method = "POST"
    uri         = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/${var.job_name}:run"

    oauth_token {
      service_account_email = google_service_account.phantom_sa.email
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
