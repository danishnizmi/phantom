#!/bin/bash
# ============================================================================
# Phantom Tech Influencer - One-Command Setup
# ============================================================================
#
# Usage:
#   ./setup.sh YOUR_PROJECT_ID
#
# Or if you have gcloud configured:
#   ./setup.sh
#
# This script will:
#   1. Create all GCP infrastructure using Terraform
#   2. Build and push the container image
#   3. Output commands to configure Twitter secrets
#   4. Test the deployment
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       Phantom Tech Influencer - Infrastructure Setup             ║"
echo "║                    AWST Timezone | AUD Currency                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Get project ID
if [ -n "$1" ]; then
    PROJECT_ID="$1"
elif [ -n "$TF_VAR_project_id" ]; then
    PROJECT_ID="$TF_VAR_project_id"
else
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
fi

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: No project ID provided.${NC}"
    echo ""
    echo "Usage: ./setup.sh YOUR_PROJECT_ID"
    echo "   or: export TF_VAR_project_id=YOUR_PROJECT_ID && ./setup.sh"
    echo "   or: gcloud config set project YOUR_PROJECT_ID && ./setup.sh"
    exit 1
fi

export TF_VAR_project_id="$PROJECT_ID"
REGION="${TF_VAR_region:-us-central1}"

echo -e "${GREEN}Project ID:${NC} $PROJECT_ID"
echo -e "${GREEN}Region:${NC}     $REGION"
echo -e "${GREEN}Timezone:${NC}   Australia/Perth (AWST)"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v terraform &> /dev/null; then
    echo -e "${RED}Error: Terraform is not installed.${NC}"
    echo "Install: https://developer.hashicorp.com/terraform/downloads"
    exit 1
fi
echo "  ✓ Terraform $(terraform version -json | grep -o '"terraform_version":"[^"]*"' | cut -d'"' -f4)"

if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed.${NC}"
    echo "Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi
echo "  ✓ gcloud CLI"

# Check gcloud auth
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &>/dev/null; then
    echo -e "${RED}Error: Not authenticated with gcloud.${NC}"
    echo "Run: gcloud auth login"
    exit 1
fi
echo "  ✓ gcloud authenticated"

# Set project
gcloud config set project "$PROJECT_ID" 2>/dev/null || true
echo ""

# ============================================================================
# Step 1: Initialize Terraform and create prerequisites (APIs, Artifact Registry)
# ============================================================================
echo -e "${BLUE}Step 1: Creating prerequisite infrastructure...${NC}"
echo ""

terraform init -input=false

echo ""
echo -e "${YELLOW}Creating APIs and Artifact Registry first...${NC}"
# First, create only the APIs and Artifact Registry so we can build the image
terraform apply -input=false -auto-approve \
    -target=google_project_service.required_apis \
    -target=google_artifact_registry_repository.phantom_repo \
    -target=google_service_account.phantom_sa

echo ""
echo -e "${GREEN}✓ Prerequisites created!${NC}"

# ============================================================================
# Step 2: Build and Push Container
# ============================================================================
echo ""
echo -e "${BLUE}Step 2: Building and pushing container image...${NC}"
echo ""

# Build image URL
IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/phantom-influencer/phantom-influencer:latest"

echo "Building image: $IMAGE_URL"
echo ""

# Navigate to project root and build
cd ..
gcloud builds submit --tag "$IMAGE_URL" .
cd terraform

echo ""
echo -e "${GREEN}✓ Container built and pushed!${NC}"

# ============================================================================
# Step 3: Apply full Terraform (Cloud Run Job, Scheduler, IAM)
# ============================================================================
echo ""
echo -e "${BLUE}Step 3: Creating Cloud Run Job and Scheduler...${NC}"
echo ""

terraform apply -input=false -auto-approve

echo ""
echo -e "${GREEN}✓ All infrastructure created!${NC}"

# Get job name for output commands
JOB_NAME=$(terraform output -raw job_name 2>/dev/null || echo "phantom-influencer")

# ============================================================================
# Step 4: Output next steps
# ============================================================================
echo ""
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE!                                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${YELLOW}NEXT STEPS - Configure Twitter API Secrets:${NC}"
echo ""
echo "Add your Twitter API credentials to Secret Manager:"
echo ""
echo -e "${GREEN}# Required secrets:${NC}"
echo "echo -n 'YOUR_CONSUMER_KEY' | gcloud secrets versions add TWITTER_CONSUMER_KEY --data-file=-"
echo "echo -n 'YOUR_CONSUMER_SECRET' | gcloud secrets versions add TWITTER_CONSUMER_SECRET --data-file=-"
echo "echo -n 'YOUR_ACCESS_TOKEN' | gcloud secrets versions add TWITTER_ACCESS_TOKEN --data-file=-"
echo "echo -n 'YOUR_ACCESS_TOKEN_SECRET' | gcloud secrets versions add TWITTER_ACCESS_TOKEN_SECRET --data-file=-"
echo ""
echo -e "${GREEN}# Optional (for trend analysis):${NC}"
echo "echo -n 'YOUR_BEARER_TOKEN' | gcloud secrets versions add TWITTER_BEARER_TOKEN --data-file=-"
echo ""

echo -e "${YELLOW}USEFUL COMMANDS:${NC}"
echo ""
echo -e "${GREEN}# Test the job (force post):${NC}"
echo "gcloud run jobs execute $JOB_NAME --region $REGION --update-env-vars FORCE_POST=true"
echo ""
echo -e "${GREEN}# View logs:${NC}"
echo "gcloud logging read 'resource.type=cloud_run_job AND resource.labels.job_name=$JOB_NAME' --limit=50"
echo ""
echo -e "${GREEN}# View scheduler status:${NC}"
echo "gcloud scheduler jobs list --location=$REGION"
echo ""
echo -e "${GREEN}# Update container after code changes:${NC}"
echo "cd .. && gcloud builds submit --tag $IMAGE_URL . && gcloud run jobs update $JOB_NAME --image $IMAGE_URL --region $REGION"
echo ""

echo -e "${BLUE}Infrastructure Summary:${NC}"
terraform output
