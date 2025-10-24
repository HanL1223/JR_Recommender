# create a gcp project first
provider "google" {
  project = "jr-data-training"
  region  = "australia-southeast1"
}

# store states in gcs
terraform {
  backend "gcs" {
    bucket = "jr-data-training-terraform-state"
    prefix = "terraform/state"
  }
}

# cloud storage for raw data
resource "google_storage_bucket" "raw_data" {
  name     = "jr-data-training-raw-data"
  location = "australia-southeast1"
}

# create an IAM Role, for all trainees
resource "google_project_iam_custom_role" "trainee" {
  role_id     = "JRTrainee"
  title       = "JR Trainee"
  description = "Trainee role for JR Data Training"
  permissions = [
    # BigQuery Read Only, but billed to their own project
    "bigquery.datasets.get",
    "bigquery.tables.getData",
    "bigquery.tables.list",
    "bigquery.tables.get",
    "bigquery.tables.export",
    # able to create tables
    "bigquery.tables.create",
    "bigquery.tables.updateData",
    "bigquery.tables.update",
    "bigquery.tables.delete",
    # able to update dataset, for dbt / data engineer
    "bigquery.datasets.update",
    "bigquery.datasets.delete",
    "bigquery.datasets.create",
    # job creation
    "bigquery.jobs.create",
  ]
}

# set a list of trainees
variable "trainees" {
  type = list(string)
  default = [
    "serviceAccount:jr-data-training-sa@jr-data-training.iam.gserviceaccount.com",
    "user:juanningshi@gmail.com",
    "user:maxinexiong2@gmail.com",
    # --- students ---
    "user:alicia.jinhui.qu@gmail.com", # start from 2025-10-27
    "user:ylin0034@gmail.com", # start from 2025-10-27
    "user:zibozhao52@gmail.com", # start from 2025-10-27
    "user:harryberlin996@gmail.com", # start from 2025-10-27
    "user:laaron1223@gmail.com", # start from 2025-10-27
    "user:lzlzyy@hotmail.com", # start from 2025-10-27
  ]
}

# Grant the trainee role to the trainees xx@xx.com
resource "google_project_iam_member" "trainee" {
  project  = "jr-data-training"
  role     = google_project_iam_custom_role.trainee.id
  for_each = toset(var.trainees)
  member   = each.value
}

# https://community.fabric.microsoft.com/t5/Service/BIgQuery-Account-Permissions/m-p/1390692
# Assign the predefined GCP IAM role "BigQuery Read Session User" 
# to allow the user to query the data in Power BI.
resource "google_project_iam_member" "bigquery_read_session_user" {
  project  = "jr-data-training"
  role     = "roles/bigquery.readSessionUser"
  for_each = toset(var.trainees)
  member   = each.value
}
