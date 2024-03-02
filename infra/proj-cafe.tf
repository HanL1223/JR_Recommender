# create bigquery dataset cafe-analytics
resource "google_bigquery_dataset" "cafe" {
  dataset_id                  = "cafe"
  friendly_name               = "Cafe Analytics"
  description                 = "Cafe Analytics"
  project                     = "jr-data-training"
  location                    = "australia-southeast1"
  default_table_expiration_ms = 3600000
  labels = {
    environment = "training"
  }
}