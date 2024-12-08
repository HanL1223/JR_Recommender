# create a notification channel for alerting
resource "google_monitoring_notification_channel" "admin-email" {
  display_name = "admin"
  type         = "email"
  labels = {
    email_address = "xg1990@gmail.com"
  }
}

# add an alerting policy
# tracking Bigquery scanned bytes billed
# The alert will be triggered if the daily billed bytes is greater than 10Gb
# refer to https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#billingtier
# based on the following code:
# {
#   "displayName": "Bigquery Bytes Billed",
#   "userLabels": {},
#   "conditions": [
#     {
#       "displayName": "BigQuery Project - Statement scanned bytes billed",
#       "conditionThreshold": {
#         "filter": "resource.type = \"bigquery_project\" AND metric.type = \"bigquery.googleapis.com/query/statement_scanned_bytes_billed\"",
#         "aggregations": [
#           {
#             "alignmentPeriod": "86400s",
#             "crossSeriesReducer": "REDUCE_NONE",
#             "perSeriesAligner": "ALIGN_SUM"
#           }
#         ],
#         "comparison": "COMPARISON_GT",
#         "duration": "0s",
#         "trigger": {
#           "count": 1
#         },
#         "thresholdValue": 5000000000
#       }
#     }
#   ],
#   "alertStrategy": {
#     "autoClose": "604800s"
#   },
#   "combiner": "OR",
#   "enabled": true,
#   "notificationChannels": [
#     "projects/jr-data-training/notificationChannels/6777780243199322547"
#   ],
#   "severity": "SEVERITY_UNSPECIFIED"
# }
resource "google_monitoring_alert_policy" "bigquery_bytes_billed" {
  display_name = "Bigquery Bytes Billed"
  conditions {
    display_name = "BigQuery Project - Statement scanned bytes billed"
    condition_threshold {
      filter = "resource.type = \"bigquery_project\" AND metric.type = \"bigquery.googleapis.com/query/statement_scanned_bytes_billed\""
      aggregations {
        alignment_period = "86400s"
        cross_series_reducer = "REDUCE_NONE"
        per_series_aligner = "ALIGN_SUM"
      }
      comparison = "COMPARISON_GT"
      duration = "0s"
      threshold_value = 5000000000 # 5Gb
      trigger {
        count = 1
      }
    }
  }
  alert_strategy {
    auto_close = "604800s"
  }
  combiner = "OR"
  enabled = true
  notification_channels = [google_monitoring_notification_channel.admin-email.id]
  severity = "CRITICAL"
  depends_on = [ google_monitoring_notification_channel.admin-email ]
}
