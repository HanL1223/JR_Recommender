WITH cust_orders AS (
  SELECT DISTINCT od.customer_id, od.date_created AS order_time,
                  od.order_id
  FROM `jr-data-training.dbt_cafeanalytics.order_details` AS od
  JOIn `jr-data-training.dbt_cafeanalytics.item_details` AS id
  ON od.order_id = id.order_id
  WHERE od.date_paid IS NOT NULL AND od.status <> 10
  ORDER BY od.customer_id, order_time, od.order_id
),
cust_next_orders AS (
  SELECT customer_id, order_time,
        LEAD(order_time) OVER (
          PARTITION BY customer_id ORDER BY order_time, order_id
        ) AS next_order_time,
        order_id
  FROM cust_orders
  ORDER BY customer_id, order_time, order_id
),
order_intervals AS (
  SELECT *,
         TIMESTAMP_DIFF(next_order_time, order_time, DAY) AS purchase_interval_days
  FROM cust_next_orders
),
interval_mean_std AS (
  SELECT customer_id,
        AVG(purchase_interval_days) AS interval_mean,
        STDDEV(purchase_interval_days) AS interval_std,
  FROM order_intervals
  GROUP BY customer_id
  ORDER BY customer_id
),
churn_thresholds AS (
  SELECT customer_id, interval_mean + 1 * interval_std AS churn_threshold
  FROM interval_mean_std
),
churn_threshold_all AS (
  SELECT AVG(purchase_interval_days) + 1 * STDDEV(purchase_interval_days) AS churn_threshold_all
  FROM order_intervals
)

SELECT customer_id,
       CASE
       WHEN c.churn_threshold IS NULL THEN ca.churn_threshold_all
       ELSE c.churn_threshold
       END AS churn_threshold
FROM churn_thresholds AS c
CROSS JOIN churn_threshold_all AS ca
ORDER BY customer_id
