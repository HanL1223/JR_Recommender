SELECT DISTINCT od.order_id, od.customer_id,
                od.cart_total_price AS order_price,
                id.item_name AS item,
                id.item_category AS category,
                id.item_quantity AS quantity, id.item_price,
                id.item_id AS item_tracking_id,
                od.date_created AS order_time
FROM `jr-data-training.dbt_cafeanalytics.order_details` AS od
JOIN `jr-data-training.dbt_cafeanalytics.item_details` AS id
ON  od.order_id = id.order_id
/* Keep the successfully paid transactions only */
WHERE od.date_paid IS NOT NULL AND od.status IN (1, 2)
ORDER BY order_time, od.order_id, item_tracking_id
