<<<<<<< HEAD
-- where you prepare the orders data from your source tables.
=======
-- extract and preparing data in the staging layer from BigQuery
-- select from your raw orders table. Ensure the items column is correctly handled as JSON. 
-- include renaming columns for consistency and possibly converting string timestamps to appropriate timestamp types.
with
    raw_orders as (
        select
            order_id,
            customer_id,
            ip_addr,
            cast(date_created as timestamp) as date_created,  -- Casting to TIMESTAMP if not already
            cast(date_paid as timestamp) as date_paid,  -- Casting to TIMESTAMP if not already
            total,
            status,
            items as items_json  -- Directly using items as JSON-formatted string
        from `jr-data-training.cafe.cafe-sales`  -- Your source table
    )

select
    order_id, customer_id, ip_addr, date_created, date_paid, total, status, items_json
from raw_orders

>>>>>>> feature/CA-14
