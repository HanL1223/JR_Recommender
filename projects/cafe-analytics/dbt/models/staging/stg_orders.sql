-- cleaning and preparing the order data
select
    order_id,
    customer_id,
    ip_addr,
    to_timestamp(date_created, 'YYYY-MM-DD HH24:MI:SS') as date_created_ts,
    to_timestamp(date_paid, 'YYYY-MM-DD HH24:MI:SS') as date_paid_ts,
    total,
    status
from {{ source("cafe", "cafe-sales") }}
