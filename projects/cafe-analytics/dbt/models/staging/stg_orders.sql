-- cleaning and preparing the order data
select
    order_id,
    customer_id,
    ip_addr,
    timestamp(date_created) as date_created_ts,
    timestamp(date_paid) as date_paid_ts,
    total,
    status
from `jr-data-training.cafe.cafe-sales`
