-- focus might be deduplicating records and ensuring clean, consistent data for
-- customer identifiers:
select distinct
    customer_id,
    first_value(ip_addr) over (
        partition by customer_id order by date_created asc
    ) as first_ip_addr
from {{ source("cafe", "cafe-sales") }}
