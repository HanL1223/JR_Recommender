with 
raw_orders as (
    select *  
    from {{ source('cafe_raw', 'cafe') }} 
),
stg_orders as (
    select
        order_id,
        customer_id,
        date_created,
        date_paid,
        -- Adjusting the total price from cents to dollars and rounding to 2 decimal places
        round(total / 100, 2) as total,
        status,
        items as items_json
    from raw_orders
)

select *
from stg_orders