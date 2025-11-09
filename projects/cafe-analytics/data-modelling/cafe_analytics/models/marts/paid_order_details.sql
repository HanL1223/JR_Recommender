{{
    config(
        materialized='incremental',
        on_schema_change='fail'    
    )
}}

with 

orders_extracted as (
    select *
    from {{ ref("int_order_details_extracted") }}
),

paid_orders as (
    select distinct
        order_id,
        customer_id,
        cart_total_price as order_price,
        item_id as item_tracking_id,
        item_name as item,
        item_category as category,
        item_quantity as quantity,
        item_price,
        date_created as order_time
    from orders_extracted
    /* Keep the successfully paid transactions only */
    where date_paid IS NOT NULL and status IN (1, 2)
)

select * 
from paid_orders 
{% if is_incremental() %}
    where
    {% if var('start_date', False) and var('end_date', False) %}
        {{ 
            log(
                'Loading ' ~ this ~ ' incrementally\nStart Date: ' ~ var('start_date') ~ '\nEnd Date: ' ~ var('end_date'), 
                info=True
            ) 
        }}
        order_time >= '{{ var("start_date") }}'
        and order_time < '{{ var("end_date") }}'
    {% else %}
        {{
            log(
                'Loading ' ~ this ~ ' incrementally with no date range specified',
                info=True
            )
        }}
        order_time > (select max(order_time) from {{ this }})
    {% endif %}
{% endif %}
order by order_time, order_id, item_tracking_id
