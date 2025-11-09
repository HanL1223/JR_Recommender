with

orders_extracted as (
    select *
    from {{ ref('int_order_details_extracted') }}
),

order_details as (
    select distinct
        order_id,
        customer_id,
        date_created,
        date_paid,
        order_time,
        order_time_verbose,
        total,
        status,
        cart_size,
        cart_surcharge,
        cart_total_price,
        cart_gst
    from orders_extracted
)

select * 
from order_details 
{% if is_incremental() %}
    where
    {% if var('start_date', False) and var('end_date', False) %}
        {{ 
            log(
                'Loading ' ~ this ~ ' incrementally\nStart Date: ' ~ var('start_date') ~ '\nEnd Date: ' ~ var('end_date'), 
                info=True
            ) 
        }}
        date_created >= '{{ var("start_date") }}'
        and date_created < '{{ var("end_date") }}'
    {% else %}
        {{
            log(
                'Loading ' ~ this ~ ' incrementally with no date range specified',
                info=True
            )
        }}
        date_created > (select max(date_created) from {{ this }})
    {% endif %}
{% endif %}
order by order_id

