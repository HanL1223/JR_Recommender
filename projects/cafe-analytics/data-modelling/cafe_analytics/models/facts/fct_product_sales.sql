with

orders_extracted as (
    select *
    from {{ ref('int_order_details_extracted') }}
),

product_sales as (
    select
        order_item_id,
        item_name,
        order_id,
        round(
            item_price / item_quantity,
            2
        ) as price,
        item_quantity as quantity,
        item_price as item_sales,
        date_created
    from orders_extracted
)

select * 
from product_sales 
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
order by order_item_id