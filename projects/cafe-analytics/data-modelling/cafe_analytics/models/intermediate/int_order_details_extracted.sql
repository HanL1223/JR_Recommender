with

orders as (
    select *
    from {{ ref("stg_orders") }}
),

orders_extracted as (
    select
        orders.order_id,
        orders.customer_id,
        orders.date_created,
        orders.date_paid,
        orders.total,
        orders.status,

        cast(
            json_extract_scalar(
                orders.items_json,
                '$.cart_size'
            ) as int64
        ) as cart_size,
        round(
            cast(
                json_extract_scalar(
                    orders.items_json,
                    '$.cart_surcharge'
                ) as float64
            ) / 100,
            2
        ) as cart_surcharge,
        round(
            cast(
                json_extract_scalar(
                    orders.items_json,
                    '$.cart_total_price'
                ) as float64
            ) / 100,
            2
        ) as cart_total_price,
        round(
            cast(
                json_extract_scalar(
                    orders.items_json,
                    '$.cart_gst'
                ) as float64
            ) / 100,
            2
        ) as cart_gst,
        cast(
            json_extract_scalar(
                orders.items_json,
                '$.order_time'
            ) as string
        ) as order_time,
        cast(
            json_extract_scalar(
                 orders.items_json,
                 '$.order_time_verbose'
            ) as string
        ) as order_time_verbose,

        cast(
            json_extract_scalar(cart_items, '$.name') as string
        ) as item_name,
        ifnull(
            safe_cast(
                json_extract_scalar(cart_items, '$.quantity') as int64
            ), 
            1
        ) as item_quantity,
        round(
            cast(
                json_extract_scalar(cart_items, '$.price') as float64
            ) / 100,
            2
        ) as item_price,
        
        cart_items

    from orders
    cross join unnest(
        json_extract_array(orders.items_json, '$.cart')
    ) as cart_items
),

orders_categorized as (
    select
        dense_rank() over (
            partition by order_id
            order by item_name, cart_items
        ) as item_id,
        dense_rank() over (
            partition by order_id
            order by item_name
        ) as item_rank,
        case
            when item_name in (
                'Cappuccino',
                'Latte',
                'Espresso',
                'White Hot Chocolate',
                'Short Black',
                'Piccolo',
                'Mocha',
                'Long Macchiato',
                'Long Black',
                'Hot Chocolate',
                'Golden Latte',
                'Short Macchiato',
                'Magic',
                'Macha Latte',
                'Flat White',
                'Dirty Chai Latte',
                'Chai Latte',
                'Babychino'
            )
            then 'Hot Drinks'
            when item_name in (
                'Croissant',
                'Toasted Roll',
                'Quiche',
                'Orange Almond Cake (VG)',
                'Orange Almond Cake (GF,VG)',
                'Homemade Sausage Roll',
                'Toastie',
                'Toasties',
                'Plain Toast',
                'Quiche',
                'Homemade Spinach Ricotta Roll',
                'Homemade Sausage Roll',
                'Hash Brown',
                'Fruit Toast (V)',
                'Fruit Toast',
                'Crepe',
                'Chips and Wedges'
            )
            then 'Food'
            when item_name in (
                'Iced Coffee',
                'Smoothie',
                'Spider',
                'Milkshake (Syrup)',
                'Thickshake (Syrup)',
                'Muffin',
                'Orange Almond Cake (GF,VG)',
                'Scone',
                'Iced Mocha with Cream',
                'Iced Coffee with Cream',
                'Iced Chocolate with Cream',
                'Ice Latte',
                'Ice Chai',
                'Friand (gf)',
                'Friand',
                'Freshly Squeezed Juice',
                'Focaccia',
                'Drinks from fridge',
                'Cookies',
                'Coffee Milkshake',
                'Coffee Frappe',
                'Carrot Cake',
                'Brownie',
                'Banana Bread',
                'Affogato',
                'Acai smoothie bowl (vo)'
            )
            then 'Cold Drinks'
            else 'Kitchen'
        end as item_category,
        *,
        case
            when status = 2
            then 'Payment is successful'
            when status = 1
            then 'Payment is successful, label not printed'
            when status = 9
            then 'Payment is unsuccessful'
            when status = 0
            then 'Order created but no payment made'
            when status = 10
            then
                case
                    when extract(year from date(date_created)) 
                         between 2019 and 2020
                    then 'Unknown (2019-2020)'
                    else 'Status Unknown'
                end
            else 'Status Unknown'
        end as status_description
    from orders_extracted
),

orders_indexed as (
    select
        concat(
            cast(order_id as string),
            '-',
            cast(item_id as string)
        ) as order_item_id,
        concat(
            cast(order_id as string),
            '-',
            cast(item_rank as string)
        ) as order_item_ranking,
        *
    from orders_categorized
),

orders_aggregated as (
    select
        order_item_id,
        order_item_ranking,
        item_id,
        item_rank,
        item_category,
        order_id,
        customer_id,
        date_created,
        date_paid,
        total,
        status,
        cart_size,
        cart_surcharge,
        cart_total_price,
        cart_gst,
        order_time,
        order_time_verbose,
        item_name,

        sum(item_quantity) over (
            partition by order_item_id
        ) as item_quantity,

        sum(item_price) over (
            partition by order_item_id
        ) as item_price,

        status_description,
        cart_items

    from orders_indexed
),

orders_distinct as (
    select distinct * from orders_aggregated
)

select *
from orders_distinct
where 
    cart_items is not null
{% if is_incremental() %}
    {% if var('start_date', False) and var('end_date', False) %}
        {{ 
            log(
                'Loading ' ~ this ~ ' incrementally\nStart Date: ' ~ var('start_date') ~ '\nEnd Date: ' ~ var('end_date'), 
                info=True
            ) 
        }}
        and date_created >= '{{ var("start_date") }}'
        and date_created < '{{ var("end_date") }}'
    {% else %}
        {{
            log(
                'Loading ' ~ this ~ ' incrementally with no date range specified',
                info=True
            )
        }}
        and date_created > (select max(date_created) from {{ this }})
    {% endif %}
{% endif %}



