select
    concat(
        cast(extracted.order_id as string), '-', cast(extracted.item_rank as string)
    ) as order_item_ranking,
    extracted.*
from
    (
        select
            o.order_id,
            cast(json_extract_scalar(item, '$.name') as string) as item_name,
            case
                when
                    cast(json_extract_scalar(item, '$.name') as string) in (
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
                when
                    cast(json_extract_scalar(item, '$.name') as string) in (
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
                when
                    cast(json_extract_scalar(item, '$.name') as string) in (
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
            ifnull(
                safe_cast(json_extract_scalar(item, '$.quantity') as int64), 1
            ) as item_quantity,
            round(
                cast(json_extract_scalar(item, '$.price') as float64) / 100, 2
            ) as item_price,
            cast(
                json_extract_scalar(o.items_json, '$.order_time') as string
            ) as order_time,
            cast(
                json_extract_scalar(o.items_json, '$.order_time_verbose') as string
            ) as order_time_verbose,
            dense_rank() over (
                partition by o.order_id
                order by cast(json_extract_scalar(item, '$.name') as string)
            ) as item_rank
        from {{ ref("stg_orders") }} as o
        cross join unnest(json_extract_array(o.items_json, '$.cart')) as item
    ) as extracted
