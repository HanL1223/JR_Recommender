with
    cte_opening_hours as (
        select
            actual_date,
            open_time,
            close_time,
            special_note,
            -- Convert 'day_category' directly, no need for case statement since
            -- 'actual_date' and 'day_category' are already correctly aligned
            day_category
        from {{ ref("stg_opening_hours") }}
    ),

    cte_order_details_enriched as (
        select
            o.*,
            oh.open_time as opening_time,
            oh.close_time as closing_time,
            oh.special_note,
            oh.day_category
        from {{ ref("order_details") }} o
        left join
            cte_opening_hours oh
            on (oh.day_category = 'special' and date(o.date_created) = oh.actual_date)
            or (
                oh.day_category = 'normal'
                and format_date('%A', o.date_created) = oh.special_note  -- Assuming 'special_note' field is used to store the weekday name for 'normal' days
            )
    )

select *
from cte_order_details_enriched
