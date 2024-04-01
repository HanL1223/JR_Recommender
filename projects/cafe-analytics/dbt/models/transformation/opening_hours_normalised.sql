with
    cte_opening_hours as (
        select
            case
                when date not like 'normal_%' then safe_cast(date as date) else null
            end as actual_date,
            case
                when open_time != '' then safe_cast(open_time as int64) else null
            end as open_time,
            case
                when close_time != '' then safe_cast(close_time as int64) else null
            end as close_time,
            special_note,
            case
                when date like 'normal_%' then replace(date, 'normal_', '') else null
            end as weekday,
            case
                when date like 'normal_%' then 'normal' else 'special'
            end as day_category
        from `{{ ref('stg_opening_hours') }}`
    ),

    cte_order_details_enriched as (
        select
            o.*,
            oh.open_time as opening_time,
            oh.close_time as closing_time,
            oh.special_note,
            oh.day_category
        from `{{ ref('order_details') }}` o
        left join
            cte_opening_hours oh
            on (
                oh.day_category = 'special'
                and format_date('%Y-%m-%d', o.date_created) = oh.actual_date
            )
            or (
                oh.day_category = 'normal'
                and format_date('%A', o.date_created) = oh.weekday
            )
    )

select *
from cte_order_details_enriched
