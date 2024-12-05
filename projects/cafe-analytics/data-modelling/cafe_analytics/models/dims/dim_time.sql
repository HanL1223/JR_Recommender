with

orders as (
    select *
    from {{ ref("stg_orders") }}
),

order_date_range as (
    select 
        date(min(date_created)) as start_date,
        date(max(date_created)) as end_date
    from orders
),

full_date_range as (
    select
        date as date_key,
        extract(day from date) as day,
        extract(month from date) as month,
        extract(year from date) as year,
        format_date('%A', date) as day_of_week,
        format_date('%B', date) as month_name,
        extract(dayofweek from date) as weekday_index,  -- Sunday=1 through Saturday=7
        extract(week from date) as week_of_year,
        extract(quarter from date) as quarter
    from order_date_range
    cross join unnest(
        generate_date_array(
            order_date_range.start_date, 
            order_date_range.end_date
        )
    ) as date
)

select * 
from full_date_range
order by date_key

