-- This script is a placeholder. In practice, you'd use a dbt package like dbt_date or
-- generate the table with a script.
-- models/dim_time.sql
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
from unnest(generate_date_array('2020-01-01', '2030-12-31')) as date
