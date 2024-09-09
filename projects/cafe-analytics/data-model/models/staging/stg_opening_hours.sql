select
    -- Attempt to cast 'date' to a DATE type, except for 'normal_%' entries
    case
        when date not like 'normal_%' then safe_cast(date as date) else null
    end as actual_date,
    -- Directly use 'open_time' and 'close_time' as they are already INTEGER and can
    -- be null
    open_time,
    close_time,
    special_note,
    -- Extract the day from 'date' when it follows the 'normal_%' pattern, else it's a
    -- special date
    case
        when date like 'normal_%' then replace(date, 'normal_', '') else 'special'
    end as day_category
from `jr-data-training.dbt_cafeanalytics.opening_hours`
