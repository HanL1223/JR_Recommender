-- models/dim_opening_hours.sql
select distinct
    actual_date,
    min(open_time) as open_time,
    max(close_time) as close_time,
    max(special_note) as special_note  -- Assuming there could be multiple notes; adjust as needed.
from
    (
        select actual_date, open_time, close_time, special_note
        from {{ ref("stg_opening_hours") }}  -- Replace with your actual source reference.
    -- This source should ideally be a table or model that contains the base opening
    -- hours data.
    ) enriched_opening_hours
group by actual_date
