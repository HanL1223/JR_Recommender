with
    enriched_opening_hours as (
        select actual_date, open_time, close_time, special_note
        from `jr-data-training.dbt_cafeanalytics.stg_opening_hours`
    ),

    orders_enriched_with_opening_info as (
        select
            od.order_id,
            od.customer_id,  -- Assuming you can directly include customer_id from the source
            od.ip_addr,
            od.date_created,
            od.date_paid,
            od.total,
            od.status,
            od.cart_size,
            od.cart_surcharge,
            od.cart_total_price,
            od.cart_gst,
            case
                when oh.special_note is not null and oh.open_time is null
                then null
                else
                    coalesce(
                        oh.open_time,
                        case
                            when
                                format_date('%u', date(od.date_created))
                                between '1' and '5'
                            then 600
                            else 700
                        end
                    )
            end as opening_time,
            case
                when oh.special_note is not null and oh.close_time is null
                then null
                else coalesce(oh.close_time, 1600)
            end as closing_time,
            coalesce(oh.special_note, 'Normal') as special_note,
            format_date('%A', date(od.date_created)) as day_of_week,
            -- Directly use the day of the week from the order's creation date.
            case
                status
                when 2
                then 'Payment is successful'
                when 1
                then 'Payment is successful, label not printed'
                when 9
                then 'Payment is unsuccessful'
                when 0
                then 'Order created but no payment made'
                when 10
                then
                    case
                        when extract(year from date(date_created)) between 2019 and 2020
                        then 'Unknown (2019-2020)'
                        else 'Status Unknown'
                    end
                else 'Status Unknown'
            end as status_description
        from `jr-data-training.dbt_cafeanalytics.order_details` od
        left join enriched_opening_hours oh on date(od.date_created) = oh.actual_date
    )

select
    order_id,
    customer_id,
    ip_addr,
    date_created,
    date_paid,
    total,
    status,
    cart_size,
    cart_surcharge,
    cart_total_price,
    cart_gst,
    opening_time,
    closing_time,
    special_note,
    day_of_week,
    status_description
from orders_enriched_with_opening_info

