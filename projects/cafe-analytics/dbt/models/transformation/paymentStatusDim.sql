-- models/dim_payment_status.sql
select distinct
    status,
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
        then 'Unknown (2019-2020)'  -- Assuming this logic is static and not date-dependent.
        else 'Status Unknown'
    end as status_description
from {{ ref("fact_orders") }}
