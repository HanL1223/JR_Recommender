-- Bronze Layer: Raw data extraction with minimal transformation
-- Source: jr-data-training.cafe.cafe-sales
-- Purpose: Extract raw orders with JSON intact for silver layer parsing

{{ config(
    materialized='table',
    schema='bronze_zzb_dev',
    tags=['bronze', 'raw']
) }}

SELECT
    order_id,
    customer_id,
    ip_addr,
    CAST(date_created AS TIMESTAMP) AS date_created,
    CAST(date_paid AS TIMESTAMP) AS date_paid,
    total,  -- Keep as cents for now
    status,
    items AS items_json  -- Keep JSON as-is

FROM {{ source('cafe', 'cafe-sales') }}
