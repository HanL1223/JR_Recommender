{{ config(
    materialized='view',
    schema='bronze'
) }}

SELECT *
FROM {{ source('cafe_raw_data', 'cafe_sales') }}