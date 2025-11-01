{{ config(materialized='view') }}

SELECT *
FROM {{ source('cafe_raw_data', 'cafe_sales') }}