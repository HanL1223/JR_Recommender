{{ config(
    materialized='view',
    schema='bronze'
) }}

SELECT *
FROM `jr-data-training.cafe.cafe-sales`