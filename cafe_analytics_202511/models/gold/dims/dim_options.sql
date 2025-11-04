{{ config(
    materialized='table',
    schema='gold'
) }}

SELECT
-- Option name + Option value = option SKU 
-- Business logic require
  {{ dbt_utils.generate_surrogate_key(['option_name', 'option_value']) }} AS option_key,
  
  option_name AS option_category,
  option_value,
  
  MAX(option_price) AS standard_price,
  
  -- Flags
  option_price = 0 AS is_default,
  option_price > 0 AS is_premium,
  option_quantity IS NOT NULL AS is_countable,
  
--   CURRENT_TIMESTAMP() AS created_at,
--   CURRENT_TIMESTAMP() AS updated_at

FROM {{ ref('int_order_item_options') }}
GROUP BY option_name, option_value, option_price, option_quantity