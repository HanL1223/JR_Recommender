{{ config(
    materialized='table',
    schema='gold'
) }}

SELECT DISTINCT
  {{ dbt_utils.generate_surrogate_key(['item_name', 'variant_name']) }} AS product_key,
  
  item_name AS product_name,
  variant_name,
  category,
 -- variant_desc,
 -- variant_image,
  
  -- Base price (from most common unit_price for this product)
  -- Business logic require
  MAX(base_variant_price) AS base_price,
  
  -- Classification
  -- Business logic require
  CASE 
    WHEN variant_name IN ('Large', 'Regular', 'Small', 'Mini', 'Extra Large', 'XLG') 
      THEN 'Size Variant'
    WHEN variant_name IS NULL OR variant_name = ''
      THEN 'Standard'
    ELSE 'Specialty'
  END AS variant_type,
  
  TRUE AS is_active,
  
--   CURRENT_TIMESTAMP() AS created_at,
--   CURRENT_TIMESTAMP() AS updated_at

FROM {{ ref('int_order_items_flattened') }}
where category is not null
GROUP BY product_name, variant_name, category
--, variant_desc, variant_image