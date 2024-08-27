-- models/fact_product_sales.sql
with
    item_sales as (
        select
            i.order_id,
            i.item_name,
            i.item_category,
            sum(i.item_quantity) as quantity_sold,
            sum(i.item_price * i.item_quantity) as total_sales
        from `jr-data-training.dbt_cafeanalytics.item_details` i
        group by i.order_id, i.item_name, i.item_category
    ),

    order_details_enriched as (
        select d.order_id, d.date_created, d.customer_id
        from `jr-data-training.dbt_cafeanalytics.combined_order_details` d
    )

select
    o.date_created,
    o.customer_id,
    s.item_name,
    s.item_category,
    s.quantity_sold,
    s.total_sales
from item_sales s
join order_details_enriched o on s.order_id = o.order_id

