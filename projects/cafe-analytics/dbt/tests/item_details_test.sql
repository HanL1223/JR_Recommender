-- Test to ensure that the item_name field is not null
select count(*)
from `jr-data-training`.`dbt_cafeanalytics`.`item_details`
where item_name is null
having count(*) = 0


