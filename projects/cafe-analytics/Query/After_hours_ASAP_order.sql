SELECT
  *
FROM
  `jr-data-training.cafe.cafe-sales`
WHERE
  TIME(date_created) NOT BETWEEN '5:45:00' AND '16:00:00'
  AND status = 2
  AND JSON_EXTRACT_SCALAR(items, '$.order_time_verbose') = 'ASAP'