# DBT Models Project Documentation - Cafe Analytics

## Overview

This document provides an overview of the dbt (data build tool) models project, which includes staging, transformation, and analytics folders.

### Staging

In the staging folder, we have the following model:

#### stg_orders

- **Description:** Staging table for orders data.
- **Fields:**
  - `order_id`: INTEGER, nullable
  - `customer_id`: INTEGER, nullable
  - `ip_addr`: STRING, nullable
  - `date_created`: TIMESTAMP, nullable
  - `date_paid`: TIMESTAMP, nullable
  - `total`: INTEGER, nullable
  - `status`: INTEGER, nullable
  - `items_json`: STRING, nullable

### Transformation

In the transformation folder, we have the following models:

#### item_details

- **Description:** Transformation model for item details.
- **Fields:**
  - `item_hash_id`: INTEGER, nullable
  - `order_id`: INTEGER, nullable
  - `item_name`: STRING, nullable
  - `item_category`: STRING, nullable
  - `item_quantity`: INTEGER, nullable
  - `item_price`: FLOAT, nullable
  - `order_time`: STRING, nullable
  - `order_time_verbose`: STRING, nullable

#### item_options

- **Description:** Transformation model for item options.
- **Fields:**
  - `order_id`: INTEGER, nullable
  - `item_name`: STRING, nullable
  - `item_hash_id`: INTEGER, nullable
  - `option_name`: STRING, nullable
  - `option_value`: STRING, nullable
  - `option_price`: FLOAT, nullable

#### order_details

- **Description:** Transformation model for order details.
- **Fields:**
  - `order_id`: INTEGER, nullable
  - `customer_id`: INTEGER, nullable
  - `ip_addr`: STRING, nullable
  - `date_created`: TIMESTAMP, nullable
  - `date_paid`: TIMESTAMP, nullable
  - `total`: FLOAT, nullable
  - `status`: INTEGER, nullable
  - `cart_size`: INTEGER, nullable
  - `cart_surcharge`: FLOAT, nullable
  - `cart_total_price`: FLOAT, nullable
  - `cart_gst`: FLOAT, nullable

### Analytics

In the analytics folder, we have the following model:

#### customer

- **Description:** Analytics model for customer insights.
- **Fields:**
  - `customer_id`: INTEGER, nullable
  - `total_orders`: INTEGER, nullable
  - `avg_order_value`: FLOAT, nullable
  - `last_order_date`: TIMESTAMP, nullable
  - `visits_last_year`: INTEGER, nullable
  - `most_frequent_item`: STRING, nullable
