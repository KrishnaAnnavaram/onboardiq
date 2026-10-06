---
role: all
level: all
topic: platform
---
# Data Platform Overview

## Environments

There are three environments: dev, staging and prod. Everyone can create tables in their personal dev schema, named `dev_<username>`. Staging mirrors production with a one-day delay and is where pull requests are tested. Only the deployment pipeline writes to prod.

## Warehouse Layers

Raw data lands in the `raw` layer exactly as received. The `staging` layer cleans types and renames columns. The `core` layer holds modelled fact and dimension tables. The `marts` layer holds business-facing tables that dashboards read. Never point a dashboard at `raw` or `staging`.

## Naming Conventions

Fact tables start with `fct_`, dimension tables with `dim_`, and intermediate models with `int_`. Columns use snake_case. Timestamps end in `_at` and are stored in UTC; dates end in `_date`. Boolean columns start with `is_` or `has_`.

## Data Catalog

Every table in `core` and `marts` has an owner, a description and a freshness expectation in the data catalog. If a table has no owner listed, treat it as unsupported and ask in the platform team channel before building on it.
