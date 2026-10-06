---
role: data_analyst, data_engineer
level: all
topic: sql
---
# SQL Style and Review Guide

## Style Rules

Write keywords in lower case and put each selected column on its own line. Use common table expressions instead of nested subqueries, and give every CTE a name that says what it contains. Always qualify columns with a table alias when a query joins more than one table. Never use `select *` in a model that other people depend on.

## Query Review

Every model change is reviewed by one teammate before merge. Reviewers check the grain of the result (one row per what?), join fan-out, and whether filters on dates use half-open intervals such as `>= start and < end`.

## Cost Limits

Interactive queries are limited to scanning 200 GB. If a query hits the limit, filter on the partition column (usually `event_date`) or query a mart instead of the raw events table. Scheduled jobs that scan more than 1 TB need sign-off from the platform team.
