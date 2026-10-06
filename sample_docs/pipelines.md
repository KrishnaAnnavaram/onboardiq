---
role: data_engineer
level: junior, mid
topic: pipelines
---
# Building and Running Pipelines

## Orchestration Basics

Pipelines are defined as DAGs in the orchestration repository. Each DAG has an owner, a schedule written as a cron expression, and a service level that says when its output must be ready. New DAGs start paused; unpause them only after a successful run in staging.

## Retries and Alerts

Every task retries three times with exponential backoff starting at five minutes. After the final retry fails, the on-call engineer is paged. Do not raise the retry count to hide a flaky source; open a ticket for the source owner instead.

## Backfills

Run backfills with the backfill command, one month at a time, and only outside the 06:00 to 09:00 UTC window when the morning dashboards refresh. Announce large backfills in the platform team channel beforehand because they compete for warehouse slots.

## Data Quality Checks

Every model has tests for uniqueness of its primary key and non-null foreign keys. A failing test blocks downstream tasks. If a test fails because the source data is genuinely late, mark the run as skipped rather than deleting the test.

## On-Call Rotation

Engineers join the on-call rotation after their third month. The on-call engineer acknowledges a page within 15 minutes and writes a short incident note for anything that delays a dashboard.
