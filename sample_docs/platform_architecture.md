---
role: data_engineer
level: senior
topic: architecture
---
# Platform Architecture Decisions

## Data Contracts

Producing teams publish a data contract for every event stream: schema, owner, freshness target and allowed breaking changes. Breaking changes require a new version of the stream and a thirty-day overlap period. Senior engineers review new contracts and own the relationship with producing teams.

## Batch Versus Streaming

We default to hourly batch. Streaming is approved only when a consumer needs data in under five minutes and can name the decision that depends on it. Every streaming job must also have a batch reconciliation that runs daily and alerts on drift above 0.5 percent.

## Ownership Model

Each domain owns its `core` models and their quality. The platform team owns the orchestration, the warehouse configuration and shared tooling. Cross-domain tables need a named owner before they can be promoted to `marts`.

## Cost Governance

Warehouse spend is reviewed monthly. Senior engineers own the cost of their domain's scheduled workloads and are expected to propose clustering, incremental models or retention changes when a workload grows faster than 10 percent per month.
