---
role: data_scientist
level: senior
topic: deployment
---
# Model Deployment and Monitoring

## Deployment Paths

Batch models write scores to a `marts` table on the pipeline schedule. Online models are packaged as containers and deployed behind the model gateway with a canary that receives 5 percent of traffic for 24 hours before full rollout.

## Monitoring and Drift

Every production model has dashboards for input feature drift, prediction distribution and the business metric. Drift alerts fire when the population stability index of a key feature exceeds 0.2. The model owner triages drift alerts within one business day.

## Rollback

Keep the previous model version deployable. If the canary degrades the business metric beyond the agreed guardrail, roll back with the gateway rollback command and write an incident note. Senior data scientists own rollback decisions for their models.

## Retraining Policy

Retrain on a fixed schedule only when drift is routine; otherwise retrain when monitoring shows decay. Every retrained model goes through the same offline evaluation and canary as a new model.
