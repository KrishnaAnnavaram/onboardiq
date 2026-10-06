---
role: data_scientist
level: all
topic: modelling
---
# Machine Learning Workflow

## Experiment Tracking

Log every training run to the experiment tracker with its code version, data snapshot, parameters and metrics. A result that is not in the tracker does not exist for review purposes. Name experiments after the business question, not the algorithm.

## Feature Store

Reusable features live in the feature store with point-in-time correct joins. Before writing a new feature, search the store; duplicated features with slightly different logic are the most common cause of training and serving skew.

## Offline Evaluation

Split data by time, never randomly, when the model will predict the future. Report the metric the business cares about next to the technical metric, and always compare against a simple baseline such as the previous week's value or a logistic regression.

## Model Review

Before a model ships, a model review covers the problem framing, the evaluation set, fairness checks on protected groups, and the monitoring plan. Junior data scientists present with their mentor; the review checklist template is in the team wiki.
