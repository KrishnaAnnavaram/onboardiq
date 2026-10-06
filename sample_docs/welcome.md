---
role: all
level: all
topic: first-week
---
# Welcome to the Data Team

## First Week Checklist

On day one, request access to the data warehouse, the code repository and the BI tool through the access portal. Approvals usually take one business day. Your manager approves warehouse access; the platform team approves production write access, which new joiners do not get in their first month.

During the first week, pair with your onboarding buddy for at least two hours, read the data platform overview, and run the starter notebook end to end in the development environment.

## Who to Ask

Your onboarding buddy is the first person to ask about anything. Questions about warehouse permissions go to the platform team channel. Questions about metric definitions go to the analytics guild channel. Security or privacy concerns go straight to the data protection officer, never to a public channel.

## Communication Norms

We write decisions down. Any change that affects a shared table needs a short design note in the team wiki before it is merged. Stand-up is asynchronous: post yesterday, today and blockers in the team channel before 10:00.

## Working with Sensitive Data

Personal data lives only in restricted schemas whose names end in `_pii`. Never copy rows from a restricted schema into a notebook, a spreadsheet or a chat message. Use the masked views for exploration; they hash identifiers and drop free-text fields.
