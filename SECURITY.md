# Security

agent-review starts Odoo processes against copies of databases, handles a model-provider key and, on Odoo 20,
runs a local stand-in for Odoo's AI endpoint. A flaw in any of these can leak a credential, send mail, or change a
database it should not touch. Please report suspected security problems privately.

## Reporting

Email **support@odexalabs.com** with "SECURITY: odoo-ai-agent-review" in the subject. Include what you observed, how
to reproduce it, and the versions involved. Do not include real credentials, customer data or private run
artifacts (`run.json`, raw diffs, Odoo logs); describe them instead.

Please do not open a public issue for a suspected vulnerability. We will acknowledge your report, work with you on
a fix, and credit you if you wish.

## Scope

In scope: this repository's code and its documented guarantees: credentials sourced only from the named key and
never passed to the wrong process or written to a report; AI provider keys stored in the database removed from the
run's copy, and on Odoo 20 every IAP token replaced with a synthetic run token, before Odoo starts; outgoing mail
blocked and verified before Odoo starts; only the tool's own run databases dropped; the redacted report withholding
free text.

Out of scope: vulnerabilities in Odoo itself, which should go to Odoo S.A., and in your model provider.
