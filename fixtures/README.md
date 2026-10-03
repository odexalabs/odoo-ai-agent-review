# Preparing a fixture template

agent-review never runs an agent against a database you care about. Every run gets a **disposable copy**
(`CREATE DATABASE … TEMPLATE`) of a prepared, frozen **template** database, and the copy is dropped when the run
ends. This directory holds the synthetic data the bundled scenarios expect, for Odoo 19 and Odoo 20. Every name,
amount and address in it is invented.

`agent-review fixture-script 19` (or `20`) prints where these files are in an installed copy.

## What you need

- **Odoo 19 or Odoo 20 with the `ai` module.** The `ai` module is part of Odoo Enterprise, so you need your own
  licensed Enterprise source. This repository contains no Odoo code.
- **PostgreSQL** with the **pgvector** extension available, and a superuser to create
  it: Odoo's `ai` module needs the `vector` extension and cannot create it as an ordinary role.
  `pg_stat_statements` is optional. When it is also in `shared_preload_libraries`, each run reports statement
  counts and execution times for the SQL the agent's Odoo process ran (the query text stays in the run's private
  record); otherwise those figures are marked unavailable.
- **Two roles.** The *execution role* is what Odoo connects as; it owns the template and every run copy. The
  *observer* is the role you run agent-review as. It must be a different role: the tool refuses to run otherwise,
  because the per-run SQL figures would include its own queries. The observer has been tested as a PostgreSQL
  superuser. A least-privilege observer would need to create and drop databases owned by the execution
  role, end backends on a run copy before dropping it, read every table of a run copy, and read the execution
  role's `pg_stat_statements` rows; that setup has not been tested.

## Steps

The example names match the bundled example profiles: template `agent_review_tpl20` (or `agent_review_tpl19`),
execution role `agent_review_exec`. Use your own if you prefer, and put them in your run profile. Commands that
run Odoo are run from your Odoo source tree; `/path/to/…` stands for your own paths.

**1. Role and database** — as a PostgreSQL superuser:

```sql
CREATE ROLE agent_review_exec LOGIN;                 -- add PASSWORD '…' if your pg_hba.conf requires one
CREATE DATABASE agent_review_tpl20 OWNER agent_review_exec ENCODING 'unicode' LC_COLLATE 'C' TEMPLATE template0;
\c agent_review_tpl20
CREATE EXTENSION vector;                             -- before the ai module is installed
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;   -- optional; reports nothing unless preloaded (see above)
```

**2. A configuration for building the template.** Copy `template.conf.example` to `/path/to/template.conf` and set
the paths. Pass it both ways, `ODOO_RC=…` and `-c …`, so Odoo does not also read a `~/.odoorc` of yours.

**3. Install the modules**, without demo data, from your Odoo tree:

```bash
cd /path/to/odoo-20.0
# Odoo 20 (demo data is off unless asked for)
ODOO_RC=/path/to/template.conf ./odoo-bin -c /path/to/template.conf -d agent_review_tpl20 -i crm,account,ai_crm --stop-after-init
# Odoo 19
ODOO_RC=/path/to/template.conf ./odoo-bin -c /path/to/template.conf -d agent_review_tpl19 -i crm,account,ai_crm --without-demo=all --stop-after-init
```

A packaged Odoo tree (a downloaded archive) has no `odoo-bin`: run `python setup/odoo …` instead, from the tree's
root, with `PYTHONPATH` set to that root.

**4. Load the synthetic data** through Odoo's shell, still from your Odoo tree. `agent-review fixture-script 20`
prints the script's path. The script refuses to run twice:

```bash
ODOO_RC=/path/to/template.conf ./odoo-bin shell -c /path/to/template.conf -d agent_review_tpl20 \
  < /path/to/odoo-ai-agent-review/fixtures/odoo20/load_fixture.py
```

It prints `FIXTURE OK`. The Odoo 20 script also replaces every IAP account token in the template with a synthetic
value (`agentreview-synthetic-…`), so the template itself never holds a real-looking token.

**5. Freeze it.** Stop every process connected to the template and never run Odoo against it again: each run
copies it, and agent-review refuses to copy a template that a client session holds open.

**6. Check** with your own profiles:

```bash
agent-review init-profile odoo20-standin-example --output my-odoo20.yaml   # set paths, template, operator
agent-review init-profile noop --output my-noop.yaml                       # set its template
agent-review inspect --profile my-odoo20.yaml                              # version, schema and tool inventory
agent-review run noop --profile my-noop.yaml                               # the lifecycle alone (copy, diff, drop)
agent-review run reassign-opportunities --profile my-odoo20.yaml --probe   # Odoo 20, no provider request
```

## Using a copy of a real database instead

You can point a template at a restored staging copy of your own Odoo. A restored database is **active state**:
it can hold live credentials and integrations. On every run's copy, before Odoo starts, agent-review archives the
outgoing mail servers and verifies that mail cannot leave, removes AI provider keys stored in the database, and on
Odoo 20 replaces every IAP account token with a synthetic run token. It does **not** block other outbound
integrations, such as payment providers, webhooks or other connectors: there is no network jail. It records what
it found in the run's private record. Review a customer-derived template for such integrations before you use it.
