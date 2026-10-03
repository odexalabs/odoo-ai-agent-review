"""Test tiers. A plain `pytest` (and CI) runs only the pure-Python tests: nothing connects to PostgreSQL,
launches Odoo or calls a provider. Two tiers run only when asked for explicitly:

  integration   a PostgreSQL cluster holding the synthetic fixture templates (fixtures/README.md).
                Opt in with AGENT_REVIEW_INTEGRATION=1. Names: AGENT_REVIEW_TEST_TEMPLATE (Odoo 19),
                AGENT_REVIEW_TEST_TEMPLATE20 (Odoo 20), AGENT_REVIEW_TEST_ROLE (the execution role).
  odoo          additionally starts real Odoo processes from your own Odoo source trees.
                Opt in with AGENT_REVIEW_ODOO=1 (implies integration). Trees: AGENT_REVIEW_TEST_ODOO19_ROOT /
                AGENT_REVIEW_TEST_ODOO19_PYTHON and AGENT_REVIEW_TEST_ODOO20_ROOT / AGENT_REVIEW_TEST_ODOO20_PYTHON.

A skipped tier is reported as skipped, with the reason: read the skip count beside the pass count."""
from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    odoo = os.environ.get("AGENT_REVIEW_ODOO") == "1"
    db = odoo or os.environ.get("AGENT_REVIEW_INTEGRATION") == "1"
    for item in items:
        if "odoo" in item.keywords and not odoo:
            item.add_marker(pytest.mark.skip(reason="starts Odoo: opt in with AGENT_REVIEW_ODOO=1 (tests/conftest.py)"))
        elif "integration" in item.keywords and not db:
            item.add_marker(pytest.mark.skip(reason="needs PostgreSQL and the fixture templates: opt in with "
                                                    "AGENT_REVIEW_INTEGRATION=1 (tests/conftest.py)"))
