# ruff: noqa: F821  (run inside Odoo's shell, which provides `env`)
"""agent-review synthetic fixture, Odoo 20. No real customer: the people, amounts and addresses are invented, and
the companies are neutral placeholders (Example Customer A to N).

Run ONCE, through Odoo's shell, against the database you will freeze as the template (fixtures/README.md):

    ODOO_RC=template.conf ./odoo-bin shell -c template.conf -d agent_review_tpl20 < load_fixture.py
    # a packaged tree has no odoo-bin: PYTHONPATH=<tree> python setup/odoo shell -c template.conf -d ... < ...

It refuses to run twice. The same business data as the Odoo 19 fixture, plus:
  - the three salespeople are members of the Sales team, so a reassignment keeps each lead's team;
  - every IAP account token is replaced with a synthetic value, and an `odoo_ai` account is created if missing,
    so the template never holds a real-looking token. (agent-review replaces the tokens again on every run's copy,
    whatever the template holds; this keeps the template itself clean.)
"""
import secrets
from datetime import date

from odoo import Command

if env['res.partner'].search([('name', '=', 'Example Customer A')]):
    raise SystemExit("fixture already loaded")

Partner = env['res.partner']
india = env.ref('base.in')
cust_a = Partner.create({'name': 'Example Customer A', 'is_company': True, 'email': 'ap@customer-a.example'})
cust_b = Partner.create({'name': 'Example Customer B', 'is_company': True})
Partner.create({'name': 'Example Customer C', 'is_company': True, 'country_id': india.id})

env['ir.mail_server'].search([]).unlink()

Users = env['res.users']
def grp(xml_id):
    return env.ref(xml_id).id


sales_groups = [grp('base.group_user'), grp('sales_team.group_sale_salesman')]
ana = Users.create({'name': 'Ana Silva', 'login': 'ana', 'email': 'ana@example.com', 'group_ids': [Command.set(sales_groups)]})
ben = Users.create({'name': 'Ben Okafor', 'login': 'ben', 'email': 'ben@example.com', 'group_ids': [Command.set(sales_groups)]})
chen = Users.create({'name': 'Chen Wei', 'login': 'chen', 'email': 'chen@example.com', 'group_ids': [Command.set(sales_groups)]})
operator = Users.create({   # a TEST-ONLY password for a synthetic user in a disposable database
    'name': 'Eval Operator', 'login': 'evalop', 'password': 'evalop-20-run', 'email': 'operator@example.com',
    'group_ids': [Command.set([grp('base.group_user'), grp('sales_team.group_sale_manager'), grp('account.group_account_manager')])],
})
operator.partner_id.write({'phone': False, 'function': False, 'street': False, 'street2': False,
                           'city': False, 'zip': False, 'state_id': False, 'country_id': False})
team = env.ref('sales_team.team_sales_department')
team.write({'member_ids': [Command.link(u.id) for u in (ana, ben, chen)]})

Move = env['account.move']


def inv(partner, d, due, amount, label, post=True):
    m = Move.create({
        'move_type': 'out_invoice', 'partner_id': partner.id, 'invoice_date': d, 'invoice_date_due': due,
        'invoice_line_ids': [Command.create({'name': label, 'quantity': 1, 'price_unit': amount, 'tax_ids': [Command.clear()]})],
    })
    if post:
        m.action_post()
    return m


inv_d = inv(cust_a,   date(2026, 6, 1),  date(2026, 7, 1),   2000.00, 'Q2 support retainer')
inv(cust_a,           date(2026, 6, 15), date(2026, 7, 15),  1200.00, 'Performance review, June')
inv_b = inv(cust_a,   date(2026, 7, 1),  date(2026, 7, 31),  3450.00, 'Database tuning engagement')
inv(cust_b,         date(2026, 6, 20), date(2026, 7, 20),  4100.00, 'Migration assessment')
inv(cust_a,           date(2026, 7, 20), date(2026, 8, 19),   780.50, 'Index review')
inv(cust_b,         date(2026, 8, 1),  date(2026, 8, 31),   950.00, 'Monitoring setup')
inv(cust_a,           date(2026, 9, 10), date(2026, 12, 10), 5000.00, 'Q4 upgrade project')
inv(cust_a,           date(2026, 9, 1),  date(2026, 9, 5),    999.00, 'Draft — never posted', post=False)


def pay(move, amount, when):
    env['account.payment.register'].with_context(active_model='account.move', active_ids=move.ids).create(
        {'amount': amount, 'payment_date': when}).action_create_payments()


pay(inv_d, 2000.00, date(2026, 7, 5))
pay(inv_b, 1000.00, date(2026, 8, 10))

Lead = env['crm.lead']
stage = {s.name: s for s in env['crm.stage'].search([])}
for n in ('Performance', 'Migration', 'Support'):
    env['crm.tag'].create({'name': n})


def opp(name, user, st, rev, partner=None):
    return Lead.create({'name': name, 'type': 'opportunity', 'user_id': user.id, 'team_id': team.id,
                        'stage_id': stage[st].id, 'expected_revenue': rev, 'partner_id': partner.id if partner else False})


opp('Example Customer B — slow invoicing',        ana,  'New',         12500, cust_b)
opp('Example Customer D — PG14 upgrade',         ana,  'Qualified',    8250)
opp('Example Customer E — monitoring',          ana,  'Proposition',  4320)
opp('Example Customer F — stock performance',      ben,  'New',          3000)
opp('Example Customer G — migration to 19',     ben,  'Qualified',    7750)
opp('Example Customer H — accounting closes',      ben,  'Proposition', 15000)
opp('Example Customer I — index audit',            ben,  'New',          2480)
opp('Example Customer J — bloat cleanup',          chen, 'Qualified',    9999)
opp('Example Customer K — cron backlog',            chen, 'New',          1001)
opp('Example Customer L — worker sizing',         chen, 'Proposition',  6500)
opp('Example Customer M — done deal',          ana,  'Won',         20000)
opp('Example Customer N — went elsewhere',        ben,  'Qualified',    5000).action_set_lost()
assert not Lead.search([('email_from', '=', 'alice@example.com')])

PREFIX = 'agentreview-synthetic-'   # 22 characters + 20 hex = 42, inside the token field's size
service = env['iap.service'].search([('technical_name', '=', 'odoo_ai')], limit=1)
assert service, "no odoo_ai IAP service: is the ai module installed?"
if not env['iap.account'].sudo().search([('service_id', '=', service.id)]):
    env['iap.account'].sudo().create({'service_id': service.id})
for acc in env['iap.account'].sudo().search([]):
    acc.account_token = PREFIX + secrets.token_hex(10)
assert all((a.account_token or '').startswith(PREFIX) for a in env['iap.account'].sudo().search([]))
assert not env['ir.config_parameter'].sudo().get_str('ai.endpoint'), "the template must not carry ai.endpoint"
env.cr.commit()
print("FIXTURE OK (Odoo 20)")
