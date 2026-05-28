def report_agent(query, answer, risk_report):

    report = f"""
=============================
FRAUD INVESTIGATION REPORT
=============================

QUERY:
{query}

-----------------------------

AI ANALYSIS:
{answer}

-----------------------------

RISK REPORT:
{risk_report}

=============================
"""

    return report