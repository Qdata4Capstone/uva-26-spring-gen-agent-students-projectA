"""
EnvPilot Agent Prompts — System prompts for each phase of the workflow.
"""

SYSTEM_PROMPT = """\
You are an expert Python developer producing code that runs correctly against \
the user's exact installed package versions. Follow each task prompt's output \
format precisely.\
"""

ANALYSIS_PROMPT = """\
Analyze the following task and identify all required third-party Python packages \
and critical APIs. Assign an Uncertainty Score (0-100) based on your confidence \
in the local environment and API stability.

Respond with ONLY a valid JSON object (no markdown fencing):
{{
    "identified_packages": ["package1", "package2"],
    "uncertainty_score": <0-100>,
    "reasoning": "brief explanation",
    "critical_apis": ["api1", "api2"]
}}

Task: {task_description}
"""

ENV_PROBE_ANALYSIS_PROMPT = """\
The environment has been probed. Here are the results:

{env_info}

Compare these installed versions against the task requirements. Identify any \
version mismatches or missing packages. Update the uncertainty score if needed.

Respond with ONLY a valid JSON object (no markdown fencing):
{{
    "version_issues": ["issue1", "issue2"],
    "missing_packages": ["pkg1"],
    "updated_uncertainty_score": <0-100>,
    "adjusted_plan": "brief description of adjustments"
}}
"""

KB_ANALYSIS_PROMPT = """\
The knowledge base returned these results for the identified packages:

{kb_results}

Environment info:
{env_info}

Determine if the KB has sufficient coverage ("API Drift" means outdated info). \
If there are gaps or the KB lacks rules for installed package versions, \
indicate that a web search is needed.

Respond with ONLY a valid JSON object (no markdown fencing):
{{
    "kb_has_gaps": true/false,
    "gap_details": ["what's missing"],
    "search_queries": ["query1 if gaps exist"],
    "updated_uncertainty_score": <0-100>
}}
"""

PLAN_PROMPT = """\
You have probed the target environment and looked up breaking-change rules. \
Now decide which exact dotted-path APIs you intend to use in the final code, \
choosing alternatives that EXIST in the env's installed package versions.

Rules:
- Output dotted paths only (e.g. "seaborn.distplot", "pandas.DataFrame.ffill").
- For each task-relevant operation, pick ONE concrete API to use.
- If KB / web indicates an API was removed in the env's version, pick its \
replacement instead (e.g. seaborn 0.10.1 has no histplot → use distplot).
- Don't propose APIs you have no reason to believe exist.

Task:
{task_description}

Environment:
{env_info}

Originally identified APIs (from task analysis, may include broken ones):
{critical_apis}

KB rules (breaking changes we know about):
{kb_results}

Web search findings (if any):
{web_results}

{previous_attempt}

Respond with ONLY a valid JSON object (no markdown fencing):
{{
    "proposed_apis": ["library.symbol", "library.Class.method", ...],
    "reasoning": "one sentence why these APIs",
    "uncertainty_score": <0-100>
}}
"""


PREFLIGHT_PROMPT = """\
Based on the analysis so far, write a minimal smoke test script that verifies \
the most "at-risk" part of the planned code. The script should:
1. Import the critical packages
2. Test the specific API calls that may have breaking changes (CALL the API,
   don't just import — we need the actual breaking-change error to surface)
3. Print a clear SUCCESS or FAILURE message

Environment info:
{env_info}

KB findings:
{kb_results}

Web search results (if any):
{web_results}

Task: {task_description}

Output the smoke test as a single fenced Python code block. Do NOT use JSON;
embed the full code inside the fence so quotes and backslashes don't need
to be escaped. Format exactly:

```python
# your smoke test here
```
"""

GENERATION_PROMPT = """\
Generate the final code for the task.

The planning + preflight steps have determined which APIs are safe to use \
in this environment. Use the corresponding API SYMBOLS (functions / classes / \
methods) — they have been verified to exist. Idiomatic aliasing is fine \
(e.g. `import numpy as np` → `np.array(...)`); do NOT write `import \
package.submodule.attr as alias` for non-module attributes (e.g. \
`matplotlib.colormaps` is a singleton, not a module — write \
`import matplotlib as mpl; mpl.colormaps.register(...)`).

Verified safe APIs (treat as a guide, not a literal substitution string):
{proposed_apis}

Avoid switching to a different alternative API than what was proposed \
above, and avoid modifying code paths unrelated to these APIs.

Environment:
{env_info}

KB Rules (relevant breaking changes):
{kb_results}

Preflight test result (PASS/FAIL per API):
{preflight_result}

Task: {task_description}

Output the complete, production-ready Python code as a single fenced Python \
code block. Do NOT use JSON. Format exactly:

```python
# your final code here (imports + function definition with body)
```

NOTES: one short line about implementation choices (optional).
"""

KB_UPDATE_PROMPT = """\
Based on the web search results below, extract any new breaking change rules \
that should be added to the Knowledge Base.

Web search results:
{web_results}

For each new rule, provide the structured data. Respond with ONLY a valid JSON \
object (no markdown fencing):
{{
    "rules_to_upsert": [
        {{
            "rule_id": "library-symbol-change",
            "library": "package_name",
            "removed_in": "version",
            "pattern_type": "attribute|import|method_call|method_access",
            "module_path": "module.path",
            "symbol": "function_or_attr_name",
            "old_api": "old usage example",
            "new_api": "new usage example",
            "error_type": "AttributeError|ImportError|TypeError",
            "description": "human readable explanation",
            "severity": "error|warning"
        }}
    ]
}}

If no new rules can be extracted, return: {{"rules_to_upsert": []}}
"""
