TARGET_AGENT = "medical"
RED_TEAM_LLM = "openai"
TARGET_LLM = "openai"
MAX_TURNS = 10
VIOLATION_STOPS_EXPERIMENT = True
ALLOWED_ATTACK_SURFACES = [
    "direct_prompt_injection",
    "goal_hijacking",
    "alignment_bypass",
    "tool_misuse",
    "memory_poisoning",
    "multi_turn_escalation",
    "obfuscation",
]
CURIOSITY_BONUS_WEIGHT = 0.3
EXPLOIT_VS_EXPLORE_EPSILON = 0.25
MAX_ATTACKS_PER_CAMPAIGN = 20
EXPERIMENT_OUTPUT_DIR = "experiments/"
EXPERIMENT_LOG_FILENAME = "campaign_log.json"
STREAMLIT_PAGE_TITLE = "Agent Alignment Testbed"
STREAMLIT_DEFAULT_TARGET = "medical"
SHOW_ALIGNMENT_WARNING_IN_UI = True
SHOW_RED_TEAM_REASONING_IN_UI = True
LOG_ALL_MEMORY_ENTRIES = False
ANTHROPIC_MODEL = "claude-sonnet-4-20250514" # not using htis anymore
ANTHROPIC_MAX_TOKENS = 1024
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MAX_TOKENS = 1024
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
VLLM_MAX_TOKENS = 1024
BASELINE_N_PROBES_PER_CATEGORY = 5
BASELINE_OUTPUT_DIR = "experiments/baseline/"
BASELINE_LOG_FILENAME = "baseline_log.json"
BASELINE_LLM_JUDGE_BACKEND = "stub"
BASELINE_JUDGE_TEMPERATURE = 0.0
BASELINE_CATEGORIES = [
    "scope_refusal",
    "tool_restriction",
    "disclaimer_compliance",
    "escalation_resistance",
    "identity_stability",
    "instruction_following_under_pressure",
]
TRANSFER_OUTPUT_DIR = "experiments/transfer/"
TRANSFER_LOG_FILENAME = "transfer_log.json"
TRANSFER_TOP_N_ATTACKS = 5
FRR_OUTPUT_DIR = "experiments/frr/"
FRR_LOG_FILENAME = "frr_log.json"
FRR_N_PROBES_PER_AGENT = 8
REPORT_OUTPUT_DIR = "experiments/reports/"
MATPLOTLIB_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_DPI = 150
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 6
