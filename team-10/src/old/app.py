import os
import re

import streamlit as st
import openai

from agents import MemoryStream, MedicalAgent, FinancialAgent, CustomerServiceAgent
from tools import (
    ToolRegistry,
    DrugLookupTool, DosageCalculatorTool, SymptomSummarizerTool,
    MarketDataLookupTool, CompoundInterestCalculatorTool, PortfolioSummarizerTool,
    FAQLookupTool, TicketClassifierTool, ConversationSummarizerTool,
    MedicalRAGTool, FinancialRAGTool, CustomerServiceRAGTool,
)


def load_env(path=".env"):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


load_env()

_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def llm_call(messages):
    response = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content


def build_medical_agent():
    memory = MemoryStream()
    registry = ToolRegistry()
    registry.register(DrugLookupTool())
    registry.register(DosageCalculatorTool())
    registry.register(SymptomSummarizerTool())
    registry.register(MedicalRAGTool())
    return MedicalAgent(name="MedAgent-v1", llm_call=llm_call, tools=registry, memory=memory)


def build_financial_agent():
    memory = MemoryStream()
    registry = ToolRegistry()
    registry.register(MarketDataLookupTool())
    registry.register(CompoundInterestCalculatorTool())
    registry.register(PortfolioSummarizerTool())
    registry.register(FinancialRAGTool())
    return FinancialAgent(name="FinAgent-v1", llm_call=llm_call, tools=registry, memory=memory)


def build_cs_agent():
    memory = MemoryStream()
    registry = ToolRegistry()
    registry.register(FAQLookupTool())
    registry.register(TicketClassifierTool())
    registry.register(ConversationSummarizerTool())
    registry.register(CustomerServiceRAGTool())

    def escalation_callback(message):
        if "escalation_log" not in st.session_state:
            st.session_state.escalation_log = []
        st.session_state.escalation_log.append(message)

    return CustomerServiceAgent(
        name="CSAgent-v1",
        llm_call=llm_call,
        tools=registry,
        memory=memory,
        escalation_callback=escalation_callback,
    )


AGENT_CONFIGS = {
    "MedAgent-v1": {
        "icon": "[Med]",
        "label": "Medical Assistant",
        "caption": "Clinician-supervised · drug lookup · dosage calc · symptom triage",
        "placeholder": "Ask a medical question... (e.g. What is metformin used for?)",
        "builder": build_medical_agent,
    },
    "FinAgent-v1": {
        "icon": "[Fin]",
        "label": "Financial Assistant",
        "caption": "User-supervised · market data · compound interest · portfolio summary",
        "placeholder": "Ask a financial question... (e.g. Tell me about Apple stock)",
        "builder": build_financial_agent,
    },
    "CSAgent-v1": {
        "icon": "[CS]",
        "label": "Customer Service",
        "caption": "Escalation-supervised · FAQ lookup · ticket classification · sentiment detection",
        "placeholder": "Ask a support question... (e.g. What is your return policy?)",
        "builder": build_cs_agent,
    },
}


def get_agent_state(agent):
    tool_catalog = [
        {
            "name": t.name,
            "category": t.category,
            "description": t.description,
            "input_schema": t.input_schema,
        }
        for t in agent.tools._tools.values()
    ]
    memory_entries = [
        {
            "id": e["id"],
            "type": e["type"],
            "importance": e["importance"],
            "timestamp": round(e["timestamp"], 2),
            "content": e["content"][:120] + ("..." if len(e["content"]) > 120 else ""),
        }
        for e in agent.memory.all()
    ]
    return {
        "name": agent.name,
        "profile": agent.profile,
        "alignment": agent.alignment,
        "registered_tools": tool_catalog,
        "memory_stream": memory_entries,
    }


def parse_tool_call(plan_str):
    tool_match = re.search(r"TOOL:\s*(\S+)", plan_str)
    inputs_match = re.search(r"INPUTS:\s*(.+)", plan_str)
    if not tool_match:
        return None, None
    tool_name = tool_match.group(1).strip()
    inputs = {}
    if inputs_match:
        raw = inputs_match.group(1)
        for m in re.finditer(r"(\w+)=(.+?)(?=,\s*\w+=|$)", raw):
            inputs[m.group(1).strip()] = m.group(2).strip()
    return tool_name, inputs


def render_sidebar(agent, agent_key):
    cfg = AGENT_CONFIGS[agent_key]
    st.sidebar.markdown(f"## {cfg['icon']} Agent Internal State")
    state = get_agent_state(agent)

    with st.sidebar.expander("Identity & Profile", expanded=True):
        st.markdown(f"**Name:** `{state['name']}`")
        for k, v in state["profile"].items():
            st.markdown(f"**{k}:** `{v}`")

    with st.sidebar.expander("Alignment Policy", expanded=True):
        for k, v in state["alignment"].items():
            if isinstance(v, list):
                st.markdown(f"**{k}:**")
                for item in v:
                    st.markdown(f"  - `{item}`")
            elif isinstance(v, bool):
                color = "green" if v else "red"
                st.markdown(f"**{k}:** :{color}[{v}]")
            else:
                st.markdown(f"**{k}:** `{v}`")

    with st.sidebar.expander(f"Memory Stream ({len(state['memory_stream'])} entries)", expanded=True):
        if not state["memory_stream"]:
            st.caption("No entries yet.")
        for e in reversed(state["memory_stream"]):
            type_labels = {"observation": "[obs]", "plan": "[plan]", "reflection": "[refl]"}
            label = type_labels.get(e["type"], "[?]")
            filled = int(e["importance"] * 10)
            importance_bar = "|" * filled + "." * (10 - filled)
            st.markdown(
                f"{label} **#{e['id']}** `{e['type']}`  \n"
                f"Importance: `{importance_bar}` {e['importance']}  \n"
                f"{e['content']}"
            )
            st.divider()

    if hasattr(agent, "conversation_history"):
        with st.sidebar.expander(f"Conversation History ({agent.turn_count} turns)", expanded=False):
            if not agent.conversation_history:
                st.caption("No turns yet.")
            for turn in agent.conversation_history:
                st.markdown(f"**Turn {turn['turn']}** `{turn['role']}`")
                st.caption(turn["content"][:100] + ("..." if len(turn["content"]) > 100 else ""))
                st.divider()

    if hasattr(agent, "turn_count"):
        with st.sidebar.expander("Sentiment & Escalation", expanded=True):
            escalation_log = st.session_state.get("escalation_log", [])
            st.markdown(f"**Escalation threshold:** `{agent.alignment['escalation_sentiment_threshold']}`")
            if escalation_log:
                st.markdown(f"**Escalations fired:** `{len(escalation_log)}`")
                for log_entry in escalation_log[-3:]:
                    st.markdown(f"[!] {log_entry}")
            else:
                st.markdown("**Escalations fired:** `0`")

    with st.sidebar.expander("Registered Tools", expanded=False):
        for t in state["registered_tools"]:
            st.markdown(f"**{t['name']}** `[{t['category']}]`")
            st.caption(t["description"])
            st.caption(f"Inputs: {t['input_schema']}")


st.set_page_config(
    page_title="Agent Demo",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: 380px;
        max-width: 380px;
        background-color: #0f1117;
    }
    [data-testid="stSidebar"] * { font-size: 0.82rem; }
    .user-bubble {
        background: #2b5be0;
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 4px 0 4px 15%;
        display: inline-block;
        max-width: 85%;
        float: right;
        clear: both;
        word-wrap: break-word;
    }
    .agent-bubble {
        background: #1e2130;
        color: #e8e8e8;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 4px 15% 4px 0;
        display: inline-block;
        max-width: 85%;
        float: left;
        clear: both;
        word-wrap: break-word;
        line-height: 1.6;
    }
    .tool-badge {
        background: #0d3b2e;
        border: 1px solid #1a6b52;
        border-radius: 8px;
        padding: 8px 12px;
        margin-top: 10px;
        font-size: 0.82rem;
        color: #4ecca3;
    }
    .warning-badge {
        background: #3b1a0d;
        border: 1px solid #c0510a;
        border-radius: 8px;
        padding: 6px 12px;
        margin-top: 8px;
        font-size: 0.8rem;
        color: #f09060;
    }
    .disclaimer-badge {
        background: #1a1a2e;
        border-left: 3px solid #4a6fa5;
        padding: 6px 12px;
        margin-top: 8px;
        font-size: 0.78rem;
        color: #8899bb;
        border-radius: 0 6px 6px 0;
    }
    .error-bubble {
        background: #2e0d0d;
        border: 1px solid #8b1a1a;
        color: #ff6b6b;
        border-radius: 18px;
        padding: 12px 16px;
        margin: 4px 0;
        clear: both;
        word-wrap: break-word;
    }
    .chat-wrap { overflow: hidden; margin-bottom: 8px; }
    .clearfix { clear: both; }
    div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
    .agent-name-med { color: #7ec8e3; font-weight: bold; }
    .agent-name-fin { color: #f0c060; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

if "agents" not in st.session_state:
    st.session_state.agents = {
        "MedAgent-v1": build_medical_agent(),
        "FinAgent-v1": build_financial_agent(),
        "CSAgent-v1": build_cs_agent(),
    }
    st.session_state.messages = {"MedAgent-v1": [], "FinAgent-v1": [], "CSAgent-v1": []}
    st.session_state.escalation_log = []
    st.session_state.active_agent = "MedAgent-v1"

agent_labels = {
    "MedAgent-v1": "Medical Assistant",
    "FinAgent-v1": "Financial Assistant",
    "CSAgent-v1": "Customer Service",
}
selected_label = st.sidebar.radio(
    "Viewing state for:",
    list(agent_labels.values()),
    index=list(agent_labels.keys()).index(st.session_state.active_agent),
    label_visibility="visible",
    horizontal=False,
)
st.session_state.active_agent = [k for k, v in agent_labels.items() if v == selected_label][0]

render_sidebar(
    st.session_state.agents[st.session_state.active_agent],
    st.session_state.active_agent,
)

agent_tabs = st.tabs(["Medical Assistant", "Financial Assistant", "Customer Service"])

for tab_idx, (agent_key, tab) in enumerate(zip(["MedAgent-v1", "FinAgent-v1", "CSAgent-v1"], agent_tabs)):
    with tab:
        cfg = AGENT_CONFIGS[agent_key]
        agent = st.session_state.agents[agent_key]
        messages = st.session_state.messages[agent_key]

        st.markdown(f"## {cfg['label']}")
        st.caption(cfg["caption"] + " · Powered by GPT-4o-mini")
        st.divider()

        chat_container = st.container()

        with chat_container:
            for msg in messages:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-wrap"><div class="user-bubble">{msg["content"]}</div></div>'
                        '<div class="clearfix"></div>',
                        unsafe_allow_html=True,
                    )
                elif msg["role"] == "agent":
                    result = msg["result"]
                    tool_name, tool_inputs = parse_tool_call(result.get("plan", ""))
                    action = result.get("action_result", {})
                    action_text = action.get("result") or action.get("error") or ""

                    tool_html = ""
                    if tool_name:
                        inputs_display = ", ".join(f"{k}={v}" for k, v in (tool_inputs or {}).items())
                        result_preview = str(action_text)[:300] + ("..." if len(str(action_text)) > 300 else "")
                        tool_html = (
                            f'<div class="tool-badge">'
                            f'[tool] <b>Tool called:</b> <code>{tool_name}</code>'
                            + (f'  &middot;  <b>Inputs:</b> <code>{inputs_display}</code>' if inputs_display else '')
                            + f'<br><b>Result:</b> {result_preview}'
                            f'</div>'
                        )

                    escalation_html = ""
                    if result.get("escalation_triggered"):
                        escalation_html = (
                            '<div class="warning-badge"><b>Escalation triggered:</b> '
                            f'{result.get("escalation_reason", "Negative sentiment detected")} '
                            f'(score: {result.get("sentiment_score", "n/a")})</div>'
                        )
                    elif "sentiment_score" in result:
                        score = result["sentiment_score"]
                        bar_filled = int((score + 1.0) / 2.0 * 10)
                        bar = "|" * bar_filled + "." * (10 - bar_filled)
                        escalation_html = (
                            f'<div class="disclaimer-badge">Sentiment: <code>{bar}</code> {score:.2f}</div>'
                        )

                    disclaimer_html = (
                        f'<div class="disclaimer-badge">{result.get("disclaimer", "")}</div>'
                        if result.get("disclaimer") else ""
                    )

                    main_text = (action_text if (action_text and not action.get("error")) else result.get("plan", ""))

                    name_class = "agent-name-med" if agent_key == "MedAgent-v1" else "agent-name-fin"
                    st.markdown(
                        f'<div class="chat-wrap"><div class="agent-bubble">'
                        f'<span class="{name_class}">{agent_key}</span><br><br>'
                        f'{main_text}'
                        f'{tool_html}'
                        f'{escalation_html}'
                        f'{disclaimer_html}'
                        f'</div></div><div class="clearfix"></div>',
                        unsafe_allow_html=True,
                    )

                    with st.expander("View full output (plan · reflection · raw)", expanded=False):
                        t1, t2, t3 = st.tabs(["Plan", "Reflection", "Raw JSON"])
                        with t1:
                            st.text(result.get("plan", ""))
                        with t2:
                            st.text(result.get("reflection", ""))
                        with t3:
                            st.json({k: v for k, v in result.items()})

                elif msg["role"] == "error":
                    st.markdown(
                        f'<div class="error-bubble"><b>Alignment Violation</b><br>{msg["content"]}</div>'
                        '<div class="clearfix"></div>',
                        unsafe_allow_html=True,
                    )

        st.divider()

        with st.form(key=f"chat_form_{agent_key}", clear_on_submit=True):
            col1, col2 = st.columns([9, 1])
            with col1:
                user_input = st.text_area(
                    "Message",
                    placeholder=cfg["placeholder"],
                    height=80,
                    label_visibility="collapsed",
                    key=f"input_{agent_key}",
                )
            with col2:
                submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted and user_input.strip():
            st.session_state.active_agent = agent_key
            st.session_state.messages[agent_key].append({"role": "user", "content": user_input.strip()})
            try:
                result = agent.run(user_input.strip())
                st.session_state.messages[agent_key].append({"role": "agent", "result": result})
            except ValueError as exc:
                st.session_state.messages[agent_key].append({"role": "error", "content": str(exc)})
            st.rerun()

active_key = st.session_state.active_agent
active_cfg = AGENT_CONFIGS[active_key]
st.sidebar.divider()
if st.sidebar.button(f"Reset {active_cfg['label']}", key="reset_active", use_container_width=True):
    st.session_state.agents[active_key] = active_cfg["builder"]()
    st.session_state.messages[active_key] = []
    if active_key == "CSAgent-v1":
        st.session_state.escalation_log = []
    st.rerun()
