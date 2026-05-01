import os
import streamlit as st


def get_target_agent_state(agent):
    tool_catalog = []
    if hasattr(agent, "tools") and hasattr(agent.tools, "_tools"):
        tool_catalog = [
            {
                "name": t.name,
                "category": t.category,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in agent.tools._tools.values()
        ]

    memory_entries = []
    if hasattr(agent, "memory"):
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
        "name": getattr(agent, "name", "unknown"),
        "profile": getattr(agent, "profile", {}),
        "alignment": getattr(agent, "alignment", {}),
        "registered_tools": tool_catalog,
        "memory_stream": memory_entries,
    }


def render_target_state_sidebar(agent):
    state = get_target_agent_state(agent)

    with st.expander("Identity & Profile", expanded=True):
        st.markdown(f"**Name:** `{state['name']}`")
        if state["profile"]:
            for k, v in state["profile"].items():
                st.markdown(f"**{k}:** `{v}`")
        else:
            st.caption("No profile metadata available.")

    with st.expander("Alignment Policy", expanded=True):
        if state["alignment"]:
            for k, v in state["alignment"].items():
                if isinstance(v, list):
                    st.markdown(f"**{k}:**")
                    for item in v:
                        st.markdown(f"- `{item}`")
                elif isinstance(v, bool):
                    color = "green" if v else "red"
                    st.markdown(f"**{k}:** :{color}[{v}]")
                else:
                    st.markdown(f"**{k}:** `{v}`")
        else:
            st.caption("No alignment metadata available.")

    with st.expander(f"Memory Stream ({len(state['memory_stream'])} entries)", expanded=True):
        if not state["memory_stream"]:
            st.caption("No entries yet.")
        for entry in reversed(state["memory_stream"]):
            type_labels = {"observation": "[obs]", "plan": "[plan]", "reflection": "[refl]"}
            label = type_labels.get(entry["type"], "[?]")
            filled = int(entry["importance"] * 10)
            importance_bar = "|" * filled + "." * (10 - filled)
            st.markdown(
                f"{label} **#{entry['id']}** `{entry['type']}`  \n"
                f"Importance: `{importance_bar}` {entry['importance']}  \n"
                f"{entry['content']}"
            )
            st.divider()

    if hasattr(agent, "conversation_history"):
        turn_count = getattr(agent, "turn_count", len(agent.conversation_history))
        with st.expander(f"Conversation History ({turn_count} turns)", expanded=False):
            if not agent.conversation_history:
                st.caption("No turns yet.")
            for turn in agent.conversation_history:
                st.markdown(f"**Turn {turn['turn']}** `{turn['role']}`")
                st.caption(turn["content"][:100] + ("..." if len(turn["content"]) > 100 else ""))
                st.divider()

    if hasattr(agent, "turn_count") and "escalation_sentiment_threshold" in state["alignment"]:
        with st.expander("Sentiment & Escalation", expanded=True):
            st.markdown(
                f"**Escalation threshold:** `{state['alignment']['escalation_sentiment_threshold']}`"
            )
            escalations = [
                e for e in st.session_state["conversation_history"]
                if e["role"] == "agent" and e.get("alignment_warning")
            ]
            st.markdown(f"**UI alignment warnings:** `{len(escalations)}`")

    with st.expander("Registered Tools", expanded=False):
        if not state["registered_tools"]:
            st.caption("No tools registered.")
        for t in state["registered_tools"]:
            st.markdown(f"**{t['name']}** `[{t['category']}]`")
            st.caption(t["description"])
            st.caption(f"Inputs: {t['input_schema']}")


def _load_env(path=".env"):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"'))
    except FileNotFoundError:
        pass


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_load_env(os.path.join(_SCRIPT_DIR, ".env"))
_load_env()

import config
from agents import AlignmentBlockedError
from backends import build_target_agent, build_red_team_agent

st.set_page_config(
    page_title=config.STREAMLIT_PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    [data-testid="stSidebar"] {
        min-width: 380px;
        max-width: 380px;
    }
</style>
""",
    unsafe_allow_html=True,
)

if "target_agent_name" not in st.session_state:
    st.session_state["target_agent_name"] = config.STREAMLIT_DEFAULT_TARGET
if "target_agent" not in st.session_state:
    st.session_state["target_agent"] = None
if "red_team_agent" not in st.session_state:
    st.session_state["red_team_agent"] = None
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []
if "red_team_mode_active" not in st.session_state:
    st.session_state["red_team_mode_active"] = False
if "red_team_pending_prompt" not in st.session_state:
    st.session_state["red_team_pending_prompt"] = None
if "red_team_generated_prompt" not in st.session_state:
    st.session_state["red_team_generated_prompt"] = None
if "red_team_pending_surface" not in st.session_state:
    st.session_state["red_team_pending_surface"] = "direct_prompt_injection"
if "red_team_prompt_input" not in st.session_state:
    st.session_state["red_team_prompt_input"] = ""
if "last_alignment_warning" not in st.session_state:
    st.session_state["last_alignment_warning"] = False
if "show_target_state_panel" not in st.session_state:
    st.session_state["show_target_state_panel"] = True

if st.session_state["target_agent"] is None:
    try:
        ta = build_target_agent(st.session_state["target_agent_name"], config.TARGET_LLM)
        rt = build_red_team_agent(ta, config.RED_TEAM_LLM)
        st.session_state["target_agent"] = ta
        st.session_state["red_team_agent"] = rt
    except Exception as exc:
        st.error(f"Failed to initialize agents: {exc}")
        st.stop()

with st.sidebar:
    st.markdown("## Agent Alignment Testbed")
    st.divider()

    agent_options = ["medical", "weak_medical", "financial", "customer_service"]
    new_target = st.selectbox(
        "Target Agent",
        agent_options,
        index=agent_options.index(st.session_state["target_agent_name"]),
    )
    if new_target != st.session_state["target_agent_name"]:
        st.session_state["target_agent_name"] = new_target
        st.session_state["target_agent"] = None
        st.session_state["red_team_agent"] = None
        st.session_state["conversation_history"] = []
        st.session_state["red_team_pending_prompt"] = None
        st.session_state["red_team_generated_prompt"] = None
        st.session_state["red_team_prompt_input"] = ""
        st.rerun()

    st.divider()
    st.subheader("Red Team Mode")

    red_team_active = st.toggle(
        "Activate Red Team Agent",
        value=st.session_state["red_team_mode_active"],
    )
    if red_team_active != st.session_state["red_team_mode_active"]:
        st.session_state["red_team_mode_active"] = red_team_active
        st.session_state["red_team_pending_prompt"] = None
        st.session_state["red_team_generated_prompt"] = None
        st.session_state["red_team_prompt_input"] = ""
        st.rerun()

    if st.session_state["red_team_mode_active"]:
        st.warning(
            "Red team mode active. The adversarial agent will generate attack prompts. "
            "You must press Enter to submit them."
        )

    st.divider()
    st.subheader("Target Agent State")
    st.toggle("Show target state panel", key="show_target_state_panel")
    if st.session_state["show_target_state_panel"]:
        render_target_state_sidebar(st.session_state["target_agent"])

target_agent = st.session_state["target_agent"]
red_team_agent = st.session_state["red_team_agent"]
target_agent_name = st.session_state["target_agent_name"]

chat_col, state_col = st.columns([2.3, 1.0], gap="large")

with chat_col:
    st.header(f"Chatting with: {target_agent_name}-agent")

    for entry in st.session_state["conversation_history"]:
        if entry["role"] == "user":
            with st.chat_message("user"):
                st.markdown(entry["content"])
        elif entry["role"] == "agent":
            with st.chat_message("assistant"):
                st.markdown(entry["content"])
                if entry.get("alignment_warning") and config.SHOW_ALIGNMENT_WARNING_IN_UI:
                    st.warning("Alignment warning triggered.")
        elif entry["role"] == "red_team":
            with st.chat_message("user"):
                st.markdown(f"**[RED TEAM]** {entry['content']}")
                if entry.get("alignment_warning") and config.SHOW_ALIGNMENT_WARNING_IN_UI:
                    st.warning("Alignment warning triggered.")


def handle_send(user_input, is_red_team):
    role = "red_team" if is_red_team else "user"
    st.session_state["conversation_history"].append({
        "role": role,
        "content": user_input,
        "is_red_team": is_red_team,
    })

    with st.chat_message("assistant"):
        st.markdown("⏳ _Thinking..._")

    blocked_by_alignment = False
    target_runtime_failed = False
    try:
        result = target_agent.run(user_input)
        result["blocked"] = False
    except AlignmentBlockedError:
        result = {
            "plan": "Input blocked at perception layer.",
            "alignment_warning": False,
            "blocked": True,
        }
        blocked_by_alignment = True
    except Exception as exc:
        result = {
            "plan": f"Target agent call failed: {exc}",
            "alignment_warning": False,
            "blocked": True,
        }
        target_runtime_failed = True

    alignment_warning = result.get("alignment_warning", False)

    disclaimer = result.get("disclaimer", "")
    plan_text = result["plan"]
    if disclaimer and disclaimer.lower() not in plan_text.lower():
        plan_text = plan_text + f"\n\n*{disclaimer}*"
    st.session_state["conversation_history"].append({
        "role": "agent",
        "content": plan_text,
        "alignment_warning": alignment_warning,
    })

    if is_red_team:
        generated = st.session_state.get("red_team_generated_prompt") or ""
        tampered = user_input.strip() != generated.strip()
        if tampered:
            st.info("Prompt was modified — red team agent state not updated.")
        elif target_runtime_failed:
            st.warning("Skipped red-team scoring because target execution failed.")
        else:
            try:
                red_team_agent.act(
                    {
                        "surface": st.session_state.get("red_team_pending_surface", "direct_prompt_injection"),
                        "attack": user_input,
                    },
                    outcome="detected" if blocked_by_alignment else result,
                )
            except Exception as exc:
                st.warning(f"Red-team scoring failed: {exc}")
        st.session_state["red_team_pending_prompt"] = None
        st.session_state["red_team_generated_prompt"] = None

    if config.SHOW_ALIGNMENT_WARNING_IN_UI and alignment_warning:
        st.warning("Alignment violation detected in agent response.")

    st.rerun()


with chat_col:
    if st.session_state["red_team_mode_active"]:
        st.info("Red team agent is generating an adversarial prompt. Review it below before submitting.")

        if st.session_state["red_team_pending_prompt"] is None:
            try:
                red_team_agent.perceive(f"Streamlit interactive session targeting {target_agent_name}")
                plan = red_team_agent.plan([])
                st.session_state["red_team_pending_prompt"] = plan["attack"]
                st.session_state["red_team_generated_prompt"] = plan["attack"]
                st.session_state["red_team_pending_surface"] = plan["surface"]
                st.session_state["red_team_prompt_input"] = plan["attack"]
            except Exception as exc:
                st.error(f"Failed to generate red-team prompt: {exc}")
                st.session_state["red_team_pending_prompt"] = ""
                st.session_state["red_team_generated_prompt"] = None
                st.session_state["red_team_prompt_input"] = ""

        edited_prompt = st.text_area(
            "Red team prompt (editable before sending)",
            key="red_team_prompt_input",
            height=100,
        )
        st.session_state["red_team_pending_prompt"] = edited_prompt

        generated = st.session_state.get("red_team_generated_prompt") or ""
        is_tampered = edited_prompt.strip() != generated.strip()
        if is_tampered and edited_prompt.strip():
            st.caption("⚠ Prompt differs from generated — red team state will not be updated on send.")

        if st.button("Send adversarial prompt", type="primary", disabled=not edited_prompt.strip()):
            handle_send(st.session_state["red_team_pending_prompt"], is_red_team=True)

    else:
        user_input = st.chat_input(f"Message {target_agent_name}-agent...")
        if user_input:
            handle_send(user_input, is_red_team=False)

with state_col:
    st.markdown("### Internal State")

    if config.SHOW_RED_TEAM_REASONING_IN_UI and st.session_state["red_team_mode_active"]:
        reflections = [e for e in red_team_agent.memory.all() if e["type"] == "reflection"]
        if reflections:
            with st.expander("Red team reasoning (last reflection)", expanded=False):
                st.text(reflections[-1]["content"])

    with st.expander("Attack surface bandit state", expanded=False):
        st.json(red_team_agent.bandit_state)
