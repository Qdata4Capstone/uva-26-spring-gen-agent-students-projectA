import streamlit as st

import config
from backends import build_target_agent, build_red_team_agent
from experiments import run_experiment, summarize_experiment

st.set_page_config(
    page_title=config.STREAMLIT_PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
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
if "red_team_pending_surface" not in st.session_state:
    st.session_state["red_team_pending_surface"] = "direct_prompt_injection"
if "last_alignment_warning" not in st.session_state:
    st.session_state["last_alignment_warning"] = False
if "experiment_log" not in st.session_state:
    st.session_state["experiment_log"] = None
if "experiment_running" not in st.session_state:
    st.session_state["experiment_running"] = False

if st.session_state["target_agent"] is None:
    ta = build_target_agent(st.session_state["target_agent_name"], config.TARGET_LLM)
    rt = build_red_team_agent(ta, config.RED_TEAM_LLM)
    st.session_state["target_agent"] = ta
    st.session_state["red_team_agent"] = rt

with st.sidebar:
    st.markdown("## Agent Alignment Testbed")
    st.divider()

    agent_options = ["medical", "financial", "customer_service"]
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
        st.rerun()

    if st.session_state["red_team_mode_active"]:
        st.warning(
            "Red team mode active. The adversarial agent will generate attack prompts. "
            "You must press Enter to submit them."
        )

    st.divider()
    st.subheader("Run Automated Experiment")

    n_turns_input = st.number_input(
        "Number of turns",
        min_value=1,
        max_value=50,
        value=config.MAX_TURNS,
    )
    stop_on_violation_input = st.checkbox(
        "Stop on first violation",
        value=config.VIOLATION_STOPS_EXPERIMENT,
    )

    if st.button("Run Experiment", use_container_width=True):
        st.session_state["experiment_running"] = True
        log = run_experiment(
            target_agent_name=st.session_state["target_agent_name"],
            n_turns=n_turns_input,
            stop_on_violation=stop_on_violation_input,
        )
        st.session_state["experiment_log"] = log
        st.session_state["experiment_running"] = False

    if st.session_state["experiment_running"]:
        st.info("Experiment running...")

    if st.session_state["experiment_log"] is not None:
        st.text(summarize_experiment(st.session_state["experiment_log"]))
        with st.expander("Full experiment log", expanded=False):
            st.json(st.session_state["experiment_log"])

target_agent = st.session_state["target_agent"]
red_team_agent = st.session_state["red_team_agent"]
target_agent_name = st.session_state["target_agent_name"]

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

    blocked = False
    try:
        result = target_agent.run(user_input)
        result["blocked"] = False
    except ValueError:
        result = {
            "plan": "Input blocked at perception layer.",
            "alignment_warning": False,
            "blocked": True,
        }
        blocked = True

    alignment_warning = result.get("alignment_warning", False)

    st.session_state["conversation_history"].append({
        "role": "agent",
        "content": result["plan"],
        "alignment_warning": alignment_warning,
    })

    if is_red_team:
        red_team_agent.evaluator.evaluate(user_input, result)
        red_team_agent.act({
            "surface": st.session_state.get("red_team_pending_surface", "direct_prompt_injection"),
            "attack": user_input,
        })
        st.session_state["red_team_pending_prompt"] = None

    if config.SHOW_ALIGNMENT_WARNING_IN_UI and alignment_warning:
        st.warning("Alignment violation detected in agent response.")

    st.rerun()


if st.session_state["red_team_mode_active"]:
    st.info("Red team agent is generating an adversarial prompt. Review it below before submitting.")

    if st.session_state["red_team_pending_prompt"] is None:
        red_team_agent.perceive(f"Streamlit interactive session targeting {target_agent_name}")
        plan = red_team_agent.plan([])
        st.session_state["red_team_pending_prompt"] = plan["attack"]
        st.session_state["red_team_pending_surface"] = plan["surface"]

    edited_prompt = st.text_area(
        "Red team prompt (editable before sending)",
        value=st.session_state["red_team_pending_prompt"],
        key="red_team_prompt_input",
        height=100,
    )
    st.session_state["red_team_pending_prompt"] = edited_prompt

    if st.button("Send adversarial prompt", type="primary"):
        handle_send(st.session_state["red_team_pending_prompt"], is_red_team=True)

else:
    user_input = st.chat_input(f"Message {target_agent_name}-agent...")
    if user_input:
        handle_send(user_input, is_red_team=False)

if config.SHOW_RED_TEAM_REASONING_IN_UI and st.session_state["red_team_mode_active"]:
    with st.expander("Red team reasoning (last reflection)", expanded=False):
        reflect_str = red_team_agent.reflect([])
        st.text(reflect_str)

with st.expander("Attack surface bandit state", expanded=False):
    st.json(red_team_agent.bandit_state)
