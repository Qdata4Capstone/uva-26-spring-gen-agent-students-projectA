"""
Tests for src/agent/agent.py — LegalComplianceAgent initialization and planning.

All Claude API calls are mocked — no ANTHROPIC_API_KEY required.

Run from team-00/:
    pytest tests/test_agent.py -v
"""
import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestLegalComplianceAgentInit:
    """Test agent initialization without invoking any real tools or APIs."""

    @patch("src.agent.agent.LLMController")
    @patch("src.agent.agent.EgoBlurTool")
    @patch("src.agent.agent.FaceBlurTool")
    @patch("src.agent.agent.ReIDCritic")
    @patch("src.agent.agent.FileManager")
    @patch("src.agent.agent.KnowledgeBase")
    def test_agent_initializes(
        self, mock_kb, mock_fm, mock_critic, mock_fb, mock_eb, mock_ctrl, tmp_path
    ):
        from src.agent.agent import LegalComplianceAgent
        agent = LegalComplianceAgent(dataset_dir=str(tmp_path))
        assert agent is not None
        assert agent.dataset_dir == tmp_path

    @patch("src.agent.agent.LLMController")
    @patch("src.agent.agent.EgoBlurTool")
    @patch("src.agent.agent.FaceBlurTool")
    @patch("src.agent.agent.ReIDCritic")
    @patch("src.agent.agent.FileManager")
    @patch("src.agent.agent.KnowledgeBase")
    def test_output_dir_defaults_to_sibling(
        self, mock_kb, mock_fm, mock_critic, mock_fb, mock_eb, mock_ctrl, tmp_path
    ):
        from src.agent.agent import LegalComplianceAgent
        dataset = tmp_path / "images"
        dataset.mkdir()
        agent = LegalComplianceAgent(dataset_dir=str(dataset))
        assert "output" in str(agent.output_dir)

    @patch("src.agent.agent.LLMController")
    @patch("src.agent.agent.EgoBlurTool")
    @patch("src.agent.agent.FaceBlurTool")
    @patch("src.agent.agent.ReIDCritic")
    @patch("src.agent.agent.FileManager")
    @patch("src.agent.agent.KnowledgeBase")
    def test_custom_output_dir(
        self, mock_kb, mock_fm, mock_critic, mock_fb, mock_eb, mock_ctrl, tmp_path
    ):
        from src.agent.agent import LegalComplianceAgent
        out_dir = tmp_path / "custom_out"
        agent = LegalComplianceAgent(
            dataset_dir=str(tmp_path),
            output_dir=str(out_dir),
        )
        assert agent.output_dir == out_dir


class TestLegalComplianceAgentRun:
    """Test the agent's run method with mocked sub-components."""

    _MOCK_PLAN = json.dumps({
        "intent_summary": "Prepare dataset for CVPR 2026",
        "platform": None,
        "conference": "cvpr2026",
        "regulations": ["gdpr"],
        "tools": ["egoblur", "knowledge_retrieval"],
    })

    @patch("src.agent.agent.LLMController")
    @patch("src.agent.agent.EgoBlurTool")
    @patch("src.agent.agent.FaceBlurTool")
    @patch("src.agent.agent.ReIDCritic")
    @patch("src.agent.agent.FileManager")
    @patch("src.agent.agent.KnowledgeBase")
    def test_run_returns_result_dict(
        self, mock_kb_cls, mock_fm_cls, mock_critic_cls,
        mock_fb_cls, mock_eb_cls, mock_ctrl_cls, tmp_path
    ):
        # Wire mock controller to return a plan + final response
        mock_ctrl = mock_ctrl_cls.return_value
        mock_ctrl.plan.return_value = self._MOCK_PLAN
        mock_ctrl.respond.return_value = "Compliance complete."

        # Mock critic to approve on first try
        mock_critic = mock_critic_cls.return_value
        mock_critic.validate.return_value = {"approved": True, "score": 0.02}

        # Mock egoblur as unavailable → falls back gracefully
        mock_eb = mock_eb_cls.return_value
        type(mock_eb).is_available = PropertyMock(return_value=False)

        from src.agent.agent import LegalComplianceAgent
        agent = LegalComplianceAgent(dataset_dir=str(tmp_path))
        result = agent.run("I want to submit this to CVPR 2026")

        assert isinstance(result, dict)

    @patch("src.agent.agent.LLMController")
    @patch("src.agent.agent.EgoBlurTool")
    @patch("src.agent.agent.FaceBlurTool")
    @patch("src.agent.agent.ReIDCritic")
    @patch("src.agent.agent.FileManager")
    @patch("src.agent.agent.KnowledgeBase")
    def test_run_retries_on_critic_rejection(
        self, mock_kb_cls, mock_fm_cls, mock_critic_cls,
        mock_fb_cls, mock_eb_cls, mock_ctrl_cls, tmp_path
    ):
        mock_ctrl = mock_ctrl_cls.return_value
        mock_ctrl.plan.return_value = self._MOCK_PLAN
        mock_ctrl.respond.return_value = "Re-processing..."

        # Critic fails twice, then approves
        mock_critic = mock_critic_cls.return_value
        mock_critic.validate.side_effect = [
            {"approved": False, "score": 0.45},
            {"approved": False, "score": 0.30},
            {"approved": True, "score": 0.01},
        ]

        mock_eb = mock_eb_cls.return_value
        type(mock_eb).is_available = PropertyMock(return_value=False)

        from src.agent.agent import LegalComplianceAgent
        agent = LegalComplianceAgent(dataset_dir=str(tmp_path))
        result = agent.run("Prepare for NeurIPS submission")

        # Critic validate should have been called at least twice
        assert mock_critic.validate.call_count >= 2
