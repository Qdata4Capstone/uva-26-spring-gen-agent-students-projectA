# Are Medical Agents Reliable Co-workers for Radiologists?

**Team:** Mengmeng Ma · Kathleen O'Donovan

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA/GPU recommended for best performance

### Steps
```bash
# Clone the repository
git clone https://github.com/mengmenm/uva-26-spring-gen-agent-students-projectA.git
cd team-07/src

# Install package
pip install -e .
```

---

## Getting Started
```bash
python main.py
```

If you run into permission issues:
```bash
sudo -E env "PATH=$PATH" python main.py
```

### Setup Checklist
- Set `model_dir` in `main.py` to your model weights directory
- Comment out any tools you don't have access to
- Add your OpenAI API key to the `.env` file

---

## Tool Selection

You can choose which tools to load when initializing the agent in `main.py`:
```python
selected_tools = [
    "ImageVisualizerTool",
    "ChestXRayClassifierTool",
    "ChestXRaySegmentationTool",
    # Add or remove tools as needed
]

agent, tools_dict = initialize_agent(
    "medrax/docs/system_prompts.txt",
    tools_to_use=selected_tools,
    model_dir="/model-weights"
)
```