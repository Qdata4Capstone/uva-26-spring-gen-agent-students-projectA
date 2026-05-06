import os
import argparse
import subprocess
import sys
import warnings
from typing import *
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from interface_v2 import create_demo
from medrax.agent import *
from medrax.tools import *
from medrax.utils import *

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
_ = load_dotenv()


def initialize_agent(
    prompt_file,
    tools_to_use=None,
    model_dir="/model-weights",
    temp_dir="temp",
    device="cuda",
    model="chatgpt-4o-latest",
    temperature=0.7,
    top_p=0.95,
    openai_kwargs={}
):
    """Initialize the MedRAX agent with specified tools and configuration.

    Args:
        prompt_file (str): Path to file containing system prompts
        tools_to_use (List[str], optional): List of tool names to initialize. If None, all tools are initialized.
        model_dir (str, optional): Directory containing model weights. Defaults to "/model-weights".
        temp_dir (str, optional): Directory for temporary files. Defaults to "temp".
        device (str, optional): Device to run models on. Defaults to "cuda".
        model (str, optional): Model to use. Defaults to "chatgpt-4o-latest".
        temperature (float, optional): Temperature for the model. Defaults to 0.7.
        top_p (float, optional): Top P for the model. Defaults to 0.95.
        openai_kwargs (dict, optional): Additional keyword arguments for OpenAI API, such as API key and base URL.

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT"]

    all_tools = {
        "ChestXRayClassifierTool": lambda: ChestXRayClassifierTool(device=device),
        "ChestXRaySegmentationTool": lambda: ChestXRaySegmentationTool(device=device),
        "LlavaMedTool": lambda: LlavaMedTool(cache_dir=model_dir, device=device, load_in_8bit=True),
        "XRayVQATool": lambda: XRayVQATool(cache_dir=model_dir, device=device),
        "ChestXRayReportGeneratorTool": lambda: ChestXRayReportGeneratorTool(
            cache_dir=model_dir, device=device
        ),
        "XRayPhraseGroundingTool": lambda: XRayPhraseGroundingTool(
            cache_dir=model_dir, temp_dir=temp_dir, load_in_8bit=True, device=device
        ),
        "ChestXRayGeneratorTool": lambda: ChestXRayGeneratorTool(
            model_path=f"{model_dir}/roentgen", temp_dir=temp_dir, device=device
        ),
        "ImageVisualizerTool": lambda: ImageVisualizerTool(),
        "DicomProcessorTool": lambda: DicomProcessorTool(temp_dir=temp_dir),
        "GradCAMExplainerTool": lambda: GradCAMExplainerTool(model_dir=model_dir, temp_dir=temp_dir, device=device),
    }

    # Initialize only selected tools or all if none specified
    tools_dict = {}
    tools_to_use = tools_to_use or all_tools.keys()
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            tools_dict[tool_name] = all_tools[tool_name]()

    checkpointer = MemorySaver()
    model = ChatOpenAI(model=model, temperature=temperature, top_p=top_p, **openai_kwargs)
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
    )

    print("Agent initialized")
    return agent, tools_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedRAX server")
    parser.add_argument(
        "--ui",
        choices=["gradio", "chainlit"],
        default="gradio",
        help="UI backend to use (default: gradio)",
    )
    parser.add_argument("--port", type=int, default=8585)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--share", action="store_true", default=True)
    args = parser.parse_args()

    selected_tools = [
        "ImageVisualizerTool",
        "DicomProcessorTool",
        "ChestXRayClassifierTool",
        "ChestXRaySegmentationTool",
        "ChestXRayReportGeneratorTool",
        "XRayVQATool",
        "LlavaMedTool",
        "XRayPhraseGroundingTool",
        "GradCAMExplainerTool",
    ]

    openai_kwargs = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = api_key
    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    src_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.environ.get("MODEL_DIR", os.path.join(src_dir, "mydownload"))
    default_temp_dir  = os.environ.get("TEMP_DIR",  os.path.join(default_model_dir, "temp"))
    default_prompt    = os.environ.get("PROMPT_FILE",
                                       os.path.join(src_dir, "medrax", "docs", "system_prompts.txt"))
    default_model     = os.environ.get("OPENAI_MODEL", "gpt-4o")
    default_device    = os.environ.get("DEVICE", "cuda")

    os.environ.setdefault("GRADIO_TEMP_DIR", default_temp_dir)

    if args.ui == "chainlit":
        # Chainlit manages its own agent initialization via @cl.on_app_startup.
        # Pass configuration through environment variables and launch in-process.
        os.environ["MODEL_DIR"]   = default_model_dir
        os.environ["TEMP_DIR"]    = default_temp_dir
        os.environ["PROMPT_FILE"] = default_prompt
        os.environ["OPENAI_MODEL"] = default_model
        os.environ["DEVICE"]      = default_device

        print(f"Starting Chainlit UI on http://{args.host}:{args.port} ...")
        subprocess.run([
            sys.executable, "-m", "chainlit", "run",
            os.path.join(src_dir, "interface_v3.py"),
            "--host", args.host,
            "--port", str(args.port),
        ])
    else:
        print("Starting Gradio server...")
        agent, tools_dict = initialize_agent(
            default_prompt,
            tools_to_use=selected_tools,
            model_dir=default_model_dir,
            temp_dir=default_temp_dir,
            device=default_device,
            model=default_model,
            temperature=0.7,
            top_p=0.95,
            openai_kwargs=openai_kwargs,
        )
        demo = create_demo(agent, tools_dict)
        demo.launch(server_name=args.host, server_port=args.port, share=args.share)
