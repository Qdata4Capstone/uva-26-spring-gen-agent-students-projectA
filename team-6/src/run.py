import sys
sys.path.insert(0, "/scratch/rzv4ve/cardioagent/")

from planner import CardioAgentPlanner

planner = CardioAgentPlanner(
    qwen_api_url="http://localhost:8000/v1",
    lingshu_api_url="http://localhost:8001/v1",
)

state = planner.run(
    patient_id="P001",
    ecg_path="/scratch/rzv4ve/cardioagent/data/40961212.dat",
    dicom_input = "/scratch/rzv4ve/cardioagent/data/brain_MRI.dcm",
    clinical_notes="diagnose with Atelectasis,Edema,Lung Opacity,Pleural Effusion",
)

# thinking process
print("=" * 60)
print("THINKING PROCESS")
print("=" * 60)
for step in state.thinking_steps:
    icon = {"success": "✅", "error": "❌", "skipped": "⏭️"}.get(step.status, "⏳")
    print(f"\n{icon} {step.step_name} ({step.duration_s}s)")
    print(f"   {step.detail}")
    for sub in step.substeps:
        s_icon = "✓" if sub.get("status") == "success" else "✗"
        print(f"     {s_icon} {sub.get('step','')}: {sub.get('detail','')}")

# visualize image
print("\n" + "=" * 60)
print("GENERATED IMAGES")
print("=" * 60)
for name, path in state.images.items():
    print(f"  {name}: {path}")

# display report
print("\n" + "=" * 60)
print("DIAGNOSTIC REPORT")
print("=" * 60)
print(state.final_report)