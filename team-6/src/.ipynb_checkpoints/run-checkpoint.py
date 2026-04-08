import sys
sys.path.insert(0, "/scratch/rzv4ve/cardioagent/")

from planner import CardioAgentPlanner

planner = CardioAgentPlanner(
    qwen_api_url="http://localhost:8000/v1",
    lingshu_api_url="http://localhost:8001/v1",
)

state = planner.run(
    patient_id="P001",
    ecg_path="/scratch/rzv4ve/cardioagent/qwen-model/46857043.hea",
    dicom_input = "/scratch/rzv4ve/cardioagent/qwen-model/sample1.jpg",
    clinical_notes="1. Normal biventricular size and systolic function. LVEF is 54%. RV EF is 57%  2. Basal inferolateral subepicardial LGE with corresponding elevation in T1 and T2 values consistent with acute myocarditis. 3. No hemodynamically significant valve disease 4. Small circumferential pericardial effusion without  thickening or pericardial LGE Jamey A Cutts, M.D., Fellow Physician, Cardiology Christopher M. Kramer MD Attending Physician, Radiology/Cardiology",
)

# 显示思考过程
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

# 显示图像（保存路径，可以手动查看）
print("\n" + "=" * 60)
print("GENERATED IMAGES")
print("=" * 60)
for name, path in state.images.items():
    print(f"  {name}: {path}")

# 显示报告
print("\n" + "=" * 60)
print("DIAGNOSTIC REPORT")
print("=" * 60)
print(state.final_report)