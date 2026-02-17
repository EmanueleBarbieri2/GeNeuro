#!/usr/bin/env python3
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to the patient JSON (e.g. explainer_reports/updrs/10001_U3.json)")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"❌ File not found: {args.json}")
        return

    with open(args.json, "r") as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"🧠 PRO-M3E LOCAL EXPLANATION REPORT")
    print(f"{'='*60}")
    print(f"👤 Subject ID: {data['subject_id']}")
    print(f"📋 Task:       {data['task']}")
    print(f"🎯 Prediction: {data['prediction']:.4f}")
    print(f"{'-'*60}")

    for mod, content in data.get("modalities", {}).items():
        print(f"\n📂 MODALITY: {mod}")
        
        # 1. Print Top Nodes
        print(f"  📍 Top Influential Brain Regions:")
        nodes = content.get("top_nodes", [])[:args.topk]
        for n in nodes:
            # Combining the ROI name and the LUT name for maximum clarity
            name = n.get('roi_name', 'Unknown')
            lut = f" ({n.get('lut_name')})" if n.get('lut_name') else ""
            imp = n.get('importance', 0.0)
            contrib = n.get('contrib_mean', 0.0)
            
            # Clinical interpretation of contribution
            direction = "🔺 increases" if contrib > 0 else "🔹 decreases"
            print(f"    - {name}{lut:30} | Imp: {imp:.4f} | {direction} score")

        # 2. Print Top Edges (Connections)
        edges = content.get("top_edges", [])[:args.topk]
        if edges:
            print(f"\n  🔗 Top Influential Connectivity Edges:")
            for e in edges:
                u_name = e['u'].get('roi_name')
                v_name = e['v'].get('roi_name')
                print(f"    - {u_name} <---> {v_name}")

if __name__ == "__main__":
    main()