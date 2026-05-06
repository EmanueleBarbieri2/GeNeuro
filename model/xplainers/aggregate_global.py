import os
import json
import argparse
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime


def _format_seconds(seconds):
    minutes, secs = divmod(float(seconds), 60.0)
    hours, minutes = divmod(minutes, 60.0)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {secs:05.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {secs:05.2f}s"
    return f"{secs:.2f}s"


def _save_timing_report(out_dir, stage_times):
    os.makedirs(out_dir, exist_ok=True)
    total = sum(stage_times.values())
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "timings_seconds": stage_times,
        "total_seconds": total,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = os.path.join(out_dir, "aggregate_global_timing_latest.json")
    run_path = os.path.join(out_dir, f"aggregate_global_timing_{ts}.json")

    for path in [latest_path, run_path]:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    print("\n📊 Aggregation Timing Summary")
    for name, seconds in stage_times.items():
        print(f"  - {name}: {_format_seconds(seconds)}")
    print(f"  - TOTAL: {_format_seconds(total)}")
    print(f"\n📝 Saved timing report to: {latest_path}")
    print(f"📝 Saved run-specific report to: {run_path}")

def aggregate_task(reports_dir, task_name, out_dir):
    t0 = time.perf_counter()
    task_dir = os.path.join(reports_dir, task_name)
    if not os.path.exists(task_dir):
        print(f"Directory not found: {task_dir}. Skipping...")
        return {"status": "missing_dir", "elapsed_seconds": time.perf_counter() - t0, "reports": 0}

    print(f"\n📊 Aggregating Global Biomarkers for: {task_name}")
    files = [f for f in os.listdir(task_dir) if f.endswith('.json')]
    if not files:
        print(f"  No JSON reports found in {task_dir}.")
        return {"status": "no_reports", "elapsed_seconds": time.perf_counter() - t0, "reports": 0}
    print(f"  Found {len(files)} patient reports.")

    is_classification = (task_name == "classification")
    roi_registry = defaultdict(set)
    edge_registry = defaultdict(set)
    all_data = []
    
    # 1. Parse JSONs and discover all unique features
    for f in files:
        with open(os.path.join(task_dir, f), 'r') as fp:
            data = json.load(fp)
            all_data.append(data)
            for mod, mod_data in data.get('modalities', {}).items():
                
                # Register Nodes
                for node in mod_data.get('top_nodes', []):
                    roi_registry[mod].add(node.get('roi_name', 'Unknown'))
                
                # Register Edges
                for edge in mod_data.get('top_edges', []):
                    src = edge.get('u', {}).get('roi_name', 'Unknown')
                    tgt = edge.get('v', {}).get('roi_name', 'Unknown')
                    # Standardize edge name direction alphabetically
                    edge_name = f"{min(src, tgt)} <-> {max(src, tgt)}"
                    edge_registry[mod].add(edge_name)

    # 2. Math & CSV Generation Logic
    def process_features(registry, feature_type_key, name_key, output_suffix):
        for mod, features in registry.items():
            records = []
            for feature in features:
                importances, contribs = [], []
                contribs_pd, contribs_hc, contribs_prod = [], [], []

                for data in all_data:
                    mod_data = data.get('modalities', {}).get(mod, {})
                    feature_list = mod_data.get(feature_type_key, [])
                    
                    # Extract feature matches
                    item_matched = None
                    for item in feature_list:
                        if feature_type_key == 'top_nodes':
                            item_name = item.get('roi_name', 'Unknown')
                        else:
                            src = item.get('u', {}).get('roi_name', 'Unknown')
                            tgt = item.get('v', {}).get('roi_name', 'Unknown')
                            item_name = f"{min(src, tgt)} <-> {max(src, tgt)}"
                        
                        if item_name == feature:
                            item_matched = item
                            break
                    
                    # Append values (or 0.0 if not found in this patient's top-k)
                    imp = item_matched.get('importance', 0.0) if item_matched else 0.0
                    cont = item_matched.get('contrib_mean', item_matched.get('contrib', 0.0)) if item_matched else 0.0
                    
                    importances.append(imp)
                    contribs.append(cont)
                    
                    if is_classification:
                        label = data.get('true_label', 'Unknown')
                        if label == 'PD': contribs_pd.append(cont)
                        elif label == 'Control': contribs_hc.append(cont)
                        elif label == 'Prodromal': contribs_prod.append(cont)

                # Population Stats
                mean_imp, mean_cont, std_cont = np.mean(importances), np.mean(contribs), np.std(contribs)
                z_score = mean_cont / (std_cont + 1e-8)

                record = {
                    name_key: feature,
                    'Global_Saliency_Mean': float(mean_imp),
                    'Global_Contrib_Mean': float(mean_cont),
                    'Contrib_Std': float(std_cont),
                    'Robustness_Z_Score': float(z_score)
                }

                if is_classification:
                    mean_pd = np.mean(contribs_pd) if contribs_pd else 0.0
                    mean_hc = np.mean(contribs_hc) if contribs_hc else 0.0
                    record['PD_Contrib_Mean'] = round(mean_pd, 6)
                    record['HC_Contrib_Mean'] = round(mean_hc, 6)
                    record['Prodromal_Contrib_Mean'] = round(np.mean(contribs_prod) if contribs_prod else 0.0, 6)
                    record['Contrastive_Delta_C'] = round(mean_pd - mean_hc, 6)

                records.append(record)

            # Export
            df = pd.DataFrame(records)
            if not df.empty:
                df = df.sort_values(by='Global_Saliency_Mean', ascending=False)
                os.makedirs(out_dir, exist_ok=True)
                out_csv = os.path.join(out_dir, f"{task_name}_{mod}_{output_suffix}.csv")
                df.to_csv(out_csv, index=False)
                print(f"  ✅ Saved {mod} {output_suffix} to {out_csv}")

    # 3. Execute Processing
    process_features(roi_registry, 'top_nodes', 'ROI_Name', 'global_biomarkers')
    if any(len(edges) > 0 for edges in edge_registry.values()):
        process_features(edge_registry, 'top_edges', 'Edge_Name', 'global_edges')

    elapsed = time.perf_counter() - t0
    print(f"  ⏱️ Completed {task_name} in {_format_seconds(elapsed)}")
    return {"status": "ok", "elapsed_seconds": elapsed, "reports": len(files)}

def main():
    t_total = time.perf_counter()
    parser = argparse.ArgumentParser(description="Aggregate JSON reports into global biomarker CSVs.")
    parser.add_argument("--reports_dir", required=True, help="Directory containing task folders.")
    parser.add_argument("--out_dir", required=True, help="Directory to save the CSV files.")
    args = parser.parse_args()

    stage_times = {}
    for task in ["classification", "updrs", "progression"]:
        task_result = aggregate_task(args.reports_dir, task, args.out_dir)
        stage_times[f"Aggregate task: {task}"] = task_result["elapsed_seconds"]
    stage_times["aggregate_global end-to-end"] = time.perf_counter() - t_total
    _save_timing_report(args.out_dir, stage_times)
    print("\n🚀 All Global Aggregations Complete!")

if __name__ == "__main__":
    main()