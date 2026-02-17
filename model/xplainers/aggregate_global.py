import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

def aggregate_task(reports_dir, task_name, out_dir):
    task_dir = os.path.join(reports_dir, task_name)
    if not os.path.exists(task_dir):
        print(f"Directory not found: {task_dir}. Skipping...")
        return

    print(f"\n📊 Aggregating Global Biomarkers for: {task_name}")
    files = [f for f in os.listdir(task_dir) if f.endswith('.json')]
    N = len(files)
    if N == 0:
        print(f"  No JSON reports found in {task_dir}.")
        return
    print(f"  Found {N} patient reports.")

    is_classification = (task_name == "classification")

    # Step 1: Discover all unique ROIs that appeared in the Top-K across all patients
    roi_registry = defaultdict(set) # modality -> set of unique ROI names
    all_data = []
    
    for f in files:
        with open(os.path.join(task_dir, f), 'r') as fp:
            data = json.load(fp)
            all_data.append(data)
            for mod, mod_data in data.get('modalities', {}).items():
                for node in mod_data.get('top_nodes', []):
                    roi_registry[mod].add(node.get('roi_name', 'Unknown'))

    # Step 2: Compute Population-Level Math
    for mod, rois in roi_registry.items():
        records = []
        for roi in rois:
            importances = []
            contribs = []
            contribs_pd = []
            contribs_hc = []

            for data in all_data:
                mod_data = data.get('modalities', {}).get(mod, {})
                nodes = mod_data.get('top_nodes', [])
                
                # Check if this ROI was important for this specific patient
                found = False
                for n in nodes:
                    if n.get('roi_name') == roi:
                        imp = n.get('importance', 0.0)
                        cont = n.get('contrib_mean', 0.0)
                        importances.append(imp)
                        contribs.append(cont)
                        
                        if is_classification:
                            label = data.get('true_label', 'Unknown')
                            if label == 'PD': contribs_pd.append(cont)
                            elif label == 'Control': contribs_hc.append(cont)
                        
                        found = True
                        break
                
                # If it wasn't in the Top-K, its contribution is effectively 0 for this patient
                if not found:
                    importances.append(0.0)
                    contribs.append(0.0)
                    if is_classification:
                        label = data.get('true_label', 'Unknown')
                        if label == 'PD': contribs_pd.append(0.0)
                        elif label == 'Control': contribs_hc.append(0.0)

            # Mathematical Aggregation (Matches your thesis methodology)
            imp_array = np.array(importances)
            cont_array = np.array(contribs)

            mean_imp = np.mean(imp_array)                  # Global Absolute Saliency
            mean_cont = np.mean(cont_array)                # Global Directional Contribution
            std_cont = np.std(cont_array)                  # Biomarker Variance
            z_score = mean_cont / (std_cont + 1e-8)        # Robustness Z-Score

            record = {
                'ROI_Name': roi,
                'Global_Saliency_Mean': float(mean_imp),
                'Global_Contrib_Mean': float(mean_cont),
                'Contrib_Std': float(std_cont),
                'Robustness_Z_Score': float(z_score)
            }

            # Contrastive Biomarkers (PD vs HC)
            if is_classification:
                mean_pd = np.mean(contribs_pd) if len(contribs_pd) > 0 else 0
                mean_hc = np.mean(contribs_hc) if len(contribs_hc) > 0 else 0
                record['PD_Contrib_Mean'] = round(mean_pd, 6)
                record['HC_Contrib_Mean'] = round(mean_hc, 6)
                record['Contrastive_Delta_C'] = round(mean_pd - mean_hc, 6)

            records.append(record)

        # Step 3: Format and Save to CSV
        df = pd.DataFrame(records)
        if not df.empty:
            # Sort by absolute saliency so the most important biomarkers are at the top
            df = df.sort_values(by='Global_Saliency_Mean', ascending=False)
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(out_dir, f"{task_name}_{mod}_global_biomarkers.csv")
            df.to_csv(out_csv, index=False)
            print(f"  ✅ Saved {mod} global biomarkers to {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="Aggregate patient-level JSON reports into global biomarker CSVs.")
    parser.add_argument("--reports_dir", required=True, help="Directory containing the 'classification', 'updrs', and 'progression' folders.")
    parser.add_argument("--out_dir", required=True, help="Directory to save the aggregated CSV files.")
    args = parser.parse_args()

    # Process all three tasks
    tasks = ["classification", "updrs", "progression"]
    for task in tasks:
        aggregate_task(args.reports_dir, task, args.out_dir)

    print("\n🚀 All Global Aggregations Complete!")

if __name__ == "__main__":
    main()