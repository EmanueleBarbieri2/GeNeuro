import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    args = parser.parse_args()

    with open(args.infile, "r") as f:
        report = json.load(f)

    print(f"Classification Explainer Report for {report['subject_id']}")
    print(f"  True label: {report['true_label']}")
    print(f"  Predicted label: {report['pred_label']} (prob={report['pred_prob']:.4f})")
    print("  Top contributing features (by |grad|):")
    grads = report["feature_grad"]
    names = report["feature_names"]
    topk = sorted(zip(names, grads), key=lambda x: abs(x[1]), reverse=True)[:10]
    for name, grad in topk:
        print(f"    {name}: grad={grad:.6f}")

if __name__ == "__main__":
    main()
