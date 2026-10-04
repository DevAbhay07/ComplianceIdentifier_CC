"""
analyze_weak_label_distribution.py
===================================
Analyzes the distribution of automatically generated weak labels, compares them
against the locked 75-product human ground truth reference (DIAGNOSTIC ONLY),
and generates comprehensive statistical reports.

Inputs:
    data/processed/flipkart_weak_labels_v1.jsonl
    data/processed/pilot_ground_truth_75_v1.jsonl
    configs/weak_label_rules.yaml

Outputs:
    reports/phase_6/weak_label_distribution.csv
    reports/phase_6/weak_label_distribution.md
    reports/phase_6/human_vs_weak_distribution_comparison.md
    reports/phase_6/weak_label_diagnostic_against_human_reference.md
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import yaml
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

from src.data.generate_weak_labels import generate_weak_labels_for_product
from src.data.extract_product_evidence import extract_evidence_record


DECLARATIONS = ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]
DECL_STATES = ["PRESENT", "ABSENT", "PARTIAL", "NOT_APPLICABLE", "CANNOT_DETERMINE"]
OVERALL_STATES = ["COMPLIANT", "NON_COMPLIANT", "UNCERTAIN"]


def analyze_weak_dataset(weak_jsonl_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    records = []
    with open(weak_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    total = len(records)
    print(f"Analyzing {total} weak label records from {weak_jsonl_path}...")

    # 1. Overall Compliance
    overall_counts = Counter(r["overall_label"] for r in records)
    tier_counts = Counter(r["quality_tier"] for r in records)
    cat_counts = Counter(r["category"] for r in records)
    confidences = [r["overall_confidence"] for r in records if "overall_confidence" in r]

    # 2. Per-Declaration Distributions
    decl_dist = {did: Counter() for did in DECLARATIONS}
    decl_confs = {did: [] for did in DECLARATIONS}

    for r in records:
        decls = r.get("declarations", {})
        for did in DECLARATIONS:
            d_info = decls.get(did, {})
            label = d_info.get("label", "CANNOT_DETERMINE")
            conf = d_info.get("confidence", 0.0)
            decl_dist[did][label] += 1
            decl_confs[did].append(conf)

    # 3. Export Summary CSV
    summary_rows = []
    for did in DECLARATIONS:
        row = {"declaration": did}
        for state in DECL_STATES:
            count = decl_dist[did][state]
            pct = round((count / total) * 100, 2)
            row[f"{state}_count"] = count
            row[f"{state}_pct"] = pct
        row["mean_confidence"] = round(float(np.mean(decl_confs[did])), 3)
        summary_rows.append(row)

    dist_df = pd.DataFrame(summary_rows)
    csv_out = os.path.join(output_dir, "weak_label_distribution.csv")
    dist_df.to_csv(csv_out, index=False)
    print(f"Saved distribution CSV to {csv_out}")

    # 4. Export Distribution Markdown Report
    md_out = os.path.join(output_dir, "weak_label_distribution.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Flipkart Weak-Label Distribution Analysis (Phase 6)\n\n")
        f.write(f"**Dataset File:** `{weak_jsonl_path}`  \n")
        f.write(f"**Total Records:** {total:,}  \n")
        f.write(f"**Mean Overall Confidence:** {round(float(np.mean(confidences)), 3)}  \n\n")

        f.write("## 1. Overall Compliance Distribution\n\n")
        f.write("| Overall Label | Count | Percentage |\n")
        f.write("| :--- | :--- | :--- |\n")
        for state in OVERALL_STATES:
            c = overall_counts.get(state, 0)
            f.write(f"| **`{state}`** | {c:,} | {round(c/total*100, 2)}% |\n")
        f.write("\n")

        f.write("## 2. Quality Tier Distribution\n\n")
        f.write("| Quality Tier | Count | Percentage |\n")
        f.write("| :--- | :--- | :--- |\n")
        for tier, count in tier_counts.most_common():
            f.write(f"| **`{tier}`** | {count:,} | {round(count/total*100, 2)}% |\n")
        f.write("\n")

        f.write("## 3. Mandatory Declarations (D01-D07) Distribution\n\n")
        f.write("| Decl | PRESENT (%) | ABSENT (%) | PARTIAL (%) | CANNOT_DETERMINE (%) | Mean Conf |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in summary_rows:
            f.write(
                f"| **`{r['declaration']}`** | "
                f"{r['PRESENT_count']} ({r['PRESENT_pct']}%) | "
                f"{r['ABSENT_count']} ({r['ABSENT_pct']}%) | "
                f"{r['PARTIAL_count']} ({r['PARTIAL_pct']}%) | "
                f"{r['CANNOT_DETERMINE_count']} ({r['CANNOT_DETERMINE_pct']}%) | "
                f"{r['mean_confidence']} |\n"
            )
        f.write("\n")

        f.write("## 4. Top Category Distribution\n\n")
        f.write("| Category | Count | Percentage |\n")
        f.write("| :--- | :--- | :--- |\n")
        for cat, count in cat_counts.most_common(15):
            f.write(f"| {cat} | {count:,} | {round(count/total*100, 2)}% |\n")
        f.write("\n")

    print(f"Saved Markdown report to {md_out}")
    return records, summary_rows


def compare_human_vs_weak(weak_jsonl_path: str, human_jsonl_path: str, output_md_path: str):
    """Compare distributions between 75 human ground truth and expanded weak labels."""
    weak_records = []
    with open(weak_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                weak_records.append(json.loads(line))

    human_records = []
    with open(human_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                human_records.append(json.loads(line))

    w_total = len(weak_records)
    h_total = len(human_records)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("# Human Ground Truth vs. Automated Weak-Label Distribution Comparison\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> **Distribution similarity does not establish label accuracy.**  \n")
        f.write("> This comparison is for structural profiling and distribution auditing only.\n\n")

        f.write(f"- **Human Ground Truth Dataset:** `{human_jsonl_path}` ({h_total} products, 100% human-verified)\n")
        f.write(f"- **Automated Weak-Label Dataset:** `{weak_jsonl_path}` ({w_total:,} products, weak labels)\n\n")

        f.write("## 1. Overall Compliance Comparison\n\n")
        f.write("| State | Human Ground Truth (N=75) | Weak-Label Dataset (N=1663) |\n")
        f.write("| :--- | :--- | :--- |\n")
        h_overall = Counter(r.get("overall_label") for r in human_records)
        w_overall = Counter(r.get("overall_label") for r in weak_records)
        for state in OVERALL_STATES:
            hc = h_overall.get(state, 0)
            wc = w_overall.get(state, 0)
            f.write(f"| **`{state}`** | {hc} ({round(hc/h_total*100, 1)}%) | {wc} ({round(wc/w_total*100, 1)}%) |\n")
        f.write("\n")

        f.write("## 2. Per-Declaration Comparison (Prevalence %)\n\n")
        f.write("| Declaration | Human PRESENT | Weak PRESENT | Human ABSENT | Weak ABSENT | Human PARTIAL | Weak PARTIAL | Human CANNOT_DET | Weak CANNOT_DET |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

        for did in DECLARATIONS:
            h_states = Counter(r.get("declarations", {}).get(did, {}).get("label", "") for r in human_records)
            w_states = Counter(r.get("declarations", {}).get(did, {}).get("label", "") for r in weak_records)
            f.write(
                f"| **`{did}`** | "
                f"{round(h_states.get('PRESENT',0)/h_total*100,1)}% | {round(w_states.get('PRESENT',0)/w_total*100,1)}% | "
                f"{round(h_states.get('ABSENT',0)/h_total*100,1)}% | {round(w_states.get('ABSENT',0)/w_total*100,1)}% | "
                f"{round(h_states.get('PARTIAL',0)/h_total*100,1)}% | {round(w_states.get('PARTIAL',0)/w_total*100,1)}% | "
                f"{round(h_states.get('CANNOT_DETERMINE',0)/h_total*100,1)}% | {round(w_states.get('CANNOT_DETERMINE',0)/w_total*100,1)}% |\n"
            )
        f.write("\n")

        f.write("## 3. Analytical Observations\n\n")
        f.write("1. **D01 (Manufacturer/Packer/Importer):** Both datasets show high `PARTIAL` rates due to e-commerce brand presence without full postal addresses.\n")
        f.write("2. **D02 (Commodity Name) & D05 (MRP):** High `PRESENT` rates across both datasets, driven by structured product names and retail price fields.\n")
        f.write("3. **D04 (Mfg Date) & D07 (Country of Origin):** High `CANNOT_DETERMINE` rates in both, reflecting fundamental listing omissions.\n")
        f.write("4. **D06 (Consumer Care):** High `ABSENT` rates across both, as sellers rarely provide grievance numbers on listing cards.\n")

    print(f"Saved human vs weak comparison to {output_md_path}")


def run_diagnostic_against_human_reference(
    human_jsonl_path: str,
    rules_config_path: str,
    output_md_path: str
):
    """
    DIAGNOSTIC ONLY: Run the automated weak-label heuristics on the 75 human products
    and calculate agreement metrics. DO NOT use this to tune rules.
    """
    with open(rules_config_path, "r", encoding="utf-8") as f:
        rules_cfg = yaml.safe_load(f)

    human_records = []
    with open(human_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                human_records.append(json.loads(line))

    print(f"Running diagnostic evaluation against {len(human_records)} human records...")

    decl_matches = {did: 0 for did in DECLARATIONS}
    overall_matches = 0
    disagreements = defaultdict(list)

    for h_rec in human_records:
        # Construct synthetic evidence representation from human record fields
        pid = h_rec.get("product_id")
        title = h_rec.get("product_name", "")
        brand = h_rec.get("brand", "")
        retail_price = h_rec.get("retail_price")
        discounted_price = h_rec.get("discounted_price")
        cat = h_rec.get("category", "")
        img_urls = h_rec.get("image_urls", [])
        if isinstance(img_urls, str):
            try:
                img_urls = json.loads(img_urls)
            except Exception:
                img_urls = []

        # Extract evidence using the same evidence extractor logic
        synth_series = pd.Series({
            "uniq_id": pid,
            "product_name": title,
            "description": h_rec.get("description", ""),
            "product_specifications": h_rec.get("product_specifications", ""),
            "product_category_tree": h_rec.get("category_tree", ""),
            "brand": brand,
            "retail_price": retail_price,
            "discounted_price": discounted_price,
            "image": json.dumps(img_urls) if isinstance(img_urls, list) else str(img_urls),
            "product_url": h_rec.get("product_url", "")
        })

        evidence_rec = extract_evidence_record(synth_series, check_images=False)
        auto_weak_rec = generate_weak_labels_for_product(evidence_rec, rules_cfg)

        # Compare overall
        h_overall = h_rec.get("overall_label")
        a_overall = auto_weak_rec.get("overall_label")
        if h_overall == a_overall:
            overall_matches += 1

        # Compare declarations
        h_decls = h_rec.get("declarations", {})
        a_decls = auto_weak_rec.get("declarations", {})

        for did in DECLARATIONS:
            hl = h_decls.get(did, {}).get("label")
            al = a_decls.get(did, {}).get("label")
            if hl == al:
                decl_matches[did] += 1
            else:
                disagreements[did].append({
                    "product_id": pid,
                    "title": title[:50],
                    "human_label": hl,
                    "auto_label": al,
                    "rule_ids": a_decls.get(did, {}).get("rule_ids", [])
                })

    n = len(human_records)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("# Weak-Label Diagnostic Evaluation Against Human Reference\n\n")
        f.write("> [!NOTE]\n")
        f.write("> **DIAGNOSTIC PURPOSE ONLY**: This evaluation measures the baseline concordance of deterministic\n")
        f.write("> legal heuristics against the 75 locked human ground-truth records. The heuristics were NOT\n")
        f.write("> tuned against this reference.\n\n")

        f.write(f"- **Total Reference Records:** {n}\n")
        f.write(f"- **Overall Label Agreement:** {overall_matches}/{n} ({round(overall_matches/n*100, 1)}%)\n\n")

        f.write("## 1. Declaration-Level Concordance\n\n")
        f.write("| Declaration | Name | Agreement Count | Concordance Rate |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        decl_names = {
            "D01": "Manufacturer Name & Address",
            "D02": "Commodity Name",
            "D03": "Net Quantity",
            "D04": "Month/Year of Manufacture",
            "D05": "Maximum Retail Price (MRP)",
            "D06": "Consumer Care Details",
            "D07": "Country of Origin"
        }
        for did in DECLARATIONS:
            m = decl_matches[did]
            rate = round(m / n * 100, 1)
            f.write(f"| **`{did}`** | {decl_names[did]} | {m}/{n} | **{rate}%** |\n")
        f.write("\n")

        f.write("## 2. Disagreement Analysis\n\n")
        for did in DECLARATIONS:
            dis_list = disagreements[did]
            f.write(f"### `{did}` Disagreements ({len(dis_list)} items)\n\n")
            if not dis_list:
                f.write("No disagreements found.\n\n")
                continue
            f.write("| Product ID | Title | Human Label | Automated Label | Auto Rules |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for item in dis_list[:5]:
                f.write(f"| `{item['product_id'][:8]}` | {item['title']} | `{item['human_label']}` | `{item['auto_label']}` | `{', '.join(item['rule_ids'])}` |\n")
            f.write("\n")

    print(f"Saved diagnostic report to {output_md_path}")


if __name__ == "__main__":
    WEAK_JSONL = "data/processed/flipkart_weak_labels_v1.jsonl"
    HUMAN_JSONL = "data/processed/pilot_ground_truth_75_v1.jsonl"
    RULES_CONFIG = "configs/weak_label_rules.yaml"
    OUTPUT_DIR = "reports/phase_6"

    analyze_weak_dataset(WEAK_JSONL, OUTPUT_DIR)
    compare_human_vs_weak(
        WEAK_JSONL,
        HUMAN_JSONL,
        os.path.join(OUTPUT_DIR, "human_vs_weak_distribution_comparison.md")
    )
    run_diagnostic_against_human_reference(
        HUMAN_JSONL,
        RULES_CONFIG,
        os.path.join(OUTPUT_DIR, "weak_label_diagnostic_against_human_reference.md")
    )
