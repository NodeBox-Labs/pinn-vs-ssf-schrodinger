import os
import re

import pandas as pd


def generate_markdown_table(csv_path):
    if not os.path.exists(csv_path):
        print(f"[error] {csv_path} not found.")
        return ""

    df = pd.read_csv(csv_path)

    # Format numbers for display
    df["Rel-L2"] = df["rel_l2_final_mean"].apply(lambda x: f"{x:.2e}")
    df["Norm Drift"] = df["norm_drift_max_mean"].apply(lambda x: f"{x:.2e}")
    df["Train (s)"] = df["train_time_mean"].apply(lambda x: f"{x:.0f}")

    # Select and rename columns for the table
    display_df = df[["case", "Rel-L2", "Norm Drift", "Train (s)", "verdict"]].copy()
    display_df.columns = ["Case", "Rel-L2 (mean)", "Norm Drift (mean)", "Train (s)", "Verdict"]

    # Map verdicts to emojis
    verdict_map = {"PASS": "✅", "BORDERLINE": "⚠️", "FAIL": "❌"}
    display_df["Verdict"] = display_df["Verdict"].map(verdict_map)

    return display_df.to_markdown(index=False)

def update_readme(csv_path, readme_path):
    md_table = generate_markdown_table(csv_path)
    if not md_table:
        return

    with open(readme_path) as f:
        content = f.read()

    start_marker = "<!-- AUTO-GENERATED-RESULTS-START -->"
    end_marker = "<!-- AUTO-GENERATED-RESULTS-END -->"

    if start_marker not in content:
        print("[warn] Markers not found in README. Please add them around the results table.")
        return

    new_content = re.sub(
        f"{start_marker}.*?{end_marker}",
        f"{start_marker}\n\n{md_table}\n\n{end_marker}",
        content,
        flags=re.DOTALL
    )

    with open(readme_path, "w") as f:
        f.write(new_content)
    print("[ok] README results table updated successfully.")

if __name__ == "__main__":
    update_readme("results/tables/master_results.csv", "README.md")
