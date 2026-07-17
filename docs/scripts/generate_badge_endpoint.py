"""
Automatic generation of coverage endpoint for shield.io badge.

Usage:
    python docs/scripts/generate_coverage_badge.py <report_file> <output_path> <pylint or coverage>

Example:
    python docs/scripts/generate_pylint_badge.py pylint_report.json docs/source/_static/pylint_badge.json pylint

    python docs/scripts/generate_coverage_badge.py coverage.json docs/source/_static/coverage_badge.json coverage
"""

import json
import os
import sys


def make_color(score):
    if score >= 90:
        color = "brightgreen"
    elif score >= 70:
        color = "green"
    elif score >= 50:
        color = "yellow"
    elif score >= 30:
        color = "orange"
    else:
        color = "red"
    return color


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    report_path = sys.argv[1]
    output_path = sys.argv[2]
    badge_type = sys.argv[3]

    if badge_type not in ["pylint", "coverage"]:
        raise ValueError(f"Third argument must be either 'pylint' or 'coverage'. Received: '{badge_type}'")

    if not os.path.isfile(report_path):
        raise FileNotFoundError(f"File not found: {report_path}.")

    try:
        with open(report_path, "r") as f:
            json_data = json.load(f)
    except Exception as exc:
        raise ValueError(f"Error while reading {report_path}.") from exc

    if badge_type == "pylint":
        score = 10 * json_data.get("statistics", {}).get("score", -1)  # Score x/10, making it x/100
        score_str = f"{score/10:.2f}/10"  # Display the right score /10
    else:
        score = json_data.get("totals", {}).get("percent_covered", -1)
        score_str = f"{score:.2f}%"  # Display the percentage

    if score == -1:
        raise ValueError(f"Score not found in {badge_type} report: {report_path}.")

    color = make_color(score)

    badge_data = {"schemaVersion": 1, "label": badge_type, "message": score_str, "color": color}

    try:
        with open(output_path, "w") as f:
            json.dump(badge_data, f, indent=2)
    except Exception as exc:
        raise ValueError(f"Error while writing in file {output_path}.") from exc

    print(f"[OK]  Badge generated in: {output_path}")


if __name__ == "__main__":
    main()
