# Experiment metrics are saved as YAML files
import yaml
# Pandas to handle the reports as table, i.e., DataFrame
import pandas as pd

# Script entrypoint
if __name__ == "__main__":
    # Open the configuration file
    with open("params.yaml") as file:
        # Load the configuration from yaml format
        params = yaml.safe_load(file)["metrics"]
    # Open the report file
    with open(params["report"]) as file:
        # Load the JSON formatted report
        report = pd.read_json(file, orient="index")

    # Collect metrics breakdown as dictionary by categories
    breakdown = {}
    # First collect the metrics breakdown by categories as configured
    for category, f in params["breakdown"].items():
        # Filter and count the number of occurrences per category
        count = len(report.filter(regex=f, axis="rows"))
        # Filter and summarize according to the category-specified rule
        summary = report.filter(regex=f, axis="rows").sum()
        # Insert into the nested breakdown dictionary
        breakdown[category] = {"COUNT": count, **summary.to_dict()}
    # Dump the per-category breakdown to its own report file
    with open("breakdown.yaml", "w") as file:
        # Dumpy python dictionary to YAML file
        yaml.safe_dump(breakdown, file)

    # Filter the reported rows according to some regex filter rule
    report = report.filter(regex=params["filter"], axis="rows")
    # Generate a summary of the total resources
    summary = report.sum()
    # Dump the metrics dictionary as yaml
    with open("metrics.yaml", "w") as file:
        # Convert the dataframe to a dictionary which can be dumped into YAML
        yaml.safe_dump(summary.to_dict(), file)
