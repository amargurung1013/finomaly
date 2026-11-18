import polars as pl
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

DATA_PATH = "data/raw/PS_20174392719_1491204439457_log.csv"

# Capture output
output = StringIO()

with redirect_stdout(output):

    df = pl.scan_csv(DATA_PATH)

    print("# Finomaly - Initial Data Inspection\n")

    print("## Dataset Schema")
    print(df.collect_schema())

    print("\n## Total Rows")
    print(df.select(pl.len()).collect())

    print("\n## Missing Values")
    print(df.select(pl.all().null_count()).collect())

    print("\n## Transaction Distribution")
    print(
        df.group_by("type")
        .len()
        .sort("len", descending=True)
        .collect()
    )

    print("\n## Fraud Distribution")
    print(df.group_by("isFraud").len().collect())

    print("\n## Amount Statistics")
    print(
        df.select(
            pl.col("amount").mean().alias("mean"),
            pl.col("amount").median().alias("median"),
            pl.col("amount").min().alias("minimum"),
            pl.col("amount").max().alias("maximum"),
        ).collect()
    )

# Save output to Markdown
Path("logs").mkdir(exist_ok=True)

Path("logs/initial_data_inspection.md").write_text(
    output.getvalue(),
    encoding="utf-8"
)

# Also display output in terminal
print(output.getvalue())
