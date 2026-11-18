import polars as pl
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

DATA_PATH = "data/raw/PS_20174392719_1491204439457_log.csv"

df = pl.scan_csv(DATA_PATH)

output = StringIO()

with redirect_stdout(output):
    print("# Finomaly - Duplicate Detection\n")
    
    total_rows = df.select(pl.len()).collect().item()
    
    unique_rows = (
        df.unique()
        .select(pl.len())
        .collect()
        .item()
    )
    
    duplicate_rows = total_rows - unique_rows
    
    print(f"Total records: {total_rows:,}")
    print(f"Unique records: {unique_rows:,}")
    print(f"Duplicate records: {duplicate_rows:,}")
    
    if duplicate_rows == 0:
        print("\nNo exact duplicate rows detected.")
    else:
        print("\nExact duplicate records detected. ")

Path("logs").mkdir(exist_ok=True)

Path("logs/duplicate_detection.md").write_text(
    output.getvalue(),
    encoding="utf-8"
)

print(output.getvalue())