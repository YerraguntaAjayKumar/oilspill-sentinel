import pandas as pd

file = "data/events/dartis_2019.txt"

# Find the actual data table
with open(file, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

# Find the first line containing the actual column header
header_line = None

for i, line in enumerate(lines):
    if line.startswith("Image set"):
        header_line = i
        break

if header_line is None:
    print("Could not find the data table header.")
    exit()

print("Data table starts at line:", header_line + 1)

# Read the actual table
df = pd.read_csv(
    file,
    sep="\t",
    skiprows=header_line,
    encoding="utf-8",
    encoding_errors="replace"
)

print("\nNumber of rows:", len(df))

print("\nColumns:")
for column in df.columns:
    print(repr(column))

print("\nFirst 5 rows:")
print(df.head().to_string())