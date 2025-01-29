import csv


def merge_csv_files(paths, out_file="merged.csv"):
    with open(out_file, "w", newline="", encoding="utf-8") as outfile:
        writer = None
        for path in paths:
            with open(path, "r", encoding="utf-8") as infile:
                reader = csv.reader(infile)
                header = next(reader)
                if writer is None:
                    writer = csv.writer(outfile)
                    writer.writerow(header)
                for row in reader:
                    writer.writerow(row)


if __name__ == "__main__":
    files = [
        "mathlib/data_merged.csv",
        "MIL/final/data.csv",
        "Compfiles/final/data.csv",
    ]
    merge_csv_files(files, "final_data.csv")
