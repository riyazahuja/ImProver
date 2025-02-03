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
    libfiles = [
        "mathlib/algtop/traj.csv",
        "mathlib/algtop/traj2.csv",
        "mathlib/computability/traj.csv",
        "mathlib/computability/traj2.csv",
    ]
    merge_csv_files(libfiles, "mathlib/traj_merged.csv")

    files = [
        "mathlib/traj_merged.csv",
        "MIL/final/traj.csv",
        "Compfiles/final/traj.csv",
    ]
    merge_csv_files(files, "final_traj.csv")
