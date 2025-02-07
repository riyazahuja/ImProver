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
    carleson_data = ["carleson/data.csv", "carleson/data2.csv"]
    merge_csv_files(carleson_data, "carleson/final/data.csv")

    carleson_traj = ["carleson/traj.csv", "carleson/traj2.csv"]
    merge_csv_files(carleson_traj, "carleson/final/traj.csv")

    htpi_data = [
        "htpi/data.csv",
        "htpi/data2.csv",
        "htpi/data3.csv",
        "htpi/data4.csv",
        "htpi/data5.csv",
    ]

    htpi_traj = [
        "htpi/traj.csv",
        "htpi/traj2.csv",
        "htpi/traj3.csv",
        "htpi/traj4.csv",
        "htpi/traj5.csv",
    ]

    merge_csv_files(htpi_data, "htpi/final/data.csv")
    merge_csv_files(htpi_traj, "htpi/final/traj.csv")

    mathlib_data = [
        "mathlib/data_merged.csv",
        "mathlib/category/data.csv",
        "mathlib/category/data2.csv",
        "mathlib/category/data3.csv",
        "mathlib/category/data4.csv",
    ]

    mathlib_traj = [
        "mathlib/traj_merged.csv",
        "mathlib/category/traj.csv",
        "mathlib/category/traj2.csv",
        "mathlib/category/traj3.csv",
        "mathlib/category/traj4.csv",
    ]

    merge_csv_files(mathlib_data, "mathlib/final/data.csv")
    merge_csv_files(mathlib_traj, "mathlib/final/traj.csv")

    files = [
        "carleson/final/data.csv",
        "Compfiles/final/data.csv",
        "hep/data.csv",
        "htpi/final/data.csv",
        "knot/data.csv",
        "mathlib/final/data.csv",
        "MIL/final/data.csv",
        "pnt/data.csv",
    ]
    merge_csv_files(files, "FINAL/data.csv")
    files_traj = [
        "carleson/final/traj.csv",
        "Compfiles/final/traj.csv",
        "hep/traj.csv",
        "htpi/final/traj.csv",
        "knot/traj.csv",
        "mathlib/final/traj.csv",
        "MIL/final/traj.csv",
        "pnt/traj.csv",
    ]
    merge_csv_files(files, "FINAL/traj.csv")
