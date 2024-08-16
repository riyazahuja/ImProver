import csv


def remove_every_third_line(input_csv, output_csv):
    with open(input_csv, "r", newline="") as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Skip the header row
        data = list(reader)

    # Remove every third line starting from line 4 (index 3)
    trimmed_data = [row for index, row in enumerate(data) if (index + 1) % 3 != 0]

    # Write the trimmed data to a new CSV file
    with open(output_csv, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header
        writer.writerows(trimmed_data)


# Example usage
input_csv = "benchmark/data/final/combo_final.csv"
output_csv = "output_trimmed.csv"
remove_every_third_line(input_csv, output_csv)
