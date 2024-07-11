import csv

def transform_method_column(input_csv, output_csv):
    with open(input_csv, mode='r') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        # Read all rows into a list
        rows = list(reader)
        
        # Iterate over rows and modify the `method` column
        for i in range(len(rows)):
            if rows[i]['method'].startswith('prompt'):
                continue
            if rows[i]['method'].startswith('best_of_n'):
                if i > 0 and rows[i-1]['method'].startswith('prompt'):
                    if 'flat' in rows[i-1]['method']:
                        rows[i]['method'] += '(flat)'
                    elif 'structured' in rows[i-1]['method']:
                        rows[i]['method'] += '(structured)'
            if rows[i]['method'].startswith('refinement'):
                if i > 0 and rows[i-2]['method'].startswith('prompt'):
                    if 'flat' in rows[i-2]['method']:
                        rows[i]['method'] += '(flat)'
                    elif 'structured' in rows[i-2]['method']:
                        rows[i]['method'] += '(structured)'

    # Write the modified data back to a new CSV file
    with open(output_csv, mode='w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# Replace 'input.csv' and 'output.csv' with your actual file paths
input_csv = 'method_data.csv'
output_csv = 'method_data_fixed.csv'

transform_method_column(input_csv, output_csv)
