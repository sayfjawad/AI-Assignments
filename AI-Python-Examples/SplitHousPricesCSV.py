import csv
import re


# Define a function to extract ZIP codes from an address string
def extract_zip(address):
    match = re.search(r'(\d{5}(?:-\d{4})?)', address)
    return match.group(1) if match else None


# Open the input CSV file and create a CSV reader object
with open('usa_housing_training.csv', mode='r') as infile:
    reader = csv.DictReader(infile)

    # Open the output CSV file and create a CSV writer object
    with open('split_usa_housing_training_output.csv', mode='w', newline='') as outfile:
        fieldnames = reader.fieldnames + ['zip']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()  # Write the header to the output CSV
        for row in reader:
            row['zip'] = extract_zip(row['address'])
            writer.writerow(row)
