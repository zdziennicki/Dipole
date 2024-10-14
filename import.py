import csv

def import_parameters(csv_file):
    """
    Reads a CSV file and returns its contents as a dictionary.

    Args:
    - csv_file: Path to the CSV file.

    Returns:
    - A dictionary where keys are values from the first column and values are values from the second column.
    """
    data = {}
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:  # Check if row has at least two columns
                key = row[0]
                value = row[1]
                data[key] = value
    return data

# Example usage:
csv_file = 'Parameters.csv'  # Replace 'example.csv' with the path to your CSV file
csv_data = csv_to_dict(csv_file)
print(csv_data)