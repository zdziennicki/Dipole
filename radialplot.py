import matplotlib.pyplot as plt
import csv
import os

def import_psi(file_path):
    """
    Import psi values from a CSV file.

    Args:
    - file_path (str): Path to the CSV file.

    Returns:
    - points (list of tuples): List of (x, y) coordinates.
    """
    points = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip header if exists
        for row in csv_reader:
            try:
                x = float(row[0])
                y = float(row[1])
                points.append((x, y))
            except (IndexError, ValueError):
                print(f"Skipping invalid data: {row}")
    return points

def plot2Dharm(directory):
    """
    Plot the 2D harmonic functions.

    Args:
    - directory (str): Path to the directory containing CSV files.

    Returns:
    - None
    """
    plt.rcdefaults()
    plt.figure(figsize=(12, 6))
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        if os.path.isfile(file):
            points = import_psi(file)
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            plt.plot(x_values, y_values, label= 'R' + filename[4])

    plt.xlim(0.1, 10)
    plt.ylim(-2.5, 2.5)
    plt.xlabel('r in van der Waals units')
    plt.ylabel('Non-nolmalised probability amplitude')
    plt.title('Interpolated Function')
    plt.legend()
    plt.grid(visible=True)
    plt.xscale('log')
    plt.show()







