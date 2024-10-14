import matplotlib.pyplot as plt
import scipy.special as sp
import seaborn as sns
import numpy as np
import csv
from scipy.interpolate import interp1d
import os
from matplotlib import ticker
import math

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
                value = float(row[1])
                data[key] = value
    return data

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

def generate_radial_function(file):
    """
    Generate a radial function from a CSV file.

    Args:
    - file (str): Path to the CSV file.

    Returns:
    - radial (interp1d): Interpolated radial function.
    """
    points = import_psi(file)
    x_values = [point[0] for point in points]
    x_values.insert(0, 0)
    y_values = [point[1] for point in points]
    y_values.insert(0, 0)
    radial = interp1d(x_values, y_values, kind='linear')
    return radial

def angular_function(m, l, theta, phi):
    """
    Calculate the angular function.

    Args:
    - m (int): Azimuthal quantum number.
    - l (int): Angular momentum quantum number.
    - theta (float): Polar angle.
    - phi (float): Azimuthal angle.

    Returns:
    - angular_function (complex): Angular function value.
    """
    legendre = sp.lpmv(m, l, np.cos(theta))
    constant_factor = ((-1) ** m) * np.sqrt(
        ((2 * l + 1) * sp.factorial(l - np.abs(m))) /
        (4 * np.pi * sp.factorial(l + np.abs(m)))
    )
    return constant_factor * legendre * np.real(np.exp(1.j * m * phi))

def compute_wavefunction(directory):
    """
    Compute the wavefunction.

    Args:
    - directory (str): Path to the directory containing CSV files.

    Returns:
    - psi (numpy.ndarray): Computed wavefunction.
    """
    psi = 0
    grid_extent = 2
    grid_resolution = 680
    z = x = np.linspace(-grid_extent, grid_extent, grid_resolution)
    z, x = np.meshgrid(z, x)
    eps = np.finfo(float).eps

    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        radial = generate_radial_function(file)
        psi += radial(
            np.sqrt((x ** 2 + z ** 2))
        ) * angular_function(
            0, 2 * (int(filename[4]) - 1), np.arctan(x / (z + eps)), 0
        )
    
    return psi

def gaussian(x, mu, sig):
    """
    Gaussian function.

    Args:
    - x (numpy.ndarray): Input values.
    - mu (float): Mean of the Gaussian.
    - sig (float): Standard deviation of the Gaussian.

    Returns:
    - result (numpy.ndarray): Gaussian values.
    """
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )

def ident(z):
    """
    Identity function.

    Args:
    - z (numpy.ndarray): Input values.

    Returns:
    - result (numpy.ndarray): Identity function values.
    """
    return 1

def compute_hermite(n, x0=1.0):
    """
    Compute the nth solution to the quantum harmonic oscillator problem.

    Args:
    - n (int): The order of the harmonic oscillator solution.
    - x0 (float): The harmonic oscillator length parameter.

    Returns:
    - psi (numpy.ndarray): Computed nth harmonic oscillator solution.
    """
    grid_extent = 2
    grid_resolution = 680
    z = x = np.linspace(-grid_extent, grid_extent, grid_resolution)
    z, x = np.meshgrid(z, x)

    # Compute the nth harmonic oscillator solution
    psi = np.exp(-0.5 * (x / (x0**2)) ** 2) * np.polynomial.hermite.hermval(x / x0, np.eye(n + 1)[-1]) / (np.sqrt(2 ** n * np.math.factorial(n)) * np.sqrt(np.pi * x0**2))

    return psi

def compute_probability_density(psi):
    """
    Compute the probability density.

    Args:
    - psi (numpy.ndarray): Wavefunction.

    Returns:
    - prob_density (numpy.ndarray): Probability density.
    """
    return np.abs(psi) ** 2

def plot_wf_probability_density(directory, dark_theme=False, colormap='rocket'):
    """
    Plot the wavefunction probability density.

    Args:
    - directory (str): Path to the directory containing CSV files.
    - dark_theme (bool): Whether to use dark theme or not.
    - colormap (str): Name of the colormap.

    Returns:
    - None
    """
    try:
        sns.color_palette(colormap)
    except ValueError:
        raise ValueError(f'{colormap} is not a recognized Seaborn colormap.')

    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.major.width'] = 4
    plt.rcParams['ytick.major.width'] = 4
    plt.rcParams['xtick.major.size'] = 15
    plt.rcParams['ytick.major.size'] = 15
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['axes.linewidth'] = 4

    fig, ax = plt.subplots(figsize=(16, 16.5))
    plt.subplots_adjust(top=0.82)
    plt.subplots_adjust(right=0.905)
    plt.subplots_adjust(left=-0.1)

    psi = compute_wavefunction(directory)
    prob_density = compute_probability_density(psi)
    
    im = ax.imshow(np.sqrt(prob_density).T, cmap=sns.color_palette(colormap, as_cmap=True))
    locator = ticker.LinearLocator(numticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)
    ax.set_xticklabels(np.round(np.linspace(-4,4,9),2))
    ax.set_yticklabels(np.round(np.linspace(-4,4,9),2))

    cbar = plt.colorbar(im, fraction=0.046, pad=0.03)
    cbar.set_ticks([])
    
    
    if dark_theme:
        theme = 'dt'
        background_color = sorted(
            sns.color_palette(colormap, n_colors=100),
            key=lambda color: 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        )[0]
        plt.rcParams['text.color'] = '#dfdfdf'
        title_color = '#dfdfdf'
        fig.patch.set_facecolor(background_color)
        cbar.outline.set_visible(False)
        ax.tick_params(axis='x', colors='#c4c4c4')
        ax.tick_params(axis='y', colors='#c4c4c4')
        for spine in ax.spines.values():
            spine.set_color('#c4c4c4')

    else:  # Apply light theme parameters
        theme = 'lt'
        plt.rcParams['text.color'] = '#000000'
        title_color = '#000000'
        ax.tick_params(axis='x', colors='#000000')
        ax.tick_params(axis='y', colors='#000000')

    ax.set_title('Bound State of Dysprosium in harmonic trap', 
                 pad=40, fontsize=44, loc='center', color=title_color)

    ax.text(30, 615, '(En = , a =)', color='#dfdfdf', fontsize=42)
    ax.text(770, 200, 'Probability distribution', rotation='vertical', fontsize=40)
    ax.text(705, 700, 'Higher\nprobability', fontsize=24)
    ax.text(705, -60, 'Lower\nprobability', fontsize=24)
    ax.text(775, 590, '+', fontsize=34)
    ax.text(769, 82, '−', fontsize=34, rotation='vertical')
    ax.invert_yaxis()

    # Save and display the plot
    plt.show()
    
    
def calculate_r_squared(shape):
    """
    Calculate the squared distance from the center for each point in a 2D grid. 
    This function is required for discrete integration

    Args:
    - shape (tuple): The shape of the grid (rows, columns).

    Returns:
    - distance_squared (numpy.ndarray): 2D array containing the squared distance from the center for each point.
    """
    x_coords, y_coords = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    center_x = shape[0] // 2
    center_y = shape[1] // 2
    distance_squared = (8 * (y_coords - center_y) / shape[0]) ** 2
    return distance_squared

def calculate_correlation(psi1, psi2, rsq):
    """
    Calculate the correlation between two wavefunctions.

    Args:
    - psi1 (numpy.ndarray): Wavefunction 1.
    - psi2 (numpy.ndarray): Wavefunction 2.
    - rsq (numpy.ndarray): Squared distance from the center.

    Returns:
    - one dimensional correlation along z axis (numpy.ndarray): Correlation between psi1 and psi2.
    """
    norm_psi1 = np.sum(psi1 * psi1 * rsq * (8 / 680) ** 2)
    norm_psi2 = np.sum(psi2 * psi2 * rsq * (8 / 680) ** 2)
    psi1_normalized = psi1 / np.sqrt(norm_psi1)
    psi2_normalized = psi2 / np.sqrt(norm_psi2)
    correlation = np.sum(psi1_normalized * psi2_normalized * rsq * (8 / 680), axis=0)
    correlation = correlation[:len(correlation) // 2][::-1]
    return correlation

def plot_correlation(x_values, correlation):
    """
    Plot the correlation values.

    Args:
    - x_values (numpy.ndarray): X-axis values.
    - correlation (numpy.ndarray): Correlation values.
    """
    plt.rcdefaults()
    plt.grid(visible=True)
    plt.xlim(0, 4)
    plt.plot(x_values, correlation ** 2 * (680 / 8))
    plt.xlabel('Distance in RvdW')
    plt.ylabel('Correlation')
    plt.title('Plot of correlation')
    plt.show()

def save_correlation_data(filename, x_values, correlation):
    """
    Save correlation data to a CSV file.

    Args:
    - filename (str): Name of the CSV file.
    - x_values (numpy.ndarray): X-axis values.
    - correlation (numpy.ndarray): Correlation values.
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Correlation Squared'])
        for i in range(len(x_values)):
            writer.writerow([x_values[i], correlation[i] ** 2])