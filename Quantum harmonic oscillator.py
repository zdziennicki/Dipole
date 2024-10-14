# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:16:48 2024

@author: Oem
"""

import numpy as np
from scipy.special import hermite
import scipy.special as sp
import csv
from scipy.interpolate import interp1d
import os
from wfplot import *
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker

def import_psi(file_path):
    points = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip header if exists
        for row in csv_reader:
            try:
                x = float(row[0])  # assuming the first column contains x coordinates
                y = float(row[1])  # assuming the second column contains y coordinates
                points.append((x, y))
            except (IndexError, ValueError):
                print(f"Skipping invalid data: {row}")
    return points

def generate_radial_function(file):
    points = import_psi(file)
    x_values = [point[0] for point in points]
    x_values.insert(0, 0)
    y_values = [point[1] for point in points]
    y_values.insert(0, 0)
    radial = interp1d(x_values, y_values, kind='linear')
    return radial

def quantum_harmonic_eigenstate(n, x, omega):
    prefactor = 1 / np.sqrt(2**n * np.math.factorial(n)) * (omega / (np.pi)) ** 0.25
    hermite_term = hermite(n)(np.sqrt(omega) * x)
    gaussian_term = np.exp(- x**2 * omega / 2)
    return prefactor * hermite_term * gaussian_term

def quantum_2d_harmonic_eigenstate(nx, ny, x, y, omega_x, omega_y):
    prefactor_x = 1 / np.sqrt(2**nx * np.math.factorial(nx)) * (omega_x / (np.pi)) ** 0.25
    prefactor_y = 1 / np.sqrt(2**ny * np.math.factorial(ny)) * (omega_y / (np.pi)) ** 0.25
    hermite_term_x = hermite(nx)(np.sqrt(omega_x) * x)
    hermite_term_y = hermite(ny)(np.sqrt(omega_y) * y)
    gaussian_term = np.exp(- (x**2 * omega_x + y**2 * omega_y) / 2)
    return prefactor_x * prefactor_y * hermite_term_x * hermite_term_y * gaussian_term

def quantum_3d_harmonic_eigenstate(nx, ny, nz, x, y, z, omega_x, omega_y, omega_z):
    prefactor_x = 1 / np.sqrt(2**nx * np.math.factorial(nx)) * (omega_x / (np.pi)) ** 0.25
    prefactor_y = 1 / np.sqrt(2**ny * np.math.factorial(ny)) * (omega_y / (np.pi)) ** 0.25
    prefactor_z = 1 / np.sqrt(2**nz * np.math.factorial(nz)) * (omega_z / (np.pi)) ** 0.25
    hermite_term_x = hermite(nx)(np.sqrt(omega_x) * x)
    hermite_term_y = hermite(ny)(np.sqrt(omega_y) * y)
    hermite_term_z = hermite(nz)(np.sqrt(omega_z) * z)
    gaussian_term = np.exp(- (x**2 * omega_x + y**2 * omega_y + z**2 * omega_z) / 2)
    return prefactor_x * prefactor_y * prefactor_z * hermite_term_x * hermite_term_y * hermite_term_z * gaussian_term

def hydrogen_3d_eigenstate(n, l, m, x, y, z):
    # Handling special case when x, y, and z are all zero
    if x == 0 and y == 0 and z == 0:
        return 0
    else:
        # Spherical harmonic (3D)
        if x == 0 and y == 0:
            if z > 0:
                phi = np.pi / 2
            else:
                phi = -np.pi / 2
        else:
            phi = np.arctan2(y, x)
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        spherical_harmonic = sp.sph_harm(m, l, theta, phi)
        # Radial part (1s orbital for hydrogen atom)
        r = np.sqrt(x**2 + y**2 + z**2)
        radial_part = np.exp(-r / 2) * sp.genlaguerre(n - l - 1, 2*l + 1)(r)
        return radial_part * spherical_harmonic
    
def angular_function(m, l, theta, phi):
    legendre = sp.lpmv(m, l, np.cos(theta))

    constant_factor = ((-1) ** m) * np.sqrt(
        ((2 * l + 1) * sp.factorial(l - np.abs(m))) /
        (4 * np.pi * sp.factorial(l + np.abs(m)))
    )
    return constant_factor * legendre * np.real(np.exp(1.j * m * phi))

def ident(x):
    return 1

def compute_wavefunction():
    psi = 0

    # x-y grid to represent electron spatial distribution
    grid_extent = 4
    grid_resolution = 680
    z = x = np.linspace(-grid_extent, grid_extent, grid_resolution)
    z, x = np.meshgrid(z, x)

    psi_unnormalized = 1/np.sqrt(2*grid_extent) * quantum_harmonic_eigenstate(0, x, 1) * ident(z)
    
    # Normalize the wavefunction
    dx = (2 * grid_extent) / (grid_resolution - 1)
    dy = dx  # assuming square grid
    normalization_constant = np.sum(np.abs(psi_unnormalized)**2) * dx * dy
    psi = psi_unnormalized / np.sqrt(normalization_constant)
    
    return psi

def compute_wavefunction_ho(n, omega):
    psi = 0

    # x-y grid to represent electron spatial distribution
    grid_extent = 4
    grid_resolution = 680
    z = x = np.linspace(-grid_extent, grid_extent, grid_resolution)
    z, x = np.meshgrid(z, x)

    psi = 1/np.sqrt(2*grid_extent) * quantum_harmonic_eigenstate(n, x, omega) * ident(z)
    
    return psi

def compute_wavefunction_dy(directory):
    psi = 0

    # x-y grid to represent electron spatial distribution
    grid_extent = 2
    grid_resolution = 680
    z = x = np.linspace(-grid_extent, grid_extent, grid_resolution)
    z, x = np.meshgrid(z, x)

    # Use epsilon to avoid division by zero during angle calculations
    eps = np.finfo(float).eps

    normalization_constant = 0

    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        radial = generate_radial_function(file)
        psi += radial(
            np.sqrt((x ** 2 + z ** 2))
        ) * angular_function(
            0, 2*(int(filename[4])-1), np.arctan(x / (z + eps)), 0
        )

        # Accumulate the sum of squares of the wavefunction for normalization
        normalization_constant += np.sum(np.abs(psi)**2)

    # Normalize the wavefunction
    dx = (2 * grid_extent) / (grid_resolution - 1)
    dy = dx  # assuming square grid
    normalization_constant *= dx * dy
    psi /= np.sqrt(normalization_constant)
    
    return psi

def compute_probability_density(psi):
    return np.abs(psi) ** 2

def plot_wf_probability_density_xz(psi, dark_theme=False, colormap='rocket'):

    # Colormap validation
    try:
        sns.color_palette(colormap)
    except ValueError:
        raise ValueError(f'{colormap} is not a recognized Seaborn colormap.')

    # Configure plot aesthetics using matplotlib rcParams settings
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

    # Compute and visualize the wavefunction probability density
    prob_density = compute_probability_density(psi)
    
    # Here we transpose the array to align the calculated z-x plane with Matplotlib's y-x imshow display
    im = ax.imshow(np.sqrt(prob_density).T, cmap=sns.color_palette(colormap, as_cmap=True))
    locator = ticker.MultipleLocator(0.5)
    ax.xaxis.set_major_locator(locator)


    cbar = plt.colorbar(im, fraction=0.046, pad=0.03)
    cbar.set_ticks([])

    # Apply dark theme parameters
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
    
def integrate_wf(psi1, psi2, dx):
    column_sums = dx * np.sum(psi1*psi2, axis=0)
    length = len(column_sums)
    correlation = column_sums[:length // 2]
    correlation = correlation[::-1]
    return correlation**2

# Example usage:
n = 2  # Energy level of the eigenstate
x = 0.5  # Position at which to evaluate the eigenstate
omega = 1.0  # Angular frequency of the harmonic oscillator
result = quantum_harmonic_eigenstate(n, x, omega)
print(f"ψ_{n}({x}) = {result}")

grid_extent = 2
grid_resolution = 680

dx = (2 * grid_extent) / (grid_resolution - 1)
dy = dx  # assuming square grid

plt.figure(figsize=(12, 6), dpi=300)
plt.grid(visible=True)

for i in range(0, 10, 2):
    psi1 = compute_wavefunction_dy("C:\\Users\Oem\Desktop\Dipole\Radial_functions")
    psi2 = compute_wavefunction_ho(i, 2)
    psi1 /= dx**(2) * np.sum(psi1*psi1)
    psi2 /= dx**(2) * np.sum(psi2*psi2)
    correlation = integrate_wf(psi1, psi2, dx)
    x_values = range(len(correlation))
    plt.plot(x_values, correlation**2)

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot of correlation')
plt.show()

print(psi1)

print(dx*np.sum(correlation))
print(dx)
