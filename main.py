import numpy as np
import matplotlib.pyplot as plt
import csv
from radialplot import plot2Dharm
from wfplot import *
from scipy.integrate import quad

def main():
    # Import parameters of the simulation
    parameters = import_parameters("Parameters.csv")
    
    # Plot radial functions
    plot2Dharm("Radial_functions")

    # Plot wavefunction probability density
    plot_wf_probability_density("Radial_functions", True, colormap='mako')

    # Compute wavefunctions
    psi1 = compute_wavefunction("Radial_functions")
    correlations = []
    
    for i in range(0,17,2):
        psi2 = compute_hermite(i,parameters["beta"])
    
        # Calculate r^2 matrix
        rows, columns = psi1.shape
        rsq = calculate_r_squared((rows, columns))
    
        # Calculate correlation
        correlation = calculate_correlation(psi1, psi2, rsq)
        correlations.append(((8 / 680)*np.sum(correlation))**2)

        # Plot correlation
        x_values = np.linspace(0, 4, len(correlation))
        plot_correlation(x_values, correlation)
    
        # Save correlation data to CSV
        save_correlation_data('correlation_data_{}.csv'.format(i), x_values, correlation)
        
    print(correlations)

if __name__ == "__main__":
    main()


