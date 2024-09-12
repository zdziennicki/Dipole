import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


# Read the CSV file
data = pd.read_csv('Spectral_colored.csv', header=None)
data1 = pd.read_csv('Scattered_01.csv', header=None)
data1 = data1.sort_values(by=0)
data2 = pd.read_csv('Before_scat_colored.csv', header=None)

# Data extraction
x = data[0]
y = data[1]
color = data[2]
x1 = data1[0]
y1 = data1[1]
x2 = data2[0]
y2 = data2[1]
color2 = data2[2]


#Before scattering code
cm = plt.cm.jet
norm = matplotlib.colors.Normalize(vmin=min(color), vmax=max(color))
sc=plt.scatter(x2, y2, marker='s', c=color2, vmin=min(color2), vmax=max(color2), cmap=cm, s=12, edgecolors='0.3', linewidths=0.1)
plt.gca().spines['top'].set_linewidth(2)  
plt.gca().spines['bottom'].set_linewidth(2)  
plt.gca().spines['left'].set_linewidth(2)  
plt.gca().spines['right'].set_linewidth(2) 
plt.grid(True, alpha=0.1)
plt.xlim(-100, 100)  # Set the range of the x-axis
plt.ylim(0, 20)  # Set the range of the x-axis
plt.xlabel('$R_{vdW}/a_{1D}$')
plt.ylabel('$E/E_{\omega}$')
plt.title('3D Bound state energy')
plt.colorbar(sc)
plt.savefig('3D_Bound_state_energy_before_scat.png')

#clear plot
plt.clf()

# Bound state energy
cm = plt.cm.jet
norm = matplotlib.colors.Normalize(vmin=min(color), vmax=max(color))
sc=plt.scatter(x, y, marker='s', c=color, vmin=min(color), vmax=max(color), cmap=cm, s=5, edgecolors='0.3', linewidths=0.1)
plt.gca().spines['top'].set_linewidth(2)  
plt.gca().spines['bottom'].set_linewidth(2)  
plt.gca().spines['left'].set_linewidth(2)  
plt.gca().spines['right'].set_linewidth(2) 
plt.grid(True, alpha=0.1)
plt.xlim(-100, 100)  # Set the range of the x-axis
plt.ylim(0, 20)  # Set the range of the x-axis
plt.xlabel('$R_{vdW}/a_{3D}$')
plt.ylabel('$E/E_{\omega}$')
plt.title('3D Bound state energy')
plt.colorbar(sc)
plt.savefig('3D_Bound_state_energy.png')

#clear plot
plt.clf()

# Bound state energy
plt.plot(x1, y1)
lin=np.linspace(-3,1,100)
plt.plot(lin, lin, '--')
plt.gca().spines['top'].set_linewidth(2)  
plt.gca().spines['bottom'].set_linewidth(2)  
plt.gca().spines['left'].set_linewidth(2)  
plt.gca().spines['right'].set_linewidth(2) 
plt.grid(True, alpha=0.1)
plt.xlim(-3, 1)  # Set the range of the x-axis
plt.ylim(-2, 2)  # Set the range of the x-axis
plt.xlabel('$a_{1D}$')
plt.ylabel('$a_{3D}$')
plt.title('Scattering length conversion')
plt.savefig('Scattering_length_conversion.png')