# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:47:20 2024

@author: Oem
"""

import pandas as pd
import matplotlib.pyplot as plt

# Import data from CSV
data = pd.read_csv('Dysprosium_30kHz.csv', header=None)
data.columns = ['x', 'y']  

print(data)
data['x']=1/(data['x'])
data['y']= - data['y']

# Plot data
plt.scatter(data['x'], data['y'], marker = '^')
plt.yscale('log')

plt.xlim(0,50)
plt.ylim(0,40)


plt.title('Your Title')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.grid(True)
plt.show()
