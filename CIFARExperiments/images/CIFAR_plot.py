
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:20:07 2024

@author: lidon
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

checkpoints = [64, 128, 192, 249]
percentages = ['25%', '50%', '75%', '100%']

SGBJ_data = pd.read_csv('SOME FILE')
SGLD_data = pd.read_csv('SOME FILE')
SGHMC_data = pd.read_csv('SOME FILE')
SGNHT_data = pd.read_csv('SOME FILE')



# Extract the step size from the 'Sampler' column and add it to the dataframe as a new column

for df in [SGBJ_data,SGLD_data,SGHMC_data,SGNHT_data]:
    df['step size'] = df['Sampler'].str.extract(r'step size=(\d+\.?\d*e?-?\d*)')[0].astype(float)

    # Sort the dataframe based on the 'step size' column to ensure correct plotting
    df = df.sort_values(by=['step size'])

# Iterate through the specified checkpoints and create a plot for each
for i, checkpoint in enumerate(checkpoints):
    print(f'printing {checkpoint}...')
    # Filter the data for the current checkpoint
    SGBJ_checkpoint = SGBJ_data[SGBJ_data['Checkpoint'] == checkpoint]
    SGLD_checkpoint = SGLD_data[SGLD_data['Checkpoint'] == checkpoint]
    SGHMC_checkpoint = SGHMC_data[SGHMC_data['Checkpoint'] == checkpoint]
    SGNHT_checkpoint = SGNHT_data[SGNHT_data['Checkpoint'] == checkpoint]

    SGBJ = SGBJ_checkpoint[SGBJ_checkpoint['Sampler'].str.contains('barker')]
    SGLD = SGLD_checkpoint[SGLD_checkpoint['Sampler'].str.contains('langevin')]
    SGHMC = SGHMC_checkpoint[SGHMC_checkpoint['Sampler'].str.contains('hamilton')]
    SGNHT = SGNHT_checkpoint[SGNHT_checkpoint['Sampler'].str.contains('hoover')]

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot ACC for each sampler
    
    plt.errorbar(SGBJ['step size'], 
                 SGBJ['ACC'], yerr=2 * SGBJ['std'] , label='SGBJ', markersize=10,
                 marker='o', linestyle='-', capsize=10)
    
    
    plt.errorbar(SGLD['step size'], 
                 SGLD['ACC'], yerr=2* SGLD['std'] , label='SGLD', markersize=10,
                 marker='x', linestyle='-', capsize=10)
    plt.errorbar(SGHMC['step size'], 
                 SGHMC['ACC'], yerr=2* SGHMC['std'] , label='SGHMC', markersize=10,
                 marker='h', linestyle='-', capsize=10)
    
    plt.errorbar(SGNHT['step size'], 
                 SGNHT['ACC'], yerr=2 *  SGNHT['std'] , label='SGNHT',markersize=10,
                 marker='d', linestyle='-', capsize=10)

    
    # Set plot labels and title
    plt.xlabel('Step Size')
    plt.ylabel('ACC')
    plt.title(f'ACC vs Step Size at Checkpoint {checkpoint} ({percentages[i]})')
    plt.legend()

    # Set x-axis to log scale
    plt.xscale('log')

    # Save the plot as an image file
    plt.savefig(f'CIFAR_{checkpoint}.png')

    # Show the plot (optional)
    plt.show()