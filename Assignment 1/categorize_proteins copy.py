import os
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.SeqUtils import seq3

# Kyte-Doolittle Hydrophobicity Scale (simplified)
hydrophobicity_scale = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 
    'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def calculate_hydrophobicity(sequence):
    """ Calculate hydrophobicity score for a given protein sequence using Kyte-Doolittle scale. """
    return sum([hydrophobicity_scale.get(aa, 0) for aa in sequence])

def parse_3line(file_path):
    """ Parse the .3line file to extract protein type, localization, and hydrophobicity. """
    protein_info = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

        # Loop through the lines, extracting protein info (customize this based on your file format)
        for i, line in enumerate(lines):
            if line.startswith('>'):  # protein name
                protein_name = line.strip().split('|')[1]  # Extract protein ID from >sp|P08987|GTFB_STRMU
                
                # Check if there is a next line for sequence data
                if i+1 < len(lines):
                    sequence = lines[i+1].strip()
                    
                    # Check if the sequence is non-empty
                    if sequence:
                        # Extract localization info (SP, TM, GLOB)
                        if 'SP' in line:
                            localization = 'secreted'
                        elif 'TM' in line:
                            localization = 'membrane'
                        else:
                            localization = 'cytosolic'

                        # Calculate hydrophobicity based on Kyte-Doolittle scale
                        hydrophobicity = calculate_hydrophobicity(sequence)
                        
                        protein_info.append([protein_name, localization, hydrophobicity])
                    else:
                        print(f"Warning: Empty sequence for {protein_name}, skipping.")
                else:
                    print(f"Warning: No sequence found for {protein_name}, skipping.")
    return protein_info

def process_3line_files(directory):
    all_proteins = []
    for filename in os.listdir(directory):
        if filename.endswith('.3line'):
            file_path = os.path.join(directory, filename)
            proteins = parse_3line(file_path)
            all_proteins.extend(proteins)
    return pd.DataFrame(all_proteins, columns=['Protein Name', 'Localization', 'Hydrophobicity'])

# Specify the directory where your .3line files are stored
directory = "/users/maya.anand/desktop/MM&P/assignment1-protein-properties-ma3457/biolib_results"

# Process all .3line files into a DataFrame
proteome_df = process_3line_files(directory)

# Visualize the hydrophobicity distribution for each localization category
proteome_df['Hydrophobicity'] = proteome_df['Hydrophobicity'].astype(float)

# Plot the distribution
plt.figure(figsize=(10,6))
for localization in proteome_df['Localization'].unique():
    subset = proteome_df[proteome_df['Localization'] == localization]
    plt.hist(subset['Hydrophobicity'], bins=20, alpha=0.5, label=f"{localization} proteins")

plt.xlabel("Hydrophobicity")
plt.ylabel("Frequency")
plt.legend(title="Protein Localization")
plt.title("Hydrophobicity Distribution by Protein Localization")
plt.show()