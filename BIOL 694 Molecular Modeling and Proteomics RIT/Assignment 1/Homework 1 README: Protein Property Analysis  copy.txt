Homework 1 README: Protein Property Analysis for Streptococcus mutans
Library Requirements:
- pandas
- maplotlib
- biopython
- deepTMHMM 
- biopython
- pyteomics
- plotly
- csv
- subprocess
- os
Input files:
- Streptococcusmutanslab1.fasta from UniProt
Scripts
- assignment1_script.py:
- split_fasta.py:
- run_deeptmhmm.py:
- parse_topologies.py:
- categorizie_proteins.py: 
How to run code:
- download the input file
- run assignment1_script.py to generate molecular weights and isoelectric points
- run split_fasta.py to split the streptococcusmutanslab1.fasta into smaller chunks
- run those chunks in the DeepTMHMM 
- run categorize_proteins.py to read .3line files, classify proteins, and produce visualizations

