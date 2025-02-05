# https://biopython.org/wiki/SeqIO
from Bio import SeqIO

# https://pyteomics.readthedocs.io/en/latest/api/mass.html
from pyteomics import mass
import plotly.graph_objects as go

# the electrochem class within pyteomics also has functions for pI and hydrophobicity
# https://pyteomics.readthedocs.io/en/latest/electrochem.html

mol_weight_list = []
# parse the fasta file using SeqIO
for record in SeqIO.parse("Streptococcusmutanslab1.fasta", "fasta"):
    # you can use the extracted sequence to calculate the MW
    mw = mass.calculate_mass(sequence=record.seq)
    # you can append the MW to the list outside of the for loop to collect all MWs
    mol_weight_list.append(mw)

print(mol_weight_list)


fig = go.Figure()
fig.add_trace(go.Histogram(x=mol_weight_list))
fig.show()
