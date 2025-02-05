# Import required Libraries
# for parsing the FASTA file
from Bio import SeqIO
# for molecular weight distribution
from pyteomics import mass
# for creating plots 
import plotly.graph_objects as go
# for saving results to a CSV file 
import csv 
# For calculating isoelectric point 
from Bio.SeqUtils.ProtParam import ProteinAnalysis
#


mol_weight_list = [] # stores molecular weights
pI_list = [] # stores isoelectric points
protein_ids = [] # stores protein ids

# Opens a CSV file for molecular weights 
with open("SMmolecular_weights.csv", "w", newline="") as mw_csvfile:
    mw_writer = csv.DictWriter(mw_csvfile, fieldnames=["Protein ID", "Molecular Weight"], lineterminator="\n")
    mw_writer.writeheader() # write the header row

    # Opens a CSV file for isoelectric points 
    with open("SM_isoelectric_points.csv", "w", newline="") as pI_csvfile:
        pI_writer = csv.DictWriter(pI_csvfile, fieldnames=["Protein ID", "Isoelectric Point (pI)"], lineterminator="\n")
        pI_writer.writeheader()

        # parse the fasta file using SeqIO
        for record in SeqIO.parse("Streptococcusmutanslab1.fasta", "fasta"):
            # calculate the MW
            mw = mass.calculate_mass(sequence=record.seq)
            mol_weight_list.append(mw)
            protein_ids.append(record.id) # store protein ids
            mw_writer.writerow({"Protein ID": record.id, "Molecular Weight": mw}) # saves mw to the csv file

            #calculate the pI
            pI = ProteinAnalysis(str(record.seq)).isoelectric_point()
            pI_list.append(pI)
            pI_writer.writerow({"Protein ID": record.id, "Isoelectric Point (pI)": pI}) # saves pI to csv file


print(mol_weight_list)

# plot mw distribution
mw_fig = go.Figure()
mw_fig.add_trace(go.Histogram(x=mol_weight_list))
mw_fig.update_layout(
    # figure descriptions 
    title="Distribution of Molecular Weights",
    xaxis_title="Molecular Weight",
    yaxis_title="Frequency"
)
mw_fig.write_image("MW_distribution.jpg") # save as a .jpg file
mw_fig.show()

# plot isoelectric point distribution 
pI_fig = go.Figure()
pI_fig.add_trace(go.Histogram(x=pI_list, marker=dict(color="red")))
pI_fig.update_layout(
    title="Distribution of Isoelectric Points",
    xaxis_title="Isoelectric Point (pI)",
    yaxis_title="Frequency",
)
pI_fig.write_image("pI_distribution.jpg") # save as a .jpg file
pI_fig.show()

# Scatterplot comparing mw and pI
mwvpI_scatterplot_fig = go.Figure()
mwvpI_scatterplot_fig.add_trace(
    go.Scatter(
        x=mol_weight_list,
        y=pI_list,
        mode="markers",
        marker=dict(size=5, color="purple"),
        text=protein_ids, # add protein ids as hover info
    )
)
mwvpI_scatterplot_fig.update_layout(
    title="Molecular Weight vs Isoelectric Point",
    xaxis_title="Molecular Weight",
    yaxis_title="Isoelectric Point",
)
mwvpI_scatterplot_fig.write_image("mw_vs_pI_scatter.jpg") # save scatter plot
mwvpI_scatterplot_fig.show() 