# you can use the csv module to write csv files
# https://docs.python.org/3/library/csv.html
import csv

# for a list of proteins, you are creating a list of properties, e.g:
proteins = ["ID1", "ID2", "ID3"]
mw_list = [123, 456, 789]

# to write them into a file, you first have to open a file:
with open("protein_properties.csv", "w") as csv_out:
    # then you can use the csv DictWriter class to define how the csv should look like
    writer = csv.DictWriter(
        csv_out, fieldnames=["Protein ID", "MW"], lineterminator="\n"
    )
    # and then you can write the header
    writer.writeheader()
    # now you can iterate over the protein entries and write them into a csv:
    for n, prot in enumerate(proteins):
        writer.writerow({"Protein ID": prot, "MW": mw[n]})
