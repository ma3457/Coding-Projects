import os
import csv

# Define the folder containing the files
input_folder = "/Users/maya.anand/Desktop/MM&P/assignment1-protein-properties-ma3457/biolib_results"
output_csv = "parsed_topologies_combined.csv"

# Open the output CSV file
with open(output_csv, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the header row
    csv_writer.writerow(["Protein ID", "Topology Prediction"])
    
    # Iterate through the files in the folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Look for files with 'predicted_topologies.3line' in the name
            if "predicted_topologies.3line" in file:
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")
                
                # Open and parse the file
                with open(file_path, "r") as f:
                    for line in f:
                        # Parse the lines with sequence ID and topology
                        if line.startswith(">"):
                            protein_id = line.strip().lstrip(">")
                            topology = next(f).strip()  # Read the next line for the topology
                            # Write to the CSV file
                            csv_writer.writerow([protein_id, topology])

print(f"Parsing complete! Results saved to {output_csv}.")