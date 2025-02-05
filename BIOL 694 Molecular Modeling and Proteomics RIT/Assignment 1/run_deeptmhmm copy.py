import os
import subprocess

# Define input folder, output folder, and DeepTMHMM command
input_folder = "fasta_chunks"
output_folder = "deeptmhmm_outputs"
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

# Iterate through each FASTA chunk file
for fasta_file in sorted(os.listdir(input_folder)):
    if fasta_file.endswith(".fasta"):  # Only process FASTA files
        input_path = os.path.join(input_folder, fasta_file)
        output_path = os.path.join(output_folder, f"{fasta_file}_results.txt")
        
        print(f"Running DeepTMHMM on {fasta_file}...")

        # Construct the biolib command
        cmd = [
            "biolib",
            "run",
            "DTU/DeepTMHMM",
            "--fasta",
            input_path,
        ]

        # Run the command and redirect the output to a file
        with open(output_path, "w") as output_file:
            subprocess.run(cmd, stdout=output_file, stderr=subprocess.STDOUT)

        print(f"Results saved to {output_path}")

print("All files processed!")