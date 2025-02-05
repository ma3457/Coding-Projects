from Bio import SeqIO
import os
import math

# Input file and desired number of chunks
input_fasta = "Streptococcusmutanslab1.fasta"
num_files = 10  # Set the desired number of files
output_folder = "fasta_chunks"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the sequences from the FASTA file
sequences = list(SeqIO.parse(input_fasta, "fasta"))
total_sequences = len(sequences)
chunk_size = math.ceil(total_sequences / num_files)  # Calculate chunk size based on the total number of sequences

print(f"Total sequences: {total_sequences}")
print(f"Splitting into {num_files} files, with up to {chunk_size} sequences per file.")

# Write the sequences into smaller FASTA files
for i in range(num_files):
    start_idx = i * chunk_size
    end_idx = min(start_idx + chunk_size, total_sequences)
    chunk_sequences = sequences[start_idx:end_idx]

    if not chunk_sequences:
        break

    chunk_filename = os.path.join(output_folder, f"chunk_{i+1}.fasta")
    with open(chunk_filename, "w") as output_handle:
        SeqIO.write(chunk_sequences, output_handle, "fasta")
    print(f"Created {chunk_filename} with {len(chunk_sequences)} sequences.")