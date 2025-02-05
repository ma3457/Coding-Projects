Principal Component Analysis (PCA) of Protein Phosphorylation Levels:
- This project analyzes protein phosphorylation levels using Principal Component Analysis (PCA) to explore patterns of variation in the dataset. 
The analysis highlights key proteins contributing to variance, which may help identify significant roles in cellular signaling pathways.

Files in the Project:
- GSE150575_ISPY2_MK2206_RawToNorm_Params_RPPA1.csv.gz
- GSE150575_ISPY2_MK2206_RawToNorm_Params_RPPA2.csv.gz
- HER2_Signaling_Analysis.R

R script performing the following tasks:
- Loading and combining the datasets.
- Preparing and standardizing the data.
- Performing Principal Component Analysis (PCA).
- Visualizing the PCA results as a scatter plot with color gradients.

Install Required Packages:
- tidyverse
- readr
- ggplot2
- factoextra