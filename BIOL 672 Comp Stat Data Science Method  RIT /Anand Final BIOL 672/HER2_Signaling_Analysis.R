## AnandMA_FinalAssingment_BIOL672.r
# Operating system : macOS Sequoia Version 15.0.1

# Load necessary libraries
library(tidyverse)
library(readr)
library(ggplot2)
library(factoextra)

# Load the data files
data1 <- read_csv("/Users/maya.anand/Desktop/Anand Comp Stat BIOL 672/Anand Final Assignment BIOL 672/GSE150575_ISPY2_MK2206_RawToNorm_Params_RPPA1.csv.gz")
data2 <- read_csv("/Users/maya.anand/Desktop/Anand Comp Stat BIOL 672/Anand Final Assignment BIOL 672/GSE150575_ISPY2_MK2206_RawToNorm_Params_RPPA2.csv.gz")

# Combine the datasets based on 'ID_Ref'
combined_data <- full_join(data1, data2, by = "ID_Ref") %>%
  mutate(across(everything(), ~replace_na(., 0)))  # Replace NA with 0, if necessary

# Prepare the data for PCA
data_pca <- select(combined_data, contains("mean"))
data_pca <- replace_na(data_pca, list(`RPPA1_raw_to_norm:mean` = 0, `RPPA2_raw_to_norm:mean` = 0))

# Standardize the data
data_pca_scaled <- scale(data_pca)

# Perform PCA
pca_result <- prcomp(data_pca_scaled, center = TRUE, scale. = TRUE)

# Check if rownames are set, if not, set them
if (is.null(rownames(pca_result$x))) {
  rownames(pca_result$x) <- 1:nrow(pca_result$x)
}

# Create a data frame that includes PCA scores and adds a color gradient based on their proximity to the origin
pca_data <- data.frame(
  Individual = rownames(pca_result$x),
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  Distance = sqrt(pca_result$x[, 1]^2 + pca_result$x[, 2]^2)
)

# Generate PCA plot with enhanced visual details
pca_plot <- ggplot(pca_data, aes(x = PC1, y = PC2, color = Distance)) +
  geom_point(alpha = 0.8) +
  scale_color_gradient(low = "#00AFBB", high = "#FC4E07") +
  theme_minimal() +
  labs(title = "Enhanced PCA of Protein Phosphorylation Levels",
       caption = 
       "This scatter plot illustrates the distribution of protein phosphorylation levels analyzed through principal component analysis (PCA). The first principal 
       component (horizontal axis) explains 73.93% of the variance, highlighting the major patterns of variation in phosphorylation across the dataset, while the 
       second principal component (vertical axis) accounts for the remaining 26.07% of the variance. Each point represents a unique protein, color-coded based on 
       their 'Distance' value, which quantifies the protein's contribution to the total variance.Points with higher 'Distance' values, shown in darker shades, 
       contribute more significantly to the data's structure, identifying them as key drivers in the phosphorylation profile. This analysis helps in identifying 
       proteins with significant roles in cellular signaling pathways, potentially pinpointing targets for further biochemical investigation or therapeutic intervention.",
       x = "Principal Component 1 - Explains 73.93%",
       y = "Principal Component 2 - Explains 26.07%") +
  theme(plot.caption = element_text(size = 8, hjust = 0.5),  # Center the caption
        plot.title = element_text(hjust = 0.5),             # Center the title
        legend.title = element_text(size = 10))

# Print the plot
print(pca_plot)

# Save the enhanced PCA plot to a file
ggsave("Enhanced_PCA_Protein_Phosphorylation_Levels.pdf", plot = pca_plot, width = 12, height = 8)

# Save the enhanced PCA plot to a PDF file
ggsave("Enhanced_PCA_Protein_Phosphorylation_Levels.pdf", plot = pca_plot, device = "pdf", width = 12, height = 8, dpi = 300)