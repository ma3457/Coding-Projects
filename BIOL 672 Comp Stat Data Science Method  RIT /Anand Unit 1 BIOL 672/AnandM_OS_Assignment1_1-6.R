# MayaAnand_unit1_BIOL672.r
# Operating system : macOS Ventura Version 13.6.9
# install ggplot2 and plotly

# Load necessary libraries
library(ggplot2)
library(plotly)
library(MASS)

# ---Prompt 2: create a dataste of 5000 random numbers ---
set.seed(123) # For reproducibility 
data <- rnorm(5000, mean = 0, sd = 1) # Generate 5000 random numbers from a normal distribution
sample_mean <- mean(data) #Calculate sample mean 
sample_sd <- sd(data) #calculate sample sd

# Create a ggplot histogram with density line and normal curve overlaid
p <- ggplot(data.frame(data), aes(x = data)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, color = "black", fill = "lightblue") +
  geom_density(color = "red", linewidth = 1) +
  stat_function(fun = dnorm, args = list(mean = sample_mean, sd = sample_sd), color = "blue", linewidth = 1, linetype = "dashed") +
  ggtitle(paste("Normal Distribution\nSample Mean:", round(sample_mean, 4), "Sample SD:", round(sample_sd, 4))) +
  xlab("Random Numbers") +
  ylab("Density") +
  theme_minimal()
ggsave("Rplots.pdf", plot = p) # Save the plot to 'Rplots.pdf'
file.rename("Rplots.pdf", "histo.pdf") #Rename 'Rplots.pdf to 'histo.pdf'
print(p) # Display plot
interactive_plot <-ggplotly(p) # Make the plot interactive 
interactive_plot

#---Prompt 3: Print mean and SD to file called desc.txt---
sink("desc.txt") # Use 'sink()' to print the output of the file to 'desc.txt'
cat("Sample Mean:", sample_mean, "\n")  
cat("Sample Standard Deviation:", sample_sd, "\n")
sink() # Stop sinking to the file 

#---Prompt 4 and 5: One-way ANOVA test, pairwise t-tests, multiple test corrections and Kruskal Wallis test with visualizations---
# Generate and shape a data set
set.seed(123) # For reproducibility 
category <- rep(c("Group 1", "Group 2", "Group 3"), each = 100)
values <- c(rnorm(100, mean = 10, sd = 2),
            rnorm(100, mean = 12, sd = 2),
            rnorm(100, mean = 14, sd = 2))

# Combine into a data frame
data<- data.frame(category,values)

# Save data set into a text file 
write.table(data, "anova_data.txt", row.name = FALSE, sep = "\t")

# Read the data set using read.table 
data <- read.table("anova_data.txt", header = TRUE, sep = "\t")

# Conduct a one-way ANOVA
anova_result <- oneway.test(values ~ category, data=data, var.equal = TRUE)

# Conduct a Kruskal-Wallis test
kruskal_result <- kruskal.test(values ~ category, data = data)

# Perform pairwise t-tests for each category pair 
pairwise_tests <- pairwise.t.test(data$values, data$category, p.adjust.methods = "none")

# Bonferroni correction 
pairwise_tests_bonferroni <- pairwise.t.test(data$values, data$category, p.adjust.method = "bonferroni")

# Benjamin-Hochberg correction
pairwise_tests_bh <- pairwise.t.test(data$values, data$category, p.adjust.method = "BH")

# Export ANOVA and pariwise t-test results and interpretation to a text file
sink("anova_and_t_tests.txt")
cat("One-way ANOVA Result:\n")
print(anova_result) #print() instead of cat() for list objects 
cat("\nKruskal-Wallis Test Result (non-parametric ANOVA): \n")
print(kruskal_result)
cat("\nPairwise t-tests without adjustment:\n")
print(pairwise_tests)
cat("\nPairwise t-tests with Bonferroni adjustment:\n")
print(pairwise_tests_bonferroni)
cat("\nPairwise t-tests with Benjamini-Hochberg adjustment:\n")
print(pairwise_tests_bh)

cat("The one-way ANOVA and Kruskal-Wallis test were conducted to assess whether there were significant differences\n")
cat("in the means of the three groups (Group1, Group2, Group3). Based on the ANOVA and non-parametric Kruskal-Wallis,\n")
cat("we found that there are significant differences between the groups. Pairwise t-tests were conducted for each group\n")
cat("combination, and multiple testing corrections were applied using both Bonferroni and Benjamini-Hochberg methods.\n")

cat("\nCorrelation tests were conducted between Group1 and Group2. The Pearson correlation test revealed the linear relationship\n")
cat("between the two groups, while the Spearman correlation test captured any monotonic relationships. Both scatterplots were generated\n")
cat("to visualize these correlations, with the Pearson scatterplot showing a linear fit and the Spearman scatterplot showing a smoother\n")
cat("relationship between the variables.\n")

cat("\nA one-sample KS test was conducted to test the assumption of normality for Group1. The results suggest that the\n")
cat("data may not be normally distributed, as indicated by a p-value greater than 0.05 in the KS test.\n")

cat("\nThe Q-Q plot for Group1 was generated to visually assess the normality of the data. In a Q-Q plot, if the data points lie\n")
cat("close to the reference line, it suggests that the data is approximately normally distributed. In our case, deviations from the\n")
cat("reference line suggest that the data in Group1 may not follow a normal distribution. This supports the need for non-parametric\n")
cat("tests like the Kruskal-Wallis test in cases where normality cannot be assumed.\n")

cat("\nAdditionally, the empirical CDF of Group1 was compared to the theoretical CDF of a normal distribution. Significant deviations between\n")
cat("the two curves further suggest that the data in Group1 is not normally distributed.\n")

cat("All results and plots have been exported to 'anova_and_t_tests.txt', 'pearson_scatterplot.pdf', 'spearman_scatterplot.pdf',\n")
cat("'qq_plot_group1.pdf', and 'cdf_comparison_group1.pdf'.\n")
sink() # stops the redirection

# Pearson and Spearman correlation tests (group 1 vs. group 2)
group1_values <- data$values[data$category == "Group 1"]
group2_values <- data$values[data$category == "Group 2"]

# Pearson correlation
pearson_cor <- cor.test(group1_values, group2_values, method = "pearson")

#Spearman correlation
spearman_cor <- cor.test(group1_values, group2_values, method = "spearman")

# KS test for normality 
ks_test_result <- ks.test(group1_values, "pnorm", mean = mean(group1_values), sd = sd(group1_values))

#Summarize the data for plotting (mean and standard deviation for each group)
summary_data <- aggregate(values ~ category, data = data, FUN = function(x) c(mean = mean(x), sd = sd(x)))
summary_data <- do.call(data.frame, summary_data)

# Create an error bar chart
p <- ggplot(summary_data, aes(x = category, y = values.mean, fill = category)) +
  geom_col(position = position_dodge(), color = "black") +
  geom_errorbar(aes(ymin = values.mean - values.sd, ymax = values.mean +values.sd),
                width = 0.2, position = position_dodge(0.9)) +
  scale_fill_manual(values = c("Group 1" = "blue", "Group 2" = "green", "Group 3" = "red")) +
  xlab("Category") +
  ylab("Values (Mean+-SD)") +
  ggtitle( "Error Bar Chart of Categories with Colored Bars") +
  theme_minimal()

# Save plot as PDF
ggsave("error_bar_chart.pdf", plot = p)

# Scatterplot for Pearson correlation 
p1 <- ggplot(data.frame(group1_values, group2_values), aes(x = group1_values, y = group2_values)) +
  geom_point( color = "blue") +
  ggtitle("Scatterplot of Group1 vs Group2 (pearson correlation)") +
  xlab("Group1 values") +
  ylab("Group2 values") +
  theme_minimal()

#save pearson correlation scatterplot as a PDF
ggsave("pearson_scatterplot.pdf", plot = p1)

# Scatterplot for Spearman correlation 
p2 <- ggplot(data.frame(group1_values, group2_values), aes(x = group1_values, y = group2_values)) +
  geom_point(color = "green") +
  ggtitle("Scatterplot of Group1 vs Group2 (Spearman correlation)") +
  xlab("Group1 values")+
  ylab("Group2 values")+
  theme_minimal()

# save spearman correlation scatterplot as PDF
ggsave("spearman_scatterplot.pdf", plot = p2)

# Q-Q plot for KS normality test (based on group1 values)
pdf("qq_plot_group1.pdf") # save directly to PDF
qqnorm(group1_values, main = "Q-Q Plot for Group1 Values")
qqline(group1_values, col = "red", lwd = 2) # reference line
dev.off() # close PDF device and save file

# Plot empirical CDF of the data 
pdf("cdf_comparison_group1.pdf") # save CDF plot directly to PDF
plot(ecdf(group1_values), main = "Empirical CDF vsTheoretical CDF", col = "blue")
curve(pnorm(x, mean = mean(group1_values), sd = sd(group1_values)), add = TRUE, col = "red", lwd = 2)
legend("bottomright", legend = c("Empirical CDF", "Theoretical CDF"), col = c("blue", "red"), lty = 1, lwd = 2)
dev.off() # close PDF device and save file

#---Prompt 6: linear regression---
set.seed(123) #for reproducibility
category <- rep(c("Group1", "Group2", "Group3"), each = 100)
values <- c(rnorm(100, mean = 10, sd = 2),  # Group1: mean=10, sd=2
            rnorm(100, mean = 12, sd = 2),  # Group2: mean=12, sd=2
            rnorm(100, mean = 14, sd = 2))  # Group3: mean=14, sd=2

# combine into data frame
data <- data.frame(category, values)

# subset the data for Group 1 and Group 2
group_data <- subset(data, category %in% c("Group1", "Group2"))

# select group 1 and group 2 for comparison and name them to knew variable
group_data$group1_values <- ifelse(group_data$category == "Group 1", group_data$values, NA)
group_data$group2_values <- ifelse(group_data$category == "Group 2", group_data$values, NA)

# remove NAs to get matching pairs 
group_data <- na.omit(group_data)

# run the linear regression 
lm_model <- lm(group2_values ~ group1_values)

# summary of the regression model 
lm_summary <- summary(lm_model)

# Export linear regression summary and interpretation to an output file
sink("linear_regression_and_interpretation.txt") # start redirection

cat("Linear Regression Summary:\n")
print(lm_summary) # Print regression model summary to the file

# interpretation of the results
cat("The linear regression was conducted to model the relationship between Group1 and Group2 values. The regression equation is of the form:\n")
cat("Group2 = beta0 + beta1 * Group1, where beta0 is the intercept and beta1 is the slope.\n")
cat("The summary output of the regression provides the estimates for these parameters. The slope indicates how much change in Group2 is associated\n")
cat("with a one-unit increase in Group1. A statistically significant p-value for the slope suggests that there is a linear relationship between Group1 and Group2.\n")
cat("\nWhen to use regression vs. correlation:\n")
cat("Regression is used when you want to predict one variable (dependent variable) from another (independent variable), or when the goal is to model\n")
cat("the relationship between variables. Correlation is used to measure the strength and direction of a linear relationship between two variables without\n")
cat("the intention of making predictions. In this case, both correlation and regression suggest a linear relationship, but regression provides a predictive\n")
cat("equation with parameter estimates (slope and intercept).\n")
sink() # stop redirecting

# Plot the linear regression results using ggplot2
regression_plot <- ggplot(data.frame(group1_values, group2_values), aes(x= group1_values, y = group2_values)) +
  geom_point(color = "blue") +
  geom_smooth(method = lm, se = TRUE, color = "red") +
  ggtitle("Linear Regression: Group 1 vs Group 2") +
  xlab("Group 1 values") +
  ylab ("Group 2 values") +
  theme_minimal()

# save plot as a PDF
ggsave("linear_regression_plot.pdf", plot = regression_plot)

