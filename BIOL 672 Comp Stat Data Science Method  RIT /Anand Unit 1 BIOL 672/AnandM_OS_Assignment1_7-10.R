# Load necessary libraries
library(readr)     # For reading CSV files
library(dplyr)     # For data manipulation
library(MASS)      # For MANOVA
library(ggplot2)   # For visualizations
library(car) #ANOVA and linear regression
library(rlang) # for symbol conversion

# --- Prompt 7: Load and Shape the Data for Multivariate Statistical Tests ---
# Load your dataset using read.csv
data <- read.csv("EPL_20_21.csv", quote = "\"", na.strings = c("", "NA"))

# Check the structure of the data to ensure it is loaded correctly
str(data)

# Inspect the first few rows of the dataset
head(data)

# --- Step to Define Quantitative and Categorical Variables ---
# Select quantitative variables
quantitative_vars <- data[c("Goals", "Assists", "Passes_Attempted", "Perc_Passes_Completed")]

# Check the selected variables to confirm they exist
print(head(quantitative_vars))

# Assign the categorical variable
categorical_var <- data$Club

# Print unique clubs to confirm the categorical variable
print(unique(categorical_var))

# --- Prompt 8: Perform MANOVA using 'manova' function in R ---

# Combine the quantitative variables into a matrix using 'cbind'
dependent_vars <- cbind(data$Goals, data$Assists, data$Passes_Attempted, data$Perc_Passes_Completed)

# Perform MANOVA using Club as the grouping variable
manova_result <- manova(dependent_vars ~ Club, data = data)

# Output the summary of the MANOVA result
manova_summary <- summary(manova_result)

# Print the MANOVA summary
print(manova_summary)

# Interpretation of results
cat("\n\n--- MANOVA Results Interpretation ---\n")
cat("The MANOVA test was performed to determine if the means of Goals, Assists, Passes_Attempted, and Perc_Passes_Completed differ significantly across Clubs.\n")
cat("If the p-value is less than 0.05, it indicates that there is a significant difference in the mean values of the quantitative variables across Clubs.\n")


# --- Prompt 9: Multiple Regression using 'lm' function ---

# Run multiple regression to predict 'Goals' using other variables in the entire dataset
lm_result <- lm(Goals ~ Assists + Passes_Attempted + Perc_Passes_Completed, data = data)
summary_lm_result <- summary(lm_result)

# Print the summary of the multiple regression
cat("\n\n--- Multiple Regression Results (Entire Dataset) ---\n")
print(summary_lm_result)

# Determine the best predictor based on the summary
best_predictor <- which.max(abs(coef(summary_lm_result)[-1, "t value"]))
best_predictor_name <- rownames(coef(summary_lm_result))[-1][best_predictor]
cat("The best predictor for Goals in the entire dataset is:", best_predictor_name, "\n")

# Multiple regression within one category (e.g., within 'Chelsea' club)
chelsea_data <- data %>% filter(Club == "Chelsea")
lm_chelsea <- lm(Goals ~ Assists + Passes_Attempted + Perc_Passes_Completed, data = chelsea_data)
summary_lm_chelsea <- summary(lm_chelsea)

# Print the summary for the Chelsea subset
cat("\n\n--- Multiple Regression Results for Chelsea ---\n")
print(summary_lm_chelsea)

# Determine the best predictor within Chelsea
best_predictor_chelsea <- which.max(abs(coef(summary_lm_chelsea)[-1, "t value"]))
best_predictor_chelsea_name <- rownames(coef(summary_lm_chelsea))[-1][best_predictor_chelsea]
cat("The best predictor for Goals within Chelsea is:", best_predictor_chelsea_name, "\n")

# Interpretation
cat("\n--- Interpretation of Results ---\n")
cat("In the entire dataset, the best predictor of Goals is", best_predictor_name, 
    "with a significant p-value indicating a strong relationship.\n")
cat("Within the Chelsea club, the best predictor is", best_predictor_chelsea_name, 
    "which may differ due to the club's specific playing style or player contributions.\n")
# --- Prompt 10: Composite Variable and ANCOVA ---

# Create a composite variable
# Composite Performance = (Goals * Assists) / Passes_Attempted
data$composite_performance <- (data$Goals * data$Assists) / data$Passes_Attempted

# Run ANCOVA: Compare the relation between 'Goals' predicted by 'Assists' while controlling for 'composite_performance'
ancova_result <- aov(Goals ~ Assists + composite_performance, data = data)
summary(ancova_result)

# Print the summary of the ANCOVA
cat("\n\n--- ANCOVA Results ---\n")
cat("This ANCOVA compares the relationship between Goals (dependent variable) and Assists (independent variable),\n")
cat("while controlling for the composite performance index created as (Goals * Assists) / Passes_Attempted.\n")
cat("The significance of the composite variable indicates its contribution to the model while controlling for Assists.\n")

# Interpretation of ANCOVA results
ancova_summary <- summary(ancova_result)
cat("\n--- Interpretation of ANCOVA Results ---\n")
cat("If the p-value for Assists is less than 0.05, it indicates that Assists significantly predict Goals,\n")
cat("while the p-value for the composite variable indicates its role in the relationship as a control.\n")

# --- Visualizations ---

# 1. Visualization for Multiple Regression
multiple_regression_plot <- ggplot(data, aes(x = Assists, y = Goals)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(title = "Multiple Regression: Goals vs. Assists",
       x = "Assists",
       y = "Goals") +
  theme_minimal()

# Print the multiple regression plot
print(multiple_regression_plot)

# 2. Visualization for ANCOVA
ancova_plot <- ggplot(data, aes(x = Assists, y = Goals, color = composite_performance)) +
  geom_point() +
  labs(title = "ANCOVA: Goals vs. Assists with Composite Performance",
       x = "Assists",
       y = "Goals",
       color = "Composite Performance") +
  theme_minimal() +
  scale_color_gradient(low = "blue", high = "red")

# Print the ANCOVA plot
print(ancova_plot)

# 3. Ribbon plot for Goals vs. Assists
ribbon_data <- data %>%
  group_by(Assists) %>%
  summarise(mean_goals = mean(Goals, na.rm = TRUE), 
            sd_goals = sd(Goals, na.rm = TRUE)) %>%
  mutate(lower = mean_goals - sd_goals, upper = mean_goals + sd_goals)

ribbon_plot <- ggplot(ribbon_data, aes(x = Assists, y = mean_goals)) +
  geom_line(color = "blue") +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "lightblue", alpha = 0.5) +
  labs(title = "Ribbon Plot: Goals vs. Assists",
       x = "Assists",
       y = "Mean Goals") +
  theme_minimal()

# Print the ribbon plot for Goals vs. Assists
print(ribbon_plot)

# 4. Ribbon plot for Goals vs. Composite Performance
ribbon_performance_data <- data %>%
  group_by(composite_performance) %>%
  summarise(mean_goals = mean(Goals, na.rm = TRUE), 
            sd_goals = sd(Goals, na.rm = TRUE)) %>%
  mutate(lower = mean_goals - sd_goals, upper = mean_goals + sd_goals)

ribbon_performance_plot <- ggplot(ribbon_performance_data, aes(x = composite_performance, y = mean_goals)) +
  geom_line(color = "blue") +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "lightblue", alpha = 0.5) +
  labs(title = "Ribbon Plot: Goals vs. Composite Performance",
       x = "Composite Performance",
       y = "Mean Goals") +
  theme_minimal()

# Print the ribbon plot for Goals vs. Composite Performance
print(ribbon_performance_plot)

# --- Visualization for Goals vs. Position ---
# Box plot to visualize the distribution of Goals across different Positions
goals_position_plot <- ggplot(data, aes(x = Position, y = Goals)) +
  geom_boxplot(fill = "lightblue", outlier.color = "red", outlier.shape = 16) +
  labs(title = "Goals Scored by Position",
       x = "Position",
       y = "Goals") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Print the box plot for Goals vs. Position
print(goals_position_plot)