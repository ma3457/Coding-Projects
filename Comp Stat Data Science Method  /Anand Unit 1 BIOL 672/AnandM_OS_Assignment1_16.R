#---Prompt 16: Using GMM and BIC values to determine best fit---
# Load necessary libraries
library('MASS')        # For distribution fitting
library('mixtools')    # For Gaussian Mixture Models
library('ggplot2')     # For plotting
library('grid')        # For arranging plots

# Load data
data <- read.csv('EPL_20_21.csv', header = TRUE)
passes_completed <- data$Perc_Passes_Completed / 100  # Convert percentage to a fraction [0,1]

# Ensure all values are positive for log-normal fitting
passes_completed[passes_completed <= 0] <- min(passes_completed[passes_completed > 0]) / 10

# Fit models
# Normal distribution
fitNORM <- fitdistr(passes_completed, densfun="normal")
print(fitNORM)

# Lognormal distribution
fitLNORM <- fitdistr(passes_completed, densfun="log-normal")
print(fitLNORM)

# Exponential distribution
fitEXP <- fitdistr(passes_completed, densfun="exponential")
print(fitEXP)

# Gaussian Mixture Model, assuming 2 components
fitGMM <- normalmixEM(passes_completed, k=2)
print(fitGMM)

# Manually calculate BIC for each model
n <- length(passes_completed)

# Normal: 2 parameters (mean, sd)
logLik_norm <- sum(dnorm(passes_completed, mean=fitNORM$estimate["mean"], sd=fitNORM$estimate["sd"], log=TRUE))
BIC_NORM <- -2 * logLik_norm + 2 * log(n)

# Log-Normal: 2 parameters (meanlog, sdlog)
logLik_lognorm <- sum(dlnorm(passes_completed, meanlog=fitLNORM$estimate["meanlog"], sdlog=fitLNORM$estimate["sdlog"], log=TRUE))
BIC_LNORM <- -2 * logLik_lognorm + 2 * log(n)

# Exponential: 1 parameter (rate)
logLik_exp <- sum(dexp(passes_completed, rate=fitEXP$estimate["rate"], log=TRUE))
BIC_EXP <- -2 * logLik_exp + 1 * log(n)

# GMM: 4 parameters (2 means, 2 sds, and mixing proportions)
fitGMM_loglik <- fitGMM$loglik
BIC_GMM <- -2 * fitGMM_loglik + 4 * log(n)  # Adjust based on the number of components and parameters
print("BIC for GMM")
print(BIC_GMM)

# Save BIC results and interpretation to a file
sink("model_results.txt")
print(fitNORM)
print(fitLNORM)
print(fitEXP)
print(fitGMM)
print("BIC Values:")
print(paste("Normal Distribution BIC:", BIC_NORM))
print(paste("Log-Normal Distribution BIC:", BIC_LNORM))
print(paste("Exponential Distribution BIC:", BIC_EXP))
print(paste("Gaussian Mixture Model BIC:", BIC_GMM))

# Determine and print the best model based on BIC
bic_values <- c(Normal = BIC_NORM, Lognormal = BIC_LNORM, Exponential = BIC_EXP, GMM = BIC_GMM)
best_model <- names(which.min(bic_values))
cat("\nThe best model based on BIC is:", best_model, "\n")

# Interpretation
cat("Interpretation:\n")
if (best_model == "GMM") {
  cat("The Gaussian Mixture Model (GMM) has the lowest BIC, suggesting latent structures or clusters in the data, indicating different player groups.\n")
} else {
  cat("The", best_model, "model has the lowest BIC, suggesting that this distribution best describes the overall pattern of passes completed.\n")
}
sink()

# Plotting each model's fit on histogram of passes completed
myplot1 <- ggplot(data, aes(x=passes_completed)) + 
  geom_histogram(aes(y=..density..), bins=30, fill="gray", color="black") +
  stat_function(fun=dnorm, args=list(mean=fitNORM$estimate["mean"], sd=fitNORM$estimate["sd"]), color="red") +
  ggtitle("Normal Distribution Fit")

myplot2 <- ggplot(data, aes(x=passes_completed)) + 
  geom_histogram(aes(y=..density..), bins=30, fill="gray", color="black") +
  stat_function(fun=dlnorm, args=list(meanlog=fitLNORM$estimate["meanlog"], sdlog=fitLNORM$estimate["sdlog"]), color="blue") +
  ggtitle("Log-Normal Distribution Fit")

myplot3 <- ggplot(data, aes(x=passes_completed)) + 
  geom_histogram(aes(y=..density..), bins=30, fill="gray", color="black") +
  stat_function(fun=dexp, args=list(rate=fitEXP$estimate["rate"]), color="green") +
  ggtitle("Exponential Distribution Fit")

myplot4 <- ggplot(data, aes(x=passes_completed)) + 
  geom_histogram(aes(y=..density..), bins=30, fill="gray", color="black") +
  geom_line(data = data.frame(x = fitGMM$x, y = fitGMM$posterior[,1] * dnorm(fitGMM$x, mean = fitGMM$mu[1], sd = fitGMM$sigma[1]) +
                                fitGMM$posterior[,2] * dnorm(fitGMM$x, mean = fitGMM$mu[2], sd = fitGMM$sigma[2])),
            aes(x = x, y = y), color="purple") +
  ggtitle("Gaussian Mixture Model Fit")

# Set up the viewport for plotting in a grid layout
pushViewport(viewport(layout = grid.layout(2, 2)))
print(myplot1, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
print(myplot2, vp = viewport(layout.pos.row = 1, layout.pos.col = 2))
print(myplot3, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
print(myplot4, vp = viewport(layout.pos.row = 2, layout.pos.col = 2))

# End script
print("Goodbye Maya...leaving R")