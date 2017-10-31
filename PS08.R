# Special thanks to Andrew!

library(tidyverse)
library(caret)

# Package for easy timing in R
library(tictoc)

# Get data ----------------------------------------------------------------
# Accelerometer Biometric Competition Kaggle competition data
# https://www.kaggle.com/c/accelerometer-biometric-competition/data
train <- read_csv("train.csv")

# It's huge!
dim(train)



# knn modeling ------------------------------------------------------------
model_formula <- as.formula(Device ~ X + Y + Z)

# Values to use:
n_values <- c(100000, seq(from = 500000, to = 3000000, by = 500000), seq(from = 3000000, to = 24000000, by = 3000000))
k_values <- c(2, 3, 4, 5) # I can make more

# Create data frame with k going from 2 through 5 for each n before n changes
runtime_dataframe <- expand.grid(n_values, k_values) %>%
  as_tibble() %>%
  rename(n=Var1, k=Var2) %>%
  mutate(runtime = n*k)

n <- c()
for (i in 1:length(n_values)){
  x <- rep(n_values[i], length(k_values))
  n <- c(n, x)
}

runtime_dataframe$n <- n
runtime_dataframe$k <- as.factor(rep(k_values, length(n_values)))

# Time knn here -----------------------------------------------------------

for (i in 1:length(n_values)) {
  train_n <- train %>% sample_n(n_values[i])
  for (j in 1:length(k_values)) {
    tic()
    model_knn <- caret::knn3(model_formula, data=train_n, k = k_values[j])
    timer_info <- toc()  
    runtime_dataframe$runtime[((i-1)*length(k_values)) + j] <- timer_info$toc - timer_info$tic
  }
}

# For loop description:
# Different sizes of training sets are randomly sampled.
# For each training sample size, it runs knn with the 4 different values of k.
# Only after the 4 values of k are gone through does the value of n increase.
# This is mediated by the increase of the index i.
# The increase of k is mediated by the index j. 
# The runtime vector has to be filled in in the right order.
# ((i-1)*length(k_values)) + j ensures that before n increases, k has gone from 2 through 5 first.

# Plot your results ---------------------------------------------------------

runtime_plot <- ggplot(runtime_dataframe, aes(x=n, y=runtime, col=k)) +
  geom_point() + labs(title = "Runtime of k Nearest Neighbors Fitting", y = "runtime (s)")

runtime_plot
ggsave(filename="leonard_yoon.png", width=16, height = 9)




# Runtime complexity ------------------------------------------------------
# Can you write out the rough Big-O runtime algorithmic complexity as a function
# of:
# -n: number of points in training set
# -k: number of neighbors to consider
# -d: number of predictors used? In this case d is fixed at 3

# ANSWER: runtime = O(n^2) because the relationship looks parabolic between n and runtime. 
# There is no significant relationship between k and runtime, at least until large values of n.
