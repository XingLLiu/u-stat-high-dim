library(rstan)
library(rstanarm)
library(loo)
rstan_options(auto_write = TRUE)

set.seed(1)
N <- 100
mu <- 3
sigma <- 1
mu0 = 0
sigma0 = 1
y <- rnorm(N, mu, sigma)
data <- list(y=y, N=N)


fit <- stan("normal_model.stan", data=data, iter=100)
print(fit)
loglik_vals <- extract_log_lik(fit, merge_chains=FALSE)


log_lik(fit, newdata=NULL)
