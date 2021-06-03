library(rstan)
# library(rstanarm)
library(loo)
library(dplyr)
rstan_options(auto_write = TRUE)

# compiled_model <- stan_model("normal_model.stan")
compiled_model <- readRDS("normal_model.rds")

set.seed(1)
N <- 100
mu <- 3
sigma <- 1
mu0 = 0
sigma0 = 1
y <- rnorm(N, mu, sigma)
data <- list(y=y, N=N, sigma=sigma, mu0=mu0, sigma0=sigma0)

fit <- sampling(compiled_model, data=data, iter=1000)
print(fit)
fit_df <- as.data.frame(fit)
mu_pos <- fit_df$mu
cat("mean:", mean(mu_pos), "sd:", sd(mu_pos), "\n")

# get loglikelihood
loglik_vals <- extract_log_lik(fit, merge_chains=TRUE)
head(rowSums(loglik_vals))

# compiled_model_pp <- stan_model("normal_model_pp.stan")
compiled_model_pp <- readRDS("normal_model_pp.rds")
y_new <- rnorm(10, mu, sigma)
data_new <- list(y=y_new, N=length(y_new), sigma=sigma, mu0=mu0, sigma0=sigma0)


posterior_den <- function(mu, y){
  true_pos_var <- (1 / sigma0**2 + length(y) / sigma**2)**(-1)
  true_pos_mean <- true_pos_var * (mu0 / sigma0**2 + sum(y) / sigma**2)
  return(dnorm(mu, mean=true_pos_mean, sd=sqrt(true_pos_var)))
}

loglik <- function(mu, sigma, y){
  term1 <- - 0.5 * log(2 * pi * sigma**2)
  term2 <- - 1 / (2 * sigma**2) * (y - mu)**2
  return(term1 + term2)
}

# plot true and empirical posteriors
hist(mu_pos, freq=FALSE, breaks=50)
abline(v=mean(mu_pos), col="green")
abline(v=mu, col="blue", lty="dashed")
x_plot <- seq(from=range(mu_pos)[1], to=range(mu_pos)[2], length.out=100)
lines(x_plot, posterior_den(x_plot, y), col="blue")


# check if posterior matches with true value
for (i in 1:5) {
  # this should be 0
  cat("loglik", sum(loglik_vals[i, ]) - sum(loglik(mu_pos[i], sigma, y)), "\n")
  # this should be constant
  cat("posterior", fit_df$lp__[i] - (sum(loglik(mu_pos[i], sigma, y)) + dnorm(mu_pos[i], mu0, sigma0, log=TRUE)), "\n")
}


# add new data and update posterior samples
n_new <- 100
y_new <- y
log_lik_old <- fit_df %>% select(matches("log_lik")) %>% rowSums
log_posterior_old <- fit_df %>% .$`lp__`
mu_hat <- rep(0, n_new + 1)
mu_hat[1] <- mean(mu_pos)
mu_hat_posterior <- mu_hat
for (i in 1:1){
  y_new = c(y_new, rnorm(mean=mu, sd=sigma, n=n_new))
  data_new <- list(y=y_new, N=length(y_new), sigma=sigma, mu0=mu0, sigma0=sigma0)
  gqs_res <- gqs(
    compiled_model_pp, 
    draws=as.matrix(fit), 
    data=data_new, 
    seed=1
  )
  df <- as.data.frame(gqs_res)

  # compute weights
  log_lik_new <- df %>% select(matches("log_lik")) %>% rowSums()
  weights <- exp(log_lik_new - log_lik_old)
  # update likelihood
  log_lik_old <- log_lik_new

  # compute posterior 
  log_posterior_new <- df %>% select(matches("log_pos")) %>% rowSums()
  weights_posterior <- exp(log_posterior_new - log_posterior_old)
  # update posterior
  log_posterior_old <- log_posterior_new

  # compute SW-IS estimate
  mu_hat[i + 1] <- sum(mu_pos * weights / sum(weights))
  mu_hat_posterior[i + 1] <- sum(mu_pos * weights_posterior / sum(weights_posterior))

  if ((i %% 10) == 0){
    cat("MSE:", (mu_hat[i] - mu)^2, "posterior MSE:", (mu_hat_posterior[i] - mu)^2, "y len:", length(y_new), "\n")
    cat("lik weights:", (weights / sum(weights))[1:5], "posterior weights", (weights_posterior / sum(weights_posterior))[1:5], "\n")
  }

}


# sample from new posterior after adding data
data_new <- list(y=y_new, N=length(y_new), sigma=sigma, mu0=mu0, sigma0=sigma0)
fit_new <- sampling(compiled_model, data=data_new, iter=1000)
fit_df_new <- as.data.frame(fit_new)
mu_pos_new <- fit_df_new$mu

# MSE of new estimates
cat("HMC: ", (mean(mu_pos_new) - mu)^2, "SN-IS:", (mu_hat[i] - mu)^2, "\n")


# plot true and empirical posteriors
# hist(mu_pos, freq=FALSE, breaks=50)
# abline(v=mean(mu_pos), col="green")
# abline(v=mu, col="blue", lty="dashed")
# x_plot <- seq(from=range(mu_pos)[1], to=range(mu_pos)[2], length.out=100)
# lines(x_plot, posterior_den(x_plot, y_new), col="blue")
hist(mu_pos_new, freq=FALSE, breaks=50)
lines(density(mu_pos_new), col="red")
lines(density(mu_pos), col="green")

fit <- sampling(compiled_model, data=data, iter=1)
