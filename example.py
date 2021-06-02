import pystan
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import norm
import multiprocessing
multiprocessing.set_start_method("fork")



np.random.seed(1)

N = 10
sigma = 1
mu = 3
mu0 = 0
sigma0 = 10
y = np.random.normal(loc=mu, scale=sigma, size=N)


#! need to change sigma in two places!
model = """
data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real mu;
}
model {
  real mu0=0;
  real sigma0=10;
  real sigma=1;
  mu ~ normal(mu0, sigma0);
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[N] log_lik;
  real sigma=1;
  for (n in 1:N) 
    log_lik[n] = normal_lpdf(y[n] | mu, sigma);
}
"""

# compile the model
sm = pystan.StanModel(model_code=model)

# put our data in a dictionary
data = {'N': N, 'y': y}

# train the model and generate samples
fit = sm.sampling(n_jobs=1, data=data, iter=1000, chains=4, warmup=500, thin=1, seed=1)

# gather into dataframe
summary_dict = fit.summary()
df = pd.DataFrame(summary_dict['summary'], 
                  columns=summary_dict['summary_colnames'], 
                  index=summary_dict['summary_rownames'])
lp = fit["lp__"]
mu_pos = fit["mu"]

loglik_val = fit["log_lik"].sum(axis=1)


def log_posterior(mu, sigma, mu0, sigma0, y):
  N = len(y)
  sd = (1 / sigma0**2 + N / sigma**2)**(-1)
  mean = sd * (mu0 / sigma0**2 + np.sum(y) / sigma**2)
  log_pos = - 1 / (2*sd**2) * (mu - mean)**2
  return log_pos


def loglik(mu, sigma, y):
  # N = len(y)
  term1 = - 0.5 * np.log(2 * np.pi * sigma**2)
  term2 = - 1/(2 * sigma**2) * (y - mu)**2
  return term1 + term2

def posterior_den(mu, y):
  true_pos_sd = (1 / sigma0**2 + len(y) / sigma**2)**(-1)
  true_pos_mean = true_pos_sd * (mu0 / sigma0**2 + np.sum(y) / sigma**2)
  return norm.pdf(mu, loc=true_pos_mean, scale=true_pos_sd)


# log_posterior(mu_pos, sigma, mu0=0, sigma0=1, y=y)
print(loglik_val[0], loglik(mu_pos[0], sigma, y).sum())


# plot posterior samples and mean
x_plot = np.linspace(2, 4, 500)
plt.hist(mu_pos, 30, density=True)
plt.plot(x_plot, posterior_den(x_plot, y), color="grey", label="true density")
plt.axvline(np.mean(mu_pos), color='r', lw=2, linestyle='--',label='mean')
plt.axvline(mu, color='k', lw=2, linestyle='-',label='true mean')
plt.legend()
plt.show()

# add new data and update posterior samples
n_new = 100
y_new = np.zeros(n_new)
mu_hat = np.zeros(n_new)
mu_hat[0] = mu_pos.mean()
for i in range(1, n_new):
  y_new_val = np.random.normal(loc=mu, scale=sigma, size=1)
  weights = np.exp(loglik(mu_pos, sigma, y_new_val))
  # print((mu_pos * weights / weights.sum()).sum())
  mu_pos_new = mu_pos * weights / weights.sum()
  mu_hat[i] = mu_pos_new.sum()
  y_new[i] = y_new_val

  if (i+1) % 10 == 0:
    print("MSE:", np.mean((mu_hat[i] - mu)**2))


y_full = np.vstack((y.reshape(-1, 1), y_new.reshape(-1, 1)))
true_pos_sd = (1 / sigma0**2 + len(y_full) / sigma**2)**(-1)
true_pos_mean = true_pos_sd * (mu0 / sigma0**2 + np.sum(y_full) / sigma**2)
x_plot = np.linspace(0, 6, 500)
# true_pos_den = norm.pdf(x_plot, loc=true_pos_mean, scale=true_pos_sd)
true_pos_sample = np.random.normal(loc=true_pos_mean, scale=true_pos_sd, size=1000)





plt.scatter(range(n_new), mu_hat)
plt.axhline(mu, color="k", lw=2, linestyle="-",label="true mean")
plt.legend()
plt.show()

plt.hist(mu_hat, 30, density=True, label="sampled posterior", alpha=0.5)
plt.hist(true_pos_sample, 30, density=True, color="grey", label="true density", alpha=0.5)
plt.show()

# plt.hist(mu_hat, 30, density=True)
# plt.scatter(x_plot, true_pos_den, color="grey", label="true density")
# plt.show()

