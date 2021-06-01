import pystan

import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import numpy as np



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


# put our data in a dictionary
data = {'N': N, 'y': y}

# compile the model
sm = pystan.StanModel(model_code=model)

# train the model and generate samples
fit = sm.sampling(data=data, iter=1000, chains=4, warmup=500, thin=1, seed=101)

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

# def sample_true_pos()

def loglik(mu, sigma, y):
  # N = len(y)
  term1 = - 0.5 * np.log(2 * np.pi * sigma**2)
  term2 = - 1/(2 * sigma**2) * (y - mu)**2
  return term1 + term2

log_posterior(mu_pos, sigma, mu0=0, sigma0=1, y=y)
print(loglik_val[0], loglik(mu_pos[0], sigma, y).sum())


# plot
plt.hist(mu_pos, 30, density=True)
plt.axvline(np.mean(mu_pos), color='r', lw=2, linestyle='--',label='mean')
plt.axvline(mu, color='k', lw=2, linestyle='-',label='true mean')
plt.legend()
plt.show()

n_new = 1000
y_new = np.zeros(n_new)
mu_hat = np.zeros(n_new)
mu_hat[0] = mu_pos.mean()
for i in range(1, n_new):
  y_new_val = np.random.normal(loc=mu, scale=sigma, size=1)
  weights = np.exp(loglik(mu_pos, sigma, y_new_val))
  # print((mu_pos * weights / weights.sum()).sum())
  mu_hat[i] = (mu_pos * weights / weights.sum()).sum()
  y_new[i] = y_new_val


# y_full = 
# true_pos_sd = (1 / sigma0**2 + N / sigma**2)**(-1)
# true_pos_mean = sd * (mu0 / sigma0**2 + np.sum(y_full) / sigma**2)
# true_pos_samples = np.random.normal(loc=true_pos_mean, scale=true_pos_sd, size=10000)


plt.scatter(range(n_new), mu_hat)
plt.axhline(mu, color='k', lw=2, linestyle='-',label='true mean')
plt.show()

# plt.hist(true_pos_samples, 30, density=True)
# plt.show()
