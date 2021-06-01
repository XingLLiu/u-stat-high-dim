data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real mu;
}
model {
  real mu0=0;
    real sigma0=1;
  real sigma=1;
  mu ~ normal(mu0, sigma0);
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[N] log_lik;
  real sigma=1;
  log_lik = normal_lpdf(y[n] | mu, sigma)
}
