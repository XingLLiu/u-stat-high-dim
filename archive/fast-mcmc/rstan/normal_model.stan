data {
  int<lower=0> N;
  vector[N] y;
  real mu0;
  real sigma0;
  real sigma;
}
parameters {
  real mu;
}
model {
  mu ~ normal(mu0, sigma0);
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) 
    log_lik[n] = normal_lpdf(y[n] | mu, sigma);
}
