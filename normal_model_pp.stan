data {
  int<lower=0> N;
  vector[N] y;
  real sigma;
  real mu0;
  real sigma0;
}
parameters {
  real mu;
}
generated quantities {
  vector[N] log_lik;
  vector[N] log_pos;
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | mu, sigma);
    log_pos[n] = log_lik[n] + normal_lpdf(mu | mu0, sigma0);
  }
}
