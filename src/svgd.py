import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tqdm import trange


class SVGD:
    def __init__(
        self,
        target: tfp.distributions.Distribution,
        kernel: tf.Module,
        sigmas: tf.Tensor,
        optimizer: tf.keras.optimizers,
        num_particles: int = 50,
        num_samples: int = 10,
    ):
        """
        Inputs:
            target (tf.distributions.Distribution): Only require the log_probability of the target distribution e.g. unnormalised posterior distribution
            kernel (tf.nn.Module): [description]
            optimizer (tf.optim.Optimizer): [description]
        """
        self.target = target
        self.k = kernel
        self.sigmas = sigmas
        self.num_particles = num_particles
        self.num_samples = num_samples
        self.optimizer = optimizer

    def phi_sigma(self, X: tf.Tensor, sigma: tf.Tensor):
        X_cp = X[:, None, :]
        z = tf.random.normal(shape=(self.num_samples, 1))  # num_samples x x_dim
        perturbed_samples = sigma * z  # num_samples x x_dim
        with tf.GradientTape() as g:
            g.watch(X_cp)
            diff_1 = X_cp - perturbed_samples  # num_particles x num_samples x x_dim
            prob_1 = self.target.prob(
                diff_1
            )  # prob() computes the unnormalised density
        grad_1 = g.gradient(prob_1, X_cp)
        grad_1 = tf.squeeze(grad_1, axis=2)
        score_X = (
            grad_1 / tf.reduce_sum(prob_1, axis=1)[:, None]
        )  # num_particles x x_dim

        # compute perturbed Q samples
        z = self.sigmas[0] * tf.random.normal(
            shape=(self.num_particles, 1)
        )  # num_particles x x_dim
        X_perturbed = X + z  # num_particles x x_dim
        self.k.bandwidth(X_perturbed, X_perturbed)
        Kxx = self.k(X_perturbed, X_perturbed)  # num_particles x num_particles

        # compute the optimal test function
        attraction = Kxx @ score_X / self.num_particles
        repulsion = tf.reduce_mean(self.k.grad_first(X_perturbed, X), axis=0)
        phi = attraction + repulsion
        return phi * sigma**2

    def phi(self, X: tf.Tensor):
        phi = tf.map_fn(lambda sigma: self.phi_sigma(X, sigma), self.sigmas)
        phi = tf.reduce_sum(phi, axis=0)
        return phi

    def step(self, X: tf.Tensor, **kwargs):
        """Gradient descent step

        Args:
            X (torch.Tensor): Particles to transport to the target distribution
        """
        phi = self.phi(X)
        self.optimizer.apply_gradients(zip([phi], [X]))

    def fit(self, X: tf.Tensor, epochs=100):
        iterator = trange(epochs)
        for i in iterator:
            self.step(X)
            iterator.set_description(f"Epoch {i}")

