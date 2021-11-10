import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def l2norm(X, Y):
    """Compute \|X - Y\|_2^2 of tensors X, Y
    """
    XY = tf.linalg.matmul(X, Y, transpose_b=True) # n x m
    XX = tf.linalg.matmul(X, X, transpose_b=True)
    XX = tf.expand_dims(tf.linalg.diag_part(XX), axis=1) # n x 1
    YY = tf.linalg.matmul(Y, Y, transpose_b=True)
    YY = tf.expand_dims(tf.linalg.diag_part(YY), axis=0) # m x 1

    dnorm2 = -2 * XY + XX + YY
    return dnorm2


def median_heuristic(dnorm2):
    """Compute median heuristic.
    Inputs:
        dnorm2: (n x n) tensor of \|X - Y\|_2^2
    Return:
        med(\|X_i - Y_j\|_2^2, 1 \leq i < j \leq n)
    """
    ind_array = tf.experimental.numpy.triu(tf.ones_like(dnorm2), k=1) == 1
    med_heuristic = tfp.stats.percentile(dnorm2[ind_array], 50.0, interpolation="midpoint")
    return med_heuristic


def bandwidth(X, Y):
        """Compute magic bandwidth
        """
        dnorm2 = l2norm(X, Y)
        med_heuristic_sq = median_heuristic(dnorm2)
        sigma2 = med_heuristic_sq / np.log(X.shape[0])
        return tf.math.sqrt(sigma2)

class RBF(tf.Module):
    """For GSVGD to work, a kernel class need to have the following methods:
        forward: kernel evaluation k(x, y)
        grad_first: grad_x k(x, y)
        grad_second: grad_y k(x, y)
        gradgrad: grad_x grad_y k(x, y)
    """

    def __init__(self, sigma_sq=None, med_heuristic=False):
        super().__init__()
        self.sigma_sq = sigma_sq
        self.med_heuristic = med_heuristic

    def bandwidth(self, X, Y):
        """Compute magic bandwidth
        """
        dnorm2 = l2norm(X, Y)
        med_heuristic_sq = median_heuristic(dnorm2)
        sigma2 = med_heuristic_sq # med_heuristic_sq / np.log(X.shape[0])
        self.sigma_sq = sigma2
    
    def __call__(self, X, Y):
        """
        Args:
            Xr: tf.Tensor of shape (n, dim)
            Yr: tf.Tensor of shape (m, dim)
        Output:
            tf.Tensor of shape (n, m)
        """
        dnorm2 = l2norm(X, Y)
        sigma2_inv = 1.0 / (self.sigma_sq + 1e-9)
        sigma2_inv = tf.expand_dims(tf.expand_dims(sigma2_inv, 0), 0)
        K_XY = tf.math.exp(- sigma2_inv * dnorm2)

        return K_XY

    def grad_first(self, X, Y):
        """Compute grad_K in wrt first argument in matrix form 
        """
        return -self.grad_second(X, Y)

    def grad_second(self, X, Y):
        """Compute grad_K in wrt second argument in matrix form.
        Args:
            Xr: tf.Tensor of shape (n, dim)
            Yr: tf.Tensor of shape (m, dim)
        Output:
            tf.Tensor of shape (n, m, dim)
        """
        sigma2_inv = 1 / (1e-9 + self.sigma_sq)
        K = tf.expand_dims(tf.math.exp(-l2norm(X, Y) * sigma2_inv), 0) # 1 x n x m
        # diff_{ijk} = y^i_j - x^i_k
        diff = tf.transpose(tf.expand_dims(Y, 1) - X, (2, 1, 0)) # n x m x dim
        # compute grad_K
        grad_K_XY = - 2 * sigma2_inv * diff * K # n x m x dim
        grad_K_XY = tf.transpose(grad_K_XY, (1, 2, 0))

        return grad_K_XY

    def gradgrad(self, X, Y):
        """
        Args:
            X: tf.Tensor of shape (n, dim)
            Y: tf.Tensor of shape (m, dim)
        Output:
            tf.Tensor of shape (n, m, dim, dim)
        """
        # Gram matrix
        sigma2_inv = 1 / (1e-9 + self.sigma_sq)
        K = tf.expand_dims(tf.math.exp(-l2norm(X, Y) * sigma2_inv), 0)
        # diff_{ijk} = y^i_j - x^i_k
        diff = tf.transpose(tf.expand_dims(Y, 1) - X, (2, 1, 0))
        # product of differences
        diff_outerprod = -tf.expand_dims(diff, 0) * tf.expand_dims(diff, 1)
        # compute gradgrad_K
        diag = 2 * sigma2_inv * tf.expand_dims(tf.expand_dims(tf.eye(X.shape[1]), -1), -1)
        gradgrad_K_all = (diag + 4 * sigma2_inv ** 2 * diff_outerprod) * K
        gradgrad_K_all = tf.transpose(gradgrad_K_all, (2, 3, 0, 1))
        return gradgrad_K_all


class IMQ(tf.Module):
    """For GSVGD to work, a kernel class need to have the following methods:
        forward: kernel evaluation k(x, y)
        grad_first: grad_x k(x, y)
        grad_second: grad_y k(x, y)
        gradgrad: grad_x grad_y k(x, y)
    """

    def __init__(self, sigma_sq=None, beta=-0.5, med_heuristic=False):
        super().__init__()
        self.sigma_sq = sigma_sq
        self.beta = beta
        self.med_heuristic = med_heuristic

    def bandwidth(self, X, Y):
        """Compute magic bandwidth
        """
        dnorm2 = l2norm(X, Y)
        med_heuristic_sq = median_heuristic(dnorm2)
        self.sigma_sq = med_heuristic_sq
        
    def __call__(self, X, Y):
        """
        Args:
            Xr: tf.Tensor of shape (n, dim)
            Yr: tf.Tensor of shape (m, dim)
        Output:
            tf.Tensor of shape (n, m)
        """
        dnorm2 = l2norm(X, Y)
        sigma2_inv = 1.0 / (self.sigma_sq + 1e-9)
        sigma2_inv = tf.expand_dims(tf.expand_dims(sigma2_inv, 0), 0)
        K_XY = tf.pow(1 + sigma2_inv * dnorm2, self.beta)

        return K_XY

    def grad_first(self, X, Y):
        """Compute grad_K in wrt first argument in matrix form 
        """
        return -self.grad_second(X, Y)

    def grad_second(self, X, Y):
        """Compute grad_K in wrt second argument in matrix form.
        Args:
            Xr: tf.Tensor of shape (n, dim)
            Yr: tf.Tensor of shape (m, dim)
        Output:
            tf.Tensor of shape (n, m, dim)
        """
        sigma2_inv = 1 / (1e-9 + self.sigma_sq)
        K = 1 + tf.expand_dims(l2norm(X, Y) * sigma2_inv, -1) # n x m x 1
        # diff_{ijk} = y^k_i - x^k_j
        diff = tf.expand_dims(Y, 0) - tf.expand_dims(X, 1) # n x m x dim
        # compute grad_K
        grad_K_XY = 2 * sigma2_inv * diff * self.beta * tf.pow(K, self.beta-1) # n x m x dim

        return grad_K_XY   

    def gradgrad(self, X, Y):
        """
        Args:
            X: tf.Tensor of shape (n, dim)
            Y: tf.Tensor of shape (m, dim)
        Output:
            tf.Tensor of shape (n, m, dim, dim)
        """
        sigma2_inv = 1 / (1e-9 + self.sigma_sq)
        K = tf.expand_dims(1 + tf.expand_dims(l2norm(X, Y) * sigma2_inv, -1), -1) # n x m x 1 x 1
        # diff_{ijk} = y^k_i - x^k_j
        diff = tf.expand_dims(Y, 0) - tf.expand_dims(X, 1) # n x m x dim
        # product of differences
        diff_outerprod = -tf.expand_dims(diff, 2) * tf.expand_dims(diff, 3) # n x m x dim x dim
        # compute gradgrad_K
        diag = - 2 * sigma2_inv * self.beta * tf.expand_dims(
            tf.expand_dims(tf.eye(X.shape[1]), 0), 0) # 1 x 1 x dim x dim
        gradgrad_K_all = (
            diag * K + self.beta * (self.beta-1) * 4 * sigma2_inv**2 * diff_outerprod
        ) * tf.pow(K, self.beta-2) # n x m x dim x dim

        return gradgrad_K_all