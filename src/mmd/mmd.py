import tensorflow as tf

class MMD:
    """MMD statistic with equal sample sizes."""
    def __init__(self, kernel: tf.Module):
        self.k = kernel

    def __call__(self, X: tf.Tensor, Y: tf.Tensor, output_dim: int=1):
        """
        Inputs:
            X: (n, d)
            Y: (m, d)
        """
        assert X.shape[-2] == Y.shape[-2], "Sample sizes must be equal"

        # median heuristic
        if self.k.med_heuristic:
            Z = tf.concat([X, Y], axis=-2)
            self.k.bandwidth(Z, tf.identity(Z))

        # TODO median heuristic needs to be computed separately as 
        # it requires both samples
        
        K_XX = self.k(X, X) # n x n
        K_YY = self.k(Y, Y) # m x m
        K_XY = self.k(X, Y) # n x m
        K_YX = self.k(Y, X) # m x n

        u_mat = K_XX + K_YY - K_XY - K_YX # n x n
        u_mat_nodiag = tf.linalg.set_diag(u_mat, tf.zeros(u_mat.shape[:-1]))

        if output_dim == 1:
            mmd = tf.reduce_sum(
                u_mat_nodiag,
                axis=[-1, -2],
            ) / (X.shape[-2] * (Y.shape[-2] - 1))
            return mmd

        elif output_dim == 2:
            return u_mat_nodiag

    def m3_test(self, X, Y):
        K_XX = self.k(X, X) # n x n
        K_YY = self.k(Y, Y) # m x m
        K_XY = self.k(X, Y) # n x m
        K_YX = self.k(Y, X) # m x n

        u_mat = K_XX + K_YY - K_XY - K_YX # n x n
        n = X.shape[0]
        res = 0.
        count = 0.
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and j != k and k != i:
                        res += u_mat[i] * u_mat[j] * u_mat[k]
                        count += 1

        return res / count
    
    def abs_cond_central_moment(self, X: tf.Tensor, Y: tf.Tensor, k: int, center: bool=True):
        n = X.shape[-2]
        u_nodiag = self.__call__(X, Y, output_dim=2) # n x n
        g = tf.math.reduce_sum(u_nodiag, axis=-1) / (n - 1) # n
        mmd = tf.math.reduce_sum(g, axis=-1) / n
        if center:
            g -= mmd

        mk = tf.math.reduce_sum(tf.math.abs(g)**k) / n
        return mk
    
    def abs_full_central_moment(self, X: tf.Tensor, Y: tf.Tensor, k: int, center: bool=True):
        n = X.shape[-2]
        u_nodiag = self.__call__(X, Y, output_dim=2) # n x n
        mmd = tf.math.reduce_sum(u_nodiag, axis=[-1, -2]) / (n * (n - 1))
        if center:
            u_nodiag -= mmd

        Mk = tf.math.reduce_sum(tf.math.abs(u_nodiag)**k) / (n * (n - 1))
        return Mk

class MMDAnalytical:
    """Analytical expressions or upper bounds for the moments of U-statistic."""
    def __init__(self, dim, mu_norm, bandwidth_power):
        self.d = dim
        self.mu_norm_sq = mu_norm**2
        self.r = bandwidth_power
        self.lmda = self.d**self.r
        
    def mmd(self):
        res = 2 * (self.lmda / (2 + self.lmda))**(self.d / 2) * (
            1 - tf.exp(- 1 / (2 * (2 + self.lmda)) * self.mu_norm_sq)
        )
        return res

    def m2_ub(self):
        res = 8 * (self.lmda / (3 + self.lmda))**(self.d / 2) * (
            self.lmda / (1 + self.lmda)
        )**(self.d / 2) * (
            1 + tf.exp(- 1 / (3 + self.lmda) * self.mu_norm_sq)
        )
        return res
    
    def cond_var(self):
        return self.m2_ub() - self.mmd()**2
        
    def M2(self):
        term1 = 2 * (self.lmda / (4 + self.lmda))**(self.d / 2) * (
            1 + tf.exp( - 1 / (4 + self.lmda) * self.mu_norm_sq )
        )
        term2 = 2 * (self.lmda / (2 + self.lmda))**self.d
        term3 = - 8 * (self.lmda / (3 + self.lmda))**(self.d / 2) * (
            self.lmda / (1 + self.lmda)
        )**(self.d / 2) * tf.exp(
            - (2 + self.lmda) / (2 * (1 + self.lmda) * (3 + self.lmda)) * self.mu_norm_sq
        )
        term4 = 2 * (self.lmda / (2 + self.lmda))**self.d * tf.exp(
            - 1 / (2 + self.lmda) * self.mu_norm_sq
        )
        return term1 + term2 + term3 + term4
    
    def full_var(self):
        return self.M2() - self.mmd()**2
    
    def abs_cond_moment_ub(self, nu):
        """
        Upper bound for *non-central* absolute conditional moments.
        """
        res = 2**(2*nu - 1) * (
            nu / (1 + nu + self.lmda)
        )**(self.d / 2) * (
            1 + tf.exp(- nu / (2 * (1 + nu + self.lmda)) * self.mu_norm_sq)
        )
        return res
    
    def abs_full_moment_ub(self, nu):
        """
        Upper bound for *non-central* absolute full moments.
        """
        res = 2**(2*nu - 1) * (
            self.lmda / (2*nu + self.lmda)
        )**(self.d / 2) * (
            1 + tf.exp(- nu / (2 * (2 * nu + self.lmda)) * self.mu_norm_sq)
        )
        return res