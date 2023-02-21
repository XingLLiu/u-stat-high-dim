
import numpy as np

class Analytical:
    """Base class for analytical formulae of U-statistics."""
    def __init__(self, dim, mu_norm, bandwidth_power, bandwidth_scale: float=None):
        self.d = dim
        self.mu_norm_sq = mu_norm**2
        self.r = bandwidth_power
        # 0.5 as here 2*gamma := bandwidth_scale * r**bandwidth_power 
        bandwidth_scale = bandwidth_scale if bandwidth_scale is not None else 2.
        self.gamma = 0.5 * bandwidth_scale * self.d**self.r

    def mean(self):
        raise NotImplementedError

    def cond_var(self):
        """*Central* 2nd condition moment"""
        raise NotImplementedError
        
    def full_var(self):
        """*Central* 2nd full moment"""
        raise NotImplementedError

    def abs_cond_moment_ub(self, nu):
        """
        Upper bound for *non-central* absolute conditional moments.
        """
        raise NotImplementedError

    def abs_full_moment_ub(self, nu):
        """
        Upper bound for *non-central* absolute full moments.
        """
        raise NotImplementedError


class MMDAnalytical(Analytical):
    """Analytical expressions or upper bounds for the moments of MMD U-statistic."""
    def __init__(self, dim, mu_norm, bandwidth_power, bandwidth_scale: float=None):
        super().__init__(dim, mu_norm, bandwidth_power, bandwidth_scale)

    def mean(self):
        res = 2 * (self.gamma / (2 + self.gamma))**(self.d / 2) * (
            1 - np.exp(- 1 / (2 * (2 + self.gamma)) * self.mu_norm_sq)
        )
        return res

    def m2(self):
        factor = (self.gamma / (1 + self.gamma))**self.d
        term1 = 2 * ((1 + self.gamma) / (3 + self.gamma))**(self.d/2)
        term2 = term1 * np.exp( - self.mu_norm_sq / (3 + self.gamma) )
        term3 = 2 * ((1 + self.gamma) / (2 + self.gamma))**self.d
        term4 = - 4 * ((1 + self.gamma) / (3 + self.gamma))**(self.d/2) * np.exp(
            - 1/4 * (1/(1 + self.gamma) + 1/(3 + self.gamma)) * self.mu_norm_sq
        )
        term5 = - 4 * ((1 + self.gamma) / (2 + self.gamma))**self.d * np.exp(
            - self.mu_norm_sq / (2 * (2 + self.gamma))
        )
        term6 = 2 * ((1 + self.gamma) / (2 + self.gamma))**self.d * np.exp(
            - self.mu_norm_sq / (2 + self.gamma)
        )
        res = factor * (term1 + term2 + term3 + term4 + term5 + term6)
    
        return res
    
    def cond_var(self):
        return self.m2() - self.mean()**2
        
    def M2(self):
        term1 = 2 * (self.gamma / (4 + self.gamma))**(self.d / 2) * (
            1 + np.exp( - 1 / (4 + self.gamma) * self.mu_norm_sq )
        )
        term2 = 2 * (self.gamma / (2 + self.gamma))**self.d
        term3 = - 8 * (self.gamma / (3 + self.gamma))**(self.d / 2) * (
            self.gamma / (1 + self.gamma)
        )**(self.d / 2) * np.exp(
            - (2 + self.gamma) / (2 * (1 + self.gamma) * (3 + self.gamma)) * self.mu_norm_sq
        )
        term4 = 2 * (self.gamma / (2 + self.gamma))**self.d * np.exp(
            - 1 / (2 + self.gamma) * self.mu_norm_sq
        )
        return (term1 + term2 + term3 + term4)
    
    def full_var(self):
        return self.M2() - self.mean()**2
    
    def abs_cond_moment_ub(self, nu):
        """
        Upper bound for *non-central* absolute conditional moments.
        """
        res = 2**(2*nu - 1) * (
            nu / (1 + nu + self.gamma)
        )**(self.d / 2) * (
            1 + np.exp(- nu / (2 * (1 + nu + self.gamma)) * self.mu_norm_sq)
        )
        return res
    
    def abs_full_moment_ub(self, nu):
        """
        Upper bound for *non-central* absolute full moments.
        """
        res = 2**(2*nu - 1) * (
            self.gamma / (2*nu + self.gamma)
        )**(self.d / 2) * (
            1 + np.exp(- nu / (2 * (2 * nu + self.gamma)) * self.mu_norm_sq)
        )
        return res

class KSDAnalytical(Analytical):
    """Analytical expressions or upper bounds for the moments of KSD U-statistic."""
    def __init__(self, dim, mu_norm, bandwidth_power, bandwidth_scale: float=None) -> None:
        super().__init__(dim, mu_norm, bandwidth_power, bandwidth_scale)
        
    def mean(self):
        res = (self.gamma / (self.gamma + 2))**(self.d / 2) * self.mu_norm_sq
        return res

    def cond_var(self):
        g_1_g_3 = (1 + self.gamma) * (3 + self.gamma)
        factor = (self.gamma**2 / g_1_g_3)**(self.d/2)
        term1 = (2 + self.gamma)**2 / g_1_g_3 * self.mu_norm_sq
        term2 = (
            1 - (g_1_g_3 / (2 + self.gamma)**2)**(self.d/2)
        ) * self.mu_norm_sq**2
        res = factor * (term1 + term2)
        return res

    def full_var(self):
        factor1 = (self.gamma / (4 + self.gamma))**(self.d / 2)
        factor2 = (
            1 - (self.gamma * (4 + self.gamma) / (2 + self.gamma)**2)**(self.d / 2)
        )
        res = factor1 * (
            self.d 
            + self.d**2 / self.gamma**2 
            + 2 * self.d * self.mu_norm_sq / self.gamma
            + 2 * self.mu_norm_sq
            + (1 - factor2) * self.mu_norm_sq**2
        )
        return res

    def abs_cond_moment_ub(self, nu):
        """
        Upper bound for *non-central* absolute conditional moments.
        """
        c1 = 1.
        c2 = 1.
        res = np.exp(- self.d * nu / self.gamma) * (
            c1 * self.mu_norm_sq**(nu / 2)
            + c2 * self.mu_norm_sq**nu
        )
        return res

    def abs_full_moment_ub(self, nu):
        """
        Upper bound for *non-central* absolute conditional moments.
        """
        c3 = 1.
        c4 = 1.
        c5 = 1.
        res = np.exp( - self.d * nu / self.gamma ) *(
            c3 * self.d**(nu / 2)
            + c4 * self.mu_norm_sq**(nu / 2)
            + c5 * self.mu_norm_sq**nu
            + (self.d / self.gamma)**nu
        )
        return res


class MMDLinearAnalytical(Analytical):
    """Analytical expressions or upper bounds for the moments of Linear-MMD U-statistic."""
    def __init__(self, dim, mu, Sigma) -> None:
        self.mu = mu
        self.Sigma = Sigma
        self.mu_sigma_mu = np.matmul(
            np.matmul(self.Sigma, self.mu,),
            self.mu,
        )
        mu_norm = np.sum(mu**2)**(1/2.)
        super().__init__(dim, mu_norm, bandwidth_power=0.)
        
    def mean(self):
        return self.mu_norm_sq
    
    def cond_var(self):
        return 2 * self.mu_sigma_mu

    def full_var(self):
        tr_sigsq = np.sum(np.matmul(self.Sigma, self.Sigma))
        return 4 * tr_sigsq + 4 * self.mu_sigma_mu

    def abs_cond_moment_ub(self, nu):
        """
        Upper bound for *non-central* absolute conditional moments.
        """
        res = None
        return res

    def abs_full_moment_ub(self, nu):
        """
        Upper bound for *non-central* absolute conditional moments.
        """
        res = None
        return res
