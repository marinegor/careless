import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.internal import tensor_util
import numpy as np


class Amoroso(tfd.TransformedDistribution):
    def __init__(self,
		   a,
		   theta,
		   alpha,
                   beta,
		   validate_args=False,
		   allow_nan_stats=True,
		   name='Amoroso'):

        parameters = dict(locals())
        with tf.name_scope(name) as name:
            self._a = tensor_util.convert_nonref_to_tensor(a)
            self._theta = tensor_util.convert_nonref_to_tensor(theta)
            self._alpha = tensor_util.convert_nonref_to_tensor(alpha)
            self._beta = tensor_util.convert_nonref_to_tensor(beta)
            gamma = tfd.Gamma(alpha, 1.)

            chain = tfb.Invert(tfb.Chain([
                tfb.Exp(),
                tfb.Scale(beta),
                tfb.Shift(-tf.math.log(theta)),
                tfb.Log(),
                tfb.Shift(-a),
            ]))

            super().__init__(
                    distribution=gamma, 
                    bijector=chain, 
                    validate_args=validate_args,
                    parameters=parameters,
                    name=name)

    @property
    def a(self):
        return self._a

    @property
    def theta(self):
        return self._theta

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    def mean(self):
        """
        The mean of of the Amoroso distribution exists for `alpha + 1/beta >= 0`.
        It can be computed analytically by

        ```
        mean = a + theta * gamma(alpha + 1/beta) / gamma(alpha)
        ```
        """
        a,theta,alpha,beta = self.a,self.theta,self.alpha,self.beta
        return a + tf.math.exp(tf.math.log(theta)+tf.math.lgamma(alpha + tf.math.reciprocal(beta)) - tf.math.lgamma(alpha))

    def variance(self):
        """
        The variance of of the Amoroso distribution exists for `alpha + 2/beta >= 0`.
        It can be computed analytically by

        ```
        variance = theta**2 * (gamma(alpha + 2/beta) / gamma(alpha) - gamma(alpha + 1/beta)**2 / gamma(alpha)**2 )
        ```
        """
        theta,alpha,beta = self.theta,self.alpha,self.beta
        return theta**2. * (tf.math.exp(tf.math.lgamma(alpha + 2./beta) - tf.math.lgamma(alpha)) - tf.math.exp(2.*tf.math.lgamma(alpha + 1/beta) - 2.*tf.math.lgamma(alpha)))

    def stddev(self):
        """
        The variance of of the Amoroso distribution exists for `alpha + 2/beta >= 0`.
        It can be computed analytically by

        ```
        variance = theta**2 * (gamma(alpha + 2/beta) / gamma(alpha) - gamma(alpha + 1/beta)**2 / gamma(alpha)**2 )
        ```

        The standard deviation is computed by
        ```amoroso.stddev() = tf.sqrt(amoroso.variance())```
        """
        return tf.math.sqrt(self.variance())


class Stacy(Amoroso):
    def __init__(self,
		 theta,
		 alpha,
                 beta,
		 validate_args=False,
		 allow_nan_stats=True,
		 name='Stacy'):

        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super().__init__(
                0.,
                theta,
                alpha,
                beta,
                validate_args,
                allow_nan_stats,
                name,
            )

    def _stacy_params(self, other):
        if isinstance(other, Stacy):
            params = (other.theta, other.alpha, other.beta)
        elif isinstance(other, tfd.Weibull):
            # Weibul(x; k, lambda) = Stacy(x; lambda, 1, k)
            k = other.concentration
            lam = other.scale
            params = (lam, 1., k)
        elif isinstance(other, tfd.HalfNormal):
            #HalfNormal(x; scale) = Stacy(x; sqrt(2)*scale, 0.5, 2)
            scale = other.scale
            params = (np.sqrt(2.) * scale, 0.5, 2.)
        else:
            raise TypeError(f"Equivalent Stacy parameters cannot be determined for distribution, {other}. " 
                             "Only tfd.Weibull, tfd.HalfNormal, or Stacy can be converted to Stacy parameterisation")
        return params

    def _bauckhage_params(self, other):
        theta, alpha, beta = self._stacy_params(other)
        bauckhage_params = (theta, alpha*beta, beta)
        return bauckhage_params 

    def kl_divergence(self, other):
        """
        The Stacy distribution has an analytical KL div. 
        However, it isn't documented in the same parameterization as the Crooks Amoroso tome. 
        To avoid confusion, I will first translate the Stacy distributions from 
        Crooks's parameterization to the one in the KL div paper. 

        ```
        Stacy(x; a,d,p) = x**(d-1) * exp(-(x/a)**p) / Z
        ```
        where
        ```
        a = theta
        d = alpha * beta
        p = beta
        ```
        Then the KL div is
        ```
        log(p1*a2**d2*gamma(d2/p2)) - log(p2*a1**d1*gamma(d1/p1)) + 
        (digamma(d1/p1)/p1 + log(a1)) * (d1 - d2) + 
        gamma((d1 + p2)/p1) * (a1/a2)**p2 / gamma(d1/p1) - d1/p1
        ```

        See Bauckhage 2014 for derivation. 
        https://arxiv.org/pdf/1401.6853.pdf

        Parameters
        ----------
        other : Stacy or tfd.Weibull or tfd.HalfNormal
        """
        a1,d1,p1 = self._bauckhage_params(self)
        a2,d2,p2 = self._bauckhage_params(other)

        #The following numerics are easier to read if you alias this
        ln = tf.math.log

        kl = ln(p1) + d2*ln(a2) + tf.math.lgamma(d2/p2) - ln(p2) - d1*ln(a1) - tf.math.lgamma(d1/p1) + \
             (tf.math.digamma(d1/p1)/p1 + ln(a1))*(d1 - d2) +  \
             tf.math.exp(tf.math.lgamma((d1 + p2)/p1) - tf.math.lgamma(d1/p1) + p2*(ln(a1) - ln(a2))) \
             - d1/p1

        return kl
