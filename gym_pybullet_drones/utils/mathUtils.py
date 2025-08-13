import math
import numpy as np

class MathUtils():

    EPSILON = 0.1
    H = 0.2
    A, B = 5, 5
    C = np.abs(A-B)/np.sqrt(4*A*B)  # phi

    R = 40
    D = 40

    @staticmethod
    def sigma_1(z):
        '''
            scale a number so it stays in the range (-1,1)(-1,1) while preserving its sign and relative size for small values.
        '''

        return z / np.sqrt(1 + z**2)

    @staticmethod
    def sigma_norm(z, e=EPSILON):
        '''
            generalised “norm” function that smooths out the magnitude of a vector
        '''
        return (np.sqrt(1 + e * np.linalg.norm(z, axis=-1, keepdims=True)**2) - 1) / e

    @staticmethod
    def sigma_norm_grad(z, e=EPSILON):
        '''
            gradient (vector derivative) of the z norm
        '''
        return z/np.sqrt(1 + e * np.linalg.norm(z, axis=-1, keepdims=True)**2)

    @staticmethod
    def bump_function(z, h=H):
        '''
            piecewise “bump” function 
        '''
        ph = np.zeros_like(z)
        ph[z <= 1] = (1 + np.cos(np.pi * (z[z <= 1] - h)/(1 - h)))/2
        ph[z < h] = 1
        ph[z < 0] = 0
        return ph

    @staticmethod
    def phi(z, a=A, b=B, c=C):
        '''
            smooth mapping from an unbounded input zz to a range [b,a][b,a].
        '''
        return ((a + b) * MathUtils.sigma_1(z + c) + (a - b)) / 2

    @staticmethod
    def phi_alpha(z, r=R, d=D):
        '''
            composite “interaction weight” function.
        '''
        r_alpha = MathUtils.sigma_norm([r])
        d_alpha = MathUtils.sigma_norm([d])
        return MathUtils.bump_function(z/r_alpha) * MathUtils.phi(z-d_alpha)

    @staticmethod
    def normalise(v, pre_computed=None):
        '''
            2D vector normalisation function
        '''
        n = pre_computed if pre_computed is not None else math.sqrt(
            v[0]**2 + v[1]**2)
        if n < 1e-13:
            return np.zeros(2)
        else:
            return np.array(v) / n
        
    @staticmethod
    def InteractionForce(xi, xj, a, c, d):
        exponent = -((abs(xi-xj) - d)/ c)
        numerator = a * (1- np.exp(exponent))
        denominator = 1 + abs(xi-xj)
        return numerator/denominator * xi-xj
