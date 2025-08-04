import numpy as np

################################################################################

#attractive/repulsive force equation
def InteractionForce(xi, xj, a, c, d):
    exponent = -((abs(xi-xj) - d)/ c)
    numerator = a * (1- np.exp(exponent))
    denominator = 1 + abs(xi-xj)
    force = numerator/denominator * xi-xj
    return force

################################################################################

