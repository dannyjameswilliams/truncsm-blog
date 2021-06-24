import numpy as np

# Truncated Score Matching objective function
def truncsm(theta, X, Px, g, dg, psi):
    p, dp = psi(X, theta)
    t1 = np.mean(np.sum(p**2, 1)*g)
    t2 = np.mean(np.sum(dp*g))
    t3 = np.mean(np.sum(p*dg, 1))
    return t1 + 2*(t2 + t3)

# Psi for multivariate normal
def psi_mvn(X, theta):
    return -(X-theta.T), -np.repeat(X.shape[1], X.shape[0])

