import numpy as np
import numpy.random as rand
import scipy.optimize as opt

# Simulate data and truncated data within a Lq ball
def simulate_data(n, d, q, mu = None, seed = None):
    
    # Set seed
    if seed is not None:
        rand.seed(seed)
    
    # Parameters
    if mu is None:
        mu = np.repeat(0, d)
        
    Sigma = np.identity(d)
    
    # Simulate until we have n truncated samples
    ntrunc = 0
    Xt = Xall = np.empty((0, d))
    while ntrunc < n:
        
        # Simulate from MVN
        Xin  = rand.multivariate_normal(mu, Sigma, n)
        Xall = np.vstack((Xall, Xin))
        
        # Truncate according to Lq norm
        Xint = Xin[np.linalg.norm(Xin, q, 1) < 1, :]
        
        # Append to Xt (truncated X)
        Xt = np.vstack((Xt, Xint))
        ntrunc = np.shape(Xt)[0]
        
    return Xt[0:n, :], Xall

# Get projections on from a Lq ball
def polydist_Lqball(X, q_bound, q_norm):
    
    # Variables
    n, d = np.shape(X)
    
    # predefined arrays
    Px = np.empty((n, d))
    g  = np.empty(n)
    
    # Define constraint function (Lq ball = 1)
    def constr(x):
        return np.linalg.norm(x, q_bound) - 1 
    constr = opt.NonlinearConstraint(constr, 0, 0)
    
    for i in range(n):
        
        # Objective function is to minimise the Lq norm from data to boundary
        def obj(x):
            return np.linalg.norm(X[i, :] - x, q_norm)
        
        # Loop over 5 different initial conditions
        res = np.empty(5)
        xx  = np.empty((5, d))
        for j in range(5):
            ini = rand.normal(0, 1, d)
            op  = opt.minimize(obj,  ini, constraints = [constr])
            
            res[j]   = op.fun
            xx[j, :] = op.x
        
        # Choose the point which best satisfied the objective function
        Px[i, :] = xx[np.argmin(res)]
        g[i]     = res[np.argmin(res)]
    
    return Px, g    
