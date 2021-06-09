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


def polydist_Lqball(X, q_bound, q_norm):
    
    # Variables
    n, d = np.shape(X)
    
    # predefined arrays
    Px = np.empty((n, d))
    g  = np.empty(n)
    
    def constr(x):
        return np.linalg.norm(x, q_bound) - 1 
    constr = opt.NonlinearConstraint(constr, 0, 0)
    
    for i in range(n):
        
        
        def obj(x):
            return np.linalg.norm(X[i, :] - x, q_norm)
        
        res = np.empty(5)
        xx  = np.empty((5, d))
        for j in range(5):
            ini = rand.normal(0, 1, d)
            op  = opt.minimize(obj,  ini, constraints = [constr])
            
            res[j]   = op.fun
            xx[j, :] = op.x
        
        
        Px[i, :] = xx[np.argmin(res)]
        g[i]     = res[np.argmin(res)]
    
    return Px, g    

def get_Px(q = 2, h = 0.005):
    
    # Generate points for the boundary (L2)
    xypoints = np.array([[x, y] for x in np.arange(-1, 1, h) for y in np.arange(-1, 1, h)])
    lqnorm = np.linalg.norm(xypoints, q, 1)
        
    Px = xypoints[abs(lqnorm - 1) < 1e-3, :]
    # Px = np.array([lqbound]).T # Set boundary points (NOT projections) as Px
    
    return Px