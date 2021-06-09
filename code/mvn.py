
import os, sys
os.chdir('/home/danny/OneDrive/Personal Stuff/nodata')
sys.path.append('/home/danny/OneDrive/Personal Stuff/nodata/code')

# other files
import data_funs as df

# Dataframes/matrices etc
import pandas as pd
import numpy as np
import numpy.random as rand

# Optimisation 
from scipy.optimize import minimize

# Regular plots
import matplotlib.pyplot as plt
import seaborn as sns

# Plotly interactive charts
import plotly.graph_objects as go
import plotly_keys as mykeys
import chart_studio
import chart_studio.plotly as py

# Set key for plotly online
chart_studio.tools.set_credentials_file(username=mykeys.username,
                                        api_key=mykeys.api)


def psi_mvn(X, theta):
    return -(X-theta.T), -np.repeat(X.shape[1], X.shape[0])

def truncsm(theta, X, Px, g, dg, psi):
    p, dp = psi(X, theta)
    t1 = np.mean(np.sum(p**2, 1)*g)
    t2 = np.mean(np.sum(dp*g))
    t3 = np.mean(np.sum(p*dg, 1))
    return t1 + 2*(t2 + t3)

def mvn_l2():
       
    # Datasets
    X   = rand.multivariate_normal([1, 1], np.eye(2), 500) # Full distribution
    Xt  = X[np.linalg.norm(X, 2, 1) < 1, :]  # Truncated to L2 ball
    Xtt = X[np.linalg.norm(X, 2, 1) >= 1, :] # Not truncated
    
    # Projection to boundary
    Px, g = df.polydist_Lqball(Xt, 2, 2)
    dg    = (Xt - Px)/g[:,None]
    
    # Set up dataframe
    dat  = pd.DataFrame(data = np.vstack((Xt, Xtt)), columns = ["x", "y"])
    dat["Label"] = np.hstack((
        np.repeat("Observed", len(Xt)),
        np.repeat("Unobserved", len(Xtt))
        ))
    
    # Estimate with truncSM
    obj = lambda theta: truncsm(theta, Xt, Px, g, dg, psi_mvn)
    mu = minimize(obj, np.random.randn(2, 1))
    print(mu.x)
    
    # Set figsize and theme
    sns.set(rc={'figure.figsize':(7.7,5.27)})    
    sns.set_style("dark")
    
    # Plot data
    ax = sns.scatterplot("x", "y", hue = "Label", data=dat,  
                         edgecolor=None, s=30, linewidth=0, palette = ["#2E00A1", "#BBBBBB"]);
    
    # Plot means
    ax.scatter(1, 1, c = "r", marker="+", label = "Mean", s = 100)
    ax.scatter(mu.x[0], mu.x[1], c = "r", label = "Estimate", s = 100)
    
    # Remove legend title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    # Remove axes
    ax.set(xlabel='', ylabel='')
    plt.savefig("plots/mvn_l2.png")
    plt.show()
    

def mvn_box():
       
    # Datasets
    X   = rand.multivariate_normal([1, 1], np.eye(2), 500) # Full distribution
    within_poly =  np.bitwise_and(X[:, 0] > -1, np.bitwise_and(X[:, 0] < 0, np.bitwise_and(X[:, 1] < 2, X[:, 1] > -1)))
    Xt  = X[within_poly, :]
    Xtt = X[~within_poly, :]
    
    # Create boundary points as a box
    boundary = np.vstack((
        np.hstack((np.linspace(-1, 0, 20)[:, None], np.repeat(2, 20)[:, None])),
        np.hstack((np.repeat(0, 60)[:, None], np.linspace(2, -1, 60)[:, None])),
        np.hstack((np.linspace(0, -1, 20)[:, None], np.repeat(-1, 20)[:, None])),
        np.hstack((np.repeat(-1, 60)[:, None], np.linspace(-1, 2, 60)[:, None]))
        ))
    
    # Loop to get projection to X
    Px = np.empty(Xt.shape)
    g  = np.empty(Xt.shape[0])
    for i in range(len(Xt)):
        dist = np.sqrt(np.sum((Xt[i, :] - boundary)**2, 1))
        Px[i, :] = boundary[np.argmin(dist), :]   
        g[i] = np.min(dist)
        
    dg = (Xt - Px)/g[:,None]
    
    # Set up dataframe
    dat  = pd.DataFrame(data = np.vstack((Xt, Xtt)), columns = ["x", "y"])
    dat["Label"] = np.hstack((
        np.repeat("Observed", len(Xt)),
        np.repeat("Unobserved", len(Xtt))
        ))
    
    # Estimate with truncSM
    obj = lambda theta: truncsm(theta, Xt, Px, g, dg, psi_mvn)
    mu = minimize(obj, np.random.randn(2, 1))
    print(mu.x)
    
    # Set figsize and theme
    sns.set(rc={'figure.figsize':(7.7,5.27)})    
    sns.set_style("dark")
    
    # Plot data
    ax = sns.scatterplot("x", "y", hue = "Label", data=dat,  
                         edgecolor=None, s=30, linewidth=0, palette = ["#2E00A1", "#BBBBBB"]);
    
    # Plot means
    ax.scatter(1, 1, c = "r", marker="+", label = "Mean", s = 100)
    ax.scatter(mu.x[0], mu.x[1], c = "r", label = "Estimate", s = 100)
    
    # Remove legend title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    # Remove axes
    ax.set(xlabel='', ylabel='')
    plt.savefig("plots/mvn_box.png")
    plt.show()


def main():
    mvn_l2() 
    mvn_box()
    
if __name__ == "__main__":
    main()
    
    