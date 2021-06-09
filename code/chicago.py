
import os, sys
os.chdir('/home/danny/OneDrive/Personal Stuff/nodata')
sys.path.append('/home/danny/OneDrive/Personal Stuff/nodata/code')

# Dataframes/matrices etc
import pandas as pd
import autograd.numpy as np
import autograd.numpy.random as rand
import autograd as ag

# Optimisation
from scipy.optimize import minimize

# MLE estimation for Gaussian mixture
from sklearn import mixture

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

def logp(X, mu, sigma = 0.06): 
        return np.log(np.exp(-np.sum((X-mu[0:2])**2/(2*sigma**2), 0)) +
                      np.exp(-np.sum((X-mu[2:4])**2/(2*sigma**2), 0)))
    
grad_logp  = ag.elementwise_grad(logp, 0)
grad2_logp = ag.elementwise_grad(grad_logp, 0) 

def psi_mixed_mvn(X, theta, sigma = 0.06):
    p  = np.zeros(X.shape)
    dp = np.zeros(X.shape)
    for i in range(len(X)):
        p[i, :]  = grad_logp(X[i, :], theta)
        dp[i, :] = grad2_logp(X[i, :], theta)
        
    return p, dp

def truncsm(theta, X, Px, g, dg, psi):
    p, dp = psi(X, theta)
    t1 = np.mean(np.sum(p**2, 1)*g)
    t2 = np.mean(np.sum(dp, 1) * g)
    t3 = np.mean(np.sum(p*dg, 1))
    return t1 + 2*(t2 + t3)

def chicago_plotly(fname = "chicago"):
    crime = pd.read_csv("data/crime.csv", index_col = 0)
    bound = pd.read_csv("data/chicago_boundaries.csv", index_col = None, header=None)
    
    fig = go.Figure([
        go.Scattermapbox(lon=bound.iloc[:, 0].values, 
                     lat=bound.iloc[:, 1].values,
                     mode='markers',
                     name='markers',
                     marker=go.scattermapbox.Marker(
                         size=6
                     )
                   # hoverinfo='skip'     
    ),
    go.Scattermapbox(lon=crime["Longitude"].values, 
                   lat=crime["Latitude"].values,
                   mode='markers',
                   name='markers',
                   marker=go.scattermapbox.Marker(
                        size=11
                   )
                   # hoverinfo='skip'     
    )
    ])

    fig.update_layout(
        xaxis_title = "x",
        yaxis_title = "y",
        template = "plotly_white",
        showlegend=False,
        mapbox_style="open-street-map"
    )
    
    fig.update_layout(
        hovermode='closest',
        mapbox=dict(
            bearing = 0,
            center = go.layout.mapbox.Center(
                lat = 41.814574 + 0.025,
                lon = -87.666467
            ),
            zoom = 9.25
            )
    )
  
    
    py.plot(fig, filename = fname, auto_open=True)

def chicago_regular():
    crime = pd.read_csv("data/crime.csv", index_col = 0)
    bound = pd.read_csv("data/chicago_boundaries.csv", index_col = None, header=None)
    
    fig, axes = plt.subplots(ncols = 2, nrows = 1, figsize = (13, 8))

    
    sns.scatterplot(x=bound.values[:, 0], y = bound.values[:, 1],
                    size = 0.5, ax = axes[1], edgecolors=None, linewidth = 0)
    sns.kdeplot(x='Longitude', y='Latitude',
                data= crime,
                cmap="viridis",
                bw=.1,
                cbar=False, 
                shade=True, 
                alpha=0.75,
                shade_lowest=False,
                ax = axes[1])
    sns.scatterplot(x=bound.values[:, 0], y = bound.values[:, 1],
                    size = 0.5, ax = axes[0], edgecolors=None, linewidth = 0)
    sns.regplot('Longitude', 'Latitude',
               data= crime[['Longitude','Latitude']],
               fit_reg=False, 
               scatter_kws={'alpha':.4, 'color':'black', 'edgecolors': None, 'linewidth': 0},
               ax=axes[0])
    axes[0].get_legend().remove()
    axes[0].set_xlim(-87.9,-87.5)
    axes[0].set_ylim(41.60,42.05)
    axes[0].set_axis_off()    
    axes[1].get_legend().remove()
    axes[1].set_xlim(-87.9,-87.5)
    axes[1].set_ylim(41.60,42.05)
    axes[1].set_axis_off()    
    fig.tight_layout()
    plt.show()

def chicago_estimate():
    
    crime = pd.read_csv("data/crime.csv", index_col = 0)
    bound = pd.read_csv("data/chicago_boundaries.csv", index_col = None, header=None).dropna()
    
    X  = crime.values
    bound = bound.values
    
    # Calculate g
    g  = np.empty(X.shape[0])
    Px = np.empty(X.shape)    
    for i in range(len(X)):
        dist = np.sqrt(np.sum((X[i, :] - bound)**2, 1))
        Px[i, :] = bound[np.argmin(dist), :]
        g[i] = np.min(dist)
        
    dg = (X - Px)/g[:,None]
    
    # Estimate with truncSM
    # obj = lambda theta: truncsm(theta, X, Px, g, dg, psi_mixed_mvn)
    # ini = np.hstack((np.mean(X, 0), np.mean(X, 0))) + rand.randn(4)*0.06
    # mu = minimize(obj, ini).x
    # print(mu.x)
    mu = [-87.59468382,  41.74283899, -87.77178438, 41.87616018];

    
    # Estimate with MLE
    clf = mixture.GaussianMixture(n_components=2, covariance_type='diag')
    clf.fit(X);

    
    # Plot options
    sns.set(rc={'figure.figsize':(7.7,8.27)})    
    sns.set_style("dark")
    
    ax = sns.scatterplot(x=bound[:, 0], y = bound[:, 1], label = "Boundary",
                    size = 0.25, edgecolors=None, linewidth = 0)
    
    sns.regplot('Longitude', 'Latitude',
               data= crime[['Longitude','Latitude']],
               fit_reg=False, 
               label = "Data",
               ax = ax,
               scatter_kws={'color':'#878787', 'edgecolors': None, 'linewidth': 0})
    
    # Plot truncsm
    ax.scatter(mu[0], mu[1], c = "r", label = "truncSM", s = 100)
    ax.scatter(mu[2], mu[3], c = "r", s = 100)
    
    # Plot MLE
    ax.scatter(clf.means_[0, 0], clf.means_[0, 1], c = "r", label = "MLE", s = 100, marker = "x")
    ax.scatter(clf.means_[1, 0], clf.means_[1, 1], c = "r", s = 100, marker = "x")
    
    # Remove legend title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[3:], labels=labels[3:])
    
    ax.set_xlim(-87.9,-87.5)
    ax.set_ylim(41.60,42.05)
    
    plt.savefig("plots/chicago_estimate.png")
    plt.show()
    
    
    
def main():
    # chicago_plotly()
    # chicago_regular()
    chicago_estimate()
if __name__ == "__main__":
    main()