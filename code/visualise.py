
import os, sys
os.chdir('/home/danny/OneDrive/Personal Stuff/nodata')
sys.path.append('/home/danny/OneDrive/Personal Stuff/nodata/code')

# Dataframes/matrices etc
import pandas as pd
import numpy as np

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

def main():
    # chicago_plotly()
    chicago_regular()

if __name__ == "__main__":
    main()