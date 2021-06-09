import os
os.chdir('/home/danny/OneDrive/Personal Stuff/nodata')
import code.data_funs as df

import numpy as np
import numpy.random as rand


import plotly.graph_objects as go
import plotly_keys as mykeys
import chart_studio
chart_studio.tools.set_credentials_file(username=mykeys.username,
                                        api_key=mykeys.api)
import chart_studio.plotly as py

# Variables
n = 500

Xall = rand.multivariate_normal([1, 1], np.eye(2), n)
Xt   = Xall[np.linalg.norm(Xall, 2, 1) < 1, :]
Px   = df.get_Px(h=0.01)


fig1 = go.Figure([
    go.Scatter(x=Xall[:, 0], 
               y=Xall[:, 1],
               mode='markers',
               name='markers',
               marker=dict(
                      size=10,
                      color="blue"
                     ),
               hoverinfo='skip'     
    ),
    go.Scatter(x=Xt[:, 0], 
               y=Xt[:, 1],
               mode='markers',
               name='markers',
               marker=dict(
                      size=11,
                      color="red"
                     ),
               hoverinfo='skip'
    )
    ])

fig1.update_layout(
    xaxis_title = "x",
    yaxis_title = "y",
    template = "plotly_white",
    showlegend=False
)

# fig1.show()

py.plot(fig1, filename = "test_mvn", auto_open=True)

