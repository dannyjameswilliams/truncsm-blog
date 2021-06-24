# truncsm-blog
Truncated score matching in Python, for "How do we do data science without all of our data?" blog post.

[Blog post on dannyjameswilliams.co.uk (Recommended)](https://dannyjameswilliams.co.uk/post/nodata/)

[Blog post on blogs.compass.ac.uk](https://compass.blogs.bristol.ac.uk/2021/06/10/student-perspectives-data-science-without-data/)

For the full paper on this project, [please see here](https://arxiv.org/pdf/1910.03834.pdf).

## Installation 

To use this work, please clone this repository to wherever you would like. It is recommended to create a virtual environment using `conda`, and then running
```bash
pip install requirements.txt
```
while in the repository directory that you cloned to.


## Usage for reproducing plots

To reproduce the plots from the blog post, you can run any of 
```python
python3 code/mvn.py
```
or 
```python
python3 code/chicago.py
```
and the plots should appear. To view the Chicago plots 'on a map' like it is displayed on the personal blog post (not the COMPASS one), you will need to run the `chicago_plotly` and `chicago_plotly_estimate` functions inside of `chicago.py`. To save these, you need plotly API keys, as well as a Mapbox API key (all free).

Note that the random seeds for data generation or initial point generation was not set, so results are going to vary (maybe significantly) than the final blog post. 

To obtain some of the MVN simulated plots in the blog post, the code was run multiple times until a plot appeared that was 'visually appealing'. This does not affect the validity of the method, as the paper demonstrates.


## Usage for truncated score matching

I have also separated some functions into a rather basic file, not packaged in anyway, containing the barebones to truncated score matching. This involves the objective function, `truncsm` as well as a basic statistical model `psi_mvn`. This will enable you to perform estimation as exemplified in the blog posts. For a simple example, see below.

```python
import numpy as np

# Simulate MVN data using numpy
X  = np.random.multivariate_normal([1, 1], np.eye(2), 500)

# Truncate to L2 ball
Xt = X[np.linalg.norm(X, 2, 1) < 1, :] 

# Project Xt to the boundary (points on a L2 ball that are closest to X)
import data_funs as df
Px, g = df.polydist_Lqball(Xt, 2, 2)

# Calculate derivative of g
dg = (Xt - Px)/g[:,None]

# Numerically minimise the truncsm function with scipy
from scipy.optimize import minimize
mu = minimize(lambda theta: truncsm(theta, Xt, Px, g, dg, psi_mvn), np.random.randn(2, 1))
```

The `polydist_Lqball` function has been used to solve the optimisation problem to find the boundary projections `Px` as well as the values of `g`. This function is also included in this repository.

This code can be adapted, for example, you can change the `psi` function from `psi_mvn` to another valid statistical model. For compatibility with `truncsm`, there needs to be two outputs from the function, the first being the first derivative of `log p(x; \theta)` and the second being the trace of the second derivative, all output in vector form.






