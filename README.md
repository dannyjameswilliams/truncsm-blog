# truncsm-blog
Truncated score matching in Python, for "How do we do data science without all of our data?" blog post.

To reproduce the plots from the blog post, you can run any of 
```
python3 code/mvn.py
```
or 
```
python3 code/chicago.py
```
and the plots should appear. 

Note that the random seeds for data generation or initial point generation was not set, so results are going to vary (somewhat significantly) than the final blog post. 

To obtain some of the plots in the blog post, the code was run multiple times until a plot appeared that was 'visually appealing'. This does not affect the validity of the method, as the paper demonstrates.

For the full paper on this project, [please see here](https://arxiv.org/pdf/1910.03834.pdf).
