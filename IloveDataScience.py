
#%% [markdown]  
# ## This is markdown.

#%%
print("hello world")
print("hello Live share")

#%%
x = 2
y = 3
print(x+y)

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
p = np.linspace(0,20,100)
plt.plot(p,np.sin(p))
plt.show()

#%%
# Let's load and review some data
df = pd.read_csv("./data/pima-data.csv") # load Pima data
df.head(5)

#%% Let's plot the correlation between data columns
import matplotlib.pyplot as plt # matplotlib.pyplot plots data

def draw_corr(df, size = 11):
    corr = df.corr() # data frame correlation function
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.matshow(corr) # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns) # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns) # draw y tick marks
    
draw_corr(df)

#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="dark")
rs = np.random.RandomState(50)

# Set up the matplotlib figure
f, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

# Rotate the starting point around the cubehelix hue circle
for ax, s in zip(axes.flat, np.linspace(0, 3, 10)):

    # Create a cubehelix colormap to use with kdeplot
    cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)

    # Generate and plot a random bivariate dataset
    x, y = rs.randn(2, 50)
    sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=ax)
    ax.set(xlim=(-3, 3), ylim=(-3, 3))

f.tight_layout()
