
#%% [markdown]  
# ## This is markdown.

#%%
print("hello world")

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
import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np

x, y, z = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 400).transpose()

trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='3d-scatter-colorscale')

#%%
import requests 
import os, json, base64
from PIL import Image as pilimage
from urllib.request import urlopen
from IPython.display import Image, HTML, display

URL = "http://www.whateverydogdeserves.com/wp-content/uploads/2016/09/husky-meme-work.jpg"

with urlopen(URL) as response:
    with open('temp.jpg', 'bw+') as f:
        shutil.copyfileobj(response, f)

Image('temp.jpg')

#%%

def imgToBase64(img):
    imgio = BytesIO()
    img.save(imgio, 'JPEG')
    img_str = base64.b64encode(imgio.getvalue())
    return img_str.decode('utf-8')

base64Img = imgToBase64(pilimage.open(test_img))

service_uri = "http://52.190.24.229:80/score"
input_data = json.dumps({'data': base64Img})
headers = {'Content-Type':'application/json'}
result = requests.post(service_uri, input_data, headers=headers).text

print(json.loads(result))