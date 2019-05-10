#%%

from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

# Generate a large random dataset
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(ascii_letters[26:]))

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


#%%

import plotly.plotly as py
import plotly.graph_objs as go

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_ebola.csv')
df.head()

cases = []
colors = ['rgb(239,243,255)','rgb(189,215,231)','rgb(107,174,214)','rgb(33,113,181)']
months = {6:'June',7:'July',8:'Aug',9:'Sept'}

for i in range(6,10)[::-1]:
    cases.append(go.Scattergeo(
            lon = df[ df['Month'] == i ]['Lon'], #-(max(range(6,10))-i),
            lat = df[ df['Month'] == i ]['Lat'],
            text = df[ df['Month'] == i ]['Value'],
            name = months[i],
            marker = go.scattergeo.Marker(
                size = df[ df['Month'] == i ]['Value']/50,
                color = colors[i-6],
                line = go.scattergeo.marker.Line(width = 0)
            )
        )
    )

cases[0]['text'] = df[ df['Month'] == 9 ]['Value'].map('{:.0f}'.format).astype(str)+' '+\
    df[ df['Month'] == 9 ]['Country']
cases[0]['mode'] = 'markers+text'
cases[0]['textposition'] = 'bottom center'

inset = [
    go.Choropleth(
        locationmode = 'country names',
        locations = df[ df['Month'] == 9 ]['Country'],
        z = df[ df['Month'] == 9 ]['Value'],
        text = df[ df['Month'] == 9 ]['Country'],
        colorscale = [[0,'rgb(0, 0, 0)'],[1,'rgb(0, 0, 0)']],
        autocolorscale = False,
        showscale = False,
        geo = 'geo2'
    ),
    go.Scattergeo(
        lon = [21.0936],
        lat = [7.1881],
        text = ['Africa'],
        mode = 'text',
        showlegend = False,
        geo = 'geo2'
    )
]

layout = go.Layout(
    title = go.layout.Title(
        text = 'Ebola cases reported by month in West Africa 2014<br> \
Source: <a href="https://data.hdx.rwlabs.org/dataset/rowca-ebola-cases">\
HDX</a>'),
    geo = go.layout.Geo(
        resolution = 50,
        scope = 'africa',
        showframe = False,
        showcoastlines = True,
        showland = True,
        landcolor = "rgb(229, 229, 229)",
        countrycolor = "rgb(255, 255, 255)" ,
        coastlinecolor = "rgb(255, 255, 255)",
        projection = go.layout.geo.Projection(
            type = 'mercator'
        ),
        lonaxis = go.layout.geo.Lonaxis(
            range= [ -15.0, -5.0 ]
        ),
        lataxis = go.layout.geo.Lataxis(
            range= [ 0.0, 12.0 ]
        ),
        domain = go.layout.geo.Domain(
            x = [ 0, 1 ],
            y = [ 0, 1 ]
        )
    ),
    geo2 = go.layout.Geo(
        scope = 'africa',
        showframe = False,
        showland = True,
        landcolor = "rgb(229, 229, 229)",
        showcountries = False,
        domain = go.layout.geo.Domain(
            x = [ 0, 0.6 ],
            y = [ 0, 0.6 ]
        ),
        bgcolor = 'rgba(255, 255, 255, 0.0)',
    ),
    legend = go.layout.Legend(
           traceorder = 'reversed'
    )
)

fig = go.Figure(layout=layout, data=cases+inset)
py.iplot(fig, filename='West Africa Ebola cases 2014')
#%%

import plotly.plotly as py
import plotly.graph_objs as go

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_us_cities.csv')
df.head()

df['text'] = df['name'] + '<br>Population ' + (df['pop']/1e6).astype(str)+' million'
limits = [(0,2),(3,10),(11,20),(21,50),(50,3000)]
colors = ["rgb(0,116,217)","rgb(255,65,54)","rgb(133,20,75)","rgb(255,133,27)","lightgrey"]
cities = []
scale = 5000

for i in range(len(limits)):
    lim = limits[i]
    df_sub = df[lim[0]:lim[1]]
    city = go.Scattergeo(
        locationmode = 'USA-states',
        lon = df_sub['lon'],
        lat = df_sub['lat'],
        text = df_sub['text'],
        marker = go.scattergeo.Marker(
            size = df_sub['pop']/scale,
            color = colors[i],
            line = go.scattergeo.marker.Line(
                width=0.5, color='rgb(40,40,40)'
            ),
            sizemode = 'area'
        ),
        name = '{0} - {1}'.format(lim[0],lim[1]) )
    cities.append(city)

layout = go.Layout(
        title = go.layout.Title(
            text = '2014 US city populations<br>(Click legend to toggle traces)'
        ),
        showlegend = True,
        geo = go.layout.Geo(
            scope = 'usa',
            projection = go.layout.geo.Projection(
                type='albers usa'
            ),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        )
    )

fig = go.Figure(data=cities, layout=layout)
py.iplot(fig, filename='d3-bubble-map-populations')

#%%

import plotly.plotly as py
import plotly.graph_objs as go

import pandas as pd
try:
    from itertools import izip# Python 2
except ImportError:
    izip = zip# Python 3

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/globe_contours.csv')
df.head()

contours = []

scl = ['rgb(213,62,79)', 'rgb(244,109,67)', 'rgb(253,174,97)', \
    'rgb(254,224,139)', 'rgb(255,255,191)', 'rgb(230,245,152)', \
    'rgb(171,221,164)', 'rgb(102,194,165)', 'rgb(50,136,189)'
]

def pairwise(iterable):
    a = iter(iterable)
    return izip(a, a)

i = 0
for lat, lon in pairwise(df.columns):
    contours.append(go.Scattergeo(
        lon = df[lon],
        lat = df[lat],
        mode = 'lines',
        line = go.scattergeo.Line(
            width = 2,
            color = scl[i]
        )))
    i = 0 if i + 1 >= len(df.columns) / 4 else i + 1

layout = go.Layout(
    title = go.layout.Title(
        text = 'Contour lines over globe<br>(Click and drag to rotate)'
    ),
    showlegend = False,
    geo = go.layout.Geo(
        showland = True,
        showlakes = True,
        showcountries = True,
        showocean = True,
        countrywidth = 0.5,
        landcolor = 'rgb(230, 145, 56)',
        lakecolor = 'rgb(0, 255, 255)',
        oceancolor = 'rgb(0, 255, 255)',
        projection = go.layout.geo.Projection(
            type = 'orthographic',
            rotation = go.layout.geo.projection.Rotation(
                lon = -100,
                lat = 40,
                roll = 0
            )
        ),
        lonaxis = go.layout.geo.Lonaxis(
            showgrid = True,
            gridcolor = 'rgb(102, 102, 102)',
            gridwidth = 0.5
        ),
        lataxis = go.layout.geo.Lataxis(
            showgrid = True,
            gridcolor = 'rgb(102, 102, 102)',
            gridwidth = 0.5
        )
    )
)

fig = go.Figure(data = contours, layout = layout)
py.iplot(fig, filename = 'd3-globe')

#%%

import altair as alt
#alt.renderers.enable('notebook')

import pandas as pd
url = 'https://raw.githubusercontent.com/blmoore/blogR/master/data/measles_incidence.csv'
data = pd.read_csv(url, skiprows=2, na_values='-')
data.head()

annual = data.drop('WEEK', axis=1).groupby('YEAR').sum()
annual.head()


measles = annual.reset_index()
measles = pd.melt(measles, 'YEAR', var_name='state', value_name='incidence')
measles.head()

alt.Chart(measles).mark_rect().encode(
    x='YEAR:O',
    y='state:N',
    color='incidence'
).properties(
    width=600,
    height=400
)

# Define a custom colormape using Hex codes & HTML color names
colormap = alt.Scale(domain=[0, 100, 200, 300, 1000, 3000],
                     range=['#F0F8FF', 'cornflowerblue', 'mediumseagreen', '#FFEE00', 'darkorange', 'firebrick'],
                     type='sqrt')

alt.Chart(measles).mark_rect().encode(
    alt.X('YEAR:O', axis=alt.Axis(title=None, ticks=False)),
    alt.Y('state:N', axis=alt.Axis(title=None, ticks=False)),
    alt.Color('incidence:Q', sort='ascending', scale=colormap, legend=None)
).properties(
    width=800,
    height=500
)
#%%
