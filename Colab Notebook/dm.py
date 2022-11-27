#!/usr/bin/env python
# coding: utf-8

# In[10]:


# !pip install -U kaleido


# In[1]:


import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots


# In[ ]:





# In[ ]:





# In[2]:


import dash
import plotly
from dash import dcc
from dash import Dash
from dash import html
from dash.dependencies import Output, Input, State
from datetime import date
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from google.oauth2 import service_account  # pip install google-auth
import pandas_gbq  # pip install pandas-gbq
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.tools as tls
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot


# In[3]:


credentials = service_account.Credentials.from_service_account_file('datamining-364220.json')
project_id = 'datamining-364220'


# In[4]:



query1  = """
SELECT
    *
FROM
    `datamining-364220.house_price.preprocessed_data`
"""


# In[5]:


house_price_querydf = pd.read_gbq(credentials=credentials,query=query1,dialect='standard',project_id=project_id)
house_price_querydf.head()


# In[12]:


fig = make_subplots(specs=[[{"secondary_y": True}]])
unique_cities=['Seattle','Renton','Bellevue','Bellevue','Redmond','Kirkland','Issaquah','Kent','Auburn','Sammamish','Federal Way','Shoreline','Woodinville']
for city in unique_cities:
    df_city = house_price_querydf[house_price_querydf["city"] == city]
    fig.add_trace(go.Histogram(x=df_city.price, y=df_city.sqft_living, name=city ))

fig.show()
    
    
    


# In[16]:


type(fig)


# In[ ]:





# In[5]:


def g1():
    corr = house_price_querydf.corr(method ='pearson') 
    f,ax=plt.subplots(figsize=(9,6))
    fig = sns.heatmap(corr,annot=True,linewidths=1.5,fmt='.2f',ax=ax)
    return fig


# In[6]:


def g2():
    sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white' , 'figure.figsize':(10,10)})
    fig = sns.kdeplot(data = house_price_querydf,x ='price', hue ='city' )
    plotly_fig = tls.mpl_to_plotly(fig)
    return plotly_fig


# In[7]:


def g3():
    sns.set_style('whitegrid') 
    fig = sns.lmplot(x ='bedrooms', y ='price', data = house_data_final,col='city', hue ='city',height=5,col_wrap=5) 
    return fig


# In[ ]:





# In[14]:


def p1():
    corr = house_price_querydf.corr(method ='pearson')
    fig = px.imshow(corr, text_auto=True)
    return fig


# In[ ]:





# In[8]:



# iplot(plotly_fig)


# In[17]:


app = Dash(__name__)


app.layout = html.Div([
    html.H1(children=' Python Dash Boards Using Dash '),
    html.H2(children=' Histogram for the Living Room Area vs House Price for all the Cities  '),
    dcc.Graph(figure=fig),
    html.H2(children=' Corelation Of Numerical Cols and House Price  '),
    dcc.Graph(figure=p1()),
])

app.run_server(debug=True)


# In[ ]:




