#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import dash
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


# In[ ]:





# In[7]:


def g1():
    corr = house_price_querydf.corr(method ='pearson') 
    f,ax=plt.subplots(figsize=(9,6))
    fig = sns.heatmap(corr,annot=True,linewidths=1.5,fmt='.2f',ax=ax)
    return fig


# In[8]:


def g2():
    sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white' , 'figure.figsize':(10,10)})
    fig = sns.kdeplot(data = house_price_querydf,x ='price', hue ='city' )
    return fig


# In[9]:


def g3():
    sns.set_style('whitegrid') 
    fig = sns.lmplot(x ='bedrooms', y ='price', data = house_data_final,col='city', hue ='city',height=5,col_wrap=5) 
    return fig


# In[10]:


app = Dash(__name__)


app.layout = html.Div([
    dcc.Graph(id = 'example-graph', figure=g1())
])

app.run_server(debug=True)


# In[ ]:




