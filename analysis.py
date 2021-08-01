# In[1]:


import pandas as pd 
import numpy as np                     # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs 
import warnings   # To ignore any warnings 
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[134]:


data = pd.read_csv('Train.csv')


# In[135]:


data.head()


# In[136]:


table = pd.DataFrame(data.groupby(['Outlet_Size','Outlet_Location_Type','Outlet_Type'],as_index = False)['Item_Outlet_Sales'].sum()).sort_values('Item_Outlet_Sales',ascending=False)
cm = sns.light_palette("seagreen", as_cmap=True)
table.style.background_gradient(cmap=cm)


# In[147]:


#total sales

plt.figure(figsize=(8,5))
type2=data.groupby(['Outlet_Type'])['Item_Outlet_Sales'].sum()
store_types=['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3']
plot1 = sns.barplot(store_types,type2, color='coral')
plot1.ticklabel_format(axis="y", style="plain")
plot1.set(xlabel='Outlet Type', ylabel='Item Outlet Sales')
ylabels = ['{:,.0f}'.format(y) + 'M USD' for y in plot1.get_yticks()/1000000]
plot1.set_yticklabels(ylabels)
plt.show(block=False)
plt.close()


# In[148]:


#price

plt.rcParams['figure.figsize'] = 16,4
plot2 =sns.boxplot(x="Item_Type",y="Item_MRP",data=data,palette='husl')
plot2.set_xticklabels(plot2.get_xticklabels(), rotation=15,horizontalalignment='right', fontweight='light',fontsize='x-large')
ylabels = ['{:,.0f}'.format(y) + ' USD' for y in plot2.get_yticks()]
plot2.set_yticklabels(ylabels)
plt.show(block=False)
plt.close()


# In[149]:


#outlet types

plt.figure(figsize=(8,5))
plot3 = sns.countplot('Outlet_Type',data=data,palette='autumn')
plt.xlabel('Outlet_Type')
plt.ylabel('Count')
ylabels = ['{:,.0f}'.format(y) + ' stores' for y in plot3.get_yticks()]
plot3.set_yticklabels(ylabels)
plt.show(block=False)
plt.close()


# In[150]:


#item breakdown
df3=data.groupby(by='Item_Type').sum()
df2=df3['Item_Outlet_Sales'].sort_values(ascending=False)
plot4 = px.pie(df2, values='Item_Outlet_Sales', names=['Fruits and Vegetables', 'Snack Foods','Household ','Frozen Foods','Dairy ', 'Canned','Baking Goods','Health and Hygiene','Meat', 'Soft Drinks','Breads','Hard Drinks','Starchy Foods', 'Others','Breakfast','Seafood'])
plot4.layout.update(showlegend=False)
#plot4.show()


# In[154]:


#sales per year

y = data.groupby(['Outlet_Establishment_Year']).sum()
y = y['Item_Outlet_Sales']
x = y.index.astype(int)

plt.figure(figsize=(16,4))
plot5 = sns.barplot(y = y, x = x, palette='summer')
ax2 = plot5.twinx()
ax2.plot(plot5.get_xticks(), y, marker = 'o', color='red', linewidth=2.5)
ylabels_ax2 = ['{:,.0f}'.format(y) + 'M USD' for y in ax2.get_yticks()/1000000]
ax2.set_yticklabels(ylabels_ax2)
plot5.set_xlabel(xlabel='Year', fontsize=16)
plot5.set_xticklabels(labels = x, fontsize=12, rotation=50)
plot5.set_ylabel(ylabel='Sales', fontsize=16)
plot5.set_title(label='Sales Per Year', fontsize=20)
ylabels = ['{:,.0f}'.format(y) + 'M USD' for y in plot5.get_yticks()/1000000]
plot5.set_yticklabels(ylabels)
plt.show(block=False)
plt.close()


# In[156]:


#fat content
data['Item_Fat_Content'].replace({'reg':'Regular','low fat':'Low Fat','LF':'Low Fat'},inplace = True)


# In[157]:


df4=data.groupby(by='Item_Fat_Content').sum()
df5=df4['Item_Outlet_Sales'].sort_values(ascending=False)
plot6 = px.pie(df5, values='Item_Outlet_Sales',names= ['Low Fat','Regular'])
#plot6.show()
