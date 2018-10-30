#!/usr/bin/env python
# coding: utf-8

# In[4]:


with open('C:/Personal/09142640/Desktop/poem.txt') as poem_file:
    text = poem_file.readlines()
    print("This file is {} lines long".format(len(text)))
    for line in text:
        print(line)


# In[5]:


import pandas as pd
df=pd.read_csv("C:/Personal/09142640/Desktop/NBA_player_of_the_week.csv")
print(df)


# 

# In[201]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(1,1))
df=pd.read_csv("C:/Personal/09142640/Desktop/NBA_player_of_the_week1.csv")
df1=df[df.Season_short==1985]
fig,ax=plt.subplots()
#All players with their playing years
ax.plot(df1['Player'],df1['Season_short'],linestyle='', marker='o')
plt.ylabel('Year')
#plt.title('Player stats')#plt.subplot(1, 2, 2)
#plt.plot(df1['Season short'], color='green')
plt.xlabel('Player')
#plt.xticks(y_pos, bars)
plt.title('Player stats')
plt.show()

#Top 5 players in subplot
plt.subplot(1, 2, 2)
#counts=df.groupby('Player')['Player'].count()
counts=df['Player'].value_counts()
#counts
df_count = pd.DataFrame(counts)
#count=df_count[columns='count']

df_counts=counts.reset_index()
df_counts
Top_5=df_counts.sort_values('Player',ascending=False).head(5)
#top 5 players  
plt.plot(Top_5['Player'],Top_5['index'])
plt.title('Top 5 NBA Players based on Awards')
plt.xlabel('Awards Won')
plt.ylabel('Player')
plt.show()
bar_width = 0.35
opacity = 0.8

#Top 5 players on barchart
bar1= plt.bar( Top_5['index'],Top_5['Player'], bar_width,
                 alpha=opacity,
                 color='b'
                 )
plt.xlabel('Player')
plt.ylabel('Awards')
#plt.legend()
plt.title('Top 5 NBA Players based on Awards')
plt.tight_layout()
plt.show()

#top5 players with their playing years
fig,ax=plt.subplots()
#ax.plot(df1['Player'],df1['Season_short']<=Top_5['index'],linestyle='', marker='o')
df3= df.loc[df['Player'] .isin(Top_5['index'])]
plt.plot(df3['Player'],df3['Season_short'],linestyle=' ', marker='o')
plt.title('Top 5 NBA Players with their playing years')
plt.ylabel('Year')
plt.xlabel('Player')
plt.show()


# In[8]:


# Make the random function consistent and replicable.
np.random.seed(1221)

# Make a blank data frame.
df = pd.DataFrame()

# Add a column of random numbers between 0 and 1.
df['rand'] = np.random.rand(100)
df['rand_sq'] = df['rand'] ** 2
df['rand_shift'] = df['rand'] + 2

# When creating a data frame an index column of counts is created, counting from 0.
# Here we do a few transforms on that index to create some extra columns.
df['counts_sq'] = df.index ** 2 
df['counts_sqrt'] = np.sqrt(df.index)
df
#plt.plot(df['rand'])
#plt.show()


# In[ ]:




