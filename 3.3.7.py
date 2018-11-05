#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
my_array = np.array([['Montgomery','Yellohammer state',52423],
                     ['Sacramento','Golden state',163707],
                     ['Oklahoma City','Sooner state',69960 ]])
b=pd.DataFrame(my_array)
b


# In[3]:


b.columns=['a1','b1','c1']
b.index=['cal','bos','ny']
b


# In[5]:


names = ['George',
         'John',
         'Thomas',
         'James',
         'Andrew',
         'Martin',
         'William',
         'Zachary',
         'Millard',
         'Franklin']
data=pd.DataFrame(index=names)
print(data)


# In[28]:


data['country']=['USA','CAN','USA','IND','IND','GRE','GER','NETH','UK','GER']
data['country']


# In[33]:


data['ad_views'] = [16, 42, 32, 13, 63, 19, 65, 23, 16, 77]
data['items_purchased'] = [2, 1, 0, 8, 0, 5, 7, 3, 0, 5]
data['avg']=data['items_purchased']/data['ad_views']
print (data)
data['items_purchased']/data['ad_views']


# In[17]:


data.loc['George']


# In[20]:


data.loc[:,'ad_views']


# In[22]:


data.iloc[2:6,2]


# data.loc[lambda b:data['items_purchased']>2,:]

# In[23]:


data.loc[lambda b:data['items_purchased']>2,:]


# In[27]:


data.loc[lambda c:data['ad_views']>10,:]


# In[25]:


data[data['items_purchased'] > 1]


# In[29]:


data.groupby('country')


# In[34]:


data.groupby('country')['ad_views'].aggregate(np.mean)


# In[35]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot([0, 1, 2, 3])
plt.show()


# In[36]:


np.random.seed(1221)


# In[37]:


df = pd.DataFrame()


# In[38]:


df


# In[39]:


df['rand'] = np.random.rand(100)


# In[44]:


df['rand_sq'] = df['rand'] ** 2
df['rand_shift'] = df['rand'] + 2
df['counts_sq'] = df.index ** 2
df['counts_sqrt'] = np.sqrt(df.index)
df


# In[45]:


plt.plot(df['rand'])
plt.show()


# In[51]:


plt.plot(df['rand'], color='purple')
plt.plot(df['rand_shift'],color='green')
plt.ylim([-0.1, 4.1])
plt.ylabel('Values')
plt.title('Random Series')
plt.show()


# In[52]:


plt.scatter(x=df['rand'], y=df['rand_sq'])
plt.show()


# In[55]:


plt.scatter(
    x=df['rand'],
    y=df['rand_sq'],
    color='purple',
    marker='x', s=10
)
plt.scatter(
    x=df['rand'],
    y=df['rand_shift'],
    color='green',
    marker='x', s=10
)
plt.show()


# In[62]:


df.plot(kind='scatter', x='counts_sq',y= 'counts_sqrt')
df.plot(kind='line')
plt.show()


# In[63]:


plt.figure(figsize=(10, 5))


# In[64]:


plt


# In[65]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(df['rand'], color='purple')
plt.ylabel('Values')
plt.title('Random Series')


# In[73]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(df['rand'], color='purple')
plt.ylabel('Values')
plt.title('Random Series')

plt.subplot(1, 2, 2)
plt.scatter(x = df['rand_sq'], y = df['rand'], color='green')
plt.ylabel('Squared Values')
plt.title('Squared Series')

plt.tight_layout()
plt.show()


# In[74]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Data to play with. Twice the histograms, twice the fun.
x = np.random.normal(10, 5, 1000)
y = np.random.normal(15, 5, 10000)

# Override bin defaults with specific bin bounds.
# FYI `alpha` controls the opacity.
plt.hist(x, color='blue', bins=np.arange(-10, 40), alpha=.5) 
plt.hist(y, color='red', bins=np.arange(-10, 40), alpha=.5)
plt.title('Manually setting bin placement')
plt.xlabel('Random Values')

plt.show()


# In[2]:


import pandas as pd

# Make a blank data frame.
df = pd.DataFrame()

# Populate it with data.
df['age'] = [14, 12, 11, 10, 8, 6, 8]
import numpy as np

np.mean(df['age'])
import statistics

statistics.median(df['age'])

# Using NumPy.
#import numpy as np

#np.median(df['age'])


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
bernoulli= np.random.binomial(1, .5, 100)

#Plot a histogram.
plt.hist(bernoulli)
plt.axvline(bernoulli.mean(), color='b', linestyle='solid', linewidth=2)

# Add a vertical line at one standard deviation above the mean.
plt.axvline(bernoulli.mean() + bernoulli.std(), color='b', linestyle='dashed', linewidth=2)
# Print the histogram


# Print the histogram.

plt.show()


# In[9]:


binomial = np.random.binomial(20, 0.5, 100)

# Plot a histogram.
plt.hist(binomial)
plt.axvline(binomial.mean(), color='b', linestyle='solid', linewidth=2)

# Add a vertical line at one standard deviation above the mean.
plt.axvline(binomial.mean() + binomial.std(), color='b', linestyle='dashed', linewidth=2)
plt.show()


# In[10]:


gamma = np.random.gamma(5,1, 100)
plt.axvline(gamma.mean(), color='b', linestyle='solid', linewidth=2)

# Add a vertical line at one standard deviation above the mean.
plt.axvline(gamma.mean() + gamma.std(), color='b', linestyle='dashed', linewidth=2)
# Plot a histogram.
plt.hist(gamma)

# Print the histogram.
plt.show()


# In[11]:


poisson = np.random.poisson(3, 100)
plt.axvline(poisson.mean(), color='b', linestyle='solid', linewidth=2)

# Add a vertical line at one standard deviation above the mean.
plt.axvline(poisson.mean() + poisson.std(), color='b', linestyle='dashed', linewidth=2)
# Plot a histogram.
plt.hist(poisson)

# Print the histogram.
plt.show()


# In[12]:


norm1 = np.random.normal(5, 0.5, 1000)
norm2 = np.random.normal(10, 1, 1000)
third=norm1+norm2
mean = np.mean(third)
sd = np.std(third)
plt.hist(third)
plt.axvline(third.mean(), color='b', linestyle='solid', linewidth=2)

# Add a vertical line at one standard deviation above the mean.
plt.axvline(third.mean() + third.std(), color='b', linestyle='dashed', linewidth=2)
plt.axvline(third.mean()-third.std(), color='b', linestyle='dashed', linewidth=2)
plt.show()


# In[23]:


pop1 = np.random.binomial(10, 0.2, 10000)
pop2 = np.random.binomial(10,0.5, 10000) 

# Let’s make histograms for the two groups. 

plt.hist(pop1, alpha=0.5, label='Population 1') 
plt.hist(pop2, alpha=0.5, label='Population 2') 
plt.legend(loc='upper right') 
plt.show()
sample1 = np.random.choice(pop1, 1000, replace=True)
sample2 = np.random.choice(pop2, 1000, replace=True)

plt.hist(sample1, alpha=0.5, label='sample 1') 
plt.hist(sample2, alpha=0.5, label='sample 2') 
plt.legend(loc='upper right') 
plt.show()
print(sample1.mean())
print(sample2.mean())
print(sample1.std())
print(sample2.std())
sample1 = np.random.choice(pop1, 20, replace=True)
sample2 = np.random.choice(pop2, 20, replace=True)

plt.hist(sample1, alpha=0.5, label='sample 1') 
plt.hist(sample2, alpha=0.5, label='sample 2') 
plt.legend(loc='upper right') 
plt.show()
print(sample1.mean())
print(sample2.mean())
print(sample1.std())
print(sample2.std())


# In[24]:


pop1 = np.random.binomial(10, 0.3, 10000)
pop2 = np.random.binomial(10,0.5, 10000) 

# Let’s make histograms for the two groups. 

plt.hist(pop1, alpha=0.5, label='Population 1') 
plt.hist(pop2, alpha=0.5, label='Population 2') 
plt.legend(loc='upper right') 
plt.show()
sample1 = np.random.choice(pop1, 100, replace=True)
sample2 = np.random.choice(pop2, 100, replace=True)

plt.hist(sample1, alpha=0.5, label='sample 1') 
plt.hist(sample2, alpha=0.5, label='sample 2') 
plt.legend(loc='upper right') 
plt.show()
from scipy.stats import ttest_ind
print(ttest_ind(sample2, sample1, equal_var=False))


# In[22]:


pop1 = np.random.binomial(10, 0.4, 10000)
pop2 = np.random.binomial(10,0.5, 10000) 

# Let’s make histograms for the two groups. 

plt.hist(pop1, alpha=0.5, label='Population 1') 
plt.hist(pop2, alpha=0.5, label='Population 2') 
plt.legend(loc='upper right') 
plt.show()
sample1 = np.random.choice(pop1, 100, replace=True)
sample2 = np.random.choice(pop2, 100, replace=True)

plt.hist(sample1, alpha=0.5, label='sample 1') 
plt.hist(sample2, alpha=0.5, label='sample 2') 
plt.legend(loc='upper right') 
plt.show()
from scipy.stats import ttest_ind
print(ttest_ind(sample2, sample1, equal_var=False))


# In[ ]:




