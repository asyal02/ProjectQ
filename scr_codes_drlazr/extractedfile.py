
# coding: utf-8

# In[1]:


import tarfile


# In[2]:


tar= tarfile.open("tstat.dtn05.nersc.gov.testb.all.csv.tar.gz")


# In[3]:


tar.extractall()


# In[4]:


tar = tarfile.open("tstat.dtn05.nersc.gov.testb.all.csv.tar.gz", "r:gz")

