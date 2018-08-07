
# coding: utf-8

# In[1]:


get_ipython().magic('load_ext watermark')
get_ipython().magic('watermark -p tensorflow,numpy,pandas -v -m')


# In[2]:


import tensorflow as tf
import numpy as np
import pandas as pd

from inspect_lm import build_graph, transform_texts, TEXT_ENCODER


# ## Load Model

# In[3]:


graph = tf.Graph()
sess =  tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=graph)
with graph.as_default():
    X, M, lm_logits, lm_losses = build_graph(sess)


# ## Prepare and Examine Test Data

# In[4]:


list_of_texts =[
    "Karen was assigned a roommate her first year of college. Her roommate asked her to go to a nearby city for a concert. Karen agreed happily. The show was absolutely exhilarating.",
    "Jim got his first credit card in college. He didn’t have a job so he bought everything on his card. After he graduated he amounted a $10,000 debt. Jim realized that he was foolish to spend so much money.",
    "Gina misplaced her phone at her grandparents. It wasn’t anywhere in the living room. She realized she was in the car before. She grabbed her dad’s keys and ran outside."
]


# In[5]:


x, m = transform_texts(list_of_texts)
x.shape


# In[6]:


DECODER = TEXT_ENCODER.decoder
restored = []
for token, mask in zip(x[0, :, 0], m[0, :]):
    # if DECODER[token] != "<unk>":
    if mask:
        restored.append(DECODER[token].replace("</w>", ""))
" ".join(restored)


# ## Model Prediction

# In[7]:


batch_lm_logits, batch_lm_losses = sess.run([lm_logits, lm_losses], {X: x, M: m})


# In[8]:


first_choices = np.argmax(batch_lm_logits, axis=-1)
first_choices.shape


# In[9]:


DECODER = TEXT_ENCODER.decoder
restored = []
preds = []
for token, mask, pred in zip(x[0, :, 0], m[0, :], first_choices[0, :]):
    # if DECODER[token] != "<unk>":
    if mask:
        restored.append(DECODER[token].replace("</w>", ""))
        preds.append(DECODER.get(pred, "<ctx_token>").replace("</w>", ""))
print(" ".join(restored))
print(" ".join(preds))


# In[10]:


def collect_predictions(idx, topk=3):
    topk_preds = [["<start>"] for _ in range(topk)]
    original = []
    for token, mask, logits in zip(x[idx, :, 0], m[idx, :], batch_lm_logits[idx, :, :]):
        top_tokens = np.argsort(logits, axis=-1)[::-1][::topk]
        if mask:
            original.append(DECODER[token].replace("</w>", ""))
            for i in range(topk):
                topk_preds[i].append(DECODER.get(top_tokens[i], "<ctx_token>").replace("</w>", ""))
    original.append("<end>")
    df = pd.DataFrame({"original": original})
    for i in range(topk):
        df[f"pred_{i+1}"] = topk_preds[i]
    return df
collect_predictions(0).transpose()


# In[11]:


collect_predictions(1).transpose()


# In[12]:


collect_predictions(2).transpose()


# In[13]:


batch_lm_losses

