#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install --upgrade transformers accelerate peft')
get_ipython().system('pip install wandb')


# In[1]:


import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score



# In[2]:


data = pd.read_csv('cellula toxic data  (1).csv')
data = pd.DataFrame(data)


# In[3]:


def combine_query_image_descriptions(data, query_col='query', image_col='image descriptions', new_col='text'):
    data[new_col] = data[query_col].astype(str) + " " + data[image_col].astype(str)
    return data
data = combine_query_image_descriptions(data)
data=data.drop(columns=['image descriptions','query'])
data = data.rename(columns={'Toxic Category':'labels'})
data.head()


# In[4]:


text_column = "text"      
label_column = "labels"   

unique_labels = data[label_column].unique().tolist()
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
data["label_id"] = data[label_column].map(label2id)


# # tokenization&Data Preparation 

# In[5]:


class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=500):
        self.texts = df[text_column].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# In[6]:


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
num_labels = len(unique_labels)

base_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)


# # LoRA

# In[7]:


from peft import LoraConfig, get_peft_model, TaskType

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.2,
    target_modules=[
        "q_lin",  
        "k_lin",  
        "v_lin",  
        "out_lin" 
    ],
)

model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters() 


# # Data Spliting

# In[8]:


train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

train_dataset = ToxicDataset(train_df, tokenizer)
val_dataset = ToxicDataset(val_df, tokenizer)


# # Train_args

# In[9]:


training_args = TrainingArguments(
    output_dir="./lora_distilbert",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    metric_for_best_model="f1",
)


# # Evaluation metrics

# In[10]:


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# # training 

# In[11]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)


# In[12]:


trainer.train()


# # Graphs

# In[13]:


logs = pd.DataFrame(trainer.state.log_history)

train_loss = logs[logs["loss"].notnull()][["step","loss"]]
eval_loss = logs[logs["eval_loss"].notnull()][["step","eval_loss","eval_accuracy"]]

plt.figure(figsize=(10,4))
plt.plot(train_loss["step"], train_loss["loss"], label="Train Loss")
plt.plot(eval_loss["step"], eval_loss["eval_loss"], label="Val Loss")
plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Loss over time"); plt.legend(); plt.show()


# In[ ]:




