# Toxicity Classification

Project materials, scripts, and notebooks for Toxicity Classification . This repository contains experiments for toxicity classification using both a fine-tuned DistilBERT model (with LoRA) and a BiLSTM baseline . 

**Project structure**
- **`cellula toxic data  (1).csv`**: Primary dataset used by notebooks and scripts. Contains columns including `query`, `image descriptions`, and `Toxic Category` (used as `labels`).
  - **`BERT/`**: Reference materials and PDFs about DistilBERT, ALBERT, LoRA, and QLoRA.
    - `DistilBERT & ALBERT.pdf`, `LoRA & QLoRA.pdf`
  - **`Bonus/`**: DistilBERT fine-tuning code and notebook.
    - **`DistilBERT.ipynb`**: Notebook performing data preparation, tokenization, LoRA configuration, training with the Hugging Face `Trainer`, and evaluation/plots.
    - **`DistilBERT.py`**: A script-version of the notebook. It includes inline pip-install commands (notebook magics) and uses: `pandas`, `torch`, `transformers`, `peft`, `accelerate`, `sklearn`, `matplotlib`, and `wandb` (optional). This file expects `cellula toxic data  (1).csv` in the working directory.
  - **`LSTM/`**: Classical RNN baseline implemented with PyTorch.
    - **`BiLSTM.ipynb`**: Notebook with data cleaning, augmentation, vocabulary building, BiLSTM model definition, training, and evaluation.
    - **`BiLSTM.py`**: Script version of the BiLSTM pipeline. Dependencies include `pandas`, `numpy`, `nltk`, `nlpaug`, `deep_translator`, `torch`, `sklearn`, `seaborn`, and `matplotlib`.
  - **`Report&Results.pdf`**,'Result for Toxicity Classification using BiLSTM' 
  - **`LoRA & QLoRA.pdf`**, 'A Quick introduction for LoRA and QLoRA'
  - **`DistilBERT & ALBERT.pdf`**, 'A Quick introduction for DistilBERT and ALBERT'


## Overview
This folder contains two main modeling approaches for toxicity classification:

- DistilBERT + LoRA (parameter-efficient fine-tuning). Implemented in `DistilBERT.ipynb` and provided as `DistilBERT.py`.
- BiLSTM baseline using custom tokenization, embedding and a bidirectional LSTM. Implemented in `BiLSTM.ipynb` and `BiLSTM.py`.

Both approaches read from the same CSV dataset and produce trained models and evaluation plots.

## Recommended environment setup (Windows PowerShell)
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install common dependencies (example). Adjust versions as needed:

```powershell
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab notebook
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # choose correct CUDA/CPU wheel
pip install transformers accelerate peft wandb nlpaug deep-translator nltk
```

Notes:
- `DistilBERT.py` contains notebook-style pip calls; it's recommended to run the accompanying `DistilBERT.ipynb` (or remove the `get_ipython()` magics) if running as a plain script.
- If you don't have a GPU, set `device='cpu'` in scripts or ensure PyTorch CPU-only build is installed.

## How to run

- Jupyter notebooks: Launch a notebook server and open the `.ipynb` files:

```powershell
jupyter lab
# or
jupyter notebook
```

- Run the DistilBERT notebook (preferred for step-by-step execution):

Open `DistilBERT.ipynb` in Jupyter and run cells in order. It will:
  - read `cellula toxic data  (1).csv`
  - combine `query` + `image descriptions` into a single `text` column
  - tokenize with `distilbert-base-uncased`
  - configure LoRA via `peft` and train with `Trainer`

- Run the BiLSTM notebook: Open `BiLSTM.ipynb` and run cells. It will:
  - read and clean the CSV
  - apply augmentation with `nlpaug` and `deep_translator` back-translation
  - build vocabulary, prepare DataLoaders and train the BiLSTM model

- Execute scripts (script usage notes):
  - `DistilBERT.py` — may require removing notebook magics at the top (the `get_ipython().system('pip install ...')` calls) and ensuring the working directory contains the CSV. Run with:

```powershell
python "DistilBERT.py"
```

  - `BiLSTM.py` — runs as a script but expects any required NLTK corpora and models installed (e.g., `stopwords`) and may require a GPU for reasonable speed. Run with:

```powershell
python "BiLSTM.py"
```

## Important implementation notes & tips
- The notebooks/scripts assume `cellula toxic data  (1).csv` is in the same working directory; set your working directory accordingly or update file paths.
- NLTK: If running `BiLSTM.py` or the LSTM notebook, ensure NLTK data is downloaded (stopwords):

```python
import nltk
nltk.download('stopwords')
```

- Data augmentation in `BiLSTM` uses `nlpaug` (contextual and back-translation). Back-translation may require significant time and resources.
- DistilBERT training uses `Trainer` and `peft`'s LoRA. Check the `TrainingArguments` in `DistilBERT.py` for batch sizes, learning rate, and fp16 behavior.

---

- Author: Mohamed Eisa.

