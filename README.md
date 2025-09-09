# Hindi Word Embeddings with FastText

This project demonstrates how to train **Hindi word embeddings** using **Facebook’s FastText** library.  
It uses both **CBOW** and **Skip-gram** models to capture semantic and syntactic relationships between words.

---

## 📌 Project Overview

1. **Dataset**
   - Hindi text is taken from the **Hindi Text Short Summarization Corpus** (`train.csv` and `test.csv`).
   - Each CSV contains an `article` column with Hindi text.

2. **Preprocessing**
   - Load CSV files using **pandas**.
   - Combine all text into a single string.
   - Clean unwanted characters:
     - Remove punctuation marks (`।!:.,;()_’?"\/@`).
     - Remove English characters `[a-zA-Z]`.
     - Remove special Unicode characters (`\u200d`, `\xa0`).
   - Normalize extra spaces.
   - Save processed text into `.txt` files (`training.txt`, `testing.txt`).

3. **Training FastText Models**
   - **CBOW Model:**
     ```python
     model = fasttext.train_unsupervised('training.txt', model='cbow')
     model.save_model("model_cbow.bin")
     ```
   - **Skip-gram Model:**
     ```python
     model = fasttext.train_unsupervised('training.txt', model='skipgram')
     model.save_model("model_skipgram.bin")
     ```

4. **Word Similarity Queries**
   - Get nearest neighbors of a word:
     ```python
     model.get_nearest_neighbors('मोदी')
     ```

---

## 🛠️ Requirements

- Python 3.x
- Libraries:
  - `fasttext`
  - `numpy`
  - `pandas`
  - `re`

Install dependencies:
```bash
pip install fasttext numpy pandas
```
---

## 🚀 Usage
1. Preprocess the dataset:
   ```python
    preprocess('/kaggle/input/hindi-text-short-summarization-corpus/train.csv', 'training')
    preprocess('/kaggle/input/hindi-text-short-summarization-corpus/test.csv', 'testing')
   ```
3. Train embeddings:
   ```python
    model = fasttext.train_unsupervised('training.txt', model='cbow')
    model.save_model("model_cbow.bin")
    
    model = fasttext.train_unsupervised('training.txt', model='skipgram')
    model.save_model("model_skipgram.bin")
   ```
5. Query nearest neighbors:
   ```python
   model.get_nearest_neighbors('भारत')
   ```
   
📊 Example Output
- Nearest Neighbors of "मोदी":
   ```bash
   [(0.81, 'प्रधानमंत्री'), 
   (0.76, 'सरकार'), 
   (0.74, 'अमित'), 
   (0.73, 'योगी')]
   ```

---

## 📂 Project Structure
   ```python
    ├── training.txt            # Preprocessed training data
    ├── testing.txt             # Preprocessed testing data
    ├── model_cbow.bin          # Trained CBOW model
    ├── model_skipgram.bin      # Trained Skip-gram model
    ├── main.py                 # Preprocessing + training script
    ├── README.md               # Documentation
   ```
