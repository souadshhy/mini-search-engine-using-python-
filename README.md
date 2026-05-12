# Mini Search Engine using Python

A mini search engine project developed in Python for the Information Retrieval Systems course.  
The system demonstrates the complete information retrieval pipeline including preprocessing, indexing, Boolean retrieval, phrase retrieval, ranked retrieval using TF-IDF, cosine similarity, and bonus search-engine features.

This project was implemented using Python and Streamlit.

---

# Project Features

## Core Features

- Document preprocessing
- Incidence Matrix
- Inverted Index
- Positional Index
- Boolean Retrieval
- Phrase Query Retrieval
- Ranked Retrieval using TF-IDF
- Cosine Similarity

## Bonus Features

- Wildcard Query Processing
- Spelling Correction
- Streamlit Web Interface

---

# Technologies Used

- Python
- Streamlit
- NLTK
- Pandas

Required libraries are listed in `requirements.txt`.

---

# Project Structure

```text
miniSearchEngine/
│
├── documents/
│   ├── doc1.txt
│   ├── doc2.txt
│   └── ...
│
├── main.py
├── preprocessing.py
├── incidence_matrix.py
├── inverted_index.py
├── positional_index.py
├── ranking.py
├── spelling_correction.py
├── wildcard_query.py
├── utils.py
├── requirements.txt
├── README.md
└── report.pdf
```

---

# Dataset

The dataset contains 20 text documents related to:

- Information Retrieval
- Artificial Intelligence
- Machine Learning
- Databases
- Cybersecurity
- Cloud Computing
- Python
- Streamlit

The documents were designed with overlapping technical vocabulary to properly demonstrate:

- Boolean Retrieval
- Phrase Queries
- TF-IDF Ranking
- Wildcard Search
- Spelling Correction

---

# Information Retrieval Techniques Implemented

## 1. Document Preprocessing

The preprocessing pipeline includes:

- Case Folding
- Tokenization
- Punctuation Removal
- Stop-word Removal
- Stemming using Porter Stemmer

Example preprocessing flow:

```text
Original:
Machine Learning improves search systems.

Processed:
['machin', 'learn', 'improv', 'search', 'system']
```

---

## 2. Incidence Matrix

The project implements a binary term-document incidence matrix.

- Rows represent terms
- Columns represent documents
- Values:
  - 1 → term exists in document
  - 0 → term does not exist

Supported queries:

- Single-term search
- Boolean retrieval

Example query:

```text
machine AND database
```

---

## 3. Inverted Index

The inverted index stores:

- Terms
- Document frequency
- Posting lists

Supported operations:

- Single-term retrieval
- AND queries
- OR queries
- NOT queries

Example:

```text
machine AND NOT database
```

---

## 4. Positional Index

The positional index stores:

- Document IDs
- Exact token positions

This allows phrase searching.

Example phrase query:

```text
"machine learning"
```

The system only returns documents where the terms appear consecutively and in order.

---

## 5. Ranked Retrieval

The project implements ranked retrieval using:

- TF (Term Frequency)
- IDF (Inverse Document Frequency)
- TF-IDF weighting
- Cosine Similarity

Formula used:

```text
TF-IDF = tf(t,d) × log(N / df(t))
```

Cosine similarity:

```text
Cosine(q,d) = (q · d) / (|q| × |d|)
```

Returned results include:

- Document ID
- Similarity Score
- Ranking Order

---

# Bonus Features

## Wildcard Query Processing

Supports:

- Prefix matching
- Suffix matching
- Contains matching

Example:

```text
mach
```

Can retrieve:

```text
machine
machining
machinery
```

---

## Spelling Correction

The project implements spelling correction using Edit Distance (Levenshtein Distance).

Example:

```text
machne
```

Automatically corrected to:

```text
machine
```

---

# User Interface

The project uses a Streamlit-based graphical interface.

Users can:

- Load documents
- Preprocess text
- View indexes
- Run Boolean queries
- Run phrase queries
- Perform ranked retrieval
- Use wildcard search
- View similarity scores

---

# Installation Guide

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd miniSearchEngine
```

---

## 2. Create Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the Streamlit application:

```bash
streamlit run main.py
```

The browser will automatically open the application.

---

# Example Queries

## Inverted Index

```text
machine
database
retrieval
```

---

## Boolean Retrieval

```text
machine AND database

machine OR cloud

machine AND NOT database
```

---

## Phrase Query

```text
machine learning

information retrieval
```

---

## Ranked Retrieval

```text
artificial intelligence

search engine optimization

cloud computing systems
```

---

## Wildcard Query

```text
mach
learn
ing
```

---

# Important Files

| File | Description |
|---|---|
| `main.py` | Streamlit interface |
| `preprocessing.py` | Text preprocessing pipeline |
| `incidence_matrix.py` | Binary term-document matrix |
| `inverted_index.py` | Inverted index and Boolean retrieval |
| `positional_index.py` | Phrase query processing |
| `ranking.py` | TF-IDF and cosine similarity |
| `spelling_correction.py` | Edit-distance spelling correction |
| `wildcard_query.py` | Wildcard and partial matching |
| `utils.py` | Corpus loading utilities |

---

# Limitations

- Boolean retrieval currently supports linear expressions without parentheses parsing.
- Ranked retrieval uses a basic TF-IDF implementation without normalization optimizations.
- The system is designed for educational purposes and small-to-medium datasets.
- Wildcard processing is vocabulary-based and not implemented using permuterm indexes or tries.
- Phrase queries require exact consecutive ordering.

---

# Learning Outcomes

Through this project, the following concepts were implemented and understood:

- Information Retrieval pipelines
- Document preprocessing
- Boolean retrieval models
- Positional indexing
- Ranked retrieval
- TF-IDF weighting
- Cosine similarity
- Search engine architecture
- Query processing

---

# References

General concepts and theory were studied using:

- Information Retrieval course materials
- NLTK documentation
- Streamlit documentation
- Wikipedia (TF-IDF and Information Retrieval concepts)
- GeeksforGeeks
- W3Schools

The project logic and implementation were written specifically for this assignment.

---

# Academic Integrity

This project was developed for educational purposes as part of the Information Retrieval Systems course.

External resources were used only for learning and reference purposes. Core logic and implementation were written manually to reflect course concepts.