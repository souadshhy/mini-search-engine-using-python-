import os
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Preprocessor:
    def __init__(self, language="english", use_stemming=True):
        self.language = language
        self.use_stemming = use_stemming

        self._ensure_nltk_resources()

        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()

    # Setup
    def _ensure_nltk_resources(self):

        resources = [
            ("corpora/stopwords", "stopwords"),
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab")
        ]

        for path, resource in resources:

            try:
                nltk.data.find(path)

            except LookupError:
                nltk.download(resource)

    # Preprocessing steps
    def case_folding(self, text):
        return text.lower()
    
    def tokenize(self, text):
        return word_tokenize(text)

    def remove_punctuation(self, tokens):
        return [
            re.sub(r"[^\w\s]", "", t)
            for t in tokens
            if re.sub(r"[^\w\s]", "", t)
        ]

    def remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stop_words]

    def stem(self, tokens):
        return [self.stemmer.stem(t) for t in tokens]

    # Pipeline
    def preprocess(self, text):
        text = self.case_folding(text)
        tokens = self.tokenize(text)
        tokens = self.remove_punctuation(tokens)
        tokens = self.remove_stopwords(tokens)

        if self.use_stemming:
            tokens = self.stem(tokens)

        return tokens

    # Document processing

    def natural_sort_key(self, filename):
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", filename)
        ]


    def preprocess_documents(self, folder_path):
        documents = {}

        files = sorted(
            os.listdir(folder_path),
            key=self.natural_sort_key
        )

        for doc_id, filename in enumerate(files, start=1):
            if filename.endswith(".txt"):
                path = os.path.join(folder_path, filename)

                with open(path, "r", encoding="utf-8") as file:
                    text = file.read()

                tokens = self.preprocess(text)
                documents[doc_id] = tokens

        return documents

    # BEFORE/AFTER visualization 
    def show_example(self, folder_path, file_name):
            path = os.path.join(folder_path, file_name)
            if not os.path.exists(path):
                print(f'File {file_name} not found.')
                return
            
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()

            processed = self.preprocess(text)

            result = {
                "filename": file_name,
                "before": text,
                "after": processed
            }
            return result
