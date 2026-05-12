from collections import defaultdict
import re


class InvertedIndex:
    def __init__(self, documents):
        self.documents = documents
        self.index = defaultdict(set)
        self.df = {}

    def build_index(self):
        for doc_id, tokens in self.documents.items():
            for token in tokens:
                if token:
                    self.index[token].add(doc_id)

        for term in self.index:
            self.df[term] = len(self.index[term])

    def search(self, term):
        return sorted(self.index.get(term, set()))

    def not_query(self, term):
        all_docs = set(self.documents.keys())
        return sorted(all_docs - self.index.get(term, set()))

    def get_posting_list(self, term):
        return {
            "term": term,
            "document_frequency": self.df.get(term, 0),
            "documents": sorted(self.index.get(term, set()))
        }

    # ------------------------------------------------
    # Boolean Query Processing
    # Supports:
    # machine and database
    # machine or database
    # machine and not database
    # ------------------------------------------------
    def boolean_query(self, processed_query_tokens):
        """
        processed_query_tokens example:
        ['machin', 'AND', 'databas']
        ['machin', 'AND', 'NOT', 'databas']
        """

        if not processed_query_tokens:
            return []

        all_docs = set(self.documents.keys())

        def get_docs(term):
            return self.index.get(term, set())

        result = None
        current_operator = None
        negate_next = False

        for token in processed_query_tokens:

            if token == "AND":
                current_operator = "AND"

            elif token == "OR":
                current_operator = "OR"

            elif token == "NOT":
                negate_next = True

            else:
                term_docs = get_docs(token)

                if negate_next:
                    term_docs = all_docs - term_docs
                    negate_next = False

                if result is None:
                    result = term_docs.copy()

                elif current_operator == "AND":
                    result = result & term_docs

                elif current_operator == "OR":
                    result = result | term_docs

                else:
                    # Default behavior if user writes terms without operator
                    result = result & term_docs

        return sorted(result) if result is not None else []

    def preprocess_boolean_query(self, query, preprocessor):
        """
        Keeps AND / OR / NOT as operators.
        Preprocesses only the normal terms.
        """

        raw_tokens = query.split()

        processed_tokens = []

        for token in raw_tokens:
            upper_token = token.upper()

            if upper_token in ["AND", "OR", "NOT"]:
                processed_tokens.append(upper_token)

            else:
                cleaned = preprocessor.preprocess(token)

                if cleaned:
                    processed_tokens.append(cleaned[0])

        return processed_tokens