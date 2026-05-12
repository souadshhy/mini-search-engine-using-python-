from collections import defaultdict


class IncidenceMatrix:
    def __init__(self, documents):
        """
        documents: dict -> {doc_id: [tokens]}
        """
        self.documents = documents
        self.vocabulary = set()
        self.matrix = defaultdict(dict)

    def build_vocabulary(self):
        for _, tokens in self.documents.items():
            for token in tokens:
                self.vocabulary.add(token)

        self.vocabulary = sorted(list(self.vocabulary))

    def build_matrix(self):
        for term in self.vocabulary:
            for doc_id, tokens in self.documents.items():
                if term in tokens:
                    self.matrix[term][doc_id] = 1
                else:
                    self.matrix[term][doc_id] = 0

    def get_docs_for_term(self, term):
        """
        Returns all documents where term exists
        """
        matching_docs = set()

        for doc_id in self.documents:
            if self.matrix.get(term, {}).get(doc_id, 0) == 1:
                matching_docs.add(doc_id)

        return matching_docs

    def boolean_query(self, processed_query_tokens):
        """
        Supports:
        machin AND databas
        machin OR databas
        machin AND NOT databas
        """

        if not processed_query_tokens:
            return []

        all_docs = set(self.documents.keys())

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
                term_docs = self.get_docs_for_term(token)

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
                    result = result & term_docs

        return sorted(result) if result is not None else []

    def display_matrix(self):
        doc_ids = sorted(self.documents.keys())

        print("\nINCIDENCE MATRIX\n")

        print("Term".ljust(15), end="")

        for doc_id in doc_ids:
            print(f"D{doc_id}".ljust(6), end="")

        print()
        print("-" * 40)

        for term in self.vocabulary:
            print(term.ljust(15), end="")

            for doc_id in doc_ids:
                print(str(self.matrix[term][doc_id]).ljust(6), end="")

            print()

    def single_term_query(self, term):
        """
        Simple one-term lookup using incidence matrix.
        """
        return sorted(self.get_docs_for_term(term))