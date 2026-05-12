import math
from collections import defaultdict


class RankedRetrieval:
    def __init__(self, documents):
        self.documents = documents

        self.tf = defaultdict(dict)
        self.df = {}
        self.idf = {}
        self.query_tfidf = {}
        self.document_weights = defaultdict(dict)

        self.vocabulary = set()

    # Build vocabulary
    def build_vocabulary(self):
        for tokens in self.documents.values():
            for token in tokens:
                self.vocabulary.add(token)

    # Compute document TF using 1 + log(tf)
    def compute_tf(self):
        for doc_id, tokens in self.documents.items():

            term_counts = {}

            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1

            for term, count in term_counts.items():
                self.tf[doc_id][term] = 1 + math.log10(count)

    # Compute document frequency
    def compute_df(self):
        for term in self.vocabulary:

            count = 0

            for tokens in self.documents.values():
                if term in tokens:
                    count += 1

            self.df[term] = count

    # Compute IDF using log10(N / df)
    def compute_idf(self):
        N = len(self.documents)

        for term, df_value in self.df.items():
            self.idf[term] = math.log10(N / df_value)

    # Document vector     
    # document weight = document TF only

    def compute_document_weights(self):
        for doc_id, terms in self.tf.items():

            for term, tf_value in terms.items():
                self.document_weights[doc_id][term] = tf_value

    # Build all
    def build(self):
        self.build_vocabulary()
        self.compute_tf()
        self.compute_df()
        self.compute_idf()
        self.compute_document_weights()

    # Query vector style
    # query weight = query TF * IDF
    def query_vector(self, query_tokens):
        query_tf = {}

        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1

        query_vector = {}

        for term, count in query_tf.items():

            if term in self.idf:
                tf = 1 + math.log10(count)
                idf = self.idf[term]

                query_vector[term] = tf * idf

        return query_vector

    # Cosine similarity
    def cosine_similarity(self, query_vector, doc_vector):

        dot_product = 0

        for term, q_weight in query_vector.items():
            d_weight = doc_vector.get(term, 0)
            dot_product += q_weight * d_weight

        query_magnitude = math.sqrt(
            sum(weight ** 2 for weight in query_vector.values())
        )

        doc_magnitude = math.sqrt(
            sum(weight ** 2 for weight in doc_vector.values())
        )

        if query_magnitude == 0 or doc_magnitude == 0:
            return 0

        return dot_product / (query_magnitude * doc_magnitude)

    # Ranked search
    def search(self, query_tokens):

        query_vec = self.query_vector(query_tokens)

        scores = []

        for doc_id, doc_vector in self.document_weights.items():

            similarity = self.cosine_similarity(
                query_vec,
                doc_vector
            )

            scores.append((doc_id, similarity))

        scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return scores