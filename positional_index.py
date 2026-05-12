from collections import defaultdict

class PositionalIndex:
    def __init__(self, documents):
        self.documents = documents
        self.index = defaultdict(lambda: defaultdict(list)) # {key:{key:[empty list]}}


    # Build positional index
    def build_index(self):
        for doc_id, tokens in self.documents.items():
            for position, token in enumerate(tokens):
                if token:
                    self.index[token][doc_id].append(position)

    # Get posting list
    def get_posting_list(self, term):
        return {
            "term": term,
            "documents": {
                doc_id: positions
                for doc_id, positions in self.index.get(term, {}).items()
            }
        }

    # Phrase query 
    def phrase_query(self, phrase_tokens):
        if not phrase_tokens:
            return []

        # Step 1: get candidate docs (must contain first term)
        first_term = phrase_tokens[0]
        candidate_docs = set(self.index.get(first_term, {}).keys())

        for term in phrase_tokens[1:]:
            candidate_docs &= set(self.index.get(term, {}).keys())

        results = []

        # !! Step 2: check positions
        for doc_id in candidate_docs:
            positions_lists = [
                self.index[term][doc_id]
                for term in phrase_tokens
            ]

            # Check consecutive positions
            for pos in positions_lists[0]:
                match = True
                for i in range(1, len(positions_lists)):
                    if (pos + i) not in positions_lists[i]:
                        match = False
                        break

                if match:
                    results.append(doc_id)
                    break

        return sorted(results)