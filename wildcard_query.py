class WildcardQuery:

    def __init__(self, vocabulary):
        self.vocabulary = set(vocabulary)

    def prefix_match(self, query):
        query = query.lower()

        matched_terms = []

        for term in self.vocabulary:
            if term.startswith(query):
                matched_terms.append(term)

        return sorted(matched_terms)

    def contains_match(self, query):
        query = query.lower()

        matched_terms = []

        for term in self.vocabulary:
            if query in term:
                matched_terms.append(term)

        return sorted(matched_terms)

    def suffix_match(self, query):
        query = query.lower()

        matched_terms = []

        for term in self.vocabulary:
            if term.endswith(query):
                matched_terms.append(term)

        return sorted(matched_terms)

    def automatic_match(self, query):
        query = query.lower().strip()

        prefix_terms = self.prefix_match(query)
        contains_terms = self.contains_match(query)
        suffix_terms = self.suffix_match(query)

        all_terms = sorted(
            set(prefix_terms) |
            set(contains_terms) |
            set(suffix_terms)
        )

        match_types = []

        if prefix_terms:
            match_types.append("Prefix")

        if contains_terms:
            match_types.append("Contains")

        if suffix_terms:
            match_types.append("Suffix")

        return {
            "query": query,
            "prefix_terms": prefix_terms,
            "contains_terms": contains_terms,
            "suffix_terms": suffix_terms,
            "all_terms": all_terms,
            "match_types": match_types
        }