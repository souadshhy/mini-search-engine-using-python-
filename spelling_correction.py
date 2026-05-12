class SpellingCorrector:
    def __init__(self, vocabulary):
        self.vocabulary = set(vocabulary)

    def edit_distance(self, word1, word2):
        rows = len(word1) + 1
        cols = len(word2) + 1

        dp = [[0 for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            dp[i][0] = i

        for j in range(cols):
            dp[0][j] = j

        for i in range(1, rows):
            for j in range(1, cols):
                if word1[i - 1] == word2[j - 1]:
                    cost = 0
                else:
                    cost = 1

                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + cost  # substitution
                )

        return dp[-1][-1]

    def suggest_word(self, word, max_distance=2):
        if word in self.vocabulary:
            return word

        best_word = None
        best_distance = float("inf")

        for vocab_word in self.vocabulary:
            distance = self.edit_distance(word, vocab_word)

            if distance < best_distance:
                best_distance = distance
                best_word = vocab_word

        if best_distance <= max_distance:
            return best_word

        return word

    def correct_query(self, tokens, max_distance=2):
        corrected_tokens = []

        for token in tokens:
            corrected_word = self.suggest_word(token, max_distance)
            corrected_tokens.append(corrected_word)

        return corrected_tokens