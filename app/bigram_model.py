"""Bigram language model, adapted from the Module 2 word-sampling notebook."""

import random
import re
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self, corpus, frequency_threshold=None):
        text = " ".join(corpus)
        self.vocab, self.bigram_probs = self._analyze_bigrams(text, frequency_threshold)

    @staticmethod
    def _tokenize(text, frequency_threshold=None):
        """Lowercase the text and split it into words; optionally drop rare words."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not frequency_threshold:
            return tokens
        counts = Counter(tokens)
        return [t for t in tokens if counts[t] >= frequency_threshold]

    def _analyze_bigrams(self, text, frequency_threshold=None):
        """Compute P(w2 | w1) = count(w1, w2) / count(w1)."""
        words = self._tokenize(text, frequency_threshold)
        bigram_counts = Counter(zip(words[:-1], words[1:]))
        unigram_counts = Counter(words)
        probs = defaultdict(dict)
        for (w1, w2), count in bigram_counts.items():
            probs[w1][w2] = count / unigram_counts[w1]
        return list(unigram_counts.keys()), probs

    def generate_text(self, start_word, length=10):
        """Sample up to `length` words, starting from `start_word`."""
        current = start_word.lower()
        words = [current]
        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current)
            if not next_words:  # no known continuation: stop early
                break
            current = random.choices(
                list(next_words.keys()), weights=list(next_words.values())
            )[0]
            words.append(current)
        return " ".join(words)
