
"""For ungarbling copy-pasted PDF text"""

import pdb
import re
import itertools

import nltk
import enchant

class NltkUngarbler:
    def __init__(self):
        self.spell_checker = enchant.Dict('en_US')

        # Cheating
        self.spell_checker.add('cloze')
        self.spell_checker.add('et')

    def is_word_probability(self, token):
        if len(token) == 0:
            return 0.0
        if token.isdigit():
            return 1.0
        if self.spell_checker.check(token) or self.spell_checker.check(token.lower()):
            return 1.0
        if len(token) == 1:
            return 0.05
        if token.isupper() or (token[:-1].isupper() and token[-1] == 's'):
            return 0.95
        if token[0].isupper() and token[1:].islower():
            return 0.9
        return 0.05

    def tokens_score(self, tokens):
        """Return a number or tuple representing likelihood of a set of tokens, to be used as a key in sorting (higher is better)."""
        badness = 0
        for tok in tokens:
            # 12 is 99th percentile based on a sample from wikipedia
            if len(tok) > 12 and not self.spell_checker.check(tok):
                badness += 1.5*len(tok)/8
        return -badness - len(tokens)

    def make_candidate_splits(self, token, max_subtokens=4):
        if max_subtokens <= 1:
            return [(token,)]

        results = []
        for i in range(0, len(token)):
            if self.is_word_probability(token[i:]) >= 0.9:
                if self.is_word_probability(token[:i]) >= 0.9:
                    results.append((token[:i], token[i:]))
                results.extend([x + (token[i:],) for x in self.make_candidate_splits(token[:i], max_subtokens=max_subtokens-1)])
        return results

    def split_into_words(self, token):
        if self.is_word_probability(token) >= 0.95:
            return (token,)
        if '-' in token:
            parts = token.partition('-')
            split1 = self.split_into_words(parts[0])
            split2 = self.split_into_words(parts[2])
            return split1[:-1] + (split1[-1] + parts[1] + split2[0],) + split2[1:]
        return max(self.make_candidate_splits(token, max_subtokens=max(3, 1+len(token)//4)), key=self.tokens_score)

    def detokenize(self, tokens):
        string = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize([tok.replace("''", '"') for tok in tokens])

        # Fix quotes
        string = string.replace(' "', '"').replace("``", ' "')

        # Fix periods (they should generally join to the left)
        string = re.sub(r' \.([^ ])', r'. \1', string)
        string = re.sub(r'([^ A-Za-z.])\.([^ ])', r'\1. \2', string)

        return string

    def ungarble(self, garbled_text):
        garbled_text = garbled_text.replace(u'“', '"').replace(u'”', '"')
        tokens = nltk.tokenize.treebank.TreebankWordTokenizer().tokenize(garbled_text)
        new_tokens = []
        for token in tokens:
            if not token.replace('-', '').isalnum():
                new_tokens.append(token)
                continue

            # TODO: Make better use of the probability rather than just thresholding it
            if self.spell_checker.check(token.replace('-', '')):
                new_tokens.append(token.replace('-', ''))
                continue

            is_word = (self.is_word_probability(token) >= 0.95)
            if is_word:
                new_tokens.append(token)
                continue

            # Try to split
            new_tokens.extend(self.split_into_words(token))
        return self.detokenize(new_tokens)

if __name__ == '__main__':
    test_texts = [
            "giventwocloze-phrasessuch  as  “Seinfeldoriginallyaired  on[MASK]”  and  “Seinfeldpremiered  on[MASK]” ",
"maybe of varying specificity:headquarterInmay beexpressed directly by open relations ",
"Using a conven-tional EL system, the first mentionAndrei Broder1can  be  easily  linked  toAndreiBroder ",
"we   pro-pose  a  novelTransformer-based  heterogeneousGNN   model ",
"vertices corre-spond tomentions of entitiesand edges toopenrelations(see Fig. 1)."  ]
    ug = NltkUngarbler()
    for text in test_texts:
        print(ug.ungarble(text))

    while True:
        text = input("Enter text: ")
        print("Ungarbled: ", ug.ungarble(text))

