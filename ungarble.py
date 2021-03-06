
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

    def is_word_probability(self, token):
        if len(token) == 0:
            return 0.0
        if token.isdigit():
            return 1.0
        if self.spell_checker.check(token) or self.spell_checker.check(token.lower()):
            return 1.0
        if len(token) == 1:
            return 0.05
        if token.isupper():
            return 0.95
        if token[0].isupper() and token[1:].islower():
            return 0.9
        return 0.05

    def split_into_words(self, token):
        if self.is_word_probability(token) >= 0.95:
            return (token,)
        if '-' in token:
            parts = token.partition('-')
            split1 = self.split_into_words(parts[0])
            split2 = self.split_into_words(parts[2])
            return split1[:-1] + (split1[-1] + parts[1] + split2[0],) + split2[1:]
        best_split = -1
        best_num_splits = 999
        other_split = ()
        for i in range(0, len(token)):
            if self.is_word_probability(token[i:]) >= 0.9:
                if self.is_word_probability(token[:i]) >= 0.9:
                    best_split = i
                    other_split = (token[:i],)
                    break
                else:
                    split = self.split_into_words(token[:i])
                    if len(split) > 1 and len(split) < best_num_splits:
                        best_split = i
                        best_num_splits = len(split)
                        other_split = split
        if best_split == -1 or best_split == (len(token)+1):
            return (token,)
        return other_split + (token[best_split:],)

    def ungarble(self, garbled_text):
        garbled_text = garbled_text.replace(u'“', '"').replace(u'”', '"')
        tokens = nltk.tokenize.treebank.TreebankWordTokenizer().tokenize(garbled_text)
        new_tokens = []
        for token in tokens:
            if token == '``' or token == "''":
                new_tokens.append('"')
                continue
            # TODO: Make better use of the probability rather than just thresholding it
            if not token.replace('-', '').isalnum():
                new_tokens.append(token)
                continue

            if (self.is_word_probability(token.replace('-', '')) >= 0.95):
                new_tokens.append(token.replace('-', ''))
                continue

            is_word = (self.is_word_probability(token) >= 0.95)
            if is_word:
                new_tokens.append(token)
                continue


            # Try to split
            new_tokens.extend(self.split_into_words(token))
        return nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(new_tokens)

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

