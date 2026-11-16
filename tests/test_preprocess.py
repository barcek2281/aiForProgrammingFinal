"""Tests for the Kazakh tokeniser, stop words and the suffix stemmer.

The stemmer is written by hand in src/preprocess.py, so these tests pin down
the behaviour the models rely on: which endings are cut, which words are
protected, and where the stemmer is known to cut too much.
"""
import unittest

from src.preprocess import (MIN_TOKEN_LENGTH, STOP_WORDS, normalize_text, remove_stop_words,
                            stem_word, tokenize, tokenize_stems, tokenize_words, vocabulary_stats)

TITLE = "Аудандық бюджет туралы 2023 жылғы шешімге өзгерістер енгізу туралы!"


class TestNormalizeText(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(normalize_text("ҚАУЛЫ"), "қаулы")

    def test_yo_becomes_ye(self):
        self.assertEqual(normalize_text("ёлка"), "елка")

    def test_latin_i_becomes_kazakh_i_in_kazakh_text(self):
        # 'i' typed on a Latin keyboard must not create a second vocabulary entry
        self.assertEqual(normalize_text("шешiм"), "шешім")

    def test_latin_text_is_left_alone(self):
        self.assertEqual(normalize_text("Finance"), "finance")

    def test_accepts_non_string(self):
        self.assertEqual(normalize_text(2023), "2023")


class TestTokenize(unittest.TestCase):
    def test_numbers_and_punctuation_are_dropped(self):
        self.assertNotIn("2023", tokenize(TITLE))
        self.assertEqual(tokenize("бюджет, 2023!"), ["бюджет"])

    def test_keeps_kazakh_specific_letters(self):
        self.assertEqual(tokenize("Әділет ғылым қоры"), ["әділет", "ғылым", "қоры"])

    def test_short_tokens_are_dropped(self):
        self.assertEqual(MIN_TOKEN_LENGTH, 2)
        self.assertEqual(tokenize("а ба"), ["ба"])

    def test_empty_text(self):
        self.assertEqual(tokenize(""), [])

    def test_stop_words_are_removed(self):
        self.assertIn("туралы", tokenize(TITLE))          # tokenised
        self.assertNotIn("туралы", tokenize_words(TITLE))  # but not kept as a feature

    def test_remove_stop_words_keeps_content_words(self):
        self.assertEqual(remove_stop_words(["бюджет", "туралы", "және"]), ["бюджет"])


class TestStemWord(unittest.TestCase):
    def test_known_stems(self):
        # one stem for the many forms of the same word
        cases = {
            "ауданы": "аудан",
            "ауданының": "аудан",
            "аудандық": "аудан",
            "аудандарда": "аудан",
            "мәслихатының": "мәслихат",
            "мәслихаты": "мәслихат",
            "шешімге": "шешім",
            "өзгерістер": "өзгеріс",
            "заңның": "заң",
            "мемлекеттік": "мемлекет",
            "комитеті": "комитет",
        }
        for word, expected in cases.items():
            with self.subTest(word=word):
                self.assertEqual(stem_word(word), expected)

    def test_forms_of_one_word_collapse_to_one_stem(self):
        forms = ["аудан", "ауданы", "ауданының", "аудандық", "аудандарда"]
        self.assertEqual(len({stem_word(form) for form in forms}), 1)

    def test_softening_is_undone(self):
        # 'бұйрық' + 'ы' is written 'бұйрығы': the stem must get 'қ' back
        self.assertEqual(stem_word("бұйрығы"), "бұйрық")
        self.assertEqual(stem_word("бұйрығына"), "бұйрық")
        self.assertEqual(stem_word("салығы"), "салық")

    def test_protected_ending_is_not_cut(self):
        # 'қазақстан' is not 'қазақс' + 'тан' (ablative case)
        self.assertEqual(stem_word("қазақстан"), "қазақстан")

    def test_short_words_are_not_cut(self):
        # a one-letter ending needs a stem of at least 5 letters
        self.assertEqual(stem_word("қаулы"), "қаулы")
        self.assertEqual(stem_word("қаржы"), "қаржы")
        self.assertEqual(stem_word("заң"), "заң")

    def test_stem_is_never_longer_than_the_word(self):
        for word in ["ауданы", "қаулы", "қазақстан", "бюджеті", "мәслихатының"]:
            with self.subTest(word=word):
                self.assertLessEqual(len(stem_word(word)), len(word))

    def test_known_overstemming_is_documented(self):
        # The stemmer has no dictionary, so it sometimes cuts too much.
        # This is listed under 'Limitations' in the README; the test records it,
        # so a future change to the rules shows up here instead of silently.
        self.assertEqual(stem_word("облысы"), "облы")


class TestTokenizeStems(unittest.TestCase):
    def test_title_becomes_stems(self):
        self.assertEqual(tokenize_stems(TITLE),
                         ["аудан", "бюджет", "шешім", "өзгеріс", "енгіз"])

    def test_no_stop_word_survives_stemming(self):
        # stop words are checked again after stemming ('жылғы' -> 'жыл' is a stop word)
        for stem in tokenize_stems(TITLE):
            self.assertNotIn(stem, STOP_WORDS)

    def test_repeated_call_gives_the_same_result(self):
        # the module caches stems in a dict; the cache must not change the answer
        self.assertEqual(tokenize_stems(TITLE), tokenize_stems(TITLE))


class TestVocabularyStats(unittest.TestCase):
    def setUp(self):
        self.train = ["Ауданы туралы шешім", "Аудандық бюджет туралы қаулы"]
        self.test = ["Ауданының бюджеті туралы шешім"]
        self.rows = {row["representation"]: row for row in vocabulary_stats(self.train, self.test)}

    def test_both_representations_are_reported(self):
        self.assertEqual(set(self.rows), {"word forms", "stems"})

    def test_stemming_shrinks_the_vocabulary(self):
        self.assertLess(self.rows["stems"]["vocabulary"], self.rows["word forms"]["vocabulary"])

    def test_stemming_reduces_unseen_test_words(self):
        self.assertLessEqual(self.rows["stems"]["test_tokens_unseen_in_train_pct"],
                             self.rows["word forms"]["test_tokens_unseen_in_train_pct"])


if __name__ == "__main__":
    unittest.main()
