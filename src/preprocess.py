"""Kazakh text preprocessing: normalisation, tokenisation, stop words, stemming.

Kazakh is agglutinative: one noun appears in dozens of forms, for example
'аудан', 'ауданы', 'ауданының', 'аудандық', 'аудандарда'. Three ways to deal
with this are compared in the project:

    1. word forms      - tokens are used as they are;
    2. suffix stemming - frequent endings are cut off (stem_word below);
    3. char n-grams    - no tokens at all, pieces of 2-5 characters
                         (built by scikit-learn, see models.py).
"""
import re
from collections import Counter

# Kazakh Cyrillic has 9 extra letters: ә ғ қ ң ө ұ ү һ і
TOKEN_PATTERN = re.compile(r"[a-zа-яёәғқңөұүһі]+")
MIN_TOKEN_LENGTH = 2

# Function words and date words that carry no information about the topic.
STOP_WORDS = frozenset("""
және мен бен пен да де та те не немесе әрі бірақ сондай ақ
туралы үшін бойынша арқылы жөнінде жөніндегі қатысты сәйкес байланысты дейін кейін бастап
бұл осы сол ол оның олардың өз өзге басқа кейбір барлық әр бір
деп болып табылады табылатын бар жоқ емес
жыл жылы жылғы жылдың жылдан жылдарға жылдары жылдарда
қаңтар ақпан наурыз сәуір мамыр маусым шілде тамыз қыркүйек қазан қараша желтоқсан
""".split())

# --------------------------------------------------------------------------
# Suffix stemmer
# --------------------------------------------------------------------------
# A Kazakh noun is built as  stem + [word-forming] + [plural] + [possessive] + [case],
# so endings are removed from the right in the opposite order, one per stage.
VOWELS = set("аәеиоөұүуыіэюя")
VOICELESS = set("кқпстфхцчшщһбвгд")      # consonants followed by the т-/қ-/к- variants
SONORANT = set("лмнңжзрйу")              # consonants followed by the д-/ғ-/г- variants


def _after_vowel(ch): return ch in VOWELS
def _after_consonant(ch): return ch not in VOWELS
def _after_voiceless(ch): return ch in VOICELESS
def _after_sonorant(ch): return ch in SONORANT
def _after_vowel_or_sonorant(ch): return ch in VOWELS or ch in SONORANT
def _after_vowel_or_nasal(ch): return ch in VOWELS or ch in "мнң"
def _after_vowel_or_glide(ch): return ch in VOWELS or ch in "рйу"


def _rules(suffixes: str, allowed_after) -> dict:
    return {suffix: allowed_after for suffix in suffixes.split()}


# Vowel harmony and consonant assimilation decide which variant may follow which
# letter. Checking it avoids false cuts such as 'мәслиха-ты' or 'бюдже-ті'.
POSSESSIVE_CASE = {                       # 3rd person possessive + case in one ending
    **_rules("ына іне ында інде ынан інен ын ін", _after_consonant),
    **_rules("сына сіне сында сінде сынан сінен сын сін ндағы ндегі", _after_vowel),
}
CASE = {
    **POSSESSIVE_CASE,
    **_rules("ның нің", _after_vowel_or_nasal),
    **_rules("дың дің", _after_sonorant),
    **_rules("тың тің та те тан тен қа ке тағы тегі қы кі", _after_voiceless),
    **_rules("ға ге да де дан ден дағы дегі", _after_vowel_or_sonorant),
    **_rules("ғы гі", _after_sonorant),      # 'жылғы'; after a vowel it is a softened stem: 'салығы'
}
POSSESSIVE = {
    **_rules("сы сі", _after_vowel),
    **_rules("ы і", _after_consonant),
}
PLURAL = {
    **_rules("лар лер", _after_vowel_or_glide),
    **_rules("дар дер", _after_sonorant),
    **_rules("тар тер", _after_voiceless),
}
WORD_FORMING = {
    **_rules("лық лік", _after_vowel_or_glide),
    **_rules("дық дік ған ген", _after_sonorant),
    **_rules("тық тік қан кен", _after_voiceless),
    **_rules("атын етін у", _after_consonant),
}
STAGES = [CASE, POSSESSIVE, PLURAL, WORD_FORMING]

MIN_STEM_LENGTH = 3            # never cut a word shorter than this
MIN_STEM_ONE_LETTER = 5        # one-letter endings are risky: 'қаулы' must stay 'қаулы'
PROTECTED_ENDINGS = ("стан",)  # 'қазақстан' is not 'қазақс' + 'тан'
SOFTENED = {"ғ": "қ", "г": "к", "б": "п"}   # 'бұйрық' + 'ы' is written 'бұйрығы'


def normalize_text(text: str) -> str:
    """Lower-case and unify symbols that look the same but are different characters."""
    text = str(text).lower()
    text = text.replace("ё", "е")
    # Latin look-alikes that appear in typed Kazakh text
    text = text.replace("i", "і") if re.search(r"[а-яәғқңөұүһі]", text) else text
    return text


def tokenize(text: str) -> list:
    """Split a title into lower-case words (letters only: numbers and symbols are dropped)."""
    return [token for token in TOKEN_PATTERN.findall(normalize_text(text))
            if len(token) >= MIN_TOKEN_LENGTH]


def remove_stop_words(tokens: list) -> list:
    return [token for token in tokens if token not in STOP_WORDS]


def strip_suffix(word: str, rules: dict) -> tuple:
    """Remove the longest allowed suffix of one stage. Returns (word, removed suffix or None)."""
    for suffix in sorted(rules, key=len, reverse=True):
        stem_length = len(word) - len(suffix)
        min_stem = MIN_STEM_ONE_LETTER if len(suffix) == 1 else MIN_STEM_LENGTH
        if word.endswith(suffix) and stem_length >= min_stem and rules[suffix](word[stem_length - 1]):
            return word[:stem_length], suffix
    return word, None


def stem_word(word: str) -> str:
    """Cut case, possessive, plural and word-forming endings: 'мәслихатының' -> 'мәслихат'."""
    if word.endswith(PROTECTED_ENDINGS):
        return word
    had_possessive = False
    for rules in STAGES:
        if rules is POSSESSIVE and had_possessive:
            continue                      # 'ына', 'сын' ... already contained the possessive
        word, suffix = strip_suffix(word, rules)
        if suffix in POSSESSIVE_CASE or (rules is POSSESSIVE and suffix):
            had_possessive = True
            if word[-1] in SOFTENED:      # undo the softening: 'бұйрығ' -> 'бұйрық'
                word = word[:-1] + SOFTENED[word[-1]]
    return word


# --- tokenisers passed to TfidfVectorizer (must be plain functions to be saved) ---
def tokenize_words(text: str) -> list:
    """Representation 1: word forms without stop words."""
    return remove_stop_words(tokenize(text))


_stem_cache = {}


def tokenize_stems(text: str) -> list:
    """Representation 2: stems without stop words (checked before and after stemming)."""
    stems = []
    for token in tokenize_words(text):
        stem = _stem_cache.get(token)
        if stem is None:
            stem = _stem_cache[token] = stem_word(token)
        if stem not in STOP_WORDS:
            stems.append(stem)
    return stems


def vocabulary_stats(train_texts, test_texts) -> list:
    """Compare word forms and stems: vocabulary size, rare words, unseen test words."""
    rows = []
    for name, tokenizer in [("word forms", tokenize_words), ("stems", tokenize_stems)]:
        train_counts = Counter()
        for text in train_texts:
            train_counts.update(tokenizer(text))
        test_tokens = [token for text in test_texts for token in tokenizer(text)]
        unseen = sum(1 for token in test_tokens if token not in train_counts)
        rows.append({
            "representation": name,
            "vocabulary": len(train_counts),
            "words_seen_once": sum(1 for n in train_counts.values() if n == 1),
            "test_tokens": len(test_tokens),
            "test_tokens_unseen_in_train_pct": round(100 * unseen / max(len(test_tokens), 1), 2),
        })
    return rows
