"""Fixed style representation. No learned content vocabulary or bibliographic inputs."""

import re
from collections import Counter

import numpy as np

LEXICONS = {
    "promotion": "novel innovative groundbreaking revolutionary unprecedented remarkable exceptional transformative promising powerful superior unique breakthrough outstanding cutting-edge state-of-the-art robust efficient effective important significant crucial exciting compelling pioneering versatile dramatic dramatically substantially substantially".split(),
    "hedging": "may might could perhaps possibly probably likely unlikely suggest suggests suggested suggesting appears appear seems seem potential potentially approximately relatively generally usually".split(),
    "certainty": "clearly obviously undoubtedly certainly demonstrate demonstrates demonstrated establish establishes established confirm confirms confirmed prove proves proven always never unequivocal definitive".split(),
    "limitations": "limitation limitations limited however although despite nevertheless nonetheless caveat caution uncertain uncertainty unclear inconsistent failure failed fails drawback drawbacks".split(),
    "evidence": "measured observed tested evaluated compared estimated quantified replicated validated analyzed examined assessed calculated recorded measured experiments experiment results data analysis measurements".split(),
    "future": "future further will shall anticipate anticipated expect expected prospective eventually".split(),
    "negation": "not no neither nor without cannot unable".split(),
    "first_person": "we our ours us i my".split(),
    "causal": "because therefore thus hence consequently accordingly causes cause caused leads lead".split(),
    "contrast": "but whereas while unlike conversely alternatively instead yet".split(),
}
LEXICONS = {key: frozenset(words) for key, words in LEXICONS.items()}
FUNCTION_WORDS = "the a an of to in for with by from on at as is are was were be been being and or that which this these it its their".split()
TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-’'][A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[A-Za-z]+(?:[-’'][A-Za-z]+)*")
STRUCTURAL = [
    "log_words",
    "log_sentences",
    "words_per_sentence",
    "sentence_length_sd",
    "mean_word_length",
    "long_word_rate",
    "type_token_ratio",
    "digit_rate",
    "comma_rate",
    "semicolon_rate",
    "colon_rate",
    "parenthesis_rate",
    "question_rate",
    "hyphen_rate",
    "uppercase_rate",
    "passive_proxy",
    "past_tense_proxy",
    "adverb_proxy",
]
FEATURE_NAMES = (
    STRUCTURAL + [f"{key}_rate" for key in LEXICONS] + [f"function_{word}" for word in FUNCTION_WORDS]
)


def tokenize(text):
    result = []
    for match in TOKEN_RE.finditer(text):
        word = match.group().lower()
        categories = [k for k, vocabulary in LEXICONS.items() if word in vocabulary]
        category = (
            categories[0]
            if categories
            else (
                "function"
                if word in FUNCTION_WORDS
                else "content"
                if WORD_RE.fullmatch(word)
                else "number"
                if word[0].isdigit()
                else "punctuation"
            )
        )
        result.append(
            {
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "category": category,
                "categories": categories,
            }
        )
    return result


def extract(text):
    raw_words = WORD_RE.findall(text)
    words = [w.lower() for w in raw_words]
    n = max(len(words), 1)
    counts = Counter(words)
    sentences = [WORD_RE.findall(s) for s in re.split(r"(?<=[.!?])\s+|[\r\n]+", text)]
    lengths = [len(s) for s in sentences if s]
    lengths = lengths or [1]
    values = [
        np.log1p(n),
        np.log1p(len(lengths)),
        np.mean(lengths),
        np.std(lengths),
        sum(map(len, words)) / n,
        sum(len(w) >= 7 for w in words) / n,
        len(counts) / n,
        len(re.findall(r"\b\d+(?:\.\d+)?\b", text)) / n,
        text.count(",") / n,
        text.count(";") / n,
        text.count(":") / n,
        text.count("(") / n,
        text.count("?") / n,
        sum("-" in w for w in words) / n,
        sum(w.isupper() and len(w) > 1 for w in raw_words) / n,
        len(re.findall(r"\b(?:is|are|was|were|be|been)\s+(?:\w+ly\s+)?\w+ed\b", text.lower())) / n,
        sum(w.endswith("ed") for w in words) / n,
        sum(w.endswith("ly") for w in words) / n,
    ]
    values += [sum(counts[w] for w in lexicon) / n for lexicon in LEXICONS.values()]
    values += [counts[w] / n for w in FUNCTION_WORDS]
    return np.asarray(values, dtype=float)


def feature_label(name):
    if name.startswith("function_"):
        return f"Function word “{name[9:]}”"
    return name.replace("_", " ").capitalize()


def matrix(texts):
    return np.vstack([extract(text) for text in texts])
