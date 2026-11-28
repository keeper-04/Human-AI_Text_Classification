import re
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been
being it its this that these those i you he she we they them his her our your their not no so than
then there here which who whom what when where why how do does did have has had will would can could
should may might must just also very more most some any all each such only own same too""".split())
WORD_RE = re.compile(r"[A-Za-z']+")
SENT_RE = re.compile(r"[.!?]+(?:\s+|$)")

FEATURE_INFO = {
    "avg_word_len": ("length", "Mean letters per word"),
    "stopword_ratio": ("lexical", "Stopwords / words"),
    "avg_sent_len": ("sentence", "Mean words per sentence"),
    "sent_len_std": ("sentence", "Std. dev. of words per sentence (burstiness)"),
    "comma_per_sent": ("punct", "Commas per sentence"),
    "digit_ratio": ("char", "Digits / characters"),
    "newline_ratio": ("format", "Newline characters per 1000 characters"),
    "avg_par_len": ("paragraph", "Mean words per paragraph (paragraphs = non-empty lines split on newlines)"),
    "contraction_ratio": ("style", "Words containing an apostrophe / words"),
    "pos_conj": ("pos", "Share of coordinating conjunctions (and, but, or; Penn tag CC) among POS-tagged "
                        "tokens (first 400 words, NLTK perceptron tagger)"),
}
FEATURE_NAMES = list(FEATURE_INFO)


def _div(a, b):
    return a / b if b else 0.0


def _pos_tag(words):
    import nltk
    try:
        return nltk.pos_tag(words)
    except LookupError:  
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        return nltk.pos_tag(words)


def extract(text: str) -> list:
    """The 10 features of one text, in FEATURE_NAMES order."""
    text = str(text)
    n_chars = len(text)
    words = WORD_RE.findall(text)
    lw = [w.lower() for w in words]
    n_words = len(words)
    sents = [s for s in SENT_RE.split(text) if s.strip()]
    n_sent = max(len(sents), 1)
    sent_lens = [len(WORD_RE.findall(s)) for s in sents] or [0]
    pars = [p for p in re.split(r"\n+", text) if p.strip()]
    par_lens = [len(WORD_RE.findall(p)) for p in pars] or [0]
    tags = Counter(t for _, t in _pos_tag(words[:400])) if words else Counter()
    return [
        _div(sum(len(w) for w in words), n_words),                    
        _div(sum(w in STOP for w in lw), n_words),                    
        float(np.mean(sent_lens)),                                    
        float(np.std(sent_lens)),                                     
        text.count(",") / n_sent,                                     
        _div(sum(c.isdigit() for c in text), n_chars),                
        text.count("\n") * 1000.0 / n_chars if n_chars else 0.0,      
        float(np.mean(par_lens)),                                     
        _div(sum("'" in w for w in words), n_words),                  
        sum(c for t, c in tags.items() if t.startswith("CC")) / max(sum(tags.values()), 1),  
    ]


class FeatureLR(BaseEstimator, ClassifierMixin):
    """Raw texts in -> labels out.

    robust=True (default): log(1+x) on every feature, standardise, clip z-scores to +-5.
    features: optional subset of FEATURE_NAMES (used for the ablations).
    """

    def __init__(self, features=None, C=1.0, class_weight=None, robust=True):
        self.features = features
        self.C = C
        self.class_weight = class_weight
        self.robust = robust

    def _idx(self):
        return [FEATURE_NAMES.index(f) for f in (self.features or FEATURE_NAMES)]

    def transform_features(self, texts):
        return np.array([extract(t) for t in texts], dtype=float)

    def _prep(self, X):
        X = np.log1p(X) if self.robust else X.copy()
        return X[:, self._idx()]

    def _scale(self, X):
        Z = self.scaler_.transform(X)
        return np.clip(Z, -5, 5) if self.robust else Z

    def fit(self, texts, y, X_cached=None):
        X = self._prep(X_cached if X_cached is not None else self.transform_features(texts))
        self.scaler_ = StandardScaler().fit(X)
        self.lr_ = LogisticRegression(C=self.C, class_weight=self.class_weight, max_iter=2000,
                                      random_state=42).fit(self._scale(X), y)
        self.classes_ = self.lr_.classes_
        return self

    def predict(self, texts, X_cached=None):
        X = self._prep(X_cached if X_cached is not None else self.transform_features(texts))
        return self.lr_.predict(self._scale(X))