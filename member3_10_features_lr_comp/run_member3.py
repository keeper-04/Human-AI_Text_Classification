import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB

from features import FEATURE_INFO, FEATURE_NAMES, WORD_RE, FeatureLR, extract

OUT = "results"
os.makedirs(OUT, exist_ok=True)
GROUPS = {}
for n, (g, _) in FEATURE_INFO.items():
    GROUPS.setdefault(g, []).append(n)


_CACHE = {}  


def _all_features():
    if "feats" not in _CACHE:
        dfs = [pd.read_csv(f"data/{c}_{s}.csv") for c in "ABC" for s in ("train", "test")]
        uniq = pd.concat(dfs).drop_duplicates("id")
        with Pool() as p:  
            rows = p.map(extract, list(uniq.text), chunksize=100)
        _CACHE["feats"] = pd.DataFrame(rows, columns=FEATURE_NAMES, index=uniq.id.values)
    return _CACHE["feats"]


def feats(df):
    return _all_features().loc[df.id].values


def mf1(y, p):
    return f1_score(y, p, average="macro")


def main():
    summary, ablation, dropone, coefs, corrs, errors = [], [], [], [], [], []
    for ch in "ABC":
        tr = pd.read_csv(f"data/{ch}_train.csv")
        te = pd.read_csv(f"data/{ch}_test.csv")
        print(ch, tr.shape, te.shape, flush=True)
        Xtr, Xte = feats(tr), feats(te)
        ytr, yte = tr.label.values, te.label.values
        full = FeatureLR().fit(tr.text, ytr, X_cached=Xtr)
        pred = full.predict(te.text, X_cached=Xte)
        Zte = full._scale(full._prep(Xte))
        proba = full.lr_.predict_proba(Zte)[:, 1]
        full_f1 = mf1(yte, pred)
        rep = classification_report(yte, pred, digits=4)
        open(f"{OUT}/classification_report_{ch}_features_LR.txt", "w").write(rep)
        print(rep, flush=True)
        pd.DataFrame({"id": te.id, "label": pred}).to_csv(f"{OUT}/predictions_{ch}_features_LR.csv", index=False)
        row = {"challenge": ch, "Features+LR (10 feats)": full_f1}

        cv = CountVectorizer(max_features=10000, min_df=5)
        Btr, Bte = cv.fit_transform(tr.text), cv.transform(te.text)
        row["BoW+NB (baseline)"] = mf1(yte, MultinomialNB(alpha=1.0).fit(Btr, ytr).predict(Bte))
        row["BoW+LR"] = mf1(yte, LogisticRegression(max_iter=1000).fit(Btr, ytr).predict(Bte))
        tv = TfidfVectorizer(max_features=10000, min_df=5)
        Ttr, Tte = tv.fit_transform(tr.text), tv.transform(te.text)
        row["TF-IDF+LR"] = mf1(yte, LogisticRegression(max_iter=1000).fit(Ttr, ytr).predict(Tte))
        Ctr = hstack([Ttr, csr_matrix(full._scale(full._prep(Xtr)))]).tocsr()
        Cte = hstack([Tte, csr_matrix(Zte)]).tocsr()
        row["TF-IDF + features + LR"] = mf1(yte, LogisticRegression(max_iter=2000).fit(Ctr, ytr).predict(Cte))
        bal = FeatureLR(class_weight="balanced").fit(None, ytr, X_cached=Xtr)
        bp = bal.predict(None, X_cached=Xte)
        row["Features+LR balanced"] = mf1(yte, bp)
        open(f"{OUT}/classification_report_{ch}_features_LR_balanced.txt", "w").write(classification_report(yte, bp, digits=4))
        pd.DataFrame({"id": te.id, "label": bp}).to_csv(f"{OUT}/predictions_{ch}_features_LR_balanced.csv", index=False)
        plain = FeatureLR(robust=False).fit(None, ytr, X_cached=Xtr)
        pp = plain.predict(None, X_cached=Xte)
        row["Features+LR plain (no log/clip)"] = mf1(yte, pp)
        open(f"{OUT}/classification_report_{ch}_features_LR_plain.txt", "w").write(classification_report(yte, pp, digits=4))
        pd.DataFrame({"id": te.id, "label": pp}).to_csv(f"{OUT}/predictions_{ch}_features_LR_plain.csv", index=False)
        summary.append(row)
        print(row, flush=True)

        for g, names in GROUPS.items():
            only = FeatureLR(features=names).fit(None, ytr, X_cached=Xtr)
            wo = FeatureLR(features=[f for f in FEATURE_NAMES if f not in names]).fit(None, ytr, X_cached=Xtr)
            ablation.append({"challenge": ch, "group": g, "n_feats": len(names),
                             "only_group_F1": mf1(yte, only.predict(None, X_cached=Xte)),
                             "all_but_group_F1": mf1(yte, wo.predict(None, X_cached=Xte)),
                             "drop_delta": mf1(yte, wo.predict(None, X_cached=Xte)) - full_f1})
        for f in FEATURE_NAMES:
            m = FeatureLR(features=[x for x in FEATURE_NAMES if x != f]).fit(None, ytr, X_cached=Xtr)
            dropone.append({"challenge": ch, "feature": f, "F1_without": mf1(yte, m.predict(None, X_cached=Xte)),
                            "delta": mf1(yte, m.predict(None, X_cached=Xte)) - full_f1})
        for i, f in enumerate(FEATURE_NAMES):
            coefs.append({"challenge": ch, "feature": f, "coef": full.lr_.coef_[0][i]})
            corrs.append({"challenge": ch, "feature": f, "pearson_with_label": np.corrcoef(Xtr[:, i], ytr)[0, 1],
                          "mean_label0": Xtr[ytr == 0, i].mean(), "mean_label1": Xtr[ytr == 1, i].mean()})
        fc = pd.DataFrame(Xtr, columns=FEATURE_NAMES).corr()
        fc.to_csv(f"{OUT}/feature_corr_matrix_{ch}.csv")

        Z = Zte
        wrong = np.where(pred != yte)[0]
        conf = np.abs(proba[wrong] - 0.5)
        for j in wrong[np.argsort(-conf)][:40]:
            contrib = Z[j] * full.lr_.coef_[0]
            top = np.argsort(-np.abs(contrib))[:3]
            errors.append({"challenge": ch, "id": te.id[j], "true": int(yte[j]), "pred": int(pred[j]),
                           "p1": round(float(proba[j]), 3), "n_words": len(WORD_RE.findall(te.text[j])),
                           "top_drivers": "; ".join(f"{FEATURE_NAMES[k]}={Xte[j, k]:.3g} (z={Z[j, k]:+.1f}, push {contrib[k]:+.2f})" for k in top),
                           "text_start": te.text[j][:300].replace("\n", " ")})

    pd.DataFrame(summary).to_csv(f"{OUT}/summary_macroF1.csv", index=False)
    pd.DataFrame(ablation).to_csv(f"{OUT}/ablation_groups.csv", index=False)
    pd.DataFrame(dropone).to_csv(f"{OUT}/ablation_drop_one.csv", index=False)
    pd.DataFrame(coefs).to_csv(f"{OUT}/coefficients.csv", index=False)
    pd.DataFrame(corrs).to_csv(f"{OUT}/correlations_with_label.csv", index=False)
    pd.DataFrame(errors).to_csv(f"{OUT}/error_analysis.csv", index=False)
    pd.DataFrame([{"feature": k, "group": v[0], "definition": v[1]} for k, v in FEATURE_INFO.items()]).to_csv(
        f"{OUT}/feature_definitions.csv", index=False)


if __name__ == "__main__":
    main()