"""Stage 1 (TF-IDF + LogReg) and Stage 2 (BiLSTM, PyTorch) on GoEmotions single-label."""
import re
import time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from common import load_all, save_result, EMOTIONS

OUT = "results"
torch.manual_seed(42); np.random.seed(42)
train, dev, test = load_all()
print(f"train {len(train)}  dev {len(dev)}  test {len(test)}  classes {len(EMOTIONS)}")
print("train class share (top 5):", (train.y.map(dict(enumerate(EMOTIONS))).value_counts(normalize=True).head(5).round(3).to_dict()))

# ---------------- Stage 1: TF-IDF + Logistic Regression ----------------
t0 = time.time()
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr, Xdv, Xte = vec.fit_transform(train.text), vec.transform(dev.text), vec.transform(test.text)
best = None
for C in [1, 3, 10]:   # tune on dev, never on test
    lr = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    lr.fit(Xtr, train.y)
    f = f1_score(dev.y, lr.predict(Xdv), average="macro")
    print(f"  LR C={C}: dev macro-F1 {f:.4f}")
    if best is None or f > best[0]: best = (f, C, lr)
lr = best[2]
print(f"TF-IDF+LR done in {time.time()-t0:.0f}s (best C={best[1]})")
save_result("tfidf_lr", test.y.values, lr.predict(Xte), test.text, OUT)

# ---------------- Stage 2: BiLSTM ----------------
tok = lambda s: re.findall(r"[a-z']+|[!?.,]", s.lower())
cnt = Counter(w for s in train.text for w in tok(s))
vocab = {w: i + 2 for i, (w, c) in enumerate(cnt.most_common()) if c >= 2}  # 0=pad 1=unk
MAXLEN = 40
def enc(texts):
    X = np.zeros((len(texts), MAXLEN), dtype=np.int64)
    for i, s in enumerate(texts):
        ids = [vocab.get(w, 1) for w in tok(s)][:MAXLEN]
        X[i, :len(ids)] = ids
    return torch.from_numpy(X)
Xtr_t, Xdv_t, Xte_t = enc(train.text), enc(dev.text), enc(test.text)
ytr_t = torch.tensor(train.y.values)

class BiLSTM(nn.Module):
    def __init__(self, V, n_cls, emb=128, hid=128):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(hid * 2, n_cls)
    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        h, _ = self.lstm(self.drop(self.emb(x)))
        h = h.masked_fill(~mask, -1e9).max(1).values   # max-pool over real tokens
        return self.fc(self.drop(h))

model = BiLSTM(len(vocab) + 2, len(EMOTIONS))
# class-weighted loss (sqrt-inverse frequency) to help macro-F1 on rare emotions
freq = np.bincount(train.y.values, minlength=len(EMOTIONS)).astype(float)
w = torch.tensor((freq.sum() / (len(freq) * np.maximum(freq, 1))) ** 0.5, dtype=torch.float)
loss_fn = nn.CrossEntropyLoss(weight=w)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)

def predict(X, bs=512):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs): out.append(model(X[i:i + bs]).argmax(1))
    return torch.cat(out).numpy()

best_f, best_state, patience = -1, None, 0
t0 = time.time()
for ep in range(1, 16):
    model.train(); perm = torch.randperm(len(Xtr_t))
    for i in range(0, len(perm), 64):
        idx = perm[i:i + 64]
        opt.zero_grad(); loss_fn(model(Xtr_t[idx]), ytr_t[idx]).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    f = f1_score(dev.y, predict(Xdv_t), average="macro")
    print(f"  epoch {ep:2d}  dev macro-F1 {f:.4f}  ({time.time()-t0:.0f}s)")
    if f > best_f: best_f, best_state, patience = f, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        patience += 1
        if patience >= 3: break
model.load_state_dict(best_state)
save_result("bilstm", test.y.values, predict(Xte_t), test.text, OUT)
