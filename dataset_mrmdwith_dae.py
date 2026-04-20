import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import json
import time

# MRmD helper functions
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

DATA_DIR = 'datasets'

# ===========================================================================
#  DAE Embedding Model (Denoising Autoencoder)
#  Berdasarkan: Vincent et al. (2008) "Extracting and Composing Robust
#  Features with Denoising Autoencoders", ICML 2008.
#
#  Arsitektur mengikuti paper secara ketat (Section 2.2 & 2.3):
#
#    Input x ∈ {0,1}^d  — one-hot gabungan semua kolom (d = Σ vocab_size_i)
#
#    Encoder  f_θ  (eq. 1):
#      y = f_θ(x̃) = sigmoid(W · x̃ + b)
#      W ∈ R^{d' × d},  b ∈ R^{d'},  d' = hidden_dim
#
#    Decoder  g_θ'  (eq. 1):
#      z = g_θ'(y) = sigmoid(W' · y + b')
#      W' ∈ R^{d × d'},  (opsional: W' = W^T, tied weights)
#
#    Corruption  qD  (Section 2.3):
#      Untuk setiap fitur, dengan prob ν: set nilai → 0  ("forced to 0")
#      Hanya aktif saat training; inference pakai x_clean langsung.
#
#    Objective (eq. 5):
#      min E_{x~data, x̃~qD(x̃|x)} [ L_H(x, g_θ'(f_θ(x̃))) ]
#      L_H = cross-entropy biner per elemen one-hot (eq. 2):
#        L_H(x, z) = -Σ_k [ x_k log z_k + (1-x_k) log(1-z_k) ]
#
#  Output embedding y [batch, hidden_dim] kompatibel dengan pipeline
#  diffusion downstream. Pipeline embedding → imputasi TIDAK BERUBAH.
# ===========================================================================





# ===========================================================================
#  MRmD Discretizer (implementasi Max-Relevance-Min-Divergence)
#  Berdasarkan: Wang et al., Pattern Recognition 149 (2024) 110236
# ===========================================================================

def _mutual_information(a_discrete: np.ndarray, c: np.ndarray) -> float:
    """
    Hitung Mutual Information I(A; C).
    Persamaan (3) di paper: I(A;C) = Σ_{a,c} P(a,c)*log[P(a,c)/(P(a)*P(c))]
    """
    n = len(a_discrete)
    bins_a = np.unique(a_discrete)
    bins_c = np.unique(c)
    mi = 0.0
    for a_val in bins_a:
        mask_a = (a_discrete == a_val)
        p_a = mask_a.sum() / n
        for c_val in bins_c:
            p_ac = ((mask_a) & (c == c_val)).sum() / n
            p_c  = (c == c_val).sum() / n
            if p_ac > 0 and p_a > 0 and p_c > 0:
                mi += p_ac * np.log(p_ac / (p_a * p_c))
    return max(mi, 0.0)


def _js_divergence(p_t: np.ndarray, p_v: np.ndarray) -> float:
    """
    Hitung Jensen-Shannon Divergence D_JS(P_t ‖ P_v).
    Persamaan (4)-(6) di paper. D_JS ∈ [0, 1].
    """
    eps    = 1e-10
    p_t    = np.clip(p_t, eps, 1.0)
    p_v    = np.clip(p_v, eps, 1.0)
    p_star = 0.5 * (p_t + p_v)
    kl_t   = np.sum(p_t * np.log(p_t / p_star))
    kl_v   = np.sum(p_v * np.log(p_v / p_star))
    return float(np.clip(0.5 * (kl_t + kl_v), 0.0, 1.0))


def _get_distributions(a_train: np.ndarray, a_val: np.ndarray):
    """Hitung P_t dan P_v dari label bin diskrit (training & validation)."""
    all_bins = np.union1d(np.unique(a_train), np.unique(a_val))
    p_t = np.array([(a_train == b).sum() for b in all_bins], dtype=float)
    p_v = np.array([(a_val   == b).sum() for b in all_bins], dtype=float)
    if p_t.sum() > 0: p_t /= p_t.sum()
    if p_v.sum() > 0: p_v /= p_v.sum()
    return p_t, p_v


def _make_bins(cut_points: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Bangun array edges bins: [x_min-ε, cp1, cp2, ..., x_max+ε]."""
    lo = x_min - 1e-10
    hi = x_max + 1e-10
    if len(cut_points) == 0:
        return np.array([lo, hi])
    return np.concatenate([[lo], np.sort(cut_points), [hi]])


def _discretize_mrmd(x: np.ndarray, cut_points: np.ndarray,
                     x_min: float, x_max: float) -> np.ndarray:
    """Diskritisasi array x menggunakan cut_points → label bin integer [0,1,2,...]."""
    if len(cut_points) == 0:
        return np.zeros(len(x), dtype=int)
    bins = _make_bins(cut_points, x_min, x_max)
    return (np.digitize(x, bins[1:-1])).astype(int)


class MRmDDiscretizer(BaseEstimator, TransformerMixin):
    """
    MRmD (Max-Relevance-Min-Divergence) Discretizer.

    Implementasi Algorithm 1 dari:
      Wang et al., Pattern Recognition 149 (2024) 110236

    Optimasi dua kriteria secara bersamaan (Persamaan 13):
      Ψ(Aj; C) = λ * I(Aj; C)  −  D_JS(P_t(aj) ‖ P_v(aj))

    di mana:
      • I(Aj; C)         = Mutual Information atribut-diskrit vs kelas
      • D_JS(P_t ‖ P_v)  = Jensen-Shannon Divergence distribusi train vs val
      • λ = exp(-|D*_j| / N_D) (bobot adaptif, Persamaan 14)

    Kompatibel dengan scikit-learn API (fit / transform / fit_transform).

    Parameters
    ----------
    val_size     : float, default=0.125  — proporsi data untuk validasi internal
    N_D          : int,   default=50     — parameter λ
    random_state : int or None           — seed untuk split train/val
    verbose      : bool,  default=False
    """

    def __init__(self, val_size: float = 0.125, N_D: int = 50,
                 random_state=None, verbose: bool = False):
        self.val_size     = val_size
        self.N_D          = N_D
        self.random_state = random_state
        self.verbose      = verbose

    def fit(self, X, y):
        """
        Fit MRmD: temukan cut point optimal untuk tiap fitur.
        X : [N, n_cols] float — fitur numerik
        y : [N]         int   — label kelas
        """
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
            X = np.array(X, dtype=float)
        else:
            X = np.array(X, dtype=float)

        y = np.array(y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Split internal: training vs validation
        rng       = np.random.RandomState(self.random_state)
        val_n     = max(1, int(n_samples * self.val_size))
        val_idx   = rng.choice(n_samples, size=val_n, replace=False)
        train_idx = np.setdiff1d(np.arange(n_samples), val_idx)

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_vl       = X[val_idx]

        if self.verbose:
            print(f'[MRmD] n_train={len(train_idx)}, n_val={len(val_idx)}, '
                  f'n_features={n_features}')

        self.cut_points_ = []
        self.x_min_      = []
        self.x_max_      = []
        # Alias agar kompatibel dengan kode lama yang pakai .n_bins_
        self.n_bins_     = []

        for j in range(n_features):
            x_tr_j = X_tr[:, j]
            x_vl_j = X_vl[:, j]

            x_min = float(x_tr_j.min())
            x_max = float(x_tr_j.max())
            self.x_min_.append(x_min)
            self.x_max_.append(x_max)

            unique_tr = np.unique(x_tr_j)
            if len(unique_tr) <= 1:
                self.cut_points_.append(np.array([]))
                self.n_bins_.append(1)
                if self.verbose:
                    print(f'  [MRmD] Col {j}: konstan, skip.')
                continue

            cp = self._fit_one_attribute(x_tr_j, x_vl_j, y_tr,
                                         unique_tr, x_min, x_max, j)
            self.cut_points_.append(cp)
            n_bins = len(cp) + 1
            self.n_bins_.append(n_bins)
            print(f'  [MRmD] Col {j}: {len(cp)} cut points → {n_bins} bins')

        return self

    def _fit_one_attribute(self, x_tr, x_vl, c_tr,
                            unique_all, x_min, x_max, j_idx):
        """Algorithm 1 untuk satu atribut. Return cut points optimal (sorted)."""
        D_star_j = np.array([])
        S_j      = unique_all.copy()
        psi_max  = -np.inf

        while len(S_j) > 0:
            best_psi = -np.inf
            best_dk  = None

            for dk in S_j:
                D_k_j     = np.append(D_star_j, dk)
                a_tr_disc = _discretize_mrmd(x_tr, D_k_j, x_min, x_max)
                a_vl_disc = _discretize_mrmd(x_vl, D_k_j, x_min, x_max)

                n_cuts  = len(D_star_j) + 1
                lam     = np.exp(-n_cuts / self.N_D)
                mi_val  = _mutual_information(a_tr_disc, c_tr)
                p_t, p_v = _get_distributions(a_tr_disc, a_vl_disc)
                jsd_val = _js_divergence(p_t, p_v)
                psi_k   = lam * mi_val - jsd_val

                if psi_k > best_psi:
                    best_psi = psi_k
                    best_dk  = dk

            if best_dk is None or best_psi <= psi_max:
                break

            psi_max  = best_psi
            D_star_j = np.append(D_star_j, best_dk)
            S_j      = S_j[S_j != best_dk]

        result = np.sort(D_star_j)
        if self.verbose:
            print(f'  Fitur [{j_idx}]: {len(result)} cut points '
                  f'→ {np.round(result, 4).tolist()}')
        return result

    def transform(self, X) -> np.ndarray:
        """
        Transform nilai kontinu → integer bin index [0, n_bins-1].
        X : [N, n_cols]
        Return : [N, n_cols]  int64
        """
        check_is_fitted(self, 'cut_points_')

        if hasattr(X, 'values'):
            X = X.values
        X = np.array(X, dtype=float)

        out = np.empty(X.shape, dtype=np.int64)
        for j, cp in enumerate(self.cut_points_):
            out[:, j] = _discretize_mrmd(
                X[:, j], cp, self.x_min_[j], self.x_max_[j]
            ).astype(np.int64)

        return out

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def get_n_bins(self) -> np.ndarray:
        """Jumlah bin per fitur setelah fit."""
        check_is_fitted(self, 'cut_points_')
        return np.array(self.n_bins_)

    def get_bin_midpoints(self, X_norm: np.ndarray,
                          X_norm_binned: np.ndarray) -> list:
        """
        Hitung nilai tengah (midpoint) setiap bin dalam skala normalisasi.

        Dipakai saat decoding: bin index → nilai kontinu dalam skala (X-mean)/std.

        X_norm        : [N, n_cols]  — data normalisasi (skala (X-mean)/std)
        X_norm_binned : [N, n_cols]  — hasil transform (integer bin index)

        Return : list[n_cols] of np.ndarray, tiap elemen panjang n_bins_[col]
        """
        n_cols    = X_norm.shape[1]
        midpoints = []

        for col in range(n_cols):
            n_bins = self.n_bins_[col]
            mids   = np.zeros(n_bins, dtype=np.float32)

            for b in range(n_bins):
                mask = X_norm_binned[:, col] == b
                if mask.sum() > 0:
                    mids[b] = float(X_norm[mask, col].mean())
                else:
                    # Bin kosong → interpolasi linear
                    mids[b] = float(b) / max(n_bins - 1, 1)

            midpoints.append(mids)

        return midpoints

def compute_embedding_size(n_categories: int) -> int:
    """
    Hitung ukuran embedding optimal berdasarkan jumlah kategori.
    Rumus: min(600, round(1.6 * n_categories^0.56))
    Referensi: Guo & Berkhahn (2016)
    """
    return min(600, round(1.6 * n_categories ** 0.56))


# ===========================================================================
#  Corruption function (qD dari Vincent et al. 2008)
# ===========================================================================

def corrupt_input(x_clean: torch.Tensor,
                  cat_dims: list,
                  corruption_prob: float = 0.3,
                  corruption_type: str = 'mask') -> torch.Tensor:
    """
    Generate x_tilde ~ qD(x_tilde | x_clean) sesuai Vincent et al. (2008).

    Corruption dilakukan di level input (integer index), BUKAN di embedding/latent.
    Ini mencegah trivial identity mapping karena encoder tidak pernah melihat
    x_clean secara langsung saat training.

    Parameter
    ----------
    x_clean          : [batch, n_cols]  — integer index bersih
    cat_dims         : list[int]        — vocab size per kolom (untuk random_replace)
    corruption_prob  : float            — probabilitas tiap feature di-corrupt [0,1]
    corruption_type  : str              — 'mask' | 'random_replace'
        'mask'         : set token → 0 (zero/padding token masking)
        'random_replace': ganti token → random valid index ≠ original (acak kategori)

    Return
    ------
    x_tilde : [batch, n_cols]  — integer index yang sudah di-corrupt, dtype sama
    """
    x_tilde = x_clean.clone()

    # Boolean mask: True = kolom ini di-corrupt untuk sample tersebut
    # Shape: [batch, n_cols]; setiap entry Bernoulli(corruption_prob)
    corrupt_mask = torch.bernoulli(
        torch.full_like(x_clean, corruption_prob, dtype=torch.float32)
    ).bool()

    if corruption_type == 'mask':
        # Set ke 0 (token index 0 = padding/mask token)
        # Model harus merekonstruksi nilai aslinya dari konteks kolom lain
        x_tilde[corrupt_mask] = 0

    elif corruption_type == 'random_replace':
        # Ganti dengan random index yang valid per kolom
        # Tidak harus berbeda dari aslinya (mengikuti konvensi paper asli)
        for col_i, n_vocab in enumerate(cat_dims):
            col_mask = corrupt_mask[:, col_i]
            if col_mask.any():
                n_corrupted = col_mask.sum().item()
                rand_idx    = torch.randint(
                    low=0, high=n_vocab,
                    size=(int(n_corrupted),),
                    device=x_clean.device
                )
                x_tilde[col_mask, col_i] = rand_idx
    else:
        raise ValueError(f"corruption_type harus 'mask' atau 'random_replace', "
                         f"dapat: '{corruption_type}'")

    return x_tilde


# ===========================================================================
#  DAE Embedding Model
#  Implementasi Denoising Autoencoder sesuai Vincent et al. (2008)
# ===========================================================================

class DAEEmbeddingModel(nn.Module):
    """
    Denoising Autoencoder (DAE) Embedding Model.

    Berdasarkan Vincent et al. (2008) ICML — Section 2.2 & 2.3.

    Arsitektur mengikuti paper secara ketat:

      Encoder  f_θ  (Section 2.2, eq. 1):
        Input x_tilde diproyeksikan menjadi one-hot per kolom, lalu
        digabung menjadi vektor biner/kontinu [batch, total_onehot_dim].
        Kemudian dipetakan ke representasi laten melalui satu lapisan
        affine + sigmoid:
            y = f_θ(x̃) = sigmoid(W · x̃ + b)
        W ∈ R^{d' × d},  b ∈ R^{d'},  d = total_onehot_dim, d' = hidden_dim.
        (Paper Section 2.2: "y = f_θ(x) = s(Wx + b)")

      Decoder  g_θ'  (Section 2.2, eq. 1):
        Representasi laten y dipetakan kembali ke ruang input melalui satu
        lapisan affine + sigmoid:
            z = g_θ'(y) = sigmoid(W'y + b')
        W' ∈ R^{d × d'},  b' ∈ R^{d}.
        Dari z kemudian diambil slice per kolom → logits rekonstruksi.
        (Paper Section 2.2: "z = g_θ'(y) = s(W'y + b')")

      Tied weights (opsional, Section 2.2):
        W' = W^T  — didukung via parameter `tied_weights`.

    Alur forward (training):
      x_clean [batch, n_cols]
        → one_hot(x_clean) → x_oh [batch, total_onehot_dim]
        → corrupt(x_oh)    → x̃_oh  [batch, total_onehot_dim]
        → f_θ(x̃_oh)       → y     [batch, hidden_dim]
        → g_θ'(y)          → z_oh  [batch, total_onehot_dim]
        → slice per kolom  → logits rekonstruksi
        → CE loss vs x_clean  (eq. 5)

    Alur forward (inference):
      x_clean → one_hot → f_θ → y  (tanpa corruption, Section 2.4)
      y dikembalikan sebagai representasi embedding.

    Objective (eq. 5 paper):
      min E_{x~data, x̃~qD(x̃|x)} [ L_H(x, g_θ'(f_θ(x̃))) ]
      L_H = cross-entropy rekonstruksi per kolom (unsupervised).

    Kompatibilitas downstream:
      - encode() mengembalikan y [batch, hidden_dim]
      - decode() mengembalikan list[n_cols] logits dari z_oh
      - forward() signature: (x_cat, add_noise=False) → (y, None, recon_logits)
    """

    def __init__(self, cat_dims: list, emb_sizes: list,
                 n_classes: int = 2,          # diabaikan, hanya untuk kompatibilitas API
                 dropout: float = 0.1,
                 hidden_dim: int = 256,        # d' dalam paper: dimensi laten y
                 use_mlp: bool = True,         # diabaikan (bukan arsitektur paper)
                 mlp_ratio: float = 1.5,       # diabaikan
                 noise_std: float = 0.0,       # diabaikan
                 corruption_prob: float = 0.3,
                 corruption_type: str = 'mask',
                 tied_weights: bool = False):  # W' = W^T (Section 2.2 paper)
        super().__init__()

        self.cat_dims        = cat_dims
        self.emb_sizes       = emb_sizes       # dipertahankan untuk kompatibilitas downstream
        self.n_cols          = len(cat_dims)
        self.total_onehot    = sum(cat_dims)   # d: dimensi input one-hot gabungan
        self.hidden_dim      = hidden_dim      # d': dimensi laten (representasi y)
        self.out_dim         = hidden_dim      # output encode() = hidden_dim
        self.total_emb_dim   = hidden_dim      # alias untuk kompatibilitas downstream
        self.corruption_prob = corruption_prob
        self.corruption_type = corruption_type
        self.tied_weights    = tied_weights

        # ── Encoder  f_θ: x̃ → y  ─────────────────────────────────────────
        # y = sigmoid(W · x̃ + b)
        # W ∈ R^{d' × d},  b ∈ R^{d'}
        # (Vincent et al. 2008, Section 2.2)
        self.W_enc = nn.Linear(self.total_onehot, hidden_dim, bias=True)

        # ── Decoder  g_θ': y → z  ────────────────────────────────────────
        # z = sigmoid(W' · y + b')
        # Jika tied_weights: W' = W^T  → hanya bias b' yang dilatih terpisah
        # Jika tidak tied: W' bebas (nn.Linear penuh)
        # (Vincent et al. 2008, Section 2.2: "optionally W' = W^T")
        if tied_weights:
            # b' tetap dilatih; W' dihitung on-the-fly dari W_enc.weight.T
            self.b_dec = nn.Parameter(torch.zeros(self.total_onehot))
        else:
            self.W_dec = nn.Linear(hidden_dim, self.total_onehot, bias=True)

        # dropout untuk regularisasi ringan (tidak mengubah alur utama)
        self.dropout = nn.Dropout(dropout)

        # lookup: offset awal tiap kolom dalam vektor one-hot gabungan
        self._col_offsets = [0]
        for d in cat_dims[:-1]:
            self._col_offsets.append(self._col_offsets[-1] + d)

    def _to_onehot(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Konversi integer index [batch, n_cols] → one-hot gabungan [batch, total_onehot].

        Setiap kolom j dengan vocab_size = cat_dims[j] diubah ke vektor
        one-hot lalu digabung secara horizontal:
            x_oh = [onehot(x[:,0]) | onehot(x[:,1]) | ... | onehot(x[:,n_cols-1])]

        Ini adalah representasi biner x ∈ {0,1}^d yang digunakan paper
        sebagai domain input encoder (Section 2.2).
        """
        parts = []
        for i, n_cat in enumerate(self.cat_dims):
            oh = torch.zeros(x_cat.shape[0], n_cat,
                             device=x_cat.device, dtype=torch.float32)
            oh.scatter_(1, x_cat[:, i].unsqueeze(1), 1.0)
            parts.append(oh)
        return torch.cat(parts, dim=1)   # [batch, total_onehot]

    def _corrupt_onehot(self, x_oh: torch.Tensor) -> torch.Tensor:
        """
        Corruption qD di ruang one-hot sesuai Vincent et al. (2008) Section 2.3.

        Untuk setiap kolom j, dengan probabilitas corruption_prob:
          'mask'          → set seluruh slice one-hot kolom j ke 0
                            ("forced to 0", seperti masking fitur)
          'random_replace'→ ganti slice one-hot kolom j dengan one-hot acak lain

        Ini sesuai dengan: "a fixed number νd of components are chosen at random,
        and their value is forced to 0" (Section 2.3).
        """
        x_tilde = x_oh.clone()
        batch   = x_oh.shape[0]

        for i, n_cat in enumerate(self.cat_dims):
            start = self._col_offsets[i]
            end   = start + n_cat

            # Bernoulli per sample untuk kolom ini
            col_corrupt = torch.bernoulli(
                torch.full((batch,), self.corruption_prob,
                           device=x_oh.device)
            ).bool()

            if not col_corrupt.any():
                continue

            if self.corruption_type == 'mask':
                x_tilde[col_corrupt, start:end] = 0.0

            elif self.corruption_type == 'random_replace':
                rand_idx = torch.randint(0, n_cat,
                                         (int(col_corrupt.sum()),),
                                         device=x_oh.device)
                rand_oh = torch.zeros(int(col_corrupt.sum()), n_cat,
                                      device=x_oh.device)
                rand_oh.scatter_(1, rand_idx.unsqueeze(1), 1.0)
                x_tilde[col_corrupt, start:end] = rand_oh

        return x_tilde

    def encode(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        f_θ: integer index → representasi laten y (tanpa corruption).

        Mengimplementasikan persamaan (Section 2.2 paper):
            y = f_θ(x̃) = sigmoid(W · x̃ + b)

        Digunakan saat:
          - training: encode(x_tilde_oh)  [dipanggil dari forward() setelah corrupt]
          - inference: encode(x_clean)    [tanpa corrupt, Section 2.4]

        x_cat  : [batch, n_cols]  — integer index tiap kolom
        return : [batch, hidden_dim]   — representasi laten y
        """
        x_oh = self._to_onehot(x_cat)           # x ∈ {0,1}^d
        x_oh = self.dropout(x_oh)
        y    = torch.sigmoid(self.W_enc(x_oh))  # y = sigmoid(Wx + b)
        return y                                  # [batch, hidden_dim]

    def decode(self, y: torch.Tensor) -> list:
        """
        g_θ': representasi laten y → logits rekonstruksi per kolom.

        PERBAIKAN (konsisten dengan Vincent et al. 2008 + CrossEntropyLoss):
        ─────────────────────────────────────────────────────────────────────
        Untuk fitur multi-class (one-hot per kolom), setiap kolom harus
        diperlakukan sebagai distribusi kategoris — bukan biner independen.

        Arsitektur decoder mengikuti paper (Section 2.2):
            z_raw = W'y + b'   ← affine projection (RAW, tanpa aktivasi)
        Kemudian z_raw di-slice per kolom:
            logits_j = z_raw[:, offset_j : offset_j + vocab_j]  [batch, vocab_j]

        CrossEntropyLoss (dipakai di training) sudah mengandung softmax internal,
        sehingga TIDAK perlu sigmoid/softmax di sini.

        Untuk argmax (prediksi kelas): argmax(logits_j) — benar karena
            argmax(softmax(logits)) = argmax(logits)

        Untuk decode numerik (weighted-sum): softmax(logits_j) @ midpoints — benar.

        Catatan: tied_weights (W' = W^T) tetap didukung, namun tanpa sigmoid.

        y      : [batch, hidden_dim]
        return : list[n_cols] of [batch, vocab_size_i]  — RAW logits
        """
        if self.tied_weights:
            # W' = W^T  (tied weights, Section 2.2) — tanpa sigmoid
            z_raw = torch.nn.functional.linear(y, self.W_enc.weight.t(), self.b_dec)
        else:
            z_raw = self.W_dec(y)  # [batch, total_onehot] — RAW logits

        # Slice per kolom → logits rekonstruksi (RAW, tanpa aktivasi)
        logits_per_col = []
        for i, n_cat in enumerate(self.cat_dims):
            start = self._col_offsets[i]
            end   = start + n_cat
            logits_per_col.append(z_raw[:, start:end])   # [batch, vocab_size_i]

        return logits_per_col

    def forward(self, x_cat: torch.Tensor,
                add_noise: bool = False):
        """
        Forward pass dengan alur DAE sesuai Vincent et al. (2008):

        Training (self.training=True):
          x_clean [batch, n_cols]
            → one_hot         → x_oh    [batch, total_onehot]
            → corrupt qD      → x̃_oh   [batch, total_onehot]
            → f_θ(x̃_oh)      → y       [batch, hidden_dim]     ← eq. 1
            → g_θ'(y)         → logits  list[n_cols]            ← eq. 1
            → CE loss vs x_clean                                ← eq. 5

        Inference (self.training=False):
          x_clean → one_hot → f_θ → y
          (corruption TIDAK digunakan — Section 2.4:
           "the corruption process qD is only used during training")

        Parameter
        ----------
        x_cat      : [batch, n_cols]  — x_clean (integer index bersih)
        add_noise  : bool             — diabaikan (kompatibilitas API lama)

        Return
        ------
        y            : [batch, hidden_dim]      — latent representation
        class_logits : None                     — placeholder (tidak ada classifier)
        recon_logits : list[n_cols] of Tensor   — logits rekonstruksi tiap kolom
        """
        if self.training and self.corruption_prob > 0:
            # Corruption HANYA saat training — mencegah identity mapping
            # Dilakukan di ruang one-hot, sesuai Section 2.3 paper
            x_oh    = self._to_onehot(x_cat)
            x_input = self._corrupt_onehot(x_oh)
            x_input = self.dropout(x_input)
            y       = torch.sigmoid(self.W_enc(x_input))   # f_θ(x̃)
        else:
            # Inference: encode langsung dari x_clean (tanpa corrupt)
            y = self.encode(x_cat)

        recon_logits = self.decode(y)   # g_θ'(y)

        return y, None, recon_logits


# ===========================================================================
#  Training DAE Embedding
#  Menggantikan train_supervised_embedding_model
# ===========================================================================

# Alias untuk kompatibilitas tipe hint di encode_with_embedding & helpers
SupervisedLearnableEmbeddingModel = DAEEmbeddingModel


def train_dae_embedding_model(cat_idx_array: np.ndarray,
                              labels: np.ndarray,          # diterima tapi TIDAK dipakai
                              cat_dims: list,
                              emb_sizes: list,
                              n_classes: int,              # diterima tapi TIDAK dipakai
                              device: str,
                              n_epochs: int = 1000,
                              batch_size: int = 1024,
                              lr: float = 1e-3,
                              dropout: float = 0.1,
                              hidden_dim: int = 256,
                              use_mlp: bool = True,
                              mlp_ratio: float = 1.5,
                              noise_std: float = 0.0,      # diabaikan, untuk kompatibilitas
                              patience: int = 40,
                              corruption_prob: float = 0.3,
                              corruption_type: str = 'mask') -> DAEEmbeddingModel:
    """
    Latih DAEEmbeddingModel dengan objective unsupervised denoising.

    Objective (eq. 5 Vincent et al. 2008):
      min E_{x~data, x_tilde~qD(x_tilde|x)} [ L(x, g(f(x_tilde))) ]

    Loss L = average categorical cross-entropy per kolom:
      L = (1/n_cols) * Σ_j CE(logits_j, x_clean_j)

    TIDAK ada classifier loss — murni unsupervised.

    Parameter
    ----------
    cat_idx_array   : [N, n_cols]  — integer index semua kolom (bin+kategori)
    labels          : [N]          — DIABAIKAN (diterima untuk kompatibilitas API)
    cat_dims        : list[int]    — vocab size per kolom
    emb_sizes       : list[int]    — embedding size per kolom
    n_classes       : int          — DIABAIKAN (diterima untuk kompatibilitas API)
    corruption_prob : float        — fraksi fitur yang di-corrupt per sample
    corruption_type : str          — 'mask' | 'random_replace'

    Return
    ------
    DAEEmbeddingModel — parameter di-freeze untuk training diffusion downstream
    """
    model = DAEEmbeddingModel(
        cat_dims        = cat_dims,
        emb_sizes       = emb_sizes,
        dropout         = dropout,
        hidden_dim      = hidden_dim,
        corruption_prob = corruption_prob,
        corruption_type = corruption_type,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Loss L_H: cross-entropy rekonstruksi per kolom (eq. 2 & 5 paper)
    #
    # PERBAIKAN: Gunakan CrossEntropyLoss per kolom, BUKAN sigmoid + BCELoss.
    #
    # Setiap kolom merepresentasikan variabel kategoris one-of-K.
    # CrossEntropyLoss(logits [B, K], target [B]) = -log softmax(logits)[target]
    # ini benar untuk distribusi kategoris karena mengandung softmax internal
    # dan menjamin ΣP = 1 per kolom.
    #
    # sigmoid + BCE salah karena memperlakukan tiap bit secara independen
    # sehingga tidak ada normalisasi across vocab per kolom.
    ce_loss = nn.CrossEntropyLoss()

    # Hanya butuh x_clean — tidak butuh label
    cat_tensor = torch.tensor(cat_idx_array, dtype=torch.long, device=device)
    dataset    = torch.utils.data.TensorDataset(cat_tensor)
    cpu_gen    = torch.Generator(device='cpu')
    loader     = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = False,
        generator   = cpu_gen,
    )

    best_loss        = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f'[DAE] corruption_prob={corruption_prob}, '
          f'corruption_type={corruption_type}')

    model.train()
    for epoch in range(n_epochs):
        total_recon_loss = 0.0
        n_batches        = 0

        for (batch_cat,) in loader:
            optimizer.zero_grad()

            # Forward: x_clean → corrupt(x_oh) → f_θ → y → g_θ' → recon_logits
            # Corruption dilakukan DI DALAM model.forward() (hanya saat training)
            y, _, recon_logits = model(batch_cat)

            # Loss sesuai eq. 2 & 5 Vincent et al. (2008) — per kolom:
            #   CE(logits_j [B, K], target_j [B]) untuk setiap kolom j
            #
            # PERBAIKAN: target adalah integer index (bukan one-hot float)
            # CrossEntropyLoss menerima: logits [B, K] dan target [B] (int)
            # Ini menggantikan: one-hot scatter + BCELoss (yang salah untuk multi-class)
            recon_loss = 0.0
            for i in range(model.n_cols):
                # logits_i : [batch, vocab_size_i]  — RAW logits dari decoder
                # target_i : [batch]                — integer class index
                recon_loss = recon_loss + ce_loss(
                    recon_logits[i],          # [B, K] raw logits
                    batch_cat[:, i].long()    # [B]    integer target
                )
            recon_loss = recon_loss / model.n_cols

            recon_loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            n_batches        += 1

        avg_loss = total_recon_loss / n_batches

        if (epoch + 1) % 10 == 0:
            # Hitung reconstruction accuracy (denoising) pada batch terakhir
            # untuk monitoring kemampuan model merekonstruksi x dari x_tilde
            with torch.no_grad():
                correct_total = 0
                total_cols    = 0
                for i in range(model.n_cols):
                    pred_i = recon_logits[i].argmax(dim=1)   # argmax(raw_logits) = argmax(softmax)
                    true_i = batch_cat[:, i].long()
                    correct_total += (pred_i == true_i).sum().item()
                    total_cols    += batch_cat.shape[0]
                recon_acc = correct_total / total_cols if total_cols > 0 else 0.0
            print(f'[DAE] Epoch {epoch+1}/{n_epochs} - '
                  f'Reconstruction Loss: {avg_loss:.4f}  '
                  f'Batch Recon Acc (corrupted→clean): {recon_acc:.4f}')

        if avg_loss < best_loss:
            best_loss        = avg_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'[DAE] Early stopping triggered at epoch {epoch+1}')
            print(f'[DAE] Best reconstruction loss: {best_loss:.4f}')
            break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device)
                               for k, v in best_model_state.items()})
        print(f'[DAE] Loaded best model (epoch {epoch + 1 - patience_counter})')

    model.eval()

    with torch.no_grad():
        sample_cat = cat_tensor[:min(2048, len(cat_tensor))]
        # Inference: encode tanpa corruption
        z_sample = model.encode(sample_cat)
        print(f'[DAE] Distribusi embedding (N={z_sample.shape[0]}):')
        print(f'  mean={z_sample.mean().item():.4f}  '
              f'std={z_sample.std().item():.4f}  '
              f'norm_mean={z_sample.norm(dim=1).mean().item():.4f}')

    # ── Evaluasi denoising pada beberapa corruption level ──────────────────
    # Ini adalah evaluasi utama DAE sesuai Vincent et al. (2008):
    #   x_clean → corrupt(ν) → encode → decode → compare x_clean
    # Memverifikasi bahwa model benar-benar belajar: g(f(x̃)) ≈ x
    eval_sample = cat_tensor[:min(4096, len(cat_tensor))].cpu().numpy()
    evaluate_dae_denoising(
        model            = model,
        cat_idx_array    = eval_sample,
        device           = device,
        corruption_levels= [0.0, 0.1, 0.2, 0.3, 0.5],
        corruption_type  = corruption_type,
        verbose          = True,
    )

    for param in model.parameters():
        param.requires_grad_(False)
    print('[DAE] Seluruh parameter embedding di-freeze untuk training diffusion.')

    return model


# Alias untuk kompatibilitas pemanggilan di load_dataset
def train_supervised_embedding_model(cat_idx_array, labels, cat_dims, emb_sizes,
                                     n_classes, device, n_epochs=1000,
                                     batch_size=1024, lr=1e-3, dropout=0.1,
                                     hidden_dim=256, use_mlp=True, mlp_ratio=1.5,
                                     noise_std=0.01, patience=40):
    """
    Wrapper: memanggil train_dae_embedding_model.
    Signature identik dengan versi lama sehingga load_dataset tidak perlu diubah.
    corruption_prob dan corruption_type menggunakan default DAE (0.3, 'mask').
    """
    return train_dae_embedding_model(
        cat_idx_array   = cat_idx_array,
        labels          = labels,
        cat_dims        = cat_dims,
        emb_sizes       = emb_sizes,
        n_classes       = n_classes,
        device          = device,
        n_epochs        = n_epochs,
        batch_size      = batch_size,
        lr              = lr,
        dropout         = dropout,
        hidden_dim      = hidden_dim,
        use_mlp         = use_mlp,
        mlp_ratio       = mlp_ratio,
        noise_std       = noise_std,
        patience        = patience,
    )


# ===========================================================================
#  Evaluasi Kemampuan Denoising (INTI DAE — Vincent et al. 2008)
#
#  Mengukur kemampuan model merekonstruksi x_clean dari x_tilde sesuai
#  objective utama paper (eq. 5):
#      g(f(x̃)) ≈ x
#
#  Pipeline evaluasi yang benar (konsisten dengan training):
#      x_clean → corrupt(qD) → f_θ → g_θ' → compare dengan x_clean
#
#  Dua metrik utama:
#    1. reconstruction_loss : CE rata-rata per kolom (sama dengan training loss)
#    2. accuracy_per_feature : akurasi rekonstruksi per kolom
#
#  Dilakukan pada berbagai corruption level untuk memverifikasi bahwa
#  model benar-benar belajar denoising (bukan identity mapping).
# ===========================================================================

def evaluate_dae_denoising(model: DAEEmbeddingModel,
                           cat_idx_array: np.ndarray,
                           device: str,
                           corruption_levels: list = None,
                           corruption_type: str = 'mask',
                           batch_size: int = 4096,
                           verbose: bool = True) -> dict:
    """
    Evaluasi kemampuan denoising DAE sesuai objective Vincent et al. (2008).

    Skenario evaluasi (WAJIB sama dengan training):
        x_clean → corrupt(ν) → model.encode → model.decode → compare x_clean

    Parameter
    ---------
    model           : DAEEmbeddingModel — model sudah dilatih
    cat_idx_array   : [N, n_cols]  — data bersih (integer index)
    device          : str
    corruption_levels : list[float] — beberapa level corruption untuk dibandingkan
                        Default: [0.0, 0.1, 0.2, 0.3, 0.5]
                        0.0 = evaluasi dengan input bersih (baseline)
    corruption_type : str — 'mask' | 'random_replace'
    verbose         : bool — cetak ringkasan per level

    Return
    ------
    dict dengan key = corruption_level (float), value = dict berisi:
        'reconstruction_loss'  : float — CE loss rata-rata (konsisten dgn training)
        'overall_accuracy'     : float — akurasi rekonstruksi semua kolom
        'per_col_accuracy'     : np.ndarray [n_cols] — akurasi per kolom
        'n_samples'            : int
    """
    if corruption_levels is None:
        corruption_levels = [0.0, 0.1, 0.2, 0.3, 0.5]

    ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    cat_tensor = torch.tensor(cat_idx_array, dtype=torch.long, device=device)
    n_cols     = model.n_cols
    results    = {}

    model.eval()

    if verbose:
        print('\n' + '=' * 65)
        print(' Evaluasi Denoising DAE — Vincent et al. (2008)')
        print(' Skenario: x_clean → corrupt(ν) → encode → decode → compare x_clean')
        print('=' * 65)
        header = f"{'ν':>6} | {'CE Loss':>10} | {'Overall Acc':>12} | {'Per-Col Acc (min→max)':>28}"
        print(header)
        print('-' * 65)

    for nu in corruption_levels:
        total_ce_loss = 0.0
        n_batches     = 0
        col_correct   = np.zeros(n_cols, dtype=np.int64)
        col_total     = np.zeros(n_cols, dtype=np.int64)

        with torch.no_grad():
            for start in range(0, len(cat_tensor), batch_size):
                x_clean = cat_tensor[start : start + batch_size]   # [B, n_cols]

                # ── Step 1: Corrupt x_clean → x_tilde ─────────────────────
                # Jika ν=0 → x_tilde = x_clean (evaluasi input bersih / baseline)
                if nu > 0.0:
                    x_oh    = model._to_onehot(x_clean)             # [B, total_OH]
                    # Terapkan corruption manual dengan level ν
                    x_tilde_oh = x_oh.clone()
                    for col_i, n_cat in enumerate(model.cat_dims):
                        start_c = model._col_offsets[col_i]
                        end_c   = start_c + n_cat
                        col_corrupt = torch.bernoulli(
                            torch.full((x_clean.shape[0],), nu, device=device)
                        ).bool()
                        if not col_corrupt.any():
                            continue
                        if corruption_type == 'mask':
                            x_tilde_oh[col_corrupt, start_c:end_c] = 0.0
                        elif corruption_type == 'random_replace':
                            rand_idx = torch.randint(0, n_cat,
                                                     (int(col_corrupt.sum()),),
                                                     device=device)
                            rand_oh = torch.zeros(int(col_corrupt.sum()), n_cat,
                                                  device=device)
                            rand_oh.scatter_(1, rand_idx.unsqueeze(1), 1.0)
                            x_tilde_oh[col_corrupt, start_c:end_c] = rand_oh
                    # ── Step 2: Encode x_tilde → y ────────────────────────
                    x_tilde_oh = model.dropout(x_tilde_oh)
                    y           = torch.sigmoid(model.W_enc(x_tilde_oh))
                else:
                    # ν=0: encode langsung x_clean (tanpa corrupt)
                    y = model.encode(x_clean)

                # ── Step 3: Decode y → recon_logits ───────────────────────
                recon_logits = model.decode(y)   # list[n_cols] of [B, K_j]

                # ── Step 4: Hitung CE loss & accuracy vs x_clean ──────────
                batch_ce = 0.0
                for j in range(n_cols):
                    logits_j = recon_logits[j]              # [B, K_j] raw logits
                    target_j = x_clean[:, j].long()         # [B] clean integer index

                    # CE loss konsisten dengan training
                    batch_ce += ce_loss_fn(logits_j, target_j).item()

                    # Accuracy: argmax(raw logits) = argmax(softmax) — benar
                    pred_j = logits_j.argmax(dim=1)
                    col_correct[j] += (pred_j == target_j).sum().item()
                    col_total[j]   += x_clean.shape[0]

                total_ce_loss += batch_ce / n_cols
                n_batches     += 1

        avg_ce          = total_ce_loss / n_batches
        per_col_acc     = col_correct / np.maximum(col_total, 1)
        overall_acc     = col_correct.sum() / np.maximum(col_total.sum(), 1)

        results[nu] = {
            'reconstruction_loss' : avg_ce,
            'overall_accuracy'    : float(overall_acc),
            'per_col_accuracy'    : per_col_acc,
            'n_samples'           : int(col_total[0]) if len(col_total) > 0 else 0,
        }

        if verbose:
            acc_min = per_col_acc.min() if len(per_col_acc) > 0 else 0.0
            acc_max = per_col_acc.max() if len(per_col_acc) > 0 else 0.0
            tag = ' ← training ν' if abs(nu - 0.3) < 1e-6 else (
                  ' ← baseline'   if nu == 0.0 else '')
            print(f'{nu:>6.1f} | {avg_ce:>10.4f} | {overall_acc:>12.4f} | '
                  f'[{acc_min:.3f} → {acc_max:.3f}]{tag}')

    if verbose:
        print('=' * 65)
        nu0   = results.get(0.0, {})
        nu_tr = results.get(0.3, {})
        if nu0 and nu_tr:
            acc_drop = nu0['overall_accuracy'] - nu_tr['overall_accuracy']
            print(f'\nAcc drop bersih (ν=0 → ν=0.3): {acc_drop:+.4f}')
            print('Interpretasi:')
            print('  Jika acc(ν=0) >> acc(ν>0): model bergantung pada input bersih')
            print('  Jika acc(ν=0) ≈ acc(ν>0): model robust — benar-benar belajar denoising')
        print('')

    return results


# ===========================================================================
#  Encode / Decode helpers (TIDAK BERUBAH)
# ===========================================================================

def encode_with_embedding(model: DAEEmbeddingModel,
                          cat_idx_array: np.ndarray,
                          device: str,
                          batch_size: int = 4096) -> np.ndarray:
    """
    Encode integer index → embedding numpy array.
    Inference: encode langsung dari x_clean tanpa corruption.
    Kompatibel dengan DAEEmbeddingModel (forward mengembalikan (z, None, logits)).
    """
    model.eval()
    cat_tensor = torch.tensor(cat_idx_array, dtype=torch.long, device=device)
    dataset    = torch.utils.data.TensorDataset(cat_tensor)
    loader     = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )

    all_z = []
    with torch.no_grad():
        for (batch,) in loader:
            z, _, _ = model(batch, add_noise=False)
            all_z.append(z.cpu().numpy())

    return np.concatenate(all_z, axis=0).astype(np.float32)


def decode_cat_from_embedding(model: DAEEmbeddingModel,
                              emb_array: np.ndarray,
                              device: str,
                              batch_size: int = 4096) -> np.ndarray:
    """
    Decode representasi laten y → prediksi kelas tiap kolom (argmax logits).

    Mengimplementasikan g_θ' dari paper (Section 2.2):
        z_raw = W'y + b'  (RAW logits, tanpa aktivasi)
    kemudian argmax per slice kolom diambil sebagai prediksi kategori.

    PERBAIKAN: Decoder kini mengeluarkan raw logits (bukan sigmoid output).
    argmax(raw_logits) = argmax(softmax(logits)) — ekuivalen dan benar
    untuk prediksi kelas kategoris.

    emb_array : [N, hidden_dim]  — output encode() / representasi laten y
    Return    : [N, n_cols]      — predicted integer index
    """
    model.eval()
    emb_tensor = torch.tensor(emb_array, dtype=torch.float32, device=device)
    dataset    = torch.utils.data.TensorDataset(emb_tensor)
    loader     = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )

    all_pred = []
    with torch.no_grad():
        for (batch,) in loader:
            recon_logits = model.decode(batch)
            pred_idx = torch.stack([
                torch.argmax(logits, dim=1)
                for logits in recon_logits
            ], dim=1)
            all_pred.append(pred_idx.cpu().numpy())

    return np.concatenate(all_pred, axis=0).astype(np.int64)


def decode_num_from_embedding(model: DAEEmbeddingModel,
                              emb_array: np.ndarray,
                              bin_midpoints: list,
                              n_num_cols: int,
                              device: str,
                              batch_size: int = 4096) -> np.ndarray:
    """
    Decode representasi laten y → nilai numerik kontinu (skala normalisasi).

    Mengimplementasikan g_θ' dari paper (Section 2.2) untuk kolom numerik:
        z_raw = W'y + b'  (RAW logits) → slice per kolom numerik
        → softmax(z_raw_col) → weighted sum atas midpoints bin

    PERBAIKAN: Decoder kini mengeluarkan raw logits (bukan sigmoid).
    softmax(raw_logits) benar karena menghasilkan distribusi probabilitas
    yang valid (ΣP=1) untuk weighted sum atas bin midpoints.
    Sebelumnya sigmoid(logits) tidak menjamin ΣP=1 sehingga weighted sum
    tidak memiliki interpretasi probabilistik yang valid.

    Alur (Weighted Sum / Soft Decode):
        y → g_θ'(y) → z_raw_col [N, n_bins] → softmax → weighted sum @ mids

    Kolom numerik diasumsikan berada di AWAL (indeks 0..n_num_cols-1).

    Parameter
    ---------
    model         : DAEEmbeddingModel
    emb_array     : [N, hidden_dim]  — output encode() (representasi laten y)
    bin_midpoints : list[n_num_cols] of np.ndarray  — midpoint per bin, skala norm
    n_num_cols    : int  — jumlah kolom numerik
    device        : str

    Return : np.ndarray [N, n_num_cols]  — nilai kontinu skala normalisasi
    """
    model.eval()
    emb_tensor = torch.tensor(emb_array, dtype=torch.float32, device=device)
    dataset    = torch.utils.data.TensorDataset(emb_tensor)
    loader     = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )

    all_preds = []
    with torch.no_grad():
        for (batch,) in loader:
            recon_logits = model.decode(batch)  # list[n_cols] of [B, vocab_size_i]

            batch_num_preds = []
            for col in range(n_num_cols):
                logits  = recon_logits[col]                          # [B, n_bins_col]
                probs   = torch.softmax(logits, dim=1)               # [B, n_bins_col]
                mids_t  = torch.tensor(
                    bin_midpoints[col], dtype=torch.float32, device=device
                )                                                     # [n_bins_col]
                # Weighted sum: probs @ mids → [B]
                pred_col = (probs * mids_t.unsqueeze(0)).sum(dim=1)  # [B]
                batch_num_preds.append(pred_col.unsqueeze(1))         # [B, 1]

            # Stack semua kolom numerik → [B, n_num_cols]
            batch_num_preds = torch.cat(batch_num_preds, dim=1)
            all_preds.append(batch_num_preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0).astype(np.float32)


# ===========================================================================
#  Load Dataset
# ===========================================================================

def load_dataset(dataname, idx=0, mask_type='MCAR', ratio='30', noise_std=0.01):
    """
    Load dataset dengan MRmD discretization untuk numerik +
    DAE Embedding untuk SEMUA kolom (numerik-bin + kategorikal).

    Perubahan dari versi MDLP ke MRmD:
    - Fitur numerik di-diskritisasi dengan MRmD → integer bin index
    - Bin index numerik di-embed BERSAMA kolom kategorikal (posisi pertama)
    - Pipeline embedding → normalisasi → diffusion → imputasi TIDAK BERUBAH
    - train_num / test_num tetap dikembalikan (nilai float asli, ternormalisasi)
      untuk keperluan evaluasi MAE/RMSE di skala normalisasi

    Output tambahan (dibanding versi sebelumnya):
    - mrmd          : MRmDDiscretizer  (untuk transform test & decode)
    - bin_midpoints : list[n_num_cols] — midpoint bin dalam skala normalisasi
    - n_num_cols    : int
    - t_mrmd        : float — waktu komputasi MRmD discretization (detik)
    - t_emb         : float — waktu komputasi embedding training (detik)

    Return
    ------
    train_X           : [N_train, total_emb_dim]           float32
    test_X            : [N_test,  total_emb_dim]           float32
    ori_train_mask    : mask asli train [N_train, total_cols]
    ori_test_mask     : mask asli test  [N_test,  total_cols]
    train_num         : [N_train, n_num_cols]  — float asli (ternormalisasi)
    test_num          : [N_test,  n_num_cols]
    train_all_idx     : [N_train, n_num_cols + n_cat_cols]  — semua bin/label idx
    test_all_idx      : [N_test,  n_num_cols + n_cat_cols]
    extend_train_mask : [N_train, total_emb_dim]
    extend_test_mask  : [N_test,  total_emb_dim]
    cat_bin_num       : None  (legacy)
    emb_model         : DAEEmbeddingModel
    emb_sizes         : list[int]
    mrmd              : MRmDDiscretizer  (atau None jika tidak ada fitur numerik)
    bin_midpoints     : list[n_num_cols] of np.ndarray  (atau None)
    n_num_cols        : int
    t_mrmd            : float — waktu komputasi MRmD discretization (detik)
    t_emb             : float — waktu komputasi embedding training (detik)
    """
    ratio = str(ratio)

    data_dir  = f'datasets/{dataname}'
    info_path = f'datasets/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx    = info['num_col_idx']
    cat_col_idx    = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    data_path       = f'{data_dir}/data.csv'
    train_path      = f'{data_dir}/train.csv'
    test_path       = f'{data_dir}/test.csv'
    train_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
    test_mask_path  = f'{data_dir}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy'

    data_df  = pd.read_csv(data_path)
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    train_mask = np.load(train_mask_path)
    test_mask  = np.load(test_mask_path)

    cols = train_df.columns

    # ── Fitur numerik (nilai float asli) ─────────────────────────────────
    data_num  = data_df[cols[num_col_idx]].values.astype(np.float32)
    train_num_raw = train_df[cols[num_col_idx]].values.astype(np.float32)
    test_num_raw  = test_df[cols[num_col_idx]].values.astype(np.float32)

    # ── Labels untuk supervised learning ─────────────────────────────────
    train_y = train_df[cols[target_col_idx]]
    test_y  = test_df[cols[target_col_idx]]

    label_encoder = LabelEncoder()
    all_labels    = pd.concat([train_y, test_y]).values.ravel()
    label_encoder.fit(all_labels.astype(str))

    train_labels = label_encoder.transform(train_y.values.ravel().astype(str))
    test_labels  = label_encoder.transform(test_y.values.ravel().astype(str))
    n_classes    = len(label_encoder.classes_)

    print(f'[Dataset] Detected {n_classes} classes for supervised learning')
    print(f'[Dataset] Classes: {label_encoder.classes_}')

    # ── Normalisasi numerik (untuk evaluasi MAE/RMSE & bin midpoints) ─────
    # Normalisasi dihitung dari observed entries train (mask=False → observed)
    n_num_cols = len(num_col_idx)

    if n_num_cols > 0:
        num_mask_train = train_mask[:, num_col_idx].astype(bool)
        mask_obs       = (~num_mask_train).astype(np.float32)
        mask_sum       = mask_obs.sum(0)
        mask_sum[mask_sum == 0] = 1.0

        num_mean = (train_num_raw * mask_obs).sum(0) / mask_sum
        num_var  = ((train_num_raw - num_mean) ** 2 * mask_obs).sum(0) / mask_sum
        num_std  = np.sqrt(num_var)
        num_std[num_std == 0] = 1.0

        # Skala normalisasi: (X - mean) / std
        train_num_norm = (train_num_raw - num_mean) / num_std
        test_num_norm  = (test_num_raw  - num_mean) / num_std

        # Simpan untuk dikembalikan (dipakai get_eval)
        train_num = train_num_norm.astype(np.float32)
        test_num  = test_num_norm.astype(np.float32)

        # ── MRmD Discretization ─────────────────────────────────────────
        print(f'[MRmD] Menjalankan MRmD discretization pada {n_num_cols} kolom numerik ...')
        mrmd = MRmDDiscretizer(val_size=0.125, N_D=50, random_state=42, verbose=False)

        # Fit MRmD pada data RAW train (bukan normalisasi) dengan label
        t_mrmd_start = time.time()
        mrmd.fit(train_num_raw, train_labels)

        train_num_bin = mrmd.transform(train_num_raw)   # [N_train, n_num_cols] int64
        test_num_bin  = mrmd.transform(test_num_raw)    # [N_test,  n_num_cols] int64

        # Hitung bin midpoints dalam skala NORMALISASI
        # (dipakai saat decoding: bin index → nilai kontinu untuk MAE/RMSE)
        bin_midpoints = mrmd.get_bin_midpoints(train_num_norm, train_num_bin)
        t_mrmd_end = time.time()
        t_mrmd = t_mrmd_end - t_mrmd_start

        print(f'[MRmD] n_bins per kolom: {mrmd.n_bins_}')
        print(f'[MRmD] Total bins: {sum(mrmd.n_bins_)}')
        print(f'[MRmD] Waktu komputasi diskritisasi: {t_mrmd:.4f}s')

    else:
        # Tidak ada fitur numerik
        train_num     = np.zeros((len(train_df), 0), dtype=np.float32)
        test_num      = np.zeros((len(test_df),  0), dtype=np.float32)
        train_num_bin = np.zeros((len(train_df), 0), dtype=np.int64)
        test_num_bin  = np.zeros((len(test_df),  0), dtype=np.int64)
        bin_midpoints = []
        mrmd          = None
        t_mrmd        = 0.0

    # ── Encoding kolom kategorikal ────────────────────────────────────────
    cat_dims_cat           = []
    train_cat_idx_list     = []
    test_cat_idx_list      = []

    if len(cat_col_idx) > 0:
        cat_columns = cols[cat_col_idx]
        data_cat    = data_df[cat_columns].astype(str)
        train_cat   = train_df[cat_columns].astype(str)
        test_cat    = test_df[cat_columns].astype(str)

        encoders = {}
        for col in cat_columns:
            le = LabelEncoder()
            le.fit(data_cat[col])
            encoders[col] = le
            cat_dims_cat.append(len(le.classes_))
            train_cat_idx_list.append(
                le.transform(train_cat[col]).astype(np.int64)
            )
            test_cat_idx_list.append(
                le.transform(test_cat[col]).astype(np.int64)
            )

        train_cat_idx = np.stack(train_cat_idx_list, axis=1)
        test_cat_idx  = np.stack(test_cat_idx_list,  axis=1)
    else:
        train_cat_idx = np.zeros((len(train_df), 0), dtype=np.int64)
        test_cat_idx  = np.zeros((len(test_df),  0), dtype=np.int64)

    # ── Gabungkan: [num_bin | cat_idx] → satu array idx untuk embedding ──
    # Urutan: numerik (bin) DULU, lalu kategorikal — konsisten di seluruh pipeline
    if n_num_cols > 0 and len(cat_col_idx) > 0:
        train_all_idx = np.concatenate([train_num_bin, train_cat_idx], axis=1)
        test_all_idx  = np.concatenate([test_num_bin,  test_cat_idx],  axis=1)
    elif n_num_cols > 0:
        train_all_idx = train_num_bin
        test_all_idx  = test_num_bin
    else:
        train_all_idx = train_cat_idx
        test_all_idx  = test_cat_idx

    # ── Dimensi embedding ─────────────────────────────────────────────────
    # Numerik: n_bins per kolom; kategorikal: n_unique per kolom
    all_dims = (mrmd.n_bins_ if mrmd is not None else []) + cat_dims_cat
    emb_sizes = [compute_embedding_size(n) for n in all_dims]

    print(f'[Embedding] all_dims (num_bin+cat)={all_dims}')
    print(f'[Embedding] emb_sizes={emb_sizes}, total_emb_dim={sum(emb_sizes)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Latih DAEEmbeddingModel ───────────────────────────────────────────
    # Input: semua kolom (numerik bin + kategorikal) sebagai integer index
    # Objective: unsupervised denoising (TIDAK ada classifier loss)
    # Sesuai Vincent et al. (2008): min E[L(x, g(f(x_tilde)))]
    print('[DAE] Melatih DAEEmbeddingModel '
          '(denoising reconstruction loss, unsupervised) ...')
    t_emb_start = time.time()
    print(noise_std)
    emb_model = train_supervised_embedding_model(
        cat_idx_array = train_all_idx,
        labels        = train_labels,
        cat_dims      = all_dims,
        emb_sizes     = emb_sizes,
        n_classes     = n_classes,
        device        = device,
        n_epochs      = 1000,
        batch_size    = 1024,
        lr            = 1e-3,
        dropout       = 0.1,
        hidden_dim    = 256,
        use_mlp       = True,
        mlp_ratio     = 1.5,
        noise_std     = noise_std,
        patience      = 40,
    )
    t_emb_end = time.time()
    t_emb = t_emb_end - t_emb_start
    print('[DAE] Training selesai. Parameter di-freeze untuk diffusion.')
    print(f'[Embedding] Waktu komputasi embedding: {t_emb:.4f}s')

    # ── Encode semua kolom → embedding vector ────────────────────────────
    # [TIDAK BERUBAH] — encode_with_embedding sama persis
    train_all_emb = encode_with_embedding(emb_model, train_all_idx, device)
    test_all_emb  = encode_with_embedding(emb_model, test_all_idx,  device)
    # shape: [N, total_emb_dim]

    # ── train_X / test_X sekarang HANYA embedding (tidak ada kolom raw num) ─
    # Karena numerik sudah masuk embedding, len_num = 0 di main
    train_X = train_all_emb
    test_X  = test_all_emb

    # ── Buat extended mask ────────────────────────────────────────────────
    # Mask asli: [N, total_original_cols]
    # Extended mask: [N, total_emb_dim] — diperluas sesuai emb_sizes
    train_num_mask = train_mask[:, num_col_idx].astype(bool) if n_num_cols > 0 else np.zeros((len(train_df), 0), dtype=bool)
    train_cat_mask = train_mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else np.zeros((len(train_df), 0), dtype=bool)
    test_num_mask  = test_mask[:, num_col_idx].astype(bool)  if n_num_cols > 0 else np.zeros((len(test_df),  0), dtype=bool)
    test_cat_mask  = test_mask[:, cat_col_idx].astype(bool)  if len(cat_col_idx) > 0 else np.zeros((len(test_df),  0), dtype=bool)

    # Gabungkan mask: [num_mask | cat_mask] — urutan sama dengan all_dims
    if n_num_cols > 0 and len(cat_col_idx) > 0:
        train_all_mask = np.concatenate([train_num_mask, train_cat_mask], axis=1)
        test_all_mask  = np.concatenate([test_num_mask,  test_cat_mask],  axis=1)
    elif n_num_cols > 0:
        train_all_mask = train_num_mask
        test_all_mask  = test_num_mask
    else:
        train_all_mask = train_cat_mask
        test_all_mask  = test_cat_mask

    emb_sizes_arr = np.array(emb_sizes, dtype=int)

    def extend_mask_emb(mask: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        """
        Perluas mask [N, n_cols] → [N, total_emb_dim].
        Kolom ke-j diperluas ke sizes[j] dimensi.
        [TIDAK BERUBAH]
        """
        N      = mask.shape[0]
        cum    = np.concatenate(([0], sizes.cumsum()))
        result = np.zeros((N, sizes.sum()), dtype=bool)
        for j in range(len(sizes)):
            col_mask = mask[:, j][:, np.newaxis]
            result[:, cum[j]:cum[j + 1]] = np.tile(col_mask, sizes[j])
        return result

    # extend_train_mask = extend_mask_emb(train_all_mask, emb_sizes_arr)
    # extend_test_mask  = extend_mask_emb(test_all_mask,  emb_sizes_arr)
    # train_X / test_X adalah output laten DAE [N, hidden_dim=256]
    # Mask harus diperluas ke hidden_dim, bukan sum(emb_sizes)
    hidden_dim = train_X.shape[1]  # ambil dari train_X langsung

    def extend_mask_to_hidden(mask_cols: np.ndarray, n_hidden: int) -> np.ndarray:
        """
        Dari mask [N, n_cols] → [N, n_hidden].
        Sample di-mark missing jika ANY kolomnya missing.
        """
        any_missing = mask_cols.any(axis=1, keepdims=True)  # [N, 1]
        return np.tile(any_missing, (1, n_hidden))           # [N, hidden_dim]

    extend_train_mask = extend_mask_to_hidden(train_all_mask, hidden_dim)
    extend_test_mask  = extend_mask_to_hidden(test_all_mask,  hidden_dim)

    # Hitung bin_midpoints dalam skala normalisasi (dibutuhkan get_eval)
    # Sudah dihitung di atas, disimpan di mrmd.bin_midpoints_ & bin_midpoints

    return (train_X, test_X,
            train_mask, test_mask,
            train_num, test_num,
            train_all_idx, test_all_idx,
            extend_train_mask, extend_test_mask,
            None,          # cat_bin_num (legacy)
            emb_model,
            emb_sizes,
            mrmd,          # [BARU] MRmDDiscretizer
            bin_midpoints, # [BARU] list[n_num_cols] midpoint per bin, skala norm
            n_num_cols,    # [BARU] jumlah kolom numerik
            t_mrmd,        # [BARU] waktu komputasi MRmD discretization (detik)
            t_emb)         # [BARU] waktu komputasi embedding training (detik)


def mean_std(data, mask):
    mask      = (~mask).astype(np.float32)
    mask_sum  = mask.sum(0)
    mask_sum[mask_sum == 0] = 1
    mean      = (data * mask).sum(0) / mask_sum
    var       = ((data - mean) ** 2 * mask).sum(0) / mask_sum
    std       = np.sqrt(var)
    return mean, std


# ===========================================================================
#  Evaluasi
# ===========================================================================

def get_eval(dataname, X_recon, X_true, truth_all_idx,
             num_num, emb_model, emb_sizes, mask,
             device='cpu', oos=False,
             bin_midpoints=None, n_num_cols=0,
             num_true_norm=None):
    """
    Hitung MAE, RMSE (numerik) dan Accuracy (kategorikal).

    CATATAN PENTING — Konsistensi Training vs Evaluasi (Vincent et al. 2008):
    ──────────────────────────────────────────────────────────────────────────
    Fungsi ini mengevaluasi hasil rekonstruksi SETELAH pipeline imputasi lengkap
    (diffusion/downstream). Untuk evaluasi kemampuan denoising DAE secara murni,
    gunakan fungsi `evaluate_dae_denoising()` yang mengikuti skenario benar:
        x_clean → corrupt(ν) → encode → decode → compare x_clean

    Fungsi ini (get_eval) memverifikasi kualitas imputasi akhir dengan metrik:
      - MAE/RMSE di skala normalisasi (untuk fitur numerik)
      - Accuracy rekonstruksi (untuk fitur kategorikal)
    pada posisi-posisi yang missing (mask=True).

    PERBAIKAN: decode_cat_from_embedding kini menggunakan argmax(raw_logits)
    yang benar (setelah decoder diperbaiki dari sigmoid → raw logits).

    [MODIFIKASI] Numerik sekarang di-embed bersama kategorikal.
    MAE/RMSE dihitung di skala normalisasi menggunakan ground truth
    nilai asli (bukan midpoint bin) yang dipass via num_true_norm.

    Konvensi input:
    ---------------
    X_recon / X_true : [N, total_emb_dim]
        Seluruh dimensi adalah embedding. Tidak ada kolom raw numerik.

    Numerik (MAE/RMSE):
        decode_num_from_embedding → bin index → midpoint (skala norm) [prediksi]
        Ground truth: num_true_norm — nilai float asli ternormalisasi (skala norm)
        MAE/RMSE dihitung di skala (X-mean)/std (normalisasi).

    Kategorikal (Accuracy):
        decode_cat_from_embedding → argmax(raw_logits) → dibandingkan truth_all_idx
        Sama persis dengan versi sebelumnya, hanya offset kolom bergeser
        karena kolom numerik (bin) ada di awal.

    Parameter
    ---------
    bin_midpoints  : list[n_num_cols] of np.ndarray  — midpoint per bin, skala norm
                     (dipakai untuk decode prediksi)
    n_num_cols     : int — jumlah kolom numerik
    num_num        : int — DIABAIKAN (legacy, selalu 0 di pipeline baru ini)
                          dipertahankan untuk kompatibilitas signature
    truth_all_idx  : [N, n_num_cols + n_cat_cols]  integer index (bin + label)
    num_true_norm  : [N, n_num_cols] float — nilai numerik asli ternormalisasi
                     (skala (X-mean)/std). Jika None, fallback ke midpoint bin.
    """
    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    # mask: True(1) = missing, False(0) = observed
    num_mask = mask[:, num_col_idx].astype(bool) if len(num_col_idx) > 0 else None
    cat_mask = mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else None

    # ── Special case: news dataset ────────────────────────────────────────
    if dataname == 'news' and oos:
        drop = 6265
        if num_mask is not None:
            num_mask = np.delete(num_mask, drop, axis=0)
        if cat_mask is not None:
            cat_mask = np.delete(cat_mask, drop, axis=0)
        if truth_all_idx is not None:
            truth_all_idx = np.delete(truth_all_idx, drop, axis=0)
        if num_true_norm is not None:
            num_true_norm = np.delete(num_true_norm, drop, axis=0)
        X_recon = np.delete(X_recon, drop, axis=0)
        X_true  = np.delete(X_true,  drop, axis=0)

    # ── Numerik: MAE & RMSE di skala normalisasi ─────────────────────────
    mae  = np.nan
    rmse = np.nan

    if (n_num_cols > 0
            and num_mask is not None
            and bin_midpoints is not None
            and emb_model is not None):

        # Decode embedding → nilai kontinu prediksi (skala normalisasi) via bin midpoints
        num_pred_norm = decode_num_from_embedding(
            emb_model, X_recon, bin_midpoints, n_num_cols, device
        )  # [N, n_num_cols]

        # Ground truth: gunakan nilai asli ternormalisasi (num_true_norm) jika tersedia.
        # Ini adalah nilai float asli (X - mean) / std, bukan midpoint bin.
        # Fallback ke midpoint bin hanya jika num_true_norm tidak dipass.
        if num_true_norm is not None:
            # Pastikan shape cocok (news dataset bisa ada row yang di-drop)
            gt_norm = num_true_norm
        else:
            # Fallback (legacy): lookup midpoint dari true bin index
            N = X_true.shape[0]
            gt_norm = np.zeros((N, n_num_cols), dtype=np.float32)
            for col in range(n_num_cols):
                mids     = bin_midpoints[col]
                true_bin = truth_all_idx[:, col].astype(int)
                true_bin = np.clip(true_bin, 0, len(mids) - 1)
                gt_norm[:, col] = mids[true_bin]

        # Hitung MAE & RMSE hanya pada posisi missing
        diff = num_pred_norm[num_mask] - gt_norm[num_mask]
        mae  = float(np.abs(diff).mean())
        rmse = float(np.sqrt((diff ** 2).mean()))

    # ── Kategorikal: Akurasi via Linear Decoder ───────────────────────────
    # [TIDAK BERUBAH] — logika sama, hanya offset kolom bergeser
    acc = np.nan
    if (truth_all_idx is not None
            and len(cat_col_idx) > 0
            and emb_model is not None
            and emb_sizes is not None
            and cat_mask is not None):

        # Decode semua kolom → predicted index
        pred_all_idx = decode_cat_from_embedding(
            emb_model, X_recon, device
        )  # [N, n_num_cols + n_cat_cols]

        # Kolom kategorikal berada di offset n_num_cols (setelah numerik)
        n_cat_cols    = len(cat_col_idx)
        correct_total = 0
        total_missing = 0

        for j in range(n_cat_cols):
            rows_miss = cat_mask[:, j]
            if rows_miss.sum() == 0:
                continue

            col_offset = n_num_cols + j      # offset di array all_idx

            pred_j = pred_all_idx[:, col_offset]
            true_j = truth_all_idx[:, col_offset].astype(int)

            correct = (pred_j[rows_miss] == true_j[rows_miss]).sum()
            correct_total += int(correct)
            total_missing += int(rows_miss.sum())

        if total_missing > 0:
            acc = correct_total / total_missing

    return mae, rmse, acc