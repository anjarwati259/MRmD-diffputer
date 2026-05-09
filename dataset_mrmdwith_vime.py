import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import json
import time
import pickle

# MRmD helper functions (inline dari mrmd_discretizer.py)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

DATA_DIR = 'datasets'

# ===========================================================================
#  Supervised Learnable Embedding Model dengan Label
#  Arsitektur: nn.Embedding per kolom kategorikal + MLP Classifier untuk prediksi label
#  Konsep: Neural Network Embedding yang dilatih secara supervised
#
#  [DISESUAIKAN] Alur encoding/decoding disejajarkan dengan versi unsupervised
#  agar perbandingan adil (apple-to-apple):
#    - Tambah 1 hidden layer opsional (Linear → SiLU → Linear) setelah concat
#    - Tambah LayerNorm setelah concat/MLP (stabilisasi skala sebelum diffusion)
#    - Tambah Gaussian noise kecil (σ≈0.01) sebelum decoding saat training
#    - Freeze seluruh parameter embedding setelah pretraining
#  Classification loss (alpha * class_loss) TETAP dipertahankan.
#
#  [BARU] Fitur numerik di-diskritisasi dengan MRmD lalu di-embed bersama
#  fitur kategorikal menggunakan SupervisedLearnableEmbeddingModel yang sama.
#  Pipeline dari embedding → imputasi TIDAK BERUBAH.
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

    def summary(self):
        """Cetak tabel ringkasan cut points."""
        check_is_fitted(self, 'cut_points_')
        print('=' * 60)
        print(f"{'MRmD Discretizer — Summary':^60}")
        print('=' * 60)
        print(f"  {'Fitur':<20} {'# Bins':>7}   Cut Points")
        print('  ' + '-' * 56)
        for j, cp in enumerate(self.cut_points_):
            name   = str(self.feature_names_in_[j])[:20] if hasattr(self, 'feature_names_in_') else f'fitur_{j}'
            cp_str = np.round(cp, 4).tolist() if len(cp) > 0 else '[ ]'
            print(f'  {name:<20} {len(cp)+1:>7}   {cp_str}')
        print('=' * 60)
        print(f'  Total cut points: {sum(len(c) for c in self.cut_points_)}')
        print(f'  N_D={self.N_D}, val_size={self.val_size}')
        print('=' * 60)


# ===========================================================================
#  VIME Embedding Model
#
#  Menggantikan SupervisedLearnableEmbeddingModel dengan VIME self-supervised
#  encoder sesuai paper "VIME: Extending the Success of Self- and
#  Semi-supervised Learning to Tabular Domain" (NeurIPS 2020) dan
#  implementasi resmi di https://github.com/jsyoon0823/VIME.
#
#  Arsitektur (sesuai paper & GitHub resmi):
#    - Encoder (e)           : Linear → ReLU → Linear
#    - Feature Estimator (sr): Linear → ReLU → Linear
#    - Mask Estimator (sm)   : Linear → ReLU → Linear → Sigmoid
#
#  Pretext tasks (sesuai paper Section 4.1):
#    (1) Mask vector estimation  : BCE loss antara m dan m_hat
#    (2) Feature vector estimation: MSE loss antara x dan x_hat
#
#  Loss function (Eq. 4 paper):
#    L = lm(m, m_hat) + alpha * lr(x, x_hat)
#    - lm : binary cross-entropy per dimensi (Eq. 5)
#    - lr : MSE reconstruction loss (Eq. 6)
#         → cross-entropy untuk fitur kategorikal (bin index)
#
#  Corrupt generation (Eq. 3 paper):
#    x_tilde = m ⊙ x_bar + (1 - m) ⊙ x
#    di mana x_bar[j] ~ empirical marginal distribution fitur ke-j
#
#  Input pipeline:
#    cat_idx_array [N, n_cols]  — integer bin/label index (MRmD + kategorikal)
#    Sebelum masuk VIME encoder, index di-one-hot encode → float tensor
#    sehingga input_dim = sum(all_dims) (total one-hot size)
#
#  Output (encode):
#    z [N, hidden_dim]  — representasi laten (output encoder e)
#    Dipakai sebagai "embedding" pengganti SupervisedLearnableEmbeddingModel.
#
#  Decode (untuk evaluasi kategorikal/numerik):
#    sr decoder → logits [N, input_dim] → split per kolom → argmax / weighted sum
# ===========================================================================

def compute_embedding_size(n_categories: int) -> int:
    """
    Ukuran hidden dim VIME encoder (dipakai juga sebagai 'emb_size' per kolom
    dalam konteks pipeline lama untuk kompatibilitas).
    Rumus: min(600, round(1.6 * n_categories^0.56))
    Referensi: Guo & Berkhahn (2016) — dipertahankan untuk konsistensi pipeline.
    """
    return min(600, round(1.6 * n_categories ** 0.56))


class VIMEEmbeddingModel(nn.Module):
    """
    VIME Self-Supervised Encoder untuk data tabular.

    Sesuai paper NeurIPS 2020 (Section 4.1) dan GitHub resmi jsyoon0823/VIME.

    Komponen utama:
      - Encoder e          : input_dim → hidden_dim → hidden_dim  (Linear→ReLU→Linear)
      - Feature Estimator sr: hidden_dim → hidden_dim → input_dim  (Linear→ReLU→Linear)
      - Mask Estimator sm   : hidden_dim → hidden_dim → input_dim  (Linear→ReLU→Linear→Sigmoid)

    Input:
      x_float [batch, input_dim]  — one-hot encoded tabular features (float)

    Corrupt generation (Eq. 3):
      x_tilde = m ⊙ x_bar + (1-m) ⊙ x
      m_j ~ Bernoulli(p_m) per fitur
      x_bar[j] ~ empirical marginal (sampled dari batch/dataset)

    Loss (Eq. 4-6):
      L = lm(m, m_hat) + alpha * lr(x, x_hat)
      lm = BCE per dimensi (mask estimation loss)
      lr = MSE per dimensi untuk numerik,
           CrossEntropy per kolom untuk kategorikal (bin index)

    Untuk kompatibilitas dengan pipeline decode (get_eval):
      - self.cat_dims  : list[n_cols] — vocab size tiap kolom
      - self.emb_sizes : list[n_cols] — semuanya = hidden_dim (placeholder)
      - decode(z)      : z [N, hidden_dim] → list[n_cols] logits
                         via linear projection sr_out → split per kolom
    """

    def __init__(self, all_dims: list, hidden_dim: int, p_m: float = 0.3,
                 alpha: float = 1.0):
        """
        Parameter
        ---------
        all_dims   : list[n_cols] — vocab size tiap kolom (n_bins atau n_unique)
        hidden_dim : int          — ukuran hidden layer encoder
        p_m        : float        — probabilitas masking Bernoulli (paper: p_m)
        alpha      : float        — bobot feature reconstruction loss (Eq. 4)
        """
        super().__init__()

        self.all_dims   = all_dims
        self.n_cols     = len(all_dims)
        self.input_dim  = sum(all_dims)   # total one-hot size
        self.hidden_dim = hidden_dim
        self.p_m        = p_m
        self.alpha      = alpha

        # Offset untuk split one-hot per kolom
        self._offsets = [0] + list(np.cumsum(all_dims))

        # Placeholder agar kompatibel dengan pipeline lama (decode, get_eval)
        self.cat_dims  = all_dims
        self.emb_sizes = [hidden_dim] * self.n_cols   # tidak dipakai untuk decode
        self.total_emb_dim = hidden_dim
        self.out_dim   = hidden_dim

        # ── Encoder e: Linear → ReLU → Linear ────────────────────────────
        # Sesuai paper Section 4.1 dan GitHub jsyoon0823/VIME
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Feature Estimator sr: Linear → ReLU → Linear ─────────────────
        # sr: Z → X̂  (output = input_dim, yaitu full one-hot size)
        # Sesuai paper: sr merekonstruksi seluruh vektor fitur x
        self.feature_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
        )

        # ── Mask Estimator sm: Linear → ReLU → Linear → Sigmoid ──────────
        # [PERBAIKAN BUG #1] Paper Eq. 5 & Figure 1:
        #   sm: Z → [0,1]^d  di mana d = n_cols (jumlah kolom/fitur)
        #   BUKAN input_dim (total one-hot size).
        #   Mask m ∈ {0,1}^d — satu bit per KOLOM, bukan per dimensi one-hot.
        self.mask_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_cols),   # output = n_cols sesuai paper
            nn.Sigmoid(),
        )

    # ── One-hot helpers ───────────────────────────────────────────────────

    def idx_to_onehot(self, x_idx: torch.Tensor) -> torch.Tensor:
        """
        Convert integer index per kolom → one-hot concatenated float tensor.
        x_idx  : [batch, n_cols]  int
        return : [batch, input_dim]  float
        """
        parts = []
        for j in range(self.n_cols):
            parts.append(
                torch.nn.functional.one_hot(
                    x_idx[:, j].long(),
                    num_classes=self.all_dims[j]
                ).float()
            )
        return torch.cat(parts, dim=1)   # [batch, input_dim]

    # ── Corrupt generation (Eq. 3, paper Section 4.1) ─────────────────────

    def corrupt(self, x_onehot: torch.Tensor) -> tuple:
        """
        Generate corrupted sample x_tilde dan mask m.

        [PERBAIKAN BUG #2 — sesuai Paper Section 4.1 & Eq. 3]
        Paper: mj ~ Bernoulli(pm) untuk setiap fitur/kolom ke-j (j=1..d).
        Mask m ∈ {0,1}^d — SATU bit per KOLOM, bukan per dimensi one-hot.

        Untuk one-hot encoded input:
          - Jika kolom j di-mask (mj=1): seluruh dimensi one-hot kolom j
            diganti dengan x_bar[j] (sampel dari empirical marginal)
          - Jika kolom j tidak di-mask (mj=0): dimensi one-hot kolom j
            dipertahankan dari x asli

        x_tilde = m_expanded ⊙ x_bar + (1 - m_expanded) ⊙ x  (Eq. 3)

        Return:
          x_tilde     : [batch, input_dim]  — corrupted sample
          m_col       : [batch, n_cols]     — mask per kolom (untuk loss lm, Eq. 5)
        """
        batch_size = x_onehot.shape[0]
        device     = x_onehot.device

        # [PERBAIKAN] Mask per KOLOM sesuai paper: mj ~ Bernoulli(pm)
        # Shape: [batch, n_cols]  — satu bit per kolom
        m_col = torch.bernoulli(
            torch.full((batch_size, self.n_cols), self.p_m, device=device)
        )  # [batch, n_cols]

        # Expand mask ke dimensi one-hot: setiap kolom j di-expand ke all_dims[j] dimensi
        # Sehingga seluruh one-hot dimensi suatu kolom di-mask bersama
        m_expanded_parts = []
        for j in range(self.n_cols):
            # mj[i] → ulang all_dims[j] kali untuk kolom j
            m_expanded_parts.append(
                m_col[:, j:j+1].expand(-1, self.all_dims[j])  # [batch, all_dims[j]]
            )
        m_expanded = torch.cat(m_expanded_parts, dim=1)  # [batch, input_dim]

        # x_bar: empirical marginal — shuffle rows secara random (per Eq. 3)
        perm  = torch.randperm(batch_size, device=device)
        x_bar = x_onehot[perm]   # [batch, input_dim]

        # x_tilde = m_expanded ⊙ x_bar + (1 - m_expanded) ⊙ x  (Eq. 3)
        x_tilde = m_expanded * x_bar + (1.0 - m_expanded) * x_onehot

        return x_tilde, m_col   # m_col [batch, n_cols] untuk loss lm

    # ── Forward pass ──────────────────────────────────────────────────────

    def encode(self, x_float: torch.Tensor) -> torch.Tensor:
        """
        Encode float input → latent representation z.
        x_float : [batch, input_dim]  (one-hot float atau embedding)
        return  : [batch, hidden_dim]
        """
        return self.encoder(x_float)

    def encode_from_idx(self, x_idx: torch.Tensor) -> torch.Tensor:
        """
        Encode dari integer index → z.
        x_idx : [batch, n_cols]  int
        return: [batch, hidden_dim]
        """
        x_onehot = self.idx_to_onehot(x_idx)
        return self.encoder(x_onehot)

    def decode(self, z: torch.Tensor) -> list:
        """
        Decode latent z → logits per kolom (untuk evaluasi).
        Menggunakan feature_estimator (sr) sebagai decoder.

        z      : [batch, hidden_dim]
        return : list[n_cols] of [batch, vocab_size_j]
        """
        x_hat_flat = self.feature_estimator(z)   # [batch, input_dim]
        logits = []
        for j in range(self.n_cols):
            s = self._offsets[j]
            e = self._offsets[j + 1]
            logits.append(x_hat_flat[:, s:e])    # [batch, all_dims[j]]
        return logits

    def forward(self, x_idx: torch.Tensor) -> tuple:
        """
        Forward pass lengkap: corrupt → encode → estimasi mask & fitur.

        x_idx : [batch, n_cols]  int
        return: (z, m_hat, x_hat_flat, m_col, x_onehot)
          z          : [batch, hidden_dim]  — representasi encoder
          m_hat      : [batch, n_cols]      — estimasi mask per kolom (Sigmoid)
          x_hat_flat : [batch, input_dim]   — estimasi fitur asli (one-hot)
          m_col      : [batch, n_cols]      — mask per kolom yang diaplikasikan
          x_onehot   : [batch, input_dim]   — input asli (one-hot float)
        """
        x_onehot           = self.idx_to_onehot(x_idx)      # [batch, input_dim]
        x_tilde, m_col     = self.corrupt(x_onehot)          # x_tilde [batch, input_dim]
                                                              # m_col   [batch, n_cols]

        z                  = self.encoder(x_tilde)            # [batch, hidden_dim]
        m_hat              = self.mask_estimator(z)            # [batch, n_cols]
        x_hat_flat         = self.feature_estimator(z)        # [batch, input_dim]

        return z, m_hat, x_hat_flat, m_col, x_onehot


# ===========================================================================
#  Training VIME Encoder (menggantikan train_supervised_embedding_model)
# ===========================================================================

def train_supervised_embedding_model(cat_idx_array: np.ndarray,
                                     labels: np.ndarray,
                                     cat_dims: list,
                                     emb_sizes: list,
                                     n_classes: int,
                                     device: str,
                                     n_epochs: int = 50,
                                     batch_size: int = 1024,
                                     lr: float = 1e-3,
                                     dropout: float = 0.1,
                                     hidden_dim: int = 256,
                                     use_mlp: bool = True,
                                     mlp_ratio: float = 1.5,
                                     noise_std: float = 0.01,
                                     patience: int = 30) -> 'VIMEEmbeddingModel':
    """
    Latih VIME self-supervised encoder.

    Signature dipertahankan agar main_mdlpwith.py tidak perlu diubah.
    Parameter yang tidak relevan untuk VIME (n_classes, dropout, use_mlp,
    mlp_ratio, noise_std) diterima tapi diabaikan.

    Loss function sesuai paper Section 4.1 (Eq. 4-6):
      L = lm(m, m̂) + α · lr(x, x̂)
      lm : BCE per KOLOM (mask estimation, m ∈ {0,1}^d, d=n_cols)
      lr : MSE untuk numerik (bin), CrossEntropy untuk kategorikal
           sesuai keterangan eksplisit paper: "For categorical variables,
           we modified Equation 6 to cross-entropy loss."
    """
    # ── Hitung hidden_dim ────────────────────────────────────────────────
    total_emb   = sum(emb_sizes)
    vime_hidden = hidden_dim if hidden_dim > 0 else min(total_emb, 256)

    # n_num_cols_for_loss: berapa kolom pertama yang merupakan numerik (bin)
    # Diturunkan dari n_classes tidak tersedia langsung, tapi caller meneruskan
    # cat_dims = all_dims = [num_bins...] + [cat_sizes...].
    # Karena kita tidak punya info terpisah di sini, kita pakai n_classes=0
    # sebagai sinyal "tidak ada info", dan default ke 0 (semua dianggap kategorikal
    # untuk loss CE). Caller yang benar harus meneruskan lewat parameter baru.
    # [Catatan: n_classes dipakai sebagai proxy n_num_cols — lihat pemanggil]
    n_num_cols_for_loss = n_classes   # diisi ulang dari caller (lihat load_dataset)

    # p_m: masking probability (paper default ~0.3)
    p_m   = 0.3
    # alpha: bobot reconstruction loss (Eq. 4, paper default ~1.0)
    alpha = 1.0

    # Fix random seed agar hasil embedding reproducible setiap run
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    model = VIMEEmbeddingModel(
        all_dims   = cat_dims,
        hidden_dim = vime_hidden,
        p_m        = p_m,
        alpha      = alpha,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss functions sesuai paper
    bce_loss = nn.BCELoss()                               # lm: mask BCE (Eq. 5)
    mse_loss = nn.MSELoss()                               # lr numerik: MSE (Eq. 6)
    ce_loss  = nn.CrossEntropyLoss()                      # lr kategorikal: CE (bawah Eq. 6)
    device   = next(model.parameters()).device.type + (
        ':' + str(next(model.parameters()).device.index)
        if next(model.parameters()).device.type == 'cuda' else ''
    )

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

    model.train()
    for epoch in range(n_epochs):
        total_loss      = 0.0
        total_mask_loss = 0.0
        total_feat_loss = 0.0
        n_batches       = 0

        for (batch_cat,) in loader:
            optimizer.zero_grad()

            # Forward: corrupt → encode → estimasi mask & fitur
            # m_col      : [batch, n_cols]     — mask per kolom (per paper Eq. 5)
            # x_hat_flat : [batch, input_dim]  — rekonstruksi one-hot
            # x_onehot   : [batch, input_dim]  — input asli one-hot
            z, m_hat, x_hat_flat, m_col, x_onehot = model(batch_cat)

            # ── Loss lm: mask estimation BCE (Paper Eq. 5) ───────────────
            # lm(m, m̂) = -(1/d) Σj [ mj log m̂j + (1-mj) log(1-m̂j) ]
            # m_hat: [batch, n_cols], m_col: [batch, n_cols]  ← sesuai paper
            loss_m = bce_loss(m_hat, m_col)

            # ── Loss lr: feature reconstruction (Paper Eq. 6 + catatan) ──
            # Paper: MSE untuk numerik, cross-entropy untuk kategorikal.
            # "For categorical variables, we modified Equation 6 to
            #  cross-entropy loss." — Paper Section 4.1
            #
            # Implementasi:
            #   - Kolom numerik (indeks 0..n_num_cols-1 di all_dims):
            #       MSE atas dimensi one-hot numerik
            #   - Kolom kategorikal (indeks n_num_cols.. di all_dims):
            #       CrossEntropy per kolom (logits vs argmax one-hot asli)
            loss_r = torch.tensor(0.0, device=device)
            n_num  = n_num_cols_for_loss   # diisi dari closure (lihat di bawah)

            offsets = model._offsets
            for j in range(model.n_cols):
                s = offsets[j]
                e = offsets[j + 1]
                x_hat_j  = x_hat_flat[:, s:e]   # [batch, vocab_j]
                x_true_j = x_onehot[:, s:e]      # [batch, vocab_j]  (one-hot float)

                if j < n_num:
                    # Numerik (bin): MSE atas one-hot representasi
                    loss_r = loss_r + mse_loss(x_hat_j, x_true_j)
                else:
                    # Kategorikal: cross-entropy
                    # target = argmax dari one-hot asli → integer index
                    target_j = x_true_j.argmax(dim=1)   # [batch]  int
                    loss_r   = loss_r + ce_loss(x_hat_j, target_j)

            loss_r = loss_r / model.n_cols   # rata-rata per kolom

            # ── Total loss (Paper Eq. 4) ──────────────────────────────────
            # L = lm + alpha * lr
            loss = loss_m + alpha * loss_r

            loss.backward()
            optimizer.step()

            total_loss      += loss.item()
            total_mask_loss += loss_m.item()
            total_feat_loss += loss_r.item()
            n_batches       += 1

        avg_loss      = total_loss      / n_batches
        avg_mask_loss = total_mask_loss / n_batches
        avg_feat_loss = total_feat_loss / n_batches

        if (epoch + 1) % 10 == 0:
            print(f'[VIME] Epoch {epoch+1}/{n_epochs} - '
                  f'Loss: {avg_loss:.4f} '
                  f'(Mask: {avg_mask_loss:.4f}, Feat: {avg_feat_loss:.4f})')

        if avg_loss < best_loss:
            best_loss        = avg_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'[VIME] Early stopping triggered at epoch {epoch+1}')
            print(f'[VIME] Best loss: {best_loss:.4f}')
            break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f'[VIME] Loaded best model (best loss: {best_loss:.4f})')

    model.eval()

    with torch.no_grad():
        sample_cat = cat_tensor[:min(2048, len(cat_tensor))]
        z_sample   = model.encode_from_idx(sample_cat)
        print(f'[VIME] Distribusi representasi encoder (N={z_sample.shape[0]}):')
        print(f'  mean={z_sample.mean().item():.4f}  '
              f'std={z_sample.std().item():.4f}  '
              f'norm_mean={z_sample.norm(dim=1).mean().item():.4f}')

    for param in model.parameters():
        param.requires_grad_(False)
    print('[VIME] Seluruh parameter encoder di-freeze untuk training diffusion.')

    return model


# ===========================================================================
#  Encode / Decode helpers (menggunakan VIMEEmbeddingModel)
# ===========================================================================

def encode_with_embedding(model: 'VIMEEmbeddingModel',
                          cat_idx_array: np.ndarray,
                          device: str,
                          batch_size: int = 4096) -> np.ndarray:
    """
    Encode integer index → representasi laten VIME encoder (z).
    Menggantikan encode via nn.Embedding lama.

    cat_idx_array : [N, n_cols]  int
    return        : [N, hidden_dim]  float32
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
            z = model.encode_from_idx(batch)   # [B, hidden_dim]
            all_z.append(z.cpu().numpy())

    return np.concatenate(all_z, axis=0).astype(np.float32)


def decode_cat_from_embedding(model: 'VIMEEmbeddingModel',
                              emb_array: np.ndarray,
                              device: str,
                              batch_size: int = 4096,
                              ref_embeddings: np.ndarray = None,
                              ref_all_idx: np.ndarray = None) -> np.ndarray:
    """
    Decode representasi laten VIME z → prediksi kelas tiap kolom.

    [PERBAIKAN BUG UTAMA] Dua mode decode:

    Mode 1 — Nearest-Neighbor di z-space (DEFAULT, dipakai saat ref tersedia):
        Untuk setiap sampel z_i (hasil diffusion), cari z_ref paling dekat
        (L2 distance) dari ref_embeddings (embedding observed/ground-truth).
        Prediksi = label dari z_ref terdekat.
        → Ini benar-benar iterasi-aware: hasil diffusion yang berbeda tiap
          iterasi menghasilkan z berbeda → neighbor berbeda → label berbeda.

    Mode 2 — VIME frozen decoder (FALLBACK, jika ref tidak tersedia):
        Menggunakan feature_estimator sebagai decoder.
        ⚠️ DETERMINISTIK: hasil sama tiap iterasi karena decoder di-freeze.
        Hanya dipakai jika ref_embeddings/ref_all_idx tidak di-pass.

    Parameter
    ---------
    emb_array       : [N, hidden_dim]  — embedding hasil diffusion (di-denorm)
    ref_embeddings  : [N_ref, hidden_dim]  — embedding referensi (observed data)
    ref_all_idx     : [N_ref, n_cols]  — label index referensi (ground truth)
    """
    if ref_embeddings is not None and ref_all_idx is not None:
        # ── Mode 1: Nearest-Neighbor di z-space ──────────────────────────
        # Dibagi batch untuk efisiensi memori
        N       = emb_array.shape[0]
        n_cols  = ref_all_idx.shape[1]
        all_pred = np.empty((N, n_cols), dtype=np.int64)

        ref_t   = torch.tensor(ref_embeddings, dtype=torch.float32, device=device)

        for start in range(0, N, batch_size):
            end     = min(start + batch_size, N)
            q       = torch.tensor(emb_array[start:end],
                                   dtype=torch.float32, device=device)  # [B, D]
            # Squared L2: ||q - ref||^2 = ||q||^2 + ||ref||^2 - 2*q@ref^T
            dist2   = (q.pow(2).sum(1, keepdim=True)
                       + ref_t.pow(2).sum(1).unsqueeze(0)
                       - 2.0 * q @ ref_t.t())                           # [B, N_ref]
            nn_idx  = dist2.argmin(dim=1).cpu().numpy()                  # [B]
            all_pred[start:end] = ref_all_idx[nn_idx]                   # [B, n_cols]

        return all_pred.astype(np.int64)

    else:
        # ── Mode 2: VIME frozen decoder (fallback deterministik) ──────────
        model.eval()
        emb_tensor = torch.tensor(emb_array, dtype=torch.float32, device=device)
        dataset    = torch.utils.data.TensorDataset(emb_tensor)
        loader     = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False,
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


def decode_num_from_embedding(model: 'VIMEEmbeddingModel',
                              emb_array: np.ndarray,
                              bin_midpoints: list,
                              n_num_cols: int,
                              device: str,
                              batch_size: int = 4096,
                              ref_embeddings: np.ndarray = None,
                              ref_num_norm: np.ndarray = None) -> np.ndarray:
    """
    Decode representasi laten VIME z → nilai numerik kontinu (skala normalisasi).

    [PERBAIKAN BUG UTAMA] Dua mode decode:

    Mode 1 — Nearest-Neighbor di z-space (DEFAULT, dipakai saat ref tersedia):
        Untuk setiap z_i (hasil diffusion), cari z_ref paling dekat dari
        ref_embeddings (embedding observed). Prediksi nilai numerik =
        nilai asli ternormalisasi dari NN tersebut.
        → Iterasi-aware: z berbeda tiap iterasi → NN berbeda → nilai berbeda.

    Mode 2 — VIME frozen weighted-sum (FALLBACK):
        z → feature_estimator → softmax → weighted sum bin_midpoints.
        ⚠️ DETERMINISTIK: hasil sama tiap iterasi.

    Parameter
    ---------
    emb_array      : [N, hidden_dim]  — embedding hasil diffusion (di-denorm)
    bin_midpoints  : list[n_num_cols] — midpoint bin, skala normalisasi (Mode 2)
    n_num_cols     : int
    ref_embeddings : [N_ref, hidden_dim]  — embedding referensi (observed)
    ref_num_norm   : [N_ref, n_num_cols]  — nilai numerik ternormalisasi (observed)
    Return         : [N, n_num_cols]  float32
    """
    if ref_embeddings is not None and ref_num_norm is not None:
        # ── Mode 1: Nearest-Neighbor di z-space ──────────────────────────
        N        = emb_array.shape[0]
        all_pred = np.empty((N, n_num_cols), dtype=np.float32)
        ref_t    = torch.tensor(ref_embeddings, dtype=torch.float32, device=device)

        for start in range(0, N, batch_size):
            end    = min(start + batch_size, N)
            q      = torch.tensor(emb_array[start:end],
                                  dtype=torch.float32, device=device)  # [B, D]
            dist2  = (q.pow(2).sum(1, keepdim=True)
                      + ref_t.pow(2).sum(1).unsqueeze(0)
                      - 2.0 * q @ ref_t.t())                           # [B, N_ref]
            nn_idx = dist2.argmin(dim=1).cpu().numpy()                  # [B]
            all_pred[start:end] = ref_num_norm[nn_idx]                 # [B, n_num_cols]

        return all_pred.astype(np.float32)

    else:
        # ── Mode 2: VIME frozen weighted-sum (fallback deterministik) ─────
        model.eval()
        emb_tensor = torch.tensor(emb_array, dtype=torch.float32, device=device)
        dataset    = torch.utils.data.TensorDataset(emb_tensor)
        loader     = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False,
        )
        all_preds = []
        with torch.no_grad():
            for (batch,) in loader:
                recon_logits = model.decode(batch)
                batch_num_preds = []
                for col in range(n_num_cols):
                    logits   = recon_logits[col]
                    probs    = torch.softmax(logits, dim=1)
                    mids_t   = torch.tensor(
                        bin_midpoints[col], dtype=torch.float32, device=device
                    )
                    pred_col = (probs * mids_t.unsqueeze(0)).sum(dim=1)
                    batch_num_preds.append(pred_col.unsqueeze(1))
                batch_num_preds = torch.cat(batch_num_preds, dim=1)
                all_preds.append(batch_num_preds.cpu().numpy())
        return np.concatenate(all_preds, axis=0).astype(np.float32)


# ===========================================================================
#  Load Dataset
# ===========================================================================

def load_dataset(dataname, idx=0, mask_type='MCAR', ratio='30', noise_std=0.01):
    """
    Load dataset dengan MRmD discretization untuk numerik +
    Supervised Embedding untuk SEMUA kolom (numerik-bin + kategorikal).

    Perubahan dari versi sebelumnya:
    - Fitur numerik di-diskritisasi dengan MRmD → integer bin index
    - Bin index numerik di-embed BERSAMA kolom kategorikal (posisi pertama)
    - Pipeline embedding → normalisasi → diffusion → imputasi TIDAK BERUBAH
    - train_num / test_num tetap dikembalikan (nilai float asli, ternormalisasi)
      untuk keperluan evaluasi MAE/RMSE di skala normalisasi

    Output tambahan (dibanding versi sebelumnya):
    - mrmd          : MRmDDiscretizer  (untuk transform test & decode)
    - bin_midpoints : list[n_num_cols] — midpoint bin dalam skala normalisasi
    - n_num_cols    : int

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
    emb_model         : VIMEEmbeddingModel
    emb_sizes         : list[int]
    mrmd              : MRmDDiscretizer  (atau None jika tidak ada fitur numerik)
    bin_midpoints     : list[n_num_cols] of np.ndarray  (atau None)
    n_num_cols        : int
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

        # ── MRmD Discretization (dengan cache) ──────────────────────────
        mrmd_cache_path = f'cache/{dataname}/mrmd.pkl'
        os.makedirs(f'cache/{dataname}', exist_ok=True)

        if os.path.exists(mrmd_cache_path):
            # Load cut points dari cache, skip fitting
            print(f'[MRmD] Cache ditemukan di {mrmd_cache_path}, skip fitting.')
            with open(mrmd_cache_path, 'rb') as f:
                mrmd = pickle.load(f)
            t_mrmd = 0.0
            print(f'[MRmD] Cut points di-load. n_bins per kolom: {mrmd.n_bins_}')
        else:
            # Fit MRmD pada data RAW train (bukan normalisasi) dengan label
            print(f'[MRmD] Cache belum ada. Menjalankan MRmD discretization '
                  f'pada {n_num_cols} kolom numerik ...')
            t_mrmd_start = time.time()
            mrmd = MRmDDiscretizer(val_size=0.125, N_D=50, random_state=42, verbose=False)
            mrmd.fit(train_num_raw, train_labels)
            t_mrmd = time.time() - t_mrmd_start

            # Simpan objek mrmd (berisi cut_points_, x_min_, x_max_, n_bins_)
            with open(mrmd_cache_path, 'wb') as f:
                pickle.dump(mrmd, f)
            print(f'[MRmD] Cache disimpan ke {mrmd_cache_path}')
            print(f'[MRmD] Waktu komputasi diskritisasi: {t_mrmd:.4f}s')

        # Transform train & test pakai cut points yang sama (fit atau cache)
        train_num_bin = mrmd.transform(train_num_raw)   # [N_train, n_num_cols] int64
        test_num_bin  = mrmd.transform(test_num_raw)    # [N_test,  n_num_cols] int64

        # Hitung bin midpoints dalam skala NORMALISASI
        # (dipakai saat decoding: bin index → nilai kontinu untuk MAE/RMSE)
        bin_midpoints = mrmd.get_bin_midpoints(train_num_norm, train_num_bin)

        print(f'[MRmD] n_bins per kolom: {mrmd.n_bins_}')
        print(f'[MRmD] Total bins: {sum(mrmd.n_bins_)}')

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

    # VIME hidden_dim: ukuran representasi encoder (output VIME = [N, hidden_dim])
    vime_hidden_dim = 256
    emb_sizes = [vime_hidden_dim] * len(all_dims)   # placeholder per kolom

    print(f'[VIME] all_dims (num_bin+cat)={all_dims}')
    print(f'[VIME] input_dim (total one-hot)={sum(all_dims)}, hidden_dim={vime_hidden_dim}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Latih VIMEEmbeddingModel ──────────────────────────────────────────
    # Self-supervised encoder sesuai paper VIME (NeurIPS 2020):
    #   Pretext tasks: mask vector estimation + feature vector estimation
    #   Loss: lm (BCE) + alpha * lr (MSE)  — Eq. 4-6 paper
    print('[VIME] Melatih VIME self-supervised encoder '
          '(mask estimation + feature reconstruction loss) ...')
    t_emb_start = time.time()
    emb_model = train_supervised_embedding_model(
        cat_idx_array = train_all_idx,
        labels        = train_labels,
        cat_dims      = all_dims,
        emb_sizes     = emb_sizes,
        n_classes     = n_num_cols,    # dipakai sebagai n_num_cols_for_loss di training
        device        = device,
        n_epochs      = 1000,
        batch_size    = 1024,
        lr            = 1e-3,
        dropout       = 0.1,
        hidden_dim    = vime_hidden_dim,
        use_mlp       = True,
        mlp_ratio     = 1.5,
        noise_std     = noise_std,
        patience      = 40,
    )
    t_emb_end = time.time()
    t_emb = t_emb_end - t_emb_start
    print('[VIME] Training selesai. Parameter di-freeze untuk diffusion.')
    print(f'[VIME] Waktu komputasi embedding: {t_emb:.4f}s')

    # ── Encode semua kolom → representasi VIME z ─────────────────────────
    # encode_with_embedding memanggil model.encode_from_idx (VIME encoder)
    # Output shape: [N, hidden_dim]
    train_all_emb = encode_with_embedding(emb_model, train_all_idx, device)
    test_all_emb  = encode_with_embedding(emb_model, test_all_idx,  device)
    # shape: [N, hidden_dim]
    print("dimensi: ", train_all_emb.shape)

    # ── train_X / test_X sekarang HANYA embedding (tidak ada kolom raw num) ─
    # Karena numerik sudah masuk embedding, len_num = 0 di main
    train_X = train_all_emb
    test_X  = test_all_emb

    # ── Buat extended mask untuk VIME ────────────────────────────────────
    # [PERBAIKAN BUG #2] Strategi mask yang benar untuk VIME:
    #
    # VIME encoder menghasilkan [N, hidden_dim] — satu vektor per sampel
    # yang merangkum SEMUA kolom. Karena encoder global, sampel dengan
    # missing value menghasilkan embedding yang "terkontaminasi" (kolom
    # missing diisi 0 atau nilai default sebelum di-encode).
    #
    # Perbaikan: mask sampel yang memiliki SETIDAKNYA SATU kolom missing
    # di seluruh dimensi hidden_dim (bukan per-dimensi), karena diffusion
    # harus merekonstruksi seluruh embedding vektor untuk sampel tersebut.
    #
    # Ini membuat:
    #   - Sampel fully-observed    → mask_train = False (semua dim)
    #   - Sampel ada kolom missing → mask_train = True  (semua dim hidden_dim)
    # Diffusion hanya memodifikasi embedding sampel yang missing,
    # sampel fully-observed dipertahankan (observed constraint).

    # Kumpulkan mask kolom yang relevan (num + cat)
    train_num_mask = train_mask[:, num_col_idx].astype(bool) if n_num_cols > 0 else np.zeros((len(train_df), 0), dtype=bool)
    train_cat_mask = train_mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else np.zeros((len(train_df), 0), dtype=bool)
    test_num_mask  = test_mask[:, num_col_idx].astype(bool)  if n_num_cols > 0 else np.zeros((len(test_df),  0), dtype=bool)
    test_cat_mask  = test_mask[:, cat_col_idx].astype(bool)  if len(cat_col_idx) > 0 else np.zeros((len(test_df),  0), dtype=bool)

    # Gabungkan mask [num | cat] → [N, n_all_cols]
    if n_num_cols > 0 and len(cat_col_idx) > 0:
        train_all_mask = np.concatenate([train_num_mask, train_cat_mask], axis=1)
        test_all_mask  = np.concatenate([test_num_mask,  test_cat_mask],  axis=1)
    elif n_num_cols > 0:
        train_all_mask = train_num_mask
        test_all_mask  = test_num_mask
    else:
        train_all_mask = train_cat_mask
        test_all_mask  = test_cat_mask

    # any_missing [N] — True jika sampel memiliki minimal 1 kolom missing
    train_any_missing = train_all_mask.any(axis=1)   # [N_train]
    test_any_missing  = test_all_mask.any(axis=1)    # [N_test]

    # Perluas ke [N, hidden_dim]: sampel dengan kolom missing →
    # seluruh hidden_dim di-mask True (karena encoder bersifat global)
    extend_train_mask = np.tile(
        train_any_missing[:, np.newaxis], (1, vime_hidden_dim)
    )   # [N_train, hidden_dim]
    extend_test_mask  = np.tile(
        test_any_missing[:, np.newaxis],  (1, vime_hidden_dim)
    )   # [N_test, hidden_dim]

    # ── Simpan observed embeddings sebagai referensi untuk NN decode ──────
    # [PERBAIKAN BUG #3] train_obs_emb / test_obs_emb adalah embedding dari
    # sampel yang fully-observed (tidak ada kolom missing sama sekali).
    # Dipakai oleh get_eval sebagai ref_embeddings untuk NN-decode yang
    # iterasi-aware (bukan frozen VIME decoder).
    #
    # Untuk in-sample:  ref = sampel training yang fully-observed
    # Untuk OOS:        ref = semua sampel training (fully-observed sebagai
    #                   referensi proxy untuk decode test)
    train_obs_mask = ~train_any_missing   # [N_train] True = fully observed
    test_obs_mask  = ~test_any_missing    # [N_test]  True = fully observed

    # Referensi untuk in-sample decode: training observed samples
    if train_obs_mask.sum() > 0:
        train_ref_emb     = train_all_emb[train_obs_mask]     # [N_obs_train, D]
        train_ref_all_idx = train_all_idx[train_obs_mask]     # [N_obs_train, n_cols]
        train_ref_num     = train_num[train_obs_mask] if n_num_cols > 0 else None
    else:
        # Fallback: pakai semua training data jika semua missing
        train_ref_emb     = train_all_emb
        train_ref_all_idx = train_all_idx
        train_ref_num     = train_num if n_num_cols > 0 else None

    # Referensi untuk OOS decode: semua training observed (proxy untuk test)
    # (kita tidak bisa pakai test_obs karena test yang ingin kita prediksi)
    test_ref_emb     = train_ref_emb      # pakai referensi yang sama
    test_ref_all_idx = train_ref_all_idx
    test_ref_num     = train_ref_num

    print(f'[Ref] Train ref samples (fully-observed): {train_ref_emb.shape[0]}')
    print(f'[Ref] Test  ref samples (dari train obs): {test_ref_emb.shape[0]}')

    return (train_X, test_X,
            train_mask, test_mask,
            train_num, test_num,
            train_all_idx, test_all_idx,
            extend_train_mask, extend_test_mask,
            None,              # cat_bin_num (legacy)
            emb_model,
            emb_sizes,
            mrmd,              # MRmDDiscretizer
            bin_midpoints,     # list[n_num_cols] midpoint per bin, skala norm
            n_num_cols,        # jumlah kolom numerik
            t_mrmd,            # waktu komputasi MRmD discretization (detik)
            t_emb,             # waktu komputasi embedding training (detik)
            train_ref_emb,     # [BARU] referensi embedding in-sample (observed)
            train_ref_all_idx, # [BARU] referensi label index in-sample
            train_ref_num,     # [BARU] referensi nilai numerik in-sample (norm)
            test_ref_emb,      # [BARU] referensi embedding OOS (dari train obs)
            test_ref_all_idx,  # [BARU] referensi label index OOS
            test_ref_num,      # [BARU] referensi nilai numerik OOS (norm)
            )


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
             num_true_norm=None,
             ref_embeddings=None, ref_all_idx=None, ref_num_norm=None):
    """
    Hitung MAE, RMSE (numerik) dan Accuracy (kategorikal).

    [PERBAIKAN] Evaluasi sekarang menggunakan Nearest-Neighbor di z-space
    (bukan VIME frozen decoder) agar hasil berubah tiap iterasi sesuai
    perbaikan diffusion.

    Alur decode yang benar (iterasi-aware):
    ----------------------------------------
    X_recon (embedding diffusion, skala asli)
      → cari NN di ref_embeddings (embedding observed)
      → ambil label/nilai dari NN tersebut
      → bandingkan dengan ground truth (truth_all_idx / num_true_norm)

    Parameter baru:
    ---------------
    ref_embeddings : [N_ref, hidden_dim]  — embedding dari data observed
                     (train_all_emb untuk in-sample, test_all_emb untuk OOS)
    ref_all_idx    : [N_ref, n_cols]      — label index observed (ground truth)
    ref_num_norm   : [N_ref, n_num_cols]  — nilai numerik ternormalisasi observed

    Parameter lama (dipertahankan untuk kompatibilitas):
    ---------------
    bin_midpoints  : dipakai sebagai fallback jika ref tidak tersedia
    num_true_norm  : [N, n_num_cols] — ground truth nilai asli ternormalisasi
    num_num        : DIABAIKAN (legacy)
    truth_all_idx  : [N, n_num_cols + n_cat_cols]  — ground truth integer index
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
            and emb_model is not None):

        # [PERBAIKAN] Decode via NN di z-space (iterasi-aware)
        # ref_num_norm = nilai numerik asli (observed) dari data referensi
        num_pred_norm = decode_num_from_embedding(
            emb_model, X_recon, bin_midpoints, n_num_cols, device,
            ref_embeddings=ref_embeddings,
            ref_num_norm=ref_num_norm,
        )  # [N, n_num_cols]

        # Ground truth: nilai asli ternormalisasi (bukan midpoint bin)
        if num_true_norm is not None:
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

    # ── Kategorikal: Akurasi via NN di z-space ───────────────────────────
    acc = np.nan
    if (truth_all_idx is not None
            and len(cat_col_idx) > 0
            and emb_model is not None
            and emb_sizes is not None
            and cat_mask is not None):

        # [PERBAIKAN] Decode via NN di z-space menggunakan ref_embeddings
        # ref_all_idx = label index dari data referensi (observed)
        pred_all_idx = decode_cat_from_embedding(
            emb_model, X_recon, device,
            ref_embeddings=ref_embeddings,
            ref_all_idx=ref_all_idx,
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