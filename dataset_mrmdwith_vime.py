import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import json
import time

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
        # Sesuai GitHub jsyoon0823/VIME (vime_self.py, encoder layer)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Feature Estimator sr: Linear → ReLU → Linear ─────────────────
        # Sesuai GitHub jsyoon0823/VIME (vime_self.py, feature_estimator layer)
        self.feature_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
        )

        # ── Mask Estimator sm: Linear → ReLU → Linear → Sigmoid ──────────
        # Sesuai GitHub jsyoon0823/VIME (vime_self.py, mask_estimator layer)
        self.mask_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
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

        x_tilde = m ⊙ x_bar + (1-m) ⊙ x
        m_j ~ Bernoulli(p_m) per fitur (dimensi one-hot)
        x_bar[j] sampled dari empirical marginal (shuffled rows)

        Return: (x_tilde, m)  — keduanya [batch, input_dim]
        """
        batch_size = x_onehot.shape[0]
        device     = x_onehot.device

        # Mask: Bernoulli per dimensi (sesuai paper Eq. 3)
        m = torch.bernoulli(
            torch.full_like(x_onehot, self.p_m)
        )  # [batch, input_dim]

        # x_bar: empirical marginal — shuffle rows secara random
        perm  = torch.randperm(batch_size, device=device)
        x_bar = x_onehot[perm]   # [batch, input_dim]

        # x_tilde = m ⊙ x_bar + (1 - m) ⊙ x  (Eq. 3)
        x_tilde = m * x_bar + (1.0 - m) * x_onehot

        return x_tilde, m

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
        return: (z, m_hat, x_hat_flat, m, x_onehot)
          z          : [batch, hidden_dim]  — representasi encoder
          m_hat      : [batch, input_dim]   — estimasi mask (Sigmoid)
          x_hat_flat : [batch, input_dim]   — estimasi fitur asli
          m          : [batch, input_dim]   — mask yang diaplikasikan
          x_onehot   : [batch, input_dim]   — input asli (one-hot)
        """
        x_onehot        = self.idx_to_onehot(x_idx)    # [batch, input_dim]
        x_tilde, m      = self.corrupt(x_onehot)        # [batch, input_dim]

        z               = self.encoder(x_tilde)          # [batch, hidden_dim]
        m_hat           = self.mask_estimator(z)          # [batch, input_dim]
        x_hat_flat      = self.feature_estimator(z)      # [batch, input_dim]

        return z, m_hat, x_hat_flat, m, x_onehot


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
    Latih VIME self-supervised encoder sebagai pengganti SupervisedLearnableEmbeddingModel.

    Signature dipertahankan agar main_mdlpwith.py tidak perlu diubah.
    Parameter yang tidak relevan untuk VIME (n_classes, dropout, use_mlp,
    mlp_ratio, noise_std) diterima tapi diabaikan.

    Loss function sesuai paper (Eq. 4-6):
      L = lm(m, m_hat) + alpha * lr(x, x_hat)
      lm : BCE per dimensi one-hot (mask estimation)
      lr : MSE per dimensi one-hot (feature reconstruction)
           → untuk kolom kategorikal, MSE atas one-hot setara dengan mendorong
             rekonstruksi ke one-hot asli (sesuai implementasi GitHub resmi
             yang menggunakan MSE atas seluruh vektor fitur)

    Catatan: paper menyebut cross-entropy untuk categorical variables (bawah Eq. 6),
    namun implementasi GitHub resmi (vime_self.py) menggunakan MSE atas seluruh
    input vector (termasuk one-hot categorical). Kami ikuti implementasi GitHub resmi.
    """
    # ── Hitung hidden_dim dari emb_sizes (ambil total emb dim sebagai hidden) ─
    total_emb = sum(emb_sizes)
    # Gunakan hidden_dim arg (dari caller) sebagai ukuran hidden VIME encoder
    # Fallback: min(total_emb, 256) jika hidden_dim tidak di-pass eksplisit
    vime_hidden = hidden_dim if hidden_dim > 0 else min(total_emb, 256)

    # p_m: masking probability (hyperparameter VIME, paper default ~0.3)
    p_m   = 0.3
    # alpha: bobot reconstruction loss (Eq. 4, paper default ~1.0)
    alpha = 1.0

    model = VIMEEmbeddingModel(
        all_dims   = cat_dims,
        hidden_dim = vime_hidden,
        p_m        = p_m,
        alpha      = alpha,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss functions sesuai paper + GitHub resmi
    bce_loss = nn.BCELoss()    # mask estimation loss lm (Eq. 5)
    mse_loss = nn.MSELoss()    # feature reconstruction loss lr (Eq. 6)

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
            z, m_hat, x_hat_flat, m, x_onehot = model(batch_cat)

            # ── Loss lm: mask estimation BCE (Eq. 5) ─────────────────────
            # lm(m, m_hat) = -(1/d) * Σ_j [m_j log m_hat_j + (1-m_j) log(1-m_hat_j)]
            loss_m = bce_loss(m_hat, m)

            # ── Loss lr: feature reconstruction MSE (Eq. 6) ──────────────
            # lr(x, x_hat) = (1/d) * Σ_j (x_j - x_hat_j)^2
            # Sesuai implementasi GitHub resmi (vime_self.py) yang menggunakan
            # MSE atas seluruh vektor one-hot (bukan cross-entropy per kolom)
            loss_r = mse_loss(x_hat_flat, x_onehot)

            # ── Total loss (Eq. 4) ────────────────────────────────────────
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
                              batch_size: int = 4096) -> np.ndarray:
    """
    Decode representasi laten VIME z → prediksi kelas tiap kolom (argmax logits).
    Menggunakan feature_estimator (sr) sebagai decoder.

    emb_array : [N, hidden_dim]
    Return    : [N, n_cols]  — predicted integer index
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
            recon_logits = model.decode(batch)   # list[n_cols] of [B, vocab_j]
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
                              batch_size: int = 4096) -> np.ndarray:
    """
    Decode representasi laten VIME z → nilai numerik kontinu (skala normalisasi).

    Alur (Weighted Sum / Soft-Max Decode):
      z → feature_estimator(z) → logits per kolom →
      softmax → weighted sum dengan bin_midpoints

    emb_array     : [N, hidden_dim]
    bin_midpoints : list[n_num_cols] of np.ndarray  — midpoint per bin, skala norm
    n_num_cols    : int — jumlah kolom numerik (index pertama di all_dims)
    Return        : [N, n_num_cols]  float32  — nilai kontinu skala normalisasi
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
            recon_logits = model.decode(batch)   # list[n_cols] of [B, vocab_j]

            batch_num_preds = []
            for col in range(n_num_cols):
                logits  = recon_logits[col]                           # [B, n_bins_col]
                probs   = torch.softmax(logits, dim=1)                # [B, n_bins_col]
                mids_t  = torch.tensor(
                    bin_midpoints[col], dtype=torch.float32, device=device
                )                                                      # [n_bins_col]
                pred_col = (probs * mids_t.unsqueeze(0)).sum(dim=1)   # [B]
                batch_num_preds.append(pred_col.unsqueeze(1))          # [B, 1]

            batch_num_preds = torch.cat(batch_num_preds, dim=1)       # [B, n_num_cols]
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

        # ── MRmD Discretization ──────────────────────────────────────────
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
        n_classes     = n_classes,
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
    # VIME encoder menghasilkan [N, hidden_dim] — satu vektor per sampel.
    # Mask di level kolom asal [N, n_cols] perlu diperluas ke [N, hidden_dim].
    #
    # Strategi: jika suatu sampel memiliki SETIDAKNYA SATU kolom yang missing,
    # maka seluruh hidden_dim dimensi representasinya ditandai sebagai missing
    # (karena encoder global merangkum semua kolom ke satu vektor).
    #
    # Ini konsisten dengan cara diffusion memanfaatkan mask: posisi True → perlu
    # diimputasi, posisi False → observed (dipertahankan).

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

    # Perluas ke [N, hidden_dim]: sampel yang ada kolom missing →
    # seluruh hidden_dim di-mask True
    extend_train_mask = np.tile(
        train_any_missing[:, np.newaxis], (1, vime_hidden_dim)
    )   # [N_train, hidden_dim]
    extend_test_mask  = np.tile(
        test_any_missing[:, np.newaxis],  (1, vime_hidden_dim)
    )   # [N_test, hidden_dim]

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
        decode_cat_from_embedding → argmax logits → dibandingkan truth_all_idx
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