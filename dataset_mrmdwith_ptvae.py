import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
#  PT-VAE Embedding Model (Liu et al., 2025)
#  Arsitektur: Variational Autoencoder with Prior Concept Transformation
#  Konsep: Membangun ruang laten dengan prior concept yang terkonstruksi baik
#          untuk memandu variabel laten dan meningkatkan interpretabilitas.
#
#  Referensi:
#    Liu, Z., Liu, Y., Yu, Z., et al. (2025). PT-VAE: Variational autoencoder
#    with prior concept transformation. Neurocomputing, 638, 130129.
#    https://doi.org/10.1016/j.neucom.2025.130129
#
#  [DIGANTI] Seluruh proses embedding (encode → latent → decode) diganti dengan PT-VAE:
#    - nn.Embedding per kolom → concat → Encoder MLP → (mu, log_var, c_prior)
#    - Prior Concept: latent space VAE terpisah (x_prior → Encoder → c_prior)
#    - Gumbel-Softmax Reparameterization: mengintegrasikan c_prior ke c_concept
#      q(c_concept|x) = exp((logT_concept + g_concept)/τ) / (... + ...)
#      q(c_prior|x)   = exp((logT_prior   + g_prior)  /τ) / (... + ...)
#    - Reparameterization Trick normal: z = mu + eps * sigma  (eps ~ N(0,I))
#    - Decoder MLP: z ⊕ c_concept → rekonstruksi logits per kolom (L_recon)
#    - Loss Total: L_Loss = L_ELBO + L_recon + L_KL
#      L_ELBO = E_q[log p(x|z,c)] - KL(q(c|x)||p(c)) - KL(q(z|x)||p(z))
#      L_recon = ||x'_concept - x'||^2  (rekonstruksi dari prior concept)
#      L_KL    = KL(q(c_prior|x) || q(c_concept|x))  (kesamaan distribusi)
#    - Saat inference (freeze): gunakan z = mu (deterministik)
#
#  [TIDAK BERUBAH] Pipeline dari embedding → normalisasi → diffusion → imputasi.
#  [TIDAK BERUBAH] Classification loss TETAP dipertahankan (sebagai auxiliary loss).
#
#  [BARU] Fitur numerik di-diskritisasi dengan MRmD lalu di-embed bersama
#  fitur kategorikal menggunakan PTVAEEmbeddingModel yang sama.
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
#  PT-VAE Embedding Model (Liu et al., 2025)
#  MENGGANTIKAN: VAEEmbeddingModel (Kingma & Welling, 2013)
# ===========================================================================

def compute_embedding_size(n_categories: int) -> int:
    """
    Hitung ukuran embedding optimal berdasarkan jumlah kategori.
    Rumus: min(600, round(1.6 * n_categories^0.56))
    Referensi: Guo & Berkhahn (2016)
    """
    return min(600, round(1.6 * n_categories ** 0.56))




class PTVAEEmbeddingModel(nn.Module):
    """
    PT-VAE Embedding Model untuk fitur kategorikal tabular.

    Berdasarkan Liu et al. (2025) "PT-VAE: Variational autoencoder with
    prior concept transformation." Neurocomputing 638, 130129.

    =========================================================================
    ARSITEKTUR PT-VAE (sesuai Fig. 1, Fig. 2, dan Algorithm 1 paper):
    =========================================================================

    [A] PRIOR CONCEPT ENCODER — x_prior → (mu_prior, log_var_prior, T_prior)
        Encoder terpisah untuk "well-constructed latent space" sebagai prior.
        Input x_prior adalah data yang sama (x), sesuai paper Section 3.1.
        p(c_prior) ~ Gumbel(0,1)
        Alur:
          x → nn.Embedding → concat → Prior Encoder MLP → (mu_prior, log_var_prior)
          T_prior = exp(0.5 * log_var_prior)

    [B] MAIN ENCODER — x → (mu, log_var, T_concept)
        q_phi(z|x): Recognition Network
          x → nn.Embedding → concat → Main Encoder MLP → (mu, log_var)
          T_concept = exp(0.5 * log_var)

    [C] GUMBEL-SOFTMAX REPARAMETERIZATION — Section 3.1, Eq. 3 & 4
        Mengintegrasikan c_prior ke c_concept via Gumbel-Softmax:

        q(c_concept|x) = exp((log T_concept + g_concept) / tau)
                       / (exp((log T_concept + g_concept) / tau)
                         + exp((log T_prior + g_prior) / tau))        (Eq. 3)

        q(c_prior|x)   = exp((log T_prior + g_prior) / tau)
                       / (exp((log T_concept + g_concept) / tau)
                         + exp((log T_prior + g_prior) / tau))        (Eq. 4)

        g_concept ~ Gumbel(0,1), g_prior ~ Gumbel(0,1)
        tau = temperature parameter

    [D] LATENT VARIABLE + FUSION — Section 3.1, Eq. 5 + Fig. 1
        z = mu + sigma * eps,  eps ~ N(0, I)                          (Eq. 5)
        z_fused = torch.cat([z, c_concept], dim=1)  (⊕ = concat sesuai paper Fig. 1)
        Output shape: [batch, fused_dim] = [batch, 2 * latent_dim]

    [E] MAIN DECODER — p_theta(x|z, c) — Section 3.2, Eq. 6
        x' = p_theta(x | z_fused)
        z_fused → Decoder MLP → per-kolom logits

    [F] PRIOR CONCEPT DECODER — L_recon — Section 3.3, Eq. 12
        x'_concept = decoder_prior(c_concept)
        Dipakai untuk: L_recon = ||x'_concept - x'||^2

    [G] LOSS TOTAL — Section 3.3, Eq. 14
        L_Loss = L_ELBO + L_recon + L_KL

        L_ELBO (bentuk minimisasi dari Eq. 10):
          = CE(recon_logits, x)          ← -E_q[log p(x|z,c)], via CrossEntropy
          + KL(q(z|x) || p(z))           ← closed-form, standard normal prior (Eq. 9)
          + KL(q(c|x) || p(c))           ← uniform categorical prior (Eq. 11)

        Setara dengan memaksimalkan:
          E_q[log p(x|z,c)] - KL(q(z|x)||p(z)) - KL(q(c|x)||p(c))   (Eq. 10)

        L_recon = ||x'_concept - x'||^2                              (Eq. 12)

        L_KL    = KL(q(c_prior|x) || q(c_concept|x))                  (Eq. 13)

    Complexity: O(LNM) — Algorithm 1

    Referensi:
        Liu, Z., Liu, Y., Yu, Z., et al. (2025). PT-VAE: Variational
        autoencoder with prior concept transformation.
        Neurocomputing, 638, 130129. DOI: 10.1016/j.neucom.2025.130129
    """

    def __init__(self, cat_dims: list, emb_sizes: list, n_classes: int,
                 dropout: float = 0.1, hidden_dim: int = 256,
                 latent_dim: int = None,
                 encoder_ratio: float = 1.5,
                 tau: float = 0.5):
        """
        Parameter
        ---------
        cat_dims      : list[int]   — jumlah kategori per kolom (vocab size)
        emb_sizes     : list[int]   — ukuran embedding per kolom
        n_classes     : int         — jumlah kelas untuk supervised classification
        dropout       : float       — dropout rate
        hidden_dim    : int         — hidden dim untuk MLP classifier
        latent_dim    : int|None    — dimensi ruang laten z (default = total_emb_dim)
        encoder_ratio : float       — rasio hidden dim encoder/decoder
        tau           : float       — temperature tau untuk Gumbel-Softmax
                                      tau > 0; tau->0 = diskrit, tau->inf = uniform
                                      (paper Section 3.1, setelah Eq. 4)
                                      tau=1.0 direkomendasikan: mencegah c_concept
                                      saturated (mendekati 0/1) yang menyebabkan
                                      KL(c) membesar hingga ~36 dari max 44.
        """
        super().__init__()

        self.total_emb_dim = sum(emb_sizes)
        self.n_cols        = len(cat_dims)
        self.cat_dims      = cat_dims
        self.emb_sizes     = emb_sizes
        self.n_classes     = n_classes
        self.tau           = tau

        self.latent_dim = latent_dim if latent_dim is not None else self.total_emb_dim
        self.out_dim    = self.total_emb_dim

        # ── [A+B] Input embedding lookup per kolom ────────────────────────
        # Shared embeddings untuk main encoder dan prior concept encoder
        # (x dan x_prior adalah data yang sama, paper Section 3.1)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=n_cat, embedding_dim=emb_dim)
            for n_cat, emb_dim in zip(cat_dims, emb_sizes)
        ])

        enc_hidden = max(self.total_emb_dim, int(self.total_emb_dim * encoder_ratio))

        # ── [B] MAIN ENCODER MLP ──────────────────────────────────────────
        # q_phi(z|x): paper Fig. 1 (jalur bawah)
        # Paper setup: D-Conv32-Conv32-Conv64-FC256-FC10 → untuk tabular: MLP analog
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.total_emb_dim, enc_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_hidden, enc_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu      = nn.Linear(enc_hidden, self.latent_dim)
        self.fc_log_var = nn.Linear(enc_hidden, self.latent_dim)

        # ── [A] PRIOR CONCEPT ENCODER MLP ────────────────────────────────
        # Encoder untuk prior concept c_prior: paper Fig. 1 (jalur atas, x_prior)
        # Arsitektur simetris dengan main encoder
        self.prior_encoder_mlp = nn.Sequential(
            nn.Linear(self.total_emb_dim, enc_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_hidden, enc_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu_prior      = nn.Linear(enc_hidden, self.latent_dim)
        self.fc_log_var_prior = nn.Linear(enc_hidden, self.latent_dim)

        # ── [E] MAIN DECODER MLP ─────────────────────────────────────────
        # p_theta(x|z,c): paper Fig. 1 (Decoder bawah), menerima z_fused
        # [FIX paper] ⊕ di Fig.1 = CONCATENATION bukan addition.
        # Decoder menerima [z; c_concept] sehingga input dim = 2 * latent_dim.
        dec_hidden = enc_hidden
        fused_dim      = self.latent_dim * 2
        self.fused_dim = fused_dim
        self.decoder_mlp = nn.Sequential(
            nn.Linear(fused_dim, dec_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_hidden, dec_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_hidden, self.total_emb_dim),
        )

        # ── [F] PRIOR CONCEPT DECODER MLP ────────────────────────────────
        # Decoder terpisah untuk x'_concept dari c_concept (Eq. 12)
        # paper Fig. 1 (Decoder atas, dari c_concept, menghasilkan L_recon)
        # Input hanya c_concept (latent_dim), sesuai paper Eq. 12
        self.prior_decoder_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, dec_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_hidden, dec_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_hidden, self.total_emb_dim),
        )

        # ── Linear Decoder per kolom ──────────────────────────────────────
        # Dipakai oleh KEDUA decoder (main + prior concept)
        self.decoders = nn.ModuleList([
            nn.Linear(emb_size, n_cat)
            for n_cat, emb_size in zip(cat_dims, emb_sizes)
        ])

        # ── MLP Classifier (auxiliary, dipertahankan) ─────────────────────
        self.dropout    = nn.Dropout(dropout)
        # [FIX paper] Classifier menerima z_fused (fused_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

        # ── LayerNorm pada z_fused (stabilisasi skala sebelum diffusion) ──
        # [FIX paper] LayerNorm pada fused_dim (z concat c_concept)
        self.layer_norm = nn.LayerNorm(fused_dim)

    # ── Embedding input ───────────────────────────────────────────────────

    def _embed_input(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Lookup embedding per kolom dan concat.
        x_cat : [batch, n_cols]
        return: [batch, total_emb_dim]
        """
        return torch.cat([
            self.embeddings[i](x_cat[:, i]) for i in range(self.n_cols)
        ], dim=1)

    # ── [B] Main Encoder ──────────────────────────────────────────────────

    def _encode_to_params(self, x_emb: torch.Tensor):
        """
        Main Encoder q_phi(z|x). Paper Fig. 1 (jalur bawah).
        Return: (mu, log_var, T_concept)
          T_concept = sigma = exp(0.5 * log_var) — transformation variable
          untuk Gumbel-Softmax (Section 3.1)
        """
        h       = self.encoder_mlp(x_emb)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        T_concept = torch.exp(0.5 * log_var)   # T_concept = sigma
        return mu, log_var, T_concept

    # ── [A] Prior Concept Encoder ─────────────────────────────────────────

    def _encode_prior_to_params(self, x_emb: torch.Tensor):
        """
        Prior Concept Encoder. Paper Fig. 1 (jalur atas, x_prior).
        Return: (mu_prior, log_var_prior, T_prior)
          T_prior = sigma_prior = exp(0.5 * log_var_prior)
        """
        h             = self.prior_encoder_mlp(x_emb)
        mu_prior      = self.fc_mu_prior(h)
        log_var_prior = self.fc_log_var_prior(h)
        log_var_prior = torch.clamp(log_var_prior, min=-10.0, max=10.0)
        T_prior = torch.exp(0.5 * log_var_prior)  # T_prior = sigma_prior
        return mu_prior, log_var_prior, T_prior

    # ── [C] Gumbel-Softmax Reparameterization ─────────────────────────────

    @staticmethod
    def _sample_gumbel(shape, device, eps: float = 1e-6) -> torch.Tensor:
        """
        Sampling Gumbel(0, 1): g = -log(-log(U)), U ~ Uniform(0,1).
        Paper Section 3.1: g_concept ~ Gumbel(0,1), g_prior ~ Gumbel(0,1).

        U di-clamp ke (eps, 1-eps) untuk menghindari log(0) = -inf
        yang menyebabkan Gumbel sample menjadi +inf → NaN setelah
        dibagi tau dan dimasukkan ke exp().
        eps=1e-6 cukup untuk float32 (resolusi ~1.2e-7).
        """
        U = torch.rand(shape, device=device).clamp(eps, 1.0 - eps)
        return -torch.log(-torch.log(U))

    def _gumbel_softmax_concept(
        self,
        T_concept: torch.Tensor,   # [batch, latent_dim]
        T_prior:   torch.Tensor,   # [batch, latent_dim]
    ):
        """
        Gumbel-Softmax Reparameterization Trick. Paper Eq. 3 & 4.

        q(c_concept|x) = exp((log T_concept + g_concept) / tau)
                       / (exp((log T_concept + g_concept) / tau)
                         + exp((log T_prior   + g_prior)  / tau))      (Eq. 3)

        q(c_prior|x)   = exp((log T_prior + g_prior) / tau)
                       / (exp((log T_concept + g_concept) / tau)
                         + exp((log T_prior   + g_prior)  / tau))      (Eq. 4)

        Saat tau->0: samples lebih diskrit
        Saat tau->inf: samples mendekati uniform
        Saat tau positif & finite: smooth & differentiable (untuk training)

        Return: (c_concept, q_c_prior)
          c_concept  : q(c_concept|x) — [batch, latent_dim]
          q_c_prior  : q(c_prior|x)   — [batch, latent_dim]
        """
        device = T_concept.device
        T_c = torch.clamp(T_concept, min=1e-8)
        T_p = torch.clamp(T_prior,   min=1e-8)

        if self.training:
            g_concept = self._sample_gumbel(T_c.shape, device)
            g_prior   = self._sample_gumbel(T_p.shape, device)
        else:
            # Inference: tanpa Gumbel noise (deterministik)
            g_concept = torch.zeros_like(T_c)
            g_prior   = torch.zeros_like(T_p)

        logit_c = (torch.log(T_c) + g_concept) / self.tau
        logit_p = (torch.log(T_p) + g_prior)   / self.tau

        # Stable Gumbel-Softmax via log-sum-exp
        denom_log = torch.logaddexp(logit_c, logit_p)
        c_concept  = torch.exp(logit_c - denom_log)   # Eq. 3
        q_c_prior  = torch.exp(logit_p - denom_log)   # Eq. 4

        return c_concept, q_c_prior

    # ── [D] Normal Reparameterization ─────────────────────────────────────

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Normal Reparameterization Trick. Paper Eq. 5:
          z = mu + sigma * eps,  eps ~ N(0, I),  sigma = exp(0.5 * log_var)
        Inference (eval mode): return mu deterministik.
        Training: sampling dengan reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    # ── [E+F] Decode ──────────────────────────────────────────────────────

    def _decode_from_z(self, z: torch.Tensor) -> torch.Tensor:
        """Main Decoder: z_fused → [batch, total_emb_dim]"""
        return self.decoder_mlp(z)

    def _decode_prior_concept(self, c_concept: torch.Tensor) -> torch.Tensor:
        """Prior Concept Decoder: c_concept → [batch, total_emb_dim]. Paper Eq. 12."""
        return self.prior_decoder_mlp(c_concept)

    def _logits_from_recon(self, recon_emb: torch.Tensor) -> list:
        """Split recon embedding → list of per-kolom logits."""
        per_col = torch.split(recon_emb, self.emb_sizes, dim=1)
        return [self.decoders[i](per_col[i]) for i in range(self.n_cols)]

    # ── Public API ────────────────────────────────────────────────────────

    def encode(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Encode integer index → z_fused (deterministik saat inference).

        Alur (Algorithm 1, Lines 3-5):
          [3] Embed input → Main Encoder → (mu, log_var, T_concept)
          [3] Embed input → Prior Encoder → T_prior
          [4] Gumbel-Softmax(T_concept, T_prior) → c_concept
          [5] z_fused = cat([mu, c_concept])  (deterministik, tanpa sampling)

        [FIX paper] ⊕ = concatenation sesuai Fig.1, bukan addition.
        Pakai mu (bukan sample z) agar representasi stabil dan reproducible.
        Output shape: [N, fused_dim] = [N, 2 * latent_dim]
        """
        x_emb = self._embed_input(x_cat)
        mu, log_var, T_concept = self._encode_to_params(x_emb)
        _, _, T_prior          = self._encode_prior_to_params(x_emb)
        c_concept, _           = self._gumbel_softmax_concept(T_concept, T_prior)
        # [FIX paper] ⊕ = concatenation sesuai Fig.1, bukan addition
        # Deterministik: pakai mu (tanpa sampling) untuk inference yang stabil
        z_fused                = torch.cat([mu, c_concept], dim=1)
        return self.layer_norm(z_fused)

    def encode_with_params(self, x_cat: torch.Tensor):
        """
        Encode dan kembalikan semua parameter untuk PT-VAE loss.
        Algorithm 1, Lines 3-5.

        Return: (z_fused, mu, log_var, c_concept, q_c_prior,
                 mu_prior, log_var_prior)
        """
        x_emb = self._embed_input(x_cat)
        mu, log_var, T_concept             = self._encode_to_params(x_emb)
        mu_prior, log_var_prior, T_prior   = self._encode_prior_to_params(x_emb)
        c_concept, q_c_prior               = self._gumbel_softmax_concept(T_concept, T_prior)
        z                                  = self.reparameterize(mu, log_var)
        # [FIX paper] ⊕ = concatenation sesuai Fig.1, bukan addition
        z_fused                            = torch.cat([z, c_concept], dim=1)
        z_normed                           = self.layer_norm(z_fused)
        return (z_normed, mu, log_var, c_concept, q_c_prior, mu_prior, log_var_prior)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """Auxiliary classifier: z_fused → logit kelas."""
        return self.classifier(z)

    def decode(self, z: torch.Tensor) -> list:
        """
        Main Decoder: z_fused → per-kolom logits.
        Paper Eq. 6: x' = p_theta(x | z, c)
        """
        return self._logits_from_recon(self._decode_from_z(z))

    def decode_prior(self, c_concept: torch.Tensor) -> list:
        """
        Prior Concept Decoder: c_concept → per-kolom logits.
        Dipakai untuk L_recon: L_recon = ||x'_concept - x'||^2 (Eq. 12)
        """
        return self._logits_from_recon(self._decode_prior_concept(c_concept))

    def forward(self, x_cat: torch.Tensor, add_noise: bool = False):
        """
        Forward pass PT-VAE untuk training. Algorithm 1, Lines 3-7.

        Return:
          z_fused            : [batch, latent_dim]
          mu, log_var        : main encoder params
          c_concept          : q(c_concept|x) dari Gumbel-Softmax (Eq. 3)
          q_c_prior          : q(c_prior|x) dari Gumbel-Softmax (Eq. 4)
          mu_prior, log_var_prior : prior concept encoder params
          class_logits       : [batch, n_classes] — auxiliary classifier
          recon_logits       : list[n_cols] — [6] logits dari z_fused
          recon_prior_logits : list[n_cols] — logits dari c_concept (L_recon)
        """
        (z_fused, mu, log_var, c_concept, q_c_prior,
         mu_prior, log_var_prior) = self.encode_with_params(x_cat)

        class_logits       = self.classify(z_fused)
        recon_logits       = self.decode(z_fused)          # [6] train decoder
        recon_prior_logits = self.decode_prior(c_concept)  # L_recon

        return (z_fused, mu, log_var, c_concept, q_c_prior,
                mu_prior, log_var_prior, class_logits,
                recon_logits, recon_prior_logits)

    # ── PT-VAE Loss Functions (Section 3.3) ──────────────────────────────

    @staticmethod
    def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        KL(q(z|x) || p(z)). Paper Eq. 9 & 10, komponen KL untuk z.
        Closed-form (Kingma & Welling 2013, Appendix B, Eq. B.3):
          KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

        Clamp mu dan log_var sebelum digunakan untuk mencegah overflow:
        - log_var.exp() overflow jika log_var > ~88 (float32)
        - mu.pow(2) overflow jika |mu| sangat besar

        [FIX] Hapus pembagian / latent_dim yang menyebabkan KL terlalu kecil
              dan memicu posterior collapse (z tidak membawa informasi).
        """
        log_var_c = torch.clamp(log_var, min=-10.0, max=10.0)
        mu_c      = torch.clamp(mu,      min=-10.0, max=10.0)
        kl = -0.5 * torch.sum(
            1 + log_var_c - mu_c.pow(2) - log_var_c.exp(), dim=1
        )
        return kl.mean()

    @staticmethod
    def kl_divergence_c(c_concept: torch.Tensor, K: int) -> torch.Tensor:
        """
        KL(q(c|x) || p(c)). Paper Eq. 11.
        c_concept adalah hasil binary Gumbel-Softmax per dimensi ∈ (0,1).
        Prior p(c) = Bernoulli(0.5) per dimensi (uniform/tidak informatif).

        KL(Bernoulli(q) || Bernoulli(0.5)) = log(2) - H(Bernoulli(q))
          H(q) = -q*log(q) - (1-q)*log(1-q)

        Formulasi ini numerically stable: tidak ada perkalian 0*(-inf).
        c_concept bisa bernilai tepat 0.0 atau 1.0 dalam float32 ketika
        Gumbel logits sangat besar — formula lama menghasilkan NaN.
        """
        # Clamp ke (1e-7, 1-1e-7): cukup jauh dari 0/1 dalam float32
        c_s = torch.clamp(c_concept, min=1e-7, max=1.0 - 1e-7)
        # Hitung entropy Bernoulli: H(c) = -c*log(c) - (1-c)*log(1-c)
        H = -(c_s * torch.log(c_s) + (1.0 - c_s) * torch.log(1.0 - c_s))
        # KL = log(2) - H(c)  [>= 0, bernilai 0 saat c=0.5]
        log2 = torch.log(torch.tensor(2.0, device=c_concept.device))
        kl_per_dim = log2 - H
        # Normalisasi per dimensi: KL(c) biasanya sum atas latent_dim dimensi
        # tanpa normalisasi nilainya ~36 (64 × 0.57) yang mendominasi loss.
        # Dibagi latent_dim agar magnitude setara dengan CE (~1.0).
        latent_dim = c_concept.shape[1]
        return kl_per_dim.sum(dim=1).mean() / latent_dim

    @staticmethod
    def reconstruction_loss_concept(
        recon_prior_logits: list,
        recon_logits: list,
        n_cols: int
    ) -> torch.Tensor:
        """
        L_recon = ||x'_concept - x'||^2. Paper Eq. 12.
        Diimplementasikan sebagai MSE antara logit prior concept
        dan logit main reconstruction.

        [FIX] Hapus .detach() pada recon_logits. Dengan .detach(), main decoder
              tidak mendapat gradient dari L_recon sehingga prior decoder mengejar
              target yang terus bergerak → L_recon naik terus bukan turun.
              Kedua decoder sekarang saling terhubung via L_recon.
        """
        loss = torch.tensor(0.0, device=recon_logits[0].device)
        for i in range(n_cols):
            loss = loss + F.mse_loss(recon_prior_logits[i], recon_logits[i])
        return loss / n_cols

    @staticmethod
    def kl_divergence_concept_prior(
        q_c_prior: torch.Tensor,
        c_concept: torch.Tensor
    ) -> torch.Tensor:
        """
        L_KL = KL(q(c_prior|x) || q(c_concept|x)). Paper Eq. 13.
        q_c_prior dan c_concept adalah output binary Gumbel-Softmax dimana
        per dimensi: q_c_prior + c_concept = 1.0 (binary competition).
        Keduanya membentuk distribusi Bernoulli per dimensi.

        [FIX v2] Gunakan KL Bernoulli yang selalu >= 0:
          KL(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
          Normalisasi / latent_dim agar skala konsisten dengan KL lainnya.
        """
        eps = 1e-7
        p   = torch.clamp(q_c_prior, min=eps, max=1.0 - eps)
        q   = torch.clamp(c_concept, min=eps, max=1.0 - eps)
        kl  = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
        latent_dim = p.shape[1]
        return kl.sum(dim=1).mean() / latent_dim


# Alias untuk kompatibilitas pemanggilan di seluruh kode
VAEEmbeddingModel = PTVAEEmbeddingModel



# ===========================================================================
#  Training PT-VAE Embedding (Liu et al., 2025)
#  Menggantikan: train_supervised_embedding_model / train_vae_embedding_model
# ===========================================================================

def train_vae_embedding_model(cat_idx_array: np.ndarray,
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
                               latent_dim: int = None,
                               encoder_ratio: float = 1.5,
                               patience: int = 30) -> PTVAEEmbeddingModel:
    """
    Latih PTVAEEmbeddingModel dengan loss PT-VAE sesuai Liu et al. (2025).

    Loss Total (paper Eq. 14):
        L_Loss = L_ELBO + L_recon + L_KL

        L_ELBO  = E_q[log p(x|z,c)]                    (reconstruction via CE)
                - KL(q(c|x) || p(c))                   (Eq. 11: uniform categorical prior)
                - KL(q(z|x) || p(z))                   (Eq. 9: standard normal prior)
                                                         (Eq. 10)
        → Dalam bentuk minimisasi (loss):
          L_ELBO = CE_recon + KL_z + KL_c
          (CE_recon ≈ -E_q[log p(x|z,c)], KL selalu positif sebagai penalti)

        L_recon = ||x'_concept - x'||^2                 (Eq. 12: MSE logits)

        L_KL    = KL(q(c_prior|x) || q(c_concept|x))   (Eq. 13)

        L_class = CrossEntropy(class_logits, labels)     (auxiliary, tidak di paper)

        L_total = L_ELBO + L_recon + L_KL + L_class
                  Semua term berbobot 1.0 — sesuai paper Section 3.3
                  "adopted equal weights to both terms during the learning process"

    Algorithm 1 (paper):
        Input: Dataset X, prior concept c
        for epoch:
          [3] train encoder dengan data X
          [4] Gumbel-Softmax → q(c_concept|x), q(c_prior|x)
          [5] infer q(c_concept|x), q(c_prior|x) dengan Eq. 2 & reparameterization Eq. 5
          [6] train decoder dengan z sebagai input
          [7] hitung total loss dengan Eq. 14

    Parameter
    ---------
    cat_idx_array : [N, n_cols]  — integer index semua kolom
    labels        : [N]          — integer label kelas
    cat_dims      : list[int]    — vocab size tiap kolom
    emb_sizes     : list[int]    — embedding dim tiap kolom
    n_classes     : int          — jumlah kelas
    device        : str
    latent_dim    : int|None     — dimensi ruang laten (default=total_emb_dim)

    Return : PTVAEEmbeddingModel (parameter di-freeze, eval mode)
    """
    model = PTVAEEmbeddingModel(
        cat_dims      = cat_dims,
        emb_sizes     = emb_sizes,
        n_classes     = n_classes,
        dropout       = dropout,
        hidden_dim    = hidden_dim,
        latent_dim    = latent_dim,
        encoder_ratio = encoder_ratio,
        tau           = 1.0,   # temperature τ — tau=1.0 mencegah c_concept saturated
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss   = nn.CrossEntropyLoss()

    cat_tensor   = torch.tensor(cat_idx_array, dtype=torch.long, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    dataset      = torch.utils.data.TensorDataset(cat_tensor, label_tensor)
    cpu_gen      = torch.Generator(device='cpu')
    loader       = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = False,
        generator   = cpu_gen,
    )

    # K = latent_dim — dipakai untuk upper bound KL_c = log K (Eq. 11)
    K = model.latent_dim

    best_loss        = float('inf')
    patience_counter = 0
    best_model_state = None

    model.train()
    for epoch in range(n_epochs):
        total_loss        = 0.0
        total_elbo_recon  = 0.0   # E_q[log p(x|z,c)] bagian rekonstruksi
        total_kl_z        = 0.0   # KL(q(z|x) || p(z))
        total_kl_c        = 0.0   # KL(q(c|x) || p(c))
        total_l_elbo      = 0.0   # L_ELBO = recon - KL_z - KL_c
        total_recon_loss  = 0.0   # L_recon (Eq. 12)
        total_kl_loss     = 0.0   # L_KL (Eq. 13)
        total_class_loss  = 0.0
        n_batches         = 0

        for batch_cat, batch_labels in loader:
            optimizer.zero_grad()

            # ── Algorithm 1, Lines 3-7 ────────────────────────────────────
            (z_fused, mu, log_var, c_concept, q_c_prior,
             mu_prior, log_var_prior, class_logits,
             recon_logits, recon_prior_logits) = model(batch_cat)

            # ── L_ELBO (Eq. 10) ───────────────────────────────────────────
            # Term 1: -E_q[log p(x|z,c)] ≈ CE loss
            elbo_recon = sum(
                ce_loss(recon_logits[i], batch_cat[:, i])
                for i in range(model.n_cols)
            ) / model.n_cols

            # Term 2: KL(q(z|x) || p(z)) — Eq. 9, closed-form, ALWAYS POSITIVE
            kl_z = PTVAEEmbeddingModel.kl_divergence(mu, log_var)

            # Term 3: KL(q(c|x) || p(c)) — Eq. 11, ALWAYS POSITIVE
            kl_c = PTVAEEmbeddingModel.kl_divergence_c(c_concept, K)

            # L_ELBO (bentuk minimisasi): elbo_recon + KL_z + KL_c
            l_elbo = elbo_recon + kl_z + kl_c

            # ── L_recon (Eq. 12) ──────────────────────────────────────────
            l_recon = PTVAEEmbeddingModel.reconstruction_loss_concept(
                recon_prior_logits, recon_logits, model.n_cols
            )

            # ── L_KL (Eq. 13) ─────────────────────────────────────────────
            l_kl = PTVAEEmbeddingModel.kl_divergence_concept_prior(
                q_c_prior, c_concept
            )

            # ── L_class (auxiliary) ───────────────────────────────────────
            class_loss = ce_loss(class_logits, batch_labels)

            # ── Total Loss (Eq. 14) ───────────────────────────────────────
            loss = l_elbo + l_recon + l_kl + class_loss

            # Guard NaN: skip batch jika ada komponen NaN
            if not torch.isfinite(loss):
                nan_info = {
                    'elbo_recon': elbo_recon.item(),
                    'kl_z':       kl_z.item(),
                    'kl_c':       kl_c.item(),
                    'l_recon':    l_recon.item(),
                    'l_kl':       l_kl.item(),
                    'class_loss': class_loss.item(),
                }
                print(f'[WARN] PT-VAE loss NaN/Inf di-skip: {nan_info}')
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss       += loss.item()
            total_elbo_recon += elbo_recon.item()
            total_kl_z       += kl_z.item()
            total_kl_c       += kl_c.item()
            total_l_elbo     += l_elbo.item()
            total_recon_loss += l_recon.item()
            total_kl_loss    += l_kl.item()
            total_class_loss += class_loss.item()
            n_batches        += 1

        avg_loss        = total_loss        / n_batches
        avg_elbo_recon  = total_elbo_recon  / n_batches
        avg_kl_z        = total_kl_z        / n_batches
        avg_kl_c        = total_kl_c        / n_batches
        avg_l_elbo      = total_l_elbo      / n_batches
        avg_recon_loss  = total_recon_loss  / n_batches
        avg_kl_loss     = total_kl_loss     / n_batches
        avg_class_loss  = total_class_loss  / n_batches

        if (epoch + 1) % 10 == 0:
            print(
                f'[PT-VAE] Epoch {epoch+1:>4}/{n_epochs} | '
                f'Loss={avg_loss:.4f} | '
                f'L_ELBO={avg_l_elbo:.4f} '
                f'[CE={avg_elbo_recon:.4f}, KL(z)={avg_kl_z:.4f}, KL(c)={avg_kl_c:.4f}] | '
                f'L_recon={avg_recon_loss:.4f} | '
                f'L_KL={avg_kl_loss:.4f} | '
                f'L_class={avg_class_loss:.4f}'
            )
            # Peringatan jika ada loss yang dominan atau nol
            losses = {
                'KL(z)':   avg_kl_z,
                'KL(c)':   avg_kl_c,
                'L_recon': avg_recon_loss,
                'L_KL':    avg_kl_loss,
            }
            for name, val in losses.items():
                if val < 1e-8:
                    print(f'  [WARN] {name} mendekati nol ({val:.2e}) — '
                          f'komponen ini mungkin tidak aktif!')
            vals = list(losses.values())
            if len(vals) > 0 and max(vals) > 10 * (sum(vals) / len(vals)):
                dominant = max(losses, key=losses.get)
                print(f'  [WARN] {dominant}={losses[dominant]:.4f} mendominasi loss '
                      f'(target masing-masing komponen ≈ 0.1–1.0)')

        if avg_loss < best_loss:
            best_loss        = avg_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'[PT-VAE Embedding] Early stopping triggered at epoch {epoch+1}')
            print(f'[PT-VAE Embedding] Best total loss: {best_loss:.4f}')
            break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f'[PT-VAE Embedding] Loaded best model state.')

    model.eval()

    with torch.no_grad():
        sample_cat = cat_tensor[:min(2048, len(cat_tensor))]
        z_sample   = model.encode(sample_cat)
        print(f'[PT-VAE Embedding] Distribusi laten z_fused (N={z_sample.shape[0]}):')
        print(f'  mean={z_sample.mean().item():.4f}  '
              f'std={z_sample.std().item():.4f}  '
              f'norm_mean={z_sample.norm(dim=1).mean().item():.4f}')

    for param in model.parameters():
        param.requires_grad_(False)
    print('[PT-VAE Embedding] Seluruh parameter PT-VAE embedding di-freeze untuk training diffusion.')

    return model




# ===========================================================================
#  Encode / Decode helpers (disesuaikan ke PTVAEEmbeddingModel)
# ===========================================================================

def encode_with_embedding(model: VAEEmbeddingModel,
                          cat_idx_array: np.ndarray,
                          device: str,
                          batch_size: int = 4096) -> np.ndarray:
    """
    Encode integer index → embedding numpy array menggunakan VAE encoder.

    Saat inference (eval mode), model.encode() mengembalikan
    z_fused = cat([mu, c_concept]) secara deterministik — tanpa sampling
    dan tanpa Gumbel noise. Output shape: [N, fused_dim] = [N, 2*latent_dim].
    PT-VAE: representasi stabil untuk downstream task (diffusion).
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
            # encode() saat eval mode → z = mu (deterministik)
            z = model.encode(batch)
            all_z.append(z.cpu().numpy())

    return np.concatenate(all_z, axis=0).astype(np.float32)


def decode_cat_from_embedding(model: VAEEmbeddingModel,
                              emb_array: np.ndarray,
                              device: str,
                              batch_size: int = 4096) -> np.ndarray:
    """
    Decode embedding (z / mu) → prediksi kelas tiap kolom (argmax logits).

    Input emb_array berisi vektor laten z (atau mu saat inference).
    PT-VAE Main Decoder memetakan z_fused → embedding space → per-kolom logits.

    emb_array : [N, fused_dim]  (= [N, 2*latent_dim] setelah FIX concatenation)
    Return    : [N, n_cols]     — predicted integer index
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
            # PT-VAE decode: z_fused → Main Decoder MLP → per-kolom logits
            recon_logits = model.decode(batch)
            pred_idx = torch.stack([
                torch.argmax(logits, dim=1)
                for logits in recon_logits
            ], dim=1)
            all_pred.append(pred_idx.cpu().numpy())

    return np.concatenate(all_pred, axis=0).astype(np.int64)


def decode_num_from_embedding(model: VAEEmbeddingModel,
                              emb_array: np.ndarray,
                              bin_midpoints: list,
                              n_num_cols: int,
                              device: str,
                              batch_size: int = 4096) -> np.ndarray:
    """
    Decode embedding (z / mu) → nilai numerik kontinu (dalam skala normalisasi).

    Alur (Weighted Sum / Soft-Max Decode via VAE Decoder):
      z → VAE Decoder MLP → embedding space → per-kolom logits → softmax → weighted sum midpoints

    Untuk kolom ke-i (kolom numerik berada di awal, indeks 0..n_num_cols-1):
        p_i  = softmax(decoder_i(VAE_decode(z)_i))   # [N, n_bins_i]
        pred = p_i @ mids_i                           # [N] — weighted sum

    Parameter
    ---------
    model         : VAEEmbeddingModel
    emb_array     : [N, latent_dim]  — vektor laten z (= mu saat inference)
    bin_midpoints : list[n_num_cols] of np.ndarray  — midpoint per bin, skala norm
    n_num_cols    : int — jumlah kolom numerik (embedding pertama)
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
            # PT-VAE decode: z_fused → Main Decoder MLP → per-kolom logits
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
    VAE Embedding untuk SEMUA kolom (numerik-bin + kategorikal).

    [DIGANTI] Proses embedding menggunakan PTVAEEmbeddingModel (Liu et al. 2025)
    menggantikan VAEEmbeddingModel sepenuhnya.
    - Main Encoder: input embedding → MLP → (mu, log_var, T_concept)
    - Prior Concept Encoder: input embedding → MLP → T_prior
    - Gumbel-Softmax: (T_concept, T_prior) → c_concept, q_c_prior (Eq. 3 & 4)
    - z = mu + sigma*eps; z_fused = cat([z, c_concept])  [FIX: concat bukan +]
    - Main Decoder: z_fused → MLP → per-kolom logits
    - Prior Concept Decoder: c_concept → MLP → per-kolom logits (L_recon)
    - Loss: L_ELBO + L_recon + L_KL + Classification (Eq. 14)

    [TIDAK BERUBAH] Pipeline MRmD discretization, normalisasi, diffusion, imputasi.
    [TIDAK BERUBAH] train_num / test_num dikembalikan untuk evaluasi MAE/RMSE.

    Parameter
    ---------
    dataname  : str
    idx       : int   — mask split index
    mask_type : str   — 'MCAR', 'MAR', 'MNAR_logistic_T2'
    ratio     : str   — masking ratio ('10', '30', '50')
    noise_std : float — DIABAIKAN (VAE memiliki stochasticity bawaan); dipertahankan
                        untuk kompatibilitas signature dengan versi sebelumnya.

    Return
    ------
    train_X           : [N_train, fused_dim]                float32  (fused_dim = 2*latent_dim)
    test_X            : [N_test,  fused_dim]                float32
    ori_train_mask    : mask asli train [N_train, total_cols]
    ori_test_mask     : mask asli test  [N_test,  total_cols]
    train_num         : [N_train, n_num_cols]  — float asli (ternormalisasi)
    test_num          : [N_test,  n_num_cols]
    train_all_idx     : [N_train, n_num_cols + n_cat_cols]  — semua bin/label idx
    test_all_idx      : [N_test,  n_num_cols + n_cat_cols]
    extend_train_mask : [N_train, total_emb_dim]
    extend_test_mask  : [N_test,  total_emb_dim]
    cat_bin_num       : None  (legacy)
    emb_model         : PTVAEEmbeddingModel
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

        t_mrmd_start = time.time()
        mrmd.fit(train_num_raw, train_labels)

        train_num_bin = mrmd.transform(train_num_raw)
        test_num_bin  = mrmd.transform(test_num_raw)

        bin_midpoints = mrmd.get_bin_midpoints(train_num_norm, train_num_bin)
        t_mrmd_end = time.time()
        t_mrmd = t_mrmd_end - t_mrmd_start

        print(f'[MRmD] n_bins per kolom: {mrmd.n_bins_}')
        print(f'[MRmD] Total bins: {sum(mrmd.n_bins_)}')
        print(f'[MRmD] Waktu komputasi diskritisasi: {t_mrmd:.4f}s')

    else:
        train_num     = np.zeros((len(train_df), 0), dtype=np.float32)
        test_num      = np.zeros((len(test_df),  0), dtype=np.float32)
        train_num_bin = np.zeros((len(train_df), 0), dtype=np.int64)
        test_num_bin  = np.zeros((len(test_df),  0), dtype=np.int64)
        bin_midpoints = []
        mrmd          = None
        t_mrmd        = 0.0

    # ── Encoding kolom kategorikal (TIDAK BERUBAH) ────────────────────────
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

    # ── Gabungkan: [num_bin | cat_idx] (TIDAK BERUBAH) ───────────────────
    if n_num_cols > 0 and len(cat_col_idx) > 0:
        train_all_idx = np.concatenate([train_num_bin, train_cat_idx], axis=1)
        test_all_idx  = np.concatenate([test_num_bin,  test_cat_idx],  axis=1)
    elif n_num_cols > 0:
        train_all_idx = train_num_bin
        test_all_idx  = test_num_bin
    else:
        train_all_idx = train_cat_idx
        test_all_idx  = test_cat_idx

    # ── Dimensi embedding (TIDAK BERUBAH) ────────────────────────────────
    all_dims  = (mrmd.n_bins_ if mrmd is not None else []) + cat_dims_cat
    emb_sizes = [compute_embedding_size(n) for n in all_dims]

    print(f'[Embedding] all_dims (num_bin+cat)={all_dims}')
    print(f'[Embedding] emb_sizes={emb_sizes}, total_emb_dim={sum(emb_sizes)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Latih VAEEmbeddingModel (MENGGANTIKAN SupervisedLearnableEmbeddingModel) ──
    # Input: semua kolom (numerik bin + kategorikal) sebagai integer index
    # Loss: ELBO murni = Recon + KL + Classification (tanpa alpha/beta)
    print('[PT-VAE Embedding] Melatih PTVAEEmbeddingModel '
          '(ELBO: Reconstruction + KL Divergence + Classification loss) ...')
    print('[PT-VAE Embedding] Referensi: Liu et al. (2025) PT-VAE: Variational Autoencoder with Prior Concept Transformation')
    t_emb_start = time.time()
    emb_model = train_vae_embedding_model(
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
        latent_dim    = None,        # default = total_emb_dim (kompatibel dengan diffusion)
        encoder_ratio = 1.5,
        patience      = 40,
    )
    t_emb_end = time.time()
    t_emb = t_emb_end - t_emb_start
    print('[PT-VAE Embedding] Training selesai. Parameter di-freeze untuk diffusion.')
    print(f'[PT-VAE Embedding] Waktu komputasi embedding: {t_emb:.4f}s')

    # ── Encode semua kolom → embedding vector (z_fused, deterministik) ────
    train_all_emb = encode_with_embedding(emb_model, train_all_idx, device)
    test_all_emb  = encode_with_embedding(emb_model, test_all_idx,  device)
    # [FIX paper] encode() output shape: [N, fused_dim] = [N, 2 * latent_dim]
    # karena z_fused = cat([mu, c_concept]) masing-masing [N, latent_dim]

    # ── train_X / test_X sekarang HANYA embedding PT-VAE ─────────────────
    train_X = train_all_emb
    test_X  = test_all_emb

    # ── Buat extended mask ────────────────────────────────────────────────
    train_num_mask = train_mask[:, num_col_idx].astype(bool) if n_num_cols > 0 else np.zeros((len(train_df), 0), dtype=bool)
    train_cat_mask = train_mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else np.zeros((len(train_df), 0), dtype=bool)
    test_num_mask  = test_mask[:, num_col_idx].astype(bool)  if n_num_cols > 0 else np.zeros((len(test_df),  0), dtype=bool)
    test_cat_mask  = test_mask[:, cat_col_idx].astype(bool)  if len(cat_col_idx) > 0 else np.zeros((len(test_df),  0), dtype=bool)

    if n_num_cols > 0 and len(cat_col_idx) > 0:
        train_all_mask = np.concatenate([train_num_mask, train_cat_mask], axis=1)
        test_all_mask  = np.concatenate([test_num_mask,  test_cat_mask],  axis=1)
    elif n_num_cols > 0:
        train_all_mask = train_num_mask
        test_all_mask  = test_num_mask
    else:
        train_all_mask = train_cat_mask
        test_all_mask  = test_cat_mask

    # [FIX paper] encode() output = [N, fused_dim] = [N, 2 * latent_dim].
    # Mask harus diperluas ke fused_dim, bukan total_emb_dim / sum(emb_sizes).
    # Sample di-mark missing jika ANY kolomnya missing (same as DAE version).
    fused_dim = emb_model.fused_dim  # 2 * latent_dim

    def extend_mask_to_fused(mask_cols: np.ndarray, n_fused: int) -> np.ndarray:
        """
        Perluas mask [N, n_cols] → [N, fused_dim].
        Sample di-mark missing jika ANY kolomnya missing.
        """
        any_missing = mask_cols.any(axis=1, keepdims=True)  # [N, 1]
        return np.tile(any_missing, (1, n_fused))            # [N, fused_dim]

    extend_train_mask = extend_mask_to_fused(train_all_mask, fused_dim)
    extend_test_mask  = extend_mask_to_fused(test_all_mask,  fused_dim)

    return (train_X, test_X,
            train_mask, test_mask,
            train_num, test_num,
            train_all_idx, test_all_idx,
            extend_train_mask, extend_test_mask,
            None,          # cat_bin_num (legacy)
            emb_model,
            emb_sizes,
            mrmd,          # MRmDDiscretizer
            bin_midpoints, # list[n_num_cols] midpoint per bin, skala norm
            n_num_cols,    # jumlah kolom numerik
            t_mrmd,        # waktu komputasi MRmD discretization (detik)
            t_emb)         # waktu komputasi embedding training (detik)


def mean_std(data, mask):
    mask      = (~mask).astype(np.float32)
    mask_sum  = mask.sum(0)
    mask_sum[mask_sum == 0] = 1
    mean      = (data * mask).sum(0) / mask_sum
    var       = ((data - mean) ** 2 * mask).sum(0) / mask_sum
    std       = np.sqrt(var)
    return mean, std


# ===========================================================================
#  Evaluasi (TIDAK BERUBAH)
# ===========================================================================

def get_eval(dataname, X_recon, X_true, truth_all_idx,
             num_num, emb_model, emb_sizes, mask,
             device='cpu', oos=False,
             bin_midpoints=None, n_num_cols=0,
             num_true_norm=None):
    """
    Hitung MAE, RMSE (numerik) dan Accuracy (kategorikal).

    [TIDAK BERUBAH] — logika evaluasi sama persis.
    emb_model sekarang adalah PTVAEEmbeddingModel, decode() tetap kompatibel.

    Numerik (MAE/RMSE):
        decode_num_from_embedding → bin index → midpoint (skala norm) [prediksi]
        Ground truth: num_true_norm — nilai float asli ternormalisasi (skala norm)

    Kategorikal (Accuracy):
        decode_cat_from_embedding → argmax logits → dibandingkan truth_all_idx
    """
    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

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

        num_pred_norm = decode_num_from_embedding(
            emb_model, X_recon, bin_midpoints, n_num_cols, device
        )  # [N, n_num_cols]

        if num_true_norm is not None:
            gt_norm = num_true_norm
        else:
            N = X_true.shape[0]
            gt_norm = np.zeros((N, n_num_cols), dtype=np.float32)
            for col in range(n_num_cols):
                mids     = bin_midpoints[col]
                true_bin = truth_all_idx[:, col].astype(int)
                true_bin = np.clip(true_bin, 0, len(mids) - 1)
                gt_norm[:, col] = mids[true_bin]

        diff = num_pred_norm[num_mask] - gt_norm[num_mask]
        mae  = float(np.abs(diff).mean())
        rmse = float(np.sqrt((diff ** 2).mean()))

    # ── Kategorikal: Akurasi via VAE Linear Decoder ───────────────────────
    acc = np.nan
    if (truth_all_idx is not None
            and len(cat_col_idx) > 0
            and emb_model is not None
            and emb_sizes is not None
            and cat_mask is not None):

        pred_all_idx = decode_cat_from_embedding(
            emb_model, X_recon, device
        )  # [N, n_num_cols + n_cat_cols]

        n_cat_cols    = len(cat_col_idx)
        correct_total = 0
        total_missing = 0

        for j in range(n_cat_cols):
            rows_miss = cat_mask[:, j]
            if rows_miss.sum() == 0:
                continue

            col_offset = n_num_cols + j

            pred_j = pred_all_idx[:, col_offset]
            true_j = truth_all_idx[:, col_offset].astype(int)

            correct = (pred_j[rows_miss] == true_j[rows_miss]).sum()
            correct_total += int(correct)
            total_missing += int(rows_miss.sum())

        if total_missing > 0:
            acc = correct_total / total_missing

    return mae, rmse, acc