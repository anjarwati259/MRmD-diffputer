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
from scipy.stats import chi2

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
#  [BARU] Fitur numerik di-diskritisasi dengan MDLP lalu di-embed bersama
#  fitur kategorikal menggunakan PTVAEEmbeddingModel yang sama.
# ===========================================================================


# ===========================================================================
#  MDLP Discretizer (implementasi dari scratch, tanpa dependency eksternal)
# ===========================================================================

class MDLPDiscretizer:
    """
    Minimum Description Length Principle (MDLP) Discretizer.

    Implementasi algoritma Fayyad & Irani (1993):
    - Recursive binary partitioning berdasarkan entropy gain
    - Stopping criterion: MDL (Minimum Description Length)
    - Fit pada data training, transform ke integer bin index

    Referensi:
        Fayyad, U. M., & Irani, K. B. (1993). Multi-interval discretization
        of continuous-valued attributes for classification learning. IJCAI.
    """

    def __init__(self, min_samples: int = 3):
        """
        Parameter
        ---------
        min_samples : int
            Minimum jumlah sample dalam satu bin (untuk mencegah over-splitting)
        """
        self.min_samples = min_samples
        self.cut_points_ = []    # list of np.ndarray, satu per kolom
        self.n_bins_      = []    # list of int, jumlah bin per kolom
        self.bin_means_   = []    # list of np.ndarray, nilai tengah bin per kolom
                                  # dalam skala NORMALISASI — diisi setelah fit+normalize

    # ── Entropy helpers ──────────────────────────────────────────────────

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        """Hitung entropy dari array label integer."""
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def _class_entropy(y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Weighted class information entropy setelah split.
        E(A, T; S) = (|S1|/|S|) * Ent(S1) + (|S2|/|S|) * Ent(S2)
        """
        n = len(y_left) + len(y_right)
        if n == 0:
            return 0.0
        ent_left  = MDLPDiscretizer._entropy(y_left)
        ent_right = MDLPDiscretizer._entropy(y_right)
        return (len(y_left) / n) * ent_left + (len(y_right) / n) * ent_right

    @staticmethod
    def _mdl_gain(y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Hitung MDL gain: apakah gain > MDL threshold?

        MDL Threshold (Fayyad & Irani 1993):
            delta = log2(3^k - 2) - (k*Ent(S) - k1*Ent(S1) - k2*Ent(S2))
            k  = jumlah kelas unik di S
            k1 = jumlah kelas unik di S1
            k2 = jumlah kelas unik di S2

        Return gain (float). Caller membandingkan dengan threshold:
            Gain(A,T;S) > (log2(N-1)/N) + (delta/N)
        """
        n  = len(y)
        k  = len(np.unique(y))
        k1 = len(np.unique(y_left))
        k2 = len(np.unique(y_right))

        ent   = MDLPDiscretizer._entropy(y)
        ent_s = MDLPDiscretizer._class_entropy(y_left, y_right)
        gain  = ent - ent_s

        delta = np.log2(3 ** k - 2) - (
            k  * ent -
            k1 * MDLPDiscretizer._entropy(y_left) -
            k2 * MDLPDiscretizer._entropy(y_right)
        )

        threshold = (np.log2(n - 1) / n) + (delta / n)
        return gain, threshold

    # ── Recursive split ──────────────────────────────────────────────────

    def _find_best_split(self, x: np.ndarray, y: np.ndarray):
        """
        Cari titik split terbaik pada array x yang sudah terurut.
        Return (best_threshold, best_gain, best_gain_threshold) atau None.
        """
        n = len(x)
        if n < 2 * self.min_samples:
            return None

        best_t     = None
        best_gain  = -np.inf
        best_thr   = np.inf

        # Kandidat cut point: titik tengah antara nilai berurutan yang berbeda
        unique_vals = np.unique(x)
        if len(unique_vals) < 2:
            return None

        candidates = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for t in candidates:
            left_mask  = x <= t
            right_mask = ~left_mask

            if left_mask.sum() < self.min_samples or right_mask.sum() < self.min_samples:
                continue

            gain, thr = self._mdl_gain(y, y[left_mask], y[right_mask])

            if gain > best_gain:
                best_gain = gain
                best_thr  = thr
                best_t    = t

        if best_t is None:
            return None
        return best_t, best_gain, best_thr

    def _recursive_split(self, x: np.ndarray, y: np.ndarray,
                          cuts: list, depth: int = 0, max_depth: int = 20):
        """Rekursif cari semua cut point yang memenuhi MDL criterion."""
        if depth >= max_depth or len(x) < 2 * self.min_samples:
            return

        result = self._find_best_split(x, y)
        if result is None:
            return

        t, gain, thr = result

        # MDLP stopping criterion: hanya split jika gain > MDL threshold
        if gain <= thr:
            return

        cuts.append(t)

        # Rekursi pada tiap sisi
        left_mask  = x <= t
        right_mask = ~left_mask

        self._recursive_split(x[left_mask],  y[left_mask],  cuts, depth + 1, max_depth)
        self._recursive_split(x[right_mask], y[right_mask], cuts, depth + 1, max_depth)

    # ── Public API ───────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MDLPDiscretizer':
        """
        Fit MDLP pada data training.

        X : [N, n_cols]  float — fitur numerik
        y : [N]          int   — label kelas
        """
        n_cols = X.shape[1]
        self.cut_points_ = []
        self.n_bins_      = []

        for col in range(n_cols):
            x_col = X[:, col]

            # Hapus NaN untuk fitting
            valid_mask = ~np.isnan(x_col)
            x_valid = x_col[valid_mask]
            y_valid = y[valid_mask]

            # Sort berdasarkan nilai fitur
            sort_idx = np.argsort(x_valid)
            x_sorted = x_valid[sort_idx]
            y_sorted = y_valid[sort_idx]

            cuts = []
            self._recursive_split(x_sorted, y_sorted, cuts)
            cuts = sorted(set(cuts))

            self.cut_points_.append(np.array(cuts))
            self.n_bins_.append(len(cuts) + 1)

            print(f'  [MDLP] Col {col}: {len(cuts)} cut points → {len(cuts)+1} bins')

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform nilai kontinu → integer bin index [0, n_bins-1].

        X : [N, n_cols]
        Return : [N, n_cols]  int64
        """
        n_cols = X.shape[1]
        out    = np.zeros_like(X, dtype=np.int64)

        for col in range(n_cols):
            cuts = self.cut_points_[col]
            out[:, col] = np.digitize(X[:, col], cuts, right=False).astype(np.int64)
            # np.digitize returns [0, len(bins)] → clip ke [0, n_bins-1]
            out[:, col] = np.clip(out[:, col], 0, self.n_bins_[col] - 1)

        return out

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_bin_midpoints(self, X_norm: np.ndarray,
                          X_norm_binned: np.ndarray) -> list:
        """
        Hitung nilai tengah (midpoint) setiap bin dalam skala normalisasi.

        Dipakai saat decoding: bin index → nilai kontinu dalam skala (X-mean)/std.

        X_norm        : [N, n_cols]  — data normalisasi (skala (X-mean)/std)
        X_norm_binned : [N, n_cols]  — hasil transform (integer bin index)

        Return : list[n_cols] of np.ndarray, tiap elemen panjang n_bins_[col]
        """
        n_cols   = X_norm.shape[1]
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
        z_fused = z + c_concept  (simbol ⊕ pada Fig. 1)

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
        # Arsitektur simetris dengan encoder
        dec_hidden = enc_hidden
        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, dec_hidden),
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
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

        # ── LayerNorm pada z_fused (stabilisasi skala sebelum diffusion) ──
        self.layer_norm = nn.LayerNorm(self.latent_dim)

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
          [5] z = mu + sigma*eps; z_fused = z + c_concept

        Saat eval: tanpa sampling (deterministik).
        """
        x_emb = self._embed_input(x_cat)
        mu, log_var, T_concept = self._encode_to_params(x_emb)
        _, _, T_prior          = self._encode_prior_to_params(x_emb)
        c_concept, _           = self._gumbel_softmax_concept(T_concept, T_prior)
        z                      = self.reparameterize(mu, log_var)
        z_fused                = z + c_concept
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
        z_fused                            = z + c_concept
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
        """
        log_var_c = torch.clamp(log_var, min=-10.0, max=10.0)
        mu_c      = torch.clamp(mu,      min=-10.0, max=10.0)
        kl = -0.5 * torch.sum(
            1 + log_var_c - mu_c.pow(2) - log_var_c.exp(), dim=1
        )
        # Normalisasi per dimensi agar setara dengan KL(c) dan CE
        latent_dim = mu.shape[1]
        return kl.mean() / latent_dim

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
        """
        loss = torch.tensor(0.0, device=recon_logits[0].device)
        for i in range(n_cols):
            loss = loss + F.mse_loss(recon_prior_logits[i], recon_logits[i].detach())
        return loss / n_cols

    @staticmethod
    def kl_divergence_concept_prior(
        q_c_prior: torch.Tensor,
        c_concept: torch.Tensor
    ) -> torch.Tensor:
        """
        L_KL = KL(q(c_prior|x) || q(c_concept|x)). Paper Eq. 13.
        q_c_prior dan c_concept adalah output Gumbel-Softmax: keduanya
        selalu positif dan menjumlah ke 1 (q_p + q_c = 1 per dimensi).
        Normalisasi ulang untuk menjaga properti ini setelah clamp.
        """
        eps = 1e-7
        q_p = torch.clamp(q_c_prior, min=eps)
        q_c = torch.clamp(c_concept, min=eps)
        # Re-normalisasi: pastikan q_p + q_c = 1 per dimensi
        total = q_p + q_c
        q_p   = q_p / total
        q_c   = q_c / total
        kl    = torch.sum(q_p * torch.log(q_p / q_c), dim=1)
        # Normalisasi per dimensi agar magnitude setara dengan komponen lain
        latent_dim = q_p.shape[1]
        return kl.mean() / latent_dim


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

    Saat inference (eval mode), model.encode() mengembalikan z_fused = mu + c_concept
    secara deterministik — tanpa sampling dan tanpa Gumbel noise.
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

    emb_array : [N, latent_dim]  (= [N, total_emb_dim] karena latent_dim default)
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
    Load dataset dengan MDLP discretization untuk numerik +
    VAE Embedding untuk SEMUA kolom (numerik-bin + kategorikal).

    [DIGANTI] Proses embedding menggunakan PTVAEEmbeddingModel (Liu et al. 2025)
    menggantikan VAEEmbeddingModel sepenuhnya.
    - Main Encoder: input embedding → MLP → (mu, log_var, T_concept)
    - Prior Concept Encoder: input embedding → MLP → T_prior
    - Gumbel-Softmax: (T_concept, T_prior) → c_concept, q_c_prior (Eq. 3 & 4)
    - z = mu + sigma*eps; z_fused = z + c_concept
    - Main Decoder: z_fused → MLP → per-kolom logits
    - Prior Concept Decoder: c_concept → MLP → per-kolom logits (L_recon)
    - Loss: L_ELBO + L_recon + L_KL + Classification (Eq. 14)

    [TIDAK BERUBAH] Pipeline MDLP discretization, normalisasi, diffusion, imputasi.
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
    emb_model         : PTVAEEmbeddingModel
    emb_sizes         : list[int]
    mdlp              : MDLPDiscretizer  (atau None jika tidak ada fitur numerik)
    bin_midpoints     : list[n_num_cols] of np.ndarray  (atau None)
    n_num_cols        : int
    t_mdlp            : float — waktu komputasi MDLP discretization (detik)
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

        # ── MDLP Discretization ──────────────────────────────────────────
        print(f'[MDLP] Menjalankan MDLP discretization pada {n_num_cols} kolom numerik ...')
        mdlp = MDLPDiscretizer(min_samples=3)

        t_mdlp_start = time.time()
        mdlp.fit(train_num_raw, train_labels)

        train_num_bin = mdlp.transform(train_num_raw)
        test_num_bin  = mdlp.transform(test_num_raw)

        bin_midpoints = mdlp.get_bin_midpoints(train_num_norm, train_num_bin)
        t_mdlp_end = time.time()
        t_mdlp = t_mdlp_end - t_mdlp_start

        print(f'[MDLP] n_bins per kolom: {mdlp.n_bins_}')
        print(f'[MDLP] Total bins: {sum(mdlp.n_bins_)}')
        print(f'[MDLP] Waktu komputasi diskritisasi: {t_mdlp:.4f}s')

    else:
        train_num     = np.zeros((len(train_df), 0), dtype=np.float32)
        test_num      = np.zeros((len(test_df),  0), dtype=np.float32)
        train_num_bin = np.zeros((len(train_df), 0), dtype=np.int64)
        test_num_bin  = np.zeros((len(test_df),  0), dtype=np.int64)
        bin_midpoints = []
        mdlp          = None
        t_mdlp        = 0.0

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
    all_dims  = (mdlp.n_bins_ if mdlp is not None else []) + cat_dims_cat
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

    # ── Encode semua kolom → embedding vector (z = mu, deterministik) ────
    train_all_emb = encode_with_embedding(emb_model, train_all_idx, device)
    test_all_emb  = encode_with_embedding(emb_model, test_all_idx,  device)
    # shape: [N, total_emb_dim]  (= latent_dim karena latent_dim=total_emb_dim)

    # ── train_X / test_X sekarang HANYA embedding VAE ────────────────────
    train_X = train_all_emb
    test_X  = test_all_emb

    # ── Buat extended mask (TIDAK BERUBAH) ───────────────────────────────
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

    extend_train_mask = extend_mask_emb(train_all_mask, emb_sizes_arr)
    extend_test_mask  = extend_mask_emb(test_all_mask,  emb_sizes_arr)

    return (train_X, test_X,
            train_mask, test_mask,
            train_num, test_num,
            train_all_idx, test_all_idx,
            extend_train_mask, extend_test_mask,
            None,          # cat_bin_num (legacy)
            emb_model,
            emb_sizes,
            mdlp,          # MDLPDiscretizer
            bin_midpoints, # list[n_num_cols] midpoint per bin, skala norm
            n_num_cols,    # jumlah kolom numerik
            t_mdlp,        # waktu komputasi MDLP discretization (detik)
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