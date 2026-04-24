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

    [E] MAIN DECODER — p_theta(x|z, c) — Section 3.2, Eq. 6
        x' = p_theta(x | z_fused)
        z_fused → Decoder MLP → per-kolom logits

    [F] PRIOR CONCEPT DECODER — L_recon — Section 3.3, Eq. 12
        x'_concept = decoder_prior(c_concept)
        Dipakai untuk: L_recon = ||x'_concept - x'||^2

    [G] LOSS TOTAL — Section 3.3, Eq. 14
        L_Loss = L_ELBO + L_recon + L_KL

        L_ELBO (bentuk minimisasi dari Eq. 10):
          = CE(recon_logits, x)          <- -E_q[log p(x|z,c)], via CrossEntropy
          + KL(q(z|x) || p(z))           <- closed-form, standard normal prior (Eq. 9)
          + KL(q(c|x) || p(c))           <- uniform categorical prior (Eq. 11)

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
        cat_dims      : list[int]   - jumlah kategori per kolom (vocab size)
        emb_sizes     : list[int]   - ukuran embedding per kolom
        n_classes     : int         - jumlah kelas untuk supervised classification
        dropout       : float       - dropout rate
        hidden_dim    : int         - hidden dim untuk MLP classifier
        latent_dim    : int|None    - dimensi ruang laten z (default = total_emb_dim)
        encoder_ratio : float       - rasio hidden dim encoder/decoder
        tau           : float       - temperature tau untuk Gumbel-Softmax
                                      tau > 0; tau->0 = diskrit, tau->inf = uniform
                                      tau=1.0 direkomendasikan (paper Section 3.1)
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

        # [A+B] Input embedding lookup per kolom
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=n_cat, embedding_dim=emb_dim)
            for n_cat, emb_dim in zip(cat_dims, emb_sizes)
        ])

        enc_hidden = max(self.total_emb_dim, int(self.total_emb_dim * encoder_ratio))

        # [B] MAIN ENCODER MLP — q_phi(z|x)
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

        # [A] PRIOR CONCEPT ENCODER MLP — simetris dengan main encoder
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

        # [E] MAIN DECODER MLP — p_theta(x|z,c)
        # [FIX paper] ⊕ di Fig.1 = CONCATENATION bukan addition.
        # Decoder menerima [z; c_concept] sehingga input dim = 2 * latent_dim.
        dec_hidden     = enc_hidden
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

        # [F] PRIOR CONCEPT DECODER MLP — untuk L_recon (Eq. 12)
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

        # Linear Decoder per kolom — dipakai oleh KEDUA decoder
        self.decoders = nn.ModuleList([
            nn.Linear(emb_size, n_cat)
            for n_cat, emb_size in zip(cat_dims, emb_sizes)
        ])

        # MLP Classifier (auxiliary, dipertahankan)
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

        # LayerNorm pada z_fused (stabilisasi skala sebelum diffusion)
        # [FIX paper] LayerNorm pada fused_dim (z concat c_concept)
        self.layer_norm = nn.LayerNorm(fused_dim)

    # ── Embedding input ───────────────────────────────────────────────────

    def _embed_input(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Lookup embedding per kolom dan concat. [batch, n_cols] → [batch, total_emb_dim]"""
        return torch.cat([
            self.embeddings[i](x_cat[:, i]) for i in range(self.n_cols)
        ], dim=1)

    # ── [B] Main Encoder ──────────────────────────────────────────────────

    def _encode_to_params(self, x_emb: torch.Tensor):
        """Main Encoder q_phi(z|x). Return: (mu, log_var, T_concept)"""
        h       = self.encoder_mlp(x_emb)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        T_concept = torch.exp(0.5 * log_var)
        return mu, log_var, T_concept

    # ── [A] Prior Concept Encoder ─────────────────────────────────────────

    def _encode_prior_to_params(self, x_emb: torch.Tensor):
        """Prior Concept Encoder. Return: (mu_prior, log_var_prior, T_prior)"""
        h             = self.prior_encoder_mlp(x_emb)
        mu_prior      = self.fc_mu_prior(h)
        log_var_prior = self.fc_log_var_prior(h)
        log_var_prior = torch.clamp(log_var_prior, min=-10.0, max=10.0)
        T_prior = torch.exp(0.5 * log_var_prior)
        return mu_prior, log_var_prior, T_prior

    # ── [C] Gumbel-Softmax Reparameterization ─────────────────────────────

    @staticmethod
    def _sample_gumbel(shape, device, eps: float = 1e-6) -> torch.Tensor:
        """Sampling Gumbel(0, 1): g = -log(-log(U)), U ~ Uniform(0,1)."""
        U = torch.rand(shape, device=device).clamp(eps, 1.0 - eps)
        return -torch.log(-torch.log(U))

    def _gumbel_softmax_concept(self, T_concept: torch.Tensor, T_prior: torch.Tensor):
        """
        Gumbel-Softmax Reparameterization Trick. Paper Eq. 3 & 4.
        Return: (c_concept, q_c_prior)
        """
        device = T_concept.device
        T_c = torch.clamp(T_concept, min=1e-8)
        T_p = torch.clamp(T_prior,   min=1e-8)

        if self.training:
            g_concept = self._sample_gumbel(T_c.shape, device)
            g_prior   = self._sample_gumbel(T_p.shape, device)
        else:
            g_concept = torch.zeros_like(T_c)
            g_prior   = torch.zeros_like(T_p)

        logit_c = (torch.log(T_c) + g_concept) / self.tau
        logit_p = (torch.log(T_p) + g_prior)   / self.tau

        denom_log = torch.logaddexp(logit_c, logit_p)
        c_concept  = torch.exp(logit_c - denom_log)   # Eq. 3
        q_c_prior  = torch.exp(logit_p - denom_log)   # Eq. 4

        return c_concept, q_c_prior

    # ── [D] Normal Reparameterization ─────────────────────────────────────

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Normal Reparameterization Trick. Paper Eq. 5.
        Inference (eval mode): return mu deterministik.
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
        Encode integer index ke z_fused deterministik untuk inference.
        Pakai mu (bukan sample z) agar representasi stabil dan reproducible.
        Output shape: [N, fused_dim] = [N, 2 * latent_dim]
        """
        x_emb = self._embed_input(x_cat)
        mu, log_var, T_concept = self._encode_to_params(x_emb)
        _, _, T_prior          = self._encode_prior_to_params(x_emb)
        c_concept, _           = self._gumbel_softmax_concept(T_concept, T_prior)
        # Deterministik: pakai mu (tanpa sampling) untuk inference yang stabil
        z_fused                = torch.cat([mu, c_concept], dim=1)
        return self.layer_norm(z_fused)

    def encode_with_params(self, x_cat: torch.Tensor):
        """
        Encode dan kembalikan semua parameter untuk PT-VAE loss.
        Return: (z_fused, mu, log_var, c_concept, q_c_prior, mu_prior, log_var_prior)
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
        """Main Decoder: z_fused → per-kolom logits. Paper Eq. 6."""
        return self._logits_from_recon(self._decode_from_z(z))

    def decode_prior(self, c_concept: torch.Tensor) -> list:
        """Prior Concept Decoder: c_concept → per-kolom logits. Untuk L_recon (Eq. 12)."""
        return self._logits_from_recon(self._decode_prior_concept(c_concept))

    def forward(self, x_cat: torch.Tensor, add_noise: bool = False):
        """
        Forward pass PT-VAE untuk training. Algorithm 1, Lines 3-7.

        Return:
          z_fused, mu, log_var, c_concept, q_c_prior, mu_prior, log_var_prior,
          class_logits, recon_logits, recon_prior_logits
        """
        (z_fused, mu, log_var, c_concept, q_c_prior,
         mu_prior, log_var_prior) = self.encode_with_params(x_cat)

        class_logits       = self.classify(z_fused)
        recon_logits       = self.decode(z_fused)
        recon_prior_logits = self.decode_prior(c_concept)

        return (z_fused, mu, log_var, c_concept, q_c_prior,
                mu_prior, log_var_prior, class_logits,
                recon_logits, recon_prior_logits)

    # ── PT-VAE Loss Functions (Section 3.3) ──────────────────────────────

    @staticmethod
    def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || p(z)). Paper Eq. 9. Closed-form, rata-rata per sample.
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
        """KL(q(c|x) || p(c)). Paper Eq. 11.
        c_concept per dimensi adalah output binary Gumbel-Softmax in (0,1)
        dimana c_concept + q_c_prior = 1.0 per dimensi (binary competition).
        Prior p(c) = Bernoulli(0.5) adalah prior natural untuk binary variable.
        [FIX v2] Normalisasi / latent_dim agar KL tidak meledak saat
                 latent_dim besar. Target skala: ~0.1-0.5.
        """
        c_s      = torch.clamp(c_concept, min=1e-7, max=1.0 - 1e-7)
        log2     = torch.log(torch.tensor(2.0, device=c_concept.device))
        H_q      = -(c_s * torch.log(c_s) + (1.0 - c_s) * torch.log(1.0 - c_s))
        kl_per_dim = log2 - H_q
        latent_dim = c_concept.shape[1]
        return kl_per_dim.sum(dim=1).mean() / latent_dim

    @staticmethod
    def reconstruction_loss_concept(recon_prior_logits: list,
                                     recon_logits: list,
                                     n_cols: int) -> torch.Tensor:
        """L_recon = ||x'_concept - x'||^2. Paper Eq. 12. MSE antara logit.
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
    def kl_divergence_concept_prior(q_c_prior: torch.Tensor,
                                     c_concept: torch.Tensor) -> torch.Tensor:
        """L_KL = KL(q(c_prior|x) || q(c_concept|x)). Paper Eq. 13.
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


# Alias untuk kompatibilitas
VAEEmbeddingModel = PTVAEEmbeddingModel


# ===========================================================================
#  Training PT-VAE Embedding (Liu et al., 2025)
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

    Loss Total (Eq. 14):
        L_total = L_ELBO + L_recon + L_KL + L_class
        L_ELBO  = CE_recon + KL_z + KL_c
        L_recon = ||x'_concept - x'||^2   (Eq. 12)
        L_KL    = KL(q(c_prior|x) || q(c_concept|x))  (Eq. 13)
        L_class = CrossEntropy(class_logits, labels)   (auxiliary)

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
        tau           = 1.0,
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

    K = model.latent_dim

    # Bobot loss: paper Section 3.3 menggunakan EQUAL WEIGHTS untuk semua komponen.
    # "we adopted equal weights to both terms during the learning process"
    # L_total = L_ELBO + L_recon + L_KL  (Eq. 14, semua bobot = 1.0)
    # L_class ditambahkan sebagai auxiliary loss dengan bobot 1.0.
    # KL(c) dinormalisasi / latent_dim agar skalanya konsisten dengan KL(z).

    best_loss        = float('inf')
    patience_counter = 0
    best_model_state = None

    model.train()
    for epoch in range(n_epochs):

        total_loss        = 0.0
        total_elbo_recon  = 0.0
        total_kl_z        = 0.0
        total_kl_c        = 0.0
        total_l_elbo      = 0.0
        total_recon_loss  = 0.0
        total_kl_loss     = 0.0
        total_class_loss  = 0.0
        n_batches         = 0

        for batch_cat, batch_labels in loader:
            optimizer.zero_grad()

            (z_fused, mu, log_var, c_concept, q_c_prior,
             mu_prior, log_var_prior, class_logits,
             recon_logits, recon_prior_logits) = model(batch_cat)

            # L_ELBO (Eq. 10)
            elbo_recon = sum(
                ce_loss(recon_logits[i], batch_cat[:, i])
                for i in range(model.n_cols)
            ) / model.n_cols

            # L_ELBO = CE + KL(z) + KL(c) -- Paper Eq. 10, equal weights
            kl_z   = PTVAEEmbeddingModel.kl_divergence(mu, log_var)
            kl_c   = PTVAEEmbeddingModel.kl_divergence_c(c_concept, K)
            l_elbo = elbo_recon + kl_z + kl_c

            # L_recon (Eq. 12)
            l_recon = PTVAEEmbeddingModel.reconstruction_loss_concept(
                recon_prior_logits, recon_logits, model.n_cols
            )

            # L_KL (Eq. 13) — bobot kecil fixed
            l_kl = PTVAEEmbeddingModel.kl_divergence_concept_prior(
                q_c_prior, c_concept
            )

            # L_class (auxiliary)
            class_loss = ce_loss(class_logits, batch_labels)

            # Total Loss (Eq. 14): L_ELBO + L_recon + L_KL, equal weights sesuai paper
            loss = l_elbo + l_recon + l_kl + class_loss

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
            losses = {
                'KL(z)':   avg_kl_z,
                'KL(c)':   avg_kl_c,
                'L_recon': avg_recon_loss,
                'L_KL':    avg_kl_loss,
            }
            # [FIX] Warning threshold yang lebih relevan: cek nilai negatif
            for name, val in losses.items():
                if val < 0:
                    print(f'  [WARN] {name}={val:.4f} NEGATIF — periksa implementasi loss!')
                elif val < 1e-8:
                    print(f'  [WARN] {name} mendekati nol ({val:.2e}) — '
                          f'komponen ini mungkin tidak aktif!')
            vals = list(losses.values())
            if len(vals) > 0 and max(vals) > 10 * (sum(vals) / len(vals)):
                dominant = max(losses, key=losses.get)
                print(f'  [WARN] {dominant}={losses[dominant]:.4f} mendominasi loss '
                      f'(target masing-masing komponen ≈ 0.1-1.0)')

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
#  Encode / Decode helpers
# ===========================================================================

def encode_with_embedding(model: PTVAEEmbeddingModel,
                          cat_idx_array: np.ndarray,
                          device: str,
                          batch_size: int = 4096) -> np.ndarray:
    """
    Encode integer index → embedding numpy array menggunakan PT-VAE encoder.

    Saat inference (eval mode), model.encode() mengembalikan z_fused = mu + c_concept
    secara deterministik — tanpa sampling dan tanpa Gumbel noise.

    Return : np.ndarray [N, latent_dim]
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
            z = model.encode(batch)
            all_z.append(z.cpu().numpy())

    return np.concatenate(all_z, axis=0).astype(np.float32)


def decode_cat_from_embedding(model: PTVAEEmbeddingModel,
                              emb_array: np.ndarray,
                              device: str,
                              batch_size: int = 4096) -> np.ndarray:
    """
    Decode embedding (z_fused) → prediksi kelas tiap kolom (argmax logits).

    PT-VAE Main Decoder: z_fused → Decoder MLP → per-kolom logits → argmax

    emb_array : [N, latent_dim]
    Return    : [N, n_cat_cols]  — predicted integer index
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


# ===========================================================================
#  Load Dataset
# ===========================================================================

def load_dataset(dataname, idx=0, mask_type='MCAR', ratio='30'):
    """
    Load dataset dengan PT-VAE embedding learning.

    Embedding diganti dengan PTVAEEmbeddingModel (Liu et al., 2025).
    Pipeline load data, masking, dan struktur output tetap sama
    dengan dataset_class.py original.

    Parameters
    ----------
    dataname : str
        Nama dataset (e.g., 'adult')
    idx : int
        Split index (default 0).
    mask_type : str
        Tipe masking ('MCAR', 'MNAR_logistic_T2', dll)
    ratio : str or int
        Rasio missing (e.g., '30', 30)

    Return
    ------
    train_X           : [N_train, num_num + total_emb_dim]  float32
    test_X            : [N_test,  num_num + total_emb_dim]  float32
    ori_train_mask    : mask asli train [N_train, total_cols]
    ori_test_mask     : mask asli test  [N_test,  total_cols]
    train_num         : [N_train, num_num]
    test_num          : [N_test,  num_num]
    train_cat_idx     : [N_train, n_cat_cols] integer index  (atau None)
    test_cat_idx      : [N_test,  n_cat_cols] integer index  (atau None)
    extend_train_mask : mask yang sudah diperluas ke dimensi X
    extend_test_mask  : mask yang sudah diperluas ke dimensi X
    cat_bin_num       : None  (tidak digunakan)
    emb_model         : PTVAEEmbeddingModel  (atau None jika no cat)
    emb_sizes         : list[int] dimensi embedding per kolom (atau None)
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

    # ── Fitur numerik ────────────────────────────────────────────────────
    data_num  = data_df[cols[num_col_idx]].values.astype(np.float32)
    train_num = train_df[cols[num_col_idx]].values.astype(np.float32)
    test_num  = test_df[cols[num_col_idx]].values.astype(np.float32)

    # ── Extract labels untuk supervised learning ─────────────────────────
    train_y = train_df[cols[target_col_idx]]
    test_y  = test_df[cols[target_col_idx]]

    label_encoder = LabelEncoder()
    all_labels = pd.concat([train_y, test_y]).values.ravel()
    label_encoder.fit(all_labels.astype(str))

    train_labels = label_encoder.transform(train_y.values.ravel().astype(str))
    test_labels  = label_encoder.transform(test_y.values.ravel().astype(str))
    n_classes    = len(label_encoder.classes_)

    print(f'[Dataset] Detected {n_classes} classes for supervised learning')
    print(f'[Dataset] Classes: {label_encoder.classes_}')

    # ── Kasus: hanya fitur numerik ───────────────────────────────────────
    if len(cat_col_idx) == 0:
        train_X = train_num
        test_X  = test_num

        extend_train_mask = train_mask[:, num_col_idx]
        extend_test_mask  = test_mask[:, num_col_idx]

        return (train_X, test_X,
                train_mask, test_mask,
                train_num, test_num,
                None, None,
                extend_train_mask, extend_test_mask,
                None,   # cat_bin_num (legacy, tidak dipakai)
                None,   # emb_model
                None)   # emb_sizes

    # ── Kasus: ada fitur kategorikal → PT-VAE Embedding ──────────────────
    cat_columns = cols[cat_col_idx]

    data_cat  = data_df[cat_columns].astype(str)
    train_cat = train_df[cat_columns].astype(str)
    test_cat  = test_df[cat_columns].astype(str)

    # Label encoding: fit pada seluruh data (data.csv) agar konsisten
    encoders           = {}
    cat_dims           = []
    train_cat_idx_list = []
    test_cat_idx_list  = []

    for col in cat_columns:
        le = LabelEncoder()
        le.fit(data_cat[col])
        encoders[col] = le
        cat_dims.append(len(le.classes_))

        train_cat_idx_list.append(
            le.transform(train_cat[col]).astype(np.int64)
        )
        test_cat_idx_list.append(
            le.transform(test_cat[col]).astype(np.int64)
        )

    train_cat_idx = np.stack(train_cat_idx_list, axis=1)  # [N_train, n_cat]
    test_cat_idx  = np.stack(test_cat_idx_list,  axis=1)  # [N_test,  n_cat]

    # Hitung ukuran embedding sesuai rumus Guo & Berkhahn (2016)
    emb_sizes = [compute_embedding_size(n) for n in cat_dims]

    print(f'[Embedding] cat_dims={cat_dims}, emb_sizes={emb_sizes}, '
          f'total_emb_dim={sum(emb_sizes)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Latih PT-VAE embedding model
    print('[PT-VAE Embedding] Melatih PTVAEEmbeddingModel '
          '(ELBO: Reconstruction + KL Divergence + Classification loss) ...')
    print('[PT-VAE Embedding] Referensi: Liu et al. (2025) PT-VAE: Variational Autoencoder with Prior Concept Transformation')
    emb_model = train_vae_embedding_model(
        cat_idx_array = train_cat_idx,
        labels        = train_labels,
        cat_dims      = cat_dims,
        emb_sizes     = emb_sizes,
        n_classes     = n_classes,
        device        = device,
        n_epochs      = 150,         # Paper Section 4.1: "number of epochs was set to 150"
        batch_size    = 128,         # Paper Section 4.1: "batch size of 128"
        lr            = 1e-3,        # Paper Section 4.1: "learning rate of 0.001"
        dropout       = 0.1,
        hidden_dim    = 256,
        latent_dim    = None,        # default = total_emb_dim
        encoder_ratio = 1.5,
        patience      = 30,          # default patience, paper tidak menyebut early stopping
    )
    print('[PT-VAE Embedding] Training selesai. Parameter di-freeze untuk diffusion.')

    # Encode: integer index → embedding vector (z = mu + c_concept, deterministik)
    train_cat_emb = encode_with_embedding(emb_model, train_cat_idx, device)
    test_cat_emb  = encode_with_embedding(emb_model, test_cat_idx,  device)
    # shape: [N, fused_dim] = [N, 2 * latent_dim]

    # Gabungkan numerik + embedding
    train_X = np.concatenate([train_num, train_cat_emb], axis=1)
    test_X  = np.concatenate([test_num,  test_cat_emb],  axis=1)

    # ── Buat extended mask ───────────────────────────────────────────────
    train_num_mask = train_mask[:, num_col_idx]
    train_cat_mask = train_mask[:, cat_col_idx]
    test_num_mask  = test_mask[:, num_col_idx]
    test_cat_mask  = test_mask[:, cat_col_idx]

    # fused_dim = 2 * latent_dim: setiap kolom kat diperluas ke fused_dim_per_col
    n_cat_cols      = len(emb_sizes)
    fused_dim_total = emb_model.fused_dim                 # 2 * latent_dim
    # Setiap kolom kategorikal mendapat porsi fused_dim_total / n_cat_cols
    # tapi yang lebih sederhana: broadcast semua kolom cat ke fused_dim_total
    # dengan bobot rata karena z_fused adalah representasi gabungan semua kolom.
    fused_per_col   = fused_dim_total // n_cat_cols
    fused_remainder = fused_dim_total  % n_cat_cols
    # Distribusikan fused_dim ke tiap kolom (kolom terakhir dapat sisa)
    fused_sizes_arr = np.array(
        [fused_per_col + (1 if j < fused_remainder else 0) for j in range(n_cat_cols)],
        dtype=int
    )

    def extend_mask_emb(mask: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        """
        Perluas mask dari [N, n_cat_cols] ke [N, sum(sizes)].
        Setiap kolom kategorikal ke-j diperluas ke sizes[j] kolom.
        """
        N       = mask.shape[0]
        cum     = np.concatenate(([0], sizes.cumsum()))
        result  = np.zeros((N, sizes.sum()), dtype=bool)
        for j in range(len(sizes)):
            col_mask = mask[:, j][:, np.newaxis]
            result[:, cum[j]:cum[j + 1]] = np.tile(col_mask, sizes[j])
        return result

    ext_train_cat_mask = extend_mask_emb(train_cat_mask, fused_sizes_arr)
    ext_test_cat_mask  = extend_mask_emb(test_cat_mask,  fused_sizes_arr)

    extend_train_mask = np.concatenate([train_num_mask, ext_train_cat_mask], axis=1)
    extend_test_mask  = np.concatenate([test_num_mask,  ext_test_cat_mask],  axis=1)

    return (train_X, test_X,
            train_mask, test_mask,
            train_num, test_num,
            train_cat_idx, test_cat_idx,
            extend_train_mask, extend_test_mask,
            None,       # cat_bin_num (legacy, tidak dipakai)
            emb_model,  # PTVAEEmbeddingModel
            emb_sizes)  # list[int]


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

def get_eval(dataname, X_recon, X_true, truth_cat_idx,
             num_num, emb_model, emb_sizes, mask,
             device='cpu', oos=False):
    """
    Hitung MAE, RMSE (numerik) dan Accuracy (kategorikal).

    Konvensi input:
    ---------------
    X_recon[:, :num_num]   → fitur numerik dalam skala TERNORMALISASI (X-mean)/std
    X_recon[:, num_num:]   → embedding kategorikal dalam skala ASLI
                             (sudah di-invers-norm: × std_emb + mean_emb)
                             → siap dikirim ke Main Decoder PTVAEEmbeddingModel

    emb_model sekarang adalah PTVAEEmbeddingModel, decode() tetap kompatibel.
    """
    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    num_mask = mask[:, num_col_idx].astype(bool)
    cat_mask = mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else None

    num_pred = X_recon[:, :num_num]
    num_true = X_true[:, :num_num]

    cat_emb_pred = X_recon[:, num_num:]

    # Special case: news dataset
    if dataname == 'news' and oos:
        drop = 6265
        num_mask  = np.delete(num_mask, drop, axis=0)
        num_pred  = np.delete(num_pred, drop, axis=0)
        num_true  = np.delete(num_true, drop, axis=0)
        if cat_mask is not None:
            cat_mask = np.delete(cat_mask, drop, axis=0)
        if truth_cat_idx is not None:
            truth_cat_idx = np.delete(truth_cat_idx, drop, axis=0)
        cat_emb_pred = np.delete(cat_emb_pred, drop, axis=0)

    # ── Numerik: MAE & RMSE (hanya pada posisi missing) ─────────────────
    div  = num_pred[num_mask] - num_true[num_mask]
    mae  = np.abs(div).mean()
    rmse = np.sqrt((div ** 2).mean())

    # ── Kategorikal: Akurasi via PT-VAE Main Decoder ─────────────────────
    acc = np.nan
    if (truth_cat_idx is not None
            and len(cat_col_idx) > 0
            and emb_model is not None
            and emb_sizes is not None):

        # Decode embedding → prediksi kelas per kolom via PT-VAE Main Decoder
        pred_cat_idx = decode_cat_from_embedding(
            emb_model, cat_emb_pred, device
        )  # [N, n_cat_cols]

        correct_total  = 0
        total_missing  = 0
        emb_sizes_arr  = np.array(emb_sizes, dtype=int)

        for j in range(len(cat_col_idx)):
            rows_miss = cat_mask[:, j]
            if rows_miss.sum() == 0:
                continue

            pred_j  = pred_cat_idx[:, j]
            true_j  = truth_cat_idx[:, j].astype(int)

            correct = (pred_j[rows_miss] == true_j[rows_miss]).sum()
            correct_total += int(correct)
            total_missing += int(rows_miss.sum())

        if total_missing > 0:
            acc = correct_total / total_missing

    return mae, rmse, acc