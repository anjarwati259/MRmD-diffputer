import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import json

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


def compute_embedding_size(n_categories: int) -> int:
    """
    Hitung ukuran embedding optimal berdasarkan jumlah kategori.
    Rumus: min(600, round(1.6 * n_categories^0.56))
    Referensi: Guo & Berkhahn (2016)
    """
    return min(600, round(1.6 * n_categories ** 0.56))


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

      Decoder  g_θ'  (Section 2.2, eq. 1):
        Representasi laten y dipetakan kembali ke ruang input melalui satu
        lapisan affine + sigmoid:
            z = g_θ'(y) = sigmoid(W'y + b')
        W' ∈ R^{d × d'},  b' ∈ R^{d}.

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
        self.W_enc = nn.Linear(self.total_onehot, hidden_dim, bias=True)

        # ── Decoder  g_θ': y → z  ────────────────────────────────────────
        # Jika tied_weights: W' = W^T  → hanya bias b' yang dilatih terpisah
        # Jika tidak tied: W' bebas (nn.Linear penuh)
        if tied_weights:
            self.b_dec = nn.Parameter(torch.zeros(self.total_onehot))
        else:
            self.W_dec = nn.Linear(hidden_dim, self.total_onehot, bias=True)

        # dropout untuk regularisasi ringan
        self.dropout = nn.Dropout(dropout)

        # lookup: offset awal tiap kolom dalam vektor one-hot gabungan
        self._col_offsets = [0]
        for d in cat_dims[:-1]:
            self._col_offsets.append(self._col_offsets[-1] + d)

    def _to_onehot(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Konversi integer index [batch, n_cols] → one-hot gabungan [batch, total_onehot].
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
        """
        x_tilde = x_oh.clone()
        batch   = x_oh.shape[0]

        for i, n_cat in enumerate(self.cat_dims):
            start = self._col_offsets[i]
            end   = start + n_cat

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

        x_cat  : [batch, n_cols]  — integer index tiap kolom
        return : [batch, hidden_dim]   — representasi laten y
        """
        x_oh = self._to_onehot(x_cat)           # x ∈ {0,1}^d
        x_oh = self.dropout(x_oh)
        y    = torch.sigmoid(self.W_enc(x_oh))  # y = sigmoid(Wx + b)
        return y                                  # [batch, hidden_dim]

    def decode(self, y: torch.Tensor) -> list:
        """
        g_θ': representasi laten y → logits rekonstruksi per kolom (RAW logits).

        y      : [batch, hidden_dim]
        return : list[n_cols] of [batch, vocab_size_i]  — RAW logits
        """
        if self.tied_weights:
            z_raw = torch.nn.functional.linear(y, self.W_enc.weight.t(), self.b_dec)
        else:
            z_raw = self.W_dec(y)  # [batch, total_onehot] — RAW logits

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
          x_clean → one_hot → corrupt qD → f_θ → y → g_θ' → recon_logits

        Inference (self.training=False):
          x_clean → one_hot → f_θ → y  (tanpa corruption)

        return : (y, None, recon_logits)
        """
        if self.training and self.corruption_prob > 0:
            x_oh    = self._to_onehot(x_cat)
            x_input = self._corrupt_onehot(x_oh)
            x_input = self.dropout(x_input)
            y       = torch.sigmoid(self.W_enc(x_input))   # f_θ(x̃)
        else:
            y = self.encode(x_cat)

        recon_logits = self.decode(y)   # g_θ'(y)

        return y, None, recon_logits


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
    ce_loss   = nn.CrossEntropyLoss()

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
            y, _, recon_logits = model(batch_cat)

            recon_loss = 0.0
            for i in range(model.n_cols):
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
            with torch.no_grad():
                correct_total = 0
                total_cols    = 0
                for i in range(model.n_cols):
                    pred_i = recon_logits[i].argmax(dim=1)
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
        z_sample   = model.encode(sample_cat)
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
        model             = model,
        cat_idx_array     = eval_sample,
        device            = device,
        corruption_levels = [0.0, 0.1, 0.2, 0.3, 0.5],
        corruption_type   = corruption_type,
        verbose           = True,
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
                    x_oh       = model._to_onehot(x_clean)         # [B, total_OH]
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
                    y          = torch.sigmoid(model.W_enc(x_tilde_oh))
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

        avg_ce      = total_ce_loss / n_batches
        per_col_acc = col_correct / np.maximum(col_total, 1)
        overall_acc = col_correct.sum() / np.maximum(col_total.sum(), 1)

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


def encode_with_embedding(model: DAEEmbeddingModel,
                          cat_idx_array: np.ndarray,
                          device: str,
                          batch_size: int = 4096) -> np.ndarray:
    """
    Encode integer index → embedding numpy array.
    Inference: encode langsung dari x_clean tanpa corruption.

    Return : np.ndarray [N, hidden_dim]
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

    emb_array : [N, hidden_dim]  — output encode()
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


# ===========================================================================
#  Load Dataset
# ===========================================================================

def load_dataset(dataname, idx=0, mask_type='MCAR', ratio='30'):
    """
    Load dataset dengan supervised embedding learning.

    Perubahan utama:
    - Membaca label dari target_col_idx di info.json
    - Melatih embedding model secara supervised dengan label
    - Arsitektur disejajarkan dengan versi unsupervised untuk perbandingan adil
    - Tetap menggunakan struktur output yang sama untuk kompatibilitas

    Parameters
    ----------
    dataname : str
        Nama dataset (e.g., 'adult')
    idx : int
        Split index (default 0). Bisa juga dipanggil sebagai 'split_idx' untuk kompatibilitas
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
    train_num         : [N_train, num_num]  — hanya numerik
    test_num          : [N_test,  num_num]
    train_cat_idx     : [N_train, n_cat_cols] integer index  (atau None)
    test_cat_idx      : [N_test,  n_cat_cols] integer index  (atau None)
    extend_train_mask : mask yang sudah diperluas ke dimensi X
    extend_test_mask  : mask yang sudah diperluas ke dimensi X
    cat_bin_num       : None  (tidak digunakan lagi, digantikan emb_sizes)
    emb_model         : SupervisedLearnableEmbeddingModel  (atau None jika no cat)
    emb_sizes         : list[int] dimensi embedding per kolom (atau None)
    """
    # Convert ratio to string if needed
    ratio = str(ratio)

    # ── Paths sama persis dengan original ────────────────────────────────
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

    # ── Load files sama persis dengan original ───────────────────────────
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

    # Label encoding untuk supervised learning
    label_encoder = LabelEncoder()
    # Fit pada train dan test untuk konsistensi
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

    # ── Kasus: ada fitur kategorikal → Supervised Learnable Embedding ────
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
        le.fit(data_cat[col])             # fit pada semua data
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

    # Tentukan device (gunakan CUDA jika tersedia; main.py akan overwrite)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Latih supervised embedding model menggunakan data TRAIN + LABELS
    print('[Embedding] Melatih SupervisedLearnableEmbeddingModel '
          '(classification + reconstruction loss) ...')
    emb_model = train_supervised_embedding_model(
        cat_idx_array = train_cat_idx,
        labels        = train_labels,
        cat_dims      = cat_dims,
        emb_sizes     = emb_sizes,
        n_classes     = n_classes,
        device        = device,
        n_epochs      = 1000,      # maksimum epochs
        batch_size    = 1024,
        lr            = 1e-3,
        dropout       = 0.1,
        hidden_dim    = 256,
        use_mlp       = True,      # [DISESUAIKAN] 1 hidden layer setelah concat
        mlp_ratio     = 1.5,       # [DISESUAIKAN] sama dengan versi unsupervised
        noise_std     = 0.01,      # [DISESUAIKAN] noise sebelum decoding
        patience      = 40,        # early stopping patience
    )
    print('[Embedding] Training selesai. Parameter di-freeze untuk diffusion.')

    # Encode: integer index → embedding vector
    train_cat_emb = encode_with_embedding(emb_model, train_cat_idx, device)
    test_cat_emb  = encode_with_embedding(emb_model, test_cat_idx,  device)
    # shape: [N, total_emb_dim]

    # Gabungkan numerik + embedding
    train_X = np.concatenate([train_num, train_cat_emb], axis=1)
    test_X  = np.concatenate([test_num,  test_cat_emb],  axis=1)

    # ── Buat extended mask ───────────────────────────────────────────────
    # Setiap kolom kategorikal dalam mask asli → diperluas ke emb_size kolom
    # (seluruh dimensi embedding kolom tsb dianggap missing/observed bersama)
    train_num_mask = train_mask[:, num_col_idx]
    train_cat_mask = train_mask[:, cat_col_idx]
    test_num_mask  = test_mask[:, num_col_idx]
    test_cat_mask  = test_mask[:, cat_col_idx]

    emb_sizes_arr = np.array(emb_sizes, dtype=int)

    def extend_mask_emb(mask: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        """
        Perluas mask dari [N, n_cat_cols] → [N, total_emb_dim].
        Setiap kolom kategorikal ke-j diperluas ke sizes[j] kolom.
        """
        N       = mask.shape[0]
        cum     = np.concatenate(([0], sizes.cumsum()))
        result  = np.zeros((N, sizes.sum()), dtype=bool)
        for j in range(len(sizes)):
            col_mask = mask[:, j][:, np.newaxis]           # [N, 1]
            result[:, cum[j]:cum[j + 1]] = np.tile(col_mask, sizes[j])
        return result

    # ext_train_cat_mask = extend_mask_emb(train_cat_mask, emb_sizes_arr)
    # ext_test_cat_mask  = extend_mask_emb(test_cat_mask,  emb_sizes_arr)

    # extend_train_mask = np.concatenate([train_num_mask, ext_train_cat_mask], axis=1)
    # extend_test_mask  = np.concatenate([test_num_mask,  ext_test_cat_mask],  axis=1)

    # SEBELUM (salah — memakai emb_sizes per kolom):
    emb_sizes_arr = np.array(emb_sizes, dtype=int)
    ext_train_cat_mask = extend_mask_emb(train_cat_mask, emb_sizes_arr)
    ext_test_cat_mask  = extend_mask_emb(test_cat_mask,  emb_sizes_arr)

    # SESUDAH (benar — DAE output = hidden_dim, bukan sum(emb_sizes)):
    total_emb_dim = emb_model.hidden_dim  # 256
    # Jika ada kolom cat yang missing, seluruh 256 dim dianggap missing
    # Caranya: OR across all cat columns → satu mask [N, 1] → tile ke [N, 256]
    cat_any_missing = train_cat_mask.any(axis=1, keepdims=True)  # [N, 1]
    ext_train_cat_mask = np.tile(cat_any_missing, (1, total_emb_dim))  # [N, 256]

    cat_any_missing_test = test_cat_mask.any(axis=1, keepdims=True)
    ext_test_cat_mask  = np.tile(cat_any_missing_test, (1, total_emb_dim))

    extend_train_mask = np.concatenate([train_num_mask, ext_train_cat_mask], axis=1)
    extend_test_mask  = np.concatenate([test_num_mask,  ext_test_cat_mask],  axis=1)
    # Sekarang shape: [N, num_num + 256] = [N, 266] ✓ cocok dengan train_X

    return (train_X, test_X,
            train_mask, test_mask,
            train_num, test_num,
            train_cat_idx, test_cat_idx,
            extend_train_mask, extend_test_mask,
            None,       # cat_bin_num (legacy, tidak dipakai)
            emb_model,  # SupervisedLearnableEmbeddingModel
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

    Konvensi input — mengikuti pola asli DiffPutter:
    ------------------------------------------------
    X_recon[:, :num_num]   → fitur numerik dalam skala TERNORMALISASI (X-mean)/std
                             (belum di-denorm ke skala asli, konsisten dengan paper)
    X_recon[:, num_num:]   → embedding kategorikal dalam skala ASLI
                             (sudah di-invers-norm: × std_emb + mean_emb)
                             → siap dikirim ke Linear Decoder emb_model

    Mengapa numerik TIDAK di-denorm?
    Kode asli DiffPutter (binary encoding) juga menghitung MAE/RMSE pada skala
    (X-mean)/std, bukan skala asli. Kita ikuti konvensi yang sama agar hasil
    bisa dibandingkan secara apple-to-apple.
    """
    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    # mask: True(1) = missing, False(0) = observed
    num_mask = mask[:, num_col_idx].astype(bool)
    cat_mask = mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else None

    num_pred = X_recon[:, :num_num]
    num_true = X_true[:, :num_num]

    # Bagian embedding dari rekonstruksi
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

    # ── Kategorikal: Akurasi via Linear Decoder ──────────────────────────
    acc = np.nan
    if (truth_cat_idx is not None
            and len(cat_col_idx) > 0
            and emb_model is not None
            and emb_sizes is not None):

        # Decode embedding → prediksi kelas per kolom
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

            # Hitung akurasi hanya pada posisi missing
            correct = (pred_j[rows_miss] == true_j[rows_miss]).sum()
            correct_total += int(correct)
            total_missing += int(rows_miss.sum())

        if total_missing > 0:
            acc = correct_total / total_missing

    return mae, rmse, acc