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
#
#  Corrupt generation (Eq. 3 paper):
#    x_tilde = m ⊙ x_bar + (1 - m) ⊙ x
#    di mana x_bar[j] ~ empirical marginal distribution fitur ke-j
#
#  Input pipeline:
#    cat_idx_array [N, n_cols]  — integer label index (kategorikal)
#    Sebelum masuk VIME encoder, index di-one-hot encode → float tensor
#    sehingga input_dim = sum(all_dims) (total one-hot size)
#
#  Output (encode):
#    z [N, hidden_dim]  — representasi laten (output encoder e)
#    Dipakai sebagai "embedding" pengganti SupervisedLearnableEmbeddingModel.
#
#  Decode (untuk evaluasi kategorikal):
#    sr decoder → logits [N, input_dim] → split per kolom → argmax
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
      lr = MSE per dimensi

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
        all_dims   : list[n_cols] — vocab size tiap kolom (n_unique per kolom kategorikal)
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
#  Training VIME Encoder
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

    Signature dipertahankan agar main_class.py tidak perlu diubah.
    Parameter yang tidak relevan untuk VIME (labels, n_classes, dropout, use_mlp,
    mlp_ratio, noise_std) diterima tapi diabaikan.

    Loss function sesuai paper (Eq. 4-6):
      L = lm(m, m_hat) + alpha * lr(x, x_hat)
      lm : BCE per dimensi one-hot (mask estimation)
      lr : MSE per dimensi one-hot (feature reconstruction)
           sesuai implementasi GitHub resmi (vime_self.py) yang menggunakan
           MSE atas seluruh vektor fitur (termasuk one-hot categorical).
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


def encode_with_embedding(model: 'VIMEEmbeddingModel',
                          cat_idx_array: np.ndarray,
                          device: str,
                          batch_size: int = 4096) -> np.ndarray:
    """
    Encode integer index → representasi laten VIME encoder (z).

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


# ===========================================================================
#  Load Dataset
# ===========================================================================

def load_dataset(dataname, idx=0, mask_type='MCAR', ratio='30'):
    """
    Load dataset dengan VIME self-supervised embedding untuk kolom kategorikal.

    Perubahan dari versi supervised:
    - Embedding menggunakan VIME self-supervised encoder (bukan supervised)
    - train_X / test_X : [N, hidden_dim] — seluruh embedding VIME
    - len_num = 0 di main karena tidak ada kolom raw numerik di train_X

    Output tambahan (dibanding versi supervised):
    - train_num / test_num tetap dikembalikan (nilai float asli, ternormalisasi)
      untuk keperluan evaluasi MAE/RMSE di skala normalisasi

    Parameters
    ----------
    dataname : str
    idx : int
    mask_type : str
    ratio : str or int

    Return
    ------
    train_X           : [N_train, hidden_dim]  float32
    test_X            : [N_test,  hidden_dim]  float32
    ori_train_mask    : mask asli train [N_train, total_cols]
    ori_test_mask     : mask asli test  [N_test,  total_cols]
    train_num         : [N_train, num_num]  — hanya numerik (ternormalisasi)
    test_num          : [N_test,  num_num]
    train_cat_idx     : [N_train, n_cat_cols] integer index  (atau None)
    test_cat_idx      : [N_test,  n_cat_cols] integer index  (atau None)
    extend_train_mask : mask yang sudah diperluas ke dimensi hidden_dim
    extend_test_mask  : mask yang sudah diperluas ke dimensi hidden_dim
    cat_bin_num       : None  (tidak digunakan lagi)
    emb_model         : VIMEEmbeddingModel  (atau None jika no cat)
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

    # ── Extract labels (untuk kompatibilitas signature) ──────────────────
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

    # ── Kasus: ada fitur kategorikal → VIME Embedding ────────────────────
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

    # VIME hidden_dim: ukuran representasi encoder (output VIME = [N, hidden_dim])
    vime_hidden_dim = 256
    emb_sizes = [vime_hidden_dim] * len(cat_dims)   # placeholder per kolom

    print(f'[VIME] cat_dims={cat_dims}')
    print(f'[VIME] input_dim (total one-hot)={sum(cat_dims)}, hidden_dim={vime_hidden_dim}')

    # Tentukan device (gunakan CUDA jika tersedia; main.py akan overwrite)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Latih VIME self-supervised encoder menggunakan data TRAIN
    print('[VIME] Melatih VIME self-supervised encoder '
          '(mask estimation + feature reconstruction loss) ...')
    emb_model = train_supervised_embedding_model(
        cat_idx_array = train_cat_idx,
        labels        = train_labels,
        cat_dims      = cat_dims,
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
        noise_std     = 0.01,
        patience      = 40,
    )
    print('[VIME] Training selesai. Parameter di-freeze untuk diffusion.')

    # Encode: integer index → representasi VIME z
    train_cat_emb = encode_with_embedding(emb_model, train_cat_idx, device)
    test_cat_emb  = encode_with_embedding(emb_model, test_cat_idx,  device)
    # shape: [N, hidden_dim]
    print("dimensi embedding: ", train_cat_emb.shape)

    # ── train_X / test_X sekarang HANYA embedding (tidak ada kolom raw num) ─
    # Numerik TIDAK digabung ke train_X — numerik disimpan terpisah di train_num
    # untuk kompatibilitas get_eval dan normalisasi diffusion.
    # len_num di main.py = 0 karena seluruh isi train_X adalah embedding.
    train_X = train_cat_emb
    test_X  = test_cat_emb

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

    # Kumpulkan mask kolom kategorikal
    train_cat_mask = train_mask[:, cat_col_idx].astype(bool)
    test_cat_mask  = test_mask[:, cat_col_idx].astype(bool)

    # any_missing [N] — True jika sampel memiliki minimal 1 kolom kategorikal missing
    train_any_missing = train_cat_mask.any(axis=1)   # [N_train]
    test_any_missing  = test_cat_mask.any(axis=1)    # [N_test]

    # Perluas ke [N, hidden_dim]: sampel yang ada kolom missing →
    # seluruh hidden_dim di-mask True
    extend_train_mask = np.tile(
        train_any_missing[:, np.newaxis], (1, vime_hidden_dim)
    )   # [N_train, hidden_dim]
    extend_test_mask  = np.tile(
        test_any_missing[:, np.newaxis],  (1, vime_hidden_dim)
    )   # [N_test, hidden_dim]

    return (train_X, test_X,
            train_mask, test_mask,
            train_num, test_num,
            train_cat_idx, test_cat_idx,
            extend_train_mask, extend_test_mask,
            None,       # cat_bin_num (legacy, tidak dipakai)
            emb_model,  # VIMEEmbeddingModel
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

    [MODIFIKASI VIME] Seluruh dimensi train_X adalah embedding VIME.
    num_num = 0 selalu di pipeline ini — tidak ada kolom raw numerik di X_recon.

    MAE/RMSE: tidak dihitung (num_num = 0, tidak ada fitur numerik di embedding).
    Accuracy: decode embedding VIME → prediksi kelas kategorikal.

    Konvensi input:
    ---------------
    X_recon / X_true : [N, hidden_dim]
        Seluruh dimensi adalah embedding VIME.
    truth_cat_idx    : [N, n_cat_cols]  integer index
    """
    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    # mask: True(1) = missing, False(0) = observed
    num_mask = mask[:, num_col_idx].astype(bool) if len(num_col_idx) > 0 else None
    cat_mask = mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else None

    # Special case: news dataset
    if dataname == 'news' and oos:
        drop = 6265
        if num_mask is not None:
            num_mask = np.delete(num_mask, drop, axis=0)
        if cat_mask is not None:
            cat_mask = np.delete(cat_mask, drop, axis=0)
        if truth_cat_idx is not None:
            truth_cat_idx = np.delete(truth_cat_idx, drop, axis=0)
        X_recon = np.delete(X_recon, drop, axis=0)
        X_true  = np.delete(X_true,  drop, axis=0)

    # ── Numerik: tidak ada kolom raw numerik di embedding VIME ───────────
    # MAE/RMSE tidak dapat dihitung langsung dari embedding
    mae  = np.nan
    rmse = np.nan

    # ── Kategorikal: Akurasi via VIME decode ─────────────────────────────
    acc = np.nan
    if (truth_cat_idx is not None
            and len(cat_col_idx) > 0
            and emb_model is not None
            and emb_sizes is not None
            and cat_mask is not None):

        # Decode embedding → prediksi kelas per kolom
        pred_all_idx = decode_cat_from_embedding(
            emb_model, X_recon, device
        )  # [N, n_cat_cols]

        n_cat_cols    = len(cat_col_idx)
        correct_total = 0
        total_missing = 0

        for j in range(n_cat_cols):
            rows_miss = cat_mask[:, j]
            if rows_miss.sum() == 0:
                continue

            pred_j = pred_all_idx[:, j]
            true_j = truth_cat_idx[:, j].astype(int)

            correct = (pred_j[rows_miss] == true_j[rows_miss]).sum()
            correct_total += int(correct)
            total_missing += int(rows_miss.sum())

        if total_missing > 0:
            acc = correct_total / total_missing

    return mae, rmse, acc