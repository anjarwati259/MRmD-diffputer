import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import json
import time
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

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
#  [BARU] Fitur numerik di-diskritisasi dengan CAIM lalu di-embed bersama
#  fitur kategorikal menggunakan SupervisedLearnableEmbeddingModel yang sama.
#  Pipeline dari embedding → imputasi TIDAK BERUBAH.
# ===========================================================================


# ===========================================================================
#  CAIM Discretizer (Class-Attribute Interdependence Maximization)
# ===========================================================================

class CAIMDiscretizer:
    """
    Class-Attribute Interdependence Maximization (CAIM) Discretizer.

    Implementasi algoritma Kurgan & Cios (2004):
    - Greedy forward interval search berdasarkan nilai CAIM
    - Stopping criterion: CAIM global tidak meningkat DAN k >= num_classes
    - Fit pada data training, transform ke integer bin index

    Referensi:
        Kurgan, L. A., & Cios, K. J. (2004). CAIM discretization algorithm.
        IEEE Transactions on Knowledge and Data Engineering, 16(2), 145-153.
    """

    def __init__(self):
        self.cut_points_         = []   # list of np.ndarray, satu per kolom
        self.n_bins_             = []   # list of int, jumlah bin per kolom
        self.bin_means_          = []   # list of np.ndarray (legacy, tidak dipakai langsung)
        self.bin_means_real_ = []   # list of np.ndarray — mean nilai asli (skala norm) per bin
        self.bin_means_real_ = []   # list of np.ndarray — mean nilai asli (skala norm) per bin

    # ── CAIM helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _build_quanta(x: np.ndarray, y: np.ndarray, intervals: np.ndarray) -> np.ndarray:
        """
        Bangun matriks quanta: baris = bin, kolom = kelas.

        intervals : array cut points termasuk batas bawah & atas
                    (panjang = n_bins + 1)
        Return    : np.ndarray [n_bins, n_classes]
        """
        bins = np.digitize(x, intervals, right=True)
        bins = np.where(bins == 0, 1, bins)         # clip agar index mulai dari 1

        n_bins   = len(intervals) - 1
        classes  = np.unique(y)
        n_classes = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        quanta = np.zeros((n_bins, n_classes), dtype=np.float64)
        for b in range(1, n_bins + 1):
            mask = bins == b
            for cls in classes:
                quanta[b - 1, class_to_idx[cls]] = np.sum(mask & (y == cls))
        return quanta

    @staticmethod
    def _compute_caim(quanta: np.ndarray) -> float:
        """
        Hitung nilai CAIM dari matriks quanta.

        CAIM = (1/n) * Σ_r (max_r^2 / M_r)
        dimana:
            max_r = max class count di bin r
            M_r   = total count di bin r
            n     = jumlah bin yang tidak kosong
        """
        M_r   = quanta.sum(axis=1)                  # [n_bins]
        max_r = quanta.max(axis=1)                  # [n_bins]

        # Hanya hitung bin yang tidak kosong
        nonempty = M_r > 0
        n = nonempty.sum()
        if n == 0:
            return 0.0

        return float(np.sum((max_r[nonempty] ** 2) / M_r[nonempty]) / n)

    # ── Core algorithm per kolom ──────────────────────────────────────────

    @staticmethod
    def _run_feature(x: np.ndarray, y: np.ndarray) -> tuple:
        """
        Jalankan CAIM untuk satu kolom numerik.

        x : [N]  float — nilai fitur (sudah bebas NaN)
        y : [N]  int   — label kelas

        Return : (global_caim, disc_interval) dimana disc_interval berisi
                 seluruh boundary termasuk min & max
        """
        num_classes = len(np.unique(y))

        # Kandidat cut points: titik tengah antar nilai unik yang berurutan
        unique_vals = np.unique(x).astype(float)
        if len(unique_vals) == 1:
            # Hanya satu nilai unik → buat 1 bin dummy
            boundary = np.array([unique_vals[0], unique_vals[0] + 0.5])
            return 0.0, boundary

        # Titik tengah antara nilai berurutan (kandidat internal)
        midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        # Batas interval awal: seluruh rentang [min, max]
        disc_interval = np.array([unique_vals[0], unique_vals[-1]])
        remaining_int = midpoints.copy()             # kandidat yang belum dipakai

        global_caim = 0.0
        k = 1

        while True:
            if len(remaining_int) == 0:
                break

            best_caim_val  = -np.inf
            best_cut       = None

            for cut in remaining_int:
                candidate = np.sort(np.append(disc_interval, cut))
                q  = CAIMDiscretizer._build_quanta(x, y, candidate)
                cv = CAIMDiscretizer._compute_caim(q)
                if cv > best_caim_val:
                    best_caim_val = cv
                    best_cut      = cut

            better = best_caim_val > global_caim

            if better:
                disc_interval = np.sort(np.append(disc_interval, best_cut))
                global_caim   = best_caim_val
                remaining_int = remaining_int[remaining_int != best_cut]
                k += 1
            elif k < num_classes:
                # Belum memenuhi syarat minimum → tetap tambahkan
                disc_interval = np.sort(np.append(disc_interval, best_cut))
                global_caim   = best_caim_val
                remaining_int = remaining_int[remaining_int != best_cut]
                k += 1
            else:
                # CAIM tidak meningkat & k >= num_classes → berhenti
                break

        return global_caim, disc_interval

    # ── Public API ───────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CAIMDiscretizer':
        """
        Fit CAIM pada data training.

        X : [N, n_cols]  float — fitur numerik
        y : [N]          int   — label kelas
        """
        n_cols = X.shape[1]
        self.cut_points_ = []
        self.n_bins_     = []

        for col in range(n_cols):
            x_col = X[:, col]

            # Hapus NaN untuk fitting
            valid_mask = ~np.isnan(x_col)
            x_valid    = x_col[valid_mask]
            y_valid    = y[valid_mask]

            _, disc_interval = self._run_feature(x_valid, y_valid)

            # Cut points internal (tanpa batas bawah & atas)
            internal_cuts = disc_interval[1:-1]
            cuts_sorted   = np.sort(internal_cuts)

            self.cut_points_.append(cuts_sorted)
            self.n_bins_.append(len(cuts_sorted) + 1)

            print(f'  [CAIM] Col {col}: {len(cuts_sorted)} cut points → '
                  f'{len(cuts_sorted) + 1} bins')

        return self

    def fit_with_norm(self, X_raw: np.ndarray, y: np.ndarray,
                      X_norm: np.ndarray) -> 'CAIMDiscretizer':
        """
        Fit CAIM pada nilai raw, lalu simpan mean nilai normalisasi per bin
        sebagai representasi rekonstruksi yang benar.

        X_raw  : [N, n_cols]  float — nilai asli (raw) untuk menentukan cut points
        y      : [N]          int   — label kelas
        X_norm : [N, n_cols]  float — nilai ternormalisasi (skala (X-mean)/std)
                               dipakai untuk hitung bin_means_real_
        """
        self.fit(X_raw, y)

        n_cols = X_raw.shape[1]
        binned = self.transform(X_raw)   # [N, n_cols] bin index

        self.bin_means_real_ = []
        for col in range(n_cols):
            n_bins = self.n_bins_[col]
            means  = np.zeros(n_bins, dtype=np.float32)
            for b in range(n_bins):
                mask = binned[:, col] == b
                if mask.sum() > 0:
                    means[b] = float(X_norm[mask, col].mean())
                else:
                    # Fallback: posisi relatif bin jika bin kosong
                    means[b] = float(b) / max(n_bins - 1, 1)
            self.bin_means_real_.append(means)

        return self

    def fit_with_norm(self, X_raw: np.ndarray, y: np.ndarray,
                      X_norm: np.ndarray) -> 'CAIMDiscretizer':
        """
        Fit CAIM pada nilai raw, lalu simpan mean nilai normalisasi per bin
        sebagai representasi rekonstruksi yang benar.

        X_raw  : [N, n_cols]  float — nilai asli (raw) untuk menentukan cut points
        y      : [N]          int   — label kelas
        X_norm : [N, n_cols]  float — nilai ternormalisasi (skala (X-mean)/std)
                               dipakai untuk hitung bin_means_real_
        """
        self.fit(X_raw, y)

        n_cols = X_raw.shape[1]
        binned = self.transform(X_raw)   # [N, n_cols] bin index

        self.bin_means_real_ = []
        for col in range(n_cols):
            n_bins = self.n_bins_[col]
            means  = np.zeros(n_bins, dtype=np.float32)
            for b in range(n_bins):
                mask = binned[:, col] == b
                if mask.sum() > 0:
                    means[b] = float(X_norm[mask, col].mean())
                else:
                    # Fallback: posisi relatif bin jika bin kosong
                    means[b] = float(b) / max(n_bins - 1, 1)
            self.bin_means_real_.append(means)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform nilai kontinu → integer bin index [0, n_bins-1].

        X      : [N, n_cols]
        Return : [N, n_cols]  int64
        """
        n_cols = X.shape[1]
        out    = np.zeros_like(X, dtype=np.int64)

        for col in range(n_cols):
            cuts         = self.cut_points_[col]
            binned       = np.digitize(X[:, col], cuts, right=False).astype(np.int64)
            out[:, col]  = np.clip(binned, 0, self.n_bins_[col] - 1)

        return out

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_bin_midpoints(self, X_norm: np.ndarray,
                          X_norm_binned: np.ndarray) -> list:
        """
        Hitung nilai tengah (midpoint) setiap bin dalam skala normalisasi.

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
                    mids[b] = float(b) / max(n_bins - 1, 1)

            midpoints.append(mids)

        return midpoints

    def get_bin_means_real(self) -> list:
        """
        Kembalikan bin_means_real_ yang sudah di-compute saat fit_with_norm().

        Return : list[n_cols] of np.ndarray  — mean nilai normalisasi per bin
                 Ini adalah representasi yang benar untuk MAE/RMSE:
                 mean nilai asli per bin lebih akurat daripada midpoint geometris
                 karena mencerminkan distribusi data yang sebenarnya.

        Raises : RuntimeError jika fit_with_norm() belum pernah dipanggil.
        """
        if not self.bin_means_real_:
            raise RuntimeError(
                "bin_means_real_ kosong. Panggil fit_with_norm() terlebih dahulu, "
                "bukan fit()."
            )
        return self.bin_means_real_


    def get_bin_means_real(self) -> list:
        """
        Kembalikan bin_means_real_ yang sudah di-compute saat fit_with_norm().

        Return : list[n_cols] of np.ndarray  — mean nilai normalisasi per bin
                 Ini adalah representasi yang benar untuk MAE/RMSE:
                 mean nilai asli per bin lebih akurat daripada midpoint geometris
                 karena mencerminkan distribusi data yang sebenarnya.

        Raises : RuntimeError jika fit_with_norm() belum pernah dipanggil.
        """
        if not self.bin_means_real_:
            raise RuntimeError(
                "bin_means_real_ kosong. Panggil fit_with_norm() terlebih dahulu, "
                "bukan fit()."
            )
        return self.bin_means_real_


#  Supervised Learnable Embedding Model (TIDAK BERUBAH)
# ===========================================================================

def compute_embedding_size(n_categories: int) -> int:
    """
    Hitung ukuran embedding optimal berdasarkan jumlah kategori.
    Rumus: min(600, round(1.6 * n_categories^0.56))
    Referensi: Guo & Berkhahn (2016)
    """
    return min(600, round(1.6 * n_categories ** 0.56))


class SupervisedLearnableEmbeddingModel(nn.Module):
    """
    Model Supervised Learnable Embedding untuk fitur kategorikal tabular.

    [TIDAK BERUBAH] — sama persis dengan versi sebelumnya.
    Sekarang juga dipakai untuk fitur numerik yang sudah di-diskritisasi CAIM.

    Alur (setelah penyesuaian arsitektur):
      cat_idx [batch, n_cols]          ← termasuk numerik yg sudah jadi bin index
        → nn.Embedding per kolom → concat → [batch, total_emb_dim]
        → (opsional) Linear → SiLU → Linear   (1 hidden layer, jika use_mlp=True)
        → LayerNorm                            (stabilisasi skala sebelum diffusion)
        → z [batch, total_emb_dim]
        → MLP Classifier → [batch, n_classes]  (supervised signal, TETAP)
        → (+ noise σ=noise_std saat training)
        → Linear Decoder per kolom → logits rekonstruksi
    """

    def __init__(self, cat_dims: list, emb_sizes: list, n_classes: int,
                 dropout: float = 0.1, hidden_dim: int = 256,
                 use_mlp: bool = True, mlp_ratio: float = 1.5,
                 noise_std: float = 0.1):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=n_cat, embedding_dim=emb_dim)
            for n_cat, emb_dim in zip(cat_dims, emb_sizes)
        ])

        self.total_emb_dim = sum(emb_sizes)
        self.n_cols        = len(cat_dims)
        self.cat_dims      = cat_dims
        self.emb_sizes     = emb_sizes
        self.n_classes     = n_classes
        self.noise_std     = noise_std
        self.use_mlp       = use_mlp

        if use_mlp:
            hidden_dim_mlp = max(self.total_emb_dim, int(self.total_emb_dim * mlp_ratio))
            self.mlp = nn.Sequential(
                nn.Linear(self.total_emb_dim, hidden_dim_mlp),
                nn.SiLU(),
                nn.Linear(hidden_dim_mlp, self.total_emb_dim),
            )
        else:
            self.mlp = None

        self.layer_norm = nn.LayerNorm(self.total_emb_dim)
        self.out_dim    = self.total_emb_dim

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.total_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

        self.decoders = nn.ModuleList([
            nn.Linear(emb_size, n_cat)
            for n_cat, emb_size in zip(cat_dims, emb_sizes)
        ])

    def encode(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Encode integer index → vektor embedding dense + LayerNorm.
        x_cat  : [batch, n_cols]  — integer index tiap kolom
        return : [batch, total_emb_dim]
        """
        embedded = [
            self.embeddings[i](x_cat[:, i])
            for i in range(self.n_cols)
        ]
        z = torch.cat(embedded, dim=1)

        if self.mlp is not None:
            z = self.mlp(z)

        z = self.layer_norm(z)
        return z

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)

    def decode(self, z: torch.Tensor) -> list:
        """
        Linear Decoder: embedding → logit tiap kolom.
        z      : [batch, total_emb_dim]
        return : list[n_cols] of [batch, vocab_size_i]
        """
        per_col = torch.split(z, self.emb_sizes, dim=1)
        return [self.decoders[i](per_col[i]) for i in range(self.n_cols)]

    def forward(self, x_cat: torch.Tensor, add_noise: bool = False):
        z            = self.encode(x_cat)
        class_logits = self.classify(z)

        if add_noise and self.training and self.noise_std > 0:
            z_noisy = z + torch.randn_like(z) * self.noise_std
        else:
            z_noisy = z

        recon_logits = self.decode(z_noisy)
        return z, class_logits, recon_logits


# ===========================================================================
#  Training Supervised Embedding (TIDAK BERUBAH)
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
                                     patience: int = 30) -> SupervisedLearnableEmbeddingModel:
    """
    Latih SupervisedLearnableEmbeddingModel.
    [TIDAK BERUBAH] — sama persis dengan versi sebelumnya.
    Sekarang cat_idx_array berisi SEMUA kolom (numerik bin + kategorikal).
    """
    model = SupervisedLearnableEmbeddingModel(
        cat_dims, emb_sizes, n_classes,
        dropout=dropout,
        hidden_dim=hidden_dim,
        use_mlp=use_mlp,
        mlp_ratio=mlp_ratio,
        noise_std=noise_std,
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

    best_loss        = float('inf')
    patience_counter = 0
    best_model_state = None

    alpha = 0.7
    beta  = 1.0

    model.train()
    for epoch in range(n_epochs):
        total_loss       = 0.0
        total_class_loss = 0.0
        total_recon_loss = 0.0
        n_batches        = 0

        for batch_cat, batch_labels in loader:
            optimizer.zero_grad()

            z, class_logits, recon_logits = model(batch_cat, add_noise=True)

            class_loss = ce_loss(class_logits, batch_labels)

            recon_loss = sum(
                ce_loss(recon_logits[i], batch_cat[:, i])
                for i in range(model.n_cols)
            ) / model.n_cols

            loss = alpha * class_loss + beta * recon_loss

            loss.backward()
            optimizer.step()

            total_loss       += loss.item()
            total_class_loss += class_loss.item()
            total_recon_loss += recon_loss.item()
            n_batches        += 1

        avg_loss       = total_loss       / n_batches
        avg_class_loss = total_class_loss / n_batches
        avg_recon_loss = total_recon_loss / n_batches

        if (epoch + 1) % 10 == 0:
            print(f'[Embedding] Epoch {epoch+1}/{n_epochs} - '
                  f'Loss: {avg_loss:.4f} (Class: {avg_class_loss:.4f}, '
                  f'Recon: {avg_recon_loss:.4f})')

        if avg_loss < best_loss:
            best_loss        = avg_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'[Embedding] Early stopping triggered at epoch {epoch+1}')
            print(f'[Embedding] Best loss: {best_loss:.4f}')
            break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f'[Embedding] Loaded best model from epoch {epoch + 1 - patience_counter}')

    model.eval()

    with torch.no_grad():
        sample_cat = cat_tensor[:min(2048, len(cat_tensor))]
        z_sample   = model.encode(sample_cat)
        print(f'[Embedding] Distribusi embedding (N={z_sample.shape[0]}):')
        print(f'  mean={z_sample.mean().item():.4f}  '
              f'std={z_sample.std().item():.4f}  '
              f'norm_mean={z_sample.norm(dim=1).mean().item():.4f}')

    for param in model.parameters():
        param.requires_grad_(False)
    print('[Embedding] Seluruh parameter embedding di-freeze untuk training diffusion.')

    return model


# ===========================================================================
#  Encode / Decode helpers (TIDAK BERUBAH)
# ===========================================================================

def encode_with_embedding(model: SupervisedLearnableEmbeddingModel,
                          cat_idx_array: np.ndarray,
                          device: str,
                          batch_size: int = 4096) -> np.ndarray:
    """
    Encode integer index → embedding numpy array.
    [TIDAK BERUBAH]
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


def decode_cat_from_embedding(model: SupervisedLearnableEmbeddingModel,
                              emb_array: np.ndarray,
                              device: str,
                              batch_size: int = 4096) -> np.ndarray:
    """
    Decode embedding → prediksi kelas tiap kolom (argmax logits).
    [TIDAK BERUBAH] — dipakai untuk kolom kategorikal (dan bisa juga untuk
    numerik-bin jika diperlukan, tapi evaluasi numerik pakai bin_midpoints).

    emb_array : [N, total_emb_dim]
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
            recon_logits = model.decode(batch)
            pred_idx = torch.stack([
                torch.argmax(logits, dim=1)
                for logits in recon_logits
            ], dim=1)
            all_pred.append(pred_idx.cpu().numpy())

    return np.concatenate(all_pred, axis=0).astype(np.int64)


def decode_num_from_embedding(model: SupervisedLearnableEmbeddingModel,
                              emb_array: np.ndarray,
                              bin_midpoints: list,
                              n_num_cols: int,
                              device: str,
                              batch_size: int = 4096) -> np.ndarray:
    """
    Decode embedding → nilai numerik kontinu (dalam skala normalisasi).

    Alur (Weighted Sum / Soft-Max Decode):
      embedding → logits → softmax (probabilitas per bin) → weighted sum midpoints

    Metode ini lebih halus daripada argmax karena mempertimbangkan distribusi
    probabilitas seluruh bin, bukan hanya bin dengan logit tertinggi.
    Untuk kolom ke-i:
        p_i  = softmax(decoder_i(emb_i))      # [N, n_bins_i]
        pred = p_i @ mids_i                   # [N] — dot product = weighted sum

    Kolom numerik diasumsikan berada di AWAL emb_model (indeks 0..n_num_cols-1),
    diikuti kolom kategorikal.

    Parameter
    ---------
    model         : SupervisedLearnableEmbeddingModel
    emb_array     : [N, total_emb_dim]
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
    Load dataset dengan CAIM discretization untuk numerik +
    Supervised Embedding untuk SEMUA kolom (numerik-bin + kategorikal).

    Perubahan dari versi sebelumnya:
    - Fitur numerik di-diskritisasi dengan CAIM → integer bin index
    - Bin index numerik di-embed BERSAMA kolom kategorikal (posisi pertama)
    - Pipeline embedding → normalisasi → diffusion → imputasi TIDAK BERUBAH
    - train_num / test_num tetap dikembalikan (nilai float asli, ternormalisasi)
      untuk keperluan evaluasi MAE/RMSE di skala normalisasi

    Output tambahan (dibanding versi sebelumnya):
    - caim          : CAIMDiscretizer  (untuk transform test & decode)
    - bin_midpoints : list[n_num_cols] — midpoint bin dalam skala normalisasi
    - n_num_cols    : int
    - t_caim        : float — waktu komputasi CAIM discretization (detik)
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
    emb_model         : SupervisedLearnableEmbeddingModel
    emb_sizes         : list[int]
    caim              : CAIMDiscretizer  (atau None jika tidak ada fitur numerik)
    bin_midpoints     : list[n_num_cols] of np.ndarray  (atau None)
    n_num_cols        : int
    t_caim            : float — waktu komputasi CAIM discretization (detik)
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

        # ── CAIM Discretization ──────────────────────────────────────────
        print(f'[CAIM] Menjalankan CAIM discretization pada {n_num_cols} kolom numerik ...')
        caim = CAIMDiscretizer()

        # Fit CAIM pada train (observed) dengan label → transform train & test
        # CAIM difit pada nilai RAW (bukan normalisasi) untuk konsistensi cut point
        t_caim_start = time.time()
        # [FIX] Gunakan fit_with_norm agar bin_means_real_ tersimpan.
        # bin_means_real_ = mean nilai normalisasi per bin → lebih akurat untuk MAE/RMSE
        # dibanding midpoint geometris karena mencerminkan distribusi data asli.
        # [FIX] Gunakan fit_with_norm agar bin_means_real_ tersimpan.
        # bin_means_real_ = mean nilai normalisasi per bin → lebih akurat untuk MAE/RMSE
        # dibanding midpoint geometris karena mencerminkan distribusi data asli.
        caim.fit_with_norm_with_norm(train_num_raw, train_labels, train_num_norm, train_num_norm)

        train_num_bin = caim.transform(train_num_raw)   # [N_train, n_num_cols] int64
        test_num_bin  = caim.transform(test_num_raw)    # [N_test,  n_num_cols] int64

        # [FIX] bin_midpoints sekarang menggunakan mean nilai asli per bin (bukan
        # midpoint geometris). Nama variabel dipertahankan agar interface tidak berubah.
        bin_midpoints = caim.get_bin_means_real()
        t_caim_end = time.time()
        t_caim = t_caim_end - t_caim_start

        print(f'[CAIM] n_bins per kolom: {caim.n_bins_}')
        print(f'[CAIM] Total bins: {sum(caim.n_bins_)}')
        print(f'[CAIM] bin_means_real per kolom (sampel):')
        for ci, bm in enumerate(bin_midpoints):
            print(f'         col {ci}: {bm}')
        print(f'[CAIM] Waktu komputasi diskritisasi: {t_caim:.4f}s')

    else:
        # Tidak ada fitur numerik
        train_num     = np.zeros((len(train_df), 0), dtype=np.float32)
        test_num      = np.zeros((len(test_df),  0), dtype=np.float32)
        train_num_bin = np.zeros((len(train_df), 0), dtype=np.int64)
        test_num_bin  = np.zeros((len(test_df),  0), dtype=np.int64)
        bin_midpoints = []
        caim          = None
        t_caim        = 0.0

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
    all_dims = (caim.n_bins_ if caim is not None else []) + cat_dims_cat
    emb_sizes = [compute_embedding_size(n) for n in all_dims]

    print(f'[Embedding] all_dims (num_bin+cat)={all_dims}')
    print(f'[Embedding] emb_sizes={emb_sizes}, total_emb_dim={sum(emb_sizes)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Latih SupervisedLearnableEmbeddingModel ───────────────────────────
    # Input: semua kolom (numerik bin + kategorikal) sebagai integer index
    # [TIDAK BERUBAH] — model, training procedure, freeze semua sama
    print('[Embedding] Melatih SupervisedLearnableEmbeddingModel '
          '(classification + reconstruction loss) ...')
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
    print('[Embedding] Training selesai. Parameter di-freeze untuk diffusion.')
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

    extend_train_mask = extend_mask_emb(train_all_mask, emb_sizes_arr)
    extend_test_mask  = extend_mask_emb(test_all_mask,  emb_sizes_arr)

    # Hitung bin_midpoints dalam skala normalisasi (dibutuhkan get_eval)
    # Sudah dihitung di atas, disimpan di caim & bin_midpoints

    return (train_X, test_X,
            train_mask, test_mask,
            train_num, test_num,
            train_all_idx, test_all_idx,
            extend_train_mask, extend_test_mask,
            None,          # cat_bin_num (legacy)
            emb_model,
            emb_sizes,
            caim,          # [BARU] CAIMDiscretizer
            bin_midpoints, # [BARU] list[n_num_cols] midpoint per bin, skala norm
            n_num_cols,    # [BARU] jumlah kolom numerik
            t_caim,        # [BARU] waktu komputasi CAIM discretization (detik)
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
    # ACC hanya dihitung dari kolom kategorikal ASLI (indeks n_num_cols ke atas).
    # Kolom numerik dievaluasi dengan MAE dan RMSE saja (di atas).
    acc = np.nan
    if (truth_all_idx is not None
            and emb_model is not None
            and emb_sizes is not None):

        # Decode semua kolom → predicted index
        pred_all_idx = decode_cat_from_embedding(
            emb_model, X_recon, device
        )  # [N, n_num_cols + n_cat_cols]

        # ── (Opsional) Akurasi bin numerik — dilaporkan terpisah ──────────
        if n_num_cols > 0 and num_mask is not None:
            correct_num   = 0
            missing_num   = 0
            for j in range(n_num_cols):
                rows_miss = num_mask[:, j]
                if rows_miss.sum() == 0:
                    continue
                pred_j = pred_all_idx[:, j]
                true_j = truth_all_idx[:, j].astype(int)
                correct_num += int((pred_j[rows_miss] == true_j[rows_miss]).sum())
                missing_num += int(rows_miss.sum())
            if missing_num > 0:
                acc_num = correct_num / missing_num

        # ── Akurasi kategorikal MURNI (kolom cat saja) ────────────────────
        if len(cat_col_idx) > 0 and cat_mask is not None:
            correct_total = 0
            total_missing = 0

            for j in range(len(cat_col_idx)):
                rows_miss = cat_mask[:, j]
                if rows_miss.sum() == 0:
                    continue

                # [FIX] offset eksplisit: kolom numerik ada di 0..n_num_cols-1,
                # kolom kategorikal mulai dari n_num_cols.
                col_offset = n_num_cols + j

                pred_j = pred_all_idx[:, col_offset]
                true_j = truth_all_idx[:, col_offset].astype(int)

                correct = (pred_j[rows_miss] == true_j[rows_miss]).sum()
                correct_total += int(correct)
                total_missing += int(rows_miss.sum())

            if total_missing > 0:
                acc = correct_total / total_missing

    return mae, rmse, acc