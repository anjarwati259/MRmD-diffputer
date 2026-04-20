import os
import torch

import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
from tqdm import tqdm

from model import MLPDiffusion, Model
from dataset_mrmd import (load_dataset, get_eval, mean_std,
                     decode_cat_from_embedding)
from diffusion_utils import sample_step, impute_mask

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname',   type=str, default='california', help='Name of dataset.')
parser.add_argument('--gpu',        type=int, default=0,            help='GPU index.')
parser.add_argument('--split_idx',  type=int, default=0,            help='Split idx.')
parser.add_argument('--max_iter',   type=int, default=10,            help='Maximum iteration.')
parser.add_argument('--ratio',      type=str, default=30,           help='Masking ratio.')
parser.add_argument('--hid_dim',    type=int, default=1024,         help='Hidden dimension.')
parser.add_argument('--mask',       type=str, default='MCAR',       help='Masking mechanism.')
parser.add_argument('--num_trials', type=int, default=20,            help='Number of sampling times.')
parser.add_argument('--num_steps',  type=int, default=50,           help='Number of diffusion steps.')
parser.add_argument('--noise_std',  type=float, default=0.01,       help='Noise std for embedding model.')
parser.add_argument('--epochs',     type=int, default=10000,        help='Number of training epochs per iteration.')
parser.add_argument('--resume_iter',type=int, default=0,            help='Resume from this iteration index.')
parser.add_argument('--stop_iter',  type=int, default=None,         help='Stop after this iteration (exclusive). Jika None, jalan sampai max_iter.')

args = parser.parse_args()

# Force GPU usage
if not torch.cuda.is_available():
    raise RuntimeError("GPU tidak tersedia! Script ini membutuhkan GPU untuk berjalan.")

args.device = f'cuda:{args.gpu}'
torch.cuda.set_device(args.gpu)


if __name__ == '__main__':

    dataname   = args.dataname
    split_idx  = args.split_idx
    device     = args.device
    hid_dim    = args.hid_dim
    mask_type  = args.mask
    ratio      = args.ratio
    num_trials = args.num_trials
    num_steps  = args.num_steps

    if mask_type == 'MNAR':
        mask_type = 'MNAR_logistic_T2'

    # =========================================================================
    #  Load Dataset
    #  PENTING: load_dataset dipanggil SEBELUM torch.set_default_device.
    #
    #  [MODIFIKASI] load_dataset sekarang mengembalikan 3 nilai tambahan:
    #    - mrmd          : MRmDDiscretizer (untuk decode numerik)
    #    - bin_midpoints : list[n_num_cols] — midpoint bin skala normalisasi
    #    - n_num_cols    : int — jumlah kolom numerik
    #
    #  train_X / test_X : [N, total_emb_dim]
    #    SEMUA fitur (numerik bin + kategorikal) sudah di-encode ke embedding.
    #    len_num = 0 karena tidak ada kolom raw numerik terpisah.
    # =========================================================================
    (train_X, test_X,
     ori_train_mask, ori_test_mask,
     train_num, test_num,
     train_all_idx, test_all_idx,   # [N, n_num_cols + n_cat_cols] — bin + label idx
     extend_train_mask, extend_test_mask,
     cat_bin_num,
     emb_model,
     emb_sizes,
     mrmd,           # [BARU] MRmDDiscretizer (atau None jika tidak ada numerik)
     bin_midpoints,  # [BARU] list[n_num_cols] midpoint per bin, skala normalisasi
     n_num_cols,     # [BARU] jumlah kolom numerik
     t_mrmd,         # [BARU] waktu komputasi MRmD discretization (detik)
     t_emb           # [BARU] waktu komputasi embedding training (detik)
     ) = load_dataset(dataname, split_idx, mask_type, ratio, args.noise_std)

    t_total_preprocessing = t_mrmd + t_emb
    print(f'\n{"="*60}')
    print(f'[TIMING] Ringkasan Waktu Komputasi Preprocessing:')
    print(f'  - MRmD Discretization : {t_mrmd:.4f}s')
    print(f'  - Embedding Training  : {t_emb:.4f}s')
    print(f'  - Total (Diskrit→Emb) : {t_total_preprocessing:.4f}s')
    print(f'{"="*60}')

    # Setelah load_dataset selesai, aktifkan default device ke CUDA.
    torch.set_default_device(device)

    # Pindahkan emb_model ke GPU
    if emb_model is not None:
        emb_model = emb_model.to(device)
        emb_model.eval()

    # =========================================================================
    #  Normalisasi
    #  [TIDAK BERUBAH] — mean & std dihitung pada observed entries,
    #  normalisasi: (X - mean) / std / 2
    #
    #  Catatan: train_X sekarang hanya berisi embedding (tidak ada raw numerik).
    #  extend_train_mask shape sudah cocok dengan train_X (total_emb_dim).
    # =========================================================================
    mean_X, std_X = mean_std(train_X, extend_train_mask)
    std_X[std_X == 0] = 1.0
    in_dim = train_X.shape[1]

    X      = torch.tensor((train_X - mean_X) / std_X / 2,
                          device=device, dtype=torch.float32)
    X_test = torch.tensor((test_X  - mean_X) / std_X / 2,
                          device=device, dtype=torch.float32)

    mask_train = torch.tensor(extend_train_mask, device=device, dtype=torch.float32)
    mask_test  = torch.tensor(extend_test_mask,  device=device, dtype=torch.float32)

    mean_X_gpu = torch.tensor(mean_X, device=device, dtype=torch.float32)
    std_X_gpu  = torch.tensor(std_X,  device=device, dtype=torch.float32)

    # [MODIFIKASI] len_num = 0 karena tidak ada kolom raw numerik di train_X.
    # Seluruh isi train_X adalah embedding (numerik bin + kategorikal).
    len_num = 0

    MAEs,  RMSEs,  ACCs  = [], [], []
    MAEs_out, RMSEs_out, ACCs_out = [], [], []

    # Tentukan batas iterasi untuk run ini
    iter_end = args.stop_iter if args.stop_iter is not None else args.max_iter
    iter_end = min(iter_end, args.max_iter)
    print(f'[INFO] Menjalankan iterasi {args.resume_iter} s/d {iter_end - 1} '
          f'(max_iter={args.max_iter})')

    start_time = time.time()

    for iteration in range(args.resume_iter, iter_end):

        # =====================================================================
        #  M-Step: Density Estimation (DiffPutter)
        #  [TIDAK BERUBAH] — algoritma diffusion sama persis.
        # =====================================================================

        ckpt_dir = (f'ckpt/{dataname}/rate{ratio}/{mask_type}/'
                    f'{split_idx}/{num_trials}_{num_steps}')
        os.makedirs(f'{ckpt_dir}/{iteration}', exist_ok=True)

        print(f'\n{"="*60}')
        print(f'Iteration: {iteration}')
        print(f'Checkpoint dir: {ckpt_dir}')

        if iteration == 0:
            X_miss     = (1. - mask_train) * X
            train_data = X_miss
        else:
            print(f'Loading X_miss from {ckpt_dir}/iter_{iteration}.npy')
            rec_prev   = torch.tensor(
                np.load(f'{ckpt_dir}/iter_{iteration}.npy') / 2,
                device=device, dtype=torch.float32
            )
            X_miss     = rec_prev * mask_train + X * (1. - mask_train)
            train_data = X_miss

        print(f'[INFO] X_miss shape: {train_data.shape}, '
              f'range: [{train_data.min():.4f}, {train_data.max():.4f}]')

        batch_size = 4096

        generator = torch.Generator(device=device)

        class GPUTensorDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        train_loader = DataLoader(
            GPUTensorDataset(train_data),
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = 0,
            pin_memory  = False,
            generator   = generator,
        )

        num_epochs  = args.epochs + 1
        denoise_fn  = MLPDiffusion(in_dim, hid_dim).to(device)

        if iteration == 0:
            print(denoise_fn)

        model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9,
                                      patience=50, verbose=False)

        model.train()
        best_loss = float('inf')
        patience  = 0

        pbar = tqdm(range(num_epochs), desc='Training')
        for epoch in pbar:
            batch_loss = 0.0
            len_input  = 0

            for batch in train_loader:
                inputs     = batch.float()
                loss       = model(inputs)
                loss       = loss.mean()
                batch_loss += loss.item() * len(inputs)
                len_input  += len(inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            curr_loss = batch_loss / len_input
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
                best_loss = curr_loss
                patience  = 0
                torch.save(model.state_dict(),
                           f'{ckpt_dir}/{iteration}/model.pt')
            else:
                patience += 1
                if patience == 500:
                    print('Early stopping')
                    break

            pbar.set_postfix(loss=curr_loss)

            if epoch % 1000 == 0:
                torch.save(model.state_dict(),
                           f'{ckpt_dir}/{iteration}/model_{epoch}.pt')

        end_time = time.time()
        print(f'Training time: {end_time - start_time:.4f}s')

        # =====================================================================
        #  E-Step: In-sample Imputation
        #  [TIDAK BERUBAH] — diffusion sampling sama persis.
        # =====================================================================

        impute_start_time = time.time()
        rec_Xs = []

        for trial in tqdm(range(num_trials), desc='In-sample imputation'):
            X_miss   = (1. - mask_train) * X
            impute_X = X_miss

            denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)
            model      = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
            model.load_state_dict(
                torch.load(f'{ckpt_dir}/{iteration}/model.pt')
            )

            net = model.denoise_fn_D
            num_samples, dim = X.shape[0], X.shape[1]

            rec_X    = impute_mask(net, impute_X, mask_train,
                                   num_samples, dim, num_steps, device)

            mask_int = mask_train.float()
            rec_X    = rec_X * mask_int + X * (1. - mask_int)

            rec_X = torch.clamp(rec_X, -10.0, 10.0)
            rec_Xs.append(rec_X)

        rec_X = torch.stack(rec_Xs, dim=0).mean(0)

        np.save(f'{ckpt_dir}/iter_{iteration+1}.npy',
                (rec_X * 2).cpu().numpy())

        # =====================================================================
        #  Denormalisasi In-sample
        #
        #  [MODIFIKASI] Semua dimensi adalah embedding (tidak ada raw numerik).
        #  Seluruh dimensi di-denorm ke skala embedding asli untuk decoding.
        #
        #  Alur:
        #    rec_X (skala /2) → * 2 → skala (X-mean)/std
        #    → * std + mean   → skala embedding asli
        #    → decode_num_from_embedding → MAE/RMSE di skala normalisasi
        #    → decode_cat_from_embedding → Accuracy kategorikal
        #
        #  MAE/RMSE dihitung di skala (X-mean)/std VIA bin midpoints,
        #  konsisten dengan konvensi asli DiffPutter (tidak di-denorm ke asli).
        # =====================================================================

        rec_X_np  = rec_X.cpu().numpy()
        X_true_np = X.cpu().numpy()
        mean_np   = mean_X_gpu.cpu().numpy()
        std_np    = std_X_gpu.cpu().numpy()

        # Step 1: undo /2 → skala (X - mean) / std
        pred_X = rec_X_np * 2
        X_true = X_true_np * 2

        # Step 2: denorm SELURUH dimensi embedding ke skala asli
        # (semua dimensi adalah embedding, tidak ada split num/emb)
        pred_X = pred_X * std_np + mean_np
        X_true = X_true * std_np + mean_np

        mae, rmse, acc = get_eval(
            dataname      = dataname,
            X_recon       = pred_X,
            X_true        = X_true,
            truth_all_idx = train_all_idx,
            num_num       = len_num,          # selalu 0 di pipeline baru
            emb_model     = emb_model,
            emb_sizes     = emb_sizes,
            mask          = ori_train_mask,
            device        = device,
            oos           = False,
            bin_midpoints = bin_midpoints,    # untuk decode prediksi numerik
            n_num_cols    = n_num_cols,
            num_true_norm = train_num,        # ground truth nilai asli ternormalisasi
        )
        MAEs.append(mae)
        RMSEs.append(rmse)
        ACCs.append(acc)

        impute_end_time = time.time()
        print(f'In-sample imputation time: '
              f'{impute_end_time - impute_start_time:.2f}s')
        print(f'In-sample  → MAE={mae:.6f}  RMSE={rmse:.6f}  ACC={acc}')

        # =====================================================================
        #  E-Step: Out-of-sample Imputation
        #  [TIDAK BERUBAH] — diffusion sampling sama persis.
        # =====================================================================

        oos_start = time.time()
        rec_Xs    = []

        for trial in tqdm(range(num_trials), desc='Out-of-sample imputation'):
            X_miss   = (1. - mask_test) * X_test
            impute_X = X_miss

            denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)
            model      = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
            model.load_state_dict(
                torch.load(f'{ckpt_dir}/{iteration}/model.pt')
            )

            net = model.denoise_fn_D
            num_samples, dim = X_test.shape[0], X_test.shape[1]

            rec_X    = impute_mask(net, impute_X, mask_test,
                                   num_samples, dim, num_steps, device)

            mask_int = mask_test.float()
            rec_X    = rec_X * mask_int + X_test * (1. - mask_int)

            rec_X = torch.clamp(rec_X, -10.0, 10.0)
            rec_Xs.append(rec_X)

        rec_X = torch.stack(rec_Xs, dim=0).mean(0)

        # ── Denormalisasi Out-of-sample (sama dengan in-sample) ──────────
        rec_X_np  = rec_X.cpu().numpy()
        X_true_np = X_test.cpu().numpy()

        pred_X = rec_X_np * 2
        X_true = X_true_np * 2

        pred_X = pred_X * std_np + mean_np
        X_true = X_true * std_np + mean_np

        mae_out, rmse_out, acc_out = get_eval(
            dataname      = dataname,
            X_recon       = pred_X,
            X_true        = X_true,
            truth_all_idx = test_all_idx,
            num_num       = len_num,
            emb_model     = emb_model,
            emb_sizes     = emb_sizes,
            mask          = ori_test_mask,
            device        = device,
            oos           = True,
            bin_midpoints = bin_midpoints,
            n_num_cols    = n_num_cols,
            num_true_norm = test_num,         # ground truth nilai asli ternormalisasi
        )
        MAEs_out.append(mae_out)
        RMSEs_out.append(rmse_out)
        ACCs_out.append(acc_out)

        oos_end = time.time()
        print(f'Out-of-sample imputation time: {oos_end - oos_start:.2f}s')
        print(f'Out-of-sample → MAE={mae_out:.6f}  RMSE={rmse_out:.6f}  ACC={acc_out}')

        # =====================================================================
        #  Simpan hasil
        # =====================================================================
        result_save_path = (f'results/{dataname}/rate{ratio}/{mask_type}/'
                            f'{split_idx}/{num_trials}_{num_steps}')
        os.makedirs(result_save_path, exist_ok=True)

        t_train          = end_time - start_time
        t_impute_in      = impute_end_time - impute_start_time
        t_impute_out     = oos_end - oos_start
        t_total_pipeline = t_mrmd + t_emb + t_train + t_impute_in + t_impute_out

        print(f'\n[TIMING] Iteration {iteration} — Ringkasan Waktu:')
        print(f'  - MRmD Discretization         : {t_mrmd:.4f}s')
        print(f'  - Embedding Training           : {t_emb:.4f}s')
        print(f'  - Diffusion Training           : {t_train:.4f}s')
        print(f'  - In-sample Imputation         : {t_impute_in:.4f}s')
        print(f'  - Out-of-sample Imputation     : {t_impute_out:.4f}s')
        print(f'  - TOTAL (Diskrit→Imputasi)     : {t_total_pipeline:.4f}s')

        with open(f'{result_save_path}/result_mrmdwith.txt', 'a+', encoding='utf-8') as f:
            f.write(
                f'iteration {iteration}, '
                f'MAE: in-sample={mae:.6f}, out-of-sample={mae_out:.6f}\n'
            )
            f.write(
                f'iteration {iteration}, '
                f'RMSE: in-sample={rmse:.6f}, out-of-sample={rmse_out:.6f}\n'
            )
            f.write(
                f'iteration {iteration}, '
                f'ACC: in-sample={acc}, out-of-sample={acc_out}\n'
            )
            f.write(
                f'iteration {iteration}, '
                f'Training time={t_train:.4f}s, '
                f'In-sample imputation={t_impute_in:.4f}s, '
                f'Out-of-sample imputation={t_impute_out:.4f}s\n'
            )
            f.write(
                f'iteration {iteration}, '
                f'MRmD discretization={t_mrmd:.4f}s, '
                f'Embedding training={t_emb:.4f}s, '
                f'Total preprocessing (Diskrit+Emb)={t_mrmd + t_emb:.4f}s\n'
            )
            f.write(
                f'iteration {iteration}, '
                f'TOTAL pipeline (Diskrit-Imputasi)={t_total_pipeline:.4f}s\n\n'
            )

        print(f'Results saved to {result_save_path}')

        # Reset timer untuk iterasi berikutnya
        start_time = time.time()

    # =========================================================================
    #  Selesai — reminder resume berikutnya
    # =========================================================================
    next_iter = iter_end
    if next_iter < args.max_iter:
        print(f'\n{"="*60}')
        print(f'[STOP] Run selesai sampai iterasi {iter_end - 1}.')
        print(f'[NEXT] Lakukan Save Version (Run All), lalu lanjut dengan:')
        print(f'       --resume_iter {next_iter} --stop_iter {min(next_iter + (iter_end - args.resume_iter), args.max_iter)}')
        print(f'{"="*60}')
    else:
        print(f'\n{"="*60}')
        print(f'[DONE] Semua {args.max_iter} iterasi selesai.')
        print(f'{"="*60}')