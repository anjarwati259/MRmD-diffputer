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
from dataset_class import (load_dataset, get_eval, mean_std,
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

args = parser.parse_args()

# Force GPU usage
if not torch.cuda.is_available():
    raise RuntimeError("GPU tidak tersedia! Script ini membutuhkan GPU untuk berjalan.")

args.device = f'cuda:{args.gpu}'
torch.cuda.set_device(args.gpu)
# torch.set_default_device dipanggil di dalam __main__ SETELAH load_dataset
# selesai, agar DataLoader di train_embedding_model (dataset.py) tidak kena
# RuntimeError "Expected cuda generator but found cpu".


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
    #  PENTING: load_dataset dipanggil SEBELUM torch.set_default_device agar
    #  DataLoader di dalam train_embedding_model (dataset.py) tidak terkena
    #  efek "RuntimeError: Expected a 'cuda' device type for generator but
    #  found 'cpu'". DataLoader PyTorch membutuhkan generator CPU untuk shuffle.
    #
    #  train_X / test_X : [N, num_num + total_emb_dim]
    #    fitur kategorik sudah di-encode ke embedding (bukan binary encoding)
    # =========================================================================
    (train_X, test_X,
     ori_train_mask, ori_test_mask,
     train_num, test_num,
     train_cat_idx, test_cat_idx,
     extend_train_mask, extend_test_mask,   # shape [N, num_num+total_emb_dim] ← untuk normalisasi & diffusion
     cat_bin_num,
     emb_model,
     emb_sizes
     ) = load_dataset(dataname, split_idx, mask_type, ratio)

    # Setelah load_dataset selesai (embedding sudah di-train),
    # baru aktifkan default device ke CUDA untuk training loop utama.
    torch.set_default_device(device)

    # Pindahkan emb_model ke GPU
    if emb_model is not None:
        emb_model = emb_model.to(device)
        emb_model.eval()

    # =========================================================================
    #  Normalisasi
    #  Sama persis dengan kode asli:
    #    - mean & std dihitung hanya pada observed entries (bukan missing)
    #    - normalisasi: (X - mean) / std / 2
    #  Catatan: embedding sudah berbasis float, sehingga normalisasi ini valid.
    # =========================================================================
    mean_X, std_X = mean_std(train_X, extend_train_mask)   # mask shape cocok dengan train_X
    std_X[std_X == 0] = 1.0    # jaga kolom konstan agar tidak NaN saat /std
    in_dim        = train_X.shape[1]

    # Konversi ke GPU tensor
    X      = torch.tensor((train_X - mean_X) / std_X / 2,
                          device=device, dtype=torch.float32)
    X_test = torch.tensor((test_X  - mean_X) / std_X / 2,
                          device=device, dtype=torch.float32)

    mask_train = torch.tensor(extend_train_mask, device=device, dtype=torch.float32)
    mask_test  = torch.tensor(extend_test_mask,  device=device, dtype=torch.float32)

    # Simpan mean/std di GPU untuk denormalisasi
    mean_X_gpu = torch.tensor(mean_X, device=device, dtype=torch.float32)
    std_X_gpu  = torch.tensor(std_X,  device=device, dtype=torch.float32)

    # Panjang fitur numerik (dipakai untuk memisahkan num / cat di output)
    len_num = train_num.shape[1]

    MAEs,  RMSEs,  ACCs  = [], [], []
    MAEs_out, RMSEs_out, ACCs_out = [], [], []

    start_time = time.time()

    for iteration in range(args.max_iter):

        # =====================================================================
        #  M-Step: Density Estimation (DiffPutter)
        #  Tidak ada perubahan pada bagian ini — algoritma diffusion tetap sama.
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
            # Load hasil imputasi iterasi sebelumnya (skala (X-mean)/std)
            rec_prev   = torch.tensor(
                np.load(f'{ckpt_dir}/iter_{iteration}.npy') / 2,
                device=device, dtype=torch.float32
            )
            # KRUSIAL: observed entries HARUS selalu dari X asli (bukan dari
            # rekonstruksi), agar tidak terjadi drift/akumulasi error tiap iterasi.
            # missing=1 → pakai imputasi iterasi lalu | observed=0 → pakai X asli
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

        num_epochs  = 10000 + 1
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
        print(f'Training time: {end_time - start_time:.2f}s')

        # =====================================================================
        #  E-Step: In-sample Imputation
        #  Bagian diffusion sama persis dengan kode asli.
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

            # Blend: posisi observed → nilai X asli, posisi missing → imputasi
            # Gunakan X langsung (bukan impute_X yang berisi 0 di posisi missing)
            # agar observed entries selalu terikat ke data asli
            mask_int = mask_train.float()
            rec_X    = rec_X * mask_int + X * (1. - mask_int)

            # Clamp ke range wajar agar tidak terjadi drift/range explosion
            # antar iterasi. Batas ±10 lebih dari cukup untuk skala /std/2.
            rec_X = torch.clamp(rec_X, -10.0, 10.0)

            rec_Xs.append(rec_X)

        rec_X = torch.stack(rec_Xs, dim=0).mean(0)

        # Simpan hasil untuk iterasi berikutnya (dalam skala normalized /2)
        np.save(f'{ckpt_dir}/iter_{iteration+1}.npy',
                (rec_X * 2).cpu().numpy())

        # =====================================================================
        #  Denormalisasi In-sample — mengikuti POLA ASLI DiffPutter
        #
        #  Pola asli (binary encoding):
        #    pred_X = rec_X * 2              → undo /2, skala (X-mean)/std
        #    pred_X[:, len_num:] = pred_X[:, len_num:] * std + mean
        #                                    → kembalikan kategorikal ke bit {0,1}
        #    MAE/RMSE dihitung pada numerik yang MASIH di skala (X-mean)/std
        #
        #  Pola embedding (analog):
        #    pred_X = rec_X * 2              → undo /2, skala (X-mean)/std
        #    pred_X[:, len_num:] = pred_X[:, len_num:] * std_emb + mean_emb
        #                                    → kembalikan embedding ke skala asli
        #                                       (skala yang dipakai saat emb_model di-train)
        #    MAE/RMSE dihitung pada numerik yang MASIH di skala (X-mean)/std
        #    → konsisten dengan metrik asli DiffPutter
        #
        #  Mengapa range tidak perlu seragam antara num dan emb setelah ini?
        #  Karena get_eval memisahkan: numerik untuk MAE/RMSE, embedding
        #  untuk decode kelas (akurasi). Keduanya tidak dibandingkan langsung.
        # =====================================================================

        rec_X_np  = rec_X.cpu().numpy()    # [N, num_num + total_emb_dim], skala /2
        X_true_np = X.cpu().numpy()        # sama
        mean_np   = mean_X_gpu.cpu().numpy()
        std_np    = std_X_gpu.cpu().numpy()

        # Step 1: undo /2 → skala (X - mean) / std
        pred_X = rec_X_np * 2
        X_true = X_true_np * 2

        # Step 2: hanya bagian EMBEDDING yang di-denorm ke skala embedding asli
        # (numerik dibiarkan di skala ternormalisasi, sama dengan pola asli)
        if len_num < pred_X.shape[1]:
            pred_X[:, len_num:] = (pred_X[:, len_num:]
                                   * std_np[len_num:] + mean_np[len_num:])
            X_true[:, len_num:] = (X_true[:, len_num:]
                                   * std_np[len_num:] + mean_np[len_num:])

        mae, rmse, acc = get_eval(
            dataname     = dataname,
            X_recon      = pred_X,
            X_true       = X_true,
            truth_cat_idx= train_cat_idx,
            num_num      = len_num,
            emb_model    = emb_model,
            emb_sizes    = emb_sizes,
            mask         = ori_train_mask,
            device       = device,
            oos          = False,
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

            # Blend: observed → X_test asli, missing → imputasi
            mask_int = mask_test.float()
            rec_X    = rec_X * mask_int + X_test * (1. - mask_int)

            # Clamp agar range tidak meledak
            rec_X = torch.clamp(rec_X, -10.0, 10.0)

            rec_Xs.append(rec_X)

        rec_X = torch.stack(rec_Xs, dim=0).mean(0)

        # =====================================================================
        #  Denormalisasi Out-of-sample — pola sama dengan in-sample
        # =====================================================================
        rec_X_np  = rec_X.cpu().numpy()
        X_true_np = X_test.cpu().numpy()

        pred_X = rec_X_np * 2
        X_true = X_true_np * 2

        if len_num < pred_X.shape[1]:
            pred_X[:, len_num:] = (pred_X[:, len_num:]
                                   * std_np[len_num:] + mean_np[len_num:])
            X_true[:, len_num:] = (X_true[:, len_num:]
                                   * std_np[len_num:] + mean_np[len_num:])

        mae_out, rmse_out, acc_out = get_eval(
            dataname     = dataname,
            X_recon      = pred_X,
            X_true       = X_true,
            truth_cat_idx= test_cat_idx,
            num_num      = len_num,
            emb_model    = emb_model,
            emb_sizes    = emb_sizes,
            mask         = ori_test_mask,
            device       = device,
            oos          = True,
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

        with open(f'{result_save_path}/result_class.txt', 'a+') as f:
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
                f'Training time={end_time - start_time:.2f}s, '
                f'In-sample imputation={impute_end_time - impute_start_time:.2f}s, '
                f'Out-of-sample imputation={oos_end - oos_start:.2f}s\n\n'
            )

        print(f'Results saved to {result_save_path}')

        # Reset timer untuk iterasi berikutnya
        start_time = time.time()