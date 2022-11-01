import cv2
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func
import torch
import torch.nn
from tqdm import tqdm

from bregman.models import aux_funs, print_sparsity
from bregman.optimizers import AdaBreg

from titan.utils import (make_experiment_name, datadir, plotsdir,
                         checkpointsdir, configsdir, get_lipschitz,
                         read_config, parse_input_args, normalize)
from titan import TITAN, SIREN

# Paths to raw Mars waveforms and the scattering covariance thereof.
DATA_PATH = datadir('super-resolution')

# GET butterfly image.
if not os.path.exists(os.path.join(DATA_PATH, 'butterfly.png')):
    os.system(
        "wget https://www.dropbox.com/s/qqm30q7v37ttj21/butterfly.png -O" +
        os.path.join(DATA_PATH, 'butterfly.png'))

# Training default hyperparameters.
MARS_CONFIG_FILE = 'super-resolution.json'

# Downsampling factor.
SCALE = 4

if __name__ == '__main__':

    # Command line arguments.
    args = read_config(os.path.join(configsdir(), MARS_CONFIG_FILE))
    args = parse_input_args(args)

    # Experiment name.
    experiment = make_experiment_name(args)

    # Setting random seeds.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Setting default device (cpu/cuda) depending on CUDA availability and
    # input arguments.
    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')
        # Read in all the data into CPU memory to avoid slowing down GPU.
    else:
        device = torch.device('cpu')
        # Read the data from disk batch by batch.

    # Read image.
    im = normalize(
        plt.imread(os.path.join(DATA_PATH,
                                'butterfly.png')).astype(np.float32), True)

    H, W, _ = im.shape
    im_lr = cv2.resize(im,
                       None,
                       fx=1 / SCALE,
                       fy=1 / SCALE,
                       interpolation=cv2.INTER_AREA)
    H2, W2, _ = im_lr.shape

    x = torch.linspace(-1, 1, W2).to(device)
    y = torch.linspace(-1, 1, H2).to(device)

    X, Y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]

    x_hr = torch.linspace(-1, 1, W).to(device)
    y_hr = torch.linspace(-1, 1, H).to(device)
    X_hr, Y_hr = torch.meshgrid(x_hr, y_hr, indexing='xy')
    coords_hr = torch.hstack((X_hr.reshape(-1, 1), Y_hr.reshape(-1, 1)))[None,
                                                                         ...]

    gt = torch.tensor(im).reshape(H * W, 3)[None, ...].to(device)
    gt_lr = torch.tensor(im_lr).reshape(H2 * W2, 3)[None, ...].to(device)

    mse_array = torch.zeros(args.niters, device=device)

    best_mse = float('inf')
    best_img = None

    if args.model_type == 'titan':
        model = TITAN(2,
                      100,
                      3,
                      depth=10,
                      resnet_activation=lambda x: torch.sin(2 * x),
                      scale=1.0)
    elif args.model_type == 'siren':
        model = SIREN(in_features=2,
                      out_features=3,
                      hidden_features=256,
                      hidden_layers=2,
                      outermost_linear=True,
                      pos_encode=False,
                      nonlinearity='sine',
                      sidelength=max(H, W),
                      first_omega_0=10.0,
                      hidden_omega_0=10.0,
                      scale=50.0)

    model.to(device)
    if args.optim_type == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optim_type == 'AdaBreg':
        aux_funs.sparsify_(model, args.spars_fac)
        optim = AdaBreg(lr=args.learning_rate, params=model.parameters())

    loss_log = []
    with tqdm(range(args.niters),
              unit='iter',
              colour='#B5F2A9',
              dynamic_ncols=True) as pb:
        for epoch in pb:

            subset = np.random.choice(gt_lr.shape[1],
                                      size=gt_lr.shape[1] // 1,
                                      replace=False)

            rec = model(coords[:, subset, :]).real

            loss = ((gt_lr[:, subset, :] - rec)**2).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_log.append(loss.detach().cpu().item())

            if epoch % 100 == 0:
                with torch.no_grad():
                    rec_hr = model(coords_hr).real
                    mse_array[epoch] = ((gt - rec_hr)**2).mean().item()

                imrec = rec_hr[0, ...].reshape(H, W, 3).detach().cpu().numpy()

                if mse_array[epoch] < best_mse:
                    best_mse = mse_array[epoch]
                    best_img = imrec

            pb.set_postfix({
                'loss': loss.item(),
                'best_mse': best_mse,
            })

    fig = plt.figure()
    plt.imshow(best_img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(plotsdir(experiment), 'recovery.png'),
                format="png",
                bbox_inches="tight",
                dpi=600,
                pad_inches=.0)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(loss_log)
    plt.savefig(os.path.join(plotsdir(experiment), 'loss.png'),
                format="png",
                bbox_inches="tight",
                dpi=600,
                pad_inches=.0)
    plt.close(fig)

    if args.optim_type == 'AdaBreg':
        sparsity_stats = print_sparsity(model)

    L = get_lipschitz(model.cpu(), w=256, h=256)
    lip_cnst = L.max().numpy()

    with open(os.path.join(checkpointsdir(experiment), 'stats.txt'), 'w') as f:
        f.write('SNR: ' + str(-10 * np.log10(best_mse.cpu().numpy())) + '\n')
        f.write('SSIM: ' + str(ssim_func(im, imrec, channel_axis=-1)) + '\n')
        f.write('Lipschitz: ' + str(lip_cnst) + '\n')
        f.write('Seed: ' + str(args.seed) + '\n')
        f.write('spars_fac: ' + str(args.spars_fac) + '\n')
        if args.optim_type == 'AdaBreg':
            f.write('sparsity factor: ' + str(sparsity_stats[2]) + '\n')

    file = h5py.File(os.path.join(checkpointsdir(experiment), 'recovery.h5'),
                     'w')
    file['img'] = best_img
    if args.optim_type == 'AdaBreg':
        file['sparsity'] = sparsity_stats[2]
    file['SNR'] = -10 * np.log10(best_mse.cpu().numpy())
    file['SSIM'] = ssim_func(im, imrec, channel_axis=-1)
    file['lip'] = lip_cnst
    file['L'] = L.detach().cpu().numpy()
    file.close()

    torch.save(
        {
            'model_state_dict': model.cpu().state_dict(),
            'optim_state_dict': optim.state_dict(),
        }, os.path.join(checkpointsdir(experiment), 'checkpoint.pth'))
