import os, sys
import numpy as np
import matplotlib as mpl
from matplotlib import colors, patheffects
from matplotlib.patches import FancyBboxPatch

mpl.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.transform import resize as imresize
import torch
import matplotlib.cm as cm
# import datautils as datautils
from PIL import Image


# acgu_path = '/home/zhuhaoran/MyNet/acgu.npz'
# chars = np.load(acgu_path,allow_pickle=True)['data']

def inference(args, model, device, test_loader):
    model.eval()
    p_all = []
    with torch.no_grad():
        for batch_idx, (x0, y0) in enumerate(test_loader):
            x, y = x0.float().to(device), y0.to(device).float()
            output = model(x)
            prob = torch.sigmoid(output)

            p_np = prob.to(device='cpu').numpy()
            p_all.append(p_np)

    p_all = np.concatenate(p_all)
    return p_all


def normalize_pwm(pwm, factor=None, MAX=None):
    if MAX is None:
        MAX = np.max(np.abs(pwm))
    pwm = pwm / MAX
    if factor:
        pwm = np.exp(pwm * factor)
    norm = np.outer(np.ones(pwm.shape[0]), np.sum(np.abs(pwm), axis=0))
    return pwm / norm


def get_nt_height(pwm, height, norm):
    def entropy(p):
        s = 0
        for i in range(len(p)):
            if p[i] > 0:
                s -= p[i] * np.log2(p[i])
        return s

    num_nt, num_seq = pwm.shape
    heights = np.zeros((num_nt, num_seq))
    for i in range(num_seq):
        if norm == 1:
            total_height = height
        else:
            total_height = (np.log2(num_nt) - entropy(pwm[:, i])) * height

        heights[:, i] = np.floor(pwm[:, i] * np.minimum(total_height, height * 2))

    return heights.astype(int)


# def seq_logo(pwm, height=30, nt_width=10, norm=0, alphabet='rna', colormap='standard'):
def seq_logo(pwm, height=30, nt_width=10, norm=0):
    acgu_path = '../acgu.npz'
    chars = np.load(acgu_path, allow_pickle=True)['data']
    heights = get_nt_height(pwm, height, norm)
    num_nt, num_seq = pwm.shape
    width = np.ceil(nt_width * num_seq).astype(int)

    max_height = height * 2
    logo = np.ones((max_height, width, 3)).astype(int) * 255
    for i in range(num_seq):
        nt_height = np.sort(heights[:, i])
        index = np.argsort(heights[:, i])
        remaining_height = np.sum(heights[:, i])
        offset = max_height - remaining_height

        for j in range(num_nt):
            if nt_height[j] <= 0:
                continue
            nt_img = imresize(chars[index[j]], output_shape=(nt_height[j], nt_width)) * 255
            height_range = range(remaining_height - nt_height[j], remaining_height)
            width_range = range(i * nt_width, i * nt_width + nt_width)
            if height_range:
                for k in range(3):
                    for m in range(len(width_range)):
                        logo[height_range + offset, width_range[m], k] = nt_img[:, m, k]

                remaining_height -= nt_height[j]

    return logo.astype(np.uint8)


def plot_saliency(X, W, nt_width=100, norm_factor=3, str_null=None, outdir="pic/"):
    plot_index = np.where(np.sum(X[:4, :], axis=0) != 0)[0]
    num_nt = len(plot_index)
    trace_width = num_nt * nt_width
    trace_height = 400

    seq_str_mode = False
    if X.shape[0] > 4:
        seq_str_mode = True
        assert str_null is not None, "Null region is not provided."

    img_seq_raw = seq_logo(X[:4, plot_index], height=nt_width, nt_width=nt_width)

    seq_sal = normalize_pwm(W[:4, plot_index], factor=norm_factor)
    img_seq_sal_logo = seq_logo(seq_sal, height=nt_width * 5, nt_width=nt_width)
    img_seq_sal = imresize(W[:4, plot_index], output_shape=(trace_height, trace_width))

    fig = plt.figure(figsize=(10, 2))

    # gs = gridspec.GridSpec(nrows=4, ncols=1, height_ratios=[2.5, 1, 0.5,1])

    gs = gridspec.GridSpec(
        nrows=2,
        ncols=1,
        height_ratios=[2, 1.5],
        hspace=-0.25
    )

    # cmap_reversed = mpl.cm.get_cmap('gist_earth')
    # cmap_reversed = mpl.cm.get_cmap('gist_ncar')
    # cmap_reversed = mpl.cm.get_cmap('PuRd')
    # cmap_reversed = mpl.cm.get_cmap('YlOrBr')
    cmap_reversed = mpl.cm.get_cmap('Greens')
    # cmap_reversed = mpl.cm.get_cmap('GnBu')
    # cmap_reversed = mpl.cm.get_cmap('Blues')

    # colors_list = ['#191970', '#00BFFF', '#FFD700', '#FF8C00', '#D54846', '#800000']
    # colors_list = ['#191970', '#87CEFA', '#FFB6C1', '#FF69B4', '#FFD700', '#C71585']

    # cmap = colors.LinearSegmentedColormap.from_list('How2matplotlib_custom', colors_list, N=10000)

    ax1 = fig.add_subplot(gs[0])
    ax1.axis('off')
    ax1.imshow(img_seq_sal_logo)
    # plt.text(x=trace_width - 400, y=10, s='CPRISP ', fontsize=6)
    # plt.text(x=0, y=20, s='CPRISP ', fontsize=6)
    plt.text(x=0, y=20, s='CRSP ', fontsize=6)

    gs_sub = gridspec.GridSpecFromSubplotSpec(
        nrows=2,
        ncols=1,
        subplot_spec=gs[1],
        height_ratios=[1, 2],
        hspace=-0.7
    )
    ax2 = fig.add_subplot(gs_sub[0])
    ax2.axis('off')
    # ax2.imshow(img_seq_sal, cmap=cmap_reversed)

    # ax2.pcolormesh(W, cmap=cmap_reversed, edgecolors='white', linewidth=0.1)
    # ax2.pcolormesh(W, cmap=cmap_reversed, edgecolors='#B1D0F3', linewidth=0.1)
    # ax2.pcolormesh(W, cmap=cmap_reversed, edgecolors='#173F91', linewidth=0.1)
    ax2.pcolormesh(W, cmap=cmap_reversed, edgecolor='none')

    rect = plt.Rectangle(
        (0, 0),
        W.shape[1],
        W.shape[0],
        fill=False,
        edgecolor='#315F1F',
        linewidth=1,
        linestyle='-.'
    )

    ax2.add_patch(rect)
    ax2.set_aspect(0.5)

    ax3 = fig.add_subplot(gs_sub[1])
    ax3.axis('off')
    ax3.imshow(img_seq_raw)

    # plt.subplots_adjust(wspace=0, hspace=-0.2)

    # plt.tight_layout(pad=0, w_pad=0, h_pad=-1)

    # ax.axhline(y=0, color='black', linewidth=2, linestyle='--')

    filepath = outdir
    fig.savefig(filepath, format='png', dpi=300, bbox_inches='tight')
    plt.close('all')


