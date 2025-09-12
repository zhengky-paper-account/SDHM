import os
import argparse
import torch
from core.dataset import MMDataLoader
from models.SDHM import build_model
from core.metric import MetricsTop
from opts import parse_opts
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from typing import List, Tuple, Dict
from sklearn.manifold import TSNE  

# 字体
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
my_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [my_font.get_name()]


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_missing_mask(batch_tensor: torch.Tensor, missing_rate: float) -> torch.Tensor:
    if missing_rate <= 0.0:
        return batch_tensor
    if missing_rate >= 1.0:
        return torch.zeros_like(batch_tensor)
    batch_size = batch_tensor.size(0)
    mask = (torch.rand(batch_size, 1, 1, device=batch_tensor.device) < missing_rate).float()
    return batch_tensor * (1.0 - mask)


# ======================== t-SNE 可视化函数 ========================

def plot_tsne_for_missing_rates(r2pairs: Dict[float, Tuple[np.ndarray, np.ndarray]], save_dir: str = "./denoising_visualization"):
    """
    绘制2x3 t-SNE网格，r∈{0.0,0.3,0.6}
    """
    os.makedirs(save_dir, exist_ok=True)
    target_order = [0.0,0.1,0.2,0.3, 0.5,0.8]

    all_base, all_diff = [], []
    for r in target_order:
        if r in r2pairs:
            base, diff = r2pairs[r]
            all_base.append(base)
            all_diff.append(diff)

    if len(all_base) == 0:
        print("没有找到有效的特征数据")
        return

    X_base = np.vstack(all_base)
    X_diff = np.vstack(all_diff)
    X_all = np.vstack([X_base, X_diff])

    # 标准化
    mu, sigma = X_all.mean(axis=0, keepdims=True), X_all.std(axis=0, keepdims=True) + 1e-6
    X_norm = (X_all - mu) / sigma

    # t-SNE
    reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42, n_iter=2000)
    Z_all = reducer.fit_transform(X_norm)

    # 分回 base/diff
    base_size = X_base.shape[0]
    Zb_all, Zd_all = Z_all[:base_size], Z_all[base_size:]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax_list = axes.ravel()

    offset = 0
    for idx, r in enumerate(target_order):
        ax = ax_list[idx]
        if r not in r2pairs:
            ax.axis('off')
            continue

        base, diff = r2pairs[r]
        n_base, n_diff = base.shape[0], diff.shape[0]
        Zb = Zb_all[offset:offset+n_base]
        Zd = Zd_all[offset:offset+n_diff]
        offset += n_base

        ax.scatter(Zb[:, 0], Zb[:, 1], s=6, c='blue', alpha=0.7, label='Base Features')
        ax.scatter(Zd[:, 0], Zd[:, 1], s=6, c='red', alpha=0.7, label='Diffused Features')

        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc='best', fontsize=8, frameon=True)

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'tsne_missing_rates_grid.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"保存t-SNE缺失率网格图: {out_path}")


def plot_feature_comparison_grid_tsne(r2pairs: Dict[float, Tuple[np.ndarray, np.ndarray]], save_dir: str = "./feature_comparison"):
    """
    用t-SNE绘制不同缺失率下的特征对比，r∈{0.0,0.3,0.8}
    """
    os.makedirs(save_dir, exist_ok=True)
    target_order = [0.0,0.1,0.2, 0.3,0.5, 0.8]

    all_base, all_diff = [], []
    for r in target_order:
        if r in r2pairs:
            base, diff = r2pairs[r]
            all_base.append(base)
            all_diff.append(diff)

    if len(all_base) == 0:
        print("没有找到有效的特征数据")
        return

    X_base = np.vstack(all_base)
    X_diff = np.vstack(all_diff)
    X_all = np.vstack([X_base, X_diff])

    mu, sigma = X_all.mean(axis=0, keepdims=True), X_all.std(axis=0, keepdims=True) + 1e-6
    X_norm = (X_all - mu) / sigma

    reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42, n_iter=2000)
    Z_all = reducer.fit_transform(X_norm)

    base_size = X_base.shape[0]
    Zb_all, Zd_all = Z_all[:base_size], Z_all[base_size:]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ax_list = axes.ravel()
    titles = ["Missing rate r=0.0", "Missing rate r=0.1",  "Missing rate r=0.2","Missing rate r=0.3",  "Missing rate r=0.5","Missing rate r=0.8"]

    offset = 0
    for idx, r in enumerate(target_order):
        ax = ax_list[idx]
        if r not in r2pairs:
            ax.axis('off')
            continue

        base, diff = r2pairs[r]
        n_base, n_diff = base.shape[0], diff.shape[0]
        Zb = Zb_all[offset:offset+n_base]
        Zd = Zd_all[offset:offset+n_diff]
        offset += n_base

        ax.scatter(Zb[:,0], Zb[:,1], s=8, c='blue', alpha=0.7, label='Base Features')
        ax.scatter(Zd[:,0], Zd[:,1], s=8, c='red', alpha=0.7, label='Diffused Features')
        ax.set_title(titles[idx], fontsize=36)
        ax.set_xticks([]); ax.set_yticks([])
        #ax.legend(loc='best', fontsize=18, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'feature_comparison_missing_rates_tsne.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"保存t-SNE特征对比图: {out_path}")


# ======================== 评估函数 ========================

def evaluate_once(model: torch.nn.Module, loader: torch.utils.data.DataLoader, metrics_fn, missing_rate: float, device: torch.device,
                 collect_vis_pairs: bool = False):
    model.eval()
    y_pred, y_true = [], []

    base_vis_np, diff_vis_np = None, None
    all_base_features, all_diffused_features = [], []

    print(f"开始评估，缺失率: {missing_rate}, 收集可视化: {collect_vis_pairs}")

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            img = data['vision'].to(device)
            audio = data['audio'].to(device)
            text = data['text'].to(device)
            audio_text = data['audio_text'].to(device)
            vision_text = data['vision_text'].to(device)
            label = data['labels']['M'].to(device).view(-1, 1)

            img_m = apply_missing_mask(img, missing_rate)
            audio_m = apply_missing_mask(audio, missing_rate)
            text_m = apply_missing_mask(text, missing_rate)

            output = model(img_m, audio_m, text_m, audio_text, vision_text, current_epoch=160, return_vis_pairs=collect_vis_pairs)

            pred = output[0] if isinstance(output, tuple) else output
            y_pred.append(pred.cpu())
            y_true.append(label.cpu())

            if collect_vis_pairs and hasattr(model, 'saved_base_vis') and hasattr(model, 'saved_final_vis'):
                base_vis, final_vis = model.saved_base_vis, model.saved_final_vis
                if base_vis is not None and final_vis is not None:
                    base_pooled = base_vis.mean(dim=1).cpu().numpy()
                    diffused_pooled = final_vis.mean(dim=1).cpu().numpy()
                    all_base_features.append(base_pooled)
                    all_diffused_features.append(diffused_pooled)

    if all_base_features and all_diffused_features:
        base_vis_np = np.vstack(all_base_features)
        diff_vis_np = np.vstack(all_diffused_features)
        print(f"收集到特征数据: base={base_vis_np.shape}, diffused={diff_vis_np.shape}")
    else:
        print(f"未收集到特征数据")

    pred, true = torch.cat(y_pred, dim=0), torch.cat(y_true, dim=0)
    metrics = metrics_fn(pred, true)
    print(f"评估完成，样本数: {len(pred)}")
    return metrics, base_vis_np, diff_vis_np


# ======================== 主程序 ========================

def main():
    print("开始运行评估脚本...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--key_eval', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--missing_rates', type=str, default='0.0,0.1,0.2,0.3,0.5,0.8')
    parser.add_argument('--seeds', type=str, default='')
    parser.add_argument('--save_feature_comparison', action='store_true')
    args_cli, unknown = parser.parse_known_args()

    opt = parse_opts()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = build_model(opt).to(device)
    ckpt_path = args_cli.checkpoint or getattr(opt, 'test_checkpoint', '')
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['state_dict'])
        print("检查点加载成功")
    else:
        print("未找到检查点，使用随机初始化")

    dataLoader = MMDataLoader(opt)
    test_loader = dataLoader['test']
    dataset_name = opt.datasetName
    metrics_fn = MetricsTop().getMetics(dataset_name)

    missing_rates = [float(x) for x in args_cli.missing_rates.split(',') if x.strip() != '']
    seed_list = [int(s) for s in args_cli.seeds.split(',')] if args_cli.seeds else []

    r2pairs: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

    for cur_r in missing_rates:
        results, base_pair, diff_pair = evaluate_once(model, test_loader, metrics_fn, cur_r, device, collect_vis_pairs=args_cli.save_feature_comparison)
        if args_cli.save_feature_comparison and base_pair is not None and diff_pair is not None and cur_r not in r2pairs:
            r2pairs[cur_r] = (base_pair, diff_pair)

    if len(r2pairs) > 0 and args_cli.save_feature_comparison:
        plot_tsne_for_missing_rates(r2pairs)
        plot_feature_comparison_grid_tsne(r2pairs)
    else:
        print("跳过可视化")

    print("\n评估完成！")


if __name__ == '__main__':
    main()
