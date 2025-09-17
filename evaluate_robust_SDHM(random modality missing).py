import os
import argparse
import torch
from core.dataset import MMDataLoader
from models.SDHM import build_model
from core.metric import MetricsTop
from opts import parse_opts
import numpy as np


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

# ======================== 评估函数 ========================

def evaluate_once(model: torch.nn.Module, loader: torch.utils.data.DataLoader, metrics_fn, missing_rate: float, device: torch.device,
                 collect_vis_pairs: bool = False):
    model.eval()
    y_pred, y_true = [], []

    print(f"开始评估，缺失率: {missing_rate})

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

            output = model(img_m, audio_m, text_m, audio_text, vision_text, current_epoch=160)

            pred = output[0] if isinstance(output, tuple) else output
            y_pred.append(pred.cpu())
            y_true.append(label.cpu())
                    
    

    pred, true = torch.cat(y_pred, dim=0), torch.cat(y_true, dim=0)
    metrics = metrics_fn(pred, true)
    print(f"评估完成，样本数: {len(pred)}")
    return metrics


# ======================== 主程序 ========================

def main():
    print("开始运行评估脚本...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--key_eval', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--missing_rates', type=str, default='0.0,0.1,0.2,0.3,0.5,0.8')
    parser.add_argument('--seeds', type=str, default='')
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
    all_results = {}  # 存储所有缺失率的结果
    for cur_r in missing_rates:
        results = evaluate_once(model, test_loader, metrics_fn, cur_r, device, collect_vis_pairs=args_cli.save_feature_comparison)
        
        # 打印每个缺失率下的指标
        print(f"\n=== 缺失率 {cur_r} 的评估结果 ===")
        for metric_name, metric_value in results.items():
            print(f"{metric_name}: {metric_value:.4f}")
        print("=" * 40)
        all_results[cur_r] = results

    if all_results:
        # 获取所有指标名称
        metric_names = list(next(iter(all_results.values())).keys())
        
        # 打印表头
        header = f"{'缺失率':<8}"
        for metric_name in metric_names:
            header += f"{metric_name:<12}"
        print(header)
        print("-" * len(header))
        
        # 打印每个缺失率的结果
        for cur_r in missing_rates:
            if cur_r in all_results:
                row = f"{cur_r:<8.1f}"
                for metric_name in metric_names:
                    value = all_results[cur_r][metric_name]
                    row += f"{value:<12.4f}"
                print(row)
    
    print("=" * 60)
        
   
    print("\n评估完成！")


if __name__ == '__main__':
    main()
