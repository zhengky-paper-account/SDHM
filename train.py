import os
import sys

# 解析参数（在导入torch之前）
sys.path.append('.')
from opts import parse_opts
opt = parse_opts()

# 设置CUDA_VISIBLE_DEVICES（必须在导入torch之前）
os.environ["CUDA_VISIBLE_DEVICES"] = opt.CUDA_VISIBLE_DEVICES

# 现在导入torch
import torch
import numpy as np
from tqdm import tqdm
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, save_model, setup_seed
from tensorboardX import SummaryWriter
from models.SDHM import build_model
from core.metric import MetricsTop
import pickle
import time
print('PyTorch version:', torch.__version__)
print('CUDA built version:', torch.version.cuda if hasattr(torch.version, 'cuda') else 'CPU Only')
print('CUDA available:', torch.cuda.is_available())

# 检查CUDA状态
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(f"✅ CUDA检测结果:")
print(f"  - 是否使用CUDA: {USE_CUDA}")
print(f"  - 设备: {device}")
print(f"  - CUDA设备数量: {torch.cuda.device_count()}")
if USE_CUDA:
    print(f"  - 当前CUDA设备: {torch.cuda.current_device()}")
    print(f"  - 设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"  - CUDA_VISIBLE_DEVICES: {opt.CUDA_VISIBLE_DEVICES}")
print(f"  - SCE扩散: {opt.SCE_diffusion}")
train_mae, val_mae = [], []

def load_model(model, checkpoint_path):
    """
    加载指定路径的模型权重
    
    Args:
        model: 待加载的模型
        checkpoint_path: 模型权重文件路径
    
    Returns:
        start_epoch: 开始的epoch
        best_acc: 最佳准确率
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"❌ 模型权重文件不存在: {checkpoint_path}")
        return 0, 0.0
    
    print(f"🔄 正在加载模型权重: {checkpoint_path}")
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['state_dict'])
        print("✅ 模型权重加载成功")
        
        # 获取epoch和最佳准确率
        start_epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('best_acc', 0.0)
        
        print(f"📊 模型加载信息:")
        print(f"  - 起始epoch: {start_epoch}")
        print(f"  - 最佳准确率: {best_acc:.4f}")
        
        return start_epoch, best_acc
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        print("将从头开始训练")
        return 0, 0.0

def save_pkl(path, obj):
    pickle_file = open(path, 'wb')
    pickle.dump(obj, pickle_file)
    pickle_file.close()
    print("保存成功")

def load_pkl(path):
    pickle_file = open(path, 'rb')
    obj = pickle.load(pickle_file)
    pickle_file.close()
    print("读取成功")
    return obj



def main():
    if opt.seed is not None:
        setup_seed(opt.seed)
    print("seed: {}".format(opt.seed))
    
    log_path = os.path.join(".", "log", opt.project_name)
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)
    print("log_path :", log_path)

    save_path = os.path.join(opt.models_save_root,  opt.project_name)
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    print("model_save_path :", save_path)

    model = build_model(opt).to(device)

    dataLoader = MMDataLoader(opt)

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)

    scheduler_warmup = get_scheduler(optimizer, opt)
    loss_fn = torch.nn.MSELoss()
    metrics = MetricsTop().getMetics(opt.datasetName)

    writer = SummaryWriter(logdir=log_path)
    start_epoch, best_acc = load_model(model, opt.load_checkpoint)

    for epoch in range(start_epoch, opt.n_epochs+1):
        show_list = []
        train(model, dataLoader['train'], optimizer, loss_fn, epoch, writer, metrics)
        evaluate(model, dataLoader['valid'], optimizer, loss_fn, epoch, writer, save_path, metrics)
        if opt.is_test is not None:
            pre_dict = test(model, dataLoader['test'], optimizer, loss_fn, epoch, writer, metrics)
            show_list.append(pre_dict)
        scheduler_warmup.step()
    writer.close()

def train(model, train_loader, optimizer, loss_fn, epoch, writer, metrics):
    
    train_pbar = tqdm(enumerate(train_loader))
    losses = AverageMeter()

    y_pred, y_true = [], []

    model.train()
    for cur_iter, data in train_pbar:
        img, audio, text, audio_text, vision_text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device), data['audio_text'].to(device), data['vision_text'].to(device)
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]

        model_output = model(img, audio, text, audio_text, vision_text,current_epoch=epoch)
        
        # 处理模型输出
        if isinstance(model_output, tuple):
            output, diffusion_loss = model_output
        else:
            output = model_output
            diffusion_loss = 0.0
        
        # 主任务loss
        main_loss = loss_fn(output, label)
        
       # 总loss = 主任务loss + 扩散loss
        #方案一
        #diffusion_loss_weight = getattr(opt, 'diffusion_loss_weight', 0.1)
        #loss = main_loss + diffusion_loss_weight * diffusion_loss
        #方案二
        # 获取模型中的可训练参数 log(sigma)
        log_sigma_main = model.log_sigma_main
        log_sigma_diff = model.log_sigma_diff
        # 计算 sigma
        sigma_main = torch.exp(log_sigma_main)
        sigma_diff = torch.exp(log_sigma_diff)
        # 不确定性加权损失（带 log sigma 正则项）
        loss = (main_loss / (2 * sigma_main**2)) + log_sigma_main + (diffusion_loss / (2 * sigma_diff**2)) + log_sigma_diff
        #方案三
        # 计算动态 lambda
        #_lambda = getattr(opt, 'diffusion_loss_weight', 0.1) / (diffusion_loss.item() + 1e-8)
        # 总损失
        #loss = main_loss + _lambda * diffusion_loss





        losses.update(loss.item(), batchsize)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

        train_pbar.set_description('train')
        train_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                'loss': '{:.5f}'.format(losses.value_avg),
                                'main_loss': '{:.5f}'.format(main_loss.item()),
                                'diff_loss': '{:.5f}'.format(diffusion_loss.item() if diffusion_loss > 0 else 0.0),
                                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    train_results = metrics(pred, true)
    print('train: ', train_results)
    train_mae.append(train_results['MAE'])

    writer.add_scalar('train/loss', losses.value_avg, epoch)
    writer.add_scalar('train/main_loss', main_loss.item(), epoch)
    if diffusion_loss > 0:
        writer.add_scalar('train/diffusion_loss', diffusion_loss.item(), epoch)

def evaluate(model, eval_loader, optimizer, loss_fn, epoch, writer, save_path, metrics):
    test_pbar = tqdm(enumerate(eval_loader))

    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:
            img, audio, text, audio_text, vision_text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device), data['audio_text'].to(device), data['vision_text'].to(device)
            
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            model_output = model(img, audio, text, audio_text, vision_text,current_epoch=epoch)
            
            # 处理模型输出
            if isinstance(model_output, tuple):
                output, diffusion_loss = model_output
            else:
                output = model_output
                diffusion_loss = 0.0
            
            # 主任务loss
            main_loss = loss_fn(output, label)
            
            # 总loss = 主任务loss + 扩散loss
            diffusion_loss_weight = getattr(opt, 'diffusion_loss_weight', 0.1)
            loss = main_loss + diffusion_loss_weight * diffusion_loss

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)

            test_pbar.set_description('eval')
            test_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                   'loss': '{:.5f}'.format(losses.value_avg),
                                   'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        #print(test_results)

        writer.add_scalar('evaluate/loss', losses.value_avg, epoch)


        
        if epoch == 40:
         save_model(save_path, epoch, model, optimizer) 
        
       
        if epoch == 50:
         save_model(save_path, epoch, model, optimizer)
        if epoch == 160:
         save_model(save_path, epoch, model, optimizer)
        

def test(model, test_loader, optimizer, loss_fn, epoch, writer, metrics):
    start_time = time.time()

    test_pbar = tqdm(enumerate(test_loader))

    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:
            img, audio, text, audio_text, vision_text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device), data['audio_text'].to(device), data['vision_text'].to(device)
            raw_text, id, labels = data['raw_text'], data['id'], data['labels']['M'].to(device).view(-1, 1)
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            output = model(img, audio, text, audio_text, vision_text,current_epoch=epoch)
            # 处理模型输出
            if isinstance(output, tuple):
                output, diffusion_loss = output
            else:
                output = output
                diffusion_loss = 0.0
            #print(id)
            #print(output.view(1, -1))
            #print(label.view(1, -1))
            #print("----")

         # 主任务loss
            main_loss = loss_fn(output, label)
            
            # 总loss = 主任务loss + 扩散loss
            diffusion_loss_weight = getattr(opt, 'diffusion_loss_weight', 0.1)
            loss = main_loss + diffusion_loss_weight * diffusion_loss

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)

            test_pbar.set_description('test')
            test_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                   'loss': '{:.5f}'.format(losses.value_avg),
                                   'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        print(test_results)

        end_time = time.time()
        elapsed_time = end_time - start_time
        #print(f"Elapsed time: {elapsed_time} seconds")  

        writer.add_scalar('test/loss', losses.value_avg, epoch)

if __name__ == '__main__':
    main()