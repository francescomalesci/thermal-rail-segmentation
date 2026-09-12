import argparse
import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import datasets
import models
import utils
from statistics import mean
import torch
from torchvision.transforms import RandomErasing

# --- Device Configuration ---
# Force single GPU usage for the training pipeline
local_rank = 0 
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

def log(text):
    print(text)

def make_data_loader(spec, tag=''):
    if spec is None:
        return None
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    log(f'{tag} dataset: size={len(dataset)}')
    
    is_train = (tag == 'train')
    loader = DataLoader(
        dataset, 
        batch_size=spec['batch_size'],
        shuffle=is_train, 
        num_workers=0, 
        pin_memory=True, 
        sampler=None
    )
    return loader

def make_data_loaders(config):
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def eval_psnr(loader, model, eval_type=None):
    model.eval()
    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        m1, m2, m3, m4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        m1, m2, m3, m4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        m1, m2, m3, m4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        m1, m2, m3, m4 = 'sm', 'em', 'wfm', 'mae'
    else:
        raise ValueError(f"Unsupported eval_type: {eval_type}")

    pbar = tqdm(total=len(loader), leave=False, desc='val')
    pred_list, gt_list = [], []

    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.cuda()
        with torch.no_grad():
            pred = torch.sigmoid(model.infer(batch['inp']))
        
        # Move to CPU immediately to prevent VRAM saturation during evaluation
        pred_list.append(pred.cpu())
        gt_list.append(batch['gt'].cpu())
        pbar.update(1)

    pbar.close()
    pred_list = torch.cat(pred_list, 0)
    gt_list = torch.cat(gt_list, 0)
    r1, r2, r3, r4 = metric_fn(pred_list, gt_list)
    return r1, r2, r3, r4, m1, m2, m3, m4

def prepare_training(config, save_path):
    model = models.make(config['model']).cuda()
    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
    
    if config.get('resume') is not None:
        # 1. RESUME: Continue an interrupted training session
        epoch_start = config.get('resume') + 1
        checkpoint_path = os.path.join(save_path, 'model_epoch_last.pth')
        print(f"Resuming: Loading adapter weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
        
    elif config.get('load') is not None:
        # 2. TRANSFER LEARNING: Start from epoch 1 using pre-trained adapter weights
        epoch_start = 1
        print(f"Transfer Learning: Loading pre-trained weights from {config['load']}")
        model.load_state_dict(torch.load(config['load'], map_location='cuda'), strict=False)
        
    else:
        # 3. CLEAN START: Initialize from base SAM2 weights
        epoch_start = 1
        print(f"Start: Loading base SAM2 weights from {config['sam_checkpoint']}")
        model.load_state_dict(torch.load(config['sam_checkpoint'], map_location='cuda'), strict=False)
        
    # Custom Learning Rate Scheduler logic (MultiStepLR based on YAML milestones)
    milestones = config.get('multi_step_lr', {}).get('milestones', [20, 35])
    gamma = config.get('multi_step_lr', {}).get('gamma', 0.1)
    lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    log(f"Model parameters: {utils.compute_num_params(model, text=True)}")
    return model, optimizer, epoch_start, lr_scheduler

def train(train_loader, model):
    model.train()
    pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    loss_list = []
    
    # Custom Data Augmentation: Random Erasing (50% probability to obscure 2-15% of the image)
    eraser = RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
    
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
            
        # Apply RandomErasing augmentation BEFORE passing tensors to the model
        batch['inp'] = eraser(batch['inp'])
        
        model.set_input(batch['inp'], batch['gt'])
        model.optimize_parameters()
        loss_list.append(model.loss_G.item())
        pbar.update(1)
        
    pbar.close()
    return mean(loss_list)

def main(config, save_path):
    if not os.path.exists(save_path): 
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f: 
        yaml.dump(config, f)

    train_loader, val_loader = make_data_loaders(config)
    model, optimizer, epoch_start, lr_scheduler = prepare_training(config, save_path)
    model.optimizer = optimizer	

    # Model Initialization Log
    print(f"Device: {torch.cuda.get_device_name(device)}")

    # Backbone Freezing Strategy
    # Freeze the image encoder to retain SAM2 generalization, train only the adapter
    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters (Adapter only): {trainable_params}')
    
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()

    for epoch in range(epoch_start, config['epoch_max'] + 1):
        t_start = timer.t()
        train_loss = train(train_loader, model)
        lr_scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(f'Epoch {epoch}/{config["epoch_max"]} | Loss: {train_loss:.4f} | LR: {current_lr:.6f}')
        
        torch.save(model.state_dict(), os.path.join(save_path, 'model_epoch_last.pth'))

        if config.get('epoch_val') and (epoch % config['epoch_val'] == 0):
            r1, r2, r3, r4, m1, m2, m3, m4 = eval_psnr(val_loader, model, config.get('eval_type'))
            print(f'Validation - {m1}: {r1:.4f}, {m2}: {r2:.4f}')
            
            if (config['eval_type'] != 'ber' and r1 > max_val_v) or (config['eval_type'] == 'ber' and r3 < max_val_v):
                max_val_v = r1 if config['eval_type'] != 'ber' else r3
                torch.save(model.state_dict(), os.path.join(save_path, 'model_epoch_best.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAM2 Adapter Custom Training Pipeline")
    parser.add_argument('--config', default="configs/thermal-rail-sam2.yaml", help="Path to YAML config file")
    parser.add_argument('--name', default="railvid_GEO_SHARP_V3", help="Name of the training run")
    parser.add_argument('--resume', type=int, default=None, help="Epoch number to resume training from")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        
    if args.resume is not None: 
        conf['resume'] = args.resume
        
    main(conf, os.path.join('./save', args.name))