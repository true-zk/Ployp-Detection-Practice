from dataloader import PloypDataset, ToTensor
import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from pyutils import HyperParam

from model import ResUnetPP
from logger import MyWriter
from metrics import BCEDiceLoss, MetricTracker, jaccard_index, dice_coeff

def train(hp):
    ckpt_dir = os.path.join(hp.ckpt_dir, hp.model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    log_dir = os.path.join(hp.log_dir, hp.model_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = MyWriter(log_dir)
    
    if hp.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(hp.device)
    
    model = ResUnetPP(3).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.step_size, gamma=hp.gamma)
    best_loss = 1e10
    start_epoch = 0
    
    # TODO: resume training
    if hp.resume and os.path.isfile(hp.resume_path):
        None
    
    train_set = PloypDataset(hp.train_data, transform=transforms.Compose([ToTensor()]), train=True, bbox=False)
    val_set = PloypDataset(hp.val_data, transform=transforms.Compose([ToTensor()]), train=False, bbox=False)
    
    train_laoder = DataLoader(train_set, batch_size=hp.batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    
    epochs = hp.epochs
    step = 0
    for epoch in range(start_epoch, epochs):
        print("Epoch {}/{}".format(epoch, epochs - 1))
        print("-" * 10)
        """
        /home/zhongken/miniconda3/envs/yolo8/lib/python3.8/site-packages/torch/optim/lr_scheduler.py:131: 
        UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. 
        In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  
        Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. 
        See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        """
        train_acc = MetricTracker()
        train_loss = MetricTracker()
        
        bar = tqdm(train_laoder, desc="Training")
        for idx, data in enumerate(bar):
            x, gd = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, gd)
            loss.backward()
            optimizer.step()
            
            # step the learning rate scheduler
            lr_scheduler.step()   
            
            train_acc.update(dice_coeff(pred, gd), pred.size(0)) 
            train_loss.update(loss.data.item(), pred.size(0))
            
            if step % hp.log_interval == 0:
                writer.log_training(train_loss.avg, train_acc.avg, step)
                bar.set_description(
                    "Train Loss: {:.4f}, Train Acc: {:.4f}".format(train_loss.avg, train_acc.avg)
                )
            
            # val
            if step % hp.val_interval == 0:
                valid_metrics = validation(
                    device, val_loader, model, criterion, writer, step
                )
                save_path = os.path.join(
                    # ckpt_dir, "%s_checkpoint_%04d.pt" % (hp.model_name, step)
                    ckpt_dir, "{}_checkpoint_{:04d}.pt".format(hp.model_name, step)
                )
                # store best loss and save a model checkpoint
                best_loss = min(valid_metrics["valid_loss"], best_loss)
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "arch": "ResUnetpp",
                        "state_dict": model.state_dict(),
                        "best_loss": best_loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )
                print("Saved checkpoint to: %s" % save_path)

            step += 1


def validation(device, valid_loader, model, criterion, logger, step):

    # logging accuracy and loss
    valid_acc = MetricTracker()
    valid_loss = MetricTracker()

    # switch to evaluate mode
    model.eval()

    # Iterate over data.
    for idx, data in enumerate(tqdm(valid_loader, desc="validation")):

        # get the inputs and wrap in Variable
        x, gd = data[0].to(device), data[1].to(device)

        outputs = model(x)
        loss = criterion(outputs, gd)
        
        valid_acc.update(dice_coeff(outputs, gd), outputs.size(0))
        valid_loss.update(loss.data.item(), outputs.size(0))
        if idx == 0:
            logger.log_images(x.cpu(), gd.cpu(), outputs.cpu(), step)
    logger.log_validation(valid_loss.avg, valid_acc.avg, step)

    print("Validation Loss: {:.4f} Acc: {:.4f}".format(valid_loss.avg, valid_acc.avg))
    model.train()
    return {"valid_loss": valid_loss.avg, "valid_acc": valid_acc.avg}


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("-c", "--config", type=str, required=True, help="yaml file for configuration")
    args = parser.parse_args()
    hp = HyperParam(args.config)
    train(hp)