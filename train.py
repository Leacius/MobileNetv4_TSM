import torch
import torch.nn as nn

from pathlib import Path
from ops.tools import setup_logger, AverageMeter, accuracy
from ops.datasets import BaseballDataset
from ops.model import MobileNetV4TSMInflated
from torch.utils.data import DataLoader, random_split
from ops.transform import train_transforms, val_transforms

class Trainer:
    def __init__(self, args):
        self.args = args
        self.is_best = False
        self.best_prec1 = 0
        self.less_loss = 100000
        self.store_name = "Mobilev4TSM"
        self.logger = setup_logger(output="./log", name="Mobilev4TSM")
        self.logger.info('storing name: ' + "Train_Mobilev4TSM")

        # 檢查資料夾
        self.check_rootfolders()

        # 優化器和損失函數及模型
        self.model = MobileNetV4TSMInflated(num_segments=self.args.num_segments, new_length=self.args.new_length, num_classes=self.args.num_classes).cuda()
        # load pre-trained
        self.load_model(self.args.weights)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr = self.args.lr, 
                                    momentum = self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, args.lr_steps, gamma=args.lr_decay_rate)
        self.criterion = nn.CrossEntropyLoss()

    def confusion_matrix(self, output, target, topk=(1, 2), cf_matrix=None):
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, False)
        pred = pred.t()[0] # for top1
            
        for i, j in zip(pred, target):
            cf_matrix[int(i.item()), j.item()] += 1

        return cf_matrix
    
    def load_model(self, weight):
        if len(weight) == 0:
            return
        self.logger.info("=> fine-tuning from '{}'".format(weight))
        ckpt = torch.load(weight, weights_only=True)
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model_dict = self.model.state_dict()

        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                self.logger.info('=> Load after remove .net: {}'.format(k))
                replace_dict.append((k, k.replace('.net', '')))
        for k in model_dict:
            if k not in sd and k.replace('.net', '') in sd:
                self.logger.info('=> Load after adding .net: {}'.format(k))
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)

        # Drop incompatible shapes
        filtered_sd = {}
        for k, v in sd.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered_sd[k] = v
            else:
                self.logger.info(f"=> Skip loading parameter: {k}, shape mismatch {v.shape} vs {model_dict.get(k, None)}")

        self.model.load_state_dict(filtered_sd, strict=False)

    def check_rootfolders(self):
        """Create log and model folder"""
        log = Path("log") / self.store_name
        checkpoint = Path("checkpoint") / self.store_name
        
        log.mkdir(parents=True, exist_ok=True)
        checkpoint.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, state, epoch):

        filename = '%s/%s/%d_epoch_ckpt.pth.tar' % ("checkpoint", self.store_name, epoch)
        torch.save(state, filename)
        if self.is_best:
            best_filename = '%s/%s/best.pth.tar' % ("checkpoint", self.store_name)
            torch.save(state, best_filename)
        
        total = Path(f"./checkpoint/{self.store_name}").glob("*_ckpt.pth.tar")
        total_list = list(Path(f"./checkpoint/{self.store_name}").glob("*_ckpt.pth.tar"))
        if len(total_list) > 5:
            total = sorted(total, key=lambda x: x.stat().st_mtime)
            total[0].unlink()  

    def train(self, epoch):
        with torch.no_grad():
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
        
        self.model.train()
        
        self.logger.info(f"Start")
        for i, (inputs, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            inp = inputs.view((-1, inputs.shape[1] // self.args.num_segments) + (inputs.shape[2::]))
            inp = inp.cuda()
            target = target.cuda()

            output = self.model(inp)
            loss = self.criterion(output, target)

            prec1, prec5 = accuracy(output.data, target, (1, 2))

            with torch.no_grad():
                losses.update(loss.item(), inp.size(0))
                top1.update(prec1.item(), inp.size(0))
                top5.update(prec5.item(), inp.size(0))

            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), 10)

            self.optimizer.step()

            if i % 20 == 0:
                self.logger.info(('Epoch: [{0}][{1}/{2}],'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                epoch, i, len(self.train_loader), loss=losses,
                                top1=top1, top5=top5, )))
        return losses.avg, top1.avg, top5.avg
    
    def validate(self,):
        with torch.no_grad():
            self.model.eval()
            cf_matrix = torch.zeros((self.args.num_classes, self.args.num_classes), dtype=torch.int64)
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            for i, (inputs, target) in enumerate(self.test_loader):
                inp = inputs.view((-1, inputs.shape[1] // self.args.num_segments) + (inputs.shape[2::]))
                inp = inp.cuda()
                target = target.cuda()
                
                output = self.model(inp)
                loss = self.criterion(output, target)
                cf_matrix = self.confusion_matrix(output, target, cf_matrix=cf_matrix)

                prec1, prec5 = accuracy(output.data, target, (1, 2))

                losses.update(loss.item(), inp.size(0))
                top1.update(prec1.item(), inp.size(0))
                top5.update(prec5.item(), inp.size(0))

                if i % 20 == 0:
                    self.logger.info(
                        ('Test: [{0}/{1}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len(self.test_loader), loss=losses, top1=top1, top5=top5))) # , batch_time=batch_time
        self.logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
            .format(top1=top1, top5=top5, loss=losses)))
        self.logger.info(f"Confusion Matrix: \n{cf_matrix}")

        return top1.avg, top5.avg, losses.avg
    
    def run(self, ):
        # 準備資料集和數據加載器
        train_dataset = BaseballDataset("./mlb-youtube/clips/", "./mlb-youtube/label_frames.txt", self.args.new_length, self.args.num_segments, self.args.frame_interval, transform=train_transforms())
        test_dataset = BaseballDataset("./mlb-youtube/clips/", "./mlb-youtube/label_frames.txt", self.args.new_length, self.args.num_segments, self.args.frame_interval, transform=val_transforms())
        total_len = len(train_dataset)
        train_len = int(0.8 * total_len)
        test_len = total_len - train_len
        generator = torch.Generator().manual_seed(0)

        train_dataset, _ = random_split(train_dataset, [train_len, test_len], generator=generator)
        _, test_dataset = random_split(test_dataset, [train_len, test_len], generator=generator)

        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

        # 訓練
        for epoch in range(self.args.epochs):
            train_loss, train_top1, train_top5 = self.train(epoch)
            self.scheduler.step()
            if (epoch + 1) % 1 == 0 or epoch == self.args.epochs - 1:
                prec1, prec5, val_loss = self.validate()
                if prec1 > self.best_prec1:
                    self.is_best = True
                    self.less_loss = val_loss
                    self.best_prec1 = prec1
                elif prec1 == self.best_prec1:
                    self.is_best = val_loss < self.less_loss
                    self.best_prec1 = prec1
                else:
                    self.is_best = False
                self.logger.info(("Best Prec@1: '{}'".format(self.best_prec1)))
                save_epoch = epoch + 1
                self.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        'prec1': prec1,
                        'best_prec1': self.best_prec1,
                    }, save_epoch)
        self.logger.info("Train_Mobilev4TSM" + "\n" + f"Best Prec@1: {self.best_prec1}") 
