import sys,time
import numpy as np
import torch

import utils

class Appr(object):

    def __init__(self,model,nepochs=20,sbatch=128,lr=0.01,clipgrad=10):
        self.model=model

        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.clipgrad=clipgrad
        _set_random()

        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=2e-4)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.nepochs)

        return


    def train(self,t,xtrain,ytrain,xvalid,yvalid):

        if t == 0:
            epochs = self.nepochs*3
        else:
            epochs = self.nepochs
        # Loop epochs
        for e in range(epochs):
            # Train
            clock0=time.time()
            self.train_epoch(t,xtrain,ytrain)
            clock1=time.time()
            train_loss,train_acc=self.eval(t,xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            print()

        return

    def train_epoch(self,t,x,y):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=torch.autograd.Variable(x[b],volatile=False)
            targets=torch.autograd.Variable(y[b],volatile=False)

            # Forward
            outputs=self.model.forward(images)
            output=outputs[t]
            loss=self.criterion(output,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return

    def eval(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r=np.arange(x.size(0))
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            with torch.no_grad():
                images=x[b]
                targets=y[b]

            # Forward
            outputs=self.model.forward(images)
            output=outputs[t]
            loss=self.criterion(output,targets)

            total_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(output.data, 1)
            total_num += targets.size(0)
            total_acc += (predicted == targets.data).sum()

        return total_loss/total_num,total_acc/total_num

def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
