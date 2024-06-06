from torch.optim import SGD, Adam, lr_scheduler
import torch
import tensorboard,tensorboardX
from tqdm import tqdm

def train(model,loaders,lr,nTrain,saveFreq,testFreq,id,save_dir):


    writer = tensorboardX.SummaryWriter(f"{save_dir}/logs/{id}", flush_secs=1)

    gpu = torch.device('cuda')
    model.cuda(gpu)
    optimizer = Adam(model.parameters(),lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.99)
    scaler = torch.cuda.amp.GradScaler()
    scaleInit = scaler.get_scale()
    for epoch in tqdm(range(nTrain),desc='training model'):
        
        for step,(x1,x2) in enumerate(loaders['train'],start=epoch*len(loaders['train'])):

            x1,x2 = x1[:,None,:].cuda(gpu).float(),x2[:,None,:].cuda(gpu).float()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                loss = model.forward(x1,x2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scaleInit:
                scheduler.step()
                scaleInit = scalar.get_scale()
            else:
                scaleInit = scalar.get_scale()

            if (step % 50) == 0:

                writer.add_scalar('train/loss',loss.item(),step)
            
        if (epoch % testFreq) == 0:
            with torch.no_grad():
                test_loss = 0.
                for (x1,x2) in loaders['test']:
                    x1,x2 = x1[:,None,:].cuda(gpu).float(),x2[:,None,:].cuda(gpu).float()
                    with torch.cuda.amp.autocast():
                        loss = model.forward(x1,x2)
                    test_loss += loss.item()

                writer.add_scalar('test/loss',test_loss/len(loaders['test']),step)
    
        if (epoch % saveFreq) == 0:
            state = dict(epoch = epoch +1,
                         model = model.state_dict(),
                         optimizer = optimizer.state_dict())
            torch.save(state,save_dir + f'/checkpoint_{epoch}.tar')

    writer.close()






