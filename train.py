from torch.optim import SGD, Adam
import torch
import tensorboard,tensorboardX

def train(model,loaders,lr,nTrain,saveFreq,testFreq,id,save_dir):


    writer = tensorboardX.SummaryWriter(f"logs/{id}", flush_secs=1)

    gpu = torch.device('cuda')

    optimizer = Adam(model.params(),lr)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(nTrain):
        
        for step,(x1,x2) in enumerate(loaders['train'],start=epoch*len(loaders['train'])):

            x1,x2 = x1.cuda(gpu),x2.cuda(gpu)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(x1,x2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (step % 50) == 0:

                writer.add_scalar('train/loss',loss.item(),step)
            
        if (epoch % testFreq) == 0:
            with torch.no_grad():
                test_loss = 0.
                for (x1,x2) in loaders['test']:
                    x1,x2 = x1.cuda(gpu),x2.cuda(gpu)
                    with torch.cuda.amp.autocast():
                        loss = model.forward()
                    test_loss += loss.item()

                writer.add_scaler('test/loss',test_loss/len(loaders['test']),step)
    
        if (epoch % saveFreq) == 0:
            state = dict(epoch = epoch +1,
                         model = model.state_dict(),
                         optimizer = optimizer.state_dict())
            torch.save(state,save_dir + f'/checkpoint_{epoch+1}.tar')

    writer.close()






