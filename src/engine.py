import torch
import config

def train_step(model,
               dataloader:torch.utils.data,
               loss_fn:torch.nn,
               optimizer:torch.optim,
               device=config.DEVICE):
    
    train_loss,train_acc = 0,0
    for batch,(X,y) in enumerate(dataloader):
        model.train()
        X, y = X.to(device), y.to(device)
            
        y_logits = model(X)

        loss = loss_fn(y_logits, y)

        train_loss += loss
        train_acc += (y_logits.argmax(1) == y).sum().item() / len(y)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
        
    return train_loss,train_acc



def test_step(model,
               dataloader:torch.utils.data,
               loss_fn:torch.nn,
               device=config.DEVICE):
    
    with torch.inference_mode():
        model.eval()
        
        test_loss,test_acc = 0,0
        for batch, (X_test, y_test) in enumerate(dataloader):
            
            X_test, y_test = X_test.to(device), y_test.to(device)
            
            y_logits = model(X_test)

            loss = loss_fn(y_logits, y_test)

            test_loss += loss.item()

            # Compute accuracy
            test_preds = y_logits.argmax(dim=1)
            test_acc += (test_preds == y_test).sum().item() / len(y_test)
            
        
        test_loss /= len(dataloader)  
        test_acc /= len(dataloader) 


    return test_loss,test_acc 

