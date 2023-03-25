
import torch.nn as nn
from torch import optim

#Initialize weights in model suited for ReLU
def init_net_weight(model):
    if type(model) == nn.Linear:
        nn.init.kaiming_normal_(model.weight, nonlinearity='relu')

    for m in model.modules():
        if isinstance(m, nn.Linear):
            #nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def make_dense_training_step(model, loss_fn, optimizer):
    def train_step(x,y):
        
        #Training mode
        model.train()
        
        #Make prediction
        yhat = model(x)
        
        #Calculate loss
        loss = loss_fn(y, yhat)
        
        #Backward propagation
        loss.backward()
        
        #Update model parameters
        optimizer.step()
        
        #Zero out the gradients
        optimizer.zero_grad()
        
        #Return the loss
        return loss.item()
    return train_step


def make_l1_reg_training_step(model, loss_fn, optimizer, lambda_, lambda_in):
    def train_step(x,y):
         #Training mode
        model.train()
        
        #Make prediction
        yhat = model(x)
        
        # l1 regularization
        l1_reg = 0
        
        for idx, param in enumerate(model.parameters()): 
            if idx==0:
                l1_reg = lambda_in*param.norm(1)

            if idx>0 and idx%2 == 0:
                l1_reg += lambda_*param.norm(1)
            
        
        #Calculate loss
        loss = loss_fn(y, yhat) + l1_reg
        
        
        #Backward propagation
        loss.backward()
        
        #Update model parameters
        optimizer.step()
        
        #Zero out the gradients
        optimizer.zero_grad()
        
        #Return the loss
        return loss.item()
    return train_step
    
def make_l1_reg_training_step_input_skip(model, loss_fn, optimizer, lambda_,lambda_skip, lambda_in):
    def train_step(x,y):
         #Training mode
        model.train()
        
        #Make prediction
        yhat = model(x)
        
        # l1 regularization
        l1_reg = 0
        
        for idx, param in enumerate(model.parameters()): 
            
            if idx ==0:
                l1_reg = lambda_in*param.norm(1)

            if idx>0 and idx%2 == 0:
                l1_reg += lambda_*param[:,0:20].norm(1)
                l1_reg += lambda_skip*param[:,20:].norm(1)
            
        
        #Calculate loss
        loss = loss_fn(y, yhat) + l1_reg
        
        
        #Backward propagation
        loss.backward()
        
        #Update model parameters
        optimizer.step()
        
        #Zero out the gradients
        optimizer.zero_grad()
        
        #Return the loss
        return loss.item()
    return train_step
def model_preparation(model, optim_, l_r, lambda_l1=None,lambda_skip=0.0, lambda_in=0.0,in_skip_mod=False):
    loss_fn = nn.MSELoss()
    model.apply(init_net_weight)

    if optim_=='Adam':
        opt = optim.Adam(model.parameters(), lr=l_r)
    else:
        print('Only Adam optimizer implemented')
        return
    
    if lambda_l1==None:
        training_step = make_dense_training_step(model, loss_fn, opt)

    elif in_skip_mod ==True:
        training_step = make_l1_reg_training_step_input_skip(model,loss_fn, opt, lambda_l1,lambda_skip, lambda_in)
    else:
        training_step = make_l1_reg_training_step(model,loss_fn,opt, lambda_l1, lambda_in)
    
    return training_step
