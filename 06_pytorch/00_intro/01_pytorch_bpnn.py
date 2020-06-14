# first_pytorch_bpnn.py
# 
import torch

N, D_in, H, D_out = 64, 500, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

learning_rate = 1e-2
t_epoch = 300

for t in range(t_epoch):
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    if( t % 10 == 0):
        print("epoch {:3d} : loss {:.3f}".format(t, loss)) 
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
    model.zero_grad()
