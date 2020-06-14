# second_pytorch_bpnn.py
# use optimizer
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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(t_epoch):
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    if( t % 10 == 0):
        print("epoch {:3d} : loss {:.3f}".format(t, loss)) 
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

