# fourth_pytorch_bpnn.py
# define your own neural networks using Module 2
# use ParallelBlock
import torch

class ParallelBlock(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_out)
        self.linear2 = torch.nn.Linear(D_in, D_out)
    
    def forward(self, x)
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        return (h1 * h2).clamp(min=0)

N, D_in, H, D_out = 64, 500, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    ParallelBlock(D_in, H),
    ParallelBlock(H, H),
    ParallelBlock(H, D_out))

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