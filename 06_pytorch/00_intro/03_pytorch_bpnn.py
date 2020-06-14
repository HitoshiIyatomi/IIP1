# third_pytorch_bpnn.py
# define your own neural networks using Module
import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, x)
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 500, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = TwoLayerNet(D_in, H, D_out)

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