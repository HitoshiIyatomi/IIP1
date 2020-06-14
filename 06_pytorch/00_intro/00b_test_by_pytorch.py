import torch

device ='cuda:0'
#device ='cpu'
N,D = 3,4

torch.manual_seed(0)

x = torch.randn(N, D, requires_grad=True, device=device)
y = torch.randn(N, D, device=device)
z = torch.randn(N, D, device=device)

a = x * y
b = a + z
c = torch.sum(b)

c.backward()
print(x.grad)
print(c)