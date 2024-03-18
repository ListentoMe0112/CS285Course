- When using categorical distributions, please make sure the dimension of action.
```python
# if len(a.shape) > 1
m = torch.distributions.Categorical(torch.tensor([[0.4,0.6],[0.7,0.3]]))
a = torch.tensor([[1],[0]])
# output [2,2]
tensor([[-0.5108, -1.2040],  # ln0.6, ln0.3
        [-0.9163, -0.3567]])  # ln0.4, ln0.7
# if len(a.shape) == 1
b = torch.tensor([1,0])
print(m.log_prob(b))
# output [2]
tensor([-0.5108, -0.3567]) # ln0.6, ln0.7
```