# import torch

# def my_softmax(x):
#     exp_x = torch.exp(x)
#     return exp_x / exp_x.sum()

# x = torch.tensor([1.0, 2.0, 3.0])

# my_result = my_softmax(x)
# torch_result = torch.softmax(x, dim=0)

# print("my softmax:", my_result)
# print("torch softmax:", torch_result)

# print("difference:", my_result - torch_result)



import torch

x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

print("x:")
print(x)

print("\nsoftmax dim=0:")
print(torch.softmax(x, dim=0))

print("\nsoftmax dim=1:")
print(torch.softmax(x, dim=1))