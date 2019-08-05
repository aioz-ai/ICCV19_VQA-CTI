import torch

def ModeProduct(tensor, matrix_1, matrix_2, matrix_3, matrix_4, n_way=3):

    #mode-1 tensor product
    tensor_1 = tensor.transpose(3,2).contiguous().view(tensor.size(0), tensor.size(1), tensor.size(2)*tensor.size(3)*tensor.size(4))
    tensor_product = torch.matmul(matrix_1, tensor_1)
    tensor_1 = tensor_product.view(-1, tensor_product.size(1),tensor.size(4), tensor.size(3), tensor.size(2)).transpose(4,2)

    #mode-2 tensor product
    tensor_2 = tensor_1.transpose(2,1).transpose(4,2).contiguous().view(-1, tensor_1.size(2), tensor_1.size(1)*tensor_1.size(3)*tensor_1.size(4))
    tensor_product = torch.matmul(matrix_2, tensor_2.float())
    tensor_2 = tensor_product.view(-1, tensor_product.size(1), tensor_1.size(4), tensor_1.size(3), tensor_1.size(1)).transpose(4,1).transpose(4,2)
    tensor_product = tensor_2
    if n_way > 2:
        #mode-3 tensor product
        tensor_3 = tensor_2.transpose(3,1).transpose(4,2).transpose(4,3).contiguous().view(-1, tensor_2.size(3), tensor_2.size(2)*tensor_2.size(1)*tensor_2.size(4))
        tensor_product = torch.matmul(matrix_3, tensor_3.float())
        tensor_3 = tensor_product.view(-1, tensor_product.size(1), tensor_2.size(4), tensor_2.size(2), tensor_2.size(1)).transpose(1,4).transpose(4,2).transpose(3,2)
        tensor_product = tensor_3
    if n_way > 3:
    #mode-4 tensor product
        tensor_4 = tensor_3.transpose(4,1).transpose(3,2).contiguous().view(-1, tensor_3.size(4), tensor_3.size(3)*tensor_3.size(2)*tensor_3.size(1))
        tensor_product = torch.matmul(matrix_4, tensor_4)
        tensor_4 = tensor_product.view(-1, tensor_product.size(1), tensor_3.size(3), tensor_3.size(2), tensor_3.size(1)).transpose(4,1).transpose(3,2)
        tensor_product = tensor_4

    return tensor_product

if __name__=='__main__':
    X = torch.tensor([[[1, 13], [4, 16], [7, 19], [10, 22]], [[2, 14], [5, 17], [8, 20], [11, 23]], [[3, 15], [6, 18], [9, 21], [12, 24]]])
    U_1 = torch.tensor([[1, 3, 5], [2, 4, 6]]).unsqueeze(0)
    U_2 = torch.randn(1,5,4)
    U_3 = torch.randn(1,2,2)
    ModeProduct(X.unsqueeze(0).unsqueeze(4), U_1, U_2, U_3, None)

