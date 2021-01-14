import torch


device = "cpu"
def assign_a_gpu(gpu_no):
    device = torch.device("cuda:%s"%(str(gpu_no)) if torch.cuda.is_available() else "cpu")
    return device