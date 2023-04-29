import time
import numpy as np
import torch
from torch.backends import cudnn
import tqdm
from PIL import Image
import glob

def path2tensor(path):
    data_lowlight = Image.open(path)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.unsqueeze(0)
    return data_lowlight

def gpu_speed_test(model,path,test_num=300):
    # model: 需要测试的模型
    # path: 输入图片路径
    image_list = glob.glob(path + "/**/" + "*.jpg", recursive=True) + glob.glob(path+ "/**/" + "*.png",
                                                                                       recursive=True)
    if(len(image_list)==0):
        print("没有找到测试图片")
        return
    device = 'cuda:0'
    repetitions = 300
    dummy_input = torch.rand(1,3,256,256)
    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

# synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()


# 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# 初始化一个时间容器
    timings = np.zeros((test_num, 1))
    with torch.no_grad():
            for i in range(test_num):
                input_tensor = path2tensor(image_list[i%len(image_list)]).cuda()
                starter.record()
                _ = model(input_tensor)
                ender.record()
                torch.cuda.synchronize() # 等待GPU任务完成
                curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
                timings[i] = curr_time
    avg = timings.sum()/test_num
    print('\navg={}\n'.format(avg))

def cpu_speed_test(model,path,test_num=300):
    # model: 需要测试的模型
    # path: 输入图片路径
    image_list = glob.glob(path + "/**/" + "*.jpg", recursive=True) + glob.glob(path + "/**/" + "*.png",
                                                                                   recursive=True)
    if(len(image_list)==0):
        print("没有找到测试图片")
        return

    print("测试集共有"+str(len(image_list))+"张图片")
# 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
# 初始化一个时间容器
    timings = np.zeros((test_num, 1))
    with torch.no_grad():
            for i in range(test_num):
                input_tensor = path2tensor(image_list[i%len(image_list)])
                t1 = time.time()
                _ = model(input_tensor)
                t2 = time.time()
                timings[i] = t2-t1
    avg = timings.sum()/test_num
    print('\navg={}\n'.format(avg))
