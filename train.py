import math
import torch
import os
import time
import datetime
import torch.backends.cudnn as cudnn
from dataset import CaptchaDataSet
from CRNN_res_v1 import CRNN_res
from log import logger
from config import TrainConfig,DataConfig,ProjectConfig
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader,random_split
from torch.nn import CTCLoss
from torchvision.transforms import v2
from torch.utils.data.dataloader import default_collate

def custom_collate_fn(batch,T=50):
    items = list(zip(*batch)) # 原本的形式是[(a,b),(a,b),...]，zip()可以理解为压缩，zip(*)可以理解为解压，从这个二元组的列表中解压出两个列表。压缩则是把两个列表压缩为二元组的列表。压缩或解压的直接结果是zip对象，而非列表，这是一个可迭代对象，需要使用list函数转为列表
    # 这里将（图像，标签）这一二元组的列表解压出来，单独处理。
    items[0] = default_collate(items[0])
    labels = list(items[1])
    items[1] = []
    target_lengths = torch.zeros((len(batch,)),dtype=torch.int)
    input_lengths = torch.zeros(len(batch,),dtype=torch.int)
    for idx,label in enumerate(labels):
        target_lengths[idx] = len(label)
        items[1].extend(label)
        input_lengths[idx] = T

    return items[0], torch.tensor(items[1]), target_lengths, input_lengths

def val(net,val_dataset,ctc_loss,max_iter=100):
    # 将模型置于评估模式下
    net.eval()

    # 加载数据集
    data_loader = DataLoader(val_dataset,train_config.batch_size,shuffle=True,num_workers=train_config.num_workers,collate_fn=custom_collate_fn)

    # 初始化参数
    loss_avg = 0.0
    max_iter = min(max_iter,len(data_loader))

    for i,(images,labels,target_lengths,input_lengths) in enumerate(data_loader):
        # 超过设定的最大迭代次数则退出
        if i >= max_iter:
            break

        predict = log_softmax(net(images),dim=2)

        cost = ctc_loss(log_probs=predict,targets=labels,target_lengths=target_lengths,input_lengths=input_lengths)
        loss_avg += cost
    logger.info(f"val loss:{loss_avg/max_iter}")
    
    # 评估结束，将模型恢复为训练模式
    net.train()

def train(net,optimizer,train_set,val_set,use_gpu):
    ctc_loss = CTCLoss(blank=0,reduction='mean')
    net.train()
    epoch = 0
    logger.info("初始化训练参数中...")

    epoch_size = math.ceil(len(train_set)/train_config.batch_size) # 一个epoch包含的batch数量
    max_iter = train_config.max_epoch*epoch_size # 最大迭代次数，每次迭代取一个batch的数据进行训练

    start_iter = 0

    logger.info("开始训练...")
    for iteration in range(start_iter,max_iter):
        # 若一个新的epoch开始
        if iteration % epoch_size == 0:
            epoch += 1
            # 重新初始化batch迭代器
            batch_iterator = iter(DataLoader(train_set,train_config.batch_size,shuffle=True,num_workers=train_config.num_workers,collate_fn=custom_collate_fn))

            if epoch%2==0 and epoch>0:
                # 每10个epoch保存一下模型
                if train_config.num_gpu>1:
                    # 这里为什么在多gpu的环境下使用这种方式进行储存？
                    torch.save(net.module.state_dict(),os.path.join(project_config.weights_save_folder,'epoch_'+str(epoch)+'.pth'))
                else:
                    torch.save(net.state_dict(),os.path.join(project_config.weights_save_folder,'epoch_'+str(epoch)+'.pth'))

                val(net,val_set,ctc_loss)

        load_t0 = time.time()
        # 这里我并不理解，姑且这样写。我觉得迭代器只会返回dataset类里面__getitem__魔法函数返回的内容，但是这里有四个返回值
        images,labels,target_lengths,input_lengths = next(batch_iterator)
        if use_gpu:
            # 这样将tensor转移到gpu上对吗？难道不是设定tensor的device属性？
            images = images.cuda()
            labels = labels.cuda()
            target_lengths = target_lengths.cuda()
            input_lengths = input_lengths.cuda()
        # 计算预测结果，即从最开始的输入层层正向传播到最后
        out = net(images)
        # 清空上一次训练的梯度信息
        optimizer.zero_grad()
        # 将输出传递到损失函数层，计算损失
        loss = ctc_loss(log_probs=out,targets=labels,target_lengths=target_lengths,input_lengths=input_lengths)
        # 调用backward()函数进行反向传播，计算出每一层的梯度
        loss.backward()
        # 用优化器结合已计算出的梯度信息和被优化器管理的学习率信息，更新参数
        optimizer.step()

        load_t1 = time.time()
        batch_time = load_t1-load_t0
        eta = int(batch_time*(max_iter-iteration))

        logger.info(f"Epoch:{epoch}/{train_config.max_epoch}||Epochiter:{(iteration % epoch_size) + 1}/{epoch_size}||Iter:{iteration+1}/{max_iter}||Loss:{loss}||BatchTime:{batch_time:.4f}s||ETA:{str(datetime.timedelta(seconds=eta))}")

    logger.info("训练结束，保存模型中..")

    if train_config.num_gpu > 1:
        torch.save(net.module.state_dict(), os.path.join(project_config.weights_save_folder, 'Final.pth'))
    else:
        torch.save(net.state_dict(), os.path.join(project_config.weights_save_folder, 'Final.pth'))

    logger.info("保存成功")

if __name__ == '__main__':
    data_config = DataConfig()
    train_config = TrainConfig()
    project_config = ProjectConfig()

    use_gpu = torch.cuda.is_available()

    net = CRNN_res(imgH=data_config.height,nc=3,nclass=len(data_config.character_set),nh=100)

    # 加载训练好的模型权重
    if train_config.pre_train:
        pretrained_dict = torch.load(os.path.join(project_config.weights_save_folder,"Final.pth"))
        model_dict = net.state_dict()
        # 只选择需要的那部分
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    if train_config.num_gpu>1 and use_gpu:
        net = torch.nn.DataParallel(net).cuda()
    elif use_gpu:
        net = net.cuda()

    cudnn.benchmark = True
    optimizer = torch.optim.Adam(net.parameters(),lr=train_config.initial_learning_rate,weight_decay=train_config.weight_decay)

    transform = v2.Compose([v2.ToImage(),
                            v2.Resize((data_config.height,data_config.width)),
                            v2.ToDtype(torch.float32)])

    dataset = CaptchaDataSet(img_dir=project_config.image_folder,annotation_path=project_config.train_annotation_path,character_set=data_config.character_set,transform=transform,transform_target=None)

    train_size = int(len(dataset)*0.5)
    val_size = int(len(dataset)*0.3)
    test_size = len(dataset) - train_size - val_size

    train_data,val_data,test_data = random_split(dataset,[train_size,val_size,test_size])

    # train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True)
    # val_loader = DataLoader(val_data,batch_size=args.batch_size,shuffle=True)
    # test_loader = DataLoader(test_data,batch_size=args.batch_size,shuffle=True)

    train(net,optimizer=optimizer,train_set=train_data,val_set=val_data,use_gpu=use_gpu)
