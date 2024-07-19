import torch
import os
import time
from log import logger
from config import args
from CRNN_res_v1 import CRNN_res
import cv2
from torchvision.transforms import v2


def decode_out(str_index, characters):
    char_list = []
    for i in range(len(str_index)):
        if str_index[i] != 0 and (not (i > 0 and str_index[i - 1] == str_index[i])):
            char_list.append(characters[str_index[i]])
    return ''.join(char_list)

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    
    net = CRNN_res(imgH=args.height,nc=3,nclass=len(args.characters),nh=100)
    device = torch.device("cpu" if args.cpu else "cuda")
    net.load_state_dict(torch.load(args.trained_model,map_location=torch.device(device)))
    net.eval()

    input_path = args.input_path
    image_paths = []

    if os.path.isdir(input_path):
        for input_file in os.listdir(input_path):
            image_paths += [input_path+input_file]
    else:
        image_paths += [input_path]

    image_paths = [input_file for input_file in image_paths if (input_file[-4:] in ['.jpg','.png','JPEG'])]

    for img_path in image_paths:
        begin = time.time()
        logger.info(f"recog: {img_path}")
        image = cv2.imread(img_path)
        transform = v2.Compose([v2.ToTensor(),
                            v2.Resize((args.height,args.width))])
        
        image = transform(image)

        out = net(image)
        _, preds = out.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        
        lab2str = decode_out(preds, args.characters)
        logger.info(lab2str)
        end = time.time()

    logger.info("预测结束")
