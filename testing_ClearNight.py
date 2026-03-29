import time
import torch
import torchvision
import argparse
import numpy as np
import random

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from networks.ClearNight import *
from datasets.dataset_pairs import my_dataset_eval
from utils.UTILS import compute_psnr
import cv2
from Retinex import *


# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device ----------------------------------------:', device)

parser = argparse.ArgumentParser()
# path setting
parser.add_argument('--experiment_name', type=str, default="testing")
parser.add_argument('--model_path', type=str, default='../../net_epoch_99.pth', help='path to the saved model')

parser.add_argument('--Retinex_decomp', type=bool, default=False)

parser.add_argument('--eval_in_path_RD', type=str, default='test/drop/')
parser.add_argument('--eval_map_path_RD', type=str, default='test/drop1/')
parser.add_argument('--eval_mapf_path_RD', type=str, default='test/drop2/')
parser.add_argument('--eval_gt_path_RD', type=str, default='test/drop_gt/')
parser.add_argument('--eval_in_path_L', type=str, default='test/snow/')
parser.add_argument('--eval_map_path_L', type=str, default='test/snow1/')
parser.add_argument('--eval_mapf_path_L', type=str, default='test/snow2/')
parser.add_argument('--eval_gt_path_L', type=str, default='test/snow_gt/')
parser.add_argument('--eval_in_path_Rain', type=str, default='test/rain/')
parser.add_argument('--eval_map_path_Rain', type=str, default='test/rain1/')
parser.add_argument('--eval_mapf_path_Rain', type=str, default='test/rain2/')
parser.add_argument('--eval_gt_path_Rain', type=str, default='test/rain_gt/')
parser.add_argument('--eval_in_path_Other', type=str, default='test/haze/')
parser.add_argument('--eval_map_path_Other', type=str, default='test/haze1/')
parser.add_argument('--eval_mapf_path_Other', type=str, default='test/haze2/')
parser.add_argument('--eval_gt_path_Other', type=str, default='test/haze_gt/')

parser.add_argument('--output_dir', type=str, default='output_images/', help='directory to save output images')

args = parser.parse_args()

# 创建输出目录及子文件夹
output_dirs = {
    'RD': os.path.join(args.output_dir, 'drop'),
    'Rain': os.path.join(args.output_dir, 'rain'),
    'Snow': os.path.join(args.output_dir, 'snow'),
    'Other': os.path.join(args.output_dir, 'other')
}

if args.Retinex_decomp:
    process_folder(args.eval_in_path_RD, args.eval_mapf_path_RD, args.eval_map_path_RD, sigma=10)
    process_folder(args.eval_in_path_Rain, args.eval_mapf_path_Rain, args.eval_map_path_Rain, sigma=10)
    process_folder(args.eval_in_path_L, args.eval_mapf_path_L, args.eval_map_path_L, sigma=10)
    process_folder(args.eval_in_path_Other, args.eval_mapf_path_Other, args.eval_map_path_Other, sigma=10)

for d in output_dirs.values():
    if not os.path.exists(d):
        os.makedirs(d)

trans_eval = transforms.Compose([transforms.ToTensor()])

def get_eval_data(val_in_path, val_gt_path, val_map_path, val_mapf_path, trans_eval=trans_eval):
    eval_data = my_dataset_eval(root_in=val_in_path, root_label=val_gt_path, root_imap=val_map_path,  root_rmap=val_mapf_path, transform=trans_eval, fix_sample=1000)
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4)
    return eval_loader

def save_image(output_dir, filename, outputs):
    save_path = os.path.join(output_dir, filename)
    torchvision.utils.save_image(outputs.cpu(), save_path, nrow=1, padding=5)

def test(net, eval_loader, output_dir, Dname='S'):
    net.eval()
    with torch.no_grad():
        eval_output_psnr = 0.0
        eval_input_psnr = 0.0
        total_classification_accuracy = 0.0
        st = time.time()
        
        all_labels = ["haze", "snow", "rain", "drop"]
        num_labels = len(all_labels)
        
        for index, (data_in, label, map, mapf, name) in enumerate(eval_loader, 0):
            inputs = Variable(data_in).to(device)
            maps = Variable(map).to(device)
            mapfs = Variable(mapf).to(device)
            labels = Variable(label).to(device)
            
            batch_size = data_in.size(0)
            target_labels = torch.zeros(batch_size, num_labels).to(device)
            for b in range(batch_size):
                for j, lbl in enumerate(all_labels):
                    if lbl in name[b]:
                        target_labels[b, j] = 1

            outputs, classification_logits, _, _ = net(inputs, maps, mapfs, name)
            
            eval_input_psnr += compute_psnr(inputs, labels)
            eval_output_psnr += compute_psnr(outputs, labels)
            
            probs = torch.sigmoid(classification_logits)
            predicted_labels = (probs > 0.5).float()
            accuracy = (predicted_labels == target_labels).float().mean()
            total_classification_accuracy += accuracy.item()
            
            save_image(output_dir, name[0], outputs)
        
        Final_output_PSNR = eval_output_psnr / len(eval_loader)
        Final_input_PSNR = eval_input_psnr / len(eval_loader)
        Final_classification_accuracy = total_classification_accuracy / len(eval_loader)
        
        print(f"Dname:{Dname} | Num_eval:{len(eval_loader)} | In_PSNR:{round(Final_input_PSNR, 2)} | "
              f"Out_PSNR:{round(Final_output_PSNR, 2)} | Clf_Acc:{round(Final_classification_accuracy, 2)} | "
              f"cost time: {time.time() - st}")

if __name__ == '__main__':
    net = ClearNight_()
    net.to(device)

    if os.path.exists(args.model_path):
        net.load_state_dict(torch.load(args.model_path), strict=True)
        print("Model loaded from", args.model_path)
    else:
        raise Exception("No model found at specified path")

    eval_loader_RD = get_eval_data(val_in_path=args.eval_in_path_RD, val_gt_path=args.eval_gt_path_RD, val_map_path=args.eval_map_path_RD, val_mapf_path=args.eval_mapf_path_RD)
    eval_loader_Rain = get_eval_data(val_in_path=args.eval_in_path_Rain, val_gt_path=args.eval_gt_path_Rain, val_map_path=args.eval_map_path_Rain, val_mapf_path=args.eval_mapf_path_Rain)
    eval_loader_L = get_eval_data(val_in_path=args.eval_in_path_L, val_gt_path=args.eval_gt_path_L, val_map_path=args.eval_map_path_L, val_mapf_path=args.eval_mapf_path_L)
    eval_loader_Other = get_eval_data(val_in_path=args.eval_in_path_Other, val_gt_path=args.eval_gt_path_Other, val_map_path=args.eval_map_path_Other, val_mapf_path=args.eval_mapf_path_Other)

    test(net=net, eval_loader=eval_loader_RD, output_dir=output_dirs['RD'], Dname='RD')
    test(net=net, eval_loader=eval_loader_Rain, output_dir=output_dirs['Rain'], Dname='Rain')
    test(net=net, eval_loader=eval_loader_L, output_dir=output_dirs['Snow'], Dname='Snow')
    test(net=net, eval_loader=eval_loader_Other, output_dir=output_dirs['Other'], Dname='Other')

    
