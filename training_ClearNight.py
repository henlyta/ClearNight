import time,torchvision,argparse,sys,os
import random
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch.optim as optim

from datasets.dataset_pairs import my_dataset, my_dataset_eval

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from networks.ClearNight_model import *
from utils.UTILS import compute_psnr
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from loss.perceptual import LossNetwork
from Retinex import *

import torchvision.transforms as transforms

sys.path.append(os.getcwd())
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device ----------------------------------------:',device)

parser = argparse.ArgumentParser()
# path setting
parser.add_argument('--experiment_name', type=str, default="ClearNight")
parser.add_argument('--unified_path', type=str, default='Checkpoint/')

parser.add_argument('--resume_training', type=bool, default=False)
parser.add_argument('--resume_checkpoint', type=int, default=0)
parser.add_argument('--Retinex_decomp', type=bool, default=False)

parser.add_argument('--training_in_path', type=str, default='data/train/snow/')
parser.add_argument('--training_imap_path', type=str, default='data/train/snow1/')
parser.add_argument('--training_rmap_path', type=str, default='data/train/snow2/')
parser.add_argument('--training_gt_path', type=str, default='data/train/snow_gt/')

parser.add_argument('--training_in_pathRain', type=str, default='data/train/rain/')
parser.add_argument('--training_imap_pathRain', type=str, default='data/train/rain1/')
parser.add_argument('--training_rmap_pathRain', type=str, default='data/train/rain2/')
parser.add_argument('--training_gt_pathRain', type=str, default='data/train/rain_gt/')

parser.add_argument('--training_in_pathRD', type=str, default='data/train/drop/')
parser.add_argument('--training_imap_pathRD', type=str, default='data/train/drop1/')
parser.add_argument('--training_rmap_pathRD', type=str, default='data/train/drop2/')
parser.add_argument('--training_gt_pathRD', type=str, default='data/train/drop_gt/')

parser.add_argument('--writer_dir', type=str, default='logs/')

parser.add_argument('--eval_in_path_RD', type=str,default='data/test/drop/')
parser.add_argument('--eval_imap_path_RD', type=str,default='data/test/drop1/')
parser.add_argument('--eval_rmap_path_RD', type=str,default='data/test/drop2')
parser.add_argument('--eval_gt_path_RD', type=str,default='data/test/drop_gt/')

parser.add_argument('--eval_in_path_L', type=str,default='data/test/snow/')
parser.add_argument('--eval_imap_path_L', type=str,default='data/test/snow1/')
parser.add_argument('--eval_rmap_path_L', type=str,default='data/test/snow2/')
parser.add_argument('--eval_gt_path_L', type=str,default='data/test/snow_gt/')

parser.add_argument('--eval_in_path_Rain', type=str,default='data/test/rain/')
parser.add_argument('--eval_imap_path_Rain', type=str,default='data/test/rain1/')
parser.add_argument('--eval_rmap_path_Rain', type=str,default='data/test/rain2/')
parser.add_argument('--eval_gt_path_Rain', type=str,default='data/test/rain_gt/')

# training setting
parser.add_argument('--EPOCH', type=int, default=100)
parser.add_argument('--T_period', type=int, default=50)  
parser.add_argument('--BATCH_SIZE', type=int, default=1)
parser.add_argument('--Crop_patches', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--print_frequency', type=int, default=100)
parser.add_argument('--max_psnr', type=int, default=10)
parser.add_argument('--fix_sample', type=int, default=5000)
parser.add_argument('--lam_VGG', type=float, default=0.1)

parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--addition_loss', type=str, default='VGG')
parser.add_argument('--depth_loss', type=bool, default=True)
parser.add_argument('--lam_DepthLoss', type=float, default=0.02)
parser.add_argument('--lam_classification_loss', type=float, default=0.001)
parser.add_argument('--lam_balance_loss', type=float, default=0.01)

parser.add_argument('--Aug_regular', type=bool, default=False)
args = parser.parse_args()

if args.debug == True:
    fix_sampleA = 1000
    fix_sampleB = 1000
    fix_sampleC = 1000
    print_frequency = 10
else:
    fix_sampleA = args.fix_sample
    fix_sampleB = args.fix_sample
    fix_sampleC = args.fix_sample
    print_frequency = args.print_frequency

exper_name = args.experiment_name
writer = SummaryWriter(args.writer_dir + exper_name)
if not os.path.exists(args.writer_dir):
    os.mkdir(args.writer_dir)

unified_path = args.unified_path
SAVE_PATH = unified_path + exper_name + '/'
if not os.path.exists(args.unified_path):
    os.mkdir(args.unified_path)
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

trans_eval = transforms.Compose([transforms.ToTensor()])
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
print("==" * 50)

def check_dataset(in_path, gt_path, name='RD'):
    print("Check {} pairs({}) ???: {}".format(name, len(in_path), os.listdir(in_path) == os.listdir(gt_path)))

if args.Retinex_decomp:
    process_folder(args.training_in_path, args.training_rmap_path, args.training_imap_path, sigma=10)
    process_folder(args.training_in_pathRD, args.training_rmap_pathRD, args.training_imap_pathRD, sigma=10)
    process_folder(args.training_in_pathRain, args.training_rmap_pathRain, args.training_imap_pathRain, sigma=10)
    process_folder(args.eval_in_path_RD, args.eval_rmap_path_RD, args.eval_imap_path_RD, sigma=10)
    process_folder(args.eval_in_path_Rain, args.eval_rmap_path_Rain, args.eval_imap_path_Rain, sigma=10)
    process_folder(args.eval_in_path_L, args.eval_rmap_path_L, args.eval_imap_path_L, sigma=10)

check_dataset(args.eval_in_path_RD, args.eval_gt_path_RD)
check_dataset(args.eval_in_path_Rain, args.eval_gt_path_Rain)
check_dataset(args.eval_in_path_L, args.eval_gt_path_L)
check_dataset(args.training_in_path, args.training_gt_path)
check_dataset(args.training_in_pathRain, args.training_gt_pathRain)
check_dataset(args.training_in_pathRD, args.training_gt_pathRD)
check_dataset(args.eval_imap_path_RD, args.eval_gt_path_RD)
check_dataset(args.eval_imap_path_Rain, args.eval_gt_path_Rain)
check_dataset(args.eval_imap_path_L, args.eval_gt_path_L)
check_dataset(args.eval_rmap_path_RD, args.eval_gt_path_RD)
check_dataset(args.eval_rmap_path_Rain, args.eval_gt_path_Rain)
check_dataset(args.eval_rmap_path_L, args.eval_gt_path_L)
check_dataset(args.training_imap_path, args.training_gt_path)
check_dataset(args.training_imap_pathRain, args.training_gt_pathRain)
check_dataset(args.training_imap_pathRD, args.training_gt_pathRD)
check_dataset(args.training_rmap_path, args.training_gt_path)
check_dataset(args.training_rmap_pathRain, args.training_gt_pathRain)
check_dataset(args.training_rmap_pathRD, args.training_gt_pathRD)
print("==" * 50)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def test(net, eval_loader, epoch=1, max_psnr_val=26, Dname='S'):
    net.eval()
    with torch.no_grad():
        eval_output_psnr = 0.0
        eval_input_psnr = 0.0
        total_classification_accuracy = 0.0
        st = time.time()
        for index, (data_in, label, imap, rmap, name) in enumerate(eval_loader, 0):
            inputs = Variable(data_in).to(device)
            imaps = Variable(imap).to(device)
            rmaps = Variable(rmap).to(device)
            labels = Variable(label).to(device)

            batch_size = data_in.size(0)
            target_labels = torch.zeros(batch_size, num_labels).to(device)
            for b in range(batch_size):
                for j, lbl in enumerate(all_labels):
                    if lbl in name[b]:
                        target_labels[b, j] = 1

            outputs, classification_logits, _, _ = net(inputs, imaps, rmaps, name)

            eval_input_psnr += compute_psnr(inputs, labels)
            eval_output_psnr += compute_psnr(outputs, labels)

            probs = torch.sigmoid(classification_logits)
            predicted_labels = (probs > 0.5).float()
            accuracy = (predicted_labels == target_labels).float().mean()
            total_classification_accuracy += accuracy.item()

        Final_output_PSNR = eval_output_psnr / len(eval_loader)
        Final_input_PSNR = eval_input_psnr / len(eval_loader)
        Final_classification_accuracy = total_classification_accuracy / len(eval_loader)

        writer.add_scalars(exper_name + '/testing', {
            'eval_PSNR_Output': Final_output_PSNR,
            'eval_PSNR_Input': Final_input_PSNR,
            'eval_classification_accuracy': Final_classification_accuracy
        }, epoch)

        print(f"epoch:{epoch}---------Dname:{Dname}--------------[Num_eval:{len(eval_loader)} In_PSNR:{round(Final_input_PSNR, 2)}  Out_PSNR:{round(Final_output_PSNR, 2)} Clf_Acc:{round(Final_classification_accuracy, 2)}]--------max_psnr_val:{round(max_psnr_val, 2)}, cost time: {time.time() - st}")
        if Final_output_PSNR > max_psnr_val:
            max_psnr_val = Final_output_PSNR
    return max_psnr_val

def save_imgs_for_visual(path, inputs, labels, outputs):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0]], path, nrow=3, padding=0)

def get_training_data(fix_sampleA=fix_sampleA, fix_sampleB=fix_sampleB, fix_sampleC=fix_sampleC, Crop_patches=args.Crop_patches):
    rootA_in = args.training_in_pathRain
    rootA_imap = args.training_imap_pathRain
    rootA_rmap = args.training_rmap_pathRain
    rootA_label = args.training_gt_pathRain
    rootB_in = args.training_in_path
    rootB_imap = args.training_imap_path
    rootB_rmap = args.training_rmap_path
    rootB_label = args.training_gt_path
    rootC_in = args.training_in_pathRD
    rootC_imap = args.training_imap_pathRD
    rootC_rmap = args.training_rmap_pathRD
    rootC_label = args.training_gt_pathRD
    train_datasets = my_dataset(rootA_in, rootA_label, rootA_imap, rootA_rmap, rootB_in, rootB_label, rootB_imap, rootB_rmap, rootC_in, rootC_label, rootC_imap, rootC_rmap, crop_size=Crop_patches,
                                fix_sample_A=fix_sampleA, fix_sample_B=fix_sampleB, fix_sample_C=fix_sampleC, regular_aug=args.Aug_regular)
    train_loader = DataLoader(dataset=train_datasets, batch_size=args.BATCH_SIZE, num_workers=8, shuffle=True)
    print('len(train_loader):', len(train_loader))
    return train_loader

def get_eval_data(val_in_path=args.eval_in_path_Rain, val_gt_path=args.eval_gt_path_Rain, val_imap_path=args.eval_imap_path_Rain, val_rmap_path=args.eval_rmap_path_Rain, trans_eval=trans_eval):
    eval_data = my_dataset_eval(
        root_in=val_in_path, root_label=val_gt_path, root_imap=val_imap_path, root_rmap=val_rmap_path, transform=trans_eval, fix_sample=500)
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4)
    return eval_loader

def print_param_number(net):
    print('#generator parameters:', sum(param.numel() for param in net.parameters()))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ClearNight_().to(device)

    all_labels = ["haze", "snow", "rain", "drop"]
    num_labels = len(all_labels)

    train_loader = get_training_data()
    eval_loader_RD = get_eval_data(val_in_path=args.eval_in_path_RD, val_gt_path=args.eval_gt_path_RD, val_imap_path=args.eval_imap_path_RD, val_rmap_path=args.eval_rmap_path_RD)
    eval_loader_Rain = get_eval_data(val_in_path=args.eval_in_path_Rain, val_gt_path=args.eval_gt_path_Rain, val_imap_path=args.eval_imap_path_Rain, val_rmap_path=args.eval_rmap_path_Rain)
    eval_loader_L = get_eval_data(val_in_path=args.eval_in_path_L, val_gt_path=args.eval_gt_path_L, val_imap_path=args.eval_imap_path_L, val_rmap_path=args.eval_rmap_path_L)

    optimizerG = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    scheduler = CosineAnnealingWarmRestarts(optimizerG, T_0=args.T_period, T_mult=1)

    # loss function
    loss_char = losses.CharbonnierLoss()
    criterion_depth = losses.depth_loss()

    vgg = models.vgg16(pretrained=False)
    vgg.load_state_dict(torch.load('loss/vgg16-397923af.pth'))
    vgg_model = vgg.features[:16].to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()

    step = 0
    max_psnr_val_RD = args.max_psnr
    max_psnr_val_Rain = args.max_psnr
    max_psnr_val_L = args.max_psnr

    total_loss = 0.0
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss3 = 0.0
    input_PSNR_all = 0
    train_PSNR_all = 0
    iter_nums = 0

    if args.resume_training:
        checkpoint_epoch = args.resume_checkpoint
        checkpoint_path = args.unified_path + args.experiment_name + '/net_epoch_{}.pth'.format(checkpoint_epoch)
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            net.load_state_dict(torch.load(checkpoint_path))
            start_epoch = checkpoint_epoch + 1
        else:
            print(f"No checkpoint found at {checkpoint_path}, training from scratch")
            start_epoch = 0
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.EPOCH):
        scheduler.step(epoch)
        st = time.time()

        epoch_input_PSNR_all = 0.0
        epoch_train_PSNR_all = 0.0
        num_batches = 0

        for i, train_data in enumerate(train_loader, 0):
            data_A, data_B, data_C = train_data
            data_in = torch.cat([data_A[0], data_B[0], data_C[0]], dim=0).to(device)
            label = torch.cat([data_A[1], data_B[1], data_C[1]], dim=0).to(device)
            imap = torch.cat([data_A[2], data_B[2], data_C[2]], dim=0).to(device)
            rmap = torch.cat([data_A[3], data_B[3], data_C[3]], dim=0).to(device)
            names = data_A[4] + data_B[4] + data_C[4]

            batch_size = data_in.size(0)
            target_labels = torch.zeros(batch_size, num_labels).to(device)
            for b in range(batch_size):
                for j, lbl in enumerate(all_labels):
                    if lbl in names[b]:
                        target_labels[b, j] = 1

            if i == 0:
                print("Check data: data.size: {} ,in_GT_mask: {}".format(data_in.size(), label.size()))
            iter_nums += 1
            net.train()
            net.zero_grad()
            optimizerG.zero_grad()
            inputs = Variable(data_in).to(device)
            imaps = Variable(imap).to(device)
            rmaps = Variable(rmap).to(device)
            labels = Variable(label).to(device)

            train_output, classification_logits, l2_regularization, load_balancing_loss = net(inputs, imaps, rmaps, names)

            loss1 = F.smooth_l1_loss(train_output, labels)
            if args.addition_loss == 'VGG':
                loss2 = args.lam_VGG * loss_network(train_output, labels)
            else:
                loss2 = 0.01 * loss1

            if args.depth_loss:
                loss3 = args.lam_DepthLoss * criterion_depth(train_output, labels)
            else:
                loss3 = 0.0

            classification_loss = F.binary_cross_entropy_with_logits(classification_logits, target_labels)
            g_loss = loss1 + loss2 + loss3 + args.lam_classification_loss * classification_loss + args.lam_balance_loss * load_balancing_loss + l2_regularization

            total_loss += g_loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()

            input_PSNR = compute_psnr(inputs, labels)
            train_PSNR = compute_psnr(train_output, labels)

            epoch_input_PSNR_all += input_PSNR
            epoch_train_PSNR_all += train_PSNR
            num_batches += 1

            input_PSNR_all += input_PSNR
            train_PSNR_all += train_PSNR
            g_loss.backward()
            optimizerG.step()

            if (i + 1) % print_frequency == 0 and i > 1:
                writer.add_scalars(exper_name + '/training', {
                    'PSNR_Output': train_PSNR_all / iter_nums,
                    'PSNR_Input': input_PSNR_all / iter_nums,
                }, iter_nums)
                writer.add_scalars(exper_name + '/training', {
                    'total_loss': total_loss / iter_nums,
                    'loss1_char': total_loss1 / iter_nums,
                    'loss2': total_loss2 / iter_nums,
                    'loss3': total_loss3 / iter_nums,
                    'classification_loss': classification_loss.item(),
                    'load_balancing_loss': load_balancing_loss.item()  # 新增
                }, iter_nums)
                print(
                    "epoch:%d,[%d / %d], [lr: %.7f ],[loss:%.5f,loss1:%.5f,loss2:%.5f,loss3:%.5f,clf_loss:%.5f,lb_loss:%.5f,avg_loss:%.5f],[in_PSNR: %.3f, out_PSNR: %.3f],time:%.3f" %
                    (epoch, i + 1, len(train_loader), optimizerG.param_groups[0]["lr"], g_loss.item(), loss1.item(),
                     loss2.item(), loss3.item(), classification_loss.item(), load_balancing_loss.item(), total_loss / iter_nums, input_PSNR, train_PSNR, time.time() - st))
                st = time.time()

        avg_epoch_input_PSNR = epoch_input_PSNR_all / num_batches
        avg_epoch_train_PSNR = epoch_train_PSNR_all / num_batches
        print(
            f"Epoch [{epoch + 1}/{args.EPOCH}] Average Input PSNR: {avg_epoch_input_PSNR:.3f}, Average Output PSNR: {avg_epoch_train_PSNR:.3f}")
        writer.add_scalars(exper_name + '/epoch', {
            'Average_Input_PSNR': avg_epoch_input_PSNR,
            'Average_Output_PSNR': avg_epoch_train_PSNR
        }, epoch + 1)

        if epoch <= 60 and (epoch + 1) % 10 == 0:
            max_psnr_val_RD = test(net=net, eval_loader=eval_loader_RD, epoch=epoch, max_psnr_val=max_psnr_val_RD, Dname='RD')
            max_psnr_val_Rain = test(net=net, eval_loader=eval_loader_Rain, epoch=epoch, max_psnr_val=max_psnr_val_Rain, Dname='HRain')
            max_psnr_val_L = test(net=net, eval_loader=eval_loader_L, epoch=epoch, max_psnr_val=max_psnr_val_L, Dname='L')
        if epoch > 60 and (epoch + 1) % 1 == 0:
            max_psnr_val_RD = test(net=net, eval_loader=eval_loader_RD, epoch=epoch, max_psnr_val=max_psnr_val_RD, Dname='RD')
            max_psnr_val_Rain = test(net=net, eval_loader=eval_loader_Rain, epoch=epoch, max_psnr_val=max_psnr_val_Rain, Dname='HRain')
            max_psnr_val_L = test(net=net, eval_loader=eval_loader_L, epoch=epoch, max_psnr_val=max_psnr_val_L, Dname='L')

        torch.save(net.state_dict(), SAVE_PATH + 'net_epoch_{}.pth'.format(epoch))
