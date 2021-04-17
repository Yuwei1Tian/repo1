import torch
import torch.nn as nn
import torch, math
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from PIL import Image
from torch.autograd import Variable
import imageio
import argparse
import model
import MyLoss
import dataloader
import torchvision.transforms as transforms

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    T_net = model.T_Net().cuda()

    T_net.apply(weights_init)
    if config.load_pretrain == True:
        T_net.load_state_dict(torch.load(config.pretrain_dir))
    train_dataset = dataloader.My_Dataset(config.input_path, config.t_ori_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)

    L_t = MyLoss.loss_t()
    #L_cs = MyLoss.loss_cs()


    optimizer = torch.optim.Adam(T_net.parameters(), lr=config.lr, betas=(0.9, 0.99), weight_decay=config.weight_decay)

    T_net.train()

    for epoch in range(config.num_epochs):
        loss_ave = 0
        for iteration, img_ in enumerate(train_loader):

            img, label = img_
            img = img.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            output_s, output_m, outputs = T_net(img)
            #pool_s = nn.AvgPool2d(4)
            #pool_m = nn.AvgPool2d(2)
            #label_s = pool_s(label)
            #label_m = pool_m(label)

            loss_t = L_t(outputs, label)
            loss_cs = MyLoss.loss_cs(label, outputs)

            # best_loss
            loss = 0.75*loss_t + 0.25*loss_cs
            print(loss.size())
            #

             
            loss.backward()
            torch.nn.utils.clip_grad_norm(T_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            loss_ave = loss_ave + float(loss)
            losses1.append(loss.cpu().data.item())
            if (iteration + 1) % config.display_iter == 0:
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (
                epoch + 1, config.num_epochs, iteration + 1, len(train_dataset) // config.train_batch_size, loss.item()))

            if epoch % 100 == 0:
                temp_t_trained = np.transpose(np.squeeze(outputs[0, :].cpu().detach().numpy()), (0, 1))
                # print(temp_t_trained.shape)
                temp_t_max = np.max((temp_t_trained), 0)

                t_max = np.max((temp_t_max), 0)
                temp_t_trained = temp_t_trained - np.maximum(0, t_max - 1)

                temp_t_trained[temp_t_trained > 1] = 1
                temp_t_trained[temp_t_trained < 0] = 0
                temp_t_trained = (temp_t_trained * 255).astype(np.uint8)

                ##temp!!!!!!!!!!!!!!!!!!!!!!!
                imageio.imwrite('%s%s' % (result_path1, train_dataset.img_list1_[i]),
                                temp_t_trained)  # 单通道，不用120

        loss_val = loss_ave / img.shape[0] * config.train_batch_size
        f = open(txt_file_name, "a+")
        f.write("%d %f\n" % (epoch, loss_val))
        f.close()

        if (epoch >= 20) and (epoch % 20) == 0:
            loss_data = np.loadtxt(txt_file_name)
            plt.plot(loss_data[:, 0], loss_data[:, 1])
            plt.savefig(loss_pic_name)
            print('loss pic saved!')

        if (epoch + 1) % 1000 == 0:
            torch.save(cnn.state_dict(), config.snapshots_folder + "model_t_0414.pth")
            print('model saved')



if __name__ == "__main__":

    result_path = 'result/t_part/'
    result_path1 = 'result/t_part/t_trained_0413/'
    txt_file_name = result_path + "/loss.txt"
    loss_pic_name = result_path + '/loss.jpg'

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--input_path', type=str, default="/home/imglab_hsr/sys_data/water/")
    parser.add_argument('--t_ori_path', type=str, default="/home/imglab_hsr/sys_data/t/")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=200)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/Epoch99.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    if not os.path.exists(result_path1):
        os.mkdir(result_path1)

    train(config)
