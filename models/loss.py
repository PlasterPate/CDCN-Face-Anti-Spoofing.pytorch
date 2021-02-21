import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def contrast_depth_conv(input, device):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''

    kernel_filter_list = [
                         [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                         [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
                         [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                         ]
    
    kernel_filter = np.array(kernel_filter_list, np.float32)
    print(kernel_filter.shape)
    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().to(device)
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)
    print(kernel_filter.shape)

    print(input.shape)
    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    print(input.shape)
    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    print(contrast_depth.shape)
    
    return contrast_depth


class ContrastDepthLoss(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self, device):
        super(ContrastDepthLoss, self).__init__()
        self.device = device

    def forward(self, out, label): 
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out, device=self.device)
        contrast_label = contrast_depth_conv(label, device=self.device)

        # image1 = contrast_out.cpu().detach().numpy()[0]
        # image1 = np.transpose(image1, (1, 2, 0))
        # image2 = contrast_label.cpu().detach().numpy()[0]
        # image2 = np.transpose(image2, (1, 2, 0))
        #
        # plt.figure()
        # f, ax = plt.subplots(1, 8)
        #
        # ax[0].imshow(image1[:, :, 0])
        # ax[1].imshow(image1[:, :, 1])
        # ax[2].imshow(image1[:, :, 2])
        # ax[3].imshow(image1[:, :, 3])
        # ax[4].imshow(image1[:, :, 4])
        # ax[5].imshow(image1[:, :, 5])
        # ax[6].imshow(image1[:, :, 6])
        # ax[7].imshow(image1[:, :, 7])

        our = np.unique(out.cpu().detach().numpy())
        # print(our)
        print(len(our))
        print(np.unique(label.cpu().detach().numpy()))

        # print(np.unique(label.cpu().detach().numpy()))
        # print(contrast_label.cpu().detach().numpy()[0])
        print("#######################################")

        # ax[1, 0].imshow(image2[:, :, 0])
        # ax[1, 1].imshow(image2[:, :, 1])
        # ax[1, 2].imshow(image2[:, :, 2])
        # ax[1, 3].imshow(image2[:, :, 3])
        # ax[1, 4].imshow(image2[:, :, 4])
        # ax[1, 5].imshow(image2[:, :, 5])
        # ax[1, 6].imshow(image2[:, :, 6])
        # ax[1, 7].imshow(image2[:, :, 7])

        # plt.show()

        criterion_MSE = nn.MSELoss()
    
        loss = criterion_MSE(contrast_out, contrast_label)
    
        return loss


class DepthLoss(nn.Module):
    def __init__(self, device):
        super(DepthLoss, self).__init__()
        self.criterion_absolute_loss = nn.MSELoss()
        self.criterion_contrastive_loss = ContrastDepthLoss(device=device)

    def forward(self, predicted_depth_map, gt_depth_map):
        absolute_loss = self.criterion_absolute_loss(predicted_depth_map, gt_depth_map)
        contrastive_loss = self.criterion_contrastive_loss(predicted_depth_map, gt_depth_map)
        return absolute_loss + contrastive_loss
