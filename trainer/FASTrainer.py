import os
import numpy as np
from random import randint
import torch
import torchvision
from trainer.base import BaseTrainer
from utils.meters import AvgMeter
from utils.eval import add_visualization_to_tensorboard, predict, calc_accuracy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

TARGET_NAMES = ['Fake', 'Real']


class FASTrainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, criterion, lr_scheduler, device, trainloader, valloader, writer):
        super(FASTrainer, self).__init__(cfg, network, optimizer, criterion, lr_scheduler, device, trainloader,
                                         valloader, writer)

        self.network = self.network.to(device)

        self.train_loss_metric = AvgMeter(writer=writer, name='Loss/train', num_iter_per_epoch=len(self.trainloader),
                                          per_iter_vis=True)
        self.train_acc_metric = AvgMeter(writer=writer, name='Accuracy/train', num_iter_per_epoch=len(self.trainloader),
                                         per_iter_vis=True)

        self.val_loss_metric = AvgMeter(writer=writer, name='Loss/val', num_iter_per_epoch=len(self.valloader))
        self.val_acc_metric = AvgMeter(writer=writer, name='Accuracy/val', num_iter_per_epoch=len(self.valloader))

        self.test_loss_metric = AvgMeter(writer=writer, name='Loss/test', num_iter_per_epoch=len(self.valloader))
        self.test_acc_metric = AvgMeter(writer=writer, name='Accuracy/test', num_iter_per_epoch=len(self.valloader))

    def load_model(self):
        saved_name = os.path.join(self.cfg['output_dir'],
                                  '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))
        state = torch.load(saved_name)

        self.optimizer.load_state_dict(state['optimizer'])
        self.network.load_state_dict(state['state_dict'])

    def save_model(self, epoch):
        if not os.path.exists(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])

        saved_name = os.path.join(self.cfg['output_dir'],
                                  '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))

        state = {
            'epoch': epoch,
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(state, saved_name)

    def train_one_epoch(self, epoch):

        self.network.train()
        self.train_loss_metric.reset(epoch)
        self.train_acc_metric.reset(epoch)

        y_pred = np.array([])
        y_true = np.array([])

        for i, (img, depth_map, label) in enumerate(self.trainloader):

            # image = img.cpu().numpy()[0]
            # image = np.transpose(image, (1, 2, 0))
            # plt.imshow(image)
            # plt.show()

            img, depth_map, label = img.to(self.device), depth_map.to(self.device), label.to(self.device)
            net_depth_map, _, _, _, _, _ = self.network(img)
            self.optimizer.zero_grad()
            loss = self.criterion(net_depth_map, depth_map)
            loss.backward()
            self.optimizer.step()

            preds, _ = predict(net_depth_map)
            targets, _ = predict(depth_map)

            y_pred = np.append(y_pred, preds.to('cpu').numpy())
            y_true = np.append(y_true, label.to('cpu').numpy())

            accuracy = calc_accuracy(preds, targets)

            # Update metrics
            self.train_loss_metric.update(loss.item())
            self.train_acc_metric.update(accuracy)

            # print('Epoch: {}, iter: {}, loss: {}, acc: {}'.format(epoch, epoch * len(self.trainloader) + i,
            #                                                       self.train_loss_metric.avg,
            #                                                       self.train_acc_metric.avg))
        print(f'Epoch: {epoch}')
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        cr = classification_report(y_true, y_pred, target_names=TARGET_NAMES)
        print(cr)


    def train(self):
        # self.load_model()
        for epoch in range(self.cfg['train']['num_epochs']):
            self.train_one_epoch(epoch)
            # epoch_acc = self.validate(epoch)
            # if epoch_acc > self.best_val_acc:
            #     self.best_val_acc = epoch_acc
            self.save_model(epoch)

    def validate(self, epoch):
        self.network.eval()
        self.val_loss_metric.reset(epoch)
        self.val_acc_metric.reset(epoch)

        seed = randint(0, len(self.valloader) - 1)
        with torch.no_grad():
            for i, (img, depth_map, label) in enumerate(self.valloader):
                img, depth_map, label = img.to(self.device), depth_map.to(self.device), label.to(self.device)
                net_depth_map, _, _, _, _, _ = self.network(img)
                loss = self.criterion(net_depth_map, depth_map)

                preds, score = predict(net_depth_map)
                targets, _ = predict(depth_map)

                accuracy = calc_accuracy(preds, targets)

                # Update metrics
                self.val_loss_metric.update(loss.item())
                self.val_acc_metric.update(accuracy)

                if i == seed:
                    add_visualization_to_tensorboard(self.cfg, epoch, img, preds, targets, score, self.writer)

            return self.val_acc_metric.avg

    def test(self):
        self.load_model()

        self.network.eval()
        self.test_loss_metric.reset(0)
        self.test_acc_metric.reset(0)

        y_pred = np.array([])
        y_true = np.array([])

        with torch.no_grad():
            for i, (img, depth_map, label) in enumerate(self.valloader):
                if i % 99 == 0:
                    plt.imshow(img.numpy()[0, 0, :, :])
                img, depth_map, label = img.to(self.device), depth_map.to(self.device), label.to(self.device)
                net_depth_map, _, _, _, _, _ = self.network(img)
                loss = self.criterion(net_depth_map, depth_map)

                preds, score = predict(net_depth_map)
                targets, _ = predict(depth_map)
                y_pred = np.append(y_pred, preds.to('cpu').numpy())
                y_true = np.append(y_true, label.to('cpu').numpy())
                accuracy = calc_accuracy(preds, targets)

                # Update metrics
                self.test_loss_metric.update(loss.item())
                self.test_acc_metric.update(accuracy)

            # print('loss: {}, acc: {}'.format(self.test_loss_metric.avg, self.test_acc_metric.avg))
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        cr = classification_report(y_true, y_pred, target_names=TARGET_NAMES)
        print(cr)

        return y_pred
