# -*- encoding:utf-8 -*-
from re import A
from tkinter import NO
import torch
from torch import nn
import sys
import torch.optim as optim
import numpy as np
import time
import pickle as pkl
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
# from utils.util import plot_loss
from utils.eval_metrics import calculate_ir_metrics, calculate_top_accuracy,calculate_top_k_accuracy,calculate_top_k_accuracy_with_feature
from model import Model
from config import DEVICE, get_args, get_config

# torch.autograd.set_detect_anomaly(True)


class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None,
                 pretrained_emb=None):
        self.args = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.info_nce = self.args.info_nce
        self.option = self.args.option

        self.dataset_name = self.args.dataset

        self.test_loader = test_loader

        self.is_train = is_train
        self.model = model


        self.update_batch = self.args.update_batch

        # initialize the model
        if model is None:
            self.model = Model(self.args)

        if torch.cuda.is_available():
            self.model = self.model.to(DEVICE)
        else:
            self.device = torch.device("cpu")

        # optimizer
        self.optimizer = {}

        if self.is_train:
            main_param = []
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    print(name)
                    main_param.append(p)

        no_decay = ['bias', 'LayerNorm.weight','LayerNorm.bias']
        self.optimizer_main_group = [
            {'params': main_param, 'weight_decay': self.args.weight_decay_main, 'lr': self.args.lr_main}
        ]

        self.optimizer_main = getattr(torch.optim, self.args.optim)(
            self.optimizer_main_group
        )
        self.scheduler_main = ReduceLROnPlateau(self.optimizer_main, mode='min', patience=self.args.when, factor=0.5,
                                                verbose=True)

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):

        def train(model, optimizer):
            epoch_loss = 0.0

            model.train()
            num_batches = self.args.n_train // self.args.batch_size
            # print(self.train_loader)

            for i_batch, batch_data in enumerate(self.train_loader):
                texts,ids,audio,alens,labels = batch_data

                # print('audio.shape',audio.shape)
                # print('texts.shape',texts.shape)
                # print('alens.shape',alens.shape)

                model.zero_grad()

                if torch.cuda.is_available():
                    with torch.cuda.device(0):
                        texts, audio, alens, labels = texts.to(DEVICE),audio.to(DEVICE),alens.to(DEVICE),labels.to(DEVICE)
                        loss, logits_per_acoustic, logits_per_text = self.model\
                            (texts,audio,alens,labels)
                        # print('labels',labels)
                        #             # torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                else:
                        texts, audio, alens = texts.to(DEVICE),audio.to(DEVICE),alens.to(DEVICE)
                        loss, logits_per_acoustic, logits_per_text = self.model \
                            (texts, audio, alens,labels)

                loss = loss.requires_grad_(True)
                loss.backward()
                self.optimizer_main.step()
                epoch_loss = epoch_loss + loss

                return epoch_loss/num_batches

        def evaluate(model, loader, n_loader=None, test=False):
            model.eval()
            # loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            if test:
                test_result = []
                with torch.no_grad():
                    for i, batch in enumerate(loader):
                        ids, texts, audio, alens, labels = batch
                        if torch.cuda.is_available():
                            with torch.cuda.device(0):
                                texts, audio, alens, labels, = texts.to(DEVICE), audio.to(DEVICE), alens.to(DEVICE), labels.to(DEVICE)
                                top_probs,top_labels = self.model(texts, audio, alens, labels, training=False)
                        else:
                            top_probs, top_labels = self.model(texts, audio, alens, labels, training=False)
                        # print('top_labels_shape:{}'.format(top_labels.shape))
                        test_result.extend(top_labels.cpu().tolist())
                        # test_result.append((top_probs,top_labels))
                return test_result
            else:
                epoch_loss = 0.0
                num_batches = self.args.n_valid // self.args.batch_size
                with torch.no_grad():
                    for i, batch in enumerate(loader):
                        ids,texts,audio,alens,labels = batch
                        if torch.cuda.is_available():
                            with torch.cuda.device(0):
                                texts, audio, alens, labels = texts.to(DEVICE), audio.to(DEVICE), alens.to(DEVICE), labels.to(DEVICE)
                                loss, logits_per_text, logits_per_text = self.model(texts,audio,alens,labels,training=True)

                        else:
                            loss, logits_per_text, logits_per_text = self.model(texts,audio,alens,labels) ###  多分类问题，但其实根据tag是个多标签多分类问题
                        epoch_loss = epoch_loss + loss

                return epoch_loss/num_batches
    
        for epoch in range(1, self.args.num_epochs + 1):
            start = time.time()

            self.epoch = epoch

            # minimize all losses left
            train_loss = train(self.model, self.optimizer_main)
            valid_loss = evaluate(self.model, self.dev_loader, test=False)
            test_results = evaluate(self.model, self.dev_loader, test=True)
            acc_top = calculate_top_accuracy(test_results)
            print('epoch:{}, train_loss:{:5.4f},val_loss:{:5.4f}, acc_top:{}'.format(epoch,train_loss,valid_loss,acc_top))
            # if epoch%2 == 0:
            #     test_results = evaluate(self.model, self.dev_loader, test=True)
            #     print('test_result:',test_results)


        sys.stdout.flush()


        sys.stdout.flush()

    def train_and_eval_with_pretrain(self):
        # model = self.model
        # optimizer_main = self.optimizer_main
        #
        # scheduler_main = self.scheduler_main

        def train(model, optimizer):
            epoch_loss = 0.0

            model.train()
            num_batches = self.args.n_train // self.args.batch_size
            # print(self.train_loader)

            for i_batch, batch_data in enumerate(self.train_loader):
                text, t5_att_mask, audio, category, labels = batch_data
                ## category means vote score for HapticGen data
                ## category means sensory, emotion and association (3-class) for collected data

                # print('audio.shape',audio.shape)
                # print('texts.shape',texts.shape)
                # print('alens.shape',alens.shape)

                model.zero_grad()

                if torch.cuda.is_available():
                    with torch.cuda.device(0):
                        text, audio, labels = text.to(DEVICE),audio.to(DEVICE),labels.to(DEVICE)
                        if self.option=='match' and self.dataset_name=='gendata':
                            category = category.to(DEVICE)
                        loss = self.model\
                            (text,audio,label=labels,category=category, mode='train')
                        # print('labels',labels)
                        #             # torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                else:
                        text, audio, labels = text.to(DEVICE), audio.to(DEVICE), labels.to(DEVICE)
                        if category != None:
                            category = category.to(DEVICE)
                        loss = self.model \
                            (text, audio, label=labels, category=category, mode='train')

                loss = loss.requires_grad_(True)
                loss.backward()
                self.optimizer_main.step()
                epoch_loss = epoch_loss + loss

                return epoch_loss/num_batches, epoch_loss

        def evaluate(model, loader, n_loader=None, mode='valid'):
            model.eval()
            # loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            total_acc = 0
            total_pk, total_pkpct, total_rk, total_rkpct, total_ap, total_rprec = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            total_ndcg = 0.0
            # num_batches = n_loader // self.args.batch_size
            num_batches = 0

            if mode == 'test':
                test_result = []
                acoustic_result = []
                text_result = []
                with torch.no_grad():
                    for i, batch_data in enumerate(loader):
                        text, t5_att_mask, audio, category, labels = batch_data
                        num_batches = num_batches + 1
                        if torch.cuda.is_available():
                            with torch.cuda.device(0):
                                text, audio, labels = text.to(DEVICE),audio.to(DEVICE),labels.to(DEVICE)
                                if self.option=='match' and self.dataset_name=='gendata':
                                    category = category.to(DEVICE)
                                acoustic_feature, text_feature, pred_probs, top_probs, top_labels = self.model\
                                    (text,audio,label=labels, category=category, mode='test')
                        else:
                            acoustic_feature, text_feature, pred_probs, top_probs, top_labels = self.model(text, audio, label=labels, mode='test')
                        # print('top_labels_shape:{}'.format(top_labels.shape))
                        tpk = min(acoustic_feature.shape[0], 10)

                        ## extract match
                        topk_accuracy = calculate_top_k_accuracy_with_feature(acoustic_feature, text_feature, k=tpk)
                        total_acc = total_acc + topk_accuracy
                        ## all metrics, similar match
                        apk, ark, aap,ndcg = calculate_ir_metrics(pred_probs,top_labels,labels,k=tpk)
                        
                        total_pk = total_pk + apk
                        total_rk = total_rk + ark
                        total_ap = total_ap + aap
                        total_ndcg = total_ndcg + ndcg

                        acoustic_result.extend(acoustic_feature)
                        text_result.extend(text_feature)
                        # test_result.append((top_probs,top_labels))
                    acoustic_pred = acoustic_result[0] 
                    for acoustic in acoustic_result:
                        acoustic_pred = torch.vstack((acoustic_pred, acoustic))

                    text_pred = text_result[0] 
                    for text in text_result:
                        text_pred = torch.vstack((text_pred, text))
                    # print('acoustic_pred.shape:{}, text_pred.shape:{}'.format(acoustic_pred.shape, text_pred.shape))
                    # ###visualization
                    # # if self.args.set_visualization:
                    # #     visualization(emotions, events)
                    # probs = (100.0 * acoustic_pred @ text_pred.T).softmax(dim=-1)
                    # print('text_probs.shape:{}'.format(probs.shape))
                    # targets = torch.range(0,probs.shape[0]-1)

                    # top_probs, top_labels = emotion_probs.cpu().topk(topk, dim=-1)

                metric = [total_pk, total_rk, total_ap, total_ndcg]
                avg_metric = [ele/num_batches for ele in metric]

                return acoustic_pred, text_pred, total_acc/num_batches, avg_metric 
            elif mode == 'valid':
                epoch_loss = 0.0
                avg_metric = 0.0
                with torch.no_grad():
                    for i, batch_data in enumerate(loader):
                        text, t5_att_mask, audio, category, labels = batch_data
                        num_batches = num_batches + 1
                        if torch.cuda.is_available():
                            with torch.cuda.device(0):
                                texts, audio,labels = text.to(DEVICE), audio.to(DEVICE), labels.to(DEVICE)
                                if self.option=='match' and self.dataset_name=='gendata':
                                    category = category.to(DEVICE)
                                loss, pred_probs, top_probs, top_labels = self.model(texts,audio,label=labels,category=category, mode='valid')

                        else:
                            loss, pred_probs, top_probs, top_labels = self.model(text,audio,label=labels, mode='valid') ###  多分类问题，但其实根据tag是个多标签多分类问题

                        epoch_acc = calculate_top_accuracy(top_labels,batch_size=top_labels.shape[0])

                        epoch_loss = epoch_loss + loss
                
                # metric = [total_pk, total_pkpct, total_rk, total_rkpct, total_ap, total_rprec]
                # avg_metric = [ele/num_batches for ele in metric]

                return epoch_loss/num_batches, epoch_loss, total_acc/num_batches, avg_metric
        best_valid_acc = 0
        best_acoustic_pred = None
        best_text_pred = None
        train_loss_dict, valid_loss_dict = {}, {}
        for epoch in range(1, self.args.num_epochs + 1):
            start = time.time()

            self.epoch = epoch

            # minimize all losses left
            avg_train_loss, train_loss = train(self.model, self.optimizer_main)
            avg_valid_loss, valid_loss, valid_acc, valid_avg_metric = evaluate(self.model, self.dev_loader, n_loader=self.args.n_valid, mode='valid')
            # avg_test_loss, test_loss, test_acc_1, test_avg_metric_1 = evaluate(self.model, self.test_loader, n_loader=self.args.n_test, mode='valid')

            train_loss_dict[epoch] = avg_train_loss.cpu().detach().numpy()
            valid_loss_dict[epoch]= avg_valid_loss.cpu().detach().numpy()

            acoustic_pred, text_pred, test_acc_2, test_avg_metric_2 = evaluate(self.model, self.test_loader, n_loader=self.args.n_test, mode='test')

            # acc_top = calculate_top_accuracy(test_results)
            # if epoch%2 == 0:
            #     test_results = evaluate(self.model, self.dev_loader, test=True)
            #     print('test_result:',test_results)
            print('epoch:{}, avg_train_loss:{:5.4f}, train_loss:{:5.4f}, avg_val_loss:{:5.4f}, val_loss: {:5.4f}, valid_acc:{}, \
                  test_acc_2:{}'.format(epoch, avg_train_loss, train_loss, avg_valid_loss, valid_loss, valid_acc, test_acc_2))
            print('IR metric, p:{}, r:{}, map:{}, ndcg:{}'.format(test_avg_metric_2[0], test_avg_metric_2[1], \
                test_avg_metric_2[2],test_avg_metric_2[3]))

            if valid_acc > best_valid_acc:
                # torch.save({
                # 'epoch': epoch,
                # 'model_state_dict': self.model.state_dict(),
                # 'loss': valid_loss
                # }, r'./models/best_model.pt')
                best_acoustic_pred = acoustic_pred
                best_text_pred = text_pred
            # print('epoch:{}, train_loss:{:5.4f}, avg_train_loss: {:5.4f}, val_loss:{:5.4f}, avg_valid_loss:{:5.4f}'.format(epoch,train_loss,avg_train_loss, valid_loss, avg_valid_loss))

        # checkpoint = torch.load(r'./models/best_model.pt')
        # self.model.load_state_dict(checkpoint['model_state_dict'])

        # Save the training loss values
        
        # with open('./loss/train_loss_2_four_block.pkl', 'wb') as file:
        #     pkl.dump(train_loss_dict, file)

 
        # Save the validation loss values
        # with open('./loss/val_loss_2_four_block', 'wb') as file:
        #     pkl.dump(valid_loss_dict, file)
            
        # tpk = min(best_acoustic_pred.shape[0], 10)

        # topk_accuracy = calculate_top_k_accuracy_with_feature(best_acoustic_pred, best_text_pred, k=10)
        # print('topk_accuracy:{}'.format(topk_accuracy))
        # P, R, AP = calculate_ir_metrics(best_acoustic_pred, best_text_pred, k=10)
        
        ##plot loss curves 
        # plot_loss('./train_loss.pkl', './val_loss.pkl', epoch_num=self.args.num_epochs)
        sys.stdout.flush()

        
            
        







