from cProfile import label
from os import path
import os
import numpy as np
import time
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import torch.nn as nn
import argparse
import torch.nn.functional as F
from distutils.util import strtobool
from loss_function import PSNRLoss, SSIMLoss, SupConLoss
import torch
from dataloader import MicroExpressionDataset, get_train_loader, get_test_loader
from Model import MicroExpressionNet, HTNet, Vit
# Some of the codes are adapted from STSNet
def reset_weights(m):  # Reset the weights for network to avoid weight leakage
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #             print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def confusionMatrix(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall


def save_reconstruction_comparison(original, reconstructed, epoch, batch_idx, output_dir):
    # 只取第一对图像
    original_img = original[0].cpu().detach().numpy()[0]  # 取batch中的第一个样本
    reconstructed_img = reconstructed.cpu().detach().numpy()[0]
    
    # 创建单行双列的对比图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 原始图像
    orig_img = np.transpose(original_img, (1, 2, 0))
    orig_img = np.clip(orig_img, 0, 1)
    axes[0].imshow(orig_img)
    axes[0].axis('off')
    axes[0].set_title('Original')
    
    # 重建图像
    recon_img = np.transpose(reconstructed_img, (1, 2, 0))
    recon_img = np.clip(recon_img, 0, 1)
    axes[1].imshow(recon_img)
    axes[1].axis('off')
    axes[1].set_title('Reconstructed')
    
    # 调整布局并保存
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'compare_epoch{epoch}_batch{batch_idx}.png'))
    plt.close()

def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}
    # Display recognition result
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except Exception:
        return '', ''

def region_mse_loss(pred, target, region_coords, patch_size=12):

    B, C, H, W = pred.shape
    total_loss = 0.0
    
    for b in range(B):
        for (y, x) in region_coords[b]:
            # 计算区域边界 (防止越界)
            y_start = int(y)
            y_end = min(y_start + patch_size, H)
            x_start = int(x)
            x_end = min(x_start + patch_size, W)
            
            # 提取预测和目标的对应区域
            pred_patch = pred[b, :, y_start:y_end, x_start:x_end]
            target_patch = target[b, :, y_start:y_end, x_start:x_end]
            
            # 计算当前区域的 MSE
            if pred_patch.numel() > 0:  # 确保区域有效
                total_loss += F.mse_loss(pred_patch, target_patch)
    
    # 返回平均损失 (总损失 / (B * 8))
    return total_loss / (B * 8)


def main(config):
    all_accuracy_dict = {}
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    loss_CE = nn.CrossEntropyLoss()
    loss_ssim = SSIMLoss().to(device)
    loss_psnr = PSNRLoss().to(device)
    loss_MSE = nn.MSELoss()
    loss_SC = SupConLoss().to(device)
    weight_dir = config.weight_dir
    if (config.train):
        if not path.exists(weight_dir):
            os.mkdir(weight_dir)

    print('lr=%f, epochs=%d, device=%s\n' % (config.lr, config.epochs, device))

    total_gt = []
    total_pred = []
    best_total_pred = []

    t = time.time()
    dataset = MicroExpressionDataset(config.root_dir, config.csv_path, config.au_path)
    all_subjects = dataset.all_subjects
    # Use 'leave one subject out' 2 evaluate the model
    print(all_subjects)

    for sub in all_subjects:
        print('Subject:', sub)
        weight_path = weight_dir + '/' + sub + '.pth'
        train_loader = get_train_loader(dataset, excluded_subjects=[], batch_size=config.batch_size, img_size=(config.image_size, config.image_size), num_workers=config.num_workers)
        test_loader = get_test_loader(dataset, test_subjects=[sub], img_size=(config.image_size, config.image_size), num_workers=config.num_workers)

        model = MicroExpressionNet(
            num_classes=config.num_classes,
            embed_dim = config.dim,
            img_size = config.image_size,
            patch_size = config.patch_size,
            onset_depth = config.onset_depth,
            flow_depth = config.flow_depth,
            heads = config.heads,
            dec_depth = config.dec_depth,
            dim_feedforward = config.dim_feedforward,
            dropout = config.dropout,
        )
        # model = HTNet(
        #     image_size=28,
        #     patch_size=7,
        #     dim=256,  # 256,--96, 56-, 192
        #     heads=3,  # 3 ---- , 6-
        #     num_hierarchies=3,  # 3----number of hierarchies
        #     block_repeats=(2, 2, 10),#(2, 2, 8),------
        #     # the number of transformer blocks at each heirarchy, starting from the bottom(2,2,20) -
        #     num_classes=3
        # )
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs')
            model = nn.DataParallel(model)
        model = model.to(device)

        if(config.train):
            print('train')
        else:
            state_dict = torch.load(weight_path)
            print(state_dict)
            if torch.cuda.device_count() > 1:
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
        optimizer = torch.optim.Adam(model.parameters(),lr=config.lr)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # optimizer = torch.optim.SGD(model.parameters(),lr=config.lr)

        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        for epoch in range(1, config.epochs + 1):
            if (config.train):
                # Training
                
                model.train()
                train_loss = 0.0
                Rec = 0.0
                Cls = 0.0
                num_train_correct = 0
                num_train_examples = 0
                batch_all = len(train_loader)

                for batch, (onset, apex, optimal, labels, AUs) in enumerate(train_loader):    
                    optimizer.zero_grad()
                    onset = onset.to(device)
                    apex = apex.to(device)
                    optimal = optimal.to(device)
                    labels = labels.to(device)
                    # (y1, y2), (rec1, rec2) = model(onset, optimal)
                    (rec1, rec2), (feat1, feat2) = model(onset, optimal)
                    # label_con = torch.concat((labels, labels))
                    if batch == 0:
                        save_reconstruction_comparison(apex, rec1, epoch, batch, './output_recimg')
                    apex1, apex2  = apex[:, 0], apex[:, 1]
                    AU1, AU2 = AUs[:, 0], AUs[:, 1]
                    # print('AU1:', AU1)
                    # print('AU2:', AU2)
                    # Rec_loss = loss_ssim(rec1, apex1) + loss_ssim(rec2, apex2) + + 0.01 * loss_psnr(rec1, apex1) + 0.01 * loss_psnr(rec2, apex2)
                    Rec_loss = loss_MSE(rec1, apex1) + loss_MSE(rec2, apex2)
                    # Cls_loss = loss_CE(y1, labels) + loss_CE(y2, labels)
                    # loss = Rec_loss + Cls_loss
                    # yhat = model(onset, optimal)
                    # Cls_loss = loss_CE(yhat, labels)
                    loss = region_mse_loss(rec1, apex1, AU1) + region_mse_loss(rec2, apex2, AU2)
                    loss.backward()
                    optimizer.step()
                    # lr_scheduler.step()

                    train_loss += loss.data.item() * onset.size(1)
                    # Rec += Rec_loss.data.item() * onset.size(1)
                    # Cls += Cls_loss.data.item() * onset.size(1)
                    if batch % 5 == 0:
                        print(f'Epoch: {epoch}/{config.epochs}, Batch Num: {batch}/{batch_all}, Train Loss: {loss.data.item():.5f}, Rec Loss: {Rec_loss.data.item():.5f}')
                        # print(f'Epoch: {epoch}/{config.epochs}, Batch Num: {batch}/{batch_all}, Train Loss: {loss.data.item():.5f}, Rec Loss: {Rec_loss.data.item():.5f}, Cls Loss: {Cls_loss.data.item():.5f}')
                    
                    # num_train_correct += (torch.max(y1, 1)[1] == labels).sum().item() + (torch.max(y2, 1)[1] == labels).sum().item()
                    # num_train_examples += labels.shape[0]
                # train_acc = num_train_correct / num_train_examples
                train_loss = train_loss / len(train_loader.dataset)
                Rec = Rec / len(train_loader.dataset)
                Cls = Cls / len(train_loader.dataset)
                weight_path = weight_dir + '/pre_train_hier_recon' + '.pth'
                torch.save(model.module.state_dict(), weight_path)

        break
            # Testing

            # if torch.cuda.device_count() > 1:
            #     model = model.module
            # model.eval()
            # val_loss = 0.0
            # num_val_correct = 0
            # num_val_examples = 0
            # for onset, optimal, labels in test_loader:
            #     onset_pair = onset.to(device)
            #     optimal_pair = optimal.to(device)
            #     labels = labels.to(device)
            #     yhat = model(onset_pair, optimal_pair, eval=True)
            #     # yhat = model(onset, optimal, training=False)
            #     loss = loss_CE(yhat, labels)
            #     val_loss += loss.data.item() * optimal.size(0)
            #     num_val_correct += (torch.max(yhat, 1)[1] == labels).sum().item()
            #     num_val_examples += labels.shape[0]

            # val_acc = num_val_correct / num_val_examples
            # val_loss = val_loss / len(test_loader.dataset)
            # #### best result
            # temp_best_each_subject_pred = []
            # if best_accuracy_for_each_subject <= val_acc:
            #     best_accuracy_for_each_subject = val_acc
            #     temp_best_each_subject_pred.extend(torch.max(yhat, 1)[1].tolist())
            #     best_each_subject_pred = temp_best_each_subject_pred
            #     # Save Weights
            #     if (config.train):
            #         # if torch.cuda.device_count() > 1:
            #         #     torch.save(model.module.state_dict(), weight_path)
            #         # else:
            #         torch.save(model.state_dict(), weight_path)
            # print(f'Sub: {sub}, Epoch: {epoch}/{config.epochs}, Train Loss: {train_loss:.5f}, Cls Loss: {Cls:.5f}, Rec Loss: {Rec:.5f}, Val Loss: {val_loss:.5f}, Val Acc: {val_acc:.5f}')
            # print('Ground Truth:', labels)
            # print(' Prediction :', torch.max(yhat, 1)[1])

            # if torch.cuda.device_count() > 1:
            #     model = nn.DataParallel(model)
            # if val_acc == 1:
            #     break



        # For UF1 and UAR computation
    #     print('Best Predicted:', best_each_subject_pred)
    #     accuracydict = {}
    #     accuracydict['pred'] = best_each_subject_pred
    #     accuracydict['truth'] = labels.tolist()
    #     all_accuracy_dict[sub] = accuracydict

    #     print(' Ground Truth :', labels.tolist())
    #     print('Evaluation until this subject: ')
    #     total_pred.extend(torch.max(yhat, 1)[1].tolist())
    #     total_gt.extend(labels.tolist())
    #     best_total_pred.extend(best_each_subject_pred)
    #     UF1, UAR = recognition_evaluation(total_gt, total_pred, show=True)
    #     best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred, show=True)
    #     print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))

    # print('Final Evaluation: ')
    # UF1, UAR = recognition_evaluation(total_gt, total_pred)
    # print(np.shape(total_gt))
    # print('Total Time Taken:', time.time() - t)
    # print(all_accuracy_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=strtobool, default=True, 
                       help='Train mode (True) or evaluation mode (False)')
    parser.add_argument('--lr', type=float, default=1e-6, 
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=500, 
                       help='Number of training epochs')
    parser.add_argument('--gpu', type=str, default='0,1,2,3', 
                    help='list of GPU(s)')

    # Model
    parser.add_argument('--image_size', type=int, default=112, 
                       help='Input image size')
    parser.add_argument('--patch_size', type=int, default=14, 
                       help='Patch size for transformer')
    parser.add_argument('--dim', type=int, default=256, 
                       help='Dimension of transformer')
    parser.add_argument('--heads', type=int, default=4, 
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, 
                       help='Dropout rate')
    parser.add_argument('--dim_feedforward', type=int, default=512,
                       help='Dimension of feedforward network')
    parser.add_argument('--onset_depth', type=int, default=1,
                       help='Depth of onset network')
    parser.add_argument('--flow_depth', type=int, default=1,
                       help='Depth of flow network')
    parser.add_argument('--dec_depth', type=int, default=3,
                       help='Depth of decoder network')
    parser.add_argument('--num_classes', type=int, default=3,
                       help='Number of output classes')
    parser.add_argument('--num_workers', type=int, default=16, 
                       help='Number of workers')

    #Path stuff
    parser.add_argument('--root_dir', type=str, default='./datasets/dataset_combined',
                   help='Root directory of the dataset')
    parser.add_argument('--csv_path', type=str, default='./datasets/combined_datasets_class.csv',
                    help='Path to the CSV file containing class labels')
    parser.add_argument('--au_path', type=str, default='./datasets/marked.csv',
                    help='Path to the CSV file containing AUs')
    parser.add_argument('--weight_dir', type=str, default='./weights_3',
                    help='Directory to save/load model weights')

    config = parser.parse_args()
    main(config)
