import torch
from torch.utils.data import DataLoader
from model import StreamVADWithXL
from dataset import UCFDataset_Whole_test_train, XDDataset_Whole_test_train
import argparse
from sklearn.metrics import roc_auc_score
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="StreamVAD+XL Testing")
    parser.add_argument('--dataset', type=str, default='ucf', choices=['ucf', 'xd'])
    parser.add_argument('--test_list', type=str, required=True, help='测试集CSV路径')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--mem_len', type=int, default=32)
    parser.add_argument('--chunk_size', type=int, default=16)
    parser.add_argument('--attnwindow', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def test(args):
    print(f"测试数据集: {args.dataset}")
    print(f"加载模型: {args.model_path}")
    
    model = StreamVADWithXL(mem_len=args.mem_len).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()
    
    # 加载测试集
    if args.dataset == 'ucf':
        test_dataset = UCFDataset_Whole_test_train(
            args.test_list, test_mode=True, normal=False, attnwindow=args.attnwindow
        )
    else:
        test_dataset = XDDataset_Whole_test_train(
            args.test_list, test_mode=True, attnwindow=args.attnwindow
        )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    all_video_scores = []
    all_video_labels = []
    
    with torch.no_grad():
        for batch_idx, (clip_feature, clip_label, clip_length, video_name) in enumerate(test_loader):
            # 标签转换：Normal=0，其他=1
            labels = torch.tensor([0 if l == 'Normal' else 1 for l in clip_label])
            
            clip_feature = clip_feature.to(args.device).float()
            clip_feature = torch.nn.functional.normalize(clip_feature, p=2, dim=-1)
            
            B, Total_T, D = clip_feature.shape
            memories = None
            video_scores = []
            
            for t in range(0, Total_T, args.chunk_size):
                chunk = clip_feature[:, t:t+args.chunk_size, :]
                if chunk.size(1) == 0:
                    continue
                
                scores, memories, _ = model(chunk, memories)
                scores = torch.sigmoid(scores)  # logits 转概率
                video_scores.append(scores.cpu())
            
            # 拼接整个视频的分数
            video_scores = torch.cat(video_scores, dim=1)  # (B, Total_T)
            
            # 视频级预测：取 top-k 均值（和训练 MIL 一致）
            topk_scores, _ = torch.topk(video_scores, k=min(10, video_scores.size(1)), dim=1)
            video_pred = torch.mean(topk_scores, dim=1).numpy()
            
            all_video_scores.extend(video_pred.tolist())
            all_video_labels.extend(labels.numpy().tolist())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"已处理 {batch_idx+1}/{len(test_loader)} 个样本")
    
    # 计算视频级 AUC
    auc = roc_auc_score(all_video_labels, all_video_scores)
    print(f"\n测试完成！视频级 AUC: {auc:.4f}")

if __name__ == "__main__":
    args = parse_args()
    test(args)