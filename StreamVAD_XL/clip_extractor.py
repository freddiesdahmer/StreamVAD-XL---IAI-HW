import clip
import torch
import torch.nn.functional as F


class CLIPFeatureExtractor:
    def __init__(self, model_name="ViT-B/16", device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_clip_feature(self, pil_images):
        """
        接收 PIL Image 列表，用 CLIP 官方 preprocess 提取特征
        Args:
            pil_images: List[PIL.Image]，长度 = clip_length
        Returns:
            features: float32 tensor, 形状 (1, T, 512)，已 L2 归一化
        """
        features = []
        for pil_img in pil_images:
            image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            feat = self.model.encode_image(image_input)   # (1, 512), float16
            feat = feat.float()                          
            feat = F.normalize(feat, p=2, dim=-1)         # L2 归一化（与训练一致）
            features.append(feat.squeeze(0))               # (512,)

        features = torch.stack(features, dim=0).unsqueeze(0)  # (1, T, 512)
        return features
