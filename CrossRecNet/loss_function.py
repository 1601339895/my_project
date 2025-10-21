import torch
import torch.nn.functional as F
import torch.nn

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, reduction='mean'):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.reduction = reduction

    def gaussian(self, window_size, sigma):
        gauss = torch.exp(-(torch.arange(window_size) - window_size//2)**2 / (2*sigma**2))
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, img1, img2):
        channel = img1.shape[1]
        window = self.create_window(self.window_size, channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.reduction == 'mean':
            return 1 - ssim_map.mean()  # SSIM Loss = 1 - SSIM
        elif self.reduction == 'none':
            return 1 - ssim_map
        else:
            raise ValueError("Invalid reduction option. Use 'mean' or 'none'.")


class PSNRLoss(torch.nn.Module):
    def __init__(self, max_val=1.0, reduction='mean'):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val
        self.reduction = reduction

    def forward(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        
        if self.reduction == 'mean':
            return -psnr.mean()
        elif self.reduction == 'none':
            return -psnr
        else:
            raise ValueError("Invalid reduction option. Use 'mean' or 'none'.")



class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)
        similarities = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask * (1 - torch.eye(features.size(0), device=device))
        
        # 数值稳定计算
        logits_max = similarities.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(similarities - logits_max)
        log_prob = similarities - logits_max - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # 只计算有正样本的样本
        valid_samples = mask.sum(dim=1) > 0
        loss = -(mask * log_prob).sum(dim=1)[valid_samples] / mask.sum(dim=1)[valid_samples]
        
        return loss.mean() if loss.numel() > 0 else torch.tensor(0., device=device)