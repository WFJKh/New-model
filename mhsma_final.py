# -*- coding: utf-8 -*-
"""
Final runnable version with:
- Best CNN checkpoints saving
- Best DFE pipeline saving (joblib)
- 5-fold cross-validation with mean ± sd
- Reproduce and inference helpers

Run:
    python run_smids_cv.py \
        --data_path SMIDS-others \
        --save_dir makale_sonuclari_yizhi3 \
        --epochs 60 \
        --kfolds 5 \
        --batch_size 16 \
        --do_gradcam 1 \
        --do_tsne 1

Author: MONSTER (modified for full reproducibility and CV)
"""

import os, sys, json, time, argparse, warnings
from collections import Counter

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
import torchvision.models as tv_models

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import dump, load
import sklearn

warnings.filterwarnings("ignore", category=UserWarning)
plt.switch_backend("Agg")  # headless safe


# -------------------------
# Utils & reproducibility
# -------------------------
def set_seed(s=2024):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except Exception:
    #     pass


# -------------------------
# Attention & Blocks
# -------------------------
class DualAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.acrosome_encoder = nn.Linear(in_dim, in_dim)
        self.pool = nn.AdaptiveAvgPool1d(1024)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, max(1, in_dim // 16), 1),
            nn.ReLU(),
            nn.Conv2d(max(1, in_dim // 16), in_dim, 1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_dim, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        acrosome_feat = x.mean(dim=[2, 3])
        acrosome_mask = torch.sigmoid(self.acrosome_encoder(acrosome_feat)).view(B, C, 1, 1)
        channel_scale = self.channel_attn(x) * acrosome_mask
        spatial_scale = self.spatial_attn(x)
        return x * channel_scale + x * spatial_scale


class ReversibleBlock(nn.Module):
    def __init__(self, in_channels, expansion=2, dropout=0.3):
        super().__init__()
        hidden_channels = in_channels * expansion
        self.segmentation_guide = nn.Sequential(
            nn.Conv2d(in_channels, max(1, in_channels // 8), 3, padding=1),
            nn.BatchNorm2d(max(1, in_channels // 8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // 8), 2, 1),
            nn.Softmax(dim=1)
        )
        self.F = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, in_channels, 1)
        )
        self.G = nn.Sequential(
            DualAttention(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        masks = self.segmentation_guide(x)
        mask1, mask2 = masks[:, 0:1], masks[:, 1:2]
        x1, x2 = x * mask1, x * mask2
        y1 = x1 + self.F(self.norm(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        y2 = x2 + self.G(y1)
        return y1 + y2


class MADRNet(nn.Module):
    def __init__(self, num_classes=2, backbone='resnet50', img_size=224,
                 use_rev_blocks=True, use_bilinear=True, pretrained=True):
        super().__init__()
        self.use_rev_blocks = use_rev_blocks
        self.use_bilinear = use_bilinear

        # Backbone
        self.backbone = None
        try:
            self.backbone = getattr(tv_models, backbone)(weights="DEFAULT" if pretrained else None)
        except Exception:
            # Fallback without internet/pretrained weights
            self.backbone = getattr(tv_models, backbone)(weights=None)

        in_channels = 2048  # resnet50

        if self.use_rev_blocks:
            self.shared_rev_block = ReversibleBlock(in_channels)
            self.rev_feat_dim = in_channels
        else:
            self.rev_feat_dim = 0

        if self.use_bilinear:
            self.bilinear_branch1 = nn.Sequential(
                nn.Conv2d(in_channels, 256, 1),
                DualAttention(256),
                nn.AdaptiveAvgPool2d(1)
            )
            self.bilinear_branch2 = nn.Sequential(
                nn.Conv2d(in_channels, 256, 1),
                DualAttention(256),
                nn.AdaptiveAvgPool2d(1)
            )
            self.bilinear_process = nn.Sequential(
                nn.Linear(256 * 256, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 512)
            )
            self.bilinear_dim = 512
        else:
            self.bilinear_dim = 0

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.base_feat_dim = in_channels

        combined_dim = 0
        if self.use_rev_blocks: combined_dim += self.rev_feat_dim
        if self.use_bilinear: combined_dim += self.bilinear_dim
        if not self.use_rev_blocks and not self.use_bilinear:
            combined_dim = self.base_feat_dim

        self.embedding_layer = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.classifier = nn.Linear(256, num_classes)
        self.extraction_points = {
            'backbone_features': None,
            'rev_features': None,
            'bilinear_features': None,
            'embedding_features': None
        }

        self.freeze_backbone()
        print(f"✅ MADRNet initialized | feat_dim={combined_dim:,} | params={self.count_parameters():,}")

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True

    def _forward_backbone(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        return self.backbone.layer4(x)

    def forward(self, x, extract_features=False):
        backbone_feat = self._forward_backbone(x)
        if extract_features:
            self.extraction_points['backbone_features'] = backbone_feat.detach()

        feats = []
        if not self.use_rev_blocks and not self.use_bilinear:
            feats.append(self.global_pool(backbone_feat).flatten(1))

        if self.use_rev_blocks:
            rev_feat = self.shared_rev_block(backbone_feat)
            rev_pooled = self.global_pool(rev_feat).flatten(1)
            feats.append(rev_pooled)
            if extract_features:
                self.extraction_points['rev_features'] = rev_feat.detach()

        if self.use_bilinear:
            f1 = self.bilinear_branch1(backbone_feat).flatten(1)
            f2 = self.bilinear_branch2(backbone_feat).flatten(1)
            bilinear = torch.bmm(f1.unsqueeze(2), f2.unsqueeze(1)).flatten(1)
            bilinear = torch.sign(bilinear) * torch.sqrt(torch.abs(bilinear) + 1e-5)
            bilinear_feat = self.bilinear_process(bilinear)
            feats.append(bilinear_feat)
            if extract_features:
                self.extraction_points['bilinear_features'] = bilinear_feat.detach()

        combined = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]
        emb = self.embedding_layer(combined)
        if extract_features:
            self.extraction_points['embedding_features'] = emb.detach()
        return self.classifier(emb)

    def get_features(self, dataloader, device, layer_names=['embedding_features']):
        self.eval()
        features_dict = {n: [] for n in layer_names}
        labels = []
        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(device)
                _ = self.forward(data, extract_features=True)
                for name in layer_names:
                    ft = self.extraction_points.get(name)
                    if ft is None: continue
                    arr = ft.detach().cpu().numpy()
                    if arr.ndim > 2:
                        arr = arr.reshape(arr.shape[0], -1)
                    features_dict[name].append(arr)
                labels.extend(target.numpy())
        final_features = {}
        for name in layer_names:
            if features_dict[name]:
                final_features[name] = np.vstack(features_dict[name])
        return final_features, np.array(labels)


# -------------------------
# Grad-CAM (safe hooks)
# -------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients, self.activations = None, None
        self.hook_layers()

    def hook_layers(self):
        def fwd_hook(module, inp, out):
            self.activations = out
        def bwd_hook(module, gin, gout):
            self.gradients = gout[0]

        self.target_layer.register_forward_hook(fwd_hook)
        try:
            self.target_layer.register_full_backward_hook(bwd_hook)
        except Exception:
            self.target_layer.register_backward_hook(bwd_hook)

    def generate_cam(self, input_image, class_idx):
        self.model.eval()
        output = self.model(input_image)
        self.model.zero_grad()
        target = output[:, class_idx]
        target.backward()
        gradients = self.gradients[0].detach().cpu().numpy()
        activations = self.activations[0].detach().cpu().numpy()
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# -------------------------
# CBAM (unchanged style)
# -------------------------
class CBAMBlock(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(1, channels // ratio)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


# -------------------------
# Dataset & transforms
# -------------------------

class SMIDSSpermDataset(Dataset):
    def __init__(self, root_dir, indices=None, transform=None, phase='train', target_balance=True):
        self.root_dir = root_dir
        self.transform = transform
        self.phase = phase
        
        # 获取所有有效类别（排除.ipynb_checkpoints）
        self.classes = []
        for d in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, d)
            if os.path.isdir(dir_path) and not d.startswith('.'):  # 排除隐藏文件夹
                self.classes.append(d)
        self.classes = sorted(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        all_images, all_labels = [], []
        for cls in self.classes:
            cdir = os.path.join(root_dir, cls)
            # 排除.ipynb_checkpoints文件
            imgs = [f for f in os.listdir(cdir) 
                    if f.lower().endswith('.jpg') and not f.startswith('.')]
            for name in imgs:
                all_images.append(os.path.join(cdir, name))
                all_labels.append(self.class_to_idx[cls])

        if indices is not None:
            self.images = [all_images[i] for i in indices]
            self.labels = [all_labels[i] for i in indices]
        else:
            self.images, self.labels = all_images, all_labels

        if target_balance and phase == 'train' and len(self.labels) > 0:
            self._balance_dataset()

        self._print_distribution()

    def _balance_dataset(self):
        class_groups = {}
        for img, lab in zip(self.images, self.labels):
            class_groups.setdefault(lab, []).append(img)
        counts = [len(v) for v in class_groups.values()]
        target = int(np.median(counts) * 1.2)
        balanced_images, balanced_labels = [], []
        for lab, imgs in class_groups.items():
            balanced_images.extend(imgs)
            balanced_labels.extend([lab]*len(imgs))
            if len(imgs) < target and len(imgs) > 0:
                need = target - len(imgs)
                extra = np.random.choice(imgs, need, replace=True)
                balanced_images.extend(list(extra))
                balanced_labels.extend([lab]*need)
        self.images, self.labels = balanced_images, balanced_labels

    def _print_distribution(self):
        class_counts = Counter(self.labels)
        total = len(self.labels)
        print(f"📊 {self.phase.upper()} ({total}):")
        for i, name in enumerate(self.classes):
            cnt = class_counts.get(i, 0)
            pct = (cnt/total*100) if total>0 else 0.0
            print(f"   {name}: {cnt} ({pct:.1f}%)")

    def __len__(self):
        return len(self.images)

    @staticmethod
    def enhance_sperm_features(image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharp = cv2.filter2D(denoised, -1, kernel)
        result = cv2.addWeighted(enhanced, 0.7, sharp, 0.3, 0)
        return np.clip(result, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        img_path = self.images[idx]; label = self.labels[idx]
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((224,224,3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.enhance_sperm_features(image)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(phase='train', img_size=224):
    if phase == 'train':
        return transforms.Compose([
            # 针对显微图像的特殊处理
            transforms.Grayscale(num_output_channels=3),  # 确保三通道
            transforms.Resize((img_size+64, img_size+64)),  # 更大尺寸以便裁剪
            
            # 针对低质量图像的增强
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.4),
            
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomRotation(15),  # 减小旋转角度
            
            # 调整颜色增强参数（显微图像对比度敏感）
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02),
            
            transforms.ToTensor(),
            
            # 针对灰度显微图像的归一化
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
            
            # 调整随机擦除参数（避免擦除关键结构）
            transforms.RandomErasing(p=0.1, scale=(0.01, 0.05), ratio=(0.3, 3.3))
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
        ])


# -------------------------
# Trainer
# -------------------------
class CBAMSpermTrainer:
    def __init__(self, model, train_loader, val_loader, device, num_classes=2):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        self.best_acc = 0.0
        self.best_model_state = None

    def train_kaggle_style(self, total_epochs=60):
        print(f"\n🚀  (TARGET: 91%+)  Total epochs: {total_epochs}")
        stage1 = total_epochs // 2
        print(f"📋 STAGE 1: Frozen backbone ({stage1} epochs)")
        self.model.freeze_backbone()
        opt1 = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=1e-3)

        for ep in range(stage1):
            tl, ta = self._train_epoch(opt1)
            vl, va = self._validate()
            self._update(tl, ta, vl, va, opt1.param_groups[0]['lr'])
            if va > self.best_acc:
                self.best_acc = va
                self.best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                print(f"Epoch {ep+1}/{stage1}: 🎉 NEW BEST! Val: {va:.2f}%")
            elif ep % 5 == 0:
                print(f"Epoch {ep+1}/{stage1}: Val: {va:.2f}%")

        print(f"\n📋 STAGE 2: Full fine-tuning ({total_epochs-stage1} epochs)")
        self.model.unfreeze_backbone()
        param_groups = [
            {'params': self.model.backbone.parameters(), 'lr': 1e-5},
        ]
        if hasattr(self.model, 'use_rev_blocks') and self.model.use_rev_blocks:
            param_groups.append({'params': self.model.shared_rev_block.parameters(), 'lr': 1e-4})
        if hasattr(self.model, 'use_bilinear') and self.model.use_bilinear:
            param_groups.append({'params': list(self.model.bilinear_branch1.parameters()) +
                                          list(self.model.bilinear_branch2.parameters()) +
                                          list(self.model.bilinear_process.parameters()), 'lr': 1e-4})
        param_groups.append({'params': list(self.model.embedding_layer.parameters()) + list(self.model.classifier.parameters()), 'lr': 1e-4})
        opt2 = optim.Adam(param_groups)

        for ep in range(total_epochs - stage1):
            tl, ta = self._train_epoch(opt2)
            vl, va = self._validate()
            self._update(tl, ta, vl, va, opt2.param_groups[0]['lr'])
            if va > self.best_acc:
                self.best_acc = va
                self.best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                print(f"Epoch {stage1+ep+1}/{total_epochs}: 🎉 NEW BEST! Val: {va:.2f}%")
            elif ep % 5 == 0:
                print(f"Epoch {stage1+ep+1}/{total_epochs}: Val: {va:.2f}%")

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        print(f"\n🏆 TRAIN DONE. Best Val Acc: {self.best_acc:.2f}%")
        return self.best_acc

    def _train_epoch(self, optimizer):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            out = self.model(data)
            loss = self.criterion(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            _, pred = out.max(1)
            total += target.size(0); correct += pred.eq(target).sum().item()
        return total_loss / len(self.train_loader), 100. * correct / total

    def _validate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                out = self.model(data)
                loss = self.criterion(out, target)
                total_loss += loss.item()
                _, pred = out.max(1)
                total += target.size(0); correct += pred.eq(target).sum().item()
        return total_loss / len(self.val_loader), 100. * correct / total

    def _update(self, tl, ta, vl, va, lr):
        self.history['train_loss'].append(tl)
        self.history['train_acc'].append(ta)
        self.history['val_loss'].append(vl)
        self.history['val_acc'].append(va)
        self.history['lr'].append(lr)


# -------------------------
# Visualization & reports
# -------------------------
def save_gradcam_results(model, dataloader, device, classes, save_dir, model_name, num_samples=12):
    try:
        target_layer = model.backbone.layer4[-1].conv3
    except Exception:
        print("Grad-CAM target layer not found; skip.")
        return
    print(f"🔍 Grad-CAM for {model_name} ...")
    gradcam = GradCAM(model, target_layer)
    class_samples = {i: [] for i in range(len(classes))}
    for data, targets in dataloader:
        for img, t in zip(data, targets):
            ci = int(t.item())
            if len(class_samples[ci]) < max(1, num_samples // max(1,len(classes))):
                class_samples[ci].append((img, ci))
        if all(len(v) >= max(1, num_samples // max(1,len(classes))) for v in class_samples.values()):
            break
    out_dir = os.path.join(save_dir, 'gradcam', model_name); os.makedirs(out_dir, exist_ok=True)
    for ci, samples in class_samples.items():
        cname = classes[ci]
        for si, (img, _) in enumerate(samples[:5]):
            img_tensor = img.unsqueeze(0).to(device)
            cam = gradcam.generate_cam(img_tensor, ci)
            orig = img.permute(1,2,0).numpy()
            orig = np.clip(orig * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]), 0, 1)
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1); plt.imshow(orig); plt.title(f'Original - {cname}'); plt.axis('off')
            plt.subplot(1,3,2); plt.imshow(cam, cmap='jet'); plt.title('Grad-CAM'); plt.axis('off')
            plt.subplot(1,3,3); plt.imshow(orig); plt.imshow(cam, cmap='jet', alpha=0.4); plt.title('Overlay'); plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{cname}_{si+1}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    print(f"✅ Grad-CAM saved: {out_dir}")


def save_tsne_visualization(features, labels, classes, save_dir, model_name, layer_name):
    print(f"📊 t-SNE for {model_name} - {layer_name} ...")
    try:
        X = features
        if X.shape[1] > 50:
            pca = PCA(n_components=50)
            X = pca.fit_transform(X)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(X)//10)), n_iter=1000)
        X2 = tsne.fit_transform(X)
        plt.figure(figsize=(12,8))
        colors = ['red','blue','green','orange','purple','brown','pink'][:len(classes)]
        for i, cname in enumerate(classes):
            mask = (labels == i)
            if mask.sum()==0: continue
            plt.scatter(X2[mask,0], X2[mask,1], label=cname, alpha=0.6, s=50)
        plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2'); plt.title(f't-SNE - {model_name} ({layer_name})')
        plt.legend(); plt.grid(True, alpha=0.3)
        out_dir = os.path.join(save_dir, 'tsne'); os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'{model_name}_{layer_name}_tsne.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ t-SNE saved.")
    except Exception as e:
        print(f"t-SNE failed: {e}")


def save_confusion_matrix(model, dataloader, device, classes, save_dir, model_name):
    print(f"📊 Confusion matrix for {model_name} ...")
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device)
            out = model(data)
            _, p = out.max(1)
            preds.extend(p.cpu().numpy()); gts.extend(targets.numpy())
    cm = confusion_matrix(gts, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}'); plt.ylabel('True'); plt.xlabel('Predicted')
    cm_dir = os.path.join(save_dir, 'confusion_matrices'); os.makedirs(cm_dir, exist_ok=True)
    plt.savefig(os.path.join(cm_dir, f'{model_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight'); plt.close()
    report = classification_report(gts, preds, target_names=classes, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(cm_dir, f'{model_name}_classification_report.csv'))
    print("✅ Confusion matrix & report saved.")
    return cm, report


def plot_training_history(history, save_dir, model_name):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(15,10))
    ax1.plot(epochs, history['train_loss'], label='Train'); ax1.plot(epochs, history['val_loss'], label='Val')
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history['train_acc'], label='Train'); ax2.plot(epochs, history['val_acc'], label='Val')
    ax2.axhline(91, linestyle='--', alpha=0.7, label='Target 91%'); ax2.legend(); ax2.set_title('Accuracy'); ax2.grid(True, alpha=0.3)
    ax3.plot(epochs, history['lr']); ax3.set_yscale('log'); ax3.set_title('LR'); ax3.grid(True, alpha=0.3)
    final_train = history['train_acc'][-1] if history['train_acc'] else 0
    final_val = history['val_acc'][-1] if history['val_acc'] else 0
    best_val = max(history['val_acc']) if history['val_acc'] else 0
    labels = ['Final Train','Final Val','Best Val','Target(91%)']; vals = [final_train, final_val, best_val, 91]
    bars = ax4.bar(labels, vals, alpha=0.8)
    ax4.set_ylim(0,100); ax4.set_title('Summary')
    for b, v in zip(bars, vals): ax4.text(b.get_x()+b.get_width()/2, v+1, f'{v:.1f}%', ha='center')
    plt.tight_layout()
    out_dir = os.path.join(save_dir, 'training_history'); os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'{model_name}_training_history.png'), dpi=300, bbox_inches='tight'); plt.close()


# -------------------------
# DFE Analysis (+ persist best)
# -------------------------
def run_dfe_analysis(model, train_loader, val_loader, device, classes, save_dir, model_name, persist_dir=None, do_tsne=True):
    print(f"🔬 DFE analysis for {model_name} ...")
    extraction_layers = ['embedding_features']
    feature_selectors = {
        'PCA': 'pca',
        'Chi2': 'chi2',
        'ANOVA': 'anova',
        'Variance': 'variance'
    }
    classifiers = {
        'SVM_RBF': SVC(kernel='rbf', C=1.0, random_state=42, probability=False),
        'SVM_Linear': SVC(kernel='linear', C=1.0, random_state=42, probability=False),
        'kNN_3': KNeighborsClassifier(n_neighbors=3, metric='euclidean'),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
    }

    try:
        train_feats_dict, train_labels = model.get_features(train_loader, device, extraction_layers)
        val_feats_dict, val_labels = model.get_features(val_loader, device, extraction_layers)
        if not train_feats_dict or not val_feats_dict:
            print("❌ No features extracted.")
            return None, None

        dfe_results = []
        best_snapshot = {"acc": -1.0, "layer": None, "selector": None, "classifier": None,
                         "selector_path": None, "classifier_path": None}
        persist_root = None
        if persist_dir is not None:
            persist_root = os.path.join(persist_dir, model_name)
            os.makedirs(persist_root, exist_ok=True)

        for layer_name, trX in train_feats_dict.items():
            vaX = val_feats_dict[layer_name]
            if do_tsne:
                try:
                    combX = np.vstack([trX, vaX]); combY = np.hstack([train_labels, val_labels])
                    save_tsne_visualization(combX, combY, classes, save_dir, model_name, layer_name)
                except Exception as e:
                    print(f"t-SNE skip ({e})")

            # Build selector objects per type
            for sel_name, sel_type in feature_selectors.items():
                try:
                    selector_obj, tr_sel, va_sel, sel_indices = None, None, None, None

                    if sel_type == 'pca':
                        pca = PCA(n_components=min(256, trX.shape[1], max(2, len(trX)//2)))
                        tr_sel = pca.fit_transform(trX)
                        va_sel = pca.transform(vaX)
                        selector_obj = pca

                    elif sel_type == 'chi2':
                        # make non-negative
                        shift = trX.min() - 1e-8 if np.min(trX) < 0 else 0.0
                        tr_adj = trX - shift
                        va_adj = vaX - shift
                        k = min(256, trX.shape[1])
                        sel = SelectKBest(chi2, k=k)
                        tr_sel = sel.fit_transform(tr_adj, train_labels)
                        va_sel = sel.transform(va_adj)
                        selector_obj = sel

                    elif sel_type == 'anova':
                        k = min(256, trX.shape[1])
                        sel = SelectKBest(f_classif, k=k)
                        tr_sel = sel.fit_transform(trX, train_labels)
                        va_sel = sel.transform(vaX)
                        selector_obj = sel

                    else:  # variance
                        variances = np.var(trX, axis=0)
                        sel_indices = np.argsort(variances)[::-1][:min(256, trX.shape[1])]
                        tr_sel = trX[:, sel_indices]
                        va_sel = vaX[:, sel_indices]

                    for clf_name, clf in classifiers.items():
                        try:
                            clf.fit(tr_sel, train_labels)
                            pred = clf.predict(va_sel)
                            acc = float(np.mean(pred == val_labels) * 100.0)
                            dfe_results.append({
                                'layer': layer_name, 'selector': sel_name, 'classifier': clf_name,
                                'accuracy': acc, 'method': f"{layer_name} + {sel_name} + {clf_name}"
                            })

                            if acc > best_snapshot["acc"]:
                                best_snapshot.update({"acc": acc, "layer": layer_name,
                                                      "selector": sel_name, "classifier": clf_name})
                                if persist_root is not None:
                                    # save selector & classifier
                                    if sel_type == 'variance':
                                        sel_path = os.path.join(persist_root, f"BEST_{layer_name}_{sel_name}_{clf_name}_indices.npy")
                                        np.save(sel_path, sel_indices)
                                        best_snapshot["selector_path"] = sel_path
                                    else:
                                        sel_path = os.path.join(persist_root, f"BEST_{layer_name}_{sel_name}_{clf_name}_selector.joblib")
                                        dump(selector_obj, sel_path)
                                        best_snapshot["selector_path"] = sel_path

                                    clf_path = os.path.join(persist_root, f"BEST_{layer_name}_{sel_name}_{clf_name}_classifier.joblib")
                                    dump(clf, clf_path)
                                    best_snapshot["classifier_path"] = clf_path

                                    with open(os.path.join(persist_root, "best_dfe_meta.json"), "w") as f:
                                        json.dump(best_snapshot, f, indent=2)

                        except Exception as e:
                            print(f"  Skip {sel_name} + {clf_name}: {e}")
                            continue

                except Exception as e:
                    print(f"  Selector {sel_name} failed: {e}")
                    continue

        if dfe_results:
            dfe_results.sort(key=lambda x: x['accuracy'], reverse=True)
            dfe_dir = os.path.join(save_dir, 'dfe_results'); os.makedirs(dfe_dir, exist_ok=True)
            pd.DataFrame(dfe_results).to_csv(os.path.join(dfe_dir, f'{model_name}_dfe_results.csv'), index=False)
            # Plot top-10
            top10 = dfe_results[:10]
            plt.figure(figsize=(12,8))
            methods = [r['method'] for r in top10]
            accs = [r['accuracy'] for r in top10]
            short = [m.replace(' + ','+\n') for m in methods]
            bars = plt.barh(range(len(short)), accs, alpha=0.85)
            plt.yticks(range(len(short)), short, fontsize=8)
            plt.xlabel('Accuracy (%)'); plt.title(f'Top 10 DFE - {model_name}')
            for i,(b,a) in enumerate(zip(bars,accs)):
                plt.text(b.get_width()+0.1, b.get_y()+b.get_height()/2, f'{a:.1f}%', va='center', fontsize=8)
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(dfe_dir, f'{model_name}_top_dfe_results.png'), dpi=300, bbox_inches='tight'); plt.close()
            print(f"✅ DFE done. Best: {dfe_results[0]['accuracy']:.2f}%")
            return dfe_results, best_snapshot
        else:
            print("❌ DFE produced no results.")
            return None, None

    except Exception as e:
        print(f"❌ DFE Error: {e}")
        import traceback; traceback.print_exc()
        return None, None


# -------------------------
# Reproduce & Inference
# -------------------------
def reproduce_best(save_dir, model_name, device=None, fold_dir=None, data_path="SMIDS-others"):
    """
    Re-evaluate saved best CNN + best DFE on the stored val split to reproduce accuracy.
    If fold_dir is provided, e.g., 'fold_1', it will look into save_dir/fold_1/...
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = os.path.join(save_dir, fold_dir) if fold_dir else save_dir

    # split & classes
    split = np.load(os.path.join(base_dir, "splits", "split_indices.npz"))
    val_idx = split["val"]
    with open(os.path.join(base_dir, "splits", "classes.json"), "r", encoding="utf-8") as f:
        cls_meta = json.load(f)
    classes = cls_meta["classes"]

    # val loader
    val_dataset = SMIDSSpermDataset(data_path, indices=val_idx,
                                    transform=get_transforms('val', 224),
                                    phase='val', target_balance=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    # CNN
    with open(os.path.join(base_dir, "checkpoints", f"{model_name}_model_config.json"), "r") as f:
        cfg = json.load(f)
    model = MADRNet(num_classes=cfg["num_classes"],
                                     backbone=cfg["backbone"], img_size=cfg["img_size"],
                                     use_rev_blocks=cfg["use_rev_blocks"],
                                     use_bilinear=cfg["use_bilinear"],
                                     pretrained=False).to(device)
    state = torch.load(os.path.join(base_dir, "checkpoints", f"{model_name}_best.pth"), map_location=device)
    model.load_state_dict(state); model.eval()

    # DFE
    dfe_root = os.path.join(base_dir, "dfe_pipelines", model_name)
    with open(os.path.join(dfe_root, "best_dfe_meta.json"), "r") as f:
        best = json.load(f)

    selector_path = best.get("selector_path")
    classifier_path = best.get("classifier_path")
    selector_obj, selector_indices = None, None
    if selector_path.endswith(".npy"):
        selector_indices = np.load(selector_path)
    else:
        selector_obj = load(selector_path)
    clf = load(classifier_path)

    feats_dict, labels = model.get_features(val_loader, device, [best["layer"]])
    X = feats_dict[best["layer"]]
    if selector_obj is not None:
        X_sel = selector_obj.transform(X)
    else:
        X_sel = X[:, selector_indices]

    preds = clf.predict(X_sel)
    acc = (preds == labels).mean() * 100.0
    print(f"[REPRODUCE] {fold_dir or ''} {model_name} + {best['layer']} + {best['selector']} + {best['classifier']}")
    print(f"Accuracy: {acc:.2f}%")
    return acc


def predict_folder_with_best_pipeline(images_dir, save_dir, model_name, device=None, fold_dir=None):
    """
    Predict .bmp images under images_dir using saved best CNN + DFE.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.join(save_dir, fold_dir) if fold_dir else save_dir

    with open(os.path.join(base_dir, "splits", "classes.json"), "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]
    class DummyDataset(Dataset):
        def __len__(self): return len(paths)
        def __getitem__(self, i):
            img = cv2.imread(paths[i]); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = SMIDSSpermDataset.enhance_sperm_features(img)
            pil = Image.fromarray(img)
            tensor = get_transforms('val', 224)(pil)
            return tensor, paths[i]

    ds = DummyDataset()
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    # load CNN
    with open(os.path.join(base_dir, "checkpoints", f"{model_name}_model_config.json"), "r") as f:
        cfg = json.load(f)
    model = MADRNet(num_classes=len(classes),
                                     backbone=cfg["backbone"], img_size=cfg["img_size"],
                                     use_rev_blocks=cfg["use_rev_blocks"],
                                     use_bilinear=cfg["use_bilinear"],
                                     pretrained=False).to(device)
    state = torch.load(os.path.join(base_dir, "checkpoints", f"{model_name}_best.pth"), map_location=device)
    model.load_state_dict(state); model.eval()

    # load DFE
    dfe_root = os.path.join(base_dir, "dfe_pipelines", model_name)
    with open(os.path.join(dfe_root, "best_dfe_meta.json"), "r") as f:
        best = json.load(f)
    selector_path, classifier_path = best["selector_path"], best["classifier_path"]
    selector_obj, selector_indices = None, None
    if selector_path.endswith(".npy"):
        selector_indices = np.load(selector_path)
    else:
        selector_obj = load(selector_path)
    clf = load(classifier_path)

    # features -> selector -> classifier
    all_feats, file_paths = [], []
    with torch.no_grad():
        for batch, batch_paths in dl:
            batch = batch.to(device)
            _ = model.forward(batch, extract_features=True)
            feats = model.extraction_points[best["layer"]].cpu().numpy()
            if feats.ndim > 2: feats = feats.reshape(feats.shape[0], -1)
            all_feats.append(feats)
            file_paths.extend(list(batch_paths))
    X = np.vstack(all_feats)
    X_sel = selector_obj.transform(X) if selector_obj is not None else X[:, selector_indices]
    preds = clf.predict(X_sel)
    return list(zip(file_paths, [classes[i] for i in preds]))


# -------------------------
# Main with K-Fold CV
# -------------------------
def run_cv(args):
    set_seed(2024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    print("🚀 ENHANCED FINE-GRAINED REVERSIBLE NET FOR SPERM CLASSIFICATION")
    print("="*72)
    print(f"📁 Dataset: {args.data_path}")
    print(f"💾 Save dir: {args.save_dir}")
    print(f"💻 Device: {device}")
    print(f"🔁 Folds: {args.kfolds} | Epochs: {args.epochs} | Batch: {args.batch_size}")

    # collect all images & labels once
    classes = sorted([d for d in os.listdir(args.data_path) 
                     if os.path.isdir(os.path.join(args.data_path, d)) and not d.startswith('.')])
    class_to_idx = {c:i for i,c in enumerate(classes)}
    all_images, all_labels = [], []
    for c in classes:
        cdir = os.path.join(args.data_path, c)
        # 排除.ipynb_checkpoints文件
        imgs = [f for f in os.listdir(cdir) 
                if f.lower().endswith('.jpg') and not f.startswith('.')]
        for name in imgs:
            all_images.append(os.path.join(cdir, name))
            all_labels.append(class_to_idx[c])
    all_labels = np.array(all_labels)
    
    print(f"📊 Total images: {len(all_images)} | Classes: {classes}")

    # env meta
    meta = {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": getattr(tv_models, "__version__", "N/A"),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "seed": 2024,
        "img_size": 224,
        "augment_train": True
    }
    with open(os.path.join(args.save_dir, "env_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # CV splitter
    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=42)

    # models to test
    model_specs = [
        ('MADRNet_Full', dict(use_rev_blocks=True, use_bilinear=False))]
    # models to test
    # model_specs = [
    #     ('MADRNet_Full', dict(use_rev_blocks=True, use_bilinear=False)),
    #     ('MADRNet_Base', dict(use_rev_blocks=False, use_bilinear=False)),
    # ]
    # to aggregate stats across folds
    cv_summary = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(all_images)), all_labels), start=1):
        fold_name = f"fold_{fold}"
        print(f"\n{'='*30}  {fold_name}  {'='*30}")
        fold_dir = os.path.join(args.save_dir, fold_name); os.makedirs(fold_dir, exist_ok=True)

        # save split & classes
        split_dir = os.path.join(fold_dir, "splits"); os.makedirs(split_dir, exist_ok=True)
        np.savez(os.path.join(split_dir, "split_indices.npz"), train=train_idx, val=val_idx)
        with open(os.path.join(split_dir, "classes.json"), "w", encoding="utf-8") as f:
            json.dump({"classes": classes, "class_to_idx": class_to_idx}, f, ensure_ascii=False, indent=2)

        # datasets & loaders
        train_ds = SMIDSSpermDataset(args.data_path, indices=train_idx, transform=get_transforms('train', 224), phase='train', target_balance=True)
        val_ds   = SMIDSSpermDataset(args.data_path, indices=val_idx,   transform=get_transforms('val', 224),   phase='val',   target_balance=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=8, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        for model_name, kwargs in model_specs:
            print(f"\n🧠 TESTING: {model_name}")
            model = MADRNet(num_classes=len(classes), backbone='resnet50', img_size=224,
                                             pretrained=True, **kwargs)

            trainer = CBAMSpermTrainer(model, train_loader, val_loader, device, len(classes))
            t0 = time.time()
            base_acc = trainer.train_kaggle_style(total_epochs=args.epochs)
            train_time = time.time() - t0

            # save best CNN checkpoint + config
            ckpt_dir = os.path.join(fold_dir, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{model_name}_best.pth"))
            with open(os.path.join(ckpt_dir, f"{model_name}_model_config.json"), "w") as f:
                json.dump({
                    "num_classes": len(classes),
                    "use_rev_blocks": getattr(model, "use_rev_blocks", None),
                    "use_bilinear": getattr(model, "use_bilinear", None),
                    "backbone": "resnet50",
                    "img_size": 224
                }, f, indent=2)

            # plots & matrices
            plot_training_history(trainer.history, fold_dir, model_name)
            cm, report = save_confusion_matrix(model, val_loader, device, classes, fold_dir, model_name)

            # grad-cam (optional)
            if args.do_gradcam:
                save_gradcam_results(model, val_loader, device, classes, fold_dir, model_name)

            # DFE + persist best pipeline
            dfe_results, best_dfe = run_dfe_analysis(
                model, train_loader, val_loader, device, classes, fold_dir, model_name,
                persist_dir=os.path.join(fold_dir, "dfe_pipelines"),
                do_tsne=bool(args.do_tsne) and (fold==1)  # 避免每折都TSNE过慢：默认仅首折绘制
            )

            # per-fold record
            best_dfe_acc = base_acc
            best_method = 'Base CNN'
            if dfe_results:
                best_dfe_acc = dfe_results[0]['accuracy']
                best_method = dfe_results[0]['method']

            cv_summary.append({
                'Fold': fold,
                'Model': model_name,
                'Base_CNN_Acc': base_acc,
                'Best_DFE_Acc': best_dfe_acc,
                'Best_DFE_Method': best_method,
                'Params_M': model.count_parameters() / 1e6,
                'Train_Time_min': train_time/60.0
            })

        # save fold summary csv
        fold_df = pd.DataFrame([r for r in cv_summary if r['Fold'] == fold])
        fold_df.to_csv(os.path.join(fold_dir, f'{fold_name}_results.csv'), index=False)

    # ---- aggregate mean ± sd by model ----
    all_df = pd.DataFrame(cv_summary)
    all_df.to_csv(os.path.join(args.save_dir, 'all_folds_results_raw.csv'), index=False)

    final_rows = []
    for model_name in all_df['Model'].unique():
        sub = all_df[all_df['Model'] == model_name]
        base_mean, base_sd = sub['Base_CNN_Acc'].mean(), sub['Base_CNN_Acc'].std(ddof=1)
        dfe_mean, dfe_sd   = sub['Best_DFE_Acc'].mean(), sub['Best_DFE_Acc'].std(ddof=1)
        imp_mean = (sub['Best_DFE_Acc'] - sub['Base_CNN_Acc']).mean()
        final_rows.append({
            'Model': model_name,
            'Base_CNN_Acc_mean': base_mean,
            'Base_CNN_Acc_sd': base_sd,
            'Best_DFE_Acc_mean': dfe_mean,
            'Best_DFE_Acc_sd': dfe_sd,
            'Improvement_mean': imp_mean
        })
        print(f"\n✅ {model_name} (5-fold):")
        print(f"   Base CNN      : {base_mean:.2f} ± {base_sd:.2f}%")
        print(f"   Best DFE      : {dfe_mean:.2f} ± {dfe_sd:.2f}%")
        print(f"   Improvement   : {imp_mean:+.2f}%")

    pd.DataFrame(final_rows).to_csv(os.path.join(args.save_dir, 'final_cv_summary.csv'), index=False)
    print(f"\n🏁 CV finished. Summary saved to {os.path.join(args.save_dir, 'final_cv_summary.csv')}")


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, default='mhsma/vacuole')
    p.add_argument('--save_dir', type=str, default='result_mhsma/vacuole')
    p.add_argument('--epochs', type=int, default=100)  # 可改回100
    p.add_argument('--kfolds', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--do_gradcam', type=int, default=1, help='1 to enable')
    p.add_argument('--do_tsne', type=int, default=1, help='1 to enable (默认仅首折画)')
    return p.parse_args()


if __name__ == "__main__":
    print("🧬 ENHANCED FINE-GRAINED REVERSIBLE NET FOR SPERM CLASSIFICATION (CV)")
    print("="*72)
    args = parse_args()
    run_cv(args)

    # 你也可以在训练结束后，手动调用如下进行复现或新图像推断：
    # reproduce_best(args.save_dir, "MADRNet_Full", fold_dir="fold_1", data_path=args.data_path)
    # preds = predict_folder_with_best_pipeline("some_images_dir", args.save_dir, "MADRNet_Full", fold_dir="fold_1")
    # print(preds)
