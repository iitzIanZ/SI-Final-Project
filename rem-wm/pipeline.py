# lora_watermark_pipeline.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from remwm import WatermarkRemover
from transformers import Trainer, TrainingArguments
from diffusers import UNet2DConditionModel
from peft import LoraConfig, get_peft_model, TaskType
from torchvision import transforms
import matplotlib.pyplot as plt

def debug_show(img, title, out_dir="debug", idx=None):
    """
    同時做到：
     1. 在 notebook / console 中用 matplotlib 顯示
     2. 儲存一張 .png 到 out_dir  
    參數：
      img     : cv2 讀進來的 BGR 或 BGRA ndarray
      title   : 圖片上要顯示的標題
      out_dir : 儲存路徑 (若不存在會自動建立)
      idx     : 檔名用的索引 (可 None)
    """
    os.makedirs(out_dir, exist_ok=True)

    # matplotlib 顯示
    plt.figure(figsize=(4,4))
    # 若是 4 通道就轉 RGBA，否則轉 RGB
    if img.shape[2] == 4:
        img2 = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img2)
    plt.title(title)
    plt.axis("off")
    plt.show()

    # 存檔
    fn = f"{title.replace(' ','_')}"
    if idx is not None:
        fn = f"{idx:02d}_{fn}"
    fn += ".png"
    cv2.imwrite(os.path.join(out_dir, fn), img)

# -----------------------------------------------------------------------------
# Step 1: 使用 Florence 偵測 logo，取得 mask 與 bbox
# -----------------------------------------------------------------------------
def detect_logo(remover: WatermarkRemover, image_path: str):
    """
    傳回：
      mask: 二值 np.ndarray, shape=(H,W)，值為 0/1
      bbox: (x1,y1,x2,y2)
    """
    mask, bbox = remover.generate_mask_florence(image_path)
    debug_show((mask[...,None]*255).repeat(3,axis=2), "Florence Mask", idx=1)
    return mask, bbox

# -----------------------------------------------------------------------------
# Step 2: 擷取 bbox 周圍的色塊 patches（預設 5 張）
# -----------------------------------------------------------------------------
def extract_patches(img: np.ndarray, bbox: tuple, num_patches=5):
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    coords = [
        (-w, 0), (w, 0), (0, -h), (0, h), (w, h)
    ][:num_patches]
    patches = []
    H, W = img.shape[:2]
    for dx, dy in coords:
        cx = np.clip(x1 + dx, 0, W - w)
        cy = np.clip(y1 + dy, 0, H - h)
        patch = img[cy:cy+h, cx:cx+w]
        if patch.shape[0]==h and patch.shape[1]==w:
            patches.append(patch.copy())
    return patches

def draw_border(img, color=(0,0,255), thickness=2):
    """在 img 四周畫紅框，回傳新影像"""
    bordered = img.copy()
    h, w = img.shape[:2]
    cv2.rectangle(bordered, (0,0), (w-1,h-1), color, thickness)
    return bordered

# -----------------------------------------------------------------------------
# Step 3: 從原圖與 mask 製作 RGBA logo 圖
# -----------------------------------------------------------------------------
def make_logo_rgba(img: np.ndarray, mask: np.ndarray):
    """
    回傳 logo_rgba: np.ndarray, shape=(H,W,4)
    """
    H, W = mask.shape
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    alpha = (mask * 255).astype(np.uint8)
    rgba[:, :, 3] = alpha
    return rgba

# -----------------------------------------------------------------------------
# Step 4: 合成 (syn_img, target_img) 訓練對
# -----------------------------------------------------------------------------
def synthesize_pairs(patches, logo_rgba):
    syn_imgs, target_imgs = [], []
    h, w = logo_rgba.shape[:2]
    for patch in patches:
        H, W = patch.shape[:2]
        x_off = (W - w) // 2
        y_off = (H - h) // 2
        comp = patch.copy()
        alpha = logo_rgba[:, :, 3] / 255.0
        for c in range(3):
            comp[y_off:y_off+h, x_off:x_off+w, c] = (
                alpha * logo_rgba[:, :, c] +
                (1-alpha) * comp[y_off:y_off+h, x_off:x_off+w, c]
            )
        syn_imgs.append(comp)
        target_imgs.append(patch)
    return syn_imgs, target_imgs

# -----------------------------------------------------------------------------
# Step 5: 定義自訂 Dataset
# -----------------------------------------------------------------------------
class LogoDataset(Dataset):
    def __init__(self, syn_imgs, tgt_imgs, transform):
        self.syn = syn_imgs
        self.tgt = tgt_imgs
        self.tf = transform

    def __len__(self):
        return len(self.syn)

    def __getitem__(self, idx):
        x = self.tf(self.syn[idx])
        y = self.tf(self.tgt[idx])
        return {"image": x, "target": y}

# -----------------------------------------------------------------------------
# Step 6: LoRA 微調 LaMa Inpainting 的 UNet
# -----------------------------------------------------------------------------
def train_lora_on_pairs(syn_imgs, target_imgs, output_dir="./lorawork"):
    # 1. 載入原版 UNet
    unet = UNet2DConditionModel.from_pretrained("lama-cleaner-unet")
    # 2. 建立 LoRA Adapter
    lora_cfg = LoraConfig(
        task_type=TaskType.IMAGE_INPAINTING,
        inference_mode=False,  # 訓練模式
        r=8, alpha=16, dropout=0.05
    )
    lora_unet = get_peft_model(unet, lora_cfg)

    # 3. 資料前處理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    dataset = LogoDataset(syn_imgs, target_imgs, transform)

    # 4. Trainer 設定
    def collate_fn(batch):
        imgs = torch.stack([b["image"] for b in batch])
        tgts = torch.stack([b["target"] for b in batch])
        return {"image": imgs, "target": tgts}

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=5,
        logging_steps=10,
    )
    trainer = Trainer(
        model=lora_unet,
        args=args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    trainer.train()
    # 回傳訓練後的 LoRA Adapter
    return lora_unet

# -----------------------------------------------------------------------------
# Step 7: Test-Time Training & Inpainting
# -----------------------------------------------------------------------------
def test_time_inpaint(remover, image_path, lora_unet=None):
    # 1. 偵測 logo，取得二值 mask 與 bbox
    #    注意：這裡直接呼叫你在 remwm.py 中新增的 generate_mask_florence
    mask, bbox = remover.generate_mask_florence(image_path)
    img = cv2.imread(image_path)
    debug_show(img, "Original Image", idx=0)

    # 2. 先把整張 RGBA 做出來，再依 bbox 裁切出 logo 小區域
    full_rgba = make_logo_rgba(img, mask)      # (H_full, W_full, 4)
    x1, y1, x2, y2 = bbox
    logo_rgba = full_rgba[y1:y2, x1:x2, :]      # (h_logo, w_logo, 4)
    debug_show(logo_rgba, "Cropped_Logo_RGBA", idx=10)

    # 3. 擷取同尺寸的周圍 patches
    patches = extract_patches(img, bbox, num_patches=5)
    for i, p in enumerate(patches):
        debug_show(p, f"Patch_{i}_raw", idx=20 + i)

    # 4. 合成 syn/tgt pair：syn 貼上 logo，tgt 保留原 patch
    syn, tgt = synthesize_pairs(patches, logo_rgba)
    for i, (s, t) in enumerate(zip(syn, tgt)):
        print(f"Patch #{i}: syn.shape={s.shape}, tgt.shape={t.shape}")
        s_b = draw_border(s)  # 在 syn patch 四周畫紅框
        debug_show(s_b, f"Syn_{i}_patch_with_logo", idx=30 + i)
        debug_show(t,   f"Tgt_{i}_patch_clean",       idx=40 + i)

    # 5. 短訓 LoRA Adapter（使用剛才的 syn/tgt）
    print("=== Ready to train LoRA on above syn/gt pairs ===")
    adapter = train_lora_on_pairs(syn, tgt, output_dir="./ttt_lorawork")

    # 6. 使用微調後的 adapter 做最終 inpainting
    inpainted = remover.inpaint_with_custom_unet(img, mask, adapter)
    debug_show(inpainted, "TTT_Inpainted", idx=100)

    return inpainted


# -----------------------------------------------------------------------------
# 主流程示範
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 0. 初始化 WatermarkRemover
    remover = WatermarkRemover()

    # 宣告輸入與輸出
    input_path = "original\\test6.png"
    output_path = "result1.png"

    # （示範）先做一般流程
    remover.process_images_florence_lama(input_path, output_path)

    # 若要用 LoRA + TTT：
    inpainted = test_time_inpaint(remover, input_path, None)
    cv2.imwrite("example_image/ttt_result.png", inpainted)
