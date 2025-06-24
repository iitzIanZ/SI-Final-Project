"""
Stable Diffusion 2 Inpainting LoRA Fine-tuning Script (Manual GPU & AMP Management)

本腳本用於對 Stable Diffusion 2 Inpainting 模型進行 LoRA 微調。
它移除了對 `accelerate` 庫的依賴，並採用手動方式管理 GPU 設備上的數據傳輸、
梯度累積以及自動混合精度 (AMP)。LoRA 配置使用 `peft` 庫，並通過 `diffusers`
的 `unet.add_adapter()` 方法將其應用到 UNet 模型上。

主要特點：
-   **模型**: `stabilityai/stable-diffusion-2-inpainting`。
-   **微調方法**: LoRA (Low-Rank Adaptation)。
-   **依賴**: `diffusers`, `transformers`, `peft`, `torch`, `Pillow`, `torchvision`.
-   **數據**: 假定數據集結構為 `data_root/train` 和 `data_root/val`，
    其中包含 `*_in.png` (輸入圖像), `*_mask.png` (遮罩), `*_gt.png` (真實圖像)。
-   **LoRA 權重保存**: 訓練完成後，LoRA 權重會保存到 `output_dir` 指定的目錄中，
    檔名通常為 `pytorch_lora_weights.safetensors`。

配置：
您可以在腳本的 `1. 配置與初始化` 部分調整以下參數：
-   `model_name`: 基礎 Inpainting 模型的名稱。
-   `data_root`: 數據集根目錄。
-   `output_dir`: 訓練完成後 LoRA 權重保存的目錄。
-   `resolution`: 圖像解析度，應與數據集圖像的解析度匹配。
-   `train_batch_size`: 訓練批次大小。
-   `gradient_accumulation_steps`: 梯度累積步數。
-   `learning_rate`: 學習率。
-   `num_train_epochs`: 訓練的 epoch 數量。
-   `seed`: 隨機種子，用於結果復現。

使用方式：
1.  **準備數據**: 確保您的圖像數據按照 `data_root/train` 和 `data_root/val`
    的結構組織，並包含 `*_in.png`, `*_mask.png`, `*_gt.png` 檔案。
    （本腳本的 `InpaintingDataset` 類假定這種檔案命名和組織方式）。
2.  **配置參數**: 修改腳本中 `1. 配置與初始化` 部分的參數，以適應您的環境和需求。
3.  **運行腳本**: 在終端機中執行以下命令：
    ```bash
    python your_script_name.py
    ```
    (將 `your_script_name.py` 替換為您的 Python 檔案名稱，例如 `diffusion_inpainting/lora_finetune.py`)。

輸出：
-   訓練過程中的損失信息會打印到控制台。
-   訓練完成後，LoRA 權重（通常是 `pytorch_lora_weights.safetensors`）
    將被保存到您指定的 `output_dir` 目錄中。

注意事項：
-   本腳本假設您有足夠的 GPU 顯存來運行訓練。
-   `num_workers` 設置為 0 以避免與 CUDA 相關的多進程問題。
-   `batch.pop("input_ids")` 是為了兼容 `peft` 和 `diffusers` 可能的互動而保留的，
    避免 `TypeError`。
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os
from pathlib import Path
import random
# import accelerate # 繼續移除 accelerate
from diffusers import (
    StableDiffusionInpaintPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    # LoraConfig # 這裡不再從 diffusers 導入 LoraConfig
)
from transformers import CLIPTokenizer, CLIPTextModel
import time
from peft import LoraConfig # 這裡從 peft 導入 LoraConfig
# from torch.cuda.amp import autocast, GradScaler # 將更新為 torch.amp
import argparse


def LoRA_train(data_root="gen_ft_data", output_dir="lora_15", num_train_epochs=10, train_batch_size=8, gradient_accumulation_steps=2, learning_rate=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 設置模型名稱和路徑
    model_name = "stabilityai/stable-diffusion-2-inpainting"
    data_root = "gen_ft_data" # 您的數據根目錄
    output_dir = "lora_15" # 更改輸出目錄名稱以區分
    resolution = 512 # 圖像解析度 (需與 make_data.py 中的 size 一致)

    # 訓練參數
    train_batch_size = 8 # 建議從 1 開始，如果記憶體足夠再增加
    gradient_accumulation_steps = 2 # 梯度累積步數 (現在手動實現)
    learning_rate = 1e-4
    num_train_epochs = 15 # 訓練 epoch 數量
    seed = 42

    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------
    # 2. 加載基礎模型組件 (全部在主進程中移到 GPU)
    # ---------------------------------------------------

    print("Loading pipeline components...")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

    # 凍結 VAE 和 Text Encoder 的權重，因為我們只微調 UNet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # ---------------------------------------------------
    # 3. 配置和應用 LoRA 到 UNet (使用 peft.LoraConfig 和 diffusers.add_adapter)
    # ---------------------------------------------------

    # 設置 LoRA 配置 (來自 peft 庫)
    lora_config_peft = LoraConfig(
        r=4,
        lora_alpha=16,
        init_lora_weights="gaussian", # 可選，初始化 LoRA 權重的方式
        target_modules=["to_q", "to_k", "to_v", "to_out.0"], # 明確指定要注入 LoRA 的模塊
    )

    # 使用 unet.add_adapter() 添加 LoRA 適配器
    unet.add_adapter(lora_config_peft)

    # 手動打印可訓練參數 (因為沒有 peft 的 print_trainable_parameters())
    trainable_params = 0
    all_params = 0
    for name, param in unet.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        all_params += param.numel()

    if all_params > 0:
        print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.4f}")
    else:
        print("No parameters found in the UNet.")


    # ---------------------------------------------------
    # 4. 資料集和 DataLoader
    # ---------------------------------------------------

    class InpaintingDataset(Dataset):
        def __init__(self, data_root, split, tokenizer, resolution=512):
            self.data_dir = os.path.join(data_root, split)
            # 確保文件路徑正確，根據您提供的文件夾結構
            self.image_paths = sorted(list(Path(self.data_dir).glob("*_in.png"))) 
            self.resolution = resolution
            self.tokenizer = tokenizer

            self.image_transform = transforms.Compose([
                transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(), # 將圖片轉換為 [0, 1] 範圍
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(), # 將遮罩轉換為 [0, 1] 範圍
            ])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            in_path = self.image_paths[idx]
            stem = in_path.stem.replace("_in", "")
            mask_path = in_path.parent / f"{stem}_mask.png"
            gt_path = in_path.parent / f"{stem}_gt.png"

            input_image = Image.open(in_path).convert("RGB")
            mask_image = Image.open(mask_path).convert("L")
            gt_image = Image.open(gt_path).convert("RGB")

            input_image_tensor = self.image_transform(input_image)
            mask_image_tensor = self.mask_transform(mask_image)
            gt_image_tensor = self.image_transform(gt_image)

            input_image_tensor = input_image_tensor * 2 - 1
            gt_image_tensor = gt_image_tensor * 2 - 1

            caption = "A man with sunglasses and a beard " # 這裡的 caption 可以根據您的需求調整或從數據中讀取
            text_input_ids = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)

            return {
                "pixel_values_input": input_image_tensor,
                "pixel_values_mask": mask_image_tensor,
                "pixel_values_gt": gt_image_tensor,
                "input_ids": text_input_ids,
            }

    train_dataset = InpaintingDataset(
        data_root=data_root,
        split="train",
        tokenizer=tokenizer,
        resolution=resolution
    )
    # 注意：num_workers 仍然建議設置為 0，以避免多進程和 CUDA 設備交互問題。
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    val_dataset = InpaintingDataset(
        data_root=data_root,
        split="val",
        tokenizer=tokenizer,
        resolution=resolution
    )
    val_dataloader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False, num_workers=0)

    # ---------------------------------------------------
    # 5. 優化器與自動混合精度 (AMP) scaler
    # ---------------------------------------------------

    # 優化器現在自動只針對已啟用 requires_grad 的 LoRA 層參數進行優化
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler('cuda') # 初始化 GradScaler for AMP

    # ---------------------------------------------------
    # 6. 訓練循環 (手動管理)
    # ---------------------------------------------------

    print("Starting training...")
    unet.train() # 確保 UNet 處於訓練模式
    vae.eval() # VAE 保持評估模式
    text_encoder.eval() # Text Encoder 保持評估模式

    num_update_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    global_step = 0
    start_time = time.time()

    for epoch in range(num_train_epochs):
        print(f"Epoch {epoch+1}/{num_train_epochs}")
        for step, batch in enumerate(train_dataloader):
            # 梯度累積的開始
            with torch.amp.autocast('cuda'): # 使用自動混合精度
                # 手動將所有從 DataLoader 獲取的張量移到 GPU
                input_image = batch["pixel_values_input"].to(device)
                mask_image = batch["pixel_values_mask"].to(device)
                gt_image = batch["pixel_values_gt"].to(device)
                
                # --- 關鍵修正：從 batch 中取出 input_ids 並從原始 batch 字典中移除 ---
                # 由於 diffusers 依賴 peft，為避免 TypeError，仍然保留此行
                input_ids = batch.pop("input_ids").to(device) 

                # 1. 將圖像編碼到 VAE 潛在空間
                with torch.no_grad(): # VAE 不參與梯度計算
                    input_image_latents = vae.encode(input_image).latent_dist.sample() * vae.config.scaling_factor
                    gt_image_latents = vae.encode(gt_image).latent_dist.sample() * vae.config.scaling_factor

                # 2. 準備遮罩的潛在表示
                # 確保 mask_latents 與圖像潛在空間維度一致
                mask_latents = transforms.functional.resize(
                    mask_image,
                    (input_image_latents.shape[2], input_image_latents.shape[3]),
                    interpolation=transforms.InterpolationMode.NEAREST
                ).to(input_image_latents.dtype)

                # 3. 生成噪聲和時間步
                noise = torch.randn_like(gt_image_latents)
                batch_size = gt_image_latents.shape[0]

                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),        # ← 使用真實的 batch 大小
                    device=device,
                    dtype=torch.long
                )
                noisy_latents = noise_scheduler.add_noise(gt_image_latents, noise, timesteps)

                # 4. 生成文本嵌入
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                # 5. 構建 UNet 輸入 (Inpainting UNet 的標準 9 通道輸入：latent_image + mask + masked_image_latent)
                # 這裡 masked_image_latent 實際上是 input_image_latents
                unet_input_sample = torch.cat([noisy_latents, mask_latents, input_image_latents], dim=1)

                # 6. 前向傳播 (明確只傳遞 UNet 期望的參數)
                model_pred = unet(
                    sample=unet_input_sample,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False
                )[0]

                # 7. 計算損失 (預測的噪音與真實添加的噪音之間的 MSE)
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # 規範化損失以進行梯度累積
                loss = loss / gradient_accumulation_steps

            # 8. 反向傳播
            scaler.scale(loss).backward() # 使用 scaler.scale() 進行反向傳播

            # 9. 梯度累積和優化器步驟
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                scaler.unscale_(optimizer) # 在裁剪前unscale梯度
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0) # 梯度裁剪 (可選)
                scaler.step(optimizer) # 使用 scaler.step() 更新優化器
                scaler.update() # 更新 scaler
                optimizer.zero_grad() # 梯度歸零

            global_step += 1
            # 打印進度
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                
                print(f"Epoch {epoch+1}, Step {step+1}/{len(train_dataloader)}, Loss: {loss.item() * gradient_accumulation_steps:.4f}") # 顯示未規範化的損失

    # ---------------------------------------------------
    # 7. 保存 LoRA 權重 (使用 diffusers 原生方法)
    # ---------------------------------------------------

    # 使用 unet.save_attn_procs() 保存 LoRA 權重
    unet.save_attn_procs(output_dir)
    print(f"LoRA weights saved to {output_dir}")

    print("Training complete!")
    end_time = time.time()
    print(f"training time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stable Diffusion 2 Inpainting with LoRA.")

    parser.add_argument("--data_root", type=str, default="gen_ft_data",
                        help="Root directory of the dataset (default: 'gen_ft_data').")

    parser.add_argument("--output_dir", type=str, default="lora_10",
                        help="Directory to save the LoRA weights (default: 'lora_10').")

    parser.add_argument("--num_train_epochs", type=int, default=10,
                        help="Number of training epochs (default: 10).")
    
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Batch size for training (default: 8).")
    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps (default: 2).")
    
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for training (default: 1e-4).")
    
    args = parser.parse_args()
    LoRA_train(data_root=args.data_root,
               output_dir=args.output_dir,
               num_train_epochs=args.num_train_epochs,
               train_batch_size=args.train_batch_size,
               gradient_accumulation_steps=args.gradient_accumulation_steps,
               learning_rate=args.learning_rate)

