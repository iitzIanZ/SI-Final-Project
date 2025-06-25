import os
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torchvision.transforms as transforms
import argparse

def inference_model(lora_weights_path, input_dir, output_dir):
    # ---------------------------------------------------
    # 1. 配置與初始化
    # ---------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_model_name = "stabilityai/stable-diffusion-2-inpainting"
    os.makedirs(output_dir, exist_ok=True)

    prompt = " "
    negative_prompt = " "

    # ---------------------------------------------------
    # 2. 加載 Inpainting Pipeline 和 LoRA 權重
    # ---------------------------------------------------
    print("Loading base inpainting pipeline...")
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16
    ).to(device)

    print(f"Loading LoRA weights from {lora_weights_path}...")
    if not os.path.exists(lora_weights_path):
        print(f"Error: LoRA weights directory not found at '{lora_weights_path}'. Please check the path.")
        return
        
    pipeline.load_lora_weights(lora_weights_path)
    print("LoRA weights loaded successfully.")

    # ---------------------------------------------------
    # 3. 定義影像與遮罩前處理 (保持不進行 Resize)
    # ---------------------------------------------------
    img_transform = transforms.Compose([]) # 保持為空
    mask_transform = transforms.Compose([]) # 保持為空

    # ---------------------------------------------------
    # 4. 批次處理資料夾
    # ---------------------------------------------------
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'. Please check the path.")
        return

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".png") or fname.endswith("_mask.png"):
            continue

        image_path = os.path.join(input_dir, fname)
        mask_path = os.path.join(input_dir, fname.replace(".png", "_mask.png"))

        if not os.path.exists(mask_path):
            print(f"找不到遮罩：{mask_path}，跳過 {fname}")
            continue

        # 讀取並前處理 (不進行 resize)
        input_image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")
        
        # 儲存原始圖片尺寸，用於後續縮放
        original_width, original_height = input_image.size

        input_image = img_transform(input_image)
        mask_image = mask_transform(mask_image)

        print(f"Processing {fname} (Input size: {input_image.size})...")
        with torch.no_grad():
            generator = torch.Generator(device=device).manual_seed(42)
            result = pipeline(
                prompt=prompt,
                image=input_image,
                mask_image=mask_image,
                num_inference_steps=50,
                strength=1.0,
                generator=generator,
                negative_prompt=negative_prompt
            ).images[0]

        # 將結果圖片縮放回原始尺寸
        # 使用 Image.LANCZOS 進行高質量縮放
        result = result.resize((original_width, original_height), Image.LANCZOS)
        
        # 儲存結果
        out_path = os.path.join(output_dir, fname)
        result.save(out_path)
        print(f"Saved: {out_path} (Output size: {result.size})")

    print("All done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stable Diffusion 2 Inpainting with LoRA fine-tuned weights.")

    parser.add_argument("--lora_weights_path", type=str, default="lora_10",
                        help="Path to the directory containing LoRA weights (e.g., 'lora_15').")
    parser.add_argument("--input_dir", type=str, default="video/ori_frames",
                        help="Directory containing input images and masks (e.g., 'video/ori_frames').")
    parser.add_argument("--output_dir", type=str, default="inpainted_results_v2",
                        help="Directory to save the inpainted results (e.g., 'inpainted_results_15').")
    
    args = parser.parse_args()

    inference_model(
        lora_weights_path=args.lora_weights_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
