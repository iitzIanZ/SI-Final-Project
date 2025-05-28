import torch
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Install necessary packages
# subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

def debug_show(img, title, out_dir="debug", idx=None):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(4,4))
    # BGR→RGB 或 BGRA→RGBA
    if img.shape[2] == 4:
        disp = cv2.cvtColor(img, cv2.COLOR_BGRA_RGBA)
    else:
        disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(disp)
    plt.title(title)
    plt.axis("off")
    plt.show()
    fn = title.replace(" ", "_")
    if idx is not None:
        fn = f"{idx:02d}_{fn}"
    cv2.imwrite(os.path.join(out_dir, fn + ".png"), img)


class WatermarkRemover:
    def __init__(self, model_id='microsoft/Florence-2-large'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize Florence model
        self.florence_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(self.device).eval()
        self.florence_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        # Initialize Llama Cleaner model
        self.model_manager = ModelManager(name="lama", device=self.device)

    def generate_mask_florence(self, image_path: str):
        """
        只跑 Florence，回傳：
          - binary mask: np.ndarray shape=(H,W)，值為 0/1
          - bbox: (x1, y1, x2, y2)
        """
        # 1. 載入影像
        img = Image.open(image_path).convert("RGB")

        # 2. 呼叫 Florence 做 segmentation
        prompt = '<REGION_TO_SEGMENTATION>watermark'
        inputs = self.florence_processor(
            text=prompt, images=img, return_tensors="pt"
        ).to(self.device)
        gen_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        gen_text = self.florence_processor.batch_decode(
            gen_ids, skip_special_tokens=False
        )[0]
        parsed = self.florence_processor.post_process_generation(
            gen_text,
            task='<REGION_TO_SEGMENTATION>',
            image_size=(img.width, img.height)
        )

        # 3. 轉成 RGBA，再 extract alpha → binary mask
        mask_rgba = self.create_mask(img, parsed['<REGION_TO_SEGMENTATION>'])
        mask_np = np.array(mask_rgba)[:,:,3]  # alpha channel
        binary_mask = (mask_np > 0).astype(np.uint8)

        # 4. 計算 bbox
        ys, xs = np.where(binary_mask)
        if len(xs)==0 or len(ys)==0:
            return binary_mask, (0,0,0,0)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        return binary_mask, (x1, y1, x2, y2)
    
    def process_image(self, image, mask, strategy=HDStrategy.RESIZE, sampler=LDMSampler.ddim, fx=1, fy=1):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if fx != 1 or fy != 1:
            image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
        
        config = Config(
            ldm_steps=1,
            ldm_sampler=sampler,
            hd_strategy=strategy,
            hd_strategy_crop_margin=32,
            hd_strategy_crop_trigger_size=200,
            hd_strategy_resize_limit=200,
        )

        result = self.model_manager(image, mask, config)
        return result

    def create_mask(self, image, prediction):
        # mask = Image.new("RGBA", image.size, (0, 0, 0, 255))  # Black background
        mask = Image.new("RGBA", image.size, (0, 0, 0, 0))  # Black background
        draw = ImageDraw.Draw(mask)
        scale = 1
        for polygons in prediction['polygons']:
            for _polygon in polygons:
                _polygon = np.array(_polygon).reshape(-1, 2)
                if len(_polygon) < 3:
                    continue
                _polygon = (_polygon * scale).reshape(-1).tolist()
                draw.polygon(_polygon, fill=(255, 255, 255, 255))  # Make selected area white
        return mask

    def process_images_florence_lama(self, input_image_path, output_image_path):
        # Load input image
        image = Image.open(input_image_path).convert("RGB")
        
        # Convert image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run Florence to get mask
        text_input = 'watermark'  # Teks untuk Florence agar mengenali watermark
        task_prompt = '<REGION_TO_SEGMENTATION>'
        inputs = self.florence_processor(text=task_prompt + text_input, images=image, return_tensors="pt").to(self.device)
        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.florence_processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        
        # Create mask and process image with Llama Cleaner
        mask_image = self.create_mask(image, parsed_answer['<REGION_TO_SEGMENTATION>'])
        result_image = self.process_image(image_cv, np.array(mask_image), HDStrategy.RESIZE, LDMSampler.ddim)
        
        # Convert result back to PIL Image
        result_image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        
        # Save output image
        result_image_pil.save(output_image_path)

    def process_batch(self, input_dir, output_dir, max_workers=4):
        input_images = [os.path.join(input_dir, img) for img in os.listdir(input_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        output_images = [os.path.join(output_dir, os.path.basename(img)) for img in input_images]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.process_images_florence_lama, input_images, output_images)
