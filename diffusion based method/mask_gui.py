import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os

class MaskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mask Generator")
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image = None
        self.mask = None
        self.tk_img = None
        self.draw = None
        self.last_x = None
        self.last_y = None
        self.brush_size = 20
        self.filename = None
        self.display_size = (0, 0)  # 新增：記錄顯示區域大小
        self.scale = 1.0            # 新增：縮放比例
        self.last_paint = None  # 新增：記錄上次塗抹座標

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Button(btn_frame, text="載入圖片", command=self.load_image).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="儲存Mask", command=self.save_mask).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="清除塗抹", command=self.clear_mask).pack(side=tk.LEFT)

        # Drag & Drop support
        self.root.drop_target_register('DND_Files')
        self.root.dnd_bind('<<Drop>>', self.drop_image)

        # Mouse events
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.paint)
        self.canvas.bind('<B3-Motion>', self.erase)
        self.canvas.bind('<Button-3>', self.erase)
        self.canvas.bind('<Configure>', self.on_resize)
        self.canvas.bind('<ButtonRelease-1>', self.reset_last)
        self.canvas.bind('<ButtonRelease-3>', self.reset_last)

    def load_image(self):
        file = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file:
            self.open_image(file)

    def drop_image(self, event):
        files = self.root.tk.splitlist(event.data)
        if files:
            self.open_image(files[0])

    def open_image(self, file):
        self.filename = file
        self.image = Image.open(file).convert('RGB')
        self.mask = Image.new('L', self.image.size, 0)
        self.draw = ImageDraw.Draw(self.mask)
        self.show_image()

    def on_resize(self, event):
        if self.image is not None:
            self.display_size = (event.width, event.height)
            self.show_image()

    def show_image(self):
        if self.image is None:
            return
        # 根據canvas大小縮放圖片
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width < 10 or canvas_height < 10:
            return
        img_w, img_h = self.image.size
        scale = min(canvas_width / img_w, canvas_height / img_h)
        self.scale = scale
        new_size = (int(img_w * scale), int(img_h * scale))
        img_resized = self.image.resize(new_size, Image.LANCZOS)
        mask_resized = self.mask.resize(new_size, Image.NEAREST)
        mask_rgb = Image.merge('RGB', [mask_resized]*3)
        display = Image.blend(img_resized, mask_rgb, 0.5)
        self.tk_img = ImageTk.PhotoImage(display)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def paint(self, event):
        if self.mask is None:
            return
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)
        if self.last_paint is not None:
            self.draw.line([self.last_paint, (x, y)], fill=255, width=self.brush_size*2)
        self.draw.ellipse((x-self.brush_size, y-self.brush_size, x+self.brush_size, y+self.brush_size), fill=255)
        self.last_paint = (x, y)
        self.show_image()

    def erase(self, event):
        if self.mask is None:
            return
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)
        if self.last_paint is not None:
            self.draw.line([self.last_paint, (x, y)], fill=0, width=self.brush_size*2)
        self.draw.ellipse((x-self.brush_size, y-self.brush_size, x+self.brush_size, y+self.brush_size), fill=0)
        self.last_paint = (x, y)
        self.show_image()

    def reset_last(self, event):
        self.last_paint = None

    def save_mask(self):
        if self.mask is None:
            messagebox.showerror("錯誤", "尚未載入圖片")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if save_path:
            self.mask.save(save_path)
            messagebox.showinfo("完成", f"已儲存: {save_path}")

    def clear_mask(self):
        if self.mask is not None:
            self.mask.paste(0, [0,0,self.mask.size[0],self.mask.size[1]])
            self.show_image()

if __name__ == "__main__":
    try:
        import tkinterdnd2 as tkdnd
        class DnDApp(tkdnd.TkinterDnD.Tk, MaskApp):
            def __init__(self):
                tkdnd.TkinterDnD.Tk.__init__(self)
                MaskApp.__init__(self, self)
        app = DnDApp()
        app.mainloop()
    except ImportError:
        root = tk.Tk()
        MaskApp(root)
        root.mainloop()
