# SI-Final Project

`#去水印` `#finetune` `#LoRA`

<aside>
💡

# 實踐規劃

## #今日討論

- lama cleaner tech stack
- 挖掘LaMa，or 嘗試其他DF相關技術如(ControlNet)
    - Result: LaMa

## # Model

- [x]  目標決定
- [x]  問題定義
- [ ]  現有研究了解
    - [x]  Deep Reasearch (progressing, blocked)
    - [x]  查看結果
    - [ ]  lama cleaner (progressing)
    - [ ]  Stable Diffusion RoRA fine-tune
- [ ]  天降code複現
- [ ]  嘗試方法

## # Dataset

- [ ]  Procedure Define
- [ ]  TBD

</aside>

---

# Goal

---

使用現有影像修復模型，並以Field Data Fine-tune模型，以完成水印去除

# Problem Formulation

---

- 現有影像修復模型對於目標移除有著強大的泛化能力與可用性，但是對於大面積的污損還是能感知明顯的修復痕跡。
- 藉由從Field Data，也就是待修復的圖片，提取與污損部相似之區域以Fine-tune修復模型，以學習待修復部分的紋理，使修復能夠更加精細。

# Dataset

---

## Validation

- Source : 風景照、人像照
- 污損:  Python加水印

# **Technology Stack**

---

## 現有的影像修覆技術

`#deep reasearch TBD`

### 

# 方法規劃

---

## Pretrain

- 影像修復模型

## Process

- Field data prepare
- Fine-tune Pretrained Model
- 影像修復

# 文獻

---

[Lama-Cleaner](https://www.notion.so/Lama-Cleaner-1fb0cd9d8a8f80a5a138fea7aec918c5?pvs=21)
