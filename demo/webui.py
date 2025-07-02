import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

def segment_with_points(img, points_and_labels):
    if img is None or not points_and_labels:
        return None, None
    image = np.array(img.convert("RGB"))
    predictor.set_image(image)
    pts = np.array([[x, y] for x, y, label in points_and_labels])
    labels = np.array([label for x, y, label in points_and_labels])
    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=labels,
        multimask_output=True,
    )
    best_idx = np.argmax(scores)
    mask = masks[best_idx]
    mask_bool = mask.astype(bool)
    mask_img = image.copy()
    mask_img[mask_bool] = [255, 0, 0]
    vis_img = Image.fromarray(mask_img)

    # 返回单通道掩码图像（PIL），尺寸与原图一致
    mask_img_pil = Image.fromarray((mask_bool * 255).astype(np.uint8)).resize(img.size)
    return vis_img, mask_img_pil


def draw_points(img, points_and_labels):
    if img is None:
        return None
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for x, y, label in points_and_labels:
        color = "green" if label == 1 else "red"
        r = 6
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    return img


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # SAM2 点选分割 Demo

        **Developer: Henry Wu. For new requirements or issues, please contact.**

        <span style="color:gray;font-size:14px;">
        1. 上传图片<br>
        2. 选择点类型（前景点/背景点）<br>
        3. 在图片上点击添加点，绿色为前景，红色为背景<br>
        4. 点击“分割”按钮进行分割，点击“重置点”清空所有点
        </span>
        """,
        elem_id="header"
    )
    gr.Markdown("---")

    with gr.Row():
        point_type = gr.Radio(
            ["前景点", "背景点"],
            value="前景点",
            label="点类型",
            info="请选择当前要添加的点类型"
        )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="上传图片", height=320)
            with gr.Row():
                clear_btn = gr.Button("重置点", size="sm")
                seg_btn = gr.Button("分割", size="sm")
        with gr.Column(scale=1):
            img_output = gr.Image(type="pil", label="分割结果", height=320)
            mask_file = gr.Image(type="pil", label="掩码编辑与下载", format="png", height=320)

    state = gr.State([])
    orig_img_state = gr.State(None)  # 新增：保存原始图片

    def on_upload(img):
        return img, img  # state, orig_img_state

    img_input.upload(
        on_upload,
        inputs=img_input,
        outputs=[img_input, orig_img_state]
    )

    def on_click(evt: gr.SelectData, points_and_labels, img, point_type):
        label = 1 if point_type == "前景点" else 0
        points_and_labels = points_and_labels.copy()
        points_and_labels.append((evt.index[0], evt.index[1], label))
        return points_and_labels, draw_points(img, points_and_labels)

    img_input.select(
        on_click,
        inputs=[state, img_input, point_type],
        outputs=[state, img_input]
    )

    def on_clear(orig_img):
        return [], orig_img  # 清空点，恢复原图

    clear_btn.click(
        on_clear,
        inputs=orig_img_state,
        outputs=[state, img_input]
    )

    seg_btn.click(
        segment_with_points,
        inputs=[orig_img_state, state],  # 用原图做输入
        outputs=[img_output, mask_file]
    )

demo.launch(server_name="0.0.0.0", server_port=7861)
