import gradio as gr
from ultralytics import YOLO
import torch
import numpy as np
from utils.tools_gradio import fast_process
from utils.tools import format_results


# Load the FastSAM model
model = YOLO("./weights/FastSAM.pt")

device = torch.device("cpu")

model.to(device)


def get_input_scale(input, input_size=1024):

    input_size = int(input_size)
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))

    return input, input_size


def segment_everything(
    input,
    iou_threshold=0.9, 
    confidence_threshold=0.4
):

    input, input_size = get_input_scale(input)

    results = model(
        input,
        device=device,
        retina_masks=True,
        iou=iou_threshold,
        conf=confidence_threshold,
        imgsz=input_size,
    )

    annotations = results[0].masks.data

    fig = fast_process(
        annotations=annotations,
        image=input,
        device=device,
        scale=(1024 // input_size),
        better_quality=False,
        mask_random_color=True,
        bbox=None,
        use_retina=True,
        withContours=True,
    )

    return fig


title = "FastSAM: Fast Segment Anything"

description_e = "Demo project of FastSAM. Adapted from Ultralytics. CPU only."

examples = [
    ["examples/sa_8776.jpg"],
    ["examples/sa_414.jpg"],
    ["examples/sa_1309.jpg"],
    ["examples/sa_11025.jpg"],
    ["examples/sa_561.jpg"],
    ["examples/sa_192.jpg"],
    ["examples/sa_10039.jpg"],
    ["examples/sa_862.jpg"],
]

default_example = examples[0]

cond_img_e = gr.Image(label="Input", value=default_example[0], type="pil")
segm_img_e = gr.Image(label="Segmented Image", interactive=False, type="pil")


css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


with gr.Blocks(css=css, title="Fast Segment Anything") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

        with gr.Column(scale=1):
            # News
            gr.Markdown(description_e)

    with gr.Tab("Everything mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_e.render()

            with gr.Column(scale=1):
                segm_img_e.render()

        # Submit & Clear
        with gr.Row():

            with gr.Column():
                segment_btn_e = gr.Button(
                    "Segment Everything", variant="primary"
                )
                clear_btn_e = gr.Button("Clear", variant="secondary")

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[cond_img_e],
                    outputs=segm_img_e,
                    fn=segment_everything,
                    cache_examples=True,
                    examples_per_page=4,
                )

            with gr.Column():
                with gr.Accordion("Advanced options", open=False):
                    iou_threshold = gr.Slider(
                        0.1,
                        0.9,
                        0.7,
                        step=0.1,
                        label="iou",
                        info="iou threshold for filtering the annotations",
                    )
                    conf_threshold = gr.Slider(
                        0.1,
                        0.9,
                        0.25,
                        step=0.05,
                        label="conf",
                        info="object confidence threshold",
                    )

                # Description
                gr.Markdown(description_e)

    segment_btn_e.click(
        segment_everything,
        inputs=[cond_img_e, iou_threshold, conf_threshold],
        outputs=segm_img_e,
    )

    def clear():
        return None, None

    clear_btn_e.click(clear, outputs=[cond_img_e, segm_img_e])
    

demo.queue()
demo.launch(debug=True)