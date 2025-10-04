import gradio as gr
from Inference import load_pipeline, inference_fn 


path = "./opus-mt-en-hi-4-bit-quantized-finetuned-model"
translator = load_pipeline(path) 


with gr.Blocks() as demo:
    gr.Markdown("# English TO Hindi Translation Web App")
    name = gr.Textbox(label="English")
    output = gr.Textbox(label="Hindi")
    greet_btn = gr.Button("Translate")
    greet_btn.click(fn=lambda English: inference_fn(English, translator), inputs=name, outputs=output, api_name="Translate")

# demo = gr.Interface(
#     fn=lambda English: inference_fn(English, translator), 
#     inputs="textbox",
#     outputs="textbox"
# )
if __name__=="__main__":
    demo.launch()