import gradio as gr
import decoders


# define the layout
with gr.Blocks() as demo:
    with gr.Tab("Greedy Search"):
        with gr.Row():
            with gr.Column(scale=1):
                max_length = gr.Number(
                    value=100,
                    label="Maximum length",
                    interactive=True,
                    precision=0
                )
                generate = gr.Button(
                    value="Generate"
                )
            with gr.Column(scale=4):
                starter = gr.Textbox(
                    lines=10,
                    max_lines=20,
                    placeholder="Type in some sentence as starter: ",
                    label="Starter",
                    interactive=True
                )
                generated = gr.Textbox(
                    lines=10,
                    max_lines=20,
                    placeholder="Generated text: ",
                    label="Generated text",
                )
        generate.click(
            fn=decoders.greedy_search,
            inputs=[starter, max_length],
            outputs=generated
        )
    with gr.Tab("Beam Search"):
        with gr.Row():
            with gr.Column(scale=1):
                max_length = gr.Number(
                    value=100,
                    label="Maximum length",
                    interactive=True,
                    precision=0
                )
                num_beams = gr.Number(
                    value=5,
                    label="Number of beams",
                    interactive=True,
                    precision=0
                )
                no_repeat_ngram_size = gr.Number(
                    value=2,
                    label="No repeat n-gram size",
                    interactive=True,
                    precision=0
                )
                num_return_sequences = gr.Slider(
                    minimum=1,
                    maximum=3,
                    value=1,
                    step=1,
                    label="Number of returned sequences",
                    interactive=True
                )
                index_return_sequences = gr.Slider(
                    minimum=1,
                    maximum=3,
                    value=1,
                    step=1,
                    label="Index of returned sequences",
                    interactive=True
                )
                keywords = gr.Textbox(
                    max_lines=1,
                    placeholder="keywords",
                    label="keywords",
                    interactive=True
                )
                is_early_stopping = gr.Radio(
                    choices=["True", "False"],
                    value="True",
                    type="value",
                    label="Is early stopping",
                    interactive=True
                )
                generate = gr.Button(
                    value="Generate"
                )
            with gr.Column(scale=4):
                starter = gr.Textbox(
                    lines=10,
                    max_lines=20,
                    placeholder="Type in some sentence as starter: ",
                    label="Starter",
                    interactive=True
                )
                generated = gr.Textbox(
                    lines=10,
                    max_lines=20,
                    placeholder="Generated text: ",
                    label="Generated text",
                )
        generate.click(
            fn=decoders.beam_search,
            inputs=[
                starter, max_length, num_beams, no_repeat_ngram_size,
                num_return_sequences, index_return_sequences,
                is_early_stopping, keywords
            ],
            outputs=generated
        )
    with gr.Tab("Sampling"):
        with gr.Row():
            with gr.Column(scale=1):
                max_length = gr.Number(
                    value=100,
                    label="Maximum length",
                    interactive=True,
                    precision=0
                )
                top_k = gr.Number(
                    value=0,
                    label="Top-k",
                    interactive=True,
                    precision=0
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    label="Top-p",
                    interactive=True
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature",
                    interactive=True
                )
                is_do_sample = gr.Radio(
                    choices=["True", "False"],
                    value="True",
                    type="value",
                    label="Is sampling",
                    interactive=True
                )
                generate = gr.Button(
                    value="Generate",
                )
            with gr.Column(scale=4):
                starter = gr.Textbox(
                    lines=10,
                    max_lines=20,
                    placeholder="Type in some sentence as starter: ",
                    label="Starter",
                    interactive=True
                )
                generated = gr.Textbox(
                    lines=10,
                    max_lines=20,
                    placeholder="Generated text: ",
                    label="Generated text",
                )
        generate.click(
            fn=decoders.sampling,
            inputs=[starter, max_length, is_do_sample, top_k, temperature, top_p],
            outputs=generated
        )

if __name__ == "main":
    demo.launch()
