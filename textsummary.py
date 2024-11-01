import torch
import gradio as gr
from transformers import pipeline


text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

# model_path = "C:\\Users\\abdul\\Documents\\genaiproj\\genai\\Models\\models--sshleifer--distilbart-cnn-12-6\\snapshots\\a4f8f3ea906ed274767e9906dbaede7531d660ff"
# text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.bfloat16)

# text="Google was founded on September 4, 1998, by American computer scientists Larry Page and Sergey Brin while they were PhD students at Stanford University in California. Together, they own about 14% of its publicly listed shares and control 56% of its stockholder voting power through super-voting stock. The company went public via an initial public offering (IPO) in 2004. In 2015, Google was reorganized as a wholly owned subsidiary of Alphabet Inc. Google is Alphabet's largest subsidiary and is a holding company for Alphabet's internet properties and interests. Sundar Pichai was appointed CEO of Google on October 24, 2015, replacing Larry Page, who became the CEO of Alphabet. On December 3, 2019, Pichai also became the CEO of Alphabet."
# print(text_summary(text))


def summary (input):
    output = text_summary(input)
    return output[0]['summary_text']

gr.close_all()

# demo = gr.Interface(fn=summary, inputs="text", outputs="text")

demo = gr.Interface(fn=summary, 
inputs=[gr.Textbox(lines=7, label="Input Text")], 
outputs=[gr.Textbox(lines=5, label="Summarized Text")], 
title="Text Summarization", 
theme="soft",
description="Summarize your text using the model distilbart-cnn-12-6")
demo.launch(share=True)