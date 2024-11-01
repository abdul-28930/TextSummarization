from youtube_transcript_api import YouTubeTranscriptApi
import re
import torch
import gradio as gr
from transformers import pipeline


# text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

model_path = "C:\\Users\\abdul\\Documents\\genaiproj\\genai\\Models\\models--sshleifer--distilbart-cnn-12-6\\snapshots\\a4f8f3ea906ed274767e9906dbaede7531d660ff"
text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.bfloat16)


def summary(input_text):
    # Check if input_text is too long
    max_input_length = 1024  # Adjust based on your model's max input length
    if len(input_text) > max_input_length:
        input_text = input_text[:max_input_length]
    
    try:
        output = text_summary(input_text)
        return output[0]['summary_text']
    except IndexError as e:
        print(f"Error summarizing text: {e}")
        return None

# Function to extract video ID from YouTube URL
def get_video_id(url):
    video_id = re.search(r"(?<=v=)[^&#]+", url) or re.search(r"(?<=be/)[^&#]+", url)
    return video_id.group(0) if video_id else None

# Function to fetch transcript
def fetch_transcript(url):
    video_id = get_video_id(url)
    if video_id:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = "\n".join([entry['text'] for entry in transcript])
        summary_text = summary(transcript_text)
        return summary_text
    else:
        return "Invalid YouTube URL"

# Input URL
# url = input("Enter YouTube URL: ")
# print(fetch_transcript(url))


gr.close_all()

# demo = gr.Interface(fn=summary, inputs="text", outputs="text")

demo = gr.Interface(fn=fetch_transcript, 
inputs=[gr.Textbox(lines=2, label="Input Youtube URL to summarize")], 
outputs=[gr.Textbox(lines=7, label="Summarized Text")], 
title="Youtube Script Summarization", 
theme="soft",
description="Summarize any Youtube Videos in seconds!")
demo.launch(share=True)