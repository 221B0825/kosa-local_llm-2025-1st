import os
import streamlit as st
import openai
from huggingface_hub import HfApi, login

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch

@st.cache_resource
def get_model_and_tokenizer(model_id):

    login(token=os.getenv("HUGGINGFACE_TOKEN"))    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return model, tokenizer

def translate(model, tokenizer, korean_text):
    new_messages = [
        {
            "role": "system",
            "content": "You are a translation expert. Please provide an English translation of the user's Korean sentence."
        },
        {
            "role": "user",
            "content": korean_text
        },
    ]
    prompt = tokenizer.apply_chat_template(new_messages, tokenize=False, add_generation_prompt=True) 
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer)

    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    return streamer, len(prompt)

def main():
    model, tokenizer = get_model_and_tokenizer("bardroh/ko-en-translate-qwen2-0.5")

    colKorean, colEnglish = st.columns(2)

    with colKorean:
        st.header('입력')
        korean_text = st.text_area("번역할 한국어 텍스트를 입력하세요", height=300)

    with colEnglish:
        st.header('출력')
        english_text = st.empty()
        
    if st.button('번역'):
        if korean_text:
            gen, length = translate(model, tokenizer, korean_text)
            full_length = 0
            full_response = ""
            is_first_printable_chunk = True
            for chunk in gen:
                full_length += len(chunk)
                if full_length > length:
                    if is_first_printable_chunk:
                        remains = full_length - length
                        chunk = chunk[-remains:]
                        is_first_printable_chunk = False
                    chunk = chunk.replace("<|im_end|>", "")
                    full_response += chunk
                    english_text.markdown(full_response + "▌")
            english_text.markdown(full_response)
        else:
            st.warning('번역할 텍스트를 입력하세요')

if __name__ == "__main__":
    main()
