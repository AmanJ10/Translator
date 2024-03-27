import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

model_checkpoint = r"C:\Users\Aman Joharapurkar\OneDrive\Desktop\NLP\model"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


def translate_text(input_text):
    tokenized = tokenizer([input_text], return_tensors='tf')
    out = model.generate(**tokenized, max_length=128)
    with tokenizer.as_target_tokenizer():
        translation = tokenizer.decode(out[0], skip_special_tokens=True)
    return translation


st.title("English to Hindi Translator")

input_text = st.text_area("Enter the English text you want to translate:")

if st.button("Translate"):
    if input_text:
        translation = translate_text(input_text)
        st.write("Translated Hindi Text:")
        st.write(translation)
    else:
        st.warning("Please enter some text to translate.")
