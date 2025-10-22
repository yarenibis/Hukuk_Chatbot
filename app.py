# -*- coding: utf-8 -*-
import os
import re
import sys
import traceback
import torch
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
# LangChain: Embedding + Vector Store
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
# Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM



# ─────────────────────────────
# 0️⃣ Ortam Değişkenleri
# ─────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

APP_TITLE = "⚖️ Türkçe Hukuk Chatbotu"
APP_CAPTION = "Türkçe hukuk sorularını Qwen3-1.7B modeliyle yanıtlar."

# Hugging Face’ten alınan Türkçe hukuk veri seti
DEFAULT_DATASET = "alibayram/hukuk_soru_cevap"
DEFAULT_SPLIT = "train"

# Embedding ve LLM model bilgileri
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_LLM_MODEL = "Qwen/Qwen3-1.7B"
CHROMA_DIR = "chroma_hukuk_db"

# RAG parametreleri
RETRIEVER_K = 6
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100


# ─────────────────────────────
# 1️⃣ Dataset ve Chroma Vektör Veritabanı Hazırlığı
# ─────────────────────────────
@st.cache_data(show_spinner=False)
def load_hf_df(dataset_name: str, split: str) -> pd.DataFrame:
    """
       Hugging Face üzerindeki veri setini yükler ve 'soru' ile 'cevap' sütunlarını düzenler.
    """
    ds = load_dataset(dataset_name, split=split, token=HF_TOKEN)
    df = ds.to_pandas()
    df = df.rename(columns={c.lower(): c for c in df.columns})
    if "soru" not in df.columns or "cevap" not in df.columns:
        raise ValueError("Dataset'te 'soru' veya 'cevap' kolonu yok.")
    df["soru"] = df["soru"].astype(str).str.strip()
    df["cevap"] = df["cevap"].astype(str).str.strip()
    return df.dropna(subset=["soru", "cevap"]).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def build_or_load_chroma(df: pd.DataFrame, persist_dir: str):
    """
        Chroma vektör veritabanını oluşturur veya mevcutsa yükler.
        Embedding modeli olarak çok dilli MiniLM kullanılır.
    """
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # Eğer zaten oluşturulmuş bir Chroma veritabanı varsa onu yükler
    if any(f.endswith(".sqlite3") for f in os.listdir(persist_dir)):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    # Yeni vektör veritabanı oluşturma süreci
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = [Document(page_content=f"Soru: {r['soru']}\n\nCevap: {r['cevap']}") for _, r in df.iterrows()]
    chunks = [Document(page_content=c) for d in docs for c in splitter.split_text(d.page_content)]
    return Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)



# ─────────────────────────────
# 2️⃣ Qwen3-1.7B Modeli
# ─────────────────────────────
@st.cache_resource(show_spinner=False)
def load_qwen_model(model_name=LOCAL_LLM_MODEL):
    """Qwen3-1.7B chat modelini yükler (CPU uyumlu)."""
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to("cpu").eval()
    return tokenizer, model



# 🔹 Qwen modellerinin reasoning (<think>) kısmını temizlemek için yardımcı fonksiyon
def clean_output(text: str) -> str:
    """
    Modelin ürettiği <think> bölümlerini temizler (içsel düşünceleri gizler).
    """
    # <think> ile başlayan tüm bölümleri sil
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    # Boşlukları düzelt
    return text.strip()



# 🔹 Kullanıcı mesajını Qwen chat formatına uygun hale getirir
def create_messages(question, context):
    """
        Qwen chat formatına uygun şekilde system + user mesajları oluşturur.
        Türkçe cevapları zorunlu kılar ve modelin içsel düşünce üretmesini engeller.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Sen bir TÜRK hukuk danışmanısın. "
                "Cevaplarını her zaman TÜRKÇE olarak ver. "
                "İngilizce veya başka bir dilde asla cevap verme. "
                "Asla içsel düşünce (<think>), açıklama veya adım adım mantık yazma. "
                "Sadece nihai cevabı yaz. "
                "Kullanıcıya yalnızca verilen bağlamdaki bilgilere dayanarak kısa, resmi ve doğru bir TÜRKÇE hukuk cevabı ver. "
                "Yorum yapma, tahmin etme, gereksiz tekrar etme."
                "Cevap en fazla 2-3 paragraf uzunluğunda olsun."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Bu soru Türkçe yazılmıştır. Lütfen yanıtı da TÜRKÇE ver.\n\n"
                f"BAĞLAM:\n{context}\n\nSORU:\n{question}"
            ),
        },
    ]
    return messages


# 🔹 Modelden yanıt üretme fonksiyonu
def generate_answer(model, tokenizer, question, context, max_new_tokens=400):
    """
    Soru + bağlam bilgisini alır, modelden yanıt üretir ve <think> kısmını temizler.
    """
    messages = create_messages(question, context)
    # Qwen modelleri için doğru encoding:
    text_input = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    # Bazı modeller dict yerine direkt tensor döndürür → kontrol ediyoruz
    if isinstance(text_input, torch.Tensor):
        inputs = {"input_ids": text_input.to(model.device)}
    else:
        inputs = text_input.to(model.device)
    # Modelden yanıt üretimi (temperature düşük = daha kararlı Türkçe cevap)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4
        )
    # Yalnızca yeni üretilen kısmı çözüyoruz
    input_length = inputs["input_ids"].shape[-1]
    text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    text = clean_output(text)  # 💥 yeni satır
    return text.strip() or "Yanıt üretilemedi."



# 🔹 Bağlamı (context) belirli uzunlukta birleştirir
def build_context(docs, max_len=6000):
    """
    Arama sonucu dönen belgeleri birleştirip modelin anlayacağı şekilde bir context oluşturur.
    """
    buf, total = [], 0
    for d in docs:
        if total + len(d.page_content) > max_len:
            break
        buf.append(d.page_content)
        total += len(d.page_content)
    return "\n\n".join(buf)


# ─────────────────────────────
# 3️⃣ Streamlit Arayüzü
# ─────────────────────────────
def main():
    # Sayfa başlığı ve ikon
    st.set_page_config(page_title=APP_TITLE, page_icon="⚖️")
    st.title(APP_TITLE)
    st.caption(APP_CAPTION)

    # Dataset yükleniyor
    with st.spinner("📂 Dataset yükleniyor..."):
        df = load_hf_df(DEFAULT_DATASET, DEFAULT_SPLIT)

    # Chroma veritabanı hazırlanıyor
    with st.spinner("📚 Chroma veritabanı hazırlanıyor..."):
        vectordb = build_or_load_chroma(df, CHROMA_DIR)
    retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVER_K})
    # Model yükleniyor
    with st.spinner(f"🤖 Model yükleniyor ({LOCAL_LLM_MODEL})..."):
        tokenizer, model = load_qwen_model(LOCAL_LLM_MODEL)
    # Sohbet geçmişi yönetimi
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Eski mesajları arayüzde göster
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    # Kullanıcıdan yeni soru al
    question = st.chat_input("Bir hukuk sorusu yazın… (örnek:... Nafaka hakkım var mı?)")
    if question:
        # Kullanıcı mesajını ekrana yaz
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        # Belgelerden bağlam oluştur ve modelden yanıt al
        with st.spinner("🧠 Belgeler aranıyor ve yanıt hazırlanıyor..."):
            docs = retriever.invoke(question)
            context = build_context(docs)
            answer = generate_answer(model, tokenizer, question, context)
        # Asistan cevabını ekrana yaz
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
