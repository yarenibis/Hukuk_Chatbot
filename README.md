# ⚖️ Türkçe Hukuk Chatbotu

**Retrieval-Augmented Generation (RAG) tabanlı yerel hukuk danışmanı**

---

## 🦩 Proje Amacı

Bu proje, Türk hukuk sistemine ait metinleri (soru–cevap veri seti) kullanarak çalışan bir **hukuk danışmanı chatbotu** geliştirmeyi amaçlamaktadır.  
Kullanıcı, doğal dilde bir Türkçe hukuk sorusu yazdığında sistem:

1. **Hugging Face'teki Türk hukuk veri setinden** ilgili belgeleri vektör benzerliğiyle bulur,  
2. **RAG mimarisi** sayesinde bağlamı Large Language Model'e (LLM) aktarır,  
3. **Qwen modelini** kullanarak kısa, resmi bir Türkçe yanıt üretir.

---

## 📚 Veri Seti Hakkında

### Kaynak
- **Platform:** Hugging Face Datasets
- **Veri Seti:** [Hugging Face](https://huggingface.co/datasets/alibayram/hukuk_soru_cevap)
- **Dil:** Türkçe
- **Tür:** Soru-Cevap çiftleri
- **Veri Seti Adı:** hukuk_soru_cevap 
- **Kapsam:** 2000+ Türkçe hukuk sorusu ve uzman yanıtı  

### 🔹 Genel Bakış  
Türkçe Hukuki Soru-Cevap Veri Seti, **kghukukankara.com** ve **hukuksorucevap.com.tr** sitelerinden toplanmış hukuki soru–cevap metinlerinden oluşan kapsamlı bir koleksiyondur.  
Bu veri seti, **araştırmacılar, hukuk profesyonelleri ve geliştiriciler** için doğal dil işleme (NLP) projelerinde, özellikle **hukuk alanındaki yapay zekâ uygulamalarında** kullanılmak üzere hazırlanmıştır.

### İşlemler
- Boş değerler temizlendi
- Metinler standartlaştırıldı
- Türkçe karakterler korundu
---

### 🔹 Örnek Kayıt

Aşağıda veri setinin örnek bir kısmı yer almaktadır 👇

![Veri Seti Örneği](ornek_kayit.png)



## 🛠️ Teknolojiler

### Backend
- **Python 3.8+**
- **LangChain** - RAG pipeline framework
- **Chroma DB** - Vektör veritabanı
- **Transformers** - Model yönetimi
- **Sentence Transformers** - Embedding modeli

### Modeller
- **Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **LLM:** `Qwen/Qwen3-1.7B` (yerel)
- **Text Splitter:** RecursiveCharacterTextSplitter

### Frontend
- **Streamlit** - Web arayüzü

## 🏗️ Mimari

```text
Kullanıcı Sorusu
        │
        ▼
 ┌──────────────────────┐
 │  Chroma Retriever    │
 │ (Embedding Search)   │
 └──────────────────────┘
        │
        ▼
 ┌──────────────────────┐
 │  RAG Pipeline        │
 │ (LangChain + Qwen)   │
 └──────────────────────┘
        │
        ▼
 ┌──────────────────────┐
 │  Türkçe Yanıt Üretimi│
 └──────────────────────┘
        │
        ▼
Streamlit Arayüzü (Chat UI)

---

### RAG Pipeline
### Bileşenler

#### 1. Veri Hazırlama
- Hugging Face dataset yükleme
- Metin temizleme ve normalleştirme
- Vektör veritabanı oluşturma

#### 2. Retrieval Sistemi
- Çok dilli embedding modeli
- Chroma vektör veritabanı
- Benzerlik tabanlı belge arama

#### 3. Generation Sistemi
- Qwen3-1.7B modeli
- Türkçe prompt optimizasyonu
- Bağlamsal yanıt üretimi

#### 4. Kullanıcı Arayüzü
- Streamlit tabanlı web arayüzü
- Gerçek zamanlı sohbet
- Geçmiş saklama

## 📊 Sonuçlar

### Başarılar
- ✅ Türkçe hukuk terminolojisi işleme
- ✅ Bağlama duyarlı yanıtlar
- ✅ Yerel çalışma (internet gerekmez)
- ✅ Kullanıcı dostu arayüz
- ✅ Hızlı yanıt süreleri

### Performans
- **Doğruluk:** Bağlamsal tutarlılık
- **Süre:** 5-15 saniye
- **Dil:** %100 Türkçe yanıt
- **Bağlam:** Otomatik bilgi çekme

  
## 🚀 Kurulum
1. **Repository klonlama**
```bash
git clone <repository-url>
cd turkce-hukuk-chatbot

2. **Sanal ortam oluşturma**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

3. **Paketleri yükleme**
```bash
pip install -r requirements.txt

3. **Uygulamayı çalıştır**
```bash
streamlit run app.py

