import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

# تحميل المتغيرات البيئية من ملف .env (مثل مفاتيح API)
load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")  # بدء عملية استيراد النص
    
    # تحميل ملف نصي من المسار المحدد
    #loader = TextLoader(r"C:\Users\ammar\Desktop\langchain_projects\RAG\mediumblog1.txt", encoding="utf-8")
    loader = TextLoader("mediumblog1.txt", encoding="utf-8")# او يمكن استخدم المسار المباشر
    document = loader.load()  # قراءة محتوى الملف بالكامل

    print("splitting...")  # بدء عملية التقطيع
    
    # تقسيم النص الطويل إلى أجزاء صغيرة (chunks)
    # chunk_size = 1000: كل جزء يحتوي على 1000 حرف كحد أقصى
    # chunk_overlap = 0: لا يوجد تداخل بين الأجزاء المتجاورة
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)  # تنفيذ عملية التقطيع
    print(f"created {len(texts)} chunks")  # طباعة عدد الأجزاء الناتجة

    # إنشاء نموذج تحويل النص إلى متجهات (Embeddings) باستخدام OpenAI
    # تستخدم هذه المتجهات لتمثيل المعنى الدلالي للنص
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("ingesting...")  # بدء عملية إدخال المتجهات إلى قاعدة البيانات
    
    # إنشاء قاعدة بيانات متجهية في Pinecone وتخزين الأجزاء النصية داخلها
    # texts: الأجزاء النصية المقسمة
    # embeddings: نموذج التحويل إلى متجهات
    # index_name: اسم الفهرس (index) في Pinecone حيث سيتم التخزين
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")  # اكتملت العملية بنجاح