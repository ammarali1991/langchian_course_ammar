import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

# تحميل المتغيرات البيئية من ملف .env، مثل مفتاح OpenAI واسم فهرس Pinecone
load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")  # بدء عملية إدخال البيانات إلى Pinecone

    # تحميل ملف النص من المشروع
    # يمكن استخدام مسار كامل إذا كان الملف في مكان آخر
    loader = TextLoader("mediumblog1.txt", encoding="utf-8")
    document = loader.load()  # قراءة الملف كنائن مستند

    print("splitting...")  # بدء تقطيع النص إلى أجزاء صغيرة

    # تقسيم النص إلى أجزاء بطول 1000 حرف بدون تداخل
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)  # قائمة الأجزاء النصية
    print(f"created {len(texts)} chunks")  # طباعة عدد الأجزاء الناتجة

    # إنشاء نموذج تحويل النص إلى متجهات باستخدام OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("ingesting...")  # بدء حفظ المتجهات في Pinecone

    # تحميل النصوص كوثائق إلى Pinecone لتكون قابلة للاسترجاع لاحقًا
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")  # انتهت عملية الاستيراد بنجاح