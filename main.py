import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# تحميل المتغيرات البيئية من ملف .env
load_dotenv()

print("Initializing components...")

# إنشاء نموذج تحويل النص إلى متجهات
embeddings = OpenAIEmbeddings()
# إنشاء نموذج المحادثة الخاص بـ OpenAI
llm = ChatOpenAI()

# إنشاء اتصال بـ Pinecone عبر اسم الفهرس من المتغيرات البيئية
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

# تحويل المتجهات إلى كائن قابل للاسترجاع
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# قالب السؤال مع سياق المستندات المسترجعة
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    """تحويل المستندات المسترجعة إلى نص واحد مفصول بأسطر فارغة."""
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# ============================================================================
def retrieval_chain_without_lcel(query: str):
    """
    سلسلة استرجاع بسيطة بدون LCEL.
    تسترجع المستندات وتعيد تنسيقها ثم تستدعي الموديل.
    """
    # الخطوة 1: استرجاع المستندات الأكثر صلة
    docs = retriever.invoke(query)

    # الخطوة 2: تحويل المستندات إلى نص سياقي واحد
    context = format_docs(docs)

    # الخطوة 3: إضافة السياق إلى قالب الرسالة
    messages = prompt_template.format_messages(context=context, question=query)

    # الخطوة 4: استدعاء الموديل للحصول على الإجابة
    response = llm.invoke(messages)

    # الخطوة 5: إعادة نص الإجابة فقط
    return response.content


# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================
def create_retrieval_chain_with_lcel():
    """
    إنشاء سلسلة استرجاع باستخدام LCEL.
    تعيد سلسلة جاهزة للاستدعاء مع السؤال فقط.
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")

    # نص الاستعلام الذي سيتم استخدامه في التجربة
    query = "what is Pinecone in machine learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer: with out rag")
    print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    print("- Better for production use")
    print("=" * 70)
     # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
    