# استيراد مكتبة البحث من وحدة re (للتعبيرات النمطية)
from re import search

# استيراد مكتبة dotenv لتحميل المتغيرات البيئية من ملف .env
from dotenv import load_dotenv

# تحميل المتغيرات البيئية (مثل مفاتيح API) من ملف .env
load_dotenv()

# استيراد وظيفة create_agent من مكتبة LangChain لإنشاء وكيل ذكي
from langchain.agents import create_agent

# استيراد HumanMessage لإنشاء رسائل بشرية يتفاعل معها الوكيل
from langchain_core.messages import HumanMessage

# استيراد ChatOpenAI للتفاعل مع نماذج GPT من OpenAI
from langchain_openai import ChatOpenAI

# استيراد ديكوراتور @tool لتحويل الدوال إلى أدوات يمكن للوكيل استخدامها
from langchain.tools import tool

# استيراد TavilyClient للبحث عبر الإنترنت باستخدام محرك Tavily
from tavily import TavilyClient

# إنشاء كائن TavilyClient للتواصل مع خدمة البحث Tavily
tavily = TavilyClient()

# تعريف أداة البحث باستخدام ديكوراتور @tool
@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    
    Args:
    query: The query to search for
    
    Returns:
    The search result
    """
    # طباعة رسالة توضح ما يتم البحث عنه حالياً
    print(f"Searching for {query}")
    
    # استدعاء خدمة Tavily للبحث وإرجاع النتائج
    return tavily.search(query)

# إنشاء نموذج اللغة الكبير (LLM) باستخدام GPT-5 من OpenAI
# temperature=0 تعني أن النموذج سيكون دقيقاً وغير عشوائي
llm = ChatOpenAI(model="gpt-5", temperature=0)

# تجميع الأدوات في قائمة لتمريرها إلى الوكيل
tools = [search]

# إنشاء وكيل ذكي يجمع بين نموذج اللغة والأدوات المتاحة
agent = create_agent(model=llm, tools=tools)

# تعريف الدالة الرئيسية للبرنامج
def main():
    # طباعة رسالة ترحيبية
    print("Hello from ammar-langchain-corse-serch-agent!")
    
    # استدعاء الوكيل مع رسالة تطلب البحث عن وظائف محددة
    result = agent.invoke({
        "messages": [
            HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
            )
        ]
    })
    
    # طباعة النتيجة النهائية من الوكيل
    print(result)

# التأكد من أن هذا الملف هو الملف الرئيسي المنفذ
if __name__ == "__main__":
    # استدعاء الدالة الرئيسية
    main()