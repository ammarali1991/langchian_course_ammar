# استيراد مكتبة البحث من وحدة re (للتعبيرات النمطية)
from re import search
from typing import List
from pydantic import BaseModel, Field

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
#from tavily import TavilyClient
from langchain_tavily import TavilySearch
class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )

# إنشاء نموذج اللغة الكبير (LLM) باستخدام GPT-5 من OpenAI
# temperature=0 تعني أن النموذج سيكون دقيقاً وغير عشوائي
llm = ChatOpenAI(model="gpt-5", temperature=0)
#test
# تجميع الأدوات في قائمة لتمريرها إلى الوكيل
tools = [TavilySearch()]  # إضافة أداة البحث عبر الإنترنت إلى قائمة الأدوات

# إنشاء وكيل ذكي يجمع بين نموذج اللغة والأدوات المتاحة
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
# تعريف الدالة الرئيسية للبرنامج
def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
            )
        }
    )
    print(result)
    

# التأكد من أن هذا الملف هو الملف الرئيسي المنفذ
if __name__ == "__main__":
    # استدعاء الدالة الرئيسية
    main()