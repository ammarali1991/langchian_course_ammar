import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    print(os.getenv("OPENAI_API_KEY"))
    
    # البيانات المدخلة (Input Data)
    information = """
    ammar  (/ˈiːlɒn/ ammar alchaeeb-lon; born June 28, 1991 in lattakia with white color) is a businessman...
    """

    # ═══════════════════════════════════════════════════════════════════════
    # المفهوم الأساسي (1): القالب (PromptTemplate)
    # ═══════════════════════════════════════════════════════════════════════
    # القالب هو "هيكل جاهز" للسؤال، نترك فيه مكاناً فارغاً {information}
    # سيتم ملؤه لاحقاً بالبيانات الحقيقية
    # ═══════════════════════════════════════════════════════════════════════
    summary_template = """
    given the information {information} about a person I want you to create:
    1. name, age, height, weight, and skin color
    2. top 3 traits (only from the AI's perspective)
    """

    # نربط القالب باسم المتغير الفارغ الموجود بداخله
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], 
        template=summary_template
    )

    # ═══════════════════════════════════════════════════════════════════════
    # المفهوم الأساسي (2): نموذج اللغة (LLM) + درجة الحرارة (Temperature)
    # ═══════════════════════════════════════════════════════════════════════
    # temperature=0 تعني: التزم بالنص المعطى ولا تبتكر من عندك (دقة وليس إبداع)
    # ═══════════════════════════════════════════════════════════════════════
    # llm = ChatOllama(temperature=0, model="gemma3:270m")  # نموذج محلي
    llm = ChatOpenAI(temperature=0, model="gpt-5")          # نموذج OpenAI السحابي

    # ═══════════════════════════════════════════════════════════════════════
    # المفهوم الأساسي (3): السلسلة (Chain) باستخدام عامل التوصيل (|)
    # ═══════════════════════════════════════════════════════════════════════
    # الرمز | يعني: خذ مخرجات اليمين ومررها كمدخلات لليسار
    # ترتيب التنفيذ: 1- املأ القالب بالبيانات -> 2- أرسل النص المكتمل للنموذج
    # ═══════════════════════════════════════════════════════════════════════
    chain = summary_prompt_template | llm

    # ═══════════════════════════════════════════════════════════════════════
    # المفهوم الأساسي (4): تنفيذ السلسلة (Invoke)
    # ═══════════════════════════════════════════════════════════════════════
    # المفتاح "information" (نص ثابت) = اسم المكان الفارغ في القالب
    # القيمة information (متغير) = البيانات الحقيقية (نص إيلون ماسك)
    # ═══════════════════════════════════════════════════════════════════════
    response = chain.invoke(input={"information": information})
    
    # طباعة النص الناتج فقط من كائن الرد
    print(response.content)
    #نهاية الجزء الاول سكشن 2 ammar branch section2


if __name__ == "__main__":
    main()