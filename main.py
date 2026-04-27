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
    Shakira Isabel Mebarak Ripoll[a] (/ʃəˈkɪərə/ shə-KEER-ə, Spanish: [ʃaˈkiɾa isaˈβel meβaˈɾak riˈpol];
    born 2 February 1977)[1] is a Colombian singer, songwriter, dancer, and record producer. Referred to
    as the "Queen of Latin Music",[b] she has had a significant impact on the musical landscape of Latin America and has been credited with popularizing Hispanophone music globally,[14] contributing to increased learning and use of the Spanish language worldwide.[15] She is also credited with opening the doors of the international market for other Latin artists.[16][17] Her accolades include 4 Grammy Awards and 15 Latin Grammy Awards.
    Shakira made her recording debut with Sony Music Colombia at the age of 14. She rose to prominence with the albums Pies Descalzos (1995) and Dónde Están los Ladrones? (1998). Her first English release, Laundry Service (2001), sold over 13 million copies worldwide,[18] becoming the best-selling album by a female Latin artist. Her success was further solidified with the Spanish-language albums Fijación Oral, Vol. 1 (2005), Sale el Sol (2010), El Dorado (2017), and Las Mujeres Ya No Lloran (2024), all of which topped the Billboard Top Latin Albums chart, making her the first woman with number-one albums across four different decades. Her English albums Oral Fixation, Vol. 2 (2005), She Wolf (2009), and Shakira (2014) achieved platinum status worldwide.
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