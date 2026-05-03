from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage,ToolMessage
from langsmith import traceable
MAX_ITERATIONS = 10 #تمثل اليمت لعدد مرات تنفيذ النموذج قبل اجبارة على اعطاء الجواب  بهدف منع دخولة في حلقة لانهائية 
MODEL="qwen3:1.7b"
@tool

def get_product_price(product_name:str)->float:
    """"look up the price of a product on the catlog."""
    print(f"looking up the price of (product='{product_name}')...")
    prices={
        "laptop": 999.99,
        "smartphone": 499.99,
        "headphones": 199.99,
    }
    return prices.get(product_name, 0)
@tool

def get_apply_discount(price: float, discount_tier: str) -> float:
    """"apply a discount  tier to a price and return the final price.
    available tiers:gold,bronze,silver
    """
    print(f"ammar Agent is working wait please applying discount (price={price}, discount_tier='{discount_tier}')...")
    discounts_percentages = {
        "gold": 10,
        "silver": 5,
        "bronze": 2,
    }
    discount=discounts_percentages.get(discount_tier, 0)
    return  round(price * (1 - discount / 100), 2)
#-----------aegent loop----------------

@traceable (name="ammar_agent_loop")

def run_agent(question :str):
    tools=[get_product_price, get_apply_discount]
    tools_dict={t.name: t for t in tools}
    llm=init_chat_model(f"ollama:{MODEL}",temperature=0)
    llm_with_tools=llm.bind_tools(tools)
    print(f"question: {question}")
    print("="*60)
    messages=[SystemMessage
              (content="you are a helpful assistant for answering questions about product prices and discounts."
               "you have access to a product catalog tools and a discount tool"
               "you must follow this exactly:\n"
               "1.never guess or assume any product price."
               "you must call get_product_price tool first to get real price.\n"
               "2.onlly call  get_apply_discount after ypu have received the price .pass the exact price"
               "returned by get_product_price -Dont pass a made up number.\n"
               "3.Never calculate the discount by yourself or use any math "
               " Always call get_apply_discount tool to apply the discount .\n"
               "4.if user dosenot specify a discount tier ,"
               "ask them which tier they want.-Do not assume one."
               )
              ,HumanMessage(content=question)]
    # بدء حلقة التكرار للعميل (Agent Loop) لضمان تنفيذ الخطوات بالتسلسل
    for itaration in range(1, MAX_ITERATIONS + 1):
        print(f"--itration {itaration}-- ")
        
        # 1. إرسال قائمة الرسائل (السؤال + التاريخ) إلى النموذج ليقرر الخطوة التالية
        ai_message = llm_with_tools.invoke(messages)
        
        # 2. التحقق مما إذا كان النموذج يريد استخدام "أداة" (Tool) أم يريد إعطاء إجابة نهائية
        tool_calls = ai_message.tool_calls
        
        # إذا لم يطلب النموذج أي أداة، فهذا يعني أنه وصل للإجابة النهائية
        if not tool_calls:
            print("agent has responded without calling any tool, stopping the loop.")
            print(f"final answer: {ai_message.content}")
            return ai_message.content # إنهاء الدالة وإرجاع النص النهائي للمستخدم

        # 3. في حال طلب النموذج أداة، نختار الطلب الأول (بفرض أنه طلب أداة واحدة)
        tool_call = tool_calls[0]
        
        # استخراج تفاصيل استدعاء الأداة (الاسم، المدخلات، والمعرف الفريد)
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f" [Tool Selected] {tool_name} with args: {tool_args}")

        # 4. البحث عن "الدالة" البرمجية المقابلة للاسم الذي اختاره النموذج (مثل get_product_price)
        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # 5. التنفيذ الفعلي للدالة (الأداة) برمجياً للحصول على المعلومة (مثل سعر اللابتوب)
        observation = tool_to_use.invoke(tool_args)
        print(f" [Tool Result] {observation}")

        # 6. تحديث "ذاكرة" العميل (Messages):
        # أولاً: نضيف رسالة النموذج التي تحتوي على "طلب استدعاء الأداة"
        messages.append(ai_message)
        
        # ثانياً: نضيف رسالة بنوع (ToolMessage) تحتوي على "نتيجة الأداة" التي حصلنا عليها
        # ملاحظة: يجب إرسال الـ tool_call_id ليعرف النموذج أن هذه النتيجة تابعة لهذا الطلب
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print(" error Reached maximum iterations without a final answer, stopping the loop.")
    return  None
   
def main():
    print("Hello from  ammar 1agent-under-the-hood!")
    result=run_agent("what is the price of smartphone with gold discount?")
if __name__ == "__main__":
    main()
