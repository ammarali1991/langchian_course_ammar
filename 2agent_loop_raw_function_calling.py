from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# --- Tools (LangChain @tool decorator) ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# 🔑 الفرق الجوهري: بدلاً من @tool decorator نكتب JSON schema يدوياً
# هذا هو ما ترسله للـ LLM ليعرف الأدوات المتاحة وكيف يستخدمها
tools_for_llm = [
    {
        "type": "function",  # نوع الأداة: دالة
        "function": {
            "name": "get_product_price",  # اسم الدالة الذي سيطلبه الـ LLM
            "description": "Look up the price of a product in the catalog.",
            # 📋 الـ schema: تخبر الـ LLM بالـ parameters المطلوبة ونوع كل واحد
            "parameters": {
                "type": "object",  # المعاملات تأتي كـ object
                "properties": {
                    "product": {
                        "type": "string",  # نوع المعامل
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],  # المعاملات الإجبارية
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],  # الدالة تحتاج المعاملين
            },
        },
    },
]


# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)

# --- Agent Loop ---


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    # 🔑 ربط اسم الأداة (string من الـ schema) بالدالة الفعلية (callable)
    # الـ LLM يرجع اسم الأداة كـ string، ونحن نبحث هنا عن الدالة المناسبة
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }



    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Difference 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        # 🔑 هيكل الـ tool call القادم من Ollama (بدون LangChain):
        # ai_message.tool_calls عبارة عن list من ToolCall objects
        # كل ToolCall يحتوي على .function.name و .function.arguments (dict)
        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        # 🔑 استخراج اسم الأداة والـ arguments من الـ ToolCall
        # الفرق هنا: نستخدم attribute access (.function.name)
        # بدلاً من dict access (["function"]["name"]) كما في LangChain
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments  # هذا dict جاهز: {"product": "laptop"}

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        # 🔑 استدعاء الدالة الفعلية باستخدام الـ dict lookup
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # 🔑 تنفيذ الدالة مع تفريغ الـ arguments كـ kwargs: **tool_args
        # مثال: tool_to_use(**{"product": "laptop"}) => get_product_price(product="laptop")
        observation = tool_to_use(**tool_args)


        print(f"  [Tool Result] {observation}")

        # 🔑 إعادة نتيجة الأداة للـ LLM في السياق
        # نضيف رسالة الـ assistant (التي طلبت الأداة)
        messages.append(ai_message)
        # ثم رسالة tool تحتوي على النتيجة
        messages.append(
            {
                "role": "tool",
                "content": str(observation),  # النتيجة كـ string
            }
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")