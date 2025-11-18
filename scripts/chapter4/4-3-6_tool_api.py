import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def multiply(a, b):
    """掛け算ツール"""
    return a * b

def chat_with_tools(user_input):
    # ツール定義
    tools = [{
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "二つの数値を掛け算します",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "最初の数値"},
                    "b": {"type": "number", "description": "二番目の数値"}
                },
                "required": ["a", "b"]
            }
        }
    }]

    messages = [
        {"role": "system", "content": "掛け算ツールを使って計算してください。"},
        {"role": "user", "content": user_input}
    ]

    # 最初のAPI呼び出し
    response = openai.chat.completions.create(
        model="gpt-5-nano",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # ツール呼び出しがある場合
    if message.tool_calls:
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.tool_calls
        })

        for tool_call in message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            print(f"ツール実行: multiply({arguments['a']}, {arguments['b']})")

            result = multiply(arguments["a"], arguments["b"])
            print(f"実行結果: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

        # 最終回答を取得
        final_response = openai.chat.completions.create(
            model="gpt-5-nano",
            messages=messages
        )
        return final_response.choices[0].message.content

    return message.content

print(chat_with_tools("7に8を掛けてください"))
