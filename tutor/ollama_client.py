def chat(messages, schema, model="qwen2.5:14b") -> str:
    import ollama

    response = ollama.chat(
        model=model, messages=messages, format=schema, options={"temperature": 0}
    )
    return response["message"]["content"]
