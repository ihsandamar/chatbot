# interfaces/cli_interface.py
def run_cli(chatbot):
    print("🧠 CLI Chatbot başlatıldı. Çıkmak için 'exit' yazın.")
    while True:
        user_input = input("👤 Siz: ")
        if user_input.lower() in {"exit", "quit"}:
            print("🔚 Görüşmek üzere!")
            break
        response = chatbot.send({"messages": [{"role": "user", "content": user_input}]})
        print(f"🤖 Bot: {response}")
