import gradio as gr
from src.chatbot import response_handler

demo = gr.ChatInterface(
    fn=response_handler,
    type="chat",
    title="Basit Chatbot",
    description="Sade bir LangGraph destekli chatbot",
    chatbot=gr.Chatbot(height=400),
    autofocus=False
)

demo.launch()

# if __name__ == "__main__":
#     demo.launch()
