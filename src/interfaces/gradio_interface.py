# src/interfaces/gradio_interface.py
import gradio as gr
from src.core.chatbot import Chatbot



def run_gradio(chatbot: Chatbot):
    # 🔹 Gradio arayüzü
    with gr.Blocks() as demo:
        with gr.Row():
            chatbot_component = gr.Chatbot(
                label="LangGraph Chatbot",
                placeholder="Merhaba! Size nasıl yardımcı olabilirim?",
                show_copy_button=True,
                height=600
            )

        with gr.Row():
            user_input = gr.Textbox(
                label="Mesajınızı yazın",
                placeholder="Buraya mesajınızı yazın...",
                show_label=False,
                lines=1
            )

        with gr.Row():
            submit_button = gr.Button("Gönder")

        # 🔹 Buton ve Enter tetikleyicileri
        submit_button.click(
            chatbot.response_handler,
            inputs=[chatbot_component, user_input],
            outputs=[chatbot_component],
            queue=False
        ).then(lambda: gr.Textbox(interactive=True),
               None, [user_input], queue=False)

        user_input.submit(
            chatbot.response_handler,
            inputs=[chatbot_component, user_input],
            outputs=[chatbot_component],
            queue=False
        ).then(lambda: gr.Textbox(interactive=True),
               None, [user_input], queue=False)

    demo.launch(server_name="0.0.0.0", share=False, pwa=True)
