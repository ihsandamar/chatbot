# src/interfaces/gradio_interface.py
import gradio as gr
import yaml
import os
from src.core.chatbot import Chatbot



def run_gradio(chatbot: Chatbot):
    # 🔹 Gradio arayüzü
    with gr.Blocks() as demo:
        # Logo banner
        with gr.Row():
            gr.Image(
                "assets/vegabot.png",
                width=200,
                height=200,
                interactive=False,
                show_label=False,
                show_download_button=False,
                container=False
            )
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot_component = gr.Chatbot(
                    label="LangGraph Chatbot",
                    placeholder="Merhaba! Size nasıl yardımcı olabilirim?",
                    show_copy_button=True,
                    height=300
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 💡 Hızlı Başlangıç Soruları:")
                
                # Load grouped prompts from YAML
                grouped_prompts = {}
                try:
                    if os.path.exists("config/default_prompts.yaml"):
                        with open("config/default_prompts.yaml", 'r', encoding='utf-8') as file:
                            config = yaml.safe_load(file)
                            grouped_prompts = config.get('groups', {})
                except Exception as e:
                    print(f"Error loading prompts: {e}")
                    grouped_prompts = {"Genel": ["Merhaba! Size nasıl yardımcı olabilirim?"]}
                
                # Fallback if no prompts loaded
                if not grouped_prompts:
                    grouped_prompts = {"Genel": ["Merhaba! Size nasıl yardımcı olabilirim?"]}
                
                # Create compact grouped interface with Accordion
                buttons = []
                for group_name, prompts in grouped_prompts.items():
                    if prompts:  # Only show groups that have prompts
                        with gr.Accordion(f"📁 {group_name}", open=False):
                            for prompt in prompts:
                                btn = gr.Button(prompt, size="sm")
                                buttons.append((btn, prompt))
                
                # Setup handlers with proper closure
                def make_handler(prompt):
                    return lambda history: chatbot.response_handler(history, prompt)
                
                for btn, prompt_text in buttons:
                    btn.click(
                        fn=make_handler(prompt_text),
                        inputs=[chatbot_component],
                        outputs=[chatbot_component]
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
