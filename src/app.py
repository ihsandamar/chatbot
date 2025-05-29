import gradio as gr
from src.builders.chatbot_builder import ChatbotBuilder
from src.chatbot import Chatbot
from src.config import OPENAI_API_KEY
from src.graphs.main_graph import MainGraph
from src.models import LLM
from src.graphs.graph_repository import GraphRepository
from src.services.container import ServiceContainer



# TODO: Graph repository declaration will change in the future
# [ ] FEATURE: Graph repository funcs will be defined as a static method
# [ ] FEATURE: Graph repository will be used to manage different graph types
# [ ] FEATURE: LLM models will be managed by the graph repository functions (setting up the LLM model, temperature, etc.)


# 🔹 1. Service Container oluştur
container = ServiceContainer()
container.register("llm", lambda: LLM(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY))
container.register("graph_repo", lambda: GraphRepository(container.resolve("llm")))

# 🔹 2. Chatbot'u builder ile oluştur
chatbot = (
    ChatbotBuilder(container)
    .with_model("gpt-4o-mini")
    .with_graph("main")
    .with_config({"configurable": {"thread_id": "1"}})
    .build()
)


#User interface

with gr.Blocks() as demo:
    # Chat component
    with gr.Row():
        chatbot_component = gr.Chatbot(
            label="LangGraph Chatbot",
            placeholder="Merhaba! Size nasıl yardımcı olabilirim?",
            show_copy_button=True,
            height=600
        )

    # Text input for user messages
    with gr.Row():
        user_input = gr.Textbox(
            label="Mesajınızı yazın",
            placeholder="Buraya mesajınızı yazın...",
            show_label=False,
            lines=1
        )

    # Submit button
    with gr.Row():
        submit_button = gr.Button("Gönder")



    text = submit_button.click(
        chatbot.response_handler,
        inputs=[chatbot_component, user_input],
        outputs=[chatbot_component],
        queue=False).then(lambda: gr.Textbox(interactive=True),
                            None, [user_input], queue=False)

    text = user_input.submit(
        chatbot.response_handler,
        inputs=[chatbot_component, user_input],
        outputs=[chatbot_component],
        queue=False).then(lambda: gr.Textbox(interactive=True),
                            None, [user_input], queue=False)






if __name__ == "__main__":
    demo.launch()
