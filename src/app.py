import gradio as gr
from src.builders.chatbot_builder import ChatbotBuilder
from src.chatbot import Chatbot
from src.config import OPENAI_API_KEY
from src.graphs.main_graph import MainGraph
from src.models import LLM
from src.graphs.graph_repository import GraphRepository
from src.services.config_loader import ConfigLoader
from src.services.container import ServiceContainer



# TODO: Graph repository declaration will change in the future
# [ ] FEATURE: Graph repository funcs will be defined as a static method
# [ ] FEATURE: Graph repository will be used to manage different graph types
# [ ] FEATURE: LLM models will be managed by the graph repository functions (setting up the LLM model, temperature, etc.)

config = ConfigLoader.load_config()


# This container will manage the dependencies of the application
container = ServiceContainer()

llm_config = config["llm"]
container.register("llm", lambda: LLM(
    model=llm_config["model"],
    temperature=llm_config["temperature"],
    api_key=llm_config["api_key"]
))

container.register("graph_repo", lambda: GraphRepository(container.resolve("llm")))


# Registering the main graph
chatbot = (
    ChatbotBuilder(container)
    .with_model(llm_config["model"], llm_config["temperature"])
    .with_graph(config["chatbot"]["graph_type"])
    .with_config({"configurable": {"thread_id": config["chatbot"]["thread_id"]}})
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
