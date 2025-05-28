import gradio as gr
from src.chatbot import Chatbot
from src.config import OPENAI_API_KEY
from src.models import LLM
from src.graphs.graph_repository import GraphRepository



llm = LLM(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
config = {"configurable": {"thread_id": "1"}}

# Graph repository to manage different graph types
graph_repo = GraphRepository(llm=llm)

# Get the main graph
main_graph = graph_repo.get("main")

# Initialize the chatbot with the main graph and configuration
chatbot = Chatbot(llm=llm, config=config, graph=main_graph)




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
