# main.py
import argparse
from src.models import LLM
from src.services.config_loader import ConfigLoader
from src.services.container import ServiceContainer
from src.graphs.graph_repository import GraphRepository
from src.builders.chatbot_builder import ChatbotBuilder

from src.interfaces.cli_interface import run_cli
from src.interfaces.run_gradio import run_gradio
from src.interfaces.flask_interface import run_flask


def build_chatbot():
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
        .with_model(llm_config["model"], llm_config["temperature"], llm_config["api_key"])
        .with_graph(config["chatbot"]["graph_type"])
        .with_config({"configurable": {"thread_id": config["chatbot"]["thread_id"]}})
        .build()
    )
    return chatbot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", choices=["cli", "gradio", "api"], default="cli", 
                        help="Choose the interface to run the chatbot: cli, gradio, or api")
    
      
    args = parser.parse_args()

    chatbot = build_chatbot()

    if args.interface == "cli":
        run_cli(chatbot)
    elif args.interface == "gradio":
        run_gradio(chatbot)
    elif args.interface == "api":
        run_flask(chatbot)
