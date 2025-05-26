
from src.graphs.main_graph import MainGraph
from src.models import LLM
from src.config import OPENAI_API_KEY


llm = LLM(model="gpt-3.5-turbo", temperature=0.0, api_key=OPENAI_API_KEY)
graph = MainGraph(llm=llm).build_graph()




def response_handler(message, history):
    """
    Gradio 'chat' tipi için uygun yanıt formatı.
    message: kullanıcıdan gelen tek mesaj
    history: [(user, bot), ...] şeklinde geçmiş
    """
    state = {"messages": [{"role": "user", "content": message}]}
    response = graph.invoke(state)

    # Beklenen: tek bir string (asistanın cevabı)
    if "messages" in response:
        message = response["messages"][-1]
        return message.content
    else:
        return "Bir hata oluştu, lütfen tekrar deneyin."




if __name__ == "__main__":
    result = graph.invoke({"messages": ["Hello, how are you?"]})
    print("Chatbot Response:", result)
