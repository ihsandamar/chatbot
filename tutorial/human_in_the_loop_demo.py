from __future__ import annotations

import uuid
from typing import TypedDict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from src.services.config_loader import ConfigLoader


# ---------------------------------------------------------------------------
# 1. Ortak durum (state) tanımı
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    approved: bool


# ---------------------------------------------------------------------------
# 2. Grafik düğümleri (nodes)
# ---------------------------------------------------------------------------
config = ConfigLoader().load_config()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=config.llm.api_key)


def draft_reply(state: ChatState) -> ChatState:
    """LLM çıktısını üretir."""
    reply = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [reply]}


def review_by_human(state: ChatState):
    """İnsan müdahalesi için grafiği durdurur.

    `interrupt`  JSON‑serileştirilebilir bir payload döner.
    Grafiğe `Command(resume=...)` ile devam edildiğinde, bu fonksiyon
    baştan çalışır ve `interrupt(...)` ilgili insan girdisini döndürür.
    """
    human_input = interrupt(
        {
            "question": "Mesajı onaylıyor musunuz? Düzenleyebilirsiniz.",
            "draft": state["messages"][-1].content,
        }
    )

    # İnsan bir string döndürdüyse doğrudan onaylıyoruz, aksi hâlde
    # dict içerisinde `edited` alanına bakıyoruz.
    if isinstance(human_input, str):
        edited_text = human_input
    else:
        edited_text = human_input.get("edited", state["messages"][-1].content)

    approved_message = AIMessage(content=edited_text)
    return {
        "messages": state["messages"][:-1] + [approved_message],
        "approved": True,
    }


def final_node(state: ChatState) -> ChatState:
    print("\n✅ Sonuç:")
    print(state["messages"][-1].content)
    return state


# ---------------------------------------------------------------------------
# 3. Grafiği kur
# ---------------------------------------------------------------------------
builder = StateGraph(ChatState)

builder.add_node("draft_reply", draft_reply)
builder.add_node("review_by_human", review_by_human)
builder.add_node("final", final_node)

builder.set_entry_point("draft_reply")

builder.add_edge("draft_reply", "review_by_human")
# Human onayından sonra final düğümüne geç.
builder.add_edge("review_by_human", "final")

builder.add_edge("final", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 4. Demo çalıştır
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    thread_cfg = {"configurable": {"thread_id": uuid.uuid4()}}

    # Başlangıç mesajı
    initial_state: ChatState = {
        "messages": [
            HumanMessage(content="LangGraph'ta human‑in‑the‑loop örneği gösterir misin?"),
        ]
    }

    # İlk çağrı: interrupt'a kadar çalışır
    result = graph.invoke(initial_state, config=thread_cfg)

    # Eğer interrupt geldi, payload'ı ekrana bas ve kullanıcıdan girdi al
    if "__interrupt__" in result:
        int_payload = result["__interrupt__"][0]
        print("\n--- Draft görüldü: ")
        print(int_payload.value["draft"])
        print("------------------")
        edited = input("Düzenlenmiş metni girin (boş bırak = onay): ")
        resume_value = edited if edited else int_payload.value["draft"]

        # Grafiği insan girdisiyle sürdür
        graph.invoke(Command(resume=resume_value), config=thread_cfg)

