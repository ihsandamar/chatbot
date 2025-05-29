# interfaces/flask_interface.py
from flask import Flask, request, jsonify

def run_flask(chatbot):
    app = Flask(__name__)

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.get_json()
        messages = data.get("messages", [])
        result = chatbot.send({"messages": messages})
        return jsonify({"response": result})

    app.run(port=5000)
