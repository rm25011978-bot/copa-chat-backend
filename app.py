import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq, GroqError

load_dotenv()

app = Flask(__name__)

# Configuração de CORS para permitir acesso apenas do domínio do frontend
cors_origin = os.getenv("CORS_ORIGIN", "*")
CORS(app, resources={r"/api/*": {"origins": cors_origin}})

SYSTEM_PROMPT = """
Você é o "GolBot", um assistente especialista e apaixonado em Copa do Mundo de Futebol Masculino e Feminino.
Sua única função e objetivo é conversar sobre a história das Copas, seleções, jogadores, títulos, regulamentos, estatísticas, curiosidades e jogos lendários.

REGRAS RÍGIDAS DE ESCOPO:
1. Responda APENAS a dúvidas, análises e conversas estritamente relacionadas à Copa do Mundo ou futebol de seleções/histórico da competição.
2. Se o usuário tentar falar sobre qualquer assunto fora de Copa do Mundo/futebol (ex: política, receita de bolo, programação, vida pessoal, outros esportes, notícias gerais):
   - NÃO responda nem forneça informações sobre o assunto solicitado.
   - Redirecione-o com polidez e entusiasmo futebolístico de volta à Copa do Mundo. Exemplo: "Como um grande apaixonado por futebol, meu foco é exclusivamente a Copa do Mundo! Que tal conversarmos sobre a final de 1970 ou o maior artilheiro das Copas?"
3. Mantenha um tom empolgado, cortês, preciso e esportivo. Use termos e analogias do futebol sempre que pertinente.
"""

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY não foi configurada nas variáveis de ambiente.")
    return Groq(api_key=api_key)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Backend do Copa Chat operando perfeitamente."})

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "messages" not in data:
        return jsonify({"error": "Formato de requisição inválido. 'messages' é obrigatório."}), 400

    incoming_messages = data.get("messages", [])
    if not isinstance(incoming_messages, list) or len(incoming_messages) == 0:
        return jsonify({"error": "O campo 'messages' deve ser uma lista não vazia."}), 400

    # Injeta o System Prompt no início do histórico de mensagens
    messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for msg in incoming_messages:
        role = msg.get("role")
        content = msg.get("content")
        if role in ["user", "assistant"] and content:
            messages_payload.append({"role": role, "content": str(content)})

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    try:
        client = get_groq_client()
        chat_completion = client.chat.completions.create(
            messages=messages_payload,
            model=model,
            temperature=0.6,
            max_tokens=1024,
        )
        bot_response = chat_completion.choices[0].message.content
        return jsonify({"response": bot_response})

    except GroqError as e:
        app.logger.error(f"Erro na API da Groq: {str(e)}")
        return jsonify({"error": "Falha na comunicação com o serviço de IA. Tente novamente em instantes."}), 502
    except Exception as e:
        app.logger.error(f"Erro interno no servidor: {str(e)}")
        return jsonify({"error": "Erro interno no servidor."}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)