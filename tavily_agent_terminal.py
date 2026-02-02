import os
import re
import math
import ast
from datetime import datetime
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults

# =========================================================
# 0) ENV
# =========================================================
load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5-chat-nextai")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not BASE_URL or not API_KEY:
    raise ValueError("Faltan BASE_URL o API_KEY en el .env")

if not TAVILY_API_KEY:
    raise ValueError("Falta TAVILY_API_KEY en el .env")

CUSTOM_HEADERS = {
    "origin": "research",
    "origin-detail": "Escuela2026",
    "provider": "AzureOpenAI",
}

# =========================================================
# 1) LLM
# =========================================================
llm = ChatOpenAI(
    api_key=SecretStr(API_KEY),
    model=MODEL_NAME,
    base_url=BASE_URL,
    default_headers=CUSTOM_HEADERS,
    temperature=0.2,
)

# =========================================================
# 2) TOOLS (Tavily + extras)
# =========================================================
tavily_tool = TavilySearchResults(
    max_results=6,
    tavily_api_key=TAVILY_API_KEY,
)

@tool
def now_madrid() -> str:
    """Devuelve fecha y hora actual en formato YYYY-MM-DD HH:MM:SS (hora local del sistema)."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _safe_eval(expr: str) -> float:
    """Calculadora segura: + - * / ** () y funciones básicas."""
    allowed_names = {
        "pi": math.pi,
        "e": math.e,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "abs": abs,
        "round": round,
    }

    node = ast.parse(expr, mode="eval")
    for subnode in ast.walk(node):
        if isinstance(subnode, ast.Name) and subnode.id not in allowed_names:
            raise ValueError(f"Nombre no permitido: {subnode.id}")
        if isinstance(subnode, ast.Call):
            if not isinstance(subnode.func, ast.Name) or subnode.func.id not in allowed_names:
                raise ValueError("Función no permitida.")
        if isinstance(subnode, (ast.Attribute, ast.Import, ast.ImportFrom, ast.Lambda)):
            raise ValueError("Expresión no permitida.")

    return eval(compile(node, "<calc>", "eval"), {"__builtins__": {}}, allowed_names)

@tool
def calculator(expression: str) -> str:
    """Calcula una expresión matemática. Ej: '12/3 + sqrt(16)'."""
    try:
        result = _safe_eval(expression.strip())
        return f"Resultado: {result}"
    except Exception as e:
        return f"Error en cálculo: {e}"

TOOLS = [tavily_tool, calculator, now_madrid]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# Modelo con tools vinculadas (tool calling)
tool_llm = llm.bind_tools(TOOLS)

# =========================================================
# 3) MEMORIA (terminal)
# =========================================================
history: List[Tuple[str, str]] = []  # [(user, final_answer), ...]

def render_history(max_turns: int = 6) -> str:
    tail = history[-max_turns:]
    if not tail:
        return "(vacío)"
    lines = []
    for u, a in tail:
        lines.append(f"User: {u}")
        lines.append(f"Agent: {a}")
    return "\n".join(lines)

# =========================================================
# 4) BRANCHING (intención → pista de estilo)
# =========================================================
def guess_intent(user_text: str) -> str:
    t = user_text.lower()
    if any(k in t for k in ["precio", "comparar", "mejor", "recomend", "comprar", "review", "opinión"]):
        return "shopping"
    if any(k in t for k in ["hoy", "última hora", "noticia", "reciente", "ayer", "esta semana"]):
        return "news"
    if any(k in t for k in ["temperatura", "tiempo", "lluvia", "clima", "previsión"]):
        return "weather"
    return "general"

intent_branch = RunnableBranch(
    (lambda x: x["intent"] == "news",
     RunnableLambda(lambda x: {**x, "style_hint": "Prioriza fuentes recientes y menciona fechas si aparecen."})),
    (lambda x: x["intent"] == "shopping",
     RunnableLambda(lambda x: {**x, "style_hint": "Incluye pros/contras y rango de precios si aparece en fuentes."})),
    (lambda x: x["intent"] == "weather",
     RunnableLambda(lambda x: {**x, "style_hint": "Aclara ubicación y cita fuentes oficiales si las hay."})),
    RunnableLambda(lambda x: {**x, "style_hint": "Da un resumen claro y añade fuentes si usas búsquedas."}),
)

# =========================================================
# 5) PARALLEL (formateo + extracción URLs)
# =========================================================
postprocess_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        "Eres un asistente que formatea respuestas para terminal.\n"
        "Responde SIEMPRE en español.\n"
        "NO inventes fuentes ni URLs.\n"
        "Solo muestra 'Fuentes' si el borrador contiene URLs.\n"
        "No uses bloques de código (```); solo texto normal.\n"
        "Evita cifras de precios o límites si no vienen explícitamente en las fuentes.\n"
        "Si la hora viene de una tool local, dilo explícitamente (hora del sistema).\n"
        ),
        ("human",
         "Pregunta:\n{question}\n\n"
         "Borrador del agente:\n{draft}\n\n"
         "Pista de estilo:\n{style_hint}\n\n"
         "Devuelve:\n"
         "1) Resumen (3-6 líneas)\n"
         "2) Puntos clave (4-8 bullets)\n"
         "3) Si el borrador contiene URLs, añade 'Fuentes:' con 3-6 URLs únicas.\n"),
    ]
)

def extract_urls(text: str) -> List[str]:
    urls = re.findall(r"https?://\S+", text)
    clean = []
    for u in urls:
        u = u.rstrip(").,]")
        if u not in clean:
            clean.append(u)
    return clean[:8]

parallel_post = RunnableParallel(
    formatted=(postprocess_prompt | llm | StrOutputParser()),
    urls=RunnableLambda(lambda x: extract_urls(x["draft"])),
)

def merge_formatted_and_urls(out: Dict[str, Any]) -> str:
    formatted = out["formatted"].strip()
    urls = out["urls"]
    if urls and "Fuentes:" not in formatted:
        formatted += "\n\nFuentes:\n" + "\n".join(f"- {u}" for u in urls[:6])
    return formatted

# =========================================================
# 6) “AGENTE” (Tool-calling loop) — LangChain 0.3.x correcto
# =========================================================
SYSTEM_TEXT = (
    "Eres un agente de investigación en tiempo real.\n"
    "Reglas:\n"
    "- Si la pregunta requiere info actual o verificable, usa Tavily.\n"
    "- Si es cálculo, usa calculator.\n"
    "- Si preguntan por fecha/hora, usa now_madrid.\n"
    "- Responde con claridad. Si usas Tavily, incluye fuentes (URLs) si aparecen.\n"
)

def run_tool_calling_agent(question: str, history_text: str, max_iters: int = 6) -> str:
    messages = [
        SystemMessage(content=SYSTEM_TEXT + f"\n\nMemoria:\n{history_text}"),
        HumanMessage(content=question),
    ]

    last_text = ""

    for _ in range(max_iters):
        try:
            ai = tool_llm.invoke(messages)
        except Exception as e:
            # ✅ FALLBACK si el LLM está caído (500, etc.)
            return fallback_answer(question, str(e))

        messages.append(ai)

        if getattr(ai, "content", None):
            last_text = ai.content

        tool_calls = getattr(ai, "tool_calls", None)

        # Si no pide tools, ya es respuesta final
        if not tool_calls:
            return ai.content or last_text or "(sin respuesta)"

        # Ejecutar tools y devolver ToolMessage al modelo
        for call in tool_calls:
            tool_name = call.get("name")
            tool_args = call.get("args", {})
            tool_call_id = call.get("id")

            tool = TOOLS_BY_NAME.get(tool_name)
            if tool is None:
                tool_result = f"Tool no encontrada: {tool_name}"
            else:
                try:
                    tool_result = tool.invoke(tool_args)
                except Exception as ex:
                    tool_result = f"Error ejecutando {tool_name}: {ex}"

            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call_id))

    return last_text or "(No se pudo completar en el límite de iteraciones)"
    messages = [
        SystemMessage(content=SYSTEM_TEXT + f"\n\nMemoria:\n{history_text}"),
        HumanMessage(content=question),
    ]

    last_text = ""

    for _ in range(max_iters):
        ai = tool_llm.invoke(messages)
        messages.append(ai)

        if getattr(ai, "content", None):
            last_text = ai.content

        tool_calls = getattr(ai, "tool_calls", None)

        # Si no pide tools, ya es respuesta final
        if not tool_calls:
            return ai.content or last_text or "(sin respuesta)"

        # Ejecutar tools y devolver ToolMessage al modelo
        for call in tool_calls:
            tool_name = call.get("name")
            tool_args = call.get("args", {})
            tool_call_id = call.get("id")

            tool = TOOLS_BY_NAME.get(tool_name)
            if tool is None:
                tool_result = f"Tool no encontrada: {tool_name}"
            else:
                try:
                    tool_result = tool.invoke(tool_args)
                except Exception as e:
                    tool_result = f"Error ejecutando {tool_name}: {e}"

            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call_id))

    return last_text or "(No se pudo completar en el límite de iteraciones)"


def fallback_answer(question: str, err: str) -> str:
    """
    Respuesta de emergencia si el endpoint LLM falla.
    - Si detecta cálculo: usa calculator
    - Si detecta hora/fecha: usa now_madrid
    - Si detecta búsqueda web: usa Tavily directamente y devuelve fuentes
    """
    q = question.lower()

    parts = []
    parts.append("⚠️ Nota: el endpoint LLM ha devuelto error (500). Respondo usando tools locales/web.\n")

    # 1) Cálculo
    # Heurística simple: si hay números u operadores típicos
    if any(ch in q for ch in ["+", "-", "*", "/", "sqrt", "log", "sin", "cos", "tan"]) and any(c.isdigit() for c in q):
        # extrae una expresión razonable del texto (si el usuario la puso explícita)
        # si no, intentamos con la frase entera
        expr = question
        # intenta rescatar solo lo que parece expresión
        expr_candidate = re.findall(r"([0-9\.\s\+\-\*\/\(\)]+|sqrt\([0-9\.]+\))", question, flags=re.IGNORECASE)
        if expr_candidate:
            expr = " ".join(expr_candidate).strip()
        parts.append("🧮 Cálculo:")
        parts.append(calculator.invoke({"expression": expr}))

    # 2) Hora/fecha
    if any(k in q for k in ["hora", "fecha", "now", "madrid"]):
        parts.append("\n🕒 Hora actual:")
        parts.append(now_madrid.invoke({}))

    # 3) Búsqueda web (Tavily)
    if any(k in q for k in ["qué es", "que es", "fuentes", "url", "busca", "buscar", "investiga", "último", "reciente", "noticia", "tavily", "serpapi", "alternativa"]):
        parts.append("\n🌐 Búsqueda web (Tavily):")
        try:
            results = tavily_tool.invoke({"query": question})
            # Formato compacto con fuentes
            for i, r in enumerate(results[:3], start=1):
                title = r.get("title", "Sin título")
                url = r.get("url", "")
                parts.append(f"{i}. {title}\n   {url}")
        except Exception as ex:
            parts.append(f"(No se pudo consultar Tavily: {ex})")

    parts.append(f"\n[Debug] Error LLM: {err}")
    return "\n".join(parts)

# =========================================================
# 7) TERMINAL LOOP
# =========================================================
def main():
    print("\n=== Tavily Agent (LangChain) + Memoria + Branching + Parallel ===")
    print("Escribe 'exit' para salir.\n")

    while True:
        user_q = input("Tú: ").strip()
        if user_q.lower() == "exit":
            break
        if not user_q:
            continue

        # 1) intención + style hint
        intent = guess_intent(user_q)
        enriched = intent_branch.invoke({"intent": intent})
        style_hint = enriched["style_hint"]

        # 2) memoria
        history_text = render_history(max_turns=6)

        # 3) input aumentado (para que se vea la memoria y se incentive Tavily si toca)
        augmented_input = (
            f"Historial:\n{history_text}\n\n"
            "Instrucción:\nSi la respuesta depende de información actual o verificable, usa Tavily.\n\n"
            f"Pregunta:\n{user_q}"
        )

        # 4) agente (borrador)
        draft_answer = run_tool_calling_agent(
            question=augmented_input,
            history_text=history_text,
            max_iters=6
        ).strip()

        # 5) postprocesado (parallel)
        post_out = parallel_post.invoke({
            "question": user_q,
            "draft": draft_answer,
            "style_hint": style_hint
        })
        final_answer = merge_formatted_and_urls(post_out)

        # 6) guardar memoria y mostrar
        history.append((user_q, final_answer))

        print("\nAI:\n" + final_answer)
        print(f"\n--- (Debug) intent: {intent} | turns: {len(history)} ---\n")

if __name__ == "__main__":
    main()
