import os
import json
import logging
import requests
import math
import boto3
from datetime import datetime
from boto3.dynamodb.conditions import Key
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import ask_sdk_core.utils as ask_utils
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import (
    AbstractRequestHandler,
    AbstractExceptionHandler
)



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "laia-historial")
AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")

MAX_HISTORIAL_DB = 50
TOP_K = 5

EMBEDDING_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-embedding-001:embedContent?key={GEMINI_API_KEY}"
)

# Reutilizar conexiones HTTP
requests_session = requests.Session()

# Clientes AWS
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(DYNAMODB_TABLE)

lambda_client = boto3.client(
    "lambda",
    region_name=AWS_REGION,
    config=Config(
        connect_timeout=1,
        read_timeout=2,
        retries={"max_attempts": 1}
    )
)


def generar_embedding(texto):
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": texto}]}
    }

    try:
        r = requests_session.post(EMBEDDING_URL, json=payload, timeout=5)

        if r.status_code != 200:
            logger.error(f"❌ EMBEDDING HTTP ERROR {r.status_code}: {r.text}")
            return None

        data = r.json()
        vector = data["embedding"]["values"]
        logger.info(f"🔢 EMBEDDING generado ({len(vector)} dims) para: {texto[:50]}")
        return vector

    except Exception as e:
        logger.error(f"❌ EMBEDDING ERROR: {e}", exc_info=True)
        return None


def similitud_coseno(a, b):
    if not a or not b:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0


def buscar_relevantes(embedding_consulta, registros, top_k=TOP_K):
    puntuados = []

    for r in registros:
        emb_str = r.get("embedding")
        if not emb_str:
            continue

        try:
            emb = json.loads(emb_str)
            score = similitud_coseno(embedding_consulta, emb)
            puntuados.append((score, r))
        except Exception as e:
            logger.warning(f"⚠️ No se pudo leer embedding guardado: {e}")
            continue

    puntuados.sort(key=lambda x: x[0], reverse=True)
    seleccionados = [r for _, r in puntuados[:top_k]]

    logger.info(
        f"🎯 TOP {top_k} relevantes | "
        f"scores: {[round(s, 3) for s, _ in puntuados[:top_k]]}"
    )

    return seleccionados


def guardar_en_db(user_id, pregunta, respuesta, embedding=None):
    """
    Guarda pregunta y respuesta en DynamoDB.
    Reutiliza el embedding ya calculado si se recibe.
    """
    if embedding is None:
        embedding = generar_embedding(pregunta)

    item = {
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "pregunta": pregunta,
        "respuesta": respuesta
    }

    if embedding:
        item["embedding"] = json.dumps(embedding)

    try:
        table.put_item(Item=item)
        logger.info(
            f"✅ DYNAMO GUARDADO | user: {user_id[:20]} | pregunta: {pregunta[:50]}"
        )
        return True
    except Exception as e:
        logger.error(f"❌ DYNAMO ERROR guardando: {e}", exc_info=True)
        return False


def obtener_historial_db(user_id):
    try:
        result = table.query(
            KeyConditionExpression=Key("user_id").eq(user_id),
            ScanIndexForward=False,
            Limit=MAX_HISTORIAL_DB
        )
        items = list(reversed(result.get("Items", [])))
        logger.info(f"📚 DYNAMO LEÍDO | user: {user_id[:20]} | registros: {len(items)}")
        return items
    except Exception as e:
        logger.error(f"❌ DYNAMO ERROR leyendo: {e}", exc_info=True)
        return []


def construir_prompt(pregunta, relevantes, historial_sesion, historial_db):
    lineas = []
    lineas.append("════════════════════════════════")
    lineas.append(f"PREGUNTA ACTUAL DEL USUARIO: {pregunta}")
    lineas.append("════════════════════════════════")
    lineas.append("")
    lineas.append("Instrucciones OBLIGATORIAS:")
    lineas.append("- Responde EXCLUSIVAMENTE a la pregunta actual.")
    lineas.append("- Responde en español, breve, máximo 3 o 4 frases.")
    lineas.append("- El usuario ya tiene las respuestas anteriores, NO las repitas.")
    lineas.append("- Usa el contexto solo para entender referencias como 'eso', 'él', 'lo anterior'.")
    lineas.append("- Si la pregunta actual depende de la pregunta anterior, relaciónalas.")
    lineas.append("- IMPORTANTE: Responde siempre en texto plano, sin usar Markdown.")
    lineas.append("- No uses asteriscos, almohadillas, guiones de lista, negritas, cursivas ni ningún otro formato.")
    lineas.append("- Escribe como si fuera un texto que se va a leer en voz alta..")
    lineas.append("")

    # Priorizar historial de sesión porque el guardado en DB ahora es asíncrono
    ultima = historial_sesion[-1] if historial_sesion else (
        historial_db[-1] if historial_db else None
    )

    if ultima:
        lineas.append("── Última interacción previa (contexto prioritario) ──")
        lineas.append(f"  · Pregunta anterior: '{ultima.get('pregunta', '')[:120]}'")
        respuesta_anterior = ultima.get("respuesta", "")
        if respuesta_anterior:
            lineas.append(f"  · Respuesta anterior: '{respuesta_anterior[:180]}'")
        lineas.append("")

    ultima_pregunta = ultima.get("pregunta") if isinstance(ultima, dict) else None

    relevantes_filtrados = [
        r for r in relevantes
        if ultima is None or r.get("pregunta") != ultima_pregunta
    ]

    if relevantes_filtrados:
        lineas.append("── Contexto adicional relevante ──")
        for h in relevantes_filtrados:
            lineas.append(f"  · '{h.get('pregunta', '')[:80]}'")
        lineas.append("")

    if historial_sesion:
        lineas.append("── Conversación de esta sesión ──")
        for t in historial_sesion[-3:]:
            lineas.append(f"  Usuario: {t.get('pregunta', '')}")
            lineas.append(f"  Asistente: {t.get('respuesta', '')}")
        lineas.append("")

    lineas.append("════════════════════════════════")
    lineas.append(f"RESPONDE SOLO A ESTO: {pregunta}")
    lineas.append("Si la pregunta depende del contexto anterior, úsalo.")
    lineas.append("No repitas literalmente respuestas previas salvo que sea necesario para aclarar.")
    lineas.append("════════════════════════════════")

    return "\n".join(lineas)


def ask_gemini(pregunta, relevantes, historial_sesion, historial_db):
    prompt = construir_prompt(pregunta, relevantes, historial_sesion, historial_db)
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    modelos = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview"
    ]

    for modelo in modelos:
        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{modelo}:generateContent?key={GEMINI_API_KEY}"
            )
            r = requests_session.post(url, json=payload, timeout=8)

            if r.status_code == 200:
                logger.info(f"🤖 GEMINI OK (Modelo: {modelo})")
                data = r.json()
                respuesta = data["candidates"][0]["content"]["parts"][0]["text"]
                return respuesta

            if r.status_code == 429:
                logger.warning(
                    f"⚠️ Cuota excedida para {modelo}. Reintentando con el siguiente..."
                )
                continue

            logger.error(f"❌ Error HTTP {r.status_code} en {modelo}: {r.text}")

        except Exception as e:
            logger.error(f"❌ Error de conexión en {modelo}: {e}", exc_info=True)

    return None


def ask_brave(pregunta, relevantes, historial_sesion, historial_db):
    headers = {
        "Content-Type": "application/json",
        "x-subscription-token": BRAVE_API_KEY
    }

    prompt_completo = (
        f"Busca en internet información actual para responder brevemente a esta pregunta: {pregunta}\n\n"
        "---\n"
        "REGLAS:\n"
        "- Responde en español.\n"
        "- Máximo 3 frases.\n"
        "- Si la pregunta depende de la inmediatamente anterior, usa ese contexto.\n"
        "- No repitas innecesariamente todo el historial.\n"
    )

    # Priorizar historial de sesión porque el guardado en DB ahora es asíncrono
    ultima = historial_sesion[-1] if historial_sesion else (
        historial_db[-1] if historial_db else None
    )

    if ultima:
        prompt_completo += (
            f"\nÚltima pregunta del usuario: '{ultima.get('pregunta', '')[:120]}'.\n"
        )
        if ultima.get("respuesta"):
            prompt_completo += (
                f"Última respuesta dada: '{ultima.get('respuesta', '')[:180]}'.\n"
            )

    ultima_pregunta = ultima.get("pregunta") if isinstance(ultima, dict) else None

    relevantes_filtrados = [
        r for r in relevantes
        if ultima is None or r.get("pregunta") != ultima_pregunta
    ]

    if relevantes_filtrados:
        prompt_completo += (
            "Temas relacionados ya tratados: " +
            "; ".join([f"'{h.get('pregunta', '')[:60]}'" for h in relevantes_filtrados]) +
            "\n"
        )

    if historial_sesion:
        prompt_completo += "Historial reciente:\n"
        for t in historial_sesion[-3:]:
            prompt_completo += f"Usuario: {t.get('pregunta', '')}\n"
            prompt_completo += f"Asistente: {t.get('respuesta', '')}\n"

    mensajes = [{"role": "user", "content": prompt_completo}]

    payload = {
        "stream": False,
        "model": "brave",
        "messages": mensajes,
        "country": "ES",
        "language": "es"
    }

    try:
        r = requests_session.post(
            "https://api.search.brave.com/res/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )

        if r.status_code != 200:
            logger.error(f"❌ BRAVE HTTP ERROR {r.status_code}: {r.text}")
            return None

        logger.info(f"🌐 BRAVE status: {r.status_code}")
        data = r.json()
        respuesta = data["choices"][0]["message"]["content"]
        logger.info(f"🌐 BRAVE OK | {respuesta[:100]}")
        return respuesta

    except Exception as e:
        logger.error(f"❌ BRAVE ERROR: {e}", exc_info=True)
        return None


def lanzar_guardado_asincrono(user_id, pregunta, respuesta, embedding=None):
    """
    Invoca la MISMA Lambda de forma asíncrona para guardar el historial
    sin bloquear la respuesta de Alexa.
    """
    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME")

    if not function_name:
        logger.warning("⚠️ AWS_LAMBDA_FUNCTION_NAME no disponible. Guardando en sync.")
        return guardar_en_db(user_id, pregunta, respuesta, embedding=embedding)

    event = {
        "action": "save_history",
        "payload": {
            "user_id": user_id,
            "pregunta": pregunta,
            "respuesta": respuesta,
            "embedding": embedding
        }
    }

    lambda_client.invoke(
        FunctionName=function_name,
        InvocationType="Event",
        Payload=json.dumps(event).encode("utf-8")
    )

    logger.info("📨 Guardado lanzado de forma asíncrona")
    return True


def process_async_event(event):
    """
    Procesa el evento asíncrono de guardado.
    """
    payload = event.get("payload", {})

    ok = guardar_en_db(
        user_id=payload.get("user_id", ""),
        pregunta=payload.get("pregunta", ""),
        respuesta=payload.get("respuesta", ""),
        embedding=payload.get("embedding")
    )

    if not ok:
        raise RuntimeError("Fallo en guardado asíncrono en DynamoDB")

    return {"statusCode": 202, "body": "ok"}





class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        logger.info("🚀 LaunchRequest")
        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["historial"] = []
        session_attr["modo"] = None

        return (
            handler_input.response_builder
            .speak("Hola, soy tu asistente. ¿Vas a hacer preguntas sobre información actual?")
            .ask("Di sí para internet, no para conocimiento general.")
            .response
        )
class SiIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("SiIntent")(handler_input)

    def handle(self, handler_input):
        session_attr = handler_input.attributes_manager.session_attributes
        if session_attr.get("modo") is None:
            session_attr["modo"] = "brave"
            logger.info("🔄 Modo: brave")
        return (
            handler_input.response_builder
            .speak("Perfecto, usaré internet. Dime tu pregunta.")
            .ask("Puedes preguntarme lo que quieras.")
            .response
        )


class NoIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("NoIntent")(handler_input)

    def handle(self, handler_input):
        session_attr = handler_input.attributes_manager.session_attributes
        if session_attr.get("modo") is None:
            session_attr["modo"] = "gemini"
            logger.info("🔄 Modo: gemini")
            return (
                handler_input.response_builder
                .speak("Entendido, usaré conocimiento general. Dime tu pregunta.")
                .ask("Puedes preguntarme lo que quieras.")
                .response
            )
        logger.info("🛑 NoIntent tras respuesta → cerrando")
        return (
            handler_input.response_builder
            .speak("¡Hasta luego!")
            .set_should_end_session(True)
            .response
        )
class NadaMasIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("NadaMasIntent")(handler_input)

    def handle(self, handler_input):
        logger.info("🛑 NadaMasIntent → cerrando")
        return (
            handler_input.response_builder
            .speak("¡Hasta luego!")
            .set_should_end_session(True)
            .response
        )


class GeminiQueryIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("GeminiQueryIntent")(handler_input)

    def handle(self, handler_input):
        t0 = perf_counter()
        session_attr = handler_input.attributes_manager.session_attributes
        user_id = handler_input.request_envelope.session.user.user_id
        historial = session_attr.get("historial", [])
        modo = session_attr.get("modo", "gemini")

        intent = handler_input.request_envelope.request.intent
        query_slot = intent.slots.get("query") if intent and intent.slots else None
        query = query_slot.value.strip() if query_slot and query_slot.value else None

        if not query:
            return (
                handler_input.response_builder
                .speak("No he captado bien la pregunta. Dímela de nuevo.")
                .ask("Puedes preguntarme lo que quieras.")
                .response
            )

        PALABRAS_CIERRE = {
            "para", "nada", "nada mas", "nada más", "adios", "adiós",
            "hasta luego", "salir", "terminar", "fin", "es todo",
            "ya esta", "ya está"
        }
        if query.lower() in PALABRAS_CIERRE:
            logger.info(f"🛑 Palabra de cierre: '{query}'")
            return (
                handler_input.response_builder
                .speak("¡Hasta luego!")
                .set_should_end_session(True)
                .response
            )

        logger.info(f"❓ PREGUNTA | modo:{modo} | query:{query[:60]}")

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_historial_db = executor.submit(obtener_historial_db, user_id)
            future_emb = executor.submit(generar_embedding, query)
            historial_db = future_historial_db.result()
            emb = future_emb.result()

        relevantes = buscar_relevantes(emb, historial_db) if emb else []

        if modo == "brave":
            respuesta = ask_brave(query, relevantes, historial, historial_db)
        else:
            respuesta = ask_gemini(query, relevantes, historial, historial_db)

        if not respuesta:
            respuesta = "Lo siento, no pude procesar tu petición."

        historial.append({"pregunta": query, "respuesta": respuesta})
        session_attr["historial"] = historial[-5:]

        try:
            lanzar_guardado_asincrono(user_id, query, respuesta, embedding=emb)
        except Exception as e:
            logger.error(f"❌ Error lanzando guardado asíncrono: {e}", exc_info=True)
            guardar_en_db(user_id, query, respuesta, embedding=emb)

        logger.info(f"⏱️ Tiempo total request: {round(perf_counter() - t0, 3)}s")

        return (
            handler_input.response_builder
            .speak(f"{respuesta} ¿Qué más quieres saber?")
            .ask("Puedes hacerme otra pregunta.")
            .response
        )



class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        logger.info("🔚 SessionEnded")
        return handler_input.response_builder.response


class StopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return (
            ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input) or
            ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input)
        )

    def handle(self, handler_input):
        logger.info("🛑 Stop/Cancel")
        return (
            handler_input.response_builder
            .speak("¡Hasta luego!")
            .set_should_end_session(True)
            .response
        )

class HelpIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        session_attr = handler_input.attributes_manager.session_attributes
        if session_attr.get("modo") in ("brave", "gemini"):
            return (
                handler_input.response_builder
                .speak("Puedes preguntarme lo que quieras. Di por ejemplo: dime, quién es Einstein.")
                .ask("Dime tu pregunta.")
                .response
            )
        return (
            handler_input.response_builder
            .speak("Di sí para usar internet, no para conocimiento general.")
            .ask("Di sí para internet, no para conocimiento general.")
            .response
        )


class FallbackIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input):
        session_attr = handler_input.attributes_manager.session_attributes
        if session_attr.get("modo") in ("brave", "gemini"):
            return (
                handler_input.response_builder
                .speak("No te entendí. Para preguntar di: dime, pregunta, o explícame, seguido de tu pregunta.")
                .ask("Dime tu pregunta.")
                .response
            )
        return (
            handler_input.response_builder
            .speak("No te entendí. ¿Quieres usar internet? Di sí o no.")
            .ask("Di sí para internet, no para conocimiento general.")
            .response
        )

class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.error(f"❌ EXCEPCIÓN: {exception}", exc_info=True)
        return (
            handler_input.response_builder
            .speak("Ha ocurrido un error. Inténtalo de nuevo.")
            .response
        )


sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GeminiQueryIntentHandler())
sb.add_request_handler(SiIntentHandler())
sb.add_request_handler(NoIntentHandler())
sb.add_request_handler(NadaMasIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_request_handler(StopIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(FallbackIntentHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

alexa_handler = sb.lambda_handler()


def lambda_handler(event, context):
    """
    Router principal:
    - Si llega un evento propio con action=save_history -> guardado asíncrono
    - Si no, procesa el evento normal de Alexa
    """
    try:
        if isinstance(event, dict) and event.get("action") == "save_history":
            logger.info("📥 Evento asíncrono de persistencia")
            return process_async_event(event)

        return alexa_handler(event, context)

    except Exception as e:
        logger.error(f"❌ Error en lambda_handler principal: {e}", exc_info=True)
        raise