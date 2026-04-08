import base64
import contextvars
import io
import logging
import os
import json
import tempfile
from typing import Any

from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http import models as qdrant_models

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.readers.file import PDFReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from openai import OpenAI
import anyio

load_dotenv()

APP_NAME = "IBA HR Policy Assistant"
SYSTEM_PROMPT = (
    "You are the IBA HR Policy Assistant (male voice). "
    "Reply in the same language as the user. "
    "If the user writes in Roman Urdu, reply in Roman Urdu. "
    "If the user writes in English, reply in English. "
    "If the user mixes, reply in a light mix. "
    "Be funny, friendly, and concise. "
    "Avoid repetitive closings like 'How can I assist you today?' and only ask a follow-up if needed. "
    "Answer employee HR policy questions clearly using the policy documents. "
    "If the user asks about academic dates or calendars, say you will check live sources. "
    "If the answer is not in the documents, say you don't have it and ask what else you can help with."
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "iba_hr_policies")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.05"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))
logger = logging.getLogger("iba_hr_assistant")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required in the environment.")
if not QDRANT_URL:
    raise RuntimeError("QDRANT_URL is required in the environment.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

Settings.llm = LlamaOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
    system_prompt=SYSTEM_PROMPT,
)
Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=120)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection() -> None:
    try:
        qdrant_client.get_collection(QDRANT_COLLECTION)
    except UnexpectedResponse as exc:
        if getattr(exc, "status_code", None) == 403:
            raise RuntimeError(
                "Qdrant authorization failed (403)."
                "Check QDRANT_URL/QDRANT_API_KEY in .env."
            ) from exc
        raise
    except Exception:
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qdrant_models.VectorParams(
                size=EMBED_DIM,
                distance=qdrant_models.Distance.COSINE,
            ),
        )


ensure_collection()

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION,
)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskTextRequest(BaseModel):
    text: str


class DeletePolicyRequest(BaseModel):
    file_name: str


SESSION_MEMORY: dict[str, ChatMemoryBuffer] = {}
LANGUAGE_CTX = contextvars.ContextVar("language", default="english")
META_SYSTEM_PROMPT = (
    "Classify the user's request. Return ONLY valid JSON with keys "
    "`intent` and `language`.\n\n"
    "intent must be one of: policy_question, list_policies, live_dates, other.\n"
    "language must be one of: english, roman, mixed (roman = Urdu in English letters).\n"
    "Example: {\"intent\":\"policy_question\",\"language\":\"roman\"}"
)
NAME_SYSTEM_PROMPT = (
    "Extract the user's name if they shared it. Return ONLY valid JSON with key "
    "`name`. If no name is present, return {\"name\": \"\"}.\n"
    "Examples:\n"
    "User: \"my name is Mujtaba\" -> {\"name\":\"Mujtaba\"}\n"
    "User: \"main Ali hoon\" -> {\"name\":\"Ali\"}\n"
    "User: \"tell me about hajj\" -> {\"name\":\"\"}"
)


def require_admin(token: str | None) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_TOKEN is not configured on the server.",
        )
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token.")


def query_hr_policies(question: str):
    try:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="version", operator=FilterOperator.EQ, value="current")
            ]
        )
        query_engine = index.as_query_engine(filters=filters, similarity_top_k=3)
        return query_engine.query(question)
    except Exception as exc:
        raise RuntimeError("Failed to query HR policy index.") from exc


def iba_web_scraper_tool(question: str) -> str:
    return (
        "Live web search is not configured yet. "
        "Please add a search provider to fetch IBA dates."
    )


def list_current_policies() -> list[dict[str, str | None]]:
    results: dict[str, dict[str, str | None]] = {}
    next_offset = None
    filter_payload = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="version",
                match=qdrant_models.MatchValue(value="current"),
            )
        ]
    )
    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=200,
            offset=next_offset,
            with_payload=True,
            scroll_filter=filter_payload,
        )
        for point in points:
            payload = point.payload or {}
            file_name = payload.get("file_name")
            effective_date = payload.get("effective_date")
            if not file_name:
                continue
            if file_name not in results:
                results[file_name] = {
                    "file_name": file_name,
                    "effective_date": effective_date,
                }
            elif not results[file_name].get("effective_date") and effective_date:
                results[file_name]["effective_date"] = effective_date
        if next_offset is None:
            break
    return [results[name] for name in sorted(results.keys())]


def list_policies_tool() -> str:
    """List current policy documents with effective dates."""
    language = LANGUAGE_CTX.get()
    policies = list_current_policies()
    if not policies:
        if language == "roman":
            return "Abhi koi current policy documents nazar nahi aa rahe."
        if language == "mixed":
            return "Abhi koi current policy documents nazar nahi aa rahe. (No documents yet.)"
        return "I don't see any current policy documents yet."
    lines = [f"- {policy['file_name']}" for policy in policies]
    if language == "roman":
        return "Yeh current policy documents hain:\n" + "\n".join(lines)
    if language == "mixed":
        return "Here are current policy documents / Yeh current policy documents hain:\n" + "\n".join(lines)
    return "Here are the current policy documents:\n" + "\n".join(lines)


def live_dates_tool(question: str) -> str:
    """Answer questions about academic calendars and dates."""
    language = LANGUAGE_CTX.get()
    if language == "roman":
        return (
            "Live web search abhi configured nahi hai. "
            "IBA dates ke liye search provider add karna hoga."
        )
    if language == "mixed":
        return (
            "Live web search abhi configured nahi hai. "
            "Please add a search provider for IBA dates."
        )
    return iba_web_scraper_tool(question)


def extract_sources(source_nodes: list[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, int | None]] = set()
    for node in source_nodes:
        metadata = getattr(node, "metadata", None) or {}
        file_name = metadata.get("file_name")
        if not file_name:
            continue
        entry = {"file_name": file_name}
        effective_date = metadata.get("effective_date")
        if effective_date:
            entry["effective_date"] = effective_date
        if "page" in metadata:
            entry["page"] = metadata["page"]
        key = (
            file_name,
            entry.get("effective_date"),
            entry.get("page"),
        )
        if key in seen:
            continue
        seen.add(key)
        results.append(entry)
    return results


def classify_meta_llm(query: str) -> tuple[str, str]:
    try:
        completion = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": META_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=24,
            response_format={"type": "json_object"},
        )
        payload = completion.choices[0].message.content.strip()
        data = json.loads(payload)
        intent = str(data.get("intent", "")).lower()
        language = str(data.get("language", "")).lower()
        if intent not in {"policy_question", "list_policies", "live_dates", "other"}:
            intent = "policy_question"
        if language not in {"english", "roman", "mixed"}:
            language = "english"
        return intent, language
    except Exception:
        return "policy_question", "english"


def chunk_text(text: str, size: int = 80) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def format_name_prompt(language: str) -> str:
    if language == "roman":
        return "Agar aap chahein to apna naam bata dein."
    if language == "mixed":
        return "Agar aap chahein to apna naam bata dein. (If you'd like, share your name.)"
    return "If you'd like, share your name."


def should_append_name_prompt(answer: str) -> bool:
    lowered = answer.lower()
    return "name" not in lowered and "naam" not in lowered


def get_user_name(memory: ChatMemoryBuffer | None) -> str | None:
    if memory is None:
        return None
    for message in memory.get_all():
        role = getattr(message.role, "value", message.role)
        if role != "system":
            continue
        content = str(message.content or "")
        if content.lower().startswith("user name:"):
            return content.split(":", 1)[1].strip() or None
    return None


def remember_user_name(memory: ChatMemoryBuffer, name: str) -> None:
    memory.put(ChatMessage(role="system", content=f"User name: {name.strip()}"))


def extract_name_llm(query: str) -> str | None:
    try:
        completion = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": NAME_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=24,
            response_format={"type": "json_object"},
        )
        payload = completion.choices[0].message.content.strip()
        data = json.loads(payload)
        name = str(data.get("name", "")).strip()
        return name or None
    except Exception:
        return None


def build_openai_messages(
    query: str, memory: ChatMemoryBuffer | None
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    history = memory.get() if memory else []
    for msg in history:
        role = getattr(msg.role, "value", msg.role)
        if role not in {"user", "assistant", "system"}:
            continue
        content = msg.content
        if content is None:
            continue
        messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": query})
    return messages


def transcribe_audio(audio_bytes: bytes, filename: str | None) -> str:
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename or "audio.webm"
    try:
        transcription = openai_client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
        )
        return transcription.text
    except Exception as exc:
        raise RuntimeError("Failed to transcribe audio.") from exc


def synthesize_audio(text: str) -> str:
    try:
        response = openai_client.audio.speech.create(
            model=TTS_MODEL,
            voice="onyx",
            input=text,
            speed=TTS_SPEED,
        )
        audio_bytes = response.content
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("Failed to generate TTS audio.") from exc


@app.post("/api/v1/ask")
async def ask(
    request: Request,
    audio: UploadFile | None = File(default=None),
    text_payload: AskTextRequest | None = Body(default=None),
    x_session_id: str | None = Header(default=None),
):
    try:
        if audio is not None:
            audio_bytes = await audio.read()
            if not audio_bytes:
                raise HTTPException(status_code=400, detail="Audio file is empty.")
            query = await anyio.to_thread.run_sync(
                transcribe_audio, audio_bytes, audio.filename
            )
        elif text_payload and text_payload.text:
            query = text_payload.text.strip()
        else:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                data = await request.json()
            else:
                data = {}
            query = (data or {}).get("text", "").strip()

        if not query:
            raise HTTPException(status_code=400, detail="Query text is required.")

        session_id = x_session_id or "default"
        memory = SESSION_MEMORY.get(session_id)
        if memory is None:
            memory = ChatMemoryBuffer.from_defaults(llm=Settings.llm)
            SESSION_MEMORY[session_id] = memory
        known_name = get_user_name(memory)
        extracted_name = await anyio.to_thread.run_sync(extract_name_llm, query)
        if not known_name and extracted_name:
            remember_user_name(memory, extracted_name)
            known_name = extracted_name
        intent, language = await anyio.to_thread.run_sync(classify_meta_llm, query)
        language_token = LANGUAGE_CTX.set(language)
        try:
            if intent == "list_policies":
                answer = list_policies_tool()
                sources = []
            elif intent == "live_dates":
                answer = live_dates_tool(query)
                sources = []
            elif intent == "policy_question":
                response = await anyio.to_thread.run_sync(query_hr_policies, query)
                answer = response.response
                sources = extract_sources(response.source_nodes or [])
            else:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                messages.extend(build_openai_messages(query, memory))
                completion = openai_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    temperature=0.4,
                )
                answer = completion.choices[0].message.content.strip()
                sources = []
        finally:
            LANGUAGE_CTX.reset(language_token)
        if not known_name and intent == "other" and should_append_name_prompt(answer):
            answer = f"{answer}\n\n{format_name_prompt(language)}"
        await memory.aput(ChatMessage(role="user", content=query))
        await memory.aput(ChatMessage(role="assistant", content=answer))
        audio_base64 = await anyio.to_thread.run_sync(synthesize_audio, answer)

        return {
            "answer": answer,
            "audio_base64": audio_base64,
            "sources": sources,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ask request failed.")
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/api/v1/ask-stream")
async def ask_stream(
    request: Request,
    text_payload: AskTextRequest | None = Body(default=None),
    x_session_id: str | None = Header(default=None),
):
    if text_payload and text_payload.text:
        query = text_payload.text.strip()
    else:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            data = await request.json()
        else:
            data = {}
        query = (data or {}).get("text", "").strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query text is required.")

    session_id = x_session_id or "default"
    memory = SESSION_MEMORY.get(session_id)
    if memory is None:
        memory = ChatMemoryBuffer.from_defaults(llm=Settings.llm)
        SESSION_MEMORY[session_id] = memory

    known_name = get_user_name(memory)
    extracted_name = await anyio.to_thread.run_sync(extract_name_llm, query)
    if not known_name and extracted_name:
        remember_user_name(memory, extracted_name)
        known_name = extracted_name
    intent, language = await anyio.to_thread.run_sync(classify_meta_llm, query)

    async def event_stream():
        answer_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        language_token = LANGUAGE_CTX.set(language)
        try:
            if intent == "list_policies":
                answer = list_policies_tool()
                for chunk in chunk_text(answer):
                    answer_parts.append(chunk)
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk}, ensure_ascii=False)}\n\n"
            elif intent == "live_dates":
                answer = live_dates_tool(query)
                for chunk in chunk_text(answer):
                    answer_parts.append(chunk)
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk}, ensure_ascii=False)}\n\n"
            elif intent == "policy_question":
                filters = MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="version", operator=FilterOperator.EQ, value="current"
                        )
                    ]
                )
                query_engine = index.as_query_engine(
                    filters=filters, similarity_top_k=3, streaming=True
                )
                response = query_engine.query(query)
                response_gen = getattr(response, "response_gen", None)
                if response_gen is not None:
                    for chunk in response_gen:
                        if not chunk:
                            continue
                        answer_parts.append(chunk)
                        yield f"data: {json.dumps({'type': 'chunk', 'text': chunk}, ensure_ascii=False)}\n\n"
                sources = extract_sources(response.source_nodes or [])
            else:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                messages.extend(build_openai_messages(query, memory))
                completion = openai_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    temperature=0.4,
                    stream=True,
                )
                for chunk in completion:
                    delta = chunk.choices[0].delta
                    token = getattr(delta, "content", None)
                    if not token:
                        continue
                    answer_parts.append(token)
                    yield f"data: {json.dumps({'type': 'chunk', 'text': token}, ensure_ascii=False)}\n\n"

            answer = "".join(answer_parts).strip()
            if not answer and intent == "policy_question":
                response = await anyio.to_thread.run_sync(query_hr_policies, query)
                answer = response.response
                sources = extract_sources(response.source_nodes or [])
                for chunk in chunk_text(answer):
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk}, ensure_ascii=False)}\n\n"
            if not known_name and intent == "other" and should_append_name_prompt(answer):
                name_prompt = format_name_prompt(language)
                answer = f"{answer}\n\n{name_prompt}" if answer else name_prompt
                for chunk in chunk_text(name_prompt):
                    answer_parts.append(chunk)
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk}, ensure_ascii=False)}\n\n"

            await memory.aput(ChatMessage(role="user", content=query))
            await memory.aput(ChatMessage(role="assistant", content=answer))
            yield f"data: {json.dumps({'type': 'done', 'sources': sources}, ensure_ascii=False)}\n\n"
            try:
                audio_base64 = await anyio.to_thread.run_sync(synthesize_audio, answer)
                yield f"data: {json.dumps({'type': 'audio', 'audio_base64': audio_base64}, ensure_ascii=False)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'audio_error', 'message': str(exc)})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            LANGUAGE_CTX.reset(language_token)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/v1/upload-policy")
async def upload_policy(
    version: str = Form(default="current"),
    effective_date: str | None = Form(default=None),
    file: UploadFile = File(...),
    x_admin_token: str | None = Header(default=None),
):
    require_admin(x_admin_token)
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, file.filename)
            with open(temp_path, "wb") as temp_file:
                temp_file.write(await file.read())

            documents = SimpleDirectoryReader(
                input_files=[temp_path],
                filename_as_id=True,
                file_extractor={".pdf": PDFReader()},
            ).load_data()

            for doc in documents:
                doc.metadata.update(
                    {
                        "source_type": "document",
                        "file_name": file.filename,
                        "version": version,
                        "effective_date": effective_date,
                    }
                )
                await anyio.to_thread.run_sync(index.insert, doc)

        return {"status": "success", "message": f"Uploaded {file.filename}."}
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Failed to upload policy."
        ) from exc


@app.delete("/api/v1/delete-policy")
async def delete_policy(
    payload: DeletePolicyRequest,
    x_admin_token: str | None = Header(default=None),
):
    require_admin(x_admin_token)
    try:
        qdrant_client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="file_name",
                        match=qdrant_models.MatchValue(value=payload.file_name),
                    )
                ]
            ),
        )
        return {
            "status": "success",
            "message": f"Deleted {payload.file_name}.",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Failed to delete policy."
        ) from exc


@app.get("/api/v1/policies")
async def list_policies(x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    try:
        results: dict[str, dict[str, str | None]] = {}
        next_offset = None
        while True:
            points, next_offset = qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=200,
                offset=next_offset,
                with_payload=True,
            )
            for point in points:
                payload = point.payload or {}
                file_name = payload.get("file_name")
                version = payload.get("version")
                effective_date = payload.get("effective_date")
                if file_name:
                    if file_name not in results:
                        results[file_name] = {
                            "file_name": file_name,
                            "version": version or "current",
                            "effective_date": effective_date,
                        }
                    elif not results[file_name].get("effective_date") and effective_date:
                        results[file_name]["effective_date"] = effective_date
            if next_offset is None:
                break
        return [
            results[name] for name in sorted(results.keys())
        ]
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Failed to list policies."
        ) from exc