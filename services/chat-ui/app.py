import os
import logging

import chainlit as cl
from chainlit.element import Element
import httpx
import engineio
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("chat-ui")

# Increase engineio max buffer size to handle larger payloads
engineio.async_server.AsyncServer.max_http_buffer_size = 1024 * 1024 * 1024  # 1GB

import re

API_URL = os.getenv("PERCEIVE8_API_URL", "http://localhost:8000")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "default-user")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "auto")

AUDIO_MIME_PREFIXES = ("audio/",)
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4")


def _is_audio_element(el: Element) -> bool:
    if el.mime and el.mime.startswith(AUDIO_MIME_PREFIXES):
        return True
    if el.name:
        return any(el.name.lower().endswith(ext) for ext in AUDIO_EXTENSIONS)
    return False


def _parse_language_prefix(text: str) -> tuple[str, str]:
    """Extract optional lang:XX prefix from message text.

    Returns (language, remaining_text). Falls back to DEFAULT_LANGUAGE.
    """
    match = re.match(r"^lang:(\w+)\s+(.+)$", text.strip(), re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()
    return DEFAULT_LANGUAGE, text.strip()


def _looks_like_file_path(text: str) -> bool:
    """Check if the message text looks like a local file path."""
    _, stripped = _parse_language_prefix(text)
    if any(stripped.lower().endswith(ext) for ext in AUDIO_EXTENSIONS):
        return True
    if stripped.startswith("/") or stripped.startswith("~"):
        return True
    return False


@cl.on_chat_start
async def on_chat_start():
    logger.info("Chat session started")
    cl.user_session.set("analysis_ids", [])

    # Health check
    logger.info(f"Checking perceive8 API at {API_URL}...")
    api_status = ""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            health_resp = await client.get(f"{API_URL}/health")
            logger.info(f"Health check response: {health_resp.status_code} - {health_resp.text[:200]}")
            api_status = f"\n\n✅ Connected to perceive8 API at `{API_URL}`"
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        api_status = f"\n\n❌ Cannot reach perceive8 API at `{API_URL}`"

    await cl.Message(
        content=(
            "👋 Welcome to **perceive8 Chat**!\n\n"
            "You can:\n"
            "- **Upload small audio files** by attaching them (drag & drop or use the attachment button)\n"
            "- **Paste a file path** for large files (e.g., `/Users/you/Downloads/recording.wav`)\n"
            "- **Ask questions** about the analyzed audio content\n"
            f"- **Override language** by prefixing with `lang:XX` (e.g., `lang:he /path/to/file.wav`)\n\n"
            f"Default language: **{DEFAULT_LANGUAGE}** (auto-detection)\n\n"
            "Start by uploading an audio file or pasting its path, then ask anything about it!"
            + api_status
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    # Check if the message text looks like a file path first (handles large files)
    if _looks_like_file_path(message.content):
        language, file_path = _parse_language_prefix(message.content)
        await _handle_file_path(file_path, language)
        return

    audio_elements = [el for el in (message.elements or []) if _is_audio_element(el)]

    if audio_elements:
        language, _ = _parse_language_prefix(message.content)
        await _handle_audio_upload(audio_elements, language)
    else:
        await _handle_question(message.content)


async def _handle_file_path(file_path: str, language: str = DEFAULT_LANGUAGE):
    """Read an audio file from a local path and send it to the API."""
    file_path = os.path.expanduser(file_path)

    if not os.path.isfile(file_path):
        await cl.Message(content=f"❌ File not found: `{file_path}`").send()
        return

    file_name = os.path.basename(file_path)
    if not any(file_name.lower().endswith(ext) for ext in AUDIO_EXTENSIONS):
        await cl.Message(
            content=f"⚠️ `{file_name}` doesn't look like a supported audio file. Supported extensions: {', '.join(AUDIO_EXTENSIONS)}"
        ).send()
        return

    file_size = os.path.getsize(file_path)
    logger.info(f"File path detected: {file_path}, size: {file_size} bytes")

    analysis_ids: list[str] = cl.user_session.get("analysis_ids")

    # Progress message that we'll update as we go
    msg = cl.Message(content="")
    await msg.send()

    # Step 1: Read file
    size_mb = file_size / (1024 * 1024)
    msg.content = f"📁 Reading file: **{file_name}** ({size_mb:.1f} MB)..."
    await msg.update()

    try:
        file_bytes = open(file_path, "rb").read()
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        msg.content = f"❌ Failed to read file: {e}"
        await msg.update()
        return

    # Step 2: Upload
    msg.content = f"📁 Reading file: **{file_name}** ({size_mb:.1f} MB)... ✓\n⬆️ Uploading to perceive8 API..."
    await msg.update()

    logger.info(f"Sending file to perceive8 API at {API_URL}/analysis/analyze")

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with cl.Step(name="Analyzing", type="tool") as step:
                step.output = f"Uploading **{file_name}** ({size_mb:.1f} MB) to perceive8 for analysis..."
                form_data = {"user_id": DEFAULT_USER_ID}
                if language != "auto":
                    form_data["language"] = language
                resp = await client.post(
                    f"{API_URL}/analysis/analyze",
                    data=form_data,
                    files={"audio_file": (file_name, file_bytes)},
                )
                logger.info(f"API response status: {resp.status_code}, body: {resp.text[:500]}")
                resp.raise_for_status()
                result = resp.json()
                step.output = "✅ Analysis complete!"

            analysis_id = result["id"]
            analysis_ids.append(analysis_id)
            cl.user_session.set("analysis_ids", analysis_ids)

            lang = result.get("language", "unknown")
            msg.content = (
                f"📁 Reading file: **{file_name}** ({size_mb:.1f} MB)... ✓\n"
                f"⬆️ Uploading to perceive8 API... ✓\n"
                f"✅ **Analysis complete!**\n\n"
                f"- Analysis ID: `{analysis_id}`\n"
                f"- Language: {lang}\n\n"
                "You can now ask questions about this audio."
            )
            await msg.update()
        except httpx.HTTPStatusError as e:
            logger.error(f"API call failed: {e}")
            msg.content = f"❌ Failed to analyze **{file_name}**: {e.response.status_code} - {e.response.text}"
            await msg.update()
        except Exception as e:
            logger.error(f"API call failed: {e}")
            msg.content = f"❌ Error analyzing **{file_name}**: {e}"
            await msg.update()


async def _handle_audio_upload(elements: list[Element], language: str = DEFAULT_LANGUAGE):
    analysis_ids: list[str] = cl.user_session.get("analysis_ids")

    async with httpx.AsyncClient(timeout=None) as client:
        for el in elements:
            msg = cl.Message(content="")
            await msg.send()

            # Step 1: Read file
            file_bytes = el.content if isinstance(el.content, bytes) else open(el.path, "rb").read()
            size_mb = len(file_bytes) / (1024 * 1024)

            logger.info(f"File path detected: {el.name}, size: {len(file_bytes)} bytes")

            msg.content = f"📁 Reading **{el.name}** ({size_mb:.1f} MB)... ✓\n⬆️ Uploading to perceive8 API..."
            await msg.update()

            logger.info(f"Sending file to perceive8 API at {API_URL}/analysis/analyze")

            try:
                async with cl.Step(name="Analyzing", type="tool") as step:
                    step.output = f"Uploading **{el.name}** ({size_mb:.1f} MB) to perceive8 for analysis..."
                    form_data = {"user_id": DEFAULT_USER_ID}
                    if language != "auto":
                        form_data["language"] = language
                    resp = await client.post(
                        f"{API_URL}/analysis/analyze",
                        data=form_data,
                        files={"audio_file": (el.name, file_bytes)},
                    )
                    logger.info(f"API response status: {resp.status_code}, body: {resp.text[:500]}")
                    resp.raise_for_status()
                    result = resp.json()
                    step.output = "✅ Analysis complete!"

                analysis_id = result["id"]
                analysis_ids.append(analysis_id)
                cl.user_session.set("analysis_ids", analysis_ids)

                lang = result.get("language", "unknown")
                msg.content = (
                    f"📁 Reading **{el.name}** ({size_mb:.1f} MB)... ✓\n"
                    f"⬆️ Uploading to perceive8 API... ✓\n"
                    f"✅ **Analysis complete!**\n\n"
                    f"- Analysis ID: `{analysis_id}`\n"
                    f"- Language: {lang}\n\n"
                    "You can now ask questions about this audio."
                )
                await msg.update()
            except httpx.HTTPStatusError as e:
                logger.error(f"API call failed: {e}")
                msg.content = f"❌ Failed to analyze **{el.name}**: {e.response.status_code} - {e.response.text}"
                await msg.update()
            except Exception as e:
                logger.error(f"API call failed: {e}")
                msg.content = f"❌ Error analyzing **{el.name}**: {e}"
                await msg.update()


async def _handle_question(question: str):
    analysis_ids: list[str] = cl.user_session.get("analysis_ids")
    analysis_id = analysis_ids[0] if len(analysis_ids) == 1 else None

    logger.info(f"Sending question to {API_URL}/query: {question}")

    msg = cl.Message(content="🔍 Searching transcripts...")
    await msg.send()

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                f"{API_URL}/query",
                json={"question": question, "analysis_id": analysis_id},
            )
            logger.info(f"Query response status: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()

            answer = data.get("answer", "No answer returned.")
            content = f"💬 {answer}"

            sources = data.get("sources")
            if sources:
                content += "\n\n---\n**Sources:**\n"
                for src in sources:
                    speaker = src.get("speaker_name", "Unknown")
                    start = src.get("start_time", 0)
                    end = src.get("end_time", 0)
                    text = src.get("text", "")
                    score = src.get("relevance_score", 0)
                    content += (
                        f"\n> 🗣 **{speaker}** [{start:.1f}s – {end:.1f}s] "
                        f"(relevance: {score:.2f})\n> {text}\n"
                    )

            msg.content = content
            await msg.update()
        except httpx.HTTPStatusError as e:
            logger.error(f"API call failed: {e}")
            msg.content = f"❌ Query failed: {e.response.status_code} - {e.response.text}"
            await msg.update()
        except Exception as e:
            logger.error(f"API call failed: {e}")
            msg.content = f"❌ Error: {e}"
            await msg.update()
