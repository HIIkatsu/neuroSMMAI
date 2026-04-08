from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from pathlib import Path

import httpx

from ai_client import ai_chat
from topic_utils import detect_topic_family

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR / "generated_images"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# Higher guidance_scale (8.5) pushes model toward prompt adherence for photorealism;
# 28 inference steps balances quality vs latency for SDXL.
_DEFAULT_GUIDANCE_SCALE = 8.5
_DEFAULT_INFERENCE_STEPS = 28

_NEGATIVE_HINT = (
    "blurry, low quality, text, watermark, logo, poster, banner, illustration, "
    "clipart, collage, distorted hands, extra fingers, cropped face, duplicate person, "
    "3d render, cgi, cartoon, anime, painting, sketch, drawing, fake looking, "
    "artificial, neon lights, rgb wallpaper, unrealistic skin, bad anatomy, "
    "deformed, mutated, ugly, oversaturated"
)

# Family-aware style mapping — uses detect_topic_family() instead of keyword matching
_FAMILY_STYLE_MAP: dict[str, str] = {
    "massage": "realistic wellness photo, massage therapy, spa interior, human body care, warm natural light",
    "food": "realistic food photography, beautifully plated dish, natural light, appetizing, editorial",
    "health": "realistic wellness photo, healthy lifestyle, natural environment, warm light",
    "beauty": "realistic beauty photo, skincare product, beauty treatment, soft lighting, editorial",
    "local_business": "realistic professional photo, workshop tools, craftsman at work, natural light",
    "education": "realistic education photo, books study desk, learning environment, natural light",
    "finance": "realistic business photo, financial planning context, professional environment",
    "marketing": "realistic professional photo, marketing workspace, analytics dashboard, clean composition",
    "lifestyle": "realistic lifestyle photo, authentic daily moment, warm natural light, editorial",
    "expert_blog": "realistic professional photo, expert workspace, thoughtful, natural light",
    "cars": "realistic automotive photo, car exterior or interior, professional, natural light",
    "gaming": "realistic gaming setup photo, monitor controller desk, ambient light",
    "hardware": "realistic technology photo, computer hardware desk setup, natural lighting",
    "tech": "realistic technology photo, software development workspace, monitor, natural lighting",
    "business": "realistic professional photo, office environment, teamwork, modern business setting",
}

# Legacy keyword-level fallback for edge cases not caught by family detection
_STYLE_MAP = {
    "массаж": "realistic wellness photo, massage therapy, spa interior, human body care, warm natural light",
    "шеи": "realistic wellness photo, neck and shoulder massage, spa interior, natural skin tones",
    "спины": "realistic wellness photo, back massage, therapy room, natural skin tones",
    "осан": "realistic wellness photo, posture correction exercise, healthy back, studio light, natural skin tones",
    "упражнен": "realistic fitness wellness photo, posture exercise, stretching, studio light",
    "компьют": "realistic technology photo, modern computer desk setup, monitor keyboard, clean workspace, natural lighting",
    "ноут": "realistic technology photo, laptop on modern desk, clean workspace, natural lighting",
    "процессор": "realistic macro technology photo, computer processor hardware, motherboard detail, studio lighting",
    "видеокарт": "realistic technology photo, graphics card computer hardware, studio lighting",
    "бизнес": "realistic professional photo, office environment, teamwork, modern business setting",
    "маркет": "realistic professional photo, marketing team, analytics, modern office setting",
}


def _proxy_url() -> str | None:
    for key in ("IMAGE_PROXY_URL", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return None


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _infer_visual_style(source: str) -> str:
    """Infer visual style for AI image generation.

    First tries family-aware detection via detect_topic_family(),
    then falls back to keyword matching for edge cases.
    """
    # Family-aware detection takes priority
    family = detect_topic_family(source)
    if family in _FAMILY_STYLE_MAP:
        return _FAMILY_STYLE_MAP[family]
    # Legacy keyword-level fallback for edge cases
    s = (source or "").lower()
    for key, style in _STYLE_MAP.items():
        if key in s:
            return style
    return "realistic professional photo, natural light, high detail, people, clean composition"


def _heuristic_prompt(topic: str, prompt: str, post_text: str) -> str:
    source = " ".join(x for x in [prompt, topic, post_text[:220]] if x).strip()
    source = _clean_text(source)
    style = _infer_visual_style(source)
    subject = _clean_text(prompt.strip() or topic.strip() or "useful social media post")
    subject = re.sub(r"^(пост|сделай|создай|напиши)\s+", "", subject, flags=re.I)
    return (
        f"{style}. Main subject: {subject}. "
        "Photorealistic, shot on camera, natural light, high resolution. "
        "One clear scene, absolutely no text inside image, no logos, no collage, "
        "no 3D renders, no illustrations, no AI artifacts, no extra fingers or limbs. "
        "High quality editorial-style social media visual. "
        "Portrait-safe composition for Telegram post cover."
    )


async def build_image_prompt(
    topic: str,
    prompt: str,
    post_text: str,
    *,
    api_key: str = "",
    model: str = "",
    base_url: str | None = None,
) -> str:
    fallback = _heuristic_prompt(topic, prompt, post_text)
    api_key = (api_key or "").strip()
    model = (model or "").strip()
    if not api_key or not model:
        return fallback
    ask = (
        "Generate one concise English image prompt for a PHOTOREALISTIC, high-quality Telegram post cover image.\n"
        "Rules:\n"
        "- The image must LITERALLY and DIRECTLY depict the main subject of the post.\n"
        "- If the post is about cars — show a real car. If about servers — show a server room.\n"
        "- NEVER replace the literal subject with a metaphor, abstraction, or artistic reinterpretation.\n"
        "- Real photograph style only — no 3D renders, no illustrations, no clipart, no cartoon.\n"
        "- No text, letters, logos, watermarks, banners, or UI elements inside the image.\n"
        "- No AI-typical artifacts: no extra limbs, no distorted faces, no weird fingers.\n"
        "- Composition must be clean, single scene, portrait-safe for a vertical/square crop.\n"
        "- Output ONLY the prompt string, nothing else.\n\n"
        f"Channel topic: {topic or '-'}\n"
        f"User request: {prompt or '-'}\n"
        f"Post text: {post_text[:500]}\n"
    )
    try:
        out = await ai_chat(
            api_key=api_key,
            model=model,
            messages=[{"role": "user", "content": ask}],
            temperature=0.3,
            max_tokens=120,
            base_url=base_url,
        )
        out = _clean_text(out)
        if out and not out.startswith(("⚠️", "⏳", "🌐")):
            return out[:500]
    except (httpx.TimeoutException, httpx.ConnectError, OSError) as exc:
        logger.warning("Prompt builder network error (transient), using fallback: %s", exc)
    except Exception as exc:
        logger.error("Prompt builder failed (unexpected), using fallback: %s", exc, exc_info=True)
    return fallback


async def generate_ai_image(
    topic: str,
    prompt: str,
    post_text: str,
    *,
    api_key: str = "",
    model: str = "",
    base_url: str | None = None,
    prebuilt_prompt: str = "",
) -> str | None:
    hf_key = (os.getenv("HF_API_KEY") or "").strip()
    hf_model = (os.getenv("HF_IMAGE_MODEL") or DEFAULT_MODEL).strip()
    if not hf_key:
        logger.info("HF_API_KEY is not configured, AI image generation skipped")
        return None

    if prebuilt_prompt:
        image_prompt = _clean_text(prebuilt_prompt)[:600]
    else:
        image_prompt = await build_image_prompt(
            topic,
            prompt,
            post_text,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )

    url = f"https://router.huggingface.co/hf-inference/models/{hf_model}"
    headers = {"Authorization": f"Bearer {hf_key}", "Accept": "image/png"}
    payload = {
        "inputs": image_prompt,
        "parameters": {
            "negative_prompt": _NEGATIVE_HINT,
            "guidance_scale": _DEFAULT_GUIDANCE_SCALE,
            "num_inference_steps": _DEFAULT_INFERENCE_STEPS,
        },
        "options": {"wait_for_model": False, "use_cache": False},
    }

    timeout = httpx.Timeout(connect=10.0, read=45.0, write=15.0, pool=15.0)
    proxy = _proxy_url()
    transport = httpx.AsyncHTTPTransport(retries=0)  # manual retry for better error handling

    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False, follow_redirects=True, proxy=proxy, transport=transport) as client:
                r = await client.post(url, headers=headers, json=payload)
            ctype = (r.headers.get("content-type") or "").lower()

            if r.status_code == 429:
                logger.warning("HF image rate-limited (429) attempt=%d/%d", attempt, max_attempts)
                if attempt < max_attempts:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None

            if r.status_code == 503:
                logger.warning("HF image model loading (503) attempt=%d/%d body=%s", attempt, max_attempts, r.text[:200])
                if attempt < max_attempts:
                    await asyncio.sleep(3)
                    continue
                return None

            if r.status_code == 401:
                logger.error("HF image auth failed (401) — check HF_API_KEY")
                return None

            if r.status_code not in range(200, 300):
                logger.warning("HF image non-2xx code=%s attempt=%d body=%s", r.status_code, attempt, r.text[:300])
                if attempt < max_attempts:
                    continue
                return None

            if "image" not in ctype:
                logger.warning("HF image unexpected content-type=%s body=%s", ctype, r.text[:300])
                return None

            ext = ".png" if "png" in ctype else ".jpg"
            digest = hashlib.sha1(f"{topic}|{prompt}|{post_text[:300]}|{image_prompt}".encode("utf-8", "ignore")).hexdigest()
            out_path = GENERATED_DIR / f"{digest}{ext}"
            out_path.write_bytes(r.content)
            logger.info("AI image generated model=%s path=%s attempt=%d", hf_model, out_path, attempt)
            return str(out_path)

        except httpx.TimeoutException as exc:
            logger.warning("HF image timeout attempt=%d/%d err=%s", attempt, max_attempts, exc)
            if attempt < max_attempts:
                continue
        except (httpx.ConnectError, OSError) as exc:
            logger.warning("HF image network error attempt=%d/%d err=%s", attempt, max_attempts, exc)
            if attempt < max_attempts:
                continue
        except Exception as exc:
            logger.error("HF image generation unexpected error: %s", exc, exc_info=True)
            return None

    return None
