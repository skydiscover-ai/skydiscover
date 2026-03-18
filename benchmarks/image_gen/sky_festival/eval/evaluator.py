"""
Sky Festival evaluator — GPT-5 LLM-as-a-judge.

Scores VLM-generated images against a 100-point rubric using GPT-5 vision.
Returns combined_score normalized to [0, 1].

The framework passes the image path via a sidecar file:
    <program_path>.image_path  ->  absolute path to the generated image

Requirements:
    pip install openai
    Environment: OPENAI_API_KEY (required), JUDGE_MODEL (optional, default gpt-5)
"""

import base64
import json
import logging
import os
import re
from typing import Dict, Union

logger = logging.getLogger(__name__)

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5")

SYSTEM_PROMPT = """\
You are an extremely strict image evaluation judge. You score images against a precise rubric.
You must output ONLY valid JSON with the exact keys specified. No markdown, no explanation outside JSON.
Be harsh — most AI-generated images fail these criteria. Award points only when clearly met.
If you cannot verify a requirement (e.g., too small to see), award 0 for that item."""

RUBRIC_PROMPT = """\
Score this image against the following rubric for a "Floating Sky Festival" scene.
Be extremely strict. Only award points when requirements are CLEARLY and UNAMBIGUOUSLY met.

## Category 1: Cloud Counting and Shapes (15 pts)
- Exactly 9 clouds visible in the sky: 5 pts (8 or 10 clouds = 0)
- At least 5 of the 9 clouds have recognizable distinct shapes (rabbit, teacup, musical note, crescent moon, whale, bicycle, crown, butterfly, number 7): 10 pts (2 pts per recognizable shape, max 10)

## Category 2: Hot Air Balloons — Count, Colors, and Passengers (20 pts)
- Exactly 5 hot air balloons visible: 4 pts (4 or 6 = 0)
- Each balloon has correct distinct color/pattern (red-striped, yellow-dotted, rainbow, purple-stars, green-peace-sign): 6 pts (deduct 2 per wrong/missing pattern)
- Correct passenger count per balloon (2 children, 1 woman, 3 cats, 1 violinist, empty): 6 pts (deduct 2 per wrong count)
- Banner on Balloon 5 reads exactly "HAPPY 100TH SKY FESTIVAL": 4 pts (any word wrong = 0)

## Category 3: Floating Island and Trees (10 pts)
- Floating island visible suspended in air: 3 pts
- Exactly 4 different trees on the island: 4 pts (3 or 5 = 0)
- Trees in correct order left to right (oak, cherry blossom, palm, pine): 3 pts

## Category 4: Party Table Items — Counting and Arrangement (20 pts)
- 3-tier cake with candle present: 3 pts
- Cake text "100 YEARS" legible on middle tier: 3 pts
- Exactly 6 cupcakes in 2 rows of 3 with different colored frostings: 4 pts
- Lemonade pitcher with 3 lemon slices and 2 ice cubes: 3 pts
- Stack of exactly 8 golden plates: 3 pts
- Exactly 5 gift boxes in pyramid (3 bottom, 2 top): 4 pts

## Category 5: Characters — Count, Identity, and Details (15 pts)
- Exactly 6 characters seated at the table (3 per side): 5 pts
- Correct characters identifiable (girl with pigtails, penguin with bowtie, giraffe, robot, grandmother, golden retriever): 5 pts (1 pt per correct character, max 5 — giraffe counts as 1 even if neck extends)
- Specific details: robot has 3 colored buttons on chest, grandmother thumbs-up with LEFT hand, dog wears striped party hat, girl has 5 fingers per hand: 5 pts (deduct 1.5 per missing detail)

## Category 6: Decorations and Flying Creatures (10 pts)
- Bunting banner with approximately 11 flags in alternating red/yellow/blue: 3 pts
- Exactly 7 paper lanterns in different colors: 3 pts
- Correct flying creatures: 4 birds (blue jay, cardinal, canary, hummingbird) + 2 butterflies (monarch, morpho): 4 pts (1 pt per 2 correct creatures)

## Category 7: Rainbow, Lighting, and Overall Composition (10 pts)
- Complete semicircular rainbow with 7 color bands in correct order: 4 pts
- Consistent warm golden lighting from upper left with shadows falling lower right: 3 pts
- Overall magical/celebratory mood, scene is joyful and cohesive: 3 pts

Respond with ONLY this JSON (no other text):
{
  "cloud_shapes": <0-15>,
  "balloons": <0-20>,
  "floating_island": <0-10>,
  "table_items": <0-20>,
  "characters": <0-15>,
  "decorations_creatures": <0-10>,
  "rainbow_lighting": <0-10>,
  "reasoning": "<brief 2-3 sentence explanation>"
}"""

# Category maximum scores for validation
CATEGORY_MAXES = {
    "cloud_shapes": 15,
    "balloons": 20,
    "floating_island": 10,
    "table_items": 20,
    "characters": 15,
    "decorations_creatures": 10,
    "rainbow_lighting": 10,
}

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _judge_image(image_path: str) -> Dict[str, Union[float, str]]:
    """Call GPT-5 to score the image. Retries once on failure."""
    client = _get_client()
    b64 = _encode_image(image_path)

    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png")
    data_url = f"data:{mime};base64,{b64}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                {"type": "text", "text": RUBRIC_PROMPT},
            ],
        },
    ]

    last_error = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=messages,
                max_completion_tokens=16384,
            )
            content = response.choices[0].message.content or ""
            raw = content.strip()
            logger.info(f"Judge raw response (first 300 chars): {raw[:300]}")

            # Extract JSON from markdown code block if present
            if "```" in raw:
                m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
                if m:
                    raw = m.group(1).strip()

            # Find JSON object in response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

            result = json.loads(raw)

            # Validate and clamp scores
            scores = {}
            for cat, max_val in CATEGORY_MAXES.items():
                val = result.get(cat, 0)
                if not isinstance(val, (int, float)):
                    val = 0
                scores[cat] = max(0, min(max_val, float(val)))

            scores["reasoning"] = str(result.get("reasoning", ""))
            return scores

        except Exception as e:
            last_error = e
            logger.warning(f"Judge attempt {attempt + 1} failed: {e}")

    logger.error(f"GPT-5 judge failed after retries: {last_error}")
    return {cat: 0.0 for cat in CATEGORY_MAXES}


def evaluate(program_path: str) -> Dict[str, Union[float, str]]:
    """Score a VLM-generated image using GPT-5 as judge.

    Args:
        program_path: Path to the text file (VLM reasoning).
            A sidecar file ``<program_path>.image_path`` contains the
            absolute path to the generated image.

    Returns:
        Dictionary with combined_score (0-1), per-category scores, and image_path.
    """
    # Read image path from sidecar
    sidecar = program_path + ".image_path"
    image_path = None
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            image_path = f.read().strip()

    if not image_path or not os.path.exists(image_path):
        logger.warning("No image found for scoring")
        return {"combined_score": 0.0, "error": "No image to score"}

    # Score with GPT-5
    scores = _judge_image(image_path)

    # Compute total out of 100, normalize to 0-1
    total = sum(v for k, v in scores.items() if k in CATEGORY_MAXES)
    combined = round(total / 100.0, 4)

    result = {"combined_score": combined, "image_path": image_path}

    # Add per-category scores (normalized to 0-1 for each category)
    for cat, max_val in CATEGORY_MAXES.items():
        result[cat] = round(scores.get(cat, 0) / max_val, 4)

    # Also store raw scores
    result["raw_total"] = round(total, 1)

    reasoning = scores.get("reasoning", "")
    if reasoning:
        result["judge_reasoning"] = reasoning

    return result


if __name__ == "__main__":
    from wrapper import run
    run(evaluate)
