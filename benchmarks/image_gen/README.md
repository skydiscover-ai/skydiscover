# Image Generation Benchmark

This benchmark evaluates whether SkyDiscover can optimize images, not just code or text. Each "solution" in the population is an image, evolved by generating and scoring variants from a candidate pool stored in the database. The evolutionary loop is the same as for code — parent selection, mutation via LLM, crossover via other context images from other islands — but instead of evolving Python programs, SkyDiscover evolves text prompts fed to GPT-5's native image generation. The VLM receives actual parent and other context images alongside text guidance, reasons about what to improve, and generates a new image. Setting `language: "image"` in the config is the only change needed.

## Benchmark: Sky Festival

**Directory:** `sky_festival/`

The system must generate a floating sky-festival image where many details must match exact structural constraints: 9 clouds with specific shapes (rabbit, teacup, musical note, crescent moon, whale, etc.), 5 hot-air balloons with exact colors, passengers, and a banner reading "HAPPY 100TH SKY FESTIVAL", a floating island with 4 trees in a specific left-to-right order, and a party table with precisely counted items (6 cupcakes, 8 golden plates, 5 gift boxes in a pyramid). The scene also includes 6 characters with specific attributes (e.g., a robot with 3 colored buttons on its chest, a grandmother giving a thumbs-up with her left hand), flying creatures, and a correctly ordered 7-band rainbow. The full specification is about 2000 words and lives in `config.yaml`'s `prompt.system_message`.

**Evaluator.** Each generated image is graded by a GPT-5 vision judge using a strict rubric. The judge receives the image and a detailed scoring sheet, then returns per-category scores across 7 dimensions — cloud shapes (15 pts), balloons (20 pts), floating island (10 pts), table items (20 pts), characters (15 pts), decorations/creatures (10 pts), and rainbow/lighting (10 pts) — for a total of 100 points. The judge is instructed to be extremely harsh: points are awarded only when requirements are clearly and unambiguously met in the image.

## Setup

1. **Set your API key:**

   ```bash
   export OPENAI_API_KEY=...
   ```

   Both the image generator (GPT-5) and the evaluator judge (GPT-5) use the OpenAI API.

## Run

```bash
cd benchmarks/image_gen/sky_festival

# AdaEvolve
uv run skydiscover-run evaluator.py -c config.yaml -s adaevolve -o sky_festival_output

# EvoX
uv run skydiscover-run evaluator.py -c config.yaml -s evox -o sky_festival_output
```

## Files

| File | Description |
|------|-------------|
| `sky_festival/evaluator.py` | GPT-5 vision judge that scores images against the 100-point rubric |
| `sky_festival/config.yaml` | Config — scene specification in `prompt.system_message` |
