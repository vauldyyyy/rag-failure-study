"""Which generator models does this key actually have quota for, and how much?

The free tier caps generation per DAY *per model*
(`GenerateRequestsPerDayPerProjectPerModel-FreeTier`). The first real run
discovered gemini-2.5-flash allows only 20/day, which cannot support a study
needing thousands of calls. Model IDs differ wildly in their allowance, and the
published numbers go stale, so probe rather than trust a table.

    python probe_models.py            # probe the usual candidates
    python probe_models.py --all      # probe every model the key can list
"""

import argparse
import re
import warnings

warnings.filterwarnings("ignore")

import google.generativeai as genai

import config

CANDIDATES = [
    "models/gemini-3.5-flash-lite",
    "models/gemini-3.1-flash-lite",
    "models/gemini-flash-lite-latest",
    "models/gemini-2.0-flash-lite",
    "models/gemini-3.6-flash",
    "models/gemini-3.5-flash",
    "models/gemini-flash-latest",
    "models/gemini-2.5-flash",
]


def probe(name):
    try:
        r = genai.GenerativeModel(name).generate_content("Reply with one word: ping")
        return "OK", (r.text or "").strip()[:20], None
    except Exception as e:
        s = str(e)
        limit = re.search(r"limit: (\d+)", s) or re.search(r"quota_value: (\d+)", s)
        if "429" in s:
            return "QUOTA", "", limit.group(1) if limit else "?"
        if "404" in s:
            return "N/A", "not available to this key", None
        return "ERR", s[:60], None


def main():
    p = argparse.ArgumentParser(description="Probe model availability and quota")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    genai.configure(api_key=config.GEMINI_API_KEY)

    names = CANDIDATES
    if args.all:
        names = [m.name for m in genai.list_models()
                 if "generateContent" in m.supported_generation_methods
                 and not any(k in m.name for k in ("vision", "tts", "image", "lyria",
                                                   "banana", "embedding"))]

    print(f"{'status':<8}{'model':<42}{'daily limit'}")
    print("-" * 68)
    usable = []
    for name in names:
        status, detail, limit = probe(name)
        note = f"{limit}/day (exhausted)" if status == "QUOTA" else detail
        print(f"{status:<8}{name:<42}{note}")
        if status == "OK":
            usable.append(name)

    print()
    if usable:
        print("Usable now:")
        for u in usable:
            print(f"  {u}")
        print(f"\nSet it for a run:  --model {usable[0]}")
        print("or put LLM_MODEL in .env to make it the default.")
    else:
        print("No model has quota right now. Free-tier daily caps reset at "
              "midnight Pacific. Checkpoints are safe; resume then.")


if __name__ == "__main__":
    main()
