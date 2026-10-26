"""
VeriRAG Claim Decomposer
========================
Decomposes complex legal text or generated responses into indivisible, atomic,
and self-contained factual/legal propositions according to definitions.md.

Supports:
1. LLM-prompted decomposition (Claude API / Anthropic SDK)
2. Deterministic rule-based baseline (regex + sentence/clause boundaries)
3. Fallback execution if API key is not configured
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from dotenv import load_dotenv

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "decomposer" / "prompts" / "decomposition_prompt.txt"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def decompose_rule_based(text: str) -> List[Dict[str, Any]]:
    """
    Deterministic rule-based baseline decomposition.
    Splits at sentence terminators and major coordinating conjunctions/semicolons.
    """
    if not text or not text.strip():
        return []

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Split into sentences
    sentence_delimiters = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
    raw_sentences = [s.strip() for s in sentence_delimiters.split(text) if s.strip()]

    claims = []
    claim_idx = 1

    # Conjunction boundary splitter for compound sentences
    subclause_delimiters = re.compile(r';|\b(?:and|but|whereas|furthermore|moreover)\b', re.IGNORECASE)

    for sent in raw_sentences:
        parts = [p.strip() for p in subclause_delimiters.split(sent) if p.strip()]
        if not parts:
            parts = [sent]

        for p in parts:
            # Capitalize first letter and ensure ending period
            sub_claim = p[0].upper() + p[1:] if len(p) > 1 else p.upper()
            if not sub_claim.endswith((".", "!", "?")):
                sub_claim += "."

            # Filter trivial fragments
            if len(sub_claim.split()) >= 3:
                claims.append({
                    "claim_index": claim_idx,
                    "claim_text": sub_claim,
                    "source_sentence": sent,
                    "method": "rule_based"
                })
                claim_idx += 1

    return claims


def decompose_llm(
    text: str,
    prompt_template_path: Path = DEFAULT_PROMPT_PATH,
    model_name: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    LLM-prompted atomic claim decomposition via Anthropic API.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package is required for LLM decomposition. Run pip install anthropic.")

    client = anthropic.Anthropic(api_key=api_key)

    with open(prompt_template_path, "r", encoding="utf-8") as f:
        template = f.read()

    prompt = template.replace("{input_text}", text)

    response = client.messages.create(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )

    content_text = response.content[0].text.strip()

    # Parse JSON output from LLM
    try:
        # Strip potential markdown fences
        clean_json_str = re.sub(r'^```(?:json)?\s*', '', content_text, flags=re.MULTILINE)
        clean_json_str = re.sub(r'```\s*$', '', clean_json_str, flags=re.MULTILINE).strip()
        parsed_claims = json.loads(clean_json_str)
        if isinstance(parsed_claims, list):
            return [
                {
                    "claim_index": i + 1,
                    "claim_text": str(c).strip(),
                    "source_sentence": text,
                    "method": "llm_prompted"
                }
                for i, c in enumerate(parsed_claims) if str(c).strip()
            ]
    except Exception as e:
        print(f"Warning: Failed to parse LLM JSON response: {e}. Raw content:\n{content_text}")

    # Fallback to rule-based if JSON extraction failed
    return decompose_rule_based(text)


def decompose(
    text: str,
    method: str = "auto",
    config_path: Path = DEFAULT_CONFIG_PATH
) -> List[Dict[str, Any]]:
    """
    Unified decomposition dispatcher.
    method: 'auto' | 'llm' | 'rule_based'
    """
    cfg = load_config(config_path)
    decomp_cfg = cfg.get("decomposer", {})

    if method == "auto":
        preferred_method = decomp_cfg.get("method", "llm-prompted")
        has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if preferred_method == "llm-prompted" and has_api_key:
            method = "llm"
        else:
            method = "rule_based"

    if method == "llm":
        prompt_path = PROJECT_ROOT / decomp_cfg.get("prompt_template", "decomposer/prompts/decomposition_prompt.txt")
        return decompose_llm(
            text=text,
            prompt_template_path=prompt_path,
            model_name=decomp_cfg.get("llm_model", "claude-3-5-sonnet-20241022"),
        )
    else:
        return decompose_rule_based(text)


def main():
    parser = argparse.ArgumentParser(description="Decompose legal text into atomic claims.")
    parser.add_argument("text", type=str, nargs="?", default=None, help="Input legal text to decompose")
    parser.add_argument("--method", choices=["auto", "llm", "rule_based"], default="auto")
    args = parser.parse_args()

    sample_text = (
        args.text or
        "The appellant, convicted under Section 302 of the Indian Penal Code by the Sessions Court, "
        "was sentenced to life imprisonment, but was subsequently acquitted by the High Court due to "
        "material discrepancies in witness testimony."
    )

    print("=" * 80)
    print("VERIRAG CLAIM DECOMPOSER")
    print(f"Method: {args.method}")
    print(f"Input Text:\n\"{sample_text}\"")
    print("=" * 80)

    claims = decompose(sample_text, method=args.method)
    print(f"\nExtracted {len(claims)} Atomic Claims:")
    for c in claims:
        print(f"  [{c['claim_index']}] ({c['method']}) {c['claim_text']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
