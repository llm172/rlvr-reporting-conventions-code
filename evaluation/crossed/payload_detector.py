"""Reference-blind reporting-payload audit for saved GSM8K completions.

Public API: detect(response) -> dict.

Use h_payload and b_payload to form requested-only / other-only / both /
neither. These report syntactic payload presence, not semantic compliance,
final commitment, correctness, or agreement with another marker.

Hash: an exact run of four hashes, followed on the SAME LINE by a payload
starting with an ASCII number (optional +, -, or Unicode minus). Leading
currency/math $ delimiters, **, backticks, and LaTeX math opening delimiters
are allowed. The payload ends at line end or the next detected convention
marker. Ordinary quoted-marker instruction mentions and prose headings do
not qualify. A numeric prefix suffices: units/fractions/multipart tails are
retained without deciding their value.

Boxed: a literal \\boxed command with an opening brace (optional whitespace),
a matched closing brace, and nonempty content. Braces may nest; escaped
braces do not affect balancing. No numeric or correctness requirement is
imposed. Empty template boxes and unclosed boxes do not qualify.

All occurrences are scanned; one qualifying occurrence suffices. Both
formats may qualify in one response. Unlike the frozen first-occurrence
evaluation readers, this module does not return an answer value.

raw_h_nonempty and raw_b_nonempty preserve the corresponding permissive
nonempty flags before the numeric-start restriction on hashes. "Raw" still
means same-line hash or balanced box, not arbitrary text after a marker.
Suspect candidates record nonnumeric hash continuations, allowed wrapped
hash numbers, empty/unbalanced markers, nonnumeric boxes, and explicit
instruction-cue contexts. They are diagnostic records; no adjudication is
silently applied and no references are accepted by this API.

CLI (no raw-input edits):
  python payload_detector.py --inputs inputs --output payload_suspects.json
This writes a bounded-context candidate audit plus syntactic counts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

DETECTOR_VERSION = "gsm8k-payload-v1"
HASH_MARKER = re.compile(r"(?<!#)####(?!#)")
BOX_MARKER = re.compile(r"\\boxed\s*\{")
NUMERIC_START = re.compile(r"[+\-\u2212]?(?:[0-9]|\.[0-9])")
OPENERS = (r"\(", r"\[", r"\$", "**", "`", "$")
INSTRUCTION_CUE = re.compile(
    r"(?:output|report|write|put|place|give|provide|present|show)\s+"
    r"(?:the\s+)?(?:final\s+)?answer\s+(?:after|within|inside)\s*[\"'`]*(?:\s*)$",
    re.IGNORECASE,
)


def _escaped(text: str, pos: int) -> bool:
    slashes = 0
    pos -= 1
    while pos >= 0 and text[pos] == "\\":
        slashes += 1
        pos -= 1
    return bool(slashes % 2)


def _strip_openers(payload: str) -> str:
    value = payload.lstrip()
    while value:
        for opener in OPENERS:
            if value.startswith(opener):
                value = value[len(opener):].lstrip()
                break
        else:
            return value
    return value


def _nonempty(payload: str) -> bool:
    # Wrapper-only punctuation is not an answer payload. A digit, letter,
    # or any non-ASCII mathematical symbol still qualifies as raw content.
    value = payload.strip()
    for token in (r"\(", r"\)", r"\[", r"\]", r"\$", r"\{", r"\}"):
        value = value.replace(token, "")
    return bool(value.strip(" \t\r\n$`*_{}[]()\"'.,:;!?"))


def _box_payload(text: str, start: int) -> tuple[str, int, bool]:
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{" and not _escaped(text, i):
            depth += 1
        elif text[i] == "}" and not _escaped(text, i):
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1, True
    return text[start:], len(text), False


def detect(response: str) -> dict:
    """Return reference-blind per-response flags and audit candidates."""
    if not isinstance(response, str):
        raise TypeError("response must be a completion string")
    hashes = list(HASH_MARKER.finditer(response))
    boxes = list(BOX_MARKER.finditer(response))
    starts = sorted(m.start() for m in hashes + boxes)
    result = {
        "detector_version": DETECTOR_VERSION,
        "h_marker": bool(hashes), "b_marker": bool(boxes),
        "raw_h_nonempty": False, "raw_b_nonempty": False,
        "h_payload": False, "b_payload": False,
        "h_occurrences": len(hashes), "b_occurrences": len(boxes),
        "candidates": [],
    }

    def candidate(kind, start, end, payload, reasons, accepted):
        result["candidates"].append({
            "convention": kind, "marker_start": start, "span_end": end,
            "payload": payload[:240], "payload_truncated": len(payload) > 240,
            "reasons": reasons, "accepted_payload": accepted,
            "context": response[max(0, start - 110):min(len(response), end + 110)][:500],
        })

    for marker in hashes:
        end = response.find("\n", marker.end())
        if end < 0:
            end = len(response)
        following = [pos for pos in starts if marker.end() <= pos < end]
        if following:
            end = min(following)
        payload = response[marker.end():end].strip()
        raw = _nonempty(payload)
        normalized = _strip_openers(payload)
        accepted = bool(NUMERIC_START.match(normalized))
        result["raw_h_nonempty"] |= raw
        result["h_payload"] |= accepted
        reasons = []
        if not raw:
            reasons.append("empty_hash_payload")
        elif not accepted:
            reasons.append("nonnumeric_hash_continuation")
        elif normalized != payload:
            reasons.append("wrapped_numeric_hash_payload")
        if INSTRUCTION_CUE.search(response[max(0,marker.start()-160):marker.start()]):
            reasons.append("instruction_cue_before_marker")
        if reasons:
            candidate("hash",marker.start(),end,payload,reasons,accepted)

    for marker in boxes:
        payload, end, closed = _box_payload(response, marker.end())
        nonempty = closed and _nonempty(payload)
        result["raw_b_nonempty"] |= nonempty
        result["b_payload"] |= nonempty
        reasons = []
        if not closed:
            reasons.append("unbalanced_box")
        elif not nonempty:
            reasons.append("empty_box_payload")
        elif not re.search(r"[0-9]", payload):
            reasons.append("box_payload_without_digits")
        if INSTRUCTION_CUE.search(response[max(0,marker.start()-160):marker.start()]):
            reasons.append("instruction_cue_before_marker")
        if reasons:
            candidate("boxed",marker.start(),end,payload,reasons,nonempty)
    result["hash_numeric_payload"] = result["h_payload"]
    result["boxed_payload"] = result["b_payload"]
    return result


def requested_category(flags: dict, request: str) -> str:
    """Map the two payload booleans to an exhaustive, exclusive category."""
    if request not in {"hash", "boxed"}:
        raise ValueError("request must be 'hash' or 'boxed'")
    requested = flags["h_payload" if request == "hash" else "b_payload"]
    other = flags["b_payload" if request == "hash" else "h_payload"]
    if requested and other:
        return "both"
    if requested:
        return "requested_only"
    if other:
        return "other_only"
    return "neither"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells, candidates = [], []
    for path in sorted(args.inputs.glob("*/*/*/responses.jsonl")):
        seed, trained, request, _ = path.relative_to(args.inputs).parts
        counts, reasons = Counter(), Counter()
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            for sample, response in enumerate(row["responses"]):
                flags = detect(response)
                counts["responses"] += 1
                for key in ("h_marker", "b_marker", "raw_h_nonempty", "raw_b_nonempty",
                            "h_payload", "b_payload"):
                    counts[key] += int(flags[key])
                counts[requested_category(flags,request)] += 1
                if flags["candidates"]:
                    counts["responses_with_candidates"] += 1
                    for c in flags["candidates"]:
                        reasons.update(c["reasons"])
                    candidates.append({"seed":seed,"trained":trained,"request":request,
                                       "item_id":row["id"],"sample_index":sample,
                                       "candidates":flags["candidates"]})
        assert sum(counts[k] for k in ("requested_only","other_only","both","neither")) == counts["responses"]
        cells.append({"seed":seed,"trained":trained,"request":request,
                      "counts":dict(counts),"candidate_reasons":dict(reasons)})
    if not cells:
        raise SystemExit("No response files found under the requested input directory")
    args.output.write_text(json.dumps({"detector_version":DETECTOR_VERSION,
                                      "cells":cells,"candidates":candidates},
                                     ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"cells":len(cells),"responses":sum(c["counts"]["responses"] for c in cells),
                      "responses_with_candidates":len(candidates),"output":str(args.output)}))


def self_test() -> None:
    """Boundary checks for the operational rules, independent of any labels."""
    fixtures = [
        ("", False, False),
        ("####", False, False),
        ('Output the final answer after "####".', False, False),
        ("#### For Steve:\nThe distance is 3 miles.", False, False),
        ("#### $2400", True, False),
        ("#### **`-24`**", True, False),
        (r"The answer is \boxed{\frac{1}{2}}.", False, True),
        (r"Output the answer within \boxed{}.", False, False),
        (r"\boxed{754", False, False),
        ("##### 12", False, False),
        ("####\n12", False, False),
        ("#### 4 blue and 6 red", True, False),
        ("#### 12\n" + r"\boxed{12}", True, True),
    ]
    for response, expected_h, expected_b in fixtures:
        flags = detect(response)
        assert (flags["h_payload"], flags["b_payload"]) == (expected_h, expected_b), repr(response)
    assert detect("#### For Steve:")["raw_h_nonempty"]
    assert requested_category(detect("#### 12\n" + r"\boxed{12}"), "boxed") == "both"


if __name__ == "__main__":
    self_test()
    main()
