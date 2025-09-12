import json
import argparse
import logging
from typing import Dict, Any, List, Tuple, Optional
from deepdiff import DeepDiff


from tabulate import tabulate
import yaml
import csv
import os
import re
import subprocess
from groq import Groq
client = Groq(api_key="")



# Optional URL support (only used if you pass --url1/--url2)
try:
    import requests
except Exception:  # keep script usable without requests if you never use URLs
    requests = None


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---- Load models (UNCHANGED) ----
 # kept for future use; not required here



def load_local_file(path: str) -> Any:
    """
    Load JSON, YAML, CSV (as dict), or plain-text AB logs into a Python object.
    - .json -> JSON
    - .yaml/.yml -> YAML
    - .csv -> CSV (wrapped as {"rows": [...]})
    - .txt/.log -> parsed ApacheBench-ish text to dict (best-effort)
    """
    ext = os.path.splitext(path)[1].lower()

    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    if ext == ".json":
        return json.loads(data)

    if ext in [".yaml", ".yml"]:
        return yaml.safe_load(data)

    if ext == ".csv":
        # Parse CSV to dict of rows
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8", newline="") as fcsv:
            reader = csv.DictReader(fcsv)
            for row in reader:
                rows.append(row)
        return {"rows": rows}

    if ext in [".txt", ".log"]:
        return parse_ab_text(data)

    # Fallback: try JSON then YAML
    try:
        return json.loads(data)
    except Exception:
        return yaml.safe_load(data)


def parse_json_string(json_str: str) -> Any:
    """Parse a raw JSON string."""
    return json.loads(json_str)


def fetch_from_url(url: str) -> Any:
    """Fetch JSON/YAML from URL. Requires 'requests'."""
    if requests is None:
        raise RuntimeError("requests is not installed. Install with: pip install requests")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    # Try JSON first
    try:
        return resp.json()
    except Exception:
        # Then YAML
        try:
            return yaml.safe_load(resp.text)
        except Exception:
            # Finally, try parsing AB text
            return parse_ab_text(resp.text)


def parse_ab_text(content: str) -> Dict[str, Any]:
    """
    Best-effort parser for ApacheBench (ab) plain text output.
    Also works with other simple key: value perf logs.
    """
    data: Dict[str, Any] = {}
    # Key: Value lines
    for line in content.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            data[key.strip()] = val.strip()
        else:
            # try to capture 'Requests per second' like entries not in key:value
            m = re.match(r"(.+?)\s+([\d\.\-eE]+)$", line.strip())
            if m:
                k, v = m.groups()
                data[k.strip()] = v.strip()
    # If still empty, store raw for transparency
    if not data and content.strip():
        data["raw"] = content.strip()
    return data


def is_openapi(spec: Any) -> bool:
    """Detect if a dict looks like an OpenAPI/Swagger spec."""
    if not isinstance(spec, dict):
        return False
    if "openapi" in spec and "paths" in spec:
        return True
    if "swagger" in spec and "paths" in spec:  # Swagger 2.0
        return True
    return False


def _collect_endpoints(spec: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Return list of (method, path) pairs from an OpenAPI/Swagger spec.
    Methods are lowercase: get, post, put, delete, patch, head, options, trace
    """
    if not is_openapi(spec):
        return []
    endpoints: List[Tuple[str, str]] = []
    paths = spec.get("paths", {}) or {}
    for path, item in paths.items():
        if not isinstance(item, dict):
            continue
        for method in ["get", "post", "put", "delete", "patch", "head", "options", "trace"]:
            if method in item:
                endpoints.append((method, path))
    return sorted(endpoints)


def _collect_params(spec: Dict[str, Any], method: str, path: str) -> List[Dict[str, Any]]:
    """Collect parameters for a given path+method."""
    paths = spec.get("paths", {}) or {}
    op = (paths.get(path) or {}).get(method) or {}
    params = op.get("parameters", []) or []
    # Also consider path-level parameters
    path_level = (paths.get(path) or {}).get("parameters", []) or []
    return list(params) + list(path_level)


def _collect_responses(spec: Dict[str, Any], method: str, path: str) -> Dict[str, Any]:
    """Collect responses for a given path+method as a simple dict of status->content-types."""
    paths = spec.get("paths", {}) or {}
    op = (paths.get(path) or {}).get(method) or {}
    responses = op.get("responses", {}) or {}
    out: Dict[str, Any] = {}
    for status, rdef in responses.items():
        content = (rdef or {}).get("content", {}) or {}
        out[status] = sorted(list(content.keys())) if isinstance(content, dict) else []
    return out


def compare_openapi(spec1: Dict[str, Any], spec2: Dict[str, Any]) -> str:
    """
    Endpoint-level comparison:
    - Added / Removed endpoints
    - Parameter presence differences (by name)
    - Response status/content-type differences
    Returns a tabulated grid string.
    """
    rows: List[List[str]] = []

    e1 = set(_collect_endpoints(spec1))
    e2 = set(_collect_endpoints(spec2))

    added = sorted(list(e2 - e1))     # present only in spec2
    removed = sorted(list(e1 - e2))   # present only in spec1
    common = sorted(list(e1 & e2))

    for method, path in added:
        rows.append(["Added", method.upper(), path, "Present only in API 2"])

    for method, path in removed:
        rows.append(["Removed", method.upper(), path, "Present only in API 1"])

    for method, path in common:
        p1 = _collect_params(spec1, method, path)
        p2 = _collect_params(spec2, method, path)

        names1 = { (p.get("name"), p.get("in")) for p in p1 if isinstance(p, dict) }
        names2 = { (p.get("name"), p.get("in")) for p in p2 if isinstance(p, dict) }

        added_params = sorted(list(names2 - names1))
        removed_params = sorted(list(names1 - names2))

        if added_params:
            rows.append([
                "Param +",
                method.upper(),
                path,
                "Added params in API 2: " + ", ".join([f"{n}@{where}" for n, where in added_params])
            ])
        if removed_params:
            rows.append([
                "Param -",
                method.upper(),
                path,
                "Missing in API 2 (present in API 1): " + ", ".join([f"{n}@{where}" for n, where in removed_params])
            ])
        r1 = _collect_responses(spec1, method, path)
        r2 = _collect_responses(spec2, method, path)

        s1 = set(r1.keys())
        s2 = set(r2.keys())

        add_status = sorted(list(s2 - s1))
        rem_status = sorted(list(s1 - s2))

        for st in add_status:
            rows.append(["Resp +", method.upper(), path, f"Status {st} added in API 2"])
        for st in rem_status:
            rows.append(["Resp -", method.upper(), path, f"Status {st} removed in API 2"])

        for st in sorted(list(s1 & s2)):
            ct1 = set(r1.get(st, []))
            ct2 = set(r2.get(st, []))
            if ct1 != ct2:
                rows.append([
                    "Resp ~",
                    method.upper(),
                    path,
                    f"Status {st} content-types differ: API1={sorted(ct1)} vs API2={sorted(ct2)}"
                ])

    if not rows:
        return "No endpoint-level differences detected."

    return tabulate(rows, headers=["Type", "Method", "Path", "Detail"], tablefmt="grid")


def compare_json_structures(json1: Dict[str, Any], json2: Dict[str, Any]) -> str:
    """Compare JSON structures and return formatted differences."""
    diff = DeepDiff(json1, json2, ignore_order=True)
    if not diff:
        return "No structural differences found."

    formatted_diff = []
    for change_type, changes in diff.items():
        for key, details in (changes.items() if isinstance(changes, dict) else []):
            old_val = details.get("old_value", "N/A")
            new_val = details.get("new_value", "N/A")
            formatted_diff.append([
                change_type.replace("_", " ").title(),
                key,
                str(old_val),
                str(new_val)
            ])
    return tabulate(formatted_diff, headers=["Change Type", "Path", "API 1 Value", "API 2 Value"], tablefmt="grid")


def generate_dynamic_prompt(json1: Dict[str, Any], json2: Dict[str, Any]) -> str:
    """Generate a clear, concise LLM prompt for API comparison."""
    return f"""
You are an expert API analyst. Compare the following two API responses and identify all differences.
Focus on:
1. Structural differences (fields, nesting, data types).
2. Semantic differences (authentication, performance, backward compatibility, error handling, etc.).
**API 1 Response:**
{json.dumps(json1, indent=2)}

**API 2 Response:**
{json.dumps(json2, indent=2)}

**Output Requirements:**
- Use a clear, concise table format with the following columns:
  | Category | API 1 Field/Value | API 2 Field/Value | Difference Description |
- Group similar changes together (e.g., all field renames under "Field Change").
- Only include rows for actual differences. Omit "Same" or unchanged fields.
- If a field or endpoint is missing in one API, use "N/A" in the relevant column.
- For data type changes, note the old and new types (e.g., "int → string").
- For changes, explain the impact (like how having different authentication methods affects client implementation).
- **As the above point states the difference should be atleast one coherent line long**.
- Redundant rows with no differences must not be included.
- Do not include introductory or closing text. Only output the table.
"""

def generate_llm_summary(json1: Dict[str, Any], json2: Dict[str, Any]) -> str:
    """Generate a tabular LLM summary using Groq's API."""
    try:
        prompt = generate_dynamic_prompt(json1, json2)
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.65,
            max_tokens=1024,
        )
        output = response.choices[0].message.content.strip()
        # Convert markdown table to grid if present
        if "|" in output:
            lines = [line for line in output.split("\n") if "|" in line]
            if len(lines) > 1:
                headers = [h.strip() for h in lines[0].split("|")[1:-1]]
                rows = [[col.strip() for col in line.split("|")[1:-1]] for line in lines[2:]]
                return tabulate(rows, headers=headers, tablefmt="grid")
        return output if output else "No differences found."
    except Exception as e:
        logger.error(f"Groq API Error: {e}")
        return "Failed to generate summary."


# =========================================
# Unified Input Resolver
# =========================================
def resolve_inputs(args) -> Tuple[Any, Any, str]:
    """
    Resolve two inputs into Python objects based on provided arguments.
    Priority order keeps backward-compat with your original usage:
      1) --json1/--json2 (as FILE PATHS, original behavior)
      2) --file1/--file2 (generic files)
      3) --openapi1/--openapi2 (alias of files but named)
      4) --url1/--url2 (fetch via HTTP)
      5) --str1/--str2 (inline raw JSON strings)
    Returns (obj1, obj2, mode_description)
    """
    # 1) Original flags: --json1/--json2 as FILE paths (backward compatible)
    if args.json1 and args.json2:
        obj1 = load_local_file(args.json1)
        obj2 = load_local_file(args.json2)
        return obj1, obj2, "json-file"

    # 2) Generic file paths
    if args.file1 and args.file2:
        obj1 = load_local_file(args.file1)
        obj2 = load_local_file(args.file2)
        return obj1, obj2, "file"

    # 3) OpenAPI-named file flags
    if args.openapi1 and args.openapi2:
        obj1 = load_local_file(args.openapi1)
        obj2 = load_local_file(args.openapi2)
        return obj1, obj2, "openapi-file"

    # 4) URLs
    if args.url1 and args.url2:
        obj1 = fetch_from_url(args.url1)
        obj2 = fetch_from_url(args.url2)
        return obj1, obj2, "url"

    # 5) Raw JSON strings
    if args.str1 and args.str2:
        obj1 = parse_json_string(args.str1)
        obj2 = parse_json_string(args.str2)
        return obj1, obj2, "json-string"

    raise ValueError(
        "You must provide a pair of inputs. Try one of:\n"
        "  --json1 file.json --json2 file.json (original mode)\n"
        "  --file1 file --file2 file\n"
        "  --openapi1 api.yaml --openapi2 api.yaml\n"
        "  --url1 https://... --url2 https://...\n"
        "  --str1 '{...}' --str2 '{...}'"
    )


# =========================================
# Main
# =========================================
def main():
    parser = argparse.ArgumentParser(description="Universal API Comparison Tool (JSON/YAML/OpenAPI/Swagger/AB/URL)")

    # ---- Original flags (backward compatible) ----
    parser.add_argument('--json1', help='Path to API 1 JSON file (original flag, kept for compatibility)')
    parser.add_argument('--json2', help='Path to API 2 JSON file (original flag, kept for compatibility)')

    # ---- New flexible flags ----
    parser.add_argument('--file1', help='Path to first file (JSON/YAML/CSV/TXT)')
    parser.add_argument('--file2', help='Path to second file (JSON/YAML/CSV/TXT)')

    parser.add_argument('--openapi1', help='Path to first OpenAPI/Swagger file (YAML/JSON)')
    parser.add_argument('--openapi2', help='Path to second OpenAPI/Swagger file (YAML/JSON)')

    parser.add_argument('--url1', help='URL to first JSON/YAML spec or AB text')
    parser.add_argument('--url2', help='URL to second JSON/YAML spec or AB text')

    parser.add_argument('--str1', help='Raw JSON string for input 1')
    parser.add_argument('--str2', help='Raw JSON string for input 2')

    args = parser.parse_args()

    try:
        obj1, obj2, mode = resolve_inputs(args)
        logger.info(f"Loaded inputs via mode: {mode}")
    except Exception as e:
        logger.error(f"Input error: {e}")
        return

    # Print OpenAPI endpoint comparison if both are OpenAPI/Swagger
    if is_openapi(obj1) and is_openapi(obj2):
        print("\n OpenAPI Endpoint-Level Differences:")
        print(compare_openapi(obj1, obj2))

    # Structural diff (works for dict-like inputs)
    if isinstance(obj1, (dict, list)) and isinstance(obj2, (dict, list)):
        print("\n JSON Structural Differences:")
        # DeepDiff prefers dicts; if lists, wrap for comparison transparency
        j1 = obj1 if isinstance(obj1, dict) else {"_root_list": obj1}
        j2 = obj2 if isinstance(obj2, dict) else {"_root_list": obj2}
        print(compare_json_structures(j1, j2))
    else:
        print("\n Structural Differences:")
        print("Inputs are not structured objects. Showing types only:")
        print(f" - Input 1 type: {type(obj1).__name__}")
        print(f" - Input 2 type: {type(obj2).__name__}")

    # LLM summary with your original prompt
    if isinstance(obj1, (dict, list)) and isinstance(obj2, (dict, list)):
        # To keep the prompt semantics (expects dict-like), wrap lists if needed
        j1 = obj1 if isinstance(obj1, dict) else {"_root_list": obj1}
        j2 = obj2 if isinstance(obj2, dict) else {"_root_list": obj2}
        print("\n Dynamic LLM Comparison Summary:")
        print(generate_llm_summary(j1, j2))
    else:
        print("\n Dynamic LLM Comparison Summary:")
        print("Skipped (inputs are not JSON/YAML-like structures).")

    # Close LLM cleanly


if __name__ == '__main__':
    main()
