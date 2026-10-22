#!/usr/bin/env python3 -I
"""Makes Notebooks/data/gemma-en-ja-200.jsonl: 200 English segments from WMT24++ (en-ja_JP), each translated into Japanese by
Gemma 4 26B-A4B on OpenRouter (full precision, temperature 0, no thinking), with the system prompt in translate-system-prompt-ja.txt.

    OR_KEY=... python3 -I make_gemma_en_ja.py [--dry-run]

The key is read from the environment variable OR_KEY and is never printed or written. Stops at the first failed call (no retries).
The segment choice is the notebook's own filter, so Notebooks/Shears.ipynb can leave exactly these segments out of its training rows.
"""
import urllib.error, collections, datetime, hashlib, json, os, pathlib, random, re, sys, time, urllib.request
from huggingface_hub import hf_hub_download

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "gemma-en-ja-200.jsonl"
SYSTEM = (HERE / "translate-system-prompt-ja.txt").read_text(encoding="utf-8")
MODEL = "google/gemma-4-26b-a4b-it"
WMT_REPO, WMT_FILE, WMT_REVISION = "google/wmt24pp", "en-ja_JP.jsonl", "fd7405c06494bc66a57b25f55d217a72f96e60dc"
QUOTA = {"speech": 5, "news": 16, "literary": 58, "social": 121}     # 200 in all; speech and news have few usable segments
SEED = 11

AVOID = re.compile(r"\b(fuck\w*|shit\w*|damn\w*|bitch\w*|ass|asshole|bastard|crap|piss\w*|sex\w*|porn\w*|nude|naked|rape\w*|kill\w*|murder\w*|suicid\w*|die[ds]?|dead|death|dying|gun\w*|shoot\w*|shot|bomb\w*|terror\w*|drug\w*|weed|cocaine|nazi\w*|hitler|racis\w*|slave\w*|stupid|idiot\w*|hate\w*|abus\w*|drunk|alcohol\w*|cancer|hospital\w*|pills?|religio\w*|god|jesus|christ\w*|muslim\w*|islam\w*|jew\w*|trump|biden|republican\w*|democrat\w*|politic\w*|president\w*|elect\w*|vote[sd]?|war|army|soldier\w*|blood\w*|crim\w*|police|arrest\w*|prison\w*|jail\w*|fat|ugly|dumb|crazy|breast\w*|penis|vagina|cheat\w*|divorc\w*|pregnan\w*|abort\w*|gay|lesbian|trans|black|white|asian|immigra\w*|ukrain\w*|russia\w*|israel\w*|palestin\w*|china|chinese|japan\w*|korea\w*|lol|wtf|omg|hell|bloody|sucks?|screw\w*|dick|cock|boob\w*|hooker|whore|tit|tits|nsfw|vomit\w*|poop|diaper|toilet|diarrhea|disease\w*|virus|covid|vaccin\w*|racial|nigg\w*|retard\w*|russif\w*|sunak|snp|nhs|conservative|liberal|labou?r|tory|tories|government\w*|minister\w*|activis\w*|incarcerat\w*|convict\w*|tax\w*|protest\w*|union\w*|strike\w*|scotland|scottish|predator\w*|bigot\w*|violen\w*|beer|elon|musk|tesla|abandon\w*|possess\w*|pistol|rifle|weapon\w*|open fire|hostage\w*|stole|stolen|steal\w*|eats|spat|poking|drill\w*|extraction)\b", re.I)

def usable(r, max_words=40):
    s, t = r["source"], r["target"]
    if r["domain"] == "canary" or r["is_bad_source"] or "CANARY" in s or not t.strip():
        return False
    if not (4 <= len(s.split()) <= max_words) or "\n" in s:
        return False
    if re.search(r"https?://|www\.|[@#<>\[\]{}|\\]|&\w+;", s):
        return False
    letters = [c for c in s if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) < 0.95:
        return False
    return not AVOID.search(s)

def choose():
    path = hf_hub_download(WMT_REPO, WMT_FILE, repo_type="dataset", revision=WMT_REVISION)
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    by = collections.defaultdict(list)
    for r in sorted((r for r in rows if usable(r)), key=lambda r: (r["document_id"], r["segment_id"])):
        by[r["domain"]].append(r)
    rng = random.Random(SEED)
    picked = {}
    for d, n in QUOTA.items():
        rng.shuffle(by[d])
        picked[d] = by[d][:n]
        assert len(picked[d]) == n, (d, len(by[d]), n)
    # the first rows are the ones the notebook shows side by side: take them round-robin over the domains
    order, rest = [], {d: list(v) for d, v in picked.items()}
    while any(rest.values()):
        for d in ("literary", "news", "social", "speech"):
            if rest[d]:
                order.append(rest[d].pop(0))
    return order[:12] + random.Random(SEED + 1).sample(order[12:], len(order) - 12)

def call(key, english, provider=None):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 700, "reasoning": {"enabled": False},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": english}]}
    if provider:
        body["provider"] = {"order": [provider], "allow_fallbacks": False}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": "shears-notebook-data"})
    try:
        import certifi, ssl
        ctx = ssl.create_default_context(cafile=certifi.where())     # python.org builds on a Mac ship no CA bundle
    except ImportError:
        ctx = None
    try:
        with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"OpenRouter answered HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")

if __name__ == "__main__":
    segs = choose()
    print(len(segs), "segments:", dict(collections.Counter(r["domain"] for r in segs)))
    if "--dry-run" in sys.argv:
        sys.exit(0)
    key = os.environ.get("OR_KEY", "").strip()
    if not key:
        sys.exit("set OR_KEY to an OpenRouter API key: stopping")
    want = os.environ.get("OR_KEY_SHA_PREFIX")        # optional: refuse to run with any other key (a guard against using the wrong one)
    if want and not hashlib.sha256(key.encode()).hexdigest().startswith(want):
        sys.exit("OR_KEY is not the key whose sha256 starts with OR_KEY_SHA_PREFIX: stopping")
    out, provider, cost = [], None, 0.0
    for i, r in enumerate(segs):
        resp = call(key, r["source"], provider)
        ch = resp["choices"][0]
        text = (ch["message"].get("content") or "")
        if ch.get("finish_reason") != "stop" or not text.strip() or ch["message"].get("reasoning"):
            sys.exit(f"segment {i}: unusable reply (finish_reason={ch.get('finish_reason')!r}, empty={not text.strip()}, reasoning={bool(ch['message'].get('reasoning'))}): stopping")
        if provider is None:
            provider = resp.get("provider")
            print("served by:", provider, "| model:", resp.get("model"))
            if not provider:
                sys.exit("the reply does not say which provider served it: stopping")
        elif resp.get("provider") != provider:
            sys.exit(f"segment {i}: served by {resp.get('provider')!r}, not {provider!r}: stopping")
        cost += float((resp.get("usage") or {}).get("cost") or 0)
        out.append({"id": f"g{i:03d}", "domain": r["domain"], "document_id": r["document_id"], "segment_id": r["segment_id"],
                    "english": r["source"], "gemma_ja": text.strip(), "wmt_ref_ja": r["target"],
                    "model": resp.get("model") or MODEL, "provider": provider, "date": datetime.date.today().isoformat()})
        if i % 20 == 19:
            print(f"{i + 1}/{len(segs)} done, cost so far ${cost:.4f}")
    with open(OUT, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("wrote", OUT, f"({OUT.stat().st_size} bytes), total cost ${cost:.4f}")
