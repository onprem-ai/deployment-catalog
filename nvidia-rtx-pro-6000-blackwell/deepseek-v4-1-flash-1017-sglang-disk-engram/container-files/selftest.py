"""Debug self-test (onprem.ai profile, not for production): send real chat requests to
the local SGLang server before the replica turns READY, so the DSV41_SM120_PREFILL_CHECK
lines of real requests land in the deployment log. Prints SELFTEST lines only."""
import json
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:%s" % (sys.argv[1] if len(sys.argv) > 1 else "8081")
S = ("The Gotthard Base Tunnel in Switzerland is a railway tunnel through the Alps, "
     "opened in 2016, and it is the longest railway tunnel in the world. ")
Q_MUL = " Question: what is 6 times 7? Answer with only the number."
Q_YEAR = " In which year did the tunnel open? Answer with only the year."


def log(*a):
    print("SELFTEST", *a, flush=True)


def chat(content, max_tokens=24):
    body = json.dumps({"model": "deepseek-v4.1-flash", "temperature": 0.0, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    text = (m.get("content") or "") + " || " + (m.get("reasoning_content") or "")[:80]
    return d["usage"]["prompt_tokens"], text.replace("\n", " ")[:160], time.time() - t


def run(name, content, max_tokens=24):
    try:
        n, text, dt = chat(content, max_tokens)
        log("%s prompt=%d %.1fs answer=%r" % (name, n, dt, text))
    except Exception as ex:
        log("%s ERROR %r" % (name, ex))


MODE_FILE = sys.argv[2] if len(sys.argv) > 2 else None


def set_mode(mode):
    with open(MODE_FILE + ".tmp", "w") as f:
        f.write(mode)
    import os
    os.replace(MODE_FILE + ".tmp", MODE_FILE)


def flush():
    last = None
    for _ in range(30):
        try:
            req = urllib.request.Request(BASE + "/flush_cache", b"", method="POST")
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status
        except Exception as ex:
            last = repr(ex)[:80]
        time.sleep(2)
    return last


CASES = [
    ("short", "What is 6 times 7? Answer with only the number.", "42"),
    ("p120", S * 3 + Q_MUL, "42"),
    ("p318", S * 9 + Q_MUL, "42"),
    ("p1669", S * 50 + Q_YEAR, "2016"),
    ("p6619", S * 200 + Q_YEAR, "2016"),
    ("needle", S * 120 + " The secret code is 7319-ORCHID. " + S * 120 +
     " What is the secret code? Answer with only the code.", "7319-ORCHID"),
]
WORDS = ("alpine railway tunnel granite engineer schedule freight passenger ventilation "
         "drill segment portal station signal gradient concrete survey").split()


def unique_cases(seed, n):
    import random
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a, b = rng.randint(11, 97), rng.randint(11, 97)
        k = rng.choice([40, 80, 200, 900, 1800])
        filler = " ".join(rng.choice(WORDS) for _ in range(k))
        out.append(("u%d-%d" % (seed, i), "Notes %d-%d: %s\nIgnore the notes. Question: what is %d times %d? "
                    "Answer with only the number." % (seed, i, filler, a, b), str(a * b)))
    return out


def run_mode(mode, seed):
    set_mode(mode)
    log("mode %s flush=%s" % (mode, flush()))
    ok = 0
    t_mode = time.time()
    cases = CASES + unique_cases(seed, 8)
    for name, content, want in cases:
        try:
            n, text, dt = chat(content)
            good = text.split("||")[0].strip().replace(",", "") == want
            ok += good
            log("mode=%s %s prompt=%d %.2fs %s answer=%r" % (mode, name, n, dt, "OK" if good else "WRONG", text[:90]))
        except Exception as ex:
            log("mode=%s %s ERROR %r" % (mode, name, ex))
    # mixed batch: concurrent prefills while one request decodes
    results = []

    def one(i):
        try:
            n, text, dt = chat(S * (3 + 7 * i) + Q_MUL)
            results.append(text.split("||")[0].strip() == "42")
        except Exception:
            results.append(False)
    th = [threading.Thread(target=one, args=(i,)) for i in range(4)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    ok += sum(results)
    total = len(cases) + 4
    log("mode=%s RESULT %d/%d correct in %.1fs" % (mode, ok, total, time.time() - t_mode))
    return ok == total


def main():
    t0 = time.time()
    while time.time() - t0 < 3600:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        time.sleep(5)
    log("server healthy after %.0fs" % (time.time() - t0))
    verdict = []
    for i, mode in enumerate(("fi_sync", "check", "fi_fresh", "fi_perpbs", "dec64", "fi_sync", "triton")):
        verdict.append((mode, run_mode(mode, 100 + i)))
    set_mode("triton")
    log("VERDICT %s -> serving with prefill mode triton, flush=%s" % (verdict, flush()))
    log("done after %.0fs" % (time.time() - t0))


main()
