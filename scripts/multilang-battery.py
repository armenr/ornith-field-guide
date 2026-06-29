#!/usr/bin/env python3
# Multi-language correctness battery. Same 3 algorithmic problems in Rust/Python/Go/TS, each with a real
# compile+behavioral-test harness and a self-fix loop. Reports per (lang,problem) convergence rate, avg
# rounds, and first-attempt pass, across seeds.
#
# IMPORTANT (2026-06-29 methodology fix): the test harnesses give ACTIONABLE feedback on failure — the
# first failing input + expected + actual (what a real test runner shows) — not a bare panic/assert.
# A model can only self-correct a logic bug it can localize; bare "assertion failed" feedback made the
# model spiral/give-up and badly understated its self-fix ability. Also: run this SINGLE-STREAM (server
# `-np 1`), because concurrent batched decode in llama.cpp is not batch-invariant (a sequence's logits
# depend on its batch neighbors), which makes per-seed results non-reproducible at temp 0.6.
#
# Args: PORT MODELNAME LABEL "rust,python,go,ts" "1,2,3" [MAXIT]   Env: MAXTOK (default 20000)
import json, re, subprocess, sys, os, urllib.request, tempfile, shutil
PORT, MODEL, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
LANGS = sys.argv[4].split(","); SEEDS = [int(s) for s in sys.argv[5].split(",")]
MAXIT = int(sys.argv[6]) if len(sys.argv) > 6 else 5
MAXTOK = int(os.environ.get("MAXTOK", "20000"))
WORK = tempfile.mkdtemp(prefix=f"mlb_{LABEL}_")

SYS = ("You are a senior engineer. Provide ONE complete, correct, idiomatic, compiling solution in a "
       "single fenced code block for the requested language. Implement EXACTLY the requested API "
       "(names/signatures). Do not add a main/test harness unless asked. When given compiler errors or "
       "failing tests, reason about the root cause and return the FULL corrected program.")

# --- problem specs: per problem, an API description (language-agnostic) + per-language test harness ---
PROBLEMS = {
 "eval": {
   "desc": ("an arithmetic expression evaluator. Support integer literals, binary + - * / with standard "
            "precedence (* / bind tighter), parentheses, unary minus, and arbitrary whitespace. Use "
            "truncating integer division. Signal an error on malformed input or division by zero."),
   "api": {
     "rust":   "fn eval(expr: &str) -> Result<i64, String>",
     "python": "def evaluate(expr: str) -> int   (raise ValueError on malformed input / div by zero)",
     "go":     "func Eval(expr string) (int, error)",
     "ts":     "function evaluate(expr: string): number   (throw an Error on malformed input / div by zero)",
   },
 },
 "lru": {
   "desc": ("a Least-Recently-Used cache with integer keys/values and O(1) get and put. get returns the "
            "value or a 'missing' sentinel; put inserts/updates and evicts the least-recently-used entry "
            "when over capacity. Both get and put count as a use."),
   "api": {
     "rust":   "struct LruCache with fn new(capacity: usize) -> Self, fn get(&mut self, key: i32) -> Option<i32>, fn put(&mut self, key: i32, value: i32)",
     "python": "class LruCache with __init__(self, capacity: int), get(self, key: int) -> int (return -1 if missing), put(self, key: int, value: int) -> None",
     "go":     "type LruCache with NewLruCache(capacity int) *LruCache, (c *LruCache) Get(key int) (int, bool), (c *LruCache) Put(key, value int)",
     "ts":     "class LruCache with constructor(capacity: number), get(key: number): number (return -1 if missing), put(key: number, value: number): void",
   },
 },
 "intervals": {
   "desc": ("merge overlapping closed intervals. Given a list of [start, end] integer intervals (in any "
            "order), return the minimal list of non-overlapping intervals that cover the same points, "
            "sorted ascending by start. Touching intervals like [1,2],[2,3] merge into [1,3]."),
   "api": {
     "rust":   "fn merge(intervals: Vec<(i64, i64)>) -> Vec<(i64, i64)>",
     "python": "def merge(intervals: list[tuple[int,int]]) -> list[tuple[int,int]]",
     "go":     "func Merge(intervals [][2]int) [][2]int",
     "ts":     "function merge(intervals: [number, number][]): [number, number][]",
   },
 },
}

# --- per-language test harnesses (appended to model code). On failure they print ONE actionable line
#     "FAIL: <call> = <actual> but expected <expected>" to stdout; on success they print ALL_PASS_OK. ---
OK = "ALL_PASS_OK"
LANG = {
 "rust": {
   "tag": "rust", "ext": "rs",
   "tests": {
     "eval": f'''
fn main() {{
    let cases: Vec<(&str, Result<i64,()>)> = vec![("2+3*4",Ok(14)),("(2+3)*4",Ok(20)),("10/2-3",Ok(2)),(" -(3 + 4) ",Ok(-7)),("2*-3",Ok(-6)),("7-2-2",Ok(3)),("1/0",Err(())),("2 3",Err(())),("(1+2",Err(())),("",Err(()))];
    for (input, want) in cases {{
        let got = eval(input);
        match (&want, &got) {{
            (Ok(w), Ok(g)) if g == w => {{}},
            (Err(()), Err(_)) => {{}},
            (Ok(w), _) => {{ println!("FAIL: eval({{:?}}) = {{:?}} but expected Ok({{}})", input, got, w); return; }},
            (Err(()), Ok(g)) => {{ println!("FAIL: eval({{:?}}) = Ok({{}}) but expected an Err (malformed input / div-by-zero)", input, g); return; }},
        }}
    }}
    println!("{OK}");
}}''',
     "lru": f'''
fn main() {{
    let mut c = LruCache::new(2);
    c.put(1,1); c.put(2,2);
    let g = c.get(1); if g != Some(1) {{ println!("FAIL: after put(1,1),put(2,2): get(1) = {{:?}} but expected Some(1)", g); return; }}
    c.put(3,3);
    let g = c.get(2); if g != None {{ println!("FAIL: after put(3,3) over capacity 2: get(2) = {{:?}} but expected None (2 is LRU, should be evicted)", g); return; }}
    c.put(4,4);
    let g = c.get(1); if g != None {{ println!("FAIL: after put(4,4): get(1) = {{:?}} but expected None (1 should be evicted)", g); return; }}
    let g = c.get(3); if g != Some(3) {{ println!("FAIL: get(3) = {{:?}} but expected Some(3)", g); return; }}
    let g = c.get(4); if g != Some(4) {{ println!("FAIL: get(4) = {{:?}} but expected Some(4)", g); return; }}
    println!("{OK}");
}}''',
     "intervals": f'''
fn main() {{
    let cases: Vec<(Vec<(i64,i64)>, Vec<(i64,i64)>)> = vec![
        (vec![(1,3),(2,6),(8,10),(15,18)], vec![(1,6),(8,10),(15,18)]),
        (vec![(1,4),(4,5)], vec![(1,5)]),
        (vec![(5,6),(1,3),(2,4)], vec![(1,4),(5,6)]),
    ];
    for (input, want) in cases {{
        let got = merge(input.clone());
        if got != want {{ println!("FAIL: merge({{:?}}) = {{:?}} but expected {{:?}}", input, got, want); return; }}
    }}
    println!("{OK}");
}}''',
   },
   "build": lambda f, b: ["rustc", "-O", f, "-o", b],
   "run": lambda b: [b],
 },
 "python": {
   "tag": "python", "ext": "py",
   "tests": {
     "eval": f'''
if __name__ == "__main__":
    cases = [("2+3*4",14),("(2+3)*4",20),("10/2-3",2),(" -(3 + 4) ",-7),("2*-3",-6),("7-2-2",3)]
    for inp,want in cases:
        try: got = evaluate(inp)
        except Exception as e:
            print(f"FAIL: evaluate({{inp!r}}) raised {{type(e).__name__}}: {{e}} but expected {{want}}"); raise SystemExit
        if got != want:
            print(f"FAIL: evaluate({{inp!r}}) = {{got!r}} but expected {{want}}"); raise SystemExit
    for bad in ["1/0","2 3","(1+2",""]:
        try:
            r = evaluate(bad); print(f"FAIL: evaluate({{bad!r}}) = {{r!r}} but expected a ValueError"); raise SystemExit
        except ValueError: pass
        except SystemExit: raise
        except Exception as e:
            print(f"FAIL: evaluate({{bad!r}}) raised {{type(e).__name__}} but expected a ValueError"); raise SystemExit
    print("{OK}")''',
     "lru": f'''
if __name__ == "__main__":
    c = LruCache(2); c.put(1,1); c.put(2,2)
    g = c.get(1)
    if g != 1: print(f"FAIL: after put(1,1),put(2,2): get(1) = {{g!r}} but expected 1"); raise SystemExit
    c.put(3,3)
    g = c.get(2)
    if g != -1: print(f"FAIL: after put(3,3) over capacity 2: get(2) = {{g!r}} but expected -1 (2 is LRU, evicted)"); raise SystemExit
    c.put(4,4)
    g = c.get(1)
    if g != -1: print(f"FAIL: after put(4,4): get(1) = {{g!r}} but expected -1 (1 evicted)"); raise SystemExit
    g = c.get(3)
    if g != 3: print(f"FAIL: get(3) = {{g!r}} but expected 3"); raise SystemExit
    g = c.get(4)
    if g != 4: print(f"FAIL: get(4) = {{g!r}} but expected 4"); raise SystemExit
    print("{OK}")''',
     "intervals": f'''
if __name__ == "__main__":
    cases = [([(1,3),(2,6),(8,10),(15,18)],[(1,6),(8,10),(15,18)]),([(1,4),(4,5)],[(1,5)]),([(5,6),(1,3),(2,4)],[(1,4),(5,6)])]
    for inp,want in cases:
        got = merge(list(inp))
        if got != want:
            print(f"FAIL: merge({{inp!r}}) = {{got!r}} but expected {{want!r}}"); raise SystemExit
    print("{OK}")''',
   },
   "build": None,
   "run": lambda f: ["python3", f],
 },
 "go": {
   "tag": "go", "ext": "go",
   "tests": {
     "eval": f'''
func main() {{
    type C struct {{ in string; want int; wantErr bool }}
    cases := []C{{{{"2+3*4",14,false}},{{"(2+3)*4",20,false}},{{"10/2-3",2,false}},{{" -(3 + 4) ",-7,false}},{{"2*-3",-6,false}},{{"7-2-2",3,false}},{{"1/0",0,true}},{{"2 3",0,true}},{{"(1+2",0,true}},{{"",0,true}}}}
    for _, c := range cases {{
        g, err := Eval(c.in)
        if c.wantErr {{
            if err == nil {{ fmt.Printf("FAIL: Eval(%q) returned (%d, nil) but an error was expected\\n", c.in, g); return }}
        }} else {{
            if err != nil {{ fmt.Printf("FAIL: Eval(%q) returned error %v but expected %d\\n", c.in, err, c.want); return }}
            if g != c.want {{ fmt.Printf("FAIL: Eval(%q) = %d but expected %d\\n", c.in, g, c.want); return }}
        }}
    }}
    fmt.Println("{OK}")
}}''',
     "lru": f'''
func main() {{
    c := NewLruCache(2)
    c.Put(1,1); c.Put(2,2)
    if v,ok := c.Get(1); !ok || v != 1 {{ fmt.Printf("FAIL: after Put(1,1),Put(2,2): Get(1) = (%d,%v) but expected (1,true)\\n", v, ok); return }}
    c.Put(3,3)
    if v,ok := c.Get(2); ok {{ fmt.Printf("FAIL: after Put(3,3) over capacity 2: Get(2) = (%d,true) but expected a miss (false); 2 is LRU and should be evicted\\n", v); return }}
    c.Put(4,4)
    if v,ok := c.Get(1); ok {{ fmt.Printf("FAIL: after Put(4,4): Get(1) = (%d,true) but expected a miss; 1 should be evicted\\n", v); return }}
    if v,ok := c.Get(3); !ok || v != 3 {{ fmt.Printf("FAIL: Get(3) = (%d,%v) but expected (3,true)\\n", v, ok); return }}
    if v,ok := c.Get(4); !ok || v != 4 {{ fmt.Printf("FAIL: Get(4) = (%d,%v) but expected (4,true)\\n", v, ok); return }}
    fmt.Println("{OK}")
}}''',
     "intervals": f'''
func main() {{
    eq := func(a,b [][2]int) bool {{ if len(a)!=len(b) {{return false}}; for i := range a {{ if a[i]!=b[i] {{return false}} }}; return true }}
    type C struct {{ in, want [][2]int }}
    cases := []C{{
        {{[][2]int{{{{1,3}},{{2,6}},{{8,10}},{{15,18}}}}, [][2]int{{{{1,6}},{{8,10}},{{15,18}}}}}},
        {{[][2]int{{{{1,4}},{{4,5}}}}, [][2]int{{{{1,5}}}}}},
        {{[][2]int{{{{5,6}},{{1,3}},{{2,4}}}}, [][2]int{{{{1,4}},{{5,6}}}}}},
    }}
    for _, c := range cases {{
        got := Merge(c.in)
        if !eq(got, c.want) {{ fmt.Printf("FAIL: Merge(%v) = %v but expected %v\\n", c.in, got, c.want); return }}
    }}
    fmt.Println("{OK}")
}}''',
   },
   "build": None,
   "run": lambda f: ["go", "run", f],
   "preamble": 'package main\nimport "fmt"\n',
 },
 "ts": {
   "tag": "typescript", "ext": "ts",
   "tests": {
     "eval": f'''
{{
const ec:[string,number][]=[["2+3*4",14],["(2+3)*4",20],["10/2-3",2],[" -(3 + 4) ",-7],["2*-3",-6],["7-2-2",3]];
for(const [inp,want] of ec){{
  let got:number; try{{got=evaluate(inp);}}catch(e){{console.log(`FAIL: evaluate(${{JSON.stringify(inp)}}) threw ${{e}} but expected ${{want}}`);process.exit(0);}}
  if(got!==want){{console.log(`FAIL: evaluate(${{JSON.stringify(inp)}}) = ${{got}} but expected ${{want}}`);process.exit(0);}}
}}
for(const bad of ["1/0","2 3","(1+2",""]){{ let threw=false; try{{evaluate(bad);}}catch{{threw=true;}} if(!threw){{console.log(`FAIL: evaluate(${{JSON.stringify(bad)}}) did not throw but expected an Error (malformed input / div-by-zero)`);process.exit(0);}} }}
console.log("{OK}");
}}''',
     "lru": f'''
{{
const c=new LruCache(2); c.put(1,1); c.put(2,2);
let g=c.get(1); if(g!==1){{console.log(`FAIL: after put(1,1),put(2,2): get(1) = ${{g}} but expected 1`);process.exit(0);}}
c.put(3,3);
g=c.get(2); if(g!==-1){{console.log(`FAIL: after put(3,3) over capacity 2: get(2) = ${{g}} but expected -1 (2 is LRU, evicted)`);process.exit(0);}}
c.put(4,4);
g=c.get(1); if(g!==-1){{console.log(`FAIL: after put(4,4): get(1) = ${{g}} but expected -1 (1 evicted)`);process.exit(0);}}
g=c.get(3); if(g!==3){{console.log(`FAIL: get(3) = ${{g}} but expected 3`);process.exit(0);}}
g=c.get(4); if(g!==4){{console.log(`FAIL: get(4) = ${{g}} but expected 4`);process.exit(0);}}
console.log("{OK}");
}}''',
     "intervals": f'''
{{
const eqi=(a:[number,number][],b:[number,number][])=>JSON.stringify(a)===JSON.stringify(b);
const ci:[[number,number][],[number,number][]][]=[
 [[[1,3],[2,6],[8,10],[15,18]],[[1,6],[8,10],[15,18]]],
 [[[1,4],[4,5]],[[1,5]]],
 [[[5,6],[1,3],[2,4]],[[1,4],[5,6]]],
];
for(const [inp,want] of ci){{ const got=merge(inp); if(!eqi(got,want)){{console.log(`FAIL: merge(${{JSON.stringify(inp)}}) = ${{JSON.stringify(got)}} but expected ${{JSON.stringify(want)}}`);process.exit(0);}} }}
console.log("{OK}");
}}''',
   },
   "build": None,
   "run": lambda f: ["bun", "run", f],
 },
}

def chat(msgs, seed):
    body = json.dumps({"model": MODEL, "messages": msgs, "temperature": 0.6, "top_p": 0.95,
                       "top_k": 20, "min_p": 0, "seed": seed, "max_tokens": MAXTOK}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
    try:
        ch = json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]
        return ch["message"].get("content") or "", ch.get("finish_reason")
    except Exception as e:
        return "", f"error:{str(e)[:80]}"

def extract(t, tag):
    b = re.findall(r"```(?:" + tag + r"|[a-z]*)?\s*\n(.*?)```", t, re.S)
    return max(b, key=len) if b else t

def feedback(stdout, stderr):
    for l in stdout.splitlines():
        if l.startswith("FAIL:"):
            return l.strip()
    return (stderr[:2500].strip() or f"no test output; stdout tail: {stdout[-400:]!r}")

def run_lang(lang, prob, seed):
    spec = LANG[lang]; P = PROBLEMS[prob]
    extra = (" Include `package main` and all needed imports; do NOT write func main (a separate test file "
             "with main is added to the same package)." if lang == "go" else "")
    task = (f"Write, in {spec['tag']}, {P['desc']}\nUse EXACTLY this API: {P['api'][lang]}\n"
            "Provide one complete program in a single ```" + spec['tag'] + " code block (definitions only, no main/tests)." + extra)
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": task}]
    first_pass = None
    for it in range(1, MAXIT + 1):
        content, finish = chat(msgs, seed)
        if not content.strip():
            # no code (reasoning ran the whole budget, or server error): nudge once and retry rather
            # than scoring it a quality failure — but bounded by MAXIT.
            if it < MAXIT:
                msgs.append({"role": "assistant", "content": "(ran out of token budget while reasoning)"})
                msgs.append({"role": "user", "content": "You produced no code (the reasoning ran long). Reason more "
                             "concisely, then output the COMPLETE program in one ```" + spec['tag'] + " code block."})
                continue
            return (False, it, (finish or "empty"), first_pass if first_pass is not None else False)
        code = extract(content, spec['tag'])
        ok = False; err = ""
        try:
            if lang == "go":  # two files in one package: model defs + separate runner with main
                f1 = os.path.join(WORK, f"sol_{prob}_{seed}_{it}.go")
                f2 = os.path.join(WORK, f"run_{prob}_{seed}_{it}.go")
                open(f1, "w").write(code)
                open(f2, "w").write('package main\nimport "fmt"\n' + spec["tests"][prob])
                r = subprocess.run(["go", "run", f1, f2], capture_output=True, text=True, timeout=120)
                ok = OK in r.stdout; err = "" if ok else feedback(r.stdout, r.stderr)
            else:
                src = code + "\n" + spec["tests"][prob]
                f = os.path.join(WORK, f"{lang}_{prob}_{seed}_{it}.{spec['ext']}")
                open(f, "w").write(src)
                if spec["build"]:
                    b = os.path.join(WORK, "bin")
                    c = subprocess.run(spec["build"](f, b), capture_output=True, text=True, timeout=120)
                    if c.returncode != 0:
                        err = c.stderr[:2500]
                    else:
                        r = subprocess.run(spec["run"](b), capture_output=True, text=True, timeout=30)
                        ok = OK in r.stdout; err = "" if ok else feedback(r.stdout, r.stderr)
                else:
                    r = subprocess.run(spec["run"](f), capture_output=True, text=True, timeout=60)
                    ok = OK in r.stdout; err = "" if ok else feedback(r.stdout, r.stderr)
        except subprocess.TimeoutExpired:
            err = "the program ran past the timeout — likely an infinite loop."
        if first_pass is None: first_pass = ok
        if ok:
            return (True, it, "pass", first_pass)
        msgs.append({"role": "assistant", "content": content})
        msgs.append({"role": "user", "content": f"Your {spec['tag']} solution failed.\n{err}\nReason about the root cause and return the COMPLETE corrected program in one ```{spec['tag']} block."})
    return (False, MAXIT, "no-converge", first_pass if first_pass is not None else False)

print(f"=== Battery {LABEL}: langs={LANGS} probs={list(PROBLEMS)} seeds={SEEDS} MAXTOK={MAXTOK} (rich-feedback) ===")
summary = {}
for lang in LANGS:
    for prob in PROBLEMS:
        rows = []
        for s in SEEDS:
            ok, rounds, why, fp = run_lang(lang, prob, s)
            rows.append((s, ok, rounds, why, fp))
            print(f"[{LABEL}] {lang:7s} {prob:10s} seed={s}: {'PASS' if ok else 'FAIL':4s} rounds={rounds} first_try={fp} ({why})")
            sys.stdout.flush()
        npass = sum(1 for r in rows if r[1]); nfp = sum(1 for r in rows if r[4]); n = len(rows)
        avg = sum(r[2] for r in rows if r[1]) / npass if npass else 0
        summary[f"{lang}/{prob}"] = (npass, n, nfp, round(avg, 1))
        print(f"[{LABEL}] >>> {lang}/{prob}: converged {npass}/{n}, first-try {nfp}/{n}, avg_rounds {avg:.1f}")
        sys.stdout.flush()
print(f"\n=== {LABEL} SUMMARY (lang/prob: converged, first-try, avg_rounds) ===")
for k, v in summary.items():
    print(f"  {k:18s}: converged {v[0]}/{v[1]}, first-try {v[2]}/{v[1]}, avg_rounds {v[3]}")
tot_c = sum(v[0] for v in summary.values()); tot_n = sum(v[1] for v in summary.values()); tot_fp = sum(v[2] for v in summary.values())
print(f"  {'TOTAL':18s}: converged {tot_c}/{tot_n}, first-try {tot_fp}/{tot_n}")
shutil.rmtree(WORK, ignore_errors=True)
