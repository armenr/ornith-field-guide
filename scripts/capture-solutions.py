#!/usr/bin/env python3
# Unified rich-feedback coding harness: runs a compile+behavioral-test SELF-FIX loop per (lang, problem,
# seed), records convergence (converged?/rounds/first-try/finish), AND saves the model's FINAL solution
# code (definitions only) for a downstream blind code-quality head-to-head. Same prompts as the family's
# battery; ACTIONABLE feedback (first failing input + expected + actual) so a model can localize bugs.
#
# RUN SINGLE-STREAM (server `-np 1`, or one process at a time): concurrent batched decode in llama.cpp
# is not batch-invariant, which makes per-seed results non-reproducible at temp 0.6.
#
# Args: PORT MODEL_LABEL OUTDIR "rust,python,go,ts" "eval,lru,intervals,trie,regex" "1,2,3" [MAXIT]
# Env:  MAXTOK (default 40000)
import json, re, subprocess, sys, os, urllib.request, tempfile, shutil, time
PORT, MODEL, OUTDIR = sys.argv[1], sys.argv[2], sys.argv[3]
LANGS = sys.argv[4].split(","); PROBS = sys.argv[5].split(","); SEEDS = [int(s) for s in sys.argv[6].split(",")]
MAXIT = int(sys.argv[7]) if len(sys.argv) > 7 else 6
MAXTOK = int(os.environ.get("MAXTOK", "40000"))
WORK = tempfile.mkdtemp(prefix=f"cap_{MODEL}_"); os.makedirs(OUTDIR, exist_ok=True)

SYS = ("You are a senior engineer. Provide ONE complete, correct, idiomatic, compiling solution in a "
       "single fenced code block for the requested language. Implement EXACTLY the requested API "
       "(names/signatures). Do not add a main/test harness unless asked. When given compiler errors or "
       "failing tests, reason about the root cause and return the FULL corrected program.")
OK = "ALL_PASS_OK"

PROBLEMS = {
 "eval": {"desc": ("an arithmetic expression evaluator. Support integer literals, binary + - * / with standard "
            "precedence (* / bind tighter), parentheses, unary minus, and arbitrary whitespace. Use truncating "
            "integer division. Signal an error on malformed input or division by zero."),
   "api": {"rust": "fn eval(expr: &str) -> Result<i64, String>",
     "python": "def evaluate(expr: str) -> int   (raise ValueError on malformed input / div by zero)",
     "go": "func Eval(expr string) (int, error)",
     "ts": "function evaluate(expr: string): number   (throw an Error on malformed input / div by zero)"}},
 "lru": {"desc": ("a Least-Recently-Used cache with integer keys/values and O(1) get and put. get returns the "
            "value or a 'missing' sentinel; put inserts/updates and evicts the least-recently-used entry when "
            "over capacity. Both get and put count as a use."),
   "api": {"rust": "struct LruCache with fn new(capacity: usize) -> Self, fn get(&mut self, key: i32) -> Option<i32>, fn put(&mut self, key: i32, value: i32)",
     "python": "class LruCache with __init__(self, capacity: int), get(self, key: int) -> int (return -1 if missing), put(self, key: int, value: int) -> None",
     "go": "type LruCache with NewLruCache(capacity int) *LruCache, (c *LruCache) Get(key int) (int, bool), (c *LruCache) Put(key, value int)",
     "ts": "class LruCache with constructor(capacity: number), get(key: number): number (return -1 if missing), put(key: number, value: number): void"}},
 "intervals": {"desc": ("merge overlapping closed intervals. Given a list of [start, end] integer intervals (in any "
            "order), return the minimal list of non-overlapping intervals that cover the same points, sorted "
            "ascending by start. Touching intervals like [1,2],[2,3] merge into [1,3]."),
   "api": {"rust": "fn merge(intervals: Vec<(i64, i64)>) -> Vec<(i64, i64)>",
     "python": "def merge(intervals: list[tuple[int,int]]) -> list[tuple[int,int]]",
     "go": "func Merge(intervals [][2]int) [][2]int",
     "ts": "function merge(intervals: [number, number][]): [number, number][]"}},
 "trie": {"desc": ("a trie (prefix tree). search returns true only for whole words that were inserted; starts_with "
            "returns true if any inserted word has the given prefix."),
   "api": {"rust": "struct Trie with fn new() -> Self, fn insert(&mut self, word: &str), fn search(&self, word: &str) -> bool, fn starts_with(&self, prefix: &str) -> bool"}},
 "regex": {"desc": ("a regular-expression matcher that FULL-matches (the pattern matches the ENTIRE text). Support: "
            "literal chars, '.' (any single char), '*' (zero or more of the preceding element), '+' (one or more), "
            "'?' (zero or one), '(' ')' grouping, and '|' alternation. Quantifiers apply to the preceding element "
            "(a literal, '.', or a group). Use backtracking."),
   "api": {"rust": "fn matches(pattern: &str, text: &str) -> bool"}},
}

LANG = {
 "rust": {"tag": "rust", "ext": "rs", "build": lambda f, b: ["rustc", "-O", f, "-o", b], "run": lambda b: [b], "tests": {
   "eval": f'''
fn main() {{
    let cases: Vec<(&str, Result<i64,()>)> = vec![("2+3*4",Ok(14)),("(2+3)*4",Ok(20)),("10/2-3",Ok(2)),(" -(3 + 4) ",Ok(-7)),("2*-3",Ok(-6)),("7-2-2",Ok(3)),("1/0",Err(())),("2 3",Err(())),("(1+2",Err(())),("",Err(()))];
    for (input, want) in cases {{
        let got = eval(input);
        match (&want, &got) {{
            (Ok(w), Ok(g)) if g == w => {{}}, (Err(()), Err(_)) => {{}},
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
   "trie": f'''
fn main() {{
    let mut t = Trie::new();
    t.insert("apple");
    if t.search("apple") != true {{ println!("FAIL: after insert(\\"apple\\"): search(\\"apple\\") = {{}} but expected true", t.search("apple")); return; }}
    if t.search("app") != false {{ println!("FAIL: search(\\"app\\") = {{}} but expected false (\\"app\\" not inserted as a whole word)", t.search("app")); return; }}
    if t.starts_with("app") != true {{ println!("FAIL: starts_with(\\"app\\") = {{}} but expected true (\\"apple\\" has prefix \\"app\\")", t.starts_with("app")); return; }}
    if t.search("appl") != false {{ println!("FAIL: search(\\"appl\\") = {{}} but expected false", t.search("appl")); return; }}
    if t.starts_with("apple") != true {{ println!("FAIL: starts_with(\\"apple\\") = {{}} but expected true", t.starts_with("apple")); return; }}
    if t.starts_with("apx") != false {{ println!("FAIL: starts_with(\\"apx\\") = {{}} but expected false", t.starts_with("apx")); return; }}
    t.insert("app");
    if t.search("app") != true {{ println!("FAIL: after insert(\\"app\\"): search(\\"app\\") = {{}} but expected true", t.search("app")); return; }}
    t.insert("banana");
    if t.search("banana") != true {{ println!("FAIL: after insert(\\"banana\\"): search(\\"banana\\") = {{}} but expected true", t.search("banana")); return; }}
    if t.starts_with("ban") != true {{ println!("FAIL: starts_with(\\"ban\\") = {{}} but expected true", t.starts_with("ban")); return; }}
    if t.search("ban") != false {{ println!("FAIL: search(\\"ban\\") = {{}} but expected false", t.search("ban")); return; }}
    if t.search("bananas") != false {{ println!("FAIL: search(\\"bananas\\") = {{}} but expected false", t.search("bananas")); return; }}
    println!("{OK}");
}}''',
   "regex": f'''
fn main() {{
    let cases: Vec<(&str,&str,bool)> = vec![("abc","abc",true),("abc","abd",false),("a.c","axc",true),("a*","",true),("a*","aaaa",true),("a+","aaa",true),("a+","",false),("ab?c","ac",true),("ab?c","abc",true),("a|b","a",true),("a|b","b",true),("a|b","c",false),("(ab)+","abab",true),("(ab)+","aba",false),("(a|b)*c","ababc",true),("a.*z","aXYZz",true),("a.*z","aXYZ",false),("abc","ab",false)];
    for (pat,text,want) in cases {{
        let got = matches(pat,text);
        if got != want {{ println!("FAIL: matches({{:?}}, {{:?}}) = {{}} but expected {{}}", pat, text, got, want); return; }}
    }}
    println!("{OK}");
}}''',
 }},
 "python": {"tag": "python", "ext": "py", "build": None, "run": lambda f: ["python3", f], "tests": {
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
 }},
 "go": {"tag": "go", "ext": "go", "build": None, "run": None, "tests": {
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
 }},
 "ts": {"tag": "typescript", "ext": "ts", "build": None, "run": lambda f: ["bun", "run", f], "tests": {
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
 }},
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

def strip_rust_main(code):
    i = code.find("fn main")
    if i == -1: return code
    b = code.find("{", i); d = 0; j = b
    while j < len(code):
        if code[j] == "{": d += 1
        elif code[j] == "}":
            d -= 1
            if d == 0: break
        j += 1
    return (code[:i] + code[j+1:]).strip()

def feedback(stdout, stderr):
    for l in stdout.splitlines():
        if l.startswith("FAIL:"): return l.strip()
    return (stderr[:2500].strip() or f"no test output; stdout tail: {stdout[-400:]!r}")

def test_code(lang, prob, code):
    spec = LANG[lang]
    try:
        if lang == "go":
            f1 = os.path.join(WORK, f"sol_{prob}.go"); f2 = os.path.join(WORK, f"run_{prob}.go")
            open(f1, "w").write(code); open(f2, "w").write('package main\nimport "fmt"\n' + spec["tests"][prob])
            r = subprocess.run(["go", "run", f1, f2], capture_output=True, text=True, timeout=120)
            return (OK in r.stdout, feedback(r.stdout, r.stderr))
        src = code + "\n" + spec["tests"][prob]
        f = os.path.join(WORK, f"{lang}_{prob}.{spec['ext']}"); open(f, "w").write(src)
        if spec["build"]:
            b = os.path.join(WORK, f"bin_{prob}")
            c = subprocess.run(spec["build"](f, b), capture_output=True, text=True, timeout=120)
            if c.returncode != 0: return (False, c.stderr[:2000])
            r = subprocess.run(spec["run"](b), capture_output=True, text=True, timeout=30)
            return (OK in r.stdout, feedback(r.stdout, r.stderr))
        r = subprocess.run(spec["run"](f), capture_output=True, text=True, timeout=60)
        return (OK in r.stdout, feedback(r.stdout, r.stderr))
    except subprocess.TimeoutExpired:
        return (False, "the program ran past the timeout — likely an infinite loop.")

def run_cell(lang, prob, seed):
    spec = LANG[lang]; P = PROBLEMS[prob]
    extra = (" Include `package main` and all needed imports; do NOT write func main (a separate test file "
             "with main is added to the same package)." if lang == "go" else "")
    task = (f"Write, in {spec['tag']}, {P['desc']}\nUse EXACTLY this API: {P['api'][lang]}\n"
            "Provide one complete program in a single ```" + spec['tag'] + " code block (definitions only, no main/tests)." + extra)
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": task}]
    last_code = ""; converged = False; rounds = 0; finish = ""; first_pass = None
    for it in range(1, MAXIT + 1):
        rounds = it
        content, finish = chat(msgs, seed)
        if not content.strip():
            if it < MAXIT:
                msgs.append({"role": "assistant", "content": "(ran out of token budget while reasoning)"})
                msgs.append({"role": "user", "content": "You produced no code (the reasoning ran long). Reason more "
                             "concisely, then output the COMPLETE program in one ```" + spec['tag'] + " code block."})
                continue
            break
        code = extract(content, spec['tag'])
        if lang == "rust": code = strip_rust_main(code)
        last_code = code
        passes, detail = test_code(lang, prob, code)
        if first_pass is None: first_pass = passes
        if passes:
            converged = True; break
        msgs.append({"role": "assistant", "content": content})
        msgs.append({"role": "user", "content": f"Your {spec['tag']} solution failed.\n{detail}\nReason about the root cause and return the COMPLETE corrected program in one ```{spec['tag']} block."})
    out = os.path.join(OUTDIR, f"{lang}__{prob}__s{seed}.{spec['ext']}")
    open(out, "w").write(last_code + "\n")
    meta = {"model": MODEL, "lang": lang, "tag": spec["tag"], "prob": prob, "seed": seed, "rounds": rounds,
            "converged": bool(converged), "first_try": bool(first_pass), "finish": finish,
            "code_chars": len(last_code), "file": os.path.basename(out)}
    print(f"[{MODEL}] {lang:7s} {prob:10s} seed={seed}: {'PASS' if converged else 'FAIL'} rounds={rounds} "
          f"first_try={first_pass} finish={finish} chars={len(last_code)}", flush=True)
    return meta

for _ in range(180):
    try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3); break
    except Exception: time.sleep(1)

print(f"=== CAPTURE {MODEL}: langs={LANGS} probs={PROBS} seeds={SEEDS} MAXTOK={MAXTOK} (rich, single-stream) ===")
metas = []
for lang in LANGS:
    for prob in PROBS:
        if prob in ("trie", "regex") and lang != "rust": continue
        if prob not in LANG[lang]["tests"]: continue
        for seed in SEEDS:
            metas.append(run_cell(lang, prob, seed))
open(os.path.join(OUTDIR, f"_meta_{MODEL}.jsonl"), "w").write("\n".join(json.dumps(m) for m in metas) + "\n")
# convergence summary
print(f"\n=== {MODEL} SUMMARY (lang/prob: converged, first-try, avg_rounds) ===")
keys = []
for m in metas:
    k = f"{m['lang']}/{m['prob']}"
    if k not in keys: keys.append(k)
tc = tn = tf = 0
for k in keys:
    rows = [m for m in metas if f"{m['lang']}/{m['prob']}" == k]
    npass = sum(1 for m in rows if m["converged"]); nfp = sum(1 for m in rows if m["first_try"]); n = len(rows)
    avg = (sum(m["rounds"] for m in rows if m["converged"]) / npass) if npass else 0
    tc += npass; tn += n; tf += nfp
    print(f"  {k:18s}: converged {npass}/{n}, first-try {nfp}/{n}, avg_rounds {avg:.1f}")
print(f"  {'TOTAL':18s}: converged {tc}/{tn}, first-try {tf}/{tn}")
print(f"\n=== {MODEL}: captured {len(metas)} solutions to {OUTDIR} ===")
shutil.rmtree(WORK, ignore_errors=True)
