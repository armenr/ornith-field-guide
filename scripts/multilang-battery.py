#!/usr/bin/env python3
# Multi-language Q4-vs-Q6 correctness battery. Same 3 algorithmic problems in Rust/Python/Go/TS,
# each with a real compile+behavioral-test harness and a self-fix loop. Reports per (lang,problem)
# convergence rate, avg rounds, and first-attempt pass, across seeds.
# Args: PORT MODELNAME LABEL "rust,python,go,ts" "1,2,3" [MAXIT]
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

# --- per-language test harnesses (appended to model code) + how to build/run + success sentinel ---
OK = "ALL_PASS_OK"
LANG = {
 "rust": {
   "tag": "rust", "ext": "rs",
   "tests": {
     "eval": f'''
fn main() {{
    assert_eq!(eval("2+3*4"), Ok(14)); assert_eq!(eval("(2+3)*4"), Ok(20));
    assert_eq!(eval("10/2-3"), Ok(2)); assert_eq!(eval(" -(3 + 4) "), Ok(-7));
    assert_eq!(eval("2*-3"), Ok(-6)); assert_eq!(eval("7-2-2"), Ok(3));
    assert!(eval("1/0").is_err()); assert!(eval("2 3").is_err()); assert!(eval("(1+2").is_err()); assert!(eval("").is_err());
    println!("{OK}");
}}''',
     "lru": f'''
fn main() {{
    let mut c = LruCache::new(2);
    c.put(1,1); c.put(2,2); assert_eq!(c.get(1), Some(1));
    c.put(3,3); assert_eq!(c.get(2), None); c.put(4,4); assert_eq!(c.get(1), None);
    assert_eq!(c.get(3), Some(3)); assert_eq!(c.get(4), Some(4));
    println!("{OK}");
}}''',
     "intervals": f'''
fn main() {{
    assert_eq!(merge(vec![(1,3),(2,6),(8,10),(15,18)]), vec![(1,6),(8,10),(15,18)]);
    assert_eq!(merge(vec![(1,4),(4,5)]), vec![(1,5)]);
    assert_eq!(merge(vec![(5,6),(1,3),(2,4)]), vec![(1,4),(5,6)]);
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
    assert evaluate("2+3*4")==14; assert evaluate("(2+3)*4")==20; assert evaluate("10/2-3")==2
    assert evaluate(" -(3 + 4) ")==-7; assert evaluate("2*-3")==-6; assert evaluate("7-2-2")==3
    for bad in ["1/0","2 3","(1+2",""]:
        try: evaluate(bad); raise SystemExit("expected error for "+repr(bad))
        except ValueError: pass
    print("{OK}")''',
     "lru": f'''
if __name__ == "__main__":
    c = LruCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1
    c.put(3,3); assert c.get(2)==-1; c.put(4,4); assert c.get(1)==-1
    assert c.get(3)==3; assert c.get(4)==4
    print("{OK}")''',
     "intervals": f'''
if __name__ == "__main__":
    assert merge([(1,3),(2,6),(8,10),(15,18)])==[(1,6),(8,10),(15,18)]
    assert merge([(1,4),(4,5)])==[(1,5)]
    assert merge([(5,6),(1,3),(2,4)])==[(1,4),(5,6)]
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
    chk := func(g int, e int, err error) {{ if err != nil || g != e {{ panic("eval mismatch") }} }}
    g,err := Eval("2+3*4"); chk(g,14,err)
    g,err = Eval("(2+3)*4"); chk(g,20,err)
    g,err = Eval("10/2-3"); chk(g,2,err)
    g,err = Eval(" -(3 + 4) "); chk(g,-7,err)
    g,err = Eval("2*-3"); chk(g,-6,err)
    for _, bad := range []string{{"1/0","2 3","(1+2",""}} {{ if _,e := Eval(bad); e == nil {{ panic("expected error") }} }}
    fmt.Println("{OK}")
}}''',
     "lru": f'''
func main() {{
    c := NewLruCache(2)
    c.Put(1,1); c.Put(2,2)
    if v,ok := c.Get(1); !ok || v != 1 {{ panic("g1") }}
    c.Put(3,3)
    if _,ok := c.Get(2); ok {{ panic("g2 should miss") }}
    c.Put(4,4)
    if _,ok := c.Get(1); ok {{ panic("g1 should miss") }}
    if v,ok := c.Get(3); !ok || v != 3 {{ panic("g3") }}
    if v,ok := c.Get(4); !ok || v != 4 {{ panic("g4") }}
    fmt.Println("{OK}")
}}''',
     "intervals": f'''
func main() {{
    eq := func(a,b [][2]int) bool {{ if len(a)!=len(b) {{return false}}; for i := range a {{ if a[i]!=b[i] {{return false}} }}; return true }}
    if !eq(Merge([][2]int{{{{1,3}},{{2,6}},{{8,10}},{{15,18}}}}), [][2]int{{{{1,6}},{{8,10}},{{15,18}}}}) {{ panic("m1") }}
    if !eq(Merge([][2]int{{{{1,4}},{{4,5}}}}), [][2]int{{{{1,5}}}}) {{ panic("m2") }}
    if !eq(Merge([][2]int{{{{5,6}},{{1,3}},{{2,4}}}}), [][2]int{{{{1,4}},{{5,6}}}}) {{ panic("m3") }}
    fmt.Println("{OK}")
}}''',
   },
   "build": None,
   "run": lambda f: ["go", "run", f],
   "preamble": 'package main\nimport "fmt"\n',  # ensure imports; model code goes after
 },
 "ts": {
   "tag": "typescript", "ext": "ts",
   "tests": {
     "eval": f'''
function ck(c:boolean){{ if(!c) throw new Error("fail"); }}
ck(evaluate("2+3*4")===14); ck(evaluate("(2+3)*4")===20); ck(evaluate("10/2-3")===2);
ck(evaluate(" -(3 + 4) ")===-7); ck(evaluate("2*-3")===-6); ck(evaluate("7-2-2")===3);
for(const bad of ["1/0","2 3","(1+2",""]){{ let threw=false; try{{evaluate(bad);}}catch{{threw=true;}} ck(threw); }}
console.log("{OK}");''',
     "lru": f'''
function ck(c:boolean){{ if(!c) throw new Error("fail"); }}
const c=new LruCache(2); c.put(1,1); c.put(2,2); ck(c.get(1)===1);
c.put(3,3); ck(c.get(2)===-1); c.put(4,4); ck(c.get(1)===-1);
ck(c.get(3)===3); ck(c.get(4)===4);
console.log("{OK}");''',
     "intervals": f'''
function ck(c:boolean){{ if(!c) throw new Error("fail"); }}
function eq(a:[number,number][],b:[number,number][]){{ return JSON.stringify(a)===JSON.stringify(b); }}
ck(eq(merge([[1,3],[2,6],[8,10],[15,18]]),[[1,6],[8,10],[15,18]]));
ck(eq(merge([[1,4],[4,5]]),[[1,5]]));
ck(eq(merge([[5,6],[1,3],[2,4]]),[[1,4],[5,6]]));
console.log("{OK}");''',
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
        ch = json.load(urllib.request.urlopen(req, timeout=1200))["choices"][0]
        return ch["message"].get("content") or "", ch.get("finish_reason")
    except Exception as e:
        return "", f"error:{str(e)[:80]}"

def extract(t, tag):
    b = re.findall(r"```(?:" + tag + r"|[a-z]*)?\s*\n(.*?)```", t, re.S)
    return max(b, key=len) if b else t

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
        if not content.strip():  # length-exhausted, server error, or empty -> fail this cell gracefully
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
                ok = OK in r.stdout; err = "" if ok else (r.stderr[:2500] or f"no OK; stdout={r.stdout[-400:]!r}")
            else:
                src = code + "\n" + spec["tests"][prob]
                f = os.path.join(WORK, f"{lang}_{prob}_{seed}_{it}.{spec['ext']}")
                open(f, "w").write(src)
                if spec["build"]:
                    b = os.path.join(WORK, "bin")
                    c = subprocess.run(spec["build"](f, b), capture_output=True, text=True, timeout=120)
                    if c.returncode != 0: err = c.stderr[:2500]
                    else:
                        r = subprocess.run(spec["run"](b), capture_output=True, text=True, timeout=30)
                        ok = OK in r.stdout; err = "" if ok else f"runtime: stdout={r.stdout[-400:]!r} stderr={r.stderr[-600:]!r}"
                else:
                    r = subprocess.run(spec["run"](f), capture_output=True, text=True, timeout=60)
                    ok = OK in r.stdout; err = "" if ok else (r.stderr[:2500] or f"no OK; stdout={r.stdout[-400:]!r}")
        except subprocess.TimeoutExpired:
            err = "timeout (likely infinite loop)"
        if first_pass is None: first_pass = ok
        if ok:
            return (True, it, "pass", first_pass)
        msgs.append({"role": "assistant", "content": content})
        msgs.append({"role": "user", "content": f"Your {spec['tag']} solution failed.\n{err}\nReason about the root cause and return the COMPLETE corrected program in one ```{spec['tag']} block."})
    return (False, MAXIT, "no-converge", first_pass if first_pass is not None else False)

print(f"=== Battery {LABEL}: langs={LANGS} probs={list(PROBLEMS)} seeds={SEEDS} ===")
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
