#!/usr/bin/env python3
# Generic agentic self-correction loop. Args: PORT LABEL MAXIT PROBLEM
import json, re, subprocess, sys, urllib.request, time, os

PORT, LABEL, MAXIT, PROBLEM = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
SCR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'); os.makedirs(SCR, exist_ok=True)

PROBLEMS = {
 "eval": (
   ("Implement an arithmetic expression evaluator in Rust. Exactly this API: "
    "fn eval(expr: &str) -> Result<i64, String>. Support integer literals, the binary operators "
    "+ - * / with standard precedence (* and / bind tighter than + and -), parentheses, unary minus, "
    "and arbitrary whitespace. Use truncating integer division. Return Err for malformed input or "
    "division by zero. Provide the full program in one ```rust code block."),
   r'''
fn main() {
    assert_eq!(eval("2+3*4"), Ok(14));
    assert_eq!(eval("(2+3)*4"), Ok(20));
    assert_eq!(eval("10/2-3"), Ok(2));
    assert_eq!(eval(" -(3 + 4) "), Ok(-7));
    assert_eq!(eval("2*-3"), Ok(-6));
    assert_eq!(eval("100/(2*5)"), Ok(10));
    assert_eq!(eval("7-2-2"), Ok(3));
    assert_eq!(eval("2*3+4*5"), Ok(26));
    assert!(eval("1/0").is_err());
    assert!(eval("2 3").is_err());
    assert!(eval("(1+2").is_err());
    assert!(eval("").is_err());
    println!("ALL EVAL ASSERTIONS PASSED");
}
''', "ALL EVAL ASSERTIONS PASSED"),

 "trie": (
   ("Implement a trie (prefix tree) in Rust. Exactly this API: struct Trie with "
    "fn new() -> Self, fn insert(&mut self, word: &str), fn search(&self, word: &str) -> bool, "
    "and fn starts_with(&self, prefix: &str) -> bool. search returns true only for whole words that "
    "were inserted; starts_with returns true if any inserted word has the given prefix. "
    "Provide the full program in one ```rust code block."),
   r'''
fn main() {
    let mut t = Trie::new();
    t.insert("apple");
    assert_eq!(t.search("apple"), true);
    assert_eq!(t.search("app"), false);
    assert_eq!(t.starts_with("app"), true);
    assert_eq!(t.search("appl"), false);
    assert_eq!(t.starts_with("apple"), true);
    assert_eq!(t.starts_with("apx"), false);
    t.insert("app");
    assert_eq!(t.search("app"), true);
    assert_eq!(t.starts_with("app"), true);
    t.insert("banana");
    assert_eq!(t.search("banana"), true);
    assert_eq!(t.starts_with("ban"), true);
    assert_eq!(t.search("ban"), false);
    assert_eq!(t.search("bananas"), false);
    println!("ALL TRIE ASSERTIONS PASSED");
}
''', "ALL TRIE ASSERTIONS PASSED"),

 "regex": (
   ("Implement a regular-expression matcher in Rust. Exactly this API: "
    "fn matches(pattern: &str, text: &str) -> bool. It must FULL-match (the pattern matches the "
    "ENTIRE text). Support: literal characters, '.' (any single char), '*' (zero or more of the "
    "preceding element), '+' (one or more), '?' (zero or one), '(' ')' grouping, and '|' alternation. "
    "Quantifiers apply to the preceding element (a literal, '.', or a group). Use backtracking. "
    "Provide the full program in one ```rust code block."),
   r'''
fn main() {
    assert!(matches("abc", "abc"));
    assert!(!matches("abc", "abd"));
    assert!(matches("a.c", "axc"));
    assert!(matches("a*", ""));
    assert!(matches("a*", "aaaa"));
    assert!(matches("a+", "aaa"));
    assert!(!matches("a+", ""));
    assert!(matches("ab?c", "ac"));
    assert!(matches("ab?c", "abc"));
    assert!(matches("a|b", "a"));
    assert!(matches("a|b", "b"));
    assert!(!matches("a|b", "c"));
    assert!(matches("(ab)+", "abab"));
    assert!(!matches("(ab)+", "aba"));
    assert!(matches("(a|b)*c", "ababc"));
    assert!(matches("a.*z", "aXYZz"));
    assert!(!matches("a.*z", "aXYZ"));
    assert!(!matches("abc", "ab"));
    println!("ALL REGEX ASSERTIONS PASSED");
}
''', "ALL REGEX ASSERTIONS PASSED"),
}

TASK, TEST_MAIN, SUCCESS = PROBLEMS[PROBLEM]
SYS = ("You are a senior Rust engineer. Provide one complete, correct, idiomatic, compiling solution "
       "in a single ```rust code block. When given compiler errors or failing tests, reason about the "
       "root cause and return the FULL corrected program.")

def chat(messages):
    # max_tokens MUST be large: Ornith reasons ~30K+ tokens on hard problems (regex needed >32K
    # of thinking alone). Too small a budget => it never closes </think>, content is empty, and the
    # loop "fails" on a phantom missing-function error. Override with MAXTOK; default fits 65536 ctx.
    body = json.dumps({"model":os.environ.get("MODEL","x"),"messages":messages,"temperature":0.6,"top_p":0.95,
                       "top_k":20,"min_p":0,"seed":int(os.environ.get("SEED","7")),
                       "max_tokens":int(os.environ.get("MAXTOK","48000"))}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"Content-Type":"application/json"})
    ch = json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]
    return ch["message"], ch.get("finish_reason")

def extract_code(t):
    b = re.findall(r"```(?:rust)?\s*\n(.*?)```", t, re.S)
    return max(b, key=len) if b else t

def strip_main(code):
    i = code.find("fn main")
    if i == -1: return code
    b = code.find("{", i); d=0; j=b
    while j < len(code):
        if code[j]=="{": d+=1
        elif code[j]=="}":
            d-=1
            if d==0: break
        j+=1
    return code[:i] + code[j+1:]

for _ in range(180):
    try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3); break
    except Exception: time.sleep(1)

messages=[{"role":"system","content":SYS},{"role":"user","content":TASK}]
ok=False
for it in range(1, MAXIT+1):
    msg,finish=chat(messages); content=msg.get("content") or ""
    # vLLM nightly exposes chain-of-thought as `reasoning`; llama.cpp/deepseek as `reasoning_content`.
    rlen=len(msg.get("reasoning") or msg.get("reasoning_content") or "")
    if finish=="length" and not content.strip():
        print(f"[{LABEL}] iter {it}: think={rlen}c  finish=length, NO code emitted "
              f"(reasoning consumed the whole budget) → raise MAXTOK")
        messages.append({"role":"assistant","content":"(ran out of token budget while reasoning)"})
        messages.append({"role":"user","content":"You exhausted the output budget while thinking and "
                         "produced no code. Reason more concisely, then output the COMPLETE program in "
                         "one ```rust block."})
        continue
    code=strip_main(extract_code(content))
    path=f"{SCR}/loop_{LABEL}_{it}.rs"; open(path,"w").write(code+"\n"+TEST_MAIN)
    comp=subprocess.run(["rustc","-O",path,"-o",f"{SCR}/loop_{LABEL}_{it}.bin"],capture_output=True,text=True)
    nerr=comp.stderr.count("error[")+comp.stderr.count("error:")
    if comp.returncode==0:
        try:
            run=subprocess.run([f"{SCR}/loop_{LABEL}_{it}.bin"],capture_output=True,text=True,timeout=10)
            if SUCCESS in run.stdout:
                print(f"[{LABEL}] iter {it}: think={rlen}c  COMPILES + PASSES ALL TESTS  ✅"); ok=True; break
            fb=(f"It compiled but a test assertion failed at runtime.\nstdout: {run.stdout!r}\n"
                f"stderr: {run.stderr[-400:]!r}\nReturn the COMPLETE corrected program in one ```rust block.")
            print(f"[{LABEL}] iter {it}: think={rlen}c  compiled but TEST FAILED")
        except subprocess.TimeoutExpired:
            fb="It compiled but hung (infinite loop). Return the COMPLETE corrected program."
            print(f"[{LABEL}] iter {it}: think={rlen}c  compiled but HUNG")
    else:
        el="\n".join(l for l in comp.stderr.splitlines() if l.startswith("error") or l.strip().startswith("-->"))[:2500]
        print(f"[{LABEL}] iter {it}: think={rlen}c  {nerr} COMPILE ERRORS")
        hint=""
        if "found macro" in comp.stderr or "cannot find function" in comp.stderr:
            hint=("NOTE: the test harness calls your entry point directly by name. Define it EXACTLY "
                  "as specified at module top level (e.g. `pub fn matches(pattern: &str, text: &str) -> bool`); "
                  "if that exact function is missing, the name resolves to a std macro and every call fails. "
                  "Errors that reference a `fn main`/assert you didn't write are from the harness — fix them by "
                  "providing the exact required function.\n\n")
        fb=f"{hint}Your code failed to compile. rustc errors:\n{el}\n\nReason about the root cause and return the COMPLETE corrected program in one ```rust block."
    messages.append({"role":"assistant","content":content}); messages.append({"role":"user","content":fb})

print(f"[{LABEL}] RESULT: {'CONVERGED' if ok else 'did NOT converge'} in <= {MAXIT} iters")
