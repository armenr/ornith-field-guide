#!/usr/bin/env python3
# Code-correctness battery: run the self-fix loop across SEEDS x PROBLEMS on one server, tally
# convergence rate + avg rounds-to-pass. Budget-safe problems (eval, trie) so the verbose-reasoner
# tax doesn't confound correctness. Args: PORT MODEL LABEL "prob1,prob2" "seed1,seed2,..." [MAXIT]
import json, re, subprocess, sys, urllib.request, time, os
PORT, MODEL, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
PROBS = sys.argv[4].split(","); SEEDS=[int(s) for s in sys.argv[5].split(",")]
MAXIT = int(sys.argv[6]) if len(sys.argv)>6 else 6
SCR="/home/v3ct0r/.claude/jobs/d566939e/tmp/battery_runs"; os.makedirs(SCR, exist_ok=True)
MAXTOK=int(os.environ.get("MAXTOK","24000"))

PROBLEMS={
 "eval": (("Implement an arithmetic expression evaluator in Rust. Exactly this API: "
   "fn eval(expr: &str) -> Result<i64, String>. Support integer literals, the binary operators "
   "+ - * / with standard precedence, parentheses, unary minus, and arbitrary whitespace. Use "
   "truncating integer division. Return Err for malformed input or division by zero. Provide the "
   "full program in one ```rust code block."),
  r'''
fn main(){assert_eq!(eval("2+3*4"),Ok(14));assert_eq!(eval("(2+3)*4"),Ok(20));assert_eq!(eval("10/2-3"),Ok(2));
assert_eq!(eval(" -(3 + 4) "),Ok(-7));assert_eq!(eval("2*-3"),Ok(-6));assert_eq!(eval("100/(2*5)"),Ok(10));
assert_eq!(eval("7-2-2"),Ok(3));assert_eq!(eval("2*3+4*5"),Ok(26));assert!(eval("1/0").is_err());
assert!(eval("2 3").is_err());assert!(eval("(1+2").is_err());assert!(eval("").is_err());
println!("ALL EVAL ASSERTIONS PASSED");}''', "ALL EVAL ASSERTIONS PASSED"),
 "trie": (("Implement a trie (prefix tree) in Rust. Exactly this API: struct Trie with "
   "fn new() -> Self, fn insert(&mut self, word: &str), fn search(&self, word: &str) -> bool, and "
   "fn starts_with(&self, prefix: &str) -> bool. search returns true only for whole inserted words; "
   "starts_with returns true if any inserted word has the prefix. Provide the full program in one ```rust code block."),
  r'''
fn main(){let mut t=Trie::new();t.insert("apple");assert_eq!(t.search("apple"),true);assert_eq!(t.search("app"),false);
assert_eq!(t.starts_with("app"),true);assert_eq!(t.search("appl"),false);assert_eq!(t.starts_with("apple"),true);
assert_eq!(t.starts_with("apx"),false);t.insert("app");assert_eq!(t.search("app"),true);t.insert("banana");
assert_eq!(t.search("banana"),true);assert_eq!(t.starts_with("ban"),true);assert_eq!(t.search("ban"),false);
assert_eq!(t.search("bananas"),false);println!("ALL TRIE ASSERTIONS PASSED");}''', "ALL TRIE ASSERTIONS PASSED"),
}
SYS=("You are a senior Rust engineer. Provide one complete, correct, idiomatic, compiling solution in a "
     "single ```rust code block. When given compiler errors or failing tests, reason about the root cause "
     "and return the FULL corrected program.")

def chat(msgs, seed):
    body=json.dumps({"model":MODEL,"messages":msgs,"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0,
                     "seed":seed,"max_tokens":MAXTOK}).encode()
    req=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",body,{"Content-Type":"application/json"})
    ch=json.load(urllib.request.urlopen(req,timeout=1200))["choices"][0]
    return ch["message"].get("content") or "", ch.get("finish_reason")

def extract(t):
    b=re.findall(r"```(?:rust)?\s*\n(.*?)```",t,re.S); return max(b,key=len) if b else t
def strip_main(code):
    i=code.find("fn main")
    if i==-1: return code
    b=code.find("{",i); d=0; j=b
    while j<len(code):
        if code[j]=="{":d+=1
        elif code[j]=="}":
            d-=1
            if d==0:break
        j+=1
    return code[:i]+code[j+1:]

def solve(prob, seed):
    task,test,success=PROBLEMS[prob]
    msgs=[{"role":"system","content":SYS},{"role":"user","content":task}]
    for it in range(1,MAXIT+1):
        content,finish=chat(msgs,seed)
        if finish=="length" and not content.strip(): return (False,it,"budget")
        code=strip_main(extract(content))
        p=f"{SCR}/{LABEL}_{prob}_{seed}_{it}.rs"; open(p,"w").write(code+"\n"+test)
        comp=subprocess.run(["rustc","-O",p,"-o",f"{SCR}/b.bin"],capture_output=True,text=True)
        if comp.returncode==0:
            run=subprocess.run([f"{SCR}/b.bin"],capture_output=True,text=True,timeout=10)
            if success in run.stdout: return (True,it,"pass")
            fb=f"Compiled but a test failed.\nstdout:{run.stdout!r}\nReturn the COMPLETE corrected program in one ```rust block."
        else:
            el="\n".join(l for l in comp.stderr.splitlines() if l.startswith("error") or l.strip().startswith("-->"))[:2000]
            fb=f"Failed to compile. rustc errors:\n{el}\nReason about the root cause; return the COMPLETE corrected program in one ```rust block."
        msgs.append({"role":"assistant","content":content}); msgs.append({"role":"user","content":fb})
    return (False,MAXIT,"no-converge")

results={}
for prob in PROBS:
    rows=[]
    for s in SEEDS:
        ok,rounds,why=solve(prob,s)
        rows.append((s,ok,rounds,why))
        print(f"[{LABEL}] {prob} seed={s}: {'PASS' if ok else 'FAIL'} rounds={rounds} ({why})"); sys.stdout.flush()
    npass=sum(1 for r in rows if r[1]); n=len(rows)
    avg=sum(r[2] for r in rows if r[1])/npass if npass else 0
    print(f"[{LABEL}] {prob} SUMMARY: converged {npass}/{n}  avg_rounds_when_pass={avg:.1f}"); sys.stdout.flush()
    results[prob]=rows
print(f"[{LABEL}] BATTERY DONE")
