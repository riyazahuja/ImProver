import subprocess
import os

cwd = os.getcwd()
subprocess.run(
    [f"repl/.lake/build/bin/repl < /Users/ahuja/Desktop/ImProver2/tmpl80v5bsy.in"],
    shell=True,
    cwd=cwd,
)
