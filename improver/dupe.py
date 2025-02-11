from pantograph import Server
import time
import re

project_path = "/Users/ahuja/Desktop/mathlib4/"


server = Server(
    imports=[
        "Mathlib.Data.Set.Lattice",
        "Mathlib.Data.Nat.Prime.Basic",
        "Mathlib.Tactic",
    ],
    project_path="/Users/ahuja/Desktop/mathlib4/",
)
# start_time = time.time()
# units = server.tactic_invocations(project_path + "Mathlib/Logic/Basic.lean")
# end_time = time.time()
# print(f"Time taken: {end_time - start_time:.2f} seconds")


# with open(project_path + "Mathlib/Logic/Basic.lean", "rb") as f:
#     content = f.read()
#     text = content[units[10].i_begin : units[10].i_end].decode("utf-8")
#     text = re.sub(r"theorem\s+\w+", "example", text)

content = """variable {α : Type*}
variable (s t u : Set α)
open Set

theorem fact : s ∩ t ∪ s ∩ u ⊆ s ∩ (t ∪ u) := by
  rintro x (⟨xs, xt⟩ | ⟨xs, xu⟩)
  · use xs; left; exact xt
  · use xs; right; exact xu
"""

out = server.load_sorry(content)
print(out)

print("\nServer output:")
for i in out[-1].invocations:
    print(f"[Before]\n{i.before}")
    print(f"[Tactic]\n{i.tactic} (using {i.used_constants})")
    print(f"[After]\n{i.after}")
