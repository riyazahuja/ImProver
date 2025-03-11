import scripts.state_comments
import Cli

open Cli

/-- Setting up command line options and help text for `lake exe state_comments` in standalone_state_comments.lean -/
def state_comments : Cmd := `[Cli|
  state_comments VIA printAnnotations; ["0.0.1"]
"Modify a Lean file by inserting comments after every tactic invocation showing the goal.
Prints the modified source code to stdout."

  FLAGS:
    "indent";  "Indent the state comments to the column of their corresponding tactic."

  ARGS:
    module : ModuleName; "Lean module to compile and annotate with state comments."
]


-- /-- `lake exe state_comments` -/
def main (args : List String) : IO UInt32 :=
  state_comments.validate args
