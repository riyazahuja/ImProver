import metrics.conjecturer.conjecturer
import metrics.completion.completion
import metrics.dependency.dependency
import metrics.declarativity.declarativity
import metrics.length.length
open Lean Elab IO

def route_metric (name : String) (cs : CompilationStep) : IO Float := match name with
| "length" => length_score cs
| "declarativity" => declarativity_score cs
| "dependency" => dependency_score cs
| "completion" => completion_score cs
| "conjecturer" => conjecturer_score cs
| _ => pure 0.0
