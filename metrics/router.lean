import metrics.length.length

def route_metric (name : String) (cs : CompilationStep) : IO Float := match name with
| "length" => length_score cs
| _ => pure 0.0
