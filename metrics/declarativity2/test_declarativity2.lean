import metrics.declarativity2.declarativity2

def main : IO Unit := do
  IO.println "Testing Declarativity2 Metric"
  IO.println "=============================="

  -- Test repeat_main_goal2 (exact duplicate: should be (0.0, 0.0, true))
  IO.println "\n=== Testing repeat_main_goal2 ==="
  let result2 ← getScore2 `FLT.Mathlib.GroupTheory.Index `AddSubgroup.index_smul repeat_main_goal2
  IO.println s!"Result: {result2}"

  -- Test repeat_main_goal3 (forall-abstraction duplicate: should be (0.0, 0.0, true))
  IO.println "\n=== Testing repeat_main_goal3 ==="
  let result3 ← getScore2 `FLT.Mathlib.GroupTheory.Index `AddSubgroup.index_smul repeat_main_goal3
  IO.println s!"Result: {result3}"

  IO.println "\n=== Tests Complete ==="
