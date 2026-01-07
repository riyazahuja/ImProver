
def gcd (n k : Nat) : Nat :=
  if n = 0 then k
  else if k = 0 then n
  else if n > k then
    gcd (n - k) k
  else
    gcd n (k - n)
termination_by n + k
decreasing_by
  all_goals simp_wf; omega
