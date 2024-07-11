def triangular : ℕ → ℕ
| 0 := 0
| n := triangular (n-1) + n
