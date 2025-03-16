instance : WellFoundedRelation (WithTop ℕ) where
  rel := (· < ·)
  wf := IsWellFounded.wf


theorem Nat.cast_withTop (n : ℕ) :  Nat.cast n = WithTop.some n :=
  rfl


theorem Nat.cast_withBot (n : ℕ) : Nat.cast n = WithBot.some n :=
  rfl

