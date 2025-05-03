lemma finite_length_eq : {l : List α | l.length = n}.Finite := List.Vector.finite


lemma finite_length_lt : {l : List α | l.length < n}.Finite := by
  /-
    α : Type u_1
    inst✝ : Finite α
    n : Nat
    ⊢ (setOf fun l => LT.lt l.length n).Finite
  -/
  convert (Finset.range n).finite_toSet.biUnion fun i _ ↦ finite_length_eq α i; ext; simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


lemma finite_length_le : {l : List α | l.length ≤ n}.Finite := by
  /-
    α : Type u_1
    inst✝ : Finite α
    n : Nat
    ⊢ (setOf fun l => LE.le l.length n).Finite
  -/
  simpa [Nat.lt_succ_iff] using finite_length_lt α (n + 1)
  /-
    🎉 no goals
  -/


