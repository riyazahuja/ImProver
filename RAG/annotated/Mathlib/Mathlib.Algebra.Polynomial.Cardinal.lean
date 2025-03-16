@[simp]
theorem cardinalMk_eq_max {R : Type u} [Semiring R] [Nontrivial R] : #(R[X]) = max #R ℵ₀ :=
  (toFinsuppIso R).toEquiv.cardinal_eq.trans <| by
    /-
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      ⊢ Eq (Cardinal.mk (AddMonoidAlgebra R Nat)) (Max.max (Cardinal.mk R) Cardinal. …
    -/
    rw [AddMonoidAlgebra, mk_finsupp_lift_of_infinite, lift_uzero, max_comm]
    /-
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      ⊢ Eq (Max.max (Cardinal.mk R) (Cardinal.lift.{u, 0} (Cardinal.mk Nat))) (Max.m …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_eq_max := cardinalMk_eq_max


theorem cardinalMk_le_max {R : Type u} [Semiring R] : #(R[X]) ≤ max #R ℵ₀ := by
  /-
    R : Type u
    inst✝ : Semiring R
    ⊢ LE.le (Cardinal.mk (Polynomial R)) (Max.max (Cardinal.mk R) Cardinal.aleph0)
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u
      inst✝ : Semiring R
      h✝ : Subsingleton R
      ⊢ LE.le (Cardinal.mk (Polynomial R)) (Max.max (Cardinal.mk R) Cardinal.aleph0)
    -/
  · exact (mk_eq_one _).trans_le (le_max_of_le_right one_le_aleph0)
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      inst✝ : Semiring R
      h✝ : Nontrivial R
      ⊢ LE.le (Cardinal.mk (Polynomial R)) (Max.max (Cardinal.mk R) Cardinal.aleph0)
    -/
  · exact cardinalMk_eq_max.le
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_le_max := cardinalMk_le_max


