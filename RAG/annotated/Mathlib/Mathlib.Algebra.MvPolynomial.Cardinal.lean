@[simp]
theorem cardinalMk_eq_max_lift [Nonempty σ] [Nontrivial R] :
    #(MvPolynomial σ R) = max (max (Cardinal.lift.{u} #R) <| Cardinal.lift.{v} #σ) ℵ₀ :=
  (mk_finsupp_lift_of_infinite _ R).trans <| by
    /-
      σ : Type u
      R : Type v
      inst✝² : CommSemiring R
      inst✝¹ : Nonempty σ
      inst✝ : Nontrivial R
      ⊢ Eq (Max.max (Cardinal.lift.{v, u} (Cardinal.mk (Finsupp σ Nat))) (Cardinal.l …
    -/
    rw [mk_finsupp_nat, max_assoc, lift_max, lift_aleph0, max_comm]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_eq_max_lift := cardinalMk_eq_max_lift


@[simp]
theorem cardinalMk_eq_lift [IsEmpty σ] : #(MvPolynomial σ R) = Cardinal.lift.{u} #R :=
  ((isEmptyRingEquiv R σ).toEquiv.trans Equiv.ulift.{u}.symm).cardinal_eq


@[deprecated (since := "2024-11-10")] alias cardinal_mk_eq_lift := cardinalMk_eq_lift


@[nontriviality]
theorem cardinalMk_eq_one [Subsingleton R] : #(MvPolynomial σ R) = 1 := mk_eq_one _


theorem cardinalMk_le_max_lift {σ : Type u} {R : Type v} [CommSemiring R] : #(MvPolynomial σ R) ≤
    max (max (Cardinal.lift.{u} #R) <| Cardinal.lift.{v} #σ) ℵ₀ := by
  /-
    σ : Type u
    R : Type v
    inst✝ : CommSemiring R
    ⊢ LE.le (Cardinal.mk (MvPolynomial σ R)) (Max.max (Max.max (Cardinal.lift.{u,  …
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      σ : Type u
      R : Type v
      inst✝ : CommSemiring R
      h✝ : Subsingleton R
      ⊢ LE.le (Cardinal.mk (MvPolynomial σ R)) (Max.max (Max.max (Cardinal.lift.{u,  …
    -/
  · exact (mk_eq_one _).trans_le (le_max_of_le_right one_le_aleph0)
    /-
      🎉 no goals
    -/
  /-
    case inr
    σ : Type u
    R : Type v
    inst✝ : CommSemiring R
    h✝ : Nontrivial R
    ⊢ LE.le (Cardinal.mk (MvPolynomial σ R)) (Max.max (Max.max (Cardinal.lift.{u,  …
  -/
  cases isEmpty_or_nonempty σ
    /-
      case inr.inl
      σ : Type u
      R : Type v
      inst✝ : CommSemiring R
      h✝¹ : Nontrivial R
      h✝ : IsEmpty σ
      ⊢ LE.le (Cardinal.mk (MvPolynomial σ R)) (Max.max (Max.max (Cardinal.lift.{u,  …
    -/
  · exact cardinalMk_eq_lift.trans_le (le_max_of_le_left <| le_max_left _ _)
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      σ : Type u
      R : Type v
      inst✝ : CommSemiring R
      h✝¹ : Nontrivial R
      h✝ : Nonempty σ
      ⊢ LE.le (Cardinal.mk (MvPolynomial σ R)) (Max.max (Max.max (Cardinal.lift.{u,  …
    -/
  · exact cardinalMk_eq_max_lift.le
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-21")] alias cardinal_lift_mk_le_max := cardinalMk_le_max_lift


theorem cardinalMk_eq_max [Nonempty σ] [Nontrivial R] :
                                                   /-
                                                     σ R : Type u
                                                     inst✝² : CommSemiring R
                                                     inst✝¹ : Nonempty σ
                                                     inst✝ : Nontrivial R
                                                     ⊢ Eq (Cardinal.mk (MvPolynomial σ R)) (Max.max (Max.max (Cardinal.mk R) (Cardi …
                                                   -/
    #(MvPolynomial σ R) = max (max #R #σ) ℵ₀ := by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_eq_max := cardinalMk_eq_max


                                                                   /-
                                                                     σ R : Type u
                                                                     inst✝¹ : CommSemiring R
                                                                     inst✝ : IsEmpty σ
                                                                     ⊢ Eq (Cardinal.mk (MvPolynomial σ R)) (Cardinal.mk R)
                                                                   -/
theorem cardinalMk_eq [IsEmpty σ] : #(MvPolynomial σ R) = #R := by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- The cardinality of the multivariate polynomial ring, `MvPolynomial σ R` is at most the maximum
of `#R`, `#σ` and `ℵ₀` -/
theorem cardinalMk_le_max : #(MvPolynomial σ R) ≤ max (max #R #σ) ℵ₀ :=
                                     /-
                                       σ R : Type u
                                       inst✝ : CommSemiring R
                                       ⊢ LE.le (Max.max (Max.max (Cardinal.lift.{u, u} (Cardinal.mk R)) (Cardinal.lif …
                                     -/
  cardinalMk_le_max_lift.trans <| by rw [lift_id, lift_id]
                                     /-
                                       🎉 no goals
                                     -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_le_max := cardinalMk_le_max


