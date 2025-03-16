instance instStarRing : StarRing (CliffordAlgebra Q) where
  star x := reverse (involute x)
  star_involutive x := by
    /-
      R : Type u_1
      inst✝² : CommRing R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q : QuadraticForm R M
      x : CliffordAlgebra Q
      ⊢ Eq (Star.star (Star.star x)) x
    -/
    simp only [reverse_involute_commute.eq, reverse_reverse, involute_involute]
    /-
      🎉 no goals
    -/
                     /-
                       R : Type u_1
                       inst✝² : CommRing R
                       M : Type u_2
                       inst✝¹ : AddCommGroup M
                       inst✝ : Module R M
                       Q : QuadraticForm R M
                       x y : CliffordAlgebra Q
                       ⊢ Eq (Star.star (HMul.hMul x y)) (HMul.hMul (Star.star y) (Star.star x))
                     -/
  star_mul x y := by simp only [map_mul, reverse.map_mul]
                     /-
                       🎉 no goals
                     -/
                     /-
                       R : Type u_1
                       inst✝² : CommRing R
                       M : Type u_2
                       inst✝¹ : AddCommGroup M
                       inst✝ : Module R M
                       Q : QuadraticForm R M
                       x y : CliffordAlgebra Q
                       ⊢ Eq (Star.star (HAdd.hAdd x y)) (HAdd.hAdd (Star.star x) (Star.star y))
                     -/
  star_add x y := by simp only [map_add]
                     /-
                       🎉 no goals
                     -/


theorem star_def (x : CliffordAlgebra Q) : star x = reverse (involute x) :=
  rfl


theorem star_def' (x : CliffordAlgebra Q) : star x = involute (reverse x) :=
  reverse_involute _


@[simp]
                                                     /-
                                                       R : Type u_1
                                                       inst✝² : CommRing R
                                                       M : Type u_2
                                                       inst✝¹ : AddCommGroup M
                                                       inst✝ : Module R M
                                                       Q : QuadraticForm R M
                                                       m : M
                                                       ⊢ Eq (Star.star ((CliffordAlgebra.ι Q) m)) (Neg.neg ((CliffordAlgebra.ι Q) m))
                                                     -/
theorem star_ι (m : M) : star (ι Q m) = -ι Q m := by rw [star_def, involute_ι, map_neg, reverse_ι]
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- Note that this not match the `star_smul` implied by `StarModule`; it certainly could if we
also conjugated all the scalars, but there appears to be nothing in the literature that advocates
doing this. -/
@[simp]
theorem star_smul (r : R) (x : CliffordAlgebra Q) : star (r • x) = r • star x := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    r : R
    x : CliffordAlgebra Q
    ⊢ Eq (Star.star (HSMul.hSMul r x)) (HSMul.hSMul r (Star.star x))
  -/
  rw [star_def, star_def, map_smul, map_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem star_algebraMap (r : R) :
    star (algebraMap R (CliffordAlgebra Q) r) = algebraMap R (CliffordAlgebra Q) r := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    r : R
    ⊢ Eq (Star.star ((algebraMap R (CliffordAlgebra Q)) r)) ((algebraMap R (Cliffo …
  -/
  rw [star_def, involute.commutes, reverse.commutes]
  /-
    🎉 no goals
  -/


