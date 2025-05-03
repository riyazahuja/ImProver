noncomputable instance : Inv (FractionalIdeal R₁⁰ K) := ⟨fun I => 1 / I⟩


theorem inv_eq : I⁻¹ = 1 / I := rfl


theorem inv_zero' : (0 : FractionalIdeal R₁⁰ K)⁻¹ = 0 := div_zero


theorem inv_nonzero {J : FractionalIdeal R₁⁰ K} (h : J ≠ 0) :
    J⁻¹ = ⟨(1 : FractionalIdeal R₁⁰ K) / J, fractional_div_of_nonzero h⟩ := div_nonzero h


theorem coe_inv_of_nonzero {J : FractionalIdeal R₁⁰ K} (h : J ≠ 0) :
    (↑J⁻¹ : Submodule R₁ K) = IsLocalization.coeSubmodule K ⊤ / (J : Submodule R₁ K) := by
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Ne J 0
    ⊢ Eq (↑(Inv.inv J)) (HDiv.hDiv (IsLocalization.coeSubmodule K Top.top) ↑J)
  -/
  simp_rw [inv_nonzero _ h, coe_one, coe_mk, IsLocalization.coeSubmodule_top]
  /-
    🎉 no goals
  -/


theorem mem_inv_iff (hI : I ≠ 0) {x : K} : x ∈ I⁻¹ ↔ ∀ y ∈ I, x * y ∈ (1 : FractionalIdeal R₁⁰ K) :=
  mem_div_iff_of_nonzero hI


theorem inv_anti_mono (hI : I ≠ 0) (hJ : J ≠ 0) (hIJ : I ≤ J) : J⁻¹ ≤ I⁻¹ := by
  -- Porting note: in Lean3, introducing `x` would just give `x ∈ J⁻¹ → x ∈ I⁻¹`, but
  --  in Lean4, it goes all the way down to the subtypes
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    hI : Ne I 0
    hJ : Ne J 0
    hIJ : LE.le I J
    ⊢ LE.le (Inv.inv J) (Inv.inv I)
  -/
  intro x
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    hI : Ne I 0
    hJ : Ne J 0
    hIJ : LE.le I J
    x : K
    ⊢ Membership.mem ((fun a => ↑a) (Inv.inv J)) x → Membership.mem ((fun a => ↑a) …
  -/
  simp only [val_eq_coe, mem_coe, mem_inv_iff hJ, mem_inv_iff hI]
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    hI : Ne I 0
    hJ : Ne J 0
    hIJ : LE.le I J
    x : K
    ⊢ (∀ (y : K), Membership.mem J y → Membership.mem 1 (HMul.hMul x y)) → ∀ (y :  …
  -/
  exact fun h y hy => h y (hIJ hy)
  /-
    🎉 no goals
  -/


theorem le_self_mul_inv {I : FractionalIdeal R₁⁰ K} (hI : I ≤ (1 : FractionalIdeal R₁⁰ K)) :
    I ≤ I * I⁻¹ :=
  le_self_mul_one_div hI


theorem coe_ideal_le_self_mul_inv (I : Ideal R₁) :
    (I : FractionalIdeal R₁⁰ K) ≤ I * (I : FractionalIdeal R₁⁰ K)⁻¹ :=
  le_self_mul_inv coeIdeal_le_one


/-- `I⁻¹` is the inverse of `I` if `I` has an inverse. -/
theorem right_inverse_eq (I J : FractionalIdeal R₁⁰ K) (h : I * J = 1) : J = I⁻¹ := by
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Eq (HMul.hMul I J) 1
    ⊢ Eq J (Inv.inv I)
  -/
  have hI : I ≠ 0 := ne_zero_of_mul_eq_one I J h
  suffices h' : I * (1 / I) = 1 from
    congr_arg Units.inv <| @Units.ext _ _ (Units.mkOfMulEqOne _ _ h) (Units.mkOfMulEqOne _ _ h') rfl
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Eq (HMul.hMul I J) 1
    hI : Ne I 0
    ⊢ Eq (HMul.hMul I (HDiv.hDiv 1 I)) 1
  -/
  apply le_antisymm
    /-
      case a
      K : Type u_3
      inst✝⁴ : Field K
      R₁ : Type u_4
      inst✝³ : CommRing R₁
      inst✝² : IsDomain R₁
      inst✝¹ : Algebra R₁ K
      inst✝ : IsFractionRing R₁ K
      I J : FractionalIdeal (nonZeroDivisors R₁) K
      h : Eq (HMul.hMul I J) 1
      hI : Ne I 0
      ⊢ LE.le (HMul.hMul I (HDiv.hDiv 1 I)) 1
    -/
  · apply mul_le.mpr _
    /-
      K : Type u_3
      inst✝⁴ : Field K
      R₁ : Type u_4
      inst✝³ : CommRing R₁
      inst✝² : IsDomain R₁
      inst✝¹ : Algebra R₁ K
      inst✝ : IsFractionRing R₁ K
      I J : FractionalIdeal (nonZeroDivisors R₁) K
      h : Eq (HMul.hMul I J) 1
      hI : Ne I 0
      ⊢ ∀ (i : K), Membership.mem I i → ∀ (j : K), Membership.mem (HDiv.hDiv 1 I) j  …
    -/
    intro x hx y hy
    /-
      K : Type u_3
      inst✝⁴ : Field K
      R₁ : Type u_4
      inst✝³ : CommRing R₁
      inst✝² : IsDomain R₁
      inst✝¹ : Algebra R₁ K
      inst✝ : IsFractionRing R₁ K
      I J : FractionalIdeal (nonZeroDivisors R₁) K
      h : Eq (HMul.hMul I J) 1
      hI : Ne I 0
      x : K
      hx : Membership.mem I x
      y : K
      hy : Membership.mem (HDiv.hDiv 1 I) y
      ⊢ Membership.mem 1 (HMul.hMul x y)
    -/
    rw [mul_comm]
    /-
      K : Type u_3
      inst✝⁴ : Field K
      R₁ : Type u_4
      inst✝³ : CommRing R₁
      inst✝² : IsDomain R₁
      inst✝¹ : Algebra R₁ K
      inst✝ : IsFractionRing R₁ K
      I J : FractionalIdeal (nonZeroDivisors R₁) K
      h : Eq (HMul.hMul I J) 1
      hI : Ne I 0
      x : K
      hx : Membership.mem I x
      y : K
      hy : Membership.mem (HDiv.hDiv 1 I) y
      ⊢ Membership.mem 1 (HMul.hMul y x)
    -/
    exact (mem_div_iff_of_nonzero hI).mp hy x hx
    /-
      🎉 no goals
    -/
  /-
    case a
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Eq (HMul.hMul I J) 1
    hI : Ne I 0
    ⊢ LE.le 1 (HMul.hMul I (HDiv.hDiv 1 I))
  -/
  rw [← h]
  /-
    case a
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Eq (HMul.hMul I J) 1
    hI : Ne I 0
    ⊢ LE.le (HMul.hMul I J) (HMul.hMul I (HDiv.hDiv (HMul.hMul I J) I))
  -/
  apply mul_left_mono I
  /-
    case a.a
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Eq (HMul.hMul I J) 1
    hI : Ne I 0
    ⊢ LE.le J (HDiv.hDiv (HMul.hMul I J) I)
  -/
  apply (le_div_iff_of_nonzero hI).mpr _
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Eq (HMul.hMul I J) 1
    hI : Ne I 0
    ⊢ ∀ (x : K), Membership.mem J x → ∀ (y : K), Membership.mem I y → Membership.m …
  -/
  intro y hy x hx
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Eq (HMul.hMul I J) 1
    hI : Ne I 0
    y : K
    hy : Membership.mem J y
    x : K
    hx : Membership.mem I x
    ⊢ Membership.mem (HMul.hMul I J) (HMul.hMul y x)
  -/
  rw [mul_comm]
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I J : FractionalIdeal (nonZeroDivisors R₁) K
    h : Eq (HMul.hMul I J) 1
    hI : Ne I 0
    y : K
    hy : Membership.mem J y
    x : K
    hx : Membership.mem I x
    ⊢ Membership.mem (HMul.hMul J I) (HMul.hMul y x)
  -/
  exact mul_mem_mul hy hx
  /-
    🎉 no goals
  -/


theorem mul_inv_cancel_iff {I : FractionalIdeal R₁⁰ K} : I * I⁻¹ = 1 ↔ ∃ J, I * J = 1 :=
                                        /-
                                          K : Type u_3
                                          inst✝⁴ : Field K
                                          R₁ : Type u_4
                                          inst✝³ : CommRing R₁
                                          inst✝² : IsDomain R₁
                                          inst✝¹ : Algebra R₁ K
                                          inst✝ : IsFractionRing R₁ K
                                          I : FractionalIdeal (nonZeroDivisors R₁) K
                                          x✝ : Exists fun J => Eq (HMul.hMul I J) 1
                                          J : FractionalIdeal (nonZeroDivisors R₁) K
                                          hJ : Eq (HMul.hMul I J) 1
                                          ⊢ Eq (HMul.hMul I (Inv.inv I)) 1
                                        -/
  ⟨fun h => ⟨I⁻¹, h⟩, fun ⟨J, hJ⟩ => by rwa [← right_inverse_eq K I J hJ]⟩
                                        /-
                                          🎉 no goals
                                        -/


theorem mul_inv_cancel_iff_isUnit {I : FractionalIdeal R₁⁰ K} : I * I⁻¹ = 1 ↔ IsUnit I :=
  (mul_inv_cancel_iff K).trans isUnit_iff_exists_inv.symm


@[simp]
theorem map_inv (I : FractionalIdeal R₁⁰ K) (h : K ≃ₐ[R₁] K') :
                                                  /-
                                                    K : Type u_3
                                                    inst✝⁷ : Field K
                                                    R₁ : Type u_4
                                                    inst✝⁶ : CommRing R₁
                                                    inst✝⁵ : IsDomain R₁
                                                    inst✝⁴ : Algebra R₁ K
                                                    inst✝³ : IsFractionRing R₁ K
                                                    K' : Type u_5
                                                    inst✝² : Field K'
                                                    inst✝¹ : Algebra R₁ K'
                                                    inst✝ : IsFractionRing R₁ K'
                                                    I : FractionalIdeal (nonZeroDivisors R₁) K
                                                    h : AlgEquiv R₁ K K'
                                                    ⊢ Eq (FractionalIdeal.map (↑h) (Inv.inv I)) (Inv.inv (FractionalIdeal.map (↑h) …
                                                  -/
    I⁻¹.map (h : K →ₐ[R₁] K') = (I.map h)⁻¹ := by rw [inv_eq, map_div, map_one, inv_eq]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem spanSingleton_inv (x : K) : (spanSingleton R₁⁰ x)⁻¹ = spanSingleton _ x⁻¹ :=
  one_div_spanSingleton x


theorem spanSingleton_div_spanSingleton (x y : K) :
    spanSingleton R₁⁰ x / spanSingleton R₁⁰ y = spanSingleton R₁⁰ (x / y) := by
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    x y : K
    ⊢ Eq (HDiv.hDiv (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) x) (Fracti …
  -/
  rw [div_spanSingleton, mul_comm, spanSingleton_mul_spanSingleton, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


theorem spanSingleton_div_self {x : K} (hx : x ≠ 0) :
    spanSingleton R₁⁰ x / spanSingleton R₁⁰ x = 1 := by
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    x : K
    hx : Ne x 0
    ⊢ Eq (HDiv.hDiv (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) x) (Fracti …
  -/
  rw [spanSingleton_div_spanSingleton, div_self hx, spanSingleton_one]
  /-
    🎉 no goals
  -/


theorem coe_ideal_span_singleton_div_self {x : R₁} (hx : x ≠ 0) :
    (Ideal.span ({x} : Set R₁) : FractionalIdeal R₁⁰ K) / Ideal.span ({x} : Set R₁) = 1 := by
  rw [coeIdeal_span_singleton,
    spanSingleton_div_self K <|
      (map_ne_zero_iff _ <| NoZeroSMulDivisors.algebraMap_injective R₁ K).mpr hx]


theorem spanSingleton_mul_inv {x : K} (hx : x ≠ 0) :
    spanSingleton R₁⁰ x * (spanSingleton R₁⁰ x)⁻¹ = 1 := by
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    x : K
    hx : Ne x 0
    ⊢ Eq (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) x) (Inv.in …
  -/
  rw [spanSingleton_inv, spanSingleton_mul_spanSingleton, mul_inv_cancel₀ hx, spanSingleton_one]
  /-
    🎉 no goals
  -/


theorem coe_ideal_span_singleton_mul_inv {x : R₁} (hx : x ≠ 0) :
    (Ideal.span ({x} : Set R₁) : FractionalIdeal R₁⁰ K) *
    (Ideal.span ({x} : Set R₁) : FractionalIdeal R₁⁰ K)⁻¹ = 1 := by
  rw [coeIdeal_span_singleton,
    spanSingleton_mul_inv K <|
      (map_ne_zero_iff _ <| NoZeroSMulDivisors.algebraMap_injective R₁ K).mpr hx]


theorem spanSingleton_inv_mul {x : K} (hx : x ≠ 0) :
    (spanSingleton R₁⁰ x)⁻¹ * spanSingleton R₁⁰ x = 1 := by
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    x : K
    hx : Ne x 0
    ⊢ Eq (HMul.hMul (Inv.inv (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) x …
  -/
  rw [mul_comm, spanSingleton_mul_inv K hx]
  /-
    🎉 no goals
  -/


theorem coe_ideal_span_singleton_inv_mul {x : R₁} (hx : x ≠ 0) :
    (Ideal.span ({x} : Set R₁) : FractionalIdeal R₁⁰ K)⁻¹ * Ideal.span ({x} : Set R₁) = 1 := by
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    x : R₁
    hx : Ne x 0
    ⊢ Eq (HMul.hMul (Inv.inv ↑(Ideal.span (Singleton.singleton x))) ↑(Ideal.span ( …
  -/
  rw [mul_comm, coe_ideal_span_singleton_mul_inv K hx]
  /-
    🎉 no goals
  -/


theorem mul_generator_self_inv {R₁ : Type*} [CommRing R₁] [Algebra R₁ K] [IsLocalization R₁⁰ K]
    (I : FractionalIdeal R₁⁰ K) [Submodule.IsPrincipal (I : Submodule R₁ K)] (h : I ≠ 0) :
    I * spanSingleton _ (generator (I : Submodule R₁ K))⁻¹ = 1 := by
  -- Rewrite only the `I` that appears alone.
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_6
    inst✝³ : CommRing R₁
    inst✝² : Algebra R₁ K
    inst✝¹ : IsLocalization (nonZeroDivisors R₁) K
    I : FractionalIdeal (nonZeroDivisors R₁) K
    inst✝ : (↑I).IsPrincipal
    h : Ne I 0
    ⊢ Eq (HMul.hMul I (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (Inv.inv …
  -/
  conv_lhs => congr; rw [eq_spanSingleton_of_principal I]
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_6
    inst✝³ : CommRing R₁
    inst✝² : Algebra R₁ K
    inst✝¹ : IsLocalization (nonZeroDivisors R₁) K
    I : FractionalIdeal (nonZeroDivisors R₁) K
    inst✝ : (↑I).IsPrincipal
    h : Ne I 0
    ⊢ Eq (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (Submodule …
  -/
  rw [spanSingleton_mul_spanSingleton, mul_inv_cancel₀, spanSingleton_one]
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_6
    inst✝³ : CommRing R₁
    inst✝² : Algebra R₁ K
    inst✝¹ : IsLocalization (nonZeroDivisors R₁) K
    I : FractionalIdeal (nonZeroDivisors R₁) K
    inst✝ : (↑I).IsPrincipal
    h : Ne I 0
    ⊢ Ne (Submodule.IsPrincipal.generator ↑I) 0
  -/
  intro generator_I_eq_zero
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_6
    inst✝³ : CommRing R₁
    inst✝² : Algebra R₁ K
    inst✝¹ : IsLocalization (nonZeroDivisors R₁) K
    I : FractionalIdeal (nonZeroDivisors R₁) K
    inst✝ : (↑I).IsPrincipal
    h : Ne I 0
    generator_I_eq_zero : Eq (Submodule.IsPrincipal.generator ↑I) 0
    ⊢ False
  -/
  apply h
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_6
    inst✝³ : CommRing R₁
    inst✝² : Algebra R₁ K
    inst✝¹ : IsLocalization (nonZeroDivisors R₁) K
    I : FractionalIdeal (nonZeroDivisors R₁) K
    inst✝ : (↑I).IsPrincipal
    h : Ne I 0
    generator_I_eq_zero : Eq (Submodule.IsPrincipal.generator ↑I) 0
    ⊢ Eq I 0
  -/
  rw [eq_spanSingleton_of_principal I, generator_I_eq_zero, spanSingleton_zero]
  /-
    🎉 no goals
  -/


theorem invertible_of_principal (I : FractionalIdeal R₁⁰ K)
    [Submodule.IsPrincipal (I : Submodule R₁ K)] (h : I ≠ 0) : I * I⁻¹ = 1 :=
  mul_div_self_cancel_iff.mpr
    ⟨spanSingleton _ (generator (I : Submodule R₁ K))⁻¹, mul_generator_self_inv _ I h⟩


theorem invertible_iff_generator_nonzero (I : FractionalIdeal R₁⁰ K)
    [Submodule.IsPrincipal (I : Submodule R₁ K)] :
    I * I⁻¹ = 1 ↔ generator (I : Submodule R₁ K) ≠ 0 := by
  /-
    K : Type u_3
    inst✝⁵ : Field K
    R₁ : Type u_4
    inst✝⁴ : CommRing R₁
    inst✝³ : IsDomain R₁
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    I : FractionalIdeal (nonZeroDivisors R₁) K
    inst✝ : (↑I).IsPrincipal
    ⊢ Iff (Eq (HMul.hMul I (Inv.inv I)) 1) (Ne (Submodule.IsPrincipal.generator ↑I …
  -/
  constructor
    /-
      case mp
      K : Type u_3
      inst✝⁵ : Field K
      R₁ : Type u_4
      inst✝⁴ : CommRing R₁
      inst✝³ : IsDomain R₁
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      I : FractionalIdeal (nonZeroDivisors R₁) K
      inst✝ : (↑I).IsPrincipal
      ⊢ Eq (HMul.hMul I (Inv.inv I)) 1 → Ne (Submodule.IsPrincipal.generator ↑I) 0
    -/
  · intro hI hg
    /-
      case mp
      K : Type u_3
      inst✝⁵ : Field K
      R₁ : Type u_4
      inst✝⁴ : CommRing R₁
      inst✝³ : IsDomain R₁
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      I : FractionalIdeal (nonZeroDivisors R₁) K
      inst✝ : (↑I).IsPrincipal
      hI : Eq (HMul.hMul I (Inv.inv I)) 1
      hg : Eq (Submodule.IsPrincipal.generator ↑I) 0
      ⊢ False
    -/
    apply ne_zero_of_mul_eq_one _ _ hI
    /-
      case mp
      K : Type u_3
      inst✝⁵ : Field K
      R₁ : Type u_4
      inst✝⁴ : CommRing R₁
      inst✝³ : IsDomain R₁
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      I : FractionalIdeal (nonZeroDivisors R₁) K
      inst✝ : (↑I).IsPrincipal
      hI : Eq (HMul.hMul I (Inv.inv I)) 1
      hg : Eq (Submodule.IsPrincipal.generator ↑I) 0
      ⊢ Eq I 0
    -/
    rw [eq_spanSingleton_of_principal I, hg, spanSingleton_zero]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u_3
      inst✝⁵ : Field K
      R₁ : Type u_4
      inst✝⁴ : CommRing R₁
      inst✝³ : IsDomain R₁
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      I : FractionalIdeal (nonZeroDivisors R₁) K
      inst✝ : (↑I).IsPrincipal
      ⊢ Ne (Submodule.IsPrincipal.generator ↑I) 0 → Eq (HMul.hMul I (Inv.inv I)) 1
    -/
  · intro hg
    /-
      case mpr
      K : Type u_3
      inst✝⁵ : Field K
      R₁ : Type u_4
      inst✝⁴ : CommRing R₁
      inst✝³ : IsDomain R₁
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      I : FractionalIdeal (nonZeroDivisors R₁) K
      inst✝ : (↑I).IsPrincipal
      hg : Ne (Submodule.IsPrincipal.generator ↑I) 0
      ⊢ Eq (HMul.hMul I (Inv.inv I)) 1
    -/
    apply invertible_of_principal
    /-
      case mpr.h
      K : Type u_3
      inst✝⁵ : Field K
      R₁ : Type u_4
      inst✝⁴ : CommRing R₁
      inst✝³ : IsDomain R₁
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      I : FractionalIdeal (nonZeroDivisors R₁) K
      inst✝ : (↑I).IsPrincipal
      hg : Ne (Submodule.IsPrincipal.generator ↑I) 0
      ⊢ Ne I 0
    -/
    rw [eq_spanSingleton_of_principal I]
    /-
      case mpr.h
      K : Type u_3
      inst✝⁵ : Field K
      R₁ : Type u_4
      inst✝⁴ : CommRing R₁
      inst✝³ : IsDomain R₁
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      I : FractionalIdeal (nonZeroDivisors R₁) K
      inst✝ : (↑I).IsPrincipal
      hg : Ne (Submodule.IsPrincipal.generator ↑I) 0
      ⊢ Ne (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (Submodule.IsPrincipa …
    -/
    intro hI
    /-
      case mpr.h
      K : Type u_3
      inst✝⁵ : Field K
      R₁ : Type u_4
      inst✝⁴ : CommRing R₁
      inst✝³ : IsDomain R₁
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      I : FractionalIdeal (nonZeroDivisors R₁) K
      inst✝ : (↑I).IsPrincipal
      hg : Ne (Submodule.IsPrincipal.generator ↑I) 0
      hI : Eq (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (Submodule.IsPrinc …
      ⊢ False
    -/
    have := mem_spanSingleton_self R₁⁰ (generator (I : Submodule R₁ K))
    /-
      case mpr.h
      K : Type u_3
      inst✝⁵ : Field K
      R₁ : Type u_4
      inst✝⁴ : CommRing R₁
      inst✝³ : IsDomain R₁
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      I : FractionalIdeal (nonZeroDivisors R₁) K
      inst✝ : (↑I).IsPrincipal
      hg : Ne (Submodule.IsPrincipal.generator ↑I) 0
      hI : Eq (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (Submodule.IsPrinc …
      this : Membership.mem (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (Sub …
      ⊢ False
    -/
    rw [hI, mem_zero_iff] at this
    /-
      case mpr.h
      K : Type u_3
      inst✝⁵ : Field K
      R₁ : Type u_4
      inst✝⁴ : CommRing R₁
      inst✝³ : IsDomain R₁
      inst✝² : Algebra R₁ K
      inst✝¹ : IsFractionRing R₁ K
      I : FractionalIdeal (nonZeroDivisors R₁) K
      inst✝ : (↑I).IsPrincipal
      hg : Ne (Submodule.IsPrincipal.generator ↑I) 0
      hI : Eq (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (Submodule.IsPrinc …
      this : Eq (Submodule.IsPrincipal.generator ↑I) 0
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/


theorem isPrincipal_inv (I : FractionalIdeal R₁⁰ K) [Submodule.IsPrincipal (I : Submodule R₁ K)]
    (h : I ≠ 0) : Submodule.IsPrincipal I⁻¹.1 := by
  /-
    K : Type u_3
    inst✝⁵ : Field K
    R₁ : Type u_4
    inst✝⁴ : CommRing R₁
    inst✝³ : IsDomain R₁
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    I : FractionalIdeal (nonZeroDivisors R₁) K
    inst✝ : (↑I).IsPrincipal
    h : Ne I 0
    ⊢ (↑(Inv.inv I)).IsPrincipal
  -/
  rw [val_eq_coe, isPrincipal_iff]
  /-
    K : Type u_3
    inst✝⁵ : Field K
    R₁ : Type u_4
    inst✝⁴ : CommRing R₁
    inst✝³ : IsDomain R₁
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    I : FractionalIdeal (nonZeroDivisors R₁) K
    inst✝ : (↑I).IsPrincipal
    h : Ne I 0
    ⊢ Exists fun x => Eq (Inv.inv I) (FractionalIdeal.spanSingleton (nonZeroDiviso …
  -/
  use (generator (I : Submodule R₁ K))⁻¹
  have hI : I * spanSingleton _ (generator (I : Submodule R₁ K))⁻¹ = 1 :=
    mul_generator_self_inv _ I h
  /-
    case h
    K : Type u_3
    inst✝⁵ : Field K
    R₁ : Type u_4
    inst✝⁴ : CommRing R₁
    inst✝³ : IsDomain R₁
    inst✝² : Algebra R₁ K
    inst✝¹ : IsFractionRing R₁ K
    I : FractionalIdeal (nonZeroDivisors R₁) K
    inst✝ : (↑I).IsPrincipal
    h : Ne I 0
    hI : Eq (HMul.hMul I (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (Inv. …
    ⊢ Eq (Inv.inv I) (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) (Inv.inv  …
  -/
  exact (right_inverse_eq _ I (spanSingleton _ (generator (I : Submodule R₁ K))⁻¹) hI).symm
  /-
    🎉 no goals
  -/


lemma den_mem_inv {I : FractionalIdeal R₁⁰ K} (hI : I ≠ ⊥) :
    (algebraMap R₁ K) (I.den : R₁) ∈ I⁻¹ := by
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I : FractionalIdeal (nonZeroDivisors R₁) K
    hI : Ne I Bot.bot
    ⊢ Membership.mem (Inv.inv I) ((algebraMap R₁ K) ↑I.den)
  -/
  rw [mem_inv_iff hI]
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I : FractionalIdeal (nonZeroDivisors R₁) K
    hI : Ne I Bot.bot
    ⊢ ∀ (y : K), Membership.mem I y → Membership.mem 1 (HMul.hMul ((algebraMap R₁  …
  -/
  intro i hi
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I : FractionalIdeal (nonZeroDivisors R₁) K
    hI : Ne I Bot.bot
    i : K
    hi : Membership.mem I i
    ⊢ Membership.mem 1 (HMul.hMul ((algebraMap R₁ K) ↑I.den) i)
  -/
  rw [← Algebra.smul_def (I.den : R₁) i, ← mem_coe, coe_one]
  suffices Submodule.map (Algebra.linearMap R₁ K) I.num ≤ 1 from
    this <| (den_mul_self_eq_num I).symm ▸ smul_mem_pointwise_smul i I.den I.coeToSubmodule hi
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I : FractionalIdeal (nonZeroDivisors R₁) K
    hI : Ne I Bot.bot
    i : K
    hi : Membership.mem I i
    ⊢ LE.le (Submodule.map (Algebra.linearMap R₁ K) I.num) 1
  -/
  apply le_trans <| map_mono (show I.num ≤ 1 by simp only [Ideal.one_eq_top, le_top, bot_eq_zero])
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I : FractionalIdeal (nonZeroDivisors R₁) K
    hI : Ne I Bot.bot
    i : K
    hi : Membership.mem I i
    ⊢ LE.le (Submodule.map (Algebra.linearMap R₁ K) 1) 1
  -/
  rw [Ideal.one_eq_top, Submodule.map_top, one_eq_range]
  /-
    🎉 no goals
  -/


lemma num_le_mul_inv (I : FractionalIdeal R₁⁰ K) : I.num ≤ I * I⁻¹ := by
  /-
    K : Type u_3
    inst✝⁴ : Field K
    R₁ : Type u_4
    inst✝³ : CommRing R₁
    inst✝² : IsDomain R₁
    inst✝¹ : Algebra R₁ K
    inst✝ : IsFractionRing R₁ K
    I : FractionalIdeal (nonZeroDivisors R₁) K
    ⊢ LE.le (↑I.num) (HMul.hMul I (Inv.inv I))
  -/
  by_cases hI : I = 0
  · rw [hI, num_zero_eq <| NoZeroSMulDivisors.algebraMap_injective R₁ K, zero_mul, zero_eq_bot,
      coeIdeal_bot]
    /-
      case neg
      K : Type u_3
      inst✝⁴ : Field K
      R₁ : Type u_4
      inst✝³ : CommRing R₁
      inst✝² : IsDomain R₁
      inst✝¹ : Algebra R₁ K
      inst✝ : IsFractionRing R₁ K
      I : FractionalIdeal (nonZeroDivisors R₁) K
      hI : Not (Eq I 0)
      ⊢ LE.le (↑I.num) (HMul.hMul I (Inv.inv I))
    -/
  · rw [mul_comm, ← den_mul_self_eq_num']
    /-
      case neg
      K : Type u_3
      inst✝⁴ : Field K
      R₁ : Type u_4
      inst✝³ : CommRing R₁
      inst✝² : IsDomain R₁
      inst✝¹ : Algebra R₁ K
      inst✝ : IsFractionRing R₁ K
      I : FractionalIdeal (nonZeroDivisors R₁) K
      hI : Not (Eq I 0)
      ⊢ LE.le (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors R₁) ((algeb …
    -/
    exact mul_right_mono I <| spanSingleton_le_iff_mem.2 (den_mem_inv hI)
    /-
      🎉 no goals
    -/


lemma bot_lt_mul_inv {I : FractionalIdeal R₁⁰ K} (hI : I ≠ ⊥) : ⊥ < I * I⁻¹ :=
  lt_of_lt_of_le (coeIdeal_ne_zero.2 (hI ∘ num_eq_zero_iff.1)).bot_lt I.num_le_mul_inv


noncomputable instance : InvOneClass (FractionalIdeal R₁⁰ K) := { inv_one := div_one }


/-- A Dedekind domain is an integral domain such that every fractional ideal has an inverse.

This is equivalent to `IsDedekindDomain`.
In particular we provide a `fractional_ideal.comm_group_with_zero` instance,
assuming `IsDedekindDomain A`, which implies `IsDedekindDomainInv`. For **integral** ideals,
`IsDedekindDomain`(`_inv`) implies only `Ideal.cancelCommMonoidWithZero`.
-/
def IsDedekindDomainInv : Prop :=
  ∀ I ≠ (⊥ : FractionalIdeal A⁰ (FractionRing A)), I * I⁻¹ = 1


theorem isDedekindDomainInv_iff [Algebra A K] [IsFractionRing A K] :
    IsDedekindDomainInv A ↔ ∀ I ≠ (⊥ : FractionalIdeal A⁰ K), I * I⁻¹ = 1 := by
  let h : FractionalIdeal A⁰ (FractionRing A) ≃+* FractionalIdeal A⁰ K :=
    FractionalIdeal.mapEquiv (FractionRing.algEquiv A K)
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : IsDomain A
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    h : RingEquiv (FractionalIdeal (nonZeroDivisors A) (FractionRing A)) (Fraction …
    ⊢ Iff (IsDedekindDomainInv A) (∀ (I : FractionalIdeal (nonZeroDivisors A) K),  …
  -/
  refine h.toEquiv.forall_congr (fun {x} => ?_)
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : IsDomain A
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    h : RingEquiv (FractionalIdeal (nonZeroDivisors A) (FractionRing A)) (Fraction …
    x : FractionalIdeal (nonZeroDivisors A) (FractionRing A)
    ⊢ Iff (Ne x Bot.bot → Eq (HMul.hMul x (Inv.inv x)) 1) (Ne (h.toEquiv x) Bot.bo …
  -/
  rw [← h.toEquiv.apply_eq_iff_eq]
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : IsDomain A
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    h : RingEquiv (FractionalIdeal (nonZeroDivisors A) (FractionRing A)) (Fraction …
    x : FractionalIdeal (nonZeroDivisors A) (FractionRing A)
    ⊢ Iff (Ne x Bot.bot → Eq (h.toEquiv (HMul.hMul x (Inv.inv x))) (h.toEquiv 1))  …
  -/
  simp [h, IsDedekindDomainInv]
  /-
    🎉 no goals
  -/


theorem FractionalIdeal.adjoinIntegral_eq_one_of_isUnit [Algebra A K] [IsFractionRing A K] (x : K)
    (hx : IsIntegral A x) (hI : IsUnit (adjoinIntegral A⁰ x hx)) : adjoinIntegral A⁰ x hx = 1 := by
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : IsDomain A
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    hx : IsIntegral A x
    hI : IsUnit (FractionalIdeal.adjoinIntegral (nonZeroDivisors A) x hx)
    ⊢ Eq (FractionalIdeal.adjoinIntegral (nonZeroDivisors A) x hx) 1
  -/
  set I := adjoinIntegral A⁰ x hx
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : IsDomain A
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    hx : IsIntegral A x
    I : FractionalIdeal (nonZeroDivisors A) K := FractionalIdeal.adjoinIntegral (n …
    hI : IsUnit I
    ⊢ Eq I 1
  -/
  have mul_self : I * I = I := by apply coeToSubmodule_injective; simp [I]
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : IsDomain A
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    hx : IsIntegral A x
    I : FractionalIdeal (nonZeroDivisors A) K := FractionalIdeal.adjoinIntegral (n …
    hI : IsUnit I
    mul_self : Eq (HMul.hMul I I) I
    ⊢ Eq I 1
  -/
  convert congr_arg (· * I⁻¹) mul_self <;>
    /-
      case h.e'_2
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : IsDomain A
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      x : K
      hx : IsIntegral A x
      I : FractionalIdeal (nonZeroDivisors A) K := FractionalIdeal.adjoinIntegral (n …
      hI : IsUnit I
      mul_self : Eq (HMul.hMul I I) I
      ⊢ Eq I (HMul.hMul (HMul.hMul I I) (Inv.inv I))
    -/
    /-
      🎉 no goals
    -/
    simp only [(mul_inv_cancel_iff_isUnit K).mpr hI, mul_assoc, mul_one]
    /-
      🎉 no goals
    -/


theorem mul_inv_eq_one {I : FractionalIdeal A⁰ K} (hI : I ≠ 0) : I * I⁻¹ = 1 :=
  isDedekindDomainInv_iff.mp h I hI


theorem inv_mul_eq_one {I : FractionalIdeal A⁰ K} (hI : I ≠ 0) : I⁻¹ * I = 1 :=
  (mul_comm _ _).trans (h.mul_inv_eq_one hI)


protected theorem isUnit {I : FractionalIdeal A⁰ K} (hI : I ≠ 0) : IsUnit I :=
  isUnit_of_mul_eq_one _ _ (h.mul_inv_eq_one hI)


theorem isNoetherianRing : IsNoetherianRing A := by
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    ⊢ IsNoetherianRing A
  -/
  refine isNoetherianRing_iff.mpr ⟨fun I : Ideal A => ?_⟩
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    I : Ideal A
    ⊢ Submodule.FG I
  -/
  by_cases hI : I = ⊥
    /-
      case pos
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDomain A
      h : IsDedekindDomainInv A
      I : Ideal A
      hI : Eq I Bot.bot
      ⊢ Submodule.FG I
    -/
  · rw [hI]; apply Submodule.fg_bot
             /-
               🎉 no goals
             -/
  /-
    case neg
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    I : Ideal A
    hI : Not (Eq I Bot.bot)
    ⊢ Submodule.FG I
  -/
  have hI : (I : FractionalIdeal A⁰ (FractionRing A)) ≠ 0 := coeIdeal_ne_zero.mpr hI
  /-
    case neg
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    I : Ideal A
    hI✝ : Not (Eq I Bot.bot)
    hI : Ne (↑I) 0
    ⊢ Submodule.FG I
  -/
  exact I.fg_of_isUnit (IsFractionRing.injective A (FractionRing A)) (h.isUnit hI)
  /-
    🎉 no goals
  -/


theorem integrallyClosed : IsIntegrallyClosed A := by
  -- It suffices to show that for integral `x`,
  -- `A[x]` (which is a fractional ideal) is in fact equal to `A`.
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    ⊢ IsIntegrallyClosed A
  -/
  refine (isIntegrallyClosed_iff (FractionRing A)).mpr (fun {x hx} => ?_)
  rw [← Set.mem_range, ← Algebra.mem_bot, ← Subalgebra.mem_toSubmodule, Algebra.toSubmodule_bot,
    Submodule.one_eq_span, ← coe_spanSingleton A⁰ (1 : FractionRing A), spanSingleton_one, ←
    FractionalIdeal.adjoinIntegral_eq_one_of_isUnit x hx (h.isUnit _)]
    /-
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDomain A
      h : IsDedekindDomainInv A
      x : FractionRing A
      hx : IsIntegral A x
      ⊢ Membership.mem (↑(FractionalIdeal.adjoinIntegral (nonZeroDivisors A) x hx)) x
    -/
  · exact mem_adjoinIntegral_self A⁰ x hx
    /-
      🎉 no goals
    -/
    /-
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDomain A
      h : IsDedekindDomainInv A
      x : FractionRing A
      hx : IsIntegral A x
      ⊢ Ne (FractionalIdeal.adjoinIntegral (nonZeroDivisors A) x hx) 0
    -/
  · exact fun h => one_ne_zero (eq_zero_iff.mp h 1 (Algebra.adjoin A {x}).one_mem)
    /-
      🎉 no goals
    -/


theorem dimensionLEOne : DimensionLEOne A := ⟨by
  -- We're going to show that `P` is maximal because any (maximal) ideal `M`
  -- that is strictly larger would be `⊤`.
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    ⊢ ∀ {p : Ideal A}, Ne p Bot.bot → p.IsPrime → p.IsMaximal
  -/
  rintro P P_ne hP
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    P : Ideal A
    P_ne : Ne P Bot.bot
    hP : P.IsPrime
    ⊢ P.IsMaximal
  -/
  refine Ideal.isMaximal_def.mpr ⟨hP.ne_top, fun M hM => ?_⟩
  -- We may assume `P` and `M` (as fractional ideals) are nonzero.
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    P : Ideal A
    P_ne : Ne P Bot.bot
    hP : P.IsPrime
    M : Ideal A
    hM : LT.lt P M
    ⊢ Eq M Top.top
  -/
  have P'_ne : (P : FractionalIdeal A⁰ (FractionRing A)) ≠ 0 := coeIdeal_ne_zero.mpr P_ne
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    P : Ideal A
    P_ne : Ne P Bot.bot
    hP : P.IsPrime
    M : Ideal A
    hM : LT.lt P M
    P'_ne : Ne (↑P) 0
    ⊢ Eq M Top.top
  -/
  have M'_ne : (M : FractionalIdeal A⁰ (FractionRing A)) ≠ 0 := coeIdeal_ne_zero.mpr hM.ne_bot
  -- In particular, we'll show `M⁻¹ * P ≤ P`
  suffices (M⁻¹ : FractionalIdeal A⁰ (FractionRing A)) * P ≤ P by
    rw [eq_top_iff, ← coeIdeal_le_coeIdeal (FractionRing A), coeIdeal_top]
    calc
      (1 : FractionalIdeal A⁰ (FractionRing A)) = _ * _ * _ := ?_
      _ ≤ _ * _ := mul_right_mono
        ((P : FractionalIdeal A⁰ (FractionRing A))⁻¹ * M : FractionalIdeal A⁰ (FractionRing A)) this
      _ = M := ?_
    · rw [mul_assoc, ← mul_assoc (P : FractionalIdeal A⁰ (FractionRing A)), h.mul_inv_eq_one P'_ne,
      one_mul, h.inv_mul_eq_one M'_ne]
    · rw [← mul_assoc (P : FractionalIdeal A⁰ (FractionRing A)), h.mul_inv_eq_one P'_ne, one_mul]
  -- Suppose we have `x ∈ M⁻¹ * P`, then in fact `x = algebraMap _ _ y` for some `y`.
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    P : Ideal A
    P_ne : Ne P Bot.bot
    hP : P.IsPrime
    M : Ideal A
    hM : LT.lt P M
    P'_ne : Ne (↑P) 0
    M'_ne : Ne (↑M) 0
    ⊢ LE.le (HMul.hMul (Inv.inv ↑M) ↑P) ↑P
  -/
  intro x hx
  have le_one : (M⁻¹ : FractionalIdeal A⁰ (FractionRing A)) * P ≤ 1 := by
    rw [← h.inv_mul_eq_one M'_ne]
    exact mul_left_mono _ ((coeIdeal_le_coeIdeal (FractionRing A)).mpr hM.le)
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    P : Ideal A
    P_ne : Ne P Bot.bot
    hP : P.IsPrime
    M : Ideal A
    hM : LT.lt P M
    P'_ne : Ne (↑P) 0
    M'_ne : Ne (↑M) 0
    x : FractionRing A
    hx : Membership.mem ((fun a => ↑a) (HMul.hMul (Inv.inv ↑M) ↑P)) x
    le_one : LE.le (HMul.hMul (Inv.inv ↑M) ↑P) 1
    ⊢ Membership.mem ((fun a => ↑a) ↑P) x
  -/
  obtain ⟨y, _hy, rfl⟩ := (mem_coeIdeal _).mp (le_one hx)
  -- Since `M` is strictly greater than `P`, let `z ∈ M \ P`.
  /-
    case intro.intro
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    P : Ideal A
    P_ne : Ne P Bot.bot
    hP : P.IsPrime
    M : Ideal A
    hM : LT.lt P M
    P'_ne : Ne (↑P) 0
    M'_ne : Ne (↑M) 0
    le_one : LE.le (HMul.hMul (Inv.inv ↑M) ↑P) 1
    y : A
    _hy : Membership.mem Top.top y
    hx : Membership.mem ((fun a => ↑a) (HMul.hMul (Inv.inv ↑M) ↑P)) ((algebraMap A …
    ⊢ Membership.mem ((fun a => ↑a) ↑P) ((algebraMap A (FractionRing A)) y)
  -/
  obtain ⟨z, hzM, hzp⟩ := SetLike.exists_of_lt hM
  -- We have `z * y ∈ M * (M⁻¹ * P) = P`.
  /-
    case intro.intro.intro.intro
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    P : Ideal A
    P_ne : Ne P Bot.bot
    hP : P.IsPrime
    M : Ideal A
    hM : LT.lt P M
    P'_ne : Ne (↑P) 0
    M'_ne : Ne (↑M) 0
    le_one : LE.le (HMul.hMul (Inv.inv ↑M) ↑P) 1
    y : A
    _hy : Membership.mem Top.top y
    hx : Membership.mem ((fun a => ↑a) (HMul.hMul (Inv.inv ↑M) ↑P)) ((algebraMap A …
    z : A
    hzM : Membership.mem M z
    hzp : Not (Membership.mem P z)
    ⊢ Membership.mem ((fun a => ↑a) ↑P) ((algebraMap A (FractionRing A)) y)
  -/
  have zy_mem := mul_mem_mul (mem_coeIdeal_of_mem A⁰ hzM) hx
  /-
    case intro.intro.intro.intro
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    P : Ideal A
    P_ne : Ne P Bot.bot
    hP : P.IsPrime
    M : Ideal A
    hM : LT.lt P M
    P'_ne : Ne (↑P) 0
    M'_ne : Ne (↑M) 0
    le_one : LE.le (HMul.hMul (Inv.inv ↑M) ↑P) 1
    y : A
    _hy : Membership.mem Top.top y
    hx : Membership.mem ((fun a => ↑a) (HMul.hMul (Inv.inv ↑M) ↑P)) ((algebraMap A …
    z : A
    hzM : Membership.mem M z
    hzp : Not (Membership.mem P z)
    zy_mem : Membership.mem (HMul.hMul (↑M) (HMul.hMul (Inv.inv ↑M) ↑P)) (HMul.hMu …
    ⊢ Membership.mem ((fun a => ↑a) ↑P) ((algebraMap A (FractionRing A)) y)
  -/
  rw [← RingHom.map_mul, ← mul_assoc, h.mul_inv_eq_one M'_ne, one_mul] at zy_mem
  /-
    case intro.intro.intro.intro
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    P : Ideal A
    P_ne : Ne P Bot.bot
    hP : P.IsPrime
    M : Ideal A
    hM : LT.lt P M
    P'_ne : Ne (↑P) 0
    M'_ne : Ne (↑M) 0
    le_one : LE.le (HMul.hMul (Inv.inv ↑M) ↑P) 1
    y : A
    _hy : Membership.mem Top.top y
    hx : Membership.mem ((fun a => ↑a) (HMul.hMul (Inv.inv ↑M) ↑P)) ((algebraMap A …
    z : A
    hzM : Membership.mem M z
    hzp : Not (Membership.mem P z)
    zy_mem : Membership.mem (↑P) ((algebraMap A (FractionRing A)) (HMul.hMul z y))
    ⊢ Membership.mem ((fun a => ↑a) ↑P) ((algebraMap A (FractionRing A)) y)
  -/
  obtain ⟨zy, hzy, zy_eq⟩ := (mem_coeIdeal A⁰).mp zy_mem
  /-
    case intro.intro.intro.intro.intro.intro
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    P : Ideal A
    P_ne : Ne P Bot.bot
    hP : P.IsPrime
    M : Ideal A
    hM : LT.lt P M
    P'_ne : Ne (↑P) 0
    M'_ne : Ne (↑M) 0
    le_one : LE.le (HMul.hMul (Inv.inv ↑M) ↑P) 1
    y : A
    _hy : Membership.mem Top.top y
    hx : Membership.mem ((fun a => ↑a) (HMul.hMul (Inv.inv ↑M) ↑P)) ((algebraMap A …
    z : A
    hzM : Membership.mem M z
    hzp : Not (Membership.mem P z)
    zy_mem : Membership.mem (↑P) ((algebraMap A (FractionRing A)) (HMul.hMul z y))
    zy : A
    hzy : Membership.mem P zy
    zy_eq : Eq ((algebraMap A (FractionRing A)) zy) ((algebraMap A (FractionRing A …
    ⊢ Membership.mem ((fun a => ↑a) ↑P) ((algebraMap A (FractionRing A)) y)
  -/
  rw [IsFractionRing.injective A (FractionRing A) zy_eq] at hzy
  -- But `P` is a prime ideal, so `z ∉ P` implies `y ∈ P`, as desired.
  /-
    case intro.intro.intro.intro.intro.intro
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    h : IsDedekindDomainInv A
    P : Ideal A
    P_ne : Ne P Bot.bot
    hP : P.IsPrime
    M : Ideal A
    hM : LT.lt P M
    P'_ne : Ne (↑P) 0
    M'_ne : Ne (↑M) 0
    le_one : LE.le (HMul.hMul (Inv.inv ↑M) ↑P) 1
    y : A
    _hy : Membership.mem Top.top y
    hx : Membership.mem ((fun a => ↑a) (HMul.hMul (Inv.inv ↑M) ↑P)) ((algebraMap A …
    z : A
    hzM : Membership.mem M z
    hzp : Not (Membership.mem P z)
    zy_mem : Membership.mem (↑P) ((algebraMap A (FractionRing A)) (HMul.hMul z y))
    zy : A
    hzy : Membership.mem P (HMul.hMul z y)
    zy_eq : Eq ((algebraMap A (FractionRing A)) zy) ((algebraMap A (FractionRing A …
    ⊢ Membership.mem ((fun a => ↑a) ↑P) ((algebraMap A (FractionRing A)) y)
  -/
  exact mem_coeIdeal_of_mem A⁰ (Or.resolve_left (hP.mem_or_mem hzy) hzp)⟩
  /-
    🎉 no goals
  -/


/-- Showing one side of the equivalence between the definitions
`IsDedekindDomainInv` and `IsDedekindDomain` of Dedekind domains. -/
theorem isDedekindDomain : IsDedekindDomain A :=
  { h.isNoetherianRing, h.dimensionLEOne, h.integrallyClosed with }


theorem one_mem_inv_coe_ideal [IsDomain A] {I : Ideal A} (hI : I ≠ ⊥) :
    (1 : K) ∈ (I : FractionalIdeal A⁰ K)⁻¹ := by
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDomain A
    I : Ideal A
    hI : Ne I Bot.bot
    ⊢ Membership.mem (Inv.inv ↑I) 1
  -/
  rw [FractionalIdeal.mem_inv_iff (FractionalIdeal.coeIdeal_ne_zero.mpr hI)]
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDomain A
    I : Ideal A
    hI : Ne I Bot.bot
    ⊢ ∀ (y : K), Membership.mem (↑I) y → Membership.mem 1 (HMul.hMul 1 y)
  -/
  intro y hy
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDomain A
    I : Ideal A
    hI : Ne I Bot.bot
    y : K
    hy : Membership.mem (↑I) y
    ⊢ Membership.mem 1 (HMul.hMul 1 y)
  -/
  rw [one_mul]
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDomain A
    I : Ideal A
    hI : Ne I Bot.bot
    y : K
    hy : Membership.mem (↑I) y
    ⊢ Membership.mem 1 y
  -/
  exact FractionalIdeal.coeIdeal_le_one hy
  /-
    🎉 no goals
  -/


/-- Specialization of `exists_primeSpectrum_prod_le_and_ne_bot_of_domain` to Dedekind domains:
Let `I : Ideal A` be a nonzero ideal, where `A` is a Dedekind domain that is not a field.
Then `exists_primeSpectrum_prod_le_and_ne_bot_of_domain` states we can find a product of prime
ideals that is contained within `I`. This lemma extends that result by making the product minimal:
let `M` be a maximal ideal that contains `I`, then the product including `M` is contained within `I`
and the product excluding `M` is not contained within `I`. -/
theorem exists_multiset_prod_cons_le_and_prod_not_le [IsDedekindDomain A] (hNF : ¬IsField A)
    {I M : Ideal A} (hI0 : I ≠ ⊥) (hIM : I ≤ M) [hM : M.IsMaximal] :
    ∃ Z : Multiset (PrimeSpectrum A),
      (M ::ₘ Z.map PrimeSpectrum.asIdeal).prod ≤ I ∧
        ¬Multiset.prod (Z.map PrimeSpectrum.asIdeal) ≤ I := by
  -- Let `Z` be a minimal set of prime ideals such that their product is contained in `J`.
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    hNF : Not (IsField A)
    I M : Ideal A
    hI0 : Ne I Bot.bot
    hIM : LE.le I M
    hM : M.IsMaximal
    ⊢ Exists fun Z => And (LE.le (Multiset.cons M (Multiset.map PrimeSpectrum.asId …
  -/
  obtain ⟨Z₀, hZ₀⟩ := PrimeSpectrum.exists_primeSpectrum_prod_le_and_ne_bot_of_domain hNF hI0
  obtain ⟨Z, ⟨hZI, hprodZ⟩, h_eraseZ⟩ :=
    wellFounded_lt.has_min
      {Z | (Z.map PrimeSpectrum.asIdeal).prod ≤ I ∧ (Z.map PrimeSpectrum.asIdeal).prod ≠ ⊥}
      ⟨Z₀, hZ₀.1, hZ₀.2⟩
  /-
    case intro.intro.intro.intro
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    hNF : Not (IsField A)
    I M : Ideal A
    hI0 : Ne I Bot.bot
    hIM : LE.le I M
    hM : M.IsMaximal
    Z₀ : Multiset (PrimeSpectrum A)
    hZ₀ : And (LE.le (Multiset.map PrimeSpectrum.asIdeal Z₀).prod I) (Ne (Multiset …
    Z : Multiset (PrimeSpectrum A)
    h_eraseZ : ∀ (x : Multiset (PrimeSpectrum A)), Membership.mem (setOf fun Z =>  …
    hZI : LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod I
    hprodZ : Ne (Multiset.map PrimeSpectrum.asIdeal Z).prod Bot.bot
    ⊢ Exists fun Z => And (LE.le (Multiset.cons M (Multiset.map PrimeSpectrum.asId …
  -/
  obtain ⟨_, hPZ', hPM⟩ := hM.isPrime.multiset_prod_le.mp (hZI.trans hIM)
  -- Then in fact there is a `P ∈ Z` with `P ≤ M`.
  /-
    case intro.intro.intro.intro.intro.intro
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    hNF : Not (IsField A)
    I M : Ideal A
    hI0 : Ne I Bot.bot
    hIM : LE.le I M
    hM : M.IsMaximal
    Z₀ : Multiset (PrimeSpectrum A)
    hZ₀ : And (LE.le (Multiset.map PrimeSpectrum.asIdeal Z₀).prod I) (Ne (Multiset …
    Z : Multiset (PrimeSpectrum A)
    h_eraseZ : ∀ (x : Multiset (PrimeSpectrum A)), Membership.mem (setOf fun Z =>  …
    hZI : LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod I
    hprodZ : Ne (Multiset.map PrimeSpectrum.asIdeal Z).prod Bot.bot
    w✝ : Ideal A
    hPZ' : Membership.mem (Multiset.map PrimeSpectrum.asIdeal Z) w✝
    hPM : LE.le w✝ M
    ⊢ Exists fun Z => And (LE.le (Multiset.cons M (Multiset.map PrimeSpectrum.asId …
  -/
  obtain ⟨P, hPZ, rfl⟩ := Multiset.mem_map.mp hPZ'
  classical
    have := Multiset.map_erase PrimeSpectrum.asIdeal (fun _ _ => PrimeSpectrum.ext) P Z
    obtain ⟨hP0, hZP0⟩ : P.asIdeal ≠ ⊥ ∧ ((Z.erase P).map PrimeSpectrum.asIdeal).prod ≠ ⊥ := by
      rwa [Ne, ← Multiset.cons_erase hPZ', Multiset.prod_cons, Ideal.mul_eq_bot, not_or, ←
        this] at hprodZ
    -- By maximality of `P` and `M`, we have that `P ≤ M` implies `P = M`.
    have hPM' := (P.isPrime.isMaximal hP0).eq_of_le hM.ne_top hPM
    subst hPM'
    -- By minimality of `Z`, erasing `P` from `Z` is exactly what we need.
    refine ⟨Z.erase P, ?_, ?_⟩
    · convert hZI
      rw [this, Multiset.cons_erase hPZ']
    · refine fun h => h_eraseZ (Z.erase P) ⟨h, ?_⟩ (Multiset.erase_lt.mpr hPZ)
      exact hZP0

lemma not_inv_le_one_of_ne_bot [IsDedekindDomain A] {I : Ideal A}
    (hI0 : I ≠ ⊥) (hI1 : I ≠ ⊤) : ¬(I⁻¹ : FractionalIdeal A⁰ K) ≤ 1 := by
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    hI1 : Ne I Top.top
    ⊢ Not (LE.le (Inv.inv ↑I) 1)
  -/
  have hNF : ¬IsField A := fun h ↦ letI := h.toField; (eq_bot_or_eq_top I).elim hI0 hI1
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    hI1 : Ne I Top.top
    hNF : Not (IsField A)
    ⊢ Not (LE.le (Inv.inv ↑I) 1)
  -/
  wlog hM : I.IsMaximal generalizing I
    /-
      case inr
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      I : Ideal A
      hI0 : Ne I Bot.bot
      hI1 : Ne I Top.top
      hNF : Not (IsField A)
      this : ∀ {I : Ideal A}, Ne I Bot.bot → Ne I Top.top → I.IsMaximal → Not (LE.le …
      hM : Not I.IsMaximal
      ⊢ Not (LE.le (Inv.inv ↑I) 1)
    -/
  · rcases I.exists_le_maximal hI1 with ⟨M, hmax, hIM⟩
    /-
      case inr.intro.intro
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      I : Ideal A
      hI0 : Ne I Bot.bot
      hI1 : Ne I Top.top
      hNF : Not (IsField A)
      this : ∀ {I : Ideal A}, Ne I Bot.bot → Ne I Top.top → I.IsMaximal → Not (LE.le …
      hM : Not I.IsMaximal
      M : Ideal A
      hmax : M.IsMaximal
      hIM : LE.le I M
      ⊢ Not (LE.le (Inv.inv ↑I) 1)
    -/
    have hMbot : M ≠ ⊥ := (M.bot_lt_of_maximal hNF).ne'
    /-
      case inr.intro.intro
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      I : Ideal A
      hI0 : Ne I Bot.bot
      hI1 : Ne I Top.top
      hNF : Not (IsField A)
      this : ∀ {I : Ideal A}, Ne I Bot.bot → Ne I Top.top → I.IsMaximal → Not (LE.le …
      hM : Not I.IsMaximal
      M : Ideal A
      hmax : M.IsMaximal
      hIM : LE.le I M
      hMbot : Ne M Bot.bot
      ⊢ Not (LE.le (Inv.inv ↑I) 1)
    -/
    refine mt (le_trans <| inv_anti_mono ?_ ?_ ?_) (this hMbot hmax.ne_top hmax) <;>
      /-
        case inr.intro.intro.refine_1
        A : Type u_2
        K : Type u_3
        inst✝⁴ : CommRing A
        inst✝³ : Field K
        inst✝² : Algebra A K
        inst✝¹ : IsFractionRing A K
        inst✝ : IsDedekindDomain A
        I : Ideal A
        hI0 : Ne I Bot.bot
        hI1 : Ne I Top.top
        hNF : Not (IsField A)
        this : ∀ {I : Ideal A}, Ne I Bot.bot → Ne I Top.top → I.IsMaximal → Not (LE.le …
        hM : Not I.IsMaximal
        M : Ideal A
        hmax : M.IsMaximal
        hIM : LE.le I M
        hMbot : Ne M Bot.bot
        ⊢ Ne (↑I) 0
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      simpa only [coeIdeal_ne_zero, coeIdeal_le_coeIdeal]
      /-
        🎉 no goals
      -/
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    hNF : Not (IsField A)
    I : Ideal A
    hI0 : Ne I Bot.bot
    hI1 : Ne I Top.top
    hM : I.IsMaximal
    ⊢ Not (LE.le (Inv.inv ↑I) 1)
  -/
  have hI0 : ⊥ < I := I.bot_lt_of_maximal hNF
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    hNF : Not (IsField A)
    I : Ideal A
    hI0✝ : Ne I Bot.bot
    hI1 : Ne I Top.top
    hM : I.IsMaximal
    hI0 : LT.lt Bot.bot I
    ⊢ Not (LE.le (Inv.inv ↑I) 1)
  -/
  obtain ⟨⟨a, haI⟩, ha0⟩ := Submodule.nonzero_mem_of_bot_lt hI0
  /-
    case intro.mk
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    hNF : Not (IsField A)
    I : Ideal A
    hI0✝ : Ne I Bot.bot
    hI1 : Ne I Top.top
    hM : I.IsMaximal
    hI0 : LT.lt Bot.bot I
    a : A
    haI : Membership.mem I a
    ha0 : Ne ⟨a, haI⟩ 0
    ⊢ Not (LE.le (Inv.inv ↑I) 1)
  -/
  replace ha0 : a ≠ 0 := Subtype.coe_injective.ne ha0
  /-
    case intro.mk
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    hNF : Not (IsField A)
    I : Ideal A
    hI0✝ : Ne I Bot.bot
    hI1 : Ne I Top.top
    hM : I.IsMaximal
    hI0 : LT.lt Bot.bot I
    a : A
    haI : Membership.mem I a
    ha0 : Ne a 0
    ⊢ Not (LE.le (Inv.inv ↑I) 1)
  -/
  let J : Ideal A := Ideal.span {a}
  /-
    case intro.mk
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    hNF : Not (IsField A)
    I : Ideal A
    hI0✝ : Ne I Bot.bot
    hI1 : Ne I Top.top
    hM : I.IsMaximal
    hI0 : LT.lt Bot.bot I
    a : A
    haI : Membership.mem I a
    ha0 : Ne a 0
    J : Ideal A := Ideal.span (Singleton.singleton a)
    ⊢ Not (LE.le (Inv.inv ↑I) 1)
  -/
  have hJ0 : J ≠ ⊥ := mt Ideal.span_singleton_eq_bot.mp ha0
  /-
    case intro.mk
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    hNF : Not (IsField A)
    I : Ideal A
    hI0✝ : Ne I Bot.bot
    hI1 : Ne I Top.top
    hM : I.IsMaximal
    hI0 : LT.lt Bot.bot I
    a : A
    haI : Membership.mem I a
    ha0 : Ne a 0
    J : Ideal A := Ideal.span (Singleton.singleton a)
    hJ0 : Ne J Bot.bot
    ⊢ Not (LE.le (Inv.inv ↑I) 1)
  -/
  have hJI : J ≤ I := I.span_singleton_le_iff_mem.2 haI
  -- Then we can find a product of prime (hence maximal) ideals contained in `J`,
  -- such that removing element `M` from the product is not contained in `J`.
  /-
    case intro.mk
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    hNF : Not (IsField A)
    I : Ideal A
    hI0✝ : Ne I Bot.bot
    hI1 : Ne I Top.top
    hM : I.IsMaximal
    hI0 : LT.lt Bot.bot I
    a : A
    haI : Membership.mem I a
    ha0 : Ne a 0
    J : Ideal A := Ideal.span (Singleton.singleton a)
    hJ0 : Ne J Bot.bot
    hJI : LE.le J I
    ⊢ Not (LE.le (Inv.inv ↑I) 1)
  -/
  obtain ⟨Z, hle, hnle⟩ := exists_multiset_prod_cons_le_and_prod_not_le hNF hJ0 hJI
  -- Choose an element `b` of the product that is not in `J`.
  /-
    case intro.mk.intro.intro
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    hNF : Not (IsField A)
    I : Ideal A
    hI0✝ : Ne I Bot.bot
    hI1 : Ne I Top.top
    hM : I.IsMaximal
    hI0 : LT.lt Bot.bot I
    a : A
    haI : Membership.mem I a
    ha0 : Ne a 0
    J : Ideal A := Ideal.span (Singleton.singleton a)
    hJ0 : Ne J Bot.bot
    hJI : LE.le J I
    Z : Multiset (PrimeSpectrum A)
    hle : LE.le (Multiset.cons I (Multiset.map PrimeSpectrum.asIdeal Z)).prod J
    hnle : Not (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod J)
    ⊢ Not (LE.le (Inv.inv ↑I) 1)
  -/
  obtain ⟨b, hbZ, hbJ⟩ := SetLike.not_le_iff_exists.mp hnle
  have hnz_fa : algebraMap A K a ≠ 0 :=
    mt ((injective_iff_map_eq_zero _).mp (IsFractionRing.injective A K) a) ha0
  -- Then `b a⁻¹ : K` is in `M⁻¹` but not in `1`.
  /-
    case intro.mk.intro.intro.intro.intro
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    hNF : Not (IsField A)
    I : Ideal A
    hI0✝ : Ne I Bot.bot
    hI1 : Ne I Top.top
    hM : I.IsMaximal
    hI0 : LT.lt Bot.bot I
    a : A
    haI : Membership.mem I a
    ha0 : Ne a 0
    J : Ideal A := Ideal.span (Singleton.singleton a)
    hJ0 : Ne J Bot.bot
    hJI : LE.le J I
    Z : Multiset (PrimeSpectrum A)
    hle : LE.le (Multiset.cons I (Multiset.map PrimeSpectrum.asIdeal Z)).prod J
    hnle : Not (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod J)
    b : A
    hbZ : Membership.mem (Multiset.map PrimeSpectrum.asIdeal Z).prod b
    hbJ : Not (Membership.mem J b)
    hnz_fa : Ne ((algebraMap A K) a) 0
    ⊢ Not (LE.le (Inv.inv ↑I) 1)
  -/
  refine Set.not_subset.2 ⟨algebraMap A K b * (algebraMap A K a)⁻¹, (mem_inv_iff ?_).mpr ?_, ?_⟩
    /-
      case intro.mk.intro.intro.intro.intro.refine_1
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      hNF : Not (IsField A)
      I : Ideal A
      hI0✝ : Ne I Bot.bot
      hI1 : Ne I Top.top
      hM : I.IsMaximal
      hI0 : LT.lt Bot.bot I
      a : A
      haI : Membership.mem I a
      ha0 : Ne a 0
      J : Ideal A := Ideal.span (Singleton.singleton a)
      hJ0 : Ne J Bot.bot
      hJI : LE.le J I
      Z : Multiset (PrimeSpectrum A)
      hle : LE.le (Multiset.cons I (Multiset.map PrimeSpectrum.asIdeal Z)).prod J
      hnle : Not (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod J)
      b : A
      hbZ : Membership.mem (Multiset.map PrimeSpectrum.asIdeal Z).prod b
      hbJ : Not (Membership.mem J b)
      hnz_fa : Ne ((algebraMap A K) a) 0
      ⊢ Ne (↑I) 0
    -/
  · exact coeIdeal_ne_zero.mpr hI0.ne'
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.intro.intro.intro.intro.refine_2
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      hNF : Not (IsField A)
      I : Ideal A
      hI0✝ : Ne I Bot.bot
      hI1 : Ne I Top.top
      hM : I.IsMaximal
      hI0 : LT.lt Bot.bot I
      a : A
      haI : Membership.mem I a
      ha0 : Ne a 0
      J : Ideal A := Ideal.span (Singleton.singleton a)
      hJ0 : Ne J Bot.bot
      hJI : LE.le J I
      Z : Multiset (PrimeSpectrum A)
      hle : LE.le (Multiset.cons I (Multiset.map PrimeSpectrum.asIdeal Z)).prod J
      hnle : Not (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod J)
      b : A
      hbZ : Membership.mem (Multiset.map PrimeSpectrum.asIdeal Z).prod b
      hbJ : Not (Membership.mem J b)
      hnz_fa : Ne ((algebraMap A K) a) 0
      ⊢ ∀ (y : K), Membership.mem (↑I) y → Membership.mem 1 (HMul.hMul (HMul.hMul (( …
    -/
  · rintro y₀ hy₀
    /-
      case intro.mk.intro.intro.intro.intro.refine_2
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      hNF : Not (IsField A)
      I : Ideal A
      hI0✝ : Ne I Bot.bot
      hI1 : Ne I Top.top
      hM : I.IsMaximal
      hI0 : LT.lt Bot.bot I
      a : A
      haI : Membership.mem I a
      ha0 : Ne a 0
      J : Ideal A := Ideal.span (Singleton.singleton a)
      hJ0 : Ne J Bot.bot
      hJI : LE.le J I
      Z : Multiset (PrimeSpectrum A)
      hle : LE.le (Multiset.cons I (Multiset.map PrimeSpectrum.asIdeal Z)).prod J
      hnle : Not (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod J)
      b : A
      hbZ : Membership.mem (Multiset.map PrimeSpectrum.asIdeal Z).prod b
      hbJ : Not (Membership.mem J b)
      hnz_fa : Ne ((algebraMap A K) a) 0
      y₀ : K
      hy₀ : Membership.mem (↑I) y₀
      ⊢ Membership.mem 1 (HMul.hMul (HMul.hMul ((algebraMap A K) b) (Inv.inv ((algeb …
    -/
    obtain ⟨y, h_Iy, rfl⟩ := (mem_coeIdeal _).mp hy₀
    /-
      case intro.mk.intro.intro.intro.intro.refine_2.intro.intro
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      hNF : Not (IsField A)
      I : Ideal A
      hI0✝ : Ne I Bot.bot
      hI1 : Ne I Top.top
      hM : I.IsMaximal
      hI0 : LT.lt Bot.bot I
      a : A
      haI : Membership.mem I a
      ha0 : Ne a 0
      J : Ideal A := Ideal.span (Singleton.singleton a)
      hJ0 : Ne J Bot.bot
      hJI : LE.le J I
      Z : Multiset (PrimeSpectrum A)
      hle : LE.le (Multiset.cons I (Multiset.map PrimeSpectrum.asIdeal Z)).prod J
      hnle : Not (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod J)
      b : A
      hbZ : Membership.mem (Multiset.map PrimeSpectrum.asIdeal Z).prod b
      hbJ : Not (Membership.mem J b)
      hnz_fa : Ne ((algebraMap A K) a) 0
      y : A
      h_Iy : Membership.mem I y
      hy₀ : Membership.mem (↑I) ((algebraMap A K) y)
      ⊢ Membership.mem 1 (HMul.hMul (HMul.hMul ((algebraMap A K) b) (Inv.inv ((algeb …
    -/
    rw [mul_comm, ← mul_assoc, ← RingHom.map_mul]
    have h_yb : y * b ∈ J := by
      apply hle
      rw [Multiset.prod_cons]
      exact Submodule.smul_mem_smul h_Iy hbZ
    /-
      case intro.mk.intro.intro.intro.intro.refine_2.intro.intro
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      hNF : Not (IsField A)
      I : Ideal A
      hI0✝ : Ne I Bot.bot
      hI1 : Ne I Top.top
      hM : I.IsMaximal
      hI0 : LT.lt Bot.bot I
      a : A
      haI : Membership.mem I a
      ha0 : Ne a 0
      J : Ideal A := Ideal.span (Singleton.singleton a)
      hJ0 : Ne J Bot.bot
      hJI : LE.le J I
      Z : Multiset (PrimeSpectrum A)
      hle : LE.le (Multiset.cons I (Multiset.map PrimeSpectrum.asIdeal Z)).prod J
      hnle : Not (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod J)
      b : A
      hbZ : Membership.mem (Multiset.map PrimeSpectrum.asIdeal Z).prod b
      hbJ : Not (Membership.mem J b)
      hnz_fa : Ne ((algebraMap A K) a) 0
      y : A
      h_Iy : Membership.mem I y
      hy₀ : Membership.mem (↑I) ((algebraMap A K) y)
      h_yb : Membership.mem J (HMul.hMul y b)
      ⊢ Membership.mem 1 (HMul.hMul ((algebraMap A K) (HMul.hMul y b)) (Inv.inv ((al …
    -/
    rw [Ideal.mem_span_singleton'] at h_yb
    /-
      case intro.mk.intro.intro.intro.intro.refine_2.intro.intro
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      hNF : Not (IsField A)
      I : Ideal A
      hI0✝ : Ne I Bot.bot
      hI1 : Ne I Top.top
      hM : I.IsMaximal
      hI0 : LT.lt Bot.bot I
      a : A
      haI : Membership.mem I a
      ha0 : Ne a 0
      J : Ideal A := Ideal.span (Singleton.singleton a)
      hJ0 : Ne J Bot.bot
      hJI : LE.le J I
      Z : Multiset (PrimeSpectrum A)
      hle : LE.le (Multiset.cons I (Multiset.map PrimeSpectrum.asIdeal Z)).prod J
      hnle : Not (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod J)
      b : A
      hbZ : Membership.mem (Multiset.map PrimeSpectrum.asIdeal Z).prod b
      hbJ : Not (Membership.mem J b)
      hnz_fa : Ne ((algebraMap A K) a) 0
      y : A
      h_Iy : Membership.mem I y
      hy₀ : Membership.mem (↑I) ((algebraMap A K) y)
      h_yb : Exists fun a_1 => Eq (HMul.hMul a_1 a) (HMul.hMul y b)
      ⊢ Membership.mem 1 (HMul.hMul ((algebraMap A K) (HMul.hMul y b)) (Inv.inv ((al …
    -/
    rcases h_yb with ⟨c, hc⟩
    /-
      case intro.mk.intro.intro.intro.intro.refine_2.intro.intro.intro
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      hNF : Not (IsField A)
      I : Ideal A
      hI0✝ : Ne I Bot.bot
      hI1 : Ne I Top.top
      hM : I.IsMaximal
      hI0 : LT.lt Bot.bot I
      a : A
      haI : Membership.mem I a
      ha0 : Ne a 0
      J : Ideal A := Ideal.span (Singleton.singleton a)
      hJ0 : Ne J Bot.bot
      hJI : LE.le J I
      Z : Multiset (PrimeSpectrum A)
      hle : LE.le (Multiset.cons I (Multiset.map PrimeSpectrum.asIdeal Z)).prod J
      hnle : Not (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod J)
      b : A
      hbZ : Membership.mem (Multiset.map PrimeSpectrum.asIdeal Z).prod b
      hbJ : Not (Membership.mem J b)
      hnz_fa : Ne ((algebraMap A K) a) 0
      y : A
      h_Iy : Membership.mem I y
      hy₀ : Membership.mem (↑I) ((algebraMap A K) y)
      c : A
      hc : Eq (HMul.hMul c a) (HMul.hMul y b)
      ⊢ Membership.mem 1 (HMul.hMul ((algebraMap A K) (HMul.hMul y b)) (Inv.inv ((al …
    -/
    rw [← hc, RingHom.map_mul, mul_assoc, mul_inv_cancel₀ hnz_fa, mul_one]
    /-
      case intro.mk.intro.intro.intro.intro.refine_2.intro.intro.intro
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      hNF : Not (IsField A)
      I : Ideal A
      hI0✝ : Ne I Bot.bot
      hI1 : Ne I Top.top
      hM : I.IsMaximal
      hI0 : LT.lt Bot.bot I
      a : A
      haI : Membership.mem I a
      ha0 : Ne a 0
      J : Ideal A := Ideal.span (Singleton.singleton a)
      hJ0 : Ne J Bot.bot
      hJI : LE.le J I
      Z : Multiset (PrimeSpectrum A)
      hle : LE.le (Multiset.cons I (Multiset.map PrimeSpectrum.asIdeal Z)).prod J
      hnle : Not (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod J)
      b : A
      hbZ : Membership.mem (Multiset.map PrimeSpectrum.asIdeal Z).prod b
      hbJ : Not (Membership.mem J b)
      hnz_fa : Ne ((algebraMap A K) a) 0
      y : A
      h_Iy : Membership.mem I y
      hy₀ : Membership.mem (↑I) ((algebraMap A K) y)
      c : A
      hc : Eq (HMul.hMul c a) (HMul.hMul y b)
      ⊢ Membership.mem 1 ((algebraMap A K) c)
    -/
    apply coe_mem_one
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.intro.intro.intro.intro.refine_3
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      hNF : Not (IsField A)
      I : Ideal A
      hI0✝ : Ne I Bot.bot
      hI1 : Ne I Top.top
      hM : I.IsMaximal
      hI0 : LT.lt Bot.bot I
      a : A
      haI : Membership.mem I a
      ha0 : Ne a 0
      J : Ideal A := Ideal.span (Singleton.singleton a)
      hJ0 : Ne J Bot.bot
      hJI : LE.le J I
      Z : Multiset (PrimeSpectrum A)
      hle : LE.le (Multiset.cons I (Multiset.map PrimeSpectrum.asIdeal Z)).prod J
      hnle : Not (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod J)
      b : A
      hbZ : Membership.mem (Multiset.map PrimeSpectrum.asIdeal Z).prod b
      hbJ : Not (Membership.mem J b)
      hnz_fa : Ne ((algebraMap A K) a) 0
      ⊢ Not (Membership.mem (↑((fun a => ↑a) 1)) (HMul.hMul ((algebraMap A K) b) (In …
    -/
  · refine mt (mem_one_iff _).mp ?_
    /-
      case intro.mk.intro.intro.intro.intro.refine_3
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      hNF : Not (IsField A)
      I : Ideal A
      hI0✝ : Ne I Bot.bot
      hI1 : Ne I Top.top
      hM : I.IsMaximal
      hI0 : LT.lt Bot.bot I
      a : A
      haI : Membership.mem I a
      ha0 : Ne a 0
      J : Ideal A := Ideal.span (Singleton.singleton a)
      hJ0 : Ne J Bot.bot
      hJI : LE.le J I
      Z : Multiset (PrimeSpectrum A)
      hle : LE.le (Multiset.cons I (Multiset.map PrimeSpectrum.asIdeal Z)).prod J
      hnle : Not (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod J)
      b : A
      hbZ : Membership.mem (Multiset.map PrimeSpectrum.asIdeal Z).prod b
      hbJ : Not (Membership.mem J b)
      hnz_fa : Ne ((algebraMap A K) a) 0
      ⊢ Not (Exists fun x' => Eq ((algebraMap A K) x') (HMul.hMul ((algebraMap A K)  …
    -/
    rintro ⟨x', h₂_abs⟩
    /-
      case intro.mk.intro.intro.intro.intro.refine_3.intro
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      hNF : Not (IsField A)
      I : Ideal A
      hI0✝ : Ne I Bot.bot
      hI1 : Ne I Top.top
      hM : I.IsMaximal
      hI0 : LT.lt Bot.bot I
      a : A
      haI : Membership.mem I a
      ha0 : Ne a 0
      J : Ideal A := Ideal.span (Singleton.singleton a)
      hJ0 : Ne J Bot.bot
      hJI : LE.le J I
      Z : Multiset (PrimeSpectrum A)
      hle : LE.le (Multiset.cons I (Multiset.map PrimeSpectrum.asIdeal Z)).prod J
      hnle : Not (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod J)
      b : A
      hbZ : Membership.mem (Multiset.map PrimeSpectrum.asIdeal Z).prod b
      hbJ : Not (Membership.mem J b)
      hnz_fa : Ne ((algebraMap A K) a) 0
      x' : A
      h₂_abs : Eq ((algebraMap A K) x') (HMul.hMul ((algebraMap A K) b) (Inv.inv ((a …
      ⊢ False
    -/
    rw [← div_eq_mul_inv, eq_div_iff_mul_eq hnz_fa, ← RingHom.map_mul] at h₂_abs
    /-
      case intro.mk.intro.intro.intro.intro.refine_3.intro
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      hNF : Not (IsField A)
      I : Ideal A
      hI0✝ : Ne I Bot.bot
      hI1 : Ne I Top.top
      hM : I.IsMaximal
      hI0 : LT.lt Bot.bot I
      a : A
      haI : Membership.mem I a
      ha0 : Ne a 0
      J : Ideal A := Ideal.span (Singleton.singleton a)
      hJ0 : Ne J Bot.bot
      hJI : LE.le J I
      Z : Multiset (PrimeSpectrum A)
      hle : LE.le (Multiset.cons I (Multiset.map PrimeSpectrum.asIdeal Z)).prod J
      hnle : Not (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod J)
      b : A
      hbZ : Membership.mem (Multiset.map PrimeSpectrum.asIdeal Z).prod b
      hbJ : Not (Membership.mem J b)
      hnz_fa : Ne ((algebraMap A K) a) 0
      x' : A
      h₂_abs : Eq ((algebraMap A K) (HMul.hMul x' a)) ((algebraMap A K) b)
      ⊢ False
    -/
    have := Ideal.mem_span_singleton'.mpr ⟨x', IsFractionRing.injective A K h₂_abs⟩
    /-
      case intro.mk.intro.intro.intro.intro.refine_3.intro
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      hNF : Not (IsField A)
      I : Ideal A
      hI0✝ : Ne I Bot.bot
      hI1 : Ne I Top.top
      hM : I.IsMaximal
      hI0 : LT.lt Bot.bot I
      a : A
      haI : Membership.mem I a
      ha0 : Ne a 0
      J : Ideal A := Ideal.span (Singleton.singleton a)
      hJ0 : Ne J Bot.bot
      hJI : LE.le J I
      Z : Multiset (PrimeSpectrum A)
      hle : LE.le (Multiset.cons I (Multiset.map PrimeSpectrum.asIdeal Z)).prod J
      hnle : Not (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod J)
      b : A
      hbZ : Membership.mem (Multiset.map PrimeSpectrum.asIdeal Z).prod b
      hbJ : Not (Membership.mem J b)
      hnz_fa : Ne ((algebraMap A K) a) 0
      x' : A
      h₂_abs : Eq ((algebraMap A K) (HMul.hMul x' a)) ((algebraMap A K) b)
      this : Membership.mem (Ideal.span (Singleton.singleton a)) b
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/


theorem exists_not_mem_one_of_ne_bot [IsDedekindDomain A] {I : Ideal A} (hI0 : I ≠ ⊥)
    (hI1 : I ≠ ⊤) : ∃ x ∈ (I⁻¹ : FractionalIdeal A⁰ K), x ∉ (1 : FractionalIdeal A⁰ K) :=
  Set.not_subset.1 <| not_inv_le_one_of_ne_bot hI0 hI1


theorem mul_inv_cancel_of_le_one [h : IsDedekindDomain A] {I : Ideal A} (hI0 : I ≠ ⊥)
    (hI : (I * (I : FractionalIdeal A⁰ K)⁻¹)⁻¹ ≤ 1) : I * (I : FractionalIdeal A⁰ K)⁻¹ = 1 := by
  -- We'll show a contradiction with `exists_not_mem_one_of_ne_bot`:
  -- `J⁻¹ = (I * I⁻¹)⁻¹` cannot have an element `x ∉ 1`, so it must equal `1`.
  obtain ⟨J, hJ⟩ : ∃ J : Ideal A, (J : FractionalIdeal A⁰ K) = I * (I : FractionalIdeal A⁰ K)⁻¹ :=
    le_one_iff_exists_coeIdeal.mp mul_one_div_le_one
  /-
    case intro
    A : Type u_2
    K : Type u_3
    inst✝³ : CommRing A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    h : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    hI : LE.le (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I))) 1
    J : Ideal A
    hJ : Eq (↑J) (HMul.hMul (↑I) (Inv.inv ↑I))
    ⊢ Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 1
  -/
  by_cases hJ0 : J = ⊥
    /-
      case pos
      A : Type u_2
      K : Type u_3
      inst✝³ : CommRing A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      h : IsDedekindDomain A
      I : Ideal A
      hI0 : Ne I Bot.bot
      hI : LE.le (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I))) 1
      J : Ideal A
      hJ : Eq (↑J) (HMul.hMul (↑I) (Inv.inv ↑I))
      hJ0 : Eq J Bot.bot
      ⊢ Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 1
    -/
  · subst hJ0
    /-
      case pos
      A : Type u_2
      K : Type u_3
      inst✝³ : CommRing A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      h : IsDedekindDomain A
      I : Ideal A
      hI0 : Ne I Bot.bot
      hI : LE.le (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I))) 1
      hJ : Eq (↑Bot.bot) (HMul.hMul (↑I) (Inv.inv ↑I))
      ⊢ Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 1
    -/
    refine absurd ?_ hI0
    /-
      case pos
      A : Type u_2
      K : Type u_3
      inst✝³ : CommRing A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      h : IsDedekindDomain A
      I : Ideal A
      hI0 : Ne I Bot.bot
      hI : LE.le (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I))) 1
      hJ : Eq (↑Bot.bot) (HMul.hMul (↑I) (Inv.inv ↑I))
      ⊢ Eq I Bot.bot
    -/
    rw [eq_bot_iff, ← coeIdeal_le_coeIdeal K, hJ]
    /-
      case pos
      A : Type u_2
      K : Type u_3
      inst✝³ : CommRing A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      h : IsDedekindDomain A
      I : Ideal A
      hI0 : Ne I Bot.bot
      hI : LE.le (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I))) 1
      hJ : Eq (↑Bot.bot) (HMul.hMul (↑I) (Inv.inv ↑I))
      ⊢ LE.le (↑I) (HMul.hMul (↑I) (Inv.inv ↑I))
    -/
    exact coe_ideal_le_self_mul_inv K I
    /-
      🎉 no goals
    -/
  /-
    case neg
    A : Type u_2
    K : Type u_3
    inst✝³ : CommRing A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    h : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    hI : LE.le (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I))) 1
    J : Ideal A
    hJ : Eq (↑J) (HMul.hMul (↑I) (Inv.inv ↑I))
    hJ0 : Not (Eq J Bot.bot)
    ⊢ Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 1
  -/
  by_cases hJ1 : J = ⊤
    /-
      case pos
      A : Type u_2
      K : Type u_3
      inst✝³ : CommRing A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      h : IsDedekindDomain A
      I : Ideal A
      hI0 : Ne I Bot.bot
      hI : LE.le (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I))) 1
      J : Ideal A
      hJ : Eq (↑J) (HMul.hMul (↑I) (Inv.inv ↑I))
      hJ0 : Not (Eq J Bot.bot)
      hJ1 : Eq J Top.top
      ⊢ Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 1
    -/
  · rw [← hJ, hJ1, coeIdeal_top]
    /-
      🎉 no goals
    -/
  /-
    case neg
    A : Type u_2
    K : Type u_3
    inst✝³ : CommRing A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    h : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    hI : LE.le (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I))) 1
    J : Ideal A
    hJ : Eq (↑J) (HMul.hMul (↑I) (Inv.inv ↑I))
    hJ0 : Not (Eq J Bot.bot)
    hJ1 : Not (Eq J Top.top)
    ⊢ Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 1
  -/
  exact (not_inv_le_one_of_ne_bot (K := K) hJ0 hJ1 (hJ ▸ hI)).elim
  /-
    🎉 no goals
  -/


/-- Nonzero integral ideals in a Dedekind domain are invertible.

We will use this to show that nonzero fractional ideals are invertible,
and finally conclude that fractional ideals in a Dedekind domain form a group with zero.
-/
theorem coe_ideal_mul_inv [h : IsDedekindDomain A] (I : Ideal A) (hI0 : I ≠ ⊥) :
    I * (I : FractionalIdeal A⁰ K)⁻¹ = 1 := by
  -- We'll show `1 ≤ J⁻¹ = (I * I⁻¹)⁻¹ ≤ 1`.
  /-
    A : Type u_2
    K : Type u_3
    inst✝³ : CommRing A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    h : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    ⊢ Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 1
  -/
  apply mul_inv_cancel_of_le_one hI0
  /-
    A : Type u_2
    K : Type u_3
    inst✝³ : CommRing A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    h : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    ⊢ LE.le (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I))) 1
  -/
  by_cases hJ0 : I * (I : FractionalIdeal A⁰ K)⁻¹ = 0
    /-
      case pos
      A : Type u_2
      K : Type u_3
      inst✝³ : CommRing A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      h : IsDedekindDomain A
      I : Ideal A
      hI0 : Ne I Bot.bot
      hJ0 : Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 0
      ⊢ LE.le (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I))) 1
    -/
  · rw [hJ0, inv_zero']; exact zero_le _
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    A : Type u_2
    K : Type u_3
    inst✝³ : CommRing A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    h : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    hJ0 : Not (Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 0)
    ⊢ LE.le (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I))) 1
  -/
  intro x hx
  -- In particular, we'll show all `x ∈ J⁻¹` are integral.
  suffices x ∈ integralClosure A K by
    rwa [IsIntegrallyClosed.integralClosure_eq_bot, Algebra.mem_bot, Set.mem_range,
      ← mem_one_iff] at this
  -- For that, we'll find a subalgebra that is f.g. as a module and contains `x`.
  -- `A` is a noetherian ring, so we just need to find a subalgebra between `{x}` and `I⁻¹`.
  /-
    case neg
    A : Type u_2
    K : Type u_3
    inst✝³ : CommRing A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    h : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    hJ0 : Not (Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 0)
    x : K
    hx : Membership.mem ((fun a => ↑a) (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I)))) x
    ⊢ Membership.mem (integralClosure A K) x
  -/
  rw [mem_integralClosure_iff_mem_fg]
  have x_mul_mem : ∀ b ∈ (I⁻¹ : FractionalIdeal A⁰ K), x * b ∈ (I⁻¹ : FractionalIdeal A⁰ K) := by
    intro b hb
    rw [mem_inv_iff (coeIdeal_ne_zero.mpr hI0)]
    dsimp only at hx
    rw [val_eq_coe, mem_coe, mem_inv_iff hJ0] at hx
    simp only [mul_assoc, mul_comm b] at hx ⊢
    intro y hy
    exact hx _ (mul_mem_mul hy hb)
  -- It turns out the subalgebra consisting of all `p(x)` for `p : A[X]` works.
  refine ⟨AlgHom.range (Polynomial.aeval x : A[X] →ₐ[A] K),
    isNoetherian_submodule.mp (isNoetherian (I : FractionalIdeal A⁰ K)⁻¹) _ fun y hy => ?_,
    ⟨Polynomial.X, Polynomial.aeval_X x⟩⟩
  /-
    case neg
    A : Type u_2
    K : Type u_3
    inst✝³ : CommRing A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    h : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    hJ0 : Not (Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 0)
    x : K
    hx : Membership.mem ((fun a => ↑a) (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I)))) x
    x_mul_mem : ∀ (b : K), Membership.mem (Inv.inv ↑I) b → Membership.mem (Inv.inv …
    y : K
    hy : Membership.mem (Subalgebra.toSubmodule (Polynomial.aeval x).range) y
    ⊢ Membership.mem (↑(Inv.inv ↑I)) y
  -/
  obtain ⟨p, rfl⟩ := (AlgHom.mem_range _).mp hy
  /-
    case neg.intro
    A : Type u_2
    K : Type u_3
    inst✝³ : CommRing A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    h : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    hJ0 : Not (Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 0)
    x : K
    hx : Membership.mem ((fun a => ↑a) (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I)))) x
    x_mul_mem : ∀ (b : K), Membership.mem (Inv.inv ↑I) b → Membership.mem (Inv.inv …
    p : Polynomial A
    hy : Membership.mem (Subalgebra.toSubmodule (Polynomial.aeval x).range) ((Poly …
    ⊢ Membership.mem (↑(Inv.inv ↑I)) ((Polynomial.aeval x) p)
  -/
  rw [Polynomial.aeval_eq_sum_range]
  /-
    case neg.intro
    A : Type u_2
    K : Type u_3
    inst✝³ : CommRing A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    h : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    hJ0 : Not (Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 0)
    x : K
    hx : Membership.mem ((fun a => ↑a) (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I)))) x
    x_mul_mem : ∀ (b : K), Membership.mem (Inv.inv ↑I) b → Membership.mem (Inv.inv …
    p : Polynomial A
    hy : Membership.mem (Subalgebra.toSubmodule (Polynomial.aeval x).range) ((Poly …
    ⊢ Membership.mem (↑(Inv.inv ↑I)) ((Finset.range (HAdd.hAdd p.natDegree 1)).sum …
  -/
  refine Submodule.sum_mem _ fun i hi => Submodule.smul_mem _ _ ?_
  /-
    case neg.intro
    A : Type u_2
    K : Type u_3
    inst✝³ : CommRing A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    h : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    hJ0 : Not (Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 0)
    x : K
    hx : Membership.mem ((fun a => ↑a) (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I)))) x
    x_mul_mem : ∀ (b : K), Membership.mem (Inv.inv ↑I) b → Membership.mem (Inv.inv …
    p : Polynomial A
    hy : Membership.mem (Subalgebra.toSubmodule (Polynomial.aeval x).range) ((Poly …
    i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) i
    ⊢ Membership.mem (↑(Inv.inv ↑I)) (HPow.hPow x i)
  -/
  clear hi
  /-
    case neg.intro
    A : Type u_2
    K : Type u_3
    inst✝³ : CommRing A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    h : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    hJ0 : Not (Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 0)
    x : K
    hx : Membership.mem ((fun a => ↑a) (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I)))) x
    x_mul_mem : ∀ (b : K), Membership.mem (Inv.inv ↑I) b → Membership.mem (Inv.inv …
    p : Polynomial A
    hy : Membership.mem (Subalgebra.toSubmodule (Polynomial.aeval x).range) ((Poly …
    i : Nat
    ⊢ Membership.mem (↑(Inv.inv ↑I)) (HPow.hPow x i)
  -/
  induction' i with i ih
    /-
      case neg.intro.zero
      A : Type u_2
      K : Type u_3
      inst✝³ : CommRing A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      h : IsDedekindDomain A
      I : Ideal A
      hI0 : Ne I Bot.bot
      hJ0 : Not (Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 0)
      x : K
      hx : Membership.mem ((fun a => ↑a) (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I)))) x
      x_mul_mem : ∀ (b : K), Membership.mem (Inv.inv ↑I) b → Membership.mem (Inv.inv …
      p : Polynomial A
      hy : Membership.mem (Subalgebra.toSubmodule (Polynomial.aeval x).range) ((Poly …
      ⊢ Membership.mem (↑(Inv.inv ↑I)) (HPow.hPow x 0)
    -/
  · rw [pow_zero]; exact one_mem_inv_coe_ideal hI0
                   /-
                     🎉 no goals
                   -/
    /-
      case neg.intro.succ
      A : Type u_2
      K : Type u_3
      inst✝³ : CommRing A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      h : IsDedekindDomain A
      I : Ideal A
      hI0 : Ne I Bot.bot
      hJ0 : Not (Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 0)
      x : K
      hx : Membership.mem ((fun a => ↑a) (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I)))) x
      x_mul_mem : ∀ (b : K), Membership.mem (Inv.inv ↑I) b → Membership.mem (Inv.inv …
      p : Polynomial A
      hy : Membership.mem (Subalgebra.toSubmodule (Polynomial.aeval x).range) ((Poly …
      i : Nat
      ih : Membership.mem (↑(Inv.inv ↑I)) (HPow.hPow x i)
      ⊢ Membership.mem (↑(Inv.inv ↑I)) (HPow.hPow x (HAdd.hAdd i 1))
    -/
  · show x ^ i.succ ∈ (I⁻¹ : FractionalIdeal A⁰ K)
    /-
      case neg.intro.succ
      A : Type u_2
      K : Type u_3
      inst✝³ : CommRing A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      h : IsDedekindDomain A
      I : Ideal A
      hI0 : Ne I Bot.bot
      hJ0 : Not (Eq (HMul.hMul (↑I) (Inv.inv ↑I)) 0)
      x : K
      hx : Membership.mem ((fun a => ↑a) (Inv.inv (HMul.hMul (↑I) (Inv.inv ↑I)))) x
      x_mul_mem : ∀ (b : K), Membership.mem (Inv.inv ↑I) b → Membership.mem (Inv.inv …
      p : Polynomial A
      hy : Membership.mem (Subalgebra.toSubmodule (Polynomial.aeval x).range) ((Poly …
      i : Nat
      ih : Membership.mem (↑(Inv.inv ↑I)) (HPow.hPow x i)
      ⊢ Membership.mem (Inv.inv ↑I) (HPow.hPow x i.succ)
    -/
    rw [pow_succ']; exact x_mul_mem _ ih
                    /-
                      🎉 no goals
                    -/


/-- Nonzero fractional ideals in a Dedekind domain are units.

This is also available as `_root_.mul_inv_cancel`, using the
`Semifield` instance defined below.
-/
protected theorem mul_inv_cancel [IsDedekindDomain A] {I : FractionalIdeal A⁰ K} (hne : I ≠ 0) :
    I * I⁻¹ = 1 := by
  obtain ⟨a, J, ha, hJ⟩ :
    ∃ (a : A) (aI : Ideal A), a ≠ 0 ∧ I = spanSingleton A⁰ (algebraMap A K a)⁻¹ * aI :=
    exists_eq_spanSingleton_mul I
  suffices h₂ : I * (spanSingleton A⁰ (algebraMap _ _ a) * (J : FractionalIdeal A⁰ K)⁻¹) = 1 by
    rw [mul_inv_cancel_iff]
    exact ⟨spanSingleton A⁰ (algebraMap _ _ a) * (J : FractionalIdeal A⁰ K)⁻¹, h₂⟩
  /-
    case intro.intro.intro
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    I : FractionalIdeal (nonZeroDivisors A) K
    hne : Ne I 0
    a : A
    J : Ideal A
    ha : Ne a 0
    hJ : Eq I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors A) (Inv.i …
    ⊢ Eq (HMul.hMul I (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors A …
  -/
  subst hJ
  rw [mul_assoc, mul_left_comm (J : FractionalIdeal A⁰ K), coe_ideal_mul_inv, mul_one,
    spanSingleton_mul_spanSingleton, inv_mul_cancel₀, spanSingleton_one]
    /-
      case intro.intro.intro
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      a : A
      J : Ideal A
      ha : Ne a 0
      hne : Ne (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors A) (Inv.in …
      ⊢ Ne ((algebraMap A K) a) 0
    -/
  · exact mt ((injective_iff_map_eq_zero (algebraMap A K)).mp (IsFractionRing.injective A K) _) ha
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.hI0
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      a : A
      J : Ideal A
      ha : Ne a 0
      hne : Ne (HMul.hMul (FractionalIdeal.spanSingleton (nonZeroDivisors A) (Inv.in …
      ⊢ Ne J Bot.bot
    -/
  · exact coeIdeal_ne_zero.mp (right_ne_zero_of_mul hne)
    /-
      🎉 no goals
    -/


theorem mul_right_le_iff [IsDedekindDomain A] {J : FractionalIdeal A⁰ K} (hJ : J ≠ 0) :
    ∀ {I I'}, I * J ≤ I' * J ↔ I ≤ I' := by
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    J : FractionalIdeal (nonZeroDivisors A) K
    hJ : Ne J 0
    ⊢ ∀ {I I' : FractionalIdeal (nonZeroDivisors A) K}, Iff (LE.le (HMul.hMul I J) …
  -/
  intro I I'
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    J : FractionalIdeal (nonZeroDivisors A) K
    hJ : Ne J 0
    I I' : FractionalIdeal (nonZeroDivisors A) K
    ⊢ Iff (LE.le (HMul.hMul I J) (HMul.hMul I' J)) (LE.le I I')
  -/
  constructor
    /-
      case mp
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      J : FractionalIdeal (nonZeroDivisors A) K
      hJ : Ne J 0
      I I' : FractionalIdeal (nonZeroDivisors A) K
      ⊢ LE.le (HMul.hMul I J) (HMul.hMul I' J) → LE.le I I'
    -/
  · intro h
    /-
      case mp
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      J : FractionalIdeal (nonZeroDivisors A) K
      hJ : Ne J 0
      I I' : FractionalIdeal (nonZeroDivisors A) K
      h : LE.le (HMul.hMul I J) (HMul.hMul I' J)
      ⊢ LE.le I I'
    -/
    convert mul_right_mono J⁻¹ h <;> dsimp only <;>
    /-
      case h.e'_3
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      J : FractionalIdeal (nonZeroDivisors A) K
      hJ : Ne J 0
      I I' : FractionalIdeal (nonZeroDivisors A) K
      h : LE.le (HMul.hMul I J) (HMul.hMul I' J)
      ⊢ Eq I (HMul.hMul (HMul.hMul I J) (Inv.inv J))
    -/
    /-
      🎉 no goals
    -/
    rw [mul_assoc, FractionalIdeal.mul_inv_cancel hJ, mul_one]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      J : FractionalIdeal (nonZeroDivisors A) K
      hJ : Ne J 0
      I I' : FractionalIdeal (nonZeroDivisors A) K
      ⊢ LE.le I I' → LE.le (HMul.hMul I J) (HMul.hMul I' J)
    -/
  · exact fun h => mul_right_mono J h
    /-
      🎉 no goals
    -/


theorem mul_left_le_iff [IsDedekindDomain A] {J : FractionalIdeal A⁰ K} (hJ : J ≠ 0) {I I'} :
                                  /-
                                    A : Type u_2
                                    K : Type u_3
                                    inst✝⁴ : CommRing A
                                    inst✝³ : Field K
                                    inst✝² : Algebra A K
                                    inst✝¹ : IsFractionRing A K
                                    inst✝ : IsDedekindDomain A
                                    J : FractionalIdeal (nonZeroDivisors A) K
                                    hJ : Ne J 0
                                    I I' : FractionalIdeal (nonZeroDivisors A) K
                                    ⊢ Iff (LE.le (HMul.hMul J I) (HMul.hMul J I')) (LE.le I I')
                                  -/
    J * I ≤ J * I' ↔ I ≤ I' := by convert mul_right_le_iff hJ using 1; simp only [mul_comm]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem mul_right_strictMono [IsDedekindDomain A] {I : FractionalIdeal A⁰ K} (hI : I ≠ 0) :
    StrictMono (· * I) :=
  strictMono_of_le_iff_le fun _ _ => (mul_right_le_iff hI).symm


theorem mul_left_strictMono [IsDedekindDomain A] {I : FractionalIdeal A⁰ K} (hI : I ≠ 0) :
    StrictMono (I * ·) :=
  strictMono_of_le_iff_le fun _ _ => (mul_left_le_iff hI).symm


/-- This is also available as `_root_.div_eq_mul_inv`, using the
`Semifield` instance defined below.
-/
protected theorem div_eq_mul_inv [IsDedekindDomain A] (I J : FractionalIdeal A⁰ K) :
    I / J = I * J⁻¹ := by
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    I J : FractionalIdeal (nonZeroDivisors A) K
    ⊢ Eq (HDiv.hDiv I J) (HMul.hMul I (Inv.inv J))
  -/
  by_cases hJ : J = 0
    /-
      case pos
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      I J : FractionalIdeal (nonZeroDivisors A) K
      hJ : Eq J 0
      ⊢ Eq (HDiv.hDiv I J) (HMul.hMul I (Inv.inv J))
    -/
  · rw [hJ, div_zero, inv_zero', mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    I J : FractionalIdeal (nonZeroDivisors A) K
    hJ : Not (Eq J 0)
    ⊢ Eq (HDiv.hDiv I J) (HMul.hMul I (Inv.inv J))
  -/
  refine le_antisymm ((mul_right_le_iff hJ).mp ?_) ((le_div_iff_mul_le hJ).mpr ?_)
    /-
      case neg.refine_1
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      I J : FractionalIdeal (nonZeroDivisors A) K
      hJ : Not (Eq J 0)
      ⊢ LE.le (HMul.hMul (HDiv.hDiv I J) J) (HMul.hMul (HMul.hMul I (Inv.inv J)) J)
    -/
  · rw [mul_assoc, mul_comm J⁻¹, FractionalIdeal.mul_inv_cancel hJ, mul_one, mul_le]
    /-
      case neg.refine_1
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      I J : FractionalIdeal (nonZeroDivisors A) K
      hJ : Not (Eq J 0)
      ⊢ ∀ (i : K), Membership.mem (HDiv.hDiv I J) i → ∀ (j : K), Membership.mem J j  …
    -/
    intro x hx y hy
    /-
      case neg.refine_1
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      I J : FractionalIdeal (nonZeroDivisors A) K
      hJ : Not (Eq J 0)
      x : K
      hx : Membership.mem (HDiv.hDiv I J) x
      y : K
      hy : Membership.mem J y
      ⊢ Membership.mem I (HMul.hMul x y)
    -/
    rw [mem_div_iff_of_nonzero hJ] at hx
    /-
      case neg.refine_1
      A : Type u_2
      K : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : IsDedekindDomain A
      I J : FractionalIdeal (nonZeroDivisors A) K
      hJ : Not (Eq J 0)
      x : K
      hx : ∀ (y : K), Membership.mem J y → Membership.mem I (HMul.hMul x y)
      y : K
      hy : Membership.mem J y
      ⊢ Membership.mem I (HMul.hMul x y)
    -/
    exact hx y hy
    /-
      🎉 no goals
    -/
  /-
    case neg.refine_2
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra A K
    inst✝¹ : IsFractionRing A K
    inst✝ : IsDedekindDomain A
    I J : FractionalIdeal (nonZeroDivisors A) K
    hJ : Not (Eq J 0)
    ⊢ LE.le (HMul.hMul (HMul.hMul I (Inv.inv J)) J) I
  -/
  rw [mul_assoc, mul_comm J⁻¹, FractionalIdeal.mul_inv_cancel hJ, mul_one]
  /-
    🎉 no goals
  -/


/-- `IsDedekindDomain` and `IsDedekindDomainInv` are equivalent ways
to express that an integral domain is a Dedekind domain. -/
theorem isDedekindDomain_iff_isDedekindDomainInv [IsDomain A] :
    IsDedekindDomain A ↔ IsDedekindDomainInv A :=
  ⟨fun _h _I hI => FractionalIdeal.mul_inv_cancel hI, fun h => h.isDedekindDomain⟩


noncomputable instance FractionalIdeal.semifield : Semifield (FractionalIdeal A⁰ K) where
  __ := coeIdeal_injective.nontrivial
  inv_zero := inv_zero' _
  div_eq_mul_inv := FractionalIdeal.div_eq_mul_inv
  mul_inv_cancel _ := FractionalIdeal.mul_inv_cancel
  nnqsmul := _
  nnqsmul_def := fun _ _ => rfl


/-- Fractional ideals have cancellative multiplication in a Dedekind domain.

Although this instance is a direct consequence of the instance
`FractionalIdeal.semifield`, we define this instance to provide
a computable alternative.
-/
instance FractionalIdeal.cancelCommMonoidWithZero :
    CancelCommMonoidWithZero (FractionalIdeal A⁰ K) where
  __ : CommSemiring (FractionalIdeal A⁰ K) := inferInstance


instance Ideal.cancelCommMonoidWithZero : CancelCommMonoidWithZero (Ideal A) :=
  { Function.Injective.cancelCommMonoidWithZero (coeIdealHom A⁰ (FractionRing A)) coeIdeal_injective
    (RingHom.map_zero _) (RingHom.map_one _) (RingHom.map_mul _) (RingHom.map_pow _) with }

-- Porting note: Lean can infer all it needs by itself

instance Ideal.isDomain : IsDomain (Ideal A) := { }


/-- For ideals in a Dedekind domain, to divide is to contain. -/
theorem Ideal.dvd_iff_le {I J : Ideal A} : I ∣ J ↔ J ≤ I :=
  ⟨Ideal.le_of_dvd, fun h => by
    /-
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      I J : Ideal A
      h : LE.le J I
      ⊢ Dvd.dvd I J
    -/
    by_cases hI : I = ⊥
      /-
        case pos
        A : Type u_2
        inst✝¹ : CommRing A
        inst✝ : IsDedekindDomain A
        I J : Ideal A
        h : LE.le J I
        hI : Eq I Bot.bot
        ⊢ Dvd.dvd I J
      -/
    · have hJ : J = ⊥ := by rwa [hI, ← eq_bot_iff] at h
      /-
        case pos
        A : Type u_2
        inst✝¹ : CommRing A
        inst✝ : IsDedekindDomain A
        I J : Ideal A
        h : LE.le J I
        hI : Eq I Bot.bot
        hJ : Eq J Bot.bot
        ⊢ Dvd.dvd I J
      -/
      rw [hI, hJ]
      /-
        🎉 no goals
      -/
    /-
      case neg
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      I J : Ideal A
      h : LE.le J I
      hI : Not (Eq I Bot.bot)
      ⊢ Dvd.dvd I J
    -/
    have hI' : (I : FractionalIdeal A⁰ (FractionRing A)) ≠ 0 := coeIdeal_ne_zero.mpr hI
    have : (I : FractionalIdeal A⁰ (FractionRing A))⁻¹ * J ≤ 1 :=
      le_trans (mul_left_mono (↑I)⁻¹ ((coeIdeal_le_coeIdeal _).mpr h))
        (le_of_eq (inv_mul_cancel₀ hI'))
    /-
      case neg
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      I J : Ideal A
      h : LE.le J I
      hI : Not (Eq I Bot.bot)
      hI' : Ne (↑I) 0
      this : LE.le (HMul.hMul (Inv.inv ↑I) ↑J) 1
      ⊢ Dvd.dvd I J
    -/
    obtain ⟨H, hH⟩ := le_one_iff_exists_coeIdeal.mp this
    /-
      case neg.intro
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      I J : Ideal A
      h : LE.le J I
      hI : Not (Eq I Bot.bot)
      hI' : Ne (↑I) 0
      this : LE.le (HMul.hMul (Inv.inv ↑I) ↑J) 1
      H : Ideal A
      hH : Eq (↑H) (HMul.hMul (Inv.inv ↑I) ↑J)
      ⊢ Dvd.dvd I J
    -/
    use H
    /-
      case h
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      I J : Ideal A
      h : LE.le J I
      hI : Not (Eq I Bot.bot)
      hI' : Ne (↑I) 0
      this : LE.le (HMul.hMul (Inv.inv ↑I) ↑J) 1
      H : Ideal A
      hH : Eq (↑H) (HMul.hMul (Inv.inv ↑I) ↑J)
      ⊢ Eq J (HMul.hMul I H)
    -/
    refine coeIdeal_injective (show (J : FractionalIdeal A⁰ (FractionRing A)) = ↑(I * H) from ?_)
    /-
      case h
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      I J : Ideal A
      h : LE.le J I
      hI : Not (Eq I Bot.bot)
      hI' : Ne (↑I) 0
      this : LE.le (HMul.hMul (Inv.inv ↑I) ↑J) 1
      H : Ideal A
      hH : Eq (↑H) (HMul.hMul (Inv.inv ↑I) ↑J)
      ⊢ Eq ↑J ↑(HMul.hMul I H)
    -/
    rw [coeIdeal_mul, hH, ← mul_assoc, mul_inv_cancel₀ hI', one_mul]⟩
    /-
      🎉 no goals
    -/


theorem Ideal.dvdNotUnit_iff_lt {I J : Ideal A} : DvdNotUnit I J ↔ J < I :=
  ⟨fun ⟨hI, H, hunit, hmul⟩ =>
    lt_of_le_of_ne (Ideal.dvd_iff_le.mp ⟨H, hmul⟩)
      (mt
        (fun h =>
                                                  /-
                                                    A : Type u_2
                                                    inst✝¹ : CommRing A
                                                    inst✝ : IsDedekindDomain A
                                                    I J : Ideal A
                                                    x✝ : DvdNotUnit I J
                                                    hI : Ne I 0
                                                    H : Ideal A
                                                    hunit : Not (IsUnit H)
                                                    hmul : Eq J (HMul.hMul I H)
                                                    h : Eq J I
                                                    ⊢ Eq (HMul.hMul I H) (HMul.hMul I 1)
                                                  -/
          have : H = 1 := mul_left_cancel₀ hI (by rw [← hmul, h, mul_one])
                                                  /-
                                                    🎉 no goals
                                                  -/
          show IsUnit H from this.symm ▸ isUnit_one)
        hunit),
    fun h =>
    dvdNotUnit_of_dvd_of_not_dvd (Ideal.dvd_iff_le.mpr (le_of_lt h))
      (mt Ideal.dvd_iff_le.mp (not_le_of_lt h))⟩


instance : WfDvdMonoid (Ideal A) where
  wf := by
    /-
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : IsDedekindDomain A
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      ⊢ WellFounded DvdNotUnit
    -/
    have : WellFoundedGT (Ideal A) := inferInstance
    /-
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : IsDedekindDomain A
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      this : WellFoundedGT (Ideal A)
      ⊢ WellFounded DvdNotUnit
    -/
    convert this.wf
    /-
      case h.e'_2.h.h.h.e
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : IsDedekindDomain A
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      this : WellFoundedGT (Ideal A)
      x✝¹ x✝ : Ideal A
      ⊢ Eq DvdNotUnit GT.gt
    -/
    ext
    /-
      case h.e'_2.h.h.h.e.h.h.a
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : IsDedekindDomain A
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      this : WellFoundedGT (Ideal A)
      x✝³ x✝² x✝¹ x✝ : Ideal A
      ⊢ Iff (DvdNotUnit x✝¹ x✝) (GT.gt x✝¹ x✝)
    -/
    rw [Ideal.dvdNotUnit_iff_lt]
    /-
      🎉 no goals
    -/


instance Ideal.uniqueFactorizationMonoid : UniqueFactorizationMonoid (Ideal A) :=
  { irreducible_iff_prime := by
      /-
        R : Type u_1
        A : Type u_2
        K : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Field K
        inst✝² : IsDedekindDomain A
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        ⊢ ∀ {a : Ideal A}, Iff (Irreducible a) (Prime a)
      -/
      intro P
      exact ⟨fun hirr => ⟨hirr.ne_zero, hirr.not_unit, fun I J => by
        have : P.IsMaximal := by
          refine ⟨⟨mt Ideal.isUnit_iff.mpr hirr.not_unit, ?_⟩⟩
          intro J hJ
          obtain ⟨_J_ne, H, hunit, P_eq⟩ := Ideal.dvdNotUnit_iff_lt.mpr hJ
          exact Ideal.isUnit_iff.mp ((hirr.isUnit_or_isUnit P_eq).resolve_right hunit)
        rw [Ideal.dvd_iff_le, Ideal.dvd_iff_le, Ideal.dvd_iff_le, SetLike.le_def, SetLike.le_def,
          SetLike.le_def]
        contrapose!
        rintro ⟨⟨x, x_mem, x_not_mem⟩, ⟨y, y_mem, y_not_mem⟩⟩
        exact
          ⟨x * y, Ideal.mul_mem_mul x_mem y_mem,
            mt this.isPrime.mem_or_mem (not_or_intro x_not_mem y_not_mem)⟩⟩, Prime.irreducible⟩ }


instance Ideal.normalizationMonoid : NormalizationMonoid (Ideal A) := .ofUniqueUnits


@[simp]
theorem Ideal.dvd_span_singleton {I : Ideal A} {x : A} : I ∣ Ideal.span {x} ↔ x ∈ I :=
  Ideal.dvd_iff_le.trans (Ideal.span_le.trans Set.singleton_subset_iff)


theorem Ideal.isPrime_of_prime {P : Ideal A} (h : Prime P) : IsPrime P := by
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    P : Ideal A
    h : Prime P
    ⊢ P.IsPrime
  -/
  refine ⟨?_, fun hxy => ?_⟩
    /-
      case refine_1
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      P : Ideal A
      h : Prime P
      ⊢ Ne P Top.top
    -/
  · rintro rfl
    /-
      case refine_1
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      h : Prime Top.top
      ⊢ False
    -/
    rw [← Ideal.one_eq_top] at h
    /-
      case refine_1
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      h : Prime 1
      ⊢ False
    -/
    exact h.not_unit isUnit_one
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      P : Ideal A
      h : Prime P
      x✝ y✝ : A
      hxy : Membership.mem P (HMul.hMul x✝ y✝)
      ⊢ Or (Membership.mem P x✝) (Membership.mem P y✝)
    -/
  · simp only [← Ideal.dvd_span_singleton, ← Ideal.span_singleton_mul_span_singleton] at hxy ⊢
    /-
      case refine_2
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      P : Ideal A
      h : Prime P
      x✝ y✝ : A
      hxy : Dvd.dvd P (HMul.hMul (Ideal.span (Singleton.singleton x✝)) (Ideal.span ( …
      ⊢ Or (Dvd.dvd P (Ideal.span (Singleton.singleton x✝))) (Dvd.dvd P (Ideal.span  …
    -/
    exact h.dvd_or_dvd hxy
    /-
      🎉 no goals
    -/


theorem Ideal.prime_of_isPrime {P : Ideal A} (hP : P ≠ ⊥) (h : IsPrime P) : Prime P := by
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    P : Ideal A
    hP : Ne P Bot.bot
    h : P.IsPrime
    ⊢ Prime P
  -/
  refine ⟨hP, mt Ideal.isUnit_iff.mp h.ne_top, fun I J hIJ => ?_⟩
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    P : Ideal A
    hP : Ne P Bot.bot
    h : P.IsPrime
    I J : Ideal A
    hIJ : Dvd.dvd P (HMul.hMul I J)
    ⊢ Or (Dvd.dvd P I) (Dvd.dvd P J)
  -/
  simpa only [Ideal.dvd_iff_le] using h.mul_le.mp (Ideal.le_of_dvd hIJ)
  /-
    🎉 no goals
  -/


/-- In a Dedekind domain, the (nonzero) prime elements of the monoid with zero `Ideal A`
are exactly the prime ideals. -/
theorem Ideal.prime_iff_isPrime {P : Ideal A} (hP : P ≠ ⊥) : Prime P ↔ IsPrime P :=
  ⟨Ideal.isPrime_of_prime, Ideal.prime_of_isPrime hP⟩


/-- In a Dedekind domain, the prime ideals are the zero ideal together with the prime elements
of the monoid with zero `Ideal A`. -/
theorem Ideal.isPrime_iff_bot_or_prime {P : Ideal A} : IsPrime P ↔ P = ⊥ ∨ Prime P :=
  ⟨fun hp => (eq_or_ne P ⊥).imp_right fun hp0 => Ideal.prime_of_isPrime hp0 hp, fun hp =>
    hp.elim (fun h => h.symm ▸ Ideal.bot_prime) Ideal.isPrime_of_prime⟩


@[simp]
theorem Ideal.prime_span_singleton_iff {a : A} : Prime (Ideal.span {a}) ↔ Prime a := by
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    a : A
    ⊢ Iff (Prime (Ideal.span (Singleton.singleton a))) (Prime a)
  -/
  rcases eq_or_ne a 0 with rfl | ha
    /-
      case inl
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      ⊢ Iff (Prime (Ideal.span (Singleton.singleton 0))) (Prime 0)
    -/
  · rw [Set.singleton_zero, span_zero, ← Ideal.zero_eq_bot, ← not_iff_not]
    /-
      case inl
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      ⊢ Iff (Not (Prime 0)) (Not (Prime 0))
    -/
    simp only [not_prime_zero, not_false_eq_true]
    /-
      🎉 no goals
    -/
    /-
      case inr
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      a : A
      ha : Ne a 0
      ⊢ Iff (Prime (Ideal.span (Singleton.singleton a))) (Prime a)
    -/
  · have ha' : span {a} ≠ ⊥ := by simpa only [ne_eq, span_singleton_eq_bot] using ha
    /-
      case inr
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      a : A
      ha : Ne a 0
      ha' : Ne (Ideal.span (Singleton.singleton a)) Bot.bot
      ⊢ Iff (Prime (Ideal.span (Singleton.singleton a))) (Prime a)
    -/
    rw [Ideal.prime_iff_isPrime ha', Ideal.span_singleton_prime ha]
    /-
      🎉 no goals
    -/


open Submodule.IsPrincipal in
theorem Ideal.prime_generator_of_prime {P : Ideal A} (h : Prime P) [P.IsPrincipal] :
    Prime (generator P) :=
  have : Ideal.IsPrime P := Ideal.isPrime_of_prime h
  prime_generator_of_isPrime _ h.ne_zero


open UniqueFactorizationMonoid in
nonrec theorem Ideal.mem_normalizedFactors_iff {p I : Ideal A} (hI : I ≠ ⊥) :
    p ∈ normalizedFactors I ↔ p.IsPrime ∧ I ≤ p := by
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    p I : Ideal A
    hI : Ne I Bot.bot
    ⊢ Iff (Membership.mem (UniqueFactorizationMonoid.normalizedFactors I) p) (And  …
  -/
  rw [← Ideal.dvd_iff_le]
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    p I : Ideal A
    hI : Ne I Bot.bot
    ⊢ Iff (Membership.mem (UniqueFactorizationMonoid.normalizedFactors I) p) (And  …
  -/
  by_cases hp : p = 0
    /-
      case pos
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      p I : Ideal A
      hI : Ne I Bot.bot
      hp : Eq p 0
      ⊢ Iff (Membership.mem (UniqueFactorizationMonoid.normalizedFactors I) p) (And  …
    -/
  · rw [← zero_eq_bot] at hI
    simp only [hp, zero_not_mem_normalizedFactors, zero_dvd_iff, hI, false_iff, not_and,
      not_false_eq_true, implies_true]
    /-
      case neg
      A : Type u_2
      inst✝¹ : CommRing A
      inst✝ : IsDedekindDomain A
      p I : Ideal A
      hI : Ne I Bot.bot
      hp : Not (Eq p 0)
      ⊢ Iff (Membership.mem (UniqueFactorizationMonoid.normalizedFactors I) p) (And  …
    -/
  · rwa [mem_normalizedFactors_iff hI, prime_iff_isPrime]
    /-
      🎉 no goals
    -/


theorem Ideal.pow_right_strictAnti (I : Ideal A) (hI0 : I ≠ ⊥) (hI1 : I ≠ ⊤) :
    StrictAnti (I ^ · : ℕ → Ideal A) :=
  strictAnti_nat_of_succ_lt fun e =>
    Ideal.dvdNotUnit_iff_lt.mp ⟨pow_ne_zero _ hI0, I, mt isUnit_iff.mp hI1, pow_succ I e⟩


theorem Ideal.pow_lt_self (I : Ideal A) (hI0 : I ≠ ⊥) (hI1 : I ≠ ⊤) (e : ℕ) (he : 2 ≤ e) :
    I ^ e < I := by
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    hI1 : Ne I Top.top
    e : Nat
    he : LE.le 2 e
    ⊢ LT.lt (HPow.hPow I e) I
  -/
  convert I.pow_right_strictAnti hI0 hI1 he
  /-
    case h.e'_4
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    hI1 : Ne I Top.top
    e : Nat
    he : LE.le 2 e
    ⊢ Eq I ((fun x => HPow.hPow I x) 1)
  -/
  dsimp only
  /-
    case h.e'_4
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    I : Ideal A
    hI0 : Ne I Bot.bot
    hI1 : Ne I Top.top
    e : Nat
    he : LE.le 2 e
    ⊢ Eq I (HPow.hPow I 1)
  -/
  rw [pow_one]
  /-
    🎉 no goals
  -/


theorem Ideal.exists_mem_pow_not_mem_pow_succ (I : Ideal A) (hI0 : I ≠ ⊥) (hI1 : I ≠ ⊤) (e : ℕ) :
    ∃ x ∈ I ^ e, x ∉ I ^ (e + 1) :=
  SetLike.exists_of_lt (I.pow_right_strictAnti hI0 hI1 e.lt_succ_self)


theorem Ideal.eq_prime_pow_of_succ_lt_of_le {P I : Ideal A} [P_prime : P.IsPrime] (hP : P ≠ ⊥)
    {i : ℕ} (hlt : P ^ (i + 1) < I) (hle : I ≤ P ^ i) : I = P ^ i := by
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    P I : Ideal A
    P_prime : P.IsPrime
    hP : Ne P Bot.bot
    i : Nat
    hlt : LT.lt (HPow.hPow P (HAdd.hAdd i 1)) I
    hle : LE.le I (HPow.hPow P i)
    ⊢ Eq I (HPow.hPow P i)
  -/
  refine le_antisymm hle ?_
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    P I : Ideal A
    P_prime : P.IsPrime
    hP : Ne P Bot.bot
    i : Nat
    hlt : LT.lt (HPow.hPow P (HAdd.hAdd i 1)) I
    hle : LE.le I (HPow.hPow P i)
    ⊢ LE.le (HPow.hPow P i) I
  -/
  have P_prime' := Ideal.prime_of_isPrime hP P_prime
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    P I : Ideal A
    P_prime : P.IsPrime
    hP : Ne P Bot.bot
    i : Nat
    hlt : LT.lt (HPow.hPow P (HAdd.hAdd i 1)) I
    hle : LE.le I (HPow.hPow P i)
    P_prime' : Prime P
    ⊢ LE.le (HPow.hPow P i) I
  -/
  have h1 : I ≠ ⊥ := (lt_of_le_of_lt bot_le hlt).ne'
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    P I : Ideal A
    P_prime : P.IsPrime
    hP : Ne P Bot.bot
    i : Nat
    hlt : LT.lt (HPow.hPow P (HAdd.hAdd i 1)) I
    hle : LE.le I (HPow.hPow P i)
    P_prime' : Prime P
    h1 : Ne I Bot.bot
    ⊢ LE.le (HPow.hPow P i) I
  -/
  have := pow_ne_zero i hP
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    P I : Ideal A
    P_prime : P.IsPrime
    hP : Ne P Bot.bot
    i : Nat
    hlt : LT.lt (HPow.hPow P (HAdd.hAdd i 1)) I
    hle : LE.le I (HPow.hPow P i)
    P_prime' : Prime P
    h1 : Ne I Bot.bot
    this : Ne (HPow.hPow P i) 0
    ⊢ LE.le (HPow.hPow P i) I
  -/
  have h3 := pow_ne_zero (i + 1) hP
  rw [← Ideal.dvdNotUnit_iff_lt, dvdNotUnit_iff_normalizedFactors_lt_normalizedFactors h1 h3,
    normalizedFactors_pow, normalizedFactors_irreducible P_prime'.irreducible,
    Multiset.nsmul_singleton, Multiset.lt_replicate_succ] at hlt
  rw [← Ideal.dvd_iff_le, dvd_iff_normalizedFactors_le_normalizedFactors, normalizedFactors_pow,
    normalizedFactors_irreducible P_prime'.irreducible, Multiset.nsmul_singleton]
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    P I : Ideal A
    P_prime : P.IsPrime
    hP : Ne P Bot.bot
    i : Nat
    hlt : LE.le (UniqueFactorizationMonoid.normalizedFactors I) (Multiset.replicat …
    hle : LE.le I (HPow.hPow P i)
    P_prime' : Prime P
    h1 : Ne I Bot.bot
    this : Ne (HPow.hPow P i) 0
    h3 : Ne (HPow.hPow P (HAdd.hAdd i 1)) 0
    ⊢ LE.le (UniqueFactorizationMonoid.normalizedFactors I) (Multiset.replicate i  …
  -/
  all_goals assumption
  /-
    🎉 no goals
  -/


theorem Ideal.pow_succ_lt_pow {P : Ideal A} [P_prime : P.IsPrime] (hP : P ≠ ⊥) (i : ℕ) :
    P ^ (i + 1) < P ^ i :=
  lt_of_le_of_ne (Ideal.pow_le_pow_right (Nat.le_succ _))
    (mt (pow_inj_of_not_isUnit (mt Ideal.isUnit_iff.mp P_prime.ne_top) hP).mp i.succ_ne_self)


theorem Associates.le_singleton_iff (x : A) (n : ℕ) (I : Ideal A) :
    Associates.mk I ^ n ≤ Associates.mk (Ideal.span {x}) ↔ x ∈ I ^ n := by
  simp_rw [← Associates.dvd_eq_le, ← Associates.mk_pow, Associates.mk_dvd_mk,
    Ideal.dvd_span_singleton]


lemma FractionalIdeal.le_inv_comm {I J : FractionalIdeal A⁰ K} (hI : I ≠ 0) (hJ : J ≠ 0) :
    I ≤ J⁻¹ ↔ J ≤ I⁻¹ := by
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : IsDedekindDomain A
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    I J : FractionalIdeal (nonZeroDivisors A) K
    hI : Ne I 0
    hJ : Ne J 0
    ⊢ Iff (LE.le I (Inv.inv J)) (LE.le J (Inv.inv I))
  -/
  rw [inv_eq, inv_eq, le_div_iff_mul_le hI, le_div_iff_mul_le hJ, mul_comm]
  /-
    🎉 no goals
  -/


lemma FractionalIdeal.inv_le_comm {I J : FractionalIdeal A⁰ K} (hI : I ≠ 0) (hJ : J ≠ 0) :
    I⁻¹ ≤ J ↔ J⁻¹ ≤ I := by
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : IsDedekindDomain A
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    I J : FractionalIdeal (nonZeroDivisors A) K
    hI : Ne I 0
    hJ : Ne J 0
    ⊢ Iff (LE.le (Inv.inv I) J) (LE.le (Inv.inv J) I)
  -/
  simpa using le_inv_comm (A := A) (K := K) (inv_ne_zero hI) (inv_ne_zero hJ)
  /-
    🎉 no goals
  -/


/-- Strengthening of `IsLocalization.exist_integer_multiples`:
Let `J ≠ ⊤` be an ideal in a Dedekind domain `A`, and `f ≠ 0` a finite collection
of elements of `K = Frac(A)`, then we can multiply the elements of `f` by some `a : K`
to find a collection of elements of `A` that is not completely contained in `J`. -/
theorem Ideal.exist_integer_multiples_not_mem {J : Ideal A} (hJ : J ≠ ⊤) {ι : Type*} (s : Finset ι)
    (f : ι → K) {j} (hjs : j ∈ s) (hjf : f j ≠ 0) :
    ∃ a : K,
      (∀ i ∈ s, IsLocalization.IsInteger A (a * f i)) ∧
        ∃ i ∈ s, a * f i ∉ (J : FractionalIdeal A⁰ K) := by
  -- Consider the fractional ideal `I` spanned by the `f`s.
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : IsDedekindDomain A
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    J : Ideal A
    hJ : Ne J Top.top
    ι : Type u_4
    s : Finset ι
    f : ι → K
    j : ι
    hjs : Membership.mem s j
    hjf : Ne (f j) 0
    ⊢ Exists fun a => And (∀ (i : ι), Membership.mem s i → IsLocalization.IsIntege …
  -/
  let I : FractionalIdeal A⁰ K := spanFinset A s f
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : IsDedekindDomain A
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    J : Ideal A
    hJ : Ne J Top.top
    ι : Type u_4
    s : Finset ι
    f : ι → K
    j : ι
    hjs : Membership.mem s j
    hjf : Ne (f j) 0
    I : FractionalIdeal (nonZeroDivisors A) K := FractionalIdeal.spanFinset A s f
    ⊢ Exists fun a => And (∀ (i : ι), Membership.mem s i → IsLocalization.IsIntege …
  -/
  have hI0 : I ≠ 0 := spanFinset_ne_zero.mpr ⟨j, hjs, hjf⟩
  -- We claim the multiplier `a` we're looking for is in `I⁻¹ \ (J / I)`.
  suffices ↑J / I < I⁻¹ by
    obtain ⟨_, a, hI, hpI⟩ := SetLike.lt_iff_le_and_exists.mp this
    rw [mem_inv_iff hI0] at hI
    refine ⟨a, fun i hi => ?_, ?_⟩
    -- By definition, `a ∈ I⁻¹` multiplies elements of `I` into elements of `1`,
    -- in other words, `a * f i` is an integer.
    · exact (mem_one_iff _).mp (hI (f i) (Submodule.subset_span (Set.mem_image_of_mem f hi)))
    · contrapose! hpI
      -- And if all `a`-multiples of `I` are an element of `J`,
      -- then `a` is actually an element of `J / I`, contradiction.
      refine (mem_div_iff_of_nonzero hI0).mpr fun y hy => Submodule.span_induction ?_ ?_ ?_ ?_ hy
      · rintro _ ⟨i, hi, rfl⟩; exact hpI i hi
      · rw [mul_zero]; exact Submodule.zero_mem _
      · intro x y _ _ hx hy; rw [mul_add]; exact Submodule.add_mem _ hx hy
      · intro b x _ hx; rw [mul_smul_comm]; exact Submodule.smul_mem _ b hx
  -- To show the inclusion of `J / I` into `I⁻¹ = 1 / I`, note that `J < I`.
  calc
    ↑J / I = ↑J * I⁻¹ := div_eq_mul_inv (↑J) I
    _ < 1 * I⁻¹ := mul_right_strictMono (inv_ne_zero hI0) ?_
    _ = I⁻¹ := one_mul _
  /-
    A : Type u_2
    K : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : IsDedekindDomain A
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    J : Ideal A
    hJ : Ne J Top.top
    ι : Type u_4
    s : Finset ι
    f : ι → K
    j : ι
    hjs : Membership.mem s j
    hjf : Ne (f j) 0
    I : FractionalIdeal (nonZeroDivisors A) K := FractionalIdeal.spanFinset A s f
    hI0 : Ne I 0
    ⊢ LT.lt (↑J) 1
  -/
  rw [← coeIdeal_top]
  -- And multiplying by `I⁻¹` is indeed strictly monotone.
  exact
    strictMono_of_le_iff_le (fun _ _ => (coeIdeal_le_coeIdeal K).symm)
      (lt_top_iff_ne_top.mpr hJ)


@[simp]
theorem sup_mul_inf (I J : Ideal A) : (I ⊔ J) * (I ⊓ J) = I * J := by
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    I J : Ideal A
    ⊢ Eq (HMul.hMul (Max.max I J) (Min.min I J)) (HMul.hMul I J)
  -/
  letI := UniqueFactorizationMonoid.toNormalizedGCDMonoid (Ideal A)
  have hgcd : gcd I J = I ⊔ J := by
    rw [gcd_eq_normalize _ _, normalize_eq]
    · rw [dvd_iff_le, sup_le_iff, ← dvd_iff_le, ← dvd_iff_le]
      exact ⟨gcd_dvd_left _ _, gcd_dvd_right _ _⟩
    · rw [dvd_gcd_iff, dvd_iff_le, dvd_iff_le]
      simp
  have hlcm : lcm I J = I ⊓ J := by
    rw [lcm_eq_normalize _ _, normalize_eq]
    · rw [lcm_dvd_iff, dvd_iff_le, dvd_iff_le]
      simp
    · rw [dvd_iff_le, le_inf_iff, ← dvd_iff_le, ← dvd_iff_le]
      exact ⟨dvd_lcm_left _ _, dvd_lcm_right _ _⟩
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    I J : Ideal A
    this : NormalizedGCDMonoid (Ideal A) := UniqueFactorizationMonoid.toNormalized …
    hgcd : Eq (GCDMonoid.gcd I J) (Max.max I J)
    hlcm : Eq (GCDMonoid.lcm I J) (Min.min I J)
    ⊢ Eq (HMul.hMul (Max.max I J) (Min.min I J)) (HMul.hMul I J)
  -/
  rw [← hgcd, ← hlcm, associated_iff_eq.mp (gcd_mul_lcm _ _)]
  /-
    🎉 no goals
  -/


/-- Ideals in a Dedekind domain have gcd and lcm operators that (trivially) are compatible with
the normalization operator. -/
instance : NormalizedGCDMonoid (Ideal A) :=
  { Ideal.normalizationMonoid with
    gcd := (· ⊔ ·)
                                  /-
                                    R : Type u_1
                                    A : Type u_2
                                    K : Type u_3
                                    inst✝⁵ : CommRing R
                                    inst✝⁴ : CommRing A
                                    inst✝³ : Field K
                                    inst✝² : IsDedekindDomain A
                                    inst✝¹ : Algebra A K
                                    inst✝ : IsFractionRing A K
                                    x✝¹ x✝ : Ideal A
                                    ⊢ Dvd.dvd ((fun x1 x2 => Max.max x1 x2) x✝¹ x✝) x✝¹
                                  -/
    gcd_dvd_left := fun _ _ => by simpa only [dvd_iff_le] using le_sup_left
                                  /-
                                    🎉 no goals
                                  -/
                                   /-
                                     R : Type u_1
                                     A : Type u_2
                                     K : Type u_3
                                     inst✝⁵ : CommRing R
                                     inst✝⁴ : CommRing A
                                     inst✝³ : Field K
                                     inst✝² : IsDedekindDomain A
                                     inst✝¹ : Algebra A K
                                     inst✝ : IsFractionRing A K
                                     x✝¹ x✝ : Ideal A
                                     ⊢ Dvd.dvd ((fun x1 x2 => Max.max x1 x2) x✝¹ x✝) x✝
                                   -/
    gcd_dvd_right := fun _ _ => by simpa only [dvd_iff_le] using le_sup_right
                                   /-
                                     🎉 no goals
                                   -/
    dvd_gcd := by
      /-
        R : Type u_1
        A : Type u_2
        K : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Field K
        inst✝² : IsDedekindDomain A
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        ⊢ ∀ {a b c : Ideal A}, Dvd.dvd a c → Dvd.dvd a b → Dvd.dvd a ((fun x1 x2 => Ma …
      -/
      simp only [dvd_iff_le]
      /-
        R : Type u_1
        A : Type u_2
        K : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Field K
        inst✝² : IsDedekindDomain A
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        ⊢ ∀ {a b c : Ideal A}, LE.le c a → LE.le b a → LE.le (Max.max c b) a
      -/
      exact fun h1 h2 => @sup_le (Ideal A) _ _ _ _ h1 h2
      /-
        🎉 no goals
      -/
    lcm := (· ⊓ ·)
                                 /-
                                   R : Type u_1
                                   A : Type u_2
                                   K : Type u_3
                                   inst✝⁵ : CommRing R
                                   inst✝⁴ : CommRing A
                                   inst✝³ : Field K
                                   inst✝² : IsDedekindDomain A
                                   inst✝¹ : Algebra A K
                                   inst✝ : IsFractionRing A K
                                   x✝ : Ideal A
                                   ⊢ Eq ((fun x1 x2 => Min.min x1 x2) 0 x✝) 0
                                 -/
    lcm_zero_left := fun _ => by simp only [zero_eq_bot, bot_inf_eq]
                                 /-
                                   R : Type u_1
                                   A : Type u_2
                                   K : Type u_3
                                   inst✝⁵ : CommRing R
                                   inst✝⁴ : CommRing A
                                   inst✝³ : Field K
                                   inst✝² : IsDedekindDomain A
                                   inst✝¹ : Algebra A K
                                   inst✝ : IsFractionRing A K
                                   x✝¹ x✝ : Ideal A
                                   ⊢ Associated (HMul.hMul ((fun x1 x2 => Max.max x1 x2) x✝¹ x✝) ((fun x1 x2 => M …
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
                                  /-
                                    R : Type u_1
                                    A : Type u_2
                                    K : Type u_3
                                    inst✝⁵ : CommRing R
                                    inst✝⁴ : CommRing A
                                    inst✝³ : Field K
                                    inst✝² : IsDedekindDomain A
                                    inst✝¹ : Algebra A K
                                    inst✝ : IsFractionRing A K
                                    x✝ : Ideal A
                                    ⊢ Eq ((fun x1 x2 => Min.min x1 x2) x✝ 0) 0
                                  -/
    lcm_zero_right := fun _ => by simp only [zero_eq_bot, inf_bot_eq]
                                  /-
                                    🎉 no goals
                                  -/
    gcd_mul_lcm := fun _ _ => by rw [associated_iff_eq, sup_mul_inf]
    normalize_gcd := fun _ _ => normalize_eq _
    normalize_lcm := fun _ _ => normalize_eq _ }

-- In fact, any lawful gcd and lcm would equal sup and inf respectively.

@[simp]
theorem gcd_eq_sup (I J : Ideal A) : gcd I J = I ⊔ J := rfl


@[simp]
theorem lcm_eq_inf (I J : Ideal A) : lcm I J = I ⊓ J := rfl


theorem isCoprime_iff_gcd {I J : Ideal A} : IsCoprime I J ↔ gcd I J = 1 := by
  /-
    A : Type u_2
    inst✝¹ : CommRing A
    inst✝ : IsDedekindDomain A
    I J : Ideal A
    ⊢ Iff (IsCoprime I J) (Eq (GCDMonoid.gcd I J) 1)
  -/
  rw [Ideal.isCoprime_iff_codisjoint, codisjoint_iff, one_eq_top, gcd_eq_sup]
  /-
    🎉 no goals
  -/


theorem factors_span_eq {p : K[X]} : factors (span {p}) = (factors p).map (fun q ↦ span {q}) := by
  /-
    K : Type u_3
    inst✝ : Field K
    p : Polynomial K
    ⊢ Eq (UniqueFactorizationMonoid.factors (Ideal.span (Singleton.singleton p)))  …
  -/
  rcases eq_or_ne p 0 with rfl | hp; · simpa [Set.singleton_zero] using normalizedFactors_zero
                                       /-
                                         🎉 no goals
                                       -/
  have : ∀ q ∈ (factors p).map (fun q ↦ span {q}), Prime q := fun q hq ↦ by
    obtain ⟨r, hr, rfl⟩ := Multiset.mem_map.mp hq
    exact prime_span_singleton_iff.mpr <| prime_of_factor r hr
  rw [← span_singleton_eq_span_singleton.mpr (factors_prod hp), ← multiset_prod_span_singleton,
    factors_eq_normalizedFactors, normalizedFactors_prod_of_prime this]


theorem prod_normalizedFactors_eq_self (hI : I ≠ ⊥) : (normalizedFactors I).prod = I :=
  associated_iff_eq.1 (prod_normalizedFactors hI)


theorem count_le_of_ideal_ge [DecidableEq (Ideal T)]
    {I J : Ideal T} (h : I ≤ J) (hI : I ≠ ⊥) (K : Ideal T) :
    count K (normalizedFactors J) ≤ count K (normalizedFactors I) :=
  le_iff_count.1 ((dvd_iff_normalizedFactors_le_normalizedFactors (ne_bot_of_le_ne_bot hI h) hI).1
    (dvd_iff_le.2 h))
    _


theorem sup_eq_prod_inf_factors [DecidableEq (Ideal T)] (hI : I ≠ ⊥) (hJ : J ≠ ⊥) :
    I ⊔ J = (normalizedFactors I ∩ normalizedFactors J).prod := by
  have H : normalizedFactors (normalizedFactors I ∩ normalizedFactors J).prod =
      normalizedFactors I ∩ normalizedFactors J := by
    apply normalizedFactors_prod_of_prime
    intro p hp
    rw [mem_inter] at hp
    exact prime_of_normalized_factor p hp.left
  have := Multiset.prod_ne_zero_of_prime (normalizedFactors I ∩ normalizedFactors J) fun _ h =>
      prime_of_normalized_factor _ (Multiset.mem_inter.1 h).1
  /-
    T : Type u_4
    inst✝² : CommRing T
    inst✝¹ : IsDedekindDomain T
    I J : Ideal T
    inst✝ : DecidableEq (Ideal T)
    hI : Ne I Bot.bot
    hJ : Ne J Bot.bot
    H : Eq (UniqueFactorizationMonoid.normalizedFactors (Inter.inter (UniqueFactor …
    this : Ne (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
    ⊢ Eq (Max.max I J) (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I …
  -/
  apply le_antisymm
    /-
      case a
      T : Type u_4
      inst✝² : CommRing T
      inst✝¹ : IsDedekindDomain T
      I J : Ideal T
      inst✝ : DecidableEq (Ideal T)
      hI : Ne I Bot.bot
      hJ : Ne J Bot.bot
      H : Eq (UniqueFactorizationMonoid.normalizedFactors (Inter.inter (UniqueFactor …
      this : Ne (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
      ⊢ LE.le (Max.max I J) (Inter.inter (UniqueFactorizationMonoid.normalizedFactor …
    -/
  · rw [sup_le_iff, ← dvd_iff_le, ← dvd_iff_le]
    /-
      case a
      T : Type u_4
      inst✝² : CommRing T
      inst✝¹ : IsDedekindDomain T
      I J : Ideal T
      inst✝ : DecidableEq (Ideal T)
      hI : Ne I Bot.bot
      hJ : Ne J Bot.bot
      H : Eq (UniqueFactorizationMonoid.normalizedFactors (Inter.inter (UniqueFactor …
      this : Ne (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
      ⊢ And (Dvd.dvd (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (U …
    -/
    constructor
      /-
        case a.left
        T : Type u_4
        inst✝² : CommRing T
        inst✝¹ : IsDedekindDomain T
        I J : Ideal T
        inst✝ : DecidableEq (Ideal T)
        hI : Ne I Bot.bot
        hJ : Ne J Bot.bot
        H : Eq (UniqueFactorizationMonoid.normalizedFactors (Inter.inter (UniqueFactor …
        this : Ne (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
        ⊢ Dvd.dvd (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
      -/
    · rw [dvd_iff_normalizedFactors_le_normalizedFactors this hI, H]
      /-
        case a.left
        T : Type u_4
        inst✝² : CommRing T
        inst✝¹ : IsDedekindDomain T
        I J : Ideal T
        inst✝ : DecidableEq (Ideal T)
        hI : Ne I Bot.bot
        hJ : Ne J Bot.bot
        H : Eq (UniqueFactorizationMonoid.normalizedFactors (Inter.inter (UniqueFactor …
        this : Ne (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
        ⊢ LE.le (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (UniqueFa …
      -/
      exact inf_le_left
      /-
        🎉 no goals
      -/
      /-
        case a.right
        T : Type u_4
        inst✝² : CommRing T
        inst✝¹ : IsDedekindDomain T
        I J : Ideal T
        inst✝ : DecidableEq (Ideal T)
        hI : Ne I Bot.bot
        hJ : Ne J Bot.bot
        H : Eq (UniqueFactorizationMonoid.normalizedFactors (Inter.inter (UniqueFactor …
        this : Ne (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
        ⊢ Dvd.dvd (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
      -/
    · rw [dvd_iff_normalizedFactors_le_normalizedFactors this hJ, H]
      /-
        case a.right
        T : Type u_4
        inst✝² : CommRing T
        inst✝¹ : IsDedekindDomain T
        I J : Ideal T
        inst✝ : DecidableEq (Ideal T)
        hI : Ne I Bot.bot
        hJ : Ne J Bot.bot
        H : Eq (UniqueFactorizationMonoid.normalizedFactors (Inter.inter (UniqueFactor …
        this : Ne (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
        ⊢ LE.le (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (UniqueFa …
      -/
      exact inf_le_right
      /-
        🎉 no goals
      -/
  · rw [← dvd_iff_le, dvd_iff_normalizedFactors_le_normalizedFactors,
      normalizedFactors_prod_of_prime, le_iff_count]
      /-
        case a
        T : Type u_4
        inst✝² : CommRing T
        inst✝¹ : IsDedekindDomain T
        I J : Ideal T
        inst✝ : DecidableEq (Ideal T)
        hI : Ne I Bot.bot
        hJ : Ne J Bot.bot
        H : Eq (UniqueFactorizationMonoid.normalizedFactors (Inter.inter (UniqueFactor …
        this : Ne (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
        ⊢ ∀ (a : Ideal T), LE.le (Multiset.count a (UniqueFactorizationMonoid.normaliz …
      -/
    · intro a
      /-
        case a
        T : Type u_4
        inst✝² : CommRing T
        inst✝¹ : IsDedekindDomain T
        I J : Ideal T
        inst✝ : DecidableEq (Ideal T)
        hI : Ne I Bot.bot
        hJ : Ne J Bot.bot
        H : Eq (UniqueFactorizationMonoid.normalizedFactors (Inter.inter (UniqueFactor …
        this : Ne (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
        a : Ideal T
        ⊢ LE.le (Multiset.count a (UniqueFactorizationMonoid.normalizedFactors (Max.ma …
      -/
      rw [Multiset.count_inter]
      /-
        case a
        T : Type u_4
        inst✝² : CommRing T
        inst✝¹ : IsDedekindDomain T
        I J : Ideal T
        inst✝ : DecidableEq (Ideal T)
        hI : Ne I Bot.bot
        hJ : Ne J Bot.bot
        H : Eq (UniqueFactorizationMonoid.normalizedFactors (Inter.inter (UniqueFactor …
        this : Ne (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
        a : Ideal T
        ⊢ LE.le (Multiset.count a (UniqueFactorizationMonoid.normalizedFactors (Max.ma …
      -/
      exact le_min (count_le_of_ideal_ge le_sup_left hI a) (count_le_of_ideal_ge le_sup_right hJ a)
      /-
        🎉 no goals
      -/
      /-
        case a
        T : Type u_4
        inst✝² : CommRing T
        inst✝¹ : IsDedekindDomain T
        I J : Ideal T
        inst✝ : DecidableEq (Ideal T)
        hI : Ne I Bot.bot
        hJ : Ne J Bot.bot
        H : Eq (UniqueFactorizationMonoid.normalizedFactors (Inter.inter (UniqueFactor …
        this : Ne (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
        ⊢ ∀ (p : Ideal T), Membership.mem (Inter.inter (UniqueFactorizationMonoid.norm …
      -/
    · intro p hp
      /-
        case a
        T : Type u_4
        inst✝² : CommRing T
        inst✝¹ : IsDedekindDomain T
        I J : Ideal T
        inst✝ : DecidableEq (Ideal T)
        hI : Ne I Bot.bot
        hJ : Ne J Bot.bot
        H : Eq (UniqueFactorizationMonoid.normalizedFactors (Inter.inter (UniqueFactor …
        this : Ne (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
        p : Ideal T
        hp : Membership.mem (Inter.inter (UniqueFactorizationMonoid.normalizedFactors  …
        ⊢ Prime p
      -/
      rw [mem_inter] at hp
      /-
        case a
        T : Type u_4
        inst✝² : CommRing T
        inst✝¹ : IsDedekindDomain T
        I J : Ideal T
        inst✝ : DecidableEq (Ideal T)
        hI : Ne I Bot.bot
        hJ : Ne J Bot.bot
        H : Eq (UniqueFactorizationMonoid.normalizedFactors (Inter.inter (UniqueFactor …
        this : Ne (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
        p : Ideal T
        hp : And (Membership.mem (UniqueFactorizationMonoid.normalizedFactors I) p) (M …
        ⊢ Prime p
      -/
      exact prime_of_normalized_factor p hp.left
      /-
        🎉 no goals
      -/
      /-
        case a.hx
        T : Type u_4
        inst✝² : CommRing T
        inst✝¹ : IsDedekindDomain T
        I J : Ideal T
        inst✝ : DecidableEq (Ideal T)
        hI : Ne I Bot.bot
        hJ : Ne J Bot.bot
        H : Eq (UniqueFactorizationMonoid.normalizedFactors (Inter.inter (UniqueFactor …
        this : Ne (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
        ⊢ Ne (Max.max I J) 0
      -/
    · exact ne_bot_of_le_ne_bot hI le_sup_left
      /-
        🎉 no goals
      -/
      /-
        case a.hy
        T : Type u_4
        inst✝² : CommRing T
        inst✝¹ : IsDedekindDomain T
        I J : Ideal T
        inst✝ : DecidableEq (Ideal T)
        hI : Ne I Bot.bot
        hJ : Ne J Bot.bot
        H : Eq (UniqueFactorizationMonoid.normalizedFactors (Inter.inter (UniqueFactor …
        this : Ne (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (Unique …
        ⊢ Ne (Inter.inter (UniqueFactorizationMonoid.normalizedFactors I) (UniqueFacto …
      -/
    · exact this
      /-
        🎉 no goals
      -/


theorem irreducible_pow_sup [DecidableEq (Ideal T)] (hI : I ≠ ⊥) (hJ : Irreducible J) (n : ℕ) :
    J ^ n ⊔ I = J ^ min ((normalizedFactors I).count J) n := by
  rw [sup_eq_prod_inf_factors (pow_ne_zero n hJ.ne_zero) hI, min_comm,
    normalizedFactors_of_irreducible_pow hJ, normalize_eq J, replicate_inter, prod_replicate]


theorem irreducible_pow_sup_of_le (hJ : Irreducible J) (n : ℕ) (hn : n ≤ emultiplicity J I) :
    J ^ n ⊔ I = J ^ n := by
  classical
  by_cases hI : I = ⊥
  · simp_all
  rw [irreducible_pow_sup hI hJ, min_eq_right]
  rw [emultiplicity_eq_count_normalizedFactors hJ hI, normalize_eq J] at hn
  exact_mod_cast hn


theorem irreducible_pow_sup_of_ge (hI : I ≠ ⊥) (hJ : Irreducible J) (n : ℕ)
    (hn : emultiplicity J I ≤ n) : J ^ n ⊔ I = J ^ multiplicity J I := by
  classical
  rw [irreducible_pow_sup hI hJ, min_eq_left]
  · congr
    rw [← Nat.cast_inj (R := ℕ∞), ← FiniteMultiplicity.emultiplicity_eq_multiplicity,
      emultiplicity_eq_count_normalizedFactors hJ hI, normalize_eq J]
    rw [← emultiplicity_lt_top]
    apply hn.trans_lt
    simp
  · rw [emultiplicity_eq_count_normalizedFactors hJ hI, normalize_eq J] at hn
    exact_mod_cast hn


theorem Ideal.eq_prime_pow_mul_coprime [DecidableEq (Ideal T)] {I : Ideal T} (hI : I ≠ ⊥)
    (P : Ideal T) [hpm : P.IsMaximal] :
    ∃ Q : Ideal T, P ⊔ Q = ⊤ ∧ I = P ^ (Multiset.count P (normalizedFactors I)) * Q := by
  /-
    T : Type u_4
    inst✝² : CommRing T
    inst✝¹ : IsDedekindDomain T
    inst✝ : DecidableEq (Ideal T)
    I : Ideal T
    hI : Ne I Bot.bot
    P : Ideal T
    hpm : P.IsMaximal
    ⊢ Exists fun Q => And (Eq (Max.max P Q) Top.top) (Eq I (HMul.hMul (HPow.hPow P …
  -/
  use (filter (¬ P = ·) (normalizedFactors I)).prod
  /-
    case h
    T : Type u_4
    inst✝² : CommRing T
    inst✝¹ : IsDedekindDomain T
    inst✝ : DecidableEq (Ideal T)
    I : Ideal T
    hI : Ne I Bot.bot
    P : Ideal T
    hpm : P.IsMaximal
    ⊢ And (Eq (Max.max P (Multiset.filter (fun x => Not (Eq P x)) (UniqueFactoriza …
  -/
  constructor
    /-
      case h.left
      T : Type u_4
      inst✝² : CommRing T
      inst✝¹ : IsDedekindDomain T
      inst✝ : DecidableEq (Ideal T)
      I : Ideal T
      hI : Ne I Bot.bot
      P : Ideal T
      hpm : P.IsMaximal
      ⊢ Eq (Max.max P (Multiset.filter (fun x => Not (Eq P x)) (UniqueFactorizationM …
    -/
  · refine P.sup_multiset_prod_eq_top (fun p hpi ↦ ?_)
    /-
      case h.left
      T : Type u_4
      inst✝² : CommRing T
      inst✝¹ : IsDedekindDomain T
      inst✝ : DecidableEq (Ideal T)
      I : Ideal T
      hI : Ne I Bot.bot
      P : Ideal T
      hpm : P.IsMaximal
      p : Ideal T
      hpi : Membership.mem (Multiset.filter (fun x => Not (Eq P x)) (UniqueFactoriza …
      ⊢ Eq (Max.max P p) Top.top
    -/
    have hp : Prime p := prime_of_normalized_factor p (filter_subset _ (normalizedFactors I) hpi)
    /-
      case h.left
      T : Type u_4
      inst✝² : CommRing T
      inst✝¹ : IsDedekindDomain T
      inst✝ : DecidableEq (Ideal T)
      I : Ideal T
      hI : Ne I Bot.bot
      P : Ideal T
      hpm : P.IsMaximal
      p : Ideal T
      hpi : Membership.mem (Multiset.filter (fun x => Not (Eq P x)) (UniqueFactoriza …
      hp : Prime p
      ⊢ Eq (Max.max P p) Top.top
    -/
    exact hpm.coprime_of_ne ((isPrime_of_prime hp).isMaximal hp.ne_zero) (of_mem_filter hpi)
    /-
      🎉 no goals
    -/
    /-
      case h.right
      T : Type u_4
      inst✝² : CommRing T
      inst✝¹ : IsDedekindDomain T
      inst✝ : DecidableEq (Ideal T)
      I : Ideal T
      hI : Ne I Bot.bot
      P : Ideal T
      hpm : P.IsMaximal
      ⊢ Eq I (HMul.hMul (HPow.hPow P (Multiset.count P (UniqueFactorizationMonoid.no …
    -/
  · nth_rw 1 [← prod_normalizedFactors_eq_self hI, ← filter_add_not (P = ·) (normalizedFactors I)]
    /-
      case h.right
      T : Type u_4
      inst✝² : CommRing T
      inst✝¹ : IsDedekindDomain T
      inst✝ : DecidableEq (Ideal T)
      I : Ideal T
      hI : Ne I Bot.bot
      P : Ideal T
      hpm : P.IsMaximal
      ⊢ Eq (HAdd.hAdd (Multiset.filter (fun x => Eq P x) (UniqueFactorizationMonoid. …
    -/
    rw [prod_add, pow_count]
    /-
      🎉 no goals
    -/


/-- The height one prime spectrum of a Dedekind domain `R` is the type of nonzero prime ideals of
`R`. Note that this equals the maximal spectrum if `R` has Krull dimension 1. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed `has_nonempty_instance`, linter doesn't exist yet
@[ext, nolint unusedArguments]
structure HeightOneSpectrum where
  asIdeal : Ideal R
  isPrime : asIdeal.IsPrime
  ne_bot : asIdeal ≠ ⊥


instance isMaximal : v.asIdeal.IsMaximal := v.isPrime.isMaximal v.ne_bot


theorem prime : Prime v.asIdeal := Ideal.prime_of_isPrime v.ne_bot v.isPrime


theorem irreducible : Irreducible v.asIdeal :=
  UniqueFactorizationMonoid.irreducible_iff_prime.mpr v.prime


theorem associates_irreducible : Irreducible <| Associates.mk v.asIdeal :=
  Associates.irreducible_mk.mpr v.irreducible


/-- An equivalence between the height one and maximal spectra for rings of Krull dimension 1. -/
def equivMaximalSpectrum (hR : ¬IsField R) : HeightOneSpectrum R ≃ MaximalSpectrum R where
  toFun v := ⟨v.asIdeal, v.isPrime.isMaximal v.ne_bot⟩
  invFun v :=
    ⟨v.asIdeal, v.IsMaximal.isPrime, Ring.ne_bot_of_isMaximal_of_not_isField v.IsMaximal hR⟩
  left_inv := fun ⟨_, _, _⟩ => rfl
  right_inv := fun ⟨_, _⟩ => rfl


/-- A Dedekind domain is equal to the intersection of its localizations at all its height one
non-zero prime ideals viewed as subalgebras of its field of fractions. -/
theorem iInf_localization_eq_bot [Algebra R K] [hK : IsFractionRing R K] :
    (⨅ v : HeightOneSpectrum R,
        Localization.subalgebra.ofField K _ v.asIdeal.primeCompl_le_nonZeroDivisors) = ⊥ := by
  /-
    R : Type u_1
    K : Type u_3
    inst✝³ : CommRing R
    inst✝² : Field K
    inst✝¹ : IsDedekindDomain R
    inst✝ : Algebra R K
    hK : IsFractionRing R K
    ⊢ Eq (iInf fun v => Localization.subalgebra.ofField K v.asIdeal.primeCompl ⋯)  …
  -/
  ext x
  /-
    case h
    R : Type u_1
    K : Type u_3
    inst✝³ : CommRing R
    inst✝² : Field K
    inst✝¹ : IsDedekindDomain R
    inst✝ : Algebra R K
    hK : IsFractionRing R K
    x : K
    ⊢ Iff (Membership.mem (iInf fun v => Localization.subalgebra.ofField K v.asIde …
  -/
  rw [Algebra.mem_iInf]
  /-
    case h
    R : Type u_1
    K : Type u_3
    inst✝³ : CommRing R
    inst✝² : Field K
    inst✝¹ : IsDedekindDomain R
    inst✝ : Algebra R K
    hK : IsFractionRing R K
    x : K
    ⊢ Iff (∀ (i : IsDedekindDomain.HeightOneSpectrum R), Membership.mem (Localizat …
  -/
  constructor
  /-
    case h.mp
    R : Type u_1
    K : Type u_3
    inst✝³ : CommRing R
    inst✝² : Field K
    inst✝¹ : IsDedekindDomain R
    inst✝ : Algebra R K
    hK : IsFractionRing R K
    x : K
    ⊢ (∀ (i : IsDedekindDomain.HeightOneSpectrum R), Membership.mem (Localization. …
  -/
  on_goal 1 => by_cases hR : IsField R
  · rcases Function.bijective_iff_has_inverse.mp
      (IsField.localization_map_bijective (Rₘ := K) (flip nonZeroDivisors.ne_zero rfl : 0 ∉ R⁰) hR)
      with ⟨algebra_map_inv, _, algebra_map_right_inv⟩
    /-
      case pos.intro.intro
      R : Type u_1
      K : Type u_3
      inst✝³ : CommRing R
      inst✝² : Field K
      inst✝¹ : IsDedekindDomain R
      inst✝ : Algebra R K
      hK : IsFractionRing R K
      x : K
      hR : IsField R
      algebra_map_inv : K → R
      left✝ : Function.LeftInverse algebra_map_inv ⇑(algebraMap R K)
      algebra_map_right_inv : Function.RightInverse algebra_map_inv ⇑(algebraMap R K)
      ⊢ (∀ (i : IsDedekindDomain.HeightOneSpectrum R), Membership.mem (Localization. …
    -/
    exact fun _ => Algebra.mem_bot.mpr ⟨algebra_map_inv x, algebra_map_right_inv x⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    K : Type u_3
    inst✝³ : CommRing R
    inst✝² : Field K
    inst✝¹ : IsDedekindDomain R
    inst✝ : Algebra R K
    hK : IsFractionRing R K
    x : K
    hR : Not (IsField R)
    ⊢ (∀ (i : IsDedekindDomain.HeightOneSpectrum R), Membership.mem (Localization. …
  -/
  all_goals rw [← MaximalSpectrum.iInf_localization_eq_bot, Algebra.mem_iInf]
    /-
      case neg
      R : Type u_1
      K : Type u_3
      inst✝³ : CommRing R
      inst✝² : Field K
      inst✝¹ : IsDedekindDomain R
      inst✝ : Algebra R K
      hK : IsFractionRing R K
      x : K
      hR : Not (IsField R)
      ⊢ (∀ (i : IsDedekindDomain.HeightOneSpectrum R), Membership.mem (Localization. …
    -/
  · exact fun hx ⟨v, hv⟩ => hx ((equivMaximalSpectrum hR).symm ⟨v, hv⟩)
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u_1
      K : Type u_3
      inst✝³ : CommRing R
      inst✝² : Field K
      inst✝¹ : IsDedekindDomain R
      inst✝ : Algebra R K
      hK : IsFractionRing R K
      x : K
      ⊢ (∀ (i : MaximalSpectrum R), Membership.mem (Localization.subalgebra.ofField  …
    -/
  · exact fun hx ⟨v, hv, hbot⟩ => hx ⟨v, hv.isMaximal hbot⟩
    /-
      🎉 no goals
    -/


/-- The map from ideals of `R` dividing `I` to the ideals of `A` dividing `J` induced by
  a homomorphism `f : R/I →+* A/J` -/
@[simps] -- Porting note: use `Subtype` instead of `Set` to make linter happy
def idealFactorsFunOfQuotHom {f : R ⧸ I →+* A ⧸ J} (hf : Function.Surjective f) :
    {p : Ideal R // p ∣ I} →o {p : Ideal A // p ∣ J} where
  toFun X := ⟨comap (Ideal.Quotient.mk J) (map f (map (Ideal.Quotient.mk I) X)), by
    have : RingHom.ker (Ideal.Quotient.mk J) ≤
        comap (Ideal.Quotient.mk J) (map f (map (Ideal.Quotient.mk I) X)) :=
      ker_le_comap (Ideal.Quotient.mk J)
    /-
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Field K
      inst✝ : IsDedekindDomain A
      I : Ideal R
      J : Ideal A
      f : RingHom (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
      hf : Function.Surjective ⇑f
      X : Subtype fun p => Dvd.dvd p I
      this : LE.le (RingHom.ker (Ideal.Quotient.mk J)) (Ideal.comap (Ideal.Quotient. …
      ⊢ Dvd.dvd (Ideal.comap (Ideal.Quotient.mk J) (Ideal.map f (Ideal.map (Ideal.Qu …
    -/
    rw [mk_ker] at this
    /-
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Field K
      inst✝ : IsDedekindDomain A
      I : Ideal R
      J : Ideal A
      f : RingHom (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
      hf : Function.Surjective ⇑f
      X : Subtype fun p => Dvd.dvd p I
      this : LE.le J (Ideal.comap (Ideal.Quotient.mk J) (Ideal.map f (Ideal.map (Ide …
      ⊢ Dvd.dvd (Ideal.comap (Ideal.Quotient.mk J) (Ideal.map f (Ideal.map (Ideal.Qu …
    -/
    exact dvd_iff_le.mpr this⟩
    /-
      🎉 no goals
    -/
  monotone' := by
    /-
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Field K
      inst✝ : IsDedekindDomain A
      I : Ideal R
      J : Ideal A
      f : RingHom (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
      hf : Function.Surjective ⇑f
      ⊢ Monotone fun X => ⟨Ideal.comap (Ideal.Quotient.mk J) (Ideal.map f (Ideal.map …
    -/
    rintro ⟨X, hX⟩ ⟨Y, hY⟩ h
    /-
      case mk.mk
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Field K
      inst✝ : IsDedekindDomain A
      I : Ideal R
      J : Ideal A
      f : RingHom (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
      hf : Function.Surjective ⇑f
      X : Ideal R
      hX : Dvd.dvd X I
      Y : Ideal R
      hY : Dvd.dvd Y I
      h : LE.le ⟨X, hX⟩ ⟨Y, hY⟩
      ⊢ LE.le ((fun X => ⟨Ideal.comap (Ideal.Quotient.mk J) (Ideal.map f (Ideal.map  …
    -/
    rw [← Subtype.coe_le_coe, Subtype.coe_mk, Subtype.coe_mk] at h ⊢
    rw [Subtype.coe_mk, comap_le_comap_iff_of_surjective (Ideal.Quotient.mk J)
      Ideal.Quotient.mk_surjective, map_le_iff_le_comap, Subtype.coe_mk,
      comap_map_of_surjective _ hf (map (Ideal.Quotient.mk I) Y)]
    suffices map (Ideal.Quotient.mk I) X ≤ map (Ideal.Quotient.mk I) Y by
      exact le_sup_of_le_left this
    rwa [map_le_iff_le_comap, comap_map_of_surjective (Ideal.Quotient.mk I)
      Ideal.Quotient.mk_surjective, ← RingHom.ker_eq_comap_bot, mk_ker,
      sup_eq_left.mpr <| le_of_dvd hY]


@[simp]
theorem idealFactorsFunOfQuotHom_id :
    idealFactorsFunOfQuotHom (RingHom.id (A ⧸ J)).surjective = OrderHom.id :=
  OrderHom.ext _ _
    (funext fun X => by
      simp only [idealFactorsFunOfQuotHom, map_id, OrderHom.coe_mk, OrderHom.id_coe, id,
        comap_map_of_surjective (Ideal.Quotient.mk J) Ideal.Quotient.mk_surjective, ←
        RingHom.ker_eq_comap_bot (Ideal.Quotient.mk J), mk_ker,
        sup_eq_left.mpr (dvd_iff_le.mp X.prop), Subtype.coe_eta])


theorem idealFactorsFunOfQuotHom_comp {f : R ⧸ I →+* A ⧸ J} {g : A ⧸ J →+* B ⧸ L}
    (hf : Function.Surjective f) (hg : Function.Surjective g) :
    (idealFactorsFunOfQuotHom hg).comp (idealFactorsFunOfQuotHom hf) =
      idealFactorsFunOfQuotHom (show Function.Surjective (g.comp f) from hg.comp hf) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : IsDedekindDomain A
    I : Ideal R
    J : Ideal A
    B : Type u_4
    inst✝¹ : CommRing B
    inst✝ : IsDedekindDomain B
    L : Ideal B
    f : RingHom (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
    g : RingHom (HasQuotient.Quotient A J) (HasQuotient.Quotient B L)
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    ⊢ Eq ((idealFactorsFunOfQuotHom hg).comp (idealFactorsFunOfQuotHom hf)) (ideal …
  -/
  refine OrderHom.ext _ _ (funext fun x => ?_)
  rw [idealFactorsFunOfQuotHom, idealFactorsFunOfQuotHom, OrderHom.comp_coe, OrderHom.coe_mk,
    OrderHom.coe_mk, Function.comp_apply, idealFactorsFunOfQuotHom, OrderHom.coe_mk,
    Subtype.mk_eq_mk, Subtype.coe_mk, map_comap_of_surjective (Ideal.Quotient.mk J)
    Ideal.Quotient.mk_surjective, map_map]


/-- The bijection between ideals of `R` dividing `I` and the ideals of `A` dividing `J` induced by
  an isomorphism `f : R/I ≅ A/J`. -/
def idealFactorsEquivOfQuotEquiv : { p : Ideal R | p ∣ I } ≃o { p : Ideal A | p ∣ J } := by
  /-
    R : Type u_1
    A : Type u_2
    K : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Field K
    inst✝³ : IsDedekindDomain A
    I : Ideal R
    J : Ideal A
    B : Type u_4
    inst✝² : CommRing B
    inst✝¹ : IsDedekindDomain B
    L : Ideal B
    inst✝ : IsDedekindDomain R
    f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
    ⊢ OrderIso ↑(setOf fun p => Dvd.dvd p I) ↑(setOf fun p => Dvd.dvd p J)
  -/
  have f_surj : Function.Surjective (f : R ⧸ I →+* A ⧸ J) := f.surjective
  /-
    R : Type u_1
    A : Type u_2
    K : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Field K
    inst✝³ : IsDedekindDomain A
    I : Ideal R
    J : Ideal A
    B : Type u_4
    inst✝² : CommRing B
    inst✝¹ : IsDedekindDomain B
    L : Ideal B
    inst✝ : IsDedekindDomain R
    f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
    f_surj : Function.Surjective ⇑↑f
    ⊢ OrderIso ↑(setOf fun p => Dvd.dvd p I) ↑(setOf fun p => Dvd.dvd p J)
  -/
  have fsym_surj : Function.Surjective (f.symm : A ⧸ J →+* R ⧸ I) := f.symm.surjective
  refine OrderIso.ofHomInv (idealFactorsFunOfQuotHom f_surj) (idealFactorsFunOfQuotHom fsym_surj)
    ?_ ?_
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Field K
      inst✝³ : IsDedekindDomain A
      I : Ideal R
      J : Ideal A
      B : Type u_4
      inst✝² : CommRing B
      inst✝¹ : IsDedekindDomain B
      L : Ideal B
      inst✝ : IsDedekindDomain R
      f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
      f_surj : Function.Surjective ⇑↑f
      fsym_surj : Function.Surjective ⇑↑f.symm
      ⊢ Eq ((↑(idealFactorsFunOfQuotHom f_surj)).comp ↑(idealFactorsFunOfQuotHom fsy …
    -/
  · have := idealFactorsFunOfQuotHom_comp fsym_surj f_surj
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Field K
      inst✝³ : IsDedekindDomain A
      I : Ideal R
      J : Ideal A
      B : Type u_4
      inst✝² : CommRing B
      inst✝¹ : IsDedekindDomain B
      L : Ideal B
      inst✝ : IsDedekindDomain R
      f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
      f_surj : Function.Surjective ⇑↑f
      fsym_surj : Function.Surjective ⇑↑f.symm
      this : Eq ((idealFactorsFunOfQuotHom f_surj).comp (idealFactorsFunOfQuotHom fs …
      ⊢ Eq ((↑(idealFactorsFunOfQuotHom f_surj)).comp ↑(idealFactorsFunOfQuotHom fsy …
    -/
    simp only [RingEquiv.comp_symm, idealFactorsFunOfQuotHom_id] at this
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Field K
      inst✝³ : IsDedekindDomain A
      I : Ideal R
      J : Ideal A
      B : Type u_4
      inst✝² : CommRing B
      inst✝¹ : IsDedekindDomain B
      L : Ideal B
      inst✝ : IsDedekindDomain R
      f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
      f_surj : Function.Surjective ⇑↑f
      fsym_surj : Function.Surjective ⇑↑f.symm
      this : Eq ((idealFactorsFunOfQuotHom f_surj).comp (idealFactorsFunOfQuotHom fs …
      ⊢ Eq ((↑(idealFactorsFunOfQuotHom f_surj)).comp ↑(idealFactorsFunOfQuotHom fsy …
    -/
    rw [← this, OrderHom.coe_eq, OrderHom.coe_eq]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Field K
      inst✝³ : IsDedekindDomain A
      I : Ideal R
      J : Ideal A
      B : Type u_4
      inst✝² : CommRing B
      inst✝¹ : IsDedekindDomain B
      L : Ideal B
      inst✝ : IsDedekindDomain R
      f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
      f_surj : Function.Surjective ⇑↑f
      fsym_surj : Function.Surjective ⇑↑f.symm
      ⊢ Eq ((↑(idealFactorsFunOfQuotHom fsym_surj)).comp ↑(idealFactorsFunOfQuotHom  …
    -/
  · have := idealFactorsFunOfQuotHom_comp f_surj fsym_surj
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Field K
      inst✝³ : IsDedekindDomain A
      I : Ideal R
      J : Ideal A
      B : Type u_4
      inst✝² : CommRing B
      inst✝¹ : IsDedekindDomain B
      L : Ideal B
      inst✝ : IsDedekindDomain R
      f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
      f_surj : Function.Surjective ⇑↑f
      fsym_surj : Function.Surjective ⇑↑f.symm
      this : Eq ((idealFactorsFunOfQuotHom fsym_surj).comp (idealFactorsFunOfQuotHom …
      ⊢ Eq ((↑(idealFactorsFunOfQuotHom fsym_surj)).comp ↑(idealFactorsFunOfQuotHom  …
    -/
    simp only [RingEquiv.symm_comp, idealFactorsFunOfQuotHom_id] at this
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Field K
      inst✝³ : IsDedekindDomain A
      I : Ideal R
      J : Ideal A
      B : Type u_4
      inst✝² : CommRing B
      inst✝¹ : IsDedekindDomain B
      L : Ideal B
      inst✝ : IsDedekindDomain R
      f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
      f_surj : Function.Surjective ⇑↑f
      fsym_surj : Function.Surjective ⇑↑f.symm
      this : Eq ((idealFactorsFunOfQuotHom fsym_surj).comp (idealFactorsFunOfQuotHom …
      ⊢ Eq ((↑(idealFactorsFunOfQuotHom fsym_surj)).comp ↑(idealFactorsFunOfQuotHom  …
    -/
    rw [← this, OrderHom.coe_eq, OrderHom.coe_eq]
    /-
      🎉 no goals
    -/


theorem idealFactorsEquivOfQuotEquiv_symm :
    (idealFactorsEquivOfQuotEquiv f).symm = idealFactorsEquivOfQuotEquiv f.symm := rfl


theorem idealFactorsEquivOfQuotEquiv_is_dvd_iso {L M : Ideal R} (hL : L ∣ I) (hM : M ∣ I) :
    (idealFactorsEquivOfQuotEquiv f ⟨L, hL⟩ : Ideal A) ∣ idealFactorsEquivOfQuotEquiv f ⟨M, hM⟩ ↔
      L ∣ M := by
  suffices
    idealFactorsEquivOfQuotEquiv f ⟨M, hM⟩ ≤ idealFactorsEquivOfQuotEquiv f ⟨L, hL⟩ ↔
      (⟨M, hM⟩ : { p : Ideal R | p ∣ I }) ≤ ⟨L, hL⟩
    by rw [dvd_iff_le, dvd_iff_le, Subtype.coe_le_coe, this, Subtype.mk_le_mk]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : IsDedekindDomain A
    I : Ideal R
    J : Ideal A
    inst✝ : IsDedekindDomain R
    f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
    L M : Ideal R
    hL : Dvd.dvd L I
    hM : Dvd.dvd M I
    ⊢ Iff (LE.le ((idealFactorsEquivOfQuotEquiv f) ⟨M, hM⟩) ((idealFactorsEquivOfQ …
  -/
  exact (idealFactorsEquivOfQuotEquiv f).le_iff_le
  /-
    🎉 no goals
  -/


theorem idealFactorsEquivOfQuotEquiv_mem_normalizedFactors_of_mem_normalizedFactors (hJ : J ≠ ⊥)
    {L : Ideal R} (hL : L ∈ normalizedFactors I) :
    ↑(idealFactorsEquivOfQuotEquiv f ⟨L, dvd_of_mem_normalizedFactors hL⟩)
      ∈ normalizedFactors J := by
  have hI : I ≠ ⊥ := by
    intro hI
    rw [hI, bot_eq_zero, normalizedFactors_zero, ← Multiset.empty_eq_zero] at hL
    exact Finset.not_mem_empty _ hL
  refine mem_normalizedFactors_factor_dvd_iso_of_mem_normalizedFactors hI hJ hL
    (d := (idealFactorsEquivOfQuotEquiv f).toEquiv) ?_
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : IsDedekindDomain A
    I : Ideal R
    J : Ideal A
    inst✝ : IsDedekindDomain R
    f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
    hJ : Ne J Bot.bot
    L : Ideal R
    hL : Membership.mem (UniqueFactorizationMonoid.normalizedFactors I) L
    hI : Ne I Bot.bot
    ⊢ ∀ (l l' : Subtype fun l => Dvd.dvd l I), Iff (Dvd.dvd ↑((idealFactorsEquivOf …
  -/
  rintro ⟨l, hl⟩ ⟨l', hl'⟩
  /-
    case mk.mk
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : IsDedekindDomain A
    I : Ideal R
    J : Ideal A
    inst✝ : IsDedekindDomain R
    f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
    hJ : Ne J Bot.bot
    L : Ideal R
    hL : Membership.mem (UniqueFactorizationMonoid.normalizedFactors I) L
    hI : Ne I Bot.bot
    l : Ideal R
    hl : Dvd.dvd l I
    l' : Ideal R
    hl' : Dvd.dvd l' I
    ⊢ Iff (Dvd.dvd ↑((idealFactorsEquivOfQuotEquiv f).toEquiv ⟨l, hl⟩) ↑((idealFac …
  -/
  rw [Subtype.coe_mk, Subtype.coe_mk]
  /-
    case mk.mk
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : IsDedekindDomain A
    I : Ideal R
    J : Ideal A
    inst✝ : IsDedekindDomain R
    f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
    hJ : Ne J Bot.bot
    L : Ideal R
    hL : Membership.mem (UniqueFactorizationMonoid.normalizedFactors I) L
    hI : Ne I Bot.bot
    l : Ideal R
    hl : Dvd.dvd l I
    l' : Ideal R
    hl' : Dvd.dvd l' I
    ⊢ Iff (Dvd.dvd ↑((idealFactorsEquivOfQuotEquiv f).toEquiv ⟨l, hl⟩) ↑((idealFac …
  -/
  apply idealFactorsEquivOfQuotEquiv_is_dvd_iso f
  /-
    🎉 no goals
  -/


/-- The bijection between the sets of normalized factors of I and J induced by a ring
    isomorphism `f : R/I ≅ A/J`. -/
def normalizedFactorsEquivOfQuotEquiv (hI : I ≠ ⊥) (hJ : J ≠ ⊥) :
    { L : Ideal R | L ∈ normalizedFactors I } ≃ { M : Ideal A | M ∈ normalizedFactors J } where
  toFun j :=
    ⟨idealFactorsEquivOfQuotEquiv f ⟨↑j, dvd_of_mem_normalizedFactors j.prop⟩,
      idealFactorsEquivOfQuotEquiv_mem_normalizedFactors_of_mem_normalizedFactors f hJ j.prop⟩
  invFun j :=
    ⟨(idealFactorsEquivOfQuotEquiv f).symm ⟨↑j, dvd_of_mem_normalizedFactors j.prop⟩, by
      /-
        R : Type u_1
        A : Type u_2
        K : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Field K
        inst✝³ : IsDedekindDomain A
        I : Ideal R
        J : Ideal A
        B : Type u_4
        inst✝² : CommRing B
        inst✝¹ : IsDedekindDomain B
        L : Ideal B
        inst✝ : IsDedekindDomain R
        f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
        hI : Ne I Bot.bot
        hJ : Ne J Bot.bot
        j : ↑(setOf fun M => Membership.mem (UniqueFactorizationMonoid.normalizedFacto …
        ⊢ Membership.mem (setOf fun L => Membership.mem (UniqueFactorizationMonoid.nor …
      -/
      rw [idealFactorsEquivOfQuotEquiv_symm]
      exact
        idealFactorsEquivOfQuotEquiv_mem_normalizedFactors_of_mem_normalizedFactors f.symm hI
          j.prop⟩
                                /-
                                  R : Type u_1
                                  A : Type u_2
                                  K : Type u_3
                                  inst✝⁶ : CommRing R
                                  inst✝⁵ : CommRing A
                                  inst✝⁴ : Field K
                                  inst✝³ : IsDedekindDomain A
                                  I : Ideal R
                                  J : Ideal A
                                  B : Type u_4
                                  inst✝² : CommRing B
                                  inst✝¹ : IsDedekindDomain B
                                  L : Ideal B
                                  inst✝ : IsDedekindDomain R
                                  f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
                                  hI : Ne I Bot.bot
                                  hJ : Ne J Bot.bot
                                  x✝ : ↑(setOf fun L => Membership.mem (UniqueFactorizationMonoid.normalizedFact …
                                  j : Ideal R
                                  hj : Membership.mem (setOf fun L => Membership.mem (UniqueFactorizationMonoid. …
                                  ⊢ Eq ((fun j => ⟨↑((idealFactorsEquivOfQuotEquiv f).symm ⟨↑j, ⋯⟩), ⋯⟩) ((fun j …
                                -/
  left_inv := fun ⟨j, hj⟩ => by simp
                                /-
                                  🎉 no goals
                                -/
  right_inv := fun ⟨j, hj⟩ => by
    /-
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Field K
      inst✝³ : IsDedekindDomain A
      I : Ideal R
      J : Ideal A
      B : Type u_4
      inst✝² : CommRing B
      inst✝¹ : IsDedekindDomain B
      L : Ideal B
      inst✝ : IsDedekindDomain R
      f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
      hI : Ne I Bot.bot
      hJ : Ne J Bot.bot
      x✝ : ↑(setOf fun M => Membership.mem (UniqueFactorizationMonoid.normalizedFact …
      j : Ideal A
      hj : Membership.mem (setOf fun M => Membership.mem (UniqueFactorizationMonoid. …
      ⊢ Eq ((fun j => ⟨↑((idealFactorsEquivOfQuotEquiv f) ⟨↑j, ⋯⟩), ⋯⟩) ((fun j => ⟨ …
    -/
    simp
    -- This used to be the end of the proof before https://github.com/leanprover/lean4/pull/2644
    /-
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Field K
      inst✝³ : IsDedekindDomain A
      I : Ideal R
      J : Ideal A
      B : Type u_4
      inst✝² : CommRing B
      inst✝¹ : IsDedekindDomain B
      L : Ideal B
      inst✝ : IsDedekindDomain R
      f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
      hI : Ne I Bot.bot
      hJ : Ne J Bot.bot
      x✝ : ↑(setOf fun M => Membership.mem (UniqueFactorizationMonoid.normalizedFact …
      j : Ideal A
      hj : Membership.mem (setOf fun M => Membership.mem (UniqueFactorizationMonoid. …
      ⊢ Eq (↑((idealFactorsEquivOfQuotEquiv f) ((idealFactorsEquivOfQuotEquiv f).sym …
    -/
    erw [OrderIso.apply_symm_apply]
    /-
      🎉 no goals
    -/


@[simp]
theorem normalizedFactorsEquivOfQuotEquiv_symm (hI : I ≠ ⊥) (hJ : J ≠ ⊥) :
    (normalizedFactorsEquivOfQuotEquiv f hI hJ).symm =
      normalizedFactorsEquivOfQuotEquiv f.symm hJ hI := rfl


/-- The map `normalizedFactorsEquivOfQuotEquiv` preserves multiplicities. -/
theorem normalizedFactorsEquivOfQuotEquiv_emultiplicity_eq_emultiplicity (hI : I ≠ ⊥) (hJ : J ≠ ⊥)
    (L : Ideal R) (hL : L ∈ normalizedFactors I) :
    emultiplicity (↑(normalizedFactorsEquivOfQuotEquiv f hI hJ ⟨L, hL⟩)) J = emultiplicity L I := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : IsDedekindDomain A
    I : Ideal R
    J : Ideal A
    inst✝ : IsDedekindDomain R
    f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
    hI : Ne I Bot.bot
    hJ : Ne J Bot.bot
    L : Ideal R
    hL : Membership.mem (UniqueFactorizationMonoid.normalizedFactors I) L
    ⊢ Eq (emultiplicity (↑((normalizedFactorsEquivOfQuotEquiv f hI hJ) ⟨L, hL⟩)) J …
  -/
  rw [normalizedFactorsEquivOfQuotEquiv, Equiv.coe_fn_mk, Subtype.coe_mk]
  refine emultiplicity_factor_dvd_iso_eq_emultiplicity_of_mem_normalizedFactors hI hJ hL
    (d := (idealFactorsEquivOfQuotEquiv f).toEquiv) ?_
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : IsDedekindDomain A
    I : Ideal R
    J : Ideal A
    inst✝ : IsDedekindDomain R
    f : RingEquiv (HasQuotient.Quotient R I) (HasQuotient.Quotient A J)
    hI : Ne I Bot.bot
    hJ : Ne J Bot.bot
    L : Ideal R
    hL : Membership.mem (UniqueFactorizationMonoid.normalizedFactors I) L
    ⊢ ∀ (l l' : Subtype fun l => Dvd.dvd l I), Iff (Dvd.dvd ↑((idealFactorsEquivOf …
  -/
  exact fun ⟨l, hl⟩ ⟨l', hl'⟩ => idealFactorsEquivOfQuotEquiv_is_dvd_iso f hl hl'
  /-
    🎉 no goals
  -/


theorem Ring.DimensionLeOne.prime_le_prime_iff_eq [Ring.DimensionLEOne R] {P Q : Ideal R}
    [hP : P.IsPrime] [hQ : Q.IsPrime] (hP0 : P ≠ ⊥) : P ≤ Q ↔ P = Q :=
  ⟨(hP.isMaximal hP0).eq_of_le hQ.ne_top, Eq.le⟩


theorem Ideal.coprime_of_no_prime_ge {I J : Ideal R} (h : ∀ P, I ≤ P → J ≤ P → ¬IsPrime P) :
    IsCoprime I J := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I J : Ideal R
    h : ∀ (P : Ideal R), LE.le I P → LE.le J P → Not P.IsPrime
    ⊢ IsCoprime I J
  -/
  rw [isCoprime_iff_sup_eq]
  /-
    R : Type u_1
    inst✝ : CommRing R
    I J : Ideal R
    h : ∀ (P : Ideal R), LE.le I P → LE.le J P → Not P.IsPrime
    ⊢ Eq (Max.max I J) Top.top
  -/
  by_contra hIJ
  /-
    R : Type u_1
    inst✝ : CommRing R
    I J : Ideal R
    h : ∀ (P : Ideal R), LE.le I P → LE.le J P → Not P.IsPrime
    hIJ : Not (Eq (Max.max I J) Top.top)
    ⊢ False
  -/
  obtain ⟨P, hP, hIJ⟩ := Ideal.exists_le_maximal _ hIJ
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    I J : Ideal R
    h : ∀ (P : Ideal R), LE.le I P → LE.le J P → Not P.IsPrime
    hIJ✝ : Not (Eq (Max.max I J) Top.top)
    P : Ideal R
    hP : P.IsMaximal
    hIJ : LE.le (Max.max I J) P
    ⊢ False
  -/
  exact h P (le_trans le_sup_left hIJ) (le_trans le_sup_right hIJ) hP.isPrime
  /-
    🎉 no goals
  -/


theorem Ideal.IsPrime.mul_mem_pow (I : Ideal R) [hI : I.IsPrime] {a b : R} {n : ℕ}
    (h : a * b ∈ I ^ n) : a ∈ I ∨ b ∈ I ^ n := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : I.IsPrime
    a b : R
    n : Nat
    h : Membership.mem (HPow.hPow I n) (HMul.hMul a b)
    ⊢ Or (Membership.mem I a) (Membership.mem (HPow.hPow I n) b)
  -/
  cases n; · simp
             /-
               🎉 no goals
             -/
  /-
    case succ
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : I.IsPrime
    a b : R
    n✝ : Nat
    h : Membership.mem (HPow.hPow I (HAdd.hAdd n✝ 1)) (HMul.hMul a b)
    ⊢ Or (Membership.mem I a) (Membership.mem (HPow.hPow I (HAdd.hAdd n✝ 1)) b)
  -/
  by_cases hI0 : I = ⊥; · simpa [pow_succ, hI0] using h
                          /-
                            🎉 no goals
                          -/
  simp only [← Submodule.span_singleton_le_iff_mem, Ideal.submodule_span_eq, ← Ideal.dvd_iff_le, ←
    Ideal.span_singleton_mul_span_singleton] at h ⊢
  /-
    case neg
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : I.IsPrime
    a b : R
    n✝ : Nat
    hI0 : Not (Eq I Bot.bot)
    h : Dvd.dvd (HPow.hPow I (HAdd.hAdd n✝ 1)) (HMul.hMul (Ideal.span (Singleton.s …
    ⊢ Or (Dvd.dvd I (Ideal.span (Singleton.singleton a))) (Dvd.dvd (HPow.hPow I (H …
  -/
  by_cases ha : I ∣ span {a}
    /-
      case pos
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDedekindDomain R
      I : Ideal R
      hI : I.IsPrime
      a b : R
      n✝ : Nat
      hI0 : Not (Eq I Bot.bot)
      h : Dvd.dvd (HPow.hPow I (HAdd.hAdd n✝ 1)) (HMul.hMul (Ideal.span (Singleton.s …
      ha : Dvd.dvd I (Ideal.span (Singleton.singleton a))
      ⊢ Or (Dvd.dvd I (Ideal.span (Singleton.singleton a))) (Dvd.dvd (HPow.hPow I (H …
    -/
  · exact Or.inl ha
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : I.IsPrime
    a b : R
    n✝ : Nat
    hI0 : Not (Eq I Bot.bot)
    h : Dvd.dvd (HPow.hPow I (HAdd.hAdd n✝ 1)) (HMul.hMul (Ideal.span (Singleton.s …
    ha : Not (Dvd.dvd I (Ideal.span (Singleton.singleton a)))
    ⊢ Or (Dvd.dvd I (Ideal.span (Singleton.singleton a))) (Dvd.dvd (HPow.hPow I (H …
  -/
  rw [mul_comm] at h
  /-
    case neg
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : I.IsPrime
    a b : R
    n✝ : Nat
    hI0 : Not (Eq I Bot.bot)
    h : Dvd.dvd (HPow.hPow I (HAdd.hAdd n✝ 1)) (HMul.hMul (Ideal.span (Singleton.s …
    ha : Not (Dvd.dvd I (Ideal.span (Singleton.singleton a)))
    ⊢ Or (Dvd.dvd I (Ideal.span (Singleton.singleton a))) (Dvd.dvd (HPow.hPow I (H …
  -/
  exact Or.inr (Prime.pow_dvd_of_dvd_mul_right ((Ideal.prime_iff_isPrime hI0).mpr hI) _ ha h)
  /-
    🎉 no goals
  -/


theorem Ideal.IsPrime.mem_pow_mul (I : Ideal R) [hI : I.IsPrime] {a b : R} {n : ℕ}
    (h : a * b ∈ I ^ n) : a ∈ I ^ n ∨ b ∈ I := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : I.IsPrime
    a b : R
    n : Nat
    h : Membership.mem (HPow.hPow I n) (HMul.hMul a b)
    ⊢ Or (Membership.mem (HPow.hPow I n) a) (Membership.mem I b)
  -/
  rw [mul_comm] at h
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : I.IsPrime
    a b : R
    n : Nat
    h : Membership.mem (HPow.hPow I n) (HMul.hMul b a)
    ⊢ Or (Membership.mem (HPow.hPow I n) a) (Membership.mem I b)
  -/
  rw [or_comm]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I : Ideal R
    hI : I.IsPrime
    a b : R
    n : Nat
    h : Membership.mem (HPow.hPow I n) (HMul.hMul b a)
    ⊢ Or (Membership.mem I b) (Membership.mem (HPow.hPow I n) a)
  -/
  exact Ideal.IsPrime.mul_mem_pow _ h
  /-
    🎉 no goals
  -/


theorem Ideal.count_normalizedFactors_eq {p x : Ideal R} [hp : p.IsPrime] {n : ℕ} (hle : x ≤ p ^ n)
    [DecidableEq (Ideal R)] (hlt : ¬x ≤ p ^ (n + 1)) : (normalizedFactors x).count p = n :=
  count_normalizedFactors_eq' ((Ideal.isPrime_iff_bot_or_prime.mp hp).imp_right Prime.irreducible)
    (normalize_eq _) (Ideal.dvd_iff_le.mpr hle) (mt Ideal.le_of_dvd hlt)


/-- The number of times an ideal `I` occurs as normalized factor of another ideal `J` is stable
  when regarding at these ideals as associated elements of the monoid of ideals.-/
theorem count_associates_factors_eq [DecidableEq (Ideal R)] [DecidableEq <| Associates (Ideal R)]
    [∀ (p : Associates <| Ideal R), Decidable (Irreducible p)]
    {I J : Ideal R} (hI : I ≠ 0) (hJ : J.IsPrime) (hJ₀ : J ≠ ⊥) :
    (Associates.mk J).count (Associates.mk I).factors = Multiset.count J (normalizedFactors I) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : DecidableEq (Ideal R)
    inst✝¹ : DecidableEq (Associates (Ideal R))
    inst✝ : (p : Associates (Ideal R)) → Decidable (Irreducible p)
    I J : Ideal R
    hI : Ne I 0
    hJ : J.IsPrime
    hJ₀ : Ne J Bot.bot
    ⊢ Eq ((Associates.mk J).count (Associates.mk I).factors) (Multiset.count J (Un …
  -/
  replace hI : Associates.mk I ≠ 0 := Associates.mk_ne_zero.mpr hI
  have hJ' : Irreducible (Associates.mk J) := by
    simpa only [Associates.irreducible_mk] using (Ideal.prime_of_isPrime hJ₀ hJ).irreducible
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : DecidableEq (Ideal R)
    inst✝¹ : DecidableEq (Associates (Ideal R))
    inst✝ : (p : Associates (Ideal R)) → Decidable (Irreducible p)
    I J : Ideal R
    hJ : J.IsPrime
    hJ₀ : Ne J Bot.bot
    hI : Ne (Associates.mk I) 0
    hJ' : Irreducible (Associates.mk J)
    ⊢ Eq ((Associates.mk J).count (Associates.mk I).factors) (Multiset.count J (Un …
  -/
  apply (Ideal.count_normalizedFactors_eq (p := J) (x := I) _ _).symm
  all_goals
    rw [← Ideal.dvd_iff_le, ← Associates.mk_dvd_mk, Associates.mk_pow]
    simp only [Associates.dvd_eq_le]
    rw [Associates.prime_pow_dvd_iff_le hI hJ']
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    inst✝² : DecidableEq (Ideal R)
    inst✝¹ : DecidableEq (Associates (Ideal R))
    inst✝ : (p : Associates (Ideal R)) → Decidable (Irreducible p)
    I J : Ideal R
    hJ : J.IsPrime
    hJ₀ : Ne J Bot.bot
    hI : Ne (Associates.mk I) 0
    hJ' : Irreducible (Associates.mk J)
    ⊢ Not (LE.le (HAdd.hAdd ((Associates.mk J).count (Associates.mk I).factors) 1) …
  -/
  omega
  /-
    🎉 no goals
  -/


theorem Ideal.le_mul_of_no_prime_factors {I J K : Ideal R}
    (coprime : ∀ P, J ≤ P → K ≤ P → ¬IsPrime P) (hJ : I ≤ J) (hK : I ≤ K) : I ≤ J * K := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I J K : Ideal R
    coprime : ∀ (P : Ideal R), LE.le J P → LE.le K P → Not P.IsPrime
    hJ : LE.le I J
    hK : LE.le I K
    ⊢ LE.le I (HMul.hMul J K)
  -/
  simp only [← Ideal.dvd_iff_le] at coprime hJ hK ⊢
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I J K : Ideal R
    coprime : ∀ (P : Ideal R), Dvd.dvd P J → Dvd.dvd P K → Not P.IsPrime
    hJ : Dvd.dvd J I
    hK : Dvd.dvd K I
    ⊢ Dvd.dvd (HMul.hMul J K) I
  -/
  by_cases hJ0 : J = 0
    /-
      case pos
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDedekindDomain R
      I J K : Ideal R
      coprime : ∀ (P : Ideal R), Dvd.dvd P J → Dvd.dvd P K → Not P.IsPrime
      hJ : Dvd.dvd J I
      hK : Dvd.dvd K I
      hJ0 : Eq J 0
      ⊢ Dvd.dvd (HMul.hMul J K) I
    -/
  · simpa only [hJ0, zero_mul] using hJ
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    I J K : Ideal R
    coprime : ∀ (P : Ideal R), Dvd.dvd P J → Dvd.dvd P K → Not P.IsPrime
    hJ : Dvd.dvd J I
    hK : Dvd.dvd K I
    hJ0 : Not (Eq J 0)
    ⊢ Dvd.dvd (HMul.hMul J K) I
  -/
  obtain ⟨I', rfl⟩ := hK
  /-
    case neg.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    J K : Ideal R
    coprime : ∀ (P : Ideal R), Dvd.dvd P J → Dvd.dvd P K → Not P.IsPrime
    hJ0 : Not (Eq J 0)
    I' : Ideal R
    hJ : Dvd.dvd J (HMul.hMul K I')
    ⊢ Dvd.dvd (HMul.hMul J K) (HMul.hMul K I')
  -/
  rw [mul_comm]
  refine mul_dvd_mul_left K
    (UniqueFactorizationMonoid.dvd_of_dvd_mul_right_of_no_prime_factors (b := K) hJ0 ?_ hJ)
  /-
    case neg.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    J K : Ideal R
    coprime : ∀ (P : Ideal R), Dvd.dvd P J → Dvd.dvd P K → Not P.IsPrime
    hJ0 : Not (Eq J 0)
    I' : Ideal R
    hJ : Dvd.dvd J (HMul.hMul K I')
    ⊢ ∀ {d : Ideal R}, Dvd.dvd d J → Dvd.dvd d K → Not (Prime d)
  -/
  exact fun hPJ hPK => mt Ideal.isPrime_of_prime (coprime _ hPJ hPK)
  /-
    🎉 no goals
  -/


/-- The intersection of distinct prime powers in a Dedekind domain is the product of these
prime powers. -/
theorem IsDedekindDomain.inf_prime_pow_eq_prod {ι : Type*} (s : Finset ι) (f : ι → Ideal R)
    (e : ι → ℕ) (prime : ∀ i ∈ s, Prime (f i))
    (coprime : ∀ᵉ (i ∈ s) (j ∈ s), i ≠ j → f i ≠ f j) :
    (s.inf fun i => f i ^ e i) = ∏ i ∈ s, f i ^ e i := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    ι : Type u_4
    s : Finset ι
    f : ι → Ideal R
    e : ι → Nat
    prime : ∀ (i : ι), Membership.mem s i → Prime (f i)
    coprime : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Ne i …
    ⊢ Eq (s.inf fun i => HPow.hPow (f i) (e i)) (s.prod fun i => HPow.hPow (f i) ( …
  -/
  letI := Classical.decEq ι
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    ι : Type u_4
    s : Finset ι
    f : ι → Ideal R
    e : ι → Nat
    prime : ∀ (i : ι), Membership.mem s i → Prime (f i)
    coprime : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Ne i …
    this : DecidableEq ι := Classical.decEq ι
    ⊢ Eq (s.inf fun i => HPow.hPow (f i) (e i)) (s.prod fun i => HPow.hPow (f i) ( …
  -/
  revert prime coprime
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    ι : Type u_4
    s : Finset ι
    f : ι → Ideal R
    e : ι → Nat
    this : DecidableEq ι := Classical.decEq ι
    ⊢ (∀ (i : ι), Membership.mem s i → Prime (f i)) → (∀ (i : ι), Membership.mem s …
  -/
  refine s.induction ?_ ?_
    /-
      case refine_1
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDedekindDomain R
      ι : Type u_4
      s : Finset ι
      f : ι → Ideal R
      e : ι → Nat
      this : DecidableEq ι := Classical.decEq ι
      ⊢ (∀ (i : ι), Membership.mem EmptyCollection.emptyCollection i → Prime (f i))  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    ι : Type u_4
    s : Finset ι
    f : ι → Ideal R
    e : ι → Nat
    this : DecidableEq ι := Classical.decEq ι
    ⊢ ∀ ⦃a : ι⦄ {s : Finset ι}, Not (Membership.mem s a) → ((∀ (i : ι), Membership …
  -/
  intro a s ha ih prime coprime
  specialize
    ih (fun i hi => prime i (Finset.mem_insert_of_mem hi)) fun i hi j hj =>
      coprime i (Finset.mem_insert_of_mem hi) j (Finset.mem_insert_of_mem hj)
  /-
    case refine_2
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    ι : Type u_4
    s✝ : Finset ι
    f : ι → Ideal R
    e : ι → Nat
    this : DecidableEq ι := Classical.decEq ι
    a : ι
    s : Finset ι
    ha : Not (Membership.mem s a)
    prime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → Prime (f i)
    coprime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → ∀ (j : ι), Members …
    ih : Eq (s.inf fun i => HPow.hPow (f i) (e i)) (s.prod fun i => HPow.hPow (f i …
    ⊢ Eq ((Insert.insert a s).inf fun i => HPow.hPow (f i) (e i)) ((Insert.insert  …
  -/
  rw [Finset.inf_insert, Finset.prod_insert ha, ih]
  /-
    case refine_2
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    ι : Type u_4
    s✝ : Finset ι
    f : ι → Ideal R
    e : ι → Nat
    this : DecidableEq ι := Classical.decEq ι
    a : ι
    s : Finset ι
    ha : Not (Membership.mem s a)
    prime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → Prime (f i)
    coprime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → ∀ (j : ι), Members …
    ih : Eq (s.inf fun i => HPow.hPow (f i) (e i)) (s.prod fun i => HPow.hPow (f i …
    ⊢ Eq (Min.min (HPow.hPow (f a) (e a)) (s.prod fun i => HPow.hPow (f i) (e i))) …
  -/
  refine le_antisymm (Ideal.le_mul_of_no_prime_factors ?_ inf_le_left inf_le_right) Ideal.mul_le_inf
  /-
    case refine_2
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    ι : Type u_4
    s✝ : Finset ι
    f : ι → Ideal R
    e : ι → Nat
    this : DecidableEq ι := Classical.decEq ι
    a : ι
    s : Finset ι
    ha : Not (Membership.mem s a)
    prime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → Prime (f i)
    coprime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → ∀ (j : ι), Members …
    ih : Eq (s.inf fun i => HPow.hPow (f i) (e i)) (s.prod fun i => HPow.hPow (f i …
    ⊢ ∀ (P : Ideal R), LE.le (HPow.hPow (f a) (e a)) P → LE.le (s.prod fun x => HP …
  -/
  intro P hPa hPs hPp
  /-
    case refine_2
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    ι : Type u_4
    s✝ : Finset ι
    f : ι → Ideal R
    e : ι → Nat
    this : DecidableEq ι := Classical.decEq ι
    a : ι
    s : Finset ι
    ha : Not (Membership.mem s a)
    prime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → Prime (f i)
    coprime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → ∀ (j : ι), Members …
    ih : Eq (s.inf fun i => HPow.hPow (f i) (e i)) (s.prod fun i => HPow.hPow (f i …
    P : Ideal R
    hPa : LE.le (HPow.hPow (f a) (e a)) P
    hPs : LE.le (s.prod fun x => HPow.hPow (f x) (e x)) P
    hPp : P.IsPrime
    ⊢ False
  -/
  obtain ⟨b, hb, hPb⟩ := hPp.prod_le.mp hPs
  /-
    case refine_2.intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    ι : Type u_4
    s✝ : Finset ι
    f : ι → Ideal R
    e : ι → Nat
    this : DecidableEq ι := Classical.decEq ι
    a : ι
    s : Finset ι
    ha : Not (Membership.mem s a)
    prime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → Prime (f i)
    coprime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → ∀ (j : ι), Members …
    ih : Eq (s.inf fun i => HPow.hPow (f i) (e i)) (s.prod fun i => HPow.hPow (f i …
    P : Ideal R
    hPa : LE.le (HPow.hPow (f a) (e a)) P
    hPs : LE.le (s.prod fun x => HPow.hPow (f x) (e x)) P
    hPp : P.IsPrime
    b : ι
    hb : Membership.mem s b
    hPb : LE.le (HPow.hPow (f b) (e b)) P
    ⊢ False
  -/
  haveI := Ideal.isPrime_of_prime (prime a (Finset.mem_insert_self a s))
  /-
    case refine_2.intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    ι : Type u_4
    s✝ : Finset ι
    f : ι → Ideal R
    e : ι → Nat
    this✝ : DecidableEq ι := Classical.decEq ι
    a : ι
    s : Finset ι
    ha : Not (Membership.mem s a)
    prime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → Prime (f i)
    coprime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → ∀ (j : ι), Members …
    ih : Eq (s.inf fun i => HPow.hPow (f i) (e i)) (s.prod fun i => HPow.hPow (f i …
    P : Ideal R
    hPa : LE.le (HPow.hPow (f a) (e a)) P
    hPs : LE.le (s.prod fun x => HPow.hPow (f x) (e x)) P
    hPp : P.IsPrime
    b : ι
    hb : Membership.mem s b
    hPb : LE.le (HPow.hPow (f b) (e b)) P
    this : (f a).IsPrime
    ⊢ False
  -/
  haveI := Ideal.isPrime_of_prime (prime b (Finset.mem_insert_of_mem hb))
  /-
    case refine_2.intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    ι : Type u_4
    s✝ : Finset ι
    f : ι → Ideal R
    e : ι → Nat
    this✝¹ : DecidableEq ι := Classical.decEq ι
    a : ι
    s : Finset ι
    ha : Not (Membership.mem s a)
    prime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → Prime (f i)
    coprime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → ∀ (j : ι), Members …
    ih : Eq (s.inf fun i => HPow.hPow (f i) (e i)) (s.prod fun i => HPow.hPow (f i …
    P : Ideal R
    hPa : LE.le (HPow.hPow (f a) (e a)) P
    hPs : LE.le (s.prod fun x => HPow.hPow (f x) (e x)) P
    hPp : P.IsPrime
    b : ι
    hb : Membership.mem s b
    hPb : LE.le (HPow.hPow (f b) (e b)) P
    this✝ : (f a).IsPrime
    this : (f b).IsPrime
    ⊢ False
  -/
  refine coprime a (Finset.mem_insert_self a s) b (Finset.mem_insert_of_mem hb) ?_ ?_
    /-
      case refine_2.intro.intro.refine_1
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDedekindDomain R
      ι : Type u_4
      s✝ : Finset ι
      f : ι → Ideal R
      e : ι → Nat
      this✝¹ : DecidableEq ι := Classical.decEq ι
      a : ι
      s : Finset ι
      ha : Not (Membership.mem s a)
      prime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → Prime (f i)
      coprime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → ∀ (j : ι), Members …
      ih : Eq (s.inf fun i => HPow.hPow (f i) (e i)) (s.prod fun i => HPow.hPow (f i …
      P : Ideal R
      hPa : LE.le (HPow.hPow (f a) (e a)) P
      hPs : LE.le (s.prod fun x => HPow.hPow (f x) (e x)) P
      hPp : P.IsPrime
      b : ι
      hb : Membership.mem s b
      hPb : LE.le (HPow.hPow (f b) (e b)) P
      this✝ : (f a).IsPrime
      this : (f b).IsPrime
      ⊢ Ne a b
    -/
  · exact (ne_of_mem_of_not_mem hb ha).symm
    /-
      🎉 no goals
    -/
  · refine ((Ring.DimensionLeOne.prime_le_prime_iff_eq ?_).mp (hPp.le_of_pow_le hPa)).trans
      ((Ring.DimensionLeOne.prime_le_prime_iff_eq ?_).mp (hPp.le_of_pow_le hPb)).symm
      /-
        case refine_2.intro.intro.refine_2.refine_1
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDedekindDomain R
        ι : Type u_4
        s✝ : Finset ι
        f : ι → Ideal R
        e : ι → Nat
        this✝¹ : DecidableEq ι := Classical.decEq ι
        a : ι
        s : Finset ι
        ha : Not (Membership.mem s a)
        prime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → Prime (f i)
        coprime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → ∀ (j : ι), Members …
        ih : Eq (s.inf fun i => HPow.hPow (f i) (e i)) (s.prod fun i => HPow.hPow (f i …
        P : Ideal R
        hPa : LE.le (HPow.hPow (f a) (e a)) P
        hPs : LE.le (s.prod fun x => HPow.hPow (f x) (e x)) P
        hPp : P.IsPrime
        b : ι
        hb : Membership.mem s b
        hPb : LE.le (HPow.hPow (f b) (e b)) P
        this✝ : (f a).IsPrime
        this : (f b).IsPrime
        ⊢ Ne (f a) Bot.bot
      -/
    · exact (prime a (Finset.mem_insert_self a s)).ne_zero
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.refine_2.refine_2
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDedekindDomain R
        ι : Type u_4
        s✝ : Finset ι
        f : ι → Ideal R
        e : ι → Nat
        this✝¹ : DecidableEq ι := Classical.decEq ι
        a : ι
        s : Finset ι
        ha : Not (Membership.mem s a)
        prime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → Prime (f i)
        coprime : ∀ (i : ι), Membership.mem (Insert.insert a s) i → ∀ (j : ι), Members …
        ih : Eq (s.inf fun i => HPow.hPow (f i) (e i)) (s.prod fun i => HPow.hPow (f i …
        P : Ideal R
        hPa : LE.le (HPow.hPow (f a) (e a)) P
        hPs : LE.le (s.prod fun x => HPow.hPow (f x) (e x)) P
        hPp : P.IsPrime
        b : ι
        hb : Membership.mem s b
        hPb : LE.le (HPow.hPow (f b) (e b)) P
        this✝ : (f a).IsPrime
        this : (f b).IsPrime
        ⊢ Ne (f b) Bot.bot
      -/
    · exact (prime b (Finset.mem_insert_of_mem hb)).ne_zero
      /-
        🎉 no goals
      -/


/-- **Chinese remainder theorem** for a Dedekind domain: if the ideal `I` factors as
`∏ i, P i ^ e i`, then `R ⧸ I` factors as `Π i, R ⧸ (P i ^ e i)`. -/
noncomputable def IsDedekindDomain.quotientEquivPiOfProdEq {ι : Type*} [Fintype ι] (I : Ideal R)
    (P : ι → Ideal R) (e : ι → ℕ) (prime : ∀ i, Prime (P i))
    (coprime : Pairwise fun i j => P i ≠ P j)
    (prod_eq : ∏ i, P i ^ e i = I) : R ⧸ I ≃+* ∀ i, R ⧸ P i ^ e i :=
  (Ideal.quotEquivOfEq
    (by
      simp only [← prod_eq, Finset.inf_eq_iInf, Finset.mem_univ, ciInf_pos,
        ← IsDedekindDomain.inf_prime_pow_eq_prod _ _ _ (fun i _ => prime i)
        (coprime.set_pairwise _)])).trans <|
    Ideal.quotientInfRingEquivPiQuotient _ fun i j hij => Ideal.coprime_of_no_prime_ge <| by
      /-
        R : Type u_1
        A : Type u_2
        K : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : CommRing A
        inst✝² : Field K
        inst✝¹ : IsDedekindDomain R
        ι : Type u_4
        inst✝ : Fintype ι
        I : Ideal R
        P : ι → Ideal R
        e : ι → Nat
        prime : ∀ (i : ι), Prime (P i)
        coprime : Pairwise fun i j => Ne (P i) (P j)
        prod_eq : Eq (Finset.univ.prod fun i => HPow.hPow (P i) (e i)) I
        i j : ι
        hij : Ne i j
        ⊢ ∀ (P_1 : Ideal R), LE.le ((fun i => HPow.hPow (P i) (e i)) i) P_1 → LE.le (( …
      -/
      intro P hPi hPj hPp
      /-
        R : Type u_1
        A : Type u_2
        K : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : CommRing A
        inst✝² : Field K
        inst✝¹ : IsDedekindDomain R
        ι : Type u_4
        inst✝ : Fintype ι
        I : Ideal R
        P✝ : ι → Ideal R
        e : ι → Nat
        prime : ∀ (i : ι), Prime (P✝ i)
        coprime : Pairwise fun i j => Ne (P✝ i) (P✝ j)
        prod_eq : Eq (Finset.univ.prod fun i => HPow.hPow (P✝ i) (e i)) I
        i j : ι
        hij : Ne i j
        P : Ideal R
        hPi : LE.le ((fun i => HPow.hPow (P✝ i) (e i)) i) P
        hPj : LE.le ((fun i => HPow.hPow (P✝ i) (e i)) j) P
        hPp : P.IsPrime
        ⊢ False
      -/
      haveI := Ideal.isPrime_of_prime (prime i)
      /-
        R : Type u_1
        A : Type u_2
        K : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : CommRing A
        inst✝² : Field K
        inst✝¹ : IsDedekindDomain R
        ι : Type u_4
        inst✝ : Fintype ι
        I : Ideal R
        P✝ : ι → Ideal R
        e : ι → Nat
        prime : ∀ (i : ι), Prime (P✝ i)
        coprime : Pairwise fun i j => Ne (P✝ i) (P✝ j)
        prod_eq : Eq (Finset.univ.prod fun i => HPow.hPow (P✝ i) (e i)) I
        i j : ι
        hij : Ne i j
        P : Ideal R
        hPi : LE.le ((fun i => HPow.hPow (P✝ i) (e i)) i) P
        hPj : LE.le ((fun i => HPow.hPow (P✝ i) (e i)) j) P
        hPp : P.IsPrime
        this : (P✝ i).IsPrime
        ⊢ False
      -/
      haveI := Ideal.isPrime_of_prime (prime j)
      exact coprime hij <| ((Ring.DimensionLeOne.prime_le_prime_iff_eq (prime i).ne_zero).mp
        (hPp.le_of_pow_le hPi)).trans <| Eq.symm <|
          (Ring.DimensionLeOne.prime_le_prime_iff_eq (prime j).ne_zero).mp (hPp.le_of_pow_le hPj)


open scoped Classical in
/-- **Chinese remainder theorem** for a Dedekind domain: `R ⧸ I` factors as `Π i, R ⧸ (P i ^ e i)`,
where `P i` ranges over the prime factors of `I` and `e i` over the multiplicities. -/
noncomputable def IsDedekindDomain.quotientEquivPiFactors {I : Ideal R} (hI : I ≠ ⊥) :
    R ⧸ I ≃+* ∀ P : (factors I).toFinset, R ⧸ (P : Ideal R) ^ (Multiset.count ↑P (factors I)) :=
  IsDedekindDomain.quotientEquivPiOfProdEq _ _ _
    (fun P : (factors I).toFinset => prime_of_factor _ (Multiset.mem_toFinset.mp P.prop))
    (fun _ _ hij => Subtype.coe_injective.ne hij)
    (calc
      (∏ P : (factors I).toFinset, (P : Ideal R) ^ (factors I).count (P : Ideal R)) =
          ∏ P ∈ (factors I).toFinset, P ^ (factors I).count P :=
        (factors I).toFinset.prod_coe_sort fun P => P ^ (factors I).count P
      _ = ((factors I).map fun P => P).prod := (Finset.prod_multiset_map_count (factors I) id).symm
                                 /-
                                   R : Type u_1
                                   A : Type u_2
                                   K : Type u_3
                                   inst✝³ : CommRing R
                                   inst✝² : CommRing A
                                   inst✝¹ : Field K
                                   inst✝ : IsDedekindDomain R
                                   I : Ideal R
                                   hI : Ne I Bot.bot
                                   ⊢ Eq (Multiset.map (fun P => P) (UniqueFactorizationMonoid.factors I)).prod (U …
                                 -/
      _ = (factors I).prod := by rw [Multiset.map_id']
                                 /-
                                   🎉 no goals
                                 -/
      _ = I := associated_iff_eq.mp (factors_prod hI)
      )


@[simp]
theorem IsDedekindDomain.quotientEquivPiFactors_mk {I : Ideal R} (hI : I ≠ ⊥) (x : R) :
    IsDedekindDomain.quotientEquivPiFactors hI (Ideal.Quotient.mk I x) = fun _P =>
      Ideal.Quotient.mk _ x := rfl


/-- **Chinese remainder theorem** for a Dedekind domain: if the ideal `I` factors as
`∏ i ∈ s, P i ^ e i`, then `R ⧸ I` factors as `Π (i : s), R ⧸ (P i ^ e i)`.

This is a version of `IsDedekindDomain.quotientEquivPiOfProdEq` where we restrict
the product to a finite subset `s` of a potentially infinite indexing type `ι`.
-/
noncomputable def IsDedekindDomain.quotientEquivPiOfFinsetProdEq {ι : Type*} {s : Finset ι}
    (I : Ideal R) (P : ι → Ideal R) (e : ι → ℕ) (prime : ∀ i ∈ s, Prime (P i))
    (coprime : ∀ᵉ (i ∈ s) (j ∈ s), i ≠ j → P i ≠ P j)
    (prod_eq : ∏ i ∈ s, P i ^ e i = I) : R ⧸ I ≃+* ∀ i : s, R ⧸ P i ^ e i :=
  IsDedekindDomain.quotientEquivPiOfProdEq I (fun i : s => P i) (fun i : s => e i)
    (fun i => prime i i.2) (fun i j h => coprime i i.2 j j.2 (Subtype.coe_injective.ne h))
    (_root_.trans (Finset.prod_coe_sort s fun i => P i ^ e i) prod_eq)


/-- Corollary of the Chinese remainder theorem: given elements `x i : R / P i ^ e i`,
we can choose a representative `y : R` such that `y ≡ x i (mod P i ^ e i)`. -/
theorem IsDedekindDomain.exists_representative_mod_finset {ι : Type*} {s : Finset ι}
    (P : ι → Ideal R) (e : ι → ℕ) (prime : ∀ i ∈ s, Prime (P i))
    (coprime : ∀ᵉ (i ∈ s) (j ∈ s), i ≠ j → P i ≠ P j) (x : ∀ i : s, R ⧸ P i ^ e i) :
    ∃ y, ∀ (i) (hi : i ∈ s), Ideal.Quotient.mk (P i ^ e i) y = x ⟨i, hi⟩ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    ι : Type u_4
    s : Finset ι
    P : ι → Ideal R
    e : ι → Nat
    prime : ∀ (i : ι), Membership.mem s i → Prime (P i)
    coprime : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Ne i …
    x : (i : Subtype fun x => Membership.mem s x) → HasQuotient.Quotient R (HPow.h …
    ⊢ Exists fun y => ∀ (i : ι) (hi : Membership.mem s i), Eq ((Ideal.Quotient.mk  …
  -/
  let f := IsDedekindDomain.quotientEquivPiOfFinsetProdEq _ P e prime coprime rfl
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    ι : Type u_4
    s : Finset ι
    P : ι → Ideal R
    e : ι → Nat
    prime : ∀ (i : ι), Membership.mem s i → Prime (P i)
    coprime : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Ne i …
    x : (i : Subtype fun x => Membership.mem s x) → HasQuotient.Quotient R (HPow.h …
    f : RingEquiv (HasQuotient.Quotient R (s.prod fun i => HPow.hPow (P i) (e i))) …
    ⊢ Exists fun y => ∀ (i : ι) (hi : Membership.mem s i), Eq ((Ideal.Quotient.mk  …
  -/
  obtain ⟨y, rfl⟩ := f.surjective x
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    ι : Type u_4
    s : Finset ι
    P : ι → Ideal R
    e : ι → Nat
    prime : ∀ (i : ι), Membership.mem s i → Prime (P i)
    coprime : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Ne i …
    f : RingEquiv (HasQuotient.Quotient R (s.prod fun i => HPow.hPow (P i) (e i))) …
    y : HasQuotient.Quotient R (s.prod fun i => HPow.hPow (P i) (e i))
    ⊢ Exists fun y_1 => ∀ (i : ι) (hi : Membership.mem s i), Eq ((Ideal.Quotient.m …
  -/
  obtain ⟨z, rfl⟩ := Ideal.Quotient.mk_surjective y
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    ι : Type u_4
    s : Finset ι
    P : ι → Ideal R
    e : ι → Nat
    prime : ∀ (i : ι), Membership.mem s i → Prime (P i)
    coprime : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Ne i …
    f : RingEquiv (HasQuotient.Quotient R (s.prod fun i => HPow.hPow (P i) (e i))) …
    z : R
    ⊢ Exists fun y => ∀ (i : ι) (hi : Membership.mem s i), Eq ((Ideal.Quotient.mk  …
  -/
  exact ⟨z, fun i _hi => rfl⟩
  /-
    🎉 no goals
  -/


/-- Corollary of the Chinese remainder theorem: given elements `x i : R`,
we can choose a representative `y : R` such that `y - x i ∈ P i ^ e i`. -/
theorem IsDedekindDomain.exists_forall_sub_mem_ideal {ι : Type*} {s : Finset ι} (P : ι → Ideal R)
    (e : ι → ℕ) (prime : ∀ i ∈ s, Prime (P i))
    (coprime : ∀ᵉ (i ∈ s) (j ∈ s), i ≠ j → P i ≠ P j) (x : s → R) :
    ∃ y, ∀ (i) (hi : i ∈ s), y - x ⟨i, hi⟩ ∈ P i ^ e i := by
  obtain ⟨y, hy⟩ :=
    IsDedekindDomain.exists_representative_mod_finset P e prime coprime fun i =>
      Ideal.Quotient.mk _ (x i)
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDedekindDomain R
    ι : Type u_4
    s : Finset ι
    P : ι → Ideal R
    e : ι → Nat
    prime : ∀ (i : ι), Membership.mem s i → Prime (P i)
    coprime : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Ne i …
    x : (Subtype fun x => Membership.mem s x) → R
    y : R
    hy : ∀ (i : ι) (hi : Membership.mem s i), Eq ((Ideal.Quotient.mk (HPow.hPow (P …
    ⊢ Exists fun y => ∀ (i : ι) (hi : Membership.mem s i), Membership.mem (HPow.hP …
  -/
  exact ⟨y, fun i hi => Ideal.Quotient.eq.mp (hy i hi)⟩
  /-
    🎉 no goals
  -/


theorem span_singleton_dvd_span_singleton_iff_dvd {a b : R} :
    Ideal.span {a} ∣ Ideal.span ({b} : Set R) ↔ a ∣ b :=
  ⟨fun h => mem_span_singleton.mp (dvd_iff_le.mp h (mem_span_singleton.mpr (dvd_refl b))), fun h =>
    dvd_iff_le.mpr fun _d hd => mem_span_singleton.mpr (dvd_trans h (mem_span_singleton.mp hd))⟩


@[simp]
theorem Ideal.squarefree_span_singleton {a : R} :
    Squarefree (span {a}) ↔ Squarefree a := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    a : R
    ⊢ Iff (Squarefree (Ideal.span (Singleton.singleton a))) (Squarefree a)
  -/
  refine ⟨fun h x hx ↦ ?_, fun h I hI ↦ ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsPrincipalIdealRing R
      a : R
      h : Squarefree (Ideal.span (Singleton.singleton a))
      x : R
      hx : Dvd.dvd (HMul.hMul x x) a
      ⊢ IsUnit x
    -/
  · rw [← span_singleton_dvd_span_singleton_iff_dvd, ← span_singleton_mul_span_singleton] at hx
    /-
      case refine_1
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsPrincipalIdealRing R
      a : R
      h : Squarefree (Ideal.span (Singleton.singleton a))
      x : R
      hx : Dvd.dvd (HMul.hMul (Ideal.span (Singleton.singleton x)) (Ideal.span (Sing …
      ⊢ IsUnit x
    -/
    simpa using h _ hx
    /-
      🎉 no goals
    -/
  · rw [← span_singleton_generator I, span_singleton_mul_span_singleton,
      span_singleton_dvd_span_singleton_iff_dvd] at hI
    /-
      case refine_2
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsPrincipalIdealRing R
      a : R
      h : Squarefree a
      I : Ideal R
      hI : Dvd.dvd (HMul.hMul (Submodule.IsPrincipal.generator I) (Submodule.IsPrinc …
      ⊢ IsUnit I
    -/
    exact isUnit_iff.mpr <| eq_top_of_isUnit_mem _ (Submodule.IsPrincipal.generator_mem I) (h _ hI)
    /-
      🎉 no goals
    -/


theorem singleton_span_mem_normalizedFactors_of_mem_normalizedFactors [NormalizationMonoid R]
    {a b : R} (ha : a ∈ normalizedFactors b) :
    Ideal.span ({a} : Set R) ∈ normalizedFactors (Ideal.span ({b} : Set R)) := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    inst✝ : NormalizationMonoid R
    a b : R
    ha : Membership.mem (UniqueFactorizationMonoid.normalizedFactors b) a
    ⊢ Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Ideal.span (Sin …
  -/
  by_cases hb : b = 0
    /-
      case pos
      R : Type u_1
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : IsPrincipalIdealRing R
      inst✝ : NormalizationMonoid R
      a b : R
      ha : Membership.mem (UniqueFactorizationMonoid.normalizedFactors b) a
      hb : Eq b 0
      ⊢ Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Ideal.span (Sin …
    -/
  · rw [Ideal.span_singleton_eq_bot.mpr hb, bot_eq_zero, normalizedFactors_zero]
    /-
      case pos
      R : Type u_1
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : IsPrincipalIdealRing R
      inst✝ : NormalizationMonoid R
      a b : R
      ha : Membership.mem (UniqueFactorizationMonoid.normalizedFactors b) a
      hb : Eq b 0
      ⊢ Membership.mem 0 (Ideal.span (Singleton.singleton a))
    -/
    rw [hb, normalizedFactors_zero] at ha
    /-
      case pos
      R : Type u_1
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : IsPrincipalIdealRing R
      inst✝ : NormalizationMonoid R
      a b : R
      ha : Membership.mem 0 a
      hb : Eq b 0
      ⊢ Membership.mem 0 (Ideal.span (Singleton.singleton a))
    -/
    exact absurd ha (Multiset.not_mem_zero a)
    /-
      🎉 no goals
    -/
  · suffices Prime (Ideal.span ({a} : Set R)) by
      obtain ⟨c, hc, hc'⟩ := exists_mem_normalizedFactors_of_dvd ?_ this.irreducible
          (dvd_iff_le.mpr (span_singleton_le_span_singleton.mpr (dvd_of_mem_normalizedFactors ha)))
      rwa [associated_iff_eq.mp hc']
      /-
        case neg.refine_2
        R : Type u_1
        inst✝³ : CommRing R
        inst✝² : IsDomain R
        inst✝¹ : IsPrincipalIdealRing R
        inst✝ : NormalizationMonoid R
        a b : R
        ha : Membership.mem (UniqueFactorizationMonoid.normalizedFactors b) a
        hb : Not (Eq b 0)
        this : Prime (Ideal.span (Singleton.singleton a))
        ⊢ Ne (Ideal.span (Singleton.singleton b)) 0
      -/
    · by_contra h
      /-
        case neg.refine_2
        R : Type u_1
        inst✝³ : CommRing R
        inst✝² : IsDomain R
        inst✝¹ : IsPrincipalIdealRing R
        inst✝ : NormalizationMonoid R
        a b : R
        ha : Membership.mem (UniqueFactorizationMonoid.normalizedFactors b) a
        hb : Not (Eq b 0)
        this : Prime (Ideal.span (Singleton.singleton a))
        h : Eq (Ideal.span (Singleton.singleton b)) 0
        ⊢ False
      -/
      exact hb (span_singleton_eq_bot.mp h)
      /-
        🎉 no goals
      -/
    /-
      case neg.refine_1
      R : Type u_1
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : IsPrincipalIdealRing R
      inst✝ : NormalizationMonoid R
      a b : R
      ha : Membership.mem (UniqueFactorizationMonoid.normalizedFactors b) a
      hb : Not (Eq b 0)
      ⊢ Prime (Ideal.span (Singleton.singleton a))
    -/
    rw [prime_iff_isPrime]
    · exact (span_singleton_prime (prime_of_normalized_factor a ha).ne_zero).mpr
        (prime_of_normalized_factor a ha)
      /-
        case neg.refine_1
        R : Type u_1
        inst✝³ : CommRing R
        inst✝² : IsDomain R
        inst✝¹ : IsPrincipalIdealRing R
        inst✝ : NormalizationMonoid R
        a b : R
        ha : Membership.mem (UniqueFactorizationMonoid.normalizedFactors b) a
        hb : Not (Eq b 0)
        ⊢ Ne (Ideal.span (Singleton.singleton a)) Bot.bot
      -/
    · by_contra h
      /-
        case neg.refine_1
        R : Type u_1
        inst✝³ : CommRing R
        inst✝² : IsDomain R
        inst✝¹ : IsPrincipalIdealRing R
        inst✝ : NormalizationMonoid R
        a b : R
        ha : Membership.mem (UniqueFactorizationMonoid.normalizedFactors b) a
        hb : Not (Eq b 0)
        h : Eq (Ideal.span (Singleton.singleton a)) Bot.bot
        ⊢ False
      -/
      exact (prime_of_normalized_factor a ha).ne_zero (span_singleton_eq_bot.mp h)
      /-
        🎉 no goals
      -/


theorem emultiplicity_eq_emultiplicity_span {a b : R} :
    emultiplicity (Ideal.span {a}) (Ideal.span ({b} : Set R)) = emultiplicity a b := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    a b : R
    ⊢ Eq (emultiplicity (Ideal.span (Singleton.singleton a)) (Ideal.span (Singleto …
  -/
  by_cases h : FiniteMultiplicity a b
    /-
      case pos
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsPrincipalIdealRing R
      a b : R
      h : FiniteMultiplicity a b
      ⊢ Eq (emultiplicity (Ideal.span (Singleton.singleton a)) (Ideal.span (Singleto …
    -/
  · rw [h.emultiplicity_eq_multiplicity]
    /-
      case pos
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsPrincipalIdealRing R
      a b : R
      h : FiniteMultiplicity a b
      ⊢ Eq (emultiplicity (Ideal.span (Singleton.singleton a)) (Ideal.span (Singleto …
    -/
    apply emultiplicity_eq_of_dvd_of_not_dvd <;>
      /-
        case pos.hk
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : IsPrincipalIdealRing R
        a b : R
        h : FiniteMultiplicity a b
        ⊢ Dvd.dvd (HPow.hPow (Ideal.span (Singleton.singleton a)) (multiplicity a b))  …
      -/
      rw [Ideal.span_singleton_pow, span_singleton_dvd_span_singleton_iff_dvd]
      /-
        case pos.hk
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : IsPrincipalIdealRing R
        a b : R
        h : FiniteMultiplicity a b
        ⊢ Dvd.dvd (HPow.hPow a (multiplicity a b)) b
      -/
    · exact pow_multiplicity_dvd a b
      /-
        🎉 no goals
      -/
      /-
        case pos.hsucc
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : IsPrincipalIdealRing R
        a b : R
        h : FiniteMultiplicity a b
        ⊢ Not (Dvd.dvd (HPow.hPow a (HAdd.hAdd (multiplicity a b) 1)) b)
      -/
    · apply h.not_pow_dvd_of_multiplicity_lt
      /-
        case pos.hsucc
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : IsPrincipalIdealRing R
        a b : R
        h : FiniteMultiplicity a b
        ⊢ LT.lt (multiplicity a b) (HAdd.hAdd (multiplicity a b) 1)
      -/
      apply lt_add_one
      /-
        🎉 no goals
      -/
  · suffices ¬FiniteMultiplicity (Ideal.span ({a} : Set R)) (Ideal.span ({b} : Set R)) by
      rw [emultiplicity_eq_top.2 h, emultiplicity_eq_top.2 this]
    exact FiniteMultiplicity.not_iff_forall.mpr fun n => by
      rw [Ideal.span_singleton_pow, span_singleton_dvd_span_singleton_iff_dvd]
      exact FiniteMultiplicity.not_iff_forall.mp h n


/-- The bijection between the (normalized) prime factors of `r` and the (normalized) prime factors
    of `span {r}` -/
noncomputable def normalizedFactorsEquivSpanNormalizedFactors {r : R} (hr : r ≠ 0) :
    { d : R | d ∈ normalizedFactors r } ≃
      { I : Ideal R | I ∈ normalizedFactors (Ideal.span ({r} : Set R)) } := by
  /-
    R : Type u_1
    A : Type u_2
    K : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    inst✝ : NormalizationMonoid R
    r : R
    hr : Ne r 0
    ⊢ Equiv ↑(setOf fun d => Membership.mem (UniqueFactorizationMonoid.normalizedF …
  -/
  refine Equiv.ofBijective ?_ ?_
  · exact fun d =>
      ⟨Ideal.span {↑d}, singleton_span_mem_normalizedFactors_of_mem_normalizedFactors d.prop⟩
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      K : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Field K
      inst✝² : IsDomain R
      inst✝¹ : IsPrincipalIdealRing R
      inst✝ : NormalizationMonoid R
      r : R
      hr : Ne r 0
      ⊢ Function.Bijective fun d => ⟨Ideal.span (Singleton.singleton ↑d), ⋯⟩
    -/
  · refine ⟨?_, ?_⟩
      /-
        case refine_2.refine_1
        R : Type u_1
        A : Type u_2
        K : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Field K
        inst✝² : IsDomain R
        inst✝¹ : IsPrincipalIdealRing R
        inst✝ : NormalizationMonoid R
        r : R
        hr : Ne r 0
        ⊢ Function.Injective fun d => ⟨Ideal.span (Singleton.singleton ↑d), ⋯⟩
      -/
    · rintro ⟨a, ha⟩ ⟨b, hb⟩ h
      rw [Subtype.mk_eq_mk, Ideal.span_singleton_eq_span_singleton, Subtype.coe_mk,
          Subtype.coe_mk] at h
      /-
        case refine_2.refine_1.mk.mk
        R : Type u_1
        A : Type u_2
        K : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Field K
        inst✝² : IsDomain R
        inst✝¹ : IsPrincipalIdealRing R
        inst✝ : NormalizationMonoid R
        r : R
        hr : Ne r 0
        a : R
        ha : Membership.mem (setOf fun d => Membership.mem (UniqueFactorizationMonoid. …
        b : R
        hb : Membership.mem (setOf fun d => Membership.mem (UniqueFactorizationMonoid. …
        h : Associated a b
        ⊢ Eq ⟨a, ha⟩ ⟨b, hb⟩
      -/
      exact Subtype.mk_eq_mk.mpr (mem_normalizedFactors_eq_of_associated ha hb h)
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        R : Type u_1
        A : Type u_2
        K : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Field K
        inst✝² : IsDomain R
        inst✝¹ : IsPrincipalIdealRing R
        inst✝ : NormalizationMonoid R
        r : R
        hr : Ne r 0
        ⊢ Function.Surjective fun d => ⟨Ideal.span (Singleton.singleton ↑d), ⋯⟩
      -/
    · rintro ⟨i, hi⟩
      /-
        case refine_2.refine_2.mk
        R : Type u_1
        A : Type u_2
        K : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Field K
        inst✝² : IsDomain R
        inst✝¹ : IsPrincipalIdealRing R
        inst✝ : NormalizationMonoid R
        r : R
        hr : Ne r 0
        i : Ideal R
        hi : Membership.mem (setOf fun I => Membership.mem (UniqueFactorizationMonoid. …
        ⊢ Exists fun a => Eq ((fun d => ⟨Ideal.span (Singleton.singleton ↑d), ⋯⟩) a) ⟨ …
      -/
      have : i.IsPrime := isPrime_of_prime (prime_of_normalized_factor i hi)
      have := exists_mem_normalizedFactors_of_dvd hr
        (Submodule.IsPrincipal.prime_generator_of_isPrime i
        (prime_of_normalized_factor i hi).ne_zero).irreducible ?_
        /-
          case refine_2.refine_2.mk.refine_2
          R : Type u_1
          A : Type u_2
          K : Type u_3
          inst✝⁵ : CommRing R
          inst✝⁴ : CommRing A
          inst✝³ : Field K
          inst✝² : IsDomain R
          inst✝¹ : IsPrincipalIdealRing R
          inst✝ : NormalizationMonoid R
          r : R
          hr : Ne r 0
          i : Ideal R
          hi : Membership.mem (setOf fun I => Membership.mem (UniqueFactorizationMonoid. …
          this✝ : i.IsPrime
          this : Exists fun q => And (Membership.mem (UniqueFactorizationMonoid.normaliz …
          ⊢ Exists fun a => Eq ((fun d => ⟨Ideal.span (Singleton.singleton ↑d), ⋯⟩) a) ⟨ …
        -/
      · obtain ⟨a, ha, ha'⟩ := this
        /-
          case refine_2.refine_2.mk.refine_2.intro.intro
          R : Type u_1
          A : Type u_2
          K : Type u_3
          inst✝⁵ : CommRing R
          inst✝⁴ : CommRing A
          inst✝³ : Field K
          inst✝² : IsDomain R
          inst✝¹ : IsPrincipalIdealRing R
          inst✝ : NormalizationMonoid R
          r : R
          hr : Ne r 0
          i : Ideal R
          hi : Membership.mem (setOf fun I => Membership.mem (UniqueFactorizationMonoid. …
          this : i.IsPrime
          a : R
          ha : Membership.mem (UniqueFactorizationMonoid.normalizedFactors r) a
          ha' : Associated (Submodule.IsPrincipal.generator i) a
          ⊢ Exists fun a => Eq ((fun d => ⟨Ideal.span (Singleton.singleton ↑d), ⋯⟩) a) ⟨ …
        -/
        use ⟨a, ha⟩
        simp only [Subtype.coe_mk, Subtype.mk_eq_mk, ← span_singleton_eq_span_singleton.mpr ha',
            Ideal.span_singleton_generator]
      · exact (Submodule.IsPrincipal.mem_iff_generator_dvd i).mp
          ((show Ideal.span {r} ≤ i from dvd_iff_le.mp (dvd_of_mem_normalizedFactors hi))
            (mem_span_singleton.mpr (dvd_refl r)))


/-- The bijection `normalizedFactorsEquivSpanNormalizedFactors` between the set of prime
    factors of `r` and the set of prime factors of the ideal `⟨r⟩` preserves multiplicities. See
    `count_normalizedFactorsSpan_eq_count` for the version stated in terms of multisets `count`.-/
theorem emultiplicity_normalizedFactorsEquivSpanNormalizedFactors_eq_emultiplicity {r d : R}
    (hr : r ≠ 0) (hd : d ∈ normalizedFactors r) :
    emultiplicity d r =
      emultiplicity (normalizedFactorsEquivSpanNormalizedFactors hr ⟨d, hd⟩ : Ideal R)
        (Ideal.span {r}) := by
  simp only [normalizedFactorsEquivSpanNormalizedFactors, emultiplicity_eq_emultiplicity_span,
    Subtype.coe_mk, Equiv.ofBijective_apply]


/-- The bijection `normalized_factors_equiv_span_normalized_factors.symm` between the set of prime
    factors of the ideal `⟨r⟩` and the set of prime factors of `r` preserves multiplicities. -/
theorem emultiplicity_normalizedFactorsEquivSpanNormalizedFactors_symm_eq_emultiplicity {r : R}
    (hr : r ≠ 0) (I : { I : Ideal R | I ∈ normalizedFactors (Ideal.span ({r} : Set R)) }) :
    emultiplicity ((normalizedFactorsEquivSpanNormalizedFactors hr).symm I : R) r =
      emultiplicity (I : Ideal R) (Ideal.span {r}) := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    inst✝ : NormalizationMonoid R
    r : R
    hr : Ne r 0
    I : ↑(setOf fun I => Membership.mem (UniqueFactorizationMonoid.normalizedFacto …
    ⊢ Eq (emultiplicity (↑((normalizedFactorsEquivSpanNormalizedFactors hr).symm I …
  -/
  obtain ⟨x, hx⟩ := (normalizedFactorsEquivSpanNormalizedFactors hr).surjective I
  /-
    case intro
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    inst✝ : NormalizationMonoid R
    r : R
    hr : Ne r 0
    I : ↑(setOf fun I => Membership.mem (UniqueFactorizationMonoid.normalizedFacto …
    x : ↑(setOf fun d => Membership.mem (UniqueFactorizationMonoid.normalizedFacto …
    hx : Eq ((normalizedFactorsEquivSpanNormalizedFactors hr) x) I
    ⊢ Eq (emultiplicity (↑((normalizedFactorsEquivSpanNormalizedFactors hr).symm I …
  -/
  obtain ⟨a, ha⟩ := x
  rw [hx.symm, Equiv.symm_apply_apply, Subtype.coe_mk,
    emultiplicity_normalizedFactorsEquivSpanNormalizedFactors_eq_emultiplicity hr ha]


/-- The bijection between the set of prime factors of the ideal `⟨r⟩` and the set of prime factors
  of `r` preserves `count` of the corresponding multisets. See
  `multiplicity_normalizedFactorsEquivSpanNormalizedFactors_eq_multiplicity` for the version
  stated in terms of multiplicity. -/
theorem count_span_normalizedFactors_eq {r X : R} (hr : r ≠ 0) (hX : Prime X) :
    Multiset.count (Ideal.span {X} : Ideal R) (normalizedFactors (Ideal.span {r}))  =
        Multiset.count (normalize X) (normalizedFactors r) := by
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : NormalizationMonoid R
    inst✝¹ : DecidableEq R
    inst✝ : DecidableEq (Ideal R)
    r X : R
    hr : Ne r 0
    hX : Prime X
    ⊢ Eq (Multiset.count (Ideal.span (Singleton.singleton X)) (UniqueFactorization …
  -/
  have := emultiplicity_eq_emultiplicity_span (R := R) (a := X) (b := r)
  rw [emultiplicity_eq_count_normalizedFactors (Prime.irreducible hX) hr,
    emultiplicity_eq_count_normalizedFactors (Prime.irreducible ?_), normalize_apply,
    normUnit_eq_one, Units.val_one, one_eq_top, mul_top, Nat.cast_inj] at this
    /-
      R : Type u_1
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : NormalizationMonoid R
      inst✝¹ : DecidableEq R
      inst✝ : DecidableEq (Ideal R)
      r X : R
      hr : Ne r 0
      hX : Prime X
      this : Eq (Multiset.count (Ideal.span (Singleton.singleton X)) (UniqueFactoriz …
      ⊢ Eq (Multiset.count (Ideal.span (Singleton.singleton X)) (UniqueFactorization …
    -/
  · simp only [normalize_apply, this]
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : NormalizationMonoid R
      inst✝¹ : DecidableEq R
      inst✝ : DecidableEq (Ideal R)
      r X : R
      hr : Ne r 0
      hX : Prime X
      this : Eq (emultiplicity (Ideal.span (Singleton.singleton X)) (Ideal.span (Sin …
      ⊢ Ne (Ideal.span (Singleton.singleton r)) 0
    -/
  · simp only [Submodule.zero_eq_bot, ne_eq, span_singleton_eq_bot, hr, not_false_eq_true]
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : NormalizationMonoid R
      inst✝¹ : DecidableEq R
      inst✝ : DecidableEq (Ideal R)
      r X : R
      hr : Ne r 0
      hX : Prime X
      this : Eq (emultiplicity (Ideal.span (Singleton.singleton X)) (Ideal.span (Sin …
      ⊢ Prime (Ideal.span (Singleton.singleton X))
    -/
  · simpa only [prime_span_singleton_iff]
    /-
      🎉 no goals
    -/


theorem count_span_normalizedFactors_eq_of_normUnit {r X : R}
    (hr : r ≠ 0) (hX₁ : normUnit X = 1) (hX : Prime X) :
      Multiset.count (Ideal.span {X} : Ideal R) (normalizedFactors (Ideal.span {r})) =
        Multiset.count X (normalizedFactors r) := by
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : NormalizationMonoid R
    inst✝¹ : DecidableEq R
    inst✝ : DecidableEq (Ideal R)
    r X : R
    hr : Ne r 0
    hX₁ : Eq (NormalizationMonoid.normUnit X) 1
    hX : Prime X
    ⊢ Eq (Multiset.count (Ideal.span (Singleton.singleton X)) (UniqueFactorization …
  -/
  simpa [hX₁, normalize_apply] using count_span_normalizedFactors_eq hr hX
  /-
    🎉 no goals
  -/


open scoped Classical in
/-- The finite set of all prime factors of the pushforward of `p`. -/
noncomputable abbrev primesOverFinset {A : Type*} [CommRing A] (p : Ideal A) (B : Type*)
    [CommRing B] [IsDedekindDomain B] [Algebra A B] : Finset (Ideal B) :=
  (factors (p.map (algebraMap A B))).toFinset


include hpb in
theorem coe_primesOverFinset : primesOverFinset p B = primesOver p B := by
  classical
  ext P
  rw [primesOverFinset, factors_eq_normalizedFactors, Finset.mem_coe, Multiset.mem_toFinset]
  exact (P.mem_normalizedFactors_iff (map_ne_bot_of_ne_bot hpb)).trans <| Iff.intro
    (fun ⟨hPp, h⟩ => ⟨hPp, ⟨hpm.eq_of_le (comap_ne_top _ hPp.ne_top) (le_comap_of_map_le h)⟩⟩)
    (fun ⟨hPp, h⟩ => ⟨hPp, map_le_of_le_comap h.1.le⟩)


theorem primesOver_finite : (primesOver p B).Finite := by
  /-
    A : Type u_4
    inst✝⁵ : CommRing A
    p : Ideal A
    hpm : p.IsMaximal
    B : Type u_5
    inst✝⁴ : CommRing B
    inst✝³ : IsDedekindDomain B
    inst✝² : Algebra A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : Algebra.IsIntegral A B
    ⊢ (primesOver p B).Finite
  -/
  by_cases hpb : p = ⊥
    /-
      case pos
      A : Type u_4
      inst✝⁵ : CommRing A
      p : Ideal A
      hpm : p.IsMaximal
      B : Type u_5
      inst✝⁴ : CommRing B
      inst✝³ : IsDedekindDomain B
      inst✝² : Algebra A B
      inst✝¹ : NoZeroSMulDivisors A B
      inst✝ : Algebra.IsIntegral A B
      hpb : Eq p Bot.bot
      ⊢ (primesOver p B).Finite
    -/
  · rw [hpb] at hpm ⊢
    /-
      case pos
      A : Type u_4
      inst✝⁵ : CommRing A
      p : Ideal A
      hpm : Bot.bot.IsMaximal
      B : Type u_5
      inst✝⁴ : CommRing B
      inst✝³ : IsDedekindDomain B
      inst✝² : Algebra A B
      inst✝¹ : NoZeroSMulDivisors A B
      inst✝ : Algebra.IsIntegral A B
      hpb : Eq p Bot.bot
      ⊢ (primesOver Bot.bot B).Finite
    -/
    haveI : IsDomain A := IsDomain.of_bot_isPrime A
    /-
      case pos
      A : Type u_4
      inst✝⁵ : CommRing A
      p : Ideal A
      hpm : Bot.bot.IsMaximal
      B : Type u_5
      inst✝⁴ : CommRing B
      inst✝³ : IsDedekindDomain B
      inst✝² : Algebra A B
      inst✝¹ : NoZeroSMulDivisors A B
      inst✝ : Algebra.IsIntegral A B
      hpb : Eq p Bot.bot
      this : IsDomain A
      ⊢ (primesOver Bot.bot B).Finite
    -/
    rw [primesOver_bot A B]
    /-
      case pos
      A : Type u_4
      inst✝⁵ : CommRing A
      p : Ideal A
      hpm : Bot.bot.IsMaximal
      B : Type u_5
      inst✝⁴ : CommRing B
      inst✝³ : IsDedekindDomain B
      inst✝² : Algebra A B
      inst✝¹ : NoZeroSMulDivisors A B
      inst✝ : Algebra.IsIntegral A B
      hpb : Eq p Bot.bot
      this : IsDomain A
      ⊢ (Singleton.singleton Bot.bot).Finite
    -/
    exact Set.finite_singleton ⊥
    /-
      🎉 no goals
    -/
    /-
      case neg
      A : Type u_4
      inst✝⁵ : CommRing A
      p : Ideal A
      hpm : p.IsMaximal
      B : Type u_5
      inst✝⁴ : CommRing B
      inst✝³ : IsDedekindDomain B
      inst✝² : Algebra A B
      inst✝¹ : NoZeroSMulDivisors A B
      inst✝ : Algebra.IsIntegral A B
      hpb : Not (Eq p Bot.bot)
      ⊢ (primesOver p B).Finite
    -/
  · rw [← coe_primesOverFinset hpb B]
    /-
      case neg
      A : Type u_4
      inst✝⁵ : CommRing A
      p : Ideal A
      hpm : p.IsMaximal
      B : Type u_5
      inst✝⁴ : CommRing B
      inst✝³ : IsDedekindDomain B
      inst✝² : Algebra A B
      inst✝¹ : NoZeroSMulDivisors A B
      inst✝ : Algebra.IsIntegral A B
      hpb : Not (Eq p Bot.bot)
      ⊢ (↑(primesOverFinset p B)).Finite
    -/
    exact (primesOverFinset p B).finite_toSet
    /-
      🎉 no goals
    -/


theorem primesOver_ncard_ne_zero : (primesOver p B).ncard ≠ 0 := by
  /-
    A : Type u_4
    inst✝⁵ : CommRing A
    p : Ideal A
    hpm : p.IsMaximal
    B : Type u_5
    inst✝⁴ : CommRing B
    inst✝³ : IsDedekindDomain B
    inst✝² : Algebra A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : Algebra.IsIntegral A B
    ⊢ Ne (primesOver p B).ncard 0
  -/
  rcases exists_ideal_liesOver_maximal_of_isIntegral p B with ⟨P, hPm, hp⟩
  /-
    case intro.intro
    A : Type u_4
    inst✝⁵ : CommRing A
    p : Ideal A
    hpm : p.IsMaximal
    B : Type u_5
    inst✝⁴ : CommRing B
    inst✝³ : IsDedekindDomain B
    inst✝² : Algebra A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : Algebra.IsIntegral A B
    P : Ideal B
    hPm : P.IsMaximal
    hp : P.LiesOver p
    ⊢ Ne (primesOver p B).ncard 0
  -/
  exact Set.ncard_ne_zero_of_mem ⟨hPm.isPrime, hp⟩ (primesOver_finite p B)
  /-
    🎉 no goals
  -/


theorem one_le_primesOver_ncard : 1 ≤ (primesOver p B).ncard :=
  Nat.one_le_iff_ne_zero.mpr (primesOver_ncard_ne_zero p B)


