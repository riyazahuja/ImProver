/-- Auxiliary lemma showing that for any `k : ℤ` the function `z → 1/(c*z+d)^k` is
differentiable on `{z : ℂ | 0 < z.im}`. -/
lemma div_linear_zpow_differentiableOn (k : ℤ) (a : Fin 2 → ℤ) :
    DifferentiableOn ℂ (fun z : ℂ => (a 0 * z + a 1) ^ (-k)) {z : ℂ | 0 < z.im} := by
  /-
    k : Int
    a : Fin 2 → Int
    ⊢ DifferentiableOn Complex (fun z => HPow.hPow (HAdd.hAdd (HMul.hMul (↑(a 0))  …
  -/
  rcases ne_or_eq a 0 with ha | rfl
    /-
      case inl
      k : Int
      a : Fin 2 → Int
      ha : Ne a 0
      ⊢ DifferentiableOn Complex (fun z => HPow.hPow (HAdd.hAdd (HMul.hMul (↑(a 0))  …
    -/
  · apply DifferentiableOn.zpow
      /-
        case inl.hf
        k : Int
        a : Fin 2 → Int
        ha : Ne a 0
        ⊢ DifferentiableOn Complex (fun x => HAdd.hAdd (HMul.hMul (↑(a 0)) x) ↑(a 1))  …
      -/
    · fun_prop
      /-
        🎉 no goals
      -/
      /-
        case inl.h
        k : Int
        a : Fin 2 → Int
        ha : Ne a 0
        ⊢ Or (∀ (x : Complex), Membership.mem (setOf fun z => LT.lt 0 z.im) x → Ne (HA …
      -/
    · left
      exact fun z hz ↦ linear_ne_zero _ ⟨z, hz⟩
        ((comp_ne_zero_iff _ Int.cast_injective Int.cast_zero).mpr ha)
    /-
      case inr
      k : Int
      ⊢ DifferentiableOn Complex (fun z => HPow.hPow (HAdd.hAdd (HMul.hMul (↑(0 0))  …
    -/
  · simp only [Fin.isValue, Pi.zero_apply, Int.cast_zero, zero_mul, add_zero, one_div]
    /-
      case inr
      k : Int
      ⊢ DifferentiableOn Complex (fun z => HPow.hPow 0 (Neg.neg k)) (setOf fun z =>  …
    -/
    apply differentiableOn_const
    /-
      🎉 no goals
    -/


/-- Auxiliary lemma showing that for any `k : ℤ` and `(a : Fin 2 → ℤ)`
the extension of `eisSummand` is differentiable on `{z : ℂ | 0 < z.im}`.-/
lemma eisSummand_extension_differentiableOn (k : ℤ) (a : Fin 2 → ℤ) :
    DifferentiableOn ℂ (↑ₕeisSummand k a) {z : ℂ | 0 < z.im} := by
  /-
    k : Int
    a : Fin 2 → Int
    ⊢ DifferentiableOn Complex (Function.comp (EisensteinSeries.eisSummand k a) ↑U …
  -/
  apply DifferentiableOn.congr (div_linear_zpow_differentiableOn k a)
  /-
    k : Int
    a : Fin 2 → Int
    ⊢ ∀ (x : Complex), Membership.mem (setOf fun z => LT.lt 0 z.im) x → Eq (Functi …
  -/
  intro z hz
  /-
    k : Int
    a : Fin 2 → Int
    z : Complex
    hz : Membership.mem (setOf fun z => LT.lt 0 z.im) z
    ⊢ Eq (Function.comp (EisensteinSeries.eisSummand k a) (↑UpperHalfPlane.ofCompl …
  -/
  lift z to ℍ using hz
  /-
    case intro
    k : Int
    a : Fin 2 → Int
    z : UpperHalfPlane
    ⊢ Eq (Function.comp (EisensteinSeries.eisSummand k a) ↑UpperHalfPlane.ofComple …
  -/
  apply comp_ofComplex
  /-
    🎉 no goals
  -/


/-- Eisenstein series are MDifferentiable (i.e. holomorphic functions from `ℍ → ℂ`). -/
theorem eisensteinSeries_SIF_MDifferentiable {k : ℤ} {N : ℕ} (hk : 3 ≤ k) (a : Fin 2 → ZMod N) :
    MDifferentiable 𝓘(ℂ) 𝓘(ℂ) (eisensteinSeries_SIF a k) := by
  /-
    k : Int
    N : Nat
    hk : LE.le 3 k
    a : Fin 2 → ZMod N
    ⊢ MDifferentiable (modelWithCornersSelf Complex Complex) (modelWithCornersSelf …
  -/
  intro τ
  suffices DifferentiableAt ℂ (↑ₕeisensteinSeries_SIF a k) τ.1 by
    convert MDifferentiableAt.comp τ (DifferentiableAt.mdifferentiableAt this) τ.mdifferentiable_coe
    exact funext fun z ↦ (comp_ofComplex (eisensteinSeries_SIF a k) z).symm
  refine DifferentiableOn.differentiableAt ?_
    ((isOpen_lt continuous_const Complex.continuous_im).mem_nhds τ.2)
  exact (eisensteinSeries_tendstoLocallyUniformlyOn hk a).differentiableOn
    (Eventually.of_forall fun s ↦ DifferentiableOn.sum
      fun _ _ ↦ eisSummand_extension_differentiableOn _ _)
        (isOpen_lt continuous_const continuous_im)


