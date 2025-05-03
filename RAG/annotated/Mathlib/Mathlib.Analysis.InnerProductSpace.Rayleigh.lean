local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


/-- The *Rayleigh quotient* of a continuous linear map `T` (over `ℝ` or `ℂ`) at a vector `x` is
the quantity `re ⟪T x, x⟫ / ‖x‖ ^ 2`. -/
noncomputable abbrev rayleighQuotient (x : E) := T.reApplyInnerSelf x / ‖(x : E)‖ ^ 2


theorem rayleigh_smul (x : E) {c : 𝕜} (hc : c ≠ 0) :
    rayleighQuotient T (c • x) = rayleighQuotient T x := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    x : E
    c : 𝕜
    hc : Ne c 0
    ⊢ Eq (T.rayleighQuotient (HSMul.hSMul c x)) (T.rayleighQuotient x)
  -/
  by_cases hx : x = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : ContinuousLinearMap (RingHom.id 𝕜) E E
      x : E
      c : 𝕜
      hc : Ne c 0
      hx : Eq x 0
      ⊢ Eq (T.rayleighQuotient (HSMul.hSMul c x)) (T.rayleighQuotient x)
    -/
  · simp [hx]
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    x : E
    c : 𝕜
    hc : Ne c 0
    hx : Not (Eq x 0)
    ⊢ Eq (T.rayleighQuotient (HSMul.hSMul c x)) (T.rayleighQuotient x)
  -/
  field_simp [norm_smul, T.reApplyInnerSelf_smul]
  /-
    case neg
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    x : E
    c : 𝕜
    hc : Ne c 0
    hx : Not (Eq x 0)
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (Norm.norm c) 2) (T.reApplyInnerSelf x)) …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem image_rayleigh_eq_image_rayleigh_sphere {r : ℝ} (hr : 0 < r) :
    rayleighQuotient T '' {0}ᶜ = rayleighQuotient T '' sphere 0 r := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq (Set.image T.rayleighQuotient (HasCompl.compl (Singleton.singleton 0))) ( …
  -/
  ext a
  /-
    case h
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    r : Real
    hr : LT.lt 0 r
    a : Real
    ⊢ Iff (Membership.mem (Set.image T.rayleighQuotient (HasCompl.compl (Singleton …
  -/
  constructor
    /-
      case h.mp
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : ContinuousLinearMap (RingHom.id 𝕜) E E
      r : Real
      hr : LT.lt 0 r
      a : Real
      ⊢ Membership.mem (Set.image T.rayleighQuotient (HasCompl.compl (Singleton.sing …
    -/
  · rintro ⟨x, hx : x ≠ 0, hxT⟩
    /-
      case h.mp.intro.intro
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : ContinuousLinearMap (RingHom.id 𝕜) E E
      r : Real
      hr : LT.lt 0 r
      a : Real
      x : E
      hx : Ne x 0
      hxT : Eq (T.rayleighQuotient x) a
      ⊢ Membership.mem (Set.image T.rayleighQuotient (Metric.sphere 0 r)) a
    -/
    have : ‖x‖ ≠ 0 := by simp [hx]
    /-
      case h.mp.intro.intro
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : ContinuousLinearMap (RingHom.id 𝕜) E E
      r : Real
      hr : LT.lt 0 r
      a : Real
      x : E
      hx : Ne x 0
      hxT : Eq (T.rayleighQuotient x) a
      this : Ne (Norm.norm x) 0
      ⊢ Membership.mem (Set.image T.rayleighQuotient (Metric.sphere 0 r)) a
    -/
    let c : 𝕜 := ↑‖x‖⁻¹ * r
    /-
      case h.mp.intro.intro
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : ContinuousLinearMap (RingHom.id 𝕜) E E
      r : Real
      hr : LT.lt 0 r
      a : Real
      x : E
      hx : Ne x 0
      hxT : Eq (T.rayleighQuotient x) a
      this : Ne (Norm.norm x) 0
      c : 𝕜 := HMul.hMul ↑(Inv.inv (Norm.norm x)) ↑r
      ⊢ Membership.mem (Set.image T.rayleighQuotient (Metric.sphere 0 r)) a
    -/
    have : c ≠ 0 := by simp [c, hx, hr.ne']
    /-
      case h.mp.intro.intro
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : ContinuousLinearMap (RingHom.id 𝕜) E E
      r : Real
      hr : LT.lt 0 r
      a : Real
      x : E
      hx : Ne x 0
      hxT : Eq (T.rayleighQuotient x) a
      this✝ : Ne (Norm.norm x) 0
      c : 𝕜 := HMul.hMul ↑(Inv.inv (Norm.norm x)) ↑r
      this : Ne c 0
      ⊢ Membership.mem (Set.image T.rayleighQuotient (Metric.sphere 0 r)) a
    -/
    refine ⟨c • x, ?_, ?_⟩
      /-
        case h.mp.intro.intro.refine_1
        𝕜 : Type u_1
        inst✝² : RCLike 𝕜
        E : Type u_2
        inst✝¹ : NormedAddCommGroup E
        inst✝ : InnerProductSpace 𝕜 E
        T : ContinuousLinearMap (RingHom.id 𝕜) E E
        r : Real
        hr : LT.lt 0 r
        a : Real
        x : E
        hx : Ne x 0
        hxT : Eq (T.rayleighQuotient x) a
        this✝ : Ne (Norm.norm x) 0
        c : 𝕜 := HMul.hMul ↑(Inv.inv (Norm.norm x)) ↑r
        this : Ne c 0
        ⊢ Membership.mem (Metric.sphere 0 r) (HSMul.hSMul c x)
      -/
    · field_simp [c, norm_smul, abs_of_pos hr]
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.intro.refine_2
        𝕜 : Type u_1
        inst✝² : RCLike 𝕜
        E : Type u_2
        inst✝¹ : NormedAddCommGroup E
        inst✝ : InnerProductSpace 𝕜 E
        T : ContinuousLinearMap (RingHom.id 𝕜) E E
        r : Real
        hr : LT.lt 0 r
        a : Real
        x : E
        hx : Ne x 0
        hxT : Eq (T.rayleighQuotient x) a
        this✝ : Ne (Norm.norm x) 0
        c : 𝕜 := HMul.hMul ↑(Inv.inv (Norm.norm x)) ↑r
        this : Ne c 0
        ⊢ Eq (T.rayleighQuotient (HSMul.hSMul c x)) a
      -/
    · rw [T.rayleigh_smul x this]
      /-
        case h.mp.intro.intro.refine_2
        𝕜 : Type u_1
        inst✝² : RCLike 𝕜
        E : Type u_2
        inst✝¹ : NormedAddCommGroup E
        inst✝ : InnerProductSpace 𝕜 E
        T : ContinuousLinearMap (RingHom.id 𝕜) E E
        r : Real
        hr : LT.lt 0 r
        a : Real
        x : E
        hx : Ne x 0
        hxT : Eq (T.rayleighQuotient x) a
        this✝ : Ne (Norm.norm x) 0
        c : 𝕜 := HMul.hMul ↑(Inv.inv (Norm.norm x)) ↑r
        this : Ne c 0
        ⊢ Eq (T.rayleighQuotient x) a
      -/
      exact hxT
      /-
        🎉 no goals
      -/
    /-
      case h.mpr
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : ContinuousLinearMap (RingHom.id 𝕜) E E
      r : Real
      hr : LT.lt 0 r
      a : Real
      ⊢ Membership.mem (Set.image T.rayleighQuotient (Metric.sphere 0 r)) a → Member …
    -/
  · rintro ⟨x, hx, hxT⟩
    /-
      case h.mpr.intro.intro
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : ContinuousLinearMap (RingHom.id 𝕜) E E
      r : Real
      hr : LT.lt 0 r
      a : Real
      x : E
      hx : Membership.mem (Metric.sphere 0 r) x
      hxT : Eq (T.rayleighQuotient x) a
      ⊢ Membership.mem (Set.image T.rayleighQuotient (HasCompl.compl (Singleton.sing …
    -/
    exact ⟨x, ne_zero_of_mem_sphere hr.ne' ⟨x, hx⟩, hxT⟩
    /-
      🎉 no goals
    -/


theorem iSup_rayleigh_eq_iSup_rayleigh_sphere {r : ℝ} (hr : 0 < r) :
    ⨆ x : { x : E // x ≠ 0 }, rayleighQuotient T x =
      ⨆ x : sphere (0 : E) r, rayleighQuotient T x :=
  show ⨆ x : ({0}ᶜ : Set E), rayleighQuotient T x = _ by
    simp only [← @sSup_image' _ _ _ _ (rayleighQuotient T),
      T.image_rayleigh_eq_image_rayleigh_sphere hr]


theorem iInf_rayleigh_eq_iInf_rayleigh_sphere {r : ℝ} (hr : 0 < r) :
    ⨅ x : { x : E // x ≠ 0 }, rayleighQuotient T x =
      ⨅ x : sphere (0 : E) r, rayleighQuotient T x :=
  show ⨅ x : ({0}ᶜ : Set E), rayleighQuotient T x = _ by
    simp only [← @sInf_image' _ _ _ _ (rayleighQuotient T),
      T.image_rayleigh_eq_image_rayleigh_sphere hr]


theorem _root_.LinearMap.IsSymmetric.hasStrictFDerivAt_reApplyInnerSelf {T : F →L[ℝ] F}
    (hT : (T : F →ₗ[ℝ] F).IsSymmetric) (x₀ : F) :
    HasStrictFDerivAt T.reApplyInnerSelf (2 • (innerSL ℝ (T x₀))) x₀ := by
  /-
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : (↑T).IsSymmetric
    x₀ : F
    ⊢ HasStrictFDerivAt T.reApplyInnerSelf (HSMul.hSMul 2 ((innerSL Real) (T x₀))) …
  -/
  convert T.hasStrictFDerivAt.inner ℝ (hasStrictFDerivAt_id x₀) using 1
  /-
    case h.e'_12.h.h.h
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : (↑T).IsSymmetric
    x₀ : F
    e_4✝ : Eq NormedAddCommGroup.toAddCommGroup SeminormedAddCommGroup.toAddCommGr …
    he✝¹ : Eq NormedSpace.toModule NormedSpace.toModule
    e_8✝ : Eq Real.instAddCommGroup SeminormedAddCommGroup.toAddCommGroup
    he✝ : Eq NormedSpace.toModule NormedSpace.toModule
    ⊢ Eq (HSMul.hSMul 2 ((innerSL Real) (T x₀))) ((fderivInnerCLM Real { fst := T  …
  -/
  ext y
  rw [ContinuousLinearMap.smul_apply, ContinuousLinearMap.comp_apply, fderivInnerCLM_apply,
    ContinuousLinearMap.prod_apply, innerSL_apply, id, ContinuousLinearMap.id_apply,
    hT.apply_clm x₀ y, real_inner_comm _ x₀, two_smul]


theorem linearly_dependent_of_isLocalExtrOn (hT : IsSelfAdjoint T) {x₀ : F}
    (hextr : IsLocalExtrOn T.reApplyInnerSelf (sphere (0 : F) ‖x₀‖) x₀) :
    ∃ a b : ℝ, (a, b) ≠ 0 ∧ a • x₀ + b • T x₀ = 0 := by
  have H : IsLocalExtrOn T.reApplyInnerSelf {x : F | ‖x‖ ^ 2 = ‖x₀‖ ^ 2} x₀ := by
    convert hextr
    ext x
    simp [dist_eq_norm]
  -- find Lagrange multipliers for the function `T.re_apply_inner_self` and the
  -- hypersurface-defining function `fun x ↦ ‖x‖ ^ 2`
  obtain ⟨a, b, h₁, h₂⟩ :=
    IsLocalExtrOn.exists_multipliers_of_hasStrictFDerivAt_1d H (hasStrictFDerivAt_norm_sq x₀)
      (hT.isSymmetric.hasStrictFDerivAt_reApplyInnerSelf x₀)
  /-
    case intro.intro.intro
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : IsSelfAdjoint T
    x₀ : F
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    H : IsLocalExtrOn T.reApplyInnerSelf (setOf fun x => Eq (HPow.hPow (Norm.norm  …
    a b : Real
    h₁ : Ne { fst := a, snd := b } 0
    h₂ : Eq (HAdd.hAdd (HSMul.hSMul a (HSMul.hSMul 2 ((innerSL Real) x₀))) (HSMul. …
    ⊢ Exists fun a => Exists fun b => And (Ne { fst := a, snd := b } 0) (Eq (HAdd. …
  -/
  refine ⟨a, b, h₁, ?_⟩
  /-
    case intro.intro.intro
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : IsSelfAdjoint T
    x₀ : F
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    H : IsLocalExtrOn T.reApplyInnerSelf (setOf fun x => Eq (HPow.hPow (Norm.norm  …
    a b : Real
    h₁ : Ne { fst := a, snd := b } 0
    h₂ : Eq (HAdd.hAdd (HSMul.hSMul a (HSMul.hSMul 2 ((innerSL Real) x₀))) (HSMul. …
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul a x₀) (HSMul.hSMul b (T x₀))) 0
  -/
  apply (InnerProductSpace.toDualMap ℝ F).injective
  /-
    case intro.intro.intro.a
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : IsSelfAdjoint T
    x₀ : F
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    H : IsLocalExtrOn T.reApplyInnerSelf (setOf fun x => Eq (HPow.hPow (Norm.norm  …
    a b : Real
    h₁ : Ne { fst := a, snd := b } 0
    h₂ : Eq (HAdd.hAdd (HSMul.hSMul a (HSMul.hSMul 2 ((innerSL Real) x₀))) (HSMul. …
    ⊢ Eq ((InnerProductSpace.toDualMap Real F) (HAdd.hAdd (HSMul.hSMul a x₀) (HSMu …
  -/
  simp only [LinearIsometry.map_add, LinearIsometry.map_smul, LinearIsometry.map_zero]
  -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 changed `map_smulₛₗ` into `map_smulₛₗ _`
  /-
    case intro.intro.intro.a
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : IsSelfAdjoint T
    x₀ : F
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    H : IsLocalExtrOn T.reApplyInnerSelf (setOf fun x => Eq (HPow.hPow (Norm.norm  …
    a b : Real
    h₁ : Ne { fst := a, snd := b } 0
    h₂ : Eq (HAdd.hAdd (HSMul.hSMul a (HSMul.hSMul 2 ((innerSL Real) x₀))) (HSMul. …
    ⊢ Eq (HAdd.hAdd ((InnerProductSpace.toDualMap Real F) (HSMul.hSMul a x₀)) ((In …
  -/
  simp only [map_smulₛₗ _, RCLike.conj_to_real]
  /-
    case intro.intro.intro.a
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : IsSelfAdjoint T
    x₀ : F
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    H : IsLocalExtrOn T.reApplyInnerSelf (setOf fun x => Eq (HPow.hPow (Norm.norm  …
    a b : Real
    h₁ : Ne { fst := a, snd := b } 0
    h₂ : Eq (HAdd.hAdd (HSMul.hSMul a (HSMul.hSMul 2 ((innerSL Real) x₀))) (HSMul. …
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul a ((InnerProductSpace.toDualMap Real F) x₀)) (HSM …
  -/
  change a • innerSL ℝ x₀ + b • innerSL ℝ (T x₀) = 0
  /-
    case intro.intro.intro.a
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : IsSelfAdjoint T
    x₀ : F
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    H : IsLocalExtrOn T.reApplyInnerSelf (setOf fun x => Eq (HPow.hPow (Norm.norm  …
    a b : Real
    h₁ : Ne { fst := a, snd := b } 0
    h₂ : Eq (HAdd.hAdd (HSMul.hSMul a (HSMul.hSMul 2 ((innerSL Real) x₀))) (HSMul. …
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul a ((innerSL Real) x₀)) (HSMul.hSMul b ((innerSL R …
  -/
  apply smul_right_injective (F →L[ℝ] ℝ) (two_ne_zero : (2 : ℝ) ≠ 0)
  /-
    case intro.intro.intro.a.a
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : IsSelfAdjoint T
    x₀ : F
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    H : IsLocalExtrOn T.reApplyInnerSelf (setOf fun x => Eq (HPow.hPow (Norm.norm  …
    a b : Real
    h₁ : Ne { fst := a, snd := b } 0
    h₂ : Eq (HAdd.hAdd (HSMul.hSMul a (HSMul.hSMul 2 ((innerSL Real) x₀))) (HSMul. …
    ⊢ Eq ((fun x => HSMul.hSMul 2 x) (HAdd.hAdd (HSMul.hSMul a ((innerSL Real) x₀) …
  -/
  simpa only [two_smul, smul_add, add_smul, add_zero] using h₂
  /-
    🎉 no goals
  -/


open scoped InnerProductSpace in
theorem eq_smul_self_of_isLocalExtrOn_real (hT : IsSelfAdjoint T) {x₀ : F}
    (hextr : IsLocalExtrOn T.reApplyInnerSelf (sphere (0 : F) ‖x₀‖) x₀) :
    T x₀ = T.rayleighQuotient x₀ • x₀ := by
  /-
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : IsSelfAdjoint T
    x₀ : F
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    ⊢ Eq (T x₀) (HSMul.hSMul (T.rayleighQuotient x₀) x₀)
  -/
  obtain ⟨a, b, h₁, h₂⟩ := hT.linearly_dependent_of_isLocalExtrOn hextr
  /-
    case intro.intro.intro
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : IsSelfAdjoint T
    x₀ : F
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    a b : Real
    h₁ : Ne { fst := a, snd := b } 0
    h₂ : Eq (HAdd.hAdd (HSMul.hSMul a x₀) (HSMul.hSMul b (T x₀))) 0
    ⊢ Eq (T x₀) (HSMul.hSMul (T.rayleighQuotient x₀) x₀)
  -/
  by_cases hx₀ : x₀ = 0
    /-
      case pos
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : InnerProductSpace Real F
      inst✝ : CompleteSpace F
      T : ContinuousLinearMap (RingHom.id Real) F F
      hT : IsSelfAdjoint T
      x₀ : F
      hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
      a b : Real
      h₁ : Ne { fst := a, snd := b } 0
      h₂ : Eq (HAdd.hAdd (HSMul.hSMul a x₀) (HSMul.hSMul b (T x₀))) 0
      hx₀ : Eq x₀ 0
      ⊢ Eq (T x₀) (HSMul.hSMul (T.rayleighQuotient x₀) x₀)
    -/
  · simp [hx₀]
    /-
      🎉 no goals
    -/
  /-
    case neg
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : IsSelfAdjoint T
    x₀ : F
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    a b : Real
    h₁ : Ne { fst := a, snd := b } 0
    h₂ : Eq (HAdd.hAdd (HSMul.hSMul a x₀) (HSMul.hSMul b (T x₀))) 0
    hx₀ : Not (Eq x₀ 0)
    ⊢ Eq (T x₀) (HSMul.hSMul (T.rayleighQuotient x₀) x₀)
  -/
  by_cases hb : b = 0
    /-
      case pos
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : InnerProductSpace Real F
      inst✝ : CompleteSpace F
      T : ContinuousLinearMap (RingHom.id Real) F F
      hT : IsSelfAdjoint T
      x₀ : F
      hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
      a b : Real
      h₁ : Ne { fst := a, snd := b } 0
      h₂ : Eq (HAdd.hAdd (HSMul.hSMul a x₀) (HSMul.hSMul b (T x₀))) 0
      hx₀ : Not (Eq x₀ 0)
      hb : Eq b 0
      ⊢ Eq (T x₀) (HSMul.hSMul (T.rayleighQuotient x₀) x₀)
    -/
  · have : a ≠ 0 := by simpa [hb] using h₁
    /-
      case pos
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : InnerProductSpace Real F
      inst✝ : CompleteSpace F
      T : ContinuousLinearMap (RingHom.id Real) F F
      hT : IsSelfAdjoint T
      x₀ : F
      hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
      a b : Real
      h₁ : Ne { fst := a, snd := b } 0
      h₂ : Eq (HAdd.hAdd (HSMul.hSMul a x₀) (HSMul.hSMul b (T x₀))) 0
      hx₀ : Not (Eq x₀ 0)
      hb : Eq b 0
      this : Ne a 0
      ⊢ Eq (T x₀) (HSMul.hSMul (T.rayleighQuotient x₀) x₀)
    -/
    refine absurd ?_ hx₀
    /-
      case pos
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : InnerProductSpace Real F
      inst✝ : CompleteSpace F
      T : ContinuousLinearMap (RingHom.id Real) F F
      hT : IsSelfAdjoint T
      x₀ : F
      hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
      a b : Real
      h₁ : Ne { fst := a, snd := b } 0
      h₂ : Eq (HAdd.hAdd (HSMul.hSMul a x₀) (HSMul.hSMul b (T x₀))) 0
      hx₀ : Not (Eq x₀ 0)
      hb : Eq b 0
      this : Ne a 0
      ⊢ Eq x₀ 0
    -/
    apply smul_right_injective F this
    /-
      case pos.a
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : InnerProductSpace Real F
      inst✝ : CompleteSpace F
      T : ContinuousLinearMap (RingHom.id Real) F F
      hT : IsSelfAdjoint T
      x₀ : F
      hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
      a b : Real
      h₁ : Ne { fst := a, snd := b } 0
      h₂ : Eq (HAdd.hAdd (HSMul.hSMul a x₀) (HSMul.hSMul b (T x₀))) 0
      hx₀ : Not (Eq x₀ 0)
      hb : Eq b 0
      this : Ne a 0
      ⊢ Eq ((fun x => HSMul.hSMul a x) x₀) ((fun x => HSMul.hSMul a x) 0)
    -/
    simpa [hb] using h₂
    /-
      🎉 no goals
    -/
  /-
    case neg
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : IsSelfAdjoint T
    x₀ : F
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    a b : Real
    h₁ : Ne { fst := a, snd := b } 0
    h₂ : Eq (HAdd.hAdd (HSMul.hSMul a x₀) (HSMul.hSMul b (T x₀))) 0
    hx₀ : Not (Eq x₀ 0)
    hb : Not (Eq b 0)
    ⊢ Eq (T x₀) (HSMul.hSMul (T.rayleighQuotient x₀) x₀)
  -/
  let c : ℝ := -b⁻¹ * a
  have hc : T x₀ = c • x₀ := by
    have : b * (b⁻¹ * a) = a := by field_simp [mul_comm]
    apply smul_right_injective F hb
    simp [c, eq_neg_of_add_eq_zero_left h₂, ← mul_smul, this]
  /-
    case neg
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : IsSelfAdjoint T
    x₀ : F
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    a b : Real
    h₁ : Ne { fst := a, snd := b } 0
    h₂ : Eq (HAdd.hAdd (HSMul.hSMul a x₀) (HSMul.hSMul b (T x₀))) 0
    hx₀ : Not (Eq x₀ 0)
    hb : Not (Eq b 0)
    c : Real := HMul.hMul (Neg.neg (Inv.inv b)) a
    hc : Eq (T x₀) (HSMul.hSMul c x₀)
    ⊢ Eq (T x₀) (HSMul.hSMul (T.rayleighQuotient x₀) x₀)
  -/
  convert hc
  /-
    case h.e'_3.h.e'_5
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : IsSelfAdjoint T
    x₀ : F
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    a b : Real
    h₁ : Ne { fst := a, snd := b } 0
    h₂ : Eq (HAdd.hAdd (HSMul.hSMul a x₀) (HSMul.hSMul b (T x₀))) 0
    hx₀ : Not (Eq x₀ 0)
    hb : Not (Eq b 0)
    c : Real := HMul.hMul (Neg.neg (Inv.inv b)) a
    hc : Eq (T x₀) (HSMul.hSMul c x₀)
    ⊢ Eq (T.rayleighQuotient x₀) c
  -/
  have := congr_arg (fun x => ⟪x, x₀⟫_ℝ) hc
  /-
    case h.e'_3.h.e'_5
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : IsSelfAdjoint T
    x₀ : F
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    a b : Real
    h₁ : Ne { fst := a, snd := b } 0
    h₂ : Eq (HAdd.hAdd (HSMul.hSMul a x₀) (HSMul.hSMul b (T x₀))) 0
    hx₀ : Not (Eq x₀ 0)
    hb : Not (Eq b 0)
    c : Real := HMul.hMul (Neg.neg (Inv.inv b)) a
    hc : Eq (T x₀) (HSMul.hSMul c x₀)
    this : Eq ((fun x => Inner.inner x x₀) (T x₀)) ((fun x => Inner.inner x x₀) (H …
    ⊢ Eq (T.rayleighQuotient x₀) c
  -/
  field_simp [inner_smul_left, real_inner_self_eq_norm_mul_norm, sq] at this ⊢
  /-
    case h.e'_3.h.e'_5
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id Real) F F
    hT : IsSelfAdjoint T
    x₀ : F
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    a b : Real
    h₁ : Ne { fst := a, snd := b } 0
    h₂ : Eq (HAdd.hAdd (HSMul.hSMul a x₀) (HSMul.hSMul b (T x₀))) 0
    hx₀ : Not (Eq x₀ 0)
    hb : Not (Eq b 0)
    c : Real := HMul.hMul (Neg.neg (Inv.inv b)) a
    hc : Eq (T x₀) (HSMul.hSMul c x₀)
    this : Eq (Inner.inner (T x₀) x₀) (HMul.hMul c (HMul.hMul (Norm.norm x₀) (Norm …
    ⊢ Eq (T.reApplyInnerSelf x₀) (HMul.hMul c (HMul.hMul (Norm.norm x₀) (Norm.norm …
  -/
  exact this
  /-
    🎉 no goals
  -/


theorem eq_smul_self_of_isLocalExtrOn (hT : IsSelfAdjoint T) {x₀ : E}
    (hextr : IsLocalExtrOn T.reApplyInnerSelf (sphere (0 : E) ‖x₀‖) x₀) :
    T x₀ = (↑(T.rayleighQuotient x₀) : 𝕜) • x₀ := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    ⊢ Eq (T x₀) (HSMul.hSMul (↑(T.rayleighQuotient x₀)) x₀)
  -/
  letI := InnerProductSpace.rclikeToReal 𝕜 E
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    this : InnerProductSpace Real E := InnerProductSpace.rclikeToReal 𝕜 E
    ⊢ Eq (T x₀) (HSMul.hSMul (↑(T.rayleighQuotient x₀)) x₀)
  -/
  let hSA := hT.isSymmetric.restrictScalars.toSelfAdjoint.prop
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    this : InnerProductSpace Real E := InnerProductSpace.rclikeToReal 𝕜 E
    hSA : Membership.mem (selfAdjoint (ContinuousLinearMap (RingHom.id Real) E E)) …
    ⊢ Eq (T x₀) (HSMul.hSMul (↑(T.rayleighQuotient x₀)) x₀)
  -/
  exact hSA.eq_smul_self_of_isLocalExtrOn_real hextr
  /-
    🎉 no goals
  -/


/-- For a self-adjoint operator `T`, a local extremum of the Rayleigh quotient of `T` on a sphere
centred at the origin is an eigenvector of `T`. -/
theorem hasEigenvector_of_isLocalExtrOn (hT : IsSelfAdjoint T) {x₀ : E} (hx₀ : x₀ ≠ 0)
    (hextr : IsLocalExtrOn T.reApplyInnerSelf (sphere (0 : E) ‖x₀‖) x₀) :
    HasEigenvector (T : E →ₗ[𝕜] E) (↑(T.rayleighQuotient x₀)) x₀ := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    ⊢ Module.End.HasEigenvector (↑T) (↑(T.rayleighQuotient x₀)) x₀
  -/
  refine ⟨?_, hx₀⟩
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    ⊢ Membership.mem ((Module.End.genEigenspace ↑T ↑(T.rayleighQuotient x₀)) 1) x₀
  -/
  rw [Module.End.mem_eigenspace_iff]
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsLocalExtrOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    ⊢ Eq (↑T x₀) (HSMul.hSMul (↑(T.rayleighQuotient x₀)) x₀)
  -/
  exact hT.eq_smul_self_of_isLocalExtrOn hextr
  /-
    🎉 no goals
  -/


/-- For a self-adjoint operator `T`, a maximum of the Rayleigh quotient of `T` on a sphere centred
at the origin is an eigenvector of `T`, with eigenvalue the global supremum of the Rayleigh
quotient. -/
theorem hasEigenvector_of_isMaxOn (hT : IsSelfAdjoint T) {x₀ : E} (hx₀ : x₀ ≠ 0)
    (hextr : IsMaxOn T.reApplyInnerSelf (sphere (0 : E) ‖x₀‖) x₀) :
    HasEigenvector (T : E →ₗ[𝕜] E) (↑(⨆ x : { x : E // x ≠ 0 }, T.rayleighQuotient x)) x₀ := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMaxOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    ⊢ Module.End.HasEigenvector (↑T) (↑(iSup fun x => T.rayleighQuotient ↑x)) x₀
  -/
  convert hT.hasEigenvector_of_isLocalExtrOn hx₀ (Or.inr hextr.localize)
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMaxOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    ⊢ Eq (iSup fun x => T.rayleighQuotient ↑x) (T.rayleighQuotient x₀)
  -/
  have hx₀' : 0 < ‖x₀‖ := by simp [hx₀]
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMaxOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    ⊢ Eq (iSup fun x => T.rayleighQuotient ↑x) (T.rayleighQuotient x₀)
  -/
  have hx₀'' : x₀ ∈ sphere (0 : E) ‖x₀‖ := by simp
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMaxOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    ⊢ Eq (iSup fun x => T.rayleighQuotient ↑x) (T.rayleighQuotient x₀)
  -/
  rw [T.iSup_rayleigh_eq_iSup_rayleigh_sphere hx₀']
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMaxOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    ⊢ Eq (iSup fun x => T.rayleighQuotient ↑x) (T.rayleighQuotient x₀)
  -/
  refine IsMaxOn.iSup_eq hx₀'' ?_
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMaxOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    ⊢ IsMaxOn T.rayleighQuotient (Metric.sphere 0 (Norm.norm x₀)) x₀
  -/
  intro x hx
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMaxOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    x : E
    hx : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x
    ⊢ Membership.mem (setOf fun x => (fun x => LE.le (T.rayleighQuotient x) (T.ray …
  -/
  dsimp
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMaxOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    x : E
    hx : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x
    ⊢ LE.le (T.rayleighQuotient x) (T.rayleighQuotient x₀)
  -/
  have : ‖x‖ = ‖x₀‖ := by simpa using hx
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMaxOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    x : E
    hx : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x
    this : Eq (Norm.norm x) (Norm.norm x₀)
    ⊢ LE.le (T.rayleighQuotient x) (T.rayleighQuotient x₀)
  -/
  simp only [ContinuousLinearMap.rayleighQuotient]
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMaxOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    x : E
    hx : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x
    this : Eq (Norm.norm x) (Norm.norm x₀)
    ⊢ LE.le (HDiv.hDiv (T.reApplyInnerSelf x) (HPow.hPow (Norm.norm x) 2)) (HDiv.h …
  -/
  rw [this]
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMaxOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    x : E
    hx : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x
    this : Eq (Norm.norm x) (Norm.norm x₀)
    ⊢ LE.le (HDiv.hDiv (T.reApplyInnerSelf x) (HPow.hPow (Norm.norm x₀) 2)) (HDiv. …
  -/
  gcongr
  /-
    case h.e'_7.h.e'_3.hab
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMaxOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    x : E
    hx : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x
    this : Eq (Norm.norm x) (Norm.norm x₀)
    ⊢ LE.le (T.reApplyInnerSelf x) (T.reApplyInnerSelf x₀)
  -/
  exact hextr hx
  /-
    🎉 no goals
  -/


/-- For a self-adjoint operator `T`, a minimum of the Rayleigh quotient of `T` on a sphere centred
at the origin is an eigenvector of `T`, with eigenvalue the global infimum of the Rayleigh
quotient. -/
theorem hasEigenvector_of_isMinOn (hT : IsSelfAdjoint T) {x₀ : E} (hx₀ : x₀ ≠ 0)
    (hextr : IsMinOn T.reApplyInnerSelf (sphere (0 : E) ‖x₀‖) x₀) :
    HasEigenvector (T : E →ₗ[𝕜] E) (↑(⨅ x : { x : E // x ≠ 0 }, T.rayleighQuotient x)) x₀ := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMinOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    ⊢ Module.End.HasEigenvector (↑T) (↑(iInf fun x => T.rayleighQuotient ↑x)) x₀
  -/
  convert hT.hasEigenvector_of_isLocalExtrOn hx₀ (Or.inl hextr.localize)
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMinOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    ⊢ Eq (iInf fun x => T.rayleighQuotient ↑x) (T.rayleighQuotient x₀)
  -/
  have hx₀' : 0 < ‖x₀‖ := by simp [hx₀]
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMinOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    ⊢ Eq (iInf fun x => T.rayleighQuotient ↑x) (T.rayleighQuotient x₀)
  -/
  have hx₀'' : x₀ ∈ sphere (0 : E) ‖x₀‖ := by simp
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMinOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    ⊢ Eq (iInf fun x => T.rayleighQuotient ↑x) (T.rayleighQuotient x₀)
  -/
  rw [T.iInf_rayleigh_eq_iInf_rayleigh_sphere hx₀']
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMinOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    ⊢ Eq (iInf fun x => T.rayleighQuotient ↑x) (T.rayleighQuotient x₀)
  -/
  refine IsMinOn.iInf_eq hx₀'' ?_
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMinOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    ⊢ IsMinOn T.rayleighQuotient (Metric.sphere 0 (Norm.norm x₀)) x₀
  -/
  intro x hx
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMinOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    x : E
    hx : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x
    ⊢ Membership.mem (setOf fun x => (fun x => LE.le (T.rayleighQuotient x₀) (T.ra …
  -/
  dsimp
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMinOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    x : E
    hx : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x
    ⊢ LE.le (T.rayleighQuotient x₀) (T.rayleighQuotient x)
  -/
  have : ‖x‖ = ‖x₀‖ := by simpa using hx
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMinOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    x : E
    hx : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x
    this : Eq (Norm.norm x) (Norm.norm x₀)
    ⊢ LE.le (T.rayleighQuotient x₀) (T.rayleighQuotient x)
  -/
  simp only [ContinuousLinearMap.rayleighQuotient]
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMinOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    x : E
    hx : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x
    this : Eq (Norm.norm x) (Norm.norm x₀)
    ⊢ LE.le (HDiv.hDiv (T.reApplyInnerSelf x₀) (HPow.hPow (Norm.norm x₀) 2)) (HDiv …
  -/
  rw [this]
  /-
    case h.e'_7.h.e'_3
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMinOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    x : E
    hx : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x
    this : Eq (Norm.norm x) (Norm.norm x₀)
    ⊢ LE.le (HDiv.hDiv (T.reApplyInnerSelf x₀) (HPow.hPow (Norm.norm x₀) 2)) (HDiv …
  -/
  gcongr
  /-
    case h.e'_7.h.e'_3.hab
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : IsSelfAdjoint T
    x₀ : E
    hx₀ : Ne x₀ 0
    hextr : IsMinOn T.reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀' : LT.lt 0 (Norm.norm x₀)
    hx₀'' : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x₀
    x : E
    hx : Membership.mem (Metric.sphere 0 (Norm.norm x₀)) x
    this : Eq (Norm.norm x) (Norm.norm x₀)
    ⊢ LE.le (T.reApplyInnerSelf x₀) (T.reApplyInnerSelf x)
  -/
  exact hextr hx
  /-
    🎉 no goals
  -/


/-- The supremum of the Rayleigh quotient of a symmetric operator `T` on a nontrivial
finite-dimensional vector space is an eigenvalue for that operator. -/
theorem hasEigenvalue_iSup_of_finiteDimensional [Nontrivial E] (hT : T.IsSymmetric) :
    HasEigenvalue T ↑(⨆ x : { x : E // x ≠ 0 }, RCLike.re ⟪T x, x⟫ / ‖(x : E)‖ ^ 2 : ℝ) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : Nontrivial E
    hT : T.IsSymmetric
    ⊢ Module.End.HasEigenvalue T ↑(iSup fun x => HDiv.hDiv (RCLike.re (Inner.inner …
  -/
  haveI := FiniteDimensional.proper_rclike 𝕜 E
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : Nontrivial E
    hT : T.IsSymmetric
    this : ProperSpace E
    ⊢ Module.End.HasEigenvalue T ↑(iSup fun x => HDiv.hDiv (RCLike.re (Inner.inner …
  -/
  let T' := hT.toSelfAdjoint
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : Nontrivial E
    hT : T.IsSymmetric
    this : ProperSpace E
    T' : Subtype fun x => Membership.mem (selfAdjoint (ContinuousLinearMap (RingHo …
    ⊢ Module.End.HasEigenvalue T ↑(iSup fun x => HDiv.hDiv (RCLike.re (Inner.inner …
  -/
  obtain ⟨x, hx⟩ : ∃ x : E, x ≠ 0 := exists_ne 0
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : Nontrivial E
    hT : T.IsSymmetric
    this : ProperSpace E
    T' : Subtype fun x => Membership.mem (selfAdjoint (ContinuousLinearMap (RingHo …
    x : E
    hx : Ne x 0
    ⊢ Module.End.HasEigenvalue T ↑(iSup fun x => HDiv.hDiv (RCLike.re (Inner.inner …
  -/
  have H₁ : IsCompact (sphere (0 : E) ‖x‖) := isCompact_sphere _ _
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : Nontrivial E
    hT : T.IsSymmetric
    this : ProperSpace E
    T' : Subtype fun x => Membership.mem (selfAdjoint (ContinuousLinearMap (RingHo …
    x : E
    hx : Ne x 0
    H₁ : IsCompact (Metric.sphere 0 (Norm.norm x))
    ⊢ Module.End.HasEigenvalue T ↑(iSup fun x => HDiv.hDiv (RCLike.re (Inner.inner …
  -/
  have H₂ : (sphere (0 : E) ‖x‖).Nonempty := ⟨x, by simp⟩
  -- key point: in finite dimension, a continuous function on the sphere has a max
  obtain ⟨x₀, hx₀', hTx₀⟩ :=
    H₁.exists_isMaxOn H₂ T'.val.reApplyInnerSelf_continuous.continuousOn
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : Nontrivial E
    hT : T.IsSymmetric
    this : ProperSpace E
    T' : Subtype fun x => Membership.mem (selfAdjoint (ContinuousLinearMap (RingHo …
    x : E
    hx : Ne x 0
    H₁ : IsCompact (Metric.sphere 0 (Norm.norm x))
    H₂ : (Metric.sphere 0 (Norm.norm x)).Nonempty
    x₀ : E
    hx₀' : Membership.mem (Metric.sphere 0 (Norm.norm x)) x₀
    hTx₀ : IsMaxOn (↑T').reApplyInnerSelf (Metric.sphere 0 (Norm.norm x)) x₀
    ⊢ Module.End.HasEigenvalue T ↑(iSup fun x => HDiv.hDiv (RCLike.re (Inner.inner …
  -/
  have hx₀ : ‖x₀‖ = ‖x‖ := by simpa using hx₀'
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : Nontrivial E
    hT : T.IsSymmetric
    this : ProperSpace E
    T' : Subtype fun x => Membership.mem (selfAdjoint (ContinuousLinearMap (RingHo …
    x : E
    hx : Ne x 0
    H₁ : IsCompact (Metric.sphere 0 (Norm.norm x))
    H₂ : (Metric.sphere 0 (Norm.norm x)).Nonempty
    x₀ : E
    hx₀' : Membership.mem (Metric.sphere 0 (Norm.norm x)) x₀
    hTx₀ : IsMaxOn (↑T').reApplyInnerSelf (Metric.sphere 0 (Norm.norm x)) x₀
    hx₀ : Eq (Norm.norm x₀) (Norm.norm x)
    ⊢ Module.End.HasEigenvalue T ↑(iSup fun x => HDiv.hDiv (RCLike.re (Inner.inner …
  -/
  have : IsMaxOn T'.val.reApplyInnerSelf (sphere 0 ‖x₀‖) x₀ := by simpa only [← hx₀] using hTx₀
  have hx₀_ne : x₀ ≠ 0 := by
    have : ‖x₀‖ ≠ 0 := by simp only [hx₀, norm_eq_zero, hx, Ne, not_false_iff]
    simpa [← norm_eq_zero, Ne]
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : Nontrivial E
    hT : T.IsSymmetric
    this✝ : ProperSpace E
    T' : Subtype fun x => Membership.mem (selfAdjoint (ContinuousLinearMap (RingHo …
    x : E
    hx : Ne x 0
    H₁ : IsCompact (Metric.sphere 0 (Norm.norm x))
    H₂ : (Metric.sphere 0 (Norm.norm x)).Nonempty
    x₀ : E
    hx₀' : Membership.mem (Metric.sphere 0 (Norm.norm x)) x₀
    hTx₀ : IsMaxOn (↑T').reApplyInnerSelf (Metric.sphere 0 (Norm.norm x)) x₀
    hx₀ : Eq (Norm.norm x₀) (Norm.norm x)
    this : IsMaxOn (↑T').reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀_ne : Ne x₀ 0
    ⊢ Module.End.HasEigenvalue T ↑(iSup fun x => HDiv.hDiv (RCLike.re (Inner.inner …
  -/
  exact hasEigenvalue_of_hasEigenvector (T'.prop.hasEigenvector_of_isMaxOn hx₀_ne this)
  /-
    🎉 no goals
  -/


/-- The infimum of the Rayleigh quotient of a symmetric operator `T` on a nontrivial
finite-dimensional vector space is an eigenvalue for that operator. -/
theorem hasEigenvalue_iInf_of_finiteDimensional [Nontrivial E] (hT : T.IsSymmetric) :
    HasEigenvalue T ↑(⨅ x : { x : E // x ≠ 0 }, RCLike.re ⟪T x, x⟫ / ‖(x : E)‖ ^ 2 : ℝ) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : Nontrivial E
    hT : T.IsSymmetric
    ⊢ Module.End.HasEigenvalue T ↑(iInf fun x => HDiv.hDiv (RCLike.re (Inner.inner …
  -/
  haveI := FiniteDimensional.proper_rclike 𝕜 E
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : Nontrivial E
    hT : T.IsSymmetric
    this : ProperSpace E
    ⊢ Module.End.HasEigenvalue T ↑(iInf fun x => HDiv.hDiv (RCLike.re (Inner.inner …
  -/
  let T' := hT.toSelfAdjoint
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : Nontrivial E
    hT : T.IsSymmetric
    this : ProperSpace E
    T' : Subtype fun x => Membership.mem (selfAdjoint (ContinuousLinearMap (RingHo …
    ⊢ Module.End.HasEigenvalue T ↑(iInf fun x => HDiv.hDiv (RCLike.re (Inner.inner …
  -/
  obtain ⟨x, hx⟩ : ∃ x : E, x ≠ 0 := exists_ne 0
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : Nontrivial E
    hT : T.IsSymmetric
    this : ProperSpace E
    T' : Subtype fun x => Membership.mem (selfAdjoint (ContinuousLinearMap (RingHo …
    x : E
    hx : Ne x 0
    ⊢ Module.End.HasEigenvalue T ↑(iInf fun x => HDiv.hDiv (RCLike.re (Inner.inner …
  -/
  have H₁ : IsCompact (sphere (0 : E) ‖x‖) := isCompact_sphere _ _
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : Nontrivial E
    hT : T.IsSymmetric
    this : ProperSpace E
    T' : Subtype fun x => Membership.mem (selfAdjoint (ContinuousLinearMap (RingHo …
    x : E
    hx : Ne x 0
    H₁ : IsCompact (Metric.sphere 0 (Norm.norm x))
    ⊢ Module.End.HasEigenvalue T ↑(iInf fun x => HDiv.hDiv (RCLike.re (Inner.inner …
  -/
  have H₂ : (sphere (0 : E) ‖x‖).Nonempty := ⟨x, by simp⟩
  -- key point: in finite dimension, a continuous function on the sphere has a min
  obtain ⟨x₀, hx₀', hTx₀⟩ :=
    H₁.exists_isMinOn H₂ T'.val.reApplyInnerSelf_continuous.continuousOn
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : Nontrivial E
    hT : T.IsSymmetric
    this : ProperSpace E
    T' : Subtype fun x => Membership.mem (selfAdjoint (ContinuousLinearMap (RingHo …
    x : E
    hx : Ne x 0
    H₁ : IsCompact (Metric.sphere 0 (Norm.norm x))
    H₂ : (Metric.sphere 0 (Norm.norm x)).Nonempty
    x₀ : E
    hx₀' : Membership.mem (Metric.sphere 0 (Norm.norm x)) x₀
    hTx₀ : IsMinOn (↑T').reApplyInnerSelf (Metric.sphere 0 (Norm.norm x)) x₀
    ⊢ Module.End.HasEigenvalue T ↑(iInf fun x => HDiv.hDiv (RCLike.re (Inner.inner …
  -/
  have hx₀ : ‖x₀‖ = ‖x‖ := by simpa using hx₀'
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : Nontrivial E
    hT : T.IsSymmetric
    this : ProperSpace E
    T' : Subtype fun x => Membership.mem (selfAdjoint (ContinuousLinearMap (RingHo …
    x : E
    hx : Ne x 0
    H₁ : IsCompact (Metric.sphere 0 (Norm.norm x))
    H₂ : (Metric.sphere 0 (Norm.norm x)).Nonempty
    x₀ : E
    hx₀' : Membership.mem (Metric.sphere 0 (Norm.norm x)) x₀
    hTx₀ : IsMinOn (↑T').reApplyInnerSelf (Metric.sphere 0 (Norm.norm x)) x₀
    hx₀ : Eq (Norm.norm x₀) (Norm.norm x)
    ⊢ Module.End.HasEigenvalue T ↑(iInf fun x => HDiv.hDiv (RCLike.re (Inner.inner …
  -/
  have : IsMinOn T'.val.reApplyInnerSelf (sphere 0 ‖x₀‖) x₀ := by simpa only [← hx₀] using hTx₀
  have hx₀_ne : x₀ ≠ 0 := by
    have : ‖x₀‖ ≠ 0 := by simp only [hx₀, norm_eq_zero, hx, Ne, not_false_iff]
    simpa [← norm_eq_zero, Ne]
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : FiniteDimensional 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : Nontrivial E
    hT : T.IsSymmetric
    this✝ : ProperSpace E
    T' : Subtype fun x => Membership.mem (selfAdjoint (ContinuousLinearMap (RingHo …
    x : E
    hx : Ne x 0
    H₁ : IsCompact (Metric.sphere 0 (Norm.norm x))
    H₂ : (Metric.sphere 0 (Norm.norm x)).Nonempty
    x₀ : E
    hx₀' : Membership.mem (Metric.sphere 0 (Norm.norm x)) x₀
    hTx₀ : IsMinOn (↑T').reApplyInnerSelf (Metric.sphere 0 (Norm.norm x)) x₀
    hx₀ : Eq (Norm.norm x₀) (Norm.norm x)
    this : IsMinOn (↑T').reApplyInnerSelf (Metric.sphere 0 (Norm.norm x₀)) x₀
    hx₀_ne : Ne x₀ 0
    ⊢ Module.End.HasEigenvalue T ↑(iInf fun x => HDiv.hDiv (RCLike.re (Inner.inner …
  -/
  exact hasEigenvalue_of_hasEigenvector (T'.prop.hasEigenvector_of_isMinOn hx₀_ne this)
  /-
    🎉 no goals
  -/


theorem subsingleton_of_no_eigenvalue_finiteDimensional (hT : T.IsSymmetric)
    (hT' : ∀ μ : 𝕜, Module.End.eigenspace (T : E →ₗ[𝕜] E) μ = ⊥) : Subsingleton E :=
  (subsingleton_or_nontrivial E).resolve_right fun _h =>
    absurd (hT' _) hT.hasEigenvalue_iSup_of_finiteDimensional


