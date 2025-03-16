theorem HasStrictDerivAt.hasStrictFDerivAt_equiv {f : 𝕜 → 𝕜} {f' x : 𝕜}
    (hf : HasStrictDerivAt f f' x) (hf' : f' ≠ 0) :
    HasStrictFDerivAt f (ContinuousLinearEquiv.unitsEquivAut 𝕜 (Units.mk0 f' hf') : 𝕜 →L[𝕜] 𝕜) x :=
  hf


theorem HasDerivAt.hasFDerivAt_equiv {f : 𝕜 → 𝕜} {f' x : 𝕜} (hf : HasDerivAt f f' x)
    (hf' : f' ≠ 0) :
    HasFDerivAt f (ContinuousLinearEquiv.unitsEquivAut 𝕜 (Units.mk0 f' hf') : 𝕜 →L[𝕜] 𝕜) x :=
  hf


/-- If `f (g y) = y` for `y` in some neighborhood of `a`, `g` is continuous at `a`, and `f` has an
invertible derivative `f'` at `g a` in the strict sense, then `g` has the derivative `f'⁻¹` at `a`
in the strict sense.

This is one of the easy parts of the inverse function theorem: it assumes that we already have an
inverse function. -/
theorem HasStrictDerivAt.of_local_left_inverse {f g : 𝕜 → 𝕜} {f' a : 𝕜} (hg : ContinuousAt g a)
    (hf : HasStrictDerivAt f f' (g a)) (hf' : f' ≠ 0) (hfg : ∀ᶠ y in 𝓝 a, f (g y) = y) :
    HasStrictDerivAt g f'⁻¹ a :=
  (hf.hasStrictFDerivAt_equiv hf').of_local_left_inverse hg hfg


/-- If `f` is a partial homeomorphism defined on a neighbourhood of `f.symm a`, and `f` has a
nonzero derivative `f'` at `f.symm a` in the strict sense, then `f.symm` has the derivative `f'⁻¹`
at `a` in the strict sense.

This is one of the easy parts of the inverse function theorem: it assumes that we already have
an inverse function. -/
theorem PartialHomeomorph.hasStrictDerivAt_symm (f : PartialHomeomorph 𝕜 𝕜) {a f' : 𝕜}
    (ha : a ∈ f.target) (hf' : f' ≠ 0) (htff' : HasStrictDerivAt f f' (f.symm a)) :
    HasStrictDerivAt f.symm f'⁻¹ a :=
  htff'.of_local_left_inverse (f.symm.continuousAt ha) hf' (f.eventually_right_inverse ha)


/-- If `f (g y) = y` for `y` in some neighborhood of `a`, `g` is continuous at `a`, and `f` has an
invertible derivative `f'` at `g a`, then `g` has the derivative `f'⁻¹` at `a`.

This is one of the easy parts of the inverse function theorem: it assumes that we already have
an inverse function. -/
theorem HasDerivAt.of_local_left_inverse {f g : 𝕜 → 𝕜} {f' a : 𝕜} (hg : ContinuousAt g a)
    (hf : HasDerivAt f f' (g a)) (hf' : f' ≠ 0) (hfg : ∀ᶠ y in 𝓝 a, f (g y) = y) :
    HasDerivAt g f'⁻¹ a :=
  (hf.hasFDerivAt_equiv hf').of_local_left_inverse hg hfg


/-- If `f` is a partial homeomorphism defined on a neighbourhood of `f.symm a`, and `f` has a
nonzero derivative `f'` at `f.symm a`, then `f.symm` has the derivative `f'⁻¹` at `a`.

This is one of the easy parts of the inverse function theorem: it assumes that we already have
an inverse function. -/
theorem PartialHomeomorph.hasDerivAt_symm (f : PartialHomeomorph 𝕜 𝕜) {a f' : 𝕜} (ha : a ∈ f.target)
    (hf' : f' ≠ 0) (htff' : HasDerivAt f f' (f.symm a)) : HasDerivAt f.symm f'⁻¹ a :=
  htff'.of_local_left_inverse (f.symm.continuousAt ha) hf' (f.eventually_right_inverse ha)


theorem HasDerivAt.eventually_ne (h : HasDerivAt f f' x) (hf' : f' ≠ 0) :
    ∀ᶠ z in 𝓝[≠] x, f z ≠ f x :=
  (hasDerivAt_iff_hasFDerivAt.1 h).eventually_ne
                         /-
                           𝕜 : Type u
                           inst✝² : NontriviallyNormedField 𝕜
                           F : Type v
                           inst✝¹ : NormedAddCommGroup F
                           inst✝ : NormedSpace 𝕜 F
                           f : 𝕜 → F
                           f' : F
                           x : 𝕜
                           h : HasDerivAt f f' x
                           hf' : Ne f' 0
                           z : 𝕜
                           ⊢ LE.le (Norm.norm z) (HMul.hMul (Inv.inv (Norm.norm f')) (Norm.norm ((Continu …
                         -/
    ⟨‖f'‖⁻¹, fun z => by field_simp [norm_smul, mt norm_eq_zero.1 hf']⟩
                         /-
                           🎉 no goals
                         -/


theorem HasDerivAt.tendsto_punctured_nhds (h : HasDerivAt f f' x) (hf' : f' ≠ 0) :
    Tendsto f (𝓝[≠] x) (𝓝[≠] f x) :=
  tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ h.continuousAt.continuousWithinAt
    (h.eventually_ne hf')


theorem not_differentiableWithinAt_of_local_left_inverse_hasDerivWithinAt_zero {f g : 𝕜 → 𝕜} {a : 𝕜}
    {s t : Set 𝕜} (ha : a ∈ s) (hsu : UniqueDiffWithinAt 𝕜 s a) (hf : HasDerivWithinAt f 0 t (g a))
    (hst : MapsTo g s t) (hfg : f ∘ g =ᶠ[𝓝[s] a] id) : ¬DifferentiableWithinAt 𝕜 g s a := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    f g : 𝕜 → 𝕜
    a : 𝕜
    s t : Set 𝕜
    ha : Membership.mem s a
    hsu : UniqueDiffWithinAt 𝕜 s a
    hf : HasDerivWithinAt f 0 t (g a)
    hst : Set.MapsTo g s t
    hfg : (nhdsWithin a s).EventuallyEq (Function.comp f g) id
    ⊢ Not (DifferentiableWithinAt 𝕜 g s a)
  -/
  intro hg
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    f g : 𝕜 → 𝕜
    a : 𝕜
    s t : Set 𝕜
    ha : Membership.mem s a
    hsu : UniqueDiffWithinAt 𝕜 s a
    hf : HasDerivWithinAt f 0 t (g a)
    hst : Set.MapsTo g s t
    hfg : (nhdsWithin a s).EventuallyEq (Function.comp f g) id
    hg : DifferentiableWithinAt 𝕜 g s a
    ⊢ False
  -/
  have := (hf.comp a hg.hasDerivWithinAt hst).congr_of_eventuallyEq_of_mem hfg.symm ha
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    f g : 𝕜 → 𝕜
    a : 𝕜
    s t : Set 𝕜
    ha : Membership.mem s a
    hsu : UniqueDiffWithinAt 𝕜 s a
    hf : HasDerivWithinAt f 0 t (g a)
    hst : Set.MapsTo g s t
    hfg : (nhdsWithin a s).EventuallyEq (Function.comp f g) id
    hg : DifferentiableWithinAt 𝕜 g s a
    this : HasDerivWithinAt id (HMul.hMul 0 (derivWithin g s a)) s a
    ⊢ False
  -/
  simpa using hsu.eq_deriv _ this (hasDerivWithinAt_id _ _)
  /-
    🎉 no goals
  -/


theorem not_differentiableAt_of_local_left_inverse_hasDerivAt_zero {f g : 𝕜 → 𝕜} {a : 𝕜}
    (hf : HasDerivAt f 0 (g a)) (hfg : f ∘ g =ᶠ[𝓝 a] id) : ¬DifferentiableAt 𝕜 g a := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    f g : 𝕜 → 𝕜
    a : 𝕜
    hf : HasDerivAt f 0 (g a)
    hfg : (nhds a).EventuallyEq (Function.comp f g) id
    ⊢ Not (DifferentiableAt 𝕜 g a)
  -/
  intro hg
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    f g : 𝕜 → 𝕜
    a : 𝕜
    hf : HasDerivAt f 0 (g a)
    hfg : (nhds a).EventuallyEq (Function.comp f g) id
    hg : DifferentiableAt 𝕜 g a
    ⊢ False
  -/
  have := (hf.comp a hg.hasDerivAt).congr_of_eventuallyEq hfg.symm
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    f g : 𝕜 → 𝕜
    a : 𝕜
    hf : HasDerivAt f 0 (g a)
    hfg : (nhds a).EventuallyEq (Function.comp f g) id
    hg : DifferentiableAt 𝕜 g a
    this : HasDerivAt id (HMul.hMul 0 (deriv g a)) a
    ⊢ False
  -/
  simpa using this.unique (hasDerivAt_id a)
  /-
    🎉 no goals
  -/

