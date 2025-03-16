@[fun_prop]
theorem HasStrictFDerivAt.restrictScalars (h : HasStrictFDerivAt f f' x) :
    HasStrictFDerivAt f (f'.restrictScalars 𝕜) x :=
  .of_isLittleO h.isLittleO


theorem HasFDerivAtFilter.restrictScalars {L} (h : HasFDerivAtFilter f f' x L) :
    HasFDerivAtFilter f (f'.restrictScalars 𝕜) x L :=
  .of_isLittleO h.isLittleO


@[fun_prop]
theorem HasFDerivAt.restrictScalars (h : HasFDerivAt f f' x) :
    HasFDerivAt f (f'.restrictScalars 𝕜) x :=
  .of_isLittleO h.isLittleO


@[fun_prop]
theorem HasFDerivWithinAt.restrictScalars (h : HasFDerivWithinAt f f' s x) :
    HasFDerivWithinAt f (f'.restrictScalars 𝕜) s x :=
  .of_isLittleO h.isLittleO


@[fun_prop]
theorem DifferentiableAt.restrictScalars (h : DifferentiableAt 𝕜' f x) : DifferentiableAt 𝕜 f x :=
  (h.hasFDerivAt.restrictScalars 𝕜).differentiableAt


@[fun_prop]
theorem DifferentiableWithinAt.restrictScalars (h : DifferentiableWithinAt 𝕜' f s x) :
    DifferentiableWithinAt 𝕜 f s x :=
  (h.hasFDerivWithinAt.restrictScalars 𝕜).differentiableWithinAt


@[fun_prop]
theorem DifferentiableOn.restrictScalars (h : DifferentiableOn 𝕜' f s) : DifferentiableOn 𝕜 f s :=
  fun x hx => (h x hx).restrictScalars 𝕜


@[fun_prop]
theorem Differentiable.restrictScalars (h : Differentiable 𝕜' f) : Differentiable 𝕜 f := fun x =>
  (h x).restrictScalars 𝕜


@[fun_prop]
theorem HasFDerivWithinAt.of_restrictScalars {g' : E →L[𝕜] F} (h : HasFDerivWithinAt f g' s x)
    (H : f'.restrictScalars 𝕜 = g') : HasFDerivWithinAt f f' s x := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    𝕜' : Type u_2
    inst✝⁹ : NontriviallyNormedField 𝕜'
    inst✝⁸ : NormedAlgebra 𝕜 𝕜'
    E : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜' E
    inst✝⁴ : IsScalarTower 𝕜 𝕜' E
    F : Type u_4
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜') E F
    s : Set E
    x : E
    g' : ContinuousLinearMap (RingHom.id 𝕜) E F
    h : HasFDerivWithinAt f g' s x
    H : Eq (ContinuousLinearMap.restrictScalars 𝕜 f') g'
    ⊢ HasFDerivWithinAt f f' s x
  -/
  rw [← H] at h
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    𝕜' : Type u_2
    inst✝⁹ : NontriviallyNormedField 𝕜'
    inst✝⁸ : NormedAlgebra 𝕜 𝕜'
    E : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜' E
    inst✝⁴ : IsScalarTower 𝕜 𝕜' E
    F : Type u_4
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜') E F
    s : Set E
    x : E
    g' : ContinuousLinearMap (RingHom.id 𝕜) E F
    h : HasFDerivWithinAt f (ContinuousLinearMap.restrictScalars 𝕜 f') s x
    H : Eq (ContinuousLinearMap.restrictScalars 𝕜 f') g'
    ⊢ HasFDerivWithinAt f f' s x
  -/
  exact .of_isLittleO h.isLittleO
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem hasFDerivAt_of_restrictScalars {g' : E →L[𝕜] F} (h : HasFDerivAt f g' x)
    (H : f'.restrictScalars 𝕜 = g') : HasFDerivAt f f' x := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    𝕜' : Type u_2
    inst✝⁹ : NontriviallyNormedField 𝕜'
    inst✝⁸ : NormedAlgebra 𝕜 𝕜'
    E : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜' E
    inst✝⁴ : IsScalarTower 𝕜 𝕜' E
    F : Type u_4
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜') E F
    x : E
    g' : ContinuousLinearMap (RingHom.id 𝕜) E F
    h : HasFDerivAt f g' x
    H : Eq (ContinuousLinearMap.restrictScalars 𝕜 f') g'
    ⊢ HasFDerivAt f f' x
  -/
  rw [← H] at h
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    𝕜' : Type u_2
    inst✝⁹ : NontriviallyNormedField 𝕜'
    inst✝⁸ : NormedAlgebra 𝕜 𝕜'
    E : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜' E
    inst✝⁴ : IsScalarTower 𝕜 𝕜' E
    F : Type u_4
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜') E F
    x : E
    g' : ContinuousLinearMap (RingHom.id 𝕜) E F
    h : HasFDerivAt f (ContinuousLinearMap.restrictScalars 𝕜 f') x
    H : Eq (ContinuousLinearMap.restrictScalars 𝕜 f') g'
    ⊢ HasFDerivAt f f' x
  -/
  exact .of_isLittleO h.isLittleO
  /-
    🎉 no goals
  -/


theorem DifferentiableAt.fderiv_restrictScalars (h : DifferentiableAt 𝕜' f x) :
    fderiv 𝕜 f x = (fderiv 𝕜' f x).restrictScalars 𝕜 :=
  (h.hasFDerivAt.restrictScalars 𝕜).fderiv


theorem differentiableWithinAt_iff_restrictScalars (hf : DifferentiableWithinAt 𝕜 f s x)
    (hs : UniqueDiffWithinAt 𝕜 s x) : DifferentiableWithinAt 𝕜' f s x ↔
      ∃ g' : E →L[𝕜'] F, g'.restrictScalars 𝕜 = fderivWithin 𝕜 f s x := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    𝕜' : Type u_2
    inst✝⁹ : NontriviallyNormedField 𝕜'
    inst✝⁸ : NormedAlgebra 𝕜 𝕜'
    E : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜' E
    inst✝⁴ : IsScalarTower 𝕜 𝕜' E
    F : Type u_4
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    f : E → F
    s : Set E
    x : E
    hf : DifferentiableWithinAt 𝕜 f s x
    hs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Iff (DifferentiableWithinAt 𝕜' f s x) (Exists fun g' => Eq (ContinuousLinear …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜'
      inst✝⁸ : NormedAlgebra 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedSpace 𝕜' E
      inst✝⁴ : IsScalarTower 𝕜 𝕜' E
      F : Type u_4
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedSpace 𝕜' F
      inst✝ : IsScalarTower 𝕜 𝕜' F
      f : E → F
      s : Set E
      x : E
      hf : DifferentiableWithinAt 𝕜 f s x
      hs : UniqueDiffWithinAt 𝕜 s x
      ⊢ DifferentiableWithinAt 𝕜' f s x → Exists fun g' => Eq (ContinuousLinearMap.r …
    -/
  · rintro ⟨g', hg'⟩
    /-
      case mp.intro
      𝕜 : Type u_1
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜'
      inst✝⁸ : NormedAlgebra 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedSpace 𝕜' E
      inst✝⁴ : IsScalarTower 𝕜 𝕜' E
      F : Type u_4
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedSpace 𝕜' F
      inst✝ : IsScalarTower 𝕜 𝕜' F
      f : E → F
      s : Set E
      x : E
      hf : DifferentiableWithinAt 𝕜 f s x
      hs : UniqueDiffWithinAt 𝕜 s x
      g' : ContinuousLinearMap (RingHom.id 𝕜') E F
      hg' : HasFDerivWithinAt f g' s x
      ⊢ Exists fun g' => Eq (ContinuousLinearMap.restrictScalars 𝕜 g') (fderivWithin …
    -/
    exact ⟨g', hs.eq (hg'.restrictScalars 𝕜) hf.hasFDerivWithinAt⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜'
      inst✝⁸ : NormedAlgebra 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedSpace 𝕜' E
      inst✝⁴ : IsScalarTower 𝕜 𝕜' E
      F : Type u_4
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedSpace 𝕜' F
      inst✝ : IsScalarTower 𝕜 𝕜' F
      f : E → F
      s : Set E
      x : E
      hf : DifferentiableWithinAt 𝕜 f s x
      hs : UniqueDiffWithinAt 𝕜 s x
      ⊢ (Exists fun g' => Eq (ContinuousLinearMap.restrictScalars 𝕜 g') (fderivWithi …
    -/
  · rintro ⟨f', hf'⟩
    /-
      case mpr.intro
      𝕜 : Type u_1
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜'
      inst✝⁸ : NormedAlgebra 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedSpace 𝕜' E
      inst✝⁴ : IsScalarTower 𝕜 𝕜' E
      F : Type u_4
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedSpace 𝕜' F
      inst✝ : IsScalarTower 𝕜 𝕜' F
      f : E → F
      s : Set E
      x : E
      hf : DifferentiableWithinAt 𝕜 f s x
      hs : UniqueDiffWithinAt 𝕜 s x
      f' : ContinuousLinearMap (RingHom.id 𝕜') E F
      hf' : Eq (ContinuousLinearMap.restrictScalars 𝕜 f') (fderivWithin 𝕜 f s x)
      ⊢ DifferentiableWithinAt 𝕜' f s x
    -/
    exact ⟨f', hf.hasFDerivWithinAt.of_restrictScalars 𝕜 hf'⟩
    /-
      🎉 no goals
    -/


theorem differentiableAt_iff_restrictScalars (hf : DifferentiableAt 𝕜 f x) :
    DifferentiableAt 𝕜' f x ↔ ∃ g' : E →L[𝕜'] F, g'.restrictScalars 𝕜 = fderiv 𝕜 f x := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    𝕜' : Type u_2
    inst✝⁹ : NontriviallyNormedField 𝕜'
    inst✝⁸ : NormedAlgebra 𝕜 𝕜'
    E : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜' E
    inst✝⁴ : IsScalarTower 𝕜 𝕜' E
    F : Type u_4
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    f : E → F
    x : E
    hf : DifferentiableAt 𝕜 f x
    ⊢ Iff (DifferentiableAt 𝕜' f x) (Exists fun g' => Eq (ContinuousLinearMap.rest …
  -/
  rw [← differentiableWithinAt_univ, ← fderivWithin_univ]
  exact
    differentiableWithinAt_iff_restrictScalars 𝕜 hf.differentiableWithinAt uniqueDiffWithinAt_univ


