theorem HasDerivAtFilter.scomp (hg : HasDerivAtFilter g₁ g₁' (h x) L')
    (hh : HasDerivAtFilter h h' x L) (hL : Tendsto h L L') :
    HasDerivAtFilter (g₁ ∘ h) (h' • g₁') x L := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : 𝕜
    L : Filter 𝕜
    𝕜' : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    h : 𝕜 → 𝕜'
    h' : 𝕜'
    g₁ : 𝕜' → F
    g₁' : F
    L' : Filter 𝕜'
    hg : HasDerivAtFilter g₁ g₁' (h x) L'
    hh : HasDerivAtFilter h h' x L
    hL : Filter.Tendsto h L L'
    ⊢ HasDerivAtFilter (Function.comp g₁ h) (HSMul.hSMul h' g₁') x L
  -/
  simpa using ((hg.restrictScalars 𝕜).comp x hh hL).hasDerivAtFilter
  /-
    🎉 no goals
  -/


theorem HasDerivAtFilter.scomp_of_eq (hg : HasDerivAtFilter g₁ g₁' y L')
    (hh : HasDerivAtFilter h h' x L) (hy : y = h x) (hL : Tendsto h L L') :
    HasDerivAtFilter (g₁ ∘ h) (h' • g₁') x L := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : 𝕜
    L : Filter 𝕜
    𝕜' : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    h : 𝕜 → 𝕜'
    h' : 𝕜'
    g₁ : 𝕜' → F
    g₁' : F
    L' : Filter 𝕜'
    y : 𝕜'
    hg : HasDerivAtFilter g₁ g₁' y L'
    hh : HasDerivAtFilter h h' x L
    hy : Eq y (h x)
    hL : Filter.Tendsto h L L'
    ⊢ HasDerivAtFilter (Function.comp g₁ h) (HSMul.hSMul h' g₁') x L
  -/
  rw [hy] at hg; exact hg.scomp x hh hL
                 /-
                   🎉 no goals
                 -/


theorem HasDerivWithinAt.scomp_hasDerivAt (hg : HasDerivWithinAt g₁ g₁' s' (h x))
    (hh : HasDerivAt h h' x) (hs : ∀ x, h x ∈ s') : HasDerivAt (g₁ ∘ h) (h' • g₁') x :=
  hg.scomp x hh <| tendsto_inf.2 ⟨hh.continuousAt, tendsto_principal.2 <| Eventually.of_forall hs⟩


theorem HasDerivWithinAt.scomp_hasDerivAt_of_eq (hg : HasDerivWithinAt g₁ g₁' s' y)
    (hh : HasDerivAt h h' x) (hs : ∀ x, h x ∈ s') (hy : y = h x) :
    HasDerivAt (g₁ ∘ h) (h' • g₁') x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : 𝕜
    𝕜' : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    s' : Set 𝕜'
    h : 𝕜 → 𝕜'
    h' : 𝕜'
    g₁ : 𝕜' → F
    g₁' : F
    y : 𝕜'
    hg : HasDerivWithinAt g₁ g₁' s' y
    hh : HasDerivAt h h' x
    hs : ∀ (x : 𝕜), Membership.mem s' (h x)
    hy : Eq y (h x)
    ⊢ HasDerivAt (Function.comp g₁ h) (HSMul.hSMul h' g₁') x
  -/
  rw [hy] at hg; exact hg.scomp_hasDerivAt x hh hs
                 /-
                   🎉 no goals
                 -/


nonrec theorem HasDerivWithinAt.scomp (hg : HasDerivWithinAt g₁ g₁' t' (h x))
    (hh : HasDerivWithinAt h h' s x) (hst : MapsTo h s t') :
    HasDerivWithinAt (g₁ ∘ h) (h' • g₁') s x :=
  hg.scomp x hh <| hh.continuousWithinAt.tendsto_nhdsWithin hst


theorem HasDerivWithinAt.scomp_of_eq (hg : HasDerivWithinAt g₁ g₁' t' y)
    (hh : HasDerivWithinAt h h' s x) (hst : MapsTo h s t') (hy : y = h x) :
    HasDerivWithinAt (g₁ ∘ h) (h' • g₁') s x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : 𝕜
    s : Set 𝕜
    𝕜' : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    t' : Set 𝕜'
    h : 𝕜 → 𝕜'
    h' : 𝕜'
    g₁ : 𝕜' → F
    g₁' : F
    y : 𝕜'
    hg : HasDerivWithinAt g₁ g₁' t' y
    hh : HasDerivWithinAt h h' s x
    hst : Set.MapsTo h s t'
    hy : Eq y (h x)
    ⊢ HasDerivWithinAt (Function.comp g₁ h) (HSMul.hSMul h' g₁') s x
  -/
  rw [hy] at hg; exact hg.scomp x hh hst
                 /-
                   🎉 no goals
                 -/


/-- The chain rule. -/
nonrec theorem HasDerivAt.scomp (hg : HasDerivAt g₁ g₁' (h x)) (hh : HasDerivAt h h' x) :
    HasDerivAt (g₁ ∘ h) (h' • g₁') x :=
  hg.scomp x hh hh.continuousAt


/-- The chain rule. -/
theorem HasDerivAt.scomp_of_eq
    (hg : HasDerivAt g₁ g₁' y) (hh : HasDerivAt h h' x) (hy : y = h x) :
    HasDerivAt (g₁ ∘ h) (h' • g₁') x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : 𝕜
    𝕜' : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    h : 𝕜 → 𝕜'
    h' : 𝕜'
    g₁ : 𝕜' → F
    g₁' : F
    y : 𝕜'
    hg : HasDerivAt g₁ g₁' y
    hh : HasDerivAt h h' x
    hy : Eq y (h x)
    ⊢ HasDerivAt (Function.comp g₁ h) (HSMul.hSMul h' g₁') x
  -/
  rw [hy] at hg; exact hg.scomp x hh
                 /-
                   🎉 no goals
                 -/


theorem HasStrictDerivAt.scomp (hg : HasStrictDerivAt g₁ g₁' (h x)) (hh : HasStrictDerivAt h h' x) :
    HasStrictDerivAt (g₁ ∘ h) (h' • g₁') x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : 𝕜
    𝕜' : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    h : 𝕜 → 𝕜'
    h' : 𝕜'
    g₁ : 𝕜' → F
    g₁' : F
    hg : HasStrictDerivAt g₁ g₁' (h x)
    hh : HasStrictDerivAt h h' x
    ⊢ HasStrictDerivAt (Function.comp g₁ h) (HSMul.hSMul h' g₁') x
  -/
  simpa using ((hg.restrictScalars 𝕜).comp x hh).hasStrictDerivAt
  /-
    🎉 no goals
  -/


theorem HasStrictDerivAt.scomp_of_eq
    (hg : HasStrictDerivAt g₁ g₁' y) (hh : HasStrictDerivAt h h' x) (hy : y = h x) :
    HasStrictDerivAt (g₁ ∘ h) (h' • g₁') x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : 𝕜
    𝕜' : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    h : 𝕜 → 𝕜'
    h' : 𝕜'
    g₁ : 𝕜' → F
    g₁' : F
    y : 𝕜'
    hg : HasStrictDerivAt g₁ g₁' y
    hh : HasStrictDerivAt h h' x
    hy : Eq y (h x)
    ⊢ HasStrictDerivAt (Function.comp g₁ h) (HSMul.hSMul h' g₁') x
  -/
  rw [hy] at hg; exact hg.scomp x hh
                 /-
                   🎉 no goals
                 -/


theorem HasDerivAt.scomp_hasDerivWithinAt (hg : HasDerivAt g₁ g₁' (h x))
    (hh : HasDerivWithinAt h h' s x) : HasDerivWithinAt (g₁ ∘ h) (h' • g₁') s x :=
  HasDerivWithinAt.scomp x hg.hasDerivWithinAt hh (mapsTo_univ _ _)


theorem HasDerivAt.scomp_hasDerivWithinAt_of_eq (hg : HasDerivAt g₁ g₁' y)
    (hh : HasDerivWithinAt h h' s x) (hy : y = h x) :
    HasDerivWithinAt (g₁ ∘ h) (h' • g₁') s x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : 𝕜
    s : Set 𝕜
    𝕜' : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    h : 𝕜 → 𝕜'
    h' : 𝕜'
    g₁ : 𝕜' → F
    g₁' : F
    y : 𝕜'
    hg : HasDerivAt g₁ g₁' y
    hh : HasDerivWithinAt h h' s x
    hy : Eq y (h x)
    ⊢ HasDerivWithinAt (Function.comp g₁ h) (HSMul.hSMul h' g₁') s x
  -/
  rw [hy] at hg; exact hg.scomp_hasDerivWithinAt x hh
                 /-
                   🎉 no goals
                 -/


theorem derivWithin.scomp (hg : DifferentiableWithinAt 𝕜' g₁ t' (h x))
    (hh : DifferentiableWithinAt 𝕜 h s x) (hs : MapsTo h s t') (hxs : UniqueDiffWithinAt 𝕜 s x) :
    derivWithin (g₁ ∘ h) s x = derivWithin h s x • derivWithin g₁ t' (h x) :=
  (HasDerivWithinAt.scomp x hg.hasDerivWithinAt hh.hasDerivWithinAt hs).derivWithin hxs


theorem derivWithin.scomp_of_eq (hg : DifferentiableWithinAt 𝕜' g₁ t' y)
    (hh : DifferentiableWithinAt 𝕜 h s x) (hs : MapsTo h s t') (hxs : UniqueDiffWithinAt 𝕜 s x)
    (hy : y = h x) :
    derivWithin (g₁ ∘ h) s x = derivWithin h s x • derivWithin g₁ t' (h x) := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : 𝕜
    s : Set 𝕜
    𝕜' : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    t' : Set 𝕜'
    h : 𝕜 → 𝕜'
    g₁ : 𝕜' → F
    y : 𝕜'
    hg : DifferentiableWithinAt 𝕜' g₁ t' y
    hh : DifferentiableWithinAt 𝕜 h s x
    hs : Set.MapsTo h s t'
    hxs : UniqueDiffWithinAt 𝕜 s x
    hy : Eq y (h x)
    ⊢ Eq (derivWithin (Function.comp g₁ h) s x) (HSMul.hSMul (derivWithin h s x) ( …
  -/
  rw [hy] at hg; exact derivWithin.scomp x hg hh hs hxs
                 /-
                   🎉 no goals
                 -/


theorem deriv.scomp (hg : DifferentiableAt 𝕜' g₁ (h x)) (hh : DifferentiableAt 𝕜 h x) :
    deriv (g₁ ∘ h) x = deriv h x • deriv g₁ (h x) :=
  (HasDerivAt.scomp x hg.hasDerivAt hh.hasDerivAt).deriv


theorem deriv.scomp_of_eq
    (hg : DifferentiableAt 𝕜' g₁ y) (hh : DifferentiableAt 𝕜 h x) (hy : y = h x) :
    deriv (g₁ ∘ h) x = deriv h x • deriv g₁ (h x) := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : 𝕜
    𝕜' : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    h : 𝕜 → 𝕜'
    g₁ : 𝕜' → F
    y : 𝕜'
    hg : DifferentiableAt 𝕜' g₁ y
    hh : DifferentiableAt 𝕜 h x
    hy : Eq y (h x)
    ⊢ Eq (deriv (Function.comp g₁ h) x) (HSMul.hSMul (deriv h x) (deriv g₁ (h x)))
  -/
  rw [hy] at hg; exact deriv.scomp x hg hh
                 /-
                   🎉 no goals
                 -/


theorem HasDerivAtFilter.comp_hasFDerivAtFilter {f : E → 𝕜'} {f' : E →L[𝕜] 𝕜'} (x) {L'' : Filter E}
    (hh₂ : HasDerivAtFilter h₂ h₂' (f x) L') (hf : HasFDerivAtFilter f f' x L'')
    (hL : Tendsto f L'' L') : HasFDerivAtFilter (h₂ ∘ f) (h₂' • f') x L'' := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h₂ : 𝕜' → 𝕜'
    h₂' : 𝕜'
    L' : Filter 𝕜'
    f : E → 𝕜'
    f' : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    x : E
    L'' : Filter E
    hh₂ : HasDerivAtFilter h₂ h₂' (f x) L'
    hf : HasFDerivAtFilter f f' x L''
    hL : Filter.Tendsto f L'' L'
    ⊢ HasFDerivAtFilter (Function.comp h₂ f) (HSMul.hSMul h₂' f') x L''
  -/
  convert (hh₂.restrictScalars 𝕜).comp x hf hL
  /-
    case h.e'_12.h.e
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h₂ : 𝕜' → 𝕜'
    h₂' : 𝕜'
    L' : Filter 𝕜'
    f : E → 𝕜'
    f' : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    x : E
    L'' : Filter E
    hh₂ : HasDerivAtFilter h₂ h₂' (f x) L'
    hf : HasFDerivAtFilter f f' x L''
    hL : Filter.Tendsto f L'' L'
    ⊢ Eq (HSMul.hSMul h₂') (ContinuousLinearMap.restrictScalars 𝕜 (ContinuousLinea …
  -/
  ext x
  /-
    case h.e'_12.h.e.h.h
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h₂ : 𝕜' → 𝕜'
    h₂' : 𝕜'
    L' : Filter 𝕜'
    f : E → 𝕜'
    f' : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    x✝¹ : E
    L'' : Filter E
    hh₂ : HasDerivAtFilter h₂ h₂' (f x✝¹) L'
    hf : HasFDerivAtFilter f f' x✝¹ L''
    hL : Filter.Tendsto f L'' L'
    x : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    x✝ : E
    ⊢ Eq ((HSMul.hSMul h₂' x) x✝) (((ContinuousLinearMap.restrictScalars 𝕜 (Contin …
  -/
  simp [mul_comm]
  /-
    🎉 no goals
  -/


theorem HasDerivAtFilter.comp_hasFDerivAtFilter_of_eq
    {f : E → 𝕜'} {f' : E →L[𝕜] 𝕜'} (x) {L'' : Filter E}
    (hh₂ : HasDerivAtFilter h₂ h₂' y L') (hf : HasFDerivAtFilter f f' x L'')
    (hL : Tendsto f L'' L') (hy : y = f x) : HasFDerivAtFilter (h₂ ∘ f) (h₂' • f') x L'' := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h₂ : 𝕜' → 𝕜'
    h₂' : 𝕜'
    L' : Filter 𝕜'
    y : 𝕜'
    f : E → 𝕜'
    f' : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    x : E
    L'' : Filter E
    hh₂ : HasDerivAtFilter h₂ h₂' y L'
    hf : HasFDerivAtFilter f f' x L''
    hL : Filter.Tendsto f L'' L'
    hy : Eq y (f x)
    ⊢ HasFDerivAtFilter (Function.comp h₂ f) (HSMul.hSMul h₂' f') x L''
  -/
  rw [hy] at hh₂; exact hh₂.comp_hasFDerivAtFilter x hf hL
                  /-
                    🎉 no goals
                  -/


theorem HasStrictDerivAt.comp_hasStrictFDerivAt {f : E → 𝕜'} {f' : E →L[𝕜] 𝕜'} (x)
    (hh : HasStrictDerivAt h₂ h₂' (f x)) (hf : HasStrictFDerivAt f f' x) :
    HasStrictFDerivAt (h₂ ∘ f) (h₂' • f') x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h₂ : 𝕜' → 𝕜'
    h₂' : 𝕜'
    f : E → 𝕜'
    f' : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    x : E
    hh : HasStrictDerivAt h₂ h₂' (f x)
    hf : HasStrictFDerivAt f f' x
    ⊢ HasStrictFDerivAt (Function.comp h₂ f) (HSMul.hSMul h₂' f') x
  -/
  rw [HasStrictDerivAt] at hh
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h₂ : 𝕜' → 𝕜'
    h₂' : 𝕜'
    f : E → 𝕜'
    f' : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    x : E
    hh : HasStrictFDerivAt h₂ (ContinuousLinearMap.smulRight 1 h₂') (f x)
    hf : HasStrictFDerivAt f f' x
    ⊢ HasStrictFDerivAt (Function.comp h₂ f) (HSMul.hSMul h₂' f') x
  -/
  convert (hh.restrictScalars 𝕜).comp x hf
  /-
    case h.e'_12.h.e
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h₂ : 𝕜' → 𝕜'
    h₂' : 𝕜'
    f : E → 𝕜'
    f' : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    x : E
    hh : HasStrictFDerivAt h₂ (ContinuousLinearMap.smulRight 1 h₂') (f x)
    hf : HasStrictFDerivAt f f' x
    ⊢ Eq (HSMul.hSMul h₂') (ContinuousLinearMap.restrictScalars 𝕜 (ContinuousLinea …
  -/
  ext x
  /-
    case h.e'_12.h.e.h.h
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h₂ : 𝕜' → 𝕜'
    h₂' : 𝕜'
    f : E → 𝕜'
    f' : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    x✝¹ : E
    hh : HasStrictFDerivAt h₂ (ContinuousLinearMap.smulRight 1 h₂') (f x✝¹)
    hf : HasStrictFDerivAt f f' x✝¹
    x : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    x✝ : E
    ⊢ Eq ((HSMul.hSMul h₂' x) x✝) (((ContinuousLinearMap.restrictScalars 𝕜 (Contin …
  -/
  simp [mul_comm]
  /-
    🎉 no goals
  -/


theorem HasStrictDerivAt.comp_hasStrictFDerivAt_of_eq {f : E → 𝕜'} {f' : E →L[𝕜] 𝕜'} (x)
    (hh : HasStrictDerivAt h₂ h₂' y) (hf : HasStrictFDerivAt f f' x) (hy : y = f x) :
    HasStrictFDerivAt (h₂ ∘ f) (h₂' • f') x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h₂ : 𝕜' → 𝕜'
    h₂' y : 𝕜'
    f : E → 𝕜'
    f' : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    x : E
    hh : HasStrictDerivAt h₂ h₂' y
    hf : HasStrictFDerivAt f f' x
    hy : Eq y (f x)
    ⊢ HasStrictFDerivAt (Function.comp h₂ f) (HSMul.hSMul h₂' f') x
  -/
  rw [hy] at hh; exact hh.comp_hasStrictFDerivAt x hf
                 /-
                   🎉 no goals
                 -/


theorem HasDerivAt.comp_hasFDerivAt {f : E → 𝕜'} {f' : E →L[𝕜] 𝕜'} (x)
    (hh : HasDerivAt h₂ h₂' (f x)) (hf : HasFDerivAt f f' x) : HasFDerivAt (h₂ ∘ f) (h₂' • f') x :=
  hh.comp_hasFDerivAtFilter x hf hf.continuousAt


theorem HasDerivAt.comp_hasFDerivAt_of_eq {f : E → 𝕜'} {f' : E →L[𝕜] 𝕜'} (x)
    (hh : HasDerivAt h₂ h₂' y) (hf : HasFDerivAt f f' x) (hy : y = f x) :
    HasFDerivAt (h₂ ∘ f) (h₂' • f') x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h₂ : 𝕜' → 𝕜'
    h₂' y : 𝕜'
    f : E → 𝕜'
    f' : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    x : E
    hh : HasDerivAt h₂ h₂' y
    hf : HasFDerivAt f f' x
    hy : Eq y (f x)
    ⊢ HasFDerivAt (Function.comp h₂ f) (HSMul.hSMul h₂' f') x
  -/
  rw [hy] at hh; exact hh.comp_hasFDerivAt x hf
                 /-
                   🎉 no goals
                 -/


theorem HasDerivAt.comp_hasFDerivWithinAt {f : E → 𝕜'} {f' : E →L[𝕜] 𝕜'} {s} (x)
    (hh : HasDerivAt h₂ h₂' (f x)) (hf : HasFDerivWithinAt f f' s x) :
    HasFDerivWithinAt (h₂ ∘ f) (h₂' • f') s x :=
  hh.comp_hasFDerivAtFilter x hf hf.continuousWithinAt


theorem HasDerivAt.comp_hasFDerivWithinAt_of_eq {f : E → 𝕜'} {f' : E →L[𝕜] 𝕜'} {s} (x)
    (hh : HasDerivAt h₂ h₂' y) (hf : HasFDerivWithinAt f f' s x) (hy : y = f x) :
    HasFDerivWithinAt (h₂ ∘ f) (h₂' • f') s x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h₂ : 𝕜' → 𝕜'
    h₂' y : 𝕜'
    f : E → 𝕜'
    f' : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    s : Set E
    x : E
    hh : HasDerivAt h₂ h₂' y
    hf : HasFDerivWithinAt f f' s x
    hy : Eq y (f x)
    ⊢ HasFDerivWithinAt (Function.comp h₂ f) (HSMul.hSMul h₂' f') s x
  -/
  rw [hy] at hh; exact hh.comp_hasFDerivWithinAt x hf
                 /-
                   🎉 no goals
                 -/


theorem HasDerivWithinAt.comp_hasFDerivWithinAt {f : E → 𝕜'} {f' : E →L[𝕜] 𝕜'} {s t} (x)
    (hh : HasDerivWithinAt h₂ h₂' t (f x)) (hf : HasFDerivWithinAt f f' s x) (hst : MapsTo f s t) :
    HasFDerivWithinAt (h₂ ∘ f) (h₂' • f') s x :=
  hh.comp_hasFDerivAtFilter x hf <| hf.continuousWithinAt.tendsto_nhdsWithin hst


theorem HasDerivWithinAt.comp_hasFDerivWithinAt_of_eq {f : E → 𝕜'} {f' : E →L[𝕜] 𝕜'} {s t} (x)
    (hh : HasDerivWithinAt h₂ h₂' t y) (hf : HasFDerivWithinAt f f' s x) (hst : MapsTo f s t)
    (hy : y = f x) :
    HasFDerivWithinAt (h₂ ∘ f) (h₂' • f') s x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h₂ : 𝕜' → 𝕜'
    h₂' y : 𝕜'
    f : E → 𝕜'
    f' : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    s : Set E
    t : Set 𝕜'
    x : E
    hh : HasDerivWithinAt h₂ h₂' t y
    hf : HasFDerivWithinAt f f' s x
    hst : Set.MapsTo f s t
    hy : Eq y (f x)
    ⊢ HasFDerivWithinAt (Function.comp h₂ f) (HSMul.hSMul h₂' f') s x
  -/
  rw [hy] at hh; exact hh.comp_hasFDerivWithinAt x hf hst
                 /-
                   🎉 no goals
                 -/


theorem HasDerivAtFilter.comp (hh₂ : HasDerivAtFilter h₂ h₂' (h x) L')
    (hh : HasDerivAtFilter h h' x L) (hL : Tendsto h L L') :
    HasDerivAtFilter (h₂ ∘ h) (h₂' * h') x L := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    L : Filter 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h : 𝕜 → 𝕜'
    h₂ : 𝕜' → 𝕜'
    h' h₂' : 𝕜'
    L' : Filter 𝕜'
    hh₂ : HasDerivAtFilter h₂ h₂' (h x) L'
    hh : HasDerivAtFilter h h' x L
    hL : Filter.Tendsto h L L'
    ⊢ HasDerivAtFilter (Function.comp h₂ h) (HMul.hMul h₂' h') x L
  -/
  rw [mul_comm]
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    L : Filter 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h : 𝕜 → 𝕜'
    h₂ : 𝕜' → 𝕜'
    h' h₂' : 𝕜'
    L' : Filter 𝕜'
    hh₂ : HasDerivAtFilter h₂ h₂' (h x) L'
    hh : HasDerivAtFilter h h' x L
    hL : Filter.Tendsto h L L'
    ⊢ HasDerivAtFilter (Function.comp h₂ h) (HMul.hMul h' h₂') x L
  -/
  exact hh₂.scomp x hh hL
  /-
    🎉 no goals
  -/


theorem HasDerivAtFilter.comp_of_eq (hh₂ : HasDerivAtFilter h₂ h₂' y L')
    (hh : HasDerivAtFilter h h' x L) (hL : Tendsto h L L') (hy : y = h x) :
    HasDerivAtFilter (h₂ ∘ h) (h₂' * h') x L := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    L : Filter 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h : 𝕜 → 𝕜'
    h₂ : 𝕜' → 𝕜'
    h' h₂' : 𝕜'
    L' : Filter 𝕜'
    y : 𝕜'
    hh₂ : HasDerivAtFilter h₂ h₂' y L'
    hh : HasDerivAtFilter h h' x L
    hL : Filter.Tendsto h L L'
    hy : Eq y (h x)
    ⊢ HasDerivAtFilter (Function.comp h₂ h) (HMul.hMul h₂' h') x L
  -/
  rw [hy] at hh₂; exact hh₂.comp x hh hL
                  /-
                    🎉 no goals
                  -/


theorem HasDerivWithinAt.comp (hh₂ : HasDerivWithinAt h₂ h₂' s' (h x))
    (hh : HasDerivWithinAt h h' s x) (hst : MapsTo h s s') :
    HasDerivWithinAt (h₂ ∘ h) (h₂' * h') s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    s' : Set 𝕜'
    h : 𝕜 → 𝕜'
    h₂ : 𝕜' → 𝕜'
    h' h₂' : 𝕜'
    hh₂ : HasDerivWithinAt h₂ h₂' s' (h x)
    hh : HasDerivWithinAt h h' s x
    hst : Set.MapsTo h s s'
    ⊢ HasDerivWithinAt (Function.comp h₂ h) (HMul.hMul h₂' h') s x
  -/
  rw [mul_comm]
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    s' : Set 𝕜'
    h : 𝕜 → 𝕜'
    h₂ : 𝕜' → 𝕜'
    h' h₂' : 𝕜'
    hh₂ : HasDerivWithinAt h₂ h₂' s' (h x)
    hh : HasDerivWithinAt h h' s x
    hst : Set.MapsTo h s s'
    ⊢ HasDerivWithinAt (Function.comp h₂ h) (HMul.hMul h' h₂') s x
  -/
  exact hh₂.scomp x hh hst
  /-
    🎉 no goals
  -/


theorem HasDerivWithinAt.comp_of_eq (hh₂ : HasDerivWithinAt h₂ h₂' s' y)
    (hh : HasDerivWithinAt h h' s x) (hst : MapsTo h s s') (hy : y = h x) :
    HasDerivWithinAt (h₂ ∘ h) (h₂' * h') s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    s' : Set 𝕜'
    h : 𝕜 → 𝕜'
    h₂ : 𝕜' → 𝕜'
    h' h₂' y : 𝕜'
    hh₂ : HasDerivWithinAt h₂ h₂' s' y
    hh : HasDerivWithinAt h h' s x
    hst : Set.MapsTo h s s'
    hy : Eq y (h x)
    ⊢ HasDerivWithinAt (Function.comp h₂ h) (HMul.hMul h₂' h') s x
  -/
  rw [hy] at hh₂; exact hh₂.comp x hh hst
                  /-
                    🎉 no goals
                  -/


/-- The chain rule.

Note that the function `h₂` is a function on an algebra. If you are looking for the chain rule
with `h₂` taking values in a vector space, use `HasDerivAt.scomp`. -/
nonrec theorem HasDerivAt.comp (hh₂ : HasDerivAt h₂ h₂' (h x)) (hh : HasDerivAt h h' x) :
    HasDerivAt (h₂ ∘ h) (h₂' * h') x :=
  hh₂.comp x hh hh.continuousAt


/-- The chain rule.

Note that the function `h₂` is a function on an algebra. If you are looking for the chain rule
with `h₂` taking values in a vector space, use `HasDerivAt.scomp_of_eq`. -/
theorem HasDerivAt.comp_of_eq
    (hh₂ : HasDerivAt h₂ h₂' y) (hh : HasDerivAt h h' x) (hy : y = h x) :
    HasDerivAt (h₂ ∘ h) (h₂' * h') x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h : 𝕜 → 𝕜'
    h₂ : 𝕜' → 𝕜'
    h' h₂' y : 𝕜'
    hh₂ : HasDerivAt h₂ h₂' y
    hh : HasDerivAt h h' x
    hy : Eq y (h x)
    ⊢ HasDerivAt (Function.comp h₂ h) (HMul.hMul h₂' h') x
  -/
  rw [hy] at hh₂; exact hh₂.comp x hh
                  /-
                    🎉 no goals
                  -/


theorem HasStrictDerivAt.comp (hh₂ : HasStrictDerivAt h₂ h₂' (h x)) (hh : HasStrictDerivAt h h' x) :
    HasStrictDerivAt (h₂ ∘ h) (h₂' * h') x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h : 𝕜 → 𝕜'
    h₂ : 𝕜' → 𝕜'
    h' h₂' : 𝕜'
    hh₂ : HasStrictDerivAt h₂ h₂' (h x)
    hh : HasStrictDerivAt h h' x
    ⊢ HasStrictDerivAt (Function.comp h₂ h) (HMul.hMul h₂' h') x
  -/
  rw [mul_comm]
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h : 𝕜 → 𝕜'
    h₂ : 𝕜' → 𝕜'
    h' h₂' : 𝕜'
    hh₂ : HasStrictDerivAt h₂ h₂' (h x)
    hh : HasStrictDerivAt h h' x
    ⊢ HasStrictDerivAt (Function.comp h₂ h) (HMul.hMul h' h₂') x
  -/
  exact hh₂.scomp x hh
  /-
    🎉 no goals
  -/


theorem HasStrictDerivAt.comp_of_eq
    (hh₂ : HasStrictDerivAt h₂ h₂' y) (hh : HasStrictDerivAt h h' x) (hy : y = h x) :
    HasStrictDerivAt (h₂ ∘ h) (h₂' * h') x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h : 𝕜 → 𝕜'
    h₂ : 𝕜' → 𝕜'
    h' h₂' y : 𝕜'
    hh₂ : HasStrictDerivAt h₂ h₂' y
    hh : HasStrictDerivAt h h' x
    hy : Eq y (h x)
    ⊢ HasStrictDerivAt (Function.comp h₂ h) (HMul.hMul h₂' h') x
  -/
  rw [hy] at hh₂; exact hh₂.comp x hh
                  /-
                    🎉 no goals
                  -/


theorem HasDerivAt.comp_hasDerivWithinAt (hh₂ : HasDerivAt h₂ h₂' (h x))
    (hh : HasDerivWithinAt h h' s x) : HasDerivWithinAt (h₂ ∘ h) (h₂' * h') s x :=
  hh₂.hasDerivWithinAt.comp x hh (mapsTo_univ _ _)


theorem HasDerivAt.comp_hasDerivWithinAt_of_eq (hh₂ : HasDerivAt h₂ h₂' y)
    (hh : HasDerivWithinAt h h' s x) (hy : y = h x) :
    HasDerivWithinAt (h₂ ∘ h) (h₂' * h') s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h : 𝕜 → 𝕜'
    h₂ : 𝕜' → 𝕜'
    h' h₂' y : 𝕜'
    hh₂ : HasDerivAt h₂ h₂' y
    hh : HasDerivWithinAt h h' s x
    hy : Eq y (h x)
    ⊢ HasDerivWithinAt (Function.comp h₂ h) (HMul.hMul h₂' h') s x
  -/
  rw [hy] at hh₂; exact hh₂.comp_hasDerivWithinAt x hh
                  /-
                    🎉 no goals
                  -/


theorem derivWithin_comp (hh₂ : DifferentiableWithinAt 𝕜' h₂ s' (h x))
    (hh : DifferentiableWithinAt 𝕜 h s x) (hs : MapsTo h s s') (hxs : UniqueDiffWithinAt 𝕜 s x) :
    derivWithin (h₂ ∘ h) s x = derivWithin h₂ s' (h x) * derivWithin h s x :=
  (hh₂.hasDerivWithinAt.comp x hh.hasDerivWithinAt hs).derivWithin hxs


@[deprecated (since := "2024-10-31")] alias derivWithin.comp := derivWithin_comp


theorem derivWithin_comp_of_eq (hh₂ : DifferentiableWithinAt 𝕜' h₂ s' y)
    (hh : DifferentiableWithinAt 𝕜 h s x) (hs : MapsTo h s s') (hxs : UniqueDiffWithinAt 𝕜 s x)
    (hy : h x = y) :
    derivWithin (h₂ ∘ h) s x = derivWithin h₂ s' (h x) * derivWithin h s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    s' : Set 𝕜'
    h : 𝕜 → 𝕜'
    h₂ : 𝕜' → 𝕜'
    y : 𝕜'
    hh₂ : DifferentiableWithinAt 𝕜' h₂ s' y
    hh : DifferentiableWithinAt 𝕜 h s x
    hs : Set.MapsTo h s s'
    hxs : UniqueDiffWithinAt 𝕜 s x
    hy : Eq (h x) y
    ⊢ Eq (derivWithin (Function.comp h₂ h) s x) (HMul.hMul (derivWithin h₂ s' (h x …
  -/
  subst hy; exact derivWithin_comp x hh₂ hh hs hxs
            /-
              🎉 no goals
            -/


@[deprecated (since := "2024-10-31")] alias derivWithin.comp_of_eq := derivWithin_comp_of_eq


theorem deriv_comp (hh₂ : DifferentiableAt 𝕜' h₂ (h x)) (hh : DifferentiableAt 𝕜 h x) :
    deriv (h₂ ∘ h) x = deriv h₂ (h x) * deriv h x :=
  (hh₂.hasDerivAt.comp x hh.hasDerivAt).deriv


@[deprecated (since := "2024-10-31")] alias deriv.comp := deriv_comp


theorem deriv_comp_of_eq (hh₂ : DifferentiableAt 𝕜' h₂ y) (hh : DifferentiableAt 𝕜 h x)
    (hy : h x = y) :
    deriv (h₂ ∘ h) x = deriv h₂ (h x) * deriv h x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    h : 𝕜 → 𝕜'
    h₂ : 𝕜' → 𝕜'
    y : 𝕜'
    hh₂ : DifferentiableAt 𝕜' h₂ y
    hh : DifferentiableAt 𝕜 h x
    hy : Eq (h x) y
    ⊢ Eq (deriv (Function.comp h₂ h) x) (HMul.hMul (deriv h₂ (h x)) (deriv h x))
  -/
  subst hy; exact deriv_comp x hh₂ hh
            /-
              🎉 no goals
            -/


@[deprecated (since := "2024-10-31")] alias deriv.comp_of_eq := deriv_comp_of_eq


protected nonrec theorem HasDerivAtFilter.iterate {f : 𝕜 → 𝕜} {f' : 𝕜}
    (hf : HasDerivAtFilter f f' x L) (hL : Tendsto f L L) (hx : f x = x) (n : ℕ) :
    HasDerivAtFilter f^[n] (f' ^ n) x L := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    L : Filter 𝕜
    f : 𝕜 → 𝕜
    f' : 𝕜
    hf : HasDerivAtFilter f f' x L
    hL : Filter.Tendsto f L L
    hx : Eq (f x) x
    n : Nat
    ⊢ HasDerivAtFilter (Nat.iterate f n) (HPow.hPow f' n) x L
  -/
  have := hf.iterate hL hx n
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    L : Filter 𝕜
    f : 𝕜 → 𝕜
    f' : 𝕜
    hf : HasDerivAtFilter f f' x L
    hL : Filter.Tendsto f L L
    hx : Eq (f x) x
    n : Nat
    this : HasFDerivAtFilter (Nat.iterate f n) (HPow.hPow (ContinuousLinearMap.smu …
    ⊢ HasDerivAtFilter (Nat.iterate f n) (HPow.hPow f' n) x L
  -/
  rwa [ContinuousLinearMap.smulRight_one_pow] at this
  /-
    🎉 no goals
  -/


protected nonrec theorem HasDerivAt.iterate {f : 𝕜 → 𝕜} {f' : 𝕜} (hf : HasDerivAt f f' x)
    (hx : f x = x) (n : ℕ) : HasDerivAt f^[n] (f' ^ n) x :=
                                                   /-
                                                     𝕜 : Type u
                                                     inst✝ : NontriviallyNormedField 𝕜
                                                     x : 𝕜
                                                     f : 𝕜 → 𝕜
                                                     f' : 𝕜
                                                     hf : HasDerivAt f f' x
                                                     hx : Eq (f x) x
                                                     n : Nat
                                                     this : Filter.Tendsto f (nhds x) (nhds (f x))
                                                     ⊢ Filter.Tendsto f (nhds x) (nhds x)
                                                   -/
  hf.iterate _ (have := hf.tendsto_nhds le_rfl; by rwa [hx] at this) hx n
                                                   /-
                                                     🎉 no goals
                                                   -/


protected theorem HasDerivWithinAt.iterate {f : 𝕜 → 𝕜} {f' : 𝕜} (hf : HasDerivWithinAt f f' s x)
    (hx : f x = x) (hs : MapsTo f s s) (n : ℕ) : HasDerivWithinAt f^[n] (f' ^ n) s x := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    f' : 𝕜
    hf : HasDerivWithinAt f f' s x
    hx : Eq (f x) x
    hs : Set.MapsTo f s s
    n : Nat
    ⊢ HasDerivWithinAt (Nat.iterate f n) (HPow.hPow f' n) s x
  -/
  have := HasFDerivWithinAt.iterate hf hx hs n
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    f' : 𝕜
    hf : HasDerivWithinAt f f' s x
    hx : Eq (f x) x
    hs : Set.MapsTo f s s
    n : Nat
    this : HasFDerivWithinAt (Nat.iterate f n) (HPow.hPow (ContinuousLinearMap.smu …
    ⊢ HasDerivWithinAt (Nat.iterate f n) (HPow.hPow f' n) s x
  -/
  rwa [ContinuousLinearMap.smulRight_one_pow] at this
  /-
    🎉 no goals
  -/


protected nonrec theorem HasStrictDerivAt.iterate {f : 𝕜 → 𝕜} {f' : 𝕜}
    (hf : HasStrictDerivAt f f' x) (hx : f x = x) (n : ℕ) :
    HasStrictDerivAt f^[n] (f' ^ n) x := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    f : 𝕜 → 𝕜
    f' : 𝕜
    hf : HasStrictDerivAt f f' x
    hx : Eq (f x) x
    n : Nat
    ⊢ HasStrictDerivAt (Nat.iterate f n) (HPow.hPow f' n) x
  -/
  have := hf.iterate hx n
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    f : 𝕜 → 𝕜
    f' : 𝕜
    hf : HasStrictDerivAt f f' x
    hx : Eq (f x) x
    n : Nat
    this : HasStrictFDerivAt (Nat.iterate f n) (HPow.hPow (ContinuousLinearMap.smu …
    ⊢ HasStrictDerivAt (Nat.iterate f n) (HPow.hPow f' n) x
  -/
  rwa [ContinuousLinearMap.smulRight_one_pow] at this
  /-
    🎉 no goals
  -/


/-- The composition `l ∘ f` where `l : F → E` and `f : 𝕜 → F`, has a derivative within a set
equal to the Fréchet derivative of `l` applied to the derivative of `f`. -/
theorem HasFDerivWithinAt.comp_hasDerivWithinAt {t : Set F} (hl : HasFDerivWithinAt l l' t (f x))
    (hf : HasDerivWithinAt f f' s x) (hst : MapsTo f s t) :
    HasDerivWithinAt (l ∘ f) (l' f') s x := by
  simpa only [one_apply, one_smul, smulRight_apply, coe_comp', (· ∘ ·)] using
    (hl.comp x hf.hasFDerivWithinAt hst).hasDerivWithinAt


/-- The composition `l ∘ f` where `l : F → E` and `f : 𝕜 → F`, has a derivative within a set
equal to the Fréchet derivative of `l` applied to the derivative of `f`. -/
theorem HasFDerivWithinAt.comp_hasDerivWithinAt_of_eq {t : Set F}
    (hl : HasFDerivWithinAt l l' t y)
    (hf : HasDerivWithinAt f f' s x) (hst : MapsTo f s t) (hy : y = f x) :
    HasDerivWithinAt (l ∘ f) (l' f') s x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type w
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → F
    f' : F
    x : 𝕜
    s : Set 𝕜
    l : F → E
    l' : ContinuousLinearMap (RingHom.id 𝕜) F E
    y : F
    t : Set F
    hl : HasFDerivWithinAt l l' t y
    hf : HasDerivWithinAt f f' s x
    hst : Set.MapsTo f s t
    hy : Eq y (f x)
    ⊢ HasDerivWithinAt (Function.comp l f) (l' f') s x
  -/
  rw [hy] at hl; exact hl.comp_hasDerivWithinAt x hf hst
                 /-
                   🎉 no goals
                 -/


theorem HasFDerivAt.comp_hasDerivWithinAt (hl : HasFDerivAt l l' (f x))
    (hf : HasDerivWithinAt f f' s x) : HasDerivWithinAt (l ∘ f) (l' f') s x :=
  hl.hasFDerivWithinAt.comp_hasDerivWithinAt x hf (mapsTo_univ _ _)


theorem HasFDerivAt.comp_hasDerivWithinAt_of_eq (hl : HasFDerivAt l l' y)
    (hf : HasDerivWithinAt f f' s x) (hy : y = f x) :
    HasDerivWithinAt (l ∘ f) (l' f') s x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type w
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → F
    f' : F
    x : 𝕜
    s : Set 𝕜
    l : F → E
    l' : ContinuousLinearMap (RingHom.id 𝕜) F E
    y : F
    hl : HasFDerivAt l l' y
    hf : HasDerivWithinAt f f' s x
    hy : Eq y (f x)
    ⊢ HasDerivWithinAt (Function.comp l f) (l' f') s x
  -/
  rw [hy] at hl; exact hl.comp_hasDerivWithinAt x hf
                 /-
                   🎉 no goals
                 -/


/-- The composition `l ∘ f` where `l : F → E` and `f : 𝕜 → F`, has a derivative equal to the
Fréchet derivative of `l` applied to the derivative of `f`. -/
theorem HasFDerivAt.comp_hasDerivAt (hl : HasFDerivAt l l' (f x)) (hf : HasDerivAt f f' x) :
    HasDerivAt (l ∘ f) (l' f') x :=
  hasDerivWithinAt_univ.mp <| hl.comp_hasDerivWithinAt x hf.hasDerivWithinAt


/-- The composition `l ∘ f` where `l : F → E` and `f : 𝕜 → F`, has a derivative equal to the
Fréchet derivative of `l` applied to the derivative of `f`. -/
theorem HasFDerivAt.comp_hasDerivAt_of_eq
    (hl : HasFDerivAt l l' y) (hf : HasDerivAt f f' x) (hy : y = f x) :
    HasDerivAt (l ∘ f) (l' f') x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type w
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → F
    f' : F
    x : 𝕜
    l : F → E
    l' : ContinuousLinearMap (RingHom.id 𝕜) F E
    y : F
    hl : HasFDerivAt l l' y
    hf : HasDerivAt f f' x
    hy : Eq y (f x)
    ⊢ HasDerivAt (Function.comp l f) (l' f') x
  -/
  rw [hy] at hl; exact hl.comp_hasDerivAt x hf
                 /-
                   🎉 no goals
                 -/


theorem HasStrictFDerivAt.comp_hasStrictDerivAt (hl : HasStrictFDerivAt l l' (f x))
    (hf : HasStrictDerivAt f f' x) : HasStrictDerivAt (l ∘ f) (l' f') x := by
  simpa only [one_apply, one_smul, smulRight_apply, coe_comp', (· ∘ ·)] using
    (hl.comp x hf.hasStrictFDerivAt).hasStrictDerivAt


theorem HasStrictFDerivAt.comp_hasStrictDerivAt_of_eq (hl : HasStrictFDerivAt l l' y)
    (hf : HasStrictDerivAt f f' x) (hy : y = f x) :
    HasStrictDerivAt (l ∘ f) (l' f') x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type w
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → F
    f' : F
    x : 𝕜
    l : F → E
    l' : ContinuousLinearMap (RingHom.id 𝕜) F E
    y : F
    hl : HasStrictFDerivAt l l' y
    hf : HasStrictDerivAt f f' x
    hy : Eq y (f x)
    ⊢ HasStrictDerivAt (Function.comp l f) (l' f') x
  -/
  rw [hy] at hl; exact hl.comp_hasStrictDerivAt x hf
                 /-
                   🎉 no goals
                 -/


theorem fderivWithin_comp_derivWithin {t : Set F} (hl : DifferentiableWithinAt 𝕜 l t (f x))
    (hf : DifferentiableWithinAt 𝕜 f s x) (hs : MapsTo f s t) (hxs : UniqueDiffWithinAt 𝕜 s x) :
    derivWithin (l ∘ f) s x = (fderivWithin 𝕜 l t (f x) : F → E) (derivWithin f s x) :=
  (hl.hasFDerivWithinAt.comp_hasDerivWithinAt x hf.hasDerivWithinAt hs).derivWithin hxs


@[deprecated (since := "2024-10-31")]
alias fderivWithin.comp_derivWithin := fderivWithin_comp_derivWithin


theorem fderivWithin_comp_derivWithin_of_eq {t : Set F} (hl : DifferentiableWithinAt 𝕜 l t y)
    (hf : DifferentiableWithinAt 𝕜 f s x) (hs : MapsTo f s t) (hxs : UniqueDiffWithinAt 𝕜 s x)
    (hy : y = f x) :
    derivWithin (l ∘ f) s x = (fderivWithin 𝕜 l t (f x) : F → E) (derivWithin f s x) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type w
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    l : F → E
    y : F
    t : Set F
    hl : DifferentiableWithinAt 𝕜 l t y
    hf : DifferentiableWithinAt 𝕜 f s x
    hs : Set.MapsTo f s t
    hxs : UniqueDiffWithinAt 𝕜 s x
    hy : Eq y (f x)
    ⊢ Eq (derivWithin (Function.comp l f) s x) ((fderivWithin 𝕜 l t (f x)) (derivW …
  -/
  rw [hy] at hl; exact fderivWithin_comp_derivWithin x hl hf hs hxs
                 /-
                   🎉 no goals
                 -/


@[deprecated (since := "2024-10-31")]
alias fderivWithin.comp_derivWithin_of_eq := fderivWithin_comp_derivWithin_of_eq


theorem fderiv_comp_deriv (hl : DifferentiableAt 𝕜 l (f x)) (hf : DifferentiableAt 𝕜 f x) :
    deriv (l ∘ f) x = (fderiv 𝕜 l (f x) : F → E) (deriv f x) :=
  (hl.hasFDerivAt.comp_hasDerivAt x hf.hasDerivAt).deriv


@[deprecated (since := "2024-10-31")]
alias fderiv.comp_deriv := fderiv_comp_deriv


theorem fderiv_comp_deriv_of_eq (hl : DifferentiableAt 𝕜 l y) (hf : DifferentiableAt 𝕜 f x)
    (hy : y = f x) :
    deriv (l ∘ f) x = (fderiv 𝕜 l (f x) : F → E) (deriv f x) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type w
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → F
    x : 𝕜
    l : F → E
    y : F
    hl : DifferentiableAt 𝕜 l y
    hf : DifferentiableAt 𝕜 f x
    hy : Eq y (f x)
    ⊢ Eq (deriv (Function.comp l f) x) ((fderiv 𝕜 l (f x)) (deriv f x))
  -/
  rw [hy] at hl; exact fderiv_comp_deriv x hl hf
                 /-
                   🎉 no goals
                 -/


@[deprecated (since := "2024-10-31")]
alias fderiv.comp_deriv_of_eq := fderiv_comp_deriv_of_eq


