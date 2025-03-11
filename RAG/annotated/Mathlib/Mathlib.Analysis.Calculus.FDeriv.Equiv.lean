@[fun_prop]
protected theorem hasStrictFDerivAt : HasStrictFDerivAt iso (iso : E →L[𝕜] F) x :=
  iso.toContinuousLinearMap.hasStrictFDerivAt


@[fun_prop]
protected theorem hasFDerivWithinAt : HasFDerivWithinAt iso (iso : E →L[𝕜] F) s x :=
  iso.toContinuousLinearMap.hasFDerivWithinAt


@[fun_prop]
protected theorem hasFDerivAt : HasFDerivAt iso (iso : E →L[𝕜] F) x :=
  iso.toContinuousLinearMap.hasFDerivAtFilter


@[fun_prop]
protected theorem differentiableAt : DifferentiableAt 𝕜 iso x :=
  iso.hasFDerivAt.differentiableAt


@[fun_prop]
protected theorem differentiableWithinAt : DifferentiableWithinAt 𝕜 iso s x :=
  iso.differentiableAt.differentiableWithinAt


protected theorem fderiv : fderiv 𝕜 iso x = iso :=
  iso.hasFDerivAt.fderiv


protected theorem fderivWithin (hxs : UniqueDiffWithinAt 𝕜 s x) : fderivWithin 𝕜 iso s x = iso :=
  iso.toContinuousLinearMap.fderivWithin hxs


@[fun_prop]
protected theorem differentiable : Differentiable 𝕜 iso := fun _ => iso.differentiableAt


@[fun_prop]
protected theorem differentiableOn : DifferentiableOn 𝕜 iso s :=
  iso.differentiable.differentiableOn


theorem comp_differentiableWithinAt_iff {f : G → E} {s : Set G} {x : G} :
    DifferentiableWithinAt 𝕜 (iso ∘ f) s x ↔ DifferentiableWithinAt 𝕜 f s x := by
  refine
    ⟨fun H => ?_, fun H => iso.differentiable.differentiableAt.comp_differentiableWithinAt x H⟩
  have : DifferentiableWithinAt 𝕜 (iso.symm ∘ iso ∘ f) s x :=
    iso.symm.differentiable.differentiableAt.comp_differentiableWithinAt x H
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : G → E
    s : Set G
    x : G
    H : DifferentiableWithinAt 𝕜 (Function.comp (⇑iso) f) s x
    this : DifferentiableWithinAt 𝕜 (Function.comp (⇑iso.symm) (Function.comp (⇑is …
    ⊢ DifferentiableWithinAt 𝕜 f s x
  -/
  rwa [← Function.comp_assoc iso.symm iso f, iso.symm_comp_self] at this
  /-
    🎉 no goals
  -/


theorem comp_differentiableAt_iff {f : G → E} {x : G} :
    DifferentiableAt 𝕜 (iso ∘ f) x ↔ DifferentiableAt 𝕜 f x := by
  rw [← differentiableWithinAt_univ, ← differentiableWithinAt_univ,
    iso.comp_differentiableWithinAt_iff]


theorem comp_differentiableOn_iff {f : G → E} {s : Set G} :
    DifferentiableOn 𝕜 (iso ∘ f) s ↔ DifferentiableOn 𝕜 f s := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : G → E
    s : Set G
    ⊢ Iff (DifferentiableOn 𝕜 (Function.comp (⇑iso) f) s) (DifferentiableOn 𝕜 f s)
  -/
  rw [DifferentiableOn, DifferentiableOn]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : G → E
    s : Set G
    ⊢ Iff (∀ (x : G), Membership.mem s x → DifferentiableWithinAt 𝕜 (Function.comp …
  -/
  simp only [iso.comp_differentiableWithinAt_iff]
  /-
    🎉 no goals
  -/


theorem comp_differentiable_iff {f : G → E} : Differentiable 𝕜 (iso ∘ f) ↔ Differentiable 𝕜 f := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : G → E
    ⊢ Iff (Differentiable 𝕜 (Function.comp (⇑iso) f)) (Differentiable 𝕜 f)
  -/
  rw [← differentiableOn_univ, ← differentiableOn_univ]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : G → E
    ⊢ Iff (DifferentiableOn 𝕜 (Function.comp (⇑iso) f) Set.univ) (DifferentiableOn …
  -/
  exact iso.comp_differentiableOn_iff
  /-
    🎉 no goals
  -/


theorem comp_hasFDerivWithinAt_iff {f : G → E} {s : Set G} {x : G} {f' : G →L[𝕜] E} :
    HasFDerivWithinAt (iso ∘ f) ((iso : E →L[𝕜] F).comp f') s x ↔ HasFDerivWithinAt f f' s x := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : G → E
    s : Set G
    x : G
    f' : ContinuousLinearMap (RingHom.id 𝕜) G E
    ⊢ Iff (HasFDerivWithinAt (Function.comp (⇑iso) f) ((↑iso).comp f') s x) (HasFD …
  -/
  refine ⟨fun H => ?_, fun H => iso.hasFDerivAt.comp_hasFDerivWithinAt x H⟩
  have A : f = iso.symm ∘ iso ∘ f := by
    rw [← Function.comp_assoc, iso.symm_comp_self]
    rfl
  have B : f' = (iso.symm : F →L[𝕜] E).comp ((iso : E →L[𝕜] F).comp f') := by
    rw [← ContinuousLinearMap.comp_assoc, iso.coe_symm_comp_coe, ContinuousLinearMap.id_comp]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : G → E
    s : Set G
    x : G
    f' : ContinuousLinearMap (RingHom.id 𝕜) G E
    H : HasFDerivWithinAt (Function.comp (⇑iso) f) ((↑iso).comp f') s x
    A : Eq f (Function.comp (⇑iso.symm) (Function.comp (⇑iso) f))
    B : Eq f' ((↑iso.symm).comp ((↑iso).comp f'))
    ⊢ HasFDerivWithinAt f f' s x
  -/
  rw [A, B]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : G → E
    s : Set G
    x : G
    f' : ContinuousLinearMap (RingHom.id 𝕜) G E
    H : HasFDerivWithinAt (Function.comp (⇑iso) f) ((↑iso).comp f') s x
    A : Eq f (Function.comp (⇑iso.symm) (Function.comp (⇑iso) f))
    B : Eq f' ((↑iso.symm).comp ((↑iso).comp f'))
    ⊢ HasFDerivWithinAt (Function.comp (⇑iso.symm) (Function.comp (⇑iso) f)) ((↑is …
  -/
  exact iso.symm.hasFDerivAt.comp_hasFDerivWithinAt x H
  /-
    🎉 no goals
  -/


theorem comp_hasStrictFDerivAt_iff {f : G → E} {x : G} {f' : G →L[𝕜] E} :
    HasStrictFDerivAt (iso ∘ f) ((iso : E →L[𝕜] F).comp f') x ↔ HasStrictFDerivAt f f' x := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : G → E
    x : G
    f' : ContinuousLinearMap (RingHom.id 𝕜) G E
    ⊢ Iff (HasStrictFDerivAt (Function.comp (⇑iso) f) ((↑iso).comp f') x) (HasStri …
  -/
  refine ⟨fun H => ?_, fun H => iso.hasStrictFDerivAt.comp x H⟩
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : G → E
    x : G
    f' : ContinuousLinearMap (RingHom.id 𝕜) G E
    H : HasStrictFDerivAt (Function.comp (⇑iso) f) ((↑iso).comp f') x
    ⊢ HasStrictFDerivAt f f' x
  -/
  convert iso.symm.hasStrictFDerivAt.comp x H using 1 <;>
    /-
      case h.e'_11
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      f : G → E
      x : G
      f' : ContinuousLinearMap (RingHom.id 𝕜) G E
      H : HasStrictFDerivAt (Function.comp (⇑iso) f) ((↑iso).comp f') x
      ⊢ Eq f fun x => iso.symm (Function.comp (⇑iso) f x)
    -/
              /-
                🎉 no goals
              -/
    ext z <;> apply (iso.symm_apply_apply _).symm
              /-
                🎉 no goals
              -/


theorem comp_hasFDerivAt_iff {f : G → E} {x : G} {f' : G →L[𝕜] E} :
    HasFDerivAt (iso ∘ f) ((iso : E →L[𝕜] F).comp f') x ↔ HasFDerivAt f f' x := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : G → E
    x : G
    f' : ContinuousLinearMap (RingHom.id 𝕜) G E
    ⊢ Iff (HasFDerivAt (Function.comp (⇑iso) f) ((↑iso).comp f') x) (HasFDerivAt f …
  -/
  simp_rw [← hasFDerivWithinAt_univ, iso.comp_hasFDerivWithinAt_iff]
  /-
    🎉 no goals
  -/


theorem comp_hasFDerivWithinAt_iff' {f : G → E} {s : Set G} {x : G} {f' : G →L[𝕜] F} :
    HasFDerivWithinAt (iso ∘ f) f' s x ↔
      HasFDerivWithinAt f ((iso.symm : F →L[𝕜] E).comp f') s x := by
  rw [← iso.comp_hasFDerivWithinAt_iff, ← ContinuousLinearMap.comp_assoc, iso.coe_comp_coe_symm,
    ContinuousLinearMap.id_comp]


theorem comp_hasFDerivAt_iff' {f : G → E} {x : G} {f' : G →L[𝕜] F} :
    HasFDerivAt (iso ∘ f) f' x ↔ HasFDerivAt f ((iso.symm : F →L[𝕜] E).comp f') x := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : G → E
    x : G
    f' : ContinuousLinearMap (RingHom.id 𝕜) G F
    ⊢ Iff (HasFDerivAt (Function.comp (⇑iso) f) f' x) (HasFDerivAt f ((↑iso.symm). …
  -/
  simp_rw [← hasFDerivWithinAt_univ, iso.comp_hasFDerivWithinAt_iff']
  /-
    🎉 no goals
  -/


theorem comp_fderivWithin {f : G → E} {s : Set G} {x : G} (hxs : UniqueDiffWithinAt 𝕜 s x) :
    fderivWithin 𝕜 (iso ∘ f) s x = (iso : E →L[𝕜] F).comp (fderivWithin 𝕜 f s x) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : G → E
    s : Set G
    x : G
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (fderivWithin 𝕜 (Function.comp (⇑iso) f) s x) ((↑iso).comp (fderivWithin  …
  -/
  by_cases h : DifferentiableWithinAt 𝕜 f s x
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      f : G → E
      s : Set G
      x : G
      hxs : UniqueDiffWithinAt 𝕜 s x
      h : DifferentiableWithinAt 𝕜 f s x
      ⊢ Eq (fderivWithin 𝕜 (Function.comp (⇑iso) f) s x) ((↑iso).comp (fderivWithin  …
    -/
  · rw [fderiv_comp_fderivWithin x iso.differentiableAt h hxs, iso.fderiv]
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      f : G → E
      s : Set G
      x : G
      hxs : UniqueDiffWithinAt 𝕜 s x
      h : Not (DifferentiableWithinAt 𝕜 f s x)
      ⊢ Eq (fderivWithin 𝕜 (Function.comp (⇑iso) f) s x) ((↑iso).comp (fderivWithin  …
    -/
  · have : ¬DifferentiableWithinAt 𝕜 (iso ∘ f) s x := mt iso.comp_differentiableWithinAt_iff.1 h
    rw [fderivWithin_zero_of_not_differentiableWithinAt h,
      fderivWithin_zero_of_not_differentiableWithinAt this, ContinuousLinearMap.comp_zero]


theorem comp_fderiv {f : G → E} {x : G} :
    fderiv 𝕜 (iso ∘ f) x = (iso : E →L[𝕜] F).comp (fderiv 𝕜 f x) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : G → E
    x : G
    ⊢ Eq (fderiv 𝕜 (Function.comp (⇑iso) f) x) ((↑iso).comp (fderiv 𝕜 f x))
  -/
  rw [← fderivWithin_univ, ← fderivWithin_univ]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : G → E
    x : G
    ⊢ Eq (fderivWithin 𝕜 (Function.comp (⇑iso) f) Set.univ x) ((↑iso).comp (fderiv …
  -/
  exact iso.comp_fderivWithin uniqueDiffWithinAt_univ
  /-
    🎉 no goals
  -/


lemma _root_.fderivWithin_continuousLinearEquiv_comp (L : G ≃L[𝕜] G') (f : E → (F →L[𝕜] G))
    (hs : UniqueDiffWithinAt 𝕜 s x) :
    fderivWithin 𝕜 (fun x ↦ (L : G →L[𝕜] G').comp (f x)) s x =
      (((ContinuousLinearEquiv.refl 𝕜 F).arrowCongr L)) ∘L (fderivWithin 𝕜 f s x) := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    G' : Type u_5
    inst✝¹ : NormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    x : E
    s : Set E
    L : ContinuousLinearEquiv (RingHom.id 𝕜) G G'
    f : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    hs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (fderivWithin 𝕜 (fun x => (↑L).comp (f x)) s x) ((↑((ContinuousLinearEqui …
  -/
  change fderivWithin 𝕜 (((ContinuousLinearEquiv.refl 𝕜 F).arrowCongr L) ∘ f) s x = _
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    G' : Type u_5
    inst✝¹ : NormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    x : E
    s : Set E
    L : ContinuousLinearEquiv (RingHom.id 𝕜) G G'
    f : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    hs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (fderivWithin 𝕜 (Function.comp (⇑((ContinuousLinearEquiv.refl 𝕜 F).arrowC …
  -/
  rw [ContinuousLinearEquiv.comp_fderivWithin _ hs]
  /-
    🎉 no goals
  -/


lemma _root_.fderiv_continuousLinearEquiv_comp (L : G ≃L[𝕜] G') (f : E → (F →L[𝕜] G)) (x : E) :
    fderiv 𝕜 (fun x ↦ (L : G →L[𝕜] G').comp (f x)) x =
      (((ContinuousLinearEquiv.refl 𝕜 F).arrowCongr L)) ∘L (fderiv 𝕜 f x) := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    G' : Type u_5
    inst✝¹ : NormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    L : ContinuousLinearEquiv (RingHom.id 𝕜) G G'
    f : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    x : E
    ⊢ Eq (fderiv 𝕜 (fun x => (↑L).comp (f x)) x) ((↑((ContinuousLinearEquiv.refl 𝕜 …
  -/
  change fderiv 𝕜 (((ContinuousLinearEquiv.refl 𝕜 F).arrowCongr L) ∘ f) x = _
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    G' : Type u_5
    inst✝¹ : NormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    L : ContinuousLinearEquiv (RingHom.id 𝕜) G G'
    f : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    x : E
    ⊢ Eq (fderiv 𝕜 (Function.comp (⇑((ContinuousLinearEquiv.refl 𝕜 F).arrowCongr L …
  -/
  rw [ContinuousLinearEquiv.comp_fderiv]
  /-
    🎉 no goals
  -/


lemma _root_.fderiv_continuousLinearEquiv_comp' (L : G ≃L[𝕜] G') (f : E → (F →L[𝕜] G)) :
    fderiv 𝕜 (fun x ↦ (L : G →L[𝕜] G').comp (f x)) =
      fun x ↦ (((ContinuousLinearEquiv.refl 𝕜 F).arrowCongr L)) ∘L (fderiv 𝕜 f x) := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    G' : Type u_5
    inst✝¹ : NormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    L : ContinuousLinearEquiv (RingHom.id 𝕜) G G'
    f : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    ⊢ Eq (fderiv 𝕜 fun x => (↑L).comp (f x)) fun x => (↑((ContinuousLinearEquiv.re …
  -/
  ext x : 1
  /-
    case h
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    G' : Type u_5
    inst✝¹ : NormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    L : ContinuousLinearEquiv (RingHom.id 𝕜) G G'
    f : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    x : E
    ⊢ Eq (fderiv 𝕜 (fun x => (↑L).comp (f x)) x) ((↑((ContinuousLinearEquiv.refl 𝕜 …
  -/
  exact fderiv_continuousLinearEquiv_comp L f x
  /-
    🎉 no goals
  -/


theorem comp_right_differentiableWithinAt_iff {f : F → G} {s : Set F} {x : E} :
    DifferentiableWithinAt 𝕜 (f ∘ iso) (iso ⁻¹' s) x ↔ DifferentiableWithinAt 𝕜 f s (iso x) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    s : Set F
    x : E
    ⊢ Iff (DifferentiableWithinAt 𝕜 (Function.comp f ⇑iso) (Set.preimage (⇑iso) s) …
  -/
  refine ⟨fun H => ?_, fun H => H.comp x iso.differentiableWithinAt (mapsTo_preimage _ s)⟩
  have : DifferentiableWithinAt 𝕜 ((f ∘ iso) ∘ iso.symm) s (iso x) := by
    rw [← iso.symm_apply_apply x] at H
    apply H.comp (iso x) iso.symm.differentiableWithinAt
    intro y hy
    simpa only [mem_preimage, apply_symm_apply] using hy
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    s : Set F
    x : E
    H : DifferentiableWithinAt 𝕜 (Function.comp f ⇑iso) (Set.preimage (⇑iso) s) x
    this : DifferentiableWithinAt 𝕜 (Function.comp (Function.comp f ⇑iso) ⇑iso.sym …
    ⊢ DifferentiableWithinAt 𝕜 f s (iso x)
  -/
  rwa [Function.comp_assoc, iso.self_comp_symm] at this
  /-
    🎉 no goals
  -/


theorem comp_right_differentiableAt_iff {f : F → G} {x : E} :
    DifferentiableAt 𝕜 (f ∘ iso) x ↔ DifferentiableAt 𝕜 f (iso x) := by
  simp only [← differentiableWithinAt_univ, ← iso.comp_right_differentiableWithinAt_iff,
    preimage_univ]


theorem comp_right_differentiableOn_iff {f : F → G} {s : Set F} :
    DifferentiableOn 𝕜 (f ∘ iso) (iso ⁻¹' s) ↔ DifferentiableOn 𝕜 f s := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    s : Set F
    ⊢ Iff (DifferentiableOn 𝕜 (Function.comp f ⇑iso) (Set.preimage (⇑iso) s)) (Dif …
  -/
  refine ⟨fun H y hy => ?_, fun H y hy => iso.comp_right_differentiableWithinAt_iff.2 (H _ hy)⟩
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    s : Set F
    H : DifferentiableOn 𝕜 (Function.comp f ⇑iso) (Set.preimage (⇑iso) s)
    y : F
    hy : Membership.mem s y
    ⊢ DifferentiableWithinAt 𝕜 f s y
  -/
  rw [← iso.apply_symm_apply y, ← comp_right_differentiableWithinAt_iff]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    s : Set F
    H : DifferentiableOn 𝕜 (Function.comp f ⇑iso) (Set.preimage (⇑iso) s)
    y : F
    hy : Membership.mem s y
    ⊢ DifferentiableWithinAt 𝕜 (Function.comp f ⇑iso) (Set.preimage (⇑iso) s) (iso …
  -/
  apply H
  /-
    case a
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    s : Set F
    H : DifferentiableOn 𝕜 (Function.comp f ⇑iso) (Set.preimage (⇑iso) s)
    y : F
    hy : Membership.mem s y
    ⊢ Membership.mem (Set.preimage (⇑iso) s) (iso.symm y)
  -/
  simpa only [mem_preimage, apply_symm_apply] using hy
  /-
    🎉 no goals
  -/


theorem comp_right_differentiable_iff {f : F → G} :
    Differentiable 𝕜 (f ∘ iso) ↔ Differentiable 𝕜 f := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    ⊢ Iff (Differentiable 𝕜 (Function.comp f ⇑iso)) (Differentiable 𝕜 f)
  -/
  simp only [← differentiableOn_univ, ← iso.comp_right_differentiableOn_iff, preimage_univ]
  /-
    🎉 no goals
  -/


theorem comp_right_hasFDerivWithinAt_iff {f : F → G} {s : Set F} {x : E} {f' : F →L[𝕜] G} :
    HasFDerivWithinAt (f ∘ iso) (f'.comp (iso : E →L[𝕜] F)) (iso ⁻¹' s) x ↔
      HasFDerivWithinAt f f' s (iso x) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    s : Set F
    x : E
    f' : ContinuousLinearMap (RingHom.id 𝕜) F G
    ⊢ Iff (HasFDerivWithinAt (Function.comp f ⇑iso) (f'.comp ↑iso) (Set.preimage ( …
  -/
  refine ⟨fun H => ?_, fun H => H.comp x iso.hasFDerivWithinAt (mapsTo_preimage _ s)⟩
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    s : Set F
    x : E
    f' : ContinuousLinearMap (RingHom.id 𝕜) F G
    H : HasFDerivWithinAt (Function.comp f ⇑iso) (f'.comp ↑iso) (Set.preimage (⇑is …
    ⊢ HasFDerivWithinAt f f' s (iso x)
  -/
  rw [← iso.symm_apply_apply x] at H
  have A : f = (f ∘ iso) ∘ iso.symm := by
    rw [Function.comp_assoc, iso.self_comp_symm]
    rfl
  have B : f' = (f'.comp (iso : E →L[𝕜] F)).comp (iso.symm : F →L[𝕜] E) := by
    rw [ContinuousLinearMap.comp_assoc, iso.coe_comp_coe_symm, ContinuousLinearMap.comp_id]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    s : Set F
    x : E
    f' : ContinuousLinearMap (RingHom.id 𝕜) F G
    H : HasFDerivWithinAt (Function.comp f ⇑iso) (f'.comp ↑iso) (Set.preimage (⇑is …
    A : Eq f (Function.comp (Function.comp f ⇑iso) ⇑iso.symm)
    B : Eq f' ((f'.comp ↑iso).comp ↑iso.symm)
    ⊢ HasFDerivWithinAt f f' s (iso x)
  -/
  rw [A, B]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    s : Set F
    x : E
    f' : ContinuousLinearMap (RingHom.id 𝕜) F G
    H : HasFDerivWithinAt (Function.comp f ⇑iso) (f'.comp ↑iso) (Set.preimage (⇑is …
    A : Eq f (Function.comp (Function.comp f ⇑iso) ⇑iso.symm)
    B : Eq f' ((f'.comp ↑iso).comp ↑iso.symm)
    ⊢ HasFDerivWithinAt (Function.comp (Function.comp f ⇑iso) ⇑iso.symm) ((f'.comp …
  -/
  apply H.comp (iso x) iso.symm.hasFDerivWithinAt
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    s : Set F
    x : E
    f' : ContinuousLinearMap (RingHom.id 𝕜) F G
    H : HasFDerivWithinAt (Function.comp f ⇑iso) (f'.comp ↑iso) (Set.preimage (⇑is …
    A : Eq f (Function.comp (Function.comp f ⇑iso) ⇑iso.symm)
    B : Eq f' ((f'.comp ↑iso).comp ↑iso.symm)
    ⊢ Set.MapsTo (⇑iso.symm) s (Set.preimage (⇑iso) s)
  -/
  intro y hy
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    s : Set F
    x : E
    f' : ContinuousLinearMap (RingHom.id 𝕜) F G
    H : HasFDerivWithinAt (Function.comp f ⇑iso) (f'.comp ↑iso) (Set.preimage (⇑is …
    A : Eq f (Function.comp (Function.comp f ⇑iso) ⇑iso.symm)
    B : Eq f' ((f'.comp ↑iso).comp ↑iso.symm)
    y : F
    hy : Membership.mem s y
    ⊢ Membership.mem (Set.preimage (⇑iso) s) (iso.symm y)
  -/
  simpa only [mem_preimage, apply_symm_apply] using hy
  /-
    🎉 no goals
  -/


theorem comp_right_hasFDerivAt_iff {f : F → G} {x : E} {f' : F →L[𝕜] G} :
    HasFDerivAt (f ∘ iso) (f'.comp (iso : E →L[𝕜] F)) x ↔ HasFDerivAt f f' (iso x) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    x : E
    f' : ContinuousLinearMap (RingHom.id 𝕜) F G
    ⊢ Iff (HasFDerivAt (Function.comp f ⇑iso) (f'.comp ↑iso) x) (HasFDerivAt f f'  …
  -/
  simp only [← hasFDerivWithinAt_univ, ← comp_right_hasFDerivWithinAt_iff, preimage_univ]
  /-
    🎉 no goals
  -/


theorem comp_right_hasFDerivWithinAt_iff' {f : F → G} {s : Set F} {x : E} {f' : E →L[𝕜] G} :
    HasFDerivWithinAt (f ∘ iso) f' (iso ⁻¹' s) x ↔
      HasFDerivWithinAt f (f'.comp (iso.symm : F →L[𝕜] E)) s (iso x) := by
  rw [← iso.comp_right_hasFDerivWithinAt_iff, ContinuousLinearMap.comp_assoc,
    iso.coe_symm_comp_coe, ContinuousLinearMap.comp_id]


theorem comp_right_hasFDerivAt_iff' {f : F → G} {x : E} {f' : E →L[𝕜] G} :
    HasFDerivAt (f ∘ iso) f' x ↔ HasFDerivAt f (f'.comp (iso.symm : F →L[𝕜] E)) (iso x) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    x : E
    f' : ContinuousLinearMap (RingHom.id 𝕜) E G
    ⊢ Iff (HasFDerivAt (Function.comp f ⇑iso) f' x) (HasFDerivAt f (f'.comp ↑iso.s …
  -/
  simp only [← hasFDerivWithinAt_univ, ← iso.comp_right_hasFDerivWithinAt_iff', preimage_univ]
  /-
    🎉 no goals
  -/


theorem comp_right_fderivWithin {f : F → G} {s : Set F} {x : E}
    (hxs : UniqueDiffWithinAt 𝕜 (iso ⁻¹' s) x) :
    fderivWithin 𝕜 (f ∘ iso) (iso ⁻¹' s) x =
      (fderivWithin 𝕜 f s (iso x)).comp (iso : E →L[𝕜] F) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    s : Set F
    x : E
    hxs : UniqueDiffWithinAt 𝕜 (Set.preimage (⇑iso) s) x
    ⊢ Eq (fderivWithin 𝕜 (Function.comp f ⇑iso) (Set.preimage (⇑iso) s) x) ((fderi …
  -/
  by_cases h : DifferentiableWithinAt 𝕜 f s (iso x)
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      f : F → G
      s : Set F
      x : E
      hxs : UniqueDiffWithinAt 𝕜 (Set.preimage (⇑iso) s) x
      h : DifferentiableWithinAt 𝕜 f s (iso x)
      ⊢ Eq (fderivWithin 𝕜 (Function.comp f ⇑iso) (Set.preimage (⇑iso) s) x) ((fderi …
    -/
  · exact (iso.comp_right_hasFDerivWithinAt_iff.2 h.hasFDerivWithinAt).fderivWithin hxs
    /-
      🎉 no goals
    -/
  · have : ¬DifferentiableWithinAt 𝕜 (f ∘ iso) (iso ⁻¹' s) x := by
      intro h'
      exact h (iso.comp_right_differentiableWithinAt_iff.1 h')
    rw [fderivWithin_zero_of_not_differentiableWithinAt h,
      fderivWithin_zero_of_not_differentiableWithinAt this, ContinuousLinearMap.zero_comp]


theorem comp_right_fderiv {f : F → G} {x : E} :
    fderiv 𝕜 (f ∘ iso) x = (fderiv 𝕜 f (iso x)).comp (iso : E →L[𝕜] F) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    x : E
    ⊢ Eq (fderiv 𝕜 (Function.comp f ⇑iso) x) ((fderiv 𝕜 f (iso x)).comp ↑iso)
  -/
  rw [← fderivWithin_univ, ← fderivWithin_univ, ← iso.comp_right_fderivWithin, preimage_univ]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    f : F → G
    x : E
    ⊢ UniqueDiffWithinAt 𝕜 (Set.preimage (⇑iso) Set.univ) x
  -/
  exact uniqueDiffWithinAt_univ
  /-
    🎉 no goals
  -/


@[fun_prop]
protected theorem hasStrictFDerivAt : HasStrictFDerivAt iso (iso : E →L[𝕜] F) x :=
  (iso : E ≃L[𝕜] F).hasStrictFDerivAt


@[fun_prop]
protected theorem hasFDerivWithinAt : HasFDerivWithinAt iso (iso : E →L[𝕜] F) s x :=
  (iso : E ≃L[𝕜] F).hasFDerivWithinAt


@[fun_prop]
protected theorem hasFDerivAt : HasFDerivAt iso (iso : E →L[𝕜] F) x :=
  (iso : E ≃L[𝕜] F).hasFDerivAt


protected theorem fderivWithin (hxs : UniqueDiffWithinAt 𝕜 s x) : fderivWithin 𝕜 iso s x = iso :=
  (iso : E ≃L[𝕜] F).fderivWithin hxs


theorem comp_differentiableWithinAt_iff {f : G → E} {s : Set G} {x : G} :
    DifferentiableWithinAt 𝕜 (iso ∘ f) s x ↔ DifferentiableWithinAt 𝕜 f s x :=
  (iso : E ≃L[𝕜] F).comp_differentiableWithinAt_iff


theorem comp_differentiableAt_iff {f : G → E} {x : G} :
    DifferentiableAt 𝕜 (iso ∘ f) x ↔ DifferentiableAt 𝕜 f x :=
  (iso : E ≃L[𝕜] F).comp_differentiableAt_iff


theorem comp_differentiableOn_iff {f : G → E} {s : Set G} :
    DifferentiableOn 𝕜 (iso ∘ f) s ↔ DifferentiableOn 𝕜 f s :=
  (iso : E ≃L[𝕜] F).comp_differentiableOn_iff


theorem comp_differentiable_iff {f : G → E} : Differentiable 𝕜 (iso ∘ f) ↔ Differentiable 𝕜 f :=
  (iso : E ≃L[𝕜] F).comp_differentiable_iff


theorem comp_hasFDerivWithinAt_iff {f : G → E} {s : Set G} {x : G} {f' : G →L[𝕜] E} :
    HasFDerivWithinAt (iso ∘ f) ((iso : E →L[𝕜] F).comp f') s x ↔ HasFDerivWithinAt f f' s x :=
  (iso : E ≃L[𝕜] F).comp_hasFDerivWithinAt_iff


theorem comp_hasStrictFDerivAt_iff {f : G → E} {x : G} {f' : G →L[𝕜] E} :
    HasStrictFDerivAt (iso ∘ f) ((iso : E →L[𝕜] F).comp f') x ↔ HasStrictFDerivAt f f' x :=
  (iso : E ≃L[𝕜] F).comp_hasStrictFDerivAt_iff


theorem comp_hasFDerivAt_iff {f : G → E} {x : G} {f' : G →L[𝕜] E} :
    HasFDerivAt (iso ∘ f) ((iso : E →L[𝕜] F).comp f') x ↔ HasFDerivAt f f' x :=
  (iso : E ≃L[𝕜] F).comp_hasFDerivAt_iff


theorem comp_hasFDerivWithinAt_iff' {f : G → E} {s : Set G} {x : G} {f' : G →L[𝕜] F} :
    HasFDerivWithinAt (iso ∘ f) f' s x ↔ HasFDerivWithinAt f ((iso.symm : F →L[𝕜] E).comp f') s x :=
  (iso : E ≃L[𝕜] F).comp_hasFDerivWithinAt_iff'


theorem comp_hasFDerivAt_iff' {f : G → E} {x : G} {f' : G →L[𝕜] F} :
    HasFDerivAt (iso ∘ f) f' x ↔ HasFDerivAt f ((iso.symm : F →L[𝕜] E).comp f') x :=
  (iso : E ≃L[𝕜] F).comp_hasFDerivAt_iff'


theorem comp_fderivWithin {f : G → E} {s : Set G} {x : G} (hxs : UniqueDiffWithinAt 𝕜 s x) :
    fderivWithin 𝕜 (iso ∘ f) s x = (iso : E →L[𝕜] F).comp (fderivWithin 𝕜 f s x) :=
  (iso : E ≃L[𝕜] F).comp_fderivWithin hxs


theorem comp_fderiv {f : G → E} {x : G} :
    fderiv 𝕜 (iso ∘ f) x = (iso : E →L[𝕜] F).comp (fderiv 𝕜 f x) :=
  (iso : E ≃L[𝕜] F).comp_fderiv


theorem comp_fderiv' {f : G → E} :
    fderiv 𝕜 (iso ∘ f) = fun x ↦ (iso : E →L[𝕜] F).comp (fderiv 𝕜 f x) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : LinearIsometryEquiv (RingHom.id 𝕜) E F
    f : G → E
    ⊢ Eq (fderiv 𝕜 (Function.comp (⇑iso) f)) fun x => (↑{ toLinearEquiv := iso.toL …
  -/
  ext x : 1
  /-
    case h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    iso : LinearIsometryEquiv (RingHom.id 𝕜) E F
    f : G → E
    x : G
    ⊢ Eq (fderiv 𝕜 (Function.comp (⇑iso) f) x) ((↑{ toLinearEquiv := iso.toLinearE …
  -/
  exact LinearIsometryEquiv.comp_fderiv iso
  /-
    🎉 no goals
  -/


/-- If `f (g y) = y` for `y` in some neighborhood of `a`, `g` is continuous at `a`, and `f` has an
invertible derivative `f'` at `g a` in the strict sense, then `g` has the derivative `f'⁻¹` at `a`
in the strict sense.

This is one of the easy parts of the inverse function theorem: it assumes that we already have an
inverse function. -/
theorem HasStrictFDerivAt.of_local_left_inverse {f : E → F} {f' : E ≃L[𝕜] F} {g : F → E} {a : F}
    (hg : ContinuousAt g a) (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) (g a))
    (hfg : ∀ᶠ y in 𝓝 a, f (g y) = y) : HasStrictFDerivAt g (f'.symm : F →L[𝕜] E) a := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    g : F → E
    a : F
    hg : ContinuousAt g a
    hf : HasStrictFDerivAt f (↑f') (g a)
    hfg : Filter.Eventually (fun y => Eq (f (g y)) y) (nhds a)
    ⊢ HasStrictFDerivAt g (↑f'.symm) a
  -/
  replace hg := hg.prodMap' hg
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    g : F → E
    a : F
    hf : HasStrictFDerivAt f (↑f') (g a)
    hfg : Filter.Eventually (fun y => Eq (f (g y)) y) (nhds a)
    hg : ContinuousAt (Prod.map g g) { fst := a, snd := a }
    ⊢ HasStrictFDerivAt g (↑f'.symm) a
  -/
  replace hfg := hfg.prod_mk_nhds hfg
  have :
    (fun p : F × F => g p.1 - g p.2 - f'.symm (p.1 - p.2)) =O[𝓝 (a, a)] fun p : F × F =>
      f' (g p.1 - g p.2) - (p.1 - p.2) := by
    refine ((f'.symm : F →L[𝕜] E).isBigO_comp _ _).congr (fun x => ?_) fun _ => rfl
    simp
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    g : F → E
    a : F
    hf : HasStrictFDerivAt f (↑f') (g a)
    hg : ContinuousAt (Prod.map g g) { fst := a, snd := a }
    hfg : Filter.Eventually (fun p => And (Eq (f (g p.1)) p.1) (Eq (f (g p.2)) p.2 …
    this : Asymptotics.IsBigO (nhds { fst := a, snd := a }) (fun p => HSub.hSub (H …
    ⊢ HasStrictFDerivAt g (↑f'.symm) a
  -/
  refine .of_isLittleO <| this.trans_isLittleO ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    g : F → E
    a : F
    hf : HasStrictFDerivAt f (↑f') (g a)
    hg : ContinuousAt (Prod.map g g) { fst := a, snd := a }
    hfg : Filter.Eventually (fun p => And (Eq (f (g p.1)) p.1) (Eq (f (g p.2)) p.2 …
    this : Asymptotics.IsBigO (nhds { fst := a, snd := a }) (fun p => HSub.hSub (H …
    ⊢ Asymptotics.IsLittleO (nhds { fst := a, snd := a }) (fun p => HSub.hSub (f'  …
  -/
  clear this
  refine ((hf.isLittleO.comp_tendsto hg).symm.congr'
    (hfg.mono ?_) (Eventually.of_forall fun _ => rfl)).trans_isBigO ?_
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      g : F → E
      a : F
      hf : HasStrictFDerivAt f (↑f') (g a)
      hg : ContinuousAt (Prod.map g g) { fst := a, snd := a }
      hfg : Filter.Eventually (fun p => And (Eq (f (g p.1)) p.1) (Eq (f (g p.2)) p.2 …
      ⊢ ∀ (x : Prod F F), And (Eq (f (g x.1)) x.1) (Eq (f (g x.2)) x.2) → Eq ((fun x …
    -/
  · rintro p ⟨hp1, hp2⟩
    /-
      case refine_1.intro
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      g : F → E
      a : F
      hf : HasStrictFDerivAt f (↑f') (g a)
      hg : ContinuousAt (Prod.map g g) { fst := a, snd := a }
      hfg : Filter.Eventually (fun p => And (Eq (f (g p.1)) p.1) (Eq (f (g p.2)) p.2 …
      p : Prod F F
      hp1 : Eq (f (g p.1)) p.1
      hp2 : Eq (f (g p.2)) p.2
      ⊢ Eq ((fun x => HSub.hSub (↑f' (HSub.hSub (Prod.map g g x).1 (Prod.map g g x). …
    -/
    simp [hp1, hp2]
    /-
      🎉 no goals
    -/
  · refine (hf.isBigO_sub_rev.comp_tendsto hg).congr' (Eventually.of_forall fun _ => rfl)
      (hfg.mono ?_)
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      g : F → E
      a : F
      hf : HasStrictFDerivAt f (↑f') (g a)
      hg : ContinuousAt (Prod.map g g) { fst := a, snd := a }
      hfg : Filter.Eventually (fun p => And (Eq (f (g p.1)) p.1) (Eq (f (g p.2)) p.2 …
      ⊢ ∀ (x : Prod F F), And (Eq (f (g x.1)) x.1) (Eq (f (g x.2)) x.2) → Eq (Functi …
    -/
    rintro p ⟨hp1, hp2⟩
    /-
      case refine_2.intro
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      g : F → E
      a : F
      hf : HasStrictFDerivAt f (↑f') (g a)
      hg : ContinuousAt (Prod.map g g) { fst := a, snd := a }
      hfg : Filter.Eventually (fun p => And (Eq (f (g p.1)) p.1) (Eq (f (g p.2)) p.2 …
      p : Prod F F
      hp1 : Eq (f (g p.1)) p.1
      hp2 : Eq (f (g p.2)) p.2
      ⊢ Eq (Function.comp (fun p => HSub.hSub (f p.1) (f p.2)) (Prod.map g g) p) ((f …
    -/
    simp only [(· ∘ ·), hp1, hp2, Prod.map]
    /-
      🎉 no goals
    -/


/-- If `f (g y) = y` for `y` in some neighborhood of `a`, `g` is continuous at `a`, and `f` has an
invertible derivative `f'` at `g a`, then `g` has the derivative `f'⁻¹` at `a`.

This is one of the easy parts of the inverse function theorem: it assumes that we already have
an inverse function. -/
theorem HasFDerivAt.of_local_left_inverse {f : E → F} {f' : E ≃L[𝕜] F} {g : F → E} {a : F}
    (hg : ContinuousAt g a) (hf : HasFDerivAt f (f' : E →L[𝕜] F) (g a))
    (hfg : ∀ᶠ y in 𝓝 a, f (g y) = y) : HasFDerivAt g (f'.symm : F →L[𝕜] E) a := by
  have : (fun x : F => g x - g a - f'.symm (x - a)) =O[𝓝 a]
      fun x : F => f' (g x - g a) - (x - a) := by
    refine ((f'.symm : F →L[𝕜] E).isBigO_comp _ _).congr (fun x => ?_) fun _ => rfl
    simp
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    g : F → E
    a : F
    hg : ContinuousAt g a
    hf : HasFDerivAt f (↑f') (g a)
    hfg : Filter.Eventually (fun y => Eq (f (g y)) y) (nhds a)
    this : Asymptotics.IsBigO (nhds a) (fun x => HSub.hSub (HSub.hSub (g x) (g a)) …
    ⊢ HasFDerivAt g (↑f'.symm) a
  -/
  refine HasFDerivAtFilter.of_isLittleO <| this.trans_isLittleO ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    g : F → E
    a : F
    hg : ContinuousAt g a
    hf : HasFDerivAt f (↑f') (g a)
    hfg : Filter.Eventually (fun y => Eq (f (g y)) y) (nhds a)
    this : Asymptotics.IsBigO (nhds a) (fun x => HSub.hSub (HSub.hSub (g x) (g a)) …
    ⊢ Asymptotics.IsLittleO (nhds a) (fun x => HSub.hSub (f' (HSub.hSub (g x) (g a …
  -/
  clear this
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    g : F → E
    a : F
    hg : ContinuousAt g a
    hf : HasFDerivAt f (↑f') (g a)
    hfg : Filter.Eventually (fun y => Eq (f (g y)) y) (nhds a)
    ⊢ Asymptotics.IsLittleO (nhds a) (fun x => HSub.hSub (f' (HSub.hSub (g x) (g a …
  -/
  refine ((hf.isLittleO.comp_tendsto hg).symm.congr' (hfg.mono ?_) .rfl).trans_isBigO ?_
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      g : F → E
      a : F
      hg : ContinuousAt g a
      hf : HasFDerivAt f (↑f') (g a)
      hfg : Filter.Eventually (fun y => Eq (f (g y)) y) (nhds a)
      ⊢ ∀ (x : F), Eq (f (g x)) x → Eq ((fun x => HSub.hSub (↑f' (HSub.hSub (g x) (g …
    -/
  · intro p hp
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      g : F → E
      a : F
      hg : ContinuousAt g a
      hf : HasFDerivAt f (↑f') (g a)
      hfg : Filter.Eventually (fun y => Eq (f (g y)) y) (nhds a)
      p : F
      hp : Eq (f (g p)) p
      ⊢ Eq ((fun x => HSub.hSub (↑f' (HSub.hSub (g x) (g a))) (HSub.hSub (f (g x)) ( …
    -/
    simp [hp, hfg.self_of_nhds]
    /-
      🎉 no goals
    -/
  · refine ((hf.isBigO_sub_rev f'.antilipschitz).comp_tendsto hg).congr'
      (Eventually.of_forall fun _ => rfl) (hfg.mono ?_)
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      g : F → E
      a : F
      hg : ContinuousAt g a
      hf : HasFDerivAt f (↑f') (g a)
      hfg : Filter.Eventually (fun y => Eq (f (g y)) y) (nhds a)
      ⊢ ∀ (x : F), Eq (f (g x)) x → Eq (Function.comp (fun x' => HSub.hSub (f x') (f …
    -/
    rintro p hp
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      g : F → E
      a : F
      hg : ContinuousAt g a
      hf : HasFDerivAt f (↑f') (g a)
      hfg : Filter.Eventually (fun y => Eq (f (g y)) y) (nhds a)
      p : F
      hp : Eq (f (g p)) p
      ⊢ Eq (Function.comp (fun x' => HSub.hSub (f x') (f (g a))) g p) ((fun x' => HS …
    -/
    simp only [(· ∘ ·), hp, hfg.self_of_nhds]
    /-
      🎉 no goals
    -/


/-- If `f` is a partial homeomorphism defined on a neighbourhood of `f.symm a`, and `f` has an
invertible derivative `f'` in the sense of strict differentiability at `f.symm a`, then `f.symm` has
the derivative `f'⁻¹` at `a`.

This is one of the easy parts of the inverse function theorem: it assumes that we already have
an inverse function. -/
theorem PartialHomeomorph.hasStrictFDerivAt_symm (f : PartialHomeomorph E F) {f' : E ≃L[𝕜] F}
    {a : F} (ha : a ∈ f.target) (htff' : HasStrictFDerivAt f (f' : E →L[𝕜] F) (f.symm a)) :
    HasStrictFDerivAt f.symm (f'.symm : F →L[𝕜] E) a :=
  htff'.of_local_left_inverse (f.symm.continuousAt ha) (f.eventually_right_inverse ha)


/-- If `f` is a partial homeomorphism defined on a neighbourhood of `f.symm a`, and `f` has an
invertible derivative `f'` at `f.symm a`, then `f.symm` has the derivative `f'⁻¹` at `a`.

This is one of the easy parts of the inverse function theorem: it assumes that we already have
an inverse function. -/
theorem PartialHomeomorph.hasFDerivAt_symm (f : PartialHomeomorph E F) {f' : E ≃L[𝕜] F} {a : F}
    (ha : a ∈ f.target) (htff' : HasFDerivAt f (f' : E →L[𝕜] F) (f.symm a)) :
    HasFDerivAt f.symm (f'.symm : F →L[𝕜] E) a :=
  htff'.of_local_left_inverse (f.symm.continuousAt ha) (f.eventually_right_inverse ha)


theorem HasFDerivWithinAt.eventually_ne (h : HasFDerivWithinAt f f' s x)
    (hf' : ∃ C, ∀ z, ‖z‖ ≤ C * ‖f' z‖) : ∀ᶠ z in 𝓝[s \ {x}] x, f z ≠ f x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s : Set E
    h : HasFDerivWithinAt f f' s x
    hf' : Exists fun C => ∀ (z : E), LE.le (Norm.norm z) (HMul.hMul C (Norm.norm ( …
    ⊢ Filter.Eventually (fun z => Ne (f z) (f x)) (nhdsWithin x (SDiff.sdiff s (Si …
  -/
  rw [nhdsWithin, diff_eq, ← inf_principal, ← inf_assoc, eventually_inf_principal]
  have A : (fun z => z - x) =O[𝓝[s] x] fun z => f' (z - x) :=
    isBigO_iff.2 <| hf'.imp fun C hC => Eventually.of_forall fun z => hC _
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s : Set E
    h : HasFDerivWithinAt f f' s x
    hf' : Exists fun C => ∀ (z : E), LE.le (Norm.norm z) (HMul.hMul C (Norm.norm ( …
    A : Asymptotics.IsBigO (nhdsWithin x s) (fun z => HSub.hSub z x) fun z => f' ( …
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (HasCompl.compl (Singleton.sing …
  -/
  have : (fun z => f z - f x) ~[𝓝[s] x] fun z => f' (z - x) := h.isLittleO.trans_isBigO A
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s : Set E
    h : HasFDerivWithinAt f f' s x
    hf' : Exists fun C => ∀ (z : E), LE.le (Norm.norm z) (HMul.hMul C (Norm.norm ( …
    A : Asymptotics.IsBigO (nhdsWithin x s) (fun z => HSub.hSub z x) fun z => f' ( …
    this : Asymptotics.IsEquivalent (nhdsWithin x s) (fun z => HSub.hSub (f z) (f  …
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (HasCompl.compl (Singleton.sing …
  -/
  simpa [not_imp_not, sub_eq_zero] using (A.trans this.isBigO_symm).eq_zero_imp
  /-
    🎉 no goals
  -/


theorem HasFDerivAt.eventually_ne (h : HasFDerivAt f f' x) (hf' : ∃ C, ∀ z, ‖z‖ ≤ C * ‖f' z‖) :
    ∀ᶠ z in 𝓝[≠] x, f z ≠ f x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    h : HasFDerivAt f f' x
    hf' : Exists fun C => ∀ (z : E), LE.le (Norm.norm z) (HMul.hMul C (Norm.norm ( …
    ⊢ Filter.Eventually (fun z => Ne (f z) (f x)) (nhdsWithin x (HasCompl.compl (S …
  -/
  simpa only [compl_eq_univ_diff] using (hasFDerivWithinAt_univ.2 h).eventually_ne hf'
  /-
    🎉 no goals
  -/


theorem has_fderiv_at_filter_real_equiv {L : Filter E} :
    Tendsto (fun x' : E => ‖x' - x‖⁻¹ * ‖f x' - f x - f' (x' - x)‖) L (𝓝 0) ↔
      Tendsto (fun x' : E => ‖x' - x‖⁻¹ • (f x' - f x - f' (x' - x))) L (𝓝 0) := by
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id Real) E F
    x : E
    L : Filter E
    ⊢ Iff (Filter.Tendsto (fun x' => HMul.hMul (Inv.inv (Norm.norm (HSub.hSub x' x …
  -/
  symm
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id Real) E F
    x : E
    L : Filter E
    ⊢ Iff (Filter.Tendsto (fun x' => HSMul.hSMul (Inv.inv (Norm.norm (HSub.hSub x' …
  -/
  rw [tendsto_iff_norm_sub_tendsto_zero]
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id Real) E F
    x : E
    L : Filter E
    ⊢ Iff (Filter.Tendsto (fun e => Norm.norm (HSub.hSub (HSMul.hSMul (Inv.inv (No …
  -/
  refine tendsto_congr fun x' => ?_
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id Real) E F
    x : E
    L : Filter E
    x' : E
    ⊢ Eq (Norm.norm (HSub.hSub (HSMul.hSMul (Inv.inv (Norm.norm (HSub.hSub x' x))) …
  -/
  simp [norm_smul]
  /-
    🎉 no goals
  -/


theorem HasFDerivAt.lim_real (hf : HasFDerivAt f f' x) (v : E) :
    Tendsto (fun c : ℝ => c • (f (x + c⁻¹ • v) - f x)) atTop (𝓝 (f' v)) := by
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id Real) E F
    x : E
    hf : HasFDerivAt f f' x
    v : E
    ⊢ Filter.Tendsto (fun c => HSMul.hSMul c (HSub.hSub (f (HAdd.hAdd x (HSMul.hSM …
  -/
  apply hf.lim v
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id Real) E F
    x : E
    hf : HasFDerivAt f f' x
    v : E
    ⊢ Filter.Tendsto (fun n => Norm.norm n) Filter.atTop Filter.atTop
  -/
  rw [tendsto_atTop_atTop]
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id Real) E F
    x : E
    hf : HasFDerivAt f f' x
    v : E
    ⊢ ∀ (b : Real), Exists fun i => ∀ (a : Real), LE.le i a → LE.le b (Norm.norm a)
  -/
  exact fun b => ⟨b, fun a ha => le_trans ha (le_abs_self _)⟩
  /-
    🎉 no goals
  -/


/-- The image of a tangent cone under the differential of a map is included in the tangent cone to
the image. -/
theorem HasFDerivWithinAt.mapsTo_tangent_cone {x : E} (h : HasFDerivWithinAt f f' s x) :
    MapsTo f' (tangentConeAt 𝕜 s x) (tangentConeAt 𝕜 (f '' s) (f x)) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    h : HasFDerivWithinAt f f' s x
    ⊢ Set.MapsTo (⇑f') (tangentConeAt 𝕜 s x) (tangentConeAt 𝕜 (Set.image f s) (f x))
  -/
  rintro v ⟨c, d, dtop, clim, cdlim⟩
  refine
    ⟨c, fun n => f (x + d n) - f x, mem_of_superset dtop ?_, clim, h.lim atTop dtop clim cdlim⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    h : HasFDerivWithinAt f f' s x
    v : E
    c : Nat → 𝕜
    d : Nat → E
    dtop : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (d n))) Filte …
    clim : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
    cdlim : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds v)
    ⊢ HasSubset.Subset (setOf fun x_1 => (fun n => Membership.mem s (HAdd.hAdd x ( …
  -/
  simp +contextual [-mem_image, mem_image_of_mem]
  /-
    🎉 no goals
  -/


/-- If a set has the unique differentiability property at a point x, then the image of this set
under a map with onto derivative has also the unique differentiability property at the image point.
-/
theorem HasFDerivWithinAt.uniqueDiffWithinAt {x : E} (h : HasFDerivWithinAt f f' s x)
    (hs : UniqueDiffWithinAt 𝕜 s x) (h' : DenseRange f') : UniqueDiffWithinAt 𝕜 (f '' s) (f x) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    h : HasFDerivWithinAt f f' s x
    hs : UniqueDiffWithinAt 𝕜 s x
    h' : DenseRange ⇑f'
    ⊢ UniqueDiffWithinAt 𝕜 (Set.image f s) (f x)
  -/
  refine ⟨h'.dense_of_mapsTo f'.continuous hs.1 ?_, h.continuousWithinAt.mem_closure_image hs.2⟩
  show
    Submodule.span 𝕜 (tangentConeAt 𝕜 s x) ≤
      (Submodule.span 𝕜 (tangentConeAt 𝕜 (f '' s) (f x))).comap f'
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    h : HasFDerivWithinAt f f' s x
    hs : UniqueDiffWithinAt 𝕜 s x
    h' : DenseRange ⇑f'
    ⊢ LE.le (Submodule.span 𝕜 (tangentConeAt 𝕜 s x)) (Submodule.comap f' (Submodul …
  -/
  rw [Submodule.span_le]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    h : HasFDerivWithinAt f f' s x
    hs : UniqueDiffWithinAt 𝕜 s x
    h' : DenseRange ⇑f'
    ⊢ HasSubset.Subset (tangentConeAt 𝕜 s x) ↑(Submodule.comap f' (Submodule.span  …
  -/
  exact h.mapsTo_tangent_cone.mono Subset.rfl Submodule.subset_span
  /-
    🎉 no goals
  -/


theorem UniqueDiffOn.image {f' : E → E →L[𝕜] F} (hs : UniqueDiffOn 𝕜 s)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (hd : ∀ x ∈ s, DenseRange (f' x)) :
    UniqueDiffOn 𝕜 (f '' s) :=
  forall_mem_image.2 fun x hx => (hf' x hx).uniqueDiffWithinAt (hs x hx) (hd x hx)


theorem HasFDerivWithinAt.uniqueDiffWithinAt_of_continuousLinearEquiv {x : E} (e' : E ≃L[𝕜] F)
    (h : HasFDerivWithinAt f (e' : E →L[𝕜] F) s x) (hs : UniqueDiffWithinAt 𝕜 s x) :
    UniqueDiffWithinAt 𝕜 (f '' s) (f x) :=
  h.uniqueDiffWithinAt hs e'.surjective.denseRange


theorem ContinuousLinearEquiv.uniqueDiffOn_image (e : E ≃L[𝕜] F) (h : UniqueDiffOn 𝕜 s) :
    UniqueDiffOn 𝕜 (e '' s) :=
  h.image (fun _ _ => e.hasFDerivWithinAt) fun _ _ => e.surjective.denseRange


@[simp]
theorem ContinuousLinearEquiv.uniqueDiffOn_image_iff (e : E ≃L[𝕜] F) :
    UniqueDiffOn 𝕜 (e '' s) ↔ UniqueDiffOn 𝕜 s :=
  ⟨fun h => e.symm_image_image s ▸ e.symm.uniqueDiffOn_image h, e.uniqueDiffOn_image⟩


@[simp]
theorem ContinuousLinearEquiv.uniqueDiffOn_preimage_iff (e : F ≃L[𝕜] E) :
    UniqueDiffOn 𝕜 (e ⁻¹' s) ↔ UniqueDiffOn 𝕜 s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    e : ContinuousLinearEquiv (RingHom.id 𝕜) F E
    ⊢ Iff (UniqueDiffOn 𝕜 (Set.preimage (⇑e) s)) (UniqueDiffOn 𝕜 s)
  -/
  rw [← e.image_symm_eq_preimage, e.symm.uniqueDiffOn_image_iff]
  /-
    🎉 no goals
  -/


