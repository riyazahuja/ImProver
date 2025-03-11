@[fun_prop]
protected theorem ContinuousLinearMap.hasStrictFDerivAt {x : E} : HasStrictFDerivAt e e x :=
                                                               /-
                                                                 𝕜 : Type u_1
                                                                 inst✝⁴ : NontriviallyNormedField 𝕜
                                                                 E : Type u_2
                                                                 inst✝³ : NormedAddCommGroup E
                                                                 inst✝² : NormedSpace 𝕜 E
                                                                 F : Type u_3
                                                                 inst✝¹ : NormedAddCommGroup F
                                                                 inst✝ : NormedSpace 𝕜 F
                                                                 e : ContinuousLinearMap (RingHom.id 𝕜) E F
                                                                 x✝ : E
                                                                 x : Prod E E
                                                                 ⊢ Eq 0 (HSub.hSub (HSub.hSub (e x.1) (e x.2)) (e (HSub.hSub x.1 x.2)))
                                                               -/
  .of_isLittleO <| (isLittleO_zero _ _).congr_left fun x => by simp only [e.map_sub, sub_self]
                                                               /-
                                                                 🎉 no goals
                                                               -/


protected theorem ContinuousLinearMap.hasFDerivAtFilter : HasFDerivAtFilter e e x L :=
                                                               /-
                                                                 𝕜 : Type u_1
                                                                 inst✝⁴ : NontriviallyNormedField 𝕜
                                                                 E : Type u_2
                                                                 inst✝³ : NormedAddCommGroup E
                                                                 inst✝² : NormedSpace 𝕜 E
                                                                 F : Type u_3
                                                                 inst✝¹ : NormedAddCommGroup F
                                                                 inst✝ : NormedSpace 𝕜 F
                                                                 e : ContinuousLinearMap (RingHom.id 𝕜) E F
                                                                 x✝ : E
                                                                 L : Filter E
                                                                 x : E
                                                                 ⊢ Eq 0 (HSub.hSub (HSub.hSub (e x) (e x✝)) (e (HSub.hSub x x✝)))
                                                               -/
  .of_isLittleO <| (isLittleO_zero _ _).congr_left fun x => by simp only [e.map_sub, sub_self]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[fun_prop]
protected theorem ContinuousLinearMap.hasFDerivWithinAt : HasFDerivWithinAt e e s x :=
  e.hasFDerivAtFilter


@[fun_prop]
protected theorem ContinuousLinearMap.hasFDerivAt : HasFDerivAt e e x :=
  e.hasFDerivAtFilter


@[simp, fun_prop]
protected theorem ContinuousLinearMap.differentiableAt : DifferentiableAt 𝕜 e x :=
  e.hasFDerivAt.differentiableAt


@[fun_prop]
protected theorem ContinuousLinearMap.differentiableWithinAt : DifferentiableWithinAt 𝕜 e s x :=
  e.differentiableAt.differentiableWithinAt


@[simp]
protected theorem ContinuousLinearMap.fderiv : fderiv 𝕜 e x = e :=
  e.hasFDerivAt.fderiv


protected theorem ContinuousLinearMap.fderivWithin (hxs : UniqueDiffWithinAt 𝕜 s x) :
    fderivWithin 𝕜 e s x = e := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    e : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s : Set E
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (fderivWithin 𝕜 (⇑e) s x) e
  -/
  rw [DifferentiableAt.fderivWithin e.differentiableAt hxs]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    e : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s : Set E
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (fderiv 𝕜 (⇑e) x) e
  -/
  exact e.fderiv
  /-
    🎉 no goals
  -/


@[simp, fun_prop]
protected theorem ContinuousLinearMap.differentiable : Differentiable 𝕜 e := fun _ =>
  e.differentiableAt


@[fun_prop]
protected theorem ContinuousLinearMap.differentiableOn : DifferentiableOn 𝕜 e s :=
  e.differentiable.differentiableOn


theorem IsBoundedLinearMap.hasFDerivAtFilter (h : IsBoundedLinearMap 𝕜 f) :
    HasFDerivAtFilter f h.toContinuousLinearMap x L :=
  h.toContinuousLinearMap.hasFDerivAtFilter


@[fun_prop]
theorem IsBoundedLinearMap.hasFDerivWithinAt (h : IsBoundedLinearMap 𝕜 f) :
    HasFDerivWithinAt f h.toContinuousLinearMap s x :=
  h.hasFDerivAtFilter


@[fun_prop]
theorem IsBoundedLinearMap.hasFDerivAt (h : IsBoundedLinearMap 𝕜 f) :
    HasFDerivAt f h.toContinuousLinearMap x :=
  h.hasFDerivAtFilter


@[fun_prop]
theorem IsBoundedLinearMap.differentiableAt (h : IsBoundedLinearMap 𝕜 f) : DifferentiableAt 𝕜 f x :=
  h.hasFDerivAt.differentiableAt


@[fun_prop]
theorem IsBoundedLinearMap.differentiableWithinAt (h : IsBoundedLinearMap 𝕜 f) :
    DifferentiableWithinAt 𝕜 f s x :=
  h.differentiableAt.differentiableWithinAt


theorem IsBoundedLinearMap.fderiv (h : IsBoundedLinearMap 𝕜 f) :
    fderiv 𝕜 f x = h.toContinuousLinearMap :=
  HasFDerivAt.fderiv h.hasFDerivAt


theorem IsBoundedLinearMap.fderivWithin (h : IsBoundedLinearMap 𝕜 f)
    (hxs : UniqueDiffWithinAt 𝕜 s x) : fderivWithin 𝕜 f s x = h.toContinuousLinearMap := by
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
    x : E
    s : Set E
    h : IsBoundedLinearMap 𝕜 f
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (_root_.fderivWithin 𝕜 f s x) h.toContinuousLinearMap
  -/
  rw [DifferentiableAt.fderivWithin h.differentiableAt hxs]
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
    x : E
    s : Set E
    h : IsBoundedLinearMap 𝕜 f
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (_root_.fderiv 𝕜 f x) h.toContinuousLinearMap
  -/
  exact h.fderiv
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem IsBoundedLinearMap.differentiable (h : IsBoundedLinearMap 𝕜 f) : Differentiable 𝕜 f :=
  fun _ => h.differentiableAt


@[fun_prop]
theorem IsBoundedLinearMap.differentiableOn (h : IsBoundedLinearMap 𝕜 f) : DifferentiableOn 𝕜 f s :=
  h.differentiable.differentiableOn


