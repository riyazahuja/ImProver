@[fun_prop]
theorem HasStrictFDerivAt.clm_comp (hc : HasStrictFDerivAt c c' x) (hd : HasStrictFDerivAt d d' x) :
    HasStrictFDerivAt (fun y => (c y).comp (d y))
      ((compL 𝕜 F G H (c x)).comp d' + ((compL 𝕜 F G H).flip (d x)).comp c') x := by
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
    x : E
    H : Type u_5
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    c : E → ContinuousLinearMap (RingHom.id 𝕜) G H
    c' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    d : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    d' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    hc : HasStrictFDerivAt c c' x
    hd : HasStrictFDerivAt d d' x
    ⊢ HasStrictFDerivAt (fun y => (c y).comp (d y)) (HAdd.hAdd (((ContinuousLinear …
  -/
  have := isBoundedBilinearMap_comp.hasStrictFDerivAt (c x, d x)
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
    x : E
    H : Type u_5
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    c : E → ContinuousLinearMap (RingHom.id 𝕜) G H
    c' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    d : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    d' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    hc : HasStrictFDerivAt c c' x
    hd : HasStrictFDerivAt d d' x
    this : HasStrictFDerivAt (fun p => p.1.comp p.2) (⋯.deriv { fst := c x, snd := …
    ⊢ HasStrictFDerivAt (fun y => (c y).comp (d y)) (HAdd.hAdd (((ContinuousLinear …
  -/
  have := this.comp x (hc.prod hd)
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
    x : E
    H : Type u_5
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    c : E → ContinuousLinearMap (RingHom.id 𝕜) G H
    c' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    d : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    d' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    hc : HasStrictFDerivAt c c' x
    hd : HasStrictFDerivAt d d' x
    this✝ : HasStrictFDerivAt (fun p => p.1.comp p.2) (⋯.deriv { fst := c x, snd : …
    this : HasStrictFDerivAt (fun x => { fst := c x, snd := d x }.1.comp { fst :=  …
    ⊢ HasStrictFDerivAt (fun y => (c y).comp (d y)) (HAdd.hAdd (((ContinuousLinear …
  -/
  exact this
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem HasFDerivWithinAt.clm_comp (hc : HasFDerivWithinAt c c' s x)
    (hd : HasFDerivWithinAt d d' s x) :
    HasFDerivWithinAt (fun y => (c y).comp (d y))
      ((compL 𝕜 F G H (c x)).comp d' + ((compL 𝕜 F G H).flip (d x)).comp c') s x := by
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
    x : E
    s : Set E
    H : Type u_5
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    c : E → ContinuousLinearMap (RingHom.id 𝕜) G H
    c' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    d : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    d' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    hc : HasFDerivWithinAt c c' s x
    hd : HasFDerivWithinAt d d' s x
    ⊢ HasFDerivWithinAt (fun y => (c y).comp (d y)) (HAdd.hAdd (((ContinuousLinear …
  -/
  exact (isBoundedBilinearMap_comp.hasFDerivAt (c x, d x) :).comp_hasFDerivWithinAt x <| hc.prod hd
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem HasFDerivAt.clm_comp (hc : HasFDerivAt c c' x) (hd : HasFDerivAt d d' x) :
    HasFDerivAt (fun y => (c y).comp (d y))
      ((compL 𝕜 F G H (c x)).comp d' + ((compL 𝕜 F G H).flip (d x)).comp c') x := by
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
    x : E
    H : Type u_5
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    c : E → ContinuousLinearMap (RingHom.id 𝕜) G H
    c' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    d : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    d' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    hc : HasFDerivAt c c' x
    hd : HasFDerivAt d d' x
    ⊢ HasFDerivAt (fun y => (c y).comp (d y)) (HAdd.hAdd (((ContinuousLinearMap.co …
  -/
  exact (isBoundedBilinearMap_comp.hasFDerivAt (c x, d x) :).comp x <| hc.prod hd
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem DifferentiableWithinAt.clm_comp (hc : DifferentiableWithinAt 𝕜 c s x)
    (hd : DifferentiableWithinAt 𝕜 d s x) :
    DifferentiableWithinAt 𝕜 (fun y => (c y).comp (d y)) s x :=
  (hc.hasFDerivWithinAt.clm_comp hd.hasFDerivWithinAt).differentiableWithinAt


@[fun_prop]
theorem DifferentiableAt.clm_comp (hc : DifferentiableAt 𝕜 c x) (hd : DifferentiableAt 𝕜 d x) :
    DifferentiableAt 𝕜 (fun y => (c y).comp (d y)) x :=
  (hc.hasFDerivAt.clm_comp hd.hasFDerivAt).differentiableAt


@[fun_prop]
theorem DifferentiableOn.clm_comp (hc : DifferentiableOn 𝕜 c s) (hd : DifferentiableOn 𝕜 d s) :
    DifferentiableOn 𝕜 (fun y => (c y).comp (d y)) s := fun x hx => (hc x hx).clm_comp (hd x hx)


@[fun_prop]
theorem Differentiable.clm_comp (hc : Differentiable 𝕜 c) (hd : Differentiable 𝕜 d) :
    Differentiable 𝕜 fun y => (c y).comp (d y) := fun x => (hc x).clm_comp (hd x)


theorem fderivWithin_clm_comp (hxs : UniqueDiffWithinAt 𝕜 s x) (hc : DifferentiableWithinAt 𝕜 c s x)
    (hd : DifferentiableWithinAt 𝕜 d s x) :
    fderivWithin 𝕜 (fun y => (c y).comp (d y)) s x =
      (compL 𝕜 F G H (c x)).comp (fderivWithin 𝕜 d s x) +
        ((compL 𝕜 F G H).flip (d x)).comp (fderivWithin 𝕜 c s x) :=
  (hc.hasFDerivWithinAt.clm_comp hd.hasFDerivWithinAt).fderivWithin hxs


theorem fderiv_clm_comp (hc : DifferentiableAt 𝕜 c x) (hd : DifferentiableAt 𝕜 d x) :
    fderiv 𝕜 (fun y => (c y).comp (d y)) x =
      (compL 𝕜 F G H (c x)).comp (fderiv 𝕜 d x) +
        ((compL 𝕜 F G H).flip (d x)).comp (fderiv 𝕜 c x) :=
  (hc.hasFDerivAt.clm_comp hd.hasFDerivAt).fderiv


@[fun_prop]
theorem HasStrictFDerivAt.clm_apply (hc : HasStrictFDerivAt c c' x)
    (hu : HasStrictFDerivAt u u' x) :
    HasStrictFDerivAt (fun y => (c y) (u y)) ((c x).comp u' + c'.flip (u x)) x :=
  (isBoundedBilinearMap_apply.hasStrictFDerivAt (c x, u x)).comp x (hc.prod hu)


@[fun_prop]
theorem HasFDerivWithinAt.clm_apply (hc : HasFDerivWithinAt c c' s x)
    (hu : HasFDerivWithinAt u u' s x) :
    HasFDerivWithinAt (fun y => (c y) (u y)) ((c x).comp u' + c'.flip (u x)) s x := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    x : E
    s : Set E
    H : Type u_5
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    c : E → ContinuousLinearMap (RingHom.id 𝕜) G H
    c' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    u : E → G
    u' : ContinuousLinearMap (RingHom.id 𝕜) E G
    hc : HasFDerivWithinAt c c' s x
    hu : HasFDerivWithinAt u u' s x
    ⊢ HasFDerivWithinAt (fun y => (c y) (u y)) (HAdd.hAdd ((c x).comp u') (c'.flip …
  -/
  exact (isBoundedBilinearMap_apply.hasFDerivAt (c x, u x) :).comp_hasFDerivWithinAt x (hc.prod hu)
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem HasFDerivAt.clm_apply (hc : HasFDerivAt c c' x) (hu : HasFDerivAt u u' x) :
    HasFDerivAt (fun y => (c y) (u y)) ((c x).comp u' + c'.flip (u x)) x := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    x : E
    H : Type u_5
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    c : E → ContinuousLinearMap (RingHom.id 𝕜) G H
    c' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    u : E → G
    u' : ContinuousLinearMap (RingHom.id 𝕜) E G
    hc : HasFDerivAt c c' x
    hu : HasFDerivAt u u' x
    ⊢ HasFDerivAt (fun y => (c y) (u y)) (HAdd.hAdd ((c x).comp u') (c'.flip (u x) …
  -/
  exact (isBoundedBilinearMap_apply.hasFDerivAt (c x, u x) :).comp x (hc.prod hu)
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem DifferentiableWithinAt.clm_apply (hc : DifferentiableWithinAt 𝕜 c s x)
    (hu : DifferentiableWithinAt 𝕜 u s x) : DifferentiableWithinAt 𝕜 (fun y => (c y) (u y)) s x :=
  (hc.hasFDerivWithinAt.clm_apply hu.hasFDerivWithinAt).differentiableWithinAt


@[fun_prop]
theorem DifferentiableAt.clm_apply (hc : DifferentiableAt 𝕜 c x) (hu : DifferentiableAt 𝕜 u x) :
    DifferentiableAt 𝕜 (fun y => (c y) (u y)) x :=
  (hc.hasFDerivAt.clm_apply hu.hasFDerivAt).differentiableAt


@[fun_prop]
theorem DifferentiableOn.clm_apply (hc : DifferentiableOn 𝕜 c s) (hu : DifferentiableOn 𝕜 u s) :
    DifferentiableOn 𝕜 (fun y => (c y) (u y)) s := fun x hx => (hc x hx).clm_apply (hu x hx)


@[fun_prop]
theorem Differentiable.clm_apply (hc : Differentiable 𝕜 c) (hu : Differentiable 𝕜 u) :
    Differentiable 𝕜 fun y => (c y) (u y) := fun x => (hc x).clm_apply (hu x)


theorem fderivWithin_clm_apply (hxs : UniqueDiffWithinAt 𝕜 s x)
    (hc : DifferentiableWithinAt 𝕜 c s x) (hu : DifferentiableWithinAt 𝕜 u s x) :
    fderivWithin 𝕜 (fun y => (c y) (u y)) s x =
      (c x).comp (fderivWithin 𝕜 u s x) + (fderivWithin 𝕜 c s x).flip (u x) :=
  (hc.hasFDerivWithinAt.clm_apply hu.hasFDerivWithinAt).fderivWithin hxs


theorem fderiv_clm_apply (hc : DifferentiableAt 𝕜 c x) (hu : DifferentiableAt 𝕜 u x) :
    fderiv 𝕜 (fun y => (c y) (u y)) x = (c x).comp (fderiv 𝕜 u x) + (fderiv 𝕜 c x).flip (u x) :=
  (hc.hasFDerivAt.clm_apply hu.hasFDerivAt).fderiv


@[fun_prop]
theorem HasStrictFDerivAt.continuousMultilinear_apply_const (hc : HasStrictFDerivAt c c' x)
    (u : ∀ i, M i) : HasStrictFDerivAt (fun y ↦ (c y) u) (c'.flipMultilinear u) x :=
  (ContinuousMultilinearMap.apply 𝕜 M H u).hasStrictFDerivAt.comp x hc


@[fun_prop]
theorem HasFDerivWithinAt.continuousMultilinear_apply_const (hc : HasFDerivWithinAt c c' s x)
    (u : ∀ i, M i) :
    HasFDerivWithinAt (fun y ↦ (c y) u) (c'.flipMultilinear u) s x :=
  (ContinuousMultilinearMap.apply 𝕜 M H u).hasFDerivAt.comp_hasFDerivWithinAt x hc


@[fun_prop]
theorem HasFDerivAt.continuousMultilinear_apply_const (hc : HasFDerivAt c c' x) (u : ∀ i, M i) :
    HasFDerivAt (fun y ↦ (c y) u) (c'.flipMultilinear u) x :=
  (ContinuousMultilinearMap.apply 𝕜 M H u).hasFDerivAt.comp x hc


@[fun_prop]
theorem DifferentiableWithinAt.continuousMultilinear_apply_const
    (hc : DifferentiableWithinAt 𝕜 c s x) (u : ∀ i, M i) :
    DifferentiableWithinAt 𝕜 (fun y ↦ (c y) u) s x :=
  (hc.hasFDerivWithinAt.continuousMultilinear_apply_const u).differentiableWithinAt


@[fun_prop]
theorem DifferentiableAt.continuousMultilinear_apply_const (hc : DifferentiableAt 𝕜 c x)
    (u : ∀ i, M i) :
    DifferentiableAt 𝕜 (fun y ↦ (c y) u) x :=
  (hc.hasFDerivAt.continuousMultilinear_apply_const u).differentiableAt


@[fun_prop]
theorem DifferentiableOn.continuousMultilinear_apply_const (hc : DifferentiableOn 𝕜 c s)
    (u : ∀ i, M i) : DifferentiableOn 𝕜 (fun y ↦ (c y) u) s :=
  fun x hx ↦ (hc x hx).continuousMultilinear_apply_const u


@[fun_prop]
theorem Differentiable.continuousMultilinear_apply_const (hc : Differentiable 𝕜 c) (u : ∀ i, M i) :
    Differentiable 𝕜 fun y ↦ (c y) u := fun x ↦ (hc x).continuousMultilinear_apply_const u


theorem fderivWithin_continuousMultilinear_apply_const (hxs : UniqueDiffWithinAt 𝕜 s x)
    (hc : DifferentiableWithinAt 𝕜 c s x) (u : ∀ i, M i) :
    fderivWithin 𝕜 (fun y ↦ (c y) u) s x = ((fderivWithin 𝕜 c s x).flipMultilinear u) :=
  (hc.hasFDerivWithinAt.continuousMultilinear_apply_const u).fderivWithin hxs


theorem fderiv_continuousMultilinear_apply_const (hc : DifferentiableAt 𝕜 c x) (u : ∀ i, M i) :
    (fderiv 𝕜 (fun y ↦ (c y) u) x) = (fderiv 𝕜 c x).flipMultilinear u :=
  (hc.hasFDerivAt.continuousMultilinear_apply_const u).fderiv


/-- Application of a `ContinuousMultilinearMap` to a constant commutes with `fderivWithin`. -/
theorem fderivWithin_continuousMultilinear_apply_const_apply (hxs : UniqueDiffWithinAt 𝕜 s x)
    (hc : DifferentiableWithinAt 𝕜 c s x) (u : ∀ i, M i) (m : E) :
    (fderivWithin 𝕜 (fun y ↦ (c y) u) s x) m = (fderivWithin 𝕜 c s x) m u := by
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    x : E
    s : Set E
    ι : Type u_5
    inst✝⁴ : Fintype ι
    M : ι → Type u_6
    inst✝³ : (i : ι) → NormedAddCommGroup (M i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (M i)
    H : Type u_7
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    c : E → ContinuousMultilinearMap 𝕜 M H
    hxs : UniqueDiffWithinAt 𝕜 s x
    hc : DifferentiableWithinAt 𝕜 c s x
    u : (i : ι) → M i
    m : E
    ⊢ Eq ((fderivWithin 𝕜 (fun y => (c y) u) s x) m) (((fderivWithin 𝕜 c s x) m) u)
  -/
  simp [fderivWithin_continuousMultilinear_apply_const hxs hc]
  /-
    🎉 no goals
  -/


/-- Application of a `ContinuousMultilinearMap` to a constant commutes with `fderiv`. -/
theorem fderiv_continuousMultilinear_apply_const_apply (hc : DifferentiableAt 𝕜 c x)
    (u : ∀ i, M i) (m : E) :
    (fderiv 𝕜 (fun y ↦ (c y) u) x) m = (fderiv 𝕜 c x) m u := by
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    x : E
    ι : Type u_5
    inst✝⁴ : Fintype ι
    M : ι → Type u_6
    inst✝³ : (i : ι) → NormedAddCommGroup (M i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (M i)
    H : Type u_7
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    c : E → ContinuousMultilinearMap 𝕜 M H
    hc : DifferentiableAt 𝕜 c x
    u : (i : ι) → M i
    m : E
    ⊢ Eq ((fderiv 𝕜 (fun y => (c y) u) x) m) (((fderiv 𝕜 c x) m) u)
  -/
  simp [fderiv_continuousMultilinear_apply_const hc]
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem HasStrictFDerivAt.smul (hc : HasStrictFDerivAt c c' x) (hf : HasStrictFDerivAt f f' x) :
    HasStrictFDerivAt (fun y => c y • f y) (c x • f' + c'.smulRight (f x)) x :=
  (isBoundedBilinearMap_smul.hasStrictFDerivAt (c x, f x)).comp x <| hc.prod hf


@[fun_prop]
theorem HasFDerivWithinAt.smul (hc : HasFDerivWithinAt c c' s x) (hf : HasFDerivWithinAt f f' s x) :
    HasFDerivWithinAt (fun y => c y • f y) (c x • f' + c'.smulRight (f x)) s x := by
  exact (isBoundedBilinearMap_smul.hasFDerivAt (𝕜 := 𝕜) (c x, f x) :).comp_hasFDerivWithinAt x <|
    hc.prod hf


@[fun_prop]
theorem HasFDerivAt.smul (hc : HasFDerivAt c c' x) (hf : HasFDerivAt f f' x) :
    HasFDerivAt (fun y => c y • f y) (c x • f' + c'.smulRight (f x)) x := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    𝕜' : Type u_5
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    c : E → 𝕜'
    c' : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    hc : HasFDerivAt c c' x
    hf : HasFDerivAt f f' x
    ⊢ HasFDerivAt (fun y => HSMul.hSMul (c y) (f y)) (HAdd.hAdd (HSMul.hSMul (c x) …
  -/
  exact (isBoundedBilinearMap_smul.hasFDerivAt (𝕜 := 𝕜) (c x, f x) :).comp x <| hc.prod hf
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem DifferentiableWithinAt.smul (hc : DifferentiableWithinAt 𝕜 c s x)
    (hf : DifferentiableWithinAt 𝕜 f s x) : DifferentiableWithinAt 𝕜 (fun y => c y • f y) s x :=
  (hc.hasFDerivWithinAt.smul hf.hasFDerivWithinAt).differentiableWithinAt


@[simp, fun_prop]
theorem DifferentiableAt.smul (hc : DifferentiableAt 𝕜 c x) (hf : DifferentiableAt 𝕜 f x) :
    DifferentiableAt 𝕜 (fun y => c y • f y) x :=
  (hc.hasFDerivAt.smul hf.hasFDerivAt).differentiableAt


@[fun_prop]
theorem DifferentiableOn.smul (hc : DifferentiableOn 𝕜 c s) (hf : DifferentiableOn 𝕜 f s) :
    DifferentiableOn 𝕜 (fun y => c y • f y) s := fun x hx => (hc x hx).smul (hf x hx)


@[simp, fun_prop]
theorem Differentiable.smul (hc : Differentiable 𝕜 c) (hf : Differentiable 𝕜 f) :
    Differentiable 𝕜 fun y => c y • f y := fun x => (hc x).smul (hf x)


theorem fderivWithin_smul (hxs : UniqueDiffWithinAt 𝕜 s x) (hc : DifferentiableWithinAt 𝕜 c s x)
    (hf : DifferentiableWithinAt 𝕜 f s x) :
    fderivWithin 𝕜 (fun y => c y • f y) s x =
      c x • fderivWithin 𝕜 f s x + (fderivWithin 𝕜 c s x).smulRight (f x) :=
  (hc.hasFDerivWithinAt.smul hf.hasFDerivWithinAt).fderivWithin hxs


theorem fderiv_smul (hc : DifferentiableAt 𝕜 c x) (hf : DifferentiableAt 𝕜 f x) :
    fderiv 𝕜 (fun y => c y • f y) x = c x • fderiv 𝕜 f x + (fderiv 𝕜 c x).smulRight (f x) :=
  (hc.hasFDerivAt.smul hf.hasFDerivAt).fderiv


@[fun_prop]
theorem HasStrictFDerivAt.smul_const (hc : HasStrictFDerivAt c c' x) (f : F) :
    HasStrictFDerivAt (fun y => c y • f) (c'.smulRight f) x := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : E
    𝕜' : Type u_5
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    c : E → 𝕜'
    c' : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    hc : HasStrictFDerivAt c c' x
    f : F
    ⊢ HasStrictFDerivAt (fun y => HSMul.hSMul (c y) f) (c'.smulRight f) x
  -/
  simpa only [smul_zero, zero_add] using hc.smul (hasStrictFDerivAt_const f x)
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem HasFDerivWithinAt.smul_const (hc : HasFDerivWithinAt c c' s x) (f : F) :
    HasFDerivWithinAt (fun y => c y • f) (c'.smulRight f) s x := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : E
    s : Set E
    𝕜' : Type u_5
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    c : E → 𝕜'
    c' : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    hc : HasFDerivWithinAt c c' s x
    f : F
    ⊢ HasFDerivWithinAt (fun y => HSMul.hSMul (c y) f) (c'.smulRight f) s x
  -/
  simpa only [smul_zero, zero_add] using hc.smul (hasFDerivWithinAt_const f x s)
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem HasFDerivAt.smul_const (hc : HasFDerivAt c c' x) (f : F) :
    HasFDerivAt (fun y => c y • f) (c'.smulRight f) x := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : E
    𝕜' : Type u_5
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    c : E → 𝕜'
    c' : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜'
    hc : HasFDerivAt c c' x
    f : F
    ⊢ HasFDerivAt (fun y => HSMul.hSMul (c y) f) (c'.smulRight f) x
  -/
  simpa only [smul_zero, zero_add] using hc.smul (hasFDerivAt_const f x)
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem DifferentiableWithinAt.smul_const (hc : DifferentiableWithinAt 𝕜 c s x) (f : F) :
    DifferentiableWithinAt 𝕜 (fun y => c y • f) s x :=
  (hc.hasFDerivWithinAt.smul_const f).differentiableWithinAt


@[fun_prop]
theorem DifferentiableAt.smul_const (hc : DifferentiableAt 𝕜 c x) (f : F) :
    DifferentiableAt 𝕜 (fun y => c y • f) x :=
  (hc.hasFDerivAt.smul_const f).differentiableAt


@[fun_prop]
theorem DifferentiableOn.smul_const (hc : DifferentiableOn 𝕜 c s) (f : F) :
    DifferentiableOn 𝕜 (fun y => c y • f) s := fun x hx => (hc x hx).smul_const f


@[fun_prop]
theorem Differentiable.smul_const (hc : Differentiable 𝕜 c) (f : F) :
    Differentiable 𝕜 fun y => c y • f := fun x => (hc x).smul_const f


theorem fderivWithin_smul_const (hxs : UniqueDiffWithinAt 𝕜 s x)
    (hc : DifferentiableWithinAt 𝕜 c s x) (f : F) :
    fderivWithin 𝕜 (fun y => c y • f) s x = (fderivWithin 𝕜 c s x).smulRight f :=
  (hc.hasFDerivWithinAt.smul_const f).fderivWithin hxs


theorem fderiv_smul_const (hc : DifferentiableAt 𝕜 c x) (f : F) :
    fderiv 𝕜 (fun y => c y • f) x = (fderiv 𝕜 c x).smulRight f :=
  (hc.hasFDerivAt.smul_const f).fderiv


@[fun_prop]
theorem HasStrictFDerivAt.mul' {x : E} (ha : HasStrictFDerivAt a a' x)
    (hb : HasStrictFDerivAt b b' x) :
    HasStrictFDerivAt (fun y => a y * b y) (a x • b' + a'.smulRight (b x)) x :=
  ((ContinuousLinearMap.mul 𝕜 𝔸).isBoundedBilinearMap.hasStrictFDerivAt (a x, b x)).comp x
    (ha.prod hb)


@[fun_prop]
theorem HasStrictFDerivAt.mul (hc : HasStrictFDerivAt c c' x) (hd : HasStrictFDerivAt d d' x) :
    HasStrictFDerivAt (fun y => c y * d y) (c x • d' + d x • c') x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c d : E → 𝔸'
    c' d' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasStrictFDerivAt c c' x
    hd : HasStrictFDerivAt d d' x
    ⊢ HasStrictFDerivAt (fun y => HMul.hMul (c y) (d y)) (HAdd.hAdd (HSMul.hSMul ( …
  -/
  convert hc.mul' hd
  /-
    case h.e'_12.h.e'_6
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c d : E → 𝔸'
    c' d' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasStrictFDerivAt c c' x
    hd : HasStrictFDerivAt d d' x
    ⊢ Eq (HSMul.hSMul (d x) c') (c'.smulRight (d x))
  -/
  ext z
  /-
    case h.e'_12.h.e'_6.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c d : E → 𝔸'
    c' d' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasStrictFDerivAt c c' x
    hd : HasStrictFDerivAt d d' x
    z : E
    ⊢ Eq ((HSMul.hSMul (d x) c') z) ((c'.smulRight (d x)) z)
  -/
  apply mul_comm
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem HasFDerivWithinAt.mul' (ha : HasFDerivWithinAt a a' s x) (hb : HasFDerivWithinAt b b' s x) :
    HasFDerivWithinAt (fun y => a y * b y) (a x • b' + a'.smulRight (b x)) s x := by
  exact ((ContinuousLinearMap.mul 𝕜 𝔸).isBoundedBilinearMap.hasFDerivAt
    (a x, b x)).comp_hasFDerivWithinAt x (ha.prod hb)


@[fun_prop]
theorem HasFDerivWithinAt.mul (hc : HasFDerivWithinAt c c' s x) (hd : HasFDerivWithinAt d d' s x) :
    HasFDerivWithinAt (fun y => c y * d y) (c x • d' + d x • c') s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    s : Set E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c d : E → 𝔸'
    c' d' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasFDerivWithinAt c c' s x
    hd : HasFDerivWithinAt d d' s x
    ⊢ HasFDerivWithinAt (fun y => HMul.hMul (c y) (d y)) (HAdd.hAdd (HSMul.hSMul ( …
  -/
  convert hc.mul' hd
  /-
    case h.e'_12.h.e'_6
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    s : Set E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c d : E → 𝔸'
    c' d' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasFDerivWithinAt c c' s x
    hd : HasFDerivWithinAt d d' s x
    ⊢ Eq (HSMul.hSMul (d x) c') (c'.smulRight (d x))
  -/
  ext z
  /-
    case h.e'_12.h.e'_6.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    s : Set E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c d : E → 𝔸'
    c' d' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasFDerivWithinAt c c' s x
    hd : HasFDerivWithinAt d d' s x
    z : E
    ⊢ Eq ((HSMul.hSMul (d x) c') z) ((c'.smulRight (d x)) z)
  -/
  apply mul_comm
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem HasFDerivAt.mul' (ha : HasFDerivAt a a' x) (hb : HasFDerivAt b b' x) :
    HasFDerivAt (fun y => a y * b y) (a x • b' + a'.smulRight (b x)) x := by
  exact ((ContinuousLinearMap.mul 𝕜 𝔸).isBoundedBilinearMap.hasFDerivAt
    (a x, b x)).comp x (ha.prod hb)


@[fun_prop]
theorem HasFDerivAt.mul (hc : HasFDerivAt c c' x) (hd : HasFDerivAt d d' x) :
    HasFDerivAt (fun y => c y * d y) (c x • d' + d x • c') x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c d : E → 𝔸'
    c' d' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasFDerivAt c c' x
    hd : HasFDerivAt d d' x
    ⊢ HasFDerivAt (fun y => HMul.hMul (c y) (d y)) (HAdd.hAdd (HSMul.hSMul (c x) d …
  -/
  convert hc.mul' hd
  /-
    case h.e'_12.h.e'_6
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c d : E → 𝔸'
    c' d' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasFDerivAt c c' x
    hd : HasFDerivAt d d' x
    ⊢ Eq (HSMul.hSMul (d x) c') (c'.smulRight (d x))
  -/
  ext z
  /-
    case h.e'_12.h.e'_6.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c d : E → 𝔸'
    c' d' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasFDerivAt c c' x
    hd : HasFDerivAt d d' x
    z : E
    ⊢ Eq ((HSMul.hSMul (d x) c') z) ((c'.smulRight (d x)) z)
  -/
  apply mul_comm
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem DifferentiableWithinAt.mul (ha : DifferentiableWithinAt 𝕜 a s x)
    (hb : DifferentiableWithinAt 𝕜 b s x) : DifferentiableWithinAt 𝕜 (fun y => a y * b y) s x :=
  (ha.hasFDerivWithinAt.mul' hb.hasFDerivWithinAt).differentiableWithinAt


@[simp, fun_prop]
theorem DifferentiableAt.mul (ha : DifferentiableAt 𝕜 a x) (hb : DifferentiableAt 𝕜 b x) :
    DifferentiableAt 𝕜 (fun y => a y * b y) x :=
  (ha.hasFDerivAt.mul' hb.hasFDerivAt).differentiableAt


@[fun_prop]
theorem DifferentiableOn.mul (ha : DifferentiableOn 𝕜 a s) (hb : DifferentiableOn 𝕜 b s) :
    DifferentiableOn 𝕜 (fun y => a y * b y) s := fun x hx => (ha x hx).mul (hb x hx)


@[simp, fun_prop]
theorem Differentiable.mul (ha : Differentiable 𝕜 a) (hb : Differentiable 𝕜 b) :
    Differentiable 𝕜 fun y => a y * b y := fun x => (ha x).mul (hb x)


@[fun_prop]
theorem DifferentiableWithinAt.pow (ha : DifferentiableWithinAt 𝕜 a s x) :
    ∀ n : ℕ, DifferentiableWithinAt 𝕜 (fun x => a x ^ n) s x
            /-
              𝕜 : Type u_1
              inst✝⁴ : NontriviallyNormedField 𝕜
              E : Type u_2
              inst✝³ : NormedAddCommGroup E
              inst✝² : NormedSpace 𝕜 E
              x : E
              s : Set E
              𝔸 : Type u_5
              inst✝¹ : NormedRing 𝔸
              inst✝ : NormedAlgebra 𝕜 𝔸
              a : E → 𝔸
              ha : DifferentiableWithinAt 𝕜 a s x
              ⊢ DifferentiableWithinAt 𝕜 (fun x => HPow.hPow (a x) 0) s x
            -/
  | 0 => by simp only [pow_zero, differentiableWithinAt_const]
            /-
              🎉 no goals
            -/
                /-
                  𝕜 : Type u_1
                  inst✝⁴ : NontriviallyNormedField 𝕜
                  E : Type u_2
                  inst✝³ : NormedAddCommGroup E
                  inst✝² : NormedSpace 𝕜 E
                  x : E
                  s : Set E
                  𝔸 : Type u_5
                  inst✝¹ : NormedRing 𝔸
                  inst✝ : NormedAlgebra 𝕜 𝔸
                  a : E → 𝔸
                  ha : DifferentiableWithinAt 𝕜 a s x
                  n : Nat
                  ⊢ DifferentiableWithinAt 𝕜 (fun x => HPow.hPow (a x) (HAdd.hAdd n 1)) s x
                -/
  | n + 1 => by simp only [pow_succ', DifferentiableWithinAt.pow ha n, ha.mul]
                /-
                  🎉 no goals
                -/


@[simp, fun_prop]
theorem DifferentiableAt.pow (ha : DifferentiableAt 𝕜 a x) (n : ℕ) :
    DifferentiableAt 𝕜 (fun x => a x ^ n) x :=
  differentiableWithinAt_univ.mp <| ha.differentiableWithinAt.pow n


@[fun_prop]
theorem DifferentiableOn.pow (ha : DifferentiableOn 𝕜 a s) (n : ℕ) :
    DifferentiableOn 𝕜 (fun x => a x ^ n) s := fun x h => (ha x h).pow n


@[simp, fun_prop]
theorem Differentiable.pow (ha : Differentiable 𝕜 a) (n : ℕ) : Differentiable 𝕜 fun x => a x ^ n :=
  fun x => (ha x).pow n


theorem fderivWithin_mul' (hxs : UniqueDiffWithinAt 𝕜 s x) (ha : DifferentiableWithinAt 𝕜 a s x)
    (hb : DifferentiableWithinAt 𝕜 b s x) :
    fderivWithin 𝕜 (fun y => a y * b y) s x =
      a x • fderivWithin 𝕜 b s x + (fderivWithin 𝕜 a s x).smulRight (b x) :=
  (ha.hasFDerivWithinAt.mul' hb.hasFDerivWithinAt).fderivWithin hxs


theorem fderivWithin_mul (hxs : UniqueDiffWithinAt 𝕜 s x) (hc : DifferentiableWithinAt 𝕜 c s x)
    (hd : DifferentiableWithinAt 𝕜 d s x) :
    fderivWithin 𝕜 (fun y => c y * d y) s x =
      c x • fderivWithin 𝕜 d s x + d x • fderivWithin 𝕜 c s x :=
  (hc.hasFDerivWithinAt.mul hd.hasFDerivWithinAt).fderivWithin hxs


theorem fderiv_mul' (ha : DifferentiableAt 𝕜 a x) (hb : DifferentiableAt 𝕜 b x) :
    fderiv 𝕜 (fun y => a y * b y) x = a x • fderiv 𝕜 b x + (fderiv 𝕜 a x).smulRight (b x) :=
  (ha.hasFDerivAt.mul' hb.hasFDerivAt).fderiv


theorem fderiv_mul (hc : DifferentiableAt 𝕜 c x) (hd : DifferentiableAt 𝕜 d x) :
    fderiv 𝕜 (fun y => c y * d y) x = c x • fderiv 𝕜 d x + d x • fderiv 𝕜 c x :=
  (hc.hasFDerivAt.mul hd.hasFDerivAt).fderiv


@[fun_prop]
theorem HasStrictFDerivAt.mul_const' (ha : HasStrictFDerivAt a a' x) (b : 𝔸) :
    HasStrictFDerivAt (fun y => a y * b) (a'.smulRight b) x :=
  ((ContinuousLinearMap.mul 𝕜 𝔸).flip b).hasStrictFDerivAt.comp x ha


@[fun_prop]
theorem HasStrictFDerivAt.mul_const (hc : HasStrictFDerivAt c c' x) (d : 𝔸') :
    HasStrictFDerivAt (fun y => c y * d) (d • c') x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c : E → 𝔸'
    c' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasStrictFDerivAt c c' x
    d : 𝔸'
    ⊢ HasStrictFDerivAt (fun y => HMul.hMul (c y) d) (HSMul.hSMul d c') x
  -/
  convert hc.mul_const' d
  /-
    case h.e'_12.h.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c : E → 𝔸'
    c' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasStrictFDerivAt c c' x
    d : 𝔸'
    e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup NormedAddCommGroup.toAddCommGr …
    he✝ : Eq NormedSpace.toModule NormedSpace.toModule
    ⊢ Eq (HSMul.hSMul d c') (c'.smulRight d)
  -/
  ext z
  /-
    case h.e'_12.h.h.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c : E → 𝔸'
    c' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasStrictFDerivAt c c' x
    d : 𝔸'
    e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup NormedAddCommGroup.toAddCommGr …
    he✝ : Eq NormedSpace.toModule NormedSpace.toModule
    z : E
    ⊢ Eq ((HSMul.hSMul d c') z) ((c'.smulRight d) z)
  -/
  apply mul_comm
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem HasFDerivWithinAt.mul_const' (ha : HasFDerivWithinAt a a' s x) (b : 𝔸) :
    HasFDerivWithinAt (fun y => a y * b) (a'.smulRight b) s x :=
  ((ContinuousLinearMap.mul 𝕜 𝔸).flip b).hasFDerivAt.comp_hasFDerivWithinAt x ha


@[fun_prop]
theorem HasFDerivWithinAt.mul_const (hc : HasFDerivWithinAt c c' s x) (d : 𝔸') :
    HasFDerivWithinAt (fun y => c y * d) (d • c') s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    s : Set E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c : E → 𝔸'
    c' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasFDerivWithinAt c c' s x
    d : 𝔸'
    ⊢ HasFDerivWithinAt (fun y => HMul.hMul (c y) d) (HSMul.hSMul d c') s x
  -/
  convert hc.mul_const' d
  /-
    case h.e'_12.h.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    s : Set E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c : E → 𝔸'
    c' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasFDerivWithinAt c c' s x
    d : 𝔸'
    e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup NormedAddCommGroup.toAddCommGr …
    he✝ : Eq NormedSpace.toModule NormedSpace.toModule
    ⊢ Eq (HSMul.hSMul d c') (c'.smulRight d)
  -/
  ext z
  /-
    case h.e'_12.h.h.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    s : Set E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c : E → 𝔸'
    c' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasFDerivWithinAt c c' s x
    d : 𝔸'
    e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup NormedAddCommGroup.toAddCommGr …
    he✝ : Eq NormedSpace.toModule NormedSpace.toModule
    z : E
    ⊢ Eq ((HSMul.hSMul d c') z) ((c'.smulRight d) z)
  -/
  apply mul_comm
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem HasFDerivAt.mul_const' (ha : HasFDerivAt a a' x) (b : 𝔸) :
    HasFDerivAt (fun y => a y * b) (a'.smulRight b) x :=
  ((ContinuousLinearMap.mul 𝕜 𝔸).flip b).hasFDerivAt.comp x ha


@[fun_prop]
theorem HasFDerivAt.mul_const (hc : HasFDerivAt c c' x) (d : 𝔸') :
    HasFDerivAt (fun y => c y * d) (d • c') x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c : E → 𝔸'
    c' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasFDerivAt c c' x
    d : 𝔸'
    ⊢ HasFDerivAt (fun y => HMul.hMul (c y) d) (HSMul.hSMul d c') x
  -/
  convert hc.mul_const' d
  /-
    case h.e'_12.h.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c : E → 𝔸'
    c' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasFDerivAt c c' x
    d : 𝔸'
    e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup NormedAddCommGroup.toAddCommGr …
    he✝ : Eq NormedSpace.toModule NormedSpace.toModule
    ⊢ Eq (HSMul.hSMul d c') (c'.smulRight d)
  -/
  ext z
  /-
    case h.e'_12.h.h.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : E
    𝔸' : Type u_6
    inst✝¹ : NormedCommRing 𝔸'
    inst✝ : NormedAlgebra 𝕜 𝔸'
    c : E → 𝔸'
    c' : ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    hc : HasFDerivAt c c' x
    d : 𝔸'
    e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup NormedAddCommGroup.toAddCommGr …
    he✝ : Eq NormedSpace.toModule NormedSpace.toModule
    z : E
    ⊢ Eq ((HSMul.hSMul d c') z) ((c'.smulRight d) z)
  -/
  apply mul_comm
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem DifferentiableWithinAt.mul_const (ha : DifferentiableWithinAt 𝕜 a s x) (b : 𝔸) :
    DifferentiableWithinAt 𝕜 (fun y => a y * b) s x :=
  (ha.hasFDerivWithinAt.mul_const' b).differentiableWithinAt


@[fun_prop]
theorem DifferentiableAt.mul_const (ha : DifferentiableAt 𝕜 a x) (b : 𝔸) :
    DifferentiableAt 𝕜 (fun y => a y * b) x :=
  (ha.hasFDerivAt.mul_const' b).differentiableAt


@[fun_prop]
theorem DifferentiableOn.mul_const (ha : DifferentiableOn 𝕜 a s) (b : 𝔸) :
    DifferentiableOn 𝕜 (fun y => a y * b) s := fun x hx => (ha x hx).mul_const b


@[fun_prop]
theorem Differentiable.mul_const (ha : Differentiable 𝕜 a) (b : 𝔸) :
    Differentiable 𝕜 fun y => a y * b := fun x => (ha x).mul_const b


theorem fderivWithin_mul_const' (hxs : UniqueDiffWithinAt 𝕜 s x)
    (ha : DifferentiableWithinAt 𝕜 a s x) (b : 𝔸) :
    fderivWithin 𝕜 (fun y => a y * b) s x = (fderivWithin 𝕜 a s x).smulRight b :=
  (ha.hasFDerivWithinAt.mul_const' b).fderivWithin hxs


theorem fderivWithin_mul_const (hxs : UniqueDiffWithinAt 𝕜 s x)
    (hc : DifferentiableWithinAt 𝕜 c s x) (d : 𝔸') :
    fderivWithin 𝕜 (fun y => c y * d) s x = d • fderivWithin 𝕜 c s x :=
  (hc.hasFDerivWithinAt.mul_const d).fderivWithin hxs


theorem fderiv_mul_const' (ha : DifferentiableAt 𝕜 a x) (b : 𝔸) :
    fderiv 𝕜 (fun y => a y * b) x = (fderiv 𝕜 a x).smulRight b :=
  (ha.hasFDerivAt.mul_const' b).fderiv


theorem fderiv_mul_const (hc : DifferentiableAt 𝕜 c x) (d : 𝔸') :
    fderiv 𝕜 (fun y => c y * d) x = d • fderiv 𝕜 c x :=
  (hc.hasFDerivAt.mul_const d).fderiv


@[fun_prop]
theorem HasStrictFDerivAt.const_mul (ha : HasStrictFDerivAt a a' x) (b : 𝔸) :
    HasStrictFDerivAt (fun y => b * a y) (b • a') x :=
  ((ContinuousLinearMap.mul 𝕜 𝔸) b).hasStrictFDerivAt.comp x ha


@[fun_prop]
theorem HasFDerivWithinAt.const_mul (ha : HasFDerivWithinAt a a' s x) (b : 𝔸) :
    HasFDerivWithinAt (fun y => b * a y) (b • a') s x :=
  ((ContinuousLinearMap.mul 𝕜 𝔸) b).hasFDerivAt.comp_hasFDerivWithinAt x ha


@[fun_prop]
theorem HasFDerivAt.const_mul (ha : HasFDerivAt a a' x) (b : 𝔸) :
    HasFDerivAt (fun y => b * a y) (b • a') x :=
  ((ContinuousLinearMap.mul 𝕜 𝔸) b).hasFDerivAt.comp x ha


@[fun_prop]
theorem DifferentiableWithinAt.const_mul (ha : DifferentiableWithinAt 𝕜 a s x) (b : 𝔸) :
    DifferentiableWithinAt 𝕜 (fun y => b * a y) s x :=
  (ha.hasFDerivWithinAt.const_mul b).differentiableWithinAt


@[fun_prop]
theorem DifferentiableAt.const_mul (ha : DifferentiableAt 𝕜 a x) (b : 𝔸) :
    DifferentiableAt 𝕜 (fun y => b * a y) x :=
  (ha.hasFDerivAt.const_mul b).differentiableAt


@[fun_prop]
theorem DifferentiableOn.const_mul (ha : DifferentiableOn 𝕜 a s) (b : 𝔸) :
    DifferentiableOn 𝕜 (fun y => b * a y) s := fun x hx => (ha x hx).const_mul b


@[fun_prop]
theorem Differentiable.const_mul (ha : Differentiable 𝕜 a) (b : 𝔸) :
    Differentiable 𝕜 fun y => b * a y := fun x => (ha x).const_mul b


theorem fderivWithin_const_mul (hxs : UniqueDiffWithinAt 𝕜 s x)
    (ha : DifferentiableWithinAt 𝕜 a s x) (b : 𝔸) :
    fderivWithin 𝕜 (fun y => b * a y) s x = b • fderivWithin 𝕜 a s x :=
  (ha.hasFDerivWithinAt.const_mul b).fderivWithin hxs


theorem fderiv_const_mul (ha : DifferentiableAt 𝕜 a x) (b : 𝔸) :
    fderiv 𝕜 (fun y => b * a y) x = b • fderiv 𝕜 a x :=
  (ha.hasFDerivAt.const_mul b).fderiv


@[fun_prop]
theorem hasStrictFDerivAt_list_prod' [Fintype ι] {l : List ι} {x : ι → 𝔸} :
    HasStrictFDerivAt (𝕜 := 𝕜) (fun x ↦ (l.map x).prod)
      (∑ i : Fin l.length, ((l.take i).map x).prod •
        smulRight (proj l[i]) ((l.drop (.succ i)).map x).prod) x := by
  induction l with
  | nil => simp [hasStrictFDerivAt_const]
  | cons a l IH =>
    simp only [List.map_cons, List.prod_cons, ← proj_apply (R := 𝕜) (φ := fun _ : ι ↦ 𝔸) a]
    exact .congr_fderiv (.mul' (ContinuousLinearMap.hasStrictFDerivAt _) IH)
      (by ext; simp [Fin.sum_univ_succ, Finset.mul_sum, mul_assoc, add_comm])


@[fun_prop]
theorem hasStrictFDerivAt_list_prod_finRange' {n : ℕ} {x : Fin n → 𝔸} :
    HasStrictFDerivAt (𝕜 := 𝕜) (fun x ↦ ((List.finRange n).map x).prod)
      (∑ i : Fin n, (((List.finRange n).take i).map x).prod •
        smulRight (proj i) (((List.finRange n).drop (.succ i)).map x).prod) x :=
  hasStrictFDerivAt_list_prod'.congr_fderiv <|
                                                             /-
                                                               𝕜 : Type u_1
                                                               inst✝² : NontriviallyNormedField 𝕜
                                                               𝔸 : Type u_6
                                                               inst✝¹ : NormedRing 𝔸
                                                               inst✝ : NormedAlgebra 𝕜 𝔸
                                                               n : Nat
                                                               x : Fin n → 𝔸
                                                               ⊢ ∀ (i : Fin (List.finRange n).length), Iff (Membership.mem Finset.univ i) (Me …
                                                             -/
                                                             /-
                                                               🎉 no goals
                                                             -/
    Finset.sum_equiv (finCongr (List.length_finRange n)) (by simp) (by simp [Fin.forall_iff])
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[fun_prop]
theorem hasStrictFDerivAt_list_prod_attach' [DecidableEq ι] {l : List ι} {x : {i // i ∈ l} → 𝔸} :
    HasStrictFDerivAt (𝕜 := 𝕜) (fun x ↦ (l.attach.map x).prod)
      (∑ i : Fin l.length, ((l.attach.take i).map x).prod •
        smulRight (proj l.attach[i.cast List.length_attach.symm])
          ((l.attach.drop (.succ i)).map x).prod) x :=
  hasStrictFDerivAt_list_prod'.congr_fderiv <| Eq.symm <|
                                                            /-
                                                              𝕜 : Type u_1
                                                              inst✝³ : NontriviallyNormedField 𝕜
                                                              ι : Type u_5
                                                              𝔸 : Type u_6
                                                              inst✝² : NormedRing 𝔸
                                                              inst✝¹ : NormedAlgebra 𝕜 𝔸
                                                              inst✝ : DecidableEq ι
                                                              l : List ι
                                                              x : (Subtype fun i => Membership.mem l i) → 𝔸
                                                              ⊢ ∀ (i : Fin l.length), Iff (Membership.mem Finset.univ i) (Membership.mem Fin …
                                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
    Finset.sum_equiv (finCongr List.length_attach.symm) (by simp) (by simp)
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[fun_prop]
theorem hasFDerivAt_list_prod' [Fintype ι] {l : List ι} {x : ι → 𝔸'} :
    HasFDerivAt (𝕜 := 𝕜) (fun x ↦ (l.map x).prod)
      (∑ i : Fin l.length, ((l.take i).map x).prod •
        smulRight (proj l[i]) ((l.drop (.succ i)).map x).prod) x :=
  hasStrictFDerivAt_list_prod'.hasFDerivAt


@[fun_prop]
theorem hasFDerivAt_list_prod_finRange' {n : ℕ} {x : Fin n → 𝔸} :
    HasFDerivAt (𝕜 := 𝕜) (fun x ↦ ((List.finRange n).map x).prod)
      (∑ i : Fin n, (((List.finRange n).take i).map x).prod •
        smulRight (proj i) (((List.finRange n).drop (.succ i)).map x).prod) x :=
  (hasStrictFDerivAt_list_prod_finRange').hasFDerivAt


@[fun_prop]
theorem hasFDerivAt_list_prod_attach' [DecidableEq ι] {l : List ι} {x : {i // i ∈ l} → 𝔸} :
    HasFDerivAt (𝕜 := 𝕜) (fun x ↦ (l.attach.map x).prod)
      (∑ i : Fin l.length, ((l.attach.take i).map x).prod •
        smulRight (proj l.attach[i.cast List.length_attach.symm])
          ((l.attach.drop (.succ i)).map x).prod) x :=
  hasStrictFDerivAt_list_prod_attach'.hasFDerivAt


/--
Auxiliary lemma for `hasStrictFDerivAt_multiset_prod`.

For `NormedCommRing 𝔸'`, can rewrite as `Multiset` using `Multiset.prod_coe`.
-/
@[fun_prop]
theorem hasStrictFDerivAt_list_prod [DecidableEq ι] [Fintype ι] {l : List ι} {x : ι → 𝔸'} :
    HasStrictFDerivAt (𝕜 := 𝕜) (fun x ↦ (l.map x).prod)
      (l.map fun i ↦ ((l.erase i).map x).prod • proj i).sum x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    ι : Type u_5
    𝔸' : Type u_7
    inst✝³ : NormedCommRing 𝔸'
    inst✝² : NormedAlgebra 𝕜 𝔸'
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    l : List ι
    x : ι → 𝔸'
    ⊢ HasStrictFDerivAt (fun x => (List.map x l).prod) (List.map (fun i => HSMul.h …
  -/
  refine hasStrictFDerivAt_list_prod'.congr_fderiv ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    ι : Type u_5
    𝔸' : Type u_7
    inst✝³ : NormedCommRing 𝔸'
    inst✝² : NormedAlgebra 𝕜 𝔸'
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    l : List ι
    x : ι → 𝔸'
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (List.map x (List.take (↑i) l)).pro …
  -/
  conv_rhs => arg 1; arg 2; rw [← List.finRange_map_get l]
  simp only [List.map_map, ← List.sum_toFinset _ (List.nodup_finRange _), List.toFinset_finRange,
    Function.comp_def, ((List.erase_getElem _).map _).prod_eq, List.eraseIdx_eq_take_drop_succ,
    List.map_append, List.prod_append, List.get_eq_getElem, Fin.getElem_fin, Nat.succ_eq_add_one]
  exact Finset.sum_congr rfl fun i _ ↦ by
    ext; simp only [smul_apply, smulRight_apply, smul_eq_mul]; ring


@[fun_prop]
theorem hasStrictFDerivAt_multiset_prod [DecidableEq ι] [Fintype ι] {u : Multiset ι} {x : ι → 𝔸'} :
    HasStrictFDerivAt (𝕜 := 𝕜) (fun x ↦ (u.map x).prod)
      (u.map (fun i ↦ ((u.erase i).map x).prod • proj i)).sum x :=
                           /-
                             𝕜 : Type u_1
                             inst✝⁴ : NontriviallyNormedField 𝕜
                             ι : Type u_5
                             𝔸' : Type u_7
                             inst✝³ : NormedCommRing 𝔸'
                             inst✝² : NormedAlgebra 𝕜 𝔸'
                             inst✝¹ : DecidableEq ι
                             inst✝ : Fintype ι
                             u : Multiset ι
                             x : ι → 𝔸'
                             l : List ι
                             ⊢ HasStrictFDerivAt (fun x => (Multiset.map x (Quotient.mk (List.isSetoid ι) l …
                           -/
  u.inductionOn fun l ↦ by simpa using hasStrictFDerivAt_list_prod
                           /-
                             🎉 no goals
                           -/


@[fun_prop]
theorem hasFDerivAt_multiset_prod [DecidableEq ι] [Fintype ι] {u : Multiset ι} {x : ι → 𝔸'} :
    HasFDerivAt (𝕜 := 𝕜) (fun x ↦ (u.map x).prod)
      (Multiset.sum (u.map (fun i ↦ ((u.erase i).map x).prod • proj i))) x :=
  hasStrictFDerivAt_multiset_prod.hasFDerivAt


theorem hasStrictFDerivAt_finset_prod [DecidableEq ι] [Fintype ι] {x : ι → 𝔸'} :
    HasStrictFDerivAt (𝕜 := 𝕜) (∏ i ∈ u, · i) (∑ i ∈ u, (∏ j ∈ u.erase i, x j) • proj i) x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    ι : Type u_5
    𝔸' : Type u_7
    inst✝³ : NormedCommRing 𝔸'
    inst✝² : NormedAlgebra 𝕜 𝔸'
    u : Finset ι
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    x : ι → 𝔸'
    ⊢ HasStrictFDerivAt (fun x => u.prod fun i => x i) (u.sum fun i => HSMul.hSMul …
  -/
  simp only [Finset.sum_eq_multiset_sum, Finset.prod_eq_multiset_prod]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    ι : Type u_5
    𝔸' : Type u_7
    inst✝³ : NormedCommRing 𝔸'
    inst✝² : NormedAlgebra 𝕜 𝔸'
    u : Finset ι
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    x : ι → 𝔸'
    ⊢ HasStrictFDerivAt (fun x => (Multiset.map x u.val).prod) (Multiset.map (fun  …
  -/
  exact hasStrictFDerivAt_multiset_prod
  /-
    🎉 no goals
  -/


theorem hasFDerivAt_finset_prod [DecidableEq ι] [Fintype ι] {x : ι → 𝔸'} :
    HasFDerivAt (𝕜 := 𝕜) (∏ i ∈ u, · i) (∑ i ∈ u, (∏ j ∈ u.erase i, x j) • proj i) x :=
  hasStrictFDerivAt_finset_prod.hasFDerivAt


@[fun_prop]
theorem HasStrictFDerivAt.list_prod' {l : List ι} {x : E}
    (h : ∀ i ∈ l, HasStrictFDerivAt (f i ·) (f' i) x) :
    HasStrictFDerivAt (fun x ↦ (l.map (f · x)).prod)
      (∑ i : Fin l.length, ((l.take i).map (f · x)).prod •
        smulRight (f' l[i]) ((l.drop (.succ i)).map (f · x)).prod) x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    ι : Type u_5
    𝔸 : Type u_6
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    f : ι → E → 𝔸
    f' : ι → ContinuousLinearMap (RingHom.id 𝕜) E 𝔸
    l : List ι
    x : E
    h : ∀ (i : ι), Membership.mem l i → HasStrictFDerivAt (fun x => f i x) (f' i) x
    ⊢ HasStrictFDerivAt (fun x => (List.map (fun x_1 => f x_1 x) l).prod) (Finset. …
  -/
  simp_rw [Fin.getElem_fin, ← l.get_eq_getElem, ← List.finRange_map_get l, List.map_map]
  -- After #19108, we have to be optimistic with `:)`s; otherwise Lean decides it need to find
  -- `NormedAddCommGroup (List 𝔸)` which is nonsense.
  refine .congr_fderiv (hasStrictFDerivAt_list_prod_finRange'.comp x
    (hasStrictFDerivAt_pi.mpr fun i ↦ h (l.get i) (List.getElem_mem ..)) :) ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    ι : Type u_5
    𝔸 : Type u_6
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    f : ι → E → 𝔸
    f' : ι → ContinuousLinearMap (RingHom.id 𝕜) E 𝔸
    l : List ι
    x : E
    h : ∀ (i : ι), Membership.mem l i → HasStrictFDerivAt (fun x => f i x) (f' i) x
    ⊢ Eq ((Finset.univ.sum fun i => HSMul.hSMul (List.map (fun i => f (l.get i) x) …
  -/
  ext m
  simp_rw [List.map_take, List.map_drop, List.map_map, comp_apply, sum_apply, smul_apply,
    smulRight_apply, proj_apply, pi_apply, Function.comp_def]


/--
Unlike `HasFDerivAt.finset_prod`, supports non-commutative multiply and duplicate elements.
-/
@[fun_prop]
theorem HasFDerivAt.list_prod' {l : List ι} {x : E}
    (h : ∀ i ∈ l, HasFDerivAt (f i ·) (f' i) x) :
    HasFDerivAt (fun x ↦ (l.map (f · x)).prod)
      (∑ i : Fin l.length, ((l.take i).map (f · x)).prod •
        smulRight (f' l[i]) ((l.drop (.succ i)).map (f · x)).prod) x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    ι : Type u_5
    𝔸 : Type u_6
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    f : ι → E → 𝔸
    f' : ι → ContinuousLinearMap (RingHom.id 𝕜) E 𝔸
    l : List ι
    x : E
    h : ∀ (i : ι), Membership.mem l i → HasFDerivAt (fun x => f i x) (f' i) x
    ⊢ HasFDerivAt (fun x => (List.map (fun x_1 => f x_1 x) l).prod) (Finset.univ.s …
  -/
  simp_rw [Fin.getElem_fin, ← l.get_eq_getElem, ← List.finRange_map_get l, List.map_map]
  refine .congr_fderiv (hasFDerivAt_list_prod_finRange'.comp x
    (hasFDerivAt_pi.mpr fun i ↦ h (l.get i) (l.get_mem i)) :) ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    ι : Type u_5
    𝔸 : Type u_6
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    f : ι → E → 𝔸
    f' : ι → ContinuousLinearMap (RingHom.id 𝕜) E 𝔸
    l : List ι
    x : E
    h : ∀ (i : ι), Membership.mem l i → HasFDerivAt (fun x => f i x) (f' i) x
    ⊢ Eq ((Finset.univ.sum fun i => HSMul.hSMul (List.map (fun i => f (l.get i) x) …
  -/
  ext m
  simp_rw [List.map_take, List.map_drop, List.map_map, comp_apply, sum_apply, smul_apply,
    smulRight_apply, proj_apply, pi_apply, Function.comp_def]


@[fun_prop]
theorem HasFDerivWithinAt.list_prod' {l : List ι} {x : E}
    (h : ∀ i ∈ l, HasFDerivWithinAt (f i ·) (f' i) s x) :
    HasFDerivWithinAt (fun x ↦ (l.map (f · x)).prod)
      (∑ i : Fin l.length, ((l.take i).map (f · x)).prod •
        smulRight (f' l[i]) ((l.drop (.succ i)).map (f · x)).prod) s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    s : Set E
    ι : Type u_5
    𝔸 : Type u_6
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    f : ι → E → 𝔸
    f' : ι → ContinuousLinearMap (RingHom.id 𝕜) E 𝔸
    l : List ι
    x : E
    h : ∀ (i : ι), Membership.mem l i → HasFDerivWithinAt (fun x => f i x) (f' i)  …
    ⊢ HasFDerivWithinAt (fun x => (List.map (fun x_1 => f x_1 x) l).prod) (Finset. …
  -/
  simp_rw [Fin.getElem_fin, ← l.get_eq_getElem, ← List.finRange_map_get l, List.map_map]
  refine .congr_fderiv (hasFDerivAt_list_prod_finRange'.comp_hasFDerivWithinAt x
    (hasFDerivWithinAt_pi.mpr fun i ↦ h (l.get i) (l.get_mem i)) :) ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    s : Set E
    ι : Type u_5
    𝔸 : Type u_6
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    f : ι → E → 𝔸
    f' : ι → ContinuousLinearMap (RingHom.id 𝕜) E 𝔸
    l : List ι
    x : E
    h : ∀ (i : ι), Membership.mem l i → HasFDerivWithinAt (fun x => f i x) (f' i)  …
    ⊢ Eq ((Finset.univ.sum fun i => HSMul.hSMul (List.map (fun i => f (l.get i) x) …
  -/
  ext m
  simp_rw [List.map_take, List.map_drop, List.map_map, comp_apply, sum_apply, smul_apply,
    smulRight_apply, proj_apply, pi_apply, Function.comp_def]


theorem fderiv_list_prod' {l : List ι} {x : E}
    (h : ∀ i ∈ l, DifferentiableAt 𝕜 (f i ·) x) :
    fderiv 𝕜 (fun x ↦ (l.map (f · x)).prod) x =
      ∑ i : Fin l.length, ((l.take i).map (f · x)).prod •
        smulRight (fderiv 𝕜 (fun x ↦ f l[i] x) x) ((l.drop (.succ i)).map (f · x)).prod :=
  (HasFDerivAt.list_prod' fun i hi ↦ (h i hi).hasFDerivAt).fderiv


theorem fderivWithin_list_prod' {l : List ι} {x : E}
    (hxs : UniqueDiffWithinAt 𝕜 s x) (h : ∀ i ∈ l, DifferentiableWithinAt 𝕜 (f i ·) s x) :
    fderivWithin 𝕜 (fun x ↦ (l.map (f · x)).prod) s x =
      ∑ i : Fin l.length, ((l.take i).map (f · x)).prod •
        smulRight (fderivWithin 𝕜 (fun x ↦ f l[i] x) s x) ((l.drop (.succ i)).map (f · x)).prod :=
  (HasFDerivWithinAt.list_prod' fun i hi ↦ (h i hi).hasFDerivWithinAt).fderivWithin hxs


@[fun_prop]
theorem HasStrictFDerivAt.multiset_prod [DecidableEq ι] {u : Multiset ι} {x : E}
    (h : ∀ i ∈ u, HasStrictFDerivAt (g i ·) (g' i) x) :
    HasStrictFDerivAt (fun x ↦ (u.map (g · x)).prod)
      (u.map fun i ↦ ((u.erase i).map (g · x)).prod • g' i).sum x := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_5
    𝔸' : Type u_7
    inst✝² : NormedCommRing 𝔸'
    inst✝¹ : NormedAlgebra 𝕜 𝔸'
    g : ι → E → 𝔸'
    g' : ι → ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    inst✝ : DecidableEq ι
    u : Multiset ι
    x : E
    h : ∀ (i : ι), Membership.mem u i → HasStrictFDerivAt (fun x => g i x) (g' i) x
    ⊢ HasStrictFDerivAt (fun x => (Multiset.map (fun x_1 => g x_1 x) u).prod) (Mul …
  -/
  simp only [← Multiset.attach_map_val u, Multiset.map_map]
  exact .congr_fderiv
    (hasStrictFDerivAt_multiset_prod.comp x <|
      hasStrictFDerivAt_pi.mpr fun i ↦ h (Subtype.val i) i.prop :)
    (by ext; simp [Finset.sum_multiset_map_count, u.erase_attach_map (g · x)])


/--
Unlike `HasFDerivAt.finset_prod`, supports duplicate elements.
-/
@[fun_prop]
theorem HasFDerivAt.multiset_prod [DecidableEq ι] {u : Multiset ι} {x : E}
    (h : ∀ i ∈ u, HasFDerivAt (g i ·) (g' i) x) :
    HasFDerivAt (fun x ↦ (u.map (g · x)).prod)
      (u.map fun i ↦ ((u.erase i).map (g · x)).prod • g' i).sum x := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_5
    𝔸' : Type u_7
    inst✝² : NormedCommRing 𝔸'
    inst✝¹ : NormedAlgebra 𝕜 𝔸'
    g : ι → E → 𝔸'
    g' : ι → ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    inst✝ : DecidableEq ι
    u : Multiset ι
    x : E
    h : ∀ (i : ι), Membership.mem u i → HasFDerivAt (fun x => g i x) (g' i) x
    ⊢ HasFDerivAt (fun x => (Multiset.map (fun x_1 => g x_1 x) u).prod) (Multiset. …
  -/
  simp only [← Multiset.attach_map_val u, Multiset.map_map]
  exact .congr_fderiv
    (hasFDerivAt_multiset_prod.comp x <| hasFDerivAt_pi.mpr fun i ↦ h (Subtype.val i) i.prop :)
    (by ext; simp [Finset.sum_multiset_map_count, u.erase_attach_map (g · x)])


@[fun_prop]
theorem HasFDerivWithinAt.multiset_prod [DecidableEq ι] {u : Multiset ι} {x : E}
    (h : ∀ i ∈ u, HasFDerivWithinAt (g i ·) (g' i) s x) :
    HasFDerivWithinAt (fun x ↦ (u.map (g · x)).prod)
      (u.map fun i ↦ ((u.erase i).map (g · x)).prod • g' i).sum s x := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    s : Set E
    ι : Type u_5
    𝔸' : Type u_7
    inst✝² : NormedCommRing 𝔸'
    inst✝¹ : NormedAlgebra 𝕜 𝔸'
    g : ι → E → 𝔸'
    g' : ι → ContinuousLinearMap (RingHom.id 𝕜) E 𝔸'
    inst✝ : DecidableEq ι
    u : Multiset ι
    x : E
    h : ∀ (i : ι), Membership.mem u i → HasFDerivWithinAt (fun x => g i x) (g' i)  …
    ⊢ HasFDerivWithinAt (fun x => (Multiset.map (fun x_1 => g x_1 x) u).prod) (Mul …
  -/
  simp only [← Multiset.attach_map_val u, Multiset.map_map]
  exact .congr_fderiv
    (hasFDerivAt_multiset_prod.comp_hasFDerivWithinAt x <|
      hasFDerivWithinAt_pi.mpr fun i ↦ h (Subtype.val i) i.prop :)
    (by ext; simp [Finset.sum_multiset_map_count, u.erase_attach_map (g · x)])


theorem fderiv_multiset_prod [DecidableEq ι] {u : Multiset ι} {x : E}
    (h : ∀ i ∈ u, DifferentiableAt 𝕜 (g i ·) x) :
    fderiv 𝕜 (fun x ↦ (u.map (g · x)).prod) x =
      (u.map fun i ↦ ((u.erase i).map (g · x)).prod • fderiv 𝕜 (g i) x).sum :=
  (HasFDerivAt.multiset_prod fun i hi ↦ (h i hi).hasFDerivAt).fderiv


theorem fderivWithin_multiset_prod [DecidableEq ι] {u : Multiset ι} {x : E}
    (hxs : UniqueDiffWithinAt 𝕜 s x) (h : ∀ i ∈ u, DifferentiableWithinAt 𝕜 (g i ·) s x) :
    fderivWithin 𝕜 (fun x ↦ (u.map (g · x)).prod) s x =
      (u.map fun i ↦ ((u.erase i).map (g · x)).prod • fderivWithin 𝕜 (g i) s x).sum :=
  (HasFDerivWithinAt.multiset_prod fun i hi ↦ (h i hi).hasFDerivWithinAt).fderivWithin hxs


theorem HasStrictFDerivAt.finset_prod [DecidableEq ι] {x : E}
    (hg : ∀ i ∈ u, HasStrictFDerivAt (g i) (g' i) x) :
    HasStrictFDerivAt (∏ i ∈ u, g i ·) (∑ i ∈ u, (∏ j ∈ u.erase i, g j x) • g' i) x := by
  simpa [← Finset.prod_attach u] using .congr_fderiv
    (hasStrictFDerivAt_finset_prod.comp x <| hasStrictFDerivAt_pi.mpr fun i ↦ hg i i.prop)
    (by ext; simp [Finset.prod_erase_attach (g · x), ← u.sum_attach])


theorem HasFDerivAt.finset_prod [DecidableEq ι] {x : E}
    (hg : ∀ i ∈ u, HasFDerivAt (g i) (g' i) x) :
    HasFDerivAt (∏ i ∈ u, g i ·) (∑ i ∈ u, (∏ j ∈ u.erase i, g j x) • g' i) x := by
  simpa [← Finset.prod_attach u] using .congr_fderiv
    (hasFDerivAt_finset_prod.comp x <| hasFDerivAt_pi.mpr fun i ↦ hg (Subtype.val i) i.prop :)
    (by ext; simp [Finset.prod_erase_attach (g · x), ← u.sum_attach])


theorem HasFDerivWithinAt.finset_prod [DecidableEq ι] {x : E}
    (hg : ∀ i ∈ u, HasFDerivWithinAt (g i) (g' i) s x) :
    HasFDerivWithinAt (∏ i ∈ u, g i ·) (∑ i ∈ u, (∏ j ∈ u.erase i, g j x) • g' i) s x := by
  simpa [← Finset.prod_attach u] using .congr_fderiv
    (hasFDerivAt_finset_prod.comp_hasFDerivWithinAt x <|
      hasFDerivWithinAt_pi.mpr fun i ↦ hg (Subtype.val i) i.prop :)
    (by ext; simp [Finset.prod_erase_attach (g · x), ← u.sum_attach])


theorem fderiv_finset_prod [DecidableEq ι] {x : E} (hg : ∀ i ∈ u, DifferentiableAt 𝕜 (g i) x) :
    fderiv 𝕜 (∏ i ∈ u, g i ·) x = ∑ i ∈ u, (∏ j ∈ u.erase i, (g j x)) • fderiv 𝕜 (g i) x :=
  (HasFDerivAt.finset_prod fun i hi ↦ (hg i hi).hasFDerivAt).fderiv


theorem fderivWithin_finset_prod [DecidableEq ι] {x : E} (hxs : UniqueDiffWithinAt 𝕜 s x)
    (hg : ∀ i ∈ u, DifferentiableWithinAt 𝕜 (g i) s x) :
    fderivWithin 𝕜 (∏ i ∈ u, g i ·) s x =
      ∑ i ∈ u, (∏ j ∈ u.erase i, (g j x)) • fderivWithin 𝕜 (g i) s x :=
  (HasFDerivWithinAt.finset_prod fun i hi ↦ (hg i hi).hasFDerivWithinAt).fderivWithin hxs


/-- At an invertible element `x` of a normed algebra `R`, the Fréchet derivative of the inversion
operation is the linear map `fun t ↦ - x⁻¹ * t * x⁻¹`.

TODO (low prio): prove a version without assumption `[HasSummableGeomSeries R]` but within the set
of units. -/
@[fun_prop]
theorem hasFDerivAt_ring_inverse (x : Rˣ) :
    HasFDerivAt Ring.inverse (-mulLeftRight 𝕜 R ↑x⁻¹ ↑x⁻¹) x :=
  have : (fun t : R => Ring.inverse (↑x + t) - ↑x⁻¹ + ↑x⁻¹ * t * ↑x⁻¹) =o[𝓝 0] id :=
    (inverse_add_norm_diff_second_order x).trans_isLittleO (isLittleO_norm_pow_id one_lt_two)
     /-
       𝕜 : Type u_1
       inst✝³ : NontriviallyNormedField 𝕜
       R : Type u_5
       inst✝² : NormedRing R
       inst✝¹ : HasSummableGeomSeries R
       inst✝ : NormedAlgebra 𝕜 R
       x : Units R
       this : Asymptotics.IsLittleO (nhds 0) (fun t => HAdd.hAdd (HSub.hSub (Ring.inv …
       ⊢ HasFDerivAt Ring.inverse (Neg.neg (((ContinuousLinearMap.mulLeftRight 𝕜 R) ↑ …
     -/
  by simpa [hasFDerivAt_iff_isLittleO_nhds_zero] using this
     /-
       🎉 no goals
     -/


@[fun_prop]
theorem differentiableAt_inverse {x : R} (hx : IsUnit x) :
    DifferentiableAt 𝕜 (@Ring.inverse R _) x :=
  let ⟨u, hu⟩ := hx; hu ▸ (hasFDerivAt_ring_inverse u).differentiableAt


@[fun_prop]
theorem differentiableWithinAt_inverse {x : R} (hx : IsUnit x) (s : Set R) :
    DifferentiableWithinAt 𝕜 (@Ring.inverse R _) s x :=
  (differentiableAt_inverse hx).differentiableWithinAt


@[fun_prop]
theorem differentiableOn_inverse : DifferentiableOn 𝕜 (@Ring.inverse R _) {x | IsUnit x} :=
  fun _x hx => differentiableWithinAt_inverse hx _


theorem fderiv_inverse (x : Rˣ) : fderiv 𝕜 (@Ring.inverse R _) x = -mulLeftRight 𝕜 R ↑x⁻¹ ↑x⁻¹ :=
  (hasFDerivAt_ring_inverse x).fderiv


theorem hasStrictFDerivAt_ring_inverse (x : Rˣ) :
    HasStrictFDerivAt Ring.inverse (-mulLeftRight 𝕜 R ↑x⁻¹ ↑x⁻¹) x := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    R : Type u_5
    inst✝² : NormedRing R
    inst✝¹ : HasSummableGeomSeries R
    inst✝ : NormedAlgebra 𝕜 R
    x : Units R
    ⊢ HasStrictFDerivAt Ring.inverse (Neg.neg (((ContinuousLinearMap.mulLeftRight  …
  -/
  convert (analyticAt_inverse (𝕜 := 𝕜) x).hasStrictFDerivAt
  /-
    case h.e'_12.h.h.h
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    R : Type u_5
    inst✝² : NormedRing R
    inst✝¹ : HasSummableGeomSeries R
    inst✝ : NormedAlgebra 𝕜 R
    x : Units R
    e_4✝ : Eq SeminormedAddCommGroup.toAddCommGroup NormedAddCommGroup.toAddCommGr …
    he✝ : Eq NormedSpace.toModule NormedSpace.toModule
    ⊢ Eq (Neg.neg (((ContinuousLinearMap.mulLeftRight 𝕜 R) ↑(Inv.inv x)) ↑(Inv.inv …
  -/
  exact (fderiv_inverse x).symm
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem DifferentiableWithinAt.inverse (hf : DifferentiableWithinAt 𝕜 h S z) (hz : IsUnit (h z)) :
    DifferentiableWithinAt 𝕜 (fun x => Ring.inverse (h x)) S z :=
  (differentiableAt_inverse hz).comp_differentiableWithinAt z hf


@[simp, fun_prop]
theorem DifferentiableAt.inverse (hf : DifferentiableAt 𝕜 h z) (hz : IsUnit (h z)) :
    DifferentiableAt 𝕜 (fun x => Ring.inverse (h x)) z :=
  (differentiableAt_inverse hz).comp z hf


@[fun_prop]
theorem DifferentiableOn.inverse (hf : DifferentiableOn 𝕜 h S) (hz : ∀ x ∈ S, IsUnit (h x)) :
    DifferentiableOn 𝕜 (fun x => Ring.inverse (h x)) S := fun x h => (hf x h).inverse (hz x h)


@[simp, fun_prop]
theorem Differentiable.inverse (hf : Differentiable 𝕜 h) (hz : ∀ x, IsUnit (h x)) :
    Differentiable 𝕜 fun x => Ring.inverse (h x) := fun x => (hf x).inverse (hz x)


/-- At an invertible element `x` of a normed division algebra `R`, the inversion is strictly
differentiable, with derivative the linear map `fun t ↦ - x⁻¹ * t * x⁻¹`. For a nicer formula in
the commutative case, see `hasStrictFDerivAt_inv`. -/
theorem hasStrictFDerivAt_inv' {x : R} (hx : x ≠ 0) :
    HasStrictFDerivAt Inv.inv (-mulLeftRight 𝕜 R x⁻¹ x⁻¹) x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    R : Type u_5
    inst✝¹ : NormedDivisionRing R
    inst✝ : NormedAlgebra 𝕜 R
    x : R
    hx : Ne x 0
    ⊢ HasStrictFDerivAt Inv.inv (Neg.neg (((ContinuousLinearMap.mulLeftRight 𝕜 R)  …
  -/
  simpa using hasStrictFDerivAt_ring_inverse (Units.mk0 _ hx)
  /-
    🎉 no goals
  -/


/-- At an invertible element `x` of a normed division algebra `R`, the Fréchet derivative of the
inversion operation is the linear map `fun t ↦ - x⁻¹ * t * x⁻¹`. For a nicer formula in the
commutative case, see `hasFDerivAt_inv`. -/
@[fun_prop]
theorem hasFDerivAt_inv' {x : R} (hx : x ≠ 0) :
    HasFDerivAt Inv.inv (-mulLeftRight 𝕜 R x⁻¹ x⁻¹) x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    R : Type u_5
    inst✝¹ : NormedDivisionRing R
    inst✝ : NormedAlgebra 𝕜 R
    x : R
    hx : Ne x 0
    ⊢ HasFDerivAt Inv.inv (Neg.neg (((ContinuousLinearMap.mulLeftRight 𝕜 R) (Inv.i …
  -/
  simpa using hasFDerivAt_ring_inverse (Units.mk0 _ hx)
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem differentiableAt_inv {x : R} (hx : x ≠ 0) : DifferentiableAt 𝕜 Inv.inv x :=
  (hasFDerivAt_inv' hx).differentiableAt


@[deprecated (since := "2024-09-21")] alias differentiableAt_inv' := differentiableAt_inv


@[fun_prop]
theorem differentiableWithinAt_inv {x : R} (hx : x ≠ 0) (s : Set R) :
    DifferentiableWithinAt 𝕜 (fun x => x⁻¹) s x :=
  (differentiableAt_inv hx).differentiableWithinAt


@[deprecated (since := "2024-09-21")]
alias differentiableWithinAt_inv' := differentiableWithinAt_inv


@[fun_prop]
theorem differentiableOn_inv : DifferentiableOn 𝕜 (fun x : R => x⁻¹) {x | x ≠ 0} := fun _x hx =>
  differentiableWithinAt_inv hx _


@[deprecated (since := "2024-09-21")] alias differentiableOn_inv' := differentiableOn_inv


/-- Non-commutative version of `fderiv_inv` -/
theorem fderiv_inv' {x : R} (hx : x ≠ 0) : fderiv 𝕜 Inv.inv x = -mulLeftRight 𝕜 R x⁻¹ x⁻¹ :=
  (hasFDerivAt_inv' hx).fderiv


/-- Non-commutative version of `fderivWithin_inv` -/
theorem fderivWithin_inv' {s : Set R} {x : R} (hx : x ≠ 0) (hxs : UniqueDiffWithinAt 𝕜 s x) :
    fderivWithin 𝕜 (fun x => x⁻¹) s x = -mulLeftRight 𝕜 R x⁻¹ x⁻¹ := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    R : Type u_5
    inst✝¹ : NormedDivisionRing R
    inst✝ : NormedAlgebra 𝕜 R
    s : Set R
    x : R
    hx : Ne x 0
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (fderivWithin 𝕜 (fun x => Inv.inv x) s x) (Neg.neg (((ContinuousLinearMap …
  -/
  rw [DifferentiableAt.fderivWithin (differentiableAt_inv hx) hxs]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    R : Type u_5
    inst✝¹ : NormedDivisionRing R
    inst✝ : NormedAlgebra 𝕜 R
    s : Set R
    x : R
    hx : Ne x 0
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (fderiv 𝕜 Inv.inv x) (Neg.neg (((ContinuousLinearMap.mulLeftRight 𝕜 R) (I …
  -/
  exact fderiv_inv' hx
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem DifferentiableWithinAt.inv (hf : DifferentiableWithinAt 𝕜 h S z) (hz : h z ≠ 0) :
    DifferentiableWithinAt 𝕜 (fun x => (h x)⁻¹) S z :=
  (differentiableAt_inv hz).comp_differentiableWithinAt z hf


@[deprecated (since := "2024-09-21")]
alias DifferentiableWithinAt.inv' := DifferentiableWithinAt.inv


@[simp, fun_prop]
theorem DifferentiableAt.inv (hf : DifferentiableAt 𝕜 h z) (hz : h z ≠ 0) :
    DifferentiableAt 𝕜 (fun x => (h x)⁻¹) z :=
  (differentiableAt_inv hz).comp z hf


@[deprecated (since := "2024-09-21")] alias DifferentiableAt.inv' := DifferentiableAt.inv


@[fun_prop]
theorem DifferentiableOn.inv (hf : DifferentiableOn 𝕜 h S) (hz : ∀ x ∈ S, h x ≠ 0) :
    DifferentiableOn 𝕜 (fun x => (h x)⁻¹) S := fun x h => (hf x h).inv (hz x h)


@[deprecated (since := "2024-09-21")] alias DifferentiableOn.inv' := DifferentiableOn.inv


@[simp, fun_prop]
theorem Differentiable.inv (hf : Differentiable 𝕜 h) (hz : ∀ x, h x ≠ 0) :
    Differentiable 𝕜 fun x => (h x)⁻¹ := fun x => (hf x).inv (hz x)


@[deprecated (since := "2024-09-21")] alias Differentiable.inv' := Differentiable.inv


