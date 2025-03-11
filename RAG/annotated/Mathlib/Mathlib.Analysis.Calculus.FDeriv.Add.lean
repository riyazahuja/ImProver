@[fun_prop]
theorem HasStrictFDerivAt.const_smul (h : HasStrictFDerivAt f f' x) (c : R) :
    HasStrictFDerivAt (fun x => c • f x) (c • f') x :=
  (c • (1 : F →L[𝕜] F)).hasStrictFDerivAt.comp x h


theorem HasFDerivAtFilter.const_smul (h : HasFDerivAtFilter f f' x L) (c : R) :
    HasFDerivAtFilter (fun x => c • f x) (c • f') x L :=
  (c • (1 : F →L[𝕜] F)).hasFDerivAtFilter.comp x h tendsto_map


@[fun_prop]
nonrec theorem HasFDerivWithinAt.const_smul (h : HasFDerivWithinAt f f' s x) (c : R) :
    HasFDerivWithinAt (fun x => c • f x) (c • f') s x :=
  h.const_smul c


@[fun_prop]
nonrec theorem HasFDerivAt.const_smul (h : HasFDerivAt f f' x) (c : R) :
    HasFDerivAt (fun x => c • f x) (c • f') x :=
  h.const_smul c


@[fun_prop]
theorem DifferentiableWithinAt.const_smul (h : DifferentiableWithinAt 𝕜 f s x) (c : R) :
    DifferentiableWithinAt 𝕜 (fun y => c • f y) s x :=
  (h.hasFDerivWithinAt.const_smul c).differentiableWithinAt


@[fun_prop]
theorem DifferentiableAt.const_smul (h : DifferentiableAt 𝕜 f x) (c : R) :
    DifferentiableAt 𝕜 (fun y => c • f y) x :=
  (h.hasFDerivAt.const_smul c).differentiableAt


@[fun_prop]
theorem DifferentiableOn.const_smul (h : DifferentiableOn 𝕜 f s) (c : R) :
    DifferentiableOn 𝕜 (fun y => c • f y) s := fun x hx => (h x hx).const_smul c


@[fun_prop]
theorem Differentiable.const_smul (h : Differentiable 𝕜 f) (c : R) :
    Differentiable 𝕜 fun y => c • f y := fun x => (h x).const_smul c


theorem fderivWithin_const_smul (hxs : UniqueDiffWithinAt 𝕜 s x)
    (h : DifferentiableWithinAt 𝕜 f s x) (c : R) :
    fderivWithin 𝕜 (fun y => c • f y) s x = c • fderivWithin 𝕜 f s x :=
  (h.hasFDerivWithinAt.const_smul c).fderivWithin hxs


/-- Version of `fderivWithin_const_smul` written with `c • f` instead of `fun y ↦ c • f y`. -/
theorem fderivWithin_const_smul' (hxs : UniqueDiffWithinAt 𝕜 s x)
    (h : DifferentiableWithinAt 𝕜 f s x) (c : R) :
    fderivWithin 𝕜 (c • f) s x = c • fderivWithin 𝕜 f s x :=
  fderivWithin_const_smul hxs h c


theorem fderiv_const_smul (h : DifferentiableAt 𝕜 f x) (c : R) :
    fderiv 𝕜 (fun y => c • f y) x = c • fderiv 𝕜 f x :=
  (h.hasFDerivAt.const_smul c).fderiv


/-- Version of `fderiv_const_smul` written with `c • f` instead of `fun y ↦ c • f y`. -/
theorem fderiv_const_smul' (h : DifferentiableAt 𝕜 f x) (c : R) :
    fderiv 𝕜 (c • f) x = c • fderiv 𝕜 f x :=
  (h.hasFDerivAt.const_smul c).fderiv


@[fun_prop]
nonrec theorem HasStrictFDerivAt.add (hf : HasStrictFDerivAt f f' x)
    (hg : HasStrictFDerivAt g g' x) : HasStrictFDerivAt (fun y => f y + g y) (f' + g') x :=
   .of_isLittleO <| (hf.isLittleO.add hg.isLittleO).congr_left fun y => by
    /-
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f g : E → F
      f' g' : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : E
      hf : HasStrictFDerivAt f f' x
      hg : HasStrictFDerivAt g g' x
      y : Prod E E
      ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (f y.1) (f y.2)) (f' (HSub.hSub y.1 y.2) …
    -/
    simp only [LinearMap.sub_apply, LinearMap.add_apply, map_sub, map_add, add_apply]
    /-
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f g : E → F
      f' g' : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : E
      hf : HasStrictFDerivAt f f' x
      hg : HasStrictFDerivAt g g' x
      y : Prod E E
      ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (f y.1) (f y.2)) (HSub.hSub (f' y.1) (f' …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/


theorem HasFDerivAtFilter.add (hf : HasFDerivAtFilter f f' x L)
    (hg : HasFDerivAtFilter g g' x L) : HasFDerivAtFilter (fun y => f y + g y) (f' + g') x L :=
  .of_isLittleO <| (hf.isLittleO.add hg.isLittleO).congr_left fun _ => by
    /-
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f g : E → F
      f' g' : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : E
      L : Filter E
      hf : HasFDerivAtFilter f f' x L
      hg : HasFDerivAtFilter g g' x L
      x✝ : E
      ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (f x✝) (f x)) (f' (HSub.hSub x✝ x))) (HS …
    -/
    simp only [LinearMap.sub_apply, LinearMap.add_apply, map_sub, map_add, add_apply]
    /-
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f g : E → F
      f' g' : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : E
      L : Filter E
      hf : HasFDerivAtFilter f f' x L
      hg : HasFDerivAtFilter g g' x L
      x✝ : E
      ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (f x✝) (f x)) (HSub.hSub (f' x✝) (f' x)) …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/


@[fun_prop]
nonrec theorem HasFDerivWithinAt.add (hf : HasFDerivWithinAt f f' s x)
    (hg : HasFDerivWithinAt g g' s x) : HasFDerivWithinAt (fun y => f y + g y) (f' + g') s x :=
  hf.add hg


@[fun_prop]
nonrec theorem HasFDerivAt.add (hf : HasFDerivAt f f' x) (hg : HasFDerivAt g g' x) :
    HasFDerivAt (fun x => f x + g x) (f' + g') x :=
  hf.add hg


@[fun_prop]
theorem DifferentiableWithinAt.add (hf : DifferentiableWithinAt 𝕜 f s x)
    (hg : DifferentiableWithinAt 𝕜 g s x) : DifferentiableWithinAt 𝕜 (fun y => f y + g y) s x :=
  (hf.hasFDerivWithinAt.add hg.hasFDerivWithinAt).differentiableWithinAt


@[simp, fun_prop]
theorem DifferentiableAt.add (hf : DifferentiableAt 𝕜 f x) (hg : DifferentiableAt 𝕜 g x) :
    DifferentiableAt 𝕜 (fun y => f y + g y) x :=
  (hf.hasFDerivAt.add hg.hasFDerivAt).differentiableAt


@[fun_prop]
theorem DifferentiableOn.add (hf : DifferentiableOn 𝕜 f s) (hg : DifferentiableOn 𝕜 g s) :
    DifferentiableOn 𝕜 (fun y => f y + g y) s := fun x hx => (hf x hx).add (hg x hx)


@[simp, fun_prop]
theorem Differentiable.add (hf : Differentiable 𝕜 f) (hg : Differentiable 𝕜 g) :
    Differentiable 𝕜 fun y => f y + g y := fun x => (hf x).add (hg x)


theorem fderivWithin_add (hxs : UniqueDiffWithinAt 𝕜 s x) (hf : DifferentiableWithinAt 𝕜 f s x)
    (hg : DifferentiableWithinAt 𝕜 g s x) :
    fderivWithin 𝕜 (fun y => f y + g y) s x = fderivWithin 𝕜 f s x + fderivWithin 𝕜 g s x :=
  (hf.hasFDerivWithinAt.add hg.hasFDerivWithinAt).fderivWithin hxs


/-- Version of `fderivWithin_add` where the function is written as `f + g` instead
of `fun y ↦ f y + g y`. -/
theorem fderivWithin_add' (hxs : UniqueDiffWithinAt 𝕜 s x) (hf : DifferentiableWithinAt 𝕜 f s x)
    (hg : DifferentiableWithinAt 𝕜 g s x) :
    fderivWithin 𝕜 (f + g) s x = fderivWithin 𝕜 f s x + fderivWithin 𝕜 g s x :=
  fderivWithin_add hxs hf hg


theorem fderiv_add (hf : DifferentiableAt 𝕜 f x) (hg : DifferentiableAt 𝕜 g x) :
    fderiv 𝕜 (fun y => f y + g y) x = fderiv 𝕜 f x + fderiv 𝕜 g x :=
  (hf.hasFDerivAt.add hg.hasFDerivAt).fderiv


/-- Version of `fderiv_add` where the function is written as `f + g` instead
of `fun y ↦ f y + g y`. -/
theorem fderiv_add' (hf : DifferentiableAt 𝕜 f x) (hg : DifferentiableAt 𝕜 g x) :
    fderiv 𝕜 (f + g) x = fderiv 𝕜 f x + fderiv 𝕜 g x :=
  fderiv_add hf hg


@[fun_prop]
theorem HasStrictFDerivAt.add_const (hf : HasStrictFDerivAt f f' x) (c : F) :
    HasStrictFDerivAt (fun y => f y + c) f' x :=
  add_zero f' ▸ hf.add (hasStrictFDerivAt_const _ _)


theorem HasFDerivAtFilter.add_const (hf : HasFDerivAtFilter f f' x L) (c : F) :
    HasFDerivAtFilter (fun y => f y + c) f' x L :=
  add_zero f' ▸ hf.add (hasFDerivAtFilter_const _ _ _)


@[fun_prop]
nonrec theorem HasFDerivWithinAt.add_const (hf : HasFDerivWithinAt f f' s x) (c : F) :
    HasFDerivWithinAt (fun y => f y + c) f' s x :=
  hf.add_const c


@[fun_prop]
nonrec theorem HasFDerivAt.add_const (hf : HasFDerivAt f f' x) (c : F) :
    HasFDerivAt (fun x => f x + c) f' x :=
  hf.add_const c


@[fun_prop]
theorem DifferentiableWithinAt.add_const (hf : DifferentiableWithinAt 𝕜 f s x) (c : F) :
    DifferentiableWithinAt 𝕜 (fun y => f y + c) s x :=
  (hf.hasFDerivWithinAt.add_const c).differentiableWithinAt


@[simp]
theorem differentiableWithinAt_add_const_iff (c : F) :
    DifferentiableWithinAt 𝕜 (fun y => f y + c) s x ↔ DifferentiableWithinAt 𝕜 f s x :=
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
                 c : F
                 h : DifferentiableWithinAt 𝕜 (fun y => HAdd.hAdd (f y) c) s x
                 ⊢ DifferentiableWithinAt 𝕜 f s x
               -/
  ⟨fun h => by simpa using h.add_const (-c), fun h => h.add_const c⟩
               /-
                 🎉 no goals
               -/


@[fun_prop]
theorem DifferentiableAt.add_const (hf : DifferentiableAt 𝕜 f x) (c : F) :
    DifferentiableAt 𝕜 (fun y => f y + c) x :=
  (hf.hasFDerivAt.add_const c).differentiableAt


@[simp]
theorem differentiableAt_add_const_iff (c : F) :
    DifferentiableAt 𝕜 (fun y => f y + c) x ↔ DifferentiableAt 𝕜 f x :=
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
                 c : F
                 h : DifferentiableAt 𝕜 (fun y => HAdd.hAdd (f y) c) x
                 ⊢ DifferentiableAt 𝕜 f x
               -/
  ⟨fun h => by simpa using h.add_const (-c), fun h => h.add_const c⟩
               /-
                 🎉 no goals
               -/


@[fun_prop]
theorem DifferentiableOn.add_const (hf : DifferentiableOn 𝕜 f s) (c : F) :
    DifferentiableOn 𝕜 (fun y => f y + c) s := fun x hx => (hf x hx).add_const c


@[simp]
theorem differentiableOn_add_const_iff (c : F) :
    DifferentiableOn 𝕜 (fun y => f y + c) s ↔ DifferentiableOn 𝕜 f s :=
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
                 c : F
                 h : DifferentiableOn 𝕜 (fun y => HAdd.hAdd (f y) c) s
                 ⊢ DifferentiableOn 𝕜 f s
               -/
  ⟨fun h => by simpa using h.add_const (-c), fun h => h.add_const c⟩
               /-
                 🎉 no goals
               -/


@[fun_prop]
theorem Differentiable.add_const (hf : Differentiable 𝕜 f) (c : F) :
    Differentiable 𝕜 fun y => f y + c := fun x => (hf x).add_const c


@[simp]
theorem differentiable_add_const_iff (c : F) :
    (Differentiable 𝕜 fun y => f y + c) ↔ Differentiable 𝕜 f :=
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
                 c : F
                 h : Differentiable 𝕜 fun y => HAdd.hAdd (f y) c
                 ⊢ Differentiable 𝕜 f
               -/
  ⟨fun h => by simpa using h.add_const (-c), fun h => h.add_const c⟩
               /-
                 🎉 no goals
               -/


theorem fderivWithin_add_const (hxs : UniqueDiffWithinAt 𝕜 s x) (c : F) :
    fderivWithin 𝕜 (fun y => f y + c) s x = fderivWithin 𝕜 f s x := by
  classical
  by_cases hf : DifferentiableWithinAt 𝕜 f s x
  · exact (hf.hasFDerivWithinAt.add_const c).fderivWithin hxs
  · rw [fderivWithin_zero_of_not_differentiableWithinAt hf,
      fderivWithin_zero_of_not_differentiableWithinAt]
    simpa


theorem fderiv_add_const (c : F) : fderiv 𝕜 (fun y => f y + c) x = fderiv 𝕜 f x := by
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
    c : F
    ⊢ Eq (fderiv 𝕜 (fun y => HAdd.hAdd (f y) c) x) (fderiv 𝕜 f x)
  -/
  simp only [← fderivWithin_univ, fderivWithin_add_const uniqueDiffWithinAt_univ]
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem HasStrictFDerivAt.const_add (hf : HasStrictFDerivAt f f' x) (c : F) :
    HasStrictFDerivAt (fun y => c + f y) f' x :=
  zero_add f' ▸ (hasStrictFDerivAt_const _ _).add hf


theorem HasFDerivAtFilter.const_add (hf : HasFDerivAtFilter f f' x L) (c : F) :
    HasFDerivAtFilter (fun y => c + f y) f' x L :=
  zero_add f' ▸ (hasFDerivAtFilter_const _ _ _).add hf


@[fun_prop]
nonrec theorem HasFDerivWithinAt.const_add (hf : HasFDerivWithinAt f f' s x) (c : F) :
    HasFDerivWithinAt (fun y => c + f y) f' s x :=
  hf.const_add c


@[fun_prop]
nonrec theorem HasFDerivAt.const_add (hf : HasFDerivAt f f' x) (c : F) :
    HasFDerivAt (fun x => c + f x) f' x :=
  hf.const_add c


@[fun_prop]
theorem DifferentiableWithinAt.const_add (hf : DifferentiableWithinAt 𝕜 f s x) (c : F) :
    DifferentiableWithinAt 𝕜 (fun y => c + f y) s x :=
  (hf.hasFDerivWithinAt.const_add c).differentiableWithinAt


@[simp]
theorem differentiableWithinAt_const_add_iff (c : F) :
    DifferentiableWithinAt 𝕜 (fun y => c + f y) s x ↔ DifferentiableWithinAt 𝕜 f s x :=
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
                 c : F
                 h : DifferentiableWithinAt 𝕜 (fun y => HAdd.hAdd c (f y)) s x
                 ⊢ DifferentiableWithinAt 𝕜 f s x
               -/
  ⟨fun h => by simpa using h.const_add (-c), fun h => h.const_add c⟩
               /-
                 🎉 no goals
               -/


@[fun_prop]
theorem DifferentiableAt.const_add (hf : DifferentiableAt 𝕜 f x) (c : F) :
    DifferentiableAt 𝕜 (fun y => c + f y) x :=
  (hf.hasFDerivAt.const_add c).differentiableAt


@[simp]
theorem differentiableAt_const_add_iff (c : F) :
    DifferentiableAt 𝕜 (fun y => c + f y) x ↔ DifferentiableAt 𝕜 f x :=
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
                 c : F
                 h : DifferentiableAt 𝕜 (fun y => HAdd.hAdd c (f y)) x
                 ⊢ DifferentiableAt 𝕜 f x
               -/
  ⟨fun h => by simpa using h.const_add (-c), fun h => h.const_add c⟩
               /-
                 🎉 no goals
               -/


@[fun_prop]
theorem DifferentiableOn.const_add (hf : DifferentiableOn 𝕜 f s) (c : F) :
    DifferentiableOn 𝕜 (fun y => c + f y) s := fun x hx => (hf x hx).const_add c


@[simp]
theorem differentiableOn_const_add_iff (c : F) :
    DifferentiableOn 𝕜 (fun y => c + f y) s ↔ DifferentiableOn 𝕜 f s :=
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
                 c : F
                 h : DifferentiableOn 𝕜 (fun y => HAdd.hAdd c (f y)) s
                 ⊢ DifferentiableOn 𝕜 f s
               -/
  ⟨fun h => by simpa using h.const_add (-c), fun h => h.const_add c⟩
               /-
                 🎉 no goals
               -/


@[fun_prop]
theorem Differentiable.const_add (hf : Differentiable 𝕜 f) (c : F) :
    Differentiable 𝕜 fun y => c + f y := fun x => (hf x).const_add c


@[simp]
theorem differentiable_const_add_iff (c : F) :
    (Differentiable 𝕜 fun y => c + f y) ↔ Differentiable 𝕜 f :=
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
                 c : F
                 h : Differentiable 𝕜 fun y => HAdd.hAdd c (f y)
                 ⊢ Differentiable 𝕜 f
               -/
  ⟨fun h => by simpa using h.const_add (-c), fun h => h.const_add c⟩
               /-
                 🎉 no goals
               -/


theorem fderivWithin_const_add (hxs : UniqueDiffWithinAt 𝕜 s x) (c : F) :
    fderivWithin 𝕜 (fun y => c + f y) s x = fderivWithin 𝕜 f s x := by
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
    hxs : UniqueDiffWithinAt 𝕜 s x
    c : F
    ⊢ Eq (fderivWithin 𝕜 (fun y => HAdd.hAdd c (f y)) s x) (fderivWithin 𝕜 f s x)
  -/
  simpa only [add_comm] using fderivWithin_add_const hxs c
  /-
    🎉 no goals
  -/


theorem fderiv_const_add (c : F) : fderiv 𝕜 (fun y => c + f y) x = fderiv 𝕜 f x := by
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
    c : F
    ⊢ Eq (fderiv 𝕜 (fun y => HAdd.hAdd c (f y)) x) (fderiv 𝕜 f x)
  -/
  simp only [add_comm c, fderiv_add_const]
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem HasStrictFDerivAt.sum (h : ∀ i ∈ u, HasStrictFDerivAt (A i) (A' i) x) :
    HasStrictFDerivAt (fun y => ∑ i ∈ u, A i y) (∑ i ∈ u, A' i) x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : E
    ι : Type u_4
    u : Finset ι
    A : ι → E → F
    A' : ι → ContinuousLinearMap (RingHom.id 𝕜) E F
    h : ∀ (i : ι), Membership.mem u i → HasStrictFDerivAt (A i) (A' i) x
    ⊢ HasStrictFDerivAt (fun y => u.sum fun i => A i y) (u.sum fun i => A' i) x
  -/
  simp only [hasStrictFDerivAt_iff_isLittleO] at *
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : E
    ι : Type u_4
    u : Finset ι
    A : ι → E → F
    A' : ι → ContinuousLinearMap (RingHom.id 𝕜) E F
    h : ∀ (i : ι), Membership.mem u i → Asymptotics.IsLittleO (nhds { fst := x, sn …
    ⊢ Asymptotics.IsLittleO (nhds { fst := x, snd := x }) (fun p => HSub.hSub (HSu …
  -/
  convert IsLittleO.sum h
  /-
    case h.e'_7.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : E
    ι : Type u_4
    u : Finset ι
    A : ι → E → F
    A' : ι → ContinuousLinearMap (RingHom.id 𝕜) E F
    h : ∀ (i : ι), Membership.mem u i → Asymptotics.IsLittleO (nhds { fst := x, sn …
    x✝ : Prod E E
    ⊢ Eq (HSub.hSub (HSub.hSub (u.sum fun i => A i x✝.1) (u.sum fun i => A i x✝.2) …
  -/
  simp [Finset.sum_sub_distrib, ContinuousLinearMap.sum_apply]
  /-
    🎉 no goals
  -/


theorem HasFDerivAtFilter.sum (h : ∀ i ∈ u, HasFDerivAtFilter (A i) (A' i) x L) :
    HasFDerivAtFilter (fun y => ∑ i ∈ u, A i y) (∑ i ∈ u, A' i) x L := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : E
    L : Filter E
    ι : Type u_4
    u : Finset ι
    A : ι → E → F
    A' : ι → ContinuousLinearMap (RingHom.id 𝕜) E F
    h : ∀ (i : ι), Membership.mem u i → HasFDerivAtFilter (A i) (A' i) x L
    ⊢ HasFDerivAtFilter (fun y => u.sum fun i => A i y) (u.sum fun i => A' i) x L
  -/
  simp only [hasFDerivAtFilter_iff_isLittleO] at *
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : E
    L : Filter E
    ι : Type u_4
    u : Finset ι
    A : ι → E → F
    A' : ι → ContinuousLinearMap (RingHom.id 𝕜) E F
    h : ∀ (i : ι), Membership.mem u i → Asymptotics.IsLittleO L (fun x' => HSub.hS …
    ⊢ Asymptotics.IsLittleO L (fun x' => HSub.hSub (HSub.hSub (u.sum fun i => A i  …
  -/
  convert IsLittleO.sum h
  /-
    case h.e'_7.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : E
    L : Filter E
    ι : Type u_4
    u : Finset ι
    A : ι → E → F
    A' : ι → ContinuousLinearMap (RingHom.id 𝕜) E F
    h : ∀ (i : ι), Membership.mem u i → Asymptotics.IsLittleO L (fun x' => HSub.hS …
    x✝ : E
    ⊢ Eq (HSub.hSub (HSub.hSub (u.sum fun i => A i x✝) (u.sum fun i => A i x)) ((u …
  -/
  simp [ContinuousLinearMap.sum_apply]
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem HasFDerivWithinAt.sum (h : ∀ i ∈ u, HasFDerivWithinAt (A i) (A' i) s x) :
    HasFDerivWithinAt (fun y => ∑ i ∈ u, A i y) (∑ i ∈ u, A' i) s x :=
  HasFDerivAtFilter.sum h


@[fun_prop]
theorem HasFDerivAt.sum (h : ∀ i ∈ u, HasFDerivAt (A i) (A' i) x) :
    HasFDerivAt (fun y => ∑ i ∈ u, A i y) (∑ i ∈ u, A' i) x :=
  HasFDerivAtFilter.sum h


@[fun_prop]
theorem DifferentiableWithinAt.sum (h : ∀ i ∈ u, DifferentiableWithinAt 𝕜 (A i) s x) :
    DifferentiableWithinAt 𝕜 (fun y => ∑ i ∈ u, A i y) s x :=
  HasFDerivWithinAt.differentiableWithinAt <|
    HasFDerivWithinAt.sum fun i hi => (h i hi).hasFDerivWithinAt


@[simp, fun_prop]
theorem DifferentiableAt.sum (h : ∀ i ∈ u, DifferentiableAt 𝕜 (A i) x) :
    DifferentiableAt 𝕜 (fun y => ∑ i ∈ u, A i y) x :=
  HasFDerivAt.differentiableAt <| HasFDerivAt.sum fun i hi => (h i hi).hasFDerivAt


@[fun_prop]
theorem DifferentiableOn.sum (h : ∀ i ∈ u, DifferentiableOn 𝕜 (A i) s) :
    DifferentiableOn 𝕜 (fun y => ∑ i ∈ u, A i y) s := fun x hx =>
  DifferentiableWithinAt.sum fun i hi => h i hi x hx


@[simp, fun_prop]
theorem Differentiable.sum (h : ∀ i ∈ u, Differentiable 𝕜 (A i)) :
    Differentiable 𝕜 fun y => ∑ i ∈ u, A i y := fun x => DifferentiableAt.sum fun i hi => h i hi x


theorem fderivWithin_sum (hxs : UniqueDiffWithinAt 𝕜 s x)
    (h : ∀ i ∈ u, DifferentiableWithinAt 𝕜 (A i) s x) :
    fderivWithin 𝕜 (fun y => ∑ i ∈ u, A i y) s x = ∑ i ∈ u, fderivWithin 𝕜 (A i) s x :=
  (HasFDerivWithinAt.sum fun i hi => (h i hi).hasFDerivWithinAt).fderivWithin hxs


theorem fderiv_sum (h : ∀ i ∈ u, DifferentiableAt 𝕜 (A i) x) :
    fderiv 𝕜 (fun y => ∑ i ∈ u, A i y) x = ∑ i ∈ u, fderiv 𝕜 (A i) x :=
  (HasFDerivAt.sum fun i hi => (h i hi).hasFDerivAt).fderiv


@[fun_prop]
theorem HasStrictFDerivAt.neg (h : HasStrictFDerivAt f f' x) :
    HasStrictFDerivAt (fun x => -f x) (-f') x :=
  (-1 : F →L[𝕜] F).hasStrictFDerivAt.comp x h


theorem HasFDerivAtFilter.neg (h : HasFDerivAtFilter f f' x L) :
    HasFDerivAtFilter (fun x => -f x) (-f') x L :=
  (-1 : F →L[𝕜] F).hasFDerivAtFilter.comp x h tendsto_map


@[fun_prop]
nonrec theorem HasFDerivWithinAt.neg (h : HasFDerivWithinAt f f' s x) :
    HasFDerivWithinAt (fun x => -f x) (-f') s x :=
  h.neg


@[fun_prop]
nonrec theorem HasFDerivAt.neg (h : HasFDerivAt f f' x) : HasFDerivAt (fun x => -f x) (-f') x :=
  h.neg


@[fun_prop]
theorem DifferentiableWithinAt.neg (h : DifferentiableWithinAt 𝕜 f s x) :
    DifferentiableWithinAt 𝕜 (fun y => -f y) s x :=
  h.hasFDerivWithinAt.neg.differentiableWithinAt


@[simp]
theorem differentiableWithinAt_neg_iff :
    DifferentiableWithinAt 𝕜 (fun y => -f y) s x ↔ DifferentiableWithinAt 𝕜 f s x :=
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
                 h : DifferentiableWithinAt 𝕜 (fun y => Neg.neg (f y)) s x
                 ⊢ DifferentiableWithinAt 𝕜 f s x
               -/
  ⟨fun h => by simpa only [neg_neg] using h.neg, fun h => h.neg⟩
               /-
                 🎉 no goals
               -/


@[fun_prop]
theorem DifferentiableAt.neg (h : DifferentiableAt 𝕜 f x) : DifferentiableAt 𝕜 (fun y => -f y) x :=
  h.hasFDerivAt.neg.differentiableAt


@[simp]
theorem differentiableAt_neg_iff : DifferentiableAt 𝕜 (fun y => -f y) x ↔ DifferentiableAt 𝕜 f x :=
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
                 h : DifferentiableAt 𝕜 (fun y => Neg.neg (f y)) x
                 ⊢ DifferentiableAt 𝕜 f x
               -/
  ⟨fun h => by simpa only [neg_neg] using h.neg, fun h => h.neg⟩
               /-
                 🎉 no goals
               -/


@[fun_prop]
theorem DifferentiableOn.neg (h : DifferentiableOn 𝕜 f s) : DifferentiableOn 𝕜 (fun y => -f y) s :=
  fun x hx => (h x hx).neg


@[simp]
theorem differentiableOn_neg_iff : DifferentiableOn 𝕜 (fun y => -f y) s ↔ DifferentiableOn 𝕜 f s :=
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
                 h : DifferentiableOn 𝕜 (fun y => Neg.neg (f y)) s
                 ⊢ DifferentiableOn 𝕜 f s
               -/
  ⟨fun h => by simpa only [neg_neg] using h.neg, fun h => h.neg⟩
               /-
                 🎉 no goals
               -/


@[fun_prop]
theorem Differentiable.neg (h : Differentiable 𝕜 f) : Differentiable 𝕜 fun y => -f y := fun x =>
  (h x).neg


@[simp]
theorem differentiable_neg_iff : (Differentiable 𝕜 fun y => -f y) ↔ Differentiable 𝕜 f :=
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
                 h : Differentiable 𝕜 fun y => Neg.neg (f y)
                 ⊢ Differentiable 𝕜 f
               -/
  ⟨fun h => by simpa only [neg_neg] using h.neg, fun h => h.neg⟩
               /-
                 🎉 no goals
               -/


theorem fderivWithin_neg (hxs : UniqueDiffWithinAt 𝕜 s x) :
    fderivWithin 𝕜 (fun y => -f y) s x = -fderivWithin 𝕜 f s x := by
  classical
  by_cases h : DifferentiableWithinAt 𝕜 f s x
  · exact h.hasFDerivWithinAt.neg.fderivWithin hxs
  · rw [fderivWithin_zero_of_not_differentiableWithinAt h,
      fderivWithin_zero_of_not_differentiableWithinAt, neg_zero]
    simpa


/-- Version of `fderivWithin_neg` where the function is written `-f` instead of `fun y ↦ - f y`. -/
theorem fderivWithin_neg' (hxs : UniqueDiffWithinAt 𝕜 s x) :
    fderivWithin 𝕜 (-f) s x = -fderivWithin 𝕜 f s x :=
  fderivWithin_neg hxs


@[simp]
theorem fderiv_neg : fderiv 𝕜 (fun y => -f y) x = -fderiv 𝕜 f x := by
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
    ⊢ Eq (fderiv 𝕜 (fun y => Neg.neg (f y)) x) (Neg.neg (fderiv 𝕜 f x))
  -/
  simp only [← fderivWithin_univ, fderivWithin_neg uniqueDiffWithinAt_univ]
  /-
    🎉 no goals
  -/


/-- Version of `fderiv_neg` where the function is written `-f` instead of `fun y ↦ - f y`. -/
theorem fderiv_neg' : fderiv 𝕜 (-f) x = -fderiv 𝕜 f x :=
  fderiv_neg


@[fun_prop]
theorem HasStrictFDerivAt.sub (hf : HasStrictFDerivAt f f' x) (hg : HasStrictFDerivAt g g' x) :
    HasStrictFDerivAt (fun x => f x - g x) (f' - g') x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    f' g' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    hf : HasStrictFDerivAt f f' x
    hg : HasStrictFDerivAt g g' x
    ⊢ HasStrictFDerivAt (fun x => HSub.hSub (f x) (g x)) (HSub.hSub f' g') x
  -/
  simpa only [sub_eq_add_neg] using hf.add hg.neg
  /-
    🎉 no goals
  -/


theorem HasFDerivAtFilter.sub (hf : HasFDerivAtFilter f f' x L) (hg : HasFDerivAtFilter g g' x L) :
    HasFDerivAtFilter (fun x => f x - g x) (f' - g') x L := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    f' g' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    L : Filter E
    hf : HasFDerivAtFilter f f' x L
    hg : HasFDerivAtFilter g g' x L
    ⊢ HasFDerivAtFilter (fun x => HSub.hSub (f x) (g x)) (HSub.hSub f' g') x L
  -/
  simpa only [sub_eq_add_neg] using hf.add hg.neg
  /-
    🎉 no goals
  -/


@[fun_prop]
nonrec theorem HasFDerivWithinAt.sub (hf : HasFDerivWithinAt f f' s x)
    (hg : HasFDerivWithinAt g g' s x) : HasFDerivWithinAt (fun x => f x - g x) (f' - g') s x :=
  hf.sub hg


@[fun_prop]
nonrec theorem HasFDerivAt.sub (hf : HasFDerivAt f f' x) (hg : HasFDerivAt g g' x) :
    HasFDerivAt (fun x => f x - g x) (f' - g') x :=
  hf.sub hg


@[fun_prop]
theorem DifferentiableWithinAt.sub (hf : DifferentiableWithinAt 𝕜 f s x)
    (hg : DifferentiableWithinAt 𝕜 g s x) : DifferentiableWithinAt 𝕜 (fun y => f y - g y) s x :=
  (hf.hasFDerivWithinAt.sub hg.hasFDerivWithinAt).differentiableWithinAt


@[simp, fun_prop]
theorem DifferentiableAt.sub (hf : DifferentiableAt 𝕜 f x) (hg : DifferentiableAt 𝕜 g x) :
    DifferentiableAt 𝕜 (fun y => f y - g y) x :=
  (hf.hasFDerivAt.sub hg.hasFDerivAt).differentiableAt


@[simp]
lemma DifferentiableAt.add_iff_left (hg : DifferentiableAt 𝕜 g x) :
    DifferentiableAt 𝕜 (fun y => f y + g y) x ↔ DifferentiableAt 𝕜 f x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    x : E
    hg : DifferentiableAt 𝕜 g x
    ⊢ Iff (DifferentiableAt 𝕜 (fun y => HAdd.hAdd (f y) (g y)) x) (DifferentiableA …
  -/
  refine ⟨fun h ↦ ?_, fun hf ↦ hf.add hg⟩
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    x : E
    hg : DifferentiableAt 𝕜 g x
    h : DifferentiableAt 𝕜 (fun y => HAdd.hAdd (f y) (g y)) x
    ⊢ DifferentiableAt 𝕜 f x
  -/
  simpa only [add_sub_cancel_right] using h.sub hg
  /-
    🎉 no goals
  -/


@[simp]
lemma DifferentiableAt.add_iff_right (hg : DifferentiableAt 𝕜 f x) :
    DifferentiableAt 𝕜 (fun y => f y + g y) x ↔ DifferentiableAt 𝕜 g x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    x : E
    hg : DifferentiableAt 𝕜 f x
    ⊢ Iff (DifferentiableAt 𝕜 (fun y => HAdd.hAdd (f y) (g y)) x) (DifferentiableA …
  -/
  simp only [add_comm (f _), hg.add_iff_left]
  /-
    🎉 no goals
  -/


@[simp]
lemma DifferentiableAt.sub_iff_left (hg : DifferentiableAt 𝕜 g x) :
    DifferentiableAt 𝕜 (fun y => f y - g y) x ↔ DifferentiableAt 𝕜 f x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    x : E
    hg : DifferentiableAt 𝕜 g x
    ⊢ Iff (DifferentiableAt 𝕜 (fun y => HSub.hSub (f y) (g y)) x) (DifferentiableA …
  -/
  simp only [sub_eq_add_neg, differentiableAt_neg_iff, hg, add_iff_left]
  /-
    🎉 no goals
  -/


@[simp]
lemma DifferentiableAt.sub_iff_right (hg : DifferentiableAt 𝕜 f x) :
    DifferentiableAt 𝕜 (fun y => f y - g y) x ↔ DifferentiableAt 𝕜 g x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    x : E
    hg : DifferentiableAt 𝕜 f x
    ⊢ Iff (DifferentiableAt 𝕜 (fun y => HSub.hSub (f y) (g y)) x) (DifferentiableA …
  -/
  simp only [sub_eq_add_neg, hg, add_iff_right, differentiableAt_neg_iff]
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem DifferentiableOn.sub (hf : DifferentiableOn 𝕜 f s) (hg : DifferentiableOn 𝕜 g s) :
    DifferentiableOn 𝕜 (fun y => f y - g y) s := fun x hx => (hf x hx).sub (hg x hx)


@[simp]
lemma DifferentiableOn.add_iff_left (hg : DifferentiableOn 𝕜 g s) :
    DifferentiableOn 𝕜 (fun y => f y + g y) s ↔ DifferentiableOn 𝕜 f s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    s : Set E
    hg : DifferentiableOn 𝕜 g s
    ⊢ Iff (DifferentiableOn 𝕜 (fun y => HAdd.hAdd (f y) (g y)) s) (DifferentiableO …
  -/
  refine ⟨fun h ↦ ?_, fun hf ↦ hf.add hg⟩
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    s : Set E
    hg : DifferentiableOn 𝕜 g s
    h : DifferentiableOn 𝕜 (fun y => HAdd.hAdd (f y) (g y)) s
    ⊢ DifferentiableOn 𝕜 f s
  -/
  simpa only [add_sub_cancel_right] using h.sub hg
  /-
    🎉 no goals
  -/


@[simp]
lemma DifferentiableOn.add_iff_right (hg : DifferentiableOn 𝕜 f s) :
    DifferentiableOn 𝕜 (fun y => f y + g y) s ↔ DifferentiableOn 𝕜 g s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    s : Set E
    hg : DifferentiableOn 𝕜 f s
    ⊢ Iff (DifferentiableOn 𝕜 (fun y => HAdd.hAdd (f y) (g y)) s) (DifferentiableO …
  -/
  simp only [add_comm (f _), hg.add_iff_left]
  /-
    🎉 no goals
  -/


@[simp]
lemma DifferentiableOn.sub_iff_left (hg : DifferentiableOn 𝕜 g s) :
    DifferentiableOn 𝕜 (fun y => f y - g y) s ↔ DifferentiableOn 𝕜 f s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    s : Set E
    hg : DifferentiableOn 𝕜 g s
    ⊢ Iff (DifferentiableOn 𝕜 (fun y => HSub.hSub (f y) (g y)) s) (DifferentiableO …
  -/
  simp only [sub_eq_add_neg, differentiableOn_neg_iff, hg, add_iff_left]
  /-
    🎉 no goals
  -/


@[simp]
lemma DifferentiableOn.sub_iff_right (hg : DifferentiableOn 𝕜 f s) :
    DifferentiableOn 𝕜 (fun y => f y - g y) s ↔ DifferentiableOn 𝕜 g s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    s : Set E
    hg : DifferentiableOn 𝕜 f s
    ⊢ Iff (DifferentiableOn 𝕜 (fun y => HSub.hSub (f y) (g y)) s) (DifferentiableO …
  -/
  simp only [sub_eq_add_neg, differentiableOn_neg_iff, hg, add_iff_right]
  /-
    🎉 no goals
  -/


@[simp, fun_prop]
theorem Differentiable.sub (hf : Differentiable 𝕜 f) (hg : Differentiable 𝕜 g) :
    Differentiable 𝕜 fun y => f y - g y := fun x => (hf x).sub (hg x)


@[simp]
lemma Differentiable.add_iff_left (hg : Differentiable 𝕜 g) :
    Differentiable 𝕜 (fun y => f y + g y) ↔ Differentiable 𝕜 f := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    hg : Differentiable 𝕜 g
    ⊢ Iff (Differentiable 𝕜 fun y => HAdd.hAdd (f y) (g y)) (Differentiable 𝕜 f)
  -/
  refine ⟨fun h ↦ ?_, fun hf ↦ hf.add hg⟩
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    hg : Differentiable 𝕜 g
    h : Differentiable 𝕜 fun y => HAdd.hAdd (f y) (g y)
    ⊢ Differentiable 𝕜 f
  -/
  simpa only [add_sub_cancel_right] using h.sub hg
  /-
    🎉 no goals
  -/


@[simp]
lemma Differentiable.add_iff_right (hg : Differentiable 𝕜 f) :
    Differentiable 𝕜 (fun y => f y + g y) ↔ Differentiable 𝕜 g := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    hg : Differentiable 𝕜 f
    ⊢ Iff (Differentiable 𝕜 fun y => HAdd.hAdd (f y) (g y)) (Differentiable 𝕜 g)
  -/
  simp only [add_comm (f _), hg.add_iff_left]
  /-
    🎉 no goals
  -/


@[simp]
lemma Differentiable.sub_iff_left (hg : Differentiable 𝕜 g) :
    Differentiable 𝕜 (fun y => f y - g y) ↔ Differentiable 𝕜 f := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    hg : Differentiable 𝕜 g
    ⊢ Iff (Differentiable 𝕜 fun y => HSub.hSub (f y) (g y)) (Differentiable 𝕜 f)
  -/
  simp only [sub_eq_add_neg, differentiable_neg_iff, hg, add_iff_left]
  /-
    🎉 no goals
  -/


@[simp]
lemma Differentiable.sub_iff_right (hg : Differentiable 𝕜 f) :
    Differentiable 𝕜 (fun y => f y - g y) ↔ Differentiable 𝕜 g := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    hg : Differentiable 𝕜 f
    ⊢ Iff (Differentiable 𝕜 fun y => HSub.hSub (f y) (g y)) (Differentiable 𝕜 g)
  -/
  simp only [sub_eq_add_neg, differentiable_neg_iff, hg, add_iff_right]
  /-
    🎉 no goals
  -/


theorem fderivWithin_sub (hxs : UniqueDiffWithinAt 𝕜 s x) (hf : DifferentiableWithinAt 𝕜 f s x)
    (hg : DifferentiableWithinAt 𝕜 g s x) :
    fderivWithin 𝕜 (fun y => f y - g y) s x = fderivWithin 𝕜 f s x - fderivWithin 𝕜 g s x :=
  (hf.hasFDerivWithinAt.sub hg.hasFDerivWithinAt).fderivWithin hxs


/-- Version of `fderivWithin_sub` where the function is written as `f - g` instead
of `fun y ↦ f y - g y`. -/
theorem fderivWithin_sub' (hxs : UniqueDiffWithinAt 𝕜 s x) (hf : DifferentiableWithinAt 𝕜 f s x)
    (hg : DifferentiableWithinAt 𝕜 g s x) :
    fderivWithin 𝕜 (f - g) s x = fderivWithin 𝕜 f s x - fderivWithin 𝕜 g s x :=
  fderivWithin_sub hxs hf hg


theorem fderiv_sub (hf : DifferentiableAt 𝕜 f x) (hg : DifferentiableAt 𝕜 g x) :
    fderiv 𝕜 (fun y => f y - g y) x = fderiv 𝕜 f x - fderiv 𝕜 g x :=
  (hf.hasFDerivAt.sub hg.hasFDerivAt).fderiv


/-- Version of `fderiv_sub` where the function is written as `f - g` instead
of `fun y ↦ f y - g y`. -/
theorem fderiv_sub' (hf : DifferentiableAt 𝕜 f x) (hg : DifferentiableAt 𝕜 g x) :
    fderiv 𝕜 (f - g) x = fderiv 𝕜 f x - fderiv 𝕜 g x :=
  fderiv_sub hf hg


@[fun_prop]
theorem HasStrictFDerivAt.sub_const (hf : HasStrictFDerivAt f f' x) (c : F) :
    HasStrictFDerivAt (fun x => f x - c) f' x := by
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
    hf : HasStrictFDerivAt f f' x
    c : F
    ⊢ HasStrictFDerivAt (fun x => HSub.hSub (f x) c) f' x
  -/
  simpa only [sub_eq_add_neg] using hf.add_const (-c)
  /-
    🎉 no goals
  -/


theorem HasFDerivAtFilter.sub_const (hf : HasFDerivAtFilter f f' x L) (c : F) :
    HasFDerivAtFilter (fun x => f x - c) f' x L := by
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
    L : Filter E
    hf : HasFDerivAtFilter f f' x L
    c : F
    ⊢ HasFDerivAtFilter (fun x => HSub.hSub (f x) c) f' x L
  -/
  simpa only [sub_eq_add_neg] using hf.add_const (-c)
  /-
    🎉 no goals
  -/


@[fun_prop]
nonrec theorem HasFDerivWithinAt.sub_const (hf : HasFDerivWithinAt f f' s x) (c : F) :
    HasFDerivWithinAt (fun x => f x - c) f' s x :=
  hf.sub_const c


@[fun_prop]
nonrec theorem HasFDerivAt.sub_const (hf : HasFDerivAt f f' x) (c : F) :
    HasFDerivAt (fun x => f x - c) f' x :=
  hf.sub_const c


@[fun_prop]
theorem hasStrictFDerivAt_sub_const {x : F} (c : F) : HasStrictFDerivAt (· - c) (id 𝕜 F) x :=
  (hasStrictFDerivAt_id x).sub_const c


@[fun_prop]
theorem hasFDerivAt_sub_const {x : F} (c : F) : HasFDerivAt (· - c) (id 𝕜 F) x :=
  (hasFDerivAt_id x).sub_const c


@[fun_prop]
theorem DifferentiableWithinAt.sub_const (hf : DifferentiableWithinAt 𝕜 f s x) (c : F) :
    DifferentiableWithinAt 𝕜 (fun y => f y - c) s x :=
  (hf.hasFDerivWithinAt.sub_const c).differentiableWithinAt


@[simp]
theorem differentiableWithinAt_sub_const_iff (c : F) :
    DifferentiableWithinAt 𝕜 (fun y => f y - c) s x ↔ DifferentiableWithinAt 𝕜 f s x := by
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
    c : F
    ⊢ Iff (DifferentiableWithinAt 𝕜 (fun y => HSub.hSub (f y) c) s x) (Differentia …
  -/
  simp only [sub_eq_add_neg, differentiableWithinAt_add_const_iff]
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem DifferentiableAt.sub_const (hf : DifferentiableAt 𝕜 f x) (c : F) :
    DifferentiableAt 𝕜 (fun y => f y - c) x :=
  (hf.hasFDerivAt.sub_const c).differentiableAt


@[deprecated DifferentiableAt.sub_iff_left (since := "2024-07-11")]
theorem differentiableAt_sub_const_iff (c : F) :
    DifferentiableAt 𝕜 (fun y => f y - c) x ↔ DifferentiableAt 𝕜 f x :=
  (differentiableAt_const _).sub_iff_left


@[fun_prop]
theorem DifferentiableOn.sub_const (hf : DifferentiableOn 𝕜 f s) (c : F) :
    DifferentiableOn 𝕜 (fun y => f y - c) s := fun x hx => (hf x hx).sub_const c


@[deprecated DifferentiableOn.sub_iff_left (since := "2024-07-11")]
theorem differentiableOn_sub_const_iff (c : F) :
    DifferentiableOn 𝕜 (fun y => f y - c) s ↔ DifferentiableOn 𝕜 f s :=
  (differentiableOn_const _).sub_iff_left


@[fun_prop]
theorem Differentiable.sub_const (hf : Differentiable 𝕜 f) (c : F) :
    Differentiable 𝕜 fun y => f y - c := fun x => (hf x).sub_const c


@[deprecated Differentiable.sub_iff_left (since := "2024-07-11")]
theorem differentiable_sub_const_iff (c : F) :
    (Differentiable 𝕜 fun y => f y - c) ↔ Differentiable 𝕜 f :=
  (differentiable_const _).sub_iff_left


theorem fderivWithin_sub_const (hxs : UniqueDiffWithinAt 𝕜 s x) (c : F) :
    fderivWithin 𝕜 (fun y => f y - c) s x = fderivWithin 𝕜 f s x := by
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
    hxs : UniqueDiffWithinAt 𝕜 s x
    c : F
    ⊢ Eq (fderivWithin 𝕜 (fun y => HSub.hSub (f y) c) s x) (fderivWithin 𝕜 f s x)
  -/
  simp only [sub_eq_add_neg, fderivWithin_add_const hxs]
  /-
    🎉 no goals
  -/


theorem fderiv_sub_const (c : F) : fderiv 𝕜 (fun y => f y - c) x = fderiv 𝕜 f x := by
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
    c : F
    ⊢ Eq (fderiv 𝕜 (fun y => HSub.hSub (f y) c) x) (fderiv 𝕜 f x)
  -/
  simp only [sub_eq_add_neg, fderiv_add_const]
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem HasStrictFDerivAt.const_sub (hf : HasStrictFDerivAt f f' x) (c : F) :
    HasStrictFDerivAt (fun x => c - f x) (-f') x := by
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
    hf : HasStrictFDerivAt f f' x
    c : F
    ⊢ HasStrictFDerivAt (fun x => HSub.hSub c (f x)) (Neg.neg f') x
  -/
  simpa only [sub_eq_add_neg] using hf.neg.const_add c
  /-
    🎉 no goals
  -/


theorem HasFDerivAtFilter.const_sub (hf : HasFDerivAtFilter f f' x L) (c : F) :
    HasFDerivAtFilter (fun x => c - f x) (-f') x L := by
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
    L : Filter E
    hf : HasFDerivAtFilter f f' x L
    c : F
    ⊢ HasFDerivAtFilter (fun x => HSub.hSub c (f x)) (Neg.neg f') x L
  -/
  simpa only [sub_eq_add_neg] using hf.neg.const_add c
  /-
    🎉 no goals
  -/


@[fun_prop]
nonrec theorem HasFDerivWithinAt.const_sub (hf : HasFDerivWithinAt f f' s x) (c : F) :
    HasFDerivWithinAt (fun x => c - f x) (-f') s x :=
  hf.const_sub c


@[fun_prop]
nonrec theorem HasFDerivAt.const_sub (hf : HasFDerivAt f f' x) (c : F) :
    HasFDerivAt (fun x => c - f x) (-f') x :=
  hf.const_sub c


@[fun_prop]
theorem DifferentiableWithinAt.const_sub (hf : DifferentiableWithinAt 𝕜 f s x) (c : F) :
    DifferentiableWithinAt 𝕜 (fun y => c - f y) s x :=
  (hf.hasFDerivWithinAt.const_sub c).differentiableWithinAt


@[simp]
theorem differentiableWithinAt_const_sub_iff (c : F) :
    DifferentiableWithinAt 𝕜 (fun y => c - f y) s x ↔ DifferentiableWithinAt 𝕜 f s x := by
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
    c : F
    ⊢ Iff (DifferentiableWithinAt 𝕜 (fun y => HSub.hSub c (f y)) s x) (Differentia …
  -/
  simp [sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem DifferentiableAt.const_sub (hf : DifferentiableAt 𝕜 f x) (c : F) :
    DifferentiableAt 𝕜 (fun y => c - f y) x :=
  (hf.hasFDerivAt.const_sub c).differentiableAt


@[deprecated DifferentiableAt.sub_iff_right (since := "2024-07-11")]
theorem differentiableAt_const_sub_iff (c : F) :
    DifferentiableAt 𝕜 (fun y => c - f y) x ↔ DifferentiableAt 𝕜 f x :=
  (differentiableAt_const _).sub_iff_right


@[fun_prop]
theorem DifferentiableOn.const_sub (hf : DifferentiableOn 𝕜 f s) (c : F) :
    DifferentiableOn 𝕜 (fun y => c - f y) s := fun x hx => (hf x hx).const_sub c


@[deprecated DifferentiableOn.sub_iff_right (since := "2024-07-11")]
theorem differentiableOn_const_sub_iff (c : F) :
    DifferentiableOn 𝕜 (fun y => c - f y) s ↔ DifferentiableOn 𝕜 f s :=
  (differentiableOn_const _).sub_iff_right


@[fun_prop]
theorem Differentiable.const_sub (hf : Differentiable 𝕜 f) (c : F) :
    Differentiable 𝕜 fun y => c - f y := fun x => (hf x).const_sub c


@[deprecated Differentiable.sub_iff_right (since := "2024-07-11")]
theorem differentiable_const_sub_iff (c : F) :
    (Differentiable 𝕜 fun y => c - f y) ↔ Differentiable 𝕜 f :=
  (differentiable_const _).sub_iff_right


theorem fderivWithin_const_sub (hxs : UniqueDiffWithinAt 𝕜 s x) (c : F) :
    fderivWithin 𝕜 (fun y => c - f y) s x = -fderivWithin 𝕜 f s x := by
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
    hxs : UniqueDiffWithinAt 𝕜 s x
    c : F
    ⊢ Eq (fderivWithin 𝕜 (fun y => HSub.hSub c (f y)) s x) (Neg.neg (fderivWithin  …
  -/
  simp only [sub_eq_add_neg, fderivWithin_const_add, fderivWithin_neg, hxs]
  /-
    🎉 no goals
  -/


theorem fderiv_const_sub (c : F) : fderiv 𝕜 (fun y => c - f y) x = -fderiv 𝕜 f x := by
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
    c : F
    ⊢ Eq (fderiv 𝕜 (fun y => HSub.hSub c (f y)) x) (Neg.neg (fderiv 𝕜 f x))
  -/
  simp only [← fderivWithin_univ, fderivWithin_const_sub uniqueDiffWithinAt_univ]
  /-
    🎉 no goals
  -/


theorem hasFDerivWithinAt_comp_add_right (a : E) :
    HasFDerivWithinAt (fun x ↦ f (x + a)) f' s x ↔ HasFDerivWithinAt f f' (a +ᵥ s) (x + a) := by
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
    a : E
    ⊢ Iff (HasFDerivWithinAt (fun x => f (HAdd.hAdd x a)) f' s x) (HasFDerivWithin …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
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
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : E
      s : Set E
      a : E
      h : HasFDerivWithinAt (fun x => f (HAdd.hAdd x a)) f' s x
      ⊢ HasFDerivWithinAt f f' (HVAdd.hVAdd a s) (HAdd.hAdd x a)
    -/
  · have A : f = (fun x ↦ f (x + a)) ∘ (fun x ↦ x - a) := by ext; simp
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
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : E
      s : Set E
      a : E
      h : HasFDerivWithinAt (fun x => f (HAdd.hAdd x a)) f' s x
      A : Eq f (Function.comp (fun x => f (HAdd.hAdd x a)) fun x => HSub.hSub x a)
      ⊢ HasFDerivWithinAt f f' (HVAdd.hVAdd a s) (HAdd.hAdd x a)
    -/
    rw [show x = (x + a) - a by abel] at h
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
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : E
      s : Set E
      a : E
      h : HasFDerivWithinAt (fun x => f (HAdd.hAdd x a)) f' s (HSub.hSub (HAdd.hAdd  …
      A : Eq f (Function.comp (fun x => f (HAdd.hAdd x a)) fun x => HSub.hSub x a)
      ⊢ HasFDerivWithinAt f f' (HVAdd.hVAdd a s) (HAdd.hAdd x a)
    -/
    rw [A]
    have : HasFDerivWithinAt (fun x ↦ x - a) (ContinuousLinearMap.id 𝕜 E) (a +ᵥ s) (x + a) := by
      simpa using (hasFDerivWithinAt_id (x + a) _).sub (hasFDerivWithinAt_const _ _ _)
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
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : E
      s : Set E
      a : E
      h : HasFDerivWithinAt (fun x => f (HAdd.hAdd x a)) f' s (HSub.hSub (HAdd.hAdd  …
      A : Eq f (Function.comp (fun x => f (HAdd.hAdd x a)) fun x => HSub.hSub x a)
      this : HasFDerivWithinAt (fun x => HSub.hSub x a) (ContinuousLinearMap.id 𝕜 E) …
      ⊢ HasFDerivWithinAt (Function.comp (fun x => f (HAdd.hAdd x a)) fun x => HSub. …
    -/
    apply h.comp (x + a) this (fun y hy ↦ ?_)
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
      a : E
      h : HasFDerivWithinAt (fun x => f (HAdd.hAdd x a)) f' s (HSub.hSub (HAdd.hAdd  …
      A : Eq f (Function.comp (fun x => f (HAdd.hAdd x a)) fun x => HSub.hSub x a)
      this : HasFDerivWithinAt (fun x => HSub.hSub x a) (ContinuousLinearMap.id 𝕜 E) …
      y : E
      hy : Membership.mem (HVAdd.hVAdd a s) y
      ⊢ Membership.mem s (HSub.hSub y a)
    -/
    simpa [Set.mem_vadd_set_iff_neg_vadd_mem, add_comm, ← sub_eq_add_neg] using hy
    /-
      🎉 no goals
    -/
  · have : HasFDerivWithinAt (fun x ↦ x + a) (ContinuousLinearMap.id 𝕜 E) s x := by
      simpa using (hasFDerivWithinAt_id x s (𝕜 := 𝕜)).add (hasFDerivWithinAt_const a x s (𝕜 := 𝕜))
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
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : E
      s : Set E
      a : E
      h : HasFDerivWithinAt f f' (HVAdd.hVAdd a s) (HAdd.hAdd x a)
      this : HasFDerivWithinAt (fun x => HAdd.hAdd x a) (ContinuousLinearMap.id 𝕜 E) …
      ⊢ HasFDerivWithinAt (fun x => f (HAdd.hAdd x a)) f' s x
    -/
    apply h.comp x this (fun y hy ↦ ?_)
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
      a : E
      h : HasFDerivWithinAt f f' (HVAdd.hVAdd a s) (HAdd.hAdd x a)
      this : HasFDerivWithinAt (fun x => HAdd.hAdd x a) (ContinuousLinearMap.id 𝕜 E) …
      y : E
      hy : Membership.mem s y
      ⊢ Membership.mem (HVAdd.hVAdd a s) (HAdd.hAdd y a)
    -/
    simp [Set.mem_vadd_set_iff_neg_vadd_mem, hy]
    /-
      🎉 no goals
    -/


theorem differentiableWithinAt_comp_add_right (a : E) :
    DifferentiableWithinAt 𝕜 (fun x ↦ f (x + a)) s x ↔
      DifferentiableWithinAt 𝕜 f (a +ᵥ s) (x + a) := by
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
    a : E
    ⊢ Iff (DifferentiableWithinAt 𝕜 (fun x => f (HAdd.hAdd x a)) s x) (Differentia …
  -/
  simp [DifferentiableWithinAt, hasFDerivWithinAt_comp_add_right]
  /-
    🎉 no goals
  -/


theorem fderivWithin_comp_add_right (a : E) :
    fderivWithin 𝕜 (fun x ↦ f (x + a)) s x = fderivWithin 𝕜 f (a +ᵥ s) (x + a) := by
  classical
  simp only [fderivWithin, hasFDerivWithinAt_comp_add_right, DifferentiableWithinAt]


theorem hasFDerivWithinAt_comp_add_left (a : E) :
    HasFDerivWithinAt (fun x ↦ f (a + x)) f' s x ↔ HasFDerivWithinAt f f' (a +ᵥ s) (a + x) := by
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
    a : E
    ⊢ Iff (HasFDerivWithinAt (fun x => f (HAdd.hAdd a x)) f' s x) (HasFDerivWithin …
  -/
  simpa [add_comm a] using hasFDerivWithinAt_comp_add_right a
  /-
    🎉 no goals
  -/


theorem differentiableWithinAt_comp_add_left (a : E) :
    DifferentiableWithinAt 𝕜 (fun x ↦ f (a + x)) s x ↔
      DifferentiableWithinAt 𝕜 f (a +ᵥ s) (a + x) := by
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
    a : E
    ⊢ Iff (DifferentiableWithinAt 𝕜 (fun x => f (HAdd.hAdd a x)) s x) (Differentia …
  -/
  simp [DifferentiableWithinAt, hasFDerivWithinAt_comp_add_left]
  /-
    🎉 no goals
  -/


theorem fderivWithin_comp_add_left (a : E) :
    fderivWithin 𝕜 (fun x ↦ f (a + x)) s x = fderivWithin 𝕜 f (a +ᵥ s) (a + x) := by
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
    a : E
    ⊢ Eq (fderivWithin 𝕜 (fun x => f (HAdd.hAdd a x)) s x) (fderivWithin 𝕜 f (HVAd …
  -/
  simpa [add_comm a] using fderivWithin_comp_add_right a
  /-
    🎉 no goals
  -/


theorem hasFDerivAt_comp_add_right (a : E) :
    HasFDerivAt (fun x ↦ f (x + a)) f' x ↔ HasFDerivAt f f' (x + a) := by
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
    x a : E
    ⊢ Iff (HasFDerivAt (fun x => f (HAdd.hAdd x a)) f' x) (HasFDerivAt f f' (HAdd. …
  -/
  simp [← hasFDerivWithinAt_univ, hasFDerivWithinAt_comp_add_right]
  /-
    🎉 no goals
  -/


theorem differentiableAt_comp_add_right (a : E) :
    DifferentiableAt 𝕜 (fun x ↦ f (x + a)) x ↔ DifferentiableAt 𝕜 f (x + a) := by
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
    x a : E
    ⊢ Iff (DifferentiableAt 𝕜 (fun x => f (HAdd.hAdd x a)) x) (DifferentiableAt 𝕜  …
  -/
  simp [DifferentiableAt, hasFDerivAt_comp_add_right]
  /-
    🎉 no goals
  -/


theorem fderiv_comp_add_right (a : E) :
    fderiv 𝕜 (fun x ↦ f (x + a)) x = fderiv 𝕜 f (x + a) := by
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
    x a : E
    ⊢ Eq (fderiv 𝕜 (fun x => f (HAdd.hAdd x a)) x) (fderiv 𝕜 f (HAdd.hAdd x a))
  -/
  simp [← fderivWithin_univ, fderivWithin_comp_add_right]
  /-
    🎉 no goals
  -/


theorem hasFDerivAt_comp_add_left (a : E) :
    HasFDerivAt (fun x ↦ f (a + x)) f' x ↔ HasFDerivAt f f' (a + x) := by
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
    x a : E
    ⊢ Iff (HasFDerivAt (fun x => f (HAdd.hAdd a x)) f' x) (HasFDerivAt f f' (HAdd. …
  -/
  simpa [add_comm a] using hasFDerivAt_comp_add_right a
  /-
    🎉 no goals
  -/


theorem differentiableAt_comp_add_left (a : E) :
    DifferentiableAt 𝕜 (fun x ↦ f (a + x)) x ↔ DifferentiableAt 𝕜 f (a + x) := by
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
    x a : E
    ⊢ Iff (DifferentiableAt 𝕜 (fun x => f (HAdd.hAdd a x)) x) (DifferentiableAt 𝕜  …
  -/
  simp [DifferentiableAt, hasFDerivAt_comp_add_left]
  /-
    🎉 no goals
  -/


theorem fderiv_comp_add_left (a : E) :
    fderiv 𝕜 (fun x ↦ f (a + x)) x = fderiv 𝕜 f (a + x) := by
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
    x a : E
    ⊢ Eq (fderiv 𝕜 (fun x => f (HAdd.hAdd a x)) x) (fderiv 𝕜 f (HAdd.hAdd a x))
  -/
  simpa [add_comm a] using fderiv_comp_add_right a
  /-
    🎉 no goals
  -/


theorem hasFDerivWithinAt_comp_sub (a : E) :
    HasFDerivWithinAt (fun x ↦ f (x - a)) f' s x ↔ HasFDerivWithinAt f f' (-a +ᵥ s) (x - a) := by
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
    a : E
    ⊢ Iff (HasFDerivWithinAt (fun x => f (HSub.hSub x a)) f' s x) (HasFDerivWithin …
  -/
  simpa [sub_eq_add_neg] using hasFDerivWithinAt_comp_add_right (-a)
  /-
    🎉 no goals
  -/


theorem differentiableWithinAt_comp_sub (a : E) :
    DifferentiableWithinAt 𝕜 (fun x ↦ f (x - a)) s x ↔
      DifferentiableWithinAt 𝕜 f (-a +ᵥ s) (x - a) := by
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
    a : E
    ⊢ Iff (DifferentiableWithinAt 𝕜 (fun x => f (HSub.hSub x a)) s x) (Differentia …
  -/
  simp [DifferentiableWithinAt, hasFDerivWithinAt_comp_sub]
  /-
    🎉 no goals
  -/


theorem fderivWithin_comp_sub (a : E) :
    fderivWithin 𝕜 (fun x ↦ f (x - a)) s x = fderivWithin 𝕜 f (-a +ᵥ s) (x - a) := by
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
    a : E
    ⊢ Eq (fderivWithin 𝕜 (fun x => f (HSub.hSub x a)) s x) (fderivWithin 𝕜 f (HVAd …
  -/
  simpa [sub_eq_add_neg] using fderivWithin_comp_add_right (-a)
  /-
    🎉 no goals
  -/


theorem hasFDerivAt_comp_sub (a : E) :
    HasFDerivAt (fun x ↦ f (x - a)) f' x ↔ HasFDerivAt f f' (x - a) := by
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
    x a : E
    ⊢ Iff (HasFDerivAt (fun x => f (HSub.hSub x a)) f' x) (HasFDerivAt f f' (HSub. …
  -/
  simp [← hasFDerivWithinAt_univ, hasFDerivWithinAt_comp_sub]
  /-
    🎉 no goals
  -/


theorem differentiableAt_comp_sub (a : E) :
    DifferentiableAt 𝕜 (fun x ↦ f (x - a)) x ↔ DifferentiableAt 𝕜 f (x - a) := by
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
    x a : E
    ⊢ Iff (DifferentiableAt 𝕜 (fun x => f (HSub.hSub x a)) x) (DifferentiableAt 𝕜  …
  -/
  simp [DifferentiableAt, hasFDerivAt_comp_sub]
  /-
    🎉 no goals
  -/


theorem fderiv_comp_sub (a : E) :
    fderiv 𝕜 (fun x ↦ f (x - a)) x = fderiv 𝕜 f (x - a) := by
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
    x a : E
    ⊢ Eq (fderiv 𝕜 (fun x => f (HSub.hSub x a)) x) (fderiv 𝕜 f (HSub.hSub x a))
  -/
  simp [← fderivWithin_univ, fderivWithin_comp_sub]
  /-
    🎉 no goals
  -/


