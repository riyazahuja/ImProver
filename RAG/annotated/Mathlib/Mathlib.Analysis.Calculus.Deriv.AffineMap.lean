theorem hasStrictDerivAt : HasStrictDerivAt f (f.linear 1) x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : AffineMap 𝕜 𝕜 E
    x : 𝕜
    ⊢ HasStrictDerivAt (⇑f) (f.linear 1) x
  -/
  rw [f.decomp]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : AffineMap 𝕜 𝕜 E
    x : 𝕜
    ⊢ HasStrictDerivAt (HAdd.hAdd ⇑f.linear fun x => f 0) (f.linear 1) x
  -/
  exact f.linear.hasStrictDerivAt.add_const (f 0)
  /-
    🎉 no goals
  -/


theorem hasDerivAtFilter : HasDerivAtFilter f (f.linear 1) x L := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : AffineMap 𝕜 𝕜 E
    L : Filter 𝕜
    x : 𝕜
    ⊢ HasDerivAtFilter (⇑f) (f.linear 1) x L
  -/
  rw [f.decomp]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : AffineMap 𝕜 𝕜 E
    L : Filter 𝕜
    x : 𝕜
    ⊢ HasDerivAtFilter (HAdd.hAdd ⇑f.linear fun x => f 0) (f.linear 1) x L
  -/
  exact f.linear.hasDerivAtFilter.add_const (f 0)
  /-
    🎉 no goals
  -/


theorem hasDerivWithinAt : HasDerivWithinAt f (f.linear 1) s x := f.hasDerivAtFilter

theorem hasDerivAt : HasDerivAt f (f.linear 1) x := f.hasDerivAtFilter


protected theorem derivWithin (hs : UniqueDiffWithinAt 𝕜 s x) :
    derivWithin f s x = f.linear 1 :=
  f.hasDerivWithinAt.derivWithin hs


@[simp] protected theorem deriv : deriv f x = f.linear 1 := f.hasDerivAt.deriv


protected theorem differentiableAt : DifferentiableAt 𝕜 f x := f.hasDerivAt.differentiableAt

protected theorem differentiable : Differentiable 𝕜 f := fun _ ↦ f.differentiableAt


protected theorem differentiableWithinAt : DifferentiableWithinAt 𝕜 f s x :=
  f.differentiableAt.differentiableWithinAt


protected theorem differentiableOn : DifferentiableOn 𝕜 f s := fun _ _ ↦ f.differentiableWithinAt


theorem hasStrictDerivAt_lineMap : HasStrictDerivAt (lineMap a b) (b - a) x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    a b : E
    x : 𝕜
    ⊢ HasStrictDerivAt (⇑(AffineMap.lineMap a b)) (HSub.hSub b a) x
  -/
  simpa using (lineMap a b : 𝕜 →ᵃ[𝕜] E).hasStrictDerivAt
  /-
    🎉 no goals
  -/


theorem hasDerivAt_lineMap :  HasDerivAt (lineMap a b) (b - a) x :=
  hasStrictDerivAt_lineMap.hasDerivAt


theorem hasDerivWithinAt_lineMap : HasDerivWithinAt (lineMap a b) (b - a) s x :=
  hasDerivAt_lineMap.hasDerivWithinAt


