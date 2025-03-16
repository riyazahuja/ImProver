theorem uniqueMDiffWithinAt_iff_uniqueDiffWithinAt :
    UniqueMDiffWithinAt 𝓘(𝕜, E) s x ↔ UniqueDiffWithinAt 𝕜 s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    x : E
    ⊢ Iff (UniqueMDiffWithinAt (modelWithCornersSelf 𝕜 E) s x) (UniqueDiffWithinAt …
  -/
  simp only [UniqueMDiffWithinAt, mfld_simps]
  /-
    🎉 no goals
  -/


alias ⟨UniqueMDiffWithinAt.uniqueDiffWithinAt, UniqueDiffWithinAt.uniqueMDiffWithinAt⟩ :=
  uniqueMDiffWithinAt_iff_uniqueDiffWithinAt


theorem uniqueMDiffOn_iff_uniqueDiffOn : UniqueMDiffOn 𝓘(𝕜, E) s ↔ UniqueDiffOn 𝕜 s := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    ⊢ Iff (UniqueMDiffOn (modelWithCornersSelf 𝕜 E) s) (UniqueDiffOn 𝕜 s)
  -/
  simp [UniqueMDiffOn, UniqueDiffOn, uniqueMDiffWithinAt_iff_uniqueDiffWithinAt]
  /-
    🎉 no goals
  -/


alias ⟨UniqueMDiffOn.uniqueDiffOn, UniqueDiffOn.uniqueMDiffOn⟩ := uniqueMDiffOn_iff_uniqueDiffOn


theorem ModelWithCorners.uniqueMDiffOn {H : Type*} [TopologicalSpace H]
    (I : ModelWithCorners 𝕜 E H) : UniqueMDiffOn 𝓘(𝕜, E) (Set.range I) :=
  I.uniqueDiffOn.uniqueMDiffOn


@[simp, mfld_simps]
theorem writtenInExtChartAt_model_space : writtenInExtChartAt 𝓘(𝕜, E) 𝓘(𝕜, E') x f = f :=
  rfl


theorem hasMFDerivWithinAt_iff_hasFDerivWithinAt {f'} :
    HasMFDerivWithinAt 𝓘(𝕜, E) 𝓘(𝕜, E') f s x f' ↔ HasFDerivWithinAt f f' s x := by
  simpa only [HasMFDerivWithinAt, and_iff_right_iff_imp, mfld_simps] using
    HasFDerivWithinAt.continuousWithinAt


alias ⟨HasMFDerivWithinAt.hasFDerivWithinAt, HasFDerivWithinAt.hasMFDerivWithinAt⟩ :=
  hasMFDerivWithinAt_iff_hasFDerivWithinAt


theorem hasMFDerivAt_iff_hasFDerivAt {f'} :
    HasMFDerivAt 𝓘(𝕜, E) 𝓘(𝕜, E') f x f' ↔ HasFDerivAt f f' x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    E' : Type u_3
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace 𝕜 E'
    f : E → E'
    x : E
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace (modelWithCornersSelf 𝕜  …
    ⊢ Iff (HasMFDerivAt (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E') f x …
  -/
  rw [← hasMFDerivWithinAt_univ, hasMFDerivWithinAt_iff_hasFDerivWithinAt, hasFDerivWithinAt_univ]
  /-
    🎉 no goals
  -/


alias ⟨HasMFDerivAt.hasFDerivAt, HasFDerivAt.hasMFDerivAt⟩ := hasMFDerivAt_iff_hasFDerivAt


/-- For maps between vector spaces, `MDifferentiableWithinAt` and `DifferentiableWithinAt`
coincide -/
theorem mdifferentiableWithinAt_iff_differentiableWithinAt :
    MDifferentiableWithinAt 𝓘(𝕜, E) 𝓘(𝕜, E') f s x ↔ DifferentiableWithinAt 𝕜 f s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    E' : Type u_3
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace 𝕜 E'
    f : E → E'
    s : Set E
    x : E
    ⊢ Iff (MDifferentiableWithinAt (modelWithCornersSelf 𝕜 E) (modelWithCornersSel …
  -/
  simp only [mdifferentiableWithinAt_iff', mfld_simps]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    E' : Type u_3
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace 𝕜 E'
    f : E → E'
    s : Set E
    x : E
    ⊢ Iff (And (ContinuousWithinAt f s x) (DifferentiableWithinAt 𝕜 f s x)) (Diffe …
  -/
  exact ⟨fun H => H.2, fun H => ⟨H.continuousWithinAt, H⟩⟩
  /-
    🎉 no goals
  -/


alias ⟨MDifferentiableWithinAt.differentiableWithinAt,
    DifferentiableWithinAt.mdifferentiableWithinAt⟩ :=
  mdifferentiableWithinAt_iff_differentiableWithinAt


/-- For maps between vector spaces, `MDifferentiableAt` and `DifferentiableAt` coincide -/
theorem mdifferentiableAt_iff_differentiableAt :
    MDifferentiableAt 𝓘(𝕜, E) 𝓘(𝕜, E') f x ↔ DifferentiableAt 𝕜 f x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    E' : Type u_3
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace 𝕜 E'
    f : E → E'
    x : E
    ⊢ Iff (MDifferentiableAt (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E' …
  -/
  simp only [mdifferentiableAt_iff, differentiableWithinAt_univ, mfld_simps]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    E' : Type u_3
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace 𝕜 E'
    f : E → E'
    x : E
    ⊢ Iff (And (ContinuousAt f x) (DifferentiableAt 𝕜 f x)) (DifferentiableAt 𝕜 f x)
  -/
  exact ⟨fun H => H.2, fun H => ⟨H.continuousAt, H⟩⟩
  /-
    🎉 no goals
  -/


alias ⟨MDifferentiableAt.differentiableAt, DifferentiableAt.mdifferentiableAt⟩ :=
  mdifferentiableAt_iff_differentiableAt


/-- For maps between vector spaces, `MDifferentiableOn` and `DifferentiableOn` coincide -/
theorem mdifferentiableOn_iff_differentiableOn :
    MDifferentiableOn 𝓘(𝕜, E) 𝓘(𝕜, E') f s ↔ DifferentiableOn 𝕜 f s := by
  simp only [MDifferentiableOn, DifferentiableOn,
    mdifferentiableWithinAt_iff_differentiableWithinAt]


alias ⟨MDifferentiableOn.differentiableOn, DifferentiableOn.mdifferentiableOn⟩ :=
  mdifferentiableOn_iff_differentiableOn


/-- For maps between vector spaces, `MDifferentiable` and `Differentiable` coincide -/
theorem mdifferentiable_iff_differentiable :
    MDifferentiable 𝓘(𝕜, E) 𝓘(𝕜, E') f ↔ Differentiable 𝕜 f := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    E' : Type u_3
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace 𝕜 E'
    f : E → E'
    ⊢ Iff (MDifferentiable (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E')  …
  -/
  simp only [MDifferentiable, Differentiable, mdifferentiableAt_iff_differentiableAt]
  /-
    🎉 no goals
  -/


alias ⟨MDifferentiable.differentiable, Differentiable.mdifferentiable⟩ :=
  mdifferentiable_iff_differentiable


/-- For maps between vector spaces, `mfderivWithin` and `fderivWithin` coincide -/
@[simp]
theorem mfderivWithin_eq_fderivWithin :
    mfderivWithin 𝓘(𝕜, E) 𝓘(𝕜, E') f s x = fderivWithin 𝕜 f s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    E' : Type u_3
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace 𝕜 E'
    f : E → E'
    s : Set E
    x : E
    ⊢ Eq (mfderivWithin (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E') f s …
  -/
  by_cases h : MDifferentiableWithinAt 𝓘(𝕜, E) 𝓘(𝕜, E') f s x
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹ : NormedAddCommGroup E'
      inst✝ : NormedSpace 𝕜 E'
      f : E → E'
      s : Set E
      x : E
      h : MDifferentiableWithinAt (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 …
      ⊢ Eq (mfderivWithin (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E') f s …
    -/
  · simp only [mfderivWithin, h, if_pos, mfld_simps]
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹ : NormedAddCommGroup E'
      inst✝ : NormedSpace 𝕜 E'
      f : E → E'
      s : Set E
      x : E
      h : Not (MDifferentiableWithinAt (modelWithCornersSelf 𝕜 E) (modelWithCornersS …
      ⊢ Eq (mfderivWithin (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E') f s …
    -/
  · simp only [mfderivWithin, h, if_neg, not_false_iff]
    /-
      case neg
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹ : NormedAddCommGroup E'
      inst✝ : NormedSpace 𝕜 E'
      f : E → E'
      s : Set E
      x : E
      h : Not (MDifferentiableWithinAt (modelWithCornersSelf 𝕜 E) (modelWithCornersS …
      ⊢ Eq 0 (fderivWithin 𝕜 f s x)
    -/
    rw [mdifferentiableWithinAt_iff_differentiableWithinAt] at h
    /-
      case neg
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹ : NormedAddCommGroup E'
      inst✝ : NormedSpace 𝕜 E'
      f : E → E'
      s : Set E
      x : E
      h : Not (DifferentiableWithinAt 𝕜 f s x)
      ⊢ Eq 0 (fderivWithin 𝕜 f s x)
    -/
    exact (fderivWithin_zero_of_not_differentiableWithinAt h).symm
    /-
      🎉 no goals
    -/


/-- For maps between vector spaces, `mfderiv` and `fderiv` coincide -/
@[simp]
theorem mfderiv_eq_fderiv : mfderiv 𝓘(𝕜, E) 𝓘(𝕜, E') f x = fderiv 𝕜 f x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    E' : Type u_3
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace 𝕜 E'
    f : E → E'
    x : E
    ⊢ Eq (mfderiv (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E') f x) (fde …
  -/
  rw [← mfderivWithin_univ, ← fderivWithin_univ]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    E' : Type u_3
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace 𝕜 E'
    f : E → E'
    x : E
    ⊢ Eq (mfderivWithin (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E') f S …
  -/
  exact mfderivWithin_eq_fderivWithin
  /-
    🎉 no goals
  -/


