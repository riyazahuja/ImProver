theorem differentiableWithinAt_piLp :
    DifferentiableWithinAt 𝕜 f t y ↔ ∀ i, DifferentiableWithinAt 𝕜 (fun x => f x i) t y := by
  rw [← (PiLp.continuousLinearEquiv p 𝕜 E).comp_differentiableWithinAt_iff,
    differentiableWithinAt_pi]
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    f : H → PiLp p E
    t : Set H
    y : H
    ⊢ Iff (∀ (i : ι), DifferentiableWithinAt 𝕜 (fun x => Function.comp (⇑(PiLp.con …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem differentiableAt_piLp :
    DifferentiableAt 𝕜 f y ↔ ∀ i, DifferentiableAt 𝕜 (fun x => f x i) y := by
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    f : H → PiLp p E
    y : H
    ⊢ Iff (DifferentiableAt 𝕜 f y) (∀ (i : ι), DifferentiableAt 𝕜 (fun x => f x i) …
  -/
  rw [← (PiLp.continuousLinearEquiv p 𝕜 E).comp_differentiableAt_iff, differentiableAt_pi]
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    f : H → PiLp p E
    y : H
    ⊢ Iff (∀ (i : ι), DifferentiableAt 𝕜 (fun x => Function.comp (⇑(PiLp.continuou …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem differentiableOn_piLp :
    DifferentiableOn 𝕜 f t ↔ ∀ i, DifferentiableOn 𝕜 (fun x => f x i) t := by
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    f : H → PiLp p E
    t : Set H
    ⊢ Iff (DifferentiableOn 𝕜 f t) (∀ (i : ι), DifferentiableOn 𝕜 (fun x => f x i) …
  -/
  rw [← (PiLp.continuousLinearEquiv p 𝕜 E).comp_differentiableOn_iff, differentiableOn_pi]
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    f : H → PiLp p E
    t : Set H
    ⊢ Iff (∀ (i : ι), DifferentiableOn 𝕜 (fun x => Function.comp (⇑(PiLp.continuou …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem differentiable_piLp : Differentiable 𝕜 f ↔ ∀ i, Differentiable 𝕜 fun x => f x i := by
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    f : H → PiLp p E
    ⊢ Iff (Differentiable 𝕜 f) (∀ (i : ι), Differentiable 𝕜 fun x => f x i)
  -/
  rw [← (PiLp.continuousLinearEquiv p 𝕜 E).comp_differentiable_iff, differentiable_pi]
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    f : H → PiLp p E
    ⊢ Iff (∀ (i : ι), Differentiable 𝕜 fun x => Function.comp (⇑(PiLp.continuousLi …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem hasStrictFDerivAt_piLp :
    HasStrictFDerivAt f f' y ↔
      ∀ i, HasStrictFDerivAt (fun x => f x i) (PiLp.proj _ _ i ∘L f') y := by
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    f : H → PiLp p E
    f' : ContinuousLinearMap (RingHom.id 𝕜) H (PiLp p E)
    y : H
    ⊢ Iff (HasStrictFDerivAt f f' y) (∀ (i : ι), HasStrictFDerivAt (fun x => f x i …
  -/
  rw [← (PiLp.continuousLinearEquiv p 𝕜 E).comp_hasStrictFDerivAt_iff, hasStrictFDerivAt_pi']
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    f : H → PiLp p E
    f' : ContinuousLinearMap (RingHom.id 𝕜) H (PiLp p E)
    y : H
    ⊢ Iff (∀ (i : ι), HasStrictFDerivAt (fun x => Function.comp (⇑(PiLp.continuous …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem hasFDerivWithinAt_piLp :
    HasFDerivWithinAt f f' t y ↔
      ∀ i, HasFDerivWithinAt (fun x => f x i) (PiLp.proj _ _ i ∘L f') t y := by
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    f : H → PiLp p E
    f' : ContinuousLinearMap (RingHom.id 𝕜) H (PiLp p E)
    t : Set H
    y : H
    ⊢ Iff (HasFDerivWithinAt f f' t y) (∀ (i : ι), HasFDerivWithinAt (fun x => f x …
  -/
  rw [← (PiLp.continuousLinearEquiv p 𝕜 E).comp_hasFDerivWithinAt_iff, hasFDerivWithinAt_pi']
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    f : H → PiLp p E
    f' : ContinuousLinearMap (RingHom.id 𝕜) H (PiLp p E)
    t : Set H
    y : H
    ⊢ Iff (∀ (i : ι), HasFDerivWithinAt (fun x => Function.comp (⇑(PiLp.continuous …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem hasStrictFDerivAt_equiv (f : PiLp p E) :
    HasStrictFDerivAt (WithLp.equiv p (∀ i, E i))
      (PiLp.continuousLinearEquiv p 𝕜 _).toContinuousLinearMap f :=
  .of_isLittleO <| (Asymptotics.isLittleO_zero _ _).congr_left fun _ => (sub_self _).symm


theorem hasStrictFDerivAt_equiv_symm (f : PiLp p E) :
    HasStrictFDerivAt (WithLp.equiv p (∀ i, E i)).symm
      (PiLp.continuousLinearEquiv p 𝕜 _).symm.toContinuousLinearMap f :=
  .of_isLittleO <| (Asymptotics.isLittleO_zero _ _).congr_left fun _ => (sub_self _).symm


nonrec theorem hasStrictFDerivAt_apply (f : PiLp p E) (i : ι) :
    HasStrictFDerivAt (𝕜 := 𝕜) (fun f : PiLp p E => f i) (proj p E i) f :=
  (hasStrictFDerivAt_apply i f).comp f (hasStrictFDerivAt_equiv (𝕜 := 𝕜) p f)


theorem hasFDerivAt_equiv (f : PiLp p E) :
    HasFDerivAt (WithLp.equiv p (∀ i, E i))
      (PiLp.continuousLinearEquiv p 𝕜 _).toContinuousLinearMap f :=
  (hasStrictFDerivAt_equiv p f).hasFDerivAt


theorem hasFDerivAt_equiv_symm (f : PiLp p E) :
    HasFDerivAt (WithLp.equiv p (∀ i, E i)).symm
      (PiLp.continuousLinearEquiv p 𝕜 _).symm.toContinuousLinearMap f :=
  (hasStrictFDerivAt_equiv_symm p f).hasFDerivAt


nonrec theorem hasFDerivAt_apply (f : PiLp p E) (i : ι) :
    HasFDerivAt (𝕜 := 𝕜) (fun f : PiLp p E => f i) (proj p E i) f :=
  (hasStrictFDerivAt_apply p f i).hasFDerivAt


