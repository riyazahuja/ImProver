/-- A real differentiable map `f` is conformal at point `x` if and only if its
    differential `fderiv ℝ f x` at that point scales every inner product by a positive scalar. -/
theorem conformalAt_iff' {f : E → F} {x : E} : ConformalAt f x ↔
    ∃ c : ℝ, 0 < c ∧ ∀ u v : E, ⟪fderiv ℝ f x u, fderiv ℝ f x v⟫ = c * ⟪u, v⟫ := by
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real E
    inst✝ : InnerProductSpace Real F
    f : E → F
    x : E
    ⊢ Iff (ConformalAt f x) (Exists fun c => And (LT.lt 0 c) (∀ (u v : E), Eq (Inn …
  -/
  rw [conformalAt_iff_isConformalMap_fderiv, isConformalMap_iff]
  /-
    🎉 no goals
  -/


/-- A real differentiable map `f` is conformal at point `x` if and only if its
    differential `f'` at that point scales every inner product by a positive scalar. -/
theorem conformalAt_iff {f : E → F} {x : E} {f' : E →L[ℝ] F} (h : HasFDerivAt f f' x) :
    ConformalAt f x ↔ ∃ c : ℝ, 0 < c ∧ ∀ u v : E, ⟪f' u, f' v⟫ = c * ⟪u, v⟫ := by
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real E
    inst✝ : InnerProductSpace Real F
    f : E → F
    x : E
    f' : ContinuousLinearMap (RingHom.id Real) E F
    h : HasFDerivAt f f' x
    ⊢ Iff (ConformalAt f x) (Exists fun c => And (LT.lt 0 c) (∀ (u v : E), Eq (Inn …
  -/
  simp only [conformalAt_iff', h.fderiv]
  /-
    🎉 no goals
  -/


/-- The conformal factor of a conformal map at some point `x`. Some authors refer to this function
    as the characteristic function of the conformal map. -/
def conformalFactorAt {f : E → F} {x : E} (h : ConformalAt f x) : ℝ :=
  Classical.choose (conformalAt_iff'.mp h)


theorem conformalFactorAt_pos {f : E → F} {x : E} (h : ConformalAt f x) : 0 < conformalFactorAt h :=
  (Classical.choose_spec <| conformalAt_iff'.mp h).1


theorem conformalFactorAt_inner_eq_mul_inner' {f : E → F} {x : E} (h : ConformalAt f x) (u v : E) :
    ⟪(fderiv ℝ f x) u, (fderiv ℝ f x) v⟫ = (conformalFactorAt h : ℝ) * ⟪u, v⟫ :=
  (Classical.choose_spec <| conformalAt_iff'.mp h).2 u v


theorem conformalFactorAt_inner_eq_mul_inner {f : E → F} {x : E} {f' : E →L[ℝ] F}
    (h : HasFDerivAt f f' x) (H : ConformalAt f x) (u v : E) :
    ⟪f' u, f' v⟫ = (conformalFactorAt H : ℝ) * ⟪u, v⟫ :=
  H.differentiableAt.hasFDerivAt.unique h ▸ conformalFactorAt_inner_eq_mul_inner' H u v

