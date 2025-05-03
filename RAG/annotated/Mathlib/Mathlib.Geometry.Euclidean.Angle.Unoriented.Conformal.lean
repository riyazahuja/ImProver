theorem IsConformalMap.preserves_angle {f' : E →L[ℝ] F} (h : IsConformalMap f') (u v : E) :
    angle (f' u) (f' v) = angle u v := by
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real E
    inst✝ : InnerProductSpace Real F
    f' : ContinuousLinearMap (RingHom.id Real) E F
    h : IsConformalMap f'
    u v : E
    ⊢ Eq (InnerProductGeometry.angle (f' u) (f' v)) (InnerProductGeometry.angle u v)
  -/
  obtain ⟨c, hc, li, rfl⟩ := h
  /-
    case intro.intro.intro
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real E
    inst✝ : InnerProductSpace Real F
    u v : E
    c : Real
    hc : Ne c 0
    li : LinearIsometry (RingHom.id Real) E F
    ⊢ Eq (InnerProductGeometry.angle ((HSMul.hSMul c li.toContinuousLinearMap) u)  …
  -/
  exact (angle_smul_smul hc _ _).trans (li.angle_map _ _)
  /-
    🎉 no goals
  -/


/-- If a real differentiable map `f` is conformal at a point `x`,
    then it preserves the angles at that point. -/
theorem ConformalAt.preserves_angle {f : E → F} {x : E} {f' : E →L[ℝ] F} (h : HasFDerivAt f f' x)
    (H : ConformalAt f x) (u v : E) : angle (f' u) (f' v) = angle u v :=
  let ⟨_, h₁, c⟩ := H
  h₁.unique h ▸ IsConformalMap.preserves_angle c u v


