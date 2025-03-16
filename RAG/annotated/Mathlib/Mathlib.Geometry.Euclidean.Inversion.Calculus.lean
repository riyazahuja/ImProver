protected theorem ContDiffWithinAt.inversion (hc : ContDiffWithinAt ℝ n c s a)
    (hR : ContDiffWithinAt ℝ n R s a) (hx : ContDiffWithinAt ℝ n x s a) (hne : x a ≠ c a) :
    ContDiffWithinAt ℝ n (fun a ↦ inversion (c a) (R a) (x a)) s a :=
  (((hR.div (hx.dist ℝ hc hne) (dist_ne_zero.2 hne)).pow _).smul (hx.sub hc)).add hc


protected theorem ContDiffOn.inversion (hc : ContDiffOn ℝ n c s) (hR : ContDiffOn ℝ n R s)
    (hx : ContDiffOn ℝ n x s) (hne : ∀ a ∈ s, x a ≠ c a) :
    ContDiffOn ℝ n (fun a ↦ inversion (c a) (R a) (x a)) s := fun a ha ↦
  (hc a ha).inversion (hR a ha) (hx a ha) (hne a ha)


protected nonrec theorem ContDiffAt.inversion (hc : ContDiffAt ℝ n c a) (hR : ContDiffAt ℝ n R a)
    (hx : ContDiffAt ℝ n x a) (hne : x a ≠ c a) :
    ContDiffAt ℝ n (fun a ↦ inversion (c a) (R a) (x a)) a :=
  hc.inversion hR hx hne


protected nonrec theorem ContDiff.inversion (hc : ContDiff ℝ n c) (hR : ContDiff ℝ n R)
    (hx : ContDiff ℝ n x) (hne : ∀ a, x a ≠ c a) :
    ContDiff ℝ n (fun a ↦ inversion (c a) (R a) (x a)) :=
  contDiff_iff_contDiffAt.2 fun a ↦ hc.contDiffAt.inversion hR.contDiffAt hx.contDiffAt (hne a)


protected theorem DifferentiableWithinAt.inversion (hc : DifferentiableWithinAt ℝ c s a)
    (hR : DifferentiableWithinAt ℝ R s a) (hx : DifferentiableWithinAt ℝ x s a) (hne : x a ≠ c a) :
    DifferentiableWithinAt ℝ (fun a ↦ inversion (c a) (R a) (x a)) s a :=
  -- TODO: Use `.div` https://github.com/leanprover-community/mathlib4/issues/5870
  (((hR.mul <| (hx.dist ℝ hc hne).inv (dist_ne_zero.2 hne)).pow _).smul (hx.sub hc)).add hc


protected theorem DifferentiableOn.inversion (hc : DifferentiableOn ℝ c s)
    (hR : DifferentiableOn ℝ R s) (hx : DifferentiableOn ℝ x s) (hne : ∀ a ∈ s, x a ≠ c a) :
    DifferentiableOn ℝ (fun a ↦ inversion (c a) (R a) (x a)) s := fun a ha ↦
  (hc a ha).inversion (hR a ha) (hx a ha) (hne a ha)


protected theorem DifferentiableAt.inversion (hc : DifferentiableAt ℝ c a)
    (hR : DifferentiableAt ℝ R a) (hx : DifferentiableAt ℝ x a) (hne : x a ≠ c a) :
    DifferentiableAt ℝ (fun a ↦ inversion (c a) (R a) (x a)) a := by
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    c x : E → F
    R : E → Real
    a : E
    hc : DifferentiableAt Real c a
    hR : DifferentiableAt Real R a
    hx : DifferentiableAt Real x a
    hne : Ne (x a) (c a)
    ⊢ DifferentiableAt Real (fun a => EuclideanGeometry.inversion (c a) (R a) (x a …
  -/
  rw [← differentiableWithinAt_univ] at *
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    c x : E → F
    R : E → Real
    a : E
    hc : DifferentiableWithinAt Real c Set.univ a
    hR : DifferentiableWithinAt Real R Set.univ a
    hx : DifferentiableWithinAt Real x Set.univ a
    hne : Ne (x a) (c a)
    ⊢ DifferentiableWithinAt Real (fun a => EuclideanGeometry.inversion (c a) (R a …
  -/
  exact hc.inversion hR hx hne
  /-
    🎉 no goals
  -/


protected theorem Differentiable.inversion (hc : Differentiable ℝ c)
    (hR : Differentiable ℝ R) (hx : Differentiable ℝ x) (hne : ∀ a, x a ≠ c a) :
    Differentiable ℝ (fun a ↦ inversion (c a) (R a) (x a)) := fun a ↦
  (hc a).inversion (hR a) (hx a) (hne a)


/-- Formula for the Fréchet derivative of `EuclideanGeometry.inversion c R`. -/
theorem hasFDerivAt_inversion (hx : x ≠ c) :
    HasFDerivAt (inversion c R)
      ((R / dist x c) ^ 2 • (reflection (ℝ ∙ (x - c))ᗮ : F →L[ℝ] F)) x := by
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    c x : F
    R : Real
    hx : Ne x c
    ⊢ HasFDerivAt (EuclideanGeometry.inversion c R) (HSMul.hSMul (HPow.hPow (HDiv. …
  -/
  rcases add_left_surjective c x with ⟨x, rfl⟩
  have : HasFDerivAt (inversion c R) (?_ : F →L[ℝ] F) (c + x) := by
    #adaptation_note /-- nightly-2024-03-16: simp was
    simp (config := { unfoldPartialApp := true }) only [inversion] -/
    simp only [inversion_def]
    simp_rw [dist_eq_norm, div_pow, div_eq_mul_inv]
    have A := (hasFDerivAt_id (𝕜 := ℝ) (c + x)).sub_const c
    have B := ((hasDerivAt_inv <| by simpa using hx).comp_hasFDerivAt _ A.norm_sq).const_mul
      (R ^ 2)
    exact (B.smul A).add_const c
  refine this.congr_fderiv (LinearMap.ext_on_codisjoint
    (Submodule.isCompl_orthogonal_of_completeSpace (K := ℝ ∙ x)).codisjoint
    (LinearMap.eqOn_span' ?_) fun y hy ↦ ?_)
  · have : ((‖x‖ ^ 2) ^ 2)⁻¹ * (‖x‖ ^ 2) = (‖x‖ ^ 2)⁻¹ := by
      rw [← div_eq_inv_mul, sq (‖x‖ ^ 2), div_self_mul_self']
    simp [reflection_orthogonalComplement_singleton_eq_neg, real_inner_self_eq_norm_sq,
      two_mul, this, div_eq_mul_inv, mul_add, add_smul, mul_pow]
  · simp [Submodule.mem_orthogonal_singleton_iff_inner_right.1 hy,
      reflection_mem_subspace_eq_self hy, div_eq_mul_inv, mul_pow]


