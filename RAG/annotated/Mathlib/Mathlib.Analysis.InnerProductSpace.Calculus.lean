local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


/-- Derivative of the inner product. -/
def fderivInnerCLM (p : E × E) : E × E →L[ℝ] 𝕜 :=
  isBoundedBilinearMap_inner.deriv p


@[simp]
theorem fderivInnerCLM_apply (p x : E × E) : fderivInnerCLM 𝕜 p x = ⟪p.1, x.2⟫ + ⟪x.1, p.2⟫ :=
  rfl


theorem contDiff_inner {n} : ContDiff ℝ n fun p : E × E => ⟪p.1, p.2⟫ :=
  isBoundedBilinearMap_inner.contDiff


theorem contDiffAt_inner {p : E × E} {n} : ContDiffAt ℝ n (fun p : E × E => ⟪p.1, p.2⟫) p :=
  ContDiff.contDiffAt contDiff_inner


theorem differentiable_inner : Differentiable ℝ fun p : E × E => ⟪p.1, p.2⟫ :=
  isBoundedBilinearMap_inner.differentiableAt


theorem ContDiffWithinAt.inner (hf : ContDiffWithinAt ℝ n f s x) (hg : ContDiffWithinAt ℝ n g s x) :
    ContDiffWithinAt ℝ n (fun x => ⟪f x, g x⟫) s x :=
  contDiffAt_inner.comp_contDiffWithinAt x (hf.prod hg)


nonrec theorem ContDiffAt.inner (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x) :
    ContDiffAt ℝ n (fun x => ⟪f x, g x⟫) x :=
  hf.inner 𝕜 hg


theorem ContDiffOn.inner (hf : ContDiffOn ℝ n f s) (hg : ContDiffOn ℝ n g s) :
    ContDiffOn ℝ n (fun x => ⟪f x, g x⟫) s := fun x hx => (hf x hx).inner 𝕜 (hg x hx)


theorem ContDiff.inner (hf : ContDiff ℝ n f) (hg : ContDiff ℝ n g) :
    ContDiff ℝ n fun x => ⟪f x, g x⟫ :=
  contDiff_inner.comp (hf.prod hg)


theorem HasFDerivWithinAt.inner (hf : HasFDerivWithinAt f f' s x)
    (hg : HasFDerivWithinAt g g' s x) :
    HasFDerivWithinAt (fun t => ⟪f t, g t⟫) ((fderivInnerCLM 𝕜 (f x, g x)).comp <| f'.prod g') s
      x := by
  exact isBoundedBilinearMap_inner (𝕜 := 𝕜) (E := E)
    |>.hasFDerivAt (f x, g x) |>.comp_hasFDerivWithinAt x (hf.prod hg)


theorem HasStrictFDerivAt.inner (hf : HasStrictFDerivAt f f' x) (hg : HasStrictFDerivAt g g' x) :
    HasStrictFDerivAt (fun t => ⟪f t, g t⟫) ((fderivInnerCLM 𝕜 (f x, g x)).comp <| f'.prod g') x :=
  isBoundedBilinearMap_inner (𝕜 := 𝕜) (E := E)
    |>.hasStrictFDerivAt (f x, g x) |>.comp x (hf.prod hg)


theorem HasFDerivAt.inner (hf : HasFDerivAt f f' x) (hg : HasFDerivAt g g' x) :
    HasFDerivAt (fun t => ⟪f t, g t⟫) ((fderivInnerCLM 𝕜 (f x, g x)).comp <| f'.prod g') x := by
  exact isBoundedBilinearMap_inner (𝕜 := 𝕜) (E := E)
    |>.hasFDerivAt (f x, g x) |>.comp x (hf.prod hg)


theorem HasDerivWithinAt.inner {f g : ℝ → E} {f' g' : E} {s : Set ℝ} {x : ℝ}
    (hf : HasDerivWithinAt f f' s x) (hg : HasDerivWithinAt g g' s x) :
    HasDerivWithinAt (fun t => ⟪f t, g t⟫) (⟪f x, g'⟫ + ⟪f', g x⟫) s x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : NormedSpace Real E
    f g : Real → E
    f' g' : E
    s : Set Real
    x : Real
    hf : HasDerivWithinAt f f' s x
    hg : HasDerivWithinAt g g' s x
    ⊢ HasDerivWithinAt (fun t => Inner.inner (f t) (g t)) (HAdd.hAdd (Inner.inner  …
  -/
  simpa using (hf.hasFDerivWithinAt.inner 𝕜 hg.hasFDerivWithinAt).hasDerivWithinAt
  /-
    🎉 no goals
  -/


theorem HasDerivAt.inner {f g : ℝ → E} {f' g' : E} {x : ℝ} :
    HasDerivAt f f' x → HasDerivAt g g' x →
      HasDerivAt (fun t => ⟪f t, g t⟫) (⟪f x, g'⟫ + ⟪f', g x⟫) x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : NormedSpace Real E
    f g : Real → E
    f' g' : E
    x : Real
    ⊢ HasDerivAt f f' x → HasDerivAt g g' x → HasDerivAt (fun t => Inner.inner (f  …
  -/
  simpa only [← hasDerivWithinAt_univ] using HasDerivWithinAt.inner 𝕜
  /-
    🎉 no goals
  -/


theorem DifferentiableWithinAt.inner (hf : DifferentiableWithinAt ℝ f s x)
    (hg : DifferentiableWithinAt ℝ g s x) : DifferentiableWithinAt ℝ (fun x => ⟪f x, g x⟫) s x :=
  (hf.hasFDerivWithinAt.inner 𝕜 hg.hasFDerivWithinAt).differentiableWithinAt


theorem DifferentiableAt.inner (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x) :
    DifferentiableAt ℝ (fun x => ⟪f x, g x⟫) x :=
  (hf.hasFDerivAt.inner 𝕜 hg.hasFDerivAt).differentiableAt


theorem DifferentiableOn.inner (hf : DifferentiableOn ℝ f s) (hg : DifferentiableOn ℝ g s) :
    DifferentiableOn ℝ (fun x => ⟪f x, g x⟫) s := fun x hx => (hf x hx).inner 𝕜 (hg x hx)


theorem Differentiable.inner (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) :
    Differentiable ℝ fun x => ⟪f x, g x⟫ := fun x => (hf x).inner 𝕜 (hg x)


theorem fderiv_inner_apply (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x) (y : G) :
    fderiv ℝ (fun t => ⟪f t, g t⟫) x y = ⟪f x, fderiv ℝ g x y⟫ + ⟪fderiv ℝ f x y, g x⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : NormedSpace Real E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    f g : G → E
    x : G
    hf : DifferentiableAt Real f x
    hg : DifferentiableAt Real g x
    y : G
    ⊢ Eq ((fderiv Real (fun t => Inner.inner (f t) (g t)) x) y) (HAdd.hAdd (Inner. …
  -/
  rw [(hf.hasFDerivAt.inner 𝕜 hg.hasFDerivAt).fderiv]; rfl
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem deriv_inner_apply {f g : ℝ → E} {x : ℝ} (hf : DifferentiableAt ℝ f x)
    (hg : DifferentiableAt ℝ g x) :
    deriv (fun t => ⟪f t, g t⟫) x = ⟪f x, deriv g x⟫ + ⟪deriv f x, g x⟫ :=
  (hf.hasDerivAt.inner 𝕜 hg.hasDerivAt).deriv


theorem contDiff_norm_sq : ContDiff ℝ n fun x : E => ‖x‖ ^ 2 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : NormedSpace Real E
    n : WithTop ENat
    ⊢ ContDiff Real n fun x => HPow.hPow (Norm.norm x) 2
  -/
  convert (reCLM : 𝕜 →L[ℝ] ℝ).contDiff.comp ((contDiff_id (E := E)).inner 𝕜 (contDiff_id (E := E)))
  /-
    case h.e'_10.h
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : NormedSpace Real E
    n : WithTop ENat
    x✝ : E
    ⊢ Eq (HPow.hPow (Norm.norm x✝) 2) (Function.comp (⇑RCLike.reCLM) (fun x => Inn …
  -/
  exact (inner_self_eq_norm_sq _).symm
  /-
    🎉 no goals
  -/


theorem ContDiff.norm_sq (hf : ContDiff ℝ n f) : ContDiff ℝ n fun x => ‖f x‖ ^ 2 :=
  (contDiff_norm_sq 𝕜).comp hf


theorem ContDiffWithinAt.norm_sq (hf : ContDiffWithinAt ℝ n f s x) :
    ContDiffWithinAt ℝ n (fun y => ‖f y‖ ^ 2) s x :=
  (contDiff_norm_sq 𝕜).contDiffAt.comp_contDiffWithinAt x hf


nonrec theorem ContDiffAt.norm_sq (hf : ContDiffAt ℝ n f x) : ContDiffAt ℝ n (‖f ·‖ ^ 2) x :=
  hf.norm_sq 𝕜


theorem contDiffAt_norm {x : E} (hx : x ≠ 0) : ContDiffAt ℝ n norm x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : NormedSpace Real E
    n : WithTop ENat
    x : E
    hx : Ne x 0
    ⊢ ContDiffAt Real n Norm.norm x
  -/
  have : ‖id x‖ ^ 2 ≠ 0 := pow_ne_zero 2 (norm_pos_iff.2 hx).ne'
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : NormedSpace Real E
    n : WithTop ENat
    x : E
    hx : Ne x 0
    this : Ne (HPow.hPow (Norm.norm (id x)) 2) 0
    ⊢ ContDiffAt Real n Norm.norm x
  -/
  simpa only [id, sqrt_sq, norm_nonneg] using (contDiffAt_id.norm_sq 𝕜).sqrt this
  /-
    🎉 no goals
  -/


theorem ContDiffAt.norm (hf : ContDiffAt ℝ n f x) (h0 : f x ≠ 0) :
    ContDiffAt ℝ n (fun y => ‖f y‖) x :=
  (contDiffAt_norm 𝕜 h0).comp x hf


theorem ContDiffAt.dist (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x) (hne : f x ≠ g x) :
    ContDiffAt ℝ n (fun y => dist (f y) (g y)) x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : NormedSpace Real E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    f g : G → E
    x : G
    n : WithTop ENat
    hf : ContDiffAt Real n f x
    hg : ContDiffAt Real n g x
    hne : Ne (f x) (g x)
    ⊢ ContDiffAt Real n (fun y => Dist.dist (f y) (g y)) x
  -/
  simp only [dist_eq_norm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : NormedSpace Real E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    f g : G → E
    x : G
    n : WithTop ENat
    hf : ContDiffAt Real n f x
    hg : ContDiffAt Real n g x
    hne : Ne (f x) (g x)
    ⊢ ContDiffAt Real n (fun y => Norm.norm (HSub.hSub (f y) (g y))) x
  -/
  exact (hf.sub hg).norm 𝕜 (sub_ne_zero.2 hne)
  /-
    🎉 no goals
  -/


theorem ContDiffWithinAt.norm (hf : ContDiffWithinAt ℝ n f s x) (h0 : f x ≠ 0) :
    ContDiffWithinAt ℝ n (fun y => ‖f y‖) s x :=
  (contDiffAt_norm 𝕜 h0).comp_contDiffWithinAt x hf


theorem ContDiffWithinAt.dist (hf : ContDiffWithinAt ℝ n f s x) (hg : ContDiffWithinAt ℝ n g s x)
    (hne : f x ≠ g x) : ContDiffWithinAt ℝ n (fun y => dist (f y) (g y)) s x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : NormedSpace Real E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    f g : G → E
    s : Set G
    x : G
    n : WithTop ENat
    hf : ContDiffWithinAt Real n f s x
    hg : ContDiffWithinAt Real n g s x
    hne : Ne (f x) (g x)
    ⊢ ContDiffWithinAt Real n (fun y => Dist.dist (f y) (g y)) s x
  -/
  simp only [dist_eq_norm]; exact (hf.sub hg).norm 𝕜 (sub_ne_zero.2 hne)
                            /-
                              🎉 no goals
                            -/


theorem ContDiffOn.norm_sq (hf : ContDiffOn ℝ n f s) : ContDiffOn ℝ n (fun y => ‖f y‖ ^ 2) s :=
  fun x hx => (hf x hx).norm_sq 𝕜


theorem ContDiffOn.norm (hf : ContDiffOn ℝ n f s) (h0 : ∀ x ∈ s, f x ≠ 0) :
    ContDiffOn ℝ n (fun y => ‖f y‖) s := fun x hx => (hf x hx).norm 𝕜 (h0 x hx)


theorem ContDiffOn.dist (hf : ContDiffOn ℝ n f s) (hg : ContDiffOn ℝ n g s)
    (hne : ∀ x ∈ s, f x ≠ g x) : ContDiffOn ℝ n (fun y => dist (f y) (g y)) s := fun x hx =>
  (hf x hx).dist 𝕜 (hg x hx) (hne x hx)


theorem ContDiff.norm (hf : ContDiff ℝ n f) (h0 : ∀ x, f x ≠ 0) : ContDiff ℝ n fun y => ‖f y‖ :=
  contDiff_iff_contDiffAt.2 fun x => hf.contDiffAt.norm 𝕜 (h0 x)


theorem ContDiff.dist (hf : ContDiff ℝ n f) (hg : ContDiff ℝ n g) (hne : ∀ x, f x ≠ g x) :
    ContDiff ℝ n fun y => dist (f y) (g y) :=
  contDiff_iff_contDiffAt.2 fun x => hf.contDiffAt.dist 𝕜 hg.contDiffAt (hne x)


theorem hasStrictFDerivAt_norm_sq (x : F) :
    HasStrictFDerivAt (fun x => ‖x‖ ^ 2) (2 • (innerSL ℝ x)) x := by
  /-
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x : F
    ⊢ HasStrictFDerivAt (fun x => HPow.hPow (Norm.norm x) 2) (HSMul.hSMul 2 ((inne …
  -/
  simp only [sq, ← @inner_self_eq_norm_mul_norm ℝ]
  /-
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x : F
    ⊢ HasStrictFDerivAt (fun x => RCLike.re (Inner.inner x x)) (HSMul.hSMul 2 ((in …
  -/
  convert (hasStrictFDerivAt_id x).inner ℝ (hasStrictFDerivAt_id x)
  /-
    case h.e'_12.h.h.h
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x : F
    e_4✝ : Eq NormedAddCommGroup.toAddCommGroup SeminormedAddCommGroup.toAddCommGr …
    he✝¹ : Eq NormedSpace.toModule NormedSpace.toModule
    e_8✝ : Eq Real.instAddCommGroup SeminormedAddCommGroup.toAddCommGroup
    he✝ : Eq NormedSpace.toModule NormedSpace.toModule
    ⊢ Eq (HSMul.hSMul 2 ((innerSL Real) x)) ((fderivInnerCLM Real { fst := id x, s …
  -/
  ext y
  /-
    case h.e'_12.h.h.h.h
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x : F
    e_4✝ : Eq NormedAddCommGroup.toAddCommGroup SeminormedAddCommGroup.toAddCommGr …
    he✝¹ : Eq NormedSpace.toModule NormedSpace.toModule
    e_8✝ : Eq Real.instAddCommGroup SeminormedAddCommGroup.toAddCommGroup
    he✝ : Eq NormedSpace.toModule NormedSpace.toModule
    y : F
    ⊢ Eq ((HSMul.hSMul 2 ((innerSL Real) x)) y) (((fderivInnerCLM Real { fst := id …
  -/
  simp [two_smul, real_inner_comm]
  /-
    🎉 no goals
  -/


theorem HasFDerivAt.norm_sq {f : G → F} {f' : G →L[ℝ] F} (hf : HasFDerivAt f f' x) :
    HasFDerivAt (‖f ·‖ ^ 2) (2 • (innerSL ℝ (f x)).comp f') x :=
  (hasStrictFDerivAt_norm_sq _).hasFDerivAt.comp x hf


theorem HasDerivAt.norm_sq {f : ℝ → F} {f' : F} {x : ℝ} (hf : HasDerivAt f f' x) :
    HasDerivAt (‖f ·‖ ^ 2) (2 * Inner.inner (f x) f') x := by
  /-
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    f : Real → F
    f' : F
    x : Real
    hf : HasDerivAt f f' x
    ⊢ HasDerivAt (fun x => HPow.hPow (Norm.norm (f x)) 2) (HMul.hMul 2 (Inner.inne …
  -/
  simpa using hf.hasFDerivAt.norm_sq.hasDerivAt
  /-
    🎉 no goals
  -/


theorem HasFDerivWithinAt.norm_sq {f : G → F} {f' : G →L[ℝ] F} (hf : HasFDerivWithinAt f f' s x) :
    HasFDerivWithinAt (‖f ·‖ ^ 2) (2 • (innerSL ℝ (f x)).comp f') s x :=
  (hasStrictFDerivAt_norm_sq _).hasFDerivAt.comp_hasFDerivWithinAt x hf


theorem HasDerivWithinAt.norm_sq {f : ℝ → F} {f' : F} {s : Set ℝ} {x : ℝ}
    (hf : HasDerivWithinAt f f' s x) :
    HasDerivWithinAt (‖f ·‖ ^ 2) (2 * Inner.inner (f x) f') s x := by
  /-
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    f : Real → F
    f' : F
    s : Set Real
    x : Real
    hf : HasDerivWithinAt f f' s x
    ⊢ HasDerivWithinAt (fun x => HPow.hPow (Norm.norm (f x)) 2) (HMul.hMul 2 (Inne …
  -/
  simpa using hf.hasFDerivWithinAt.norm_sq.hasDerivWithinAt
  /-
    🎉 no goals
  -/


theorem DifferentiableAt.norm_sq (hf : DifferentiableAt ℝ f x) :
    DifferentiableAt ℝ (fun y => ‖f y‖ ^ 2) x :=
  ((contDiffAt_id.norm_sq 𝕜).differentiableAt le_rfl).comp x hf


theorem DifferentiableAt.norm (hf : DifferentiableAt ℝ f x) (h0 : f x ≠ 0) :
    DifferentiableAt ℝ (fun y => ‖f y‖) x :=
  ((contDiffAt_norm 𝕜 h0).differentiableAt le_rfl).comp x hf


theorem DifferentiableAt.dist (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (hne : f x ≠ g x) : DifferentiableAt ℝ (fun y => dist (f y) (g y)) x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : NormedSpace Real E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    f g : G → E
    x : G
    hf : DifferentiableAt Real f x
    hg : DifferentiableAt Real g x
    hne : Ne (f x) (g x)
    ⊢ DifferentiableAt Real (fun y => Dist.dist (f y) (g y)) x
  -/
  simp only [dist_eq_norm]; exact (hf.sub hg).norm 𝕜 (sub_ne_zero.2 hne)
                            /-
                              🎉 no goals
                            -/


theorem Differentiable.norm_sq (hf : Differentiable ℝ f) : Differentiable ℝ fun y => ‖f y‖ ^ 2 :=
  fun x => (hf x).norm_sq 𝕜


theorem Differentiable.norm (hf : Differentiable ℝ f) (h0 : ∀ x, f x ≠ 0) :
    Differentiable ℝ fun y => ‖f y‖ := fun x => (hf x).norm 𝕜 (h0 x)


theorem Differentiable.dist (hf : Differentiable ℝ f) (hg : Differentiable ℝ g)
    (hne : ∀ x, f x ≠ g x) : Differentiable ℝ fun y => dist (f y) (g y) := fun x =>
  (hf x).dist 𝕜 (hg x) (hne x)


theorem DifferentiableWithinAt.norm_sq (hf : DifferentiableWithinAt ℝ f s x) :
    DifferentiableWithinAt ℝ (fun y => ‖f y‖ ^ 2) s x :=
  ((contDiffAt_id.norm_sq 𝕜).differentiableAt le_rfl).comp_differentiableWithinAt x hf


theorem DifferentiableWithinAt.norm (hf : DifferentiableWithinAt ℝ f s x) (h0 : f x ≠ 0) :
    DifferentiableWithinAt ℝ (fun y => ‖f y‖) s x :=
  ((contDiffAt_id.norm 𝕜 h0).differentiableAt le_rfl).comp_differentiableWithinAt x hf


theorem DifferentiableWithinAt.dist (hf : DifferentiableWithinAt ℝ f s x)
    (hg : DifferentiableWithinAt ℝ g s x) (hne : f x ≠ g x) :
    DifferentiableWithinAt ℝ (fun y => dist (f y) (g y)) s x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : NormedSpace Real E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    f g : G → E
    s : Set G
    x : G
    hf : DifferentiableWithinAt Real f s x
    hg : DifferentiableWithinAt Real g s x
    hne : Ne (f x) (g x)
    ⊢ DifferentiableWithinAt Real (fun y => Dist.dist (f y) (g y)) s x
  -/
  simp only [dist_eq_norm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : NormedSpace Real E
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    f g : G → E
    s : Set G
    x : G
    hf : DifferentiableWithinAt Real f s x
    hg : DifferentiableWithinAt Real g s x
    hne : Ne (f x) (g x)
    ⊢ DifferentiableWithinAt Real (fun y => Norm.norm (HSub.hSub (f y) (g y))) s x
  -/
  exact (hf.sub hg).norm 𝕜 (sub_ne_zero.2 hne)
  /-
    🎉 no goals
  -/


theorem DifferentiableOn.norm_sq (hf : DifferentiableOn ℝ f s) :
    DifferentiableOn ℝ (fun y => ‖f y‖ ^ 2) s := fun x hx => (hf x hx).norm_sq 𝕜


theorem DifferentiableOn.norm (hf : DifferentiableOn ℝ f s) (h0 : ∀ x ∈ s, f x ≠ 0) :
    DifferentiableOn ℝ (fun y => ‖f y‖) s := fun x hx => (hf x hx).norm 𝕜 (h0 x hx)


theorem DifferentiableOn.dist (hf : DifferentiableOn ℝ f s) (hg : DifferentiableOn ℝ g s)
    (hne : ∀ x ∈ s, f x ≠ g x) : DifferentiableOn ℝ (fun y => dist (f y) (g y)) s := fun x hx =>
  (hf x hx).dist 𝕜 (hg x hx) (hne x hx)


theorem differentiableWithinAt_euclidean :
    DifferentiableWithinAt 𝕜 f t y ↔ ∀ i, DifferentiableWithinAt 𝕜 (fun x => f x i) t y :=
  differentiableWithinAt_piLp _


theorem differentiableAt_euclidean :
    DifferentiableAt 𝕜 f y ↔ ∀ i, DifferentiableAt 𝕜 (fun x => f x i) y :=
  differentiableAt_piLp _


theorem differentiableOn_euclidean :
    DifferentiableOn 𝕜 f t ↔ ∀ i, DifferentiableOn 𝕜 (fun x => f x i) t :=
  differentiableOn_piLp _


theorem differentiable_euclidean : Differentiable 𝕜 f ↔ ∀ i, Differentiable 𝕜 fun x => f x i :=
  differentiable_piLp _


theorem hasStrictFDerivAt_euclidean :
    HasStrictFDerivAt f f' y ↔
      ∀ i, HasStrictFDerivAt (fun x => f x i) (PiLp.proj _ _ i ∘L f') y :=
  hasStrictFDerivAt_piLp _


theorem hasFDerivWithinAt_euclidean :
    HasFDerivWithinAt f f' t y ↔
      ∀ i, HasFDerivWithinAt (fun x => f x i) (PiLp.proj _ _ i ∘L f') t y :=
  hasFDerivWithinAt_piLp _


theorem contDiffWithinAt_euclidean {n : WithTop ℕ∞} :
    ContDiffWithinAt 𝕜 n f t y ↔ ∀ i, ContDiffWithinAt 𝕜 n (fun x => f x i) t y :=
  contDiffWithinAt_piLp _


theorem contDiffAt_euclidean {n : WithTop ℕ∞} :
    ContDiffAt 𝕜 n f y ↔ ∀ i, ContDiffAt 𝕜 n (fun x => f x i) y :=
  contDiffAt_piLp _


theorem contDiffOn_euclidean {n : WithTop ℕ∞} :
    ContDiffOn 𝕜 n f t ↔ ∀ i, ContDiffOn 𝕜 n (fun x => f x i) t :=
  contDiffOn_piLp _


theorem contDiff_euclidean {n : WithTop ℕ∞} : ContDiff 𝕜 n f ↔ ∀ i, ContDiff 𝕜 n fun x => f x i :=
  contDiff_piLp _


theorem PartialHomeomorph.contDiff_univUnitBall : ContDiff ℝ n (univUnitBall : E → E) := by
  /-
    n : ENat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    ⊢ ContDiff Real ↑n ↑PartialHomeomorph.univUnitBall
  -/
  suffices ContDiff ℝ n fun x : E => (√(1 + ‖x‖ ^ 2 : ℝ))⁻¹ from this.smul contDiff_id
  /-
    n : ENat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    ⊢ ContDiff Real ↑n fun x => Inv.inv (HAdd.hAdd 1 (HPow.hPow (Norm.norm x) 2)). …
  -/
  have h : ∀ x : E, (0 : ℝ) < (1 : ℝ) + ‖x‖ ^ 2 := fun x => by positivity
  /-
    n : ENat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    h : ∀ (x : E), LT.lt 0 (HAdd.hAdd 1 (HPow.hPow (Norm.norm x) 2))
    ⊢ ContDiff Real ↑n fun x => Inv.inv (HAdd.hAdd 1 (HPow.hPow (Norm.norm x) 2)). …
  -/
  refine ContDiff.inv ?_ fun x => Real.sqrt_ne_zero'.mpr (h x)
  /-
    n : ENat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    h : ∀ (x : E), LT.lt 0 (HAdd.hAdd 1 (HPow.hPow (Norm.norm x) 2))
    ⊢ ContDiff Real ↑n fun x => (HAdd.hAdd 1 (HPow.hPow (Norm.norm x) 2)).sqrt
  -/
  exact (contDiff_const.add <| contDiff_norm_sq ℝ).sqrt fun x => (h x).ne'
  /-
    🎉 no goals
  -/


theorem PartialHomeomorph.contDiffOn_univUnitBall_symm :
    ContDiffOn ℝ n univUnitBall.symm (ball (0 : E) 1) := fun y hy ↦ by
  /-
    n : ENat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    y : E
    hy : Membership.mem (Metric.ball 0 1) y
    ⊢ ContDiffWithinAt Real (↑n) (↑PartialHomeomorph.univUnitBall.symm) (Metric.ba …
  -/
  apply ContDiffAt.contDiffWithinAt
  /-
    case h
    n : ENat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    y : E
    hy : Membership.mem (Metric.ball 0 1) y
    ⊢ ContDiffAt Real (↑n) (↑PartialHomeomorph.univUnitBall.symm) y
  -/
  suffices ContDiffAt ℝ n (fun y : E => (√(1 - ‖y‖ ^ 2 : ℝ))⁻¹) y from this.smul contDiffAt_id
  have h : (0 : ℝ) < (1 : ℝ) - ‖(y : E)‖ ^ 2 := by
    rwa [mem_ball_zero_iff, ← _root_.abs_one, ← abs_norm, ← sq_lt_sq, one_pow, ← sub_pos] at hy
  /-
    case h
    n : ENat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    y : E
    hy : Membership.mem (Metric.ball 0 1) y
    h : LT.lt 0 (HSub.hSub 1 (HPow.hPow (Norm.norm y) 2))
    ⊢ ContDiffAt Real (↑n) (fun y => Inv.inv (HSub.hSub 1 (HPow.hPow (Norm.norm y) …
  -/
  refine ContDiffAt.inv ?_ (Real.sqrt_ne_zero'.mpr h)
  /-
    case h
    n : ENat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    y : E
    hy : Membership.mem (Metric.ball 0 1) y
    h : LT.lt 0 (HSub.hSub 1 (HPow.hPow (Norm.norm y) 2))
    ⊢ ContDiffAt Real (↑n) (fun y => (HSub.hSub 1 (HPow.hPow (Norm.norm y) 2)).sqr …
  -/
  change ContDiffAt ℝ n ((fun y ↦ √(y)) ∘ fun y ↦ (1 - ‖y‖ ^ 2)) y
  /-
    case h
    n : ENat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    y : E
    hy : Membership.mem (Metric.ball 0 1) y
    h : LT.lt 0 (HSub.hSub 1 (HPow.hPow (Norm.norm y) 2))
    ⊢ ContDiffAt Real (↑n) (Function.comp (fun y => y.sqrt) fun y => HSub.hSub 1 ( …
  -/
  refine (contDiffAt_sqrt h.ne').comp y ?_
  /-
    case h
    n : ENat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    y : E
    hy : Membership.mem (Metric.ball 0 1) y
    h : LT.lt 0 (HSub.hSub 1 (HPow.hPow (Norm.norm y) 2))
    ⊢ ContDiffAt Real (↑n) (fun y => HSub.hSub 1 (HPow.hPow (Norm.norm y) 2)) y
  -/
  exact contDiffAt_const.sub (contDiff_norm_sq ℝ).contDiffAt
  /-
    🎉 no goals
  -/


theorem Homeomorph.contDiff_unitBall : ContDiff ℝ n fun x : E => (unitBall x : E) :=
  PartialHomeomorph.contDiff_univUnitBall


theorem contDiff_unitBallBall (hr : 0 < r) : ContDiff ℝ n (unitBallBall c r hr) :=
  (contDiff_id.const_smul _).add contDiff_const


theorem contDiff_unitBallBall_symm (hr : 0 < r) : ContDiff ℝ n (unitBallBall c r hr).symm :=
  (contDiff_id.sub contDiff_const).const_smul _


theorem contDiff_univBall : ContDiff ℝ n (univBall c r) := by
  /-
    n : ENat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    c : E
    r : Real
    ⊢ ContDiff Real ↑n ↑(PartialHomeomorph.univBall c r)
  -/
  unfold univBall; split_ifs with h
    /-
      case pos
      n : ENat
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      c : E
      r : Real
      h : LT.lt 0 r
      ⊢ ContDiff Real ↑n ↑(PartialHomeomorph.univUnitBall.trans' (PartialHomeomorph. …
    -/
  · exact (contDiff_unitBallBall h).comp contDiff_univUnitBall
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : ENat
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      c : E
      r : Real
      h : Not (LT.lt 0 r)
      ⊢ ContDiff Real ↑n ↑(IsometryEquiv.vaddConst c).toHomeomorph.toPartialHomeomorph
    -/
  · exact contDiff_id.add contDiff_const
    /-
      🎉 no goals
    -/


theorem contDiffOn_univBall_symm :
    ContDiffOn ℝ n (univBall c r).symm (ball c r) := by
  /-
    n : ENat
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    c : E
    r : Real
    ⊢ ContDiffOn Real (↑n) (↑(PartialHomeomorph.univBall c r).symm) (Metric.ball c …
  -/
  unfold univBall; split_ifs with h
    /-
      case pos
      n : ENat
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      c : E
      r : Real
      h : LT.lt 0 r
      ⊢ ContDiffOn Real (↑n) (↑(PartialHomeomorph.univUnitBall.trans' (PartialHomeom …
    -/
  · refine contDiffOn_univUnitBall_symm.comp (contDiff_unitBallBall_symm h).contDiffOn ?_
    /-
      case pos
      n : ENat
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      c : E
      r : Real
      h : LT.lt 0 r
      ⊢ Set.MapsTo (↑(PartialHomeomorph.unitBallBall c r h).symm) (Metric.ball c r)  …
    -/
    rw [← unitBallBall_source c r h, ← unitBallBall_target c r h]
    /-
      case pos
      n : ENat
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      c : E
      r : Real
      h : LT.lt 0 r
      ⊢ Set.MapsTo (↑(PartialHomeomorph.unitBallBall c r h).symm) (PartialHomeomorph …
    -/
    apply PartialHomeomorph.symm_mapsTo
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : ENat
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      c : E
      r : Real
      h : Not (LT.lt 0 r)
      ⊢ ContDiffOn Real (↑n) (↑(IsometryEquiv.vaddConst c).toHomeomorph.toPartialHom …
    -/
  · exact contDiffOn_id.sub contDiffOn_const
    /-
      🎉 no goals
    -/


