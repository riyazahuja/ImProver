/-- `f` has the derivative `f'` at the point `x` along the direction `v` in the set `s`.
That is, `f (x + t v) = f x + t • f' + o (t)` when `t` tends to `0` and `x + t v ∈ s`.
Note that this definition is less well behaved than the total Fréchet derivative, which
should generally be favored over this one. -/
def HasLineDerivWithinAt (f : E → F) (f' : F) (s : Set E) (x : E) (v : E) :=
  HasDerivWithinAt (fun t ↦ f (x + t • v)) f' ((fun t ↦ x + t • v) ⁻¹' s) (0 : 𝕜)


/-- `f` has the derivative `f'` at the point `x` along the direction `v`.
That is, `f (x + t v) = f x + t • f' + o (t)` when `t` tends to `0`.
Note that this definition is less well behaved than the total Fréchet derivative, which
should generally be favored over this one. -/
def HasLineDerivAt (f : E → F) (f' : F) (x : E) (v : E) :=
  HasDerivAt (fun t ↦ f (x + t • v)) f' (0 : 𝕜)


/-- `f` is line-differentiable at the point `x` in the direction `v` in the set `s` if there
exists `f'` such that `f (x + t v) = f x + t • f' + o (t)` when `t` tends to `0` and `x + t v ∈ s`.
-/
def LineDifferentiableWithinAt (f : E → F) (s : Set E) (x : E) (v : E) : Prop :=
  DifferentiableWithinAt 𝕜 (fun t ↦ f (x + t • v)) ((fun t ↦ x + t • v) ⁻¹' s) (0 : 𝕜)


/-- `f` is line-differentiable at the point `x` in the direction `v` if there
exists `f'` such that `f (x + t v) = f x + t • f' + o (t)` when `t` tends to `0`. -/
def LineDifferentiableAt (f : E → F) (x : E) (v : E) : Prop :=
  DifferentiableAt 𝕜 (fun t ↦ f (x + t • v)) (0 : 𝕜)


/-- Line derivative of `f` at the point `x` in the direction `v` within the set `s`, if it exists.
Zero otherwise.

If the line derivative exists (i.e., `∃ f', HasLineDerivWithinAt 𝕜 f f' s x v`), then
`f (x + t v) = f x + t lineDerivWithin 𝕜 f s x v + o (t)` when `t` tends to `0` and `x + t v ∈ s`.
-/
def lineDerivWithin (f : E → F) (s : Set E) (x : E) (v : E) : F :=
  derivWithin (fun t ↦ f (x + t • v)) ((fun t ↦ x + t • v) ⁻¹' s) (0 : 𝕜)


/-- Line derivative of `f` at the point `x` in the direction `v`, if it exists.  Zero otherwise.

If the line derivative exists (i.e., `∃ f', HasLineDerivAt 𝕜 f f' x v`), then
`f (x + t v) = f x + t lineDeriv 𝕜 f x v + o (t)` when `t` tends to `0`.
-/
def lineDeriv (f : E → F) (x : E) (v : E) : F :=
  deriv (fun t ↦ f (x + t • v)) (0 : 𝕜)


lemma HasLineDerivWithinAt.mono (hf : HasLineDerivWithinAt 𝕜 f f' s x v) (hst : t ⊆ s) :
    HasLineDerivWithinAt 𝕜 f f' t x v :=
  HasDerivWithinAt.mono hf (preimage_mono hst)


lemma HasLineDerivAt.hasLineDerivWithinAt (hf : HasLineDerivAt 𝕜 f f' x v) (s : Set E) :
    HasLineDerivWithinAt 𝕜 f f' s x v :=
  HasDerivAt.hasDerivWithinAt hf


lemma HasLineDerivWithinAt.lineDifferentiableWithinAt (hf : HasLineDerivWithinAt 𝕜 f f' s x v) :
    LineDifferentiableWithinAt 𝕜 f s x v :=
  HasDerivWithinAt.differentiableWithinAt hf


theorem HasLineDerivAt.lineDifferentiableAt (hf : HasLineDerivAt 𝕜 f f' x v) :
    LineDifferentiableAt 𝕜 f x v :=
  HasDerivAt.differentiableAt hf


theorem LineDifferentiableWithinAt.hasLineDerivWithinAt (h : LineDifferentiableWithinAt 𝕜 f s x v) :
    HasLineDerivWithinAt 𝕜 f (lineDerivWithin 𝕜 f s x v) s x v :=
  DifferentiableWithinAt.hasDerivWithinAt h


theorem LineDifferentiableAt.hasLineDerivAt (h : LineDifferentiableAt 𝕜 f x v) :
    HasLineDerivAt 𝕜 f (lineDeriv 𝕜 f x v) x v :=
  DifferentiableAt.hasDerivAt h


@[simp] lemma hasLineDerivWithinAt_univ :
    HasLineDerivWithinAt 𝕜 f f' univ x v ↔ HasLineDerivAt 𝕜 f f' x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    f' : F
    x v : E
    ⊢ Iff (HasLineDerivWithinAt 𝕜 f f' Set.univ x v) (HasLineDerivAt 𝕜 f f' x v)
  -/
  simp only [HasLineDerivWithinAt, HasLineDerivAt, preimage_univ, hasDerivWithinAt_univ]
  /-
    🎉 no goals
  -/


theorem lineDerivWithin_zero_of_not_lineDifferentiableWithinAt
    (h : ¬LineDifferentiableWithinAt 𝕜 f s x v) :
    lineDerivWithin 𝕜 f s x v = 0 :=
  derivWithin_zero_of_not_differentiableWithinAt h


theorem lineDeriv_zero_of_not_lineDifferentiableAt (h : ¬LineDifferentiableAt 𝕜 f x v) :
    lineDeriv 𝕜 f x v = 0 :=
  deriv_zero_of_not_differentiableAt h


theorem hasLineDerivAt_iff_isLittleO_nhds_zero :
    HasLineDerivAt 𝕜 f f' x v ↔
      (fun t : 𝕜 => f (x + t • v) - f x - t • f') =o[𝓝 0] fun t => t := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    f' : F
    x v : E
    ⊢ Iff (HasLineDerivAt 𝕜 f f' x v) (Asymptotics.IsLittleO (nhds 0) (fun t => HS …
  -/
  simp only [HasLineDerivAt, hasDerivAt_iff_isLittleO_nhds_zero, zero_add, zero_smul, add_zero]
  /-
    🎉 no goals
  -/


theorem HasLineDerivAt.unique (h₀ : HasLineDerivAt 𝕜 f f₀' x v) (h₁ : HasLineDerivAt 𝕜 f f₁' x v) :
    f₀' = f₁' :=
  HasDerivAt.unique h₀ h₁


protected theorem HasLineDerivAt.lineDeriv (h : HasLineDerivAt 𝕜 f f' x v) :
    lineDeriv 𝕜 f x v = f' := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    f' : F
    x v : E
    h : HasLineDerivAt 𝕜 f f' x v
    ⊢ Eq (lineDeriv 𝕜 f x v) f'
  -/
  rw [h.unique h.lineDifferentiableAt.hasLineDerivAt]
  /-
    🎉 no goals
  -/


theorem lineDifferentiableWithinAt_univ :
    LineDifferentiableWithinAt 𝕜 f univ x v ↔ LineDifferentiableAt 𝕜 f x v := by
  simp only [LineDifferentiableWithinAt, LineDifferentiableAt, preimage_univ,
    differentiableWithinAt_univ]


theorem LineDifferentiableAt.lineDifferentiableWithinAt (h : LineDifferentiableAt 𝕜 f x v) :
    LineDifferentiableWithinAt 𝕜 f s x v :=
  (differentiableWithinAt_univ.2 h).mono (subset_univ _)


@[simp]
theorem lineDerivWithin_univ : lineDerivWithin 𝕜 f univ x v = lineDeriv 𝕜 f x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    x v : E
    ⊢ Eq (lineDerivWithin 𝕜 f Set.univ x v) (lineDeriv 𝕜 f x v)
  -/
  simp [lineDerivWithin, lineDeriv]
  /-
    🎉 no goals
  -/


theorem LineDifferentiableWithinAt.mono (h : LineDifferentiableWithinAt 𝕜 f t x v) (st : s ⊆ t) :
    LineDifferentiableWithinAt 𝕜 f s x v :=
  (h.hasLineDerivWithinAt.mono st).lineDifferentiableWithinAt


theorem HasLineDerivWithinAt.congr_mono (h : HasLineDerivWithinAt 𝕜 f f' s x v) (ht : EqOn f₁ f t)
    (hx : f₁ x = f x) (h₁ : t ⊆ s) : HasLineDerivWithinAt 𝕜 f₁ f' t x v :=
                                                       /-
                                                         𝕜 : Type u_1
                                                         inst✝⁴ : NontriviallyNormedField 𝕜
                                                         F : Type u_2
                                                         inst✝³ : NormedAddCommGroup F
                                                         inst✝² : NormedSpace 𝕜 F
                                                         E : Type u_3
                                                         inst✝¹ : AddCommGroup E
                                                         inst✝ : Module 𝕜 E
                                                         f f₁ : E → F
                                                         f' : F
                                                         s t : Set E
                                                         x v : E
                                                         h : HasLineDerivWithinAt 𝕜 f f' s x v
                                                         ht : Set.EqOn f₁ f t
                                                         hx : Eq (f₁ x) (f x)
                                                         h₁ : HasSubset.Subset t s
                                                         ⊢ Eq (f₁ (HAdd.hAdd x (HSMul.hSMul 0 v))) (f (HAdd.hAdd x (HSMul.hSMul 0 v)))
                                                       -/
  HasDerivWithinAt.congr_mono h (fun _ hy ↦ ht hy) (by simpa using hx) (preimage_mono h₁)
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem HasLineDerivWithinAt.congr (h : HasLineDerivWithinAt 𝕜 f f' s x v) (hs : EqOn f₁ f s)
    (hx : f₁ x = f x) : HasLineDerivWithinAt 𝕜 f₁ f' s x v :=
  h.congr_mono hs hx (Subset.refl _)


theorem HasLineDerivWithinAt.congr' (h : HasLineDerivWithinAt 𝕜 f f' s x v)
    (hs : EqOn f₁ f s) (hx : x ∈ s) :
    HasLineDerivWithinAt 𝕜 f₁ f' s x v :=
  h.congr hs (hs hx)


theorem LineDifferentiableWithinAt.congr_mono (h : LineDifferentiableWithinAt 𝕜 f s x v)
    (ht : EqOn f₁ f t) (hx : f₁ x = f x) (h₁ : t ⊆ s) :
    LineDifferentiableWithinAt 𝕜 f₁ t x v :=
  (HasLineDerivWithinAt.congr_mono h.hasLineDerivWithinAt ht hx h₁).differentiableWithinAt


theorem LineDifferentiableWithinAt.congr (h : LineDifferentiableWithinAt 𝕜 f s x v)
    (ht : ∀ x ∈ s, f₁ x = f x) (hx : f₁ x = f x) :
    LineDifferentiableWithinAt 𝕜 f₁ s x v :=
  LineDifferentiableWithinAt.congr_mono h ht hx (Subset.refl _)


theorem lineDerivWithin_congr (hs : EqOn f₁ f s) (hx : f₁ x = f x) :
    lineDerivWithin 𝕜 f₁ s x v = lineDerivWithin 𝕜 f s x v :=
                                           /-
                                             𝕜 : Type u_1
                                             inst✝⁴ : NontriviallyNormedField 𝕜
                                             F : Type u_2
                                             inst✝³ : NormedAddCommGroup F
                                             inst✝² : NormedSpace 𝕜 F
                                             E : Type u_3
                                             inst✝¹ : AddCommGroup E
                                             inst✝ : Module 𝕜 E
                                             f f₁ : E → F
                                             s : Set E
                                             x v : E
                                             hs : Set.EqOn f₁ f s
                                             hx : Eq (f₁ x) (f x)
                                             ⊢ Eq (f₁ (HAdd.hAdd x (HSMul.hSMul 0 v))) (f (HAdd.hAdd x (HSMul.hSMul 0 v)))
                                           -/
  derivWithin_congr (fun _ hy ↦ hs hy) (by simpa using hx)
                                           /-
                                             🎉 no goals
                                           -/


theorem lineDerivWithin_congr' (hs : EqOn f₁ f s) (hx : x ∈ s) :
    lineDerivWithin 𝕜 f₁ s x v = lineDerivWithin 𝕜 f s x v :=
  lineDerivWithin_congr hs (hs hx)


theorem hasLineDerivAt_iff_tendsto_slope_zero :
    HasLineDerivAt 𝕜 f f' x v ↔
      Tendsto (fun (t : 𝕜) ↦ t⁻¹ • (f (x + t • v) - f x)) (𝓝[≠] 0) (𝓝 f') := by
  simp only [HasLineDerivAt, hasDerivAt_iff_tendsto_slope_zero, zero_add,
    zero_smul, add_zero]


alias ⟨HasLineDerivAt.tendsto_slope_zero, _⟩ := hasLineDerivAt_iff_tendsto_slope_zero


theorem HasLineDerivAt.tendsto_slope_zero_right [PartialOrder 𝕜] (h : HasLineDerivAt 𝕜 f f' x v) :
    Tendsto (fun (t : 𝕜) ↦ t⁻¹ • (f (x + t • v) - f x)) (𝓝[>] 0) (𝓝 f') :=
  h.tendsto_slope_zero.mono_left (nhdsGT_le_nhdsNE 0)


theorem HasLineDerivAt.tendsto_slope_zero_left [PartialOrder 𝕜] (h : HasLineDerivAt 𝕜 f f' x v) :
    Tendsto (fun (t : 𝕜) ↦ t⁻¹ • (f (x + t • v) - f x)) (𝓝[<] 0) (𝓝 f') :=
  h.tendsto_slope_zero.mono_left (nhdsLT_le_nhdsNE 0)


theorem HasLineDerivWithinAt.hasLineDerivAt'
    (h : HasLineDerivWithinAt 𝕜 f f' s x v) (hs : ∀ᶠ t : 𝕜 in 𝓝 0, x + t • v ∈ s) :
    HasLineDerivAt 𝕜 f f' x v :=
  h.hasDerivAt hs


theorem HasLineDerivWithinAt.mono_of_mem_nhdsWithin
    (h : HasLineDerivWithinAt 𝕜 f f' t x v) (hst : t ∈ 𝓝[s] x) :
    HasLineDerivWithinAt 𝕜 f f' s x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F
    f' : F
    s t : Set E
    x v : E
    h : HasLineDerivWithinAt 𝕜 f f' t x v
    hst : Membership.mem (nhdsWithin x s) t
    ⊢ HasLineDerivWithinAt 𝕜 f f' s x v
  -/
  apply HasDerivWithinAt.mono_of_mem_nhdsWithin h
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F
    f' : F
    s t : Set E
    x v : E
    h : HasLineDerivWithinAt 𝕜 f f' t x v
    hst : Membership.mem (nhdsWithin x s) t
    ⊢ Membership.mem (nhdsWithin 0 (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMu …
  -/
  apply ContinuousWithinAt.preimage_mem_nhdsWithin'' _ hst (by simp)
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F
    f' : F
    s t : Set E
    x v : E
    h : HasLineDerivWithinAt 𝕜 f f' t x v
    hst : Membership.mem (nhdsWithin x s) t
    ⊢ ContinuousWithinAt (fun t => HAdd.hAdd x (HSMul.hSMul t v)) (Set.preimage (f …
  -/
  apply Continuous.continuousWithinAt; fun_prop
                                       /-
                                         🎉 no goals
                                       -/


@[deprecated (since := "2024-10-31")]
alias HasLineDerivWithinAt.mono_of_mem := HasLineDerivWithinAt.mono_of_mem_nhdsWithin


theorem HasLineDerivWithinAt.hasLineDerivAt
    (h : HasLineDerivWithinAt 𝕜 f f' s x v) (hs : s ∈ 𝓝 x) :
    HasLineDerivAt 𝕜 f f' x v :=
                                                /-
                                                  𝕜 : Type u_1
                                                  inst✝⁴ : NontriviallyNormedField 𝕜
                                                  F : Type u_2
                                                  inst✝³ : NormedAddCommGroup F
                                                  inst✝² : NormedSpace 𝕜 F
                                                  E : Type u_3
                                                  inst✝¹ : NormedAddCommGroup E
                                                  inst✝ : NormedSpace 𝕜 E
                                                  f : E → F
                                                  f' : F
                                                  s : Set E
                                                  x v : E
                                                  h : HasLineDerivWithinAt 𝕜 f f' s x v
                                                  hs : Membership.mem (nhds x) s
                                                  ⊢ Continuous fun t => HAdd.hAdd x (HSMul.hSMul t v)
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
  h.hasLineDerivAt' <| (Continuous.tendsto' (by fun_prop) 0 _ (by simp)).eventually hs
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem LineDifferentiableWithinAt.lineDifferentiableAt (h : LineDifferentiableWithinAt 𝕜 f s x v)
    (hs : s ∈ 𝓝 x) : LineDifferentiableAt 𝕜 f x v :=
  (h.hasLineDerivWithinAt.hasLineDerivAt hs).lineDifferentiableAt


lemma HasFDerivWithinAt.hasLineDerivWithinAt (hf : HasFDerivWithinAt f L s x) (v : E) :
    HasLineDerivWithinAt 𝕜 f (L v) s x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F
    s : Set E
    x : E
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : HasFDerivWithinAt f L s x
    v : E
    ⊢ HasLineDerivWithinAt 𝕜 f (L v) s x v
  -/
  let F := fun (t : 𝕜) ↦ x + t • v
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F✝ : Type u_2
    inst✝³ : NormedAddCommGroup F✝
    inst✝² : NormedSpace 𝕜 F✝
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F✝
    s : Set E
    x : E
    L : ContinuousLinearMap (RingHom.id 𝕜) E F✝
    hf : HasFDerivWithinAt f L s x
    v : E
    F : 𝕜 → E := fun t => HAdd.hAdd x (HSMul.hSMul t v)
    ⊢ HasLineDerivWithinAt 𝕜 f (L v) s x v
  -/
  rw [show x = F (0 : 𝕜) by simp [F]] at hf
  have A : HasDerivWithinAt F (0 + (1 : 𝕜) • v) (F ⁻¹' s) 0 :=
    ((hasDerivAt_const (0 : 𝕜) x).add ((hasDerivAt_id' (0 : 𝕜)).smul_const v)).hasDerivWithinAt
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F✝ : Type u_2
    inst✝³ : NormedAddCommGroup F✝
    inst✝² : NormedSpace 𝕜 F✝
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F✝
    s : Set E
    x : E
    L : ContinuousLinearMap (RingHom.id 𝕜) E F✝
    v : E
    F : 𝕜 → E := fun t => HAdd.hAdd x (HSMul.hSMul t v)
    hf : HasFDerivWithinAt f L s (F 0)
    A : HasDerivWithinAt F (HAdd.hAdd 0 (HSMul.hSMul 1 v)) (Set.preimage F s) 0
    ⊢ HasLineDerivWithinAt 𝕜 f (L v) s x v
  -/
  simp only [one_smul, zero_add] at A
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F✝ : Type u_2
    inst✝³ : NormedAddCommGroup F✝
    inst✝² : NormedSpace 𝕜 F✝
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F✝
    s : Set E
    x : E
    L : ContinuousLinearMap (RingHom.id 𝕜) E F✝
    v : E
    F : 𝕜 → E := fun t => HAdd.hAdd x (HSMul.hSMul t v)
    hf : HasFDerivWithinAt f L s (F 0)
    A : HasDerivWithinAt F v (Set.preimage F s) 0
    ⊢ HasLineDerivWithinAt 𝕜 f (L v) s x v
  -/
  exact hf.comp_hasDerivWithinAt (x := (0 : 𝕜)) A (mapsTo_preimage F s)
  /-
    🎉 no goals
  -/


lemma HasFDerivAt.hasLineDerivAt (hf : HasFDerivAt f L x) (v : E) :
    HasLineDerivAt 𝕜 f (L v) x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F
    x : E
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : HasFDerivAt f L x
    v : E
    ⊢ HasLineDerivAt 𝕜 f (L v) x v
  -/
  rw [← hasLineDerivWithinAt_univ]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F
    x : E
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : HasFDerivAt f L x
    v : E
    ⊢ HasLineDerivWithinAt 𝕜 f (L v) Set.univ x v
  -/
  exact hf.hasFDerivWithinAt.hasLineDerivWithinAt v
  /-
    🎉 no goals
  -/


lemma DifferentiableAt.lineDeriv_eq_fderiv (hf : DifferentiableAt 𝕜 f x) :
    lineDeriv 𝕜 f x v = fderiv 𝕜 f x v :=
  (hf.hasFDerivAt.hasLineDerivAt v).lineDeriv


theorem LineDifferentiableWithinAt.mono_of_mem_nhdsWithin (h : LineDifferentiableWithinAt 𝕜 f s x v)
    (hst : s ∈ 𝓝[t] x) : LineDifferentiableWithinAt 𝕜 f t x v :=
  (h.hasLineDerivWithinAt.mono_of_mem_nhdsWithin hst).lineDifferentiableWithinAt


@[deprecated (since := "2024-10-31")]
alias LineDifferentiableWithinAt.mono_of_mem := LineDifferentiableWithinAt.mono_of_mem_nhdsWithin


theorem lineDerivWithin_of_mem_nhds (h : s ∈ 𝓝 x) :
    lineDerivWithin 𝕜 f s x v = lineDeriv 𝕜 f x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F
    s : Set E
    x v : E
    h : Membership.mem (nhds x) s
    ⊢ Eq (lineDerivWithin 𝕜 f s x v) (lineDeriv 𝕜 f x v)
  -/
  apply derivWithin_of_mem_nhds
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F
    s : Set E
    x v : E
    h : Membership.mem (nhds x) s
    ⊢ Membership.mem (nhds 0) (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v …
  -/
  apply (Continuous.continuousAt _).preimage_mem_nhds (by simpa using h)
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F
    s : Set E
    x v : E
    h : Membership.mem (nhds x) s
    ⊢ Continuous fun t => HAdd.hAdd x (HSMul.hSMul t v)
  -/
  fun_prop
  /-
    🎉 no goals
  -/


theorem lineDerivWithin_of_isOpen (hs : IsOpen s) (hx : x ∈ s) :
    lineDerivWithin 𝕜 f s x v = lineDeriv 𝕜 f x v :=
  lineDerivWithin_of_mem_nhds (hs.mem_nhds hx)


theorem hasLineDerivWithinAt_congr_set (h : s =ᶠ[𝓝 x] t) :
    HasLineDerivWithinAt 𝕜 f f' s x v ↔ HasLineDerivWithinAt 𝕜 f f' t x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F
    f' : F
    s t : Set E
    x v : E
    h : (nhds x).EventuallyEq s t
    ⊢ Iff (HasLineDerivWithinAt 𝕜 f f' s x v) (HasLineDerivWithinAt 𝕜 f f' t x v)
  -/
  apply hasDerivWithinAt_congr_set
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F
    f' : F
    s t : Set E
    x v : E
    h : (nhds x).EventuallyEq s t
    ⊢ (nhds 0).EventuallyEq (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) …
  -/
  let F := fun (t : 𝕜) ↦ x + t • v
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F✝ : Type u_2
    inst✝³ : NormedAddCommGroup F✝
    inst✝² : NormedSpace 𝕜 F✝
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F✝
    f' : F✝
    s t : Set E
    x v : E
    h : (nhds x).EventuallyEq s t
    F : 𝕜 → E := fun t => HAdd.hAdd x (HSMul.hSMul t v)
    ⊢ (nhds 0).EventuallyEq (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) …
  -/
  have B : ContinuousAt F 0 := by apply Continuous.continuousAt; fun_prop
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F✝ : Type u_2
    inst✝³ : NormedAddCommGroup F✝
    inst✝² : NormedSpace 𝕜 F✝
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F✝
    f' : F✝
    s t : Set E
    x v : E
    h : (nhds x).EventuallyEq s t
    F : 𝕜 → E := fun t => HAdd.hAdd x (HSMul.hSMul t v)
    B : ContinuousAt F 0
    ⊢ (nhds 0).EventuallyEq (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) …
  -/
  have : s =ᶠ[𝓝 (F 0)] t := by convert h; simp [F]
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F✝ : Type u_2
    inst✝³ : NormedAddCommGroup F✝
    inst✝² : NormedSpace 𝕜 F✝
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F✝
    f' : F✝
    s t : Set E
    x v : E
    h : (nhds x).EventuallyEq s t
    F : 𝕜 → E := fun t => HAdd.hAdd x (HSMul.hSMul t v)
    B : ContinuousAt F 0
    this : (nhds (F 0)).EventuallyEq s t
    ⊢ (nhds 0).EventuallyEq (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) …
  -/
  exact B.preimage_mem_nhds this
  /-
    🎉 no goals
  -/


theorem lineDifferentiableWithinAt_congr_set (h : s =ᶠ[𝓝 x] t) :
    LineDifferentiableWithinAt 𝕜 f s x v ↔ LineDifferentiableWithinAt 𝕜 f t x v :=
  ⟨fun h' ↦ ((hasLineDerivWithinAt_congr_set h).1
    h'.hasLineDerivWithinAt).lineDifferentiableWithinAt,
  fun h' ↦ ((hasLineDerivWithinAt_congr_set h.symm).1
    h'.hasLineDerivWithinAt).lineDifferentiableWithinAt⟩


theorem lineDerivWithin_congr_set (h : s =ᶠ[𝓝 x] t) :
    lineDerivWithin 𝕜 f s x v = lineDerivWithin 𝕜 f t x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F
    s t : Set E
    x v : E
    h : (nhds x).EventuallyEq s t
    ⊢ Eq (lineDerivWithin 𝕜 f s x v) (lineDerivWithin 𝕜 f t x v)
  -/
  apply derivWithin_congr_set
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F
    s t : Set E
    x v : E
    h : (nhds x).EventuallyEq s t
    ⊢ (nhds 0).EventuallyEq (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) …
  -/
  let F := fun (t : 𝕜) ↦ x + t • v
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F✝ : Type u_2
    inst✝³ : NormedAddCommGroup F✝
    inst✝² : NormedSpace 𝕜 F✝
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F✝
    s t : Set E
    x v : E
    h : (nhds x).EventuallyEq s t
    F : 𝕜 → E := fun t => HAdd.hAdd x (HSMul.hSMul t v)
    ⊢ (nhds 0).EventuallyEq (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) …
  -/
  have B : ContinuousAt F 0 := by apply Continuous.continuousAt; fun_prop
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F✝ : Type u_2
    inst✝³ : NormedAddCommGroup F✝
    inst✝² : NormedSpace 𝕜 F✝
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F✝
    s t : Set E
    x v : E
    h : (nhds x).EventuallyEq s t
    F : 𝕜 → E := fun t => HAdd.hAdd x (HSMul.hSMul t v)
    B : ContinuousAt F 0
    ⊢ (nhds 0).EventuallyEq (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) …
  -/
  have : s =ᶠ[𝓝 (F 0)] t := by convert h; simp [F]
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F✝ : Type u_2
    inst✝³ : NormedAddCommGroup F✝
    inst✝² : NormedSpace 𝕜 F✝
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → F✝
    s t : Set E
    x v : E
    h : (nhds x).EventuallyEq s t
    F : 𝕜 → E := fun t => HAdd.hAdd x (HSMul.hSMul t v)
    B : ContinuousAt F 0
    this : (nhds (F 0)).EventuallyEq s t
    ⊢ (nhds 0).EventuallyEq (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) …
  -/
  exact B.preimage_mem_nhds this
  /-
    🎉 no goals
  -/


theorem Filter.EventuallyEq.hasLineDerivAt_iff (h : f₀ =ᶠ[𝓝 x] f₁) :
    HasLineDerivAt 𝕜 f₀ f' x v ↔ HasLineDerivAt 𝕜 f₁ f' x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f₀ f₁ : E → F
    f' : F
    x v : E
    h : (nhds x).EventuallyEq f₀ f₁
    ⊢ Iff (HasLineDerivAt 𝕜 f₀ f' x v) (HasLineDerivAt 𝕜 f₁ f' x v)
  -/
  apply hasDerivAt_iff
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f₀ f₁ : E → F
    f' : F
    x v : E
    h : (nhds x).EventuallyEq f₀ f₁
    ⊢ (nhds 0).EventuallyEq (fun t => f₀ (HAdd.hAdd x (HSMul.hSMul t v))) fun t => …
  -/
  let F := fun (t : 𝕜) ↦ x + t • v
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F✝ : Type u_2
    inst✝³ : NormedAddCommGroup F✝
    inst✝² : NormedSpace 𝕜 F✝
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f₀ f₁ : E → F✝
    f' : F✝
    x v : E
    h : (nhds x).EventuallyEq f₀ f₁
    F : 𝕜 → E := fun t => HAdd.hAdd x (HSMul.hSMul t v)
    ⊢ (nhds 0).EventuallyEq (fun t => f₀ (HAdd.hAdd x (HSMul.hSMul t v))) fun t => …
  -/
  have B : ContinuousAt F 0 := by apply Continuous.continuousAt; fun_prop
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F✝ : Type u_2
    inst✝³ : NormedAddCommGroup F✝
    inst✝² : NormedSpace 𝕜 F✝
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f₀ f₁ : E → F✝
    f' : F✝
    x v : E
    h : (nhds x).EventuallyEq f₀ f₁
    F : 𝕜 → E := fun t => HAdd.hAdd x (HSMul.hSMul t v)
    B : ContinuousAt F 0
    ⊢ (nhds 0).EventuallyEq (fun t => f₀ (HAdd.hAdd x (HSMul.hSMul t v))) fun t => …
  -/
  have : f₀ =ᶠ[𝓝 (F 0)] f₁ := by convert h; simp [F]
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F✝ : Type u_2
    inst✝³ : NormedAddCommGroup F✝
    inst✝² : NormedSpace 𝕜 F✝
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f₀ f₁ : E → F✝
    f' : F✝
    x v : E
    h : (nhds x).EventuallyEq f₀ f₁
    F : 𝕜 → E := fun t => HAdd.hAdd x (HSMul.hSMul t v)
    B : ContinuousAt F 0
    this : (nhds (F 0)).EventuallyEq f₀ f₁
    ⊢ (nhds 0).EventuallyEq (fun t => f₀ (HAdd.hAdd x (HSMul.hSMul t v))) fun t => …
  -/
  exact B.preimage_mem_nhds this
  /-
    🎉 no goals
  -/


theorem Filter.EventuallyEq.lineDifferentiableAt_iff (h : f₀ =ᶠ[𝓝 x] f₁) :
    LineDifferentiableAt 𝕜 f₀ x v ↔ LineDifferentiableAt 𝕜 f₁ x v :=
  ⟨fun h' ↦ (h.hasLineDerivAt_iff.1 h'.hasLineDerivAt).lineDifferentiableAt,
  fun h' ↦ (h.hasLineDerivAt_iff.2 h'.hasLineDerivAt).lineDifferentiableAt⟩


theorem Filter.EventuallyEq.hasLineDerivWithinAt_iff (h : f₀ =ᶠ[𝓝[s] x] f₁) (hx : f₀ x = f₁ x) :
    HasLineDerivWithinAt 𝕜 f₀ f' s x v ↔ HasLineDerivWithinAt 𝕜 f₁ f' s x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f₀ f₁ : E → F
    f' : F
    s : Set E
    x v : E
    h : (nhdsWithin x s).EventuallyEq f₀ f₁
    hx : Eq (f₀ x) (f₁ x)
    ⊢ Iff (HasLineDerivWithinAt 𝕜 f₀ f' s x v) (HasLineDerivWithinAt 𝕜 f₁ f' s x v)
  -/
  apply hasDerivWithinAt_iff
    /-
      case h₁
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      F : Type u_2
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f₀ f₁ : E → F
      f' : F
      s : Set E
      x v : E
      h : (nhdsWithin x s).EventuallyEq f₀ f₁
      hx : Eq (f₀ x) (f₁ x)
      ⊢ (nhdsWithin 0 (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) s)).Eve …
    -/
  · have A : Continuous (fun (t : 𝕜) ↦ x + t • v) := by fun_prop
    /-
      case h₁
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      F : Type u_2
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f₀ f₁ : E → F
      f' : F
      s : Set E
      x v : E
      h : (nhdsWithin x s).EventuallyEq f₀ f₁
      hx : Eq (f₀ x) (f₁ x)
      A : Continuous fun t => HAdd.hAdd x (HSMul.hSMul t v)
      ⊢ (nhdsWithin 0 (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) s)).Eve …
    -/
    exact A.continuousWithinAt.preimage_mem_nhdsWithin'' h (by simp)
    /-
      🎉 no goals
    -/
    /-
      case hx
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      F : Type u_2
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f₀ f₁ : E → F
      f' : F
      s : Set E
      x v : E
      h : (nhdsWithin x s).EventuallyEq f₀ f₁
      hx : Eq (f₀ x) (f₁ x)
      ⊢ Eq (f₀ (HAdd.hAdd x (HSMul.hSMul 0 v))) (f₁ (HAdd.hAdd x (HSMul.hSMul 0 v)))
    -/
  · simpa using hx
    /-
      🎉 no goals
    -/


theorem Filter.EventuallyEq.hasLineDerivWithinAt_iff_of_mem (h : f₀ =ᶠ[𝓝[s] x] f₁) (hx : x ∈ s) :
    HasLineDerivWithinAt 𝕜 f₀ f' s x v ↔ HasLineDerivWithinAt 𝕜 f₁ f' s x v :=
  h.hasLineDerivWithinAt_iff (h.eq_of_nhdsWithin hx)


theorem Filter.EventuallyEq.lineDifferentiableWithinAt_iff
    (h : f₀ =ᶠ[𝓝[s] x] f₁) (hx : f₀ x = f₁ x) :
    LineDifferentiableWithinAt 𝕜 f₀ s x v ↔ LineDifferentiableWithinAt 𝕜 f₁ s x v :=
  ⟨fun h' ↦ ((h.hasLineDerivWithinAt_iff hx).1 h'.hasLineDerivWithinAt).lineDifferentiableWithinAt,
  fun h' ↦ ((h.hasLineDerivWithinAt_iff hx).2 h'.hasLineDerivWithinAt).lineDifferentiableWithinAt⟩


theorem Filter.EventuallyEq.lineDifferentiableWithinAt_iff_of_mem
    (h : f₀ =ᶠ[𝓝[s] x] f₁) (hx : x ∈ s) :
    LineDifferentiableWithinAt 𝕜 f₀ s x v ↔ LineDifferentiableWithinAt 𝕜 f₁ s x v :=
  h.lineDifferentiableWithinAt_iff (h.eq_of_nhdsWithin hx)


lemma HasLineDerivWithinAt.congr_of_eventuallyEq (hf : HasLineDerivWithinAt 𝕜 f f' s x v)
    (h'f : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) : HasLineDerivWithinAt 𝕜 f₁ f' s x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f f₁ : E → F
    f' : F
    s : Set E
    x v : E
    hf : HasLineDerivWithinAt 𝕜 f f' s x v
    h'f : (nhdsWithin x s).EventuallyEq f₁ f
    hx : Eq (f₁ x) (f x)
    ⊢ HasLineDerivWithinAt 𝕜 f₁ f' s x v
  -/
  apply HasDerivWithinAt.congr_of_eventuallyEq hf _ (by simp [hx])
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f f₁ : E → F
    f' : F
    s : Set E
    x v : E
    hf : HasLineDerivWithinAt 𝕜 f f' s x v
    h'f : (nhdsWithin x s).EventuallyEq f₁ f
    hx : Eq (f₁ x) (f x)
    ⊢ (nhdsWithin 0 (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) s)).Eve …
  -/
  have A : Continuous (fun (t : 𝕜) ↦ x + t • v) := by fun_prop
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f f₁ : E → F
    f' : F
    s : Set E
    x v : E
    hf : HasLineDerivWithinAt 𝕜 f f' s x v
    h'f : (nhdsWithin x s).EventuallyEq f₁ f
    hx : Eq (f₁ x) (f x)
    A : Continuous fun t => HAdd.hAdd x (HSMul.hSMul t v)
    ⊢ (nhdsWithin 0 (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) s)).Eve …
  -/
  exact A.continuousWithinAt.preimage_mem_nhdsWithin'' h'f (by simp)
  /-
    🎉 no goals
  -/


theorem HasLineDerivAt.congr_of_eventuallyEq (h : HasLineDerivAt 𝕜 f f' x v) (h₁ : f₁ =ᶠ[𝓝 x] f) :
    HasLineDerivAt 𝕜 f₁ f' x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f f₁ : E → F
    f' : F
    x v : E
    h : HasLineDerivAt 𝕜 f f' x v
    h₁ : (nhds x).EventuallyEq f₁ f
    ⊢ HasLineDerivAt 𝕜 f₁ f' x v
  -/
  apply HasDerivAt.congr_of_eventuallyEq h
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f f₁ : E → F
    f' : F
    x v : E
    h : HasLineDerivAt 𝕜 f f' x v
    h₁ : (nhds x).EventuallyEq f₁ f
    ⊢ (nhds 0).EventuallyEq (fun t => f₁ (HAdd.hAdd x (HSMul.hSMul t v))) fun t => …
  -/
  let F := fun (t : 𝕜) ↦ x + t • v
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F✝ : Type u_2
    inst✝³ : NormedAddCommGroup F✝
    inst✝² : NormedSpace 𝕜 F✝
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f f₁ : E → F✝
    f' : F✝
    x v : E
    h : HasLineDerivAt 𝕜 f f' x v
    h₁ : (nhds x).EventuallyEq f₁ f
    F : 𝕜 → E := fun t => HAdd.hAdd x (HSMul.hSMul t v)
    ⊢ (nhds 0).EventuallyEq (fun t => f₁ (HAdd.hAdd x (HSMul.hSMul t v))) fun t => …
  -/
  rw [show x = F 0 by simp [F]] at h₁
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F✝ : Type u_2
    inst✝³ : NormedAddCommGroup F✝
    inst✝² : NormedSpace 𝕜 F✝
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f f₁ : E → F✝
    f' : F✝
    x v : E
    h : HasLineDerivAt 𝕜 f f' x v
    F : 𝕜 → E := fun t => HAdd.hAdd x (HSMul.hSMul t v)
    h₁ : (nhds (F 0)).EventuallyEq f₁ f
    ⊢ (nhds 0).EventuallyEq (fun t => f₁ (HAdd.hAdd x (HSMul.hSMul t v))) fun t => …
  -/
  exact (Continuous.continuousAt (by fun_prop)).preimage_mem_nhds h₁
  /-
    🎉 no goals
  -/


theorem LineDifferentiableWithinAt.congr_of_eventuallyEq (h : LineDifferentiableWithinAt 𝕜 f s x v)
    (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) : LineDifferentiableWithinAt 𝕜 f₁ s x v :=
  (h.hasLineDerivWithinAt.congr_of_eventuallyEq h₁ hx).differentiableWithinAt


theorem LineDifferentiableAt.congr_of_eventuallyEq
    (h : LineDifferentiableAt 𝕜 f x v) (hL : f₁ =ᶠ[𝓝 x] f) :
    LineDifferentiableAt 𝕜 f₁ x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f f₁ : E → F
    x v : E
    h : LineDifferentiableAt 𝕜 f x v
    hL : (nhds x).EventuallyEq f₁ f
    ⊢ LineDifferentiableAt 𝕜 f₁ x v
  -/
  apply DifferentiableAt.congr_of_eventuallyEq h
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f f₁ : E → F
    x v : E
    h : LineDifferentiableAt 𝕜 f x v
    hL : (nhds x).EventuallyEq f₁ f
    ⊢ (nhds 0).EventuallyEq (fun t => f₁ (HAdd.hAdd x (HSMul.hSMul t v))) fun t => …
  -/
  let F := fun (t : 𝕜) ↦ x + t • v
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F✝ : Type u_2
    inst✝³ : NormedAddCommGroup F✝
    inst✝² : NormedSpace 𝕜 F✝
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f f₁ : E → F✝
    x v : E
    h : LineDifferentiableAt 𝕜 f x v
    hL : (nhds x).EventuallyEq f₁ f
    F : 𝕜 → E := fun t => HAdd.hAdd x (HSMul.hSMul t v)
    ⊢ (nhds 0).EventuallyEq (fun t => f₁ (HAdd.hAdd x (HSMul.hSMul t v))) fun t => …
  -/
  rw [show x = F 0 by simp [F]] at hL
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F✝ : Type u_2
    inst✝³ : NormedAddCommGroup F✝
    inst✝² : NormedSpace 𝕜 F✝
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f f₁ : E → F✝
    x v : E
    h : LineDifferentiableAt 𝕜 f x v
    F : 𝕜 → E := fun t => HAdd.hAdd x (HSMul.hSMul t v)
    hL : (nhds (F 0)).EventuallyEq f₁ f
    ⊢ (nhds 0).EventuallyEq (fun t => f₁ (HAdd.hAdd x (HSMul.hSMul t v))) fun t => …
  -/
  exact (Continuous.continuousAt (by fun_prop)).preimage_mem_nhds hL
  /-
    🎉 no goals
  -/


theorem Filter.EventuallyEq.lineDerivWithin_eq (hs : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) :
    lineDerivWithin 𝕜 f₁ s x v = lineDerivWithin 𝕜 f s x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f f₁ : E → F
    s : Set E
    x v : E
    hs : (nhdsWithin x s).EventuallyEq f₁ f
    hx : Eq (f₁ x) (f x)
    ⊢ Eq (lineDerivWithin 𝕜 f₁ s x v) (lineDerivWithin 𝕜 f s x v)
  -/
  apply derivWithin_eq ?_ (by simpa using hx)
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f f₁ : E → F
    s : Set E
    x v : E
    hs : (nhdsWithin x s).EventuallyEq f₁ f
    hx : Eq (f₁ x) (f x)
    ⊢ (nhdsWithin 0 (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) s)).Eve …
  -/
  have A : Continuous (fun (t : 𝕜) ↦ x + t • v) := by fun_prop
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f f₁ : E → F
    s : Set E
    x v : E
    hs : (nhdsWithin x s).EventuallyEq f₁ f
    hx : Eq (f₁ x) (f x)
    A : Continuous fun t => HAdd.hAdd x (HSMul.hSMul t v)
    ⊢ (nhdsWithin 0 (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) s)).Eve …
  -/
  exact A.continuousWithinAt.preimage_mem_nhdsWithin'' hs (by simp)
  /-
    🎉 no goals
  -/


theorem Filter.EventuallyEq.lineDerivWithin_eq_nhds (h : f₁ =ᶠ[𝓝 x] f) :
    lineDerivWithin 𝕜 f₁ s x v = lineDerivWithin 𝕜 f s x v :=
  (h.filter_mono nhdsWithin_le_nhds).lineDerivWithin_eq h.self_of_nhds


theorem Filter.EventuallyEq.lineDeriv_eq (h : f₁ =ᶠ[𝓝 x] f) :
    lineDeriv 𝕜 f₁ x v = lineDeriv 𝕜 f x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f f₁ : E → F
    x v : E
    h : (nhds x).EventuallyEq f₁ f
    ⊢ Eq (lineDeriv 𝕜 f₁ x v) (lineDeriv 𝕜 f x v)
  -/
  rw [← lineDerivWithin_univ, ← lineDerivWithin_univ, h.lineDerivWithin_eq_nhds]
  /-
    🎉 no goals
  -/


/-- Converse to the mean value inequality: if `f` is line differentiable at `x₀` and `C`-lipschitz
on a neighborhood of `x₀` then its line derivative at `x₀` in the direction `v` has norm
bounded by `C * ‖v‖`. This version only assumes that `‖f x - f x₀‖ ≤ C * ‖x - x₀‖` in a
neighborhood of `x`. -/
theorem HasLineDerivAt.le_of_lip' {f : E → F} {f' : F} {x₀ : E} (hf : HasLineDerivAt 𝕜 f f' x₀ v)
    {C : ℝ} (hC₀ : 0 ≤ C) (hlip : ∀ᶠ x in 𝓝 x₀, ‖f x - f x₀‖ ≤ C * ‖x - x₀‖) :
    ‖f'‖ ≤ C * ‖v‖ := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    v : E
    f : E → F
    f' : F
    x₀ : E
    hf : HasLineDerivAt 𝕜 f f' x₀ v
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    ⊢ LE.le (Norm.norm f') (HMul.hMul C (Norm.norm v))
  -/
  apply HasDerivAt.le_of_lip' hf (by positivity)
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    v : E
    f : E → F
    f' : F
    x₀ : E
    hf : HasLineDerivAt 𝕜 f f' x₀ v
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f (HAdd.hAdd x₀ (HS …
  -/
  have A : Continuous (fun (t : 𝕜) ↦ x₀ + t • v) := by fun_prop
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    v : E
    f : E → F
    f' : F
    x₀ : E
    hf : HasLineDerivAt 𝕜 f f' x₀ v
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    A : Continuous fun t => HAdd.hAdd x₀ (HSMul.hSMul t v)
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f (HAdd.hAdd x₀ (HS …
  -/
  have : ∀ᶠ x in 𝓝 (x₀ + (0 : 𝕜) • v), ‖f x - f x₀‖ ≤ C * ‖x - x₀‖ := by simpa using hlip
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    v : E
    f : E → F
    f' : F
    x₀ : E
    hf : HasLineDerivAt 𝕜 f f' x₀ v
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    A : Continuous fun t => HAdd.hAdd x₀ (HSMul.hSMul t v)
    this : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f (HAdd.hAdd x₀ (HS …
  -/
  filter_upwards [(A.continuousAt (x := 0)).preimage_mem_nhds this] with t ht
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    v : E
    f : E → F
    f' : F
    x₀ : E
    hf : HasLineDerivAt 𝕜 f f' x₀ v
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    A : Continuous fun t => HAdd.hAdd x₀ (HSMul.hSMul t v)
    this : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    t : 𝕜
    ht : Membership.mem (Set.preimage (fun t => HAdd.hAdd x₀ (HSMul.hSMul t v)) (s …
    ⊢ LE.le (Norm.norm (HSub.hSub (f (HAdd.hAdd x₀ (HSMul.hSMul t v))) (f (HAdd.hA …
  -/
  simp only [preimage_setOf_eq, add_sub_cancel_left, norm_smul, mem_setOf_eq, mul_comm (‖t‖)] at ht
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    v : E
    f : E → F
    f' : F
    x₀ : E
    hf : HasLineDerivAt 𝕜 f f' x₀ v
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    A : Continuous fun t => HAdd.hAdd x₀ (HSMul.hSMul t v)
    this : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    t : 𝕜
    ht : LE.le (Norm.norm (HSub.hSub (f (HAdd.hAdd x₀ (HSMul.hSMul t v))) (f x₀))) …
    ⊢ LE.le (Norm.norm (HSub.hSub (f (HAdd.hAdd x₀ (HSMul.hSMul t v))) (f (HAdd.hA …
  -/
  simpa [mul_assoc] using ht
  /-
    🎉 no goals
  -/


/-- Converse to the mean value inequality: if `f` is line differentiable at `x₀` and `C`-lipschitz
on a neighborhood of `x₀` then its line derivative at `x₀` in the direction `v` has norm
bounded by `C * ‖v‖`. This version only assumes that `‖f x - f x₀‖ ≤ C * ‖x - x₀‖` in a
neighborhood of `x`. -/
theorem HasLineDerivAt.le_of_lipschitzOn
    {f : E → F} {f' : F} {x₀ : E} (hf : HasLineDerivAt 𝕜 f f' x₀ v)
    {s : Set E} (hs : s ∈ 𝓝 x₀) {C : ℝ≥0} (hlip : LipschitzOnWith C f s) :
    ‖f'‖ ≤ C * ‖v‖ := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    v : E
    f : E → F
    f' : F
    x₀ : E
    hf : HasLineDerivAt 𝕜 f f' x₀ v
    s : Set E
    hs : Membership.mem (nhds x₀) s
    C : NNReal
    hlip : LipschitzOnWith C f s
    ⊢ LE.le (Norm.norm f') (HMul.hMul (↑C) (Norm.norm v))
  -/
  refine hf.le_of_lip' C.coe_nonneg ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    v : E
    f : E → F
    f' : F
    x₀ : E
    hf : HasLineDerivAt 𝕜 f f' x₀ v
    s : Set E
    hs : Membership.mem (nhds x₀) s
    C : NNReal
    hlip : LipschitzOnWith C f s
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀))) (HMul …
  -/
  filter_upwards [hs] with x hx using hlip.norm_sub_le hx (mem_of_mem_nhds hs)
  /-
    🎉 no goals
  -/


/-- Converse to the mean value inequality: if `f` is line differentiable at `x₀` and `C`-lipschitz
then its line derivative at `x₀` in the direction `v` has norm bounded by `C * ‖v‖`. -/
theorem HasLineDerivAt.le_of_lipschitz
    {f : E → F} {f' : F} {x₀ : E} (hf : HasLineDerivAt 𝕜 f f' x₀ v)
    {C : ℝ≥0} (hlip : LipschitzWith C f) : ‖f'‖ ≤ C * ‖v‖ :=
  hf.le_of_lipschitzOn univ_mem (lipschitzOnWith_univ.2 hlip)


/-- Converse to the mean value inequality: if `f` is `C`-lipschitz
on a neighborhood of `x₀` then its line derivative at `x₀` in the direction `v` has norm
bounded by `C * ‖v‖`. This version only assumes that `‖f x - f x₀‖ ≤ C * ‖x - x₀‖` in a
neighborhood of `x`.
Version using `lineDeriv`. -/
theorem norm_lineDeriv_le_of_lip' {f : E → F} {x₀ : E}
    {C : ℝ} (hC₀ : 0 ≤ C) (hlip : ∀ᶠ x in 𝓝 x₀, ‖f x - f x₀‖ ≤ C * ‖x - x₀‖) :
    ‖lineDeriv 𝕜 f x₀ v‖ ≤ C * ‖v‖ := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    v : E
    f : E → F
    x₀ : E
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    ⊢ LE.le (Norm.norm (lineDeriv 𝕜 f x₀ v)) (HMul.hMul C (Norm.norm v))
  -/
  apply norm_deriv_le_of_lip' (by positivity)
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    v : E
    f : E → F
    x₀ : E
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f (HAdd.hAdd x₀ (HS …
  -/
  have A : Continuous (fun (t : 𝕜) ↦ x₀ + t • v) := by fun_prop
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    v : E
    f : E → F
    x₀ : E
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    A : Continuous fun t => HAdd.hAdd x₀ (HSMul.hSMul t v)
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f (HAdd.hAdd x₀ (HS …
  -/
  have : ∀ᶠ x in 𝓝 (x₀ + (0 : 𝕜) • v), ‖f x - f x₀‖ ≤ C * ‖x - x₀‖ := by simpa using hlip
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    v : E
    f : E → F
    x₀ : E
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    A : Continuous fun t => HAdd.hAdd x₀ (HSMul.hSMul t v)
    this : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f (HAdd.hAdd x₀ (HS …
  -/
  filter_upwards [(A.continuousAt (x := 0)).preimage_mem_nhds this] with t ht
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    v : E
    f : E → F
    x₀ : E
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    A : Continuous fun t => HAdd.hAdd x₀ (HSMul.hSMul t v)
    this : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    t : 𝕜
    ht : Membership.mem (Set.preimage (fun t => HAdd.hAdd x₀ (HSMul.hSMul t v)) (s …
    ⊢ LE.le (Norm.norm (HSub.hSub (f (HAdd.hAdd x₀ (HSMul.hSMul t v))) (f (HAdd.hA …
  -/
  simp only [preimage_setOf_eq, add_sub_cancel_left, norm_smul, mem_setOf_eq, mul_comm (‖t‖)] at ht
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    v : E
    f : E → F
    x₀ : E
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    A : Continuous fun t => HAdd.hAdd x₀ (HSMul.hSMul t v)
    this : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    t : 𝕜
    ht : LE.le (Norm.norm (HSub.hSub (f (HAdd.hAdd x₀ (HSMul.hSMul t v))) (f x₀))) …
    ⊢ LE.le (Norm.norm (HSub.hSub (f (HAdd.hAdd x₀ (HSMul.hSMul t v))) (f (HAdd.hA …
  -/
  simpa [mul_assoc] using ht
  /-
    🎉 no goals
  -/


/-- Converse to the mean value inequality: if `f` is `C`-lipschitz on a neighborhood of `x₀`
then its line derivative at `x₀` in the direction `v` has norm bounded by `C * ‖v‖`.
Version using `lineDeriv`. -/
theorem norm_lineDeriv_le_of_lipschitzOn {f : E → F} {x₀ : E} {s : Set E} (hs : s ∈ 𝓝 x₀)
    {C : ℝ≥0} (hlip : LipschitzOnWith C f s) : ‖lineDeriv 𝕜 f x₀ v‖ ≤ C * ‖v‖ := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    v : E
    f : E → F
    x₀ : E
    s : Set E
    hs : Membership.mem (nhds x₀) s
    C : NNReal
    hlip : LipschitzOnWith C f s
    ⊢ LE.le (Norm.norm (lineDeriv 𝕜 f x₀ v)) (HMul.hMul (↑C) (Norm.norm v))
  -/
  refine norm_lineDeriv_le_of_lip' 𝕜 C.coe_nonneg ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    v : E
    f : E → F
    x₀ : E
    s : Set E
    hs : Membership.mem (nhds x₀) s
    C : NNReal
    hlip : LipschitzOnWith C f s
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀))) (HMul …
  -/
  filter_upwards [hs] with x hx using hlip.norm_sub_le hx (mem_of_mem_nhds hs)
  /-
    🎉 no goals
  -/


/-- Converse to the mean value inequality: if `f` is `C`-lipschitz then
its line derivative at `x₀` in the direction `v` has norm bounded by `C * ‖v‖`.
Version using `lineDeriv`. -/
theorem norm_lineDeriv_le_of_lipschitz {f : E → F} {x₀ : E}
    {C : ℝ≥0} (hlip : LipschitzWith C f) : ‖lineDeriv 𝕜 f x₀ v‖ ≤ C * ‖v‖ :=
  norm_lineDeriv_le_of_lipschitzOn 𝕜 univ_mem (lipschitzOnWith_univ.2 hlip)


theorem hasLineDerivWithinAt_zero : HasLineDerivWithinAt 𝕜 f 0 s x 0 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    s : Set E
    x : E
    ⊢ HasLineDerivWithinAt 𝕜 f 0 s x 0
  -/
  simp [HasLineDerivWithinAt, hasDerivWithinAt_const]
  /-
    🎉 no goals
  -/


theorem hasLineDerivAt_zero : HasLineDerivAt 𝕜 f 0 x 0 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    x : E
    ⊢ HasLineDerivAt 𝕜 f 0 x 0
  -/
  simp [HasLineDerivAt, hasDerivAt_const]
  /-
    🎉 no goals
  -/


theorem lineDifferentiableWithinAt_zero : LineDifferentiableWithinAt 𝕜 f s x 0 :=
  hasLineDerivWithinAt_zero.lineDifferentiableWithinAt


theorem lineDifferentiableAt_zero : LineDifferentiableAt 𝕜 f x 0 :=
  hasLineDerivAt_zero.lineDifferentiableAt


theorem lineDeriv_zero : lineDeriv 𝕜 f x 0 = 0 :=
  hasLineDerivAt_zero.lineDeriv


theorem HasLineDerivAt.of_comp {v : E'} (hf : HasLineDerivAt 𝕜 (f ∘ L) f' x v) :
    HasLineDerivAt 𝕜 f f' (L x) (L v) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    E : Type u_3
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    E' : Type u_4
    inst✝¹ : AddCommGroup E'
    inst✝ : Module 𝕜 E'
    f : E → F
    f' : F
    x : E'
    L : LinearMap (RingHom.id 𝕜) E' E
    v : E'
    hf : HasLineDerivAt 𝕜 (Function.comp f ⇑L) f' x v
    ⊢ HasLineDerivAt 𝕜 f f' (L x) (L v)
  -/
  simpa [HasLineDerivAt] using hf
  /-
    🎉 no goals
  -/


theorem LineDifferentiableAt.of_comp {v : E'} (hf : LineDifferentiableAt 𝕜 (f ∘ L) x v) :
    LineDifferentiableAt 𝕜 f (L x) (L v) :=
  hf.hasLineDerivAt.of_comp.lineDifferentiableAt


theorem HasLineDerivWithinAt.smul (h : HasLineDerivWithinAt 𝕜 f f' s x v) (c : 𝕜) :
    HasLineDerivWithinAt 𝕜 f (c • f') s x (c • v) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    s : Set E
    x v : E
    f' : F
    h : HasLineDerivWithinAt 𝕜 f f' s x v
    c : 𝕜
    ⊢ HasLineDerivWithinAt 𝕜 f (HSMul.hSMul c f') s x (HSMul.hSMul c v)
  -/
  simp only [HasLineDerivWithinAt] at h ⊢
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    s : Set E
    x v : E
    f' : F
    h : HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t v))) f' (Set.prei …
    c : 𝕜
    ⊢ HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t (HSMul.hSMul c v))) …
  -/
  let g := fun (t : 𝕜) ↦ c • t
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    s : Set E
    x v : E
    f' : F
    h : HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t v))) f' (Set.prei …
    c : 𝕜
    g : 𝕜 → 𝕜 := fun t => HSMul.hSMul c t
    ⊢ HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t (HSMul.hSMul c v))) …
  -/
  let s' := (fun (t : 𝕜) ↦ x + t • v) ⁻¹' s
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    s : Set E
    x v : E
    f' : F
    h : HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t v))) f' (Set.prei …
    c : 𝕜
    g : 𝕜 → 𝕜 := fun t => HSMul.hSMul c t
    s' : Set 𝕜 := Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) s
    ⊢ HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t (HSMul.hSMul c v))) …
  -/
  have A : HasDerivAt g c 0 := by simpa using (hasDerivAt_id (0 : 𝕜)).const_smul c
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    s : Set E
    x v : E
    f' : F
    h : HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t v))) f' (Set.prei …
    c : 𝕜
    g : 𝕜 → 𝕜 := fun t => HSMul.hSMul c t
    s' : Set 𝕜 := Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) s
    A : HasDerivAt g c 0
    ⊢ HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t (HSMul.hSMul c v))) …
  -/
  have B : HasDerivWithinAt (fun t ↦ f (x + t • v)) f' s' (g 0) := by simpa [g] using h
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    s : Set E
    x v : E
    f' : F
    h : HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t v))) f' (Set.prei …
    c : 𝕜
    g : 𝕜 → 𝕜 := fun t => HSMul.hSMul c t
    s' : Set 𝕜 := Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) s
    A : HasDerivAt g c 0
    B : HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t v))) f' s' (g 0)
    ⊢ HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t (HSMul.hSMul c v))) …
  -/
  have Z := B.scomp (0 : 𝕜) A.hasDerivWithinAt (mapsTo_preimage g s')
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    s : Set E
    x v : E
    f' : F
    h : HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t v))) f' (Set.prei …
    c : 𝕜
    g : 𝕜 → 𝕜 := fun t => HSMul.hSMul c t
    s' : Set 𝕜 := Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) s
    A : HasDerivAt g c 0
    B : HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t v))) f' s' (g 0)
    Z : HasDerivWithinAt (Function.comp (fun t => f (HAdd.hAdd x (HSMul.hSMul t v) …
    ⊢ HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t (HSMul.hSMul c v))) …
  -/
  simp only [g, s', Function.comp_def, smul_eq_mul, mul_comm c, ← smul_smul] at Z
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    s : Set E
    x v : E
    f' : F
    h : HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t v))) f' (Set.prei …
    c : 𝕜
    g : 𝕜 → 𝕜 := fun t => HSMul.hSMul c t
    s' : Set 𝕜 := Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) s
    A : HasDerivAt g c 0
    B : HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t v))) f' s' (g 0)
    Z : HasDerivWithinAt (fun x_1 => f (HAdd.hAdd x (HSMul.hSMul x_1 (HSMul.hSMul  …
    ⊢ HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t (HSMul.hSMul c v))) …
  -/
  convert Z
  /-
    case h.e'_10
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    s : Set E
    x v : E
    f' : F
    h : HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t v))) f' (Set.prei …
    c : 𝕜
    g : 𝕜 → 𝕜 := fun t => HSMul.hSMul c t
    s' : Set 𝕜 := Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) s
    A : HasDerivAt g c 0
    B : HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t v))) f' s' (g 0)
    Z : HasDerivWithinAt (fun x_1 => f (HAdd.hAdd x (HSMul.hSMul x_1 (HSMul.hSMul  …
    ⊢ Eq (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t (HSMul.hSMul c v))) s) …
  -/
  ext t
  /-
    case h.e'_10.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    s : Set E
    x v : E
    f' : F
    h : HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t v))) f' (Set.prei …
    c : 𝕜
    g : 𝕜 → 𝕜 := fun t => HSMul.hSMul c t
    s' : Set 𝕜 := Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t v)) s
    A : HasDerivAt g c 0
    B : HasDerivWithinAt (fun t => f (HAdd.hAdd x (HSMul.hSMul t v))) f' s' (g 0)
    Z : HasDerivWithinAt (fun x_1 => f (HAdd.hAdd x (HSMul.hSMul x_1 (HSMul.hSMul  …
    t : 𝕜
    ⊢ Iff (Membership.mem (Set.preimage (fun t => HAdd.hAdd x (HSMul.hSMul t (HSMu …
  -/
  simp [← smul_smul]
  /-
    🎉 no goals
  -/


theorem hasLineDerivWithinAt_smul_iff {c : 𝕜} (hc : c ≠ 0) :
    HasLineDerivWithinAt 𝕜 f (c • f') s x (c • v) ↔ HasLineDerivWithinAt 𝕜 f f' s x v :=
              /-
                𝕜 : Type u_1
                inst✝⁴ : NontriviallyNormedField 𝕜
                F : Type u_2
                inst✝³ : NormedAddCommGroup F
                inst✝² : NormedSpace 𝕜 F
                E : Type u_3
                inst✝¹ : AddCommGroup E
                inst✝ : Module 𝕜 E
                f : E → F
                s : Set E
                x v : E
                f' : F
                c : 𝕜
                hc : Ne c 0
                h : HasLineDerivWithinAt 𝕜 f (HSMul.hSMul c f') s x (HSMul.hSMul c v)
                ⊢ HasLineDerivWithinAt 𝕜 f f' s x v
              -/
  ⟨fun h ↦ by simpa [smul_smul, inv_mul_cancel₀ hc] using h.smul (c ⁻¹), fun h ↦ h.smul c⟩
              /-
                🎉 no goals
              -/


theorem HasLineDerivAt.smul (h : HasLineDerivAt 𝕜 f f' x v) (c : 𝕜) :
    HasLineDerivAt 𝕜 f (c • f') x (c • v) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    x v : E
    f' : F
    h : HasLineDerivAt 𝕜 f f' x v
    c : 𝕜
    ⊢ HasLineDerivAt 𝕜 f (HSMul.hSMul c f') x (HSMul.hSMul c v)
  -/
  simp only [← hasLineDerivWithinAt_univ] at h ⊢
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    x v : E
    f' : F
    c : 𝕜
    h : HasLineDerivWithinAt 𝕜 f f' Set.univ x v
    ⊢ HasLineDerivWithinAt 𝕜 f (HSMul.hSMul c f') Set.univ x (HSMul.hSMul c v)
  -/
  exact HasLineDerivWithinAt.smul h c
  /-
    🎉 no goals
  -/


theorem hasLineDerivAt_smul_iff {c : 𝕜} (hc : c ≠ 0) :
    HasLineDerivAt 𝕜 f (c • f') x (c • v) ↔ HasLineDerivAt 𝕜 f f' x v :=
              /-
                𝕜 : Type u_1
                inst✝⁴ : NontriviallyNormedField 𝕜
                F : Type u_2
                inst✝³ : NormedAddCommGroup F
                inst✝² : NormedSpace 𝕜 F
                E : Type u_3
                inst✝¹ : AddCommGroup E
                inst✝ : Module 𝕜 E
                f : E → F
                x v : E
                f' : F
                c : 𝕜
                hc : Ne c 0
                h : HasLineDerivAt 𝕜 f (HSMul.hSMul c f') x (HSMul.hSMul c v)
                ⊢ HasLineDerivAt 𝕜 f f' x v
              -/
  ⟨fun h ↦ by simpa [smul_smul, inv_mul_cancel₀ hc] using h.smul (c ⁻¹), fun h ↦ h.smul c⟩
              /-
                🎉 no goals
              -/


theorem LineDifferentiableWithinAt.smul (h : LineDifferentiableWithinAt 𝕜 f s x v) (c : 𝕜) :
    LineDifferentiableWithinAt 𝕜 f s x (c • v) :=
  (h.hasLineDerivWithinAt.smul c).lineDifferentiableWithinAt


theorem lineDifferentiableWithinAt_smul_iff {c : 𝕜} (hc : c ≠ 0) :
    LineDifferentiableWithinAt 𝕜 f s x (c • v) ↔ LineDifferentiableWithinAt 𝕜 f s x v :=
              /-
                𝕜 : Type u_1
                inst✝⁴ : NontriviallyNormedField 𝕜
                F : Type u_2
                inst✝³ : NormedAddCommGroup F
                inst✝² : NormedSpace 𝕜 F
                E : Type u_3
                inst✝¹ : AddCommGroup E
                inst✝ : Module 𝕜 E
                f : E → F
                s : Set E
                x v : E
                c : 𝕜
                hc : Ne c 0
                h : LineDifferentiableWithinAt 𝕜 f s x (HSMul.hSMul c v)
                ⊢ LineDifferentiableWithinAt 𝕜 f s x v
              -/
  ⟨fun h ↦ by simpa [smul_smul, inv_mul_cancel₀ hc] using h.smul (c ⁻¹), fun h ↦ h.smul c⟩
              /-
                🎉 no goals
              -/


theorem LineDifferentiableAt.smul (h : LineDifferentiableAt 𝕜 f x v) (c : 𝕜) :
    LineDifferentiableAt 𝕜 f x (c • v) :=
  (h.hasLineDerivAt.smul c).lineDifferentiableAt


theorem lineDifferentiableAt_smul_iff {c : 𝕜} (hc : c ≠ 0) :
    LineDifferentiableAt 𝕜 f x (c • v) ↔ LineDifferentiableAt 𝕜 f x v :=
              /-
                𝕜 : Type u_1
                inst✝⁴ : NontriviallyNormedField 𝕜
                F : Type u_2
                inst✝³ : NormedAddCommGroup F
                inst✝² : NormedSpace 𝕜 F
                E : Type u_3
                inst✝¹ : AddCommGroup E
                inst✝ : Module 𝕜 E
                f : E → F
                x v : E
                c : 𝕜
                hc : Ne c 0
                h : LineDifferentiableAt 𝕜 f x (HSMul.hSMul c v)
                ⊢ LineDifferentiableAt 𝕜 f x v
              -/
  ⟨fun h ↦ by simpa [smul_smul, inv_mul_cancel₀ hc] using h.smul (c ⁻¹), fun h ↦ h.smul c⟩
              /-
                🎉 no goals
              -/


theorem lineDeriv_smul {c : 𝕜} : lineDeriv 𝕜 f x (c • v) = c • lineDeriv 𝕜 f x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    x v : E
    c : 𝕜
    ⊢ Eq (lineDeriv 𝕜 f x (HSMul.hSMul c v)) (HSMul.hSMul c (lineDeriv 𝕜 f x v))
  -/
  rcases eq_or_ne c 0 with rfl|hc
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      F : Type u_2
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      E : Type u_3
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      f : E → F
      x v : E
      ⊢ Eq (lineDeriv 𝕜 f x (HSMul.hSMul 0 v)) (HSMul.hSMul 0 (lineDeriv 𝕜 f x v))
    -/
  · simp [lineDeriv_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    x v : E
    c : 𝕜
    hc : Ne c 0
    ⊢ Eq (lineDeriv 𝕜 f x (HSMul.hSMul c v)) (HSMul.hSMul c (lineDeriv 𝕜 f x v))
  -/
  by_cases H : LineDifferentiableAt 𝕜 f x v
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      F : Type u_2
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      E : Type u_3
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      f : E → F
      x v : E
      c : 𝕜
      hc : Ne c 0
      H : LineDifferentiableAt 𝕜 f x v
      ⊢ Eq (lineDeriv 𝕜 f x (HSMul.hSMul c v)) (HSMul.hSMul c (lineDeriv 𝕜 f x v))
    -/
  · exact (H.hasLineDerivAt.smul c).lineDeriv
    /-
      🎉 no goals
    -/
  · have H' : ¬ (LineDifferentiableAt 𝕜 f x (c • v)) := by
      simpa [lineDifferentiableAt_smul_iff hc] using H
    /-
      case neg
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      F : Type u_2
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      E : Type u_3
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      f : E → F
      x v : E
      c : 𝕜
      hc : Ne c 0
      H : Not (LineDifferentiableAt 𝕜 f x v)
      H' : Not (LineDifferentiableAt 𝕜 f x (HSMul.hSMul c v))
      ⊢ Eq (lineDeriv 𝕜 f x (HSMul.hSMul c v)) (HSMul.hSMul c (lineDeriv 𝕜 f x v))
    -/
    simp [lineDeriv_zero_of_not_lineDifferentiableAt, H, H']
    /-
      🎉 no goals
    -/


theorem lineDeriv_neg : lineDeriv 𝕜 f x (-v) = - lineDeriv 𝕜 f x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E : Type u_3
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : E → F
    x v : E
    ⊢ Eq (lineDeriv 𝕜 f x (Neg.neg v)) (Neg.neg (lineDeriv 𝕜 f x v))
  -/
  rw [← neg_one_smul (R := 𝕜) v, lineDeriv_smul, neg_one_smul]
  /-
    🎉 no goals
  -/


