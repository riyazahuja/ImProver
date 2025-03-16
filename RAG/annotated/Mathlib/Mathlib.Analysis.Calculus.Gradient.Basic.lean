/-- A function `f` has the gradient `f'` as derivative along the filter `L` if
  `f x' = f x + ⟨f', x' - x⟩ + o (x' - x)` when `x'` converges along the filter `L`. -/
def HasGradientAtFilter (f : F → 𝕜) (f' x : F) (L : Filter F) :=
  HasFDerivAtFilter f (toDual 𝕜 F f') x L


/-- `f` has the gradient `f'` at the point `x` within the subset `s` if
  `f x' = f x + ⟨f', x' - x⟩ + o (x' - x)` where `x'` converges to `x` inside `s`. -/
def HasGradientWithinAt (f : F → 𝕜) (f' : F) (s : Set F) (x : F) :=
  HasGradientAtFilter f f' x (𝓝[s] x)


/-- `f` has the gradient `f'` at the point `x` if
  `f x' = f x + ⟨f', x' - x⟩ + o (x' - x)` where `x'` converges to `x`. -/
def HasGradientAt (f : F → 𝕜) (f' x : F) :=
  HasGradientAtFilter f f' x (𝓝 x)


/-- Gradient of `f` at the point `x` within the set `s`, if it exists.  Zero otherwise.

If the derivative exists (i.e., `∃ f', HasGradientWithinAt f f' s x`), then
`f x' = f x + ⟨f', x' - x⟩ + o (x' - x)` where `x'` converges to `x` inside `s`. -/
def gradientWithin (f : F → 𝕜) (s : Set F) (x : F) : F :=
  (toDual 𝕜 F).symm (fderivWithin 𝕜 f s x)


/-- Gradient of `f` at the point `x`, if it exists.  Zero otherwise.

If the derivative exists (i.e., `∃ f', HasGradientAt f f' x`), then
`f x' = f x + ⟨f', x' - x⟩ + o (x' - x)` where `x'` converges to `x`. -/
def gradient (f : F → 𝕜) (x : F) : F :=
  (toDual 𝕜 F).symm (fderiv 𝕜 f x)


@[inherit_doc]
scoped[Gradient] notation "∇" => gradient


local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


theorem hasGradientWithinAt_iff_hasFDerivWithinAt {s : Set F} :
    HasGradientWithinAt f f' s x ↔ HasFDerivWithinAt f (toDual 𝕜 F f') s x :=
  Iff.rfl


theorem hasFDerivWithinAt_iff_hasGradientWithinAt {frechet : F →L[𝕜] 𝕜} {s : Set F} :
    HasFDerivWithinAt f frechet s x ↔ HasGradientWithinAt f ((toDual 𝕜 F).symm frechet) s x := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : F → 𝕜
    x : F
    frechet : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜
    s : Set F
    ⊢ Iff (HasFDerivWithinAt f frechet s x) (HasGradientWithinAt f ((InnerProductS …
  -/
  rw [hasGradientWithinAt_iff_hasFDerivWithinAt, (toDual 𝕜 F).apply_symm_apply frechet]
  /-
    🎉 no goals
  -/


theorem hasGradientAt_iff_hasFDerivAt :
    HasGradientAt f f' x ↔ HasFDerivAt f (toDual 𝕜 F f') x :=
  Iff.rfl


theorem hasFDerivAt_iff_hasGradientAt {frechet : F →L[𝕜] 𝕜} :
    HasFDerivAt f frechet x ↔ HasGradientAt f ((toDual 𝕜 F).symm frechet) x := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : F → 𝕜
    x : F
    frechet : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜
    ⊢ Iff (HasFDerivAt f frechet x) (HasGradientAt f ((InnerProductSpace.toDual 𝕜  …
  -/
  rw [hasGradientAt_iff_hasFDerivAt, (toDual 𝕜 F).apply_symm_apply frechet]
  /-
    🎉 no goals
  -/


alias ⟨HasGradientWithinAt.hasFDerivWithinAt, _⟩ := hasGradientWithinAt_iff_hasFDerivWithinAt


alias ⟨HasFDerivWithinAt.hasGradientWithinAt, _⟩ := hasFDerivWithinAt_iff_hasGradientWithinAt


alias ⟨HasGradientAt.hasFDerivAt, _⟩ := hasGradientAt_iff_hasFDerivAt


alias ⟨HasFDerivAt.hasGradientAt, _⟩ := hasFDerivAt_iff_hasGradientAt


theorem gradient_eq_zero_of_not_differentiableAt (h : ¬DifferentiableAt 𝕜 f x) : ∇ f x = 0 := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : F → 𝕜
    x : F
    h : Not (DifferentiableAt 𝕜 f x)
    ⊢ Eq (gradient f x) 0
  -/
  rw [gradient, fderiv_zero_of_not_differentiableAt h, map_zero]
  /-
    🎉 no goals
  -/


theorem HasGradientAt.unique {gradf gradg : F}
    (hf : HasGradientAt f gradf x) (hg : HasGradientAt f gradg x) :
    gradf = gradg :=
  (toDual 𝕜 F).injective (hf.hasFDerivAt.unique hg.hasFDerivAt)


theorem DifferentiableAt.hasGradientAt (h : DifferentiableAt 𝕜 f x) :
    HasGradientAt f (∇ f x) x := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : F → 𝕜
    x : F
    h : DifferentiableAt 𝕜 f x
    ⊢ HasGradientAt f (gradient f x) x
  -/
  rw [hasGradientAt_iff_hasFDerivAt, gradient, (toDual 𝕜 F).apply_symm_apply (fderiv 𝕜 f x)]
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : F → 𝕜
    x : F
    h : DifferentiableAt 𝕜 f x
    ⊢ HasFDerivAt f (fderiv 𝕜 f x) x
  -/
  exact h.hasFDerivAt
  /-
    🎉 no goals
  -/


theorem HasGradientAt.differentiableAt (h : HasGradientAt f f' x) :
    DifferentiableAt 𝕜 f x :=
  h.hasFDerivAt.differentiableAt


theorem DifferentiableWithinAt.hasGradientWithinAt (h : DifferentiableWithinAt 𝕜 f s x) :
    HasGradientWithinAt f (gradientWithin f s x) s x := by
  rw [hasGradientWithinAt_iff_hasFDerivWithinAt, gradientWithin,
    (toDual 𝕜 F).apply_symm_apply (fderivWithin 𝕜 f s x)]
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : F → 𝕜
    x : F
    s : Set F
    h : DifferentiableWithinAt 𝕜 f s x
    ⊢ HasFDerivWithinAt f (fderivWithin 𝕜 f s x) s x
  -/
  exact h.hasFDerivWithinAt
  /-
    🎉 no goals
  -/


theorem HasGradientWithinAt.differentiableWithinAt (h : HasGradientWithinAt f f' s x) :
    DifferentiableWithinAt 𝕜 f s x :=
  h.hasFDerivWithinAt.differentiableWithinAt


@[simp]
theorem hasGradientWithinAt_univ : HasGradientWithinAt f f' univ x ↔ HasGradientAt f f' x := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : F → 𝕜
    f' x : F
    ⊢ Iff (HasGradientWithinAt f f' Set.univ x) (HasGradientAt f f' x)
  -/
  rw [hasGradientWithinAt_iff_hasFDerivWithinAt, hasGradientAt_iff_hasFDerivAt]
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : F → 𝕜
    f' x : F
    ⊢ Iff (HasFDerivWithinAt f ((InnerProductSpace.toDual 𝕜 F) f') Set.univ x) (Ha …
  -/
  exact hasFDerivWithinAt_univ
  /-
    🎉 no goals
  -/


theorem DifferentiableOn.hasGradientAt (h : DifferentiableOn 𝕜 f s) (hs : s ∈ 𝓝 x) :
    HasGradientAt f (∇ f x) x :=
  (h.hasFDerivAt hs).hasGradientAt


theorem HasGradientAt.gradient (h : HasGradientAt f f' x) : ∇ f x = f' :=
  h.differentiableAt.hasGradientAt.unique h


theorem gradient_eq {f' : F → F} (h : ∀ x, HasGradientAt f (f' x) x) : ∇ f = f' :=
  funext fun x => (h x).gradient


theorem HasGradientAtFilter.hasDerivAtFilter (h : HasGradientAtFilter g g' u L') :
    HasDerivAtFilter g (starRingEnd 𝕜 g') u L' := by
  have : ContinuousLinearMap.smulRight (1 : 𝕜 →L[𝕜] 𝕜) (starRingEnd 𝕜 g') = (toDual 𝕜 𝕜) g' := by
    ext; simp
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    g : 𝕜 → 𝕜
    g' u : 𝕜
    L' : Filter 𝕜
    h : HasGradientAtFilter g g' u L'
    this : Eq (ContinuousLinearMap.smulRight 1 ((starRingEnd 𝕜) g')) ((InnerProduc …
    ⊢ HasDerivAtFilter g ((starRingEnd 𝕜) g') u L'
  -/
  rwa [HasDerivAtFilter, this]
  /-
    🎉 no goals
  -/


theorem HasDerivAtFilter.hasGradientAtFilter (h : HasDerivAtFilter g g' u L') :
    HasGradientAtFilter g (starRingEnd 𝕜 g') u L' := by
  have : ContinuousLinearMap.smulRight (1 : 𝕜 →L[𝕜] 𝕜) g' = (toDual 𝕜 𝕜) (starRingEnd 𝕜 g') := by
    ext; simp
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    g : 𝕜 → 𝕜
    g' u : 𝕜
    L' : Filter 𝕜
    h : HasDerivAtFilter g g' u L'
    this : Eq (ContinuousLinearMap.smulRight 1 g') ((InnerProductSpace.toDual 𝕜 𝕜) …
    ⊢ HasGradientAtFilter g ((starRingEnd 𝕜) g') u L'
  -/
  rwa [HasGradientAtFilter, ← this]
  /-
    🎉 no goals
  -/


theorem HasGradientAt.hasDerivAt (h : HasGradientAt g g' u) :
    HasDerivAt g (starRingEnd 𝕜 g') u := by
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    g : 𝕜 → 𝕜
    g' u : 𝕜
    h : HasGradientAt g g' u
    ⊢ HasDerivAt g ((starRingEnd 𝕜) g') u
  -/
  rw [hasGradientAt_iff_hasFDerivAt, hasFDerivAt_iff_hasDerivAt] at h
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    g : 𝕜 → 𝕜
    g' u : 𝕜
    h : HasDerivAt g (((InnerProductSpace.toDual 𝕜 𝕜) g') 1) u
    ⊢ HasDerivAt g ((starRingEnd 𝕜) g') u
  -/
  simpa using h
  /-
    🎉 no goals
  -/


theorem HasDerivAt.hasGradientAt (h : HasDerivAt g g' u) :
    HasGradientAt g (starRingEnd 𝕜 g') u := by
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    g : 𝕜 → 𝕜
    g' u : 𝕜
    h : HasDerivAt g g' u
    ⊢ HasGradientAt g ((starRingEnd 𝕜) g') u
  -/
  rw [hasGradientAt_iff_hasFDerivAt, hasFDerivAt_iff_hasDerivAt]
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    g : 𝕜 → 𝕜
    g' u : 𝕜
    h : HasDerivAt g g' u
    ⊢ HasDerivAt g (((InnerProductSpace.toDual 𝕜 𝕜) ((starRingEnd 𝕜) g')) 1) u
  -/
  simpa
  /-
    🎉 no goals
  -/


theorem gradient_eq_deriv : ∇ g u = starRingEnd 𝕜 (deriv g u) := by
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    g : 𝕜 → 𝕜
    u : 𝕜
    ⊢ Eq (gradient g u) ((starRingEnd 𝕜) (deriv g u))
  -/
  by_cases h : DifferentiableAt 𝕜 g u
    /-
      case pos
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      g : 𝕜 → 𝕜
      u : 𝕜
      h : DifferentiableAt 𝕜 g u
      ⊢ Eq (gradient g u) ((starRingEnd 𝕜) (deriv g u))
    -/
  · rw [h.hasGradientAt.hasDerivAt.deriv, RCLike.conj_conj]
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      g : 𝕜 → 𝕜
      u : 𝕜
      h : Not (DifferentiableAt 𝕜 g u)
      ⊢ Eq (gradient g u) ((starRingEnd 𝕜) (deriv g u))
    -/
  · rw [gradient_eq_zero_of_not_differentiableAt h, deriv_zero_of_not_differentiableAt h, map_zero]
    /-
      🎉 no goals
    -/


theorem HasGradientAtFilter.hasDerivAtFilter' (h : HasGradientAtFilter g g' u L') :
    HasDerivAtFilter g g' u L' := h.hasDerivAtFilter


theorem HasDerivAtFilter.hasGradientAtFilter' (h : HasDerivAtFilter g g' u L') :
    HasGradientAtFilter g g' u L' := h.hasGradientAtFilter


theorem HasGradientAt.hasDerivAt' (h : HasGradientAt g g' u) :
    HasDerivAt g g' u := h.hasDerivAt


theorem HasDerivAt.hasGradientAt' (h : HasDerivAt g g' u) :
    HasGradientAt g g' u := h.hasGradientAt


theorem gradient_eq_deriv' : ∇ g u = deriv g u := gradient_eq_deriv


theorem hasGradientAtFilter_iff_isLittleO :
    HasGradientAtFilter f f' x L ↔
    (fun x' : F => f x' - f x - ⟪f', x' - x⟫) =o[L] fun x' => x' - x :=
  hasFDerivAtFilter_iff_isLittleO ..


theorem hasGradientWithinAt_iff_isLittleO :
    HasGradientWithinAt f f' s x ↔
    (fun x' : F => f x' - f x - ⟪f', x' - x⟫) =o[𝓝[s] x] fun x' => x' - x :=
  hasGradientAtFilter_iff_isLittleO


theorem hasGradientWithinAt_iff_tendsto :
    HasGradientWithinAt f f' s x ↔
    Tendsto (fun x' => ‖x' - x‖⁻¹ * ‖f x' - f x - ⟪f', x' - x⟫‖) (𝓝[s] x) (𝓝 0) :=
  hasFDerivAtFilter_iff_tendsto


theorem hasGradientAt_iff_isLittleO : HasGradientAt f f' x ↔
    (fun x' : F => f x' - f x - ⟪f', x' - x⟫) =o[𝓝 x] fun x' => x' - x :=
  hasGradientAtFilter_iff_isLittleO


theorem hasGradientAt_iff_tendsto :
    HasGradientAt f f' x ↔
    Tendsto (fun x' => ‖x' - x‖⁻¹ * ‖f x' - f x - ⟪f', x' - x⟫‖) (𝓝 x) (𝓝 0) :=
  hasFDerivAtFilter_iff_tendsto


theorem HasGradientAtFilter.isBigO_sub (h : HasGradientAtFilter f f' x L) :
    (fun x' => f x' - f x) =O[L] fun x' => x' - x :=
  HasFDerivAtFilter.isBigO_sub h


theorem hasGradientWithinAt_congr_set' {s t : Set F} (y : F) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
    HasGradientWithinAt f f' s x ↔ HasGradientWithinAt f f' t x :=
  hasFDerivWithinAt_congr_set' y h


theorem hasGradientWithinAt_congr_set {s t : Set F} (h : s =ᶠ[𝓝 x] t) :
    HasGradientWithinAt f f' s x ↔ HasGradientWithinAt f f' t x :=
  hasFDerivWithinAt_congr_set h


theorem hasGradientAt_iff_isLittleO_nhds_zero : HasGradientAt f f' x ↔
    (fun h => f (x + h) - f x - ⟪f', h⟫) =o[𝓝 0] fun h => h :=
  hasFDerivAt_iff_isLittleO_nhds_zero


theorem Filter.EventuallyEq.hasGradientAtFilter_iff (h₀ : f₀ =ᶠ[L] f₁) (hx : f₀ x = f₁ x)
    (h₁ : f₀' = f₁') : HasGradientAtFilter f₀ f₀' x L ↔ HasGradientAtFilter f₁ f₁' x L :=
                                  /-
                                    𝕜 : Type u_1
                                    F : Type u_2
                                    inst✝³ : RCLike 𝕜
                                    inst✝² : NormedAddCommGroup F
                                    inst✝¹ : InnerProductSpace 𝕜 F
                                    inst✝ : CompleteSpace F
                                    x : F
                                    L : Filter F
                                    f₀ f₁ : F → 𝕜
                                    f₀' f₁' : F
                                    h₀ : L.EventuallyEq f₀ f₁
                                    hx : Eq (f₀ x) (f₁ x)
                                    h₁ : Eq f₀' f₁'
                                    ⊢ ∀ (x : F), Eq (((InnerProductSpace.toDual 𝕜 F) f₀') x) (((InnerProductSpace. …
                                  -/
  h₀.hasFDerivAtFilter_iff hx (by simp [h₁])
                                  /-
                                    🎉 no goals
                                  -/


theorem HasGradientAtFilter.congr_of_eventuallyEq (h : HasGradientAtFilter f f' x L)
    (hL : f₁ =ᶠ[L] f) (hx : f₁ x = f x) : HasGradientAtFilter f₁ f' x L := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : F → 𝕜
    f' x : F
    L : Filter F
    f₁ : F → 𝕜
    h : HasGradientAtFilter f f' x L
    hL : L.EventuallyEq f₁ f
    hx : Eq (f₁ x) (f x)
    ⊢ HasGradientAtFilter f₁ f' x L
  -/
  rwa [hL.hasGradientAtFilter_iff hx rfl]
  /-
    🎉 no goals
  -/


theorem HasGradientWithinAt.congr_mono (h : HasGradientWithinAt f f' s x) (ht : ∀ x ∈ t, f₁ x = f x)
    (hx : f₁ x = f x) (h₁ : t ⊆ s) : HasGradientWithinAt f₁ f' t x :=
  HasFDerivWithinAt.congr_mono h ht hx h₁


theorem HasGradientWithinAt.congr (h : HasGradientWithinAt f f' s x) (hs : ∀ x ∈ s, f₁ x = f x)
    (hx : f₁ x = f x) : HasGradientWithinAt f₁ f' s x :=
                         /-
                           𝕜 : Type u_1
                           F : Type u_2
                           inst✝³ : RCLike 𝕜
                           inst✝² : NormedAddCommGroup F
                           inst✝¹ : InnerProductSpace 𝕜 F
                           inst✝ : CompleteSpace F
                           f : F → 𝕜
                           f' x : F
                           s : Set F
                           f₁ : F → 𝕜
                           h : HasGradientWithinAt f f' s x
                           hs : ∀ (x : F), Membership.mem s x → Eq (f₁ x) (f x)
                           hx : Eq (f₁ x) (f x)
                           ⊢ HasSubset.Subset s s
                         -/
  h.congr_mono hs hx (by tauto)
                         /-
                           🎉 no goals
                         -/


theorem HasGradientWithinAt.congr_of_mem (h : HasGradientWithinAt f f' s x)
    (hs : ∀ x ∈ s, f₁ x = f x) (hx : x ∈ s) : HasGradientWithinAt f₁ f' s x :=
  h.congr hs (hs _ hx)


theorem HasGradientWithinAt.congr_of_eventuallyEq (h : HasGradientWithinAt f f' s x)
    (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) : HasGradientWithinAt f₁ f' s x :=
  HasGradientAtFilter.congr_of_eventuallyEq h h₁ hx


theorem HasGradientWithinAt.congr_of_eventuallyEq_of_mem (h : HasGradientWithinAt f f' s x)
    (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : x ∈ s) : HasGradientWithinAt f₁ f' s x :=
  h.congr_of_eventuallyEq h₁ (h₁.eq_of_nhdsWithin hx)


theorem HasGradientAt.congr_of_eventuallyEq (h : HasGradientAt f f' x) (h₁ : f₁ =ᶠ[𝓝 x] f) :
    HasGradientAt f₁ f' x :=
  HasGradientAtFilter.congr_of_eventuallyEq h h₁ (mem_of_mem_nhds h₁ : _)


theorem Filter.EventuallyEq.gradient_eq (hL : f₁ =ᶠ[𝓝 x] f) : ∇ f₁ x = ∇ f x := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : F → 𝕜
    x : F
    f₁ : F → 𝕜
    hL : (nhds x).EventuallyEq f₁ f
    ⊢ Eq (gradient f₁ x) (gradient f x)
  -/
  unfold gradient
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : F → 𝕜
    x : F
    f₁ : F → 𝕜
    hL : (nhds x).EventuallyEq f₁ f
    ⊢ Eq ((InnerProductSpace.toDual 𝕜 F).symm (fderiv 𝕜 f₁ x)) ((InnerProductSpace …
  -/
  rwa [Filter.EventuallyEq.fderiv_eq]
  /-
    🎉 no goals
  -/


protected theorem Filter.EventuallyEq.gradient (h : f₁ =ᶠ[𝓝 x] f) : ∇ f₁ =ᶠ[𝓝 x] ∇ f :=
  h.eventuallyEq_nhds.mono fun _ h => h.gradient_eq


theorem hasGradientAtFilter_const : HasGradientAtFilter (fun _ => c) 0 x L := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    inst✝ : CompleteSpace F
    x : F
    L : Filter F
    c : 𝕜
    ⊢ HasGradientAtFilter (fun x => c) 0 x L
  -/
  rw [HasGradientAtFilter, map_zero]; apply hasFDerivAtFilter_const c x L
                                      /-
                                        🎉 no goals
                                      -/


theorem hasGradientWithinAt_const : HasGradientWithinAt (fun _ => c) 0 s x :=
  hasGradientAtFilter_const _ _ _


theorem hasGradientAt_const : HasGradientAt (fun _ => c) 0 x :=
  hasGradientAtFilter_const _ _ _


theorem gradient_const : ∇ (fun _ => c) x = 0 := by
  /-
    𝕜 : Type u_1
    F : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace 𝕜 F
    inst✝ : CompleteSpace F
    x : F
    c : 𝕜
    ⊢ Eq (gradient (fun x => c) x) 0
  -/
  rw [gradient, fderiv_const, Pi.zero_apply, map_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem gradient_const' : (∇ fun _ : 𝕜 => c) = fun _ => 0 :=
  funext fun x => gradient_const x c


nonrec theorem HasGradientAtFilter.tendsto_nhds (hL : L ≤ 𝓝 x) (h : HasGradientAtFilter f f' x L) :
    Tendsto f L (𝓝 (f x)) :=
  h.tendsto_nhds hL


theorem HasGradientWithinAt.continuousWithinAt (h : HasGradientWithinAt f f' s x) :
    ContinuousWithinAt f s x :=
  HasGradientAtFilter.tendsto_nhds inf_le_left h


theorem HasGradientAt.continuousAt (h : HasGradientAt f f' x) : ContinuousAt f x :=
  HasGradientAtFilter.tendsto_nhds le_rfl h


protected theorem HasGradientAt.continuousOn {f' : F → F} (h : ∀ x ∈ s, HasGradientAt f (f' x) x) :
    ContinuousOn f s :=
  fun x hx => (h x hx).continuousAt.continuousWithinAt


