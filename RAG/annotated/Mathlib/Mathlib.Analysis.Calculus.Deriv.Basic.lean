/-- `f` has the derivative `f'` at the point `x` as `x` goes along the filter `L`.

That is, `f x' = f x + (x' - x) • f' + o(x' - x)` where `x'` converges along the filter `L`.
-/
def HasDerivAtFilter (f : 𝕜 → F) (f' : F) (x : 𝕜) (L : Filter 𝕜) :=
  HasFDerivAtFilter f (smulRight (1 : 𝕜 →L[𝕜] 𝕜) f') x L


/-- `f` has the derivative `f'` at the point `x` within the subset `s`.

That is, `f x' = f x + (x' - x) • f' + o(x' - x)` where `x'` converges to `x` inside `s`.
-/
def HasDerivWithinAt (f : 𝕜 → F) (f' : F) (s : Set 𝕜) (x : 𝕜) :=
  HasDerivAtFilter f f' x (𝓝[s] x)


/-- `f` has the derivative `f'` at the point `x`.

That is, `f x' = f x + (x' - x) • f' + o(x' - x)` where `x'` converges to `x`.
-/
def HasDerivAt (f : 𝕜 → F) (f' : F) (x : 𝕜) :=
  HasDerivAtFilter f f' x (𝓝 x)


/-- `f` has the derivative `f'` at the point `x` in the sense of strict differentiability.

That is, `f y - f z = (y - z) • f' + o(y - z)` as `y, z → x`. -/
def HasStrictDerivAt (f : 𝕜 → F) (f' : F) (x : 𝕜) :=
  HasStrictFDerivAt f (smulRight (1 : 𝕜 →L[𝕜] 𝕜) f') x


/-- Derivative of `f` at the point `x` within the set `s`, if it exists.  Zero otherwise.

If the derivative exists (i.e., `∃ f', HasDerivWithinAt f f' s x`), then
`f x' = f x + (x' - x) • derivWithin f s x + o(x' - x)` where `x'` converges to `x` inside `s`.
-/
def derivWithin (f : 𝕜 → F) (s : Set 𝕜) (x : 𝕜) :=
  fderivWithin 𝕜 f s x 1


/-- Derivative of `f` at the point `x`, if it exists.  Zero otherwise.

If the derivative exists (i.e., `∃ f', HasDerivAt f f' x`), then
`f x' = f x + (x' - x) • deriv f x + o(x' - x)` where `x'` converges to `x`.
-/
def deriv (f : 𝕜 → F) (x : 𝕜) :=
  fderiv 𝕜 f x 1


/-- Expressing `HasFDerivAtFilter f f' x L` in terms of `HasDerivAtFilter` -/
theorem hasFDerivAtFilter_iff_hasDerivAtFilter {f' : 𝕜 →L[𝕜] F} :
                                                                     /-
                                                                       𝕜 : Type u
                                                                       inst✝⁴ : NontriviallyNormedField 𝕜
                                                                       F : Type v
                                                                       inst✝³ : AddCommGroup F
                                                                       inst✝² : Module 𝕜 F
                                                                       inst✝¹ : TopologicalSpace F
                                                                       f : 𝕜 → F
                                                                       x : 𝕜
                                                                       L : Filter 𝕜
                                                                       inst✝ : ContinuousSMul 𝕜 F
                                                                       f' : ContinuousLinearMap (RingHom.id 𝕜) 𝕜 F
                                                                       ⊢ Iff (HasFDerivAtFilter f f' x L) (HasDerivAtFilter f (f' 1) x L)
                                                                     -/
    HasFDerivAtFilter f f' x L ↔ HasDerivAtFilter f (f' 1) x L := by simp [HasDerivAtFilter]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem HasFDerivAtFilter.hasDerivAtFilter {f' : 𝕜 →L[𝕜] F} :
    HasFDerivAtFilter f f' x L → HasDerivAtFilter f (f' 1) x L :=
  hasFDerivAtFilter_iff_hasDerivAtFilter.mp


/-- Expressing `HasFDerivWithinAt f f' s x` in terms of `HasDerivWithinAt` -/
theorem hasFDerivWithinAt_iff_hasDerivWithinAt {f' : 𝕜 →L[𝕜] F} :
    HasFDerivWithinAt f f' s x ↔ HasDerivWithinAt f (f' 1) s x :=
  hasFDerivAtFilter_iff_hasDerivAtFilter


/-- Expressing `HasDerivWithinAt f f' s x` in terms of `HasFDerivWithinAt` -/
theorem hasDerivWithinAt_iff_hasFDerivWithinAt {f' : F} :
    HasDerivWithinAt f f' s x ↔ HasFDerivWithinAt f (smulRight (1 : 𝕜 →L[𝕜] 𝕜) f') s x :=
  Iff.rfl


theorem HasFDerivWithinAt.hasDerivWithinAt {f' : 𝕜 →L[𝕜] F} :
    HasFDerivWithinAt f f' s x → HasDerivWithinAt f (f' 1) s x :=
  hasFDerivWithinAt_iff_hasDerivWithinAt.mp


theorem HasDerivWithinAt.hasFDerivWithinAt {f' : F} :
    HasDerivWithinAt f f' s x → HasFDerivWithinAt f (smulRight (1 : 𝕜 →L[𝕜] 𝕜) f') s x :=
  hasDerivWithinAt_iff_hasFDerivWithinAt.mp


/-- Expressing `HasFDerivAt f f' x` in terms of `HasDerivAt` -/
theorem hasFDerivAt_iff_hasDerivAt {f' : 𝕜 →L[𝕜] F} : HasFDerivAt f f' x ↔ HasDerivAt f (f' 1) x :=
  hasFDerivAtFilter_iff_hasDerivAtFilter


theorem HasFDerivAt.hasDerivAt {f' : 𝕜 →L[𝕜] F} : HasFDerivAt f f' x → HasDerivAt f (f' 1) x :=
  hasFDerivAt_iff_hasDerivAt.mp


theorem hasStrictFDerivAt_iff_hasStrictDerivAt {f' : 𝕜 →L[𝕜] F} :
    HasStrictFDerivAt f f' x ↔ HasStrictDerivAt f (f' 1) x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜 F
    inst✝¹ : TopologicalSpace F
    f : 𝕜 → F
    x : 𝕜
    inst✝ : ContinuousSMul 𝕜 F
    f' : ContinuousLinearMap (RingHom.id 𝕜) 𝕜 F
    ⊢ Iff (HasStrictFDerivAt f f' x) (HasStrictDerivAt f (f' 1) x)
  -/
  simp [HasStrictDerivAt, HasStrictFDerivAt]
  /-
    🎉 no goals
  -/


protected theorem HasStrictFDerivAt.hasStrictDerivAt {f' : 𝕜 →L[𝕜] F} :
    HasStrictFDerivAt f f' x → HasStrictDerivAt f (f' 1) x :=
  hasStrictFDerivAt_iff_hasStrictDerivAt.mp


theorem hasStrictDerivAt_iff_hasStrictFDerivAt :
    HasStrictDerivAt f f' x ↔ HasStrictFDerivAt f (smulRight (1 : 𝕜 →L[𝕜] 𝕜) f') x :=
  Iff.rfl


alias ⟨HasStrictDerivAt.hasStrictFDerivAt, _⟩ := hasStrictDerivAt_iff_hasStrictFDerivAt


/-- Expressing `HasDerivAt f f' x` in terms of `HasFDerivAt` -/
theorem hasDerivAt_iff_hasFDerivAt {f' : F} :
    HasDerivAt f f' x ↔ HasFDerivAt f (smulRight (1 : 𝕜 →L[𝕜] 𝕜) f') x :=
  Iff.rfl


alias ⟨HasDerivAt.hasFDerivAt, _⟩ := hasDerivAt_iff_hasFDerivAt


theorem derivWithin_zero_of_not_differentiableWithinAt (h : ¬DifferentiableWithinAt 𝕜 f s x) :
    derivWithin f s x = 0 := by
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 F
    inst✝ : TopologicalSpace F
    f : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    h : Not (DifferentiableWithinAt 𝕜 f s x)
    ⊢ Eq (derivWithin f s x) 0
  -/
  unfold derivWithin
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 F
    inst✝ : TopologicalSpace F
    f : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    h : Not (DifferentiableWithinAt 𝕜 f s x)
    ⊢ Eq ((fderivWithin 𝕜 f s x) 1) 0
  -/
  rw [fderivWithin_zero_of_not_differentiableWithinAt h]
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 F
    inst✝ : TopologicalSpace F
    f : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    h : Not (DifferentiableWithinAt 𝕜 f s x)
    ⊢ Eq (0 1) 0
  -/
  simp
  /-
    🎉 no goals
  -/


theorem differentiableWithinAt_of_derivWithin_ne_zero (h : derivWithin f s x ≠ 0) :
    DifferentiableWithinAt 𝕜 f s x :=
  not_imp_comm.1 derivWithin_zero_of_not_differentiableWithinAt h


theorem derivWithin_zero_of_isolated (h : 𝓝[s \ {x}] x = ⊥) : derivWithin f s x = 0 := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    h : Eq (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))) Bot.bot
    ⊢ Eq (derivWithin f s x) 0
  -/
  rw [derivWithin, fderivWithin_zero_of_isolated h, ContinuousLinearMap.zero_apply]
  /-
    🎉 no goals
  -/


theorem derivWithin_zero_of_nmem_closure (h : x ∉ closure s) : derivWithin f s x = 0 := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    h : Not (Membership.mem (closure s) x)
    ⊢ Eq (derivWithin f s x) 0
  -/
  rw [derivWithin, fderivWithin_zero_of_nmem_closure h, ContinuousLinearMap.zero_apply]
  /-
    🎉 no goals
  -/


theorem deriv_zero_of_not_differentiableAt (h : ¬DifferentiableAt 𝕜 f x) : deriv f x = 0 := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    h : Not (DifferentiableAt 𝕜 f x)
    ⊢ Eq (deriv f x) 0
  -/
  unfold deriv
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    h : Not (DifferentiableAt 𝕜 f x)
    ⊢ Eq ((fderiv 𝕜 f x) 1) 0
  -/
  rw [fderiv_zero_of_not_differentiableAt h]
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    h : Not (DifferentiableAt 𝕜 f x)
    ⊢ Eq (0 1) 0
  -/
  simp
  /-
    🎉 no goals
  -/


theorem differentiableAt_of_deriv_ne_zero (h : deriv f x ≠ 0) : DifferentiableAt 𝕜 f x :=
  not_imp_comm.1 deriv_zero_of_not_differentiableAt h


theorem UniqueDiffWithinAt.eq_deriv (s : Set 𝕜) (H : UniqueDiffWithinAt 𝕜 s x)
    (h : HasDerivWithinAt f f' s x) (h₁ : HasDerivWithinAt f f₁' s x) : f' = f₁' :=
  smulRight_one_eq_iff.mp <| UniqueDiffWithinAt.eq H h h₁


theorem hasDerivAtFilter_iff_isLittleO :
    HasDerivAtFilter f f' x L ↔ (fun x' : 𝕜 => f x' - f x - (x' - x) • f') =o[L] fun x' => x' - x :=
  hasFDerivAtFilter_iff_isLittleO ..


theorem hasDerivAtFilter_iff_tendsto :
    HasDerivAtFilter f f' x L ↔
      Tendsto (fun x' : 𝕜 => ‖x' - x‖⁻¹ * ‖f x' - f x - (x' - x) • f'‖) L (𝓝 0) :=
  hasFDerivAtFilter_iff_tendsto


theorem hasDerivWithinAt_iff_isLittleO :
    HasDerivWithinAt f f' s x ↔
      (fun x' : 𝕜 => f x' - f x - (x' - x) • f') =o[𝓝[s] x] fun x' => x' - x :=
  hasFDerivAtFilter_iff_isLittleO ..


theorem hasDerivWithinAt_iff_tendsto :
    HasDerivWithinAt f f' s x ↔
      Tendsto (fun x' => ‖x' - x‖⁻¹ * ‖f x' - f x - (x' - x) • f'‖) (𝓝[s] x) (𝓝 0) :=
  hasFDerivAtFilter_iff_tendsto


theorem hasDerivAt_iff_isLittleO :
    HasDerivAt f f' x ↔ (fun x' : 𝕜 => f x' - f x - (x' - x) • f') =o[𝓝 x] fun x' => x' - x :=
  hasFDerivAtFilter_iff_isLittleO ..


theorem hasDerivAt_iff_tendsto :
    HasDerivAt f f' x ↔ Tendsto (fun x' => ‖x' - x‖⁻¹ * ‖f x' - f x - (x' - x) • f'‖) (𝓝 x) (𝓝 0) :=
  hasFDerivAtFilter_iff_tendsto


theorem HasDerivAtFilter.isBigO_sub (h : HasDerivAtFilter f f' x L) :
    (fun x' => f x' - f x) =O[L] fun x' => x' - x :=
  HasFDerivAtFilter.isBigO_sub h


nonrec theorem HasDerivAtFilter.isBigO_sub_rev (hf : HasDerivAtFilter f f' x L) (hf' : f' ≠ 0) :
    (fun x' => x' - x) =O[L] fun x' => f x' - f x :=
  suffices AntilipschitzWith ‖f'‖₊⁻¹ (smulRight (1 : 𝕜 →L[𝕜] 𝕜) f') from hf.isBigO_sub_rev this
  AddMonoidHomClass.antilipschitz_of_bound (smulRight (1 : 𝕜 →L[𝕜] 𝕜) f') fun x => by
    /-
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : 𝕜 → F
      f' : F
      x✝ : 𝕜
      L : Filter 𝕜
      hf : HasDerivAtFilter f f' x✝ L
      hf' : Ne f' 0
      x : 𝕜
      ⊢ LE.le (Norm.norm x) (HMul.hMul (↑(Inv.inv (NNNorm.nnnorm f'))) (Norm.norm (( …
    -/
    simp [norm_smul, ← div_eq_inv_mul, mul_div_cancel_right₀ _ (mt norm_eq_zero.1 hf')]
    /-
      🎉 no goals
    -/


theorem HasStrictDerivAt.hasDerivAt (h : HasStrictDerivAt f f' x) : HasDerivAt f f' x :=
  h.hasFDerivAt


theorem hasDerivWithinAt_congr_set' {s t : Set 𝕜} (y : 𝕜) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
    HasDerivWithinAt f f' s x ↔ HasDerivWithinAt f f' t x :=
  hasFDerivWithinAt_congr_set' y h


theorem hasDerivWithinAt_congr_set {s t : Set 𝕜} (h : s =ᶠ[𝓝 x] t) :
    HasDerivWithinAt f f' s x ↔ HasDerivWithinAt f f' t x :=
  hasFDerivWithinAt_congr_set h


alias ⟨HasDerivWithinAt.congr_set, _⟩ := hasDerivWithinAt_congr_set


@[simp]
theorem hasDerivWithinAt_diff_singleton :
    HasDerivWithinAt f f' (s \ {x}) x ↔ HasDerivWithinAt f f' s x :=
  hasFDerivWithinAt_diff_singleton _


@[simp]
theorem hasDerivWithinAt_Ioi_iff_Ici [PartialOrder 𝕜] :
    HasDerivWithinAt f f' (Ioi x) x ↔ HasDerivWithinAt f f' (Ici x) x := by
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x : 𝕜
    inst✝ : PartialOrder 𝕜
    ⊢ Iff (HasDerivWithinAt f f' (Set.Ioi x) x) (HasDerivWithinAt f f' (Set.Ici x) …
  -/
  rw [← Ici_diff_left, hasDerivWithinAt_diff_singleton]
  /-
    🎉 no goals
  -/


alias ⟨HasDerivWithinAt.Ici_of_Ioi, HasDerivWithinAt.Ioi_of_Ici⟩ := hasDerivWithinAt_Ioi_iff_Ici


@[simp]
theorem hasDerivWithinAt_Iio_iff_Iic [PartialOrder 𝕜] :
    HasDerivWithinAt f f' (Iio x) x ↔ HasDerivWithinAt f f' (Iic x) x := by
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x : 𝕜
    inst✝ : PartialOrder 𝕜
    ⊢ Iff (HasDerivWithinAt f f' (Set.Iio x) x) (HasDerivWithinAt f f' (Set.Iic x) …
  -/
  rw [← Iic_diff_right, hasDerivWithinAt_diff_singleton]
  /-
    🎉 no goals
  -/


alias ⟨HasDerivWithinAt.Iic_of_Iio, HasDerivWithinAt.Iio_of_Iic⟩ := hasDerivWithinAt_Iio_iff_Iic


theorem HasDerivWithinAt.Ioi_iff_Ioo [LinearOrder 𝕜] [OrderClosedTopology 𝕜] {x y : 𝕜} (h : x < y) :
    HasDerivWithinAt f f' (Ioo x y) x ↔ HasDerivWithinAt f f' (Ioi x) x :=
  hasFDerivWithinAt_inter <| Iio_mem_nhds h


alias ⟨HasDerivWithinAt.Ioi_of_Ioo, HasDerivWithinAt.Ioo_of_Ioi⟩ := HasDerivWithinAt.Ioi_iff_Ioo


theorem hasDerivAt_iff_isLittleO_nhds_zero :
    HasDerivAt f f' x ↔ (fun h => f (x + h) - f x - h • f') =o[𝓝 0] fun h => h :=
  hasFDerivAt_iff_isLittleO_nhds_zero


theorem HasDerivAtFilter.mono (h : HasDerivAtFilter f f' x L₂) (hst : L₁ ≤ L₂) :
    HasDerivAtFilter f f' x L₁ :=
  HasFDerivAtFilter.mono h hst


theorem HasDerivWithinAt.mono (h : HasDerivWithinAt f f' t x) (hst : s ⊆ t) :
    HasDerivWithinAt f f' s x :=
  HasFDerivWithinAt.mono h hst


theorem HasDerivWithinAt.mono_of_mem_nhdsWithin (h : HasDerivWithinAt f f' t x) (hst : t ∈ 𝓝[s] x) :
    HasDerivWithinAt f f' s x :=
  HasFDerivWithinAt.mono_of_mem_nhdsWithin h hst


@[deprecated (since := "2024-10-31")]
alias HasDerivWithinAt.mono_of_mem := HasDerivWithinAt.mono_of_mem_nhdsWithin


theorem HasDerivAt.hasDerivAtFilter (h : HasDerivAt f f' x) (hL : L ≤ 𝓝 x) :
    HasDerivAtFilter f f' x L :=
  HasFDerivAt.hasFDerivAtFilter h hL


theorem HasDerivAt.hasDerivWithinAt (h : HasDerivAt f f' x) : HasDerivWithinAt f f' s x :=
  HasFDerivAt.hasFDerivWithinAt h


theorem HasDerivWithinAt.differentiableWithinAt (h : HasDerivWithinAt f f' s x) :
    DifferentiableWithinAt 𝕜 f s x :=
  HasFDerivWithinAt.differentiableWithinAt h


theorem HasDerivAt.differentiableAt (h : HasDerivAt f f' x) : DifferentiableAt 𝕜 f x :=
  HasFDerivAt.differentiableAt h


@[simp]
theorem hasDerivWithinAt_univ : HasDerivWithinAt f f' univ x ↔ HasDerivAt f f' x :=
  hasFDerivWithinAt_univ


theorem HasDerivAt.unique (h₀ : HasDerivAt f f₀' x) (h₁ : HasDerivAt f f₁' x) : f₀' = f₁' :=
  smulRight_one_eq_iff.mp <| h₀.hasFDerivAt.unique h₁


theorem hasDerivWithinAt_inter' (h : t ∈ 𝓝[s] x) :
    HasDerivWithinAt f f' (s ∩ t) x ↔ HasDerivWithinAt f f' s x :=
  hasFDerivWithinAt_inter' h


theorem hasDerivWithinAt_inter (h : t ∈ 𝓝 x) :
    HasDerivWithinAt f f' (s ∩ t) x ↔ HasDerivWithinAt f f' s x :=
  hasFDerivWithinAt_inter h


theorem HasDerivWithinAt.union (hs : HasDerivWithinAt f f' s x) (ht : HasDerivWithinAt f f' t x) :
    HasDerivWithinAt f f' (s ∪ t) x :=
  hs.hasFDerivWithinAt.union ht.hasFDerivWithinAt


theorem HasDerivWithinAt.hasDerivAt (h : HasDerivWithinAt f f' s x) (hs : s ∈ 𝓝 x) :
    HasDerivAt f f' x :=
  HasFDerivWithinAt.hasFDerivAt h hs


theorem DifferentiableWithinAt.hasDerivWithinAt (h : DifferentiableWithinAt 𝕜 f s x) :
    HasDerivWithinAt f (derivWithin f s x) s x :=
  h.hasFDerivWithinAt.hasDerivWithinAt


theorem DifferentiableAt.hasDerivAt (h : DifferentiableAt 𝕜 f x) : HasDerivAt f (deriv f x) x :=
  h.hasFDerivAt.hasDerivAt


@[simp]
theorem hasDerivAt_deriv_iff : HasDerivAt f (deriv f x) x ↔ DifferentiableAt 𝕜 f x :=
  ⟨fun h => h.differentiableAt, fun h => h.hasDerivAt⟩


@[simp]
theorem hasDerivWithinAt_derivWithin_iff :
    HasDerivWithinAt f (derivWithin f s x) s x ↔ DifferentiableWithinAt 𝕜 f s x :=
  ⟨fun h => h.differentiableWithinAt, fun h => h.hasDerivWithinAt⟩


theorem DifferentiableOn.hasDerivAt (h : DifferentiableOn 𝕜 f s) (hs : s ∈ 𝓝 x) :
    HasDerivAt f (deriv f x) x :=
  (h.hasFDerivAt hs).hasDerivAt


theorem HasDerivAt.deriv (h : HasDerivAt f f' x) : deriv f x = f' :=
  h.differentiableAt.hasDerivAt.unique h


theorem deriv_eq {f' : 𝕜 → F} (h : ∀ x, HasDerivAt f (f' x) x) : deriv f = f' :=
  funext fun x => (h x).deriv


theorem HasDerivWithinAt.derivWithin (h : HasDerivWithinAt f f' s x)
    (hxs : UniqueDiffWithinAt 𝕜 s x) : derivWithin f s x = f' :=
  hxs.eq_deriv _ h.differentiableWithinAt.hasDerivWithinAt h


theorem fderivWithin_derivWithin : (fderivWithin 𝕜 f s x : 𝕜 → F) 1 = derivWithin f s x :=
  rfl


theorem derivWithin_fderivWithin :
                                                                               /-
                                                                                 𝕜 : Type u
                                                                                 inst✝² : NontriviallyNormedField 𝕜
                                                                                 F : Type v
                                                                                 inst✝¹ : NormedAddCommGroup F
                                                                                 inst✝ : NormedSpace 𝕜 F
                                                                                 f : 𝕜 → F
                                                                                 x : 𝕜
                                                                                 s : Set 𝕜
                                                                                 ⊢ Eq (ContinuousLinearMap.smulRight 1 (derivWithin f s x)) (fderivWithin 𝕜 f s …
                                                                               -/
    smulRight (1 : 𝕜 →L[𝕜] 𝕜) (derivWithin f s x) = fderivWithin 𝕜 f s x := by simp [derivWithin]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


theorem norm_derivWithin_eq_norm_fderivWithin : ‖derivWithin f s x‖ = ‖fderivWithin 𝕜 f s x‖ := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    ⊢ Eq (Norm.norm (derivWithin f s x)) (Norm.norm (fderivWithin 𝕜 f s x))
  -/
  simp [← derivWithin_fderivWithin]
  /-
    🎉 no goals
  -/


theorem fderiv_deriv : (fderiv 𝕜 f x : 𝕜 → F) 1 = deriv f x :=
  rfl


@[simp]
theorem fderiv_eq_smul_deriv (y : 𝕜) : (fderiv 𝕜 f x : 𝕜 → F) y = y • deriv f x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x y : 𝕜
    ⊢ Eq ((fderiv 𝕜 f x) y) (HSMul.hSMul y (deriv f x))
  -/
  rw [← fderiv_deriv, ← ContinuousLinearMap.map_smul]
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x y : 𝕜
    ⊢ Eq ((fderiv 𝕜 f x) y) ((fderiv 𝕜 f x) (HSMul.hSMul y 1))
  -/
  simp only [smul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


theorem deriv_fderiv : smulRight (1 : 𝕜 →L[𝕜] 𝕜) (deriv f x) = fderiv 𝕜 f x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    ⊢ Eq (ContinuousLinearMap.smulRight 1 (deriv f x)) (fderiv 𝕜 f x)
  -/
  simp only [deriv, ContinuousLinearMap.smulRight_one_one]
  /-
    🎉 no goals
  -/


lemma fderiv_eq_deriv_mul {f : 𝕜 → 𝕜} {x y : 𝕜} : (fderiv 𝕜 f x : 𝕜 → 𝕜) y = (deriv f x) * y := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    f : 𝕜 → 𝕜
    x y : 𝕜
    ⊢ Eq ((fderiv 𝕜 f x) y) (HMul.hMul (deriv f x) y)
  -/
  simp [mul_comm]
  /-
    🎉 no goals
  -/


theorem norm_deriv_eq_norm_fderiv : ‖deriv f x‖ = ‖fderiv 𝕜 f x‖ := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    ⊢ Eq (Norm.norm (deriv f x)) (Norm.norm (fderiv 𝕜 f x))
  -/
  simp [← deriv_fderiv]
  /-
    🎉 no goals
  -/


theorem DifferentiableAt.derivWithin (h : DifferentiableAt 𝕜 f x) (hxs : UniqueDiffWithinAt 𝕜 s x) :
    derivWithin f s x = deriv f x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    h : DifferentiableAt 𝕜 f x
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (_root_.derivWithin f s x) (deriv f x)
  -/
  unfold _root_.derivWithin deriv
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    h : DifferentiableAt 𝕜 f x
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq ((fderivWithin 𝕜 f s x) 1) ((fderiv 𝕜 f x) 1)
  -/
  rw [h.fderivWithin hxs]
  /-
    🎉 no goals
  -/


theorem HasDerivWithinAt.deriv_eq_zero (hd : HasDerivWithinAt f 0 s x)
    (H : UniqueDiffWithinAt 𝕜 s x) : deriv f x = 0 :=
  (em' (DifferentiableAt 𝕜 f x)).elim deriv_zero_of_not_differentiableAt fun h =>
    H.eq_deriv _ h.hasDerivAt.hasDerivWithinAt hd


theorem derivWithin_of_mem_nhdsWithin (st : t ∈ 𝓝[s] x) (ht : UniqueDiffWithinAt 𝕜 s x)
    (h : DifferentiableWithinAt 𝕜 f t x) : derivWithin f s x = derivWithin f t x :=
  ((DifferentiableWithinAt.hasDerivWithinAt h).mono_of_mem_nhdsWithin st).derivWithin ht


@[deprecated (since := "2024-10-31")] alias derivWithin_of_mem := derivWithin_of_mem_nhdsWithin


theorem derivWithin_subset (st : s ⊆ t) (ht : UniqueDiffWithinAt 𝕜 s x)
    (h : DifferentiableWithinAt 𝕜 f t x) : derivWithin f s x = derivWithin f t x :=
  ((DifferentiableWithinAt.hasDerivWithinAt h).mono st).derivWithin ht


theorem derivWithin_congr_set' (y : 𝕜) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
                                                /-
                                                  𝕜 : Type u
                                                  inst✝² : NontriviallyNormedField 𝕜
                                                  F : Type v
                                                  inst✝¹ : NormedAddCommGroup F
                                                  inst✝ : NormedSpace 𝕜 F
                                                  f : 𝕜 → F
                                                  x : 𝕜
                                                  s t : Set 𝕜
                                                  y : 𝕜
                                                  h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
                                                  ⊢ Eq (derivWithin f s x) (derivWithin f t x)
                                                -/
    derivWithin f s x = derivWithin f t x := by simp only [derivWithin, fderivWithin_congr_set' y h]
                                                /-
                                                  🎉 no goals
                                                -/


theorem derivWithin_congr_set (h : s =ᶠ[𝓝 x] t) : derivWithin f s x = derivWithin f t x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    s t : Set 𝕜
    h : (nhds x).EventuallyEq s t
    ⊢ Eq (derivWithin f s x) (derivWithin f t x)
  -/
  simp only [derivWithin, fderivWithin_congr_set h]
  /-
    🎉 no goals
  -/


@[simp]
theorem derivWithin_univ : derivWithin f univ = deriv f := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    ⊢ Eq (derivWithin f Set.univ) (deriv f)
  -/
  ext
  /-
    case h
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x✝ : 𝕜
    ⊢ Eq (derivWithin f Set.univ x✝) (deriv f x✝)
  -/
  unfold derivWithin deriv
  /-
    case h
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x✝ : 𝕜
    ⊢ Eq ((fderivWithin 𝕜 f Set.univ x✝) 1) ((fderiv 𝕜 f x✝) 1)
  -/
  rw [fderivWithin_univ]
  /-
    🎉 no goals
  -/


theorem derivWithin_inter (ht : t ∈ 𝓝 x) : derivWithin f (s ∩ t) x = derivWithin f s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    s t : Set 𝕜
    ht : Membership.mem (nhds x) t
    ⊢ Eq (derivWithin f (Inter.inter s t) x) (derivWithin f s x)
  -/
  unfold derivWithin
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    s t : Set 𝕜
    ht : Membership.mem (nhds x) t
    ⊢ Eq ((fderivWithin 𝕜 f (Inter.inter s t) x) 1) ((fderivWithin 𝕜 f s x) 1)
  -/
  rw [fderivWithin_inter ht]
  /-
    🎉 no goals
  -/


theorem derivWithin_of_mem_nhds (h : s ∈ 𝓝 x) : derivWithin f s x = deriv f x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    h : Membership.mem (nhds x) s
    ⊢ Eq (derivWithin f s x) (deriv f x)
  -/
  simp only [derivWithin, deriv, fderivWithin_of_mem_nhds h]
  /-
    🎉 no goals
  -/


theorem derivWithin_of_isOpen (hs : IsOpen s) (hx : x ∈ s) : derivWithin f s x = deriv f x :=
  derivWithin_of_mem_nhds (hs.mem_nhds hx)


lemma deriv_eqOn {f' : 𝕜 → F} (hs : IsOpen s) (hf' : ∀ x ∈ s, HasDerivWithinAt f (f' x) s x) :
    s.EqOn (deriv f) f' := fun x hx ↦ by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s : Set 𝕜
    f' : 𝕜 → F
    hs : IsOpen s
    hf' : ∀ (x : 𝕜), Membership.mem s x → HasDerivWithinAt f (f' x) s x
    x : 𝕜
    hx : Membership.mem s x
    ⊢ Eq (deriv f x) (f' x)
  -/
  rw [← derivWithin_of_isOpen hs hx, (hf' _ hx).derivWithin <| hs.uniqueDiffWithinAt hx]
  /-
    🎉 no goals
  -/


theorem deriv_mem_iff {f : 𝕜 → F} {s : Set F} {x : 𝕜} :
    deriv f x ∈ s ↔
      DifferentiableAt 𝕜 f x ∧ deriv f x ∈ s ∨ ¬DifferentiableAt 𝕜 f x ∧ (0 : F) ∈ s := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s : Set F
    x : 𝕜
    ⊢ Iff (Membership.mem s (deriv f x)) (Or (And (DifferentiableAt 𝕜 f x) (Member …
  -/
                                           /-
                                             🎉 no goals
                                           -/
  by_cases hx : DifferentiableAt 𝕜 f x <;> simp [deriv_zero_of_not_differentiableAt, *]
                                           /-
                                             🎉 no goals
                                           -/


theorem derivWithin_mem_iff {f : 𝕜 → F} {t : Set 𝕜} {s : Set F} {x : 𝕜} :
    derivWithin f t x ∈ s ↔
      DifferentiableWithinAt 𝕜 f t x ∧ derivWithin f t x ∈ s ∨
        ¬DifferentiableWithinAt 𝕜 f t x ∧ (0 : F) ∈ s := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    t : Set 𝕜
    s : Set F
    x : 𝕜
    ⊢ Iff (Membership.mem s (derivWithin f t x)) (Or (And (DifferentiableWithinAt  …
  -/
  by_cases hx : DifferentiableWithinAt 𝕜 f t x <;>
    /-
      case pos
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : 𝕜 → F
      t : Set 𝕜
      s : Set F
      x : 𝕜
      hx : DifferentiableWithinAt 𝕜 f t x
      ⊢ Iff (Membership.mem s (derivWithin f t x)) (Or (And (DifferentiableWithinAt  …
    -/
    /-
      🎉 no goals
    -/
    simp [derivWithin_zero_of_not_differentiableWithinAt, *]
    /-
      🎉 no goals
    -/


theorem differentiableWithinAt_Ioi_iff_Ici [PartialOrder 𝕜] :
    DifferentiableWithinAt 𝕜 f (Ioi x) x ↔ DifferentiableWithinAt 𝕜 f (Ici x) x :=
  ⟨fun h => h.hasDerivWithinAt.Ici_of_Ioi.differentiableWithinAt, fun h =>
    h.hasDerivWithinAt.Ioi_of_Ici.differentiableWithinAt⟩

-- Golfed while splitting the file

theorem derivWithin_Ioi_eq_Ici {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] (f : ℝ → E)
    (x : ℝ) : derivWithin f (Ioi x) x = derivWithin f (Ici x) x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    x : Real
    ⊢ Eq (derivWithin f (Set.Ioi x) x) (derivWithin f (Set.Ici x) x)
  -/
  by_cases H : DifferentiableWithinAt ℝ f (Ioi x) x
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      x : Real
      H : DifferentiableWithinAt Real f (Set.Ioi x) x
      ⊢ Eq (derivWithin f (Set.Ioi x) x) (derivWithin f (Set.Ici x) x)
    -/
  · have A := H.hasDerivWithinAt.Ici_of_Ioi
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      x : Real
      H : DifferentiableWithinAt Real f (Set.Ioi x) x
      A : HasDerivWithinAt f (derivWithin f (Set.Ioi x) x) (Set.Ici x) x
      ⊢ Eq (derivWithin f (Set.Ioi x) x) (derivWithin f (Set.Ici x) x)
    -/
    have B := (differentiableWithinAt_Ioi_iff_Ici.1 H).hasDerivWithinAt
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      x : Real
      H : DifferentiableWithinAt Real f (Set.Ioi x) x
      A : HasDerivWithinAt f (derivWithin f (Set.Ioi x) x) (Set.Ici x) x
      B : HasDerivWithinAt f (derivWithin f (Set.Ici x) x) (Set.Ici x) x
      ⊢ Eq (derivWithin f (Set.Ioi x) x) (derivWithin f (Set.Ici x) x)
    -/
    simpa using (uniqueDiffOn_Ici x).eq left_mem_Ici A B
    /-
      🎉 no goals
    -/
  · rw [derivWithin_zero_of_not_differentiableWithinAt H,
      derivWithin_zero_of_not_differentiableWithinAt]
    /-
      case neg
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      x : Real
      H : Not (DifferentiableWithinAt Real f (Set.Ioi x) x)
      ⊢ Not (DifferentiableWithinAt Real f (Set.Ici x) x)
    -/
    rwa [differentiableWithinAt_Ioi_iff_Ici] at H
    /-
      🎉 no goals
    -/


theorem Filter.EventuallyEq.hasDerivAtFilter_iff (h₀ : f₀ =ᶠ[L] f₁) (hx : f₀ x = f₁ x)
    (h₁ : f₀' = f₁') : HasDerivAtFilter f₀ f₀' x L ↔ HasDerivAtFilter f₁ f₁' x L :=
                                  /-
                                    𝕜 : Type u
                                    inst✝² : NontriviallyNormedField 𝕜
                                    F : Type v
                                    inst✝¹ : NormedAddCommGroup F
                                    inst✝ : NormedSpace 𝕜 F
                                    f₀ f₁ : 𝕜 → F
                                    f₀' f₁' : F
                                    x : 𝕜
                                    L : Filter 𝕜
                                    h₀ : L.EventuallyEq f₀ f₁
                                    hx : Eq (f₀ x) (f₁ x)
                                    h₁ : Eq f₀' f₁'
                                    ⊢ ∀ (x : 𝕜), Eq ((ContinuousLinearMap.smulRight 1 f₀') x) ((ContinuousLinearMa …
                                  -/
  h₀.hasFDerivAtFilter_iff hx (by simp [h₁])
                                  /-
                                    🎉 no goals
                                  -/


theorem HasDerivAtFilter.congr_of_eventuallyEq (h : HasDerivAtFilter f f' x L) (hL : f₁ =ᶠ[L] f)
                                                         /-
                                                           𝕜 : Type u
                                                           inst✝² : NontriviallyNormedField 𝕜
                                                           F : Type v
                                                           inst✝¹ : NormedAddCommGroup F
                                                           inst✝ : NormedSpace 𝕜 F
                                                           f f₁ : 𝕜 → F
                                                           f' : F
                                                           x : 𝕜
                                                           L : Filter 𝕜
                                                           h : HasDerivAtFilter f f' x L
                                                           hL : L.EventuallyEq f₁ f
                                                           hx : Eq (f₁ x) (f x)
                                                           ⊢ HasDerivAtFilter f₁ f' x L
                                                         -/
    (hx : f₁ x = f x) : HasDerivAtFilter f₁ f' x L := by rwa [hL.hasDerivAtFilter_iff hx rfl]
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem HasDerivWithinAt.congr_mono (h : HasDerivWithinAt f f' s x) (ht : ∀ x ∈ t, f₁ x = f x)
    (hx : f₁ x = f x) (h₁ : t ⊆ s) : HasDerivWithinAt f₁ f' t x :=
  HasFDerivWithinAt.congr_mono h ht hx h₁


theorem HasDerivWithinAt.congr (h : HasDerivWithinAt f f' s x) (hs : ∀ x ∈ s, f₁ x = f x)
    (hx : f₁ x = f x) : HasDerivWithinAt f₁ f' s x :=
  h.congr_mono hs hx (Subset.refl _)


theorem HasDerivWithinAt.congr_of_mem (h : HasDerivWithinAt f f' s x) (hs : ∀ x ∈ s, f₁ x = f x)
    (hx : x ∈ s) : HasDerivWithinAt f₁ f' s x :=
  h.congr hs (hs _ hx)


theorem HasDerivWithinAt.congr_of_eventuallyEq (h : HasDerivWithinAt f f' s x)
    (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) : HasDerivWithinAt f₁ f' s x :=
  HasDerivAtFilter.congr_of_eventuallyEq h h₁ hx


theorem Filter.EventuallyEq.hasDerivWithinAt_iff (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) :
    HasDerivWithinAt f₁ f' s x ↔ HasDerivWithinAt f f' s x :=
  ⟨fun h' ↦ h'.congr_of_eventuallyEq h₁.symm hx.symm, fun h' ↦ h'.congr_of_eventuallyEq h₁ hx⟩


theorem HasDerivWithinAt.congr_of_eventuallyEq_of_mem (h : HasDerivWithinAt f f' s x)
    (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : x ∈ s) : HasDerivWithinAt f₁ f' s x :=
  h.congr_of_eventuallyEq h₁ (h₁.eq_of_nhdsWithin hx)


theorem Filter.EventuallyEq.hasDerivWithinAt_iff_of_mem (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : x ∈ s) :
    HasDerivWithinAt f₁ f' s x ↔ HasDerivWithinAt f f' s x :=
  ⟨fun h' ↦ h'.congr_of_eventuallyEq_of_mem h₁.symm hx,
  fun h' ↦ h'.congr_of_eventuallyEq_of_mem h₁ hx⟩


theorem HasStrictDerivAt.congr_deriv (h : HasStrictDerivAt f f' x) (h' : f' = g') :
    HasStrictDerivAt f g' x :=
  h.congr_fderiv <| congr_arg _ h'


theorem HasDerivAt.congr_deriv (h : HasDerivAt f f' x) (h' : f' = g') : HasDerivAt f g' x :=
  HasFDerivAt.congr_fderiv h <| congr_arg _ h'


theorem HasDerivWithinAt.congr_deriv (h : HasDerivWithinAt f f' s x) (h' : f' = g') :
    HasDerivWithinAt f g' s x :=
  HasFDerivWithinAt.congr_fderiv h <| congr_arg _ h'


theorem HasDerivAt.congr_of_eventuallyEq (h : HasDerivAt f f' x) (h₁ : f₁ =ᶠ[𝓝 x] f) :
    HasDerivAt f₁ f' x :=
  HasDerivAtFilter.congr_of_eventuallyEq h h₁ (mem_of_mem_nhds h₁ : _)


theorem Filter.EventuallyEq.hasDerivAt_iff (h : f₀ =ᶠ[𝓝 x] f₁) :
    HasDerivAt f₀ f' x ↔ HasDerivAt f₁ f' x :=
  ⟨fun h' ↦ h'.congr_of_eventuallyEq h.symm, fun h' ↦ h'.congr_of_eventuallyEq h⟩


theorem Filter.EventuallyEq.derivWithin_eq (hs : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) :
    derivWithin f₁ s x = derivWithin f s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f f₁ : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    hs : (nhdsWithin x s).EventuallyEq f₁ f
    hx : Eq (f₁ x) (f x)
    ⊢ Eq (derivWithin f₁ s x) (derivWithin f s x)
  -/
  unfold derivWithin
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f f₁ : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    hs : (nhdsWithin x s).EventuallyEq f₁ f
    hx : Eq (f₁ x) (f x)
    ⊢ Eq ((fderivWithin 𝕜 f₁ s x) 1) ((fderivWithin 𝕜 f s x) 1)
  -/
  rw [hs.fderivWithin_eq hx]
  /-
    🎉 no goals
  -/


theorem derivWithin_congr (hs : EqOn f₁ f s) (hx : f₁ x = f x) :
    derivWithin f₁ s x = derivWithin f s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f f₁ : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    hs : Set.EqOn f₁ f s
    hx : Eq (f₁ x) (f x)
    ⊢ Eq (derivWithin f₁ s x) (derivWithin f s x)
  -/
  unfold derivWithin
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f f₁ : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    hs : Set.EqOn f₁ f s
    hx : Eq (f₁ x) (f x)
    ⊢ Eq ((fderivWithin 𝕜 f₁ s x) 1) ((fderivWithin 𝕜 f s x) 1)
  -/
  rw [fderivWithin_congr hs hx]
  /-
    🎉 no goals
  -/


theorem Filter.EventuallyEq.deriv_eq (hL : f₁ =ᶠ[𝓝 x] f) : deriv f₁ x = deriv f x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f f₁ : 𝕜 → F
    x : 𝕜
    hL : (nhds x).EventuallyEq f₁ f
    ⊢ Eq (deriv f₁ x) (deriv f x)
  -/
  unfold deriv
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f f₁ : 𝕜 → F
    x : 𝕜
    hL : (nhds x).EventuallyEq f₁ f
    ⊢ Eq ((fderiv 𝕜 f₁ x) 1) ((fderiv 𝕜 f x) 1)
  -/
  rwa [Filter.EventuallyEq.fderiv_eq]
  /-
    🎉 no goals
  -/


protected theorem Filter.EventuallyEq.deriv (h : f₁ =ᶠ[𝓝 x] f) : deriv f₁ =ᶠ[𝓝 x] deriv f :=
  h.eventuallyEq_nhds.mono fun _ h => h.deriv_eq


theorem hasDerivAtFilter_id : HasDerivAtFilter id 1 x L :=
  (hasFDerivAtFilter_id x L).hasDerivAtFilter


theorem hasDerivWithinAt_id : HasDerivWithinAt id 1 s x :=
  hasDerivAtFilter_id _ _


theorem hasDerivAt_id : HasDerivAt id 1 x :=
  hasDerivAtFilter_id _ _


theorem hasDerivAt_id' : HasDerivAt (fun x : 𝕜 => x) 1 x :=
  hasDerivAtFilter_id _ _


theorem hasStrictDerivAt_id : HasStrictDerivAt id 1 x :=
  (hasStrictFDerivAt_id x).hasStrictDerivAt


theorem deriv_id : deriv id x = 1 :=
  HasDerivAt.deriv (hasDerivAt_id x)


@[simp]
theorem deriv_id' : deriv (@id 𝕜) = fun _ => 1 :=
  funext deriv_id


/-- Variant with `fun x => x` rather than `id` -/
@[simp]
theorem deriv_id'' : (deriv fun x : 𝕜 => x) = fun _ => 1 :=
  deriv_id'


theorem derivWithin_id (hxs : UniqueDiffWithinAt 𝕜 s x) : derivWithin id s x = 1 :=
  (hasDerivWithinAt_id x s).derivWithin hxs


/-- Variant with `fun x => x` rather than `id` -/
theorem derivWithin_id' (hxs : UniqueDiffWithinAt 𝕜 s x) : derivWithin (fun x => x) s x = 1 :=
  derivWithin_id x s hxs


theorem hasDerivAtFilter_const : HasDerivAtFilter (fun _ => c) 0 x L :=
  (hasFDerivAtFilter_const c x L).hasDerivAtFilter


theorem hasStrictDerivAt_const : HasStrictDerivAt (fun _ => c) 0 x :=
  (hasStrictFDerivAt_const c x).hasStrictDerivAt


theorem hasDerivWithinAt_const : HasDerivWithinAt (fun _ => c) 0 s x :=
  hasDerivAtFilter_const _ _ _


theorem hasDerivAt_const : HasDerivAt (fun _ => c) 0 x :=
  hasDerivAtFilter_const _ _ _


theorem deriv_const : deriv (fun _ => c) x = 0 :=
  HasDerivAt.deriv (hasDerivAt_const x c)


@[simp]
theorem deriv_const' : (deriv fun _ : 𝕜 => c) = fun _ => 0 :=
  funext fun x => deriv_const x c


@[simp]
theorem derivWithin_const : derivWithin (fun _ => c) s = 0 := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set 𝕜
    c : F
    ⊢ Eq (derivWithin (fun x => c) s) 0
  -/
  ext; simp [derivWithin]
       /-
         🎉 no goals
       -/


nonrec theorem HasDerivAtFilter.tendsto_nhds (hL : L ≤ 𝓝 x) (h : HasDerivAtFilter f f' x L) :
    Tendsto f L (𝓝 (f x)) :=
  h.tendsto_nhds hL


theorem HasDerivWithinAt.continuousWithinAt (h : HasDerivWithinAt f f' s x) :
    ContinuousWithinAt f s x :=
  HasDerivAtFilter.tendsto_nhds inf_le_left h


theorem HasDerivAt.continuousAt (h : HasDerivAt f f' x) : ContinuousAt f x :=
  HasDerivAtFilter.tendsto_nhds le_rfl h


protected theorem HasDerivAt.continuousOn {f f' : 𝕜 → F} (hderiv : ∀ x ∈ s, HasDerivAt f (f' x) x) :
    ContinuousOn f s := fun x hx => (hderiv x hx).continuousAt.continuousWithinAt


/-- Converse to the mean value inequality: if `f` is differentiable at `x₀` and `C`-lipschitz
on a neighborhood of `x₀` then its derivative at `x₀` has norm bounded by `C`. This version
only assumes that `‖f x - f x₀‖ ≤ C * ‖x - x₀‖` in a neighborhood of `x`. -/
theorem HasDerivAt.le_of_lip' {f : 𝕜 → F} {f' : F} {x₀ : 𝕜} (hf : HasDerivAt f f' x₀)
    {C : ℝ} (hC₀ : 0 ≤ C) (hlip : ∀ᶠ x in 𝓝 x₀, ‖f x - f x₀‖ ≤ C * ‖x - x₀‖) :
    ‖f'‖ ≤ C := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x₀ : 𝕜
    hf : HasDerivAt f f' x₀
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    ⊢ LE.le (Norm.norm f') C
  -/
  simpa using HasFDerivAt.le_of_lip' hf.hasFDerivAt hC₀ hlip
  /-
    🎉 no goals
  -/


/-- Converse to the mean value inequality: if `f` is differentiable at `x₀` and `C`-lipschitz
on a neighborhood of `x₀` then its derivative at `x₀` has norm bounded by `C`. -/
theorem HasDerivAt.le_of_lipschitzOn {f : 𝕜 → F} {f' : F} {x₀ : 𝕜} (hf : HasDerivAt f f' x₀)
    {s : Set 𝕜} (hs : s ∈ 𝓝 x₀) {C : ℝ≥0} (hlip : LipschitzOnWith C f s) : ‖f'‖ ≤ C := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x₀ : 𝕜
    hf : HasDerivAt f f' x₀
    s : Set 𝕜
    hs : Membership.mem (nhds x₀) s
    C : NNReal
    hlip : LipschitzOnWith C f s
    ⊢ LE.le (Norm.norm f') ↑C
  -/
  simpa using HasFDerivAt.le_of_lipschitzOn hf.hasFDerivAt hs hlip
  /-
    🎉 no goals
  -/


/-- Converse to the mean value inequality: if `f` is differentiable at `x₀` and `C`-lipschitz
then its derivative at `x₀` has norm bounded by `C`. -/
theorem HasDerivAt.le_of_lipschitz {f : 𝕜 → F} {f' : F} {x₀ : 𝕜} (hf : HasDerivAt f f' x₀)
    {C : ℝ≥0} (hlip : LipschitzWith C f) : ‖f'‖ ≤ C := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x₀ : 𝕜
    hf : HasDerivAt f f' x₀
    C : NNReal
    hlip : LipschitzWith C f
    ⊢ LE.le (Norm.norm f') ↑C
  -/
  simpa using HasFDerivAt.le_of_lipschitz hf.hasFDerivAt hlip
  /-
    🎉 no goals
  -/


/-- Converse to the mean value inequality: if `f` is `C`-lipschitz
on a neighborhood of `x₀` then its derivative at `x₀` has norm bounded by `C`. This version
only assumes that `‖f x - f x₀‖ ≤ C * ‖x - x₀‖` in a neighborhood of `x`. -/
theorem norm_deriv_le_of_lip' {f : 𝕜 → F} {x₀ : 𝕜}
    {C : ℝ} (hC₀ : 0 ≤ C) (hlip : ∀ᶠ x in 𝓝 x₀, ‖f x - f x₀‖ ≤ C * ‖x - x₀‖) :
    ‖deriv f x₀‖ ≤ C := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x₀ : 𝕜
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    ⊢ LE.le (Norm.norm (deriv f x₀)) C
  -/
  simpa [norm_deriv_eq_norm_fderiv] using norm_fderiv_le_of_lip' 𝕜 hC₀ hlip
  /-
    🎉 no goals
  -/


/-- Converse to the mean value inequality: if `f` is `C`-lipschitz
on a neighborhood of `x₀` then its derivative at `x₀` has norm bounded by `C`.
Version using `deriv`. -/
theorem norm_deriv_le_of_lipschitzOn {f : 𝕜 → F} {x₀ : 𝕜} {s : Set 𝕜} (hs : s ∈ 𝓝 x₀)
    {C : ℝ≥0} (hlip : LipschitzOnWith C f s) : ‖deriv f x₀‖ ≤ C := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x₀ : 𝕜
    s : Set 𝕜
    hs : Membership.mem (nhds x₀) s
    C : NNReal
    hlip : LipschitzOnWith C f s
    ⊢ LE.le (Norm.norm (deriv f x₀)) ↑C
  -/
  simpa [norm_deriv_eq_norm_fderiv] using norm_fderiv_le_of_lipschitzOn 𝕜 hs hlip
  /-
    🎉 no goals
  -/


/-- Converse to the mean value inequality: if `f` is `C`-lipschitz then
its derivative at `x₀` has norm bounded by `C`.
Version using `deriv`. -/
theorem norm_deriv_le_of_lipschitz {f : 𝕜 → F} {x₀ : 𝕜}
    {C : ℝ≥0} (hlip : LipschitzWith C f) : ‖deriv f x₀‖ ≤ C := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x₀ : 𝕜
    C : NNReal
    hlip : LipschitzWith C f
    ⊢ LE.le (Norm.norm (deriv f x₀)) ↑C
  -/
  simpa [norm_deriv_eq_norm_fderiv] using norm_fderiv_le_of_lipschitz 𝕜 hlip
  /-
    🎉 no goals
  -/


