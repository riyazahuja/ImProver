/-- A function `f` has the continuous linear map `f'` as derivative along the filter `L` if
`f x' = f x + f' (x' - x) + o (x' - x)` when `x'` converges along the filter `L`. This definition
is designed to be specialized for `L = 𝓝 x` (in `HasFDerivAt`), giving rise to the usual notion
of Fréchet derivative, and for `L = 𝓝[s] x` (in `HasFDerivWithinAt`), giving rise to
the notion of Fréchet derivative along the set `s`. -/
@[mk_iff hasFDerivAtFilter_iff_isLittleOTVS]
structure HasFDerivAtFilter (f : E → F) (f' : E →L[𝕜] F) (x : E) (L : Filter E) : Prop where
  of_isLittleOTVS ::
    isLittleOTVS : (fun x' => f x' - f x - f' (x' - x)) =o[𝕜;L] (fun x' => x' - x)


/-- A function `f` has the continuous linear map `f'` as derivative at `x` within a set `s` if
`f x' = f x + f' (x' - x) + o (x' - x)` when `x'` tends to `x` inside `s`. -/
@[fun_prop]
def HasFDerivWithinAt (f : E → F) (f' : E →L[𝕜] F) (s : Set E) (x : E) :=
  HasFDerivAtFilter f f' x (𝓝[s] x)


/-- A function `f` has the continuous linear map `f'` as derivative at `x` if
`f x' = f x + f' (x' - x) + o (x' - x)` when `x'` tends to `x`. -/
@[fun_prop]
def HasFDerivAt (f : E → F) (f' : E →L[𝕜] F) (x : E) :=
  HasFDerivAtFilter f f' x (𝓝 x)


/-- A function `f` has derivative `f'` at `a` in the sense of *strict differentiability*
if `f x - f y - f' (x - y) = o(x - y)` as `x, y → a`. This form of differentiability is required,
e.g., by the inverse function theorem. Any `C^1` function on a vector space over `ℝ` is strictly
differentiable but this definition works, e.g., for vector spaces over `p`-adic numbers. -/
@[fun_prop, mk_iff hasStrictFDerivAt_iff_isLittleOTVS]
structure HasStrictFDerivAt (f : E → F) (f' : E →L[𝕜] F) (x : E) where
  of_isLittleOTVS ::
    isLittleOTVS :
      (fun p : E × E => f p.1 - f p.2 - f' (p.1 - p.2))
        =o[𝕜;𝓝 (x, x)] (fun p : E × E => p.1 - p.2)


/-- A function `f` is differentiable at a point `x` within a set `s` if it admits a derivative
there (possibly non-unique). -/
@[fun_prop]
def DifferentiableWithinAt (f : E → F) (s : Set E) (x : E) :=
  ∃ f' : E →L[𝕜] F, HasFDerivWithinAt f f' s x


/-- A function `f` is differentiable at a point `x` if it admits a derivative there (possibly
non-unique). -/
@[fun_prop]
def DifferentiableAt (f : E → F) (x : E) :=
  ∃ f' : E →L[𝕜] F, HasFDerivAt f f' x


open scoped Classical in
/-- If `f` has a derivative at `x` within `s`, then `fderivWithin 𝕜 f s x` is such a derivative.
Otherwise, it is set to `0`. We also set it to be zero, if zero is one of possible derivatives. -/
irreducible_def fderivWithin (f : E → F) (s : Set E) (x : E) : E →L[𝕜] F :=
  if HasFDerivWithinAt f (0 : E →L[𝕜] F) s x
    then 0
  else if h : DifferentiableWithinAt 𝕜 f s x
    then Classical.choose h
  else 0


/-- If `f` has a derivative at `x`, then `fderiv 𝕜 f x` is such a derivative. Otherwise, it is
set to `0`. -/
irreducible_def fderiv (f : E → F) (x : E) : E →L[𝕜] F :=
  fderivWithin 𝕜 f univ x


/-- `DifferentiableOn 𝕜 f s` means that `f` is differentiable within `s` at any point of `s`. -/
@[fun_prop]
def DifferentiableOn (f : E → F) (s : Set E) :=
  ∀ x ∈ s, DifferentiableWithinAt 𝕜 f s x


/-- `Differentiable 𝕜 f` means that `f` is differentiable at any point. -/
@[fun_prop]
def Differentiable (f : E → F) :=
  ∀ x, DifferentiableAt 𝕜 f x


theorem fderivWithin_zero_of_not_differentiableWithinAt (h : ¬DifferentiableWithinAt 𝕜 f s x) :
    fderivWithin 𝕜 f s x = 0 := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : TopologicalSpace E
    F : Type u_3
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 F
    inst✝ : TopologicalSpace F
    f : E → F
    x : E
    s : Set E
    h : Not (DifferentiableWithinAt 𝕜 f s x)
    ⊢ Eq (fderivWithin 𝕜 f s x) 0
  -/
  simp [fderivWithin, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem fderivWithin_univ : fderivWithin 𝕜 f univ = fderiv 𝕜 f := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : TopologicalSpace E
    F : Type u_3
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 F
    inst✝ : TopologicalSpace F
    f : E → F
    ⊢ Eq (fderivWithin 𝕜 f Set.univ) (fderiv 𝕜 f)
  -/
  ext
  /-
    case h.h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : TopologicalSpace E
    F : Type u_3
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 F
    inst✝ : TopologicalSpace F
    f : E → F
    x✝¹ x✝ : E
    ⊢ Eq ((fderivWithin 𝕜 f Set.univ x✝¹) x✝) ((fderiv 𝕜 f x✝¹) x✝)
  -/
  rw [fderiv]
  /-
    🎉 no goals
  -/


theorem hasFDerivAtFilter_iff_isLittleO :
    HasFDerivAtFilter f f' x L ↔ (fun x' => f x' - f x - f' (x' - x)) =o[L] fun x' => x' - x :=
  (hasFDerivAtFilter_iff_isLittleOTVS ..).trans isLittleOTVS_iff_isLittleO


alias ⟨HasFDerivAtFilter.isLittleO, HasFDerivAtFilter.of_isLittleO⟩ :=
  hasFDerivAtFilter_iff_isLittleO


theorem hasStrictFDerivAt_iff_isLittleO :
    HasStrictFDerivAt f f' x ↔
      (fun p : E × E => f p.1 - f p.2 - f' (p.1 - p.2)) =o[𝓝 (x, x)] fun p : E × E => p.1 - p.2 :=
  (hasStrictFDerivAt_iff_isLittleOTVS ..).trans isLittleOTVS_iff_isLittleO


alias ⟨HasStrictFDerivAt.isLittleO, HasStrictFDerivAt.of_isLittleO⟩ :=
  hasStrictFDerivAt_iff_isLittleO


/-- If a function f has a derivative f' at x, a rescaled version of f around x converges to f',
i.e., `n (f (x + (1/n) v) - f x)` converges to `f' v`. More generally, if `c n` tends to infinity
and `c n * d n` tends to `v`, then `c n * (f (x + d n) - f x)` tends to `f' v`. This lemma expresses
this fact, for functions having a derivative within a set. Its specific formulation is useful for
tangent cone related discussions. -/
theorem HasFDerivWithinAt.lim (h : HasFDerivWithinAt f f' s x) {α : Type*} (l : Filter α)
    {c : α → 𝕜} {d : α → E} {v : E} (dtop : ∀ᶠ n in l, x + d n ∈ s)
    (clim : Tendsto (fun n => ‖c n‖) l atTop) (cdlim : Tendsto (fun n => c n • d n) l (𝓝 v)) :
    Tendsto (fun n => c n • (f (x + d n) - f x)) l (𝓝 (f' v)) := by
  have tendsto_arg : Tendsto (fun n => x + d n) l (𝓝[s] x) := by
    conv in 𝓝[s] x => rw [← add_zero x]
    rw [nhdsWithin, tendsto_inf]
    constructor
    · apply tendsto_const_nhds.add (tangentConeAt.lim_zero l clim cdlim)
    · rwa [tendsto_principal]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s : Set E
    h : HasFDerivWithinAt f f' s x
    α : Type u_4
    l : Filter α
    c : α → 𝕜
    d : α → E
    v : E
    dtop : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (d n))) l
    clim : Filter.Tendsto (fun n => Norm.norm (c n)) l Filter.atTop
    cdlim : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) l (nhds v)
    tendsto_arg : Filter.Tendsto (fun n => HAdd.hAdd x (d n)) l (nhdsWithin x s)
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (c n) (HSub.hSub (f (HAdd.hAdd x (d n)) …
  -/
  have : (fun y => f y - f x - f' (y - x)) =o[𝓝[s] x] fun y => y - x := h.isLittleO
  have : (fun n => f (x + d n) - f x - f' (x + d n - x)) =o[l] fun n => x + d n - x :=
    this.comp_tendsto tendsto_arg
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s : Set E
    h : HasFDerivWithinAt f f' s x
    α : Type u_4
    l : Filter α
    c : α → 𝕜
    d : α → E
    v : E
    dtop : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (d n))) l
    clim : Filter.Tendsto (fun n => Norm.norm (c n)) l Filter.atTop
    cdlim : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) l (nhds v)
    tendsto_arg : Filter.Tendsto (fun n => HAdd.hAdd x (d n)) l (nhdsWithin x s)
    this✝ : Asymptotics.IsLittleO (nhdsWithin x s) (fun y => HSub.hSub (HSub.hSub  …
    this : Asymptotics.IsLittleO l (fun n => HSub.hSub (HSub.hSub (f (HAdd.hAdd x  …
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (c n) (HSub.hSub (f (HAdd.hAdd x (d n)) …
  -/
  have : (fun n => f (x + d n) - f x - f' (d n)) =o[l] d := by simpa only [add_sub_cancel_left]
  have : (fun n => c n • (f (x + d n) - f x - f' (d n))) =o[l] fun n => c n • d n :=
    (isBigO_refl c l).smul_isLittleO this
  have : (fun n => c n • (f (x + d n) - f x - f' (d n))) =o[l] fun _ => (1 : ℝ) :=
    this.trans_isBigO (cdlim.isBigO_one ℝ)
  have L1 : Tendsto (fun n => c n • (f (x + d n) - f x - f' (d n))) l (𝓝 0) :=
    (isLittleO_one_iff ℝ).1 this
  have L2 : Tendsto (fun n => f' (c n • d n)) l (𝓝 (f' v)) :=
    Tendsto.comp f'.cont.continuousAt cdlim
  have L3 :
    Tendsto (fun n => c n • (f (x + d n) - f x - f' (d n)) + f' (c n • d n)) l (𝓝 (0 + f' v)) :=
    L1.add L2
  have :
    (fun n => c n • (f (x + d n) - f x - f' (d n)) + f' (c n • d n)) = fun n =>
      c n • (f (x + d n) - f x) := by
    ext n
    simp [smul_add, smul_sub]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s : Set E
    h : HasFDerivWithinAt f f' s x
    α : Type u_4
    l : Filter α
    c : α → 𝕜
    d : α → E
    v : E
    dtop : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (d n))) l
    clim : Filter.Tendsto (fun n => Norm.norm (c n)) l Filter.atTop
    cdlim : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) l (nhds v)
    tendsto_arg : Filter.Tendsto (fun n => HAdd.hAdd x (d n)) l (nhdsWithin x s)
    this✝⁴ : Asymptotics.IsLittleO (nhdsWithin x s) (fun y => HSub.hSub (HSub.hSub …
    this✝³ : Asymptotics.IsLittleO l (fun n => HSub.hSub (HSub.hSub (f (HAdd.hAdd  …
    this✝² : Asymptotics.IsLittleO l (fun n => HSub.hSub (HSub.hSub (f (HAdd.hAdd  …
    this✝¹ : Asymptotics.IsLittleO l (fun n => HSMul.hSMul (c n) (HSub.hSub (HSub. …
    this✝ : Asymptotics.IsLittleO l (fun n => HSMul.hSMul (c n) (HSub.hSub (HSub.h …
    L1 : Filter.Tendsto (fun n => HSMul.hSMul (c n) (HSub.hSub (HSub.hSub (f (HAdd …
    L2 : Filter.Tendsto (fun n => f' (HSMul.hSMul (c n) (d n))) l (nhds (f' v))
    L3 : Filter.Tendsto (fun n => HAdd.hAdd (HSMul.hSMul (c n) (HSub.hSub (HSub.hS …
    this : Eq (fun n => HAdd.hAdd (HSMul.hSMul (c n) (HSub.hSub (HSub.hSub (f (HAd …
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (c n) (HSub.hSub (f (HAdd.hAdd x (d n)) …
  -/
  rwa [this, zero_add] at L3
  /-
    🎉 no goals
  -/


/-- If `f'` and `f₁'` are two derivatives of `f` within `s` at `x`, then they are equal on the
tangent cone to `s` at `x` -/
theorem HasFDerivWithinAt.unique_on (hf : HasFDerivWithinAt f f' s x)
    (hg : HasFDerivWithinAt f f₁' s x) : EqOn f' f₁' (tangentConeAt 𝕜 s x) :=
  fun _ ⟨_, _, dtop, clim, cdlim⟩ =>
  tendsto_nhds_unique (hf.lim atTop dtop clim cdlim) (hg.lim atTop dtop clim cdlim)


/-- `UniqueDiffWithinAt` achieves its goal: it implies the uniqueness of the derivative. -/
theorem UniqueDiffWithinAt.eq (H : UniqueDiffWithinAt 𝕜 s x) (hf : HasFDerivWithinAt f f' s x)
    (hg : HasFDerivWithinAt f f₁' s x) : f' = f₁' :=
  ContinuousLinearMap.ext_on H.1 (hf.unique_on hg)


theorem UniqueDiffOn.eq (H : UniqueDiffOn 𝕜 s) (hx : x ∈ s) (h : HasFDerivWithinAt f f' s x)
    (h₁ : HasFDerivWithinAt f f₁' s x) : f' = f₁' :=
  (H x hx).eq h h₁


theorem hasFDerivAtFilter_iff_tendsto :
    HasFDerivAtFilter f f' x L ↔
      Tendsto (fun x' => ‖x' - x‖⁻¹ * ‖f x' - f x - f' (x' - x)‖) L (𝓝 0) := by
  have h : ∀ x', ‖x' - x‖ = 0 → ‖f x' - f x - f' (x' - x)‖ = 0 := fun x' hx' => by
    rw [sub_eq_zero.1 (norm_eq_zero.1 hx')]
    simp
  rw [hasFDerivAtFilter_iff_isLittleO, ← isLittleO_norm_left, ← isLittleO_norm_right,
    isLittleO_iff_tendsto h]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    L : Filter E
    h : ∀ (x' : E), Eq (Norm.norm (HSub.hSub x' x)) 0 → Eq (Norm.norm (HSub.hSub ( …
    ⊢ Iff (Filter.Tendsto (fun x_1 => HDiv.hDiv (Norm.norm (HSub.hSub (HSub.hSub ( …
  -/
  exact tendsto_congr fun _ => div_eq_inv_mul _ _
  /-
    🎉 no goals
  -/


theorem hasFDerivWithinAt_iff_tendsto :
    HasFDerivWithinAt f f' s x ↔
      Tendsto (fun x' => ‖x' - x‖⁻¹ * ‖f x' - f x - f' (x' - x)‖) (𝓝[s] x) (𝓝 0) :=
  hasFDerivAtFilter_iff_tendsto


theorem hasFDerivAt_iff_tendsto :
    HasFDerivAt f f' x ↔ Tendsto (fun x' => ‖x' - x‖⁻¹ * ‖f x' - f x - f' (x' - x)‖) (𝓝 x) (𝓝 0) :=
  hasFDerivAtFilter_iff_tendsto


theorem hasFDerivAt_iff_isLittleO_nhds_zero :
    HasFDerivAt f f' x ↔ (fun h : E => f (x + h) - f x - f' h) =o[𝓝 0] fun h => h := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    ⊢ Iff (HasFDerivAt f f' x) (Asymptotics.IsLittleO (nhds 0) (fun h => HSub.hSub …
  -/
  rw [HasFDerivAt, hasFDerivAtFilter_iff_isLittleO, ← map_add_left_nhds_zero x, isLittleO_map]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    ⊢ Iff (Asymptotics.IsLittleO (nhds 0) (Function.comp (fun x' => HSub.hSub (HSu …
  -/
  simp [Function.comp_def]
  /-
    🎉 no goals
  -/


nonrec theorem HasFDerivAtFilter.mono (h : HasFDerivAtFilter f f' x L₂) (hst : L₁ ≤ L₂) :
    HasFDerivAtFilter f f' x L₁ :=
  .of_isLittleO <| h.isLittleO.mono hst


theorem HasFDerivWithinAt.mono_of_mem_nhdsWithin
    (h : HasFDerivWithinAt f f' t x) (hst : t ∈ 𝓝[s] x) :
    HasFDerivWithinAt f f' s x :=
  h.mono <| nhdsWithin_le_iff.mpr hst


@[deprecated (since := "2024-10-31")]
alias HasFDerivWithinAt.mono_of_mem := HasFDerivWithinAt.mono_of_mem_nhdsWithin


nonrec theorem HasFDerivWithinAt.mono (h : HasFDerivWithinAt f f' t x) (hst : s ⊆ t) :
    HasFDerivWithinAt f f' s x :=
  h.mono <| nhdsWithin_mono _ hst


theorem HasFDerivAt.hasFDerivAtFilter (h : HasFDerivAt f f' x) (hL : L ≤ 𝓝 x) :
    HasFDerivAtFilter f f' x L :=
  h.mono hL


@[fun_prop]
theorem HasFDerivAt.hasFDerivWithinAt (h : HasFDerivAt f f' x) : HasFDerivWithinAt f f' s x :=
  h.hasFDerivAtFilter inf_le_left


@[fun_prop]
theorem HasFDerivWithinAt.differentiableWithinAt (h : HasFDerivWithinAt f f' s x) :
    DifferentiableWithinAt 𝕜 f s x :=
  ⟨f', h⟩


@[fun_prop]
theorem HasFDerivAt.differentiableAt (h : HasFDerivAt f f' x) : DifferentiableAt 𝕜 f x :=
  ⟨f', h⟩


@[simp]
theorem hasFDerivWithinAt_univ : HasFDerivWithinAt f f' univ x ↔ HasFDerivAt f f' x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    ⊢ Iff (HasFDerivWithinAt f f' Set.univ x) (HasFDerivAt f f' x)
  -/
  simp only [HasFDerivWithinAt, nhdsWithin_univ, HasFDerivAt]
  /-
    🎉 no goals
  -/


alias ⟨HasFDerivWithinAt.hasFDerivAt_of_univ, _⟩ := hasFDerivWithinAt_univ


theorem differentiableWithinAt_univ :
    DifferentiableWithinAt 𝕜 f univ x ↔ DifferentiableAt 𝕜 f x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    ⊢ Iff (DifferentiableWithinAt 𝕜 f Set.univ x) (DifferentiableAt 𝕜 f x)
  -/
  simp only [DifferentiableWithinAt, hasFDerivWithinAt_univ, DifferentiableAt]
  /-
    🎉 no goals
  -/


theorem fderiv_zero_of_not_differentiableAt (h : ¬DifferentiableAt 𝕜 f x) : fderiv 𝕜 f x = 0 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    h : Not (DifferentiableAt 𝕜 f x)
    ⊢ Eq (fderiv 𝕜 f x) 0
  -/
  rw [fderiv, fderivWithin_zero_of_not_differentiableWithinAt]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    h : Not (DifferentiableAt 𝕜 f x)
    ⊢ Not (DifferentiableWithinAt 𝕜 f Set.univ x)
  -/
  rwa [differentiableWithinAt_univ]
  /-
    🎉 no goals
  -/


theorem hasFDerivWithinAt_of_mem_nhds (h : s ∈ 𝓝 x) :
    HasFDerivWithinAt f f' s x ↔ HasFDerivAt f f' x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s : Set E
    h : Membership.mem (nhds x) s
    ⊢ Iff (HasFDerivWithinAt f f' s x) (HasFDerivAt f f' x)
  -/
  rw [HasFDerivAt, HasFDerivWithinAt, nhdsWithin_eq_nhds.mpr h]
  /-
    🎉 no goals
  -/


lemma hasFDerivWithinAt_of_isOpen (h : IsOpen s) (hx : x ∈ s) :
    HasFDerivWithinAt f f' s x ↔ HasFDerivAt f f' x :=
  hasFDerivWithinAt_of_mem_nhds (h.mem_nhds hx)


theorem hasFDerivWithinAt_insert {y : E} :
    HasFDerivWithinAt f f' (insert y s) x ↔ HasFDerivWithinAt f f' s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s : Set E
    y : E
    ⊢ Iff (HasFDerivWithinAt f f' (Insert.insert y s) x) (HasFDerivWithinAt f f' s …
  -/
  rcases eq_or_ne x y with (rfl | h)
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : E
      s : Set E
      ⊢ Iff (HasFDerivWithinAt f f' (Insert.insert x s) x) (HasFDerivWithinAt f f' s …
    -/
  · simp_rw [HasFDerivWithinAt, hasFDerivAtFilter_iff_isLittleO]
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : E
      s : Set E
      ⊢ Iff (Asymptotics.IsLittleO (nhdsWithin x (Insert.insert x s)) (fun x' => HSu …
    -/
    apply Asymptotics.isLittleO_insert
    /-
      case inl.h
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : E
      s : Set E
      ⊢ Eq (HSub.hSub (HSub.hSub (f x) (f x)) (f' (HSub.hSub x x))) 0
    -/
    simp only [sub_self, map_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s : Set E
    y : E
    h : Ne x y
    ⊢ Iff (HasFDerivWithinAt f f' (Insert.insert y s) x) (HasFDerivWithinAt f f' s …
  -/
  refine ⟨fun h => h.mono <| subset_insert y s, fun hf => hf.mono_of_mem_nhdsWithin ?_⟩
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s : Set E
    y : E
    h : Ne x y
    hf : HasFDerivWithinAt f f' s x
    ⊢ Membership.mem (nhdsWithin x (Insert.insert y s)) s
  -/
  simp_rw [nhdsWithin_insert_of_ne h, self_mem_nhdsWithin]
  /-
    🎉 no goals
  -/


alias ⟨HasFDerivWithinAt.of_insert, HasFDerivWithinAt.insert'⟩ := hasFDerivWithinAt_insert


protected theorem HasFDerivWithinAt.insert (h : HasFDerivWithinAt g g' s x) :
    HasFDerivWithinAt g g' (insert x s) x :=
  h.insert'


theorem hasFDerivWithinAt_diff_singleton (y : E) :
    HasFDerivWithinAt f f' (s \ {y}) x ↔ HasFDerivWithinAt f f' s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s : Set E
    y : E
    ⊢ Iff (HasFDerivWithinAt f f' (SDiff.sdiff s (Singleton.singleton y)) x) (HasF …
  -/
  rw [← hasFDerivWithinAt_insert, insert_diff_singleton, hasFDerivWithinAt_insert]
  /-
    🎉 no goals
  -/


theorem HasStrictFDerivAt.isBigO_sub (hf : HasStrictFDerivAt f f' x) :
    (fun p : E × E => f p.1 - f p.2) =O[𝓝 (x, x)] fun p : E × E => p.1 - p.2 :=
  hf.isLittleO.isBigO.congr_of_sub.2 (f'.isBigO_comp _ _)


theorem HasFDerivAtFilter.isBigO_sub (h : HasFDerivAtFilter f f' x L) :
    (fun x' => f x' - f x) =O[L] fun x' => x' - x :=
  h.isLittleO.isBigO.congr_of_sub.2 (f'.isBigO_sub _ _)


@[fun_prop]
protected theorem HasStrictFDerivAt.hasFDerivAt (hf : HasStrictFDerivAt f f' x) :
    HasFDerivAt f f' x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    hf : HasStrictFDerivAt f f' x
    ⊢ HasFDerivAt f f' x
  -/
  rw [HasFDerivAt, hasFDerivAtFilter_iff_isLittleO, isLittleO_iff]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    hf : HasStrictFDerivAt f f' x
    ⊢ ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x_1 => LE.le (Norm.norm (HS …
  -/
  exact fun c hc => tendsto_id.prod_mk_nhds tendsto_const_nhds (isLittleO_iff.1 hf.isLittleO hc)
  /-
    🎉 no goals
  -/


protected theorem HasStrictFDerivAt.differentiableAt (hf : HasStrictFDerivAt f f' x) :
    DifferentiableAt 𝕜 f x :=
  hf.hasFDerivAt.differentiableAt


/-- If `f` is strictly differentiable at `x` with derivative `f'` and `K > ‖f'‖₊`, then `f` is
`K`-Lipschitz in a neighborhood of `x`. -/
theorem HasStrictFDerivAt.exists_lipschitzOnWith_of_nnnorm_lt (hf : HasStrictFDerivAt f f' x)
    (K : ℝ≥0) (hK : ‖f'‖₊ < K) : ∃ s ∈ 𝓝 x, LipschitzOnWith K f s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    hf : HasStrictFDerivAt f f' x
    K : NNReal
    hK : LT.lt (NNNorm.nnnorm f') K
    ⊢ Exists fun s => And (Membership.mem (nhds x) s) (LipschitzOnWith K f s)
  -/
  have := hf.isLittleO.add_isBigOWith (f'.isBigOWith_comp _ _) hK
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    hf : HasStrictFDerivAt f f' x
    K : NNReal
    hK : LT.lt (NNNorm.nnnorm f') K
    this : Asymptotics.IsBigOWith ((fun a => ↑a) K) (nhds { fst := x, snd := x })  …
    ⊢ Exists fun s => And (Membership.mem (nhds x) s) (LipschitzOnWith K f s)
  -/
  simp only [sub_add_cancel, IsBigOWith] at this
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    hf : HasStrictFDerivAt f f' x
    K : NNReal
    hK : LT.lt (NNNorm.nnnorm f') K
    this : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x.1) (f x.2) …
    ⊢ Exists fun s => And (Membership.mem (nhds x) s) (LipschitzOnWith K f s)
  -/
  rcases exists_nhds_square this with ⟨U, Uo, xU, hU⟩
  exact
    ⟨U, Uo.mem_nhds xU, lipschitzOnWith_iff_norm_sub_le.2 fun x hx y hy => hU (mk_mem_prod hx hy)⟩


/-- If `f` is strictly differentiable at `x` with derivative `f'`, then `f` is Lipschitz in a
neighborhood of `x`. See also `HasStrictFDerivAt.exists_lipschitzOnWith_of_nnnorm_lt` for a
more precise statement. -/
theorem HasStrictFDerivAt.exists_lipschitzOnWith (hf : HasStrictFDerivAt f f' x) :
    ∃ K, ∃ s ∈ 𝓝 x, LipschitzOnWith K f s :=
  (exists_gt _).imp hf.exists_lipschitzOnWith_of_nnnorm_lt


/-- Directional derivative agrees with `HasFDeriv`. -/
theorem HasFDerivAt.lim (hf : HasFDerivAt f f' x) (v : E) {α : Type*} {c : α → 𝕜} {l : Filter α}
    (hc : Tendsto (fun n => ‖c n‖) l atTop) :
    Tendsto (fun n => c n • (f (x + (c n)⁻¹ • v) - f x)) l (𝓝 (f' v)) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    hf : HasFDerivAt f f' x
    v : E
    α : Type u_4
    c : α → 𝕜
    l : Filter α
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) l Filter.atTop
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (c n) (HSub.hSub (f (HAdd.hAdd x (HSMul …
  -/
  refine (hasFDerivWithinAt_univ.2 hf).lim _ univ_mem hc ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    hf : HasFDerivAt f f' x
    v : E
    α : Type u_4
    c : α → 𝕜
    l : Filter α
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) l Filter.atTop
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (c n) (HSMul.hSMul (Inv.inv (c n)) v))  …
  -/
  intro U hU
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    hf : HasFDerivAt f f' x
    v : E
    α : Type u_4
    c : α → 𝕜
    l : Filter α
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) l Filter.atTop
    U : Set E
    hU : Membership.mem (nhds v) U
    ⊢ Membership.mem (Filter.map (fun n => HSMul.hSMul (c n) (HSMul.hSMul (Inv.inv …
  -/
  refine (eventually_ne_of_tendsto_norm_atTop hc (0 : 𝕜)).mono fun y hy => ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    hf : HasFDerivAt f f' x
    v : E
    α : Type u_4
    c : α → 𝕜
    l : Filter α
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) l Filter.atTop
    U : Set E
    hU : Membership.mem (nhds v) U
    y : α
    hy : Ne (c y) 0
    ⊢ Membership.mem U ((fun n => HSMul.hSMul (c n) (HSMul.hSMul (Inv.inv (c n)) v …
  -/
  convert mem_of_mem_nhds hU
  /-
    case h.e'_5
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    hf : HasFDerivAt f f' x
    v : E
    α : Type u_4
    c : α → 𝕜
    l : Filter α
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) l Filter.atTop
    U : Set E
    hU : Membership.mem (nhds v) U
    y : α
    hy : Ne (c y) 0
    ⊢ Eq ((fun n => HSMul.hSMul (c n) (HSMul.hSMul (Inv.inv (c n)) v)) y) v
  -/
  dsimp only
  /-
    case h.e'_5
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    hf : HasFDerivAt f f' x
    v : E
    α : Type u_4
    c : α → 𝕜
    l : Filter α
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) l Filter.atTop
    U : Set E
    hU : Membership.mem (nhds v) U
    y : α
    hy : Ne (c y) 0
    ⊢ Eq (HSMul.hSMul (c y) (HSMul.hSMul (Inv.inv (c y)) v)) v
  -/
  rw [← mul_smul, mul_inv_cancel₀ hy, one_smul]
  /-
    🎉 no goals
  -/


theorem HasFDerivAt.unique (h₀ : HasFDerivAt f f₀' x) (h₁ : HasFDerivAt f f₁' x) : f₀' = f₁' := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f₀' f₁' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    h₀ : HasFDerivAt f f₀' x
    h₁ : HasFDerivAt f f₁' x
    ⊢ Eq f₀' f₁'
  -/
  rw [← hasFDerivWithinAt_univ] at h₀ h₁
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f₀' f₁' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    h₀ : HasFDerivWithinAt f f₀' Set.univ x
    h₁ : HasFDerivWithinAt f f₁' Set.univ x
    ⊢ Eq f₀' f₁'
  -/
  exact uniqueDiffWithinAt_univ.eq h₀ h₁
  /-
    🎉 no goals
  -/


theorem hasFDerivWithinAt_inter' (h : t ∈ 𝓝[s] x) :
    HasFDerivWithinAt f f' (s ∩ t) x ↔ HasFDerivWithinAt f f' s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s t : Set E
    h : Membership.mem (nhdsWithin x s) t
    ⊢ Iff (HasFDerivWithinAt f f' (Inter.inter s t) x) (HasFDerivWithinAt f f' s x)
  -/
  simp [HasFDerivWithinAt, nhdsWithin_restrict'' s h]
  /-
    🎉 no goals
  -/


theorem hasFDerivWithinAt_inter (h : t ∈ 𝓝 x) :
    HasFDerivWithinAt f f' (s ∩ t) x ↔ HasFDerivWithinAt f f' s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s t : Set E
    h : Membership.mem (nhds x) t
    ⊢ Iff (HasFDerivWithinAt f f' (Inter.inter s t) x) (HasFDerivWithinAt f f' s x)
  -/
  simp [HasFDerivWithinAt, nhdsWithin_restrict' s h]
  /-
    🎉 no goals
  -/


theorem HasFDerivWithinAt.union (hs : HasFDerivWithinAt f f' s x)
    (ht : HasFDerivWithinAt f f' t x) : HasFDerivWithinAt f f' (s ∪ t) x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s t : Set E
    hs : HasFDerivWithinAt f f' s x
    ht : HasFDerivWithinAt f f' t x
    ⊢ HasFDerivWithinAt f f' (Union.union s t) x
  -/
  simp only [HasFDerivWithinAt, nhdsWithin_union]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s t : Set E
    hs : HasFDerivWithinAt f f' s x
    ht : HasFDerivWithinAt f f' t x
    ⊢ HasFDerivAtFilter f f' x (Max.max (nhdsWithin x s) (nhdsWithin x t))
  -/
  exact .of_isLittleO <| hs.isLittleO.sup ht.isLittleO
  /-
    🎉 no goals
  -/


theorem HasFDerivWithinAt.hasFDerivAt (h : HasFDerivWithinAt f f' s x) (hs : s ∈ 𝓝 x) :
    HasFDerivAt f f' x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s : Set E
    h : HasFDerivWithinAt f f' s x
    hs : Membership.mem (nhds x) s
    ⊢ HasFDerivAt f f' x
  -/
  rwa [← univ_inter s, hasFDerivWithinAt_inter hs, hasFDerivWithinAt_univ] at h
  /-
    🎉 no goals
  -/


theorem DifferentiableWithinAt.differentiableAt (h : DifferentiableWithinAt 𝕜 f s x)
    (hs : s ∈ 𝓝 x) : DifferentiableAt 𝕜 f x :=
  h.imp fun _ hf' => hf'.hasFDerivAt hs


/-- If `x` is isolated in `s`, then `f` has any derivative at `x` within `s`,
as this statement is empty. -/
theorem HasFDerivWithinAt.of_nhdsWithin_eq_bot (h : 𝓝[s\{x}] x = ⊥) :
    HasFDerivWithinAt f f' s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s : Set E
    h : Eq (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))) Bot.bot
    ⊢ HasFDerivWithinAt f f' s x
  -/
  rw [← hasFDerivWithinAt_diff_singleton x, HasFDerivWithinAt, h, hasFDerivAtFilter_iff_isLittleO]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    s : Set E
    h : Eq (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))) Bot.bot
    ⊢ Asymptotics.IsLittleO Bot.bot (fun x' => HSub.hSub (HSub.hSub (f x') (f x))  …
  -/
  apply isLittleO_bot
  /-
    🎉 no goals
  -/


/-- If `x` is not in the closure of `s`, then `f` has any derivative at `x` within `s`,
as this statement is empty. -/
theorem hasFDerivWithinAt_of_nmem_closure (h : x ∉ closure s) : HasFDerivWithinAt f f' s x :=
  .of_nhdsWithin_eq_bot <| eq_bot_mono (nhdsWithin_mono _ diff_subset) <| by
    /-
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      x : E
      s : Set E
      h : Not (Membership.mem (closure s) x)
      ⊢ Eq (nhdsWithin x s) Bot.bot
    -/
    rwa [mem_closure_iff_nhdsWithin_neBot, not_neBot] at h
    /-
      🎉 no goals
    -/


theorem fderivWithin_zero_of_isolated (h : 𝓝[s \ {x}] x = ⊥) : fderivWithin 𝕜 f s x = 0 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    h : Eq (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))) Bot.bot
    ⊢ Eq (fderivWithin 𝕜 f s x) 0
  -/
  rw [fderivWithin, if_pos (.of_nhdsWithin_eq_bot h)]
  /-
    🎉 no goals
  -/


theorem fderivWithin_zero_of_nmem_closure (h : x ∉ closure s) : fderivWithin 𝕜 f s x = 0 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    h : Not (Membership.mem (closure s) x)
    ⊢ Eq (fderivWithin 𝕜 f s x) 0
  -/
  rw [fderivWithin, if_pos (hasFDerivWithinAt_of_nmem_closure h)]
  /-
    🎉 no goals
  -/


theorem DifferentiableWithinAt.hasFDerivWithinAt (h : DifferentiableWithinAt 𝕜 f s x) :
    HasFDerivWithinAt f (fderivWithin 𝕜 f s x) s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    h : DifferentiableWithinAt 𝕜 f s x
    ⊢ HasFDerivWithinAt f (fderivWithin 𝕜 f s x) s x
  -/
  simp only [fderivWithin, dif_pos h]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    h : DifferentiableWithinAt 𝕜 f s x
    ⊢ HasFDerivWithinAt f (ite (HasFDerivWithinAt f 0 s x) 0 (Classical.choose h)) …
  -/
  split_ifs with h₀
  /-
    case pos
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    h : DifferentiableWithinAt 𝕜 f s x
    h₀ : HasFDerivWithinAt f 0 s x
    ⊢ HasFDerivWithinAt f 0 s x
  -/
  exacts [h₀, Classical.choose_spec h]
  /-
    🎉 no goals
  -/


theorem DifferentiableAt.hasFDerivAt (h : DifferentiableAt 𝕜 f x) :
    HasFDerivAt f (fderiv 𝕜 f x) x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    h : DifferentiableAt 𝕜 f x
    ⊢ HasFDerivAt f (fderiv 𝕜 f x) x
  -/
  rw [fderiv, ← hasFDerivWithinAt_univ]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    h : DifferentiableAt 𝕜 f x
    ⊢ HasFDerivWithinAt f (fderivWithin 𝕜 f Set.univ x) Set.univ x
  -/
  rw [← differentiableWithinAt_univ] at h
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    h : DifferentiableWithinAt 𝕜 f Set.univ x
    ⊢ HasFDerivWithinAt f (fderivWithin 𝕜 f Set.univ x) Set.univ x
  -/
  exact h.hasFDerivWithinAt
  /-
    🎉 no goals
  -/


theorem DifferentiableOn.hasFDerivAt (h : DifferentiableOn 𝕜 f s) (hs : s ∈ 𝓝 x) :
    HasFDerivAt f (fderiv 𝕜 f x) x :=
  ((h x (mem_of_mem_nhds hs)).differentiableAt hs).hasFDerivAt


theorem DifferentiableOn.differentiableAt (h : DifferentiableOn 𝕜 f s) (hs : s ∈ 𝓝 x) :
    DifferentiableAt 𝕜 f x :=
  (h.hasFDerivAt hs).differentiableAt


theorem DifferentiableOn.eventually_differentiableAt (h : DifferentiableOn 𝕜 f s) (hs : s ∈ 𝓝 x) :
    ∀ᶠ y in 𝓝 x, DifferentiableAt 𝕜 f y :=
  (eventually_eventually_nhds.2 hs).mono fun _ => h.differentiableAt


protected theorem HasFDerivAt.fderiv (h : HasFDerivAt f f' x) : fderiv 𝕜 f x = f' := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    h : HasFDerivAt f f' x
    ⊢ Eq (fderiv 𝕜 f x) f'
  -/
  ext
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    h : HasFDerivAt f f' x
    x✝ : E
    ⊢ Eq ((fderiv 𝕜 f x) x✝) (f' x✝)
  -/
  rw [h.unique h.differentiableAt.hasFDerivAt]
  /-
    🎉 no goals
  -/


theorem fderiv_eq {f' : E → E →L[𝕜] F} (h : ∀ x, HasFDerivAt f (f' x) x) : fderiv 𝕜 f = f' :=
  funext fun x => (h x).fderiv


protected theorem HasFDerivWithinAt.fderivWithin (h : HasFDerivWithinAt f f' s x)
    (hxs : UniqueDiffWithinAt 𝕜 s x) : fderivWithin 𝕜 f s x = f' :=
  (hxs.eq h h.differentiableWithinAt.hasFDerivWithinAt).symm


theorem DifferentiableWithinAt.mono (h : DifferentiableWithinAt 𝕜 f t x) (st : s ⊆ t) :
    DifferentiableWithinAt 𝕜 f s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s t : Set E
    h : DifferentiableWithinAt 𝕜 f t x
    st : HasSubset.Subset s t
    ⊢ DifferentiableWithinAt 𝕜 f s x
  -/
  rcases h with ⟨f', hf'⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s t : Set E
    st : HasSubset.Subset s t
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf' : HasFDerivWithinAt f f' t x
    ⊢ DifferentiableWithinAt 𝕜 f s x
  -/
  exact ⟨f', hf'.mono st⟩
  /-
    🎉 no goals
  -/


theorem DifferentiableWithinAt.mono_of_mem_nhdsWithin
    (h : DifferentiableWithinAt 𝕜 f s x) {t : Set E} (hst : s ∈ 𝓝[t] x) :
    DifferentiableWithinAt 𝕜 f t x :=
  (h.hasFDerivWithinAt.mono_of_mem_nhdsWithin hst).differentiableWithinAt


@[deprecated (since := "2024-10-31")]
alias DifferentiableWithinAt.mono_of_mem := DifferentiableWithinAt.mono_of_mem_nhdsWithin


theorem DifferentiableWithinAt.congr_nhds (h : DifferentiableWithinAt 𝕜 f s x) {t : Set E}
    (hst : 𝓝[s] x = 𝓝[t] x) : DifferentiableWithinAt 𝕜 f t x :=
  h.mono_of_mem_nhdsWithin <| hst ▸ self_mem_nhdsWithin


theorem differentiableWithinAt_congr_nhds {t : Set E} (hst : 𝓝[s] x = 𝓝[t] x) :
    DifferentiableWithinAt 𝕜 f s x ↔ DifferentiableWithinAt 𝕜 f t x :=
  ⟨fun h => h.congr_nhds hst, fun h => h.congr_nhds hst.symm⟩


theorem differentiableWithinAt_inter (ht : t ∈ 𝓝 x) :
    DifferentiableWithinAt 𝕜 f (s ∩ t) x ↔ DifferentiableWithinAt 𝕜 f s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s t : Set E
    ht : Membership.mem (nhds x) t
    ⊢ Iff (DifferentiableWithinAt 𝕜 f (Inter.inter s t) x) (DifferentiableWithinAt …
  -/
  simp only [DifferentiableWithinAt, hasFDerivWithinAt_inter ht]
  /-
    🎉 no goals
  -/


theorem differentiableWithinAt_inter' (ht : t ∈ 𝓝[s] x) :
    DifferentiableWithinAt 𝕜 f (s ∩ t) x ↔ DifferentiableWithinAt 𝕜 f s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s t : Set E
    ht : Membership.mem (nhdsWithin x s) t
    ⊢ Iff (DifferentiableWithinAt 𝕜 f (Inter.inter s t) x) (DifferentiableWithinAt …
  -/
  simp only [DifferentiableWithinAt, hasFDerivWithinAt_inter' ht]
  /-
    🎉 no goals
  -/


theorem differentiableWithinAt_insert_self :
    DifferentiableWithinAt 𝕜 f (insert x s) x ↔ DifferentiableWithinAt 𝕜 f s x :=
  ⟨fun h ↦ h.mono (subset_insert x s), fun h ↦ h.hasFDerivWithinAt.insert.differentiableWithinAt⟩


theorem differentiableWithinAt_insert {y : E} :
    DifferentiableWithinAt 𝕜 f (insert y s) x ↔ DifferentiableWithinAt 𝕜 f s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    y : E
    ⊢ Iff (DifferentiableWithinAt 𝕜 f (Insert.insert y s) x) (DifferentiableWithin …
  -/
  rcases eq_or_ne x y with (rfl | h)
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      x : E
      s : Set E
      ⊢ Iff (DifferentiableWithinAt 𝕜 f (Insert.insert x s) x) (DifferentiableWithin …
    -/
  · exact differentiableWithinAt_insert_self
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    y : E
    h : Ne x y
    ⊢ Iff (DifferentiableWithinAt 𝕜 f (Insert.insert y s) x) (DifferentiableWithin …
  -/
  apply differentiableWithinAt_congr_nhds
  /-
    case inr.hst
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    y : E
    h : Ne x y
    ⊢ Eq (nhdsWithin x (Insert.insert y s)) (nhdsWithin x s)
  -/
  exact nhdsWithin_insert_of_ne h
  /-
    🎉 no goals
  -/


alias ⟨DifferentiableWithinAt.of_insert, DifferentiableWithinAt.insert'⟩ :=
differentiableWithinAt_insert


protected theorem DifferentiableWithinAt.insert (h : DifferentiableWithinAt 𝕜 f s x) :
    DifferentiableWithinAt 𝕜 f (insert x s) x :=
  h.insert'


theorem DifferentiableAt.differentiableWithinAt (h : DifferentiableAt 𝕜 f x) :
    DifferentiableWithinAt 𝕜 f s x :=
  (differentiableWithinAt_univ.2 h).mono (subset_univ _)


@[fun_prop]
theorem Differentiable.differentiableAt (h : Differentiable 𝕜 f) : DifferentiableAt 𝕜 f x :=
  h x


protected theorem DifferentiableAt.fderivWithin (h : DifferentiableAt 𝕜 f x)
    (hxs : UniqueDiffWithinAt 𝕜 s x) : fderivWithin 𝕜 f s x = fderiv 𝕜 f x :=
  h.hasFDerivAt.hasFDerivWithinAt.fderivWithin hxs


theorem DifferentiableOn.mono (h : DifferentiableOn 𝕜 f t) (st : s ⊆ t) : DifferentiableOn 𝕜 f s :=
  fun x hx => (h x (st hx)).mono st


theorem differentiableOn_univ : DifferentiableOn 𝕜 f univ ↔ Differentiable 𝕜 f := by
  simp only [DifferentiableOn, Differentiable, differentiableWithinAt_univ, mem_univ,
    forall_true_left]


@[fun_prop]
theorem Differentiable.differentiableOn (h : Differentiable 𝕜 f) : DifferentiableOn 𝕜 f s :=
  (differentiableOn_univ.2 h).mono (subset_univ _)


theorem differentiableOn_of_locally_differentiableOn
    (h : ∀ x ∈ s, ∃ u, IsOpen u ∧ x ∈ u ∧ DifferentiableOn 𝕜 f (s ∩ u)) :
    DifferentiableOn 𝕜 f s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : ∀ (x : E), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    ⊢ DifferentiableOn 𝕜 f s
  -/
  intro x xs
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : ∀ (x : E), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    x : E
    xs : Membership.mem s x
    ⊢ DifferentiableWithinAt 𝕜 f s x
  -/
  rcases h x xs with ⟨t, t_open, xt, ht⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : ∀ (x : E), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    x : E
    xs : Membership.mem s x
    t : Set E
    t_open : IsOpen t
    xt : Membership.mem t x
    ht : DifferentiableOn 𝕜 f (Inter.inter s t)
    ⊢ DifferentiableWithinAt 𝕜 f s x
  -/
  exact (differentiableWithinAt_inter (IsOpen.mem_nhds t_open xt)).1 (ht x ⟨xs, xt⟩)
  /-
    🎉 no goals
  -/


theorem fderivWithin_of_mem_nhdsWithin (st : t ∈ 𝓝[s] x) (ht : UniqueDiffWithinAt 𝕜 s x)
    (h : DifferentiableWithinAt 𝕜 f t x) : fderivWithin 𝕜 f s x = fderivWithin 𝕜 f t x :=
  ((DifferentiableWithinAt.hasFDerivWithinAt h).mono_of_mem_nhdsWithin st).fderivWithin ht


@[deprecated (since := "2024-10-31")]
alias fderivWithin_of_mem := fderivWithin_of_mem_nhdsWithin


theorem fderivWithin_subset (st : s ⊆ t) (ht : UniqueDiffWithinAt 𝕜 s x)
    (h : DifferentiableWithinAt 𝕜 f t x) : fderivWithin 𝕜 f s x = fderivWithin 𝕜 f t x :=
  fderivWithin_of_mem_nhdsWithin (nhdsWithin_mono _ st self_mem_nhdsWithin) ht h


theorem fderivWithin_inter (ht : t ∈ 𝓝 x) : fderivWithin 𝕜 f (s ∩ t) x = fderivWithin 𝕜 f s x := by
  classical
  simp [fderivWithin, hasFDerivWithinAt_inter ht, DifferentiableWithinAt]


theorem fderivWithin_of_mem_nhds (h : s ∈ 𝓝 x) : fderivWithin 𝕜 f s x = fderiv 𝕜 f x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    h : Membership.mem (nhds x) s
    ⊢ Eq (fderivWithin 𝕜 f s x) (fderiv 𝕜 f x)
  -/
  rw [← fderivWithin_univ, ← univ_inter s, fderivWithin_inter h]
  /-
    🎉 no goals
  -/


theorem fderivWithin_of_isOpen (hs : IsOpen s) (hx : x ∈ s) : fderivWithin 𝕜 f s x = fderiv 𝕜 f x :=
  fderivWithin_of_mem_nhds (hs.mem_nhds hx)


theorem fderivWithin_eq_fderiv (hs : UniqueDiffWithinAt 𝕜 s x) (h : DifferentiableAt 𝕜 f x) :
    fderivWithin 𝕜 f s x = fderiv 𝕜 f x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    hs : UniqueDiffWithinAt 𝕜 s x
    h : DifferentiableAt 𝕜 f x
    ⊢ Eq (fderivWithin 𝕜 f s x) (fderiv 𝕜 f x)
  -/
  rw [← fderivWithin_univ]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    hs : UniqueDiffWithinAt 𝕜 s x
    h : DifferentiableAt 𝕜 f x
    ⊢ Eq (fderivWithin 𝕜 f s x) (fderivWithin 𝕜 f Set.univ x)
  -/
  exact fderivWithin_subset (subset_univ _) hs h.differentiableWithinAt
  /-
    🎉 no goals
  -/


theorem fderiv_mem_iff {f : E → F} {s : Set (E →L[𝕜] F)} {x : E} : fderiv 𝕜 f x ∈ s ↔
    DifferentiableAt 𝕜 f x ∧ fderiv 𝕜 f x ∈ s ∨ ¬DifferentiableAt 𝕜 f x ∧ (0 : E →L[𝕜] F) ∈ s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    x : E
    ⊢ Iff (Membership.mem s (fderiv 𝕜 f x)) (Or (And (DifferentiableAt 𝕜 f x) (Mem …
  -/
                                           /-
                                             🎉 no goals
                                           -/
  by_cases hx : DifferentiableAt 𝕜 f x <;> simp [fderiv_zero_of_not_differentiableAt, *]
                                           /-
                                             🎉 no goals
                                           -/


theorem fderivWithin_mem_iff {f : E → F} {t : Set E} {s : Set (E →L[𝕜] F)} {x : E} :
    fderivWithin 𝕜 f t x ∈ s ↔
      DifferentiableWithinAt 𝕜 f t x ∧ fderivWithin 𝕜 f t x ∈ s ∨
        ¬DifferentiableWithinAt 𝕜 f t x ∧ (0 : E →L[𝕜] F) ∈ s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    t : Set E
    s : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
    x : E
    ⊢ Iff (Membership.mem s (fderivWithin 𝕜 f t x)) (Or (And (DifferentiableWithin …
  -/
  by_cases hx : DifferentiableWithinAt 𝕜 f t x <;>
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      t : Set E
      s : Set (ContinuousLinearMap (RingHom.id 𝕜) E F)
      x : E
      hx : DifferentiableWithinAt 𝕜 f t x
      ⊢ Iff (Membership.mem s (fderivWithin 𝕜 f t x)) (Or (And (DifferentiableWithin …
    -/
    /-
      🎉 no goals
    -/
    simp [fderivWithin_zero_of_not_differentiableWithinAt, *]
    /-
      🎉 no goals
    -/


theorem Asymptotics.IsBigO.hasFDerivWithinAt {s : Set E} {x₀ : E} {n : ℕ}
    (h : f =O[𝓝[s] x₀] fun x => ‖x - x₀‖ ^ n) (hx₀ : x₀ ∈ s) (hn : 1 < n) :
    HasFDerivWithinAt f (0 : E →L[𝕜] F) s x₀ := by
  simp_rw [HasFDerivWithinAt, hasFDerivAtFilter_iff_isLittleO,
    h.eq_zero_of_norm_pow_within hx₀ hn.ne_bot, zero_apply, sub_zero,
    h.trans_isLittleO ((isLittleO_pow_sub_sub x₀ hn).mono nhdsWithin_le_nhds)]


theorem Asymptotics.IsBigO.hasFDerivAt {x₀ : E} {n : ℕ} (h : f =O[𝓝 x₀] fun x => ‖x - x₀‖ ^ n)
    (hn : 1 < n) : HasFDerivAt f (0 : E →L[𝕜] F) x₀ := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x₀ : E
    n : Nat
    h : Asymptotics.IsBigO (nhds x₀) f fun x => HPow.hPow (Norm.norm (HSub.hSub x  …
    hn : LT.lt 1 n
    ⊢ HasFDerivAt f 0 x₀
  -/
  rw [← nhdsWithin_univ] at h
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x₀ : E
    n : Nat
    h : Asymptotics.IsBigO (nhdsWithin x₀ Set.univ) f fun x => HPow.hPow (Norm.nor …
    hn : LT.lt 1 n
    ⊢ HasFDerivAt f 0 x₀
  -/
  exact (h.hasFDerivWithinAt (mem_univ _) hn).hasFDerivAt_of_univ
  /-
    🎉 no goals
  -/


nonrec theorem HasFDerivWithinAt.isBigO_sub {f : E → F} {s : Set E} {x₀ : E} {f' : E →L[𝕜] F}
    (h : HasFDerivWithinAt f f' s x₀) : (f · - f x₀) =O[𝓝[s] x₀] (· - x₀) :=
  h.isBigO_sub


lemma DifferentiableWithinAt.isBigO_sub {f : E → F} {s : Set E} {x₀ : E}
    (h : DifferentiableWithinAt 𝕜 f s x₀) : (f · - f x₀) =O[𝓝[s] x₀] (· - x₀) :=
  h.hasFDerivWithinAt.isBigO_sub


nonrec theorem HasFDerivAt.isBigO_sub {f : E → F} {x₀ : E} {f' : E →L[𝕜] F}
    (h : HasFDerivAt f f' x₀) : (f · - f x₀) =O[𝓝 x₀] (· - x₀) :=
  h.isBigO_sub


nonrec theorem DifferentiableAt.isBigO_sub {f : E → F} {x₀ : E} (h : DifferentiableAt 𝕜 f x₀) :
    (f · - f x₀) =O[𝓝 x₀] (· - x₀) :=
  h.hasFDerivAt.isBigO_sub


theorem HasFDerivAtFilter.tendsto_nhds (hL : L ≤ 𝓝 x) (h : HasFDerivAtFilter f f' x L) :
    Tendsto f L (𝓝 (f x)) := by
  have : Tendsto (fun x' => f x' - f x) L (𝓝 0) := by
    refine h.isBigO_sub.trans_tendsto (Tendsto.mono_left ?_ hL)
    rw [← sub_self x]
    exact tendsto_id.sub tendsto_const_nhds
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    L : Filter E
    hL : LE.le L (nhds x)
    h : HasFDerivAtFilter f f' x L
    this : Filter.Tendsto (fun x' => HSub.hSub (f x') (f x)) L (nhds 0)
    ⊢ Filter.Tendsto f L (nhds (f x))
  -/
  have := this.add (tendsto_const_nhds (x := f x))
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    L : Filter E
    hL : LE.le L (nhds x)
    h : HasFDerivAtFilter f f' x L
    this✝ : Filter.Tendsto (fun x' => HSub.hSub (f x') (f x)) L (nhds 0)
    this : Filter.Tendsto (fun x_1 => HAdd.hAdd (HSub.hSub (f x_1) (f x)) (f x)) L …
    ⊢ Filter.Tendsto f L (nhds (f x))
  -/
  rw [zero_add (f x)] at this
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    L : Filter E
    hL : LE.le L (nhds x)
    h : HasFDerivAtFilter f f' x L
    this✝ : Filter.Tendsto (fun x' => HSub.hSub (f x') (f x)) L (nhds 0)
    this : Filter.Tendsto (fun x_1 => HAdd.hAdd (HSub.hSub (f x_1) (f x)) (f x)) L …
    ⊢ Filter.Tendsto f L (nhds (f x))
  -/
  exact this.congr (by simp only [sub_add_cancel, eq_self_iff_true, forall_const])
  /-
    🎉 no goals
  -/


theorem HasFDerivWithinAt.continuousWithinAt (h : HasFDerivWithinAt f f' s x) :
    ContinuousWithinAt f s x :=
  HasFDerivAtFilter.tendsto_nhds inf_le_left h


theorem HasFDerivAt.continuousAt (h : HasFDerivAt f f' x) : ContinuousAt f x :=
  HasFDerivAtFilter.tendsto_nhds le_rfl h


@[fun_prop]
theorem DifferentiableWithinAt.continuousWithinAt (h : DifferentiableWithinAt 𝕜 f s x) :
    ContinuousWithinAt f s x :=
  let ⟨_, hf'⟩ := h
  hf'.continuousWithinAt


@[fun_prop]
theorem DifferentiableAt.continuousAt (h : DifferentiableAt 𝕜 f x) : ContinuousAt f x :=
  let ⟨_, hf'⟩ := h
  hf'.continuousAt


@[fun_prop]
theorem DifferentiableOn.continuousOn (h : DifferentiableOn 𝕜 f s) : ContinuousOn f s := fun x hx =>
  (h x hx).continuousWithinAt


@[fun_prop]
theorem Differentiable.continuous (h : Differentiable 𝕜 f) : Continuous f :=
  continuous_iff_continuousAt.2 fun x => (h x).continuousAt


protected theorem HasStrictFDerivAt.continuousAt (hf : HasStrictFDerivAt f f' x) :
    ContinuousAt f x :=
  hf.hasFDerivAt.continuousAt


theorem HasStrictFDerivAt.isBigO_sub_rev {f' : E ≃L[𝕜] F}
    (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) x) :
    (fun p : E × E => p.1 - p.2) =O[𝓝 (x, x)] fun p : E × E => f p.1 - f p.2 :=
  ((f'.isBigO_comp_rev _ _).trans
      (hf.isLittleO.trans_isBigO (f'.isBigO_comp_rev _ _)).right_isBigO_add).congr
    (fun _ => rfl) fun _ => sub_add_cancel _ _


theorem HasFDerivAtFilter.isBigO_sub_rev (hf : HasFDerivAtFilter f f' x L) {C}
    (hf' : AntilipschitzWith C f') : (fun x' => x' - x) =O[L] fun x' => f x' - f x :=
  have : (fun x' => x' - x) =O[L] fun x' => f' (x' - x) :=
    isBigO_iff.2 ⟨C, Eventually.of_forall fun _ => ZeroHomClass.bound_of_antilipschitz f' hf' _⟩
  (this.trans (hf.isLittleO.trans_isBigO this).right_isBigO_add).congr (fun _ => rfl) fun _ =>
    sub_add_cancel _ _


theorem hasFDerivWithinAt_congr_set' (y : E) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
    HasFDerivWithinAt f f' s x ↔ HasFDerivWithinAt f f' t x :=
  calc
    HasFDerivWithinAt f f' s x ↔ HasFDerivWithinAt f f' (s \ {y}) x :=
      (hasFDerivWithinAt_diff_singleton _).symm
    _ ↔ HasFDerivWithinAt f f' (t \ {y}) x := by
      /-
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        f' : ContinuousLinearMap (RingHom.id 𝕜) E F
        x : E
        s t : Set E
        y : E
        h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
        ⊢ Iff (HasFDerivWithinAt f f' (SDiff.sdiff s (Singleton.singleton y)) x) (HasF …
      -/
      suffices 𝓝[s \ {y}] x = 𝓝[t \ {y}] x by simp only [HasFDerivWithinAt, this]
      simpa only [set_eventuallyEq_iff_inf_principal, ← nhdsWithin_inter', diff_eq,
        inter_comm] using h
    _ ↔ HasFDerivWithinAt f f' t x := hasFDerivWithinAt_diff_singleton _


theorem hasFDerivWithinAt_congr_set (h : s =ᶠ[𝓝 x] t) :
    HasFDerivWithinAt f f' s x ↔ HasFDerivWithinAt f f' t x :=
  hasFDerivWithinAt_congr_set' x <| h.filter_mono inf_le_left


theorem differentiableWithinAt_congr_set' (y : E) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
    DifferentiableWithinAt 𝕜 f s x ↔ DifferentiableWithinAt 𝕜 f t x :=
  exists_congr fun _ => hasFDerivWithinAt_congr_set' _ h


theorem differentiableWithinAt_congr_set (h : s =ᶠ[𝓝 x] t) :
    DifferentiableWithinAt 𝕜 f s x ↔ DifferentiableWithinAt 𝕜 f t x :=
  exists_congr fun _ => hasFDerivWithinAt_congr_set h


theorem fderivWithin_congr_set' (y : E) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
    fderivWithin 𝕜 f s x = fderivWithin 𝕜 f t x := by
  classical
  simp only [fderivWithin, differentiableWithinAt_congr_set' _ h, hasFDerivWithinAt_congr_set' _ h]


theorem fderivWithin_congr_set (h : s =ᶠ[𝓝 x] t) : fderivWithin 𝕜 f s x = fderivWithin 𝕜 f t x :=
  fderivWithin_congr_set' x <| h.filter_mono inf_le_left


theorem fderivWithin_eventually_congr_set' (y : E) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
    fderivWithin 𝕜 f s =ᶠ[𝓝 x] fderivWithin 𝕜 f t :=
  (eventually_nhds_nhdsWithin.2 h).mono fun _ => fderivWithin_congr_set' y


theorem fderivWithin_eventually_congr_set (h : s =ᶠ[𝓝 x] t) :
    fderivWithin 𝕜 f s =ᶠ[𝓝 x] fderivWithin 𝕜 f t :=
  fderivWithin_eventually_congr_set' x <| h.filter_mono inf_le_left


theorem Filter.EventuallyEq.hasStrictFDerivAt_iff (h : f₀ =ᶠ[𝓝 x] f₁) (h' : ∀ y, f₀' y = f₁' y) :
    HasStrictFDerivAt f₀ f₀' x ↔ HasStrictFDerivAt f₁ f₁' x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f₀ f₁ : E → F
    f₀' f₁' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    h : (nhds x).EventuallyEq f₀ f₁
    h' : ∀ (y : E), Eq (f₀' y) (f₁' y)
    ⊢ Iff (HasStrictFDerivAt f₀ f₀' x) (HasStrictFDerivAt f₁ f₁' x)
  -/
  rw [hasStrictFDerivAt_iff_isLittleO, hasStrictFDerivAt_iff_isLittleO]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f₀ f₁ : E → F
    f₀' f₁' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    h : (nhds x).EventuallyEq f₀ f₁
    h' : ∀ (y : E), Eq (f₀' y) (f₁' y)
    ⊢ Iff (Asymptotics.IsLittleO (nhds { fst := x, snd := x }) (fun p => HSub.hSub …
  -/
  refine isLittleO_congr ((h.prod_mk_nhds h).mono ?_) .rfl
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f₀ f₁ : E → F
    f₀' f₁' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    h : (nhds x).EventuallyEq f₀ f₁
    h' : ∀ (y : E), Eq (f₀' y) (f₁' y)
    ⊢ ∀ (x : Prod E E), And (Eq (f₀ x.1) (f₁ x.1)) (Eq (f₀ x.2) (f₁ x.2)) → Eq ((f …
  -/
  rintro p ⟨hp₁, hp₂⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f₀ f₁ : E → F
    f₀' f₁' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    h : (nhds x).EventuallyEq f₀ f₁
    h' : ∀ (y : E), Eq (f₀' y) (f₁' y)
    p : Prod E E
    hp₁ : Eq (f₀ p.1) (f₁ p.1)
    hp₂ : Eq (f₀ p.2) (f₁ p.2)
    ⊢ Eq ((fun p => HSub.hSub (HSub.hSub (f₀ p.1) (f₀ p.2)) (f₀' (HSub.hSub p.1 p. …
  -/
  simp only [*]
  /-
    🎉 no goals
  -/


theorem HasStrictFDerivAt.congr_fderiv (h : HasStrictFDerivAt f f' x) (h' : f' = g') :
    HasStrictFDerivAt f g' x :=
  h' ▸ h


theorem HasFDerivAt.congr_fderiv (h : HasFDerivAt f f' x) (h' : f' = g') : HasFDerivAt f g' x :=
  h' ▸ h


theorem HasFDerivWithinAt.congr_fderiv (h : HasFDerivWithinAt f f' s x) (h' : f' = g') :
    HasFDerivWithinAt f g' s x :=
  h' ▸ h


theorem HasStrictFDerivAt.congr_of_eventuallyEq (h : HasStrictFDerivAt f f' x)
    (h₁ : f =ᶠ[𝓝 x] f₁) : HasStrictFDerivAt f₁ f' x :=
  (h₁.hasStrictFDerivAt_iff fun _ => rfl).1 h


theorem Filter.EventuallyEq.hasFDerivAtFilter_iff (h₀ : f₀ =ᶠ[L] f₁) (hx : f₀ x = f₁ x)
    (h₁ : ∀ x, f₀' x = f₁' x) : HasFDerivAtFilter f₀ f₀' x L ↔ HasFDerivAtFilter f₁ f₁' x L := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f₀ f₁ : E → F
    f₀' f₁' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    L : Filter E
    h₀ : L.EventuallyEq f₀ f₁
    hx : Eq (f₀ x) (f₁ x)
    h₁ : ∀ (x : E), Eq (f₀' x) (f₁' x)
    ⊢ Iff (HasFDerivAtFilter f₀ f₀' x L) (HasFDerivAtFilter f₁ f₁' x L)
  -/
  simp only [hasFDerivAtFilter_iff_isLittleO]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f₀ f₁ : E → F
    f₀' f₁' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    L : Filter E
    h₀ : L.EventuallyEq f₀ f₁
    hx : Eq (f₀ x) (f₁ x)
    h₁ : ∀ (x : E), Eq (f₀' x) (f₁' x)
    ⊢ Iff (Asymptotics.IsLittleO L (fun x' => HSub.hSub (HSub.hSub (f₀ x') (f₀ x)) …
  -/
  exact isLittleO_congr (h₀.mono fun y hy => by simp only [hy, h₁, hx]) .rfl
  /-
    🎉 no goals
  -/


theorem HasFDerivAtFilter.congr_of_eventuallyEq (h : HasFDerivAtFilter f f' x L) (hL : f₁ =ᶠ[L] f)
    (hx : f₁ x = f x) : HasFDerivAtFilter f₁ f' x L :=
  (hL.hasFDerivAtFilter_iff hx fun _ => rfl).2 h


theorem Filter.EventuallyEq.hasFDerivAt_iff (h : f₀ =ᶠ[𝓝 x] f₁) :
    HasFDerivAt f₀ f' x ↔ HasFDerivAt f₁ f' x :=
  h.hasFDerivAtFilter_iff h.eq_of_nhds fun _ => _root_.rfl


theorem Filter.EventuallyEq.differentiableAt_iff (h : f₀ =ᶠ[𝓝 x] f₁) :
    DifferentiableAt 𝕜 f₀ x ↔ DifferentiableAt 𝕜 f₁ x :=
  exists_congr fun _ => h.hasFDerivAt_iff


theorem Filter.EventuallyEq.hasFDerivWithinAt_iff (h : f₀ =ᶠ[𝓝[s] x] f₁) (hx : f₀ x = f₁ x) :
    HasFDerivWithinAt f₀ f' s x ↔ HasFDerivWithinAt f₁ f' s x :=
  h.hasFDerivAtFilter_iff hx fun _ => _root_.rfl


theorem Filter.EventuallyEq.hasFDerivWithinAt_iff_of_mem (h : f₀ =ᶠ[𝓝[s] x] f₁) (hx : x ∈ s) :
    HasFDerivWithinAt f₀ f' s x ↔ HasFDerivWithinAt f₁ f' s x :=
  h.hasFDerivWithinAt_iff (h.eq_of_nhdsWithin hx)


theorem Filter.EventuallyEq.differentiableWithinAt_iff (h : f₀ =ᶠ[𝓝[s] x] f₁) (hx : f₀ x = f₁ x) :
    DifferentiableWithinAt 𝕜 f₀ s x ↔ DifferentiableWithinAt 𝕜 f₁ s x :=
  exists_congr fun _ => h.hasFDerivWithinAt_iff hx


theorem Filter.EventuallyEq.differentiableWithinAt_iff_of_mem (h : f₀ =ᶠ[𝓝[s] x] f₁) (hx : x ∈ s) :
    DifferentiableWithinAt 𝕜 f₀ s x ↔ DifferentiableWithinAt 𝕜 f₁ s x :=
  h.differentiableWithinAt_iff (h.eq_of_nhdsWithin hx)


theorem HasFDerivWithinAt.congr_mono (h : HasFDerivWithinAt f f' s x) (ht : EqOn f₁ f t)
    (hx : f₁ x = f x) (h₁ : t ⊆ s) : HasFDerivWithinAt f₁ f' t x :=
  HasFDerivAtFilter.congr_of_eventuallyEq (h.mono h₁) (Filter.mem_inf_of_right ht) hx


theorem HasFDerivWithinAt.congr (h : HasFDerivWithinAt f f' s x) (hs : EqOn f₁ f s)
    (hx : f₁ x = f x) : HasFDerivWithinAt f₁ f' s x :=
  h.congr_mono hs hx (Subset.refl _)


theorem HasFDerivWithinAt.congr' (h : HasFDerivWithinAt f f' s x) (hs : EqOn f₁ f s) (hx : x ∈ s) :
    HasFDerivWithinAt f₁ f' s x :=
  h.congr hs (hs hx)


theorem HasFDerivWithinAt.congr_of_eventuallyEq (h : HasFDerivWithinAt f f' s x)
    (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) : HasFDerivWithinAt f₁ f' s x :=
  HasFDerivAtFilter.congr_of_eventuallyEq h h₁ hx


theorem HasFDerivAt.congr_of_eventuallyEq (h : HasFDerivAt f f' x) (h₁ : f₁ =ᶠ[𝓝 x] f) :
    HasFDerivAt f₁ f' x :=
  HasFDerivAtFilter.congr_of_eventuallyEq h h₁ (mem_of_mem_nhds h₁ : _)


theorem DifferentiableWithinAt.congr_mono (h : DifferentiableWithinAt 𝕜 f s x) (ht : EqOn f₁ f t)
    (hx : f₁ x = f x) (h₁ : t ⊆ s) : DifferentiableWithinAt 𝕜 f₁ t x :=
  (HasFDerivWithinAt.congr_mono h.hasFDerivWithinAt ht hx h₁).differentiableWithinAt


theorem DifferentiableWithinAt.congr (h : DifferentiableWithinAt 𝕜 f s x) (ht : ∀ x ∈ s, f₁ x = f x)
    (hx : f₁ x = f x) : DifferentiableWithinAt 𝕜 f₁ s x :=
  DifferentiableWithinAt.congr_mono h ht hx (Subset.refl _)


theorem DifferentiableWithinAt.congr_of_eventuallyEq (h : DifferentiableWithinAt 𝕜 f s x)
    (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) : DifferentiableWithinAt 𝕜 f₁ s x :=
  (h.hasFDerivWithinAt.congr_of_eventuallyEq h₁ hx).differentiableWithinAt


theorem DifferentiableWithinAt.congr_of_eventuallyEq_of_mem (h : DifferentiableWithinAt 𝕜 f s x)
    (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : x ∈ s) : DifferentiableWithinAt 𝕜 f₁ s x :=
  h.congr_of_eventuallyEq h₁ (mem_of_mem_nhdsWithin hx h₁ :)


theorem DifferentiableWithinAt.congr_of_eventuallyEq_insert (h : DifferentiableWithinAt 𝕜 f s x)
    (h₁ : f₁ =ᶠ[𝓝[insert x s] x] f) : DifferentiableWithinAt 𝕜 f₁ s x :=
  (h.insert.congr_of_eventuallyEq_of_mem h₁ (mem_insert _ _)).of_insert


theorem DifferentiableOn.congr_mono (h : DifferentiableOn 𝕜 f s) (h' : ∀ x ∈ t, f₁ x = f x)
    (h₁ : t ⊆ s) : DifferentiableOn 𝕜 f₁ t := fun x hx => (h x (h₁ hx)).congr_mono h' (h' x hx) h₁


theorem DifferentiableOn.congr (h : DifferentiableOn 𝕜 f s) (h' : ∀ x ∈ s, f₁ x = f x) :
    DifferentiableOn 𝕜 f₁ s := fun x hx => (h x hx).congr h' (h' x hx)


theorem differentiableOn_congr (h' : ∀ x ∈ s, f₁ x = f x) :
    DifferentiableOn 𝕜 f₁ s ↔ DifferentiableOn 𝕜 f s :=
  ⟨fun h => DifferentiableOn.congr h fun y hy => (h' y hy).symm, fun h =>
    DifferentiableOn.congr h h'⟩


theorem DifferentiableAt.congr_of_eventuallyEq (h : DifferentiableAt 𝕜 f x) (hL : f₁ =ᶠ[𝓝 x] f) :
    DifferentiableAt 𝕜 f₁ x :=
  hL.differentiableAt_iff.2 h


theorem DifferentiableWithinAt.fderivWithin_congr_mono (h : DifferentiableWithinAt 𝕜 f s x)
    (hs : EqOn f₁ f t) (hx : f₁ x = f x) (hxt : UniqueDiffWithinAt 𝕜 t x) (h₁ : t ⊆ s) :
    fderivWithin 𝕜 f₁ t x = fderivWithin 𝕜 f s x :=
  (HasFDerivWithinAt.congr_mono h.hasFDerivWithinAt hs hx h₁).fderivWithin hxt


theorem Filter.EventuallyEq.fderivWithin_eq (hs : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) :
    fderivWithin 𝕜 f₁ s x = fderivWithin 𝕜 f s x := by
  classical
  simp only [fderivWithin, DifferentiableWithinAt, hs.hasFDerivWithinAt_iff hx]


theorem Filter.EventuallyEq.fderivWithin_eq_of_mem (hs : f₁ =ᶠ[𝓝[s] x] f) (hx : x ∈ s) :
    fderivWithin 𝕜 f₁ s x = fderivWithin 𝕜 f s x :=
  hs.fderivWithin_eq (mem_of_mem_nhdsWithin hx hs :)


theorem Filter.EventuallyEq.fderivWithin_eq_of_insert (hs : f₁ =ᶠ[𝓝[insert x s] x] f) :
    fderivWithin 𝕜 f₁ s x = fderivWithin 𝕜 f s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f f₁ : E → F
    x : E
    s : Set E
    hs : (nhdsWithin x (Insert.insert x s)).EventuallyEq f₁ f
    ⊢ Eq (fderivWithin 𝕜 f₁ s x) (fderivWithin 𝕜 f s x)
  -/
  apply Filter.EventuallyEq.fderivWithin_eq (nhdsWithin_mono _ (subset_insert x s) hs)
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f f₁ : E → F
    x : E
    s : Set E
    hs : (nhdsWithin x (Insert.insert x s)).EventuallyEq f₁ f
    ⊢ Eq (f₁ x) (f x)
  -/
  exact (mem_of_mem_nhdsWithin (mem_insert x s) hs :)
  /-
    🎉 no goals
  -/


theorem Filter.EventuallyEq.fderivWithin' (hs : f₁ =ᶠ[𝓝[s] x] f) (ht : t ⊆ s) :
    fderivWithin 𝕜 f₁ t =ᶠ[𝓝[s] x] fderivWithin 𝕜 f t :=
  (eventually_eventually_nhdsWithin.2 hs).mp <|
    eventually_mem_nhdsWithin.mono fun _y hys hs =>
      EventuallyEq.fderivWithin_eq (hs.filter_mono <| nhdsWithin_mono _ ht)
        (hs.self_of_nhdsWithin hys)


protected theorem Filter.EventuallyEq.fderivWithin (hs : f₁ =ᶠ[𝓝[s] x] f) :
    fderivWithin 𝕜 f₁ s =ᶠ[𝓝[s] x] fderivWithin 𝕜 f s :=
  hs.fderivWithin' Subset.rfl


theorem Filter.EventuallyEq.fderivWithin_eq_nhds (h : f₁ =ᶠ[𝓝 x] f) :
    fderivWithin 𝕜 f₁ s x = fderivWithin 𝕜 f s x :=
  (h.filter_mono nhdsWithin_le_nhds).fderivWithin_eq h.self_of_nhds


theorem fderivWithin_congr (hs : EqOn f₁ f s) (hx : f₁ x = f x) :
    fderivWithin 𝕜 f₁ s x = fderivWithin 𝕜 f s x :=
  (hs.eventuallyEq.filter_mono inf_le_right).fderivWithin_eq hx


theorem fderivWithin_congr' (hs : EqOn f₁ f s) (hx : x ∈ s) :
    fderivWithin 𝕜 f₁ s x = fderivWithin 𝕜 f s x :=
  fderivWithin_congr hs (hs hx)


theorem Filter.EventuallyEq.fderiv_eq (h : f₁ =ᶠ[𝓝 x] f) : fderiv 𝕜 f₁ x = fderiv 𝕜 f x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f f₁ : E → F
    x : E
    h : (nhds x).EventuallyEq f₁ f
    ⊢ Eq (fderiv 𝕜 f₁ x) (fderiv 𝕜 f x)
  -/
  rw [← fderivWithin_univ, ← fderivWithin_univ, h.fderivWithin_eq_nhds]
  /-
    🎉 no goals
  -/


protected theorem Filter.EventuallyEq.fderiv (h : f₁ =ᶠ[𝓝 x] f) : fderiv 𝕜 f₁ =ᶠ[𝓝 x] fderiv 𝕜 f :=
  h.eventuallyEq_nhds.mono fun _ h => h.fderiv_eq


@[fun_prop]
theorem hasStrictFDerivAt_id (x : E) : HasStrictFDerivAt id (id 𝕜 E) x :=
                                                         /-
                                                           𝕜 : Type u_1
                                                           inst✝² : NontriviallyNormedField 𝕜
                                                           E : Type u_2
                                                           inst✝¹ : NormedAddCommGroup E
                                                           inst✝ : NormedSpace 𝕜 E
                                                           x : E
                                                           ⊢ ∀ (x : Prod E E), Eq 0 (HSub.hSub (HSub.hSub (id x.1) (id x.2)) ((Continuous …
                                                         -/
  .of_isLittleO <| (isLittleO_zero _ _).congr_left <| by simp
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem hasFDerivAtFilter_id (x : E) (L : Filter E) : HasFDerivAtFilter id (id 𝕜 E) x L :=
                                                         /-
                                                           𝕜 : Type u_1
                                                           inst✝² : NontriviallyNormedField 𝕜
                                                           E : Type u_2
                                                           inst✝¹ : NormedAddCommGroup E
                                                           inst✝ : NormedSpace 𝕜 E
                                                           x : E
                                                           L : Filter E
                                                           ⊢ ∀ (x_1 : E), Eq 0 (HSub.hSub (HSub.hSub (id x_1) (id x)) ((ContinuousLinearM …
                                                         -/
  .of_isLittleO <| (isLittleO_zero _ _).congr_left <| by simp
                                                         /-
                                                           🎉 no goals
                                                         -/


@[fun_prop]
theorem hasFDerivWithinAt_id (x : E) (s : Set E) : HasFDerivWithinAt id (id 𝕜 E) s x :=
  hasFDerivAtFilter_id _ _


@[fun_prop]
theorem hasFDerivAt_id (x : E) : HasFDerivAt id (id 𝕜 E) x :=
  hasFDerivAtFilter_id _ _


@[simp, fun_prop]
theorem differentiableAt_id : DifferentiableAt 𝕜 id x :=
  (hasFDerivAt_id x).differentiableAt


/-- Variant with `fun x => x` rather than `id` -/
@[simp]
theorem differentiableAt_id' : DifferentiableAt 𝕜 (fun x => x) x :=
  (hasFDerivAt_id x).differentiableAt


@[fun_prop]
theorem differentiableWithinAt_id : DifferentiableWithinAt 𝕜 id s x :=
  differentiableAt_id.differentiableWithinAt


/-- Variant with `fun x => x` rather than `id` -/
@[fun_prop]
theorem differentiableWithinAt_id' : DifferentiableWithinAt 𝕜 (fun x => x) s x :=
  differentiableWithinAt_id


@[simp, fun_prop]
theorem differentiable_id : Differentiable 𝕜 (id : E → E) := fun _ => differentiableAt_id


/-- Variant with `fun x => x` rather than `id` -/
@[simp]
theorem differentiable_id' : Differentiable 𝕜 fun x : E => x := fun _ => differentiableAt_id


@[fun_prop]
theorem differentiableOn_id : DifferentiableOn 𝕜 id s :=
  differentiable_id.differentiableOn


@[simp]
theorem fderiv_id : fderiv 𝕜 id x = id 𝕜 E :=
  HasFDerivAt.fderiv (hasFDerivAt_id x)


@[simp]
theorem fderiv_id' : fderiv 𝕜 (fun x : E => x) x = ContinuousLinearMap.id 𝕜 E :=
  fderiv_id


theorem fderivWithin_id (hxs : UniqueDiffWithinAt 𝕜 s x) : fderivWithin 𝕜 id s x = id 𝕜 E := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (fderivWithin 𝕜 id s x) (ContinuousLinearMap.id 𝕜 E)
  -/
  rw [DifferentiableAt.fderivWithin differentiableAt_id hxs]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (fderiv 𝕜 id x) (ContinuousLinearMap.id 𝕜 E)
  -/
  exact fderiv_id
  /-
    🎉 no goals
  -/


theorem fderivWithin_id' (hxs : UniqueDiffWithinAt 𝕜 s x) :
    fderivWithin 𝕜 (fun x : E => x) s x = ContinuousLinearMap.id 𝕜 E :=
  fderivWithin_id hxs


@[fun_prop]
theorem hasStrictFDerivAt_const (c : F) (x : E) :
    HasStrictFDerivAt (fun _ => c) (0 : E →L[𝕜] F) x :=
                                                               /-
                                                                 𝕜 : Type u_1
                                                                 inst✝⁴ : NontriviallyNormedField 𝕜
                                                                 E : Type u_2
                                                                 inst✝³ : NormedAddCommGroup E
                                                                 inst✝² : NormedSpace 𝕜 E
                                                                 F : Type u_3
                                                                 inst✝¹ : NormedAddCommGroup F
                                                                 inst✝ : NormedSpace 𝕜 F
                                                                 c : F
                                                                 x : E
                                                                 x✝ : Prod E E
                                                                 ⊢ Eq 0 (HSub.hSub (HSub.hSub c c) (0 (HSub.hSub x✝.1 x✝.2)))
                                                               -/
  .of_isLittleO <| (isLittleO_zero _ _).congr_left fun _ => by simp only [zero_apply, sub_self]
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem hasFDerivAtFilter_const (c : F) (x : E) (L : Filter E) :
    HasFDerivAtFilter (fun _ => c) (0 : E →L[𝕜] F) x L :=
                                                               /-
                                                                 𝕜 : Type u_1
                                                                 inst✝⁴ : NontriviallyNormedField 𝕜
                                                                 E : Type u_2
                                                                 inst✝³ : NormedAddCommGroup E
                                                                 inst✝² : NormedSpace 𝕜 E
                                                                 F : Type u_3
                                                                 inst✝¹ : NormedAddCommGroup F
                                                                 inst✝ : NormedSpace 𝕜 F
                                                                 c : F
                                                                 x : E
                                                                 L : Filter E
                                                                 x✝ : E
                                                                 ⊢ Eq 0 (HSub.hSub (HSub.hSub c c) (0 (HSub.hSub x✝ x)))
                                                               -/
  .of_isLittleO <| (isLittleO_zero _ _).congr_left fun _ => by simp only [zero_apply, sub_self]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[fun_prop]
theorem hasFDerivWithinAt_const (c : F) (x : E) (s : Set E) :
    HasFDerivWithinAt (fun _ => c) (0 : E →L[𝕜] F) s x :=
  hasFDerivAtFilter_const _ _ _


@[fun_prop]
theorem hasFDerivAt_const (c : F) (x : E) : HasFDerivAt (fun _ => c) (0 : E →L[𝕜] F) x :=
  hasFDerivAtFilter_const _ _ _


@[simp, fun_prop]
theorem differentiableAt_const (c : F) : DifferentiableAt 𝕜 (fun _ => c) x :=
  ⟨0, hasFDerivAt_const c x⟩


@[fun_prop]
theorem differentiableWithinAt_const (c : F) : DifferentiableWithinAt 𝕜 (fun _ => c) s x :=
  DifferentiableAt.differentiableWithinAt (differentiableAt_const _)


theorem fderivWithin_const_apply (c : F) : fderivWithin 𝕜 (fun _ => c) s x = 0 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : E
    s : Set E
    c : F
    ⊢ Eq (fderivWithin 𝕜 (fun x => c) s x) 0
  -/
  rw [fderivWithin, if_pos]
  /-
    case hc
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : E
    s : Set E
    c : F
    ⊢ HasFDerivWithinAt (fun x => c) 0 s x
  -/
  apply hasFDerivWithinAt_const
  /-
    🎉 no goals
  -/


@[simp]
theorem fderivWithin_const (c : F) : fderivWithin 𝕜 (fun _ ↦ c) s = 0 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    c : F
    ⊢ Eq (fderivWithin 𝕜 (fun x => c) s) 0
  -/
  ext
  /-
    case h.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    c : F
    x✝¹ x✝ : E
    ⊢ Eq ((fderivWithin 𝕜 (fun x => c) s x✝¹) x✝) ((0 x✝¹) x✝)
  -/
  rw [fderivWithin_const_apply, Pi.zero_apply]
  /-
    🎉 no goals
  -/


theorem fderiv_const_apply (c : F) : fderiv 𝕜 (fun _ => c) x = 0 :=
  (hasFDerivAt_const c x).fderiv


@[simp]
theorem fderiv_const (c : F) : (fderiv 𝕜 fun _ : E => c) = 0 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    c : F
    ⊢ Eq (fderiv 𝕜 fun x => c) 0
  -/
  rw [← fderivWithin_univ, fderivWithin_const]
  /-
    🎉 no goals
  -/


@[simp, fun_prop]
theorem differentiable_const (c : F) : Differentiable 𝕜 fun _ : E => c := fun _ =>
  differentiableAt_const _


@[simp, fun_prop]
theorem differentiableOn_const (c : F) : DifferentiableOn 𝕜 (fun _ => c) s :=
  (differentiable_const _).differentiableOn


@[fun_prop]
theorem hasFDerivWithinAt_singleton (f : E → F) (x : E) :
    HasFDerivWithinAt f (0 : E →L[𝕜] F) {x} x := by
  simp only [HasFDerivWithinAt, nhdsWithin_singleton, hasFDerivAtFilter_iff_isLittleO,
    isLittleO_pure, ContinuousLinearMap.zero_apply, sub_self]


@[fun_prop]
theorem hasFDerivAt_of_subsingleton [h : Subsingleton E] (f : E → F) (x : E) :
    HasFDerivAt f (0 : E →L[𝕜] F) x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    h : Subsingleton E
    f : E → F
    x : E
    ⊢ HasFDerivAt f 0 x
  -/
  rw [← hasFDerivWithinAt_univ, subsingleton_univ.eq_singleton_of_mem (mem_univ x)]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    h : Subsingleton E
    f : E → F
    x : E
    ⊢ HasFDerivWithinAt f 0 (Singleton.singleton x) x
  -/
  exact hasFDerivWithinAt_singleton f x
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem differentiableOn_empty : DifferentiableOn 𝕜 f ∅ := fun _ => False.elim


@[fun_prop]
theorem differentiableOn_singleton : DifferentiableOn 𝕜 f {x} :=
  forall_eq.2 (hasFDerivWithinAt_singleton f x).differentiableWithinAt


@[fun_prop]
theorem Set.Subsingleton.differentiableOn (hs : s.Subsingleton) : DifferentiableOn 𝕜 f s :=
  hs.induction_on differentiableOn_empty fun _ => differentiableOn_singleton


theorem hasFDerivAt_zero_of_eventually_const (c : F) (hf : f =ᶠ[𝓝 x] fun _ => c) :
    HasFDerivAt f (0 : E →L[𝕜] F) x :=
  (hasFDerivAt_const _ _).congr_of_eventuallyEq hf


/-- Converse to the mean value inequality: if `f` is differentiable at `x₀` and `C`-lipschitz
on a neighborhood of `x₀` then its derivative at `x₀` has norm bounded by `C`. This version
only assumes that `‖f x - f x₀‖ ≤ C * ‖x - x₀‖` in a neighborhood of `x`. -/
theorem HasFDerivAt.le_of_lip' {f : E → F} {f' : E →L[𝕜] F} {x₀ : E} (hf : HasFDerivAt f f' x₀)
    {C : ℝ} (hC₀ : 0 ≤ C) (hlip : ∀ᶠ x in 𝓝 x₀, ‖f x - f x₀‖ ≤ C * ‖x - x₀‖) : ‖f'‖ ≤ C := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x₀ : E
    hf : HasFDerivAt f f' x₀
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    ⊢ LE.le (Norm.norm f') C
  -/
  refine le_of_forall_pos_le_add fun ε ε0 => opNorm_le_of_nhds_zero ?_ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      x₀ : E
      hf : HasFDerivAt f f' x₀
      C : Real
      hC₀ : LE.le 0 C
      hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
      ε : Real
      ε0 : LT.lt 0 ε
      ⊢ LE.le 0 (HAdd.hAdd C ε)
    -/
  · exact add_nonneg hC₀ ε0.le
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x₀ : E
    hf : HasFDerivAt f f' x₀
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    ε : Real
    ε0 : LT.lt 0 ε
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f' x)) (HMul.hMul (HAdd.hAdd C …
  -/
  rw [← map_add_left_nhds_zero x₀, eventually_map] at hlip
  /-
    case refine_2
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x₀ : E
    hf : HasFDerivAt f f' x₀
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun a => LE.le (Norm.norm (HSub.hSub (f (HAdd.hAdd x …
    ε : Real
    ε0 : LT.lt 0 ε
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f' x)) (HMul.hMul (HAdd.hAdd C …
  -/
  filter_upwards [isLittleO_iff.1 (hasFDerivAt_iff_isLittleO_nhds_zero.1 hf) ε0, hlip] with y hy hyC
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x₀ : E
    hf : HasFDerivAt f f' x₀
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun a => LE.le (Norm.norm (HSub.hSub (f (HAdd.hAdd x …
    ε : Real
    ε0 : LT.lt 0 ε
    y : E
    hy : LE.le (Norm.norm (HSub.hSub (HSub.hSub (f (HAdd.hAdd x₀ y)) (f x₀)) (f' y …
    hyC : LE.le (Norm.norm (HSub.hSub (f (HAdd.hAdd x₀ y)) (f x₀))) (HMul.hMul C ( …
    ⊢ LE.le (Norm.norm (f' y)) (HMul.hMul (HAdd.hAdd C ε) (Norm.norm y))
  -/
  rw [add_sub_cancel_left] at hyC
  calc
    ‖f' y‖ ≤ ‖f (x₀ + y) - f x₀‖ + ‖f (x₀ + y) - f x₀ - f' y‖ := norm_le_insert _ _
    _ ≤ C * ‖y‖ + ε * ‖y‖ := add_le_add hyC hy
    _ = (C + ε) * ‖y‖ := (add_mul _ _ _).symm


/-- Converse to the mean value inequality: if `f` is differentiable at `x₀` and `C`-lipschitz
on a neighborhood of `x₀` then its derivative at `x₀` has norm bounded by `C`. -/
theorem HasFDerivAt.le_of_lipschitzOn
    {f : E → F} {f' : E →L[𝕜] F} {x₀ : E} (hf : HasFDerivAt f f' x₀)
    {s : Set E} (hs : s ∈ 𝓝 x₀) {C : ℝ≥0} (hlip : LipschitzOnWith C f s) : ‖f'‖ ≤ C := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x₀ : E
    hf : HasFDerivAt f f' x₀
    s : Set E
    hs : Membership.mem (nhds x₀) s
    C : NNReal
    hlip : LipschitzOnWith C f s
    ⊢ LE.le (Norm.norm f') ↑C
  -/
  refine hf.le_of_lip' C.coe_nonneg ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x₀ : E
    hf : HasFDerivAt f f' x₀
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


/-- Converse to the mean value inequality: if `f` is differentiable at `x₀` and `C`-lipschitz
then its derivative at `x₀` has norm bounded by `C`. -/
theorem HasFDerivAt.le_of_lipschitz {f : E → F} {f' : E →L[𝕜] F} {x₀ : E} (hf : HasFDerivAt f f' x₀)
    {C : ℝ≥0} (hlip : LipschitzWith C f) : ‖f'‖ ≤ C :=
  hf.le_of_lipschitzOn univ_mem (lipschitzOnWith_univ.2 hlip)


/-- Converse to the mean value inequality: if `f` is `C`-lipschitz
on a neighborhood of `x₀` then its derivative at `x₀` has norm bounded by `C`. This version
only assumes that `‖f x - f x₀‖ ≤ C * ‖x - x₀‖` in a neighborhood of `x`. -/
theorem norm_fderiv_le_of_lip' {f : E → F} {x₀ : E}
    {C : ℝ} (hC₀ : 0 ≤ C) (hlip : ∀ᶠ x in 𝓝 x₀, ‖f x - f x₀‖ ≤ C * ‖x - x₀‖) :
    ‖fderiv 𝕜 f x₀‖ ≤ C := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x₀ : E
    C : Real
    hC₀ : LE.le 0 C
    hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
    ⊢ LE.le (Norm.norm (fderiv 𝕜 f x₀)) C
  -/
  by_cases hf : DifferentiableAt 𝕜 f x₀
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      x₀ : E
      C : Real
      hC₀ : LE.le 0 C
      hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
      hf : DifferentiableAt 𝕜 f x₀
      ⊢ LE.le (Norm.norm (fderiv 𝕜 f x₀)) C
    -/
  · exact hf.hasFDerivAt.le_of_lip' hC₀ hlip
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
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      x₀ : E
      C : Real
      hC₀ : LE.le 0 C
      hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
      hf : Not (DifferentiableAt 𝕜 f x₀)
      ⊢ LE.le (Norm.norm (fderiv 𝕜 f x₀)) C
    -/
  · rw [fderiv_zero_of_not_differentiableAt hf]
    /-
      case neg
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      x₀ : E
      C : Real
      hC₀ : LE.le 0 C
      hlip : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (f x) (f x₀)))  …
      hf : Not (DifferentiableAt 𝕜 f x₀)
      ⊢ LE.le (Norm.norm 0) C
    -/
    simp [hC₀]
    /-
      🎉 no goals
    -/


/-- Converse to the mean value inequality: if `f` is `C`-lipschitz
on a neighborhood of `x₀` then its derivative at `x₀` has norm bounded by `C`.
Version using `fderiv`. -/
-- Porting note: renamed so that dot-notation makes sense
theorem norm_fderiv_le_of_lipschitzOn {f : E → F} {x₀ : E} {s : Set E} (hs : s ∈ 𝓝 x₀)
    {C : ℝ≥0} (hlip : LipschitzOnWith C f s) : ‖fderiv 𝕜 f x₀‖ ≤ C := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x₀ : E
    s : Set E
    hs : Membership.mem (nhds x₀) s
    C : NNReal
    hlip : LipschitzOnWith C f s
    ⊢ LE.le (Norm.norm (fderiv 𝕜 f x₀)) ↑C
  -/
  refine norm_fderiv_le_of_lip' 𝕜 C.coe_nonneg ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
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
its derivative at `x₀` has norm bounded by `C`.
Version using `fderiv`. -/
theorem norm_fderiv_le_of_lipschitz {f : E → F} {x₀ : E}
    {C : ℝ≥0} (hlip : LipschitzWith C f) : ‖fderiv 𝕜 f x₀‖ ≤ C :=
  norm_fderiv_le_of_lipschitzOn 𝕜 univ_mem (lipschitzOnWith_univ.2 hlip)


theorem HasStrictFDerivAt.of_nmem_tsupport (h : x ∉ tsupport f) :
    HasStrictFDerivAt f (0 : E →L[𝕜] F) x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    h : Not (Membership.mem (tsupport f) x)
    ⊢ HasStrictFDerivAt f 0 x
  -/
  rw [not_mem_tsupport_iff_eventuallyEq] at h
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    h : (nhds x).EventuallyEq f 0
    ⊢ HasStrictFDerivAt f 0 x
  -/
  exact (hasStrictFDerivAt_const (0 : F) x).congr_of_eventuallyEq h.symm
  /-
    🎉 no goals
  -/


theorem HasFDerivAt.of_nmem_tsupport (h : x ∉ tsupport f) :
    HasFDerivAt f (0 : E →L[𝕜] F) x :=
  (HasStrictFDerivAt.of_nmem_tsupport 𝕜 h).hasFDerivAt


theorem HasFDerivWithinAt.of_not_mem_tsupport {s : Set E} {x : E} (h : x ∉ tsupport f) :
    HasFDerivWithinAt f (0 : E →L[𝕜] F) s x :=
  (HasFDerivAt.of_nmem_tsupport 𝕜 h).hasFDerivWithinAt


theorem fderiv_of_not_mem_tsupport (h : x ∉ tsupport f) : fderiv 𝕜 f x = 0 :=
  (HasFDerivAt.of_nmem_tsupport 𝕜 h).fderiv


theorem support_fderiv_subset : support (fderiv 𝕜 f) ⊆ tsupport f := fun x ↦ by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    ⊢ Membership.mem (Function.support (fderiv 𝕜 f)) x → Membership.mem (tsupport  …
  -/
  rw [← not_imp_not, nmem_support]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    ⊢ Not (Membership.mem (tsupport f) x) → Eq (fderiv 𝕜 f x) 0
  -/
  exact fderiv_of_not_mem_tsupport _
  /-
    🎉 no goals
  -/


theorem tsupport_fderiv_subset : tsupport (fderiv 𝕜 f) ⊆ tsupport f :=
  closure_minimal (support_fderiv_subset 𝕜) isClosed_closure


protected theorem HasCompactSupport.fderiv (hf : HasCompactSupport f) :
    HasCompactSupport (fderiv 𝕜 f) :=
  hf.mono' <| support_fderiv_subset 𝕜


protected theorem HasCompactSupport.fderiv_apply (hf : HasCompactSupport f) (v : E) :
    HasCompactSupport (fderiv 𝕜 f · v) :=
  hf.fderiv 𝕜 |>.comp_left (g := fun L : E →L[𝕜] F ↦ L v) rfl


