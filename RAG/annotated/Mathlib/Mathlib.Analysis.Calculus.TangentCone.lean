/-- The set of all tangent directions to the set `s` at the point `x`. -/
def tangentConeAt (s : Set E) (x : E) : Set E :=
  { y : E | ∃ (c : ℕ → 𝕜) (d : ℕ → E),
    (∀ᶠ n in atTop, x + d n ∈ s) ∧
    Tendsto (fun n => ‖c n‖) atTop atTop ∧
    Tendsto (fun n => c n • d n) atTop (𝓝 y) }


/-- A property ensuring that the tangent cone to `s` at `x` spans a dense subset of the whole space.
The main role of this property is to ensure that the differential within `s` at `x` is unique,
hence this name. The uniqueness it asserts is proved in `UniqueDiffWithinAt.eq` in
`Mathlib.Analysis.Calculus.FDeriv.Basic`.
To avoid pathologies in dimension 0, we also require that `x` belongs to the closure of `s` (which
is automatic when `E` is not `0`-dimensional). -/
@[mk_iff]
structure UniqueDiffWithinAt (s : Set E) (x : E) : Prop where
  dense_tangentCone : Dense (Submodule.span 𝕜 (tangentConeAt 𝕜 s x) : Set E)
  mem_closure : x ∈ closure s


/-- A property ensuring that the tangent cone to `s` at any of its points spans a dense subset of
the whole space. The main role of this property is to ensure that the differential along `s` is
unique, hence this name. The uniqueness it asserts is proved in `UniqueDiffOn.eq` in
`Mathlib.Analysis.Calculus.FDeriv.Basic`. -/
def UniqueDiffOn (s : Set E) : Prop :=
  ∀ x ∈ s, UniqueDiffWithinAt 𝕜 s x


theorem mem_tangentConeAt_of_pow_smul {r : 𝕜} (hr₀ : r ≠ 0) (hr : ‖r‖ < 1)
    (hs : ∀ᶠ n : ℕ in atTop, x + r ^ n • y ∈ s) : y ∈ tangentConeAt 𝕜 s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x y : E
    s : Set E
    r : 𝕜
    hr₀ : Ne r 0
    hr : LT.lt (Norm.norm r) 1
    hs : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (HSMul.hSMul (H …
    ⊢ Membership.mem (tangentConeAt 𝕜 s x) y
  -/
  refine ⟨fun n ↦ (r ^ n)⁻¹, fun n ↦ r ^ n • y, hs, ?_, ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x y : E
      s : Set E
      r : 𝕜
      hr₀ : Ne r 0
      hr : LT.lt (Norm.norm r) 1
      hs : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (HSMul.hSMul (H …
      ⊢ Filter.Tendsto (fun n => Norm.norm ((fun n => Inv.inv (HPow.hPow r n)) n)) F …
    -/
  · simp only [norm_inv, norm_pow, ← inv_pow]
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x y : E
      s : Set E
      r : 𝕜
      hr₀ : Ne r 0
      hr : LT.lt (Norm.norm r) 1
      hs : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (HSMul.hSMul (H …
      ⊢ Filter.Tendsto (fun n => HPow.hPow (Inv.inv (Norm.norm r)) n) Filter.atTop F …
    -/
    exact tendsto_pow_atTop_atTop_of_one_lt <| (one_lt_inv₀ (norm_pos_iff.2 hr₀)).2 hr
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x y : E
      s : Set E
      r : 𝕜
      hr₀ : Ne r 0
      hr : LT.lt (Norm.norm r) 1
      hs : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (HSMul.hSMul (H …
      ⊢ Filter.Tendsto (fun n => HSMul.hSMul ((fun n => Inv.inv (HPow.hPow r n)) n)  …
    -/
  · simp only [inv_smul_smul₀ (pow_ne_zero _ hr₀), tendsto_const_nhds]
    /-
      🎉 no goals
    -/


theorem tangentCone_univ : tangentConeAt 𝕜 univ x = univ :=
  let ⟨_r, hr₀, hr⟩ := exists_norm_lt_one 𝕜
  eq_univ_of_forall fun _ ↦ mem_tangentConeAt_of_pow_smul (norm_pos_iff.1 hr₀) hr <|
    Eventually.of_forall fun _ ↦ mem_univ _


theorem tangentCone_mono (h : s ⊆ t) : tangentConeAt 𝕜 s x ⊆ tangentConeAt 𝕜 t x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s t : Set E
    h : HasSubset.Subset s t
    ⊢ HasSubset.Subset (tangentConeAt 𝕜 s x) (tangentConeAt 𝕜 t x)
  -/
  rintro y ⟨c, d, ds, ctop, clim⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s t : Set E
    h : HasSubset.Subset s t
    y : E
    c : Nat → 𝕜
    d : Nat → E
    ds : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (d n))) Filter. …
    ctop : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
    clim : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds y)
    ⊢ Membership.mem (tangentConeAt 𝕜 t x) y
  -/
  exact ⟨c, d, mem_of_superset ds fun n hn => h hn, ctop, clim⟩
  /-
    🎉 no goals
  -/


/-- Auxiliary lemma ensuring that, under the assumptions defining the tangent cone,
the sequence `d` tends to 0 at infinity. -/
theorem tangentConeAt.lim_zero {α : Type*} (l : Filter α) {c : α → 𝕜} {d : α → E}
    (hc : Tendsto (fun n => ‖c n‖) l atTop) (hd : Tendsto (fun n => c n • d n) l (𝓝 y)) :
    Tendsto d l (𝓝 0) := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    y : E
    α : Type u_5
    l : Filter α
    c : α → 𝕜
    d : α → E
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) l Filter.atTop
    hd : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) l (nhds y)
    ⊢ Filter.Tendsto d l (nhds 0)
  -/
  have A : Tendsto (fun n => ‖c n‖⁻¹) l (𝓝 0) := tendsto_inv_atTop_zero.comp hc
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    y : E
    α : Type u_5
    l : Filter α
    c : α → 𝕜
    d : α → E
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) l Filter.atTop
    hd : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) l (nhds y)
    A : Filter.Tendsto (fun n => Inv.inv (Norm.norm (c n))) l (nhds 0)
    ⊢ Filter.Tendsto d l (nhds 0)
  -/
  have B : Tendsto (fun n => ‖c n • d n‖) l (𝓝 ‖y‖) := (continuous_norm.tendsto _).comp hd
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    y : E
    α : Type u_5
    l : Filter α
    c : α → 𝕜
    d : α → E
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) l Filter.atTop
    hd : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) l (nhds y)
    A : Filter.Tendsto (fun n => Inv.inv (Norm.norm (c n))) l (nhds 0)
    B : Filter.Tendsto (fun n => Norm.norm (HSMul.hSMul (c n) (d n))) l (nhds (Nor …
    ⊢ Filter.Tendsto d l (nhds 0)
  -/
  have C : Tendsto (fun n => ‖c n‖⁻¹ * ‖c n • d n‖) l (𝓝 (0 * ‖y‖)) := A.mul B
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    y : E
    α : Type u_5
    l : Filter α
    c : α → 𝕜
    d : α → E
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) l Filter.atTop
    hd : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) l (nhds y)
    A : Filter.Tendsto (fun n => Inv.inv (Norm.norm (c n))) l (nhds 0)
    B : Filter.Tendsto (fun n => Norm.norm (HSMul.hSMul (c n) (d n))) l (nhds (Nor …
    C : Filter.Tendsto (fun n => HMul.hMul (Inv.inv (Norm.norm (c n))) (Norm.norm  …
    ⊢ Filter.Tendsto d l (nhds 0)
  -/
  rw [zero_mul] at C
  have : ∀ᶠ n in l, ‖c n‖⁻¹ * ‖c n • d n‖ = ‖d n‖ := by
    refine (eventually_ne_of_tendsto_norm_atTop hc 0).mono fun n hn => ?_
    rw [norm_smul, ← mul_assoc, inv_mul_cancel₀, one_mul]
    rwa [Ne, norm_eq_zero]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    y : E
    α : Type u_5
    l : Filter α
    c : α → 𝕜
    d : α → E
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) l Filter.atTop
    hd : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) l (nhds y)
    A : Filter.Tendsto (fun n => Inv.inv (Norm.norm (c n))) l (nhds 0)
    B : Filter.Tendsto (fun n => Norm.norm (HSMul.hSMul (c n) (d n))) l (nhds (Nor …
    C : Filter.Tendsto (fun n => HMul.hMul (Inv.inv (Norm.norm (c n))) (Norm.norm  …
    this : Filter.Eventually (fun n => Eq (HMul.hMul (Inv.inv (Norm.norm (c n))) ( …
    ⊢ Filter.Tendsto d l (nhds 0)
  -/
  have D : Tendsto (fun n => ‖d n‖) l (𝓝 0) := Tendsto.congr' this C
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    y : E
    α : Type u_5
    l : Filter α
    c : α → 𝕜
    d : α → E
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) l Filter.atTop
    hd : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) l (nhds y)
    A : Filter.Tendsto (fun n => Inv.inv (Norm.norm (c n))) l (nhds 0)
    B : Filter.Tendsto (fun n => Norm.norm (HSMul.hSMul (c n) (d n))) l (nhds (Nor …
    C : Filter.Tendsto (fun n => HMul.hMul (Inv.inv (Norm.norm (c n))) (Norm.norm  …
    this : Filter.Eventually (fun n => Eq (HMul.hMul (Inv.inv (Norm.norm (c n))) ( …
    D : Filter.Tendsto (fun n => Norm.norm (d n)) l (nhds 0)
    ⊢ Filter.Tendsto d l (nhds 0)
  -/
  rw [tendsto_zero_iff_norm_tendsto_zero]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    y : E
    α : Type u_5
    l : Filter α
    c : α → 𝕜
    d : α → E
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) l Filter.atTop
    hd : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) l (nhds y)
    A : Filter.Tendsto (fun n => Inv.inv (Norm.norm (c n))) l (nhds 0)
    B : Filter.Tendsto (fun n => Norm.norm (HSMul.hSMul (c n) (d n))) l (nhds (Nor …
    C : Filter.Tendsto (fun n => HMul.hMul (Inv.inv (Norm.norm (c n))) (Norm.norm  …
    this : Filter.Eventually (fun n => Eq (HMul.hMul (Inv.inv (Norm.norm (c n))) ( …
    D : Filter.Tendsto (fun n => Norm.norm (d n)) l (nhds 0)
    ⊢ Filter.Tendsto (fun x => Norm.norm (d x)) l (nhds 0)
  -/
  exact D
  /-
    🎉 no goals
  -/


theorem tangentCone_mono_nhds (h : 𝓝[s] x ≤ 𝓝[t] x) :
    tangentConeAt 𝕜 s x ⊆ tangentConeAt 𝕜 t x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s t : Set E
    h : LE.le (nhdsWithin x s) (nhdsWithin x t)
    ⊢ HasSubset.Subset (tangentConeAt 𝕜 s x) (tangentConeAt 𝕜 t x)
  -/
  rintro y ⟨c, d, ds, ctop, clim⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s t : Set E
    h : LE.le (nhdsWithin x s) (nhdsWithin x t)
    y : E
    c : Nat → 𝕜
    d : Nat → E
    ds : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (d n))) Filter. …
    ctop : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
    clim : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds y)
    ⊢ Membership.mem (tangentConeAt 𝕜 t x) y
  -/
  refine ⟨c, d, ?_, ctop, clim⟩
  suffices Tendsto (fun n => x + d n) atTop (𝓝[t] x) from
    tendsto_principal.1 (tendsto_inf.1 this).2
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s t : Set E
    h : LE.le (nhdsWithin x s) (nhdsWithin x t)
    y : E
    c : Nat → 𝕜
    d : Nat → E
    ds : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (d n))) Filter. …
    ctop : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
    clim : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds y)
    ⊢ Filter.Tendsto (fun n => HAdd.hAdd x (d n)) Filter.atTop (nhdsWithin x t)
  -/
  refine (tendsto_inf.2 ⟨?_, tendsto_principal.2 ds⟩).mono_right h
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s t : Set E
    h : LE.le (nhdsWithin x s) (nhdsWithin x t)
    y : E
    c : Nat → 𝕜
    d : Nat → E
    ds : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (d n))) Filter. …
    ctop : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
    clim : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds y)
    ⊢ Filter.Tendsto (fun a => HAdd.hAdd x (d a)) Filter.atTop (nhds x)
  -/
  simpa only [add_zero] using tendsto_const_nhds.add (tangentConeAt.lim_zero atTop ctop clim)
  /-
    🎉 no goals
  -/


/-- Tangent cone of `s` at `x` depends only on `𝓝[s] x`. -/
theorem tangentCone_congr (h : 𝓝[s] x = 𝓝[t] x) : tangentConeAt 𝕜 s x = tangentConeAt 𝕜 t x :=
  Subset.antisymm (tangentCone_mono_nhds <| le_of_eq h) (tangentCone_mono_nhds <| le_of_eq h.symm)


/-- Intersecting with a neighborhood of the point does not change the tangent cone. -/
theorem tangentCone_inter_nhds (ht : t ∈ 𝓝 x) : tangentConeAt 𝕜 (s ∩ t) x = tangentConeAt 𝕜 s x :=
  tangentCone_congr (nhdsWithin_restrict' _ ht).symm


/-- The tangent cone of a product contains the tangent cone of its left factor. -/
theorem subset_tangentCone_prod_left {t : Set F} {y : F} (ht : y ∈ closure t) :
    LinearMap.inl 𝕜 E F '' tangentConeAt 𝕜 s x ⊆ tangentConeAt 𝕜 (s ×ˢ t) (x, y) := by
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
    t : Set F
    y : F
    ht : Membership.mem (closure t) y
    ⊢ HasSubset.Subset (Set.image (⇑(LinearMap.inl 𝕜 E F)) (tangentConeAt 𝕜 s x))  …
  -/
  rintro _ ⟨v, ⟨c, d, hd, hc, hy⟩, rfl⟩
  have : ∀ n, ∃ d', y + d' ∈ t ∧ ‖c n • d'‖ < ((1 : ℝ) / 2) ^ n := by
    intro n
    rcases mem_closure_iff_nhds.1 ht _
        (eventually_nhds_norm_smul_sub_lt (c n) y (pow_pos one_half_pos n)) with
      ⟨z, hz, hzt⟩
    exact ⟨z - y, by simpa using hzt, by simpa using hz⟩
  /-
    case intro.intro.intro.intro.intro.intro
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
    t : Set F
    y : F
    ht : Membership.mem (closure t) y
    v : E
    c : Nat → 𝕜
    d : Nat → E
    hd : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (d n))) Filter. …
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
    hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds v)
    this : ∀ (n : Nat), Exists fun d' => And (Membership.mem t (HAdd.hAdd y d')) ( …
    ⊢ Membership.mem (tangentConeAt 𝕜 (SProd.sprod s t) { fst := x, snd := y }) (( …
  -/
  choose d' hd' using this
  /-
    case intro.intro.intro.intro.intro.intro
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
    t : Set F
    y : F
    ht : Membership.mem (closure t) y
    v : E
    c : Nat → 𝕜
    d : Nat → E
    hd : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (d n))) Filter. …
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
    hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds v)
    d' : Nat → F
    hd' : ∀ (n : Nat), And (Membership.mem t (HAdd.hAdd y (d' n))) (LT.lt (Norm.no …
    ⊢ Membership.mem (tangentConeAt 𝕜 (SProd.sprod s t) { fst := x, snd := y }) (( …
  -/
  refine ⟨c, fun n => (d n, d' n), ?_, hc, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
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
      t : Set F
      y : F
      ht : Membership.mem (closure t) y
      v : E
      c : Nat → 𝕜
      d : Nat → E
      hd : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (d n))) Filter. …
      hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
      hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds v)
      d' : Nat → F
      hd' : ∀ (n : Nat), And (Membership.mem t (HAdd.hAdd y (d' n))) (LT.lt (Norm.no …
      ⊢ Filter.Eventually (fun n => Membership.mem (SProd.sprod s t) (HAdd.hAdd { fs …
    -/
  · show ∀ᶠ n in atTop, (x, y) + (d n, d' n) ∈ s ×ˢ t
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
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
      t : Set F
      y : F
      ht : Membership.mem (closure t) y
      v : E
      c : Nat → 𝕜
      d : Nat → E
      hd : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (d n))) Filter. …
      hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
      hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds v)
      d' : Nat → F
      hd' : ∀ (n : Nat), And (Membership.mem t (HAdd.hAdd y (d' n))) (LT.lt (Norm.no …
      ⊢ Filter.Eventually (fun n => Membership.mem (SProd.sprod s t) (HAdd.hAdd { fs …
    -/
    filter_upwards [hd] with n hn
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
      x : E
      s : Set E
      t : Set F
      y : F
      ht : Membership.mem (closure t) y
      v : E
      c : Nat → 𝕜
      d : Nat → E
      hd : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (d n))) Filter. …
      hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
      hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds v)
      d' : Nat → F
      hd' : ∀ (n : Nat), And (Membership.mem t (HAdd.hAdd y (d' n))) (LT.lt (Norm.no …
      n : Nat
      hn : Membership.mem s (HAdd.hAdd x (d n))
      ⊢ Membership.mem (SProd.sprod s t) (HAdd.hAdd { fst := x, snd := y } { fst :=  …
    -/
    simp [hn, (hd' n).1]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
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
      t : Set F
      y : F
      ht : Membership.mem (closure t) y
      v : E
      c : Nat → 𝕜
      d : Nat → E
      hd : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (d n))) Filter. …
      hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
      hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds v)
      d' : Nat → F
      hd' : ∀ (n : Nat), And (Membership.mem t (HAdd.hAdd y (d' n))) (LT.lt (Norm.no …
      ⊢ Filter.Tendsto (fun n => HSMul.hSMul (c n) ((fun n => { fst := d n, snd := d …
    -/
  · apply Tendsto.prod_mk_nhds hy _
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
      t : Set F
      y : F
      ht : Membership.mem (closure t) y
      v : E
      c : Nat → 𝕜
      d : Nat → E
      hd : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (d n))) Filter. …
      hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
      hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds v)
      d' : Nat → F
      hd' : ∀ (n : Nat), And (Membership.mem t (HAdd.hAdd y (d' n))) (LT.lt (Norm.no …
      ⊢ Filter.Tendsto (fun c_1 => HSMul.hSMul (c c_1) ((fun n => { fst := d n, snd  …
    -/
    refine squeeze_zero_norm (fun n => (hd' n).2.le) ?_
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
      t : Set F
      y : F
      ht : Membership.mem (closure t) y
      v : E
      c : Nat → 𝕜
      d : Nat → E
      hd : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (d n))) Filter. …
      hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
      hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds v)
      d' : Nat → F
      hd' : ∀ (n : Nat), And (Membership.mem t (HAdd.hAdd y (d' n))) (LT.lt (Norm.no …
      ⊢ Filter.Tendsto (HPow.hPow (1 / 2)) Filter.atTop (nhds 0)
    -/
    exact tendsto_pow_atTop_nhds_zero_of_lt_one one_half_pos.le one_half_lt_one
    /-
      🎉 no goals
    -/


/-- The tangent cone of a product contains the tangent cone of its right factor. -/
theorem subset_tangentCone_prod_right {t : Set F} {y : F} (hs : x ∈ closure s) :
    LinearMap.inr 𝕜 E F '' tangentConeAt 𝕜 t y ⊆ tangentConeAt 𝕜 (s ×ˢ t) (x, y) := by
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
    t : Set F
    y : F
    hs : Membership.mem (closure s) x
    ⊢ HasSubset.Subset (Set.image (⇑(LinearMap.inr 𝕜 E F)) (tangentConeAt 𝕜 t y))  …
  -/
  rintro _ ⟨w, ⟨c, d, hd, hc, hy⟩, rfl⟩
  have : ∀ n, ∃ d', x + d' ∈ s ∧ ‖c n • d'‖ < ((1 : ℝ) / 2) ^ n := by
    intro n
    rcases mem_closure_iff_nhds.1 hs _
        (eventually_nhds_norm_smul_sub_lt (c n) x (pow_pos one_half_pos n)) with
      ⟨z, hz, hzs⟩
    exact ⟨z - x, by simpa using hzs, by simpa using hz⟩
  /-
    case intro.intro.intro.intro.intro.intro
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
    t : Set F
    y : F
    hs : Membership.mem (closure s) x
    w : F
    c : Nat → 𝕜
    d : Nat → F
    hd : Filter.Eventually (fun n => Membership.mem t (HAdd.hAdd y (d n))) Filter. …
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
    hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds w)
    this : ∀ (n : Nat), Exists fun d' => And (Membership.mem s (HAdd.hAdd x d')) ( …
    ⊢ Membership.mem (tangentConeAt 𝕜 (SProd.sprod s t) { fst := x, snd := y }) (( …
  -/
  choose d' hd' using this
  /-
    case intro.intro.intro.intro.intro.intro
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
    t : Set F
    y : F
    hs : Membership.mem (closure s) x
    w : F
    c : Nat → 𝕜
    d : Nat → F
    hd : Filter.Eventually (fun n => Membership.mem t (HAdd.hAdd y (d n))) Filter. …
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
    hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds w)
    d' : Nat → E
    hd' : ∀ (n : Nat), And (Membership.mem s (HAdd.hAdd x (d' n))) (LT.lt (Norm.no …
    ⊢ Membership.mem (tangentConeAt 𝕜 (SProd.sprod s t) { fst := x, snd := y }) (( …
  -/
  refine ⟨c, fun n => (d' n, d n), ?_, hc, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
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
      t : Set F
      y : F
      hs : Membership.mem (closure s) x
      w : F
      c : Nat → 𝕜
      d : Nat → F
      hd : Filter.Eventually (fun n => Membership.mem t (HAdd.hAdd y (d n))) Filter. …
      hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
      hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds w)
      d' : Nat → E
      hd' : ∀ (n : Nat), And (Membership.mem s (HAdd.hAdd x (d' n))) (LT.lt (Norm.no …
      ⊢ Filter.Eventually (fun n => Membership.mem (SProd.sprod s t) (HAdd.hAdd { fs …
    -/
  · show ∀ᶠ n in atTop, (x, y) + (d' n, d n) ∈ s ×ˢ t
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
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
      t : Set F
      y : F
      hs : Membership.mem (closure s) x
      w : F
      c : Nat → 𝕜
      d : Nat → F
      hd : Filter.Eventually (fun n => Membership.mem t (HAdd.hAdd y (d n))) Filter. …
      hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
      hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds w)
      d' : Nat → E
      hd' : ∀ (n : Nat), And (Membership.mem s (HAdd.hAdd x (d' n))) (LT.lt (Norm.no …
      ⊢ Filter.Eventually (fun n => Membership.mem (SProd.sprod s t) (HAdd.hAdd { fs …
    -/
    filter_upwards [hd] with n hn
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
      x : E
      s : Set E
      t : Set F
      y : F
      hs : Membership.mem (closure s) x
      w : F
      c : Nat → 𝕜
      d : Nat → F
      hd : Filter.Eventually (fun n => Membership.mem t (HAdd.hAdd y (d n))) Filter. …
      hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
      hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds w)
      d' : Nat → E
      hd' : ∀ (n : Nat), And (Membership.mem s (HAdd.hAdd x (d' n))) (LT.lt (Norm.no …
      n : Nat
      hn : Membership.mem t (HAdd.hAdd y (d n))
      ⊢ Membership.mem (SProd.sprod s t) (HAdd.hAdd { fst := x, snd := y } { fst :=  …
    -/
    simp [hn, (hd' n).1]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
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
      t : Set F
      y : F
      hs : Membership.mem (closure s) x
      w : F
      c : Nat → 𝕜
      d : Nat → F
      hd : Filter.Eventually (fun n => Membership.mem t (HAdd.hAdd y (d n))) Filter. …
      hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
      hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds w)
      d' : Nat → E
      hd' : ∀ (n : Nat), And (Membership.mem s (HAdd.hAdd x (d' n))) (LT.lt (Norm.no …
      ⊢ Filter.Tendsto (fun n => HSMul.hSMul (c n) ((fun n => { fst := d' n, snd :=  …
    -/
  · apply Tendsto.prod_mk_nhds _ hy
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
      t : Set F
      y : F
      hs : Membership.mem (closure s) x
      w : F
      c : Nat → 𝕜
      d : Nat → F
      hd : Filter.Eventually (fun n => Membership.mem t (HAdd.hAdd y (d n))) Filter. …
      hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
      hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds w)
      d' : Nat → E
      hd' : ∀ (n : Nat), And (Membership.mem s (HAdd.hAdd x (d' n))) (LT.lt (Norm.no …
      ⊢ Filter.Tendsto (fun c_1 => HSMul.hSMul (c c_1) ((fun n => { fst := d' n, snd …
    -/
    refine squeeze_zero_norm (fun n => (hd' n).2.le) ?_
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
      t : Set F
      y : F
      hs : Membership.mem (closure s) x
      w : F
      c : Nat → 𝕜
      d : Nat → F
      hd : Filter.Eventually (fun n => Membership.mem t (HAdd.hAdd y (d n))) Filter. …
      hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
      hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds w)
      d' : Nat → E
      hd' : ∀ (n : Nat), And (Membership.mem s (HAdd.hAdd x (d' n))) (LT.lt (Norm.no …
      ⊢ Filter.Tendsto (HPow.hPow (1 / 2)) Filter.atTop (nhds 0)
    -/
    exact tendsto_pow_atTop_nhds_zero_of_lt_one one_half_pos.le one_half_lt_one
    /-
      🎉 no goals
    -/


/-- The tangent cone of a product contains the tangent cone of each factor. -/
theorem mapsTo_tangentCone_pi {ι : Type*} [DecidableEq ι] {E : ι → Type*}
    [∀ i, NormedAddCommGroup (E i)] [∀ i, NormedSpace 𝕜 (E i)] {s : ∀ i, Set (E i)} {x : ∀ i, E i}
    {i : ι} (hi : ∀ j ≠ i, x j ∈ closure (s j)) :
    MapsTo (LinearMap.single 𝕜 E i) (tangentConeAt 𝕜 (s i) (x i))
      (tangentConeAt 𝕜 (Set.pi univ s) x) := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    ι : Type u_5
    inst✝² : DecidableEq ι
    E : ι → Type u_6
    inst✝¹ : (i : ι) → NormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    s : (i : ι) → Set (E i)
    x : (i : ι) → E i
    i : ι
    hi : ∀ (j : ι), Ne j i → Membership.mem (closure (s j)) (x j)
    ⊢ Set.MapsTo (⇑(LinearMap.single 𝕜 E i)) (tangentConeAt 𝕜 (s i) (x i)) (tangen …
  -/
  rintro w ⟨c, d, hd, hc, hy⟩
  have : ∀ n, ∀ j ≠ i, ∃ d', x j + d' ∈ s j ∧ ‖c n • d'‖ < (1 / 2 : ℝ) ^ n := fun n j hj ↦ by
    rcases mem_closure_iff_nhds.1 (hi j hj) _
        (eventually_nhds_norm_smul_sub_lt (c n) (x j) (pow_pos one_half_pos n)) with
      ⟨z, hz, hzs⟩
    exact ⟨z - x j, by simpa using hzs, by simpa using hz⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    ι : Type u_5
    inst✝² : DecidableEq ι
    E : ι → Type u_6
    inst✝¹ : (i : ι) → NormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    s : (i : ι) → Set (E i)
    x : (i : ι) → E i
    i : ι
    hi : ∀ (j : ι), Ne j i → Membership.mem (closure (s j)) (x j)
    w : E i
    c : Nat → 𝕜
    d : Nat → E i
    hd : Filter.Eventually (fun n => Membership.mem (s i) (HAdd.hAdd (x i) (d n))) …
    hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
    hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds w)
    this : ∀ (n : Nat) (j : ι), Ne j i → Exists fun d' => And (Membership.mem (s j …
    ⊢ Membership.mem (tangentConeAt 𝕜 (Set.univ.pi s) x) ((LinearMap.single 𝕜 E i) …
  -/
  choose! d' hd's hcd' using this
  refine ⟨c, fun n => Function.update (d' n) i (d n), hd.mono fun n hn j _ => ?_, hc,
      tendsto_pi_nhds.2 fun j => ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      ι : Type u_5
      inst✝² : DecidableEq ι
      E : ι → Type u_6
      inst✝¹ : (i : ι) → NormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      s : (i : ι) → Set (E i)
      x : (i : ι) → E i
      i : ι
      hi : ∀ (j : ι), Ne j i → Membership.mem (closure (s j)) (x j)
      w : E i
      c : Nat → 𝕜
      d : Nat → E i
      hd : Filter.Eventually (fun n => Membership.mem (s i) (HAdd.hAdd (x i) (d n))) …
      hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
      hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds w)
      d' : Nat → (j : ι) → E j
      hd's : ∀ (n : Nat) (j : ι), Ne j i → Membership.mem (s j) (HAdd.hAdd (x j) (d' …
      hcd' : ∀ (n : Nat) (j : ι), Ne j i → LT.lt (Norm.norm (HSMul.hSMul (c n) (d' n …
      n : Nat
      hn : Membership.mem (s i) (HAdd.hAdd (x i) (d n))
      j : ι
      x✝ : Membership.mem Set.univ j
      ⊢ Membership.mem (s j) (HAdd.hAdd x ((fun n => Function.update (d' n) i (d n)) …
    -/
                                          /-
                                            🎉 no goals
                                          -/
  · rcases em (j = i) with (rfl | hj) <;> simp [*]
                                          /-
                                            🎉 no goals
                                          -/
    /-
      case intro.intro.intro.intro.refine_2
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      ι : Type u_5
      inst✝² : DecidableEq ι
      E : ι → Type u_6
      inst✝¹ : (i : ι) → NormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      s : (i : ι) → Set (E i)
      x : (i : ι) → E i
      i : ι
      hi : ∀ (j : ι), Ne j i → Membership.mem (closure (s j)) (x j)
      w : E i
      c : Nat → 𝕜
      d : Nat → E i
      hd : Filter.Eventually (fun n => Membership.mem (s i) (HAdd.hAdd (x i) (d n))) …
      hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
      hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds w)
      d' : Nat → (j : ι) → E j
      hd's : ∀ (n : Nat) (j : ι), Ne j i → Membership.mem (s j) (HAdd.hAdd (x j) (d' …
      hcd' : ∀ (n : Nat) (j : ι), Ne j i → LT.lt (Norm.norm (HSMul.hSMul (c n) (d' n …
      j : ι
      ⊢ Filter.Tendsto (fun i_1 => HSMul.hSMul (c i_1) ((fun n => Function.update (d …
    -/
  · rcases em (j = i) with (rfl | hj)
      /-
        case intro.intro.intro.intro.refine_2.inl
        𝕜 : Type u_1
        inst✝³ : NontriviallyNormedField 𝕜
        ι : Type u_5
        inst✝² : DecidableEq ι
        E : ι → Type u_6
        inst✝¹ : (i : ι) → NormedAddCommGroup (E i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
        s : (i : ι) → Set (E i)
        x : (i : ι) → E i
        c : Nat → 𝕜
        hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
        d' : Nat → (j : ι) → E j
        j : ι
        hi : ∀ (j_1 : ι), Ne j_1 j → Membership.mem (closure (s j_1)) (x j_1)
        w : E j
        d : Nat → E j
        hd : Filter.Eventually (fun n => Membership.mem (s j) (HAdd.hAdd (x j) (d n))) …
        hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds w)
        hd's : ∀ (n : Nat) (j_1 : ι), Ne j_1 j → Membership.mem (s j_1) (HAdd.hAdd (x  …
        hcd' : ∀ (n : Nat) (j_1 : ι), Ne j_1 j → LT.lt (Norm.norm (HSMul.hSMul (c n) ( …
        ⊢ Filter.Tendsto (fun i => HSMul.hSMul (c i) ((fun n => Function.update (d' n) …
      -/
    · simp [hy]
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.refine_2.inr
        𝕜 : Type u_1
        inst✝³ : NontriviallyNormedField 𝕜
        ι : Type u_5
        inst✝² : DecidableEq ι
        E : ι → Type u_6
        inst✝¹ : (i : ι) → NormedAddCommGroup (E i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
        s : (i : ι) → Set (E i)
        x : (i : ι) → E i
        i : ι
        hi : ∀ (j : ι), Ne j i → Membership.mem (closure (s j)) (x j)
        w : E i
        c : Nat → 𝕜
        d : Nat → E i
        hd : Filter.Eventually (fun n => Membership.mem (s i) (HAdd.hAdd (x i) (d n))) …
        hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
        hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds w)
        d' : Nat → (j : ι) → E j
        hd's : ∀ (n : Nat) (j : ι), Ne j i → Membership.mem (s j) (HAdd.hAdd (x j) (d' …
        hcd' : ∀ (n : Nat) (j : ι), Ne j i → LT.lt (Norm.norm (HSMul.hSMul (c n) (d' n …
        j : ι
        hj : Not (Eq j i)
        ⊢ Filter.Tendsto (fun i_1 => HSMul.hSMul (c i_1) ((fun n => Function.update (d …
      -/
    · suffices Tendsto (fun n => c n • d' n j) atTop (𝓝 0) by simpa [hj]
      /-
        case intro.intro.intro.intro.refine_2.inr
        𝕜 : Type u_1
        inst✝³ : NontriviallyNormedField 𝕜
        ι : Type u_5
        inst✝² : DecidableEq ι
        E : ι → Type u_6
        inst✝¹ : (i : ι) → NormedAddCommGroup (E i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
        s : (i : ι) → Set (E i)
        x : (i : ι) → E i
        i : ι
        hi : ∀ (j : ι), Ne j i → Membership.mem (closure (s j)) (x j)
        w : E i
        c : Nat → 𝕜
        d : Nat → E i
        hd : Filter.Eventually (fun n => Membership.mem (s i) (HAdd.hAdd (x i) (d n))) …
        hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
        hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds w)
        d' : Nat → (j : ι) → E j
        hd's : ∀ (n : Nat) (j : ι), Ne j i → Membership.mem (s j) (HAdd.hAdd (x j) (d' …
        hcd' : ∀ (n : Nat) (j : ι), Ne j i → LT.lt (Norm.norm (HSMul.hSMul (c n) (d' n …
        j : ι
        hj : Not (Eq j i)
        ⊢ Filter.Tendsto (fun n => HSMul.hSMul (c n) (d' n j)) Filter.atTop (nhds 0)
      -/
      refine squeeze_zero_norm (fun n => (hcd' n j hj).le) ?_
      /-
        case intro.intro.intro.intro.refine_2.inr
        𝕜 : Type u_1
        inst✝³ : NontriviallyNormedField 𝕜
        ι : Type u_5
        inst✝² : DecidableEq ι
        E : ι → Type u_6
        inst✝¹ : (i : ι) → NormedAddCommGroup (E i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
        s : (i : ι) → Set (E i)
        x : (i : ι) → E i
        i : ι
        hi : ∀ (j : ι), Ne j i → Membership.mem (closure (s j)) (x j)
        w : E i
        c : Nat → 𝕜
        d : Nat → E i
        hd : Filter.Eventually (fun n => Membership.mem (s i) (HAdd.hAdd (x i) (d n))) …
        hc : Filter.Tendsto (fun n => Norm.norm (c n)) Filter.atTop Filter.atTop
        hy : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds w)
        d' : Nat → (j : ι) → E j
        hd's : ∀ (n : Nat) (j : ι), Ne j i → Membership.mem (s j) (HAdd.hAdd (x j) (d' …
        hcd' : ∀ (n : Nat) (j : ι), Ne j i → LT.lt (Norm.norm (HSMul.hSMul (c n) (d' n …
        j : ι
        hj : Not (Eq j i)
        ⊢ Filter.Tendsto (HPow.hPow (1 / 2)) Filter.atTop (nhds 0)
      -/
      exact tendsto_pow_atTop_nhds_zero_of_lt_one one_half_pos.le one_half_lt_one
      /-
        🎉 no goals
      -/


/-- If a subset of a real vector space contains an open segment, then the direction of this
segment belongs to the tangent cone at its endpoints. -/
theorem mem_tangentCone_of_openSegment_subset {s : Set G} {x y : G} (h : openSegment ℝ x y ⊆ s) :
    y - x ∈ tangentConeAt ℝ s x := by
  /-
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    s : Set G
    x y : G
    h : HasSubset.Subset (openSegment Real x y) s
    ⊢ Membership.mem (tangentConeAt Real s x) (HSub.hSub y x)
  -/
  refine mem_tangentConeAt_of_pow_smul one_half_pos.ne' (by norm_num) ?_
  /-
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    s : Set G
    x y : G
    h : HasSubset.Subset (openSegment Real x y) s
    ⊢ Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd x (HSMul.hSMul (HPow …
  -/
  refine (eventually_ne_atTop 0).mono fun n hn ↦ (h ?_)
  /-
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    s : Set G
    x y : G
    h : HasSubset.Subset (openSegment Real x y) s
    n : Nat
    hn : Ne n 0
    ⊢ Membership.mem (openSegment Real x y) (HAdd.hAdd x (HSMul.hSMul (HPow.hPow ( …
  -/
  rw [openSegment_eq_image]
  /-
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    s : Set G
    x y : G
    h : HasSubset.Subset (openSegment Real x y) s
    n : Nat
    hn : Ne n 0
    ⊢ Membership.mem (Set.image (fun θ => HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 θ) x …
  -/
  refine ⟨(1 / 2) ^ n, ⟨?_, ?_⟩, ?_⟩
    /-
      case refine_1
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      s : Set G
      x y : G
      h : HasSubset.Subset (openSegment Real x y) s
      n : Nat
      hn : Ne n 0
      ⊢ LT.lt 0 (HPow.hPow (1 / 2) n)
    -/
  · exact pow_pos one_half_pos _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      s : Set G
      x y : G
      h : HasSubset.Subset (openSegment Real x y) s
      n : Nat
      hn : Ne n 0
      ⊢ LT.lt (HPow.hPow (1 / 2) n) 1
    -/
  · exact pow_lt_one₀ one_half_pos.le one_half_lt_one hn
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      s : Set G
      x y : G
      h : HasSubset.Subset (openSegment Real x y) s
      n : Nat
      hn : Ne n 0
      ⊢ Eq ((fun θ => HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 θ) x) (HSMul.hSMul θ y)) ( …
    -/
                                              /-
                                                🎉 no goals
                                              -/
  · simp only [sub_smul, one_smul, smul_sub]; abel
                                              /-
                                                🎉 no goals
                                              -/


/-- If a subset of a real vector space contains a segment, then the direction of this
segment belongs to the tangent cone at its endpoints. -/
theorem mem_tangentCone_of_segment_subset {s : Set G} {x y : G} (h : segment ℝ x y ⊆ s) :
    y - x ∈ tangentConeAt ℝ s x :=
  mem_tangentCone_of_openSegment_subset ((openSegment_subset_segment ℝ x y).trans h)


theorem UniqueDiffOn.uniqueDiffWithinAt {s : Set E} {x} (hs : UniqueDiffOn 𝕜 s) (h : x ∈ s) :
    UniqueDiffWithinAt 𝕜 s x :=
  hs x h


theorem uniqueDiffWithinAt_univ : UniqueDiffWithinAt 𝕜 univ x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    ⊢ UniqueDiffWithinAt 𝕜 Set.univ x
  -/
  rw [uniqueDiffWithinAt_iff, tangentCone_univ]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    ⊢ And (Dense ↑(Submodule.span 𝕜 Set.univ)) (Membership.mem (closure Set.univ) x)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem uniqueDiffOn_univ : UniqueDiffOn 𝕜 (univ : Set E) :=
  fun _ _ => uniqueDiffWithinAt_univ


theorem uniqueDiffOn_empty : UniqueDiffOn 𝕜 (∅ : Set E) :=
  fun _ hx => hx.elim


theorem UniqueDiffWithinAt.congr_pt (h : UniqueDiffWithinAt 𝕜 s x) (hy : x = y) :
    UniqueDiffWithinAt 𝕜 s y := hy ▸ h


theorem UniqueDiffWithinAt.mono_nhds (h : UniqueDiffWithinAt 𝕜 s x) (st : 𝓝[s] x ≤ 𝓝[t] x) :
    UniqueDiffWithinAt 𝕜 t x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s t : Set E
    h : UniqueDiffWithinAt 𝕜 s x
    st : LE.le (nhdsWithin x s) (nhdsWithin x t)
    ⊢ UniqueDiffWithinAt 𝕜 t x
  -/
  simp only [uniqueDiffWithinAt_iff] at *
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s t : Set E
    st : LE.le (nhdsWithin x s) (nhdsWithin x t)
    h : And (Dense ↑(Submodule.span 𝕜 (tangentConeAt 𝕜 s x))) (Membership.mem (clo …
    ⊢ And (Dense ↑(Submodule.span 𝕜 (tangentConeAt 𝕜 t x))) (Membership.mem (closu …
  -/
  rw [mem_closure_iff_nhdsWithin_neBot] at h ⊢
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s t : Set E
    st : LE.le (nhdsWithin x s) (nhdsWithin x t)
    h : And (Dense ↑(Submodule.span 𝕜 (tangentConeAt 𝕜 s x))) (nhdsWithin x s).NeBot
    ⊢ And (Dense ↑(Submodule.span 𝕜 (tangentConeAt 𝕜 t x))) (nhdsWithin x t).NeBot
  -/
  exact ⟨h.1.mono <| Submodule.span_mono <| tangentCone_mono_nhds st, h.2.mono st⟩
  /-
    🎉 no goals
  -/


theorem UniqueDiffWithinAt.mono (h : UniqueDiffWithinAt 𝕜 s x) (st : s ⊆ t) :
    UniqueDiffWithinAt 𝕜 t x :=
  h.mono_nhds <| nhdsWithin_mono _ st


theorem uniqueDiffWithinAt_congr (st : 𝓝[s] x = 𝓝[t] x) :
    UniqueDiffWithinAt 𝕜 s x ↔ UniqueDiffWithinAt 𝕜 t x :=
  ⟨fun h => h.mono_nhds <| le_of_eq st, fun h => h.mono_nhds <| le_of_eq st.symm⟩


theorem uniqueDiffWithinAt_inter (ht : t ∈ 𝓝 x) :
    UniqueDiffWithinAt 𝕜 (s ∩ t) x ↔ UniqueDiffWithinAt 𝕜 s x :=
  uniqueDiffWithinAt_congr <| (nhdsWithin_restrict' _ ht).symm


theorem UniqueDiffWithinAt.inter (hs : UniqueDiffWithinAt 𝕜 s x) (ht : t ∈ 𝓝 x) :
    UniqueDiffWithinAt 𝕜 (s ∩ t) x :=
  (uniqueDiffWithinAt_inter ht).2 hs


theorem uniqueDiffWithinAt_inter' (ht : t ∈ 𝓝[s] x) :
    UniqueDiffWithinAt 𝕜 (s ∩ t) x ↔ UniqueDiffWithinAt 𝕜 s x :=
  uniqueDiffWithinAt_congr <| (nhdsWithin_restrict'' _ ht).symm


theorem UniqueDiffWithinAt.inter' (hs : UniqueDiffWithinAt 𝕜 s x) (ht : t ∈ 𝓝[s] x) :
    UniqueDiffWithinAt 𝕜 (s ∩ t) x :=
  (uniqueDiffWithinAt_inter' ht).2 hs


theorem uniqueDiffWithinAt_of_mem_nhds (h : s ∈ 𝓝 x) : UniqueDiffWithinAt 𝕜 s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    h : Membership.mem (nhds x) s
    ⊢ UniqueDiffWithinAt 𝕜 s x
  -/
  simpa only [univ_inter] using uniqueDiffWithinAt_univ.inter h
  /-
    🎉 no goals
  -/


theorem IsOpen.uniqueDiffWithinAt (hs : IsOpen s) (xs : x ∈ s) : UniqueDiffWithinAt 𝕜 s x :=
  uniqueDiffWithinAt_of_mem_nhds (IsOpen.mem_nhds hs xs)


theorem UniqueDiffOn.inter (hs : UniqueDiffOn 𝕜 s) (ht : IsOpen t) : UniqueDiffOn 𝕜 (s ∩ t) :=
  fun x hx => (hs x hx.1).inter (IsOpen.mem_nhds ht hx.2)


theorem IsOpen.uniqueDiffOn (hs : IsOpen s) : UniqueDiffOn 𝕜 s :=
  fun _ hx => IsOpen.uniqueDiffWithinAt hs hx


/-- The product of two sets of unique differentiability at points `x` and `y` has unique
differentiability at `(x, y)`. -/
theorem UniqueDiffWithinAt.prod {t : Set F} {y : F} (hs : UniqueDiffWithinAt 𝕜 s x)
    (ht : UniqueDiffWithinAt 𝕜 t y) : UniqueDiffWithinAt 𝕜 (s ×ˢ t) (x, y) := by
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
    t : Set F
    y : F
    hs : UniqueDiffWithinAt 𝕜 s x
    ht : UniqueDiffWithinAt 𝕜 t y
    ⊢ UniqueDiffWithinAt 𝕜 (SProd.sprod s t) { fst := x, snd := y }
  -/
  rw [uniqueDiffWithinAt_iff] at hs ht ⊢
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
    t : Set F
    y : F
    hs : And (Dense ↑(Submodule.span 𝕜 (tangentConeAt 𝕜 s x))) (Membership.mem (cl …
    ht : And (Dense ↑(Submodule.span 𝕜 (tangentConeAt 𝕜 t y))) (Membership.mem (cl …
    ⊢ And (Dense ↑(Submodule.span 𝕜 (tangentConeAt 𝕜 (SProd.sprod s t) { fst := x, …
  -/
  rw [closure_prod_eq]
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
    t : Set F
    y : F
    hs : And (Dense ↑(Submodule.span 𝕜 (tangentConeAt 𝕜 s x))) (Membership.mem (cl …
    ht : And (Dense ↑(Submodule.span 𝕜 (tangentConeAt 𝕜 t y))) (Membership.mem (cl …
    ⊢ And (Dense ↑(Submodule.span 𝕜 (tangentConeAt 𝕜 (SProd.sprod s t) { fst := x, …
  -/
  refine ⟨?_, hs.2, ht.2⟩
  have : _ ≤ Submodule.span 𝕜 (tangentConeAt 𝕜 (s ×ˢ t) (x, y)) := Submodule.span_mono
    (union_subset (subset_tangentCone_prod_left ht.2) (subset_tangentCone_prod_right hs.2))
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
    t : Set F
    y : F
    hs : And (Dense ↑(Submodule.span 𝕜 (tangentConeAt 𝕜 s x))) (Membership.mem (cl …
    ht : And (Dense ↑(Submodule.span 𝕜 (tangentConeAt 𝕜 t y))) (Membership.mem (cl …
    this : LE.le (Submodule.span 𝕜 (Union.union (Set.image (⇑(LinearMap.inl 𝕜 E F) …
    ⊢ Dense ↑(Submodule.span 𝕜 (tangentConeAt 𝕜 (SProd.sprod s t) { fst := x, snd  …
  -/
  rw [LinearMap.span_inl_union_inr, SetLike.le_def] at this
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
    t : Set F
    y : F
    hs : And (Dense ↑(Submodule.span 𝕜 (tangentConeAt 𝕜 s x))) (Membership.mem (cl …
    ht : And (Dense ↑(Submodule.span 𝕜 (tangentConeAt 𝕜 t y))) (Membership.mem (cl …
    this : ∀ ⦃x_1 : Prod E F⦄, Membership.mem ((Submodule.span 𝕜 (tangentConeAt 𝕜  …
    ⊢ Dense ↑(Submodule.span 𝕜 (tangentConeAt 𝕜 (SProd.sprod s t) { fst := x, snd  …
  -/
  exact (hs.1.prod ht.1).mono this
  /-
    🎉 no goals
  -/


theorem UniqueDiffWithinAt.univ_pi (ι : Type*) [Finite ι] (E : ι → Type*)
    [∀ i, NormedAddCommGroup (E i)] [∀ i, NormedSpace 𝕜 (E i)] (s : ∀ i, Set (E i)) (x : ∀ i, E i)
    (h : ∀ i, UniqueDiffWithinAt 𝕜 (s i) (x i)) : UniqueDiffWithinAt 𝕜 (Set.pi univ s) x := by
  classical
  simp only [uniqueDiffWithinAt_iff, closure_pi_set] at h ⊢
  refine ⟨(dense_pi univ fun i _ => (h i).1).mono ?_, fun i _ => (h i).2⟩
  norm_cast
  simp only [← Submodule.iSup_map_single, iSup_le_iff, LinearMap.map_span, Submodule.span_le,
    ← mapsTo']
  exact fun i => (mapsTo_tangentCone_pi fun j _ => (h j).2).mono Subset.rfl Submodule.subset_span


theorem UniqueDiffWithinAt.pi (ι : Type*) [Finite ι] (E : ι → Type*)
    [∀ i, NormedAddCommGroup (E i)] [∀ i, NormedSpace 𝕜 (E i)] (s : ∀ i, Set (E i)) (x : ∀ i, E i)
    (I : Set ι) (h : ∀ i ∈ I, UniqueDiffWithinAt 𝕜 (s i) (x i)) :
    UniqueDiffWithinAt 𝕜 (Set.pi I s) x := by
  classical
  rw [← Set.univ_pi_piecewise_univ]
  refine UniqueDiffWithinAt.univ_pi ι E _ _ fun i => ?_
  by_cases hi : i ∈ I <;> simp [*, uniqueDiffWithinAt_univ]


/-- The product of two sets of unique differentiability is a set of unique differentiability. -/
theorem UniqueDiffOn.prod {t : Set F} (hs : UniqueDiffOn 𝕜 s) (ht : UniqueDiffOn 𝕜 t) :
    UniqueDiffOn 𝕜 (s ×ˢ t) :=
  fun ⟨x, y⟩ h => UniqueDiffWithinAt.prod (hs x h.1) (ht y h.2)


/-- The finite product of a family of sets of unique differentiability is a set of unique
differentiability. -/
theorem UniqueDiffOn.pi (ι : Type*) [Finite ι] (E : ι → Type*) [∀ i, NormedAddCommGroup (E i)]
    [∀ i, NormedSpace 𝕜 (E i)] (s : ∀ i, Set (E i)) (I : Set ι)
    (h : ∀ i ∈ I, UniqueDiffOn 𝕜 (s i)) : UniqueDiffOn 𝕜 (Set.pi I s) :=
  fun x hx => UniqueDiffWithinAt.pi _ _ _ _ _ fun i hi => h i hi (x i) (hx i hi)


/-- The finite product of a family of sets of unique differentiability is a set of unique
differentiability. -/
theorem UniqueDiffOn.univ_pi (ι : Type*) [Finite ι] (E : ι → Type*)
    [∀ i, NormedAddCommGroup (E i)] [∀ i, NormedSpace 𝕜 (E i)] (s : ∀ i, Set (E i))
    (h : ∀ i, UniqueDiffOn 𝕜 (s i)) : UniqueDiffOn 𝕜 (Set.pi univ s) :=
  UniqueDiffOn.pi _ _ _ _ fun i _ => h i


/-- In a real vector space, a convex set with nonempty interior is a set of unique
differentiability at every point of its closure. -/
theorem uniqueDiffWithinAt_convex {s : Set G} (conv : Convex ℝ s) (hs : (interior s).Nonempty)
    {x : G} (hx : x ∈ closure s) : UniqueDiffWithinAt ℝ s x := by
  /-
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    s : Set G
    conv : Convex Real s
    hs : (interior s).Nonempty
    x : G
    hx : Membership.mem (closure s) x
    ⊢ UniqueDiffWithinAt Real s x
  -/
  rcases hs with ⟨y, hy⟩
  suffices y - x ∈ interior (tangentConeAt ℝ s x) by
    refine ⟨Dense.of_closure ?_, hx⟩
    simp [(Submodule.span ℝ (tangentConeAt ℝ s x)).eq_top_of_nonempty_interior'
        ⟨y - x, interior_mono Submodule.subset_span this⟩]
  /-
    case intro
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    s : Set G
    conv : Convex Real s
    x : G
    hx : Membership.mem (closure s) x
    y : G
    hy : Membership.mem (interior s) y
    ⊢ Membership.mem (interior (tangentConeAt Real s x)) (HSub.hSub y x)
  -/
  rw [mem_interior_iff_mem_nhds]
  /-
    case intro
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    s : Set G
    conv : Convex Real s
    x : G
    hx : Membership.mem (closure s) x
    y : G
    hy : Membership.mem (interior s) y
    ⊢ Membership.mem (nhds (HSub.hSub y x)) (tangentConeAt Real s x)
  -/
  replace hy : interior s ∈ 𝓝 y := IsOpen.mem_nhds isOpen_interior hy
  /-
    case intro
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    s : Set G
    conv : Convex Real s
    x : G
    hx : Membership.mem (closure s) x
    y : G
    hy : Membership.mem (nhds y) (interior s)
    ⊢ Membership.mem (nhds (HSub.hSub y x)) (tangentConeAt Real s x)
  -/
  apply mem_of_superset ((isOpenMap_sub_right x).image_mem_nhds hy)
  /-
    case intro
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    s : Set G
    conv : Convex Real s
    x : G
    hx : Membership.mem (closure s) x
    y : G
    hy : Membership.mem (nhds y) (interior s)
    ⊢ HasSubset.Subset (Set.image (fun x_1 => HSub.hSub x_1 x) (interior s)) (tang …
  -/
  rintro _ ⟨z, zs, rfl⟩
  /-
    case intro.intro.intro
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    s : Set G
    conv : Convex Real s
    x : G
    hx : Membership.mem (closure s) x
    y : G
    hy : Membership.mem (nhds y) (interior s)
    z : G
    zs : Membership.mem (interior s) z
    ⊢ Membership.mem (tangentConeAt Real s x) ((fun x_1 => HSub.hSub x_1 x) z)
  -/
  refine mem_tangentCone_of_openSegment_subset (Subset.trans ?_ interior_subset)
  /-
    case intro.intro.intro
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    s : Set G
    conv : Convex Real s
    x : G
    hx : Membership.mem (closure s) x
    y : G
    hy : Membership.mem (nhds y) (interior s)
    z : G
    zs : Membership.mem (interior s) z
    ⊢ HasSubset.Subset (openSegment Real x z) (interior s)
  -/
  exact conv.openSegment_closure_interior_subset_interior hx zs
  /-
    🎉 no goals
  -/


/-- In a real vector space, a convex set with nonempty interior is a set of unique
differentiability. -/
theorem uniqueDiffOn_convex {s : Set G} (conv : Convex ℝ s) (hs : (interior s).Nonempty) :
    UniqueDiffOn ℝ s :=
  fun _ xs => uniqueDiffWithinAt_convex conv hs (subset_closure xs)


theorem uniqueDiffOn_Ici (a : ℝ) : UniqueDiffOn ℝ (Ici a) :=
                                           /-
                                             a : Real
                                             ⊢ (interior (Set.Ici a)).Nonempty
                                           -/
  uniqueDiffOn_convex (convex_Ici a) <| by simp only [interior_Ici, nonempty_Ioi]
                                           /-
                                             🎉 no goals
                                           -/


theorem uniqueDiffOn_Iic (a : ℝ) : UniqueDiffOn ℝ (Iic a) :=
                                           /-
                                             a : Real
                                             ⊢ (interior (Set.Iic a)).Nonempty
                                           -/
  uniqueDiffOn_convex (convex_Iic a) <| by simp only [interior_Iic, nonempty_Iio]
                                           /-
                                             🎉 no goals
                                           -/


theorem uniqueDiffOn_Ioi (a : ℝ) : UniqueDiffOn ℝ (Ioi a) :=
  isOpen_Ioi.uniqueDiffOn


theorem uniqueDiffOn_Iio (a : ℝ) : UniqueDiffOn ℝ (Iio a) :=
  isOpen_Iio.uniqueDiffOn


theorem uniqueDiffOn_Icc {a b : ℝ} (hab : a < b) : UniqueDiffOn ℝ (Icc a b) :=
                                             /-
                                               a b : Real
                                               hab : LT.lt a b
                                               ⊢ (interior (Set.Icc a b)).Nonempty
                                             -/
  uniqueDiffOn_convex (convex_Icc a b) <| by simp only [interior_Icc, nonempty_Ioo, hab]
                                             /-
                                               🎉 no goals
                                             -/


theorem uniqueDiffOn_Ico (a b : ℝ) : UniqueDiffOn ℝ (Ico a b) :=
  if hab : a < b then
                                               /-
                                                 a b : Real
                                                 hab : LT.lt a b
                                                 ⊢ (interior (Set.Ico a b)).Nonempty
                                               -/
    uniqueDiffOn_convex (convex_Ico a b) <| by simp only [interior_Ico, nonempty_Ioo, hab]
                                               /-
                                                 🎉 no goals
                                               -/
          /-
            a b : Real
            hab : Not (LT.lt a b)
            ⊢ UniqueDiffOn Real (Set.Ico a b)
          -/
  else by simp only [Ico_eq_empty hab, uniqueDiffOn_empty]
          /-
            🎉 no goals
          -/


theorem uniqueDiffOn_Ioc (a b : ℝ) : UniqueDiffOn ℝ (Ioc a b) :=
  if hab : a < b then
                                               /-
                                                 a b : Real
                                                 hab : LT.lt a b
                                                 ⊢ (interior (Set.Ioc a b)).Nonempty
                                               -/
    uniqueDiffOn_convex (convex_Ioc a b) <| by simp only [interior_Ioc, nonempty_Ioo, hab]
                                               /-
                                                 🎉 no goals
                                               -/
          /-
            a b : Real
            hab : Not (LT.lt a b)
            ⊢ UniqueDiffOn Real (Set.Ioc a b)
          -/
  else by simp only [Ioc_eq_empty hab, uniqueDiffOn_empty]
          /-
            🎉 no goals
          -/


theorem uniqueDiffOn_Ioo (a b : ℝ) : UniqueDiffOn ℝ (Ioo a b) :=
  isOpen_Ioo.uniqueDiffOn


/-- The real interval `[0, 1]` is a set of unique differentiability. -/
theorem uniqueDiffOn_Icc_zero_one : UniqueDiffOn ℝ (Icc (0 : ℝ) 1) :=
  uniqueDiffOn_Icc zero_lt_one


theorem uniqueDiffWithinAt_Ioo {a b t : ℝ} (ht : t ∈ Set.Ioo a b) :
    UniqueDiffWithinAt ℝ (Set.Ioo a b) t :=
  IsOpen.uniqueDiffWithinAt isOpen_Ioo ht


theorem uniqueDiffWithinAt_Ioi (a : ℝ) : UniqueDiffWithinAt ℝ (Ioi a) a :=
                                               /-
                                                 a : Real
                                                 ⊢ (interior (Set.Ioi a)).Nonempty
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
  uniqueDiffWithinAt_convex (convex_Ioi a) (by simp) (by simp)
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem uniqueDiffWithinAt_Iio (a : ℝ) : UniqueDiffWithinAt ℝ (Iio a) a :=
                                               /-
                                                 a : Real
                                                 ⊢ (interior (Set.Iio a)).Nonempty
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
  uniqueDiffWithinAt_convex (convex_Iio a) (by simp) (by simp)
                                                         /-
                                                           🎉 no goals
                                                         -/


