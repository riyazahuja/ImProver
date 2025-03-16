/-- We say that a measure `μ` is *inner regular* with respect to predicates `p q : Set α → Prop`,
if for every `U` such that `q U` and `r < μ U`, there exists a subset `K ⊆ U` satisfying `p K`
of measure greater than `r`.

This definition is used to prove some facts about regular and weakly regular measures without
repeating the proofs. -/
def InnerRegularWRT {α} {_ : MeasurableSpace α} (μ : Measure α) (p q : Set α → Prop) :=
  ∀ ⦃U⦄, q U → ∀ r < μ U, ∃ K, K ⊆ U ∧ p K ∧ r < μ K


theorem measure_eq_iSup (H : InnerRegularWRT μ p q) (hU : q U) :
    μ U = ⨆ (K) (_ : K ⊆ U) (_ : p K), μ K := by
  refine
    le_antisymm (le_of_forall_lt fun r hr => ?_) (iSup₂_le fun K hK => iSup_le fun _ => μ.mono hK)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Set α → Prop
    U : Set α
    H : μ.InnerRegularWRT p q
    hU : q U
    r : ENNReal
    hr : LT.lt r (μ U)
    ⊢ LT.lt r (iSup fun K => iSup fun x => iSup fun x => μ K)
  -/
  simpa only [lt_iSup_iff, exists_prop] using H hU r hr
  /-
    🎉 no goals
  -/


theorem exists_subset_lt_add (H : InnerRegularWRT μ p q) (h0 : p ∅) (hU : q U) (hμU : μ U ≠ ∞)
    (hε : ε ≠ 0) : ∃ K, K ⊆ U ∧ p K ∧ μ U < μ K + ε := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Set α → Prop
    U : Set α
    ε : ENNReal
    H : μ.InnerRegularWRT p q
    h0 : p EmptyCollection.emptyCollection
    hU : q U
    hμU : Ne (μ U) Top.top
    hε : Ne ε 0
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (p K) (LT.lt (μ U) (HAdd.hAd …
  -/
  rcases eq_or_ne (μ U) 0 with h₀ | h₀
    /-
      case inl
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p q : Set α → Prop
      U : Set α
      ε : ENNReal
      H : μ.InnerRegularWRT p q
      h0 : p EmptyCollection.emptyCollection
      hU : q U
      hμU : Ne (μ U) Top.top
      hε : Ne ε 0
      h₀ : Eq (μ U) 0
      ⊢ Exists fun K => And (HasSubset.Subset K U) (And (p K) (LT.lt (μ U) (HAdd.hAd …
    -/
  · refine ⟨∅, empty_subset _, h0, ?_⟩
    /-
      case inl
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p q : Set α → Prop
      U : Set α
      ε : ENNReal
      H : μ.InnerRegularWRT p q
      h0 : p EmptyCollection.emptyCollection
      hU : q U
      hμU : Ne (μ U) Top.top
      hε : Ne ε 0
      h₀ : Eq (μ U) 0
      ⊢ LT.lt (μ U) (HAdd.hAdd (μ EmptyCollection.emptyCollection) ε)
    -/
    rwa [measure_empty, h₀, zero_add, pos_iff_ne_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p q : Set α → Prop
      U : Set α
      ε : ENNReal
      H : μ.InnerRegularWRT p q
      h0 : p EmptyCollection.emptyCollection
      hU : q U
      hμU : Ne (μ U) Top.top
      hε : Ne ε 0
      h₀ : Ne (μ U) 0
      ⊢ Exists fun K => And (HasSubset.Subset K U) (And (p K) (LT.lt (μ U) (HAdd.hAd …
    -/
  · rcases H hU _ (ENNReal.sub_lt_self hμU h₀ hε) with ⟨K, hKU, hKc, hrK⟩
    /-
      case inr.intro.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p q : Set α → Prop
      U : Set α
      ε : ENNReal
      H : μ.InnerRegularWRT p q
      h0 : p EmptyCollection.emptyCollection
      hU : q U
      hμU : Ne (μ U) Top.top
      hε : Ne ε 0
      h₀ : Ne (μ U) 0
      K : Set α
      hKU : HasSubset.Subset K U
      hKc : p K
      hrK : LT.lt (HSub.hSub (μ U) ε) (μ K)
      ⊢ Exists fun K => And (HasSubset.Subset K U) (And (p K) (LT.lt (μ U) (HAdd.hAd …
    -/
    exact ⟨K, hKU, hKc, ENNReal.lt_add_of_sub_lt_right (Or.inl hμU) hrK⟩
    /-
      🎉 no goals
    -/


protected theorem map {α β} [MeasurableSpace α] [MeasurableSpace β]
    {μ : Measure α} {pa qa : Set α → Prop}
    (H : InnerRegularWRT μ pa qa) {f : α → β} (hf : AEMeasurable f μ) {pb qb : Set β → Prop}
    (hAB : ∀ U, qb U → qa (f ⁻¹' U)) (hAB' : ∀ K, pa K → pb (f '' K))
    (hB₂ : ∀ U, qb U → MeasurableSet U) :
    InnerRegularWRT (map f μ) pb qb := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    pa qa : Set α → Prop
    H : μ.InnerRegularWRT pa qa
    f : α → β
    hf : AEMeasurable f μ
    pb qb : Set β → Prop
    hAB : ∀ (U : Set β), qb U → qa (Set.preimage f U)
    hAB' : ∀ (K : Set α), pa K → pb (Set.image f K)
    hB₂ : ∀ (U : Set β), qb U → MeasurableSet U
    ⊢ (MeasureTheory.Measure.map f μ).InnerRegularWRT pb qb
  -/
  intro U hU r hr
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    pa qa : Set α → Prop
    H : μ.InnerRegularWRT pa qa
    f : α → β
    hf : AEMeasurable f μ
    pb qb : Set β → Prop
    hAB : ∀ (U : Set β), qb U → qa (Set.preimage f U)
    hAB' : ∀ (K : Set α), pa K → pb (Set.image f K)
    hB₂ : ∀ (U : Set β), qb U → MeasurableSet U
    U : Set β
    hU : qb U
    r : ENNReal
    hr : LT.lt r ((MeasureTheory.Measure.map f μ) U)
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (pb K) (LT.lt r ((MeasureThe …
  -/
  rw [map_apply_of_aemeasurable hf (hB₂ _ hU)] at hr
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    pa qa : Set α → Prop
    H : μ.InnerRegularWRT pa qa
    f : α → β
    hf : AEMeasurable f μ
    pb qb : Set β → Prop
    hAB : ∀ (U : Set β), qb U → qa (Set.preimage f U)
    hAB' : ∀ (K : Set α), pa K → pb (Set.image f K)
    hB₂ : ∀ (U : Set β), qb U → MeasurableSet U
    U : Set β
    hU : qb U
    r : ENNReal
    hr : LT.lt r (μ (Set.preimage f U))
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (pb K) (LT.lt r ((MeasureThe …
  -/
  rcases H (hAB U hU) r hr with ⟨K, hKU, hKc, hK⟩
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    pa qa : Set α → Prop
    H : μ.InnerRegularWRT pa qa
    f : α → β
    hf : AEMeasurable f μ
    pb qb : Set β → Prop
    hAB : ∀ (U : Set β), qb U → qa (Set.preimage f U)
    hAB' : ∀ (K : Set α), pa K → pb (Set.image f K)
    hB₂ : ∀ (U : Set β), qb U → MeasurableSet U
    U : Set β
    hU : qb U
    r : ENNReal
    hr : LT.lt r (μ (Set.preimage f U))
    K : Set α
    hKU : HasSubset.Subset K (Set.preimage f U)
    hKc : pa K
    hK : LT.lt r (μ K)
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (pb K) (LT.lt r ((MeasureThe …
  -/
  refine ⟨f '' K, image_subset_iff.2 hKU, hAB' _ hKc, ?_⟩
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    pa qa : Set α → Prop
    H : μ.InnerRegularWRT pa qa
    f : α → β
    hf : AEMeasurable f μ
    pb qb : Set β → Prop
    hAB : ∀ (U : Set β), qb U → qa (Set.preimage f U)
    hAB' : ∀ (K : Set α), pa K → pb (Set.image f K)
    hB₂ : ∀ (U : Set β), qb U → MeasurableSet U
    U : Set β
    hU : qb U
    r : ENNReal
    hr : LT.lt r (μ (Set.preimage f U))
    K : Set α
    hKU : HasSubset.Subset K (Set.preimage f U)
    hKc : pa K
    hK : LT.lt r (μ K)
    ⊢ LT.lt r ((MeasureTheory.Measure.map f μ) (Set.image f K))
  -/
  exact hK.trans_le (le_map_apply_image hf _)
  /-
    🎉 no goals
  -/


theorem map' {α β} [MeasurableSpace α] [MeasurableSpace β] {μ : Measure α} {pa qa : Set α → Prop}
    (H : InnerRegularWRT μ pa qa) (f : α ≃ᵐ β) {pb qb : Set β → Prop}
    (hAB : ∀ U, qb U → qa (f ⁻¹' U)) (hAB' : ∀ K, pa K → pb (f '' K)) :
    InnerRegularWRT (map f μ) pb qb := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    pa qa : Set α → Prop
    H : μ.InnerRegularWRT pa qa
    f : MeasurableEquiv α β
    pb qb : Set β → Prop
    hAB : ∀ (U : Set β), qb U → qa (Set.preimage (⇑f) U)
    hAB' : ∀ (K : Set α), pa K → pb (Set.image (⇑f) K)
    ⊢ (MeasureTheory.Measure.map (⇑f) μ).InnerRegularWRT pb qb
  -/
  intro U hU r hr
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    pa qa : Set α → Prop
    H : μ.InnerRegularWRT pa qa
    f : MeasurableEquiv α β
    pb qb : Set β → Prop
    hAB : ∀ (U : Set β), qb U → qa (Set.preimage (⇑f) U)
    hAB' : ∀ (K : Set α), pa K → pb (Set.image (⇑f) K)
    U : Set β
    hU : qb U
    r : ENNReal
    hr : LT.lt r ((MeasureTheory.Measure.map (⇑f) μ) U)
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (pb K) (LT.lt r ((MeasureThe …
  -/
  rw [f.map_apply U] at hr
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    pa qa : Set α → Prop
    H : μ.InnerRegularWRT pa qa
    f : MeasurableEquiv α β
    pb qb : Set β → Prop
    hAB : ∀ (U : Set β), qb U → qa (Set.preimage (⇑f) U)
    hAB' : ∀ (K : Set α), pa K → pb (Set.image (⇑f) K)
    U : Set β
    hU : qb U
    r : ENNReal
    hr : LT.lt r (μ (Set.preimage (⇑f) U))
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (pb K) (LT.lt r ((MeasureThe …
  -/
  rcases H (hAB U hU) r hr with ⟨K, hKU, hKc, hK⟩
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    pa qa : Set α → Prop
    H : μ.InnerRegularWRT pa qa
    f : MeasurableEquiv α β
    pb qb : Set β → Prop
    hAB : ∀ (U : Set β), qb U → qa (Set.preimage (⇑f) U)
    hAB' : ∀ (K : Set α), pa K → pb (Set.image (⇑f) K)
    U : Set β
    hU : qb U
    r : ENNReal
    hr : LT.lt r (μ (Set.preimage (⇑f) U))
    K : Set α
    hKU : HasSubset.Subset K (Set.preimage (⇑f) U)
    hKc : pa K
    hK : LT.lt r (μ K)
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (pb K) (LT.lt r ((MeasureThe …
  -/
  refine ⟨f '' K, image_subset_iff.2 hKU, hAB' _ hKc, ?_⟩
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    pa qa : Set α → Prop
    H : μ.InnerRegularWRT pa qa
    f : MeasurableEquiv α β
    pb qb : Set β → Prop
    hAB : ∀ (U : Set β), qb U → qa (Set.preimage (⇑f) U)
    hAB' : ∀ (K : Set α), pa K → pb (Set.image (⇑f) K)
    U : Set β
    hU : qb U
    r : ENNReal
    hr : LT.lt r (μ (Set.preimage (⇑f) U))
    K : Set α
    hKU : HasSubset.Subset K (Set.preimage (⇑f) U)
    hKc : pa K
    hK : LT.lt r (μ K)
    ⊢ LT.lt r ((MeasureTheory.Measure.map (⇑f) μ) (Set.image (⇑f) K))
  -/
  rwa [f.map_apply, f.preimage_image]
  /-
    🎉 no goals
  -/


theorem smul (H : InnerRegularWRT μ p q) (c : ℝ≥0∞) : InnerRegularWRT (c • μ) p q := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Set α → Prop
    H : μ.InnerRegularWRT p q
    c : ENNReal
    ⊢ (HSMul.hSMul c μ).InnerRegularWRT p q
  -/
  intro U hU r hr
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Set α → Prop
    H : μ.InnerRegularWRT p q
    c : ENNReal
    U : Set α
    hU : q U
    r : ENNReal
    hr : LT.lt r ((HSMul.hSMul c μ) U)
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (p K) (LT.lt r ((HSMul.hSMul …
  -/
  rw [smul_apply, H.measure_eq_iSup hU, smul_eq_mul] at hr
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Set α → Prop
    H : μ.InnerRegularWRT p q
    c : ENNReal
    U : Set α
    hU : q U
    r : ENNReal
    hr : LT.lt r (HMul.hMul c (iSup fun K => iSup fun x => iSup fun x => μ K))
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (p K) (LT.lt r ((HSMul.hSMul …
  -/
  simpa only [ENNReal.mul_iSup, lt_iSup_iff, exists_prop] using hr
  /-
    🎉 no goals
  -/


theorem trans {q' : Set α → Prop} (H : InnerRegularWRT μ p q) (H' : InnerRegularWRT μ q q') :
    InnerRegularWRT μ p q' := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q q' : Set α → Prop
    H : μ.InnerRegularWRT p q
    H' : μ.InnerRegularWRT q q'
    ⊢ μ.InnerRegularWRT p q'
  -/
  intro U hU r hr
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q q' : Set α → Prop
    H : μ.InnerRegularWRT p q
    H' : μ.InnerRegularWRT q q'
    U : Set α
    hU : q' U
    r : ENNReal
    hr : LT.lt r (μ U)
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (p K) (LT.lt r (μ K)))
  -/
  rcases H' hU r hr with ⟨F, hFU, hqF, hF⟩; rcases H hqF _ hF with ⟨K, hKF, hpK, hrK⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q q' : Set α → Prop
    H : μ.InnerRegularWRT p q
    H' : μ.InnerRegularWRT q q'
    U : Set α
    hU : q' U
    r : ENNReal
    hr : LT.lt r (μ U)
    F : Set α
    hFU : HasSubset.Subset F U
    hqF : q F
    hF : LT.lt r (μ F)
    K : Set α
    hKF : HasSubset.Subset K F
    hpK : p K
    hrK : LT.lt r (μ K)
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (p K) (LT.lt r (μ K)))
  -/
  exact ⟨K, hKF.trans hFU, hpK, hrK⟩
  /-
    🎉 no goals
  -/


theorem rfl {p : Set α → Prop} : InnerRegularWRT μ p p :=
  fun U hU _r hr ↦ ⟨U, Subset.rfl, hU, hr⟩


theorem of_imp (h : ∀ s, q s → p s) : InnerRegularWRT μ p q :=
  fun U hU _ hr ↦ ⟨U, Subset.rfl, h U hU, hr⟩


theorem mono {p' q' : Set α → Prop} (H : InnerRegularWRT μ p q)
    (h : ∀ s, q' s → q s) (h' : ∀ s, p s → p' s) : InnerRegularWRT μ p' q' :=
  of_imp h' |>.trans H |>.trans (of_imp h)


/-- A measure `μ` is outer regular if `μ(A) = inf {μ(U) | A ⊆ U open}` for a measurable set `A`.

This definition implies the same equality for any (not necessarily measurable) set, see
`Set.measure_eq_iInf_isOpen`. -/
class OuterRegular (μ : Measure α) : Prop where
  protected outerRegular :
    ∀ ⦃A : Set α⦄, MeasurableSet A → ∀ r > μ A, ∃ U, U ⊇ A ∧ IsOpen U ∧ μ U < r


/-- A measure `μ` is regular if
  - it is finite on all compact sets;
  - it is outer regular: `μ(A) = inf {μ(U) | A ⊆ U open}` for `A` measurable;
  - it is inner regular for open sets, using compact sets:
    `μ(U) = sup {μ(K) | K ⊆ U compact}` for `U` open. -/
class Regular (μ : Measure α) extends IsFiniteMeasureOnCompacts μ, OuterRegular μ : Prop where
  innerRegular : InnerRegularWRT μ IsCompact IsOpen


/-- A measure `μ` is weakly regular if
  - it is outer regular: `μ(A) = inf {μ(U) | A ⊆ U open}` for `A` measurable;
  - it is inner regular for open sets, using closed sets:
    `μ(U) = sup {μ(F) | F ⊆ U closed}` for `U` open. -/
class WeaklyRegular (μ : Measure α) extends OuterRegular μ : Prop where
  protected innerRegular : InnerRegularWRT μ IsClosed IsOpen


/-- A measure `μ` is inner regular if, for any measurable set `s`, then
`μ(s) = sup {μ(K) | K ⊆ s compact}`. -/
class InnerRegular (μ : Measure α) : Prop where
  protected innerRegular : InnerRegularWRT μ IsCompact MeasurableSet


/-- A measure `μ` is inner regular for finite measure sets with respect to compact sets:
for any measurable set `s` with finite measure, then `μ(s) = sup {μ(K) | K ⊆ s compact}`.
The main interest of this class is that it is satisfied for both natural Haar measures (the
regular one and the inner regular one). -/
class InnerRegularCompactLTTop (μ : Measure α) : Prop where
  protected innerRegular : InnerRegularWRT μ IsCompact (fun s ↦ MeasurableSet s ∧ μ s ≠ ∞)

-- see Note [lower instance priority]

/-- A regular measure is weakly regular in an R₁ space. -/
instance (priority := 100) Regular.weaklyRegular [R1Space α] [Regular μ] :
    WeaklyRegular μ where
  innerRegular := fun _U hU r hr ↦
    let ⟨K, KU, K_comp, hK⟩ := Regular.innerRegular hU r hr
    ⟨closure K, K_comp.closure_subset_of_isOpen hU KU, isClosed_closure,
      hK.trans_le (measure_mono subset_closure)⟩


instance zero : OuterRegular (0 : Measure α) :=
  ⟨fun A _ _r hr => ⟨univ, subset_univ A, isOpen_univ, hr⟩⟩


/-- Given `r` larger than the measure of a set `A`, there exists an open superset of `A` with
measure less than `r`. -/
theorem _root_.Set.exists_isOpen_lt_of_lt [OuterRegular μ] (A : Set α) (r : ℝ≥0∞) (hr : μ A < r) :
    ∃ U, U ⊇ A ∧ IsOpen U ∧ μ U < r := by
  rcases OuterRegular.outerRegular (measurableSet_toMeasurable μ A) r
      (by rwa [measure_toMeasurable]) with
    ⟨U, hAU, hUo, hU⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace α
    inst✝ : μ.OuterRegular
    A : Set α
    r : ENNReal
    hr : LT.lt (μ A) r
    U : Set α
    hAU : Superset U (MeasureTheory.toMeasurable μ A)
    hUo : IsOpen U
    hU : LT.lt (μ U) r
    ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LT.lt (μ U) r))
  -/
  exact ⟨U, (subset_toMeasurable _ _).trans hAU, hUo, hU⟩
  /-
    🎉 no goals
  -/


/-- For an outer regular measure, the measure of a set is the infimum of the measures of open sets
containing it. -/
theorem _root_.Set.measure_eq_iInf_isOpen (A : Set α) (μ : Measure α) [OuterRegular μ] :
    μ A = ⨅ (U : Set α) (_ : A ⊆ U) (_ : IsOpen U), μ U := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    A : Set α
    μ : MeasureTheory.Measure α
    inst✝ : μ.OuterRegular
    ⊢ Eq (μ A) (iInf fun U => iInf fun x => iInf fun x => μ U)
  -/
  refine le_antisymm (le_iInf₂ fun s hs => le_iInf fun _ => μ.mono hs) ?_
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    A : Set α
    μ : MeasureTheory.Measure α
    inst✝ : μ.OuterRegular
    ⊢ LE.le (iInf fun U => iInf fun x => iInf fun x => μ U) (μ A)
  -/
  refine le_of_forall_lt' fun r hr => ?_
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    A : Set α
    μ : MeasureTheory.Measure α
    inst✝ : μ.OuterRegular
    r : ENNReal
    hr : LT.lt (μ A) r
    ⊢ LT.lt (iInf fun U => iInf fun x => iInf fun x => μ U) r
  -/
  simpa only [iInf_lt_iff, exists_prop] using A.exists_isOpen_lt_of_lt r hr
  /-
    🎉 no goals
  -/


theorem _root_.Set.exists_isOpen_lt_add [OuterRegular μ] (A : Set α) (hA : μ A ≠ ∞) {ε : ℝ≥0∞}
    (hε : ε ≠ 0) : ∃ U, U ⊇ A ∧ IsOpen U ∧ μ U < μ A + ε :=
  A.exists_isOpen_lt_of_lt _ (ENNReal.lt_add_right hA hε)


theorem _root_.Set.exists_isOpen_le_add (A : Set α) (μ : Measure α) [OuterRegular μ] {ε : ℝ≥0∞}
    (hε : ε ≠ 0) : ∃ U, U ⊇ A ∧ IsOpen U ∧ μ U ≤ μ A + ε := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    A : Set α
    μ : MeasureTheory.Measure α
    inst✝ : μ.OuterRegular
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LE.le (μ U) (HAdd.hAdd ( …
  -/
  rcases eq_or_ne (μ A) ∞ with (H | H)
    /-
      case inl
      α : Type u_1
      inst✝² : MeasurableSpace α
      inst✝¹ : TopologicalSpace α
      A : Set α
      μ : MeasureTheory.Measure α
      inst✝ : μ.OuterRegular
      ε : ENNReal
      hε : Ne ε 0
      H : Eq (μ A) Top.top
      ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LE.le (μ U) (HAdd.hAdd ( …
    -/
  · exact ⟨univ, subset_univ _, isOpen_univ, by simp only [H, _root_.top_add, le_top]⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝² : MeasurableSpace α
      inst✝¹ : TopologicalSpace α
      A : Set α
      μ : MeasureTheory.Measure α
      inst✝ : μ.OuterRegular
      ε : ENNReal
      hε : Ne ε 0
      H : Ne (μ A) Top.top
      ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LE.le (μ U) (HAdd.hAdd ( …
    -/
  · rcases A.exists_isOpen_lt_add H hε with ⟨U, AU, U_open, hU⟩
    /-
      case inr.intro.intro.intro
      α : Type u_1
      inst✝² : MeasurableSpace α
      inst✝¹ : TopologicalSpace α
      A : Set α
      μ : MeasureTheory.Measure α
      inst✝ : μ.OuterRegular
      ε : ENNReal
      hε : Ne ε 0
      H : Ne (μ A) Top.top
      U : Set α
      AU : Superset U A
      U_open : IsOpen U
      hU : LT.lt (μ U) (HAdd.hAdd (μ A) ε)
      ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LE.le (μ U) (HAdd.hAdd ( …
    -/
    exact ⟨U, AU, U_open, hU.le⟩
    /-
      🎉 no goals
    -/


theorem _root_.MeasurableSet.exists_isOpen_diff_lt [OuterRegular μ] {A : Set α}
    (hA : MeasurableSet A) (hA' : μ A ≠ ∞) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ U, U ⊇ A ∧ IsOpen U ∧ μ U < ∞ ∧ μ (U \ A) < ε := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace α
    inst✝ : μ.OuterRegular
    A : Set α
    hA : MeasurableSet A
    hA' : Ne (μ A) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (And (LT.lt (μ U) Top.top …
  -/
  rcases A.exists_isOpen_lt_add hA' hε with ⟨U, hAU, hUo, hU⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace α
    inst✝ : μ.OuterRegular
    A : Set α
    hA : MeasurableSet A
    hA' : Ne (μ A) Top.top
    ε : ENNReal
    hε : Ne ε 0
    U : Set α
    hAU : Superset U A
    hUo : IsOpen U
    hU : LT.lt (μ U) (HAdd.hAdd (μ A) ε)
    ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (And (LT.lt (μ U) Top.top …
  -/
  use U, hAU, hUo, hU.trans_le le_top
  /-
    case right
    α : Type u_1
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace α
    inst✝ : μ.OuterRegular
    A : Set α
    hA : MeasurableSet A
    hA' : Ne (μ A) Top.top
    ε : ENNReal
    hε : Ne ε 0
    U : Set α
    hAU : Superset U A
    hUo : IsOpen U
    hU : LT.lt (μ U) (HAdd.hAdd (μ A) ε)
    ⊢ LT.lt (μ (SDiff.sdiff U A)) ε
  -/
  exact measure_diff_lt_of_lt_add hA.nullMeasurableSet hAU hA' hU
  /-
    🎉 no goals
  -/


protected theorem map [OpensMeasurableSpace α] [MeasurableSpace β] [TopologicalSpace β]
    [BorelSpace β] (f : α ≃ₜ β) (μ : Measure α) [OuterRegular μ] :
    (Measure.map f μ).OuterRegular := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OpensMeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : TopologicalSpace β
    inst✝¹ : BorelSpace β
    f : Homeomorph α β
    μ : MeasureTheory.Measure α
    inst✝ : μ.OuterRegular
    ⊢ (MeasureTheory.Measure.map (⇑f) μ).OuterRegular
  -/
  refine ⟨fun A hA r hr => ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OpensMeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : TopologicalSpace β
    inst✝¹ : BorelSpace β
    f : Homeomorph α β
    μ : MeasureTheory.Measure α
    inst✝ : μ.OuterRegular
    A : Set β
    hA : MeasurableSet A
    r : ENNReal
    hr : GT.gt r ((MeasureTheory.Measure.map (⇑f) μ) A)
    ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LT.lt ((MeasureTheory.Me …
  -/
  rw [map_apply f.measurable hA, ← f.image_symm] at hr
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OpensMeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : TopologicalSpace β
    inst✝¹ : BorelSpace β
    f : Homeomorph α β
    μ : MeasureTheory.Measure α
    inst✝ : μ.OuterRegular
    A : Set β
    hA : MeasurableSet A
    r : ENNReal
    hr : GT.gt r (μ (Set.image (⇑f.symm) A))
    ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LT.lt ((MeasureTheory.Me …
  -/
  rcases Set.exists_isOpen_lt_of_lt _ r hr with ⟨U, hAU, hUo, hU⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OpensMeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : TopologicalSpace β
    inst✝¹ : BorelSpace β
    f : Homeomorph α β
    μ : MeasureTheory.Measure α
    inst✝ : μ.OuterRegular
    A : Set β
    hA : MeasurableSet A
    r : ENNReal
    hr : GT.gt r (μ (Set.image (⇑f.symm) A))
    U : Set α
    hAU : Superset U (Set.image (⇑f.symm) A)
    hUo : IsOpen U
    hU : LT.lt (μ U) r
    ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LT.lt ((MeasureTheory.Me …
  -/
  have : IsOpen (f.symm ⁻¹' U) := hUo.preimage f.symm.continuous
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OpensMeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : TopologicalSpace β
    inst✝¹ : BorelSpace β
    f : Homeomorph α β
    μ : MeasureTheory.Measure α
    inst✝ : μ.OuterRegular
    A : Set β
    hA : MeasurableSet A
    r : ENNReal
    hr : GT.gt r (μ (Set.image (⇑f.symm) A))
    U : Set α
    hAU : Superset U (Set.image (⇑f.symm) A)
    hUo : IsOpen U
    hU : LT.lt (μ U) r
    this : IsOpen (Set.preimage (⇑f.symm) U)
    ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LT.lt ((MeasureTheory.Me …
  -/
  refine ⟨f.symm ⁻¹' U, image_subset_iff.1 hAU, this, ?_⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OpensMeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : TopologicalSpace β
    inst✝¹ : BorelSpace β
    f : Homeomorph α β
    μ : MeasureTheory.Measure α
    inst✝ : μ.OuterRegular
    A : Set β
    hA : MeasurableSet A
    r : ENNReal
    hr : GT.gt r (μ (Set.image (⇑f.symm) A))
    U : Set α
    hAU : Superset U (Set.image (⇑f.symm) A)
    hUo : IsOpen U
    hU : LT.lt (μ U) r
    this : IsOpen (Set.preimage (⇑f.symm) U)
    ⊢ LT.lt ((MeasureTheory.Measure.map (⇑f) μ) (Set.preimage (⇑f.symm) U)) r
  -/
  rwa [map_apply f.measurable this.measurableSet, f.preimage_symm, f.preimage_image]
  /-
    🎉 no goals
  -/


protected theorem smul (μ : Measure α) [OuterRegular μ] {x : ℝ≥0∞} (hx : x ≠ ∞) :
    (x • μ).OuterRegular := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    μ : MeasureTheory.Measure α
    inst✝ : μ.OuterRegular
    x : ENNReal
    hx : Ne x Top.top
    ⊢ (HSMul.hSMul x μ).OuterRegular
  -/
  rcases eq_or_ne x 0 with (rfl | h0)
    /-
      case inl
      α : Type u_1
      inst✝² : MeasurableSpace α
      inst✝¹ : TopologicalSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.OuterRegular
      hx : Ne 0 Top.top
      ⊢ (HSMul.hSMul 0 μ).OuterRegular
    -/
  · rw [zero_smul]
    /-
      case inl
      α : Type u_1
      inst✝² : MeasurableSpace α
      inst✝¹ : TopologicalSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.OuterRegular
      hx : Ne 0 Top.top
      ⊢ MeasureTheory.Measure.OuterRegular 0
    -/
    exact OuterRegular.zero
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝² : MeasurableSpace α
      inst✝¹ : TopologicalSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.OuterRegular
      x : ENNReal
      hx : Ne x Top.top
      h0 : Ne x 0
      ⊢ (HSMul.hSMul x μ).OuterRegular
    -/
  · refine ⟨fun A _ r hr => ?_⟩
    /-
      case inr
      α : Type u_1
      inst✝² : MeasurableSpace α
      inst✝¹ : TopologicalSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.OuterRegular
      x : ENNReal
      hx : Ne x Top.top
      h0 : Ne x 0
      A : Set α
      x✝ : MeasurableSet A
      r : ENNReal
      hr : GT.gt r ((HSMul.hSMul x μ) A)
      ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LT.lt ((HSMul.hSMul x μ) …
    -/
    rw [smul_apply, A.measure_eq_iInf_isOpen, smul_eq_mul] at hr
    /-
      case inr
      α : Type u_1
      inst✝² : MeasurableSpace α
      inst✝¹ : TopologicalSpace α
      μ : MeasureTheory.Measure α
      inst✝ : μ.OuterRegular
      x : ENNReal
      hx : Ne x Top.top
      h0 : Ne x 0
      A : Set α
      x✝ : MeasurableSet A
      r : ENNReal
      hr : GT.gt r (HMul.hMul x (iInf fun U => iInf fun x => iInf fun x => μ U))
      ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LT.lt ((HSMul.hSMul x μ) …
    -/
    simpa only [ENNReal.mul_iInf_of_ne h0 hx, gt_iff_lt, iInf_lt_iff, exists_prop] using hr
    /-
      🎉 no goals
    -/


instance smul_nnreal (μ : Measure α) [OuterRegular μ] (c : ℝ≥0) :
    OuterRegular (c • μ) :=
  OuterRegular.smul μ coe_ne_top


/-- If the restrictions of a measure to countably many open sets covering the space are
outer regular, then the measure itself is outer regular. -/
lemma of_restrict [OpensMeasurableSpace α] {μ : Measure α} {s : ℕ → Set α}
    (h : ∀ n, OuterRegular (μ.restrict (s n))) (h' : ∀ n, IsOpen (s n)) (h'' : univ ⊆ ⋃ n, s n) :
    OuterRegular μ := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    inst✝ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Nat → Set α
    h : ∀ (n : Nat), (μ.restrict (s n)).OuterRegular
    h' : ∀ (n : Nat), IsOpen (s n)
    h'' : HasSubset.Subset Set.univ (Set.iUnion fun n => s n)
    ⊢ μ.OuterRegular
  -/
  refine ⟨fun A hA r hr => ?_⟩
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    inst✝ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Nat → Set α
    h : ∀ (n : Nat), (μ.restrict (s n)).OuterRegular
    h' : ∀ (n : Nat), IsOpen (s n)
    h'' : HasSubset.Subset Set.univ (Set.iUnion fun n => s n)
    A : Set α
    hA : MeasurableSet A
    r : ENNReal
    hr : GT.gt r (μ A)
    ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LT.lt (μ U) r))
  -/
  have HA : μ A < ∞ := lt_of_lt_of_le hr le_top
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    inst✝ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Nat → Set α
    h : ∀ (n : Nat), (μ.restrict (s n)).OuterRegular
    h' : ∀ (n : Nat), IsOpen (s n)
    h'' : HasSubset.Subset Set.univ (Set.iUnion fun n => s n)
    A : Set α
    hA : MeasurableSet A
    r : ENNReal
    hr : GT.gt r (μ A)
    HA : LT.lt (μ A) Top.top
    ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LT.lt (μ U) r))
  -/
  have hm : ∀ n, MeasurableSet (s n) := fun n => (h' n).measurableSet
  -- Note that `A = ⋃ n, A ∩ disjointed s n`. We replace `A` with this sequence.
  obtain ⟨A, hAm, hAs, hAd, rfl⟩ :
    ∃ A' : ℕ → Set α,
      (∀ n, MeasurableSet (A' n)) ∧
        (∀ n, A' n ⊆ s n) ∧ Pairwise (Disjoint on A') ∧ A = ⋃ n, A' n := by
    refine
      ⟨fun n => A ∩ disjointed s n, fun n => hA.inter (MeasurableSet.disjointed hm _), fun n =>
        inter_subset_right.trans (disjointed_subset _ _),
        (disjoint_disjointed s).mono fun k l hkl => hkl.mono inf_le_right inf_le_right, ?_⟩
    rw [← inter_iUnion, iUnion_disjointed, univ_subset_iff.mp h'', inter_univ]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    inst✝ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Nat → Set α
    h : ∀ (n : Nat), (μ.restrict (s n)).OuterRegular
    h' : ∀ (n : Nat), IsOpen (s n)
    h'' : HasSubset.Subset Set.univ (Set.iUnion fun n => s n)
    r : ENNReal
    hm : ∀ (n : Nat), MeasurableSet (s n)
    A : Nat → Set α
    hAm : ∀ (n : Nat), MeasurableSet (A n)
    hAs : ∀ (n : Nat), HasSubset.Subset (A n) (s n)
    hAd : Pairwise (Function.onFun Disjoint A)
    hA : MeasurableSet (Set.iUnion fun n => A n)
    hr : GT.gt r (μ (Set.iUnion fun n => A n))
    HA : LT.lt (μ (Set.iUnion fun n => A n)) Top.top
    ⊢ Exists fun U => And (Superset U (Set.iUnion fun n => A n)) (And (IsOpen U) ( …
  -/
  rcases ENNReal.exists_pos_sum_of_countable' (tsub_pos_iff_lt.2 hr).ne' ℕ with ⟨δ, δ0, hδε⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    inst✝ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Nat → Set α
    h : ∀ (n : Nat), (μ.restrict (s n)).OuterRegular
    h' : ∀ (n : Nat), IsOpen (s n)
    h'' : HasSubset.Subset Set.univ (Set.iUnion fun n => s n)
    r : ENNReal
    hm : ∀ (n : Nat), MeasurableSet (s n)
    A : Nat → Set α
    hAm : ∀ (n : Nat), MeasurableSet (A n)
    hAs : ∀ (n : Nat), HasSubset.Subset (A n) (s n)
    hAd : Pairwise (Function.onFun Disjoint A)
    hA : MeasurableSet (Set.iUnion fun n => A n)
    hr : GT.gt r (μ (Set.iUnion fun n => A n))
    HA : LT.lt (μ (Set.iUnion fun n => A n)) Top.top
    δ : Nat → ENNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    hδε : LT.lt (tsum fun i => δ i) (HSub.hSub r (μ (Set.iUnion fun n => A n)))
    ⊢ Exists fun U => And (Superset U (Set.iUnion fun n => A n)) (And (IsOpen U) ( …
  -/
  rw [lt_tsub_iff_right, add_comm] at hδε
  have : ∀ n, ∃ U ⊇ A n, IsOpen U ∧ μ U < μ (A n) + δ n := by
    intro n
    have H₁ : ∀ t, μ.restrict (s n) t = μ (t ∩ s n) := fun t => restrict_apply' (hm n)
    have Ht : μ.restrict (s n) (A n) ≠ ∞ := by
      rw [H₁]
      exact ((measure_mono (inter_subset_left.trans (subset_iUnion A n))).trans_lt HA).ne
    rcases (A n).exists_isOpen_lt_add Ht (δ0 n).ne' with ⟨U, hAU, hUo, hU⟩
    rw [H₁, H₁, inter_eq_self_of_subset_left (hAs _)] at hU
    exact ⟨U ∩ s n, subset_inter hAU (hAs _), hUo.inter (h' n), hU⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    inst✝ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Nat → Set α
    h : ∀ (n : Nat), (μ.restrict (s n)).OuterRegular
    h' : ∀ (n : Nat), IsOpen (s n)
    h'' : HasSubset.Subset Set.univ (Set.iUnion fun n => s n)
    r : ENNReal
    hm : ∀ (n : Nat), MeasurableSet (s n)
    A : Nat → Set α
    hAm : ∀ (n : Nat), MeasurableSet (A n)
    hAs : ∀ (n : Nat), HasSubset.Subset (A n) (s n)
    hAd : Pairwise (Function.onFun Disjoint A)
    hA : MeasurableSet (Set.iUnion fun n => A n)
    hr : GT.gt r (μ (Set.iUnion fun n => A n))
    HA : LT.lt (μ (Set.iUnion fun n => A n)) Top.top
    δ : Nat → ENNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    hδε : LT.lt (HAdd.hAdd (μ (Set.iUnion fun n => A n)) (tsum fun i => δ i)) r
    this : ∀ (n : Nat), Exists fun U => And (Superset U (A n)) (And (IsOpen U) (LT …
    ⊢ Exists fun U => And (Superset U (Set.iUnion fun n => A n)) (And (IsOpen U) ( …
  -/
  choose U hAU hUo hU using this
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    inst✝ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Nat → Set α
    h : ∀ (n : Nat), (μ.restrict (s n)).OuterRegular
    h' : ∀ (n : Nat), IsOpen (s n)
    h'' : HasSubset.Subset Set.univ (Set.iUnion fun n => s n)
    r : ENNReal
    hm : ∀ (n : Nat), MeasurableSet (s n)
    A : Nat → Set α
    hAm : ∀ (n : Nat), MeasurableSet (A n)
    hAs : ∀ (n : Nat), HasSubset.Subset (A n) (s n)
    hAd : Pairwise (Function.onFun Disjoint A)
    hA : MeasurableSet (Set.iUnion fun n => A n)
    hr : GT.gt r (μ (Set.iUnion fun n => A n))
    HA : LT.lt (μ (Set.iUnion fun n => A n)) Top.top
    δ : Nat → ENNReal
    δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
    hδε : LT.lt (HAdd.hAdd (μ (Set.iUnion fun n => A n)) (tsum fun i => δ i)) r
    U : Nat → Set α
    hAU : ∀ (n : Nat), Superset (U n) (A n)
    hUo : ∀ (n : Nat), IsOpen (U n)
    hU : ∀ (n : Nat), LT.lt (μ (U n)) (HAdd.hAdd (μ (A n)) (δ n))
    ⊢ Exists fun U => And (Superset U (Set.iUnion fun n => A n)) (And (IsOpen U) ( …
  -/
  refine ⟨⋃ n, U n, iUnion_mono hAU, isOpen_iUnion hUo, ?_⟩
  calc
    μ (⋃ n, U n) ≤ ∑' n, μ (U n) := measure_iUnion_le _
    _ ≤ ∑' n, (μ (A n) + δ n) := ENNReal.tsum_le_tsum fun n => (hU n).le
    _ = ∑' n, μ (A n) + ∑' n, δ n := ENNReal.tsum_add
    _ = μ (⋃ n, A n) + ∑' n, δ n := (congr_arg₂ (· + ·) (measure_iUnion hAd hAm).symm rfl)
    _ < r := hδε


/-- See also `IsCompact.measure_closure` for a version
that assumes the `σ`-algebra to be the Borel `σ`-algebra but makes no assumptions on `μ`. -/
lemma measure_closure_eq_of_isCompact [R1Space α] [OuterRegular μ]
    {k : Set α} (hk : IsCompact k) : μ (closure k) = μ k := by
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace α
    inst✝¹ : R1Space α
    inst✝ : μ.OuterRegular
    k : Set α
    hk : IsCompact k
    ⊢ Eq (μ (closure k)) (μ k)
  -/
  apply le_antisymm ?_ (measure_mono subset_closure)
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace α
    inst✝¹ : R1Space α
    inst✝ : μ.OuterRegular
    k : Set α
    hk : IsCompact k
    ⊢ LE.le (μ (closure k)) (μ k)
  -/
  simp only [measure_eq_iInf_isOpen k, le_iInf_iff]
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace α
    inst✝¹ : R1Space α
    inst✝ : μ.OuterRegular
    k : Set α
    hk : IsCompact k
    ⊢ ∀ (i : Set α), HasSubset.Subset k i → IsOpen i → LE.le (μ (closure k)) (μ i)
  -/
  intro u ku u_open
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace α
    inst✝¹ : R1Space α
    inst✝ : μ.OuterRegular
    k : Set α
    hk : IsCompact k
    u : Set α
    ku : HasSubset.Subset k u
    u_open : IsOpen u
    ⊢ LE.le (μ (closure k)) (μ u)
  -/
  exact measure_mono (hk.closure_subset_of_isOpen u_open ku)
  /-
    🎉 no goals
  -/


/-- If a measure `μ` admits finite spanning open sets such that the restriction of `μ` to each set
is outer regular, then the original measure is outer regular as well. -/
protected theorem FiniteSpanningSetsIn.outerRegular
    [TopologicalSpace α] [OpensMeasurableSpace α] {μ : Measure α}
    (s : μ.FiniteSpanningSetsIn { U | IsOpen U ∧ OuterRegular (μ.restrict U) }) :
    OuterRegular μ :=
  OuterRegular.of_restrict (s := fun n ↦ s.set n) (fun n ↦ (s.set_mem n).2)
    (fun n ↦ (s.set_mem n).1) s.spanning.symm.subset


/-- If the restrictions of a measure to a monotone sequence of sets covering the space are
inner regular for some property `p` and all measurable sets, then the measure itself is
inner regular. -/
lemma of_restrict {μ : Measure α} {s : ℕ → Set α}
    (h : ∀ n, InnerRegularWRT (μ.restrict (s n)) p MeasurableSet)
    (hs : univ ⊆ ⋃ n, s n) (hmono : Monotone s) : InnerRegularWRT μ p MeasurableSet := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    p : Set α → Prop
    μ : MeasureTheory.Measure α
    s : Nat → Set α
    h : ∀ (n : Nat), (μ.restrict (s n)).InnerRegularWRT p MeasurableSet
    hs : HasSubset.Subset Set.univ (Set.iUnion fun n => s n)
    hmono : Monotone s
    ⊢ μ.InnerRegularWRT p MeasurableSet
  -/
  intro F hF r hr
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    p : Set α → Prop
    μ : MeasureTheory.Measure α
    s : Nat → Set α
    h : ∀ (n : Nat), (μ.restrict (s n)).InnerRegularWRT p MeasurableSet
    hs : HasSubset.Subset Set.univ (Set.iUnion fun n => s n)
    hmono : Monotone s
    F : Set α
    hF : MeasurableSet F
    r : ENNReal
    hr : LT.lt r (μ F)
    ⊢ Exists fun K => And (HasSubset.Subset K F) (And (p K) (LT.lt r (μ K)))
  -/
  have hBU : ⋃ n, F ∩ s n = F := by rw [← inter_iUnion, univ_subset_iff.mp hs, inter_univ]
  have : μ F = ⨆ n, μ (F ∩ s n) := by
    rw [← (monotone_const.inter hmono).measure_iUnion, hBU]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    p : Set α → Prop
    μ : MeasureTheory.Measure α
    s : Nat → Set α
    h : ∀ (n : Nat), (μ.restrict (s n)).InnerRegularWRT p MeasurableSet
    hs : HasSubset.Subset Set.univ (Set.iUnion fun n => s n)
    hmono : Monotone s
    F : Set α
    hF : MeasurableSet F
    r : ENNReal
    hr : LT.lt r (μ F)
    hBU : Eq (Set.iUnion fun n => Inter.inter F (s n)) F
    this : Eq (μ F) (iSup fun n => μ (Inter.inter F (s n)))
    ⊢ Exists fun K => And (HasSubset.Subset K F) (And (p K) (LT.lt r (μ K)))
  -/
  rw [this] at hr
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    p : Set α → Prop
    μ : MeasureTheory.Measure α
    s : Nat → Set α
    h : ∀ (n : Nat), (μ.restrict (s n)).InnerRegularWRT p MeasurableSet
    hs : HasSubset.Subset Set.univ (Set.iUnion fun n => s n)
    hmono : Monotone s
    F : Set α
    hF : MeasurableSet F
    r : ENNReal
    hr : LT.lt r (iSup fun n => μ (Inter.inter F (s n)))
    hBU : Eq (Set.iUnion fun n => Inter.inter F (s n)) F
    this : Eq (μ F) (iSup fun n => μ (Inter.inter F (s n)))
    ⊢ Exists fun K => And (HasSubset.Subset K F) (And (p K) (LT.lt r (μ K)))
  -/
  rcases lt_iSup_iff.1 hr with ⟨n, hn⟩
  /-
    case intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    p : Set α → Prop
    μ : MeasureTheory.Measure α
    s : Nat → Set α
    h : ∀ (n : Nat), (μ.restrict (s n)).InnerRegularWRT p MeasurableSet
    hs : HasSubset.Subset Set.univ (Set.iUnion fun n => s n)
    hmono : Monotone s
    F : Set α
    hF : MeasurableSet F
    r : ENNReal
    hr : LT.lt r (iSup fun n => μ (Inter.inter F (s n)))
    hBU : Eq (Set.iUnion fun n => Inter.inter F (s n)) F
    this : Eq (μ F) (iSup fun n => μ (Inter.inter F (s n)))
    n : Nat
    hn : LT.lt r (μ (Inter.inter F (s n)))
    ⊢ Exists fun K => And (HasSubset.Subset K F) (And (p K) (LT.lt r (μ K)))
  -/
  rw [← restrict_apply hF] at hn
  /-
    case intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    p : Set α → Prop
    μ : MeasureTheory.Measure α
    s : Nat → Set α
    h : ∀ (n : Nat), (μ.restrict (s n)).InnerRegularWRT p MeasurableSet
    hs : HasSubset.Subset Set.univ (Set.iUnion fun n => s n)
    hmono : Monotone s
    F : Set α
    hF : MeasurableSet F
    r : ENNReal
    hr : LT.lt r (iSup fun n => μ (Inter.inter F (s n)))
    hBU : Eq (Set.iUnion fun n => Inter.inter F (s n)) F
    this : Eq (μ F) (iSup fun n => μ (Inter.inter F (s n)))
    n : Nat
    hn : LT.lt r ((μ.restrict (s n)) F)
    ⊢ Exists fun K => And (HasSubset.Subset K F) (And (p K) (LT.lt r (μ K)))
  -/
  rcases h n hF _ hn with ⟨K, KF, hKp, hK⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    p : Set α → Prop
    μ : MeasureTheory.Measure α
    s : Nat → Set α
    h : ∀ (n : Nat), (μ.restrict (s n)).InnerRegularWRT p MeasurableSet
    hs : HasSubset.Subset Set.univ (Set.iUnion fun n => s n)
    hmono : Monotone s
    F : Set α
    hF : MeasurableSet F
    r : ENNReal
    hr : LT.lt r (iSup fun n => μ (Inter.inter F (s n)))
    hBU : Eq (Set.iUnion fun n => Inter.inter F (s n)) F
    this : Eq (μ F) (iSup fun n => μ (Inter.inter F (s n)))
    n : Nat
    hn : LT.lt r ((μ.restrict (s n)) F)
    K : Set α
    KF : HasSubset.Subset K F
    hKp : p K
    hK : LT.lt r ((μ.restrict (s n)) K)
    ⊢ Exists fun K => And (HasSubset.Subset K F) (And (p K) (LT.lt r (μ K)))
  -/
  exact ⟨K, KF, hKp, hK.trans_le (restrict_apply_le _ _)⟩
  /-
    🎉 no goals
  -/


/-- If `μ` is inner regular for measurable finite measure sets with respect to some class of sets,
then its restriction to any set is also inner regular for measurable finite measure sets, with
respect to the same class of sets. -/
lemma restrict (h : InnerRegularWRT μ p (fun s ↦ MeasurableSet s ∧ μ s ≠ ∞)) (A : Set α) :
    InnerRegularWRT (μ.restrict A) p (fun s ↦ MeasurableSet s ∧ μ.restrict A s ≠ ∞) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Set α → Prop
    h : μ.InnerRegularWRT p fun s => And (MeasurableSet s) (Ne (μ s) Top.top)
    A : Set α
    ⊢ (μ.restrict A).InnerRegularWRT p fun s => And (MeasurableSet s) (Ne ((μ.rest …
  -/
  rintro s ⟨s_meas, hs⟩ r hr
  /-
    case intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Set α → Prop
    h : μ.InnerRegularWRT p fun s => And (MeasurableSet s) (Ne (μ s) Top.top)
    A s : Set α
    s_meas : MeasurableSet s
    hs : Ne ((μ.restrict A) s) Top.top
    r : ENNReal
    hr : LT.lt r ((μ.restrict A) s)
    ⊢ Exists fun K => And (HasSubset.Subset K s) (And (p K) (LT.lt r ((μ.restrict  …
  -/
  rw [restrict_apply s_meas] at hs
  obtain ⟨K, K_subs, pK, rK⟩ : ∃ K, K ⊆ (toMeasurable μ (s ∩ A)) ∩ s ∧ p K ∧ r < μ K := by
    have : r < μ ((toMeasurable μ (s ∩ A)) ∩ s) := by
      apply hr.trans_le
      rw [restrict_apply s_meas]
      exact measure_mono <| subset_inter (subset_toMeasurable μ (s ∩ A)) inter_subset_left
    refine h ⟨(measurableSet_toMeasurable _ _).inter s_meas, ?_⟩ _ this
    apply (lt_of_le_of_lt _ hs.lt_top).ne
    rw [← measure_toMeasurable (s ∩ A)]
    exact measure_mono inter_subset_left
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Set α → Prop
    h : μ.InnerRegularWRT p fun s => And (MeasurableSet s) (Ne (μ s) Top.top)
    A s : Set α
    s_meas : MeasurableSet s
    hs : Ne (μ (Inter.inter s A)) Top.top
    r : ENNReal
    hr : LT.lt r ((μ.restrict A) s)
    K : Set α
    K_subs : HasSubset.Subset K (Inter.inter (MeasureTheory.toMeasurable μ (Inter. …
    pK : p K
    rK : LT.lt r (μ K)
    ⊢ Exists fun K => And (HasSubset.Subset K s) (And (p K) (LT.lt r ((μ.restrict  …
  -/
  refine ⟨K, K_subs.trans inter_subset_right, pK, ?_⟩
  calc
  r < μ K := rK
  _ = μ.restrict (toMeasurable μ (s ∩ A)) K := by
    rw [restrict_apply' (measurableSet_toMeasurable μ (s ∩ A))]
    congr
    apply (inter_eq_left.2 ?_).symm
    exact K_subs.trans inter_subset_left
  _ = μ.restrict (s ∩ A) K := by rwa [restrict_toMeasurable]
  _ ≤ μ.restrict A K := Measure.le_iff'.1 (restrict_mono inter_subset_right le_rfl) K


/-- If `μ` is inner regular for measurable finite measure sets with respect to some class of sets,
then its restriction to any finite measure set is also inner regular for measurable sets with
respect to the same class of sets. -/
lemma restrict_of_measure_ne_top (h : InnerRegularWRT μ p (fun s ↦ MeasurableSet s ∧ μ s ≠ ∞))
    {A : Set α} (hA : μ A ≠ ∞) :
    InnerRegularWRT (μ.restrict A) p (fun s ↦ MeasurableSet s) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Set α → Prop
    h : μ.InnerRegularWRT p fun s => And (MeasurableSet s) (Ne (μ s) Top.top)
    A : Set α
    hA : Ne (μ A) Top.top
    ⊢ (μ.restrict A).InnerRegularWRT p fun s => MeasurableSet s
  -/
  have : Fact (μ A < ∞) := ⟨hA.lt_top⟩
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Set α → Prop
    h : μ.InnerRegularWRT p fun s => And (MeasurableSet s) (Ne (μ s) Top.top)
    A : Set α
    hA : Ne (μ A) Top.top
    this : Fact (LT.lt (μ A) Top.top)
    ⊢ (μ.restrict A).InnerRegularWRT p fun s => MeasurableSet s
  -/
  exact (restrict h A).trans (of_imp (fun s hs ↦ ⟨hs, measure_ne_top _ _⟩))
  /-
    🎉 no goals
  -/


/-- Given a σ-finite measure, any measurable set can be approximated from inside by a measurable
set of finite measure. -/
lemma of_sigmaFinite [SigmaFinite μ] :
    InnerRegularWRT μ (fun s ↦ MeasurableSet s ∧ μ s ≠ ∞) (fun s ↦ MeasurableSet s) := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    ⊢ μ.InnerRegularWRT (fun s => And (MeasurableSet s) (Ne (μ s) Top.top)) fun s  …
  -/
  intro s hs r hr
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    hs : MeasurableSet s
    r : ENNReal
    hr : LT.lt r (μ s)
    ⊢ Exists fun K => And (HasSubset.Subset K s) (And ((fun s => And (MeasurableSe …
  -/
  set B : ℕ → Set α := spanningSets μ
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    hs : MeasurableSet s
    r : ENNReal
    hr : LT.lt r (μ s)
    B : Nat → Set α := MeasureTheory.spanningSets μ
    ⊢ Exists fun K => And (HasSubset.Subset K s) (And ((fun s => And (MeasurableSe …
  -/
  have hBU : ⋃ n, s ∩ B n = s := by rw [← inter_iUnion, iUnion_spanningSets, inter_univ]
  have : μ s = ⨆ n, μ (s ∩ B n) := by
    rw [← (monotone_const.inter (monotone_spanningSets μ)).measure_iUnion, hBU]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    hs : MeasurableSet s
    r : ENNReal
    hr : LT.lt r (μ s)
    B : Nat → Set α := MeasureTheory.spanningSets μ
    hBU : Eq (Set.iUnion fun n => Inter.inter s (B n)) s
    this : Eq (μ s) (iSup fun n => μ (Inter.inter s (B n)))
    ⊢ Exists fun K => And (HasSubset.Subset K s) (And ((fun s => And (MeasurableSe …
  -/
  rw [this] at hr
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    hs : MeasurableSet s
    r : ENNReal
    B : Nat → Set α := MeasureTheory.spanningSets μ
    hr : LT.lt r (iSup fun n => μ (Inter.inter s (B n)))
    hBU : Eq (Set.iUnion fun n => Inter.inter s (B n)) s
    this : Eq (μ s) (iSup fun n => μ (Inter.inter s (B n)))
    ⊢ Exists fun K => And (HasSubset.Subset K s) (And ((fun s => And (MeasurableSe …
  -/
  rcases lt_iSup_iff.1 hr with ⟨n, hn⟩
  /-
    case intro
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    hs : MeasurableSet s
    r : ENNReal
    B : Nat → Set α := MeasureTheory.spanningSets μ
    hr : LT.lt r (iSup fun n => μ (Inter.inter s (B n)))
    hBU : Eq (Set.iUnion fun n => Inter.inter s (B n)) s
    this : Eq (μ s) (iSup fun n => μ (Inter.inter s (B n)))
    n : Nat
    hn : LT.lt r (μ (Inter.inter s (B n)))
    ⊢ Exists fun K => And (HasSubset.Subset K s) (And ((fun s => And (MeasurableSe …
  -/
  refine ⟨s ∩ B n, inter_subset_left, ⟨hs.inter (measurableSet_spanningSets μ n), ?_⟩, hn⟩
  /-
    case intro
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    hs : MeasurableSet s
    r : ENNReal
    B : Nat → Set α := MeasureTheory.spanningSets μ
    hr : LT.lt r (iSup fun n => μ (Inter.inter s (B n)))
    hBU : Eq (Set.iUnion fun n => Inter.inter s (B n)) s
    this : Eq (μ s) (iSup fun n => μ (Inter.inter s (B n)))
    n : Nat
    hn : LT.lt r (μ (Inter.inter s (B n)))
    ⊢ Ne (μ (Inter.inter s (B n))) Top.top
  -/
  exact ((measure_mono inter_subset_right).trans_lt (measure_spanningSets_lt_top μ n)).ne
  /-
    🎉 no goals
  -/


/-- If a measure is inner regular (using closed or compact sets) for open sets, then every
measurable set of finite measure can be approximated by a (closed or compact) subset. -/
theorem measurableSet_of_isOpen [OuterRegular μ] (H : InnerRegularWRT μ p IsOpen)
    (hd : ∀ ⦃s U⦄, p s → IsOpen U → p (s \ U)) :
    InnerRegularWRT μ p fun s => MeasurableSet s ∧ μ s ≠ ∞ := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Set α → Prop
    inst✝¹ : TopologicalSpace α
    inst✝ : μ.OuterRegular
    H : μ.InnerRegularWRT p IsOpen
    hd : ∀ ⦃s U : Set α⦄, p s → IsOpen U → p (SDiff.sdiff s U)
    ⊢ μ.InnerRegularWRT p fun s => And (MeasurableSet s) (Ne (μ s) Top.top)
  -/
  rintro s ⟨hs, hμs⟩ r hr
  have h0 : p ∅ := by
    have : 0 < μ univ := (bot_le.trans_lt hr).trans_le (measure_mono (subset_univ _))
    obtain ⟨K, -, hK, -⟩ : ∃ K, K ⊆ univ ∧ p K ∧ 0 < μ K := H isOpen_univ _ this
    simpa using hd hK isOpen_univ
  obtain ⟨ε, hε, hεs, rfl⟩ : ∃ ε ≠ 0, ε + ε ≤ μ s ∧ r = μ s - (ε + ε) := by
    use (μ s - r) / 2
    simp [*, hr.le, ENNReal.add_halves, ENNReal.sub_sub_cancel, le_add_right, tsub_eq_zero_iff_le]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Set α → Prop
    inst✝¹ : TopologicalSpace α
    inst✝ : μ.OuterRegular
    H : μ.InnerRegularWRT p IsOpen
    hd : ∀ ⦃s U : Set α⦄, p s → IsOpen U → p (SDiff.sdiff s U)
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    h0 : p EmptyCollection.emptyCollection
    ε : ENNReal
    hε : Ne ε 0
    hεs : LE.le (HAdd.hAdd ε ε) (μ s)
    hr : LT.lt (HSub.hSub (μ s) (HAdd.hAdd ε ε)) (μ s)
    ⊢ Exists fun K => And (HasSubset.Subset K s) (And (p K) (LT.lt (HSub.hSub (μ s …
  -/
  rcases hs.exists_isOpen_diff_lt hμs hε with ⟨U, hsU, hUo, hUt, hμU⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Set α → Prop
    inst✝¹ : TopologicalSpace α
    inst✝ : μ.OuterRegular
    H : μ.InnerRegularWRT p IsOpen
    hd : ∀ ⦃s U : Set α⦄, p s → IsOpen U → p (SDiff.sdiff s U)
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    h0 : p EmptyCollection.emptyCollection
    ε : ENNReal
    hε : Ne ε 0
    hεs : LE.le (HAdd.hAdd ε ε) (μ s)
    hr : LT.lt (HSub.hSub (μ s) (HAdd.hAdd ε ε)) (μ s)
    U : Set α
    hsU : Superset U s
    hUo : IsOpen U
    hUt : LT.lt (μ U) Top.top
    hμU : LT.lt (μ (SDiff.sdiff U s)) ε
    ⊢ Exists fun K => And (HasSubset.Subset K s) (And (p K) (LT.lt (HSub.hSub (μ s …
  -/
  rcases (U \ s).exists_isOpen_lt_of_lt _ hμU with ⟨U', hsU', hU'o, hμU'⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Set α → Prop
    inst✝¹ : TopologicalSpace α
    inst✝ : μ.OuterRegular
    H : μ.InnerRegularWRT p IsOpen
    hd : ∀ ⦃s U : Set α⦄, p s → IsOpen U → p (SDiff.sdiff s U)
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    h0 : p EmptyCollection.emptyCollection
    ε : ENNReal
    hε : Ne ε 0
    hεs : LE.le (HAdd.hAdd ε ε) (μ s)
    hr : LT.lt (HSub.hSub (μ s) (HAdd.hAdd ε ε)) (μ s)
    U : Set α
    hsU : Superset U s
    hUo : IsOpen U
    hUt : LT.lt (μ U) Top.top
    hμU : LT.lt (μ (SDiff.sdiff U s)) ε
    U' : Set α
    hsU' : Superset U' (SDiff.sdiff U s)
    hU'o : IsOpen U'
    hμU' : LT.lt (μ U') ε
    ⊢ Exists fun K => And (HasSubset.Subset K s) (And (p K) (LT.lt (HSub.hSub (μ s …
  -/
  replace hsU' := diff_subset_comm.1 hsU'
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Set α → Prop
    inst✝¹ : TopologicalSpace α
    inst✝ : μ.OuterRegular
    H : μ.InnerRegularWRT p IsOpen
    hd : ∀ ⦃s U : Set α⦄, p s → IsOpen U → p (SDiff.sdiff s U)
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    h0 : p EmptyCollection.emptyCollection
    ε : ENNReal
    hε : Ne ε 0
    hεs : LE.le (HAdd.hAdd ε ε) (μ s)
    hr : LT.lt (HSub.hSub (μ s) (HAdd.hAdd ε ε)) (μ s)
    U : Set α
    hsU : Superset U s
    hUo : IsOpen U
    hUt : LT.lt (μ U) Top.top
    hμU : LT.lt (μ (SDiff.sdiff U s)) ε
    U' : Set α
    hU'o : IsOpen U'
    hμU' : LT.lt (μ U') ε
    hsU' : HasSubset.Subset (SDiff.sdiff U U') s
    ⊢ Exists fun K => And (HasSubset.Subset K s) (And (p K) (LT.lt (HSub.hSub (μ s …
  -/
  rcases H.exists_subset_lt_add h0 hUo hUt.ne hε with ⟨K, hKU, hKc, hKr⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : Set α → Prop
    inst✝¹ : TopologicalSpace α
    inst✝ : μ.OuterRegular
    H : μ.InnerRegularWRT p IsOpen
    hd : ∀ ⦃s U : Set α⦄, p s → IsOpen U → p (SDiff.sdiff s U)
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    h0 : p EmptyCollection.emptyCollection
    ε : ENNReal
    hε : Ne ε 0
    hεs : LE.le (HAdd.hAdd ε ε) (μ s)
    hr : LT.lt (HSub.hSub (μ s) (HAdd.hAdd ε ε)) (μ s)
    U : Set α
    hsU : Superset U s
    hUo : IsOpen U
    hUt : LT.lt (μ U) Top.top
    hμU : LT.lt (μ (SDiff.sdiff U s)) ε
    U' : Set α
    hU'o : IsOpen U'
    hμU' : LT.lt (μ U') ε
    hsU' : HasSubset.Subset (SDiff.sdiff U U') s
    K : Set α
    hKU : HasSubset.Subset K U
    hKc : p K
    hKr : LT.lt (μ U) (HAdd.hAdd (μ K) ε)
    ⊢ Exists fun K => And (HasSubset.Subset K s) (And (p K) (LT.lt (HSub.hSub (μ s …
  -/
  refine ⟨K \ U', fun x hx => hsU' ⟨hKU hx.1, hx.2⟩, hd hKc hU'o, ENNReal.sub_lt_of_lt_add hεs ?_⟩
  calc
    μ s ≤ μ U := μ.mono hsU
    _ < μ K + ε := hKr
    _ ≤ μ (K \ U') + μ U' + ε := add_le_add_right (tsub_le_iff_right.1 le_measure_diff) _
    _ ≤ μ (K \ U') + ε + ε := by gcongr
    _ = μ (K \ U') + (ε + ε) := add_assoc _ _ _


open Finset in
/-- In a finite measure space, assume that any open set can be approximated from inside by closed
sets. Then the measure is weakly regular. -/
theorem weaklyRegular_of_finite [BorelSpace α] (μ : Measure α) [IsFiniteMeasure μ]
    (H : InnerRegularWRT μ IsClosed IsOpen) : WeaklyRegular μ := by
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    inst✝² : TopologicalSpace α
    inst✝¹ : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    H : μ.InnerRegularWRT IsClosed IsOpen
    ⊢ μ.WeaklyRegular
  -/
  have hfin : ∀ {s}, μ s ≠ ∞ := @(measure_ne_top μ)
  suffices ∀ s, MeasurableSet s → ∀ ε, ε ≠ 0 → ∃ F, F ⊆ s ∧ ∃ U, U ⊇ s ∧
      IsClosed F ∧ IsOpen U ∧ μ s ≤ μ F + ε ∧ μ U ≤ μ s + ε by
    refine
      { outerRegular := fun s hs r hr => ?_
        innerRegular := H }
    rcases exists_between hr with ⟨r', hsr', hr'r⟩
    rcases this s hs _ (tsub_pos_iff_lt.2 hsr').ne' with ⟨-, -, U, hsU, -, hUo, -, H⟩
    refine ⟨U, hsU, hUo, ?_⟩
    rw [add_tsub_cancel_of_le hsr'.le] at H
    exact H.trans_lt hr'r
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    inst✝² : TopologicalSpace α
    inst✝¹ : BorelSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    H : μ.InnerRegularWRT IsClosed IsOpen
    hfin : ∀ {s : Set α}, Ne (μ s) Top.top
    ⊢ ∀ (s : Set α), MeasurableSet s → ∀ (ε : ENNReal), Ne ε 0 → Exists fun F => A …
  -/
  apply MeasurableSet.induction_on_open
  /- The proof is by measurable induction: we should check that the property is true for the empty
    set, for open sets, and is stable by taking the complement and by taking countable disjoint
    unions. The point of the property we are proving is that it is stable by taking complements
    (exchanging the roles of closed and open sets and thanks to the finiteness of the measure). -/
  -- check for open set
    /-
      case isOpen
      α : Type u_1
      inst✝³ : MeasurableSpace α
      inst✝² : TopologicalSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      H : μ.InnerRegularWRT IsClosed IsOpen
      hfin : ∀ {s : Set α}, Ne (μ s) Top.top
      ⊢ ∀ (U : Set α), IsOpen U → ∀ (ε : ENNReal), Ne ε 0 → Exists fun F => And (Has …
    -/
  · intro U hU ε hε
    /-
      case isOpen
      α : Type u_1
      inst✝³ : MeasurableSpace α
      inst✝² : TopologicalSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      H : μ.InnerRegularWRT IsClosed IsOpen
      hfin : ∀ {s : Set α}, Ne (μ s) Top.top
      U : Set α
      hU : IsOpen U
      ε : ENNReal
      hε : Ne ε 0
      ⊢ Exists fun F => And (HasSubset.Subset F U) (Exists fun U_1 => And (Superset  …
    -/
    rcases H.exists_subset_lt_add isClosed_empty hU hfin hε with ⟨F, hsF, hFc, hF⟩
    /-
      case isOpen.intro.intro.intro
      α : Type u_1
      inst✝³ : MeasurableSpace α
      inst✝² : TopologicalSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      H : μ.InnerRegularWRT IsClosed IsOpen
      hfin : ∀ {s : Set α}, Ne (μ s) Top.top
      U : Set α
      hU : IsOpen U
      ε : ENNReal
      hε : Ne ε 0
      F : Set α
      hsF : HasSubset.Subset F U
      hFc : IsClosed F
      hF : LT.lt (μ U) (HAdd.hAdd (μ F) ε)
      ⊢ Exists fun F => And (HasSubset.Subset F U) (Exists fun U_1 => And (Superset  …
    -/
    exact ⟨F, hsF, U, Subset.rfl, hFc, hU, hF.le, le_self_add⟩
    /-
      🎉 no goals
    -/
  -- check for complements
    /-
      case compl
      α : Type u_1
      inst✝³ : MeasurableSpace α
      inst✝² : TopologicalSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      H : μ.InnerRegularWRT IsClosed IsOpen
      hfin : ∀ {s : Set α}, Ne (μ s) Top.top
      ⊢ ∀ (t : Set α), MeasurableSet t → (∀ (ε : ENNReal), Ne ε 0 → Exists fun F =>  …
    -/
  · rintro s hs H ε hε
    /-
      case compl
      α : Type u_1
      inst✝³ : MeasurableSpace α
      inst✝² : TopologicalSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      H✝ : μ.InnerRegularWRT IsClosed IsOpen
      hfin : ∀ {s : Set α}, Ne (μ s) Top.top
      s : Set α
      hs : MeasurableSet s
      H : ∀ (ε : ENNReal), Ne ε 0 → Exists fun F => And (HasSubset.Subset F s) (Exis …
      ε : ENNReal
      hε : Ne ε 0
      ⊢ Exists fun F => And (HasSubset.Subset F (HasCompl.compl s)) (Exists fun U => …
    -/
    rcases H ε hε with ⟨F, hFs, U, hsU, hFc, hUo, hF, hU⟩
    refine
      ⟨Uᶜ, compl_subset_compl.2 hsU, Fᶜ, compl_subset_compl.2 hFs, hUo.isClosed_compl,
        hFc.isOpen_compl, ?_⟩
    /-
      case compl.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝³ : MeasurableSpace α
      inst✝² : TopologicalSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      H✝ : μ.InnerRegularWRT IsClosed IsOpen
      hfin : ∀ {s : Set α}, Ne (μ s) Top.top
      s : Set α
      hs : MeasurableSet s
      H : ∀ (ε : ENNReal), Ne ε 0 → Exists fun F => And (HasSubset.Subset F s) (Exis …
      ε : ENNReal
      hε : Ne ε 0
      F : Set α
      hFs : HasSubset.Subset F s
      U : Set α
      hsU : Superset U s
      hFc : IsClosed F
      hUo : IsOpen U
      hF : LE.le (μ s) (HAdd.hAdd (μ F) ε)
      hU : LE.le (μ U) (HAdd.hAdd (μ s) ε)
      ⊢ And (LE.le (μ (HasCompl.compl s)) (HAdd.hAdd (μ (HasCompl.compl U)) ε)) (LE. …
    -/
    simp only [measure_compl_le_add_iff, *, hUo.measurableSet, hFc.measurableSet, true_and]
    /-
      🎉 no goals
    -/
  -- check for disjoint unions
    /-
      case iUnion
      α : Type u_1
      inst✝³ : MeasurableSpace α
      inst✝² : TopologicalSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      H : μ.InnerRegularWRT IsClosed IsOpen
      hfin : ∀ {s : Set α}, Ne (μ s) Top.top
      ⊢ ∀ (f : Nat → Set α), Pairwise (Function.onFun Disjoint f) → (∀ (i : Nat), Me …
    -/
  · intro s hsd hsm H ε ε0
    /-
      case iUnion
      α : Type u_1
      inst✝³ : MeasurableSpace α
      inst✝² : TopologicalSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      H✝ : μ.InnerRegularWRT IsClosed IsOpen
      hfin : ∀ {s : Set α}, Ne (μ s) Top.top
      s : Nat → Set α
      hsd : Pairwise (Function.onFun Disjoint s)
      hsm : ∀ (i : Nat), MeasurableSet (s i)
      H : ∀ (i : Nat) (ε : ENNReal), Ne ε 0 → Exists fun F => And (HasSubset.Subset  …
      ε : ENNReal
      ε0 : Ne ε 0
      ⊢ Exists fun F => And (HasSubset.Subset F (Set.iUnion fun i => s i)) (Exists f …
    -/
    have ε0' : ε / 2 ≠ 0 := (ENNReal.half_pos ε0).ne'
    /-
      case iUnion
      α : Type u_1
      inst✝³ : MeasurableSpace α
      inst✝² : TopologicalSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      H✝ : μ.InnerRegularWRT IsClosed IsOpen
      hfin : ∀ {s : Set α}, Ne (μ s) Top.top
      s : Nat → Set α
      hsd : Pairwise (Function.onFun Disjoint s)
      hsm : ∀ (i : Nat), MeasurableSet (s i)
      H : ∀ (i : Nat) (ε : ENNReal), Ne ε 0 → Exists fun F => And (HasSubset.Subset  …
      ε : ENNReal
      ε0 : Ne ε 0
      ε0' : Ne (HDiv.hDiv ε 2) 0
      ⊢ Exists fun F => And (HasSubset.Subset F (Set.iUnion fun i => s i)) (Exists f …
    -/
    rcases ENNReal.exists_pos_sum_of_countable' ε0' ℕ with ⟨δ, δ0, hδε⟩
    /-
      case iUnion.intro.intro
      α : Type u_1
      inst✝³ : MeasurableSpace α
      inst✝² : TopologicalSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      H✝ : μ.InnerRegularWRT IsClosed IsOpen
      hfin : ∀ {s : Set α}, Ne (μ s) Top.top
      s : Nat → Set α
      hsd : Pairwise (Function.onFun Disjoint s)
      hsm : ∀ (i : Nat), MeasurableSet (s i)
      H : ∀ (i : Nat) (ε : ENNReal), Ne ε 0 → Exists fun F => And (HasSubset.Subset  …
      ε : ENNReal
      ε0 : Ne ε 0
      ε0' : Ne (HDiv.hDiv ε 2) 0
      δ : Nat → ENNReal
      δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
      hδε : LT.lt (tsum fun i => δ i) (HDiv.hDiv ε 2)
      ⊢ Exists fun F => And (HasSubset.Subset F (Set.iUnion fun i => s i)) (Exists f …
    -/
    choose F hFs U hsU hFc hUo hF hU using fun n => H n (δ n) (δ0 n).ne'
    -- the approximating closed set is constructed by considering finitely many sets `s i`, which
    -- cover all the measure up to `ε/2`, approximating each of these by a closed set `F i`, and
    -- taking the union of these (finitely many) `F i`.
    have : Tendsto (fun t => (∑ k ∈ t, μ (s k)) + ε / 2) atTop (𝓝 <| μ (⋃ n, s n) + ε / 2) := by
      rw [measure_iUnion hsd hsm]
      exact Tendsto.add ENNReal.summable.hasSum tendsto_const_nhds
    /-
      case iUnion.intro.intro
      α : Type u_1
      inst✝³ : MeasurableSpace α
      inst✝² : TopologicalSpace α
      inst✝¹ : BorelSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      H✝ : μ.InnerRegularWRT IsClosed IsOpen
      hfin : ∀ {s : Set α}, Ne (μ s) Top.top
      s : Nat → Set α
      hsd : Pairwise (Function.onFun Disjoint s)
      hsm : ∀ (i : Nat), MeasurableSet (s i)
      H : ∀ (i : Nat) (ε : ENNReal), Ne ε 0 → Exists fun F => And (HasSubset.Subset  …
      ε : ENNReal
      ε0 : Ne ε 0
      ε0' : Ne (HDiv.hDiv ε 2) 0
      δ : Nat → ENNReal
      δ0 : ∀ (i : Nat), LT.lt 0 (δ i)
      hδε : LT.lt (tsum fun i => δ i) (HDiv.hDiv ε 2)
      F : Nat → Set α
      hFs : ∀ (n : Nat), HasSubset.Subset (F n) (s n)
      U : Nat → Set α
      hsU : ∀ (n : Nat), Superset (U n) (s n)
      hFc : ∀ (n : Nat), IsClosed (F n)
      hUo : ∀ (n : Nat), IsOpen (U n)
      hF : ∀ (n : Nat), LE.le (μ (s n)) (HAdd.hAdd (μ (F n)) (δ n))
      hU : ∀ (n : Nat), LE.le (μ (U n)) (HAdd.hAdd (μ (s n)) (δ n))
      this : Filter.Tendsto (fun t => HAdd.hAdd (t.sum fun k => μ (s k)) (HDiv.hDiv  …
      ⊢ Exists fun F => And (HasSubset.Subset F (Set.iUnion fun i => s i)) (Exists f …
    -/
    rcases (this.eventually <| lt_mem_nhds <| ENNReal.lt_add_right hfin ε0').exists with ⟨t, ht⟩
    -- the approximating open set is constructed by taking for each `s n` an approximating open set
    -- `U n` with measure at most `μ (s n) + δ n` for a summable `δ`, and taking the union of these.
    refine
      ⟨⋃ k ∈ t, F k, iUnion_mono fun k => iUnion_subset fun _ => hFs _, ⋃ n, U n, iUnion_mono hsU,
        isClosed_biUnion_finset fun k _ => hFc k, isOpen_iUnion hUo, ht.le.trans ?_, ?_⟩
    · calc
        (∑ k ∈ t, μ (s k)) + ε / 2 ≤ ((∑ k ∈ t, μ (F k)) + ∑ k ∈ t, δ k) + ε / 2 := by
          rw [← sum_add_distrib]
          gcongr
          apply hF
        _ ≤ (∑ k ∈ t, μ (F k)) + ε / 2 + ε / 2 := by
          gcongr
          exact (ENNReal.sum_le_tsum _).trans hδε.le
        _ = μ (⋃ k ∈ t, F k) + ε := by
          rw [measure_biUnion_finset, add_assoc, ENNReal.add_halves]
          exacts [fun k _ n _ hkn => (hsd hkn).mono (hFs k) (hFs n),
            fun k _ => (hFc k).measurableSet]
    · calc
        μ (⋃ n, U n) ≤ ∑' n, μ (U n) := measure_iUnion_le _
        _ ≤ ∑' n, (μ (s n) + δ n) := ENNReal.tsum_le_tsum hU
        _ = μ (⋃ n, s n) + ∑' n, δ n := by rw [measure_iUnion hsd hsm, ENNReal.tsum_add]
        _ ≤ μ (⋃ n, s n) + ε := add_le_add_left (hδε.le.trans ENNReal.half_le_self) _


/-- In a metrizable space (or even a pseudo metrizable space), an open set can be approximated from
inside by closed sets. -/
theorem of_pseudoMetrizableSpace {X : Type*} [TopologicalSpace X] [PseudoMetrizableSpace X]
    [MeasurableSpace X] (μ : Measure X) : InnerRegularWRT μ IsClosed IsOpen := by
  /-
    X : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace X
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    ⊢ μ.InnerRegularWRT IsClosed IsOpen
  -/
  let A : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetric X
  /-
    X : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace X
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    A : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetric X
    ⊢ μ.InnerRegularWRT IsClosed IsOpen
  -/
  intro U hU r hr
  /-
    X : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace X
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    A : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetric X
    U : Set X
    hU : IsOpen U
    r : ENNReal
    hr : LT.lt r (μ U)
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (IsClosed K) (LT.lt r (μ K)))
  -/
  rcases hU.exists_iUnion_isClosed with ⟨F, F_closed, -, rfl, F_mono⟩
  /-
    case intro.intro.intro.intro
    X : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace X
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    A : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetric X
    r : ENNReal
    F : Nat → Set X
    F_closed : ∀ (n : Nat), IsClosed (F n)
    F_mono : Monotone F
    hU : IsOpen (Set.iUnion fun n => F n)
    hr : LT.lt r (μ (Set.iUnion fun n => F n))
    ⊢ Exists fun K => And (HasSubset.Subset K (Set.iUnion fun n => F n)) (And (IsC …
  -/
  rw [F_mono.measure_iUnion] at hr
  /-
    case intro.intro.intro.intro
    X : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace X
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    A : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetric X
    r : ENNReal
    F : Nat → Set X
    F_closed : ∀ (n : Nat), IsClosed (F n)
    F_mono : Monotone F
    hU : IsOpen (Set.iUnion fun n => F n)
    hr : LT.lt r (iSup fun i => μ (F i))
    ⊢ Exists fun K => And (HasSubset.Subset K (Set.iUnion fun n => F n)) (And (IsC …
  -/
  rcases lt_iSup_iff.1 hr with ⟨n, hn⟩
  /-
    case intro.intro.intro.intro.intro
    X : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace X
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    A : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetric X
    r : ENNReal
    F : Nat → Set X
    F_closed : ∀ (n : Nat), IsClosed (F n)
    F_mono : Monotone F
    hU : IsOpen (Set.iUnion fun n => F n)
    hr : LT.lt r (iSup fun i => μ (F i))
    n : Nat
    hn : LT.lt r (μ (F n))
    ⊢ Exists fun K => And (HasSubset.Subset K (Set.iUnion fun n => F n)) (And (IsC …
  -/
  exact ⟨F n, subset_iUnion _ _, F_closed n, hn⟩
  /-
    🎉 no goals
  -/


/-- In a `σ`-compact space, any closed set can be approximated by a compact subset. -/
theorem isCompact_isClosed {X : Type*} [TopologicalSpace X] [SigmaCompactSpace X]
    [MeasurableSpace X] (μ : Measure X) : InnerRegularWRT μ IsCompact IsClosed := by
  /-
    X : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : SigmaCompactSpace X
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    ⊢ μ.InnerRegularWRT IsCompact IsClosed
  -/
  intro F hF r hr
  /-
    X : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : SigmaCompactSpace X
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    F : Set X
    hF : IsClosed F
    r : ENNReal
    hr : LT.lt r (μ F)
    ⊢ Exists fun K => And (HasSubset.Subset K F) (And (IsCompact K) (LT.lt r (μ K)))
  -/
  set B : ℕ → Set X := compactCovering X
  /-
    X : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : SigmaCompactSpace X
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    F : Set X
    hF : IsClosed F
    r : ENNReal
    hr : LT.lt r (μ F)
    B : Nat → Set X := compactCovering X
    ⊢ Exists fun K => And (HasSubset.Subset K F) (And (IsCompact K) (LT.lt r (μ K)))
  -/
  have hBc : ∀ n, IsCompact (F ∩ B n) := fun n => (isCompact_compactCovering X n).inter_left hF
  /-
    X : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : SigmaCompactSpace X
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    F : Set X
    hF : IsClosed F
    r : ENNReal
    hr : LT.lt r (μ F)
    B : Nat → Set X := compactCovering X
    hBc : ∀ (n : Nat), IsCompact (Inter.inter F (B n))
    ⊢ Exists fun K => And (HasSubset.Subset K F) (And (IsCompact K) (LT.lt r (μ K)))
  -/
  have hBU : ⋃ n, F ∩ B n = F := by rw [← inter_iUnion, iUnion_compactCovering, Set.inter_univ]
  have : μ F = ⨆ n, μ (F ∩ B n) := by
    rw [← Monotone.measure_iUnion, hBU]
    exact monotone_const.inter monotone_accumulate
  /-
    X : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : SigmaCompactSpace X
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    F : Set X
    hF : IsClosed F
    r : ENNReal
    hr : LT.lt r (μ F)
    B : Nat → Set X := compactCovering X
    hBc : ∀ (n : Nat), IsCompact (Inter.inter F (B n))
    hBU : Eq (Set.iUnion fun n => Inter.inter F (B n)) F
    this : Eq (μ F) (iSup fun n => μ (Inter.inter F (B n)))
    ⊢ Exists fun K => And (HasSubset.Subset K F) (And (IsCompact K) (LT.lt r (μ K)))
  -/
  rw [this] at hr
  /-
    X : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : SigmaCompactSpace X
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    F : Set X
    hF : IsClosed F
    r : ENNReal
    B : Nat → Set X := compactCovering X
    hr : LT.lt r (iSup fun n => μ (Inter.inter F (B n)))
    hBc : ∀ (n : Nat), IsCompact (Inter.inter F (B n))
    hBU : Eq (Set.iUnion fun n => Inter.inter F (B n)) F
    this : Eq (μ F) (iSup fun n => μ (Inter.inter F (B n)))
    ⊢ Exists fun K => And (HasSubset.Subset K F) (And (IsCompact K) (LT.lt r (μ K)))
  -/
  rcases lt_iSup_iff.1 hr with ⟨n, hn⟩
  /-
    case intro
    X : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : SigmaCompactSpace X
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    F : Set X
    hF : IsClosed F
    r : ENNReal
    B : Nat → Set X := compactCovering X
    hr : LT.lt r (iSup fun n => μ (Inter.inter F (B n)))
    hBc : ∀ (n : Nat), IsCompact (Inter.inter F (B n))
    hBU : Eq (Set.iUnion fun n => Inter.inter F (B n)) F
    this : Eq (μ F) (iSup fun n => μ (Inter.inter F (B n)))
    n : Nat
    hn : LT.lt r (μ (Inter.inter F (B n)))
    ⊢ Exists fun K => And (HasSubset.Subset K F) (And (IsCompact K) (LT.lt r (μ K)))
  -/
  exact ⟨_, inter_subset_left, hBc n, hn⟩
  /-
    🎉 no goals
  -/


/-- The measure of a measurable set is the supremum of the measures of compact sets it contains. -/
theorem _root_.MeasurableSet.measure_eq_iSup_isCompact ⦃U : Set α⦄ (hU : MeasurableSet U)
    (μ : Measure α) [InnerRegular μ] :
    μ U = ⨆ (K : Set α) (_ : K ⊆ U) (_ : IsCompact K), μ K :=
  InnerRegular.innerRegular.measure_eq_iSup hU


instance zero : InnerRegular (0 : Measure α) :=
  ⟨fun _ _ _r hr => ⟨∅, empty_subset _, isCompact_empty, hr⟩⟩


instance smul [h : InnerRegular μ] (c : ℝ≥0∞) : InnerRegular (c • μ) :=
  ⟨InnerRegularWRT.smul h.innerRegular c⟩


instance smul_nnreal [InnerRegular μ] (c : ℝ≥0) : InnerRegular (c • μ) := smul (c : ℝ≥0∞)


instance (priority := 100) [InnerRegular μ] : InnerRegularCompactLTTop μ :=
  ⟨fun _s hs r hr ↦ InnerRegular.innerRegular hs.1 r hr⟩


lemma innerRegularWRT_isClosed_isOpen [R1Space α] [OpensMeasurableSpace α] [h : InnerRegular μ] :
    InnerRegularWRT μ IsClosed IsOpen := by
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace α
    inst✝¹ : R1Space α
    inst✝ : OpensMeasurableSpace α
    h : μ.InnerRegular
    ⊢ μ.InnerRegularWRT IsClosed IsOpen
  -/
  intro U hU r hr
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace α
    inst✝¹ : R1Space α
    inst✝ : OpensMeasurableSpace α
    h : μ.InnerRegular
    U : Set α
    hU : IsOpen U
    r : ENNReal
    hr : LT.lt r (μ U)
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (IsClosed K) (LT.lt r (μ K)))
  -/
  rcases h.innerRegular hU.measurableSet r hr with ⟨K, KU, K_comp, hK⟩
  exact ⟨closure K, K_comp.closure_subset_of_isOpen hU KU, isClosed_closure,
    hK.trans_le (measure_mono subset_closure)⟩


theorem exists_isCompact_not_null [InnerRegular μ] : (∃ K, IsCompact K ∧ μ K ≠ 0) ↔ μ ≠ 0 := by
  simp_rw [Ne, ← measure_univ_eq_zero, MeasurableSet.univ.measure_eq_iSup_isCompact,
    ENNReal.iSup_eq_zero, not_forall, exists_prop, subset_univ, true_and]

@[deprecated (since := "2024-11-19")] alias exists_compact_not_null := exists_isCompact_not_null


/-- If `μ` is inner regular, then any measurable set can be approximated by a compact subset.
See also `MeasurableSet.exists_isCompact_lt_add_of_ne_top`. -/
theorem _root_.MeasurableSet.exists_lt_isCompact [InnerRegular μ] ⦃A : Set α⦄
    (hA : MeasurableSet A) {r : ℝ≥0∞} (hr : r < μ A) :
    ∃ K, K ⊆ A ∧ IsCompact K ∧ r < μ K :=
  InnerRegular.innerRegular hA _ hr


protected theorem map_of_continuous [BorelSpace α] [MeasurableSpace β] [TopologicalSpace β]
    [BorelSpace β] [h : InnerRegular μ] {f : α → β} (hf : Continuous f) :
    InnerRegular (Measure.map f μ) :=
  ⟨InnerRegularWRT.map h.innerRegular hf.aemeasurable (fun _s hs ↦ hf.measurable hs)
    (fun _K hK ↦ hK.image hf) (fun _s hs ↦ hs)⟩


protected theorem map [BorelSpace α] [MeasurableSpace β] [TopologicalSpace β]
    [BorelSpace β] [InnerRegular μ] (f : α ≃ₜ β) : (Measure.map f μ).InnerRegular :=
  InnerRegular.map_of_continuous f.continuous


protected theorem map_iff [BorelSpace α] [MeasurableSpace β] [TopologicalSpace β]
    [BorelSpace β] (f : α ≃ₜ β) :
    InnerRegular (Measure.map f μ) ↔ InnerRegular μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : TopologicalSpace β
    inst✝ : BorelSpace β
    f : Homeomorph α β
    ⊢ Iff (MeasureTheory.Measure.map (⇑f) μ).InnerRegular μ.InnerRegular
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ h.map f⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : TopologicalSpace β
    inst✝ : BorelSpace β
    f : Homeomorph α β
    h : (MeasureTheory.Measure.map (⇑f) μ).InnerRegular
    ⊢ μ.InnerRegular
  -/
  convert h.map f.symm
  /-
    case h.e'_4
    α : Type u_1
    β : Type u_2
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : TopologicalSpace β
    inst✝ : BorelSpace β
    f : Homeomorph α β
    h : (MeasureTheory.Measure.map (⇑f) μ).InnerRegular
    ⊢ Eq μ (MeasureTheory.Measure.map (⇑f.symm) (MeasureTheory.Measure.map (⇑f) μ))
  -/
  rw [map_map f.symm.continuous.measurable f.continuous.measurable]
  /-
    case h.e'_4
    α : Type u_1
    β : Type u_2
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : TopologicalSpace β
    inst✝ : BorelSpace β
    f : Homeomorph α β
    h : (MeasureTheory.Measure.map (⇑f) μ).InnerRegular
    ⊢ Eq μ (MeasureTheory.Measure.map (Function.comp ⇑f.symm ⇑f) μ)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If `μ` is inner regular for finite measure sets with respect to compact sets,
then any measurable set of finite measure can be approximated by a
compact subset. See also `MeasurableSet.exists_lt_isCompact_of_ne_top`. -/
theorem _root_.MeasurableSet.exists_isCompact_lt_add [InnerRegularCompactLTTop μ]
    ⦃A : Set α⦄ (hA : MeasurableSet A) (h'A : μ A ≠ ∞) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ K, K ⊆ A ∧ IsCompact K ∧ μ A < μ K + ε :=
  InnerRegularCompactLTTop.innerRegular.exists_subset_lt_add isCompact_empty ⟨hA, h'A⟩ h'A hε


/-- If `μ` is inner regular for finite measure sets with respect to compact sets,
then any measurable set of finite measure can be approximated by a compact closed subset.
Compared to `MeasurableSet.exists_isCompact_lt_add`,
this version additionally assumes that `α` is an R₁ space with Borel σ-algebra.
-/
theorem _root_.MeasurableSet.exists_isCompact_isClosed_lt_add
    [InnerRegularCompactLTTop μ] [R1Space α] [BorelSpace α]
    ⦃A : Set α⦄ (hA : MeasurableSet A) (h'A : μ A ≠ ∞) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ K, K ⊆ A ∧ IsCompact K ∧ IsClosed K ∧ μ A < μ K + ε :=
  let ⟨K, hKA, hK, hμK⟩ := hA.exists_isCompact_lt_add h'A hε
  ⟨closure K, hK.closure_subset_measurableSet hA hKA, hK.closure, isClosed_closure,
       /-
         α : Type u_1
         inst✝⁴ : MeasurableSpace α
         μ : MeasureTheory.Measure α
         inst✝³ : TopologicalSpace α
         inst✝² : μ.InnerRegularCompactLTTop
         inst✝¹ : R1Space α
         inst✝ : BorelSpace α
         A : Set α
         hA : MeasurableSet A
         h'A : Ne (μ A) Top.top
         ε : ENNReal
         hε : Ne ε 0
         K : Set α
         hKA : HasSubset.Subset K A
         hK : IsCompact K
         hμK : LT.lt (μ A) (HAdd.hAdd (μ K) ε)
         ⊢ LT.lt (μ A) (HAdd.hAdd (μ (closure K)) ε)
       -/
    by rwa [hK.measure_closure]⟩
       /-
         🎉 no goals
       -/


/-- If `μ` is inner regular for finite measure sets with respect to compact sets,
then any measurable set of finite measure can be approximated by a
compact subset. See also `MeasurableSet.exists_isCompact_lt_add` and
`MeasurableSet.exists_lt_isCompact_of_ne_top`. -/
theorem _root_.MeasurableSet.exists_isCompact_diff_lt [OpensMeasurableSpace α] [T2Space α]
    [InnerRegularCompactLTTop μ]  ⦃A : Set α⦄ (hA : MeasurableSet A) (h'A : μ A ≠ ∞)
    {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ K, K ⊆ A ∧ IsCompact K ∧ μ (A \ K) < ε := by
  /-
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : T2Space α
    inst✝ : μ.InnerRegularCompactLTTop
    A : Set α
    hA : MeasurableSet A
    h'A : Ne (μ A) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun K => And (HasSubset.Subset K A) (And (IsCompact K) (LT.lt (μ (SDi …
  -/
  rcases hA.exists_isCompact_lt_add h'A hε with ⟨K, hKA, hKc, hK⟩
  exact ⟨K, hKA, hKc, measure_diff_lt_of_lt_add hKc.nullMeasurableSet hKA
    (ne_top_of_le_ne_top h'A <| measure_mono hKA) hK⟩


/-- If `μ` is inner regular for finite measure sets with respect to compact sets,
then any measurable set of finite measure can be approximated by a compact closed subset.
Compared to `MeasurableSet.exists_isCompact_diff_lt`,
this lemma additionally assumes that `α` is an R₁ space with Borel σ-algebra. -/
theorem _root_.MeasurableSet.exists_isCompact_isClosed_diff_lt [BorelSpace α] [R1Space α]
    [InnerRegularCompactLTTop μ] ⦃A : Set α⦄ (hA : MeasurableSet A) (h'A : μ A ≠ ∞)
    {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ K, K ⊆ A ∧ IsCompact K ∧ IsClosed K ∧ μ (A \ K) < ε := by
  /-
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace α
    inst✝² : BorelSpace α
    inst✝¹ : R1Space α
    inst✝ : μ.InnerRegularCompactLTTop
    A : Set α
    hA : MeasurableSet A
    h'A : Ne (μ A) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun K => And (HasSubset.Subset K A) (And (IsCompact K) (And (IsClosed …
  -/
  rcases hA.exists_isCompact_isClosed_lt_add h'A hε with ⟨K, hKA, hKco, hKcl, hK⟩
  exact ⟨K, hKA, hKco, hKcl, measure_diff_lt_of_lt_add hKcl.nullMeasurableSet hKA
    (ne_top_of_le_ne_top h'A <| measure_mono hKA) hK⟩


/-- If `μ` is inner regular for finite measure sets with respect to compact sets,
then any measurable set of finite measure can be approximated by a
compact subset. See also `MeasurableSet.exists_isCompact_lt_add`. -/
theorem _root_.MeasurableSet.exists_lt_isCompact_of_ne_top [InnerRegularCompactLTTop μ] ⦃A : Set α⦄
    (hA : MeasurableSet A) (h'A : μ A ≠ ∞) {r : ℝ≥0∞} (hr : r < μ A) :
    ∃ K, K ⊆ A ∧ IsCompact K ∧ r < μ K :=
  InnerRegularCompactLTTop.innerRegular ⟨hA, h'A⟩ _ hr


/-- If `μ` is inner regular for finite measure sets with respect to compact sets,
any measurable set of finite mass can be approximated from inside by compact sets. -/
theorem _root_.MeasurableSet.measure_eq_iSup_isCompact_of_ne_top [InnerRegularCompactLTTop μ]
    ⦃A : Set α⦄ (hA : MeasurableSet A) (h'A : μ A ≠ ∞) :
    μ A = ⨆ (K) (_ : K ⊆ A) (_ : IsCompact K), μ K :=
  InnerRegularCompactLTTop.innerRegular.measure_eq_iSup ⟨hA, h'A⟩


/-- If `μ` is inner regular for finite measure sets with respect to compact sets, then its
restriction to any set also is. -/
instance restrict [h : InnerRegularCompactLTTop μ] (A : Set α) :
    InnerRegularCompactLTTop (μ.restrict A) :=
  ⟨InnerRegularWRT.restrict h.innerRegular A⟩


instance (priority := 50) [h : InnerRegularCompactLTTop μ] [IsFiniteMeasure μ] :
    InnerRegular μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace α
    h : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ μ.InnerRegular
  -/
  constructor
  /-
    case innerRegular
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace α
    h : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ μ.InnerRegularWRT IsCompact MeasurableSet
  -/
  convert h.innerRegular with s
  /-
    case h.e'_5.h.a
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace α
    h : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Set α
    ⊢ Iff (MeasurableSet s) (And (MeasurableSet s) (Ne (μ s) Top.top))
  -/
  simp [measure_ne_top μ s]
  /-
    🎉 no goals
  -/


instance (priority := 50) [BorelSpace α] [R1Space α] [InnerRegularCompactLTTop μ]
    [IsFiniteMeasure μ] : WeaklyRegular μ :=
  InnerRegular.innerRegularWRT_isClosed_isOpen.weaklyRegular_of_finite _


instance (priority := 50) [BorelSpace α] [R1Space α] [h : InnerRegularCompactLTTop μ]
    [IsFiniteMeasure μ] : Regular μ where
  innerRegular := InnerRegularWRT.trans h.innerRegular <|
    InnerRegularWRT.of_imp (fun U hU ↦ ⟨hU.measurableSet, measure_ne_top μ U⟩)


protected lemma _root_.IsCompact.exists_isOpen_lt_of_lt [InnerRegularCompactLTTop μ]
    [IsLocallyFiniteMeasure μ] [R1Space α] [BorelSpace α] {K : Set α}
    (hK : IsCompact K) (r : ℝ≥0∞) (hr : μ K < r) :
    ∃ U, K ⊆ U ∧ IsOpen U ∧ μ U < r := by
  /-
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : R1Space α
    inst✝ : BorelSpace α
    K : Set α
    hK : IsCompact K
    r : ENNReal
    hr : LT.lt (μ K) r
    ⊢ Exists fun U => And (HasSubset.Subset K U) (And (IsOpen U) (LT.lt (μ U) r))
  -/
  rcases hK.exists_open_superset_measure_lt_top μ with ⟨V, hKV, hVo, hμV⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : R1Space α
    inst✝ : BorelSpace α
    K : Set α
    hK : IsCompact K
    r : ENNReal
    hr : LT.lt (μ K) r
    V : Set α
    hKV : Superset V K
    hVo : IsOpen V
    hμV : LT.lt (μ V) Top.top
    ⊢ Exists fun U => And (HasSubset.Subset K U) (And (IsOpen U) (LT.lt (μ U) r))
  -/
  have := Fact.mk hμV
  obtain ⟨U, hKU, hUo, hμU⟩ : ∃ U, K ⊆ U ∧ IsOpen U ∧ μ.restrict V U < r :=
    exists_isOpen_lt_of_lt K r <| (restrict_apply_le _ _).trans_lt hr
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : R1Space α
    inst✝ : BorelSpace α
    K : Set α
    hK : IsCompact K
    r : ENNReal
    hr : LT.lt (μ K) r
    V : Set α
    hKV : Superset V K
    hVo : IsOpen V
    hμV : LT.lt (μ V) Top.top
    this : Fact (LT.lt (μ V) Top.top)
    U : Set α
    hKU : HasSubset.Subset K U
    hUo : IsOpen U
    hμU : LT.lt ((μ.restrict V) U) r
    ⊢ Exists fun U => And (HasSubset.Subset K U) (And (IsOpen U) (LT.lt (μ U) r))
  -/
  refine ⟨U ∩ V, subset_inter hKU hKV, hUo.inter hVo, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : R1Space α
    inst✝ : BorelSpace α
    K : Set α
    hK : IsCompact K
    r : ENNReal
    hr : LT.lt (μ K) r
    V : Set α
    hKV : Superset V K
    hVo : IsOpen V
    hμV : LT.lt (μ V) Top.top
    this : Fact (LT.lt (μ V) Top.top)
    U : Set α
    hKU : HasSubset.Subset K U
    hUo : IsOpen U
    hμU : LT.lt ((μ.restrict V) U) r
    ⊢ LT.lt (μ (Inter.inter U V)) r
  -/
  rwa [restrict_apply hUo.measurableSet] at hμU
  /-
    🎉 no goals
  -/


/-- If `μ` is inner regular for finite measure sets with respect to compact sets
and is locally finite in an R₁ space,
then any compact set can be approximated from outside by open sets. -/
protected lemma _root_.IsCompact.measure_eq_iInf_isOpen [InnerRegularCompactLTTop μ]
    [IsLocallyFiniteMeasure μ] [R1Space α] [BorelSpace α] {K : Set α} (hK : IsCompact K) :
    μ K = ⨅ (U : Set α) (_ : K ⊆ U) (_ : IsOpen U), μ U := by
  /-
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : R1Space α
    inst✝ : BorelSpace α
    K : Set α
    hK : IsCompact K
    ⊢ Eq (μ K) (iInf fun U => iInf fun x => iInf fun x => μ U)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      inst✝⁵ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : μ.InnerRegularCompactLTTop
      inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝¹ : R1Space α
      inst✝ : BorelSpace α
      K : Set α
      hK : IsCompact K
      ⊢ LE.le (μ K) (iInf fun U => iInf fun x => iInf fun x => μ U)
    -/
  · simp only [le_iInf_iff]
    /-
      case a
      α : Type u_1
      inst✝⁵ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : μ.InnerRegularCompactLTTop
      inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝¹ : R1Space α
      inst✝ : BorelSpace α
      K : Set α
      hK : IsCompact K
      ⊢ ∀ (i : Set α), HasSubset.Subset K i → IsOpen i → LE.le (μ K) (μ i)
    -/
    exact fun U KU _ ↦ measure_mono KU
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝⁵ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : μ.InnerRegularCompactLTTop
      inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝¹ : R1Space α
      inst✝ : BorelSpace α
      K : Set α
      hK : IsCompact K
      ⊢ LE.le (iInf fun U => iInf fun x => iInf fun x => μ U) (μ K)
    -/
  · apply le_of_forall_lt'
    /-
      case a.H
      α : Type u_1
      inst✝⁵ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : μ.InnerRegularCompactLTTop
      inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝¹ : R1Space α
      inst✝ : BorelSpace α
      K : Set α
      hK : IsCompact K
      ⊢ ∀ (c : ENNReal), LT.lt (μ K) c → LT.lt (iInf fun U => iInf fun x => iInf fun …
    -/
    simpa only [iInf_lt_iff, exists_prop, exists_and_left] using hK.exists_isOpen_lt_of_lt
    /-
      🎉 no goals
    -/


protected theorem _root_.IsCompact.exists_isOpen_lt_add [InnerRegularCompactLTTop μ]
    [IsLocallyFiniteMeasure μ] [R1Space α] [BorelSpace α]
    {K : Set α} (hK : IsCompact K) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ U, K ⊆ U ∧ IsOpen U ∧ μ U < μ K + ε :=
  hK.exists_isOpen_lt_of_lt _ (ENNReal.lt_add_right hK.measure_lt_top.ne hε)


/-- Let `μ` be a locally finite measure on an R₁ topological space with Borel σ-algebra.
If `μ` is inner regular for finite measure sets with respect to compact sets,
then any measurable set of finite measure can be approximated in measure by an open set.
See also `Set.exists_isOpen_lt_of_lt` and `MeasurableSet.exists_isOpen_diff_lt`
for the case of an outer regular measure. -/
protected theorem _root_.MeasurableSet.exists_isOpen_symmDiff_lt [InnerRegularCompactLTTop μ]
    [IsLocallyFiniteMeasure μ] [R1Space α] [BorelSpace α]
    {s : Set α} (hs : MeasurableSet s) (hμs : μ s ≠ ∞) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ U, IsOpen U ∧ μ U < ∞ ∧ μ (U ∆ s) < ε := by
  /-
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : R1Space α
    inst✝ : BorelSpace α
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun U => And (IsOpen U) (And (LT.lt (μ U) Top.top) (LT.lt (μ (symmDif …
  -/
  have : ε / 2 ≠ 0 := (ENNReal.half_pos hε).ne'
  /-
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : R1Space α
    inst✝ : BorelSpace α
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    ε : ENNReal
    hε : Ne ε 0
    this : Ne (HDiv.hDiv ε 2) 0
    ⊢ Exists fun U => And (IsOpen U) (And (LT.lt (μ U) Top.top) (LT.lt (μ (symmDif …
  -/
  rcases hs.exists_isCompact_isClosed_diff_lt hμs this with ⟨K, hKs, hKco, hKcl, hμK⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : R1Space α
    inst✝ : BorelSpace α
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    ε : ENNReal
    hε : Ne ε 0
    this : Ne (HDiv.hDiv ε 2) 0
    K : Set α
    hKs : HasSubset.Subset K s
    hKco : IsCompact K
    hKcl : IsClosed K
    hμK : LT.lt (μ (SDiff.sdiff s K)) (HDiv.hDiv ε 2)
    ⊢ Exists fun U => And (IsOpen U) (And (LT.lt (μ U) Top.top) (LT.lt (μ (symmDif …
  -/
  rcases hKco.exists_isOpen_lt_add (μ := μ) this with ⟨U, hKU, hUo, hμU⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : R1Space α
    inst✝ : BorelSpace α
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    ε : ENNReal
    hε : Ne ε 0
    this : Ne (HDiv.hDiv ε 2) 0
    K : Set α
    hKs : HasSubset.Subset K s
    hKco : IsCompact K
    hKcl : IsClosed K
    hμK : LT.lt (μ (SDiff.sdiff s K)) (HDiv.hDiv ε 2)
    U : Set α
    hKU : HasSubset.Subset K U
    hUo : IsOpen U
    hμU : LT.lt (μ U) (HAdd.hAdd (μ K) (HDiv.hDiv ε 2))
    ⊢ Exists fun U => And (IsOpen U) (And (LT.lt (μ U) Top.top) (LT.lt (μ (symmDif …
  -/
  refine ⟨U, hUo, hμU.trans_le le_top, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : R1Space α
    inst✝ : BorelSpace α
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    ε : ENNReal
    hε : Ne ε 0
    this : Ne (HDiv.hDiv ε 2) 0
    K : Set α
    hKs : HasSubset.Subset K s
    hKco : IsCompact K
    hKcl : IsClosed K
    hμK : LT.lt (μ (SDiff.sdiff s K)) (HDiv.hDiv ε 2)
    U : Set α
    hKU : HasSubset.Subset K U
    hUo : IsOpen U
    hμU : LT.lt (μ U) (HAdd.hAdd (μ K) (HDiv.hDiv ε 2))
    ⊢ LT.lt (μ (symmDiff U s)) ε
  -/
  rw [← ENNReal.add_halves ε, measure_symmDiff_eq hUo.nullMeasurableSet hs.nullMeasurableSet]
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : R1Space α
    inst✝ : BorelSpace α
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    ε : ENNReal
    hε : Ne ε 0
    this : Ne (HDiv.hDiv ε 2) 0
    K : Set α
    hKs : HasSubset.Subset K s
    hKco : IsCompact K
    hKcl : IsClosed K
    hμK : LT.lt (μ (SDiff.sdiff s K)) (HDiv.hDiv ε 2)
    U : Set α
    hKU : HasSubset.Subset K U
    hUo : IsOpen U
    hμU : LT.lt (μ U) (HAdd.hAdd (μ K) (HDiv.hDiv ε 2))
    ⊢ LT.lt (HAdd.hAdd (μ (SDiff.sdiff U s)) (μ (SDiff.sdiff s U))) (HAdd.hAdd (HD …
  -/
  gcongr
  · calc
      μ (U \ s) ≤ μ (U \ K) := by gcongr
      _ < ε / 2 := by
        apply measure_diff_lt_of_lt_add hKcl.nullMeasurableSet hKU _ hμU
        exact ne_top_of_le_ne_top hμs (by gcongr)
    /-
      case intro.intro.intro.intro.intro.intro.intro.bd
      α : Type u_1
      inst✝⁵ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : μ.InnerRegularCompactLTTop
      inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝¹ : R1Space α
      inst✝ : BorelSpace α
      s : Set α
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ε : ENNReal
      hε : Ne ε 0
      this : Ne (HDiv.hDiv ε 2) 0
      K : Set α
      hKs : HasSubset.Subset K s
      hKco : IsCompact K
      hKcl : IsClosed K
      hμK : LT.lt (μ (SDiff.sdiff s K)) (HDiv.hDiv ε 2)
      U : Set α
      hKU : HasSubset.Subset K U
      hUo : IsOpen U
      hμU : LT.lt (μ U) (HAdd.hAdd (μ K) (HDiv.hDiv ε 2))
      ⊢ LT.lt (μ (SDiff.sdiff s U)) (HDiv.hDiv ε 2)
    -/
  · exact lt_of_le_of_lt (by gcongr) hμK
    /-
      🎉 no goals
    -/


/-- Let `μ` be a locally finite measure on an R₁ topological space with Borel σ-algebra.
If `μ` is inner regular for finite measure sets with respect to compact sets,
then any null measurable set of finite measure can be approximated in measure by an open set.
See also `Set.exists_isOpen_lt_of_lt` and `MeasurableSet.exists_isOpen_diff_lt`
for the case of an outer regular measure. -/
protected theorem _root_.MeasureTheory.NullMeasurableSet.exists_isOpen_symmDiff_lt
    [InnerRegularCompactLTTop μ] [IsLocallyFiniteMeasure μ] [R1Space α] [BorelSpace α]
    {s : Set α} (hs : NullMeasurableSet s μ) (hμs : μ s ≠ ∞) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ U, IsOpen U ∧ μ U < ∞ ∧ μ (U ∆ s) < ε := by
  /-
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : R1Space α
    inst✝ : BorelSpace α
    s : Set α
    hs : MeasureTheory.NullMeasurableSet s μ
    hμs : Ne (μ s) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun U => And (IsOpen U) (And (LT.lt (μ U) Top.top) (LT.lt (μ (symmDif …
  -/
  rcases hs with ⟨t, htm, hst⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : R1Space α
    inst✝ : BorelSpace α
    s : Set α
    hμs : Ne (μ s) Top.top
    ε : ENNReal
    hε : Ne ε 0
    t : Set α
    htm : MeasurableSet t
    hst : (MeasureTheory.ae μ).EventuallyEq s t
    ⊢ Exists fun U => And (IsOpen U) (And (LT.lt (μ U) Top.top) (LT.lt (μ (symmDif …
  -/
  rcases htm.exists_isOpen_symmDiff_lt (by rwa [← measure_congr hst]) hε with ⟨U, hUo, hμU, hUs⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : R1Space α
    inst✝ : BorelSpace α
    s : Set α
    hμs : Ne (μ s) Top.top
    ε : ENNReal
    hε : Ne ε 0
    t : Set α
    htm : MeasurableSet t
    hst : (MeasureTheory.ae μ).EventuallyEq s t
    U : Set α
    hUo : IsOpen U
    hμU : LT.lt (μ U) Top.top
    hUs : LT.lt (μ (symmDiff U t)) ε
    ⊢ Exists fun U => And (IsOpen U) (And (LT.lt (μ U) Top.top) (LT.lt (μ (symmDif …
  -/
  refine ⟨U, hUo, hμU, ?_⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : R1Space α
    inst✝ : BorelSpace α
    s : Set α
    hμs : Ne (μ s) Top.top
    ε : ENNReal
    hε : Ne ε 0
    t : Set α
    htm : MeasurableSet t
    hst : (MeasureTheory.ae μ).EventuallyEq s t
    U : Set α
    hUo : IsOpen U
    hμU : LT.lt (μ U) Top.top
    hUs : LT.lt (μ (symmDiff U t)) ε
    ⊢ LT.lt (μ (symmDiff U s)) ε
  -/
  rwa [measure_congr <| (ae_eq_refl _).symmDiff hst]
  /-
    🎉 no goals
  -/


instance smul [h : InnerRegularCompactLTTop μ] (c : ℝ≥0∞) : InnerRegularCompactLTTop (c • μ) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : TopologicalSpace α
    h : μ.InnerRegularCompactLTTop
    c : ENNReal
    ⊢ (HSMul.hSMul c μ).InnerRegularCompactLTTop
  -/
  by_cases hc : c = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : TopologicalSpace α
      h : μ.InnerRegularCompactLTTop
      c : ENNReal
      hc : Eq c 0
      ⊢ (HSMul.hSMul c μ).InnerRegularCompactLTTop
    -/
  · simp only [hc, zero_smul]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : TopologicalSpace α
      h : μ.InnerRegularCompactLTTop
      c : ENNReal
      hc : Eq c 0
      ⊢ MeasureTheory.Measure.InnerRegularCompactLTTop 0
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : TopologicalSpace α
    h : μ.InnerRegularCompactLTTop
    c : ENNReal
    hc : Not (Eq c 0)
    ⊢ (HSMul.hSMul c μ).InnerRegularCompactLTTop
  -/
  by_cases h'c : c = ∞
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : TopologicalSpace α
      h : μ.InnerRegularCompactLTTop
      c : ENNReal
      hc : Not (Eq c 0)
      h'c : Eq c Top.top
      ⊢ (HSMul.hSMul c μ).InnerRegularCompactLTTop
    -/
  · constructor
    /-
      case pos.innerRegular
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : TopologicalSpace α
      h : μ.InnerRegularCompactLTTop
      c : ENNReal
      hc : Not (Eq c 0)
      h'c : Eq c Top.top
      ⊢ (HSMul.hSMul c μ).InnerRegularWRT IsCompact fun s => And (MeasurableSet s) ( …
    -/
    intro s hs r hr
    /-
      case pos.innerRegular
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : TopologicalSpace α
      h : μ.InnerRegularCompactLTTop
      c : ENNReal
      hc : Not (Eq c 0)
      h'c : Eq c Top.top
      s : Set α
      hs : And (MeasurableSet s) (Ne ((HSMul.hSMul c μ) s) Top.top)
      r : ENNReal
      hr : LT.lt r ((HSMul.hSMul c μ) s)
      ⊢ Exists fun K => And (HasSubset.Subset K s) (And (IsCompact K) (LT.lt r ((HSM …
    -/
    simp only [h'c, smul_toOuterMeasure, OuterMeasure.coe_smul, Pi.smul_apply, smul_eq_mul] at hr
    /-
      case pos.innerRegular
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : TopologicalSpace α
      h : μ.InnerRegularCompactLTTop
      c : ENNReal
      hc : Not (Eq c 0)
      h'c : Eq c Top.top
      s : Set α
      hs : And (MeasurableSet s) (Ne ((HSMul.hSMul c μ) s) Top.top)
      r : ENNReal
      hr : LT.lt r ((HSMul.hSMul Top.top μ) s)
      ⊢ Exists fun K => And (HasSubset.Subset K s) (And (IsCompact K) (LT.lt r ((HSM …
    -/
    by_cases h's : μ s = 0
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝¹ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : TopologicalSpace α
        h : μ.InnerRegularCompactLTTop
        c : ENNReal
        hc : Not (Eq c 0)
        h'c : Eq c Top.top
        s : Set α
        hs : And (MeasurableSet s) (Ne ((HSMul.hSMul c μ) s) Top.top)
        r : ENNReal
        hr : LT.lt r ((HSMul.hSMul Top.top μ) s)
        h's : Eq (μ s) 0
        ⊢ Exists fun K => And (HasSubset.Subset K s) (And (IsCompact K) (LT.lt r ((HSM …
      -/
    · simp [h's] at hr
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝¹ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : TopologicalSpace α
        h : μ.InnerRegularCompactLTTop
        c : ENNReal
        hc : Not (Eq c 0)
        h'c : Eq c Top.top
        s : Set α
        hs : And (MeasurableSet s) (Ne ((HSMul.hSMul c μ) s) Top.top)
        r : ENNReal
        hr : LT.lt r ((HSMul.hSMul Top.top μ) s)
        h's : Not (Eq (μ s) 0)
        ⊢ Exists fun K => And (HasSubset.Subset K s) (And (IsCompact K) (LT.lt r ((HSM …
      -/
    · simp [h'c, ENNReal.mul_eq_top, h's] at hs
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : TopologicalSpace α
      h : μ.InnerRegularCompactLTTop
      c : ENNReal
      hc : Not (Eq c 0)
      h'c : Not (Eq c Top.top)
      ⊢ (HSMul.hSMul c μ).InnerRegularCompactLTTop
    -/
  · constructor
    /-
      case neg.innerRegular
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : TopologicalSpace α
      h : μ.InnerRegularCompactLTTop
      c : ENNReal
      hc : Not (Eq c 0)
      h'c : Not (Eq c Top.top)
      ⊢ (HSMul.hSMul c μ).InnerRegularWRT IsCompact fun s => And (MeasurableSet s) ( …
    -/
    convert InnerRegularWRT.smul h.innerRegular c using 2 with s
    /-
      case h.e'_5.h.a
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : TopologicalSpace α
      h : μ.InnerRegularCompactLTTop
      c : ENNReal
      hc : Not (Eq c 0)
      h'c : Not (Eq c Top.top)
      s : Set α
      ⊢ Iff (And (MeasurableSet s) (Ne ((HSMul.hSMul c μ) s) Top.top)) (And (Measura …
    -/
    have : (c • μ) s ≠ ∞ ↔ μ s ≠ ∞ := by simp [not_iff_not, ENNReal.mul_eq_top, hc, h'c]
    /-
      case h.e'_5.h.a
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : TopologicalSpace α
      h : μ.InnerRegularCompactLTTop
      c : ENNReal
      hc : Not (Eq c 0)
      h'c : Not (Eq c Top.top)
      s : Set α
      this : Iff (Ne ((HSMul.hSMul c μ) s) Top.top) (Ne (μ s) Top.top)
      ⊢ Iff (And (MeasurableSet s) (Ne ((HSMul.hSMul c μ) s) Top.top)) (And (Measura …
    -/
    simp only [this]
    /-
      🎉 no goals
    -/


instance smul_nnreal [InnerRegularCompactLTTop μ] (c : ℝ≥0) :
    InnerRegularCompactLTTop (c • μ) :=
  inferInstanceAs (InnerRegularCompactLTTop ((c : ℝ≥0∞) • μ))


instance (priority := 80) [InnerRegularCompactLTTop μ] [SigmaFinite μ] : InnerRegular μ :=
  ⟨InnerRegularCompactLTTop.innerRegular.trans InnerRegularWRT.of_sigmaFinite⟩


protected theorem map_of_continuous [BorelSpace α] [MeasurableSpace β] [TopologicalSpace β]
    [BorelSpace β] [h : InnerRegularCompactLTTop μ] {f : α → β} (hf : Continuous f) :
    InnerRegularCompactLTTop (Measure.map f μ) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : TopologicalSpace β
    inst✝ : BorelSpace β
    h : μ.InnerRegularCompactLTTop
    f : α → β
    hf : Continuous f
    ⊢ (MeasureTheory.Measure.map f μ).InnerRegularCompactLTTop
  -/
  constructor
  /-
    case innerRegular
    α : Type u_1
    β : Type u_2
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : TopologicalSpace β
    inst✝ : BorelSpace β
    h : μ.InnerRegularCompactLTTop
    f : α → β
    hf : Continuous f
    ⊢ (MeasureTheory.Measure.map f μ).InnerRegularWRT IsCompact fun s => And (Meas …
  -/
  refine InnerRegularWRT.map h.innerRegular hf.aemeasurable ?_ (fun K hK ↦ hK.image hf) ?_
    /-
      case innerRegular.refine_1
      α : Type u_1
      β : Type u_2
      inst✝⁵ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : BorelSpace α
      inst✝² : MeasurableSpace β
      inst✝¹ : TopologicalSpace β
      inst✝ : BorelSpace β
      h : μ.InnerRegularCompactLTTop
      f : α → β
      hf : Continuous f
      ⊢ ∀ (U : Set β), And (MeasurableSet U) (Ne ((MeasureTheory.Measure.map f μ) U) …
    -/
  · rintro s ⟨hs, h's⟩
    /-
      case innerRegular.refine_1.intro
      α : Type u_1
      β : Type u_2
      inst✝⁵ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : BorelSpace α
      inst✝² : MeasurableSpace β
      inst✝¹ : TopologicalSpace β
      inst✝ : BorelSpace β
      h : μ.InnerRegularCompactLTTop
      f : α → β
      hf : Continuous f
      s : Set β
      hs : MeasurableSet s
      h's : Ne ((MeasureTheory.Measure.map f μ) s) Top.top
      ⊢ And (MeasurableSet (Set.preimage f s)) (Ne (μ (Set.preimage f s)) Top.top)
    -/
    exact ⟨hf.measurable hs, by rwa [map_apply hf.measurable hs] at h's⟩
    /-
      🎉 no goals
    -/
    /-
      case innerRegular.refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁵ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : BorelSpace α
      inst✝² : MeasurableSpace β
      inst✝¹ : TopologicalSpace β
      inst✝ : BorelSpace β
      h : μ.InnerRegularCompactLTTop
      f : α → β
      hf : Continuous f
      ⊢ ∀ (U : Set β), And (MeasurableSet U) (Ne ((MeasureTheory.Measure.map f μ) U) …
    -/
  · rintro s ⟨hs, -⟩
    /-
      case innerRegular.refine_2.intro
      α : Type u_1
      β : Type u_2
      inst✝⁵ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : BorelSpace α
      inst✝² : MeasurableSpace β
      inst✝¹ : TopologicalSpace β
      inst✝ : BorelSpace β
      h : μ.InnerRegularCompactLTTop
      f : α → β
      hf : Continuous f
      s : Set β
      hs : MeasurableSet s
      ⊢ MeasurableSet s
    -/
    exact hs
    /-
      🎉 no goals
    -/


instance zero : WeaklyRegular (0 : Measure α) :=
  ⟨fun _ _ _r hr => ⟨∅, empty_subset _, isClosed_empty, hr⟩⟩


/-- If `μ` is a weakly regular measure, then any open set can be approximated by a closed subset. -/
theorem _root_.IsOpen.exists_lt_isClosed [WeaklyRegular μ] ⦃U : Set α⦄ (hU : IsOpen U) {r : ℝ≥0∞}
    (hr : r < μ U) : ∃ F, F ⊆ U ∧ IsClosed F ∧ r < μ F :=
  WeaklyRegular.innerRegular hU r hr


/-- If `μ` is a weakly regular measure, then any open set can be approximated by a closed subset. -/
theorem _root_.IsOpen.measure_eq_iSup_isClosed ⦃U : Set α⦄ (hU : IsOpen U) (μ : Measure α)
    [WeaklyRegular μ] : μ U = ⨆ (F) (_ : F ⊆ U) (_ : IsClosed F), μ F :=
  WeaklyRegular.innerRegular.measure_eq_iSup hU


theorem innerRegular_measurable [WeaklyRegular μ] :
    InnerRegularWRT μ IsClosed fun s => MeasurableSet s ∧ μ s ≠ ∞ :=
  WeaklyRegular.innerRegular.measurableSet_of_isOpen (fun _ _ h₁ h₂ ↦ h₁.inter h₂.isClosed_compl)


/-- If `s` is a measurable set, a weakly regular measure `μ` is finite on `s`, and `ε` is a positive
number, then there exist a closed set `K ⊆ s` such that `μ s < μ K + ε`. -/
theorem _root_.MeasurableSet.exists_isClosed_lt_add [WeaklyRegular μ] {s : Set α}
    (hs : MeasurableSet s) (hμs : μ s ≠ ∞) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ K, K ⊆ s ∧ IsClosed K ∧ μ s < μ K + ε :=
  innerRegular_measurable.exists_subset_lt_add isClosed_empty ⟨hs, hμs⟩ hμs hε


theorem _root_.MeasurableSet.exists_isClosed_diff_lt [OpensMeasurableSpace α] [WeaklyRegular μ]
    ⦃A : Set α⦄ (hA : MeasurableSet A) (h'A : μ A ≠ ∞) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ F, F ⊆ A ∧ IsClosed F ∧ μ (A \ F) < ε := by
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : μ.WeaklyRegular
    A : Set α
    hA : MeasurableSet A
    h'A : Ne (μ A) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun F => And (HasSubset.Subset F A) (And (IsClosed F) (LT.lt (μ (SDif …
  -/
  rcases hA.exists_isClosed_lt_add h'A hε with ⟨F, hFA, hFc, hF⟩
  exact ⟨F, hFA, hFc, measure_diff_lt_of_lt_add hFc.nullMeasurableSet hFA
    (ne_top_of_le_ne_top h'A <| measure_mono hFA) hF⟩


/-- Given a weakly regular measure, any measurable set of finite mass can be approximated from
inside by closed sets. -/
theorem _root_.MeasurableSet.exists_lt_isClosed_of_ne_top [WeaklyRegular μ] ⦃A : Set α⦄
    (hA : MeasurableSet A) (h'A : μ A ≠ ∞) {r : ℝ≥0∞} (hr : r < μ A) :
    ∃ K, K ⊆ A ∧ IsClosed K ∧ r < μ K :=
  innerRegular_measurable ⟨hA, h'A⟩ _ hr


/-- Given a weakly regular measure, any measurable set of finite mass can be approximated from
inside by closed sets. -/
theorem _root_.MeasurableSet.measure_eq_iSup_isClosed_of_ne_top [WeaklyRegular μ] ⦃A : Set α⦄
    (hA : MeasurableSet A) (h'A : μ A ≠ ∞) : μ A = ⨆ (K) (_ : K ⊆ A) (_ : IsClosed K), μ K :=
  innerRegular_measurable.measure_eq_iSup ⟨hA, h'A⟩


/-- The restriction of a weakly regular measure to a measurable set of finite measure is
weakly regular. -/
theorem restrict_of_measure_ne_top [BorelSpace α] [WeaklyRegular μ] {A : Set α}
    (h'A : μ A ≠ ∞) : WeaklyRegular (μ.restrict A) := by
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace α
    inst✝¹ : BorelSpace α
    inst✝ : μ.WeaklyRegular
    A : Set α
    h'A : Ne (μ A) Top.top
    ⊢ (μ.restrict A).WeaklyRegular
  -/
  haveI : Fact (μ A < ∞) := ⟨h'A.lt_top⟩
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace α
    inst✝¹ : BorelSpace α
    inst✝ : μ.WeaklyRegular
    A : Set α
    h'A : Ne (μ A) Top.top
    this : Fact (LT.lt (μ A) Top.top)
    ⊢ (μ.restrict A).WeaklyRegular
  -/
  refine InnerRegularWRT.weaklyRegular_of_finite (μ.restrict A) (fun V V_open r hr ↦ ?_)
  have : InnerRegularWRT (μ.restrict A) IsClosed (fun s ↦ MeasurableSet s) :=
    InnerRegularWRT.restrict_of_measure_ne_top innerRegular_measurable h'A
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace α
    inst✝¹ : BorelSpace α
    inst✝ : μ.WeaklyRegular
    A : Set α
    h'A : Ne (μ A) Top.top
    this✝ : Fact (LT.lt (μ A) Top.top)
    V : Set α
    V_open : IsOpen V
    r : ENNReal
    hr : LT.lt r ((μ.restrict A) V)
    this : (μ.restrict A).InnerRegularWRT IsClosed fun s => MeasurableSet s
    ⊢ Exists fun K => And (HasSubset.Subset K V) (And (IsClosed K) (LT.lt r ((μ.re …
  -/
  exact this V_open.measurableSet r hr
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

/-- Any finite measure on a metrizable space (or even a pseudo metrizable space)
is weakly regular. -/
instance (priority := 100) of_pseudoMetrizableSpace_of_isFiniteMeasure {X : Type*}
    [TopologicalSpace X] [PseudoMetrizableSpace X] [MeasurableSpace X] [BorelSpace X]
    (μ : Measure X) [IsFiniteMeasure μ] :
    WeaklyRegular μ :=
  (InnerRegularWRT.of_pseudoMetrizableSpace μ).weaklyRegular_of_finite μ

-- see Note [lower instance priority]

/-- Any locally finite measure on a second countable metrizable space
(or even a pseudo metrizable space) is weakly regular. -/
instance (priority := 100) of_pseudoMetrizableSpace_secondCountable_of_locallyFinite {X : Type*}
    [TopologicalSpace X] [PseudoMetrizableSpace X] [SecondCountableTopology X] [MeasurableSpace X]
    [BorelSpace X] (μ : Measure X) [IsLocallyFiniteMeasure μ] : WeaklyRegular μ :=
  have : OuterRegular μ := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁷ : MeasurableSpace α
      μ✝ : MeasureTheory.Measure α
      inst✝⁶ : TopologicalSpace α
      X : Type u_3
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace.PseudoMetrizableSpace X
      inst✝³ : SecondCountableTopology X
      inst✝² : MeasurableSpace X
      inst✝¹ : BorelSpace X
      μ : MeasureTheory.Measure X
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      ⊢ μ.OuterRegular
    -/
    refine (μ.finiteSpanningSetsInOpen'.mono' fun U hU => ?_).outerRegular
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁷ : MeasurableSpace α
      μ✝ : MeasureTheory.Measure α
      inst✝⁶ : TopologicalSpace α
      X : Type u_3
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace.PseudoMetrizableSpace X
      inst✝³ : SecondCountableTopology X
      inst✝² : MeasurableSpace X
      inst✝¹ : BorelSpace X
      μ : MeasureTheory.Measure X
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      U : Set X
      hU : Membership.mem (Inter.inter (setOf fun K => IsOpen K) (setOf fun s => LT. …
      ⊢ Membership.mem (setOf fun U => And (IsOpen U) (μ.restrict U).OuterRegular) U
    -/
    have : Fact (μ U < ∞) := ⟨hU.2⟩
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁷ : MeasurableSpace α
      μ✝ : MeasureTheory.Measure α
      inst✝⁶ : TopologicalSpace α
      X : Type u_3
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace.PseudoMetrizableSpace X
      inst✝³ : SecondCountableTopology X
      inst✝² : MeasurableSpace X
      inst✝¹ : BorelSpace X
      μ : MeasureTheory.Measure X
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      U : Set X
      hU : Membership.mem (Inter.inter (setOf fun K => IsOpen K) (setOf fun s => LT. …
      this : Fact (LT.lt (μ U) Top.top)
      ⊢ Membership.mem (setOf fun U => And (IsOpen U) (μ.restrict U).OuterRegular) U
    -/
    exact ⟨hU.1, inferInstance⟩
    /-
      🎉 no goals
    -/
  ⟨InnerRegularWRT.of_pseudoMetrizableSpace μ⟩


protected theorem smul [WeaklyRegular μ] {x : ℝ≥0∞} (hx : x ≠ ∞) : (x • μ).WeaklyRegular := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace α
    inst✝ : μ.WeaklyRegular
    x : ENNReal
    hx : Ne x Top.top
    ⊢ (HSMul.hSMul x μ).WeaklyRegular
  -/
  haveI := OuterRegular.smul μ hx
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace α
    inst✝ : μ.WeaklyRegular
    x : ENNReal
    hx : Ne x Top.top
    this : (HSMul.hSMul x μ).OuterRegular
    ⊢ (HSMul.hSMul x μ).WeaklyRegular
  -/
  exact ⟨WeaklyRegular.innerRegular.smul x⟩
  /-
    🎉 no goals
  -/


instance smul_nnreal [WeaklyRegular μ] (c : ℝ≥0) : WeaklyRegular (c • μ) :=
  WeaklyRegular.smul coe_ne_top


instance zero : Regular (0 : Measure α) :=
  ⟨fun _ _ _r hr => ⟨∅, empty_subset _, isCompact_empty, hr⟩⟩


/-- If `μ` is a regular measure, then any open set can be approximated by a compact subset. -/
theorem _root_.IsOpen.exists_lt_isCompact [Regular μ] ⦃U : Set α⦄ (hU : IsOpen U) {r : ℝ≥0∞}
    (hr : r < μ U) : ∃ K, K ⊆ U ∧ IsCompact K ∧ r < μ K :=
  Regular.innerRegular hU r hr


/-- The measure of an open set is the supremum of the measures of compact sets it contains. -/
theorem _root_.IsOpen.measure_eq_iSup_isCompact ⦃U : Set α⦄ (hU : IsOpen U) (μ : Measure α)
    [Regular μ] : μ U = ⨆ (K : Set α) (_ : K ⊆ U) (_ : IsCompact K), μ K :=
  Regular.innerRegular.measure_eq_iSup hU


theorem exists_isCompact_not_null [Regular μ] : (∃ K, IsCompact K ∧ μ K ≠ 0) ↔ μ ≠ 0 := by
  simp_rw [Ne, ← measure_univ_eq_zero, isOpen_univ.measure_eq_iSup_isCompact,
    ENNReal.iSup_eq_zero, not_forall, exists_prop, subset_univ, true_and]

/-- If `μ` is a regular measure, then any measurable set of finite measure can be approximated by a
compact subset. See also `MeasurableSet.exists_isCompact_lt_add` and
`MeasurableSet.exists_lt_isCompact_of_ne_top`. -/
instance (priority := 100) [Regular μ] : InnerRegularCompactLTTop μ :=
  ⟨Regular.innerRegular.measurableSet_of_isOpen (fun _ _ hs hU ↦ hs.diff hU)⟩


protected theorem map [BorelSpace α] [MeasurableSpace β] [TopologicalSpace β]
    [BorelSpace β] [Regular μ] (f : α ≃ₜ β) : (Measure.map f μ).Regular := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : BorelSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : TopologicalSpace β
    inst✝¹ : BorelSpace β
    inst✝ : μ.Regular
    f : Homeomorph α β
    ⊢ (MeasureTheory.Measure.map (⇑f) μ).Regular
  -/
  haveI := OuterRegular.map f μ
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : BorelSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : TopologicalSpace β
    inst✝¹ : BorelSpace β
    inst✝ : μ.Regular
    f : Homeomorph α β
    this : (MeasureTheory.Measure.map (⇑f) μ).OuterRegular
    ⊢ (MeasureTheory.Measure.map (⇑f) μ).Regular
  -/
  haveI := IsFiniteMeasureOnCompacts.map μ f
  exact
    ⟨Regular.innerRegular.map' f.toMeasurableEquiv
        (fun U hU => hU.preimage f.continuous)
        (fun K hK => hK.image f.continuous)⟩


protected theorem map_iff [BorelSpace α] [MeasurableSpace β] [TopologicalSpace β]
    [BorelSpace β] (f : α ≃ₜ β) :
    Regular (Measure.map f μ) ↔ Regular μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : TopologicalSpace β
    inst✝ : BorelSpace β
    f : Homeomorph α β
    ⊢ Iff (MeasureTheory.Measure.map (⇑f) μ).Regular μ.Regular
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ h.map f⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : TopologicalSpace β
    inst✝ : BorelSpace β
    f : Homeomorph α β
    h : (MeasureTheory.Measure.map (⇑f) μ).Regular
    ⊢ μ.Regular
  -/
  convert h.map f.symm
  /-
    case h.e'_4
    α : Type u_1
    β : Type u_2
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : TopologicalSpace β
    inst✝ : BorelSpace β
    f : Homeomorph α β
    h : (MeasureTheory.Measure.map (⇑f) μ).Regular
    ⊢ Eq μ (MeasureTheory.Measure.map (⇑f.symm) (MeasureTheory.Measure.map (⇑f) μ))
  -/
  rw [map_map f.symm.continuous.measurable f.continuous.measurable]
  /-
    case h.e'_4
    α : Type u_1
    β : Type u_2
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : TopologicalSpace β
    inst✝ : BorelSpace β
    f : Homeomorph α β
    h : (MeasureTheory.Measure.map (⇑f) μ).Regular
    ⊢ Eq μ (MeasureTheory.Measure.map (Function.comp ⇑f.symm ⇑f) μ)
  -/
  simp
  /-
    🎉 no goals
  -/


protected theorem smul [Regular μ] {x : ℝ≥0∞} (hx : x ≠ ∞) : (x • μ).Regular := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace α
    inst✝ : μ.Regular
    x : ENNReal
    hx : Ne x Top.top
    ⊢ (HSMul.hSMul x μ).Regular
  -/
  haveI := OuterRegular.smul μ hx
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace α
    inst✝ : μ.Regular
    x : ENNReal
    hx : Ne x Top.top
    this : (HSMul.hSMul x μ).OuterRegular
    ⊢ (HSMul.hSMul x μ).Regular
  -/
  haveI := IsFiniteMeasureOnCompacts.smul μ hx
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace α
    inst✝ : μ.Regular
    x : ENNReal
    hx : Ne x Top.top
    this✝ : (HSMul.hSMul x μ).OuterRegular
    this : MeasureTheory.IsFiniteMeasureOnCompacts (HSMul.hSMul x μ)
    ⊢ (HSMul.hSMul x μ).Regular
  -/
  exact ⟨Regular.innerRegular.smul x⟩
  /-
    🎉 no goals
  -/


instance smul_nnreal [Regular μ] (c : ℝ≥0) : Regular (c • μ) := Regular.smul coe_ne_top


/-- The restriction of a regular measure to a set of finite measure is regular. -/
theorem restrict_of_measure_ne_top [R1Space α] [BorelSpace α] [Regular μ]
    {A : Set α} (h'A : μ A ≠ ∞) : Regular (μ.restrict A) := by
  /-
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace α
    inst✝² : R1Space α
    inst✝¹ : BorelSpace α
    inst✝ : μ.Regular
    A : Set α
    h'A : Ne (μ A) Top.top
    ⊢ (μ.restrict A).Regular
  -/
  have : WeaklyRegular (μ.restrict A) := WeaklyRegular.restrict_of_measure_ne_top h'A
  /-
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace α
    inst✝² : R1Space α
    inst✝¹ : BorelSpace α
    inst✝ : μ.Regular
    A : Set α
    h'A : Ne (μ A) Top.top
    this : (μ.restrict A).WeaklyRegular
    ⊢ (μ.restrict A).Regular
  -/
  constructor
  /-
    case innerRegular
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace α
    inst✝² : R1Space α
    inst✝¹ : BorelSpace α
    inst✝ : μ.Regular
    A : Set α
    h'A : Ne (μ A) Top.top
    this : (μ.restrict A).WeaklyRegular
    ⊢ (μ.restrict A).InnerRegularWRT IsCompact IsOpen
  -/
  intro V hV r hr
  have R : restrict μ A V ≠ ∞ := by
    rw [restrict_apply hV.measurableSet]
    exact ((measure_mono inter_subset_right).trans_lt h'A.lt_top).ne
  /-
    case innerRegular
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace α
    inst✝² : R1Space α
    inst✝¹ : BorelSpace α
    inst✝ : μ.Regular
    A : Set α
    h'A : Ne (μ A) Top.top
    this : (μ.restrict A).WeaklyRegular
    V : Set α
    hV : IsOpen V
    r : ENNReal
    hr : LT.lt r ((μ.restrict A) V)
    R : Ne ((μ.restrict A) V) Top.top
    ⊢ Exists fun K => And (HasSubset.Subset K V) (And (IsCompact K) (LT.lt r ((μ.r …
  -/
  exact MeasurableSet.exists_lt_isCompact_of_ne_top hV.measurableSet R hr
  /-
    🎉 no goals
  -/


/-- Any locally finite measure on a `σ`-compact pseudometrizable space is regular. -/
instance (priority := 100) Regular.of_sigmaCompactSpace_of_isLocallyFiniteMeasure {X : Type*}
    [TopologicalSpace X] [PseudoMetrizableSpace X] [SigmaCompactSpace X] [MeasurableSpace X]
    [BorelSpace X] (μ : Measure X) [IsLocallyFiniteMeasure μ] : Regular μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    X : Type u_3
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : TopologicalSpace.PseudoMetrizableSpace X
    inst✝³ : SigmaCompactSpace X
    inst✝² : MeasurableSpace X
    inst✝¹ : BorelSpace X
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ⊢ μ.Regular
  -/
  let A : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetric X
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    X : Type u_3
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : TopologicalSpace.PseudoMetrizableSpace X
    inst✝³ : SigmaCompactSpace X
    inst✝² : MeasurableSpace X
    inst✝¹ : BorelSpace X
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    A : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetric X
    ⊢ μ.Regular
  -/
  exact ⟨(InnerRegularWRT.isCompact_isClosed μ).trans (InnerRegularWRT.of_pseudoMetrizableSpace μ)⟩
  /-
    🎉 no goals
  -/


/-- Any sigma finite measure on a `σ`-compact pseudometrizable space is inner regular. -/
instance (priority := 100) {X : Type*}
    [TopologicalSpace X] [PseudoMetrizableSpace X] [SigmaCompactSpace X] [MeasurableSpace X]
    [BorelSpace X] (μ : Measure X) [SigmaFinite μ] : InnerRegular μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    X : Type u_3
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : TopologicalSpace.PseudoMetrizableSpace X
    inst✝³ : SigmaCompactSpace X
    inst✝² : MeasurableSpace X
    inst✝¹ : BorelSpace X
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.SigmaFinite μ
    ⊢ μ.InnerRegular
  -/
  refine ⟨(InnerRegularWRT.isCompact_isClosed μ).trans ?_⟩
  refine InnerRegularWRT.of_restrict (fun n ↦ ?_) (iUnion_spanningSets μ).superset
    (monotone_spanningSets μ)
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    X : Type u_3
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : TopologicalSpace.PseudoMetrizableSpace X
    inst✝³ : SigmaCompactSpace X
    inst✝² : MeasurableSpace X
    inst✝¹ : BorelSpace X
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.SigmaFinite μ
    n : Nat
    ⊢ (μ.restrict (MeasureTheory.spanningSets μ n)).InnerRegularWRT IsClosed Measu …
  -/
  have : Fact (μ (spanningSets μ n) < ∞) := ⟨measure_spanningSets_lt_top μ n⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    X : Type u_3
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : TopologicalSpace.PseudoMetrizableSpace X
    inst✝³ : SigmaCompactSpace X
    inst✝² : MeasurableSpace X
    inst✝¹ : BorelSpace X
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.SigmaFinite μ
    n : Nat
    this : Fact (LT.lt (μ (MeasureTheory.spanningSets μ n)) Top.top)
    ⊢ (μ.restrict (MeasureTheory.spanningSets μ n)).InnerRegularWRT IsClosed Measu …
  -/
  exact WeaklyRegular.innerRegular_measurable.trans InnerRegularWRT.of_sigmaFinite
  /-
    🎉 no goals
  -/


