/-- The oscillation of `f : E → F` at `x`. -/
noncomputable def oscillation [TopologicalSpace E] (f : E → F) (x : E) : ENNReal :=
  ⨅ S ∈ (𝓝 x).map f, diam S


/-- The oscillation of `f : E → F` within `D` at `x`. -/
noncomputable def oscillationWithin [TopologicalSpace E] (f : E → F) (D : Set E) (x : E) :
  ENNReal := ⨅ S ∈ (𝓝[D] x).map f, diam S


/-- The oscillation of `f` at `x` within a neighborhood `D` of `x` is equal to `oscillation f x` -/
theorem oscillationWithin_nhd_eq_oscillation [TopologicalSpace E] (f : E → F) (D : Set E) (x : E)
    (hD : D ∈ 𝓝 x) : oscillationWithin f D x = oscillation f x := by
  /-
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : TopologicalSpace E
    f : E → F
    D : Set E
    x : E
    hD : Membership.mem (nhds x) D
    ⊢ Eq (oscillationWithin f D x) (oscillation f x)
  -/
  rw [oscillation, oscillationWithin, nhdsWithin_eq_nhds.2 hD]
  /-
    🎉 no goals
  -/


/-- The oscillation of `f` at `x` within `univ` is equal to `oscillation f x` -/
theorem oscillationWithin_univ_eq_oscillation [TopologicalSpace E] (f : E → F) (x : E) :
    oscillationWithin f univ x = oscillation f x :=
  oscillationWithin_nhd_eq_oscillation f univ x Filter.univ_mem


theorem oscillationWithin_eq_zero [TopologicalSpace E] {f : E → F} {D : Set E}
    {x : E} (hf : ContinuousWithinAt f D x) : oscillationWithin f D x = 0 := by
  /-
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : TopologicalSpace E
    f : E → F
    D : Set E
    x : E
    hf : ContinuousWithinAt f D x
    ⊢ Eq (oscillationWithin f D x) 0
  -/
  refine le_antisymm (_root_.le_of_forall_pos_le_add fun ε hε ↦ ?_) (zero_le _)
  /-
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : TopologicalSpace E
    f : E → F
    D : Set E
    x : E
    hf : ContinuousWithinAt f D x
    ε : ENNReal
    hε : LT.lt 0 ε
    ⊢ LE.le (oscillationWithin f D x) (HAdd.hAdd 0 ε)
  -/
  rw [zero_add]
  /-
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : TopologicalSpace E
    f : E → F
    D : Set E
    x : E
    hf : ContinuousWithinAt f D x
    ε : ENNReal
    hε : LT.lt 0 ε
    ⊢ LE.le (oscillationWithin f D x) ε
  -/
  have : ball (f x) (ε / 2) ∈ (𝓝[D] x).map f := hf <| ball_mem_nhds _ (by simp [ne_of_gt hε])
  /-
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : TopologicalSpace E
    f : E → F
    D : Set E
    x : E
    hf : ContinuousWithinAt f D x
    ε : ENNReal
    hε : LT.lt 0 ε
    this : Membership.mem (Filter.map f (nhdsWithin x D)) (EMetric.ball (f x) (HDi …
    ⊢ LE.le (oscillationWithin f D x) ε
  -/
  refine (biInf_le diam this).trans (le_of_le_of_eq diam_ball ?_)
  /-
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : TopologicalSpace E
    f : E → F
    D : Set E
    x : E
    hf : ContinuousWithinAt f D x
    ε : ENNReal
    hε : LT.lt 0 ε
    this : Membership.mem (Filter.map f (nhdsWithin x D)) (EMetric.ball (f x) (HDi …
    ⊢ Eq (HMul.hMul 2 (HDiv.hDiv ε 2)) ε
  -/
  exact (ENNReal.mul_div_cancel (by norm_num) (by norm_num))
  /-
    🎉 no goals
  -/


theorem oscillation_eq_zero [TopologicalSpace E] {f : E → F} {x : E} (hf : ContinuousAt f x) :
    oscillation f x = 0 := by
  /-
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : TopologicalSpace E
    f : E → F
    x : E
    hf : ContinuousAt f x
    ⊢ Eq (oscillation f x) 0
  -/
  rw [← continuousWithinAt_univ f x] at hf
  /-
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : TopologicalSpace E
    f : E → F
    x : E
    hf : ContinuousWithinAt f Set.univ x
    ⊢ Eq (oscillation f x) 0
  -/
  exact oscillationWithin_univ_eq_oscillation f x ▸ hf.oscillationWithin_eq_zero
  /-
    🎉 no goals
  -/


/-- The oscillation within `D` of `f` at `x ∈ D` is 0 if and only if `ContinuousWithinAt f D x`. -/
theorem eq_zero_iff_continuousWithinAt [TopologicalSpace E] (f : E → F) {D : Set E}
    {x : E} (xD : x ∈ D) : oscillationWithin f D x = 0 ↔ ContinuousWithinAt f D x := by
  /-
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : TopologicalSpace E
    f : E → F
    D : Set E
    x : E
    xD : Membership.mem D x
    ⊢ Iff (Eq (oscillationWithin f D x) 0) (ContinuousWithinAt f D x)
  -/
  refine ⟨fun hf ↦ EMetric.tendsto_nhds.mpr (fun ε ε0 ↦ ?_), fun hf ↦ hf.oscillationWithin_eq_zero⟩
  /-
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : TopologicalSpace E
    f : E → F
    D : Set E
    x : E
    xD : Membership.mem D x
    hf : Eq (oscillationWithin f D x) 0
    ε : ENNReal
    ε0 : GT.gt ε 0
    ⊢ Filter.Eventually (fun x_1 => LT.lt (EDist.edist (f x_1) (f x)) ε) (nhdsWith …
  -/
  simp_rw [← hf, oscillationWithin, iInf_lt_iff] at ε0
  /-
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : TopologicalSpace E
    f : E → F
    D : Set E
    x : E
    xD : Membership.mem D x
    hf : Eq (oscillationWithin f D x) 0
    ε : ENNReal
    ε0 : Exists fun i => Exists fun i_1 => LT.lt (EMetric.diam i) ε
    ⊢ Filter.Eventually (fun x_1 => LT.lt (EDist.edist (f x_1) (f x)) ε) (nhdsWith …
  -/
  obtain ⟨S, hS, Sε⟩ := ε0
  /-
    case intro.intro
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : TopologicalSpace E
    f : E → F
    D : Set E
    x : E
    xD : Membership.mem D x
    hf : Eq (oscillationWithin f D x) 0
    ε : ENNReal
    S : Set F
    hS : Membership.mem (Filter.map f (nhdsWithin x D)) S
    Sε : LT.lt (EMetric.diam S) ε
    ⊢ Filter.Eventually (fun x_1 => LT.lt (EDist.edist (f x_1) (f x)) ε) (nhdsWith …
  -/
  refine Filter.mem_of_superset hS (fun y hy ↦ lt_of_le_of_lt ?_ Sε)
  /-
    case intro.intro
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : TopologicalSpace E
    f : E → F
    D : Set E
    x : E
    xD : Membership.mem D x
    hf : Eq (oscillationWithin f D x) 0
    ε : ENNReal
    S : Set F
    hS : Membership.mem (Filter.map f (nhdsWithin x D)) S
    Sε : LT.lt (EMetric.diam S) ε
    y : E
    hy : Membership.mem (Set.preimage f S) y
    ⊢ LE.le (EDist.edist (f y) (f x)) (EMetric.diam S)
  -/
  exact edist_le_diam_of_mem (mem_preimage.1 hy) <| mem_preimage.1 (mem_of_mem_nhdsWithin xD hS)
  /-
    🎉 no goals
  -/


/-- The oscillation of `f` at `x` is 0 if and only if `f` is continuous at `x`. -/
theorem eq_zero_iff_continuousAt [TopologicalSpace E] (f : E → F) (x : E) :
    oscillation f x = 0 ↔ ContinuousAt f x := by
  /-
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : TopologicalSpace E
    f : E → F
    x : E
    ⊢ Iff (Eq (oscillation f x) 0) (ContinuousAt f x)
  -/
  rw [← oscillationWithin_univ_eq_oscillation, ← continuousWithinAt_univ f x]
  /-
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : TopologicalSpace E
    f : E → F
    x : E
    ⊢ Iff (Eq (oscillationWithin f Set.univ x) 0) (ContinuousWithinAt f Set.univ x)
  -/
  exact OscillationWithin.eq_zero_iff_continuousWithinAt f (mem_univ x)
  /-
    🎉 no goals
  -/


/-- If `oscillationWithin f D x < ε` at every `x` in a compact set `K`, then there exists `δ > 0`
such that the oscillation of `f` on `ball x δ ∩ D` is less than `ε` for every `x` in `K`. -/
theorem uniform_oscillationWithin (comp : IsCompact K) (hK : ∀ x ∈ K, oscillationWithin f D x < ε) :
    ∃ δ > 0, ∀ x ∈ K, diam (f '' (ball x (ENNReal.ofReal δ) ∩ D)) ≤ ε := by
  /-
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : PseudoEMetricSpace E
    K : Set E
    f : E → F
    D : Set E
    ε : ENNReal
    comp : IsCompact K
    hK : ∀ (x : E), Membership.mem K x → LT.lt (oscillationWithin f D x) ε
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (x : E), Membership.mem K x → LE.le (EMet …
  -/
  let S := fun r ↦ { x : E | ∃ (a : ℝ), (a > r ∧ diam (f '' (ball x (ENNReal.ofReal a) ∩ D)) ≤ ε) }
  have S_open : ∀ r > 0, IsOpen (S r) := by
    refine fun r _ ↦ isOpen_iff.mpr fun x ⟨a, ar, ha⟩ ↦
      ⟨ENNReal.ofReal ((a - r) / 2), by simp [ar], ?_⟩
    refine fun y hy ↦ ⟨a - (a - r) / 2, by linarith,
      le_trans (diam_mono (image_mono fun z hz ↦ ?_)) ha⟩
    refine ⟨lt_of_le_of_lt (edist_triangle z y x) (lt_of_lt_of_eq (ENNReal.add_lt_add hz.1 hy) ?_),
      hz.2⟩
    rw [← ofReal_add (by linarith) (by linarith), sub_add_cancel]
  have S_cover : K ⊆ ⋃ r > 0, S r := by
    intro x hx
    have : oscillationWithin f D x < ε := hK x hx
    simp only [oscillationWithin, Filter.mem_map, iInf_lt_iff] at this
    obtain ⟨n, hn₁, hn₂⟩ := this
    obtain ⟨r, r0, hr⟩ := mem_nhdsWithin_iff.1 hn₁
    simp only [gt_iff_lt, mem_iUnion, exists_prop]
    have : ∀ r', (ENNReal.ofReal r') ≤ r → diam (f '' (ball x (ENNReal.ofReal r') ∩ D)) ≤ ε := by
      intro r' hr'
      refine le_trans (diam_mono (subset_trans ?_ (image_subset_iff.2 hr))) (le_of_lt hn₂)
      exact image_mono (inter_subset_inter_left D (ball_subset_ball hr'))
    by_cases r_top : r = ⊤
    · use 1, one_pos, 2, one_lt_two, this 2 (by simp only [r_top, le_top])
    · obtain ⟨r', hr'⟩ := exists_between (toReal_pos (ne_of_gt r0) r_top)
      use r', hr'.1, r.toReal, hr'.2, this r.toReal ofReal_toReal_le
  have S_antitone : ∀ (r₁ r₂ : ℝ), r₁ ≤ r₂ → S r₂ ⊆ S r₁ :=
    fun r₁ r₂ hr x ⟨a, ar₂, ha⟩ ↦ ⟨a, lt_of_le_of_lt hr ar₂, ha⟩
  obtain ⟨δ, δ0, hδ⟩ : ∃ r > 0, K ⊆ S r := by
    obtain ⟨T, Tb, Tfin, hT⟩ := comp.elim_finite_subcover_image S_open S_cover
    by_cases T_nonempty : T.Nonempty
    · use Tfin.isWF.min T_nonempty, Tb (Tfin.isWF.min_mem T_nonempty)
      intro x hx
      obtain ⟨r, hr⟩ := mem_iUnion.1 (hT hx)
      simp only [mem_iUnion, exists_prop] at hr
      exact (S_antitone _ r (IsWF.min_le Tfin.isWF T_nonempty hr.1)) hr.2
    · rw [not_nonempty_iff_eq_empty] at T_nonempty
      use 1, one_pos, subset_trans hT (by simp [T_nonempty])
  /-
    case intro.intro
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : PseudoEMetricSpace E
    K : Set E
    f : E → F
    D : Set E
    ε : ENNReal
    comp : IsCompact K
    hK : ∀ (x : E), Membership.mem K x → LT.lt (oscillationWithin f D x) ε
    S : Real → Set E := fun r => setOf fun x => Exists fun a => And (GT.gt a r) (L …
    S_open : ∀ (r : Real), GT.gt r 0 → IsOpen (S r)
    S_cover : HasSubset.Subset K (Set.iUnion fun r => Set.iUnion fun h => S r)
    S_antitone : ∀ (r₁ r₂ : Real), LE.le r₁ r₂ → HasSubset.Subset (S r₂) (S r₁)
    δ : Real
    δ0 : GT.gt δ 0
    hδ : HasSubset.Subset K (S δ)
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (x : E), Membership.mem K x → LE.le (EMet …
  -/
  use δ, δ0
  /-
    case right
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : PseudoEMetricSpace E
    K : Set E
    f : E → F
    D : Set E
    ε : ENNReal
    comp : IsCompact K
    hK : ∀ (x : E), Membership.mem K x → LT.lt (oscillationWithin f D x) ε
    S : Real → Set E := fun r => setOf fun x => Exists fun a => And (GT.gt a r) (L …
    S_open : ∀ (r : Real), GT.gt r 0 → IsOpen (S r)
    S_cover : HasSubset.Subset K (Set.iUnion fun r => Set.iUnion fun h => S r)
    S_antitone : ∀ (r₁ r₂ : Real), LE.le r₁ r₂ → HasSubset.Subset (S r₂) (S r₁)
    δ : Real
    δ0 : GT.gt δ 0
    hδ : HasSubset.Subset K (S δ)
    ⊢ ∀ (x : E), Membership.mem K x → LE.le (EMetric.diam (Set.image f (Inter.inte …
  -/
  intro x xK
  /-
    case right
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : PseudoEMetricSpace E
    K : Set E
    f : E → F
    D : Set E
    ε : ENNReal
    comp : IsCompact K
    hK : ∀ (x : E), Membership.mem K x → LT.lt (oscillationWithin f D x) ε
    S : Real → Set E := fun r => setOf fun x => Exists fun a => And (GT.gt a r) (L …
    S_open : ∀ (r : Real), GT.gt r 0 → IsOpen (S r)
    S_cover : HasSubset.Subset K (Set.iUnion fun r => Set.iUnion fun h => S r)
    S_antitone : ∀ (r₁ r₂ : Real), LE.le r₁ r₂ → HasSubset.Subset (S r₂) (S r₁)
    δ : Real
    δ0 : GT.gt δ 0
    hδ : HasSubset.Subset K (S δ)
    x : E
    xK : Membership.mem K x
    ⊢ LE.le (EMetric.diam (Set.image f (Inter.inter (EMetric.ball x (ENNReal.ofRea …
  -/
  obtain ⟨a, δa, ha⟩ := hδ xK
  exact (diam_mono <| image_mono <| inter_subset_inter_left D <| ball_subset_ball <|
    coe_le_coe.2 <| Real.toNNReal_mono (le_of_lt δa)).trans ha


/-- If `oscillation f x < ε` at every `x` in a compact set `K`, then there exists `δ > 0` such
that the oscillation of `f` on `ball x δ` is less than `ε` for every `x` in `K`. -/
theorem uniform_oscillation {K : Set E} (comp : IsCompact K)
    {f : E → F} {ε : ENNReal} (hK : ∀ x ∈ K, oscillation f x < ε) :
    ∃ δ > 0, ∀ x ∈ K, diam (f '' (ball x (ENNReal.ofReal δ))) ≤ ε := by
  /-
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : PseudoEMetricSpace E
    K : Set E
    comp : IsCompact K
    f : E → F
    ε : ENNReal
    hK : ∀ (x : E), Membership.mem K x → LT.lt (oscillation f x) ε
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (x : E), Membership.mem K x → LE.le (EMet …
  -/
  simp only [← oscillationWithin_univ_eq_oscillation] at hK
  /-
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : PseudoEMetricSpace E
    K : Set E
    comp : IsCompact K
    f : E → F
    ε : ENNReal
    hK : ∀ (x : E), Membership.mem K x → LT.lt (oscillationWithin f Set.univ x) ε
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (x : E), Membership.mem K x → LE.le (EMet …
  -/
  convert ← comp.uniform_oscillationWithin hK
  /-
    case h.e'_2.h.h.e'_2.h.h'.h.e'_3.h.e'_3.h.e'_4
    E : Type u
    F : Type v
    inst✝¹ : PseudoEMetricSpace F
    inst✝ : PseudoEMetricSpace E
    K : Set E
    comp : IsCompact K
    f : E → F
    ε : ENNReal
    hK : ∀ (x : E), Membership.mem K x → LT.lt (oscillationWithin f Set.univ x) ε
    x✝ : Real
    a✝¹ : E
    a✝ : Membership.mem K a✝¹
    ⊢ Eq (Inter.inter (EMetric.ball a✝¹ (ENNReal.ofReal x✝)) Set.univ) (EMetric.ba …
  -/
  exact inter_univ _
  /-
    🎉 no goals
  -/


