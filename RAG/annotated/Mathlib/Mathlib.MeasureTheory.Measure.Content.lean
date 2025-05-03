/-- A content is an additive function on compact sets taking values in `ℝ≥0`. It is a device
from which one can define a measure. -/
structure Content (G : Type w) [TopologicalSpace G] where
  toFun : Compacts G → ℝ≥0
  mono' : ∀ K₁ K₂ : Compacts G, (K₁ : Set G) ⊆ K₂ → toFun K₁ ≤ toFun K₂
  sup_disjoint' :
    ∀ K₁ K₂ : Compacts G, Disjoint (K₁ : Set G) K₂ → IsClosed (K₁ : Set G) → IsClosed (K₂ : Set G)
      → toFun (K₁ ⊔ K₂) = toFun K₁ + toFun K₂
  sup_le' : ∀ K₁ K₂ : Compacts G, toFun (K₁ ⊔ K₂) ≤ toFun K₁ + toFun K₂


instance : Inhabited (Content G) :=
  ⟨{  toFun := fun _ => 0
                  /-
                    G : Type w
                    inst✝ : TopologicalSpace G
                    ⊢ ∀ (K₁ K₂ : TopologicalSpace.Compacts G), HasSubset.Subset ↑K₁ ↑K₂ → LE.le (( …
                  -/
      mono' := by simp
                  /-
                    🎉 no goals
                  -/
                          /-
                            G : Type w
                            inst✝ : TopologicalSpace G
                            ⊢ ∀ (K₁ K₂ : TopologicalSpace.Compacts G), Disjoint ↑K₁ ↑K₂ → IsClosed ↑K₁ → I …
                          -/
      sup_disjoint' := by simp
                          /-
                            🎉 no goals
                          -/
                    /-
                      G : Type w
                      inst✝ : TopologicalSpace G
                      ⊢ ∀ (K₁ K₂ : TopologicalSpace.Compacts G), LE.le ((fun x => 0) (Max.max K₁ K₂) …
                    -/
      sup_le' := by simp }⟩
                    /-
                      🎉 no goals
                    -/


/-- Although the `toFun` field of a content takes values in `ℝ≥0`, we register a coercion to
functions taking values in `ℝ≥0∞` as most constructions below rely on taking iSups and iInfs, which
is more convenient in a complete lattice, and aim at constructing a measure. -/
instance : CoeFun (Content G) fun _ => Compacts G → ℝ≥0∞ :=
  ⟨fun μ s => μ.toFun s⟩


theorem apply_eq_coe_toFun (K : Compacts G) : μ K = μ.toFun K :=
  rfl


theorem mono (K₁ K₂ : Compacts G) (h : (K₁ : Set G) ⊆ K₂) : μ K₁ ≤ μ K₂ := by
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : HasSubset.Subset ↑K₁ ↑K₂
    ⊢ LE.le ((fun s => ↑(μ.toFun s)) K₁) ((fun s => ↑(μ.toFun s)) K₂)
  -/
  simp [apply_eq_coe_toFun, μ.mono' _ _ h]
  /-
    🎉 no goals
  -/


theorem sup_disjoint (K₁ K₂ : Compacts G) (h : Disjoint (K₁ : Set G) K₂)
    (h₁ : IsClosed (K₁ : Set G)) (h₂ : IsClosed (K₂ : Set G)) :
    μ (K₁ ⊔ K₂) = μ K₁ + μ K₂ := by
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    K₁ K₂ : TopologicalSpace.Compacts G
    h : Disjoint ↑K₁ ↑K₂
    h₁ : IsClosed ↑K₁
    h₂ : IsClosed ↑K₂
    ⊢ Eq ((fun s => ↑(μ.toFun s)) (Max.max K₁ K₂)) (HAdd.hAdd ((fun s => ↑(μ.toFun …
  -/
  simp [apply_eq_coe_toFun, μ.sup_disjoint' _ _ h]
  /-
    🎉 no goals
  -/


theorem sup_le (K₁ K₂ : Compacts G) : μ (K₁ ⊔ K₂) ≤ μ K₁ + μ K₂ := by
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    K₁ K₂ : TopologicalSpace.Compacts G
    ⊢ LE.le ((fun s => ↑(μ.toFun s)) (Max.max K₁ K₂)) (HAdd.hAdd ((fun s => ↑(μ.to …
  -/
  simp only [apply_eq_coe_toFun]
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    K₁ K₂ : TopologicalSpace.Compacts G
    ⊢ LE.le (↑(μ.toFun (Max.max K₁ K₂))) (HAdd.hAdd ↑(μ.toFun K₁) ↑(μ.toFun K₂))
  -/
  norm_cast
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    K₁ K₂ : TopologicalSpace.Compacts G
    ⊢ LE.le (μ.toFun (Max.max K₁ K₂)) (HAdd.hAdd (μ.toFun K₁) (μ.toFun K₂))
  -/
  exact μ.sup_le' _ _
  /-
    🎉 no goals
  -/


theorem lt_top (K : Compacts G) : μ K < ∞ :=
  ENNReal.coe_lt_top


theorem empty : μ ⊥ = 0 := by
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    ⊢ Eq ((fun s => ↑(μ.toFun s)) Bot.bot) 0
  -/
  have := μ.sup_disjoint' ⊥ ⊥
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    this : Disjoint ↑Bot.bot ↑Bot.bot → IsClosed ↑Bot.bot → IsClosed ↑Bot.bot → Eq …
    ⊢ Eq ((fun s => ↑(μ.toFun s)) Bot.bot) 0
  -/
  simpa [apply_eq_coe_toFun] using this
  /-
    🎉 no goals
  -/


/-- Constructing the inner content of a content. From a content defined on the compact sets, we
  obtain a function defined on all open sets, by taking the supremum of the content of all compact
  subsets. -/
def innerContent (U : Opens G) : ℝ≥0∞ :=
  ⨆ (K : Compacts G) (_ : (K : Set G) ⊆ U), μ K


theorem le_innerContent (K : Compacts G) (U : Opens G) (h2 : (K : Set G) ⊆ U) :
    μ K ≤ μ.innerContent U :=
  le_iSup_of_le K <| le_iSup (fun _ ↦ (μ.toFun K : ℝ≥0∞)) h2


theorem innerContent_le (U : Opens G) (K : Compacts G) (h2 : (U : Set G) ⊆ K) :
    μ.innerContent U ≤ μ K :=
  iSup₂_le fun _ hK' => μ.mono _ _ (Subset.trans hK' h2)


theorem innerContent_of_isCompact {K : Set G} (h1K : IsCompact K) (h2K : IsOpen K) :
    μ.innerContent ⟨K, h2K⟩ = μ ⟨K, h1K⟩ :=
  le_antisymm (iSup₂_le fun _ hK' => μ.mono _ ⟨K, h1K⟩ hK') (μ.le_innerContent _ _ Subset.rfl)


theorem innerContent_bot : μ.innerContent ⊥ = 0 := by
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    ⊢ Eq (μ.innerContent Bot.bot) 0
  -/
  refine le_antisymm ?_ (zero_le _)
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    ⊢ LE.le (μ.innerContent Bot.bot) 0
  -/
  rw [← μ.empty]
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    ⊢ LE.le (μ.innerContent Bot.bot) ((fun s => ↑(μ.toFun s)) Bot.bot)
  -/
  refine iSup₂_le fun K hK => ?_
  have : K = ⊥ := by
    ext1
    rw [subset_empty_iff.mp hK, Compacts.coe_bot]
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    K : TopologicalSpace.Compacts G
    hK : HasSubset.Subset ↑K ↑Bot.bot
    this : Eq K Bot.bot
    ⊢ LE.le ((fun s => ↑(μ.toFun s)) K) ((fun s => ↑(μ.toFun s)) Bot.bot)
  -/
  rw [this]
  /-
    🎉 no goals
  -/


/-- This is "unbundled", because that is required for the API of `inducedOuterMeasure`. -/
theorem innerContent_mono ⦃U V : Set G⦄ (hU : IsOpen U) (hV : IsOpen V) (h2 : U ⊆ V) :
    μ.innerContent ⟨U, hU⟩ ≤ μ.innerContent ⟨V, hV⟩ :=
  biSup_mono fun _ hK => hK.trans h2


theorem innerContent_exists_compact {U : Opens G} (hU : μ.innerContent U ≠ ∞) {ε : ℝ≥0}
    (hε : ε ≠ 0) : ∃ K : Compacts G, (K : Set G) ⊆ U ∧ μ.innerContent U ≤ μ K + ε := by
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    U : TopologicalSpace.Opens G
    hU : Ne (μ.innerContent U) Top.top
    ε : NNReal
    hε : Ne ε 0
    ⊢ Exists fun K => And (HasSubset.Subset ↑K ↑U) (LE.le (μ.innerContent U) (HAdd …
  -/
  have h'ε := ENNReal.coe_ne_zero.2 hε
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    U : TopologicalSpace.Opens G
    hU : Ne (μ.innerContent U) Top.top
    ε : NNReal
    hε : Ne ε 0
    h'ε : Ne (↑ε) 0
    ⊢ Exists fun K => And (HasSubset.Subset ↑K ↑U) (LE.le (μ.innerContent U) (HAdd …
  -/
  rcases le_or_lt (μ.innerContent U) ε with h | h
    /-
      case inl
      G : Type w
      inst✝ : TopologicalSpace G
      μ : MeasureTheory.Content G
      U : TopologicalSpace.Opens G
      hU : Ne (μ.innerContent U) Top.top
      ε : NNReal
      hε : Ne ε 0
      h'ε : Ne (↑ε) 0
      h : LE.le (μ.innerContent U) ↑ε
      ⊢ Exists fun K => And (HasSubset.Subset ↑K ↑U) (LE.le (μ.innerContent U) (HAdd …
    -/
  · exact ⟨⊥, empty_subset _, le_add_left h⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    U : TopologicalSpace.Opens G
    hU : Ne (μ.innerContent U) Top.top
    ε : NNReal
    hε : Ne ε 0
    h'ε : Ne (↑ε) 0
    h : LT.lt (↑ε) (μ.innerContent U)
    ⊢ Exists fun K => And (HasSubset.Subset ↑K ↑U) (LE.le (μ.innerContent U) (HAdd …
  -/
  have h₂ := ENNReal.sub_lt_self hU h.ne_bot h'ε
  /-
    case inr
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    U : TopologicalSpace.Opens G
    hU : Ne (μ.innerContent U) Top.top
    ε : NNReal
    hε : Ne ε 0
    h'ε : Ne (↑ε) 0
    h : LT.lt (↑ε) (μ.innerContent U)
    h₂ : LT.lt (HSub.hSub (μ.innerContent U) ↑ε) (μ.innerContent U)
    ⊢ Exists fun K => And (HasSubset.Subset ↑K ↑U) (LE.le (μ.innerContent U) (HAdd …
  -/
  conv at h₂ => rhs; rw [innerContent]
  /-
    case inr
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    U : TopologicalSpace.Opens G
    hU : Ne (μ.innerContent U) Top.top
    ε : NNReal
    hε : Ne ε 0
    h'ε : Ne (↑ε) 0
    h : LT.lt (↑ε) (μ.innerContent U)
    h₂ : LT.lt (HSub.hSub (μ.innerContent U) ↑ε) (iSup fun K => iSup fun x => (fun …
    ⊢ Exists fun K => And (HasSubset.Subset ↑K ↑U) (LE.le (μ.innerContent U) (HAdd …
  -/
  simp only [lt_iSup_iff] at h₂
  /-
    case inr
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    U : TopologicalSpace.Opens G
    hU : Ne (μ.innerContent U) Top.top
    ε : NNReal
    hε : Ne ε 0
    h'ε : Ne (↑ε) 0
    h : LT.lt (↑ε) (μ.innerContent U)
    h₂ : Exists fun i => Exists fun i_1 => LT.lt (HSub.hSub (μ.innerContent U) ↑ε) …
    ⊢ Exists fun K => And (HasSubset.Subset ↑K ↑U) (LE.le (μ.innerContent U) (HAdd …
  -/
  rcases h₂ with ⟨U, h1U, h2U⟩; refine ⟨U, h1U, ?_⟩
  /-
    case inr.intro.intro
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    U✝ : TopologicalSpace.Opens G
    hU : Ne (μ.innerContent U✝) Top.top
    ε : NNReal
    hε : Ne ε 0
    h'ε : Ne (↑ε) 0
    h : LT.lt (↑ε) (μ.innerContent U✝)
    U : TopologicalSpace.Compacts G
    h1U : HasSubset.Subset ↑U ↑U✝
    h2U : LT.lt (HSub.hSub (μ.innerContent U✝) ↑ε) ↑(μ.toFun U)
    ⊢ LE.le (μ.innerContent U✝) (HAdd.hAdd ((fun s => ↑(μ.toFun s)) U) ↑ε)
  -/
  rw [← tsub_le_iff_right]; exact le_of_lt h2U
                            /-
                              🎉 no goals
                            -/


/-- The inner content of a supremum of opens is at most the sum of the individual inner contents. -/
theorem innerContent_iSup_nat [R1Space G] (U : ℕ → Opens G) :
    μ.innerContent (⨆ i : ℕ, U i) ≤ ∑' i : ℕ, μ.innerContent (U i) := by
  have h3 : ∀ (t : Finset ℕ) (K : ℕ → Compacts G), μ (t.sup K) ≤ t.sum fun i => μ (K i) := by
    intro t K
    refine Finset.induction_on t ?_ ?_
    · simp only [μ.empty, nonpos_iff_eq_zero, Finset.sum_empty, Finset.sup_empty]
    · intro n s hn ih
      rw [Finset.sup_insert, Finset.sum_insert hn]
      exact le_trans (μ.sup_le _ _) (add_le_add_left ih _)
  /-
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    U : Nat → TopologicalSpace.Opens G
    h3 : ∀ (t : Finset Nat) (K : Nat → TopologicalSpace.Compacts G), LE.le ((fun s …
    ⊢ LE.le (μ.innerContent (iSup fun i => U i)) (tsum fun i => μ.innerContent (U  …
  -/
  refine iSup₂_le fun K hK => ?_
  obtain ⟨t, ht⟩ :=
    K.isCompact.elim_finite_subcover _ (fun i => (U i).isOpen) (by rwa [← Opens.coe_iSup])
  rcases K.isCompact.finite_compact_cover t (SetLike.coe ∘ U) (fun i _ => (U i).isOpen) ht with
    ⟨K', h1K', h2K', h3K'⟩
  /-
    case intro.intro.intro.intro
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    U : Nat → TopologicalSpace.Opens G
    h3 : ∀ (t : Finset Nat) (K : Nat → TopologicalSpace.Compacts G), LE.le ((fun s …
    K : TopologicalSpace.Compacts G
    hK : HasSubset.Subset ↑K ↑(iSup fun i => U i)
    t : Finset Nat
    ht : HasSubset.Subset (↑K) (Set.iUnion fun i => Set.iUnion fun h => ↑(U i))
    K' : Nat → Set G
    h1K' : ∀ (i : Nat), IsCompact (K' i)
    h2K' : ∀ (i : Nat), HasSubset.Subset (K' i) (Function.comp SetLike.coe U i)
    h3K' : Eq (↑K) (Set.iUnion fun i => Set.iUnion fun h => K' i)
    ⊢ LE.le ((fun s => ↑(μ.toFun s)) K) (tsum fun i => μ.innerContent (U i))
  -/
  let L : ℕ → Compacts G := fun n => ⟨K' n, h1K' n⟩
  /-
    case intro.intro.intro.intro
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    U : Nat → TopologicalSpace.Opens G
    h3 : ∀ (t : Finset Nat) (K : Nat → TopologicalSpace.Compacts G), LE.le ((fun s …
    K : TopologicalSpace.Compacts G
    hK : HasSubset.Subset ↑K ↑(iSup fun i => U i)
    t : Finset Nat
    ht : HasSubset.Subset (↑K) (Set.iUnion fun i => Set.iUnion fun h => ↑(U i))
    K' : Nat → Set G
    h1K' : ∀ (i : Nat), IsCompact (K' i)
    h2K' : ∀ (i : Nat), HasSubset.Subset (K' i) (Function.comp SetLike.coe U i)
    h3K' : Eq (↑K) (Set.iUnion fun i => Set.iUnion fun h => K' i)
    L : Nat → TopologicalSpace.Compacts G := fun n => { carrier := K' n, isCompact …
    ⊢ LE.le ((fun s => ↑(μ.toFun s)) K) (tsum fun i => μ.innerContent (U i))
  -/
  convert le_trans (h3 t L) _
    /-
      case h.e'_3.h.e'_1
      G : Type w
      inst✝¹ : TopologicalSpace G
      μ : MeasureTheory.Content G
      inst✝ : R1Space G
      U : Nat → TopologicalSpace.Opens G
      h3 : ∀ (t : Finset Nat) (K : Nat → TopologicalSpace.Compacts G), LE.le ((fun s …
      K : TopologicalSpace.Compacts G
      hK : HasSubset.Subset ↑K ↑(iSup fun i => U i)
      t : Finset Nat
      ht : HasSubset.Subset (↑K) (Set.iUnion fun i => Set.iUnion fun h => ↑(U i))
      K' : Nat → Set G
      h1K' : ∀ (i : Nat), IsCompact (K' i)
      h2K' : ∀ (i : Nat), HasSubset.Subset (K' i) (Function.comp SetLike.coe U i)
      h3K' : Eq (↑K) (Set.iUnion fun i => Set.iUnion fun h => K' i)
      L : Nat → TopologicalSpace.Compacts G := fun n => { carrier := K' n, isCompact …
      ⊢ Eq K (t.sup L)
    -/
  · ext1
    /-
      case h.e'_3.h.e'_1.h
      G : Type w
      inst✝¹ : TopologicalSpace G
      μ : MeasureTheory.Content G
      inst✝ : R1Space G
      U : Nat → TopologicalSpace.Opens G
      h3 : ∀ (t : Finset Nat) (K : Nat → TopologicalSpace.Compacts G), LE.le ((fun s …
      K : TopologicalSpace.Compacts G
      hK : HasSubset.Subset ↑K ↑(iSup fun i => U i)
      t : Finset Nat
      ht : HasSubset.Subset (↑K) (Set.iUnion fun i => Set.iUnion fun h => ↑(U i))
      K' : Nat → Set G
      h1K' : ∀ (i : Nat), IsCompact (K' i)
      h2K' : ∀ (i : Nat), HasSubset.Subset (K' i) (Function.comp SetLike.coe U i)
      h3K' : Eq (↑K) (Set.iUnion fun i => Set.iUnion fun h => K' i)
      L : Nat → TopologicalSpace.Compacts G := fun n => { carrier := K' n, isCompact …
      ⊢ Eq ↑K ↑(t.sup L)
    -/
    rw [Compacts.coe_finset_sup, Finset.sup_eq_iSup]
    /-
      case h.e'_3.h.e'_1.h
      G : Type w
      inst✝¹ : TopologicalSpace G
      μ : MeasureTheory.Content G
      inst✝ : R1Space G
      U : Nat → TopologicalSpace.Opens G
      h3 : ∀ (t : Finset Nat) (K : Nat → TopologicalSpace.Compacts G), LE.le ((fun s …
      K : TopologicalSpace.Compacts G
      hK : HasSubset.Subset ↑K ↑(iSup fun i => U i)
      t : Finset Nat
      ht : HasSubset.Subset (↑K) (Set.iUnion fun i => Set.iUnion fun h => ↑(U i))
      K' : Nat → Set G
      h1K' : ∀ (i : Nat), IsCompact (K' i)
      h2K' : ∀ (i : Nat), HasSubset.Subset (K' i) (Function.comp SetLike.coe U i)
      h3K' : Eq (↑K) (Set.iUnion fun i => Set.iUnion fun h => K' i)
      L : Nat → TopologicalSpace.Compacts G := fun n => { carrier := K' n, isCompact …
      ⊢ Eq (↑K) (iSup fun a => iSup fun h => ↑(L a))
    -/
    exact h3K'
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.intro.convert_2
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    U : Nat → TopologicalSpace.Opens G
    h3 : ∀ (t : Finset Nat) (K : Nat → TopologicalSpace.Compacts G), LE.le ((fun s …
    K : TopologicalSpace.Compacts G
    hK : HasSubset.Subset ↑K ↑(iSup fun i => U i)
    t : Finset Nat
    ht : HasSubset.Subset (↑K) (Set.iUnion fun i => Set.iUnion fun h => ↑(U i))
    K' : Nat → Set G
    h1K' : ∀ (i : Nat), IsCompact (K' i)
    h2K' : ∀ (i : Nat), HasSubset.Subset (K' i) (Function.comp SetLike.coe U i)
    h3K' : Eq (↑K) (Set.iUnion fun i => Set.iUnion fun h => K' i)
    L : Nat → TopologicalSpace.Compacts G := fun n => { carrier := K' n, isCompact …
    ⊢ LE.le (t.sum fun i => (fun s => ↑(μ.toFun s)) (L i)) (tsum fun i => μ.innerC …
  -/
  refine le_trans (Finset.sum_le_sum ?_) (ENNReal.sum_le_tsum t)
  /-
    case intro.intro.intro.intro.convert_2
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    U : Nat → TopologicalSpace.Opens G
    h3 : ∀ (t : Finset Nat) (K : Nat → TopologicalSpace.Compacts G), LE.le ((fun s …
    K : TopologicalSpace.Compacts G
    hK : HasSubset.Subset ↑K ↑(iSup fun i => U i)
    t : Finset Nat
    ht : HasSubset.Subset (↑K) (Set.iUnion fun i => Set.iUnion fun h => ↑(U i))
    K' : Nat → Set G
    h1K' : ∀ (i : Nat), IsCompact (K' i)
    h2K' : ∀ (i : Nat), HasSubset.Subset (K' i) (Function.comp SetLike.coe U i)
    h3K' : Eq (↑K) (Set.iUnion fun i => Set.iUnion fun h => K' i)
    L : Nat → TopologicalSpace.Compacts G := fun n => { carrier := K' n, isCompact …
    ⊢ ∀ (i : Nat), Membership.mem t i → LE.le ((fun s => ↑(μ.toFun s)) (L i)) (μ.i …
  -/
  intro i _
  /-
    case intro.intro.intro.intro.convert_2
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    U : Nat → TopologicalSpace.Opens G
    h3 : ∀ (t : Finset Nat) (K : Nat → TopologicalSpace.Compacts G), LE.le ((fun s …
    K : TopologicalSpace.Compacts G
    hK : HasSubset.Subset ↑K ↑(iSup fun i => U i)
    t : Finset Nat
    ht : HasSubset.Subset (↑K) (Set.iUnion fun i => Set.iUnion fun h => ↑(U i))
    K' : Nat → Set G
    h1K' : ∀ (i : Nat), IsCompact (K' i)
    h2K' : ∀ (i : Nat), HasSubset.Subset (K' i) (Function.comp SetLike.coe U i)
    h3K' : Eq (↑K) (Set.iUnion fun i => Set.iUnion fun h => K' i)
    L : Nat → TopologicalSpace.Compacts G := fun n => { carrier := K' n, isCompact …
    i : Nat
    a✝ : Membership.mem t i
    ⊢ LE.le ((fun s => ↑(μ.toFun s)) (L i)) (μ.innerContent (U i))
  -/
  refine le_trans ?_ (le_iSup _ (L i))
  /-
    case intro.intro.intro.intro.convert_2
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    U : Nat → TopologicalSpace.Opens G
    h3 : ∀ (t : Finset Nat) (K : Nat → TopologicalSpace.Compacts G), LE.le ((fun s …
    K : TopologicalSpace.Compacts G
    hK : HasSubset.Subset ↑K ↑(iSup fun i => U i)
    t : Finset Nat
    ht : HasSubset.Subset (↑K) (Set.iUnion fun i => Set.iUnion fun h => ↑(U i))
    K' : Nat → Set G
    h1K' : ∀ (i : Nat), IsCompact (K' i)
    h2K' : ∀ (i : Nat), HasSubset.Subset (K' i) (Function.comp SetLike.coe U i)
    h3K' : Eq (↑K) (Set.iUnion fun i => Set.iUnion fun h => K' i)
    L : Nat → TopologicalSpace.Compacts G := fun n => { carrier := K' n, isCompact …
    i : Nat
    a✝ : Membership.mem t i
    ⊢ LE.le ((fun s => ↑(μ.toFun s)) (L i)) (iSup fun x => (fun s => ↑(μ.toFun s)) …
  -/
  refine le_trans ?_ (le_iSup _ (h2K' i))
  /-
    case intro.intro.intro.intro.convert_2
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    U : Nat → TopologicalSpace.Opens G
    h3 : ∀ (t : Finset Nat) (K : Nat → TopologicalSpace.Compacts G), LE.le ((fun s …
    K : TopologicalSpace.Compacts G
    hK : HasSubset.Subset ↑K ↑(iSup fun i => U i)
    t : Finset Nat
    ht : HasSubset.Subset (↑K) (Set.iUnion fun i => Set.iUnion fun h => ↑(U i))
    K' : Nat → Set G
    h1K' : ∀ (i : Nat), IsCompact (K' i)
    h2K' : ∀ (i : Nat), HasSubset.Subset (K' i) (Function.comp SetLike.coe U i)
    h3K' : Eq (↑K) (Set.iUnion fun i => Set.iUnion fun h => K' i)
    L : Nat → TopologicalSpace.Compacts G := fun n => { carrier := K' n, isCompact …
    i : Nat
    a✝ : Membership.mem t i
    ⊢ LE.le ((fun s => ↑(μ.toFun s)) (L i)) ((fun s => ↑(μ.toFun s)) (L i))
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The inner content of a union of sets is at most the sum of the individual inner contents.
  This is the "unbundled" version of `innerContent_iSup_nat`.
  It is required for the API of `inducedOuterMeasure`. -/
theorem innerContent_iUnion_nat [R1Space G] ⦃U : ℕ → Set G⦄
    (hU : ∀ i : ℕ, IsOpen (U i)) :
    μ.innerContent ⟨⋃ i : ℕ, U i, isOpen_iUnion hU⟩ ≤ ∑' i : ℕ, μ.innerContent ⟨U i, hU i⟩ := by
  /-
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    U : Nat → Set G
    hU : ∀ (i : Nat), IsOpen (U i)
    ⊢ LE.le (μ.innerContent { carrier := Set.iUnion fun i => U i, is_open' := ⋯ }) …
  -/
  have := μ.innerContent_iSup_nat fun i => ⟨U i, hU i⟩
  /-
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    U : Nat → Set G
    hU : ∀ (i : Nat), IsOpen (U i)
    this : LE.le (μ.innerContent (iSup fun i => { carrier := U i, is_open' := ⋯ }) …
    ⊢ LE.le (μ.innerContent { carrier := Set.iUnion fun i => U i, is_open' := ⋯ }) …
  -/
  rwa [Opens.iSup_def] at this
  /-
    🎉 no goals
  -/


theorem innerContent_comap (f : G ≃ₜ G) (h : ∀ ⦃K : Compacts G⦄, μ (K.map f f.continuous) = μ K)
    (U : Opens G) : μ.innerContent (Opens.comap f U) = μ.innerContent U := by
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    f : Homeomorph G G
    h : ∀ ⦃K : TopologicalSpace.Compacts G⦄, Eq ((fun s => ↑(μ.toFun s)) (Topologi …
    U : TopologicalSpace.Opens G
    ⊢ Eq (μ.innerContent ((TopologicalSpace.Opens.comap ↑f) U)) (μ.innerContent U)
  -/
  refine (Compacts.equiv f).surjective.iSup_congr _ fun K => iSup_congr_Prop image_subset_iff ?_
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    f : Homeomorph G G
    h : ∀ ⦃K : TopologicalSpace.Compacts G⦄, Eq ((fun s => ↑(μ.toFun s)) (Topologi …
    U : TopologicalSpace.Opens G
    K : TopologicalSpace.Compacts G
    ⊢ HasSubset.Subset ↑K ↑((TopologicalSpace.Opens.comap ↑f) U) → Eq ((fun s => ↑ …
  -/
  intro hK
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    f : Homeomorph G G
    h : ∀ ⦃K : TopologicalSpace.Compacts G⦄, Eq ((fun s => ↑(μ.toFun s)) (Topologi …
    U : TopologicalSpace.Opens G
    K : TopologicalSpace.Compacts G
    hK : HasSubset.Subset ↑K ↑((TopologicalSpace.Opens.comap ↑f) U)
    ⊢ Eq ((fun s => ↑(μ.toFun s)) ((TopologicalSpace.Compacts.equiv f) K)) ((fun s …
  -/
  simp only [Equiv.coe_fn_mk, Subtype.mk_eq_mk, Compacts.equiv]
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    f : Homeomorph G G
    h : ∀ ⦃K : TopologicalSpace.Compacts G⦄, Eq ((fun s => ↑(μ.toFun s)) (Topologi …
    U : TopologicalSpace.Opens G
    K : TopologicalSpace.Compacts G
    hK : HasSubset.Subset ↑K ↑((TopologicalSpace.Opens.comap ↑f) U)
    ⊢ Eq ↑(μ.toFun (TopologicalSpace.Compacts.map ⇑f ⋯ K)) ↑(μ.toFun K)
  -/
  apply h
  /-
    🎉 no goals
  -/


@[to_additive]
theorem is_mul_left_invariant_innerContent [Group G] [TopologicalGroup G]
    (h : ∀ (g : G) {K : Compacts G}, μ (K.map _ <| continuous_mul_left g) = μ K) (g : G)
    (U : Opens G) :
    μ.innerContent (Opens.comap (Homeomorph.mulLeft g) U) = μ.innerContent U := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    h : ∀ (g : G) {K : TopologicalSpace.Compacts G}, Eq ((fun s => ↑(μ.toFun s)) ( …
    g : G
    U : TopologicalSpace.Opens G
    ⊢ Eq (μ.innerContent ((TopologicalSpace.Opens.comap ↑(Homeomorph.mulLeft g)) U …
  -/
  convert μ.innerContent_comap (Homeomorph.mulLeft g) (fun K => h g) U
  /-
    🎉 no goals
  -/


@[to_additive]
theorem innerContent_pos_of_is_mul_left_invariant [Group G] [TopologicalGroup G]
    (h3 : ∀ (g : G) {K : Compacts G}, μ (K.map _ <| continuous_mul_left g) = μ K) (K : Compacts G)
    (hK : μ K ≠ 0) (U : Opens G) (hU : (U : Set G).Nonempty) : 0 < μ.innerContent U := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    h3 : ∀ (g : G) {K : TopologicalSpace.Compacts G}, Eq ((fun s => ↑(μ.toFun s))  …
    K : TopologicalSpace.Compacts G
    hK : Ne ((fun s => ↑(μ.toFun s)) K) 0
    U : TopologicalSpace.Opens G
    hU : (↑U).Nonempty
    ⊢ LT.lt 0 (μ.innerContent U)
  -/
  have : (interior (U : Set G)).Nonempty := by rwa [U.isOpen.interior_eq]
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    h3 : ∀ (g : G) {K : TopologicalSpace.Compacts G}, Eq ((fun s => ↑(μ.toFun s))  …
    K : TopologicalSpace.Compacts G
    hK : Ne ((fun s => ↑(μ.toFun s)) K) 0
    U : TopologicalSpace.Opens G
    hU : (↑U).Nonempty
    this : (interior ↑U).Nonempty
    ⊢ LT.lt 0 (μ.innerContent U)
  -/
  rcases compact_covered_by_mul_left_translates K.2 this with ⟨s, hs⟩
  suffices μ K ≤ s.card * μ.innerContent U by
    exact (ENNReal.mul_pos_iff.mp <| hK.bot_lt.trans_le this).2
  have : (K : Set G) ⊆ ↑(⨆ g ∈ s, Opens.comap (Homeomorph.mulLeft g : C(G, G)) U) := by
    simpa only [Opens.iSup_def, Opens.coe_comap, Subtype.coe_mk]
  /-
    case intro
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    h3 : ∀ (g : G) {K : TopologicalSpace.Compacts G}, Eq ((fun s => ↑(μ.toFun s))  …
    K : TopologicalSpace.Compacts G
    hK : Ne ((fun s => ↑(μ.toFun s)) K) 0
    U : TopologicalSpace.Opens G
    hU : (↑U).Nonempty
    this✝ : (interior ↑U).Nonempty
    s : Finset G
    hs : HasSubset.Subset K.carrier (Set.iUnion fun g => Set.iUnion fun h => Set.p …
    this : HasSubset.Subset ↑K ↑(iSup fun g => iSup fun h => (TopologicalSpace.Ope …
    ⊢ LE.le ((fun s => ↑(μ.toFun s)) K) (HMul.hMul (↑s.card) (μ.innerContent U))
  -/
  refine (μ.le_innerContent _ _ this).trans ?_
  refine
    (rel_iSup_sum μ.innerContent μ.innerContent_bot (· ≤ ·) μ.innerContent_iSup_nat _ _).trans ?_
  /-
    case intro
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    h3 : ∀ (g : G) {K : TopologicalSpace.Compacts G}, Eq ((fun s => ↑(μ.toFun s))  …
    K : TopologicalSpace.Compacts G
    hK : Ne ((fun s => ↑(μ.toFun s)) K) 0
    U : TopologicalSpace.Opens G
    hU : (↑U).Nonempty
    this✝ : (interior ↑U).Nonempty
    s : Finset G
    hs : HasSubset.Subset K.carrier (Set.iUnion fun g => Set.iUnion fun h => Set.p …
    this : HasSubset.Subset ↑K ↑(iSup fun g => iSup fun h => (TopologicalSpace.Ope …
    ⊢ LE.le (s.sum fun d => μ.innerContent ((TopologicalSpace.Opens.comap ↑(Homeom …
  -/
  simp only [μ.is_mul_left_invariant_innerContent h3, Finset.sum_const, nsmul_eq_mul, le_refl]
  /-
    🎉 no goals
  -/


theorem innerContent_mono' ⦃U V : Set G⦄ (hU : IsOpen U) (hV : IsOpen V) (h2 : U ⊆ V) :
    μ.innerContent ⟨U, hU⟩ ≤ μ.innerContent ⟨V, hV⟩ :=
  biSup_mono fun _ hK => hK.trans h2


/-- Extending a content on compact sets to an outer measure on all sets. -/
protected def outerMeasure : OuterMeasure G :=
  inducedOuterMeasure (fun U hU => μ.innerContent ⟨U, hU⟩) isOpen_empty μ.innerContent_bot


theorem outerMeasure_opens (U : Opens G) : μ.outerMeasure U = μ.innerContent U :=
  inducedOuterMeasure_eq' (fun _ => isOpen_iUnion) μ.innerContent_iUnion_nat μ.innerContent_mono U.2


theorem outerMeasure_of_isOpen (U : Set G) (hU : IsOpen U) :
    μ.outerMeasure U = μ.innerContent ⟨U, hU⟩ :=
  μ.outerMeasure_opens ⟨U, hU⟩


theorem outerMeasure_le (U : Opens G) (K : Compacts G) (hUK : (U : Set G) ⊆ K) :
    μ.outerMeasure U ≤ μ K :=
  (μ.outerMeasure_opens U).le.trans <| μ.innerContent_le U K hUK


theorem le_outerMeasure_compacts (K : Compacts G) : μ K ≤ μ.outerMeasure K := by
  /-
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    K : TopologicalSpace.Compacts G
    ⊢ LE.le ((fun s => ↑(μ.toFun s)) K) (μ.outerMeasure ↑K)
  -/
  rw [Content.outerMeasure, inducedOuterMeasure_eq_iInf]
    /-
      G : Type w
      inst✝¹ : TopologicalSpace G
      μ : MeasureTheory.Content G
      inst✝ : R1Space G
      K : TopologicalSpace.Compacts G
      ⊢ LE.le ((fun s => ↑(μ.toFun s)) K) (iInf fun t => iInf fun ht => iInf fun x = …
    -/
  · exact le_iInf fun U => le_iInf fun hU => le_iInf <| μ.le_innerContent K ⟨U, hU⟩
    /-
      🎉 no goals
    -/
    /-
      case PU
      G : Type w
      inst✝¹ : TopologicalSpace G
      μ : MeasureTheory.Content G
      inst✝ : R1Space G
      K : TopologicalSpace.Compacts G
      ⊢ ∀ ⦃f : Nat → Set G⦄, (∀ (i : Nat), IsOpen (f i)) → IsOpen (Set.iUnion fun i  …
    -/
  · exact fun U hU => isOpen_iUnion hU
    /-
      🎉 no goals
    -/
    /-
      case msU
      G : Type w
      inst✝¹ : TopologicalSpace G
      μ : MeasureTheory.Content G
      inst✝ : R1Space G
      K : TopologicalSpace.Compacts G
      ⊢ ∀ ⦃f : Nat → Set G⦄ (hm : ∀ (i : Nat), IsOpen (f i)), LE.le (μ.innerContent  …
    -/
  · exact μ.innerContent_iUnion_nat
    /-
      🎉 no goals
    -/
    /-
      case m_mono
      G : Type w
      inst✝¹ : TopologicalSpace G
      μ : MeasureTheory.Content G
      inst✝ : R1Space G
      K : TopologicalSpace.Compacts G
      ⊢ ∀ ⦃s₁ s₂ : Set G⦄ (hs₁ : IsOpen s₁) (hs₂ : IsOpen s₂), HasSubset.Subset s₁ s …
    -/
  · exact μ.innerContent_mono
    /-
      🎉 no goals
    -/


theorem outerMeasure_eq_iInf (A : Set G) :
    μ.outerMeasure A = ⨅ (U : Set G) (hU : IsOpen U) (_ : A ⊆ U), μ.innerContent ⟨U, hU⟩ :=
  inducedOuterMeasure_eq_iInf _ μ.innerContent_iUnion_nat μ.innerContent_mono A


theorem outerMeasure_interior_compacts (K : Compacts G) : μ.outerMeasure (interior K) ≤ μ K :=
  (μ.outerMeasure_opens <| Opens.interior K).le.trans <| μ.innerContent_le _ _ interior_subset


theorem outerMeasure_exists_compact {U : Opens G} (hU : μ.outerMeasure U ≠ ∞) {ε : ℝ≥0}
    (hε : ε ≠ 0) : ∃ K : Compacts G, (K : Set G) ⊆ U ∧ μ.outerMeasure U ≤ μ.outerMeasure K + ε := by
  /-
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    U : TopologicalSpace.Opens G
    hU : Ne (μ.outerMeasure ↑U) Top.top
    ε : NNReal
    hε : Ne ε 0
    ⊢ Exists fun K => And (HasSubset.Subset ↑K ↑U) (LE.le (μ.outerMeasure ↑U) (HAd …
  -/
  rw [μ.outerMeasure_opens] at hU ⊢
  /-
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    U : TopologicalSpace.Opens G
    hU : Ne (μ.innerContent U) Top.top
    ε : NNReal
    hε : Ne ε 0
    ⊢ Exists fun K => And (HasSubset.Subset ↑K ↑U) (LE.le (μ.innerContent U) (HAdd …
  -/
  rcases μ.innerContent_exists_compact hU hε with ⟨K, h1K, h2K⟩
  /-
    case intro.intro
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    U : TopologicalSpace.Opens G
    hU : Ne (μ.innerContent U) Top.top
    ε : NNReal
    hε : Ne ε 0
    K : TopologicalSpace.Compacts G
    h1K : HasSubset.Subset ↑K ↑U
    h2K : LE.le (μ.innerContent U) (HAdd.hAdd ((fun s => ↑(μ.toFun s)) K) ↑ε)
    ⊢ Exists fun K => And (HasSubset.Subset ↑K ↑U) (LE.le (μ.innerContent U) (HAdd …
  -/
  exact ⟨K, h1K, le_trans h2K <| add_le_add_right (μ.le_outerMeasure_compacts K) _⟩
  /-
    🎉 no goals
  -/


theorem outerMeasure_exists_open {A : Set G} (hA : μ.outerMeasure A ≠ ∞) {ε : ℝ≥0} (hε : ε ≠ 0) :
    ∃ U : Opens G, A ⊆ U ∧ μ.outerMeasure U ≤ μ.outerMeasure A + ε := by
  rcases inducedOuterMeasure_exists_set _ μ.innerContent_iUnion_nat μ.innerContent_mono hA
      (ENNReal.coe_ne_zero.2 hε) with
    ⟨U, hU, h2U, h3U⟩
  /-
    case intro.intro.intro
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    A : Set G
    hA : Ne (μ.outerMeasure A) Top.top
    ε : NNReal
    hε : Ne ε 0
    U : Set G
    hU : IsOpen U
    h2U : HasSubset.Subset A U
    h3U : LE.le ((MeasureTheory.inducedOuterMeasure (fun s₁ hs₁ => μ.innerContent  …
    ⊢ Exists fun U => And (HasSubset.Subset A ↑U) (LE.le (μ.outerMeasure ↑U) (HAdd …
  -/
  exact ⟨⟨U, hU⟩, h2U, h3U⟩
  /-
    🎉 no goals
  -/


theorem outerMeasure_preimage (f : G ≃ₜ G) (h : ∀ ⦃K : Compacts G⦄, μ (K.map f f.continuous) = μ K)
    (A : Set G) : μ.outerMeasure (f ⁻¹' A) = μ.outerMeasure A := by
  refine inducedOuterMeasure_preimage _ μ.innerContent_iUnion_nat μ.innerContent_mono _
    (fun _ => f.isOpen_preimage) ?_
  /-
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    f : Homeomorph G G
    h : ∀ ⦃K : TopologicalSpace.Compacts G⦄, Eq ((fun s => ↑(μ.toFun s)) (Topologi …
    A : Set G
    ⊢ ∀ (s : Set G) (hs : IsOpen s), Eq (μ.innerContent { carrier := Set.preimage  …
  -/
  intro s hs
  /-
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    f : Homeomorph G G
    h : ∀ ⦃K : TopologicalSpace.Compacts G⦄, Eq ((fun s => ↑(μ.toFun s)) (Topologi …
    A s : Set G
    hs : IsOpen s
    ⊢ Eq (μ.innerContent { carrier := Set.preimage (⇑f.toEquiv) s, is_open' := ⋯ } …
  -/
  convert μ.innerContent_comap f h ⟨s, hs⟩
  /-
    🎉 no goals
  -/


theorem outerMeasure_lt_top_of_isCompact [WeaklyLocallyCompactSpace G]
    {K : Set G} (hK : IsCompact K) :
    μ.outerMeasure K < ∞ := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    inst✝ : WeaklyLocallyCompactSpace G
    K : Set G
    hK : IsCompact K
    ⊢ LT.lt (μ.outerMeasure K) Top.top
  -/
  rcases exists_compact_superset hK with ⟨F, h1F, h2F⟩
  calc
    μ.outerMeasure K ≤ μ.outerMeasure (interior F) := measure_mono h2F
    _ ≤ μ ⟨F, h1F⟩ := by
      apply μ.outerMeasure_le ⟨interior F, isOpen_interior⟩ ⟨F, h1F⟩ interior_subset
    _ < ⊤ := μ.lt_top _


@[to_additive]
theorem is_mul_left_invariant_outerMeasure [Group G] [TopologicalGroup G]
    (h : ∀ (g : G) {K : Compacts G}, μ (K.map _ <| continuous_mul_left g) = μ K) (g : G)
    (A : Set G) : μ.outerMeasure ((g * ·) ⁻¹' A) = μ.outerMeasure A := by
  /-
    G : Type w
    inst✝³ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝² : R1Space G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    h : ∀ (g : G) {K : TopologicalSpace.Compacts G}, Eq ((fun s => ↑(μ.toFun s)) ( …
    g : G
    A : Set G
    ⊢ Eq (μ.outerMeasure (Set.preimage (fun x => HMul.hMul g x) A)) (μ.outerMeasur …
  -/
  convert μ.outerMeasure_preimage (Homeomorph.mulLeft g) (fun K => h g) A
  /-
    🎉 no goals
  -/


theorem outerMeasure_caratheodory (A : Set G) :
    MeasurableSet[μ.outerMeasure.caratheodory] A ↔
      ∀ U : Opens G, μ.outerMeasure (U ∩ A) + μ.outerMeasure (U \ A) ≤ μ.outerMeasure U := by
  /-
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    A : Set G
    ⊢ Iff (MeasurableSet A) (∀ (U : TopologicalSpace.Opens G), LE.le (HAdd.hAdd (μ …
  -/
  rw [Opens.forall]
  /-
    G : Type w
    inst✝¹ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝ : R1Space G
    A : Set G
    ⊢ Iff (MeasurableSet A) (∀ (U : Set G) (hU : IsOpen U), LE.le (HAdd.hAdd (μ.ou …
  -/
  apply inducedOuterMeasure_caratheodory
    /-
      case msU
      G : Type w
      inst✝¹ : TopologicalSpace G
      μ : MeasureTheory.Content G
      inst✝ : R1Space G
      A : Set G
      ⊢ ∀ ⦃f : Nat → Set G⦄ (hm : ∀ (i : Nat), IsOpen (f i)), LE.le (μ.innerContent  …
    -/
  · apply innerContent_iUnion_nat
    /-
      🎉 no goals
    -/
    /-
      case m_mono
      G : Type w
      inst✝¹ : TopologicalSpace G
      μ : MeasureTheory.Content G
      inst✝ : R1Space G
      A : Set G
      ⊢ ∀ ⦃s₁ s₂ : Set G⦄ (hs₁ : IsOpen s₁) (hs₂ : IsOpen s₂), HasSubset.Subset s₁ s …
    -/
  · apply innerContent_mono'
    /-
      🎉 no goals
    -/


@[to_additive]
theorem outerMeasure_pos_of_is_mul_left_invariant [Group G] [TopologicalGroup G]
    (h3 : ∀ (g : G) {K : Compacts G}, μ (K.map _ <| continuous_mul_left g) = μ K) (K : Compacts G)
    (hK : μ K ≠ 0) {U : Set G} (h1U : IsOpen U) (h2U : U.Nonempty) : 0 < μ.outerMeasure U := by
  /-
    G : Type w
    inst✝³ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝² : R1Space G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    h3 : ∀ (g : G) {K : TopologicalSpace.Compacts G}, Eq ((fun s => ↑(μ.toFun s))  …
    K : TopologicalSpace.Compacts G
    hK : Ne ((fun s => ↑(μ.toFun s)) K) 0
    U : Set G
    h1U : IsOpen U
    h2U : U.Nonempty
    ⊢ LT.lt 0 (μ.outerMeasure U)
  -/
  convert μ.innerContent_pos_of_is_mul_left_invariant h3 K hK ⟨U, h1U⟩ h2U
  /-
    case h.e'_4
    G : Type w
    inst✝³ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝² : R1Space G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    h3 : ∀ (g : G) {K : TopologicalSpace.Compacts G}, Eq ((fun s => ↑(μ.toFun s))  …
    K : TopologicalSpace.Compacts G
    hK : Ne ((fun s => ↑(μ.toFun s)) K) 0
    U : Set G
    h1U : IsOpen U
    h2U : U.Nonempty
    ⊢ Eq (μ.outerMeasure U) (μ.innerContent { carrier := U, is_open' := h1U })
  -/
  exact μ.outerMeasure_opens ⟨U, h1U⟩
  /-
    🎉 no goals
  -/


/-- For the outer measure coming from a content, all Borel sets are measurable. -/
theorem borel_le_caratheodory : S ≤ μ.outerMeasure.caratheodory := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    ⊢ LE.le S μ.outerMeasure.caratheodory
  -/
  rw [BorelSpace.measurable_eq (α := G)]
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    ⊢ LE.le (borel G) μ.outerMeasure.caratheodory
  -/
  refine MeasurableSpace.generateFrom_le ?_
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    ⊢ ∀ (t : Set G), Membership.mem (setOf fun s => IsOpen s) t → MeasurableSet t
  -/
  intro U hU
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    ⊢ MeasurableSet U
  -/
  rw [μ.outerMeasure_caratheodory]
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    ⊢ ∀ (U_1 : TopologicalSpace.Opens G), LE.le (HAdd.hAdd (μ.outerMeasure (Inter. …
  -/
  intro U'
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    ⊢ LE.le (HAdd.hAdd (μ.outerMeasure (Inter.inter (↑U') U)) (μ.outerMeasure (SDi …
  -/
  rw [μ.outerMeasure_of_isOpen ((U' : Set G) ∩ U) (U'.isOpen.inter hU)]
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    ⊢ LE.le (HAdd.hAdd (μ.innerContent { carrier := Inter.inter (↑U') U, is_open'  …
  -/
  simp only [innerContent, iSup_subtype']
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    ⊢ LE.le (HAdd.hAdd (iSup fun x => ↑(μ.toFun ↑x)) (μ.outerMeasure (SDiff.sdiff  …
  -/
  rw [Opens.coe_mk]
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    ⊢ LE.le (HAdd.hAdd (iSup fun x => ↑(μ.toFun ↑x)) (μ.outerMeasure (SDiff.sdiff  …
  -/
  haveI : Nonempty { L : Compacts G // (L : Set G) ⊆ U' ∩ U } := ⟨⟨⊥, empty_subset _⟩⟩
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    ⊢ LE.le (HAdd.hAdd (iSup fun x => ↑(μ.toFun ↑x)) (μ.outerMeasure (SDiff.sdiff  …
  -/
  rw [ENNReal.iSup_add]
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    ⊢ LE.le (iSup fun i => HAdd.hAdd (↑(μ.toFun ↑i)) (μ.outerMeasure (SDiff.sdiff  …
  -/
  refine iSup_le ?_
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    ⊢ ∀ (i : Subtype fun i => HasSubset.Subset (↑i) (Inter.inter (↑U') U)), LE.le  …
  -/
  rintro ⟨L, hL⟩
  /-
    case mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    hL : HasSubset.Subset (↑L) (Inter.inter (↑U') U)
    ⊢ LE.le (HAdd.hAdd (↑(μ.toFun ↑⟨L, hL⟩)) (μ.outerMeasure (SDiff.sdiff (↑U') U) …
  -/
  let L' : Compacts G := ⟨closure L, L.isCompact.closure⟩
  suffices μ L' + μ.outerMeasure (↑U' \ U) ≤ μ.outerMeasure U' by
    have A : μ L ≤ μ L' := μ.mono _ _ subset_closure
    exact (add_le_add_right A _).trans this
  /-
    case mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    hL : HasSubset.Subset (↑L) (Inter.inter (↑U') U)
    L' : TopologicalSpace.Compacts G := { carrier := closure ↑L, isCompact' := ⋯ }
    ⊢ LE.le (HAdd.hAdd ((fun s => ↑(μ.toFun s)) L') (μ.outerMeasure (SDiff.sdiff ( …
  -/
  simp only [subset_inter_iff] at hL
  /-
    case mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    L' : TopologicalSpace.Compacts G := { carrier := closure ↑L, isCompact' := ⋯ }
    hL : And (HasSubset.Subset ↑L ↑U') (HasSubset.Subset (↑L) U)
    ⊢ LE.le (HAdd.hAdd ((fun s => ↑(μ.toFun s)) L') (μ.outerMeasure (SDiff.sdiff ( …
  -/
  have hL'U : (L' : Set G) ⊆ U := IsCompact.closure_subset_of_isOpen L.2 hU hL.2
  /-
    case mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    L' : TopologicalSpace.Compacts G := { carrier := closure ↑L, isCompact' := ⋯ }
    hL : And (HasSubset.Subset ↑L ↑U') (HasSubset.Subset (↑L) U)
    hL'U : HasSubset.Subset (↑L') U
    ⊢ LE.le (HAdd.hAdd ((fun s => ↑(μ.toFun s)) L') (μ.outerMeasure (SDiff.sdiff ( …
  -/
  have hL'U' : (L' : Set G) ⊆ (U' : Set G) := IsCompact.closure_subset_of_isOpen L.2 U'.2 hL.1
  /-
    case mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    L' : TopologicalSpace.Compacts G := { carrier := closure ↑L, isCompact' := ⋯ }
    hL : And (HasSubset.Subset ↑L ↑U') (HasSubset.Subset (↑L) U)
    hL'U : HasSubset.Subset (↑L') U
    hL'U' : HasSubset.Subset ↑L' ↑U'
    ⊢ LE.le (HAdd.hAdd ((fun s => ↑(μ.toFun s)) L') (μ.outerMeasure (SDiff.sdiff ( …
  -/
  have : ↑U' \ U ⊆ U' \ L' := diff_subset_diff_right hL'U
  /-
    case mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this✝ : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    L' : TopologicalSpace.Compacts G := { carrier := closure ↑L, isCompact' := ⋯ }
    hL : And (HasSubset.Subset ↑L ↑U') (HasSubset.Subset (↑L) U)
    hL'U : HasSubset.Subset (↑L') U
    hL'U' : HasSubset.Subset ↑L' ↑U'
    this : HasSubset.Subset (SDiff.sdiff (↑U') U) (SDiff.sdiff ↑U' ↑L')
    ⊢ LE.le (HAdd.hAdd ((fun s => ↑(μ.toFun s)) L') (μ.outerMeasure (SDiff.sdiff ( …
  -/
  refine le_trans (add_le_add_left (measure_mono this) _) ?_
  /-
    case mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this✝ : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    L' : TopologicalSpace.Compacts G := { carrier := closure ↑L, isCompact' := ⋯ }
    hL : And (HasSubset.Subset ↑L ↑U') (HasSubset.Subset (↑L) U)
    hL'U : HasSubset.Subset (↑L') U
    hL'U' : HasSubset.Subset ↑L' ↑U'
    this : HasSubset.Subset (SDiff.sdiff (↑U') U) (SDiff.sdiff ↑U' ↑L')
    ⊢ LE.le (HAdd.hAdd ((fun s => ↑(μ.toFun s)) L') (μ.outerMeasure (SDiff.sdiff ↑ …
  -/
  rw [μ.outerMeasure_of_isOpen (↑U' \ L') (IsOpen.sdiff U'.2 isClosed_closure)]
  /-
    case mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this✝ : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    L' : TopologicalSpace.Compacts G := { carrier := closure ↑L, isCompact' := ⋯ }
    hL : And (HasSubset.Subset ↑L ↑U') (HasSubset.Subset (↑L) U)
    hL'U : HasSubset.Subset (↑L') U
    hL'U' : HasSubset.Subset ↑L' ↑U'
    this : HasSubset.Subset (SDiff.sdiff (↑U') U) (SDiff.sdiff ↑U' ↑L')
    ⊢ LE.le (HAdd.hAdd ((fun s => ↑(μ.toFun s)) L') (μ.innerContent { carrier := S …
  -/
  simp only [innerContent, iSup_subtype']
  /-
    case mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this✝ : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    L' : TopologicalSpace.Compacts G := { carrier := closure ↑L, isCompact' := ⋯ }
    hL : And (HasSubset.Subset ↑L ↑U') (HasSubset.Subset (↑L) U)
    hL'U : HasSubset.Subset (↑L') U
    hL'U' : HasSubset.Subset ↑L' ↑U'
    this : HasSubset.Subset (SDiff.sdiff (↑U') U) (SDiff.sdiff ↑U' ↑L')
    ⊢ LE.le (HAdd.hAdd (↑(μ.toFun L')) (iSup fun x => ↑(μ.toFun ↑x))) (μ.outerMeas …
  -/
  rw [Opens.coe_mk]
  /-
    case mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this✝ : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    L' : TopologicalSpace.Compacts G := { carrier := closure ↑L, isCompact' := ⋯ }
    hL : And (HasSubset.Subset ↑L ↑U') (HasSubset.Subset (↑L) U)
    hL'U : HasSubset.Subset (↑L') U
    hL'U' : HasSubset.Subset ↑L' ↑U'
    this : HasSubset.Subset (SDiff.sdiff (↑U') U) (SDiff.sdiff ↑U' ↑L')
    ⊢ LE.le (HAdd.hAdd (↑(μ.toFun L')) (iSup fun x => ↑(μ.toFun ↑x))) (μ.outerMeas …
  -/
  haveI : Nonempty { M : Compacts G // (M : Set G) ⊆ ↑U' \ closure L } := ⟨⟨⊥, empty_subset _⟩⟩
  /-
    case mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this✝¹ : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    L' : TopologicalSpace.Compacts G := { carrier := closure ↑L, isCompact' := ⋯ }
    hL : And (HasSubset.Subset ↑L ↑U') (HasSubset.Subset (↑L) U)
    hL'U : HasSubset.Subset (↑L') U
    hL'U' : HasSubset.Subset ↑L' ↑U'
    this✝ : HasSubset.Subset (SDiff.sdiff (↑U') U) (SDiff.sdiff ↑U' ↑L')
    this : Nonempty (Subtype fun M => HasSubset.Subset (↑M) (SDiff.sdiff (↑U') (cl …
    ⊢ LE.le (HAdd.hAdd (↑(μ.toFun L')) (iSup fun x => ↑(μ.toFun ↑x))) (μ.outerMeas …
  -/
  rw [ENNReal.add_iSup]
  /-
    case mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this✝¹ : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    L' : TopologicalSpace.Compacts G := { carrier := closure ↑L, isCompact' := ⋯ }
    hL : And (HasSubset.Subset ↑L ↑U') (HasSubset.Subset (↑L) U)
    hL'U : HasSubset.Subset (↑L') U
    hL'U' : HasSubset.Subset ↑L' ↑U'
    this✝ : HasSubset.Subset (SDiff.sdiff (↑U') U) (SDiff.sdiff ↑U' ↑L')
    this : Nonempty (Subtype fun M => HasSubset.Subset (↑M) (SDiff.sdiff (↑U') (cl …
    ⊢ LE.le (iSup fun i => HAdd.hAdd ↑(μ.toFun L') ↑(μ.toFun ↑i)) (μ.outerMeasure  …
  -/
  refine iSup_le ?_
  /-
    case mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this✝¹ : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    L' : TopologicalSpace.Compacts G := { carrier := closure ↑L, isCompact' := ⋯ }
    hL : And (HasSubset.Subset ↑L ↑U') (HasSubset.Subset (↑L) U)
    hL'U : HasSubset.Subset (↑L') U
    hL'U' : HasSubset.Subset ↑L' ↑U'
    this✝ : HasSubset.Subset (SDiff.sdiff (↑U') U) (SDiff.sdiff ↑U' ↑L')
    this : Nonempty (Subtype fun M => HasSubset.Subset (↑M) (SDiff.sdiff (↑U') (cl …
    ⊢ ∀ (i : Subtype fun i => HasSubset.Subset (↑i) (SDiff.sdiff ↑U' ↑L')), LE.le  …
  -/
  rintro ⟨M, hM⟩
  /-
    case mk.mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this✝¹ : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    L' : TopologicalSpace.Compacts G := { carrier := closure ↑L, isCompact' := ⋯ }
    hL : And (HasSubset.Subset ↑L ↑U') (HasSubset.Subset (↑L) U)
    hL'U : HasSubset.Subset (↑L') U
    hL'U' : HasSubset.Subset ↑L' ↑U'
    this✝ : HasSubset.Subset (SDiff.sdiff (↑U') U) (SDiff.sdiff ↑U' ↑L')
    this : Nonempty (Subtype fun M => HasSubset.Subset (↑M) (SDiff.sdiff (↑U') (cl …
    M : TopologicalSpace.Compacts G
    hM : HasSubset.Subset (↑M) (SDiff.sdiff ↑U' ↑L')
    ⊢ LE.le (HAdd.hAdd ↑(μ.toFun L') ↑(μ.toFun ↑⟨M, hM⟩)) (μ.outerMeasure ↑U')
  -/
  let M' : Compacts G := ⟨closure M, M.isCompact.closure⟩
  suffices μ L' + μ M' ≤ μ.outerMeasure U' by
    have A : μ M ≤ μ M' := μ.mono _ _ subset_closure
    exact (add_le_add_left A _).trans this
  have hM' : (M' : Set G) ⊆ U' \ L' :=
    IsCompact.closure_subset_of_isOpen M.2 (IsOpen.sdiff U'.2 isClosed_closure) hM
  have : (↑(L' ⊔ M') : Set G) ⊆ U' := by
    simp only [Compacts.coe_sup, union_subset_iff, hL'U', true_and]
    exact hM'.trans diff_subset
  /-
    case mk.mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this✝² : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    L' : TopologicalSpace.Compacts G := { carrier := closure ↑L, isCompact' := ⋯ }
    hL : And (HasSubset.Subset ↑L ↑U') (HasSubset.Subset (↑L) U)
    hL'U : HasSubset.Subset (↑L') U
    hL'U' : HasSubset.Subset ↑L' ↑U'
    this✝¹ : HasSubset.Subset (SDiff.sdiff (↑U') U) (SDiff.sdiff ↑U' ↑L')
    this✝ : Nonempty (Subtype fun M => HasSubset.Subset (↑M) (SDiff.sdiff (↑U') (c …
    M : TopologicalSpace.Compacts G
    hM : HasSubset.Subset (↑M) (SDiff.sdiff ↑U' ↑L')
    M' : TopologicalSpace.Compacts G := { carrier := closure ↑M, isCompact' := ⋯ }
    hM' : HasSubset.Subset (↑M') (SDiff.sdiff ↑U' ↑L')
    this : HasSubset.Subset ↑(Max.max L' M') ↑U'
    ⊢ LE.le (HAdd.hAdd ((fun s => ↑(μ.toFun s)) L') ((fun s => ↑(μ.toFun s)) M'))  …
  -/
  rw [μ.outerMeasure_of_isOpen (↑U') U'.2]
  /-
    case mk.mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this✝² : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    L' : TopologicalSpace.Compacts G := { carrier := closure ↑L, isCompact' := ⋯ }
    hL : And (HasSubset.Subset ↑L ↑U') (HasSubset.Subset (↑L) U)
    hL'U : HasSubset.Subset (↑L') U
    hL'U' : HasSubset.Subset ↑L' ↑U'
    this✝¹ : HasSubset.Subset (SDiff.sdiff (↑U') U) (SDiff.sdiff ↑U' ↑L')
    this✝ : Nonempty (Subtype fun M => HasSubset.Subset (↑M) (SDiff.sdiff (↑U') (c …
    M : TopologicalSpace.Compacts G
    hM : HasSubset.Subset (↑M) (SDiff.sdiff ↑U' ↑L')
    M' : TopologicalSpace.Compacts G := { carrier := closure ↑M, isCompact' := ⋯ }
    hM' : HasSubset.Subset (↑M') (SDiff.sdiff ↑U' ↑L')
    this : HasSubset.Subset ↑(Max.max L' M') ↑U'
    ⊢ LE.le (HAdd.hAdd ((fun s => ↑(μ.toFun s)) L') ((fun s => ↑(μ.toFun s)) M'))  …
  -/
  refine le_trans (ge_of_eq ?_) (μ.le_innerContent _ _ this)
  /-
    case mk.mk
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    U : Set G
    hU : Membership.mem (setOf fun s => IsOpen s) U
    U' : TopologicalSpace.Opens G
    this✝² : Nonempty (Subtype fun L => HasSubset.Subset (↑L) (Inter.inter (↑U') U))
    L : TopologicalSpace.Compacts G
    L' : TopologicalSpace.Compacts G := { carrier := closure ↑L, isCompact' := ⋯ }
    hL : And (HasSubset.Subset ↑L ↑U') (HasSubset.Subset (↑L) U)
    hL'U : HasSubset.Subset (↑L') U
    hL'U' : HasSubset.Subset ↑L' ↑U'
    this✝¹ : HasSubset.Subset (SDiff.sdiff (↑U') U) (SDiff.sdiff ↑U' ↑L')
    this✝ : Nonempty (Subtype fun M => HasSubset.Subset (↑M) (SDiff.sdiff (↑U') (c …
    M : TopologicalSpace.Compacts G
    hM : HasSubset.Subset (↑M) (SDiff.sdiff ↑U' ↑L')
    M' : TopologicalSpace.Compacts G := { carrier := closure ↑M, isCompact' := ⋯ }
    hM' : HasSubset.Subset (↑M') (SDiff.sdiff ↑U' ↑L')
    this : HasSubset.Subset ↑(Max.max L' M') ↑U'
    ⊢ Eq ((fun s => ↑(μ.toFun s)) (Max.max L' M')) (HAdd.hAdd ((fun s => ↑(μ.toFun …
  -/
  exact μ.sup_disjoint L' M' (subset_diff.1 hM').2.symm isClosed_closure isClosed_closure
  /-
    🎉 no goals
  -/


/-- The measure induced by the outer measure coming from a content, on the Borel sigma-algebra. -/
protected def measure : Measure G :=
  μ.outerMeasure.toMeasure μ.borel_le_caratheodory


theorem measure_apply {s : Set G} (hs : MeasurableSet s) : μ.measure s = μ.outerMeasure s :=
  toMeasure_apply _ _ hs


instance outerRegular : μ.measure.OuterRegular := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    ⊢ μ.measure.OuterRegular
  -/
  refine ⟨fun A hA r (hr : _ < _) ↦ ?_⟩
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    A : Set G
    hA : MeasurableSet A
    r : ENNReal
    hr : LT.lt (μ.measure A) r
    ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LT.lt (μ.measure U) r))
  -/
  rw [μ.measure_apply hA, outerMeasure_eq_iInf] at hr
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    A : Set G
    hA : MeasurableSet A
    r : ENNReal
    hr : LT.lt (iInf fun U => iInf fun hU => iInf fun x => μ.innerContent { carrie …
    ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LT.lt (μ.measure U) r))
  -/
  simp only [iInf_lt_iff] at hr
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    A : Set G
    hA : MeasurableSet A
    r : ENNReal
    hr : Exists fun i => Exists fun h => Exists fun i_1 => LT.lt (μ.innerContent { …
    ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LT.lt (μ.measure U) r))
  -/
  rcases hr with ⟨U, hUo, hAU, hr⟩
  /-
    case intro.intro.intro
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    A : Set G
    hA : MeasurableSet A
    r : ENNReal
    U : Set G
    hUo : IsOpen U
    hAU : HasSubset.Subset A U
    hr : LT.lt (μ.innerContent { carrier := U, is_open' := ⋯ }) r
    ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LT.lt (μ.measure U) r))
  -/
  rw [← μ.outerMeasure_of_isOpen U hUo, ← μ.measure_apply hUo.measurableSet] at hr
  /-
    case intro.intro.intro
    G : Type w
    inst✝² : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝¹ : R1Space G
    S : MeasurableSpace G
    inst✝ : BorelSpace G
    A : Set G
    hA : MeasurableSet A
    r : ENNReal
    U : Set G
    hUo : IsOpen U
    hAU : HasSubset.Subset A U
    hr : LT.lt (μ.measure U) r
    ⊢ Exists fun U => And (Superset U A) (And (IsOpen U) (LT.lt (μ.measure U) r))
  -/
  exact ⟨U, hAU, hUo, hr⟩
  /-
    🎉 no goals
  -/


/-- In a locally compact space, any measure constructed from a content is regular. -/
instance regular [WeaklyLocallyCompactSpace G] : μ.measure.Regular := by
  have : IsFiniteMeasureOnCompacts μ.measure := by
    refine ⟨fun K hK => ?_⟩
    apply (measure_mono subset_closure).trans_lt _
    rw [measure_apply _ isClosed_closure.measurableSet]
    exact μ.outerMeasure_lt_top_of_isCompact hK.closure
  /-
    G : Type w
    inst✝³ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝² : R1Space G
    S : MeasurableSpace G
    inst✝¹ : BorelSpace G
    inst✝ : WeaklyLocallyCompactSpace G
    this : MeasureTheory.IsFiniteMeasureOnCompacts μ.measure
    ⊢ μ.measure.Regular
  -/
  refine ⟨fun U hU r hr => ?_⟩
  /-
    G : Type w
    inst✝³ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝² : R1Space G
    S : MeasurableSpace G
    inst✝¹ : BorelSpace G
    inst✝ : WeaklyLocallyCompactSpace G
    this : MeasureTheory.IsFiniteMeasureOnCompacts μ.measure
    U : Set G
    hU : IsOpen U
    r : ENNReal
    hr : LT.lt r (μ.measure U)
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (IsCompact K) (LT.lt r (μ.me …
  -/
  rw [measure_apply _ hU.measurableSet, μ.outerMeasure_of_isOpen U hU] at hr
  /-
    G : Type w
    inst✝³ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝² : R1Space G
    S : MeasurableSpace G
    inst✝¹ : BorelSpace G
    inst✝ : WeaklyLocallyCompactSpace G
    this : MeasureTheory.IsFiniteMeasureOnCompacts μ.measure
    U : Set G
    hU : IsOpen U
    r : ENNReal
    hr : LT.lt r (μ.innerContent { carrier := U, is_open' := hU })
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (IsCompact K) (LT.lt r (μ.me …
  -/
  simp only [innerContent, lt_iSup_iff] at hr
  /-
    G : Type w
    inst✝³ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝² : R1Space G
    S : MeasurableSpace G
    inst✝¹ : BorelSpace G
    inst✝ : WeaklyLocallyCompactSpace G
    this : MeasureTheory.IsFiniteMeasureOnCompacts μ.measure
    U : Set G
    hU : IsOpen U
    r : ENNReal
    hr : Exists fun i => Exists fun i_1 => LT.lt r ↑(μ.toFun i)
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (IsCompact K) (LT.lt r (μ.me …
  -/
  rcases hr with ⟨K, hKU, hr⟩
  /-
    case intro.intro
    G : Type w
    inst✝³ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝² : R1Space G
    S : MeasurableSpace G
    inst✝¹ : BorelSpace G
    inst✝ : WeaklyLocallyCompactSpace G
    this : MeasureTheory.IsFiniteMeasureOnCompacts μ.measure
    U : Set G
    hU : IsOpen U
    r : ENNReal
    K : TopologicalSpace.Compacts G
    hKU : HasSubset.Subset ↑K ↑{ carrier := U, is_open' := hU }
    hr : LT.lt r ↑(μ.toFun K)
    ⊢ Exists fun K => And (HasSubset.Subset K U) (And (IsCompact K) (LT.lt r (μ.me …
  -/
  refine ⟨K, hKU, K.2, hr.trans_le ?_⟩
  /-
    case intro.intro
    G : Type w
    inst✝³ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝² : R1Space G
    S : MeasurableSpace G
    inst✝¹ : BorelSpace G
    inst✝ : WeaklyLocallyCompactSpace G
    this : MeasureTheory.IsFiniteMeasureOnCompacts μ.measure
    U : Set G
    hU : IsOpen U
    r : ENNReal
    K : TopologicalSpace.Compacts G
    hKU : HasSubset.Subset ↑K ↑{ carrier := U, is_open' := hU }
    hr : LT.lt r ↑(μ.toFun K)
    ⊢ LE.le (↑(μ.toFun K)) (μ.measure ↑K)
  -/
  exact (μ.le_outerMeasure_compacts K).trans (le_toMeasure_apply _ _ _)
  /-
    🎉 no goals
  -/


/-- A content `μ` is called regular if for every compact set `K`,
  `μ(K) = inf {μ(K') : K ⊂ int K' ⊂ K'}`. See Paul Halmos (1950), Measure Theory, §54-/
def ContentRegular :=
  ∀ ⦃K : TopologicalSpace.Compacts G⦄,
    μ K = ⨅ (K' : TopologicalSpace.Compacts G) (_ : (K : Set G) ⊆ interior (K' : Set G)), μ K'


theorem contentRegular_exists_compact (H : ContentRegular μ) (K : TopologicalSpace.Compacts G)
    {ε : NNReal} (hε : ε ≠ 0) :
    ∃ K' : TopologicalSpace.Compacts G, K.carrier ⊆ interior K'.carrier ∧ μ K' ≤ μ K + ε := by
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    H : μ.ContentRegular
    K : TopologicalSpace.Compacts G
    ε : NNReal
    hε : Ne ε 0
    ⊢ Exists fun K' => And (HasSubset.Subset K.carrier (interior K'.carrier)) (LE. …
  -/
  by_contra hc
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    H : μ.ContentRegular
    K : TopologicalSpace.Compacts G
    ε : NNReal
    hε : Ne ε 0
    hc : Not (Exists fun K' => And (HasSubset.Subset K.carrier (interior K'.carrie …
    ⊢ False
  -/
  simp only [not_exists, not_and, not_le] at hc
  have lower_bound_iInf : μ K + ε ≤
      ⨅ (K' : TopologicalSpace.Compacts G) (_ : (K : Set G) ⊆ interior (K' : Set G)), μ K' :=
    le_iInf fun K' => le_iInf fun K'_hyp => le_of_lt (hc K' K'_hyp)
  /-
    G : Type w
    inst✝ : TopologicalSpace G
    μ : MeasureTheory.Content G
    H : μ.ContentRegular
    K : TopologicalSpace.Compacts G
    ε : NNReal
    hε : Ne ε 0
    hc : ∀ (x : TopologicalSpace.Compacts G), HasSubset.Subset K.carrier (interior …
    lower_bound_iInf : LE.le (HAdd.hAdd ((fun s => ↑(μ.toFun s)) K) ↑ε) (iInf fun  …
    ⊢ False
  -/
  rw [← H] at lower_bound_iInf
  exact (lt_self_iff_false (μ K)).mp (lt_of_le_of_lt' lower_bound_iInf
    (ENNReal.lt_add_right (ne_top_of_lt (μ.lt_top K)) (ENNReal.coe_ne_zero.mpr hε)))


/-- If `μ` is a regular content, then the measure induced by `μ` will agree with `μ`
  on compact sets. -/
theorem measure_eq_content_of_regular (H : MeasureTheory.Content.ContentRegular μ)
    (K : TopologicalSpace.Compacts G) : μ.measure ↑K = μ K := by
  /-
    G : Type w
    inst✝³ : TopologicalSpace G
    μ : MeasureTheory.Content G
    inst✝² : MeasurableSpace G
    inst✝¹ : R1Space G
    inst✝ : BorelSpace G
    H : μ.ContentRegular
    K : TopologicalSpace.Compacts G
    ⊢ Eq (μ.measure ↑K) ((fun s => ↑(μ.toFun s)) K)
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      G : Type w
      inst✝³ : TopologicalSpace G
      μ : MeasureTheory.Content G
      inst✝² : MeasurableSpace G
      inst✝¹ : R1Space G
      inst✝ : BorelSpace G
      H : μ.ContentRegular
      K : TopologicalSpace.Compacts G
      ⊢ LE.le (μ.measure ↑K) ((fun s => ↑(μ.toFun s)) K)
    -/
  · apply ENNReal.le_of_forall_pos_le_add
    /-
      case refine_1.h
      G : Type w
      inst✝³ : TopologicalSpace G
      μ : MeasureTheory.Content G
      inst✝² : MeasurableSpace G
      inst✝¹ : R1Space G
      inst✝ : BorelSpace G
      H : μ.ContentRegular
      K : TopologicalSpace.Compacts G
      ⊢ ∀ (ε : NNReal), LT.lt 0 ε → LT.lt ((fun s => ↑(μ.toFun s)) K) Top.top → LE.l …
    -/
    intro ε εpos _
    /-
      case refine_1.h
      G : Type w
      inst✝³ : TopologicalSpace G
      μ : MeasureTheory.Content G
      inst✝² : MeasurableSpace G
      inst✝¹ : R1Space G
      inst✝ : BorelSpace G
      H : μ.ContentRegular
      K : TopologicalSpace.Compacts G
      ε : NNReal
      εpos : LT.lt 0 ε
      a✝ : LT.lt ((fun s => ↑(μ.toFun s)) K) Top.top
      ⊢ LE.le (μ.measure ↑K) (HAdd.hAdd ((fun s => ↑(μ.toFun s)) K) ↑ε)
    -/
    obtain ⟨K', K'_hyp⟩ := contentRegular_exists_compact μ H K (ne_bot_of_gt εpos)
    calc
      μ.measure ↑K ≤ μ.measure (interior ↑K') := measure_mono K'_hyp.1
      _ ≤ μ K' := by
        rw [μ.measure_apply (IsOpen.measurableSet isOpen_interior)]
        exact μ.outerMeasure_interior_compacts K'
      _ ≤ μ K + ε := K'_hyp.right
  · calc
    μ K ≤ μ ⟨closure K, K.2.closure⟩ := μ.mono _ _ subset_closure
    _ ≤ μ.measure (closure K) := by
      rw [μ.measure_apply (isClosed_closure.measurableSet)]
      exact μ.le_outerMeasure_compacts _
    _ = μ.measure K := K.2.measure_closure _


