/-- If the distance between consecutive points of a sequence is estimated by a summable series,
then the original sequence is a Cauchy sequence. -/
theorem cauchySeq_of_dist_le_of_summable (d : ℕ → ℝ) (hf : ∀ n, dist (f n) (f n.succ) ≤ d n)
    (hd : Summable d) : CauchySeq f := by
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    f : Nat → α
    d : Nat → Real
    hf : ∀ (n : Nat), LE.le (Dist.dist (f n) (f n.succ)) (d n)
    hd : Summable d
    ⊢ CauchySeq f
  -/
  lift d to ℕ → ℝ≥0 using fun n ↦ dist_nonneg.trans (hf n)
  /-
    case intro
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    f : Nat → α
    d : Nat → NNReal
    hf : ∀ (n : Nat), LE.le (Dist.dist (f n) (f n.succ)) ((fun i => ↑(d i)) n)
    hd : Summable fun i => ↑(d i)
    ⊢ CauchySeq f
  -/
  apply cauchySeq_of_edist_le_of_summable d (α := α) (f := f)
    /-
      case intro.hf
      α : Type u_1
      inst✝ : PseudoMetricSpace α
      f : Nat → α
      d : Nat → NNReal
      hf : ∀ (n : Nat), LE.le (Dist.dist (f n) (f n.succ)) ((fun i => ↑(d i)) n)
      hd : Summable fun i => ↑(d i)
      ⊢ ∀ (n : Nat), LE.le (EDist.edist (f n) (f n.succ)) ↑(d n)
    -/
  · exact_mod_cast hf
    /-
      🎉 no goals
    -/
    /-
      case intro.hd
      α : Type u_1
      inst✝ : PseudoMetricSpace α
      f : Nat → α
      d : Nat → NNReal
      hf : ∀ (n : Nat), LE.le (Dist.dist (f n) (f n.succ)) ((fun i => ↑(d i)) n)
      hd : Summable fun i => ↑(d i)
      ⊢ Summable d
    -/
  · exact_mod_cast hd
    /-
      🎉 no goals
    -/


theorem cauchySeq_of_summable_dist (h : Summable fun n ↦ dist (f n) (f n.succ)) : CauchySeq f :=
  cauchySeq_of_dist_le_of_summable _ (fun _ ↦ le_rfl) h


theorem dist_le_tsum_of_dist_le_of_tendsto (d : ℕ → ℝ) (hf : ∀ n, dist (f n) (f n.succ) ≤ d n)
    (hd : Summable d) {a : α} (ha : Tendsto f atTop (𝓝 a)) (n : ℕ) :
    dist (f n) a ≤ ∑' m, d (n + m) := by
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    f : Nat → α
    d : Nat → Real
    hf : ∀ (n : Nat), LE.le (Dist.dist (f n) (f n.succ)) (d n)
    hd : Summable d
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    ⊢ LE.le (Dist.dist (f n) a) (tsum fun m => d (HAdd.hAdd n m))
  -/
  refine le_of_tendsto (tendsto_const_nhds.dist ha) (eventually_atTop.2 ⟨n, fun m hnm ↦ ?_⟩)
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    f : Nat → α
    d : Nat → Real
    hf : ∀ (n : Nat), LE.le (Dist.dist (f n) (f n.succ)) (d n)
    hd : Summable d
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n m : Nat
    hnm : GE.ge m n
    ⊢ LE.le (Dist.dist (f n) (f m)) (tsum fun m => d (HAdd.hAdd n m))
  -/
  refine le_trans (dist_le_Ico_sum_of_dist_le hnm fun _ _ ↦ hf _) ?_
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    f : Nat → α
    d : Nat → Real
    hf : ∀ (n : Nat), LE.le (Dist.dist (f n) (f n.succ)) (d n)
    hd : Summable d
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n m : Nat
    hnm : GE.ge m n
    ⊢ LE.le ((Finset.Ico n m).sum fun i => d i) (tsum fun m => d (HAdd.hAdd n m))
  -/
  rw [sum_Ico_eq_sum_range]
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    f : Nat → α
    d : Nat → Real
    hf : ∀ (n : Nat), LE.le (Dist.dist (f n) (f n.succ)) (d n)
    hd : Summable d
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n m : Nat
    hnm : GE.ge m n
    ⊢ LE.le ((Finset.range (HSub.hSub m n)).sum fun k => d (HAdd.hAdd n k)) (tsum  …
  -/
  refine sum_le_tsum (range _) (fun _ _ ↦ le_trans dist_nonneg (hf _)) ?_
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    f : Nat → α
    d : Nat → Real
    hf : ∀ (n : Nat), LE.le (Dist.dist (f n) (f n.succ)) (d n)
    hd : Summable d
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n m : Nat
    hnm : GE.ge m n
    ⊢ Summable fun k => d (HAdd.hAdd n k)
  -/
  exact hd.comp_injective (add_right_injective n)
  /-
    🎉 no goals
  -/


theorem dist_le_tsum_of_dist_le_of_tendsto₀ (d : ℕ → ℝ) (hf : ∀ n, dist (f n) (f n.succ) ≤ d n)
    (hd : Summable d) (ha : Tendsto f atTop (𝓝 a)) : dist (f 0) a ≤ tsum d := by
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    f : Nat → α
    a : α
    d : Nat → Real
    hf : ∀ (n : Nat), LE.le (Dist.dist (f n) (f n.succ)) (d n)
    hd : Summable d
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    ⊢ LE.le (Dist.dist (f 0) a) (tsum d)
  -/
  simpa only [zero_add] using dist_le_tsum_of_dist_le_of_tendsto d hf hd ha 0
  /-
    🎉 no goals
  -/


theorem dist_le_tsum_dist_of_tendsto (h : Summable fun n ↦ dist (f n) (f n.succ))
    (ha : Tendsto f atTop (𝓝 a)) (n) : dist (f n) a ≤ ∑' m, dist (f (n + m)) (f (n + m).succ) :=
  show dist (f n) a ≤ ∑' m, (fun x ↦ dist (f x) (f x.succ)) (n + m) from
    dist_le_tsum_of_dist_le_of_tendsto (fun n ↦ dist (f n) (f n.succ)) (fun _ ↦ le_rfl) h ha n


theorem dist_le_tsum_dist_of_tendsto₀ (h : Summable fun n ↦ dist (f n) (f n.succ))
    (ha : Tendsto f atTop (𝓝 a)) : dist (f 0) a ≤ ∑' n, dist (f n) (f n.succ) := by
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    f : Nat → α
    a : α
    h : Summable fun n => Dist.dist (f n) (f n.succ)
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    ⊢ LE.le (Dist.dist (f 0) a) (tsum fun n => Dist.dist (f n) (f n.succ))
  -/
  simpa only [zero_add] using dist_le_tsum_dist_of_tendsto h ha 0
  /-
    🎉 no goals
  -/


theorem not_summable_iff_tendsto_nat_atTop_of_nonneg {f : ℕ → ℝ} (hf : ∀ n, 0 ≤ f n) :
    ¬Summable f ↔ Tendsto (fun n : ℕ => ∑ i ∈ Finset.range n, f i) atTop atTop := by
  /-
    f : Nat → Real
    hf : ∀ (n : Nat), LE.le 0 (f n)
    ⊢ Iff (Not (Summable f)) (Filter.Tendsto (fun n => (Finset.range n).sum fun i  …
  -/
  lift f to ℕ → ℝ≥0 using hf
  /-
    case intro
    f : Nat → NNReal
    ⊢ Iff (Not (Summable fun i => ↑(f i))) (Filter.Tendsto (fun n => (Finset.range …
  -/
  simpa using mod_cast NNReal.not_summable_iff_tendsto_nat_atTop
  /-
    🎉 no goals
  -/


theorem summable_iff_not_tendsto_nat_atTop_of_nonneg {f : ℕ → ℝ} (hf : ∀ n, 0 ≤ f n) :
    Summable f ↔ ¬Tendsto (fun n : ℕ => ∑ i ∈ Finset.range n, f i) atTop atTop := by
  /-
    f : Nat → Real
    hf : ∀ (n : Nat), LE.le 0 (f n)
    ⊢ Iff (Summable f) (Not (Filter.Tendsto (fun n => (Finset.range n).sum fun i = …
  -/
  rw [← not_iff_not, Classical.not_not, not_summable_iff_tendsto_nat_atTop_of_nonneg hf]
  /-
    🎉 no goals
  -/


theorem summable_sigma_of_nonneg {α} {β : α → Type*} {f : (Σ x, β x) → ℝ} (hf : ∀ x, 0 ≤ f x) :
    Summable f ↔ (∀ x, Summable fun y => f ⟨x, y⟩) ∧ Summable fun x => ∑' y, f ⟨x, y⟩ := by
  /-
    α : Type u_4
    β : α → Type u_3
    f : (Sigma fun x => β x) → Real
    hf : ∀ (x : Sigma fun x => β x), LE.le 0 (f x)
    ⊢ Iff (Summable f) (And (∀ (x : α), Summable fun y => f ⟨x, y⟩) (Summable fun  …
  -/
  lift f to (Σx, β x) → ℝ≥0 using hf
  /-
    case intro
    α : Type u_4
    β : α → Type u_3
    f : (Sigma fun x => β x) → NNReal
    ⊢ Iff (Summable fun i => ↑(f i)) (And (∀ (x : α), Summable fun y => (fun i =>  …
  -/
  simpa using mod_cast NNReal.summable_sigma
  /-
    🎉 no goals
  -/


lemma summable_partition {α β : Type*} {f : β → ℝ} (hf : 0 ≤ f) {s : α  → Set β}
    (hs : ∀ i, ∃! j, i ∈ s j) : Summable f ↔
      (∀ j, Summable fun i : s j ↦ f i) ∧ Summable fun j ↦ ∑' i : s j, f i := by
  /-
    α : Type u_3
    β : Type u_4
    f : β → Real
    hf : LE.le 0 f
    s : α → Set β
    hs : ∀ (i : β), ExistsUnique fun j => Membership.mem (s j) i
    ⊢ Iff (Summable f) (And (∀ (j : α), Summable fun i => f ↑i) (Summable fun j => …
  -/
  simpa only [← (Set.sigmaEquiv s hs).summable_iff] using summable_sigma_of_nonneg (fun _ ↦ hf _)
  /-
    🎉 no goals
  -/


theorem summable_prod_of_nonneg {α β} {f : (α × β) → ℝ} (hf : 0 ≤ f) :
    Summable f ↔ (∀ x, Summable fun y ↦ f (x, y)) ∧ Summable fun x ↦ ∑' y, f (x, y) :=
  (Equiv.sigmaEquivProd _ _).summable_iff.symm.trans <| summable_sigma_of_nonneg fun _ ↦ hf _


theorem summable_of_sum_le {ι : Type*} {f : ι → ℝ} {c : ℝ} (hf : 0 ≤ f)
    (h : ∀ u : Finset ι, ∑ x ∈ u, f x ≤ c) : Summable f :=
  ⟨⨆ u : Finset ι, ∑ x ∈ u, f x,
    tendsto_atTop_ciSup (Finset.sum_mono_set_of_nonneg hf) ⟨c, fun _ ⟨u, hu⟩ => hu ▸ h u⟩⟩


theorem summable_of_sum_range_le {f : ℕ → ℝ} {c : ℝ} (hf : ∀ n, 0 ≤ f n)
    (h : ∀ n, ∑ i ∈ Finset.range n, f i ≤ c) : Summable f := by
  /-
    f : Nat → Real
    c : Real
    hf : ∀ (n : Nat), LE.le 0 (f n)
    h : ∀ (n : Nat), LE.le ((Finset.range n).sum fun i => f i) c
    ⊢ Summable f
  -/
  refine (summable_iff_not_tendsto_nat_atTop_of_nonneg hf).2 fun H => ?_
  /-
    f : Nat → Real
    c : Real
    hf : ∀ (n : Nat), LE.le 0 (f n)
    h : ∀ (n : Nat), LE.le ((Finset.range n).sum fun i => f i) c
    H : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop F …
    ⊢ False
  -/
  rcases exists_lt_of_tendsto_atTop H 0 c with ⟨n, -, hn⟩
  /-
    case intro.intro
    f : Nat → Real
    c : Real
    hf : ∀ (n : Nat), LE.le 0 (f n)
    h : ∀ (n : Nat), LE.le ((Finset.range n).sum fun i => f i) c
    H : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop F …
    n : Nat
    hn : LT.lt c ((Finset.range n).sum fun i => f i)
    ⊢ False
  -/
  exact lt_irrefl _ (hn.trans_le (h n))
  /-
    🎉 no goals
  -/


theorem Real.tsum_le_of_sum_range_le {f : ℕ → ℝ} {c : ℝ} (hf : ∀ n, 0 ≤ f n)
    (h : ∀ n, ∑ i ∈ Finset.range n, f i ≤ c) : ∑' n, f n ≤ c :=
  _root_.tsum_le_of_sum_range_le (summable_of_sum_range_le hf h) h


/-- If a sequence `f` with non-negative terms is dominated by a sequence `g` with summable
series and at least one term of `f` is strictly smaller than the corresponding term in `g`,
then the series of `f` is strictly smaller than the series of `g`. -/
theorem tsum_lt_tsum_of_nonneg {i : ℕ} {f g : ℕ → ℝ} (h0 : ∀ b : ℕ, 0 ≤ f b)
    (h : ∀ b : ℕ, f b ≤ g b) (hi : f i < g i) (hg : Summable g) : ∑' n, f n < ∑' n, g n :=
  tsum_lt_tsum h hi (.of_nonneg_of_le h0 h hg) hg


