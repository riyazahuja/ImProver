instance : TopologicalSpace ℝ≥0 := inferInstance

-- short-circuit type class inference

instance : TopologicalSemiring ℝ≥0 where
  toContinuousAdd := continuousAdd_induced toRealHom
  toContinuousMul := continuousMul_induced toRealHom


instance : SecondCountableTopology ℝ≥0 :=
  inferInstanceAs (SecondCountableTopology { x : ℝ | 0 ≤ x })


instance : OrderTopology ℝ≥0 :=
  orderTopology_of_ordConnected (t := Ici 0)


instance : CompleteSpace ℝ≥0 :=
  isClosed_Ici.completeSpace_coe


instance : ContinuousStar ℝ≥0 where
  continuous_star := continuous_id

-- TODO: generalize this to a broader class of subtypes

instance : IsOrderBornology ℝ≥0 where
  isBounded_iff_bddBelow_bddAbove s := by
    /-
      s : Set NNReal
      ⊢ Iff (Bornology.IsBounded s) (And (BddBelow s) (BddAbove s))
    -/
    refine ⟨fun bdd ↦ ?_, fun h ↦ isBounded_of_bddAbove_of_bddBelow h.2 h.1⟩
    obtain ⟨r, hr⟩ : ∃ r : ℝ≥0, s ⊆ Icc 0 r := by
      obtain ⟨rreal, hrreal⟩ := bdd.subset_closedBall 0
      use rreal.toNNReal
      simp only [← NNReal.closedBall_zero_eq_Icc', Real.coe_toNNReal']
      exact subset_trans hrreal (Metric.closedBall_subset_closedBall (le_max_left rreal 0))
    /-
      case intro
      s : Set NNReal
      bdd : Bornology.IsBounded s
      r : NNReal
      hr : HasSubset.Subset s (Set.Icc 0 r)
      ⊢ And (BddBelow s) (BddAbove s)
    -/
    exact ⟨bddBelow_Icc.mono hr, bddAbove_Icc.mono hr⟩
    /-
      🎉 no goals
    -/


lemma isOpen_Ico_zero {x : NNReal} : IsOpen (Set.Ico 0 x) :=
  Ico_bot (a := x) ▸ isOpen_Iio


theorem _root_.continuous_real_toNNReal : Continuous Real.toNNReal :=
  (continuous_id.max continuous_const).subtype_mk _


/-- `Real.toNNReal` bundled as a continuous map for convenience. -/
@[simps (config := .asFn)]
noncomputable def _root_.ContinuousMap.realToNNReal : C(ℝ, ℝ≥0) :=
  .mk Real.toNNReal continuous_real_toNNReal


theorem continuous_coe : Continuous ((↑) : ℝ≥0 → ℝ) :=
  continuous_subtype_val


lemma _root_.ContinuousOn.ofReal_map_toNNReal {f : ℝ≥0 → ℝ≥0} {s : Set ℝ} {t : Set ℝ≥0}
    (hf : ContinuousOn f t) (h : Set.MapsTo Real.toNNReal s t) :
    ContinuousOn (fun x ↦ f x.toNNReal : ℝ → ℝ) s :=
  continuous_subtype_val.comp_continuousOn <| hf.comp continuous_real_toNNReal.continuousOn h


/-- Embedding of `ℝ≥0` to `ℝ` as a bundled continuous map. -/
@[simps (config := .asFn)]
def _root_.ContinuousMap.coeNNRealReal : C(ℝ≥0, ℝ) :=
  ⟨(↑), continuous_coe⟩


instance ContinuousMap.canLift {X : Type*} [TopologicalSpace X] :
    CanLift C(X, ℝ) C(X, ℝ≥0) ContinuousMap.coeNNRealReal.comp fun f => ∀ x, 0 ≤ f x where
  prf f hf := ⟨⟨fun x => ⟨f x, hf x⟩, f.2.subtype_mk _⟩, DFunLike.ext' rfl⟩


@[simp, norm_cast]
theorem tendsto_coe {f : Filter α} {m : α → ℝ≥0} {x : ℝ≥0} :
    Tendsto (fun a => (m a : ℝ)) f (𝓝 (x : ℝ)) ↔ Tendsto m f (𝓝 x) :=
  tendsto_subtype_rng.symm


theorem tendsto_coe' {f : Filter α} [NeBot f] {m : α → ℝ≥0} {x : ℝ} :
    Tendsto (fun a => m a : α → ℝ) f (𝓝 x) ↔ ∃ hx : 0 ≤ x, Tendsto m f (𝓝 ⟨x, hx⟩) :=
  ⟨fun h => ⟨ge_of_tendsto' h fun c => (m c).2, tendsto_coe.1 h⟩, fun ⟨_, hm⟩ => tendsto_coe.2 hm⟩


@[simp] theorem map_coe_atTop : map toReal atTop = atTop := map_val_Ici_atTop 0


@[simp]
theorem comap_coe_atTop : comap toReal atTop = atTop := (atTop_Ici_eq 0).symm


@[simp, norm_cast]
theorem tendsto_coe_atTop {f : Filter α} {m : α → ℝ≥0} :
    Tendsto (fun a => (m a : ℝ)) f atTop ↔ Tendsto m f atTop :=
  tendsto_Ici_atTop.symm


theorem _root_.tendsto_real_toNNReal {f : Filter α} {m : α → ℝ} {x : ℝ} (h : Tendsto m f (𝓝 x)) :
    Tendsto (fun a => Real.toNNReal (m a)) f (𝓝 (Real.toNNReal x)) :=
  (continuous_real_toNNReal.tendsto _).comp h


@[simp]
theorem _root_.Real.map_toNNReal_atTop : map Real.toNNReal atTop = atTop := by
  /-
    ⊢ Eq (Filter.map Real.toNNReal Filter.atTop) Filter.atTop
  -/
  rw [← map_coe_atTop, Function.LeftInverse.filter_map @Real.toNNReal_coe]
  /-
    🎉 no goals
  -/


theorem _root_.tendsto_real_toNNReal_atTop : Tendsto Real.toNNReal atTop atTop :=
  Real.map_toNNReal_atTop.le


@[simp]
theorem _root_.Real.comap_toNNReal_atTop : comap Real.toNNReal atTop = atTop := by
  /-
    ⊢ Eq (Filter.comap Real.toNNReal Filter.atTop) Filter.atTop
  -/
  refine le_antisymm ?_ tendsto_real_toNNReal_atTop.le_comap
  /-
    ⊢ LE.le (Filter.comap Real.toNNReal Filter.atTop) Filter.atTop
  -/
  refine (atTop_basis_Ioi' 0).ge_iff.2 fun a ha ↦ ?_
  /-
    a : Real
    ha : LT.lt 0 a
    ⊢ Membership.mem (Filter.comap Real.toNNReal Filter.atTop) (Set.Ioi a)
  -/
  filter_upwards [preimage_mem_comap (Ioi_mem_atTop a.toNNReal)] with x hx
  /-
    case h
    a : Real
    ha : LT.lt 0 a
    x : Real
    hx : Membership.mem (Set.preimage Real.toNNReal (Set.Ioi a.toNNReal)) x
    ⊢ Membership.mem (Set.Ioi a) x
  -/
  exact (Real.toNNReal_lt_toNNReal_iff_of_nonneg ha.le).1 hx
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.Real.tendsto_toNNReal_atTop_iff {l : Filter α} {f : α → ℝ} :
    Tendsto (fun x ↦ (f x).toNNReal) l atTop ↔ Tendsto f l atTop := by
  /-
    α : Type u_1
    l : Filter α
    f : α → Real
    ⊢ Iff (Filter.Tendsto (fun x => (f x).toNNReal) l Filter.atTop) (Filter.Tendst …
  -/
  rw [← Real.comap_toNNReal_atTop, tendsto_comap_iff, Function.comp_def]
  /-
    🎉 no goals
  -/


theorem _root_.Real.tendsto_toNNReal_atTop : Tendsto Real.toNNReal atTop atTop :=
  Real.tendsto_toNNReal_atTop_iff.2 tendsto_id


theorem nhds_zero : 𝓝 (0 : ℝ≥0) = ⨅ (a : ℝ≥0) (_ : a ≠ 0), 𝓟 (Iio a) :=
                             /-
                               ⊢ Eq (iInf fun l => iInf fun x => Filter.principal (Set.Iio l)) (iInf fun a => …
                             -/
  nhds_bot_order.trans <| by simp only [bot_lt_iff_ne_bot]; rfl
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem nhds_zero_basis : (𝓝 (0 : ℝ≥0)).HasBasis (fun a : ℝ≥0 => 0 < a) fun a => Iio a :=
  nhds_bot_basis


instance : ContinuousSub ℝ≥0 :=
  ⟨((continuous_coe.fst'.sub continuous_coe.snd').max continuous_const).subtype_mk _⟩


instance : HasContinuousInv₀ ℝ≥0 := inferInstance


instance [TopologicalSpace α] [MulAction ℝ α] [ContinuousSMul ℝ α] :
    ContinuousSMul ℝ≥0 α where
  continuous_smul := continuous_induced_dom.fst'.smul continuous_snd


@[norm_cast]
theorem hasSum_coe {f : α → ℝ≥0} {r : ℝ≥0} : HasSum (fun a => (f a : ℝ)) (r : ℝ) ↔ HasSum f r := by
  /-
    α : Type u_1
    f : α → NNReal
    r : NNReal
    ⊢ Iff (HasSum (fun a => ↑(f a)) ↑r) (HasSum f r)
  -/
  simp only [HasSum, ← coe_sum, tendsto_coe]
  /-
    🎉 no goals
  -/


protected theorem _root_.HasSum.toNNReal {f : α → ℝ} {y : ℝ} (hf₀ : ∀ n, 0 ≤ f n)
    (hy : HasSum f y) : HasSum (fun x => Real.toNNReal (f x)) y.toNNReal := by
  /-
    α : Type u_1
    f : α → Real
    y : Real
    hf₀ : ∀ (n : α), LE.le 0 (f n)
    hy : HasSum f y
    ⊢ HasSum (fun x => (f x).toNNReal) y.toNNReal
  -/
  lift y to ℝ≥0 using hy.nonneg hf₀
  /-
    case intro
    α : Type u_1
    f : α → Real
    hf₀ : ∀ (n : α), LE.le 0 (f n)
    y : NNReal
    hy : HasSum f ↑y
    ⊢ HasSum (fun x => (f x).toNNReal) (↑y).toNNReal
  -/
  lift f to α → ℝ≥0 using hf₀
  /-
    case intro.intro
    α : Type u_1
    y : NNReal
    f : α → NNReal
    hy : HasSum (fun i => ↑(f i)) ↑y
    ⊢ HasSum (fun x => ((fun i => ↑(f i)) x).toNNReal) (↑y).toNNReal
  -/
  simpa [hasSum_coe] using hy
  /-
    🎉 no goals
  -/


theorem hasSum_real_toNNReal_of_nonneg {f : α → ℝ} (hf_nonneg : ∀ n, 0 ≤ f n) (hf : Summable f) :
    HasSum (fun n => Real.toNNReal (f n)) (Real.toNNReal (∑' n, f n)) :=
  hf.hasSum.toNNReal hf_nonneg


@[norm_cast]
theorem summable_coe {f : α → ℝ≥0} : (Summable fun a => (f a : ℝ)) ↔ Summable f := by
  /-
    α : Type u_1
    f : α → NNReal
    ⊢ Iff (Summable fun a => ↑(f a)) (Summable f)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      f : α → NNReal
      ⊢ (Summable fun a => ↑(f a)) → Summable f
    -/
  · exact fun ⟨a, ha⟩ => ⟨⟨a, ha.nonneg fun x => (f x).2⟩, hasSum_coe.1 ha⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      f : α → NNReal
      ⊢ Summable f → Summable fun a => ↑(f a)
    -/
  · exact fun ⟨a, ha⟩ => ⟨a.1, hasSum_coe.2 ha⟩
    /-
      🎉 no goals
    -/


theorem summable_mk {f : α → ℝ} (hf : ∀ n, 0 ≤ f n) :
    (@Summable ℝ≥0 _ _ _ fun n => ⟨f n, hf n⟩) ↔ Summable f :=
  Iff.symm <| summable_coe (f := fun x => ⟨f x, hf x⟩)


@[norm_cast]
theorem coe_tsum {f : α → ℝ≥0} : ↑(∑' a, f a) = ∑' a, (f a : ℝ) := by
  classical
  exact if hf : Summable f then Eq.symm <| (hasSum_coe.2 <| hf.hasSum).tsum_eq
  else by simp [tsum_def, hf, mt summable_coe.1 hf]


theorem coe_tsum_of_nonneg {f : α → ℝ} (hf₁ : ∀ n, 0 ≤ f n) :
    (⟨∑' n, f n, tsum_nonneg hf₁⟩ : ℝ≥0) = (∑' n, ⟨f n, hf₁ n⟩ : ℝ≥0) :=
  NNReal.eq <| Eq.symm <| coe_tsum (f := fun x => ⟨f x, hf₁ x⟩)


nonrec theorem tsum_mul_left (a : ℝ≥0) (f : α → ℝ≥0) : ∑' x, a * f x = a * ∑' x, f x :=
                  /-
                    α : Type u_1
                    a : NNReal
                    f : α → NNReal
                    ⊢ Eq ↑(tsum fun x => HMul.hMul a (f x)) ↑(HMul.hMul a (tsum fun x => f x))
                  -/
  NNReal.eq <| by simp only [coe_tsum, NNReal.coe_mul, tsum_mul_left]
                  /-
                    🎉 no goals
                  -/


nonrec theorem tsum_mul_right (f : α → ℝ≥0) (a : ℝ≥0) : ∑' x, f x * a = (∑' x, f x) * a :=
                  /-
                    α : Type u_1
                    f : α → NNReal
                    a : NNReal
                    ⊢ Eq ↑(tsum fun x => HMul.hMul (f x) a) ↑(HMul.hMul (tsum fun x => f x) a)
                  -/
  NNReal.eq <| by simp only [coe_tsum, NNReal.coe_mul, tsum_mul_right]
                  /-
                    🎉 no goals
                  -/


theorem summable_comp_injective {β : Type*} {f : α → ℝ≥0} (hf : Summable f) {i : β → α}
    (hi : Function.Injective i) : Summable (f ∘ i) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → NNReal
    hf : Summable f
    i : β → α
    hi : Function.Injective i
    ⊢ Summable (Function.comp f i)
  -/
  rw [← summable_coe] at hf ⊢
  /-
    α : Type u_1
    β : Type u_2
    f : α → NNReal
    hf : Summable fun a => ↑(f a)
    i : β → α
    hi : Function.Injective i
    ⊢ Summable fun a => ↑(Function.comp f i a)
  -/
  exact hf.comp_injective hi
  /-
    🎉 no goals
  -/


theorem summable_nat_add (f : ℕ → ℝ≥0) (hf : Summable f) (k : ℕ) : Summable fun i => f (i + k) :=
  summable_comp_injective hf <| add_left_injective k


nonrec theorem summable_nat_add_iff {f : ℕ → ℝ≥0} (k : ℕ) :
    (Summable fun i => f (i + k)) ↔ Summable f := by
  /-
    f : Nat → NNReal
    k : Nat
    ⊢ Iff (Summable fun i => f (HAdd.hAdd i k)) (Summable f)
  -/
  rw [← summable_coe, ← summable_coe]
  /-
    f : Nat → NNReal
    k : Nat
    ⊢ Iff (Summable fun a => ↑(f (HAdd.hAdd a k))) (Summable fun a => ↑(f a))
  -/
  exact @summable_nat_add_iff ℝ _ _ _ (fun i => (f i : ℝ)) k
  /-
    🎉 no goals
  -/


nonrec theorem hasSum_nat_add_iff {f : ℕ → ℝ≥0} (k : ℕ) {a : ℝ≥0} :
    HasSum (fun n => f (n + k)) a ↔ HasSum f (a + ∑ i ∈ range k, f i) := by
  /-
    f : Nat → NNReal
    k : Nat
    a : NNReal
    ⊢ Iff (HasSum (fun n => f (HAdd.hAdd n k)) a) (HasSum f (HAdd.hAdd a ((Finset. …
  -/
  rw [← hasSum_coe, hasSum_nat_add_iff (f := fun n => toReal (f n)) k]; norm_cast
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem sum_add_tsum_nat_add {f : ℕ → ℝ≥0} (k : ℕ) (hf : Summable f) :
    ∑' i, f i = (∑ i ∈ range k, f i) + ∑' i, f (i + k) :=
  (sum_add_tsum_nat_add' <| (summable_nat_add_iff k).2 hf).symm


theorem iInf_real_pos_eq_iInf_nnreal_pos [CompleteLattice α] {f : ℝ → α} :
    ⨅ (n : ℝ) (_ : 0 < n), f n = ⨅ (n : ℝ≥0) (_ : 0 < n), f n :=
  le_antisymm (iInf_mono' fun r => ⟨r, le_rfl⟩) (iInf₂_mono' fun r hr => ⟨⟨r, hr.le⟩, hr, le_rfl⟩)


theorem tendsto_cofinite_zero_of_summable {α} {f : α → ℝ≥0} (hf : Summable f) :
    Tendsto f cofinite (𝓝 0) := by
  /-
    α : Type u_1
    f : α → NNReal
    hf : Summable f
    ⊢ Filter.Tendsto f Filter.cofinite (nhds 0)
  -/
  simp only [← summable_coe, ← tendsto_coe] at hf ⊢
  /-
    α : Type u_1
    f : α → NNReal
    hf : Summable fun a => ↑(f a)
    ⊢ Filter.Tendsto (fun a => ↑(f a)) Filter.cofinite (nhds ↑0)
  -/
  exact hf.tendsto_cofinite_zero
  /-
    🎉 no goals
  -/


theorem tendsto_atTop_zero_of_summable {f : ℕ → ℝ≥0} (hf : Summable f) : Tendsto f atTop (𝓝 0) := by
  /-
    f : Nat → NNReal
    hf : Summable f
    ⊢ Filter.Tendsto f Filter.atTop (nhds 0)
  -/
  rw [← Nat.cofinite_eq_atTop]
  /-
    f : Nat → NNReal
    hf : Summable f
    ⊢ Filter.Tendsto f Filter.cofinite (nhds 0)
  -/
  exact tendsto_cofinite_zero_of_summable hf
  /-
    🎉 no goals
  -/


/-- The sum over the complement of a finset tends to `0` when the finset grows to cover the whole
space. This does not need a summability assumption, as otherwise all sums are zero. -/
nonrec theorem tendsto_tsum_compl_atTop_zero {α : Type*} (f : α → ℝ≥0) :
    Tendsto (fun s : Finset α => ∑' b : { x // x ∉ s }, f b) atTop (𝓝 0) := by
  /-
    α : Type u_1
    f : α → NNReal
    ⊢ Filter.Tendsto (fun s => tsum fun b => f ↑b) Filter.atTop (nhds 0)
  -/
  simp_rw [← tendsto_coe, coe_tsum, NNReal.coe_zero]
  /-
    α : Type u_1
    f : α → NNReal
    ⊢ Filter.Tendsto (fun a => tsum fun a_1 => ↑(f ↑a_1)) Filter.atTop (nhds 0)
  -/
  exact tendsto_tsum_compl_atTop_zero fun a : α => (f a : ℝ)
  /-
    🎉 no goals
  -/


/-- `x ↦ x ^ n` as an order isomorphism of `ℝ≥0`. -/
def powOrderIso (n : ℕ) (hn : n ≠ 0) : ℝ≥0 ≃o ℝ≥0 :=
  StrictMono.orderIsoOfSurjective (fun x ↦ x ^ n) (fun x y h =>
      pow_left_strictMonoOn₀ hn (zero_le x) (zero_le y) h) <|
    (continuous_id.pow _).surjective (tendsto_pow_atTop hn) <| by
      /-
        n : Nat
        hn : Ne n 0
        ⊢ Filter.Tendsto (fun b => HPow.hPow (id b) n) Filter.atBot Filter.atBot
      -/
      simpa [OrderBot.atBot_eq, pos_iff_ne_zero]
      /-
        🎉 no goals
      -/


/-- A monotone, bounded above sequence `f : ℕ → ℝ` has a finite limit. -/
theorem _root_.Real.tendsto_of_bddAbove_monotone {f : ℕ → ℝ} (h_bdd : BddAbove (Set.range f))
    (h_mon : Monotone f) : ∃ r : ℝ, Tendsto f atTop (𝓝 r) := by
  /-
    f : Nat → Real
    h_bdd : BddAbove (Set.range f)
    h_mon : Monotone f
    ⊢ Exists fun r => Filter.Tendsto f Filter.atTop (nhds r)
  -/
  obtain ⟨B, hB⟩ := Real.exists_isLUB (Set.range_nonempty f) h_bdd
  /-
    case intro
    f : Nat → Real
    h_bdd : BddAbove (Set.range f)
    h_mon : Monotone f
    B : Real
    hB : IsLUB (Set.range f) B
    ⊢ Exists fun r => Filter.Tendsto f Filter.atTop (nhds r)
  -/
  exact ⟨B, tendsto_atTop_isLUB h_mon hB⟩
  /-
    🎉 no goals
  -/


/-- An antitone, bounded below sequence `f : ℕ → ℝ` has a finite limit. -/
theorem _root_.Real.tendsto_of_bddBelow_antitone {f : ℕ → ℝ} (h_bdd : BddBelow (Set.range f))
    (h_ant : Antitone f) : ∃ r : ℝ, Tendsto f atTop (𝓝 r) := by
  /-
    f : Nat → Real
    h_bdd : BddBelow (Set.range f)
    h_ant : Antitone f
    ⊢ Exists fun r => Filter.Tendsto f Filter.atTop (nhds r)
  -/
  obtain ⟨B, hB⟩ := Real.exists_isGLB (Set.range_nonempty f) h_bdd
  /-
    case intro
    f : Nat → Real
    h_bdd : BddBelow (Set.range f)
    h_ant : Antitone f
    B : Real
    hB : IsGLB (Set.range f) B
    ⊢ Exists fun r => Filter.Tendsto f Filter.atTop (nhds r)
  -/
  exact ⟨B, tendsto_atTop_isGLB h_ant hB⟩
  /-
    🎉 no goals
  -/


/-- An antitone sequence `f : ℕ → ℝ≥0` has a finite limit. -/
theorem tendsto_of_antitone {f : ℕ → ℝ≥0} (h_ant : Antitone f) :
    ∃ r : ℝ≥0, Tendsto f atTop (𝓝 r) := by
  have h_bdd_0 : (0 : ℝ) ∈ lowerBounds (Set.range fun n : ℕ => (f n : ℝ)) := by
    rintro r ⟨n, hn⟩
    simp_rw [← hn]
    exact NNReal.coe_nonneg _
  /-
    f : Nat → NNReal
    h_ant : Antitone f
    h_bdd_0 : Membership.mem (lowerBounds (Set.range fun n => ↑(f n))) 0
    ⊢ Exists fun r => Filter.Tendsto f Filter.atTop (nhds r)
  -/
  obtain ⟨L, hL⟩ := Real.tendsto_of_bddBelow_antitone ⟨0, h_bdd_0⟩ h_ant
  have hL0 : 0 ≤ L :=
    haveI h_glb : IsGLB (Set.range fun n => (f n : ℝ)) L := isGLB_of_tendsto_atTop h_ant hL
    (le_isGLB_iff h_glb).mpr h_bdd_0
  /-
    case intro
    f : Nat → NNReal
    h_ant : Antitone f
    h_bdd_0 : Membership.mem (lowerBounds (Set.range fun n => ↑(f n))) 0
    L : Real
    hL : Filter.Tendsto (fun n => ↑(f n)) Filter.atTop (nhds L)
    hL0 : LE.le 0 L
    ⊢ Exists fun r => Filter.Tendsto f Filter.atTop (nhds r)
  -/
  exact ⟨⟨L, hL0⟩, NNReal.tendsto_coe.mp hL⟩
  /-
    🎉 no goals
  -/


instance instProperSpace : ProperSpace ℝ≥0 where
  isCompact_closedBall x r := by
    /-
      x : NNReal
      r : Real
      ⊢ IsCompact (Metric.closedBall x r)
    -/
    have emb : IsClosedEmbedding ((↑) : ℝ≥0 → ℝ) := Isometry.isClosedEmbedding fun _ ↦ congrFun rfl
    /-
      x : NNReal
      r : Real
      emb : Topology.IsClosedEmbedding NNReal.toReal
      ⊢ IsCompact (Metric.closedBall x r)
    -/
    exact emb.isCompact_preimage (K := Metric.closedBall x r) (isCompact_closedBall _ _)
    /-
      🎉 no goals
    -/


