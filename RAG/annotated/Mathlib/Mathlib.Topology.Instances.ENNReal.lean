/-- Topology on `ℝ≥0∞`.

Note: this is different from the `EMetricSpace` topology. The `EMetricSpace` topology has
`IsOpen {∞}`, while this topology doesn't have singleton elements. -/
instance : TopologicalSpace ℝ≥0∞ := Preorder.topology ℝ≥0∞


instance : OrderTopology ℝ≥0∞ := ⟨rfl⟩

-- short-circuit type class inference

instance : T2Space ℝ≥0∞ := inferInstance

instance : T5Space ℝ≥0∞ := inferInstance

instance : T4Space ℝ≥0∞ := inferInstance


instance : SecondCountableTopology ℝ≥0∞ :=
  orderIsoUnitIntervalBirational.toHomeomorph.isEmbedding.secondCountableTopology


instance : MetrizableSpace ENNReal :=
  orderIsoUnitIntervalBirational.toHomeomorph.isEmbedding.metrizableSpace


theorem isEmbedding_coe : IsEmbedding ((↑) : ℝ≥0 → ℝ≥0∞) :=
                                                   /-
                                                     ⊢ (Set.range ENNReal.ofNNReal).OrdConnected
                                                   -/
  coe_strictMono.isEmbedding_of_ordConnected <| by rw [range_coe']; exact ordConnected_Iio
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[deprecated (since := "2024-10-26")]
alias embedding_coe := isEmbedding_coe


theorem isOpen_ne_top : IsOpen { a : ℝ≥0∞ | a ≠ ∞ } := isOpen_ne


theorem isOpen_Ico_zero : IsOpen (Ico 0 b) := by
  /-
    b : ENNReal
    ⊢ IsOpen (Set.Ico 0 b)
  -/
  rw [ENNReal.Ico_eq_Iio]
  /-
    b : ENNReal
    ⊢ IsOpen (Set.Iio b)
  -/
  exact isOpen_Iio
  /-
    🎉 no goals
  -/


theorem isOpenEmbedding_coe : IsOpenEmbedding ((↑) : ℝ≥0 → ℝ≥0∞) :=
                       /-
                         ⊢ IsOpen (Set.range ENNReal.ofNNReal)
                       -/
  ⟨isEmbedding_coe, by rw [range_coe']; exact isOpen_Iio⟩
                                        /-
                                          🎉 no goals
                                        -/


@[deprecated (since := "2024-10-18")]
alias openEmbedding_coe := isOpenEmbedding_coe


theorem coe_range_mem_nhds : range ((↑) : ℝ≥0 → ℝ≥0∞) ∈ 𝓝 (r : ℝ≥0∞) :=
  IsOpen.mem_nhds isOpenEmbedding_coe.isOpen_range <| mem_range_self _


@[norm_cast]
theorem tendsto_coe {f : Filter α} {m : α → ℝ≥0} {a : ℝ≥0} :
    Tendsto (fun a => (m a : ℝ≥0∞)) f (𝓝 ↑a) ↔ Tendsto m f (𝓝 a) :=
  isEmbedding_coe.tendsto_nhds_iff.symm


@[fun_prop]
theorem continuous_coe : Continuous ((↑) : ℝ≥0 → ℝ≥0∞) :=
  isEmbedding_coe.continuous


theorem continuous_coe_iff {α} [TopologicalSpace α] {f : α → ℝ≥0} :
    (Continuous fun a => (f a : ℝ≥0∞)) ↔ Continuous f :=
  isEmbedding_coe.continuous_iff.symm


theorem nhds_coe {r : ℝ≥0} : 𝓝 (r : ℝ≥0∞) = (𝓝 r).map (↑) :=
  (isOpenEmbedding_coe.map_nhds_eq r).symm


theorem tendsto_nhds_coe_iff {α : Type*} {l : Filter α} {x : ℝ≥0} {f : ℝ≥0∞ → α} :
    Tendsto f (𝓝 ↑x) l ↔ Tendsto (f ∘ (↑) : ℝ≥0 → α) (𝓝 x) l := by
  /-
    α : Type u_4
    l : Filter α
    x : NNReal
    f : ENNReal → α
    ⊢ Iff (Filter.Tendsto f (nhds ↑x) l) (Filter.Tendsto (Function.comp f ENNReal. …
  -/
  rw [nhds_coe, tendsto_map'_iff]
  /-
    🎉 no goals
  -/


theorem continuousAt_coe_iff {α : Type*} [TopologicalSpace α] {x : ℝ≥0} {f : ℝ≥0∞ → α} :
    ContinuousAt f ↑x ↔ ContinuousAt (f ∘ (↑) : ℝ≥0 → α) x :=
  tendsto_nhds_coe_iff


theorem nhds_coe_coe {r p : ℝ≥0} :
    𝓝 ((r : ℝ≥0∞), (p : ℝ≥0∞)) = (𝓝 (r, p)).map fun p : ℝ≥0 × ℝ≥0 => (↑p.1, ↑p.2) :=
  ((isOpenEmbedding_coe.prodMap isOpenEmbedding_coe).map_nhds_eq (r, p)).symm


theorem continuous_ofReal : Continuous ENNReal.ofReal :=
  (continuous_coe_iff.2 continuous_id).comp continuous_real_toNNReal


theorem tendsto_ofReal {f : Filter α} {m : α → ℝ} {a : ℝ} (h : Tendsto m f (𝓝 a)) :
    Tendsto (fun a => ENNReal.ofReal (m a)) f (𝓝 (ENNReal.ofReal a)) :=
  (continuous_ofReal.tendsto a).comp h


theorem tendsto_toNNReal {a : ℝ≥0∞} (ha : a ≠ ∞) :
    Tendsto ENNReal.toNNReal (𝓝 a) (𝓝 a.toNNReal) := by
  /-
    a : ENNReal
    ha : Ne a Top.top
    ⊢ Filter.Tendsto ENNReal.toNNReal (nhds a) (nhds a.toNNReal)
  -/
  lift a to ℝ≥0 using ha
  /-
    case intro
    a : NNReal
    ⊢ Filter.Tendsto ENNReal.toNNReal (nhds ↑a) (nhds (↑a).toNNReal)
  -/
  rw [nhds_coe, tendsto_map'_iff]
  /-
    case intro
    a : NNReal
    ⊢ Filter.Tendsto (Function.comp ENNReal.toNNReal ENNReal.ofNNReal) (nhds a) (n …
  -/
  exact tendsto_id
  /-
    🎉 no goals
  -/


theorem eventuallyEq_of_toReal_eventuallyEq {l : Filter α} {f g : α → ℝ≥0∞}
    (hfi : ∀ᶠ x in l, f x ≠ ∞) (hgi : ∀ᶠ x in l, g x ≠ ∞)
    (hfg : (fun x => (f x).toReal) =ᶠ[l] fun x => (g x).toReal) : f =ᶠ[l] g := by
  /-
    α : Type u_1
    l : Filter α
    f g : α → ENNReal
    hfi : Filter.Eventually (fun x => Ne (f x) Top.top) l
    hgi : Filter.Eventually (fun x => Ne (g x) Top.top) l
    hfg : l.EventuallyEq (fun x => (f x).toReal) fun x => (g x).toReal
    ⊢ l.EventuallyEq f g
  -/
  filter_upwards [hfi, hgi, hfg] with _ hfx hgx _
  /-
    case h
    α : Type u_1
    l : Filter α
    f g : α → ENNReal
    hfi : Filter.Eventually (fun x => Ne (f x) Top.top) l
    hgi : Filter.Eventually (fun x => Ne (g x) Top.top) l
    hfg : l.EventuallyEq (fun x => (f x).toReal) fun x => (g x).toReal
    a✝¹ : α
    hfx : Ne (f a✝¹) Top.top
    hgx : Ne (g a✝¹) Top.top
    a✝ : Eq (f a✝¹).toReal (g a✝¹).toReal
    ⊢ Eq (f a✝¹) (g a✝¹)
  -/
  rwa [← ENNReal.toReal_eq_toReal hfx hgx]
  /-
    🎉 no goals
  -/


theorem continuousOn_toNNReal : ContinuousOn ENNReal.toNNReal { a | a ≠ ∞ } := fun _a ha =>
  ContinuousAt.continuousWithinAt (tendsto_toNNReal ha)


theorem tendsto_toReal {a : ℝ≥0∞} (ha : a ≠ ∞) : Tendsto ENNReal.toReal (𝓝 a) (𝓝 a.toReal) :=
  NNReal.tendsto_coe.2 <| tendsto_toNNReal ha


lemma continuousOn_toReal : ContinuousOn ENNReal.toReal { a | a ≠ ∞ } :=
  NNReal.continuous_coe.comp_continuousOn continuousOn_toNNReal


lemma continuousAt_toReal (hx : x ≠ ∞) : ContinuousAt ENNReal.toReal x :=
  continuousOn_toReal.continuousAt (isOpen_ne_top.mem_nhds_iff.mpr hx)


/-- The set of finite `ℝ≥0∞` numbers is homeomorphic to `ℝ≥0`. -/
def neTopHomeomorphNNReal : { a | a ≠ ∞ } ≃ₜ ℝ≥0 where
  toEquiv := neTopEquivNNReal
  continuous_toFun := continuousOn_iff_continuous_restrict.1 continuousOn_toNNReal
  continuous_invFun := continuous_coe.subtype_mk _


/-- The set of finite `ℝ≥0∞` numbers is homeomorphic to `ℝ≥0`. -/
def ltTopHomeomorphNNReal : { a | a < ∞ } ≃ₜ ℝ≥0 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    a b : ENNReal
    r : NNReal
    x ε : ENNReal
    ⊢ Homeomorph (↑(setOf fun a => LT.lt a Top.top)) NNReal
  -/
  refine (Homeomorph.setCongr ?_).trans neTopHomeomorphNNReal
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    a b : ENNReal
    r : NNReal
    x ε : ENNReal
    ⊢ Eq (setOf fun a => LT.lt a Top.top) (setOf fun a => Ne a Top.top)
  -/
  simp only [mem_setOf_eq, lt_top_iff_ne_top]
  /-
    🎉 no goals
  -/


theorem nhds_top : 𝓝 ∞ = ⨅ (a) (_ : a ≠ ∞), 𝓟 (Ioi a) :=
                             /-
                               ⊢ Eq (iInf fun l => iInf fun x => Filter.principal (Set.Ioi l)) (iInf fun a => …
                             -/
  nhds_top_order.trans <| by simp [lt_top_iff_ne_top, Ioi]
                             /-
                               🎉 no goals
                             -/


theorem nhds_top' : 𝓝 ∞ = ⨅ r : ℝ≥0, 𝓟 (Ioi ↑r) :=
  nhds_top.trans <| iInf_ne_top _


theorem nhds_top_basis : (𝓝 ∞).HasBasis (fun a => a < ∞) fun a => Ioi a :=
  _root_.nhds_top_basis


theorem tendsto_nhds_top_iff_nnreal {m : α → ℝ≥0∞} {f : Filter α} :
    Tendsto m f (𝓝 ∞) ↔ ∀ x : ℝ≥0, ∀ᶠ a in f, ↑x < m a := by
  /-
    α : Type u_1
    m : α → ENNReal
    f : Filter α
    ⊢ Iff (Filter.Tendsto m f (nhds Top.top)) (∀ (x : NNReal), Filter.Eventually ( …
  -/
  simp only [nhds_top', tendsto_iInf, tendsto_principal, mem_Ioi]
  /-
    🎉 no goals
  -/


theorem tendsto_nhds_top_iff_nat {m : α → ℝ≥0∞} {f : Filter α} :
    Tendsto m f (𝓝 ∞) ↔ ∀ n : ℕ, ∀ᶠ a in f, ↑n < m a :=
  tendsto_nhds_top_iff_nnreal.trans
                   /-
                     α : Type u_1
                     m : α → ENNReal
                     f : Filter α
                     h : ∀ (x : NNReal), Filter.Eventually (fun a => LT.lt (↑x) (m a)) f
                     n : Nat
                     ⊢ Filter.Eventually (fun a => LT.lt (↑n) (m a)) f
                   -/
    ⟨fun h n => by simpa only [ENNReal.coe_natCast] using h n, fun h x =>
                   /-
                     🎉 no goals
                   -/
      let ⟨n, hn⟩ := exists_nat_gt x
                                         /-
                                           α : Type u_1
                                           m : α → ENNReal
                                           f : Filter α
                                           h : ∀ (n : Nat), Filter.Eventually (fun a => LT.lt (↑n) (m a)) f
                                           x : NNReal
                                           n : Nat
                                           hn : LT.lt x ↑n
                                           x✝ : α
                                           ⊢ LT.lt ↑x ↑n
                                         -/
      (h n).mono fun _ => lt_trans <| by rwa [← ENNReal.coe_natCast, coe_lt_coe]⟩
                                         /-
                                           🎉 no goals
                                         -/


theorem tendsto_nhds_top {m : α → ℝ≥0∞} {f : Filter α} (h : ∀ n : ℕ, ∀ᶠ a in f, ↑n < m a) :
    Tendsto m f (𝓝 ∞) :=
  tendsto_nhds_top_iff_nat.2 h


theorem tendsto_nat_nhds_top : Tendsto (fun n : ℕ => ↑n) atTop (𝓝 ∞) :=
  tendsto_nhds_top fun n =>
    mem_atTop_sets.2 ⟨n + 1, fun _m hm => mem_setOf.2 <| Nat.cast_lt.2 <| Nat.lt_of_succ_le hm⟩


@[simp, norm_cast]
theorem tendsto_coe_nhds_top {f : α → ℝ≥0} {l : Filter α} :
    Tendsto (fun x => (f x : ℝ≥0∞)) l (𝓝 ∞) ↔ Tendsto f l atTop := by
  /-
    α : Type u_1
    f : α → NNReal
    l : Filter α
    ⊢ Iff (Filter.Tendsto (fun x => ↑(f x)) l (nhds Top.top)) (Filter.Tendsto f l  …
  -/
  rw [tendsto_nhds_top_iff_nnreal, atTop_basis_Ioi.tendsto_right_iff]; simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem tendsto_ofReal_nhds_top {f : α → ℝ} {l : Filter α} :
    Tendsto (fun x ↦ ENNReal.ofReal (f x)) l (𝓝 ∞) ↔ Tendsto f l atTop :=
  tendsto_coe_nhds_top.trans Real.tendsto_toNNReal_atTop_iff


theorem tendsto_ofReal_atTop : Tendsto ENNReal.ofReal atTop (𝓝 ∞) :=
  tendsto_ofReal_nhds_top.2 tendsto_id


theorem nhds_zero : 𝓝 (0 : ℝ≥0∞) = ⨅ (a) (_ : a ≠ 0), 𝓟 (Iio a) :=
                             /-
                               ⊢ Eq (iInf fun l => iInf fun x => Filter.principal (Set.Iio l)) (iInf fun a => …
                             -/
  nhds_bot_order.trans <| by simp [pos_iff_ne_zero, Iio]
                             /-
                               🎉 no goals
                             -/


theorem nhds_zero_basis : (𝓝 (0 : ℝ≥0∞)).HasBasis (fun a : ℝ≥0∞ => 0 < a) fun a => Iio a :=
  nhds_bot_basis


theorem nhds_zero_basis_Iic : (𝓝 (0 : ℝ≥0∞)).HasBasis (fun a : ℝ≥0∞ => 0 < a) Iic :=
  nhds_bot_basis_Iic

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: add a TC for `≠ ∞`?

@[instance]
theorem nhdsGT_coe_neBot {r : ℝ≥0} : (𝓝[>] (r : ℝ≥0∞)).NeBot :=
  nhdsGT_neBot_of_exists_gt ⟨∞, ENNReal.coe_lt_top⟩


@[deprecated (since := "2024-12-22")] alias nhdsWithin_Ioi_coe_neBot := nhdsGT_coe_neBot


@[instance] theorem nhdsGT_zero_neBot : (𝓝[>] (0 : ℝ≥0∞)).NeBot := nhdsGT_coe_neBot


@[deprecated (since := "2024-12-22")] alias nhdsWithin_Ioi_zero_neBot := nhdsGT_zero_neBot


@[instance] theorem nhdsGT_one_neBot : (𝓝[>] (1 : ℝ≥0∞)).NeBot := nhdsGT_coe_neBot


@[deprecated (since := "2024-12-22")] alias nhdsWithin_Ioi_one_neBot := nhdsGT_one_neBot


@[instance] theorem nhdsGT_nat_neBot (n : ℕ) : (𝓝[>] (n : ℝ≥0∞)).NeBot := nhdsGT_coe_neBot


@[deprecated (since := "2024-12-22")] alias nhdsWithin_Ioi_nat_neBot := nhdsGT_nat_neBot


@[instance]
theorem nhdsGT_ofNat_neBot (n : ℕ) [n.AtLeastTwo] : (𝓝[>] (OfNat.ofNat n : ℝ≥0∞)).NeBot :=
  nhdsGT_coe_neBot


@[deprecated (since := "2024-12-22")] alias nhdsWithin_Ioi_ofNat_nebot := nhdsGT_ofNat_neBot


@[instance]
theorem nhdsLT_neBot [NeZero x] : (𝓝[<] x).NeBot :=
  nhdsWithin_Iio_self_neBot' ⟨0, NeZero.pos x⟩


@[deprecated (since := "2024-12-22")] alias nhdsWithin_Iio_neBot := nhdsLT_neBot


/-- Closed intervals `Set.Icc (x - ε) (x + ε)`, `ε ≠ 0`, form a basis of neighborhoods of an
extended nonnegative real number `x ≠ ∞`. We use `Set.Icc` instead of `Set.Ioo` because this way the
statement works for `x = 0`.
-/
theorem hasBasis_nhds_of_ne_top' (xt : x ≠ ∞) :
    (𝓝 x).HasBasis (· ≠ 0) (fun ε => Icc (x - ε) (x + ε)) := by
  /-
    x : ENNReal
    xt : Ne x Top.top
    ⊢ (nhds x).HasBasis (fun x => Ne x 0) fun ε => Set.Icc (HSub.hSub x ε) (HAdd.h …
  -/
  rcases (zero_le x).eq_or_gt with rfl | x0
    /-
      case inl
      xt : Ne 0 Top.top
      ⊢ (nhds 0).HasBasis (fun x => Ne x 0) fun ε => Set.Icc (HSub.hSub 0 ε) (HAdd.h …
    -/
  · simp_rw [zero_tsub, zero_add, ← bot_eq_zero, Icc_bot, ← bot_lt_iff_ne_bot]
    /-
      case inl
      xt : Ne 0 Top.top
      ⊢ (nhds Bot.bot).HasBasis (fun x => LT.lt Bot.bot x) fun ε => Set.Iic ε
    -/
    exact nhds_bot_basis_Iic
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : ENNReal
      xt : Ne x Top.top
      x0 : LT.lt 0 x
      ⊢ (nhds x).HasBasis (fun x => Ne x 0) fun ε => Set.Icc (HSub.hSub x ε) (HAdd.h …
    -/
  · refine (nhds_basis_Ioo' ⟨_, x0⟩ ⟨_, xt.lt_top⟩).to_hasBasis ?_ fun ε ε0 => ?_
      /-
        case inr.refine_1
        x : ENNReal
        xt : Ne x Top.top
        x0 : LT.lt 0 x
        ⊢ ∀ (i : Prod ENNReal ENNReal), And (LT.lt i.1 x) (LT.lt x i.2) → Exists fun i …
      -/
    · rintro ⟨a, b⟩ ⟨ha, hb⟩
      /-
        case inr.refine_1.mk.intro
        x : ENNReal
        xt : Ne x Top.top
        x0 : LT.lt 0 x
        a b : ENNReal
        ha : LT.lt { fst := a, snd := b }.1 x
        hb : LT.lt x { fst := a, snd := b }.2
        ⊢ Exists fun i' => And (Ne i' 0) (HasSubset.Subset (Set.Icc (HSub.hSub x i') ( …
      -/
      rcases exists_between (tsub_pos_of_lt ha) with ⟨ε, ε0, hε⟩
      /-
        case inr.refine_1.mk.intro.intro.intro
        x : ENNReal
        xt : Ne x Top.top
        x0 : LT.lt 0 x
        a b : ENNReal
        ha : LT.lt { fst := a, snd := b }.1 x
        hb : LT.lt x { fst := a, snd := b }.2
        ε : ENNReal
        ε0 : LT.lt 0 ε
        hε : LT.lt ε (HSub.hSub x { fst := a, snd := b }.1)
        ⊢ Exists fun i' => And (Ne i' 0) (HasSubset.Subset (Set.Icc (HSub.hSub x i') ( …
      -/
      rcases lt_iff_exists_add_pos_lt.1 hb with ⟨δ, δ0, hδ⟩
      /-
        case inr.refine_1.mk.intro.intro.intro.intro.intro
        x : ENNReal
        xt : Ne x Top.top
        x0 : LT.lt 0 x
        a b : ENNReal
        ha : LT.lt { fst := a, snd := b }.1 x
        hb : LT.lt x { fst := a, snd := b }.2
        ε : ENNReal
        ε0 : LT.lt 0 ε
        hε : LT.lt ε (HSub.hSub x { fst := a, snd := b }.1)
        δ : NNReal
        δ0 : LT.lt 0 δ
        hδ : LT.lt (HAdd.hAdd x ↑δ) { fst := a, snd := b }.2
        ⊢ Exists fun i' => And (Ne i' 0) (HasSubset.Subset (Set.Icc (HSub.hSub x i') ( …
      -/
      refine ⟨min ε δ, (lt_min ε0 (coe_pos.2 δ0)).ne', Icc_subset_Ioo ?_ ?_⟩
        /-
          case inr.refine_1.mk.intro.intro.intro.intro.intro.refine_1
          x : ENNReal
          xt : Ne x Top.top
          x0 : LT.lt 0 x
          a b : ENNReal
          ha : LT.lt { fst := a, snd := b }.1 x
          hb : LT.lt x { fst := a, snd := b }.2
          ε : ENNReal
          ε0 : LT.lt 0 ε
          hε : LT.lt ε (HSub.hSub x { fst := a, snd := b }.1)
          δ : NNReal
          δ0 : LT.lt 0 δ
          hδ : LT.lt (HAdd.hAdd x ↑δ) { fst := a, snd := b }.2
          ⊢ LT.lt { fst := a, snd := b }.1 (HSub.hSub x (Min.min ε ↑δ))
        -/
      · exact lt_tsub_comm.2 ((min_le_left _ _).trans_lt hε)
        /-
          🎉 no goals
        -/
        /-
          case inr.refine_1.mk.intro.intro.intro.intro.intro.refine_2
          x : ENNReal
          xt : Ne x Top.top
          x0 : LT.lt 0 x
          a b : ENNReal
          ha : LT.lt { fst := a, snd := b }.1 x
          hb : LT.lt x { fst := a, snd := b }.2
          ε : ENNReal
          ε0 : LT.lt 0 ε
          hε : LT.lt ε (HSub.hSub x { fst := a, snd := b }.1)
          δ : NNReal
          δ0 : LT.lt 0 δ
          hδ : LT.lt (HAdd.hAdd x ↑δ) { fst := a, snd := b }.2
          ⊢ LT.lt (HAdd.hAdd x (Min.min ε ↑δ)) { fst := a, snd := b }.2
        -/
      · exact (add_le_add_left (min_le_right _ _) _).trans_lt hδ
        /-
          🎉 no goals
        -/
    · exact ⟨(x - ε, x + ε), ⟨ENNReal.sub_lt_self xt x0.ne' ε0,
        lt_add_right xt ε0⟩, Ioo_subset_Icc_self⟩


theorem hasBasis_nhds_of_ne_top (xt : x ≠ ∞) :
    (𝓝 x).HasBasis (0 < ·) (fun ε => Icc (x - ε) (x + ε)) := by
  /-
    x : ENNReal
    xt : Ne x Top.top
    ⊢ (nhds x).HasBasis (fun x => LT.lt 0 x) fun ε => Set.Icc (HSub.hSub x ε) (HAd …
  -/
  simpa only [pos_iff_ne_zero] using hasBasis_nhds_of_ne_top' xt
  /-
    🎉 no goals
  -/


theorem Icc_mem_nhds (xt : x ≠ ∞) (ε0 : ε ≠ 0) : Icc (x - ε) (x + ε) ∈ 𝓝 x :=
  (hasBasis_nhds_of_ne_top' xt).mem_of_mem ε0


theorem nhds_of_ne_top (xt : x ≠ ∞) : 𝓝 x = ⨅ ε > 0, 𝓟 (Icc (x - ε) (x + ε)) :=
  (hasBasis_nhds_of_ne_top xt).eq_biInf


theorem biInf_le_nhds : ∀ x : ℝ≥0∞, ⨅ ε > 0, 𝓟 (Icc (x - ε) (x + ε)) ≤ 𝓝 x
  | ∞ => iInf₂_le_of_le 1 one_pos <| by
    /-
      ⊢ LE.le (Filter.principal (Set.Icc (HSub.hSub Top.top 1) (HAdd.hAdd Top.top 1) …
    -/
    simpa only [← coe_one, top_sub_coe, top_add, Icc_self, principal_singleton] using pure_le_nhds _
    /-
      🎉 no goals
    -/
  | (x : ℝ≥0) => (nhds_of_ne_top coe_ne_top).ge


protected theorem tendsto_nhds_of_Icc {f : Filter α} {u : α → ℝ≥0∞} {a : ℝ≥0∞}
    (h : ∀ ε > 0, ∀ᶠ x in f, u x ∈ Icc (a - ε) (a + ε)) : Tendsto u f (𝓝 a) := by
  /-
    α : Type u_1
    f : Filter α
    u : α → ENNReal
    a : ENNReal
    h : ∀ (ε : ENNReal), GT.gt ε 0 → Filter.Eventually (fun x => Membership.mem (S …
    ⊢ Filter.Tendsto u f (nhds a)
  -/
  refine Tendsto.mono_right ?_ (biInf_le_nhds _)
  /-
    α : Type u_1
    f : Filter α
    u : α → ENNReal
    a : ENNReal
    h : ∀ (ε : ENNReal), GT.gt ε 0 → Filter.Eventually (fun x => Membership.mem (S …
    ⊢ Filter.Tendsto u f (iInf fun ε => iInf fun h => Filter.principal (Set.Icc (H …
  -/
  simpa only [tendsto_iInf, tendsto_principal]
  /-
    🎉 no goals
  -/


/-- Characterization of neighborhoods for `ℝ≥0∞` numbers. See also `tendsto_order`
for a version with strict inequalities. -/
protected theorem tendsto_nhds {f : Filter α} {u : α → ℝ≥0∞} {a : ℝ≥0∞} (ha : a ≠ ∞) :
    Tendsto u f (𝓝 a) ↔ ∀ ε > 0, ∀ᶠ x in f, u x ∈ Icc (a - ε) (a + ε) := by
  /-
    α : Type u_1
    f : Filter α
    u : α → ENNReal
    a : ENNReal
    ha : Ne a Top.top
    ⊢ Iff (Filter.Tendsto u f (nhds a)) (∀ (ε : ENNReal), GT.gt ε 0 → Filter.Event …
  -/
  simp only [nhds_of_ne_top ha, tendsto_iInf, tendsto_principal]
  /-
    🎉 no goals
  -/


protected theorem tendsto_nhds_zero {f : Filter α} {u : α → ℝ≥0∞} :
    Tendsto u f (𝓝 0) ↔ ∀ ε > 0, ∀ᶠ x in f, u x ≤ ε :=
  nhds_zero_basis_Iic.tendsto_right_iff


protected theorem tendsto_atTop [Nonempty β] [SemilatticeSup β] {f : β → ℝ≥0∞} {a : ℝ≥0∞}
    (ha : a ≠ ∞) : Tendsto f atTop (𝓝 a) ↔ ∀ ε > 0, ∃ N, ∀ n ≥ N, f n ∈ Icc (a - ε) (a + ε) :=
                                                                    /-
                                                                      β : Type u_2
                                                                      inst✝¹ : Nonempty β
                                                                      inst✝ : SemilatticeSup β
                                                                      f : β → ENNReal
                                                                      a : ENNReal
                                                                      ha : Ne a Top.top
                                                                      ⊢ Iff (∀ (ib : ENNReal), LT.lt 0 ib → Exists fun ia => And True (∀ (x : β), Me …
                                                                    -/
  .trans (atTop_basis.tendsto_iff (hasBasis_nhds_of_ne_top ha)) (by simp only [true_and]; rfl)
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


instance : ContinuousAdd ℝ≥0∞ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    a b : ENNReal
    r : NNReal
    x ε : ENNReal
    ⊢ ContinuousAdd ENNReal
  -/
  refine ⟨continuous_iff_continuousAt.2 ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    a b : ENNReal
    r : NNReal
    x ε : ENNReal
    ⊢ ∀ (x : Prod ENNReal ENNReal), ContinuousAt (fun p => HAdd.hAdd p.1 p.2) x
  -/
  rintro ⟨_ | a, b⟩
    /-
      case mk.none
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      a b✝ : ENNReal
      r : NNReal
      x ε b : ENNReal
      ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := Option.none, snd := b }
    -/
  · exact tendsto_nhds_top_mono' continuousAt_fst fun p => le_add_right le_rfl
    /-
      🎉 no goals
    -/
  /-
    case mk.some
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    a✝ b✝ : ENNReal
    r : NNReal
    x ε b : ENNReal
    a : NNReal
    ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := Option.some a, snd := b }
  -/
  rcases b with (_ | b)
    /-
      case mk.some.none
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      a✝ b : ENNReal
      r : NNReal
      x ε : ENNReal
      a : NNReal
      ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := Option.some a, snd := Opt …
    -/
  · exact tendsto_nhds_top_mono' continuousAt_snd fun p => le_add_left le_rfl
    /-
      🎉 no goals
    -/
  simp only [ContinuousAt, some_eq_coe, nhds_coe_coe, ← coe_add, tendsto_map'_iff,
    Function.comp_def, tendsto_coe, tendsto_add]


protected theorem tendsto_atTop_zero [Nonempty β] [SemilatticeSup β] {f : β → ℝ≥0∞} :
    Tendsto f atTop (𝓝 0) ↔ ∀ ε > 0, ∃ N, ∀ n ≥ N, f n ≤ ε :=
                                                           /-
                                                             β : Type u_2
                                                             inst✝¹ : Nonempty β
                                                             inst✝ : SemilatticeSup β
                                                             f : β → ENNReal
                                                             ⊢ Iff (∀ (ib : ENNReal), LT.lt 0 ib → Exists fun ia => And True (∀ (x : β), Me …
                                                           -/
  .trans (atTop_basis.tendsto_iff nhds_zero_basis_Iic) (by simp only [true_and]; rfl)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem tendsto_sub : ∀ {a b : ℝ≥0∞}, (a ≠ ∞ ∨ b ≠ ∞) →
    Tendsto (fun p : ℝ≥0∞ × ℝ≥0∞ => p.1 - p.2) (𝓝 (a, b)) (𝓝 (a - b))
                  /-
                    h : Or (Ne Top.top Top.top) (Ne Top.top Top.top)
                    ⊢ Filter.Tendsto (fun p => HSub.hSub p.1 p.2) (nhds { fst := Top.top, snd := T …
                  -/
  | ∞, ∞, h => by simp only [ne_eq, not_true_eq_false, or_self] at h
                  /-
                    🎉 no goals
                  -/
  | ∞, (b : ℝ≥0), _ => by
    /-
      b : NNReal
      x✝ : Or (Ne Top.top Top.top) (Ne (↑b) Top.top)
      ⊢ Filter.Tendsto (fun p => HSub.hSub p.1 p.2) (nhds { fst := Top.top, snd := ↑ …
    -/
    rw [top_sub_coe, tendsto_nhds_top_iff_nnreal]
    refine fun x => ((lt_mem_nhds <| @coe_lt_top (b + 1 + x)).prod_nhds
      (ge_mem_nhds <| coe_lt_coe.2 <| lt_add_one b)).mono fun y hy => ?_
    /-
      b : NNReal
      x✝ : Or (Ne Top.top Top.top) (Ne (↑b) Top.top)
      x : NNReal
      y : Prod ENNReal ENNReal
      hy : And (LT.lt (↑(HAdd.hAdd (HAdd.hAdd b 1) x)) y.1) (LE.le y.2 ↑(HAdd.hAdd b …
      ⊢ LT.lt (↑x) (HSub.hSub y.1 y.2)
    -/
    rw [lt_tsub_iff_left]
    calc y.2 + x ≤ ↑(b + 1) + x := add_le_add_right hy.2 _
    _ < y.1 := hy.1
  | (a : ℝ≥0), ∞, _ => by
    /-
      a : NNReal
      x✝ : Or (Ne (↑a) Top.top) (Ne Top.top Top.top)
      ⊢ Filter.Tendsto (fun p => HSub.hSub p.1 p.2) (nhds { fst := ↑a, snd := Top.to …
    -/
    rw [sub_top]
    /-
      a : NNReal
      x✝ : Or (Ne (↑a) Top.top) (Ne Top.top Top.top)
      ⊢ Filter.Tendsto (fun p => HSub.hSub p.1 p.2) (nhds { fst := ↑a, snd := Top.to …
    -/
    refine (tendsto_pure.2 ?_).mono_right (pure_le_nhds _)
    exact ((gt_mem_nhds <| coe_lt_coe.2 <| lt_add_one a).prod_nhds
      (lt_mem_nhds <| @coe_lt_top (a + 1))).mono fun x hx =>
        tsub_eq_zero_iff_le.2 (hx.1.trans hx.2).le
  | (a : ℝ≥0), (b : ℝ≥0), _ => by
    /-
      a b : NNReal
      x✝ : Or (Ne (↑a) Top.top) (Ne (↑b) Top.top)
      ⊢ Filter.Tendsto (fun p => HSub.hSub p.1 p.2) (nhds { fst := ↑a, snd := ↑b })  …
    -/
    simp only [nhds_coe_coe, tendsto_map'_iff, ← ENNReal.coe_sub, Function.comp_def, tendsto_coe]
    /-
      a b : NNReal
      x✝ : Or (Ne (↑a) Top.top) (Ne (↑b) Top.top)
      ⊢ Filter.Tendsto (fun a => HSub.hSub a.1 a.2) (nhds { fst := a, snd := b }) (n …
    -/
    exact continuous_sub.tendsto (a, b)
    /-
      🎉 no goals
    -/


protected theorem Tendsto.sub {f : Filter α} {ma : α → ℝ≥0∞} {mb : α → ℝ≥0∞} {a b : ℝ≥0∞}
    (hma : Tendsto ma f (𝓝 a)) (hmb : Tendsto mb f (𝓝 b)) (h : a ≠ ∞ ∨ b ≠ ∞) :
    Tendsto (fun a => ma a - mb a) f (𝓝 (a - b)) :=
  show Tendsto ((fun p : ℝ≥0∞ × ℝ≥0∞ => p.1 - p.2) ∘ fun a => (ma a, mb a)) f (𝓝 (a - b)) from
    Tendsto.comp (ENNReal.tendsto_sub h) (hma.prod_mk_nhds hmb)


protected theorem tendsto_mul (ha : a ≠ 0 ∨ b ≠ ∞) (hb : b ≠ 0 ∨ a ≠ ∞) :
    Tendsto (fun p : ℝ≥0∞ × ℝ≥0∞ => p.1 * p.2) (𝓝 (a, b)) (𝓝 (a * b)) := by
  have ht : ∀ b : ℝ≥0∞, b ≠ 0 →
      Tendsto (fun p : ℝ≥0∞ × ℝ≥0∞ => p.1 * p.2) (𝓝 (∞, b)) (𝓝 ∞) := fun b hb => by
    refine tendsto_nhds_top_iff_nnreal.2 fun n => ?_
    rcases lt_iff_exists_nnreal_btwn.1 (pos_iff_ne_zero.2 hb) with ⟨ε, hε, hεb⟩
    have : ∀ᶠ c : ℝ≥0∞ × ℝ≥0∞ in 𝓝 (∞, b), ↑n / ↑ε < c.1 ∧ ↑ε < c.2 :=
      (lt_mem_nhds <| div_lt_top coe_ne_top hε.ne').prod_nhds (lt_mem_nhds hεb)
    refine this.mono fun c hc => ?_
    exact (ENNReal.div_mul_cancel hε.ne' coe_ne_top).symm.trans_lt (mul_lt_mul hc.1 hc.2)
  induction a with
  | top => simp only [ne_eq, or_false, not_true_eq_false] at hb; simp [ht b hb, top_mul hb]
  | coe a =>
    induction b with
    | top =>
      simp only [ne_eq, or_false, not_true_eq_false] at ha
      simpa [Function.comp_def, mul_comm, mul_top ha]
        using (ht a ha).comp (continuous_swap.tendsto (ofNNReal a, ∞))
    | coe b =>
      simp only [nhds_coe_coe, ← coe_mul, tendsto_coe, tendsto_map'_iff, Function.comp_def,
        tendsto_mul]


protected theorem Tendsto.mul {f : Filter α} {ma : α → ℝ≥0∞} {mb : α → ℝ≥0∞} {a b : ℝ≥0∞}
    (hma : Tendsto ma f (𝓝 a)) (ha : a ≠ 0 ∨ b ≠ ∞) (hmb : Tendsto mb f (𝓝 b))
    (hb : b ≠ 0 ∨ a ≠ ∞) : Tendsto (fun a => ma a * mb a) f (𝓝 (a * b)) :=
  show Tendsto ((fun p : ℝ≥0∞ × ℝ≥0∞ => p.1 * p.2) ∘ fun a => (ma a, mb a)) f (𝓝 (a * b)) from
    Tendsto.comp (ENNReal.tendsto_mul ha hb) (hma.prod_mk_nhds hmb)


theorem _root_.ContinuousOn.ennreal_mul [TopologicalSpace α] {f g : α → ℝ≥0∞} {s : Set α}
    (hf : ContinuousOn f s) (hg : ContinuousOn g s) (h₁ : ∀ x ∈ s, f x ≠ 0 ∨ g x ≠ ∞)
    (h₂ : ∀ x ∈ s, g x ≠ 0 ∨ f x ≠ ∞) : ContinuousOn (fun x => f x * g x) s := fun x hx =>
  ENNReal.Tendsto.mul (hf x hx) (h₁ x hx) (hg x hx) (h₂ x hx)


theorem _root_.Continuous.ennreal_mul [TopologicalSpace α] {f g : α → ℝ≥0∞} (hf : Continuous f)
    (hg : Continuous g) (h₁ : ∀ x, f x ≠ 0 ∨ g x ≠ ∞) (h₂ : ∀ x, g x ≠ 0 ∨ f x ≠ ∞) :
    Continuous fun x => f x * g x :=
  continuous_iff_continuousAt.2 fun x =>
    ENNReal.Tendsto.mul hf.continuousAt (h₁ x) hg.continuousAt (h₂ x)


protected theorem Tendsto.const_mul {f : Filter α} {m : α → ℝ≥0∞} {a b : ℝ≥0∞}
    (hm : Tendsto m f (𝓝 b)) (hb : b ≠ 0 ∨ a ≠ ∞) : Tendsto (fun b => a * m b) f (𝓝 (a * b)) :=
                                     /-
                                       α : Type u_1
                                       f : Filter α
                                       m : α → ENNReal
                                       a b : ENNReal
                                       hm : Filter.Tendsto m f (nhds b)
                                       hb : Or (Ne b 0) (Ne a Top.top)
                                       this : Eq a 0
                                       ⊢ Filter.Tendsto (fun b => HMul.hMul a (m b)) f (nhds (HMul.hMul a b))
                                     -/
  by_cases (fun (this : a = 0) => by simp [this, tendsto_const_nhds]) fun ha : a ≠ 0 =>
                                     /-
                                       🎉 no goals
                                     -/
    ENNReal.Tendsto.mul tendsto_const_nhds (Or.inl ha) hm hb


protected theorem Tendsto.mul_const {f : Filter α} {m : α → ℝ≥0∞} {a b : ℝ≥0∞}
    (hm : Tendsto m f (𝓝 a)) (ha : a ≠ 0 ∨ b ≠ ∞) : Tendsto (fun x => m x * b) f (𝓝 (a * b)) := by
  /-
    α : Type u_1
    f : Filter α
    m : α → ENNReal
    a b : ENNReal
    hm : Filter.Tendsto m f (nhds a)
    ha : Or (Ne a 0) (Ne b Top.top)
    ⊢ Filter.Tendsto (fun x => HMul.hMul (m x) b) f (nhds (HMul.hMul a b))
  -/
  simpa only [mul_comm] using ENNReal.Tendsto.const_mul hm ha
  /-
    🎉 no goals
  -/


theorem tendsto_finset_prod_of_ne_top {ι : Type*} {f : ι → α → ℝ≥0∞} {x : Filter α} {a : ι → ℝ≥0∞}
    (s : Finset ι) (h : ∀ i ∈ s, Tendsto (f i) x (𝓝 (a i))) (h' : ∀ i ∈ s, a i ≠ ∞) :
    Tendsto (fun b => ∏ c ∈ s, f c b) x (𝓝 (∏ c ∈ s, a c)) := by
  classical
  induction' s using Finset.induction with a s has IH
  · simp [tendsto_const_nhds]
  simp only [Finset.prod_insert has]
  apply Tendsto.mul (h _ (Finset.mem_insert_self _ _))
  · right
    exact prod_ne_top fun i hi => h' _ (Finset.mem_insert_of_mem hi)
  · exact IH (fun i hi => h _ (Finset.mem_insert_of_mem hi)) fun i hi =>
      h' _ (Finset.mem_insert_of_mem hi)
  · exact Or.inr (h' _ (Finset.mem_insert_self _ _))


protected theorem continuousAt_const_mul {a b : ℝ≥0∞} (h : a ≠ ∞ ∨ b ≠ 0) :
    ContinuousAt (a * ·) b :=
  Tendsto.const_mul tendsto_id h.symm


protected theorem continuousAt_mul_const {a b : ℝ≥0∞} (h : a ≠ ∞ ∨ b ≠ 0) :
    ContinuousAt (fun x => x * a) b :=
  Tendsto.mul_const tendsto_id h.symm


@[fun_prop]
protected theorem continuous_const_mul {a : ℝ≥0∞} (ha : a ≠ ∞) : Continuous (a * ·) :=
  continuous_iff_continuousAt.2 fun _ => ENNReal.continuousAt_const_mul (Or.inl ha)


@[fun_prop]
protected theorem continuous_mul_const {a : ℝ≥0∞} (ha : a ≠ ∞) : Continuous fun x => x * a :=
  continuous_iff_continuousAt.2 fun _ => ENNReal.continuousAt_mul_const (Or.inl ha)


@[fun_prop]
protected theorem continuous_div_const (c : ℝ≥0∞) (c_ne_zero : c ≠ 0) :
    Continuous fun x : ℝ≥0∞ => x / c :=
  ENNReal.continuous_mul_const <| ENNReal.inv_ne_top.2 c_ne_zero


@[continuity, fun_prop]
protected theorem continuous_pow (n : ℕ) : Continuous fun a : ℝ≥0∞ => a ^ n := by
  /-
    n : Nat
    ⊢ Continuous fun a => HPow.hPow a n
  -/
  induction' n with n IH
    /-
      case zero
      ⊢ Continuous fun a => HPow.hPow a 0
    -/
  · simp [continuous_const]
    /-
      🎉 no goals
    -/
  /-
    case succ
    n : Nat
    IH : Continuous fun a => HPow.hPow a n
    ⊢ Continuous fun a => HPow.hPow a (HAdd.hAdd n 1)
  -/
  simp_rw [pow_add, pow_one, continuous_iff_continuousAt]
  /-
    case succ
    n : Nat
    IH : Continuous fun a => HPow.hPow a n
    ⊢ ∀ (x : ENNReal), ContinuousAt (fun a => HMul.hMul (HPow.hPow a n) a) x
  -/
  intro x
  /-
    case succ
    n : Nat
    IH : Continuous fun a => HPow.hPow a n
    x : ENNReal
    ⊢ ContinuousAt (fun a => HMul.hMul (HPow.hPow a n) a) x
  -/
  refine ENNReal.Tendsto.mul (IH.tendsto _) ?_ tendsto_id ?_ <;> by_cases H : x = 0
    /-
      case pos
      n : Nat
      IH : Continuous fun a => HPow.hPow a n
      x : ENNReal
      H : Eq x 0
      ⊢ Or (Ne (HPow.hPow x n) 0) (Ne x Top.top)
    -/
  · simp only [H, zero_ne_top, Ne, or_true, not_false_iff]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      IH : Continuous fun a => HPow.hPow a n
      x : ENNReal
      H : Not (Eq x 0)
      ⊢ Or (Ne (HPow.hPow x n) 0) (Ne x Top.top)
    -/
  · exact Or.inl fun h => H (pow_eq_zero h)
    /-
      🎉 no goals
    -/
  · simp only [H, pow_eq_top_iff, zero_ne_top, false_or, eq_self_iff_true, not_true, Ne,
      not_false_iff, false_and]
    /-
      case neg
      n : Nat
      IH : Continuous fun a => HPow.hPow a n
      x : ENNReal
      H : Not (Eq x 0)
      ⊢ Or (Ne x 0) (Ne (HPow.hPow x n) Top.top)
    -/
  · simp only [H, true_or, Ne, not_false_iff]
    /-
      🎉 no goals
    -/


theorem continuousOn_sub :
    ContinuousOn (fun p : ℝ≥0∞ × ℝ≥0∞ => p.fst - p.snd) { p : ℝ≥0∞ × ℝ≥0∞ | p ≠ ⟨∞, ∞⟩ } := by
  /-
    ⊢ ContinuousOn (fun p => HSub.hSub p.1 p.2) (setOf fun p => Ne p { fst := Top. …
  -/
  rw [ContinuousOn]
  /-
    ⊢ ∀ (x : Prod ENNReal ENNReal), Membership.mem (setOf fun p => Ne p { fst := T …
  -/
  rintro ⟨x, y⟩ hp
  /-
    case mk
    x y : ENNReal
    hp : Membership.mem (setOf fun p => Ne p { fst := Top.top, snd := Top.top }) { …
    ⊢ ContinuousWithinAt (fun p => HSub.hSub p.1 p.2) (setOf fun p => Ne p { fst : …
  -/
  simp only [Ne, Set.mem_setOf_eq, Prod.mk.inj_iff] at hp
  /-
    case mk
    x y : ENNReal
    hp : Not (And (Eq x Top.top) (Eq y Top.top))
    ⊢ ContinuousWithinAt (fun p => HSub.hSub p.1 p.2) (setOf fun p => Ne p { fst : …
  -/
  exact tendsto_nhdsWithin_of_tendsto_nhds (tendsto_sub (not_and_or.mp hp))
  /-
    🎉 no goals
  -/


theorem continuous_sub_left {a : ℝ≥0∞} (a_ne_top : a ≠ ∞) : Continuous (a - ·) := by
  /-
    a : ENNReal
    a_ne_top : Ne a Top.top
    ⊢ Continuous fun x => HSub.hSub a x
  -/
  change Continuous (Function.uncurry Sub.sub ∘ (a, ·))
  /-
    a : ENNReal
    a_ne_top : Ne a Top.top
    ⊢ Continuous (Function.comp (Function.uncurry Sub.sub) fun x => { fst := a, sn …
  -/
  refine continuousOn_sub.comp_continuous (Continuous.Prod.mk a) fun x => ?_
  /-
    a : ENNReal
    a_ne_top : Ne a Top.top
    x : ENNReal
    ⊢ Membership.mem (setOf fun p => Ne p { fst := Top.top, snd := Top.top }) { fs …
  -/
  simp only [a_ne_top, Ne, mem_setOf_eq, Prod.mk.inj_iff, false_and, not_false_iff]
  /-
    🎉 no goals
  -/


theorem continuous_nnreal_sub {a : ℝ≥0} : Continuous fun x : ℝ≥0∞ => (a : ℝ≥0∞) - x :=
  continuous_sub_left coe_ne_top


theorem continuousOn_sub_left (a : ℝ≥0∞) : ContinuousOn (a - ·) { x : ℝ≥0∞ | x ≠ ∞ } := by
  /-
    a : ENNReal
    ⊢ ContinuousOn (fun x => HSub.hSub a x) (setOf fun x => Ne x Top.top)
  -/
  rw [show (fun x => a - x) = (fun p : ℝ≥0∞ × ℝ≥0∞ => p.fst - p.snd) ∘ fun x => ⟨a, x⟩ by rfl]
  /-
    a : ENNReal
    ⊢ ContinuousOn (Function.comp (fun p => HSub.hSub p.1 p.2) fun x => { fst := a …
  -/
  apply ContinuousOn.comp continuousOn_sub (Continuous.continuousOn (Continuous.Prod.mk a))
  /-
    a : ENNReal
    ⊢ Set.MapsTo (fun y => { fst := a, snd := y }) (setOf fun x => Ne x Top.top) ( …
  -/
  rintro _ h (_ | _)
  /-
    case refl
    h : Membership.mem (setOf fun x => Ne x Top.top) Top.top
    ⊢ False
  -/
  exact h none_eq_top
  /-
    🎉 no goals
  -/


theorem continuous_sub_right (a : ℝ≥0∞) : Continuous fun x : ℝ≥0∞ => x - a := by
  /-
    a : ENNReal
    ⊢ Continuous fun x => HSub.hSub x a
  -/
  by_cases a_infty : a = ∞
    /-
      case pos
      a : ENNReal
      a_infty : Eq a Top.top
      ⊢ Continuous fun x => HSub.hSub x a
    -/
  · simp [a_infty, continuous_const, tsub_eq_zero_of_le]
    /-
      🎉 no goals
    -/
    /-
      case neg
      a : ENNReal
      a_infty : Not (Eq a Top.top)
      ⊢ Continuous fun x => HSub.hSub x a
    -/
  · rw [show (fun x => x - a) = (fun p : ℝ≥0∞ × ℝ≥0∞ => p.fst - p.snd) ∘ fun x => ⟨x, a⟩ by rfl]
    /-
      case neg
      a : ENNReal
      a_infty : Not (Eq a Top.top)
      ⊢ Continuous (Function.comp (fun p => HSub.hSub p.1 p.2) fun x => { fst := x,  …
    -/
    apply ContinuousOn.comp_continuous continuousOn_sub (continuous_id'.prod_mk continuous_const)
    /-
      case neg
      a : ENNReal
      a_infty : Not (Eq a Top.top)
      ⊢ ∀ (x : ENNReal), Membership.mem (setOf fun p => Ne p { fst := Top.top, snd : …
    -/
    intro x
    /-
      case neg
      a : ENNReal
      a_infty : Not (Eq a Top.top)
      x : ENNReal
      ⊢ Membership.mem (setOf fun p => Ne p { fst := Top.top, snd := Top.top }) { fs …
    -/
    simp only [a_infty, Ne, mem_setOf_eq, Prod.mk.inj_iff, and_false, not_false_iff]
    /-
      🎉 no goals
    -/


protected theorem Tendsto.pow {f : Filter α} {m : α → ℝ≥0∞} {a : ℝ≥0∞} {n : ℕ}
    (hm : Tendsto m f (𝓝 a)) : Tendsto (fun x => m x ^ n) f (𝓝 (a ^ n)) :=
  ((ENNReal.continuous_pow n).tendsto a).comp hm


theorem le_of_forall_lt_one_mul_le {x y : ℝ≥0∞} (h : ∀ a < 1, a * x ≤ y) : x ≤ y := by
  have : Tendsto (· * x) (𝓝[<] 1) (𝓝 (1 * x)) :=
    (ENNReal.continuousAt_mul_const (Or.inr one_ne_zero)).mono_left inf_le_left
  /-
    x y : ENNReal
    h : ∀ (a : ENNReal), LT.lt a 1 → LE.le (HMul.hMul a x) y
    this : Filter.Tendsto (fun x_1 => HMul.hMul x_1 x) (nhdsWithin 1 (Set.Iio 1))  …
    ⊢ LE.le x y
  -/
  rw [one_mul] at this
  /-
    x y : ENNReal
    h : ∀ (a : ENNReal), LT.lt a 1 → LE.le (HMul.hMul a x) y
    this : Filter.Tendsto (fun x_1 => HMul.hMul x_1 x) (nhdsWithin 1 (Set.Iio 1))  …
    ⊢ LE.le x y
  -/
  exact le_of_tendsto this (eventually_nhdsWithin_iff.2 <| Eventually.of_forall h)
  /-
    🎉 no goals
  -/


@[deprecated mul_iInf' (since := "2024-09-12")]
theorem iInf_mul_left' {ι} {f : ι → ℝ≥0∞} {a : ℝ≥0∞} (h : a = ∞ → ⨅ i, f i = 0 → ∃ i, f i = 0)
    (h0 : a = 0 → Nonempty ι) : ⨅ i, a * f i = a * ⨅ i, f i := .symm <| mul_iInf' h h0


@[deprecated mul_iInf (since := "2024-09-12")]
theorem iInf_mul_left {ι} [Nonempty ι] {f : ι → ℝ≥0∞} {a : ℝ≥0∞}
    (h : a = ∞ → ⨅ i, f i = 0 → ∃ i, f i = 0) : ⨅ i, a * f i = a * ⨅ i, f i :=
  .symm <| mul_iInf h


@[deprecated iInf_mul' (since := "2024-09-12")]
theorem iInf_mul_right' {ι} {f : ι → ℝ≥0∞} {a : ℝ≥0∞} (h : a = ∞ → ⨅ i, f i = 0 → ∃ i, f i = 0)
    (h0 : a = 0 → Nonempty ι) : ⨅ i, f i * a = (⨅ i, f i) * a := .symm <| iInf_mul' h h0


@[deprecated iInf_mul (since := "2024-09-12")]
theorem iInf_mul_right {ι} [Nonempty ι] {f : ι → ℝ≥0∞} {a : ℝ≥0∞}
    (h : a = ∞ → ⨅ i, f i = 0 → ∃ i, f i = 0) : ⨅ i, f i * a = (⨅ i, f i) * a := .symm <| iInf_mul h


@[deprecated inv_iInf (since := "2024-09-12")]
theorem inv_map_iInf {ι : Sort*} {x : ι → ℝ≥0∞} : (iInf x)⁻¹ = ⨆ i, (x i)⁻¹ :=
  OrderIso.invENNReal.map_iInf x


@[deprecated inv_iSup (since := "2024-09-12")]
theorem inv_map_iSup {ι : Sort*} {x : ι → ℝ≥0∞} : (iSup x)⁻¹ = ⨅ i, (x i)⁻¹ :=
  OrderIso.invENNReal.map_iSup x


theorem inv_limsup {ι : Sort _} {x : ι → ℝ≥0∞} {l : Filter ι} :
    (limsup x l)⁻¹ = liminf (fun i => (x i)⁻¹) l :=
  /-
    ι : Type u_4
    x : ι → ENNReal
    l : Filter ι
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l x
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  OrderIso.invENNReal.limsup_apply
  /-
    🎉 no goals
  -/


theorem inv_liminf {ι : Sort _} {x : ι → ℝ≥0∞} {l : Filter ι} :
    (liminf x l)⁻¹ = limsup (fun i => (x i)⁻¹) l :=
  /-
    ι : Type u_4
    x : ι → ENNReal
    l : Filter ι
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) l x
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  OrderIso.invENNReal.liminf_apply
  /-
    🎉 no goals
  -/


instance : ContinuousInv ℝ≥0∞ := ⟨OrderIso.invENNReal.continuous⟩


@[fun_prop]
protected theorem continuous_zpow : ∀ n : ℤ, Continuous (· ^ n : ℝ≥0∞ → ℝ≥0∞)
  | (n : ℕ) => mod_cast ENNReal.continuous_pow n
                     /-
                       n : Nat
                       ⊢ Continuous fun x => HPow.hPow x (Int.negSucc n)
                     -/
  | .negSucc n => by simpa using (ENNReal.continuous_pow _).inv
                     /-
                       🎉 no goals
                     -/


@[simp] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: generalize to `[InvolutiveInv _] [ContinuousInv _]`
protected theorem tendsto_inv_iff {f : Filter α} {m : α → ℝ≥0∞} {a : ℝ≥0∞} :
    Tendsto (fun x => (m x)⁻¹) f (𝓝 a⁻¹) ↔ Tendsto m f (𝓝 a) :=
               /-
                 α : Type u_1
                 f : Filter α
                 m : α → ENNReal
                 a : ENNReal
                 h : Filter.Tendsto (fun x => Inv.inv (m x)) f (nhds (Inv.inv a))
                 ⊢ Filter.Tendsto m f (nhds a)
               -/
  ⟨fun h => by simpa only [inv_inv] using Tendsto.inv h, Tendsto.inv⟩
               /-
                 🎉 no goals
               -/


protected theorem Tendsto.div {f : Filter α} {ma : α → ℝ≥0∞} {mb : α → ℝ≥0∞} {a b : ℝ≥0∞}
    (hma : Tendsto ma f (𝓝 a)) (ha : a ≠ 0 ∨ b ≠ 0) (hmb : Tendsto mb f (𝓝 b))
    (hb : b ≠ ∞ ∨ a ≠ ∞) : Tendsto (fun a => ma a / mb a) f (𝓝 (a / b)) := by
  /-
    α : Type u_1
    f : Filter α
    ma mb : α → ENNReal
    a b : ENNReal
    hma : Filter.Tendsto ma f (nhds a)
    ha : Or (Ne a 0) (Ne b 0)
    hmb : Filter.Tendsto mb f (nhds b)
    hb : Or (Ne b Top.top) (Ne a Top.top)
    ⊢ Filter.Tendsto (fun a => HDiv.hDiv (ma a) (mb a)) f (nhds (HDiv.hDiv a b))
  -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
  apply Tendsto.mul hma _ (ENNReal.tendsto_inv_iff.2 hmb) _ <;> simp [ha, hb]
                                                                /-
                                                                  🎉 no goals
                                                                -/


protected theorem Tendsto.const_div {f : Filter α} {m : α → ℝ≥0∞} {a b : ℝ≥0∞}
    (hm : Tendsto m f (𝓝 b)) (hb : b ≠ ∞ ∨ a ≠ ∞) : Tendsto (fun b => a / m b) f (𝓝 (a / b)) := by
  /-
    α : Type u_1
    f : Filter α
    m : α → ENNReal
    a b : ENNReal
    hm : Filter.Tendsto m f (nhds b)
    hb : Or (Ne b Top.top) (Ne a Top.top)
    ⊢ Filter.Tendsto (fun b => HDiv.hDiv a (m b)) f (nhds (HDiv.hDiv a b))
  -/
  apply Tendsto.const_mul (ENNReal.tendsto_inv_iff.2 hm)
  /-
    α : Type u_1
    f : Filter α
    m : α → ENNReal
    a b : ENNReal
    hm : Filter.Tendsto m f (nhds b)
    hb : Or (Ne b Top.top) (Ne a Top.top)
    ⊢ Or (Ne (Inv.inv b) 0) (Ne a Top.top)
  -/
  simp [hb]
  /-
    🎉 no goals
  -/


protected theorem Tendsto.div_const {f : Filter α} {m : α → ℝ≥0∞} {a b : ℝ≥0∞}
    (hm : Tendsto m f (𝓝 a)) (ha : a ≠ 0 ∨ b ≠ 0) : Tendsto (fun x => m x / b) f (𝓝 (a / b)) := by
  /-
    α : Type u_1
    f : Filter α
    m : α → ENNReal
    a b : ENNReal
    hm : Filter.Tendsto m f (nhds a)
    ha : Or (Ne a 0) (Ne b 0)
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (m x) b) f (nhds (HDiv.hDiv a b))
  -/
  apply Tendsto.mul_const hm
  /-
    α : Type u_1
    f : Filter α
    m : α → ENNReal
    a b : ENNReal
    hm : Filter.Tendsto m f (nhds a)
    ha : Or (Ne a 0) (Ne b 0)
    ⊢ Or (Ne a 0) (Ne (Inv.inv b) Top.top)
  -/
  simp [ha]
  /-
    🎉 no goals
  -/


protected theorem tendsto_inv_nat_nhds_zero : Tendsto (fun n : ℕ => (n : ℝ≥0∞)⁻¹) atTop (𝓝 0) :=
  ENNReal.inv_top ▸ ENNReal.tendsto_inv_iff.2 tendsto_nat_nhds_top


protected theorem tendsto_coe_sub {b : ℝ≥0∞} :
    Tendsto (fun b : ℝ≥0∞ => ↑r - b) (𝓝 b) (𝓝 (↑r - b)) :=
  continuous_nnreal_sub.tendsto _


theorem exists_countable_dense_no_zero_top :
    ∃ s : Set ℝ≥0∞, s.Countable ∧ Dense s ∧ 0 ∉ s ∧ ∞ ∉ s := by
  obtain ⟨s, s_count, s_dense, hs⟩ :
    ∃ s : Set ℝ≥0∞, s.Countable ∧ Dense s ∧ (∀ x, IsBot x → x ∉ s) ∧ ∀ x, IsTop x → x ∉ s :=
    exists_countable_dense_no_bot_top ℝ≥0∞
  /-
    case intro.intro.intro
    s : Set ENNReal
    s_count : s.Countable
    s_dense : Dense s
    hs : And (∀ (x : ENNReal), IsBot x → Not (Membership.mem s x)) (∀ (x : ENNReal …
    ⊢ Exists fun s => And s.Countable (And (Dense s) (And (Not (Membership.mem s 0 …
  -/
  exact ⟨s, s_count, s_dense, fun h => hs.1 0 (by simp) h, fun h => hs.2 ∞ (by simp) h⟩
  /-
    🎉 no goals
  -/


@[deprecated ofReal_iInf (since := "2024-09-12")]
theorem ofReal_cinfi (f : α → ℝ) [Nonempty α] :
    ENNReal.ofReal (⨅ i, f i) = ⨅ i, ENNReal.ofReal (f i) := by
  /-
    α : Type u_1
    f : α → Real
    inst✝ : Nonempty α
    ⊢ Eq (ENNReal.ofReal (iInf fun i => f i)) (iInf fun i => ENNReal.ofReal (f i))
  -/
  by_cases hf : BddBelow (range f)
  · exact
      Monotone.map_ciInf_of_continuousAt ENNReal.continuous_ofReal.continuousAt
        (fun i j hij => ENNReal.ofReal_le_ofReal hij) hf
    /-
      case neg
      α : Type u_1
      f : α → Real
      inst✝ : Nonempty α
      hf : Not (BddBelow (Set.range f))
      ⊢ Eq (ENNReal.ofReal (iInf fun i => f i)) (iInf fun i => ENNReal.ofReal (f i))
    -/
  · symm
    /-
      case neg
      α : Type u_1
      f : α → Real
      inst✝ : Nonempty α
      hf : Not (BddBelow (Set.range f))
      ⊢ Eq (iInf fun i => ENNReal.ofReal (f i)) (ENNReal.ofReal (iInf fun i => f i))
    -/
    rw [Real.iInf_of_not_bddBelow hf, ENNReal.ofReal_zero, ← ENNReal.bot_eq_zero, iInf_eq_bot]
    /-
      case neg
      α : Type u_1
      f : α → Real
      inst✝ : Nonempty α
      hf : Not (BddBelow (Set.range f))
      ⊢ ∀ (b : ENNReal), GT.gt b Bot.bot → Exists fun i => LT.lt (ENNReal.ofReal (f  …
    -/
    obtain ⟨y, hy_mem, hy_neg⟩ := not_bddBelow_iff.mp hf 0
    /-
      case neg.intro.intro
      α : Type u_1
      f : α → Real
      inst✝ : Nonempty α
      hf : Not (BddBelow (Set.range f))
      y : Real
      hy_mem : Membership.mem (Set.range f) y
      hy_neg : LT.lt y 0
      ⊢ ∀ (b : ENNReal), GT.gt b Bot.bot → Exists fun i => LT.lt (ENNReal.ofReal (f  …
    -/
    obtain ⟨i, rfl⟩ := mem_range.mpr hy_mem
    /-
      case neg.intro.intro.intro
      α : Type u_1
      f : α → Real
      inst✝ : Nonempty α
      hf : Not (BddBelow (Set.range f))
      i : α
      hy_mem : Membership.mem (Set.range f) (f i)
      hy_neg : LT.lt (f i) 0
      ⊢ ∀ (b : ENNReal), GT.gt b Bot.bot → Exists fun i => LT.lt (ENNReal.ofReal (f  …
    -/
    refine fun x hx => ⟨i, ?_⟩
    /-
      case neg.intro.intro.intro
      α : Type u_1
      f : α → Real
      inst✝ : Nonempty α
      hf : Not (BddBelow (Set.range f))
      i : α
      hy_mem : Membership.mem (Set.range f) (f i)
      hy_neg : LT.lt (f i) 0
      x : ENNReal
      hx : GT.gt x Bot.bot
      ⊢ LT.lt (ENNReal.ofReal (f i)) x
    -/
    rwa [ENNReal.ofReal_of_nonpos hy_neg.le]
    /-
      🎉 no goals
    -/


theorem exists_frequently_lt_of_liminf_ne_top {ι : Type*} {l : Filter ι} {x : ι → ℝ}
    (hx : liminf (fun n => (Real.nnabs (x n) : ℝ≥0∞)) l ≠ ∞) : ∃ R, ∃ᶠ n in l, x n < R := by
  /-
    ι : Type u_4
    l : Filter ι
    x : ι → Real
    hx : Ne (Filter.liminf (fun n => ↑(Real.nnabs (x n))) l) Top.top
    ⊢ Exists fun R => Filter.Frequently (fun n => LT.lt (x n) R) l
  -/
  by_contra h
  /-
    ι : Type u_4
    l : Filter ι
    x : ι → Real
    hx : Ne (Filter.liminf (fun n => ↑(Real.nnabs (x n))) l) Top.top
    h : Not (Exists fun R => Filter.Frequently (fun n => LT.lt (x n) R) l)
    ⊢ False
  -/
  simp_rw [not_exists, not_frequently, not_lt] at h
  /-
    ι : Type u_4
    l : Filter ι
    x : ι → Real
    hx : Ne (Filter.liminf (fun n => ↑(Real.nnabs (x n))) l) Top.top
    h : ∀ (x_1 : Real), Filter.Eventually (fun x_2 => LE.le x_1 (x x_2)) l
    ⊢ False
  -/
  refine hx (ENNReal.eq_top_of_forall_nnreal_le fun r => le_limsInf_of_le (by isBoundedDefault) ?_)
  /-
    ι : Type u_4
    l : Filter ι
    x : ι → Real
    hx : Ne (Filter.liminf (fun n => ↑(Real.nnabs (x n))) l) Top.top
    h : ∀ (x_1 : Real), Filter.Eventually (fun x_2 => LE.le x_1 (x x_2)) l
    r : NNReal
    ⊢ Filter.Eventually (fun n => LE.le (↑r) n) (Filter.map (fun n => ↑(Real.nnabs …
  -/
  simp only [eventually_map, ENNReal.coe_le_coe]
  /-
    ι : Type u_4
    l : Filter ι
    x : ι → Real
    hx : Ne (Filter.liminf (fun n => ↑(Real.nnabs (x n))) l) Top.top
    h : ∀ (x_1 : Real), Filter.Eventually (fun x_2 => LE.le x_1 (x x_2)) l
    r : NNReal
    ⊢ Filter.Eventually (fun a => LE.le r (Real.nnabs (x a))) l
  -/
  filter_upwards [h r] with i hi using hi.trans (le_abs_self (x i))
  /-
    🎉 no goals
  -/


theorem exists_frequently_lt_of_liminf_ne_top' {ι : Type*} {l : Filter ι} {x : ι → ℝ}
    (hx : liminf (fun n => (Real.nnabs (x n) : ℝ≥0∞)) l ≠ ∞) : ∃ R, ∃ᶠ n in l, R < x n := by
  /-
    ι : Type u_4
    l : Filter ι
    x : ι → Real
    hx : Ne (Filter.liminf (fun n => ↑(Real.nnabs (x n))) l) Top.top
    ⊢ Exists fun R => Filter.Frequently (fun n => LT.lt R (x n)) l
  -/
  by_contra h
  /-
    ι : Type u_4
    l : Filter ι
    x : ι → Real
    hx : Ne (Filter.liminf (fun n => ↑(Real.nnabs (x n))) l) Top.top
    h : Not (Exists fun R => Filter.Frequently (fun n => LT.lt R (x n)) l)
    ⊢ False
  -/
  simp_rw [not_exists, not_frequently, not_lt] at h
  /-
    ι : Type u_4
    l : Filter ι
    x : ι → Real
    hx : Ne (Filter.liminf (fun n => ↑(Real.nnabs (x n))) l) Top.top
    h : ∀ (x_1 : Real), Filter.Eventually (fun x_2 => LE.le (x x_2) x_1) l
    ⊢ False
  -/
  refine hx (ENNReal.eq_top_of_forall_nnreal_le fun r => le_limsInf_of_le (by isBoundedDefault) ?_)
  /-
    ι : Type u_4
    l : Filter ι
    x : ι → Real
    hx : Ne (Filter.liminf (fun n => ↑(Real.nnabs (x n))) l) Top.top
    h : ∀ (x_1 : Real), Filter.Eventually (fun x_2 => LE.le (x x_2) x_1) l
    r : NNReal
    ⊢ Filter.Eventually (fun n => LE.le (↑r) n) (Filter.map (fun n => ↑(Real.nnabs …
  -/
  simp only [eventually_map, ENNReal.coe_le_coe]
  /-
    ι : Type u_4
    l : Filter ι
    x : ι → Real
    hx : Ne (Filter.liminf (fun n => ↑(Real.nnabs (x n))) l) Top.top
    h : ∀ (x_1 : Real), Filter.Eventually (fun x_2 => LE.le (x x_2) x_1) l
    r : NNReal
    ⊢ Filter.Eventually (fun a => LE.le r (Real.nnabs (x a))) l
  -/
  filter_upwards [h (-r)] with i hi using(le_neg.1 hi).trans (neg_le_abs _)
  /-
    🎉 no goals
  -/


theorem exists_upcrossings_of_not_bounded_under {ι : Type*} {l : Filter ι} {x : ι → ℝ}
    (hf : liminf (fun i => (Real.nnabs (x i) : ℝ≥0∞)) l ≠ ∞)
    (hbdd : ¬IsBoundedUnder (· ≤ ·) l fun i => |x i|) :
    ∃ a b : ℚ, a < b ∧ (∃ᶠ i in l, x i < a) ∧ ∃ᶠ i in l, ↑b < x i := by
  /-
    ι : Type u_4
    l : Filter ι
    x : ι → Real
    hf : Ne (Filter.liminf (fun i => ↑(Real.nnabs (x i))) l) Top.top
    hbdd : Not (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun i => abs (x …
    ⊢ Exists fun a => Exists fun b => And (LT.lt a b) (And (Filter.Frequently (fun …
  -/
  rw [isBoundedUnder_le_abs, not_and_or] at hbdd
  /-
    ι : Type u_4
    l : Filter ι
    x : ι → Real
    hf : Ne (Filter.liminf (fun i => ↑(Real.nnabs (x i))) l) Top.top
    hbdd : Or (Not (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l x)) (Not (F …
    ⊢ Exists fun a => Exists fun b => And (LT.lt a b) (And (Filter.Frequently (fun …
  -/
  obtain hbdd | hbdd := hbdd
    /-
      case inl
      ι : Type u_4
      l : Filter ι
      x : ι → Real
      hf : Ne (Filter.liminf (fun i => ↑(Real.nnabs (x i))) l) Top.top
      hbdd : Not (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l x)
      ⊢ Exists fun a => Exists fun b => And (LT.lt a b) (And (Filter.Frequently (fun …
    -/
  · obtain ⟨R, hR⟩ := exists_frequently_lt_of_liminf_ne_top hf
    /-
      case inl.intro
      ι : Type u_4
      l : Filter ι
      x : ι → Real
      hf : Ne (Filter.liminf (fun i => ↑(Real.nnabs (x i))) l) Top.top
      hbdd : Not (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l x)
      R : Real
      hR : Filter.Frequently (fun n => LT.lt (x n) R) l
      ⊢ Exists fun a => Exists fun b => And (LT.lt a b) (And (Filter.Frequently (fun …
    -/
    obtain ⟨q, hq⟩ := exists_rat_gt R
    /-
      case inl.intro.intro
      ι : Type u_4
      l : Filter ι
      x : ι → Real
      hf : Ne (Filter.liminf (fun i => ↑(Real.nnabs (x i))) l) Top.top
      hbdd : Not (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l x)
      R : Real
      hR : Filter.Frequently (fun n => LT.lt (x n) R) l
      q : Rat
      hq : LT.lt R ↑q
      ⊢ Exists fun a => Exists fun b => And (LT.lt a b) (And (Filter.Frequently (fun …
    -/
    refine ⟨q, q + 1, (lt_add_iff_pos_right _).2 zero_lt_one, ?_, ?_⟩
      /-
        case inl.intro.intro.refine_1
        ι : Type u_4
        l : Filter ι
        x : ι → Real
        hf : Ne (Filter.liminf (fun i => ↑(Real.nnabs (x i))) l) Top.top
        hbdd : Not (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l x)
        R : Real
        hR : Filter.Frequently (fun n => LT.lt (x n) R) l
        q : Rat
        hq : LT.lt R ↑q
        ⊢ Filter.Frequently (fun i => LT.lt (x i) ↑q) l
      -/
    · refine fun hcon => hR ?_
      /-
        case inl.intro.intro.refine_1
        ι : Type u_4
        l : Filter ι
        x : ι → Real
        hf : Ne (Filter.liminf (fun i => ↑(Real.nnabs (x i))) l) Top.top
        hbdd : Not (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l x)
        R : Real
        hR : Filter.Frequently (fun n => LT.lt (x n) R) l
        q : Rat
        hq : LT.lt R ↑q
        hcon : Filter.Eventually (fun x_1 => Not ((fun i => LT.lt (x i) ↑q) x_1)) l
        ⊢ Filter.Eventually (fun x_1 => Not ((fun n => LT.lt (x n) R) x_1)) l
      -/
      filter_upwards [hcon] with x hx using not_lt.2 (lt_of_lt_of_le hq (not_lt.1 hx)).le
      /-
        🎉 no goals
      -/
    · simp only [IsBoundedUnder, IsBounded, eventually_map, eventually_atTop, not_exists,
        not_forall, not_le, exists_prop] at hbdd
      /-
        case inl.intro.intro.refine_2
        ι : Type u_4
        l : Filter ι
        x : ι → Real
        hf : Ne (Filter.liminf (fun i => ↑(Real.nnabs (x i))) l) Top.top
        R : Real
        hR : Filter.Frequently (fun n => LT.lt (x n) R) l
        q : Rat
        hq : LT.lt R ↑q
        hbdd : ∀ (x_1 : Real), Not (Filter.Eventually (fun a => LE.le (x a) x_1) l)
        ⊢ Filter.Frequently (fun i => LT.lt (↑(HAdd.hAdd q 1)) (x i)) l
      -/
      refine fun hcon => hbdd ↑(q + 1) ?_
      /-
        case inl.intro.intro.refine_2
        ι : Type u_4
        l : Filter ι
        x : ι → Real
        hf : Ne (Filter.liminf (fun i => ↑(Real.nnabs (x i))) l) Top.top
        R : Real
        hR : Filter.Frequently (fun n => LT.lt (x n) R) l
        q : Rat
        hq : LT.lt R ↑q
        hbdd : ∀ (x_1 : Real), Not (Filter.Eventually (fun a => LE.le (x a) x_1) l)
        hcon : Filter.Eventually (fun x_1 => Not ((fun i => LT.lt (↑(HAdd.hAdd q 1)) ( …
        ⊢ Filter.Eventually (fun a => LE.le (x a) ↑(HAdd.hAdd q 1)) l
      -/
      filter_upwards [hcon] with x hx using not_lt.1 hx
      /-
        🎉 no goals
      -/
    /-
      case inr
      ι : Type u_4
      l : Filter ι
      x : ι → Real
      hf : Ne (Filter.liminf (fun i => ↑(Real.nnabs (x i))) l) Top.top
      hbdd : Not (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) l x)
      ⊢ Exists fun a => Exists fun b => And (LT.lt a b) (And (Filter.Frequently (fun …
    -/
  · obtain ⟨R, hR⟩ := exists_frequently_lt_of_liminf_ne_top' hf
    /-
      case inr.intro
      ι : Type u_4
      l : Filter ι
      x : ι → Real
      hf : Ne (Filter.liminf (fun i => ↑(Real.nnabs (x i))) l) Top.top
      hbdd : Not (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) l x)
      R : Real
      hR : Filter.Frequently (fun n => LT.lt R (x n)) l
      ⊢ Exists fun a => Exists fun b => And (LT.lt a b) (And (Filter.Frequently (fun …
    -/
    obtain ⟨q, hq⟩ := exists_rat_lt R
    /-
      case inr.intro.intro
      ι : Type u_4
      l : Filter ι
      x : ι → Real
      hf : Ne (Filter.liminf (fun i => ↑(Real.nnabs (x i))) l) Top.top
      hbdd : Not (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) l x)
      R : Real
      hR : Filter.Frequently (fun n => LT.lt R (x n)) l
      q : Rat
      hq : LT.lt (↑q) R
      ⊢ Exists fun a => Exists fun b => And (LT.lt a b) (And (Filter.Frequently (fun …
    -/
    refine ⟨q - 1, q, (sub_lt_self_iff _).2 zero_lt_one, ?_, ?_⟩
    · simp only [IsBoundedUnder, IsBounded, eventually_map, eventually_atTop, not_exists,
        not_forall, not_le, exists_prop] at hbdd
      /-
        case inr.intro.intro.refine_1
        ι : Type u_4
        l : Filter ι
        x : ι → Real
        hf : Ne (Filter.liminf (fun i => ↑(Real.nnabs (x i))) l) Top.top
        R : Real
        hR : Filter.Frequently (fun n => LT.lt R (x n)) l
        q : Rat
        hq : LT.lt (↑q) R
        hbdd : ∀ (x_1 : Real), Not (Filter.Eventually (fun a => GE.ge (x a) x_1) l)
        ⊢ Filter.Frequently (fun i => LT.lt (x i) ↑(HSub.hSub q 1)) l
      -/
      refine fun hcon => hbdd ↑(q - 1) ?_
      /-
        case inr.intro.intro.refine_1
        ι : Type u_4
        l : Filter ι
        x : ι → Real
        hf : Ne (Filter.liminf (fun i => ↑(Real.nnabs (x i))) l) Top.top
        R : Real
        hR : Filter.Frequently (fun n => LT.lt R (x n)) l
        q : Rat
        hq : LT.lt (↑q) R
        hbdd : ∀ (x_1 : Real), Not (Filter.Eventually (fun a => GE.ge (x a) x_1) l)
        hcon : Filter.Eventually (fun x_1 => Not ((fun i => LT.lt (x i) ↑(HSub.hSub q  …
        ⊢ Filter.Eventually (fun a => GE.ge (x a) ↑(HSub.hSub q 1)) l
      -/
      filter_upwards [hcon] with x hx using not_lt.1 hx
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.intro.refine_2
        ι : Type u_4
        l : Filter ι
        x : ι → Real
        hf : Ne (Filter.liminf (fun i => ↑(Real.nnabs (x i))) l) Top.top
        hbdd : Not (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) l x)
        R : Real
        hR : Filter.Frequently (fun n => LT.lt R (x n)) l
        q : Rat
        hq : LT.lt (↑q) R
        ⊢ Filter.Frequently (fun i => LT.lt (↑q) (x i)) l
      -/
    · refine fun hcon => hR ?_
      /-
        case inr.intro.intro.refine_2
        ι : Type u_4
        l : Filter ι
        x : ι → Real
        hf : Ne (Filter.liminf (fun i => ↑(Real.nnabs (x i))) l) Top.top
        hbdd : Not (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) l x)
        R : Real
        hR : Filter.Frequently (fun n => LT.lt R (x n)) l
        q : Rat
        hq : LT.lt (↑q) R
        hcon : Filter.Eventually (fun x_1 => Not ((fun i => LT.lt (↑q) (x i)) x_1)) l
        ⊢ Filter.Eventually (fun x_1 => Not ((fun n => LT.lt R (x n)) x_1)) l
      -/
      filter_upwards [hcon] with x hx using not_lt.2 ((not_lt.1 hx).trans hq.le)
      /-
        🎉 no goals
      -/


@[norm_cast]
protected theorem hasSum_coe {f : α → ℝ≥0} {r : ℝ≥0} :
    HasSum (fun a => (f a : ℝ≥0∞)) ↑r ↔ HasSum f r := by
  /-
    α : Type u_1
    f : α → NNReal
    r : NNReal
    ⊢ Iff (HasSum (fun a => ↑(f a)) ↑r) (HasSum f r)
  -/
  simp only [HasSum, ← coe_finset_sum, tendsto_coe]
  /-
    🎉 no goals
  -/


protected theorem tsum_coe_eq {f : α → ℝ≥0} (h : HasSum f r) : (∑' a, (f a : ℝ≥0∞)) = r :=
  (ENNReal.hasSum_coe.2 h).tsum_eq


protected theorem coe_tsum {f : α → ℝ≥0} : Summable f → ↑(tsum f) = ∑' a, (f a : ℝ≥0∞)
                  /-
                    α : Type u_1
                    f : α → NNReal
                    r : NNReal
                    hr : HasSum f r
                    ⊢ Eq (↑(tsum f)) (tsum fun a => ↑(f a))
                  -/
  | ⟨r, hr⟩ => by rw [hr.tsum_eq, ENNReal.tsum_coe_eq hr]
                  /-
                    🎉 no goals
                  -/


protected theorem hasSum : HasSum f (⨆ s : Finset α, ∑ a ∈ s, f a) :=
  tendsto_atTop_iSup fun _ _ => Finset.sum_le_sum_of_subset


@[simp]
protected theorem summable : Summable f :=
  ⟨_, ENNReal.hasSum⟩


theorem tsum_coe_ne_top_iff_summable {f : β → ℝ≥0} : (∑' b, (f b : ℝ≥0∞)) ≠ ∞ ↔ Summable f := by
  /-
    β : Type u_2
    f : β → NNReal
    ⊢ Iff (Ne (tsum fun b => ↑(f b)) Top.top) (Summable f)
  -/
  refine ⟨fun h => ?_, fun h => ENNReal.coe_tsum h ▸ ENNReal.coe_ne_top⟩
  /-
    β : Type u_2
    f : β → NNReal
    h : Ne (tsum fun b => ↑(f b)) Top.top
    ⊢ Summable f
  -/
  lift ∑' b, (f b : ℝ≥0∞) to ℝ≥0 using h with a ha
  /-
    case intro
    β : Type u_2
    f : β → NNReal
    a : NNReal
    ha : Eq (↑a) (tsum fun b => ↑(f b))
    ⊢ Summable f
  -/
  refine ⟨a, ENNReal.hasSum_coe.1 ?_⟩
  /-
    case intro
    β : Type u_2
    f : β → NNReal
    a : NNReal
    ha : Eq (↑a) (tsum fun b => ↑(f b))
    ⊢ HasSum (fun a => ↑(f a)) ↑a
  -/
  rw [ha]
  /-
    case intro
    β : Type u_2
    f : β → NNReal
    a : NNReal
    ha : Eq (↑a) (tsum fun b => ↑(f b))
    ⊢ HasSum (fun a => ↑(f a)) (tsum fun b => ↑(f b))
  -/
  exact ENNReal.summable.hasSum
  /-
    🎉 no goals
  -/


protected theorem tsum_eq_iSup_sum : ∑' a, f a = ⨆ s : Finset α, ∑ a ∈ s, f a :=
  ENNReal.hasSum.tsum_eq


protected theorem tsum_eq_iSup_sum' {ι : Type*} (s : ι → Finset α) (hs : ∀ t, ∃ i, t ⊆ s i) :
    ∑' a, f a = ⨆ i, ∑ a ∈ s i, f a := by
  /-
    α : Type u_1
    f : α → ENNReal
    ι : Type u_4
    s : ι → Finset α
    hs : ∀ (t : Finset α), Exists fun i => HasSubset.Subset t (s i)
    ⊢ Eq (tsum fun a => f a) (iSup fun i => (s i).sum fun a => f a)
  -/
  rw [ENNReal.tsum_eq_iSup_sum]
  /-
    α : Type u_1
    f : α → ENNReal
    ι : Type u_4
    s : ι → Finset α
    hs : ∀ (t : Finset α), Exists fun i => HasSubset.Subset t (s i)
    ⊢ Eq (iSup fun s => s.sum fun a => f a) (iSup fun i => (s i).sum fun a => f a)
  -/
  symm
  /-
    α : Type u_1
    f : α → ENNReal
    ι : Type u_4
    s : ι → Finset α
    hs : ∀ (t : Finset α), Exists fun i => HasSubset.Subset t (s i)
    ⊢ Eq (iSup fun i => (s i).sum fun a => f a) (iSup fun s => s.sum fun a => f a)
  -/
  change ⨆ i : ι, (fun t : Finset α => ∑ a ∈ t, f a) (s i) = ⨆ s : Finset α, ∑ a ∈ s, f a
  /-
    α : Type u_1
    f : α → ENNReal
    ι : Type u_4
    s : ι → Finset α
    hs : ∀ (t : Finset α), Exists fun i => HasSubset.Subset t (s i)
    ⊢ Eq (iSup fun i => (fun t => t.sum fun a => f a) (s i)) (iSup fun s => s.sum  …
  -/
  exact (Finset.sum_mono_set f).iSup_comp_eq hs
  /-
    🎉 no goals
  -/


protected theorem tsum_sigma {β : α → Type*} (f : ∀ a, β a → ℝ≥0∞) :
    ∑' p : Σa, β a, f p.1 p.2 = ∑' (a) (b), f a b :=
  tsum_sigma' (fun _ => ENNReal.summable) ENNReal.summable


protected theorem tsum_sigma' {β : α → Type*} (f : (Σa, β a) → ℝ≥0∞) :
    ∑' p : Σa, β a, f p = ∑' (a) (b), f ⟨a, b⟩ :=
  tsum_sigma' (fun _ => ENNReal.summable) ENNReal.summable


protected theorem tsum_prod {f : α → β → ℝ≥0∞} : ∑' p : α × β, f p.1 p.2 = ∑' (a) (b), f a b :=
  tsum_prod' ENNReal.summable fun _ => ENNReal.summable


protected theorem tsum_prod' {f : α × β → ℝ≥0∞} : ∑' p : α × β, f p = ∑' (a) (b), f (a, b) :=
  tsum_prod' ENNReal.summable fun _ => ENNReal.summable


protected theorem tsum_comm {f : α → β → ℝ≥0∞} : ∑' a, ∑' b, f a b = ∑' b, ∑' a, f a b :=
  tsum_comm' ENNReal.summable (fun _ => ENNReal.summable) fun _ => ENNReal.summable


protected theorem tsum_add : ∑' a, (f a + g a) = ∑' a, f a + ∑' a, g a :=
  tsum_add ENNReal.summable ENNReal.summable


protected theorem tsum_le_tsum (h : ∀ a, f a ≤ g a) : ∑' a, f a ≤ ∑' a, g a :=
  tsum_le_tsum h ENNReal.summable ENNReal.summable


@[gcongr]
protected theorem _root_.GCongr.ennreal_tsum_le_tsum (h : ∀ a, f a ≤ g a) : tsum f ≤ tsum g :=
  ENNReal.tsum_le_tsum h


protected theorem sum_le_tsum {f : α → ℝ≥0∞} (s : Finset α) : ∑ x ∈ s, f x ≤ ∑' x, f x :=
  sum_le_tsum s (fun _ _ => zero_le _) ENNReal.summable


protected theorem tsum_eq_iSup_nat' {f : ℕ → ℝ≥0∞} {N : ℕ → ℕ} (hN : Tendsto N atTop atTop) :
    ∑' i : ℕ, f i = ⨆ i : ℕ, ∑ a ∈ Finset.range (N i), f a :=
  ENNReal.tsum_eq_iSup_sum' _ fun t =>
    let ⟨n, hn⟩ := t.exists_nat_subset_range
    let ⟨k, _, hk⟩ := exists_le_of_tendsto_atTop hN 0 n
    ⟨k, Finset.Subset.trans hn (Finset.range_mono hk)⟩


protected theorem tsum_eq_iSup_nat {f : ℕ → ℝ≥0∞} :
    ∑' i : ℕ, f i = ⨆ i : ℕ, ∑ a ∈ Finset.range i, f a :=
  ENNReal.tsum_eq_iSup_sum' _ Finset.exists_nat_subset_range


protected theorem tsum_eq_liminf_sum_nat {f : ℕ → ℝ≥0∞} :
    ∑' i, f i = liminf (fun n => ∑ i ∈ Finset.range n, f i) atTop :=
  ENNReal.summable.hasSum.tendsto_sum_nat.liminf_eq.symm


protected theorem tsum_eq_limsup_sum_nat {f : ℕ → ℝ≥0∞} :
    ∑' i, f i = limsup (fun n => ∑ i ∈ Finset.range n, f i) atTop :=
  ENNReal.summable.hasSum.tendsto_sum_nat.limsup_eq.symm


protected theorem le_tsum (a : α) : f a ≤ ∑' a, f a :=
  le_tsum' ENNReal.summable a


@[simp]
protected theorem tsum_eq_zero : ∑' i, f i = 0 ↔ ∀ i, f i = 0 :=
  tsum_eq_zero_iff ENNReal.summable


protected theorem tsum_eq_top_of_eq_top : (∃ a, f a = ∞) → ∑' a, f a = ∞
  | ⟨a, ha⟩ => top_unique <| ha ▸ ENNReal.le_tsum a


protected theorem lt_top_of_tsum_ne_top {a : α → ℝ≥0∞} (tsum_ne_top : ∑' i, a i ≠ ∞) (j : α) :
    a j < ∞ := by
  /-
    α : Type u_1
    a : α → ENNReal
    tsum_ne_top : Ne (tsum fun i => a i) Top.top
    j : α
    ⊢ LT.lt (a j) Top.top
  -/
  contrapose! tsum_ne_top with h
  /-
    α : Type u_1
    a : α → ENNReal
    j : α
    h : LE.le Top.top (a j)
    ⊢ Eq (tsum fun i => a i) Top.top
  -/
  exact ENNReal.tsum_eq_top_of_eq_top ⟨j, top_unique h⟩
  /-
    🎉 no goals
  -/


@[simp]
protected theorem tsum_top [Nonempty α] : ∑' _ : α, ∞ = ∞ :=
  let ⟨a⟩ := ‹Nonempty α›
  ENNReal.tsum_eq_top_of_eq_top ⟨a, rfl⟩


theorem tsum_const_eq_top_of_ne_zero {α : Type*} [Infinite α] {c : ℝ≥0∞} (hc : c ≠ 0) :
    ∑' _ : α, c = ∞ := by
  have A : Tendsto (fun n : ℕ => (n : ℝ≥0∞) * c) atTop (𝓝 (∞ * c)) := by
    apply ENNReal.Tendsto.mul_const tendsto_nat_nhds_top
    simp only [true_or, top_ne_zero, Ne, not_false_iff]
  have B : ∀ n : ℕ, (n : ℝ≥0∞) * c ≤ ∑' _ : α, c := fun n => by
    rcases Infinite.exists_subset_card_eq α n with ⟨s, hs⟩
    simpa [hs] using @ENNReal.sum_le_tsum α (fun _ => c) s
  /-
    α : Type u_4
    inst✝ : Infinite α
    c : ENNReal
    hc : Ne c 0
    A : Filter.Tendsto (fun n => HMul.hMul (↑n) c) Filter.atTop (nhds (HMul.hMul T …
    B : ∀ (n : Nat), LE.le (HMul.hMul (↑n) c) (tsum fun x => c)
    ⊢ Eq (tsum fun x => c) Top.top
  -/
  simpa [hc] using le_of_tendsto' A B
  /-
    🎉 no goals
  -/


protected theorem ne_top_of_tsum_ne_top (h : ∑' a, f a ≠ ∞) (a : α) : f a ≠ ∞ := fun ha =>
  h <| ENNReal.tsum_eq_top_of_eq_top ⟨a, ha⟩


protected theorem tsum_mul_left : ∑' i, a * f i = a * ∑' i, f i := by
  /-
    α : Type u_1
    a : ENNReal
    f : α → ENNReal
    ⊢ Eq (tsum fun i => HMul.hMul a (f i)) (HMul.hMul a (tsum fun i => f i))
  -/
  by_cases hf : ∀ i, f i = 0
    /-
      case pos
      α : Type u_1
      a : ENNReal
      f : α → ENNReal
      hf : ∀ (i : α), Eq (f i) 0
      ⊢ Eq (tsum fun i => HMul.hMul a (f i)) (HMul.hMul a (tsum fun i => f i))
    -/
  · simp [hf]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      a : ENNReal
      f : α → ENNReal
      hf : Not (∀ (i : α), Eq (f i) 0)
      ⊢ Eq (tsum fun i => HMul.hMul a (f i)) (HMul.hMul a (tsum fun i => f i))
    -/
  · rw [← ENNReal.tsum_eq_zero] at hf
    have : Tendsto (fun s : Finset α => ∑ j ∈ s, a * f j) atTop (𝓝 (a * ∑' i, f i)) := by
      simp only [← Finset.mul_sum]
      exact ENNReal.Tendsto.const_mul ENNReal.summable.hasSum (Or.inl hf)
    /-
      case neg
      α : Type u_1
      a : ENNReal
      f : α → ENNReal
      hf : Not (Eq (tsum fun i => f i) 0)
      this : Filter.Tendsto (fun s => s.sum fun j => HMul.hMul a (f j)) Filter.atTop …
      ⊢ Eq (tsum fun i => HMul.hMul a (f i)) (HMul.hMul a (tsum fun i => f i))
    -/
    exact HasSum.tsum_eq this
    /-
      🎉 no goals
    -/


protected theorem tsum_mul_right : ∑' i, f i * a = (∑' i, f i) * a := by
  /-
    α : Type u_1
    a : ENNReal
    f : α → ENNReal
    ⊢ Eq (tsum fun i => HMul.hMul (f i) a) (HMul.hMul (tsum fun i => f i) a)
  -/
  simp [mul_comm, ENNReal.tsum_mul_left]
  /-
    🎉 no goals
  -/


protected theorem tsum_const_smul {R} [SMul R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞] (a : R) :
    ∑' i, a • f i = a • ∑' i, f i := by
  /-
    α : Type u_1
    f : α → ENNReal
    R : Type u_4
    inst✝¹ : SMul R ENNReal
    inst✝ : IsScalarTower R ENNReal ENNReal
    a : R
    ⊢ Eq (tsum fun i => HSMul.hSMul a (f i)) (HSMul.hSMul a (tsum fun i => f i))
  -/
  simpa only [smul_one_mul] using @ENNReal.tsum_mul_left _ (a • (1 : ℝ≥0∞)) _
  /-
    🎉 no goals
  -/


@[simp]
theorem tsum_iSup_eq {α : Type*} (a : α) {f : α → ℝ≥0∞} : (∑' b : α, ⨆ _ : a = b, f b) = f a :=
                                  /-
                                    α : Type u_4
                                    a : α
                                    f : α → ENNReal
                                    x✝ : α
                                    h : Ne x✝ a
                                    ⊢ Eq (iSup fun x => f x✝) 0
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
  (tsum_eq_single a fun _ h => by simp [h.symm]).trans <| by simp
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem hasSum_iff_tendsto_nat {f : ℕ → ℝ≥0∞} (r : ℝ≥0∞) :
    HasSum f r ↔ Tendsto (fun n : ℕ => ∑ i ∈ Finset.range n, f i) atTop (𝓝 r) := by
  /-
    f : Nat → ENNReal
    r : ENNReal
    ⊢ Iff (HasSum f r) (Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i …
  -/
  refine ⟨HasSum.tendsto_sum_nat, fun h => ?_⟩
  /-
    f : Nat → ENNReal
    r : ENNReal
    h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
    ⊢ HasSum f r
  -/
  rw [← iSup_eq_of_tendsto _ h, ← ENNReal.tsum_eq_iSup_nat]
    /-
      f : Nat → ENNReal
      r : ENNReal
      h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
      ⊢ HasSum f (tsum fun i => f i)
    -/
  · exact ENNReal.summable.hasSum
    /-
      🎉 no goals
    -/
    /-
      f : Nat → ENNReal
      r : ENNReal
      h : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop ( …
      ⊢ Monotone fun n => (Finset.range n).sum fun i => f i
    -/
  · exact fun s t hst => Finset.sum_le_sum_of_subset (Finset.range_subset.2 hst)
    /-
      🎉 no goals
    -/


theorem tendsto_nat_tsum (f : ℕ → ℝ≥0∞) :
    Tendsto (fun n : ℕ => ∑ i ∈ Finset.range n, f i) atTop (𝓝 (∑' n, f n)) := by
  /-
    f : Nat → ENNReal
    ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop (nh …
  -/
  rw [← hasSum_iff_tendsto_nat]
  /-
    f : Nat → ENNReal
    ⊢ HasSum f (tsum fun n => f n)
  -/
  exact ENNReal.summable.hasSum
  /-
    🎉 no goals
  -/


theorem toNNReal_apply_of_tsum_ne_top {α : Type*} {f : α → ℝ≥0∞} (hf : ∑' i, f i ≠ ∞) (x : α) :
    (((ENNReal.toNNReal ∘ f) x : ℝ≥0) : ℝ≥0∞) = f x :=
  coe_toNNReal <| ENNReal.ne_top_of_tsum_ne_top hf _


theorem summable_toNNReal_of_tsum_ne_top {α : Type*} {f : α → ℝ≥0∞} (hf : ∑' i, f i ≠ ∞) :
    Summable (ENNReal.toNNReal ∘ f) := by
  /-
    α : Type u_4
    f : α → ENNReal
    hf : Ne (tsum fun i => f i) Top.top
    ⊢ Summable (Function.comp ENNReal.toNNReal f)
  -/
  simpa only [← tsum_coe_ne_top_iff_summable, toNNReal_apply_of_tsum_ne_top hf] using hf
  /-
    🎉 no goals
  -/


theorem tendsto_cofinite_zero_of_tsum_ne_top {α} {f : α → ℝ≥0∞} (hf : ∑' x, f x ≠ ∞) :
    Tendsto f cofinite (𝓝 0) := by
  /-
    α : Type u_4
    f : α → ENNReal
    hf : Ne (tsum fun x => f x) Top.top
    ⊢ Filter.Tendsto f Filter.cofinite (nhds 0)
  -/
  have f_ne_top : ∀ n, f n ≠ ∞ := ENNReal.ne_top_of_tsum_ne_top hf
  have h_f_coe : f = fun n => ((f n).toNNReal : ENNReal) :=
    funext fun n => (coe_toNNReal (f_ne_top n)).symm
  /-
    α : Type u_4
    f : α → ENNReal
    hf : Ne (tsum fun x => f x) Top.top
    f_ne_top : ∀ (n : α), Ne (f n) Top.top
    h_f_coe : Eq f fun n => ↑(f n).toNNReal
    ⊢ Filter.Tendsto f Filter.cofinite (nhds 0)
  -/
  rw [h_f_coe, ← @coe_zero, tendsto_coe]
  /-
    α : Type u_4
    f : α → ENNReal
    hf : Ne (tsum fun x => f x) Top.top
    f_ne_top : ∀ (n : α), Ne (f n) Top.top
    h_f_coe : Eq f fun n => ↑(f n).toNNReal
    ⊢ Filter.Tendsto (fun n => (f n).toNNReal) Filter.cofinite (nhds 0)
  -/
  exact NNReal.tendsto_cofinite_zero_of_summable (summable_toNNReal_of_tsum_ne_top hf)
  /-
    🎉 no goals
  -/


theorem tendsto_atTop_zero_of_tsum_ne_top {f : ℕ → ℝ≥0∞} (hf : ∑' x, f x ≠ ∞) :
    Tendsto f atTop (𝓝 0) := by
  /-
    f : Nat → ENNReal
    hf : Ne (tsum fun x => f x) Top.top
    ⊢ Filter.Tendsto f Filter.atTop (nhds 0)
  -/
  rw [← Nat.cofinite_eq_atTop]
  /-
    f : Nat → ENNReal
    hf : Ne (tsum fun x => f x) Top.top
    ⊢ Filter.Tendsto f Filter.cofinite (nhds 0)
  -/
  exact tendsto_cofinite_zero_of_tsum_ne_top hf
  /-
    🎉 no goals
  -/


/-- The sum over the complement of a finset tends to `0` when the finset grows to cover the whole
space. This does not need a summability assumption, as otherwise all sums are zero. -/
theorem tendsto_tsum_compl_atTop_zero {α : Type*} {f : α → ℝ≥0∞} (hf : ∑' x, f x ≠ ∞) :
    Tendsto (fun s : Finset α => ∑' b : { x // x ∉ s }, f b) atTop (𝓝 0) := by
  /-
    α : Type u_4
    f : α → ENNReal
    hf : Ne (tsum fun x => f x) Top.top
    ⊢ Filter.Tendsto (fun s => tsum fun b => f ↑b) Filter.atTop (nhds 0)
  -/
  lift f to α → ℝ≥0 using ENNReal.ne_top_of_tsum_ne_top hf
  /-
    case intro
    α : Type u_4
    f : α → NNReal
    hf : Ne (tsum fun x => (fun i => ↑(f i)) x) Top.top
    ⊢ Filter.Tendsto (fun s => tsum fun b => (fun i => ↑(f i)) ↑b) Filter.atTop (n …
  -/
  convert ENNReal.tendsto_coe.2 (NNReal.tendsto_tsum_compl_atTop_zero f)
  /-
    case h.e'_3.h
    α : Type u_4
    f : α → NNReal
    hf : Ne (tsum fun x => (fun i => ↑(f i)) x) Top.top
    x✝ : Finset α
    ⊢ Eq (tsum fun b => (fun i => ↑(f i)) ↑b) ↑(tsum fun b => f ↑b)
  -/
  rw [ENNReal.coe_tsum]
  /-
    case h.e'_3.h
    α : Type u_4
    f : α → NNReal
    hf : Ne (tsum fun x => (fun i => ↑(f i)) x) Top.top
    x✝ : Finset α
    ⊢ Summable fun b => f ↑b
  -/
  exact NNReal.summable_comp_injective (tsum_coe_ne_top_iff_summable.1 hf) Subtype.coe_injective
  /-
    🎉 no goals
  -/


protected theorem tsum_apply {ι α : Type*} {f : ι → α → ℝ≥0∞} {x : α} :
    (∑' i, f i) x = ∑' i, f i x :=
  tsum_apply <| Pi.summable.mpr fun _ => ENNReal.summable


theorem tsum_sub {f : ℕ → ℝ≥0∞} {g : ℕ → ℝ≥0∞} (h₁ : ∑' i, g i ≠ ∞) (h₂ : g ≤ f) :
    ∑' i, (f i - g i) = ∑' i, f i - ∑' i, g i :=
  have : ∀ i, f i - g i + g i = f i := fun i => tsub_add_cancel_of_le (h₂ i)
                                    /-
                                      f g : Nat → ENNReal
                                      h₁ : Ne (tsum fun i => g i) Top.top
                                      h₂ : LE.le g f
                                      this : ∀ (i : Nat), Eq (HAdd.hAdd (HSub.hSub (f i) (g i)) (g i)) (f i)
                                      ⊢ Eq (HAdd.hAdd (tsum fun i => HSub.hSub (f i) (g i)) (tsum fun i => g i)) (ts …
                                    -/
  ENNReal.eq_sub_of_add_eq h₁ <| by simp only [← ENNReal.tsum_add, this]
                                    /-
                                      🎉 no goals
                                    -/


theorem tsum_comp_le_tsum_of_injective {f : α → β} (hf : Injective f) (g : β → ℝ≥0∞) :
    ∑' x, g (f x) ≤ ∑' y, g y :=
  tsum_le_tsum_of_inj f hf (fun _ _ => zero_le _) (fun _ => le_rfl) ENNReal.summable
    ENNReal.summable


theorem tsum_le_tsum_comp_of_surjective {f : α → β} (hf : Surjective f) (g : β → ℝ≥0∞) :
    ∑' y, g y ≤ ∑' x, g (f x) :=
                                                    /-
                                                      α : Type u_1
                                                      β : Type u_2
                                                      f : α → β
                                                      hf : Function.Surjective f
                                                      g : β → ENNReal
                                                      ⊢ Eq (tsum fun y => g y) (tsum fun y => g (f (Function.surjInv hf y)))
                                                    -/
  calc ∑' y, g y = ∑' y, g (f (surjInv hf y)) := by simp only [surjInv_eq hf]
                                                    /-
                                                      🎉 no goals
                                                    -/
  _ ≤ ∑' x, g (f x) := tsum_comp_le_tsum_of_injective (injective_surjInv hf) _


theorem tsum_mono_subtype (f : α → ℝ≥0∞) {s t : Set α} (h : s ⊆ t) :
    ∑' x : s, f x ≤ ∑' x : t, f x :=
  tsum_comp_le_tsum_of_injective (inclusion_injective h) _


theorem tsum_iUnion_le_tsum {ι : Type*} (f : α → ℝ≥0∞) (t : ι → Set α) :
    ∑' x : ⋃ i, t i, f x ≤ ∑' i, ∑' x : t i, f x :=
  calc ∑' x : ⋃ i, t i, f x ≤ ∑' x : Σ i, t i, f x.2 :=
    tsum_le_tsum_comp_of_surjective (sigmaToiUnion_surjective t) _
  _ = ∑' i, ∑' x : t i, f x := ENNReal.tsum_sigma' _


theorem tsum_biUnion_le_tsum {ι : Type*} (f : α → ℝ≥0∞) (s : Set ι) (t : ι → Set α) :
    ∑' x : ⋃ i ∈ s , t i, f x ≤ ∑' i : s, ∑' x : t i, f x :=
                                                                                         /-
                                                                                           α : Type u_1
                                                                                           ι : Type u_4
                                                                                           f : α → ENNReal
                                                                                           s : Set ι
                                                                                           t : ι → Set α
                                                                                           ⊢ Eq (Set.iUnion fun i => Set.iUnion fun h => t i) (Set.iUnion fun i => t ↑i)
                                                                                         -/
  calc ∑' x : ⋃ i ∈ s, t i, f x = ∑' x : ⋃ i : s, t i, f x := tsum_congr_set_coe _ <| by simp
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
  _ ≤ ∑' i : s, ∑' x : t i, f x := tsum_iUnion_le_tsum _ _


theorem tsum_biUnion_le {ι : Type*} (f : α → ℝ≥0∞) (s : Finset ι) (t : ι → Set α) :
    ∑' x : ⋃ i ∈ s, t i, f x ≤ ∑ i ∈ s, ∑' x : t i, f x :=
  (tsum_biUnion_le_tsum f s.toSet t).trans_eq (Finset.tsum_subtype s fun i => ∑' x : t i, f x)


theorem tsum_iUnion_le {ι : Type*} [Fintype ι] (f : α → ℝ≥0∞) (t : ι → Set α) :
    ∑' x : ⋃ i, t i, f x ≤ ∑ i, ∑' x : t i, f x := by
  /-
    α : Type u_1
    ι : Type u_4
    inst✝ : Fintype ι
    f : α → ENNReal
    t : ι → Set α
    ⊢ LE.le (tsum fun x => f ↑x) (Finset.univ.sum fun i => tsum fun x => f ↑x)
  -/
  rw [← tsum_fintype]
  /-
    α : Type u_1
    ι : Type u_4
    inst✝ : Fintype ι
    f : α → ENNReal
    t : ι → Set α
    ⊢ LE.le (tsum fun x => f ↑x) (tsum fun b => tsum fun x => f ↑x)
  -/
  exact tsum_iUnion_le_tsum f t
  /-
    🎉 no goals
  -/


theorem tsum_union_le (f : α → ℝ≥0∞) (s t : Set α) :
    ∑' x : ↑(s ∪ t), f x ≤ ∑' x : s, f x + ∑' x : t, f x :=
  calc ∑' x : ↑(s ∪ t), f x = ∑' x : ⋃ b, cond b s t, f x := tsum_congr_set_coe _ union_eq_iUnion
              /-
                α : Type u_1
                f : α → ENNReal
                s t : Set α
                ⊢ LE.le (tsum fun x => f ↑x) (HAdd.hAdd (tsum fun x => f ↑x) (tsum fun x => f  …
              -/
  _ ≤ _ := by simpa using tsum_iUnion_le f (cond · s t)
              /-
                🎉 no goals
              -/


open Classical in
theorem tsum_eq_add_tsum_ite {f : β → ℝ≥0∞} (b : β) :
    ∑' x, f x = f b + ∑' x, ite (x = b) 0 (f x) :=
  tsum_eq_add_tsum_ite' b ENNReal.summable


theorem tsum_add_one_eq_top {f : ℕ → ℝ≥0∞} (hf : ∑' n, f n = ∞) (hf0 : f 0 ≠ ∞) :
    ∑' n, f (n + 1) = ∞ := by
  /-
    f : Nat → ENNReal
    hf : Eq (tsum fun n => f n) Top.top
    hf0 : Ne (f 0) Top.top
    ⊢ Eq (tsum fun n => f (HAdd.hAdd n 1)) Top.top
  -/
  rw [tsum_eq_zero_add' ENNReal.summable, add_eq_top] at hf
  /-
    f : Nat → ENNReal
    hf : Or (Eq (f 0) Top.top) (Eq (tsum fun b => f (HAdd.hAdd b 1)) Top.top)
    hf0 : Ne (f 0) Top.top
    ⊢ Eq (tsum fun n => f (HAdd.hAdd n 1)) Top.top
  -/
  exact hf.resolve_left hf0
  /-
    🎉 no goals
  -/


/-- A sum of extended nonnegative reals which is finite can have only finitely many terms
above any positive threshold. -/
theorem finite_const_le_of_tsum_ne_top {ι : Type*} {a : ι → ℝ≥0∞} (tsum_ne_top : ∑' i, a i ≠ ∞)
    {ε : ℝ≥0∞} (ε_ne_zero : ε ≠ 0) : { i : ι | ε ≤ a i }.Finite := by
  /-
    ι : Type u_4
    a : ι → ENNReal
    tsum_ne_top : Ne (tsum fun i => a i) Top.top
    ε : ENNReal
    ε_ne_zero : Ne ε 0
    ⊢ (setOf fun i => LE.le ε (a i)).Finite
  -/
  by_contra h
  /-
    ι : Type u_4
    a : ι → ENNReal
    tsum_ne_top : Ne (tsum fun i => a i) Top.top
    ε : ENNReal
    ε_ne_zero : Ne ε 0
    h : Not (setOf fun i => LE.le ε (a i)).Finite
    ⊢ False
  -/
  have := Infinite.to_subtype h
  /-
    ι : Type u_4
    a : ι → ENNReal
    tsum_ne_top : Ne (tsum fun i => a i) Top.top
    ε : ENNReal
    ε_ne_zero : Ne ε 0
    h : Not (setOf fun i => LE.le ε (a i)).Finite
    this : Infinite ↑(setOf fun i => LE.le ε (a i))
    ⊢ False
  -/
  refine tsum_ne_top (top_unique ?_)
  calc ∞ = ∑' _ : { i | ε ≤ a i }, ε := (tsum_const_eq_top_of_ne_zero ε_ne_zero).symm
  _ ≤ ∑' i, a i := tsum_le_tsum_of_inj (↑) Subtype.val_injective (fun _ _ => zero_le _)
    (fun i => i.2) ENNReal.summable ENNReal.summable


/-- Markov's inequality for `Finset.card` and `tsum` in `ℝ≥0∞`. -/
theorem finset_card_const_le_le_of_tsum_le {ι : Type*} {a : ι → ℝ≥0∞} {c : ℝ≥0∞} (c_ne_top : c ≠ ∞)
    (tsum_le_c : ∑' i, a i ≤ c) {ε : ℝ≥0∞} (ε_ne_zero : ε ≠ 0) :
    ∃ hf : { i : ι | ε ≤ a i }.Finite, #hf.toFinset ≤ c / ε := by
  have hf : { i : ι | ε ≤ a i }.Finite :=
    finite_const_le_of_tsum_ne_top (ne_top_of_le_ne_top c_ne_top tsum_le_c) ε_ne_zero
  /-
    ι : Type u_4
    a : ι → ENNReal
    c : ENNReal
    c_ne_top : Ne c Top.top
    tsum_le_c : LE.le (tsum fun i => a i) c
    ε : ENNReal
    ε_ne_zero : Ne ε 0
    hf : (setOf fun i => LE.le ε (a i)).Finite
    ⊢ Exists fun hf => LE.le (↑hf.toFinset.card) (HDiv.hDiv c ε)
  -/
  refine ⟨hf, (ENNReal.le_div_iff_mul_le (.inl ε_ne_zero) (.inr c_ne_top)).2 ?_⟩
  calc #hf.toFinset * ε = ∑ _i ∈ hf.toFinset, ε := by rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ ∑ i ∈ hf.toFinset, a i := Finset.sum_le_sum fun i => hf.mem_toFinset.1
    _ ≤ ∑' i, a i := ENNReal.sum_le_tsum _
    _ ≤ c := tsum_le_c


theorem tsum_fiberwise (f : β → ℝ≥0∞) (g : β → γ) :
    ∑' x, ∑' b : g ⁻¹' {x}, f b = ∑' i, f i := by
  /-
    β : Type u_2
    γ : Type u_3
    f : β → ENNReal
    g : β → γ
    ⊢ Eq (tsum fun x => tsum fun b => f ↑b) (tsum fun i => f i)
  -/
  apply HasSum.tsum_eq
  /-
    case ha
    β : Type u_2
    γ : Type u_3
    f : β → ENNReal
    g : β → γ
    ⊢ HasSum (fun b => tsum fun b_1 => f ↑b_1) (tsum fun i => f i)
  -/
  let equiv := Equiv.sigmaFiberEquiv g
  /-
    case ha
    β : Type u_2
    γ : Type u_3
    f : β → ENNReal
    g : β → γ
    equiv : Equiv (Sigma fun y => Subtype fun x => Eq (g x) y) β := Equiv.sigmaFib …
    ⊢ HasSum (fun b => tsum fun b_1 => f ↑b_1) (tsum fun i => f i)
  -/
  apply (equiv.hasSum_iff.mpr ENNReal.summable.hasSum).sigma
  /-
    case ha
    β : Type u_2
    γ : Type u_3
    f : β → ENNReal
    g : β → γ
    equiv : Equiv (Sigma fun y => Subtype fun x => Eq (g x) y) β := Equiv.sigmaFib …
    ⊢ ∀ (b : γ), HasSum (fun c => Function.comp f ⇑equiv ⟨b, c⟩) (tsum fun b_1 =>  …
  -/
  exact fun _ ↦ ENNReal.summable.hasSum_iff.mpr rfl
  /-
    🎉 no goals
  -/


theorem tendsto_toReal_iff {ι} {fi : Filter ι} {f : ι → ℝ≥0∞} (hf : ∀ i, f i ≠ ∞) {x : ℝ≥0∞}
    (hx : x ≠ ∞) : Tendsto (fun n => (f n).toReal) fi (𝓝 x.toReal) ↔ Tendsto f fi (𝓝 x) := by
  /-
    ι : Type u_4
    fi : Filter ι
    f : ι → ENNReal
    hf : ∀ (i : ι), Ne (f i) Top.top
    x : ENNReal
    hx : Ne x Top.top
    ⊢ Iff (Filter.Tendsto (fun n => (f n).toReal) fi (nhds x.toReal)) (Filter.Tend …
  -/
  lift f to ι → ℝ≥0 using hf
  /-
    case intro
    ι : Type u_4
    fi : Filter ι
    x : ENNReal
    hx : Ne x Top.top
    f : ι → NNReal
    ⊢ Iff (Filter.Tendsto (fun n => ((fun i => ↑(f i)) n).toReal) fi (nhds x.toRea …
  -/
  lift x to ℝ≥0 using hx
  /-
    case intro.intro
    ι : Type u_4
    fi : Filter ι
    f : ι → NNReal
    x : NNReal
    ⊢ Iff (Filter.Tendsto (fun n => ((fun i => ↑(f i)) n).toReal) fi (nhds (↑x).to …
  -/
  simp [tendsto_coe]
  /-
    🎉 no goals
  -/


theorem tsum_coe_ne_top_iff_summable_coe {f : α → ℝ≥0} :
    (∑' a, (f a : ℝ≥0∞)) ≠ ∞ ↔ Summable fun a => (f a : ℝ) := by
  /-
    α : Type u_1
    f : α → NNReal
    ⊢ Iff (Ne (tsum fun a => ↑(f a)) Top.top) (Summable fun a => ↑(f a))
  -/
  rw [NNReal.summable_coe]
  /-
    α : Type u_1
    f : α → NNReal
    ⊢ Iff (Ne (tsum fun a => ↑(f a)) Top.top) (Summable f)
  -/
  exact tsum_coe_ne_top_iff_summable
  /-
    🎉 no goals
  -/


theorem tsum_coe_eq_top_iff_not_summable_coe {f : α → ℝ≥0} :
    (∑' a, (f a : ℝ≥0∞)) = ∞ ↔ ¬Summable fun a => (f a : ℝ) :=
  tsum_coe_ne_top_iff_summable_coe.not_right


theorem hasSum_toReal {f : α → ℝ≥0∞} (hsum : ∑' x, f x ≠ ∞) :
    HasSum (fun x => (f x).toReal) (∑' x, (f x).toReal) := by
  /-
    α : Type u_1
    f : α → ENNReal
    hsum : Ne (tsum fun x => f x) Top.top
    ⊢ HasSum (fun x => (f x).toReal) (tsum fun x => (f x).toReal)
  -/
  lift f to α → ℝ≥0 using ENNReal.ne_top_of_tsum_ne_top hsum
  /-
    case intro
    α : Type u_1
    f : α → NNReal
    hsum : Ne (tsum fun x => (fun i => ↑(f i)) x) Top.top
    ⊢ HasSum (fun x => ((fun i => ↑(f i)) x).toReal) (tsum fun x => ((fun i => ↑(f …
  -/
  simp only [coe_toReal, ← NNReal.coe_tsum, NNReal.hasSum_coe]
  /-
    case intro
    α : Type u_1
    f : α → NNReal
    hsum : Ne (tsum fun x => (fun i => ↑(f i)) x) Top.top
    ⊢ HasSum f (tsum fun a => f a)
  -/
  exact (tsum_coe_ne_top_iff_summable.1 hsum).hasSum
  /-
    🎉 no goals
  -/


theorem summable_toReal {f : α → ℝ≥0∞} (hsum : ∑' x, f x ≠ ∞) : Summable fun x => (f x).toReal :=
  (hasSum_toReal hsum).summable


theorem tsum_eq_toNNReal_tsum {f : β → ℝ≥0} : ∑' b, f b = (∑' b, (f b : ℝ≥0∞)).toNNReal := by
  /-
    β : Type u_2
    f : β → NNReal
    ⊢ Eq (tsum fun b => f b) (tsum fun b => ↑(f b)).toNNReal
  -/
  by_cases h : Summable f
    /-
      case pos
      β : Type u_2
      f : β → NNReal
      h : Summable f
      ⊢ Eq (tsum fun b => f b) (tsum fun b => ↑(f b)).toNNReal
    -/
  · rw [← ENNReal.coe_tsum h, ENNReal.toNNReal_coe]
    /-
      🎉 no goals
    -/
    /-
      case neg
      β : Type u_2
      f : β → NNReal
      h : Not (Summable f)
      ⊢ Eq (tsum fun b => f b) (tsum fun b => ↑(f b)).toNNReal
    -/
  · have A := tsum_eq_zero_of_not_summable h
    /-
      case neg
      β : Type u_2
      f : β → NNReal
      h : Not (Summable f)
      A : Eq (tsum fun b => f b) 0
      ⊢ Eq (tsum fun b => f b) (tsum fun b => ↑(f b)).toNNReal
    -/
    simp only [← ENNReal.tsum_coe_ne_top_iff_summable, Classical.not_not] at h
    /-
      case neg
      β : Type u_2
      f : β → NNReal
      A : Eq (tsum fun b => f b) 0
      h : Eq (tsum fun b => ↑(f b)) Top.top
      ⊢ Eq (tsum fun b => f b) (tsum fun b => ↑(f b)).toNNReal
    -/
    simp only [h, ENNReal.top_toNNReal, A]
    /-
      🎉 no goals
    -/


/-- Comparison test of convergence of `ℝ≥0`-valued series. -/
theorem exists_le_hasSum_of_le {f g : β → ℝ≥0} {r : ℝ≥0} (hgf : ∀ b, g b ≤ f b) (hfr : HasSum f r) :
    ∃ p ≤ r, HasSum g p :=
  have : (∑' b, (g b : ℝ≥0∞)) ≤ r := by
    /-
      β : Type u_2
      f g : β → NNReal
      r : NNReal
      hgf : ∀ (b : β), LE.le (g b) (f b)
      hfr : HasSum f r
      ⊢ LE.le (tsum fun b => ↑(g b)) ↑r
    -/
    refine hasSum_le (fun b => ?_) ENNReal.summable.hasSum (ENNReal.hasSum_coe.2 hfr)
    /-
      β : Type u_2
      f g : β → NNReal
      r : NNReal
      hgf : ∀ (b : β), LE.le (g b) (f b)
      hfr : HasSum f r
      b : β
      ⊢ LE.le ↑(g b) ↑(f b)
    -/
    exact ENNReal.coe_le_coe.2 (hgf _)
    /-
      🎉 no goals
    -/
  let ⟨p, Eq, hpr⟩ := ENNReal.le_coe_iff.1 this
  ⟨p, hpr, ENNReal.hasSum_coe.1 <| Eq ▸ ENNReal.summable.hasSum⟩


/-- Comparison test of convergence of `ℝ≥0`-valued series. -/
theorem summable_of_le {f g : β → ℝ≥0} (hgf : ∀ b, g b ≤ f b) : Summable f → Summable g
  | ⟨_r, hfr⟩ =>
    let ⟨_p, _, hp⟩ := exists_le_hasSum_of_le hgf hfr
    hp.summable


/-- Summable non-negative functions have countable support -/
theorem _root_.Summable.countable_support_nnreal (f : α → ℝ≥0) (h : Summable f) :
    f.support.Countable := by
  /-
    α : Type u_1
    f : α → NNReal
    h : Summable f
    ⊢ (Function.support f).Countable
  -/
  rw [← NNReal.summable_coe] at h
  /-
    α : Type u_1
    f : α → NNReal
    h : Summable fun a => ↑(f a)
    ⊢ (Function.support f).Countable
  -/
  simpa [support] using h.countable_support
  /-
    🎉 no goals
  -/


/-- A series of non-negative real numbers converges to `r` in the sense of `HasSum` if and only if
the sequence of partial sum converges to `r`. -/
theorem hasSum_iff_tendsto_nat {f : ℕ → ℝ≥0} {r : ℝ≥0} :
    HasSum f r ↔ Tendsto (fun n : ℕ => ∑ i ∈ Finset.range n, f i) atTop (𝓝 r) := by
  /-
    f : Nat → NNReal
    r : NNReal
    ⊢ Iff (HasSum f r) (Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i …
  -/
  rw [← ENNReal.hasSum_coe, ENNReal.hasSum_iff_tendsto_nat]
  /-
    f : Nat → NNReal
    r : NNReal
    ⊢ Iff (Filter.Tendsto (fun n => (Finset.range n).sum fun i => ↑(f i)) Filter.a …
  -/
  simp only [← ENNReal.coe_finset_sum]
  /-
    f : Nat → NNReal
    r : NNReal
    ⊢ Iff (Filter.Tendsto (fun n => ↑((Finset.range n).sum fun a => f a)) Filter.a …
  -/
  exact ENNReal.tendsto_coe
  /-
    🎉 no goals
  -/


theorem not_summable_iff_tendsto_nat_atTop {f : ℕ → ℝ≥0} :
    ¬Summable f ↔ Tendsto (fun n : ℕ => ∑ i ∈ Finset.range n, f i) atTop atTop := by
  /-
    f : Nat → NNReal
    ⊢ Iff (Not (Summable f)) (Filter.Tendsto (fun n => (Finset.range n).sum fun i  …
  -/
  constructor
    /-
      case mp
      f : Nat → NNReal
      ⊢ Not (Summable f) → Filter.Tendsto (fun n => (Finset.range n).sum fun i => f  …
    -/
  · intro h
    /-
      case mp
      f : Nat → NNReal
      h : Not (Summable f)
      ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop Fil …
    -/
    refine ((tendsto_of_monotone ?_).resolve_right h).comp ?_
    /-
      case mp.refine_1
      f : Nat → NNReal
      h : Not (Summable f)
      ⊢ Monotone fun s => s.sum fun b => f b
    -/
    exacts [Finset.sum_mono_set _, tendsto_finset_range]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      f : Nat → NNReal
      ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop Fil …
    -/
  · rintro hnat ⟨r, hr⟩
    /-
      case mpr.intro
      f : Nat → NNReal
      hnat : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTo …
      r : NNReal
      hr : HasSum f r
      ⊢ False
    -/
    exact not_tendsto_nhds_of_tendsto_atTop hnat _ (hasSum_iff_tendsto_nat.1 hr)
    /-
      🎉 no goals
    -/


theorem summable_iff_not_tendsto_nat_atTop {f : ℕ → ℝ≥0} :
    Summable f ↔ ¬Tendsto (fun n : ℕ => ∑ i ∈ Finset.range n, f i) atTop atTop := by
  /-
    f : Nat → NNReal
    ⊢ Iff (Summable f) (Not (Filter.Tendsto (fun n => (Finset.range n).sum fun i = …
  -/
  rw [← not_iff_not, Classical.not_not, not_summable_iff_tendsto_nat_atTop]
  /-
    🎉 no goals
  -/


theorem summable_of_sum_range_le {f : ℕ → ℝ≥0} {c : ℝ≥0}
    (h : ∀ n, ∑ i ∈ Finset.range n, f i ≤ c) : Summable f := by
  /-
    f : Nat → NNReal
    c : NNReal
    h : ∀ (n : Nat), LE.le ((Finset.range n).sum fun i => f i) c
    ⊢ Summable f
  -/
  refine summable_iff_not_tendsto_nat_atTop.2 fun H => ?_
  /-
    f : Nat → NNReal
    c : NNReal
    h : ∀ (n : Nat), LE.le ((Finset.range n).sum fun i => f i) c
    H : Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop F …
    ⊢ False
  -/
  rcases exists_lt_of_tendsto_atTop H 0 c with ⟨n, -, hn⟩
  /-
    case intro.intro
    f : Nat → NNReal
    c : NNReal
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


theorem tsum_le_of_sum_range_le {f : ℕ → ℝ≥0} {c : ℝ≥0}
    (h : ∀ n, ∑ i ∈ Finset.range n, f i ≤ c) : ∑' n, f n ≤ c :=
  _root_.tsum_le_of_sum_range_le (summable_of_sum_range_le h) h


theorem tsum_comp_le_tsum_of_inj {β : Type*} {f : α → ℝ≥0} (hf : Summable f) {i : β → α}
    (hi : Function.Injective i) : (∑' x, f (i x)) ≤ ∑' x, f x :=
  tsum_le_tsum_of_inj i hi (fun _ _ => zero_le _) (fun _ => le_rfl) (summable_comp_injective hf hi)
    hf


theorem summable_sigma {β : α → Type*} {f : (Σ x, β x) → ℝ≥0} :
    Summable f ↔ (∀ x, Summable fun y => f ⟨x, y⟩) ∧ Summable fun x => ∑' y, f ⟨x, y⟩ := by
  /-
    α : Type u_1
    β : α → Type u_4
    f : (Sigma fun x => β x) → NNReal
    ⊢ Iff (Summable f) (And (∀ (x : α), Summable fun y => f ⟨x, y⟩) (Summable fun  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : α → Type u_4
      f : (Sigma fun x => β x) → NNReal
      ⊢ Summable f → And (∀ (x : α), Summable fun y => f ⟨x, y⟩) (Summable fun x =>  …
    -/
  · simp only [← NNReal.summable_coe, NNReal.coe_tsum]
    /-
      case mp
      α : Type u_1
      β : α → Type u_4
      f : (Sigma fun x => β x) → NNReal
      ⊢ (Summable fun a => ↑(f a)) → And (∀ (x : α), Summable fun a => ↑(f ⟨x, a⟩))  …
    -/
    exact fun h => ⟨h.sigma_factor, h.sigma⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : α → Type u_4
      f : (Sigma fun x => β x) → NNReal
      ⊢ And (∀ (x : α), Summable fun y => f ⟨x, y⟩) (Summable fun x => tsum fun y => …
    -/
  · rintro ⟨h₁, h₂⟩
    simpa only [← ENNReal.tsum_coe_ne_top_iff_summable, ENNReal.tsum_sigma',
      ENNReal.coe_tsum (h₁ _)] using h₂


theorem indicator_summable {f : α → ℝ≥0} (hf : Summable f) (s : Set α) :
    Summable (s.indicator f) := by
  classical
  refine NNReal.summable_of_le (fun a => le_trans (le_of_eq (s.indicator_apply f a)) ?_) hf
  split_ifs
  · exact le_refl (f a)
  · exact zero_le_coe


theorem tsum_indicator_ne_zero {f : α → ℝ≥0} (hf : Summable f) {s : Set α} (h : ∃ a ∈ s, f a ≠ 0) :
    (∑' x, (s.indicator f) x) ≠ 0 := fun h' =>
  let ⟨a, ha, hap⟩ := h
  hap ((Set.indicator_apply_eq_self.mpr (absurd ha)).symm.trans
    ((tsum_eq_zero_iff (indicator_summable hf s)).1 h' a))


/-- For `f : ℕ → ℝ≥0`, then `∑' k, f (k + i)` tends to zero. This does not require a summability
assumption on `f`, as otherwise all sums are zero. -/
theorem tendsto_sum_nat_add (f : ℕ → ℝ≥0) : Tendsto (fun i => ∑' k, f (k + i)) atTop (𝓝 0) := by
  /-
    f : Nat → NNReal
    ⊢ Filter.Tendsto (fun i => tsum fun k => f (HAdd.hAdd k i)) Filter.atTop (nhds …
  -/
  rw [← tendsto_coe]
  /-
    f : Nat → NNReal
    ⊢ Filter.Tendsto (fun a => ↑(tsum fun k => f (HAdd.hAdd k a))) Filter.atTop (n …
  -/
  convert _root_.tendsto_sum_nat_add fun i => (f i : ℝ)
  /-
    case h.e'_3.h
    f : Nat → NNReal
    x✝ : Nat
    ⊢ Eq (↑(tsum fun k => f (HAdd.hAdd k x✝))) (tsum fun k => ↑(f (HAdd.hAdd k x✝)))
  -/
  norm_cast
  /-
    🎉 no goals
  -/


nonrec theorem hasSum_lt {f g : α → ℝ≥0} {sf sg : ℝ≥0} {i : α} (h : ∀ a : α, f a ≤ g a)
    (hi : f i < g i) (hf : HasSum f sf) (hg : HasSum g sg) : sf < sg := by
  /-
    α : Type u_1
    f g : α → NNReal
    sf sg : NNReal
    i : α
    h : ∀ (a : α), LE.le (f a) (g a)
    hi : LT.lt (f i) (g i)
    hf : HasSum f sf
    hg : HasSum g sg
    ⊢ LT.lt sf sg
  -/
  have A : ∀ a : α, (f a : ℝ) ≤ g a := fun a => NNReal.coe_le_coe.2 (h a)
  /-
    α : Type u_1
    f g : α → NNReal
    sf sg : NNReal
    i : α
    h : ∀ (a : α), LE.le (f a) (g a)
    hi : LT.lt (f i) (g i)
    hf : HasSum f sf
    hg : HasSum g sg
    A : ∀ (a : α), LE.le ↑(f a) ↑(g a)
    ⊢ LT.lt sf sg
  -/
  have : (sf : ℝ) < sg := hasSum_lt A (NNReal.coe_lt_coe.2 hi) (hasSum_coe.2 hf) (hasSum_coe.2 hg)
  /-
    α : Type u_1
    f g : α → NNReal
    sf sg : NNReal
    i : α
    h : ∀ (a : α), LE.le (f a) (g a)
    hi : LT.lt (f i) (g i)
    hf : HasSum f sf
    hg : HasSum g sg
    A : ∀ (a : α), LE.le ↑(f a) ↑(g a)
    this : LT.lt ↑sf ↑sg
    ⊢ LT.lt sf sg
  -/
  exact NNReal.coe_lt_coe.1 this
  /-
    🎉 no goals
  -/


@[mono]
theorem hasSum_strict_mono {f g : α → ℝ≥0} {sf sg : ℝ≥0} (hf : HasSum f sf) (hg : HasSum g sg)
    (h : f < g) : sf < sg :=
  let ⟨hle, _i, hi⟩ := Pi.lt_def.mp h
  hasSum_lt hle hi hf hg


theorem tsum_lt_tsum {f g : α → ℝ≥0} {i : α} (h : ∀ a : α, f a ≤ g a) (hi : f i < g i)
    (hg : Summable g) : ∑' n, f n < ∑' n, g n :=
  hasSum_lt h hi (summable_of_le h hg).hasSum hg.hasSum


@[mono]
theorem tsum_strict_mono {f g : α → ℝ≥0} (hg : Summable g) (h : f < g) : ∑' n, f n < ∑' n, g n :=
  let ⟨hle, _i, hi⟩ := Pi.lt_def.mp h
  tsum_lt_tsum hle hi hg


theorem tsum_pos {g : α → ℝ≥0} (hg : Summable g) (i : α) (hi : 0 < g i) : 0 < ∑' b, g b := by
  /-
    α : Type u_1
    g : α → NNReal
    hg : Summable g
    i : α
    hi : LT.lt 0 (g i)
    ⊢ LT.lt 0 (tsum fun b => g b)
  -/
  rw [← tsum_zero]
  /-
    α : Type u_1
    g : α → NNReal
    hg : Summable g
    i : α
    hi : LT.lt 0 (g i)
    ⊢ LT.lt (tsum fun x => 0) (tsum fun b => g b)
  -/
  exact tsum_lt_tsum (fun a => zero_le _) hi hg
  /-
    🎉 no goals
  -/


open Classical in
theorem tsum_eq_add_tsum_ite {f : α → ℝ≥0} (hf : Summable f) (i : α) :
    ∑' x, f x = f i + ∑' x, ite (x = i) 0 (f x) := by
  /-
    α : Type u_1
    f : α → NNReal
    hf : Summable f
    i : α
    ⊢ Eq (tsum fun x => f x) (HAdd.hAdd (f i) (tsum fun x => ite (Eq x i) 0 (f x)))
  -/
  refine tsum_eq_add_tsum_ite' i (NNReal.summable_of_le (fun i' => ?_) hf)
  /-
    α : Type u_1
    f : α → NNReal
    hf : Summable f
    i i' : α
    ⊢ LE.le (Function.update f i 0 i') (f i')
  -/
  rw [Function.update_apply]
  /-
    α : Type u_1
    f : α → NNReal
    hf : Summable f
    i i' : α
    ⊢ LE.le (ite (Eq i' i) 0 (f i')) (f i')
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp only [zero_le', le_rfl]
                /-
                  🎉 no goals
                -/


theorem tsum_toNNReal_eq {f : α → ℝ≥0∞} (hf : ∀ a, f a ≠ ∞) :
    (∑' a, f a).toNNReal = ∑' a, (f a).toNNReal :=
  (congr_arg ENNReal.toNNReal (tsum_congr fun x => (coe_toNNReal (hf x)).symm)).trans
    NNReal.tsum_eq_toNNReal_tsum.symm


theorem tsum_toReal_eq {f : α → ℝ≥0∞} (hf : ∀ a, f a ≠ ∞) :
    (∑' a, f a).toReal = ∑' a, (f a).toReal := by
  /-
    α : Type u_1
    f : α → ENNReal
    hf : ∀ (a : α), Ne (f a) Top.top
    ⊢ Eq (tsum fun a => f a).toReal (tsum fun a => (f a).toReal)
  -/
  simp only [ENNReal.toReal, tsum_toNNReal_eq hf, NNReal.coe_tsum]
  /-
    🎉 no goals
  -/


theorem tendsto_sum_nat_add (f : ℕ → ℝ≥0∞) (hf : ∑' i, f i ≠ ∞) :
    Tendsto (fun i => ∑' k, f (k + i)) atTop (𝓝 0) := by
  /-
    f : Nat → ENNReal
    hf : Ne (tsum fun i => f i) Top.top
    ⊢ Filter.Tendsto (fun i => tsum fun k => f (HAdd.hAdd k i)) Filter.atTop (nhds …
  -/
  lift f to ℕ → ℝ≥0 using ENNReal.ne_top_of_tsum_ne_top hf
  /-
    case intro
    f : Nat → NNReal
    hf : Ne (tsum fun i => (fun i => ↑(f i)) i) Top.top
    ⊢ Filter.Tendsto (fun i => tsum fun k => (fun i => ↑(f i)) (HAdd.hAdd k i)) Fi …
  -/
  replace hf : Summable f := tsum_coe_ne_top_iff_summable.1 hf
  /-
    case intro
    f : Nat → NNReal
    hf : Summable f
    ⊢ Filter.Tendsto (fun i => tsum fun k => (fun i => ↑(f i)) (HAdd.hAdd k i)) Fi …
  -/
  simp only [← ENNReal.coe_tsum, NNReal.summable_nat_add _ hf, ← ENNReal.coe_zero]
  /-
    case intro
    f : Nat → NNReal
    hf : Summable f
    ⊢ Filter.Tendsto (fun i => ↑(tsum fun a => f (HAdd.hAdd a i))) Filter.atTop (n …
  -/
  exact mod_cast NNReal.tendsto_sum_nat_add f
  /-
    🎉 no goals
  -/


theorem tsum_le_of_sum_range_le {f : ℕ → ℝ≥0∞} {c : ℝ≥0∞}
    (h : ∀ n, ∑ i ∈ Finset.range n, f i ≤ c) : ∑' n, f n ≤ c :=
  _root_.tsum_le_of_sum_range_le ENNReal.summable h


theorem hasSum_lt {f g : α → ℝ≥0∞} {sf sg : ℝ≥0∞} {i : α} (h : ∀ a : α, f a ≤ g a) (hi : f i < g i)
    (hsf : sf ≠ ∞) (hf : HasSum f sf) (hg : HasSum g sg) : sf < sg := by
  /-
    α : Type u_1
    f g : α → ENNReal
    sf sg : ENNReal
    i : α
    h : ∀ (a : α), LE.le (f a) (g a)
    hi : LT.lt (f i) (g i)
    hsf : Ne sf Top.top
    hf : HasSum f sf
    hg : HasSum g sg
    ⊢ LT.lt sf sg
  -/
  by_cases hsg : sg = ∞
    /-
      case pos
      α : Type u_1
      f g : α → ENNReal
      sf sg : ENNReal
      i : α
      h : ∀ (a : α), LE.le (f a) (g a)
      hi : LT.lt (f i) (g i)
      hsf : Ne sf Top.top
      hf : HasSum f sf
      hg : HasSum g sg
      hsg : Eq sg Top.top
      ⊢ LT.lt sf sg
    -/
  · exact hsg.symm ▸ lt_of_le_of_ne le_top hsf
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      f g : α → ENNReal
      sf sg : ENNReal
      i : α
      h : ∀ (a : α), LE.le (f a) (g a)
      hi : LT.lt (f i) (g i)
      hsf : Ne sf Top.top
      hf : HasSum f sf
      hg : HasSum g sg
      hsg : Not (Eq sg Top.top)
      ⊢ LT.lt sf sg
    -/
  · have hg' : ∀ x, g x ≠ ∞ := ENNReal.ne_top_of_tsum_ne_top (hg.tsum_eq.symm ▸ hsg)
    lift f to α → ℝ≥0 using fun x =>
      ne_of_lt (lt_of_le_of_lt (h x) <| lt_of_le_of_ne le_top (hg' x))
    /-
      case neg.intro
      α : Type u_1
      g : α → ENNReal
      sf sg : ENNReal
      i : α
      hsf : Ne sf Top.top
      hg : HasSum g sg
      hsg : Not (Eq sg Top.top)
      hg' : ∀ (x : α), Ne (g x) Top.top
      f : α → NNReal
      h : ∀ (a : α), LE.le ((fun i => ↑(f i)) a) (g a)
      hi : LT.lt ((fun i => ↑(f i)) i) (g i)
      hf : HasSum (fun i => ↑(f i)) sf
      ⊢ LT.lt sf sg
    -/
    lift g to α → ℝ≥0 using hg'
    /-
      case neg.intro.intro
      α : Type u_1
      sf sg : ENNReal
      i : α
      hsf : Ne sf Top.top
      hsg : Not (Eq sg Top.top)
      f : α → NNReal
      hf : HasSum (fun i => ↑(f i)) sf
      g : α → NNReal
      hg : HasSum (fun i => ↑(g i)) sg
      h : ∀ (a : α), LE.le ((fun i => ↑(f i)) a) ((fun i => ↑(g i)) a)
      hi : LT.lt ((fun i => ↑(f i)) i) ((fun i => ↑(g i)) i)
      ⊢ LT.lt sf sg
    -/
    lift sf to ℝ≥0 using hsf
    /-
      case neg.intro.intro.intro
      α : Type u_1
      sg : ENNReal
      i : α
      hsg : Not (Eq sg Top.top)
      f g : α → NNReal
      hg : HasSum (fun i => ↑(g i)) sg
      h : ∀ (a : α), LE.le ((fun i => ↑(f i)) a) ((fun i => ↑(g i)) a)
      hi : LT.lt ((fun i => ↑(f i)) i) ((fun i => ↑(g i)) i)
      sf : NNReal
      hf : HasSum (fun i => ↑(f i)) ↑sf
      ⊢ LT.lt (↑sf) sg
    -/
    lift sg to ℝ≥0 using hsg
    /-
      case neg.intro.intro.intro.intro
      α : Type u_1
      i : α
      f g : α → NNReal
      h : ∀ (a : α), LE.le ((fun i => ↑(f i)) a) ((fun i => ↑(g i)) a)
      hi : LT.lt ((fun i => ↑(f i)) i) ((fun i => ↑(g i)) i)
      sf : NNReal
      hf : HasSum (fun i => ↑(f i)) ↑sf
      sg : NNReal
      hg : HasSum (fun i => ↑(g i)) ↑sg
      ⊢ LT.lt ↑sf ↑sg
    -/
    simp only [coe_le_coe, coe_lt_coe] at h hi ⊢
    /-
      case neg.intro.intro.intro.intro
      α : Type u_1
      i : α
      f g : α → NNReal
      sf : NNReal
      hf : HasSum (fun i => ↑(f i)) ↑sf
      sg : NNReal
      hg : HasSum (fun i => ↑(g i)) ↑sg
      h : ∀ (a : α), LE.le (f a) (g a)
      hi : LT.lt (f i) (g i)
      ⊢ LT.lt sf sg
    -/
    exact NNReal.hasSum_lt h hi (ENNReal.hasSum_coe.1 hf) (ENNReal.hasSum_coe.1 hg)
    /-
      🎉 no goals
    -/


theorem tsum_lt_tsum {f g : α → ℝ≥0∞} {i : α} (hfi : tsum f ≠ ∞) (h : ∀ a : α, f a ≤ g a)
    (hi : f i < g i) : ∑' x, f x < ∑' x, g x :=
  hasSum_lt h hi hfi ENNReal.summable.hasSum ENNReal.summable.hasSum


theorem tsum_comp_le_tsum_of_inj {β : Type*} {f : α → ℝ} (hf : Summable f) (hn : ∀ a, 0 ≤ f a)
    {i : β → α} (hi : Function.Injective i) : tsum (f ∘ i) ≤ tsum f := by
  /-
    α : Type u_1
    β : Type u_4
    f : α → Real
    hf : Summable f
    hn : ∀ (a : α), LE.le 0 (f a)
    i : β → α
    hi : Function.Injective i
    ⊢ LE.le (tsum (Function.comp f i)) (tsum f)
  -/
  lift f to α → ℝ≥0 using hn
  /-
    case intro
    α : Type u_1
    β : Type u_4
    i : β → α
    hi : Function.Injective i
    f : α → NNReal
    hf : Summable fun i => ↑(f i)
    ⊢ LE.le (tsum (Function.comp (fun i => ↑(f i)) i)) (tsum fun i => ↑(f i))
  -/
  rw [NNReal.summable_coe] at hf
  /-
    case intro
    α : Type u_1
    β : Type u_4
    i : β → α
    hi : Function.Injective i
    f : α → NNReal
    hf : Summable f
    ⊢ LE.le (tsum (Function.comp (fun i => ↑(f i)) i)) (tsum fun i => ↑(f i))
  -/
  simpa only [Function.comp_def, ← NNReal.coe_tsum] using NNReal.tsum_comp_le_tsum_of_inj hf hi
  /-
    🎉 no goals
  -/


/-- Comparison test of convergence of series of non-negative real numbers. -/
theorem Summable.of_nonneg_of_le {f g : β → ℝ} (hg : ∀ b, 0 ≤ g b) (hgf : ∀ b, g b ≤ f b)
    (hf : Summable f) : Summable g := by
  /-
    β : Type u_2
    f g : β → Real
    hg : ∀ (b : β), LE.le 0 (g b)
    hgf : ∀ (b : β), LE.le (g b) (f b)
    hf : Summable f
    ⊢ Summable g
  -/
  lift f to β → ℝ≥0 using fun b => (hg b).trans (hgf b)
  /-
    case intro
    β : Type u_2
    g : β → Real
    hg : ∀ (b : β), LE.le 0 (g b)
    f : β → NNReal
    hgf : ∀ (b : β), LE.le (g b) ((fun i => ↑(f i)) b)
    hf : Summable fun i => ↑(f i)
    ⊢ Summable g
  -/
  lift g to β → ℝ≥0 using hg
  /-
    case intro.intro
    β : Type u_2
    f : β → NNReal
    hf : Summable fun i => ↑(f i)
    g : β → NNReal
    hgf : ∀ (b : β), LE.le ((fun i => ↑(g i)) b) ((fun i => ↑(f i)) b)
    ⊢ Summable fun i => ↑(g i)
  -/
  rw [NNReal.summable_coe] at hf ⊢
  /-
    case intro.intro
    β : Type u_2
    f : β → NNReal
    hf : Summable f
    g : β → NNReal
    hgf : ∀ (b : β), LE.le ((fun i => ↑(g i)) b) ((fun i => ↑(f i)) b)
    ⊢ Summable g
  -/
  exact NNReal.summable_of_le (fun b => NNReal.coe_le_coe.1 (hgf b)) hf
  /-
    🎉 no goals
  -/


theorem Summable.toNNReal {f : α → ℝ} (hf : Summable f) : Summable fun n => (f n).toNNReal := by
  /-
    α : Type u_1
    f : α → Real
    hf : Summable f
    ⊢ Summable fun n => (f n).toNNReal
  -/
  apply NNReal.summable_coe.1
  /-
    α : Type u_1
    f : α → Real
    hf : Summable f
    ⊢ Summable fun a => ↑(f a).toNNReal
  -/
  refine .of_nonneg_of_le (fun n => NNReal.coe_nonneg _) (fun n => ?_) hf.abs
  /-
    α : Type u_1
    f : α → Real
    hf : Summable f
    n : α
    ⊢ LE.le (↑(f n).toNNReal) (_root_.abs (f n))
  -/
  simp only [le_abs_self, Real.coe_toNNReal', max_le_iff, abs_nonneg, and_self_iff]
  /-
    🎉 no goals
  -/


/-- Finitely summable non-negative functions have countable support -/
theorem _root_.Summable.countable_support_ennreal {f : α → ℝ≥0∞} (h : ∑' (i : α), f i ≠ ∞) :
    f.support.Countable := by
  /-
    α : Type u_1
    f : α → ENNReal
    h : Ne (tsum fun i => f i) Top.top
    ⊢ (Function.support f).Countable
  -/
  lift f to α → ℝ≥0 using ENNReal.ne_top_of_tsum_ne_top h
  /-
    case intro
    α : Type u_1
    f : α → NNReal
    h : Ne (tsum fun i => (fun i => ↑(f i)) i) Top.top
    ⊢ (Function.support fun i => ↑(f i)).Countable
  -/
  simpa [support] using (ENNReal.tsum_coe_ne_top_iff_summable.1 h).countable_support_nnreal
  /-
    🎉 no goals
  -/


/-- A series of non-negative real numbers converges to `r` in the sense of `HasSum` if and only if
the sequence of partial sum converges to `r`. -/
theorem hasSum_iff_tendsto_nat_of_nonneg {f : ℕ → ℝ} (hf : ∀ i, 0 ≤ f i) (r : ℝ) :
    HasSum f r ↔ Tendsto (fun n : ℕ => ∑ i ∈ Finset.range n, f i) atTop (𝓝 r) := by
  /-
    f : Nat → Real
    hf : ∀ (i : Nat), LE.le 0 (f i)
    r : Real
    ⊢ Iff (HasSum f r) (Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i …
  -/
  lift f to ℕ → ℝ≥0 using hf
  /-
    case intro
    r : Real
    f : Nat → NNReal
    ⊢ Iff (HasSum (fun i => ↑(f i)) r) (Filter.Tendsto (fun n => (Finset.range n). …
  -/
  simp only [HasSum, ← NNReal.coe_sum, NNReal.tendsto_coe']
  /-
    case intro
    r : Real
    f : Nat → NNReal
    ⊢ Iff (Exists fun hx => Filter.Tendsto (fun a => a.sum fun i => f i) Filter.at …
  -/
  exact exists_congr fun hr => NNReal.hasSum_iff_tendsto_nat
  /-
    🎉 no goals
  -/


theorem ENNReal.ofReal_tsum_of_nonneg {f : α → ℝ} (hf_nonneg : ∀ n, 0 ≤ f n) (hf : Summable f) :
    ENNReal.ofReal (∑' n, f n) = ∑' n, ENNReal.ofReal (f n) := by
  /-
    α : Type u_1
    f : α → Real
    hf_nonneg : ∀ (n : α), LE.le 0 (f n)
    hf : Summable f
    ⊢ Eq (ENNReal.ofReal (tsum fun n => f n)) (tsum fun n => ENNReal.ofReal (f n))
  -/
  simp_rw [ENNReal.ofReal, ENNReal.tsum_coe_eq (NNReal.hasSum_real_toNNReal_of_nonneg hf_nonneg hf)]
  /-
    🎉 no goals
  -/


/-- In an emetric ball, the distance between points is everywhere finite -/
theorem edist_ne_top_of_mem_ball {a : β} {r : ℝ≥0∞} (x y : ball a r) : edist x.1 y.1 ≠ ∞ :=
  ne_of_lt <|
    calc
      edist x y ≤ edist a x + edist a y := edist_triangle_left x.1 y.1 a
                      /-
                        β : Type u_2
                        inst✝ : EMetricSpace β
                        a : β
                        r : ENNReal
                        x y : ↑(EMetric.ball a r)
                        ⊢ LT.lt (HAdd.hAdd (EDist.edist a ↑x) (EDist.edist a ↑y)) (HAdd.hAdd r r)
                      -/
      _ < r + r := by rw [edist_comm a x, edist_comm a y]; exact ENNReal.add_lt_add x.2 y.2
                                                           /-
                                                             🎉 no goals
                                                           -/
      _ ≤ ∞ := le_top


/-- Each ball in an extended metric space gives us a metric space, as the edist
is everywhere finite. -/
def metricSpaceEMetricBall (a : β) (r : ℝ≥0∞) : MetricSpace (ball a r) :=
  EMetricSpace.toMetricSpace edist_ne_top_of_mem_ball


theorem nhds_eq_nhds_emetric_ball (a x : β) (r : ℝ≥0∞) (h : x ∈ ball a r) :
    𝓝 x = map ((↑) : ball a r → β) (𝓝 ⟨x, h⟩) :=
  (map_nhds_subtype_coe_eq_nhds _ <| IsOpen.mem_nhds EMetric.isOpen_ball h).symm


theorem tendsto_iff_edist_tendsto_0 {l : Filter β} {f : β → α} {y : α} :
    Tendsto f l (𝓝 y) ↔ Tendsto (fun x => edist (f x) y) l (𝓝 0) := by
  simp only [EMetric.nhds_basis_eball.tendsto_right_iff, EMetric.mem_ball,
    @tendsto_order ℝ≥0∞ β _ _, forall_prop_of_false ENNReal.not_lt_zero, forall_const, true_and]


/-- Yet another metric characterization of Cauchy sequences on integers. This one is often the
most efficient. -/
theorem EMetric.cauchySeq_iff_le_tendsto_0 [Nonempty β] [SemilatticeSup β] {s : β → α} :
    CauchySeq s ↔ ∃ b : β → ℝ≥0∞, (∀ n m N : β, N ≤ n → N ≤ m → edist (s n) (s m) ≤ b N) ∧
      Tendsto b atTop (𝓝 0) := EMetric.cauchySeq_iff.trans <| by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PseudoEMetricSpace α
    inst✝¹ : Nonempty β
    inst✝ : SemilatticeSup β
    s : β → α
    ⊢ Iff (∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (m : β), LE.le N m → ∀ ( …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝² : PseudoEMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      s : β → α
      ⊢ (∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (m : β), LE.le N m → ∀ (n :  …
    -/
  · intro hs
    /- `s` is Cauchy sequence. Let `b n` be the diameter of the set `s '' Set.Ici n`. -/
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝² : PseudoEMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      s : β → α
      hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (m : β), LE.le N m → ∀ (n  …
      ⊢ Exists fun b => And (∀ (n m N : β), LE.le N n → LE.le N m → LE.le (EDist.edi …
    -/
    refine ⟨fun N => EMetric.diam (s '' Ici N), fun n m N hn hm => ?_, ?_⟩
    -- Prove that it bounds the distances of points in the Cauchy sequence
      /-
        case mp.refine_1
        α : Type u_1
        β : Type u_2
        inst✝² : PseudoEMetricSpace α
        inst✝¹ : Nonempty β
        inst✝ : SemilatticeSup β
        s : β → α
        hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (m : β), LE.le N m → ∀ (n  …
        n m N : β
        hn : LE.le N n
        hm : LE.le N m
        ⊢ LE.le (EDist.edist (s n) (s m)) ((fun N => EMetric.diam (Set.image s (Set.Ic …
      -/
    · exact EMetric.edist_le_diam_of_mem (mem_image_of_mem _ hn) (mem_image_of_mem _ hm)
      /-
        🎉 no goals
      -/
    -- Prove that it tends to `0`, by using the Cauchy property of `s`
      /-
        case mp.refine_2
        α : Type u_1
        β : Type u_2
        inst✝² : PseudoEMetricSpace α
        inst✝¹ : Nonempty β
        inst✝ : SemilatticeSup β
        s : β → α
        hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (m : β), LE.le N m → ∀ (n  …
        ⊢ Filter.Tendsto (fun N => EMetric.diam (Set.image s (Set.Ici N))) Filter.atTo …
      -/
    · refine ENNReal.tendsto_nhds_zero.2 fun ε ε0 => ?_
      /-
        case mp.refine_2
        α : Type u_1
        β : Type u_2
        inst✝² : PseudoEMetricSpace α
        inst✝¹ : Nonempty β
        inst✝ : SemilatticeSup β
        s : β → α
        hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (m : β), LE.le N m → ∀ (n  …
        ε : ENNReal
        ε0 : GT.gt ε 0
        ⊢ Filter.Eventually (fun x => LE.le (EMetric.diam (Set.image s (Set.Ici x))) ε …
      -/
      rcases hs ε ε0 with ⟨N, hN⟩
      /-
        case mp.refine_2.intro
        α : Type u_1
        β : Type u_2
        inst✝² : PseudoEMetricSpace α
        inst✝¹ : Nonempty β
        inst✝ : SemilatticeSup β
        s : β → α
        hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (m : β), LE.le N m → ∀ (n  …
        ε : ENNReal
        ε0 : GT.gt ε 0
        N : β
        hN : ∀ (m : β), LE.le N m → ∀ (n : β), LE.le N n → LT.lt (EDist.edist (s m) (s …
        ⊢ Filter.Eventually (fun x => LE.le (EMetric.diam (Set.image s (Set.Ici x))) ε …
      -/
      refine (eventually_ge_atTop N).mono fun n hn => EMetric.diam_le ?_
      /-
        case mp.refine_2.intro
        α : Type u_1
        β : Type u_2
        inst✝² : PseudoEMetricSpace α
        inst✝¹ : Nonempty β
        inst✝ : SemilatticeSup β
        s : β → α
        hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (m : β), LE.le N m → ∀ (n  …
        ε : ENNReal
        ε0 : GT.gt ε 0
        N : β
        hN : ∀ (m : β), LE.le N m → ∀ (n : β), LE.le N n → LT.lt (EDist.edist (s m) (s …
        n : β
        hn : LE.le N n
        ⊢ ∀ (x : α), Membership.mem (Set.image s (Set.Ici n)) x → ∀ (y : α), Membershi …
      -/
      rintro _ ⟨k, hk, rfl⟩ _ ⟨l, hl, rfl⟩
      /-
        case mp.refine_2.intro.intro.intro.intro.intro
        α : Type u_1
        β : Type u_2
        inst✝² : PseudoEMetricSpace α
        inst✝¹ : Nonempty β
        inst✝ : SemilatticeSup β
        s : β → α
        hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (m : β), LE.le N m → ∀ (n  …
        ε : ENNReal
        ε0 : GT.gt ε 0
        N : β
        hN : ∀ (m : β), LE.le N m → ∀ (n : β), LE.le N n → LT.lt (EDist.edist (s m) (s …
        n : β
        hn : LE.le N n
        k : β
        hk : Membership.mem (Set.Ici n) k
        l : β
        hl : Membership.mem (Set.Ici n) l
        ⊢ LE.le (EDist.edist (s k) (s l)) ε
      -/
      exact (hN _ (hn.trans hk) _ (hn.trans hl)).le
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝² : PseudoEMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      s : β → α
      ⊢ (Exists fun b => And (∀ (n m N : β), LE.le N n → LE.le N m → LE.le (EDist.ed …
    -/
  · rintro ⟨b, ⟨b_bound, b_lim⟩⟩ ε εpos
    /-
      case mpr.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : PseudoEMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      s : β → α
      b : β → ENNReal
      b_bound : ∀ (n m N : β), LE.le N n → LE.le N m → LE.le (EDist.edist (s n) (s m …
      b_lim : Filter.Tendsto b Filter.atTop (nhds 0)
      ε : ENNReal
      εpos : GT.gt ε 0
      ⊢ Exists fun N => ∀ (m : β), LE.le N m → ∀ (n : β), LE.le N n → LT.lt (EDist.e …
    -/
    have : ∀ᶠ n in atTop, b n < ε := b_lim.eventually (gt_mem_nhds εpos)
    /-
      case mpr.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : PseudoEMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      s : β → α
      b : β → ENNReal
      b_bound : ∀ (n m N : β), LE.le N n → LE.le N m → LE.le (EDist.edist (s n) (s m …
      b_lim : Filter.Tendsto b Filter.atTop (nhds 0)
      ε : ENNReal
      εpos : GT.gt ε 0
      this : Filter.Eventually (fun n => LT.lt (b n) ε) Filter.atTop
      ⊢ Exists fun N => ∀ (m : β), LE.le N m → ∀ (n : β), LE.le N n → LT.lt (EDist.e …
    -/
    rcases this.exists with ⟨N, hN⟩
    /-
      case mpr.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : PseudoEMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      s : β → α
      b : β → ENNReal
      b_bound : ∀ (n m N : β), LE.le N n → LE.le N m → LE.le (EDist.edist (s n) (s m …
      b_lim : Filter.Tendsto b Filter.atTop (nhds 0)
      ε : ENNReal
      εpos : GT.gt ε 0
      this : Filter.Eventually (fun n => LT.lt (b n) ε) Filter.atTop
      N : β
      hN : LT.lt (b N) ε
      ⊢ Exists fun N => ∀ (m : β), LE.le N m → ∀ (n : β), LE.le N n → LT.lt (EDist.e …
    -/
    refine ⟨N, fun m hm n hn => ?_⟩
    calc edist (s m) (s n) ≤ b N := b_bound m n N hm hn
    _ < ε := hN


theorem continuous_of_le_add_edist {f : α → ℝ≥0∞} (C : ℝ≥0∞) (hC : C ≠ ∞)
    (h : ∀ x y, f x ≤ f y + C * edist x y) : Continuous f := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : α → ENNReal
    C : ENNReal
    hC : Ne C Top.top
    h : ∀ (x y : α), LE.le (f x) (HAdd.hAdd (f y) (HMul.hMul C (EDist.edist x y)))
    ⊢ Continuous f
  -/
  refine continuous_iff_continuousAt.2 fun x => ENNReal.tendsto_nhds_of_Icc fun ε ε0 => ?_
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : α → ENNReal
    C : ENNReal
    hC : Ne C Top.top
    h : ∀ (x y : α), LE.le (f x) (HAdd.hAdd (f y) (HMul.hMul C (EDist.edist x y)))
    x : α
    ε : ENNReal
    ε0 : GT.gt ε 0
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (Set.Icc (HSub.hSub (f x) ε) (H …
  -/
  rcases ENNReal.exists_nnreal_pos_mul_lt hC ε0.ne' with ⟨δ, δ0, hδ⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : α → ENNReal
    C : ENNReal
    hC : Ne C Top.top
    h : ∀ (x y : α), LE.le (f x) (HAdd.hAdd (f y) (HMul.hMul C (EDist.edist x y)))
    x : α
    ε : ENNReal
    ε0 : GT.gt ε 0
    δ : NNReal
    δ0 : GT.gt δ 0
    hδ : LT.lt (HMul.hMul (↑δ) C) ε
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (Set.Icc (HSub.hSub (f x) ε) (H …
  -/
  rw [mul_comm] at hδ
  /-
    case intro.intro
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : α → ENNReal
    C : ENNReal
    hC : Ne C Top.top
    h : ∀ (x y : α), LE.le (f x) (HAdd.hAdd (f y) (HMul.hMul C (EDist.edist x y)))
    x : α
    ε : ENNReal
    ε0 : GT.gt ε 0
    δ : NNReal
    δ0 : GT.gt δ 0
    hδ : LT.lt (HMul.hMul C ↑δ) ε
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (Set.Icc (HSub.hSub (f x) ε) (H …
  -/
  filter_upwards [EMetric.closedBall_mem_nhds x (ENNReal.coe_pos.2 δ0)] with y hy
  /-
    case h
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : α → ENNReal
    C : ENNReal
    hC : Ne C Top.top
    h : ∀ (x y : α), LE.le (f x) (HAdd.hAdd (f y) (HMul.hMul C (EDist.edist x y)))
    x : α
    ε : ENNReal
    ε0 : GT.gt ε 0
    δ : NNReal
    δ0 : GT.gt δ 0
    hδ : LT.lt (HMul.hMul C ↑δ) ε
    y : α
    hy : Membership.mem (EMetric.closedBall x ↑δ) y
    ⊢ Membership.mem (Set.Icc (HSub.hSub (f x) ε) (HAdd.hAdd (f x) ε)) (f y)
  -/
  refine ⟨tsub_le_iff_right.2 <| (h x y).trans ?_, (h y x).trans ?_⟩ <;>
    /-
      case h.refine_1
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      f : α → ENNReal
      C : ENNReal
      hC : Ne C Top.top
      h : ∀ (x y : α), LE.le (f x) (HAdd.hAdd (f y) (HMul.hMul C (EDist.edist x y)))
      x : α
      ε : ENNReal
      ε0 : GT.gt ε 0
      δ : NNReal
      δ0 : GT.gt δ 0
      hδ : LT.lt (HMul.hMul C ↑δ) ε
      y : α
      hy : Membership.mem (EMetric.closedBall x ↑δ) y
      ⊢ LE.le (HAdd.hAdd (f y) (HMul.hMul C (EDist.edist x y))) (HAdd.hAdd (f y) ε)
    -/
    refine add_le_add_left (le_trans (mul_le_mul_left' ?_ _) hδ.le) _
  /-
    case h.refine_1
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : α → ENNReal
    C : ENNReal
    hC : Ne C Top.top
    h : ∀ (x y : α), LE.le (f x) (HAdd.hAdd (f y) (HMul.hMul C (EDist.edist x y)))
    x : α
    ε : ENNReal
    ε0 : GT.gt ε 0
    δ : NNReal
    δ0 : GT.gt δ 0
    hδ : LT.lt (HMul.hMul C ↑δ) ε
    y : α
    hy : Membership.mem (EMetric.closedBall x ↑δ) y
    ⊢ LE.le (EDist.edist x y) ↑δ
  -/
  exacts [EMetric.mem_closedBall'.1 hy, EMetric.mem_closedBall.1 hy]
  /-
    🎉 no goals
  -/


theorem continuous_edist : Continuous fun p : α × α => edist p.1 p.2 := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    ⊢ Continuous fun p => EDist.edist p.1 p.2
  -/
  apply continuous_of_le_add_edist 2 (by decide)
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    ⊢ ∀ (x y : Prod α α), LE.le (EDist.edist x.1 x.2) (HAdd.hAdd (EDist.edist y.1  …
  -/
  rintro ⟨x, y⟩ ⟨x', y'⟩
  calc
    edist x y ≤ edist x x' + edist x' y' + edist y' y := edist_triangle4 _ _ _ _
    _ = edist x' y' + (edist x x' + edist y y') := by simp only [edist_comm]; ac_rfl
    _ ≤ edist x' y' + (edist (x, y) (x', y') + edist (x, y) (x', y')) :=
      (add_le_add_left (add_le_add (le_max_left _ _) (le_max_right _ _)) _)
    _ = edist x' y' + 2 * edist (x, y) (x', y') := by rw [← mul_two, mul_comm]


@[continuity, fun_prop]
theorem Continuous.edist [TopologicalSpace β] {f g : β → α} (hf : Continuous f)
    (hg : Continuous g) : Continuous fun b => edist (f b) (g b) :=
  continuous_edist.comp (hf.prod_mk hg : _)


theorem Filter.Tendsto.edist {f g : β → α} {x : Filter β} {a b : α} (hf : Tendsto f x (𝓝 a))
    (hg : Tendsto g x (𝓝 b)) : Tendsto (fun x => edist (f x) (g x)) x (𝓝 (edist a b)) :=
  (continuous_edist.tendsto (a, b)).comp (hf.prod_mk_nhds hg)


/-- If the extended distance between consecutive points of a sequence is estimated
by a summable series of `NNReal`s, then the original sequence is a Cauchy sequence. -/
theorem cauchySeq_of_edist_le_of_summable {f : ℕ → α} (d : ℕ → ℝ≥0)
    (hf : ∀ n, edist (f n) (f n.succ) ≤ d n) (hd : Summable d) : CauchySeq f := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : Nat → α
    d : Nat → NNReal
    hf : ∀ (n : Nat), LE.le (EDist.edist (f n) (f n.succ)) ↑(d n)
    hd : Summable d
    ⊢ CauchySeq f
  -/
  refine EMetric.cauchySeq_iff_NNReal.2 fun ε εpos ↦ ?_
  -- Actually we need partial sums of `d` to be a Cauchy sequence.
  replace hd : CauchySeq fun n : ℕ ↦ ∑ x ∈ Finset.range n, d x :=
    let ⟨_, H⟩ := hd
    H.tendsto_sum_nat.cauchySeq
  -- Now we take the same `N` as in one of the definitions of a Cauchy sequence.
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : Nat → α
    d : Nat → NNReal
    hf : ∀ (n : Nat), LE.le (EDist.edist (f n) (f n.succ)) ↑(d n)
    ε : NNReal
    εpos : LT.lt 0 ε
    hd : CauchySeq fun n => (Finset.range n).sum fun x => d x
    ⊢ Exists fun N => ∀ (n : Nat), LE.le N n → LT.lt (EDist.edist (f n) (f N)) ↑ε
  -/
  refine (Metric.cauchySeq_iff'.1 hd ε (NNReal.coe_pos.2 εpos)).imp fun N hN n hn ↦ ?_
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : Nat → α
    d : Nat → NNReal
    hf : ∀ (n : Nat), LE.le (EDist.edist (f n) (f n.succ)) ↑(d n)
    ε : NNReal
    εpos : LT.lt 0 ε
    hd : CauchySeq fun n => (Finset.range n).sum fun x => d x
    N : Nat
    hN : ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist ((Finset.range n).sum fun x =>  …
    n : Nat
    hn : LE.le N n
    ⊢ LT.lt (EDist.edist (f n) (f N)) ↑ε
  -/
  specialize hN n hn
  -- We simplify the known inequality.
  rw [dist_nndist, NNReal.nndist_eq, ← Finset.sum_range_add_sum_Ico _ hn, add_tsub_cancel_left,
    NNReal.coe_lt_coe, max_lt_iff] at hN
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : Nat → α
    d : Nat → NNReal
    hf : ∀ (n : Nat), LE.le (EDist.edist (f n) (f n.succ)) ↑(d n)
    ε : NNReal
    εpos : LT.lt 0 ε
    hd : CauchySeq fun n => (Finset.range n).sum fun x => d x
    N n : Nat
    hn : LE.le N n
    hN : And (LT.lt ((Finset.Ico N n).sum fun k => d k) ε) (LT.lt (HSub.hSub ((Fin …
    ⊢ LT.lt (EDist.edist (f n) (f N)) ↑ε
  -/
  rw [edist_comm]
  -- Then use `hf` to simplify the goal to the same form.
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : Nat → α
    d : Nat → NNReal
    hf : ∀ (n : Nat), LE.le (EDist.edist (f n) (f n.succ)) ↑(d n)
    ε : NNReal
    εpos : LT.lt 0 ε
    hd : CauchySeq fun n => (Finset.range n).sum fun x => d x
    N n : Nat
    hn : LE.le N n
    hN : And (LT.lt ((Finset.Ico N n).sum fun k => d k) ε) (LT.lt (HSub.hSub ((Fin …
    ⊢ LT.lt (EDist.edist (f N) (f n)) ↑ε
  -/
  refine lt_of_le_of_lt (edist_le_Ico_sum_of_edist_le hn fun _ _ ↦ hf _) ?_
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : Nat → α
    d : Nat → NNReal
    hf : ∀ (n : Nat), LE.le (EDist.edist (f n) (f n.succ)) ↑(d n)
    ε : NNReal
    εpos : LT.lt 0 ε
    hd : CauchySeq fun n => (Finset.range n).sum fun x => d x
    N n : Nat
    hn : LE.le N n
    hN : And (LT.lt ((Finset.Ico N n).sum fun k => d k) ε) (LT.lt (HSub.hSub ((Fin …
    ⊢ LT.lt ((Finset.Ico N n).sum fun i => ↑(d i)) ↑ε
  -/
  exact mod_cast hN.1
  /-
    🎉 no goals
  -/


theorem cauchySeq_of_edist_le_of_tsum_ne_top {f : ℕ → α} (d : ℕ → ℝ≥0∞)
    (hf : ∀ n, edist (f n) (f n.succ) ≤ d n) (hd : tsum d ≠ ∞) : CauchySeq f := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : Nat → α
    d : Nat → ENNReal
    hf : ∀ (n : Nat), LE.le (EDist.edist (f n) (f n.succ)) (d n)
    hd : Ne (tsum d) Top.top
    ⊢ CauchySeq f
  -/
  lift d to ℕ → NNReal using fun i => ENNReal.ne_top_of_tsum_ne_top hd i
  /-
    case intro
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : Nat → α
    d : Nat → NNReal
    hf : ∀ (n : Nat), LE.le (EDist.edist (f n) (f n.succ)) ((fun i => ↑(d i)) n)
    hd : Ne (tsum fun i => ↑(d i)) Top.top
    ⊢ CauchySeq f
  -/
  rw [ENNReal.tsum_coe_ne_top_iff_summable] at hd
  /-
    case intro
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : Nat → α
    d : Nat → NNReal
    hf : ∀ (n : Nat), LE.le (EDist.edist (f n) (f n.succ)) ((fun i => ↑(d i)) n)
    hd : Summable d
    ⊢ CauchySeq f
  -/
  exact cauchySeq_of_edist_le_of_summable d hf hd
  /-
    🎉 no goals
  -/


theorem EMetric.isClosed_ball {a : α} {r : ℝ≥0∞} : IsClosed (closedBall a r) :=
  isClosed_le (continuous_id.edist continuous_const) continuous_const


@[simp]
theorem EMetric.diam_closure (s : Set α) : diam (closure s) = diam s := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    s : Set α
    ⊢ Eq (EMetric.diam (closure s)) (EMetric.diam s)
  -/
  refine le_antisymm (diam_le fun x hx y hy => ?_) (diam_mono subset_closure)
  have : edist x y ∈ closure (Iic (diam s)) :=
    map_mem_closure₂ continuous_edist hx hy fun x hx y hy => edist_le_diam_of_mem hx hy
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    s : Set α
    x : α
    hx : Membership.mem (closure s) x
    y : α
    hy : Membership.mem (closure s) y
    this : Membership.mem (closure (Set.Iic (EMetric.diam s))) (EDist.edist x y)
    ⊢ LE.le (EDist.edist x y) (EMetric.diam s)
  -/
  rwa [closure_Iic] at this
  /-
    🎉 no goals
  -/


@[simp]
theorem Metric.diam_closure {α : Type*} [PseudoMetricSpace α] (s : Set α) :
                                           /-
                                             α : Type u_4
                                             inst✝ : PseudoMetricSpace α
                                             s : Set α
                                             ⊢ Eq (Metric.diam (closure s)) (Metric.diam s)
                                           -/
    Metric.diam (closure s) = diam s := by simp only [Metric.diam, EMetric.diam_closure]
                                           /-
                                             🎉 no goals
                                           -/


theorem isClosed_setOf_lipschitzOnWith {α β} [PseudoEMetricSpace α] [PseudoEMetricSpace β] (K : ℝ≥0)
    (s : Set α) : IsClosed { f : α → β | LipschitzOnWith K f s } := by
  /-
    α : Type u_4
    β : Type u_5
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    K : NNReal
    s : Set α
    ⊢ IsClosed (setOf fun f => LipschitzOnWith K f s)
  -/
  simp only [LipschitzOnWith, setOf_forall]
  /-
    α : Type u_4
    β : Type u_5
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    K : NNReal
    s : Set α
    ⊢ IsClosed (Set.iInter fun i => Set.iInter fun x => Set.iInter fun i_1 => Set. …
  -/
  refine isClosed_biInter fun x _ => isClosed_biInter fun y _ => isClosed_le ?_ ?_
  /-
    case refine_1
    α : Type u_4
    β : Type u_5
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    K : NNReal
    s : Set α
    x : α
    x✝¹ : Membership.mem s x
    y : α
    x✝ : Membership.mem s y
    ⊢ Continuous fun x_1 => EDist.edist (x_1 x) (x_1 y)
  -/
  exacts [.edist (continuous_apply x) (continuous_apply y), continuous_const]
  /-
    🎉 no goals
  -/


theorem isClosed_setOf_lipschitzWith {α β} [PseudoEMetricSpace α] [PseudoEMetricSpace β] (K : ℝ≥0) :
    IsClosed { f : α → β | LipschitzWith K f } := by
  /-
    α : Type u_4
    β : Type u_5
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    K : NNReal
    ⊢ IsClosed (setOf fun f => LipschitzWith K f)
  -/
  simp only [← lipschitzOnWith_univ, isClosed_setOf_lipschitzOnWith]
  /-
    🎉 no goals
  -/


/-- For a bounded set `s : Set ℝ`, its `EMetric.diam` is equal to `sSup s - sInf s` reinterpreted as
`ℝ≥0∞`. -/
theorem ediam_eq {s : Set ℝ} (h : Bornology.IsBounded s) :
    EMetric.diam s = ENNReal.ofReal (sSup s - sInf s) := by
  /-
    s : Set Real
    h : Bornology.IsBounded s
    ⊢ Eq (EMetric.diam s) (ENNReal.ofReal (HSub.hSub (SupSet.sSup s) (InfSet.sInf  …
  -/
  rcases eq_empty_or_nonempty s with (rfl | hne)
    /-
      case inl
      h : Bornology.IsBounded EmptyCollection.emptyCollection
      ⊢ Eq (EMetric.diam EmptyCollection.emptyCollection) (ENNReal.ofReal (HSub.hSub …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    s : Set Real
    h : Bornology.IsBounded s
    hne : s.Nonempty
    ⊢ Eq (EMetric.diam s) (ENNReal.ofReal (HSub.hSub (SupSet.sSup s) (InfSet.sInf  …
  -/
  refine le_antisymm (Metric.ediam_le_of_forall_dist_le fun x hx y hy => ?_) ?_
    /-
      case inr.refine_1
      s : Set Real
      h : Bornology.IsBounded s
      hne : s.Nonempty
      x : Real
      hx : Membership.mem s x
      y : Real
      hy : Membership.mem s y
      ⊢ LE.le (Dist.dist x y) (HSub.hSub (SupSet.sSup s) (InfSet.sInf s))
    -/
  · exact Real.dist_le_of_mem_Icc (h.subset_Icc_sInf_sSup hx) (h.subset_Icc_sInf_sSup hy)
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_2
      s : Set Real
      h : Bornology.IsBounded s
      hne : s.Nonempty
      ⊢ LE.le (ENNReal.ofReal (HSub.hSub (SupSet.sSup s) (InfSet.sInf s))) (EMetric. …
    -/
  · apply ENNReal.ofReal_le_of_le_toReal
    /-
      case inr.refine_2.h
      s : Set Real
      h : Bornology.IsBounded s
      hne : s.Nonempty
      ⊢ LE.le (HSub.hSub (SupSet.sSup s) (InfSet.sInf s)) (EMetric.diam s).toReal
    -/
    rw [← Metric.diam, ← Metric.diam_closure]
    calc sSup s - sInf s ≤ dist (sSup s) (sInf s) := le_abs_self _
    _ ≤ Metric.diam (closure s) := dist_le_diam_of_mem h.closure (csSup_mem_closure hne h.bddAbove)
        (csInf_mem_closure hne h.bddBelow)


/-- For a bounded set `s : Set ℝ`, its `Metric.diam` is equal to `sSup s - sInf s`. -/
theorem diam_eq {s : Set ℝ} (h : Bornology.IsBounded s) : Metric.diam s = sSup s - sInf s := by
  /-
    s : Set Real
    h : Bornology.IsBounded s
    ⊢ Eq (Metric.diam s) (HSub.hSub (SupSet.sSup s) (InfSet.sInf s))
  -/
  rw [Metric.diam, Real.ediam_eq h, ENNReal.toReal_ofReal]
  /-
    s : Set Real
    h : Bornology.IsBounded s
    ⊢ LE.le 0 (HSub.hSub (SupSet.sSup s) (InfSet.sInf s))
  -/
  exact sub_nonneg.2 (Real.sInf_le_sSup s h.bddBelow h.bddAbove)
  /-
    🎉 no goals
  -/


@[simp]
theorem ediam_Ioo (a b : ℝ) : EMetric.diam (Ioo a b) = ENNReal.ofReal (b - a) := by
  /-
    a b : Real
    ⊢ Eq (EMetric.diam (Set.Ioo a b)) (ENNReal.ofReal (HSub.hSub b a))
  -/
  rcases le_or_lt b a with (h | h)
    /-
      case inl
      a b : Real
      h : LE.le b a
      ⊢ Eq (EMetric.diam (Set.Ioo a b)) (ENNReal.ofReal (HSub.hSub b a))
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      a b : Real
      h : LT.lt a b
      ⊢ Eq (EMetric.diam (Set.Ioo a b)) (ENNReal.ofReal (HSub.hSub b a))
    -/
  · rw [Real.ediam_eq (isBounded_Ioo _ _), csSup_Ioo h, csInf_Ioo h]
    /-
      🎉 no goals
    -/


@[simp]
theorem ediam_Icc (a b : ℝ) : EMetric.diam (Icc a b) = ENNReal.ofReal (b - a) := by
  /-
    a b : Real
    ⊢ Eq (EMetric.diam (Set.Icc a b)) (ENNReal.ofReal (HSub.hSub b a))
  -/
  rcases le_or_lt a b with (h | h)
    /-
      case inl
      a b : Real
      h : LE.le a b
      ⊢ Eq (EMetric.diam (Set.Icc a b)) (ENNReal.ofReal (HSub.hSub b a))
    -/
  · rw [Real.ediam_eq (isBounded_Icc _ _), csSup_Icc h, csInf_Icc h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      a b : Real
      h : LT.lt b a
      ⊢ Eq (EMetric.diam (Set.Icc a b)) (ENNReal.ofReal (HSub.hSub b a))
    -/
  · simp [h, h.le]
    /-
      🎉 no goals
    -/


@[simp]
theorem ediam_Ico (a b : ℝ) : EMetric.diam (Ico a b) = ENNReal.ofReal (b - a) :=
  le_antisymm (ediam_Icc a b ▸ diam_mono Ico_subset_Icc_self)
    (ediam_Ioo a b ▸ diam_mono Ioo_subset_Ico_self)


@[simp]
theorem ediam_Ioc (a b : ℝ) : EMetric.diam (Ioc a b) = ENNReal.ofReal (b - a) :=
  le_antisymm (ediam_Icc a b ▸ diam_mono Ioc_subset_Icc_self)
    (ediam_Ioo a b ▸ diam_mono Ioo_subset_Ioc_self)


theorem diam_Icc {a b : ℝ} (h : a ≤ b) : Metric.diam (Icc a b) = b - a := by
  /-
    a b : Real
    h : LE.le a b
    ⊢ Eq (Metric.diam (Set.Icc a b)) (HSub.hSub b a)
  -/
  simp [Metric.diam, ENNReal.toReal_ofReal (sub_nonneg.2 h)]
  /-
    🎉 no goals
  -/


theorem diam_Ico {a b : ℝ} (h : a ≤ b) : Metric.diam (Ico a b) = b - a := by
  /-
    a b : Real
    h : LE.le a b
    ⊢ Eq (Metric.diam (Set.Ico a b)) (HSub.hSub b a)
  -/
  simp [Metric.diam, ENNReal.toReal_ofReal (sub_nonneg.2 h)]
  /-
    🎉 no goals
  -/


theorem diam_Ioc {a b : ℝ} (h : a ≤ b) : Metric.diam (Ioc a b) = b - a := by
  /-
    a b : Real
    h : LE.le a b
    ⊢ Eq (Metric.diam (Set.Ioc a b)) (HSub.hSub b a)
  -/
  simp [Metric.diam, ENNReal.toReal_ofReal (sub_nonneg.2 h)]
  /-
    🎉 no goals
  -/


theorem diam_Ioo {a b : ℝ} (h : a ≤ b) : Metric.diam (Ioo a b) = b - a := by
  /-
    a b : Real
    h : LE.le a b
    ⊢ Eq (Metric.diam (Set.Ioo a b)) (HSub.hSub b a)
  -/
  simp [Metric.diam, ENNReal.toReal_ofReal (sub_nonneg.2 h)]
  /-
    🎉 no goals
  -/


/-- If `edist (f n) (f (n+1))` is bounded above by a function `d : ℕ → ℝ≥0∞`,
then the distance from `f n` to the limit is bounded by `∑'_{k=n}^∞ d k`. -/
theorem edist_le_tsum_of_edist_le_of_tendsto {f : ℕ → α} (d : ℕ → ℝ≥0∞)
    (hf : ∀ n, edist (f n) (f n.succ) ≤ d n) {a : α} (ha : Tendsto f atTop (𝓝 a)) (n : ℕ) :
    edist (f n) a ≤ ∑' m, d (n + m) := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : Nat → α
    d : Nat → ENNReal
    hf : ∀ (n : Nat), LE.le (EDist.edist (f n) (f n.succ)) (d n)
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    ⊢ LE.le (EDist.edist (f n) a) (tsum fun m => d (HAdd.hAdd n m))
  -/
  refine le_of_tendsto (tendsto_const_nhds.edist ha) (mem_atTop_sets.2 ⟨n, fun m hnm => ?_⟩)
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : Nat → α
    d : Nat → ENNReal
    hf : ∀ (n : Nat), LE.le (EDist.edist (f n) (f n.succ)) (d n)
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n m : Nat
    hnm : GE.ge m n
    ⊢ Membership.mem (setOf fun x => (fun c => LE.le (EDist.edist (f n) (f c)) (ts …
  -/
  change edist _ _ ≤ _
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : Nat → α
    d : Nat → ENNReal
    hf : ∀ (n : Nat), LE.le (EDist.edist (f n) (f n.succ)) (d n)
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n m : Nat
    hnm : GE.ge m n
    ⊢ LE.le (EDist.edist (f n) (f m)) (tsum fun m => d (HAdd.hAdd n m))
  -/
  refine le_trans (edist_le_Ico_sum_of_edist_le hnm fun _ _ => hf _) ?_
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : Nat → α
    d : Nat → ENNReal
    hf : ∀ (n : Nat), LE.le (EDist.edist (f n) (f n.succ)) (d n)
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n m : Nat
    hnm : GE.ge m n
    ⊢ LE.le ((Finset.Ico n m).sum fun i => d i) (tsum fun m => d (HAdd.hAdd n m))
  -/
  rw [Finset.sum_Ico_eq_sum_range]
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f : Nat → α
    d : Nat → ENNReal
    hf : ∀ (n : Nat), LE.le (EDist.edist (f n) (f n.succ)) (d n)
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n m : Nat
    hnm : GE.ge m n
    ⊢ LE.le ((Finset.range (HSub.hSub m n)).sum fun k => d (HAdd.hAdd n k)) (tsum  …
  -/
  exact sum_le_tsum _ (fun _ _ => zero_le _) ENNReal.summable
  /-
    🎉 no goals
  -/


/-- If `edist (f n) (f (n+1))` is bounded above by a function `d : ℕ → ℝ≥0∞`,
then the distance from `f 0` to the limit is bounded by `∑'_{k=0}^∞ d k`. -/
theorem edist_le_tsum_of_edist_le_of_tendsto₀ {f : ℕ → α} (d : ℕ → ℝ≥0∞)
    (hf : ∀ n, edist (f n) (f n.succ) ≤ d n) {a : α} (ha : Tendsto f atTop (𝓝 a)) :
                                    /-
                                      α : Type u_1
                                      inst✝ : PseudoEMetricSpace α
                                      f : Nat → α
                                      d : Nat → ENNReal
                                      hf : ∀ (n : Nat), LE.le (EDist.edist (f n) (f n.succ)) (d n)
                                      a : α
                                      ha : Filter.Tendsto f Filter.atTop (nhds a)
                                      ⊢ LE.le (EDist.edist (f 0) a) (tsum fun m => d m)
                                    -/
    edist (f 0) a ≤ ∑' m, d m := by simpa using edist_le_tsum_of_edist_le_of_tendsto d hf ha 0
                                    /-
                                      🎉 no goals
                                    -/


/-- With truncation level `t`, the truncated cast `ℝ≥0∞ → ℝ` is given by `x ↦ (min t x).toReal`.
Unlike `ENNReal.toReal`, this cast is continuous and monotone when `t ≠ ∞`. -/
noncomputable def truncateToReal (t x : ℝ≥0∞) : ℝ := (min t x).toReal


lemma truncateToReal_eq_toReal {t x : ℝ≥0∞} (t_ne_top : t ≠ ∞) (x_le : x ≤ t) :
    truncateToReal t x = x.toReal := by
  /-
    t x : ENNReal
    t_ne_top : Ne t Top.top
    x_le : LE.le x t
    ⊢ Eq (t.truncateToReal x) x.toReal
  -/
  have x_lt_top : x < ∞ := lt_of_le_of_lt x_le t_ne_top.lt_top
  have obs : min t x ≠ ∞ := by
    simp_all only [ne_eq, min_eq_top, false_and, not_false_eq_true]
  /-
    t x : ENNReal
    t_ne_top : Ne t Top.top
    x_le : LE.le x t
    x_lt_top : LT.lt x Top.top
    obs : Ne (Min.min t x) Top.top
    ⊢ Eq (t.truncateToReal x) x.toReal
  -/
  exact (ENNReal.toReal_eq_toReal obs x_lt_top.ne).mpr (min_eq_right x_le)
  /-
    🎉 no goals
  -/


lemma truncateToReal_le {t : ℝ≥0∞} (t_ne_top : t ≠ ∞) {x : ℝ≥0∞} :
    truncateToReal t x ≤ t.toReal := by
  /-
    t : ENNReal
    t_ne_top : Ne t Top.top
    x : ENNReal
    ⊢ LE.le (t.truncateToReal x) t.toReal
  -/
  rw [truncateToReal]
  /-
    t : ENNReal
    t_ne_top : Ne t Top.top
    x : ENNReal
    ⊢ LE.le (Min.min t x).toReal t.toReal
  -/
  gcongr
  /-
    case hb
    t : ENNReal
    t_ne_top : Ne t Top.top
    x : ENNReal
    ⊢ Ne t Top.top
  -/
  exacts [t_ne_top, min_le_left t x]
  /-
    🎉 no goals
  -/


lemma truncateToReal_nonneg {t x : ℝ≥0∞} : 0 ≤ truncateToReal t x := toReal_nonneg


/-- The truncated cast `ENNReal.truncateToReal t : ℝ≥0∞ → ℝ` is monotone when `t ≠ ∞`. -/
lemma monotone_truncateToReal {t : ℝ≥0∞} (t_ne_top : t ≠ ∞) : Monotone (truncateToReal t) := by
  /-
    t : ENNReal
    t_ne_top : Ne t Top.top
    ⊢ Monotone t.truncateToReal
  -/
  intro x y x_le_y
  /-
    t : ENNReal
    t_ne_top : Ne t Top.top
    x y : ENNReal
    x_le_y : LE.le x y
    ⊢ LE.le (t.truncateToReal x) (t.truncateToReal y)
  -/
  simp only [truncateToReal]
  /-
    t : ENNReal
    t_ne_top : Ne t Top.top
    x y : ENNReal
    x_le_y : LE.le x y
    ⊢ LE.le (Min.min t x).toReal (Min.min t y).toReal
  -/
  gcongr
  /-
    case hb
    t : ENNReal
    t_ne_top : Ne t Top.top
    x y : ENNReal
    x_le_y : LE.le x y
    ⊢ Ne (Min.min t y) Top.top
  -/
  exact ne_top_of_le_ne_top t_ne_top (min_le_left _ _)
  /-
    🎉 no goals
  -/


/-- The truncated cast `ENNReal.truncateToReal t : ℝ≥0∞ → ℝ` is continuous when `t ≠ ∞`. -/
lemma continuous_truncateToReal {t : ℝ≥0∞} (t_ne_top : t ≠ ∞) : Continuous (truncateToReal t) := by
  /-
    t : ENNReal
    t_ne_top : Ne t Top.top
    ⊢ Continuous t.truncateToReal
  -/
  apply continuousOn_toReal.comp_continuous (continuous_min.comp (Continuous.Prod.mk t))
  /-
    t : ENNReal
    t_ne_top : Ne t Top.top
    ⊢ ∀ (x : ENNReal), Membership.mem (setOf fun a => Ne a Top.top) (Function.comp …
  -/
  simp [t_ne_top]
  /-
    🎉 no goals
  -/


lemma limsup_sub_const (F : Filter ι) (f : ι → ℝ≥0∞) (c : ℝ≥0∞) :
    Filter.limsup (fun i ↦ f i - c) F = Filter.limsup f F - c := by
  /-
    ι : Type u_4
    F : Filter ι
    f : ι → ENNReal
    c : ENNReal
    ⊢ Eq (Filter.limsup (fun i => HSub.hSub (f i) c) F) (HSub.hSub (Filter.limsup  …
  -/
  rcases F.eq_or_neBot with rfl | _
    /-
      case inl
      ι : Type u_4
      f : ι → ENNReal
      c : ENNReal
      ⊢ Eq (Filter.limsup (fun i => HSub.hSub (f i) c) Bot.bot) (HSub.hSub (Filter.l …
    -/
  · simp only [limsup_bot, bot_eq_zero', zero_le, tsub_eq_zero_of_le]
    /-
      🎉 no goals
    -/
  · exact (Monotone.map_limsSup_of_continuousAt (F := F.map f) (f := fun (x : ℝ≥0∞) ↦ x - c)
    (fun _ _ h ↦ tsub_le_tsub_right h c) (continuous_sub_right c).continuousAt).symm


lemma liminf_sub_const (F : Filter ι) [NeBot F] (f : ι → ℝ≥0∞) (c : ℝ≥0∞) :
    Filter.liminf (fun i ↦ f i - c) F = Filter.liminf f F - c :=
   /-
     ι : Type u_4
     F : Filter ι
     inst✝ : F.NeBot
     f : ι → ENNReal
     c : ENNReal
     ⊢ Filter.IsCobounded (fun x1 x2 => GE.ge x1 x2) (Filter.map f F)
   -/
   /-
     🎉 no goals
   -/
  (Monotone.map_limsInf_of_continuousAt (F := F.map f) (f := fun (x : ℝ≥0∞) ↦ x - c)
   /-
     🎉 no goals
   -/
    (fun _ _ h ↦ tsub_le_tsub_right h c) (continuous_sub_right c).continuousAt).symm


lemma limsup_const_sub (F : Filter ι) (f : ι → ℝ≥0∞) {c : ℝ≥0∞} (c_ne_top : c ≠ ∞) :
    Filter.limsup (fun i ↦ c - f i) F = c - Filter.liminf f F := by
  /-
    ι : Type u_4
    F : Filter ι
    f : ι → ENNReal
    c : ENNReal
    c_ne_top : Ne c Top.top
    ⊢ Eq (Filter.limsup (fun i => HSub.hSub c (f i)) F) (HSub.hSub c (Filter.limin …
  -/
  rcases F.eq_or_neBot with rfl | _
    /-
      case inl
      ι : Type u_4
      f : ι → ENNReal
      c : ENNReal
      c_ne_top : Ne c Top.top
      ⊢ Eq (Filter.limsup (fun i => HSub.hSub c (f i)) Bot.bot) (HSub.hSub c (Filter …
    -/
  · simp only [limsup_bot, bot_eq_zero', liminf_bot, le_top, tsub_eq_zero_of_le]
    /-
      🎉 no goals
    -/
  · exact (Antitone.map_limsInf_of_continuousAt (F := F.map f) (f := fun (x : ℝ≥0∞) ↦ c - x)
    (fun _ _ h ↦ tsub_le_tsub_left h c) (continuous_sub_left c_ne_top).continuousAt).symm


lemma liminf_const_sub (F : Filter ι) [NeBot F] (f : ι → ℝ≥0∞) {c : ℝ≥0∞} (c_ne_top : c ≠ ∞) :
    Filter.liminf (fun i ↦ c - f i) F = c - Filter.limsup f F :=
   /-
     ι : Type u_4
     F : Filter ι
     inst✝ : F.NeBot
     f : ι → ENNReal
     c : ENNReal
     c_ne_top : Ne c Top.top
     ⊢ Filter.IsBounded (fun x1 x2 => LE.le x1 x2) (Filter.map f F)
   -/
   /-
     🎉 no goals
   -/
  (Antitone.map_limsSup_of_continuousAt (F := F.map f) (f := fun (x : ℝ≥0∞) ↦ c - x)
   /-
     🎉 no goals
   -/
    (fun _ _ h ↦ tsub_le_tsub_left h c) (continuous_sub_left c_ne_top).continuousAt).symm


lemma le_limsup_mul {α : Type*} {f : Filter α} {u v : α → ℝ≥0∞} :
    limsup u f * liminf v f ≤ limsup (u * v) f :=
                                         /-
                                           α : Type u_5
                                           f : Filter α
                                           u v : α → ENNReal
                                           a : ENNReal
                                           a_u : LT.lt a (Filter.limsup u f)
                                           b : ENNReal
                                           b_v : LT.lt b (Filter.liminf v f)
                                           ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f (HMul.hMul u v)
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
  mul_le_of_forall_lt fun a a_u b b_v ↦ (le_limsup_iff).2 fun c c_ab ↦
                                         /-
                                           🎉 no goals
                                         -/
                                                 /-
                                                   α : Type u_5
                                                   f : Filter α
                                                   u v : α → ENNReal
                                                   a : ENNReal
                                                   a_u : LT.lt a (Filter.limsup u f)
                                                   b : ENNReal
                                                   b_v : LT.lt b (Filter.liminf v f)
                                                   c : ENNReal
                                                   c_ab : LT.lt c (HMul.hMul a b)
                                                   ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u
                                                 -/
    Frequently.mono (Frequently.and_eventually ((frequently_lt_of_lt_limsup) a_u)
                                                 /-
                                                   🎉 no goals
                                                 -/
      /-
        α : Type u_5
        f : Filter α
        u v : α → ENNReal
        a : ENNReal
        a_u : LT.lt a (Filter.limsup u f)
        b : ENNReal
        b_v : LT.lt b (Filter.liminf v f)
        c : ENNReal
        c_ab : LT.lt c (HMul.hMul a b)
        h : LT.lt b (Filter.liminf v f)
        ⊢ Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v
      -/
    ((eventually_lt_of_lt_liminf) b_v)) fun _ ab_x ↦ c_ab.trans (mul_lt_mul ab_x.1 ab_x.2)
      /-
        🎉 no goals
      -/


/-- See also `ENNReal.limsup_mul_le`.-/
lemma limsup_mul_le' {α : Type*} {f : Filter α} {u v : α → ℝ≥0∞}
    (h : limsup u f ≠ 0 ∨ limsup v f ≠ ∞) (h' : limsup u f ≠ ∞ ∨ limsup v f ≠ 0) :
    limsup (u * v) f ≤ limsup u f * limsup v f := by
  /-
    α : Type u_5
    f : Filter α
    u v : α → ENNReal
    h : Or (Ne (Filter.limsup u f) 0) (Ne (Filter.limsup v f) Top.top)
    h' : Or (Ne (Filter.limsup u f) Top.top) (Ne (Filter.limsup v f) 0)
    ⊢ LE.le (Filter.limsup (HMul.hMul u v) f) (HMul.hMul (Filter.limsup u f) (Filt …
  -/
  refine le_mul_of_forall_lt h h' fun a a_u b b_v ↦ (limsup_le_iff).2 fun c c_ab ↦ ?_
  /-
    α : Type u_5
    f : Filter α
    u v : α → ENNReal
    h : Or (Ne (Filter.limsup u f) 0) (Ne (Filter.limsup v f) Top.top)
    h' : Or (Ne (Filter.limsup u f) Top.top) (Ne (Filter.limsup v f) 0)
    a : ENNReal
    a_u : GT.gt a (Filter.limsup u f)
    b : ENNReal
    b_v : GT.gt b (Filter.limsup v f)
    c : ENNReal
    c_ab : GT.gt c (HMul.hMul a b)
    ⊢ Filter.Eventually (fun a => LT.lt (HMul.hMul u v a) c) f
  -/
  filter_upwards [eventually_lt_of_limsup_lt a_u, eventually_lt_of_limsup_lt b_v] with x a_x b_x
  /-
    case h
    α : Type u_5
    f : Filter α
    u v : α → ENNReal
    h : Or (Ne (Filter.limsup u f) 0) (Ne (Filter.limsup v f) Top.top)
    h' : Or (Ne (Filter.limsup u f) Top.top) (Ne (Filter.limsup v f) 0)
    a : ENNReal
    a_u : GT.gt a (Filter.limsup u f)
    b : ENNReal
    b_v : GT.gt b (Filter.limsup v f)
    c : ENNReal
    c_ab : GT.gt c (HMul.hMul a b)
    x : α
    a_x : LT.lt (u x) a
    b_x : LT.lt (v x) b
    ⊢ LT.lt (HMul.hMul u v x) c
  -/
  exact (mul_lt_mul a_x b_x).trans c_ab
  /-
    🎉 no goals
  -/


lemma le_liminf_mul {α : Type*} {f : Filter α} {u v : α → ℝ≥0∞} :
    liminf u f * liminf v f ≤ liminf (u * v) f := by
  /-
    α : Type u_5
    f : Filter α
    u v : α → ENNReal
    ⊢ LE.le (HMul.hMul (Filter.liminf u f) (Filter.liminf v f)) (Filter.liminf (HM …
  -/
  refine mul_le_of_forall_lt fun a a_u b b_v ↦ (le_liminf_iff).2 fun c c_ab ↦ ?_
  /-
    α : Type u_5
    f : Filter α
    u v : α → ENNReal
    a : ENNReal
    a_u : LT.lt a (Filter.liminf u f)
    b : ENNReal
    b_v : LT.lt b (Filter.liminf v f)
    c : ENNReal
    c_ab : LT.lt c (HMul.hMul a b)
    ⊢ Filter.Eventually (fun a => LT.lt c (HMul.hMul u v a)) f
  -/
  filter_upwards [eventually_lt_of_lt_liminf a_u, eventually_lt_of_lt_liminf b_v] with x a_x b_x
  /-
    case h
    α : Type u_5
    f : Filter α
    u v : α → ENNReal
    a : ENNReal
    a_u : LT.lt a (Filter.liminf u f)
    b : ENNReal
    b_v : LT.lt b (Filter.liminf v f)
    c : ENNReal
    c_ab : LT.lt c (HMul.hMul a b)
    x : α
    a_x : LT.lt a (u x)
    b_x : LT.lt b (v x)
    ⊢ LT.lt c (HMul.hMul u v x)
  -/
  exact c_ab.trans (mul_lt_mul a_x b_x)
  /-
    🎉 no goals
  -/


lemma liminf_mul_le {α : Type*} {f : Filter α} {u v : α → ℝ≥0∞}
    (h : limsup u f ≠ 0 ∨ liminf v f ≠ ∞) (h' : limsup u f ≠ ∞ ∨ liminf v f ≠ 0) :
    liminf (u * v) f ≤ limsup u f * liminf v f :=
                                              /-
                                                α : Type u_5
                                                f : Filter α
                                                u v : α → ENNReal
                                                h : Or (Ne (Filter.limsup u f) 0) (Ne (Filter.liminf v f) Top.top)
                                                h' : Or (Ne (Filter.limsup u f) Top.top) (Ne (Filter.liminf v f) 0)
                                                a : ENNReal
                                                a_u : GT.gt a (Filter.limsup u f)
                                                b : ENNReal
                                                b_v : GT.gt b (Filter.liminf v f)
                                                ⊢ Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HMul.hMul u v)
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
  le_mul_of_forall_lt h h' fun a a_u b b_v ↦ (liminf_le_iff).2 fun c c_ab ↦
                                              /-
                                                🎉 no goals
                                              -/
                       /-
                         α : Type u_5
                         f : Filter α
                         u v : α → ENNReal
                         h : Or (Ne (Filter.limsup u f) 0) (Ne (Filter.liminf v f) Top.top)
                         h' : Or (Ne (Filter.limsup u f) Top.top) (Ne (Filter.liminf v f) 0)
                         a : ENNReal
                         a_u : GT.gt a (Filter.limsup u f)
                         b : ENNReal
                         b_v : GT.gt b (Filter.liminf v f)
                         c : ENNReal
                         c_ab : GT.gt c (HMul.hMul a b)
                         ⊢ Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v
                       -/
    Frequently.mono (((frequently_lt_of_liminf_lt) b_v).and_eventually
                       /-
                         🎉 no goals
                       -/
      /-
        α : Type u_5
        f : Filter α
        u v : α → ENNReal
        h✝ : Or (Ne (Filter.limsup u f) 0) (Ne (Filter.liminf v f) Top.top)
        h' : Or (Ne (Filter.limsup u f) Top.top) (Ne (Filter.liminf v f) 0)
        a : ENNReal
        a_u : GT.gt a (Filter.limsup u f)
        b : ENNReal
        b_v : GT.gt b (Filter.liminf v f)
        c : ENNReal
        c_ab : GT.gt c (HMul.hMul a b)
        h : LT.lt (Filter.limsup u f) a
        ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
      -/
    ((eventually_lt_of_limsup_lt) a_u)) fun _ ab_x ↦ (mul_lt_mul ab_x.2 ab_x.1).trans c_ab
      /-
        🎉 no goals
      -/


/-- If `xs : ι → ℝ≥0∞` is bounded, then we have `liminf (toReal ∘ xs) = toReal (liminf xs)`. -/
lemma liminf_toReal_eq {ι : Type*} {F : Filter ι} [NeBot F] {b : ℝ≥0∞} (b_ne_top : b ≠ ∞)
    {xs : ι → ℝ≥0∞} (le_b : ∀ᶠ i in F, xs i ≤ b) :
    F.liminf (fun i ↦ (xs i).toReal) = (F.liminf xs).toReal := by
  have liminf_le : F.liminf xs ≤ b := by
    apply liminf_le_of_le ⟨0, by simp⟩
    intro y h
    obtain ⟨i, hi⟩ := (Eventually.and h le_b).exists
    exact hi.1.trans hi.2
  have aux : ∀ᶠ i in F, (xs i).toReal = ENNReal.truncateToReal b (xs i) := by
    filter_upwards [le_b] with i i_le_b
    simp only [truncateToReal_eq_toReal b_ne_top i_le_b, implies_true]
  have aux' : (F.liminf xs).toReal = ENNReal.truncateToReal b (F.liminf xs) := by
    rw [truncateToReal_eq_toReal b_ne_top liminf_le]
  /-
    ι : Type u_5
    F : Filter ι
    inst✝ : F.NeBot
    b : ENNReal
    b_ne_top : Ne b Top.top
    xs : ι → ENNReal
    le_b : Filter.Eventually (fun i => LE.le (xs i) b) F
    liminf_le : LE.le (Filter.liminf xs F) b
    aux : Filter.Eventually (fun i => Eq (xs i).toReal (b.truncateToReal (xs i))) F
    aux' : Eq (Filter.liminf xs F).toReal (b.truncateToReal (Filter.liminf xs F))
    ⊢ Eq (Filter.liminf (fun i => (xs i).toReal) F) (Filter.liminf xs F).toReal
  -/
  simp_rw [liminf_congr aux, aux']
  have key := Monotone.map_liminf_of_continuousAt (F := F) (monotone_truncateToReal b_ne_top) xs
          (continuous_truncateToReal b_ne_top).continuousAt
          (IsBoundedUnder.isCoboundedUnder_ge ⟨b, by simpa only [eventually_map] using le_b⟩)
          ⟨0, Eventually.of_forall (by simp)⟩
  /-
    ι : Type u_5
    F : Filter ι
    inst✝ : F.NeBot
    b : ENNReal
    b_ne_top : Ne b Top.top
    xs : ι → ENNReal
    le_b : Filter.Eventually (fun i => LE.le (xs i) b) F
    liminf_le : LE.le (Filter.liminf xs F) b
    aux : Filter.Eventually (fun i => Eq (xs i).toReal (b.truncateToReal (xs i))) F
    aux' : Eq (Filter.liminf xs F).toReal (b.truncateToReal (Filter.liminf xs F))
    key : Eq (b.truncateToReal (Filter.liminf xs F)) (Filter.liminf (Function.comp …
    ⊢ Eq (Filter.liminf (fun a => b.truncateToReal (xs a)) F) (b.truncateToReal (F …
  -/
  rw [key]
  /-
    ι : Type u_5
    F : Filter ι
    inst✝ : F.NeBot
    b : ENNReal
    b_ne_top : Ne b Top.top
    xs : ι → ENNReal
    le_b : Filter.Eventually (fun i => LE.le (xs i) b) F
    liminf_le : LE.le (Filter.liminf xs F) b
    aux : Filter.Eventually (fun i => Eq (xs i).toReal (b.truncateToReal (xs i))) F
    aux' : Eq (Filter.liminf xs F).toReal (b.truncateToReal (Filter.liminf xs F))
    key : Eq (b.truncateToReal (Filter.liminf xs F)) (Filter.liminf (Function.comp …
    ⊢ Eq (Filter.liminf (fun a => b.truncateToReal (xs a)) F) (Filter.liminf (Func …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `xs : ι → ℝ≥0∞` is bounded, then we have `liminf (toReal ∘ xs) = toReal (liminf xs)`. -/
lemma limsup_toReal_eq {ι : Type*} {F : Filter ι} [NeBot F] {b : ℝ≥0∞} (b_ne_top : b ≠ ∞)
    {xs : ι → ℝ≥0∞} (le_b : ∀ᶠ i in F, xs i ≤ b) :
    F.limsup (fun i ↦ (xs i).toReal) = (F.limsup xs).toReal := by
  have aux : ∀ᶠ i in F, (xs i).toReal = ENNReal.truncateToReal b (xs i) := by
    filter_upwards [le_b] with i i_le_b
    simp only [truncateToReal_eq_toReal b_ne_top i_le_b, implies_true]
  have aux' : (F.limsup xs).toReal = ENNReal.truncateToReal b (F.limsup xs) := by
    rw [truncateToReal_eq_toReal b_ne_top (limsup_le_of_le ⟨0, by simp⟩ le_b)]
  /-
    ι : Type u_5
    F : Filter ι
    inst✝ : F.NeBot
    b : ENNReal
    b_ne_top : Ne b Top.top
    xs : ι → ENNReal
    le_b : Filter.Eventually (fun i => LE.le (xs i) b) F
    aux : Filter.Eventually (fun i => Eq (xs i).toReal (b.truncateToReal (xs i))) F
    aux' : Eq (Filter.limsup xs F).toReal (b.truncateToReal (Filter.limsup xs F))
    ⊢ Eq (Filter.limsup (fun i => (xs i).toReal) F) (Filter.limsup xs F).toReal
  -/
  simp_rw [limsup_congr aux, aux']
  have key := Monotone.map_limsup_of_continuousAt (F := F) (monotone_truncateToReal b_ne_top) xs
          (continuous_truncateToReal b_ne_top).continuousAt
          ⟨b, by simpa only [eventually_map] using le_b⟩
          (IsBoundedUnder.isCoboundedUnder_le ⟨0, Eventually.of_forall (by simp)⟩)
  /-
    ι : Type u_5
    F : Filter ι
    inst✝ : F.NeBot
    b : ENNReal
    b_ne_top : Ne b Top.top
    xs : ι → ENNReal
    le_b : Filter.Eventually (fun i => LE.le (xs i) b) F
    aux : Filter.Eventually (fun i => Eq (xs i).toReal (b.truncateToReal (xs i))) F
    aux' : Eq (Filter.limsup xs F).toReal (b.truncateToReal (Filter.limsup xs F))
    key : Eq (b.truncateToReal (Filter.limsup xs F)) (Filter.limsup (Function.comp …
    ⊢ Eq (Filter.limsup (fun a => b.truncateToReal (xs a)) F) (b.truncateToReal (F …
  -/
  rw [key]
  /-
    ι : Type u_5
    F : Filter ι
    inst✝ : F.NeBot
    b : ENNReal
    b_ne_top : Ne b Top.top
    xs : ι → ENNReal
    le_b : Filter.Eventually (fun i => LE.le (xs i) b) F
    aux : Filter.Eventually (fun i => Eq (xs i).toReal (b.truncateToReal (xs i))) F
    aux' : Eq (Filter.limsup xs F).toReal (b.truncateToReal (Filter.limsup xs F))
    key : Eq (b.truncateToReal (Filter.limsup xs F)) (Filter.limsup (Function.comp …
    ⊢ Eq (Filter.limsup (fun a => b.truncateToReal (xs a)) F) (Filter.limsup (Func …
  -/
  rfl
  /-
    🎉 no goals
  -/


