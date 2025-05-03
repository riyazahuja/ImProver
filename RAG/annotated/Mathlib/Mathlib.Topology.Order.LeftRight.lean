lemma frequently_lt_nhds (a : α) [NeBot (𝓝[<] a)] : ∃ᶠ x in 𝓝 a, x < a :=
  frequently_iff_neBot.2 ‹_›


lemma frequently_gt_nhds (a : α) [NeBot (𝓝[>] a)] : ∃ᶠ x in 𝓝 a, a < x :=
  frequently_iff_neBot.2 ‹_›


theorem Filter.Eventually.exists_lt {a : α} [NeBot (𝓝[<] a)] {p : α → Prop}
    (h : ∀ᶠ x in 𝓝 a, p x) : ∃ b < a, p b :=
  ((frequently_lt_nhds a).and_eventually h).exists


theorem Filter.Eventually.exists_gt {a : α} [NeBot (𝓝[>] a)] {p : α → Prop}
    (h : ∀ᶠ x in 𝓝 a, p x) : ∃ b > a, p b :=
  ((frequently_gt_nhds a).and_eventually h).exists


theorem nhdsWithin_Ici_neBot {a b : α} (H₂ : a ≤ b) : NeBot (𝓝[Ici a] b) :=
  nhdsWithin_neBot_of_mem H₂


instance nhdsGE_neBot (a : α) : NeBot (𝓝[≥] a) := nhdsWithin_Ici_neBot (le_refl a)


@[deprecated nhdsGE_neBot (since := "2024-12-21")]
theorem nhdsWithin_Ici_self_neBot (a : α) : NeBot (𝓝[≥] a) := nhdsGE_neBot a


theorem nhdsWithin_Iic_neBot {a b : α} (H : a ≤ b) : NeBot (𝓝[Iic b] a) :=
  nhdsWithin_neBot_of_mem H


instance nhdsLE_neBot (a : α) : NeBot (𝓝[≤] a) := nhdsWithin_Iic_neBot (le_refl a)


@[deprecated nhdsLE_neBot (since := "2024-12-21")]
theorem nhdsWithin_Iic_self_neBot (a : α) : NeBot (𝓝[≤] a) := nhdsLE_neBot a


theorem nhdsLT_le_nhdsNE (a : α) : 𝓝[<] a ≤ 𝓝[≠] a :=
  nhdsWithin_mono a fun _ => ne_of_lt


@[deprecated (since := "2024-12-21")] alias nhds_left'_le_nhds_ne := nhdsLT_le_nhdsNE


theorem nhdsGT_le_nhdsNE (a : α) : 𝓝[>] a ≤ 𝓝[≠] a := nhdsWithin_mono a fun _ => ne_of_gt


@[deprecated (since := "2024-12-21")] alias nhds_right'_le_nhds_ne := nhdsGT_le_nhdsNE

-- TODO: add instances for `NeBot (𝓝[<] x)` on (indexed) product types


lemma IsAntichain.interior_eq_empty [∀ x : α, (𝓝[<] x).NeBot] {s : Set α}
    (hs : IsAntichain (· ≤ ·) s) : interior s = ∅ := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : Preorder α
    inst✝ : ∀ (x : α), (nhdsWithin x (Set.Iio x)).NeBot
    s : Set α
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) s
    ⊢ Eq (interior s) EmptyCollection.emptyCollection
  -/
  refine eq_empty_of_forall_not_mem fun x hx ↦ ?_
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : Preorder α
    inst✝ : ∀ (x : α), (nhdsWithin x (Set.Iio x)).NeBot
    s : Set α
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) s
    x : α
    hx : Membership.mem (interior s) x
    ⊢ False
  -/
  have : ∀ᶠ y in 𝓝 x, y ∈ s := mem_interior_iff_mem_nhds.1 hx
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : Preorder α
    inst✝ : ∀ (x : α), (nhdsWithin x (Set.Iio x)).NeBot
    s : Set α
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) s
    x : α
    hx : Membership.mem (interior s) x
    this : Filter.Eventually (fun y => Membership.mem s y) (nhds x)
    ⊢ False
  -/
  rcases this.exists_lt with ⟨y, hyx, hys⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : Preorder α
    inst✝ : ∀ (x : α), (nhdsWithin x (Set.Iio x)).NeBot
    s : Set α
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) s
    x : α
    hx : Membership.mem (interior s) x
    this : Filter.Eventually (fun y => Membership.mem s y) (nhds x)
    y : α
    hyx : LT.lt y x
    hys : Membership.mem s y
    ⊢ False
  -/
  exact hs hys (interior_subset hx) hyx.ne hyx.le
  /-
    🎉 no goals
  -/


lemma IsAntichain.interior_eq_empty' [∀ x : α, (𝓝[>] x).NeBot] {s : Set α}
    (hs : IsAntichain (· ≤ ·) s) : interior s = ∅ :=
  have : ∀ x : αᵒᵈ, NeBot (𝓝[<] x) := ‹_›
  hs.to_dual.interior_eq_empty


theorem continuousWithinAt_Ioi_iff_Ici {a : α} {f : α → β} :
    ContinuousWithinAt f (Ioi a) a ↔ ContinuousWithinAt f (Ici a) a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : PartialOrder α
    inst✝ : TopologicalSpace β
    a : α
    f : α → β
    ⊢ Iff (ContinuousWithinAt f (Set.Ioi a) a) (ContinuousWithinAt f (Set.Ici a) a)
  -/
  simp only [← Ici_diff_left, continuousWithinAt_diff_self]
  /-
    🎉 no goals
  -/


theorem continuousWithinAt_Iio_iff_Iic {a : α} {f : α → β} :
    ContinuousWithinAt f (Iio a) a ↔ ContinuousWithinAt f (Iic a) a :=
  @continuousWithinAt_Ioi_iff_Ici αᵒᵈ _ _ _ _ _ f


theorem nhdsLE_sup_nhdsGE (a : α) : 𝓝[≤] a ⊔ 𝓝[≥] a = 𝓝 a := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : LinearOrder α
    a : α
    ⊢ Eq (Max.max (nhdsWithin a (Set.Iic a)) (nhdsWithin a (Set.Ici a))) (nhds a)
  -/
  rw [← nhdsWithin_union, Iic_union_Ici, nhdsWithin_univ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-21")] alias nhds_left_sup_nhds_right := nhdsLE_sup_nhdsGE


theorem nhdsLT_sup_nhdsGE (a : α) : 𝓝[<] a ⊔ 𝓝[≥] a = 𝓝 a := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : LinearOrder α
    a : α
    ⊢ Eq (Max.max (nhdsWithin a (Set.Iio a)) (nhdsWithin a (Set.Ici a))) (nhds a)
  -/
  rw [← nhdsWithin_union, Iio_union_Ici, nhdsWithin_univ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-21")] alias nhds_left'_sup_nhds_right := nhdsLT_sup_nhdsGE


theorem nhdsLE_sup_nhdsGT (a : α) : 𝓝[≤] a ⊔ 𝓝[>] a = 𝓝 a := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : LinearOrder α
    a : α
    ⊢ Eq (Max.max (nhdsWithin a (Set.Iic a)) (nhdsWithin a (Set.Ioi a))) (nhds a)
  -/
  rw [← nhdsWithin_union, Iic_union_Ioi, nhdsWithin_univ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-21")] alias nhds_left_sup_nhds_right' := nhdsLE_sup_nhdsGT


theorem nhdsLT_sup_nhdsGT (a : α) : 𝓝[<] a ⊔ 𝓝[>] a = 𝓝[≠] a := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : LinearOrder α
    a : α
    ⊢ Eq (Max.max (nhdsWithin a (Set.Iio a)) (nhdsWithin a (Set.Ioi a))) (nhdsWith …
  -/
  rw [← nhdsWithin_union, Iio_union_Ioi]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-21")] alias nhds_left'_sup_nhds_right' := nhdsLT_sup_nhdsGT


lemma nhdsWithin_right_sup_nhds_singleton (a : α) :
    𝓝[>] a ⊔ 𝓝[{a}] a = 𝓝[≥] a := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : LinearOrder α
    a : α
    ⊢ Eq (Max.max (nhdsWithin a (Set.Ioi a)) (nhdsWithin a (Singleton.singleton a) …
  -/
  simp only [union_singleton, Ioi_insert, ← nhdsWithin_union]
  /-
    🎉 no goals
  -/


theorem continuousAt_iff_continuous_left_right {a : α} {f : α → β} :
    ContinuousAt f a ↔ ContinuousWithinAt f (Iic a) a ∧ ContinuousWithinAt f (Ici a) a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : TopologicalSpace β
    a : α
    f : α → β
    ⊢ Iff (ContinuousAt f a) (And (ContinuousWithinAt f (Set.Iic a) a) (Continuous …
  -/
  simp only [ContinuousWithinAt, ContinuousAt, ← tendsto_sup, nhdsLE_sup_nhdsGE]
  /-
    🎉 no goals
  -/


theorem continuousAt_iff_continuous_left'_right' {a : α} {f : α → β} :
    ContinuousAt f a ↔ ContinuousWithinAt f (Iio a) a ∧ ContinuousWithinAt f (Ioi a) a := by
  rw [continuousWithinAt_Ioi_iff_Ici, continuousWithinAt_Iio_iff_Iic,
    continuousAt_iff_continuous_left_right]


