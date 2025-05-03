open List in
/-- The following statements are equivalent:

0. `s` is a neighborhood of `a` within `(a, +∞)`;
1. `s` is a neighborhood of `a` within `(a, b]`;
2. `s` is a neighborhood of `a` within `(a, b)`;
3. `s` includes `(a, u)` for some `u ∈ (a, b]`;
4. `s` includes `(a, u)` for some `u > a`.
-/
theorem TFAE_mem_nhdsGT {a b : α} (hab : a < b) (s : Set α) :
    TFAE [s ∈ 𝓝[>] a,
      s ∈ 𝓝[Ioc a b] a,
      s ∈ 𝓝[Ioo a b] a,
      ∃ u ∈ Ioc a b, Ioo a u ⊆ s,
      ∃ u ∈ Ioi a, Ioo a u ⊆ s] := by
  tfae_have 1 ↔ 2 := by
    rw [nhdsWithin_Ioc_eq_nhdsGT hab]
  tfae_have 1 ↔ 3 := by
    rw [nhdsWithin_Ioo_eq_nhdsGT hab]
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    a b : α
    hab : LT.lt a b
    s : Set α
    tfae_1_iff_2 : Iff (Membership.mem (nhdsWithin a (Set.Ioi a)) s) (Membership.m …
    tfae_1_iff_3 : Iff (Membership.mem (nhdsWithin a (Set.Ioi a)) s) (Membership.m …
    ⊢ (List.cons (Membership.mem (nhdsWithin a (Set.Ioi a)) s) (List.cons (Members …
  -/
  tfae_have 4 → 5 := fun ⟨u, umem, hu⟩ => ⟨u, umem.1, hu⟩
  tfae_have 5 → 1
  | ⟨u, hau, hu⟩ => mem_of_superset (Ioo_mem_nhdsGT hau) hu
  tfae_have 1 → 4
  | h => by
    rcases mem_nhdsWithin_iff_exists_mem_nhds_inter.1 h with ⟨v, va, hv⟩
    rcases exists_Ico_subset_of_mem_nhds' va hab with ⟨u, au, hu⟩
    exact ⟨u, au, fun x hx => hv ⟨hu ⟨le_of_lt hx.1, hx.2⟩, hx.1⟩⟩
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    a b : α
    hab : LT.lt a b
    s : Set α
    tfae_1_iff_2 : Iff (Membership.mem (nhdsWithin a (Set.Ioi a)) s) (Membership.m …
    tfae_1_iff_3 : Iff (Membership.mem (nhdsWithin a (Set.Ioi a)) s) (Membership.m …
    tfae_4_to_5 : (Exists fun u => And (Membership.mem (Set.Ioc a b) u) (HasSubset …
    tfae_5_to_1 : (Exists fun u => And (Membership.mem (Set.Ioi a) u) (HasSubset.S …
    tfae_1_to_4 : Membership.mem (nhdsWithin a (Set.Ioi a)) s → Exists fun u => An …
    ⊢ (List.cons (Membership.mem (nhdsWithin a (Set.Ioi a)) s) (List.cons (Members …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias TFAE_mem_nhdsWithin_Ioi := TFAE_mem_nhdsGT


theorem mem_nhdsGT_iff_exists_mem_Ioc_Ioo_subset {a u' : α} {s : Set α} (hu' : a < u') :
    s ∈ 𝓝[>] a ↔ ∃ u ∈ Ioc a u', Ioo a u ⊆ s :=
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    a u' : α
    s : Set α
    hu' : LT.lt a u'
    ⊢ Eq ((List.cons (Membership.mem (nhdsWithin a (Set.Ioi a)) s) (List.cons (Mem …
  -/
  /-
    🎉 no goals
  -/
  (TFAE_mem_nhdsGT hu' s).out 0 3
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias mem_nhdsWithin_Ioi_iff_exists_mem_Ioc_Ioo_subset := mem_nhdsGT_iff_exists_mem_Ioc_Ioo_subset


/-- A set is a neighborhood of `a` within `(a, +∞)` if and only if it contains an interval `(a, u)`
with `a < u < u'`, provided `a` is not a top element. -/
theorem mem_nhdsGT_iff_exists_Ioo_subset' {a u' : α} {s : Set α} (hu' : a < u') :
    s ∈ 𝓝[>] a ↔ ∃ u ∈ Ioi a, Ioo a u ⊆ s :=
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    a u' : α
    s : Set α
    hu' : LT.lt a u'
    ⊢ Eq ((List.cons (Membership.mem (nhdsWithin a (Set.Ioi a)) s) (List.cons (Mem …
  -/
  /-
    🎉 no goals
  -/
  (TFAE_mem_nhdsGT hu' s).out 0 4
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias mem_nhdsWithin_Ioi_iff_exists_Ioo_subset' := mem_nhdsGT_iff_exists_Ioo_subset'


theorem nhdsGT_basis_of_exists_gt {a : α} (h : ∃ b, a < b) : (𝓝[>] a).HasBasis (a < ·) (Ioo a) :=
  let ⟨_, h⟩ := h
  ⟨fun _ => mem_nhdsGT_iff_exists_Ioo_subset' h⟩


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_Ioi_basis' := nhdsGT_basis_of_exists_gt


lemma nhdsGT_basis [NoMaxOrder α] (a : α) : (𝓝[>] a).HasBasis (a < ·) (Ioo a) :=
  nhdsGT_basis_of_exists_gt <| exists_gt a


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_Ioi_basis := nhdsGT_basis


theorem nhdsGT_eq_bot_iff {a : α} : 𝓝[>] a = ⊥ ↔ IsTop a ∨ ∃ b, a ⋖ b := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    a : α
    ⊢ Iff (Eq (nhdsWithin a (Set.Ioi a)) Bot.bot) (Or (IsTop a) (Exists fun b => C …
  -/
  by_cases ha : IsTop a
    /-
      case pos
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      a : α
      ha : IsTop a
      ⊢ Iff (Eq (nhdsWithin a (Set.Ioi a)) Bot.bot) (Or (IsTop a) (Exists fun b => C …
    -/
  · simp [ha, ha.isMax.Ioi_eq]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      a : α
      ha : Not (IsTop a)
      ⊢ Iff (Eq (nhdsWithin a (Set.Ioi a)) Bot.bot) (Or (IsTop a) (Exists fun b => C …
    -/
  · simp only [ha, false_or]
    /-
      case neg
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      a : α
      ha : Not (IsTop a)
      ⊢ Iff (Eq (nhdsWithin a (Set.Ioi a)) Bot.bot) (Exists fun b => CovBy a b)
    -/
    rw [isTop_iff_isMax, not_isMax_iff] at ha
    /-
      case neg
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      a : α
      ha : Exists fun b => LT.lt a b
      ⊢ Iff (Eq (nhdsWithin a (Set.Ioi a)) Bot.bot) (Exists fun b => CovBy a b)
    -/
    simp only [(nhdsGT_basis_of_exists_gt ha).eq_bot_iff, covBy_iff_Ioo_eq]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_Ioi_eq_bot_iff := nhdsGT_eq_bot_iff


/-- A set is a neighborhood of `a` within `(a, +∞)` if and only if it contains an interval `(a, u)`
with `a < u`. -/
theorem mem_nhdsGT_iff_exists_Ioo_subset [NoMaxOrder α] {a : α} {s : Set α} :
    s ∈ 𝓝[>] a ↔ ∃ u ∈ Ioi a, Ioo a u ⊆ s :=
  let ⟨_u', hu'⟩ := exists_gt a
  mem_nhdsGT_iff_exists_Ioo_subset' hu'


@[deprecated (since := "2024-12-22")]
alias mem_nhdsWithin_Ioi_iff_exists_Ioo_subset := mem_nhdsGT_iff_exists_Ioo_subset


/-- The set of points which are isolated on the right is countable when the space is
second-countable. -/
theorem countable_setOf_isolated_right [SecondCountableTopology α] :
    { x : α | 𝓝[>] x = ⊥ }.Countable := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    ⊢ (setOf fun x => Eq (nhdsWithin x (Set.Ioi x)) Bot.bot).Countable
  -/
  simp only [nhdsGT_eq_bot_iff, setOf_or]
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    ⊢ (Union.union (setOf fun a => IsTop a) (setOf fun a => Exists fun b => CovBy  …
  -/
  exact (subsingleton_isTop α).countable.union countable_setOf_covBy_right
  /-
    🎉 no goals
  -/


/-- The set of points which are isolated on the left is countable when the space is
second-countable. -/
theorem countable_setOf_isolated_left [SecondCountableTopology α] :
    { x : α | 𝓝[<] x = ⊥ }.Countable :=
  countable_setOf_isolated_right (α := αᵒᵈ)


/-- A set is a neighborhood of `a` within `(a, +∞)` if and only if it contains an interval `(a, u]`
with `a < u`. -/
theorem mem_nhdsGT_iff_exists_Ioc_subset [NoMaxOrder α] [DenselyOrdered α] {a : α} {s : Set α} :
    s ∈ 𝓝[>] a ↔ ∃ u ∈ Ioi a, Ioc a u ⊆ s := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : NoMaxOrder α
    inst✝ : DenselyOrdered α
    a : α
    s : Set α
    ⊢ Iff (Membership.mem (nhdsWithin a (Set.Ioi a)) s) (Exists fun u => And (Memb …
  -/
  rw [mem_nhdsGT_iff_exists_Ioo_subset]
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : NoMaxOrder α
    inst✝ : DenselyOrdered α
    a : α
    s : Set α
    ⊢ Iff (Exists fun u => And (Membership.mem (Set.Ioi a) u) (HasSubset.Subset (S …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : NoMaxOrder α
      inst✝ : DenselyOrdered α
      a : α
      s : Set α
      ⊢ (Exists fun u => And (Membership.mem (Set.Ioi a) u) (HasSubset.Subset (Set.I …
    -/
  · rintro ⟨u, au, as⟩
    /-
      case mp.intro.intro
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : NoMaxOrder α
      inst✝ : DenselyOrdered α
      a : α
      s : Set α
      u : α
      au : Membership.mem (Set.Ioi a) u
      as : HasSubset.Subset (Set.Ioo a u) s
      ⊢ Exists fun u => And (Membership.mem (Set.Ioi a) u) (HasSubset.Subset (Set.Io …
    -/
    rcases exists_between au with ⟨v, hv⟩
    /-
      case mp.intro.intro.intro
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : NoMaxOrder α
      inst✝ : DenselyOrdered α
      a : α
      s : Set α
      u : α
      au : Membership.mem (Set.Ioi a) u
      as : HasSubset.Subset (Set.Ioo a u) s
      v : α
      hv : And (LT.lt a v) (LT.lt v u)
      ⊢ Exists fun u => And (Membership.mem (Set.Ioi a) u) (HasSubset.Subset (Set.Io …
    -/
    exact ⟨v, hv.1, fun x hx => as ⟨hx.1, lt_of_le_of_lt hx.2 hv.2⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : NoMaxOrder α
      inst✝ : DenselyOrdered α
      a : α
      s : Set α
      ⊢ (Exists fun u => And (Membership.mem (Set.Ioi a) u) (HasSubset.Subset (Set.I …
    -/
  · rintro ⟨u, au, as⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : NoMaxOrder α
      inst✝ : DenselyOrdered α
      a : α
      s : Set α
      u : α
      au : Membership.mem (Set.Ioi a) u
      as : HasSubset.Subset (Set.Ioc a u) s
      ⊢ Exists fun u => And (Membership.mem (Set.Ioi a) u) (HasSubset.Subset (Set.Io …
    -/
    exact ⟨u, au, Subset.trans Ioo_subset_Ioc_self as⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-22")]
alias mem_nhdsWithin_Ioi_iff_exists_Ioc_subset := mem_nhdsGT_iff_exists_Ioc_subset


open List in
/-- The following statements are equivalent:

0. `s` is a neighborhood of `b` within `(-∞, b)`
1. `s` is a neighborhood of `b` within `[a, b)`
2. `s` is a neighborhood of `b` within `(a, b)`
3. `s` includes `(l, b)` for some `l ∈ [a, b)`
4. `s` includes `(l, b)` for some `l < b` -/
theorem TFAE_mem_nhdsLT {a b : α} (h : a < b) (s : Set α) :
    TFAE [s ∈ 𝓝[<] b,-- 0 : `s` is a neighborhood of `b` within `(-∞, b)`
        s ∈ 𝓝[Ico a b] b,-- 1 : `s` is a neighborhood of `b` within `[a, b)`
        s ∈ 𝓝[Ioo a b] b,-- 2 : `s` is a neighborhood of `b` within `(a, b)`
        ∃ l ∈ Ico a b, Ioo l b ⊆ s,-- 3 : `s` includes `(l, b)` for some `l ∈ [a, b)`
        ∃ l ∈ Iio b, Ioo l b ⊆ s] := by-- 4 : `s` includes `(l, b)` for some `l < b`
  simpa only [exists_prop, OrderDual.exists, dual_Ioi, dual_Ioc, dual_Ioo] using
    TFAE_mem_nhdsGT h.dual (ofDual ⁻¹' s)


@[deprecated (since := "2024-12-22")]
alias TFAE_mem_nhdsWithin_Iio := TFAE_mem_nhdsLT


theorem mem_nhdsLT_iff_exists_mem_Ico_Ioo_subset {a l' : α} {s : Set α} (hl' : l' < a) :
    s ∈ 𝓝[<] a ↔ ∃ l ∈ Ico l' a, Ioo l a ⊆ s :=
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    a l' : α
    s : Set α
    hl' : LT.lt l' a
    ⊢ Eq ((List.cons (Membership.mem (nhdsWithin a (Set.Iio a)) s) (List.cons (Mem …
  -/
  /-
    🎉 no goals
  -/
  (TFAE_mem_nhdsLT hl' s).out 0 3
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias mem_nhdsWithin_Iio_iff_exists_mem_Ico_Ioo_subset := mem_nhdsLT_iff_exists_mem_Ico_Ioo_subset


/-- A set is a neighborhood of `a` within `(-∞, a)` if and only if it contains an interval `(l, a)`
with `l < a`, provided `a` is not a bottom element. -/
theorem mem_nhdsLT_iff_exists_Ioo_subset' {a l' : α} {s : Set α} (hl' : l' < a) :
    s ∈ 𝓝[<] a ↔ ∃ l ∈ Iio a, Ioo l a ⊆ s :=
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    a l' : α
    s : Set α
    hl' : LT.lt l' a
    ⊢ Eq ((List.cons (Membership.mem (nhdsWithin a (Set.Iio a)) s) (List.cons (Mem …
  -/
  /-
    🎉 no goals
  -/
  (TFAE_mem_nhdsLT hl' s).out 0 4
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias mem_nhdsWithin_Iio_iff_exists_Ioo_subset' := mem_nhdsLT_iff_exists_Ioo_subset'


/-- A set is a neighborhood of `a` within `(-∞, a)` if and only if it contains an interval `(l, a)`
with `l < a`. -/
theorem mem_nhdsLT_iff_exists_Ioo_subset [NoMinOrder α] {a : α} {s : Set α} :
    s ∈ 𝓝[<] a ↔ ∃ l ∈ Iio a, Ioo l a ⊆ s :=
  let ⟨_, h⟩ := exists_lt a
  mem_nhdsLT_iff_exists_Ioo_subset' h


@[deprecated (since := "2024-12-22")]
alias mem_nhdsWithin_Iio_iff_exists_Ioo_subset := mem_nhdsLT_iff_exists_Ioo_subset


/-- A set is a neighborhood of `a` within `(-∞, a)` if and only if it contains an interval `[l, a)`
with `l < a`. -/
theorem mem_nhdsLT_iff_exists_Ico_subset [NoMinOrder α] [DenselyOrdered α] {a : α} {s : Set α} :
    s ∈ 𝓝[<] a ↔ ∃ l ∈ Iio a, Ico l a ⊆ s := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : NoMinOrder α
    inst✝ : DenselyOrdered α
    a : α
    s : Set α
    ⊢ Iff (Membership.mem (nhdsWithin a (Set.Iio a)) s) (Exists fun l => And (Memb …
  -/
  have : ofDual ⁻¹' s ∈ 𝓝[>] toDual a ↔ _ := mem_nhdsGT_iff_exists_Ioc_subset
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : NoMinOrder α
    inst✝ : DenselyOrdered α
    a : α
    s : Set α
    this : Iff (Membership.mem (nhdsWithin (OrderDual.toDual a) (Set.Ioi (OrderDua …
    ⊢ Iff (Membership.mem (nhdsWithin a (Set.Iio a)) s) (Exists fun l => And (Memb …
  -/
  simpa only [OrderDual.exists, exists_prop, dual_Ioc] using this
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias mem_nhdsWithin_Iio_iff_exists_Ico_subset := mem_nhdsLT_iff_exists_Ico_subset


theorem nhdsLT_basis_of_exists_lt {a : α} (h : ∃ b, b < a) : (𝓝[<] a).HasBasis (· < a) (Ioo · a) :=
  let ⟨_, h⟩ := h
  ⟨fun _ => mem_nhdsLT_iff_exists_Ioo_subset' h⟩


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_Iio_basis' := nhdsLT_basis_of_exists_lt


theorem nhdsLT_basis [NoMinOrder α] (a : α) : (𝓝[<] a).HasBasis (· < a) (Ioo · a) :=
  nhdsLT_basis_of_exists_lt <| exists_lt a


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_Iio_basis := nhdsLT_basis


theorem nhdsLT_eq_bot_iff {a : α} : 𝓝[<] a = ⊥ ↔ IsBot a ∨ ∃ b, b ⋖ a := by
  convert (config := { preTransparency := .default }) nhdsGT_eq_bot_iff (a := OrderDual.toDual a)
    using 4
  /-
    case h.e'_2.h.e'_2.h.e'_2.h.h.a
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    a : α
    e_1✝ : Eq α (OrderDual α)
    x✝ : α
    ⊢ Iff (CovBy x✝ a) (CovBy (OrderDual.toDual a) x✝)
  -/
  exact ofDual_covBy_ofDual_iff
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_Iio_eq_bot_iff := nhdsLT_eq_bot_iff


open List in
/-- The following statements are equivalent:

0. `s` is a neighborhood of `a` within `[a, +∞)`;
1. `s` is a neighborhood of `a` within `[a, b]`;
2. `s` is a neighborhood of `a` within `[a, b)`;
3. `s` includes `[a, u)` for some `u ∈ (a, b]`;
4. `s` includes `[a, u)` for some `u > a`.
-/
theorem TFAE_mem_nhdsGE {a b : α} (hab : a < b) (s : Set α) :
    TFAE [s ∈ 𝓝[≥] a,
      s ∈ 𝓝[Icc a b] a,
      s ∈ 𝓝[Ico a b] a,
      ∃ u ∈ Ioc a b, Ico a u ⊆ s,
      ∃ u ∈ Ioi a , Ico a u ⊆ s] := by
  tfae_have 1 ↔ 2 := by
    rw [nhdsWithin_Icc_eq_nhdsGE hab]
  tfae_have 1 ↔ 3 := by
    rw [nhdsWithin_Ico_eq_nhdsGE hab]
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    a b : α
    hab : LT.lt a b
    s : Set α
    tfae_1_iff_2 : Iff (Membership.mem (nhdsWithin a (Set.Ici a)) s) (Membership.m …
    tfae_1_iff_3 : Iff (Membership.mem (nhdsWithin a (Set.Ici a)) s) (Membership.m …
    ⊢ (List.cons (Membership.mem (nhdsWithin a (Set.Ici a)) s) (List.cons (Members …
  -/
  tfae_have 1 ↔ 5 := (nhdsGE_basis_of_exists_gt ⟨b, hab⟩).mem_iff
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    a b : α
    hab : LT.lt a b
    s : Set α
    tfae_1_iff_2 : Iff (Membership.mem (nhdsWithin a (Set.Ici a)) s) (Membership.m …
    tfae_1_iff_3 : Iff (Membership.mem (nhdsWithin a (Set.Ici a)) s) (Membership.m …
    tfae_1_iff_5 : Iff (Membership.mem (nhdsWithin a (Set.Ici a)) s) (Exists fun u …
    ⊢ (List.cons (Membership.mem (nhdsWithin a (Set.Ici a)) s) (List.cons (Members …
  -/
  tfae_have 4 → 5 := fun ⟨u, umem, hu⟩ => ⟨u, umem.1, hu⟩
  tfae_have 5 → 4
  | ⟨u, hua, hus⟩ => ⟨min u b, ⟨lt_min hua hab, min_le_right _ _⟩,
      (Ico_subset_Ico_right <| min_le_left _ _).trans hus⟩
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    a b : α
    hab : LT.lt a b
    s : Set α
    tfae_1_iff_2 : Iff (Membership.mem (nhdsWithin a (Set.Ici a)) s) (Membership.m …
    tfae_1_iff_3 : Iff (Membership.mem (nhdsWithin a (Set.Ici a)) s) (Membership.m …
    tfae_1_iff_5 : Iff (Membership.mem (nhdsWithin a (Set.Ici a)) s) (Exists fun u …
    tfae_4_to_5 : (Exists fun u => And (Membership.mem (Set.Ioc a b) u) (HasSubset …
    tfae_5_to_4 : (Exists fun u => And (Membership.mem (Set.Ioi a) u) (HasSubset.S …
    ⊢ (List.cons (Membership.mem (nhdsWithin a (Set.Ici a)) s) (List.cons (Members …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias TFAE_mem_nhdsWithin_Ici := TFAE_mem_nhdsGE


theorem mem_nhdsGE_iff_exists_mem_Ioc_Ico_subset {a u' : α} {s : Set α} (hu' : a < u') :
    s ∈ 𝓝[≥] a ↔ ∃ u ∈ Ioc a u', Ico a u ⊆ s :=
                                      /-
                                        α : Type u_1
                                        inst✝² : TopologicalSpace α
                                        inst✝¹ : LinearOrder α
                                        inst✝ : OrderTopology α
                                        a u' : α
                                        s : Set α
                                        hu' : LT.lt a u'
                                        ⊢ Eq ((List.cons (Membership.mem (nhdsWithin a (Set.Ici a)) s) (List.cons (Mem …
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
  (TFAE_mem_nhdsGE hu' s).out 0 3 (by norm_num) (by norm_num)
                                                    /-
                                                      🎉 no goals
                                                    -/


@[deprecated (since := "2024-12-22")]
alias mem_nhdsWithin_Ici_iff_exists_mem_Ioc_Ico_subset := mem_nhdsGE_iff_exists_mem_Ioc_Ico_subset


/-- A set is a neighborhood of `a` within `[a, +∞)` if and only if it contains an interval `[a, u)`
with `a < u < u'`, provided `a` is not a top element. -/
theorem mem_nhdsGE_iff_exists_Ico_subset' {a u' : α} {s : Set α} (hu' : a < u') :
    s ∈ 𝓝[≥] a ↔ ∃ u ∈ Ioi a, Ico a u ⊆ s :=
                                      /-
                                        α : Type u_1
                                        inst✝² : TopologicalSpace α
                                        inst✝¹ : LinearOrder α
                                        inst✝ : OrderTopology α
                                        a u' : α
                                        s : Set α
                                        hu' : LT.lt a u'
                                        ⊢ Eq ((List.cons (Membership.mem (nhdsWithin a (Set.Ici a)) s) (List.cons (Mem …
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
  (TFAE_mem_nhdsGE hu' s).out 0 4 (by norm_num) (by norm_num)
                                                    /-
                                                      🎉 no goals
                                                    -/


@[deprecated (since := "2024-12-22")]
alias mem_nhdsWithin_Ici_iff_exists_Ico_subset' := mem_nhdsGE_iff_exists_Ico_subset'


/-- A set is a neighborhood of `a` within `[a, +∞)` if and only if it contains an interval `[a, u)`
with `a < u`. -/
theorem mem_nhdsGE_iff_exists_Ico_subset [NoMaxOrder α] {a : α} {s : Set α} :
    s ∈ 𝓝[≥] a ↔ ∃ u ∈ Ioi a, Ico a u ⊆ s :=
  let ⟨_, hu'⟩ := exists_gt a
  mem_nhdsGE_iff_exists_Ico_subset' hu'


@[deprecated (since := "2024-12-22")]
alias mem_nhdsWithin_Ici_iff_exists_Ico_subset := mem_nhdsGE_iff_exists_Ico_subset


theorem nhdsGE_basis_Ico [NoMaxOrder α] (a : α) : (𝓝[≥] a).HasBasis (fun u => a < u) (Ico a) :=
  ⟨fun _ => mem_nhdsGE_iff_exists_Ico_subset⟩


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_Ici_basis_Ico := nhdsGE_basis_Ico


/-- The filter of right neighborhoods has a basis of closed intervals. -/
theorem nhdsGE_basis_Icc [NoMaxOrder α] [DenselyOrdered α] {a : α} :
    (𝓝[≥] a).HasBasis (a < ·) (Icc a) :=
  (nhdsGE_basis _).to_hasBasis
    (fun _u hu ↦ (exists_between hu).imp fun _v hv ↦ hv.imp_right Icc_subset_Ico_right) fun u hu ↦
    ⟨u, hu, Ico_subset_Icc_self⟩


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_Ici_basis_Icc := nhdsGE_basis_Icc


/-- A set is a neighborhood of `a` within `[a, +∞)` if and only if it contains an interval `[a, u]`
with `a < u`. -/
theorem mem_nhdsGE_iff_exists_Icc_subset [NoMaxOrder α] [DenselyOrdered α] {a : α} {s : Set α} :
    s ∈ 𝓝[≥] a ↔ ∃ u, a < u ∧ Icc a u ⊆ s :=
  nhdsGE_basis_Icc.mem_iff


@[deprecated (since := "2024-12-22")]
alias mem_nhdsWithin_Ici_iff_exists_Icc_subset := mem_nhdsGE_iff_exists_Icc_subset


open List in
/-- The following statements are equivalent:

0. `s` is a neighborhood of `b` within `(-∞, b]`
1. `s` is a neighborhood of `b` within `[a, b]`
2. `s` is a neighborhood of `b` within `(a, b]`
3. `s` includes `(l, b]` for some `l ∈ [a, b)`
4. `s` includes `(l, b]` for some `l < b` -/
theorem TFAE_mem_nhdsLE {a b : α} (h : a < b) (s : Set α) :
    TFAE [s ∈ 𝓝[≤] b,-- 0 : `s` is a neighborhood of `b` within `(-∞, b]`
      s ∈ 𝓝[Icc a b] b,-- 1 : `s` is a neighborhood of `b` within `[a, b]`
      s ∈ 𝓝[Ioc a b] b,-- 2 : `s` is a neighborhood of `b` within `(a, b]`
      ∃ l ∈ Ico a b, Ioc l b ⊆ s,-- 3 : `s` includes `(l, b]` for some `l ∈ [a, b)`
      ∃ l ∈ Iio b, Ioc l b ⊆ s] := by-- 4 : `s` includes `(l, b]` for some `l < b`
  simpa only [exists_prop, OrderDual.exists, dual_Ici, dual_Ioc, dual_Icc, dual_Ico] using
    TFAE_mem_nhdsGE h.dual (ofDual ⁻¹' s)


@[deprecated (since := "2024-12-22")]
alias TFAE_mem_nhdsWithin_Iic := TFAE_mem_nhdsLE


theorem mem_nhdsLE_iff_exists_mem_Ico_Ioc_subset {a l' : α} {s : Set α} (hl' : l' < a) :
    s ∈ 𝓝[≤] a ↔ ∃ l ∈ Ico l' a, Ioc l a ⊆ s :=
                                      /-
                                        α : Type u_1
                                        inst✝² : TopologicalSpace α
                                        inst✝¹ : LinearOrder α
                                        inst✝ : OrderTopology α
                                        a l' : α
                                        s : Set α
                                        hl' : LT.lt l' a
                                        ⊢ Eq ((List.cons (Membership.mem (nhdsWithin a (Set.Iic a)) s) (List.cons (Mem …
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
  (TFAE_mem_nhdsLE hl' s).out 0 3 (by norm_num) (by norm_num)
                                                    /-
                                                      🎉 no goals
                                                    -/


@[deprecated (since := "2024-12-22")]
alias mem_nhdsWithin_Iic_iff_exists_mem_Ico_Ioc_subset := mem_nhdsLE_iff_exists_mem_Ico_Ioc_subset


/-- A set is a neighborhood of `a` within `(-∞, a]` if and only if it contains an interval `(l, a]`
with `l < a`, provided `a` is not a bottom element. -/
theorem mem_nhdsLE_iff_exists_Ioc_subset' {a l' : α} {s : Set α} (hl' : l' < a) :
    s ∈ 𝓝[≤] a ↔ ∃ l ∈ Iio a, Ioc l a ⊆ s :=
                                      /-
                                        α : Type u_1
                                        inst✝² : TopologicalSpace α
                                        inst✝¹ : LinearOrder α
                                        inst✝ : OrderTopology α
                                        a l' : α
                                        s : Set α
                                        hl' : LT.lt l' a
                                        ⊢ Eq ((List.cons (Membership.mem (nhdsWithin a (Set.Iic a)) s) (List.cons (Mem …
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
  (TFAE_mem_nhdsLE hl' s).out 0 4 (by norm_num) (by norm_num)
                                                    /-
                                                      🎉 no goals
                                                    -/


@[deprecated (since := "2024-12-22")]
alias mem_nhdsWithin_Iic_iff_exists_Ioc_subset' := mem_nhdsLE_iff_exists_Ioc_subset'


/-- A set is a neighborhood of `a` within `(-∞, a]` if and only if it contains an interval `(l, a]`
with `l < a`. -/
theorem mem_nhdsLE_iff_exists_Ioc_subset [NoMinOrder α] {a : α} {s : Set α} :
    s ∈ 𝓝[≤] a ↔ ∃ l ∈ Iio a, Ioc l a ⊆ s :=
  let ⟨_, hl'⟩ := exists_lt a
  mem_nhdsLE_iff_exists_Ioc_subset' hl'


@[deprecated (since := "2024-12-22")]
alias mem_nhdsWithin_Iic_iff_exists_Ioc_subset := mem_nhdsLE_iff_exists_Ioc_subset


/-- A set is a neighborhood of `a` within `(-∞, a]` if and only if it contains an interval `[l, a]`
with `l < a`. -/
theorem mem_nhdsLE_iff_exists_Icc_subset [NoMinOrder α] [DenselyOrdered α] {a : α}
    {s : Set α} : s ∈ 𝓝[≤] a ↔ ∃ l, l < a ∧ Icc l a ⊆ s :=
  calc s ∈ 𝓝[≤] a ↔ ofDual ⁻¹' s ∈ 𝓝[≥] (toDual a) := Iff.rfl
  _ ↔ ∃ u : α, toDual a < toDual u ∧ Icc (toDual a) (toDual u) ⊆ ofDual ⁻¹' s :=
    mem_nhdsGE_iff_exists_Icc_subset
                                     /-
                                       α : Type u_1
                                       inst✝⁴ : TopologicalSpace α
                                       inst✝³ : LinearOrder α
                                       inst✝² : OrderTopology α
                                       inst✝¹ : NoMinOrder α
                                       inst✝ : DenselyOrdered α
                                       a : α
                                       s : Set α
                                       ⊢ Iff (Exists fun u => And (LT.lt (OrderDual.toDual a) (OrderDual.toDual u)) ( …
                                     -/
  _ ↔ ∃ l, l < a ∧ Icc l a ⊆ s := by simp only [dual_Icc]; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/


@[deprecated (since := "2024-12-22")]
alias mem_nhdsWithin_Iic_iff_exists_Icc_subset := mem_nhdsLE_iff_exists_Icc_subset


/-- The filter of left neighborhoods has a basis of closed intervals. -/
theorem nhdsLE_basis_Icc [NoMinOrder α] [DenselyOrdered α] {a : α} :
    (𝓝[≤] a).HasBasis (· < a) (Icc · a) :=
  ⟨fun _ ↦ mem_nhdsLE_iff_exists_Icc_subset⟩


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_Iic_basis_Icc := nhdsLE_basis_Icc


theorem nhds_eq_iInf_abs_sub (a : α) : 𝓝 a = ⨅ r > 0, 𝓟 { b | |a - b| < r } := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : OrderTopology α
    a : α
    ⊢ Eq (nhds a) (iInf fun r => iInf fun h => Filter.principal (setOf fun b => LT …
  -/
  simp only [nhds_eq_order, abs_lt, setOf_and, ← inf_principal, iInf_inf_eq]
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : OrderTopology α
    a : α
    ⊢ Eq (Min.min (iInf fun b => iInf fun h => Filter.principal (Set.Ioi b)) (iInf …
  -/
  refine (congr_arg₂ _ ?_ ?_).trans (inf_comm ..)
    /-
      case refine_1
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrderedAddCommGroup α
      inst✝ : OrderTopology α
      a : α
      ⊢ Eq (iInf fun b => iInf fun h => Filter.principal (Set.Ioi b)) (iInf fun x => …
    -/
  · refine (Equiv.subLeft a).iInf_congr fun x => ?_; simp [Ioi]
                                                     /-
                                                       🎉 no goals
                                                     -/
    /-
      case refine_2
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrderedAddCommGroup α
      inst✝ : OrderTopology α
      a : α
      ⊢ Eq (iInf fun b => iInf fun h => Filter.principal (Set.Iio b)) (iInf fun x => …
    -/
  · refine (Equiv.subRight a).iInf_congr fun x => ?_; simp [Iio]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem orderTopology_of_nhds_abs {α : Type*} [TopologicalSpace α] [LinearOrderedAddCommGroup α]
    (h_nhds : ∀ a : α, 𝓝 a = ⨅ r > 0, 𝓟 { b | |a - b| < r }) : OrderTopology α := by
  /-
    α : Type u_4
    inst✝¹ : TopologicalSpace α
    inst✝ : LinearOrderedAddCommGroup α
    h_nhds : ∀ (a : α), Eq (nhds a) (iInf fun r => iInf fun h => Filter.principal  …
    ⊢ OrderTopology α
  -/
  refine ⟨TopologicalSpace.ext_nhds fun a => ?_⟩
  /-
    α : Type u_4
    inst✝¹ : TopologicalSpace α
    inst✝ : LinearOrderedAddCommGroup α
    h_nhds : ∀ (a : α), Eq (nhds a) (iInf fun r => iInf fun h => Filter.principal  …
    a : α
    ⊢ Eq (nhds a) (nhds a)
  -/
  rw [h_nhds]
  /-
    α : Type u_4
    inst✝¹ : TopologicalSpace α
    inst✝ : LinearOrderedAddCommGroup α
    h_nhds : ∀ (a : α), Eq (nhds a) (iInf fun r => iInf fun h => Filter.principal  …
    a : α
    ⊢ Eq (iInf fun r => iInf fun h => Filter.principal (setOf fun b => LT.lt (abs  …
  -/
  letI := Preorder.topology α; letI : OrderTopology α := ⟨rfl⟩
  /-
    α : Type u_4
    inst✝¹ : TopologicalSpace α
    inst✝ : LinearOrderedAddCommGroup α
    h_nhds : ∀ (a : α), Eq (nhds a) (iInf fun r => iInf fun h => Filter.principal  …
    a : α
    this✝ : TopologicalSpace α := Preorder.topology α
    this : OrderTopology α := { topology_eq_generate_intervals := rfl }
    ⊢ Eq (iInf fun r => iInf fun h => Filter.principal (setOf fun b => LT.lt (abs  …
  -/
  exact (nhds_eq_iInf_abs_sub a).symm
  /-
    🎉 no goals
  -/


theorem LinearOrderedAddCommGroup.tendsto_nhds {x : Filter β} {a : α} :
    Tendsto f x (𝓝 a) ↔ ∀ ε > (0 : α), ∀ᶠ b in x, |f b - a| < ε := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : OrderTopology α
    f : β → α
    x : Filter β
    a : α
    ⊢ Iff (Filter.Tendsto f x (nhds a)) (∀ (ε : α), GT.gt ε 0 → Filter.Eventually  …
  -/
  simp [nhds_eq_iInf_abs_sub, abs_sub_comm a]
  /-
    🎉 no goals
  -/


theorem eventually_abs_sub_lt (a : α) {ε : α} (hε : 0 < ε) : ∀ᶠ x in 𝓝 a, |x - a| < ε :=
  (nhds_eq_iInf_abs_sub a).symm ▸
                                                /-
                                                  α : Type u_1
                                                  inst✝² : TopologicalSpace α
                                                  inst✝¹ : LinearOrderedAddCommGroup α
                                                  inst✝ : OrderTopology α
                                                  a ε : α
                                                  hε : LT.lt 0 ε
                                                  ⊢ Membership.mem (Filter.principal (setOf fun b => LT.lt (abs (HSub.hSub a b)) …
                                                -/
    mem_iInf_of_mem ε (mem_iInf_of_mem hε <| by simp only [abs_sub_comm, mem_principal_self])
                                                /-
                                                  🎉 no goals
                                                -/


/-- In a linearly ordered additive commutative group with the order topology, if `f` tends to `C`
and `g` tends to `atTop` then `f + g` tends to `atTop`. -/
theorem Filter.Tendsto.add_atTop {C : α} (hf : Tendsto f l (𝓝 C)) (hg : Tendsto g l atTop) :
    Tendsto (fun x => f x + g x) l atTop := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : OrderTopology α
    l : Filter β
    f g : β → α
    C : α
    hf : Filter.Tendsto f l (nhds C)
    hg : Filter.Tendsto g l Filter.atTop
    ⊢ Filter.Tendsto (fun x => HAdd.hAdd (f x) (g x)) l Filter.atTop
  -/
  nontriviality α
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : OrderTopology α
    l : Filter β
    f g : β → α
    C : α
    hf : Filter.Tendsto f l (nhds C)
    hg : Filter.Tendsto g l Filter.atTop
    a✝ : Nontrivial α
    ⊢ Filter.Tendsto (fun x => HAdd.hAdd (f x) (g x)) l Filter.atTop
  -/
  obtain ⟨C', hC'⟩ : ∃ C', C' < C := exists_lt C
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : OrderTopology α
    l : Filter β
    f g : β → α
    C : α
    hf : Filter.Tendsto f l (nhds C)
    hg : Filter.Tendsto g l Filter.atTop
    a✝ : Nontrivial α
    C' : α
    hC' : LT.lt C' C
    ⊢ Filter.Tendsto (fun x => HAdd.hAdd (f x) (g x)) l Filter.atTop
  -/
  refine tendsto_atTop_add_left_of_le' _ C' ?_ hg
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : OrderTopology α
    l : Filter β
    f g : β → α
    C : α
    hf : Filter.Tendsto f l (nhds C)
    hg : Filter.Tendsto g l Filter.atTop
    a✝ : Nontrivial α
    C' : α
    hC' : LT.lt C' C
    ⊢ Filter.Eventually (fun x => LE.le C' (f x)) l
  -/
  exact (hf.eventually (lt_mem_nhds hC')).mono fun x => le_of_lt
  /-
    🎉 no goals
  -/


/-- In a linearly ordered additive commutative group with the order topology, if `f` tends to `C`
and `g` tends to `atBot` then `f + g` tends to `atBot`. -/
theorem Filter.Tendsto.add_atBot {C : α} (hf : Tendsto f l (𝓝 C)) (hg : Tendsto g l atBot) :
    Tendsto (fun x => f x + g x) l atBot :=
  Filter.Tendsto.add_atTop (α := αᵒᵈ) hf hg


/-- In a linearly ordered additive commutative group with the order topology, if `f` tends to
`atTop` and `g` tends to `C` then `f + g` tends to `atTop`. -/
theorem Filter.Tendsto.atTop_add {C : α} (hf : Tendsto f l atTop) (hg : Tendsto g l (𝓝 C)) :
    Tendsto (fun x => f x + g x) l atTop := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : OrderTopology α
    l : Filter β
    f g : β → α
    C : α
    hf : Filter.Tendsto f l Filter.atTop
    hg : Filter.Tendsto g l (nhds C)
    ⊢ Filter.Tendsto (fun x => HAdd.hAdd (f x) (g x)) l Filter.atTop
  -/
  conv in _ + _ => rw [add_comm]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : OrderTopology α
    l : Filter β
    f g : β → α
    C : α
    hf : Filter.Tendsto f l Filter.atTop
    hg : Filter.Tendsto g l (nhds C)
    ⊢ Filter.Tendsto (fun x => HAdd.hAdd (g x) (f x)) l Filter.atTop
  -/
  exact hg.add_atTop hf
  /-
    🎉 no goals
  -/


/-- In a linearly ordered additive commutative group with the order topology, if `f` tends to
`atBot` and `g` tends to `C` then `f + g` tends to `atBot`. -/
theorem Filter.Tendsto.atBot_add {C : α} (hf : Tendsto f l atBot) (hg : Tendsto g l (𝓝 C)) :
    Tendsto (fun x => f x + g x) l atBot := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : OrderTopology α
    l : Filter β
    f g : β → α
    C : α
    hf : Filter.Tendsto f l Filter.atBot
    hg : Filter.Tendsto g l (nhds C)
    ⊢ Filter.Tendsto (fun x => HAdd.hAdd (f x) (g x)) l Filter.atBot
  -/
  conv in _ + _ => rw [add_comm]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : OrderTopology α
    l : Filter β
    f g : β → α
    C : α
    hf : Filter.Tendsto f l Filter.atBot
    hg : Filter.Tendsto g l (nhds C)
    ⊢ Filter.Tendsto (fun x => HAdd.hAdd (g x) (f x)) l Filter.atBot
  -/
  exact hg.add_atBot hf
  /-
    🎉 no goals
  -/


theorem nhds_basis_abs_sub_lt [NoMaxOrder α] (a : α) :
    (𝓝 a).HasBasis (fun ε : α => (0 : α) < ε) fun ε => { b | |b - a| < ε } := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrderedAddCommGroup α
    inst✝¹ : OrderTopology α
    inst✝ : NoMaxOrder α
    a : α
    ⊢ (nhds a).HasBasis (fun ε => LT.lt 0 ε) fun ε => setOf fun b => LT.lt (abs (H …
  -/
  simp only [nhds_eq_iInf_abs_sub, abs_sub_comm (a := a)]
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrderedAddCommGroup α
    inst✝¹ : OrderTopology α
    inst✝ : NoMaxOrder α
    a : α
    ⊢ (iInf fun r => iInf fun x => Filter.principal (setOf fun b => LT.lt (abs (HS …
  -/
  refine hasBasis_biInf_principal' (fun x hx y hy => ?_) (exists_gt _)
  exact ⟨min x y, lt_min hx hy, fun _ hz => hz.trans_le (min_le_left _ _),
    fun _ hz => hz.trans_le (min_le_right _ _)⟩


theorem nhds_basis_Ioo_pos [NoMaxOrder α] (a : α) :
    (𝓝 a).HasBasis (fun ε : α => (0 : α) < ε) fun ε => Ioo (a - ε) (a + ε) := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrderedAddCommGroup α
    inst✝¹ : OrderTopology α
    inst✝ : NoMaxOrder α
    a : α
    ⊢ (nhds a).HasBasis (fun ε => LT.lt 0 ε) fun ε => Set.Ioo (HSub.hSub a ε) (HAd …
  -/
  convert nhds_basis_abs_sub_lt a
  /-
    case h.e'_5.h
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrderedAddCommGroup α
    inst✝¹ : OrderTopology α
    inst✝ : NoMaxOrder α
    a x✝ : α
    ⊢ Eq (Set.Ioo (HSub.hSub a x✝) (HAdd.hAdd a x✝)) (setOf fun b => LT.lt (abs (H …
  -/
  simp only [Ioo, abs_lt, ← sub_lt_iff_lt_add, neg_lt_sub_iff_lt_add, sub_lt_comm]
  /-
    🎉 no goals
  -/


theorem nhds_basis_Icc_pos [NoMaxOrder α] [DenselyOrdered α] (a : α) :
    (𝓝 a).HasBasis ((0 : α) < ·) fun ε ↦ Icc (a - ε) (a + ε) :=
  (nhds_basis_Ioo_pos a).to_hasBasis
    (fun _ε ε₀ ↦ let ⟨δ, δ₀, δε⟩ := exists_between ε₀
      ⟨δ, δ₀, Icc_subset_Ioo (sub_lt_sub_left δε _) (add_lt_add_left δε _)⟩)
    (fun ε ε₀ ↦ ⟨ε, ε₀, Ioo_subset_Icc_self⟩)


theorem nhds_basis_zero_abs_sub_lt [NoMaxOrder α] :
    (𝓝 (0 : α)).HasBasis (fun ε : α => (0 : α) < ε) fun ε => { b | |b| < ε } := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrderedAddCommGroup α
    inst✝¹ : OrderTopology α
    inst✝ : NoMaxOrder α
    ⊢ (nhds 0).HasBasis (fun ε => LT.lt 0 ε) fun ε => setOf fun b => LT.lt (abs b) ε
  -/
  simpa using nhds_basis_abs_sub_lt (0 : α)
  /-
    🎉 no goals
  -/


/-- If `a` is positive we can form a basis from only nonnegative `Set.Ioo` intervals -/
theorem nhds_basis_Ioo_pos_of_pos [NoMaxOrder α] {a : α} (ha : 0 < a) :
    (𝓝 a).HasBasis (fun ε : α => (0 : α) < ε ∧ ε ≤ a) fun ε => Ioo (a - ε) (a + ε) :=
  (nhds_basis_Ioo_pos a).restrict fun ε hε => ⟨min a ε, lt_min ha hε, min_le_left _ _,
    Ioo_subset_Ioo (sub_le_sub_left (min_le_right _ _) _) (add_le_add_left (min_le_right _ _) _)⟩


/-- If `S` is order-connected and contains two points `x < y`,
then `S` is a right neighbourhood of `x`. -/
lemma mem_nhdsGE (hS : OrdConnected S) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) : S ∈ 𝓝[≥] x :=
  mem_of_superset (Icc_mem_nhdsGE hxy) <| hS.out hx hy


@[deprecated (since := "2024-12-22")] alias mem_nhdsWithin_Ici := mem_nhdsGE


/-- If `S` is order-connected and contains two points `x < y`,
then `S` is a punctured right neighbourhood of `x`. -/
lemma mem_nhdsGT (hS : OrdConnected S) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) : S ∈ 𝓝[>] x :=
  nhdsWithin_mono _ Ioi_subset_Ici_self <| hS.mem_nhdsGE hx hy hxy


@[deprecated (since := "2024-12-22")] alias mem_nhdsWithin_Ioi := mem_nhdsGT


/-- If `S` is order-connected and contains two points `x < y`, then `S` is a left neighbourhood
of `y`. -/
lemma mem_nhdsLE (hS : OrdConnected S) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) : S ∈ 𝓝[≤] y :=
  hS.dual.mem_nhdsGE hy hx hxy


@[deprecated (since := "2024-12-22")] alias mem_nhdsWithin_Iic := mem_nhdsLE


/-- If `S` is order-connected and contains two points `x < y`, then `S` is a punctured left
neighbourhood of `y`. -/
lemma mem_nhdsLT (hS : OrdConnected S) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) : S ∈ 𝓝[<] y :=
  hS.dual.mem_nhdsGT hy hx hxy


@[deprecated (since := "2024-12-22")] alias mem_nhdsWithin_Iio := mem_nhdsLT


