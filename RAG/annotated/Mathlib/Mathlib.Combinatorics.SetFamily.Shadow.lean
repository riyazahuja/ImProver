/-- The shadow of a set family `𝒜` is all sets we can get by removing one element from any set in
`𝒜`, and the (`k` times) iterated shadow (`shadow^[k]`) is all sets we can get by removing `k`
elements from any set in `𝒜`. -/
def shadow (𝒜 : Finset (Finset α)) : Finset (Finset α) :=
  𝒜.sup fun s => s.image (erase s)

-- Porting note: added `inherit_doc` to calm linter

@[inherit_doc] scoped[FinsetFamily] notation:max "∂ " => Finset.shadow
-- Porting note: had to open FinsetFamily

/-- The shadow of the empty set is empty. -/
@[simp]
theorem shadow_empty : ∂ (∅ : Finset (Finset α)) = ∅ :=
  rfl


@[simp] lemma shadow_iterate_empty (k : ℕ) : ∂^[k] (∅ : Finset (Finset α)) = ∅ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    k : Nat
    ⊢ Eq (Nat.iterate Finset.shadow k EmptyCollection.emptyCollection) EmptyCollec …
  -/
                   /-
                     🎉 no goals
                   -/
  induction' k <;> simp [*, shadow_empty]
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem shadow_singleton_empty : ∂ ({∅} : Finset (Finset α)) = ∅ :=
  rfl


@[simp]
theorem shadow_singleton (a : α) : ∂ {{a}} = {∅} := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    ⊢ Eq (Singleton.singleton (Singleton.singleton a)).shadow (Singleton.singleton …
  -/
  simp [shadow]
  /-
    🎉 no goals
  -/


/-- The shadow is monotone. -/
@[mono]
theorem shadow_monotone : Monotone (shadow : Finset (Finset α) → Finset (Finset α)) := fun _ _ =>
  sup_mono


@[gcongr] lemma shadow_mono (h𝒜ℬ : 𝒜 ⊆ ℬ) : ∂ 𝒜 ⊆ ∂ ℬ := shadow_monotone h𝒜ℬ


/-- `t` is in the shadow of `𝒜` iff there is a `s ∈ 𝒜` from which we can remove one element to
get `t`. -/
lemma mem_shadow_iff : t ∈ ∂ 𝒜 ↔ ∃ s ∈ 𝒜, ∃ a ∈ s, erase s a = t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    t : Finset α
    ⊢ Iff (Membership.mem 𝒜.shadow t) (Exists fun s => And (Membership.mem 𝒜 s) (E …
  -/
  simp only [shadow, mem_sup, mem_image]
  /-
    🎉 no goals
  -/


theorem erase_mem_shadow (hs : s ∈ 𝒜) (ha : a ∈ s) : erase s a ∈ ∂ 𝒜 :=
  mem_shadow_iff.2 ⟨s, hs, a, ha, rfl⟩


/-- `t ∈ ∂𝒜` iff `t` is exactly one element less than something from `𝒜`.

See also `Finset.mem_shadow_iff_exists_mem_card_add_one`. -/
lemma mem_shadow_iff_exists_sdiff : t ∈ ∂ 𝒜 ↔ ∃ s ∈ 𝒜, t ⊆ s ∧ #(s \ t) = 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    t : Finset α
    ⊢ Iff (Membership.mem 𝒜.shadow t) (Exists fun s => And (Membership.mem 𝒜 s) (A …
  -/
  simp_rw [mem_shadow_iff, ← covBy_iff_card_sdiff_eq_one, covBy_iff_exists_erase]
  /-
    🎉 no goals
  -/


/-- `t` is in the shadow of `𝒜` iff we can add an element to it so that the resulting finset is in
`𝒜`. -/
lemma mem_shadow_iff_insert_mem : t ∈ ∂ 𝒜 ↔ ∃ a ∉ t, insert a t ∈ 𝒜 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    t : Finset α
    ⊢ Iff (Membership.mem 𝒜.shadow t) (Exists fun a => And (Not (Membership.mem t  …
  -/
  simp_rw [mem_shadow_iff_exists_sdiff, ← covBy_iff_card_sdiff_eq_one, covBy_iff_exists_insert]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    t : Finset α
    ⊢ Iff (Exists fun s => And (Membership.mem 𝒜 s) (Exists fun a => And (Not (Mem …
  -/
  aesop
  /-
    🎉 no goals
  -/


/-- `s ∈ ∂ 𝒜` iff `s` is exactly one element less than something from `𝒜`.

See also `Finset.mem_shadow_iff_exists_sdiff`. -/
lemma mem_shadow_iff_exists_mem_card_add_one : t ∈ ∂ 𝒜 ↔ ∃ s ∈ 𝒜, t ⊆ s ∧ #s = #t + 1 := by
  refine mem_shadow_iff_exists_sdiff.trans <| exists_congr fun t ↦ and_congr_right fun _ ↦
    and_congr_right fun hst ↦ ?_
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    t✝ t : Finset α
    x✝ : Membership.mem 𝒜 t
    hst : HasSubset.Subset t✝ t
    ⊢ Iff (Eq (SDiff.sdiff t t✝).card 1) (Eq t.card (HAdd.hAdd t✝.card 1))
  -/
  rw [card_sdiff hst, tsub_eq_iff_eq_add_of_le, add_comm]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    t✝ t : Finset α
    x✝ : Membership.mem 𝒜 t
    hst : HasSubset.Subset t✝ t
    ⊢ LE.le t✝.card t.card
  -/
  exact card_mono hst
  /-
    🎉 no goals
  -/


lemma mem_shadow_iterate_iff_exists_card :
    t ∈ ∂^[k] 𝒜 ↔ ∃ u : Finset α, #u = k ∧ Disjoint t u ∧ t ∪ u ∈ 𝒜 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    t : Finset α
    k : Nat
    ⊢ Iff (Membership.mem (Nat.iterate Finset.shadow k 𝒜) t) (Exists fun u => And  …
  -/
  induction' k with k ih generalizing t
    /-
      case zero
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      t : Finset α
      ⊢ Iff (Membership.mem (Nat.iterate Finset.shadow 0 𝒜) t) (Exists fun u => And  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  set_option tactic.skipAssignedInstances false in
  simp only [mem_shadow_iff_insert_mem, ih, Function.iterate_succ_apply', card_eq_succ]
  aesop


/-- `t ∈ ∂^k 𝒜` iff `t` is exactly `k` elements less than something from `𝒜`.

See also `Finset.mem_shadow_iff_exists_mem_card_add`. -/
lemma mem_shadow_iterate_iff_exists_sdiff : t ∈ ∂^[k] 𝒜 ↔ ∃ s ∈ 𝒜, t ⊆ s ∧ #(s \ t) = k := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    t : Finset α
    k : Nat
    ⊢ Iff (Membership.mem (Nat.iterate Finset.shadow k 𝒜) t) (Exists fun s => And  …
  -/
  rw [mem_shadow_iterate_iff_exists_card]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    t : Finset α
    k : Nat
    ⊢ Iff (Exists fun u => And (Eq u.card k) (And (Disjoint t u) (Membership.mem 𝒜 …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      t : Finset α
      k : Nat
      ⊢ (Exists fun u => And (Eq u.card k) (And (Disjoint t u) (Membership.mem 𝒜 (Un …
    -/
  · rintro ⟨u, rfl, htu, hsuA⟩
    /-
      case mp.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      t u : Finset α
      htu : Disjoint t u
      hsuA : Membership.mem 𝒜 (Union.union t u)
      ⊢ Exists fun s => And (Membership.mem 𝒜 s) (And (HasSubset.Subset t s) (Eq (SD …
    -/
    exact ⟨_, hsuA, subset_union_left, by rw [union_sdiff_cancel_left htu]⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      t : Finset α
      k : Nat
      ⊢ (Exists fun s => And (Membership.mem 𝒜 s) (And (HasSubset.Subset t s) (Eq (S …
    -/
  · rintro ⟨s, hs, hts, rfl⟩
    /-
      case mpr.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      t s : Finset α
      hs : Membership.mem 𝒜 s
      hts : HasSubset.Subset t s
      ⊢ Exists fun u => And (Eq u.card (SDiff.sdiff s t).card) (And (Disjoint t u) ( …
    -/
    refine ⟨s \ t, rfl, disjoint_sdiff, ?_⟩
    /-
      case mpr.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      t s : Finset α
      hs : Membership.mem 𝒜 s
      hts : HasSubset.Subset t s
      ⊢ Membership.mem 𝒜 (Union.union t (SDiff.sdiff s t))
    -/
    rwa [union_sdiff_self_eq_union, union_eq_right.2 hts]
    /-
      🎉 no goals
    -/


/-- `t ∈ ∂^k 𝒜` iff `t` is exactly `k` elements less than something in `𝒜`.

See also `Finset.mem_shadow_iterate_iff_exists_sdiff`. -/
lemma mem_shadow_iterate_iff_exists_mem_card_add :
    t ∈ ∂^[k] 𝒜 ↔ ∃ s ∈ 𝒜, t ⊆ s ∧ #s = #t + k := by
  refine mem_shadow_iterate_iff_exists_sdiff.trans <| exists_congr fun t ↦ and_congr_right fun _ ↦
    and_congr_right fun hst ↦ ?_
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    t✝ : Finset α
    k : Nat
    t : Finset α
    x✝ : Membership.mem 𝒜 t
    hst : HasSubset.Subset t✝ t
    ⊢ Iff (Eq (SDiff.sdiff t t✝).card k) (Eq t.card (HAdd.hAdd t✝.card k))
  -/
  rw [card_sdiff hst, tsub_eq_iff_eq_add_of_le, add_comm]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    t✝ : Finset α
    k : Nat
    t : Finset α
    x✝ : Membership.mem 𝒜 t
    hst : HasSubset.Subset t✝ t
    ⊢ LE.le t✝.card t.card
  -/
  exact card_mono hst
  /-
    🎉 no goals
  -/


/-- The shadow of a family of `r`-sets is a family of `r - 1`-sets. -/
protected theorem _root_.Set.Sized.shadow (h𝒜 : (𝒜 : Set (Finset α)).Sized r) :
    (∂ 𝒜 : Set (Finset α)).Sized (r - 1) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    ⊢ Set.Sized (HSub.hSub r 1) ↑𝒜.shadow
  -/
  intro A h
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    A : Finset α
    h : Membership.mem (↑𝒜.shadow) A
    ⊢ Eq A.card (HSub.hSub r 1)
  -/
  obtain ⟨A, hA, i, hi, rfl⟩ := mem_shadow_iff.1 h
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    A : Finset α
    hA : Membership.mem 𝒜 A
    i : α
    hi : Membership.mem A i
    h : Membership.mem (↑𝒜.shadow) (A.erase i)
    ⊢ Eq (A.erase i).card (HSub.hSub r 1)
  -/
  rw [card_erase_of_mem hi, h𝒜 hA]
  /-
    🎉 no goals
  -/


/-- The `k`-th shadow of a family of `r`-sets is a family of `r - k`-sets. -/
lemma _root_.Set.Sized.shadow_iterate (h𝒜 : (𝒜 : Set (Finset α)).Sized r) :
    (∂^[k] 𝒜 : Set (Finset α)).Sized (r - k) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    k r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    ⊢ Set.Sized (HSub.hSub r k) ↑(Nat.iterate Finset.shadow k 𝒜)
  -/
  simp_rw [Set.Sized, mem_coe, mem_shadow_iterate_iff_exists_sdiff]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    k r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    ⊢ ∀ ⦃x : Finset α⦄, (Exists fun s => And (Membership.mem 𝒜 s) (And (HasSubset. …
  -/
  rintro t ⟨s, hs, hts, rfl⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    t s : Finset α
    hs : Membership.mem 𝒜 s
    hts : HasSubset.Subset t s
    ⊢ Eq t.card (HSub.hSub r (SDiff.sdiff s t).card)
  -/
  rw [card_sdiff hts, ← h𝒜 hs, Nat.sub_sub_self (card_le_card hts)]
  /-
    🎉 no goals
  -/


theorem sized_shadow_iff (h : ∅ ∉ 𝒜) :
    (∂ 𝒜 : Set (Finset α)).Sized r ↔ (𝒜 : Set (Finset α)).Sized (r + 1) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    r : Nat
    h : Not (Membership.mem 𝒜 EmptyCollection.emptyCollection)
    ⊢ Iff (Set.Sized r ↑𝒜.shadow) (Set.Sized (HAdd.hAdd r 1) ↑𝒜)
  -/
  refine ⟨fun h𝒜 s hs => ?_, Set.Sized.shadow⟩
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    r : Nat
    h : Not (Membership.mem 𝒜 EmptyCollection.emptyCollection)
    h𝒜 : Set.Sized r ↑𝒜.shadow
    s : Finset α
    hs : Membership.mem (↑𝒜) s
    ⊢ Eq s.card (HAdd.hAdd r 1)
  -/
  obtain ⟨a, ha⟩ := nonempty_iff_ne_empty.2 (ne_of_mem_of_not_mem hs h)
  /-
    case intro
    α : Type u_1
    inst✝ : DecidableEq α
    𝒜 : Finset (Finset α)
    r : Nat
    h : Not (Membership.mem 𝒜 EmptyCollection.emptyCollection)
    h𝒜 : Set.Sized r ↑𝒜.shadow
    s : Finset α
    hs : Membership.mem (↑𝒜) s
    a : α
    ha : Membership.mem s a
    ⊢ Eq s.card (HAdd.hAdd r 1)
  -/
  rw [← h𝒜 (erase_mem_shadow hs ha), card_erase_add_one ha]
  /-
    🎉 no goals
  -/


/-- Being in the shadow of `𝒜` means we have a superset in `𝒜`. -/
lemma exists_subset_of_mem_shadow (hs : t ∈ ∂ 𝒜) : ∃ s ∈ 𝒜, t ⊆ s :=
  let ⟨t, ht, hst⟩ := mem_shadow_iff_exists_mem_card_add_one.1 hs
  ⟨t, ht, hst.1⟩


/-- The upper shadow of a set family `𝒜` is all sets we can get by adding one element to any set in
`𝒜`, and the (`k` times) iterated upper shadow (`upShadow^[k]`) is all sets we can get by adding
`k` elements from any set in `𝒜`. -/
def upShadow (𝒜 : Finset (Finset α)) : Finset (Finset α) :=
  𝒜.sup fun s => sᶜ.image fun a => insert a s

-- Porting note: added `inherit_doc` to calm linter

@[inherit_doc] scoped[FinsetFamily] notation:max "∂⁺ " => Finset.upShadow


/-- The upper shadow of the empty set is empty. -/
@[simp]
theorem upShadow_empty : ∂⁺ (∅ : Finset (Finset α)) = ∅ :=
  rfl


/-- The upper shadow is monotone. -/
@[mono]
theorem upShadow_monotone : Monotone (upShadow : Finset (Finset α) → Finset (Finset α)) :=
  fun _ _ => sup_mono


/-- `t` is in the upper shadow of `𝒜` iff there is a `s ∈ 𝒜` from which we can remove one element
to get `t`. -/
lemma mem_upShadow_iff : t ∈ ∂⁺ 𝒜 ↔ ∃ s ∈ 𝒜, ∃ a ∉ s, insert a s = t := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    t : Finset α
    ⊢ Iff (Membership.mem 𝒜.upShadow t) (Exists fun s => And (Membership.mem 𝒜 s)  …
  -/
  simp_rw [upShadow, mem_sup, mem_image, mem_compl]
  /-
    🎉 no goals
  -/


theorem insert_mem_upShadow (hs : s ∈ 𝒜) (ha : a ∉ s) : insert a s ∈ ∂⁺ 𝒜 :=
  mem_upShadow_iff.2 ⟨s, hs, a, ha, rfl⟩


/-- `t` is in the upper shadow of `𝒜` iff `t` is exactly one element more than something from `𝒜`.

See also `Finset.mem_upShadow_iff_exists_mem_card_add_one`. -/
lemma mem_upShadow_iff_exists_sdiff : t ∈ ∂⁺ 𝒜 ↔ ∃ s ∈ 𝒜, s ⊆ t ∧ #(t \ s) = 1 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    t : Finset α
    ⊢ Iff (Membership.mem 𝒜.upShadow t) (Exists fun s => And (Membership.mem 𝒜 s)  …
  -/
  simp_rw [mem_upShadow_iff, ← covBy_iff_card_sdiff_eq_one, covBy_iff_exists_insert]
  /-
    🎉 no goals
  -/


/-- `t` is in the upper shadow of `𝒜` iff we can remove an element from it so that the resulting
finset is in `𝒜`. -/
lemma mem_upShadow_iff_erase_mem : t ∈ ∂⁺ 𝒜 ↔ ∃ a, a ∈ t ∧ erase t a ∈ 𝒜 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    t : Finset α
    ⊢ Iff (Membership.mem 𝒜.upShadow t) (Exists fun a => And (Membership.mem t a)  …
  -/
  simp_rw [mem_upShadow_iff_exists_sdiff, ← covBy_iff_card_sdiff_eq_one, covBy_iff_exists_erase]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    t : Finset α
    ⊢ Iff (Exists fun s => And (Membership.mem 𝒜 s) (Exists fun a => And (Membersh …
  -/
  aesop
  /-
    🎉 no goals
  -/


/-- `t` is in the upper shadow of `𝒜` iff `t` is exactly one element less than something from `𝒜`.

See also `Finset.mem_upShadow_iff_exists_sdiff`. -/
lemma mem_upShadow_iff_exists_mem_card_add_one :
    t ∈ ∂⁺ 𝒜 ↔ ∃ s ∈ 𝒜, s ⊆ t ∧ #t = #s + 1 := by
  refine mem_upShadow_iff_exists_sdiff.trans <| exists_congr fun t ↦ and_congr_right fun _ ↦
    and_congr_right fun hst ↦ ?_
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    t✝ t : Finset α
    x✝ : Membership.mem 𝒜 t
    hst : HasSubset.Subset t t✝
    ⊢ Iff (Eq (SDiff.sdiff t✝ t).card 1) (Eq t✝.card (HAdd.hAdd t.card 1))
  -/
  rw [card_sdiff hst, tsub_eq_iff_eq_add_of_le, add_comm]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    t✝ t : Finset α
    x✝ : Membership.mem 𝒜 t
    hst : HasSubset.Subset t t✝
    ⊢ LE.le t.card t✝.card
  -/
  exact card_mono hst
  /-
    🎉 no goals
  -/


lemma mem_upShadow_iterate_iff_exists_card :
    t ∈ ∂⁺^[k] 𝒜 ↔ ∃ u : Finset α, #u = k ∧ u ⊆ t ∧ t \ u ∈ 𝒜 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    t : Finset α
    k : Nat
    ⊢ Iff (Membership.mem (Nat.iterate Finset.upShadow k 𝒜) t) (Exists fun u => An …
  -/
  induction' k with k ih generalizing t
    /-
      case zero
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      t : Finset α
      ⊢ Iff (Membership.mem (Nat.iterate Finset.upShadow 0 𝒜) t) (Exists fun u => An …
    -/
  · simp
    /-
      🎉 no goals
    -/
  simp only [mem_upShadow_iff_erase_mem, ih, Function.iterate_succ_apply', card_eq_succ,
    subset_erase, erase_sdiff_comm, ← sdiff_insert]
  /-
    case succ
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    k : Nat
    ih : ∀ {t : Finset α}, Iff (Membership.mem (Nat.iterate Finset.upShadow k 𝒜) t …
    t : Finset α
    ⊢ Iff (Exists fun a => And (Membership.mem t a) (Exists fun u => And (Eq u.car …
  -/
  constructor
    /-
      case succ.mp
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      k : Nat
      ih : ∀ {t : Finset α}, Iff (Membership.mem (Nat.iterate Finset.upShadow k 𝒜) t …
      t : Finset α
      ⊢ (Exists fun a => And (Membership.mem t a) (Exists fun u => And (Eq u.card k) …
    -/
  · rintro ⟨a, hat, u, rfl, ⟨hut, hau⟩, htu⟩
    /-
      case succ.mp.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      t : Finset α
      a : α
      hat : Membership.mem t a
      u : Finset α
      ih : ∀ {t : Finset α}, Iff (Membership.mem (Nat.iterate Finset.upShadow u.card …
      htu : Membership.mem 𝒜 (SDiff.sdiff t (Insert.insert a u))
      hut : HasSubset.Subset u t
      hau : Not (Membership.mem u a)
      ⊢ Exists fun u_1 => And (Exists fun a => Exists fun t => And (Not (Membership. …
    -/
    exact ⟨_, ⟨_, _, hau, rfl, rfl⟩, insert_subset hat hut, htu⟩
    /-
      🎉 no goals
    -/
    /-
      case succ.mpr
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      k : Nat
      ih : ∀ {t : Finset α}, Iff (Membership.mem (Nat.iterate Finset.upShadow k 𝒜) t …
      t : Finset α
      ⊢ (Exists fun u => And (Exists fun a => Exists fun t => And (Not (Membership.m …
    -/
  · rintro ⟨_, ⟨a, u, hau, rfl, rfl⟩, hut, htu⟩
    /-
      case succ.mpr.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      t : Finset α
      a : α
      u : Finset α
      hau : Not (Membership.mem u a)
      ih : ∀ {t : Finset α}, Iff (Membership.mem (Nat.iterate Finset.upShadow u.card …
      hut : HasSubset.Subset (Insert.insert a u) t
      htu : Membership.mem 𝒜 (SDiff.sdiff t (Insert.insert a u))
      ⊢ Exists fun a => And (Membership.mem t a) (Exists fun u_1 => And (Eq u_1.card …
    -/
    rw [insert_subset_iff] at hut
    /-
      case succ.mpr.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      t : Finset α
      a : α
      u : Finset α
      hau : Not (Membership.mem u a)
      ih : ∀ {t : Finset α}, Iff (Membership.mem (Nat.iterate Finset.upShadow u.card …
      hut : And (Membership.mem t a) (HasSubset.Subset u t)
      htu : Membership.mem 𝒜 (SDiff.sdiff t (Insert.insert a u))
      ⊢ Exists fun a => And (Membership.mem t a) (Exists fun u_1 => And (Eq u_1.card …
    -/
    exact ⟨a, hut.1, _, rfl, ⟨hut.2, hau⟩, htu⟩
    /-
      🎉 no goals
    -/


/-- `t` is in the upper shadow of `𝒜` iff `t` is exactly `k` elements less than something from `𝒜`.

See also `Finset.mem_upShadow_iff_exists_mem_card_add`. -/
lemma mem_upShadow_iterate_iff_exists_sdiff : t ∈ ∂⁺^[k] 𝒜 ↔ ∃ s ∈ 𝒜, s ⊆ t ∧ #(t \ s) = k := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    t : Finset α
    k : Nat
    ⊢ Iff (Membership.mem (Nat.iterate Finset.upShadow k 𝒜) t) (Exists fun s => An …
  -/
  rw [mem_upShadow_iterate_iff_exists_card]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    t : Finset α
    k : Nat
    ⊢ Iff (Exists fun u => And (Eq u.card k) (And (HasSubset.Subset u t) (Membersh …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      t : Finset α
      k : Nat
      ⊢ (Exists fun u => And (Eq u.card k) (And (HasSubset.Subset u t) (Membership.m …
    -/
  · rintro ⟨u, rfl, hut, htu⟩
    /-
      case mp.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      t u : Finset α
      hut : HasSubset.Subset u t
      htu : Membership.mem 𝒜 (SDiff.sdiff t u)
      ⊢ Exists fun s => And (Membership.mem 𝒜 s) (And (HasSubset.Subset s t) (Eq (SD …
    -/
    exact ⟨_, htu, sdiff_subset, by rw [sdiff_sdiff_eq_self hut]⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      t : Finset α
      k : Nat
      ⊢ (Exists fun s => And (Membership.mem 𝒜 s) (And (HasSubset.Subset s t) (Eq (S …
    -/
  · rintro ⟨s, hs, hst, rfl⟩
    /-
      case mpr.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      t s : Finset α
      hs : Membership.mem 𝒜 s
      hst : HasSubset.Subset s t
      ⊢ Exists fun u => And (Eq u.card (SDiff.sdiff t s).card) (And (HasSubset.Subse …
    -/
    exact ⟨_, rfl, sdiff_subset, by rwa [sdiff_sdiff_eq_self hst]⟩
    /-
      🎉 no goals
    -/


/-- `t ∈ ∂⁺^k 𝒜` iff `t` is exactly `k` elements less than something in `𝒜`.

See also `Finset.mem_upShadow_iterate_iff_exists_sdiff`. -/
lemma mem_upShadow_iterate_iff_exists_mem_card_add :
    t ∈ ∂⁺^[k] 𝒜 ↔ ∃ s ∈ 𝒜, s ⊆ t ∧ #t = #s + k := by
  refine mem_upShadow_iterate_iff_exists_sdiff.trans <| exists_congr fun t ↦ and_congr_right fun _ ↦
    and_congr_right fun hst ↦ ?_
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    t✝ : Finset α
    k : Nat
    t : Finset α
    x✝ : Membership.mem 𝒜 t
    hst : HasSubset.Subset t t✝
    ⊢ Iff (Eq (SDiff.sdiff t✝ t).card k) (Eq t✝.card (HAdd.hAdd t.card k))
  -/
  rw [card_sdiff hst, tsub_eq_iff_eq_add_of_le, add_comm]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    t✝ : Finset α
    k : Nat
    t : Finset α
    x✝ : Membership.mem 𝒜 t
    hst : HasSubset.Subset t t✝
    ⊢ LE.le t.card t✝.card
  -/
  exact card_mono hst
  /-
    🎉 no goals
  -/


/-- The upper shadow of a family of `r`-sets is a family of `r + 1`-sets. -/
protected lemma _root_.Set.Sized.upShadow (h𝒜 : (𝒜 : Set (Finset α)).Sized r) :
    (∂⁺ 𝒜 : Set (Finset α)).Sized (r + 1) := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    ⊢ Set.Sized (HAdd.hAdd r 1) ↑𝒜.upShadow
  -/
  intro A h
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    A : Finset α
    h : Membership.mem (↑𝒜.upShadow) A
    ⊢ Eq A.card (HAdd.hAdd r 1)
  -/
  obtain ⟨A, hA, i, hi, rfl⟩ := mem_upShadow_iff.1 h
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    A : Finset α
    hA : Membership.mem 𝒜 A
    i : α
    hi : Not (Membership.mem A i)
    h : Membership.mem (↑𝒜.upShadow) (Insert.insert i A)
    ⊢ Eq (Insert.insert i A).card (HAdd.hAdd r 1)
  -/
  rw [card_insert_of_not_mem hi, h𝒜 hA]
  /-
    🎉 no goals
  -/


/-- Being in the upper shadow of `𝒜` means we have a superset in `𝒜`. -/
theorem exists_subset_of_mem_upShadow (hs : s ∈ ∂⁺ 𝒜) : ∃ t ∈ 𝒜, t ⊆ s :=
  let ⟨t, ht, hts, _⟩ := mem_upShadow_iff_exists_mem_card_add_one.1 hs
  ⟨t, ht, hts⟩


/-- `t ∈ ∂^k 𝒜` iff `t` is exactly `k` elements more than something in `𝒜`. -/
theorem mem_upShadow_iff_exists_mem_card_add :
    s ∈ ∂⁺ ^[k] 𝒜 ↔ ∃ t ∈ 𝒜, t ⊆ s ∧ #t + k = #s := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    s : Finset α
    k : Nat
    ⊢ Iff (Membership.mem (Nat.iterate Finset.upShadow k 𝒜) s) (Exists fun t => An …
  -/
  induction' k with k ih generalizing 𝒜 s
    /-
      case zero
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      s : Finset α
      ⊢ Iff (Membership.mem (Nat.iterate Finset.upShadow 0 𝒜) s) (Exists fun t => An …
    -/
  · refine ⟨fun hs => ⟨s, hs, Subset.refl _, rfl⟩, ?_⟩
    /-
      case zero
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      s : Finset α
      ⊢ (Exists fun t => And (Membership.mem 𝒜 t) (And (HasSubset.Subset t s) (Eq (H …
    -/
    rintro ⟨t, ht, hst, hcard⟩
    /-
      case zero.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      s t : Finset α
      ht : Membership.mem 𝒜 t
      hst : HasSubset.Subset t s
      hcard : Eq (HAdd.hAdd t.card 0) s.card
      ⊢ Membership.mem (Nat.iterate Finset.upShadow 0 𝒜) s
    -/
    rwa [← eq_of_subset_of_card_le hst hcard.ge]
    /-
      🎉 no goals
    -/
  /-
    case succ
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    k : Nat
    ih : ∀ {𝒜 : Finset (Finset α)} {s : Finset α}, Iff (Membership.mem (Nat.iterat …
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Membership.mem (Nat.iterate Finset.upShadow (HAdd.hAdd k 1) 𝒜) s) (Exis …
  -/
  simp only [exists_prop, Function.comp_apply, Function.iterate_succ]
  /-
    case succ
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    k : Nat
    ih : ∀ {𝒜 : Finset (Finset α)} {s : Finset α}, Iff (Membership.mem (Nat.iterat …
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Membership.mem (Nat.iterate Finset.upShadow k 𝒜.upShadow) s) (Exists fu …
  -/
  refine ih.trans ?_
  /-
    case succ
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    k : Nat
    ih : ∀ {𝒜 : Finset (Finset α)} {s : Finset α}, Iff (Membership.mem (Nat.iterat …
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Exists fun t => And (Membership.mem 𝒜.upShadow t) (And (HasSubset.Subse …
  -/
  clear ih
  /-
    case succ
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    k : Nat
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Exists fun t => And (Membership.mem 𝒜.upShadow t) (And (HasSubset.Subse …
  -/
  constructor
    /-
      case succ.mp
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      k : Nat
      𝒜 : Finset (Finset α)
      s : Finset α
      ⊢ (Exists fun t => And (Membership.mem 𝒜.upShadow t) (And (HasSubset.Subset t  …
    -/
  · rintro ⟨t, ht, hts, hcardst⟩
    /-
      case succ.mp.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      k : Nat
      𝒜 : Finset (Finset α)
      s t : Finset α
      ht : Membership.mem 𝒜.upShadow t
      hts : HasSubset.Subset t s
      hcardst : Eq (HAdd.hAdd t.card k) s.card
      ⊢ Exists fun t => And (Membership.mem 𝒜 t) (And (HasSubset.Subset t s) (Eq (HA …
    -/
    obtain ⟨u, hu, hut, hcardtu⟩ := mem_upShadow_iff_exists_mem_card_add_one.1 ht
    /-
      case succ.mp.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      k : Nat
      𝒜 : Finset (Finset α)
      s t : Finset α
      ht : Membership.mem 𝒜.upShadow t
      hts : HasSubset.Subset t s
      hcardst : Eq (HAdd.hAdd t.card k) s.card
      u : Finset α
      hu : Membership.mem 𝒜 u
      hut : HasSubset.Subset u t
      hcardtu : Eq t.card (HAdd.hAdd u.card 1)
      ⊢ Exists fun t => And (Membership.mem 𝒜 t) (And (HasSubset.Subset t s) (Eq (HA …
    -/
    refine ⟨u, hu, hut.trans hts, ?_⟩
    /-
      case succ.mp.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      k : Nat
      𝒜 : Finset (Finset α)
      s t : Finset α
      ht : Membership.mem 𝒜.upShadow t
      hts : HasSubset.Subset t s
      hcardst : Eq (HAdd.hAdd t.card k) s.card
      u : Finset α
      hu : Membership.mem 𝒜 u
      hut : HasSubset.Subset u t
      hcardtu : Eq t.card (HAdd.hAdd u.card 1)
      ⊢ Eq (HAdd.hAdd u.card (HAdd.hAdd k 1)) s.card
    -/
    rw [← hcardst, hcardtu, add_right_comm]
    /-
      case succ.mp.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      k : Nat
      𝒜 : Finset (Finset α)
      s t : Finset α
      ht : Membership.mem 𝒜.upShadow t
      hts : HasSubset.Subset t s
      hcardst : Eq (HAdd.hAdd t.card k) s.card
      u : Finset α
      hu : Membership.mem 𝒜 u
      hut : HasSubset.Subset u t
      hcardtu : Eq t.card (HAdd.hAdd u.card 1)
      ⊢ Eq (HAdd.hAdd u.card (HAdd.hAdd k 1)) (HAdd.hAdd (HAdd.hAdd u.card k) 1)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case succ.mpr
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      k : Nat
      𝒜 : Finset (Finset α)
      s : Finset α
      ⊢ (Exists fun t => And (Membership.mem 𝒜 t) (And (HasSubset.Subset t s) (Eq (H …
    -/
  · rintro ⟨t, ht, hts, hcard⟩
    obtain ⟨u, htu, hus, hu⟩ := Finset.exists_subsuperset_card_eq hts (Nat.le_add_right _ 1)
      (by omega)
    /-
      case succ.mpr.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      k : Nat
      𝒜 : Finset (Finset α)
      s t : Finset α
      ht : Membership.mem 𝒜 t
      hts : HasSubset.Subset t s
      hcard : Eq (HAdd.hAdd t.card (HAdd.hAdd k 1)) s.card
      u : Finset α
      htu : HasSubset.Subset t u
      hus : HasSubset.Subset u s
      hu : Eq u.card (HAdd.hAdd t.card 1)
      ⊢ Exists fun t => And (Membership.mem 𝒜.upShadow t) (And (HasSubset.Subset t s …
    -/
    refine ⟨u, mem_upShadow_iff_exists_mem_card_add_one.2 ⟨t, ht, htu, hu⟩, hus, ?_⟩
    /-
      case succ.mpr.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      k : Nat
      𝒜 : Finset (Finset α)
      s t : Finset α
      ht : Membership.mem 𝒜 t
      hts : HasSubset.Subset t s
      hcard : Eq (HAdd.hAdd t.card (HAdd.hAdd k 1)) s.card
      u : Finset α
      htu : HasSubset.Subset t u
      hus : HasSubset.Subset u s
      hu : Eq u.card (HAdd.hAdd t.card 1)
      ⊢ Eq (HAdd.hAdd u.card k) s.card
    -/
    rw [hu, ← hcard, add_right_comm]
    /-
      case succ.mpr.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      k : Nat
      𝒜 : Finset (Finset α)
      s t : Finset α
      ht : Membership.mem 𝒜 t
      hts : HasSubset.Subset t s
      hcard : Eq (HAdd.hAdd t.card (HAdd.hAdd k 1)) s.card
      u : Finset α
      htu : HasSubset.Subset t u
      hus : HasSubset.Subset u s
      hu : Eq u.card (HAdd.hAdd t.card 1)
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd t.card k) 1) (HAdd.hAdd t.card (HAdd.hAdd k 1))
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp] lemma shadow_compls : ∂ 𝒜ᶜˢ = (∂⁺ 𝒜)ᶜˢ := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    ⊢ Eq 𝒜.compls.shadow 𝒜.upShadow.compls
  -/
  ext s
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Membership.mem 𝒜.compls.shadow s) (Membership.mem 𝒜.upShadow.compls s)
  -/
  simp only [mem_image, exists_prop, mem_shadow_iff, mem_upShadow_iff, mem_compls]
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Exists fun s_1 => And (Membership.mem 𝒜 (HasCompl.compl s_1)) (Exists f …
  -/
  refine (compl_involutive.toPerm _).exists_congr_left.trans ?_
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Exists fun b => And (Membership.mem 𝒜 (HasCompl.compl ((Equiv.symm (Fun …
  -/
  simp [← compl_involutive.eq_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma upShadow_compls : ∂⁺ 𝒜ᶜˢ = (∂ 𝒜)ᶜˢ := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    ⊢ Eq 𝒜.compls.upShadow 𝒜.shadow.compls
  -/
  ext s
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Membership.mem 𝒜.compls.upShadow s) (Membership.mem 𝒜.shadow.compls s)
  -/
  simp only [mem_image, exists_prop, mem_shadow_iff, mem_upShadow_iff, mem_compls]
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Exists fun s_1 => And (Membership.mem 𝒜 (HasCompl.compl s_1)) (Exists f …
  -/
  refine (compl_involutive.toPerm _).exists_congr_left.trans ?_
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Exists fun b => And (Membership.mem 𝒜 (HasCompl.compl ((Equiv.symm (Fun …
  -/
  simp [← compl_involutive.eq_iff]
  /-
    🎉 no goals
  -/


