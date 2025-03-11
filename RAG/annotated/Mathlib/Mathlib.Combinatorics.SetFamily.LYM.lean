/-- The downward **local LYM inequality**, with cancelled denominators. `𝒜` takes up less of `α^(r)`
(the finsets of card `r`) than `∂𝒜` takes up of `α^(r - 1)`. -/
theorem card_mul_le_card_shadow_mul (h𝒜 : (𝒜 : Set (Finset α)).Sized r) :
    #𝒜 * r ≤ #(∂ 𝒜) * (Fintype.card α - r + 1) := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    ⊢ LE.le (HMul.hMul 𝒜.card r) (HMul.hMul 𝒜.shadow.card (HAdd.hAdd (HSub.hSub (F …
  -/
  let i : DecidableRel ((· ⊆ ·) : Finset α → Finset α → Prop) := fun _ _ => Classical.dec _
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    i : DecidableRel fun x1 x2 => HasSubset.Subset x1 x2 := fun x x_1 => Classical …
    ⊢ LE.le (HMul.hMul 𝒜.card r) (HMul.hMul 𝒜.shadow.card (HAdd.hAdd (HSub.hSub (F …
  -/
  refine card_mul_le_card_mul' (· ⊆ ·) (fun s hs => ?_) (fun s hs => ?_)
    /-
      case refine_1
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      r : Nat
      h𝒜 : Set.Sized r ↑𝒜
      i : DecidableRel fun x1 x2 => HasSubset.Subset x1 x2 := fun x x_1 => Classical …
      s : Finset α
      hs : Membership.mem 𝒜 s
      ⊢ LE.le r (Finset.bipartiteBelow (fun x1 x2 => HasSubset.Subset x1 x2) 𝒜.shado …
    -/
  · rw [← h𝒜 hs, ← card_image_of_injOn s.erase_injOn]
    /-
      case refine_1
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      r : Nat
      h𝒜 : Set.Sized r ↑𝒜
      i : DecidableRel fun x1 x2 => HasSubset.Subset x1 x2 := fun x x_1 => Classical …
      s : Finset α
      hs : Membership.mem 𝒜 s
      ⊢ LE.le (Finset.image s.erase s).card (Finset.bipartiteBelow (fun x1 x2 => Has …
    -/
    refine card_le_card ?_
    /-
      case refine_1
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      r : Nat
      h𝒜 : Set.Sized r ↑𝒜
      i : DecidableRel fun x1 x2 => HasSubset.Subset x1 x2 := fun x x_1 => Classical …
      s : Finset α
      hs : Membership.mem 𝒜 s
      ⊢ HasSubset.Subset (Finset.image s.erase s) (Finset.bipartiteBelow (fun x1 x2  …
    -/
    simp_rw [image_subset_iff, mem_bipartiteBelow]
    /-
      case refine_1
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      r : Nat
      h𝒜 : Set.Sized r ↑𝒜
      i : DecidableRel fun x1 x2 => HasSubset.Subset x1 x2 := fun x x_1 => Classical …
      s : Finset α
      hs : Membership.mem 𝒜 s
      ⊢ ∀ (x : α), Membership.mem s x → And (Membership.mem 𝒜.shadow (s.erase x)) (H …
    -/
    exact fun a ha => ⟨erase_mem_shadow hs ha, erase_subset _ _⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    i : DecidableRel fun x1 x2 => HasSubset.Subset x1 x2 := fun x x_1 => Classical …
    s : Finset α
    hs : Membership.mem 𝒜.shadow s
    ⊢ LE.le (Finset.bipartiteAbove (fun x1 x2 => HasSubset.Subset x1 x2) 𝒜 s).card …
  -/
  refine le_trans ?_ tsub_tsub_le_tsub_add
  /-
    case refine_2
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    i : DecidableRel fun x1 x2 => HasSubset.Subset x1 x2 := fun x x_1 => Classical …
    s : Finset α
    hs : Membership.mem 𝒜.shadow s
    ⊢ LE.le (Finset.bipartiteAbove (fun x1 x2 => HasSubset.Subset x1 x2) 𝒜 s).card …
  -/
  rw [← (Set.Sized.shadow h𝒜) hs, ← card_compl, ← card_image_of_injOn (insert_inj_on' _)]
  /-
    case refine_2
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    i : DecidableRel fun x1 x2 => HasSubset.Subset x1 x2 := fun x x_1 => Classical …
    s : Finset α
    hs : Membership.mem 𝒜.shadow s
    ⊢ LE.le (Finset.bipartiteAbove (fun x1 x2 => HasSubset.Subset x1 x2) 𝒜 s).card …
  -/
  refine card_le_card fun t ht => ?_
  -- Porting note: commented out the following line
  -- infer_instance
  /-
    case refine_2
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    i : DecidableRel fun x1 x2 => HasSubset.Subset x1 x2 := fun x x_1 => Classical …
    s : Finset α
    hs : Membership.mem 𝒜.shadow s
    t : Finset α
    ht : Membership.mem (Finset.bipartiteAbove (fun x1 x2 => HasSubset.Subset x1 x …
    ⊢ Membership.mem (Finset.image (fun a => Insert.insert a s) (HasCompl.compl s) …
  -/
  rw [mem_bipartiteAbove] at ht
  have : ∅ ∉ 𝒜 := by
    rw [← mem_coe, h𝒜.empty_mem_iff, coe_eq_singleton]
    rintro rfl
    rw [shadow_singleton_empty] at hs
    exact not_mem_empty s hs
  have h := exists_eq_insert_iff.2 ⟨ht.2, by
    rw [(sized_shadow_iff this).1 (Set.Sized.shadow h𝒜) ht.1, (Set.Sized.shadow h𝒜) hs]⟩
  /-
    case refine_2
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    i : DecidableRel fun x1 x2 => HasSubset.Subset x1 x2 := fun x x_1 => Classical …
    s : Finset α
    hs : Membership.mem 𝒜.shadow s
    t : Finset α
    ht : And (Membership.mem 𝒜 t) (HasSubset.Subset s t)
    this : Not (Membership.mem 𝒜 EmptyCollection.emptyCollection)
    h : Exists fun a => And (Not (Membership.mem s a)) (Eq (Insert.insert a s) t)
    ⊢ Membership.mem (Finset.image (fun a => Insert.insert a s) (HasCompl.compl s) …
  -/
  rcases h with ⟨a, ha, rfl⟩
  /-
    case refine_2.intro.intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    r : Nat
    h𝒜 : Set.Sized r ↑𝒜
    i : DecidableRel fun x1 x2 => HasSubset.Subset x1 x2 := fun x x_1 => Classical …
    s : Finset α
    hs : Membership.mem 𝒜.shadow s
    this : Not (Membership.mem 𝒜 EmptyCollection.emptyCollection)
    a : α
    ha : Not (Membership.mem s a)
    ht : And (Membership.mem 𝒜 (Insert.insert a s)) (HasSubset.Subset s (Insert.in …
    ⊢ Membership.mem (Finset.image (fun a => Insert.insert a s) (HasCompl.compl s) …
  -/
  exact mem_image_of_mem _ (mem_compl.2 ha)
  /-
    🎉 no goals
  -/


/-- The downward **local LYM inequality**. `𝒜` takes up less of `α^(r)` (the finsets of card `r`)
than `∂𝒜` takes up of `α^(r - 1)`. -/
theorem card_div_choose_le_card_shadow_div_choose (hr : r ≠ 0)
    (h𝒜 : (𝒜 : Set (Finset α)).Sized r) : (#𝒜 : 𝕜) / (Fintype.card α).choose r
    ≤ #(∂ 𝒜) / (Fintype.card α).choose (r - 1) := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    r : Nat
    hr : Ne r 0
    h𝒜 : Set.Sized r ↑𝒜
    ⊢ LE.le (HDiv.hDiv ↑𝒜.card ↑((Fintype.card α).choose r)) (HDiv.hDiv ↑𝒜.shadow. …
  -/
  obtain hr' | hr' := lt_or_le (Fintype.card α) r
    /-
      case inl
      𝕜 : Type u_1
      α : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      r : Nat
      hr : Ne r 0
      h𝒜 : Set.Sized r ↑𝒜
      hr' : LT.lt (Fintype.card α) r
      ⊢ LE.le (HDiv.hDiv ↑𝒜.card ↑((Fintype.card α).choose r)) (HDiv.hDiv ↑𝒜.shadow. …
    -/
  · rw [choose_eq_zero_of_lt hr', cast_zero, div_zero]
    /-
      case inl
      𝕜 : Type u_1
      α : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      r : Nat
      hr : Ne r 0
      h𝒜 : Set.Sized r ↑𝒜
      hr' : LT.lt (Fintype.card α) r
      ⊢ LE.le 0 (HDiv.hDiv ↑𝒜.shadow.card ↑((Fintype.card α).choose (HSub.hSub r 1)))
    -/
    exact div_nonneg (cast_nonneg _) (cast_nonneg _)
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    r : Nat
    hr : Ne r 0
    h𝒜 : Set.Sized r ↑𝒜
    hr' : LE.le r (Fintype.card α)
    ⊢ LE.le (HDiv.hDiv ↑𝒜.card ↑((Fintype.card α).choose r)) (HDiv.hDiv ↑𝒜.shadow. …
  -/
  replace h𝒜 := card_mul_le_card_shadow_mul h𝒜
  /-
    case inr
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    𝒜 : Finset (Finset α)
    r : Nat
    hr : Ne r 0
    hr' : LE.le r (Fintype.card α)
    h𝒜 : LE.le (HMul.hMul 𝒜.card r) (HMul.hMul 𝒜.shadow.card (HAdd.hAdd (HSub.hSub …
    ⊢ LE.le (HDiv.hDiv ↑𝒜.card ↑((Fintype.card α).choose r)) (HDiv.hDiv ↑𝒜.shadow. …
  -/
  rw [div_le_div_iff₀] <;> norm_cast
    /-
      case inr
      𝕜 : Type u_1
      α : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      r : Nat
      hr : Ne r 0
      hr' : LE.le r (Fintype.card α)
      h𝒜 : LE.le (HMul.hMul 𝒜.card r) (HMul.hMul 𝒜.shadow.card (HAdd.hAdd (HSub.hSub …
      ⊢ LE.le (HMul.hMul 𝒜.card ((Fintype.card α).choose (HSub.hSub r 1))) (HMul.hMu …
    -/
  · cases' r with r
      /-
        case inr.zero
        𝕜 : Type u_1
        α : Type u_2
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        𝒜 : Finset (Finset α)
        hr : Ne 0 0
        hr' : LE.le 0 (Fintype.card α)
        h𝒜 : LE.le (HMul.hMul 𝒜.card 0) (HMul.hMul 𝒜.shadow.card (HAdd.hAdd (HSub.hSub …
        ⊢ LE.le (HMul.hMul 𝒜.card ((Fintype.card α).choose (HSub.hSub 0 1))) (HMul.hMu …
      -/
    · exact (hr rfl).elim
      /-
        🎉 no goals
      -/
    /-
      case inr.succ
      𝕜 : Type u_1
      α : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      r : Nat
      hr : Ne (HAdd.hAdd r 1) 0
      hr' : LE.le (HAdd.hAdd r 1) (Fintype.card α)
      h𝒜 : LE.le (HMul.hMul 𝒜.card (HAdd.hAdd r 1)) (HMul.hMul 𝒜.shadow.card (HAdd.h …
      ⊢ LE.le (HMul.hMul 𝒜.card ((Fintype.card α).choose (HSub.hSub (HAdd.hAdd r 1)  …
    -/
    rw [tsub_add_eq_add_tsub hr', add_tsub_add_eq_tsub_right] at h𝒜
    /-
      case inr.succ
      𝕜 : Type u_1
      α : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      r : Nat
      hr : Ne (HAdd.hAdd r 1) 0
      hr' : LE.le (HAdd.hAdd r 1) (Fintype.card α)
      h𝒜 : LE.le (HMul.hMul 𝒜.card (HAdd.hAdd r 1)) (HMul.hMul 𝒜.shadow.card (HSub.h …
      ⊢ LE.le (HMul.hMul 𝒜.card ((Fintype.card α).choose (HSub.hSub (HAdd.hAdd r 1)  …
    -/
    apply le_of_mul_le_mul_right _ (pos_iff_ne_zero.2 hr)
    /-
      𝕜 : Type u_1
      α : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      r : Nat
      hr : Ne (HAdd.hAdd r 1) 0
      hr' : LE.le (HAdd.hAdd r 1) (Fintype.card α)
      h𝒜 : LE.le (HMul.hMul 𝒜.card (HAdd.hAdd r 1)) (HMul.hMul 𝒜.shadow.card (HSub.h …
      ⊢ LE.le (HMul.hMul (HMul.hMul 𝒜.card ((Fintype.card α).choose (HSub.hSub (HAdd …
    -/
    convert Nat.mul_le_mul_right ((Fintype.card α).choose r) h𝒜 using 1
      /-
        case h.e'_3
        𝕜 : Type u_1
        α : Type u_2
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        𝒜 : Finset (Finset α)
        r : Nat
        hr : Ne (HAdd.hAdd r 1) 0
        hr' : LE.le (HAdd.hAdd r 1) (Fintype.card α)
        h𝒜 : LE.le (HMul.hMul 𝒜.card (HAdd.hAdd r 1)) (HMul.hMul 𝒜.shadow.card (HSub.h …
        ⊢ Eq (HMul.hMul (HMul.hMul 𝒜.card ((Fintype.card α).choose (HSub.hSub (HAdd.hA …
      -/
    · simpa [mul_assoc, Nat.choose_succ_right_eq] using Or.inl (mul_comm _ _)
      /-
        🎉 no goals
      -/
      /-
        case h.e'_4
        𝕜 : Type u_1
        α : Type u_2
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        𝒜 : Finset (Finset α)
        r : Nat
        hr : Ne (HAdd.hAdd r 1) 0
        hr' : LE.le (HAdd.hAdd r 1) (Fintype.card α)
        h𝒜 : LE.le (HMul.hMul 𝒜.card (HAdd.hAdd r 1)) (HMul.hMul 𝒜.shadow.card (HSub.h …
        ⊢ Eq (HMul.hMul (HMul.hMul 𝒜.shadow.card ((Fintype.card α).choose (HAdd.hAdd r …
      -/
    · simp only [mul_assoc, choose_succ_right_eq, mul_eq_mul_left_iff]
      /-
        case h.e'_4
        𝕜 : Type u_1
        α : Type u_2
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        𝒜 : Finset (Finset α)
        r : Nat
        hr : Ne (HAdd.hAdd r 1) 0
        hr' : LE.le (HAdd.hAdd r 1) (Fintype.card α)
        h𝒜 : LE.le (HMul.hMul 𝒜.card (HAdd.hAdd r 1)) (HMul.hMul 𝒜.shadow.card (HSub.h …
        ⊢ Or (Eq (HMul.hMul ((Fintype.card α).choose r) (HSub.hSub (Fintype.card α) r) …
      -/
      exact Or.inl (mul_comm _ _)
      /-
        🎉 no goals
      -/
    /-
      case inr.hb
      𝕜 : Type u_1
      α : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      r : Nat
      hr : Ne r 0
      hr' : LE.le r (Fintype.card α)
      h𝒜 : LE.le (HMul.hMul 𝒜.card r) (HMul.hMul 𝒜.shadow.card (HAdd.hAdd (HSub.hSub …
      ⊢ LT.lt 0 ((Fintype.card α).choose r)
    -/
  · exact Nat.choose_pos hr'
    /-
      🎉 no goals
    -/
    /-
      case inr.hd
      𝕜 : Type u_1
      α : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      𝒜 : Finset (Finset α)
      r : Nat
      hr : Ne r 0
      hr' : LE.le r (Fintype.card α)
      h𝒜 : LE.le (HMul.hMul 𝒜.card r) (HMul.hMul 𝒜.shadow.card (HAdd.hAdd (HSub.hSub …
      ⊢ LT.lt 0 ((Fintype.card α).choose (HSub.hSub r 1))
    -/
  · exact Nat.choose_pos (r.pred_le.trans hr')
    /-
      🎉 no goals
    -/


/-- `falling k 𝒜` is all the finsets of cardinality `k` which are a subset of something in `𝒜`. -/
def falling : Finset (Finset α) :=
  𝒜.sup <| powersetCard k


theorem mem_falling : s ∈ falling k 𝒜 ↔ (∃ t ∈ 𝒜, s ⊆ t) ∧ #s = k := by
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    k : Nat
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Membership.mem (Finset.falling k 𝒜) s) (And (Exists fun t => And (Membe …
  -/
  simp_rw [falling, mem_sup, mem_powersetCard]
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    k : Nat
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Exists fun i => And (Membership.mem 𝒜 i) (And (HasSubset.Subset s i) (E …
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem sized_falling : (falling k 𝒜 : Set (Finset α)).Sized k := fun _ hs => (mem_falling.1 hs).2


theorem slice_subset_falling : 𝒜 # k ⊆ falling k 𝒜 := fun s hs =>
  mem_falling.2 <| (mem_slice.1 hs).imp_left fun h => ⟨s, h, Subset.refl _⟩


theorem falling_zero_subset : falling 0 𝒜 ⊆ {∅} :=
  subset_singleton_iff'.2 fun _ ht => card_eq_zero.1 <| sized_falling _ _ ht


theorem slice_union_shadow_falling_succ : 𝒜 # k ∪ ∂ (falling (k + 1) 𝒜) = falling k 𝒜 := by
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    k : Nat
    𝒜 : Finset (Finset α)
    ⊢ Eq (Union.union (𝒜.slice k) (Finset.falling (HAdd.hAdd k 1) 𝒜).shadow) (Fins …
  -/
  ext s
  /-
    case h
    α : Type u_2
    inst✝ : DecidableEq α
    k : Nat
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Membership.mem (Union.union (𝒜.slice k) (Finset.falling (HAdd.hAdd k 1) …
  -/
  simp_rw [mem_union, mem_slice, mem_shadow_iff, mem_falling]
  /-
    case h
    α : Type u_2
    inst✝ : DecidableEq α
    k : Nat
    𝒜 : Finset (Finset α)
    s : Finset α
    ⊢ Iff (Or (And (Membership.mem 𝒜 s) (Eq s.card k)) (Exists fun s_1 => And (And …
  -/
  constructor
    /-
      case h.mp
      α : Type u_2
      inst✝ : DecidableEq α
      k : Nat
      𝒜 : Finset (Finset α)
      s : Finset α
      ⊢ Or (And (Membership.mem 𝒜 s) (Eq s.card k)) (Exists fun s_1 => And (And (Exi …
    -/
  · rintro (h | ⟨s, ⟨⟨t, ht, hst⟩, hs⟩, a, ha, rfl⟩)
      /-
        case h.mp.inl
        α : Type u_2
        inst✝ : DecidableEq α
        k : Nat
        𝒜 : Finset (Finset α)
        s : Finset α
        h : And (Membership.mem 𝒜 s) (Eq s.card k)
        ⊢ And (Exists fun t => And (Membership.mem 𝒜 t) (HasSubset.Subset s t)) (Eq s. …
      -/
    · exact ⟨⟨s, h.1, Subset.refl _⟩, h.2⟩
      /-
        🎉 no goals
      -/
    /-
      case h.mp.inr.intro.intro.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DecidableEq α
      k : Nat
      𝒜 : Finset (Finset α)
      s : Finset α
      hs : Eq s.card (HAdd.hAdd k 1)
      t : Finset α
      ht : Membership.mem 𝒜 t
      hst : HasSubset.Subset s t
      a : α
      ha : Membership.mem s a
      ⊢ And (Exists fun t => And (Membership.mem 𝒜 t) (HasSubset.Subset (s.erase a)  …
    -/
    refine ⟨⟨t, ht, (erase_subset _ _).trans hst⟩, ?_⟩
    /-
      case h.mp.inr.intro.intro.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DecidableEq α
      k : Nat
      𝒜 : Finset (Finset α)
      s : Finset α
      hs : Eq s.card (HAdd.hAdd k 1)
      t : Finset α
      ht : Membership.mem 𝒜 t
      hst : HasSubset.Subset s t
      a : α
      ha : Membership.mem s a
      ⊢ Eq (s.erase a).card k
    -/
    rw [card_erase_of_mem ha, hs]
    /-
      case h.mp.inr.intro.intro.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DecidableEq α
      k : Nat
      𝒜 : Finset (Finset α)
      s : Finset α
      hs : Eq s.card (HAdd.hAdd k 1)
      t : Finset α
      ht : Membership.mem 𝒜 t
      hst : HasSubset.Subset s t
      a : α
      ha : Membership.mem s a
      ⊢ Eq (HSub.hSub (HAdd.hAdd k 1) 1) k
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_2
      inst✝ : DecidableEq α
      k : Nat
      𝒜 : Finset (Finset α)
      s : Finset α
      ⊢ And (Exists fun t => And (Membership.mem 𝒜 t) (HasSubset.Subset s t)) (Eq s. …
    -/
  · rintro ⟨⟨t, ht, hst⟩, hs⟩
    /-
      case h.mpr.intro.intro.intro
      α : Type u_2
      inst✝ : DecidableEq α
      k : Nat
      𝒜 : Finset (Finset α)
      s : Finset α
      hs : Eq s.card k
      t : Finset α
      ht : Membership.mem 𝒜 t
      hst : HasSubset.Subset s t
      ⊢ Or (And (Membership.mem 𝒜 s) (Eq s.card k)) (Exists fun s_1 => And (And (Exi …
    -/
    by_cases h : s ∈ 𝒜
      /-
        case pos
        α : Type u_2
        inst✝ : DecidableEq α
        k : Nat
        𝒜 : Finset (Finset α)
        s : Finset α
        hs : Eq s.card k
        t : Finset α
        ht : Membership.mem 𝒜 t
        hst : HasSubset.Subset s t
        h : Membership.mem 𝒜 s
        ⊢ Or (And (Membership.mem 𝒜 s) (Eq s.card k)) (Exists fun s_1 => And (And (Exi …
      -/
    · exact Or.inl ⟨h, hs⟩
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_2
      inst✝ : DecidableEq α
      k : Nat
      𝒜 : Finset (Finset α)
      s : Finset α
      hs : Eq s.card k
      t : Finset α
      ht : Membership.mem 𝒜 t
      hst : HasSubset.Subset s t
      h : Not (Membership.mem 𝒜 s)
      ⊢ Or (And (Membership.mem 𝒜 s) (Eq s.card k)) (Exists fun s_1 => And (And (Exi …
    -/
    obtain ⟨a, ha, hst⟩ := ssubset_iff.1 (ssubset_of_subset_of_ne hst (ht.ne_of_not_mem h).symm)
    /-
      case neg.intro.intro
      α : Type u_2
      inst✝ : DecidableEq α
      k : Nat
      𝒜 : Finset (Finset α)
      s : Finset α
      hs : Eq s.card k
      t : Finset α
      ht : Membership.mem 𝒜 t
      hst✝ : HasSubset.Subset s t
      h : Not (Membership.mem 𝒜 s)
      a : α
      ha : Not (Membership.mem s a)
      hst : HasSubset.Subset (Insert.insert a s) t
      ⊢ Or (And (Membership.mem 𝒜 s) (Eq s.card k)) (Exists fun s_1 => And (And (Exi …
    -/
    refine Or.inr ⟨insert a s, ⟨⟨t, ht, hst⟩, ?_⟩, a, mem_insert_self _ _, erase_insert ha⟩
    /-
      case neg.intro.intro
      α : Type u_2
      inst✝ : DecidableEq α
      k : Nat
      𝒜 : Finset (Finset α)
      s : Finset α
      hs : Eq s.card k
      t : Finset α
      ht : Membership.mem 𝒜 t
      hst✝ : HasSubset.Subset s t
      h : Not (Membership.mem 𝒜 s)
      a : α
      ha : Not (Membership.mem s a)
      hst : HasSubset.Subset (Insert.insert a s) t
      ⊢ Eq (Insert.insert a s).card (HAdd.hAdd k 1)
    -/
    rw [card_insert_of_not_mem ha, hs]
    /-
      🎉 no goals
    -/


/-- The shadow of `falling m 𝒜` is disjoint from the `n`-sized elements of `𝒜`, thanks to the
antichain property. -/
theorem IsAntichain.disjoint_slice_shadow_falling {m n : ℕ}
    (h𝒜 : IsAntichain (· ⊆ ·) (𝒜 : Set (Finset α))) : Disjoint (𝒜 # m) (∂ (falling n 𝒜)) :=
  disjoint_right.2 fun s h₁ h₂ => by
    /-
      α : Type u_2
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      m n : Nat
      h𝒜 : IsAntichain (fun x1 x2 => HasSubset.Subset x1 x2) ↑𝒜
      s : Finset α
      h₁ : Membership.mem (Finset.falling n 𝒜).shadow s
      h₂ : Membership.mem (𝒜.slice m) s
      ⊢ False
    -/
    simp_rw [mem_shadow_iff, mem_falling] at h₁
    /-
      α : Type u_2
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      m n : Nat
      h𝒜 : IsAntichain (fun x1 x2 => HasSubset.Subset x1 x2) ↑𝒜
      s : Finset α
      h₂ : Membership.mem (𝒜.slice m) s
      h₁ : Exists fun s_1 => And (And (Exists fun t => And (Membership.mem 𝒜 t) (Has …
      ⊢ False
    -/
    obtain ⟨s, ⟨⟨t, ht, hst⟩, _⟩, a, ha, rfl⟩ := h₁
    /-
      case intro.intro.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      m n : Nat
      h𝒜 : IsAntichain (fun x1 x2 => HasSubset.Subset x1 x2) ↑𝒜
      s : Finset α
      right✝ : Eq s.card n
      t : Finset α
      ht : Membership.mem 𝒜 t
      hst : HasSubset.Subset s t
      a : α
      ha : Membership.mem s a
      h₂ : Membership.mem (𝒜.slice m) (s.erase a)
      ⊢ False
    -/
    refine h𝒜 (slice_subset h₂) ht ?_ ((erase_subset _ _).trans hst)
    /-
      case intro.intro.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      m n : Nat
      h𝒜 : IsAntichain (fun x1 x2 => HasSubset.Subset x1 x2) ↑𝒜
      s : Finset α
      right✝ : Eq s.card n
      t : Finset α
      ht : Membership.mem 𝒜 t
      hst : HasSubset.Subset s t
      a : α
      ha : Membership.mem s a
      h₂ : Membership.mem (𝒜.slice m) (s.erase a)
      ⊢ Ne (s.erase a) t
    -/
    rintro rfl
    /-
      case intro.intro.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DecidableEq α
      𝒜 : Finset (Finset α)
      m n : Nat
      h𝒜 : IsAntichain (fun x1 x2 => HasSubset.Subset x1 x2) ↑𝒜
      s : Finset α
      right✝ : Eq s.card n
      a : α
      ha : Membership.mem s a
      h₂ : Membership.mem (𝒜.slice m) (s.erase a)
      ht : Membership.mem 𝒜 (s.erase a)
      hst : HasSubset.Subset s (s.erase a)
      ⊢ False
    -/
    exact not_mem_erase _ _ (hst ha)
    /-
      🎉 no goals
    -/


/-- A bound on any top part of the sum in LYM in terms of the size of `falling k 𝒜`. -/
theorem le_card_falling_div_choose [Fintype α] (hk : k ≤ Fintype.card α)
    (h𝒜 : IsAntichain (· ⊆ ·) (𝒜 : Set (Finset α))) :
    (∑ r ∈ range (k + 1),
        (#(𝒜 # (Fintype.card α - r)) : 𝕜) / (Fintype.card α).choose (Fintype.card α - r)) ≤
      (falling (Fintype.card α - k) 𝒜).card / (Fintype.card α).choose (Fintype.card α - k) := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    k : Nat
    𝒜 : Finset (Finset α)
    inst✝ : Fintype α
    hk : LE.le k (Fintype.card α)
    h𝒜 : IsAntichain (fun x1 x2 => HasSubset.Subset x1 x2) ↑𝒜
    ⊢ LE.le ((Finset.range (HAdd.hAdd k 1)).sum fun r => HDiv.hDiv ↑(𝒜.slice (HSub …
  -/
  induction' k with k ih
  · simp only [tsub_zero, cast_one, cast_le, sum_singleton, div_one, choose_self, range_one,
      zero_eq, zero_add, range_one, sum_singleton, nonpos_iff_eq_zero, tsub_zero,
      choose_self, cast_one, div_one, cast_le]
    /-
      case zero
      𝕜 : Type u_1
      α : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      k : Nat
      𝒜 : Finset (Finset α)
      inst✝ : Fintype α
      h𝒜 : IsAntichain (fun x1 x2 => HasSubset.Subset x1 x2) ↑𝒜
      hk : LE.le 0 (Fintype.card α)
      ⊢ LE.le (𝒜.slice (Fintype.card α)).card (Finset.falling (Fintype.card α) 𝒜).card
    -/
    exact card_le_card (slice_subset_falling _ _)
    /-
      🎉 no goals
    -/
  rw [sum_range_succ, ← slice_union_shadow_falling_succ,
    card_union_of_disjoint (IsAntichain.disjoint_slice_shadow_falling h𝒜), cast_add, _root_.add_div,
    add_comm]
  /-
    case succ
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    k✝ : Nat
    𝒜 : Finset (Finset α)
    inst✝ : Fintype α
    h𝒜 : IsAntichain (fun x1 x2 => HasSubset.Subset x1 x2) ↑𝒜
    k : Nat
    ih : LE.le k (Fintype.card α) → LE.le ((Finset.range (HAdd.hAdd k 1)).sum fun  …
    hk : LE.le (HAdd.hAdd k 1) (Fintype.card α)
    ⊢ LE.le (HAdd.hAdd (HDiv.hDiv ↑(𝒜.slice (HSub.hSub (Fintype.card α) (HAdd.hAdd …
  -/
  rw [← tsub_tsub, tsub_add_cancel_of_le (le_tsub_of_add_le_left hk)]
  exact
    add_le_add_left
      ((ih <| le_of_succ_le hk).trans <|
        card_div_choose_le_card_shadow_div_choose (tsub_pos_iff_lt.2 <| Nat.succ_le_iff.1 hk).ne' <|
          sized_falling _ _) _


/-- The **Lubell-Yamamoto-Meshalkin inequality**. If `𝒜` is an antichain, then the sum of the
proportion of elements it takes from each layer is less than `1`. -/
theorem sum_card_slice_div_choose_le_one [Fintype α]
    (h𝒜 : IsAntichain (· ⊆ ·) (𝒜 : Set (Finset α))) :
    (∑ r ∈ range (Fintype.card α + 1), (#(𝒜 # r) : 𝕜) / (Fintype.card α).choose r) ≤ 1 := by
  classical
    rw [← sum_flip]
    refine (le_card_falling_div_choose le_rfl h𝒜).trans ?_
    rw [div_le_iff₀] <;> norm_cast
    · simpa only [Nat.sub_self, one_mul, Nat.choose_zero_right, falling] using
        Set.Sized.card_le (sized_falling 0 𝒜)
    · rw [tsub_self, choose_zero_right]
      exact zero_lt_one


/-- **Sperner's theorem**. The size of an antichain in `Finset α` is bounded by the size of the
maximal layer in `Finset α`. This precisely means that `Finset α` is a Sperner order. -/
theorem IsAntichain.sperner [Fintype α] {𝒜 : Finset (Finset α)}
    (h𝒜 : IsAntichain (· ⊆ ·) (𝒜 : Set (Finset α))) :
    #𝒜 ≤ (Fintype.card α).choose (Fintype.card α / 2) := by
  classical
    suffices (∑ r ∈ Iic (Fintype.card α),
        (#(𝒜 # r) : ℚ) / (Fintype.card α).choose (Fintype.card α / 2)) ≤ 1 by
      rw [← sum_div, ← Nat.cast_sum, div_le_one] at this
      · simp only [cast_le] at this
        rwa [sum_card_slice] at this
      simp only [cast_pos]
      exact choose_pos (Nat.div_le_self _ _)
    rw [Iic_eq_Icc, ← Ico_succ_right, bot_eq_zero, Ico_zero_eq_range]
    refine (sum_le_sum fun r hr => ?_).trans (sum_card_slice_div_choose_le_one h𝒜)
    rw [mem_range] at hr
    refine div_le_div_of_nonneg_left ?_ ?_ ?_ <;> norm_cast
    · exact Nat.zero_le _
    · exact choose_pos (Nat.lt_succ_iff.1 hr)
    · exact choose_le_middle _ _


