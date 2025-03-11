/-- **Kleitman's theorem**. An intersecting family on `n` elements contains at most `2ⁿ⁻¹` sets, and
each further intersecting family takes at most half of the sets that are in no previous family. -/
theorem Finset.card_biUnion_le_of_intersecting (s : Finset ι) (f : ι → Finset (Finset α))
    (hf : ∀ i ∈ s, (f i : Set (Finset α)).Intersecting) :
    #(s.biUnion f) ≤ 2 ^ Fintype.card α - 2 ^ (Fintype.card α - #s) := by
  have : DecidableEq ι := by
    classical
    infer_instance
  /-
    ι : Type u_1
    α : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    s : Finset ι
    f : ι → Finset (Finset α)
    hf : ∀ (i : ι), Membership.mem s i → (↑(f i)).Intersecting
    this : DecidableEq ι
    ⊢ LE.le (s.biUnion f).card (HSub.hSub (HPow.hPow 2 (Fintype.card α)) (HPow.hPo …
  -/
  obtain hs | hs := le_total (Fintype.card α) #s
    /-
      case inl
      ι : Type u_1
      α : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : Nonempty α
      s : Finset ι
      f : ι → Finset (Finset α)
      hf : ∀ (i : ι), Membership.mem s i → (↑(f i)).Intersecting
      this : DecidableEq ι
      hs : LE.le (Fintype.card α) s.card
      ⊢ LE.le (s.biUnion f).card (HSub.hSub (HPow.hPow 2 (Fintype.card α)) (HPow.hPo …
    -/
  · rw [tsub_eq_zero_of_le hs, pow_zero]
    refine (card_le_card <| biUnion_subset.2 fun i hi a ha ↦
      mem_compl.2 <| not_mem_singleton.2 <| (hf _ hi).ne_bot ha).trans_eq ?_
    /-
      case inl
      ι : Type u_1
      α : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : Nonempty α
      s : Finset ι
      f : ι → Finset (Finset α)
      hf : ∀ (i : ι), Membership.mem s i → (↑(f i)).Intersecting
      this : DecidableEq ι
      hs : LE.le (Fintype.card α) s.card
      ⊢ Eq (HasCompl.compl (Singleton.singleton Bot.bot)).card (HSub.hSub (HPow.hPow …
    -/
    rw [card_compl, Fintype.card_finset, card_singleton]
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Type u_1
    α : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    s : Finset ι
    f : ι → Finset (Finset α)
    hf : ∀ (i : ι), Membership.mem s i → (↑(f i)).Intersecting
    this : DecidableEq ι
    hs : LE.le s.card (Fintype.card α)
    ⊢ LE.le (s.biUnion f).card (HSub.hSub (HPow.hPow 2 (Fintype.card α)) (HPow.hPo …
  -/
  induction' s using Finset.cons_induction with i s hi ih generalizing f
    /-
      case inr.empty
      ι : Type u_1
      α : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : Nonempty α
      this : DecidableEq ι
      f : ι → Finset (Finset α)
      hf : ∀ (i : ι), Membership.mem EmptyCollection.emptyCollection i → (↑(f i)).In …
      hs : LE.le EmptyCollection.emptyCollection.card (Fintype.card α)
      ⊢ LE.le (EmptyCollection.emptyCollection.biUnion f).card (HSub.hSub (HPow.hPow …
    -/
  · simp
    /-
      🎉 no goals
    -/
  set f' : ι → Finset (Finset α) :=
    fun j ↦ if hj : j ∈ cons i s hi then (hf j hj).exists_card_eq.choose else ∅
  have hf₁ : ∀ j, j ∈ cons i s hi → f j ⊆ f' j ∧ 2 * #(f' j) =
      2 ^ Fintype.card α ∧ (f' j : Set (Finset α)).Intersecting := by
    rintro j hj
    simp_rw [f', dif_pos hj, ← Fintype.card_finset]
    exact Classical.choose_spec (hf j hj).exists_card_eq
  have hf₂ : ∀ j, j ∈ cons i s hi → IsUpperSet (f' j : Set (Finset α)) := by
    refine fun j hj ↦ (hf₁ _ hj).2.2.isUpperSet' ((hf₁ _ hj).2.2.is_max_iff_card_eq.2 ?_)
    rw [Fintype.card_finset]
    exact (hf₁ _ hj).2.1
  /-
    case inr.cons
    ι : Type u_1
    α : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    this : DecidableEq ι
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    ih : ∀ (f : ι → Finset (Finset α)), (∀ (i : ι), Membership.mem s i → (↑(f i)). …
    f : ι → Finset (Finset α)
    hf : ∀ (i_1 : ι), Membership.mem (Finset.cons i s hi) i_1 → (↑(f i_1)).Interse …
    hs : LE.le (Finset.cons i s hi).card (Fintype.card α)
    f' : ι → Finset (Finset α) := fun j => dite (Membership.mem (Finset.cons i s h …
    hf₁ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → And (HasSubset.Subset …
    hf₂ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → IsUpperSet ↑(f' j)
    ⊢ LE.le ((Finset.cons i s hi).biUnion f).card (HSub.hSub (HPow.hPow 2 (Fintype …
  -/
  refine (card_le_card <| biUnion_mono fun j hj ↦ (hf₁ _ hj).1).trans ?_
  /-
    case inr.cons
    ι : Type u_1
    α : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    this : DecidableEq ι
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    ih : ∀ (f : ι → Finset (Finset α)), (∀ (i : ι), Membership.mem s i → (↑(f i)). …
    f : ι → Finset (Finset α)
    hf : ∀ (i_1 : ι), Membership.mem (Finset.cons i s hi) i_1 → (↑(f i_1)).Interse …
    hs : LE.le (Finset.cons i s hi).card (Fintype.card α)
    f' : ι → Finset (Finset α) := fun j => dite (Membership.mem (Finset.cons i s h …
    hf₁ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → And (HasSubset.Subset …
    hf₂ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → IsUpperSet ↑(f' j)
    ⊢ LE.le ((Finset.cons i s hi).biUnion f').card (HSub.hSub (HPow.hPow 2 (Fintyp …
  -/
  nth_rw 1 [cons_eq_insert i]
  /-
    case inr.cons
    ι : Type u_1
    α : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    this : DecidableEq ι
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    ih : ∀ (f : ι → Finset (Finset α)), (∀ (i : ι), Membership.mem s i → (↑(f i)). …
    f : ι → Finset (Finset α)
    hf : ∀ (i_1 : ι), Membership.mem (Finset.cons i s hi) i_1 → (↑(f i_1)).Interse …
    hs : LE.le (Finset.cons i s hi).card (Fintype.card α)
    f' : ι → Finset (Finset α) := fun j => dite (Membership.mem (Finset.cons i s h …
    hf₁ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → And (HasSubset.Subset …
    hf₂ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → IsUpperSet ↑(f' j)
    ⊢ LE.le ((Insert.insert i s).biUnion f').card (HSub.hSub (HPow.hPow 2 (Fintype …
  -/
  rw [biUnion_insert]
  /-
    case inr.cons
    ι : Type u_1
    α : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    this : DecidableEq ι
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    ih : ∀ (f : ι → Finset (Finset α)), (∀ (i : ι), Membership.mem s i → (↑(f i)). …
    f : ι → Finset (Finset α)
    hf : ∀ (i_1 : ι), Membership.mem (Finset.cons i s hi) i_1 → (↑(f i_1)).Interse …
    hs : LE.le (Finset.cons i s hi).card (Fintype.card α)
    f' : ι → Finset (Finset α) := fun j => dite (Membership.mem (Finset.cons i s h …
    hf₁ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → And (HasSubset.Subset …
    hf₂ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → IsUpperSet ↑(f' j)
    ⊢ LE.le (Union.union (f' i) (s.biUnion f')).card (HSub.hSub (HPow.hPow 2 (Fint …
  -/
  refine (card_mono <| @le_sup_sdiff _ _ _ <| f' i).trans ((card_union_le _ _).trans ?_)
  /-
    case inr.cons
    ι : Type u_1
    α : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    this : DecidableEq ι
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    ih : ∀ (f : ι → Finset (Finset α)), (∀ (i : ι), Membership.mem s i → (↑(f i)). …
    f : ι → Finset (Finset α)
    hf : ∀ (i_1 : ι), Membership.mem (Finset.cons i s hi) i_1 → (↑(f i_1)).Interse …
    hs : LE.le (Finset.cons i s hi).card (Fintype.card α)
    f' : ι → Finset (Finset α) := fun j => dite (Membership.mem (Finset.cons i s h …
    hf₁ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → And (HasSubset.Subset …
    hf₂ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → IsUpperSet ↑(f' j)
    ⊢ LE.le (HAdd.hAdd (f' i).card (SDiff.sdiff (Union.union (f' i) (s.biUnion f') …
  -/
  rw [union_sdiff_left, sdiff_eq_inter_compl]
  /-
    case inr.cons
    ι : Type u_1
    α : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    this : DecidableEq ι
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    ih : ∀ (f : ι → Finset (Finset α)), (∀ (i : ι), Membership.mem s i → (↑(f i)). …
    f : ι → Finset (Finset α)
    hf : ∀ (i_1 : ι), Membership.mem (Finset.cons i s hi) i_1 → (↑(f i_1)).Interse …
    hs : LE.le (Finset.cons i s hi).card (Fintype.card α)
    f' : ι → Finset (Finset α) := fun j => dite (Membership.mem (Finset.cons i s h …
    hf₁ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → And (HasSubset.Subset …
    hf₂ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → IsUpperSet ↑(f' j)
    ⊢ LE.le (HAdd.hAdd (f' i).card (Inter.inter (s.biUnion f') (HasCompl.compl (f' …
  -/
  refine le_of_mul_le_mul_left ?_ (pow_pos (zero_lt_two' ℕ) <| Fintype.card α + 1)
  /-
    case inr.cons
    ι : Type u_1
    α : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    this : DecidableEq ι
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    ih : ∀ (f : ι → Finset (Finset α)), (∀ (i : ι), Membership.mem s i → (↑(f i)). …
    f : ι → Finset (Finset α)
    hf : ∀ (i_1 : ι), Membership.mem (Finset.cons i s hi) i_1 → (↑(f i_1)).Interse …
    hs : LE.le (Finset.cons i s hi).card (Fintype.card α)
    f' : ι → Finset (Finset α) := fun j => dite (Membership.mem (Finset.cons i s h …
    hf₁ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → And (HasSubset.Subset …
    hf₂ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → IsUpperSet ↑(f' j)
    ⊢ LE.le (HMul.hMul (HPow.hPow 2 (HAdd.hAdd (Fintype.card α) 1)) (HAdd.hAdd (f' …
  -/
  rw [pow_succ, mul_add, mul_assoc, mul_comm _ 2, mul_assoc]
  refine (add_le_add
      ((mul_le_mul_left <| pow_pos (zero_lt_two' ℕ) _).2
      (hf₁ _ <| mem_cons_self _ _).2.2.card_le) <|
      (mul_le_mul_left <| zero_lt_two' ℕ).2 <| IsUpperSet.card_inter_le_finset ?_ ?_).trans ?_
    /-
      case inr.cons.refine_1
      ι : Type u_1
      α : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : Nonempty α
      this : DecidableEq ι
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      ih : ∀ (f : ι → Finset (Finset α)), (∀ (i : ι), Membership.mem s i → (↑(f i)). …
      f : ι → Finset (Finset α)
      hf : ∀ (i_1 : ι), Membership.mem (Finset.cons i s hi) i_1 → (↑(f i_1)).Interse …
      hs : LE.le (Finset.cons i s hi).card (Fintype.card α)
      f' : ι → Finset (Finset α) := fun j => dite (Membership.mem (Finset.cons i s h …
      hf₁ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → And (HasSubset.Subset …
      hf₂ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → IsUpperSet ↑(f' j)
      ⊢ IsUpperSet ↑(s.biUnion f')
    -/
  · rw [coe_biUnion]
    /-
      case inr.cons.refine_1
      ι : Type u_1
      α : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : Nonempty α
      this : DecidableEq ι
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      ih : ∀ (f : ι → Finset (Finset α)), (∀ (i : ι), Membership.mem s i → (↑(f i)). …
      f : ι → Finset (Finset α)
      hf : ∀ (i_1 : ι), Membership.mem (Finset.cons i s hi) i_1 → (↑(f i_1)).Interse …
      hs : LE.le (Finset.cons i s hi).card (Fintype.card α)
      f' : ι → Finset (Finset α) := fun j => dite (Membership.mem (Finset.cons i s h …
      hf₁ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → And (HasSubset.Subset …
      hf₂ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → IsUpperSet ↑(f' j)
      ⊢ IsUpperSet (Set.iUnion fun x => Set.iUnion fun h => ↑(f' x))
    -/
    exact isUpperSet_iUnion₂ fun i hi ↦ hf₂ _ <| subset_cons _ hi
    /-
      🎉 no goals
    -/
    /-
      case inr.cons.refine_2
      ι : Type u_1
      α : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : Nonempty α
      this : DecidableEq ι
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      ih : ∀ (f : ι → Finset (Finset α)), (∀ (i : ι), Membership.mem s i → (↑(f i)). …
      f : ι → Finset (Finset α)
      hf : ∀ (i_1 : ι), Membership.mem (Finset.cons i s hi) i_1 → (↑(f i_1)).Interse …
      hs : LE.le (Finset.cons i s hi).card (Fintype.card α)
      f' : ι → Finset (Finset α) := fun j => dite (Membership.mem (Finset.cons i s h …
      hf₁ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → And (HasSubset.Subset …
      hf₂ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → IsUpperSet ↑(f' j)
      ⊢ IsLowerSet ↑(HasCompl.compl (f' i))
    -/
  · rw [coe_compl]
    /-
      case inr.cons.refine_2
      ι : Type u_1
      α : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : Nonempty α
      this : DecidableEq ι
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      ih : ∀ (f : ι → Finset (Finset α)), (∀ (i : ι), Membership.mem s i → (↑(f i)). …
      f : ι → Finset (Finset α)
      hf : ∀ (i_1 : ι), Membership.mem (Finset.cons i s hi) i_1 → (↑(f i_1)).Interse …
      hs : LE.le (Finset.cons i s hi).card (Fintype.card α)
      f' : ι → Finset (Finset α) := fun j => dite (Membership.mem (Finset.cons i s h …
      hf₁ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → And (HasSubset.Subset …
      hf₂ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → IsUpperSet ↑(f' j)
      ⊢ IsLowerSet (HasCompl.compl ↑(f' i))
    -/
    exact (hf₂ _ <| mem_cons_self _ _).compl
    /-
      🎉 no goals
    -/
  rw [mul_tsub, card_compl, Fintype.card_finset, mul_left_comm, mul_tsub,
    (hf₁ _ <| mem_cons_self _ _).2.1, two_mul, add_tsub_cancel_left, ← mul_tsub, ← mul_two,
    mul_assoc, ← add_mul, mul_comm]
  /-
    case inr.cons.refine_3
    ι : Type u_1
    α : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nonempty α
    this : DecidableEq ι
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    ih : ∀ (f : ι → Finset (Finset α)), (∀ (i : ι), Membership.mem s i → (↑(f i)). …
    f : ι → Finset (Finset α)
    hf : ∀ (i_1 : ι), Membership.mem (Finset.cons i s hi) i_1 → (↑(f i_1)).Interse …
    hs : LE.le (Finset.cons i s hi).card (Fintype.card α)
    f' : ι → Finset (Finset α) := fun j => dite (Membership.mem (Finset.cons i s h …
    hf₁ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → And (HasSubset.Subset …
    hf₂ : ∀ (j : ι), Membership.mem (Finset.cons i s hi) j → IsUpperSet ↑(f' j)
    ⊢ LE.le (HMul.hMul (HPow.hPow 2 (Fintype.card α)) (HAdd.hAdd (HPow.hPow 2 (Fin …
  -/
  refine mul_le_mul_left' ?_ _
  refine (add_le_add_left
    (ih _ (fun i hi ↦ (hf₁ _ <| subset_cons _ hi).2.2)
    ((card_le_card <| subset_cons _).trans hs)) _).trans ?_
  rw [mul_tsub, two_mul, ← pow_succ',
    ← add_tsub_assoc_of_le (pow_right_mono₀ (one_le_two : (1 : ℕ) ≤ 2) tsub_le_self),
    tsub_add_eq_add_tsub hs, card_cons, add_tsub_add_eq_tsub_right]

