lemma prod_indicator_biUnion_sub_indicator (hs : s.Nonempty) (S : ι → Finset α) (a : α) :
    ∏ i ∈ s, (Set.indicator (s.biUnion S) 1 a - Set.indicator (S i) 1 a) = (0 : ℤ) := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : DecidableEq α
    s : Finset ι
    hs : s.Nonempty
    S : ι → Finset α
    a : α
    ⊢ Eq (s.prod fun i => HSub.hSub ((↑(s.biUnion S)).indicator 1 a) ((↑(S i)).ind …
  -/
  by_cases ha : a ∈ s.biUnion S
    /-
      case pos
      ι : Type u_1
      α : Type u_2
      inst✝ : DecidableEq α
      s : Finset ι
      hs : s.Nonempty
      S : ι → Finset α
      a : α
      ha : Membership.mem (s.biUnion S) a
      ⊢ Eq (s.prod fun i => HSub.hSub ((↑(s.biUnion S)).indicator 1 a) ((↑(S i)).ind …
    -/
  · obtain ⟨i, hi, ha⟩ := mem_biUnion.1 ha
    /-
      case pos.intro.intro
      ι : Type u_1
      α : Type u_2
      inst✝ : DecidableEq α
      s : Finset ι
      hs : s.Nonempty
      S : ι → Finset α
      a : α
      ha✝ : Membership.mem (s.biUnion S) a
      i : ι
      hi : Membership.mem s i
      ha : Membership.mem (S i) a
      ⊢ Eq (s.prod fun i => HSub.hSub ((↑(s.biUnion S)).indicator 1 a) ((↑(S i)).ind …
    -/
    exact prod_eq_zero hi <| by simp [*, -coe_biUnion]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      α : Type u_2
      inst✝ : DecidableEq α
      s : Finset ι
      hs : s.Nonempty
      S : ι → Finset α
      a : α
      ha : Not (Membership.mem (s.biUnion S) a)
      ⊢ Eq (s.prod fun i => HSub.hSub ((↑(s.biUnion S)).indicator 1 a) ((↑(S i)).ind …
    -/
  · obtain ⟨i, hi⟩ := hs
    /-
      case neg.intro
      ι : Type u_1
      α : Type u_2
      inst✝ : DecidableEq α
      s : Finset ι
      S : ι → Finset α
      a : α
      ha : Not (Membership.mem (s.biUnion S) a)
      i : ι
      hi : Membership.mem s i
      ⊢ Eq (s.prod fun i => HSub.hSub ((↑(s.biUnion S)).indicator 1 a) ((↑(S i)).ind …
    -/
    have ha : a ∉ S i := fun h ↦ ha <| subset_biUnion_of_mem _ hi h
    /-
      case neg.intro
      ι : Type u_1
      α : Type u_2
      inst✝ : DecidableEq α
      s : Finset ι
      S : ι → Finset α
      a : α
      ha✝ : Not (Membership.mem (s.biUnion S) a)
      i : ι
      hi : Membership.mem s i
      ha : Not (Membership.mem (S i) a)
      ⊢ Eq (s.prod fun i => HSub.hSub ((↑(s.biUnion S)).indicator 1 a) ((↑(S i)).ind …
    -/
    exact prod_eq_zero hi <| by simp [*, -coe_biUnion]
    /-
      🎉 no goals
    -/


/-- **Inclusion-exclusion principle** for the sum of a function over a union.

The sum of a function `f` over the union of the `S i` over `i ∈ s` is the alternating sum of the
sums of `f` over the intersections of the `S i`. -/
theorem inclusion_exclusion_sum_biUnion (s : Finset ι) (S : ι → Finset α) (f : α → G) :
    ∑ a ∈ s.biUnion S, f a = ∑ t : s.powerset.filter (·.Nonempty),
      (-1) ^ (#t.1 + 1) • ∑ a ∈ t.1.inf' (mem_filter.1 t.2).2 S, f a := by
  classical
  rw [← sub_eq_zero]
  calc
    ∑ a ∈ s.biUnion S, f a - ∑ t : s.powerset.filter (·.Nonempty),
      (-1) ^ (#t.1 + 1) • ∑ a ∈ t.1.inf' (mem_filter.1 t.2).2 S, f a
      = ∑ t : s.powerset.filter (·.Nonempty),
          (-1) ^ #t.1 • ∑ a ∈ t.1.inf' (mem_filter.1 t.2).2 S, f a +
          ∑ t ∈ s.powerset.filter (¬ ·.Nonempty), (-1) ^ #t • ∑ a ∈ s.biUnion S, f a := by
      simp [sub_eq_neg_add, ← sum_neg_distrib, filter_eq', pow_succ]
    _ = ∑ t ∈ s.powerset, (-1) ^ #t •
          if ht : t.Nonempty then ∑ a ∈ t.inf' ht S, f a else ∑ a ∈ s.biUnion S, f a := by
      rw [← sum_attach (filter ..)]; simp [sum_dite, filter_eq', sum_attach]
    _ = ∑ a ∈ s.biUnion S, (∏ i ∈ s, (1 - Set.indicator (S i) 1 a : ℤ)) • f a := by
      simp only [Int.reduceNeg, mem_coe, prod_sub, sum_comm (s := s.biUnion S), sum_smul, mul_assoc]
      congr! with t
      split_ifs with ht
      · obtain ⟨i, hi⟩ := ht
        simp only [prod_const_one, mul_one, prod_indicator_apply]
        simp only [smul_sum, Set.indicator, Set.mem_iInter, mem_coe, Pi.one_apply, mul_ite, mul_one,
          mul_zero, ite_smul, zero_smul, sum_ite, not_forall, sum_const_zero, add_zero]
        congr
        aesop
      · obtain rfl := not_nonempty_iff_eq_empty.1 ht
        simp
    _ = ∑ a ∈ s.biUnion S, (∏ i ∈ s,
          (Set.indicator (s.biUnion S) 1 a - Set.indicator (S i) 1 a) : ℤ) • f a := by
      congr! with t; rw [Set.indicator_of_mem ‹_›, Pi.one_apply]
    _ = 0 := by
      obtain rfl | hs := s.eq_empty_or_nonempty <;>
        simp [-coe_biUnion, prod_indicator_biUnion_sub_indicator, *]


/-- **Inclusion-exclusion principle** for the cardinality of a union.

The cardinality of the union of the `S i` over `i ∈ s` is the alternating sum of the cardinalities
of the intersections of the `S i`. -/
theorem inclusion_exclusion_card_biUnion (s : Finset ι) (S : ι → Finset α) :
    #(s.biUnion S) = ∑ t : s.powerset.filter (·.Nonempty),
      (-1 : ℤ) ^ (#t.1 + 1) * #(t.1.inf' (mem_filter.1 t.2).2 S) := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : DecidableEq α
    s : Finset ι
    S : ι → Finset α
    ⊢ Eq (↑(s.biUnion S).card) (Finset.univ.sum fun t => HMul.hMul (HPow.hPow (-1) …
  -/
  simpa using inclusion_exclusion_sum_biUnion (G := ℤ) s S (f := 1)
  /-
    🎉 no goals
  -/


/-- **Inclusion-exclusion principle** for the sum of a function over an intersection of complements.

The sum of a function `f` over the intersection of the complements of the `S i` over `i ∈ s` is the
alternating sum of the sums of `f` over the intersections of the `S i`. -/
theorem inclusion_exclusion_sum_inf_compl (s : Finset ι) (S : ι → Finset α) (f : α → G) :
    ∑ a ∈ s.inf fun i ↦ (S i)ᶜ, f a = ∑ t ∈ s.powerset, (-1) ^ #t • ∑ a ∈ t.inf S, f a := by
  classical
  calc
    ∑ a ∈ s.inf fun i ↦ (S i)ᶜ, f a
      = ∑ a, f a - ∑ a ∈ s.biUnion S, f a := by
      rw [← Finset.compl_sup, sup_eq_biUnion, eq_sub_iff_add_eq, sum_compl_add_sum]
    _ = ∑ t ∈ s.powerset.filter (¬ ·.Nonempty), (-1) ^ #t • ∑ a ∈ t.inf S, f a
          + ∑ t ∈ s.powerset.filter (·.Nonempty), (-1) ^ #t • ∑ a ∈ t.inf S, f a := by
      simp [← sum_attach (filter ..), inclusion_exclusion_sum_biUnion, inf'_eq_inf, filter_eq',
        sub_eq_add_neg, pow_succ]
    _ = ∑ t ∈ s.powerset, (-1) ^ #t • ∑ a ∈ t.inf S, f a := sum_filter_not_add_sum_filter ..


/-- **Inclusion-exclusion principle** for the cardinality of an intersection of complements.

The cardinality of the intersection of the complements of the `S i` over `i ∈ s` is the
alternating sum of the cardinalities of the intersections of the `S i`. -/
theorem inclusion_exclusion_card_inf_compl (s : Finset ι) (S : ι → Finset α) :
    #(s.inf fun i ↦ (S i)ᶜ) = ∑ t ∈ s.powerset, (-1 : ℤ) ^ #t * #(t.inf S) := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    s : Finset ι
    S : ι → Finset α
    ⊢ Eq (↑(s.inf fun i => HasCompl.compl (S i)).card) (s.powerset.sum fun t => HM …
  -/
  simpa using inclusion_exclusion_sum_inf_compl (G := ℤ) s S (f := 1)
  /-
    🎉 no goals
  -/


