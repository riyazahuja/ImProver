/-- A graph is strongly regular with parameters `n k ℓ μ` if
 * its vertex set has cardinality `n`
 * it is regular with degree `k`
 * every pair of adjacent vertices has `ℓ` common neighbors
 * every pair of nonadjacent vertices has `μ` common neighbors
-/
structure IsSRGWith (n k ℓ μ : ℕ) : Prop where
  card : Fintype.card V = n
  regular : G.IsRegularOfDegree k
  of_adj : ∀ v w : V, G.Adj v w → Fintype.card (G.commonNeighbors v w) = ℓ
  of_not_adj : Pairwise fun v w => ¬G.Adj v w → Fintype.card (G.commonNeighbors v w) = μ


/-- Empty graphs are strongly regular. Note that `ℓ` can take any value
for empty graphs, since there are no pairs of adjacent vertices. -/
theorem bot_strongly_regular : (⊥ : SimpleGraph V).IsSRGWith (Fintype.card V) 0 ℓ 0 where
  card := rfl
  regular := bot_degree
  of_adj := fun _ _ h => h.elim
  of_not_adj := fun v w _h => by
    /-
      V : Type u
      inst✝ : Fintype V
      ℓ : Nat
      v w : V
      _h : Ne v w
      ⊢ (fun v w => Not (Bot.bot.Adj v w) → Eq (Fintype.card ↑(Bot.bot.commonNeighbo …
    -/
    simp only [card_eq_zero, Fintype.card_ofFinset, forall_true_left, not_false_iff, bot_adj]
    /-
      V : Type u
      inst✝ : Fintype V
      ℓ : Nat
      v w : V
      _h : Ne v w
      ⊢ Eq (Finset.filter (Membership.mem (Bot.bot.commonNeighbors v w)) Finset.univ …
    -/
    ext
    /-
      case h
      V : Type u
      inst✝ : Fintype V
      ℓ : Nat
      v w : V
      _h : Ne v w
      a✝ : V
      ⊢ Iff (Membership.mem (Finset.filter (Membership.mem (Bot.bot.commonNeighbors  …
    -/
    simp [mem_commonNeighbors]
    /-
      🎉 no goals
    -/


/-- Complete graphs are strongly regular. Note that `μ` can take any value
for complete graphs, since there are no distinct pairs of non-adjacent vertices. -/
theorem IsSRGWith.top :
    (⊤ : SimpleGraph V).IsSRGWith (Fintype.card V) (Fintype.card V - 1) (Fintype.card V - 2) μ where
  card := rfl
  regular := IsRegularOfDegree.top
  of_adj := fun v w h => by
    /-
      V : Type u
      inst✝¹ : Fintype V
      μ : Nat
      inst✝ : DecidableEq V
      v w : V
      h : Top.top.Adj v w
      ⊢ Eq (Fintype.card ↑(Top.top.commonNeighbors v w)) (HSub.hSub (Fintype.card V) …
    -/
    rw [card_commonNeighbors_top]
    /-
      V : Type u
      inst✝¹ : Fintype V
      μ : Nat
      inst✝ : DecidableEq V
      v w : V
      h : Top.top.Adj v w
      ⊢ Ne v w
    -/
    exact h
    /-
      🎉 no goals
    -/
  of_not_adj := fun v w h h' => False.elim (h' ((top_adj v w).2 h))


theorem IsSRGWith.card_neighborFinset_union_eq {v w : V} (h : G.IsSRGWith n k ℓ μ) :
    #(G.neighborFinset v ∪ G.neighborFinset w) =
      2 * k - Fintype.card (G.commonNeighbors v w) := by
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝ : DecidableEq V
    v w : V
    h : G.IsSRGWith n k ℓ μ
    ⊢ Eq (Union.union (G.neighborFinset v) (G.neighborFinset w)).card (HSub.hSub ( …
  -/
  apply Nat.add_right_cancel (m := Fintype.card (G.commonNeighbors v w))
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝ : DecidableEq V
    v w : V
    h : G.IsSRGWith n k ℓ μ
    ⊢ Eq (HAdd.hAdd (Union.union (G.neighborFinset v) (G.neighborFinset w)).card ( …
  -/
  rw [Nat.sub_add_cancel, ← Set.toFinset_card]
  -- Porting note: Set.toFinset_inter needs workaround to use unification to solve for one of the
  -- instance arguments:
  · simp [commonNeighbors, @Set.toFinset_inter _ _ _ _ _ _ (_),
      ← neighborFinset_def, Finset.card_union_add_card_inter, card_neighborFinset_eq_degree,
      h.regular.degree_eq, two_mul]
    /-
      V : Type u
      inst✝² : Fintype V
      G : SimpleGraph V
      inst✝¹ : DecidableRel G.Adj
      n k ℓ μ : Nat
      inst✝ : DecidableEq V
      v w : V
      h : G.IsSRGWith n k ℓ μ
      ⊢ LE.le (Fintype.card ↑(G.commonNeighbors v w)) (HMul.hMul 2 k)
    -/
  · apply le_trans (card_commonNeighbors_le_degree_left _ _ _)
    /-
      V : Type u
      inst✝² : Fintype V
      G : SimpleGraph V
      inst✝¹ : DecidableRel G.Adj
      n k ℓ μ : Nat
      inst✝ : DecidableEq V
      v w : V
      h : G.IsSRGWith n k ℓ μ
      ⊢ LE.le (G.degree v) (HMul.hMul 2 k)
    -/
    simp [h.regular.degree_eq, two_mul]
    /-
      🎉 no goals
    -/


/-- Assuming `G` is strongly regular, `2*(k + 1) - m` in `G` is the number of vertices that are
adjacent to either `v` or `w` when `¬G.Adj v w`. So it's the cardinality of
`G.neighborSet v ∪ G.neighborSet w`. -/
theorem IsSRGWith.card_neighborFinset_union_of_not_adj {v w : V} (h : G.IsSRGWith n k ℓ μ)
    (hne : v ≠ w) (ha : ¬G.Adj v w) :
    #(G.neighborFinset v ∪ G.neighborFinset w) = 2 * k - μ := by
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝ : DecidableEq V
    v w : V
    h : G.IsSRGWith n k ℓ μ
    hne : Ne v w
    ha : Not (G.Adj v w)
    ⊢ Eq (Union.union (G.neighborFinset v) (G.neighborFinset w)).card (HSub.hSub ( …
  -/
  rw [← h.of_not_adj hne ha]
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝ : DecidableEq V
    v w : V
    h : G.IsSRGWith n k ℓ μ
    hne : Ne v w
    ha : Not (G.Adj v w)
    ⊢ Eq (Union.union (G.neighborFinset v) (G.neighborFinset w)).card (HSub.hSub ( …
  -/
  apply h.card_neighborFinset_union_eq
  /-
    🎉 no goals
  -/


theorem IsSRGWith.card_neighborFinset_union_of_adj {v w : V} (h : G.IsSRGWith n k ℓ μ)
    (ha : G.Adj v w) : #(G.neighborFinset v ∪ G.neighborFinset w) = 2 * k - ℓ := by
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝ : DecidableEq V
    v w : V
    h : G.IsSRGWith n k ℓ μ
    ha : G.Adj v w
    ⊢ Eq (Union.union (G.neighborFinset v) (G.neighborFinset w)).card (HSub.hSub ( …
  -/
  rw [← h.of_adj v w ha]
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝ : DecidableEq V
    v w : V
    h : G.IsSRGWith n k ℓ μ
    ha : G.Adj v w
    ⊢ Eq (Union.union (G.neighborFinset v) (G.neighborFinset w)).card (HSub.hSub ( …
  -/
  apply h.card_neighborFinset_union_eq
  /-
    🎉 no goals
  -/


theorem compl_neighborFinset_sdiff_inter_eq {v w : V} :
    (G.neighborFinset v)ᶜ \ {v} ∩ ((G.neighborFinset w)ᶜ \ {w}) =
      ((G.neighborFinset v)ᶜ ∩ (G.neighborFinset w)ᶜ) \ ({w} ∪ {v}) := by
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    v w : V
    ⊢ Eq (Inter.inter (SDiff.sdiff (HasCompl.compl (G.neighborFinset v)) (Singleto …
  -/
  ext
  /-
    case h
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    v w a✝ : V
    ⊢ Iff (Membership.mem (Inter.inter (SDiff.sdiff (HasCompl.compl (G.neighborFin …
  -/
  rw [← not_iff_not]
  /-
    case h
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    v w a✝ : V
    ⊢ Iff (Not (Membership.mem (Inter.inter (SDiff.sdiff (HasCompl.compl (G.neighb …
  -/
  simp [imp_iff_not_or, or_assoc, or_comm, or_left_comm]
  /-
    🎉 no goals
  -/


theorem sdiff_compl_neighborFinset_inter_eq {v w : V} (h : G.Adj v w) :
    ((G.neighborFinset v)ᶜ ∩ (G.neighborFinset w)ᶜ) \ ({w} ∪ {v}) =
      (G.neighborFinset v)ᶜ ∩ (G.neighborFinset w)ᶜ := by
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    v w : V
    h : G.Adj v w
    ⊢ Eq (SDiff.sdiff (Inter.inter (HasCompl.compl (G.neighborFinset v)) (HasCompl …
  -/
  ext
  simp only [and_imp, mem_union, mem_sdiff, mem_compl, and_iff_left_iff_imp, mem_neighborFinset,
    mem_inter, mem_singleton]
  /-
    case h
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    v w : V
    h : G.Adj v w
    a✝ : V
    ⊢ Not (G.Adj v a✝) → Not (G.Adj w a✝) → Not (Or (Eq a✝ w) (Eq a✝ v))
  -/
  rintro hnv hnw (rfl | rfl)
    /-
      case h.inl
      V : Type u
      inst✝² : Fintype V
      G : SimpleGraph V
      inst✝¹ : DecidableRel G.Adj
      inst✝ : DecidableEq V
      v a✝ : V
      hnv : Not (G.Adj v a✝)
      h : G.Adj v a✝
      hnw : Not (G.Adj a✝ a✝)
      ⊢ False
    -/
  · exact hnv h
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      V : Type u
      inst✝² : Fintype V
      G : SimpleGraph V
      inst✝¹ : DecidableRel G.Adj
      inst✝ : DecidableEq V
      w a✝ : V
      hnw : Not (G.Adj w a✝)
      h : G.Adj a✝ w
      hnv : Not (G.Adj a✝ a✝)
      ⊢ False
    -/
  · apply hnw
    /-
      case h.inr
      V : Type u
      inst✝² : Fintype V
      G : SimpleGraph V
      inst✝¹ : DecidableRel G.Adj
      inst✝ : DecidableEq V
      w a✝ : V
      hnw : Not (G.Adj w a✝)
      h : G.Adj a✝ w
      hnv : Not (G.Adj a✝ a✝)
      ⊢ G.Adj w a✝
    -/
    rwa [adj_comm]
    /-
      🎉 no goals
    -/


theorem IsSRGWith.compl_is_regular (h : G.IsSRGWith n k ℓ μ) :
    Gᶜ.IsRegularOfDegree (n - k - 1) := by
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝ : DecidableEq V
    h : G.IsSRGWith n k ℓ μ
    ⊢ (HasCompl.compl G).IsRegularOfDegree (HSub.hSub (HSub.hSub n k) 1)
  -/
  rw [← h.card, Nat.sub_sub, add_comm, ← Nat.sub_sub]
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝ : DecidableEq V
    h : G.IsSRGWith n k ℓ μ
    ⊢ (HasCompl.compl G).IsRegularOfDegree (HSub.hSub (HSub.hSub (Fintype.card V)  …
  -/
  exact h.regular.compl
  /-
    🎉 no goals
  -/


theorem IsSRGWith.card_commonNeighbors_eq_of_adj_compl (h : G.IsSRGWith n k ℓ μ) {v w : V}
    (ha : Gᶜ.Adj v w) : Fintype.card (Gᶜ.commonNeighbors v w) = n - (2 * k - μ) - 2 := by
  simp only [← Set.toFinset_card, commonNeighbors, Set.toFinset_inter, neighborSet_compl,
    Set.toFinset_diff, Set.toFinset_singleton, Set.toFinset_compl, ← neighborFinset_def]
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝ : DecidableEq V
    h : G.IsSRGWith n k ℓ μ
    v w : V
    ha : (HasCompl.compl G).Adj v w
    ⊢ Eq (Inter.inter (SDiff.sdiff (HasCompl.compl (G.neighborFinset v)) (Singleto …
  -/
  simp_rw [compl_neighborFinset_sdiff_inter_eq]
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝ : DecidableEq V
    h : G.IsSRGWith n k ℓ μ
    v w : V
    ha : (HasCompl.compl G).Adj v w
    ⊢ Eq (SDiff.sdiff (Inter.inter (HasCompl.compl (G.neighborFinset v)) (HasCompl …
  -/
  have hne : v ≠ w := ne_of_adj _ ha
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝ : DecidableEq V
    h : G.IsSRGWith n k ℓ μ
    v w : V
    ha : (HasCompl.compl G).Adj v w
    hne : Ne v w
    ⊢ Eq (SDiff.sdiff (Inter.inter (HasCompl.compl (G.neighborFinset v)) (HasCompl …
  -/
  rw [compl_adj] at ha
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝ : DecidableEq V
    h : G.IsSRGWith n k ℓ μ
    v w : V
    ha : And (Ne v w) (Not (G.Adj v w))
    hne : Ne v w
    ⊢ Eq (SDiff.sdiff (Inter.inter (HasCompl.compl (G.neighborFinset v)) (HasCompl …
  -/
  rw [card_sdiff, ← insert_eq, card_insert_of_not_mem, card_singleton, ← Finset.compl_union]
    /-
      V : Type u
      inst✝² : Fintype V
      G : SimpleGraph V
      inst✝¹ : DecidableRel G.Adj
      n k ℓ μ : Nat
      inst✝ : DecidableEq V
      h : G.IsSRGWith n k ℓ μ
      v w : V
      ha : And (Ne v w) (Not (G.Adj v w))
      hne : Ne v w
      ⊢ Eq (HSub.hSub (HasCompl.compl (Union.union (G.neighborFinset v) (G.neighborF …
    -/
  · rw [card_compl, h.card_neighborFinset_union_of_not_adj hne ha.2, ← h.card]
    /-
      🎉 no goals
    -/
    /-
      V : Type u
      inst✝² : Fintype V
      G : SimpleGraph V
      inst✝¹ : DecidableRel G.Adj
      n k ℓ μ : Nat
      inst✝ : DecidableEq V
      h : G.IsSRGWith n k ℓ μ
      v w : V
      ha : And (Ne v w) (Not (G.Adj v w))
      hne : Ne v w
      ⊢ Not (Membership.mem (Singleton.singleton v) w)
    -/
  · simp only [hne.symm, not_false_iff, mem_singleton]
    /-
      🎉 no goals
    -/
    /-
      V : Type u
      inst✝² : Fintype V
      G : SimpleGraph V
      inst✝¹ : DecidableRel G.Adj
      n k ℓ μ : Nat
      inst✝ : DecidableEq V
      h : G.IsSRGWith n k ℓ μ
      v w : V
      ha : And (Ne v w) (Not (G.Adj v w))
      hne : Ne v w
      ⊢ HasSubset.Subset (Union.union (Singleton.singleton w) (Singleton.singleton v …
    -/
  · intro u
    /-
      V : Type u
      inst✝² : Fintype V
      G : SimpleGraph V
      inst✝¹ : DecidableRel G.Adj
      n k ℓ μ : Nat
      inst✝ : DecidableEq V
      h : G.IsSRGWith n k ℓ μ
      v w : V
      ha : And (Ne v w) (Not (G.Adj v w))
      hne : Ne v w
      u : V
      ⊢ Membership.mem (Union.union (Singleton.singleton w) (Singleton.singleton v)) …
    -/
    simp only [mem_union, mem_compl, mem_neighborFinset, mem_inter, mem_singleton]
    /-
      V : Type u
      inst✝² : Fintype V
      G : SimpleGraph V
      inst✝¹ : DecidableRel G.Adj
      n k ℓ μ : Nat
      inst✝ : DecidableEq V
      h : G.IsSRGWith n k ℓ μ
      v w : V
      ha : And (Ne v w) (Not (G.Adj v w))
      hne : Ne v w
      u : V
      ⊢ Or (Eq u w) (Eq u v) → And (Not (G.Adj v u)) (Not (G.Adj w u))
    -/
                           /-
                             🎉 no goals
                           -/
    rintro (rfl | rfl) <;> simpa [adj_comm] using ha.2
                           /-
                             🎉 no goals
                           -/


theorem IsSRGWith.card_commonNeighbors_eq_of_not_adj_compl (h : G.IsSRGWith n k ℓ μ) {v w : V}
    (hn : v ≠ w) (hna : ¬Gᶜ.Adj v w) :
    Fintype.card (Gᶜ.commonNeighbors v w) = n - (2 * k - ℓ) := by
  simp only [← Set.toFinset_card, commonNeighbors, Set.toFinset_inter, neighborSet_compl,
    Set.toFinset_diff, Set.toFinset_singleton, Set.toFinset_compl, ← neighborFinset_def]
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝ : DecidableEq V
    h : G.IsSRGWith n k ℓ μ
    v w : V
    hn : Ne v w
    hna : Not ((HasCompl.compl G).Adj v w)
    ⊢ Eq (Inter.inter (SDiff.sdiff (HasCompl.compl (G.neighborFinset v)) (Singleto …
  -/
  simp only [not_and, Classical.not_not, compl_adj] at hna
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝ : DecidableEq V
    h : G.IsSRGWith n k ℓ μ
    v w : V
    hn : Ne v w
    hna : Ne v w → G.Adj v w
    ⊢ Eq (Inter.inter (SDiff.sdiff (HasCompl.compl (G.neighborFinset v)) (Singleto …
  -/
  have h2' := hna hn
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝ : DecidableEq V
    h : G.IsSRGWith n k ℓ μ
    v w : V
    hn : Ne v w
    hna : Ne v w → G.Adj v w
    h2' : G.Adj v w
    ⊢ Eq (Inter.inter (SDiff.sdiff (HasCompl.compl (G.neighborFinset v)) (Singleto …
  -/
  simp_rw [compl_neighborFinset_sdiff_inter_eq, sdiff_compl_neighborFinset_inter_eq h2']
  /-
    V : Type u
    inst✝² : Fintype V
    G : SimpleGraph V
    inst✝¹ : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝ : DecidableEq V
    h : G.IsSRGWith n k ℓ μ
    v w : V
    hn : Ne v w
    hna : Ne v w → G.Adj v w
    h2' : G.Adj v w
    ⊢ Eq (Inter.inter (HasCompl.compl (G.neighborFinset v)) (HasCompl.compl (G.nei …
  -/
  rwa [← Finset.compl_union, card_compl, h.card_neighborFinset_union_of_adj, ← h.card]
  /-
    🎉 no goals
  -/


/-- The complement of a strongly regular graph is strongly regular. -/
theorem IsSRGWith.compl (h : G.IsSRGWith n k ℓ μ) :
    Gᶜ.IsSRGWith n (n - k - 1) (n - (2 * k - μ) - 2) (n - (2 * k - ℓ)) where
  card := h.card
  regular := h.compl_is_regular
  of_adj := fun _v _w ha => h.card_commonNeighbors_eq_of_adj_compl ha
  of_not_adj := fun _v _w hn hna => h.card_commonNeighbors_eq_of_not_adj_compl hn hna


/-- The parameters of a strongly regular graph with at least one vertex satisfy
`k * (k - ℓ - 1) = (n - k - 1) * μ`. -/
theorem IsSRGWith.param_eq
    {V : Type u} [Fintype V] (G : SimpleGraph V) [DecidableRel G.Adj]
    (h : G.IsSRGWith n k ℓ μ) (hn : 0 < n) :
    k * (k - ℓ - 1) = (n - k - 1) * μ := by
  /-
    n k ℓ μ : Nat
    V : Type u
    inst✝¹ : Fintype V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    h : G.IsSRGWith n k ℓ μ
    hn : LT.lt 0 n
    ⊢ Eq (HMul.hMul k (HSub.hSub (HSub.hSub k ℓ) 1)) (HMul.hMul (HSub.hSub (HSub.h …
  -/
  letI := Classical.decEq V
  /-
    n k ℓ μ : Nat
    V : Type u
    inst✝¹ : Fintype V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    h : G.IsSRGWith n k ℓ μ
    hn : LT.lt 0 n
    this : DecidableEq V := Classical.decEq V
    ⊢ Eq (HMul.hMul k (HSub.hSub (HSub.hSub k ℓ) 1)) (HMul.hMul (HSub.hSub (HSub.h …
  -/
  rw [← h.card, Fintype.card_pos_iff] at hn
  /-
    n k ℓ μ : Nat
    V : Type u
    inst✝¹ : Fintype V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    h : G.IsSRGWith n k ℓ μ
    hn : Nonempty V
    this : DecidableEq V := Classical.decEq V
    ⊢ Eq (HMul.hMul k (HSub.hSub (HSub.hSub k ℓ) 1)) (HMul.hMul (HSub.hSub (HSub.h …
  -/
  obtain ⟨v⟩ := hn
  /-
    case intro
    n k ℓ μ : Nat
    V : Type u
    inst✝¹ : Fintype V
    G : SimpleGraph V
    inst✝ : DecidableRel G.Adj
    h : G.IsSRGWith n k ℓ μ
    this : DecidableEq V := Classical.decEq V
    v : V
    ⊢ Eq (HMul.hMul k (HSub.hSub (HSub.hSub k ℓ) 1)) (HMul.hMul (HSub.hSub (HSub.h …
  -/
  convert card_mul_eq_card_mul G.Adj (s := G.neighborFinset v) (t := Gᶜ.neighborFinset v) _ _
    /-
      case h.e'_2.h.e'_5
      n k ℓ μ : Nat
      V : Type u
      inst✝¹ : Fintype V
      G : SimpleGraph V
      inst✝ : DecidableRel G.Adj
      h : G.IsSRGWith n k ℓ μ
      this : DecidableEq V := Classical.decEq V
      v : V
      ⊢ Eq k (G.neighborFinset v).card
    -/
  · simp [h.regular v]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_5
      n k ℓ μ : Nat
      V : Type u
      inst✝¹ : Fintype V
      G : SimpleGraph V
      inst✝ : DecidableRel G.Adj
      h : G.IsSRGWith n k ℓ μ
      this : DecidableEq V := Classical.decEq V
      v : V
      ⊢ Eq (HSub.hSub (HSub.hSub n k) 1) ((HasCompl.compl G).neighborFinset v).card
    -/
  · simp [h.compl.regular v]
    /-
      🎉 no goals
    -/
    /-
      case intro.convert_3
      n k ℓ μ : Nat
      V : Type u
      inst✝¹ : Fintype V
      G : SimpleGraph V
      inst✝ : DecidableRel G.Adj
      h : G.IsSRGWith n k ℓ μ
      this : DecidableEq V := Classical.decEq V
      v : V
      ⊢ ∀ (a : V), Membership.mem (G.neighborFinset v) a → Eq (Finset.bipartiteAbove …
    -/
  · intro w hw
    /-
      case intro.convert_3
      n k ℓ μ : Nat
      V : Type u
      inst✝¹ : Fintype V
      G : SimpleGraph V
      inst✝ : DecidableRel G.Adj
      h : G.IsSRGWith n k ℓ μ
      this : DecidableEq V := Classical.decEq V
      v w : V
      hw : Membership.mem (G.neighborFinset v) w
      ⊢ Eq (Finset.bipartiteAbove G.Adj ((HasCompl.compl G).neighborFinset v) w).car …
    -/
    rw [mem_neighborFinset] at hw
    /-
      case intro.convert_3
      n k ℓ μ : Nat
      V : Type u
      inst✝¹ : Fintype V
      G : SimpleGraph V
      inst✝ : DecidableRel G.Adj
      h : G.IsSRGWith n k ℓ μ
      this : DecidableEq V := Classical.decEq V
      v w : V
      hw : G.Adj v w
      ⊢ Eq (Finset.bipartiteAbove G.Adj ((HasCompl.compl G).neighborFinset v) w).car …
    -/
    simp_rw [bipartiteAbove, ← mem_neighborFinset, filter_mem_eq_inter]
    have s : {v} ⊆ G.neighborFinset w \ G.neighborFinset v := by
      rw [singleton_subset_iff, mem_sdiff, mem_neighborFinset]
      exact ⟨hw.symm, G.not_mem_neighborFinset_self v⟩
    rw [inter_comm, neighborFinset_compl, ← inter_sdiff_assoc, ← sdiff_eq_inter_compl, card_sdiff s,
      card_singleton, ← sdiff_inter_self_left, card_sdiff (by apply inter_subset_left)]
    /-
      case intro.convert_3
      n k ℓ μ : Nat
      V : Type u
      inst✝¹ : Fintype V
      G : SimpleGraph V
      inst✝ : DecidableRel G.Adj
      h : G.IsSRGWith n k ℓ μ
      this : DecidableEq V := Classical.decEq V
      v w : V
      hw : G.Adj v w
      s : HasSubset.Subset (Singleton.singleton v) (SDiff.sdiff (G.neighborFinset w) …
      ⊢ Eq (HSub.hSub (HSub.hSub (G.neighborFinset w).card (Inter.inter (G.neighborF …
    -/
    congr
      /-
        case intro.convert_3.e_a.e_a
        n k ℓ μ : Nat
        V : Type u
        inst✝¹ : Fintype V
        G : SimpleGraph V
        inst✝ : DecidableRel G.Adj
        h : G.IsSRGWith n k ℓ μ
        this : DecidableEq V := Classical.decEq V
        v w : V
        hw : G.Adj v w
        s : HasSubset.Subset (Singleton.singleton v) (SDiff.sdiff (G.neighborFinset w) …
        ⊢ Eq (G.neighborFinset w).card k
      -/
    · simp [h.regular w]
      /-
        🎉 no goals
      -/
    · simp_rw [inter_comm, neighborFinset_def, ← Set.toFinset_inter, ← h.of_adj v w hw,
        ← Set.toFinset_card]
      /-
        case intro.convert_3.e_a.e_a
        n k ℓ μ : Nat
        V : Type u
        inst✝¹ : Fintype V
        G : SimpleGraph V
        inst✝ : DecidableRel G.Adj
        h : G.IsSRGWith n k ℓ μ
        this : DecidableEq V := Classical.decEq V
        v w : V
        hw : G.Adj v w
        s : HasSubset.Subset (Singleton.singleton v) (SDiff.sdiff (G.neighborFinset w) …
        ⊢ Eq (Inter.inter (G.neighborSet v) (G.neighborSet w)).toFinset.card (G.common …
      -/
      congr!
      /-
        🎉 no goals
      -/
    /-
      case intro.convert_4
      n k ℓ μ : Nat
      V : Type u
      inst✝¹ : Fintype V
      G : SimpleGraph V
      inst✝ : DecidableRel G.Adj
      h : G.IsSRGWith n k ℓ μ
      this : DecidableEq V := Classical.decEq V
      v : V
      ⊢ ∀ (b : V), Membership.mem ((HasCompl.compl G).neighborFinset v) b → Eq (Fins …
    -/
  · intro w hw
    simp_rw [neighborFinset_compl, mem_sdiff, mem_compl, mem_singleton, mem_neighborFinset,
      ← Ne.eq_def] at hw
    simp_rw [bipartiteBelow, adj_comm, ← mem_neighborFinset, filter_mem_eq_inter,
      neighborFinset_def, ← Set.toFinset_inter, ← h.of_not_adj hw.2.symm hw.1,
      ← Set.toFinset_card]
    /-
      case intro.convert_4
      n k ℓ μ : Nat
      V : Type u
      inst✝¹ : Fintype V
      G : SimpleGraph V
      inst✝ : DecidableRel G.Adj
      h : G.IsSRGWith n k ℓ μ
      this : DecidableEq V := Classical.decEq V
      v w : V
      hw : And (Not (G.Adj v w)) (Ne w v)
      ⊢ Eq (Inter.inter (G.neighborSet v) (G.neighborSet w)).toFinset.card (G.common …
    -/
    congr!
    /-
      🎉 no goals
    -/


/-- Let `A` and `C` be the adjacency matrices of a strongly regular graph with parameters `n k ℓ μ`
and its complement respectively and `I` be the identity matrix,
then `A ^ 2 = k • I + ℓ • A + μ • C`. `C` is equivalent to the expression `J - I - A`
more often found in the literature, where `J` is the all-ones matrix. -/
theorem IsSRGWith.matrix_eq {α : Type*} [Semiring α] (h : G.IsSRGWith n k ℓ μ) :
    G.adjMatrix α ^ 2 = k • (1 : Matrix V V α) + ℓ • G.adjMatrix α + μ • Gᶜ.adjMatrix α := by
  /-
    V : Type u
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝¹ : DecidableEq V
    α : Type u_1
    inst✝ : Semiring α
    h : G.IsSRGWith n k ℓ μ
    ⊢ Eq (HPow.hPow (SimpleGraph.adjMatrix α G) 2) (HAdd.hAdd (HAdd.hAdd (HSMul.hS …
  -/
  ext v w
  simp only [adjMatrix_pow_apply_eq_card_walk, Set.coe_setOf, Matrix.add_apply, Matrix.smul_apply,
    adjMatrix_apply, compl_adj]
  /-
    case a
    V : Type u
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝¹ : DecidableEq V
    α : Type u_1
    inst✝ : Semiring α
    h : G.IsSRGWith n k ℓ μ
    v w : V
    ⊢ Eq (↑(Fintype.card (Subtype fun x => Eq x.length 2))) (HAdd.hAdd (HAdd.hAdd  …
  -/
  rw [Fintype.card_congr (G.walkLengthTwoEquivCommonNeighbors v w)]
  /-
    case a
    V : Type u
    inst✝³ : Fintype V
    G : SimpleGraph V
    inst✝² : DecidableRel G.Adj
    n k ℓ μ : Nat
    inst✝¹ : DecidableEq V
    α : Type u_1
    inst✝ : Semiring α
    h : G.IsSRGWith n k ℓ μ
    v w : V
    ⊢ Eq (↑(Fintype.card ↑(G.commonNeighbors v w))) (HAdd.hAdd (HAdd.hAdd (HSMul.h …
  -/
  obtain rfl | hn := eq_or_ne v w
    /-
      case a.inl
      V : Type u
      inst✝³ : Fintype V
      G : SimpleGraph V
      inst✝² : DecidableRel G.Adj
      n k ℓ μ : Nat
      inst✝¹ : DecidableEq V
      α : Type u_1
      inst✝ : Semiring α
      h : G.IsSRGWith n k ℓ μ
      v : V
      ⊢ Eq (↑(Fintype.card ↑(G.commonNeighbors v v))) (HAdd.hAdd (HAdd.hAdd (HSMul.h …
    -/
  · rw [← Set.toFinset_card]
    /-
      case a.inl
      V : Type u
      inst✝³ : Fintype V
      G : SimpleGraph V
      inst✝² : DecidableRel G.Adj
      n k ℓ μ : Nat
      inst✝¹ : DecidableEq V
      α : Type u_1
      inst✝ : Semiring α
      h : G.IsSRGWith n k ℓ μ
      v : V
      ⊢ Eq (↑(G.commonNeighbors v v).toFinset.card) (HAdd.hAdd (HAdd.hAdd (HSMul.hSM …
    -/
    simp [commonNeighbors, ← neighborFinset_def, h.regular v]
    /-
      🎉 no goals
    -/
    /-
      case a.inr
      V : Type u
      inst✝³ : Fintype V
      G : SimpleGraph V
      inst✝² : DecidableRel G.Adj
      n k ℓ μ : Nat
      inst✝¹ : DecidableEq V
      α : Type u_1
      inst✝ : Semiring α
      h : G.IsSRGWith n k ℓ μ
      v w : V
      hn : Ne v w
      ⊢ Eq (↑(Fintype.card ↑(G.commonNeighbors v w))) (HAdd.hAdd (HAdd.hAdd (HSMul.h …
    -/
  · simp only [Matrix.one_apply_ne' hn.symm, ne_eq, hn]
    /-
      case a.inr
      V : Type u
      inst✝³ : Fintype V
      G : SimpleGraph V
      inst✝² : DecidableRel G.Adj
      n k ℓ μ : Nat
      inst✝¹ : DecidableEq V
      α : Type u_1
      inst✝ : Semiring α
      h : G.IsSRGWith n k ℓ μ
      v w : V
      hn : Ne v w
      ⊢ Eq (↑(Fintype.card ↑(G.commonNeighbors v w))) (HAdd.hAdd (HAdd.hAdd (HSMul.h …
    -/
    by_cases ha : G.Adj v w <;>
      simp only [ha, ite_true, ite_false, add_zero, zero_add, nsmul_eq_mul, smul_zero, mul_one,
        not_true_eq_false, not_false_eq_true, and_false, and_self]
      /-
        case pos
        V : Type u
        inst✝³ : Fintype V
        G : SimpleGraph V
        inst✝² : DecidableRel G.Adj
        n k ℓ μ : Nat
        inst✝¹ : DecidableEq V
        α : Type u_1
        inst✝ : Semiring α
        h : G.IsSRGWith n k ℓ μ
        v w : V
        hn : Ne v w
        ha : G.Adj v w
        ⊢ Eq ↑(Fintype.card ↑(G.commonNeighbors v w)) ↑ℓ
      -/
    · rw [h.of_adj v w ha]
      /-
        🎉 no goals
      -/
      /-
        case neg
        V : Type u
        inst✝³ : Fintype V
        G : SimpleGraph V
        inst✝² : DecidableRel G.Adj
        n k ℓ μ : Nat
        inst✝¹ : DecidableEq V
        α : Type u_1
        inst✝ : Semiring α
        h : G.IsSRGWith n k ℓ μ
        v w : V
        hn : Ne v w
        ha : Not (G.Adj v w)
        ⊢ Eq ↑(Fintype.card ↑(G.commonNeighbors v w)) ↑μ
      -/
    · rw [h.of_not_adj hn ha]
      /-
        🎉 no goals
      -/


