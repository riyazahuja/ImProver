/-- A pair of finsets of vertices is `ε`-uniform (aka `ε`-regular) iff their edge density is close
to the density of any big enough pair of subsets. Intuitively, the edges between them are
random-like. -/
def IsUniform (s t : Finset α) : Prop :=
  ∀ ⦃s'⦄, s' ⊆ s → ∀ ⦃t'⦄, t' ⊆ t → (#s : 𝕜) * ε ≤ #s' →
    (#t : 𝕜) * ε ≤ #t' → |(G.edgeDensity s' t' : 𝕜) - (G.edgeDensity s t : 𝕜)| < ε


instance IsUniform.instDecidableRel : DecidableRel (G.IsUniform ε) := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    a b : α
    ⊢ DecidableRel (G.IsUniform ε)
  -/
  unfold IsUniform; infer_instance
                    /-
                      🎉 no goals
                    -/


theorem IsUniform.mono {ε' : 𝕜} (h : ε ≤ ε') (hε : IsUniform G ε s t) : IsUniform G ε' s t :=
  fun s' hs' t' ht' hs ht => by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    ε' : 𝕜
    h : LE.le ε ε'
    hε : G.IsUniform ε s t
    s' : Finset α
    hs' : HasSubset.Subset s' s
    t' : Finset α
    ht' : HasSubset.Subset t' t
    hs : LE.le (HMul.hMul (↑s.card) ε') ↑s'.card
    ht : LE.le (HMul.hMul (↑t.card) ε') ↑t'.card
    ⊢ LT.lt (abs (HSub.hSub ↑(G.edgeDensity s' t') ↑(G.edgeDensity s t))) ε'
  -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  refine (hε hs' ht' (le_trans ?_ hs) (le_trans ?_ ht)).trans_le h <;> gcongr
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem IsUniform.symm : Symmetric (IsUniform G ε) := fun s t h t' ht' s' hs' ht hs => by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h : G.IsUniform ε s t
    t' : Finset α
    ht' : HasSubset.Subset t' t
    s' : Finset α
    hs' : HasSubset.Subset s' s
    ht : LE.le (HMul.hMul (↑t.card) ε) ↑t'.card
    hs : LE.le (HMul.hMul (↑s.card) ε) ↑s'.card
    ⊢ LT.lt (abs (HSub.hSub ↑(G.edgeDensity t' s') ↑(G.edgeDensity t s))) ε
  -/
  rw [edgeDensity_comm _ t', edgeDensity_comm _ t]
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h : G.IsUniform ε s t
    t' : Finset α
    ht' : HasSubset.Subset t' t
    s' : Finset α
    hs' : HasSubset.Subset s' s
    ht : LE.le (HMul.hMul (↑t.card) ε) ↑t'.card
    hs : LE.le (HMul.hMul (↑s.card) ε) ↑s'.card
    ⊢ LT.lt (abs (HSub.hSub ↑(G.edgeDensity s' t') ↑(G.edgeDensity s t))) ε
  -/
  exact h hs' ht' hs ht
  /-
    🎉 no goals
  -/


theorem isUniform_comm : IsUniform G ε s t ↔ IsUniform G ε t s :=
  ⟨fun h => h.symm, fun h => h.symm⟩


lemma isUniform_one : G.IsUniform (1 : 𝕜) s t := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    s t : Finset α
    ⊢ G.IsUniform 1 s t
  -/
  intro s' hs' t' ht' hs ht
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    s t s' : Finset α
    hs' : HasSubset.Subset s' s
    t' : Finset α
    ht' : HasSubset.Subset t' t
    hs : LE.le (HMul.hMul (↑s.card) 1) ↑s'.card
    ht : LE.le (HMul.hMul (↑t.card) 1) ↑t'.card
    ⊢ LT.lt (abs (HSub.hSub ↑(G.edgeDensity s' t') ↑(G.edgeDensity s t))) 1
  -/
  rw [mul_one] at hs ht
  rw [eq_of_subset_of_card_le hs' (Nat.cast_le.1 hs),
    eq_of_subset_of_card_le ht' (Nat.cast_le.1 ht), sub_self, abs_zero]
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    s t s' : Finset α
    hs' : HasSubset.Subset s' s
    t' : Finset α
    ht' : HasSubset.Subset t' t
    hs : LE.le ↑s.card ↑s'.card
    ht : LE.le ↑t.card ↑t'.card
    ⊢ LT.lt 0 1
  -/
  exact zero_lt_one
  /-
    🎉 no goals
  -/


lemma IsUniform.pos (hG : G.IsUniform ε s t) : 0 < ε :=
  not_le.1 fun hε ↦ (hε.trans <| abs_nonneg _).not_lt <| hG (empty_subset _) (empty_subset _)
        /-
          α : Type u_1
          𝕜 : Type u_2
          inst✝¹ : LinearOrderedField 𝕜
          G : SimpleGraph α
          inst✝ : DecidableRel G.Adj
          ε : 𝕜
          s t : Finset α
          hG : G.IsUniform ε s t
          hε : LE.le ε 0
          ⊢ LE.le (HMul.hMul (↑s.card) ε) ↑EmptyCollection.emptyCollection.card
        -/
    (by simpa using mul_nonpos_of_nonneg_of_nonpos (Nat.cast_nonneg _) hε)
        /-
          🎉 no goals
        -/
        /-
          α : Type u_1
          𝕜 : Type u_2
          inst✝¹ : LinearOrderedField 𝕜
          G : SimpleGraph α
          inst✝ : DecidableRel G.Adj
          ε : 𝕜
          s t : Finset α
          hG : G.IsUniform ε s t
          hε : LE.le ε 0
          ⊢ LE.le (HMul.hMul (↑t.card) ε) ↑EmptyCollection.emptyCollection.card
        -/
    (by simpa using mul_nonpos_of_nonneg_of_nonpos (Nat.cast_nonneg _) hε)
        /-
          🎉 no goals
        -/


@[simp] lemma isUniform_singleton : G.IsUniform ε {a} {b} ↔ 0 < ε := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    a b : α
    ⊢ Iff (G.IsUniform ε (Singleton.singleton a) (Singleton.singleton b)) (LT.lt 0 …
  -/
  refine ⟨IsUniform.pos, fun hε s' hs' t' ht' hs ht ↦ ?_⟩
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    a b : α
    hε : LT.lt 0 ε
    s' : Finset α
    hs' : HasSubset.Subset s' (Singleton.singleton a)
    t' : Finset α
    ht' : HasSubset.Subset t' (Singleton.singleton b)
    hs : LE.le (HMul.hMul (↑(Singleton.singleton a).card) ε) ↑s'.card
    ht : LE.le (HMul.hMul (↑(Singleton.singleton b).card) ε) ↑t'.card
    ⊢ LT.lt (abs (HSub.hSub ↑(G.edgeDensity s' t') ↑(G.edgeDensity (Singleton.sing …
  -/
  rw [card_singleton, Nat.cast_one, one_mul] at hs ht
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    a b : α
    hε : LT.lt 0 ε
    s' : Finset α
    hs' : HasSubset.Subset s' (Singleton.singleton a)
    t' : Finset α
    ht' : HasSubset.Subset t' (Singleton.singleton b)
    hs : LE.le ε ↑s'.card
    ht : LE.le ε ↑t'.card
    ⊢ LT.lt (abs (HSub.hSub ↑(G.edgeDensity s' t') ↑(G.edgeDensity (Singleton.sing …
  -/
  obtain rfl | rfl := Finset.subset_singleton_iff.1 hs'
    /-
      case inl
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      a b : α
      hε : LT.lt 0 ε
      t' : Finset α
      ht' : HasSubset.Subset t' (Singleton.singleton b)
      ht : LE.le ε ↑t'.card
      hs' : HasSubset.Subset EmptyCollection.emptyCollection (Singleton.singleton a)
      hs : LE.le ε ↑EmptyCollection.emptyCollection.card
      ⊢ LT.lt (abs (HSub.hSub ↑(G.edgeDensity EmptyCollection.emptyCollection t') ↑( …
    -/
  · replace hs : ε ≤ 0 := by simpa using hs
    /-
      case inl
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      a b : α
      hε : LT.lt 0 ε
      t' : Finset α
      ht' : HasSubset.Subset t' (Singleton.singleton b)
      ht : LE.le ε ↑t'.card
      hs' : HasSubset.Subset EmptyCollection.emptyCollection (Singleton.singleton a)
      hs : LE.le ε 0
      ⊢ LT.lt (abs (HSub.hSub ↑(G.edgeDensity EmptyCollection.emptyCollection t') ↑( …
    -/
    exact (hε.not_le hs).elim
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    a b : α
    hε : LT.lt 0 ε
    t' : Finset α
    ht' : HasSubset.Subset t' (Singleton.singleton b)
    ht : LE.le ε ↑t'.card
    hs' : HasSubset.Subset (Singleton.singleton a) (Singleton.singleton a)
    hs : LE.le ε ↑(Singleton.singleton a).card
    ⊢ LT.lt (abs (HSub.hSub ↑(G.edgeDensity (Singleton.singleton a) t') ↑(G.edgeDe …
  -/
  obtain rfl | rfl := Finset.subset_singleton_iff.1 ht'
    /-
      case inr.inl
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      a b : α
      hε : LT.lt 0 ε
      hs' : HasSubset.Subset (Singleton.singleton a) (Singleton.singleton a)
      hs : LE.le ε ↑(Singleton.singleton a).card
      ht' : HasSubset.Subset EmptyCollection.emptyCollection (Singleton.singleton b)
      ht : LE.le ε ↑EmptyCollection.emptyCollection.card
      ⊢ LT.lt (abs (HSub.hSub ↑(G.edgeDensity (Singleton.singleton a) EmptyCollectio …
    -/
  · replace ht : ε ≤ 0 := by simpa using ht
    /-
      case inr.inl
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      a b : α
      hε : LT.lt 0 ε
      hs' : HasSubset.Subset (Singleton.singleton a) (Singleton.singleton a)
      hs : LE.le ε ↑(Singleton.singleton a).card
      ht' : HasSubset.Subset EmptyCollection.emptyCollection (Singleton.singleton b)
      ht : LE.le ε 0
      ⊢ LT.lt (abs (HSub.hSub ↑(G.edgeDensity (Singleton.singleton a) EmptyCollectio …
    -/
    exact (hε.not_le ht).elim
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      a b : α
      hε : LT.lt 0 ε
      hs' : HasSubset.Subset (Singleton.singleton a) (Singleton.singleton a)
      hs : LE.le ε ↑(Singleton.singleton a).card
      ht' : HasSubset.Subset (Singleton.singleton b) (Singleton.singleton b)
      ht : LE.le ε ↑(Singleton.singleton b).card
      ⊢ LT.lt (abs (HSub.hSub ↑(G.edgeDensity (Singleton.singleton a) (Singleton.sin …
    -/
  · rwa [sub_self, abs_zero]
    /-
      🎉 no goals
    -/


theorem not_isUniform_zero : ¬G.IsUniform (0 : 𝕜) s t := fun h =>
                                                                   /-
                                                                     α : Type u_1
                                                                     𝕜 : Type u_2
                                                                     inst✝¹ : LinearOrderedField 𝕜
                                                                     G : SimpleGraph α
                                                                     inst✝ : DecidableRel G.Adj
                                                                     s t : Finset α
                                                                     h : G.IsUniform 0 s t
                                                                     ⊢ LE.le (HMul.hMul (↑s.card) 0) ↑EmptyCollection.emptyCollection.card
                                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  (abs_nonneg _).not_lt <| h (empty_subset _) (empty_subset _) (by simp) (by simp)
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem not_isUniform_iff :
    ¬G.IsUniform ε s t ↔ ∃ s', s' ⊆ s ∧ ∃ t', t' ⊆ t ∧ #s * ε ≤ #s' ∧
      #t * ε ≤ #t' ∧ ε ≤ |G.edgeDensity s' t' - G.edgeDensity s t| := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    ⊢ Iff (Not (G.IsUniform ε s t)) (Exists fun s' => And (HasSubset.Subset s' s)  …
  -/
  unfold IsUniform
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    ⊢ Iff (Not (∀ ⦃s' : Finset α⦄, HasSubset.Subset s' s → ∀ ⦃t' : Finset α⦄, HasS …
  -/
  simp only [not_forall, not_lt, exists_prop, exists_and_left, Rat.cast_abs, Rat.cast_sub]
  /-
    🎉 no goals
  -/


/-- An arbitrary pair of subsets witnessing the non-uniformity of `(s, t)`. If `(s, t)` is uniform,
returns `(s, t)`. Witnesses for `(s, t)` and `(t, s)` don't necessarily match. See
`SimpleGraph.nonuniformWitness`. -/
noncomputable def nonuniformWitnesses (ε : 𝕜) (s t : Finset α) : Finset α × Finset α :=
  if h : ¬G.IsUniform ε s t then
    ((not_isUniform_iff.1 h).choose, (not_isUniform_iff.1 h).choose_spec.2.choose)
  else (s, t)


theorem left_nonuniformWitnesses_subset (h : ¬G.IsUniform ε s t) :
    (G.nonuniformWitnesses ε s t).1 ⊆ s := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h : Not (G.IsUniform ε s t)
    ⊢ HasSubset.Subset (G.nonuniformWitnesses ε s t).1 s
  -/
  rw [nonuniformWitnesses, dif_pos h]
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h : Not (G.IsUniform ε s t)
    ⊢ HasSubset.Subset { fst := ⋯.choose, snd := ⋯.choose }.1 s
  -/
  exact (not_isUniform_iff.1 h).choose_spec.1
  /-
    🎉 no goals
  -/


theorem left_nonuniformWitnesses_card (h : ¬G.IsUniform ε s t) :
    #s * ε ≤ #(G.nonuniformWitnesses ε s t).1 := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h : Not (G.IsUniform ε s t)
    ⊢ LE.le (HMul.hMul (↑s.card) ε) ↑(G.nonuniformWitnesses ε s t).1.card
  -/
  rw [nonuniformWitnesses, dif_pos h]
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h : Not (G.IsUniform ε s t)
    ⊢ LE.le (HMul.hMul (↑s.card) ε) ↑{ fst := ⋯.choose, snd := ⋯.choose }.1.card
  -/
  exact (not_isUniform_iff.1 h).choose_spec.2.choose_spec.2.1
  /-
    🎉 no goals
  -/


theorem right_nonuniformWitnesses_subset (h : ¬G.IsUniform ε s t) :
    (G.nonuniformWitnesses ε s t).2 ⊆ t := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h : Not (G.IsUniform ε s t)
    ⊢ HasSubset.Subset (G.nonuniformWitnesses ε s t).2 t
  -/
  rw [nonuniformWitnesses, dif_pos h]
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h : Not (G.IsUniform ε s t)
    ⊢ HasSubset.Subset { fst := ⋯.choose, snd := ⋯.choose }.2 t
  -/
  exact (not_isUniform_iff.1 h).choose_spec.2.choose_spec.1
  /-
    🎉 no goals
  -/


theorem right_nonuniformWitnesses_card (h : ¬G.IsUniform ε s t) :
    #t * ε ≤ #(G.nonuniformWitnesses ε s t).2 := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h : Not (G.IsUniform ε s t)
    ⊢ LE.le (HMul.hMul (↑t.card) ε) ↑(G.nonuniformWitnesses ε s t).2.card
  -/
  rw [nonuniformWitnesses, dif_pos h]
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h : Not (G.IsUniform ε s t)
    ⊢ LE.le (HMul.hMul (↑t.card) ε) ↑{ fst := ⋯.choose, snd := ⋯.choose }.2.card
  -/
  exact (not_isUniform_iff.1 h).choose_spec.2.choose_spec.2.2.1
  /-
    🎉 no goals
  -/


theorem nonuniformWitnesses_spec (h : ¬G.IsUniform ε s t) :
    ε ≤
      |G.edgeDensity (G.nonuniformWitnesses ε s t).1 (G.nonuniformWitnesses ε s t).2 -
          G.edgeDensity s t| := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h : Not (G.IsUniform ε s t)
    ⊢ LE.le ε ↑(abs (HSub.hSub (G.edgeDensity (G.nonuniformWitnesses ε s t).1 (G.n …
  -/
  rw [nonuniformWitnesses, dif_pos h]
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h : Not (G.IsUniform ε s t)
    ⊢ LE.le ε ↑(abs (HSub.hSub (G.edgeDensity { fst := ⋯.choose, snd := ⋯.choose } …
  -/
  exact (not_isUniform_iff.1 h).choose_spec.2.choose_spec.2.2.2
  /-
    🎉 no goals
  -/


open scoped Classical in
/-- Arbitrary witness of non-uniformity. `G.nonuniformWitness ε s t` and
`G.nonuniformWitness ε t s` form a pair of subsets witnessing the non-uniformity of `(s, t)`. If
`(s, t)` is uniform, returns `s`. -/
noncomputable def nonuniformWitness (ε : 𝕜) (s t : Finset α) : Finset α :=
  if WellOrderingRel s t then (G.nonuniformWitnesses ε s t).1 else (G.nonuniformWitnesses ε t s).2


theorem nonuniformWitness_subset (h : ¬G.IsUniform ε s t) : G.nonuniformWitness ε s t ⊆ s := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h : Not (G.IsUniform ε s t)
    ⊢ HasSubset.Subset (G.nonuniformWitness ε s t) s
  -/
  unfold nonuniformWitness
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h : Not (G.IsUniform ε s t)
    ⊢ HasSubset.Subset (ite (WellOrderingRel s t) (G.nonuniformWitnesses ε s t).1  …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      s t : Finset α
      h : Not (G.IsUniform ε s t)
      h✝ : WellOrderingRel s t
      ⊢ HasSubset.Subset (G.nonuniformWitnesses ε s t).1 s
    -/
  · exact G.left_nonuniformWitnesses_subset h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      s t : Finset α
      h : Not (G.IsUniform ε s t)
      h✝ : Not (WellOrderingRel s t)
      ⊢ HasSubset.Subset (G.nonuniformWitnesses ε t s).2 s
    -/
  · exact G.right_nonuniformWitnesses_subset fun i => h i.symm
    /-
      🎉 no goals
    -/


theorem le_card_nonuniformWitness (h : ¬G.IsUniform ε s t) :
    #s * ε ≤ #(G.nonuniformWitness ε s t) := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h : Not (G.IsUniform ε s t)
    ⊢ LE.le (HMul.hMul (↑s.card) ε) ↑(G.nonuniformWitness ε s t).card
  -/
  unfold nonuniformWitness
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h : Not (G.IsUniform ε s t)
    ⊢ LE.le (HMul.hMul (↑s.card) ε) ↑(ite (WellOrderingRel s t) (G.nonuniformWitne …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      s t : Finset α
      h : Not (G.IsUniform ε s t)
      h✝ : WellOrderingRel s t
      ⊢ LE.le (HMul.hMul (↑s.card) ε) ↑(G.nonuniformWitnesses ε s t).1.card
    -/
  · exact G.left_nonuniformWitnesses_card h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      s t : Finset α
      h : Not (G.IsUniform ε s t)
      h✝ : Not (WellOrderingRel s t)
      ⊢ LE.le (HMul.hMul (↑s.card) ε) ↑(G.nonuniformWitnesses ε t s).2.card
    -/
  · exact G.right_nonuniformWitnesses_card fun i => h i.symm
    /-
      🎉 no goals
    -/


theorem nonuniformWitness_spec (h₁ : s ≠ t) (h₂ : ¬G.IsUniform ε s t) : ε ≤ |G.edgeDensity
    (G.nonuniformWitness ε s t) (G.nonuniformWitness ε t s) - G.edgeDensity s t| := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h₁ : Ne s t
    h₂ : Not (G.IsUniform ε s t)
    ⊢ LE.le ε ↑(abs (HSub.hSub (G.edgeDensity (G.nonuniformWitness ε s t) (G.nonun …
  -/
  unfold nonuniformWitness
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    s t : Finset α
    h₁ : Ne s t
    h₂ : Not (G.IsUniform ε s t)
    ⊢ LE.le ε ↑(abs (HSub.hSub (G.edgeDensity (ite (WellOrderingRel s t) (G.nonuni …
  -/
  rcases trichotomous_of WellOrderingRel s t with (lt | rfl | gt)
    /-
      case inl
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      s t : Finset α
      h₁ : Ne s t
      h₂ : Not (G.IsUniform ε s t)
      lt : WellOrderingRel s t
      ⊢ LE.le ε ↑(abs (HSub.hSub (G.edgeDensity (ite (WellOrderingRel s t) (G.nonuni …
    -/
  · rw [if_pos lt, if_neg (asymm lt)]
    /-
      case inl
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      s t : Finset α
      h₁ : Ne s t
      h₂ : Not (G.IsUniform ε s t)
      lt : WellOrderingRel s t
      ⊢ LE.le ε ↑(abs (HSub.hSub (G.edgeDensity (G.nonuniformWitnesses ε s t).1 (G.n …
    -/
    exact G.nonuniformWitnesses_spec h₂
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      s : Finset α
      h₁ : Ne s s
      h₂ : Not (G.IsUniform ε s s)
      ⊢ LE.le ε ↑(abs (HSub.hSub (G.edgeDensity (ite (WellOrderingRel s s) (G.nonuni …
    -/
  · cases h₁ rfl
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      s t : Finset α
      h₁ : Ne s t
      h₂ : Not (G.IsUniform ε s t)
      gt : WellOrderingRel t s
      ⊢ LE.le ε ↑(abs (HSub.hSub (G.edgeDensity (ite (WellOrderingRel s t) (G.nonuni …
    -/
  · rw [if_neg (asymm gt), if_pos gt, edgeDensity_comm, edgeDensity_comm _ s]
    /-
      case inr.inr
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      s t : Finset α
      h₁ : Ne s t
      h₂ : Not (G.IsUniform ε s t)
      gt : WellOrderingRel t s
      ⊢ LE.le ε ↑(abs (HSub.hSub (G.edgeDensity (G.nonuniformWitnesses ε t s).1 (G.n …
    -/
    apply G.nonuniformWitnesses_spec fun i => h₂ i.symm
    /-
      🎉 no goals
    -/


/-- The pairs of parts of a partition `P` which are not `ε`-dense in a graph `G`. Note that we
dismiss the diagonal. We do not care whether `s` is `ε`-dense with itself. -/
def sparsePairs (ε : 𝕜) : Finset (Finset α × Finset α) :=
  P.parts.offDiag.filter fun (u, v) ↦ G.edgeDensity u v < ε


@[simp]
lemma mk_mem_sparsePairs (u v : Finset α) (ε : 𝕜) :
    (u, v) ∈ P.sparsePairs G ε ↔ u ∈ P.parts ∧ v ∈ P.parts ∧ u ≠ v ∧ G.edgeDensity u v < ε := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    P : Finpartition A
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    u v : Finset α
    ε : 𝕜
    ⊢ Iff (Membership.mem (P.sparsePairs G ε) { fst := u, snd := v }) (And (Member …
  -/
  rw [sparsePairs, mem_filter, mem_offDiag, and_assoc, and_assoc]
  /-
    🎉 no goals
  -/


lemma sparsePairs_mono {ε ε' : 𝕜} (h : ε ≤ ε') : P.sparsePairs G ε ⊆ P.sparsePairs G ε' :=
  monotone_filter_right _ fun _ ↦ h.trans_lt'


/-- The pairs of parts of a partition `P` which are not `ε`-uniform in a graph `G`. Note that we
dismiss the diagonal. We do not care whether `s` is `ε`-uniform with itself. -/
def nonUniforms (ε : 𝕜) : Finset (Finset α × Finset α) :=
  P.parts.offDiag.filter fun (u, v) ↦ ¬G.IsUniform ε u v


@[simp] lemma mk_mem_nonUniforms :
    (u, v) ∈ P.nonUniforms G ε ↔ u ∈ P.parts ∧ v ∈ P.parts ∧ u ≠ v ∧ ¬G.IsUniform ε u v := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    P : Finpartition A
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    u v : Finset α
    ⊢ Iff (Membership.mem (P.nonUniforms G ε) { fst := u, snd := v }) (And (Member …
  -/
  rw [nonUniforms, mem_filter, mem_offDiag, and_assoc, and_assoc]
  /-
    🎉 no goals
  -/


theorem nonUniforms_mono {ε ε' : 𝕜} (h : ε ≤ ε') : P.nonUniforms G ε' ⊆ P.nonUniforms G ε :=
  monotone_filter_right _ fun _ => mt <| SimpleGraph.IsUniform.mono h


theorem nonUniforms_bot (hε : 0 < ε) : (⊥ : Finpartition A).nonUniforms G ε = ∅ := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    hε : LT.lt 0 ε
    ⊢ Eq (Bot.bot.nonUniforms G ε) EmptyCollection.emptyCollection
  -/
  rw [eq_empty_iff_forall_not_mem]
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    hε : LT.lt 0 ε
    ⊢ ∀ (x : Prod (Finset α) (Finset α)), Not (Membership.mem (Bot.bot.nonUniforms …
  -/
  rintro ⟨u, v⟩
  simp only [mk_mem_nonUniforms, parts_bot, mem_map, not_and,
                                    /-
                                      case mk
                                      α : Type u_1
                                      𝕜 : Type u_2
                                      inst✝² : LinearOrderedField 𝕜
                                      inst✝¹ : DecidableEq α
                                      A : Finset α
                                      G : SimpleGraph α
                                      inst✝ : DecidableRel G.Adj
                                      ε : 𝕜
                                      hε : LT.lt 0 ε
                                      u v : Finset α
                                      ⊢ ∀ (x : α), And (Membership.mem A x) (Eq ({ toFun := Singleton.singleton, inj …
                                    -/
    Classical.not_not, exists_imp]; dsimp
  /-
    case mk
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    hε : LT.lt 0 ε
    u v : Finset α
    ⊢ ∀ (x : α), And (Membership.mem A x) (Eq (Singleton.singleton x) u) → ∀ (x :  …
  -/
  rintro x ⟨_, rfl⟩ y ⟨_,rfl⟩ _
  /-
    case mk.intro.intro
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    hε : LT.lt 0 ε
    x : α
    left✝¹ : Membership.mem A x
    y : α
    left✝ : Membership.mem A y
    a✝ : Not (Eq (Singleton.singleton x) (Singleton.singleton y))
    ⊢ G.IsUniform ε (Singleton.singleton x) (Singleton.singleton y)
  -/
  rwa [SimpleGraph.isUniform_singleton]
  /-
    🎉 no goals
  -/


/-- A finpartition of a graph's vertex set is `ε`-uniform (aka `ε`-regular) iff the proportion of
its pairs of parts that are not `ε`-uniform is at most `ε`. -/
def IsUniform (ε : 𝕜) : Prop :=
  (#(P.nonUniforms G ε) : 𝕜) ≤ (#P.parts * (#P.parts - 1) : ℕ) * ε


lemma bot_isUniform (hε : 0 < ε) : (⊥ : Finpartition A).IsUniform G ε := by
  rw [Finpartition.IsUniform, Finpartition.card_bot, nonUniforms_bot _ hε, Finset.card_empty,
    Nat.cast_zero]
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    hε : LT.lt 0 ε
    ⊢ LE.le 0 (HMul.hMul (↑(HMul.hMul A.card (HSub.hSub A.card 1))) ε)
  -/
  exact mul_nonneg (Nat.cast_nonneg _) hε.le
  /-
    🎉 no goals
  -/


lemma isUniform_one : P.IsUniform G (1 : 𝕜) := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    P : Finpartition A
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ⊢ P.IsUniform G 1
  -/
  rw [IsUniform, mul_one, Nat.cast_le]
  refine (card_filter_le _
    (fun uv => ¬SimpleGraph.IsUniform G 1 (Prod.fst uv) (Prod.snd uv))).trans ?_
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    P : Finpartition A
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ⊢ LE.le P.parts.offDiag.card (HMul.hMul P.parts.card (HSub.hSub P.parts.card 1))
  -/
  rw [offDiag_card, Nat.mul_sub_left_distrib, mul_one]
  /-
    🎉 no goals
  -/


theorem IsUniform.mono {ε ε' : 𝕜} (hP : P.IsUniform G ε) (h : ε ≤ ε') : P.IsUniform G ε' :=
                                                                                   /-
                                                                                     α : Type u_1
                                                                                     𝕜 : Type u_2
                                                                                     inst✝² : LinearOrderedField 𝕜
                                                                                     inst✝¹ : DecidableEq α
                                                                                     A : Finset α
                                                                                     P : Finpartition A
                                                                                     G : SimpleGraph α
                                                                                     inst✝ : DecidableRel G.Adj
                                                                                     ε ε' : 𝕜
                                                                                     hP : P.IsUniform G ε
                                                                                     h : LE.le ε ε'
                                                                                     ⊢ LE.le (HMul.hMul (↑(HMul.hMul P.parts.card (HSub.hSub P.parts.card 1))) ε) ( …
                                                                                   -/
  ((Nat.cast_le.2 <| card_le_card <| P.nonUniforms_mono G h).trans hP).trans <| by gcongr
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


theorem isUniformOfEmpty (hP : P.parts = ∅) : P.IsUniform G ε := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    P : Finpartition A
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    hP : Eq P.parts EmptyCollection.emptyCollection
    ⊢ P.IsUniform G ε
  -/
  simp [IsUniform, hP, nonUniforms]
  /-
    🎉 no goals
  -/


theorem nonempty_of_not_uniform (h : ¬P.IsUniform G ε) : P.parts.Nonempty :=
  nonempty_of_ne_empty fun h₁ => h <| isUniformOfEmpty h₁


/-- A choice of witnesses of non-uniformity among the parts of a finpartition. -/
noncomputable def nonuniformWitnesses : Finset (Finset α) :=
  {t ∈ P.parts | s ≠ t ∧ ¬G.IsUniform ε s t}.image (G.nonuniformWitness ε s)


theorem nonuniformWitness_mem_nonuniformWitnesses (h : ¬G.IsUniform ε s t) (ht : t ∈ P.parts)
    (hst : s ≠ t) : G.nonuniformWitness ε s t ∈ P.nonuniformWitnesses G ε s :=
  mem_image_of_mem _ <| mem_filter.2 ⟨ht, hst, h⟩


open SimpleGraph in
lemma IsEquipartition.card_interedges_sparsePairs_le' (hP : P.IsEquipartition)
    (hε : 0 ≤ ε) :
    #((P.sparsePairs G ε).biUnion fun (U, V) ↦ G.interedges U V) ≤ ε * (#A + #P.parts) ^ 2 := by
  calc
    _ ≤ ∑ UV ∈ P.sparsePairs G ε, (#(G.interedges UV.1 UV.2) : 𝕜) := mod_cast card_biUnion_le
    _ ≤ ∑ UV ∈ P.sparsePairs G ε, ε * (#UV.1 * #UV.2) := ?_
    _ ≤ _ := sum_le_sum_of_subset_of_nonneg (filter_subset _ _) fun i _ _ ↦ by positivity
    _ = _ := (mul_sum _ _ _).symm
    _ ≤ _ := mul_le_mul_of_nonneg_left ?_ hε
    /-
      case calc_1
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      hP : P.IsEquipartition
      hε : LE.le 0 ε
      ⊢ LE.le ((P.sparsePairs G ε).sum fun UV => ↑(G.interedges UV.1 UV.2).card) ((P …
    -/
  · gcongr with UV hUV
    /-
      case calc_1.h
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      hP : P.IsEquipartition
      hε : LE.le 0 ε
      UV : Prod (Finset α) (Finset α)
      hUV : Membership.mem (P.sparsePairs G ε) UV
      ⊢ LE.le (↑(G.interedges UV.1 UV.2).card) (HMul.hMul ε (HMul.hMul ↑UV.1.card ↑U …
    -/
    obtain ⟨U, V⟩ := UV
    /-
      case calc_1.h.mk
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      hP : P.IsEquipartition
      hε : LE.le 0 ε
      U V : Finset α
      hUV : Membership.mem (P.sparsePairs G ε) { fst := U, snd := V }
      ⊢ LE.le (↑(G.interedges { fst := U, snd := V }.1 { fst := U, snd := V }.2).car …
    -/
    simp [mk_mem_sparsePairs, ← card_interedges_div_card] at hUV
    /-
      case calc_1.h.mk
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      hP : P.IsEquipartition
      hε : LE.le 0 ε
      U V : Finset α
      hUV : And (Membership.mem P.parts U) (And (Membership.mem P.parts V) (And (Not …
      ⊢ LE.le (↑(G.interedges { fst := U, snd := V }.1 { fst := U, snd := V }.2).car …
    -/
    refine ((div_lt_iff₀ ?_).1 hUV.2.2.2).le
    exact mul_pos (Nat.cast_pos.2 (P.nonempty_of_mem_parts hUV.1).card_pos)
      (Nat.cast_pos.2 (P.nonempty_of_mem_parts hUV.2.1).card_pos)
  /-
    case calc_2
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    P : Finpartition A
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    hP : P.IsEquipartition
    hε : LE.le 0 ε
    ⊢ LE.le (P.parts.offDiag.sum fun i => HMul.hMul ↑i.1.card ↑i.2.card) (HPow.hPo …
  -/
  norm_cast
  calc
    (_ : ℕ) ≤ _ := sum_le_card_nsmul P.parts.offDiag (fun i ↦ #i.1 * #i.2)
            ((#A / #P.parts + 1)^2 : ℕ) ?_
    _ ≤ (#P.parts * (#A / #P.parts) + #P.parts) ^ 2 := ?_
    _ ≤ _ := Nat.pow_le_pow_of_le_left (add_le_add_right (Nat.mul_div_le _ _) _) _
    /-
      case calc_2.calc_1
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      hP : P.IsEquipartition
      hε : LE.le 0 ε
      ⊢ ∀ (x : Prod (Finset α) (Finset α)), Membership.mem P.parts.offDiag x → LE.le …
    -/
  · simp only [Prod.forall, Finpartition.mk_mem_nonUniforms, and_imp, mem_offDiag, sq]
    /-
      case calc_2.calc_1
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      hP : P.IsEquipartition
      hε : LE.le 0 ε
      ⊢ ∀ (a b : Finset α), Membership.mem P.parts a → Membership.mem P.parts b → Ne …
    -/
    rintro U V hU hV -
    exact_mod_cast Nat.mul_le_mul (hP.card_part_le_average_add_one hU)
      (hP.card_part_le_average_add_one hV)
    /-
      case calc_2.calc_2
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      hP : P.IsEquipartition
      hε : LE.le 0 ε
      ⊢ LE.le (HSMul.hSMul P.parts.offDiag.card (HPow.hPow (HAdd.hAdd (HDiv.hDiv A.c …
    -/
  · rw [smul_eq_mul, offDiag_card, Nat.mul_sub_right_distrib, ← sq, ← mul_pow, mul_add_one (α := ℕ)]
    /-
      case calc_2.calc_2
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      hP : P.IsEquipartition
      hε : LE.le 0 ε
      ⊢ LE.le (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul P.parts.card (HDiv.hDiv A. …
    -/
    exact Nat.sub_le _ _
    /-
      🎉 no goals
    -/


lemma IsEquipartition.card_interedges_sparsePairs_le (hP : P.IsEquipartition) (hε : 0 ≤ ε) :
    #((P.sparsePairs G ε).biUnion fun (U, V) ↦ G.interedges U V) ≤ 4 * ε * #A ^ 2 := by
  calc
    _ ≤ _ := hP.card_interedges_sparsePairs_le' hε
    _ ≤ ε * (#A + #A)^2 := by gcongr; exact P.card_parts_le_card
    _ = _ := by ring


private lemma aux {i j : ℕ} (hj : 0 < j) : j * (j - 1) * (i / j + 1) ^ 2 < (i + j) ^ 2 := by
  have : j * (j - 1) < j ^ 2 := by
    rw [sq]; exact Nat.mul_lt_mul_of_pos_left (Nat.sub_lt hj zero_lt_one) hj
  /-
    i j : Nat
    hj : LT.lt 0 j
    this : LT.lt (HMul.hMul j (HSub.hSub j 1)) (HPow.hPow j 2)
    ⊢ LT.lt (HMul.hMul (HMul.hMul j (HSub.hSub j 1)) (HPow.hPow (HAdd.hAdd (HDiv.h …
  -/
  apply (Nat.mul_lt_mul_of_pos_right this <| pow_pos Nat.succ_pos' _).trans_le
  /-
    i j : Nat
    hj : LT.lt 0 j
    this : LT.lt (HMul.hMul j (HSub.hSub j 1)) (HPow.hPow j 2)
    ⊢ LE.le (HMul.hMul (HPow.hPow j 2) (HPow.hPow (HDiv.hDiv i j).succ 2)) (HPow.h …
  -/
  rw [← mul_pow]
  /-
    i j : Nat
    hj : LT.lt 0 j
    this : LT.lt (HMul.hMul j (HSub.hSub j 1)) (HPow.hPow j 2)
    ⊢ LE.le (HPow.hPow (HMul.hMul j (HDiv.hDiv i j).succ) 2) (HPow.hPow (HAdd.hAdd …
  -/
  exact Nat.pow_le_pow_of_le_left (add_le_add_right (Nat.mul_div_le i j) _) _
  /-
    🎉 no goals
  -/


lemma IsEquipartition.card_biUnion_offDiag_le' (hP : P.IsEquipartition) :
    (#(P.parts.biUnion offDiag) : 𝕜) ≤ #A * (#A + #P.parts) / #P.parts := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    inst✝ : DecidableEq α
    A : Finset α
    P : Finpartition A
    hP : P.IsEquipartition
    ⊢ LE.le (↑(P.parts.biUnion Finset.offDiag).card) (HDiv.hDiv (HMul.hMul (↑A.car …
  -/
  obtain h | h := P.parts.eq_empty_or_nonempty
    /-
      case inl
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      inst✝ : DecidableEq α
      A : Finset α
      P : Finpartition A
      hP : P.IsEquipartition
      h : Eq P.parts EmptyCollection.emptyCollection
      ⊢ LE.le (↑(P.parts.biUnion Finset.offDiag).card) (HDiv.hDiv (HMul.hMul (↑A.car …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  calc
    _ ≤ (#P.parts : 𝕜) * (↑(#A / #P.parts) * ↑(#A / #P.parts + 1)) :=
        mod_cast card_biUnion_le_card_mul _ _ _ fun U hU ↦ ?_
    _ = #P.parts * ↑(#A / #P.parts) * ↑(#A / #P.parts + 1) := by rw [mul_assoc]
    _ ≤ #A * (#A / #P.parts + 1) :=
        mul_le_mul (mod_cast Nat.mul_div_le _ _) ?_ (by positivity) (by positivity)
    _ = _ := by rw [← div_add_same (mod_cast h.card_pos.ne'), mul_div_assoc]
    /-
      case inr.calc_1
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      inst✝ : DecidableEq α
      A : Finset α
      P : Finpartition A
      hP : P.IsEquipartition
      h : P.parts.Nonempty
      ⊢ LE.le (↑(HAdd.hAdd (HDiv.hDiv A.card P.parts.card) 1)) (HAdd.hAdd (HDiv.hDiv …
    -/
  · simpa using Nat.cast_div_le
    /-
      🎉 no goals
    -/
  suffices (#U - 1) * #U ≤ #A / #P.parts * (#A / #P.parts + 1) by
    rwa [Nat.mul_sub_right_distrib, one_mul, ← offDiag_card] at this
  /-
    case inr.calc_2
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    inst✝ : DecidableEq α
    A : Finset α
    P : Finpartition A
    hP : P.IsEquipartition
    h : P.parts.Nonempty
    U : Finset α
    hU : Membership.mem P.parts U
    ⊢ LE.le (HMul.hMul (HSub.hSub U.card 1) U.card) (HMul.hMul (HDiv.hDiv A.card P …
  -/
  have := hP.card_part_le_average_add_one hU
  /-
    case inr.calc_2
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    inst✝ : DecidableEq α
    A : Finset α
    P : Finpartition A
    hP : P.IsEquipartition
    h : P.parts.Nonempty
    U : Finset α
    hU : Membership.mem P.parts U
    this : LE.le U.card (HAdd.hAdd (HDiv.hDiv A.card P.parts.card) 1)
    ⊢ LE.le (HMul.hMul (HSub.hSub U.card 1) U.card) (HMul.hMul (HDiv.hDiv A.card P …
  -/
  refine Nat.mul_le_mul ((Nat.sub_le_sub_right this 1).trans ?_) this
  /-
    case inr.calc_2
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    inst✝ : DecidableEq α
    A : Finset α
    P : Finpartition A
    hP : P.IsEquipartition
    h : P.parts.Nonempty
    U : Finset α
    hU : Membership.mem P.parts U
    this : LE.le U.card (HAdd.hAdd (HDiv.hDiv A.card P.parts.card) 1)
    ⊢ LE.le (HSub.hSub (HAdd.hAdd (HDiv.hDiv A.card P.parts.card) 1) 1) (HDiv.hDiv …
  -/
  simp only [Nat.add_succ_sub_one, add_zero, card_univ, le_rfl]
  /-
    🎉 no goals
  -/


lemma IsEquipartition.card_biUnion_offDiag_le (hε : 0 < ε) (hP : P.IsEquipartition)
    (hP' : 4 / ε ≤ #P.parts) : #(P.parts.biUnion offDiag) ≤ ε / 2 * #A ^ 2 := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    inst✝ : DecidableEq α
    A : Finset α
    P : Finpartition A
    ε : 𝕜
    hε : LT.lt 0 ε
    hP : P.IsEquipartition
    hP' : LE.le (HDiv.hDiv 4 ε) ↑P.parts.card
    ⊢ LE.le (↑(P.parts.biUnion Finset.offDiag).card) (HMul.hMul (HDiv.hDiv ε 2) (H …
  -/
  obtain rfl | hA : A = ⊥ ∨ _ := A.eq_empty_or_nonempty
    /-
      case inl
      α : Type u_1
      𝕜 : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      inst✝ : DecidableEq α
      ε : 𝕜
      hε : LT.lt 0 ε
      P : Finpartition Bot.bot
      hP : P.IsEquipartition
      hP' : LE.le (HDiv.hDiv 4 ε) ↑P.parts.card
      ⊢ LE.le (↑(P.parts.biUnion Finset.offDiag).card) (HMul.hMul (HDiv.hDiv ε 2) (H …
    -/
  · simp [Subsingleton.elim P ⊥]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    inst✝ : DecidableEq α
    A : Finset α
    P : Finpartition A
    ε : 𝕜
    hε : LT.lt 0 ε
    hP : P.IsEquipartition
    hP' : LE.le (HDiv.hDiv 4 ε) ↑P.parts.card
    hA : A.Nonempty
    ⊢ LE.le (↑(P.parts.biUnion Finset.offDiag).card) (HMul.hMul (HDiv.hDiv ε 2) (H …
  -/
  apply hP.card_biUnion_offDiag_le'.trans
  /-
    case inr
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    inst✝ : DecidableEq α
    A : Finset α
    P : Finpartition A
    ε : 𝕜
    hε : LT.lt 0 ε
    hP : P.IsEquipartition
    hP' : LE.le (HDiv.hDiv 4 ε) ↑P.parts.card
    hA : A.Nonempty
    ⊢ LE.le (HDiv.hDiv (HMul.hMul (↑A.card) (HAdd.hAdd ↑A.card ↑P.parts.card)) ↑P. …
  -/
  rw [div_le_iff₀ (Nat.cast_pos.2 (P.parts_nonempty hA.ne_empty).card_pos)]
  have : (#A : 𝕜) + #P.parts ≤ 2 * #A := by
    rw [two_mul]; exact add_le_add_left (Nat.cast_le.2 P.card_parts_le_card) _
  /-
    case inr
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    inst✝ : DecidableEq α
    A : Finset α
    P : Finpartition A
    ε : 𝕜
    hε : LT.lt 0 ε
    hP : P.IsEquipartition
    hP' : LE.le (HDiv.hDiv 4 ε) ↑P.parts.card
    hA : A.Nonempty
    this : LE.le (HAdd.hAdd ↑A.card ↑P.parts.card) (HMul.hMul 2 ↑A.card)
    ⊢ LE.le (HMul.hMul (↑A.card) (HAdd.hAdd ↑A.card ↑P.parts.card)) (HMul.hMul (HM …
  -/
  refine (mul_le_mul_of_nonneg_left this <| by positivity).trans ?_
  suffices 1 ≤ ε/4 * #P.parts by
    rw [mul_left_comm, ← sq]
    convert mul_le_mul_of_nonneg_left this (mul_nonneg zero_le_two <| sq_nonneg (#A : 𝕜))
      using 1 <;> ring
  /-
    case inr
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    inst✝ : DecidableEq α
    A : Finset α
    P : Finpartition A
    ε : 𝕜
    hε : LT.lt 0 ε
    hP : P.IsEquipartition
    hP' : LE.le (HDiv.hDiv 4 ε) ↑P.parts.card
    hA : A.Nonempty
    this : LE.le (HAdd.hAdd ↑A.card ↑P.parts.card) (HMul.hMul 2 ↑A.card)
    ⊢ LE.le 1 (HMul.hMul (HDiv.hDiv ε 4) ↑P.parts.card)
  -/
  rwa [← div_le_iff₀', one_div_div]
  /-
    case inr
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : LinearOrderedField 𝕜
    inst✝ : DecidableEq α
    A : Finset α
    P : Finpartition A
    ε : 𝕜
    hε : LT.lt 0 ε
    hP : P.IsEquipartition
    hP' : LE.le (HDiv.hDiv 4 ε) ↑P.parts.card
    hA : A.Nonempty
    this : LE.le (HAdd.hAdd ↑A.card ↑P.parts.card) (HMul.hMul 2 ↑A.card)
    ⊢ LT.lt 0 (HDiv.hDiv ε 4)
  -/
  positivity
  /-
    🎉 no goals
  -/


lemma IsEquipartition.sum_nonUniforms_lt' (hA : A.Nonempty) (hε : 0 < ε) (hP : P.IsEquipartition)
    (hG : P.IsUniform G ε) :
    ∑ i ∈ P.nonUniforms G ε, (#i.1 * #i.2 : 𝕜) < ε * (#A + #P.parts) ^ 2 := by
  calc
    _ ≤ #(P.nonUniforms G ε) • (↑(#A / #P.parts + 1) : 𝕜) ^ 2 :=
      sum_le_card_nsmul _ _ _ ?_
    _ = _ := nsmul_eq_mul _ _
    _ ≤ _ := mul_le_mul_of_nonneg_right hG <| by positivity
    _ < _ := ?_
    /-
      case calc_1
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      hA : A.Nonempty
      hε : LT.lt 0 ε
      hP : P.IsEquipartition
      hG : P.IsUniform G ε
      ⊢ ∀ (x : Prod (Finset α) (Finset α)), Membership.mem (P.nonUniforms G ε) x → L …
    -/
  · simp only [Prod.forall, Finpartition.mk_mem_nonUniforms, and_imp]
    /-
      case calc_1
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      hA : A.Nonempty
      hε : LT.lt 0 ε
      hP : P.IsEquipartition
      hG : P.IsUniform G ε
      ⊢ ∀ (a b : Finset α), Membership.mem P.parts a → Membership.mem P.parts b → Ne …
    -/
    rintro U V hU hV - -
    /-
      case calc_1
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      hA : A.Nonempty
      hε : LT.lt 0 ε
      hP : P.IsEquipartition
      hG : P.IsUniform G ε
      U V : Finset α
      hU : Membership.mem P.parts U
      hV : Membership.mem P.parts V
      ⊢ LE.le (HMul.hMul ↑U.card ↑V.card) (HPow.hPow (↑(HAdd.hAdd (HDiv.hDiv A.card  …
    -/
    rw [sq, ← Nat.cast_mul, ← Nat.cast_mul, Nat.cast_le]
    exact Nat.mul_le_mul (hP.card_part_le_average_add_one hU)
      (hP.card_part_le_average_add_one hV)
    /-
      case calc_2
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      hA : A.Nonempty
      hε : LT.lt 0 ε
      hP : P.IsEquipartition
      hG : P.IsUniform G ε
      ⊢ LT.lt (HMul.hMul (HMul.hMul (↑(HMul.hMul P.parts.card (HSub.hSub P.parts.car …
    -/
  · rw [mul_right_comm _ ε, mul_comm ε]
    /-
      case calc_2
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      hA : A.Nonempty
      hε : LT.lt 0 ε
      hP : P.IsEquipartition
      hG : P.IsUniform G ε
      ⊢ LT.lt (HMul.hMul (HMul.hMul (↑(HMul.hMul P.parts.card (HSub.hSub P.parts.car …
    -/
    apply mul_lt_mul_of_pos_right _ hε
    /-
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      hA : A.Nonempty
      hε : LT.lt 0 ε
      hP : P.IsEquipartition
      hG : P.IsUniform G ε
      ⊢ LT.lt (HMul.hMul (↑(HMul.hMul P.parts.card (HSub.hSub P.parts.card 1))) (HPo …
    -/
    norm_cast
    /-
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      hA : A.Nonempty
      hε : LT.lt 0 ε
      hP : P.IsEquipartition
      hG : P.IsUniform G ε
      ⊢ LT.lt (HMul.hMul (HMul.hMul P.parts.card (HSub.hSub P.parts.card 1)) (HPow.h …
    -/
    exact aux (P.parts_nonempty hA.ne_empty).card_pos
    /-
      🎉 no goals
    -/


lemma IsEquipartition.sum_nonUniforms_lt (hA : A.Nonempty) (hε : 0 < ε) (hP : P.IsEquipartition)
    (hG : P.IsUniform G ε) :
    #((P.nonUniforms G ε).biUnion fun (U, V) ↦ U ×ˢ V) < 4 * ε * #A ^ 2 := by
  calc
    _ ≤ ∑ i ∈ P.nonUniforms G ε, (#i.1 * #i.2 : 𝕜) := by
        norm_cast; simp_rw [← card_product]; exact card_biUnion_le
    _ < _ := hP.sum_nonUniforms_lt' hA hε hG
    _ ≤ ε * (#A + #A) ^ 2 := by gcongr; exact P.card_parts_le_card
    _ = _ := by ring


/-- The reduction of the graph `G` along partition `P` has edges between `ε`-uniform pairs of parts
that have edge density at least `δ`. -/
@[simps] def regularityReduced (ε δ : 𝕜) : SimpleGraph α where
  Adj a b := G.Adj a b ∧
    ∃ U ∈ P.parts, ∃ V ∈ P.parts, a ∈ U ∧ b ∈ V ∧ U ≠ V ∧ G.IsUniform ε U V ∧ δ ≤ G.edgeDensity U V
  symm a b := by
    /-
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε✝ δ✝ : 𝕜
      u v : Finset α
      ε δ : 𝕜
      a b : α
      ⊢ (fun a b => And (G.Adj a b) (Exists fun U => And (Membership.mem P.parts U)  …
    -/
    rintro ⟨ab, U, UP, V, VP, xU, yV, UV, GUV, εUV⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε✝ δ✝ : 𝕜
      u v : Finset α
      ε δ : 𝕜
      a b : α
      ab : G.Adj a b
      U : Finset α
      UP : Membership.mem P.parts U
      V : Finset α
      VP : Membership.mem P.parts V
      xU : Membership.mem U a
      yV : Membership.mem V b
      UV : Ne U V
      GUV : G.IsUniform ε U V
      εUV : LE.le δ ↑(G.edgeDensity U V)
      ⊢ And (G.Adj b a) (Exists fun U => And (Membership.mem P.parts U) (Exists fun  …
    -/
    refine ⟨G.symm ab, V, VP, U, UP, yV, xU, UV.symm, GUV.symm, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε✝ δ✝ : 𝕜
      u v : Finset α
      ε δ : 𝕜
      a b : α
      ab : G.Adj a b
      U : Finset α
      UP : Membership.mem P.parts U
      V : Finset α
      VP : Membership.mem P.parts V
      xU : Membership.mem U a
      yV : Membership.mem V b
      UV : Ne U V
      GUV : G.IsUniform ε U V
      εUV : LE.le δ ↑(G.edgeDensity U V)
      ⊢ LE.le δ ↑(G.edgeDensity V U)
    -/
    rwa [edgeDensity_comm]
    /-
      🎉 no goals
    -/
  loopless a h := G.loopless a h.1


instance regularityReduced.instDecidableRel_adj : DecidableRel (G.regularityReduced P ε δ).Adj := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    P : Finpartition A
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε δ : 𝕜
    u v : Finset α
    ⊢ DecidableRel (SimpleGraph.regularityReduced P G ε δ).Adj
  -/
  unfold regularityReduced; infer_instance
                            /-
                              🎉 no goals
                            -/


lemma regularityReduced_le : G.regularityReduced P ε δ ≤ G := fun _ _ ↦ And.left


lemma regularityReduced_mono {ε₁ ε₂ : 𝕜} (hε : ε₁ ≤ ε₂) :
    G.regularityReduced P ε₁ δ ≤ G.regularityReduced P ε₂ δ :=
  fun _a _b ⟨hab, U, hU, V, hV, ha, hb, hUV, hGε, hGδ⟩ ↦
    ⟨hab, U, hU, V, hV, ha, hb, hUV, hGε.mono hε, hGδ⟩


lemma regularityReduced_anti {δ₁ δ₂ : 𝕜} (hδ : δ₁ ≤ δ₂) :
    G.regularityReduced P ε δ₂ ≤ G.regularityReduced P ε δ₁ :=
  fun _a _b ⟨hab, U, hU, V, hV, ha, hb, hUV, hUVε, hUVδ⟩ ↦
    ⟨hab, U, hU, V, hV, ha, hb, hUV, hUVε, hδ.trans hUVδ⟩


lemma unreduced_edges_subset :
    (A ×ˢ A).filter (fun (x, y) ↦ G.Adj x y ∧ ¬ (G.regularityReduced P (ε/8) (ε/4)).Adj x y) ⊆
      (P.nonUniforms G (ε/8)).biUnion (fun (U, V) ↦ U ×ˢ V) ∪ P.parts.biUnion offDiag ∪
        (P.sparsePairs G (ε/4)).biUnion fun (U, V) ↦ G.interedges U V := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    P : Finpartition A
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    ⊢ HasSubset.Subset (Finset.filter (fun x => SimpleGraph.unreduced_edges_subset …
  -/
  rintro ⟨x, y⟩
  simp only [mem_sdiff, mem_filter, mem_univ, true_and, regularityReduced_adj, not_and, not_exists,
    not_le, mem_biUnion, mem_union, exists_prop, mem_product, Prod.exists, mem_offDiag, and_imp,
    or_assoc, and_assoc, P.mk_mem_nonUniforms, Finpartition.mk_mem_sparsePairs, mem_interedges_iff]
  /-
    case mk
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    P : Finpartition A
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    x y : α
    ⊢ Membership.mem A x → Membership.mem A y → G.Adj x y → (G.Adj x y → ∀ (x_1 :  …
  -/
  intros hx hy h h'
  /-
    case mk
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    P : Finpartition A
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    x y : α
    hx : Membership.mem A x
    hy : Membership.mem A y
    h : G.Adj x y
    h' : G.Adj x y → ∀ (x_1 : Finset α), Membership.mem P.parts x_1 → ∀ (x_2 : Fin …
    ⊢ Or (Exists fun a => Exists fun b => And (Membership.mem P.parts a) (And (Mem …
  -/
  replace h' := h' h
  /-
    case mk
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    P : Finpartition A
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    x y : α
    hx : Membership.mem A x
    hy : Membership.mem A y
    h : G.Adj x y
    h' : ∀ (x_1 : Finset α), Membership.mem P.parts x_1 → ∀ (x_2 : Finset α), Memb …
    ⊢ Or (Exists fun a => Exists fun b => And (Membership.mem P.parts a) (And (Mem …
  -/
  obtain ⟨U, hU, hx⟩ := P.exists_mem hx
  /-
    case mk.intro.intro
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    P : Finpartition A
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    x y : α
    hx✝ : Membership.mem A x
    hy : Membership.mem A y
    h : G.Adj x y
    h' : ∀ (x_1 : Finset α), Membership.mem P.parts x_1 → ∀ (x_2 : Finset α), Memb …
    U : Finset α
    hU : Membership.mem P.parts U
    hx : Membership.mem U x
    ⊢ Or (Exists fun a => Exists fun b => And (Membership.mem P.parts a) (And (Mem …
  -/
  obtain ⟨V, hV, hy⟩ := P.exists_mem hy
  /-
    case mk.intro.intro.intro.intro
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    P : Finpartition A
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    x y : α
    hx✝ : Membership.mem A x
    hy✝ : Membership.mem A y
    h : G.Adj x y
    h' : ∀ (x_1 : Finset α), Membership.mem P.parts x_1 → ∀ (x_2 : Finset α), Memb …
    U : Finset α
    hU : Membership.mem P.parts U
    hx : Membership.mem U x
    V : Finset α
    hV : Membership.mem P.parts V
    hy : Membership.mem V y
    ⊢ Or (Exists fun a => Exists fun b => And (Membership.mem P.parts a) (And (Mem …
  -/
  obtain rfl | hUV := eq_or_ne U V
    /-
      case mk.intro.intro.intro.intro.inl
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      x y : α
      hx✝ : Membership.mem A x
      hy✝ : Membership.mem A y
      h : G.Adj x y
      h' : ∀ (x_1 : Finset α), Membership.mem P.parts x_1 → ∀ (x_2 : Finset α), Memb …
      U : Finset α
      hU : Membership.mem P.parts U
      hx : Membership.mem U x
      hV : Membership.mem P.parts U
      hy : Membership.mem U y
      ⊢ Or (Exists fun a => Exists fun b => And (Membership.mem P.parts a) (And (Mem …
    -/
  · exact Or.inr (Or.inl ⟨U, hU, hx, hy, G.ne_of_adj h⟩)
    /-
      🎉 no goals
    -/
  /-
    case mk.intro.intro.intro.intro.inr
    α : Type u_1
    𝕜 : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : DecidableEq α
    A : Finset α
    P : Finpartition A
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : 𝕜
    x y : α
    hx✝ : Membership.mem A x
    hy✝ : Membership.mem A y
    h : G.Adj x y
    h' : ∀ (x_1 : Finset α), Membership.mem P.parts x_1 → ∀ (x_2 : Finset α), Memb …
    U : Finset α
    hU : Membership.mem P.parts U
    hx : Membership.mem U x
    V : Finset α
    hV : Membership.mem P.parts V
    hy : Membership.mem V y
    hUV : Ne U V
    ⊢ Or (Exists fun a => Exists fun b => And (Membership.mem P.parts a) (And (Mem …
  -/
  by_cases h₂ : G.IsUniform (ε/8) U V
    /-
      case pos
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      x y : α
      hx✝ : Membership.mem A x
      hy✝ : Membership.mem A y
      h : G.Adj x y
      h' : ∀ (x_1 : Finset α), Membership.mem P.parts x_1 → ∀ (x_2 : Finset α), Memb …
      U : Finset α
      hU : Membership.mem P.parts U
      hx : Membership.mem U x
      V : Finset α
      hV : Membership.mem P.parts V
      hy : Membership.mem V y
      hUV : Ne U V
      h₂ : G.IsUniform (HDiv.hDiv ε 8) U V
      ⊢ Or (Exists fun a => Exists fun b => And (Membership.mem P.parts a) (And (Mem …
    -/
  · exact Or.inr <| Or.inr ⟨U, V, hU, hV, hUV, h' _ hU _ hV hx hy hUV h₂, hx, hy, h⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      𝕜 : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : DecidableEq α
      A : Finset α
      P : Finpartition A
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : 𝕜
      x y : α
      hx✝ : Membership.mem A x
      hy✝ : Membership.mem A y
      h : G.Adj x y
      h' : ∀ (x_1 : Finset α), Membership.mem P.parts x_1 → ∀ (x_2 : Finset α), Memb …
      U : Finset α
      hU : Membership.mem P.parts U
      hx : Membership.mem U x
      V : Finset α
      hV : Membership.mem P.parts V
      hy : Membership.mem V y
      hUV : Ne U V
      h₂ : Not (G.IsUniform (HDiv.hDiv ε 8) U V)
      ⊢ Or (Exists fun a => Exists fun b => And (Membership.mem P.parts a) (And (Mem …
    -/
  · exact Or.inl ⟨U, V, hU, hV, hUV, h₂, hx, hy⟩
    /-
      🎉 no goals
    -/


