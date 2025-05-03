local notation3 "m" => (card α / stepBound #P.parts : ℕ)


/-- The **increment partition** in Szemerédi's Regularity Lemma.

If an equipartition is *not* uniform, then the increment partition is a (much bigger) equipartition
with a slightly higher energy. This is helpful since the energy is bounded by a constant (see
`Finpartition.energy_le_one`), so this process eventually terminates and yields a
not-too-big uniform equipartition. -/
noncomputable def increment : Finpartition (univ : Finset α) :=
  P.bind fun _ => chunk hP G ε


/-- The increment partition has a prescribed (very big) size in terms of the original partition. -/
theorem card_increment (hPα : #P.parts * 16 ^ #P.parts ≤ card α) (hPG : ¬P.IsUniform G ε) :
    #(increment hP G ε).parts = stepBound #P.parts := by
  have hPα' : stepBound #P.parts ≤ card α :=
    (mul_le_mul_left' (pow_le_pow_left' (by norm_num) _) _).trans hPα
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPG : Not (P.IsUniform G ε)
    hPα' : LE.le (SzemerediRegularity.stepBound P.parts.card) (Fintype.card α)
    ⊢ Eq (SzemerediRegularity.increment hP G ε).parts.card (SzemerediRegularity.st …
  -/
  have hPpos : 0 < stepBound #P.parts := stepBound_pos (nonempty_of_not_uniform hPG).card_pos
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPG : Not (P.IsUniform G ε)
    hPα' : LE.le (SzemerediRegularity.stepBound P.parts.card) (Fintype.card α)
    hPpos : LT.lt 0 (SzemerediRegularity.stepBound P.parts.card)
    ⊢ Eq (SzemerediRegularity.increment hP G ε).parts.card (SzemerediRegularity.st …
  -/
  rw [increment, card_bind]
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPG : Not (P.IsUniform G ε)
    hPα' : LE.le (SzemerediRegularity.stepBound P.parts.card) (Fintype.card α)
    hPpos : LT.lt 0 (SzemerediRegularity.stepBound P.parts.card)
    ⊢ Eq (P.parts.attach.sum fun A => (SzemerediRegularity.chunk hP G ε ⋯).parts.c …
  -/
  simp_rw [chunk, apply_dite Finpartition.parts, apply_dite card, sum_dite]
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPG : Not (P.IsUniform G ε)
    hPα' : LE.le (SzemerediRegularity.stepBound P.parts.card) (Fintype.card α)
    hPpos : LT.lt 0 (SzemerediRegularity.stepBound P.parts.card)
    ⊢ Eq (HAdd.hAdd (Finset.univ.sum fun x => (Finpartition.equitabilise ⋯).parts. …
  -/
  rw [sum_const_nat, sum_const_nat, univ_eq_attach, univ_eq_attach, card_attach, card_attach]
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPG : Not (P.IsUniform G ε)
    hPα' : LE.le (SzemerediRegularity.stepBound P.parts.card) (Fintype.card α)
    hPpos : LT.lt 0 (SzemerediRegularity.stepBound P.parts.card)
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Finset.filter (fun x => Eq (↑x).card (HAdd.hAdd (H …
  -/
  any_goals exact fun x hx => card_parts_equitabilise _ _ (Nat.div_pos hPα' hPpos).ne'
  rw [Nat.sub_add_cancel a_add_one_le_four_pow_parts_card,
    Nat.sub_add_cancel ((Nat.le_succ _).trans a_add_one_le_four_pow_parts_card), ← add_mul]
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPG : Not (P.IsUniform G ε)
    hPα' : LE.le (SzemerediRegularity.stepBound P.parts.card) (Fintype.card α)
    hPpos : LT.lt 0 (SzemerediRegularity.stepBound P.parts.card)
    ⊢ Eq (HMul.hMul (HAdd.hAdd (Finset.filter (fun x => Eq (↑x).card (HAdd.hAdd (H …
  -/
  congr
  /-
    case e_a
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPG : Not (P.IsUniform G ε)
    hPα' : LE.le (SzemerediRegularity.stepBound P.parts.card) (Fintype.card α)
    hPpos : LT.lt 0 (SzemerediRegularity.stepBound P.parts.card)
    ⊢ Eq (HAdd.hAdd (Finset.filter (fun x => Eq (↑x).card (HAdd.hAdd (HMul.hMul (H …
  -/
  rw [filter_card_add_filter_neg_card_eq_card, card_attach]
  /-
    🎉 no goals
  -/


theorem increment_isEquipartition : (increment hP G ε).IsEquipartition := by
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    ⊢ (SzemerediRegularity.increment hP G ε).IsEquipartition
  -/
  simp_rw [IsEquipartition, Set.equitableOn_iff_exists_eq_eq_add_one]
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    ⊢ Exists fun b => ∀ (a : Finset α), Membership.mem (↑(SzemerediRegularity.incr …
  -/
  refine ⟨m, fun A hA => ?_⟩
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    A : Finset α
    hA : Membership.mem (↑(SzemerediRegularity.increment hP G ε).parts) A
    ⊢ Or (Eq A.card (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P.p …
  -/
  rw [mem_coe, increment, mem_bind] at hA
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    A : Finset α
    hA : Exists fun A_1 => Exists fun hA => Membership.mem (SzemerediRegularity.ch …
    ⊢ Or (Eq A.card (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P.p …
  -/
  obtain ⟨U, hU, hA⟩ := hA
  /-
    case intro.intro
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    A U : Finset α
    hU : Membership.mem P.parts U
    hA : Membership.mem (SzemerediRegularity.chunk hP G ε hU).parts A
    ⊢ Or (Eq A.card (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P.p …
  -/
  exact card_eq_of_mem_parts_chunk hA
  /-
    🎉 no goals
  -/


/-- The contribution to `Finpartition.energy` of a pair of distinct parts of a `Finpartition`. -/
private noncomputable def distinctPairs (x : {x // x ∈ P.parts.offDiag}) :
    Finset (Finset α × Finset α) :=
  (chunk hP G ε (mem_offDiag.1 x.2).1).parts ×ˢ (chunk hP G ε (mem_offDiag.1 x.2).2.1).parts


private theorem distinctPairs_increment :
    P.parts.offDiag.attach.biUnion (distinctPairs hP G ε) ⊆ (increment hP G ε).parts.offDiag := by
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    ⊢ HasSubset.Subset (P.parts.offDiag.attach.biUnion (SzemerediRegularity.distin …
  -/
  rintro ⟨Ui, Vj⟩
  simp only [distinctPairs, increment, mem_offDiag, bind_parts, mem_biUnion, Prod.exists,
    exists_and_left, exists_prop, mem_product, mem_attach, true_and, Subtype.exists, and_imp,
    mem_offDiag, forall_exists_index, exists₂_imp, Ne]
  /-
    case mk
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    Ui Vj : Finset α
    ⊢ ∀ (x x_1 : Finset α) (x_2 : And (Membership.mem P.parts x) (And (Membership. …
  -/
  refine fun U V hUV hUi hVj => ⟨⟨_, hUV.1, hUi⟩, ⟨_, hUV.2.1, hVj⟩, ?_⟩
  /-
    case mk
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    Ui Vj U V : Finset α
    hUV : And (Membership.mem P.parts U) (And (Membership.mem P.parts V) (Not (Eq  …
    hUi : Membership.mem (SzemerediRegularity.chunk hP G ε ⋯).parts Ui
    hVj : Membership.mem (SzemerediRegularity.chunk hP G ε ⋯).parts Vj
    ⊢ Not (Eq Ui Vj)
  -/
  rintro rfl
  /-
    case mk
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    Ui U V : Finset α
    hUV : And (Membership.mem P.parts U) (And (Membership.mem P.parts V) (Not (Eq  …
    hUi : Membership.mem (SzemerediRegularity.chunk hP G ε ⋯).parts Ui
    hVj : Membership.mem (SzemerediRegularity.chunk hP G ε ⋯).parts Ui
    ⊢ False
  -/
  obtain ⟨i, hi⟩ := nonempty_of_mem_parts _ hUi
  exact hUV.2.2 (P.disjoint.elim_finset hUV.1 hUV.2.1 i (Finpartition.le _ hUi hi) <|
    Finpartition.le _ hVj hi)


private lemma pairwiseDisjoint_distinctPairs :
    (P.parts.offDiag.attach : Set {x // x ∈ P.parts.offDiag}).PairwiseDisjoint
      (distinctPairs hP G ε) := by
  simp (config := { unfoldPartialApp := true }) only [distinctPairs, Set.PairwiseDisjoint,
    Function.onFun, disjoint_left, inf_eq_inter, mem_inter, mem_product]
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    ⊢ (↑P.parts.offDiag.attach).Pairwise fun x y => ∀ ⦃a : Prod (Finset α) (Finset …
  -/
  rintro ⟨⟨s₁, s₂⟩, hs⟩ _ ⟨⟨t₁, t₂⟩, ht⟩ _ hst ⟨u, v⟩ huv₁ huv₂
  /-
    case mk.mk.mk.mk.mk
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    s₁ s₂ : Finset α
    hs : Membership.mem P.parts.offDiag { fst := s₁, snd := s₂ }
    a✝¹ : Membership.mem ↑P.parts.offDiag.attach ⟨{ fst := s₁, snd := s₂ }, hs⟩
    t₁ t₂ : Finset α
    ht : Membership.mem P.parts.offDiag { fst := t₁, snd := t₂ }
    a✝ : Membership.mem ↑P.parts.offDiag.attach ⟨{ fst := t₁, snd := t₂ }, ht⟩
    hst : Ne ⟨{ fst := s₁, snd := s₂ }, hs⟩ ⟨{ fst := t₁, snd := t₂ }, ht⟩
    u v : Finset α
    huv₁ : And (Membership.mem (SzemerediRegularity.chunk hP G ε ⋯).parts { fst := …
    huv₂ : And (Membership.mem (SzemerediRegularity.chunk hP G ε ⋯).parts { fst := …
    ⊢ False
  -/
  rw [mem_offDiag] at hs ht
  /-
    case mk.mk.mk.mk.mk
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    s₁ s₂ : Finset α
    hs✝ : Membership.mem P.parts.offDiag { fst := s₁, snd := s₂ }
    hs : And (Membership.mem P.parts { fst := s₁, snd := s₂ }.1) (And (Membership. …
    a✝¹ : Membership.mem ↑P.parts.offDiag.attach ⟨{ fst := s₁, snd := s₂ }, hs✝⟩
    t₁ t₂ : Finset α
    ht✝ : Membership.mem P.parts.offDiag { fst := t₁, snd := t₂ }
    ht : And (Membership.mem P.parts { fst := t₁, snd := t₂ }.1) (And (Membership. …
    a✝ : Membership.mem ↑P.parts.offDiag.attach ⟨{ fst := t₁, snd := t₂ }, ht✝⟩
    hst : Ne ⟨{ fst := s₁, snd := s₂ }, hs✝⟩ ⟨{ fst := t₁, snd := t₂ }, ht✝⟩
    u v : Finset α
    huv₁ : And (Membership.mem (SzemerediRegularity.chunk hP G ε ⋯).parts { fst := …
    huv₂ : And (Membership.mem (SzemerediRegularity.chunk hP G ε ⋯).parts { fst := …
    ⊢ False
  -/
  obtain ⟨a, ha⟩ := Finpartition.nonempty_of_mem_parts _ huv₁.1
  /-
    case mk.mk.mk.mk.mk.intro
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    s₁ s₂ : Finset α
    hs✝ : Membership.mem P.parts.offDiag { fst := s₁, snd := s₂ }
    hs : And (Membership.mem P.parts { fst := s₁, snd := s₂ }.1) (And (Membership. …
    a✝¹ : Membership.mem ↑P.parts.offDiag.attach ⟨{ fst := s₁, snd := s₂ }, hs✝⟩
    t₁ t₂ : Finset α
    ht✝ : Membership.mem P.parts.offDiag { fst := t₁, snd := t₂ }
    ht : And (Membership.mem P.parts { fst := t₁, snd := t₂ }.1) (And (Membership. …
    a✝ : Membership.mem ↑P.parts.offDiag.attach ⟨{ fst := t₁, snd := t₂ }, ht✝⟩
    hst : Ne ⟨{ fst := s₁, snd := s₂ }, hs✝⟩ ⟨{ fst := t₁, snd := t₂ }, ht✝⟩
    u v : Finset α
    huv₁ : And (Membership.mem (SzemerediRegularity.chunk hP G ε ⋯).parts { fst := …
    huv₂ : And (Membership.mem (SzemerediRegularity.chunk hP G ε ⋯).parts { fst := …
    a : α
    ha : Membership.mem { fst := u, snd := v }.1 a
    ⊢ False
  -/
  obtain ⟨b, hb⟩ := Finpartition.nonempty_of_mem_parts _ huv₁.2
  exact hst <| Subtype.ext_val <| Prod.ext
    (P.disjoint.elim_finset hs.1 ht.1 a (Finpartition.le _ huv₁.1 ha) <|
      Finpartition.le _ huv₂.1 ha) <|
        P.disjoint.elim_finset hs.2.1 ht.2.1 b (Finpartition.le _ huv₁.2 hb) <|
          Finpartition.le _ huv₂.2 hb


lemma le_sum_distinctPairs_edgeDensity_sq (x : {i // i ∈ P.parts.offDiag}) (hε₁ : ε ≤ 1)
    (hPα : #P.parts * 16 ^ #P.parts ≤ card α) (hPε : ↑100 ≤ ↑4 ^ #P.parts * ε ^ 5) :
    (G.edgeDensity x.1.1 x.1.2 : ℝ) ^ 2 +
      ((if G.IsUniform ε x.1.1 x.1.2 then 0 else ε ^ 4 / 3) - ε ^ 5 / 25) ≤
    (∑ i ∈ distinctPairs hP G ε x, G.edgeDensity i.1 i.2 ^ 2 : ℝ) / 16 ^ #P.parts := by
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    inst✝ : Nonempty α
    x : Subtype fun i => Membership.mem P.parts.offDiag i
    hε₁ : LE.le ε 1
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    ⊢ LE.le (HAdd.hAdd (HPow.hPow (↑(G.edgeDensity (↑x).1 (↑x).2)) 2) (HSub.hSub ( …
  -/
  rw [distinctPairs, ← add_sub_assoc, add_sub_right_comm]
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    inst✝ : Nonempty α
    x : Subtype fun i => Membership.mem P.parts.offDiag i
    hε₁ : LE.le ε 1
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    ⊢ LE.le (HAdd.hAdd (HSub.hSub (HPow.hPow (↑(G.edgeDensity (↑x).1 (↑x).2)) 2) ( …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      inst✝ : Nonempty α
      x : Subtype fun i => Membership.mem P.parts.offDiag i
      hε₁ : LE.le ε 1
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      h : G.IsUniform ε (↑x).1 (↑x).2
      ⊢ LE.le (HAdd.hAdd (HSub.hSub (HPow.hPow (↑(G.edgeDensity (↑x).1 (↑x).2)) 2) ( …
    -/
  · rw [add_zero]
    /-
      case pos
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      inst✝ : Nonempty α
      x : Subtype fun i => Membership.mem P.parts.offDiag i
      hε₁ : LE.le ε 1
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      h : G.IsUniform ε (↑x).1 (↑x).2
      ⊢ LE.le (HSub.hSub (HPow.hPow (↑(G.edgeDensity (↑x).1 (↑x).2)) 2) (HDiv.hDiv ( …
    -/
    exact edgeDensity_chunk_uniform hPα hPε _ _
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      inst✝ : Nonempty α
      x : Subtype fun i => Membership.mem P.parts.offDiag i
      hε₁ : LE.le ε 1
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      h : Not (G.IsUniform ε (↑x).1 (↑x).2)
      ⊢ LE.le (HAdd.hAdd (HSub.hSub (HPow.hPow (↑(G.edgeDensity (↑x).1 (↑x).2)) 2) ( …
    -/
  · exact edgeDensity_chunk_not_uniform hPα hPε hε₁ (mem_offDiag.1 x.2).2.2 h
    /-
      🎉 no goals
    -/


/-- The increment partition has energy greater than the original one by a known fixed amount. -/
theorem energy_increment (hP : P.IsEquipartition) (hP₇ : 7 ≤ #P.parts)
    (hPε : 100 ≤ 4 ^ #P.parts * ε ^ 5) (hPα : #P.parts * 16 ^ #P.parts ≤ card α)
    (hPG : ¬P.IsUniform G ε) (hε₀ : 0 ≤ ε) (hε₁ : ε ≤ 1) :
    ↑(P.energy G) + ε ^ 5 / 4 ≤ (increment hP G ε).energy G := by
  calc
    _ = (∑ x ∈ P.parts.offDiag, (G.edgeDensity x.1 x.2 : ℝ) ^ 2 +
          #P.parts ^ 2 * (ε ^ 5 / 4) : ℝ) / #P.parts ^ 2 := by
        rw [coe_energy, add_div, mul_div_cancel_left₀]; positivity
    _ ≤ (∑ x ∈ P.parts.offDiag.attach, (∑ i ∈ distinctPairs hP G ε x,
          G.edgeDensity i.1 i.2 ^ 2 : ℝ) / 16 ^ #P.parts) / #P.parts ^ 2 := ?_
    _ = (∑ x ∈ P.parts.offDiag.attach, ∑ i ∈ distinctPairs hP G ε x,
          G.edgeDensity i.1 i.2 ^ 2 : ℝ) / #(increment hP G ε).parts ^ 2 := by
        rw [card_increment hPα hPG, coe_stepBound, mul_pow, pow_right_comm,
          div_mul_eq_div_div_swap, ← sum_div]; norm_num
    _ ≤ _ := by
        rw [coe_energy]
        gcongr
        rw [← sum_biUnion pairwiseDisjoint_distinctPairs]
        exact sum_le_sum_of_subset_of_nonneg distinctPairs_increment fun i _ _ ↦ sq_nonneg _
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    inst✝ : Nonempty α
    hP : P.IsEquipartition
    hP₇ : LE.le 7 P.parts.card
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPG : Not (P.IsUniform G ε)
    hε₀ : LE.le 0 ε
    hε₁ : LE.le ε 1
    ⊢ LE.le (HDiv.hDiv (HAdd.hAdd (P.parts.offDiag.sum fun x => HPow.hPow (↑(G.edg …
  -/
  gcongr
  /-
    case hab
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    inst✝ : Nonempty α
    hP : P.IsEquipartition
    hP₇ : LE.le 7 P.parts.card
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPG : Not (P.IsUniform G ε)
    hε₀ : LE.le 0 ε
    hε₁ : LE.le ε 1
    ⊢ LE.le (HAdd.hAdd (P.parts.offDiag.sum fun x => HPow.hPow (↑(G.edgeDensity x. …
  -/
  rw [Finpartition.IsUniform, not_le, mul_tsub, mul_one, ← offDiag_card] at hPG
  calc
    _ ≤ ∑ x ∈ P.parts.offDiag, (edgeDensity G x.1 x.2 : ℝ) ^ 2 +
        (#(nonUniforms P G ε) * (ε ^ 4 / 3) - #P.parts.offDiag * (ε ^ 5 / 25)) :=
        add_le_add_left ?_ _
    _ = ∑ x ∈ P.parts.offDiag, ((G.edgeDensity x.1 x.2 : ℝ) ^ 2 +
        ((if G.IsUniform ε x.1 x.2 then (0 : ℝ) else ε ^ 4 / 3) - ε ^ 5 / 25) : ℝ) := by
        rw [sum_add_distrib, sum_sub_distrib, sum_const, nsmul_eq_mul, sum_ite, sum_const_zero,
          zero_add, sum_const, nsmul_eq_mul, ← Finpartition.nonUniforms, ← add_sub_assoc,
          add_sub_right_comm]
    _ = _ := (sum_attach ..).symm
    _ ≤ _ := sum_le_sum fun i _ ↦ le_sum_distinctPairs_edgeDensity_sq i hε₁ hPα hPε
  calc
    _ = (6/7 * #P.parts ^ 2) * ε ^ 5 * (7 / 24) := by ring
    _ ≤ #P.parts.offDiag * ε ^ 5 * (22 / 75) := by
        gcongr ?_ * _ * ?_
        · rw [← mul_div_right_comm, div_le_iff₀ (by norm_num), offDiag_card]
          norm_cast
          rw [tsub_mul]
          refine le_tsub_of_add_le_left ?_
          nlinarith
        · norm_num
    _ = (#P.parts.offDiag * ε * (ε ^ 4 / 3) - #P.parts.offDiag * (ε ^ 5 / 25)) := by ring
    _ ≤ (#(nonUniforms P G ε) * (ε ^ 4 / 3) - #P.parts.offDiag * (ε ^ 5 / 25)) := by gcongr


