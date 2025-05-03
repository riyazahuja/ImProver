/-- A graph has edge-disjoint triangles if each edge belongs to at most one triangle. -/
def EdgeDisjointTriangles (G : SimpleGraph α) : Prop :=
  (G.cliqueSet 3).Pairwise fun x y ↦ (x ∩ y : Set α).Subsingleton


/-- A graph is locally linear if each edge belongs to exactly one triangle. -/
def LocallyLinear (G : SimpleGraph α) : Prop :=
  G.EdgeDisjointTriangles ∧ ∀ ⦃x y⦄, G.Adj x y → ∃ s, G.IsNClique 3 s ∧ x ∈ s ∧ y ∈ s


protected lemma LocallyLinear.edgeDisjointTriangles : G.LocallyLinear → G.EdgeDisjointTriangles :=
  And.left


nonrec lemma EdgeDisjointTriangles.mono (h : G ≤ H) (hH : H.EdgeDisjointTriangles) :
    G.EdgeDisjointTriangles := hH.mono <| cliqueSet_mono h


@[simp] lemma edgeDisjointTriangles_bot : (⊥ : SimpleGraph α).EdgeDisjointTriangles := by
  /-
    α : Type u_1
    ⊢ Bot.bot.EdgeDisjointTriangles
  -/
  simp [EdgeDisjointTriangles]
  /-
    🎉 no goals
  -/


                                                                          /-
                                                                            α : Type u_1
                                                                            ⊢ Bot.bot.LocallyLinear
                                                                          -/
@[simp] lemma locallyLinear_bot : (⊥ : SimpleGraph α).LocallyLinear := by simp [LocallyLinear]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


lemma EdgeDisjointTriangles.map (f : α ↪ β) (hG : G.EdgeDisjointTriangles) :
    (G.map f).EdgeDisjointTriangles := by
  rw [EdgeDisjointTriangles, cliqueSet_map (by norm_num : 3 ≠ 1),
    (Finset.map_injective f).injOn.pairwise_image]
  classical
  rintro s hs t ht hst
  dsimp [Function.onFun]
  rw [← coe_inter, ← map_inter, coe_map, coe_inter]
  exact (hG hs ht hst).image _


lemma LocallyLinear.map (f : α ↪ β) (hG : G.LocallyLinear) : (G.map f).LocallyLinear := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    hG : G.LocallyLinear
    ⊢ (SimpleGraph.map f G).LocallyLinear
  -/
  refine ⟨hG.1.map _, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    hG : G.LocallyLinear
    ⊢ ∀ ⦃x y : β⦄, (SimpleGraph.map f G).Adj x y → Exists fun s => And ((SimpleGra …
  -/
  rintro _ _ ⟨a, b, h, rfl, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    hG : G.LocallyLinear
    a b : α
    h : G.Adj a b
    ⊢ Exists fun s => And ((SimpleGraph.map f G).IsNClique 3 s) (And (Membership.m …
  -/
  obtain ⟨s, hs, ha, hb⟩ := hG.2 h
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    hG : G.LocallyLinear
    a b : α
    h : G.Adj a b
    s : Finset α
    hs : G.IsNClique 3 s
    ha : Membership.mem s a
    hb : Membership.mem s b
    ⊢ Exists fun s => And ((SimpleGraph.map f G).IsNClique 3 s) (And (Membership.m …
  -/
  exact ⟨s.map f, hs.map, mem_map_of_mem _ ha, mem_map_of_mem _ hb⟩
  /-
    🎉 no goals
  -/


@[simp] lemma locallyLinear_comap {G : SimpleGraph β} {e : α ≃ β} :
    (G.comap e).LocallyLinear ↔ G.LocallyLinear := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph β
    e : Equiv α β
    ⊢ Iff (SimpleGraph.comap (⇑e) G).LocallyLinear G.LocallyLinear
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      G : SimpleGraph β
      e : Equiv α β
      h : (SimpleGraph.comap (⇑e) G).LocallyLinear
      ⊢ G.LocallyLinear
    -/
  · rw [← comap_map_eq e.symm.toEmbedding G, comap_symm, map_symm]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      G : SimpleGraph β
      e : Equiv α β
      h : (SimpleGraph.comap (⇑e) G).LocallyLinear
      ⊢ (SimpleGraph.map e.toEmbedding (SimpleGraph.comap (⇑e.toEmbedding) G)).Local …
    -/
    exact h.map _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      G : SimpleGraph β
      e : Equiv α β
      ⊢ G.LocallyLinear → (SimpleGraph.comap (⇑e) G).LocallyLinear
    -/
  · rw [← Equiv.coe_toEmbedding, ← map_symm]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      G : SimpleGraph β
      e : Equiv α β
      ⊢ G.LocallyLinear → (SimpleGraph.map e.symm.toEmbedding G).LocallyLinear
    -/
    exact LocallyLinear.map _
    /-
      🎉 no goals
    -/


lemma edgeDisjointTriangles_iff_mem_sym2_subsingleton :
    G.EdgeDisjointTriangles ↔
      ∀ ⦃e : Sym2 α⦄, ¬ e.IsDiag → {s ∈ G.cliqueSet 3 | e ∈ (s : Finset α).sym2}.Subsingleton := by
  classical
  have (a b) (hab : a ≠ b) : {s ∈ (G.cliqueSet 3 : Set (Finset α)) | s(a, b) ∈ (s : Finset α).sym2}
    = {s | G.Adj a b ∧ ∃ c, G.Adj a c ∧ G.Adj b c ∧ s = {a, b, c}} := by
    ext s
    simp only [mem_sym2_iff, Sym2.mem_iff, forall_eq_or_imp, forall_eq, Set.sep_and,
      Set.mem_inter_iff, Set.mem_sep_iff, mem_cliqueSet_iff, Set.mem_setOf_eq,
      and_and_and_comm (b := _ ∈ _), and_self, is3Clique_iff]
    constructor
    · rintro ⟨⟨c, d, e, hcd, hce, hde, rfl⟩, hab⟩
      simp only [mem_insert, mem_singleton] at hab
      obtain ⟨rfl | rfl | rfl, rfl | rfl | rfl⟩ := hab
      any_goals
        simp only [*, adj_comm, true_and, Ne, eq_self_iff_true, not_true] at *
      any_goals
        first
        | exact ⟨c, by aesop⟩
        | exact ⟨d, by aesop⟩
        | exact ⟨e, by aesop⟩
        | simp only [*, adj_comm, true_and, Ne, eq_self_iff_true, not_true] at *
          exact ⟨c, by aesop⟩
        | simp only [*, adj_comm, true_and, Ne, eq_self_iff_true, not_true] at *
          exact ⟨d, by aesop⟩
        | simp only [*, adj_comm, true_and, Ne, eq_self_iff_true, not_true] at *
          exact ⟨e, by aesop⟩
    · rintro ⟨hab, c, hac, hbc, rfl⟩
      refine ⟨⟨a, b, c, ?_⟩, ?_⟩ <;> simp [*]
  constructor
  · rw [Sym2.forall]
    rintro hG a b hab
    simp only [Sym2.isDiag_iff_proj_eq] at hab
    rw [this _ _ (Sym2.mk_isDiag_iff.not.2 hab)]
    rintro _ ⟨hab, c, hac, hbc, rfl⟩ _ ⟨-, d, had, hbd, rfl⟩
    refine hG.eq ?_ ?_ (Set.Nontrivial.not_subsingleton ⟨a, ?_, b, ?_, hab.ne⟩) <;>
      simp [is3Clique_triple_iff, *]
  · simp only [EdgeDisjointTriangles, is3Clique_iff, Set.Pairwise, mem_cliqueSet_iff, Ne,
      forall_exists_index, and_imp, ← Set.not_nontrivial_iff (s := _ ∩ _), not_imp_not,
      Set.Nontrivial, Set.mem_inter_iff, mem_coe]
    rintro hG _ a b c hab hac hbc rfl _ d e f hde hdf hef rfl g hg₁ hg₂ h hh₁ hh₂ hgh
    refine hG (Sym2.mk_isDiag_iff.not.2 hgh) ⟨⟨a, b, c, ?_⟩, by simpa using And.intro hg₁ hh₁⟩
      ⟨⟨d, e, f, ?_⟩, by simpa using And.intro hg₂ hh₂⟩ <;> simp [is3Clique_triple_iff, *]


alias ⟨EdgeDisjointTriangles.mem_sym2_subsingleton, _⟩ :=
  edgeDisjointTriangles_iff_mem_sym2_subsingleton


instance EdgeDisjointTriangles.instDecidable : Decidable G.EdgeDisjointTriangles :=
  decidable_of_iff ((G.cliqueFinset 3 : Set (Finset α)).Pairwise fun x y ↦ (#(x ∩ y) ≤ 1)) <| by
    /-
      α : Type u_1
      β : Type u_2
      𝕜 : Type u_3
      inst✝³ : LinearOrderedField 𝕜
      G H : SimpleGraph α
      ε δ : 𝕜
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      inst✝ : DecidableRel G.Adj
      ⊢ Iff ((↑(G.cliqueFinset 3)).Pairwise fun x y => LE.le (Inter.inter x y).card  …
    -/
    simp only [coe_cliqueFinset, EdgeDisjointTriangles, Finset.card_le_one, ← coe_inter]; rfl
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


instance LocallyLinear.instDecidable : Decidable G.LocallyLinear :=
  inferInstanceAs (Decidable (_ ∧ _))


lemma EdgeDisjointTriangles.card_edgeFinset_le (hG : G.EdgeDisjointTriangles) :
    3 * #(G.cliqueFinset 3) ≤ #G.edgeFinset := by
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    inst✝ : DecidableRel G.Adj
    hG : G.EdgeDisjointTriangles
    ⊢ LE.le (HMul.hMul 3 (G.cliqueFinset 3).card) G.edgeFinset.card
  -/
  rw [mul_comm, ← mul_one #G.edgeFinset]
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    inst✝ : DecidableRel G.Adj
    hG : G.EdgeDisjointTriangles
    ⊢ LE.le (HMul.hMul (G.cliqueFinset 3).card 3) (HMul.hMul G.edgeFinset.card 1)
  -/
  refine card_mul_le_card_mul (fun s e ↦ e ∈ s.sym2) ?_ (fun e he ↦ ?_)
    /-
      case refine_1
      α : Type u_1
      G : SimpleGraph α
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      inst✝ : DecidableRel G.Adj
      hG : G.EdgeDisjointTriangles
      ⊢ ∀ (a : Finset α), Membership.mem (G.cliqueFinset 3) a → LE.le 3 (Finset.bipa …
    -/
  · simp only [is3Clique_iff, mem_cliqueFinset_iff, mem_sym2_iff, forall_exists_index, and_imp]
    /-
      case refine_1
      α : Type u_1
      G : SimpleGraph α
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      inst✝ : DecidableRel G.Adj
      hG : G.EdgeDisjointTriangles
      ⊢ ∀ (a : Finset α) (x x_1 x_2 : α), G.Adj x x_1 → G.Adj x x_2 → G.Adj x_1 x_2  …
    -/
    rintro _ a b c hab hac hbc rfl
    have : #{s(a, b), s(a, c), s(b, c)} = 3 := by
      refine card_eq_three.2 ⟨_, _, _, ?_, ?_, ?_, rfl⟩ <;> simp [hab.ne, hac.ne, hbc.ne]
    /-
      case refine_1
      α : Type u_1
      G : SimpleGraph α
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      inst✝ : DecidableRel G.Adj
      hG : G.EdgeDisjointTriangles
      a b c : α
      hab : G.Adj a b
      hac : G.Adj a c
      hbc : G.Adj b c
      this : Eq (Insert.insert (Sym2.mk { fst := a, snd := b }) (Insert.insert (Sym2 …
      ⊢ LE.le 3 (Finset.bipartiteAbove (fun s e => ∀ (a : α), Membership.mem e a → M …
    -/
    rw [← this]
    /-
      case refine_1
      α : Type u_1
      G : SimpleGraph α
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      inst✝ : DecidableRel G.Adj
      hG : G.EdgeDisjointTriangles
      a b c : α
      hab : G.Adj a b
      hac : G.Adj a c
      hbc : G.Adj b c
      this : Eq (Insert.insert (Sym2.mk { fst := a, snd := b }) (Insert.insert (Sym2 …
      ⊢ LE.le (Insert.insert (Sym2.mk { fst := a, snd := b }) (Insert.insert (Sym2.m …
    -/
    refine card_mono ?_
    /-
      case refine_1
      α : Type u_1
      G : SimpleGraph α
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      inst✝ : DecidableRel G.Adj
      hG : G.EdgeDisjointTriangles
      a b c : α
      hab : G.Adj a b
      hac : G.Adj a c
      hbc : G.Adj b c
      this : Eq (Insert.insert (Sym2.mk { fst := a, snd := b }) (Insert.insert (Sym2 …
      ⊢ LE.le (Insert.insert (Sym2.mk { fst := a, snd := b }) (Insert.insert (Sym2.m …
    -/
    simp [insert_subset, *]
    /-
      🎉 no goals
    -/
  · simpa only [card_le_one, mem_bipartiteBelow, and_imp, Set.Subsingleton, Set.mem_setOf_eq,
      mem_cliqueFinset_iff, mem_cliqueSet_iff]
      using hG.mem_sym2_subsingleton (G.not_isDiag_of_mem_edgeSet <| mem_edgeFinset.1 he)


lemma LocallyLinear.card_edgeFinset (hG : G.LocallyLinear) :
    #G.edgeFinset = 3 * #(G.cliqueFinset 3) := by
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    inst✝ : DecidableRel G.Adj
    hG : G.LocallyLinear
    ⊢ Eq G.edgeFinset.card (HMul.hMul 3 (G.cliqueFinset 3).card)
  -/
  refine hG.edgeDisjointTriangles.card_edgeFinset_le.antisymm' ?_
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    inst✝ : DecidableRel G.Adj
    hG : G.LocallyLinear
    ⊢ LE.le G.edgeFinset.card (HMul.hMul 3 (G.cliqueFinset 3).card)
  -/
  rw [← mul_comm, ← mul_one #_]
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    inst✝ : DecidableRel G.Adj
    hG : G.LocallyLinear
    ⊢ LE.le (HMul.hMul G.edgeFinset.card 1) (HMul.hMul (G.cliqueFinset 3).card 3)
  -/
  refine card_mul_le_card_mul (fun e s ↦ e ∈ s.sym2) ?_ ?_
  · simpa [Sym2.forall, Nat.one_le_iff_ne_zero, -Finset.card_eq_zero, Finset.card_ne_zero,
        Finset.Nonempty]
      using hG.2
  /-
    case refine_2
    α : Type u_1
    G : SimpleGraph α
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    inst✝ : DecidableRel G.Adj
    hG : G.LocallyLinear
    ⊢ ∀ (b : Finset α), Membership.mem (G.cliqueFinset 3) b → LE.le (Finset.bipart …
  -/
  simp only [mem_cliqueFinset_iff, is3Clique_iff, forall_exists_index, and_imp]
  /-
    case refine_2
    α : Type u_1
    G : SimpleGraph α
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    inst✝ : DecidableRel G.Adj
    hG : G.LocallyLinear
    ⊢ ∀ (b : Finset α) (x x_1 x_2 : α), G.Adj x x_1 → G.Adj x x_2 → G.Adj x_1 x_2  …
  -/
  rintro _ a b c hab hac hbc rfl
  calc
    _ ≤ #{s(a, b), s(a, c), s(b, c)} := card_le_card ?_
    _ ≤ 3 := (card_insert_le _ _).trans (succ_le_succ <| (card_insert_le _ _).trans_eq <| by
      rw [card_singleton])
  simp only [subset_iff, Sym2.forall, mem_sym2_iff, le_eq_subset, mem_bipartiteBelow, mem_insert,
    mem_edgeFinset, mem_singleton, and_imp, mem_edgeSet, Sym2.mem_iff, forall_eq_or_imp,
    forall_eq, Quotient.eq, Sym2.rel_iff]
  /-
    case refine_2
    α : Type u_1
    G : SimpleGraph α
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    inst✝ : DecidableRel G.Adj
    hG : G.LocallyLinear
    a b c : α
    hab : G.Adj a b
    hac : G.Adj a c
    hbc : G.Adj b c
    ⊢ ∀ (x y : α), G.Adj x y → Or (Eq x a) (Or (Eq x b) (Eq x c)) → Or (Eq y a) (O …
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
                                                         /-
                                                           🎉 no goals
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
                                                         /-
                                                           🎉 no goals
                                                         -/
  rintro d e hde (rfl | rfl | rfl) (rfl | rfl | rfl) <;> simp [*] at *
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- A simple graph is *`ε`-far from triangle-free* if one must remove at least
`ε * (card α) ^ 2` edges to make it triangle-free. -/
def FarFromTriangleFree : Prop := G.DeleteFar (fun H ↦ H.CliqueFree 3) <| ε * (card α ^ 2 : ℕ)


theorem farFromTriangleFree_iff :
    G.FarFromTriangleFree ε ↔ ∀ ⦃H : SimpleGraph α⦄, [DecidableRel H.Adj] → H ≤ G → H.CliqueFree 3 →
      ε * (card α ^ 2 : ℕ) ≤ #G.edgeFinset - #H.edgeFinset := deleteFar_iff


alias ⟨farFromTriangleFree.le_card_sub_card, _⟩ := farFromTriangleFree_iff


nonrec theorem FarFromTriangleFree.mono (hε : G.FarFromTriangleFree ε) (h : δ ≤ ε) :
                                             /-
                                               α : Type u_1
                                               𝕜 : Type u_3
                                               inst✝² : LinearOrderedField 𝕜
                                               G : SimpleGraph α
                                               ε δ : 𝕜
                                               inst✝¹ : Fintype α
                                               inst✝ : DecidableRel G.Adj
                                               hε : G.FarFromTriangleFree ε
                                               h : LE.le δ ε
                                               ⊢ LE.le (HMul.hMul δ ↑(HPow.hPow (Fintype.card α) 2)) (HMul.hMul ε ↑(HPow.hPow …
                                             -/
    G.FarFromTriangleFree δ := hε.mono <| by gcongr
                                             /-
                                               🎉 no goals
                                             -/


theorem FarFromTriangleFree.cliqueFinset_nonempty' (hH : H ≤ G) (hG : G.FarFromTriangleFree ε)
    (hcard : #G.edgeFinset - #H.edgeFinset < ε * (card α ^ 2 : ℕ)) :
    (H.cliqueFinset 3).Nonempty :=
  nonempty_of_ne_empty <|
    cliqueFinset_eq_empty_iff.not.2 fun hH' => (hG.le_card_sub_card hH hH').not_lt hcard


private lemma farFromTriangleFree_of_disjoint_triangles_aux {tris : Finset (Finset α)}
    (htris : tris ⊆ G.cliqueFinset 3)
    (pd : (tris : Set (Finset α)).Pairwise fun x y ↦ (x ∩ y : Set α).Subsingleton) (hHG : H ≤ G)
    (hH : H.CliqueFree 3) : #tris ≤ #G.edgeFinset - #H.edgeFinset := by
  /-
    α : Type u_1
    G H : SimpleGraph α
    inst✝³ : Fintype α
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableRel H.Adj
    inst✝ : DecidableEq α
    tris : Finset (Finset α)
    htris : HasSubset.Subset tris (G.cliqueFinset 3)
    pd : (↑tris).Pairwise fun x y => (Inter.inter ↑x ↑y).Subsingleton
    hHG : LE.le H G
    hH : H.CliqueFree 3
    ⊢ LE.le tris.card (HSub.hSub G.edgeFinset.card H.edgeFinset.card)
  -/
  rw [← card_sdiff (edgeFinset_mono hHG), ← card_attach]
  /-
    α : Type u_1
    G H : SimpleGraph α
    inst✝³ : Fintype α
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableRel H.Adj
    inst✝ : DecidableEq α
    tris : Finset (Finset α)
    htris : HasSubset.Subset tris (G.cliqueFinset 3)
    pd : (↑tris).Pairwise fun x y => (Inter.inter ↑x ↑y).Subsingleton
    hHG : LE.le H G
    hH : H.CliqueFree 3
    ⊢ LE.le tris.attach.card (SDiff.sdiff G.edgeFinset H.edgeFinset).card
  -/
  by_contra! hG
  have ⦃t⦄ (ht : t ∈ tris) :
    ∃ x y, x ∈ t ∧ y ∈ t ∧ x ≠ y ∧ s(x, y) ∈ G.edgeFinset \ H.edgeFinset := by
    by_contra! h
    refine hH t ?_
    simp only [not_and, mem_sdiff, not_not, mem_edgeFinset, mem_edgeSet] at h
    obtain ⟨x, y, z, xy, xz, yz, rfl⟩ := is3Clique_iff.1 (mem_cliqueFinset_iff.1 <| htris ht)
    rw [is3Clique_triple_iff]
    refine ⟨h _ _ ?_ ?_ xy.ne xy, h _ _ ?_ ?_ xz.ne xz, h _ _ ?_ ?_ yz.ne yz⟩ <;> simp
  /-
    α : Type u_1
    G H : SimpleGraph α
    inst✝³ : Fintype α
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableRel H.Adj
    inst✝ : DecidableEq α
    tris : Finset (Finset α)
    htris : HasSubset.Subset tris (G.cliqueFinset 3)
    pd : (↑tris).Pairwise fun x y => (Inter.inter ↑x ↑y).Subsingleton
    hHG : LE.le H G
    hH : H.CliqueFree 3
    hG : LT.lt (SDiff.sdiff G.edgeFinset H.edgeFinset).card tris.attach.card
    this : ∀ ⦃t : Finset α⦄, Membership.mem tris t → Exists fun x => Exists fun y  …
    ⊢ False
  -/
  choose fx fy hfx hfy hfne fmem using this
  /-
    α : Type u_1
    G H : SimpleGraph α
    inst✝³ : Fintype α
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableRel H.Adj
    inst✝ : DecidableEq α
    tris : Finset (Finset α)
    htris : HasSubset.Subset tris (G.cliqueFinset 3)
    pd : (↑tris).Pairwise fun x y => (Inter.inter ↑x ↑y).Subsingleton
    hHG : LE.le H G
    hH : H.CliqueFree 3
    hG : LT.lt (SDiff.sdiff G.edgeFinset H.edgeFinset).card tris.attach.card
    fx fy : ⦃t : Finset α⦄ → Membership.mem tris t → α
    hfx : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem t (fx ht)
    hfy : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem t (fy ht)
    hfne : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Ne (fx ht) (fy ht)
    fmem : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem (SDiff.sd …
    ⊢ False
  -/
  let f (t : {x // x ∈ tris}) : Sym2 α := s(fx t.2, fy t.2)
  /-
    α : Type u_1
    G H : SimpleGraph α
    inst✝³ : Fintype α
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableRel H.Adj
    inst✝ : DecidableEq α
    tris : Finset (Finset α)
    htris : HasSubset.Subset tris (G.cliqueFinset 3)
    pd : (↑tris).Pairwise fun x y => (Inter.inter ↑x ↑y).Subsingleton
    hHG : LE.le H G
    hH : H.CliqueFree 3
    hG : LT.lt (SDiff.sdiff G.edgeFinset H.edgeFinset).card tris.attach.card
    fx fy : ⦃t : Finset α⦄ → Membership.mem tris t → α
    hfx : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem t (fx ht)
    hfy : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem t (fy ht)
    hfne : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Ne (fx ht) (fy ht)
    fmem : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem (SDiff.sd …
    f : (Subtype fun x => Membership.mem tris x) → Sym2 α := fun t => Sym2.mk { fs …
    ⊢ False
  -/
  have hf (x) (_ : x ∈ tris.attach) : f x ∈ G.edgeFinset \ H.edgeFinset := fmem _
  obtain ⟨⟨t₁, ht₁⟩, -, ⟨t₂, ht₂⟩, -, tne, t : s(_, _) = s(_, _)⟩ :=
    exists_ne_map_eq_of_card_lt_of_maps_to hG hf
  /-
    case intro.mk.intro.intro.mk.intro.intro
    α : Type u_1
    G H : SimpleGraph α
    inst✝³ : Fintype α
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableRel H.Adj
    inst✝ : DecidableEq α
    tris : Finset (Finset α)
    htris : HasSubset.Subset tris (G.cliqueFinset 3)
    pd : (↑tris).Pairwise fun x y => (Inter.inter ↑x ↑y).Subsingleton
    hHG : LE.le H G
    hH : H.CliqueFree 3
    hG : LT.lt (SDiff.sdiff G.edgeFinset H.edgeFinset).card tris.attach.card
    fx fy : ⦃t : Finset α⦄ → Membership.mem tris t → α
    hfx : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem t (fx ht)
    hfy : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem t (fy ht)
    hfne : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Ne (fx ht) (fy ht)
    fmem : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem (SDiff.sd …
    f : (Subtype fun x => Membership.mem tris x) → Sym2 α := fun t => Sym2.mk { fs …
    hf : ∀ (x : Subtype fun x => Membership.mem tris x), Membership.mem tris.attac …
    t₁ : Finset α
    ht₁ : Membership.mem tris t₁
    t₂ : Finset α
    ht₂ : Membership.mem tris t₂
    tne : Ne ⟨t₁, ht₁⟩ ⟨t₂, ht₂⟩
    t : Eq (Sym2.mk { fst := fx ⋯, snd := fy ⋯ }) (Sym2.mk { fst := fx ⋯, snd := f …
    ⊢ False
  -/
  dsimp at t
  /-
    case intro.mk.intro.intro.mk.intro.intro
    α : Type u_1
    G H : SimpleGraph α
    inst✝³ : Fintype α
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableRel H.Adj
    inst✝ : DecidableEq α
    tris : Finset (Finset α)
    htris : HasSubset.Subset tris (G.cliqueFinset 3)
    pd : (↑tris).Pairwise fun x y => (Inter.inter ↑x ↑y).Subsingleton
    hHG : LE.le H G
    hH : H.CliqueFree 3
    hG : LT.lt (SDiff.sdiff G.edgeFinset H.edgeFinset).card tris.attach.card
    fx fy : ⦃t : Finset α⦄ → Membership.mem tris t → α
    hfx : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem t (fx ht)
    hfy : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem t (fy ht)
    hfne : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Ne (fx ht) (fy ht)
    fmem : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem (SDiff.sd …
    f : (Subtype fun x => Membership.mem tris x) → Sym2 α := fun t => Sym2.mk { fs …
    hf : ∀ (x : Subtype fun x => Membership.mem tris x), Membership.mem tris.attac …
    t₁ : Finset α
    ht₁ : Membership.mem tris t₁
    t₂ : Finset α
    ht₂ : Membership.mem tris t₂
    tne : Ne ⟨t₁, ht₁⟩ ⟨t₂, ht₂⟩
    t : Eq (Sym2.mk { fst := fx ⋯, snd := fy ⋯ }) (Sym2.mk { fst := fx ⋯, snd := f …
    ⊢ False
  -/
  have i := pd ht₁ ht₂ (Subtype.val_injective.ne tne)
  /-
    case intro.mk.intro.intro.mk.intro.intro
    α : Type u_1
    G H : SimpleGraph α
    inst✝³ : Fintype α
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableRel H.Adj
    inst✝ : DecidableEq α
    tris : Finset (Finset α)
    htris : HasSubset.Subset tris (G.cliqueFinset 3)
    pd : (↑tris).Pairwise fun x y => (Inter.inter ↑x ↑y).Subsingleton
    hHG : LE.le H G
    hH : H.CliqueFree 3
    hG : LT.lt (SDiff.sdiff G.edgeFinset H.edgeFinset).card tris.attach.card
    fx fy : ⦃t : Finset α⦄ → Membership.mem tris t → α
    hfx : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem t (fx ht)
    hfy : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem t (fy ht)
    hfne : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Ne (fx ht) (fy ht)
    fmem : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem (SDiff.sd …
    f : (Subtype fun x => Membership.mem tris x) → Sym2 α := fun t => Sym2.mk { fs …
    hf : ∀ (x : Subtype fun x => Membership.mem tris x), Membership.mem tris.attac …
    t₁ : Finset α
    ht₁ : Membership.mem tris t₁
    t₂ : Finset α
    ht₂ : Membership.mem tris t₂
    tne : Ne ⟨t₁, ht₁⟩ ⟨t₂, ht₂⟩
    t : Eq (Sym2.mk { fst := fx ⋯, snd := fy ⋯ }) (Sym2.mk { fst := fx ⋯, snd := f …
    i : (fun x y => (Inter.inter ↑x ↑y).Subsingleton) t₁ t₂
    ⊢ False
  -/
  rw [Sym2.eq_iff] at t
  /-
    case intro.mk.intro.intro.mk.intro.intro
    α : Type u_1
    G H : SimpleGraph α
    inst✝³ : Fintype α
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableRel H.Adj
    inst✝ : DecidableEq α
    tris : Finset (Finset α)
    htris : HasSubset.Subset tris (G.cliqueFinset 3)
    pd : (↑tris).Pairwise fun x y => (Inter.inter ↑x ↑y).Subsingleton
    hHG : LE.le H G
    hH : H.CliqueFree 3
    hG : LT.lt (SDiff.sdiff G.edgeFinset H.edgeFinset).card tris.attach.card
    fx fy : ⦃t : Finset α⦄ → Membership.mem tris t → α
    hfx : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem t (fx ht)
    hfy : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem t (fy ht)
    hfne : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Ne (fx ht) (fy ht)
    fmem : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem (SDiff.sd …
    f : (Subtype fun x => Membership.mem tris x) → Sym2 α := fun t => Sym2.mk { fs …
    hf : ∀ (x : Subtype fun x => Membership.mem tris x), Membership.mem tris.attac …
    t₁ : Finset α
    ht₁ : Membership.mem tris t₁
    t₂ : Finset α
    ht₂ : Membership.mem tris t₂
    tne : Ne ⟨t₁, ht₁⟩ ⟨t₂, ht₂⟩
    t : Or (And (Eq (fx ⋯) (fx ⋯)) (Eq (fy ⋯) (fy ⋯))) (And (Eq (fx ⋯) (fy ⋯)) (Eq …
    i : (Inter.inter ↑t₁ ↑t₂).Subsingleton
    ⊢ False
  -/
  obtain t | t := t
    /-
      case intro.mk.intro.intro.mk.intro.intro.inl
      α : Type u_1
      G H : SimpleGraph α
      inst✝³ : Fintype α
      inst✝² : DecidableRel G.Adj
      inst✝¹ : DecidableRel H.Adj
      inst✝ : DecidableEq α
      tris : Finset (Finset α)
      htris : HasSubset.Subset tris (G.cliqueFinset 3)
      pd : (↑tris).Pairwise fun x y => (Inter.inter ↑x ↑y).Subsingleton
      hHG : LE.le H G
      hH : H.CliqueFree 3
      hG : LT.lt (SDiff.sdiff G.edgeFinset H.edgeFinset).card tris.attach.card
      fx fy : ⦃t : Finset α⦄ → Membership.mem tris t → α
      hfx : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem t (fx ht)
      hfy : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem t (fy ht)
      hfne : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Ne (fx ht) (fy ht)
      fmem : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem (SDiff.sd …
      f : (Subtype fun x => Membership.mem tris x) → Sym2 α := fun t => Sym2.mk { fs …
      hf : ∀ (x : Subtype fun x => Membership.mem tris x), Membership.mem tris.attac …
      t₁ : Finset α
      ht₁ : Membership.mem tris t₁
      t₂ : Finset α
      ht₂ : Membership.mem tris t₂
      tne : Ne ⟨t₁, ht₁⟩ ⟨t₂, ht₂⟩
      i : (Inter.inter ↑t₁ ↑t₂).Subsingleton
      t : And (Eq (fx ⋯) (fx ⋯)) (Eq (fy ⋯) (fy ⋯))
      ⊢ False
    -/
  · exact hfne _ (i ⟨hfx ht₁, t.1.symm ▸ hfx ht₂⟩ ⟨hfy ht₁, t.2.symm ▸ hfy ht₂⟩)
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.intro.intro.mk.intro.intro.inr
      α : Type u_1
      G H : SimpleGraph α
      inst✝³ : Fintype α
      inst✝² : DecidableRel G.Adj
      inst✝¹ : DecidableRel H.Adj
      inst✝ : DecidableEq α
      tris : Finset (Finset α)
      htris : HasSubset.Subset tris (G.cliqueFinset 3)
      pd : (↑tris).Pairwise fun x y => (Inter.inter ↑x ↑y).Subsingleton
      hHG : LE.le H G
      hH : H.CliqueFree 3
      hG : LT.lt (SDiff.sdiff G.edgeFinset H.edgeFinset).card tris.attach.card
      fx fy : ⦃t : Finset α⦄ → Membership.mem tris t → α
      hfx : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem t (fx ht)
      hfy : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem t (fy ht)
      hfne : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Ne (fx ht) (fy ht)
      fmem : ∀ ⦃t : Finset α⦄ (ht : Membership.mem tris t), Membership.mem (SDiff.sd …
      f : (Subtype fun x => Membership.mem tris x) → Sym2 α := fun t => Sym2.mk { fs …
      hf : ∀ (x : Subtype fun x => Membership.mem tris x), Membership.mem tris.attac …
      t₁ : Finset α
      ht₁ : Membership.mem tris t₁
      t₂ : Finset α
      ht₂ : Membership.mem tris t₂
      tne : Ne ⟨t₁, ht₁⟩ ⟨t₂, ht₂⟩
      i : (Inter.inter ↑t₁ ↑t₂).Subsingleton
      t : And (Eq (fx ⋯) (fy ⋯)) (Eq (fy ⋯) (fx ⋯))
      ⊢ False
    -/
  · exact hfne _ (i ⟨hfx ht₁, t.1.symm ▸ hfy ht₂⟩ ⟨hfy ht₁, t.2.symm ▸ hfx ht₂⟩)
    /-
      🎉 no goals
    -/


/-- If there are `ε * (card α)^2` disjoint triangles, then the graph is `ε`-far from being
triangle-free. -/
lemma farFromTriangleFree_of_disjoint_triangles (tris : Finset (Finset α))
    (htris : tris ⊆ G.cliqueFinset 3)
    (pd : (tris : Set (Finset α)).Pairwise fun x y ↦ (x ∩ y : Set α).Subsingleton)
    (tris_big : ε * (card α ^ 2 : ℕ) ≤ #tris) :
    G.FarFromTriangleFree ε := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    G : SimpleGraph α
    ε : 𝕜
    inst✝² : Fintype α
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq α
    tris : Finset (Finset α)
    htris : HasSubset.Subset tris (G.cliqueFinset 3)
    pd : (↑tris).Pairwise fun x y => (Inter.inter ↑x ↑y).Subsingleton
    tris_big : LE.le (HMul.hMul ε ↑(HPow.hPow (Fintype.card α) 2)) ↑tris.card
    ⊢ G.FarFromTriangleFree ε
  -/
  rw [farFromTriangleFree_iff]
  /-
    α : Type u_1
    𝕜 : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    G : SimpleGraph α
    ε : 𝕜
    inst✝² : Fintype α
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq α
    tris : Finset (Finset α)
    htris : HasSubset.Subset tris (G.cliqueFinset 3)
    pd : (↑tris).Pairwise fun x y => (Inter.inter ↑x ↑y).Subsingleton
    tris_big : LE.le (HMul.hMul ε ↑(HPow.hPow (Fintype.card α) 2)) ↑tris.card
    ⊢ ∀ ⦃H : SimpleGraph α⦄ [inst : DecidableRel H.Adj], LE.le H G → H.CliqueFree  …
  -/
  intros H _ hG hH
  /-
    α : Type u_1
    𝕜 : Type u_3
    inst✝⁴ : LinearOrderedField 𝕜
    G : SimpleGraph α
    ε : 𝕜
    inst✝³ : Fintype α
    inst✝² : DecidableRel G.Adj
    inst✝¹ : DecidableEq α
    tris : Finset (Finset α)
    htris : HasSubset.Subset tris (G.cliqueFinset 3)
    pd : (↑tris).Pairwise fun x y => (Inter.inter ↑x ↑y).Subsingleton
    tris_big : LE.le (HMul.hMul ε ↑(HPow.hPow (Fintype.card α) 2)) ↑tris.card
    H : SimpleGraph α
    inst✝ : DecidableRel H.Adj
    hG : LE.le H G
    hH : H.CliqueFree 3
    ⊢ LE.le (HMul.hMul ε ↑(HPow.hPow (Fintype.card α) 2)) (HSub.hSub ↑G.edgeFinset …
  -/
  rw [← Nat.cast_sub (card_le_card <| edgeFinset_mono hG)]
  exact tris_big.trans
    (Nat.cast_le.2 <| farFromTriangleFree_of_disjoint_triangles_aux htris pd hG hH)


protected lemma EdgeDisjointTriangles.farFromTriangleFree (hG : G.EdgeDisjointTriangles)
    (tris_big : ε * (card α ^ 2 : ℕ) ≤ #(G.cliqueFinset 3)) :
    G.FarFromTriangleFree ε :=
                                                             /-
                                                               α : Type u_1
                                                               𝕜 : Type u_3
                                                               inst✝³ : LinearOrderedField 𝕜
                                                               G : SimpleGraph α
                                                               ε : 𝕜
                                                               inst✝² : Fintype α
                                                               inst✝¹ : DecidableRel G.Adj
                                                               inst✝ : DecidableEq α
                                                               hG : G.EdgeDisjointTriangles
                                                               tris_big : LE.le (HMul.hMul ε ↑(HPow.hPow (Fintype.card α) 2)) ↑(G.cliqueFinse …
                                                               ⊢ (↑(G.cliqueFinset 3)).Pairwise fun x y => (Inter.inter ↑x ↑y).Subsingleton
                                                             -/
  farFromTriangleFree_of_disjoint_triangles _ Subset.rfl (by simpa using hG) tris_big
                                                             /-
                                                               🎉 no goals
                                                             -/


lemma FarFromTriangleFree.lt_half (hG : G.FarFromTriangleFree ε) : ε < 2⁻¹ := by
  classical
  by_contra! hε
  refine lt_irrefl (ε * card α ^ 2) ?_
  have hε₀ : 0 < ε := hε.trans_lt' (by norm_num)
  rw [inv_le_iff_one_le_mul₀ (zero_lt_two' 𝕜)] at hε
  calc
    _ ≤ (#G.edgeFinset : 𝕜) := by
      simpa using hG.le_card_sub_card bot_le (cliqueFree_bot (le_succ _))
    _ ≤ ε * 2 * #G.edgeFinset := le_mul_of_one_le_left (by positivity) (by assumption)
    _ < ε * card α ^ 2 := ?_
  rw [mul_assoc, mul_lt_mul_left hε₀]
  norm_cast
  calc
    _ ≤ 2 * (⊤ : SimpleGraph α).edgeFinset.card := by gcongr; exact le_top
    _ < card α ^ 2 := ?_
  rw [edgeFinset_top, filter_not, card_sdiff (subset_univ _), card_univ, Sym2.card]
  simp_rw [choose_two_right, Nat.add_sub_cancel, Nat.mul_comm _ (card α),
    funext (propext <| Sym2.isDiag_iff_mem_range_diag ·), univ_filter_mem_range, mul_tsub,
    Nat.mul_div_cancel' (card α).even_mul_succ_self.two_dvd]
  rw [card_image_of_injective _ Sym2.diag_injective, card_univ, mul_add_one (α := ℕ), two_mul, sq,
    add_tsub_add_eq_tsub_right]
  apply tsub_lt_self <;> positivity


lemma FarFromTriangleFree.lt_one (hG : G.FarFromTriangleFree ε) : ε < 1 :=
  hG.lt_half.trans two_inv_lt_one


theorem FarFromTriangleFree.nonpos (h₀ : G.FarFromTriangleFree ε) (h₁ : G.CliqueFree 3) :
    ε ≤ 0 := by
  /-
    α : Type u_1
    𝕜 : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    G : SimpleGraph α
    ε : 𝕜
    inst✝² : Fintype α
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty α
    h₀ : G.FarFromTriangleFree ε
    h₁ : G.CliqueFree 3
    ⊢ LE.le ε 0
  -/
  have := h₀ (empty_subset _)
  /-
    α : Type u_1
    𝕜 : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    G : SimpleGraph α
    ε : 𝕜
    inst✝² : Fintype α
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty α
    h₀ : G.FarFromTriangleFree ε
    h₁ : G.CliqueFree 3
    this : (fun H => H.CliqueFree 3) (G.deleteEdges ↑EmptyCollection.emptyCollecti …
    ⊢ LE.le ε 0
  -/
  rw [coe_empty, Finset.card_empty, cast_zero, deleteEdges_empty] at this
  /-
    α : Type u_1
    𝕜 : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    G : SimpleGraph α
    ε : 𝕜
    inst✝² : Fintype α
    inst✝¹ : DecidableRel G.Adj
    inst✝ : Nonempty α
    h₀ : G.FarFromTriangleFree ε
    h₁ : G.CliqueFree 3
    this : (fun H => H.CliqueFree 3) G → LE.le (HMul.hMul ε ↑(HPow.hPow (Fintype.c …
    ⊢ LE.le ε 0
  -/
  exact nonpos_of_mul_nonpos_left (this h₁) (cast_pos.2 <| sq_pos_of_pos Fintype.card_pos)
  /-
    🎉 no goals
  -/


theorem CliqueFree.not_farFromTriangleFree (hG : G.CliqueFree 3) (hε : 0 < ε) :
    ¬G.FarFromTriangleFree ε := fun h => (h.nonpos hG).not_lt hε


theorem FarFromTriangleFree.not_cliqueFree (hG : G.FarFromTriangleFree ε) (hε : 0 < ε) :
    ¬G.CliqueFree 3 := fun h => (hG.nonpos h).not_lt hε


theorem FarFromTriangleFree.cliqueFinset_nonempty [DecidableEq α]
    (hG : G.FarFromTriangleFree ε) (hε : 0 < ε) : (G.cliqueFinset 3).Nonempty :=
  nonempty_of_ne_empty <| cliqueFinset_eq_empty_iff.not.2 <| hG.not_cliqueFree hε


