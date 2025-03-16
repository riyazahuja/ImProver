/-- A clique in a graph is a set of vertices that are pairwise adjacent. -/
abbrev IsClique (s : Set α) : Prop :=
  s.Pairwise G.Adj


theorem isClique_iff : G.IsClique s ↔ s.Pairwise G.Adj :=
  Iff.rfl


/-- A clique is a set of vertices whose induced graph is complete. -/
theorem isClique_iff_induce_eq : G.IsClique s ↔ G.induce s = ⊤ := by
  /-
    α : Type u_1
    G : SimpleGraph α
    s : Set α
    ⊢ Iff (G.IsClique s) (Eq (SimpleGraph.induce s G) Top.top)
  -/
  rw [isClique_iff]
  /-
    α : Type u_1
    G : SimpleGraph α
    s : Set α
    ⊢ Iff (s.Pairwise G.Adj) (Eq (SimpleGraph.induce s G) Top.top)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      G : SimpleGraph α
      s : Set α
      ⊢ s.Pairwise G.Adj → Eq (SimpleGraph.induce s G) Top.top
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      G : SimpleGraph α
      s : Set α
      h : s.Pairwise G.Adj
      ⊢ Eq (SimpleGraph.induce s G) Top.top
    -/
    ext ⟨v, hv⟩ ⟨w, hw⟩
    /-
      case mp.Adj.h.mk.h.mk.a
      α : Type u_1
      G : SimpleGraph α
      s : Set α
      h : s.Pairwise G.Adj
      v : α
      hv : Membership.mem s v
      w : α
      hw : Membership.mem s w
      ⊢ Iff ((SimpleGraph.induce s G).Adj ⟨v, hv⟩ ⟨w, hw⟩) (Top.top.Adj ⟨v, hv⟩ ⟨w,  …
    -/
    simp only [comap_adj, Subtype.coe_mk, top_adj, Ne, Subtype.mk_eq_mk]
    /-
      case mp.Adj.h.mk.h.mk.a
      α : Type u_1
      G : SimpleGraph α
      s : Set α
      h : s.Pairwise G.Adj
      v : α
      hv : Membership.mem s v
      w : α
      hw : Membership.mem s w
      ⊢ Iff (G.Adj ((Function.Embedding.subtype fun x => Membership.mem s x) ⟨v, hv⟩ …
    -/
    exact ⟨Adj.ne, h hv hw⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      G : SimpleGraph α
      s : Set α
      ⊢ Eq (SimpleGraph.induce s G) Top.top → s.Pairwise G.Adj
    -/
  · intro h v hv w hw hne
    /-
      case mpr
      α : Type u_1
      G : SimpleGraph α
      s : Set α
      h : Eq (SimpleGraph.induce s G) Top.top
      v : α
      hv : Membership.mem s v
      w : α
      hw : Membership.mem s w
      hne : Ne v w
      ⊢ G.Adj v w
    -/
    have h2 : (G.induce s).Adj ⟨v, hv⟩ ⟨w, hw⟩ = _ := rfl
    /-
      case mpr
      α : Type u_1
      G : SimpleGraph α
      s : Set α
      h : Eq (SimpleGraph.induce s G) Top.top
      v : α
      hv : Membership.mem s v
      w : α
      hw : Membership.mem s w
      hne : Ne v w
      h2 : Eq ((SimpleGraph.induce s G).Adj ⟨v, hv⟩ ⟨w, hw⟩) ((SimpleGraph.induce s  …
      ⊢ G.Adj v w
    -/
    conv_lhs at h2 => rw [h]
    /-
      case mpr
      α : Type u_1
      G : SimpleGraph α
      s : Set α
      h : Eq (SimpleGraph.induce s G) Top.top
      v : α
      hv : Membership.mem s v
      w : α
      hw : Membership.mem s w
      hne : Ne v w
      h2 : Eq (Top.top.Adj ⟨v, hv⟩ ⟨w, hw⟩) ((SimpleGraph.induce s G).Adj ⟨v, hv⟩ ⟨w …
      ⊢ G.Adj v w
    -/
    simp only [top_adj, ne_eq, Subtype.mk.injEq, eq_iff_iff] at h2
    /-
      case mpr
      α : Type u_1
      G : SimpleGraph α
      s : Set α
      h : Eq (SimpleGraph.induce s G) Top.top
      v : α
      hv : Membership.mem s v
      w : α
      hw : Membership.mem s w
      hne : Ne v w
      h2 : Iff (Not (Eq v w)) ((SimpleGraph.induce s G).Adj ⟨v, hv⟩ ⟨w, hw⟩)
      ⊢ G.Adj v w
    -/
    exact h2.1 hne
    /-
      🎉 no goals
    -/


instance [DecidableEq α] [DecidableRel G.Adj] {s : Finset α} : Decidable (G.IsClique s) :=
  decidable_of_iff' _ G.isClique_iff


                                          /-
                                            α : Type u_1
                                            G : SimpleGraph α
                                            ⊢ G.IsClique EmptyCollection.emptyCollection
                                          -/
lemma isClique_empty : G.IsClique ∅ := by simp
                                          /-
                                            🎉 no goals
                                          -/


                                                        /-
                                                          α : Type u_1
                                                          G : SimpleGraph α
                                                          a : α
                                                          ⊢ G.IsClique (Singleton.singleton a)
                                                        -/
lemma isClique_singleton (a : α) : G.IsClique {a} := by simp
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem IsClique.of_subsingleton {G : SimpleGraph α} (hs : s.Subsingleton) : G.IsClique s :=
  hs.pairwise G.Adj


lemma isClique_pair : G.IsClique {a, b} ↔ a ≠ b → G.Adj a b := Set.pairwise_pair_of_symmetric G.symm


@[simp]
lemma isClique_insert : G.IsClique (insert a s) ↔ G.IsClique s ∧ ∀ b ∈ s, a ≠ b → G.Adj a b :=
  Set.pairwise_insert_of_symmetric G.symm


lemma isClique_insert_of_not_mem (ha : a ∉ s) :
    G.IsClique (insert a s) ↔ G.IsClique s ∧ ∀ b ∈ s, G.Adj a b :=
  Set.pairwise_insert_of_symmetric_of_not_mem G.symm ha


lemma IsClique.insert (hs : G.IsClique s) (h : ∀ b ∈ s, a ≠ b → G.Adj a b) :
    G.IsClique (insert a s) := hs.insert_of_symmetric G.symm h


theorem IsClique.mono (h : G ≤ H) : G.IsClique s → H.IsClique s := Set.Pairwise.mono' h


theorem IsClique.subset (h : t ⊆ s) : G.IsClique s → G.IsClique t := Set.Pairwise.mono h


@[simp]
theorem isClique_bot_iff : (⊥ : SimpleGraph α).IsClique s ↔ (s : Set α).Subsingleton :=
  Set.pairwise_bot_iff


alias ⟨IsClique.subsingleton, _⟩ := isClique_bot_iff


protected theorem IsClique.map (h : G.IsClique s) {f : α ↪ β} : (G.map f).IsClique (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    s : Set α
    h : G.IsClique s
    f : Function.Embedding α β
    ⊢ (SimpleGraph.map f G).IsClique (Set.image (⇑f) s)
  -/
  rintro _ ⟨a, ha, rfl⟩ _ ⟨b, hb, rfl⟩ hab
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    s : Set α
    h : G.IsClique s
    f : Function.Embedding α β
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : Ne (f a) (f b)
    ⊢ (SimpleGraph.map f G).Adj (f a) (f b)
  -/
  exact ⟨a, b, h ha hb <| ne_of_apply_ne _ hab, rfl, rfl⟩
  /-
    🎉 no goals
  -/


theorem isClique_map_iff_of_nontrivial {f : α ↪ β} {t : Set β} (ht : t.Nontrivial) :
    (G.map f).IsClique t ↔ ∃ (s : Set α), G.IsClique s ∧ f '' s = t := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    t : Set β
    ht : t.Nontrivial
    ⊢ Iff ((SimpleGraph.map f G).IsClique t) (Exists fun s => And (G.IsClique s) ( …
  -/
  refine ⟨fun h ↦ ⟨f ⁻¹' t, ?_, ?_⟩, by rintro ⟨x, hs, rfl⟩; exact hs.map⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      f : Function.Embedding α β
      t : Set β
      ht : t.Nontrivial
      h : (SimpleGraph.map f G).IsClique t
      ⊢ G.IsClique (Set.preimage (⇑f) t)
    -/
  · rintro x (hx : f x ∈ t) y (hy : f y ∈ t) hne
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      f : Function.Embedding α β
      t : Set β
      ht : t.Nontrivial
      h : (SimpleGraph.map f G).IsClique t
      x : α
      hx : Membership.mem t (f x)
      y : α
      hy : Membership.mem t (f y)
      hne : Ne x y
      ⊢ G.Adj x y
    -/
    obtain ⟨u,v, huv, hux, hvy⟩ := h hx hy (by simpa)
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      f : Function.Embedding α β
      t : Set β
      ht : t.Nontrivial
      h : (SimpleGraph.map f G).IsClique t
      x : α
      hx : Membership.mem t (f x)
      y : α
      hy : Membership.mem t (f y)
      hne : Ne x y
      u v : α
      huv : G.Adj u v
      hux : Eq (f u) (f x)
      hvy : Eq (f v) (f y)
      ⊢ G.Adj x y
    -/
    rw [EmbeddingLike.apply_eq_iff_eq] at hux hvy
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      f : Function.Embedding α β
      t : Set β
      ht : t.Nontrivial
      h : (SimpleGraph.map f G).IsClique t
      x : α
      hx : Membership.mem t (f x)
      y : α
      hy : Membership.mem t (f y)
      hne : Ne x y
      u v : α
      huv : G.Adj u v
      hux : Eq u x
      hvy : Eq v y
      ⊢ G.Adj x y
    -/
    rwa [← hux, ← hvy]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    t : Set β
    ht : t.Nontrivial
    h : (SimpleGraph.map f G).IsClique t
    ⊢ Eq (Set.image (⇑f) (Set.preimage (⇑f) t)) t
  -/
  rw [Set.image_preimage_eq_iff]
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    t : Set β
    ht : t.Nontrivial
    h : (SimpleGraph.map f G).IsClique t
    ⊢ HasSubset.Subset t (Set.range ⇑f)
  -/
  intro x hxt
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    t : Set β
    ht : t.Nontrivial
    h : (SimpleGraph.map f G).IsClique t
    x : β
    hxt : Membership.mem t x
    ⊢ Membership.mem (Set.range ⇑f) x
  -/
  obtain ⟨y,hyt, hyne⟩ := ht.exists_ne x
  /-
    case refine_2.intro.intro
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    t : Set β
    ht : t.Nontrivial
    h : (SimpleGraph.map f G).IsClique t
    x : β
    hxt : Membership.mem t x
    y : β
    hyt : Membership.mem t y
    hyne : Ne y x
    ⊢ Membership.mem (Set.range ⇑f) x
  -/
  obtain ⟨u,v, -, rfl, rfl⟩ := h hyt hxt hyne
  /-
    case refine_2.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    t : Set β
    ht : t.Nontrivial
    h : (SimpleGraph.map f G).IsClique t
    u v : α
    hyt : Membership.mem t (f u)
    hxt : Membership.mem t (f v)
    hyne : Ne (f u) (f v)
    ⊢ Membership.mem (Set.range ⇑f) (f v)
  -/
  exact Set.mem_range_self _
  /-
    🎉 no goals
  -/


theorem isClique_map_iff {f : α ↪ β} {t : Set β} :
    (G.map f).IsClique t ↔ t.Subsingleton ∨ ∃ (s : Set α), G.IsClique s ∧ f '' s = t := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    t : Set β
    ⊢ Iff ((SimpleGraph.map f G).IsClique t) (Or t.Subsingleton (Exists fun s => A …
  -/
  obtain (ht | ht) := t.subsingleton_or_nontrivial
    /-
      case inl
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      f : Function.Embedding α β
      t : Set β
      ht : t.Subsingleton
      ⊢ Iff ((SimpleGraph.map f G).IsClique t) (Or t.Subsingleton (Exists fun s => A …
    -/
  · simp [IsClique.of_subsingleton, ht]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    t : Set β
    ht : t.Nontrivial
    ⊢ Iff ((SimpleGraph.map f G).IsClique t) (Or t.Subsingleton (Exists fun s => A …
  -/
  simp [isClique_map_iff_of_nontrivial ht, ht.not_subsingleton]
  /-
    🎉 no goals
  -/


@[simp] theorem isClique_map_image_iff {f : α ↪ β} :
    (G.map f).IsClique (f '' s) ↔ G.IsClique s := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    s : Set α
    f : Function.Embedding α β
    ⊢ Iff ((SimpleGraph.map f G).IsClique (Set.image (⇑f) s)) (G.IsClique s)
  -/
  rw [isClique_map_iff, f.injective.subsingleton_image_iff]
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    s : Set α
    f : Function.Embedding α β
    ⊢ Iff (Or s.Subsingleton (Exists fun s_1 => And (G.IsClique s_1) (Eq (Set.imag …
  -/
  obtain (hs | hs) := s.subsingleton_or_nontrivial
    /-
      case inl
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      s : Set α
      f : Function.Embedding α β
      hs : s.Subsingleton
      ⊢ Iff (Or s.Subsingleton (Exists fun s_1 => And (G.IsClique s_1) (Eq (Set.imag …
    -/
  · simp [hs, IsClique.of_subsingleton]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    s : Set α
    f : Function.Embedding α β
    hs : s.Nontrivial
    ⊢ Iff (Or s.Subsingleton (Exists fun s_1 => And (G.IsClique s_1) (Eq (Set.imag …
  -/
  simp [or_iff_right hs.not_subsingleton, Set.image_eq_image f.injective]
  /-
    🎉 no goals
  -/


theorem isClique_map_finset_iff_of_nontrivial (ht : t.Nontrivial) :
    (G.map f).IsClique t ↔ ∃ (s : Finset α), G.IsClique s ∧ s.map f = t := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    t : Finset β
    ht : t.Nontrivial
    ⊢ Iff ((SimpleGraph.map f G).IsClique ↑t) (Exists fun s => And (G.IsClique ↑s) …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      f : Function.Embedding α β
      t : Finset β
      ht : t.Nontrivial
      ⊢ (SimpleGraph.map f G).IsClique ↑t → Exists fun s => And (G.IsClique ↑s) (Eq  …
    -/
  · rw [isClique_map_iff_of_nontrivial (by simpa)]
    /-
      case mp
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      f : Function.Embedding α β
      t : Finset β
      ht : t.Nontrivial
      ⊢ (Exists fun s => And (G.IsClique s) (Eq (Set.image (⇑f) s) ↑t)) → Exists fun …
    -/
    rintro ⟨s, hs, hst⟩
    obtain ⟨s, rfl⟩ := Set.Finite.exists_finset_coe <|
      (show s.Finite from Set.Finite.of_finite_image (by simp [hst]) f.injective.injOn)
    /-
      case mp.intro.intro.intro
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      f : Function.Embedding α β
      t : Finset β
      ht : t.Nontrivial
      s : Finset α
      hs : G.IsClique ↑s
      hst : Eq (Set.image ⇑f ↑s) ↑t
      ⊢ Exists fun s => And (G.IsClique ↑s) (Eq (Finset.map f s) t)
    -/
    exact ⟨s,hs, Finset.coe_inj.1 (by simpa)⟩
    /-
      🎉 no goals
    -/
  /-
    case mpr
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    t : Finset β
    ht : t.Nontrivial
    ⊢ (Exists fun s => And (G.IsClique ↑s) (Eq (Finset.map f s) t)) → (SimpleGraph …
  -/
  rintro ⟨s, hs, rfl⟩
  /-
    case mpr.intro.intro
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    s : Finset α
    hs : G.IsClique ↑s
    ht : (Finset.map f s).Nontrivial
    ⊢ (SimpleGraph.map f G).IsClique ↑(Finset.map f s)
  -/
  simpa using hs.map (f := f)
  /-
    🎉 no goals
  -/


theorem isClique_map_finset_iff :
    (G.map f).IsClique t ↔ #t ≤ 1 ∨ ∃ (s : Finset α), G.IsClique s ∧ s.map f = t := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    t : Finset β
    ⊢ Iff ((SimpleGraph.map f G).IsClique ↑t) (Or (LE.le t.card 1) (Exists fun s = …
  -/
  obtain (ht | ht) := le_or_lt #t 1
    /-
      case inl
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      f : Function.Embedding α β
      t : Finset β
      ht : LE.le t.card 1
      ⊢ Iff ((SimpleGraph.map f G).IsClique ↑t) (Or (LE.le t.card 1) (Exists fun s = …
    -/
  · simp only [ht, true_or, iff_true]
    /-
      case inl
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      f : Function.Embedding α β
      t : Finset β
      ht : LE.le t.card 1
      ⊢ (SimpleGraph.map f G).IsClique ↑t
    -/
    exact IsClique.of_subsingleton <| card_le_one.1 ht
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    t : Finset β
    ht : LT.lt 1 t.card
    ⊢ Iff ((SimpleGraph.map f G).IsClique ↑t) (Or (LE.le t.card 1) (Exists fun s = …
  -/
  rw [isClique_map_finset_iff_of_nontrivial, ← not_lt]
    /-
      case inr
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      f : Function.Embedding α β
      t : Finset β
      ht : LT.lt 1 t.card
      ⊢ Iff (Exists fun s => And (G.IsClique ↑s) (Eq (Finset.map f s) t)) (Or (Not ( …
    -/
  · simp [ht, Finset.map_eq_image]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    t : Finset β
    ht : LT.lt 1 t.card
    ⊢ t.Nontrivial
  -/
  exact Finset.one_lt_card_iff_nontrivial.mp ht
  /-
    🎉 no goals
  -/


protected theorem IsClique.finsetMap {f : α ↪ β} {s : Finset α} (h : G.IsClique s) :
    (G.map f).IsClique (s.map f) := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    f : Function.Embedding α β
    s : Finset α
    h : G.IsClique ↑s
    ⊢ (SimpleGraph.map f G).IsClique ↑(Finset.map f s)
  -/
  simpa
  /-
    🎉 no goals
  -/


/-- An `n`-clique in a graph is a set of `n` vertices which are pairwise connected. -/
structure IsNClique (n : ℕ) (s : Finset α) : Prop where
  isClique : G.IsClique s
  card_eq : #s = n


theorem isNClique_iff : G.IsNClique n s ↔ G.IsClique s ∧ #s = n :=
  ⟨fun h ↦ ⟨h.1, h.2⟩, fun h ↦ ⟨h.1, h.2⟩⟩


instance [DecidableEq α] [DecidableRel G.Adj] {n : ℕ} {s : Finset α} :
    Decidable (G.IsNClique n s) :=
  decidable_of_iff' _ G.isNClique_iff


                                                              /-
                                                                α : Type u_1
                                                                G : SimpleGraph α
                                                                n : Nat
                                                                ⊢ Iff (G.IsNClique n EmptyCollection.emptyCollection) (Eq n 0)
                                                              -/
@[simp] lemma isNClique_empty : G.IsNClique n ∅ ↔ n = 0 := by simp [isNClique_iff, eq_comm]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
                                                            /-
                                                              α : Type u_1
                                                              G : SimpleGraph α
                                                              n : Nat
                                                              a : α
                                                              ⊢ Iff (G.IsNClique n (Singleton.singleton a)) (Eq n 1)
                                                            -/
lemma isNClique_singleton : G.IsNClique n {a} ↔ n = 1 := by simp [isNClique_iff, eq_comm]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem IsNClique.mono (h : G ≤ H) : G.IsNClique n s → H.IsNClique n s := by
  /-
    α : Type u_1
    G H : SimpleGraph α
    n : Nat
    s : Finset α
    h : LE.le G H
    ⊢ G.IsNClique n s → H.IsNClique n s
  -/
  simp_rw [isNClique_iff]
  /-
    α : Type u_1
    G H : SimpleGraph α
    n : Nat
    s : Finset α
    h : LE.le G H
    ⊢ And (G.IsClique ↑s) (Eq s.card n) → And (H.IsClique ↑s) (Eq s.card n)
  -/
  exact And.imp_left (IsClique.mono h)
  /-
    🎉 no goals
  -/


protected theorem IsNClique.map (h : G.IsNClique n s) {f : α ↪ β} :
    (G.map f).IsNClique n (s.map f) :=
      /-
        α : Type u_1
        β : Type u_2
        G : SimpleGraph α
        n : Nat
        s : Finset α
        h : G.IsNClique n s
        f : Function.Embedding α β
        ⊢ (SimpleGraph.map f G).IsClique ↑(Finset.map f s)
      -/
  ⟨by rw [coe_map]; exact h.1.map, (card_map _).trans h.2⟩
                    /-
                      🎉 no goals
                    -/


theorem isNClique_map_iff (hn : 1 < n) {t : Finset β} {f : α ↪ β} :
    (G.map f).IsNClique n t ↔ ∃ s : Finset α, G.IsNClique n s ∧ s.map f = t := by
  rw [isNClique_iff, isClique_map_finset_iff, or_and_right,
    or_iff_right (by rintro ⟨h', rfl⟩; exact h'.not_lt hn)]
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    n : Nat
    hn : LT.lt 1 n
    t : Finset β
    f : Function.Embedding α β
    ⊢ Iff (And (Exists fun s => And (G.IsClique ↑s) (Eq (Finset.map f s) t)) (Eq t …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      n : Nat
      hn : LT.lt 1 n
      t : Finset β
      f : Function.Embedding α β
      ⊢ And (Exists fun s => And (G.IsClique ↑s) (Eq (Finset.map f s) t)) (Eq t.card …
    -/
  · rintro ⟨⟨s, hs, rfl⟩, rfl⟩
    /-
      case mp.intro.intro.intro
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      f : Function.Embedding α β
      s : Finset α
      hs : G.IsClique ↑s
      hn : LT.lt 1 (Finset.map f s).card
      ⊢ Exists fun s_1 => And (G.IsNClique (Finset.map f s).card s_1) (Eq (Finset.ma …
    -/
    simp [isNClique_iff, hs]
    /-
      🎉 no goals
    -/
  /-
    case mpr
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    n : Nat
    hn : LT.lt 1 n
    t : Finset β
    f : Function.Embedding α β
    ⊢ (Exists fun s => And (G.IsNClique n s) (Eq (Finset.map f s) t)) → And (Exist …
  -/
  rintro ⟨s, hs, rfl⟩
  /-
    case mpr.intro.intro
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    n : Nat
    hn : LT.lt 1 n
    f : Function.Embedding α β
    s : Finset α
    hs : G.IsNClique n s
    ⊢ And (Exists fun s_1 => And (G.IsClique ↑s_1) (Eq (Finset.map f s_1) (Finset. …
  -/
  simp [hs.card_eq, hs.isClique]
  /-
    🎉 no goals
  -/


@[simp]
theorem isNClique_bot_iff : (⊥ : SimpleGraph α).IsNClique n s ↔ n ≤ 1 ∧ #s = n := by
  /-
    α : Type u_1
    n : Nat
    s : Finset α
    ⊢ Iff (Bot.bot.IsNClique n s) (And (LE.le n 1) (Eq s.card n))
  -/
  rw [isNClique_iff, isClique_bot_iff]
  /-
    α : Type u_1
    n : Nat
    s : Finset α
    ⊢ Iff (And (↑s).Subsingleton (Eq s.card n)) (And (LE.le n 1) (Eq s.card n))
  -/
  refine and_congr_left ?_
  /-
    α : Type u_1
    n : Nat
    s : Finset α
    ⊢ Eq s.card n → Iff (↑s).Subsingleton (LE.le n 1)
  -/
  rintro rfl
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (↑s).Subsingleton (LE.le s.card 1)
  -/
  exact card_le_one.symm
  /-
    🎉 no goals
  -/


@[simp]
theorem isNClique_zero : G.IsNClique 0 s ↔ s = ∅ := by
  /-
    α : Type u_1
    G : SimpleGraph α
    s : Finset α
    ⊢ Iff (G.IsNClique 0 s) (Eq s EmptyCollection.emptyCollection)
  -/
  simp only [isNClique_iff, Finset.card_eq_zero, and_iff_right_iff_imp]; rintro rfl; simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp]
theorem isNClique_one : G.IsNClique 1 s ↔ ∃ a, s = {a} := by
  /-
    α : Type u_1
    G : SimpleGraph α
    s : Finset α
    ⊢ Iff (G.IsNClique 1 s) (Exists fun a => Eq s (Singleton.singleton a))
  -/
  simp only [isNClique_iff, card_eq_one, and_iff_right_iff_imp]; rintro ⟨a, rfl⟩; simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem IsNClique.insert (hs : G.IsNClique n s) (h : ∀ b ∈ s, G.Adj a b) :
    G.IsNClique (n + 1) (insert a s) := by
  /-
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    s : Finset α
    a : α
    inst✝ : DecidableEq α
    hs : G.IsNClique n s
    h : ∀ (b : α), Membership.mem s b → G.Adj a b
    ⊢ G.IsNClique (HAdd.hAdd n 1) (Insert.insert a s)
  -/
  constructor
    /-
      case isClique
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      s : Finset α
      a : α
      inst✝ : DecidableEq α
      hs : G.IsNClique n s
      h : ∀ (b : α), Membership.mem s b → G.Adj a b
      ⊢ G.IsClique ↑(Insert.insert a s)
    -/
  · push_cast
    /-
      case isClique
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      s : Finset α
      a : α
      inst✝ : DecidableEq α
      hs : G.IsNClique n s
      h : ∀ (b : α), Membership.mem s b → G.Adj a b
      ⊢ G.IsClique (Insert.insert a ↑s)
    -/
    exact hs.1.insert fun b hb _ => h _ hb
    /-
      🎉 no goals
    -/
    /-
      case card_eq
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      s : Finset α
      a : α
      inst✝ : DecidableEq α
      hs : G.IsNClique n s
      h : ∀ (b : α), Membership.mem s b → G.Adj a b
      ⊢ Eq (Insert.insert a s).card (HAdd.hAdd n 1)
    -/
  · rw [card_insert_of_not_mem fun ha => (h _ ha).ne rfl, hs.2]
    /-
      🎉 no goals
    -/


theorem is3Clique_triple_iff : G.IsNClique 3 {a, b, c} ↔ G.Adj a b ∧ G.Adj a c ∧ G.Adj b c := by
  /-
    α : Type u_1
    G : SimpleGraph α
    a b c : α
    inst✝ : DecidableEq α
    ⊢ Iff (G.IsNClique 3 (Insert.insert a (Insert.insert b (Singleton.singleton c) …
  -/
  simp only [isNClique_iff, isClique_iff, Set.pairwise_insert_of_symmetric G.symm, coe_insert]
  /-
    α : Type u_1
    G : SimpleGraph α
    a b c : α
    inst✝ : DecidableEq α
    ⊢ Iff (And (And (And ((↑(Singleton.singleton c)).Pairwise G.Adj) (∀ (b_1 : α), …
  -/
  by_cases hab : a = b <;> by_cases hbc : b = c <;> by_cases hac : a = c <;> subst_vars <;>
    /-
      case pos
      α : Type u_1
      G : SimpleGraph α
      c : α
      inst✝ : DecidableEq α
      hac : Eq c c
      ⊢ Iff (And (And (And ((↑(Singleton.singleton c)).Pairwise G.Adj) (∀ (b : α), M …
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
    simp [G.ne_of_adj, and_rotate, *]
    /-
      🎉 no goals
    -/


theorem is3Clique_iff :
    G.IsNClique 3 s ↔ ∃ a b c, G.Adj a b ∧ G.Adj a c ∧ G.Adj b c ∧ s = {a, b, c} := by
  /-
    α : Type u_1
    G : SimpleGraph α
    s : Finset α
    inst✝ : DecidableEq α
    ⊢ Iff (G.IsNClique 3 s) (Exists fun a => Exists fun b => Exists fun c => And ( …
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      G : SimpleGraph α
      s : Finset α
      inst✝ : DecidableEq α
      h : G.IsNClique 3 s
      ⊢ Exists fun a => Exists fun b => Exists fun c => And (G.Adj a b) (And (G.Adj  …
    -/
  · obtain ⟨a, b, c, -, -, -, hs⟩ := card_eq_three.1 h.card_eq
    /-
      case refine_1.intro.intro.intro.intro.intro.intro
      α : Type u_1
      G : SimpleGraph α
      s : Finset α
      inst✝ : DecidableEq α
      h : G.IsNClique 3 s
      a b c : α
      hs : Eq s (Insert.insert a (Insert.insert b (Singleton.singleton c)))
      ⊢ Exists fun a => Exists fun b => Exists fun c => And (G.Adj a b) (And (G.Adj  …
    -/
    refine ⟨a, b, c, ?_⟩
    /-
      case refine_1.intro.intro.intro.intro.intro.intro
      α : Type u_1
      G : SimpleGraph α
      s : Finset α
      inst✝ : DecidableEq α
      h : G.IsNClique 3 s
      a b c : α
      hs : Eq s (Insert.insert a (Insert.insert b (Singleton.singleton c)))
      ⊢ And (G.Adj a b) (And (G.Adj a c) (And (G.Adj b c) (Eq s (Insert.insert a (In …
    -/
    rwa [hs, eq_self_iff_true, and_true, is3Clique_triple_iff.symm, ← hs]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      G : SimpleGraph α
      s : Finset α
      inst✝ : DecidableEq α
      ⊢ (Exists fun a => Exists fun b => Exists fun c => And (G.Adj a b) (And (G.Adj …
    -/
  · rintro ⟨a, b, c, hab, hbc, hca, rfl⟩
    /-
      case refine_2.intro.intro.intro.intro.intro.intro
      α : Type u_1
      G : SimpleGraph α
      inst✝ : DecidableEq α
      a b c : α
      hab : G.Adj a b
      hbc : G.Adj a c
      hca : G.Adj b c
      ⊢ G.IsNClique 3 (Insert.insert a (Insert.insert b (Singleton.singleton c)))
    -/
    exact is3Clique_triple_iff.2 ⟨hab, hbc, hca⟩
    /-
      🎉 no goals
    -/


theorem is3Clique_iff_exists_cycle_length_three :
    (∃ s : Finset α, G.IsNClique 3 s) ↔ ∃ (u : α) (w : G.Walk u u), w.IsCycle ∧ w.length = 3 := by
  classical
  simp_rw [is3Clique_iff, isCycle_def]
  exact
    ⟨(fun ⟨_, a, _, _, hab, hac, hbc, _⟩ => ⟨a, cons hab (cons hbc (cons hac.symm nil)), by aesop⟩),
    (fun ⟨_, .cons hab (.cons hbc (.cons hca nil)), _, _⟩ => ⟨_, _, _, _, hab, hca.symm, hbc, rfl⟩)⟩


/-- `G.CliqueFree n` means that `G` has no `n`-cliques. -/
def CliqueFree (n : ℕ) : Prop :=
  ∀ t, ¬G.IsNClique n t


theorem IsNClique.not_cliqueFree (hG : G.IsNClique n s) : ¬G.CliqueFree n :=
  fun h ↦ h _ hG


theorem not_cliqueFree_of_top_embedding {n : ℕ} (f : (⊤ : SimpleGraph (Fin n)) ↪g G) :
    ¬G.CliqueFree n := by
  /-
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    f : Top.top.Embedding G
    ⊢ Not (G.CliqueFree n)
  -/
  simp only [CliqueFree, isNClique_iff, isClique_iff_induce_eq, not_forall, Classical.not_not]
  /-
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    f : Top.top.Embedding G
    ⊢ Exists fun x => And (Eq (SimpleGraph.induce (↑x) G) Top.top) (Eq x.card n)
  -/
  use Finset.univ.map f.toEmbedding
  /-
    case h
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    f : Top.top.Embedding G
    ⊢ And (Eq (SimpleGraph.induce (↑(Finset.map f.toEmbedding Finset.univ)) G) Top …
  -/
  simp only [card_map, Finset.card_fin, eq_self_iff_true, and_true]
  /-
    case h
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    f : Top.top.Embedding G
    ⊢ Eq (SimpleGraph.induce (↑(Finset.map f.toEmbedding Finset.univ)) G) Top.top
  -/
  ext ⟨v, hv⟩ ⟨w, hw⟩
  /-
    case h.Adj.h.mk.h.mk.a
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    f : Top.top.Embedding G
    v : α
    hv : Membership.mem (↑(Finset.map f.toEmbedding Finset.univ)) v
    w : α
    hw : Membership.mem (↑(Finset.map f.toEmbedding Finset.univ)) w
    ⊢ Iff ((SimpleGraph.induce (↑(Finset.map f.toEmbedding Finset.univ)) G).Adj ⟨v …
  -/
  simp only [coe_map, Set.mem_image, coe_univ, Set.mem_univ, true_and] at hv hw
  /-
    case h.Adj.h.mk.h.mk.a
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    f : Top.top.Embedding G
    v : α
    hv✝ : Membership.mem (↑(Finset.map f.toEmbedding Finset.univ)) v
    w : α
    hw✝ : Membership.mem (↑(Finset.map f.toEmbedding Finset.univ)) w
    hv : Exists fun x => Eq (f.toEmbedding x) v
    hw : Exists fun x => Eq (f.toEmbedding x) w
    ⊢ Iff ((SimpleGraph.induce (↑(Finset.map f.toEmbedding Finset.univ)) G).Adj ⟨v …
  -/
  obtain ⟨v', rfl⟩ := hv
  /-
    case h.Adj.h.mk.h.mk.a.intro
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    f : Top.top.Embedding G
    w : α
    hw✝ : Membership.mem (↑(Finset.map f.toEmbedding Finset.univ)) w
    hw : Exists fun x => Eq (f.toEmbedding x) w
    v' : Fin n
    hv : Membership.mem (↑(Finset.map f.toEmbedding Finset.univ)) (f.toEmbedding v')
    ⊢ Iff ((SimpleGraph.induce (↑(Finset.map f.toEmbedding Finset.univ)) G).Adj ⟨f …
  -/
  obtain ⟨w', rfl⟩ := hw
  simp only [coe_sort_coe, RelEmbedding.coe_toEmbedding, comap_adj, Function.Embedding.coe_subtype,
    f.map_adj_iff, top_adj, ne_eq, Subtype.mk.injEq, RelEmbedding.inj]
  -- This used to be the end of the proof before https://github.com/leanprover/lean4/pull/2644
  /-
    case h.Adj.h.mk.h.mk.a.intro.intro
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    f : Top.top.Embedding G
    v' : Fin n
    hv : Membership.mem (↑(Finset.map f.toEmbedding Finset.univ)) (f.toEmbedding v')
    w' : Fin n
    hw : Membership.mem (↑(Finset.map f.toEmbedding Finset.univ)) (f.toEmbedding w')
    ⊢ Iff (G.Adj ((Function.Embedding.subtype fun x => Membership.mem (↑(Finset.ma …
  -/
  erw [Function.Embedding.coe_subtype, f.map_adj_iff]
  /-
    case h.Adj.h.mk.h.mk.a.intro.intro
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    f : Top.top.Embedding G
    v' : Fin n
    hv : Membership.mem (↑(Finset.map f.toEmbedding Finset.univ)) (f.toEmbedding v')
    w' : Fin n
    hw : Membership.mem (↑(Finset.map f.toEmbedding Finset.univ)) (f.toEmbedding w')
    ⊢ Iff (Top.top.Adj v' w') (Not (Eq v' w'))
  -/
  simp
  /-
    🎉 no goals
  -/


/-- An embedding of a complete graph that witnesses the fact that the graph is not clique-free. -/
noncomputable def topEmbeddingOfNotCliqueFree {n : ℕ} (h : ¬G.CliqueFree n) :
    (⊤ : SimpleGraph (Fin n)) ↪g G := by
  /-
    α : Type u_1
    β : Type u_2
    G H : SimpleGraph α
    m n✝ : Nat
    s : Finset α
    n : Nat
    h : Not (G.CliqueFree n)
    ⊢ Top.top.Embedding G
  -/
  simp only [CliqueFree, isNClique_iff, isClique_iff_induce_eq, not_forall, Classical.not_not] at h
  /-
    α : Type u_1
    β : Type u_2
    G H : SimpleGraph α
    m n✝ : Nat
    s : Finset α
    n : Nat
    h : Exists fun x => And (Eq (SimpleGraph.induce (↑x) G) Top.top) (Eq x.card n)
    ⊢ Top.top.Embedding G
  -/
  obtain ⟨ha, hb⟩ := h.choose_spec
  have : (⊤ : SimpleGraph (Fin #h.choose)) ≃g (⊤ : SimpleGraph h.choose) := by
    apply Iso.completeGraph
    simpa using (Fintype.equivFin h.choose).symm
  /-
    case intro
    α : Type u_1
    β : Type u_2
    G H : SimpleGraph α
    m n✝ : Nat
    s : Finset α
    n : Nat
    h : Exists fun x => And (Eq (SimpleGraph.induce (↑x) G) Top.top) (Eq x.card n)
    ha : Eq (SimpleGraph.induce (↑h.choose) G) Top.top
    hb : Eq h.choose.card n
    this : Top.top.Iso Top.top
    ⊢ Top.top.Embedding G
  -/
  rw [← ha] at this
  /-
    case intro
    α : Type u_1
    β : Type u_2
    G H : SimpleGraph α
    m n✝ : Nat
    s : Finset α
    n : Nat
    h : Exists fun x => And (Eq (SimpleGraph.induce (↑x) G) Top.top) (Eq x.card n)
    ha : Eq (SimpleGraph.induce (↑h.choose) G) Top.top
    hb : Eq h.choose.card n
    this : Top.top.Iso (SimpleGraph.induce (↑h.choose) G)
    ⊢ Top.top.Embedding G
  -/
  convert (Embedding.induce ↑h.choose.toSet).comp this.toEmbedding
  /-
    case h.e'_1.h.e'_1
    α : Type u_1
    β : Type u_2
    G H : SimpleGraph α
    m n✝ : Nat
    s : Finset α
    n : Nat
    h : Exists fun x => And (Eq (SimpleGraph.induce (↑x) G) Top.top) (Eq x.card n)
    ha : Eq (SimpleGraph.induce (↑h.choose) G) Top.top
    hb : Eq h.choose.card n
    this : Top.top.Iso (SimpleGraph.induce (↑h.choose) G)
    ⊢ Eq n h.choose.card
  -/
  exact hb.symm
  /-
    🎉 no goals
  -/


theorem not_cliqueFree_iff (n : ℕ) : ¬G.CliqueFree n ↔ Nonempty ((⊤ : SimpleGraph (Fin n)) ↪g G) :=
  ⟨fun h ↦ ⟨topEmbeddingOfNotCliqueFree h⟩, fun ⟨f⟩ ↦ not_cliqueFree_of_top_embedding f⟩


theorem cliqueFree_iff {n : ℕ} : G.CliqueFree n ↔ IsEmpty ((⊤ : SimpleGraph (Fin n)) ↪g G) := by
  /-
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    ⊢ Iff (G.CliqueFree n) (IsEmpty (Top.top.Embedding G))
  -/
  rw [← not_iff_not, not_cliqueFree_iff, not_isEmpty_iff]
  /-
    🎉 no goals
  -/


theorem not_cliqueFree_card_of_top_embedding [Fintype α] (f : (⊤ : SimpleGraph α) ↪g G) :
    ¬G.CliqueFree (card α) := by
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Fintype α
    f : Top.top.Embedding G
    ⊢ Not (G.CliqueFree (Fintype.card α))
  -/
  rw [not_cliqueFree_iff]
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Fintype α
    f : Top.top.Embedding G
    ⊢ Nonempty (Top.top.Embedding G)
  -/
  exact ⟨(Iso.completeGraph (Fintype.equivFin α)).symm.toEmbedding.trans f⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem cliqueFree_bot (h : 2 ≤ n) : (⊥ : SimpleGraph α).CliqueFree n := by
  /-
    α : Type u_1
    n : Nat
    h : LE.le 2 n
    ⊢ Bot.bot.CliqueFree n
  -/
  intro t ht
  /-
    α : Type u_1
    n : Nat
    h : LE.le 2 n
    t : Finset α
    ht : Bot.bot.IsNClique n t
    ⊢ False
  -/
  have := le_trans h (isNClique_bot_iff.1 ht).1
  /-
    α : Type u_1
    n : Nat
    h : LE.le 2 n
    t : Finset α
    ht : Bot.bot.IsNClique n t
    this : LE.le 2 1
    ⊢ False
  -/
  contradiction
  /-
    🎉 no goals
  -/


theorem CliqueFree.mono (h : m ≤ n) : G.CliqueFree m → G.CliqueFree n := by
  /-
    α : Type u_1
    G : SimpleGraph α
    m n : Nat
    h : LE.le m n
    ⊢ G.CliqueFree m → G.CliqueFree n
  -/
  intro hG s hs
  /-
    α : Type u_1
    G : SimpleGraph α
    m n : Nat
    h : LE.le m n
    hG : G.CliqueFree m
    s : Finset α
    hs : G.IsNClique n s
    ⊢ False
  -/
  obtain ⟨t, hts, ht⟩ := exists_subset_card_eq (h.trans hs.card_eq.ge)
  /-
    case intro.intro
    α : Type u_1
    G : SimpleGraph α
    m n : Nat
    h : LE.le m n
    hG : G.CliqueFree m
    s : Finset α
    hs : G.IsNClique n s
    t : Finset α
    hts : HasSubset.Subset t s
    ht : Eq t.card m
    ⊢ False
  -/
  exact hG _ ⟨hs.isClique.subset hts, ht⟩
  /-
    🎉 no goals
  -/


theorem CliqueFree.anti (h : G ≤ H) : H.CliqueFree n → G.CliqueFree n :=
  forall_imp fun _ ↦ mt <| IsNClique.mono h


/-- If a graph is cliquefree, any graph that embeds into it is also cliquefree. -/
theorem CliqueFree.comap {H : SimpleGraph β} (f : H ↪g G) : G.CliqueFree n → H.CliqueFree n := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    n : Nat
    H : SimpleGraph β
    f : H.Embedding G
    ⊢ G.CliqueFree n → H.CliqueFree n
  -/
  intro h; contrapose h
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    n : Nat
    H : SimpleGraph β
    f : H.Embedding G
    h : Not (H.CliqueFree n)
    ⊢ Not (G.CliqueFree n)
  -/
  exact not_cliqueFree_of_top_embedding <| f.comp (topEmbeddingOfNotCliqueFree h)
  /-
    🎉 no goals
  -/


@[simp] theorem cliqueFree_map_iff {f : α ↪ β} [Nonempty α] :
    (G.map f).CliqueFree n ↔ G.CliqueFree n := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    n : Nat
    f : Function.Embedding α β
    inst✝ : Nonempty α
    ⊢ Iff ((SimpleGraph.map f G).CliqueFree n) (G.CliqueFree n)
  -/
  obtain (hle | hlt) := le_or_lt n 1
    /-
      case inl
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      n : Nat
      f : Function.Embedding α β
      inst✝ : Nonempty α
      hle : LE.le n 1
      ⊢ Iff ((SimpleGraph.map f G).CliqueFree n) (G.CliqueFree n)
    -/
  · obtain (rfl | rfl) := Nat.le_one_iff_eq_zero_or_eq_one.1 hle
      /-
        case inl.inl
        α : Type u_1
        β : Type u_2
        G : SimpleGraph α
        f : Function.Embedding α β
        inst✝ : Nonempty α
        hle : LE.le 0 1
        ⊢ Iff ((SimpleGraph.map f G).CliqueFree 0) (G.CliqueFree 0)
      -/
    · simp [CliqueFree]
      /-
        🎉 no goals
      -/
    /-
      case inl.inr
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      f : Function.Embedding α β
      inst✝ : Nonempty α
      hle : LE.le 1 1
      ⊢ Iff ((SimpleGraph.map f G).CliqueFree 1) (G.CliqueFree 1)
    -/
    simp [CliqueFree, show ∃ (_ : β), True from ⟨f (Classical.arbitrary _), trivial⟩]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    n : Nat
    f : Function.Embedding α β
    inst✝ : Nonempty α
    hlt : LT.lt 1 n
    ⊢ Iff ((SimpleGraph.map f G).CliqueFree n) (G.CliqueFree n)
  -/
  simp [CliqueFree, isNClique_map_iff hlt]
  /-
    🎉 no goals
  -/


/-- See `SimpleGraph.cliqueFree_of_chromaticNumber_lt` for a tighter bound. -/
theorem cliqueFree_of_card_lt [Fintype α] (hc : card α < n) : G.CliqueFree n := by
  /-
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    inst✝ : Fintype α
    hc : LT.lt (Fintype.card α) n
    ⊢ G.CliqueFree n
  -/
  by_contra h
  /-
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    inst✝ : Fintype α
    hc : LT.lt (Fintype.card α) n
    h : Not (G.CliqueFree n)
    ⊢ False
  -/
  refine Nat.lt_le_asymm hc ?_
  /-
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    inst✝ : Fintype α
    hc : LT.lt (Fintype.card α) n
    h : Not (G.CliqueFree n)
    ⊢ LE.le n (Fintype.card α)
  -/
  rw [cliqueFree_iff, not_isEmpty_iff] at h
  /-
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    inst✝ : Fintype α
    hc : LT.lt (Fintype.card α) n
    h : Nonempty (Top.top.Embedding G)
    ⊢ LE.le n (Fintype.card α)
  -/
  simpa only [Fintype.card_fin] using Fintype.card_le_of_embedding h.some.toEmbedding
  /-
    🎉 no goals
  -/


/-- A complete `r`-partite graph has no `n`-cliques for `r < n`. -/
theorem cliqueFree_completeMultipartiteGraph {ι : Type*} [Fintype ι] (V : ι → Type*)
    (hc : card ι < n) : (completeMultipartiteGraph V).CliqueFree n := by
  /-
    n : Nat
    ι : Type u_3
    inst✝ : Fintype ι
    V : ι → Type u_4
    hc : LT.lt (Fintype.card ι) n
    ⊢ (SimpleGraph.completeMultipartiteGraph V).CliqueFree n
  -/
  rw [cliqueFree_iff, isEmpty_iff]
  /-
    n : Nat
    ι : Type u_3
    inst✝ : Fintype ι
    V : ι → Type u_4
    hc : LT.lt (Fintype.card ι) n
    ⊢ Top.top.Embedding (SimpleGraph.completeMultipartiteGraph V) → False
  -/
  intro f
  /-
    n : Nat
    ι : Type u_3
    inst✝ : Fintype ι
    V : ι → Type u_4
    hc : LT.lt (Fintype.card ι) n
    f : Top.top.Embedding (SimpleGraph.completeMultipartiteGraph V)
    ⊢ False
  -/
  obtain ⟨v, w, hn, he⟩ := exists_ne_map_eq_of_card_lt (Sigma.fst ∘ f) (by simp [hc])
  /-
    case intro.intro.intro
    n : Nat
    ι : Type u_3
    inst✝ : Fintype ι
    V : ι → Type u_4
    hc : LT.lt (Fintype.card ι) n
    f : Top.top.Embedding (SimpleGraph.completeMultipartiteGraph V)
    v w : Fin n
    hn : Ne v w
    he : Eq (Function.comp Sigma.fst (⇑f) v) (Function.comp Sigma.fst (⇑f) w)
    ⊢ False
  -/
  rw [← top_adj, ← f.map_adj_iff, comap_adj, top_adj] at hn
  /-
    case intro.intro.intro
    n : Nat
    ι : Type u_3
    inst✝ : Fintype ι
    V : ι → Type u_4
    hc : LT.lt (Fintype.card ι) n
    f : Top.top.Embedding (SimpleGraph.completeMultipartiteGraph V)
    v w : Fin n
    hn : Ne (f v).fst (f w).fst
    he : Eq (Function.comp Sigma.fst (⇑f) v) (Function.comp Sigma.fst (⇑f) w)
    ⊢ False
  -/
  exact absurd he hn
  /-
    🎉 no goals
  -/


/-- Clique-freeness is preserved by `replaceVertex`. -/
protected theorem CliqueFree.replaceVertex [DecidableEq α] (h : G.CliqueFree n) (s t : α) :
    (G.replaceVertex s t).CliqueFree n := by
  /-
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    inst✝ : DecidableEq α
    h : G.CliqueFree n
    s t : α
    ⊢ (G.replaceVertex s t).CliqueFree n
  -/
  contrapose h
  /-
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    inst✝ : DecidableEq α
    s t : α
    h : Not ((G.replaceVertex s t).CliqueFree n)
    ⊢ Not (G.CliqueFree n)
  -/
  obtain ⟨φ, hφ⟩ := topEmbeddingOfNotCliqueFree h
  /-
    case mk
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    inst✝ : DecidableEq α
    s t : α
    h : Not ((G.replaceVertex s t).CliqueFree n)
    φ : Function.Embedding (Fin n) α
    hφ : ∀ {a b : Fin n}, Iff ((G.replaceVertex s t).Adj (φ a) (φ b)) (Top.top.Adj …
    ⊢ Not (G.CliqueFree n)
  -/
  rw [not_cliqueFree_iff]
  /-
    case mk
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    inst✝ : DecidableEq α
    s t : α
    h : Not ((G.replaceVertex s t).CliqueFree n)
    φ : Function.Embedding (Fin n) α
    hφ : ∀ {a b : Fin n}, Iff ((G.replaceVertex s t).Adj (φ a) (φ b)) (Top.top.Adj …
    ⊢ Nonempty (Top.top.Embedding G)
  -/
  by_cases mt : t ∈ Set.range φ
    /-
      case pos
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      inst✝ : DecidableEq α
      s t : α
      h : Not ((G.replaceVertex s t).CliqueFree n)
      φ : Function.Embedding (Fin n) α
      hφ : ∀ {a b : Fin n}, Iff ((G.replaceVertex s t).Adj (φ a) (φ b)) (Top.top.Adj …
      mt : Membership.mem (Set.range ⇑φ) t
      ⊢ Nonempty (Top.top.Embedding G)
    -/
  · obtain ⟨x, hx⟩ := mt
    /-
      case pos.intro
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      inst✝ : DecidableEq α
      s t : α
      h : Not ((G.replaceVertex s t).CliqueFree n)
      φ : Function.Embedding (Fin n) α
      hφ : ∀ {a b : Fin n}, Iff ((G.replaceVertex s t).Adj (φ a) (φ b)) (Top.top.Adj …
      x : Fin n
      hx : Eq (φ x) t
      ⊢ Nonempty (Top.top.Embedding G)
    -/
    by_cases ms : s ∈ Set.range φ
      /-
        case pos
        α : Type u_1
        G : SimpleGraph α
        n : Nat
        inst✝ : DecidableEq α
        s t : α
        h : Not ((G.replaceVertex s t).CliqueFree n)
        φ : Function.Embedding (Fin n) α
        hφ : ∀ {a b : Fin n}, Iff ((G.replaceVertex s t).Adj (φ a) (φ b)) (Top.top.Adj …
        x : Fin n
        hx : Eq (φ x) t
        ms : Membership.mem (Set.range ⇑φ) s
        ⊢ Nonempty (Top.top.Embedding G)
      -/
    · obtain ⟨y, hy⟩ := ms
      /-
        case pos.intro
        α : Type u_1
        G : SimpleGraph α
        n : Nat
        inst✝ : DecidableEq α
        s t : α
        h : Not ((G.replaceVertex s t).CliqueFree n)
        φ : Function.Embedding (Fin n) α
        hφ : ∀ {a b : Fin n}, Iff ((G.replaceVertex s t).Adj (φ a) (φ b)) (Top.top.Adj …
        x : Fin n
        hx : Eq (φ x) t
        y : Fin n
        hy : Eq (φ y) s
        ⊢ Nonempty (Top.top.Embedding G)
      -/
      have e := @hφ x y
      /-
        case pos.intro
        α : Type u_1
        G : SimpleGraph α
        n : Nat
        inst✝ : DecidableEq α
        s t : α
        h : Not ((G.replaceVertex s t).CliqueFree n)
        φ : Function.Embedding (Fin n) α
        hφ : ∀ {a b : Fin n}, Iff ((G.replaceVertex s t).Adj (φ a) (φ b)) (Top.top.Adj …
        x : Fin n
        hx : Eq (φ x) t
        y : Fin n
        hy : Eq (φ y) s
        e : Iff ((G.replaceVertex s t).Adj (φ x) (φ y)) (Top.top.Adj x y)
        ⊢ Nonempty (Top.top.Embedding G)
      -/
      simp_rw [hx, hy, adj_comm, not_adj_replaceVertex_same, top_adj, false_iff, not_ne_iff] at e
      /-
        case pos.intro
        α : Type u_1
        G : SimpleGraph α
        n : Nat
        inst✝ : DecidableEq α
        s t : α
        h : Not ((G.replaceVertex s t).CliqueFree n)
        φ : Function.Embedding (Fin n) α
        hφ : ∀ {a b : Fin n}, Iff ((G.replaceVertex s t).Adj (φ a) (φ b)) (Top.top.Adj …
        x : Fin n
        hx : Eq (φ x) t
        y : Fin n
        hy : Eq (φ y) s
        e : Eq x y
        ⊢ Nonempty (Top.top.Embedding G)
      -/
      rwa [← hx, e, hy, replaceVertex_self, not_cliqueFree_iff] at h
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        G : SimpleGraph α
        n : Nat
        inst✝ : DecidableEq α
        s t : α
        h : Not ((G.replaceVertex s t).CliqueFree n)
        φ : Function.Embedding (Fin n) α
        hφ : ∀ {a b : Fin n}, Iff ((G.replaceVertex s t).Adj (φ a) (φ b)) (Top.top.Adj …
        x : Fin n
        hx : Eq (φ x) t
        ms : Not (Membership.mem (Set.range ⇑φ) s)
        ⊢ Nonempty (Top.top.Embedding G)
      -/
    · unfold replaceVertex at hφ
      /-
        case neg
        α : Type u_1
        G : SimpleGraph α
        n : Nat
        inst✝ : DecidableEq α
        s t : α
        h : Not ((G.replaceVertex s t).CliqueFree n)
        φ : Function.Embedding (Fin n) α
        hφ : ∀ {a b : Fin n}, Iff ({ Adj := fun v w => ite (Eq v t) (ite (Eq w t) Fals …
        x : Fin n
        hx : Eq (φ x) t
        ms : Not (Membership.mem (Set.range ⇑φ) s)
        ⊢ Nonempty (Top.top.Embedding G)
      -/
      use φ.setValue x s
      /-
        case map_rel_iff'
        α : Type u_1
        G : SimpleGraph α
        n : Nat
        inst✝ : DecidableEq α
        s t : α
        h : Not ((G.replaceVertex s t).CliqueFree n)
        φ : Function.Embedding (Fin n) α
        hφ : ∀ {a b : Fin n}, Iff ({ Adj := fun v w => ite (Eq v t) (ite (Eq w t) Fals …
        x : Fin n
        hx : Eq (φ x) t
        ms : Not (Membership.mem (Set.range ⇑φ) s)
        ⊢ ∀ {a b : Fin n}, Iff (G.Adj ((φ.setValue x s) a) ((φ.setValue x s) b)) (Top. …
      -/
      intro a b
      /-
        case map_rel_iff'
        α : Type u_1
        G : SimpleGraph α
        n : Nat
        inst✝ : DecidableEq α
        s t : α
        h : Not ((G.replaceVertex s t).CliqueFree n)
        φ : Function.Embedding (Fin n) α
        hφ : ∀ {a b : Fin n}, Iff ({ Adj := fun v w => ite (Eq v t) (ite (Eq w t) Fals …
        x : Fin n
        hx : Eq (φ x) t
        ms : Not (Membership.mem (Set.range ⇑φ) s)
        a b : Fin n
        ⊢ Iff (G.Adj ((φ.setValue x s) a) ((φ.setValue x s) b)) (Top.top.Adj a b)
      -/
      simp only [Embedding.coeFn_mk, Embedding.setValue, not_exists.mp ms, ite_false]
      /-
        case map_rel_iff'
        α : Type u_1
        G : SimpleGraph α
        n : Nat
        inst✝ : DecidableEq α
        s t : α
        h : Not ((G.replaceVertex s t).CliqueFree n)
        φ : Function.Embedding (Fin n) α
        hφ : ∀ {a b : Fin n}, Iff ({ Adj := fun v w => ite (Eq v t) (ite (Eq w t) Fals …
        x : Fin n
        hx : Eq (φ x) t
        ms : Not (Membership.mem (Set.range ⇑φ) s)
        a b : Fin n
        ⊢ Iff (G.Adj (ite (Eq a x) s (φ a)) (ite (Eq b x) s (φ b))) (Top.top.Adj a b)
      -/
      rw [apply_ite (G.Adj · _), apply_ite (G.Adj _ ·), apply_ite (G.Adj _ ·)]
      /-
        case map_rel_iff'
        α : Type u_1
        G : SimpleGraph α
        n : Nat
        inst✝ : DecidableEq α
        s t : α
        h : Not ((G.replaceVertex s t).CliqueFree n)
        φ : Function.Embedding (Fin n) α
        hφ : ∀ {a b : Fin n}, Iff ({ Adj := fun v w => ite (Eq v t) (ite (Eq w t) Fals …
        x : Fin n
        hx : Eq (φ x) t
        ms : Not (Membership.mem (Set.range ⇑φ) s)
        a b : Fin n
        ⊢ Iff (ite (Eq a x) (ite (Eq b x) (G.Adj s s) (G.Adj s (φ b))) (ite (Eq b x) ( …
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
      convert @hφ a b <;> simp only [← φ.apply_eq_iff_eq, SimpleGraph.irrefl, hx]
                          /-
                            🎉 no goals
                          -/
    /-
      case neg
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      inst✝ : DecidableEq α
      s t : α
      h : Not ((G.replaceVertex s t).CliqueFree n)
      φ : Function.Embedding (Fin n) α
      hφ : ∀ {a b : Fin n}, Iff ((G.replaceVertex s t).Adj (φ a) (φ b)) (Top.top.Adj …
      mt : Not (Membership.mem (Set.range ⇑φ) t)
      ⊢ Nonempty (Top.top.Embedding G)
    -/
  · use φ
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      inst✝ : DecidableEq α
      s t : α
      h : Not ((G.replaceVertex s t).CliqueFree n)
      φ : Function.Embedding (Fin n) α
      hφ : ∀ {a b : Fin n}, Iff ((G.replaceVertex s t).Adj (φ a) (φ b)) (Top.top.Adj …
      mt : Not (Membership.mem (Set.range ⇑φ) t)
      ⊢ ∀ {a b : Fin n}, Iff (G.Adj (φ a) (φ b)) (Top.top.Adj a b)
    -/
    simp_rw [Set.mem_range, not_exists, ← ne_eq] at mt
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      inst✝ : DecidableEq α
      s t : α
      h : Not ((G.replaceVertex s t).CliqueFree n)
      φ : Function.Embedding (Fin n) α
      hφ : ∀ {a b : Fin n}, Iff ((G.replaceVertex s t).Adj (φ a) (φ b)) (Top.top.Adj …
      mt : ∀ (x : Fin n), Ne (φ x) t
      ⊢ ∀ {a b : Fin n}, Iff (G.Adj (φ a) (φ b)) (Top.top.Adj a b)
    -/
    conv at hφ => enter [a, b]; rw [G.adj_replaceVertex_iff_of_ne _ (mt a) (mt b)]
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      inst✝ : DecidableEq α
      s t : α
      h : Not ((G.replaceVertex s t).CliqueFree n)
      φ : Function.Embedding (Fin n) α
      hφ : ∀ {a b : Fin n}, Iff (G.Adj (φ a) (φ b)) (Top.top.Adj a b)
      mt : ∀ (x : Fin n), Ne (φ x) t
      ⊢ ∀ {a b : Fin n}, Iff (G.Adj (φ a) (φ b)) (Top.top.Adj a b)
    -/
    exact hφ
    /-
      🎉 no goals
    -/


@[simp]
theorem cliqueFree_two : G.CliqueFree 2 ↔ G = ⊥ := by
  classical
  constructor
  · simp_rw [← edgeSet_eq_empty, Set.eq_empty_iff_forall_not_mem, Sym2.forall, mem_edgeSet]
    exact fun h a b hab => h _ ⟨by simpa [hab.ne], card_pair hab.ne⟩
  · rintro rfl
    exact cliqueFree_bot le_rfl


/-- Adding an edge increases the clique number by at most one. -/
protected theorem CliqueFree.sup_edge (h : G.CliqueFree n) (v w : α) :
    (G ⊔ edge v w).CliqueFree (n + 1) := by
  /-
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    h : G.CliqueFree n
    v w : α
    ⊢ (Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1)
  -/
  contrapose h
  /-
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    v w : α
    h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
    ⊢ Not (G.CliqueFree n)
  -/
  obtain ⟨f, ha⟩ := topEmbeddingOfNotCliqueFree h
  /-
    case mk
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    v w : α
    h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
    f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
    ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
    ⊢ Not (G.CliqueFree n)
  -/
  simp only [ne_eq, top_adj] at ha
  /-
    case mk
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    v w : α
    h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
    f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
    ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
    ⊢ Not (G.CliqueFree n)
  -/
  rw [not_cliqueFree_iff]
  /-
    case mk
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    v w : α
    h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
    f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
    ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
    ⊢ Nonempty (Top.top.Embedding G)
  -/
  by_cases mw : w ∈ Set.range f
    /-
      case pos
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      mw : Membership.mem (Set.range ⇑f) w
      ⊢ Nonempty (Top.top.Embedding G)
    -/
  · obtain ⟨x, hx⟩ := mw
    /-
      case pos.intro
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      x : Fin (HAdd.hAdd n 1)
      hx : Eq (f x) w
      ⊢ Nonempty (Top.top.Embedding G)
    -/
    use ⟨f ∘ x.succAboveEmb, f.2.comp Fin.succAbove_right_injective⟩
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      x : Fin (HAdd.hAdd n 1)
      hx : Eq (f x) w
      ⊢ ∀ {a b : Fin n}, Iff (G.Adj ({ toFun := Function.comp ⇑f ⇑x.succAboveEmb, in …
    -/
    intro a b
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      x : Fin (HAdd.hAdd n 1)
      hx : Eq (f x) w
      a b : Fin n
      ⊢ Iff (G.Adj ({ toFun := Function.comp ⇑f ⇑x.succAboveEmb, inj' := ⋯ } a) ({ t …
    -/
    simp_rw [Embedding.coeFn_mk, comp_apply, Fin.succAboveEmb_apply, top_adj]
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      x : Fin (HAdd.hAdd n 1)
      hx : Eq (f x) w
      a b : Fin n
      ⊢ Iff (G.Adj (f (x.succAbove a)) (f (x.succAbove b))) (Ne a b)
    -/
    have hs := @ha (x.succAbove a) (x.succAbove b)
    have ia : w ≠ f (x.succAbove a) :=
      (hx ▸ f.apply_eq_iff_eq x (x.succAbove a)).ne.mpr (x.succAbove_ne a).symm
    have ib : w ≠ f (x.succAbove b) :=
      (hx ▸ f.apply_eq_iff_eq x (x.succAbove b)).ne.mpr (x.succAbove_ne b).symm
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      x : Fin (HAdd.hAdd n 1)
      hx : Eq (f x) w
      a b : Fin n
      hs : Iff ((Max.max G (SimpleGraph.edge v w)).Adj (f (x.succAbove a)) (f (x.suc …
      ia : Ne w (f (x.succAbove a))
      ib : Ne w (f (x.succAbove b))
      ⊢ Iff (G.Adj (f (x.succAbove a)) (f (x.succAbove b))) (Ne a b)
    -/
    rw [sup_adj, edge_adj] at hs
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      x : Fin (HAdd.hAdd n 1)
      hx : Eq (f x) w
      a b : Fin n
      hs : Iff (Or (G.Adj (f (x.succAbove a)) (f (x.succAbove b))) (And (Or (And (Eq …
      ia : Ne w (f (x.succAbove a))
      ib : Ne w (f (x.succAbove b))
      ⊢ Iff (G.Adj (f (x.succAbove a)) (f (x.succAbove b))) (Ne a b)
    -/
    simp only [ia.symm, ib.symm, and_false, false_and, or_false] at hs
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      x : Fin (HAdd.hAdd n 1)
      hx : Eq (f x) w
      a b : Fin n
      ia : Ne w (f (x.succAbove a))
      ib : Ne w (f (x.succAbove b))
      hs : Iff (G.Adj (f (x.succAbove a)) (f (x.succAbove b))) (Not (Eq (x.succAbove …
      ⊢ Iff (G.Adj (f (x.succAbove a)) (f (x.succAbove b))) (Ne a b)
    -/
    rw [hs, Fin.succAbove_right_inj]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      mw : Not (Membership.mem (Set.range ⇑f) w)
      ⊢ Nonempty (Top.top.Embedding G)
    -/
  · use ⟨f ∘ Fin.succEmb n, (f.2.of_comp_iff _).mpr (Fin.succ_injective _)⟩
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      mw : Not (Membership.mem (Set.range ⇑f) w)
      ⊢ ∀ {a b : Fin n}, Iff (G.Adj ({ toFun := Function.comp ⇑f ⇑(Fin.succEmb n), i …
    -/
    intro a b
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      mw : Not (Membership.mem (Set.range ⇑f) w)
      a b : Fin n
      ⊢ Iff (G.Adj ({ toFun := Function.comp ⇑f ⇑(Fin.succEmb n), inj' := ⋯ } a) ({  …
    -/
    simp only [Fin.val_succEmb, Embedding.coeFn_mk, comp_apply, top_adj]
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      mw : Not (Membership.mem (Set.range ⇑f) w)
      a b : Fin n
      ⊢ Iff (G.Adj (f a.succ) (f b.succ)) (Ne a b)
    -/
    have hs := @ha a.succ b.succ
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      mw : Not (Membership.mem (Set.range ⇑f) w)
      a b : Fin n
      hs : Iff ((Max.max G (SimpleGraph.edge v w)).Adj (f a.succ) (f b.succ)) (Not ( …
      ⊢ Iff (G.Adj (f a.succ) (f b.succ)) (Ne a b)
    -/
    have ia : f a.succ ≠ w := by simp_all
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      mw : Not (Membership.mem (Set.range ⇑f) w)
      a b : Fin n
      hs : Iff ((Max.max G (SimpleGraph.edge v w)).Adj (f a.succ) (f b.succ)) (Not ( …
      ia : Ne (f a.succ) w
      ⊢ Iff (G.Adj (f a.succ) (f b.succ)) (Ne a b)
    -/
    have ib : f b.succ ≠ w := by simp_all
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      mw : Not (Membership.mem (Set.range ⇑f) w)
      a b : Fin n
      hs : Iff ((Max.max G (SimpleGraph.edge v w)).Adj (f a.succ) (f b.succ)) (Not ( …
      ia : Ne (f a.succ) w
      ib : Ne (f b.succ) w
      ⊢ Iff (G.Adj (f a.succ) (f b.succ)) (Ne a b)
    -/
    rw [sup_adj, edge_adj] at hs
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      mw : Not (Membership.mem (Set.range ⇑f) w)
      a b : Fin n
      hs : Iff (Or (G.Adj (f a.succ) (f b.succ)) (And (Or (And (Eq (f a.succ) v) (Eq …
      ia : Ne (f a.succ) w
      ib : Ne (f b.succ) w
      ⊢ Iff (G.Adj (f a.succ) (f b.succ)) (Ne a b)
    -/
    simp only [ia, ib, and_false, false_and, or_false] at hs
    /-
      case map_rel_iff'
      α : Type u_1
      G : SimpleGraph α
      n : Nat
      v w : α
      h : Not ((Max.max G (SimpleGraph.edge v w)).CliqueFree (HAdd.hAdd n 1))
      f : Function.Embedding (Fin (HAdd.hAdd n 1)) α
      ha : ∀ {a b : Fin (HAdd.hAdd n 1)}, Iff ((Max.max G (SimpleGraph.edge v w)).Ad …
      mw : Not (Membership.mem (Set.range ⇑f) w)
      a b : Fin n
      ia : Ne (f a.succ) w
      ib : Ne (f b.succ) w
      hs : Iff (G.Adj (f a.succ) (f b.succ)) (Not (Eq a.succ b.succ))
      ⊢ Iff (G.Adj (f a.succ) (f b.succ)) (Ne a b)
    -/
    rw [hs, Fin.succ_inj]
    /-
      🎉 no goals
    -/


/-- `G.CliqueFreeOn s n` means that `G` has no `n`-cliques contained in `s`. -/
def CliqueFreeOn (G : SimpleGraph α) (s : Set α) (n : ℕ) : Prop :=
  ∀ ⦃t⦄, ↑t ⊆ s → ¬G.IsNClique n t


theorem CliqueFreeOn.subset (hs : s₁ ⊆ s₂) (h₂ : G.CliqueFreeOn s₂ n) : G.CliqueFreeOn s₁ n :=
  fun _t hts => h₂ <| hts.trans hs


theorem CliqueFreeOn.mono (hmn : m ≤ n) (hG : G.CliqueFreeOn s m) : G.CliqueFreeOn s n := by
  /-
    α : Type u_1
    G : SimpleGraph α
    s : Set α
    m n : Nat
    hmn : LE.le m n
    hG : G.CliqueFreeOn s m
    ⊢ G.CliqueFreeOn s n
  -/
  rintro t hts ht
  /-
    α : Type u_1
    G : SimpleGraph α
    s : Set α
    m n : Nat
    hmn : LE.le m n
    hG : G.CliqueFreeOn s m
    t : Finset α
    hts : HasSubset.Subset (↑t) s
    ht : G.IsNClique n t
    ⊢ False
  -/
  obtain ⟨u, hut, hu⟩ := exists_subset_card_eq (hmn.trans ht.card_eq.ge)
  /-
    case intro.intro
    α : Type u_1
    G : SimpleGraph α
    s : Set α
    m n : Nat
    hmn : LE.le m n
    hG : G.CliqueFreeOn s m
    t : Finset α
    hts : HasSubset.Subset (↑t) s
    ht : G.IsNClique n t
    u : Finset α
    hut : HasSubset.Subset u t
    hu : Eq u.card m
    ⊢ False
  -/
  exact hG ((coe_subset.2 hut).trans hts) ⟨ht.isClique.subset hut, hu⟩
  /-
    🎉 no goals
  -/


theorem CliqueFreeOn.anti (hGH : G ≤ H) (hH : H.CliqueFreeOn s n) : G.CliqueFreeOn s n :=
  fun _t hts ht => hH hts <| ht.mono hGH


@[simp]
theorem cliqueFreeOn_empty : G.CliqueFreeOn ∅ n ↔ n ≠ 0 := by
  /-
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    ⊢ Iff (G.CliqueFreeOn EmptyCollection.emptyCollection n) (Ne n 0)
  -/
  simp [CliqueFreeOn, Set.subset_empty_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem cliqueFreeOn_singleton : G.CliqueFreeOn {a} n ↔ 1 < n := by
  /-
    α : Type u_1
    G : SimpleGraph α
    a : α
    n : Nat
    ⊢ Iff (G.CliqueFreeOn (Singleton.singleton a) n) (LT.lt 1 n)
  -/
  obtain _ | _ | n := n <;>
    /-
      case zero
      α : Type u_1
      G : SimpleGraph α
      a : α
      ⊢ Iff (G.CliqueFreeOn (Singleton.singleton a) 0) (LT.lt 1 0)
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    simp [CliqueFreeOn, isNClique_iff, ← subset_singleton_iff', (Nat.succ_ne_zero _).symm]
    /-
      🎉 no goals
    -/


@[simp]
theorem cliqueFreeOn_univ : G.CliqueFreeOn Set.univ n ↔ G.CliqueFree n := by
  /-
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    ⊢ Iff (G.CliqueFreeOn Set.univ n) (G.CliqueFree n)
  -/
  simp [CliqueFree, CliqueFreeOn]
  /-
    🎉 no goals
  -/


protected theorem CliqueFree.cliqueFreeOn (hG : G.CliqueFree n) : G.CliqueFreeOn s n :=
  fun _t _ ↦ hG _


theorem cliqueFreeOn_of_card_lt {s : Finset α} (h : #s < n) : G.CliqueFreeOn s n :=
  fun _t hts ht => h.not_le <| ht.2.symm.trans_le <| card_mono hts

-- TODO: Restate using `SimpleGraph.IndepSet` once we have it

@[simp]
theorem cliqueFreeOn_two : G.CliqueFreeOn s 2 ↔ s.Pairwise (G.Adjᶜ) := by
  classical
  refine ⟨fun h a ha b hb _ hab => h ?_ ⟨by simpa [hab.ne], card_pair hab.ne⟩, ?_⟩
  · push_cast
    exact Set.insert_subset_iff.2 ⟨ha, Set.singleton_subset_iff.2 hb⟩
  simp only [CliqueFreeOn, isNClique_iff, card_eq_two, coe_subset, not_and, not_exists]
  rintro h t hst ht a b hab rfl
  simp only [coe_insert, coe_singleton, Set.insert_subset_iff, Set.singleton_subset_iff] at hst
  refine h hst.1 hst.2 hab (ht ?_ ?_ hab) <;> simp


theorem CliqueFreeOn.of_succ (hs : G.CliqueFreeOn s (n + 1)) (ha : a ∈ s) :
    G.CliqueFreeOn (s ∩ G.neighborSet a) n := by
  classical
  refine fun t hts ht => hs ?_ (ht.insert fun b hb => (hts hb).2)
  push_cast
  exact Set.insert_subset_iff.2 ⟨ha, hts.trans Set.inter_subset_left⟩


/-- The `n`-cliques in a graph as a set. -/
def cliqueSet (n : ℕ) : Set (Finset α) :=
  { s | G.IsNClique n s }


@[simp]
theorem mem_cliqueSet_iff : s ∈ G.cliqueSet n ↔ G.IsNClique n s :=
  Iff.rfl


@[simp]
theorem cliqueSet_eq_empty_iff : G.cliqueSet n = ∅ ↔ G.CliqueFree n := by
  /-
    α : Type u_1
    G : SimpleGraph α
    n : Nat
    ⊢ Iff (Eq (G.cliqueSet n) EmptyCollection.emptyCollection) (G.CliqueFree n)
  -/
  simp_rw [CliqueFree, Set.eq_empty_iff_forall_not_mem, mem_cliqueSet_iff]
  /-
    🎉 no goals
  -/


protected alias ⟨_, CliqueFree.cliqueSet⟩ := cliqueSet_eq_empty_iff


@[gcongr, mono]
theorem cliqueSet_mono (h : G ≤ H) : G.cliqueSet n ⊆ H.cliqueSet n :=
  fun _ ↦ IsNClique.mono h


theorem cliqueSet_mono' (h : G ≤ H) : G.cliqueSet ≤ H.cliqueSet :=
  fun _ ↦ cliqueSet_mono h


@[simp]
                                                                                        /-
                                                                                          α : Type u_1
                                                                                          G : SimpleGraph α
                                                                                          s : Finset α
                                                                                          ⊢ Iff (Membership.mem (G.cliqueSet 0) s) (Membership.mem (Singleton.singleton  …
                                                                                        -/
theorem cliqueSet_zero (G : SimpleGraph α) : G.cliqueSet 0 = {∅} := Set.ext fun s => by simp
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


@[simp]
theorem cliqueSet_one (G : SimpleGraph α) : G.cliqueSet 1 = Set.range singleton :=
                      /-
                        α : Type u_1
                        G : SimpleGraph α
                        s : Finset α
                        ⊢ Iff (Membership.mem (G.cliqueSet 1) s) (Membership.mem (Set.range Singleton. …
                      -/
  Set.ext fun s => by simp [eq_comm]
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem cliqueSet_bot (hn : 1 < n) : (⊥ : SimpleGraph α).cliqueSet n = ∅ :=
  (cliqueFree_bot hn).cliqueSet


@[simp]
theorem cliqueSet_map (hn : n ≠ 1) (G : SimpleGraph α) (f : α ↪ β) :
    (G.map f).cliqueSet n = map f '' G.cliqueSet n := by
  /-
    α : Type u_1
    β : Type u_2
    n : Nat
    hn : Ne n 1
    G : SimpleGraph α
    f : Function.Embedding α β
    ⊢ Eq ((SimpleGraph.map f G).cliqueSet n) (Set.image (Finset.map f) (G.cliqueSe …
  -/
  ext s
  /-
    case h
    α : Type u_1
    β : Type u_2
    n : Nat
    hn : Ne n 1
    G : SimpleGraph α
    f : Function.Embedding α β
    s : Finset β
    ⊢ Iff (Membership.mem ((SimpleGraph.map f G).cliqueSet n) s) (Membership.mem ( …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      β : Type u_2
      n : Nat
      hn : Ne n 1
      G : SimpleGraph α
      f : Function.Embedding α β
      s : Finset β
      ⊢ Membership.mem ((SimpleGraph.map f G).cliqueSet n) s → Membership.mem (Set.i …
    -/
  · rintro ⟨hs, rfl⟩
    have hs' : (s.preimage f f.injective.injOn).map f = s := by
      classical
      rw [map_eq_image, image_preimage, filter_true_of_mem]
      rintro a ha
      obtain ⟨b, hb, hba⟩ := exists_mem_ne (hn.lt_of_le' <| Finset.card_pos.2 ⟨a, ha⟩) a
      obtain ⟨c, _, _, hc, _⟩ := hs ha hb hba.symm
      exact ⟨c, hc⟩
    /-
      case h.mp.mk
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      f : Function.Embedding α β
      s : Finset β
      hs : (SimpleGraph.map f G).IsClique ↑s
      hn : Ne s.card 1
      hs' : Eq (Finset.map f (s.preimage ⇑f ⋯)) s
      ⊢ Membership.mem (Set.image (Finset.map f) (G.cliqueSet s.card)) s
    -/
    refine ⟨s.preimage f f.injective.injOn, ⟨?_, by rw [← card_map f, hs']⟩, hs'⟩
    /-
      case h.mp.mk
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      f : Function.Embedding α β
      s : Finset β
      hs : (SimpleGraph.map f G).IsClique ↑s
      hn : Ne s.card 1
      hs' : Eq (Finset.map f (s.preimage ⇑f ⋯)) s
      ⊢ G.IsClique ↑(s.preimage ⇑f ⋯)
    -/
    rw [coe_preimage]
    /-
      case h.mp.mk
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      f : Function.Embedding α β
      s : Finset β
      hs : (SimpleGraph.map f G).IsClique ↑s
      hn : Ne s.card 1
      hs' : Eq (Finset.map f (s.preimage ⇑f ⋯)) s
      ⊢ G.IsClique (Set.preimage ⇑f ↑s)
    -/
    exact fun a ha b hb hab => map_adj_apply.1 (hs ha hb <| f.injective.ne hab)
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      β : Type u_2
      n : Nat
      hn : Ne n 1
      G : SimpleGraph α
      f : Function.Embedding α β
      s : Finset β
      ⊢ Membership.mem (Set.image (Finset.map f) (G.cliqueSet n)) s → Membership.mem …
    -/
  · rintro ⟨s, hs, rfl⟩
    /-
      case h.mpr.intro.intro
      α : Type u_1
      β : Type u_2
      n : Nat
      hn : Ne n 1
      G : SimpleGraph α
      f : Function.Embedding α β
      s : Finset α
      hs : Membership.mem (G.cliqueSet n) s
      ⊢ Membership.mem ((SimpleGraph.map f G).cliqueSet n) (Finset.map f s)
    -/
    exact hs.map
    /-
      🎉 no goals
    -/


@[simp]
theorem cliqueSet_map_of_equiv (G : SimpleGraph α) (e : α ≃ β) (n : ℕ) :
    (G.map e.toEmbedding).cliqueSet n = map e.toEmbedding '' G.cliqueSet n := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    e : Equiv α β
    n : Nat
    ⊢ Eq ((SimpleGraph.map e.toEmbedding G).cliqueSet n) (Set.image (Finset.map e. …
  -/
  obtain rfl | hn := eq_or_ne n 1
    /-
      case inl
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      e : Equiv α β
      ⊢ Eq ((SimpleGraph.map e.toEmbedding G).cliqueSet 1) (Set.image (Finset.map e. …
    -/
  · ext
    /-
      case inl.h
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      e : Equiv α β
      x✝ : Finset β
      ⊢ Iff (Membership.mem ((SimpleGraph.map e.toEmbedding G).cliqueSet 1) x✝) (Mem …
    -/
    simp [e.exists_congr_left]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      e : Equiv α β
      n : Nat
      hn : Ne n 1
      ⊢ Eq ((SimpleGraph.map e.toEmbedding G).cliqueSet n) (Set.image (Finset.map e. …
    -/
  · exact cliqueSet_map hn _ _
    /-
      🎉 no goals
    -/


/-- The maximum number of vertices in a clique of a graph `G`. -/
noncomputable def cliqueNum (G : SimpleGraph α) : ℕ := sSup {n | ∃ s, G.IsNClique n s}


private lemma fintype_cliqueNum_bddAbove [Fintype α] : BddAbove {n | ∃ s, G.IsNClique n s} := by
  /-
    α : Type u_3
    G : SimpleGraph α
    inst✝ : Fintype α
    ⊢ BddAbove (setOf fun n => Exists fun s => G.IsNClique n s)
  -/
  use Fintype.card α
  /-
    case h
    α : Type u_3
    G : SimpleGraph α
    inst✝ : Fintype α
    ⊢ Membership.mem (upperBounds (setOf fun n => Exists fun s => G.IsNClique n s) …
  -/
  rintro y ⟨s, syc⟩
  /-
    case h.intro
    α : Type u_3
    G : SimpleGraph α
    inst✝ : Fintype α
    y : Nat
    s : Finset α
    syc : G.IsNClique y s
    ⊢ LE.le y (Fintype.card α)
  -/
  rw [isNClique_iff] at syc
  /-
    case h.intro
    α : Type u_3
    G : SimpleGraph α
    inst✝ : Fintype α
    y : Nat
    s : Finset α
    syc : And (G.IsClique ↑s) (Eq s.card y)
    ⊢ LE.le y (Fintype.card α)
  -/
  rw [← syc.right]
  /-
    case h.intro
    α : Type u_3
    G : SimpleGraph α
    inst✝ : Fintype α
    y : Nat
    s : Finset α
    syc : And (G.IsClique ↑s) (Eq s.card y)
    ⊢ LE.le s.card (Fintype.card α)
  -/
  exact Finset.card_le_card (Finset.subset_univ s)
  /-
    🎉 no goals
  -/


lemma IsClique.card_le_cliqueNum [Fintype α] {t : Finset α} {tc : G.IsClique t} :
    #t ≤ G.cliqueNum :=
  le_csSup G.fintype_cliqueNum_bddAbove (Exists.intro t ⟨tc, rfl⟩)


lemma exists_isNClique_cliqueNum [Fintype α] : ∃ s, G.IsNClique G.cliqueNum s :=
                      /-
                        α : Type u_3
                        G : SimpleGraph α
                        inst✝ : Fintype α
                        ⊢ Membership.mem (setOf fun n => Exists fun s => G.IsNClique n s) 0
                      -/
  Nat.sSup_mem ⟨0, by simp [isNClique_empty.mpr rfl]⟩ G.fintype_cliqueNum_bddAbove
                      /-
                        🎉 no goals
                      -/


/-- A maximum clique in a graph `G` is a clique with the largest possible size. -/
structure IsMaximumClique [Fintype α] (G : SimpleGraph α) (s : Finset α) : Prop where
  (isClique : G.IsClique s)
  (maximum : ∀ t : Finset α, G.IsClique t → #t ≤ #s)


theorem isMaximumClique_iff [Fintype α] {s : Finset α} :
    G.IsMaximumClique s ↔ G.IsClique s ∧ ∀ t : Finset α, G.IsClique t → #t ≤ #s :=
  ⟨fun h ↦ ⟨h.1, h.2⟩, fun h ↦ ⟨h.1, h.2⟩⟩


/-- A maximal clique in a graph `G` is a clique that cannot be extended by adding more vertices. -/
theorem isMaximalClique_iff {s : Set α} :
    Maximal G.IsClique s ↔ G.IsClique s ∧ ∀ t : Set α, G.IsClique t → s ⊆ t → t ⊆ s :=
  Iff.rfl


lemma IsMaximumClique.isMaximalClique [Fintype α] (s : Finset α) (M : G.IsMaximumClique s) :
    Maximal G.IsClique s :=
  ⟨ M.isClique,
    fun t ht hsub => by
      /-
        α : Type u_3
        G : SimpleGraph α
        inst✝ : Fintype α
        s : Finset α
        M : G.IsMaximumClique s
        t : Set α
        ht : G.IsClique t
        hsub : LE.le (↑s) t
        ⊢ LE.le t ↑s
      -/
      by_contra hc
      /-
        α : Type u_3
        G : SimpleGraph α
        inst✝ : Fintype α
        s : Finset α
        M : G.IsMaximumClique s
        t : Set α
        ht : G.IsClique t
        hsub : LE.le (↑s) t
        hc : Not (LE.le t ↑s)
        ⊢ False
      -/
      have fint : Fintype t := ofFinite ↑t
      /-
        α : Type u_3
        G : SimpleGraph α
        inst✝ : Fintype α
        s : Finset α
        M : G.IsMaximumClique s
        t : Set α
        ht : G.IsClique t
        hsub : LE.le (↑s) t
        hc : Not (LE.le t ↑s)
        fint : Fintype ↑t
        ⊢ False
      -/
      have ne : s ≠ t.toFinset := fun a ↦ by subst a; simp_all[Set.coe_toFinset, not_true_eq_false]
      /-
        α : Type u_3
        G : SimpleGraph α
        inst✝ : Fintype α
        s : Finset α
        M : G.IsMaximumClique s
        t : Set α
        ht : G.IsClique t
        hsub : LE.le (↑s) t
        hc : Not (LE.le t ↑s)
        fint : Fintype ↑t
        ne : Ne s t.toFinset
        ⊢ False
      -/
      have hle : #t.toFinset ≤ #s := M.maximum t.toFinset (by simp [Set.coe_toFinset, ht])
      have hlt : #s < #t.toFinset :=
        card_lt_card (ssubset_of_ne_of_subset ne (Set.subset_toFinset.mpr hsub))
      /-
        α : Type u_3
        G : SimpleGraph α
        inst✝ : Fintype α
        s : Finset α
        M : G.IsMaximumClique s
        t : Set α
        ht : G.IsClique t
        hsub : LE.le (↑s) t
        hc : Not (LE.le t ↑s)
        fint : Fintype ↑t
        ne : Ne s t.toFinset
        hle : LE.le t.toFinset.card s.card
        hlt : LT.lt s.card t.toFinset.card
        ⊢ False
      -/
      exact lt_irrefl _ (lt_of_lt_of_le hlt hle) ⟩
      /-
        🎉 no goals
      -/


lemma maximumClique_card_eq_cliqueNum [Fintype α] (s : Finset α) (sm : G.IsMaximumClique s) :
    #s = G.cliqueNum := by
  /-
    α : Type u_3
    G : SimpleGraph α
    inst✝ : Fintype α
    s : Finset α
    sm : G.IsMaximumClique s
    ⊢ Eq s.card G.cliqueNum
  -/
  obtain ⟨sc, sm⟩ := sm
  /-
    case mk
    α : Type u_3
    G : SimpleGraph α
    inst✝ : Fintype α
    s : Finset α
    sc : G.IsClique ↑s
    sm : ∀ (t : Finset α), G.IsClique ↑t → LE.le t.card s.card
    ⊢ Eq s.card G.cliqueNum
  -/
  obtain ⟨t, tc, tcard⟩ := G.exists_isNClique_cliqueNum
  /-
    case mk.intro.mk
    α : Type u_3
    G : SimpleGraph α
    inst✝ : Fintype α
    s : Finset α
    sc : G.IsClique ↑s
    sm : ∀ (t : Finset α), G.IsClique ↑t → LE.le t.card s.card
    t : Finset α
    tc : G.IsClique ↑t
    tcard : Eq t.card G.cliqueNum
    ⊢ Eq s.card G.cliqueNum
  -/
  exact eq_of_le_of_not_lt sc.card_le_cliqueNum (by simp [← tcard, sm t tc])
  /-
    🎉 no goals
  -/


lemma maximumClique_exists [Fintype α] : ∃ (s : Finset α), G.IsMaximumClique s := by
  /-
    α : Type u_3
    G : SimpleGraph α
    inst✝ : Fintype α
    ⊢ Exists fun s => G.IsMaximumClique s
  -/
  obtain ⟨s, snc⟩ := G.exists_isNClique_cliqueNum
  /-
    case intro
    α : Type u_3
    G : SimpleGraph α
    inst✝ : Fintype α
    s : Finset α
    snc : G.IsNClique G.cliqueNum s
    ⊢ Exists fun s => G.IsMaximumClique s
  -/
  exact ⟨s, ⟨snc.isClique, fun t ht => snc.card_eq.symm ▸ ht.card_le_cliqueNum⟩⟩
  /-
    🎉 no goals
  -/


/-- The `n`-cliques in a graph as a finset. -/
def cliqueFinset (n : ℕ) : Finset (Finset α) := {s | G.IsNClique n s}


variable {G} in
@[simp]
theorem mem_cliqueFinset_iff : s ∈ G.cliqueFinset n ↔ G.IsNClique n s :=
  mem_filter.trans <| and_iff_right <| mem_univ _


@[simp, norm_cast]
theorem coe_cliqueFinset (n : ℕ) : (G.cliqueFinset n : Set (Finset α)) = G.cliqueSet n :=
  Set.ext fun _ ↦ mem_cliqueFinset_iff


@[simp]
theorem cliqueFinset_eq_empty_iff : G.cliqueFinset n = ∅ ↔ G.CliqueFree n := by
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : DecidableRel G.Adj
    n : Nat
    ⊢ Iff (Eq (G.cliqueFinset n) EmptyCollection.emptyCollection) (G.CliqueFree n)
  -/
  simp_rw [CliqueFree, eq_empty_iff_forall_not_mem, mem_cliqueFinset_iff]
  /-
    🎉 no goals
  -/


protected alias ⟨_, CliqueFree.cliqueFinset⟩ := cliqueFinset_eq_empty_iff


theorem card_cliqueFinset_le : #(G.cliqueFinset n) ≤ (card α).choose n := by
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : DecidableRel G.Adj
    n : Nat
    ⊢ LE.le (G.cliqueFinset n).card ((Fintype.card α).choose n)
  -/
  rw [← card_univ, ← card_powersetCard]
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : DecidableRel G.Adj
    n : Nat
    ⊢ LE.le (G.cliqueFinset n).card (Finset.powersetCard n Finset.univ).card
  -/
  refine card_mono fun s => ?_
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : DecidableRel G.Adj
    n : Nat
    s : Finset α
    ⊢ Membership.mem (G.cliqueFinset n) s → Membership.mem (Finset.powersetCard n  …
  -/
  simpa [mem_powersetCard_univ] using IsNClique.card_eq
  /-
    🎉 no goals
  -/


@[gcongr, mono]
theorem cliqueFinset_mono (h : G ≤ H) : G.cliqueFinset n ⊆ H.cliqueFinset n :=
  monotone_filter_right _ fun _ ↦ IsNClique.mono h


@[simp]
theorem cliqueFinset_map (f : α ↪ β) (hn : n ≠ 1) :
    (G.map f).cliqueFinset n = (G.cliqueFinset n).map ⟨map f, Finset.map_injective _⟩ :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      inst✝⁴ : Fintype α
      inst✝³ : DecidableEq α
      inst✝² : DecidableRel G.Adj
      n : Nat
      inst✝¹ : Fintype β
      inst✝ : DecidableEq β
      f : Function.Embedding α β
      hn : Ne n 1
      ⊢ Eq ↑((SimpleGraph.map f G).cliqueFinset n) ↑(Finset.map { toFun := Finset.ma …
    -/
    simp_rw [coe_cliqueFinset, cliqueSet_map hn, coe_map, coe_cliqueFinset, Embedding.coeFn_mk]
    /-
      🎉 no goals
    -/


@[simp]
theorem cliqueFinset_map_of_equiv (e : α ≃ β) (n : ℕ) :
    (G.map e.toEmbedding).cliqueFinset n =
      (G.cliqueFinset n).map ⟨map e.toEmbedding, Finset.map_injective _⟩ :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        G : SimpleGraph α
                        inst✝⁴ : Fintype α
                        inst✝³ : DecidableEq α
                        inst✝² : DecidableRel G.Adj
                        inst✝¹ : Fintype β
                        inst✝ : DecidableEq β
                        e : Equiv α β
                        n : Nat
                        ⊢ Eq ↑((SimpleGraph.map e.toEmbedding G).cliqueFinset n) ↑(Finset.map { toFun  …
                      -/
  coe_injective <| by push_cast; exact cliqueSet_map_of_equiv _ _ _
                                 /-
                                   🎉 no goals
                                 -/


