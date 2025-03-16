/-- Definition of a Baire space. -/
theorem dense_iInter_of_isOpen_nat {f : ℕ → Set X} (ho : ∀ n, IsOpen (f n))
    (hd : ∀ n, Dense (f n)) : Dense (⋂ n, f n) :=
  BaireSpace.baire_property f ho hd


/-- Baire theorem: a countable intersection of dense open sets is dense. Formulated here with ⋂₀. -/
theorem dense_sInter_of_isOpen {S : Set (Set X)} (ho : ∀ s ∈ S, IsOpen s) (hS : S.Countable)
    (hd : ∀ s ∈ S, Dense s) : Dense (⋂₀ S) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : BaireSpace X
    S : Set (Set X)
    ho : ∀ (s : Set X), Membership.mem S s → IsOpen s
    hS : S.Countable
    hd : ∀ (s : Set X), Membership.mem S s → Dense s
    ⊢ Dense S.sInter
  -/
  rcases S.eq_empty_or_nonempty with h | h
    /-
      case inl
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : BaireSpace X
      S : Set (Set X)
      ho : ∀ (s : Set X), Membership.mem S s → IsOpen s
      hS : S.Countable
      hd : ∀ (s : Set X), Membership.mem S s → Dense s
      h : Eq S EmptyCollection.emptyCollection
      ⊢ Dense S.sInter
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : BaireSpace X
      S : Set (Set X)
      ho : ∀ (s : Set X), Membership.mem S s → IsOpen s
      hS : S.Countable
      hd : ∀ (s : Set X), Membership.mem S s → Dense s
      h : S.Nonempty
      ⊢ Dense S.sInter
    -/
  · rcases hS.exists_eq_range h with ⟨f, rfl⟩
    /-
      case inr.intro
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : BaireSpace X
      f : Nat → Set X
      ho : ∀ (s : Set X), Membership.mem (Set.range f) s → IsOpen s
      hS : (Set.range f).Countable
      hd : ∀ (s : Set X), Membership.mem (Set.range f) s → Dense s
      h : (Set.range f).Nonempty
      ⊢ Dense (Set.range f).sInter
    -/
    exact dense_iInter_of_isOpen_nat (forall_mem_range.1 ho) (forall_mem_range.1 hd)
    /-
      🎉 no goals
    -/


/-- Baire theorem: a countable intersection of dense open sets is dense. Formulated here with
an index set which is a countable set in any type. -/
theorem dense_biInter_of_isOpen {S : Set α} {f : α → Set X} (ho : ∀ s ∈ S, IsOpen (f s))
    (hS : S.Countable) (hd : ∀ s ∈ S, Dense (f s)) : Dense (⋂ s ∈ S, f s) := by
  /-
    X : Type u_1
    α : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : BaireSpace X
    S : Set α
    f : α → Set X
    ho : ∀ (s : α), Membership.mem S s → IsOpen (f s)
    hS : S.Countable
    hd : ∀ (s : α), Membership.mem S s → Dense (f s)
    ⊢ Dense (Set.iInter fun s => Set.iInter fun h => f s)
  -/
  rw [← sInter_image]
  /-
    X : Type u_1
    α : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : BaireSpace X
    S : Set α
    f : α → Set X
    ho : ∀ (s : α), Membership.mem S s → IsOpen (f s)
    hS : S.Countable
    hd : ∀ (s : α), Membership.mem S s → Dense (f s)
    ⊢ Dense (Set.image f S).sInter
  -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  refine dense_sInter_of_isOpen ?_ (hS.image _) ?_ <;> rwa [forall_mem_image]
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- Baire theorem: a countable intersection of dense open sets is dense. Formulated here with
an index set which is a countable type. -/
theorem dense_iInter_of_isOpen [Countable ι] {f : ι → Set X} (ho : ∀ i, IsOpen (f i))
    (hd : ∀ i, Dense (f i)) : Dense (⋂ s, f s) :=
  dense_sInter_of_isOpen (forall_mem_range.2 ho) (countable_range _) (forall_mem_range.2 hd)


/-- A set is residual (comeagre) if and only if it includes a dense `Gδ` set. -/
theorem mem_residual {s : Set X} : s ∈ residual X ↔ ∃ t ⊆ s, IsGδ t ∧ Dense t := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : BaireSpace X
    s : Set X
    ⊢ Iff (Membership.mem (residual X) s) (Exists fun t => And (HasSubset.Subset t …
  -/
  constructor
    /-
      case mp
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : BaireSpace X
      s : Set X
      ⊢ Membership.mem (residual X) s → Exists fun t => And (HasSubset.Subset t s) ( …
    -/
  · rw [mem_residual_iff]
    /-
      case mp
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : BaireSpace X
      s : Set X
      ⊢ (Exists fun S => And (∀ (t : Set X), Membership.mem S t → IsOpen t) (And (∀  …
    -/
    rintro ⟨S, hSo, hSd, Sct, Ss⟩
    /-
      case mp.intro.intro.intro.intro
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : BaireSpace X
      s : Set X
      S : Set (Set X)
      hSo : ∀ (t : Set X), Membership.mem S t → IsOpen t
      hSd : ∀ (t : Set X), Membership.mem S t → Dense t
      Sct : S.Countable
      Ss : HasSubset.Subset S.sInter s
      ⊢ Exists fun t => And (HasSubset.Subset t s) (And (IsGδ t) (Dense t))
    -/
    refine ⟨_, Ss, ⟨_, fun t ht => hSo _ ht, Sct, rfl⟩, ?_⟩
    /-
      case mp.intro.intro.intro.intro
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : BaireSpace X
      s : Set X
      S : Set (Set X)
      hSo : ∀ (t : Set X), Membership.mem S t → IsOpen t
      hSd : ∀ (t : Set X), Membership.mem S t → Dense t
      Sct : S.Countable
      Ss : HasSubset.Subset S.sInter s
      ⊢ Dense S.sInter
    -/
    exact dense_sInter_of_isOpen hSo Sct hSd
    /-
      🎉 no goals
    -/
  /-
    case mpr
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : BaireSpace X
    s : Set X
    ⊢ (Exists fun t => And (HasSubset.Subset t s) (And (IsGδ t) (Dense t))) → Memb …
  -/
  rintro ⟨t, ts, ho, hd⟩
  /-
    case mpr.intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : BaireSpace X
    s t : Set X
    ts : HasSubset.Subset t s
    ho : IsGδ t
    hd : Dense t
    ⊢ Membership.mem (residual X) s
  -/
  exact mem_of_superset (residual_of_dense_Gδ ho hd) ts
  /-
    🎉 no goals
  -/


/-- A property holds on a residual (comeagre) set if and only if it holds on some dense `Gδ` set. -/
theorem eventually_residual {p : X → Prop} :
    (∀ᶠ x in residual X, p x) ↔ ∃ t : Set X, IsGδ t ∧ Dense t ∧ ∀ x ∈ t, p x := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : BaireSpace X
    p : X → Prop
    ⊢ Iff (Filter.Eventually (fun x => p x) (residual X)) (Exists fun t => And (Is …
  -/
  simp only [Filter.Eventually, mem_residual, subset_def, mem_setOf_eq]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : BaireSpace X
    p : X → Prop
    ⊢ Iff (Exists fun t => And (∀ (x : X), Membership.mem t x → p x) (And (IsGδ t) …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem dense_of_mem_residual {s : Set X} (hs : s ∈ residual X) : Dense s :=
  let ⟨_, hts, _, hd⟩ := mem_residual.1 hs
  hd.mono hts


/-- Baire theorem: a countable intersection of dense Gδ sets is dense. Formulated here with ⋂₀. -/
theorem dense_sInter_of_Gδ {S : Set (Set X)} (ho : ∀ s ∈ S, IsGδ s) (hS : S.Countable)
    (hd : ∀ s ∈ S, Dense s) : Dense (⋂₀ S) :=
  dense_of_mem_residual ((countable_sInter_mem hS).mpr
    (fun _ hs => residual_of_dense_Gδ (ho _ hs) (hd _ hs)))


/-- Baire theorem: a countable intersection of dense Gδ sets is dense. Formulated here with
an index set which is a countable type. -/
theorem dense_iInter_of_Gδ [Countable ι] {f : ι → Set X} (ho : ∀ s, IsGδ (f s))
    (hd : ∀ s, Dense (f s)) : Dense (⋂ s, f s) :=
  dense_sInter_of_Gδ (forall_mem_range.2 ‹_›) (countable_range _) (forall_mem_range.2 ‹_›)


/-- Baire theorem: a countable intersection of dense Gδ sets is dense. Formulated here with
an index set which is a countable set in any type. -/
theorem dense_biInter_of_Gδ {S : Set α} {f : ∀ x ∈ S, Set X} (ho : ∀ s (H : s ∈ S), IsGδ (f s H))
    (hS : S.Countable) (hd : ∀ s (H : s ∈ S), Dense (f s H)) : Dense (⋂ s ∈ S, f s ‹_›) := by
  /-
    X : Type u_1
    α : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : BaireSpace X
    S : Set α
    f : (x : α) → Membership.mem S x → Set X
    ho : ∀ (s : α) (H : Membership.mem S s), IsGδ (f s H)
    hS : S.Countable
    hd : ∀ (s : α) (H : Membership.mem S s), Dense (f s H)
    ⊢ Dense (Set.iInter fun s => Set.iInter fun h => f s h)
  -/
  rw [biInter_eq_iInter]
  /-
    X : Type u_1
    α : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : BaireSpace X
    S : Set α
    f : (x : α) → Membership.mem S x → Set X
    ho : ∀ (s : α) (H : Membership.mem S s), IsGδ (f s H)
    hS : S.Countable
    hd : ∀ (s : α) (H : Membership.mem S s), Dense (f s H)
    ⊢ Dense (Set.iInter fun x => f ↑x ⋯)
  -/
  haveI := hS.to_subtype
  /-
    X : Type u_1
    α : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : BaireSpace X
    S : Set α
    f : (x : α) → Membership.mem S x → Set X
    ho : ∀ (s : α) (H : Membership.mem S s), IsGδ (f s H)
    hS : S.Countable
    hd : ∀ (s : α) (H : Membership.mem S s), Dense (f s H)
    this : Countable ↑S
    ⊢ Dense (Set.iInter fun x => f ↑x ⋯)
  -/
  exact dense_iInter_of_Gδ (fun s => ho s s.2) fun s => hd s s.2
  /-
    🎉 no goals
  -/


/-- Baire theorem: the intersection of two dense Gδ sets is dense. -/
theorem Dense.inter_of_Gδ {s t : Set X} (hs : IsGδ s) (ht : IsGδ t) (hsc : Dense s)
    (htc : Dense t) : Dense (s ∩ t) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : BaireSpace X
    s t : Set X
    hs : IsGδ s
    ht : IsGδ t
    hsc : Dense s
    htc : Dense t
    ⊢ Dense (Inter.inter s t)
  -/
  rw [inter_eq_iInter]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : BaireSpace X
    s t : Set X
    hs : IsGδ s
    ht : IsGδ t
    hsc : Dense s
    htc : Dense t
    ⊢ Dense (Set.iInter fun b => cond b s t)
  -/
                               /-
                                 🎉 no goals
                               -/
  apply dense_iInter_of_Gδ <;> simp [Bool.forall_bool, *]
                               /-
                                 🎉 no goals
                               -/


/-- If a countable family of closed sets cover a dense `Gδ` set, then the union of their interiors
is dense. Formulated here with `⋃`. -/
theorem IsGδ.dense_iUnion_interior_of_closed [Countable ι] {s : Set X} (hs : IsGδ s) (hd : Dense s)
    {f : ι → Set X} (hc : ∀ i, IsClosed (f i)) (hU : s ⊆ ⋃ i, f i) :
    Dense (⋃ i, interior (f i)) := by
  /-
    X : Type u_1
    ι : Sort u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : BaireSpace X
    inst✝ : Countable ι
    s : Set X
    hs : IsGδ s
    hd : Dense s
    f : ι → Set X
    hc : ∀ (i : ι), IsClosed (f i)
    hU : HasSubset.Subset s (Set.iUnion fun i => f i)
    ⊢ Dense (Set.iUnion fun i => interior (f i))
  -/
  let g i := (frontier (f i))ᶜ
  /-
    X : Type u_1
    ι : Sort u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : BaireSpace X
    inst✝ : Countable ι
    s : Set X
    hs : IsGδ s
    hd : Dense s
    f : ι → Set X
    hc : ∀ (i : ι), IsClosed (f i)
    hU : HasSubset.Subset s (Set.iUnion fun i => f i)
    g : ι → Set X := fun i => HasCompl.compl (frontier (f i))
    ⊢ Dense (Set.iUnion fun i => interior (f i))
  -/
  have hgo : ∀ i, IsOpen (g i) := fun i => isClosed_frontier.isOpen_compl
  have hgd : Dense (⋂ i, g i) := by
    refine dense_iInter_of_isOpen hgo fun i x => ?_
    rw [closure_compl, interior_frontier (hc _)]
    exact id
  /-
    X : Type u_1
    ι : Sort u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : BaireSpace X
    inst✝ : Countable ι
    s : Set X
    hs : IsGδ s
    hd : Dense s
    f : ι → Set X
    hc : ∀ (i : ι), IsClosed (f i)
    hU : HasSubset.Subset s (Set.iUnion fun i => f i)
    g : ι → Set X := fun i => HasCompl.compl (frontier (f i))
    hgo : ∀ (i : ι), IsOpen (g i)
    hgd : Dense (Set.iInter fun i => g i)
    ⊢ Dense (Set.iUnion fun i => interior (f i))
  -/
  refine (hd.inter_of_Gδ hs (.iInter_of_isOpen fun i => (hgo i)) hgd).mono ?_
  /-
    X : Type u_1
    ι : Sort u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : BaireSpace X
    inst✝ : Countable ι
    s : Set X
    hs : IsGδ s
    hd : Dense s
    f : ι → Set X
    hc : ∀ (i : ι), IsClosed (f i)
    hU : HasSubset.Subset s (Set.iUnion fun i => f i)
    g : ι → Set X := fun i => HasCompl.compl (frontier (f i))
    hgo : ∀ (i : ι), IsOpen (g i)
    hgd : Dense (Set.iInter fun i => g i)
    ⊢ HasSubset.Subset (Inter.inter s (Set.iInter fun i => g i)) (Set.iUnion fun i …
  -/
  rintro x ⟨hxs, hxg⟩
  /-
    case intro
    X : Type u_1
    ι : Sort u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : BaireSpace X
    inst✝ : Countable ι
    s : Set X
    hs : IsGδ s
    hd : Dense s
    f : ι → Set X
    hc : ∀ (i : ι), IsClosed (f i)
    hU : HasSubset.Subset s (Set.iUnion fun i => f i)
    g : ι → Set X := fun i => HasCompl.compl (frontier (f i))
    hgo : ∀ (i : ι), IsOpen (g i)
    hgd : Dense (Set.iInter fun i => g i)
    x : X
    hxs : Membership.mem s x
    hxg : Membership.mem (Set.iInter fun i => g i) x
    ⊢ Membership.mem (Set.iUnion fun i => interior (f i)) x
  -/
  rw [mem_iInter] at hxg
  /-
    case intro
    X : Type u_1
    ι : Sort u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : BaireSpace X
    inst✝ : Countable ι
    s : Set X
    hs : IsGδ s
    hd : Dense s
    f : ι → Set X
    hc : ∀ (i : ι), IsClosed (f i)
    hU : HasSubset.Subset s (Set.iUnion fun i => f i)
    g : ι → Set X := fun i => HasCompl.compl (frontier (f i))
    hgo : ∀ (i : ι), IsOpen (g i)
    hgd : Dense (Set.iInter fun i => g i)
    x : X
    hxs : Membership.mem s x
    hxg : ∀ (i : ι), Membership.mem (g i) x
    ⊢ Membership.mem (Set.iUnion fun i => interior (f i)) x
  -/
  rcases mem_iUnion.1 (hU hxs) with ⟨i, hi⟩
  /-
    case intro.intro
    X : Type u_1
    ι : Sort u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : BaireSpace X
    inst✝ : Countable ι
    s : Set X
    hs : IsGδ s
    hd : Dense s
    f : ι → Set X
    hc : ∀ (i : ι), IsClosed (f i)
    hU : HasSubset.Subset s (Set.iUnion fun i => f i)
    g : ι → Set X := fun i => HasCompl.compl (frontier (f i))
    hgo : ∀ (i : ι), IsOpen (g i)
    hgd : Dense (Set.iInter fun i => g i)
    x : X
    hxs : Membership.mem s x
    hxg : ∀ (i : ι), Membership.mem (g i) x
    i : ι
    hi : Membership.mem (f i) x
    ⊢ Membership.mem (Set.iUnion fun i => interior (f i)) x
  -/
  exact mem_iUnion.2 ⟨i, self_diff_frontier (f i) ▸ ⟨hi, hxg _⟩⟩
  /-
    🎉 no goals
  -/


/-- If a countable family of closed sets cover a dense `Gδ` set, then the union of their interiors
is dense. Formulated here with a union over a countable set in any type. -/
theorem IsGδ.dense_biUnion_interior_of_closed {t : Set α} {s : Set X} (hs : IsGδ s) (hd : Dense s)
    (ht : t.Countable) {f : α → Set X} (hc : ∀ i ∈ t, IsClosed (f i)) (hU : s ⊆ ⋃ i ∈ t, f i) :
    Dense (⋃ i ∈ t, interior (f i)) := by
  /-
    X : Type u_1
    α : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : BaireSpace X
    t : Set α
    s : Set X
    hs : IsGδ s
    hd : Dense s
    ht : t.Countable
    f : α → Set X
    hc : ∀ (i : α), Membership.mem t i → IsClosed (f i)
    hU : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => f i)
    ⊢ Dense (Set.iUnion fun i => Set.iUnion fun h => interior (f i))
  -/
  haveI := ht.to_subtype
  /-
    X : Type u_1
    α : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : BaireSpace X
    t : Set α
    s : Set X
    hs : IsGδ s
    hd : Dense s
    ht : t.Countable
    f : α → Set X
    hc : ∀ (i : α), Membership.mem t i → IsClosed (f i)
    hU : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => f i)
    this : Countable ↑t
    ⊢ Dense (Set.iUnion fun i => Set.iUnion fun h => interior (f i))
  -/
  simp only [biUnion_eq_iUnion, SetCoe.forall'] at *
  /-
    X : Type u_1
    α : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : BaireSpace X
    t : Set α
    s : Set X
    hs : IsGδ s
    hd : Dense s
    ht : t.Countable
    f : α → Set X
    this : Countable ↑t
    hc : ∀ (x : ↑t), IsClosed (f ↑x)
    hU : HasSubset.Subset s (Set.iUnion fun x => f ↑x)
    ⊢ Dense (Set.iUnion fun x => interior (f ↑x))
  -/
  exact hs.dense_iUnion_interior_of_closed hd hc hU
  /-
    🎉 no goals
  -/


/-- If a countable family of closed sets cover a dense `Gδ` set, then the union of their interiors
is dense. Formulated here with `⋃₀`. -/
theorem IsGδ.dense_sUnion_interior_of_closed {T : Set (Set X)} {s : Set X} (hs : IsGδ s)
    (hd : Dense s) (hc : T.Countable) (hc' : ∀ t ∈ T, IsClosed t) (hU : s ⊆ ⋃₀ T) :
    Dense (⋃ t ∈ T, interior t) :=
                                                      /-
                                                        X : Type u_1
                                                        inst✝¹ : TopologicalSpace X
                                                        inst✝ : BaireSpace X
                                                        T : Set (Set X)
                                                        s : Set X
                                                        hs : IsGδ s
                                                        hd : Dense s
                                                        hc : T.Countable
                                                        hc' : ∀ (t : Set X), Membership.mem T t → IsClosed t
                                                        hU : HasSubset.Subset s T.sUnion
                                                        ⊢ HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => i)
                                                      -/
  hs.dense_biUnion_interior_of_closed hd hc hc' <| by rwa [← sUnion_eq_biUnion]
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- Baire theorem: if countably many closed sets cover the whole space, then their interiors
are dense. Formulated here with an index set which is a countable set in any type. -/
theorem dense_biUnion_interior_of_closed {S : Set α} {f : α → Set X} (hc : ∀ s ∈ S, IsClosed (f s))
    (hS : S.Countable) (hU : ⋃ s ∈ S, f s = univ) : Dense (⋃ s ∈ S, interior (f s)) :=
  IsGδ.univ.dense_biUnion_interior_of_closed dense_univ hS hc hU.ge


/-- Baire theorem: if countably many closed sets cover the whole space, then their interiors
are dense. Formulated here with `⋃₀`. -/
theorem dense_sUnion_interior_of_closed {S : Set (Set X)} (hc : ∀ s ∈ S, IsClosed s)
    (hS : S.Countable) (hU : ⋃₀ S = univ) : Dense (⋃ s ∈ S, interior s) :=
  IsGδ.univ.dense_sUnion_interior_of_closed dense_univ hS hc hU.ge


/-- Baire theorem: if countably many closed sets cover the whole space, then their interiors
are dense. Formulated here with an index set which is a countable type. -/
theorem dense_iUnion_interior_of_closed [Countable ι] {f : ι → Set X} (hc : ∀ i, IsClosed (f i))
    (hU : ⋃ i, f i = univ) : Dense (⋃ i, interior (f i)) :=
  IsGδ.univ.dense_iUnion_interior_of_closed dense_univ hc hU.ge


/-- One of the most useful consequences of Baire theorem: if a countable union of closed sets
covers the space, then one of the sets has nonempty interior. -/
theorem nonempty_interior_of_iUnion_of_closed [Nonempty X] [Countable ι] {f : ι → Set X}
    (hc : ∀ i, IsClosed (f i)) (hU : ⋃ i, f i = univ) : ∃ i, (interior <| f i).Nonempty := by
  /-
    X : Type u_1
    ι : Sort u_3
    inst✝³ : TopologicalSpace X
    inst✝² : BaireSpace X
    inst✝¹ : Nonempty X
    inst✝ : Countable ι
    f : ι → Set X
    hc : ∀ (i : ι), IsClosed (f i)
    hU : Eq (Set.iUnion fun i => f i) Set.univ
    ⊢ Exists fun i => (interior (f i)).Nonempty
  -/
  simpa using (dense_iUnion_interior_of_closed hc hU).nonempty
  /-
    🎉 no goals
  -/


