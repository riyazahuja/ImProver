instance (α : Type u) (β : Type v) [Fintype α] [Fintype β] : Fintype (α ⊕ β) where
  elems := univ.disjSum univ
                 /-
                   α✝ : Type u_1
                   β✝ : Type u_2
                   α : Type u
                   β : Type v
                   inst✝¹ : Fintype α
                   inst✝ : Fintype β
                   ⊢ ∀ (x : Sum α β), Membership.mem (Finset.univ.disjSum Finset.univ) x
                 -/
                                    /-
                                      🎉 no goals
                                    -/
  complete := by rintro (_ | _) <;> simp
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem Finset.univ_disjSum_univ {α β : Type*} [Fintype α] [Fintype β] :
    univ.disjSum univ = (univ : Finset (α ⊕ β)) :=
  rfl


@[simp]
theorem Fintype.card_sum [Fintype α] [Fintype β] :
    Fintype.card (α ⊕ β) = Fintype.card α + Fintype.card β :=
  card_disjSum _ _


/-- If the subtype of all-but-one elements is a `Fintype` then the type itself is a `Fintype`. -/
def fintypeOfFintypeNe (a : α) (_ : Fintype { b // b ≠ a }) : Fintype α :=
  Fintype.ofBijective (Sum.elim ((↑) : { b // b = a } → α) ((↑) : { b // b ≠ a } → α)) <| by
    /-
      α : Type u_1
      β : Type u_2
      a : α
      x✝ : Fintype (Subtype fun b => Ne b a)
      ⊢ Function.Bijective (Sum.elim Subtype.val Subtype.val)
    -/
    classical exact (Equiv.sumCompl (· = a)).bijective
    /-
      🎉 no goals
    -/


theorem image_subtype_ne_univ_eq_image_erase [Fintype α] [DecidableEq β] (k : β) (b : α → β) :
    image (fun i : { a // b a ≠ k } => b ↑i) univ = (image b univ).erase k := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq β
    k : β
    b : α → β
    ⊢ Eq (Finset.image (fun i => b ↑i) Finset.univ) ((Finset.image b Finset.univ). …
  -/
  apply subset_antisymm
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq β
      k : β
      b : α → β
      ⊢ HasSubset.Subset (Finset.image (fun i => b ↑i) Finset.univ) ((Finset.image b …
    -/
  · rw [image_subset_iff]
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq β
      k : β
      b : α → β
      ⊢ ∀ (x : Subtype fun a => Ne (b a) k), Membership.mem Finset.univ x → Membersh …
    -/
    intro i _
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq β
      k : β
      b : α → β
      i : Subtype fun a => Ne (b a) k
      a✝ : Membership.mem Finset.univ i
      ⊢ Membership.mem ((Finset.image b Finset.univ).erase k) (b ↑i)
    -/
    apply mem_erase_of_ne_of_mem i.2 (mem_image_of_mem _ (mem_univ _))
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq β
      k : β
      b : α → β
      ⊢ HasSubset.Subset ((Finset.image b Finset.univ).erase k) (Finset.image (fun i …
    -/
  · intro i hi
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq β
      k : β
      b : α → β
      i : β
      hi : Membership.mem ((Finset.image b Finset.univ).erase k) i
      ⊢ Membership.mem (Finset.image (fun i => b ↑i) Finset.univ) i
    -/
    rw [mem_image]
    /-
      case a
      α : Type u_1
      β : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq β
      k : β
      b : α → β
      i : β
      hi : Membership.mem ((Finset.image b Finset.univ).erase k) i
      ⊢ Exists fun a => And (Membership.mem Finset.univ a) (Eq (b ↑a) i)
    -/
    rcases mem_image.1 (erase_subset _ _ hi) with ⟨a, _, ha⟩
    /-
      case a.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq β
      k : β
      b : α → β
      i : β
      hi : Membership.mem ((Finset.image b Finset.univ).erase k) i
      a : α
      left✝ : Membership.mem Finset.univ a
      ha : Eq (b a) i
      ⊢ Exists fun a => And (Membership.mem Finset.univ a) (Eq (b ↑a) i)
    -/
    subst ha
    /-
      case a.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : Fintype α
      inst✝ : DecidableEq β
      k : β
      b : α → β
      a : α
      left✝ : Membership.mem Finset.univ a
      hi : Membership.mem ((Finset.image b Finset.univ).erase k) (b a)
      ⊢ Exists fun a_1 => And (Membership.mem Finset.univ a_1) (Eq (b ↑a_1) (b a))
    -/
    exact ⟨⟨a, ne_of_mem_erase hi⟩, mem_univ _, rfl⟩
    /-
      🎉 no goals
    -/


theorem image_subtype_univ_ssubset_image_univ [Fintype α] [DecidableEq β] (k : β) (b : α → β)
    (hk : k ∈ Finset.image b univ) (p : β → Prop) [DecidablePred p] (hp : ¬p k) :
    image (fun i : { a // p (b a) } => b ↑i) univ ⊂ image b univ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq β
    k : β
    b : α → β
    hk : Membership.mem (Finset.image b Finset.univ) k
    p : β → Prop
    inst✝ : DecidablePred p
    hp : Not (p k)
    ⊢ HasSSubset.SSubset (Finset.image (fun i => b ↑i) Finset.univ) (Finset.image  …
  -/
  constructor
    /-
      case left
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq β
      k : β
      b : α → β
      hk : Membership.mem (Finset.image b Finset.univ) k
      p : β → Prop
      inst✝ : DecidablePred p
      hp : Not (p k)
      ⊢ HasSubset.Subset (Finset.image (fun i => b ↑i) Finset.univ) (Finset.image b  …
    -/
  · intro x hx
    /-
      case left
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq β
      k : β
      b : α → β
      hk : Membership.mem (Finset.image b Finset.univ) k
      p : β → Prop
      inst✝ : DecidablePred p
      hp : Not (p k)
      x : β
      hx : Membership.mem (Finset.image (fun i => b ↑i) Finset.univ) x
      ⊢ Membership.mem (Finset.image b Finset.univ) x
    -/
    rcases mem_image.1 hx with ⟨y, _, hy⟩
    /-
      case left.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq β
      k : β
      b : α → β
      hk : Membership.mem (Finset.image b Finset.univ) k
      p : β → Prop
      inst✝ : DecidablePred p
      hp : Not (p k)
      x : β
      hx : Membership.mem (Finset.image (fun i => b ↑i) Finset.univ) x
      y : Subtype fun a => p (b a)
      left✝ : Membership.mem Finset.univ y
      hy : Eq (b ↑y) x
      ⊢ Membership.mem (Finset.image b Finset.univ) x
    -/
    exact hy ▸ mem_image_of_mem b (mem_univ (y : α))
    /-
      🎉 no goals
    -/
    /-
      case right
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq β
      k : β
      b : α → β
      hk : Membership.mem (Finset.image b Finset.univ) k
      p : β → Prop
      inst✝ : DecidablePred p
      hp : Not (p k)
      ⊢ Not (HasSubset.Subset (Finset.image b Finset.univ) (Finset.image (fun i => b …
    -/
  · intro h
    /-
      case right
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq β
      k : β
      b : α → β
      hk : Membership.mem (Finset.image b Finset.univ) k
      p : β → Prop
      inst✝ : DecidablePred p
      hp : Not (p k)
      h : HasSubset.Subset (Finset.image b Finset.univ) (Finset.image (fun i => b ↑i …
      ⊢ False
    -/
    rw [mem_image] at hk
    /-
      case right
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq β
      k : β
      b : α → β
      hk : Exists fun a => And (Membership.mem Finset.univ a) (Eq (b a) k)
      p : β → Prop
      inst✝ : DecidablePred p
      hp : Not (p k)
      h : HasSubset.Subset (Finset.image b Finset.univ) (Finset.image (fun i => b ↑i …
      ⊢ False
    -/
    rcases hk with ⟨k', _, hk'⟩
    /-
      case right.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq β
      k : β
      b : α → β
      p : β → Prop
      inst✝ : DecidablePred p
      hp : Not (p k)
      h : HasSubset.Subset (Finset.image b Finset.univ) (Finset.image (fun i => b ↑i …
      k' : α
      left✝ : Membership.mem Finset.univ k'
      hk' : Eq (b k') k
      ⊢ False
    -/
    subst hk'
    /-
      case right.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq β
      b : α → β
      p : β → Prop
      inst✝ : DecidablePred p
      h : HasSubset.Subset (Finset.image b Finset.univ) (Finset.image (fun i => b ↑i …
      k' : α
      left✝ : Membership.mem Finset.univ k'
      hp : Not (p (b k'))
      ⊢ False
    -/
    have := h (mem_image_of_mem b (mem_univ k'))
    /-
      case right.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq β
      b : α → β
      p : β → Prop
      inst✝ : DecidablePred p
      h : HasSubset.Subset (Finset.image b Finset.univ) (Finset.image (fun i => b ↑i …
      k' : α
      left✝ : Membership.mem Finset.univ k'
      hp : Not (p (b k'))
      this : Membership.mem (Finset.image (fun i => b ↑i) Finset.univ) (b k')
      ⊢ False
    -/
    rw [mem_image] at this
    /-
      case right.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq β
      b : α → β
      p : β → Prop
      inst✝ : DecidablePred p
      h : HasSubset.Subset (Finset.image b Finset.univ) (Finset.image (fun i => b ↑i …
      k' : α
      left✝ : Membership.mem Finset.univ k'
      hp : Not (p (b k'))
      this : Exists fun a => And (Membership.mem Finset.univ a) (Eq (b ↑a) (b k'))
      ⊢ False
    -/
    rcases this with ⟨j, _, hj'⟩
    /-
      case right.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : DecidableEq β
      b : α → β
      p : β → Prop
      inst✝ : DecidablePred p
      h : HasSubset.Subset (Finset.image b Finset.univ) (Finset.image (fun i => b ↑i …
      k' : α
      left✝¹ : Membership.mem Finset.univ k'
      hp : Not (p (b k'))
      j : Subtype fun a => p (b a)
      left✝ : Membership.mem Finset.univ j
      hj' : Eq (b ↑j) (b k')
      ⊢ False
    -/
    exact hp (hj' ▸ j.2)
    /-
      🎉 no goals
    -/


/-- Any injection from a finset `s` in a fintype `α` to a finset `t` of the same cardinality as `α`
can be extended to a bijection between `α` and `t`. -/
theorem Finset.exists_equiv_extend_of_card_eq [Fintype α] [DecidableEq β] {t : Finset β}
    (hαt : Fintype.card α = #t) {s : Finset α} {f : α → β} (hfst : Finset.image f s ⊆ t)
    (hfs : Set.InjOn f s) : ∃ g : α ≃ t, ∀ i ∈ s, (g i : β) = f i := by
  classical
    induction' s using Finset.induction with a s has H generalizing f
    · obtain ⟨e⟩ : Nonempty (α ≃ ↥t) := by rwa [← Fintype.card_eq, Fintype.card_coe]
      use e
      simp
    have hfst' : Finset.image f s ⊆ t := (Finset.image_mono _ (s.subset_insert a)).trans hfst
    have hfs' : Set.InjOn f s := hfs.mono (s.subset_insert a)
    obtain ⟨g', hg'⟩ := H hfst' hfs'
    have hfat : f a ∈ t := hfst (mem_image_of_mem _ (s.mem_insert_self a))
    use g'.trans (Equiv.swap (⟨f a, hfat⟩ : t) (g' a))
    simp_rw [mem_insert]
    rintro i (rfl | hi)
    · simp
    rw [Equiv.trans_apply, Equiv.swap_apply_of_ne_of_ne, hg' _ hi]
    · exact
        ne_of_apply_ne Subtype.val
          (ne_of_eq_of_ne (hg' _ hi) <|
            hfs.ne (subset_insert _ _ hi) (mem_insert_self _ _) <| ne_of_mem_of_not_mem hi has)
    · exact g'.injective.ne (ne_of_mem_of_not_mem hi has)


/-- Any injection from a set `s` in a fintype `α` to a finset `t` of the same cardinality as `α`
can be extended to a bijection between `α` and `t`. -/
theorem Set.MapsTo.exists_equiv_extend_of_card_eq [Fintype α] {t : Finset β}
    (hαt : Fintype.card α = #t) {s : Set α} {f : α → β} (hfst : s.MapsTo f t)
    (hfs : Set.InjOn f s) : ∃ g : α ≃ t, ∀ i ∈ s, (g i : β) = f i := by
  classical
    let s' : Finset α := s.toFinset
    have hfst' : s'.image f ⊆ t := by simpa [s', ← Finset.coe_subset] using hfst
    have hfs' : Set.InjOn f s' := by simpa [s'] using hfs
    obtain ⟨g, hg⟩ := Finset.exists_equiv_extend_of_card_eq hαt hfst' hfs'
    refine ⟨g, fun i hi => ?_⟩
    apply hg
    simpa [s'] using hi


theorem Fintype.card_subtype_or (p q : α → Prop) [Fintype { x // p x }] [Fintype { x // q x }]
    [Fintype { x // p x ∨ q x }] :
    Fintype.card { x // p x ∨ q x } ≤ Fintype.card { x // p x } + Fintype.card { x // q x } := by
  classical
    convert Fintype.card_le_of_embedding (subtypeOrLeftEmbedding p q)
    rw [Fintype.card_sum]


theorem Fintype.card_subtype_or_disjoint (p q : α → Prop) (h : Disjoint p q) [Fintype { x // p x }]
    [Fintype { x // q x }] [Fintype { x // p x ∨ q x }] :
    Fintype.card { x // p x ∨ q x } = Fintype.card { x // p x } + Fintype.card { x // q x } := by
  classical
    convert Fintype.card_congr (subtypeOrEquiv p q h)
    simp


@[simp]
theorem infinite_sum : Infinite (α ⊕ β) ↔ Infinite α ∨ Infinite β := by
  /-
    α : Type u_1
    β : Type u_2
    ⊢ Iff (Infinite (Sum α β)) (Or (Infinite α) (Infinite β))
  -/
  refine ⟨fun H => ?_, fun H => H.elim (@Sum.infinite_of_left α β) (@Sum.infinite_of_right α β)⟩
  /-
    α : Type u_1
    β : Type u_2
    H : Infinite (Sum α β)
    ⊢ Or (Infinite α) (Infinite β)
  -/
  contrapose! H; haveI := fintypeOfNotInfinite H.1; haveI := fintypeOfNotInfinite H.2
  /-
    α : Type u_1
    β : Type u_2
    H : And (Not (Infinite α)) (Not (Infinite β))
    this✝ : Fintype α
    this : Fintype β
    ⊢ Not (Infinite (Sum α β))
  -/
  exact Infinite.false
  /-
    🎉 no goals
  -/


