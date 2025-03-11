/-- A finite partition of `a : α` is a pairwise disjoint finite set of elements whose supremum is
`a`. We forbid `⊥` as a part. -/
@[ext]
structure Finpartition [Lattice α] [OrderBot α] (a : α) where
  -- Porting note: Docstrings added
  /-- The elements of the finite partition of `a` -/
  parts : Finset α
  /-- The partition is supremum-independent -/
  protected supIndep : parts.SupIndep id
  /-- The supremum of the partition is `a` -/
  sup_parts : parts.sup id = a
  /-- No element of the partition is bottom -/
  not_bot_mem : ⊥ ∉ parts
  deriving DecidableEq


/-- A `Finpartition` constructor which does not insist on `⊥` not being a part. -/
@[simps]
def ofErase [DecidableEq α] {a : α} (parts : Finset α) (sup_indep : parts.SupIndep id)
    (sup_parts : parts.sup id = a) : Finpartition a where
  parts := parts.erase ⊥
  supIndep := sup_indep.subset (erase_subset _ _)
  sup_parts := (sup_erase_bot _).trans sup_parts
  not_bot_mem := not_mem_erase _ _


/-- A `Finpartition` constructor from a bigger existing finpartition. -/
@[simps]
def ofSubset {a b : α} (P : Finpartition a) {parts : Finset α} (subset : parts ⊆ P.parts)
    (sup_parts : parts.sup id = b) : Finpartition b :=
  { parts := parts
    supIndep := P.supIndep.subset subset
    sup_parts := sup_parts
    not_bot_mem := fun h ↦ P.not_bot_mem (subset h) }


/-- Changes the type of a finpartition to an equal one. -/
@[simps]
def copy {a b : α} (P : Finpartition a) (h : a = b) : Finpartition b where
  parts := P.parts
  supIndep := P.supIndep
  sup_parts := h ▸ P.sup_parts
  not_bot_mem := P.not_bot_mem


/-- Transfer a finpartition over an order isomorphism. -/
def map {β : Type*} [Lattice β] [OrderBot β] {a : α} (e : α ≃o β) (P : Finpartition a) :
    Finpartition (e a) where
  parts := P.parts.map e
  supIndep u hu _ hb hbu _ hx hxu := by
    /-
      α : Type u_1
      inst✝³ : Lattice α
      inst✝² : OrderBot α
      β : Type u_2
      inst✝¹ : Lattice β
      inst✝ : OrderBot β
      a : α
      e : OrderIso α β
      P : Finpartition a
      u : Finset β
      hu : HasSubset.Subset u (Finset.map (↑e).toEmbedding P.parts)
      x✝¹ : β
      hb : Membership.mem (Finset.map (↑e).toEmbedding P.parts) x✝¹
      hbu : Not (Membership.mem u x✝¹)
      x✝ : β
      hx : LE.le x✝ (id x✝¹)
      hxu : LE.le x✝ (u.sup id)
      ⊢ LE.le x✝ Bot.bot
    -/
    rw [← map_symm_subset] at hu
    /-
      α : Type u_1
      inst✝³ : Lattice α
      inst✝² : OrderBot α
      β : Type u_2
      inst✝¹ : Lattice β
      inst✝ : OrderBot β
      a : α
      e : OrderIso α β
      P : Finpartition a
      u : Finset β
      hu : HasSubset.Subset (Finset.map (↑e).symm.toEmbedding u) P.parts
      x✝¹ : β
      hb : Membership.mem (Finset.map (↑e).toEmbedding P.parts) x✝¹
      hbu : Not (Membership.mem u x✝¹)
      x✝ : β
      hx : LE.le x✝ (id x✝¹)
      hxu : LE.le x✝ (u.sup id)
      ⊢ LE.le x✝ Bot.bot
    -/
    simp only [mem_map_equiv] at hb
    /-
      α : Type u_1
      inst✝³ : Lattice α
      inst✝² : OrderBot α
      β : Type u_2
      inst✝¹ : Lattice β
      inst✝ : OrderBot β
      a : α
      e : OrderIso α β
      P : Finpartition a
      u : Finset β
      hu : HasSubset.Subset (Finset.map (↑e).symm.toEmbedding u) P.parts
      x✝¹ : β
      hbu : Not (Membership.mem u x✝¹)
      x✝ : β
      hx : LE.le x✝ (id x✝¹)
      hxu : LE.le x✝ (u.sup id)
      hb : Membership.mem P.parts ((↑e).symm x✝¹)
      ⊢ LE.le x✝ Bot.bot
    -/
    have := P.supIndep hu hb (by simp [hbu]) (map_rel e.symm hx) ?_
      /-
        case refine_2
        α : Type u_1
        inst✝³ : Lattice α
        inst✝² : OrderBot α
        β : Type u_2
        inst✝¹ : Lattice β
        inst✝ : OrderBot β
        a : α
        e : OrderIso α β
        P : Finpartition a
        u : Finset β
        hu : HasSubset.Subset (Finset.map (↑e).symm.toEmbedding u) P.parts
        x✝¹ : β
        hbu : Not (Membership.mem u x✝¹)
        x✝ : β
        hx : LE.le x✝ (id x✝¹)
        hxu : LE.le x✝ (u.sup id)
        hb : Membership.mem P.parts ((↑e).symm x✝¹)
        this : LE.le (e.symm x✝) Bot.bot
        ⊢ LE.le x✝ Bot.bot
      -/
    · rw [← e.symm.map_bot] at this
      /-
        case refine_2
        α : Type u_1
        inst✝³ : Lattice α
        inst✝² : OrderBot α
        β : Type u_2
        inst✝¹ : Lattice β
        inst✝ : OrderBot β
        a : α
        e : OrderIso α β
        P : Finpartition a
        u : Finset β
        hu : HasSubset.Subset (Finset.map (↑e).symm.toEmbedding u) P.parts
        x✝¹ : β
        hbu : Not (Membership.mem u x✝¹)
        x✝ : β
        hx : LE.le x✝ (id x✝¹)
        hxu : LE.le x✝ (u.sup id)
        hb : Membership.mem P.parts ((↑e).symm x✝¹)
        this : LE.le (e.symm x✝) (e.symm Bot.bot)
        ⊢ LE.le x✝ Bot.bot
      -/
      exact e.symm.map_rel_iff.mp this
      /-
        🎉 no goals
      -/
      /-
        case refine_1
        α : Type u_1
        inst✝³ : Lattice α
        inst✝² : OrderBot α
        β : Type u_2
        inst✝¹ : Lattice β
        inst✝ : OrderBot β
        a : α
        e : OrderIso α β
        P : Finpartition a
        u : Finset β
        hu : HasSubset.Subset (Finset.map (↑e).symm.toEmbedding u) P.parts
        x✝¹ : β
        hbu : Not (Membership.mem u x✝¹)
        x✝ : β
        hx : LE.le x✝ (id x✝¹)
        hxu : LE.le x✝ (u.sup id)
        hb : Membership.mem P.parts ((↑e).symm x✝¹)
        ⊢ LE.le (e.symm x✝) ((Finset.map (↑e).symm.toEmbedding u).sup id)
      -/
    · convert e.symm.map_rel_iff.mpr hxu
      /-
        case h.e'_4
        α : Type u_1
        inst✝³ : Lattice α
        inst✝² : OrderBot α
        β : Type u_2
        inst✝¹ : Lattice β
        inst✝ : OrderBot β
        a : α
        e : OrderIso α β
        P : Finpartition a
        u : Finset β
        hu : HasSubset.Subset (Finset.map (↑e).symm.toEmbedding u) P.parts
        x✝¹ : β
        hbu : Not (Membership.mem u x✝¹)
        x✝ : β
        hx : LE.le x✝ (id x✝¹)
        hxu : LE.le x✝ (u.sup id)
        hb : Membership.mem P.parts ((↑e).symm x✝¹)
        ⊢ Eq ((Finset.map (↑e).symm.toEmbedding u).sup id) (e.symm (u.sup id))
      -/
      rw [map_finset_sup, sup_map]
      /-
        case h.e'_4
        α : Type u_1
        inst✝³ : Lattice α
        inst✝² : OrderBot α
        β : Type u_2
        inst✝¹ : Lattice β
        inst✝ : OrderBot β
        a : α
        e : OrderIso α β
        P : Finpartition a
        u : Finset β
        hu : HasSubset.Subset (Finset.map (↑e).symm.toEmbedding u) P.parts
        x✝¹ : β
        hbu : Not (Membership.mem u x✝¹)
        x✝ : β
        hx : LE.le x✝ (id x✝¹)
        hxu : LE.le x✝ (u.sup id)
        hb : Membership.mem P.parts ((↑e).symm x✝¹)
        ⊢ Eq (u.sup (Function.comp id ⇑(↑e).symm.toEmbedding)) (u.sup (Function.comp ( …
      -/
      rfl
      /-
        🎉 no goals
      -/
                  /-
                    α : Type u_1
                    inst✝³ : Lattice α
                    inst✝² : OrderBot α
                    β : Type u_2
                    inst✝¹ : Lattice β
                    inst✝ : OrderBot β
                    a : α
                    e : OrderIso α β
                    P : Finpartition a
                    ⊢ Eq ((Finset.map (↑e).toEmbedding P.parts).sup id) (e a)
                  -/
  sup_parts := by simp [← P.sup_parts]
                  /-
                    🎉 no goals
                  -/
  not_bot_mem := by
    /-
      α : Type u_1
      inst✝³ : Lattice α
      inst✝² : OrderBot α
      β : Type u_2
      inst✝¹ : Lattice β
      inst✝ : OrderBot β
      a : α
      e : OrderIso α β
      P : Finpartition a
      ⊢ Not (Membership.mem (Finset.map (↑e).toEmbedding P.parts) Bot.bot)
    -/
    rw [mem_map_equiv]
    /-
      α : Type u_1
      inst✝³ : Lattice α
      inst✝² : OrderBot α
      β : Type u_2
      inst✝¹ : Lattice β
      inst✝ : OrderBot β
      a : α
      e : OrderIso α β
      P : Finpartition a
      ⊢ Not (Membership.mem P.parts ((↑e).symm Bot.bot))
    -/
    convert P.not_bot_mem
    /-
      case h.e'_1.h.e'_5
      α : Type u_1
      inst✝³ : Lattice α
      inst✝² : OrderBot α
      β : Type u_2
      inst✝¹ : Lattice β
      inst✝ : OrderBot β
      a : α
      e : OrderIso α β
      P : Finpartition a
      ⊢ Eq ((↑e).symm Bot.bot) Bot.bot
    -/
    exact e.symm.map_bot
    /-
      🎉 no goals
    -/


@[simp]
theorem parts_map {β : Type*} [Lattice β] [OrderBot β] {a : α} {e : α ≃o β} {P : Finpartition a} :
    (P.map e).parts = P.parts.map e := rfl


/-- The empty finpartition. -/
@[simps]
protected def empty : Finpartition (⊥ : α) where
  parts := ∅
  supIndep := supIndep_empty _
  sup_parts := Finset.sup_empty
  not_bot_mem := not_mem_empty ⊥


instance : Inhabited (Finpartition (⊥ : α)) :=
  ⟨Finpartition.empty α⟩


@[simp]
theorem default_eq_empty : (default : Finpartition (⊥ : α)) = Finpartition.empty α :=
  rfl


/-- The finpartition in one part, aka indiscrete finpartition. -/
@[simps]
def indiscrete (ha : a ≠ ⊥) : Finpartition a where
  parts := {a}
  supIndep := supIndep_singleton _ _
  sup_parts := Finset.sup_singleton
  not_bot_mem h := ha (mem_singleton.1 h).symm


protected theorem le {b : α} (hb : b ∈ P.parts) : b ≤ a :=
  (le_sup hb).trans P.sup_parts.le


theorem ne_bot {b : α} (hb : b ∈ P.parts) : b ≠ ⊥ := by
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    a : α
    P : Finpartition a
    b : α
    hb : Membership.mem P.parts b
    ⊢ Ne b Bot.bot
  -/
  intro h
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    a : α
    P : Finpartition a
    b : α
    hb : Membership.mem P.parts b
    h : Eq b Bot.bot
    ⊢ False
  -/
  refine P.not_bot_mem (?_)
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    a : α
    P : Finpartition a
    b : α
    hb : Membership.mem P.parts b
    h : Eq b Bot.bot
    ⊢ Membership.mem P.parts Bot.bot
  -/
  rw [h] at hb
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    a : α
    P : Finpartition a
    b : α
    hb : Membership.mem P.parts Bot.bot
    h : Eq b Bot.bot
    ⊢ Membership.mem P.parts Bot.bot
  -/
  exact hb
  /-
    🎉 no goals
  -/


protected theorem disjoint : (P.parts : Set α).PairwiseDisjoint id :=
  P.supIndep.pairwiseDisjoint


theorem parts_eq_empty_iff : P.parts = ∅ ↔ a = ⊥ := by
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    a : α
    P : Finpartition a
    ⊢ Iff (Eq P.parts EmptyCollection.emptyCollection) (Eq a Bot.bot)
  -/
  simp_rw [← P.sup_parts]
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    a : α
    P : Finpartition a
    ⊢ Iff (Eq P.parts EmptyCollection.emptyCollection) (Eq (P.parts.sup id) Bot.bot)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ eq_empty_iff_forall_not_mem.2 fun b hb ↦ P.not_bot_mem ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : OrderBot α
      a : α
      P : Finpartition a
      h : Eq P.parts EmptyCollection.emptyCollection
      ⊢ Eq (P.parts.sup id) Bot.bot
    -/
  · rw [h]
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : OrderBot α
      a : α
      P : Finpartition a
      h : Eq P.parts EmptyCollection.emptyCollection
      ⊢ Eq (EmptyCollection.emptyCollection.sup id) Bot.bot
    -/
    exact Finset.sup_empty
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : OrderBot α
      a : α
      P : Finpartition a
      h : Eq (P.parts.sup id) Bot.bot
      b : α
      hb : Membership.mem P.parts b
      ⊢ Membership.mem P.parts Bot.bot
    -/
  · rwa [← le_bot_iff.1 ((le_sup hb).trans h.le)]
    /-
      🎉 no goals
    -/


theorem parts_nonempty_iff : P.parts.Nonempty ↔ a ≠ ⊥ := by
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    a : α
    P : Finpartition a
    ⊢ Iff P.parts.Nonempty (Ne a Bot.bot)
  -/
  rw [nonempty_iff_ne_empty, not_iff_not, parts_eq_empty_iff]
  /-
    🎉 no goals
  -/


theorem parts_nonempty (P : Finpartition a) (ha : a ≠ ⊥) : P.parts.Nonempty :=
  parts_nonempty_iff.2 ha


instance : Unique (Finpartition (⊥ : α)) :=
  { (inferInstance : Inhabited (Finpartition (⊥ : α))) with
    uniq := fun P ↦ by
      /-
        α : Type u_1
        inst✝¹ : Lattice α
        inst✝ : OrderBot α
        a : α
        P✝ : Finpartition a
        P : Finpartition Bot.bot
        ⊢ Eq P Inhabited.default
      -/
      ext a
      /-
        case parts.h
        α : Type u_1
        inst✝¹ : Lattice α
        inst✝ : OrderBot α
        a✝ : α
        P✝ : Finpartition a✝
        P : Finpartition Bot.bot
        a : α
        ⊢ Iff (Membership.mem P.parts a) (Membership.mem Inhabited.default.parts a)
      -/
      exact iff_of_false (fun h ↦ P.ne_bot h <| le_bot_iff.1 <| P.le h) (not_mem_empty a) }
      /-
        🎉 no goals
      -/

-- See note [reducible non instances]

/-- There's a unique partition of an atom. -/
abbrev _root_.IsAtom.uniqueFinpartition (ha : IsAtom a) : Unique (Finpartition a) where
  default := indiscrete ha.1
  uniq P := by
    have h : ∀ b ∈ P.parts, b = a := fun _ hb ↦
      (ha.le_iff.mp <| P.le hb).resolve_left (P.ne_bot hb)
    /-
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : OrderBot α
      a : α
      P✝ : Finpartition a
      ha : IsAtom a
      P : Finpartition a
      h : ∀ (b : α), Membership.mem P.parts b → Eq b a
      ⊢ Eq P Inhabited.default
    -/
    ext b
    /-
      case parts.h
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : OrderBot α
      a : α
      P✝ : Finpartition a
      ha : IsAtom a
      P : Finpartition a
      h : ∀ (b : α), Membership.mem P.parts b → Eq b a
      b : α
      ⊢ Iff (Membership.mem P.parts b) (Membership.mem Inhabited.default.parts b)
    -/
    refine Iff.trans ⟨h b, ?_⟩ mem_singleton.symm
    /-
      case parts.h
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : OrderBot α
      a : α
      P✝ : Finpartition a
      ha : IsAtom a
      P : Finpartition a
      h : ∀ (b : α), Membership.mem P.parts b → Eq b a
      b : α
      ⊢ Eq b a → Membership.mem P.parts b
    -/
    rintro rfl
    /-
      case parts.h
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : OrderBot α
      b : α
      P✝ : Finpartition b
      ha : IsAtom b
      P : Finpartition b
      h : ∀ (b_1 : α), Membership.mem P.parts b_1 → Eq b_1 b
      ⊢ Membership.mem P.parts b
    -/
    obtain ⟨c, hc⟩ := P.parts_nonempty ha.1
    /-
      case parts.h.intro
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : OrderBot α
      b : α
      P✝ : Finpartition b
      ha : IsAtom b
      P : Finpartition b
      h : ∀ (b_1 : α), Membership.mem P.parts b_1 → Eq b_1 b
      c : α
      hc : Membership.mem P.parts c
      ⊢ Membership.mem P.parts b
    -/
    simp_rw [← h c hc]
    /-
      case parts.h.intro
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : OrderBot α
      b : α
      P✝ : Finpartition b
      ha : IsAtom b
      P : Finpartition b
      h : ∀ (b_1 : α), Membership.mem P.parts b_1 → Eq b_1 b
      c : α
      hc : Membership.mem P.parts c
      ⊢ Membership.mem P.parts c
    -/
    exact hc
    /-
      🎉 no goals
    -/


instance [Fintype α] [DecidableEq α] (a : α) : Fintype (Finpartition a) :=
  @Fintype.ofSurjective { p : Finset α // p.SupIndep id ∧ p.sup id = a ∧ ⊥ ∉ p } (Finpartition a) _
    (Subtype.fintype _) (fun i ↦ ⟨i.1, i.2.1, i.2.2.1, i.2.2.2⟩) fun ⟨_, y, z, w⟩ ↦
    ⟨⟨_, y, z, w⟩, rfl⟩


/-- We say that `P ≤ Q` if `P` refines `Q`: each part of `P` is less than some part of `Q`. -/
instance : LE (Finpartition a) :=
  ⟨fun P Q ↦ ∀ ⦃b⦄, b ∈ P.parts → ∃ c ∈ Q.parts, b ≤ c⟩


instance : PartialOrder (Finpartition a) :=
  { (inferInstance : LE (Finpartition a)) with
    le_refl := fun _ b hb ↦ ⟨b, hb, le_rfl⟩
    le_trans := fun _ Q R hPQ hQR b hb ↦ by
      /-
        α : Type u_1
        inst✝¹ : Lattice α
        inst✝ : OrderBot α
        a : α
        P x✝ Q R : Finpartition a
        hPQ : LE.le x✝ Q
        hQR : LE.le Q R
        b : α
        hb : Membership.mem x✝.parts b
        ⊢ Exists fun c => And (Membership.mem R.parts c) (LE.le b c)
      -/
      obtain ⟨c, hc, hbc⟩ := hPQ hb
      /-
        case intro.intro
        α : Type u_1
        inst✝¹ : Lattice α
        inst✝ : OrderBot α
        a : α
        P x✝ Q R : Finpartition a
        hPQ : LE.le x✝ Q
        hQR : LE.le Q R
        b : α
        hb : Membership.mem x✝.parts b
        c : α
        hc : Membership.mem Q.parts c
        hbc : LE.le b c
        ⊢ Exists fun c => And (Membership.mem R.parts c) (LE.le b c)
      -/
      obtain ⟨d, hd, hcd⟩ := hQR hc
      /-
        case intro.intro.intro.intro
        α : Type u_1
        inst✝¹ : Lattice α
        inst✝ : OrderBot α
        a : α
        P x✝ Q R : Finpartition a
        hPQ : LE.le x✝ Q
        hQR : LE.le Q R
        b : α
        hb : Membership.mem x✝.parts b
        c : α
        hc : Membership.mem Q.parts c
        hbc : LE.le b c
        d : α
        hd : Membership.mem R.parts d
        hcd : LE.le c d
        ⊢ Exists fun c => And (Membership.mem R.parts c) (LE.le b c)
      -/
      exact ⟨d, hd, hbc.trans hcd⟩
      /-
        🎉 no goals
      -/
    le_antisymm := fun P Q hPQ hQP ↦ by
      /-
        α : Type u_1
        inst✝¹ : Lattice α
        inst✝ : OrderBot α
        a : α
        P✝ P Q : Finpartition a
        hPQ : LE.le P Q
        hQP : LE.le Q P
        ⊢ Eq P Q
      -/
      ext b
      /-
        case parts.h
        α : Type u_1
        inst✝¹ : Lattice α
        inst✝ : OrderBot α
        a : α
        P✝ P Q : Finpartition a
        hPQ : LE.le P Q
        hQP : LE.le Q P
        b : α
        ⊢ Iff (Membership.mem P.parts b) (Membership.mem Q.parts b)
      -/
      refine ⟨fun hb ↦ ?_, fun hb ↦ ?_⟩
        /-
          case parts.h.refine_1
          α : Type u_1
          inst✝¹ : Lattice α
          inst✝ : OrderBot α
          a : α
          P✝ P Q : Finpartition a
          hPQ : LE.le P Q
          hQP : LE.le Q P
          b : α
          hb : Membership.mem P.parts b
          ⊢ Membership.mem Q.parts b
        -/
      · obtain ⟨c, hc, hbc⟩ := hPQ hb
        /-
          case parts.h.refine_1.intro.intro
          α : Type u_1
          inst✝¹ : Lattice α
          inst✝ : OrderBot α
          a : α
          P✝ P Q : Finpartition a
          hPQ : LE.le P Q
          hQP : LE.le Q P
          b : α
          hb : Membership.mem P.parts b
          c : α
          hc : Membership.mem Q.parts c
          hbc : LE.le b c
          ⊢ Membership.mem Q.parts b
        -/
        obtain ⟨d, hd, hcd⟩ := hQP hc
        /-
          case parts.h.refine_1.intro.intro.intro.intro
          α : Type u_1
          inst✝¹ : Lattice α
          inst✝ : OrderBot α
          a : α
          P✝ P Q : Finpartition a
          hPQ : LE.le P Q
          hQP : LE.le Q P
          b : α
          hb : Membership.mem P.parts b
          c : α
          hc : Membership.mem Q.parts c
          hbc : LE.le b c
          d : α
          hd : Membership.mem P.parts d
          hcd : LE.le c d
          ⊢ Membership.mem Q.parts b
        -/
        rwa [hbc.antisymm]
        /-
          case parts.h.refine_1.intro.intro.intro.intro
          α : Type u_1
          inst✝¹ : Lattice α
          inst✝ : OrderBot α
          a : α
          P✝ P Q : Finpartition a
          hPQ : LE.le P Q
          hQP : LE.le Q P
          b : α
          hb : Membership.mem P.parts b
          c : α
          hc : Membership.mem Q.parts c
          hbc : LE.le b c
          d : α
          hd : Membership.mem P.parts d
          hcd : LE.le c d
          ⊢ LE.le c b
        -/
        rwa [P.disjoint.eq_of_le hb hd (P.ne_bot hb) (hbc.trans hcd)]
        /-
          🎉 no goals
        -/
        /-
          case parts.h.refine_2
          α : Type u_1
          inst✝¹ : Lattice α
          inst✝ : OrderBot α
          a : α
          P✝ P Q : Finpartition a
          hPQ : LE.le P Q
          hQP : LE.le Q P
          b : α
          hb : Membership.mem Q.parts b
          ⊢ Membership.mem P.parts b
        -/
      · obtain ⟨c, hc, hbc⟩ := hQP hb
        /-
          case parts.h.refine_2.intro.intro
          α : Type u_1
          inst✝¹ : Lattice α
          inst✝ : OrderBot α
          a : α
          P✝ P Q : Finpartition a
          hPQ : LE.le P Q
          hQP : LE.le Q P
          b : α
          hb : Membership.mem Q.parts b
          c : α
          hc : Membership.mem P.parts c
          hbc : LE.le b c
          ⊢ Membership.mem P.parts b
        -/
        obtain ⟨d, hd, hcd⟩ := hPQ hc
        /-
          case parts.h.refine_2.intro.intro.intro.intro
          α : Type u_1
          inst✝¹ : Lattice α
          inst✝ : OrderBot α
          a : α
          P✝ P Q : Finpartition a
          hPQ : LE.le P Q
          hQP : LE.le Q P
          b : α
          hb : Membership.mem Q.parts b
          c : α
          hc : Membership.mem P.parts c
          hbc : LE.le b c
          d : α
          hd : Membership.mem Q.parts d
          hcd : LE.le c d
          ⊢ Membership.mem P.parts b
        -/
        rwa [hbc.antisymm]
        /-
          case parts.h.refine_2.intro.intro.intro.intro
          α : Type u_1
          inst✝¹ : Lattice α
          inst✝ : OrderBot α
          a : α
          P✝ P Q : Finpartition a
          hPQ : LE.le P Q
          hQP : LE.le Q P
          b : α
          hb : Membership.mem Q.parts b
          c : α
          hc : Membership.mem P.parts c
          hbc : LE.le b c
          d : α
          hd : Membership.mem Q.parts d
          hcd : LE.le c d
          ⊢ LE.le c b
        -/
        rwa [Q.disjoint.eq_of_le hb hd (Q.ne_bot hb) (hbc.trans hcd)] }
        /-
          🎉 no goals
        -/


instance [Decidable (a = ⊥)] : OrderTop (Finpartition a) where
  top := if ha : a = ⊥ then (Finpartition.empty α).copy ha.symm else indiscrete ha
  le_top P := by
    /-
      α : Type u_1
      inst✝² : Lattice α
      inst✝¹ : OrderBot α
      a : α
      P✝ : Finpartition a
      inst✝ : Decidable (Eq a Bot.bot)
      P : Finpartition a
      ⊢ LE.le P Top.top
    -/
    split_ifs with h
      /-
        case pos
        α : Type u_1
        inst✝² : Lattice α
        inst✝¹ : OrderBot α
        a : α
        P✝ : Finpartition a
        inst✝ : Decidable (Eq a Bot.bot)
        P : Finpartition a
        h : Eq a Bot.bot
        ⊢ LE.le P Top.top
      -/
    · intro x hx
      /-
        case pos
        α : Type u_1
        inst✝² : Lattice α
        inst✝¹ : OrderBot α
        a : α
        P✝ : Finpartition a
        inst✝ : Decidable (Eq a Bot.bot)
        P : Finpartition a
        h : Eq a Bot.bot
        x : α
        hx : Membership.mem P.parts x
        ⊢ Exists fun c => And (Membership.mem Top.top.parts c) (LE.le x c)
      -/
      simpa [h, P.ne_bot hx] using P.le hx
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝² : Lattice α
        inst✝¹ : OrderBot α
        a : α
        P✝ : Finpartition a
        inst✝ : Decidable (Eq a Bot.bot)
        P : Finpartition a
        h : Not (Eq a Bot.bot)
        ⊢ LE.le P Top.top
      -/
    · exact fun b hb ↦ ⟨a, mem_singleton_self _, P.le hb⟩
      /-
        🎉 no goals
      -/


theorem parts_top_subset (a : α) [Decidable (a = ⊥)] : (⊤ : Finpartition a).parts ⊆ {a} := by
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    a : α
    inst✝ : Decidable (Eq a Bot.bot)
    ⊢ HasSubset.Subset Top.top.parts (Singleton.singleton a)
  -/
  intro b hb
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    a : α
    inst✝ : Decidable (Eq a Bot.bot)
    b : α
    hb : Membership.mem Top.top.parts b
    ⊢ Membership.mem (Singleton.singleton a) b
  -/
  have hb : b ∈ Finpartition.parts (dite _ _ _) := hb
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    a : α
    inst✝ : Decidable (Eq a Bot.bot)
    b : α
    hb✝ : Membership.mem Top.top.parts b
    hb : Membership.mem (dite (Eq a Bot.bot) (fun ha => (Finpartition.empty α).cop …
    ⊢ Membership.mem (Singleton.singleton a) b
  -/
  split_ifs at hb
    /-
      case pos
      α : Type u_1
      inst✝² : Lattice α
      inst✝¹ : OrderBot α
      a : α
      inst✝ : Decidable (Eq a Bot.bot)
      b : α
      hb✝ : Membership.mem Top.top.parts b
      h✝ : Eq a Bot.bot
      hb : Membership.mem ((Finpartition.empty α).copy ⋯).parts b
      ⊢ Membership.mem (Singleton.singleton a) b
    -/
  · simp only [copy_parts, empty_parts, not_mem_empty] at hb
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝² : Lattice α
      inst✝¹ : OrderBot α
      a : α
      inst✝ : Decidable (Eq a Bot.bot)
      b : α
      hb✝ : Membership.mem Top.top.parts b
      h✝ : Not (Eq a Bot.bot)
      hb : Membership.mem (Finpartition.indiscrete h✝).parts b
      ⊢ Membership.mem (Singleton.singleton a) b
    -/
  · exact hb
    /-
      🎉 no goals
    -/


theorem parts_top_subsingleton (a : α) [Decidable (a = ⊥)] :
    ((⊤ : Finpartition a).parts : Set α).Subsingleton :=
  Set.subsingleton_of_subset_singleton fun _ hb ↦ mem_singleton.1 <| parts_top_subset _ hb

-- TODO: this instance takes double-exponential time to generate all partitions, find a faster way

instance [DecidableEq α] {s : Finset α} : Fintype (Finpartition s) where
  elems := s.powerset.powerset.image
    fun ps ↦ if h : ps.sup id = s ∧ ⊥ ∉ ps ∧ ps.SupIndep id then ⟨ps, h.2.2, h.1, h.2.1⟩ else ⊤
  complete P := by
    /-
      α : Type u_1
      inst✝² : Lattice α
      inst✝¹ : OrderBot α
      a : α
      P✝ : Finpartition a
      inst✝ : DecidableEq α
      s : Finset α
      P : Finpartition s
      ⊢ Membership.mem (Finset.image (fun ps => dite (And (Eq (ps.sup id) s) (And (N …
    -/
    refine mem_image.mpr ⟨P.parts, ?_, ?_⟩
      /-
        case refine_1
        α : Type u_1
        inst✝² : Lattice α
        inst✝¹ : OrderBot α
        a : α
        P✝ : Finpartition a
        inst✝ : DecidableEq α
        s : Finset α
        P : Finpartition s
        ⊢ Membership.mem s.powerset.powerset P.parts
      -/
    · rw [mem_powerset]; intro p hp; rw [mem_powerset]; exact P.le hp
                                                        /-
                                                          🎉 no goals
                                                        -/
      /-
        case refine_2
        α : Type u_1
        inst✝² : Lattice α
        inst✝¹ : OrderBot α
        a : α
        P✝ : Finpartition a
        inst✝ : DecidableEq α
        s : Finset α
        P : Finpartition s
        ⊢ Eq (dite (And (Eq (P.parts.sup id) s) (And (Not (Membership.mem P.parts Bot. …
      -/
    · simp [P.supIndep, P.sup_parts, P.not_bot_mem, -bot_eq_empty]
      /-
        🎉 no goals
      -/


instance : Min (Finpartition a) :=
  ⟨fun P Q ↦
    ofErase ((P.parts ×ˢ Q.parts).image fun bc ↦ bc.1 ⊓ bc.2)
      (by
        /-
          α : Type u_1
          inst✝² : DistribLattice α
          inst✝¹ : OrderBot α
          inst✝ : DecidableEq α
          a b c : α
          P Q : Finpartition a
          ⊢ (Finset.image (fun bc => Min.min bc.1 bc.2) (SProd.sprod P.parts Q.parts)).S …
        -/
        rw [supIndep_iff_disjoint_erase]
        simp only [mem_image, and_imp, exists_prop, forall_exists_index, id, Prod.exists,
          mem_product, Finset.disjoint_sup_right, mem_erase, Ne]
        /-
          α : Type u_1
          inst✝² : DistribLattice α
          inst✝¹ : OrderBot α
          inst✝ : DecidableEq α
          a b c : α
          P Q : Finpartition a
          ⊢ ∀ (i x x_1 : α), Membership.mem P.parts x → Membership.mem Q.parts x_1 → Eq  …
        -/
        rintro _ x₁ y₁ hx₁ hy₁ rfl _ h x₂ y₂ hx₂ hy₂ rfl
        /-
          α : Type u_1
          inst✝² : DistribLattice α
          inst✝¹ : OrderBot α
          inst✝ : DecidableEq α
          a b c : α
          P Q : Finpartition a
          x₁ y₁ : α
          hx₁ : Membership.mem P.parts x₁
          hy₁ : Membership.mem Q.parts y₁
          x₂ y₂ : α
          hx₂ : Membership.mem P.parts x₂
          hy₂ : Membership.mem Q.parts y₂
          h : Not (Eq (Min.min x₂ y₂) (Min.min x₁ y₁))
          ⊢ Disjoint (Min.min x₁ y₁) (Min.min x₂ y₂)
        -/
        rcases eq_or_ne x₁ x₂ with (rfl | xdiff)
          /-
            case inl
            α : Type u_1
            inst✝² : DistribLattice α
            inst✝¹ : OrderBot α
            inst✝ : DecidableEq α
            a b c : α
            P Q : Finpartition a
            x₁ y₁ : α
            hx₁ : Membership.mem P.parts x₁
            hy₁ : Membership.mem Q.parts y₁
            y₂ : α
            hy₂ : Membership.mem Q.parts y₂
            hx₂ : Membership.mem P.parts x₁
            h : Not (Eq (Min.min x₁ y₂) (Min.min x₁ y₁))
            ⊢ Disjoint (Min.min x₁ y₁) (Min.min x₁ y₂)
          -/
        · refine Disjoint.mono inf_le_right inf_le_right (Q.disjoint hy₁ hy₂ ?_)
          /-
            case inl
            α : Type u_1
            inst✝² : DistribLattice α
            inst✝¹ : OrderBot α
            inst✝ : DecidableEq α
            a b c : α
            P Q : Finpartition a
            x₁ y₁ : α
            hx₁ : Membership.mem P.parts x₁
            hy₁ : Membership.mem Q.parts y₁
            y₂ : α
            hy₂ : Membership.mem Q.parts y₂
            hx₂ : Membership.mem P.parts x₁
            h : Not (Eq (Min.min x₁ y₂) (Min.min x₁ y₁))
            ⊢ Ne y₁ y₂
          -/
          intro t
          /-
            case inl
            α : Type u_1
            inst✝² : DistribLattice α
            inst✝¹ : OrderBot α
            inst✝ : DecidableEq α
            a b c : α
            P Q : Finpartition a
            x₁ y₁ : α
            hx₁ : Membership.mem P.parts x₁
            hy₁ : Membership.mem Q.parts y₁
            y₂ : α
            hy₂ : Membership.mem Q.parts y₂
            hx₂ : Membership.mem P.parts x₁
            h : Not (Eq (Min.min x₁ y₂) (Min.min x₁ y₁))
            t : Eq y₁ y₂
            ⊢ False
          -/
          simp [t] at h
          /-
            🎉 no goals
          -/
        /-
          case inr
          α : Type u_1
          inst✝² : DistribLattice α
          inst✝¹ : OrderBot α
          inst✝ : DecidableEq α
          a b c : α
          P Q : Finpartition a
          x₁ y₁ : α
          hx₁ : Membership.mem P.parts x₁
          hy₁ : Membership.mem Q.parts y₁
          x₂ y₂ : α
          hx₂ : Membership.mem P.parts x₂
          hy₂ : Membership.mem Q.parts y₂
          h : Not (Eq (Min.min x₂ y₂) (Min.min x₁ y₁))
          xdiff : Ne x₁ x₂
          ⊢ Disjoint (Min.min x₁ y₁) (Min.min x₂ y₂)
        -/
        exact Disjoint.mono inf_le_left inf_le_left (P.disjoint hx₁ hx₂ xdiff))
        /-
          🎉 no goals
        -/
      (by
        /-
          α : Type u_1
          inst✝² : DistribLattice α
          inst✝¹ : OrderBot α
          inst✝ : DecidableEq α
          a b c : α
          P Q : Finpartition a
          ⊢ Eq ((Finset.image (fun bc => Min.min bc.1 bc.2) (SProd.sprod P.parts Q.parts …
        -/
        rw [sup_image, id_comp, sup_product_left]
        /-
          α : Type u_1
          inst✝² : DistribLattice α
          inst✝¹ : OrderBot α
          inst✝ : DecidableEq α
          a b c : α
          P Q : Finpartition a
          ⊢ Eq (P.parts.sup fun i => Q.parts.sup fun i' => Min.min { fst := i, snd := i' …
        -/
        trans P.parts.sup id ⊓ Q.parts.sup id
          /-
            α : Type u_1
            inst✝² : DistribLattice α
            inst✝¹ : OrderBot α
            inst✝ : DecidableEq α
            a b c : α
            P Q : Finpartition a
            ⊢ Eq (P.parts.sup fun i => Q.parts.sup fun i' => Min.min { fst := i, snd := i' …
          -/
        · simp_rw [Finset.sup_inf_distrib_right, Finset.sup_inf_distrib_left]
          /-
            α : Type u_1
            inst✝² : DistribLattice α
            inst✝¹ : OrderBot α
            inst✝ : DecidableEq α
            a b c : α
            P Q : Finpartition a
            ⊢ Eq (P.parts.sup fun i => Q.parts.sup fun i' => Min.min i i') (P.parts.sup fu …
          -/
          rfl
          /-
            🎉 no goals
          -/
          /-
            α : Type u_1
            inst✝² : DistribLattice α
            inst✝¹ : OrderBot α
            inst✝ : DecidableEq α
            a b c : α
            P Q : Finpartition a
            ⊢ Eq (Min.min (P.parts.sup id) (Q.parts.sup id)) a
          -/
        · rw [P.sup_parts, Q.sup_parts, inf_idem])⟩
          /-
            🎉 no goals
          -/


@[simp]
theorem parts_inf (P Q : Finpartition a) :
    (P ⊓ Q).parts = ((P.parts ×ˢ Q.parts).image fun bc : α × α ↦ bc.1 ⊓ bc.2).erase ⊥ :=
  rfl


instance : SemilatticeInf (Finpartition a) :=
  { inf := Min.min
    inf_le_left := fun P Q b hb ↦ by
      /-
        α : Type u_1
        inst✝² : DistribLattice α
        inst✝¹ : OrderBot α
        inst✝ : DecidableEq α
        a b✝ c : α
        P Q : Finpartition a
        b : α
        hb : Membership.mem (Min.min P Q).parts b
        ⊢ Exists fun c => And (Membership.mem P.parts c) (LE.le b c)
      -/
      obtain ⟨c, hc, rfl⟩ := mem_image.1 (mem_of_mem_erase hb)
      /-
        case intro.intro
        α : Type u_1
        inst✝² : DistribLattice α
        inst✝¹ : OrderBot α
        inst✝ : DecidableEq α
        a b c✝ : α
        P Q : Finpartition a
        c : Prod α α
        hc : Membership.mem (SProd.sprod P.parts Q.parts) c
        hb : Membership.mem (Min.min P Q).parts (Min.min c.1 c.2)
        ⊢ Exists fun c_1 => And (Membership.mem P.parts c_1) (LE.le (Min.min c.1 c.2)  …
      -/
      rw [mem_product] at hc
      /-
        case intro.intro
        α : Type u_1
        inst✝² : DistribLattice α
        inst✝¹ : OrderBot α
        inst✝ : DecidableEq α
        a b c✝ : α
        P Q : Finpartition a
        c : Prod α α
        hc : And (Membership.mem P.parts c.1) (Membership.mem Q.parts c.2)
        hb : Membership.mem (Min.min P Q).parts (Min.min c.1 c.2)
        ⊢ Exists fun c_1 => And (Membership.mem P.parts c_1) (LE.le (Min.min c.1 c.2)  …
      -/
      exact ⟨c.1, hc.1, inf_le_left⟩
      /-
        🎉 no goals
      -/
    inf_le_right := fun P Q b hb ↦ by
      /-
        α : Type u_1
        inst✝² : DistribLattice α
        inst✝¹ : OrderBot α
        inst✝ : DecidableEq α
        a b✝ c : α
        P Q : Finpartition a
        b : α
        hb : Membership.mem (Min.min P Q).parts b
        ⊢ Exists fun c => And (Membership.mem Q.parts c) (LE.le b c)
      -/
      obtain ⟨c, hc, rfl⟩ := mem_image.1 (mem_of_mem_erase hb)
      /-
        case intro.intro
        α : Type u_1
        inst✝² : DistribLattice α
        inst✝¹ : OrderBot α
        inst✝ : DecidableEq α
        a b c✝ : α
        P Q : Finpartition a
        c : Prod α α
        hc : Membership.mem (SProd.sprod P.parts Q.parts) c
        hb : Membership.mem (Min.min P Q).parts (Min.min c.1 c.2)
        ⊢ Exists fun c_1 => And (Membership.mem Q.parts c_1) (LE.le (Min.min c.1 c.2)  …
      -/
      rw [mem_product] at hc
      /-
        case intro.intro
        α : Type u_1
        inst✝² : DistribLattice α
        inst✝¹ : OrderBot α
        inst✝ : DecidableEq α
        a b c✝ : α
        P Q : Finpartition a
        c : Prod α α
        hc : And (Membership.mem P.parts c.1) (Membership.mem Q.parts c.2)
        hb : Membership.mem (Min.min P Q).parts (Min.min c.1 c.2)
        ⊢ Exists fun c_1 => And (Membership.mem Q.parts c_1) (LE.le (Min.min c.1 c.2)  …
      -/
      exact ⟨c.2, hc.2, inf_le_right⟩
      /-
        🎉 no goals
      -/
    le_inf := fun P Q R hPQ hPR b hb ↦ by
      /-
        α : Type u_1
        inst✝² : DistribLattice α
        inst✝¹ : OrderBot α
        inst✝ : DecidableEq α
        a b✝ c : α
        P Q R : Finpartition a
        hPQ : LE.le P Q
        hPR : LE.le P R
        b : α
        hb : Membership.mem P.parts b
        ⊢ Exists fun c => And (Membership.mem (Min.min Q R).parts c) (LE.le b c)
      -/
      obtain ⟨c, hc, hbc⟩ := hPQ hb
      /-
        case intro.intro
        α : Type u_1
        inst✝² : DistribLattice α
        inst✝¹ : OrderBot α
        inst✝ : DecidableEq α
        a b✝ c✝ : α
        P Q R : Finpartition a
        hPQ : LE.le P Q
        hPR : LE.le P R
        b : α
        hb : Membership.mem P.parts b
        c : α
        hc : Membership.mem Q.parts c
        hbc : LE.le b c
        ⊢ Exists fun c => And (Membership.mem (Min.min Q R).parts c) (LE.le b c)
      -/
      obtain ⟨d, hd, hbd⟩ := hPR hb
      /-
        case intro.intro.intro.intro
        α : Type u_1
        inst✝² : DistribLattice α
        inst✝¹ : OrderBot α
        inst✝ : DecidableEq α
        a b✝ c✝ : α
        P Q R : Finpartition a
        hPQ : LE.le P Q
        hPR : LE.le P R
        b : α
        hb : Membership.mem P.parts b
        c : α
        hc : Membership.mem Q.parts c
        hbc : LE.le b c
        d : α
        hd : Membership.mem R.parts d
        hbd : LE.le b d
        ⊢ Exists fun c => And (Membership.mem (Min.min Q R).parts c) (LE.le b c)
      -/
      have h := _root_.le_inf hbc hbd
      refine
        ⟨c ⊓ d,
          mem_erase_of_ne_of_mem (ne_bot_of_le_ne_bot (P.ne_bot hb) h)
            (mem_image.2 ⟨(c, d), mem_product.2 ⟨hc, hd⟩, rfl⟩),
          h⟩ }


theorem exists_le_of_le {a b : α} {P Q : Finpartition a} (h : P ≤ Q) (hb : b ∈ Q.parts) :
    ∃ c ∈ P.parts, c ≤ b := by
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    a b : α
    P Q : Finpartition a
    h : LE.le P Q
    hb : Membership.mem Q.parts b
    ⊢ Exists fun c => And (Membership.mem P.parts c) (LE.le c b)
  -/
  by_contra H
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    a b : α
    P Q : Finpartition a
    h : LE.le P Q
    hb : Membership.mem Q.parts b
    H : Not (Exists fun c => And (Membership.mem P.parts c) (LE.le c b))
    ⊢ False
  -/
  refine Q.ne_bot hb (disjoint_self.1 <| Disjoint.mono_right (Q.le hb) ?_)
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    a b : α
    P Q : Finpartition a
    h : LE.le P Q
    hb : Membership.mem Q.parts b
    H : Not (Exists fun c => And (Membership.mem P.parts c) (LE.le c b))
    ⊢ Disjoint b a
  -/
  rw [← P.sup_parts, Finset.disjoint_sup_right]
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    a b : α
    P Q : Finpartition a
    h : LE.le P Q
    hb : Membership.mem Q.parts b
    H : Not (Exists fun c => And (Membership.mem P.parts c) (LE.le c b))
    ⊢ ∀ ⦃i : α⦄, Membership.mem P.parts i → Disjoint b (id i)
  -/
  rintro c hc
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    a b : α
    P Q : Finpartition a
    h : LE.le P Q
    hb : Membership.mem Q.parts b
    H : Not (Exists fun c => And (Membership.mem P.parts c) (LE.le c b))
    c : α
    hc : Membership.mem P.parts c
    ⊢ Disjoint b (id c)
  -/
  obtain ⟨d, hd, hcd⟩ := h hc
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    a b : α
    P Q : Finpartition a
    h : LE.le P Q
    hb : Membership.mem Q.parts b
    H : Not (Exists fun c => And (Membership.mem P.parts c) (LE.le c b))
    c : α
    hc : Membership.mem P.parts c
    d : α
    hd : Membership.mem Q.parts d
    hcd : LE.le c d
    ⊢ Disjoint b (id c)
  -/
  refine (Q.disjoint hb hd ?_).mono_right hcd
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    a b : α
    P Q : Finpartition a
    h : LE.le P Q
    hb : Membership.mem Q.parts b
    H : Not (Exists fun c => And (Membership.mem P.parts c) (LE.le c b))
    c : α
    hc : Membership.mem P.parts c
    d : α
    hd : Membership.mem Q.parts d
    hcd : LE.le c d
    ⊢ Ne b d
  -/
  rintro rfl
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    a b : α
    P Q : Finpartition a
    h : LE.le P Q
    hb : Membership.mem Q.parts b
    H : Not (Exists fun c => And (Membership.mem P.parts c) (LE.le c b))
    c : α
    hc : Membership.mem P.parts c
    hd : Membership.mem Q.parts b
    hcd : LE.le c b
    ⊢ False
  -/
  simp only [not_exists, not_and] at H
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    a b : α
    P Q : Finpartition a
    h : LE.le P Q
    hb : Membership.mem Q.parts b
    c : α
    hc : Membership.mem P.parts c
    hd : Membership.mem Q.parts b
    hcd : LE.le c b
    H : ∀ (x : α), Membership.mem P.parts x → Not (LE.le x b)
    ⊢ False
  -/
  exact H _ hc hcd
  /-
    🎉 no goals
  -/


theorem card_mono {a : α} {P Q : Finpartition a} (h : P ≤ Q) : #Q.parts ≤ #P.parts := by
  classical
    have : ∀ b ∈ Q.parts, ∃ c ∈ P.parts, c ≤ b := fun b ↦ exists_le_of_le h
    choose f hP hf using this
    rw [← card_attach]
    refine card_le_card_of_injOn (fun b ↦ f _ b.2) (fun b _ ↦ hP _ b.2) fun b _ c _ h ↦ ?_
    exact
      Subtype.coe_injective
        (Q.disjoint.elim b.2 c.2 fun H ↦
          P.ne_bot (hP _ b.2) <| disjoint_self.1 <| H.mono (hf _ b.2) <| h.le.trans <| hf _ c.2)


/-- Given a finpartition `P` of `a` and finpartitions of each part of `P`, this yields the
finpartition of `a` obtained by juxtaposing all the subpartitions. -/
@[simps]
def bind (P : Finpartition a) (Q : ∀ i ∈ P.parts, Finpartition i) : Finpartition a where
  parts := P.parts.attach.biUnion fun i ↦ (Q i.1 i.2).parts
  supIndep := by
    /-
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a b c : α
      P✝ : Finpartition a
      Q✝ : (i : α) → Membership.mem P✝.parts i → Finpartition i
      P : Finpartition a
      Q : (i : α) → Membership.mem P.parts i → Finpartition i
      ⊢ (P.parts.attach.biUnion fun i => (Q ↑i ⋯).parts).SupIndep id
    -/
    rw [supIndep_iff_pairwiseDisjoint]
    /-
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a b c : α
      P✝ : Finpartition a
      Q✝ : (i : α) → Membership.mem P✝.parts i → Finpartition i
      P : Finpartition a
      Q : (i : α) → Membership.mem P.parts i → Finpartition i
      ⊢ (↑(P.parts.attach.biUnion fun i => (Q ↑i ⋯).parts)).PairwiseDisjoint id
    -/
    rintro a ha b hb h
    /-
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a✝ b✝ c : α
      P✝ : Finpartition a✝
      Q✝ : (i : α) → Membership.mem P✝.parts i → Finpartition i
      P : Finpartition a✝
      Q : (i : α) → Membership.mem P.parts i → Finpartition i
      a : α
      ha : Membership.mem (↑(P.parts.attach.biUnion fun i => (Q ↑i ⋯).parts)) a
      b : α
      hb : Membership.mem (↑(P.parts.attach.biUnion fun i => (Q ↑i ⋯).parts)) b
      h : Ne a b
      ⊢ Function.onFun Disjoint id a b
    -/
    rw [Finset.mem_coe, Finset.mem_biUnion] at ha hb
    /-
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a✝ b✝ c : α
      P✝ : Finpartition a✝
      Q✝ : (i : α) → Membership.mem P✝.parts i → Finpartition i
      P : Finpartition a✝
      Q : (i : α) → Membership.mem P.parts i → Finpartition i
      a : α
      ha : Exists fun a_1 => And (Membership.mem P.parts.attach a_1) (Membership.mem …
      b : α
      hb : Exists fun a => And (Membership.mem P.parts.attach a) (Membership.mem (Q  …
      h : Ne a b
      ⊢ Function.onFun Disjoint id a b
    -/
    obtain ⟨⟨A, hA⟩, -, ha⟩ := ha
    /-
      case intro.mk.intro
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a✝ b✝ c : α
      P✝ : Finpartition a✝
      Q✝ : (i : α) → Membership.mem P✝.parts i → Finpartition i
      P : Finpartition a✝
      Q : (i : α) → Membership.mem P.parts i → Finpartition i
      a b : α
      hb : Exists fun a => And (Membership.mem P.parts.attach a) (Membership.mem (Q  …
      h : Ne a b
      A : α
      hA : Membership.mem P.parts A
      ha : Membership.mem (Q ↑⟨A, hA⟩ ⋯).parts a
      ⊢ Function.onFun Disjoint id a b
    -/
    obtain ⟨⟨B, hB⟩, -, hb⟩ := hb
    /-
      case intro.mk.intro.intro.mk.intro
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a✝ b✝ c : α
      P✝ : Finpartition a✝
      Q✝ : (i : α) → Membership.mem P✝.parts i → Finpartition i
      P : Finpartition a✝
      Q : (i : α) → Membership.mem P.parts i → Finpartition i
      a b : α
      h : Ne a b
      A : α
      hA : Membership.mem P.parts A
      ha : Membership.mem (Q ↑⟨A, hA⟩ ⋯).parts a
      B : α
      hB : Membership.mem P.parts B
      hb : Membership.mem (Q ↑⟨B, hB⟩ ⋯).parts b
      ⊢ Function.onFun Disjoint id a b
    -/
    obtain rfl | hAB := eq_or_ne A B
      /-
        case intro.mk.intro.intro.mk.intro.inl
        α : Type u_1
        inst✝² : DistribLattice α
        inst✝¹ : OrderBot α
        inst✝ : DecidableEq α
        a✝ b✝ c : α
        P✝ : Finpartition a✝
        Q✝ : (i : α) → Membership.mem P✝.parts i → Finpartition i
        P : Finpartition a✝
        Q : (i : α) → Membership.mem P.parts i → Finpartition i
        a b : α
        h : Ne a b
        A : α
        hA : Membership.mem P.parts A
        ha : Membership.mem (Q ↑⟨A, hA⟩ ⋯).parts a
        hB : Membership.mem P.parts A
        hb : Membership.mem (Q ↑⟨A, hB⟩ ⋯).parts b
        ⊢ Function.onFun Disjoint id a b
      -/
    · exact (Q A hA).disjoint ha hb h
      /-
        🎉 no goals
      -/
      /-
        case intro.mk.intro.intro.mk.intro.inr
        α : Type u_1
        inst✝² : DistribLattice α
        inst✝¹ : OrderBot α
        inst✝ : DecidableEq α
        a✝ b✝ c : α
        P✝ : Finpartition a✝
        Q✝ : (i : α) → Membership.mem P✝.parts i → Finpartition i
        P : Finpartition a✝
        Q : (i : α) → Membership.mem P.parts i → Finpartition i
        a b : α
        h : Ne a b
        A : α
        hA : Membership.mem P.parts A
        ha : Membership.mem (Q ↑⟨A, hA⟩ ⋯).parts a
        B : α
        hB : Membership.mem P.parts B
        hb : Membership.mem (Q ↑⟨B, hB⟩ ⋯).parts b
        hAB : Ne A B
        ⊢ Function.onFun Disjoint id a b
      -/
    · exact (P.disjoint hA hB hAB).mono ((Q A hA).le ha) ((Q B hB).le hb)
      /-
        🎉 no goals
      -/
  sup_parts := by
    /-
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a b c : α
      P✝ : Finpartition a
      Q✝ : (i : α) → Membership.mem P✝.parts i → Finpartition i
      P : Finpartition a
      Q : (i : α) → Membership.mem P.parts i → Finpartition i
      ⊢ Eq ((P.parts.attach.biUnion fun i => (Q ↑i ⋯).parts).sup id) a
    -/
    simp_rw [sup_biUnion]
    /-
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a b c : α
      P✝ : Finpartition a
      Q✝ : (i : α) → Membership.mem P✝.parts i → Finpartition i
      P : Finpartition a
      Q : (i : α) → Membership.mem P.parts i → Finpartition i
      ⊢ Eq (P.parts.attach.sup fun x => (Q ↑x ⋯).parts.sup id) a
    -/
    trans (sup P.parts id)
      /-
        α : Type u_1
        inst✝² : DistribLattice α
        inst✝¹ : OrderBot α
        inst✝ : DecidableEq α
        a b c : α
        P✝ : Finpartition a
        Q✝ : (i : α) → Membership.mem P✝.parts i → Finpartition i
        P : Finpartition a
        Q : (i : α) → Membership.mem P.parts i → Finpartition i
        ⊢ Eq (P.parts.attach.sup fun x => (Q ↑x ⋯).parts.sup id) (P.parts.sup id)
      -/
    · rw [eq_comm, ← Finset.sup_attach]
      /-
        α : Type u_1
        inst✝² : DistribLattice α
        inst✝¹ : OrderBot α
        inst✝ : DecidableEq α
        a b c : α
        P✝ : Finpartition a
        Q✝ : (i : α) → Membership.mem P✝.parts i → Finpartition i
        P : Finpartition a
        Q : (i : α) → Membership.mem P.parts i → Finpartition i
        ⊢ Eq (P.parts.attach.sup fun x => id ↑x) (P.parts.attach.sup fun x => (Q ↑x ⋯) …
      -/
      exact sup_congr rfl fun b _hb ↦ (Q b.1 b.2).sup_parts.symm
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        inst✝² : DistribLattice α
        inst✝¹ : OrderBot α
        inst✝ : DecidableEq α
        a b c : α
        P✝ : Finpartition a
        Q✝ : (i : α) → Membership.mem P✝.parts i → Finpartition i
        P : Finpartition a
        Q : (i : α) → Membership.mem P.parts i → Finpartition i
        ⊢ Eq (P.parts.sup id) a
      -/
    · exact P.sup_parts
      /-
        🎉 no goals
      -/
  not_bot_mem h := by
    /-
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a b c : α
      P✝ : Finpartition a
      Q✝ : (i : α) → Membership.mem P✝.parts i → Finpartition i
      P : Finpartition a
      Q : (i : α) → Membership.mem P.parts i → Finpartition i
      h : Membership.mem (P.parts.attach.biUnion fun i => (Q ↑i ⋯).parts) Bot.bot
      ⊢ False
    -/
    rw [Finset.mem_biUnion] at h
    /-
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a b c : α
      P✝ : Finpartition a
      Q✝ : (i : α) → Membership.mem P✝.parts i → Finpartition i
      P : Finpartition a
      Q : (i : α) → Membership.mem P.parts i → Finpartition i
      h : Exists fun a_1 => And (Membership.mem P.parts.attach a_1) (Membership.mem  …
      ⊢ False
    -/
    obtain ⟨⟨A, hA⟩, -, h⟩ := h
    /-
      case intro.mk.intro
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a b c : α
      P✝ : Finpartition a
      Q✝ : (i : α) → Membership.mem P✝.parts i → Finpartition i
      P : Finpartition a
      Q : (i : α) → Membership.mem P.parts i → Finpartition i
      A : α
      hA : Membership.mem P.parts A
      h : Membership.mem (Q ↑⟨A, hA⟩ ⋯).parts Bot.bot
      ⊢ False
    -/
    exact (Q A hA).not_bot_mem h
    /-
      🎉 no goals
    -/


theorem mem_bind : b ∈ (P.bind Q).parts ↔ ∃ A hA, b ∈ (Q A hA).parts := by
  /-
    α : Type u_1
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    a b : α
    P : Finpartition a
    Q : (i : α) → Membership.mem P.parts i → Finpartition i
    ⊢ Iff (Membership.mem (P.bind Q).parts b) (Exists fun A => Exists fun hA => Me …
  -/
  rw [bind, mem_biUnion]
  /-
    α : Type u_1
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    a b : α
    P : Finpartition a
    Q : (i : α) → Membership.mem P.parts i → Finpartition i
    ⊢ Iff (Exists fun a_1 => And (Membership.mem P.parts.attach a_1) (Membership.m …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a b : α
      P : Finpartition a
      Q : (i : α) → Membership.mem P.parts i → Finpartition i
      ⊢ (Exists fun a_1 => And (Membership.mem P.parts.attach a_1) (Membership.mem ( …
    -/
  · rintro ⟨⟨A, hA⟩, -, h⟩
    /-
      case mp.intro.mk.intro
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a b : α
      P : Finpartition a
      Q : (i : α) → Membership.mem P.parts i → Finpartition i
      A : α
      hA : Membership.mem P.parts A
      h : Membership.mem (Q ↑⟨A, hA⟩ ⋯).parts b
      ⊢ Exists fun A => Exists fun hA => Membership.mem (Q A hA).parts b
    -/
    exact ⟨A, hA, h⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a b : α
      P : Finpartition a
      Q : (i : α) → Membership.mem P.parts i → Finpartition i
      ⊢ (Exists fun A => Exists fun hA => Membership.mem (Q A hA).parts b) → Exists  …
    -/
  · rintro ⟨A, hA, h⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a b : α
      P : Finpartition a
      Q : (i : α) → Membership.mem P.parts i → Finpartition i
      A : α
      hA : Membership.mem P.parts A
      h : Membership.mem (Q A hA).parts b
      ⊢ Exists fun a_1 => And (Membership.mem P.parts.attach a_1) (Membership.mem (Q …
    -/
    exact ⟨⟨A, hA⟩, mem_attach _ ⟨A, hA⟩, h⟩
    /-
      🎉 no goals
    -/


theorem card_bind (Q : ∀ i ∈ P.parts, Finpartition i) :
    #(P.bind Q).parts = ∑ A ∈ P.parts.attach, #(Q _ A.2).parts := by
  /-
    α : Type u_1
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    a : α
    P : Finpartition a
    Q : (i : α) → Membership.mem P.parts i → Finpartition i
    ⊢ Eq (P.bind Q).parts.card (P.parts.attach.sum fun A => (Q ↑A ⋯).parts.card)
  -/
  apply card_biUnion
  /-
    case h
    α : Type u_1
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    a : α
    P : Finpartition a
    Q : (i : α) → Membership.mem P.parts i → Finpartition i
    ⊢ ∀ (x : Subtype fun x => Membership.mem P.parts x), Membership.mem P.parts.at …
  -/
  rintro ⟨b, hb⟩ - ⟨c, hc⟩ - hbc
  /-
    case h.mk.mk
    α : Type u_1
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    a : α
    P : Finpartition a
    Q : (i : α) → Membership.mem P.parts i → Finpartition i
    b : α
    hb : Membership.mem P.parts b
    c : α
    hc : Membership.mem P.parts c
    hbc : Ne ⟨b, hb⟩ ⟨c, hc⟩
    ⊢ Disjoint (Q ↑⟨b, hb⟩ ⋯).parts (Q ↑⟨c, hc⟩ ⋯).parts
  -/
  rw [Finset.disjoint_left]
  /-
    case h.mk.mk
    α : Type u_1
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    a : α
    P : Finpartition a
    Q : (i : α) → Membership.mem P.parts i → Finpartition i
    b : α
    hb : Membership.mem P.parts b
    c : α
    hc : Membership.mem P.parts c
    hbc : Ne ⟨b, hb⟩ ⟨c, hc⟩
    ⊢ ∀ ⦃a_1 : α⦄, Membership.mem (Q ↑⟨b, hb⟩ ⋯).parts a_1 → Not (Membership.mem ( …
  -/
  rintro d hdb hdc
  /-
    case h.mk.mk
    α : Type u_1
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    a : α
    P : Finpartition a
    Q : (i : α) → Membership.mem P.parts i → Finpartition i
    b : α
    hb : Membership.mem P.parts b
    c : α
    hc : Membership.mem P.parts c
    hbc : Ne ⟨b, hb⟩ ⟨c, hc⟩
    d : α
    hdb : Membership.mem (Q ↑⟨b, hb⟩ ⋯).parts d
    hdc : Membership.mem (Q ↑⟨c, hc⟩ ⋯).parts d
    ⊢ False
  -/
  rw [Ne, Subtype.mk_eq_mk] at hbc
  exact
    (Q b hb).ne_bot hdb
      (eq_bot_iff.2 <|
        (le_inf ((Q b hb).le hdb) <| (Q c hc).le hdc).trans <| (P.disjoint hb hc hbc).le_bot)


/-- Adds `b` to a finpartition of `a` to make a finpartition of `a ⊔ b`. -/
@[simps]
def extend (P : Finpartition a) (hb : b ≠ ⊥) (hab : Disjoint a b) (hc : a ⊔ b = c) :
    Finpartition c where
  parts := insert b P.parts
  supIndep := by
    /-
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a b c : α
      P : Finpartition a
      hb : Ne b Bot.bot
      hab : Disjoint a b
      hc : Eq (Max.max a b) c
      ⊢ (Insert.insert b P.parts).SupIndep id
    -/
    rw [supIndep_iff_pairwiseDisjoint, coe_insert]
    /-
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      a b c : α
      P : Finpartition a
      hb : Ne b Bot.bot
      hab : Disjoint a b
      hc : Eq (Max.max a b) c
      ⊢ (Insert.insert b ↑P.parts).PairwiseDisjoint id
    -/
    exact P.disjoint.insert fun d hd _ ↦ hab.symm.mono_right <| P.le hd
    /-
      🎉 no goals
    -/
                  /-
                    α : Type u_1
                    inst✝² : DistribLattice α
                    inst✝¹ : OrderBot α
                    inst✝ : DecidableEq α
                    a b c : α
                    P : Finpartition a
                    hb : Ne b Bot.bot
                    hab : Disjoint a b
                    hc : Eq (Max.max a b) c
                    ⊢ Eq ((Insert.insert b P.parts).sup id) c
                  -/
  sup_parts := by rwa [sup_insert, P.sup_parts, id, _root_.sup_comm]
                  /-
                    🎉 no goals
                  -/
  not_bot_mem h := (mem_insert.1 h).elim hb.symm P.not_bot_mem


theorem card_extend (P : Finpartition a) (b c : α) {hb : b ≠ ⊥} {hab : Disjoint a b}
    {hc : a ⊔ b = c} : #(P.extend hb hab hc).parts = #P.parts + 1 :=
  card_insert_of_not_mem fun h ↦ hb <| hab.symm.eq_bot_of_le <| P.le h


/-- Restricts a finpartition to avoid a given element. -/
@[simps!]
def avoid (b : α) : Finpartition (a \ b) :=
  ofErase
    (P.parts.image (· \ b))
    (P.disjoint.image_finset_of_le fun _ ↦ sdiff_le).supIndep
        /-
          α : Type u_1
          inst✝¹ : GeneralizedBooleanAlgebra α
          inst✝ : DecidableEq α
          a b✝ c : α
          P : Finpartition a
          b : α
          ⊢ Eq ((Finset.image (fun x => SDiff.sdiff x b) P.parts).sup id) (SDiff.sdiff a …
        -/
    (by rw [sup_image, id_comp, Finset.sup_sdiff_right, ← Function.id_def, P.sup_parts])
        /-
          🎉 no goals
        -/


@[simp]
theorem mem_avoid : c ∈ (P.avoid b).parts ↔ ∃ d ∈ P.parts, ¬d ≤ b ∧ d \ b = c := by
  simp only [avoid, ofErase, mem_erase, Ne, mem_image, exists_prop, ← exists_and_left,
    @and_left_comm (c ≠ ⊥)]
  /-
    α : Type u_1
    inst✝¹ : GeneralizedBooleanAlgebra α
    inst✝ : DecidableEq α
    a b c : α
    P : Finpartition a
    ⊢ Iff (Exists fun x => And (Membership.mem P.parts x) (And (Not (Eq c Bot.bot) …
  -/
  refine exists_congr fun d ↦ and_congr_right' <| and_congr_left ?_
  /-
    α : Type u_1
    inst✝¹ : GeneralizedBooleanAlgebra α
    inst✝ : DecidableEq α
    a b c : α
    P : Finpartition a
    d : α
    ⊢ Eq (SDiff.sdiff d b) c → Iff (Not (Eq c Bot.bot)) (Not (LE.le d b))
  -/
  rintro rfl
  /-
    α : Type u_1
    inst✝¹ : GeneralizedBooleanAlgebra α
    inst✝ : DecidableEq α
    a b : α
    P : Finpartition a
    d : α
    ⊢ Iff (Not (Eq (SDiff.sdiff d b) Bot.bot)) (Not (LE.le d b))
  -/
  rw [sdiff_eq_bot_iff]
  /-
    🎉 no goals
  -/


theorem nonempty_of_mem_parts {a : Finset α} (ha : a ∈ P.parts) : a.Nonempty :=
  nonempty_iff_ne_empty.2 <| P.ne_bot ha


lemma eq_of_mem_parts (ht : t ∈ P.parts) (hu : u ∈ P.parts) (hat : a ∈ t) (hau : a ∈ u) : t = u :=
  P.disjoint.elim ht hu <| not_disjoint_iff.2 ⟨a, hat, hau⟩


theorem exists_mem (ha : a ∈ s) : ∃ t ∈ P.parts, a ∈ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    a : α
    ha : Membership.mem s a
    ⊢ Exists fun t => And (Membership.mem P.parts t) (Membership.mem t a)
  -/
  simp_rw [← P.sup_parts] at ha
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    a : α
    ha : Membership.mem (P.parts.sup id) a
    ⊢ Exists fun t => And (Membership.mem P.parts t) (Membership.mem t a)
  -/
  exact mem_sup.1 ha
  /-
    🎉 no goals
  -/


theorem biUnion_parts : P.parts.biUnion id = s :=
  (sup_eq_biUnion _ _).symm.trans P.sup_parts


theorem existsUnique_mem (ha : a ∈ s) : ∃! t, t ∈ P.parts ∧ a ∈ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    a : α
    ha : Membership.mem s a
    ⊢ ExistsUnique fun t => And (Membership.mem P.parts t) (Membership.mem t a)
  -/
  obtain ⟨t, ht, ht'⟩ := P.exists_mem ha
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    a : α
    ha : Membership.mem s a
    t : Finset α
    ht : Membership.mem P.parts t
    ht' : Membership.mem t a
    ⊢ ExistsUnique fun t => And (Membership.mem P.parts t) (Membership.mem t a)
  -/
  refine ⟨t, ⟨ht, ht'⟩, ?_⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    a : α
    ha : Membership.mem s a
    t : Finset α
    ht : Membership.mem P.parts t
    ht' : Membership.mem t a
    ⊢ ∀ (y : Finset α), (fun t => And (Membership.mem P.parts t) (Membership.mem t …
  -/
  rintro u ⟨hu, hu'⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    a : α
    ha : Membership.mem s a
    t : Finset α
    ht : Membership.mem P.parts t
    ht' : Membership.mem t a
    u : Finset α
    hu : Membership.mem P.parts u
    hu' : Membership.mem u a
    ⊢ Eq u t
  -/
  exact P.eq_of_mem_parts hu ht hu' ht'
  /-
    🎉 no goals
  -/


/-- The part of the finpartition that `a` lies in. -/
def part (a : α) : Finset α := if ha : a ∈ s then choose (hp := P.existsUnique_mem ha) else ∅


                                                       /-
                                                         α : Type u_1
                                                         inst✝ : DecidableEq α
                                                         s : Finset α
                                                         P : Finpartition s
                                                         a : α
                                                         ha : Membership.mem s a
                                                         ⊢ Membership.mem P.parts (P.part a)
                                                       -/
lemma part_mem (ha : a ∈ s) : P.part a ∈ P.parts := by simp [part, ha, choose_mem]
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma mem_part (ha : a ∈ s) : a ∈ P.part a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    a : α
    ha : Membership.mem s a
    ⊢ Membership.mem (P.part a) a
  -/
  simp [part, ha, choose_property (p := fun s => a ∈ s) P.parts (P.existsUnique_mem ha)]
  /-
    🎉 no goals
  -/


lemma part_eq_of_mem (ht : t ∈ P.parts) (hat : a ∈ t) : P.part a = t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    P : Finpartition s
    a : α
    ht : Membership.mem P.parts t
    hat : Membership.mem t a
    ⊢ Eq (P.part a) t
  -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  apply P.eq_of_mem_parts (P.part_mem _) ht (P.mem_part _) hat <;> exact mem_of_subset (P.le ht) hat
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


lemma mem_part_iff_part_eq_part {b : α} (ha : a ∈ s) (hb : b ∈ s) :
    a ∈ P.part b ↔ P.part a = P.part b :=
  ⟨fun c ↦ (P.part_eq_of_mem (P.part_mem hb) c), fun c ↦ c ▸ P.mem_part ha⟩


theorem part_surjOn : Set.SurjOn P.part s P.parts := fun p hp ↦ by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    p : Finset α
    hp : Membership.mem (↑P.parts) p
    ⊢ Membership.mem (Set.image P.part ↑s) p
  -/
  obtain ⟨x, hx⟩ := P.nonempty_of_mem_parts hp
  /-
    case intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    p : Finset α
    hp : Membership.mem (↑P.parts) p
    x : α
    hx : Membership.mem p x
    ⊢ Membership.mem (Set.image P.part ↑s) p
  -/
  have hx' := mem_of_subset (P.le hp) hx
  /-
    case intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    p : Finset α
    hp : Membership.mem (↑P.parts) p
    x : α
    hx : Membership.mem p x
    hx' : Membership.mem s x
    ⊢ Membership.mem (Set.image P.part ↑s) p
  -/
  use x, hx', (P.existsUnique_mem hx').unique ⟨P.part_mem hx', P.mem_part hx'⟩ ⟨hp, hx⟩
  /-
    🎉 no goals
  -/


theorem exists_subset_part_bijOn : ∃ r ⊆ s, Set.BijOn P.part r P.parts := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    ⊢ Exists fun r => And (HasSubset.Subset r s) (Set.BijOn P.part ↑r ↑P.parts)
  -/
  obtain ⟨r, hrs, hr⟩ := P.part_surjOn.exists_bijOn_subset
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    r : Set α
    hrs : HasSubset.Subset r ↑s
    hr : Set.BijOn P.part r ↑P.parts
    ⊢ Exists fun r => And (HasSubset.Subset r s) (Set.BijOn P.part ↑r ↑P.parts)
  -/
  lift r to Finset α using s.finite_toSet.subset hrs
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    r : Finset α
    hrs : HasSubset.Subset ↑r ↑s
    hr : Set.BijOn P.part ↑r ↑P.parts
    ⊢ Exists fun r => And (HasSubset.Subset r s) (Set.BijOn P.part ↑r ↑P.parts)
  -/
  exact ⟨r, mod_cast hrs, hr⟩
  /-
    🎉 no goals
  -/


/-- Equivalence between a finpartition's parts as a dependent sum and the partitioned set. -/
def equivSigmaParts : s ≃ Σ t : P.parts, t.1 where
  toFun x := ⟨⟨P.part x.1, P.part_mem x.2⟩, ⟨x, P.mem_part x.2⟩⟩
  invFun x := ⟨x.2, mem_of_subset (P.le x.1.2) x.2.2⟩
                   /-
                     α : Type u_1
                     inst✝ : DecidableEq α
                     s t u : Finset α
                     P : Finpartition s
                     a : α
                     x : Subtype fun x => Membership.mem s x
                     ⊢ Eq ((fun x => ⟨↑x.snd, ⋯⟩) ((fun x => ⟨⟨P.part ↑x, ⋯⟩, ⟨↑x, ⋯⟩⟩) x)) x
                   -/
  left_inv x := by simp
                   /-
                     🎉 no goals
                   -/
  right_inv x := by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s t u : Finset α
      P : Finpartition s
      a : α
      x : Sigma fun t => Subtype fun x => Membership.mem (↑t) x
      ⊢ Eq ((fun x => ⟨⟨P.part ↑x, ⋯⟩, ⟨↑x, ⋯⟩⟩) ((fun x => ⟨↑x.snd, ⋯⟩) x)) x
    -/
    ext e
      /-
        case a.a.h
        α : Type u_1
        inst✝ : DecidableEq α
        s t u : Finset α
        P : Finpartition s
        a : α
        x : Sigma fun t => Subtype fun x => Membership.mem (↑t) x
        e : α
        ⊢ Iff (Membership.mem (↑((fun x => ⟨⟨P.part ↑x, ⋯⟩, ⟨↑x, ⋯⟩⟩) ((fun x => ⟨↑x.s …
      -/
    · obtain ⟨⟨p, mp⟩, ⟨f, mf⟩⟩ := x
      /-
        case a.a.h.mk.mk.mk
        α : Type u_1
        inst✝ : DecidableEq α
        s t u : Finset α
        P : Finpartition s
        a e : α
        p : Finset α
        mp : Membership.mem P.parts p
        f : α
        mf : Membership.mem (↑⟨p, mp⟩) f
        ⊢ Iff (Membership.mem (↑((fun x => ⟨⟨P.part ↑x, ⋯⟩, ⟨↑x, ⋯⟩⟩) ((fun x => ⟨↑x.s …
      -/
      dsimp only at mf ⊢
      /-
        case a.a.h.mk.mk.mk
        α : Type u_1
        inst✝ : DecidableEq α
        s t u : Finset α
        P : Finpartition s
        a e : α
        p : Finset α
        mp : Membership.mem P.parts p
        f : α
        mf : Membership.mem p f
        ⊢ Iff (Membership.mem (P.part f) e) (Membership.mem p e)
      -/
      rw [P.part_eq_of_mem mp mf]
      /-
        🎉 no goals
      -/
      /-
        case a
        α : Type u_1
        inst✝ : DecidableEq α
        s t u : Finset α
        P : Finpartition s
        a : α
        x : Sigma fun t => Subtype fun x => Membership.mem (↑t) x
        ⊢ Eq ↑((fun x => ⟨⟨P.part ↑x, ⋯⟩, ⟨↑x, ⋯⟩⟩) ((fun x => ⟨↑x.snd, ⋯⟩) x)).snd ↑x …
      -/
    · simp
      /-
        🎉 no goals
      -/


lemma exists_enumeration : ∃ f : s ≃ Σ t : P.parts, Fin #t.1,
    ∀ a b : s, P.part a = P.part b ↔ (f a).1 = (f b).1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    ⊢ Exists fun f => ∀ (a b : Subtype fun x => Membership.mem s x), Iff (Eq (P.pa …
  -/
  use P.equivSigmaParts.trans ((Equiv.refl _).sigmaCongr (fun t ↦ t.1.equivFin))
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    ⊢ ∀ (a b : Subtype fun x => Membership.mem s x), Iff (Eq (P.part ↑a) (P.part ↑ …
  -/
  simp [equivSigmaParts, Equiv.sigmaCongr, Equiv.sigmaCongrLeft]
  /-
    🎉 no goals
  -/


theorem sum_card_parts : ∑ i ∈ P.parts, #i = #s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    ⊢ Eq (P.parts.sum fun i => i.card) s.card
  -/
  convert congr_arg Finset.card P.biUnion_parts
  /-
    case h.e'_2
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    ⊢ Eq (P.parts.sum fun i => i.card) (P.parts.biUnion id).card
  -/
  rw [card_biUnion P.supIndep.pairwiseDisjoint]
  /-
    case h.e'_2
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    ⊢ Eq (P.parts.sum fun i => i.card) (P.parts.sum fun u => (id u).card)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `⊥` is the partition in singletons, aka discrete partition. -/
instance (s : Finset α) : Bot (Finpartition s) :=
  ⟨{  parts := s.map ⟨singleton, singleton_injective⟩
      supIndep :=
        Set.PairwiseDisjoint.supIndep
          (by
            /-
              α : Type u_1
              inst✝ : DecidableEq α
              s✝ t u : Finset α
              P : Finpartition s✝
              a : α
              s : Finset α
              ⊢ (↑(Finset.map { toFun := Singleton.singleton, inj' := ⋯ } s)).PairwiseDisjoi …
            -/
            rw [Finset.coe_map]
            /-
              α : Type u_1
              inst✝ : DecidableEq α
              s✝ t u : Finset α
              P : Finpartition s✝
              a : α
              s : Finset α
              ⊢ (Set.image ⇑{ toFun := Singleton.singleton, inj' := ⋯ } ↑s).PairwiseDisjoint …
            -/
            exact Finset.pairwiseDisjoint_range_singleton.subset (Set.image_subset_range _ _))
            /-
              🎉 no goals
            -/
                      /-
                        α : Type u_1
                        inst✝ : DecidableEq α
                        s✝ t u : Finset α
                        P : Finpartition s✝
                        a : α
                        s : Finset α
                        ⊢ Eq ((Finset.map { toFun := Singleton.singleton, inj' := ⋯ } s).sup id) s
                      -/
      sup_parts := by rw [sup_map, id_comp, Embedding.coeFn_mk, Finset.sup_singleton']
                      /-
                        🎉 no goals
                      -/
                        /-
                          α : Type u_1
                          inst✝ : DecidableEq α
                          s✝ t u : Finset α
                          P : Finpartition s✝
                          a : α
                          s : Finset α
                          ⊢ Not (Membership.mem (Finset.map { toFun := Singleton.singleton, inj' := ⋯ }  …
                        -/
      not_bot_mem := by simp }⟩
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem parts_bot (s : Finset α) :
    (⊥ : Finpartition s).parts = s.map ⟨singleton, singleton_injective⟩ :=
  rfl


theorem card_bot (s : Finset α) : #(⊥ : Finpartition s).parts = #s := Finset.card_map _


theorem mem_bot_iff : t ∈ (⊥ : Finpartition s).parts ↔ ∃ a ∈ s, {a} = t :=
  mem_map


instance (s : Finset α) : OrderBot (Finpartition s) :=
  { (inferInstance : Bot (Finpartition s)) with
    bot_le := fun P t ht ↦ by
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        s✝ t✝ u : Finset α
        P✝ : Finpartition s✝
        a : α
        s : Finset α
        P : Finpartition s
        t : Finset α
        ht : Membership.mem Bot.bot.parts t
        ⊢ Exists fun c => And (Membership.mem P.parts c) (LE.le t c)
      -/
      rw [mem_bot_iff] at ht
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        s✝ t✝ u : Finset α
        P✝ : Finpartition s✝
        a : α
        s : Finset α
        P : Finpartition s
        t : Finset α
        ht : Exists fun a => And (Membership.mem s a) (Eq (Singleton.singleton a) t)
        ⊢ Exists fun c => And (Membership.mem P.parts c) (LE.le t c)
      -/
      obtain ⟨a, ha, rfl⟩ := ht
      /-
        case intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        s✝ t u : Finset α
        P✝ : Finpartition s✝
        a✝ : α
        s : Finset α
        P : Finpartition s
        a : α
        ha : Membership.mem s a
        ⊢ Exists fun c => And (Membership.mem P.parts c) (LE.le (Singleton.singleton a …
      -/
      obtain ⟨t, ht, hat⟩ := P.exists_mem ha
      /-
        case intro.intro.intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        s✝ t✝ u : Finset α
        P✝ : Finpartition s✝
        a✝ : α
        s : Finset α
        P : Finpartition s
        a : α
        ha : Membership.mem s a
        t : Finset α
        ht : Membership.mem P.parts t
        hat : Membership.mem t a
        ⊢ Exists fun c => And (Membership.mem P.parts c) (LE.le (Singleton.singleton a …
      -/
      exact ⟨t, ht, singleton_subset_iff.2 hat⟩ }
      /-
        🎉 no goals
      -/


theorem card_parts_le_card : #P.parts ≤ #s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    ⊢ LE.le P.parts.card s.card
  -/
  rw [← card_bot s]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    ⊢ LE.le P.parts.card Bot.bot.parts.card
  -/
  exact card_mono bot_le
  /-
    🎉 no goals
  -/


lemma card_mod_card_parts_le : #s % #P.parts ≤ #P.parts := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    ⊢ LE.le (HMod.hMod s.card P.parts.card) P.parts.card
  -/
  obtain h | h := (#P.parts).eq_zero_or_pos
    /-
      case inl
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      P : Finpartition s
      h : Eq P.parts.card 0
      ⊢ LE.le (HMod.hMod s.card P.parts.card) P.parts.card
    -/
  · rw [h]
    /-
      case inl
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      P : Finpartition s
      h : Eq P.parts.card 0
      ⊢ LE.le (HMod.hMod s.card 0) 0
    -/
    rw [Finset.card_eq_zero, parts_eq_empty_iff, bot_eq_empty, ← Finset.card_eq_zero] at h
    /-
      case inl
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      P : Finpartition s
      h : Eq s.card 0
      ⊢ LE.le (HMod.hMod s.card 0) 0
    -/
    rw [h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      P : Finpartition s
      h : GT.gt P.parts.card 0
      ⊢ LE.le (HMod.hMod s.card P.parts.card) P.parts.card
    -/
  · exact (Nat.mod_lt _ h).le
    /-
      🎉 no goals
    -/


/-- A setoid over a finite type induces a finpartition of the type's elements,
where the parts are the setoid's equivalence classes. -/
def ofSetoid (s : Setoid α) [DecidableRel s.r] : Finpartition (univ : Finset α) where
  parts := univ.image fun a ↦ ({b | s.r a b} : Finset α)
  supIndep := by
    simp only [mem_univ, forall_true_left, supIndep_iff_pairwiseDisjoint, Set.PairwiseDisjoint,
      Set.Pairwise, coe_image, coe_univ, Set.image_univ, Set.mem_range, ne_eq,
      forall_exists_index, forall_apply_eq_imp_iff]
    /-
      α : Type u_1
      inst✝² : DecidableEq α
      s✝ t u : Finset α
      P : Finpartition s✝
      a : α
      inst✝¹ : Fintype α
      s : Setoid α
      inst✝ : DecidableRel ⇑s
      ⊢ ∀ (a a_1 : α), Not (Eq (Finset.filter (fun b => s a b) Finset.univ) (Finset. …
    -/
    intro _ _ q
    /-
      α : Type u_1
      inst✝² : DecidableEq α
      s✝ t u : Finset α
      P : Finpartition s✝
      a : α
      inst✝¹ : Fintype α
      s : Setoid α
      inst✝ : DecidableRel ⇑s
      a✝¹ a✝ : α
      q : Not (Eq (Finset.filter (fun b => s a✝¹ b) Finset.univ) (Finset.filter (fun …
      ⊢ Function.onFun Disjoint id (Finset.filter (fun b => s a✝¹ b) Finset.univ) (F …
    -/
    contrapose! q
    /-
      α : Type u_1
      inst✝² : DecidableEq α
      s✝ t u : Finset α
      P : Finpartition s✝
      a : α
      inst✝¹ : Fintype α
      s : Setoid α
      inst✝ : DecidableRel ⇑s
      a✝¹ a✝ : α
      q : Not (Function.onFun Disjoint id (Finset.filter (fun b => s a✝¹ b) Finset.u …
      ⊢ Eq (Finset.filter (fun b => s a✝¹ b) Finset.univ) (Finset.filter (fun b => s …
    -/
    rw [not_disjoint_iff] at q
    /-
      α : Type u_1
      inst✝² : DecidableEq α
      s✝ t u : Finset α
      P : Finpartition s✝
      a : α
      inst✝¹ : Fintype α
      s : Setoid α
      inst✝ : DecidableRel ⇑s
      a✝¹ a✝ : α
      q : Exists fun a => And (Membership.mem (id (Finset.filter (fun b => s a✝¹ b)  …
      ⊢ Eq (Finset.filter (fun b => s a✝¹ b) Finset.univ) (Finset.filter (fun b => s …
    -/
    obtain ⟨c, ⟨d1, d2⟩⟩ := q
    /-
      case intro.intro
      α : Type u_1
      inst✝² : DecidableEq α
      s✝ t u : Finset α
      P : Finpartition s✝
      a : α
      inst✝¹ : Fintype α
      s : Setoid α
      inst✝ : DecidableRel ⇑s
      a✝¹ a✝ c : α
      d1 : Membership.mem (id (Finset.filter (fun b => s a✝¹ b) Finset.univ)) c
      d2 : Membership.mem (id (Finset.filter (fun b => s a✝ b) Finset.univ)) c
      ⊢ Eq (Finset.filter (fun b => s a✝¹ b) Finset.univ) (Finset.filter (fun b => s …
    -/
    rw [id_eq, mem_filter] at d1 d2
    /-
      case intro.intro
      α : Type u_1
      inst✝² : DecidableEq α
      s✝ t u : Finset α
      P : Finpartition s✝
      a : α
      inst✝¹ : Fintype α
      s : Setoid α
      inst✝ : DecidableRel ⇑s
      a✝¹ a✝ c : α
      d1 : And (Membership.mem Finset.univ c) (s a✝¹ c)
      d2 : And (Membership.mem Finset.univ c) (s a✝ c)
      ⊢ Eq (Finset.filter (fun b => s a✝¹ b) Finset.univ) (Finset.filter (fun b => s …
    -/
    ext y
    /-
      case intro.intro.h
      α : Type u_1
      inst✝² : DecidableEq α
      s✝ t u : Finset α
      P : Finpartition s✝
      a : α
      inst✝¹ : Fintype α
      s : Setoid α
      inst✝ : DecidableRel ⇑s
      a✝¹ a✝ c : α
      d1 : And (Membership.mem Finset.univ c) (s a✝¹ c)
      d2 : And (Membership.mem Finset.univ c) (s a✝ c)
      y : α
      ⊢ Iff (Membership.mem (Finset.filter (fun b => s a✝¹ b) Finset.univ) y) (Membe …
    -/
    simp only [mem_univ, forall_true_left, mem_filter, true_and]
    exact ⟨fun r1 => s.trans (s.trans d2.2 (s.symm d1.2)) r1,
           fun r2 => s.trans (s.trans d1.2 (s.symm d2.2)) r2⟩
  sup_parts := by
    /-
      α : Type u_1
      inst✝² : DecidableEq α
      s✝ t u : Finset α
      P : Finpartition s✝
      a : α
      inst✝¹ : Fintype α
      s : Setoid α
      inst✝ : DecidableRel ⇑s
      ⊢ Eq ((Finset.image (fun a => Finset.filter (fun b => s a b) Finset.univ) Fins …
    -/
    ext a
    /-
      case h
      α : Type u_1
      inst✝² : DecidableEq α
      s✝ t u : Finset α
      P : Finpartition s✝
      a✝ : α
      inst✝¹ : Fintype α
      s : Setoid α
      inst✝ : DecidableRel ⇑s
      a : α
      ⊢ Iff (Membership.mem ((Finset.image (fun a => Finset.filter (fun b => s a b)  …
    -/
    simp only [sup_image, Function.id_comp, mem_univ, mem_sup, mem_filter, true_and, iff_true]
    /-
      case h
      α : Type u_1
      inst✝² : DecidableEq α
      s✝ t u : Finset α
      P : Finpartition s✝
      a✝ : α
      inst✝¹ : Fintype α
      s : Setoid α
      inst✝ : DecidableRel ⇑s
      a : α
      ⊢ Exists fun i => s i a
    -/
    use a
    /-
      🎉 no goals
    -/
  not_bot_mem := by
    /-
      α : Type u_1
      inst✝² : DecidableEq α
      s✝ t u : Finset α
      P : Finpartition s✝
      a : α
      inst✝¹ : Fintype α
      s : Setoid α
      inst✝ : DecidableRel ⇑s
      ⊢ Not (Membership.mem (Finset.image (fun a => Finset.filter (fun b => s a b) F …
    -/
    rw [bot_eq_empty, mem_image, not_exists]
    /-
      α : Type u_1
      inst✝² : DecidableEq α
      s✝ t u : Finset α
      P : Finpartition s✝
      a : α
      inst✝¹ : Fintype α
      s : Setoid α
      inst✝ : DecidableRel ⇑s
      ⊢ ∀ (x : α), Not (And (Membership.mem Finset.univ x) (Eq (Finset.filter (fun b …
    -/
    intro a
    /-
      α : Type u_1
      inst✝² : DecidableEq α
      s✝ t u : Finset α
      P : Finpartition s✝
      a✝ : α
      inst✝¹ : Fintype α
      s : Setoid α
      inst✝ : DecidableRel ⇑s
      a : α
      ⊢ Not (And (Membership.mem Finset.univ a) (Eq (Finset.filter (fun b => s a b)  …
    -/
    simp only [filter_eq_empty_iff, not_forall, mem_univ, forall_true_left, true_and, not_not]
    /-
      α : Type u_1
      inst✝² : DecidableEq α
      s✝ t u : Finset α
      P : Finpartition s✝
      a✝ : α
      inst✝¹ : Fintype α
      s : Setoid α
      inst✝ : DecidableRel ⇑s
      a : α
      ⊢ Exists fun x => s a x
    -/
    use a
    /-
      🎉 no goals
    -/


theorem mem_part_ofSetoid_iff_rel {s : Setoid α} [DecidableRel s.r] {b : α} :
    b ∈ (ofSetoid s).part a ↔ s.r a b := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    a : α
    inst✝¹ : Fintype α
    s : Setoid α
    inst✝ : DecidableRel ⇑s
    b : α
    ⊢ Iff (Membership.mem ((Finpartition.ofSetoid s).part a) b) (s a b)
  -/
  simp_rw [part, ofSetoid, mem_univ, reduceDIte]
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    a : α
    inst✝¹ : Fintype α
    s : Setoid α
    inst✝ : DecidableRel ⇑s
    b : α
    ⊢ Iff (Membership.mem (Finset.choose (fun a_1 => Membership.mem a_1 a) (Finset …
  -/
  generalize_proofs H
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    a : α
    inst✝¹ : Fintype α
    s : Setoid α
    inst✝ : DecidableRel ⇑s
    b : α
    H : ExistsUnique fun a_1 => And (Membership.mem (Finset.image (fun a => Finset …
    ⊢ Iff (Membership.mem (Finset.choose (fun a_1 => Membership.mem a_1 a) (Finset …
  -/
  have := choose_spec _ _ H
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    a : α
    inst✝¹ : Fintype α
    s : Setoid α
    inst✝ : DecidableRel ⇑s
    b : α
    H : ExistsUnique fun a_1 => And (Membership.mem (Finset.image (fun a => Finset …
    this : And (Membership.mem (Finset.image (fun a => Finset.filter (fun b => s a …
    ⊢ Iff (Membership.mem (Finset.choose (fun a_1 => Membership.mem a_1 a) (Finset …
  -/
  simp only [mem_univ, mem_image, true_and] at this
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    a : α
    inst✝¹ : Fintype α
    s : Setoid α
    inst✝ : DecidableRel ⇑s
    b : α
    H : ExistsUnique fun a_1 => And (Membership.mem (Finset.image (fun a => Finset …
    this : And (Exists fun a_1 => Eq (Finset.filter (fun b => s a_1 b) Finset.univ …
    ⊢ Iff (Membership.mem (Finset.choose (fun a_1 => Membership.mem a_1 a) (Finset …
  -/
  obtain ⟨⟨_, hc⟩, this⟩ := this
  /-
    case intro.intro
    α : Type u_1
    inst✝² : DecidableEq α
    a : α
    inst✝¹ : Fintype α
    s : Setoid α
    inst✝ : DecidableRel ⇑s
    b : α
    H : ExistsUnique fun a_1 => And (Membership.mem (Finset.image (fun a => Finset …
    this : Membership.mem (Finset.choose (fun a_1 => Membership.mem a_1 a) (Finset …
    w✝ : α
    hc : Eq (Finset.filter (fun b => s w✝ b) Finset.univ) (Finset.choose (fun a_1  …
    ⊢ Iff (Membership.mem (Finset.choose (fun a_1 => Membership.mem a_1 a) (Finset …
  -/
  simp only [← hc, mem_univ, mem_filter, true_and] at this ⊢
  /-
    case intro.intro
    α : Type u_1
    inst✝² : DecidableEq α
    a : α
    inst✝¹ : Fintype α
    s : Setoid α
    inst✝ : DecidableRel ⇑s
    b : α
    H : ExistsUnique fun a_1 => And (Membership.mem (Finset.image (fun a => Finset …
    w✝ : α
    hc : Eq (Finset.filter (fun b => s w✝ b) Finset.univ) (Finset.choose (fun a_1  …
    this : s w✝ a
    ⊢ Iff (s w✝ b) (s a b)
  -/
  exact ⟨s.trans (s.symm this), s.trans this⟩
  /-
    🎉 no goals
  -/


/-- Cuts `s` along the finsets in `F`: Two elements of `s` will be in the same part if they are
in the same finsets of `F`. -/
def atomise (s : Finset α) (F : Finset (Finset α)) : Finpartition s :=
  ofErase (F.powerset.image fun Q ↦ {i ∈ s | ∀ t ∈ F, t ∈ Q ↔ i ∈ t})
    (Set.PairwiseDisjoint.supIndep fun x hx y hy h ↦
      disjoint_left.mpr fun z hz1 hz2 ↦
        h (by
            /-
              α : Type u_1
              inst✝ : DecidableEq α
              s✝ t u : Finset α
              P : Finpartition s✝
              a : α
              s : Finset α
              F : Finset (Finset α)
              x : Finset α
              hx : Membership.mem (↑(Finset.image (fun Q => Finset.filter (fun i => ∀ (t : F …
              y : Finset α
              hy : Membership.mem (↑(Finset.image (fun Q => Finset.filter (fun i => ∀ (t : F …
              h : Ne x y
              z : α
              hz1 : Membership.mem (id x) z
              hz2 : Membership.mem (id y) z
              ⊢ Eq x y
            -/
            rw [mem_coe, mem_image] at hx hy
            /-
              α : Type u_1
              inst✝ : DecidableEq α
              s✝ t u : Finset α
              P : Finpartition s✝
              a : α
              s : Finset α
              F : Finset (Finset α)
              x : Finset α
              hx : Exists fun a => And (Membership.mem F.powerset a) (Eq (Finset.filter (fun …
              y : Finset α
              hy : Exists fun a => And (Membership.mem F.powerset a) (Eq (Finset.filter (fun …
              h : Ne x y
              z : α
              hz1 : Membership.mem (id x) z
              hz2 : Membership.mem (id y) z
              ⊢ Eq x y
            -/
            obtain ⟨Q, hQ, rfl⟩ := hx
            /-
              case intro.intro
              α : Type u_1
              inst✝ : DecidableEq α
              s✝ t u : Finset α
              P : Finpartition s✝
              a : α
              s : Finset α
              F : Finset (Finset α)
              y : Finset α
              hy : Exists fun a => And (Membership.mem F.powerset a) (Eq (Finset.filter (fun …
              z : α
              hz2 : Membership.mem (id y) z
              Q : Finset (Finset α)
              hQ : Membership.mem F.powerset Q
              h : Ne (Finset.filter (fun i => ∀ (t : Finset α), Membership.mem F t → Iff (Me …
              hz1 : Membership.mem (id (Finset.filter (fun i => ∀ (t : Finset α), Membership …
              ⊢ Eq (Finset.filter (fun i => ∀ (t : Finset α), Membership.mem F t → Iff (Memb …
            -/
            obtain ⟨R, hR, rfl⟩ := hy
            suffices h' : Q = R by
              subst h'
              exact of_eq_true (eq_self {i ∈ s | ∀ t ∈ F, t ∈ Q ↔ i ∈ t})
            /-
              case intro.intro.intro.intro
              α : Type u_1
              inst✝ : DecidableEq α
              s✝ t u : Finset α
              P : Finpartition s✝
              a : α
              s : Finset α
              F : Finset (Finset α)
              z : α
              Q : Finset (Finset α)
              hQ : Membership.mem F.powerset Q
              hz1 : Membership.mem (id (Finset.filter (fun i => ∀ (t : Finset α), Membership …
              R : Finset (Finset α)
              hR : Membership.mem F.powerset R
              hz2 : Membership.mem (id (Finset.filter (fun i => ∀ (t : Finset α), Membership …
              h : Ne (Finset.filter (fun i => ∀ (t : Finset α), Membership.mem F t → Iff (Me …
              ⊢ Eq Q R
            -/
            rw [id, mem_filter] at hz1 hz2
            /-
              case intro.intro.intro.intro
              α : Type u_1
              inst✝ : DecidableEq α
              s✝ t u : Finset α
              P : Finpartition s✝
              a : α
              s : Finset α
              F : Finset (Finset α)
              z : α
              Q : Finset (Finset α)
              hQ : Membership.mem F.powerset Q
              hz1 : And (Membership.mem s z) (∀ (t : Finset α), Membership.mem F t → Iff (Me …
              R : Finset (Finset α)
              hR : Membership.mem F.powerset R
              hz2 : And (Membership.mem s z) (∀ (t : Finset α), Membership.mem F t → Iff (Me …
              h : Ne (Finset.filter (fun i => ∀ (t : Finset α), Membership.mem F t → Iff (Me …
              ⊢ Eq Q R
            -/
            rw [mem_powerset] at hQ hR
            /-
              case intro.intro.intro.intro
              α : Type u_1
              inst✝ : DecidableEq α
              s✝ t u : Finset α
              P : Finpartition s✝
              a : α
              s : Finset α
              F : Finset (Finset α)
              z : α
              Q : Finset (Finset α)
              hQ : HasSubset.Subset Q F
              hz1 : And (Membership.mem s z) (∀ (t : Finset α), Membership.mem F t → Iff (Me …
              R : Finset (Finset α)
              hR : HasSubset.Subset R F
              hz2 : And (Membership.mem s z) (∀ (t : Finset α), Membership.mem F t → Iff (Me …
              h : Ne (Finset.filter (fun i => ∀ (t : Finset α), Membership.mem F t → Iff (Me …
              ⊢ Eq Q R
            -/
            ext i
            /-
              case intro.intro.intro.intro.h
              α : Type u_1
              inst✝ : DecidableEq α
              s✝ t u : Finset α
              P : Finpartition s✝
              a : α
              s : Finset α
              F : Finset (Finset α)
              z : α
              Q : Finset (Finset α)
              hQ : HasSubset.Subset Q F
              hz1 : And (Membership.mem s z) (∀ (t : Finset α), Membership.mem F t → Iff (Me …
              R : Finset (Finset α)
              hR : HasSubset.Subset R F
              hz2 : And (Membership.mem s z) (∀ (t : Finset α), Membership.mem F t → Iff (Me …
              h : Ne (Finset.filter (fun i => ∀ (t : Finset α), Membership.mem F t → Iff (Me …
              i : Finset α
              ⊢ Iff (Membership.mem Q i) (Membership.mem R i)
            -/
            refine ⟨fun hi ↦ ?_, fun hi ↦ ?_⟩
              /-
                case intro.intro.intro.intro.h.refine_1
                α : Type u_1
                inst✝ : DecidableEq α
                s✝ t u : Finset α
                P : Finpartition s✝
                a : α
                s : Finset α
                F : Finset (Finset α)
                z : α
                Q : Finset (Finset α)
                hQ : HasSubset.Subset Q F
                hz1 : And (Membership.mem s z) (∀ (t : Finset α), Membership.mem F t → Iff (Me …
                R : Finset (Finset α)
                hR : HasSubset.Subset R F
                hz2 : And (Membership.mem s z) (∀ (t : Finset α), Membership.mem F t → Iff (Me …
                h : Ne (Finset.filter (fun i => ∀ (t : Finset α), Membership.mem F t → Iff (Me …
                i : Finset α
                hi : Membership.mem Q i
                ⊢ Membership.mem R i
              -/
            · rwa [hz2.2 _ (hQ hi), ← hz1.2 _ (hQ hi)]
              /-
                🎉 no goals
              -/
              /-
                case intro.intro.intro.intro.h.refine_2
                α : Type u_1
                inst✝ : DecidableEq α
                s✝ t u : Finset α
                P : Finpartition s✝
                a : α
                s : Finset α
                F : Finset (Finset α)
                z : α
                Q : Finset (Finset α)
                hQ : HasSubset.Subset Q F
                hz1 : And (Membership.mem s z) (∀ (t : Finset α), Membership.mem F t → Iff (Me …
                R : Finset (Finset α)
                hR : HasSubset.Subset R F
                hz2 : And (Membership.mem s z) (∀ (t : Finset α), Membership.mem F t → Iff (Me …
                h : Ne (Finset.filter (fun i => ∀ (t : Finset α), Membership.mem F t → Iff (Me …
                i : Finset α
                hi : Membership.mem R i
                ⊢ Membership.mem Q i
              -/
            · rwa [hz1.2 _ (hR hi), ← hz2.2 _ (hR hi)]))
              /-
                🎉 no goals
              -/
    (by
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        s✝ t u : Finset α
        P : Finpartition s✝
        a : α
        s : Finset α
        F : Finset (Finset α)
        ⊢ Eq ((Finset.image (fun Q => Finset.filter (fun i => ∀ (t : Finset α), Member …
      -/
      refine (Finset.sup_le fun t ht ↦ ?_).antisymm fun a ha ↦ ?_
        /-
          case refine_1
          α : Type u_1
          inst✝ : DecidableEq α
          s✝ t✝ u : Finset α
          P : Finpartition s✝
          a : α
          s : Finset α
          F : Finset (Finset α)
          t : Finset α
          ht : Membership.mem (Finset.image (fun Q => Finset.filter (fun i => ∀ (t : Fin …
          ⊢ LE.le (id t) s
        -/
      · rw [mem_image] at ht
        /-
          case refine_1
          α : Type u_1
          inst✝ : DecidableEq α
          s✝ t✝ u : Finset α
          P : Finpartition s✝
          a : α
          s : Finset α
          F : Finset (Finset α)
          t : Finset α
          ht : Exists fun a => And (Membership.mem F.powerset a) (Eq (Finset.filter (fun …
          ⊢ LE.le (id t) s
        -/
        obtain ⟨A, _, rfl⟩ := ht
        /-
          case refine_1.intro.intro
          α : Type u_1
          inst✝ : DecidableEq α
          s✝ t u : Finset α
          P : Finpartition s✝
          a : α
          s : Finset α
          F A : Finset (Finset α)
          left✝ : Membership.mem F.powerset A
          ⊢ LE.le (id (Finset.filter (fun i => ∀ (t : Finset α), Membership.mem F t → If …
        -/
        exact s.filter_subset _
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          α : Type u_1
          inst✝ : DecidableEq α
          s✝ t u : Finset α
          P : Finpartition s✝
          a✝ : α
          s : Finset α
          F : Finset (Finset α)
          a : α
          ha : Membership.mem s a
          ⊢ Membership.mem ((Finset.image (fun Q => Finset.filter (fun i => ∀ (t : Finse …
        -/
      · rw [mem_sup]
        refine
          ⟨{i ∈ s | ∀ t ∈ F, t ∈ {u ∈ F | a ∈ u} ↔ i ∈ t},
            mem_image_of_mem _ (mem_powerset.2 <| filter_subset _ _),
            mem_filter.2 ⟨ha, fun t ht ↦ ?_⟩⟩
        /-
          case refine_2
          α : Type u_1
          inst✝ : DecidableEq α
          s✝ t✝ u : Finset α
          P : Finpartition s✝
          a✝ : α
          s : Finset α
          F : Finset (Finset α)
          a : α
          ha : Membership.mem s a
          t : Finset α
          ht : Membership.mem F t
          ⊢ Iff (Membership.mem (Finset.filter (fun u => Membership.mem u a) F) t) (Memb …
        -/
        rw [mem_filter]
        /-
          case refine_2
          α : Type u_1
          inst✝ : DecidableEq α
          s✝ t✝ u : Finset α
          P : Finpartition s✝
          a✝ : α
          s : Finset α
          F : Finset (Finset α)
          a : α
          ha : Membership.mem s a
          t : Finset α
          ht : Membership.mem F t
          ⊢ Iff (And (Membership.mem F t) (Membership.mem t a)) (Membership.mem t a)
        -/
        exact and_iff_right ht)
        /-
          🎉 no goals
        -/


theorem mem_atomise :
    t ∈ (atomise s F).parts ↔
      t.Nonempty ∧ ∃ Q ⊆ F, {i ∈ s | ∀ u ∈ F, u ∈ Q ↔ i ∈ u} = t := by
  simp only [atomise, ofErase, bot_eq_empty, mem_erase, mem_image, nonempty_iff_ne_empty,
    mem_singleton, and_comm, mem_powerset, exists_prop]


theorem atomise_empty (hs : s.Nonempty) : (atomise s ∅).parts = {s} := by
  simp only [atomise, powerset_empty, image_singleton, not_mem_empty, IsEmpty.forall_iff,
    imp_true_iff, filter_True]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    hs : s.Nonempty
    ⊢ Eq (Finpartition.ofErase (Singleton.singleton s) ⋯ ⋯).parts (Singleton.singl …
  -/
  exact erase_eq_of_not_mem (not_mem_singleton.2 hs.ne_empty.symm)
  /-
    🎉 no goals
  -/


theorem card_atomise_le : #(atomise s F).parts ≤ 2 ^ #F :=
  (card_le_card <| erase_subset _ _).trans <| Finset.card_image_le.trans (card_powerset _).le


theorem biUnion_filter_atomise (ht : t ∈ F) (hts : t ⊆ s) :
    {u ∈ (atomise s F).parts | u ⊆ t ∧ u.Nonempty}.biUnion id = t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    F : Finset (Finset α)
    ht : Membership.mem F t
    hts : HasSubset.Subset t s
    ⊢ Eq ((Finset.filter (fun u => And (HasSubset.Subset u t) u.Nonempty) (Finpart …
  -/
  ext a
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    F : Finset (Finset α)
    ht : Membership.mem F t
    hts : HasSubset.Subset t s
    a : α
    ⊢ Iff (Membership.mem ((Finset.filter (fun u => And (HasSubset.Subset u t) u.N …
  -/
  refine mem_biUnion.trans ⟨fun ⟨u, hu, ha⟩ ↦ (mem_filter.1 hu).2.1 ha, fun ha ↦ ?_⟩
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    F : Finset (Finset α)
    ht : Membership.mem F t
    hts : HasSubset.Subset t s
    a : α
    ha : Membership.mem t a
    ⊢ Exists fun a_1 => And (Membership.mem (Finset.filter (fun u => And (HasSubse …
  -/
  obtain ⟨u, hu, hau⟩ := (atomise s F).exists_mem (hts ha)
  /-
    case h.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    F : Finset (Finset α)
    ht : Membership.mem F t
    hts : HasSubset.Subset t s
    a : α
    ha : Membership.mem t a
    u : Finset α
    hu : Membership.mem (Finpartition.atomise s F).parts u
    hau : Membership.mem u a
    ⊢ Exists fun a_1 => And (Membership.mem (Finset.filter (fun u => And (HasSubse …
  -/
  refine ⟨u, mem_filter.2 ⟨hu, fun b hb ↦ ?_, _, hau⟩, hau⟩
  /-
    case h.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    F : Finset (Finset α)
    ht : Membership.mem F t
    hts : HasSubset.Subset t s
    a : α
    ha : Membership.mem t a
    u : Finset α
    hu : Membership.mem (Finpartition.atomise s F).parts u
    hau : Membership.mem u a
    b : α
    hb : Membership.mem u b
    ⊢ Membership.mem t b
  -/
  obtain ⟨Q, _hQ, rfl⟩ := (mem_atomise.1 hu).2
  /-
    case h.intro.intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    F : Finset (Finset α)
    ht : Membership.mem F t
    hts : HasSubset.Subset t s
    a : α
    ha : Membership.mem t a
    b : α
    Q : Finset (Finset α)
    _hQ : HasSubset.Subset Q F
    hu : Membership.mem (Finpartition.atomise s F).parts (Finset.filter (fun i =>  …
    hau : Membership.mem (Finset.filter (fun i => ∀ (u : Finset α), Membership.mem …
    hb : Membership.mem (Finset.filter (fun i => ∀ (u : Finset α), Membership.mem  …
    ⊢ Membership.mem t b
  -/
  rw [mem_filter] at hau hb
  /-
    case h.intro.intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    F : Finset (Finset α)
    ht : Membership.mem F t
    hts : HasSubset.Subset t s
    a : α
    ha : Membership.mem t a
    b : α
    Q : Finset (Finset α)
    _hQ : HasSubset.Subset Q F
    hu : Membership.mem (Finpartition.atomise s F).parts (Finset.filter (fun i =>  …
    hau : And (Membership.mem s a) (∀ (u : Finset α), Membership.mem F u → Iff (Me …
    hb : And (Membership.mem s b) (∀ (u : Finset α), Membership.mem F u → Iff (Mem …
    ⊢ Membership.mem t b
  -/
  rwa [← hb.2 _ ht, hau.2 _ ht]
  /-
    🎉 no goals
  -/


theorem card_filter_atomise_le_two_pow (ht : t ∈ F) :
    #{u ∈ (atomise s F).parts | u ⊆ t ∧ u.Nonempty} ≤ 2 ^ (#F - 1) := by
  suffices h :
    {u ∈ (atomise s F).parts | u ⊆ t ∧ u.Nonempty} ⊆
      (F.erase t).powerset.image fun P ↦ {i ∈ s | ∀ x ∈ F, x ∈ insert t P ↔ i ∈ x} by
    refine (card_le_card h).trans (card_image_le.trans ?_)
    rw [card_powerset, card_erase_of_mem ht]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    F : Finset (Finset α)
    ht : Membership.mem F t
    ⊢ HasSubset.Subset (Finset.filter (fun u => And (HasSubset.Subset u t) u.Nonem …
  -/
  rw [subset_iff]
  simp_rw [mem_image, mem_powerset, mem_filter, and_imp, Finset.Nonempty, exists_imp, mem_atomise,
    and_imp, Finset.Nonempty, exists_imp, and_imp]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    F : Finset (Finset α)
    ht : Membership.mem F t
    ⊢ ∀ ⦃x : Finset α⦄ (x_1 : α), Membership.mem x x_1 → ∀ (x_2 : Finset (Finset α …
  -/
  rintro P' i hi P PQ rfl hy₂ j _hj
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    F : Finset (Finset α)
    ht : Membership.mem F t
    i : α
    P : Finset (Finset α)
    PQ : HasSubset.Subset P F
    hi : Membership.mem (Finset.filter (fun i => ∀ (u : Finset α), Membership.mem  …
    hy₂ : HasSubset.Subset (Finset.filter (fun i => ∀ (u : Finset α), Membership.m …
    j : α
    _hj : Membership.mem (Finset.filter (fun i => ∀ (u : Finset α), Membership.mem …
    ⊢ Exists fun a => And (HasSubset.Subset a (F.erase t)) (Eq (Finset.filter (fun …
  -/
  refine ⟨P.erase t, erase_subset_erase _ PQ, ?_⟩
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    F : Finset (Finset α)
    ht : Membership.mem F t
    i : α
    P : Finset (Finset α)
    PQ : HasSubset.Subset P F
    hi : Membership.mem (Finset.filter (fun i => ∀ (u : Finset α), Membership.mem  …
    hy₂ : HasSubset.Subset (Finset.filter (fun i => ∀ (u : Finset α), Membership.m …
    j : α
    _hj : Membership.mem (Finset.filter (fun i => ∀ (u : Finset α), Membership.mem …
    ⊢ Eq (Finset.filter (fun i => ∀ (x : Finset α), Membership.mem F x → Iff (Memb …
  -/
  simp only [insert_erase (((mem_filter.1 hi).2 _ ht).2 <| hy₂ hi)]
  /-
    🎉 no goals
  -/


