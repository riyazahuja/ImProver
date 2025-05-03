/-- A set `s` is *sup-closed* if `a ⊔ b ∈ s` for all `a ∈ s`, `b ∈ s`. -/
def SupClosed (s : Set α) : Prop := ∀ ⦃a⦄, a ∈ s → ∀ ⦃b⦄, b ∈ s → a ⊔ b ∈ s


                                                            /-
                                                              α : Type u_3
                                                              inst✝ : SemilatticeSup α
                                                              ⊢ SupClosed EmptyCollection.emptyCollection
                                                            -/
@[simp] lemma supClosed_empty : SupClosed (∅ : Set α) := by simp [SupClosed]
                                                            /-
                                                              🎉 no goals
                                                            -/

                                                                  /-
                                                                    α : Type u_3
                                                                    inst✝ : SemilatticeSup α
                                                                    a : α
                                                                    ⊢ SupClosed (Singleton.singleton a)
                                                                  -/
@[simp] lemma supClosed_singleton : SupClosed ({a} : Set α) := by simp [SupClosed]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


                                                              /-
                                                                α : Type u_3
                                                                inst✝ : SemilatticeSup α
                                                                ⊢ SupClosed Set.univ
                                                              -/
@[simp] lemma supClosed_univ : SupClosed (univ : Set α) := by simp [SupClosed]
                                                              /-
                                                                🎉 no goals
                                                              -/

lemma SupClosed.inter (hs : SupClosed s) (ht : SupClosed t) : SupClosed (s ∩ t) :=
  fun _a ha _b hb ↦ ⟨hs ha.1 hb.1, ht ha.2 hb.2⟩


lemma supClosed_sInter (hS : ∀ s ∈ S, SupClosed s) : SupClosed (⋂₀ S) :=
  fun _a ha _b hb _s hs ↦ hS _ hs (ha _ hs) (hb _ hs)


lemma supClosed_iInter (hf : ∀ i, SupClosed (f i)) : SupClosed (⋂ i, f i) :=
  supClosed_sInter <| forall_mem_range.2 hf


lemma SupClosed.directedOn (hs : SupClosed s) : DirectedOn (· ≤ ·) s :=
  fun _a ha _b hb ↦ ⟨_, hs ha hb, le_sup_left, le_sup_right⟩


lemma IsUpperSet.supClosed (hs : IsUpperSet s) : SupClosed s := fun _a _ _b ↦ hs le_sup_right


lemma SupClosed.preimage [FunLike F β α] [SupHomClass F β α] (hs : SupClosed s) (f : F) :
    SupClosed (f ⁻¹' s) :=
                     /-
                       F : Type u_2
                       α : Type u_3
                       β : Type u_4
                       inst✝³ : SemilatticeSup α
                       inst✝² : SemilatticeSup β
                       s : Set α
                       inst✝¹ : FunLike F β α
                       inst✝ : SupHomClass F β α
                       hs : SupClosed s
                       f : F
                       a : β
                       ha : Membership.mem (Set.preimage (⇑f) s) a
                       b : β
                       hb : Membership.mem (Set.preimage (⇑f) s) b
                       ⊢ Membership.mem (Set.preimage (⇑f) s) (Max.max a b)
                     -/
  fun a ha b hb ↦ by simpa [map_sup] using hs ha hb
                     /-
                       🎉 no goals
                     -/


lemma SupClosed.image [FunLike F α β] [SupHomClass F α β] (hs : SupClosed s) (f : F) :
    SupClosed (f '' s) := by
  /-
    F : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝³ : SemilatticeSup α
    inst✝² : SemilatticeSup β
    s : Set α
    inst✝¹ : FunLike F α β
    inst✝ : SupHomClass F α β
    hs : SupClosed s
    f : F
    ⊢ SupClosed (Set.image (⇑f) s)
  -/
  rintro _ ⟨a, ha, rfl⟩ _ ⟨b, hb, rfl⟩
  /-
    case intro.intro.intro.intro
    F : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝³ : SemilatticeSup α
    inst✝² : SemilatticeSup β
    s : Set α
    inst✝¹ : FunLike F α β
    inst✝ : SupHomClass F α β
    hs : SupClosed s
    f : F
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    ⊢ Membership.mem (Set.image (⇑f) s) (Max.max (f a) (f b))
  -/
  rw [← map_sup]
  /-
    case intro.intro.intro.intro
    F : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝³ : SemilatticeSup α
    inst✝² : SemilatticeSup β
    s : Set α
    inst✝¹ : FunLike F α β
    inst✝ : SupHomClass F α β
    hs : SupClosed s
    f : F
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    ⊢ Membership.mem (Set.image (⇑f) s) (f (Max.max a b))
  -/
  exact Set.mem_image_of_mem _ <| hs ha hb
  /-
    🎉 no goals
  -/


lemma supClosed_range [FunLike F α β] [SupHomClass F α β] (f : F) : SupClosed (Set.range f) := by
  /-
    F : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝³ : SemilatticeSup α
    inst✝² : SemilatticeSup β
    inst✝¹ : FunLike F α β
    inst✝ : SupHomClass F α β
    f : F
    ⊢ SupClosed (Set.range ⇑f)
  -/
  simpa using supClosed_univ.image f
  /-
    🎉 no goals
  -/


lemma SupClosed.prod {t : Set β} (hs : SupClosed s) (ht : SupClosed t) : SupClosed (s ×ˢ t) :=
  fun _a ha _b hb ↦ ⟨hs ha.1 hb.1, ht ha.2 hb.2⟩


lemma supClosed_pi {ι : Type*} {α : ι → Type*} [∀ i, SemilatticeSup (α i)] {s : Set ι}
    {t : ∀ i, Set (α i)} (ht : ∀ i ∈ s, SupClosed (t i)) : SupClosed (s.pi t) :=
  fun _a ha _b hb _i hi ↦ ht _ hi (ha _ hi) (hb _ hi)


lemma SupClosed.insert_upperBounds {s : Set α} {a : α} (hs : SupClosed s) (ha : a ∈ upperBounds s) :
    SupClosed (insert a s) := by
  /-
    α : Type u_3
    inst✝ : SemilatticeSup α
    s : Set α
    a : α
    hs : SupClosed s
    ha : Membership.mem (upperBounds s) a
    ⊢ SupClosed (Insert.insert a s)
  -/
  rw [SupClosed]
  /-
    α : Type u_3
    inst✝ : SemilatticeSup α
    s : Set α
    a : α
    hs : SupClosed s
    ha : Membership.mem (upperBounds s) a
    ⊢ ∀ ⦃a_1 : α⦄, Membership.mem (Insert.insert a s) a_1 → ∀ ⦃b : α⦄, Membership. …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma SupClosed.insert_lowerBounds {s : Set α} {a : α} (h : SupClosed s) (ha : a ∈ lowerBounds s) :
    SupClosed (insert a s) := by
  /-
    α : Type u_3
    inst✝ : SemilatticeSup α
    s : Set α
    a : α
    h : SupClosed s
    ha : Membership.mem (lowerBounds s) a
    ⊢ SupClosed (Insert.insert a s)
  -/
  rw [SupClosed]
  /-
    α : Type u_3
    inst✝ : SemilatticeSup α
    s : Set α
    a : α
    h : SupClosed s
    ha : Membership.mem (lowerBounds s) a
    ⊢ ∀ ⦃a_1 : α⦄, Membership.mem (Insert.insert a s) a_1 → ∀ ⦃b : α⦄, Membership. …
  -/
  have ha' : ∀ b ∈ s, a ≤ b := fun _ a ↦ ha a
  /-
    α : Type u_3
    inst✝ : SemilatticeSup α
    s : Set α
    a : α
    h : SupClosed s
    ha : Membership.mem (lowerBounds s) a
    ha' : ∀ (b : α), Membership.mem s b → LE.le a b
    ⊢ ∀ ⦃a_1 : α⦄, Membership.mem (Insert.insert a s) a_1 → ∀ ⦃b : α⦄, Membership. …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma SupClosed.finsetSup'_mem (hs : SupClosed s) (ht : t.Nonempty) :
    (∀ i ∈ t, f i ∈ s) → t.sup' ht f ∈ s :=
  sup'_induction _ _ hs


lemma SupClosed.finsetSup_mem [OrderBot α] (hs : SupClosed s) (ht : t.Nonempty) :
    (∀ i ∈ t, f i ∈ s) → t.sup f ∈ s :=
  sup'_eq_sup ht f ▸ hs.finsetSup'_mem ht


/-- A set `s` is *inf-closed* if `a ⊓ b ∈ s` for all `a ∈ s`, `b ∈ s`. -/
def InfClosed (s : Set α) : Prop := ∀ ⦃a⦄, a ∈ s → ∀ ⦃b⦄, b ∈ s → a ⊓ b ∈ s


                                                            /-
                                                              α : Type u_3
                                                              inst✝ : SemilatticeInf α
                                                              ⊢ InfClosed EmptyCollection.emptyCollection
                                                            -/
@[simp] lemma infClosed_empty : InfClosed (∅ : Set α) := by simp [InfClosed]
                                                            /-
                                                              🎉 no goals
                                                            -/

                                                                  /-
                                                                    α : Type u_3
                                                                    inst✝ : SemilatticeInf α
                                                                    a : α
                                                                    ⊢ InfClosed (Singleton.singleton a)
                                                                  -/
@[simp] lemma infClosed_singleton : InfClosed ({a} : Set α) := by simp [InfClosed]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


                                                              /-
                                                                α : Type u_3
                                                                inst✝ : SemilatticeInf α
                                                                ⊢ InfClosed Set.univ
                                                              -/
@[simp] lemma infClosed_univ : InfClosed (univ : Set α) := by simp [InfClosed]
                                                              /-
                                                                🎉 no goals
                                                              -/

lemma InfClosed.inter (hs : InfClosed s) (ht : InfClosed t) : InfClosed (s ∩ t) :=
  fun _a ha _b hb ↦ ⟨hs ha.1 hb.1, ht ha.2 hb.2⟩


lemma infClosed_sInter (hS : ∀ s ∈ S, InfClosed s) : InfClosed (⋂₀ S) :=
  fun _a ha _b hb _s hs ↦ hS _ hs (ha _ hs) (hb _ hs)


lemma infClosed_iInter (hf : ∀ i, InfClosed (f i)) : InfClosed (⋂ i, f i) :=
  infClosed_sInter <| forall_mem_range.2 hf


lemma InfClosed.codirectedOn (hs : InfClosed s) : DirectedOn (· ≥ ·) s :=
  fun _a ha _b hb ↦ ⟨_, hs ha hb, inf_le_left, inf_le_right⟩


lemma IsLowerSet.infClosed (hs : IsLowerSet s) : InfClosed s := fun _a _ _b ↦ hs inf_le_right


lemma InfClosed.preimage [FunLike F β α] [InfHomClass F β α] (hs : InfClosed s) (f : F) :
    InfClosed (f ⁻¹' s) :=
                     /-
                       F : Type u_2
                       α : Type u_3
                       β : Type u_4
                       inst✝³ : SemilatticeInf α
                       inst✝² : SemilatticeInf β
                       s : Set α
                       inst✝¹ : FunLike F β α
                       inst✝ : InfHomClass F β α
                       hs : InfClosed s
                       f : F
                       a : β
                       ha : Membership.mem (Set.preimage (⇑f) s) a
                       b : β
                       hb : Membership.mem (Set.preimage (⇑f) s) b
                       ⊢ Membership.mem (Set.preimage (⇑f) s) (Min.min a b)
                     -/
  fun a ha b hb ↦ by simpa [map_inf] using hs ha hb
                     /-
                       🎉 no goals
                     -/


lemma InfClosed.image [FunLike F α β] [InfHomClass F α β] (hs : InfClosed s) (f : F) :
    InfClosed (f '' s) := by
  /-
    F : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝³ : SemilatticeInf α
    inst✝² : SemilatticeInf β
    s : Set α
    inst✝¹ : FunLike F α β
    inst✝ : InfHomClass F α β
    hs : InfClosed s
    f : F
    ⊢ InfClosed (Set.image (⇑f) s)
  -/
  rintro _ ⟨a, ha, rfl⟩ _ ⟨b, hb, rfl⟩
  /-
    case intro.intro.intro.intro
    F : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝³ : SemilatticeInf α
    inst✝² : SemilatticeInf β
    s : Set α
    inst✝¹ : FunLike F α β
    inst✝ : InfHomClass F α β
    hs : InfClosed s
    f : F
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    ⊢ Membership.mem (Set.image (⇑f) s) (Min.min (f a) (f b))
  -/
  rw [← map_inf]
  /-
    case intro.intro.intro.intro
    F : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝³ : SemilatticeInf α
    inst✝² : SemilatticeInf β
    s : Set α
    inst✝¹ : FunLike F α β
    inst✝ : InfHomClass F α β
    hs : InfClosed s
    f : F
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    ⊢ Membership.mem (Set.image (⇑f) s) (f (Min.min a b))
  -/
  exact Set.mem_image_of_mem _ <| hs ha hb
  /-
    🎉 no goals
  -/


lemma infClosed_range [FunLike F α β] [InfHomClass F α β] (f : F) : InfClosed (Set.range f) := by
  /-
    F : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝³ : SemilatticeInf α
    inst✝² : SemilatticeInf β
    inst✝¹ : FunLike F α β
    inst✝ : InfHomClass F α β
    f : F
    ⊢ InfClosed (Set.range ⇑f)
  -/
  simpa using infClosed_univ.image f
  /-
    🎉 no goals
  -/


lemma InfClosed.prod {t : Set β} (hs : InfClosed s) (ht : InfClosed t) : InfClosed (s ×ˢ t) :=
  fun _a ha _b hb ↦ ⟨hs ha.1 hb.1, ht ha.2 hb.2⟩


lemma infClosed_pi {ι : Type*} {α : ι → Type*} [∀ i, SemilatticeInf (α i)] {s : Set ι}
    {t : ∀ i, Set (α i)} (ht : ∀ i ∈ s, InfClosed (t i)) : InfClosed (s.pi t) :=
  fun _a ha _b hb _i hi ↦ ht _ hi (ha _ hi) (hb _ hi)


lemma InfClosed.insert_upperBounds {s : Set α} {a : α} (hs : InfClosed s) (ha : a ∈ upperBounds s) :
    InfClosed (insert a s) := by
  /-
    α : Type u_3
    inst✝ : SemilatticeInf α
    s : Set α
    a : α
    hs : InfClosed s
    ha : Membership.mem (upperBounds s) a
    ⊢ InfClosed (Insert.insert a s)
  -/
  rw [InfClosed]
  /-
    α : Type u_3
    inst✝ : SemilatticeInf α
    s : Set α
    a : α
    hs : InfClosed s
    ha : Membership.mem (upperBounds s) a
    ⊢ ∀ ⦃a_1 : α⦄, Membership.mem (Insert.insert a s) a_1 → ∀ ⦃b : α⦄, Membership. …
  -/
  have ha' : ∀ b ∈ s, b ≤ a := fun _ a ↦ ha a
  /-
    α : Type u_3
    inst✝ : SemilatticeInf α
    s : Set α
    a : α
    hs : InfClosed s
    ha : Membership.mem (upperBounds s) a
    ha' : ∀ (b : α), Membership.mem s b → LE.le b a
    ⊢ ∀ ⦃a_1 : α⦄, Membership.mem (Insert.insert a s) a_1 → ∀ ⦃b : α⦄, Membership. …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma InfClosed.insert_lowerBounds {s : Set α} {a : α} (h : InfClosed s) (ha : a ∈ lowerBounds s) :
    InfClosed (insert a s) := by
  /-
    α : Type u_3
    inst✝ : SemilatticeInf α
    s : Set α
    a : α
    h : InfClosed s
    ha : Membership.mem (lowerBounds s) a
    ⊢ InfClosed (Insert.insert a s)
  -/
  rw [InfClosed]
  /-
    α : Type u_3
    inst✝ : SemilatticeInf α
    s : Set α
    a : α
    h : InfClosed s
    ha : Membership.mem (lowerBounds s) a
    ⊢ ∀ ⦃a_1 : α⦄, Membership.mem (Insert.insert a s) a_1 → ∀ ⦃b : α⦄, Membership. …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma InfClosed.finsetInf'_mem (hs : InfClosed s) (ht : t.Nonempty) :
    (∀ i ∈ t, f i ∈ s) → t.inf' ht f ∈ s :=
  inf'_induction _ _ hs


lemma InfClosed.finsetInf_mem [OrderTop α] (hs : InfClosed s) (ht : t.Nonempty) :
    (∀ i ∈ t, f i ∈ s) → t.inf f ∈ s :=
  inf'_eq_inf ht f ▸ hs.finsetInf'_mem ht


/-- A set `s` is a *sublattice* if `a ⊔ b ∈ s` and `a ⊓ b ∈ s` for all `a ∈ s`, `b ∈ s`.
Note: This is not the preferred way to declare a sublattice. One should instead use `Sublattice`.
TODO: Define `Sublattice`. -/
structure IsSublattice (s : Set α) : Prop where
  supClosed : SupClosed s
  infClosed : InfClosed s


@[simp] lemma isSublattice_empty : IsSublattice (∅ : Set α) := ⟨supClosed_empty, infClosed_empty⟩

@[simp] lemma isSublattice_singleton : IsSublattice ({a} : Set α) :=
  ⟨supClosed_singleton, infClosed_singleton⟩


@[simp] lemma isSublattice_univ : IsSublattice (Set.univ : Set α) :=
  ⟨supClosed_univ, infClosed_univ⟩


lemma IsSublattice.inter (hs : IsSublattice s) (ht : IsSublattice t) : IsSublattice (s ∩ t) :=
  ⟨hs.1.inter ht.1, hs.2.inter ht.2⟩


lemma isSublattice_sInter (hS : ∀ s ∈ S, IsSublattice s) : IsSublattice (⋂₀ S) :=
  ⟨supClosed_sInter fun _s hs ↦ (hS _ hs).1, infClosed_sInter fun _s hs ↦ (hS _ hs).2⟩


lemma isSublattice_iInter (hf : ∀ i, IsSublattice (f i)) : IsSublattice (⋂ i, f i) :=
  ⟨supClosed_iInter fun _i ↦ (hf _).1, infClosed_iInter fun _i ↦ (hf _).2⟩


lemma IsSublattice.preimage [FunLike F β α] [LatticeHomClass F β α]
    (hs : IsSublattice s) (f : F) :
    IsSublattice (f ⁻¹' s) := ⟨hs.1.preimage _, hs.2.preimage _⟩


lemma IsSublattice.image [FunLike F α β] [LatticeHomClass F α β] (hs : IsSublattice s) (f : F) :
    IsSublattice (f '' s) := ⟨hs.1.image _, hs.2.image _⟩


lemma IsSublattice_range [FunLike F α β] [LatticeHomClass F α β] (f : F) :
    IsSublattice (Set.range f) :=
  ⟨supClosed_range _, infClosed_range _⟩


lemma IsSublattice.prod {t : Set β} (hs : IsSublattice s) (ht : IsSublattice t) :
    IsSublattice (s ×ˢ t) := ⟨hs.1.prod ht.1, hs.2.prod ht.2⟩


lemma isSublattice_pi {ι : Type*} {α : ι → Type*} [∀ i, Lattice (α i)] {s : Set ι}
    {t : ∀ i, Set (α i)} (ht : ∀ i ∈ s, IsSublattice (t i)) : IsSublattice (s.pi t) :=
  ⟨supClosed_pi fun _i hi ↦ (ht _ hi).1, infClosed_pi fun _i hi ↦ (ht _ hi).2⟩


@[simp] lemma supClosed_preimage_toDual {s : Set αᵒᵈ} :
    SupClosed (toDual ⁻¹' s) ↔ InfClosed s := Iff.rfl


@[simp] lemma infClosed_preimage_toDual {s : Set αᵒᵈ} :
    InfClosed (toDual ⁻¹' s) ↔ SupClosed s := Iff.rfl


@[simp] lemma supClosed_preimage_ofDual {s : Set α} :
    SupClosed (ofDual ⁻¹' s) ↔ InfClosed s := Iff.rfl


@[simp] lemma infClosed_preimage_ofDual {s : Set α} :
    InfClosed (ofDual ⁻¹' s) ↔ SupClosed s := Iff.rfl


@[simp] lemma isSublattice_preimage_toDual {s : Set αᵒᵈ} :
    IsSublattice (toDual ⁻¹' s) ↔ IsSublattice s := ⟨fun h ↦ ⟨h.2, h.1⟩, fun h ↦ ⟨h.2, h.1⟩⟩


@[simp] lemma isSublattice_preimage_ofDual :
    IsSublattice (ofDual ⁻¹' s) ↔ IsSublattice s := ⟨fun h ↦ ⟨h.2, h.1⟩, fun h ↦ ⟨h.2, h.1⟩⟩


alias ⟨_, InfClosed.dual⟩ := supClosed_preimage_ofDual

alias ⟨_, SupClosed.dual⟩ := infClosed_preimage_ofDual

alias ⟨_, IsSublattice.dual⟩ := isSublattice_preimage_ofDual

alias ⟨_, IsSublattice.of_dual⟩ := isSublattice_preimage_toDual


@[simp] protected lemma LinearOrder.supClosed (s : Set α) : SupClosed s :=
                     /-
                       α : Type u_3
                       inst✝ : LinearOrder α
                       s : Set α
                       a : α
                       ha : Membership.mem s a
                       b : α
                       hb : Membership.mem s b
                       ⊢ Membership.mem s (Max.max a b)
                     -/
                                            /-
                                              🎉 no goals
                                            -/
  fun a ha b hb ↦ by cases le_total a b <;> simp [*]
                                            /-
                                              🎉 no goals
                                            -/


@[simp] protected lemma LinearOrder.infClosed (s : Set α) : InfClosed s :=
                     /-
                       α : Type u_3
                       inst✝ : LinearOrder α
                       s : Set α
                       a : α
                       ha : Membership.mem s a
                       b : α
                       hb : Membership.mem s b
                       ⊢ Membership.mem s (Min.min a b)
                     -/
                                            /-
                                              🎉 no goals
                                            -/
  fun a ha b hb ↦ by cases le_total a b <;> simp [*]
                                            /-
                                              🎉 no goals
                                            -/


@[simp] protected lemma LinearOrder.isSublattice (s : Set α) : IsSublattice s :=
  ⟨LinearOrder.supClosed _, LinearOrder.infClosed _⟩


/-- Every set in a join-semilattice generates a set closed under join. -/
@[simps! isClosed]
def supClosure : ClosureOperator (Set α) := .ofPred
  (fun s ↦ {a | ∃ (t : Finset α) (ht : t.Nonempty), ↑t ⊆ s ∧ t.sup' ht id = a})
  SupClosed
                                               /-
                                                 ι : Sort u_1
                                                 F : Type u_2
                                                 α : Type u_3
                                                 β : Type u_4
                                                 inst✝¹ : SemilatticeSup α
                                                 inst✝ : SemilatticeSup β
                                                 s✝ t : Set α
                                                 a✝ b : α
                                                 s : Set α
                                                 a : α
                                                 ha : Membership.mem s a
                                                 ⊢ And (HasSubset.Subset (↑(Singleton.singleton a)) s) (Eq ((Singleton.singleto …
                                               -/
  (fun s a ha ↦ ⟨{a}, singleton_nonempty _, by simpa⟩)
                                               /-
                                                 🎉 no goals
                                               -/
  (by
    classical
    rintro s _ ⟨t, ht, hts, rfl⟩ _ ⟨u, hu, hus, rfl⟩
    refine ⟨_, ht.mono subset_union_left, ?_, sup'_union ht hu _⟩
    rw [coe_union]
    exact Set.union_subset hts hus)
      /-
        ι : Sort u_1
        F : Type u_2
        α : Type u_3
        β : Type u_4
        inst✝¹ : SemilatticeSup α
        inst✝ : SemilatticeSup β
        s t : Set α
        a b : α
        ⊢ ∀ ⦃x y : Set α⦄, LE.le x y → SupClosed y → LE.le ((fun s => setOf fun a => E …
      -/
  (by rintro s₁ s₂ hs h₂ _ ⟨t, ht, hts, rfl⟩; exact h₂.finsetSup'_mem ht fun i hi ↦ hs <| hts hi)
                                              /-
                                                🎉 no goals
                                              -/


@[simp] lemma subset_supClosure {s : Set α} : s ⊆ supClosure s := supClosure.le_closure _


@[simp] lemma supClosed_supClosure : SupClosed (supClosure s) := supClosure.isClosed_closure _


lemma supClosure_mono : Monotone (supClosure : Set α → Set α) := supClosure.monotone


@[simp] lemma supClosure_eq_self : supClosure s = s ↔ SupClosed s := supClosure.isClosed_iff.symm


alias ⟨_, SupClosed.supClosure_eq⟩ := supClosure_eq_self


lemma supClosure_idem (s : Set α) : supClosure (supClosure s) = supClosure s :=
  supClosure.idempotent _


                                                                  /-
                                                                    α : Type u_3
                                                                    inst✝ : SemilatticeSup α
                                                                    ⊢ Eq (supClosure EmptyCollection.emptyCollection) EmptyCollection.emptyCollect …
                                                                  -/
@[simp] lemma supClosure_empty : supClosure (∅ : Set α) = ∅ := by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/

                                                                /-
                                                                  α : Type u_3
                                                                  inst✝ : SemilatticeSup α
                                                                  a : α
                                                                  ⊢ Eq (supClosure (Singleton.singleton a)) (Singleton.singleton a)
                                                                -/
@[simp] lemma supClosure_singleton : supClosure {a} = {a} := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/

                                                                               /-
                                                                                 α : Type u_3
                                                                                 inst✝ : SemilatticeSup α
                                                                                 ⊢ Eq (supClosure Set.univ) Set.univ
                                                                               -/
@[simp] lemma supClosure_univ : supClosure (Set.univ : Set α) = Set.univ := by simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[simp] lemma upperBounds_supClosure (s : Set α) : upperBounds (supClosure s) = upperBounds s :=
  (upperBounds_mono_set subset_supClosure).antisymm <| by
    /-
      α : Type u_3
      inst✝ : SemilatticeSup α
      s : Set α
      ⊢ HasSubset.Subset (upperBounds s) (upperBounds (supClosure s))
    -/
    rintro a ha _ ⟨t, ht, hts, rfl⟩
    /-
      case intro.intro.intro
      α : Type u_3
      inst✝ : SemilatticeSup α
      s : Set α
      a : α
      ha : Membership.mem (upperBounds s) a
      t : Finset α
      ht : t.Nonempty
      hts : HasSubset.Subset (↑t) s
      ⊢ LE.le (t.sup' ht id) a
    -/
    exact sup'_le _ _ fun b hb ↦ ha <| hts hb
    /-
      🎉 no goals
    -/


                                                                          /-
                                                                            α : Type u_3
                                                                            inst✝ : SemilatticeSup α
                                                                            s : Set α
                                                                            a : α
                                                                            ⊢ Iff (IsLUB (supClosure s) a) (IsLUB s a)
                                                                          -/
@[simp] lemma isLUB_supClosure : IsLUB (supClosure s) a ↔ IsLUB s a := by simp [IsLUB]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


lemma sup_mem_supClosure (ha : a ∈ s) (hb : b ∈ s) : a ⊔ b ∈ supClosure s :=
  supClosed_supClosure (subset_supClosure ha) (subset_supClosure hb)


lemma finsetSup'_mem_supClosure {ι : Type*} {t : Finset ι} (ht : t.Nonempty) {f : ι → α}
    (hf : ∀ i ∈ t, f i ∈ s) : t.sup' ht f ∈ supClosure s :=
  supClosed_supClosure.finsetSup'_mem _ fun _i hi ↦ subset_supClosure <| hf _ hi


lemma supClosure_min : s ⊆ t → SupClosed t → supClosure s ⊆ t := supClosure.closure_min


/-- The semilatice generated by a finite set is finite. -/
protected lemma Set.Finite.supClosure (hs : s.Finite) : (supClosure s).Finite := by
  /-
    α : Type u_3
    inst✝ : SemilatticeSup α
    s : Set α
    hs : s.Finite
    ⊢ (supClosure s).Finite
  -/
  lift s to Finset α using hs
  classical
  refine ({t ∈ s.powerset | t.Nonempty}.attach.image
    fun t ↦ t.1.sup' (mem_filter.1 t.2).2 id).finite_toSet.subset ?_
  rintro _ ⟨t, ht, hts, rfl⟩
  simp only [id_eq, coe_image, mem_image, mem_coe, mem_attach, true_and, Subtype.exists,
    Finset.mem_powerset, Finset.not_nonempty_iff_eq_empty, mem_filter]
  exact ⟨t, ⟨hts, ht⟩, rfl⟩


@[simp] lemma supClosure_prod (s : Set α) (t : Set β) :
    supClosure (s ×ˢ t) = supClosure s ×ˢ supClosure t :=
  le_antisymm (supClosure_min (Set.prod_mono subset_supClosure subset_supClosure) <|
    supClosed_supClosure.prod supClosed_supClosure) <| by
      /-
        α : Type u_3
        β : Type u_4
        inst✝¹ : SemilatticeSup α
        inst✝ : SemilatticeSup β
        s : Set α
        t : Set β
        ⊢ LE.le (SProd.sprod (supClosure s) (supClosure t)) (supClosure (SProd.sprod s …
      -/
      rintro ⟨_, _⟩ ⟨⟨u, hu, hus, rfl⟩, v, hv, hvt, rfl⟩
      /-
        case mk.intro.intro.intro.intro.intro.intro.intro
        α : Type u_3
        β : Type u_4
        inst✝¹ : SemilatticeSup α
        inst✝ : SemilatticeSup β
        s : Set α
        t : Set β
        u : Finset α
        hu : u.Nonempty
        hus : HasSubset.Subset (↑u) s
        v : Finset β
        hv : v.Nonempty
        hvt : HasSubset.Subset (↑v) t
        ⊢ Membership.mem (supClosure (SProd.sprod s t)) { fst := u.sup' hu id, snd :=  …
      -/
      refine ⟨u ×ˢ v, hu.product hv, ?_, ?_⟩
        /-
          case mk.intro.intro.intro.intro.intro.intro.intro.refine_1
          α : Type u_3
          β : Type u_4
          inst✝¹ : SemilatticeSup α
          inst✝ : SemilatticeSup β
          s : Set α
          t : Set β
          u : Finset α
          hu : u.Nonempty
          hus : HasSubset.Subset (↑u) s
          v : Finset β
          hv : v.Nonempty
          hvt : HasSubset.Subset (↑v) t
          ⊢ HasSubset.Subset (↑(SProd.sprod u v)) (SProd.sprod s t)
        -/
      · simpa only [coe_product] using Set.prod_mono hus hvt
        /-
          🎉 no goals
        -/
        /-
          case mk.intro.intro.intro.intro.intro.intro.intro.refine_2
          α : Type u_3
          β : Type u_4
          inst✝¹ : SemilatticeSup α
          inst✝ : SemilatticeSup β
          s : Set α
          t : Set β
          u : Finset α
          hu : u.Nonempty
          hus : HasSubset.Subset (↑u) s
          v : Finset β
          hv : v.Nonempty
          hvt : HasSubset.Subset (↑v) t
          ⊢ Eq ((SProd.sprod u v).sup' ⋯ id) { fst := u.sup' hu id, snd := v.sup' hv id }
        -/
      · simp [prodMk_sup'_sup']
        /-
          🎉 no goals
        -/


/-- Every set in a join-semilattice generates a set closed under join. -/
@[simps! isClosed]
def infClosure : ClosureOperator (Set α) := ClosureOperator.ofPred
  (fun s ↦ {a | ∃ (t : Finset α) (ht : t.Nonempty), ↑t ⊆ s ∧ t.inf' ht id = a})
  InfClosed
                                               /-
                                                 ι : Sort u_1
                                                 F : Type u_2
                                                 α : Type u_3
                                                 β : Type u_4
                                                 inst✝¹ : SemilatticeInf α
                                                 inst✝ : SemilatticeInf β
                                                 s✝ t : Set α
                                                 a✝ b : α
                                                 s : Set α
                                                 a : α
                                                 ha : Membership.mem s a
                                                 ⊢ And (HasSubset.Subset (↑(Singleton.singleton a)) s) (Eq ((Singleton.singleto …
                                               -/
  (fun s a ha ↦ ⟨{a}, singleton_nonempty _, by simpa⟩)
                                               /-
                                                 🎉 no goals
                                               -/
  (by
    classical
    rintro s _ ⟨t, ht, hts, rfl⟩ _ ⟨u, hu, hus, rfl⟩
    refine ⟨_, ht.mono subset_union_left, ?_, inf'_union ht hu _⟩
    rw [coe_union]
    exact Set.union_subset hts hus)
      /-
        ι : Sort u_1
        F : Type u_2
        α : Type u_3
        β : Type u_4
        inst✝¹ : SemilatticeInf α
        inst✝ : SemilatticeInf β
        s t : Set α
        a b : α
        ⊢ ∀ ⦃x y : Set α⦄, LE.le x y → InfClosed y → LE.le ((fun s => setOf fun a => E …
      -/
  (by rintro s₁ s₂ hs h₂ _ ⟨t, ht, hts, rfl⟩; exact h₂.finsetInf'_mem ht fun i hi ↦ hs <| hts hi)
                                              /-
                                                🎉 no goals
                                              -/


@[simp] lemma subset_infClosure {s : Set α} : s ⊆ infClosure s := infClosure.le_closure _


@[simp] lemma infClosed_infClosure : InfClosed (infClosure s) := infClosure.isClosed_closure _


lemma infClosure_mono : Monotone (infClosure : Set α → Set α) := infClosure.monotone


@[simp] lemma infClosure_eq_self : infClosure s = s ↔ InfClosed s := infClosure.isClosed_iff.symm


alias ⟨_, InfClosed.infClosure_eq⟩ := infClosure_eq_self


lemma infClosure_idem (s : Set α) : infClosure (infClosure s) = infClosure s :=
  infClosure.idempotent _


                                                                  /-
                                                                    α : Type u_3
                                                                    inst✝ : SemilatticeInf α
                                                                    ⊢ Eq (infClosure EmptyCollection.emptyCollection) EmptyCollection.emptyCollect …
                                                                  -/
@[simp] lemma infClosure_empty : infClosure (∅ : Set α) = ∅ := by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/

                                                                /-
                                                                  α : Type u_3
                                                                  inst✝ : SemilatticeInf α
                                                                  a : α
                                                                  ⊢ Eq (infClosure (Singleton.singleton a)) (Singleton.singleton a)
                                                                -/
@[simp] lemma infClosure_singleton : infClosure {a} = {a} := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/

                                                                               /-
                                                                                 α : Type u_3
                                                                                 inst✝ : SemilatticeInf α
                                                                                 ⊢ Eq (infClosure Set.univ) Set.univ
                                                                               -/
@[simp] lemma infClosure_univ : infClosure (Set.univ : Set α) = Set.univ := by simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[simp] lemma lowerBounds_infClosure (s : Set α) : lowerBounds (infClosure s) = lowerBounds s :=
  (lowerBounds_mono_set subset_infClosure).antisymm <| by
    /-
      α : Type u_3
      inst✝ : SemilatticeInf α
      s : Set α
      ⊢ HasSubset.Subset (lowerBounds s) (lowerBounds (infClosure s))
    -/
    rintro a ha _ ⟨t, ht, hts, rfl⟩
    /-
      case intro.intro.intro
      α : Type u_3
      inst✝ : SemilatticeInf α
      s : Set α
      a : α
      ha : Membership.mem (lowerBounds s) a
      t : Finset α
      ht : t.Nonempty
      hts : HasSubset.Subset (↑t) s
      ⊢ LE.le a (t.inf' ht id)
    -/
    exact le_inf' _ _ fun b hb ↦ ha <| hts hb
    /-
      🎉 no goals
    -/


                                                                          /-
                                                                            α : Type u_3
                                                                            inst✝ : SemilatticeInf α
                                                                            s : Set α
                                                                            a : α
                                                                            ⊢ Iff (IsGLB (infClosure s) a) (IsGLB s a)
                                                                          -/
@[simp] lemma isGLB_infClosure : IsGLB (infClosure s) a ↔ IsGLB s a := by simp [IsGLB]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


lemma inf_mem_infClosure (ha : a ∈ s) (hb : b ∈ s) : a ⊓ b ∈ infClosure s :=
  infClosed_infClosure (subset_infClosure ha) (subset_infClosure hb)


lemma finsetInf'_mem_infClosure {ι : Type*} {t : Finset ι} (ht : t.Nonempty) {f : ι → α}
    (hf : ∀ i ∈ t, f i ∈ s) : t.inf' ht f ∈ infClosure s :=
  infClosed_infClosure.finsetInf'_mem _ fun _i hi ↦ subset_infClosure <| hf _ hi


lemma infClosure_min : s ⊆ t → InfClosed t → infClosure s ⊆ t := infClosure.closure_min


/-- The semilatice generated by a finite set is finite. -/
protected lemma Set.Finite.infClosure (hs : s.Finite) : (infClosure s).Finite := by
  /-
    α : Type u_3
    inst✝ : SemilatticeInf α
    s : Set α
    hs : s.Finite
    ⊢ (infClosure s).Finite
  -/
  lift s to Finset α using hs
  classical
  refine ({t ∈ s.powerset | t.Nonempty}.attach.image
    fun t ↦ t.1.inf' (mem_filter.1 t.2).2 id).finite_toSet.subset ?_
  rintro _ ⟨t, ht, hts, rfl⟩
  simp only [id_eq, coe_image, mem_image, mem_coe, mem_attach, true_and, Subtype.exists,
    Finset.mem_powerset, Finset.not_nonempty_iff_eq_empty, mem_filter]
  exact ⟨t, ⟨hts, ht⟩, rfl⟩


@[simp] lemma infClosure_prod (s : Set α) (t : Set β) :
    infClosure (s ×ˢ t) = infClosure s ×ˢ infClosure t :=
  le_antisymm (infClosure_min (Set.prod_mono subset_infClosure subset_infClosure) <|
    infClosed_infClosure.prod infClosed_infClosure) <| by
      /-
        α : Type u_3
        β : Type u_4
        inst✝¹ : SemilatticeInf α
        inst✝ : SemilatticeInf β
        s : Set α
        t : Set β
        ⊢ LE.le (SProd.sprod (infClosure s) (infClosure t)) (infClosure (SProd.sprod s …
      -/
      rintro ⟨_, _⟩ ⟨⟨u, hu, hus, rfl⟩, v, hv, hvt, rfl⟩
      /-
        case mk.intro.intro.intro.intro.intro.intro.intro
        α : Type u_3
        β : Type u_4
        inst✝¹ : SemilatticeInf α
        inst✝ : SemilatticeInf β
        s : Set α
        t : Set β
        u : Finset α
        hu : u.Nonempty
        hus : HasSubset.Subset (↑u) s
        v : Finset β
        hv : v.Nonempty
        hvt : HasSubset.Subset (↑v) t
        ⊢ Membership.mem (infClosure (SProd.sprod s t)) { fst := u.inf' hu id, snd :=  …
      -/
      refine ⟨u ×ˢ v, hu.product hv, ?_, ?_⟩
        /-
          case mk.intro.intro.intro.intro.intro.intro.intro.refine_1
          α : Type u_3
          β : Type u_4
          inst✝¹ : SemilatticeInf α
          inst✝ : SemilatticeInf β
          s : Set α
          t : Set β
          u : Finset α
          hu : u.Nonempty
          hus : HasSubset.Subset (↑u) s
          v : Finset β
          hv : v.Nonempty
          hvt : HasSubset.Subset (↑v) t
          ⊢ HasSubset.Subset (↑(SProd.sprod u v)) (SProd.sprod s t)
        -/
      · simpa only [coe_product] using Set.prod_mono hus hvt
        /-
          🎉 no goals
        -/
        /-
          case mk.intro.intro.intro.intro.intro.intro.intro.refine_2
          α : Type u_3
          β : Type u_4
          inst✝¹ : SemilatticeInf α
          inst✝ : SemilatticeInf β
          s : Set α
          t : Set β
          u : Finset α
          hu : u.Nonempty
          hus : HasSubset.Subset (↑u) s
          v : Finset β
          hv : v.Nonempty
          hvt : HasSubset.Subset (↑v) t
          ⊢ Eq ((SProd.sprod u v).inf' ⋯ id) { fst := u.inf' hu id, snd := v.inf' hv id }
        -/
      · simp [prodMk_inf'_inf']
        /-
          🎉 no goals
        -/


/-- Every set in a join-semilattice generates a set closed under join. -/
@[simps! isClosed]
def latticeClosure : ClosureOperator (Set α) :=
  .ofCompletePred IsSublattice fun _ ↦ isSublattice_sInter


@[simp] lemma subset_latticeClosure : s ⊆ latticeClosure s := latticeClosure.le_closure _


@[simp] lemma isSublattice_latticeClosure : IsSublattice (latticeClosure s) :=
  latticeClosure.isClosed_closure _


lemma latticeClosure_min : s ⊆ t → IsSublattice t → latticeClosure s ⊆ t :=
  latticeClosure.closure_min


lemma latticeClosure_mono : Monotone (latticeClosure : Set α → Set α) := latticeClosure.monotone


@[simp] lemma latticeClosure_eq_self : latticeClosure s = s ↔ IsSublattice s :=
  latticeClosure.isClosed_iff.symm


alias ⟨_, IsSublattice.latticeClosure_eq⟩ := latticeClosure_eq_self


lemma latticeClosure_idem (s : Set α) : latticeClosure (latticeClosure s) = latticeClosure s :=
  latticeClosure.idempotent _


                                                                          /-
                                                                            α : Type u_3
                                                                            inst✝ : Lattice α
                                                                            ⊢ Eq (latticeClosure EmptyCollection.emptyCollection) EmptyCollection.emptyCol …
                                                                          -/
@[simp] lemma latticeClosure_empty : latticeClosure (∅ : Set α) = ∅ := by simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/

                                                                                /-
                                                                                  α : Type u_3
                                                                                  inst✝ : Lattice α
                                                                                  a : α
                                                                                  ⊢ Eq (latticeClosure (Singleton.singleton a)) (Singleton.singleton a)
                                                                                -/
@[simp] lemma latticeClosure_singleton (a : α) : latticeClosure {a} = {a} := by simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/

                                                                                       /-
                                                                                         α : Type u_3
                                                                                         inst✝ : Lattice α
                                                                                         ⊢ Eq (latticeClosure Set.univ) Set.univ
                                                                                       -/
@[simp] lemma latticeClosure_univ : latticeClosure (Set.univ : Set α) = Set.univ := by simp
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


protected lemma SupClosed.infClosure (hs : SupClosed s) : SupClosed (infClosure s) := by
  /-
    α : Type u_3
    inst✝ : DistribLattice α
    s : Set α
    hs : SupClosed s
    ⊢ SupClosed (infClosure s)
  -/
  rintro _ ⟨t, ht, hts, rfl⟩ _ ⟨u, hu, hus, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_3
    inst✝ : DistribLattice α
    s : Set α
    hs : SupClosed s
    t : Finset α
    ht : t.Nonempty
    hts : HasSubset.Subset (↑t) s
    u : Finset α
    hu : u.Nonempty
    hus : HasSubset.Subset (↑u) s
    ⊢ Membership.mem (infClosure s) (Max.max (t.inf' ht id) (u.inf' hu id))
  -/
  rw [inf'_sup_inf']
  exact finsetInf'_mem_infClosure _
    fun i hi ↦ hs (hts (mem_product.1 hi).1) (hus (mem_product.1 hi).2)


protected lemma InfClosed.supClosure (hs : InfClosed s) : InfClosed (supClosure s) := by
  /-
    α : Type u_3
    inst✝ : DistribLattice α
    s : Set α
    hs : InfClosed s
    ⊢ InfClosed (supClosure s)
  -/
  rintro _ ⟨t, ht, hts, rfl⟩ _ ⟨u, hu, hus, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_3
    inst✝ : DistribLattice α
    s : Set α
    hs : InfClosed s
    t : Finset α
    ht : t.Nonempty
    hts : HasSubset.Subset (↑t) s
    u : Finset α
    hu : u.Nonempty
    hus : HasSubset.Subset (↑u) s
    ⊢ Membership.mem (supClosure s) (Min.min (t.sup' ht id) (u.sup' hu id))
  -/
  rw [sup'_inf_sup']
  exact finsetSup'_mem_supClosure _
    fun i hi ↦ hs (hts (mem_product.1 hi).1) (hus (mem_product.1 hi).2)


@[simp] lemma supClosure_infClosure (s : Set α) : supClosure (infClosure s) = latticeClosure s :=
  le_antisymm (supClosure_min (infClosure_min subset_latticeClosure isSublattice_latticeClosure.2)
    isSublattice_latticeClosure.1) <| latticeClosure_min (subset_infClosure.trans subset_supClosure)
      ⟨supClosed_supClosure, infClosed_infClosure.supClosure⟩


@[simp] lemma infClosure_supClosure (s : Set α) : infClosure (supClosure s) = latticeClosure s :=
  le_antisymm (infClosure_min (supClosure_min subset_latticeClosure isSublattice_latticeClosure.1)
    isSublattice_latticeClosure.2) <| latticeClosure_min (subset_supClosure.trans subset_infClosure)
      ⟨supClosed_supClosure.infClosure, infClosed_infClosure⟩


lemma Set.Finite.latticeClosure (hs : s.Finite) : (latticeClosure s).Finite := by
  /-
    α : Type u_3
    inst✝ : DistribLattice α
    s : Set α
    hs : s.Finite
    ⊢ (_root_.latticeClosure s).Finite
  -/
  rw [← supClosure_infClosure]; exact hs.infClosure.supClosure
                                /-
                                  🎉 no goals
                                -/


@[simp] lemma latticeClosure_prod (s : Set α) (t : Set β) :
    latticeClosure (s ×ˢ t) = latticeClosure s ×ˢ latticeClosure t := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝¹ : DistribLattice α
    inst✝ : DistribLattice β
    s : Set α
    t : Set β
    ⊢ Eq (latticeClosure (SProd.sprod s t)) (SProd.sprod (latticeClosure s) (latti …
  -/
  simp_rw [← supClosure_infClosure]; simp
                                     /-
                                       🎉 no goals
                                     -/


/-- A join-semilattice where every sup-closed set has a least upper bound is automatically complete.
-/
def SemilatticeSup.toCompleteSemilatticeSup [SemilatticeSup α] (sSup : Set α → α)
    (h : ∀ s, SupClosed s → IsLUB s (sSup s)) : CompleteSemilatticeSup α where
  sSup := fun s => sSup (supClosure s)
  le_sSup _ _ ha := (h _ supClosed_supClosure).1 <| subset_supClosure ha
                                                                       /-
                                                                         ι : Sort u_1
                                                                         F : Type u_2
                                                                         α : Type u_3
                                                                         β : Type u_4
                                                                         inst✝ : SemilatticeSup α
                                                                         sSup : Set α → α
                                                                         h : ∀ (s : Set α), SupClosed s → IsLUB s (sSup s)
                                                                         s : Set α
                                                                         a : α
                                                                         ha : ∀ (b : α), Membership.mem s b → LE.le b a
                                                                         ⊢ Membership.mem (upperBounds (supClosure s)) a
                                                                       -/
  sSup_le s a ha := (isLUB_le_iff <| h _ supClosed_supClosure).2 <| by rwa [upperBounds_supClosure]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- A meet-semilattice where every inf-closed set has a greatest lower bound is automatically
complete. -/
def SemilatticeInf.toCompleteSemilatticeInf [SemilatticeInf α] (sInf : Set α → α)
    (h : ∀ s, InfClosed s → IsGLB s (sInf s)) : CompleteSemilatticeInf α where
  sInf := fun s => sInf (infClosure s)
  sInf_le _ _ ha := (h _ infClosed_infClosure).1 <| subset_infClosure ha
                                                                       /-
                                                                         ι : Sort u_1
                                                                         F : Type u_2
                                                                         α : Type u_3
                                                                         β : Type u_4
                                                                         inst✝ : SemilatticeInf α
                                                                         sInf : Set α → α
                                                                         h : ∀ (s : Set α), InfClosed s → IsGLB s (sInf s)
                                                                         s : Set α
                                                                         a : α
                                                                         ha : ∀ (b : α), Membership.mem s b → LE.le a b
                                                                         ⊢ Membership.mem (lowerBounds (infClosure s)) a
                                                                       -/
  le_sInf s a ha := (le_isGLB_iff <| h _ infClosed_infClosure).2 <| by rwa [lowerBounds_infClosure]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/



lemma SupClosed.iSup_mem_of_nonempty [Finite ι] [Nonempty ι] (hs : SupClosed s)
    (hf : ∀ i, f i ∈ s) : ⨆ i, f i ∈ s := by
  /-
    ι : Sort u_1
    α : Type u_3
    inst✝² : ConditionallyCompleteLattice α
    f : ι → α
    s : Set α
    inst✝¹ : Finite ι
    inst✝ : Nonempty ι
    hs : SupClosed s
    hf : ∀ (i : ι), Membership.mem s (f i)
    ⊢ Membership.mem s (iSup fun i => f i)
  -/
  cases nonempty_fintype (PLift ι)
  /-
    case intro
    ι : Sort u_1
    α : Type u_3
    inst✝² : ConditionallyCompleteLattice α
    f : ι → α
    s : Set α
    inst✝¹ : Finite ι
    inst✝ : Nonempty ι
    hs : SupClosed s
    hf : ∀ (i : ι), Membership.mem s (f i)
    val✝ : Fintype (PLift ι)
    ⊢ Membership.mem s (iSup fun i => f i)
  -/
  rw [← iSup_plift_down, ← Finset.sup'_univ_eq_ciSup]
  /-
    case intro
    ι : Sort u_1
    α : Type u_3
    inst✝² : ConditionallyCompleteLattice α
    f : ι → α
    s : Set α
    inst✝¹ : Finite ι
    inst✝ : Nonempty ι
    hs : SupClosed s
    hf : ∀ (i : ι), Membership.mem s (f i)
    val✝ : Fintype (PLift ι)
    ⊢ Membership.mem s (Finset.univ.sup' ⋯ fun i => f i.down)
  -/
  exact hs.finsetSup'_mem Finset.univ_nonempty fun _ _ ↦ hf _
  /-
    🎉 no goals
  -/


lemma InfClosed.iInf_mem_of_nonempty [Finite ι] [Nonempty ι] (hs : InfClosed s)
    (hf : ∀ i, f i ∈ s) : ⨅ i, f i ∈ s := hs.dual.iSup_mem_of_nonempty hf


lemma SupClosed.sSup_mem_of_nonempty (hs : SupClosed s) (ht : t.Finite) (ht' : t.Nonempty)
    (hts : t ⊆ s) : sSup t ∈ s := by
  /-
    α : Type u_3
    inst✝ : ConditionallyCompleteLattice α
    s t : Set α
    hs : SupClosed s
    ht : t.Finite
    ht' : t.Nonempty
    hts : HasSubset.Subset t s
    ⊢ Membership.mem s (SupSet.sSup t)
  -/
  have := ht.to_subtype
  /-
    α : Type u_3
    inst✝ : ConditionallyCompleteLattice α
    s t : Set α
    hs : SupClosed s
    ht : t.Finite
    ht' : t.Nonempty
    hts : HasSubset.Subset t s
    this : Finite ↑t
    ⊢ Membership.mem s (SupSet.sSup t)
  -/
  have := ht'.to_subtype
  /-
    α : Type u_3
    inst✝ : ConditionallyCompleteLattice α
    s t : Set α
    hs : SupClosed s
    ht : t.Finite
    ht' : t.Nonempty
    hts : HasSubset.Subset t s
    this✝ : Finite ↑t
    this : Nonempty ↑t
    ⊢ Membership.mem s (SupSet.sSup t)
  -/
  rw [sSup_eq_iSup']
  /-
    α : Type u_3
    inst✝ : ConditionallyCompleteLattice α
    s t : Set α
    hs : SupClosed s
    ht : t.Finite
    ht' : t.Nonempty
    hts : HasSubset.Subset t s
    this✝ : Finite ↑t
    this : Nonempty ↑t
    ⊢ Membership.mem s (iSup fun a => ↑a)
  -/
  exact hs.iSup_mem_of_nonempty (by simpa)
  /-
    🎉 no goals
  -/


lemma InfClosed.sInf_mem_of_nonempty (hs : InfClosed s) (ht : t.Finite) (ht' : t.Nonempty)
    (hts : t ⊆ s) : sInf t ∈ s := hs.dual.sSup_mem_of_nonempty ht ht' hts


lemma SupClosed.biSup_mem_of_nonempty {ι : Type*} {t : Set ι} {f : ι → α} (hs : SupClosed s)
    (ht : t.Finite) (ht' : t.Nonempty) (hf : ∀ i ∈ t, f i ∈ s) : ⨆ i ∈ t, f i ∈ s := by
  /-
    α : Type u_3
    inst✝ : CompleteLattice α
    s : Set α
    ι : Type u_5
    t : Set ι
    f : ι → α
    hs : SupClosed s
    ht : t.Finite
    ht' : t.Nonempty
    hf : ∀ (i : ι), Membership.mem t i → Membership.mem s (f i)
    ⊢ Membership.mem s (iSup fun i => iSup fun h => f i)
  -/
  rw [← sSup_image]
  /-
    α : Type u_3
    inst✝ : CompleteLattice α
    s : Set α
    ι : Type u_5
    t : Set ι
    f : ι → α
    hs : SupClosed s
    ht : t.Finite
    ht' : t.Nonempty
    hf : ∀ (i : ι), Membership.mem t i → Membership.mem s (f i)
    ⊢ Membership.mem s (SupSet.sSup (Set.image f t))
  -/
  exact hs.sSup_mem_of_nonempty (ht.image _) (by simpa) (by simpa)
  /-
    🎉 no goals
  -/


lemma InfClosed.biInf_mem_of_nonempty {ι : Type*} {t : Set ι} {f : ι → α} (hs : InfClosed s)
    (ht : t.Finite) (ht' : t.Nonempty) (hf : ∀ i ∈ t, f i ∈ s) : ⨅ i ∈ t, f i ∈ s :=
  hs.dual.biSup_mem_of_nonempty ht ht' hf


lemma SupClosed.iSup_mem [Finite ι] (hs : SupClosed s) (hbot : ⊥ ∈ s) (hf : ∀ i, f i ∈ s) :
    ⨆ i, f i ∈ s := by
  /-
    ι : Sort u_1
    α : Type u_3
    inst✝¹ : CompleteLattice α
    f : ι → α
    s : Set α
    inst✝ : Finite ι
    hs : SupClosed s
    hbot : Membership.mem s Bot.bot
    hf : ∀ (i : ι), Membership.mem s (f i)
    ⊢ Membership.mem s (iSup fun i => f i)
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      ι : Sort u_1
      α : Type u_3
      inst✝¹ : CompleteLattice α
      f : ι → α
      s : Set α
      inst✝ : Finite ι
      hs : SupClosed s
      hbot : Membership.mem s Bot.bot
      hf : ∀ (i : ι), Membership.mem s (f i)
      h✝ : IsEmpty ι
      ⊢ Membership.mem s (iSup fun i => f i)
    -/
  · simpa [iSup_of_empty]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Sort u_1
      α : Type u_3
      inst✝¹ : CompleteLattice α
      f : ι → α
      s : Set α
      inst✝ : Finite ι
      hs : SupClosed s
      hbot : Membership.mem s Bot.bot
      hf : ∀ (i : ι), Membership.mem s (f i)
      h✝ : Nonempty ι
      ⊢ Membership.mem s (iSup fun i => f i)
    -/
  · exact hs.iSup_mem_of_nonempty hf
    /-
      🎉 no goals
    -/


lemma InfClosed.iInf_mem [Finite ι] (hs : InfClosed s) (htop : ⊤ ∈ s) (hf : ∀ i, f i ∈ s) :
    ⨅ i, f i ∈ s := hs.dual.iSup_mem htop hf


lemma SupClosed.sSup_mem (hs : SupClosed s) (ht : t.Finite) (hbot : ⊥ ∈ s) (hts : t ⊆ s) :
    sSup t ∈ s := by
  /-
    α : Type u_3
    inst✝ : CompleteLattice α
    s t : Set α
    hs : SupClosed s
    ht : t.Finite
    hbot : Membership.mem s Bot.bot
    hts : HasSubset.Subset t s
    ⊢ Membership.mem s (SupSet.sSup t)
  -/
  have := ht.to_subtype
  /-
    α : Type u_3
    inst✝ : CompleteLattice α
    s t : Set α
    hs : SupClosed s
    ht : t.Finite
    hbot : Membership.mem s Bot.bot
    hts : HasSubset.Subset t s
    this : Finite ↑t
    ⊢ Membership.mem s (SupSet.sSup t)
  -/
  rw [sSup_eq_iSup']
  /-
    α : Type u_3
    inst✝ : CompleteLattice α
    s t : Set α
    hs : SupClosed s
    ht : t.Finite
    hbot : Membership.mem s Bot.bot
    hts : HasSubset.Subset t s
    this : Finite ↑t
    ⊢ Membership.mem s (iSup fun a => ↑a)
  -/
  exact hs.iSup_mem hbot (by simpa)
  /-
    🎉 no goals
  -/


lemma InfClosed.sInf_mem (hs : InfClosed s) (ht : t.Finite) (htop : ⊤ ∈ s) (hts : t ⊆ s) :
    sInf t ∈ s := hs.dual.sSup_mem ht htop hts


lemma SupClosed.biSup_mem {ι : Type*} {t : Set ι} {f : ι → α} (hs : SupClosed s)
    (ht : t.Finite) (hbot : ⊥ ∈ s) (hf : ∀ i ∈ t, f i ∈ s) : ⨆ i ∈ t, f i ∈ s := by
  /-
    α : Type u_3
    inst✝ : CompleteLattice α
    s : Set α
    ι : Type u_5
    t : Set ι
    f : ι → α
    hs : SupClosed s
    ht : t.Finite
    hbot : Membership.mem s Bot.bot
    hf : ∀ (i : ι), Membership.mem t i → Membership.mem s (f i)
    ⊢ Membership.mem s (iSup fun i => iSup fun h => f i)
  -/
  rw [← sSup_image]
  /-
    α : Type u_3
    inst✝ : CompleteLattice α
    s : Set α
    ι : Type u_5
    t : Set ι
    f : ι → α
    hs : SupClosed s
    ht : t.Finite
    hbot : Membership.mem s Bot.bot
    hf : ∀ (i : ι), Membership.mem t i → Membership.mem s (f i)
    ⊢ Membership.mem s (SupSet.sSup (Set.image f t))
  -/
  exact hs.sSup_mem (ht.image _) hbot (by simpa)
  /-
    🎉 no goals
  -/


lemma InfClosed.biInf_mem {ι : Type*} {t : Set ι} {f : ι → α} (hs : InfClosed s)
    (ht : t.Finite) (htop : ⊤ ∈ s) (hf : ∀ i ∈ t, f i ∈ s) : ⨅ i ∈ t, f i ∈ s :=
  hs.dual.biSup_mem ht htop hf

