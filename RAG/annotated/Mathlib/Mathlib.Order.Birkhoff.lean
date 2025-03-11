@[simp] lemma infIrred_Ici (a : α) : InfIrred (Ici a) := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a : α
    ⊢ InfIrred (UpperSet.Ici a)
  -/
  refine ⟨fun h ↦ Ici_ne_top h.eq_top, fun s t hst ↦ ?_⟩
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a : α
    s t : UpperSet α
    hst : Eq (Min.min s t) (UpperSet.Ici a)
    ⊢ Or (Eq s (UpperSet.Ici a)) (Eq t (UpperSet.Ici a))
  -/
  have := mem_Ici_iff.2 (le_refl a)
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a : α
    s t : UpperSet α
    hst : Eq (Min.min s t) (UpperSet.Ici a)
    this : Membership.mem (UpperSet.Ici a) a
    ⊢ Or (Eq s (UpperSet.Ici a)) (Eq t (UpperSet.Ici a))
  -/
  rw [← hst] at this
  exact this.imp (fun ha ↦ le_antisymm (le_Ici.2 ha) <| hst.ge.trans inf_le_left) fun ha ↦
      le_antisymm (le_Ici.2 ha) <| hst.ge.trans inf_le_right


@[simp] lemma infIrred_iff_of_finite : InfIrred s ↔ ∃ a, Ici a = s := by
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    s : UpperSet α
    inst✝ : Finite α
    ⊢ Iff (InfIrred s) (Exists fun a => Eq (UpperSet.Ici a) s)
  -/
  refine ⟨fun hs ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : PartialOrder α
      s : UpperSet α
      inst✝ : Finite α
      hs : InfIrred s
      ⊢ Exists fun a => Eq (UpperSet.Ici a) s
    -/
  · obtain ⟨a, ha, has⟩ := (s : Set α).toFinite.exists_minimal_wrt id _ (coe_nonempty.2 hs.ne_top)
    exact ⟨a, (hs.2 <| erase_inf_Ici ha <| by simpa [eq_comm] using has).resolve_left
      (lt_erase.2 ha).ne'⟩
    /-
      case refine_2
      α : Type u_1
      inst✝¹ : PartialOrder α
      s : UpperSet α
      inst✝ : Finite α
      ⊢ (Exists fun a => Eq (UpperSet.Ici a) s) → InfIrred s
    -/
  · rintro ⟨a, rfl⟩
    /-
      case refine_2.intro
      α : Type u_1
      inst✝¹ : PartialOrder α
      inst✝ : Finite α
      a : α
      ⊢ InfIrred (UpperSet.Ici a)
    -/
    exact infIrred_Ici _
    /-
      🎉 no goals
    -/


@[simp] lemma supIrred_Iic (a : α) : SupIrred (Iic a) := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a : α
    ⊢ SupIrred (LowerSet.Iic a)
  -/
  refine ⟨fun h ↦ Iic_ne_bot h.eq_bot, fun s t hst ↦ ?_⟩
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a : α
    s t : LowerSet α
    hst : Eq (Max.max s t) (LowerSet.Iic a)
    ⊢ Or (Eq s (LowerSet.Iic a)) (Eq t (LowerSet.Iic a))
  -/
  have := mem_Iic_iff.2 (le_refl a)
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a : α
    s t : LowerSet α
    hst : Eq (Max.max s t) (LowerSet.Iic a)
    this : Membership.mem (LowerSet.Iic a) a
    ⊢ Or (Eq s (LowerSet.Iic a)) (Eq t (LowerSet.Iic a))
  -/
  rw [← hst] at this
  exact this.imp (fun ha ↦ (le_sup_left.trans_eq hst).antisymm <| Iic_le.2 ha) fun ha ↦
    (le_sup_right.trans_eq hst).antisymm <| Iic_le.2 ha


@[simp] lemma supIrred_iff_of_finite : SupIrred s ↔ ∃ a, Iic a = s := by
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    s : LowerSet α
    inst✝ : Finite α
    ⊢ Iff (SupIrred s) (Exists fun a => Eq (LowerSet.Iic a) s)
  -/
  refine ⟨fun hs ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : PartialOrder α
      s : LowerSet α
      inst✝ : Finite α
      hs : SupIrred s
      ⊢ Exists fun a => Eq (LowerSet.Iic a) s
    -/
  · obtain ⟨a, ha, has⟩ := (s : Set α).toFinite.exists_maximal_wrt id _ (coe_nonempty.2 hs.ne_bot)
    exact ⟨a, (hs.2 <| erase_sup_Iic ha <| by simpa [eq_comm] using has).resolve_left
      (erase_lt.2 ha).ne⟩
    /-
      case refine_2
      α : Type u_1
      inst✝¹ : PartialOrder α
      s : LowerSet α
      inst✝ : Finite α
      ⊢ (Exists fun a => Eq (LowerSet.Iic a) s) → SupIrred s
    -/
  · rintro ⟨a, rfl⟩
    /-
      case refine_2.intro
      α : Type u_1
      inst✝¹ : PartialOrder α
      inst✝ : Finite α
      a : α
      ⊢ SupIrred (LowerSet.Iic a)
    -/
    exact supIrred_Iic _
    /-
      🎉 no goals
    -/


/-- The **Birkhoff Embedding** of a finite partial order as sup-irreducible elements in its
lattice of lower sets. -/
def supIrredLowerSet : α ↪o {s : LowerSet α // SupIrred s} where
  toFun a := ⟨Iic a, supIrred_Iic _⟩
               /-
                 α : Type u_1
                 inst✝ : PartialOrder α
                 x✝ : α
                 ⊢ ∀ ⦃a₂ : α⦄, Eq ((fun a => ⟨LowerSet.Iic a, ⋯⟩) x✝) ((fun a => ⟨LowerSet.Iic  …
               -/
  inj' _ := by simp
               /-
                 🎉 no goals
               -/
                     /-
                       α : Type u_1
                       inst✝ : PartialOrder α
                       ⊢ ∀ {a b : α}, Iff (LE.le ({ toFun := fun a => ⟨LowerSet.Iic a, ⋯⟩, inj' := ⋯  …
                     -/
  map_rel_iff' := by simp
                     /-
                       🎉 no goals
                     -/


/-- The **Birkhoff Embedding** of a finite partial order as inf-irreducible elements in its
lattice of lower sets. -/
def infIrredUpperSet : α ↪o {s : UpperSet α // InfIrred s} where
  toFun a := ⟨Ici a, infIrred_Ici _⟩
               /-
                 α : Type u_1
                 inst✝ : PartialOrder α
                 x✝ : α
                 ⊢ ∀ ⦃a₂ : α⦄, Eq ((fun a => ⟨UpperSet.Ici a, ⋯⟩) x✝) ((fun a => ⟨UpperSet.Ici  …
               -/
  inj' _ := by simp
               /-
                 🎉 no goals
               -/
                     /-
                       α : Type u_1
                       inst✝ : PartialOrder α
                       ⊢ ∀ {a b : α}, Iff (LE.le ({ toFun := fun a => ⟨UpperSet.Ici a, ⋯⟩, inj' := ⋯  …
                     -/
  map_rel_iff' := by simp
                     /-
                       🎉 no goals
                     -/


@[simp] lemma supIrredLowerSet_apply (a : α) : supIrredLowerSet a = ⟨Iic a, supIrred_Iic _⟩ := rfl

@[simp] lemma infIrredUpperSet_apply (a : α) : infIrredUpperSet a = ⟨Ici a, infIrred_Ici _⟩ := rfl


lemma supIrredLowerSet_surjective : Surjective (supIrredLowerSet (α := α)) := by
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : Finite α
    ⊢ Function.Surjective ⇑OrderEmbedding.supIrredLowerSet
  -/
  aesop (add simp Surjective)
  /-
    🎉 no goals
  -/


lemma infIrredUpperSet_surjective : Surjective (infIrredUpperSet (α := α)) := by
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : Finite α
    ⊢ Function.Surjective ⇑OrderEmbedding.infIrredUpperSet
  -/
  aesop (add simp Surjective)
  /-
    🎉 no goals
  -/


/-- **Birkhoff Representation for partial orders.** Any partial order is isomorphic
to the partial order of sup-irreducible elements in its lattice of lower sets. -/
noncomputable def supIrredLowerSet : α ≃o {s : LowerSet α // SupIrred s} :=
  RelIso.ofSurjective _ OrderEmbedding.supIrredLowerSet_surjective


/-- **Birkhoff Representation for partial orders.** Any partial order is isomorphic
to the partial order of inf-irreducible elements in its lattice of upper sets. -/
noncomputable def infIrredUpperSet : α ≃o {s : UpperSet α // InfIrred s} :=
  RelIso.ofSurjective _ OrderEmbedding.infIrredUpperSet_surjective


@[simp] lemma supIrredLowerSet_symm_apply (s : {s : LowerSet α // SupIrred s}) [Fintype s] :
    supIrredLowerSet.symm s = (s.1 : Set α).toFinset.sup id := by
  classical
  obtain ⟨s, hs⟩ := s
  obtain ⟨a, rfl⟩ := supIrred_iff_of_finite.1 hs
  cases nonempty_fintype α
  have : LocallyFiniteOrder α := Fintype.toLocallyFiniteOrder
  simp [symm_apply_eq]


@[simp] lemma infIrredUpperSet_symm_apply (s : {s : UpperSet α // InfIrred s}) [Fintype s] :
    infIrredUpperSet.symm s = (s.1 : Set α).toFinset.inf id := by
  classical
  obtain ⟨s, hs⟩ := s
  obtain ⟨a, rfl⟩ := infIrred_iff_of_finite.1 hs
  cases nonempty_fintype α
  have : LocallyFiniteOrder α := Fintype.toLocallyFiniteOrder
  simp [symm_apply_eq]


open Classical in
/-- **Birkhoff Representation for finite distributive lattices**. Any nonempty finite distributive
lattice is isomorphic to the lattice of lower sets of its sup-irreducible elements. -/
noncomputable def OrderIso.lowerSetSupIrred [OrderBot α] : α ≃o LowerSet {a : α // SupIrred a} :=
  Equiv.toOrderIso
    { toFun := fun a ↦ ⟨{b | ↑b ≤ a}, fun _ _ hcb hba ↦ hba.trans' hcb⟩
      invFun := fun s ↦ (s : Set {a : α // SupIrred a}).toFinset.sup (↑)
      left_inv := fun a ↦ by
        /-
          α : Type u_1
          inst✝³ : DistribLattice α
          inst✝² : Fintype α
          inst✝¹ : DecidablePred SupIrred
          inst✝ : OrderBot α
          a : α
          ⊢ Eq ((fun s => (↑s).toFinset.sup Subtype.val) ((fun a => { carrier := setOf f …
        -/
        refine le_antisymm (Finset.sup_le fun b ↦ Set.mem_toFinset.1) ?_
        /-
          α : Type u_1
          inst✝³ : DistribLattice α
          inst✝² : Fintype α
          inst✝¹ : DecidablePred SupIrred
          inst✝ : OrderBot α
          a : α
          ⊢ LE.le a ((fun s => (↑s).toFinset.sup Subtype.val) ((fun a => { carrier := se …
        -/
        obtain ⟨s, rfl, hs⟩ := exists_supIrred_decomposition a
        exact Finset.sup_le fun i hi ↦
          le_sup_of_le (b := ⟨i, hs hi⟩) (Set.mem_toFinset.2 <| le_sup (f := id) hi) le_rfl
      right_inv := fun s ↦ by
        /-
          α : Type u_1
          inst✝³ : DistribLattice α
          inst✝² : Fintype α
          inst✝¹ : DecidablePred SupIrred
          inst✝ : OrderBot α
          s : LowerSet (Subtype fun a => SupIrred a)
          ⊢ Eq ((fun a => { carrier := setOf fun b => LE.le (↑b) a, lower' := ⋯ }) ((fun …
        -/
        ext a
        /-
          case a.h
          α : Type u_1
          inst✝³ : DistribLattice α
          inst✝² : Fintype α
          inst✝¹ : DecidablePred SupIrred
          inst✝ : OrderBot α
          s : LowerSet (Subtype fun a => SupIrred a)
          a : Subtype fun a => SupIrred a
          ⊢ Iff (Membership.mem (↑((fun a => { carrier := setOf fun b => LE.le (↑b) a, l …
        -/
        dsimp
        /-
          case a.h
          α : Type u_1
          inst✝³ : DistribLattice α
          inst✝² : Fintype α
          inst✝¹ : DecidablePred SupIrred
          inst✝ : OrderBot α
          s : LowerSet (Subtype fun a => SupIrred a)
          a : Subtype fun a => SupIrred a
          ⊢ Iff (LE.le (↑a) ((↑s).toFinset.sup Subtype.val)) (Membership.mem (↑s) a)
        -/
        refine ⟨fun ha ↦ ?_, fun ha ↦ ?_⟩
          /-
            case a.h.refine_1
            α : Type u_1
            inst✝³ : DistribLattice α
            inst✝² : Fintype α
            inst✝¹ : DecidablePred SupIrred
            inst✝ : OrderBot α
            s : LowerSet (Subtype fun a => SupIrred a)
            a : Subtype fun a => SupIrred a
            ha : LE.le (↑a) ((↑s).toFinset.sup Subtype.val)
            ⊢ Membership.mem (↑s) a
          -/
        · obtain ⟨i, hi, ha⟩ := a.2.supPrime.le_finset_sup.1 ha
          /-
            case a.h.refine_1.intro.intro
            α : Type u_1
            inst✝³ : DistribLattice α
            inst✝² : Fintype α
            inst✝¹ : DecidablePred SupIrred
            inst✝ : OrderBot α
            s : LowerSet (Subtype fun a => SupIrred a)
            a : Subtype fun a => SupIrred a
            ha✝ : LE.le (↑a) ((↑s).toFinset.sup Subtype.val)
            i : Subtype fun a => SupIrred a
            hi : Membership.mem (↑s).toFinset i
            ha : LE.le ↑a ↑i
            ⊢ Membership.mem (↑s) a
          -/
          exact s.lower ha (Set.mem_toFinset.1 hi)
          /-
            🎉 no goals
          -/
          /-
            case a.h.refine_2
            α : Type u_1
            inst✝³ : DistribLattice α
            inst✝² : Fintype α
            inst✝¹ : DecidablePred SupIrred
            inst✝ : OrderBot α
            s : LowerSet (Subtype fun a => SupIrred a)
            a : Subtype fun a => SupIrred a
            ha : Membership.mem (↑s) a
            ⊢ LE.le (↑a) ((↑s).toFinset.sup Subtype.val)
          -/
        · dsimp
          /-
            case a.h.refine_2
            α : Type u_1
            inst✝³ : DistribLattice α
            inst✝² : Fintype α
            inst✝¹ : DecidablePred SupIrred
            inst✝ : OrderBot α
            s : LowerSet (Subtype fun a => SupIrred a)
            a : Subtype fun a => SupIrred a
            ha : Membership.mem (↑s) a
            ⊢ LE.le (↑a) ((↑s).toFinset.sup Subtype.val)
          -/
          exact le_sup (Set.mem_toFinset.2 ha) }
          /-
            🎉 no goals
          -/
    (fun _ _ hbc _ ↦ le_trans' hbc) fun _ _ hst ↦ Finset.sup_mono <| Set.toFinset_mono hst


/-- **Birkhoff's Representation Theorem**. Any finite distributive lattice can be embedded in a
powerset lattice. -/
noncomputable def birkhoffSet : α ↪o Set {a : α // SupIrred a} := by
  /-
    α : Type u_1
    inst✝² : DistribLattice α
    inst✝¹ : Fintype α
    inst✝ : DecidablePred SupIrred
    ⊢ OrderEmbedding α (Set (Subtype fun a => SupIrred a))
  -/
  by_cases h : IsEmpty α
    /-
      case pos
      α : Type u_1
      inst✝² : DistribLattice α
      inst✝¹ : Fintype α
      inst✝ : DecidablePred SupIrred
      h : IsEmpty α
      ⊢ OrderEmbedding α (Set (Subtype fun a => SupIrred a))
    -/
  · exact OrderEmbedding.ofIsEmpty
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝² : DistribLattice α
    inst✝¹ : Fintype α
    inst✝ : DecidablePred SupIrred
    h : Not (IsEmpty α)
    ⊢ OrderEmbedding α (Set (Subtype fun a => SupIrred a))
  -/
  rw [not_isEmpty_iff] at h
  /-
    case neg
    α : Type u_1
    inst✝² : DistribLattice α
    inst✝¹ : Fintype α
    inst✝ : DecidablePred SupIrred
    h : Nonempty α
    ⊢ OrderEmbedding α (Set (Subtype fun a => SupIrred a))
  -/
  have := Fintype.toOrderBot α
  /-
    case neg
    α : Type u_1
    inst✝² : DistribLattice α
    inst✝¹ : Fintype α
    inst✝ : DecidablePred SupIrred
    h : Nonempty α
    this : OrderBot α
    ⊢ OrderEmbedding α (Set (Subtype fun a => SupIrred a))
  -/
  exact OrderIso.lowerSetSupIrred.toOrderEmbedding.trans ⟨⟨_, SetLike.coe_injective⟩, Iff.rfl⟩
  /-
    🎉 no goals
  -/


/-- **Birkhoff's Representation Theorem**. Any finite distributive lattice can be embedded in a
powerset lattice. -/
noncomputable def birkhoffFinset : α ↪o Finset {a : α // SupIrred a} := by
  /-
    α : Type u_1
    inst✝² : DistribLattice α
    inst✝¹ : Fintype α
    inst✝ : DecidablePred SupIrred
    ⊢ OrderEmbedding α (Finset (Subtype fun a => SupIrred a))
  -/
  exact birkhoffSet.trans Fintype.finsetOrderIsoSet.symm.toOrderEmbedding
  /-
    🎉 no goals
  -/


@[simp] lemma coe_birkhoffFinset (a : α) : birkhoffFinset a = birkhoffSet a := by
  classical
  -- TODO: This should be a single `simp` call but `simp` refuses to use
  -- `OrderIso.coe_toOrderEmbedding` and `Fintype.coe_finsetOrderIsoSet_symm`
  simp [birkhoffFinset]
  rw [OrderIso.coe_toOrderEmbedding, Fintype.coe_finsetOrderIsoSet_symm]
  simp


@[simp] lemma birkhoffSet_sup (a b : α) : birkhoffSet (a ⊔ b) = birkhoffSet a ∪ birkhoffSet b := by
  /-
    α : Type u_1
    inst✝² : DistribLattice α
    inst✝¹ : Fintype α
    inst✝ : DecidablePred SupIrred
    a b : α
    ⊢ Eq (OrderEmbedding.birkhoffSet (Max.max a b)) (Union.union (OrderEmbedding.b …
  -/
                                               /-
                                                 🎉 no goals
                                               -/
  unfold OrderEmbedding.birkhoffSet; split <;> simp [eq_iff_true_of_subsingleton]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp] lemma birkhoffSet_inf (a b : α) : birkhoffSet (a ⊓ b) = birkhoffSet a ∩ birkhoffSet b := by
  /-
    α : Type u_1
    inst✝² : DistribLattice α
    inst✝¹ : Fintype α
    inst✝ : DecidablePred SupIrred
    a b : α
    ⊢ Eq (OrderEmbedding.birkhoffSet (Min.min a b)) (Inter.inter (OrderEmbedding.b …
  -/
                                               /-
                                                 🎉 no goals
                                               -/
  unfold OrderEmbedding.birkhoffSet; split <;> simp [eq_iff_true_of_subsingleton]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp] lemma birkhoffSet_apply [OrderBot α] (a : α) :
    birkhoffSet a = OrderIso.lowerSetSupIrred a := by
  /-
    α : Type u_1
    inst✝³ : DistribLattice α
    inst✝² : Fintype α
    inst✝¹ : DecidablePred SupIrred
    inst✝ : OrderBot α
    a : α
    ⊢ Eq (OrderEmbedding.birkhoffSet a) ↑(OrderIso.lowerSetSupIrred a)
  -/
  simp [birkhoffSet]; have : Subsingleton (OrderBot α) := inferInstance; convert rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp] lemma birkhoffFinset_sup (a b : α) :
    birkhoffFinset (a ⊔ b) = birkhoffFinset a ∪ birkhoffFinset b := by
  classical
  dsimp [OrderEmbedding.birkhoffFinset]
  rw [birkhoffSet_sup, OrderIso.coe_toOrderEmbedding]
  simp


@[simp] lemma birkhoffFinset_inf (a b : α) :
    birkhoffFinset (a ⊓ b) = birkhoffFinset a ∩ birkhoffFinset b := by
  classical
  dsimp [OrderEmbedding.birkhoffFinset]
  rw [birkhoffSet_inf, OrderIso.coe_toOrderEmbedding]
  simp


/-- **Birkhoff's Representation Theorem**. Any finite distributive lattice can be embedded in a
powerset lattice. -/
noncomputable def birkhoffSet : LatticeHom α (Set {a : α // SupIrred a}) where
  toFun := OrderEmbedding.birkhoffSet
  map_sup' := OrderEmbedding.birkhoffSet_sup
  map_inf' := OrderEmbedding.birkhoffSet_inf


open Classical in
/-- **Birkhoff's Representation Theorem**. Any finite distributive lattice can be embedded in a
powerset lattice. -/
noncomputable def birkhoffFinset : LatticeHom α (Finset {a : α // SupIrred a}) where
  toFun := OrderEmbedding.birkhoffFinset
  map_sup' := OrderEmbedding.birkhoffFinset_sup
  map_inf' := OrderEmbedding.birkhoffFinset_inf


lemma birkhoffFinset_injective : Injective (birkhoffFinset (α := α)) :=
  OrderEmbedding.birkhoffFinset.injective


lemma exists_birkhoff_representation.{u} (α : Type u) [Finite α] [DistribLattice α] :
    ∃ (β : Type u) (_ : DecidableEq β) (_ : Fintype β) (f : LatticeHom α (Finset β)),
      Injective f := by
  classical
  cases nonempty_fintype α
  exact ⟨{a : α // SupIrred a}, _, inferInstance, _, LatticeHom.birkhoffFinset_injective⟩


