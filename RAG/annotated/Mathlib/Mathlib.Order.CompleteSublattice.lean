/-- A complete sublattice is a subset of a complete lattice that is closed under arbitrary suprema
and infima. -/
structure CompleteSublattice extends Sublattice α where
  sSupClosed' : ∀ ⦃s : Set α⦄, s ⊆ carrier → sSup s ∈ carrier
  sInfClosed' : ∀ ⦃s : Set α⦄, s ⊆ carrier → sInf s ∈ carrier


/-- To check that a subset is a complete sublattice, one does not need to check that it is closed
under binary `Sup` since this follows from the stronger `sSup` condition. Likewise for infima. -/
@[simps] def mk' (carrier : Set α)
    (sSupClosed' : ∀ ⦃s : Set α⦄, s ⊆ carrier → sSup s ∈ carrier)
    (sInfClosed' : ∀ ⦃s : Set α⦄, s ⊆ carrier → sInf s ∈ carrier) :
  CompleteSublattice α where
    carrier := carrier
    sSupClosed' := sSupClosed'
    sInfClosed' := sInfClosed'
    supClosed' := fun x hx y hy ↦ by
      /-
        α : Type u_1
        β : Type u_2
        inst✝¹ : CompleteLattice α
        inst✝ : CompleteLattice β
        f : CompleteLatticeHom α β
        carrier : Set α
        sSupClosed' : ∀ ⦃s : Set α⦄, HasSubset.Subset s carrier → Membership.mem carri …
        sInfClosed' : ∀ ⦃s : Set α⦄, HasSubset.Subset s carrier → Membership.mem carri …
        x : α
        hx : Membership.mem carrier x
        y : α
        hy : Membership.mem carrier y
        ⊢ Membership.mem carrier (Max.max x y)
      -/
      suffices x ⊔ y = sSup {x, y} by exact this ▸ sSupClosed' (fun z hz ↦ by aesop)
      /-
        α : Type u_1
        β : Type u_2
        inst✝¹ : CompleteLattice α
        inst✝ : CompleteLattice β
        f : CompleteLatticeHom α β
        carrier : Set α
        sSupClosed' : ∀ ⦃s : Set α⦄, HasSubset.Subset s carrier → Membership.mem carri …
        sInfClosed' : ∀ ⦃s : Set α⦄, HasSubset.Subset s carrier → Membership.mem carri …
        x : α
        hx : Membership.mem carrier x
        y : α
        hy : Membership.mem carrier y
        ⊢ Eq (Max.max x y) (SupSet.sSup (Insert.insert x (Singleton.singleton y)))
      -/
      simp [sSup_singleton]
      /-
        🎉 no goals
      -/
    infClosed' := fun x hx y hy ↦ by
      /-
        α : Type u_1
        β : Type u_2
        inst✝¹ : CompleteLattice α
        inst✝ : CompleteLattice β
        f : CompleteLatticeHom α β
        carrier : Set α
        sSupClosed' : ∀ ⦃s : Set α⦄, HasSubset.Subset s carrier → Membership.mem carri …
        sInfClosed' : ∀ ⦃s : Set α⦄, HasSubset.Subset s carrier → Membership.mem carri …
        x : α
        hx : Membership.mem carrier x
        y : α
        hy : Membership.mem carrier y
        ⊢ Membership.mem carrier (Min.min x y)
      -/
      suffices x ⊓ y = sInf {x, y} by exact this ▸ sInfClosed' (fun z hz ↦ by aesop)
      /-
        α : Type u_1
        β : Type u_2
        inst✝¹ : CompleteLattice α
        inst✝ : CompleteLattice β
        f : CompleteLatticeHom α β
        carrier : Set α
        sSupClosed' : ∀ ⦃s : Set α⦄, HasSubset.Subset s carrier → Membership.mem carri …
        sInfClosed' : ∀ ⦃s : Set α⦄, HasSubset.Subset s carrier → Membership.mem carri …
        x : α
        hx : Membership.mem carrier x
        y : α
        hy : Membership.mem carrier y
        ⊢ Eq (Min.min x y) (InfSet.sInf (Insert.insert x (Singleton.singleton y)))
      -/
      simp [sInf_singleton]
      /-
        🎉 no goals
      -/


instance instSetLike : SetLike (CompleteSublattice α) α where
  coe L := L.carrier
                             /-
                               α : Type u_1
                               β : Type u_2
                               inst✝¹ : CompleteLattice α
                               inst✝ : CompleteLattice β
                               f : CompleteLatticeHom α β
                               L✝ L M : CompleteSublattice α
                               h : Eq ((fun L => L.carrier) L) ((fun L => L.carrier) M)
                               ⊢ Eq L M
                             -/
  coe_injective' L M h := by cases L; cases M; congr; exact SetLike.coe_injective' h
                                                      /-
                                                        🎉 no goals
                                                      -/


instance instBot : Bot L where
                /-
                  α : Type u_1
                  β : Type u_2
                  inst✝¹ : CompleteLattice α
                  inst✝ : CompleteLattice β
                  f : CompleteLatticeHom α β
                  L : CompleteSublattice α
                  ⊢ Membership.mem L Bot.bot
                -/
  bot := ⟨⊥, by simpa using L.sSupClosed' <| empty_subset _⟩
                /-
                  🎉 no goals
                -/


instance instTop : Top L where
                /-
                  α : Type u_1
                  β : Type u_2
                  inst✝¹ : CompleteLattice α
                  inst✝ : CompleteLattice β
                  f : CompleteLatticeHom α β
                  L : CompleteSublattice α
                  ⊢ Membership.mem L Top.top
                -/
  top := ⟨⊤, by simpa using L.sInfClosed' <| empty_subset _⟩
                /-
                  🎉 no goals
                -/


instance instSupSet : SupSet L where
  sSup s := ⟨sSup <| (↑) '' s, L.sSupClosed' image_val_subset⟩


instance instInfSet : InfSet L where
  sInf s := ⟨sInf <| (↑) '' s, L.sInfClosed' image_val_subset⟩


theorem sSupClosed {s : Set α} (h : s ⊆ L) : sSup s ∈ L := L.sSupClosed' h


theorem sInfClosed {s : Set α} (h : s ⊆ L) : sInf s ∈ L := L.sInfClosed' h


@[simp] theorem coe_bot : (↑(⊥ : L) : α) = ⊥ := rfl


@[simp] theorem coe_top : (↑(⊤ : L) : α) = ⊤ := rfl


@[simp] theorem coe_sSup (S : Set L) : (↑(sSup S) : α) = sSup {(s : α) | s ∈ S} := rfl


theorem coe_sSup' (S : Set L) : (↑(sSup S) : α) = ⨆ N ∈ S, (N : α) := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    L : CompleteSublattice α
    S : Set (Subtype fun x => Membership.mem L x)
    ⊢ Eq (↑(SupSet.sSup S)) (iSup fun N => iSup fun h => ↑N)
  -/
  rw [coe_sSup, ← Set.image, sSup_image]
  /-
    🎉 no goals
  -/


@[simp] theorem coe_sInf (S : Set L) : (↑(sInf S) : α) = sInf {(s : α) | s ∈ S} := rfl


theorem coe_sInf' (S : Set L) : (↑(sInf S) : α) = ⨅ N ∈ S, (N : α) := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    L : CompleteSublattice α
    S : Set (Subtype fun x => Membership.mem L x)
    ⊢ Eq (↑(InfSet.sInf S)) (iInf fun N => iInf fun h => ↑N)
  -/
  rw [coe_sInf, ← Set.image, sInf_image]
  /-
    🎉 no goals
  -/

-- Redeclaring to get proper keys for these instances

instance : Max {x // x ∈ L} := Sublattice.instSupCoe

instance : Min {x // x ∈ L} := Sublattice.instInfCoe


instance instCompleteLattice : CompleteLattice L :=
  Subtype.coe_injective.completeLattice _
    Sublattice.coe_sup Sublattice.coe_inf coe_sSup' coe_sInf' coe_top coe_bot


/-- The natural complete lattice hom from a complete sublattice to the original lattice. -/
def subtype (L : CompleteSublattice α) : CompleteLatticeHom L α where
  toFun := Subtype.val
  map_sInf' _ := rfl
  map_sSup' _ := rfl


@[simp, norm_cast] lemma coe_subtype (L : CompleteSublattice α) : L.subtype = ((↑) : L → α) := rfl

lemma subtype_apply (L : Sublattice α) (a : L) : L.subtype a = a := rfl


lemma subtype_injective (L : CompleteSublattice α) :
    Injective <| subtype L := Subtype.coe_injective


/-- The push forward of a complete sublattice under a complete lattice hom is a complete
sublattice. -/
@[simps] def map (L : CompleteSublattice α) : CompleteSublattice β where
  carrier := f '' L
  supClosed' := L.supClosed.image f
  infClosed' := L.infClosed.image f
  sSupClosed' := fun s hs ↦ by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : CompleteLattice β
      f : CompleteLatticeHom α β
      L✝ L : CompleteSublattice α
      s : Set β
      hs : HasSubset.Subset s { carrier := Set.image ⇑f ↑L, supClosed' := ⋯, infClos …
      ⊢ Membership.mem { carrier := Set.image ⇑f ↑L, supClosed' := ⋯, infClosed' :=  …
    -/
    obtain ⟨t, ht, rfl⟩ := subset_image_iff.mp hs
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : CompleteLattice β
      f : CompleteLatticeHom α β
      L✝ L : CompleteSublattice α
      t : Set α
      ht : HasSubset.Subset t ↑L
      hs : HasSubset.Subset (Set.image (⇑f) t) { carrier := Set.image ⇑f ↑L, supClos …
      ⊢ Membership.mem { carrier := Set.image ⇑f ↑L, supClosed' := ⋯, infClosed' :=  …
    -/
    rw [← map_sSup]
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : CompleteLattice β
      f : CompleteLatticeHom α β
      L✝ L : CompleteSublattice α
      t : Set α
      ht : HasSubset.Subset t ↑L
      hs : HasSubset.Subset (Set.image (⇑f) t) { carrier := Set.image ⇑f ↑L, supClos …
      ⊢ Membership.mem { carrier := Set.image ⇑f ↑L, supClosed' := ⋯, infClosed' :=  …
    -/
    exact mem_image_of_mem f (sSupClosed ht)
    /-
      🎉 no goals
    -/
  sInfClosed' := fun s hs ↦ by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : CompleteLattice β
      f : CompleteLatticeHom α β
      L✝ L : CompleteSublattice α
      s : Set β
      hs : HasSubset.Subset s { carrier := Set.image ⇑f ↑L, supClosed' := ⋯, infClos …
      ⊢ Membership.mem { carrier := Set.image ⇑f ↑L, supClosed' := ⋯, infClosed' :=  …
    -/
    obtain ⟨t, ht, rfl⟩ := subset_image_iff.mp hs
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : CompleteLattice β
      f : CompleteLatticeHom α β
      L✝ L : CompleteSublattice α
      t : Set α
      ht : HasSubset.Subset t ↑L
      hs : HasSubset.Subset (Set.image (⇑f) t) { carrier := Set.image ⇑f ↑L, supClos …
      ⊢ Membership.mem { carrier := Set.image ⇑f ↑L, supClosed' := ⋯, infClosed' :=  …
    -/
    rw [← map_sInf]
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : CompleteLattice β
      f : CompleteLatticeHom α β
      L✝ L : CompleteSublattice α
      t : Set α
      ht : HasSubset.Subset t ↑L
      hs : HasSubset.Subset (Set.image (⇑f) t) { carrier := Set.image ⇑f ↑L, supClos …
      ⊢ Membership.mem { carrier := Set.image ⇑f ↑L, supClosed' := ⋯, infClosed' :=  …
    -/
    exact mem_image_of_mem f (sInfClosed ht)
    /-
      🎉 no goals
    -/


@[simp] theorem mem_map {b : β} : b ∈ L.map f ↔ ∃ a ∈ L, f a = b := Iff.rfl


/-- The pull back of a complete sublattice under a complete lattice hom is a complete sublattice. -/
@[simps] def comap (L : CompleteSublattice β) : CompleteSublattice α where
  carrier := f ⁻¹' L
  supClosed' := L.supClosed.preimage f
  infClosed' := L.infClosed.preimage f
  sSupClosed' s hs := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : CompleteLattice β
      f : CompleteLatticeHom α β
      L✝ : CompleteSublattice α
      L : CompleteSublattice β
      s : Set α
      hs : HasSubset.Subset s { carrier := Set.preimage ⇑f ↑L, supClosed' := ⋯, infC …
      ⊢ Membership.mem { carrier := Set.preimage ⇑f ↑L, supClosed' := ⋯, infClosed'  …
    -/
    simpa only [mem_preimage, map_sSup, SetLike.mem_coe] using sSupClosed <| mapsTo'.mp hs
    /-
      🎉 no goals
    -/
  sInfClosed' s hs := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : CompleteLattice α
      inst✝ : CompleteLattice β
      f : CompleteLatticeHom α β
      L✝ : CompleteSublattice α
      L : CompleteSublattice β
      s : Set α
      hs : HasSubset.Subset s { carrier := Set.preimage ⇑f ↑L, supClosed' := ⋯, infC …
      ⊢ Membership.mem { carrier := Set.preimage ⇑f ↑L, supClosed' := ⋯, infClosed'  …
    -/
    simpa only [mem_preimage, map_sInf, SetLike.mem_coe] using sInfClosed <| mapsTo'.mp hs
    /-
      🎉 no goals
    -/


@[simp] theorem mem_comap {L : CompleteSublattice β} {a : α} : a ∈ L.comap f ↔ f a ∈ L := Iff.rfl


protected lemma disjoint_iff {a b : L} :
    Disjoint a b ↔ Disjoint (a : α) (b : α) := by
  rw [disjoint_iff, disjoint_iff, ← Sublattice.coe_inf, ← coe_bot (L := L),
    Subtype.coe_injective.eq_iff]


protected lemma codisjoint_iff {a b : L} :
    Codisjoint a b ↔ Codisjoint (a : α) (b : α) := by
  rw [codisjoint_iff, codisjoint_iff, ← Sublattice.coe_sup, ← coe_top (L := L),
    Subtype.coe_injective.eq_iff]


protected lemma isCompl_iff {a b : L} :
    IsCompl a b ↔ IsCompl (a : α) (b : α) := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    L : CompleteSublattice α
    a b : Subtype fun x => Membership.mem L x
    ⊢ Iff (IsCompl a b) (IsCompl ↑a ↑b)
  -/
  rw [isCompl_iff, isCompl_iff, CompleteSublattice.disjoint_iff, CompleteSublattice.codisjoint_iff]
  /-
    🎉 no goals
  -/


lemma isComplemented_iff : ComplementedLattice L ↔ ∀ a ∈ L, ∃ b ∈ L, IsCompl a b := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    L : CompleteSublattice α
    ⊢ Iff (ComplementedLattice (Subtype fun x => Membership.mem L x)) (∀ (a : α),  …
  -/
  refine ⟨fun ⟨h⟩ a ha ↦ ?_, fun h ↦ ⟨fun ⟨a, ha⟩ ↦ ?_⟩⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : CompleteLattice α
      L : CompleteSublattice α
      x✝ : ComplementedLattice (Subtype fun x => Membership.mem L x)
      a : α
      ha : Membership.mem L a
      h : ∀ (a : Subtype fun x => Membership.mem L x), Exists fun b => IsCompl a b
      ⊢ Exists fun b => And (Membership.mem L b) (IsCompl a b)
    -/
  · obtain ⟨b, hb⟩ := h ⟨a, ha⟩
    /-
      case refine_1.intro
      α : Type u_1
      inst✝ : CompleteLattice α
      L : CompleteSublattice α
      x✝ : ComplementedLattice (Subtype fun x => Membership.mem L x)
      a : α
      ha : Membership.mem L a
      h : ∀ (a : Subtype fun x => Membership.mem L x), Exists fun b => IsCompl a b
      b : Subtype fun x => Membership.mem L x
      hb : IsCompl ⟨a, ha⟩ b
      ⊢ Exists fun b => And (Membership.mem L b) (IsCompl a b)
    -/
    exact ⟨b, b.property, CompleteSublattice.isCompl_iff.mp hb⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : CompleteLattice α
      L : CompleteSublattice α
      h : ∀ (a : α), Membership.mem L a → Exists fun b => And (Membership.mem L b) ( …
      x✝ : Subtype fun x => Membership.mem L x
      a : α
      ha : Membership.mem L a
      ⊢ Exists fun b => IsCompl ⟨a, ha⟩ b
    -/
  · obtain ⟨b, hb, hb'⟩ := h a ha
    /-
      case refine_2.intro.intro
      α : Type u_1
      inst✝ : CompleteLattice α
      L : CompleteSublattice α
      h : ∀ (a : α), Membership.mem L a → Exists fun b => And (Membership.mem L b) ( …
      x✝ : Subtype fun x => Membership.mem L x
      a : α
      ha : Membership.mem L a
      b : α
      hb : Membership.mem L b
      hb' : IsCompl a b
      ⊢ Exists fun b => IsCompl ⟨a, ha⟩ b
    -/
    exact ⟨⟨b, hb⟩, CompleteSublattice.isCompl_iff.mpr hb'⟩
    /-
      🎉 no goals
    -/


instance : Top (CompleteSublattice α) := ⟨mk' univ (fun _ _ ↦ mem_univ _) (fun _ _ ↦ mem_univ _)⟩


/-- Copy of a complete sublattice with a new `carrier` equal to the old one. Useful to fix
definitional equalities. -/
protected def copy (s : Set α) (hs : s = L) : CompleteSublattice α :=
  mk' s (hs ▸ L.sSupClosed') (hs ▸ L.sInfClosed')


@[simp, norm_cast] lemma coe_copy (s : Set α) (hs) : L.copy s hs = s := rfl


lemma copy_eq (s : Set α) (hs) : L.copy s hs = L := SetLike.coe_injective hs


/-- The range of a `CompleteLatticeHom` is a `CompleteSublattice`.

See Note [range copy pattern]. -/
protected def range : CompleteSublattice β :=
  (CompleteSublattice.map f ⊤).copy (range f) image_univ.symm


theorem range_coe : (f.range : Set β) = range f := rfl


/-- We can regard a complete lattice homomorphism as an order equivalence to its range. -/
@[simps! apply] noncomputable def toOrderIsoRangeOfInjective (hf : Injective f) : α ≃o f.range :=
  (orderEmbeddingOfInjective f hf).orderIso


