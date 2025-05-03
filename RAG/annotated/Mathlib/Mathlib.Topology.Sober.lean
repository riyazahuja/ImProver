/-- `x` is a generic point of `S` if `S` is the closure of `x`. -/
def IsGenericPoint (x : α) (S : Set α) : Prop :=
  closure ({x} : Set α) = S


theorem isGenericPoint_def {x : α} {S : Set α} : IsGenericPoint x S ↔ closure ({x} : Set α) = S :=
  Iff.rfl


theorem IsGenericPoint.def {x : α} {S : Set α} (h : IsGenericPoint x S) :
    closure ({x} : Set α) = S :=
  h


theorem isGenericPoint_closure {x : α} : IsGenericPoint x (closure ({x} : Set α)) :=
  refl _


theorem isGenericPoint_iff_specializes : IsGenericPoint x S ↔ ∀ y, x ⤳ y ↔ y ∈ S := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    x : α
    S : Set α
    ⊢ Iff (IsGenericPoint x S) (∀ (y : α), Iff (Specializes x y) (Membership.mem S …
  -/
  simp only [specializes_iff_mem_closure, IsGenericPoint, Set.ext_iff]
  /-
    🎉 no goals
  -/


theorem specializes_iff_mem (h : IsGenericPoint x S) : x ⤳ y ↔ y ∈ S :=
  isGenericPoint_iff_specializes.1 h y


protected theorem specializes (h : IsGenericPoint x S) (h' : y ∈ S) : x ⤳ y :=
  h.specializes_iff_mem.2 h'


protected theorem mem (h : IsGenericPoint x S) : x ∈ S :=
  h.specializes_iff_mem.1 specializes_rfl


protected theorem isClosed (h : IsGenericPoint x S) : IsClosed S :=
  h.def ▸ isClosed_closure


protected theorem isIrreducible (h : IsGenericPoint x S) : IsIrreducible S :=
  h.def ▸ isIrreducible_singleton.closure


protected theorem inseparable (h : IsGenericPoint x S) (h' : IsGenericPoint y S) :
    Inseparable x y :=
  (h.specializes h'.mem).antisymm (h'.specializes h.mem)


/-- In a T₀ space, each set has at most one generic point. -/
protected theorem eq [T0Space α] (h : IsGenericPoint x S) (h' : IsGenericPoint y S) : x = y :=
  (h.inseparable h').eq


theorem mem_open_set_iff (h : IsGenericPoint x S) (hU : IsOpen U) : x ∈ U ↔ (S ∩ U).Nonempty :=
  ⟨fun h' => ⟨x, h.mem, h'⟩, fun ⟨_y, hyS, hyU⟩ => (h.specializes hyS).mem_open hU hyU⟩


theorem disjoint_iff (h : IsGenericPoint x S) (hU : IsOpen U) : Disjoint S U ↔ x ∉ U := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    x : α
    S U : Set α
    h : IsGenericPoint x S
    hU : IsOpen U
    ⊢ Iff (Disjoint S U) (Not (Membership.mem U x))
  -/
  rw [h.mem_open_set_iff hU, ← not_disjoint_iff_nonempty_inter, Classical.not_not]
  /-
    🎉 no goals
  -/


theorem mem_closed_set_iff (h : IsGenericPoint x S) (hZ : IsClosed Z) : x ∈ Z ↔ S ⊆ Z := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    x : α
    S Z : Set α
    h : IsGenericPoint x S
    hZ : IsClosed Z
    ⊢ Iff (Membership.mem Z x) (HasSubset.Subset S Z)
  -/
  rw [← h.def, hZ.closure_subset_iff, singleton_subset_iff]
  /-
    🎉 no goals
  -/


protected theorem image (h : IsGenericPoint x S) {f : α → β} (hf : Continuous f) :
    IsGenericPoint (f x) (closure (f '' S)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    x : α
    S : Set α
    h : IsGenericPoint x S
    f : α → β
    hf : Continuous f
    ⊢ IsGenericPoint (f x) (closure (Set.image f S))
  -/
  rw [isGenericPoint_def, ← h.def, ← image_singleton, closure_image_closure hf]
  /-
    🎉 no goals
  -/


theorem isGenericPoint_iff_forall_closed (hS : IsClosed S) (hxS : x ∈ S) :
    IsGenericPoint x S ↔ ∀ Z : Set α, IsClosed Z → x ∈ Z → S ⊆ Z := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    x : α
    S : Set α
    hS : IsClosed S
    hxS : Membership.mem S x
    ⊢ Iff (IsGenericPoint x S) (∀ (Z : Set α), IsClosed Z → Membership.mem Z x → H …
  -/
  have : closure {x} ⊆ S := closure_minimal (singleton_subset_iff.2 hxS) hS
  simp_rw [IsGenericPoint, subset_antisymm_iff, this, true_and, closure, subset_sInter_iff,
    mem_setOf_eq, and_imp, singleton_subset_iff]


/-- A space is sober if every irreducible closed subset has a generic point. -/
@[mk_iff]
class QuasiSober (α : Type*) [TopologicalSpace α] : Prop where
  sober : ∀ {S : Set α}, IsIrreducible S → IsClosed S → ∃ x, IsGenericPoint x S


/-- A generic point of the closure of an irreducible space. -/
noncomputable def IsIrreducible.genericPoint [QuasiSober α] {S : Set α} (hS : IsIrreducible S) :
    α :=
  (QuasiSober.sober hS.closure isClosed_closure).choose


theorem IsIrreducible.isGenericPoint_genericPoint_closure
    [QuasiSober α] {S : Set α} (hS : IsIrreducible S) :
    IsGenericPoint hS.genericPoint (closure S) :=
  (QuasiSober.sober hS.closure isClosed_closure).choose_spec


theorem IsIrreducible.isGenericPoint_genericPoint [QuasiSober α] {S : Set α}
    (hS : IsIrreducible S) (hS' : IsClosed S) :
    IsGenericPoint hS.genericPoint S := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : QuasiSober α
    S : Set α
    hS : IsIrreducible S
    hS' : IsClosed S
    ⊢ IsGenericPoint hS.genericPoint S
  -/
  convert hS.isGenericPoint_genericPoint_closure; exact hS'.closure_eq.symm
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem IsIrreducible.genericPoint_closure_eq [QuasiSober α] {S : Set α} (hS : IsIrreducible S) :
    closure ({hS.genericPoint} : Set α) = closure S :=
  hS.isGenericPoint_genericPoint_closure


theorem IsIrreducible.closure_genericPoint [QuasiSober α] {S : Set α}
    (hS : IsIrreducible S) (hS' : IsClosed S) :
    closure ({hS.genericPoint} : Set α) = S :=
  hS.isGenericPoint_genericPoint_closure.trans hS'.closure_eq


@[deprecated (since := "2024-10-03")]
alias IsIrreducible.genericPoint_spec := IsIrreducible.isGenericPoint_genericPoint_closure


/-- A generic point of a sober irreducible space. -/
noncomputable def genericPoint [QuasiSober α] [IrreducibleSpace α] : α :=
  (IrreducibleSpace.isIrreducible_univ α).genericPoint


theorem genericPoint_spec [QuasiSober α] [IrreducibleSpace α] :
    IsGenericPoint (genericPoint α) univ := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : QuasiSober α
    inst✝ : IrreducibleSpace α
    ⊢ IsGenericPoint (genericPoint α) Set.univ
  -/
  simpa using (IrreducibleSpace.isIrreducible_univ α).isGenericPoint_genericPoint_closure
  /-
    🎉 no goals
  -/


@[simp]
theorem genericPoint_closure [QuasiSober α] [IrreducibleSpace α] :
    closure ({genericPoint α} : Set α) = univ :=
  genericPoint_spec α


theorem genericPoint_specializes [QuasiSober α] [IrreducibleSpace α] (x : α) : genericPoint α ⤳ x :=
                                                                        /-
                                                                          α : Type u_1
                                                                          inst✝² : TopologicalSpace α
                                                                          inst✝¹ : QuasiSober α
                                                                          inst✝ : IrreducibleSpace α
                                                                          x : α
                                                                          ⊢ Membership.mem (closure Set.univ) x
                                                                        -/
  (IsIrreducible.isGenericPoint_genericPoint_closure _).specializes (by simp)
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- The closed irreducible subsets of a sober space bijects with the points of the space. -/
noncomputable def irreducibleSetEquivPoints [QuasiSober α] [T0Space α] :
    TopologicalSpace.IrreducibleCloseds α ≃o α where
  toFun s := s.2.genericPoint
  invFun x := ⟨closure ({x} : Set α), isIrreducible_singleton.closure, isClosed_closure⟩
  left_inv s := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : QuasiSober α
      inst✝ : T0Space α
      s : TopologicalSpace.IrreducibleCloseds α
      ⊢ Eq ((fun x => { carrier := closure (Singleton.singleton x), is_irreducible'  …
    -/
    refine TopologicalSpace.IrreducibleCloseds.ext ?_
    simp only [IsIrreducible.genericPoint_closure_eq, TopologicalSpace.IrreducibleCloseds.coe_mk,
      closure_eq_iff_isClosed.mpr s.3]
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : QuasiSober α
      inst✝ : T0Space α
      s : TopologicalSpace.IrreducibleCloseds α
      ⊢ Eq s.carrier ↑s
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv x := isIrreducible_singleton.closure.isGenericPoint_genericPoint_closure.eq
          /-
            α : Type u_1
            β : Type u_2
            inst✝³ : TopologicalSpace α
            inst✝² : TopologicalSpace β
            inst✝¹ : QuasiSober α
            inst✝ : T0Space α
            x : α
            ⊢ IsGenericPoint x (closure (closure (Singleton.singleton x)))
          -/
      (by rw [closure_closure]; exact isGenericPoint_closure)
                                /-
                                  🎉 no goals
                                -/
  map_rel_iff' := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : QuasiSober α
      inst✝ : T0Space α
      ⊢ ∀ {a b : TopologicalSpace.IrreducibleCloseds α}, Iff (LE.le ({ toFun := fun  …
    -/
    rintro ⟨s, hs, hs'⟩ ⟨t, ht, ht'⟩
    /-
      case mk.mk
      α : Type u_1
      β : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : QuasiSober α
      inst✝ : T0Space α
      s : Set α
      hs : IsIrreducible s
      hs' : IsClosed s
      t : Set α
      ht : IsIrreducible t
      ht' : IsClosed t
      ⊢ Iff (LE.le ({ toFun := fun s => ⋯.genericPoint, invFun := fun x => { carrier …
    -/
    refine specializes_iff_closure_subset.trans ?_
    /-
      case mk.mk
      α : Type u_1
      β : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : QuasiSober α
      inst✝ : T0Space α
      s : Set α
      hs : IsIrreducible s
      hs' : IsClosed s
      t : Set α
      ht : IsIrreducible t
      ht' : IsClosed t
      ⊢ Iff (HasSubset.Subset (closure (Singleton.singleton ({ toFun := fun s => ⋯.g …
    -/
    simp [hs'.closure_eq, ht'.closure_eq]
    /-
      case mk.mk
      α : Type u_1
      β : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : QuasiSober α
      inst✝ : T0Space α
      s : Set α
      hs : IsIrreducible s
      hs' : IsClosed s
      t : Set α
      ht : IsIrreducible t
      ht' : IsClosed t
      ⊢ Iff (HasSubset.Subset s t) (LE.le { carrier := s, is_irreducible' := hs, is_ …
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma Topology.IsClosedEmbedding.quasiSober {f : α → β} (hf : IsClosedEmbedding f) [QuasiSober β] :
    QuasiSober α where
  sober hS hS' := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      hf : Topology.IsClosedEmbedding f
      inst✝ : QuasiSober β
      S✝ : Set α
      hS : IsIrreducible S✝
      hS' : IsClosed S✝
      ⊢ Exists fun x => IsGenericPoint x S✝
    -/
    have hS'' := hS.image f hf.continuous.continuousOn
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      hf : Topology.IsClosedEmbedding f
      inst✝ : QuasiSober β
      S✝ : Set α
      hS : IsIrreducible S✝
      hS' : IsClosed S✝
      hS'' : IsIrreducible (Set.image f S✝)
      ⊢ Exists fun x => IsGenericPoint x S✝
    -/
    obtain ⟨x, hx⟩ := QuasiSober.sober hS'' (hf.isClosedMap _ hS')
    /-
      case intro
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      hf : Topology.IsClosedEmbedding f
      inst✝ : QuasiSober β
      S✝ : Set α
      hS : IsIrreducible S✝
      hS' : IsClosed S✝
      hS'' : IsIrreducible (Set.image f S✝)
      x : β
      hx : IsGenericPoint x (Set.image f S✝)
      ⊢ Exists fun x => IsGenericPoint x S✝
    -/
    obtain ⟨y, -, rfl⟩ := hx.mem
    /-
      case intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      hf : Topology.IsClosedEmbedding f
      inst✝ : QuasiSober β
      S✝ : Set α
      hS : IsIrreducible S✝
      hS' : IsClosed S✝
      hS'' : IsIrreducible (Set.image f S✝)
      y : α
      hx : IsGenericPoint (f y) (Set.image f S✝)
      ⊢ Exists fun x => IsGenericPoint x S✝
    -/
    use y
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      hf : Topology.IsClosedEmbedding f
      inst✝ : QuasiSober β
      S✝ : Set α
      hS : IsIrreducible S✝
      hS' : IsClosed S✝
      hS'' : IsIrreducible (Set.image f S✝)
      y : α
      hx : IsGenericPoint (f y) (Set.image f S✝)
      ⊢ IsGenericPoint y S✝
    -/
    apply image_injective.mpr hf.injective
    /-
      case h.a
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      hf : Topology.IsClosedEmbedding f
      inst✝ : QuasiSober β
      S✝ : Set α
      hS : IsIrreducible S✝
      hS' : IsClosed S✝
      hS'' : IsIrreducible (Set.image f S✝)
      y : α
      hx : IsGenericPoint (f y) (Set.image f S✝)
      ⊢ Eq (Set.image f (closure (Singleton.singleton y))) (Set.image f S✝)
    -/
    rw [← hx.def, ← hf.closure_image_eq, image_singleton]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-20")]
alias ClosedEmbedding.quasiSober := Topology.IsClosedEmbedding.quasiSober


theorem Topology.IsOpenEmbedding.quasiSober {f : α → β} (hf : IsOpenEmbedding f) [QuasiSober β] :
    QuasiSober α where
  sober hS hS' := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      hf : Topology.IsOpenEmbedding f
      inst✝ : QuasiSober β
      S✝ : Set α
      hS : IsIrreducible S✝
      hS' : IsClosed S✝
      ⊢ Exists fun x => IsGenericPoint x S✝
    -/
    have hS'' := hS.image f hf.continuous.continuousOn
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      hf : Topology.IsOpenEmbedding f
      inst✝ : QuasiSober β
      S✝ : Set α
      hS : IsIrreducible S✝
      hS' : IsClosed S✝
      hS'' : IsIrreducible (Set.image f S✝)
      ⊢ Exists fun x => IsGenericPoint x S✝
    -/
    obtain ⟨x, hx⟩ := QuasiSober.sober hS''.closure isClosed_closure
    /-
      case intro
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      hf : Topology.IsOpenEmbedding f
      inst✝ : QuasiSober β
      S✝ : Set α
      hS : IsIrreducible S✝
      hS' : IsClosed S✝
      hS'' : IsIrreducible (Set.image f S✝)
      x : β
      hx : IsGenericPoint x (closure (Set.image f S✝))
      ⊢ Exists fun x => IsGenericPoint x S✝
    -/
    obtain ⟨T, hT, rfl⟩ := hf.isInducing.isClosed_iff.mp hS'
    /-
      case intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      hf : Topology.IsOpenEmbedding f
      inst✝ : QuasiSober β
      x : β
      T : Set β
      hT : IsClosed T
      hS : IsIrreducible (Set.preimage f T)
      hS' : IsClosed (Set.preimage f T)
      hS'' : IsIrreducible (Set.image f (Set.preimage f T))
      hx : IsGenericPoint x (closure (Set.image f (Set.preimage f T)))
      ⊢ Exists fun x => IsGenericPoint x (Set.preimage f T)
    -/
    rw [image_preimage_eq_inter_range] at hx hS''
    have hxT : x ∈ T := by
      rw [← hT.closure_eq]
      exact closure_mono inter_subset_left hx.mem
    obtain ⟨y, rfl⟩ : x ∈ range f := by
      rw [hx.mem_open_set_iff hf.isOpen_range]
      refine Nonempty.mono ?_ hS''.1
      simpa using subset_closure
    /-
      case intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      hf : Topology.IsOpenEmbedding f
      inst✝ : QuasiSober β
      T : Set β
      hT : IsClosed T
      hS : IsIrreducible (Set.preimage f T)
      hS' : IsClosed (Set.preimage f T)
      hS'' : IsIrreducible (Inter.inter T (Set.range f))
      y : α
      hx : IsGenericPoint (f y) (closure (Inter.inter T (Set.range f)))
      hxT : Membership.mem T (f y)
      ⊢ Exists fun x => IsGenericPoint x (Set.preimage f T)
    -/
    use y
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      hf : Topology.IsOpenEmbedding f
      inst✝ : QuasiSober β
      T : Set β
      hT : IsClosed T
      hS : IsIrreducible (Set.preimage f T)
      hS' : IsClosed (Set.preimage f T)
      hS'' : IsIrreducible (Inter.inter T (Set.range f))
      y : α
      hx : IsGenericPoint (f y) (closure (Inter.inter T (Set.range f)))
      hxT : Membership.mem T (f y)
      ⊢ IsGenericPoint y (Set.preimage f T)
    -/
    change _ = _
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      hf : Topology.IsOpenEmbedding f
      inst✝ : QuasiSober β
      T : Set β
      hT : IsClosed T
      hS : IsIrreducible (Set.preimage f T)
      hS' : IsClosed (Set.preimage f T)
      hS'' : IsIrreducible (Inter.inter T (Set.range f))
      y : α
      hx : IsGenericPoint (f y) (closure (Inter.inter T (Set.range f)))
      hxT : Membership.mem T (f y)
      ⊢ Eq (closure (Singleton.singleton y)) (Set.preimage f T)
    -/
    rw [hf.isEmbedding.closure_eq_preimage_closure_image, image_singleton, show _ = _ from hx]
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      hf : Topology.IsOpenEmbedding f
      inst✝ : QuasiSober β
      T : Set β
      hT : IsClosed T
      hS : IsIrreducible (Set.preimage f T)
      hS' : IsClosed (Set.preimage f T)
      hS'' : IsIrreducible (Inter.inter T (Set.range f))
      y : α
      hx : IsGenericPoint (f y) (closure (Inter.inter T (Set.range f)))
      hxT : Membership.mem T (f y)
      ⊢ Eq (Set.preimage f (closure (Inter.inter T (Set.range f)))) (Set.preimage f T)
    -/
    apply image_injective.mpr hf.injective
    /-
      case h.a
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      hf : Topology.IsOpenEmbedding f
      inst✝ : QuasiSober β
      T : Set β
      hT : IsClosed T
      hS : IsIrreducible (Set.preimage f T)
      hS' : IsClosed (Set.preimage f T)
      hS'' : IsIrreducible (Inter.inter T (Set.range f))
      y : α
      hx : IsGenericPoint (f y) (closure (Inter.inter T (Set.range f)))
      hxT : Membership.mem T (f y)
      ⊢ Eq (Set.image f (Set.preimage f (closure (Inter.inter T (Set.range f))))) (S …
    -/
    ext z
    /-
      case h.a.h
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      f : α → β
      hf : Topology.IsOpenEmbedding f
      inst✝ : QuasiSober β
      T : Set β
      hT : IsClosed T
      hS : IsIrreducible (Set.preimage f T)
      hS' : IsClosed (Set.preimage f T)
      hS'' : IsIrreducible (Inter.inter T (Set.range f))
      y : α
      hx : IsGenericPoint (f y) (closure (Inter.inter T (Set.range f)))
      hxT : Membership.mem T (f y)
      z : β
      ⊢ Iff (Membership.mem (Set.image f (Set.preimage f (closure (Inter.inter T (Se …
    -/
    simp only [image_preimage_eq_inter_range, mem_inter_iff, and_congr_left_iff]
    exact fun hy => ⟨fun h => hT.closure_eq ▸ closure_mono inter_subset_left h,
      fun h => subset_closure ⟨h, hy⟩⟩


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.quasiSober := Topology.IsOpenEmbedding.quasiSober


/-- A space is quasi sober if it can be covered by open quasi sober subsets. -/
theorem quasiSober_of_open_cover (S : Set (Set α)) (hS : ∀ s : S, IsOpen (s : Set α))
    [hS' : ∀ s : S, QuasiSober s] (hS'' : ⋃₀ S = ⊤) : QuasiSober α := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    hS : ∀ (s : ↑S), IsOpen ↑s
    hS' : ∀ (s : ↑S), QuasiSober ↑↑s
    hS'' : Eq S.sUnion Top.top
    ⊢ QuasiSober α
  -/
  rw [quasiSober_iff]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    hS : ∀ (s : ↑S), IsOpen ↑s
    hS' : ∀ (s : ↑S), QuasiSober ↑↑s
    hS'' : Eq S.sUnion Top.top
    ⊢ ∀ {S : Set α}, IsIrreducible S → IsClosed S → Exists fun x => IsGenericPoint …
  -/
  intro t h h'
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    hS : ∀ (s : ↑S), IsOpen ↑s
    hS' : ∀ (s : ↑S), QuasiSober ↑↑s
    hS'' : Eq S.sUnion Top.top
    t : Set α
    h : IsIrreducible t
    h' : IsClosed t
    ⊢ Exists fun x => IsGenericPoint x t
  -/
  obtain ⟨x, hx⟩ := h.1
  obtain ⟨U, hU, hU'⟩ : x ∈ ⋃₀ S := by
    rw [hS'']
    trivial
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    hS : ∀ (s : ↑S), IsOpen ↑s
    hS' : ∀ (s : ↑S), QuasiSober ↑↑s
    hS'' : Eq S.sUnion Top.top
    t : Set α
    h : IsIrreducible t
    h' : IsClosed t
    x : α
    hx : Membership.mem t x
    U : Set α
    hU : Membership.mem S U
    hU' : Membership.mem U x
    ⊢ Exists fun x => IsGenericPoint x t
  -/
  haveI : QuasiSober U := hS' ⟨U, hU⟩
  have H : IsPreirreducible ((↑) ⁻¹' t : Set U) :=
    h.2.preimage (hS ⟨U, hU⟩).isOpenEmbedding_subtypeVal
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    hS : ∀ (s : ↑S), IsOpen ↑s
    hS' : ∀ (s : ↑S), QuasiSober ↑↑s
    hS'' : Eq S.sUnion Top.top
    t : Set α
    h : IsIrreducible t
    h' : IsClosed t
    x : α
    hx : Membership.mem t x
    U : Set α
    hU : Membership.mem S U
    hU' : Membership.mem U x
    this : QuasiSober ↑U
    H : IsPreirreducible (Set.preimage Subtype.val t)
    ⊢ Exists fun x => IsGenericPoint x t
  -/
  replace H : IsIrreducible ((↑) ⁻¹' t : Set U) := ⟨⟨⟨x, hU'⟩, by simpa using hx⟩, H⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    hS : ∀ (s : ↑S), IsOpen ↑s
    hS' : ∀ (s : ↑S), QuasiSober ↑↑s
    hS'' : Eq S.sUnion Top.top
    t : Set α
    h : IsIrreducible t
    h' : IsClosed t
    x : α
    hx : Membership.mem t x
    U : Set α
    hU : Membership.mem S U
    hU' : Membership.mem U x
    this : QuasiSober ↑U
    H : IsIrreducible (Set.preimage Subtype.val t)
    ⊢ Exists fun x => IsGenericPoint x t
  -/
  use H.genericPoint
  /-
    case h
    α : Type u_1
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    hS : ∀ (s : ↑S), IsOpen ↑s
    hS' : ∀ (s : ↑S), QuasiSober ↑↑s
    hS'' : Eq S.sUnion Top.top
    t : Set α
    h : IsIrreducible t
    h' : IsClosed t
    x : α
    hx : Membership.mem t x
    U : Set α
    hU : Membership.mem S U
    hU' : Membership.mem U x
    this : QuasiSober ↑U
    H : IsIrreducible (Set.preimage Subtype.val t)
    ⊢ IsGenericPoint (↑H.genericPoint) t
  -/
  have := continuous_subtype_val.closure_preimage_subset _ H.isGenericPoint_genericPoint_closure.mem
  /-
    case h
    α : Type u_1
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    hS : ∀ (s : ↑S), IsOpen ↑s
    hS' : ∀ (s : ↑S), QuasiSober ↑↑s
    hS'' : Eq S.sUnion Top.top
    t : Set α
    h : IsIrreducible t
    h' : IsClosed t
    x : α
    hx : Membership.mem t x
    U : Set α
    hU : Membership.mem S U
    hU' : Membership.mem U x
    this✝ : QuasiSober ↑U
    H : IsIrreducible (Set.preimage Subtype.val t)
    this : Membership.mem (Set.preimage Subtype.val (closure t)) H.genericPoint
    ⊢ IsGenericPoint (↑H.genericPoint) t
  -/
  rw [h'.closure_eq] at this
  /-
    case h
    α : Type u_1
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    hS : ∀ (s : ↑S), IsOpen ↑s
    hS' : ∀ (s : ↑S), QuasiSober ↑↑s
    hS'' : Eq S.sUnion Top.top
    t : Set α
    h : IsIrreducible t
    h' : IsClosed t
    x : α
    hx : Membership.mem t x
    U : Set α
    hU : Membership.mem S U
    hU' : Membership.mem U x
    this✝ : QuasiSober ↑U
    H : IsIrreducible (Set.preimage Subtype.val t)
    this : Membership.mem (Set.preimage Subtype.val t) H.genericPoint
    ⊢ IsGenericPoint (↑H.genericPoint) t
  -/
  apply le_antisymm
    /-
      case h.a
      α : Type u_1
      inst✝ : TopologicalSpace α
      S : Set (Set α)
      hS : ∀ (s : ↑S), IsOpen ↑s
      hS' : ∀ (s : ↑S), QuasiSober ↑↑s
      hS'' : Eq S.sUnion Top.top
      t : Set α
      h : IsIrreducible t
      h' : IsClosed t
      x : α
      hx : Membership.mem t x
      U : Set α
      hU : Membership.mem S U
      hU' : Membership.mem U x
      this✝ : QuasiSober ↑U
      H : IsIrreducible (Set.preimage Subtype.val t)
      this : Membership.mem (Set.preimage Subtype.val t) H.genericPoint
      ⊢ LE.le (closure (Singleton.singleton ↑H.genericPoint)) t
    -/
  · apply h'.closure_subset_iff.mpr
    /-
      case h.a
      α : Type u_1
      inst✝ : TopologicalSpace α
      S : Set (Set α)
      hS : ∀ (s : ↑S), IsOpen ↑s
      hS' : ∀ (s : ↑S), QuasiSober ↑↑s
      hS'' : Eq S.sUnion Top.top
      t : Set α
      h : IsIrreducible t
      h' : IsClosed t
      x : α
      hx : Membership.mem t x
      U : Set α
      hU : Membership.mem S U
      hU' : Membership.mem U x
      this✝ : QuasiSober ↑U
      H : IsIrreducible (Set.preimage Subtype.val t)
      this : Membership.mem (Set.preimage Subtype.val t) H.genericPoint
      ⊢ HasSubset.Subset (Singleton.singleton ↑H.genericPoint) t
    -/
    simpa using this
    /-
      🎉 no goals
    -/
  rw [← image_singleton, ← closure_image_closure continuous_subtype_val,
    H.isGenericPoint_genericPoint_closure.def]
  refine (subset_closure_inter_of_isPreirreducible_of_isOpen h.2 (hS ⟨U, hU⟩) ⟨x, hx, hU'⟩).trans
    (closure_mono ?_)
  /-
    case h.a
    α : Type u_1
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    hS : ∀ (s : ↑S), IsOpen ↑s
    hS' : ∀ (s : ↑S), QuasiSober ↑↑s
    hS'' : Eq S.sUnion Top.top
    t : Set α
    h : IsIrreducible t
    h' : IsClosed t
    x : α
    hx : Membership.mem t x
    U : Set α
    hU : Membership.mem S U
    hU' : Membership.mem U x
    this✝ : QuasiSober ↑U
    H : IsIrreducible (Set.preimage Subtype.val t)
    this : Membership.mem (Set.preimage Subtype.val t) H.genericPoint
    ⊢ HasSubset.Subset (Inter.inter t ↑⟨U, hU⟩) (Set.image Subtype.val (closure (S …
  -/
  rw [inter_comm t, ← Subtype.image_preimage_coe]
  /-
    case h.a
    α : Type u_1
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    hS : ∀ (s : ↑S), IsOpen ↑s
    hS' : ∀ (s : ↑S), QuasiSober ↑↑s
    hS'' : Eq S.sUnion Top.top
    t : Set α
    h : IsIrreducible t
    h' : IsClosed t
    x : α
    hx : Membership.mem t x
    U : Set α
    hU : Membership.mem S U
    hU' : Membership.mem U x
    this✝ : QuasiSober ↑U
    H : IsIrreducible (Set.preimage Subtype.val t)
    this : Membership.mem (Set.preimage Subtype.val t) H.genericPoint
    ⊢ HasSubset.Subset (Set.image Subtype.val (Set.preimage Subtype.val t)) (Set.i …
  -/
  exact Set.image_subset _ subset_closure
  /-
    🎉 no goals
  -/


/-- Any Hausdorff space is a quasi-sober space because any irreducible set is a singleton. -/
instance (priority := 100) T2Space.quasiSober [T2Space α] : QuasiSober α where
  sober h _ := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : T2Space α
      S✝ : Set α
      h : IsIrreducible S✝
      x✝ : IsClosed S✝
      ⊢ Exists fun x => IsGenericPoint x S✝
    -/
    obtain ⟨x, rfl⟩ := isIrreducible_iff_singleton.mp h
    /-
      case intro
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : T2Space α
      x : α
      h : IsIrreducible (Singleton.singleton x)
      x✝ : IsClosed (Singleton.singleton x)
      ⊢ Exists fun x_1 => IsGenericPoint x_1 (Singleton.singleton x)
    -/
    exact ⟨x, closure_singleton⟩
    /-
      🎉 no goals
    -/


variable (α) in
/-- The set of generic points of irreducible components. -/
def genericPoints : Set α := { x | closure {x} ∈ irreducibleComponents α }


/-- The irreducible component of a generic point -/
def component (x : genericPoints α) : irreducibleComponents α :=
  ⟨closure {x.1}, x.2⟩


lemma isGenericPoint (x : genericPoints α) : IsGenericPoint x.1 (component x).1 := rfl


lemma component_injective [T0Space α] : Function.Injective (component (α := α)) :=
  fun x y e ↦ Subtype.ext ((isGenericPoint x).eq (e ▸ isGenericPoint y))


/-- The generic point of an irreducible component. -/
noncomputable
def ofComponent [QuasiSober α] (x : irreducibleComponents α) : genericPoints α :=
  ⟨x.2.1.genericPoint, show _ ∈ irreducibleComponents α from
    (x.2.1.isGenericPoint_genericPoint (isClosed_of_mem_irreducibleComponents x.1 x.2)).symm ▸ x.2⟩


lemma isGenericPoint_ofComponent [QuasiSober α] (x : irreducibleComponents α) :
    IsGenericPoint (ofComponent x).1 x :=
    x.2.1.isGenericPoint_genericPoint (isClosed_of_mem_irreducibleComponents x.1 x.2)


@[simp]
lemma component_ofComponent [QuasiSober α] (x : irreducibleComponents α) :
    component (ofComponent x) = x :=
  Subtype.ext (isGenericPoint_ofComponent x)


@[simp]
lemma ofComponent_component [T0Space α] [QuasiSober α] (x : genericPoints α) :
    ofComponent (component x) = x :=
  component_injective (component_ofComponent _)


lemma component_surjective [QuasiSober α] : Function.Surjective (component (α := α)) :=
  Function.HasRightInverse.surjective ⟨ofComponent, component_ofComponent⟩


lemma finite [T0Space α] (h : (irreducibleComponents α).Finite) : (genericPoints α).Finite :=
  @Finite.of_injective _ _ h _ component_injective


/-- In a sober space, the generic points corresponds bijectively to irreducible components -/
@[simps]
noncomputable
def equiv [T0Space α] [QuasiSober α] : genericPoints α ≃ irreducibleComponents α :=
  ⟨component, ofComponent, ofComponent_component, component_ofComponent⟩


lemma closure [QuasiSober α] : closure (genericPoints α) = Set.univ := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : QuasiSober α
    ⊢ Eq (_root_.closure (genericPoints α)) Set.univ
  -/
  refine Set.eq_univ_iff_forall.mpr fun x ↦ Set.subset_def.mp ?_ x mem_irreducibleComponent
  refine (isGenericPoint_ofComponent
    ⟨_, irreducibleComponent_mem_irreducibleComponents x⟩).symm.trans_subset (closure_mono ?_)
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : QuasiSober α
    x : α
    ⊢ HasSubset.Subset (Singleton.singleton ↑(genericPoints.ofComponent ⟨irreducib …
  -/
  exact Set.singleton_subset_iff.mpr (ofComponent _).2
  /-
    🎉 no goals
  -/


lemma genericPoints_eq_singleton [QuasiSober α] [IrreducibleSpace α] [T0Space α] :
    genericPoints α = {genericPoint α} := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : QuasiSober α
    inst✝¹ : IrreducibleSpace α
    inst✝ : T0Space α
    ⊢ Eq (genericPoints α) (Singleton.singleton (genericPoint α))
  -/
  ext x
  /-
    case h
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : QuasiSober α
    inst✝¹ : IrreducibleSpace α
    inst✝ : T0Space α
    x : α
    ⊢ Iff (Membership.mem (genericPoints α) x) (Membership.mem (Singleton.singleto …
  -/
  rw [genericPoints, irreducibleComponents_eq_singleton]
  /-
    case h
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : QuasiSober α
    inst✝¹ : IrreducibleSpace α
    inst✝ : T0Space α
    x : α
    ⊢ Iff (Membership.mem (setOf fun x => Membership.mem (Singleton.singleton Set. …
  -/
  exact ⟨((genericPoint_spec α).eq · |>.symm), (· ▸ genericPoint_spec α)⟩
  /-
    🎉 no goals
  -/


