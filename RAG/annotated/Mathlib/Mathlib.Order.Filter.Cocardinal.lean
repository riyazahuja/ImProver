variable (α) in
/-- The filter defined by all sets that have a complement with at most cardinality `c`. For a union
of `c` sets of `c` elements to have `c` elements, we need that `c` is a regular cardinal. -/
def cocardinal (hreg : c.IsRegular) : Filter α := by
  /-
    α : Type u
    c : Cardinal.{u}
    hreg✝ hreg : c.IsRegular
    ⊢ Filter α
  -/
  apply ofCardinalUnion {s | Cardinal.mk s < c} (lt_of_lt_of_le (nat_lt_aleph0 2) hreg.aleph0_le)
    /-
      case hUnion
      α : Type u
      c : Cardinal.{u}
      hreg✝ hreg : c.IsRegular
      ⊢ ∀ (S : Set (Set α)), LT.lt (Cardinal.mk ↑S) c → (∀ (s : Set α), Membership.m …
    -/
  · refine fun s hS hSc ↦ lt_of_le_of_lt (mk_sUnion_le _) <| mul_lt_of_lt hreg.aleph0_le hS ?_
    /-
      case hUnion
      α : Type u
      c : Cardinal.{u}
      hreg✝ hreg : c.IsRegular
      s : Set (Set α)
      hS : LT.lt (Cardinal.mk ↑s) c
      hSc : ∀ (s_1 : Set α), Membership.mem s s_1 → Membership.mem (setOf fun s => L …
      ⊢ LT.lt (iSup fun s_1 => Cardinal.mk ↑↑s_1) c
    -/
    exact iSup_lt_of_isRegular hreg hS fun i ↦ hSc i i.property
    /-
      🎉 no goals
    -/
    /-
      case hmono
      α : Type u
      c : Cardinal.{u}
      hreg✝ hreg : c.IsRegular
      ⊢ ∀ (t : Set α), Membership.mem (setOf fun s => LT.lt (Cardinal.mk ↑s) c) t →  …
    -/
  · exact fun _ hSc _ ht ↦ lt_of_le_of_lt (mk_le_mk_of_subset ht) hSc
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_cocardinal {s : Set α} :
    s ∈ cocardinal α hreg ↔ Cardinal.mk (sᶜ : Set α) < c := Iff.rfl


@[simp] lemma cocardinal_aleph0_eq_cofinite :
    cocardinal (α := α) isRegular_aleph0 = cofinite := by
  /-
    α : Type u
    ⊢ Eq (Filter.cocardinal α Cardinal.isRegular_aleph0) Filter.cofinite
  -/
  aesop
  /-
    🎉 no goals
  -/


instance instCardinalInterFilter_cocardinal : CardinalInterFilter (cocardinal (α := α) hreg) c where
  cardinal_sInter_mem S hS hSs := by
    /-
      α : Type u
      c : Cardinal.{u}
      hreg : c.IsRegular
      S : Set (Set α)
      hS : LT.lt (Cardinal.mk ↑S) c
      hSs : ∀ (s : Set α), Membership.mem S s → Membership.mem (Filter.cocardinal α  …
      ⊢ Membership.mem (Filter.cocardinal α hreg) S.sInter
    -/
    rw [mem_cocardinal, Set.compl_sInter]
    /-
      α : Type u
      c : Cardinal.{u}
      hreg : c.IsRegular
      S : Set (Set α)
      hS : LT.lt (Cardinal.mk ↑S) c
      hSs : ∀ (s : Set α), Membership.mem S s → Membership.mem (Filter.cocardinal α  …
      ⊢ LT.lt (Cardinal.mk ↑(Set.image HasCompl.compl S).sUnion) c
    -/
    apply lt_of_le_of_lt (mk_sUnion_le _)
    /-
      α : Type u
      c : Cardinal.{u}
      hreg : c.IsRegular
      S : Set (Set α)
      hS : LT.lt (Cardinal.mk ↑S) c
      hSs : ∀ (s : Set α), Membership.mem S s → Membership.mem (Filter.cocardinal α  …
      ⊢ LT.lt (HMul.hMul (Cardinal.mk ↑(Set.image HasCompl.compl S)) (iSup fun s =>  …
    -/
    apply mul_lt_of_lt hreg.aleph0_le (lt_of_le_of_lt mk_image_le hS)
    /-
      α : Type u
      c : Cardinal.{u}
      hreg : c.IsRegular
      S : Set (Set α)
      hS : LT.lt (Cardinal.mk ↑S) c
      hSs : ∀ (s : Set α), Membership.mem S s → Membership.mem (Filter.cocardinal α  …
      ⊢ LT.lt (iSup fun s => Cardinal.mk ↑↑s) c
    -/
    apply iSup_lt_of_isRegular hreg <| lt_of_le_of_lt mk_image_le hS
    /-
      α : Type u
      c : Cardinal.{u}
      hreg : c.IsRegular
      S : Set (Set α)
      hS : LT.lt (Cardinal.mk ↑S) c
      hSs : ∀ (s : Set α), Membership.mem S s → Membership.mem (Filter.cocardinal α  …
      ⊢ ∀ (i : ↑(Set.image HasCompl.compl S)), LT.lt (Cardinal.mk ↑↑i) c
    -/
    aesop
    /-
      🎉 no goals
    -/


@[simp]
theorem eventually_cocardinal {p : α → Prop} :
    (∀ᶠ x in cocardinal α hreg, p x) ↔ #{ x | ¬p x } < c := Iff.rfl


theorem hasBasis_cocardinal : HasBasis (cocardinal α hreg) {s : Set α | #s < c} compl :=
  ⟨fun s =>
    ⟨fun h => ⟨sᶜ, h, (compl_compl s).subset⟩, fun ⟨_t, htf, hts⟩ => by
      have : #↑sᶜ < c := by
        apply lt_of_le_of_lt _ htf
        rw [compl_subset_comm] at hts
        apply Cardinal.mk_le_mk_of_subset hts
      /-
        α : Type u
        c : Cardinal.{u}
        hreg : c.IsRegular
        s : Set α
        x✝ : Exists fun i => And (setOf (fun s => LT.lt (Cardinal.mk ↑s) c) i) (HasSub …
        _t : Set α
        htf : setOf (fun s => LT.lt (Cardinal.mk ↑s) c) _t
        hts : HasSubset.Subset (HasCompl.compl _t) s
        this : LT.lt (Cardinal.mk ↑(HasCompl.compl s)) c
        ⊢ Membership.mem (Filter.cocardinal α hreg) s
      -/
      simp_all only [mem_cocardinal] ⟩⟩
      /-
        🎉 no goals
      -/


theorem frequently_cocardinal {p : α → Prop} :
    (∃ᶠ x in cocardinal α hreg, p x) ↔ c ≤ # { x | p x } := by
  /-
    α : Type u
    c : Cardinal.{u}
    hreg : c.IsRegular
    p : α → Prop
    ⊢ Iff (Filter.Frequently (fun x => p x) (Filter.cocardinal α hreg)) (LE.le c ( …
  -/
  simp only [Filter.Frequently, eventually_cocardinal, not_not,coe_setOf, not_lt]
  /-
    🎉 no goals
  -/


lemma frequently_cocardinal_mem {s : Set α} :
    (∃ᶠ x in cocardinal α hreg, x ∈ s) ↔ c ≤ #s := frequently_cocardinal


@[simp]
lemma cocardinal_inf_principal_neBot_iff {s : Set α} :
    (cocardinal α hreg ⊓ 𝓟 s).NeBot ↔ c ≤ #s :=
  frequently_mem_iff_neBot.symm.trans frequently_cocardinal


theorem compl_mem_cocardinal_of_card_lt {s : Set α} (hs : #s < c) :
    sᶜ ∈ cocardinal α hreg :=
  mem_cocardinal.2 <| (compl_compl s).symm ▸ hs


theorem _root_.Set.Finite.compl_mem_cocardinal {s : Set α} (hs : s.Finite) :
    sᶜ ∈ cocardinal α hreg :=
  compl_mem_cocardinal_of_card_lt <| lt_of_lt_of_le (Finite.lt_aleph0 hs) (hreg.aleph0_le)


theorem eventually_cocardinal_nmem_of_card_lt  {s : Set α} (hs : #s < c) :
    ∀ᶠ x in cocardinal α hreg, x ∉ s :=
  compl_mem_cocardinal_of_card_lt hs


theorem _root_.Finset.eventually_cocardinal_nmem (s : Finset α) :
    ∀ᶠ x in cocardinal α hreg, x ∉ s :=
  eventually_cocardinal_nmem_of_card_lt <| lt_of_lt_of_le (finset_card_lt_aleph0 s) (hreg.aleph0_le)


theorem eventually_cocardinal_ne (x : α) : ∀ᶠ a in cocardinal α hreg, a ≠ x := by
  /-
    α : Type u
    c : Cardinal.{u}
    hreg : c.IsRegular
    x : α
    ⊢ Filter.Eventually (fun a => Ne a x) (Filter.cocardinal α hreg)
  -/
  simpa [Set.finite_singleton x] using hreg.nat_lt 1
  /-
    🎉 no goals
  -/


/-- The filter defined by all sets that have countable complements. -/
abbrev cocountable : Filter α := cocardinal α Cardinal.isRegular_aleph_one


theorem mem_cocountable {s : Set α} :
    s ∈ cocountable ↔ (sᶜ : Set α).Countable := by
  /-
    α : Type u
    s : Set α
    ⊢ Iff (Membership.mem Filter.cocountable s) (HasCompl.compl s).Countable
  -/
  rw [Cardinal.countable_iff_lt_aleph_one, mem_cocardinal]
  /-
    🎉 no goals
  -/


