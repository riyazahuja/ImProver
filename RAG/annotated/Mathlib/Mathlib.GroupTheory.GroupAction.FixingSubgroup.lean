/-- The submonoid fixing a set under a `MulAction`. -/
@[to_additive " The additive submonoid fixing a set under an `AddAction`. "]
def fixingSubmonoid (s : Set α) : Submonoid M where
  carrier := { ϕ : M | ∀ x : s, ϕ • (x : α) = x }
  one_mem' _ := one_smul _ _
                               /-
                                 M : Type u_1
                                 α : Type u_2
                                 inst✝¹ : Monoid M
                                 inst✝ : MulAction M α
                                 s : Set α
                                 x y : M
                                 hx : Membership.mem (setOf fun ϕ => ∀ (x : ↑s), Eq (HSMul.hSMul ϕ ↑x) ↑x) x
                                 hy : Membership.mem (setOf fun ϕ => ∀ (x : ↑s), Eq (HSMul.hSMul ϕ ↑x) ↑x) y
                                 z : ↑s
                                 ⊢ Eq (HSMul.hSMul (HMul.hMul x y) ↑z) ↑z
                               -/
  mul_mem' {x y} hx hy z := by rw [mul_smul, hy z, hx z]
                               /-
                                 🎉 no goals
                               -/


theorem mem_fixingSubmonoid_iff {s : Set α} {m : M} :
    m ∈ fixingSubmonoid M s ↔ ∀ y ∈ s, m • y = y :=
  ⟨fun hg y hy => hg ⟨y, hy⟩, fun h ⟨y, hy⟩ => h y hy⟩


/-- The Galois connection between fixing submonoids and fixed points of a monoid action -/
theorem fixingSubmonoid_fixedPoints_gc :
    GaloisConnection (OrderDual.toDual ∘ fixingSubmonoid M)
      ((fun P : Submonoid M => fixedPoints P α) ∘ OrderDual.ofDual) :=
  fun _s _P => ⟨fun h s hs p => h p.2 ⟨s, hs⟩, fun h p hp s => h s.2 ⟨p, hp⟩⟩


theorem fixingSubmonoid_antitone : Antitone fun s : Set α => fixingSubmonoid M s :=
  (fixingSubmonoid_fixedPoints_gc M α).monotone_l


theorem fixedPoints_antitone : Antitone fun P : Submonoid M => fixedPoints P α :=
  (fixingSubmonoid_fixedPoints_gc M α).monotone_u.dual_left


/-- Fixing submonoid of union is intersection -/
theorem fixingSubmonoid_union {s t : Set α} :
    fixingSubmonoid M (s ∪ t) = fixingSubmonoid M s ⊓ fixingSubmonoid M t :=
  (fixingSubmonoid_fixedPoints_gc M α).l_sup


/-- Fixing submonoid of iUnion is intersection -/
theorem fixingSubmonoid_iUnion {ι : Sort*} {s : ι → Set α} :
    fixingSubmonoid M (⋃ i, s i) = ⨅ i, fixingSubmonoid M (s i) :=
  (fixingSubmonoid_fixedPoints_gc M α).l_iSup


/-- Fixed points of sup of submonoids is intersection -/
theorem fixedPoints_submonoid_sup {P Q : Submonoid M} :
    fixedPoints (↥(P ⊔ Q)) α = fixedPoints P α ∩ fixedPoints Q α :=
  (fixingSubmonoid_fixedPoints_gc M α).u_inf


/-- Fixed points of iSup of submonoids is intersection -/
theorem fixedPoints_submonoid_iSup {ι : Sort*} {P : ι → Submonoid M} :
    fixedPoints (↥(iSup P)) α = ⋂ i, fixedPoints (P i) α :=
  (fixingSubmonoid_fixedPoints_gc M α).u_iInf


/-- The subgroup fixing a set under a `MulAction`. -/
@[to_additive " The additive subgroup fixing a set under an `AddAction`. "]
def fixingSubgroup (s : Set α) : Subgroup M :=
                                                        /-
                                                          M : Type u_1
                                                          α : Type u_2
                                                          inst✝¹ : Group M
                                                          inst✝ : MulAction M α
                                                          s : Set α
                                                          x✝ : M
                                                          hx : Membership.mem __src✝.carrier x✝
                                                          z : ↑s
                                                          ⊢ Eq (HSMul.hSMul (Inv.inv x✝) ↑z) ↑z
                                                        -/
  { fixingSubmonoid M s with inv_mem' := fun hx z => by rw [inv_smul_eq_iff, hx z] }
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem mem_fixingSubgroup_iff {s : Set α} {m : M} : m ∈ fixingSubgroup M s ↔ ∀ y ∈ s, m • y = y :=
  ⟨fun hg y hy => hg ⟨y, hy⟩, fun h ⟨y, hy⟩ => h y hy⟩


theorem mem_fixingSubgroup_iff_subset_fixedBy {s : Set α} {m : M} :
    m ∈ fixingSubgroup M s ↔ s ⊆ fixedBy α m := by
  /-
    M : Type u_1
    α : Type u_2
    inst✝¹ : Group M
    inst✝ : MulAction M α
    s : Set α
    m : M
    ⊢ Iff (Membership.mem (fixingSubgroup M s) m) (HasSubset.Subset s (MulAction.f …
  -/
  simp_rw [mem_fixingSubgroup_iff, Set.subset_def, mem_fixedBy]
  /-
    🎉 no goals
  -/


theorem mem_fixingSubgroup_compl_iff_movedBy_subset {s : Set α} {m : M} :
    m ∈ fixingSubgroup M sᶜ ↔ (fixedBy α m)ᶜ ⊆ s := by
  /-
    M : Type u_1
    α : Type u_2
    inst✝¹ : Group M
    inst✝ : MulAction M α
    s : Set α
    m : M
    ⊢ Iff (Membership.mem (fixingSubgroup M (HasCompl.compl s)) m) (HasSubset.Subs …
  -/
  rw [mem_fixingSubgroup_iff_subset_fixedBy, Set.compl_subset_comm]
  /-
    🎉 no goals
  -/


/-- The Galois connection between fixing subgroups and fixed points of a group action -/
theorem fixingSubgroup_fixedPoints_gc :
    GaloisConnection (OrderDual.toDual ∘ fixingSubgroup M)
      ((fun P : Subgroup M => fixedPoints P α) ∘ OrderDual.ofDual) :=
  fun _s _P => ⟨fun h s hs p => h p.2 ⟨s, hs⟩, fun h p hp s => h s.2 ⟨p, hp⟩⟩


theorem fixingSubgroup_antitone : Antitone (fixingSubgroup M : Set α → Subgroup M) :=
  (fixingSubgroup_fixedPoints_gc M α).monotone_l


theorem fixedPoints_subgroup_antitone : Antitone fun P : Subgroup M => fixedPoints P α :=
  (fixingSubgroup_fixedPoints_gc M α).monotone_u.dual_left


/-- Fixing subgroup of union is intersection -/
theorem fixingSubgroup_union {s t : Set α} :
    fixingSubgroup M (s ∪ t) = fixingSubgroup M s ⊓ fixingSubgroup M t :=
  (fixingSubgroup_fixedPoints_gc M α).l_sup


/-- Fixing subgroup of iUnion is intersection -/
theorem fixingSubgroup_iUnion {ι : Sort*} {s : ι → Set α} :
    fixingSubgroup M (⋃ i, s i) = ⨅ i, fixingSubgroup M (s i) :=
  (fixingSubgroup_fixedPoints_gc M α).l_iSup


/-- Fixed points of sup of subgroups is intersection -/
theorem fixedPoints_subgroup_sup {P Q : Subgroup M} :
    fixedPoints (↥(P ⊔ Q)) α = fixedPoints P α ∩ fixedPoints Q α :=
  (fixingSubgroup_fixedPoints_gc M α).u_inf


/-- Fixed points of iSup of subgroups is intersection -/
theorem fixedPoints_subgroup_iSup {ι : Sort*} {P : ι → Subgroup M} :
    fixedPoints (↥(iSup P)) α = ⋂ i, fixedPoints (P i) α :=
  (fixingSubgroup_fixedPoints_gc M α).u_iInf


/-- The orbit of the fixing subgroup of `sᶜ` (ie. the moving subgroup of `s`) is a subset of `s` -/
theorem orbit_fixingSubgroup_compl_subset {s : Set α} {a : α} (a_in_s : a ∈ s) :
    MulAction.orbit (fixingSubgroup M sᶜ) a ⊆ s := by
  /-
    M : Type u_1
    α : Type u_2
    inst✝¹ : Group M
    inst✝ : MulAction M α
    s : Set α
    a : α
    a_in_s : Membership.mem s a
    ⊢ HasSubset.Subset (MulAction.orbit (Subtype fun x => Membership.mem (fixingSu …
  -/
  intro b b_in_orbit
  /-
    M : Type u_1
    α : Type u_2
    inst✝¹ : Group M
    inst✝ : MulAction M α
    s : Set α
    a : α
    a_in_s : Membership.mem s a
    b : α
    b_in_orbit : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem  …
    ⊢ Membership.mem s b
  -/
  let ⟨⟨g, g_fixing⟩, g_eq⟩ := MulAction.mem_orbit_iff.mp b_in_orbit
  /-
    M : Type u_1
    α : Type u_2
    inst✝¹ : Group M
    inst✝ : MulAction M α
    s : Set α
    a : α
    a_in_s : Membership.mem s a
    b : α
    b_in_orbit : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem  …
    g : M
    g_fixing : Membership.mem (fixingSubgroup M (HasCompl.compl s)) g
    g_eq : Eq (HSMul.hSMul ⟨g, g_fixing⟩ a) b
    ⊢ Membership.mem s b
  -/
  rw [Submonoid.mk_smul] at g_eq
  /-
    M : Type u_1
    α : Type u_2
    inst✝¹ : Group M
    inst✝ : MulAction M α
    s : Set α
    a : α
    a_in_s : Membership.mem s a
    b : α
    b_in_orbit : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem  …
    g : M
    g_fixing : Membership.mem (fixingSubgroup M (HasCompl.compl s)) g
    g_eq : Eq (HSMul.hSMul g a) b
    ⊢ Membership.mem s b
  -/
  rw [mem_fixingSubgroup_compl_iff_movedBy_subset] at g_fixing
  /-
    M : Type u_1
    α : Type u_2
    inst✝¹ : Group M
    inst✝ : MulAction M α
    s : Set α
    a : α
    a_in_s : Membership.mem s a
    b : α
    b_in_orbit : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem  …
    g : M
    g_fixing : HasSubset.Subset (HasCompl.compl (MulAction.fixedBy α g)) s
    g_eq : Eq (HSMul.hSMul g a) b
    ⊢ Membership.mem s b
  -/
  rwa [← g_eq, smul_mem_of_set_mem_fixedBy (set_mem_fixedBy_of_movedBy_subset g_fixing)]
  /-
    🎉 no goals
  -/


