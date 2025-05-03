/-- The type of closed subgroups of a topological group. -/
@[ext]
structure ClosedSubgroup (G : Type u) [Group G] [TopologicalSpace G] extends Subgroup G where
  isClosed' : IsClosed carrier


/-- The type of closed subgroups of an additive topological group. -/
@[ext]
structure ClosedAddSubgroup (G : Type u) [AddGroup G] [TopologicalSpace G] extends
    AddSubgroup G where
  isClosed' : IsClosed carrier


variable {G} in
@[to_additive]
theorem toSubgroup_injective : Function.Injective
    (ClosedSubgroup.toSubgroup : ClosedSubgroup G → Subgroup G) :=
  fun A B h ↦ by
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : TopologicalSpace G
    A B : ClosedSubgroup G
    h : Eq ↑A ↑B
    ⊢ Eq A B
  -/
  ext
  /-
    case carrier.h
    G : Type u
    inst✝¹ : Group G
    inst✝ : TopologicalSpace G
    A B : ClosedSubgroup G
    h : Eq ↑A ↑B
    x✝ : G
    ⊢ Iff (Membership.mem (↑A).carrier x✝) (Membership.mem (↑B).carrier x✝)
  -/
  rw [h]
  /-
    🎉 no goals
  -/


@[to_additive]
instance : SetLike (ClosedSubgroup G) G where
  coe U := U.1
  coe_injective' _ _ h := toSubgroup_injective <| SetLike.ext' h


@[to_additive]
instance : SubgroupClass (ClosedSubgroup G) G where
  mul_mem := Subsemigroup.mul_mem' _
  one_mem U := U.one_mem'
  inv_mem := Subgroup.inv_mem' _


@[to_additive]
instance : Coe (ClosedSubgroup G) (Subgroup G) where
  coe := toSubgroup


@[to_additive]
instance instInfClosedSubgroup : Min (ClosedSubgroup G) :=
  ⟨fun U V ↦ ⟨U ⊓ V, U.isClosed'.inter V.isClosed'⟩⟩


@[to_additive]
instance instSemilatticeInfClosedSubgroup : SemilatticeInf (ClosedSubgroup G) :=
  SetLike.coe_injective.semilatticeInf ((↑) : ClosedSubgroup G → Set G) fun _ _ ↦ rfl


@[to_additive]
instance [CompactSpace G] (H : ClosedSubgroup G) : CompactSpace H :=
  isCompact_iff_compactSpace.mp (IsClosed.isCompact H.isClosed')


lemma normalCore_isClosed (H : Subgroup G) (h : IsClosed (H : Set G)) :
    IsClosed (H.normalCore : Set G) := by
  /-
    G : Type u
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousMul G
    H : Subgroup G
    h : IsClosed ↑H
    ⊢ IsClosed ↑H.normalCore
  -/
  rw [normalCore_eq_iInf_conjAct]
  /-
    G : Type u
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousMul G
    H : Subgroup G
    h : IsClosed ↑H
    ⊢ IsClosed ↑(iInf fun g => HSMul.hSMul g H)
  -/
  push_cast
  /-
    G : Type u
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousMul G
    H : Subgroup G
    h : IsClosed ↑H
    ⊢ IsClosed (Set.iInter fun i => ↑(HSMul.hSMul i H))
  -/
  apply isClosed_iInter
  /-
    case h
    G : Type u
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousMul G
    H : Subgroup G
    h : IsClosed ↑H
    ⊢ ∀ (i : ConjAct G), IsClosed ↑(HSMul.hSMul i H)
  -/
  intro g
  /-
    case h
    G : Type u
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousMul G
    H : Subgroup G
    h : IsClosed ↑H
    g : ConjAct G
    ⊢ IsClosed ↑(HSMul.hSMul g H)
  -/
  convert IsClosed.preimage (TopologicalGroup.continuous_conj (ConjAct.ofConjAct g⁻¹)) h
  /-
    case h.e'_3
    G : Type u
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousMul G
    H : Subgroup G
    h : IsClosed ↑H
    g : ConjAct G
    ⊢ Eq (↑(HSMul.hSMul g H)) (Set.preimage (fun h => HMul.hMul (HMul.hMul (ConjAc …
  -/
  exact Set.ext (fun t ↦ Set.mem_smul_set_iff_inv_smul_mem)
  /-
    🎉 no goals
  -/


@[to_additive]
lemma isOpen_of_isClosed_of_finiteIndex (H : Subgroup G) [H.FiniteIndex]
    (h : IsClosed (H : Set G)) : IsOpen (H : Set G) := by
  /-
    G : Type u
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : ContinuousMul G
    H : Subgroup G
    inst✝ : H.FiniteIndex
    h : IsClosed ↑H
    ⊢ IsOpen ↑H
  -/
  apply isClosed_compl_iff.mp
  convert isClosed_iUnion_of_finite <| fun (x : {x : (G ⧸ H) // x ≠ QuotientGroup.mk 1})
    ↦ IsClosed.smul h (Quotient.out x.1)
  /-
    case h.e'_3
    G : Type u
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : ContinuousMul G
    H : Subgroup G
    inst✝ : H.FiniteIndex
    h : IsClosed ↑H
    ⊢ Eq (HasCompl.compl ↑H) (Set.iUnion fun i => HSMul.hSMul (Quotient.out ↑i) ↑H)
  -/
  ext x
  /-
    case h.e'_3.h
    G : Type u
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : ContinuousMul G
    H : Subgroup G
    inst✝ : H.FiniteIndex
    h : IsClosed ↑H
    x : G
    ⊢ Iff (Membership.mem (HasCompl.compl ↑H) x) (Membership.mem (Set.iUnion fun i …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
  · have : QuotientGroup.mk 1 ≠ QuotientGroup.mk (s := H) x := by
      apply QuotientGroup.eq.not.mpr
      simpa only [inv_one, one_mul, ne_eq]
    /-
      case h.e'_3.h.refine_1
      G : Type u
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : ContinuousMul G
      H : Subgroup G
      inst✝ : H.FiniteIndex
      h✝ : IsClosed ↑H
      x : G
      h : Membership.mem (HasCompl.compl ↑H) x
      this : Ne ↑1 ↑x
      ⊢ Membership.mem (Set.iUnion fun i => HSMul.hSMul (Quotient.out ↑i) ↑H) x
    -/
    simp only [ne_eq, Set.mem_iUnion]
    use ⟨QuotientGroup.mk (s := H) x, this.symm⟩,
      (Quotient.out (QuotientGroup.mk (s := H) x))⁻¹ * x
    /-
      case h
      G : Type u
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : ContinuousMul G
      H : Subgroup G
      inst✝ : H.FiniteIndex
      h✝ : IsClosed ↑H
      x : G
      h : Membership.mem (HasCompl.compl ↑H) x
      this : Ne ↑1 ↑x
      ⊢ And (Membership.mem (↑H) (HMul.hMul (Inv.inv (Quotient.out ↑x)) x)) (Eq ((fu …
    -/
    simp only [SetLike.mem_coe, smul_eq_mul, mul_inv_cancel_left, and_true]
    /-
      case h
      G : Type u
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : ContinuousMul G
      H : Subgroup G
      inst✝ : H.FiniteIndex
      h✝ : IsClosed ↑H
      x : G
      h : Membership.mem (HasCompl.compl ↑H) x
      this : Ne ↑1 ↑x
      ⊢ Membership.mem H (HMul.hMul (Inv.inv (Quotient.out ↑x)) x)
    -/
    exact QuotientGroup.eq.mp <| QuotientGroup.out_eq' (QuotientGroup.mk (s := H) x)
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.refine_2
      G : Type u
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : ContinuousMul G
      H : Subgroup G
      inst✝ : H.FiniteIndex
      h✝ : IsClosed ↑H
      x : G
      h : Membership.mem (Set.iUnion fun i => HSMul.hSMul (Quotient.out ↑i) ↑H) x
      ⊢ Membership.mem (HasCompl.compl ↑H) x
    -/
  · rcases h with ⟨S,⟨y,hS⟩,mem⟩
    /-
      case h.e'_3.h.refine_2.intro.intro.intro
      G : Type u
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : ContinuousMul G
      H : Subgroup G
      inst✝ : H.FiniteIndex
      h : IsClosed ↑H
      x : G
      S : Set G
      mem : Membership.mem S x
      y : Subtype fun x => Ne x ↑1
      hS : Eq ((fun i => HSMul.hSMul (Quotient.out ↑i) ↑H) y) S
      ⊢ Membership.mem (HasCompl.compl ↑H) x
    -/
    simp only [← hS] at mem
    /-
      case h.e'_3.h.refine_2.intro.intro.intro
      G : Type u
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : ContinuousMul G
      H : Subgroup G
      inst✝ : H.FiniteIndex
      h : IsClosed ↑H
      x : G
      S : Set G
      y : Subtype fun x => Ne x ↑1
      hS : Eq ((fun i => HSMul.hSMul (Quotient.out ↑i) ↑H) y) S
      mem : Membership.mem (HSMul.hSMul (Quotient.out ↑y) ↑H) x
      ⊢ Membership.mem (HasCompl.compl ↑H) x
    -/
    rcases mem with ⟨h,hh,eq⟩
    /-
      case h.e'_3.h.refine_2.intro.intro.intro.intro.intro
      G : Type u
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : ContinuousMul G
      H : Subgroup G
      inst✝ : H.FiniteIndex
      h✝ : IsClosed ↑H
      x : G
      S : Set G
      y : Subtype fun x => Ne x ↑1
      hS : Eq ((fun i => HSMul.hSMul (Quotient.out ↑i) ↑H) y) S
      h : G
      hh : Membership.mem (↑H) h
      eq : Eq ((fun x => HSMul.hSMul (Quotient.out ↑y) x) h) x
      ⊢ Membership.mem (HasCompl.compl ↑H) x
    -/
    simp only [Set.mem_compl_iff, SetLike.mem_coe]
    /-
      case h.e'_3.h.refine_2.intro.intro.intro.intro.intro
      G : Type u
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : ContinuousMul G
      H : Subgroup G
      inst✝ : H.FiniteIndex
      h✝ : IsClosed ↑H
      x : G
      S : Set G
      y : Subtype fun x => Ne x ↑1
      hS : Eq ((fun i => HSMul.hSMul (Quotient.out ↑i) ↑H) y) S
      h : G
      hh : Membership.mem (↑H) h
      eq : Eq ((fun x => HSMul.hSMul (Quotient.out ↑y) x) h) x
      ⊢ Not (Membership.mem H x)
    -/
    by_contra mH
    /-
      case h.e'_3.h.refine_2.intro.intro.intro.intro.intro
      G : Type u
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : ContinuousMul G
      H : Subgroup G
      inst✝ : H.FiniteIndex
      h✝ : IsClosed ↑H
      x : G
      S : Set G
      y : Subtype fun x => Ne x ↑1
      hS : Eq ((fun i => HSMul.hSMul (Quotient.out ↑i) ↑H) y) S
      h : G
      hh : Membership.mem (↑H) h
      eq : Eq ((fun x => HSMul.hSMul (Quotient.out ↑y) x) h) x
      mH : Membership.mem H x
      ⊢ False
    -/
    simp only [← eq, ne_eq, smul_eq_mul] at mH
    /-
      case h.e'_3.h.refine_2.intro.intro.intro.intro.intro
      G : Type u
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : ContinuousMul G
      H : Subgroup G
      inst✝ : H.FiniteIndex
      h✝ : IsClosed ↑H
      x : G
      S : Set G
      y : Subtype fun x => Ne x ↑1
      hS : Eq ((fun i => HSMul.hSMul (Quotient.out ↑i) ↑H) y) S
      h : G
      hh : Membership.mem (↑H) h
      eq : Eq ((fun x => HSMul.hSMul (Quotient.out ↑y) x) h) x
      mH : Membership.mem H (HMul.hMul (Quotient.out ↑y) h)
      ⊢ False
    -/
    absurd y.2.symm
    /-
      case h.e'_3.h.refine_2.intro.intro.intro.intro.intro
      G : Type u
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : ContinuousMul G
      H : Subgroup G
      inst✝ : H.FiniteIndex
      h✝ : IsClosed ↑H
      x : G
      S : Set G
      y : Subtype fun x => Ne x ↑1
      hS : Eq ((fun i => HSMul.hSMul (Quotient.out ↑i) ↑H) y) S
      h : G
      hh : Membership.mem (↑H) h
      eq : Eq ((fun x => HSMul.hSMul (Quotient.out ↑y) x) h) x
      mH : Membership.mem H (HMul.hMul (Quotient.out ↑y) h)
      ⊢ Eq ↑1 ↑y
    -/
    rw [← QuotientGroup.out_eq' y.1, QuotientGroup.eq]
    /-
      case h.e'_3.h.refine_2.intro.intro.intro.intro.intro
      G : Type u
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : ContinuousMul G
      H : Subgroup G
      inst✝ : H.FiniteIndex
      h✝ : IsClosed ↑H
      x : G
      S : Set G
      y : Subtype fun x => Ne x ↑1
      hS : Eq ((fun i => HSMul.hSMul (Quotient.out ↑i) ↑H) y) S
      h : G
      hh : Membership.mem (↑H) h
      eq : Eq ((fun x => HSMul.hSMul (Quotient.out ↑y) x) h) x
      mH : Membership.mem H (HMul.hMul (Quotient.out ↑y) h)
      ⊢ Membership.mem H (HMul.hMul (Inv.inv 1) (Quotient.out ↑y))
    -/
    simp only [inv_one, ne_eq, one_mul, (Subgroup.mul_mem_cancel_right H hh).mp mH]
    /-
      🎉 no goals
    -/


