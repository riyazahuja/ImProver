/-- A pair of an `Order.Ideal` and an `Order.PFilter` which form a partition of `P`.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
structure PrimePair (P : Type*) [Preorder P] where
  I : Ideal P
  F : PFilter P
  isCompl_I_F : IsCompl (I : Set P) F


theorem compl_I_eq_F : (IF.I : Set P)ᶜ = IF.F :=
  IF.isCompl_I_F.compl_eq


theorem compl_F_eq_I : (IF.F : Set P)ᶜ = IF.I :=
  IF.isCompl_I_F.eq_compl.symm


theorem I_isProper : IsProper IF.I := by
  /-
    P : Type u_1
    inst✝ : Preorder P
    IF : Order.Ideal.PrimePair P
    ⊢ IF.I.IsProper
  -/
  cases' IF.F.nonempty with w h
  /-
    case intro
    P : Type u_1
    inst✝ : Preorder P
    IF : Order.Ideal.PrimePair P
    w : P
    h : Membership.mem (↑IF.F) w
    ⊢ IF.I.IsProper
  -/
  apply isProper_of_not_mem (_ : w ∉ IF.I)
  /-
    P : Type u_1
    inst✝ : Preorder P
    IF : Order.Ideal.PrimePair P
    w : P
    h : Membership.mem (↑IF.F) w
    ⊢ Not (Membership.mem IF.I w)
  -/
  rwa [← IF.compl_I_eq_F] at h
  /-
    🎉 no goals
  -/


protected theorem disjoint : Disjoint (IF.I : Set P) IF.F :=
  IF.isCompl_I_F.disjoint


theorem I_union_F : (IF.I : Set P) ∪ IF.F = Set.univ :=
  IF.isCompl_I_F.sup_eq_top


theorem F_union_I : (IF.F : Set P) ∪ IF.I = Set.univ :=
  IF.isCompl_I_F.symm.sup_eq_top


/-- An ideal `I` is prime if its complement is a filter.
-/
@[mk_iff]
class IsPrime [Preorder P] (I : Ideal P) extends IsProper I : Prop where
  compl_filter : IsPFilter (I : Set P)ᶜ


/-- Create an element of type `Order.Ideal.PrimePair` from an ideal satisfying the predicate
`Order.Ideal.IsPrime`. -/
def IsPrime.toPrimePair {I : Ideal P} (h : IsPrime I) : PrimePair P :=
  { I
    F := h.compl_filter.toPFilter
    isCompl_I_F := isCompl_compl }


theorem PrimePair.I_isPrime (IF : PrimePair P) : IsPrime IF.I :=
  { IF.I_isProper with
    compl_filter := by
      /-
        P : Type u_1
        inst✝ : Preorder P
        IF : Order.Ideal.PrimePair P
        ⊢ Order.IsPFilter (HasCompl.compl ↑IF.I)
      -/
      rw [IF.compl_I_eq_F]
      /-
        P : Type u_1
        inst✝ : Preorder P
        IF : Order.Ideal.PrimePair P
        ⊢ Order.IsPFilter ↑IF.F
      -/
      exact IF.F.isPFilter }
      /-
        🎉 no goals
      -/


theorem IsPrime.mem_or_mem (hI : IsPrime I) {x y : P} : x ⊓ y ∈ I → x ∈ I ∨ y ∈ I := by
  /-
    P : Type u_1
    inst✝ : SemilatticeInf P
    I : Order.Ideal P
    hI : I.IsPrime
    x y : P
    ⊢ Membership.mem I (Min.min x y) → Or (Membership.mem I x) (Membership.mem I y)
  -/
  contrapose!
  /-
    P : Type u_1
    inst✝ : SemilatticeInf P
    I : Order.Ideal P
    hI : I.IsPrime
    x y : P
    ⊢ And (Not (Membership.mem I x)) (Not (Membership.mem I y)) → Not (Membership. …
  -/
  let F := hI.compl_filter.toPFilter
  /-
    P : Type u_1
    inst✝ : SemilatticeInf P
    I : Order.Ideal P
    hI : I.IsPrime
    x y : P
    F : Order.PFilter P := ⋯.toPFilter
    ⊢ And (Not (Membership.mem I x)) (Not (Membership.mem I y)) → Not (Membership. …
  -/
  show x ∈ F ∧ y ∈ F → x ⊓ y ∈ F
  /-
    P : Type u_1
    inst✝ : SemilatticeInf P
    I : Order.Ideal P
    hI : I.IsPrime
    x y : P
    F : Order.PFilter P := ⋯.toPFilter
    ⊢ And (Membership.mem F x) (Membership.mem F y) → Membership.mem F (Min.min x y)
  -/
  exact fun h => inf_mem h.1 h.2
  /-
    🎉 no goals
  -/


theorem IsPrime.of_mem_or_mem [IsProper I] (hI : ∀ {x y : P}, x ⊓ y ∈ I → x ∈ I ∨ y ∈ I) :
    IsPrime I := by
  /-
    P : Type u_1
    inst✝¹ : SemilatticeInf P
    I : Order.Ideal P
    inst✝ : I.IsProper
    hI : ∀ {x y : P}, Membership.mem I (Min.min x y) → Or (Membership.mem I x) (Me …
    ⊢ I.IsPrime
  -/
  rw [isPrime_iff]
  /-
    P : Type u_1
    inst✝¹ : SemilatticeInf P
    I : Order.Ideal P
    inst✝ : I.IsProper
    hI : ∀ {x y : P}, Membership.mem I (Min.min x y) → Or (Membership.mem I x) (Me …
    ⊢ And I.IsProper (Order.IsPFilter (HasCompl.compl ↑I))
  -/
  use ‹_›
  /-
    case right
    P : Type u_1
    inst✝¹ : SemilatticeInf P
    I : Order.Ideal P
    inst✝ : I.IsProper
    hI : ∀ {x y : P}, Membership.mem I (Min.min x y) → Or (Membership.mem I x) (Me …
    ⊢ Order.IsPFilter (HasCompl.compl ↑I)
  -/
  refine .of_def ?_ ?_ ?_
    /-
      case right.refine_1
      P : Type u_1
      inst✝¹ : SemilatticeInf P
      I : Order.Ideal P
      inst✝ : I.IsProper
      hI : ∀ {x y : P}, Membership.mem I (Min.min x y) → Or (Membership.mem I x) (Me …
      ⊢ (HasCompl.compl ↑I).Nonempty
    -/
  · exact Set.nonempty_compl.2 (I.isProper_iff.1 ‹_›)
    /-
      🎉 no goals
    -/
    /-
      case right.refine_2
      P : Type u_1
      inst✝¹ : SemilatticeInf P
      I : Order.Ideal P
      inst✝ : I.IsProper
      hI : ∀ {x y : P}, Membership.mem I (Min.min x y) → Or (Membership.mem I x) (Me …
      ⊢ DirectedOn (fun x1 x2 => GE.ge x1 x2) (HasCompl.compl ↑I)
    -/
  · intro x hx y hy
    /-
      case right.refine_2
      P : Type u_1
      inst✝¹ : SemilatticeInf P
      I : Order.Ideal P
      inst✝ : I.IsProper
      hI : ∀ {x y : P}, Membership.mem I (Min.min x y) → Or (Membership.mem I x) (Me …
      x : P
      hx : Membership.mem (HasCompl.compl ↑I) x
      y : P
      hy : Membership.mem (HasCompl.compl ↑I) y
      ⊢ Exists fun z => And (Membership.mem (HasCompl.compl ↑I) z) (And ((fun x1 x2  …
    -/
    exact ⟨x ⊓ y, fun h => (hI h).elim hx hy, inf_le_left, inf_le_right⟩
    /-
      🎉 no goals
    -/
    /-
      case right.refine_3
      P : Type u_1
      inst✝¹ : SemilatticeInf P
      I : Order.Ideal P
      inst✝ : I.IsProper
      hI : ∀ {x y : P}, Membership.mem I (Min.min x y) → Or (Membership.mem I x) (Me …
      ⊢ ∀ {x y : P}, LE.le x y → Membership.mem (HasCompl.compl ↑I) x → Membership.m …
    -/
  · exact @mem_compl_of_ge _ _ _
    /-
      🎉 no goals
    -/


theorem isPrime_iff_mem_or_mem [IsProper I] : IsPrime I ↔ ∀ {x y : P}, x ⊓ y ∈ I → x ∈ I ∨ y ∈ I :=
  ⟨IsPrime.mem_or_mem, IsPrime.of_mem_or_mem⟩


instance (priority := 100) IsMaximal.isPrime [IsMaximal I] : IsPrime I := by
  /-
    P : Type u_1
    inst✝¹ : DistribLattice P
    I : Order.Ideal P
    inst✝ : I.IsMaximal
    ⊢ I.IsPrime
  -/
  rw [isPrime_iff_mem_or_mem]
  /-
    P : Type u_1
    inst✝¹ : DistribLattice P
    I : Order.Ideal P
    inst✝ : I.IsMaximal
    ⊢ ∀ {x y : P}, Membership.mem I (Min.min x y) → Or (Membership.mem I x) (Membe …
  -/
  intro x y
  /-
    P : Type u_1
    inst✝¹ : DistribLattice P
    I : Order.Ideal P
    inst✝ : I.IsMaximal
    x y : P
    ⊢ Membership.mem I (Min.min x y) → Or (Membership.mem I x) (Membership.mem I y)
  -/
  contrapose!
  /-
    P : Type u_1
    inst✝¹ : DistribLattice P
    I : Order.Ideal P
    inst✝ : I.IsMaximal
    x y : P
    ⊢ And (Not (Membership.mem I x)) (Not (Membership.mem I y)) → Not (Membership. …
  -/
  rintro ⟨hx, hynI⟩ hxy
  /-
    case intro
    P : Type u_1
    inst✝¹ : DistribLattice P
    I : Order.Ideal P
    inst✝ : I.IsMaximal
    x y : P
    hx : Not (Membership.mem I x)
    hynI : Not (Membership.mem I y)
    hxy : Membership.mem I (Min.min x y)
    ⊢ False
  -/
  apply hynI
  /-
    case intro
    P : Type u_1
    inst✝¹ : DistribLattice P
    I : Order.Ideal P
    inst✝ : I.IsMaximal
    x y : P
    hx : Not (Membership.mem I x)
    hynI : Not (Membership.mem I y)
    hxy : Membership.mem I (Min.min x y)
    ⊢ Membership.mem I y
  -/
  let J := I ⊔ principal x
  have hJuniv : (J : Set P) = Set.univ :=
    IsMaximal.maximal_proper (lt_sup_principal_of_not_mem ‹_›)
  /-
    case intro
    P : Type u_1
    inst✝¹ : DistribLattice P
    I : Order.Ideal P
    inst✝ : I.IsMaximal
    x y : P
    hx : Not (Membership.mem I x)
    hynI : Not (Membership.mem I y)
    hxy : Membership.mem I (Min.min x y)
    J : Order.Ideal P := Max.max I (Order.Ideal.principal x)
    hJuniv : Eq (↑J) Set.univ
    ⊢ Membership.mem I y
  -/
  have hyJ : y ∈ (J : Set P) := Set.eq_univ_iff_forall.mp hJuniv y
  /-
    case intro
    P : Type u_1
    inst✝¹ : DistribLattice P
    I : Order.Ideal P
    inst✝ : I.IsMaximal
    x y : P
    hx : Not (Membership.mem I x)
    hynI : Not (Membership.mem I y)
    hxy : Membership.mem I (Min.min x y)
    J : Order.Ideal P := Max.max I (Order.Ideal.principal x)
    hJuniv : Eq (↑J) Set.univ
    hyJ : Membership.mem (↑J) y
    ⊢ Membership.mem I y
  -/
  rw [coe_sup_eq] at hyJ
  /-
    case intro
    P : Type u_1
    inst✝¹ : DistribLattice P
    I : Order.Ideal P
    inst✝ : I.IsMaximal
    x y : P
    hx : Not (Membership.mem I x)
    hynI : Not (Membership.mem I y)
    hxy : Membership.mem I (Min.min x y)
    J : Order.Ideal P := Max.max I (Order.Ideal.principal x)
    hJuniv : Eq (↑J) Set.univ
    hyJ : Membership.mem (setOf fun x_1 => Exists fun i => And (Membership.mem I i …
    ⊢ Membership.mem I y
  -/
  rcases hyJ with ⟨a, ha, b, hb, hy⟩
  /-
    case intro.intro.intro.intro.intro
    P : Type u_1
    inst✝¹ : DistribLattice P
    I : Order.Ideal P
    inst✝ : I.IsMaximal
    x y : P
    hx : Not (Membership.mem I x)
    hynI : Not (Membership.mem I y)
    hxy : Membership.mem I (Min.min x y)
    J : Order.Ideal P := Max.max I (Order.Ideal.principal x)
    hJuniv : Eq (↑J) Set.univ
    a : P
    ha : Membership.mem I a
    b : P
    hb : Membership.mem (Order.Ideal.principal x) b
    hy : Eq y (Max.max a b)
    ⊢ Membership.mem I y
  -/
  rw [hy]
  /-
    case intro.intro.intro.intro.intro
    P : Type u_1
    inst✝¹ : DistribLattice P
    I : Order.Ideal P
    inst✝ : I.IsMaximal
    x y : P
    hx : Not (Membership.mem I x)
    hynI : Not (Membership.mem I y)
    hxy : Membership.mem I (Min.min x y)
    J : Order.Ideal P := Max.max I (Order.Ideal.principal x)
    hJuniv : Eq (↑J) Set.univ
    a : P
    ha : Membership.mem I a
    b : P
    hb : Membership.mem (Order.Ideal.principal x) b
    hy : Eq y (Max.max a b)
    ⊢ Membership.mem I (Max.max a b)
  -/
  refine sup_mem ha (I.lower (le_inf hb ?_) hxy)
  /-
    case intro.intro.intro.intro.intro
    P : Type u_1
    inst✝¹ : DistribLattice P
    I : Order.Ideal P
    inst✝ : I.IsMaximal
    x y : P
    hx : Not (Membership.mem I x)
    hynI : Not (Membership.mem I y)
    hxy : Membership.mem I (Min.min x y)
    J : Order.Ideal P := Max.max I (Order.Ideal.principal x)
    hJuniv : Eq (↑J) Set.univ
    a : P
    ha : Membership.mem I a
    b : P
    hb : Membership.mem (Order.Ideal.principal x) b
    hy : Eq y (Max.max a b)
    ⊢ LE.le b y
  -/
  rw [hy]
  /-
    case intro.intro.intro.intro.intro
    P : Type u_1
    inst✝¹ : DistribLattice P
    I : Order.Ideal P
    inst✝ : I.IsMaximal
    x y : P
    hx : Not (Membership.mem I x)
    hynI : Not (Membership.mem I y)
    hxy : Membership.mem I (Min.min x y)
    J : Order.Ideal P := Max.max I (Order.Ideal.principal x)
    hJuniv : Eq (↑J) Set.univ
    a : P
    ha : Membership.mem I a
    b : P
    hb : Membership.mem (Order.Ideal.principal x) b
    hy : Eq y (Max.max a b)
    ⊢ LE.le b (Max.max a b)
  -/
  exact le_sup_right
  /-
    🎉 no goals
  -/


theorem IsPrime.mem_or_compl_mem (hI : IsPrime I) : x ∈ I ∨ xᶜ ∈ I := by
  /-
    P : Type u_1
    inst✝ : BooleanAlgebra P
    x : P
    I : Order.Ideal P
    hI : I.IsPrime
    ⊢ Or (Membership.mem I x) (Membership.mem I (HasCompl.compl x))
  -/
  apply hI.mem_or_mem
  /-
    P : Type u_1
    inst✝ : BooleanAlgebra P
    x : P
    I : Order.Ideal P
    hI : I.IsPrime
    ⊢ Membership.mem I (Min.min x (HasCompl.compl x))
  -/
  rw [inf_compl_eq_bot]
  /-
    P : Type u_1
    inst✝ : BooleanAlgebra P
    x : P
    I : Order.Ideal P
    hI : I.IsPrime
    ⊢ Membership.mem I Bot.bot
  -/
  exact I.bot_mem
  /-
    🎉 no goals
  -/


theorem IsPrime.mem_compl_of_not_mem (hI : IsPrime I) (hxnI : x ∉ I) : xᶜ ∈ I :=
  hI.mem_or_compl_mem.resolve_left hxnI


theorem isPrime_of_mem_or_compl_mem [IsProper I] (h : ∀ {x : P}, x ∈ I ∨ xᶜ ∈ I) : IsPrime I := by
  /-
    P : Type u_1
    inst✝¹ : BooleanAlgebra P
    I : Order.Ideal P
    inst✝ : I.IsProper
    h : ∀ {x : P}, Or (Membership.mem I x) (Membership.mem I (HasCompl.compl x))
    ⊢ I.IsPrime
  -/
  simp only [isPrime_iff_mem_or_mem, or_iff_not_imp_left]
  /-
    P : Type u_1
    inst✝¹ : BooleanAlgebra P
    I : Order.Ideal P
    inst✝ : I.IsProper
    h : ∀ {x : P}, Or (Membership.mem I x) (Membership.mem I (HasCompl.compl x))
    ⊢ ∀ {x y : P}, Membership.mem I (Min.min x y) → Not (Membership.mem I x) → Mem …
  -/
  intro x y hxy hxI
  /-
    P : Type u_1
    inst✝¹ : BooleanAlgebra P
    I : Order.Ideal P
    inst✝ : I.IsProper
    h : ∀ {x : P}, Or (Membership.mem I x) (Membership.mem I (HasCompl.compl x))
    x y : P
    hxy : Membership.mem I (Min.min x y)
    hxI : Not (Membership.mem I x)
    ⊢ Membership.mem I y
  -/
  have hxcI : xᶜ ∈ I := h.resolve_left hxI
  /-
    P : Type u_1
    inst✝¹ : BooleanAlgebra P
    I : Order.Ideal P
    inst✝ : I.IsProper
    h : ∀ {x : P}, Or (Membership.mem I x) (Membership.mem I (HasCompl.compl x))
    x y : P
    hxy : Membership.mem I (Min.min x y)
    hxI : Not (Membership.mem I x)
    hxcI : Membership.mem I (HasCompl.compl x)
    ⊢ Membership.mem I y
  -/
  have ass : x ⊓ y ⊔ y ⊓ xᶜ ∈ I := sup_mem hxy (I.lower inf_le_right hxcI)
  /-
    P : Type u_1
    inst✝¹ : BooleanAlgebra P
    I : Order.Ideal P
    inst✝ : I.IsProper
    h : ∀ {x : P}, Or (Membership.mem I x) (Membership.mem I (HasCompl.compl x))
    x y : P
    hxy : Membership.mem I (Min.min x y)
    hxI : Not (Membership.mem I x)
    hxcI : Membership.mem I (HasCompl.compl x)
    ass : Membership.mem I (Max.max (Min.min x y) (Min.min y (HasCompl.compl x)))
    ⊢ Membership.mem I y
  -/
  rwa [inf_comm, sup_inf_inf_compl] at ass
  /-
    🎉 no goals
  -/


theorem isPrime_iff_mem_or_compl_mem [IsProper I] : IsPrime I ↔ ∀ {x : P}, x ∈ I ∨ xᶜ ∈ I :=
  ⟨fun h _ => h.mem_or_compl_mem, isPrime_of_mem_or_compl_mem⟩


instance (priority := 100) IsPrime.isMaximal [IsPrime I] : IsMaximal I := by
  /-
    P : Type u_1
    inst✝¹ : BooleanAlgebra P
    x : P
    I : Order.Ideal P
    inst✝ : I.IsPrime
    ⊢ I.IsMaximal
  -/
  simp only [isMaximal_iff, Set.eq_univ_iff_forall, IsPrime.toIsProper, true_and]
  /-
    P : Type u_1
    inst✝¹ : BooleanAlgebra P
    x : P
    I : Order.Ideal P
    inst✝ : I.IsPrime
    ⊢ ∀ ⦃J : Order.Ideal P⦄, LT.lt I J → ∀ (x : P), Membership.mem (↑J) x
  -/
  intro J hIJ x
  /-
    P : Type u_1
    inst✝¹ : BooleanAlgebra P
    x✝ : P
    I : Order.Ideal P
    inst✝ : I.IsPrime
    J : Order.Ideal P
    hIJ : LT.lt I J
    x : P
    ⊢ Membership.mem (↑J) x
  -/
  rcases Set.exists_of_ssubset hIJ with ⟨y, hyJ, hyI⟩
  /-
    case intro.intro
    P : Type u_1
    inst✝¹ : BooleanAlgebra P
    x✝ : P
    I : Order.Ideal P
    inst✝ : I.IsPrime
    J : Order.Ideal P
    hIJ : LT.lt I J
    x y : P
    hyJ : Membership.mem (↑J) y
    hyI : Not (Membership.mem (↑I) y)
    ⊢ Membership.mem (↑J) x
  -/
  suffices ass : x ⊓ y ⊔ x ⊓ yᶜ ∈ J by rwa [sup_inf_inf_compl] at ass
  exact
    sup_mem (J.lower inf_le_right hyJ)
      (hIJ.le <| I.lower inf_le_right <| IsPrime.mem_compl_of_not_mem ‹_› hyI)


/-- A filter `F` is prime if its complement is an ideal.
-/
@[mk_iff]
class IsPrime (F : PFilter P) : Prop where
  compl_ideal : IsIdeal (F : Set P)ᶜ


/-- Create an element of type `Order.Ideal.PrimePair` from a filter satisfying the predicate
`Order.PFilter.IsPrime`. -/
def IsPrime.toPrimePair {F : PFilter P} (h : IsPrime F) : Ideal.PrimePair P :=
  { I := h.compl_ideal.toIdeal
    F
    isCompl_I_F := isCompl_compl.symm }


theorem _root_.Order.Ideal.PrimePair.F_isPrime (IF : Ideal.PrimePair P) : IsPrime IF.F :=
  {
    compl_ideal := by
      /-
        P : Type u_1
        inst✝ : Preorder P
        IF : Order.Ideal.PrimePair P
        ⊢ Order.IsIdeal (HasCompl.compl ↑IF.F)
      -/
      rw [IF.compl_F_eq_I]
      /-
        P : Type u_1
        inst✝ : Preorder P
        IF : Order.Ideal.PrimePair P
        ⊢ Order.IsIdeal ↑IF.I
      -/
      exact IF.I.isIdeal }
      /-
        🎉 no goals
      -/


