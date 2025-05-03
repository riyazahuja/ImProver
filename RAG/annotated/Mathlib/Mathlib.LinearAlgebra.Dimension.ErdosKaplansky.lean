/-- Key lemma towards the Erdős-Kaplansky theorem from https://mathoverflow.net/a/168624 -/
theorem max_aleph0_card_le_rank_fun_nat : max ℵ₀ #K ≤ Module.rank K (ℕ → K) := by
  have aleph0_le : ℵ₀ ≤ Module.rank K (ℕ → K) := (rank_finsupp_self K ℕ).symm.trans_le
    (Finsupp.lcoeFun.rank_le_of_injective <| by exact DFunLike.coe_injective)
  /-
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    ⊢ LE.le (Max.max Cardinal.aleph0 (Cardinal.mk K)) (Module.rank K (Nat → K))
  -/
  refine max_le aleph0_le ?_
  /-
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    ⊢ LE.le (Cardinal.mk K) (Module.rank K (Nat → K))
  -/
  obtain card_K | card_K := le_or_lt #K ℵ₀
    /-
      case inl
      K : Type u
      inst✝ : DivisionRing K
      aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
      card_K : LE.le (Cardinal.mk K) Cardinal.aleph0
      ⊢ LE.le (Cardinal.mk K) (Module.rank K (Nat → K))
    -/
  · exact card_K.trans aleph0_le
    /-
      🎉 no goals
    -/
  /-
    case inr
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    ⊢ LE.le (Cardinal.mk K) (Module.rank K (Nat → K))
  -/
  by_contra!
  /-
    case inr
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ⊢ False
  -/
  obtain ⟨⟨ιK, bK⟩⟩ := Module.Free.exists_basis (R := K) (M := ℕ → K)
  /-
    case inr.intro.mk
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    ⊢ False
  -/
  let L := Subfield.closure (Set.range (fun i : ιK × ℕ ↦ bK i.1 i.2))
  have hLK : #L < #K := by
    refine (Subfield.cardinalMk_closure_le_max _).trans_lt
      (max_lt_iff.mpr ⟨mk_range_le.trans_lt ?_, card_K⟩)
    rwa [mk_prod, ← aleph0, lift_uzero, bK.mk_eq_rank'', mul_aleph0_eq aleph0_le]
  /-
    case inr.intro.mk
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    ⊢ False
  -/
  letI := Module.compHom K (RingHom.op L.subtype)
  /-
    case inr.intro.mk
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝ : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Module. …
    ⊢ False
  -/
  obtain ⟨⟨ιL, bL⟩⟩ := Module.Free.exists_basis (R := Lᵐᵒᵖ) (M := K)
  have card_ιL : ℵ₀ ≤ #ιL := by
    contrapose! hLK
    haveI := @Fintype.ofFinite _ (lt_aleph0_iff_finite.mp hLK)
    rw [bL.repr.toEquiv.cardinal_eq, mk_finsupp_of_fintype,
        ← MulOpposite.opEquiv.cardinal_eq] at card_K ⊢
    apply power_nat_le
    contrapose! card_K
    exact (power_lt_aleph0 card_K <| nat_lt_aleph0 _).le
  /-
    case inr.intro.mk.intro.mk
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝ : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Module. …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    ⊢ False
  -/
  obtain ⟨e⟩ := lift_mk_le'.mp (card_ιL.trans_eq (lift_uzero #ιL).symm)
  /-
    case inr.intro.mk.intro.mk.intro
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝ : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Module. …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    e : Function.Embedding Nat ιL
    ⊢ False
  -/
  have rep_e := bK.linearCombination_repr (bL ∘ e)
  /-
    case inr.intro.mk.intro.mk.intro
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝ : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Module. …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    e : Function.Embedding Nat ιL
    rep_e : Eq ((Finsupp.linearCombination K ⇑bK) (bK.repr (Function.comp ⇑bL ⇑e)) …
    ⊢ False
  -/
  rw [Finsupp.linearCombination_apply, Finsupp.sum] at rep_e
  /-
    case inr.intro.mk.intro.mk.intro
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝ : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Module. …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    e : Function.Embedding Nat ιL
    rep_e : Eq ((bK.repr (Function.comp ⇑bL ⇑e)).support.sum fun a => HSMul.hSMul  …
    ⊢ False
  -/
  set c := bK.repr (bL ∘ e)
  /-
    case inr.intro.mk.intro.mk.intro
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝ : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Module. …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    e : Function.Embedding Nat ιL
    c : Finsupp ιK K := bK.repr (Function.comp ⇑bL ⇑e)
    rep_e : Eq (c.support.sum fun a => HSMul.hSMul (c a) (bK a)) (Function.comp ⇑b …
    ⊢ False
  -/
  set s := c.support
  /-
    case inr.intro.mk.intro.mk.intro
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝ : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Module. …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    e : Function.Embedding Nat ιL
    c : Finsupp ιK K := bK.repr (Function.comp ⇑bL ⇑e)
    s : Finset ιK := c.support
    rep_e : Eq (s.sum fun a => HSMul.hSMul (c a) (bK a)) (Function.comp ⇑bL ⇑e)
    ⊢ False
  -/
  let f i (j : s) : L := ⟨bK j i, Subfield.subset_closure ⟨(j, i), rfl⟩⟩
  have : ¬LinearIndependent Lᵐᵒᵖ f := fun h ↦ by
    have := h.cardinal_lift_le_rank
    rw [lift_uzero, (LinearEquiv.piCongrRight fun _ ↦ MulOpposite.opLinearEquiv Lᵐᵒᵖ).rank_eq,
        rank_fun'] at this
    exact (nat_lt_aleph0 _).not_le this
  /-
    case inr.intro.mk.intro.mk.intro
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝¹ : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this✝ : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Module …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    e : Function.Embedding Nat ιL
    c : Finsupp ιK K := bK.repr (Function.comp ⇑bL ⇑e)
    s : Finset ιK := c.support
    rep_e : Eq (s.sum fun a => HSMul.hSMul (c a) (bK a)) (Function.comp ⇑bL ⇑e)
    f : Nat → (Subtype fun x => Membership.mem s x) → Subtype fun x => Membership. …
    this : Not (LinearIndependent (MulOpposite (Subtype fun x => Membership.mem L  …
    ⊢ False
  -/
  obtain ⟨t, g, eq0, i, hi, hgi⟩ := not_linearIndependent_iff.mp this
  /-
    case inr.intro.mk.intro.mk.intro.intro.intro.intro.intro.intro
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝¹ : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this✝ : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Module …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    e : Function.Embedding Nat ιL
    c : Finsupp ιK K := bK.repr (Function.comp ⇑bL ⇑e)
    s : Finset ιK := c.support
    rep_e : Eq (s.sum fun a => HSMul.hSMul (c a) (bK a)) (Function.comp ⇑bL ⇑e)
    f : Nat → (Subtype fun x => Membership.mem s x) → Subtype fun x => Membership. …
    this : Not (LinearIndependent (MulOpposite (Subtype fun x => Membership.mem L  …
    t : Finset Nat
    g : Nat → MulOpposite (Subtype fun x => Membership.mem L x)
    eq0 : Eq (t.sum fun i => HSMul.hSMul (g i) (f i)) 0
    i : Nat
    hi : Membership.mem t i
    hgi : Ne (g i) 0
    ⊢ False
  -/
  refine hgi (linearIndependent_iff'.mp (bL.linearIndependent.comp e e.injective) t g ?_ i hi)
  /-
    case inr.intro.mk.intro.mk.intro.intro.intro.intro.intro.intro
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝¹ : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this✝ : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Module …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    e : Function.Embedding Nat ιL
    c : Finsupp ιK K := bK.repr (Function.comp ⇑bL ⇑e)
    s : Finset ιK := c.support
    rep_e : Eq (s.sum fun a => HSMul.hSMul (c a) (bK a)) (Function.comp ⇑bL ⇑e)
    f : Nat → (Subtype fun x => Membership.mem s x) → Subtype fun x => Membership. …
    this : Not (LinearIndependent (MulOpposite (Subtype fun x => Membership.mem L  …
    t : Finset Nat
    g : Nat → MulOpposite (Subtype fun x => Membership.mem L x)
    eq0 : Eq (t.sum fun i => HSMul.hSMul (g i) (f i)) 0
    i : Nat
    hi : Membership.mem t i
    hgi : Ne (g i) 0
    ⊢ Eq (t.sum fun i => HSMul.hSMul (g i) (Function.comp (⇑bL) (⇑e) i)) 0
  -/
  clear_value c s
  /-
    case inr.intro.mk.intro.mk.intro.intro.intro.intro.intro.intro
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝¹ : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this✝ : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Module …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    e : Function.Embedding Nat ιL
    t : Finset Nat
    g : Nat → MulOpposite (Subtype fun x => Membership.mem L x)
    i : Nat
    hi : Membership.mem t i
    hgi : Ne (g i) 0
    s : Finset ιK
    f : Nat → (Subtype fun x => Membership.mem s x) → Subtype fun x => Membership. …
    this : Not (LinearIndependent (MulOpposite (Subtype fun x => Membership.mem L  …
    eq0 : Eq (t.sum fun i => HSMul.hSMul (g i) (f i)) 0
    c : Finsupp ιK K
    rep_e : Eq (s.sum fun a => HSMul.hSMul (c a) (bK a)) (Function.comp ⇑bL ⇑e)
    ⊢ Eq (t.sum fun i => HSMul.hSMul (g i) (Function.comp (⇑bL) (⇑e) i)) 0
  -/
  simp_rw [← rep_e, Finset.sum_apply, Pi.smul_apply, Finset.smul_sum]
  /-
    case inr.intro.mk.intro.mk.intro.intro.intro.intro.intro.intro
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝¹ : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this✝ : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Module …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    e : Function.Embedding Nat ιL
    t : Finset Nat
    g : Nat → MulOpposite (Subtype fun x => Membership.mem L x)
    i : Nat
    hi : Membership.mem t i
    hgi : Ne (g i) 0
    s : Finset ιK
    f : Nat → (Subtype fun x => Membership.mem s x) → Subtype fun x => Membership. …
    this : Not (LinearIndependent (MulOpposite (Subtype fun x => Membership.mem L  …
    eq0 : Eq (t.sum fun i => HSMul.hSMul (g i) (f i)) 0
    c : Finsupp ιK K
    rep_e : Eq (s.sum fun a => HSMul.hSMul (c a) (bK a)) (Function.comp ⇑bL ⇑e)
    ⊢ Eq (t.sum fun x => s.sum fun x_1 => HSMul.hSMul (g x) (HSMul.hSMul (c x_1) ( …
  -/
  rw [Finset.sum_comm]
  /-
    case inr.intro.mk.intro.mk.intro.intro.intro.intro.intro.intro
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝¹ : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this✝ : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Module …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    e : Function.Embedding Nat ιL
    t : Finset Nat
    g : Nat → MulOpposite (Subtype fun x => Membership.mem L x)
    i : Nat
    hi : Membership.mem t i
    hgi : Ne (g i) 0
    s : Finset ιK
    f : Nat → (Subtype fun x => Membership.mem s x) → Subtype fun x => Membership. …
    this : Not (LinearIndependent (MulOpposite (Subtype fun x => Membership.mem L  …
    eq0 : Eq (t.sum fun i => HSMul.hSMul (g i) (f i)) 0
    c : Finsupp ιK K
    rep_e : Eq (s.sum fun a => HSMul.hSMul (c a) (bK a)) (Function.comp ⇑bL ⇑e)
    ⊢ Eq (s.sum fun y => t.sum fun x => HSMul.hSMul (g x) (HSMul.hSMul (c y) (bK y …
  -/
  refine Finset.sum_eq_zero fun i hi ↦ ?_
  /-
    case inr.intro.mk.intro.mk.intro.intro.intro.intro.intro.intro
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝¹ : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this✝ : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Module …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    e : Function.Embedding Nat ιL
    t : Finset Nat
    g : Nat → MulOpposite (Subtype fun x => Membership.mem L x)
    i✝ : Nat
    hi✝ : Membership.mem t i✝
    hgi : Ne (g i✝) 0
    s : Finset ιK
    f : Nat → (Subtype fun x => Membership.mem s x) → Subtype fun x => Membership. …
    this : Not (LinearIndependent (MulOpposite (Subtype fun x => Membership.mem L  …
    eq0 : Eq (t.sum fun i => HSMul.hSMul (g i) (f i)) 0
    c : Finsupp ιK K
    rep_e : Eq (s.sum fun a => HSMul.hSMul (c a) (bK a)) (Function.comp ⇑bL ⇑e)
    i : ιK
    hi : Membership.mem s i
    ⊢ Eq (t.sum fun x => HSMul.hSMul (g x) (HSMul.hSMul (c i) (bK i x))) 0
  -/
  replace eq0 := congr_arg L.subtype (congr_fun eq0 ⟨i, hi⟩)
  /-
    case inr.intro.mk.intro.mk.intro.intro.intro.intro.intro.intro
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝¹ : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this✝ : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Module …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    e : Function.Embedding Nat ιL
    t : Finset Nat
    g : Nat → MulOpposite (Subtype fun x => Membership.mem L x)
    i✝ : Nat
    hi✝ : Membership.mem t i✝
    hgi : Ne (g i✝) 0
    s : Finset ιK
    f : Nat → (Subtype fun x => Membership.mem s x) → Subtype fun x => Membership. …
    this : Not (LinearIndependent (MulOpposite (Subtype fun x => Membership.mem L  …
    c : Finsupp ιK K
    rep_e : Eq (s.sum fun a => HSMul.hSMul (c a) (bK a)) (Function.comp ⇑bL ⇑e)
    i : ιK
    hi : Membership.mem s i
    eq0 : Eq (L.subtype (t.sum (fun i => HSMul.hSMul (g i) (f i)) ⟨i, hi⟩)) (L.sub …
    ⊢ Eq (t.sum fun x => HSMul.hSMul (g x) (HSMul.hSMul (c i) (bK i x))) 0
  -/
  rw [Finset.sum_apply, map_sum] at eq0
  /-
    case inr.intro.mk.intro.mk.intro.intro.intro.intro.intro.intro
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝¹ : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this✝ : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Module …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    e : Function.Embedding Nat ιL
    t : Finset Nat
    g : Nat → MulOpposite (Subtype fun x => Membership.mem L x)
    i✝ : Nat
    hi✝ : Membership.mem t i✝
    hgi : Ne (g i✝) 0
    s : Finset ιK
    f : Nat → (Subtype fun x => Membership.mem s x) → Subtype fun x => Membership. …
    this : Not (LinearIndependent (MulOpposite (Subtype fun x => Membership.mem L  …
    c : Finsupp ιK K
    rep_e : Eq (s.sum fun a => HSMul.hSMul (c a) (bK a)) (Function.comp ⇑bL ⇑e)
    i : ιK
    hi : Membership.mem s i
    eq0 : Eq (t.sum fun x => L.subtype (HSMul.hSMul (g x) (f x) ⟨i, hi⟩)) (L.subty …
    ⊢ Eq (t.sum fun x => HSMul.hSMul (g x) (HSMul.hSMul (c i) (bK i x))) 0
  -/
  have : SMulCommClass Lᵐᵒᵖ K K := ⟨fun _ _ _ ↦ mul_assoc _ _ _⟩
  /-
    case inr.intro.mk.intro.mk.intro.intro.intro.intro.intro.intro
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝² : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this✝¹ : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Modul …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    e : Function.Embedding Nat ιL
    t : Finset Nat
    g : Nat → MulOpposite (Subtype fun x => Membership.mem L x)
    i✝ : Nat
    hi✝ : Membership.mem t i✝
    hgi : Ne (g i✝) 0
    s : Finset ιK
    f : Nat → (Subtype fun x => Membership.mem s x) → Subtype fun x => Membership. …
    this✝ : Not (LinearIndependent (MulOpposite (Subtype fun x => Membership.mem L …
    c : Finsupp ιK K
    rep_e : Eq (s.sum fun a => HSMul.hSMul (c a) (bK a)) (Function.comp ⇑bL ⇑e)
    i : ιK
    hi : Membership.mem s i
    eq0 : Eq (t.sum fun x => L.subtype (HSMul.hSMul (g x) (f x) ⟨i, hi⟩)) (L.subty …
    this : SMulCommClass (MulOpposite (Subtype fun x => Membership.mem L x)) K K
    ⊢ Eq (t.sum fun x => HSMul.hSMul (g x) (HSMul.hSMul (c i) (bK i x))) 0
  -/
  simp_rw [smul_comm _ (c i), ← Finset.smul_sum]
  /-
    case inr.intro.mk.intro.mk.intro.intro.intro.intro.intro.intro
    K : Type u
    inst✝ : DivisionRing K
    aleph0_le : LE.le Cardinal.aleph0 (Module.rank K (Nat → K))
    card_K : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    this✝² : LT.lt (Module.rank K (Nat → K)) (Cardinal.mk K)
    ιK : Type u
    bK : Basis ιK K (Nat → K)
    L : Subfield K := Subfield.closure (Set.range fun i => bK i.1 i.2)
    hLK : LT.lt (Cardinal.mk (Subtype fun x => Membership.mem L x)) (Cardinal.mk K)
    this✝¹ : Module (MulOpposite (Subtype fun x => Membership.mem L x)) K := Modul …
    ιL : Type u
    bL : Basis ιL (MulOpposite (Subtype fun x => Membership.mem L x)) K
    card_ιL : LE.le Cardinal.aleph0 (Cardinal.mk ιL)
    e : Function.Embedding Nat ιL
    t : Finset Nat
    g : Nat → MulOpposite (Subtype fun x => Membership.mem L x)
    i✝ : Nat
    hi✝ : Membership.mem t i✝
    hgi : Ne (g i✝) 0
    s : Finset ιK
    f : Nat → (Subtype fun x => Membership.mem s x) → Subtype fun x => Membership. …
    this✝ : Not (LinearIndependent (MulOpposite (Subtype fun x => Membership.mem L …
    c : Finsupp ιK K
    rep_e : Eq (s.sum fun a => HSMul.hSMul (c a) (bK a)) (Function.comp ⇑bL ⇑e)
    i : ιK
    hi : Membership.mem s i
    eq0 : Eq (t.sum fun x => L.subtype (HSMul.hSMul (g x) (f x) ⟨i, hi⟩)) (L.subty …
    this : SMulCommClass (MulOpposite (Subtype fun x => Membership.mem L x)) K K
    ⊢ Eq (HSMul.hSMul (c i) (t.sum fun x => HSMul.hSMul (g x) (bK i x))) 0
  -/
  erw [eq0, smul_zero]
  /-
    🎉 no goals
  -/


open Function in
theorem rank_fun_infinite {ι : Type v} [hι : Infinite ι] : Module.rank K (ι → K) = #(ι → K) := by
  /-
    K : Type u
    inst✝ : DivisionRing K
    ι : Type v
    hι : Infinite ι
    ⊢ Eq (Module.rank K (ι → K)) (Cardinal.mk (ι → K))
  -/
  obtain ⟨⟨ιK, bK⟩⟩ := Module.Free.exists_basis (R := K) (M := ι → K)
  /-
    case intro.mk
    K : Type u
    inst✝ : DivisionRing K
    ι : Type v
    hι : Infinite ι
    ιK : Type (max u v)
    bK : Basis ιK K (ι → K)
    ⊢ Eq (Module.rank K (ι → K)) (Cardinal.mk (ι → K))
  -/
  obtain ⟨e⟩ := lift_mk_le'.mp ((aleph0_le_mk_iff.mpr hι).trans_eq (lift_uzero #ι).symm)
  have := LinearMap.lift_rank_le_of_injective _ <|
    LinearMap.funLeft_injective_of_surjective K K _ (invFun_surjective e.injective)
  /-
    case intro.mk.intro
    K : Type u
    inst✝ : DivisionRing K
    ι : Type v
    hι : Infinite ι
    ιK : Type (max u v)
    bK : Basis ιK K (ι → K)
    e : Function.Embedding Nat ι
    this : LE.le (Cardinal.lift.{max u v, u} (Module.rank K (Nat → K))) (Cardinal. …
    ⊢ Eq (Module.rank K (ι → K)) (Cardinal.mk (ι → K))
  -/
  rw [lift_umax.{u,v}, lift_id'.{u,v}] at this
  /-
    case intro.mk.intro
    K : Type u
    inst✝ : DivisionRing K
    ι : Type v
    hι : Infinite ι
    ιK : Type (max u v)
    bK : Basis ιK K (ι → K)
    e : Function.Embedding Nat ι
    this : LE.le (Cardinal.lift.{v, u} (Module.rank K (Nat → K))) (Module.rank K ( …
    ⊢ Eq (Module.rank K (ι → K)) (Cardinal.mk (ι → K))
  -/
  have key := (lift_le.{v}.mpr <| max_aleph0_card_le_rank_fun_nat K).trans this
  /-
    case intro.mk.intro
    K : Type u
    inst✝ : DivisionRing K
    ι : Type v
    hι : Infinite ι
    ιK : Type (max u v)
    bK : Basis ιK K (ι → K)
    e : Function.Embedding Nat ι
    this : LE.le (Cardinal.lift.{v, u} (Module.rank K (Nat → K))) (Module.rank K ( …
    key : LE.le (Cardinal.lift.{v, u} (Max.max Cardinal.aleph0 (Cardinal.mk K))) ( …
    ⊢ Eq (Module.rank K (ι → K)) (Cardinal.mk (ι → K))
  -/
  rw [lift_max, lift_aleph0, max_le_iff] at key
  haveI : Infinite ιK := by
    rw [← aleph0_le_mk_iff, bK.mk_eq_rank'']; exact key.1
  rw [bK.repr.toEquiv.cardinal_eq, mk_finsupp_lift_of_infinite,
      lift_umax.{u,v}, lift_id'.{u,v}, bK.mk_eq_rank'', eq_comm, max_eq_left]
  /-
    case intro.mk.intro
    K : Type u
    inst✝ : DivisionRing K
    ι : Type v
    hι : Infinite ι
    ιK : Type (max u v)
    bK : Basis ιK K (ι → K)
    e : Function.Embedding Nat ι
    this✝ : LE.le (Cardinal.lift.{v, u} (Module.rank K (Nat → K))) (Module.rank K  …
    key : And (LE.le Cardinal.aleph0 (Module.rank K (ι → K))) (LE.le (Cardinal.lif …
    this : Infinite ιK
    ⊢ LE.le (Cardinal.lift.{v, u} (Cardinal.mk K)) (Module.rank K (ι → K))
  -/
  exact key.2
  /-
    🎉 no goals
  -/


/-- The **Erdős-Kaplansky Theorem**: the dual of an infinite-dimensional vector space
  over a division ring has dimension equal to its cardinality. -/
theorem rank_dual_eq_card_dual_of_aleph0_le_rank' {V : Type*} [AddCommGroup V] [Module K V]
    (h : ℵ₀ ≤ Module.rank K V) : Module.rank Kᵐᵒᵖ (V →ₗ[K] K) = #(V →ₗ[K] K) := by
  /-
    K : Type u
    inst✝² : DivisionRing K
    V : Type u_1
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    h : LE.le Cardinal.aleph0 (Module.rank K V)
    ⊢ Eq (Module.rank (MulOpposite K) (LinearMap (RingHom.id K) V K)) (Cardinal.mk …
  -/
  obtain ⟨⟨ι, b⟩⟩ := Module.Free.exists_basis (R := K) (M := V)
  /-
    case intro.mk
    K : Type u
    inst✝² : DivisionRing K
    V : Type u_1
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    h : LE.le Cardinal.aleph0 (Module.rank K V)
    ι : Type u_1
    b : Basis ι K V
    ⊢ Eq (Module.rank (MulOpposite K) (LinearMap (RingHom.id K) V K)) (Cardinal.mk …
  -/
  rw [← b.mk_eq_rank'', aleph0_le_mk_iff] at h
  have e := (b.constr Kᵐᵒᵖ (M' := K)).symm.trans
    (LinearEquiv.piCongrRight fun _ ↦ MulOpposite.opLinearEquiv Kᵐᵒᵖ)
  /-
    case intro.mk
    K : Type u
    inst✝² : DivisionRing K
    V : Type u_1
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    ι : Type u_1
    h : Infinite ι
    b : Basis ι K V
    e : LinearEquiv (RingHom.id (MulOpposite K)) (LinearMap (RingHom.id K) V K) (ι …
    ⊢ Eq (Module.rank (MulOpposite K) (LinearMap (RingHom.id K) V K)) (Cardinal.mk …
  -/
  rw [e.rank_eq, e.toEquiv.cardinal_eq]
  /-
    case intro.mk
    K : Type u
    inst✝² : DivisionRing K
    V : Type u_1
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    ι : Type u_1
    h : Infinite ι
    b : Basis ι K V
    e : LinearEquiv (RingHom.id (MulOpposite K)) (LinearMap (RingHom.id K) V K) (ι …
    ⊢ Eq (Module.rank (MulOpposite K) (ι → MulOpposite K)) (Cardinal.mk (ι → MulOp …
  -/
  apply rank_fun_infinite
  /-
    🎉 no goals
  -/


/-- The **Erdős-Kaplansky Theorem** over a field. -/
theorem rank_dual_eq_card_dual_of_aleph0_le_rank {K V} [Field K] [AddCommGroup V] [Module K V]
    (h : ℵ₀ ≤ Module.rank K V) : Module.rank K (V →ₗ[K] K) = #(V →ₗ[K] K) := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    h : LE.le Cardinal.aleph0 (Module.rank K V)
    ⊢ Eq (Module.rank K (LinearMap (RingHom.id K) V K)) (Cardinal.mk (LinearMap (R …
  -/
  obtain ⟨⟨ι, b⟩⟩ := Module.Free.exists_basis (R := K) (M := V)
  /-
    case intro.mk
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    h : LE.le Cardinal.aleph0 (Module.rank K V)
    ι : Type u_2
    b : Basis ι K V
    ⊢ Eq (Module.rank K (LinearMap (RingHom.id K) V K)) (Cardinal.mk (LinearMap (R …
  -/
  rw [← b.mk_eq_rank'', aleph0_le_mk_iff] at h
  /-
    case intro.mk
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    ι : Type u_2
    h : Infinite ι
    b : Basis ι K V
    ⊢ Eq (Module.rank K (LinearMap (RingHom.id K) V K)) (Cardinal.mk (LinearMap (R …
  -/
  have e := (b.constr K (M' := K)).symm
  /-
    case intro.mk
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    ι : Type u_2
    h : Infinite ι
    b : Basis ι K V
    e : LinearEquiv (RingHom.id K) (LinearMap (RingHom.id K) V K) (ι → K)
    ⊢ Eq (Module.rank K (LinearMap (RingHom.id K) V K)) (Cardinal.mk (LinearMap (R …
  -/
  rw [e.rank_eq, e.toEquiv.cardinal_eq]
  /-
    case intro.mk
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    ι : Type u_2
    h : Infinite ι
    b : Basis ι K V
    e : LinearEquiv (RingHom.id K) (LinearMap (RingHom.id K) V K) (ι → K)
    ⊢ Eq (Module.rank K (ι → K)) (Cardinal.mk (ι → K))
  -/
  apply rank_fun_infinite
  /-
    🎉 no goals
  -/


theorem lift_rank_lt_rank_dual' {V : Type v} [AddCommGroup V] [Module K V]
    (h : ℵ₀ ≤ Module.rank K V) :
    Cardinal.lift.{u} (Module.rank K V) < Module.rank Kᵐᵒᵖ (V →ₗ[K] K) := by
  /-
    K : Type u
    inst✝² : DivisionRing K
    V : Type v
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    h : LE.le Cardinal.aleph0 (Module.rank K V)
    ⊢ LT.lt (Cardinal.lift.{u, v} (Module.rank K V)) (Module.rank (MulOpposite K)  …
  -/
  obtain ⟨⟨ι, b⟩⟩ := Module.Free.exists_basis (R := K) (M := V)
  rw [← b.mk_eq_rank'', rank_dual_eq_card_dual_of_aleph0_le_rank' h,
      ← (b.constr ℕ (M' := K)).toEquiv.cardinal_eq, mk_arrow]
  /-
    case intro.mk
    K : Type u
    inst✝² : DivisionRing K
    V : Type v
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    h : LE.le Cardinal.aleph0 (Module.rank K V)
    ι : Type v
    b : Basis ι K V
    ⊢ LT.lt (Cardinal.lift.{u, v} (Cardinal.mk ι)) (HPow.hPow (Cardinal.lift.{v, u …
  -/
  apply cantor'
  /-
    case intro.mk.hb
    K : Type u
    inst✝² : DivisionRing K
    V : Type v
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    h : LE.le Cardinal.aleph0 (Module.rank K V)
    ι : Type v
    b : Basis ι K V
    ⊢ LT.lt 1 (Cardinal.lift.{v, u} (Cardinal.mk K))
  -/
  erw [nat_lt_lift_iff, one_lt_iff_nontrivial]
  /-
    case intro.mk.hb
    K : Type u
    inst✝² : DivisionRing K
    V : Type v
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    h : LE.le Cardinal.aleph0 (Module.rank K V)
    ι : Type v
    b : Basis ι K V
    ⊢ Nontrivial K
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem lift_rank_lt_rank_dual {K : Type u} {V : Type v} [Field K] [AddCommGroup V] [Module K V]
    (h : ℵ₀ ≤ Module.rank K V) :
    Cardinal.lift.{u} (Module.rank K V) < Module.rank K (V →ₗ[K] K) := by
  /-
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    h : LE.le Cardinal.aleph0 (Module.rank K V)
    ⊢ LT.lt (Cardinal.lift.{u, v} (Module.rank K V)) (Module.rank K (LinearMap (Ri …
  -/
  rw [rank_dual_eq_card_dual_of_aleph0_le_rank h, ← rank_dual_eq_card_dual_of_aleph0_le_rank' h]
  /-
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    h : LE.le Cardinal.aleph0 (Module.rank K V)
    ⊢ LT.lt (Cardinal.lift.{u, v} (Module.rank K V)) (Module.rank (MulOpposite K)  …
  -/
  exact lift_rank_lt_rank_dual' h
  /-
    🎉 no goals
  -/


theorem rank_lt_rank_dual' {V : Type u} [AddCommGroup V] [Module K V] (h : ℵ₀ ≤ Module.rank K V) :
    Module.rank K V < Module.rank Kᵐᵒᵖ (V →ₗ[K] K) := by
  /-
    K : Type u
    inst✝² : DivisionRing K
    V : Type u
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    h : LE.le Cardinal.aleph0 (Module.rank K V)
    ⊢ LT.lt (Module.rank K V) (Module.rank (MulOpposite K) (LinearMap (RingHom.id  …
  -/
  convert lift_rank_lt_rank_dual' h; rw [lift_id]
                                     /-
                                       🎉 no goals
                                     -/


theorem rank_lt_rank_dual {K V : Type u} [Field K] [AddCommGroup V] [Module K V]
    (h : ℵ₀ ≤ Module.rank K V) : Module.rank K V < Module.rank K (V →ₗ[K] K) := by
  /-
    K V : Type u
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    h : LE.le Cardinal.aleph0 (Module.rank K V)
    ⊢ LT.lt (Module.rank K V) (Module.rank K (LinearMap (RingHom.id K) V K))
  -/
  convert lift_rank_lt_rank_dual h; rw [lift_id]
                                    /-
                                      🎉 no goals
                                    -/


