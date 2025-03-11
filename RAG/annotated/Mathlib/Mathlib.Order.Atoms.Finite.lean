instance (priority := 200) [DecidableEq α] : Fintype α :=
  Fintype.ofEquiv Bool equivBool.symm


                                            /-
                                              α : Type u_1
                                              β : Type u_2
                                              inst✝² : LE α
                                              inst✝¹ : BoundedOrder α
                                              inst✝ : IsSimpleOrder α
                                              ⊢ Finite α
                                            -/
instance (priority := 200) : Finite α := by classical infer_instance
                                            /-
                                              🎉 no goals
                                            -/


theorem univ : (Finset.univ : Finset α) = {⊤, ⊥} := by
  /-
    α : Type u_1
    inst✝³ : LE α
    inst✝² : BoundedOrder α
    inst✝¹ : IsSimpleOrder α
    inst✝ : DecidableEq α
    ⊢ Eq Finset.univ (Insert.insert Top.top (Singleton.singleton Bot.bot))
  -/
  change Finset.map _ (Finset.univ : Finset Bool) = _
  /-
    α : Type u_1
    inst✝³ : LE α
    inst✝² : BoundedOrder α
    inst✝¹ : IsSimpleOrder α
    inst✝ : DecidableEq α
    ⊢ Eq (Finset.map { toFun := ⇑IsSimpleOrder.equivBool.symm, inj' := ⋯ } Finset. …
  -/
  rw [Fintype.univ_bool]
  /-
    α : Type u_1
    inst✝³ : LE α
    inst✝² : BoundedOrder α
    inst✝¹ : IsSimpleOrder α
    inst✝ : DecidableEq α
    ⊢ Eq (Finset.map { toFun := ⇑IsSimpleOrder.equivBool.symm, inj' := ⋯ } (Insert …
  -/
  simp only [Finset.map_insert, Function.Embedding.coeFn_mk, Finset.map_singleton]
  /-
    α : Type u_1
    inst✝³ : LE α
    inst✝² : BoundedOrder α
    inst✝¹ : IsSimpleOrder α
    inst✝ : DecidableEq α
    ⊢ Eq (Insert.insert (IsSimpleOrder.equivBool.symm Bool.true) (Singleton.single …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem card : Fintype.card α = 2 :=
  (Fintype.ofEquiv_card _).trans Fintype.card_bool


instance : IsSimpleOrder Bool :=
  ⟨fun a => by
    rw [← Finset.mem_singleton, Or.comm, ← Finset.mem_insert, top_eq_true, bot_eq_false, ←
      Fintype.univ_bool]
    /-
      α : Type u_1
      β : Type u_2
      a : Bool
      ⊢ Membership.mem Finset.univ a
    -/
    apply Finset.mem_univ⟩
    /-
      🎉 no goals
    -/


instance (priority := 100) Finite.to_isCoatomic [PartialOrder α] [OrderTop α] [Finite α] :
    IsCoatomic α := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : OrderTop α
    inst✝ : Finite α
    ⊢ IsCoatomic α
  -/
  refine IsCoatomic.mk fun b => or_iff_not_imp_left.2 fun ht => ?_
  obtain ⟨c, hc, hmax⟩ :=
    Set.Finite.exists_maximal_wrt id { x : α | b ≤ x ∧ x ≠ ⊤ } (Set.toFinite _) ⟨b, le_rfl, ht⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : OrderTop α
    inst✝ : Finite α
    b : α
    ht : Not (Eq b Top.top)
    c : α
    hc : Membership.mem (setOf fun x => And (LE.le b x) (Ne x Top.top)) c
    hmax : ∀ (a' : α), Membership.mem (setOf fun x => And (LE.le b x) (Ne x Top.to …
    ⊢ Exists fun a => And (IsCoatom a) (LE.le b a)
  -/
  refine ⟨c, ⟨hc.2, fun y hcy => ?_⟩, hc.1⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : OrderTop α
    inst✝ : Finite α
    b : α
    ht : Not (Eq b Top.top)
    c : α
    hc : Membership.mem (setOf fun x => And (LE.le b x) (Ne x Top.top)) c
    hmax : ∀ (a' : α), Membership.mem (setOf fun x => And (LE.le b x) (Ne x Top.to …
    y : α
    hcy : LT.lt c y
    ⊢ Eq y Top.top
  -/
  by_contra hyt
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : OrderTop α
    inst✝ : Finite α
    b : α
    ht : Not (Eq b Top.top)
    c : α
    hc : Membership.mem (setOf fun x => And (LE.le b x) (Ne x Top.top)) c
    hmax : ∀ (a' : α), Membership.mem (setOf fun x => And (LE.le b x) (Ne x Top.to …
    y : α
    hcy : LT.lt c y
    hyt : Not (Eq y Top.top)
    ⊢ False
  -/
  obtain rfl : c = y := hmax y ⟨hc.1.trans hcy.le, hyt⟩ hcy.le
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : OrderTop α
    inst✝ : Finite α
    b : α
    ht : Not (Eq b Top.top)
    c : α
    hc : Membership.mem (setOf fun x => And (LE.le b x) (Ne x Top.top)) c
    hmax : ∀ (a' : α), Membership.mem (setOf fun x => And (LE.le b x) (Ne x Top.to …
    hcy : LT.lt c c
    hyt : Not (Eq c Top.top)
    ⊢ False
  -/
  exact (lt_self_iff_false _).mp hcy
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

instance (priority := 100) Finite.to_isAtomic [PartialOrder α] [OrderBot α] [Finite α] :
    IsAtomic α :=
  isCoatomic_dual_iff_isAtomic.mp Finite.to_isCoatomic


instance : IsStronglyAtomic α where
  exists_covBy_le_of_lt a b hab := by
    obtain ⟨x, hxmem, hx⟩ := (LocallyFiniteOrder.finsetIoc a b).exists_minimal
      ⟨b, by simpa [LocallyFiniteOrder.finset_mem_Ioc]⟩
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : LocallyFiniteOrder α
      a b : α
      hab : LT.lt a b
      x : α
      hxmem : Membership.mem (LocallyFiniteOrder.finsetIoc a b) x
      hx : ∀ (x_1 : α), Membership.mem (LocallyFiniteOrder.finsetIoc a b) x_1 → Not  …
      ⊢ Exists fun x => And (CovBy a x) (LE.le x b)
    -/
    simp only [LocallyFiniteOrder.finset_mem_Ioc, and_imp] at hxmem hx
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : LocallyFiniteOrder α
      a b : α
      hab : LT.lt a b
      x : α
      hxmem : And (LT.lt a x) (LE.le x b)
      hx : ∀ (x_1 : α), LT.lt a x_1 → LE.le x_1 b → Not (LT.lt x_1 x)
      ⊢ Exists fun x => And (CovBy a x) (LE.le x b)
    -/
    exact ⟨x, ⟨hxmem.1, fun c hac hcx ↦ hx _ hac (hcx.le.trans hxmem.2) hcx⟩, hxmem.2⟩
    /-
      🎉 no goals
    -/


instance : IsStronglyCoatomic α := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ IsStronglyCoatomic α
  -/
  rw [← isStronglyAtomic_dual_iff_is_stronglyCoatomic]; infer_instance
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem exists_covby_infinite_Ici_of_infinite_Ici [IsStronglyAtomic α]
    (ha : (Set.Ici a).Infinite) (hfin : {x | a ⋖ x}.Finite) :
    ∃ b, a ⋖ b ∧ (Set.Ici b).Infinite := by
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    a : α
    inst✝ : IsStronglyAtomic α
    ha : (Set.Ici a).Infinite
    hfin : (setOf fun x => CovBy a x).Finite
    ⊢ Exists fun b => And (CovBy a b) (Set.Ici b).Infinite
  -/
  by_contra! h
  refine ((hfin.biUnion (t := Set.Ici) (by simpa using h)).subset (fun b hb ↦ ?_)).not_infinite
    (ha.diff (Set.finite_singleton a))
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    a : α
    inst✝ : IsStronglyAtomic α
    ha : (Set.Ici a).Infinite
    hfin : (setOf fun x => CovBy a x).Finite
    h : ∀ (b : α), CovBy a b → Not (Set.Ici b).Infinite
    b : α
    hb : Membership.mem (SDiff.sdiff (Set.Ici a) (Singleton.singleton a)) b
    ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun h => Set.Ici i) b
  -/
  obtain ⟨x, hax, hxb⟩ := ((show a ≤ b from hb.1).lt_of_ne (Ne.symm hb.2)).exists_covby_le
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : PartialOrder α
    a : α
    inst✝ : IsStronglyAtomic α
    ha : (Set.Ici a).Infinite
    hfin : (setOf fun x => CovBy a x).Finite
    h : ∀ (b : α), CovBy a b → Not (Set.Ici b).Infinite
    b : α
    hb : Membership.mem (SDiff.sdiff (Set.Ici a) (Singleton.singleton a)) b
    x : α
    hax : CovBy a x
    hxb : LE.le x b
    ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun h => Set.Ici i) b
  -/
  exact Set.mem_biUnion hax hxb
  /-
    🎉 no goals
  -/


theorem exists_covby_infinite_Iic_of_infinite_Iic [IsStronglyCoatomic α]
    (ha : (Set.Iic a).Infinite) (hfin : {x | x ⋖ a}.Finite) :
    ∃ b, b ⋖ a ∧ (Set.Iic b).Infinite := by
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    a : α
    inst✝ : IsStronglyCoatomic α
    ha : (Set.Iic a).Infinite
    hfin : (setOf fun x => CovBy x a).Finite
    ⊢ Exists fun b => And (CovBy b a) (Set.Iic b).Infinite
  -/
  simp_rw [← toDual_covBy_toDual_iff (α := α)] at hfin ⊢
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    a : α
    inst✝ : IsStronglyCoatomic α
    ha : (Set.Iic a).Infinite
    hfin : (setOf fun x => CovBy (OrderDual.toDual a) (OrderDual.toDual x)).Finite
    ⊢ Exists fun b => And (CovBy (OrderDual.toDual a) (OrderDual.toDual b)) (Set.I …
  -/
  exact exists_covby_infinite_Ici_of_infinite_Ici (α := αᵒᵈ) ha hfin
  /-
    🎉 no goals
  -/


