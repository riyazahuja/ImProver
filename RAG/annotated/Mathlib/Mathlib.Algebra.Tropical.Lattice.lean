instance instSemilatticeInfTropical [SemilatticeInf R] : SemilatticeInf (Tropical R) :=
  { Tropical.instPartialOrderTropical with
    inf := fun x y ↦ trop (untrop x ⊓ untrop y)
    le_inf := fun _ _ _ ↦ @SemilatticeInf.le_inf R _ _ _ _
    inf_le_left := fun _ _ ↦ inf_le_left
    inf_le_right := fun _ _ ↦ inf_le_right }


instance instSemilatticeSupTropical [SemilatticeSup R] : SemilatticeSup (Tropical R) :=
  { Tropical.instPartialOrderTropical with
    sup := fun x y ↦ trop (untrop x ⊔ untrop y)
    sup_le := fun _ _ _ ↦ @SemilatticeSup.sup_le R _ _ _ _
    le_sup_left := fun _ _ ↦ le_sup_left
    le_sup_right := fun _ _ ↦ le_sup_right }


instance instLatticeTropical [Lattice R] : Lattice (Tropical R) :=
  { instSemilatticeInfTropical, instSemilatticeSupTropical with }


instance [SupSet R] : SupSet (Tropical R) where sSup s := trop (sSup (untrop '' s))


instance [InfSet R] : InfSet (Tropical R) where sInf s := trop (sInf (untrop '' s))


instance instConditionallyCompleteLatticeTropical [ConditionallyCompleteLattice R] :
    ConditionallyCompleteLattice (Tropical R) :=
  { instLatticeTropical with
    le_csSup := fun _s _x hs hx ↦
      le_csSup (untrop_monotone.map_bddAbove hs) (Set.mem_image_of_mem untrop hx)
    csSup_le := fun _s _x hs hx ↦
      csSup_le (hs.image untrop) (untrop_monotone.mem_upperBounds_image hx)
    le_csInf := fun _s _x hs hx ↦
      le_csInf (hs.image untrop) (untrop_monotone.mem_lowerBounds_image hx)
    csInf_le := fun _s _x hs hx ↦
      csInf_le (untrop_monotone.map_bddBelow hs) (Set.mem_image_of_mem untrop hx) }


instance [ConditionallyCompleteLinearOrder R] : ConditionallyCompleteLinearOrder (Tropical R) :=
  { instConditionallyCompleteLatticeTropical, Tropical.instLinearOrderTropical with
    csSup_of_not_bddAbove := by
      /-
        R : Type u_1
        S : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder R
        ⊢ ∀ (s : Set (Tropical R)), Not (BddAbove s) → Eq (SupSet.sSup s) (SupSet.sSup …
      -/
      intro s hs
      /-
        R : Type u_1
        S : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder R
        s : Set (Tropical R)
        hs : Not (BddAbove s)
        ⊢ Eq (SupSet.sSup s) (SupSet.sSup EmptyCollection.emptyCollection)
      -/
      have : Set.range untrop = (Set.univ : Set R) := Equiv.range_eq_univ tropEquiv.symm
      /-
        R : Type u_1
        S : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder R
        s : Set (Tropical R)
        hs : Not (BddAbove s)
        this : Eq (Set.range Tropical.untrop) Set.univ
        ⊢ Eq (SupSet.sSup s) (SupSet.sSup EmptyCollection.emptyCollection)
      -/
      simp only [sSup, Set.image_empty, trop_inj_iff]
      /-
        R : Type u_1
        S : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder R
        s : Set (Tropical R)
        hs : Not (BddAbove s)
        this : Eq (Set.range Tropical.untrop) Set.univ
        ⊢ Eq (SupSet.sSup (Set.image Tropical.untrop s)) (SupSet.sSup EmptyCollection. …
      -/
      apply csSup_of_not_bddAbove
      /-
        case hs
        R : Type u_1
        S : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder R
        s : Set (Tropical R)
        hs : Not (BddAbove s)
        this : Eq (Set.range Tropical.untrop) Set.univ
        ⊢ Not (BddAbove (Set.image Tropical.untrop s))
      -/
      contrapose! hs
      /-
        case hs
        R : Type u_1
        S : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder R
        s : Set (Tropical R)
        this : Eq (Set.range Tropical.untrop) Set.univ
        hs : BddAbove (Set.image Tropical.untrop s)
        ⊢ BddAbove s
      -/
      change BddAbove (tropOrderIso.symm '' s) at hs
      /-
        case hs
        R : Type u_1
        S : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder R
        s : Set (Tropical R)
        this : Eq (Set.range Tropical.untrop) Set.univ
        hs : BddAbove (Set.image (⇑Tropical.tropOrderIso.symm) s)
        ⊢ BddAbove s
      -/
      exact tropOrderIso.symm.bddAbove_image.1 hs
      /-
        🎉 no goals
      -/
    csInf_of_not_bddBelow := by
      /-
        R : Type u_1
        S : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder R
        ⊢ ∀ (s : Set (Tropical R)), Not (BddBelow s) → Eq (InfSet.sInf s) (InfSet.sInf …
      -/
      intro s hs
      /-
        R : Type u_1
        S : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder R
        s : Set (Tropical R)
        hs : Not (BddBelow s)
        ⊢ Eq (InfSet.sInf s) (InfSet.sInf EmptyCollection.emptyCollection)
      -/
      have : Set.range untrop = (Set.univ : Set R) := Equiv.range_eq_univ tropEquiv.symm
      /-
        R : Type u_1
        S : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder R
        s : Set (Tropical R)
        hs : Not (BddBelow s)
        this : Eq (Set.range Tropical.untrop) Set.univ
        ⊢ Eq (InfSet.sInf s) (InfSet.sInf EmptyCollection.emptyCollection)
      -/
      simp only [sInf, Set.image_empty, trop_inj_iff]
      /-
        R : Type u_1
        S : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder R
        s : Set (Tropical R)
        hs : Not (BddBelow s)
        this : Eq (Set.range Tropical.untrop) Set.univ
        ⊢ Eq (InfSet.sInf (Set.image Tropical.untrop s)) (InfSet.sInf EmptyCollection. …
      -/
      apply csInf_of_not_bddBelow
      /-
        case hs
        R : Type u_1
        S : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder R
        s : Set (Tropical R)
        hs : Not (BddBelow s)
        this : Eq (Set.range Tropical.untrop) Set.univ
        ⊢ Not (BddBelow (Set.image Tropical.untrop s))
      -/
      contrapose! hs
      /-
        case hs
        R : Type u_1
        S : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder R
        s : Set (Tropical R)
        this : Eq (Set.range Tropical.untrop) Set.univ
        hs : BddBelow (Set.image Tropical.untrop s)
        ⊢ BddBelow s
      -/
      change BddBelow (tropOrderIso.symm '' s) at hs
      /-
        case hs
        R : Type u_1
        S : Type u_2
        inst✝ : ConditionallyCompleteLinearOrder R
        s : Set (Tropical R)
        this : Eq (Set.range Tropical.untrop) Set.univ
        hs : BddBelow (Set.image (⇑Tropical.tropOrderIso.symm) s)
        ⊢ BddBelow s
      -/
      exact tropOrderIso.symm.bddBelow_image.1 hs }
      /-
        🎉 no goals
      -/

