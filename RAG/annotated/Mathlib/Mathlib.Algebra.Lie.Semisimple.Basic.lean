lemma LieModule.nontrivial_of_isIrreducible [LieModule.IsIrreducible R L M] : Nontrivial M where
  exists_pair_ne := by
    /-
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : LieRing L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule.IsIrreducible R L M
      ⊢ Exists fun x => Exists fun y => Ne x y
    -/
    have aux : (⊥ : LieSubmodule R L M) ≠ ⊤ := bot_ne_top
    /-
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : LieRing L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule.IsIrreducible R L M
      aux : Ne Bot.bot Top.top
      ⊢ Exists fun x => Exists fun y => Ne x y
    -/
    contrapose! aux
    /-
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : LieRing L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule.IsIrreducible R L M
      aux : ∀ (x y : M), Eq x y
      ⊢ Eq Bot.bot Top.top
    -/
    ext m
    /-
      case h
      R : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : LieRing L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule.IsIrreducible R L M
      aux : ∀ (x y : M), Eq x y
      m : M
      ⊢ Iff (Membership.mem Bot.bot m) (Membership.mem Top.top m)
    -/
    simpa using aux m 0
    /-
      🎉 no goals
    -/


variable {R L} in
theorem HasTrivialRadical.eq_bot_of_isSolvable [HasTrivialRadical R L]
    (I : LieIdeal R L) [hI : IsSolvable R I] : I = ⊥ :=
  sSup_eq_bot.mp radical_eq_bot _ hI


@[simp]
theorem HasTrivialRadical.center_eq_bot [HasTrivialRadical R L] : center R L = ⊥ :=
  HasTrivialRadical.eq_bot_of_isSolvable _


variable {R L} in
theorem hasTrivialRadical_of_no_solvable_ideals (h : ∀ I : LieIdeal R L, IsSolvable R I → I = ⊥) :
    HasTrivialRadical R L :=
  ⟨sSup_eq_bot.mpr h⟩


theorem hasTrivialRadical_iff_no_solvable_ideals :
    HasTrivialRadical R L ↔ ∀ I : LieIdeal R L, IsSolvable R I → I = ⊥ :=
  ⟨@HasTrivialRadical.eq_bot_of_isSolvable _ _ _ _ _, hasTrivialRadical_of_no_solvable_ideals⟩


theorem hasTrivialRadical_iff_no_abelian_ideals :
    HasTrivialRadical R L ↔ ∀ I : LieIdeal R L, IsLieAbelian I → I = ⊥ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    ⊢ Iff (LieAlgebra.HasTrivialRadical R L) (∀ (I : LieIdeal R L), IsLieAbelian ( …
  -/
  rw [hasTrivialRadical_iff_no_solvable_ideals]
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    ⊢ Iff (∀ (I : LieIdeal R L), LieAlgebra.IsSolvable R (Subtype fun x => Members …
  -/
  constructor <;> intro h₁ I h₂
    /-
      case mp
      R : Type u_1
      L : Type u_2
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      h₁ : ∀ (I : LieIdeal R L), LieAlgebra.IsSolvable R (Subtype fun x => Membershi …
      I : LieIdeal R L
      h₂ : IsLieAbelian (Subtype fun x => Membership.mem I x)
      ⊢ Eq I Bot.bot
    -/
  · exact h₁ _ <| LieAlgebra.ofAbelianIsSolvable R I
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      L : Type u_2
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      h₁ : ∀ (I : LieIdeal R L), IsLieAbelian (Subtype fun x => Membership.mem I x)  …
      I : LieIdeal R L
      h₂ : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
      ⊢ Eq I Bot.bot
    -/
  · rw [← abelian_of_solvable_ideal_eq_bot_iff]
    /-
      case mpr
      R : Type u_1
      L : Type u_2
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      h₁ : ∀ (I : LieIdeal R L), IsLieAbelian (Subtype fun x => Membership.mem I x)  …
      I : LieIdeal R L
      h₂ : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
      ⊢ Eq (LieAlgebra.derivedAbelianOfIdeal I) Bot.bot
    -/
    exact h₁ _ <| abelian_derivedAbelianOfIdeal I
    /-
      🎉 no goals
    -/


instance : LieModule.IsIrreducible R L L := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSimple R L
    ⊢ LieModule.IsIrreducible R L L
  -/
  suffices Nontrivial (LieIdeal R L) from ⟨IsSimple.eq_bot_or_eq_top⟩
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSimple R L
    ⊢ Nontrivial (LieIdeal R L)
  -/
  rw [LieSubmodule.nontrivial_iff, ← not_subsingleton_iff_nontrivial]
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSimple R L
    ⊢ Not (Subsingleton L)
  -/
  have _i : ¬ IsLieAbelian L := IsSimple.non_abelian R
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSimple R L
    _i : Not (IsLieAbelian L)
    ⊢ Not (Subsingleton L)
  -/
  contrapose! _i
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSimple R L
    _i : Subsingleton L
    ⊢ IsLieAbelian L
  -/
  infer_instance
  /-
    🎉 no goals
  -/


variable {R L} in
lemma eq_top_of_isAtom (I : LieIdeal R L) (hI : IsAtom I) : I = ⊤ :=
  (IsSimple.eq_bot_or_eq_top I).resolve_left hI.1


lemma isAtom_top : IsAtom (⊤ : LieIdeal R L) :=
  ⟨bot_ne_top.symm, fun _ h ↦ h.eq_bot⟩


variable {R L} in
@[simp] lemma isAtom_iff_eq_top (I : LieIdeal R L) : IsAtom I ↔ I = ⊤ :=
  ⟨eq_top_of_isAtom I, fun h ↦ h ▸ isAtom_top R L⟩


instance : HasTrivialRadical R L := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSimple R L
    ⊢ LieAlgebra.HasTrivialRadical R L
  -/
  rw [hasTrivialRadical_iff_no_abelian_ideals]
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSimple R L
    ⊢ ∀ (I : LieIdeal R L), IsLieAbelian (Subtype fun x => Membership.mem I x) → E …
  -/
  intro I hI
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSimple R L
    I : LieIdeal R L
    hI : IsLieAbelian (Subtype fun x => Membership.mem I x)
    ⊢ Eq I Bot.bot
  -/
  apply (IsSimple.eq_bot_or_eq_top I).resolve_right
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSimple R L
    I : LieIdeal R L
    hI : IsLieAbelian (Subtype fun x => Membership.mem I x)
    ⊢ Not (Eq I Top.top)
  -/
  rintro rfl
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSimple R L
    hI : IsLieAbelian (Subtype fun x => Membership.mem Top.top x)
    ⊢ False
  -/
  rw [lie_abelian_iff_equiv_lie_abelian LieIdeal.topEquiv] at hI
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSimple R L
    hI : IsLieAbelian L
    ⊢ False
  -/
  exact IsSimple.non_abelian R (L := L) hI
  /-
    🎉 no goals
  -/


lemma isSimple_of_isAtom (I : LieIdeal R L) (hI : IsAtom I) : IsSimple R I where
  non_abelian := IsSemisimple.non_abelian_of_isAtom I hI
  eq_bot_or_eq_top := by
    -- Suppose that `J` is an ideal of `I`.
    /-
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSemisimple R L
      I : LieIdeal R L
      hI : IsAtom I
      ⊢ ∀ (I_1 : LieIdeal R (Subtype fun x => Membership.mem I x)), Or (Eq I_1 Bot.b …
    -/
    intro J
    -- We first show that `J` is also an ideal of the ambient Lie algebra `L`.
    let J' : LieIdeal R L :=
    { __ := J.toSubmodule.map I.incl.toLinearMap
      lie_mem := by
        rintro x _ ⟨y, hy, rfl⟩
        dsimp
        -- We need to show that `⁅x, y⁆ ∈ J` for any `x ∈ L` and `y ∈ J`.
        -- Since `L` is semisimple, `x` is contained
        -- in the supremum of `I` and the atoms not equal to `I`.
        have hx : x ∈ I ⊔ sSup ({I' : LieIdeal R L | IsAtom I'} \ {I}) := by
          nth_rewrite 1 [← sSup_singleton (a := I)]
          rw [← sSup_union, Set.union_diff_self, Set.union_eq_self_of_subset_left,
            IsSemisimple.sSup_atoms_eq_top]
          · apply LieSubmodule.mem_top
          · simp only [Set.singleton_subset_iff, Set.mem_setOf_eq, hI]
        -- Hence we can write `x` as `a + b` with `a ∈ I`
        -- and `b` in the supremum of the atoms not equal to `I`.
        rw [LieSubmodule.mem_sup] at hx
        obtain ⟨a, ha, b, hb, rfl⟩ := hx
        -- Therefore it suffices to show that `⁅a, y⁆ ∈ J` and `⁅b, y⁆ ∈ J`.
        simp only [add_lie, AddSubsemigroup.mem_carrier, AddSubmonoid.mem_toSubsemigroup,
          Submodule.mem_toAddSubmonoid]
        apply add_mem
        -- Now `⁅a, y⁆ ∈ J` since `a ∈ I`, `y ∈ J`, and `J` is an ideal of `I`.
        · simp only [Submodule.mem_map, LieSubmodule.mem_toSubmodule, Subtype.exists]
          erw [Submodule.coe_subtype]
          simp only [exists_and_right, exists_eq_right, ha, lie_mem_left, exists_true_left]
          exact lie_mem_right R I J ⟨a, ha⟩ y hy
        -- Finally `⁅b, y⁆ = 0`, by the independence of the atoms.
        · suffices ⁅b, y.val⁆ = 0 by erw [this]; simp only [zero_mem]
          rw [← LieSubmodule.mem_bot (R := R) (L := L),
              ← (IsSemisimple.sSupIndep_isAtom hI).eq_bot]
          exact ⟨lie_mem_right R L I b y y.2, lie_mem_left _ _ _ _ _ hb⟩ }
    -- Now that we know that `J` is an ideal of `L`,
    -- we start with the proof that `I` is a simple Lie algebra.
    -- Assume that `J ≠ ⊤`.
    /-
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSemisimple R L
      I : LieIdeal R L
      hI : IsAtom I
      J : LieIdeal R (Subtype fun x => Membership.mem I x)
      J' : LieIdeal R L :=
        let __spread.0 := Submodule.map ↑I.incl ↑J;
        { toSubmodule := __spread.0, lie_mem := ⋯ }
      ⊢ Or (Eq J Bot.bot) (Eq J Top.top)
    -/
    rw [or_iff_not_imp_right]
    /-
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSemisimple R L
      I : LieIdeal R L
      hI : IsAtom I
      J : LieIdeal R (Subtype fun x => Membership.mem I x)
      J' : LieIdeal R L :=
        let __spread.0 := Submodule.map ↑I.incl ↑J;
        { toSubmodule := __spread.0, lie_mem := ⋯ }
      ⊢ Not (Eq J Top.top) → Eq J Bot.bot
    -/
    intro hJ
    suffices J' = ⊥ by
      rw [eq_bot_iff] at this ⊢
      intro x hx
      suffices x ∈ J → x = 0 from this hx
      have := @this x.1
      simp only [LieIdeal.incl_coe, LieIdeal.toLieSubalgebra_toSubmodule,
        LieSubmodule.mem_mk_iff', Submodule.mem_map, LieSubmodule.mem_toSubmodule, Subtype.exists,
        LieSubmodule.mem_bot, ZeroMemClass.coe_eq_zero, forall_exists_index, and_imp, J'] at this
      exact fun _ ↦ this (↑x) x.property hx rfl
    -- We need to show that `J = ⊥`.
    -- Since `J` is an ideal of `L`, and `I` is an atom,
    -- it suffices to show that `J < I`.
    /-
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSemisimple R L
      I : LieIdeal R L
      hI : IsAtom I
      J : LieIdeal R (Subtype fun x => Membership.mem I x)
      J' : LieIdeal R L :=
        let __spread.0 := Submodule.map ↑I.incl ↑J;
        { toSubmodule := __spread.0, lie_mem := ⋯ }
      hJ : Not (Eq J Top.top)
      ⊢ Eq J' Bot.bot
    -/
    apply hI.2
    /-
      case a
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSemisimple R L
      I : LieIdeal R L
      hI : IsAtom I
      J : LieIdeal R (Subtype fun x => Membership.mem I x)
      J' : LieIdeal R L :=
        let __spread.0 := Submodule.map ↑I.incl ↑J;
        { toSubmodule := __spread.0, lie_mem := ⋯ }
      hJ : Not (Eq J Top.top)
      ⊢ LT.lt J' I
    -/
    rw [lt_iff_le_and_ne]
    /-
      case a
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSemisimple R L
      I : LieIdeal R L
      hI : IsAtom I
      J : LieIdeal R (Subtype fun x => Membership.mem I x)
      J' : LieIdeal R L :=
        let __spread.0 := Submodule.map ↑I.incl ↑J;
        { toSubmodule := __spread.0, lie_mem := ⋯ }
      hJ : Not (Eq J Top.top)
      ⊢ And (LE.le J' I) (Ne J' I)
    -/
    constructor
    -- We know that `J ≤ I` since `J` is an ideal of `I`.
      /-
        case a.left
        R : Type u_1
        L : Type u_2
        inst✝³ : CommRing R
        inst✝² : LieRing L
        inst✝¹ : LieAlgebra R L
        inst✝ : LieAlgebra.IsSemisimple R L
        I : LieIdeal R L
        hI : IsAtom I
        J : LieIdeal R (Subtype fun x => Membership.mem I x)
        J' : LieIdeal R L :=
          let __spread.0 := Submodule.map ↑I.incl ↑J;
          { toSubmodule := __spread.0, lie_mem := ⋯ }
        hJ : Not (Eq J Top.top)
        ⊢ LE.le J' I
      -/
    · rintro _ ⟨x, -, rfl⟩
      /-
        case a.left.intro.intro
        R : Type u_1
        L : Type u_2
        inst✝³ : CommRing R
        inst✝² : LieRing L
        inst✝¹ : LieAlgebra R L
        inst✝ : LieAlgebra.IsSemisimple R L
        I : LieIdeal R L
        hI : IsAtom I
        J : LieIdeal R (Subtype fun x => Membership.mem I x)
        J' : LieIdeal R L :=
          let __spread.0 := Submodule.map ↑I.incl ↑J;
          { toSubmodule := __spread.0, lie_mem := ⋯ }
        hJ : Not (Eq J Top.top)
        x : Subtype fun x => Membership.mem I x
        ⊢ Membership.mem I (↑I.incl x)
      -/
      exact x.2
      /-
        🎉 no goals
      -/
    -- So we need to show `J ≠ I` as ideals of `L`.
    -- This follows from our assumption that `J ≠ ⊤` as ideals of `I`.
    /-
      case a.right
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSemisimple R L
      I : LieIdeal R L
      hI : IsAtom I
      J : LieIdeal R (Subtype fun x => Membership.mem I x)
      J' : LieIdeal R L :=
        let __spread.0 := Submodule.map ↑I.incl ↑J;
        { toSubmodule := __spread.0, lie_mem := ⋯ }
      hJ : Not (Eq J Top.top)
      ⊢ Ne J' I
    -/
    contrapose! hJ
    /-
      case a.right
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSemisimple R L
      I : LieIdeal R L
      hI : IsAtom I
      J : LieIdeal R (Subtype fun x => Membership.mem I x)
      J' : LieIdeal R L :=
        let __spread.0 := Submodule.map ↑I.incl ↑J;
        { toSubmodule := __spread.0, lie_mem := ⋯ }
      hJ : Eq J' I
      ⊢ Eq J Top.top
    -/
    rw [eq_top_iff]
    /-
      case a.right
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSemisimple R L
      I : LieIdeal R L
      hI : IsAtom I
      J : LieIdeal R (Subtype fun x => Membership.mem I x)
      J' : LieIdeal R L :=
        let __spread.0 := Submodule.map ↑I.incl ↑J;
        { toSubmodule := __spread.0, lie_mem := ⋯ }
      hJ : Eq J' I
      ⊢ LE.le Top.top J
    -/
    rintro ⟨x, hx⟩ -
    /-
      case a.right.mk
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSemisimple R L
      I : LieIdeal R L
      hI : IsAtom I
      J : LieIdeal R (Subtype fun x => Membership.mem I x)
      J' : LieIdeal R L :=
        let __spread.0 := Submodule.map ↑I.incl ↑J;
        { toSubmodule := __spread.0, lie_mem := ⋯ }
      hJ : Eq J' I
      x : L
      hx : Membership.mem I x
      ⊢ Membership.mem J ⟨x, hx⟩
    -/
    rw [← hJ] at hx
    /-
      case a.right.mk
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSemisimple R L
      I : LieIdeal R L
      hI : IsAtom I
      J : LieIdeal R (Subtype fun x => Membership.mem I x)
      J' : LieIdeal R L :=
        let __spread.0 := Submodule.map ↑I.incl ↑J;
        { toSubmodule := __spread.0, lie_mem := ⋯ }
      hJ : Eq J' I
      x : L
      hx✝ : Membership.mem I x
      hx : Membership.mem J' x
      ⊢ Membership.mem J ⟨x, hx✝⟩
    -/
    rcases hx with ⟨y, hy, rfl⟩
    /-
      case a.right.mk.intro.intro
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSemisimple R L
      I : LieIdeal R L
      hI : IsAtom I
      J : LieIdeal R (Subtype fun x => Membership.mem I x)
      J' : LieIdeal R L :=
        let __spread.0 := Submodule.map ↑I.incl ↑J;
        { toSubmodule := __spread.0, lie_mem := ⋯ }
      hJ : Eq J' I
      y : Subtype fun x => Membership.mem I x
      hy : Membership.mem (↑↑J) y
      hx : Membership.mem I (↑I.incl y)
      ⊢ Membership.mem J ⟨↑I.incl y, hx⟩
    -/
    exact hy
    /-
      🎉 no goals
    -/


/--
In a semisimple Lie algebra,
Lie ideals that are contained in the supremum of a finite collection of atoms
are themselves the supremum of a finite subcollection of those atoms.

By a compactness argument, this statement can be extended to arbitrary sets of atoms.
See `atomistic`.

The proof is by induction on the finite set of atoms.
-/
private
lemma finitelyAtomistic : ∀ s : Finset (LieIdeal R L), ↑s ⊆ {I : LieIdeal R L | IsAtom I} →
    ∀ I : LieIdeal R L, I ≤ s.sup id → ∃ t ⊆ s, I = t.sup id := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSemisimple R L
    ⊢ ∀ (s : Finset (LieIdeal R L)), HasSubset.Subset (↑s) (setOf fun I => IsAtom  …
  -/
  intro s hs I hI
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSemisimple R L
    s : Finset (LieIdeal R L)
    hs : HasSubset.Subset (↑s) (setOf fun I => IsAtom I)
    I : LieIdeal R L
    hI : LE.le I (s.sup id)
    ⊢ Exists fun t => And (HasSubset.Subset t s) (Eq I (t.sup id))
  -/
  let S := {I : LieIdeal R L | IsAtom I}
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSemisimple R L
    s : Finset (LieIdeal R L)
    hs : HasSubset.Subset (↑s) (setOf fun I => IsAtom I)
    I : LieIdeal R L
    hI : LE.le I (s.sup id)
    S : Set (LieIdeal R L) := setOf fun I => IsAtom I
    ⊢ Exists fun t => And (HasSubset.Subset t s) (Eq I (t.sup id))
  -/
  obtain rfl | hI := hI.eq_or_lt
    /-
      case inl
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSemisimple R L
      s : Finset (LieIdeal R L)
      hs : HasSubset.Subset (↑s) (setOf fun I => IsAtom I)
      S : Set (LieIdeal R L) := setOf fun I => IsAtom I
      hI : LE.le (s.sup id) (s.sup id)
      ⊢ Exists fun t => And (HasSubset.Subset t s) (Eq (s.sup id) (t.sup id))
    -/
  · exact ⟨s, Finset.Subset.rfl, rfl⟩
    /-
      🎉 no goals
    -/
  -- We assume that `I` is strictly smaller than the supremum of `s`.
  -- Hence there must exist an atom `J` that is not contained in `I`.
  obtain ⟨J, hJs, hJI⟩ : ∃ J ∈ s, ¬ J ≤ I := by
    by_contra! H
    exact hI.ne (le_antisymm hI.le (s.sup_le H))
  classical
  let s' := s.erase J
  have hs' : s' ⊂ s := Finset.erase_ssubset hJs
  have hs'S : ↑s' ⊆ S := Set.Subset.trans (Finset.coe_subset.mpr hs'.subset) hs
  -- If we show that `I` is contained in the supremum `K` of the complement of `J` in `s`,
  -- then we are done by recursion.
  set K := s'.sup id
  suffices I ≤ K by
    obtain ⟨t, hts', htI⟩ := finitelyAtomistic s' hs'S I this
    #adaptation_note
    /-- Prior to https://github.com/leanprover/lean4/pull/6024
    we could write `hts'.trans hs'.subset` instead of
    `Finset.Subset.trans hts' hs'.subset` in the next line. -/
    exact ⟨t, Finset.Subset.trans hts' hs'.subset, htI⟩
  -- Since `I` is contained in the supremum of `J` with the supremum of `s'`,
  -- any element `x` of `I` can be written as `y + z` for some `y ∈ J` and `z ∈ K`.
  intro x hx
  obtain ⟨y, hy, z, hz, rfl⟩ : ∃ y ∈ id J, ∃ z ∈ K, y + z = x := by
    rw [← LieSubmodule.mem_sup, ← Finset.sup_insert, Finset.insert_erase hJs]
    exact hI.le hx
  -- If we show that `y` is contained in the center of `J`,
  -- then we find `x = z`, and hence `x` is contained in the supremum of `s'`.
  -- Since `x` was arbitrary, we have shown that `I` is contained in the supremum of `s'`.
  suffices ⟨y, hy⟩ ∈ LieAlgebra.center R J by
    have _inst := isSimple_of_isAtom J (hs hJs)
    rw [HasTrivialRadical.center_eq_bot R J, LieSubmodule.mem_bot] at this
    apply_fun Subtype.val at this
    dsimp at this
    rwa [this, zero_add]
  -- To show that `y` is in the center of `J`,
  -- we show that any `j ∈ J` brackets to `0` with `z` and with `x = y + z`.
  -- By a simple computation, that implies `⁅j, y⁆ = 0`, for all `j`, as desired.
  intro j
  suffices ⁅(j : L), z⁆ = 0 ∧ ⁅(j : L), y + z⁆ = 0 by
    rw [lie_add, this.1, add_zero] at this
    ext
    exact this.2
  rw [← LieSubmodule.mem_bot (R := R) (L := L), ← LieSubmodule.mem_bot (R := R) (L := L)]
  constructor
  -- `j` brackets to `0` with `z`, since `⁅j, z⁆` is contained in `⁅J, K⁆ ≤ J ⊓ K`,
  -- and `J ⊓ K = ⊥` by the independence of the atoms.
  · apply (sSupIndep_isAtom.disjoint_sSup (hs hJs) hs'S (Finset.not_mem_erase _ _)).le_bot
    apply LieSubmodule.lie_le_inf
    apply LieSubmodule.lie_mem_lie j.2
    simpa only [K, Finset.sup_id_eq_sSup] using hz
  -- By similar reasoning, `j` brackets to `0` with `x = y + z ∈ I`, if we show `J ⊓ I = ⊥`.
  suffices J ⊓ I = ⊥ by
    apply this.le
    apply LieSubmodule.lie_le_inf
    exact LieSubmodule.lie_mem_lie j.2 hx
  -- Indeed `J ⊓ I = ⊥`, since `J` is an atom that is not contained in `I`.
  apply ((hs hJs).le_iff.mp _).resolve_right
  · contrapose! hJI
    rw [← hJI]
    exact inf_le_right
  exact inf_le_left
termination_by s => s.card
/-
  R : Type u_1
  L : Type u_2
  inst✝³ : CommRing R
  inst✝² : LieRing L
  inst✝¹ : LieAlgebra R L
  inst✝ : LieAlgebra.IsSemisimple R L
  x✝ : PSigma fun s => PSigma fun a => PSigma fun I => LE.le I (s.sup id)
  a✝⁶ : ∀ (y : PSigma fun s => PSigma fun a => PSigma fun I => LE.le I (s.sup id …
  s : Finset (LieIdeal R L)
  a✝⁵ : PSigma fun hs => PSigma fun I => LE.le I (s.sup id)
  a✝⁴ : ∀ (y : PSigma fun s => PSigma fun a => PSigma fun I => LE.le I (s.sup id …
  a✝³ : HasSubset.Subset (↑s) (setOf fun I => IsAtom I)
  I✝ : PSigma fun I => LE.le I (s.sup id)
  a✝² : ∀ (y : PSigma fun s => PSigma fun a => PSigma fun I => LE.le I (s.sup id …
  I : LieIdeal R L
  a✝¹ : LE.le I (s.sup id)
  a✝ : ∀ (y : PSigma fun s => PSigma fun a => PSigma fun I => LE.le I (s.sup id) …
  S : Set (LieIdeal R L) := setOf fun I => IsAtom I
  hI : LT.lt I (s.sup id)
  J : LieIdeal R L
  h✝ : And (Membership.mem s J) (Not (LE.le J I))
  hJs : Membership.mem s J
  hJI : Not (LE.le J I)
  s' : Finset (LieIdeal R L) := s.erase J
  hs' : HasSSubset.SSubset s' s
  hs'S : HasSubset.Subset (↑s') S
  K : LieIdeal R L := s'.sup id
  this : LE.le I K
  ⊢ LT.lt (s.erase J).card s.card
-/
decreasing_by exact Finset.card_lt_card hs'
/-
  🎉 no goals
-/


variable (R L) in
lemma booleanGenerators : BooleanGenerators {I : LieIdeal R L | IsAtom I} where
  isAtom _ hI := hI
  finitelyAtomistic _ _ hs _ hIs := finitelyAtomistic _ hs _ hIs


instance (priority := 100) instDistribLattice : DistribLattice (LieIdeal R L) :=
  (booleanGenerators R L).distribLattice_of_sSup_eq_top sSup_atoms_eq_top


noncomputable
instance (priority := 100) instBooleanAlgebra : BooleanAlgebra (LieIdeal R L) :=
  (booleanGenerators R L).booleanAlgebra_of_sSup_eq_top sSup_atoms_eq_top


/-- A semisimple Lie algebra has trivial radical. -/
instance (priority := 100) instHasTrivialRadical : HasTrivialRadical R L := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSemisimple R L
    ⊢ LieAlgebra.HasTrivialRadical R L
  -/
  rw [hasTrivialRadical_iff_no_abelian_ideals]
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSemisimple R L
    ⊢ ∀ (I : LieIdeal R L), IsLieAbelian (Subtype fun x => Membership.mem I x) → E …
  -/
  intro I hI
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSemisimple R L
    I : LieIdeal R L
    hI : IsLieAbelian (Subtype fun x => Membership.mem I x)
    ⊢ Eq I Bot.bot
  -/
  apply (eq_bot_or_exists_atom_le I).resolve_right
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSemisimple R L
    I : LieIdeal R L
    hI : IsLieAbelian (Subtype fun x => Membership.mem I x)
    ⊢ Not (Exists fun a => And (IsAtom a) (LE.le a I))
  -/
  rintro ⟨J, hJ, hJ'⟩
  /-
    case intro.intro
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSemisimple R L
    I : LieIdeal R L
    hI : IsLieAbelian (Subtype fun x => Membership.mem I x)
    J : LieIdeal R L
    hJ : IsAtom J
    hJ' : LE.le J I
    ⊢ False
  -/
  apply IsSemisimple.non_abelian_of_isAtom J hJ
  /-
    case intro.intro
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSemisimple R L
    I : LieIdeal R L
    hI : IsLieAbelian (Subtype fun x => Membership.mem I x)
    J : LieIdeal R L
    hJ : IsAtom J
    hJ' : LE.le J I
    ⊢ IsLieAbelian (Subtype fun x => Membership.mem J x)
  -/
  constructor
  /-
    case intro.intro.trivial
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSemisimple R L
    I : LieIdeal R L
    hI : IsLieAbelian (Subtype fun x => Membership.mem I x)
    J : LieIdeal R L
    hJ : IsAtom J
    hJ' : LE.le J I
    ⊢ ∀ (x m : Subtype fun x => Membership.mem J x), Eq (Bracket.bracket x m) 0
  -/
  intro x y
  /-
    case intro.intro.trivial
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSemisimple R L
    I : LieIdeal R L
    hI : IsLieAbelian (Subtype fun x => Membership.mem I x)
    J : LieIdeal R L
    hJ : IsAtom J
    hJ' : LE.le J I
    x y : Subtype fun x => Membership.mem J x
    ⊢ Eq (Bracket.bracket x y) 0
  -/
  ext
  /-
    case intro.intro.trivial.a
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSemisimple R L
    I : LieIdeal R L
    hI : IsLieAbelian (Subtype fun x => Membership.mem I x)
    J : LieIdeal R L
    hJ : IsAtom J
    hJ' : LE.le J I
    x y : Subtype fun x => Membership.mem J x
    ⊢ Eq ↑(Bracket.bracket x y) ↑0
  -/
  simp only [LieIdeal.coe_bracket_of_module, LieSubmodule.coe_bracket, ZeroMemClass.coe_zero]
  /-
    case intro.intro.trivial.a
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSemisimple R L
    I : LieIdeal R L
    hI : IsLieAbelian (Subtype fun x => Membership.mem I x)
    J : LieIdeal R L
    hJ : IsAtom J
    hJ' : LE.le J I
    x y : Subtype fun x => Membership.mem J x
    ⊢ Eq (Bracket.bracket ↑x ↑y) 0
  -/
  have : (⁅(⟨x, hJ' x.2⟩ : I), ⟨y, hJ' y.2⟩⁆ : I) = 0 := trivial_lie_zero _ _ _ _
  /-
    case intro.intro.trivial.a
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSemisimple R L
    I : LieIdeal R L
    hI : IsLieAbelian (Subtype fun x => Membership.mem I x)
    J : LieIdeal R L
    hJ : IsAtom J
    hJ' : LE.le J I
    x y : Subtype fun x => Membership.mem J x
    this : Eq (Bracket.bracket ⟨↑x, ⋯⟩ ⟨↑y, ⋯⟩) 0
    ⊢ Eq (Bracket.bracket ↑x ↑y) 0
  -/
  apply_fun Subtype.val at this
  /-
    case intro.intro.trivial.a
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSemisimple R L
    I : LieIdeal R L
    hI : IsLieAbelian (Subtype fun x => Membership.mem I x)
    J : LieIdeal R L
    hJ : IsAtom J
    hJ' : LE.le J I
    x y : Subtype fun x => Membership.mem J x
    this : Eq ↑(Bracket.bracket ⟨↑x, ⋯⟩ ⟨↑y, ⋯⟩) ↑0
    ⊢ Eq (Bracket.bracket ↑x ↑y) 0
  -/
  exact this
  /-
    🎉 no goals
  -/


/-- A simple Lie algebra is semisimple. -/
instance (priority := 100) IsSimple.instIsSemisimple [IsSimple R L] :
    IsSemisimple R L := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsSimple R L
    ⊢ LieAlgebra.IsSemisimple R L
  -/
  constructor
    /-
      case sSup_atoms_eq_top
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSimple R L
      ⊢ Eq (SupSet.sSup (setOf fun I => IsAtom I)) Top.top
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case sSupIndep_isAtom
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSimple R L
      ⊢ sSupIndep (setOf fun I => IsAtom I)
    -/
  · simpa using sSupIndep_singleton _
    /-
      🎉 no goals
    -/
    /-
      case non_abelian_of_isAtom
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSimple R L
      ⊢ ∀ (I : LieIdeal R L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Members …
    -/
  · intro I hI₁ hI₂
    /-
      case non_abelian_of_isAtom
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSimple R L
      I : LieIdeal R L
      hI₁ : IsAtom I
      hI₂ : IsLieAbelian (Subtype fun x => Membership.mem I x)
      ⊢ False
    -/
    apply IsSimple.non_abelian (R := R) (L := L)
    /-
      case non_abelian_of_isAtom
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSimple R L
      I : LieIdeal R L
      hI₁ : IsAtom I
      hI₂ : IsLieAbelian (Subtype fun x => Membership.mem I x)
      ⊢ IsLieAbelian L
    -/
    rw [IsSimple.isAtom_iff_eq_top] at hI₁
    /-
      case non_abelian_of_isAtom
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : LieAlgebra.IsSimple R L
      I : LieIdeal R L
      hI₁ : Eq I Top.top
      hI₂ : IsLieAbelian (Subtype fun x => Membership.mem I x)
      ⊢ IsLieAbelian L
    -/
    rwa [hI₁, lie_abelian_iff_equiv_lie_abelian LieIdeal.topEquiv] at hI₂
    /-
      🎉 no goals
    -/


/-- An abelian Lie algebra with trivial radical is trivial. -/
theorem subsingleton_of_hasTrivialRadical_lie_abelian [HasTrivialRadical R L] [h : IsLieAbelian L] :
    Subsingleton L := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.HasTrivialRadical R L
    h : IsLieAbelian L
    ⊢ Subsingleton L
  -/
  rw [isLieAbelian_iff_center_eq_top R L, HasTrivialRadical.center_eq_bot] at h
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.HasTrivialRadical R L
    h : Eq Bot.bot Top.top
    ⊢ Subsingleton L
  -/
  exact (LieSubmodule.subsingleton_iff R L L).mp (subsingleton_of_bot_eq_top h)
  /-
    🎉 no goals
  -/


theorem abelian_radical_of_hasTrivialRadical [HasTrivialRadical R L] :
    IsLieAbelian (radical R L) := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.HasTrivialRadical R L
    ⊢ IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.radical R L) x)
  -/
  rw [HasTrivialRadical.radical_eq_bot]; exact LieIdeal.isLieAbelian_of_trivial ..
                                         /-
                                           🎉 no goals
                                         -/


/-- The two properties shown to be equivalent here are possible definitions for a Lie algebra
to be reductive.

Note that there is absolutely [no agreement](https://mathoverflow.net/questions/284713/) on what
the label 'reductive' should mean when the coefficients are not a field of characteristic zero. -/
theorem abelian_radical_iff_solvable_is_abelian [IsNoetherian R L] :
    IsLieAbelian (radical R L) ↔ ∀ I : LieIdeal R L, IsSolvable R I → IsLieAbelian I := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    ⊢ Iff (IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.radical R L)  …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsNoetherian R L
      ⊢ IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.radical R L) x) →  …
    -/
  · rintro h₁ I h₂
    /-
      case mp
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsNoetherian R L
      h₁ : IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.radical R L) x)
      I : LieIdeal R L
      h₂ : LieAlgebra.IsSolvable R (Subtype fun x => Membership.mem I x)
      ⊢ IsLieAbelian (Subtype fun x => Membership.mem I x)
    -/
    rw [LieIdeal.solvable_iff_le_radical] at h₂
    /-
      case mp
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsNoetherian R L
      h₁ : IsLieAbelian (Subtype fun x => Membership.mem (LieAlgebra.radical R L) x)
      I : LieIdeal R L
      h₂ : LE.le I (LieAlgebra.radical R L)
      ⊢ IsLieAbelian (Subtype fun x => Membership.mem I x)
    -/
    exact (LieIdeal.inclusion_injective h₂).isLieAbelian h₁
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsNoetherian R L
      ⊢ (∀ (I : LieIdeal R L), LieAlgebra.IsSolvable R (Subtype fun x => Membership. …
    -/
  · intro h; apply h; infer_instance
                      /-
                        🎉 no goals
                      -/


                                                                                            /-
                                                                                              R : Type u_1
                                                                                              L : Type u_2
                                                                                              inst✝³ : CommRing R
                                                                                              inst✝² : LieRing L
                                                                                              inst✝¹ : LieAlgebra R L
                                                                                              inst✝ : LieAlgebra.HasTrivialRadical R L
                                                                                              ⊢ Eq (LieAlgebra.ad R L).ker Bot.bot
                                                                                            -/
theorem ad_ker_eq_bot_of_hasTrivialRadical [HasTrivialRadical R L] : (ad R L).ker = ⊥ := by simp
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


