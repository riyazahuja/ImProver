/-- A typeclass encoding the fact that a given Lie module has linear weights, vanishing on the
derived ideal. -/
class LinearWeights [LieAlgebra.IsNilpotent R L] : Prop where
  map_add : ∀ χ : L → R, genWeightSpace M χ ≠ ⊥ → ∀ x y, χ (x + y) = χ x + χ y
  map_smul : ∀ χ : L → R, genWeightSpace M χ ≠ ⊥ → ∀ (t : R) x, χ (t • x) = t • χ x
  map_lie : ∀ χ : L → R, genWeightSpace M χ ≠ ⊥ → ∀ x y : L, χ ⁅x, y⁆ = 0


/-- A weight of a Lie module, bundled as a linear map. -/
@[simps]
def toLinear : L →ₗ[R] R where
  toFun := χ
  map_add' := LinearWeights.map_add χ χ.genWeightSpace_ne_bot
  map_smul' := LinearWeights.map_smul χ χ.genWeightSpace_ne_bot


instance instCoeLinearMap : CoeOut (Weight R L M) (L →ₗ[R] R) where
  coe := Weight.toLinear R L M


instance instLinearMapClass : LinearMapClass (Weight R L M) R L R where
  map_add χ := LinearWeights.map_add χ χ.genWeightSpace_ne_bot
  map_smulₛₗ χ := LinearWeights.map_smul χ χ.genWeightSpace_ne_bot


@[simp]
lemma apply_lie (x y : L) :
    χ ⁅x, y⁆ = 0 :=
  LinearWeights.map_lie χ χ.genWeightSpace_ne_bot x y


@[simp] lemma coe_coe : (↑(χ : L →ₗ[R] R) : L → R) = (χ : L → R) := rfl


@[simp] lemma coe_toLinear_eq_zero_iff : (χ : L →ₗ[R] R) = 0 ↔ χ.IsZero :=
                                                              /-
                                                                R : Type u_2
                                                                L : Type u_3
                                                                M : Type u_4
                                                                inst✝⁸ : CommRing R
                                                                inst✝⁷ : LieRing L
                                                                inst✝⁶ : LieAlgebra R L
                                                                inst✝⁵ : AddCommGroup M
                                                                inst✝⁴ : Module R M
                                                                inst✝³ : LieRingModule L M
                                                                inst✝² : LieModule R L M
                                                                inst✝¹ : LieAlgebra.IsNilpotent R L
                                                                inst✝ : LieModule.LinearWeights R L M
                                                                χ : LieModule.Weight R L M
                                                                h : χ.IsZero
                                                                ⊢ Eq (LieModule.Weight.toLinear R L M χ) 0
                                                              -/
  ⟨fun h ↦ funext fun x ↦ LinearMap.congr_fun h x, fun h ↦ by ext; simp [h.eq]⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


                                                                         /-
                                                                           R : Type u_2
                                                                           L : Type u_3
                                                                           M : Type u_4
                                                                           inst✝⁸ : CommRing R
                                                                           inst✝⁷ : LieRing L
                                                                           inst✝⁶ : LieAlgebra R L
                                                                           inst✝⁵ : AddCommGroup M
                                                                           inst✝⁴ : Module R M
                                                                           inst✝³ : LieRingModule L M
                                                                           inst✝² : LieModule R L M
                                                                           inst✝¹ : LieAlgebra.IsNilpotent R L
                                                                           inst✝ : LieModule.LinearWeights R L M
                                                                           χ : LieModule.Weight R L M
                                                                           ⊢ Iff (Ne (LieModule.Weight.toLinear R L M χ) 0) χ.IsNonZero
                                                                         -/
lemma coe_toLinear_ne_zero_iff : (χ : L →ₗ[R] R) ≠ 0 ↔ χ.IsNonZero := by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- The kernel of a weight of a Lie module with linear weights. -/
abbrev ker := LinearMap.ker (χ : L →ₗ[R] R)


/-- For an Abelian Lie algebra, the weights of any Lie module are linear. -/
instance instLinearWeightsOfIsLieAbelian [IsLieAbelian L] [NoZeroSMulDivisors R M] :
    LinearWeights R L M :=
  have aux : ∀ (χ : L → R), genWeightSpace M χ ≠ ⊥ → ∀ (x y : L), χ (x + y) = χ x + χ y := by
    have h : ∀ x y, Commute (toEnd R L M x) (toEnd R L M y) := fun x y ↦ by
      rw [commute_iff_lie_eq, ← LieHom.map_lie, trivial_lie_zero, LieHom.map_zero]
    /-
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : IsLieAbelian L
      inst✝ : NoZeroSMulDivisors R M
      h : ∀ (x y : L), Commute ((LieModule.toEnd R L M) x) ((LieModule.toEnd R L M) y)
      ⊢ ∀ (χ : L → R), Ne (LieModule.genWeightSpace M χ) Bot.bot → ∀ (x y : L), Eq ( …
    -/
    intro χ hχ x y
    simp_rw [Ne, ← LieSubmodule.toSubmodule_inj, genWeightSpace, genWeightSpaceOf,
      LieSubmodule.iInf_toSubmodule, LieSubmodule.bot_toSubmodule] at hχ
    exact Module.End.map_add_of_iInf_genEigenspace_ne_bot_of_commute
      (toEnd R L M).toLinearMap χ _ hχ h x y
  { map_add := aux
    map_smul := fun χ hχ t x ↦ by
      simp_rw [Ne, ← LieSubmodule.toSubmodule_inj, genWeightSpace, genWeightSpaceOf,
        LieSubmodule.iInf_toSubmodule, LieSubmodule.bot_toSubmodule] at hχ
      exact Module.End.map_smul_of_iInf_genEigenspace_ne_bot
        (toEnd R L M).toLinearMap χ _ hχ t x
    map_lie := fun χ hχ t x ↦ by
      /-
        k : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : LieRingModule L M
        inst✝² : LieModule R L M
        inst✝¹ : IsLieAbelian L
        inst✝ : NoZeroSMulDivisors R M
        aux : ∀ (χ : L → R), Ne (LieModule.genWeightSpace M χ) Bot.bot → ∀ (x y : L),  …
        χ : L → R
        hχ : Ne (LieModule.genWeightSpace M χ) Bot.bot
        t x : L
        ⊢ Eq (χ (Bracket.bracket t x)) 0
      -/
      rw [trivial_lie_zero, ← add_left_inj (χ 0), ← aux χ hχ, zero_add, zero_add] }
      /-
        🎉 no goals
      -/


lemma trace_comp_toEnd_genWeightSpace_eq (χ : L → R) :
    LinearMap.trace R _ ∘ₗ (toEnd R L (genWeightSpace M χ)).toLinearMap =
    finrank R (genWeightSpace M χ) • χ := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : L → R
    ⊢ Eq (⇑((LinearMap.trace R (Subtype fun x => Membership.mem (LieModule.genWeig …
  -/
  ext x
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : L → R
    x : L
    ⊢ Eq (((LinearMap.trace R (Subtype fun x => Membership.mem (LieModule.genWeigh …
  -/
  let n := toEnd R L (genWeightSpace M χ) x - χ x • LinearMap.id
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : L → R
    x : L
    n : Module.End R (Subtype fun x => Membership.mem (LieModule.genWeightSpace M  …
    ⊢ Eq (((LinearMap.trace R (Subtype fun x => Membership.mem (LieModule.genWeigh …
  -/
  have h₁ : toEnd R L (genWeightSpace M χ) x = n + χ x • LinearMap.id := eq_add_of_sub_eq rfl
  have h₂ : LinearMap.trace R _ n = 0 := IsReduced.eq_zero _ <|
    LinearMap.isNilpotent_trace_of_isNilpotent <| isNilpotent_toEnd_sub_algebraMap M χ x
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : L → R
    x : L
    n : Module.End R (Subtype fun x => Membership.mem (LieModule.genWeightSpace M  …
    h₁ : Eq ((LieModule.toEnd R L (Subtype fun x => Membership.mem (LieModule.genW …
    h₂ : Eq ((LinearMap.trace R (Subtype fun x => Membership.mem (LieModule.genWei …
    ⊢ Eq (((LinearMap.trace R (Subtype fun x => Membership.mem (LieModule.genWeigh …
  -/
  rw [LinearMap.comp_apply, LieHom.coe_toLinearMap, h₁, map_add, h₂]
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : L → R
    x : L
    n : Module.End R (Subtype fun x => Membership.mem (LieModule.genWeightSpace M  …
    h₁ : Eq ((LieModule.toEnd R L (Subtype fun x => Membership.mem (LieModule.genW …
    h₂ : Eq ((LinearMap.trace R (Subtype fun x => Membership.mem (LieModule.genWei …
    ⊢ Eq (HAdd.hAdd 0 ((LinearMap.trace R (Subtype fun x => Membership.mem (LieMod …
  -/
  simp [mul_comm (χ x)]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-06")]
alias trace_comp_toEnd_weight_space_eq := trace_comp_toEnd_genWeightSpace_eq


variable {R L M} in
lemma zero_lt_finrank_genWeightSpace {χ : L → R} (hχ : genWeightSpace M χ ≠ ⊥) :
    0 < finrank R (genWeightSpace M χ) := by
  rwa [← LieSubmodule.nontrivial_iff_ne_bot, ← rank_pos_iff_nontrivial (R := R), ← finrank_eq_rank,
    Nat.cast_pos] at hχ


/-- In characteristic zero, the weights of any finite-dimensional Lie module are linear and vanish
on the derived ideal. -/
instance instLinearWeightsOfCharZero [CharZero R] :
    LinearWeights R L M where
  map_add χ hχ x y := by
    rw [← smul_right_inj (zero_lt_finrank_genWeightSpace hχ).ne', smul_add, ← Pi.smul_apply,
      ← Pi.smul_apply, ← Pi.smul_apply, ← trace_comp_toEnd_genWeightSpace_eq, map_add]
  map_smul χ hχ t x := by
    rw [← smul_right_inj (zero_lt_finrank_genWeightSpace hχ).ne', smul_comm, ← Pi.smul_apply,
      ← Pi.smul_apply (finrank R _), ← trace_comp_toEnd_genWeightSpace_eq, map_smul]
  map_lie χ hχ x y := by
    rw [← smul_right_inj (zero_lt_finrank_genWeightSpace hχ).ne', nsmul_zero, ← Pi.smul_apply,
      ← trace_comp_toEnd_genWeightSpace_eq, LinearMap.comp_apply, LieHom.coe_toLinearMap,
      LieHom.map_lie, Ring.lie_def, map_sub, LinearMap.trace_mul_comm, sub_self]


/-- A type synonym for the `χ`-weight space but with the action of `x : L`
on `m : genWeightSpace M χ`, shifted to act as `⁅x, m⁆ - χ x • m`. -/
def shiftedGenWeightSpace := genWeightSpace M χ


private lemma aux [h : Nontrivial (shiftedGenWeightSpace R L M χ)] : genWeightSpace M χ ≠ ⊥ :=
  (LieSubmodule.nontrivial_iff_ne_bot _ _ _).mp h


instance : LieRingModule L (shiftedGenWeightSpace R L M χ) where
  bracket x m := ⁅x, m⁆ - χ x • m
  add_lie x y m := by
    /-
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      x y : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      ⊢ Eq (Bracket.bracket (HAdd.hAdd x y) m) (HAdd.hAdd (Bracket.bracket x m) (Bra …
    -/
    nontriviality shiftedGenWeightSpace R L M χ
    /-
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      x y : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      a✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
      ⊢ Eq (Bracket.bracket (HAdd.hAdd x y) m) (HAdd.hAdd (Bracket.bracket x m) (Bra …
    -/
    simp only [add_lie, LinearWeights.map_add χ (aux R L M χ), add_smul]
    /-
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      x y : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      a✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
      ⊢ Eq (HSub.hSub (HAdd.hAdd (Bracket.bracket x m) (Bracket.bracket y m)) (HAdd. …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/
  lie_add x m n := by
    /-
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      x : L
      m n : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ …
      ⊢ Eq (Bracket.bracket x (HAdd.hAdd m n)) (HAdd.hAdd (Bracket.bracket x m) (Bra …
    -/
    nontriviality shiftedGenWeightSpace R L M χ
    /-
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      x : L
      m n : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ …
      a✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
      ⊢ Eq (Bracket.bracket x (HAdd.hAdd m n)) (HAdd.hAdd (Bracket.bracket x m) (Bra …
    -/
    simp only [lie_add, LinearWeights.map_add χ (aux R L M χ), smul_add]
    /-
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      x : L
      m n : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ …
      a✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
      ⊢ Eq (HSub.hSub (HAdd.hAdd (Bracket.bracket x m) (Bracket.bracket x n)) (HAdd. …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/
  leibniz_lie x y m := by
    /-
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      x y : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      ⊢ Eq (Bracket.bracket x (Bracket.bracket y m)) (HAdd.hAdd (Bracket.bracket (Br …
    -/
    nontriviality shiftedGenWeightSpace R L M χ
    simp only [lie_sub, lie_smul, lie_lie, LinearWeights.map_lie χ (aux R L M χ), zero_smul,
      sub_zero, smul_sub, smul_comm (χ x)]
    /-
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      x y : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      a✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
      ⊢ Eq (HSub.hSub (HSub.hSub (Bracket.bracket x (Bracket.bracket y m)) (HSMul.hS …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/


@[simp] lemma coe_lie_shiftedGenWeightSpace_apply (x : L) (m : shiftedGenWeightSpace R L M χ) :
    letI : Bracket L (shiftedGenWeightSpace R L M χ) := LieRingModule.toBracket
    ⁅x, m⁆ = ⁅x, (m : M)⁆ - χ x • m :=
  rfl


instance : LieModule R L (shiftedGenWeightSpace R L M χ) where
  smul_lie t x m := by
    /-
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      t : R
      x : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      ⊢ Eq (Bracket.bracket (HSMul.hSMul t x) m) (HSMul.hSMul t (Bracket.bracket x m))
    -/
    nontriviality shiftedGenWeightSpace R L M χ
    /-
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      t : R
      x : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      a✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
      ⊢ Eq (Bracket.bracket (HSMul.hSMul t x) m) (HSMul.hSMul t (Bracket.bracket x m))
    -/
    apply Subtype.ext
    /-
      case a
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      t : R
      x : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      a✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
      ⊢ Eq ↑(Bracket.bracket (HSMul.hSMul t x) m) ↑(HSMul.hSMul t (Bracket.bracket x …
    -/
    rw [coe_lie_shiftedGenWeightSpace_apply]
    /-
      case a
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      t : R
      x : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      a✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
      ⊢ Eq (HSub.hSub (Bracket.bracket (HSMul.hSMul t x) ↑m) (HSMul.hSMul (χ (HSMul. …
    -/
    simp only [smul_lie, LinearWeights.map_smul χ (aux R L M χ), smul_assoc t, SetLike.val_smul]
    /-
      case a
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      t : R
      x : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      a✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
      ⊢ Eq (HSub.hSub (HSMul.hSMul t (Bracket.bracket x ↑m)) (HSMul.hSMul t (HSMul.h …
    -/
    rw [← smul_sub]
    /-
      case a
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      t : R
      x : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      a✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
      ⊢ Eq (HSMul.hSMul t (HSub.hSub (Bracket.bracket x ↑m) (HSMul.hSMul (χ x) ↑m))) …
    -/
    congr
    /-
      🎉 no goals
    -/
  lie_smul t x m := by
    /-
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      t : R
      x : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      ⊢ Eq (Bracket.bracket x (HSMul.hSMul t m)) (HSMul.hSMul t (Bracket.bracket x m))
    -/
    nontriviality shiftedGenWeightSpace R L M χ
    /-
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      t : R
      x : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      a✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
      ⊢ Eq (Bracket.bracket x (HSMul.hSMul t m)) (HSMul.hSMul t (Bracket.bracket x m))
    -/
    apply Subtype.ext
    /-
      case a
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      t : R
      x : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      a✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
      ⊢ Eq ↑(Bracket.bracket x (HSMul.hSMul t m)) ↑(HSMul.hSMul t (Bracket.bracket x …
    -/
    rw [coe_lie_shiftedGenWeightSpace_apply]
    /-
      case a
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      t : R
      x : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      a✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
      ⊢ Eq (HSub.hSub (Bracket.bracket x ↑(HSMul.hSMul t m)) (HSMul.hSMul (χ x) ↑(HS …
    -/
    simp only [SetLike.val_smul, lie_smul]
    /-
      case a
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      t : R
      x : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      a✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
      ⊢ Eq (HSub.hSub (HSMul.hSMul t (Bracket.bracket x ↑m)) (HSMul.hSMul (χ x) (HSM …
    -/
    rw [smul_comm (χ x), ← smul_sub]
    /-
      case a
      k : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      χ : L → R
      inst✝ : LieModule.LinearWeights R L M
      t : R
      x : L
      m : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
      a✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
      ⊢ Eq (HSMul.hSMul t (HSub.hSub (Bracket.bracket x ↑m) (HSMul.hSMul (χ x) ↑m))) …
    -/
    congr
    /-
      🎉 no goals
    -/


/-- Forgetting the action of `L`,
the spaces `genWeightSpace M χ` and `shiftedGenWeightSpace R L M χ` are equivalent. -/
@[simps!] def shift : genWeightSpace M χ ≃ₗ[R] shiftedGenWeightSpace R L M χ := LinearEquiv.refl R _


lemma toEnd_eq (x : L) :
    toEnd R L (shiftedGenWeightSpace R L M χ) x =
    (shift R L M χ).conj (toEnd R L (genWeightSpace M χ) x - χ x • LinearMap.id) := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    χ : L → R
    inst✝ : LieModule.LinearWeights R L M
    x : L
    ⊢ Eq ((LieModule.toEnd R L (Subtype fun x => Membership.mem (LieModule.shifted …
  -/
  ext
  simp only [toEnd_apply_apply, map_sub, LinearEquiv.conj_apply, map_smul, LinearMap.comp_id,
    LinearEquiv.comp_coe, LinearEquiv.symm_trans_self, LinearEquiv.refl_toLinearMap,
    LinearMap.sub_apply, LinearMap.coe_comp, LinearEquiv.coe_coe, Function.comp_apply,
    shift_symm_apply, shift_apply, LinearMap.smul_apply, LinearMap.id_coe, id_eq,
    AddSubgroupClass.coe_sub, SetLike.val_smul]
  /-
    case h.a
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    χ : L → R
    inst✝ : LieModule.LinearWeights R L M
    x : L
    x✝ : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
    ⊢ Eq (↑(Bracket.bracket x x✝)) (HSub.hSub (↑(Bracket.bracket x x✝)) (HSMul.hSM …
  -/
  rw [LieSubmodule.coe_bracket]
  /-
    case h.a
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    χ : L → R
    inst✝ : LieModule.LinearWeights R L M
    x : L
    x✝ : Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSpace R L M χ) x
    ⊢ Eq (↑(Bracket.bracket x x✝)) (HSub.hSub (Bracket.bracket x ↑x✝) (HSMul.hSMul …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- By Engel's theorem, if `M` is Noetherian, the shifted action `⁅x, m⁆ - χ x • m` makes the
`χ`-weight space into a nilpotent Lie module. -/
instance [IsNoetherian R M] : IsNilpotent R L (shiftedGenWeightSpace R L M χ) :=
  LieModule.isNilpotent_iff_forall'.mpr fun x ↦ isNilpotent_toEnd_sub_algebraMap M χ x


open shiftedGenWeightSpace in
/-- Given a Lie module `M` of a Lie algebra `L` with coefficients in `R`, if a function `χ : L → R`
has a simultaneous generalized eigenvector for the action of `L` then it has a simultaneous true
eigenvector, provided `M` is Noetherian and has linear weights. -/
lemma exists_forall_lie_eq_smul [LinearWeights R L M] [IsNoetherian R M] (χ : Weight R L M) :
    ∃ m : M, m ≠ 0 ∧ ∀ x : L, ⁅x, m⁆ = χ x • m := by
  replace hχ : Nontrivial (shiftedGenWeightSpace R L M χ) :=
    (LieSubmodule.nontrivial_iff_ne_bot R L M).mpr χ.genWeightSpace_ne_bot
  obtain ⟨⟨⟨m, _⟩, hm₁⟩, hm₂⟩ :=
    @exists_ne _ (nontrivial_max_triv_of_isNilpotent R L (shiftedGenWeightSpace R L M χ)) 0
  simp_rw [mem_maxTrivSubmodule, Subtype.ext_iff,
    ZeroMemClass.coe_zero] at hm₁
  /-
    case intro.mk.mk
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : LieModule.LinearWeights R L M
    inst✝ : IsNoetherian R M
    χ : LieModule.Weight R L M
    hχ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
    m : M
    property✝ : Membership.mem (LieModule.shiftedGenWeightSpace R L M ⇑χ) m
    hm₁✝ : Membership.mem (LieModule.maxTrivSubmodule R L (Subtype fun x => Member …
    hm₂ : Ne ⟨⟨m, property✝⟩, hm₁✝⟩ 0
    hm₁ : ∀ (x : L), Eq (↑(Bracket.bracket x ⟨m, property✝⟩)) 0
    ⊢ Exists fun m => And (Ne m 0) (∀ (x : L), Eq (Bracket.bracket x m) (HSMul.hSM …
  -/
  refine ⟨m, by simpa [LieSubmodule.mk_eq_zero] using hm₂, ?_⟩
  /-
    case intro.mk.mk
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : LieModule.LinearWeights R L M
    inst✝ : IsNoetherian R M
    χ : LieModule.Weight R L M
    hχ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
    m : M
    property✝ : Membership.mem (LieModule.shiftedGenWeightSpace R L M ⇑χ) m
    hm₁✝ : Membership.mem (LieModule.maxTrivSubmodule R L (Subtype fun x => Member …
    hm₂ : Ne ⟨⟨m, property✝⟩, hm₁✝⟩ 0
    hm₁ : ∀ (x : L), Eq (↑(Bracket.bracket x ⟨m, property✝⟩)) 0
    ⊢ ∀ (x : L), Eq (Bracket.bracket x m) (HSMul.hSMul (χ x) m)
  -/
  intro x
  /-
    case intro.mk.mk
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : LieModule.LinearWeights R L M
    inst✝ : IsNoetherian R M
    χ : LieModule.Weight R L M
    hχ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
    m : M
    property✝ : Membership.mem (LieModule.shiftedGenWeightSpace R L M ⇑χ) m
    hm₁✝ : Membership.mem (LieModule.maxTrivSubmodule R L (Subtype fun x => Member …
    hm₂ : Ne ⟨⟨m, property✝⟩, hm₁✝⟩ 0
    hm₁ : ∀ (x : L), Eq (↑(Bracket.bracket x ⟨m, property✝⟩)) 0
    x : L
    ⊢ Eq (Bracket.bracket x m) (HSMul.hSMul (χ x) m)
  -/
  have := hm₁ x
  /-
    case intro.mk.mk
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : LieModule.LinearWeights R L M
    inst✝ : IsNoetherian R M
    χ : LieModule.Weight R L M
    hχ : Nontrivial (Subtype fun x => Membership.mem (LieModule.shiftedGenWeightSp …
    m : M
    property✝ : Membership.mem (LieModule.shiftedGenWeightSpace R L M ⇑χ) m
    hm₁✝ : Membership.mem (LieModule.maxTrivSubmodule R L (Subtype fun x => Member …
    hm₂ : Ne ⟨⟨m, property✝⟩, hm₁✝⟩ 0
    hm₁ : ∀ (x : L), Eq (↑(Bracket.bracket x ⟨m, property✝⟩)) 0
    x : L
    this : Eq (↑(Bracket.bracket x ⟨m, property✝⟩)) 0
    ⊢ Eq (Bracket.bracket x m) (HSMul.hSMul (χ x) m)
  -/
  rwa [coe_lie_shiftedGenWeightSpace_apply, sub_eq_zero] at this
  /-
    🎉 no goals
  -/


/-- See `LieModule.exists_nontrivial_weightSpace_of_isSolvable` for the variant that
only assumes that `L` is solvable but additionally requires `k` to be of characteristic zero. -/
lemma exists_nontrivial_weightSpace_of_isNilpotent [Field k] [LieAlgebra k L] [Module k M]
    [Module.Finite k M] [LieModule k L M] [LieAlgebra.IsNilpotent k L] [LinearWeights k L M]
    [IsTriangularizable k L M] [Nontrivial M] :
    ∃ χ : Module.Dual k L, Nontrivial (weightSpace M χ) := by
  obtain ⟨χ⟩ : Nonempty (Weight k L M) := by
    by_contra contra
    rw [not_nonempty_iff] at contra
    simpa only [iSup_of_empty, bot_ne_top] using LieModule.iSup_genWeightSpace_eq_top' k L M
  /-
    case intro
    k : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : LieRing L
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : LieRingModule L M
    inst✝⁸ : Field k
    inst✝⁷ : LieAlgebra k L
    inst✝⁶ : Module k M
    inst✝⁵ : Module.Finite k M
    inst✝⁴ : LieModule k L M
    inst✝³ : LieAlgebra.IsNilpotent k L
    inst✝² : LieModule.LinearWeights k L M
    inst✝¹ : LieModule.IsTriangularizable k L M
    inst✝ : Nontrivial M
    χ : LieModule.Weight k L M
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  obtain ⟨m, hm₀, hm⟩ := exists_forall_lie_eq_smul k L M χ
  simp only [LieSubmodule.nontrivial_iff_ne_bot, LieSubmodule.eq_bot_iff, Weight.coe_coe, ne_eq,
    not_forall, Classical.not_imp]
  /-
    case intro.intro.intro
    k : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : LieRing L
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : LieRingModule L M
    inst✝⁸ : Field k
    inst✝⁷ : LieAlgebra k L
    inst✝⁶ : Module k M
    inst✝⁵ : Module.Finite k M
    inst✝⁴ : LieModule k L M
    inst✝³ : LieAlgebra.IsNilpotent k L
    inst✝² : LieModule.LinearWeights k L M
    inst✝¹ : LieModule.IsTriangularizable k L M
    inst✝ : Nontrivial M
    χ : LieModule.Weight k L M
    m : M
    hm₀ : Ne m 0
    hm : ∀ (x : L), Eq (Bracket.bracket x m) (HSMul.hSMul (χ x) m)
    ⊢ Exists fun χ => Exists fun x => Exists fun x_1 => Not (Eq x 0)
  -/
  exact ⟨χ.toLinear, m, by simpa [mem_weightSpace], hm₀⟩
  /-
    🎉 no goals
  -/


