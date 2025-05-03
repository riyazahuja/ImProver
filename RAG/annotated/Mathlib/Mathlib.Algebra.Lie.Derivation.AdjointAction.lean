/-- The adjoint action of a Lie algebra `L` on itself, seen as a morphism of Lie algebras from
`L` to its derivations.
Note the minus sign: this is chosen to so that `ad ⁅x, y⁆ = ⁅ad x, ad y⁆`. -/
@[simps!]
def ad : L →ₗ⁅R⁆ LieDerivation R L L :=
  { __ := - inner R L L
    map_lie' := by
      /-
        R : Type u_1
        L : Type u_2
        inst✝² : CommRing R
        inst✝¹ : LieRing L
        inst✝ : LieAlgebra R L
        ⊢ ∀ {x y : L}, Eq (__spread✝⁻⁰.toFun (Bracket.bracket x y)) (Bracket.bracket ( …
      -/
      intro x y
      /-
        R : Type u_1
        L : Type u_2
        inst✝² : CommRing R
        inst✝¹ : LieRing L
        inst✝ : LieAlgebra R L
        x y : L
        ⊢ Eq (__spread✝⁻⁰.toFun (Bracket.bracket x y)) (Bracket.bracket (__spread✝⁻⁰.t …
      -/
      ext z
      simp only [AddHom.toFun_eq_coe, LinearMap.coe_toAddHom, LinearMap.neg_apply, coe_neg,
        Pi.neg_apply, inner_apply_apply, commutator_apply]
      /-
        case H
        R : Type u_1
        L : Type u_2
        inst✝² : CommRing R
        inst✝¹ : LieRing L
        inst✝ : LieAlgebra R L
        x y z : L
        ⊢ Eq (Neg.neg (Bracket.bracket z (Bracket.bracket x y))) (HSub.hSub (Neg.neg ( …
      -/
      rw [leibniz_lie, neg_lie, neg_lie, ← lie_skew x]
      /-
        case H
        R : Type u_1
        L : Type u_2
        inst✝² : CommRing R
        inst✝¹ : LieRing L
        inst✝ : LieAlgebra R L
        x y z : L
        ⊢ Eq (Neg.neg (HAdd.hAdd (Bracket.bracket (Bracket.bracket z x) y) (Neg.neg (B …
      -/
      /-
        🎉 no goals
      -/
      abel }
      /-
        🎉 no goals
      -/


/-- The definitions `LieDerivation.ad` and `LieAlgebra.ad` agree. -/
                                                                                      /-
                                                                                        R : Type u_1
                                                                                        L : Type u_2
                                                                                        inst✝² : CommRing R
                                                                                        inst✝¹ : LieRing L
                                                                                        inst✝ : LieAlgebra R L
                                                                                        x : L
                                                                                        ⊢ Eq (↑((LieDerivation.ad R L) x)) ((LieAlgebra.ad R L) x)
                                                                                      -/
@[simp] lemma coe_ad_apply_eq_ad_apply (x : L) : ad R L x = LieAlgebra.ad R L x := by ext; simp
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


lemma ad_apply_lieDerivation (x : L) (D : LieDerivation R L L) : ad R L (D x) = - ⁅x, D⁆ := rfl


                                                                              /-
                                                                                R : Type u_1
                                                                                L : Type u_2
                                                                                inst✝² : CommRing R
                                                                                inst✝¹ : LieRing L
                                                                                inst✝ : LieAlgebra R L
                                                                                x : L
                                                                                D : LieDerivation R L L
                                                                                ⊢ Eq (Bracket.bracket ((LieDerivation.ad R L) x) D) (Bracket.bracket x D)
                                                                              -/
lemma lie_ad (x : L) (D : LieDerivation R L L) : ⁅ad R L x, D⁆ = ⁅x, D⁆ := by ext; simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


variable (R L) in
/-- The kernel of the adjoint action on a Lie algebra is equal to its center. -/
lemma ad_ker_eq_center : (ad R L).ker = LieAlgebra.center R L := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    ⊢ Eq (LieDerivation.ad R L).ker (LieAlgebra.center R L)
  -/
  ext x
  /-
    case h
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x : L
    ⊢ Iff (Membership.mem (LieDerivation.ad R L).ker x) (Membership.mem (LieAlgebr …
  -/
  rw [← LieAlgebra.self_module_ker_eq_center, LieHom.mem_ker, LieModule.mem_ker]
  /-
    case h
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x : L
    ⊢ Iff (Eq ((LieDerivation.ad R L) x) 0) (∀ (m : L), Eq (Bracket.bracket x m) 0)
  -/
  simp [DFunLike.ext_iff]
  /-
    🎉 no goals
  -/


/-- If the center of a Lie algebra is trivial, then the adjoint action is injective. -/
lemma injective_ad_of_center_eq_bot (h : LieAlgebra.center R L = ⊥) :
    Function.Injective (ad R L) := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    h : Eq (LieAlgebra.center R L) Bot.bot
    ⊢ Function.Injective ⇑(LieDerivation.ad R L)
  -/
  rw [← LieHom.ker_eq_bot, ad_ker_eq_center, h]
  /-
    🎉 no goals
  -/


/-- The commutator of a derivation `D` and a derivation of the form `ad x` is `ad (D x)`. -/
lemma lie_der_ad_eq_ad_der (D : LieDerivation R L L) (x : L) : ⁅D, ad R L x⁆ = ad R L (D x) := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    D : LieDerivation R L L
    x : L
    ⊢ Eq (Bracket.bracket D ((LieDerivation.ad R L) x)) ((LieDerivation.ad R L) (D …
  -/
  rw [ad_apply_lieDerivation, ← lie_ad, lie_skew]
  /-
    🎉 no goals
  -/


variable (R L) in
/-- The range of the adjoint action homomorphism from a Lie algebra `L` to the Lie algebra of its
derivations is an ideal of the latter. -/
lemma ad_isIdealMorphism : (ad R L).IsIdealMorphism := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    ⊢ (LieDerivation.ad R L).IsIdealMorphism
  -/
  simp_rw [LieHom.isIdealMorphism_iff, lie_der_ad_eq_ad_der]
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    ⊢ ∀ (x : LieDerivation R L L) (y : L), Exists fun z => Eq ((LieDerivation.ad R …
  -/
  tauto
  /-
    🎉 no goals
  -/


/-- A derivation `D` belongs to the ideal range of the adjoint action iff it is of the form `ad x`
for some `x` in the Lie algebra `L`. -/
lemma mem_ad_idealRange_iff {D : LieDerivation R L L} :
    D ∈ (ad R L).idealRange ↔ ∃ x : L, ad R L x = D :=
  (ad R L).mem_idealRange_iff (ad_isIdealMorphism R L)


lemma maxTrivSubmodule_eq_bot_of_center_eq_bot (h : LieAlgebra.center R L = ⊥) :
    LieModule.maxTrivSubmodule R L (LieDerivation R L L) = ⊥ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    h : Eq (LieAlgebra.center R L) Bot.bot
    ⊢ Eq (LieModule.maxTrivSubmodule R L (LieDerivation R L L)) Bot.bot
  -/
  refine (LieSubmodule.eq_bot_iff _).mpr fun D hD ↦ ext fun x ↦ ?_
  have : ad R L (D x) = 0 := by
    rw [LieModule.mem_maxTrivSubmodule] at hD
    simp [ad_apply_lieDerivation, hD]
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    h : Eq (LieAlgebra.center R L) Bot.bot
    D : LieDerivation R L L
    hD : Membership.mem (LieModule.maxTrivSubmodule R L (LieDerivation R L L)) D
    x : L
    this : Eq ((LieDerivation.ad R L) (D x)) 0
    ⊢ Eq (D x) (0 x)
  -/
  rw [← LieHom.mem_ker, ad_ker_eq_center, h, LieSubmodule.mem_bot] at this
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    h : Eq (LieAlgebra.center R L) Bot.bot
    D : LieDerivation R L L
    hD : Membership.mem (LieModule.maxTrivSubmodule R L (LieDerivation R L L)) D
    x : L
    this : Eq (D x) 0
    ⊢ Eq (D x) (0 x)
  -/
  simp [this]
  /-
    🎉 no goals
  -/


