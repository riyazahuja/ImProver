/--
An `R`-algebra `A` is formally unramified if `Ω[A⁄R]` is trivial.

This is equivalent to "for every `R`-algebra, every square-zero ideal
`I : Ideal B` and `f : A →ₐ[R] B ⧸ I`, there exists at most one lift `A →ₐ[R] B`".
See `Algebra.FormallyUnramified.iff_comp_injective`.

See <https://stacks.math.columbia.edu/tag/00UM>. -/
@[mk_iff]
class FormallyUnramified : Prop where
  subsingleton_kaehlerDifferential : Subsingleton (Ω[A⁄R])


theorem comp_injective [FormallyUnramified R A] (hI : I ^ 2 = ⊥) :
    Function.Injective ((Ideal.Quotient.mkₐ R I).comp : (A →ₐ[R] B) → A →ₐ[R] B ⧸ I) := by
  /-
    R : Type v
    inst✝⁵ : CommRing R
    A : Type u
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    B : Type w
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    I : Ideal B
    inst✝ : Algebra.FormallyUnramified R A
    hI : Eq (HPow.hPow I 2) Bot.bot
    ⊢ Function.Injective (Ideal.Quotient.mkₐ R I).comp
  -/
  intro f₁ f₂ e
  /-
    R : Type v
    inst✝⁵ : CommRing R
    A : Type u
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    B : Type w
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    I : Ideal B
    inst✝ : Algebra.FormallyUnramified R A
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R A B
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    ⊢ Eq f₁ f₂
  -/
  letI := f₁.toRingHom.toAlgebra
  /-
    R : Type v
    inst✝⁵ : CommRing R
    A : Type u
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    B : Type w
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    I : Ideal B
    inst✝ : Algebra.FormallyUnramified R A
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R A B
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    this : Algebra A B := f₁.toAlgebra
    ⊢ Eq f₁ f₂
  -/
  haveI := IsScalarTower.of_algebraMap_eq' f₁.comp_algebraMap.symm
  have :=
    ((KaehlerDifferential.linearMapEquivDerivation R A).toEquiv.trans
          (derivationToSquareZeroEquivLift I hI)).surjective.subsingleton
  /-
    R : Type v
    inst✝⁵ : CommRing R
    A : Type u
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    B : Type w
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    I : Ideal B
    inst✝ : Algebra.FormallyUnramified R A
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R A B
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    this✝¹ : Algebra A B := f₁.toAlgebra
    this✝ : IsScalarTower R A B
    this : Subsingleton (Subtype fun f => Eq ((Ideal.Quotient.mkₐ R I).comp f) (Is …
    ⊢ Eq f₁ f₂
  -/
  exact Subtype.ext_iff.mp (@Subsingleton.elim _ this ⟨f₁, rfl⟩ ⟨f₂, e.symm⟩)
  /-
    🎉 no goals
  -/


theorem iff_comp_injective :
    FormallyUnramified R A ↔
      ∀ ⦃B : Type u⦄ [CommRing B],
        ∀ [Algebra R B] (I : Ideal B) (_ : I ^ 2 = ⊥),
          Function.Injective ((Ideal.Quotient.mkₐ R I).comp : (A →ₐ[R] B) → A →ₐ[R] B ⧸ I) := by
  /-
    R : Type v
    inst✝² : CommRing R
    A : Type u
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Iff (Algebra.FormallyUnramified R A) (∀ ⦃B : Type u⦄ [inst : CommRing B] [in …
  -/
  constructor
    /-
      case mp
      R : Type v
      inst✝² : CommRing R
      A : Type u
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      ⊢ Algebra.FormallyUnramified R A → ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1  …
    -/
  · intros; exact comp_injective _ ‹_›
            /-
              🎉 no goals
            -/
    /-
      case mpr
      R : Type v
      inst✝² : CommRing R
      A : Type u
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      ⊢ (∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), Eq …
    -/
  · intro H
    /-
      case mpr
      R : Type v
      inst✝² : CommRing R
      A : Type u
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      H : ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), E …
      ⊢ Algebra.FormallyUnramified R A
    -/
    constructor
    /-
      case mpr.subsingleton_kaehlerDifferential
      R : Type v
      inst✝² : CommRing R
      A : Type u
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      H : ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), E …
      ⊢ Subsingleton (KaehlerDifferential R A)
    -/
    rw [← not_nontrivial_iff_subsingleton]
    /-
      case mpr.subsingleton_kaehlerDifferential
      R : Type v
      inst✝² : CommRing R
      A : Type u
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      H : ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), E …
      ⊢ Not (Nontrivial (KaehlerDifferential R A))
    -/
    intro h
    /-
      case mpr.subsingleton_kaehlerDifferential
      R : Type v
      inst✝² : CommRing R
      A : Type u
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      H : ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), E …
      h : Nontrivial (KaehlerDifferential R A)
      ⊢ False
    -/
    obtain ⟨f₁, f₂, e⟩ := (KaehlerDifferential.endEquiv R A).injective.nontrivial
    /-
      case mpr.subsingleton_kaehlerDifferential.mk.intro.intro
      R : Type v
      inst✝² : CommRing R
      A : Type u
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      H : ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), E …
      h : Nontrivial (KaehlerDifferential R A)
      f₁ f₂ : Subtype fun f => Eq ((Algebra.TensorProduct.lmul' R).kerSquareLift.com …
      e : Ne f₁ f₂
      ⊢ False
    -/
    apply e
    /-
      case mpr.subsingleton_kaehlerDifferential.mk.intro.intro
      R : Type v
      inst✝² : CommRing R
      A : Type u
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      H : ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), E …
      h : Nontrivial (KaehlerDifferential R A)
      f₁ f₂ : Subtype fun f => Eq ((Algebra.TensorProduct.lmul' R).kerSquareLift.com …
      e : Ne f₁ f₂
      ⊢ Eq f₁ f₂
    -/
    ext1
    refine H
      (RingHom.ker (TensorProduct.lmul' R (S := A)).kerSquareLift.toRingHom) ?_ ?_
      /-
        case mpr.subsingleton_kaehlerDifferential.mk.intro.intro.a.refine_1
        R : Type v
        inst✝² : CommRing R
        A : Type u
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        H : ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), E …
        h : Nontrivial (KaehlerDifferential R A)
        f₁ f₂ : Subtype fun f => Eq ((Algebra.TensorProduct.lmul' R).kerSquareLift.com …
        e : Ne f₁ f₂
        ⊢ Eq (HPow.hPow (RingHom.ker (Algebra.TensorProduct.lmul' R).kerSquareLift.toR …
      -/
    · rw [AlgHom.ker_kerSquareLift]
      /-
        case mpr.subsingleton_kaehlerDifferential.mk.intro.intro.a.refine_1
        R : Type v
        inst✝² : CommRing R
        A : Type u
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        H : ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), E …
        h : Nontrivial (KaehlerDifferential R A)
        f₁ f₂ : Subtype fun f => Eq ((Algebra.TensorProduct.lmul' R).kerSquareLift.com …
        e : Ne f₁ f₂
        ⊢ Eq (HPow.hPow (RingHom.ker (Algebra.TensorProduct.lmul' R).toRingHom).cotang …
      -/
      exact Ideal.cotangentIdeal_square _
      /-
        🎉 no goals
      -/
      /-
        case mpr.subsingleton_kaehlerDifferential.mk.intro.intro.a.refine_2
        R : Type v
        inst✝² : CommRing R
        A : Type u
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        H : ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), E …
        h : Nontrivial (KaehlerDifferential R A)
        f₁ f₂ : Subtype fun f => Eq ((Algebra.TensorProduct.lmul' R).kerSquareLift.com …
        e : Ne f₁ f₂
        ⊢ Eq ((Ideal.Quotient.mkₐ R (RingHom.ker (Algebra.TensorProduct.lmul' R).kerSq …
      -/
    · ext x
      /-
        case mpr.subsingleton_kaehlerDifferential.mk.intro.intro.a.refine_2.H
        R : Type v
        inst✝² : CommRing R
        A : Type u
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        H : ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), E …
        h : Nontrivial (KaehlerDifferential R A)
        f₁ f₂ : Subtype fun f => Eq ((Algebra.TensorProduct.lmul' R).kerSquareLift.com …
        e : Ne f₁ f₂
        x : A
        ⊢ Eq (((Ideal.Quotient.mkₐ R (RingHom.ker (Algebra.TensorProduct.lmul' R).kerS …
      -/
      apply RingHom.kerLift_injective (TensorProduct.lmul' R (S := A)).kerSquareLift.toRingHom
      /-
        case mpr.subsingleton_kaehlerDifferential.mk.intro.intro.a.refine_2.H.a
        R : Type v
        inst✝² : CommRing R
        A : Type u
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        H : ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), E …
        h : Nontrivial (KaehlerDifferential R A)
        f₁ f₂ : Subtype fun f => Eq ((Algebra.TensorProduct.lmul' R).kerSquareLift.com …
        e : Ne f₁ f₂
        x : A
        ⊢ Eq ((Algebra.TensorProduct.lmul' R).kerSquareLift.kerLift (((Ideal.Quotient. …
      -/
      simpa using DFunLike.congr_fun (f₁.2.trans f₂.2.symm) x
      /-
        🎉 no goals
      -/


theorem lift_unique
    [FormallyUnramified R A] (I : Ideal B) (hI : IsNilpotent I) (g₁ g₂ : A →ₐ[R] B)
    (h : (Ideal.Quotient.mkₐ R I).comp g₁ = (Ideal.Quotient.mkₐ R I).comp g₂) : g₁ = g₂ := by
  /-
    R : Type v
    inst✝⁵ : CommRing R
    A : Type u
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    B : Type w
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FormallyUnramified R A
    I : Ideal B
    hI : IsNilpotent I
    g₁ g₂ : AlgHom R A B
    h : Eq ((Ideal.Quotient.mkₐ R I).comp g₁) ((Ideal.Quotient.mkₐ R I).comp g₂)
    ⊢ Eq g₁ g₂
  -/
  revert g₁ g₂
  /-
    R : Type v
    inst✝⁵ : CommRing R
    A : Type u
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    B : Type w
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FormallyUnramified R A
    I : Ideal B
    hI : IsNilpotent I
    ⊢ ∀ (g₁ g₂ : AlgHom R A B), Eq ((Ideal.Quotient.mkₐ R I).comp g₁) ((Ideal.Quot …
  -/
  change Function.Injective (Ideal.Quotient.mkₐ R I).comp
  /-
    R : Type v
    inst✝⁵ : CommRing R
    A : Type u
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    B : Type w
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FormallyUnramified R A
    I : Ideal B
    hI : IsNilpotent I
    ⊢ Function.Injective (Ideal.Quotient.mkₐ R I).comp
  -/
  revert ‹Algebra R B›
  /-
    R : Type v
    inst✝⁴ : CommRing R
    A : Type u
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    B : Type w
    inst✝¹ : CommRing B
    inst✝ : Algebra.FormallyUnramified R A
    I : Ideal B
    hI : IsNilpotent I
    ⊢ ∀ [inst : Algebra R B], Function.Injective (Ideal.Quotient.mkₐ R I).comp
  -/
  apply Ideal.IsNilpotent.induction_on (S := B) I hI
    /-
      case h₁
      R : Type v
      inst✝⁴ : CommRing R
      A : Type u
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      B : Type w
      inst✝¹ : CommRing B
      inst✝ : Algebra.FormallyUnramified R A
      I : Ideal B
      hI : IsNilpotent I
      ⊢ ∀ ⦃S : Type w⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bot.bot …
    -/
  · intro B _ I hI _; exact FormallyUnramified.comp_injective I hI
                      /-
                        🎉 no goals
                      -/
    /-
      case h₂
      R : Type v
      inst✝⁴ : CommRing R
      A : Type u
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      B : Type w
      inst✝¹ : CommRing B
      inst✝ : Algebra.FormallyUnramified R A
      I : Ideal B
      hI : IsNilpotent I
      ⊢ ∀ ⦃S : Type w⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → (∀ [inst_1 : …
    -/
  · intro B _ I J hIJ h₁ h₂ _ g₁ g₂ e
    /-
      case h₂
      R : Type v
      inst✝⁶ : CommRing R
      A : Type u
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      B✝ : Type w
      inst✝³ : CommRing B✝
      inst✝² : Algebra.FormallyUnramified R A
      I✝ : Ideal B✝
      hI : IsNilpotent I✝
      B : Type w
      inst✝¹ : CommRing B
      I J : Ideal B
      hIJ : LE.le I J
      h₁ : ∀ [inst : Algebra R B], Function.Injective (Ideal.Quotient.mkₐ R I).comp
      h₂ : ∀ [inst : Algebra R (HasQuotient.Quotient B I)], Function.Injective (Idea …
      inst✝ : Algebra R B
      g₁ g₂ : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R J).comp g₁) ((Ideal.Quotient.mkₐ R J).comp g₂)
      ⊢ Eq g₁ g₂
    -/
    apply h₁
    /-
      case h₂.a
      R : Type v
      inst✝⁶ : CommRing R
      A : Type u
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      B✝ : Type w
      inst✝³ : CommRing B✝
      inst✝² : Algebra.FormallyUnramified R A
      I✝ : Ideal B✝
      hI : IsNilpotent I✝
      B : Type w
      inst✝¹ : CommRing B
      I J : Ideal B
      hIJ : LE.le I J
      h₁ : ∀ [inst : Algebra R B], Function.Injective (Ideal.Quotient.mkₐ R I).comp
      h₂ : ∀ [inst : Algebra R (HasQuotient.Quotient B I)], Function.Injective (Idea …
      inst✝ : Algebra R B
      g₁ g₂ : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R J).comp g₁) ((Ideal.Quotient.mkₐ R J).comp g₂)
      ⊢ Eq ((Ideal.Quotient.mkₐ R I).comp g₁) ((Ideal.Quotient.mkₐ R I).comp g₂)
    -/
    apply h₂
    /-
      case h₂.a.a
      R : Type v
      inst✝⁶ : CommRing R
      A : Type u
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      B✝ : Type w
      inst✝³ : CommRing B✝
      inst✝² : Algebra.FormallyUnramified R A
      I✝ : Ideal B✝
      hI : IsNilpotent I✝
      B : Type w
      inst✝¹ : CommRing B
      I J : Ideal B
      hIJ : LE.le I J
      h₁ : ∀ [inst : Algebra R B], Function.Injective (Ideal.Quotient.mkₐ R I).comp
      h₂ : ∀ [inst : Algebra R (HasQuotient.Quotient B I)], Function.Injective (Idea …
      inst✝ : Algebra R B
      g₁ g₂ : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R J).comp g₁) ((Ideal.Quotient.mkₐ R J).comp g₂)
      ⊢ Eq ((Ideal.Quotient.mkₐ R (Ideal.map (Ideal.Quotient.mk I) J)).comp ((Ideal. …
    -/
    ext x
    /-
      case h₂.a.a.H
      R : Type v
      inst✝⁶ : CommRing R
      A : Type u
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      B✝ : Type w
      inst✝³ : CommRing B✝
      inst✝² : Algebra.FormallyUnramified R A
      I✝ : Ideal B✝
      hI : IsNilpotent I✝
      B : Type w
      inst✝¹ : CommRing B
      I J : Ideal B
      hIJ : LE.le I J
      h₁ : ∀ [inst : Algebra R B], Function.Injective (Ideal.Quotient.mkₐ R I).comp
      h₂ : ∀ [inst : Algebra R (HasQuotient.Quotient B I)], Function.Injective (Idea …
      inst✝ : Algebra R B
      g₁ g₂ : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R J).comp g₁) ((Ideal.Quotient.mkₐ R J).comp g₂)
      x : A
      ⊢ Eq (((Ideal.Quotient.mkₐ R (Ideal.map (Ideal.Quotient.mk I) J)).comp ((Ideal …
    -/
    replace e := AlgHom.congr_fun e x
    /-
      case h₂.a.a.H
      R : Type v
      inst✝⁶ : CommRing R
      A : Type u
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      B✝ : Type w
      inst✝³ : CommRing B✝
      inst✝² : Algebra.FormallyUnramified R A
      I✝ : Ideal B✝
      hI : IsNilpotent I✝
      B : Type w
      inst✝¹ : CommRing B
      I J : Ideal B
      hIJ : LE.le I J
      h₁ : ∀ [inst : Algebra R B], Function.Injective (Ideal.Quotient.mkₐ R I).comp
      h₂ : ∀ [inst : Algebra R (HasQuotient.Quotient B I)], Function.Injective (Idea …
      inst✝ : Algebra R B
      g₁ g₂ : AlgHom R A B
      x : A
      e : Eq (((Ideal.Quotient.mkₐ R J).comp g₁) x) (((Ideal.Quotient.mkₐ R J).comp  …
      ⊢ Eq (((Ideal.Quotient.mkₐ R (Ideal.map (Ideal.Quotient.mk I) J)).comp ((Ideal …
    -/
    dsimp only [AlgHom.comp_apply, Ideal.Quotient.mkₐ_eq_mk] at e ⊢
    /-
      case h₂.a.a.H
      R : Type v
      inst✝⁶ : CommRing R
      A : Type u
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      B✝ : Type w
      inst✝³ : CommRing B✝
      inst✝² : Algebra.FormallyUnramified R A
      I✝ : Ideal B✝
      hI : IsNilpotent I✝
      B : Type w
      inst✝¹ : CommRing B
      I J : Ideal B
      hIJ : LE.le I J
      h₁ : ∀ [inst : Algebra R B], Function.Injective (Ideal.Quotient.mkₐ R I).comp
      h₂ : ∀ [inst : Algebra R (HasQuotient.Quotient B I)], Function.Injective (Idea …
      inst✝ : Algebra R B
      g₁ g₂ : AlgHom R A B
      x : A
      e : Eq ((Ideal.Quotient.mk J) (g₁ x)) ((Ideal.Quotient.mk J) (g₂ x))
      ⊢ Eq ((Ideal.Quotient.mk (Ideal.map (Ideal.Quotient.mk I) J)) ((Ideal.Quotient …
    -/
    rwa [Ideal.Quotient.eq, ← map_sub, Ideal.mem_quotient_iff_mem hIJ, ← Ideal.Quotient.eq]
    /-
      🎉 no goals
    -/


theorem ext [FormallyUnramified R A] (hI : IsNilpotent I) {g₁ g₂ : A →ₐ[R] B}
    (H : ∀ x, Ideal.Quotient.mk I (g₁ x) = Ideal.Quotient.mk I (g₂ x)) : g₁ = g₂ :=
  FormallyUnramified.lift_unique I hI g₁ g₂ (AlgHom.ext H)


theorem lift_unique_of_ringHom [FormallyUnramified R A] {C : Type*} [CommRing C]
    (f : B →+* C) (hf : IsNilpotent <| RingHom.ker f) (g₁ g₂ : A →ₐ[R] B)
    (h : f.comp ↑g₁ = f.comp (g₂ : A →+* B)) : g₁ = g₂ :=
  FormallyUnramified.lift_unique _ hf _ _
    (by
      /-
        R : Type v
        inst✝⁶ : CommRing R
        A : Type u
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        B : Type w
        inst✝³ : CommRing B
        inst✝² : Algebra R B
        inst✝¹ : Algebra.FormallyUnramified R A
        C : Type u_1
        inst✝ : CommRing C
        f : RingHom B C
        hf : IsNilpotent (RingHom.ker f)
        g₁ g₂ : AlgHom R A B
        h : Eq (f.comp ↑g₁) (f.comp ↑g₂)
        ⊢ Eq ((Ideal.Quotient.mkₐ R (RingHom.ker f)).comp g₁) ((Ideal.Quotient.mkₐ R ( …
      -/
      ext x
      /-
        case H
        R : Type v
        inst✝⁶ : CommRing R
        A : Type u
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        B : Type w
        inst✝³ : CommRing B
        inst✝² : Algebra R B
        inst✝¹ : Algebra.FormallyUnramified R A
        C : Type u_1
        inst✝ : CommRing C
        f : RingHom B C
        hf : IsNilpotent (RingHom.ker f)
        g₁ g₂ : AlgHom R A B
        h : Eq (f.comp ↑g₁) (f.comp ↑g₂)
        x : A
        ⊢ Eq (((Ideal.Quotient.mkₐ R (RingHom.ker f)).comp g₁) x) (((Ideal.Quotient.mk …
      -/
      have := RingHom.congr_fun h x
      simpa only [Ideal.Quotient.eq, Function.comp_apply, AlgHom.coe_comp, Ideal.Quotient.mkₐ_eq_mk,
        RingHom.mem_ker, map_sub, sub_eq_zero])


theorem ext' [FormallyUnramified R A] {C : Type*} [CommRing C] (f : B →+* C)
    (hf : IsNilpotent <| RingHom.ker f) (g₁ g₂ : A →ₐ[R] B) (h : ∀ x, f (g₁ x) = f (g₂ x)) :
    g₁ = g₂ :=
  FormallyUnramified.lift_unique_of_ringHom f hf g₁ g₂ (RingHom.ext h)


theorem lift_unique' [FormallyUnramified R A] {C : Type*} [CommRing C]
    [Algebra R C] (f : B →ₐ[R] C) (hf : IsNilpotent <| RingHom.ker (f : B →+* C))
    (g₁ g₂ : A →ₐ[R] B) (h : f.comp g₁ = f.comp g₂) : g₁ = g₂ :=
  FormallyUnramified.ext' _ hf g₁ g₂ (AlgHom.congr_fun h)


instance {R : Type*} [CommRing R] : FormallyUnramified R R := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Algebra.FormallyUnramified R R
  -/
  rw [iff_comp_injective]
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ ∀ ⦃B : Type u_1⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), E …
  -/
  intros B _ _ _ _ f₁ f₂ _
  /-
    R : Type u_1
    inst✝² : CommRing R
    B : Type u_1
    inst✝¹ : CommRing B
    inst✝ : Algebra R B
    I✝ : Ideal B
    x✝ : Eq (HPow.hPow I✝ 2) Bot.bot
    f₁ f₂ : AlgHom R R B
    a✝ : Eq ((Ideal.Quotient.mkₐ R I✝).comp f₁) ((Ideal.Quotient.mkₐ R I✝).comp f₂)
    ⊢ Eq f₁ f₂
  -/
  exact Subsingleton.elim _ _
  /-
    🎉 no goals
  -/


theorem of_equiv [FormallyUnramified R A] (e : A ≃ₐ[R] B) :
    FormallyUnramified R B := by
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FormallyUnramified R A
    e : AlgEquiv R A B
    ⊢ Algebra.FormallyUnramified R B
  -/
  rw [iff_comp_injective]
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FormallyUnramified R A
    e : AlgEquiv R A B
    ⊢ ∀ ⦃B_1 : Type u_3⦄ [inst : CommRing B_1] [inst_1 : Algebra R B_1] (I : Ideal …
  -/
  intro C _ _ I hI f₁ f₂ e'
  /-
    R : Type u_1
    inst✝⁷ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallyUnramified R A
    e : AlgEquiv R A B
    C : Type u_3
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B C
    e' : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    ⊢ Eq f₁ f₂
  -/
  rw [← f₁.comp_id, ← f₂.comp_id, ← e.comp_symm, ← AlgHom.comp_assoc, ← AlgHom.comp_assoc]
  /-
    R : Type u_1
    inst✝⁷ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallyUnramified R A
    e : AlgEquiv R A B
    C : Type u_3
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B C
    e' : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    ⊢ Eq ((f₁.comp ↑e).comp ↑e.symm) ((f₂.comp ↑e).comp ↑e.symm)
  -/
  congr 1
  /-
    case e_φ₁
    R : Type u_1
    inst✝⁷ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallyUnramified R A
    e : AlgEquiv R A B
    C : Type u_3
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B C
    e' : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    ⊢ Eq (f₁.comp ↑e) (f₂.comp ↑e)
  -/
  refine FormallyUnramified.comp_injective I hI ?_
  /-
    case e_φ₁
    R : Type u_1
    inst✝⁷ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallyUnramified R A
    e : AlgEquiv R A B
    C : Type u_3
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B C
    e' : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    ⊢ Eq ((Ideal.Quotient.mkₐ R I).comp (f₁.comp ↑e)) ((Ideal.Quotient.mkₐ R I).co …
  -/
  rw [← AlgHom.comp_assoc, e', AlgHom.comp_assoc]
  /-
    🎉 no goals
  -/


theorem comp [FormallyUnramified R A] [FormallyUnramified A B] :
    FormallyUnramified R B := by
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    A : Type u_2
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    B : Type u_3
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra R B
    inst✝³ : Algebra A B
    inst✝² : IsScalarTower R A B
    inst✝¹ : Algebra.FormallyUnramified R A
    inst✝ : Algebra.FormallyUnramified A B
    ⊢ Algebra.FormallyUnramified R B
  -/
  rw [iff_comp_injective]
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    A : Type u_2
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    B : Type u_3
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra R B
    inst✝³ : Algebra A B
    inst✝² : IsScalarTower R A B
    inst✝¹ : Algebra.FormallyUnramified R A
    inst✝ : Algebra.FormallyUnramified A B
    ⊢ ∀ ⦃B_1 : Type u_3⦄ [inst : CommRing B_1] [inst_1 : Algebra R B_1] (I : Ideal …
  -/
  intro C _ _ I hI f₁ f₂ e
  have e' :=
    FormallyUnramified.lift_unique I ⟨2, hI⟩ (f₁.comp <| IsScalarTower.toAlgHom R A B)
      (f₂.comp <| IsScalarTower.toAlgHom R A B) (by rw [← AlgHom.comp_assoc, e, AlgHom.comp_assoc])
  /-
    R : Type u_1
    inst✝¹⁰ : CommRing R
    A : Type u_2
    inst✝⁹ : CommRing A
    inst✝⁸ : Algebra R A
    B : Type u_3
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R B
    inst✝⁵ : Algebra A B
    inst✝⁴ : IsScalarTower R A B
    inst✝³ : Algebra.FormallyUnramified R A
    inst✝² : Algebra.FormallyUnramified A B
    C : Type u_3
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B C
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    e' : Eq (f₁.comp (IsScalarTower.toAlgHom R A B)) (f₂.comp (IsScalarTower.toAlg …
    ⊢ Eq f₁ f₂
  -/
  letI := (f₁.restrictDomain A).toAlgebra
  /-
    R : Type u_1
    inst✝¹⁰ : CommRing R
    A : Type u_2
    inst✝⁹ : CommRing A
    inst✝⁸ : Algebra R A
    B : Type u_3
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R B
    inst✝⁵ : Algebra A B
    inst✝⁴ : IsScalarTower R A B
    inst✝³ : Algebra.FormallyUnramified R A
    inst✝² : Algebra.FormallyUnramified A B
    C : Type u_3
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B C
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    e' : Eq (f₁.comp (IsScalarTower.toAlgHom R A B)) (f₂.comp (IsScalarTower.toAlg …
    this : Algebra A C := (AlgHom.restrictDomain A f₁).toAlgebra
    ⊢ Eq f₁ f₂
  -/
  let F₁ : B →ₐ[A] C := { f₁ with commutes' := fun r => rfl }
  /-
    R : Type u_1
    inst✝¹⁰ : CommRing R
    A : Type u_2
    inst✝⁹ : CommRing A
    inst✝⁸ : Algebra R A
    B : Type u_3
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R B
    inst✝⁵ : Algebra A B
    inst✝⁴ : IsScalarTower R A B
    inst✝³ : Algebra.FormallyUnramified R A
    inst✝² : Algebra.FormallyUnramified A B
    C : Type u_3
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B C
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    e' : Eq (f₁.comp (IsScalarTower.toAlgHom R A B)) (f₂.comp (IsScalarTower.toAlg …
    this : Algebra A C := (AlgHom.restrictDomain A f₁).toAlgebra
    F₁ : AlgHom A B C := { toRingHom := f₁.toRingHom, commutes' := ⋯ }
    ⊢ Eq f₁ f₂
  -/
  let F₂ : B →ₐ[A] C := { f₂ with commutes' := AlgHom.congr_fun e'.symm }
  /-
    R : Type u_1
    inst✝¹⁰ : CommRing R
    A : Type u_2
    inst✝⁹ : CommRing A
    inst✝⁸ : Algebra R A
    B : Type u_3
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R B
    inst✝⁵ : Algebra A B
    inst✝⁴ : IsScalarTower R A B
    inst✝³ : Algebra.FormallyUnramified R A
    inst✝² : Algebra.FormallyUnramified A B
    C : Type u_3
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B C
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    e' : Eq (f₁.comp (IsScalarTower.toAlgHom R A B)) (f₂.comp (IsScalarTower.toAlg …
    this : Algebra A C := (AlgHom.restrictDomain A f₁).toAlgebra
    F₁ : AlgHom A B C := { toRingHom := f₁.toRingHom, commutes' := ⋯ }
    F₂ : AlgHom A B C := { toRingHom := f₂.toRingHom, commutes' := ⋯ }
    ⊢ Eq f₁ f₂
  -/
  ext1 x
  /-
    case H
    R : Type u_1
    inst✝¹⁰ : CommRing R
    A : Type u_2
    inst✝⁹ : CommRing A
    inst✝⁸ : Algebra R A
    B : Type u_3
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R B
    inst✝⁵ : Algebra A B
    inst✝⁴ : IsScalarTower R A B
    inst✝³ : Algebra.FormallyUnramified R A
    inst✝² : Algebra.FormallyUnramified A B
    C : Type u_3
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B C
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    e' : Eq (f₁.comp (IsScalarTower.toAlgHom R A B)) (f₂.comp (IsScalarTower.toAlg …
    this : Algebra A C := (AlgHom.restrictDomain A f₁).toAlgebra
    F₁ : AlgHom A B C := { toRingHom := f₁.toRingHom, commutes' := ⋯ }
    F₂ : AlgHom A B C := { toRingHom := f₂.toRingHom, commutes' := ⋯ }
    x : B
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  change F₁ x = F₂ x
  /-
    case H
    R : Type u_1
    inst✝¹⁰ : CommRing R
    A : Type u_2
    inst✝⁹ : CommRing A
    inst✝⁸ : Algebra R A
    B : Type u_3
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R B
    inst✝⁵ : Algebra A B
    inst✝⁴ : IsScalarTower R A B
    inst✝³ : Algebra.FormallyUnramified R A
    inst✝² : Algebra.FormallyUnramified A B
    C : Type u_3
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B C
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    e' : Eq (f₁.comp (IsScalarTower.toAlgHom R A B)) (f₂.comp (IsScalarTower.toAlg …
    this : Algebra A C := (AlgHom.restrictDomain A f₁).toAlgebra
    F₁ : AlgHom A B C := { toRingHom := f₁.toRingHom, commutes' := ⋯ }
    F₂ : AlgHom A B C := { toRingHom := f₂.toRingHom, commutes' := ⋯ }
    x : B
    ⊢ Eq (F₁ x) (F₂ x)
  -/
  congr
  /-
    case H.e_a
    R : Type u_1
    inst✝¹⁰ : CommRing R
    A : Type u_2
    inst✝⁹ : CommRing A
    inst✝⁸ : Algebra R A
    B : Type u_3
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R B
    inst✝⁵ : Algebra A B
    inst✝⁴ : IsScalarTower R A B
    inst✝³ : Algebra.FormallyUnramified R A
    inst✝² : Algebra.FormallyUnramified A B
    C : Type u_3
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B C
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    e' : Eq (f₁.comp (IsScalarTower.toAlgHom R A B)) (f₂.comp (IsScalarTower.toAlg …
    this : Algebra A C := (AlgHom.restrictDomain A f₁).toAlgebra
    F₁ : AlgHom A B C := { toRingHom := f₁.toRingHom, commutes' := ⋯ }
    F₂ : AlgHom A B C := { toRingHom := f₂.toRingHom, commutes' := ⋯ }
    x : B
    ⊢ Eq F₁ F₂
  -/
  exact FormallyUnramified.ext I ⟨2, hI⟩ (AlgHom.congr_fun e)
  /-
    🎉 no goals
  -/


theorem of_comp [FormallyUnramified R B] : FormallyUnramified A B := by
  /-
    R : Type u_1
    inst✝⁷ : CommRing R
    A : Type u_2
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    B : Type u_3
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    inst✝² : Algebra A B
    inst✝¹ : IsScalarTower R A B
    inst✝ : Algebra.FormallyUnramified R B
    ⊢ Algebra.FormallyUnramified A B
  -/
  rw [iff_comp_injective]
  /-
    R : Type u_1
    inst✝⁷ : CommRing R
    A : Type u_2
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    B : Type u_3
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    inst✝² : Algebra A B
    inst✝¹ : IsScalarTower R A B
    inst✝ : Algebra.FormallyUnramified R B
    ⊢ ∀ ⦃B_1 : Type u_3⦄ [inst : CommRing B_1] [inst_1 : Algebra A B_1] (I : Ideal …
  -/
  intro Q _ _ I e f₁ f₂ e'
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    A : Type u_2
    inst✝⁸ : CommRing A
    inst✝⁷ : Algebra R A
    B : Type u_3
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : IsScalarTower R A B
    inst✝² : Algebra.FormallyUnramified R B
    Q : Type u_3
    inst✝¹ : CommRing Q
    inst✝ : Algebra A Q
    I : Ideal Q
    e : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom A B Q
    e' : Eq ((Ideal.Quotient.mkₐ A I).comp f₁) ((Ideal.Quotient.mkₐ A I).comp f₂)
    ⊢ Eq f₁ f₂
  -/
  letI := ((algebraMap A Q).comp (algebraMap R A)).toAlgebra
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    A : Type u_2
    inst✝⁸ : CommRing A
    inst✝⁷ : Algebra R A
    B : Type u_3
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : IsScalarTower R A B
    inst✝² : Algebra.FormallyUnramified R B
    Q : Type u_3
    inst✝¹ : CommRing Q
    inst✝ : Algebra A Q
    I : Ideal Q
    e : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom A B Q
    e' : Eq ((Ideal.Quotient.mkₐ A I).comp f₁) ((Ideal.Quotient.mkₐ A I).comp f₂)
    this : Algebra R Q := ((algebraMap A Q).comp (algebraMap R A)).toAlgebra
    ⊢ Eq f₁ f₂
  -/
  letI : IsScalarTower R A Q := IsScalarTower.of_algebraMap_eq' rfl
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    A : Type u_2
    inst✝⁸ : CommRing A
    inst✝⁷ : Algebra R A
    B : Type u_3
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : IsScalarTower R A B
    inst✝² : Algebra.FormallyUnramified R B
    Q : Type u_3
    inst✝¹ : CommRing Q
    inst✝ : Algebra A Q
    I : Ideal Q
    e : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom A B Q
    e' : Eq ((Ideal.Quotient.mkₐ A I).comp f₁) ((Ideal.Quotient.mkₐ A I).comp f₂)
    this✝ : Algebra R Q := ((algebraMap A Q).comp (algebraMap R A)).toAlgebra
    this : IsScalarTower R A Q := IsScalarTower.of_algebraMap_eq' rfl
    ⊢ Eq f₁ f₂
  -/
  refine AlgHom.restrictScalars_injective R ?_
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    A : Type u_2
    inst✝⁸ : CommRing A
    inst✝⁷ : Algebra R A
    B : Type u_3
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : IsScalarTower R A B
    inst✝² : Algebra.FormallyUnramified R B
    Q : Type u_3
    inst✝¹ : CommRing Q
    inst✝ : Algebra A Q
    I : Ideal Q
    e : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom A B Q
    e' : Eq ((Ideal.Quotient.mkₐ A I).comp f₁) ((Ideal.Quotient.mkₐ A I).comp f₂)
    this✝ : Algebra R Q := ((algebraMap A Q).comp (algebraMap R A)).toAlgebra
    this : IsScalarTower R A Q := IsScalarTower.of_algebraMap_eq' rfl
    ⊢ Eq (AlgHom.restrictScalars R f₁) (AlgHom.restrictScalars R f₂)
  -/
  refine FormallyUnramified.ext I ⟨2, e⟩ ?_
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    A : Type u_2
    inst✝⁸ : CommRing A
    inst✝⁷ : Algebra R A
    B : Type u_3
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : IsScalarTower R A B
    inst✝² : Algebra.FormallyUnramified R B
    Q : Type u_3
    inst✝¹ : CommRing Q
    inst✝ : Algebra A Q
    I : Ideal Q
    e : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom A B Q
    e' : Eq ((Ideal.Quotient.mkₐ A I).comp f₁) ((Ideal.Quotient.mkₐ A I).comp f₂)
    this✝ : Algebra R Q := ((algebraMap A Q).comp (algebraMap R A)).toAlgebra
    this : IsScalarTower R A Q := IsScalarTower.of_algebraMap_eq' rfl
    ⊢ ∀ (x : B), Eq ((Ideal.Quotient.mk I) ((AlgHom.restrictScalars R f₁) x)) ((Id …
  -/
  intro x
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    A : Type u_2
    inst✝⁸ : CommRing A
    inst✝⁷ : Algebra R A
    B : Type u_3
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : IsScalarTower R A B
    inst✝² : Algebra.FormallyUnramified R B
    Q : Type u_3
    inst✝¹ : CommRing Q
    inst✝ : Algebra A Q
    I : Ideal Q
    e : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom A B Q
    e' : Eq ((Ideal.Quotient.mkₐ A I).comp f₁) ((Ideal.Quotient.mkₐ A I).comp f₂)
    this✝ : Algebra R Q := ((algebraMap A Q).comp (algebraMap R A)).toAlgebra
    this : IsScalarTower R A Q := IsScalarTower.of_algebraMap_eq' rfl
    x : B
    ⊢ Eq ((Ideal.Quotient.mk I) ((AlgHom.restrictScalars R f₁) x)) ((Ideal.Quotien …
  -/
  exact AlgHom.congr_fun e' x
  /-
    🎉 no goals
  -/


/-- This holds in general for epimorphisms. -/
theorem of_surjective [FormallyUnramified R A] (f : A →ₐ[R] B) (H : Function.Surjective f) :
    FormallyUnramified R B := by
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FormallyUnramified R A
    f : AlgHom R A B
    H : Function.Surjective ⇑f
    ⊢ Algebra.FormallyUnramified R B
  -/
  rw [iff_comp_injective]
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FormallyUnramified R A
    f : AlgHom R A B
    H : Function.Surjective ⇑f
    ⊢ ∀ ⦃B_1 : Type u_3⦄ [inst : CommRing B_1] [inst_1 : Algebra R B_1] (I : Ideal …
  -/
  intro Q _ _ I hI f₁ f₂ e
  /-
    R : Type u_1
    inst✝⁷ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallyUnramified R A
    f : AlgHom R A B
    H : Function.Surjective ⇑f
    Q : Type u_3
    inst✝¹ : CommRing Q
    inst✝ : Algebra R Q
    I : Ideal Q
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B Q
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    ⊢ Eq f₁ f₂
  -/
  ext x
  /-
    case H
    R : Type u_1
    inst✝⁷ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallyUnramified R A
    f : AlgHom R A B
    H : Function.Surjective ⇑f
    Q : Type u_3
    inst✝¹ : CommRing Q
    inst✝ : Algebra R Q
    I : Ideal Q
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B Q
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    x : B
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  obtain ⟨x, rfl⟩ := H x
  /-
    case H.intro
    R : Type u_1
    inst✝⁷ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallyUnramified R A
    f : AlgHom R A B
    H : Function.Surjective ⇑f
    Q : Type u_3
    inst✝¹ : CommRing Q
    inst✝ : Algebra R Q
    I : Ideal Q
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B Q
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    x : A
    ⊢ Eq (f₁ (f x)) (f₂ (f x))
  -/
  rw [← AlgHom.comp_apply, ← AlgHom.comp_apply]
  /-
    case H.intro
    R : Type u_1
    inst✝⁷ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallyUnramified R A
    f : AlgHom R A B
    H : Function.Surjective ⇑f
    Q : Type u_3
    inst✝¹ : CommRing Q
    inst✝ : Algebra R Q
    I : Ideal Q
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B Q
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    x : A
    ⊢ Eq ((f₁.comp f) x) ((f₂.comp f) x)
  -/
  congr 1
  /-
    case H.intro.e_a
    R : Type u_1
    inst✝⁷ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallyUnramified R A
    f : AlgHom R A B
    H : Function.Surjective ⇑f
    Q : Type u_3
    inst✝¹ : CommRing Q
    inst✝ : Algebra R Q
    I : Ideal Q
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B Q
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    x : A
    ⊢ Eq (f₁.comp f) (f₂.comp f)
  -/
  apply FormallyUnramified.comp_injective I hI
  /-
    case H.intro.e_a.a
    R : Type u_1
    inst✝⁷ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallyUnramified R A
    f : AlgHom R A B
    H : Function.Surjective ⇑f
    Q : Type u_3
    inst✝¹ : CommRing Q
    inst✝ : Algebra R Q
    I : Ideal Q
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R B Q
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    x : A
    ⊢ Eq ((Ideal.Quotient.mkₐ R I).comp (f₁.comp f)) ((Ideal.Quotient.mkₐ R I).com …
  -/
  ext x; exact DFunLike.congr_fun e (f x)
         /-
           🎉 no goals
         -/


instance quotient {A} [CommRing A] [Algebra R A] [FormallyUnramified R A] (I : Ideal A) :
    FormallyUnramified R (A ⧸ I) :=
  FormallyUnramified.of_surjective (IsScalarTower.toAlgHom R A (A ⧸ I)) Ideal.Quotient.mk_surjective


theorem iff_of_equiv (e : A ≃ₐ[R] B) : FormallyUnramified R A ↔ FormallyUnramified R B :=
  ⟨fun _ ↦ of_equiv e, fun _ ↦ of_equiv e.symm⟩


instance base_change [FormallyUnramified R A] :
    FormallyUnramified B (B ⊗[R] A) := by
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    A : Type u_2
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    B : Type u_3
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FormallyUnramified R A
    ⊢ Algebra.FormallyUnramified B (TensorProduct R B A)
  -/
  rw [iff_comp_injective]
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    A : Type u_2
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    B : Type u_3
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FormallyUnramified R A
    ⊢ ∀ ⦃B_1 : Type (max u_2 u_3)⦄ [inst : CommRing B_1] [inst_1 : Algebra B B_1]  …
  -/
  intro C _ _ I hI f₁ f₂ e
  /-
    R : Type u_1
    inst✝⁷ : CommRing R
    A : Type u_2
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    B : Type u_3
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallyUnramified R A
    C : Type (max u_2 u_3)
    inst✝¹ : CommRing C
    inst✝ : Algebra B C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom B (TensorProduct R B A) C
    e : Eq ((Ideal.Quotient.mkₐ B I).comp f₁) ((Ideal.Quotient.mkₐ B I).comp f₂)
    ⊢ Eq f₁ f₂
  -/
  letI := ((algebraMap B C).comp (algebraMap R B)).toAlgebra
  /-
    R : Type u_1
    inst✝⁷ : CommRing R
    A : Type u_2
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    B : Type u_3
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallyUnramified R A
    C : Type (max u_2 u_3)
    inst✝¹ : CommRing C
    inst✝ : Algebra B C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom B (TensorProduct R B A) C
    e : Eq ((Ideal.Quotient.mkₐ B I).comp f₁) ((Ideal.Quotient.mkₐ B I).comp f₂)
    this : Algebra R C := ((algebraMap B C).comp (algebraMap R B)).toAlgebra
    ⊢ Eq f₁ f₂
  -/
  haveI : IsScalarTower R B C := IsScalarTower.of_algebraMap_eq' rfl
  /-
    R : Type u_1
    inst✝⁷ : CommRing R
    A : Type u_2
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    B : Type u_3
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallyUnramified R A
    C : Type (max u_2 u_3)
    inst✝¹ : CommRing C
    inst✝ : Algebra B C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom B (TensorProduct R B A) C
    e : Eq ((Ideal.Quotient.mkₐ B I).comp f₁) ((Ideal.Quotient.mkₐ B I).comp f₂)
    this✝ : Algebra R C := ((algebraMap B C).comp (algebraMap R B)).toAlgebra
    this : IsScalarTower R B C
    ⊢ Eq f₁ f₂
  -/
  ext : 1
    /-
      case ha
      R : Type u_1
      inst✝⁷ : CommRing R
      A : Type u_2
      inst✝⁶ : CommRing A
      inst✝⁵ : Algebra R A
      B : Type u_3
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R B
      inst✝² : Algebra.FormallyUnramified R A
      C : Type (max u_2 u_3)
      inst✝¹ : CommRing C
      inst✝ : Algebra B C
      I : Ideal C
      hI : Eq (HPow.hPow I 2) Bot.bot
      f₁ f₂ : AlgHom B (TensorProduct R B A) C
      e : Eq ((Ideal.Quotient.mkₐ B I).comp f₁) ((Ideal.Quotient.mkₐ B I).comp f₂)
      this✝ : Algebra R C := ((algebraMap B C).comp (algebraMap R B)).toAlgebra
      this : IsScalarTower R B C
      ⊢ Eq (f₁.comp Algebra.TensorProduct.includeLeft) (f₂.comp Algebra.TensorProduc …
    -/
  · subsingleton
    /-
      🎉 no goals
    -/
    /-
      case hb
      R : Type u_1
      inst✝⁷ : CommRing R
      A : Type u_2
      inst✝⁶ : CommRing A
      inst✝⁵ : Algebra R A
      B : Type u_3
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R B
      inst✝² : Algebra.FormallyUnramified R A
      C : Type (max u_2 u_3)
      inst✝¹ : CommRing C
      inst✝ : Algebra B C
      I : Ideal C
      hI : Eq (HPow.hPow I 2) Bot.bot
      f₁ f₂ : AlgHom B (TensorProduct R B A) C
      e : Eq ((Ideal.Quotient.mkₐ B I).comp f₁) ((Ideal.Quotient.mkₐ B I).comp f₂)
      this✝ : Algebra R C := ((algebraMap B C).comp (algebraMap R B)).toAlgebra
      this : IsScalarTower R B C
      ⊢ Eq ((AlgHom.restrictScalars R f₁).comp Algebra.TensorProduct.includeRight) ( …
    -/
  · exact FormallyUnramified.ext I ⟨2, hI⟩ fun x => AlgHom.congr_fun e (1 ⊗ₜ x)
    /-
      🎉 no goals
    -/


/-- This holds in general for epimorphisms. -/
theorem of_isLocalization [IsLocalization M Rₘ] : FormallyUnramified R Rₘ := by
  /-
    R : Type u_1
    Rₘ : Type u_3
    inst✝³ : CommRing R
    inst✝² : CommRing Rₘ
    M : Submonoid R
    inst✝¹ : Algebra R Rₘ
    inst✝ : IsLocalization M Rₘ
    ⊢ Algebra.FormallyUnramified R Rₘ
  -/
  rw [iff_comp_injective]
  /-
    R : Type u_1
    Rₘ : Type u_3
    inst✝³ : CommRing R
    inst✝² : CommRing Rₘ
    M : Submonoid R
    inst✝¹ : Algebra R Rₘ
    inst✝ : IsLocalization M Rₘ
    ⊢ ∀ ⦃B : Type u_3⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), E …
  -/
  intro Q _ _ I _ f₁ f₂ _
  /-
    R : Type u_1
    Rₘ : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing Rₘ
    M : Submonoid R
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization M Rₘ
    Q : Type u_3
    inst✝¹ : CommRing Q
    inst✝ : Algebra R Q
    I : Ideal Q
    x✝ : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R Rₘ Q
    a✝ : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    ⊢ Eq f₁ f₂
  -/
  apply AlgHom.coe_ringHom_injective
  /-
    case a
    R : Type u_1
    Rₘ : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing Rₘ
    M : Submonoid R
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization M Rₘ
    Q : Type u_3
    inst✝¹ : CommRing Q
    inst✝ : Algebra R Q
    I : Ideal Q
    x✝ : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R Rₘ Q
    a✝ : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    ⊢ Eq ↑f₁ ↑f₂
  -/
  refine IsLocalization.ringHom_ext M ?_
  /-
    case a
    R : Type u_1
    Rₘ : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing Rₘ
    M : Submonoid R
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization M Rₘ
    Q : Type u_3
    inst✝¹ : CommRing Q
    inst✝ : Algebra R Q
    I : Ideal Q
    x✝ : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R Rₘ Q
    a✝ : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    ⊢ Eq ((↑f₁).comp (algebraMap R Rₘ)) ((↑f₂).comp (algebraMap R Rₘ))
  -/
  ext
  /-
    case a.a
    R : Type u_1
    Rₘ : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing Rₘ
    M : Submonoid R
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization M Rₘ
    Q : Type u_3
    inst✝¹ : CommRing Q
    inst✝ : Algebra R Q
    I : Ideal Q
    x✝¹ : Eq (HPow.hPow I 2) Bot.bot
    f₁ f₂ : AlgHom R Rₘ Q
    a✝ : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
    x✝ : R
    ⊢ Eq (((↑f₁).comp (algebraMap R Rₘ)) x✝) (((↑f₂).comp (algebraMap R Rₘ)) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- This actually does not need the localization instance, and is stated here again for
consistency. See `Algebra.FormallyUnramified.of_comp` instead.

 The intended use is for copying proofs between `Formally{Unramified, Smooth, Etale}`
 without the need to change anything (including removing redundant arguments). -/
-- @[nolint unusedArguments] -- Porting note: removed
theorem localization_base [FormallyUnramified R Sₘ] : FormallyUnramified Rₘ Sₘ :=
  -- Porting note: added
  let _ := M
  FormallyUnramified.of_comp R Rₘ Sₘ


theorem localization_map [FormallyUnramified R S] :
    FormallyUnramified Rₘ Sₘ := by
  haveI : FormallyUnramified S Sₘ :=
    FormallyUnramified.of_isLocalization (M.map (algebraMap R S))
  /-
    R : Type u_1
    S : Type u_2
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝¹² : CommRing R
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : CommRing Rₘ
    inst✝⁹ : CommRing Sₘ
    M : Submonoid R
    inst✝⁸ : Algebra R S
    inst✝⁷ : Algebra R Sₘ
    inst✝⁶ : Algebra S Sₘ
    inst✝⁵ : Algebra R Rₘ
    inst✝⁴ : Algebra Rₘ Sₘ
    inst✝³ : IsScalarTower R Rₘ Sₘ
    inst✝² : IsScalarTower R S Sₘ
    inst✝¹ : IsLocalization (Submonoid.map (algebraMap R S) M) Sₘ
    inst✝ : Algebra.FormallyUnramified R S
    this : Algebra.FormallyUnramified S Sₘ
    ⊢ Algebra.FormallyUnramified Rₘ Sₘ
  -/
  haveI : FormallyUnramified R Sₘ := FormallyUnramified.comp R S Sₘ
  /-
    R : Type u_1
    S : Type u_2
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝¹² : CommRing R
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : CommRing Rₘ
    inst✝⁹ : CommRing Sₘ
    M : Submonoid R
    inst✝⁸ : Algebra R S
    inst✝⁷ : Algebra R Sₘ
    inst✝⁶ : Algebra S Sₘ
    inst✝⁵ : Algebra R Rₘ
    inst✝⁴ : Algebra Rₘ Sₘ
    inst✝³ : IsScalarTower R Rₘ Sₘ
    inst✝² : IsScalarTower R S Sₘ
    inst✝¹ : IsLocalization (Submonoid.map (algebraMap R S) M) Sₘ
    inst✝ : Algebra.FormallyUnramified R S
    this✝ : Algebra.FormallyUnramified S Sₘ
    this : Algebra.FormallyUnramified R Sₘ
    ⊢ Algebra.FormallyUnramified Rₘ Sₘ
  -/
  exact FormallyUnramified.localization_base M
  /-
    🎉 no goals
  -/


/-- An `R`-algebra `A` is unramified if it is formally unramified and of finite type.

Note that the Stacks project has a different definition of unramified, and tag
<https://stacks.math.columbia.edu/tag/00UU> shows that their definition is the
same as this one.
-/
class Unramified : Prop where
  formallyUnramified : FormallyUnramified R A := by infer_instance
  finiteType : FiniteType R A := by infer_instance


/-- Being unramified is transported via algebra isomorphisms. -/
theorem of_equiv [Unramified R A] (e : A ≃ₐ[R] B) : Unramified R B where
  formallyUnramified := FormallyUnramified.of_equiv e
  finiteType := FiniteType.equiv Unramified.finiteType e


/-- Localization at an element is unramified. -/
theorem of_isLocalization_Away (r : R) [IsLocalization.Away r A] : Unramified R A where
  formallyUnramified := Algebra.FormallyUnramified.of_isLocalization (Submonoid.powers r)
  finiteType :=
    haveI : FinitePresentation R A := IsLocalization.Away.finitePresentation r
    inferInstance


/-- Unramified is stable under composition. -/
theorem comp [Algebra A B] [IsScalarTower R A B] [Unramified R A] [Unramified A B] :
    Unramified R B where
  formallyUnramified := FormallyUnramified.comp R A B
  finiteType := FiniteType.trans (S := A) Unramified.finiteType
    Unramified.finiteType


/-- Unramified is stable under base change. -/
instance baseChange [Unramified R A] : Unramified B (B ⊗[R] A) where


