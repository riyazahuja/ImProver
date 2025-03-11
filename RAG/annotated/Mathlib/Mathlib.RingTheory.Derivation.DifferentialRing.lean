/-- A derivation from a ring to itself, as a typeclass. -/
class Differential (R : Type*) [CommRing R] where
  /-- The `Derivation` associated with the ring. -/
  deriv : Derivation ℤ R R


@[inherit_doc]
scoped[Differential] postfix:max "′" => Differential.deriv


open Lean PrettyPrinter Delaborator SubExpr in
/--
A delaborator for the x′ notation. This is required because it's not direct function application,
so the default delaborator doesn't work.
-/
@[app_delab DFunLike.coe]
def delabDeriv : Delab := do
  let e ← getExpr
  guard <| e.isAppOfArity' ``DFunLike.coe 6
  guard <| (e.getArg!' 4).isAppOf' ``Differential.deriv
  let arg ← withAppArg delab
  `($arg′)


/--
A differential algebra is an `Algebra` where the derivation commutes with `algebraMap`.
-/
class DifferentialAlgebra (A B : Type*) [CommRing A] [CommRing B] [Algebra A B]
    [Differential A] [Differential B] : Prop where
  deriv_algebraMap : ∀ a : A, (algebraMap A B a)′ = algebraMap A B a′


@[norm_cast]
lemma algebraMap.coe_deriv {A : Type*} {B : Type*} [CommRing A] [CommRing B] [Algebra A B]
    [Differential A] [Differential B] [DifferentialAlgebra A B] (a : A) :
    (a′ : A) = (a : B)′ :=
  (DifferentialAlgebra.deriv_algebraMap _).symm


/--
A differential ring `A` and an algebra over it `B` share constants if all
constants in B are in the range of `algberaMap A B`.
-/
class Differential.ContainConstants (A B : Type*) [CommRing A] [CommRing B]
    [Algebra A B] [Differential B] : Prop where
  /-- If the derivative of x is 0, then it's in the range of `algberaMap A B`. -/
  protected mem_range_of_deriv_eq_zero {x : B} (h : x′ = 0) : x ∈ (algebraMap A B).range


lemma mem_range_of_deriv_eq_zero (A : Type*) {B : Type*} [CommRing A] [CommRing B] [Algebra A B]
    [Differential B] [Differential.ContainConstants A B] {x : B} (h : x′ = 0) :
    x ∈ (algebraMap A B).range :=
  Differential.ContainConstants.mem_range_of_deriv_eq_zero h


instance (A : Type*) [CommRing A] [Differential A] : DifferentialAlgebra A A where
  deriv_algebraMap _ := rfl


instance (A : Type*) [CommRing A] [Differential A] : Differential.ContainConstants A A where
  mem_range_of_deriv_eq_zero {x} _ := ⟨x, rfl⟩


/-- Transfer a `Differential` instance across a `RingEquiv`. -/
@[reducible]
def Differential.equiv {R R₂ : Type*} [CommRing R] [CommRing R₂] [Differential R₂]
    (h : R ≃+* R₂) : Differential R :=
  ⟨Derivation.mk' (h.symm.toAddMonoidHom.toIntLinearMap ∘ₗ
                                                                           /-
                                                                             R : Type u_1
                                                                             R₂ : Type u_2
                                                                             inst✝² : CommRing R
                                                                             inst✝¹ : CommRing R₂
                                                                             inst✝ : Differential R₂
                                                                             h : RingEquiv R R₂
                                                                             ⊢ ∀ (a b : R), Eq ((h.symm.toAddMonoidHom.toIntLinearMap.comp ((↑Differential. …
                                                                           -/
    Differential.deriv.toLinearMap ∘ₗ h.toAddMonoidHom.toIntLinearMap) (by simp)⟩
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/--
Transfer a `DifferentialAlgebra` instance across a `AlgEquiv`.
-/
lemma DifferentialAlgebra.equiv {A : Type*} [CommRing A] [Differential A]
    {R R₂ : Type*} [CommRing R] [CommRing R₂] [Differential R₂] [Algebra A R]
    [Algebra A R₂] [DifferentialAlgebra A R₂] (h : R ≃ₐ[A] R₂) :
    letI := Differential.equiv h.toRingEquiv
    DifferentialAlgebra A R :=
  letI := Differential.equiv h.toRingEquiv
  ⟨fun a ↦ by
    /-
      A : Type u_1
      inst✝⁷ : CommRing A
      inst✝⁶ : Differential A
      R : Type u_2
      R₂ : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing R₂
      inst✝³ : Differential R₂
      inst✝² : Algebra A R
      inst✝¹ : Algebra A R₂
      inst✝ : DifferentialAlgebra A R₂
      h : AlgEquiv A R R₂
      this : Differential R := Differential.equiv h.toRingEquiv
      a : A
      ⊢ Eq ((algebraMap A R) a)′ ((algebraMap A R) a′)
    -/
    change (LinearMap.comp ..) _ = _
    simp only [AlgEquiv.toRingEquiv_eq_coe, RingHom.toAddMonoidHom_eq_coe,
      RingEquiv.toRingHom_eq_coe, AlgEquiv.toRingEquiv_toRingHom, LinearMap.coe_comp,
      AddMonoidHom.coe_toIntLinearMap, AddMonoidHom.coe_coe, RingHom.coe_coe, Derivation.coeFn_coe,
      Function.comp_apply, AlgEquiv.commutes, deriv_algebraMap]
    /-
      A : Type u_1
      inst✝⁷ : CommRing A
      inst✝⁶ : Differential A
      R : Type u_2
      R₂ : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing R₂
      inst✝³ : Differential R₂
      inst✝² : Algebra A R
      inst✝¹ : Algebra A R₂
      inst✝ : DifferentialAlgebra A R₂
      h : AlgEquiv A R R₂
      this : Differential R := Differential.equiv h.toRingEquiv
      a : A
      ⊢ Eq ((↑h).symm ((algebraMap A R₂) a′)) ((algebraMap A R) a′)
    -/
    apply h.symm.commutes⟩
    /-
      🎉 no goals
    -/

