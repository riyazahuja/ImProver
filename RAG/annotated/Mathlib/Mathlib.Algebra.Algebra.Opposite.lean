instance instAlgebra : Algebra R Aᵐᵒᵖ where
  toRingHom := (algebraMap R A).toOpposite fun _ _ => Algebra.commutes _ _
  smul_def' c x := unop_injective <| by
    simp only [unop_smul, RingHom.toOpposite_apply, Function.comp_apply, unop_mul, op_mul,
      Algebra.smul_def, Algebra.commutes, op_unop, unop_op]
  commutes' r := MulOpposite.rec' fun x => by
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring R
      inst✝⁸ : CommSemiring S
      inst✝⁷ : Semiring A
      inst✝⁶ : Semiring B
      inst✝⁵ : Algebra R S
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      inst✝² : Algebra S A
      inst✝¹ : SMulCommClass R S A
      inst✝ : IsScalarTower R S A
      r : R
      x : A
      ⊢ Eq (HMul.hMul (((algebraMap R A).toOpposite ⋯) r) (MulOpposite.op x)) (HMul. …
    -/
    simp only [RingHom.toOpposite_apply, Function.comp_apply, ← op_mul, Algebra.commutes]
    /-
      🎉 no goals
    -/


@[simp]
theorem algebraMap_apply (c : R) : algebraMap R Aᵐᵒᵖ c = op (algebraMap R A c) :=
  rfl


/-- An algebra is isomorphic to the opposite of its opposite. -/
@[simps!]
def opOp : A ≃ₐ[R] Aᵐᵒᵖᵐᵒᵖ where
  __ := RingEquiv.opOp A
  commutes' _ := rfl


@[simp] theorem toRingEquiv_opOp : (opOp R A : A ≃+* Aᵐᵒᵖᵐᵒᵖ) = RingEquiv.opOp A := rfl


/--
An algebra homomorphism `f : A →ₐ[R] B` such that `f x` commutes with `f y` for all `x, y` defines
an algebra homomorphism from `Aᵐᵒᵖ`. -/
@[simps (config := .asFn)]
def fromOpposite (f : A →ₐ[R] B) (hf : ∀ x y, Commute (f x) (f y)) : Aᵐᵒᵖ →ₐ[R] B :=
  { f.toRingHom.fromOpposite hf with
    toFun := f ∘ unop
    commutes' := fun r => f.commutes r }


@[simp]
theorem toLinearMap_fromOpposite (f : A →ₐ[R] B) (hf : ∀ x y, Commute (f x) (f y)) :
    (f.fromOpposite hf).toLinearMap = f.toLinearMap ∘ₗ (opLinearEquiv R (M := A)).symm :=
  rfl


@[simp]
theorem toRingHom_fromOpposite (f : A →ₐ[R] B) (hf : ∀ x y, Commute (f x) (f y)) :
    (f.fromOpposite hf : Aᵐᵒᵖ →+* B) = (f : A →+* B).fromOpposite hf :=
  rfl


/--
An algebra homomorphism `f : A →ₐ[R] B` such that `f x` commutes with `f y` for all `x, y` defines
an algebra homomorphism to `Bᵐᵒᵖ`. -/
@[simps (config := .asFn)]
def toOpposite (f : A →ₐ[R] B) (hf : ∀ x y, Commute (f x) (f y)) : A →ₐ[R] Bᵐᵒᵖ :=
  { f.toRingHom.toOpposite hf with
    toFun := op ∘ f
    commutes' := fun r => unop_injective <| f.commutes r }


@[simp]
theorem toLinearMap_toOpposite (f : A →ₐ[R] B) (hf : ∀ x y, Commute (f x) (f y)) :
    (f.toOpposite hf).toLinearMap = (opLinearEquiv R : B ≃ₗ[R] Bᵐᵒᵖ) ∘ₗ f.toLinearMap :=
  rfl


@[simp]
theorem toRingHom_toOpposite (f : A →ₐ[R] B) (hf : ∀ x y, Commute (f x) (f y)) :
    (f.toOpposite hf : A →+* Bᵐᵒᵖ) = (f : A →+* B).toOpposite hf :=
  rfl


/-- An algebra hom `A →ₐ[R] B` can equivalently be viewed as an algebra hom `Aᵐᵒᵖ →ₐ[R] Bᵐᵒᵖ`.
This is the action of the (fully faithful) `ᵐᵒᵖ`-functor on morphisms. -/
@[simps!]
protected def op : (A →ₐ[R] B) ≃ (Aᵐᵒᵖ →ₐ[R] Bᵐᵒᵖ) where
  toFun f := { RingHom.op f.toRingHom with commutes' := fun r => unop_injective <| f.commutes r }
  invFun f := { RingHom.unop f.toRingHom with commutes' := fun r => op_injective <| f.commutes r }
  left_inv _f := AlgHom.ext fun _a => rfl
  right_inv _f := AlgHom.ext fun _a => rfl


theorem toRingHom_op (f : A →ₐ[R] B) : f.op.toRingHom = RingHom.op f.toRingHom :=
  rfl


/-- The 'unopposite' of an algebra hom `Aᵐᵒᵖ →ₐ[R] Bᵐᵒᵖ`. Inverse to `RingHom.op`. -/
abbrev unop : (Aᵐᵒᵖ →ₐ[R] Bᵐᵒᵖ) ≃ (A →ₐ[R] B) := AlgHom.op.symm


theorem toRingHom_unop (f : Aᵐᵒᵖ →ₐ[R] Bᵐᵒᵖ) : f.unop.toRingHom = RingHom.unop f.toRingHom :=
  rfl


/-- Swap the `ᵐᵒᵖ` on an algebra hom to the opposite side. -/
@[simps!]
def opComm : (A →ₐ[R] Bᵐᵒᵖ) ≃ (Aᵐᵒᵖ →ₐ[R] B) :=
  AlgHom.op.trans <| AlgEquiv.refl.arrowCongr (AlgEquiv.opOp R B).symm


/-- An algebra iso `A ≃ₐ[R] B` can equivalently be viewed as an algebra iso `Aᵐᵒᵖ ≃ₐ[R] Bᵐᵒᵖ`.
This is the action of the (fully faithful) `ᵐᵒᵖ`-functor on morphisms. -/
@[simps!]
def op : (A ≃ₐ[R] B) ≃ Aᵐᵒᵖ ≃ₐ[R] Bᵐᵒᵖ where
  toFun f :=
    { RingEquiv.op f.toRingEquiv with
      commutes' := fun r => MulOpposite.unop_injective <| f.commutes r }
  invFun f :=
    { RingEquiv.unop f.toRingEquiv with
      commutes' := fun r => MulOpposite.op_injective <| f.commutes r }
  left_inv _f := AlgEquiv.ext fun _a => rfl
  right_inv _f := AlgEquiv.ext fun _a => rfl


theorem toAlgHom_op (f : A ≃ₐ[R] B) :
    (AlgEquiv.op f).toAlgHom = AlgHom.op f.toAlgHom :=
  rfl


theorem toRingEquiv_op (f : A ≃ₐ[R] B) :
    (AlgEquiv.op f).toRingEquiv = RingEquiv.op f.toRingEquiv :=
  rfl


/-- The 'unopposite' of an algebra iso `Aᵐᵒᵖ ≃ₐ[R] Bᵐᵒᵖ`. Inverse to `AlgEquiv.op`. -/
abbrev unop : (Aᵐᵒᵖ ≃ₐ[R] Bᵐᵒᵖ) ≃ A ≃ₐ[R] B := AlgEquiv.op.symm


theorem toAlgHom_unop (f : Aᵐᵒᵖ ≃ₐ[R] Bᵐᵒᵖ) : f.unop.toAlgHom = AlgHom.unop f.toAlgHom :=
  rfl


theorem toRingEquiv_unop (f : Aᵐᵒᵖ ≃ₐ[R] Bᵐᵒᵖ) :
    (AlgEquiv.unop f).toRingEquiv = RingEquiv.unop f.toRingEquiv :=
  rfl


/-- Swap the `ᵐᵒᵖ` on an algebra isomorphism to the opposite side. -/
@[simps!]
def opComm : (A ≃ₐ[R] Bᵐᵒᵖ) ≃ (Aᵐᵒᵖ ≃ₐ[R] B) :=
  AlgEquiv.op.trans <| AlgEquiv.refl.equivCongr (opOp R B).symm


/-- A commutative algebra is isomorphic to its opposite. -/
@[simps!]
def toOpposite : A ≃ₐ[R] Aᵐᵒᵖ where
  __ := RingEquiv.toOpposite A
  commutes' _r := rfl


@[simp] lemma toRingEquiv_toOpposite : (toOpposite R A : A ≃+* Aᵐᵒᵖ) = RingEquiv.toOpposite A := rfl

@[simp] lemma toLinearEquiv_toOpposite : toLinearEquiv (toOpposite R A) = opLinearEquiv R := rfl


