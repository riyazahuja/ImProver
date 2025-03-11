instance : StarMul (FreeMonoid α) where
  star := List.reverse
  star_involutive := List.reverse_reverse
  star_mul := List.reverse_append


@[simp]
theorem star_of (x : α) : star (of x) = of x :=
  rfl


/-- Note that `star_one` is already a global simp lemma, but this one works with dsimp too -/
@[simp]
theorem star_one : star (1 : FreeMonoid α) = 1 :=
  rfl


/-- The star ring formed by reversing the elements of products -/
instance : StarRing (FreeAlgebra R X) where
  star := MulOpposite.unop ∘ lift R (MulOpposite.op ∘ ι R)
  star_involutive x := by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      x : FreeAlgebra R X
      ⊢ Eq (Star.star (Star.star x)) x
    -/
    unfold Star.star
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      x : FreeAlgebra R X
      ⊢ Eq ({ star := Function.comp MulOpposite.unop ⇑((FreeAlgebra.lift R) (Functio …
    -/
    simp only [Function.comp_apply]
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      x : FreeAlgebra R X
      ⊢ Eq (MulOpposite.unop (((FreeAlgebra.lift R) (Function.comp MulOpposite.op (F …
    -/
    let y := lift R (X := X) (MulOpposite.op ∘ ι R)
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      x : FreeAlgebra R X
      y : AlgHom R (FreeAlgebra R X) (MulOpposite (FreeAlgebra R X)) := (FreeAlgebra …
      ⊢ Eq (MulOpposite.unop (((FreeAlgebra.lift R) (Function.comp MulOpposite.op (F …
    -/
    refine induction (C := fun x ↦ (y (y x).unop).unop = x) _ _ ?_ ?_ ?_ ?_ x
      /-
        case refine_1
        R : Type u_1
        inst✝ : CommSemiring R
        X : Type u_2
        x : FreeAlgebra R X
        y : AlgHom R (FreeAlgebra R X) (MulOpposite (FreeAlgebra R X)) := (FreeAlgebra …
        ⊢ ∀ (r : R), (fun x => Eq (MulOpposite.unop (y (MulOpposite.unop (y x)))) x) ( …
      -/
    · intros
      /-
        case refine_1
        R : Type u_1
        inst✝ : CommSemiring R
        X : Type u_2
        x : FreeAlgebra R X
        y : AlgHom R (FreeAlgebra R X) (MulOpposite (FreeAlgebra R X)) := (FreeAlgebra …
        r✝ : R
        ⊢ Eq (MulOpposite.unop (y (MulOpposite.unop (y ((algebraMap R (FreeAlgebra R X …
      -/
      simp only [AlgHom.commutes, MulOpposite.algebraMap_apply, MulOpposite.unop_op]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        R : Type u_1
        inst✝ : CommSemiring R
        X : Type u_2
        x : FreeAlgebra R X
        y : AlgHom R (FreeAlgebra R X) (MulOpposite (FreeAlgebra R X)) := (FreeAlgebra …
        ⊢ ∀ (x : X), (fun x => Eq (MulOpposite.unop (y (MulOpposite.unop (y x)))) x) ( …
      -/
    · intros
      /-
        case refine_2
        R : Type u_1
        inst✝ : CommSemiring R
        X : Type u_2
        x : FreeAlgebra R X
        y : AlgHom R (FreeAlgebra R X) (MulOpposite (FreeAlgebra R X)) := (FreeAlgebra …
        x✝ : X
        ⊢ Eq (MulOpposite.unop (y (MulOpposite.unop (y (FreeAlgebra.ι R x✝))))) (FreeA …
      -/
      simp only [y, lift_ι_apply, Function.comp_apply, MulOpposite.unop_op]
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        R : Type u_1
        inst✝ : CommSemiring R
        X : Type u_2
        x : FreeAlgebra R X
        y : AlgHom R (FreeAlgebra R X) (MulOpposite (FreeAlgebra R X)) := (FreeAlgebra …
        ⊢ ∀ (a b : FreeAlgebra R X), (fun x => Eq (MulOpposite.unop (y (MulOpposite.un …
      -/
    · intros
      /-
        case refine_3
        R : Type u_1
        inst✝ : CommSemiring R
        X : Type u_2
        x : FreeAlgebra R X
        y : AlgHom R (FreeAlgebra R X) (MulOpposite (FreeAlgebra R X)) := (FreeAlgebra …
        a✝² b✝ : FreeAlgebra R X
        a✝¹ : Eq (MulOpposite.unop (y (MulOpposite.unop (y a✝²)))) a✝²
        a✝ : Eq (MulOpposite.unop (y (MulOpposite.unop (y b✝)))) b✝
        ⊢ Eq (MulOpposite.unop (y (MulOpposite.unop (y (HMul.hMul a✝² b✝))))) (HMul.hM …
      -/
      simp only [*, map_mul, MulOpposite.unop_mul]
      /-
        🎉 no goals
      -/
      /-
        case refine_4
        R : Type u_1
        inst✝ : CommSemiring R
        X : Type u_2
        x : FreeAlgebra R X
        y : AlgHom R (FreeAlgebra R X) (MulOpposite (FreeAlgebra R X)) := (FreeAlgebra …
        ⊢ ∀ (a b : FreeAlgebra R X), (fun x => Eq (MulOpposite.unop (y (MulOpposite.un …
      -/
    · intros
      /-
        case refine_4
        R : Type u_1
        inst✝ : CommSemiring R
        X : Type u_2
        x : FreeAlgebra R X
        y : AlgHom R (FreeAlgebra R X) (MulOpposite (FreeAlgebra R X)) := (FreeAlgebra …
        a✝² b✝ : FreeAlgebra R X
        a✝¹ : Eq (MulOpposite.unop (y (MulOpposite.unop (y a✝²)))) a✝²
        a✝ : Eq (MulOpposite.unop (y (MulOpposite.unop (y b✝)))) b✝
        ⊢ Eq (MulOpposite.unop (y (MulOpposite.unop (y (HAdd.hAdd a✝² b✝))))) (HAdd.hA …
      -/
      simp only [*, map_add, MulOpposite.unop_add]
      /-
        🎉 no goals
      -/
                     /-
                       R : Type u_1
                       inst✝ : CommSemiring R
                       X : Type u_2
                       a b : FreeAlgebra R X
                       ⊢ Eq (Star.star (HMul.hMul a b)) (HMul.hMul (Star.star b) (Star.star a))
                     -/
  star_mul a b := by simp only [Function.comp_apply, map_mul, MulOpposite.unop_mul]
                     /-
                       🎉 no goals
                     -/
                     /-
                       R : Type u_1
                       inst✝ : CommSemiring R
                       X : Type u_2
                       a b : FreeAlgebra R X
                       ⊢ Eq (Star.star (HAdd.hAdd a b)) (HAdd.hAdd (Star.star a) (Star.star b))
                     -/
  star_add a b := by simp only [Function.comp_apply, map_add, MulOpposite.unop_add]
                     /-
                       🎉 no goals
                     -/


@[simp]
                                                    /-
                                                      R : Type u_1
                                                      inst✝ : CommSemiring R
                                                      X : Type u_2
                                                      x : X
                                                      ⊢ Eq (Star.star (FreeAlgebra.ι R x)) (FreeAlgebra.ι R x)
                                                    -/
theorem star_ι (x : X) : star (ι R x) = ι R x := by simp [star, Star.star]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem star_algebraMap (r : R) : star (algebraMap R (FreeAlgebra R X) r) = algebraMap R _ r := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    X : Type u_2
    r : R
    ⊢ Eq (Star.star ((algebraMap R (FreeAlgebra R X)) r)) ((algebraMap R (FreeAlge …
  -/
  simp [star, Star.star]
  /-
    🎉 no goals
  -/


/-- `star` as an `AlgEquiv` -/
def starHom : FreeAlgebra R X ≃ₐ[R] (FreeAlgebra R X)ᵐᵒᵖ :=
                                                /-
                                                  R : Type u_1
                                                  inst✝ : CommSemiring R
                                                  X : Type u_2
                                                  r : R
                                                  ⊢ Eq (__src✝.toFun ((algebraMap R (FreeAlgebra R X)) r)) ((algebraMap R (MulOp …
                                                -/
  { starRingEquiv with commutes' := fun r => by simp [star_algebraMap] }
                                                /-
                                                  🎉 no goals
                                                -/


