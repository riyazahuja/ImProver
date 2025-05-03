/-- The type of dual numbers, numbers of the form $a + bε$ where $ε^2 = 0$.
`R[ε]` is notation for `DualNumber R`. -/
abbrev DualNumber (R : Type*) : Type _ :=
  TrivSqZeroExt R R


/-- The unit element $ε$ that squares to zero, with notation `ε`. -/
def DualNumber.eps [Zero R] [One R] : DualNumber R :=
  TrivSqZeroExt.inr 1


@[inherit_doc]
scoped[DualNumber] notation "ε" => DualNumber.eps


@[inherit_doc]
scoped[DualNumber] postfix:1024 "[ε]" => DualNumber


@[simp]
theorem fst_eps [Zero R] [One R] : fst ε = (0 : R) :=
  fst_inr _ _


@[simp]
theorem snd_eps [Zero R] [One R] : snd ε = (1 : R) :=
  snd_inr _ _


/-- A version of `TrivSqZeroExt.snd_mul` with `*` instead of `•`. -/
@[simp]
theorem snd_mul [Semiring R] (x y : R[ε]) : snd (x * y) = fst x * snd y + snd x * fst y :=
  TrivSqZeroExt.snd_mul _ _


@[simp]
theorem eps_mul_eps [Semiring R] : (ε * ε : R[ε]) = 0 :=
  inr_mul_inr _ _ _


@[simp]
theorem inv_eps [DivisionRing R] : (ε : R[ε])⁻¹ = 0 :=
  TrivSqZeroExt.inv_inr 1


@[simp]
theorem inr_eq_smul_eps [MulZeroOneClass R] (r : R) : inr r = (r • ε : R[ε]) :=
  ext (mul_zero r).symm (mul_one r).symm


/-- `ε` commutes with every element of the algebra. -/
theorem commute_eps_left [Semiring R] (x : DualNumber R) : Commute ε x := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x : DualNumber R
    ⊢ Commute DualNumber.eps x
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


/-- `ε` commutes with every element of the algebra. -/
theorem commute_eps_right [Semiring R] (x : DualNumber R) : Commute x ε := (commute_eps_left x).symm


/-- For two `R`-algebra morphisms out of `A[ε]` to agree, it suffices for them to agree on the
elements of `A` and the `A`-multiples of `ε`. -/
@[ext 1100]
nonrec theorem algHom_ext' ⦃f g : A[ε] →ₐ[R] B⦄
    (hinl : f.comp (inlAlgHom _ _ _) = g.comp (inlAlgHom _ _ _))
    (hinr : f.toLinearMap ∘ₗ (LinearMap.toSpanSingleton A A[ε] ε).restrictScalars R =
        g.toLinearMap ∘ₗ (LinearMap.toSpanSingleton A A[ε] ε).restrictScalars R) :
      f = g :=
  algHom_ext' hinl (by
    /-
      R : Type u_1
      B : Type u_3
      A : Type u_4
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f g : AlgHom R (DualNumber A) B
      hinl : Eq (f.comp (TrivSqZeroExt.inlAlgHom R A A)) (g.comp (TrivSqZeroExt.inlA …
      hinr : Eq (f.toLinearMap.comp (↑R (LinearMap.toSpanSingleton A (DualNumber A)  …
      ⊢ Eq (f.toLinearMap.comp (↑R (TrivSqZeroExt.inrHom A A))) (g.toLinearMap.comp  …
    -/
    ext a
    /-
      case h
      R : Type u_1
      B : Type u_3
      A : Type u_4
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f g : AlgHom R (DualNumber A) B
      hinl : Eq (f.comp (TrivSqZeroExt.inlAlgHom R A A)) (g.comp (TrivSqZeroExt.inlA …
      hinr : Eq (f.toLinearMap.comp (↑R (LinearMap.toSpanSingleton A (DualNumber A)  …
      a : A
      ⊢ Eq ((f.toLinearMap.comp (↑R (TrivSqZeroExt.inrHom A A))) a) ((g.toLinearMap. …
    -/
    show f (inr a) = g (inr a)
    /-
      case h
      R : Type u_1
      B : Type u_3
      A : Type u_4
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f g : AlgHom R (DualNumber A) B
      hinl : Eq (f.comp (TrivSqZeroExt.inlAlgHom R A A)) (g.comp (TrivSqZeroExt.inlA …
      hinr : Eq (f.toLinearMap.comp (↑R (LinearMap.toSpanSingleton A (DualNumber A)  …
      a : A
      ⊢ Eq (f (TrivSqZeroExt.inr a)) (g (TrivSqZeroExt.inr a))
    -/
    simpa only [inr_eq_smul_eps] using DFunLike.congr_fun hinr a)
    /-
      🎉 no goals
    -/


/-- For two `R`-algebra morphisms out of `R[ε]` to agree, it suffices for them to agree on `ε`. -/
@[ext 1200]
nonrec theorem algHom_ext ⦃f g : R[ε] →ₐ[R] A⦄ (hε : f ε = g ε) : f = g := by
  /-
    R : Type u_1
    A : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f g : AlgHom R (DualNumber R) A
    hε : Eq (f DualNumber.eps) (g DualNumber.eps)
    ⊢ Eq f g
  -/
  ext
  /-
    case hinr.h
    R : Type u_1
    A : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f g : AlgHom R (DualNumber R) A
    hε : Eq (f DualNumber.eps) (g DualNumber.eps)
    ⊢ Eq ((f.toLinearMap.comp (↑R (LinearMap.toSpanSingleton R (DualNumber R) Dual …
  -/
  dsimp
  /-
    case hinr.h
    R : Type u_1
    A : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f g : AlgHom R (DualNumber R) A
    hε : Eq (f DualNumber.eps) (g DualNumber.eps)
    ⊢ Eq (f (HSMul.hSMul 1 DualNumber.eps)) (g (HSMul.hSMul 1 DualNumber.eps))
  -/
  simp only [one_smul, hε]
  /-
    🎉 no goals
  -/


/-- A universal property of the dual numbers, providing a unique `A[ε] →ₐ[R] B` for every map
`f : A →ₐ[R] B` and a choice of element `e : B` which squares to `0` and commutes with the range of
`f`.

This isomorphism is named to match the similar `Complex.lift`.
Note that when `f : R →ₐ[R] B := Algebra.ofId R B`, the commutativity assumption is automatic, and
we are free to choose any element `e : B`. -/
def lift :
    {fe : (A →ₐ[R] B) × B // fe.2 * fe.2 = 0 ∧ ∀ a, Commute fe.2 (fe.1 a)} ≃ (A[ε] →ₐ[R] B) := by
  /-
    R : Type u_1
    A✝ : Type u_2
    B : Type u_3
    A : Type u_4
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    ⊢ Equiv (Subtype fun fe => And (Eq (HMul.hMul fe.2 fe.2) 0) (∀ (a : A), Commut …
  -/
  refine Equiv.trans ?_ TrivSqZeroExt.liftEquiv
  exact {
    toFun := fun fe => ⟨
      (fe.val.1, MulOpposite.op fe.val.2 • fe.val.1.toLinearMap),
      fun x y => show (fe.val.1 x * fe.val.2) * (fe.val.1 y * fe.val.2) = 0 by
        rw [(fe.prop.2 _).mul_mul_mul_comm, fe.prop.1, mul_zero],
      fun r x => show fe.val.1 (r * x) * fe.val.2 = fe.val.1 r * (fe.val.1 x * fe.val.2) by
        rw [map_mul, mul_assoc],
      fun r x => show fe.val.1 (x * r) * fe.val.2 = (fe.val.1 x * fe.val.2) * fe.val.1 r by
        rw [map_mul, (fe.prop.2 _).right_comm]⟩
    invFun := fun fg => ⟨
      (fg.val.1, fg.val.2 1),
      fg.prop.1 _ _,
      fun a => show fg.val.2 1 * fg.val.1 a = fg.val.1 a * fg.val.2 1 by
        rw [← fg.prop.2.1, ← fg.prop.2.2, smul_eq_mul, op_smul_eq_mul, mul_one, one_mul]⟩
    left_inv := fun fe => Subtype.ext <| Prod.ext rfl <|
      show fe.val.1 1 * fe.val.2 = fe.val.2 by
        rw [map_one, one_mul]
    right_inv := fun fg => Subtype.ext <| Prod.ext rfl <| LinearMap.ext fun x =>
      show fg.val.1 x * fg.val.2 1 = fg.val.2 x by
        rw [← fg.prop.2.1, smul_eq_mul, mul_one] }


theorem lift_apply_apply (fe : {_fe : (A →ₐ[R] B) × B // _}) (a : A[ε]) :
    lift fe a = fe.val.1 a.fst + fe.val.1 a.snd * fe.val.2 := rfl


@[simp] theorem coe_lift_symm_apply (F : A[ε] →ₐ[R] B) :
    (lift.symm F).val = (F.comp (inlAlgHom _ _ _), F ε) := rfl


set_option linter.unusedVariables false in
/-- When applied to `inl`, `DualNumber.lift` applies the map `f : A →ₐ[R] B`. -/
@[simp] theorem lift_apply_inl (fe : {fe : (A →ₐ[R] B) × B // _}) (a : A) :
    lift fe (inl a : A[ε]) = fe.val.1 a := by
  /-
    R : Type u_1
    B : Type u_3
    A : Type u_4
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    fe : Subtype fun fe => And (Eq (HMul.hMul fe.2 fe.2) 0) (∀ (a : A), Commute fe …
    a : A
    ⊢ Eq ((DualNumber.lift fe) (TrivSqZeroExt.inl a)) ((↑fe).1 a)
  -/
  rw [lift_apply_apply, fst_inl, snd_inl, map_zero, zero_mul, add_zero]
  /-
    🎉 no goals
  -/


set_option linter.unusedVariables false in
/-- Scaling on the left is sent by `DualNumber.lift` to multiplication on the left -/
@[simp] theorem lift_smul (fe : {fe : (A →ₐ[R] B) × B // _}) (a : A) (ad : A[ε]) :
    lift fe (a • ad) = fe.val.1 a * lift fe ad := by
  /-
    R : Type u_1
    B : Type u_3
    A : Type u_4
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    fe : Subtype fun fe => And (Eq (HMul.hMul fe.2 fe.2) 0) (∀ (a : A), Commute fe …
    a : A
    ad : DualNumber A
    ⊢ Eq ((DualNumber.lift fe) (HSMul.hSMul a ad)) (HMul.hMul ((↑fe).1 a) ((DualNu …
  -/
  rw [← inl_mul_eq_smul, map_mul, lift_apply_inl]
  /-
    🎉 no goals
  -/


set_option linter.unusedVariables false in
/-- Scaling on the right is sent by `DualNumber.lift` to multiplication on the right -/
@[simp] theorem lift_op_smul (fe : {fe : (A →ₐ[R] B) × B // _}) (a : A) (ad : A[ε]) :
    lift fe (MulOpposite.op a • ad) = lift fe ad * fe.val.1 a := by
  /-
    R : Type u_1
    B : Type u_3
    A : Type u_4
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    fe : Subtype fun fe => And (Eq (HMul.hMul fe.2 fe.2) 0) (∀ (a : A), Commute fe …
    a : A
    ad : DualNumber A
    ⊢ Eq ((DualNumber.lift fe) (HSMul.hSMul (MulOpposite.op a) ad)) (HMul.hMul ((D …
  -/
  rw [← mul_inl_eq_op_smul, map_mul, lift_apply_inl]
  /-
    🎉 no goals
  -/


/-- When applied to `ε`, `DualNumber.lift` produces the element of `B` that squares to 0. -/
@[simp] theorem lift_apply_eps
    (fe : {fe : (A →ₐ[R] B) × B // fe.2 * fe.2 = 0 ∧ ∀ a, Commute fe.2 (fe.1 a)}) :
    lift fe (ε : A[ε]) = fe.val.2 := by
  /-
    R : Type u_1
    B : Type u_3
    A : Type u_4
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    fe : Subtype fun fe => And (Eq (HMul.hMul fe.2 fe.2) 0) (∀ (a : A), Commute fe …
    ⊢ Eq ((DualNumber.lift fe) DualNumber.eps) (↑fe).2
  -/
  simp only [lift_apply_apply, fst_eps, map_zero, snd_eps, map_one, one_mul, zero_add]
  /-
    🎉 no goals
  -/


/-- Lifting `DualNumber.eps` itself gives the identity. -/
@[simp]
theorem lift_inlAlgHom_eps :
    lift ⟨(inlAlgHom _ _ _, ε), eps_mul_eps, fun _ => commute_eps_left _⟩ = AlgHom.id R A[ε] :=
  lift.apply_symm_apply <| AlgHom.id R A[ε]


/-- Show DualNumber with values x and y as an "x + y*ε" string -/
instance instRepr [Repr R] : Repr (DualNumber R) where
  reprPrec f p :=
    (if p > 65 then (Std.Format.bracket "(" · ")") else (·)) <|
      reprPrec f.fst 65 ++ " + " ++ reprPrec f.snd 70 ++ "*ε"


