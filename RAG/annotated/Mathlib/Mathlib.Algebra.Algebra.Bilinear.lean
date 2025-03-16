/-- The multiplication on the left in a algebra is a linear map.

Note that this only assumes `SMulCommClass R A A`, so that it also works for `R := Aᵐᵒᵖ`.

When `A` is unital and associative, this is the same as `DistribMulAction.toLinearMap R A a` -/
def mulLeft (a : A) : A →ₗ[R] A where
  toFun := (a * ·)
  map_add' := mul_add _
  map_smul' _ := mul_smul_comm _ _


@[simp]
theorem mulLeft_apply (a b : A) : mulLeft R a b = a * b := rfl


@[simp]
theorem mulLeft_toAddMonoidHom (a : A) : (mulLeft R a : A →+ A) = AddMonoidHom.mulLeft a := rfl


variable (A) in
@[simp]
theorem mulLeft_zero_eq_zero : mulLeft R (0 : A) = 0 := ext fun _ => zero_mul _


/-- The multiplication on the right in an algebra is a linear map.

Note that this only assumes `IsScalarTower R A A`, so that it also works for `R := A`.

When `A` is unital and associative, this is the same as
`DistribMulAction.toLinearMap R A (MulOpposite.op b)`. -/
def mulRight (b : A) : A →ₗ[R] A where
  toFun := (· * b)
  map_add' _ _ := add_mul _ _ _
  map_smul' _ _ := smul_mul_assoc _ _ _


@[simp]
theorem mulRight_apply (a b : A) : mulRight R a b = b * a := rfl


@[simp]
theorem mulRight_toAddMonoidHom (a : A) : (mulRight R a : A →+ A) = AddMonoidHom.mulRight a := rfl


variable (A) in
@[simp]
theorem mulRight_zero_eq_zero : mulRight R (0 : A) = 0 := ext fun _ => mul_zero _


/-- The multiplication in a non-unital non-associative algebra is a bilinear map.

A weaker version of this for semirings exists as `AddMonoidHom.mul`. -/
@[simps!]
def mul : A →ₗ[R] A →ₗ[R] A :=
  LinearMap.mk₂ R (· * ·) add_mul smul_mul_assoc mul_add mul_smul_comm


/-- The multiplication map on a non-unital algebra, as an `R`-linear map from `A ⊗[R] A` to `A`. -/
-- TODO: upgrade to A-linear map if A is a semiring.
noncomputable def mul' : A ⊗[R] A →ₗ[R] A :=
  TensorProduct.lift (mul R A)


/-- Simultaneous multiplication on the left and right is a linear map. -/
def mulLeftRight (ab : A × A) : A →ₗ[R] A :=
  (mulRight R ab.snd).comp (mulLeft R ab.fst)


@[simp]
theorem mul_apply' (a b : A) : mul R A a b = a * b :=
  rfl


@[simp]
theorem mulLeftRight_apply (a b x : A) : mulLeftRight R (a, b) x = a * x * b :=
  rfl


@[simp]
theorem mul'_apply {a b : A} : mul' R A (a ⊗ₜ b) = a * b :=
  rfl


@[simp]
theorem mulLeft_mul [SMulCommClass R A A] (a b : A) :
    mulLeft R (a * b) = (mulLeft R a).comp (mulLeft R b) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : Semiring R
    inst✝² : NonUnitalSemiring A
    inst✝¹ : Module R A
    inst✝ : SMulCommClass R A A
    a b : A
    ⊢ Eq (LinearMap.mulLeft R (HMul.hMul a b)) ((LinearMap.mulLeft R a).comp (Line …
  -/
  ext
  /-
    case h
    R : Type u_1
    A : Type u_2
    inst✝³ : Semiring R
    inst✝² : NonUnitalSemiring A
    inst✝¹ : Module R A
    inst✝ : SMulCommClass R A A
    a b x✝ : A
    ⊢ Eq ((LinearMap.mulLeft R (HMul.hMul a b)) x✝) (((LinearMap.mulLeft R a).comp …
  -/
  simp only [mulLeft_apply, comp_apply, mul_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem mulRight_mul [IsScalarTower R A A] (a b : A) :
    mulRight R (a * b) = (mulRight R b).comp (mulRight R a) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : Semiring R
    inst✝² : NonUnitalSemiring A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R A A
    a b : A
    ⊢ Eq (LinearMap.mulRight R (HMul.hMul a b)) ((LinearMap.mulRight R b).comp (Li …
  -/
  ext
  /-
    case h
    R : Type u_1
    A : Type u_2
    inst✝³ : Semiring R
    inst✝² : NonUnitalSemiring A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R A A
    a b x✝ : A
    ⊢ Eq ((LinearMap.mulRight R (HMul.hMul a b)) x✝) (((LinearMap.mulRight R b).co …
  -/
  simp only [mulRight_apply, comp_apply, mul_assoc]
  /-
    🎉 no goals
  -/


/-- The multiplication in a non-unital algebra is a bilinear map.

A weaker version of this for non-unital non-associative algebras exists as `LinearMap.mul`. -/
def _root_.NonUnitalAlgHom.lmul : A →ₙₐ[R] End R A where
  __ := mul R A
  map_mul' := mulLeft_mul _ _
  map_zero' := mulLeft_zero_eq_zero _ _


@[simp]
theorem _root_.NonUnitalAlgHom.coe_lmul_eq_mul : ⇑(NonUnitalAlgHom.lmul R A) = mul R A :=
  rfl


theorem commute_mulLeft_right (a b : A) : Commute (mulLeft R a) (mulRight R b) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalSemiring A
    inst✝² : Module R A
    inst✝¹ : SMulCommClass R A A
    inst✝ : IsScalarTower R A A
    a b : A
    ⊢ Commute (LinearMap.mulLeft R a) (LinearMap.mulRight R b)
  -/
  ext c
  /-
    case h
    R : Type u_1
    A : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalSemiring A
    inst✝² : Module R A
    inst✝¹ : SMulCommClass R A A
    inst✝ : IsScalarTower R A A
    a b c : A
    ⊢ Eq ((HMul.hMul (LinearMap.mulLeft R a) (LinearMap.mulRight R b)) c) ((HMul.h …
  -/
  exact (mul_assoc a c b).symm
  /-
    🎉 no goals
  -/


/-- A `LinearMap` preserves multiplication if pre- and post- composition with `LinearMap.mul` are
equivalent. By converting the statement into an equality of `LinearMap`s, this lemma allows various
specialized `ext` lemmas about `→ₗ[R]` to then be applied.

This is the `LinearMap` version of `AddMonoidHom.map_mul_iff`. -/
theorem map_mul_iff (f : A →ₗ[R] B) :
    (∀ x y, f (x * y) = f x * f y) ↔
      (LinearMap.mul R A).compr₂ f = (LinearMap.mul R B ∘ₗ f).compl₂ f :=
  Iff.symm LinearMap.ext_iff₂


@[simp]
theorem mulLeft_one : mulLeft R (1 : A) = LinearMap.id := ext fun _ => one_mul _


@[simp]
theorem mulLeft_eq_zero_iff (a : A) : mulLeft R a = 0 ↔ a = 0 := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : Semiring R
    inst✝² : Semiring A
    inst✝¹ : Module R A
    inst✝ : SMulCommClass R A A
    a : A
    ⊢ Iff (Eq (LinearMap.mulLeft R a) 0) (Eq a 0)
  -/
  constructor <;> intro h
  -- Porting note: had to supply `R` explicitly in `@mulLeft_apply` below
    /-
      case mp
      R : Type u_1
      A : Type u_2
      inst✝³ : Semiring R
      inst✝² : Semiring A
      inst✝¹ : Module R A
      inst✝ : SMulCommClass R A A
      a : A
      h : Eq (LinearMap.mulLeft R a) 0
      ⊢ Eq a 0
    -/
  · rw [← mul_one a, ← mulLeft_apply R a 1, h, LinearMap.zero_apply]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      A : Type u_2
      inst✝³ : Semiring R
      inst✝² : Semiring A
      inst✝¹ : Module R A
      inst✝ : SMulCommClass R A A
      a : A
      h : Eq a 0
      ⊢ Eq (LinearMap.mulLeft R a) 0
    -/
  · rw [h]
    /-
      case mpr
      R : Type u_1
      A : Type u_2
      inst✝³ : Semiring R
      inst✝² : Semiring A
      inst✝¹ : Module R A
      inst✝ : SMulCommClass R A A
      a : A
      h : Eq a 0
      ⊢ Eq (LinearMap.mulLeft R 0) 0
    -/
    exact mulLeft_zero_eq_zero _ _
    /-
      🎉 no goals
    -/


@[simp]
theorem pow_mulLeft (a : A) (n : ℕ) : mulLeft R a ^ n = mulLeft R (a ^ n) :=
  match n with
            /-
              R : Type u_1
              A : Type u_2
              inst✝³ : Semiring R
              inst✝² : Semiring A
              inst✝¹ : Module R A
              inst✝ : SMulCommClass R A A
              a : A
              n : Nat
              ⊢ Eq (HPow.hPow (LinearMap.mulLeft R a) 0) (LinearMap.mulLeft R (HPow.hPow a 0))
            -/
  | 0 => by rw [pow_zero, pow_zero, mulLeft_one, LinearMap.one_eq_id]
            /-
              🎉 no goals
            -/
                  /-
                    R : Type u_1
                    A : Type u_2
                    inst✝³ : Semiring R
                    inst✝² : Semiring A
                    inst✝¹ : Module R A
                    inst✝ : SMulCommClass R A A
                    a : A
                    n✝ n : Nat
                    ⊢ Eq (HPow.hPow (LinearMap.mulLeft R a) (HAdd.hAdd n 1)) (LinearMap.mulLeft R  …
                  -/
  | (n + 1) => by rw [pow_succ, pow_succ, mulLeft_mul, LinearMap.mul_eq_comp, pow_mulLeft]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem mulRight_one : mulRight R (1 : A) = LinearMap.id := ext fun _ => mul_one _


@[simp]
theorem mulRight_eq_zero_iff (a : A) : mulRight R a = 0 ↔ a = 0 := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : Semiring R
    inst✝² : Semiring A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R A A
    a : A
    ⊢ Iff (Eq (LinearMap.mulRight R a) 0) (Eq a 0)
  -/
  constructor <;> intro h
  -- Porting note: had to supply `R` explicitly in `@mulRight_apply` below
    /-
      case mp
      R : Type u_1
      A : Type u_2
      inst✝³ : Semiring R
      inst✝² : Semiring A
      inst✝¹ : Module R A
      inst✝ : IsScalarTower R A A
      a : A
      h : Eq (LinearMap.mulRight R a) 0
      ⊢ Eq a 0
    -/
  · rw [← one_mul a, ← mulRight_apply R a 1, h, LinearMap.zero_apply]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      A : Type u_2
      inst✝³ : Semiring R
      inst✝² : Semiring A
      inst✝¹ : Module R A
      inst✝ : IsScalarTower R A A
      a : A
      h : Eq a 0
      ⊢ Eq (LinearMap.mulRight R a) 0
    -/
  · rw [h]
    /-
      case mpr
      R : Type u_1
      A : Type u_2
      inst✝³ : Semiring R
      inst✝² : Semiring A
      inst✝¹ : Module R A
      inst✝ : IsScalarTower R A A
      a : A
      h : Eq a 0
      ⊢ Eq (LinearMap.mulRight R 0) 0
    -/
    exact mulRight_zero_eq_zero _ _
    /-
      🎉 no goals
    -/


@[simp]
theorem pow_mulRight (a : A) (n : ℕ) : mulRight R a ^ n = mulRight R (a ^ n) :=
  match n with
            /-
              R : Type u_1
              A : Type u_2
              inst✝³ : Semiring R
              inst✝² : Semiring A
              inst✝¹ : Module R A
              inst✝ : IsScalarTower R A A
              a : A
              n : Nat
              ⊢ Eq (HPow.hPow (LinearMap.mulRight R a) 0) (LinearMap.mulRight R (HPow.hPow a …
            -/
  | 0 => by rw [pow_zero, pow_zero, mulRight_one, LinearMap.one_eq_id]
            /-
              🎉 no goals
            -/
                  /-
                    R : Type u_1
                    A : Type u_2
                    inst✝³ : Semiring R
                    inst✝² : Semiring A
                    inst✝¹ : Module R A
                    inst✝ : IsScalarTower R A A
                    a : A
                    n✝ n : Nat
                    ⊢ Eq (HPow.hPow (LinearMap.mulRight R a) (HAdd.hAdd n 1)) (LinearMap.mulRight  …
                  -/
  | (n + 1) => by rw [pow_succ, pow_succ', mulRight_mul, LinearMap.mul_eq_comp, pow_mulRight]
                  /-
                    🎉 no goals
                  -/


/-- The multiplication in an algebra is an algebra homomorphism into the endomorphisms on
the algebra.

A weaker version of this for non-unital algebras exists as `NonUnitalAlgHom.lmul`. -/
def _root_.Algebra.lmul : A →ₐ[R] End R A where
  __ := NonUnitalAlgHom.lmul R A
  map_one' := mulLeft_one _ _
  commutes' r := ext fun a => (Algebra.smul_def r a).symm


@[simp]
theorem _root_.Algebra.coe_lmul_eq_mul : ⇑(Algebra.lmul R A) = mul R A :=
  rfl


theorem _root_.Algebra.lmul_injective : Function.Injective (Algebra.lmul R A) :=
                   /-
                     R : Type u_1
                     A : Type u_2
                     inst✝² : CommSemiring R
                     inst✝¹ : Semiring A
                     inst✝ : Algebra R A
                     a₁ a₂ : A
                     h : Eq ((Algebra.lmul R A) a₁) ((Algebra.lmul R A) a₂)
                     ⊢ Eq a₁ a₂
                   -/
  fun a₁ a₂ h ↦ by simpa using DFunLike.congr_fun h 1
                   /-
                     🎉 no goals
                   -/


theorem _root_.Algebra.lmul_isUnit_iff {x : A} :
    IsUnit (Algebra.lmul R A x) ↔ IsUnit x := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : A
    ⊢ Iff (IsUnit ((Algebra.lmul R A) x)) (IsUnit x)
  -/
  rw [Module.End_isUnit_iff, Iff.comm]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : A
    ⊢ Iff (IsUnit x) (Function.Bijective ⇑((Algebra.lmul R A) x))
  -/
  exact IsUnit.isUnit_iff_mulLeft_bijective
  /-
    🎉 no goals
  -/


theorem toSpanSingleton_eq_algebra_linearMap : toSpanSingleton R A 1 = Algebra.linearMap R A := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    ⊢ Eq (LinearMap.toSpanSingleton R A 1) (Algebra.linearMap R A)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[deprecated mul_right_injective₀ (since := "2024-11-18")]
theorem mulLeft_injective [NoZeroDivisors A] {x : A} (hx : x ≠ 0) :
    Function.Injective (mulLeft R x) :=
  mul_right_injective₀ hx


@[deprecated mul_left_injective₀ (since := "2024-11-18")]
theorem mulRight_injective [NoZeroDivisors A] {x : A} (hx : x ≠ 0) :
    Function.Injective (mulRight R x) :=
  mul_left_injective₀ hx


@[deprecated mul_right_injective₀ (since := "2024-11-18")]
theorem mul_injective [NoZeroDivisors A] {x : A} (hx : x ≠ 0) : Function.Injective (mul R A x) :=
   mul_right_injective₀ hx


