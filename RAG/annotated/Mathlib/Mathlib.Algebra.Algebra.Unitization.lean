/-- The minimal unitization of a non-unital `R`-algebra `A`. This is just a type synonym for
`R × A`. -/
def Unitization (R A : Type*) :=
  R × A


/-- The canonical inclusion `R → Unitization R A`. -/
def inl [Zero A] (r : R) : Unitization R A :=
  (r, 0)

-- Porting note: we need a def to which we can attach `@[coe]`

/-- The canonical inclusion `A → Unitization R A`. -/
@[coe]
def inr [Zero R] (a : A) : Unitization R A :=
  (0, a)


instance [Zero R] : CoeTC A (Unitization R A) where
  coe := inr


/-- The canonical projection `Unitization R A → R`. -/
def fst (x : Unitization R A) : R :=
  x.1


/-- The canonical projection `Unitization R A → A`. -/
def snd (x : Unitization R A) : A :=
  x.2


@[ext]
theorem ext {x y : Unitization R A} (h1 : x.fst = y.fst) (h2 : x.snd = y.snd) : x = y :=
  Prod.ext h1 h2


@[simp]
theorem fst_inl [Zero A] (r : R) : (inl r : Unitization R A).fst = r :=
  rfl


@[simp]
theorem snd_inl [Zero A] (r : R) : (inl r : Unitization R A).snd = 0 :=
  rfl


@[simp]
theorem fst_inr [Zero R] (a : A) : (a : Unitization R A).fst = 0 :=
  rfl


@[simp]
theorem snd_inr [Zero R] (a : A) : (a : Unitization R A).snd = a :=
  rfl


theorem inl_injective [Zero A] : Function.Injective (inl : R → Unitization R A) :=
  Function.LeftInverse.injective <| fst_inl _


theorem inr_injective [Zero R] : Function.Injective ((↑) : A → Unitization R A) :=
  Function.LeftInverse.injective <| snd_inr _


instance instNontrivialLeft {𝕜 A} [Nontrivial 𝕜] [Nonempty A] :
    Nontrivial (Unitization 𝕜 A) :=
  nontrivial_prod_left


instance instNontrivialRight {𝕜 A} [Nonempty 𝕜] [Nontrivial A] :
    Nontrivial (Unitization 𝕜 A) :=
  nontrivial_prod_right


instance instCanLift [Zero R] : CanLift (Unitization R A) A inr (fun x ↦ x.fst = 0) where
  prf x hx := ⟨x.snd, ext (hx ▸ fst_inr R (snd x)) rfl⟩


instance instInhabited [Inhabited R] [Inhabited A] : Inhabited (Unitization R A) :=
  instInhabitedProd


instance instZero [Zero R] [Zero A] : Zero (Unitization R A) :=
  Prod.instZero


instance instAdd [Add R] [Add A] : Add (Unitization R A) :=
  Prod.instAdd


instance instNeg [Neg R] [Neg A] : Neg (Unitization R A) :=
  Prod.instNeg


instance instAddSemigroup [AddSemigroup R] [AddSemigroup A] : AddSemigroup (Unitization R A) :=
  Prod.instAddSemigroup


instance instAddZeroClass [AddZeroClass R] [AddZeroClass A] : AddZeroClass (Unitization R A) :=
  Prod.instAddZeroClass


instance instAddMonoid [AddMonoid R] [AddMonoid A] : AddMonoid (Unitization R A) :=
  Prod.instAddMonoid


instance instAddGroup [AddGroup R] [AddGroup A] : AddGroup (Unitization R A) :=
  Prod.instAddGroup


instance instAddCommSemigroup [AddCommSemigroup R] [AddCommSemigroup A] :
    AddCommSemigroup (Unitization R A) :=
  Prod.instAddCommSemigroup


instance instAddCommMonoid [AddCommMonoid R] [AddCommMonoid A] : AddCommMonoid (Unitization R A) :=
  Prod.instAddCommMonoid


instance instAddCommGroup [AddCommGroup R] [AddCommGroup A] : AddCommGroup (Unitization R A) :=
  Prod.instAddCommGroup


instance instSMul [SMul S R] [SMul S A] : SMul S (Unitization R A) :=
  Prod.smul


instance instIsScalarTower [SMul T R] [SMul T A] [SMul S R] [SMul S A] [SMul T S]
    [IsScalarTower T S R] [IsScalarTower T S A] : IsScalarTower T S (Unitization R A) :=
  Prod.isScalarTower


instance instSMulCommClass [SMul T R] [SMul T A] [SMul S R] [SMul S A] [SMulCommClass T S R]
    [SMulCommClass T S A] : SMulCommClass T S (Unitization R A) :=
  Prod.smulCommClass


instance instIsCentralScalar [SMul S R] [SMul S A] [SMul Sᵐᵒᵖ R] [SMul Sᵐᵒᵖ A] [IsCentralScalar S R]
    [IsCentralScalar S A] : IsCentralScalar S (Unitization R A) :=
  Prod.isCentralScalar


instance instMulAction [Monoid S] [MulAction S R] [MulAction S A] : MulAction S (Unitization R A) :=
  Prod.mulAction


instance instDistribMulAction [Monoid S] [AddMonoid R] [AddMonoid A] [DistribMulAction S R]
    [DistribMulAction S A] : DistribMulAction S (Unitization R A) :=
  Prod.distribMulAction


instance instModule [Semiring S] [AddCommMonoid R] [AddCommMonoid A] [Module S R] [Module S A] :
    Module S (Unitization R A) :=
  Prod.instModule


variable (R A) in
/-- The identity map between `Unitization R A` and `R × A` as an `AddEquiv`. -/
def addEquiv [Add R] [Add A] : Unitization R A ≃+ R × A :=
  AddEquiv.refl _


@[simp]
theorem fst_zero [Zero R] [Zero A] : (0 : Unitization R A).fst = 0 :=
  rfl


@[simp]
theorem snd_zero [Zero R] [Zero A] : (0 : Unitization R A).snd = 0 :=
  rfl


@[simp]
theorem fst_add [Add R] [Add A] (x₁ x₂ : Unitization R A) : (x₁ + x₂).fst = x₁.fst + x₂.fst :=
  rfl


@[simp]
theorem snd_add [Add R] [Add A] (x₁ x₂ : Unitization R A) : (x₁ + x₂).snd = x₁.snd + x₂.snd :=
  rfl


@[simp]
theorem fst_neg [Neg R] [Neg A] (x : Unitization R A) : (-x).fst = -x.fst :=
  rfl


@[simp]
theorem snd_neg [Neg R] [Neg A] (x : Unitization R A) : (-x).snd = -x.snd :=
  rfl


@[simp]
theorem fst_smul [SMul S R] [SMul S A] (s : S) (x : Unitization R A) : (s • x).fst = s • x.fst :=
  rfl


@[simp]
theorem snd_smul [SMul S R] [SMul S A] (s : S) (x : Unitization R A) : (s • x).snd = s • x.snd :=
  rfl


@[simp]
theorem inl_zero [Zero R] [Zero A] : (inl 0 : Unitization R A) = 0 :=
  rfl


@[simp]
theorem inl_add [Add R] [AddZeroClass A] (r₁ r₂ : R) :
    (inl (r₁ + r₂) : Unitization R A) = inl r₁ + inl r₂ :=
  ext rfl (add_zero 0).symm


@[simp]
theorem inl_neg [Neg R] [AddGroup A] (r : R) : (inl (-r) : Unitization R A) = -inl r :=
  ext rfl neg_zero.symm


@[simp]
theorem inl_sub [AddGroup R] [AddGroup A] (r₁ r₂ : R) :
    (inl (r₁ - r₂) : Unitization R A) = inl r₁ - inl r₂ :=
  ext rfl (sub_zero 0).symm


@[simp]
theorem inl_smul [Monoid S] [AddMonoid A] [SMul S R] [DistribMulAction S A] (s : S) (r : R) :
    (inl (s • r) : Unitization R A) = s • inl r :=
  ext rfl (smul_zero s).symm


@[simp]
theorem inr_zero [Zero R] [Zero A] : ↑(0 : A) = (0 : Unitization R A) :=
  rfl


@[simp]
theorem inr_add [AddZeroClass R] [Add A] (m₁ m₂ : A) : (↑(m₁ + m₂) : Unitization R A) = m₁ + m₂ :=
  ext (add_zero 0).symm rfl


@[simp]
theorem inr_neg [AddGroup R] [Neg A] (m : A) : (↑(-m) : Unitization R A) = -m :=
  ext neg_zero.symm rfl


@[simp]
theorem inr_sub [AddGroup R] [AddGroup A] (m₁ m₂ : A) : (↑(m₁ - m₂) : Unitization R A) = m₁ - m₂ :=
  ext (sub_zero 0).symm rfl


@[simp]
theorem inr_smul [Zero R] [Zero S] [SMulWithZero S R] [SMul S A] (r : S) (m : A) :
    (↑(r • m) : Unitization R A) = r • (m : Unitization R A) :=
  ext (smul_zero _).symm rfl


theorem inl_fst_add_inr_snd_eq [AddZeroClass R] [AddZeroClass A] (x : Unitization R A) :
    inl x.fst + (x.snd : Unitization R A) = x :=
  ext (add_zero x.1) (zero_add x.2)


/-- To show a property hold on all `Unitization R A` it suffices to show it holds
on terms of the form `inl r + a`.

This can be used as `induction x`. -/
@[elab_as_elim, induction_eliminator, cases_eliminator]
theorem ind {R A} [AddZeroClass R] [AddZeroClass A] {P : Unitization R A → Prop}
    (inl_add_inr : ∀ (r : R) (a : A), P (inl r + (a : Unitization R A))) (x) : P x :=
  inl_fst_add_inr_snd_eq x ▸ inl_add_inr x.1 x.2


/-- This cannot be marked `@[ext]` as it ends up being used instead of `LinearMap.prod_ext` when
working with `R × A`. -/
theorem linearMap_ext {N} [Semiring S] [AddCommMonoid R] [AddCommMonoid A] [AddCommMonoid N]
    [Module S R] [Module S A] [Module S N] ⦃f g : Unitization R A →ₗ[S] N⦄
    (hl : ∀ r, f (inl r) = g (inl r)) (hr : ∀ a : A, f a = g a) : f = g :=
  LinearMap.prod_ext (LinearMap.ext hl) (LinearMap.ext hr)


/-- The canonical `R`-linear inclusion `A → Unitization R A`. -/
@[simps apply]
def inrHom [Semiring R] [AddCommMonoid A] [Module R A] : A →ₗ[R] Unitization R A :=
  { LinearMap.inr R R A with toFun := (↑) }


/-- The canonical `R`-linear projection `Unitization R A → A`. -/
@[simps apply]
def sndHom [Semiring R] [AddCommMonoid A] [Module R A] : Unitization R A →ₗ[R] A :=
  { LinearMap.snd _ _ _ with toFun := snd }


instance instOne [One R] [Zero A] : One (Unitization R A) :=
  ⟨(1, 0)⟩


instance instMul [Mul R] [Add A] [Mul A] [SMul R A] : Mul (Unitization R A) :=
  ⟨fun x y => (x.1 * y.1, x.1 • y.2 + y.1 • x.2 + x.2 * y.2)⟩


@[simp]
theorem fst_one [One R] [Zero A] : (1 : Unitization R A).fst = 1 :=
  rfl


@[simp]
theorem snd_one [One R] [Zero A] : (1 : Unitization R A).snd = 0 :=
  rfl


@[simp]
theorem fst_mul [Mul R] [Add A] [Mul A] [SMul R A] (x₁ x₂ : Unitization R A) :
    (x₁ * x₂).fst = x₁.fst * x₂.fst :=
  rfl


@[simp]
theorem snd_mul [Mul R] [Add A] [Mul A] [SMul R A] (x₁ x₂ : Unitization R A) :
    (x₁ * x₂).snd = x₁.fst • x₂.snd + x₂.fst • x₁.snd + x₁.snd * x₂.snd :=
  rfl


@[simp]
theorem inl_one [One R] [Zero A] : (inl 1 : Unitization R A) = 1 :=
  rfl


@[simp]
theorem inl_mul [Monoid R] [NonUnitalNonAssocSemiring A] [DistribMulAction R A] (r₁ r₂ : R) :
    (inl (r₁ * r₂) : Unitization R A) = inl r₁ * inl r₂ :=
  ext rfl <|
    show (0 : A) = r₁ • (0 : A) + r₂ • (0 : A) + 0 * 0 by
      /-
        R : Type u_1
        A : Type u_2
        inst✝² : Monoid R
        inst✝¹ : NonUnitalNonAssocSemiring A
        inst✝ : DistribMulAction R A
        r₁ r₂ : R
        ⊢ Eq 0 (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul r₁ 0) (HSMul.hSMul r₂ 0)) (HMul.hMul …
      -/
      simp only [smul_zero, add_zero, mul_zero]
      /-
        🎉 no goals
      -/


theorem inl_mul_inl [Monoid R] [NonUnitalNonAssocSemiring A] [DistribMulAction R A] (r₁ r₂ : R) :
    (inl r₁ * inl r₂ : Unitization R A) = inl (r₁ * r₂) :=
  (inl_mul A r₁ r₂).symm


@[simp]
theorem inr_mul [Semiring R] [AddCommMonoid A] [Mul A] [SMulWithZero R A] (a₁ a₂ : A) :
    (↑(a₁ * a₂) : Unitization R A) = a₁ * a₂ :=
  ext (mul_zero _).symm <|
                                                            /-
                                                              R : Type u_1
                                                              A : Type u_2
                                                              inst✝³ : Semiring R
                                                              inst✝² : AddCommMonoid A
                                                              inst✝¹ : Mul A
                                                              inst✝ : SMulWithZero R A
                                                              a₁ a₂ : A
                                                              ⊢ Eq (HMul.hMul a₁ a₂) (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul 0 a₂) (HSMul.hSMul 0 …
                                                            -/
    show a₁ * a₂ = (0 : R) • a₂ + (0 : R) • a₁ + a₁ * a₂ by simp only [zero_smul, zero_add]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem inl_mul_inr [Semiring R] [NonUnitalNonAssocSemiring A] [DistribMulAction R A] (r : R)
    (a : A) : ((inl r : Unitization R A) * a) = ↑(r • a) :=
  ext (mul_zero r) <|
    show r • a + (0 : R) • (0 : A) + 0 * a = r • a by
      /-
        R : Type u_1
        A : Type u_2
        inst✝² : Semiring R
        inst✝¹ : NonUnitalNonAssocSemiring A
        inst✝ : DistribMulAction R A
        r : R
        a : A
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul r a) (HSMul.hSMul 0 0)) (HMul.hMul 0 a …
      -/
      rw [smul_zero, add_zero, zero_mul, add_zero]
      /-
        🎉 no goals
      -/


theorem inr_mul_inl [Semiring R] [NonUnitalNonAssocSemiring A] [DistribMulAction R A] (r : R)
    (a : A) : a * (inl r : Unitization R A) = ↑(r • a) :=
  ext (zero_mul r) <|
    show (0 : R) • (0 : A) + r • a + a * 0 = r • a by
      /-
        R : Type u_1
        A : Type u_2
        inst✝² : Semiring R
        inst✝¹ : NonUnitalNonAssocSemiring A
        inst✝ : DistribMulAction R A
        r : R
        a : A
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul 0 0) (HSMul.hSMul r a)) (HMul.hMul a 0 …
      -/
      rw [smul_zero, zero_add, mul_zero, add_zero]
      /-
        🎉 no goals
      -/


instance instMulOneClass [Monoid R] [NonUnitalNonAssocSemiring A] [DistribMulAction R A] :
    MulOneClass (Unitization R A) :=
  { Unitization.instOne, Unitization.instMul with
    one_mul := fun x =>
      ext (one_mul x.1) <|
        show (1 : R) • x.2 + x.1 • (0 : A) + 0 * x.2 = x.2 by
          /-
            R : Type u_1
            A : Type u_2
            inst✝² : Monoid R
            inst✝¹ : NonUnitalNonAssocSemiring A
            inst✝ : DistribMulAction R A
            x : Unitization R A
            ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul 1 x.2) (HSMul.hSMul x.1 0)) (HMul.hMul …
          -/
          rw [one_smul, smul_zero, add_zero, zero_mul, add_zero]
          /-
            🎉 no goals
          -/
    mul_one := fun x =>
      ext (mul_one x.1) <|
        show (x.1 • (0 : A)) + (1 : R) • x.2 + x.2 * (0 : A) = x.2 by
          /-
            R : Type u_1
            A : Type u_2
            inst✝² : Monoid R
            inst✝¹ : NonUnitalNonAssocSemiring A
            inst✝ : DistribMulAction R A
            x : Unitization R A
            ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul x.1 0) (HSMul.hSMul 1 x.2)) (HMul.hMul …
          -/
          rw [smul_zero, zero_add, one_smul, mul_zero, add_zero] }
          /-
            🎉 no goals
          -/


instance instNonAssocSemiring [Semiring R] [NonUnitalNonAssocSemiring A] [Module R A] :
    NonAssocSemiring (Unitization R A) :=
  { Unitization.instMulOneClass,
    Unitization.instAddCommMonoid with
    zero_mul := fun x =>
      ext (zero_mul x.1) <|
        show (0 : R) • x.2 + x.1 • (0 : A) + 0 * x.2 = 0 by
          /-
            R : Type u_1
            A : Type u_2
            inst✝² : Semiring R
            inst✝¹ : NonUnitalNonAssocSemiring A
            inst✝ : Module R A
            x : Unitization R A
            ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul 0 x.2) (HSMul.hSMul x.1 0)) (HMul.hMul …
          -/
          rw [zero_smul, zero_add, smul_zero, zero_mul, add_zero]
          /-
            🎉 no goals
          -/
    mul_zero := fun x =>
      ext (mul_zero x.1) <|
        show x.1 • (0 : A) + (0 : R) • x.2 + x.2 * 0 = 0 by
          /-
            R : Type u_1
            A : Type u_2
            inst✝² : Semiring R
            inst✝¹ : NonUnitalNonAssocSemiring A
            inst✝ : Module R A
            x : Unitization R A
            ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul x.1 0) (HSMul.hSMul 0 x.2)) (HMul.hMul …
          -/
          rw [smul_zero, zero_add, zero_smul, mul_zero, add_zero]
          /-
            🎉 no goals
          -/
          /-
            R : Type u_1
            A : Type u_2
            inst✝² : Semiring R
            inst✝¹ : NonUnitalNonAssocSemiring A
            inst✝ : Module R A
            x₁ x₂ x₃ : Unitization R A
            ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul x₁.1 (HAdd.hAdd x₂.2 x₃.2)) (HSMul.hSM …
          -/
    left_distrib := fun x₁ x₂ x₃ =>
          /-
            R : Type u_1
            A : Type u_2
            inst✝² : Semiring R
            inst✝¹ : NonUnitalNonAssocSemiring A
            inst✝ : Module R A
            x₁ x₂ x₃ : Unitization R A
            ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul x₁.1 x₂.2) (HSMul.hSMul x₁. …
          -/
          /-
            🎉 no goals
          -/
      ext (mul_add x₁.1 x₂.1 x₃.1) <|
          /-
            🎉 no goals
          -/
        show x₁.1 • (x₂.2 + x₃.2) + (x₂.1 + x₃.1) • x₁.2 + x₁.2 * (x₂.2 + x₃.2) =
            x₁.1 • x₂.2 + x₂.1 • x₁.2 + x₁.2 * x₂.2 + (x₁.1 • x₃.2 + x₃.1 • x₁.2 + x₁.2 * x₃.2) by
          simp only [smul_add, add_smul, mul_add]
          abel
          /-
            R : Type u_1
            A : Type u_2
            inst✝² : Semiring R
            inst✝¹ : NonUnitalNonAssocSemiring A
            inst✝ : Module R A
            x₁ x₂ x₃ : Unitization R A
            ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd x₁.1 x₂.1) x₃.2) (HSMul.hSM …
          -/
    right_distrib := fun x₁ x₂ x₃ =>
          /-
            R : Type u_1
            A : Type u_2
            inst✝² : Semiring R
            inst✝¹ : NonUnitalNonAssocSemiring A
            inst✝ : Module R A
            x₁ x₂ x₃ : Unitization R A
            ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul x₁.1 x₃.2) (HSMul.hSMul x₂. …
          -/
          /-
            🎉 no goals
          -/
      ext (add_mul x₁.1 x₂.1 x₃.1) <|
          /-
            🎉 no goals
          -/
        show (x₁.1 + x₂.1) • x₃.2 + x₃.1 • (x₁.2 + x₂.2) + (x₁.2 + x₂.2) * x₃.2 =
            x₁.1 • x₃.2 + x₃.1 • x₁.2 + x₁.2 * x₃.2 + (x₂.1 • x₃.2 + x₃.1 • x₂.2 + x₂.2 * x₃.2) by
          simp only [add_smul, smul_add, add_mul]
          abel }


instance instMonoid [CommMonoid R] [NonUnitalSemiring A] [DistribMulAction R A]
    [IsScalarTower R A A] [SMulCommClass R A A] : Monoid (Unitization R A) :=
  { Unitization.instMulOneClass with
    mul_assoc := fun x y z =>
      ext (mul_assoc x.1 y.1 z.1) <|
        show (x.1 * y.1) • z.2 + z.1 • (x.1 • y.2 + y.1 • x.2 + x.2 * y.2) +
            (x.1 • y.2 + y.1 • x.2 + x.2 * y.2) * z.2 =
            x.1 • (y.1 • z.2 + z.1 • y.2 + y.2 * z.2) + (y.1 * z.1) • x.2 +
            x.2 * (y.1 • z.2 + z.1 • y.2 + y.2 * z.2) by
          simp only [smul_add, mul_add, add_mul, smul_smul, smul_mul_assoc, mul_smul_comm,
            mul_assoc]
          /-
            R : Type u_1
            A : Type u_2
            inst✝⁴ : CommMonoid R
            inst✝³ : NonUnitalSemiring A
            inst✝² : DistribMulAction R A
            inst✝¹ : IsScalarTower R A A
            inst✝ : SMulCommClass R A A
            x y z : Unitization R A
            ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (HMul.hMul x.1 y.1) z.2) (HAdd.hAdd (H …
          -/
          rw [mul_comm z.1 x.1]
          /-
            R : Type u_1
            A : Type u_2
            inst✝⁴ : CommMonoid R
            inst✝³ : NonUnitalSemiring A
            inst✝² : DistribMulAction R A
            inst✝¹ : IsScalarTower R A A
            inst✝ : SMulCommClass R A A
            x y z : Unitization R A
            ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (HMul.hMul x.1 y.1) z.2) (HAdd.hAdd (H …
          -/
          rw [mul_comm z.1 y.1]
          /-
            R : Type u_1
            A : Type u_2
            inst✝⁴ : CommMonoid R
            inst✝³ : NonUnitalSemiring A
            inst✝² : DistribMulAction R A
            inst✝¹ : IsScalarTower R A A
            inst✝ : SMulCommClass R A A
            x y z : Unitization R A
            ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (HMul.hMul x.1 y.1) z.2) (HAdd.hAdd (H …
          -/
          /-
            🎉 no goals
          -/
          abel }
          /-
            🎉 no goals
          -/


instance instCommMonoid [CommMonoid R] [NonUnitalCommSemiring A] [DistribMulAction R A]
    [IsScalarTower R A A] [SMulCommClass R A A] : CommMonoid (Unitization R A) :=
  { Unitization.instMonoid with
    mul_comm := fun x₁ x₂ =>
      ext (mul_comm x₁.1 x₂.1) <|
        show x₁.1 • x₂.2 + x₂.1 • x₁.2 + x₁.2 * x₂.2 = x₂.1 • x₁.2 + x₁.1 • x₂.2 + x₂.2 * x₁.2 by
          /-
            R : Type u_1
            A : Type u_2
            inst✝⁴ : CommMonoid R
            inst✝³ : NonUnitalCommSemiring A
            inst✝² : DistribMulAction R A
            inst✝¹ : IsScalarTower R A A
            inst✝ : SMulCommClass R A A
            x₁ x₂ : Unitization R A
            ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul x₁.1 x₂.2) (HSMul.hSMul x₂.1 x₁.2)) (H …
          -/
          rw [add_comm (x₁.1 • x₂.2), mul_comm] }
          /-
            🎉 no goals
          -/


instance instSemiring [CommSemiring R] [NonUnitalSemiring A] [Module R A] [IsScalarTower R A A]
    [SMulCommClass R A A] : Semiring (Unitization R A) :=
  { Unitization.instMonoid, Unitization.instNonAssocSemiring with }


instance instCommSemiring [CommSemiring R] [NonUnitalCommSemiring A] [Module R A]
    [IsScalarTower R A A] [SMulCommClass R A A] : CommSemiring (Unitization R A) :=
  { Unitization.instCommMonoid, Unitization.instNonAssocSemiring with }


instance instNonAssocRing [CommRing R] [NonUnitalNonAssocRing A] [Module R A] :
    NonAssocRing (Unitization R A) :=
  { Unitization.instAddCommGroup, Unitization.instNonAssocSemiring with }


instance instRing [CommRing R] [NonUnitalRing A] [Module R A] [IsScalarTower R A A]
    [SMulCommClass R A A] : Ring (Unitization R A) :=
  { Unitization.instAddCommGroup, Unitization.instSemiring with }


instance instCommRing [CommRing R] [NonUnitalCommRing A] [Module R A] [IsScalarTower R A A]
    [SMulCommClass R A A] : CommRing (Unitization R A) :=
  { Unitization.instAddCommGroup, Unitization.instCommSemiring with }


/-- The canonical inclusion of rings `R →+* Unitization R A`. -/
@[simps apply]
def inlRingHom [Semiring R] [NonUnitalSemiring A] [Module R A] : R →+* Unitization R A where
  toFun := inl
  map_one' := inl_one A
  map_mul' := inl_mul A
  map_zero' := inl_zero A
  map_add' := inl_add A


instance instStar [Star R] [Star A] : Star (Unitization R A) :=
  ⟨fun ra => (star ra.fst, star ra.snd)⟩


@[simp]
theorem fst_star [Star R] [Star A] (x : Unitization R A) : (star x).fst = star x.fst :=
  rfl


@[simp]
theorem snd_star [Star R] [Star A] (x : Unitization R A) : (star x).snd = star x.snd :=
  rfl


@[simp]
theorem inl_star [Star R] [AddMonoid A] [StarAddMonoid A] (r : R) :
    inl (star r) = star (inl r : Unitization R A) :=
              /-
                R : Type u_1
                A : Type u_2
                inst✝² : Star R
                inst✝¹ : AddMonoid A
                inst✝ : StarAddMonoid A
                r : R
                ⊢ Eq (Unitization.inl (Star.star r)).snd (Star.star (Unitization.inl r)).snd
              -/
  ext rfl (by simp only [snd_star, star_zero, snd_inl])
              /-
                🎉 no goals
              -/


@[simp]
theorem inr_star [AddMonoid R] [StarAddMonoid R] [Star A] (a : A) :
    ↑(star a) = star (a : Unitization R A) :=
          /-
            R : Type u_1
            A : Type u_2
            inst✝² : AddMonoid R
            inst✝¹ : StarAddMonoid R
            inst✝ : Star A
            a : A
            ⊢ Eq (↑(Star.star a)).fst (Star.star ↑a).fst
          -/
  ext (by simp only [fst_star, star_zero, fst_inr]) rfl
          /-
            🎉 no goals
          -/


instance instStarAddMonoid [AddMonoid R] [AddMonoid A] [StarAddMonoid R] [StarAddMonoid A] :
    StarAddMonoid (Unitization R A) where
  star_involutive x := ext (star_star x.fst) (star_star x.snd)
  star_add x y := ext (star_add x.fst y.fst) (star_add x.snd y.snd)


instance instStarModule [CommSemiring R] [StarRing R] [AddCommMonoid A] [StarAddMonoid A]
    [Module R A] [StarModule R A] : StarModule R (Unitization R A) where
                           /-
                             R : Type u_1
                             A : Type u_2
                             inst✝⁵ : CommSemiring R
                             inst✝⁴ : StarRing R
                             inst✝³ : AddCommMonoid A
                             inst✝² : StarAddMonoid A
                             inst✝¹ : Module R A
                             inst✝ : StarModule R A
                             r : R
                             x : Unitization R A
                             ⊢ Eq (Star.star (HSMul.hSMul r x)).fst (HSMul.hSMul (Star.star r) (Star.star x …
                           -/
                           /-
                             🎉 no goals
                           -/
  star_smul r x := ext (by simp) (by simp)
                                     /-
                                       🎉 no goals
                                     -/


instance instStarRing [CommSemiring R] [StarRing R] [NonUnitalNonAssocSemiring A] [StarRing A]
    [Module R A] [StarModule R A] :
    StarRing (Unitization R A) :=
  { Unitization.instStarAddMonoid with
    star_mul := fun x y =>
              /-
                R : Type u_1
                A : Type u_2
                inst✝⁵ : CommSemiring R
                inst✝⁴ : StarRing R
                inst✝³ : NonUnitalNonAssocSemiring A
                inst✝² : StarRing A
                inst✝¹ : Module R A
                inst✝ : StarModule R A
                x y : Unitization R A
                ⊢ Eq (Star.star (HMul.hMul x y)).fst (HMul.hMul (Star.star y) (Star.star x)).fst
              -/
              /-
                🎉 no goals
              -/
      ext (by simp [-star_mul']) (by simp [-star_mul', add_comm (star x.fst • star y.snd)]) }
                                     /-
                                       🎉 no goals
                                     -/


instance instAlgebra : Algebra S (Unitization R A) :=
  { (Unitization.inlRingHom R A).comp (algebraMap S R) with
    commutes' := fun s x => by
      induction x with
      | inl_add_inr =>
        show inl (algebraMap S R s) * _ = _ * inl (algebraMap S R s)
        rw [mul_add, add_mul, inl_mul_inl, inl_mul_inl, inl_mul_inr, inr_mul_inl, mul_comm]
    smul_def' := fun s x => by
      induction x with
      | inl_add_inr =>
        show _ = inl (algebraMap S R s) * _
        rw [mul_add, smul_add,Algebra.algebraMap_eq_smul_one, inl_mul_inl, inl_mul_inr,
          smul_one_mul, inl_smul, inr_smul, smul_one_smul] }


theorem algebraMap_eq_inl_comp : ⇑(algebraMap S (Unitization R A)) = inl ∘ algebraMap S R :=
  rfl


theorem algebraMap_eq_inlRingHom_comp :
    algebraMap S (Unitization R A) = (inlRingHom R A).comp (algebraMap S R) :=
  rfl


theorem algebraMap_eq_inl : ⇑(algebraMap R (Unitization R A)) = inl :=
  rfl


theorem algebraMap_eq_inlRingHom : algebraMap R (Unitization R A) = inlRingHom R A :=
  rfl


/-- The canonical `R`-algebra projection `Unitization R A → R`. -/
@[simps]
def fstHom : Unitization R A →ₐ[R] R where
  toFun := fst
  map_one' := fst_one
  map_mul' := fst_mul
  map_zero' := fst_zero (A := A)
  map_add' := fst_add
  commutes' := fst_inl A


/-- The coercion from a non-unital `R`-algebra `A` to its unitization `Unitization R A`
realized as a non-unital algebra homomorphism. -/
@[simps]
def inrNonUnitalAlgHom (R A : Type*) [CommSemiring R] [NonUnitalSemiring A] [Module R A] :
    A →ₙₐ[R] Unitization R A where
  toFun := (↑)
  map_smul' := inr_smul R
  map_zero' := inr_zero R
  map_add' := inr_add R
  map_mul' := inr_mul R


/-- The coercion from a non-unital `R`-algebra `A` to its unitization `Unitization R A`
realized as a non-unital star algebra homomorphism. -/
@[simps!]
def inrNonUnitalStarAlgHom (R A : Type*) [CommSemiring R] [StarAddMonoid R]
    [NonUnitalSemiring A] [Star A] [Module R A] :
    A →⋆ₙₐ[R] Unitization R A where
  toNonUnitalAlgHom := inrNonUnitalAlgHom R A
  map_star' := inr_star


/-- The star algebra equivalence obtained by restricting `Unitization.inrNonUnitalStarAlgHom`
to its range. -/
@[simps!]
def inrRangeEquiv (R A : Type*) [CommSemiring R] [StarAddMonoid R] [NonUnitalSemiring A]
    [Star A] [Module R A] [IsScalarTower R A A] [SMulCommClass R A A] :
    A ≃⋆ₐ[R] NonUnitalStarAlgHom.range (inrNonUnitalStarAlgHom R A) :=
  StarAlgEquiv.ofLeftInverse' (snd_inr R)


theorem algHom_ext {F : Type*}
    [FunLike F (Unitization R A) B] [AlgHomClass F S (Unitization R A) B] {φ ψ : F}
    (h : ∀ a : A, φ a = ψ a)
    (h' : ∀ r, φ (algebraMap R (Unitization R A) r) = ψ (algebraMap R (Unitization R A) r)) :
    φ = ψ := by
  /-
    S : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝¹² : CommSemiring S
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : Module R A
    inst✝⁸ : SMulCommClass R A A
    inst✝⁷ : IsScalarTower R A A
    B : Type u_4
    inst✝⁶ : Semiring B
    inst✝⁵ : Algebra S B
    inst✝⁴ : Algebra S R
    inst✝³ : DistribMulAction S A
    inst✝² : IsScalarTower S R A
    F : Type u_6
    inst✝¹ : FunLike F (Unitization R A) B
    inst✝ : AlgHomClass F S (Unitization R A) B
    φ ψ : F
    h : ∀ (a : A), Eq (φ ↑a) (ψ ↑a)
    h' : ∀ (r : R), Eq (φ ((algebraMap R (Unitization R A)) r)) (ψ ((algebraMap R  …
    ⊢ Eq φ ψ
  -/
  refine DFunLike.ext φ ψ (fun x ↦ ?_)
  /-
    S : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝¹² : CommSemiring S
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : Module R A
    inst✝⁸ : SMulCommClass R A A
    inst✝⁷ : IsScalarTower R A A
    B : Type u_4
    inst✝⁶ : Semiring B
    inst✝⁵ : Algebra S B
    inst✝⁴ : Algebra S R
    inst✝³ : DistribMulAction S A
    inst✝² : IsScalarTower S R A
    F : Type u_6
    inst✝¹ : FunLike F (Unitization R A) B
    inst✝ : AlgHomClass F S (Unitization R A) B
    φ ψ : F
    h : ∀ (a : A), Eq (φ ↑a) (ψ ↑a)
    h' : ∀ (r : R), Eq (φ ((algebraMap R (Unitization R A)) r)) (ψ ((algebraMap R  …
    x : Unitization R A
    ⊢ Eq (φ x) (ψ x)
  -/
  induction x
  /-
    case inl_add_inr
    S : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝¹² : CommSemiring S
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : Module R A
    inst✝⁸ : SMulCommClass R A A
    inst✝⁷ : IsScalarTower R A A
    B : Type u_4
    inst✝⁶ : Semiring B
    inst✝⁵ : Algebra S B
    inst✝⁴ : Algebra S R
    inst✝³ : DistribMulAction S A
    inst✝² : IsScalarTower S R A
    F : Type u_6
    inst✝¹ : FunLike F (Unitization R A) B
    inst✝ : AlgHomClass F S (Unitization R A) B
    φ ψ : F
    h : ∀ (a : A), Eq (φ ↑a) (ψ ↑a)
    h' : ∀ (r : R), Eq (φ ((algebraMap R (Unitization R A)) r)) (ψ ((algebraMap R  …
    r✝ : R
    a✝ : A
    ⊢ Eq (φ (HAdd.hAdd (Unitization.inl r✝) ↑a✝)) (ψ (HAdd.hAdd (Unitization.inl r …
  -/
  simp only [map_add, ← algebraMap_eq_inl, h, h']
  /-
    🎉 no goals
  -/


lemma algHom_ext'' {F : Type*}
    [FunLike F (Unitization R A) C] [AlgHomClass F R (Unitization R A) C] {φ ψ : F}
    (h : ∀ a : A, φ a = ψ a) : φ = ψ :=
                            /-
                              R : Type u_2
                              A : Type u_3
                              inst✝⁸ : CommSemiring R
                              inst✝⁷ : NonUnitalSemiring A
                              inst✝⁶ : Module R A
                              inst✝⁵ : SMulCommClass R A A
                              inst✝⁴ : IsScalarTower R A A
                              C : Type u_5
                              inst✝³ : Semiring C
                              inst✝² : Algebra R C
                              F : Type u_6
                              inst✝¹ : FunLike F (Unitization R A) C
                              inst✝ : AlgHomClass F R (Unitization R A) C
                              φ ψ : F
                              h : ∀ (a : A), Eq (φ ↑a) (ψ ↑a)
                              r : R
                              ⊢ Eq (φ ((algebraMap R (Unitization R A)) r)) (ψ ((algebraMap R (Unitization R …
                            -/
  algHom_ext h (fun r => by simp only [AlgHomClass.commutes])
                            /-
                              🎉 no goals
                            -/


/-- See note [partially-applied ext lemmas] -/
@[ext 1100]
theorem algHom_ext' {φ ψ : Unitization R A →ₐ[R] C}
    (h :
      φ.toNonUnitalAlgHom.comp (inrNonUnitalAlgHom R A) =
        ψ.toNonUnitalAlgHom.comp (inrNonUnitalAlgHom R A)) :
    φ = ψ :=
  algHom_ext'' (NonUnitalAlgHom.congr_fun h)

/- porting note: this was extracted from `Unitization.lift` below, where it had previously
been inlined. Unfortunately, `Unitization.lift` was relatively slow in Lean 3, but in Lean 4 it
just times out. -/

/-- A non-unital algebra homomorphism from `A` into a unital `R`-algebra `C` lifts to a unital
algebra homomorphism from the unitization into `C`. This is extended to an `Equiv` in
`Unitization.lift` and that should be used instead. This declaration only exists for performance
reasons. -/
@[simps]
def _root_.NonUnitalAlgHom.toAlgHom (φ : A →ₙₐ[R] C) : Unitization R A →ₐ[R] C where
  toFun := fun x => algebraMap R C x.fst + φ x.snd
                 /-
                   S : Type u_1
                   R : Type u_2
                   A : Type u_3
                   inst✝¹² : CommSemiring S
                   inst✝¹¹ : CommSemiring R
                   inst✝¹⁰ : NonUnitalSemiring A
                   inst✝⁹ : Module R A
                   inst✝⁸ : SMulCommClass R A A
                   inst✝⁷ : IsScalarTower R A A
                   B : Type u_4
                   inst✝⁶ : Semiring B
                   inst✝⁵ : Algebra S B
                   inst✝⁴ : Algebra S R
                   inst✝³ : DistribMulAction S A
                   inst✝² : IsScalarTower S R A
                   C : Type u_5
                   inst✝¹ : Semiring C
                   inst✝ : Algebra R C
                   φ : NonUnitalAlgHom (MonoidHom.id R) A C
                   ⊢ Eq ((fun x => HAdd.hAdd ((algebraMap R C) x.fst) (φ x.snd)) 1) 1
                 -/
  map_one' := by simp only [fst_one, map_one, snd_one, φ.map_zero, add_zero]
                 /-
                   🎉 no goals
                 -/
  map_mul' := fun x y => by
    induction x with
    | inl_add_inr x_r x_a =>
      induction y with
      | inl_add_inr =>
        simp only [fst_mul, fst_add, fst_inl, fst_inr, snd_mul, snd_add, snd_inl, snd_inr, add_zero,
          map_mul, zero_add, map_add, map_smul φ]
        rw [add_mul, mul_add, mul_add]
        rw [← Algebra.commutes _ (φ x_a)]
        simp only [Algebra.algebraMap_eq_smul_one, smul_one_mul, add_assoc]
                  /-
                    S : Type u_1
                    R : Type u_2
                    A : Type u_3
                    inst✝¹² : CommSemiring S
                    inst✝¹¹ : CommSemiring R
                    inst✝¹⁰ : NonUnitalSemiring A
                    inst✝⁹ : Module R A
                    inst✝⁸ : SMulCommClass R A A
                    inst✝⁷ : IsScalarTower R A A
                    B : Type u_4
                    inst✝⁶ : Semiring B
                    inst✝⁵ : Algebra S B
                    inst✝⁴ : Algebra S R
                    inst✝³ : DistribMulAction S A
                    inst✝² : IsScalarTower S R A
                    C : Type u_5
                    inst✝¹ : Semiring C
                    inst✝ : Algebra R C
                    φ : NonUnitalAlgHom (MonoidHom.id R) A C
                    ⊢ Eq ((↑{ toFun := fun x => HAdd.hAdd ((algebraMap R C) x.fst) (φ x.snd), map_ …
                  -/
  map_zero' := by simp only [fst_zero, map_zero, snd_zero, φ.map_zero, add_zero]
                  /-
                    🎉 no goals
                  -/
  map_add' := fun x y => by
    induction x with
    | inl_add_inr =>
      induction y with
      | inl_add_inr =>
        simp only [fst_add, fst_inl, fst_inr, add_zero, map_add, snd_add, snd_inl, snd_inr,
          zero_add, φ.map_add]
        rw [add_add_add_comm]
  commutes' := fun r => by
    /-
      S : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝¹² : CommSemiring S
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : NonUnitalSemiring A
      inst✝⁹ : Module R A
      inst✝⁸ : SMulCommClass R A A
      inst✝⁷ : IsScalarTower R A A
      B : Type u_4
      inst✝⁶ : Semiring B
      inst✝⁵ : Algebra S B
      inst✝⁴ : Algebra S R
      inst✝³ : DistribMulAction S A
      inst✝² : IsScalarTower S R A
      C : Type u_5
      inst✝¹ : Semiring C
      inst✝ : Algebra R C
      φ : NonUnitalAlgHom (MonoidHom.id R) A C
      r : R
      ⊢ Eq ((↑↑{ toFun := fun x => HAdd.hAdd ((algebraMap R C) x.fst) (φ x.snd), map …
    -/
    simp only [algebraMap_eq_inl, fst_inl, snd_inl, φ.map_zero, add_zero]
    /-
      🎉 no goals
    -/



/-- Non-unital algebra homomorphisms from `A` into a unital `R`-algebra `C` lift uniquely to
`Unitization R A →ₐ[R] C`. This is the universal property of the unitization. -/
@[simps! apply symm_apply apply_apply]
def lift : (A →ₙₐ[R] C) ≃ (Unitization R A →ₐ[R] C) where
  toFun := NonUnitalAlgHom.toAlgHom
  invFun φ := φ.toNonUnitalAlgHom.comp (inrNonUnitalAlgHom R A)
                   /-
                     S : Type u_1
                     R : Type u_2
                     A : Type u_3
                     inst✝¹² : CommSemiring S
                     inst✝¹¹ : CommSemiring R
                     inst✝¹⁰ : NonUnitalSemiring A
                     inst✝⁹ : Module R A
                     inst✝⁸ : SMulCommClass R A A
                     inst✝⁷ : IsScalarTower R A A
                     B : Type u_4
                     inst✝⁶ : Semiring B
                     inst✝⁵ : Algebra S B
                     inst✝⁴ : Algebra S R
                     inst✝³ : DistribMulAction S A
                     inst✝² : IsScalarTower S R A
                     C : Type u_5
                     inst✝¹ : Semiring C
                     inst✝ : Algebra R C
                     φ : NonUnitalAlgHom (MonoidHom.id R) A C
                     ⊢ Eq ((fun φ => (↑φ).comp (Unitization.inrNonUnitalAlgHom R A)) φ.toAlgHom) φ
                   -/
  left_inv φ := by ext; simp [NonUnitalAlgHomClass.toNonUnitalAlgHom]
                        /-
                          🎉 no goals
                        -/
                    /-
                      S : Type u_1
                      R : Type u_2
                      A : Type u_3
                      inst✝¹² : CommSemiring S
                      inst✝¹¹ : CommSemiring R
                      inst✝¹⁰ : NonUnitalSemiring A
                      inst✝⁹ : Module R A
                      inst✝⁸ : SMulCommClass R A A
                      inst✝⁷ : IsScalarTower R A A
                      B : Type u_4
                      inst✝⁶ : Semiring B
                      inst✝⁵ : Algebra S B
                      inst✝⁴ : Algebra S R
                      inst✝³ : DistribMulAction S A
                      inst✝² : IsScalarTower S R A
                      C : Type u_5
                      inst✝¹ : Semiring C
                      inst✝ : Algebra R C
                      φ : AlgHom R (Unitization R A) C
                      ⊢ Eq ((fun φ => (↑φ).comp (Unitization.inrNonUnitalAlgHom R A)) φ).toAlgHom φ
                    -/
  right_inv φ := by ext; simp [NonUnitalAlgHomClass.toNonUnitalAlgHom]
                         /-
                           🎉 no goals
                         -/


theorem lift_symm_apply_apply (φ : Unitization R A →ₐ[R] C) (a : A) :
    Unitization.lift.symm φ a = φ a :=
  rfl


@[simp]
lemma _root_.NonUnitalAlgHom.toAlgHom_zero :
    ⇑(0 : A →ₙₐ[R] R).toAlgHom = Unitization.fst := by
  /-
    R : Type u_2
    A : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalSemiring A
    inst✝² : Module R A
    inst✝¹ : SMulCommClass R A A
    inst✝ : IsScalarTower R A A
    ⊢ Eq (⇑(NonUnitalAlgHom.toAlgHom 0)) Unitization.fst
  -/
  ext
  /-
    case h
    R : Type u_2
    A : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalSemiring A
    inst✝² : Module R A
    inst✝¹ : SMulCommClass R A A
    inst✝ : IsScalarTower R A A
    x✝ : Unitization R A
    ⊢ Eq ((NonUnitalAlgHom.toAlgHom 0) x✝) x✝.fst
  -/
  simp
  /-
    🎉 no goals
  -/


/-- See note [partially-applied ext lemmas] -/
@[ext]
theorem starAlgHom_ext {φ ψ : Unitization R A →⋆ₐ[R] C}
    (h : (φ : Unitization R A →⋆ₙₐ[R] C).comp (Unitization.inrNonUnitalStarAlgHom R A) =
      (ψ : Unitization R A →⋆ₙₐ[R] C).comp (Unitization.inrNonUnitalStarAlgHom R A)) :
    φ = ψ :=
  Unitization.algHom_ext'' <| DFunLike.congr_fun h


/-- Non-unital star algebra homomorphisms from `A` into a unital star `R`-algebra `C` lift uniquely
to `Unitization R A →⋆ₐ[R] C`. This is the universal property of the unitization. -/
@[simps! apply symm_apply apply_apply]
def starLift : (A →⋆ₙₐ[R] C) ≃ (Unitization R A →⋆ₐ[R] C) :=
{ toFun := fun φ ↦
  { toAlgHom := Unitization.lift φ.toNonUnitalAlgHom
    map_star' := fun x => by
      /-
        R : Type u_1
        A : Type u_2
        C : Type u_3
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : StarRing R
        inst✝⁸ : NonUnitalSemiring A
        inst✝⁷ : StarRing A
        inst✝⁶ : Module R A
        inst✝⁵ : SMulCommClass R A A
        inst✝⁴ : IsScalarTower R A A
        inst✝³ : Semiring C
        inst✝² : Algebra R C
        inst✝¹ : StarRing C
        inst✝ : StarModule R C
        φ : NonUnitalStarAlgHom R A C
        x : Unitization R A
        ⊢ Eq ((↑↑(Unitization.lift φ.toNonUnitalAlgHom).toRingHom).toFun (Star.star x) …
      -/
      induction x
      /-
        case inl_add_inr
        R : Type u_1
        A : Type u_2
        C : Type u_3
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : StarRing R
        inst✝⁸ : NonUnitalSemiring A
        inst✝⁷ : StarRing A
        inst✝⁶ : Module R A
        inst✝⁵ : SMulCommClass R A A
        inst✝⁴ : IsScalarTower R A A
        inst✝³ : Semiring C
        inst✝² : Algebra R C
        inst✝¹ : StarRing C
        inst✝ : StarModule R C
        φ : NonUnitalStarAlgHom R A C
        r✝ : R
        a✝ : A
        ⊢ Eq ((↑↑(Unitization.lift φ.toNonUnitalAlgHom).toRingHom).toFun (Star.star (H …
      -/
      simp [map_star] }
      /-
        🎉 no goals
      -/
  invFun := fun φ ↦ φ.toNonUnitalStarAlgHom.comp (inrNonUnitalStarAlgHom R A),
                          /-
                            R : Type u_1
                            A : Type u_2
                            C : Type u_3
                            inst✝¹⁰ : CommSemiring R
                            inst✝⁹ : StarRing R
                            inst✝⁸ : NonUnitalSemiring A
                            inst✝⁷ : StarRing A
                            inst✝⁶ : Module R A
                            inst✝⁵ : SMulCommClass R A A
                            inst✝⁴ : IsScalarTower R A A
                            inst✝³ : Semiring C
                            inst✝² : Algebra R C
                            inst✝¹ : StarRing C
                            inst✝ : StarModule R C
                            φ : NonUnitalStarAlgHom R A C
                            ⊢ Eq ((fun φ => φ.toNonUnitalStarAlgHom.comp (Unitization.inrNonUnitalStarAlgH …
                          -/
  left_inv := fun φ => by ext; simp,
                               /-
                                 🎉 no goals
                               -/
  right_inv := fun φ => Unitization.algHom_ext'' <| by
    /-
      R : Type u_1
      A : Type u_2
      C : Type u_3
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : StarRing R
      inst✝⁸ : NonUnitalSemiring A
      inst✝⁷ : StarRing A
      inst✝⁶ : Module R A
      inst✝⁵ : SMulCommClass R A A
      inst✝⁴ : IsScalarTower R A A
      inst✝³ : Semiring C
      inst✝² : Algebra R C
      inst✝¹ : StarRing C
      inst✝ : StarModule R C
      φ : StarAlgHom R (Unitization R A) C
      ⊢ ∀ (a : A), Eq (((fun φ => { toAlgHom := Unitization.lift φ.toNonUnitalAlgHom …
    -/
    simp }
    /-
      🎉 no goals
    -/


@[simp high]
theorem starLift_symm_apply_apply (φ : Unitization R A →⋆ₐ[R] C) (a : A) :
    Unitization.starLift.symm φ a = φ a :=
  rfl


/-- The functorial map on morphisms between the category of non-unital C⋆-algebras with non-unital
star homomorphisms and unital C⋆-algebras with unital star homomorphisms.

This sends `φ : A →⋆ₙₐ[R] B` to a map `Unitization R A →⋆ₐ[R] Unitization R B` given by the formula
`(r, a) ↦ (r, φ a)` (or perhaps more precisely,
`algebraMap R _ r + ↑a ↦ algebraMap R _ r + ↑(φ a)`). -/
@[simps!]
def starMap (φ : A →⋆ₙₐ[R] B) : Unitization R A →⋆ₐ[R] Unitization R B :=
  Unitization.starLift <| (Unitization.inrNonUnitalStarAlgHom R B).comp φ


@[simp high]
lemma starMap_inr (φ : A →⋆ₙₐ[R] B) (a : A) :
    starMap φ (inr a) = inr (φ a) := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝¹² : CommSemiring R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : StarRing A
    inst✝⁸ : Module R A
    inst✝⁷ : SMulCommClass R A A
    inst✝⁶ : IsScalarTower R A A
    inst✝⁵ : NonUnitalSemiring B
    inst✝⁴ : StarRing B
    inst✝³ : Module R B
    inst✝² : SMulCommClass R B B
    inst✝¹ : IsScalarTower R B B
    inst✝ : StarModule R B
    φ : NonUnitalStarAlgHom R A B
    a : A
    ⊢ Eq ((Unitization.starMap φ) ↑a) ↑(φ a)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp high]
lemma starMap_inl (φ : A →⋆ₙₐ[R] B) (r : R) :
    starMap φ (inl r) = algebraMap R (Unitization R B) r := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝¹² : CommSemiring R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : StarRing A
    inst✝⁸ : Module R A
    inst✝⁷ : SMulCommClass R A A
    inst✝⁶ : IsScalarTower R A A
    inst✝⁵ : NonUnitalSemiring B
    inst✝⁴ : StarRing B
    inst✝³ : Module R B
    inst✝² : SMulCommClass R B B
    inst✝¹ : IsScalarTower R B B
    inst✝ : StarModule R B
    φ : NonUnitalStarAlgHom R A B
    r : R
    ⊢ Eq ((Unitization.starMap φ) (Unitization.inl r)) ((algebraMap R (Unitization …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If `φ : A →⋆ₙₐ[R] B` is injective, the lift `starMap φ : Unitization R A →⋆ₐ[R] Unitization R B`
is also injective. -/
lemma starMap_injective {φ : A →⋆ₙₐ[R] B} (hφ : Function.Injective φ) :
    Function.Injective (starMap φ) := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝¹² : CommSemiring R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : StarRing A
    inst✝⁸ : Module R A
    inst✝⁷ : SMulCommClass R A A
    inst✝⁶ : IsScalarTower R A A
    inst✝⁵ : NonUnitalSemiring B
    inst✝⁴ : StarRing B
    inst✝³ : Module R B
    inst✝² : SMulCommClass R B B
    inst✝¹ : IsScalarTower R B B
    inst✝ : StarModule R B
    φ : NonUnitalStarAlgHom R A B
    hφ : Function.Injective ⇑φ
    ⊢ Function.Injective ⇑(Unitization.starMap φ)
  -/
  intro x y h
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝¹² : CommSemiring R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : StarRing A
    inst✝⁸ : Module R A
    inst✝⁷ : SMulCommClass R A A
    inst✝⁶ : IsScalarTower R A A
    inst✝⁵ : NonUnitalSemiring B
    inst✝⁴ : StarRing B
    inst✝³ : Module R B
    inst✝² : SMulCommClass R B B
    inst✝¹ : IsScalarTower R B B
    inst✝ : StarModule R B
    φ : NonUnitalStarAlgHom R A B
    hφ : Function.Injective ⇑φ
    x y : Unitization R A
    h : Eq ((Unitization.starMap φ) x) ((Unitization.starMap φ) y)
    ⊢ Eq x y
  -/
  ext
    /-
      case h1
      R : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝¹² : CommSemiring R
      inst✝¹¹ : StarRing R
      inst✝¹⁰ : NonUnitalSemiring A
      inst✝⁹ : StarRing A
      inst✝⁸ : Module R A
      inst✝⁷ : SMulCommClass R A A
      inst✝⁶ : IsScalarTower R A A
      inst✝⁵ : NonUnitalSemiring B
      inst✝⁴ : StarRing B
      inst✝³ : Module R B
      inst✝² : SMulCommClass R B B
      inst✝¹ : IsScalarTower R B B
      inst✝ : StarModule R B
      φ : NonUnitalStarAlgHom R A B
      hφ : Function.Injective ⇑φ
      x y : Unitization R A
      h : Eq ((Unitization.starMap φ) x) ((Unitization.starMap φ) y)
      ⊢ Eq x.fst y.fst
    -/
  · simpa using congr(fst $(h))
    /-
      🎉 no goals
    -/
    /-
      case h2
      R : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝¹² : CommSemiring R
      inst✝¹¹ : StarRing R
      inst✝¹⁰ : NonUnitalSemiring A
      inst✝⁹ : StarRing A
      inst✝⁸ : Module R A
      inst✝⁷ : SMulCommClass R A A
      inst✝⁶ : IsScalarTower R A A
      inst✝⁵ : NonUnitalSemiring B
      inst✝⁴ : StarRing B
      inst✝³ : Module R B
      inst✝² : SMulCommClass R B B
      inst✝¹ : IsScalarTower R B B
      inst✝ : StarModule R B
      φ : NonUnitalStarAlgHom R A B
      hφ : Function.Injective ⇑φ
      x y : Unitization R A
      h : Eq ((Unitization.starMap φ) x) ((Unitization.starMap φ) y)
      ⊢ Eq x.snd y.snd
    -/
  · exact hφ <| by simpa [algebraMap_eq_inl] using congr(snd $(h))
    /-
      🎉 no goals
    -/


/-- If `φ : A →⋆ₙₐ[R] B` is surjective, the lift
`starMap φ : Unitization R A →⋆ₐ[R] Unitization R B` is also surjective. -/
lemma starMap_surjective {φ : A →⋆ₙₐ[R] B} (hφ : Function.Surjective φ) :
    Function.Surjective (starMap φ) := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝¹² : CommSemiring R
    inst✝¹¹ : StarRing R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : StarRing A
    inst✝⁸ : Module R A
    inst✝⁷ : SMulCommClass R A A
    inst✝⁶ : IsScalarTower R A A
    inst✝⁵ : NonUnitalSemiring B
    inst✝⁴ : StarRing B
    inst✝³ : Module R B
    inst✝² : SMulCommClass R B B
    inst✝¹ : IsScalarTower R B B
    inst✝ : StarModule R B
    φ : NonUnitalStarAlgHom R A B
    hφ : Function.Surjective ⇑φ
    ⊢ Function.Surjective ⇑(Unitization.starMap φ)
  -/
  intro x
  induction x using Unitization.ind with
  | inl_add_inr r b =>
    obtain ⟨a, rfl⟩ := hφ b
    exact ⟨(r, a), by rfl⟩


/-- `starMap` is functorial: `starMap (ψ.comp φ) = (starMap ψ).comp (starMap φ)`. -/
lemma starMap_comp {φ : A →⋆ₙₐ[R] B} {ψ : B →⋆ₙₐ[R] C} :
    starMap (ψ.comp φ) = (starMap ψ).comp (starMap φ) := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    C : Type u_4
    inst✝¹⁸ : CommSemiring R
    inst✝¹⁷ : StarRing R
    inst✝¹⁶ : NonUnitalSemiring A
    inst✝¹⁵ : StarRing A
    inst✝¹⁴ : Module R A
    inst✝¹³ : SMulCommClass R A A
    inst✝¹² : IsScalarTower R A A
    inst✝¹¹ : NonUnitalSemiring B
    inst✝¹⁰ : StarRing B
    inst✝⁹ : Module R B
    inst✝⁸ : SMulCommClass R B B
    inst✝⁷ : IsScalarTower R B B
    inst✝⁶ : NonUnitalSemiring C
    inst✝⁵ : StarRing C
    inst✝⁴ : Module R C
    inst✝³ : SMulCommClass R C C
    inst✝² : IsScalarTower R C C
    inst✝¹ : StarModule R B
    inst✝ : StarModule R C
    φ : NonUnitalStarAlgHom R A B
    ψ : NonUnitalStarAlgHom R B C
    ⊢ Eq (Unitization.starMap (ψ.comp φ)) ((Unitization.starMap ψ).comp (Unitizati …
  -/
  ext; all_goals simp
       /-
         🎉 no goals
       -/


/-- `starMap` is functorial:
`starMap (NonUnitalStarAlgHom.id R B) = StarAlgHom.id R (Unitization R B)`. -/
@[simp]
lemma starMap_id : starMap (NonUnitalStarAlgHom.id R B) = StarAlgHom.id R (Unitization R B) := by
  /-
    R : Type u_1
    B : Type u_3
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : NonUnitalSemiring B
    inst✝⁴ : StarRing B
    inst✝³ : Module R B
    inst✝² : SMulCommClass R B B
    inst✝¹ : IsScalarTower R B B
    inst✝ : StarModule R B
    ⊢ Eq (Unitization.starMap (NonUnitalStarAlgHom.id R B)) (StarAlgHom.id R (Unit …
  -/
  ext; all_goals simp
       /-
         🎉 no goals
       -/


@[simp]
lemma isSelfAdjoint_inr : IsSelfAdjoint (a : Unitization R A) ↔ IsSelfAdjoint a := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : Semiring R
    inst✝¹ : StarAddMonoid R
    inst✝ : Star A
    a : A
    ⊢ Iff (IsSelfAdjoint ↑a) (IsSelfAdjoint a)
  -/
  simp only [isSelfAdjoint_iff, ← inr_star, ← inr_mul, inr_injective.eq_iff]
  /-
    🎉 no goals
  -/


alias ⟨_root_.IsSelfAdjoint.of_inr, _⟩ := isSelfAdjoint_inr


variable (R) in
lemma _root_.IsSelfAdjoint.inr (ha : IsSelfAdjoint a) : IsSelfAdjoint (a : Unitization R A) :=
  isSelfAdjoint_inr.mpr ha


@[simp]
lemma isStarNormal_inr : IsStarNormal (a : Unitization R A) ↔ IsStarNormal a := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁵ : Semiring R
    inst✝⁴ : StarAddMonoid R
    inst✝³ : Star A
    a : A
    inst✝² : AddCommMonoid A
    inst✝¹ : Mul A
    inst✝ : SMulWithZero R A
    ⊢ Iff (IsStarNormal ↑a) (IsStarNormal a)
  -/
  simp only [isStarNormal_iff, commute_iff_eq, ← inr_star, ← inr_mul, inr_injective.eq_iff]
  /-
    🎉 no goals
  -/


alias ⟨_root_.IsStarNormal.of_inr, _⟩ := isStarNormal_inr


variable (R a) in
instance instIsStarNormal (a : A) [IsStarNormal a] :
    IsStarNormal (a : Unitization R A) :=
  isStarNormal_inr.mpr ‹_›


