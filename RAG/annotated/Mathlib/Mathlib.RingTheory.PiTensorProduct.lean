instance instOne : One (⨂[R] i, A i) where
  one := tprod R 1


lemma one_def : 1 = tprod R (1 : Π i, A i) := rfl


instance instAddCommMonoidWithOne : AddCommMonoidWithOne (⨂[R] i, A i) where
  __ := inferInstanceAs (AddCommMonoid (⨂[R] i, A i))
  __ := instOne


attribute [aesop safe] mul_add mul_smul_comm smul_mul_assoc add_mul in
/--
The multiplication in tensor product of rings is induced by `(xᵢ) * (yᵢ) = (xᵢ * yᵢ)`
-/
def mul : (⨂[R] i, A i) →ₗ[R] (⨂[R] i, A i) →ₗ[R] (⨂[R] i, A i) :=
  PiTensorProduct.piTensorHomMap₂ <| tprod R fun _ ↦ LinearMap.mul _ _


@[simp] lemma mul_tprod_tprod (x y : (i : ι) → A i) :
    mul (tprod R x) (tprod R y) = tprod R (x * y) := by
  /-
    ι : Type u_1
    R : Type u_3
    A : ι → Type u_4
    inst✝⁴ : CommSemiring R
    inst✝³ : (i : ι) → NonUnitalNonAssocSemiring (A i)
    inst✝² : (i : ι) → Module R (A i)
    inst✝¹ : ∀ (i : ι), SMulCommClass R (A i) (A i)
    inst✝ : ∀ (i : ι), IsScalarTower R (A i) (A i)
    x y : (i : ι) → A i
    ⊢ Eq ((PiTensorProduct.mul ((PiTensorProduct.tprod R) x)) ((PiTensorProduct.tp …
  -/
  simp only [mul, piTensorHomMap₂_tprod_tprod_tprod, LinearMap.mul_apply', Pi.mul_def]
  /-
    🎉 no goals
  -/


instance instMul : Mul (⨂[R] i, A i) where
  mul x y := mul x y


lemma mul_def (x y : ⨂[R] i, A i) : x * y = mul x y := rfl


@[simp] lemma tprod_mul_tprod (x y : (i : ι) → A i) :
    tprod R x * tprod R y = tprod R (x * y) :=
  mul_tprod_tprod x y


theorem _root_.SemiconjBy.tprod {a₁ a₂ a₃ : Π i, A i}
    (ha : SemiconjBy a₁ a₂ a₃) :
    SemiconjBy (tprod R a₁) (tprod R a₂) (tprod R a₃) := by
  /-
    ι : Type u_1
    R : Type u_3
    A : ι → Type u_4
    inst✝⁴ : CommSemiring R
    inst✝³ : (i : ι) → NonUnitalNonAssocSemiring (A i)
    inst✝² : (i : ι) → Module R (A i)
    inst✝¹ : ∀ (i : ι), SMulCommClass R (A i) (A i)
    inst✝ : ∀ (i : ι), IsScalarTower R (A i) (A i)
    a₁ a₂ a₃ : (i : ι) → A i
    ha : SemiconjBy a₁ a₂ a₃
    ⊢ SemiconjBy ((PiTensorProduct.tprod R) a₁) ((PiTensorProduct.tprod R) a₂) ((P …
  -/
  rw [SemiconjBy, tprod_mul_tprod, tprod_mul_tprod, ha]
  /-
    🎉 no goals
  -/


nonrec theorem _root_.Commute.tprod {a₁ a₂ : Π i, A i} (ha : Commute a₁ a₂) :
    Commute (tprod R a₁) (tprod R a₂) :=
  ha.tprod


lemma smul_tprod_mul_smul_tprod (r s : R) (x y : Π i, A i) :
    (r • tprod R x) * (s • tprod R y) = (r * s) • tprod R (x * y) := by
  /-
    ι : Type u_1
    R : Type u_3
    A : ι → Type u_4
    inst✝⁴ : CommSemiring R
    inst✝³ : (i : ι) → NonUnitalNonAssocSemiring (A i)
    inst✝² : (i : ι) → Module R (A i)
    inst✝¹ : ∀ (i : ι), SMulCommClass R (A i) (A i)
    inst✝ : ∀ (i : ι), IsScalarTower R (A i) (A i)
    r s : R
    x y : (i : ι) → A i
    ⊢ Eq (HMul.hMul (HSMul.hSMul r ((PiTensorProduct.tprod R) x)) (HSMul.hSMul s ( …
  -/
  simp only [mul_def, map_smul, LinearMap.smul_apply, mul_tprod_tprod, mul_comm r s, mul_smul]
  /-
    🎉 no goals
  -/


instance instNonUnitalNonAssocSemiring : NonUnitalNonAssocSemiring (⨂[R] i, A i) where
  __ := instMul
  __ := inferInstanceAs (AddCommMonoid (⨂[R] i, A i))
  left_distrib _ _ _ := (mul _).map_add _ _
  right_distrib _ _ _ := mul.map_add₂ _ _ _
  zero_mul _ := mul.map_zero₂ _
  mul_zero _ := map_zero (mul _)


protected lemma one_mul (x : ⨂[R] i, A i) : mul (tprod R 1) x = x := by
  induction x using PiTensorProduct.induction_on with
  | smul_tprod => simp
  | add _ _ h1 h2 => simp [map_add, h1, h2]


protected lemma mul_one (x : ⨂[R] i, A i) : mul x (tprod R 1) = x := by
  induction x using PiTensorProduct.induction_on with
  | smul_tprod => simp
  | add _ _ h1 h2 => simp [h1, h2]


instance instNonAssocSemiring : NonAssocSemiring (⨂[R] i, A i) where
  __ := instNonUnitalNonAssocSemiring
  one_mul := PiTensorProduct.one_mul
  mul_one := PiTensorProduct.mul_one


variable (R) in
/-- `PiTensorProduct.tprod` as a `MonoidHom`. -/
@[simps]
def tprodMonoidHom : (Π i, A i) →* ⨂[R] i, A i where
  toFun := tprod R
  map_one' := rfl
  map_mul' x y := (tprod_mul_tprod x y).symm


protected lemma mul_assoc (x y z : ⨂[R] i, A i) : mul (mul x y) z = mul x (mul y z) := by
  -- restate as an equality of morphisms so that we can use `ext`
  suffices LinearMap.llcomp R _ _ _ mul ∘ₗ mul =
      (LinearMap.llcomp R _ _ _ LinearMap.lflip <| LinearMap.llcomp R _ _ _ mul.flip ∘ₗ mul).flip by
    exact DFunLike.congr_fun (DFunLike.congr_fun (DFunLike.congr_fun this x) y) z
  /-
    ι : Type u_1
    R : Type u_3
    A : ι → Type u_4
    inst✝⁴ : CommSemiring R
    inst✝³ : (i : ι) → NonUnitalSemiring (A i)
    inst✝² : (i : ι) → Module R (A i)
    inst✝¹ : ∀ (i : ι), SMulCommClass R (A i) (A i)
    inst✝ : ∀ (i : ι), IsScalarTower R (A i) (A i)
    x y z : PiTensorProduct R fun i => A i
    ⊢ Eq (((LinearMap.llcomp R (PiTensorProduct R fun i => A i) (PiTensorProduct R …
  -/
  ext x y z
  /-
    case H.H.H.H.H.H
    ι : Type u_1
    R : Type u_3
    A : ι → Type u_4
    inst✝⁴ : CommSemiring R
    inst✝³ : (i : ι) → NonUnitalSemiring (A i)
    inst✝² : (i : ι) → Module R (A i)
    inst✝¹ : ∀ (i : ι), SMulCommClass R (A i) (A i)
    inst✝ : ∀ (i : ι), IsScalarTower R (A i) (A i)
    x✝ y✝ z✝ : PiTensorProduct R fun i => A i
    x y z : (i : ι) → A i
    ⊢ Eq (((((((((LinearMap.llcomp R (PiTensorProduct R fun i => A i) (PiTensorPro …
  -/
  dsimp [← mul_def]
  /-
    case H.H.H.H.H.H
    ι : Type u_1
    R : Type u_3
    A : ι → Type u_4
    inst✝⁴ : CommSemiring R
    inst✝³ : (i : ι) → NonUnitalSemiring (A i)
    inst✝² : (i : ι) → Module R (A i)
    inst✝¹ : ∀ (i : ι), SMulCommClass R (A i) (A i)
    inst✝ : ∀ (i : ι), IsScalarTower R (A i) (A i)
    x✝ y✝ z✝ : PiTensorProduct R fun i => A i
    x y z : (i : ι) → A i
    ⊢ Eq (HMul.hMul (HMul.hMul ((PiTensorProduct.tprod R) x) ((PiTensorProduct.tpr …
  -/
  simpa only [tprod_mul_tprod] using congr_arg (tprod R) (mul_assoc x y z)
  /-
    🎉 no goals
  -/


instance instNonUnitalSemiring : NonUnitalSemiring (⨂[R] i, A i) where
  __ := instNonUnitalNonAssocSemiring
  mul_assoc := PiTensorProduct.mul_assoc


instance instSemiring : Semiring (⨂[R] i, A i) where
  __ := instNonUnitalSemiring
  __ := instNonAssocSemiring


instance instAlgebra : Algebra R' (⨂[R] i, A i) where
  __ := hasSMul'
  toFun := (· • 1)
                 /-
                   ι : Type u_1
                   R' : Type u_2
                   R : Type u_3
                   A : ι → Type u_4
                   inst✝⁶ : CommSemiring R'
                   inst✝⁵ : CommSemiring R
                   inst✝⁴ : (i : ι) → Semiring (A i)
                   inst✝³ : Algebra R' R
                   inst✝² : (i : ι) → Algebra R (A i)
                   inst✝¹ : (i : ι) → Algebra R' (A i)
                   inst✝ : ∀ (i : ι), IsScalarTower R' R (A i)
                   ⊢ Eq ((fun x => HSMul.hSMul x 1) 1) 1
                 -/
  map_one' := by simp
                 /-
                   🎉 no goals
                 -/
  map_mul' r s := show (r * s) • 1 = mul (r • 1) (s • 1)  by
    rw [LinearMap.map_smul_of_tower, LinearMap.map_smul_of_tower, LinearMap.smul_apply, mul_comm,
      mul_smul]
    /-
      ι : Type u_1
      R' : Type u_2
      R : Type u_3
      A : ι → Type u_4
      inst✝⁶ : CommSemiring R'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → Semiring (A i)
      inst✝³ : Algebra R' R
      inst✝² : (i : ι) → Algebra R (A i)
      inst✝¹ : (i : ι) → Algebra R' (A i)
      inst✝ : ∀ (i : ι), IsScalarTower R' R (A i)
      r s : R'
      ⊢ Eq (HSMul.hSMul s (HSMul.hSMul r 1)) (HSMul.hSMul s (HSMul.hSMul r ((PiTenso …
    -/
    congr
    /-
      case e_a.e_a
      ι : Type u_1
      R' : Type u_2
      R : Type u_3
      A : ι → Type u_4
      inst✝⁶ : CommSemiring R'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → Semiring (A i)
      inst✝³ : Algebra R' R
      inst✝² : (i : ι) → Algebra R (A i)
      inst✝¹ : (i : ι) → Algebra R' (A i)
      inst✝ : ∀ (i : ι), IsScalarTower R' R (A i)
      r s : R'
      ⊢ Eq 1 ((PiTensorProduct.mul 1) 1)
    -/
    show (1 : ⨂[R] i, A i) = 1 * 1
    /-
      case e_a.e_a
      ι : Type u_1
      R' : Type u_2
      R : Type u_3
      A : ι → Type u_4
      inst✝⁶ : CommSemiring R'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → Semiring (A i)
      inst✝³ : Algebra R' R
      inst✝² : (i : ι) → Algebra R (A i)
      inst✝¹ : (i : ι) → Algebra R' (A i)
      inst✝ : ∀ (i : ι), IsScalarTower R' R (A i)
      r s : R'
      ⊢ Eq 1 (HMul.hMul 1 1)
    -/
    rw [mul_one]
    /-
      🎉 no goals
    -/
                  /-
                    ι : Type u_1
                    R' : Type u_2
                    R : Type u_3
                    A : ι → Type u_4
                    inst✝⁶ : CommSemiring R'
                    inst✝⁵ : CommSemiring R
                    inst✝⁴ : (i : ι) → Semiring (A i)
                    inst✝³ : Algebra R' R
                    inst✝² : (i : ι) → Algebra R (A i)
                    inst✝¹ : (i : ι) → Algebra R' (A i)
                    inst✝ : ∀ (i : ι), IsScalarTower R' R (A i)
                    ⊢ Eq ((↑{ toFun := fun x => HSMul.hSMul x 1, map_one' := ⋯, map_mul' := ⋯ }).t …
                  -/
  map_zero' := by simp
                  /-
                    🎉 no goals
                  -/
                 /-
                   ι : Type u_1
                   R' : Type u_2
                   R : Type u_3
                   A : ι → Type u_4
                   inst✝⁶ : CommSemiring R'
                   inst✝⁵ : CommSemiring R
                   inst✝⁴ : (i : ι) → Semiring (A i)
                   inst✝³ : Algebra R' R
                   inst✝² : (i : ι) → Algebra R (A i)
                   inst✝¹ : (i : ι) → Algebra R' (A i)
                   inst✝ : ∀ (i : ι), IsScalarTower R' R (A i)
                   ⊢ ∀ (x y : R'), Eq ((↑{ toFun := fun x => HSMul.hSMul x 1, map_one' := ⋯, map_ …
                 -/
  map_add' := by simp [add_smul]
                 /-
                   🎉 no goals
                 -/
  commutes' r x := by
    /-
      ι : Type u_1
      R' : Type u_2
      R : Type u_3
      A : ι → Type u_4
      inst✝⁶ : CommSemiring R'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → Semiring (A i)
      inst✝³ : Algebra R' R
      inst✝² : (i : ι) → Algebra R (A i)
      inst✝¹ : (i : ι) → Algebra R' (A i)
      inst✝ : ∀ (i : ι), IsScalarTower R' R (A i)
      r : R'
      x : PiTensorProduct R fun i => A i
      ⊢ Eq (HMul.hMul ({ toFun := fun x => HSMul.hSMul x 1, map_one' := ⋯, map_mul'  …
    -/
    simp only [RingHom.coe_mk, MonoidHom.coe_mk, OneHom.coe_mk]
    /-
      ι : Type u_1
      R' : Type u_2
      R : Type u_3
      A : ι → Type u_4
      inst✝⁶ : CommSemiring R'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → Semiring (A i)
      inst✝³ : Algebra R' R
      inst✝² : (i : ι) → Algebra R (A i)
      inst✝¹ : (i : ι) → Algebra R' (A i)
      inst✝ : ∀ (i : ι), IsScalarTower R' R (A i)
      r : R'
      x : PiTensorProduct R fun i => A i
      ⊢ Eq (HMul.hMul (HSMul.hSMul r 1) x) (HMul.hMul x (HSMul.hSMul r 1))
    -/
    change mul _ _ = mul _ _
    /-
      ι : Type u_1
      R' : Type u_2
      R : Type u_3
      A : ι → Type u_4
      inst✝⁶ : CommSemiring R'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → Semiring (A i)
      inst✝³ : Algebra R' R
      inst✝² : (i : ι) → Algebra R (A i)
      inst✝¹ : (i : ι) → Algebra R' (A i)
      inst✝ : ∀ (i : ι), IsScalarTower R' R (A i)
      r : R'
      x : PiTensorProduct R fun i => A i
      ⊢ Eq ((PiTensorProduct.mul (HSMul.hSMul r 1)) x) ((PiTensorProduct.mul x) (HSM …
    -/
    rw [LinearMap.map_smul_of_tower, LinearMap.map_smul_of_tower, LinearMap.smul_apply]
    /-
      ι : Type u_1
      R' : Type u_2
      R : Type u_3
      A : ι → Type u_4
      inst✝⁶ : CommSemiring R'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → Semiring (A i)
      inst✝³ : Algebra R' R
      inst✝² : (i : ι) → Algebra R (A i)
      inst✝¹ : (i : ι) → Algebra R' (A i)
      inst✝ : ∀ (i : ι), IsScalarTower R' R (A i)
      r : R'
      x : PiTensorProduct R fun i => A i
      ⊢ Eq (HSMul.hSMul r ((PiTensorProduct.mul 1) x)) (HSMul.hSMul r ((PiTensorProd …
    -/
    change r • (1 * x) = r • (x * 1)
    /-
      ι : Type u_1
      R' : Type u_2
      R : Type u_3
      A : ι → Type u_4
      inst✝⁶ : CommSemiring R'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → Semiring (A i)
      inst✝³ : Algebra R' R
      inst✝² : (i : ι) → Algebra R (A i)
      inst✝¹ : (i : ι) → Algebra R' (A i)
      inst✝ : ∀ (i : ι), IsScalarTower R' R (A i)
      r : R'
      x : PiTensorProduct R fun i => A i
      ⊢ Eq (HSMul.hSMul r (HMul.hMul 1 x)) (HSMul.hSMul r (HMul.hMul x 1))
    -/
    rw [mul_one, one_mul]
    /-
      🎉 no goals
    -/
  smul_def' r x := by
    /-
      ι : Type u_1
      R' : Type u_2
      R : Type u_3
      A : ι → Type u_4
      inst✝⁶ : CommSemiring R'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → Semiring (A i)
      inst✝³ : Algebra R' R
      inst✝² : (i : ι) → Algebra R (A i)
      inst✝¹ : (i : ι) → Algebra R' (A i)
      inst✝ : ∀ (i : ι), IsScalarTower R' R (A i)
      r : R'
      x : PiTensorProduct R fun i => A i
      ⊢ Eq (HSMul.hSMul r x) (HMul.hMul ({ toFun := fun x => HSMul.hSMul x 1, map_on …
    -/
    simp only [RingHom.coe_mk, MonoidHom.coe_mk, OneHom.coe_mk]
    /-
      ι : Type u_1
      R' : Type u_2
      R : Type u_3
      A : ι → Type u_4
      inst✝⁶ : CommSemiring R'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → Semiring (A i)
      inst✝³ : Algebra R' R
      inst✝² : (i : ι) → Algebra R (A i)
      inst✝¹ : (i : ι) → Algebra R' (A i)
      inst✝ : ∀ (i : ι), IsScalarTower R' R (A i)
      r : R'
      x : PiTensorProduct R fun i => A i
      ⊢ Eq (HSMul.hSMul r x) (HMul.hMul (HSMul.hSMul r 1) x)
    -/
    change _ = mul _ _
    /-
      ι : Type u_1
      R' : Type u_2
      R : Type u_3
      A : ι → Type u_4
      inst✝⁶ : CommSemiring R'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → Semiring (A i)
      inst✝³ : Algebra R' R
      inst✝² : (i : ι) → Algebra R (A i)
      inst✝¹ : (i : ι) → Algebra R' (A i)
      inst✝ : ∀ (i : ι), IsScalarTower R' R (A i)
      r : R'
      x : PiTensorProduct R fun i => A i
      ⊢ Eq (HSMul.hSMul r x) ((PiTensorProduct.mul (HSMul.hSMul r 1)) x)
    -/
    rw [LinearMap.map_smul_of_tower, LinearMap.smul_apply]
    /-
      ι : Type u_1
      R' : Type u_2
      R : Type u_3
      A : ι → Type u_4
      inst✝⁶ : CommSemiring R'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → Semiring (A i)
      inst✝³ : Algebra R' R
      inst✝² : (i : ι) → Algebra R (A i)
      inst✝¹ : (i : ι) → Algebra R' (A i)
      inst✝ : ∀ (i : ι), IsScalarTower R' R (A i)
      r : R'
      x : PiTensorProduct R fun i => A i
      ⊢ Eq (HSMul.hSMul r x) (HSMul.hSMul r ((PiTensorProduct.mul 1) x))
    -/
    change _ = r • (1 * x)
    /-
      ι : Type u_1
      R' : Type u_2
      R : Type u_3
      A : ι → Type u_4
      inst✝⁶ : CommSemiring R'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → Semiring (A i)
      inst✝³ : Algebra R' R
      inst✝² : (i : ι) → Algebra R (A i)
      inst✝¹ : (i : ι) → Algebra R' (A i)
      inst✝ : ∀ (i : ι), IsScalarTower R' R (A i)
      r : R'
      x : PiTensorProduct R fun i => A i
      ⊢ Eq (HSMul.hSMul r x) (HSMul.hSMul r (HMul.hMul 1 x))
    -/
    rw [one_mul]
    /-
      🎉 no goals
    -/


lemma algebraMap_apply (r : R') (i : ι) [DecidableEq ι] :
    algebraMap R' (⨂[R] i, A i) r = tprod R (Pi.mulSingle i (algebraMap R' (A i) r)) := by
  /-
    ι : Type u_1
    R' : Type u_2
    R : Type u_3
    A : ι → Type u_4
    inst✝⁷ : CommSemiring R'
    inst✝⁶ : CommSemiring R
    inst✝⁵ : (i : ι) → Semiring (A i)
    inst✝⁴ : Algebra R' R
    inst✝³ : (i : ι) → Algebra R (A i)
    inst✝² : (i : ι) → Algebra R' (A i)
    inst✝¹ : ∀ (i : ι), IsScalarTower R' R (A i)
    r : R'
    i : ι
    inst✝ : DecidableEq ι
    ⊢ Eq ((algebraMap R' (PiTensorProduct R fun i => A i)) r) ((PiTensorProduct.tp …
  -/
  change r • tprod R 1 = _
  have : Pi.mulSingle i (algebraMap R' (A i) r) = update (fun i ↦ 1) i (r • 1) := by
    rw [Algebra.algebraMap_eq_smul_one]; rfl
  rw [this, ← smul_one_smul R r (1 : A i), MultilinearMap.map_update_smul, update_eq_self,
    smul_one_smul, Pi.one_def]


/--
The map `Aᵢ ⟶ ⨂ᵢ Aᵢ` given by `a ↦ 1 ⊗ ... ⊗ a ⊗ 1 ⊗ ...`
-/
@[simps]
def singleAlgHom [DecidableEq ι] (i : ι) : A i →ₐ[R] ⨂[R] i, A i where
  toFun a := tprod R (MonoidHom.mulSingle _ i a)
                 /-
                   ι : Type u_1
                   R' : Type u_2
                   R : Type u_3
                   A : ι → Type u_4
                   inst✝⁷ : CommSemiring R'
                   inst✝⁶ : CommSemiring R
                   inst✝⁵ : (i : ι) → Semiring (A i)
                   inst✝⁴ : Algebra R' R
                   inst✝³ : (i : ι) → Algebra R (A i)
                   inst✝² : (i : ι) → Algebra R' (A i)
                   inst✝¹ : ∀ (i : ι), IsScalarTower R' R (A i)
                   inst✝ : DecidableEq ι
                   i : ι
                   ⊢ Eq ((fun a => (PiTensorProduct.tprod R) ((MonoidHom.mulSingle A i) a)) 1) 1
                 -/
  map_one' := by simp only [_root_.map_one]; rfl
                                             /-
                                               🎉 no goals
                                             -/
                      /-
                        ι : Type u_1
                        R' : Type u_2
                        R : Type u_3
                        A : ι → Type u_4
                        inst✝⁷ : CommSemiring R'
                        inst✝⁶ : CommSemiring R
                        inst✝⁵ : (i : ι) → Semiring (A i)
                        inst✝⁴ : Algebra R' R
                        inst✝³ : (i : ι) → Algebra R (A i)
                        inst✝² : (i : ι) → Algebra R' (A i)
                        inst✝¹ : ∀ (i : ι), IsScalarTower R' R (A i)
                        inst✝ : DecidableEq ι
                        i : ι
                        a a' : A i
                        ⊢ Eq ({ toFun := fun a => (PiTensorProduct.tprod R) ((MonoidHom.mulSingle A i) …
                      -/
  map_mul' a a' := by simp [_root_.map_mul]
                      /-
                        🎉 no goals
                      -/
  map_zero' := MultilinearMap.map_update_zero _ _ _
  map_add' _ _ := MultilinearMap.map_update_add _ _ _ _ _
  commutes' r := show tprodCoeff R _ _ = r • tprodCoeff R _ _ by
    /-
      ι : Type u_1
      R' : Type u_2
      R : Type u_3
      A : ι → Type u_4
      inst✝⁷ : CommSemiring R'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : ι) → Semiring (A i)
      inst✝⁴ : Algebra R' R
      inst✝³ : (i : ι) → Algebra R (A i)
      inst✝² : (i : ι) → Algebra R' (A i)
      inst✝¹ : ∀ (i : ι), IsScalarTower R' R (A i)
      inst✝ : DecidableEq ι
      i : ι
      r : R
      ⊢ Eq (PiTensorProduct.tprodCoeff R 1 ((MonoidHom.mulSingle A i) ((algebraMap R …
    -/
    rw [Algebra.algebraMap_eq_smul_one]
    /-
      ι : Type u_1
      R' : Type u_2
      R : Type u_3
      A : ι → Type u_4
      inst✝⁷ : CommSemiring R'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : ι) → Semiring (A i)
      inst✝⁴ : Algebra R' R
      inst✝³ : (i : ι) → Algebra R (A i)
      inst✝² : (i : ι) → Algebra R' (A i)
      inst✝¹ : ∀ (i : ι), IsScalarTower R' R (A i)
      inst✝ : DecidableEq ι
      i : ι
      r : R
      ⊢ Eq (PiTensorProduct.tprodCoeff R 1 ((MonoidHom.mulSingle A i) (HSMul.hSMul r …
    -/
    erw [smul_tprodCoeff]
    /-
      ι : Type u_1
      R' : Type u_2
      R : Type u_3
      A : ι → Type u_4
      inst✝⁷ : CommSemiring R'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : ι) → Semiring (A i)
      inst✝⁴ : Algebra R' R
      inst✝³ : (i : ι) → Algebra R (A i)
      inst✝² : (i : ι) → Algebra R' (A i)
      inst✝¹ : ∀ (i : ι), IsScalarTower R' R (A i)
      inst✝ : DecidableEq ι
      i : ι
      r : R
      ⊢ Eq (PiTensorProduct.tprodCoeff R (HSMul.hSMul r 1) 1) (HSMul.hSMul r (PiTens …
    -/
    rfl
    /-
      🎉 no goals
    -/


/--
Lifting a multilinear map to an algebra homomorphism from tensor product
-/
@[simps!]
def liftAlgHom {S : Type*} [Semiring S] [Algebra R S]
    (f : MultilinearMap R A S)
    (one : f 1 = 1) (mul : ∀ x y, f (x * y) = f x * f y) : (⨂[R] i, A i) →ₐ[R] S :=
                                                              /-
                                                                ι : Type u_1
                                                                R' : Type u_2
                                                                R : Type u_3
                                                                A : ι → Type u_4
                                                                inst✝⁸ : CommSemiring R'
                                                                inst✝⁷ : CommSemiring R
                                                                inst✝⁶ : (i : ι) → Semiring (A i)
                                                                inst✝⁵ : Algebra R' R
                                                                inst✝⁴ : (i : ι) → Algebra R (A i)
                                                                inst✝³ : (i : ι) → Algebra R' (A i)
                                                                inst✝² : ∀ (i : ι), IsScalarTower R' R (A i)
                                                                S : Type u_5
                                                                inst✝¹ : Semiring S
                                                                inst✝ : Algebra R S
                                                                f : MultilinearMap R A S
                                                                one : Eq (f 1) 1
                                                                mul : ∀ (x y : (i : ι) → A i), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                                                ⊢ Eq ((PiTensorProduct.lift f) ((PiTensorProduct.tprod R) 1)) 1
                                                              -/
  AlgHom.ofLinearMap (lift f) (show lift f (tprod R 1) = 1 by simp [one]) <|
                                                              /-
                                                                🎉 no goals
                                                              -/
                                         /-
                                           ι : Type u_1
                                           R' : Type u_2
                                           R : Type u_3
                                           A : ι → Type u_4
                                           inst✝⁸ : CommSemiring R'
                                           inst✝⁷ : CommSemiring R
                                           inst✝⁶ : (i : ι) → Semiring (A i)
                                           inst✝⁵ : Algebra R' R
                                           inst✝⁴ : (i : ι) → Algebra R (A i)
                                           inst✝³ : (i : ι) → Algebra R' (A i)
                                           inst✝² : ∀ (i : ι), IsScalarTower R' R (A i)
                                           S : Type u_5
                                           inst✝¹ : Semiring S
                                           inst✝ : Algebra R S
                                           f : MultilinearMap R A S
                                           one : Eq (f 1) 1
                                           mul : ∀ (x y : (i : ι) → A i), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                                           ⊢ Eq ((LinearMap.mul R (PiTensorProduct R fun i => A i)).compr₂ (PiTensorProdu …
                                         -/
    LinearMap.map_mul_iff _ |>.mpr <| by aesop
                                         /-
                                           🎉 no goals
                                         -/


@[simp] lemma tprod_noncommProd {κ : Type*} (s : Finset κ) (x : κ → Π i, A i) (hx) :
    tprod R (s.noncommProd x hx) = s.noncommProd (fun k => tprod R (x k))
      (hx.imp fun _ _ => Commute.tprod) :=
  Finset.map_noncommProd s x _ (tprodMonoidHom R)


/-- To show two algebra morphisms from finite tensor products are equal, it suffices to show that
they agree on elements of the form $1 ⊗ ⋯ ⊗ a ⊗ 1 ⊗ ⋯$. -/
@[ext high]
theorem algHom_ext {S : Type*} [Finite ι] [DecidableEq ι] [Semiring S] [Algebra R S]
    ⦃f g : (⨂[R] i, A i) →ₐ[R] S⦄ (h : ∀ i, f.comp (singleAlgHom i) = g.comp (singleAlgHom i)) :
    f = g :=
  AlgHom.toLinearMap_injective <| PiTensorProduct.ext <| MultilinearMap.ext fun x =>
    suffices f.toMonoidHom.comp (tprodMonoidHom R) = g.toMonoidHom.comp (tprodMonoidHom R) from
      DFunLike.congr_fun this x
    MonoidHom.pi_ext fun i xi => DFunLike.congr_fun (h i) xi


instance instRing : Ring (⨂[R] i, A i) where
  __ := instSemiring
  __ := inferInstanceAs <| AddCommGroup (⨂[R] i, A i)


protected lemma mul_comm (x y : ⨂[R] i, A i) : mul x y = mul y x := by
  suffices mul (R := R) (A := A) = mul.flip from
    DFunLike.congr_fun (DFunLike.congr_fun this x) y
  /-
    ι : Type u_1
    R : Type u_3
    A : ι → Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : (i : ι) → CommSemiring (A i)
    inst✝ : (i : ι) → Algebra R (A i)
    x y : PiTensorProduct R fun i => A i
    ⊢ Eq PiTensorProduct.mul PiTensorProduct.mul.flip
  -/
  ext x y
  /-
    case H.H.H.H
    ι : Type u_1
    R : Type u_3
    A : ι → Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : (i : ι) → CommSemiring (A i)
    inst✝ : (i : ι) → Algebra R (A i)
    x✝ y✝ : PiTensorProduct R fun i => A i
    x y : (i : ι) → A i
    ⊢ Eq ((((PiTensorProduct.mul.compMultilinearMap (PiTensorProduct.tprod R)) x). …
  -/
  dsimp
  /-
    case H.H.H.H
    ι : Type u_1
    R : Type u_3
    A : ι → Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : (i : ι) → CommSemiring (A i)
    inst✝ : (i : ι) → Algebra R (A i)
    x✝ y✝ : PiTensorProduct R fun i => A i
    x y : (i : ι) → A i
    ⊢ Eq ((PiTensorProduct.mul ((PiTensorProduct.tprod R) x)) ((PiTensorProduct.tp …
  -/
  simp only [mul_tprod_tprod, mul_tprod_tprod, mul_comm x y]
  /-
    🎉 no goals
  -/


instance instCommSemiring : CommSemiring (⨂[R] i, A i) where
  __ := instSemiring
  __ := inferInstanceAs <| AddCommMonoid (⨂[R] i, A i)
  mul_comm := PiTensorProduct.mul_comm


@[simp] lemma tprod_prod {κ : Type*} (s : Finset κ) (x : κ → Π i, A i) :
    tprod R (∏ k ∈ s, x k) = ∏ k ∈ s, tprod R (x k) :=
  map_prod (tprodMonoidHom R) x s


/--
The algebra equivalence from the tensor product of the constant family with
value `R` to `R`, given by multiplication of the entries.
-/
noncomputable def constantBaseRingEquiv : (⨂[R] _ : ι, R) ≃ₐ[R] R :=
  letI toFun := lift (MultilinearMap.mkPiAlgebra R ι R)
  AlgEquiv.ofAlgHom
    (AlgHom.ofLinearMap
      toFun
      ((lift.tprod _).trans Finset.prod_const_one)
      (by
        -- one of these is required, the other is a performance optimization
        letI : IsScalarTower R (⨂[R] x : ι, R) (⨂[R] x : ι, R) :=
          IsScalarTower.right (R := R) (A := ⨂[R] (x : ι), R)
        letI : SMulCommClass R (⨂[R] x : ι, R) (⨂[R] x : ι, R) :=
          Algebra.to_smulCommClass (R := R) (A := ⨂[R] x : ι, R)
        /-
          ι : Type u_1
          R' : Type u_2
          R : Type u_3
          A : ι → Type u_4
          inst✝³ : CommSemiring R
          inst✝² : (i : ι) → CommSemiring (A i)
          inst✝¹ : (i : ι) → Algebra R (A i)
          inst✝ : Fintype ι
          toFun : LinearMap (RingHom.id R) (PiTensorProduct R fun i => R) R := PiTensorP …
          this✝ : IsScalarTower R (PiTensorProduct R fun x => R) (PiTensorProduct R fun  …
          this : SMulCommClass R (PiTensorProduct R fun x => R) (PiTensorProduct R fun x …
          ⊢ ∀ (x y : PiTensorProduct R fun x => R), Eq (toFun (HMul.hMul x y)) (HMul.hMu …
        -/
        rw [LinearMap.map_mul_iff]
        /-
          ι : Type u_1
          R' : Type u_2
          R : Type u_3
          A : ι → Type u_4
          inst✝³ : CommSemiring R
          inst✝² : (i : ι) → CommSemiring (A i)
          inst✝¹ : (i : ι) → Algebra R (A i)
          inst✝ : Fintype ι
          toFun : LinearMap (RingHom.id R) (PiTensorProduct R fun i => R) R := PiTensorP …
          this✝ : IsScalarTower R (PiTensorProduct R fun x => R) (PiTensorProduct R fun  …
          this : SMulCommClass R (PiTensorProduct R fun x => R) (PiTensorProduct R fun x …
          ⊢ Eq ((LinearMap.mul R (PiTensorProduct R fun x => R)).compr₂ toFun) (((Linear …
        -/
        ext x y
        /-
          case H.H.H.H
          ι : Type u_1
          R' : Type u_2
          R : Type u_3
          A : ι → Type u_4
          inst✝³ : CommSemiring R
          inst✝² : (i : ι) → CommSemiring (A i)
          inst✝¹ : (i : ι) → Algebra R (A i)
          inst✝ : Fintype ι
          toFun : LinearMap (RingHom.id R) (PiTensorProduct R fun i => R) R := PiTensorP …
          this✝ : IsScalarTower R (PiTensorProduct R fun x => R) (PiTensorProduct R fun  …
          this : SMulCommClass R (PiTensorProduct R fun x => R) (PiTensorProduct R fun x …
          x y : ι → R
          ⊢ Eq ((((((LinearMap.mul R (PiTensorProduct R fun x => R)).compr₂ toFun).compM …
        -/
        show toFun (tprod R x * tprod R y) = toFun (tprod R x) * toFun (tprod R y)
        simp_rw [tprod_mul_tprod, toFun, lift.tprod, MultilinearMap.mkPiAlgebra_apply,
          Pi.mul_apply, Finset.prod_mul_distrib]))
    (Algebra.ofId _ _)
        /-
          ι : Type u_1
          R' : Type u_2
          R : Type u_3
          A : ι → Type u_4
          inst✝³ : CommSemiring R
          inst✝² : (i : ι) → CommSemiring (A i)
          inst✝¹ : (i : ι) → Algebra R (A i)
          inst✝ : Fintype ι
          toFun : LinearMap (RingHom.id R) (PiTensorProduct R fun i => R) R := PiTensorP …
          ⊢ Eq ((AlgHom.ofLinearMap toFun ⋯ ⋯).comp (Algebra.ofId R (PiTensorProduct R f …
        -/
    (by ext)
        /-
          🎉 no goals
        -/
        /-
          ι : Type u_1
          R' : Type u_2
          R : Type u_3
          A : ι → Type u_4
          inst✝³ : CommSemiring R
          inst✝² : (i : ι) → CommSemiring (A i)
          inst✝¹ : (i : ι) → Algebra R (A i)
          inst✝ : Fintype ι
          toFun : LinearMap (RingHom.id R) (PiTensorProduct R fun i => R) R := PiTensorP …
          ⊢ Eq ((Algebra.ofId R (PiTensorProduct R fun x => R)).comp (AlgHom.ofLinearMap …
        -/
    (by classical ext)
        /-
          🎉 no goals
        -/


@[simp]
theorem constantBaseRingEquiv_tprod (x : ι → R) :
    constantBaseRingEquiv ι R (tprod R x) = ∏ i, x i := by
  /-
    ι : Type u_1
    R : Type u_3
    inst✝¹ : CommSemiring R
    inst✝ : Fintype ι
    x : ι → R
    ⊢ Eq ((PiTensorProduct.constantBaseRingEquiv ι R) ((PiTensorProduct.tprod R) x …
  -/
  simp [constantBaseRingEquiv]
  /-
    🎉 no goals
  -/


@[simp]
theorem constantBaseRingEquiv_symm (r : R) :
    (constantBaseRingEquiv ι R).symm r = algebraMap _ _ r := rfl


instance instCommRing : CommRing (⨂[R] i, A i) where
  __ := instCommSemiring
  __ := inferInstanceAs <| AddCommGroup (⨂[R] i, A i)


