/-- A version of `LinearMap.mulLeft` for matrix multiplication. -/
@[simps]
def mulLeftLinearMap (X : Matrix l m A) :
    Matrix m n A →ₗ[R] Matrix l n A where
  toFun := (X * ·)
  map_smul' := Matrix.mul_smul _
  map_add' := Matrix.mul_add _


/-- On square matrices, `Matrix.mulLeftLinearMap` and `LinearMap.mulLeft` coincide. -/
theorem mulLeftLinearMap_eq_mulLeft :
  mulLeftLinearMap m R = LinearMap.mulLeft R (A := Matrix m m A) := rfl


/-- A version of `LinearMap.mulLeft_zero_eq_zero` for matrix multiplication. -/
@[simp]
theorem mulLeftLinearMap_zero_eq_zero :
  mulLeftLinearMap n R (0 : Matrix l m A) = 0 := LinearMap.ext fun _ => Matrix.zero_mul _


/-- A version of `LinearMap.mulRight` for matrix multiplication. -/
@[simps]
def mulRightLinearMap (Y : Matrix m n A) :
    Matrix l m A →ₗ[R] Matrix l n A where
  toFun := (· * Y)
  map_smul' _ _ := Matrix.smul_mul _ _ _
  map_add' _ _ := Matrix.add_mul _ _ _


/-- On square matrices, `Matrix.mulRightLinearMap` and `LinearMap.mulRight` coincide. -/
theorem mulRightLinearMap_eq_mulRight :
  mulRightLinearMap m R = LinearMap.mulRight R (A := Matrix m m A) := rfl


/-- A version of `LinearMap.mulLeft_zero_eq_zero` for matrix multiplication. -/
@[simp]
theorem mulRightLinearMap_zero_eq_zero :
  mulRightLinearMap l R (0 : Matrix m n A) = 0 := LinearMap.ext fun _ => Matrix.mul_zero _


/-- A version of `LinearMap.mul` for matrix multiplication. -/
@[simps!]
def mulLinearMap : Matrix l m A →ₗ[R] Matrix m n A →ₗ[R] Matrix l n A where
  toFun := mulLeftLinearMap n R
  map_add' _ _ := LinearMap.ext fun _ => Matrix.add_mul _ _ _
  map_smul' _ _ := LinearMap.ext fun _ => Matrix.smul_mul _ _ _


/-- On square matrices, `Matrix.mulLinearMap` and `LinearMap.mul` coincide. -/
theorem mulLinearMap_eq_mul :
  mulLinearMap R = LinearMap.mul R (A := Matrix m m A) := rfl


/-- A version of `LinearMap.mulLeft_mul` for matrix multiplication. -/
@[simp]
theorem mulLeftLinearMap_mul [SMulCommClass R A A] (a : Matrix l m A) (b : Matrix m n A) :
    mulLeftLinearMap o R (a * b) = (mulLeftLinearMap o R a).comp (mulLeftLinearMap o R b) := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    R : Type u_5
    A : Type u_6
    inst✝⁵ : Fintype m
    inst✝⁴ : Fintype n
    inst✝³ : Semiring R
    inst✝² : NonUnitalSemiring A
    inst✝¹ : Module R A
    inst✝ : SMulCommClass R A A
    a : Matrix l m A
    b : Matrix m n A
    ⊢ Eq (mulLeftLinearMap o R (HMul.hMul a b)) ((mulLeftLinearMap o R a).comp (mu …
  -/
  ext
  /-
    case h.a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    R : Type u_5
    A : Type u_6
    inst✝⁵ : Fintype m
    inst✝⁴ : Fintype n
    inst✝³ : Semiring R
    inst✝² : NonUnitalSemiring A
    inst✝¹ : Module R A
    inst✝ : SMulCommClass R A A
    a : Matrix l m A
    b : Matrix m n A
    x✝ : Matrix n o A
    i✝ : l
    j✝ : o
    ⊢ Eq ((mulLeftLinearMap o R (HMul.hMul a b)) x✝ i✝ j✝) (((mulLeftLinearMap o R …
  -/
  simp only [mulLeftLinearMap_apply, LinearMap.comp_apply, Matrix.mul_assoc]
  /-
    🎉 no goals
  -/


/-- A version of `LinearMap.mulRight_mul` for matrix multiplication. -/
@[simp]
theorem mulRightLinearMap_mul [IsScalarTower R A A] (a : Matrix m n A) (b : Matrix n o A) :
    mulRightLinearMap l R (a * b) = (mulRightLinearMap l R b).comp (mulRightLinearMap l R a) := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    R : Type u_5
    A : Type u_6
    inst✝⁵ : Fintype m
    inst✝⁴ : Fintype n
    inst✝³ : Semiring R
    inst✝² : NonUnitalSemiring A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R A A
    a : Matrix m n A
    b : Matrix n o A
    ⊢ Eq (mulRightLinearMap l R (HMul.hMul a b)) ((mulRightLinearMap l R b).comp ( …
  -/
  ext
  /-
    case h.a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    R : Type u_5
    A : Type u_6
    inst✝⁵ : Fintype m
    inst✝⁴ : Fintype n
    inst✝³ : Semiring R
    inst✝² : NonUnitalSemiring A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R A A
    a : Matrix m n A
    b : Matrix n o A
    x✝ : Matrix l m A
    i✝ : l
    j✝ : o
    ⊢ Eq ((mulRightLinearMap l R (HMul.hMul a b)) x✝ i✝ j✝) (((mulRightLinearMap l …
  -/
  simp only [mulRightLinearMap_apply, LinearMap.comp_apply, Matrix.mul_assoc]
  /-
    🎉 no goals
  -/


/-- A version of `LinearMap.commute_mulLeft_right` for matrix multiplication. -/
theorem commute_mulLeftLinearMap_mulRightLinearMap (a : Matrix l m A) (b : Matrix n o A) :
    mulLeftLinearMap o R a ∘ₗ mulRightLinearMap m R b =
      mulRightLinearMap l R b ∘ₗ mulLeftLinearMap n R a := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    R : Type u_5
    A : Type u_6
    inst✝⁶ : Fintype m
    inst✝⁵ : Fintype n
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalSemiring A
    inst✝² : Module R A
    inst✝¹ : SMulCommClass R A A
    inst✝ : IsScalarTower R A A
    a : Matrix l m A
    b : Matrix n o A
    ⊢ Eq ((mulLeftLinearMap o R a).comp (mulRightLinearMap m R b)) ((mulRightLinea …
  -/
  ext c : 1
  /-
    case h
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    R : Type u_5
    A : Type u_6
    inst✝⁶ : Fintype m
    inst✝⁵ : Fintype n
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalSemiring A
    inst✝² : Module R A
    inst✝¹ : SMulCommClass R A A
    inst✝ : IsScalarTower R A A
    a : Matrix l m A
    b : Matrix n o A
    c : Matrix m n A
    ⊢ Eq (((mulLeftLinearMap o R a).comp (mulRightLinearMap m R b)) c) (((mulRight …
  -/
  exact (Matrix.mul_assoc a c b).symm
  /-
    🎉 no goals
  -/


/-- A version of `LinearMap.mulLeft_one` for matrix multiplication. -/
@[simp]
theorem mulLeftLinearMap_one : mulLeftLinearMap n R (1 : Matrix m m A) = LinearMap.id :=
  LinearMap.ext fun _ => Matrix.one_mul _


/-- A version of `LinearMap.mulLeft_eq_zero_iff` for matrix multiplication. -/
@[simp]
theorem mulLeftLinearMap_eq_zero_iff [Nonempty n] (a : Matrix l m A) :
    mulLeftLinearMap n R a = 0 ↔ a = 0 := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    R : Type u_5
    A : Type u_6
    inst✝⁶ : Fintype m
    inst✝⁵ : DecidableEq m
    inst✝⁴ : Semiring R
    inst✝³ : Semiring A
    inst✝² : Module R A
    inst✝¹ : SMulCommClass R A A
    inst✝ : Nonempty n
    a : Matrix l m A
    ⊢ Iff (Eq (mulLeftLinearMap n R a) 0) (Eq a 0)
  -/
  constructor <;> intro h
    /-
      case mp
      l : Type u_1
      m : Type u_2
      n : Type u_3
      R : Type u_5
      A : Type u_6
      inst✝⁶ : Fintype m
      inst✝⁵ : DecidableEq m
      inst✝⁴ : Semiring R
      inst✝³ : Semiring A
      inst✝² : Module R A
      inst✝¹ : SMulCommClass R A A
      inst✝ : Nonempty n
      a : Matrix l m A
      h : Eq (mulLeftLinearMap n R a) 0
      ⊢ Eq a 0
    -/
  · inhabit n
    /-
      case mp
      l : Type u_1
      m : Type u_2
      n : Type u_3
      R : Type u_5
      A : Type u_6
      inst✝⁶ : Fintype m
      inst✝⁵ : DecidableEq m
      inst✝⁴ : Semiring R
      inst✝³ : Semiring A
      inst✝² : Module R A
      inst✝¹ : SMulCommClass R A A
      inst✝ : Nonempty n
      a : Matrix l m A
      h : Eq (mulLeftLinearMap n R a) 0
      inhabited_h : Inhabited n
      ⊢ Eq a 0
    -/
    ext i j
    classical
    replace h := DFunLike.congr_fun h (Matrix.stdBasisMatrix j (default : n) 1)
    simpa using Matrix.ext_iff.2 h i default
    /-
      case mpr
      l : Type u_1
      m : Type u_2
      n : Type u_3
      R : Type u_5
      A : Type u_6
      inst✝⁶ : Fintype m
      inst✝⁵ : DecidableEq m
      inst✝⁴ : Semiring R
      inst✝³ : Semiring A
      inst✝² : Module R A
      inst✝¹ : SMulCommClass R A A
      inst✝ : Nonempty n
      a : Matrix l m A
      h : Eq a 0
      ⊢ Eq (mulLeftLinearMap n R a) 0
    -/
  · rw [h]
    /-
      case mpr
      l : Type u_1
      m : Type u_2
      n : Type u_3
      R : Type u_5
      A : Type u_6
      inst✝⁶ : Fintype m
      inst✝⁵ : DecidableEq m
      inst✝⁴ : Semiring R
      inst✝³ : Semiring A
      inst✝² : Module R A
      inst✝¹ : SMulCommClass R A A
      inst✝ : Nonempty n
      a : Matrix l m A
      h : Eq a 0
      ⊢ Eq (mulLeftLinearMap n R 0) 0
    -/
    exact mulLeftLinearMap_zero_eq_zero _ _
    /-
      🎉 no goals
    -/


/-- A version of `LinearMap.pow_mulLeft` for matrix multiplication. -/
@[simp]
theorem pow_mulLeftLinearMap (a : Matrix m m A) (k : ℕ) :
    mulLeftLinearMap n R a ^ k = mulLeftLinearMap n R (a ^ k) :=
  match k with
            /-
              m : Type u_2
              n : Type u_3
              R : Type u_5
              A : Type u_6
              inst✝⁵ : Fintype m
              inst✝⁴ : DecidableEq m
              inst✝³ : Semiring R
              inst✝² : Semiring A
              inst✝¹ : Module R A
              inst✝ : SMulCommClass R A A
              a : Matrix m m A
              k : Nat
              ⊢ Eq (HPow.hPow (mulLeftLinearMap n R a) 0) (mulLeftLinearMap n R (HPow.hPow a …
            -/
  | 0 => by rw [pow_zero, pow_zero, mulLeftLinearMap_one, LinearMap.one_eq_id]
            /-
              🎉 no goals
            -/
  | (n + 1) => by
    /-
      m : Type u_2
      n✝ : Type u_3
      R : Type u_5
      A : Type u_6
      inst✝⁵ : Fintype m
      inst✝⁴ : DecidableEq m
      inst✝³ : Semiring R
      inst✝² : Semiring A
      inst✝¹ : Module R A
      inst✝ : SMulCommClass R A A
      a : Matrix m m A
      k n : Nat
      ⊢ Eq (HPow.hPow (mulLeftLinearMap n✝ R a) (HAdd.hAdd n 1)) (mulLeftLinearMap n …
    -/
    rw [pow_succ, pow_succ, mulLeftLinearMap_mul, LinearMap.mul_eq_comp, pow_mulLeftLinearMap]
    /-
      🎉 no goals
    -/


/-- A version of `LinearMap.mulRight_one` for matrix multiplication. -/
@[simp]
theorem mulRightLinearMap_one : mulRightLinearMap l R (1 : Matrix m m A) = LinearMap.id :=
  LinearMap.ext fun _ => Matrix.mul_one _


/-- A version of `LinearMap.mulRight_eq_zero_iff` for matrix multiplication. -/
@[simp]
theorem mulRightLinearMap_eq_zero_iff (a : Matrix m n A) [Nonempty l] :
    mulRightLinearMap l R a = 0 ↔ a = 0 := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    R : Type u_5
    A : Type u_6
    inst✝⁶ : Fintype m
    inst✝⁵ : DecidableEq m
    inst✝⁴ : Semiring R
    inst✝³ : Semiring A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    a : Matrix m n A
    inst✝ : Nonempty l
    ⊢ Iff (Eq (mulRightLinearMap l R a) 0) (Eq a 0)
  -/
  constructor <;> intro h
    /-
      case mp
      l : Type u_1
      m : Type u_2
      n : Type u_3
      R : Type u_5
      A : Type u_6
      inst✝⁶ : Fintype m
      inst✝⁵ : DecidableEq m
      inst✝⁴ : Semiring R
      inst✝³ : Semiring A
      inst✝² : Module R A
      inst✝¹ : IsScalarTower R A A
      a : Matrix m n A
      inst✝ : Nonempty l
      h : Eq (mulRightLinearMap l R a) 0
      ⊢ Eq a 0
    -/
  · inhabit l
    /-
      case mp
      l : Type u_1
      m : Type u_2
      n : Type u_3
      R : Type u_5
      A : Type u_6
      inst✝⁶ : Fintype m
      inst✝⁵ : DecidableEq m
      inst✝⁴ : Semiring R
      inst✝³ : Semiring A
      inst✝² : Module R A
      inst✝¹ : IsScalarTower R A A
      a : Matrix m n A
      inst✝ : Nonempty l
      h : Eq (mulRightLinearMap l R a) 0
      inhabited_h : Inhabited l
      ⊢ Eq a 0
    -/
    ext i j
    classical
    replace h := DFunLike.congr_fun h (Matrix.stdBasisMatrix (default : l) i 1)
    simpa using Matrix.ext_iff.2 h default j
    /-
      case mpr
      l : Type u_1
      m : Type u_2
      n : Type u_3
      R : Type u_5
      A : Type u_6
      inst✝⁶ : Fintype m
      inst✝⁵ : DecidableEq m
      inst✝⁴ : Semiring R
      inst✝³ : Semiring A
      inst✝² : Module R A
      inst✝¹ : IsScalarTower R A A
      a : Matrix m n A
      inst✝ : Nonempty l
      h : Eq a 0
      ⊢ Eq (mulRightLinearMap l R a) 0
    -/
  · rw [h]
    /-
      case mpr
      l : Type u_1
      m : Type u_2
      n : Type u_3
      R : Type u_5
      A : Type u_6
      inst✝⁶ : Fintype m
      inst✝⁵ : DecidableEq m
      inst✝⁴ : Semiring R
      inst✝³ : Semiring A
      inst✝² : Module R A
      inst✝¹ : IsScalarTower R A A
      a : Matrix m n A
      inst✝ : Nonempty l
      h : Eq a 0
      ⊢ Eq (mulRightLinearMap l R 0) 0
    -/
    exact mulRightLinearMap_zero_eq_zero _ _
    /-
      🎉 no goals
    -/


/-- A version of `LinearMap.pow_mulRight` for matrix multiplication. -/
@[simp]
theorem pow_mulRightLinearMap (a : Matrix m m A) (k : ℕ) :
    mulRightLinearMap l R a ^ k = mulRightLinearMap l R (a ^ k) :=
  match k with
            /-
              l : Type u_1
              m : Type u_2
              R : Type u_5
              A : Type u_6
              inst✝⁵ : Fintype m
              inst✝⁴ : DecidableEq m
              inst✝³ : Semiring R
              inst✝² : Semiring A
              inst✝¹ : Module R A
              inst✝ : IsScalarTower R A A
              a : Matrix m m A
              k : Nat
              ⊢ Eq (HPow.hPow (mulRightLinearMap l R a) 0) (mulRightLinearMap l R (HPow.hPow …
            -/
  | 0 => by rw [pow_zero, pow_zero, mulRightLinearMap_one, LinearMap.one_eq_id]
            /-
              🎉 no goals
            -/
  | (n + 1) => by
    /-
      l : Type u_1
      m : Type u_2
      R : Type u_5
      A : Type u_6
      inst✝⁵ : Fintype m
      inst✝⁴ : DecidableEq m
      inst✝³ : Semiring R
      inst✝² : Semiring A
      inst✝¹ : Module R A
      inst✝ : IsScalarTower R A A
      a : Matrix m m A
      k n : Nat
      ⊢ Eq (HPow.hPow (mulRightLinearMap l R a) (HAdd.hAdd n 1)) (mulRightLinearMap  …
    -/
    rw [pow_succ, pow_succ', mulRightLinearMap_mul, LinearMap.mul_eq_comp, pow_mulRightLinearMap]
    /-
      🎉 no goals
    -/


