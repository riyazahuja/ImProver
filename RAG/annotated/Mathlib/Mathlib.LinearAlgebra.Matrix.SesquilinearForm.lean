/-- The map from `Matrix n n R` to bilinear maps on `n → R`.

This is an auxiliary definition for the equivalence `Matrix.toLinearMap₂'`. -/
def Matrix.toLinearMap₂'Aux (f : Matrix n m N₂) : (n → R₁) →ₛₗ[σ₁] (m → R₂) →ₛₗ[σ₂] N₂ :=
  -- porting note: we don't seem to have `∑ i j` as valid notation yet
  mk₂'ₛₗ σ₁ σ₂ (fun (v : n → R₁) (w : m → R₂) => ∑ i, ∑ j, σ₂ (w j) • σ₁ (v i) • f i j)
                     /-
                       R : Type u_1
                       R₁ : Type u_2
                       S₁ : Type u_3
                       R₂ : Type u_4
                       S₂ : Type u_5
                       M₁ : Type u_6
                       M₂ : Type u_7
                       M₁' : Type u_8
                       M₂' : Type u_9
                       N₂ : Type u_10
                       n : Type u_11
                       m : Type u_12
                       n' : Type u_13
                       m' : Type u_14
                       ι : Type u_15
                       inst✝⁹ : Semiring R₁
                       inst✝⁸ : Semiring S₁
                       inst✝⁷ : Semiring R₂
                       inst✝⁶ : Semiring S₂
                       inst✝⁵ : AddCommMonoid N₂
                       inst✝⁴ : Module S₁ N₂
                       inst✝³ : Module S₂ N₂
                       inst✝² : SMulCommClass S₂ S₁ N₂
                       inst✝¹ : Fintype n
                       inst✝ : Fintype m
                       σ₁ : RingHom R₁ S₁
                       σ₂ : RingHom R₂ S₂
                       f : Matrix n m N₂
                       x✝² x✝¹ : n → R₁
                       x✝ : m → R₂
                       ⊢ Eq ((fun v w => Finset.univ.sum fun i => Finset.univ.sum fun j => HSMul.hSMu …
                     -/
    (fun _ _ _ => by simp only [Pi.add_apply, map_add, smul_add, sum_add_distrib, add_smul])
                     /-
                       🎉 no goals
                     -/
    (fun c v w => by
      simp only [Pi.smul_apply, smul_sum, smul_eq_mul, σ₁.map_mul, ← smul_comm _ (σ₁ c),
        MulAction.mul_smul])
                     /-
                       R : Type u_1
                       R₁ : Type u_2
                       S₁ : Type u_3
                       R₂ : Type u_4
                       S₂ : Type u_5
                       M₁ : Type u_6
                       M₂ : Type u_7
                       M₁' : Type u_8
                       M₂' : Type u_9
                       N₂ : Type u_10
                       n : Type u_11
                       m : Type u_12
                       n' : Type u_13
                       m' : Type u_14
                       ι : Type u_15
                       inst✝⁹ : Semiring R₁
                       inst✝⁸ : Semiring S₁
                       inst✝⁷ : Semiring R₂
                       inst✝⁶ : Semiring S₂
                       inst✝⁵ : AddCommMonoid N₂
                       inst✝⁴ : Module S₁ N₂
                       inst✝³ : Module S₂ N₂
                       inst✝² : SMulCommClass S₂ S₁ N₂
                       inst✝¹ : Fintype n
                       inst✝ : Fintype m
                       σ₁ : RingHom R₁ S₁
                       σ₂ : RingHom R₂ S₂
                       f : Matrix n m N₂
                       x✝² : n → R₁
                       x✝¹ x✝ : m → R₂
                       ⊢ Eq ((fun v w => Finset.univ.sum fun i => Finset.univ.sum fun j => HSMul.hSMu …
                     -/
    (fun _ _ _ => by simp only [Pi.add_apply, map_add, add_smul, smul_add, sum_add_distrib])
                     /-
                       🎉 no goals
                     -/
    (fun _ v w => by
      /-
        R : Type u_1
        R₁ : Type u_2
        S₁ : Type u_3
        R₂ : Type u_4
        S₂ : Type u_5
        M₁ : Type u_6
        M₂ : Type u_7
        M₁' : Type u_8
        M₂' : Type u_9
        N₂ : Type u_10
        n : Type u_11
        m : Type u_12
        n' : Type u_13
        m' : Type u_14
        ι : Type u_15
        inst✝⁹ : Semiring R₁
        inst✝⁸ : Semiring S₁
        inst✝⁷ : Semiring R₂
        inst✝⁶ : Semiring S₂
        inst✝⁵ : AddCommMonoid N₂
        inst✝⁴ : Module S₁ N₂
        inst✝³ : Module S₂ N₂
        inst✝² : SMulCommClass S₂ S₁ N₂
        inst✝¹ : Fintype n
        inst✝ : Fintype m
        σ₁ : RingHom R₁ S₁
        σ₂ : RingHom R₂ S₂
        f : Matrix n m N₂
        x✝ : R₂
        v : n → R₁
        w : m → R₂
        ⊢ Eq ((fun v w => Finset.univ.sum fun i => Finset.univ.sum fun j => HSMul.hSMu …
      -/
      simp only [Pi.smul_apply, smul_eq_mul, _root_.map_mul, MulAction.mul_smul, smul_sum])
      /-
        🎉 no goals
      -/


theorem Matrix.toLinearMap₂'Aux_single (f : Matrix n m N₂) (i : n) (j : m) :
    f.toLinearMap₂'Aux σ₁ σ₂ (Pi.single i 1) (Pi.single j 1) = f i j := by
  /-
    R₁ : Type u_2
    S₁ : Type u_3
    R₂ : Type u_4
    S₂ : Type u_5
    N₂ : Type u_10
    n : Type u_11
    m : Type u_12
    inst✝¹¹ : Semiring R₁
    inst✝¹⁰ : Semiring S₁
    inst✝⁹ : Semiring R₂
    inst✝⁸ : Semiring S₂
    inst✝⁷ : AddCommMonoid N₂
    inst✝⁶ : Module S₁ N₂
    inst✝⁵ : Module S₂ N₂
    inst✝⁴ : SMulCommClass S₂ S₁ N₂
    inst✝³ : Fintype n
    inst✝² : Fintype m
    σ₁ : RingHom R₁ S₁
    σ₂ : RingHom R₂ S₂
    inst✝¹ : DecidableEq n
    inst✝ : DecidableEq m
    f : Matrix n m N₂
    i : n
    j : m
    ⊢ Eq (((Matrix.toLinearMap₂'Aux σ₁ σ₂ f) (Pi.single i 1)) (Pi.single j 1)) (f  …
  -/
  rw [Matrix.toLinearMap₂'Aux, mk₂'ₛₗ_apply]
  have : (∑ i', ∑ j', (if i = i' then (1 : S₁) else (0 : S₁)) •
        (if j = j' then (1 : S₂) else (0 : S₂)) • f i' j') =
      f i j := by
    simp_rw [← Finset.smul_sum]
    simp only [op_smul_eq_smul, ite_smul, one_smul, zero_smul, sum_ite_eq, mem_univ, ↓reduceIte]
  /-
    R₁ : Type u_2
    S₁ : Type u_3
    R₂ : Type u_4
    S₂ : Type u_5
    N₂ : Type u_10
    n : Type u_11
    m : Type u_12
    inst✝¹¹ : Semiring R₁
    inst✝¹⁰ : Semiring S₁
    inst✝⁹ : Semiring R₂
    inst✝⁸ : Semiring S₂
    inst✝⁷ : AddCommMonoid N₂
    inst✝⁶ : Module S₁ N₂
    inst✝⁵ : Module S₂ N₂
    inst✝⁴ : SMulCommClass S₂ S₁ N₂
    inst✝³ : Fintype n
    inst✝² : Fintype m
    σ₁ : RingHom R₁ S₁
    σ₂ : RingHom R₂ S₂
    inst✝¹ : DecidableEq n
    inst✝ : DecidableEq m
    f : Matrix n m N₂
    i : n
    j : m
    this : Eq (Finset.univ.sum fun i' => Finset.univ.sum fun j' => HSMul.hSMul (it …
    ⊢ Eq (Finset.univ.sum fun i_1 => Finset.univ.sum fun j_1 => HSMul.hSMul (σ₂ (P …
  -/
  rw [← this]
  /-
    R₁ : Type u_2
    S₁ : Type u_3
    R₂ : Type u_4
    S₂ : Type u_5
    N₂ : Type u_10
    n : Type u_11
    m : Type u_12
    inst✝¹¹ : Semiring R₁
    inst✝¹⁰ : Semiring S₁
    inst✝⁹ : Semiring R₂
    inst✝⁸ : Semiring S₂
    inst✝⁷ : AddCommMonoid N₂
    inst✝⁶ : Module S₁ N₂
    inst✝⁵ : Module S₂ N₂
    inst✝⁴ : SMulCommClass S₂ S₁ N₂
    inst✝³ : Fintype n
    inst✝² : Fintype m
    σ₁ : RingHom R₁ S₁
    σ₂ : RingHom R₂ S₂
    inst✝¹ : DecidableEq n
    inst✝ : DecidableEq m
    f : Matrix n m N₂
    i : n
    j : m
    this : Eq (Finset.univ.sum fun i' => Finset.univ.sum fun j' => HSMul.hSMul (it …
    ⊢ Eq (Finset.univ.sum fun i_1 => Finset.univ.sum fun j_1 => HSMul.hSMul (σ₂ (P …
  -/
  exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by aesop
  /-
    🎉 no goals
  -/


/-- The linear map from sesquilinear maps to `Matrix n m N₂` given an `n`-indexed basis for `M₁`
and an `m`-indexed basis for `M₂`.

This is an auxiliary definition for the equivalence `Matrix.toLinearMapₛₗ₂'`. -/
def LinearMap.toMatrix₂Aux (b₁ : n → M₁) (b₂ : m → M₂) :
    (M₁ →ₛₗ[σ₁] M₂ →ₛₗ[σ₂] N₂) →ₗ[R] Matrix n m N₂ where
  toFun f := of fun i j => f (b₁ i) (b₂ j)
  map_add' _f _g := rfl
  map_smul' _f _g := rfl


@[simp]
theorem LinearMap.toMatrix₂Aux_apply (f : M₁ →ₛₗ[σ₁] M₂ →ₛₗ[σ₂] N₂) (b₁ : n → M₁) (b₂ : m → M₂)
    (i : n) (j : m) : LinearMap.toMatrix₂Aux R b₁ b₂ f i j = f (b₁ i) (b₂ j) :=
  rfl


theorem LinearMap.toLinearMap₂'Aux_toMatrix₂Aux (f : (n → R₁) →ₛₗ[σ₁] (m → R₂) →ₛₗ[σ₂] N₂) :
    Matrix.toLinearMap₂'Aux σ₁ σ₂
        (LinearMap.toMatrix₂Aux R (fun i => Pi.single i 1) (fun j => Pi.single j 1) f) =
      f := by
  /-
    R : Type u_1
    R₁ : Type u_2
    S₁ : Type u_3
    R₂ : Type u_4
    S₂ : Type u_5
    N₂ : Type u_10
    n : Type u_11
    m : Type u_12
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : Semiring R₁
    inst✝¹³ : Semiring S₁
    inst✝¹² : Semiring R₂
    inst✝¹¹ : Semiring S₂
    inst✝¹⁰ : AddCommMonoid N₂
    inst✝⁹ : Module R N₂
    inst✝⁸ : Module S₁ N₂
    inst✝⁷ : Module S₂ N₂
    inst✝⁶ : SMulCommClass S₁ R N₂
    inst✝⁵ : SMulCommClass S₂ R N₂
    inst✝⁴ : SMulCommClass S₂ S₁ N₂
    σ₁ : RingHom R₁ S₁
    σ₂ : RingHom R₂ S₂
    inst✝³ : Fintype n
    inst✝² : Fintype m
    inst✝¹ : DecidableEq n
    inst✝ : DecidableEq m
    f : LinearMap σ₁ (n → R₁) (LinearMap σ₂ (m → R₂) N₂)
    ⊢ Eq (Matrix.toLinearMap₂'Aux σ₁ σ₂ ((LinearMap.toMatrix₂Aux R (fun i => Pi.si …
  -/
  refine ext_basis (Pi.basisFun R₁ n) (Pi.basisFun R₂ m) fun i j => ?_
  /-
    R : Type u_1
    R₁ : Type u_2
    S₁ : Type u_3
    R₂ : Type u_4
    S₂ : Type u_5
    N₂ : Type u_10
    n : Type u_11
    m : Type u_12
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : Semiring R₁
    inst✝¹³ : Semiring S₁
    inst✝¹² : Semiring R₂
    inst✝¹¹ : Semiring S₂
    inst✝¹⁰ : AddCommMonoid N₂
    inst✝⁹ : Module R N₂
    inst✝⁸ : Module S₁ N₂
    inst✝⁷ : Module S₂ N₂
    inst✝⁶ : SMulCommClass S₁ R N₂
    inst✝⁵ : SMulCommClass S₂ R N₂
    inst✝⁴ : SMulCommClass S₂ S₁ N₂
    σ₁ : RingHom R₁ S₁
    σ₂ : RingHom R₂ S₂
    inst✝³ : Fintype n
    inst✝² : Fintype m
    inst✝¹ : DecidableEq n
    inst✝ : DecidableEq m
    f : LinearMap σ₁ (n → R₁) (LinearMap σ₂ (m → R₂) N₂)
    i : n
    j : m
    ⊢ Eq (((Matrix.toLinearMap₂'Aux σ₁ σ₂ ((LinearMap.toMatrix₂Aux R (fun i => Pi. …
  -/
  simp_rw [Pi.basisFun_apply, Matrix.toLinearMap₂'Aux_single, LinearMap.toMatrix₂Aux_apply]
  /-
    🎉 no goals
  -/


theorem Matrix.toMatrix₂Aux_toLinearMap₂'Aux (f : Matrix n m N₂) :
    LinearMap.toMatrix₂Aux R (fun i => Pi.single i 1)
        (fun j => Pi.single j 1) (f.toLinearMap₂'Aux σ₁ σ₂) =
      f := by
  /-
    R : Type u_1
    R₁ : Type u_2
    S₁ : Type u_3
    R₂ : Type u_4
    S₂ : Type u_5
    N₂ : Type u_10
    n : Type u_11
    m : Type u_12
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : Semiring R₁
    inst✝¹³ : Semiring S₁
    inst✝¹² : Semiring R₂
    inst✝¹¹ : Semiring S₂
    inst✝¹⁰ : AddCommMonoid N₂
    inst✝⁹ : Module R N₂
    inst✝⁸ : Module S₁ N₂
    inst✝⁷ : Module S₂ N₂
    inst✝⁶ : SMulCommClass S₁ R N₂
    inst✝⁵ : SMulCommClass S₂ R N₂
    inst✝⁴ : SMulCommClass S₂ S₁ N₂
    σ₁ : RingHom R₁ S₁
    σ₂ : RingHom R₂ S₂
    inst✝³ : Fintype n
    inst✝² : Fintype m
    inst✝¹ : DecidableEq n
    inst✝ : DecidableEq m
    f : Matrix n m N₂
    ⊢ Eq ((LinearMap.toMatrix₂Aux R (fun i => Pi.single i 1) fun j => Pi.single j  …
  -/
  ext i j
  /-
    case a
    R : Type u_1
    R₁ : Type u_2
    S₁ : Type u_3
    R₂ : Type u_4
    S₂ : Type u_5
    N₂ : Type u_10
    n : Type u_11
    m : Type u_12
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : Semiring R₁
    inst✝¹³ : Semiring S₁
    inst✝¹² : Semiring R₂
    inst✝¹¹ : Semiring S₂
    inst✝¹⁰ : AddCommMonoid N₂
    inst✝⁹ : Module R N₂
    inst✝⁸ : Module S₁ N₂
    inst✝⁷ : Module S₂ N₂
    inst✝⁶ : SMulCommClass S₁ R N₂
    inst✝⁵ : SMulCommClass S₂ R N₂
    inst✝⁴ : SMulCommClass S₂ S₁ N₂
    σ₁ : RingHom R₁ S₁
    σ₂ : RingHom R₂ S₂
    inst✝³ : Fintype n
    inst✝² : Fintype m
    inst✝¹ : DecidableEq n
    inst✝ : DecidableEq m
    f : Matrix n m N₂
    i : n
    j : m
    ⊢ Eq ((LinearMap.toMatrix₂Aux R (fun i => Pi.single i 1) fun j => Pi.single j  …
  -/
  simp_rw [LinearMap.toMatrix₂Aux_apply, Matrix.toLinearMap₂'Aux_single]
  /-
    🎉 no goals
  -/


/-- The linear equivalence between sesquilinear maps and `n × m` matrices -/
def LinearMap.toMatrixₛₗ₂' : ((n → R₁) →ₛₗ[σ₁] (m → R₂) →ₛₗ[σ₂] N₂) ≃ₗ[R] Matrix n m N₂ :=
  { LinearMap.toMatrix₂Aux R (fun i => Pi.single i 1) (fun j => Pi.single j 1) with
    toFun := LinearMap.toMatrix₂Aux R _ _
    invFun := Matrix.toLinearMap₂'Aux σ₁ σ₂
    left_inv := LinearMap.toLinearMap₂'Aux_toMatrix₂Aux R
    right_inv := Matrix.toMatrix₂Aux_toLinearMap₂'Aux R }


/-- The linear equivalence between bilinear maps and `n × m` matrices -/
def LinearMap.toMatrix₂' : ((n → S₁) →ₗ[S₁] (m → S₂) →ₗ[S₂] N₂) ≃ₗ[R] Matrix n m N₂ :=
  LinearMap.toMatrixₛₗ₂' R


/-- The linear equivalence between `n × n` matrices and sesquilinear maps on `n → R` -/
def Matrix.toLinearMapₛₗ₂' : Matrix n m N₂ ≃ₗ[R] (n → R₁) →ₛₗ[σ₁] (m → R₂) →ₛₗ[σ₂] N₂ :=
  (LinearMap.toMatrixₛₗ₂' R).symm


/-- The linear equivalence between `n × n` matrices and bilinear maps on `n → R` -/
def Matrix.toLinearMap₂' : Matrix n m N₂ ≃ₗ[R] (n → S₁) →ₗ[S₁] (m → S₂) →ₗ[S₂] N₂ :=
  (LinearMap.toMatrix₂' R).symm


theorem Matrix.toLinearMapₛₗ₂'_aux_eq (M : Matrix n m N₂) :
    Matrix.toLinearMap₂'Aux σ₁ σ₂ M = Matrix.toLinearMapₛₗ₂' R σ₁ σ₂ M :=
  rfl


theorem Matrix.toLinearMapₛₗ₂'_apply (M : Matrix n m N₂) (x : n → R₁) (y : m → R₂) :
    -- porting note: we don't seem to have `∑ i j` as valid notation yet
    Matrix.toLinearMapₛₗ₂' R σ₁ σ₂ M x y = ∑ i, ∑ j, σ₁ (x i) •  σ₂ (y j) • M i j := by
  /-
    R : Type u_1
    R₁ : Type u_2
    S₁ : Type u_3
    R₂ : Type u_4
    S₂ : Type u_5
    N₂ : Type u_10
    n : Type u_11
    m : Type u_12
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid N₂
    inst✝¹³ : Module R N₂
    inst✝¹² : Semiring R₁
    inst✝¹¹ : Semiring R₂
    inst✝¹⁰ : Semiring S₁
    inst✝⁹ : Semiring S₂
    inst✝⁸ : Module S₁ N₂
    inst✝⁷ : Module S₂ N₂
    inst✝⁶ : SMulCommClass S₁ R N₂
    inst✝⁵ : SMulCommClass S₂ R N₂
    inst✝⁴ : SMulCommClass S₂ S₁ N₂
    σ₁ : RingHom R₁ S₁
    σ₂ : RingHom R₂ S₂
    inst✝³ : Fintype n
    inst✝² : Fintype m
    inst✝¹ : DecidableEq n
    inst✝ : DecidableEq m
    M : Matrix n m N₂
    x : n → R₁
    y : m → R₂
    ⊢ Eq ((((Matrix.toLinearMapₛₗ₂' R σ₁ σ₂) M) x) y) (Finset.univ.sum fun i => Fi …
  -/
  rw [toLinearMapₛₗ₂', toMatrixₛₗ₂', LinearEquiv.coe_symm_mk, toLinearMap₂'Aux, mk₂'ₛₗ_apply]
  apply Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by
    rw [smul_comm]


theorem Matrix.toLinearMap₂'_apply (M : Matrix n m N₂) (x : n → S₁) (y : m → S₂) :
    -- porting note: we don't seem to have `∑ i j` as valid notation yet
    Matrix.toLinearMap₂' R M x y = ∑ i, ∑ j, x i • y j • M i j :=
  Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by
    /-
      R : Type u_1
      S₁ : Type u_3
      S₂ : Type u_5
      N₂ : Type u_10
      n : Type u_11
      m : Type u_12
      inst✝¹³ : CommSemiring R
      inst✝¹² : AddCommMonoid N₂
      inst✝¹¹ : Module R N₂
      inst✝¹⁰ : Semiring S₁
      inst✝⁹ : Semiring S₂
      inst✝⁸ : Module S₁ N₂
      inst✝⁷ : Module S₂ N₂
      inst✝⁶ : SMulCommClass S₁ R N₂
      inst✝⁵ : SMulCommClass S₂ R N₂
      inst✝⁴ : SMulCommClass S₂ S₁ N₂
      inst✝³ : Fintype n
      inst✝² : Fintype m
      inst✝¹ : DecidableEq n
      inst✝ : DecidableEq m
      M : Matrix n m N₂
      x : n → S₁
      y : m → S₂
      x✝³ : n
      x✝² : Membership.mem Finset.univ x✝³
      x✝¹ : m
      x✝ : Membership.mem Finset.univ x✝¹
      ⊢ Eq (HSMul.hSMul ((RingHom.id S₂) (y x✝¹)) (HSMul.hSMul ((RingHom.id S₁) (x x …
    -/
    rw [RingHom.id_apply, RingHom.id_apply, smul_comm]
    /-
      🎉 no goals
    -/


theorem Matrix.toLinearMap₂'_apply' {T : Type*} [CommSemiring T] (M : Matrix n m T) (v : n → T)
    (w : m → T) : Matrix.toLinearMap₂' T M v w = dotProduct v (M *ᵥ w) := by
  /-
    n : Type u_11
    m : Type u_12
    inst✝⁴ : Fintype n
    inst✝³ : Fintype m
    inst✝² : DecidableEq n
    inst✝¹ : DecidableEq m
    T : Type u_16
    inst✝ : CommSemiring T
    M : Matrix n m T
    v : n → T
    w : m → T
    ⊢ Eq ((((Matrix.toLinearMap₂' T) M) v) w) (dotProduct v (M.mulVec w))
  -/
  simp_rw [Matrix.toLinearMap₂'_apply, dotProduct, Matrix.mulVec, dotProduct]
  /-
    n : Type u_11
    m : Type u_12
    inst✝⁴ : Fintype n
    inst✝³ : Fintype m
    inst✝² : DecidableEq n
    inst✝¹ : DecidableEq m
    T : Type u_16
    inst✝ : CommSemiring T
    M : Matrix n m T
    v : n → T
    w : m → T
    ⊢ Eq (Finset.univ.sum fun i => Finset.univ.sum fun j => HSMul.hSMul (v i) (HSM …
  -/
  refine Finset.sum_congr rfl fun _ _ => ?_
  /-
    n : Type u_11
    m : Type u_12
    inst✝⁴ : Fintype n
    inst✝³ : Fintype m
    inst✝² : DecidableEq n
    inst✝¹ : DecidableEq m
    T : Type u_16
    inst✝ : CommSemiring T
    M : Matrix n m T
    v : n → T
    w : m → T
    x✝¹ : n
    x✝ : Membership.mem Finset.univ x✝¹
    ⊢ Eq (Finset.univ.sum fun j => HSMul.hSMul (v x✝¹) (HSMul.hSMul (w j) (M x✝¹ j …
  -/
  rw [Finset.mul_sum]
  /-
    n : Type u_11
    m : Type u_12
    inst✝⁴ : Fintype n
    inst✝³ : Fintype m
    inst✝² : DecidableEq n
    inst✝¹ : DecidableEq m
    T : Type u_16
    inst✝ : CommSemiring T
    M : Matrix n m T
    v : n → T
    w : m → T
    x✝¹ : n
    x✝ : Membership.mem Finset.univ x✝¹
    ⊢ Eq (Finset.univ.sum fun j => HSMul.hSMul (v x✝¹) (HSMul.hSMul (w j) (M x✝¹ j …
  -/
  refine Finset.sum_congr rfl fun _ _ => ?_
  /-
    n : Type u_11
    m : Type u_12
    inst✝⁴ : Fintype n
    inst✝³ : Fintype m
    inst✝² : DecidableEq n
    inst✝¹ : DecidableEq m
    T : Type u_16
    inst✝ : CommSemiring T
    M : Matrix n m T
    v : n → T
    w : m → T
    x✝³ : n
    x✝² : Membership.mem Finset.univ x✝³
    x✝¹ : m
    x✝ : Membership.mem Finset.univ x✝¹
    ⊢ Eq (HSMul.hSMul (v x✝³) (HSMul.hSMul (w x✝¹) (M x✝³ x✝¹))) (HMul.hMul (v x✝³ …
  -/
  rw [smul_eq_mul, smul_eq_mul, mul_comm (w _), ← mul_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem Matrix.toLinearMapₛₗ₂'_single (M : Matrix n m N₂) (i : n) (j : m) :
    Matrix.toLinearMapₛₗ₂' R σ₁ σ₂ M (Pi.single i 1) (Pi.single j 1) = M i j :=
  Matrix.toLinearMap₂'Aux_single σ₁ σ₂ M i j


set_option linter.deprecated false in
@[simp, deprecated Matrix.toLinearMapₛₗ₂'_single (since := "2024-08-09")]
theorem Matrix.toLinearMapₛₗ₂'_stdBasis (M : Matrix n m N₂) (i : n) (j : m) :
    Matrix.toLinearMapₛₗ₂' R σ₁ σ₂ M (LinearMap.stdBasis R₁ (fun _ => R₁) i 1)
      (LinearMap.stdBasis R₂ (fun _ => R₂) j 1) = M i j :=
  Matrix.toLinearMapₛₗ₂'_single ..


@[simp]
theorem Matrix.toLinearMap₂'_single (M : Matrix n m N₂) (i : n) (j : m) :
    Matrix.toLinearMap₂' R M (Pi.single i 1) (Pi.single j 1) = M i j :=
  Matrix.toLinearMap₂'Aux_single _ _ M i j


set_option linter.deprecated false in
@[simp, deprecated Matrix.toLinearMap₂'_single (since := "2024-08-09")]
theorem Matrix.toLinearMap₂'_stdBasis (M : Matrix n m N₂) (i : n) (j : m) :
    Matrix.toLinearMap₂' R M (LinearMap.stdBasis R (fun _ => R) i 1)
      (LinearMap.stdBasis R (fun _ => R) j 1) = M i j :=
  show Matrix.toLinearMap₂' R M (Pi.single i 1) (Pi.single j 1) = M i j
  from Matrix.toLinearMap₂'Aux_single _ _ M i j


@[simp]
theorem LinearMap.toMatrixₛₗ₂'_symm :
    ((LinearMap.toMatrixₛₗ₂' R).symm : Matrix n m N₂ ≃ₗ[R] _) = Matrix.toLinearMapₛₗ₂' R σ₁ σ₂ :=
  rfl


@[simp]
theorem Matrix.toLinearMapₛₗ₂'_symm :
    ((Matrix.toLinearMapₛₗ₂' R σ₁ σ₂).symm : _ ≃ₗ[R] Matrix n m N₂) = LinearMap.toMatrixₛₗ₂' R :=
  (LinearMap.toMatrixₛₗ₂' R).symm_symm


@[simp]
theorem Matrix.toLinearMapₛₗ₂'_toMatrix' (B : (n → R₁) →ₛₗ[σ₁] (m → R₂) →ₛₗ[σ₂] N₂) :
    Matrix.toLinearMapₛₗ₂' R σ₁ σ₂ (LinearMap.toMatrixₛₗ₂' R B) = B :=
  (Matrix.toLinearMapₛₗ₂' R σ₁ σ₂).apply_symm_apply B


@[simp]
theorem Matrix.toLinearMap₂'_toMatrix' (B : (n → S₁) →ₗ[S₁] (m → S₂) →ₗ[S₂] N₂) :
    Matrix.toLinearMap₂' R (LinearMap.toMatrix₂' R B) = B :=
  (Matrix.toLinearMap₂' R).apply_symm_apply B


@[simp]
theorem LinearMap.toMatrix'_toLinearMapₛₗ₂' (M : Matrix n m N₂) :
    LinearMap.toMatrixₛₗ₂' R (Matrix.toLinearMapₛₗ₂' R σ₁ σ₂ M) = M :=
  (LinearMap.toMatrixₛₗ₂' R).apply_symm_apply M


@[simp]
theorem LinearMap.toMatrix'_toLinearMap₂' (M : Matrix n m N₂) :
    LinearMap.toMatrix₂' R (Matrix.toLinearMap₂' R (S₁ := S₁) (S₂ := S₂) M) = M :=
  (LinearMap.toMatrixₛₗ₂' R).apply_symm_apply M


@[simp]
theorem LinearMap.toMatrixₛₗ₂'_apply (B : (n → R₁) →ₛₗ[σ₁] (m → R₂) →ₛₗ[σ₂] N₂) (i : n) (j : m) :
    LinearMap.toMatrixₛₗ₂' R B i j = B (Pi.single i 1) (Pi.single j 1) :=
  rfl


@[simp]
theorem LinearMap.toMatrix₂'_apply (B : (n → S₁) →ₗ[S₁] (m → S₂) →ₗ[S₂] N₂) (i : n) (j : m) :
    LinearMap.toMatrix₂' R B i j = B (Pi.single i 1) (Pi.single j 1) :=
  rfl


@[simp]
theorem LinearMap.toMatrix₂'_compl₁₂ (B : (n → R) →ₗ[R] (m → R) →ₗ[R] R) (l : (n' → R) →ₗ[R] n → R)
    (r : (m' → R) →ₗ[R] m → R) :
    toMatrix₂' R (B.compl₁₂ l r) = (toMatrix' l)ᵀ * toMatrix₂' R B * toMatrix' r := by
  /-
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    m' : Type u_14
    R : Type u_16
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Fintype n
    inst✝⁶ : Fintype m
    inst✝⁵ : DecidableEq n
    inst✝⁴ : DecidableEq m
    inst✝³ : Fintype n'
    inst✝² : Fintype m'
    inst✝¹ : DecidableEq n'
    inst✝ : DecidableEq m'
    B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
    l : LinearMap (RingHom.id R) (n' → R) (n → R)
    r : LinearMap (RingHom.id R) (m' → R) (m → R)
    ⊢ Eq ((LinearMap.toMatrix₂' R) (B.compl₁₂ l r)) (HMul.hMul (HMul.hMul (LinearM …
  -/
  ext i j
  simp only [LinearMap.toMatrix₂'_apply, LinearMap.compl₁₂_apply, transpose_apply, Matrix.mul_apply,
    LinearMap.toMatrix', LinearEquiv.coe_mk, sum_mul]
  /-
    case a
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    m' : Type u_14
    R : Type u_16
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Fintype n
    inst✝⁶ : Fintype m
    inst✝⁵ : DecidableEq n
    inst✝⁴ : DecidableEq m
    inst✝³ : Fintype n'
    inst✝² : Fintype m'
    inst✝¹ : DecidableEq n'
    inst✝ : DecidableEq m'
    B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
    l : LinearMap (RingHom.id R) (n' → R) (n → R)
    r : LinearMap (RingHom.id R) (m' → R) (m → R)
    i : n'
    j : m'
    ⊢ Eq ((B (l (Pi.single i 1))) (r (Pi.single j 1))) (Finset.univ.sum fun x => F …
  -/
  rw [sum_comm]
  /-
    case a
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    m' : Type u_14
    R : Type u_16
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Fintype n
    inst✝⁶ : Fintype m
    inst✝⁵ : DecidableEq n
    inst✝⁴ : DecidableEq m
    inst✝³ : Fintype n'
    inst✝² : Fintype m'
    inst✝¹ : DecidableEq n'
    inst✝ : DecidableEq m'
    B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
    l : LinearMap (RingHom.id R) (n' → R) (n → R)
    r : LinearMap (RingHom.id R) (m' → R) (m → R)
    i : n'
    j : m'
    ⊢ Eq ((B (l (Pi.single i 1))) (r (Pi.single j 1))) (Finset.univ.sum fun y => F …
  -/
  conv_lhs => rw [← LinearMap.sum_repr_mul_repr_mul (Pi.basisFun R n) (Pi.basisFun R m) (l _) (r _)]
  /-
    case a
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    m' : Type u_14
    R : Type u_16
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Fintype n
    inst✝⁶ : Fintype m
    inst✝⁵ : DecidableEq n
    inst✝⁴ : DecidableEq m
    inst✝³ : Fintype n'
    inst✝² : Fintype m'
    inst✝¹ : DecidableEq n'
    inst✝ : DecidableEq m'
    B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
    l : LinearMap (RingHom.id R) (n' → R) (n → R)
    r : LinearMap (RingHom.id R) (m' → R) (m → R)
    i : n'
    j : m'
    ⊢ Eq (((Pi.basisFun R n).repr (l (Pi.single i 1))).sum fun i xi => ((Pi.basisF …
  -/
  rw [Finsupp.sum_fintype]
    /-
      case a
      n : Type u_11
      m : Type u_12
      n' : Type u_13
      m' : Type u_14
      R : Type u_16
      inst✝⁸ : CommSemiring R
      inst✝⁷ : Fintype n
      inst✝⁶ : Fintype m
      inst✝⁵ : DecidableEq n
      inst✝⁴ : DecidableEq m
      inst✝³ : Fintype n'
      inst✝² : Fintype m'
      inst✝¹ : DecidableEq n'
      inst✝ : DecidableEq m'
      B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
      l : LinearMap (RingHom.id R) (n' → R) (n → R)
      r : LinearMap (RingHom.id R) (m' → R) (m → R)
      i : n'
      j : m'
      ⊢ Eq (Finset.univ.sum fun i_1 => ((Pi.basisFun R m).repr (r (Pi.single j 1))). …
    -/
  · apply sum_congr rfl
    /-
      case a
      n : Type u_11
      m : Type u_12
      n' : Type u_13
      m' : Type u_14
      R : Type u_16
      inst✝⁸ : CommSemiring R
      inst✝⁷ : Fintype n
      inst✝⁶ : Fintype m
      inst✝⁵ : DecidableEq n
      inst✝⁴ : DecidableEq m
      inst✝³ : Fintype n'
      inst✝² : Fintype m'
      inst✝¹ : DecidableEq n'
      inst✝ : DecidableEq m'
      B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
      l : LinearMap (RingHom.id R) (n' → R) (n → R)
      r : LinearMap (RingHom.id R) (m' → R) (m → R)
      i : n'
      j : m'
      ⊢ ∀ (x : n), Membership.mem Finset.univ x → Eq (((Pi.basisFun R m).repr (r (Pi …
    -/
    rintro i' -
    /-
      case a
      n : Type u_11
      m : Type u_12
      n' : Type u_13
      m' : Type u_14
      R : Type u_16
      inst✝⁸ : CommSemiring R
      inst✝⁷ : Fintype n
      inst✝⁶ : Fintype m
      inst✝⁵ : DecidableEq n
      inst✝⁴ : DecidableEq m
      inst✝³ : Fintype n'
      inst✝² : Fintype m'
      inst✝¹ : DecidableEq n'
      inst✝ : DecidableEq m'
      B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
      l : LinearMap (RingHom.id R) (n' → R) (n → R)
      r : LinearMap (RingHom.id R) (m' → R) (m → R)
      i : n'
      j : m'
      i' : n
      ⊢ Eq (((Pi.basisFun R m).repr (r (Pi.single j 1))).sum fun j yj => HSMul.hSMul …
    -/
    rw [Finsupp.sum_fintype]
      /-
        case a
        n : Type u_11
        m : Type u_12
        n' : Type u_13
        m' : Type u_14
        R : Type u_16
        inst✝⁸ : CommSemiring R
        inst✝⁷ : Fintype n
        inst✝⁶ : Fintype m
        inst✝⁵ : DecidableEq n
        inst✝⁴ : DecidableEq m
        inst✝³ : Fintype n'
        inst✝² : Fintype m'
        inst✝¹ : DecidableEq n'
        inst✝ : DecidableEq m'
        B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
        l : LinearMap (RingHom.id R) (n' → R) (n → R)
        r : LinearMap (RingHom.id R) (m' → R) (m → R)
        i : n'
        j : m'
        i' : n
        ⊢ Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (((Pi.basisFun R n).repr (l (Pi.s …
      -/
    · apply sum_congr rfl
      /-
        case a
        n : Type u_11
        m : Type u_12
        n' : Type u_13
        m' : Type u_14
        R : Type u_16
        inst✝⁸ : CommSemiring R
        inst✝⁷ : Fintype n
        inst✝⁶ : Fintype m
        inst✝⁵ : DecidableEq n
        inst✝⁴ : DecidableEq m
        inst✝³ : Fintype n'
        inst✝² : Fintype m'
        inst✝¹ : DecidableEq n'
        inst✝ : DecidableEq m'
        B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
        l : LinearMap (RingHom.id R) (n' → R) (n → R)
        r : LinearMap (RingHom.id R) (m' → R) (m → R)
        i : n'
        j : m'
        i' : n
        ⊢ ∀ (x : m), Membership.mem Finset.univ x → Eq (HSMul.hSMul (((Pi.basisFun R n …
      -/
      rintro j' -
      simp only [smul_eq_mul, Pi.basisFun_repr, mul_assoc, mul_comm, mul_left_comm,
        Pi.basisFun_apply, of_apply]
      /-
        case a.h
        n : Type u_11
        m : Type u_12
        n' : Type u_13
        m' : Type u_14
        R : Type u_16
        inst✝⁸ : CommSemiring R
        inst✝⁷ : Fintype n
        inst✝⁶ : Fintype m
        inst✝⁵ : DecidableEq n
        inst✝⁴ : DecidableEq m
        inst✝³ : Fintype n'
        inst✝² : Fintype m'
        inst✝¹ : DecidableEq n'
        inst✝ : DecidableEq m'
        B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
        l : LinearMap (RingHom.id R) (n' → R) (n → R)
        r : LinearMap (RingHom.id R) (m' → R) (m → R)
        i : n'
        j : m'
        i' : n
        ⊢ ∀ (i_1 : m), Eq (HSMul.hSMul (((Pi.basisFun R n).repr (l (Pi.single i 1))) i …
      -/
    · intros
      /-
        case a.h
        n : Type u_11
        m : Type u_12
        n' : Type u_13
        m' : Type u_14
        R : Type u_16
        inst✝⁸ : CommSemiring R
        inst✝⁷ : Fintype n
        inst✝⁶ : Fintype m
        inst✝⁵ : DecidableEq n
        inst✝⁴ : DecidableEq m
        inst✝³ : Fintype n'
        inst✝² : Fintype m'
        inst✝¹ : DecidableEq n'
        inst✝ : DecidableEq m'
        B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
        l : LinearMap (RingHom.id R) (n' → R) (n → R)
        r : LinearMap (RingHom.id R) (m' → R) (m → R)
        i : n'
        j : m'
        i' : n
        i✝ : m
        ⊢ Eq (HSMul.hSMul (((Pi.basisFun R n).repr (l (Pi.single i 1))) i') (HSMul.hSM …
      -/
      simp only [zero_smul, smul_zero]
      /-
        🎉 no goals
      -/
    /-
      case a.h
      n : Type u_11
      m : Type u_12
      n' : Type u_13
      m' : Type u_14
      R : Type u_16
      inst✝⁸ : CommSemiring R
      inst✝⁷ : Fintype n
      inst✝⁶ : Fintype m
      inst✝⁵ : DecidableEq n
      inst✝⁴ : DecidableEq m
      inst✝³ : Fintype n'
      inst✝² : Fintype m'
      inst✝¹ : DecidableEq n'
      inst✝ : DecidableEq m'
      B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
      l : LinearMap (RingHom.id R) (n' → R) (n → R)
      r : LinearMap (RingHom.id R) (m' → R) (m → R)
      i : n'
      j : m'
      ⊢ ∀ (i : n), Eq (((Pi.basisFun R m).repr (r (Pi.single j 1))).sum fun j yj =>  …
    -/
  · intros
    /-
      case a.h
      n : Type u_11
      m : Type u_12
      n' : Type u_13
      m' : Type u_14
      R : Type u_16
      inst✝⁸ : CommSemiring R
      inst✝⁷ : Fintype n
      inst✝⁶ : Fintype m
      inst✝⁵ : DecidableEq n
      inst✝⁴ : DecidableEq m
      inst✝³ : Fintype n'
      inst✝² : Fintype m'
      inst✝¹ : DecidableEq n'
      inst✝ : DecidableEq m'
      B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
      l : LinearMap (RingHom.id R) (n' → R) (n → R)
      r : LinearMap (RingHom.id R) (m' → R) (m → R)
      i : n'
      j : m'
      i✝ : n
      ⊢ Eq (((Pi.basisFun R m).repr (r (Pi.single j 1))).sum fun j yj => HSMul.hSMul …
    -/
    simp only [zero_smul, Finsupp.sum_zero]
    /-
      🎉 no goals
    -/


theorem LinearMap.toMatrix₂'_comp (B : (n → R) →ₗ[R] (m → R) →ₗ[R] R) (f : (n' → R) →ₗ[R] n → R) :
    toMatrix₂' R (B.comp f) = (toMatrix' f)ᵀ * toMatrix₂' R B := by
  /-
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    R : Type u_16
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Fintype n
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype n'
    inst✝ : DecidableEq n'
    B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
    f : LinearMap (RingHom.id R) (n' → R) (n → R)
    ⊢ Eq ((LinearMap.toMatrix₂' R) (B.comp f)) (HMul.hMul (LinearMap.toMatrix' f). …
  -/
  rw [← LinearMap.compl₂_id (B.comp f), ← LinearMap.compl₁₂]
  /-
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    R : Type u_16
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Fintype n
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype n'
    inst✝ : DecidableEq n'
    B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
    f : LinearMap (RingHom.id R) (n' → R) (n → R)
    ⊢ Eq ((LinearMap.toMatrix₂' R) (B.compl₁₂ f LinearMap.id)) (HMul.hMul (LinearM …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem LinearMap.toMatrix₂'_compl₂ (B : (n → R) →ₗ[R] (m → R) →ₗ[R] R) (f : (m' → R) →ₗ[R] m → R) :
    toMatrix₂' R (B.compl₂ f) = toMatrix₂' R B * toMatrix' f := by
  /-
    n : Type u_11
    m : Type u_12
    m' : Type u_14
    R : Type u_16
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Fintype n
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m'
    inst✝ : DecidableEq m'
    B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
    f : LinearMap (RingHom.id R) (m' → R) (m → R)
    ⊢ Eq ((LinearMap.toMatrix₂' R) (B.compl₂ f)) (HMul.hMul ((LinearMap.toMatrix₂' …
  -/
  rw [← LinearMap.comp_id B, ← LinearMap.compl₁₂]
  /-
    n : Type u_11
    m : Type u_12
    m' : Type u_14
    R : Type u_16
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Fintype n
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m'
    inst✝ : DecidableEq m'
    B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
    f : LinearMap (RingHom.id R) (m' → R) (m → R)
    ⊢ Eq ((LinearMap.toMatrix₂' R) (B.compl₁₂ LinearMap.id f)) (HMul.hMul ((Linear …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem LinearMap.mul_toMatrix₂'_mul (B : (n → R) →ₗ[R] (m → R) →ₗ[R] R) (M : Matrix n' n R)
    (N : Matrix m m' R) :
    M * toMatrix₂' R B * N = toMatrix₂' R (B.compl₁₂ (toLin' Mᵀ) (toLin' N)) := by
  /-
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    m' : Type u_14
    R : Type u_16
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Fintype n
    inst✝⁶ : Fintype m
    inst✝⁵ : DecidableEq n
    inst✝⁴ : DecidableEq m
    inst✝³ : Fintype n'
    inst✝² : Fintype m'
    inst✝¹ : DecidableEq n'
    inst✝ : DecidableEq m'
    B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
    M : Matrix n' n R
    N : Matrix m m' R
    ⊢ Eq (HMul.hMul (HMul.hMul M ((LinearMap.toMatrix₂' R) B)) N) ((LinearMap.toMa …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem LinearMap.mul_toMatrix' (B : (n → R) →ₗ[R] (m → R) →ₗ[R] R) (M : Matrix n' n R) :
    M * toMatrix₂' R B = toMatrix₂' R (B.comp <| toLin' Mᵀ) := by
  /-
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    R : Type u_16
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Fintype n
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype n'
    inst✝ : DecidableEq n'
    B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
    M : Matrix n' n R
    ⊢ Eq (HMul.hMul M ((LinearMap.toMatrix₂' R) B)) ((LinearMap.toMatrix₂' R) (B.c …
  -/
  simp only [B.toMatrix₂'_comp, transpose_transpose, toMatrix'_toLin']
  /-
    🎉 no goals
  -/


theorem LinearMap.toMatrix₂'_mul (B : (n → R) →ₗ[R] (m → R) →ₗ[R] R) (M : Matrix m m' R) :
    toMatrix₂' R B * M = toMatrix₂' R (B.compl₂ <| toLin' M) := by
  /-
    n : Type u_11
    m : Type u_12
    m' : Type u_14
    R : Type u_16
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Fintype n
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m'
    inst✝ : DecidableEq m'
    B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) R)
    M : Matrix m m' R
    ⊢ Eq (HMul.hMul ((LinearMap.toMatrix₂' R) B) M) ((LinearMap.toMatrix₂' R) (B.c …
  -/
  simp only [B.toMatrix₂'_compl₂, toMatrix'_toLin']
  /-
    🎉 no goals
  -/


theorem Matrix.toLinearMap₂'_comp (M : Matrix n m R) (P : Matrix n n' R) (Q : Matrix m m' R) :
    LinearMap.compl₁₂ (Matrix.toLinearMap₂' R M) (toLin' P) (toLin' Q) =
      toLinearMap₂' R (Pᵀ * M * Q) :=
                                         /-
                                           n : Type u_11
                                           m : Type u_12
                                           n' : Type u_13
                                           m' : Type u_14
                                           R : Type u_16
                                           inst✝⁸ : CommSemiring R
                                           inst✝⁷ : Fintype n
                                           inst✝⁶ : Fintype m
                                           inst✝⁵ : DecidableEq n
                                           inst✝⁴ : DecidableEq m
                                           inst✝³ : Fintype n'
                                           inst✝² : Fintype m'
                                           inst✝¹ : DecidableEq n'
                                           inst✝ : DecidableEq m'
                                           M : Matrix n m R
                                           P : Matrix n n' R
                                           Q : Matrix m m' R
                                           ⊢ Eq ((LinearMap.toMatrix₂' R) (((Matrix.toLinearMap₂' R) M).compl₁₂ (Matrix.t …
                                         -/
  (LinearMap.toMatrix₂' R).injective (by simp)
                                         /-
                                           🎉 no goals
                                         -/


/-- `LinearMap.toMatrix₂ b₁ b₂` is the equivalence between `R`-bilinear maps on `M` and
`n`-by-`m` matrices with entries in `R`, if `b₁` and `b₂` are `R`-bases for `M₁` and `M₂`,
respectively. -/
noncomputable def LinearMap.toMatrix₂ : (M₁ →ₗ[R] M₂ →ₗ[R] N₂) ≃ₗ[R] Matrix n m N₂ :=
  (b₁.equivFun.arrowCongr (b₂.equivFun.arrowCongr (LinearEquiv.refl R N₂))).trans
    (LinearMap.toMatrix₂' R)


/-- `Matrix.toLinearMap₂ b₁ b₂` is the equivalence between `R`-bilinear maps on `M` and
`n`-by-`m` matrices with entries in `R`, if `b₁` and `b₂` are `R`-bases for `M₁` and `M₂`,
respectively; this is the reverse direction of `LinearMap.toMatrix₂ b₁ b₂`. -/
noncomputable def Matrix.toLinearMap₂ : Matrix n m N₂ ≃ₗ[R] M₁ →ₗ[R] M₂ →ₗ[R] N₂ :=
  (LinearMap.toMatrix₂ b₁ b₂).symm

-- We make this and not `LinearMap.toMatrix₂` a `simp` lemma to avoid timeouts

@[simp]
theorem LinearMap.toMatrix₂_apply (B : M₁ →ₗ[R] M₂ →ₗ[R] N₂) (i : n) (j : m) :
    LinearMap.toMatrix₂ b₁ b₂ B i j = B (b₁ i) (b₂ j) := by
  simp only [toMatrix₂, LinearEquiv.trans_apply, toMatrix₂'_apply, LinearEquiv.arrowCongr_apply,
    Basis.equivFun_symm_apply, Pi.single_apply, ite_smul, one_smul, zero_smul, sum_ite_eq',
    mem_univ, ↓reduceIte, LinearEquiv.refl_apply]


@[simp]
theorem Matrix.toLinearMap₂_apply (M : Matrix n m N₂) (x : M₁) (y : M₂) :
    Matrix.toLinearMap₂ b₁ b₂ M x y = ∑ i, ∑ j, b₁.repr x i • b₂.repr y j • M i j :=
  Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ =>
    smul_algebra_smul_comm ((RingHom.id R) ((Basis.equivFun b₁) x _))
    ((RingHom.id R) ((Basis.equivFun b₂) y _)) (M _ _)

-- Not a `simp` lemma since `LinearMap.toMatrix₂` needs an extra argument

theorem LinearMap.toMatrix₂Aux_eq (B : M₁ →ₗ[R] M₂ →ₗ[R] N₂) :
    LinearMap.toMatrix₂Aux R b₁ b₂ B = LinearMap.toMatrix₂ b₁ b₂ B :=
                           /-
                             R : Type u_1
                             M₁ : Type u_6
                             M₂ : Type u_7
                             N₂ : Type u_10
                             n : Type u_11
                             m : Type u_12
                             inst✝¹⁰ : CommSemiring R
                             inst✝⁹ : AddCommMonoid M₁
                             inst✝⁸ : Module R M₁
                             inst✝⁷ : AddCommMonoid M₂
                             inst✝⁶ : Module R M₂
                             inst✝⁵ : AddCommMonoid N₂
                             inst✝⁴ : Module R N₂
                             inst✝³ : DecidableEq n
                             inst✝² : Fintype n
                             inst✝¹ : DecidableEq m
                             inst✝ : Fintype m
                             b₁ : Basis n R M₁
                             b₂ : Basis m R M₂
                             B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ N₂)
                             i : n
                             j : m
                             ⊢ Eq ((LinearMap.toMatrix₂Aux R ⇑b₁ ⇑b₂) B i j) ((LinearMap.toMatrix₂ b₁ b₂) B …
                           -/
  Matrix.ext fun i j => by rw [LinearMap.toMatrix₂_apply, LinearMap.toMatrix₂Aux_apply]
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem LinearMap.toMatrix₂_symm :
    (LinearMap.toMatrix₂ b₁ b₂).symm = Matrix.toLinearMap₂ (N₂ := N₂) b₁ b₂ :=
  rfl


@[simp]
theorem Matrix.toLinearMap₂_symm :
    (Matrix.toLinearMap₂ b₁ b₂).symm = LinearMap.toMatrix₂ (N₂ := N₂) b₁ b₂ :=
  (LinearMap.toMatrix₂ b₁ b₂).symm_symm


theorem Matrix.toLinearMap₂_basisFun :
    Matrix.toLinearMap₂ (Pi.basisFun R n) (Pi.basisFun R m) =
      Matrix.toLinearMap₂' R (N₂ := N₂) := by
  /-
    R : Type u_1
    N₂ : Type u_10
    n : Type u_11
    m : Type u_12
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid N₂
    inst✝⁴ : Module R N₂
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : Fintype m
    ⊢ Eq (Matrix.toLinearMap₂ (Pi.basisFun R n) (Pi.basisFun R m)) (Matrix.toLinea …
  -/
  ext M
  simp only [coe_comp, coe_single, Function.comp_apply, toLinearMap₂_apply, Pi.basisFun_repr,
    toLinearMap₂'_apply]


theorem LinearMap.toMatrix₂_basisFun :
    LinearMap.toMatrix₂ (Pi.basisFun R n) (Pi.basisFun R m) =
    LinearMap.toMatrix₂' R (N₂ := N₂) := by
  /-
    R : Type u_1
    N₂ : Type u_10
    n : Type u_11
    m : Type u_12
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid N₂
    inst✝⁴ : Module R N₂
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : Fintype m
    ⊢ Eq (LinearMap.toMatrix₂ (Pi.basisFun R n) (Pi.basisFun R m)) (LinearMap.toMa …
  -/
  ext B
  /-
    case h.a
    R : Type u_1
    N₂ : Type u_10
    n : Type u_11
    m : Type u_12
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid N₂
    inst✝⁴ : Module R N₂
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : Fintype m
    B : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (m → R) N₂)
    i✝ : n
    j✝ : m
    ⊢ Eq ((LinearMap.toMatrix₂ (Pi.basisFun R n) (Pi.basisFun R m)) B i✝ j✝) ((Lin …
  -/
  rw [LinearMap.toMatrix₂_apply, LinearMap.toMatrix₂'_apply, Pi.basisFun_apply, Pi.basisFun_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem Matrix.toLinearMap₂_toMatrix₂ (B : M₁ →ₗ[R] M₂ →ₗ[R] N₂) :
    Matrix.toLinearMap₂ b₁ b₂ (LinearMap.toMatrix₂ b₁ b₂ B) = B :=
  (Matrix.toLinearMap₂ b₁ b₂).apply_symm_apply B


@[simp]
theorem LinearMap.toMatrix₂_toLinearMap₂ (M : Matrix n m N₂) :
    LinearMap.toMatrix₂ b₁ b₂ (Matrix.toLinearMap₂ b₁ b₂ M) = M :=
  (LinearMap.toMatrix₂ b₁ b₂).apply_symm_apply M


theorem LinearMap.toMatrix₂_compl₁₂ (B : M₁ →ₗ[R] M₂ →ₗ[R] R) (l : M₁' →ₗ[R] M₁)
    (r : M₂' →ₗ[R] M₂) :
    LinearMap.toMatrix₂ b₁' b₂' (B.compl₁₂ l r) =
      (toMatrix b₁' b₁ l)ᵀ * LinearMap.toMatrix₂ b₁ b₂ B * toMatrix b₂' b₂ r := by
  /-
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    M₁' : Type u_8
    M₂' : Type u_9
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    m' : Type u_14
    inst✝¹⁶ : CommSemiring R
    inst✝¹⁵ : AddCommMonoid M₁
    inst✝¹⁴ : Module R M₁
    inst✝¹³ : AddCommMonoid M₂
    inst✝¹² : Module R M₂
    inst✝¹¹ : DecidableEq n
    inst✝¹⁰ : Fintype n
    inst✝⁹ : DecidableEq m
    inst✝⁸ : Fintype m
    b₁ : Basis n R M₁
    b₂ : Basis m R M₂
    inst✝⁷ : AddCommMonoid M₁'
    inst✝⁶ : Module R M₁'
    inst✝⁵ : AddCommMonoid M₂'
    inst✝⁴ : Module R M₂'
    b₁' : Basis n' R M₁'
    b₂' : Basis m' R M₂'
    inst✝³ : Fintype n'
    inst✝² : Fintype m'
    inst✝¹ : DecidableEq n'
    inst✝ : DecidableEq m'
    B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
    l : LinearMap (RingHom.id R) M₁' M₁
    r : LinearMap (RingHom.id R) M₂' M₂
    ⊢ Eq ((LinearMap.toMatrix₂ b₁' b₂') (B.compl₁₂ l r)) (HMul.hMul (HMul.hMul ((L …
  -/
  ext i j
  simp only [LinearMap.toMatrix₂_apply, compl₁₂_apply, transpose_apply, Matrix.mul_apply,
    LinearMap.toMatrix_apply, LinearEquiv.coe_mk, sum_mul]
  /-
    case a
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    M₁' : Type u_8
    M₂' : Type u_9
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    m' : Type u_14
    inst✝¹⁶ : CommSemiring R
    inst✝¹⁵ : AddCommMonoid M₁
    inst✝¹⁴ : Module R M₁
    inst✝¹³ : AddCommMonoid M₂
    inst✝¹² : Module R M₂
    inst✝¹¹ : DecidableEq n
    inst✝¹⁰ : Fintype n
    inst✝⁹ : DecidableEq m
    inst✝⁸ : Fintype m
    b₁ : Basis n R M₁
    b₂ : Basis m R M₂
    inst✝⁷ : AddCommMonoid M₁'
    inst✝⁶ : Module R M₁'
    inst✝⁵ : AddCommMonoid M₂'
    inst✝⁴ : Module R M₂'
    b₁' : Basis n' R M₁'
    b₂' : Basis m' R M₂'
    inst✝³ : Fintype n'
    inst✝² : Fintype m'
    inst✝¹ : DecidableEq n'
    inst✝ : DecidableEq m'
    B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
    l : LinearMap (RingHom.id R) M₁' M₁
    r : LinearMap (RingHom.id R) M₂' M₂
    i : n'
    j : m'
    ⊢ Eq ((B (l (b₁' i))) (r (b₂' j))) (Finset.univ.sum fun x => Finset.univ.sum f …
  -/
  rw [sum_comm]
  /-
    case a
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    M₁' : Type u_8
    M₂' : Type u_9
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    m' : Type u_14
    inst✝¹⁶ : CommSemiring R
    inst✝¹⁵ : AddCommMonoid M₁
    inst✝¹⁴ : Module R M₁
    inst✝¹³ : AddCommMonoid M₂
    inst✝¹² : Module R M₂
    inst✝¹¹ : DecidableEq n
    inst✝¹⁰ : Fintype n
    inst✝⁹ : DecidableEq m
    inst✝⁸ : Fintype m
    b₁ : Basis n R M₁
    b₂ : Basis m R M₂
    inst✝⁷ : AddCommMonoid M₁'
    inst✝⁶ : Module R M₁'
    inst✝⁵ : AddCommMonoid M₂'
    inst✝⁴ : Module R M₂'
    b₁' : Basis n' R M₁'
    b₂' : Basis m' R M₂'
    inst✝³ : Fintype n'
    inst✝² : Fintype m'
    inst✝¹ : DecidableEq n'
    inst✝ : DecidableEq m'
    B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
    l : LinearMap (RingHom.id R) M₁' M₁
    r : LinearMap (RingHom.id R) M₂' M₂
    i : n'
    j : m'
    ⊢ Eq ((B (l (b₁' i))) (r (b₂' j))) (Finset.univ.sum fun y => Finset.univ.sum f …
  -/
  conv_lhs => rw [← LinearMap.sum_repr_mul_repr_mul b₁ b₂]
  /-
    case a
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    M₁' : Type u_8
    M₂' : Type u_9
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    m' : Type u_14
    inst✝¹⁶ : CommSemiring R
    inst✝¹⁵ : AddCommMonoid M₁
    inst✝¹⁴ : Module R M₁
    inst✝¹³ : AddCommMonoid M₂
    inst✝¹² : Module R M₂
    inst✝¹¹ : DecidableEq n
    inst✝¹⁰ : Fintype n
    inst✝⁹ : DecidableEq m
    inst✝⁸ : Fintype m
    b₁ : Basis n R M₁
    b₂ : Basis m R M₂
    inst✝⁷ : AddCommMonoid M₁'
    inst✝⁶ : Module R M₁'
    inst✝⁵ : AddCommMonoid M₂'
    inst✝⁴ : Module R M₂'
    b₁' : Basis n' R M₁'
    b₂' : Basis m' R M₂'
    inst✝³ : Fintype n'
    inst✝² : Fintype m'
    inst✝¹ : DecidableEq n'
    inst✝ : DecidableEq m'
    B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
    l : LinearMap (RingHom.id R) M₁' M₁
    r : LinearMap (RingHom.id R) M₂' M₂
    i : n'
    j : m'
    ⊢ Eq ((b₁.repr (l (b₁' i))).sum fun i xi => (b₂.repr (r (b₂' j))).sum fun j yj …
  -/
  rw [Finsupp.sum_fintype]
    /-
      case a
      R : Type u_1
      M₁ : Type u_6
      M₂ : Type u_7
      M₁' : Type u_8
      M₂' : Type u_9
      n : Type u_11
      m : Type u_12
      n' : Type u_13
      m' : Type u_14
      inst✝¹⁶ : CommSemiring R
      inst✝¹⁵ : AddCommMonoid M₁
      inst✝¹⁴ : Module R M₁
      inst✝¹³ : AddCommMonoid M₂
      inst✝¹² : Module R M₂
      inst✝¹¹ : DecidableEq n
      inst✝¹⁰ : Fintype n
      inst✝⁹ : DecidableEq m
      inst✝⁸ : Fintype m
      b₁ : Basis n R M₁
      b₂ : Basis m R M₂
      inst✝⁷ : AddCommMonoid M₁'
      inst✝⁶ : Module R M₁'
      inst✝⁵ : AddCommMonoid M₂'
      inst✝⁴ : Module R M₂'
      b₁' : Basis n' R M₁'
      b₂' : Basis m' R M₂'
      inst✝³ : Fintype n'
      inst✝² : Fintype m'
      inst✝¹ : DecidableEq n'
      inst✝ : DecidableEq m'
      B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
      l : LinearMap (RingHom.id R) M₁' M₁
      r : LinearMap (RingHom.id R) M₂' M₂
      i : n'
      j : m'
      ⊢ Eq (Finset.univ.sum fun i_1 => (b₂.repr (r (b₂' j))).sum fun j yj => HSMul.h …
    -/
  · apply sum_congr rfl
    /-
      case a
      R : Type u_1
      M₁ : Type u_6
      M₂ : Type u_7
      M₁' : Type u_8
      M₂' : Type u_9
      n : Type u_11
      m : Type u_12
      n' : Type u_13
      m' : Type u_14
      inst✝¹⁶ : CommSemiring R
      inst✝¹⁵ : AddCommMonoid M₁
      inst✝¹⁴ : Module R M₁
      inst✝¹³ : AddCommMonoid M₂
      inst✝¹² : Module R M₂
      inst✝¹¹ : DecidableEq n
      inst✝¹⁰ : Fintype n
      inst✝⁹ : DecidableEq m
      inst✝⁸ : Fintype m
      b₁ : Basis n R M₁
      b₂ : Basis m R M₂
      inst✝⁷ : AddCommMonoid M₁'
      inst✝⁶ : Module R M₁'
      inst✝⁵ : AddCommMonoid M₂'
      inst✝⁴ : Module R M₂'
      b₁' : Basis n' R M₁'
      b₂' : Basis m' R M₂'
      inst✝³ : Fintype n'
      inst✝² : Fintype m'
      inst✝¹ : DecidableEq n'
      inst✝ : DecidableEq m'
      B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
      l : LinearMap (RingHom.id R) M₁' M₁
      r : LinearMap (RingHom.id R) M₂' M₂
      i : n'
      j : m'
      ⊢ ∀ (x : n), Membership.mem Finset.univ x → Eq ((b₂.repr (r (b₂' j))).sum fun  …
    -/
    rintro i' -
    /-
      case a
      R : Type u_1
      M₁ : Type u_6
      M₂ : Type u_7
      M₁' : Type u_8
      M₂' : Type u_9
      n : Type u_11
      m : Type u_12
      n' : Type u_13
      m' : Type u_14
      inst✝¹⁶ : CommSemiring R
      inst✝¹⁵ : AddCommMonoid M₁
      inst✝¹⁴ : Module R M₁
      inst✝¹³ : AddCommMonoid M₂
      inst✝¹² : Module R M₂
      inst✝¹¹ : DecidableEq n
      inst✝¹⁰ : Fintype n
      inst✝⁹ : DecidableEq m
      inst✝⁸ : Fintype m
      b₁ : Basis n R M₁
      b₂ : Basis m R M₂
      inst✝⁷ : AddCommMonoid M₁'
      inst✝⁶ : Module R M₁'
      inst✝⁵ : AddCommMonoid M₂'
      inst✝⁴ : Module R M₂'
      b₁' : Basis n' R M₁'
      b₂' : Basis m' R M₂'
      inst✝³ : Fintype n'
      inst✝² : Fintype m'
      inst✝¹ : DecidableEq n'
      inst✝ : DecidableEq m'
      B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
      l : LinearMap (RingHom.id R) M₁' M₁
      r : LinearMap (RingHom.id R) M₂' M₂
      i : n'
      j : m'
      i' : n
      ⊢ Eq ((b₂.repr (r (b₂' j))).sum fun j yj => HSMul.hSMul ((b₁.repr (l (b₁' i))) …
    -/
    rw [Finsupp.sum_fintype]
      /-
        case a
        R : Type u_1
        M₁ : Type u_6
        M₂ : Type u_7
        M₁' : Type u_8
        M₂' : Type u_9
        n : Type u_11
        m : Type u_12
        n' : Type u_13
        m' : Type u_14
        inst✝¹⁶ : CommSemiring R
        inst✝¹⁵ : AddCommMonoid M₁
        inst✝¹⁴ : Module R M₁
        inst✝¹³ : AddCommMonoid M₂
        inst✝¹² : Module R M₂
        inst✝¹¹ : DecidableEq n
        inst✝¹⁰ : Fintype n
        inst✝⁹ : DecidableEq m
        inst✝⁸ : Fintype m
        b₁ : Basis n R M₁
        b₂ : Basis m R M₂
        inst✝⁷ : AddCommMonoid M₁'
        inst✝⁶ : Module R M₁'
        inst✝⁵ : AddCommMonoid M₂'
        inst✝⁴ : Module R M₂'
        b₁' : Basis n' R M₁'
        b₂' : Basis m' R M₂'
        inst✝³ : Fintype n'
        inst✝² : Fintype m'
        inst✝¹ : DecidableEq n'
        inst✝ : DecidableEq m'
        B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
        l : LinearMap (RingHom.id R) M₁' M₁
        r : LinearMap (RingHom.id R) M₂' M₂
        i : n'
        j : m'
        i' : n
        ⊢ Eq (Finset.univ.sum fun i_1 => HSMul.hSMul ((b₁.repr (l (b₁' i))) i') (HSMul …
      -/
    · apply sum_congr rfl
      /-
        case a
        R : Type u_1
        M₁ : Type u_6
        M₂ : Type u_7
        M₁' : Type u_8
        M₂' : Type u_9
        n : Type u_11
        m : Type u_12
        n' : Type u_13
        m' : Type u_14
        inst✝¹⁶ : CommSemiring R
        inst✝¹⁵ : AddCommMonoid M₁
        inst✝¹⁴ : Module R M₁
        inst✝¹³ : AddCommMonoid M₂
        inst✝¹² : Module R M₂
        inst✝¹¹ : DecidableEq n
        inst✝¹⁰ : Fintype n
        inst✝⁹ : DecidableEq m
        inst✝⁸ : Fintype m
        b₁ : Basis n R M₁
        b₂ : Basis m R M₂
        inst✝⁷ : AddCommMonoid M₁'
        inst✝⁶ : Module R M₁'
        inst✝⁵ : AddCommMonoid M₂'
        inst✝⁴ : Module R M₂'
        b₁' : Basis n' R M₁'
        b₂' : Basis m' R M₂'
        inst✝³ : Fintype n'
        inst✝² : Fintype m'
        inst✝¹ : DecidableEq n'
        inst✝ : DecidableEq m'
        B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
        l : LinearMap (RingHom.id R) M₁' M₁
        r : LinearMap (RingHom.id R) M₂' M₂
        i : n'
        j : m'
        i' : n
        ⊢ ∀ (x : m), Membership.mem Finset.univ x → Eq (HSMul.hSMul ((b₁.repr (l (b₁'  …
      -/
      rintro j' -
      simp only [smul_eq_mul, LinearMap.toMatrix_apply, Basis.equivFun_apply, mul_assoc, mul_comm,
        mul_left_comm]
      /-
        case a.h
        R : Type u_1
        M₁ : Type u_6
        M₂ : Type u_7
        M₁' : Type u_8
        M₂' : Type u_9
        n : Type u_11
        m : Type u_12
        n' : Type u_13
        m' : Type u_14
        inst✝¹⁶ : CommSemiring R
        inst✝¹⁵ : AddCommMonoid M₁
        inst✝¹⁴ : Module R M₁
        inst✝¹³ : AddCommMonoid M₂
        inst✝¹² : Module R M₂
        inst✝¹¹ : DecidableEq n
        inst✝¹⁰ : Fintype n
        inst✝⁹ : DecidableEq m
        inst✝⁸ : Fintype m
        b₁ : Basis n R M₁
        b₂ : Basis m R M₂
        inst✝⁷ : AddCommMonoid M₁'
        inst✝⁶ : Module R M₁'
        inst✝⁵ : AddCommMonoid M₂'
        inst✝⁴ : Module R M₂'
        b₁' : Basis n' R M₁'
        b₂' : Basis m' R M₂'
        inst✝³ : Fintype n'
        inst✝² : Fintype m'
        inst✝¹ : DecidableEq n'
        inst✝ : DecidableEq m'
        B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
        l : LinearMap (RingHom.id R) M₁' M₁
        r : LinearMap (RingHom.id R) M₂' M₂
        i : n'
        j : m'
        i' : n
        ⊢ ∀ (i_1 : m), Eq (HSMul.hSMul ((b₁.repr (l (b₁' i))) i') (HSMul.hSMul 0 ((B ( …
      -/
    · intros
      /-
        case a.h
        R : Type u_1
        M₁ : Type u_6
        M₂ : Type u_7
        M₁' : Type u_8
        M₂' : Type u_9
        n : Type u_11
        m : Type u_12
        n' : Type u_13
        m' : Type u_14
        inst✝¹⁶ : CommSemiring R
        inst✝¹⁵ : AddCommMonoid M₁
        inst✝¹⁴ : Module R M₁
        inst✝¹³ : AddCommMonoid M₂
        inst✝¹² : Module R M₂
        inst✝¹¹ : DecidableEq n
        inst✝¹⁰ : Fintype n
        inst✝⁹ : DecidableEq m
        inst✝⁸ : Fintype m
        b₁ : Basis n R M₁
        b₂ : Basis m R M₂
        inst✝⁷ : AddCommMonoid M₁'
        inst✝⁶ : Module R M₁'
        inst✝⁵ : AddCommMonoid M₂'
        inst✝⁴ : Module R M₂'
        b₁' : Basis n' R M₁'
        b₂' : Basis m' R M₂'
        inst✝³ : Fintype n'
        inst✝² : Fintype m'
        inst✝¹ : DecidableEq n'
        inst✝ : DecidableEq m'
        B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
        l : LinearMap (RingHom.id R) M₁' M₁
        r : LinearMap (RingHom.id R) M₂' M₂
        i : n'
        j : m'
        i' : n
        i✝ : m
        ⊢ Eq (HSMul.hSMul ((b₁.repr (l (b₁' i))) i') (HSMul.hSMul 0 ((B (b₁ i')) (b₂ i …
      -/
      simp only [zero_smul, smul_zero]
      /-
        🎉 no goals
      -/
    /-
      case a.h
      R : Type u_1
      M₁ : Type u_6
      M₂ : Type u_7
      M₁' : Type u_8
      M₂' : Type u_9
      n : Type u_11
      m : Type u_12
      n' : Type u_13
      m' : Type u_14
      inst✝¹⁶ : CommSemiring R
      inst✝¹⁵ : AddCommMonoid M₁
      inst✝¹⁴ : Module R M₁
      inst✝¹³ : AddCommMonoid M₂
      inst✝¹² : Module R M₂
      inst✝¹¹ : DecidableEq n
      inst✝¹⁰ : Fintype n
      inst✝⁹ : DecidableEq m
      inst✝⁸ : Fintype m
      b₁ : Basis n R M₁
      b₂ : Basis m R M₂
      inst✝⁷ : AddCommMonoid M₁'
      inst✝⁶ : Module R M₁'
      inst✝⁵ : AddCommMonoid M₂'
      inst✝⁴ : Module R M₂'
      b₁' : Basis n' R M₁'
      b₂' : Basis m' R M₂'
      inst✝³ : Fintype n'
      inst✝² : Fintype m'
      inst✝¹ : DecidableEq n'
      inst✝ : DecidableEq m'
      B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
      l : LinearMap (RingHom.id R) M₁' M₁
      r : LinearMap (RingHom.id R) M₂' M₂
      i : n'
      j : m'
      ⊢ ∀ (i : n), Eq ((b₂.repr (r (b₂' j))).sum fun j yj => HSMul.hSMul 0 (HSMul.hS …
    -/
  · intros
    /-
      case a.h
      R : Type u_1
      M₁ : Type u_6
      M₂ : Type u_7
      M₁' : Type u_8
      M₂' : Type u_9
      n : Type u_11
      m : Type u_12
      n' : Type u_13
      m' : Type u_14
      inst✝¹⁶ : CommSemiring R
      inst✝¹⁵ : AddCommMonoid M₁
      inst✝¹⁴ : Module R M₁
      inst✝¹³ : AddCommMonoid M₂
      inst✝¹² : Module R M₂
      inst✝¹¹ : DecidableEq n
      inst✝¹⁰ : Fintype n
      inst✝⁹ : DecidableEq m
      inst✝⁸ : Fintype m
      b₁ : Basis n R M₁
      b₂ : Basis m R M₂
      inst✝⁷ : AddCommMonoid M₁'
      inst✝⁶ : Module R M₁'
      inst✝⁵ : AddCommMonoid M₂'
      inst✝⁴ : Module R M₂'
      b₁' : Basis n' R M₁'
      b₂' : Basis m' R M₂'
      inst✝³ : Fintype n'
      inst✝² : Fintype m'
      inst✝¹ : DecidableEq n'
      inst✝ : DecidableEq m'
      B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
      l : LinearMap (RingHom.id R) M₁' M₁
      r : LinearMap (RingHom.id R) M₂' M₂
      i : n'
      j : m'
      i✝ : n
      ⊢ Eq ((b₂.repr (r (b₂' j))).sum fun j yj => HSMul.hSMul 0 (HSMul.hSMul yj ((B  …
    -/
    simp only [zero_smul, Finsupp.sum_zero]
    /-
      🎉 no goals
    -/


theorem LinearMap.toMatrix₂_comp (B : M₁ →ₗ[R] M₂ →ₗ[R] R) (f : M₁' →ₗ[R] M₁) :
    LinearMap.toMatrix₂ b₁' b₂ (B.comp f) =
      (toMatrix b₁' b₁ f)ᵀ * LinearMap.toMatrix₂ b₁ b₂ B := by
  /-
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    M₁' : Type u_8
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    inst✝¹² : CommSemiring R
    inst✝¹¹ : AddCommMonoid M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommMonoid M₂
    inst✝⁸ : Module R M₂
    inst✝⁷ : DecidableEq n
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq m
    inst✝⁴ : Fintype m
    b₁ : Basis n R M₁
    b₂ : Basis m R M₂
    inst✝³ : AddCommMonoid M₁'
    inst✝² : Module R M₁'
    b₁' : Basis n' R M₁'
    inst✝¹ : Fintype n'
    inst✝ : DecidableEq n'
    B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
    f : LinearMap (RingHom.id R) M₁' M₁
    ⊢ Eq ((LinearMap.toMatrix₂ b₁' b₂) (B.comp f)) (HMul.hMul ((LinearMap.toMatrix …
  -/
  rw [← LinearMap.compl₂_id (B.comp f), ← LinearMap.compl₁₂, LinearMap.toMatrix₂_compl₁₂ b₁ b₂]
  /-
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    M₁' : Type u_8
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    inst✝¹² : CommSemiring R
    inst✝¹¹ : AddCommMonoid M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommMonoid M₂
    inst✝⁸ : Module R M₂
    inst✝⁷ : DecidableEq n
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq m
    inst✝⁴ : Fintype m
    b₁ : Basis n R M₁
    b₂ : Basis m R M₂
    inst✝³ : AddCommMonoid M₁'
    inst✝² : Module R M₁'
    b₁' : Basis n' R M₁'
    inst✝¹ : Fintype n'
    inst✝ : DecidableEq n'
    B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
    f : LinearMap (RingHom.id R) M₁' M₁
    ⊢ Eq (HMul.hMul (HMul.hMul ((LinearMap.toMatrix b₁' b₁) f).transpose ((LinearM …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem LinearMap.toMatrix₂_compl₂ (B : M₁ →ₗ[R] M₂ →ₗ[R] R) (f : M₂' →ₗ[R] M₂) :
    LinearMap.toMatrix₂ b₁ b₂' (B.compl₂ f) =
      LinearMap.toMatrix₂ b₁ b₂ B * toMatrix b₂' b₂ f := by
  /-
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    M₂' : Type u_9
    n : Type u_11
    m : Type u_12
    m' : Type u_14
    inst✝¹² : CommSemiring R
    inst✝¹¹ : AddCommMonoid M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommMonoid M₂
    inst✝⁸ : Module R M₂
    inst✝⁷ : DecidableEq n
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq m
    inst✝⁴ : Fintype m
    b₁ : Basis n R M₁
    b₂ : Basis m R M₂
    inst✝³ : AddCommMonoid M₂'
    inst✝² : Module R M₂'
    b₂' : Basis m' R M₂'
    inst✝¹ : Fintype m'
    inst✝ : DecidableEq m'
    B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
    f : LinearMap (RingHom.id R) M₂' M₂
    ⊢ Eq ((LinearMap.toMatrix₂ b₁ b₂') (B.compl₂ f)) (HMul.hMul ((LinearMap.toMatr …
  -/
  rw [← LinearMap.comp_id B, ← LinearMap.compl₁₂, LinearMap.toMatrix₂_compl₁₂ b₁ b₂]
  /-
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    M₂' : Type u_9
    n : Type u_11
    m : Type u_12
    m' : Type u_14
    inst✝¹² : CommSemiring R
    inst✝¹¹ : AddCommMonoid M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommMonoid M₂
    inst✝⁸ : Module R M₂
    inst✝⁷ : DecidableEq n
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq m
    inst✝⁴ : Fintype m
    b₁ : Basis n R M₁
    b₂ : Basis m R M₂
    inst✝³ : AddCommMonoid M₂'
    inst✝² : Module R M₂'
    b₂' : Basis m' R M₂'
    inst✝¹ : Fintype m'
    inst✝ : DecidableEq m'
    B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
    f : LinearMap (RingHom.id R) M₂' M₂
    ⊢ Eq (HMul.hMul (HMul.hMul ((LinearMap.toMatrix b₁ b₁) LinearMap.id).transpose …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem LinearMap.toMatrix₂_mul_basis_toMatrix (c₁ : Basis n' R M₁) (c₂ : Basis m' R M₂)
    (B : M₁ →ₗ[R] M₂ →ₗ[R] R) :
    (b₁.toMatrix c₁)ᵀ * LinearMap.toMatrix₂ b₁ b₂ B * b₂.toMatrix c₂ =
      LinearMap.toMatrix₂ c₁ c₂ B := by
  /-
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    m' : Type u_14
    inst✝¹² : CommSemiring R
    inst✝¹¹ : AddCommMonoid M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommMonoid M₂
    inst✝⁸ : Module R M₂
    inst✝⁷ : DecidableEq n
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq m
    inst✝⁴ : Fintype m
    b₁ : Basis n R M₁
    b₂ : Basis m R M₂
    inst✝³ : Fintype n'
    inst✝² : Fintype m'
    inst✝¹ : DecidableEq n'
    inst✝ : DecidableEq m'
    c₁ : Basis n' R M₁
    c₂ : Basis m' R M₂
    B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
    ⊢ Eq (HMul.hMul (HMul.hMul (b₁.toMatrix ⇑c₁).transpose ((LinearMap.toMatrix₂ b …
  -/
  simp_rw [← LinearMap.toMatrix_id_eq_basis_toMatrix]
  /-
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    m' : Type u_14
    inst✝¹² : CommSemiring R
    inst✝¹¹ : AddCommMonoid M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommMonoid M₂
    inst✝⁸ : Module R M₂
    inst✝⁷ : DecidableEq n
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq m
    inst✝⁴ : Fintype m
    b₁ : Basis n R M₁
    b₂ : Basis m R M₂
    inst✝³ : Fintype n'
    inst✝² : Fintype m'
    inst✝¹ : DecidableEq n'
    inst✝ : DecidableEq m'
    c₁ : Basis n' R M₁
    c₂ : Basis m' R M₂
    B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
    ⊢ Eq (HMul.hMul (HMul.hMul ((LinearMap.toMatrix c₁ b₁) LinearMap.id).transpose …
  -/
  rw [← LinearMap.toMatrix₂_compl₁₂, LinearMap.compl₁₂_id_id]
  /-
    🎉 no goals
  -/


theorem LinearMap.mul_toMatrix₂_mul (B : M₁ →ₗ[R] M₂ →ₗ[R] R) (M : Matrix n' n R)
    (N : Matrix m m' R) :
    M * LinearMap.toMatrix₂ b₁ b₂ B * N =
      LinearMap.toMatrix₂ b₁' b₂' (B.compl₁₂ (toLin b₁' b₁ Mᵀ) (toLin b₂' b₂ N)) := by
  /-
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    M₁' : Type u_8
    M₂' : Type u_9
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    m' : Type u_14
    inst✝¹⁶ : CommSemiring R
    inst✝¹⁵ : AddCommMonoid M₁
    inst✝¹⁴ : Module R M₁
    inst✝¹³ : AddCommMonoid M₂
    inst✝¹² : Module R M₂
    inst✝¹¹ : DecidableEq n
    inst✝¹⁰ : Fintype n
    inst✝⁹ : DecidableEq m
    inst✝⁸ : Fintype m
    b₁ : Basis n R M₁
    b₂ : Basis m R M₂
    inst✝⁷ : AddCommMonoid M₁'
    inst✝⁶ : Module R M₁'
    inst✝⁵ : AddCommMonoid M₂'
    inst✝⁴ : Module R M₂'
    b₁' : Basis n' R M₁'
    b₂' : Basis m' R M₂'
    inst✝³ : Fintype n'
    inst✝² : Fintype m'
    inst✝¹ : DecidableEq n'
    inst✝ : DecidableEq m'
    B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
    M : Matrix n' n R
    N : Matrix m m' R
    ⊢ Eq (HMul.hMul (HMul.hMul M ((LinearMap.toMatrix₂ b₁ b₂) B)) N) ((LinearMap.t …
  -/
  simp_rw [LinearMap.toMatrix₂_compl₁₂ b₁ b₂, toMatrix_toLin, transpose_transpose]
  /-
    🎉 no goals
  -/


theorem LinearMap.mul_toMatrix₂ (B : M₁ →ₗ[R] M₂ →ₗ[R] R) (M : Matrix n' n R) :
    M * LinearMap.toMatrix₂ b₁ b₂ B =
      LinearMap.toMatrix₂ b₁' b₂ (B.comp (toLin b₁' b₁ Mᵀ)) := by
  /-
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    M₁' : Type u_8
    n : Type u_11
    m : Type u_12
    n' : Type u_13
    inst✝¹² : CommSemiring R
    inst✝¹¹ : AddCommMonoid M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommMonoid M₂
    inst✝⁸ : Module R M₂
    inst✝⁷ : DecidableEq n
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq m
    inst✝⁴ : Fintype m
    b₁ : Basis n R M₁
    b₂ : Basis m R M₂
    inst✝³ : AddCommMonoid M₁'
    inst✝² : Module R M₁'
    b₁' : Basis n' R M₁'
    inst✝¹ : Fintype n'
    inst✝ : DecidableEq n'
    B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
    M : Matrix n' n R
    ⊢ Eq (HMul.hMul M ((LinearMap.toMatrix₂ b₁ b₂) B)) ((LinearMap.toMatrix₂ b₁' b …
  -/
  rw [LinearMap.toMatrix₂_comp b₁, toMatrix_toLin, transpose_transpose]
  /-
    🎉 no goals
  -/


theorem LinearMap.toMatrix₂_mul (B : M₁ →ₗ[R] M₂ →ₗ[R] R) (M : Matrix m m' R) :
    LinearMap.toMatrix₂ b₁ b₂ B * M =
      LinearMap.toMatrix₂ b₁ b₂' (B.compl₂ (toLin b₂' b₂ M)) := by
  /-
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    M₂' : Type u_9
    n : Type u_11
    m : Type u_12
    m' : Type u_14
    inst✝¹² : CommSemiring R
    inst✝¹¹ : AddCommMonoid M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommMonoid M₂
    inst✝⁸ : Module R M₂
    inst✝⁷ : DecidableEq n
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq m
    inst✝⁴ : Fintype m
    b₁ : Basis n R M₁
    b₂ : Basis m R M₂
    inst✝³ : AddCommMonoid M₂'
    inst✝² : Module R M₂'
    b₂' : Basis m' R M₂'
    inst✝¹ : Fintype m'
    inst✝ : DecidableEq m'
    B : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)
    M : Matrix m m' R
    ⊢ Eq (HMul.hMul ((LinearMap.toMatrix₂ b₁ b₂) B) M) ((LinearMap.toMatrix₂ b₁ b₂ …
  -/
  rw [LinearMap.toMatrix₂_compl₂ b₁ b₂, toMatrix_toLin]
  /-
    🎉 no goals
  -/


theorem Matrix.toLinearMap₂_compl₁₂ (M : Matrix n m R) (P : Matrix n n' R) (Q : Matrix m m' R) :
    (Matrix.toLinearMap₂ b₁ b₂ M).compl₁₂ (toLin b₁' b₁ P) (toLin b₂' b₂ Q) =
      Matrix.toLinearMap₂ b₁' b₂' (Pᵀ * M * Q) :=
  (LinearMap.toMatrix₂ b₁' b₂').injective
    (by
      simp only [LinearMap.toMatrix₂_compl₁₂ b₁ b₂, LinearMap.toMatrix₂_toLinearMap₂,
        toMatrix_toLin])


/-- The condition for the matrices `A`, `A'` to be an adjoint pair with respect to the square
matrices `J`, `J₃`. -/
def Matrix.IsAdjointPair :=
  Aᵀ * J' = J * A'


/-- The condition for a square matrix `A` to be self-adjoint with respect to the square matrix
`J`. -/
def Matrix.IsSelfAdjoint :=
  Matrix.IsAdjointPair J J A₁ A₁


/-- The condition for a square matrix `A` to be skew-adjoint with respect to the square matrix
`J`. -/
def Matrix.IsSkewAdjoint :=
  Matrix.IsAdjointPair J J A₁ (-A₁)


@[simp]
theorem isAdjointPair_toLinearMap₂' :
    LinearMap.IsAdjointPair (Matrix.toLinearMap₂' R J) (Matrix.toLinearMap₂' R J')
        (Matrix.toLin' A) (Matrix.toLin' A') ↔
      Matrix.IsAdjointPair J J' A A' := by
  /-
    R : Type u_1
    n : Type u_11
    n' : Type u_13
    inst✝⁴ : CommRing R
    inst✝³ : Fintype n
    inst✝² : Fintype n'
    J : Matrix n n R
    J' : Matrix n' n' R
    A : Matrix n' n R
    A' : Matrix n n' R
    inst✝¹ : DecidableEq n
    inst✝ : DecidableEq n'
    ⊢ Iff (((Matrix.toLinearMap₂' R) J).IsAdjointPair ((Matrix.toLinearMap₂' R) J' …
  -/
  rw [isAdjointPair_iff_comp_eq_compl₂]
  have h :
    ∀ B B' : (n → R) →ₗ[R] (n' → R) →ₗ[R] R,
      B = B' ↔ LinearMap.toMatrix₂' R B = LinearMap.toMatrix₂' R B' := by
    intro B B'
    constructor <;> intro h
    · rw [h]
    · exact (LinearMap.toMatrix₂' R).injective h
  simp_rw [h, LinearMap.toMatrix₂'_comp, LinearMap.toMatrix₂'_compl₂,
    LinearMap.toMatrix'_toLin', LinearMap.toMatrix'_toLinearMap₂']
  /-
    R : Type u_1
    n : Type u_11
    n' : Type u_13
    inst✝⁴ : CommRing R
    inst✝³ : Fintype n
    inst✝² : Fintype n'
    J : Matrix n n R
    J' : Matrix n' n' R
    A : Matrix n' n R
    A' : Matrix n n' R
    inst✝¹ : DecidableEq n
    inst✝ : DecidableEq n'
    h : ∀ (B B' : LinearMap (RingHom.id R) (n → R) (LinearMap (RingHom.id R) (n' → …
    ⊢ Iff (Eq (HMul.hMul A.transpose J') (HMul.hMul J A')) (J.IsAdjointPair J' A A')
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem isAdjointPair_toLinearMap₂ :
    LinearMap.IsAdjointPair (Matrix.toLinearMap₂ b₁ b₁ J)
      (Matrix.toLinearMap₂ b₂ b₂ J') (Matrix.toLin b₁ b₂ A) (Matrix.toLin b₂ b₁ A') ↔
      Matrix.IsAdjointPair J J' A A' := by
  /-
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    n : Type u_11
    n' : Type u_13
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : Module R M₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M₂
    inst✝³ : Fintype n
    inst✝² : Fintype n'
    b₁ : Basis n R M₁
    b₂ : Basis n' R M₂
    J : Matrix n n R
    J' : Matrix n' n' R
    A : Matrix n' n R
    A' : Matrix n n' R
    inst✝¹ : DecidableEq n
    inst✝ : DecidableEq n'
    ⊢ Iff (((Matrix.toLinearMap₂ b₁ b₁) J).IsAdjointPair ((Matrix.toLinearMap₂ b₂  …
  -/
  rw [isAdjointPair_iff_comp_eq_compl₂]
  have h :
    ∀ B B' : M₁ →ₗ[R] M₂ →ₗ[R] R,
      B = B' ↔ LinearMap.toMatrix₂ b₁ b₂ B = LinearMap.toMatrix₂ b₁ b₂ B' := by
    intro B B'
    constructor <;> intro h
    · rw [h]
    · exact (LinearMap.toMatrix₂ b₁ b₂).injective h
  simp_rw [h, LinearMap.toMatrix₂_comp b₂ b₂, LinearMap.toMatrix₂_compl₂ b₁ b₁,
    LinearMap.toMatrix_toLin, LinearMap.toMatrix₂_toLinearMap₂]
  /-
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    n : Type u_11
    n' : Type u_13
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : Module R M₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M₂
    inst✝³ : Fintype n
    inst✝² : Fintype n'
    b₁ : Basis n R M₁
    b₂ : Basis n' R M₂
    J : Matrix n n R
    J' : Matrix n' n' R
    A : Matrix n' n R
    A' : Matrix n n' R
    inst✝¹ : DecidableEq n
    inst✝ : DecidableEq n'
    h : ∀ (B B' : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ R)), If …
    ⊢ Iff (Eq (HMul.hMul A.transpose J') (HMul.hMul J A')) (J.IsAdjointPair J' A A')
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Matrix.isAdjointPair_equiv (P : Matrix n n R) (h : IsUnit P) :
    (Pᵀ * J * P).IsAdjointPair (Pᵀ * J * P) A₁ A₂ ↔
      J.IsAdjointPair J (P * A₁ * P⁻¹) (P * A₂ * P⁻¹) := by
  /-
    R : Type u_1
    n : Type u_11
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    J A₁ A₂ : Matrix n n R
    inst✝ : DecidableEq n
    P : Matrix n n R
    h : IsUnit P
    ⊢ Iff ((HMul.hMul (HMul.hMul P.transpose J) P).IsAdjointPair (HMul.hMul (HMul. …
  -/
  have h' : IsUnit P.det := P.isUnit_iff_isUnit_det.mp h
  /-
    R : Type u_1
    n : Type u_11
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    J A₁ A₂ : Matrix n n R
    inst✝ : DecidableEq n
    P : Matrix n n R
    h : IsUnit P
    h' : IsUnit P.det
    ⊢ Iff ((HMul.hMul (HMul.hMul P.transpose J) P).IsAdjointPair (HMul.hMul (HMul. …
  -/
  let u := P.nonsingInvUnit h'
  /-
    R : Type u_1
    n : Type u_11
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    J A₁ A₂ : Matrix n n R
    inst✝ : DecidableEq n
    P : Matrix n n R
    h : IsUnit P
    h' : IsUnit P.det
    u : Units (Matrix n n R) := P.nonsingInvUnit h'
    ⊢ Iff ((HMul.hMul (HMul.hMul P.transpose J) P).IsAdjointPair (HMul.hMul (HMul. …
  -/
  let v := Pᵀ.nonsingInvUnit (P.isUnit_det_transpose h')
  /-
    R : Type u_1
    n : Type u_11
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    J A₁ A₂ : Matrix n n R
    inst✝ : DecidableEq n
    P : Matrix n n R
    h : IsUnit P
    h' : IsUnit P.det
    u : Units (Matrix n n R) := P.nonsingInvUnit h'
    v : Units (Matrix n n R) := P.transpose.nonsingInvUnit ⋯
    ⊢ Iff ((HMul.hMul (HMul.hMul P.transpose J) P).IsAdjointPair (HMul.hMul (HMul. …
  -/
  let x := A₁ᵀ * Pᵀ * J
  /-
    R : Type u_1
    n : Type u_11
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    J A₁ A₂ : Matrix n n R
    inst✝ : DecidableEq n
    P : Matrix n n R
    h : IsUnit P
    h' : IsUnit P.det
    u : Units (Matrix n n R) := P.nonsingInvUnit h'
    v : Units (Matrix n n R) := P.transpose.nonsingInvUnit ⋯
    x : Matrix n n R := HMul.hMul (HMul.hMul A₁.transpose P.transpose) J
    ⊢ Iff ((HMul.hMul (HMul.hMul P.transpose J) P).IsAdjointPair (HMul.hMul (HMul. …
  -/
  let y := J * P * A₂
  suffices x * u = v * y ↔ v⁻¹ * x = y * u⁻¹ by
    dsimp only [Matrix.IsAdjointPair]
    simp only [Matrix.transpose_mul]
    simp only [← mul_assoc, P.transpose_nonsing_inv]
    convert this using 2
    · rw [mul_assoc, mul_assoc, ← mul_assoc J]
      rfl
    · rw [mul_assoc, mul_assoc, ← mul_assoc _ _ J]
      rfl
  /-
    R : Type u_1
    n : Type u_11
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    J A₁ A₂ : Matrix n n R
    inst✝ : DecidableEq n
    P : Matrix n n R
    h : IsUnit P
    h' : IsUnit P.det
    u : Units (Matrix n n R) := P.nonsingInvUnit h'
    v : Units (Matrix n n R) := P.transpose.nonsingInvUnit ⋯
    x : Matrix n n R := HMul.hMul (HMul.hMul A₁.transpose P.transpose) J
    y : Matrix n n R := HMul.hMul (HMul.hMul J P) A₂
    ⊢ Iff (Eq (HMul.hMul x ↑u) (HMul.hMul (↑v) y)) (Eq (HMul.hMul (↑(Inv.inv v)) x …
  -/
  rw [Units.eq_mul_inv_iff_mul_eq]
  /-
    R : Type u_1
    n : Type u_11
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    J A₁ A₂ : Matrix n n R
    inst✝ : DecidableEq n
    P : Matrix n n R
    h : IsUnit P
    h' : IsUnit P.det
    u : Units (Matrix n n R) := P.nonsingInvUnit h'
    v : Units (Matrix n n R) := P.transpose.nonsingInvUnit ⋯
    x : Matrix n n R := HMul.hMul (HMul.hMul A₁.transpose P.transpose) J
    y : Matrix n n R := HMul.hMul (HMul.hMul J P) A₂
    ⊢ Iff (Eq (HMul.hMul x ↑u) (HMul.hMul (↑v) y)) (Eq (HMul.hMul (HMul.hMul (↑(In …
  -/
  conv_rhs => rw [mul_assoc]
  /-
    R : Type u_1
    n : Type u_11
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    J A₁ A₂ : Matrix n n R
    inst✝ : DecidableEq n
    P : Matrix n n R
    h : IsUnit P
    h' : IsUnit P.det
    u : Units (Matrix n n R) := P.nonsingInvUnit h'
    v : Units (Matrix n n R) := P.transpose.nonsingInvUnit ⋯
    x : Matrix n n R := HMul.hMul (HMul.hMul A₁.transpose P.transpose) J
    y : Matrix n n R := HMul.hMul (HMul.hMul J P) A₂
    ⊢ Iff (Eq (HMul.hMul x ↑u) (HMul.hMul (↑v) y)) (Eq (HMul.hMul (↑(Inv.inv v)) ( …
  -/
  rw [v.inv_mul_eq_iff_eq_mul]
  /-
    🎉 no goals
  -/


/-- The submodule of pair-self-adjoint matrices with respect to bilinear forms corresponding to
given matrices `J`, `J₂`. -/
def pairSelfAdjointMatricesSubmodule : Submodule R (Matrix n n R) :=
  (isPairSelfAdjointSubmodule (Matrix.toLinearMap₂' R J)
    (Matrix.toLinearMap₂' R J₂)).map
    ((LinearMap.toMatrix' : ((n → R) →ₗ[R] n → R) ≃ₗ[R] Matrix n n R) :
      ((n → R) →ₗ[R] n → R) →ₗ[R] Matrix n n R)


@[simp]
theorem mem_pairSelfAdjointMatricesSubmodule :
    A₁ ∈ pairSelfAdjointMatricesSubmodule J J₂ ↔ Matrix.IsAdjointPair J J₂ A₁ A₁ := by
  simp only [pairSelfAdjointMatricesSubmodule, LinearEquiv.coe_coe, LinearMap.toMatrix'_apply,
    Submodule.mem_map, mem_isPairSelfAdjointSubmodule]
  /-
    R : Type u_1
    n : Type u_11
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    J J₂ A₁ : Matrix n n R
    inst✝ : DecidableEq n
    ⊢ Iff (Exists fun y => And (((Matrix.toLinearMap₂' R) J).IsPairSelfAdjoint ((M …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      n : Type u_11
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      J J₂ A₁ : Matrix n n R
      inst✝ : DecidableEq n
      ⊢ (Exists fun y => And (((Matrix.toLinearMap₂' R) J).IsPairSelfAdjoint ((Matri …
    -/
  · rintro ⟨f, hf, hA⟩
    /-
      case mp.intro.intro
      R : Type u_1
      n : Type u_11
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      J J₂ A₁ : Matrix n n R
      inst✝ : DecidableEq n
      f : Module.End R (n → R)
      hf : ((Matrix.toLinearMap₂' R) J).IsPairSelfAdjoint ((Matrix.toLinearMap₂' R)  …
      hA : Eq (LinearMap.toMatrix' f) A₁
      ⊢ J.IsAdjointPair J₂ A₁ A₁
    -/
    have hf' : f = toLin' A₁ := by rw [← hA, Matrix.toLin'_toMatrix']
    /-
      case mp.intro.intro
      R : Type u_1
      n : Type u_11
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      J J₂ A₁ : Matrix n n R
      inst✝ : DecidableEq n
      f : Module.End R (n → R)
      hf : ((Matrix.toLinearMap₂' R) J).IsPairSelfAdjoint ((Matrix.toLinearMap₂' R)  …
      hA : Eq (LinearMap.toMatrix' f) A₁
      hf' : Eq f (Matrix.toLin' A₁)
      ⊢ J.IsAdjointPair J₂ A₁ A₁
    -/
    rw [hf'] at hf
    /-
      case mp.intro.intro
      R : Type u_1
      n : Type u_11
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      J J₂ A₁ : Matrix n n R
      inst✝ : DecidableEq n
      f : Module.End R (n → R)
      hf : ((Matrix.toLinearMap₂' R) J).IsPairSelfAdjoint ((Matrix.toLinearMap₂' R)  …
      hA : Eq (LinearMap.toMatrix' f) A₁
      hf' : Eq f (Matrix.toLin' A₁)
      ⊢ J.IsAdjointPair J₂ A₁ A₁
    -/
    rw [← isAdjointPair_toLinearMap₂']
    /-
      case mp.intro.intro
      R : Type u_1
      n : Type u_11
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      J J₂ A₁ : Matrix n n R
      inst✝ : DecidableEq n
      f : Module.End R (n → R)
      hf : ((Matrix.toLinearMap₂' R) J).IsPairSelfAdjoint ((Matrix.toLinearMap₂' R)  …
      hA : Eq (LinearMap.toMatrix' f) A₁
      hf' : Eq f (Matrix.toLin' A₁)
      ⊢ ((Matrix.toLinearMap₂' R) J).IsAdjointPair ((Matrix.toLinearMap₂' R) J₂) ⇑(M …
    -/
    exact hf
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      n : Type u_11
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      J J₂ A₁ : Matrix n n R
      inst✝ : DecidableEq n
      ⊢ J.IsAdjointPair J₂ A₁ A₁ → Exists fun y => And (((Matrix.toLinearMap₂' R) J) …
    -/
  · intro h
    /-
      case mpr
      R : Type u_1
      n : Type u_11
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      J J₂ A₁ : Matrix n n R
      inst✝ : DecidableEq n
      h : J.IsAdjointPair J₂ A₁ A₁
      ⊢ Exists fun y => And (((Matrix.toLinearMap₂' R) J).IsPairSelfAdjoint ((Matrix …
    -/
    refine ⟨toLin' A₁, ?_, LinearMap.toMatrix'_toLin' _⟩
    /-
      case mpr
      R : Type u_1
      n : Type u_11
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      J J₂ A₁ : Matrix n n R
      inst✝ : DecidableEq n
      h : J.IsAdjointPair J₂ A₁ A₁
      ⊢ ((Matrix.toLinearMap₂' R) J).IsPairSelfAdjoint ((Matrix.toLinearMap₂' R) J₂) …
    -/
    exact (isAdjointPair_toLinearMap₂' _ _ _ _).mpr h
    /-
      🎉 no goals
    -/


/-- The submodule of self-adjoint matrices with respect to the bilinear form corresponding to
the matrix `J`. -/
def selfAdjointMatricesSubmodule : Submodule R (Matrix n n R) :=
  pairSelfAdjointMatricesSubmodule J J


@[simp]
theorem mem_selfAdjointMatricesSubmodule :
    A₁ ∈ selfAdjointMatricesSubmodule J ↔ J.IsSelfAdjoint A₁ := by
  /-
    R : Type u_1
    n : Type u_11
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    J A₁ : Matrix n n R
    inst✝ : DecidableEq n
    ⊢ Iff (Membership.mem (selfAdjointMatricesSubmodule J) A₁) (J.IsSelfAdjoint A₁)
  -/
  erw [mem_pairSelfAdjointMatricesSubmodule]
  /-
    R : Type u_1
    n : Type u_11
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    J A₁ : Matrix n n R
    inst✝ : DecidableEq n
    ⊢ Iff (J.IsAdjointPair J A₁ A₁) (J.IsSelfAdjoint A₁)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The submodule of skew-adjoint matrices with respect to the bilinear form corresponding to
the matrix `J`. -/
def skewAdjointMatricesSubmodule : Submodule R (Matrix n n R) :=
  pairSelfAdjointMatricesSubmodule (-J) J


@[simp]
theorem mem_skewAdjointMatricesSubmodule :
    A₁ ∈ skewAdjointMatricesSubmodule J ↔ J.IsSkewAdjoint A₁ := by
  /-
    R : Type u_1
    n : Type u_11
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    J A₁ : Matrix n n R
    inst✝ : DecidableEq n
    ⊢ Iff (Membership.mem (skewAdjointMatricesSubmodule J) A₁) (J.IsSkewAdjoint A₁)
  -/
  erw [mem_pairSelfAdjointMatricesSubmodule]
  /-
    R : Type u_1
    n : Type u_11
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    J A₁ : Matrix n n R
    inst✝ : DecidableEq n
    ⊢ Iff ((Neg.neg J).IsAdjointPair J A₁ A₁) (J.IsSkewAdjoint A₁)
  -/
  simp [Matrix.IsSkewAdjoint, Matrix.IsAdjointPair]
  /-
    🎉 no goals
  -/


theorem _root_.Matrix.separatingLeft_toLinearMap₂'_iff_separatingLeft_toLinearMap₂
    {M : Matrix ι ι R₁} (b : Basis ι R₁ M₁) :
    (Matrix.toLinearMap₂' R₁ M).SeparatingLeft (R := R₁) ↔
      (Matrix.toLinearMap₂ b b M).SeparatingLeft :=
  (separatingLeft_congr_iff b.equivFun.symm b.equivFun.symm).symm

-- Lemmas transferring nondegeneracy between a matrix and its associated bilinear form

theorem _root_.Matrix.Nondegenerate.toLinearMap₂' {M : Matrix ι ι R₁} (h : M.Nondegenerate) :
    (Matrix.toLinearMap₂' R₁ M).SeparatingLeft (R := R₁) := fun x hx =>
                                 /-
                                   R₁ : Type u_2
                                   ι : Type u_15
                                   inst✝² : CommRing R₁
                                   inst✝¹ : DecidableEq ι
                                   inst✝ : Fintype ι
                                   M : Matrix ι ι R₁
                                   h : M.Nondegenerate
                                   x : ι → R₁
                                   hx : ∀ (y : ι → R₁), Eq ((((Matrix.toLinearMap₂' R₁) M) x) y) 0
                                   y : ι → R₁
                                   ⊢ Eq (dotProduct x (M.mulVec y)) 0
                                 -/
  h.eq_zero_of_ortho fun y => by simpa only [toLinearMap₂'_apply'] using hx y
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem _root_.Matrix.separatingLeft_toLinearMap₂'_iff {M : Matrix ι ι R₁} :
    (Matrix.toLinearMap₂' R₁ M).SeparatingLeft (R := R₁) ↔ M.Nondegenerate :=
  ⟨fun h v hv => h v fun w => (M.toLinearMap₂'_apply' _ _).trans <| hv w,
    Matrix.Nondegenerate.toLinearMap₂'⟩


theorem _root_.Matrix.Nondegenerate.toLinearMap₂ {M : Matrix ι ι R₁} (h : M.Nondegenerate)
    (b : Basis ι R₁ M₁) : (toLinearMap₂ b b M).SeparatingLeft :=
  (Matrix.separatingLeft_toLinearMap₂'_iff_separatingLeft_toLinearMap₂ b).mp h.toLinearMap₂'


@[simp]
theorem _root_.Matrix.separatingLeft_toLinearMap₂_iff {M : Matrix ι ι R₁} (b : Basis ι R₁ M₁) :
    (toLinearMap₂ b b M).SeparatingLeft ↔ M.Nondegenerate := by
  rw [← Matrix.separatingLeft_toLinearMap₂'_iff_separatingLeft_toLinearMap₂,
    Matrix.separatingLeft_toLinearMap₂'_iff]

-- Lemmas transferring nondegeneracy between a bilinear form and its associated matrix

@[simp]
theorem nondegenerate_toMatrix₂'_iff {B : (ι → R₁) →ₗ[R₁] (ι → R₁) →ₗ[R₁] R₁} :
    (LinearMap.toMatrix₂' R₁ B).Nondegenerate ↔ B.SeparatingLeft :=
  Matrix.separatingLeft_toLinearMap₂'_iff.symm.trans <|
    (Matrix.toLinearMap₂'_toMatrix' (R := R₁) B).symm ▸ Iff.rfl


theorem SeparatingLeft.toMatrix₂' {B : (ι → R₁) →ₗ[R₁] (ι → R₁) →ₗ[R₁] R₁} (h : B.SeparatingLeft) :
    (LinearMap.toMatrix₂' R₁ B).Nondegenerate :=
  nondegenerate_toMatrix₂'_iff.mpr h


@[simp]
theorem nondegenerate_toMatrix_iff {B : M₁ →ₗ[R₁] M₁ →ₗ[R₁] R₁} (b : Basis ι R₁ M₁) :
    (toMatrix₂ b b B).Nondegenerate ↔ B.SeparatingLeft :=
  (Matrix.separatingLeft_toLinearMap₂_iff b).symm.trans <|
    (Matrix.toLinearMap₂_toMatrix₂ b b B).symm ▸ Iff.rfl


theorem SeparatingLeft.toMatrix₂ {B : M₁ →ₗ[R₁] M₁ →ₗ[R₁] R₁} (h : B.SeparatingLeft)
    (b : Basis ι R₁ M₁) : (toMatrix₂ b b B).Nondegenerate :=
  (nondegenerate_toMatrix_iff b).mpr h

-- Some shorthands for combining the above with `Matrix.nondegenerate_of_det_ne_zero`

theorem separatingLeft_toLinearMap₂'_iff_det_ne_zero {M : Matrix ι ι R₁} :
    (Matrix.toLinearMap₂' R₁ M).SeparatingLeft (R := R₁) ↔ M.det ≠ 0 := by
  /-
    R₁ : Type u_2
    ι : Type u_15
    inst✝³ : CommRing R₁
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    inst✝ : IsDomain R₁
    M : Matrix ι ι R₁
    ⊢ Iff ((Matrix.toLinearMap₂' R₁) M).SeparatingLeft (Ne M.det 0)
  -/
  rw [Matrix.separatingLeft_toLinearMap₂'_iff, Matrix.nondegenerate_iff_det_ne_zero]
  /-
    🎉 no goals
  -/


theorem separatingLeft_toLinearMap₂'_of_det_ne_zero' (M : Matrix ι ι R₁) (h : M.det ≠ 0) :
    (Matrix.toLinearMap₂' R₁ M).SeparatingLeft (R := R₁) :=
  separatingLeft_toLinearMap₂'_iff_det_ne_zero.mpr h


theorem separatingLeft_iff_det_ne_zero {B : M₁ →ₗ[R₁] M₁ →ₗ[R₁] R₁} (b : Basis ι R₁ M₁) :
    B.SeparatingLeft ↔ (toMatrix₂ b b B).det ≠ 0 := by
  /-
    R₁ : Type u_2
    M₁ : Type u_6
    ι : Type u_15
    inst✝⁵ : CommRing R₁
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : Module R₁ M₁
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    inst✝ : IsDomain R₁
    B : LinearMap (RingHom.id R₁) M₁ (LinearMap (RingHom.id R₁) M₁ R₁)
    b : Basis ι R₁ M₁
    ⊢ Iff B.SeparatingLeft (Ne ((LinearMap.toMatrix₂ b b) B).det 0)
  -/
  rw [← Matrix.nondegenerate_iff_det_ne_zero, nondegenerate_toMatrix_iff]
  /-
    🎉 no goals
  -/


theorem separatingLeft_of_det_ne_zero {B : M₁ →ₗ[R₁] M₁ →ₗ[R₁] R₁} (b : Basis ι R₁ M₁)
    (h : (toMatrix₂ b b B).det ≠ 0) : B.SeparatingLeft :=
  (separatingLeft_iff_det_ne_zero b).mpr h


