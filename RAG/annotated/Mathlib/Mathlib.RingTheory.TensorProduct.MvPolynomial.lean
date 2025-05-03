/-- The tensor product of a polynomial ring by a module is
  linearly equivalent to a Finsupp of a tensor product -/
noncomputable def rTensor :
    MvPolynomial σ S ⊗[R] N ≃ₗ[S] (σ →₀ ℕ) →₀ (S ⊗[R] N) :=
  TensorProduct.finsuppLeft' _ _ _ _ _


lemma rTensor_apply_tmul (p : MvPolynomial σ S) (n : N) :
    rTensor (p ⊗ₜ[R] n) = p.sum (fun i m ↦ Finsupp.single i (m ⊗ₜ[R] n)) :=
  TensorProduct.finsuppLeft_apply_tmul p n


lemma rTensor_apply_tmul_apply (p : MvPolynomial σ S) (n : N) (d : σ →₀ ℕ) :
    rTensor (p ⊗ₜ[R] n) d = (coeff d p) ⊗ₜ[R] n :=
  TensorProduct.finsuppLeft_apply_tmul_apply p n d


lemma rTensor_apply_monomial_tmul (e : σ →₀ ℕ) (s : S) (n : N) (d : σ →₀ ℕ) :
    rTensor (monomial e s ⊗ₜ[R] n) d = if e = d then s ⊗ₜ[R] n else 0 := by
  /-
    R : Type u
    N : Type v
    inst✝⁵ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra R S
    inst✝² : DecidableEq σ
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    e : Finsupp σ Nat
    s : S
    n : N
    d : Finsupp σ Nat
    ⊢ Eq ((MvPolynomial.rTensor (TensorProduct.tmul R ((MvPolynomial.monomial e) s …
  -/
  simp only [rTensor_apply_tmul_apply, coeff_monomial, ite_tmul]
  /-
    🎉 no goals
  -/


lemma rTensor_apply_X_tmul (s : σ) (n : N) (d : σ →₀ ℕ) :
    rTensor (X s ⊗ₜ[R] n) d = if Finsupp.single s 1 = d then (1 : S) ⊗ₜ[R] n else 0 := by
  /-
    R : Type u
    N : Type v
    inst✝⁵ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra R S
    inst✝² : DecidableEq σ
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    s : σ
    n : N
    d : Finsupp σ Nat
    ⊢ Eq ((MvPolynomial.rTensor (TensorProduct.tmul R (MvPolynomial.X s) n)) d) (i …
  -/
  rw [rTensor_apply_tmul_apply, coeff_X', ite_tmul]
  /-
    🎉 no goals
  -/


lemma rTensor_apply (t : MvPolynomial σ S ⊗[R] N) (d : σ →₀ ℕ) :
    rTensor t d = ((lcoeff S d).restrictScalars R).rTensor N t :=
  TensorProduct.finsuppLeft_apply t d


@[simp]
lemma rTensor_symm_apply_single (d : σ →₀ ℕ) (s : S) (n : N) :
    rTensor.symm (Finsupp.single d (s ⊗ₜ n)) =
      (monomial d s) ⊗ₜ[R] n :=
  TensorProduct.finsuppLeft_symm_apply_single (R := R) d s n


/-- The tensor product of the polynomial algebra by a module
  is linearly equivalent to a Finsupp of that module -/
noncomputable def scalarRTensor :
    MvPolynomial σ R ⊗[R] N ≃ₗ[R] (σ →₀ ℕ) →₀ N :=
  TensorProduct.finsuppScalarLeft _ _ _


lemma scalarRTensor_apply_tmul (p : MvPolynomial σ R) (n : N) :
    scalarRTensor (p ⊗ₜ[R] n) = p.sum (fun i m ↦ Finsupp.single i (m • n)) :=
  TensorProduct.finsuppScalarLeft_apply_tmul p n


lemma scalarRTensor_apply_tmul_apply (p : MvPolynomial σ R) (n : N) (d : σ →₀ ℕ) :
    scalarRTensor (p ⊗ₜ[R] n) d = coeff d p • n :=
  TensorProduct.finsuppScalarLeft_apply_tmul_apply p n d


lemma scalarRTensor_apply_monomial_tmul (e : σ →₀ ℕ) (r : R) (n : N) (d : σ →₀ ℕ) :
    scalarRTensor (monomial e r ⊗ₜ[R] n) d = if e = d then r • n else 0 := by
  /-
    R : Type u
    N : Type v
    inst✝³ : CommSemiring R
    σ : Type u_1
    inst✝² : DecidableEq σ
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    e : Finsupp σ Nat
    r : R
    n : N
    d : Finsupp σ Nat
    ⊢ Eq ((MvPolynomial.scalarRTensor (TensorProduct.tmul R ((MvPolynomial.monomia …
  -/
  rw [scalarRTensor_apply_tmul_apply, coeff_monomial, ite_smul, zero_smul]
  /-
    🎉 no goals
  -/


lemma scalarRTensor_apply_X_tmul_apply (s : σ) (n : N) (d : σ →₀ ℕ) :
    scalarRTensor (X s ⊗ₜ[R] n) d = if Finsupp.single s 1 = d then n else 0 := by
  /-
    R : Type u
    N : Type v
    inst✝³ : CommSemiring R
    σ : Type u_1
    inst✝² : DecidableEq σ
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    s : σ
    n : N
    d : Finsupp σ Nat
    ⊢ Eq ((MvPolynomial.scalarRTensor (TensorProduct.tmul R (MvPolynomial.X s) n)) …
  -/
  rw [scalarRTensor_apply_tmul_apply, coeff_X', ite_smul, one_smul, zero_smul]
  /-
    🎉 no goals
  -/


lemma scalarRTensor_symm_apply_single (d : σ →₀ ℕ) (n : N) :
    scalarRTensor.symm (Finsupp.single d n) = (monomial d 1) ⊗ₜ[R] n :=
  TensorProduct.finsuppScalarLeft_symm_apply_single d n


/-- The algebra morphism from a tensor product of a polynomial algebra
  by an algebra to a polynomial algebra -/
noncomputable def rTensorAlgHom :
    (MvPolynomial σ S) ⊗[R] N →ₐ[S] MvPolynomial σ (S ⊗[R] N) :=
  Algebra.TensorProduct.lift
    (mapAlgHom Algebra.TensorProduct.includeLeft)
    ((IsScalarTower.toAlgHom R (S ⊗[R] N) _).comp Algebra.TensorProduct.includeRight)
                   /-
                     R : Type u
                     N : Type v
                     inst✝⁴ : CommSemiring R
                     σ : Type u_1
                     S : Type u_2
                     inst✝³ : CommSemiring S
                     inst✝² : Algebra R S
                     inst✝¹ : CommSemiring N
                     inst✝ : Algebra R N
                     p : MvPolynomial σ S
                     n : N
                     ⊢ Commute ((MvPolynomial.mapAlgHom Algebra.TensorProduct.includeLeft) p) (((Is …
                   -/
    (fun p n => by simp [commute_iff_eq, algebraMap_eq, mul_comm])
                   /-
                     🎉 no goals
                   -/


@[simp]
lemma coeff_rTensorAlgHom_tmul
    (p : MvPolynomial σ S) (n : N) (d : σ →₀ ℕ) :
    coeff d (rTensorAlgHom (p ⊗ₜ[R] n)) = (coeff d p) ⊗ₜ[R] n := by
  /-
    R : Type u
    N : Type v
    inst✝⁴ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    inst✝¹ : CommSemiring N
    inst✝ : Algebra R N
    p : MvPolynomial σ S
    n : N
    d : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff d (MvPolynomial.rTensorAlgHom (TensorProduct.tmul R p …
  -/
  rw [rTensorAlgHom, Algebra.TensorProduct.lift_tmul]
  rw [AlgHom.coe_comp, IsScalarTower.coe_toAlgHom', Function.comp_apply,
    Algebra.TensorProduct.includeRight_apply]
  /-
    R : Type u
    N : Type v
    inst✝⁴ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    inst✝¹ : CommSemiring N
    inst✝ : Algebra R N
    p : MvPolynomial σ S
    n : N
    d : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff d (HMul.hMul ((MvPolynomial.mapAlgHom Algebra.TensorP …
  -/
  rw [algebraMap_eq, mul_comm, coeff_C_mul]
  /-
    R : Type u
    N : Type v
    inst✝⁴ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    inst✝¹ : CommSemiring N
    inst✝ : Algebra R N
    p : MvPolynomial σ S
    n : N
    d : Finsupp σ Nat
    ⊢ Eq (HMul.hMul (TensorProduct.tmul R 1 n) (MvPolynomial.coeff d ((MvPolynomia …
  -/
  simp [mapAlgHom, coeff_map]
  /-
    🎉 no goals
  -/


lemma coeff_rTensorAlgHom_monomial_tmul
    (e : σ →₀ ℕ) (s : S) (n : N) (d : σ →₀ ℕ) :
    coeff d (rTensorAlgHom (monomial e s ⊗ₜ[R] n)) =
      if e = d then s ⊗ₜ[R] n else 0 := by
  /-
    R : Type u
    N : Type v
    inst✝⁵ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra R S
    inst✝² : CommSemiring N
    inst✝¹ : Algebra R N
    inst✝ : DecidableEq σ
    e : Finsupp σ Nat
    s : S
    n : N
    d : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff d (MvPolynomial.rTensorAlgHom (TensorProduct.tmul R ( …
  -/
  simp [ite_tmul]
  /-
    🎉 no goals
  -/


lemma rTensorAlgHom_toLinearMap :
    (rTensorAlgHom :
      MvPolynomial σ S ⊗[R] N →ₐ[S] MvPolynomial σ (S ⊗[R] N)).toLinearMap =
      rTensor.toLinearMap := by
  /-
    R : Type u
    N : Type v
    inst✝⁵ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra R S
    inst✝² : CommSemiring N
    inst✝¹ : Algebra R N
    inst✝ : DecidableEq σ
    ⊢ Eq MvPolynomial.rTensorAlgHom.toLinearMap ↑MvPolynomial.rTensor
  -/
  ext d n e
  dsimp only [AlgebraTensorModule.curry_apply, TensorProduct.curry_apply,
    LinearMap.coe_restrictScalars, AlgHom.toLinearMap_apply]
  simp only [coe_comp, Function.comp_apply, AlgebraTensorModule.curry_apply, curry_apply,
    LinearMap.coe_restrictScalars, AlgHom.toLinearMap_apply]
  /-
    case a.h.h.h.a
    R : Type u
    N : Type v
    inst✝⁵ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra R S
    inst✝² : CommSemiring N
    inst✝¹ : Algebra R N
    inst✝ : DecidableEq σ
    d : Finsupp σ Nat
    n : N
    e : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff e (MvPolynomial.rTensorAlgHom (TensorProduct.tmul R ( …
  -/
  rw [coeff_rTensorAlgHom_tmul]
  /-
    case a.h.h.h.a
    R : Type u
    N : Type v
    inst✝⁵ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra R S
    inst✝² : CommSemiring N
    inst✝¹ : Algebra R N
    inst✝ : DecidableEq σ
    d : Finsupp σ Nat
    n : N
    e : Finsupp σ Nat
    ⊢ Eq (TensorProduct.tmul R (MvPolynomial.coeff e ((MvPolynomial.monomial d) 1) …
  -/
  simp only [coeff]
  /-
    case a.h.h.h.a
    R : Type u
    N : Type v
    inst✝⁵ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra R S
    inst✝² : CommSemiring N
    inst✝¹ : Algebra R N
    inst✝ : DecidableEq σ
    d : Finsupp σ Nat
    n : N
    e : Finsupp σ Nat
    ⊢ Eq (TensorProduct.tmul R (((MvPolynomial.monomial d) 1) e) n) ((↑MvPolynomia …
  -/
  erw [finsuppLeft_apply_tmul_apply]
  /-
    🎉 no goals
  -/


lemma rTensorAlgHom_apply_eq (p : MvPolynomial σ S ⊗[R] N) :
    rTensorAlgHom (S := S) p = rTensor p := by
  /-
    R : Type u
    N : Type v
    inst✝⁵ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra R S
    inst✝² : CommSemiring N
    inst✝¹ : Algebra R N
    inst✝ : DecidableEq σ
    p : TensorProduct R (MvPolynomial σ S) N
    ⊢ Eq (MvPolynomial.rTensorAlgHom p) (MvPolynomial.rTensor p)
  -/
  rw [← AlgHom.toLinearMap_apply, rTensorAlgHom_toLinearMap]
  /-
    R : Type u
    N : Type v
    inst✝⁵ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra R S
    inst✝² : CommSemiring N
    inst✝¹ : Algebra R N
    inst✝ : DecidableEq σ
    p : TensorProduct R (MvPolynomial σ S) N
    ⊢ Eq (↑MvPolynomial.rTensor p) (MvPolynomial.rTensor p)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The tensor product of a polynomial algebra by an algebra
  is algebraically equivalent to a polynomial algebra -/
noncomputable def rTensorAlgEquiv :
    (MvPolynomial σ S) ⊗[R] N ≃ₐ[S] MvPolynomial σ (S ⊗[R] N) := by
  /-
    R : Type u
    N : Type v
    inst✝⁵ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra R S
    inst✝² : CommSemiring N
    inst✝¹ : Algebra R N
    inst✝ : DecidableEq σ
    ⊢ AlgEquiv S (TensorProduct R (MvPolynomial σ S) N) (MvPolynomial σ (TensorPro …
  -/
  apply AlgEquiv.ofLinearEquiv rTensor
    /-
      case map_one
      R : Type u
      N : Type v
      inst✝⁵ : CommSemiring R
      σ : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      inst✝² : CommSemiring N
      inst✝¹ : Algebra R N
      inst✝ : DecidableEq σ
      ⊢ Eq (MvPolynomial.rTensor 1) 1
    -/
  · simp only [Algebra.TensorProduct.one_def]
    /-
      case map_one
      R : Type u
      N : Type v
      inst✝⁵ : CommSemiring R
      σ : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      inst✝² : CommSemiring N
      inst✝¹ : Algebra R N
      inst✝ : DecidableEq σ
      ⊢ Eq (MvPolynomial.rTensor (TensorProduct.tmul R 1 1)) 1
    -/
    apply symm
    /-
      case map_one.a
      R : Type u
      N : Type v
      inst✝⁵ : CommSemiring R
      σ : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      inst✝² : CommSemiring N
      inst✝¹ : Algebra R N
      inst✝ : DecidableEq σ
      ⊢ Eq 1 (MvPolynomial.rTensor (TensorProduct.tmul R 1 1))
    -/
    rw [← LinearEquiv.symm_apply_eq]
    /-
      case map_one.a
      R : Type u
      N : Type v
      inst✝⁵ : CommSemiring R
      σ : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      inst✝² : CommSemiring N
      inst✝¹ : Algebra R N
      inst✝ : DecidableEq σ
      ⊢ Eq (MvPolynomial.rTensor.symm 1) (TensorProduct.tmul R 1 1)
    -/
    exact finsuppLeft_symm_apply_single (R := R) (0 : σ →₀ ℕ) (1 : S) (1 : N)
    /-
      🎉 no goals
    -/
    /-
      case map_mul
      R : Type u
      N : Type v
      inst✝⁵ : CommSemiring R
      σ : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      inst✝² : CommSemiring N
      inst✝¹ : Algebra R N
      inst✝ : DecidableEq σ
      ⊢ ∀ (x y : TensorProduct R (MvPolynomial σ S) N), Eq (MvPolynomial.rTensor (HM …
    -/
  · intro x y
    /-
      case map_mul
      R : Type u
      N : Type v
      inst✝⁵ : CommSemiring R
      σ : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      inst✝² : CommSemiring N
      inst✝¹ : Algebra R N
      inst✝ : DecidableEq σ
      x y : TensorProduct R (MvPolynomial σ S) N
      ⊢ Eq (MvPolynomial.rTensor (HMul.hMul x y)) (HMul.hMul (MvPolynomial.rTensor x …
    -/
    erw [← rTensorAlgHom_apply_eq (S := S)]
    /-
      case map_mul
      R : Type u
      N : Type v
      inst✝⁵ : CommSemiring R
      σ : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      inst✝² : CommSemiring N
      inst✝¹ : Algebra R N
      inst✝ : DecidableEq σ
      x y : TensorProduct R (MvPolynomial σ S) N
      ⊢ Eq (MvPolynomial.rTensorAlgHom (HMul.hMul x y)) (HMul.hMul (MvPolynomial.rTe …
    -/
    simp only [_root_.map_mul, rTensorAlgHom_apply_eq]
    /-
      case map_mul
      R : Type u
      N : Type v
      inst✝⁵ : CommSemiring R
      σ : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      inst✝² : CommSemiring N
      inst✝¹ : Algebra R N
      inst✝ : DecidableEq σ
      x y : TensorProduct R (MvPolynomial σ S) N
      ⊢ Eq (HMul.hMul (MvPolynomial.rTensor x) (MvPolynomial.rTensor y)) (HMul.hMul  …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
lemma rTensorAlgEquiv_apply (x : (MvPolynomial σ S) ⊗[R] N) :
    rTensorAlgEquiv x = rTensorAlgHom x := by
  /-
    R : Type u
    N : Type v
    inst✝⁵ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra R S
    inst✝² : CommSemiring N
    inst✝¹ : Algebra R N
    inst✝ : DecidableEq σ
    x : TensorProduct R (MvPolynomial σ S) N
    ⊢ Eq (MvPolynomial.rTensorAlgEquiv x) (MvPolynomial.rTensorAlgHom x)
  -/
  rw [← AlgHom.coe_coe, ← AlgEquiv.toAlgHom_eq_coe]
  /-
    R : Type u
    N : Type v
    inst✝⁵ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra R S
    inst✝² : CommSemiring N
    inst✝¹ : Algebra R N
    inst✝ : DecidableEq σ
    x : TensorProduct R (MvPolynomial σ S) N
    ⊢ Eq (↑MvPolynomial.rTensorAlgEquiv x) (MvPolynomial.rTensorAlgHom x)
  -/
  congr 1
  /-
    case e_a
    R : Type u
    N : Type v
    inst✝⁵ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra R S
    inst✝² : CommSemiring N
    inst✝¹ : Algebra R N
    inst✝ : DecidableEq σ
    x : TensorProduct R (MvPolynomial σ S) N
    ⊢ Eq (↑MvPolynomial.rTensorAlgEquiv) MvPolynomial.rTensorAlgHom
  -/
              /-
                🎉 no goals
              -/
  ext _ d <;> simpa [rTensorAlgEquiv] using rTensor_apply_tmul_apply _ _ d
              /-
                🎉 no goals
              -/


/-- The tensor product of the polynomial algebra by an algebra
  is algebraically equivalent to a polynomial algebra with
  coefficients in that algegra -/
noncomputable def scalarRTensorAlgEquiv :
    MvPolynomial σ R ⊗[R] N ≃ₐ[R] MvPolynomial σ N :=
  rTensorAlgEquiv.trans (mapAlgEquiv σ (Algebra.TensorProduct.lid R N))


/-- Tensoring `MvPolynomial σ R` on the left by an `R`-algebra `A` is algebraically
equivalent to `M̀vPolynomial σ A`. -/
noncomputable def algebraTensorAlgEquiv :
    A ⊗[R] MvPolynomial σ R ≃ₐ[A] MvPolynomial σ A := AlgEquiv.ofAlgHom
  (Algebra.TensorProduct.lift
    (Algebra.ofId A (MvPolynomial σ A))
    (MvPolynomial.mapAlgHom <| Algebra.ofId R A) (fun _ _ ↦ Commute.all _ _))
  (aeval (fun s ↦ 1 ⊗ₜ X s))
      /-
        R : Type u
        N : Type v
        inst✝⁶ : CommSemiring R
        σ : Type u_1
        S : Type u_2
        inst✝⁵ : CommSemiring S
        inst✝⁴ : Algebra R S
        inst✝³ : CommSemiring N
        inst✝² : Algebra R N
        A : Type u_3
        inst✝¹ : CommSemiring A
        inst✝ : Algebra R A
        ⊢ Eq ((Algebra.TensorProduct.lift (Algebra.ofId A (MvPolynomial σ A)) (MvPolyn …
      -/
  (by ext s; simp)
             /-
               🎉 no goals
             -/
      /-
        R : Type u
        N : Type v
        inst✝⁶ : CommSemiring R
        σ : Type u_1
        S : Type u_2
        inst✝⁵ : CommSemiring S
        inst✝⁴ : Algebra R S
        inst✝³ : CommSemiring N
        inst✝² : Algebra R N
        A : Type u_3
        inst✝¹ : CommSemiring A
        inst✝ : Algebra R A
        ⊢ Eq ((MvPolynomial.aeval fun s => TensorProduct.tmul R 1 (MvPolynomial.X s)). …
      -/
  (by ext s; simp)
             /-
               🎉 no goals
             -/


@[simp]
lemma algebraTensorAlgEquiv_tmul (a : A) (p : MvPolynomial σ R) :
    algebraTensorAlgEquiv R A (a ⊗ₜ p) = a • MvPolynomial.map (algebraMap R A) p := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    σ : Type u_1
    A : Type u_3
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    a : A
    p : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.algebraTensorAlgEquiv R A) (TensorProduct.tmul R a p)) (HS …
  -/
  simp [algebraTensorAlgEquiv, Algebra.smul_def]
  /-
    R : Type u
    inst✝² : CommSemiring R
    σ : Type u_1
    A : Type u_3
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    a : A
    p : MvPolynomial σ R
    ⊢ Eq (HMul.hMul ((Algebra.ofId A (MvPolynomial σ A)) a) (MvPolynomial.eval₂ (M …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma algebraTensorAlgEquiv_symm_X (s : σ) :
    (algebraTensorAlgEquiv R A).symm (X s) = 1 ⊗ₜ X s := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    σ : Type u_1
    A : Type u_3
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    s : σ
    ⊢ Eq ((MvPolynomial.algebraTensorAlgEquiv R A).symm (MvPolynomial.X s)) (Tenso …
  -/
  simp [algebraTensorAlgEquiv]
  /-
    🎉 no goals
  -/


@[simp]
lemma algebraTensorAlgEquiv_symm_monomial (m : σ →₀ ℕ) (a : A) :
    (algebraTensorAlgEquiv R A).symm (monomial m a) = a ⊗ₜ monomial m 1 := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    σ : Type u_1
    A : Type u_3
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    m : Finsupp σ Nat
    a : A
    ⊢ Eq ((MvPolynomial.algebraTensorAlgEquiv R A).symm ((MvPolynomial.monomial m) …
  -/
  apply @Finsupp.induction σ ℕ _ _ m
    /-
      case h0
      R : Type u
      inst✝² : CommSemiring R
      σ : Type u_1
      A : Type u_3
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      m : Finsupp σ Nat
      a : A
      ⊢ Eq ((MvPolynomial.algebraTensorAlgEquiv R A).symm ((MvPolynomial.monomial 0) …
    -/
  · simp [algebraTensorAlgEquiv]
    /-
      🎉 no goals
    -/
    /-
      case ha
      R : Type u
      inst✝² : CommSemiring R
      σ : Type u_1
      A : Type u_3
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      m : Finsupp σ Nat
      a : A
      ⊢ ∀ (a_1 : σ) (b : Nat) (f : Finsupp σ Nat), Not (Membership.mem f.support a_1 …
    -/
  · intro i n f _ _ hfa
    /-
      case ha
      R : Type u
      inst✝² : CommSemiring R
      σ : Type u_1
      A : Type u_3
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      m : Finsupp σ Nat
      a : A
      i : σ
      n : Nat
      f : Finsupp σ Nat
      a✝¹ : Not (Membership.mem f.support i)
      a✝ : Ne n 0
      hfa : Eq ((MvPolynomial.algebraTensorAlgEquiv R A).symm ((MvPolynomial.monomia …
      ⊢ Eq ((MvPolynomial.algebraTensorAlgEquiv R A).symm ((MvPolynomial.monomial (H …
    -/
    simp only [algebraTensorAlgEquiv, AlgEquiv.ofAlgHom_symm_apply] at hfa ⊢
    simp only [add_comm, monomial_add_single, _root_.map_mul, map_pow, aeval_X,
      Algebra.TensorProduct.tmul_pow, one_pow, hfa]
    /-
      case ha
      R : Type u
      inst✝² : CommSemiring R
      σ : Type u_1
      A : Type u_3
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      m : Finsupp σ Nat
      a : A
      i : σ
      n : Nat
      f : Finsupp σ Nat
      a✝¹ : Not (Membership.mem f.support i)
      a✝ : Ne n 0
      hfa : Eq ((MvPolynomial.aeval fun s => TensorProduct.tmul R 1 (MvPolynomial.X  …
      ⊢ Eq (HMul.hMul (TensorProduct.tmul R a ((MvPolynomial.monomial f) 1)) (Tensor …
    -/
    nth_rw 2 [← mul_one a]
    /-
      case ha
      R : Type u
      inst✝² : CommSemiring R
      σ : Type u_1
      A : Type u_3
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      m : Finsupp σ Nat
      a : A
      i : σ
      n : Nat
      f : Finsupp σ Nat
      a✝¹ : Not (Membership.mem f.support i)
      a✝ : Ne n 0
      hfa : Eq ((MvPolynomial.aeval fun s => TensorProduct.tmul R 1 (MvPolynomial.X  …
      ⊢ Eq (HMul.hMul (TensorProduct.tmul R a ((MvPolynomial.monomial f) 1)) (Tensor …
    -/
    rw [Algebra.TensorProduct.tmul_mul_tmul]
    /-
      🎉 no goals
    -/


lemma aeval_one_tmul (f : σ → S) (p : MvPolynomial σ R) :
    (aeval fun x ↦ (1 ⊗ₜ[R] f x : N ⊗[R] S)) p = 1 ⊗ₜ[R] (aeval f) p := by
  /-
    R : Type u
    N : Type v
    inst✝⁴ : CommSemiring R
    σ : Type u_1
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    inst✝¹ : CommSemiring N
    inst✝ : Algebra R N
    f : σ → S
    p : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.aeval fun x => TensorProduct.tmul R 1 (f x)) p) (TensorPro …
  -/
  induction' p using MvPolynomial.induction_on with a p q hp hq p i h
  · simp only [map_C, algHom_C, Algebra.TensorProduct.algebraMap_apply,
      RingHomCompTriple.comp_apply]
    /-
      case h_C
      R : Type u
      N : Type v
      inst✝⁴ : CommSemiring R
      σ : Type u_1
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      inst✝¹ : CommSemiring N
      inst✝ : Algebra R N
      f : σ → S
      a : R
      ⊢ Eq (TensorProduct.tmul R ((algebraMap R N) a) 1) (TensorProduct.tmul R 1 ((a …
    -/
    rw [← mul_one ((algebraMap R N) a), ← Algebra.smul_def, smul_tmul, Algebra.smul_def, mul_one]
    /-
      🎉 no goals
    -/
    /-
      case h_add
      R : Type u
      N : Type v
      inst✝⁴ : CommSemiring R
      σ : Type u_1
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      inst✝¹ : CommSemiring N
      inst✝ : Algebra R N
      f : σ → S
      p q : MvPolynomial σ R
      hp : Eq ((MvPolynomial.aeval fun x => TensorProduct.tmul R 1 (f x)) p) (Tensor …
      hq : Eq ((MvPolynomial.aeval fun x => TensorProduct.tmul R 1 (f x)) q) (Tensor …
      ⊢ Eq ((MvPolynomial.aeval fun x => TensorProduct.tmul R 1 (f x)) (HAdd.hAdd p  …
    -/
  · simp [hp, hq, tmul_add]
    /-
      🎉 no goals
    -/
    /-
      case h_X
      R : Type u
      N : Type v
      inst✝⁴ : CommSemiring R
      σ : Type u_1
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      inst✝¹ : CommSemiring N
      inst✝ : Algebra R N
      f : σ → S
      p : MvPolynomial σ R
      i : σ
      h : Eq ((MvPolynomial.aeval fun x => TensorProduct.tmul R 1 (f x)) p) (TensorP …
      ⊢ Eq ((MvPolynomial.aeval fun x => TensorProduct.tmul R 1 (f x)) (HMul.hMul p  …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/


