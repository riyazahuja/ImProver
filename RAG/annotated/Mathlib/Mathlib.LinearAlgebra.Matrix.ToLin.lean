/-- `Matrix.vecMul M` is a linear map. -/
def Matrix.vecMulLinear [Fintype m] (M : Matrix m n R) : (m → R) →ₗ[R] n → R where
  toFun x := x ᵥ* M
  map_add' _ _ := funext fun _ ↦ add_dotProduct _ _ _
  map_smul' _ _ := funext fun _ ↦ smul_dotProduct _ _ _


@[simp] theorem Matrix.vecMulLinear_apply [Fintype m] (M : Matrix m n R) (x : m → R) :
    M.vecMulLinear x = x ᵥ* M := rfl


theorem Matrix.coe_vecMulLinear [Fintype m] (M : Matrix m n R) :
    (M.vecMulLinear : _ → _) = M.vecMul := rfl


set_option linter.deprecated false in
@[simp, deprecated Matrix.single_one_vecMul (since := "2024-08-09")]
theorem Matrix.vecMul_stdBasis [DecidableEq m] (M : Matrix m n R) (i j) :
    (LinearMap.stdBasis R (fun _ ↦ R) i 1 ᵥ* M) j = M i j :=
  congr_fun (Matrix.single_one_vecMul ..) j


theorem range_vecMulLinear (M : Matrix m n R) :
    LinearMap.range M.vecMulLinear = span R (range M) := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    m : Type u_3
    n : Type u_4
    inst✝ : Fintype m
    M : Matrix m n R
    ⊢ Eq (LinearMap.range M.vecMulLinear) (Submodule.span R (Set.range M))
  -/
  letI := Classical.decEq m
  simp_rw [range_eq_map, ← iSup_range_single, Submodule.map_iSup, range_eq_map, ←
    Ideal.span_singleton_one, Ideal.span, Submodule.map_span, image_image, image_singleton,
    Matrix.vecMulLinear_apply, iSup_span, range_eq_iUnion, iUnion_singleton_eq_range,
    LinearMap.single, LinearMap.coe_mk, AddHom.coe_mk]
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    m : Type u_3
    n : Type u_4
    inst✝ : Fintype m
    M : Matrix m n R
    this : DecidableEq m := Classical.decEq m
    ⊢ Eq (Submodule.span R (Set.range fun x => Matrix.vecMul (Pi.single x 1) M)) ( …
  -/
  unfold vecMul
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    m : Type u_3
    n : Type u_4
    inst✝ : Fintype m
    M : Matrix m n R
    this : DecidableEq m := Classical.decEq m
    ⊢ Eq
        (Submodule.span R
          (Set.range fun x x_1 =>
            let j := x_1;
            dotProduct (Pi.single x 1) fun i => M i j))
        (Submodule.span R (Set.range M))
  -/
  simp_rw [single_dotProduct, one_mul]
  /-
    🎉 no goals
  -/


theorem Matrix.vecMul_injective_iff {R : Type*} [CommRing R] {M : Matrix m n R} :
    Function.Injective M.vecMul ↔ LinearIndependent R (fun i ↦ M i) := by
  /-
    m : Type u_3
    n : Type u_4
    inst✝¹ : Fintype m
    R : Type u_5
    inst✝ : CommRing R
    M : Matrix m n R
    ⊢ Iff (Function.Injective fun v => Matrix.vecMul v M) (LinearIndependent R fun …
  -/
  rw [← coe_vecMulLinear]
  simp only [← LinearMap.ker_eq_bot, Fintype.linearIndependent_iff, Submodule.eq_bot_iff,
    LinearMap.mem_ker, vecMulLinear_apply]
  /-
    m : Type u_3
    n : Type u_4
    inst✝¹ : Fintype m
    R : Type u_5
    inst✝ : CommRing R
    M : Matrix m n R
    ⊢ Iff (∀ (x : m → R), Eq (Matrix.vecMul x M) 0 → Eq x 0) (∀ (g : m → R), Eq (F …
  -/
  refine ⟨fun h c h0 ↦ congr_fun <| h c ?_, fun h c h0 ↦ funext <| h c ?_⟩
    /-
      case refine_1
      m : Type u_3
      n : Type u_4
      inst✝¹ : Fintype m
      R : Type u_5
      inst✝ : CommRing R
      M : Matrix m n R
      h : ∀ (x : m → R), Eq (Matrix.vecMul x M) 0 → Eq x 0
      c : m → R
      h0 : Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) (M i)) 0
      ⊢ Eq (Matrix.vecMul c M) 0
    -/
  · rw [← h0]
    /-
      case refine_1
      m : Type u_3
      n : Type u_4
      inst✝¹ : Fintype m
      R : Type u_5
      inst✝ : CommRing R
      M : Matrix m n R
      h : ∀ (x : m → R), Eq (Matrix.vecMul x M) 0 → Eq x 0
      c : m → R
      h0 : Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) (M i)) 0
      ⊢ Eq (Matrix.vecMul c M) (Finset.univ.sum fun i => HSMul.hSMul (c i) (M i))
    -/
    ext i
    /-
      case refine_1.h
      m : Type u_3
      n : Type u_4
      inst✝¹ : Fintype m
      R : Type u_5
      inst✝ : CommRing R
      M : Matrix m n R
      h : ∀ (x : m → R), Eq (Matrix.vecMul x M) 0 → Eq x 0
      c : m → R
      h0 : Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) (M i)) 0
      i : n
      ⊢ Eq (Matrix.vecMul c M i) (Finset.univ.sum (fun i => HSMul.hSMul (c i) (M i)) …
    -/
    simp [vecMul, dotProduct]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      m : Type u_3
      n : Type u_4
      inst✝¹ : Fintype m
      R : Type u_5
      inst✝ : CommRing R
      M : Matrix m n R
      h : ∀ (g : m → R), Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (M i)) 0 → ∀ …
      c : m → R
      h0 : Eq (Matrix.vecMul c M) 0
      ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) (M i)) 0
    -/
  · rw [← h0]
    /-
      case refine_2
      m : Type u_3
      n : Type u_4
      inst✝¹ : Fintype m
      R : Type u_5
      inst✝ : CommRing R
      M : Matrix m n R
      h : ∀ (g : m → R), Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (M i)) 0 → ∀ …
      c : m → R
      h0 : Eq (Matrix.vecMul c M) 0
      ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) (M i)) (Matrix.vecMul c M)
    -/
    ext j
    /-
      case refine_2.h
      m : Type u_3
      n : Type u_4
      inst✝¹ : Fintype m
      R : Type u_5
      inst✝ : CommRing R
      M : Matrix m n R
      h : ∀ (g : m → R), Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (M i)) 0 → ∀ …
      c : m → R
      h0 : Eq (Matrix.vecMul c M) 0
      j : n
      ⊢ Eq (Finset.univ.sum (fun i => HSMul.hSMul (c i) (M i)) j) (Matrix.vecMul c M …
    -/
    simp [vecMul, dotProduct]
    /-
      🎉 no goals
    -/


/-- Linear maps `(m → R) →ₗ[R] (n → R)` are linearly equivalent over `Rᵐᵒᵖ` to `Matrix m n R`,
by having matrices act by right multiplication.
 -/
def LinearMap.toMatrixRight' : ((m → R) →ₗ[R] n → R) ≃ₗ[Rᵐᵒᵖ] Matrix m n R where
  toFun f i j := f (single R (fun _ ↦ R) i 1) j
  invFun := Matrix.vecMulLinear
  right_inv M := by
    /-
      R : Type u_1
      inst✝² : Semiring R
      l : Type u_2
      m : Type u_3
      n : Type u_4
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      M : Matrix m n R
      ⊢ Eq ({ toFun := fun f i j => f ((LinearMap.single R (fun x => R) i) 1) j, map …
    -/
    ext i j
    /-
      case a
      R : Type u_1
      inst✝² : Semiring R
      l : Type u_2
      m : Type u_3
      n : Type u_4
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      M : Matrix m n R
      i : m
      j : n
      ⊢ Eq ({ toFun := fun f i j => f ((LinearMap.single R (fun x => R) i) 1) j, map …
    -/
    /-
      R : Type u_1
      inst✝² : Semiring R
      l : Type u_2
      m : Type u_3
      n : Type u_4
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      f : LinearMap (RingHom.id R) (m → R) (n → R)
      ⊢ Eq ({ toFun := fun f i j => f ((LinearMap.single R (fun x => R) i) 1) j, map …
    -/
    simp
    /-
      R : Type u_1
      inst✝² : Semiring R
      l : Type u_2
      m : Type u_3
      n : Type u_4
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      f : LinearMap (RingHom.id R) (m → R) (n → R)
      ⊢ ∀ (i : m), Eq (({ toFun := fun f i j => f ((LinearMap.single R (fun x => R)  …
    -/
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝² : Semiring R
      l : Type u_2
      m : Type u_3
      n : Type u_4
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      f g : LinearMap (RingHom.id R) (m → R) (n → R)
      ⊢ Eq ((fun f i j => f ((LinearMap.single R (fun x => R) i) 1) j) (HAdd.hAdd f  …
    -/
    /-
      case h
      R : Type u_1
      inst✝² : Semiring R
      l : Type u_2
      m : Type u_3
      n : Type u_4
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      f : LinearMap (RingHom.id R) (m → R) (n → R)
      j : m
      i : n
      ⊢ Eq (({ toFun := fun f i j => f ((LinearMap.single R (fun x => R) i) 1) j, ma …
    -/
    /-
      case a
      R : Type u_1
      inst✝² : Semiring R
      l : Type u_2
      m : Type u_3
      n : Type u_4
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      f g : LinearMap (RingHom.id R) (m → R) (n → R)
      i : m
      j : n
      ⊢ Eq ((fun f i j => f ((LinearMap.single R (fun x => R) i) 1) j) (HAdd.hAdd f  …
    -/
  left_inv f := by
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝² : Semiring R
      l : Type u_2
      m : Type u_3
      n : Type u_4
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      c : MulOpposite R
      f : LinearMap (RingHom.id R) (m → R) (n → R)
      ⊢ Eq ({ toFun := fun f i j => f ((LinearMap.single R (fun x => R) i) 1) j, map …
    -/
    apply (Pi.basisFun R m).ext
    /-
      case a
      R : Type u_1
      inst✝² : Semiring R
      l : Type u_2
      m : Type u_3
      n : Type u_4
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      c : MulOpposite R
      f : LinearMap (RingHom.id R) (m → R) (n → R)
      i : m
      j : n
      ⊢ Eq ({ toFun := fun f i j => f ((LinearMap.single R (fun x => R) i) 1) j, map …
    -/
    intro j; ext i
    /-
      🎉 no goals
    -/
    simp
  map_add' f g := by
    ext i j
    simp only [Pi.add_apply, LinearMap.add_apply, Matrix.add_apply]
  map_smul' c f := by
    ext i j
    simp only [Pi.smul_apply, LinearMap.smul_apply, RingHom.id_apply, Matrix.smul_apply]


/-- A `Matrix m n R` is linearly equivalent over `Rᵐᵒᵖ` to a linear map `(m → R) →ₗ[R] (n → R)`,
by having matrices act by right multiplication. -/
abbrev Matrix.toLinearMapRight' [DecidableEq m] : Matrix m n R ≃ₗ[Rᵐᵒᵖ] (m → R) →ₗ[R] n → R :=
  LinearEquiv.symm LinearMap.toMatrixRight'


@[simp]
theorem Matrix.toLinearMapRight'_apply (M : Matrix m n R) (v : m → R) :
    (Matrix.toLinearMapRight') M v = v ᵥ* M := rfl


@[simp]
theorem Matrix.toLinearMapRight'_mul [Fintype l] [DecidableEq l] (M : Matrix l m R)
    (N : Matrix m n R) :
    Matrix.toLinearMapRight' (M * N) =
      (Matrix.toLinearMapRight' N).comp (Matrix.toLinearMapRight' M) :=
  LinearMap.ext fun _x ↦ (vecMul_vecMul _ M N).symm


theorem Matrix.toLinearMapRight'_mul_apply [Fintype l] [DecidableEq l] (M : Matrix l m R)
    (N : Matrix m n R) (x) :
    Matrix.toLinearMapRight' (M * N) x =
      Matrix.toLinearMapRight' N (Matrix.toLinearMapRight' M x) :=
  (vecMul_vecMul _ M N).symm


@[simp]
theorem Matrix.toLinearMapRight'_one :
    Matrix.toLinearMapRight' (1 : Matrix m m R) = LinearMap.id := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    m : Type u_3
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    ⊢ Eq (Matrix.toLinearMapRight' 1) LinearMap.id
  -/
  ext
  /-
    case h.h.h
    R : Type u_1
    inst✝² : Semiring R
    m : Type u_3
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    i✝ x✝ : m
    ⊢ Eq (((Matrix.toLinearMapRight' 1).comp (LinearMap.single R (fun i => R) i✝)) …
  -/
  simp [LinearMap.one_apply]
  /-
    🎉 no goals
  -/


/-- If `M` and `M'` are each other's inverse matrices, they provide an equivalence between `n → A`
and `m → A` corresponding to `M.vecMul` and `M'.vecMul`. -/
@[simps]
def Matrix.toLinearEquivRight'OfInv [Fintype n] [DecidableEq n] {M : Matrix m n R}
    {M' : Matrix n m R} (hMM' : M * M' = 1) (hM'M : M' * M = 1) : (n → R) ≃ₗ[R] m → R :=
  { LinearMap.toMatrixRight'.symm M' with
    toFun := Matrix.toLinearMapRight' M'
    invFun := Matrix.toLinearMapRight' M
    left_inv := fun x ↦ by
      /-
        R : Type u_1
        inst✝⁴ : Semiring R
        l : Type u_2
        m : Type u_3
        n : Type u_4
        inst✝³ : Fintype m
        inst✝² : DecidableEq m
        inst✝¹ : Fintype n
        inst✝ : DecidableEq n
        M : Matrix m n R
        M' : Matrix n m R
        hMM' : Eq (HMul.hMul M M') 1
        hM'M : Eq (HMul.hMul M' M) 1
        x : n → R
        ⊢ Eq ((Matrix.toLinearMapRight' M) ({ toFun := ⇑(Matrix.toLinearMapRight' M'), …
      -/
      rw [← Matrix.toLinearMapRight'_mul_apply, hM'M, Matrix.toLinearMapRight'_one, id_apply]
      /-
        🎉 no goals
      -/
    right_inv := fun x ↦ by
      /-
        R : Type u_1
        inst✝⁴ : Semiring R
        l : Type u_2
        m : Type u_3
        n : Type u_4
        inst✝³ : Fintype m
        inst✝² : DecidableEq m
        inst✝¹ : Fintype n
        inst✝ : DecidableEq n
        M : Matrix m n R
        M' : Matrix n m R
        hMM' : Eq (HMul.hMul M M') 1
        hM'M : Eq (HMul.hMul M' M) 1
        x : m → R
        ⊢ Eq ({ toFun := ⇑(Matrix.toLinearMapRight' M'), map_add' := ⋯, map_smul' := ⋯ …
      -/
      dsimp only -- Porting note: needed due to non-flat structures
      /-
        R : Type u_1
        inst✝⁴ : Semiring R
        l : Type u_2
        m : Type u_3
        n : Type u_4
        inst✝³ : Fintype m
        inst✝² : DecidableEq m
        inst✝¹ : Fintype n
        inst✝ : DecidableEq n
        M : Matrix m n R
        M' : Matrix n m R
        hMM' : Eq (HMul.hMul M M') 1
        hM'M : Eq (HMul.hMul M' M) 1
        x : m → R
        ⊢ Eq ((Matrix.toLinearMapRight' M') ((Matrix.toLinearMapRight' M) x)) x
      -/
      rw [← Matrix.toLinearMapRight'_mul_apply, hMM', Matrix.toLinearMapRight'_one, id_apply] }
      /-
        🎉 no goals
      -/


/-- `Matrix.mulVec M` is a linear map. -/
def Matrix.mulVecLin [Fintype n] (M : Matrix m n R) : (n → R) →ₗ[R] m → R where
  toFun := M.mulVec
  map_add' _ _ := funext fun _ ↦ dotProduct_add _ _ _
  map_smul' _ _ := funext fun _ ↦ dotProduct_smul _ _ _


theorem Matrix.coe_mulVecLin [Fintype n] (M : Matrix m n R) :
    (M.mulVecLin : _ → _) = M.mulVec := rfl


@[simp]
theorem Matrix.mulVecLin_apply [Fintype n] (M : Matrix m n R) (v : n → R) :
    M.mulVecLin v = M *ᵥ v :=
  rfl


@[simp]
theorem Matrix.mulVecLin_zero [Fintype n] : Matrix.mulVecLin (0 : Matrix m n R) = 0 :=
  LinearMap.ext zero_mulVec


@[simp]
theorem Matrix.mulVecLin_add [Fintype n] (M N : Matrix m n R) :
    (M + N).mulVecLin = M.mulVecLin + N.mulVecLin :=
  LinearMap.ext fun _ ↦ add_mulVec _ _ _


@[simp] theorem Matrix.mulVecLin_transpose [Fintype m] (M : Matrix m n R) :
    Mᵀ.mulVecLin = M.vecMulLinear := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    m : Type u_4
    n : Type u_5
    inst✝ : Fintype m
    M : Matrix m n R
    ⊢ Eq M.transpose.mulVecLin M.vecMulLinear
  -/
  ext; simp [mulVec_transpose]
       /-
         🎉 no goals
       -/


@[simp] theorem Matrix.vecMulLinear_transpose [Fintype n] (M : Matrix m n R) :
    Mᵀ.vecMulLinear = M.mulVecLin := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    m : Type u_4
    n : Type u_5
    inst✝ : Fintype n
    M : Matrix m n R
    ⊢ Eq M.transpose.vecMulLinear M.mulVecLin
  -/
  ext; simp [vecMul_transpose]
       /-
         🎉 no goals
       -/


theorem Matrix.mulVecLin_submatrix [Fintype n] [Fintype l] (f₁ : m → k) (e₂ : n ≃ l)
    (M : Matrix k l R) :
    (M.submatrix f₁ e₂).mulVecLin = funLeft R R f₁ ∘ₗ M.mulVecLin ∘ₗ funLeft _ _ e₂.symm :=
  LinearMap.ext fun _ ↦ submatrix_mulVec_equiv _ _ _ _


/-- A variant of `Matrix.mulVecLin_submatrix` that keeps around `LinearEquiv`s. -/
theorem Matrix.mulVecLin_reindex [Fintype n] [Fintype l] (e₁ : k ≃ m) (e₂ : l ≃ n)
    (M : Matrix k l R) :
    (reindex e₁ e₂ M).mulVecLin =
      ↑(LinearEquiv.funCongrLeft R R e₁.symm) ∘ₗ
        M.mulVecLin ∘ₗ ↑(LinearEquiv.funCongrLeft R R e₂) :=
  Matrix.mulVecLin_submatrix _ _ _


@[simp]
theorem Matrix.mulVecLin_one [DecidableEq n] :
    Matrix.mulVecLin (1 : Matrix n n R) = LinearMap.id := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type u_5
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    ⊢ Eq (Matrix.mulVecLin 1) LinearMap.id
  -/
  ext; simp [Matrix.one_apply, Pi.single_apply]
       /-
         🎉 no goals
       -/


@[simp]
theorem Matrix.mulVecLin_mul [Fintype m] (M : Matrix l m R) (N : Matrix m n R) :
    Matrix.mulVecLin (M * N) = (Matrix.mulVecLin M).comp (Matrix.mulVecLin N) :=
  LinearMap.ext fun _ ↦ (mulVec_mulVec _ _ _).symm


theorem Matrix.ker_mulVecLin_eq_bot_iff {M : Matrix m n R} :
    (LinearMap.ker M.mulVecLin) = ⊥ ↔ ∀ v, M *ᵥ v = 0 → v = 0 := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    m : Type u_4
    n : Type u_5
    inst✝ : Fintype n
    M : Matrix m n R
    ⊢ Iff (Eq (LinearMap.ker M.mulVecLin) Bot.bot) (∀ (v : n → R), Eq (M.mulVec v) …
  -/
  simp only [Submodule.eq_bot_iff, LinearMap.mem_ker, Matrix.mulVecLin_apply]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated Matrix.mulVec_single_one (since := "2024-08-09")]
theorem Matrix.mulVec_stdBasis [DecidableEq n] (M : Matrix m n R) (i j) :
    (M *ᵥ LinearMap.stdBasis R (fun _ ↦ R) j 1) i = M i j :=
  congr_fun (Matrix.mulVec_single_one ..) i


set_option linter.deprecated false in
@[simp, deprecated Matrix.mulVec_single_one (since := "2024-08-09")]
theorem Matrix.mulVec_stdBasis_apply [DecidableEq n] (M : Matrix m n R) (j) :
    M *ᵥ LinearMap.stdBasis R (fun _ ↦ R) j 1 = Mᵀ j :=
  Matrix.mulVec_single_one ..


theorem Matrix.range_mulVecLin (M : Matrix m n R) :
    LinearMap.range M.mulVecLin = span R (range Mᵀ) := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    m : Type u_4
    n : Type u_5
    inst✝ : Fintype n
    M : Matrix m n R
    ⊢ Eq (LinearMap.range M.mulVecLin) (Submodule.span R (Set.range M.transpose))
  -/
  rw [← vecMulLinear_transpose, range_vecMulLinear]
  /-
    🎉 no goals
  -/


theorem Matrix.mulVec_injective_iff {R : Type*} [CommRing R] {M : Matrix m n R} :
    Function.Injective M.mulVec ↔ LinearIndependent R (fun i ↦ Mᵀ i) := by
  /-
    m : Type u_4
    n : Type u_5
    inst✝¹ : Fintype n
    R : Type u_6
    inst✝ : CommRing R
    M : Matrix m n R
    ⊢ Iff (Function.Injective M.mulVec) (LinearIndependent R fun i => M.transpose i)
  -/
  change Function.Injective (fun x ↦ _) ↔ _
  /-
    m : Type u_4
    n : Type u_5
    inst✝¹ : Fintype n
    R : Type u_6
    inst✝ : CommRing R
    M : Matrix m n R
    ⊢ Iff (Function.Injective fun x => M.mulVec x) (LinearIndependent R fun i => M …
  -/
  simp_rw [← M.vecMul_transpose, vecMul_injective_iff]
  /-
    🎉 no goals
  -/


/-- Linear maps `(n → R) →ₗ[R] (m → R)` are linearly equivalent to `Matrix m n R`. -/
def LinearMap.toMatrix' : ((n → R) →ₗ[R] m → R) ≃ₗ[R] Matrix m n R where
  toFun f := of fun i j ↦ f (Pi.single j 1) i
  invFun := Matrix.mulVecLin
  right_inv M := by
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      k : Type u_2
      l : Type u_3
      m : Type u_4
      n : Type u_5
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix m n R
      ⊢ Eq ({ toFun := fun f => Matrix.of fun i j => f (Pi.single j 1) i, map_add' : …
    -/
    ext i j
    /-
      case a
      R : Type u_1
      inst✝² : CommSemiring R
      k : Type u_2
      l : Type u_3
      m : Type u_4
      n : Type u_5
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix m n R
      i : m
      j : n
      ⊢ Eq ({ toFun := fun f => Matrix.of fun i j => f (Pi.single j 1) i, map_add' : …
    -/
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      k : Type u_2
      l : Type u_3
      m : Type u_4
      n : Type u_5
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      f : LinearMap (RingHom.id R) (n → R) (m → R)
      ⊢ Eq ({ toFun := fun f => Matrix.of fun i j => f (Pi.single j 1) i, map_add' : …
    -/
    simp only [Matrix.mulVec_single_one, Matrix.mulVecLin_apply, of_apply, transpose_apply]
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      k : Type u_2
      l : Type u_3
      m : Type u_4
      n : Type u_5
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      f : LinearMap (RingHom.id R) (n → R) (m → R)
      ⊢ ∀ (i : n), Eq (({ toFun := fun f => Matrix.of fun i j => f (Pi.single j 1) i …
    -/
    /-
      🎉 no goals
    -/
  left_inv f := by
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      k : Type u_2
      l : Type u_3
      m : Type u_4
      n : Type u_5
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      f g : LinearMap (RingHom.id R) (n → R) (m → R)
      ⊢ Eq ((fun f => Matrix.of fun i j => f (Pi.single j 1) i) (HAdd.hAdd f g)) (HA …
    -/
    apply (Pi.basisFun R n).ext
    /-
      case a
      R : Type u_1
      inst✝² : CommSemiring R
      k : Type u_2
      l : Type u_3
      m : Type u_4
      n : Type u_5
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      f g : LinearMap (RingHom.id R) (n → R) (m → R)
      i : m
      j : n
      ⊢ Eq ((fun f => Matrix.of fun i j => f (Pi.single j 1) i) (HAdd.hAdd f g) i j) …
    -/
    intro j; ext i
    /-
      🎉 no goals
    -/
    simp only [Pi.basisFun_apply, Matrix.mulVec_single_one,
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      k : Type u_2
      l : Type u_3
      m : Type u_4
      n : Type u_5
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      c : R
      f : LinearMap (RingHom.id R) (n → R) (m → R)
      ⊢ Eq ({ toFun := fun f => Matrix.of fun i j => f (Pi.single j 1) i, map_add' : …
    -/
      Matrix.mulVecLin_apply, of_apply, transpose_apply]
    /-
      case a
      R : Type u_1
      inst✝² : CommSemiring R
      k : Type u_2
      l : Type u_3
      m : Type u_4
      n : Type u_5
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      c : R
      f : LinearMap (RingHom.id R) (n → R) (m → R)
      i : m
      j : n
      ⊢ Eq ({ toFun := fun f => Matrix.of fun i j => f (Pi.single j 1) i, map_add' : …
    -/
  map_add' f g := by
    /-
      🎉 no goals
    -/
    ext i j
    simp only [Pi.add_apply, LinearMap.add_apply, of_apply, Matrix.add_apply]
  map_smul' c f := by
    ext i j
    simp only [Pi.smul_apply, LinearMap.smul_apply, RingHom.id_apply, of_apply, Matrix.smul_apply]


/-- A `Matrix m n R` is linearly equivalent to a linear map `(n → R) →ₗ[R] (m → R)`.

Note that the forward-direction does not require `DecidableEq` and is `Matrix.vecMulLin`. -/
def Matrix.toLin' : Matrix m n R ≃ₗ[R] (n → R) →ₗ[R] m → R :=
  LinearMap.toMatrix'.symm


theorem Matrix.toLin'_apply' (M : Matrix m n R) : Matrix.toLin' M = M.mulVecLin :=
  rfl


@[simp]
theorem LinearMap.toMatrix'_symm :
    (LinearMap.toMatrix'.symm : Matrix m n R ≃ₗ[R] _) = Matrix.toLin' :=
  rfl


@[simp]
theorem Matrix.toLin'_symm :
    (Matrix.toLin'.symm : ((n → R) →ₗ[R] m → R) ≃ₗ[R] _) = LinearMap.toMatrix' :=
  rfl


@[simp]
theorem LinearMap.toMatrix'_toLin' (M : Matrix m n R) : LinearMap.toMatrix' (Matrix.toLin' M) = M :=
  LinearMap.toMatrix'.apply_symm_apply M


@[simp]
theorem Matrix.toLin'_toMatrix' (f : (n → R) →ₗ[R] m → R) :
    Matrix.toLin' (LinearMap.toMatrix' f) = f :=
  Matrix.toLin'.apply_symm_apply f


@[simp]
theorem LinearMap.toMatrix'_apply (f : (n → R) →ₗ[R] m → R) (i j) :
    LinearMap.toMatrix' f i j = f (fun j' ↦ if j' = j then 1 else 0) i := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    m : Type u_4
    n : Type u_5
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    f : LinearMap (RingHom.id R) (n → R) (m → R)
    i : m
    j : n
    ⊢ Eq (LinearMap.toMatrix' f i j) (f (fun j' => ite (Eq j' j) 1 0) i)
  -/
  simp only [LinearMap.toMatrix', LinearEquiv.coe_mk, of_apply]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    m : Type u_4
    n : Type u_5
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    f : LinearMap (RingHom.id R) (n → R) (m → R)
    i : m
    j : n
    ⊢ Eq (f (Pi.single j 1) i) (f (fun j' => ite (Eq j' j) 1 0) i)
  -/
  refine congr_fun ?_ _  -- Porting note: `congr` didn't do this
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    m : Type u_4
    n : Type u_5
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    f : LinearMap (RingHom.id R) (n → R) (m → R)
    i : m
    j : n
    ⊢ Eq (f (Pi.single j 1)) (f fun j' => ite (Eq j' j) 1 0)
  -/
  congr
  /-
    case h.e_6.h
    R : Type u_1
    inst✝² : CommSemiring R
    m : Type u_4
    n : Type u_5
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    f : LinearMap (RingHom.id R) (n → R) (m → R)
    i : m
    j : n
    ⊢ Eq (Pi.single j 1) fun j' => ite (Eq j' j) 1 0
  -/
  ext j'
  /-
    case h.e_6.h.h
    R : Type u_1
    inst✝² : CommSemiring R
    m : Type u_4
    n : Type u_5
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    f : LinearMap (RingHom.id R) (n → R) (m → R)
    i : m
    j j' : n
    ⊢ Eq (Pi.single j 1 j') (ite (Eq j' j) 1 0)
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝² : CommSemiring R
      m : Type u_4
      n : Type u_5
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      f : LinearMap (RingHom.id R) (n → R) (m → R)
      i : m
      j j' : n
      h : Eq j' j
      ⊢ Eq (Pi.single j 1 j') 1
    -/
  · rw [h, Pi.single_eq_same]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝² : CommSemiring R
    m : Type u_4
    n : Type u_5
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    f : LinearMap (RingHom.id R) (n → R) (m → R)
    i : m
    j j' : n
    h : Not (Eq j' j)
    ⊢ Eq (Pi.single j 1 j') 0
  -/
  apply Pi.single_eq_of_ne h
  /-
    🎉 no goals
  -/


@[simp]
theorem Matrix.toLin'_apply (M : Matrix m n R) (v : n → R) : Matrix.toLin' M v = M *ᵥ v :=
  rfl


@[simp]
theorem Matrix.toLin'_one : Matrix.toLin' (1 : Matrix n n R) = LinearMap.id :=
  Matrix.mulVecLin_one


@[simp]
theorem LinearMap.toMatrix'_id : LinearMap.toMatrix' (LinearMap.id : (n → R) →ₗ[R] n → R) = 1 := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type u_5
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    ⊢ Eq (LinearMap.toMatrix' LinearMap.id) 1
  -/
  ext
  /-
    case a
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type u_5
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    i✝ j✝ : n
    ⊢ Eq (LinearMap.toMatrix' LinearMap.id i✝ j✝) (1 i✝ j✝)
  -/
  rw [Matrix.one_apply, LinearMap.toMatrix'_apply, id_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem LinearMap.toMatrix'_one : LinearMap.toMatrix' (1 : (n → R) →ₗ[R] n → R) = 1 :=
  LinearMap.toMatrix'_id


@[simp]
theorem Matrix.toLin'_mul [Fintype m] [DecidableEq m] (M : Matrix l m R) (N : Matrix m n R) :
    Matrix.toLin' (M * N) = (Matrix.toLin' M).comp (Matrix.toLin' N) :=
  Matrix.mulVecLin_mul _ _


@[simp]
theorem Matrix.toLin'_submatrix [Fintype l] [DecidableEq l] (f₁ : m → k) (e₂ : n ≃ l)
    (M : Matrix k l R) :
    Matrix.toLin' (M.submatrix f₁ e₂) =
      funLeft R R f₁ ∘ₗ (Matrix.toLin' M) ∘ₗ funLeft _ _ e₂.symm :=
  Matrix.mulVecLin_submatrix _ _ _


/-- A variant of `Matrix.toLin'_submatrix` that keeps around `LinearEquiv`s. -/
theorem Matrix.toLin'_reindex [Fintype l] [DecidableEq l] (e₁ : k ≃ m) (e₂ : l ≃ n)
    (M : Matrix k l R) :
    Matrix.toLin' (reindex e₁ e₂ M) =
      ↑(LinearEquiv.funCongrLeft R R e₁.symm) ∘ₗ (Matrix.toLin' M) ∘ₗ
        ↑(LinearEquiv.funCongrLeft R R e₂) :=
  Matrix.mulVecLin_reindex _ _ _


/-- Shortcut lemma for `Matrix.toLin'_mul` and `LinearMap.comp_apply` -/
theorem Matrix.toLin'_mul_apply [Fintype m] [DecidableEq m] (M : Matrix l m R) (N : Matrix m n R)
    (x) : Matrix.toLin' (M * N) x = Matrix.toLin' M (Matrix.toLin' N x) := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    l : Type u_3
    m : Type u_4
    n : Type u_5
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    M : Matrix l m R
    N : Matrix m n R
    x : n → R
    ⊢ Eq ((Matrix.toLin' (HMul.hMul M N)) x) ((Matrix.toLin' M) ((Matrix.toLin' N) …
  -/
  rw [Matrix.toLin'_mul, LinearMap.comp_apply]
  /-
    🎉 no goals
  -/


theorem LinearMap.toMatrix'_comp [Fintype l] [DecidableEq l] (f : (n → R) →ₗ[R] m → R)
    (g : (l → R) →ₗ[R] n → R) :
    LinearMap.toMatrix' (f.comp g) = LinearMap.toMatrix' f * LinearMap.toMatrix' g := by
  suffices f.comp g = Matrix.toLin' (LinearMap.toMatrix' f * LinearMap.toMatrix' g) by
    rw [this, LinearMap.toMatrix'_toLin']
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    l : Type u_3
    m : Type u_4
    n : Type u_5
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    inst✝¹ : Fintype l
    inst✝ : DecidableEq l
    f : LinearMap (RingHom.id R) (n → R) (m → R)
    g : LinearMap (RingHom.id R) (l → R) (n → R)
    ⊢ Eq (f.comp g) (Matrix.toLin' (HMul.hMul (LinearMap.toMatrix' f) (LinearMap.t …
  -/
  rw [Matrix.toLin'_mul, Matrix.toLin'_toMatrix', Matrix.toLin'_toMatrix']
  /-
    🎉 no goals
  -/


theorem LinearMap.toMatrix'_mul [Fintype m] [DecidableEq m] (f g : (m → R) →ₗ[R] m → R) :
    LinearMap.toMatrix' (f * g) = LinearMap.toMatrix' f * LinearMap.toMatrix' g :=
  LinearMap.toMatrix'_comp f g


@[simp]
theorem LinearMap.toMatrix'_algebraMap (x : R) :
    LinearMap.toMatrix' (algebraMap R (Module.End R (n → R)) x) = scalar n x := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type u_5
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    x : R
    ⊢ Eq (LinearMap.toMatrix' ((algebraMap R (Module.End R (n → R))) x)) ((Matrix. …
  -/
  simp [Module.algebraMap_end_eq_smul_id, smul_eq_diagonal_mul]
  /-
    🎉 no goals
  -/


theorem Matrix.ker_toLin'_eq_bot_iff {M : Matrix n n R} :
    LinearMap.ker (Matrix.toLin' M) = ⊥ ↔ ∀ v, M *ᵥ v = 0 → v = 0 :=
  Matrix.ker_mulVecLin_eq_bot_iff


theorem Matrix.range_toLin' (M : Matrix m n R) :
    LinearMap.range (Matrix.toLin' M) = span R (range Mᵀ) :=
  Matrix.range_mulVecLin _


/-- If `M` and `M'` are each other's inverse matrices, they provide an equivalence between `m → A`
and `n → A` corresponding to `M.mulVec` and `M'.mulVec`. -/
@[simps]
def Matrix.toLin'OfInv [Fintype m] [DecidableEq m] {M : Matrix m n R} {M' : Matrix n m R}
    (hMM' : M * M' = 1) (hM'M : M' * M = 1) : (m → R) ≃ₗ[R] n → R :=
  { Matrix.toLin' M' with
    toFun := Matrix.toLin' M'
    invFun := Matrix.toLin' M
                           /-
                             R : Type u_1
                             inst✝⁴ : CommSemiring R
                             k : Type u_2
                             l : Type u_3
                             m : Type u_4
                             n : Type u_5
                             inst✝³ : DecidableEq n
                             inst✝² : Fintype n
                             inst✝¹ : Fintype m
                             inst✝ : DecidableEq m
                             M : Matrix m n R
                             M' : Matrix n m R
                             hMM' : Eq (HMul.hMul M M') 1
                             hM'M : Eq (HMul.hMul M' M) 1
                             x : m → R
                             ⊢ Eq ((Matrix.toLin' M) ({ toFun := ⇑(Matrix.toLin' M'), map_add' := ⋯, map_sm …
                           -/
    left_inv := fun x ↦ by rw [← Matrix.toLin'_mul_apply, hMM', Matrix.toLin'_one, id_apply]
                           /-
                             🎉 no goals
                           -/
    right_inv := fun x ↦ by
      /-
        R : Type u_1
        inst✝⁴ : CommSemiring R
        k : Type u_2
        l : Type u_3
        m : Type u_4
        n : Type u_5
        inst✝³ : DecidableEq n
        inst✝² : Fintype n
        inst✝¹ : Fintype m
        inst✝ : DecidableEq m
        M : Matrix m n R
        M' : Matrix n m R
        hMM' : Eq (HMul.hMul M M') 1
        hM'M : Eq (HMul.hMul M' M) 1
        x : n → R
        ⊢ Eq ({ toFun := ⇑(Matrix.toLin' M'), map_add' := ⋯, map_smul' := ⋯ }.toFun (( …
      -/
      simp only
      /-
        R : Type u_1
        inst✝⁴ : CommSemiring R
        k : Type u_2
        l : Type u_3
        m : Type u_4
        n : Type u_5
        inst✝³ : DecidableEq n
        inst✝² : Fintype n
        inst✝¹ : Fintype m
        inst✝ : DecidableEq m
        M : Matrix m n R
        M' : Matrix n m R
        hMM' : Eq (HMul.hMul M M') 1
        hM'M : Eq (HMul.hMul M' M) 1
        x : n → R
        ⊢ Eq ((Matrix.toLin' M') ((Matrix.toLin' M) x)) x
      -/
      rw [← Matrix.toLin'_mul_apply, hM'M, Matrix.toLin'_one, id_apply] }
      /-
        🎉 no goals
      -/


/-- Linear maps `(n → R) →ₗ[R] (n → R)` are algebra equivalent to `Matrix n n R`. -/
def LinearMap.toMatrixAlgEquiv' : ((n → R) →ₗ[R] n → R) ≃ₐ[R] Matrix n n R :=
  AlgEquiv.ofLinearEquiv LinearMap.toMatrix' LinearMap.toMatrix'_one LinearMap.toMatrix'_mul


/-- A `Matrix n n R` is algebra equivalent to a linear map `(n → R) →ₗ[R] (n → R)`. -/
def Matrix.toLinAlgEquiv' : Matrix n n R ≃ₐ[R] (n → R) →ₗ[R] n → R :=
  LinearMap.toMatrixAlgEquiv'.symm


@[simp]
theorem LinearMap.toMatrixAlgEquiv'_symm :
    (LinearMap.toMatrixAlgEquiv'.symm : Matrix n n R ≃ₐ[R] _) = Matrix.toLinAlgEquiv' :=
  rfl


@[simp]
theorem Matrix.toLinAlgEquiv'_symm :
    (Matrix.toLinAlgEquiv'.symm : ((n → R) →ₗ[R] n → R) ≃ₐ[R] _) = LinearMap.toMatrixAlgEquiv' :=
  rfl


@[simp]
theorem LinearMap.toMatrixAlgEquiv'_toLinAlgEquiv' (M : Matrix n n R) :
    LinearMap.toMatrixAlgEquiv' (Matrix.toLinAlgEquiv' M) = M :=
  LinearMap.toMatrixAlgEquiv'.apply_symm_apply M


@[simp]
theorem Matrix.toLinAlgEquiv'_toMatrixAlgEquiv' (f : (n → R) →ₗ[R] n → R) :
    Matrix.toLinAlgEquiv' (LinearMap.toMatrixAlgEquiv' f) = f :=
  Matrix.toLinAlgEquiv'.apply_symm_apply f


@[simp]
theorem LinearMap.toMatrixAlgEquiv'_apply (f : (n → R) →ₗ[R] n → R) (i j) :
    LinearMap.toMatrixAlgEquiv' f i j = f (fun j' ↦ if j' = j then 1 else 0) i := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    n : Type u_5
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    f : LinearMap (RingHom.id R) (n → R) (n → R)
    i j : n
    ⊢ Eq (LinearMap.toMatrixAlgEquiv' f i j) (f (fun j' => ite (Eq j' j) 1 0) i)
  -/
  simp [LinearMap.toMatrixAlgEquiv']
  /-
    🎉 no goals
  -/


@[simp]
theorem Matrix.toLinAlgEquiv'_apply (M : Matrix n n R) (v : n → R) :
    Matrix.toLinAlgEquiv' M v = M *ᵥ v :=
  rfl

-- Porting note: the simpNF linter rejects this, as `simp` already simplifies the lhs
-- to `(1 : (n → R) →ₗ[R] n → R)`.
-- @[simp]

theorem Matrix.toLinAlgEquiv'_one : Matrix.toLinAlgEquiv' (1 : Matrix n n R) = LinearMap.id :=
  Matrix.toLin'_one


@[simp]
theorem LinearMap.toMatrixAlgEquiv'_id :
    LinearMap.toMatrixAlgEquiv' (LinearMap.id : (n → R) →ₗ[R] n → R) = 1 :=
  LinearMap.toMatrix'_id


theorem LinearMap.toMatrixAlgEquiv'_comp (f g : (n → R) →ₗ[R] n → R) :
    LinearMap.toMatrixAlgEquiv' (f.comp g) =
      LinearMap.toMatrixAlgEquiv' f * LinearMap.toMatrixAlgEquiv' g :=
  LinearMap.toMatrix'_comp _ _


theorem LinearMap.toMatrixAlgEquiv'_mul (f g : (n → R) →ₗ[R] n → R) :
    LinearMap.toMatrixAlgEquiv' (f * g) =
      LinearMap.toMatrixAlgEquiv' f * LinearMap.toMatrixAlgEquiv' g :=
  LinearMap.toMatrixAlgEquiv'_comp f g


/-- Given bases of two modules `M₁` and `M₂` over a commutative ring `R`, we get a linear
equivalence between linear maps `M₁ →ₗ M₂` and matrices over `R` indexed by the bases. -/
def LinearMap.toMatrix : (M₁ →ₗ[R] M₂) ≃ₗ[R] Matrix m n R :=
  LinearEquiv.trans (LinearEquiv.arrowCongr v₁.equivFun v₂.equivFun) LinearMap.toMatrix'


/-- `LinearMap.toMatrix'` is a particular case of `LinearMap.toMatrix`, for the standard basis
`Pi.basisFun R n`. -/
theorem LinearMap.toMatrix_eq_toMatrix' :
    LinearMap.toMatrix (Pi.basisFun R n) (Pi.basisFun R n) = LinearMap.toMatrix' :=
  rfl


/-- Given bases of two modules `M₁` and `M₂` over a commutative ring `R`, we get a linear
equivalence between matrices over `R` indexed by the bases and linear maps `M₁ →ₗ M₂`. -/
def Matrix.toLin : Matrix m n R ≃ₗ[R] M₁ →ₗ[R] M₂ :=
  (LinearMap.toMatrix v₁ v₂).symm


/-- `Matrix.toLin'` is a particular case of `Matrix.toLin`, for the standard basis
`Pi.basisFun R n`. -/
theorem Matrix.toLin_eq_toLin' : Matrix.toLin (Pi.basisFun R n) (Pi.basisFun R m) = Matrix.toLin' :=
  rfl


@[simp]
theorem LinearMap.toMatrix_symm : (LinearMap.toMatrix v₁ v₂).symm = Matrix.toLin v₁ v₂ :=
  rfl


@[simp]
theorem Matrix.toLin_symm : (Matrix.toLin v₁ v₂).symm = LinearMap.toMatrix v₁ v₂ :=
  rfl


@[simp]
theorem Matrix.toLin_toMatrix (f : M₁ →ₗ[R] M₂) :
    Matrix.toLin v₁ v₂ (LinearMap.toMatrix v₁ v₂ f) = f := by
  /-
    R : Type u_1
    inst✝⁷ : CommSemiring R
    m : Type u_3
    n : Type u_4
    inst✝⁶ : Fintype n
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝³ : AddCommMonoid M₁
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    ⊢ Eq ((Matrix.toLin v₁ v₂) ((LinearMap.toMatrix v₁ v₂) f)) f
  -/
  rw [← Matrix.toLin_symm, LinearEquiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem LinearMap.toMatrix_toLin (M : Matrix m n R) :
    LinearMap.toMatrix v₁ v₂ (Matrix.toLin v₁ v₂ M) = M := by
  /-
    R : Type u_1
    inst✝⁷ : CommSemiring R
    m : Type u_3
    n : Type u_4
    inst✝⁶ : Fintype n
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝³ : AddCommMonoid M₁
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    M : Matrix m n R
    ⊢ Eq ((LinearMap.toMatrix v₁ v₂) ((Matrix.toLin v₁ v₂) M)) M
  -/
  rw [← Matrix.toLin_symm, LinearEquiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


theorem LinearMap.toMatrix_apply (f : M₁ →ₗ[R] M₂) (i : m) (j : n) :
    LinearMap.toMatrix v₁ v₂ f i j = v₂.repr (f (v₁ j)) i := by
  rw [LinearMap.toMatrix, LinearEquiv.trans_apply, LinearMap.toMatrix'_apply,
    LinearEquiv.arrowCongr_apply, Basis.equivFun_symm_apply, Finset.sum_eq_single j, if_pos rfl,
    one_smul, Basis.equivFun_apply]
    /-
      case h₀
      R : Type u_1
      inst✝⁷ : CommSemiring R
      m : Type u_3
      n : Type u_4
      inst✝⁶ : Fintype n
      inst✝⁵ : Finite m
      inst✝⁴ : DecidableEq n
      M₁ : Type u_5
      M₂ : Type u_6
      inst✝³ : AddCommMonoid M₁
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M₁
      inst✝ : Module R M₂
      v₁ : Basis n R M₁
      v₂ : Basis m R M₂
      f : LinearMap (RingHom.id R) M₁ M₂
      i : m
      j : n
      ⊢ ∀ (b : n), Membership.mem Finset.univ b → Ne b j → Eq (HSMul.hSMul (ite (Eq  …
    -/
  · intro j' _ hj'
    /-
      case h₀
      R : Type u_1
      inst✝⁷ : CommSemiring R
      m : Type u_3
      n : Type u_4
      inst✝⁶ : Fintype n
      inst✝⁵ : Finite m
      inst✝⁴ : DecidableEq n
      M₁ : Type u_5
      M₂ : Type u_6
      inst✝³ : AddCommMonoid M₁
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M₁
      inst✝ : Module R M₂
      v₁ : Basis n R M₁
      v₂ : Basis m R M₂
      f : LinearMap (RingHom.id R) M₁ M₂
      i : m
      j j' : n
      a✝ : Membership.mem Finset.univ j'
      hj' : Ne j' j
      ⊢ Eq (HSMul.hSMul (ite (Eq j' j) 1 0) (v₁ j')) 0
    -/
    rw [if_neg hj', zero_smul]
    /-
      🎉 no goals
    -/
    /-
      case h₁
      R : Type u_1
      inst✝⁷ : CommSemiring R
      m : Type u_3
      n : Type u_4
      inst✝⁶ : Fintype n
      inst✝⁵ : Finite m
      inst✝⁴ : DecidableEq n
      M₁ : Type u_5
      M₂ : Type u_6
      inst✝³ : AddCommMonoid M₁
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M₁
      inst✝ : Module R M₂
      v₁ : Basis n R M₁
      v₂ : Basis m R M₂
      f : LinearMap (RingHom.id R) M₁ M₂
      i : m
      j : n
      ⊢ Not (Membership.mem Finset.univ j) → Eq (HSMul.hSMul (ite (Eq j j) 1 0) (v₁  …
    -/
  · intro hj
    /-
      case h₁
      R : Type u_1
      inst✝⁷ : CommSemiring R
      m : Type u_3
      n : Type u_4
      inst✝⁶ : Fintype n
      inst✝⁵ : Finite m
      inst✝⁴ : DecidableEq n
      M₁ : Type u_5
      M₂ : Type u_6
      inst✝³ : AddCommMonoid M₁
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M₁
      inst✝ : Module R M₂
      v₁ : Basis n R M₁
      v₂ : Basis m R M₂
      f : LinearMap (RingHom.id R) M₁ M₂
      i : m
      j : n
      hj : Not (Membership.mem Finset.univ j)
      ⊢ Eq (HSMul.hSMul (ite (Eq j j) 1 0) (v₁ j)) 0
    -/
    have := Finset.mem_univ j
    /-
      case h₁
      R : Type u_1
      inst✝⁷ : CommSemiring R
      m : Type u_3
      n : Type u_4
      inst✝⁶ : Fintype n
      inst✝⁵ : Finite m
      inst✝⁴ : DecidableEq n
      M₁ : Type u_5
      M₂ : Type u_6
      inst✝³ : AddCommMonoid M₁
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M₁
      inst✝ : Module R M₂
      v₁ : Basis n R M₁
      v₂ : Basis m R M₂
      f : LinearMap (RingHom.id R) M₁ M₂
      i : m
      j : n
      hj : Not (Membership.mem Finset.univ j)
      this : Membership.mem Finset.univ j
      ⊢ Eq (HSMul.hSMul (ite (Eq j j) 1 0) (v₁ j)) 0
    -/
    contradiction
    /-
      🎉 no goals
    -/


theorem LinearMap.toMatrix_transpose_apply (f : M₁ →ₗ[R] M₂) (j : n) :
    (LinearMap.toMatrix v₁ v₂ f)ᵀ j = v₂.repr (f (v₁ j)) :=
  funext fun i ↦ f.toMatrix_apply _ _ i j


theorem LinearMap.toMatrix_apply' (f : M₁ →ₗ[R] M₂) (i : m) (j : n) :
    LinearMap.toMatrix v₁ v₂ f i j = v₂.repr (f (v₁ j)) i :=
  LinearMap.toMatrix_apply v₁ v₂ f i j


theorem LinearMap.toMatrix_transpose_apply' (f : M₁ →ₗ[R] M₂) (j : n) :
    (LinearMap.toMatrix v₁ v₂ f)ᵀ j = v₂.repr (f (v₁ j)) :=
  LinearMap.toMatrix_transpose_apply v₁ v₂ f j


/-- This will be a special case of `LinearMap.toMatrix_id_eq_basis_toMatrix`. -/
theorem LinearMap.toMatrix_id : LinearMap.toMatrix v₁ v₁ id = 1 := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    n : Type u_4
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    M₁ : Type u_5
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    v₁ : Basis n R M₁
    ⊢ Eq ((LinearMap.toMatrix v₁ v₁) LinearMap.id) 1
  -/
  ext i j
  /-
    case a
    R : Type u_1
    inst✝⁴ : CommSemiring R
    n : Type u_4
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    M₁ : Type u_5
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    v₁ : Basis n R M₁
    i j : n
    ⊢ Eq ((LinearMap.toMatrix v₁ v₁) LinearMap.id i j) (1 i j)
  -/
  simp [LinearMap.toMatrix_apply, Matrix.one_apply, Finsupp.single_apply, eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem LinearMap.toMatrix_one : LinearMap.toMatrix v₁ v₁ 1 = 1 :=
  LinearMap.toMatrix_id v₁


@[simp]
lemma LinearMap.toMatrix_singleton {ι : Type*} [Unique ι] (f : R →ₗ[R] R) (i j : ι) :
    f.toMatrix (.singleton ι R) (.singleton ι R) i j = f 1 := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    ι : Type u_7
    inst✝ : Unique ι
    f : LinearMap (RingHom.id R) R R
    i j : ι
    ⊢ Eq ((LinearMap.toMatrix (Basis.singleton ι R) (Basis.singleton ι R)) f i j)  …
  -/
  simp [toMatrix, Subsingleton.elim j default]
  /-
    🎉 no goals
  -/


@[simp]
theorem Matrix.toLin_one : Matrix.toLin v₁ v₁ 1 = LinearMap.id := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    n : Type u_4
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    M₁ : Type u_5
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    v₁ : Basis n R M₁
    ⊢ Eq ((Matrix.toLin v₁ v₁) 1) LinearMap.id
  -/
  rw [← LinearMap.toMatrix_id v₁, Matrix.toLin_toMatrix]
  /-
    🎉 no goals
  -/


theorem LinearMap.toMatrix_reindexRange [DecidableEq M₁] (f : M₁ →ₗ[R] M₂) (k : m) (i : n) :
    LinearMap.toMatrix v₁.reindexRange v₂.reindexRange f ⟨v₂ k, Set.mem_range_self k⟩
        ⟨v₁ i, Set.mem_range_self i⟩ =
      LinearMap.toMatrix v₁ v₂ f k i := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    m : Type u_3
    n : Type u_4
    inst✝⁷ : Fintype n
    inst✝⁶ : Finite m
    inst✝⁵ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    inst✝ : DecidableEq M₁
    f : LinearMap (RingHom.id R) M₁ M₂
    k : m
    i : n
    ⊢ Eq ((LinearMap.toMatrix v₁.reindexRange v₂.reindexRange) f ⟨v₂ k, ⋯⟩ ⟨v₁ i,  …
  -/
  simp_rw [LinearMap.toMatrix_apply, Basis.reindexRange_self, Basis.reindexRange_repr]
  /-
    🎉 no goals
  -/


@[simp]
theorem LinearMap.toMatrix_algebraMap (x : R) :
    LinearMap.toMatrix v₁ v₁ (algebraMap R (Module.End R M₁) x) = scalar n x := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    n : Type u_4
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    M₁ : Type u_5
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    v₁ : Basis n R M₁
    x : R
    ⊢ Eq ((LinearMap.toMatrix v₁ v₁) ((algebraMap R (Module.End R M₁)) x)) ((Matri …
  -/
  simp [Module.algebraMap_end_eq_smul_id, LinearMap.toMatrix_id, smul_eq_diagonal_mul]
  /-
    🎉 no goals
  -/


theorem LinearMap.toMatrix_mulVec_repr (f : M₁ →ₗ[R] M₂) (x : M₁) :
    LinearMap.toMatrix v₁ v₂ f *ᵥ v₁.repr x = v₂.repr (f x) := by
  /-
    R : Type u_1
    inst✝⁷ : CommSemiring R
    m : Type u_3
    n : Type u_4
    inst✝⁶ : Fintype n
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝³ : AddCommMonoid M₁
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    x : M₁
    ⊢ Eq (((LinearMap.toMatrix v₁ v₂) f).mulVec ⇑(v₁.repr x)) ⇑(v₂.repr (f x))
  -/
  ext i
  rw [← Matrix.toLin'_apply, LinearMap.toMatrix, LinearEquiv.trans_apply, Matrix.toLin'_toMatrix',
    LinearEquiv.arrowCongr_apply, v₂.equivFun_apply]
  /-
    case h
    R : Type u_1
    inst✝⁷ : CommSemiring R
    m : Type u_3
    n : Type u_4
    inst✝⁶ : Fintype n
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝³ : AddCommMonoid M₁
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    x : M₁
    i : m
    ⊢ Eq ((v₂.repr (f (v₁.equivFun.symm ⇑(v₁.repr x)))) i) ((v₂.repr (f x)) i)
  -/
  congr
  /-
    case h.e_a.h.e_6.h.h.e_6.h
    R : Type u_1
    inst✝⁷ : CommSemiring R
    m : Type u_3
    n : Type u_4
    inst✝⁶ : Fintype n
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝³ : AddCommMonoid M₁
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    x : M₁
    i : m
    ⊢ Eq (v₁.equivFun.symm ⇑(v₁.repr x)) x
  -/
  exact v₁.equivFun.symm_apply_apply x
  /-
    🎉 no goals
  -/


@[simp]
theorem LinearMap.toMatrix_basis_equiv [Fintype l] [DecidableEq l] (b : Basis l R M₁)
    (b' : Basis l R M₂) :
    LinearMap.toMatrix b' b (b'.equiv b (Equiv.refl l) : M₂ →ₗ[R] M₁) = 1 := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    l : Type u_2
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R M₁
    inst✝² : Module R M₂
    inst✝¹ : Fintype l
    inst✝ : DecidableEq l
    b : Basis l R M₁
    b' : Basis l R M₂
    ⊢ Eq ((LinearMap.toMatrix b' b) ↑(b'.equiv b (Equiv.refl l))) 1
  -/
  ext i j
  /-
    case a
    R : Type u_1
    inst✝⁶ : CommSemiring R
    l : Type u_2
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R M₁
    inst✝² : Module R M₂
    inst✝¹ : Fintype l
    inst✝ : DecidableEq l
    b : Basis l R M₁
    b' : Basis l R M₂
    i j : l
    ⊢ Eq ((LinearMap.toMatrix b' b) (↑(b'.equiv b (Equiv.refl l))) i j) (1 i j)
  -/
  simp [LinearMap.toMatrix_apply, Matrix.one_apply, Finsupp.single_apply, eq_comm]
  /-
    🎉 no goals
  -/


theorem LinearMap.toMatrix_smulBasis_left {G} [Group G] [DistribMulAction G M₁]
    [SMulCommClass G R M₁] (g : G) (f : M₁ →ₗ[R] M₂) :
    LinearMap.toMatrix (g • v₁) v₂ f =
      LinearMap.toMatrix v₁ v₂ (f ∘ₗ DistribMulAction.toLinearMap _ _ g) := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    m : Type u_3
    n : Type u_4
    inst✝⁹ : Fintype n
    inst✝⁸ : Finite m
    inst✝⁷ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁶ : AddCommMonoid M₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    G : Type u_7
    inst✝² : Group G
    inst✝¹ : DistribMulAction G M₁
    inst✝ : SMulCommClass G R M₁
    g : G
    f : LinearMap (RingHom.id R) M₁ M₂
    ⊢ Eq ((LinearMap.toMatrix (HSMul.hSMul g v₁) v₂) f) ((LinearMap.toMatrix v₁ v₂ …
  -/
  ext
  /-
    case a
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    m : Type u_3
    n : Type u_4
    inst✝⁹ : Fintype n
    inst✝⁸ : Finite m
    inst✝⁷ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁶ : AddCommMonoid M₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    G : Type u_7
    inst✝² : Group G
    inst✝¹ : DistribMulAction G M₁
    inst✝ : SMulCommClass G R M₁
    g : G
    f : LinearMap (RingHom.id R) M₁ M₂
    i✝ : m
    j✝ : n
    ⊢ Eq ((LinearMap.toMatrix (HSMul.hSMul g v₁) v₂) f i✝ j✝) ((LinearMap.toMatrix …
  -/
  rw [LinearMap.toMatrix_apply, LinearMap.toMatrix_apply]
  /-
    case a
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    m : Type u_3
    n : Type u_4
    inst✝⁹ : Fintype n
    inst✝⁸ : Finite m
    inst✝⁷ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁶ : AddCommMonoid M₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    G : Type u_7
    inst✝² : Group G
    inst✝¹ : DistribMulAction G M₁
    inst✝ : SMulCommClass G R M₁
    g : G
    f : LinearMap (RingHom.id R) M₁ M₂
    i✝ : m
    j✝ : n
    ⊢ Eq ((v₂.repr (f ((HSMul.hSMul g v₁) j✝))) i✝) ((v₂.repr ((f.comp (DistribMul …
  -/
  dsimp
  /-
    🎉 no goals
  -/


theorem LinearMap.toMatrix_smulBasis_right {G} [Group G] [DistribMulAction G M₂]
    [SMulCommClass G R M₂] (g : G) (f : M₁ →ₗ[R] M₂) :
    LinearMap.toMatrix v₁ (g • v₂) f =
      LinearMap.toMatrix v₁ v₂ (DistribMulAction.toLinearMap _ _ g⁻¹ ∘ₗ f) := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    m : Type u_3
    n : Type u_4
    inst✝⁹ : Fintype n
    inst✝⁸ : Finite m
    inst✝⁷ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁶ : AddCommMonoid M₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    G : Type u_7
    inst✝² : Group G
    inst✝¹ : DistribMulAction G M₂
    inst✝ : SMulCommClass G R M₂
    g : G
    f : LinearMap (RingHom.id R) M₁ M₂
    ⊢ Eq ((LinearMap.toMatrix v₁ (HSMul.hSMul g v₂)) f) ((LinearMap.toMatrix v₁ v₂ …
  -/
  ext
  /-
    case a
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    m : Type u_3
    n : Type u_4
    inst✝⁹ : Fintype n
    inst✝⁸ : Finite m
    inst✝⁷ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁶ : AddCommMonoid M₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    G : Type u_7
    inst✝² : Group G
    inst✝¹ : DistribMulAction G M₂
    inst✝ : SMulCommClass G R M₂
    g : G
    f : LinearMap (RingHom.id R) M₁ M₂
    i✝ : m
    j✝ : n
    ⊢ Eq ((LinearMap.toMatrix v₁ (HSMul.hSMul g v₂)) f i✝ j✝) ((LinearMap.toMatrix …
  -/
  rw [LinearMap.toMatrix_apply, LinearMap.toMatrix_apply]
  /-
    case a
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    m : Type u_3
    n : Type u_4
    inst✝⁹ : Fintype n
    inst✝⁸ : Finite m
    inst✝⁷ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁶ : AddCommMonoid M₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    G : Type u_7
    inst✝² : Group G
    inst✝¹ : DistribMulAction G M₂
    inst✝ : SMulCommClass G R M₂
    g : G
    f : LinearMap (RingHom.id R) M₁ M₂
    i✝ : m
    j✝ : n
    ⊢ Eq (((HSMul.hSMul g v₂).repr (f (v₁ j✝))) i✝) ((v₂.repr (((DistribMulAction. …
  -/
  dsimp
  /-
    🎉 no goals
  -/


theorem Matrix.toLin_apply (M : Matrix m n R) (v : M₁) :
    Matrix.toLin v₁ v₂ M v = ∑ j, (M *ᵥ v₁.repr v) j • v₂ j :=
  show v₂.equivFun.symm (Matrix.toLin' M (v₁.repr v)) = _ by
    /-
      R : Type u_1
      inst✝⁷ : CommSemiring R
      m : Type u_3
      n : Type u_4
      inst✝⁶ : Fintype n
      inst✝⁵ : Fintype m
      inst✝⁴ : DecidableEq n
      M₁ : Type u_5
      M₂ : Type u_6
      inst✝³ : AddCommMonoid M₁
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M₁
      inst✝ : Module R M₂
      v₁ : Basis n R M₁
      v₂ : Basis m R M₂
      M : Matrix m n R
      v : M₁
      ⊢ Eq (v₂.equivFun.symm ((Matrix.toLin' M) ⇑(v₁.repr v))) (Finset.univ.sum fun  …
    -/
    rw [Matrix.toLin'_apply, v₂.equivFun_symm_apply]
    /-
      🎉 no goals
    -/


@[simp]
theorem Matrix.toLin_self (M : Matrix m n R) (i : n) :
    Matrix.toLin v₁ v₂ M (v₁ i) = ∑ j, M j i • v₂ j := by
  /-
    R : Type u_1
    inst✝⁷ : CommSemiring R
    m : Type u_3
    n : Type u_4
    inst✝⁶ : Fintype n
    inst✝⁵ : Fintype m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝³ : AddCommMonoid M₁
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    M : Matrix m n R
    i : n
    ⊢ Eq (((Matrix.toLin v₁ v₂) M) (v₁ i)) (Finset.univ.sum fun j => HSMul.hSMul ( …
  -/
  rw [Matrix.toLin_apply, Finset.sum_congr rfl fun j _hj ↦ ?_]
  rw [Basis.repr_self, Matrix.mulVec, dotProduct, Finset.sum_eq_single i, Finsupp.single_eq_same,
    mul_one]
    /-
      case h₀
      R : Type u_1
      inst✝⁷ : CommSemiring R
      m : Type u_3
      n : Type u_4
      inst✝⁶ : Fintype n
      inst✝⁵ : Fintype m
      inst✝⁴ : DecidableEq n
      M₁ : Type u_5
      M₂ : Type u_6
      inst✝³ : AddCommMonoid M₁
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M₁
      inst✝ : Module R M₂
      v₁ : Basis n R M₁
      v₂ : Basis m R M₂
      M : Matrix m n R
      i : n
      j : m
      _hj : Membership.mem Finset.univ j
      ⊢ ∀ (b : n), Membership.mem Finset.univ b → Ne b i → Eq (HMul.hMul (M j b) ((F …
    -/
  · intro i' _ i'_ne
    /-
      case h₀
      R : Type u_1
      inst✝⁷ : CommSemiring R
      m : Type u_3
      n : Type u_4
      inst✝⁶ : Fintype n
      inst✝⁵ : Fintype m
      inst✝⁴ : DecidableEq n
      M₁ : Type u_5
      M₂ : Type u_6
      inst✝³ : AddCommMonoid M₁
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M₁
      inst✝ : Module R M₂
      v₁ : Basis n R M₁
      v₂ : Basis m R M₂
      M : Matrix m n R
      i : n
      j : m
      _hj : Membership.mem Finset.univ j
      i' : n
      a✝ : Membership.mem Finset.univ i'
      i'_ne : Ne i' i
      ⊢ Eq (HMul.hMul (M j i') ((Finsupp.single i 1) i')) 0
    -/
    rw [Finsupp.single_eq_of_ne i'_ne.symm, mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case h₁
      R : Type u_1
      inst✝⁷ : CommSemiring R
      m : Type u_3
      n : Type u_4
      inst✝⁶ : Fintype n
      inst✝⁵ : Fintype m
      inst✝⁴ : DecidableEq n
      M₁ : Type u_5
      M₂ : Type u_6
      inst✝³ : AddCommMonoid M₁
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M₁
      inst✝ : Module R M₂
      v₁ : Basis n R M₁
      v₂ : Basis m R M₂
      M : Matrix m n R
      i : n
      j : m
      _hj : Membership.mem Finset.univ j
      ⊢ Not (Membership.mem Finset.univ i) → Eq (HMul.hMul (M j i) ((Finsupp.single  …
    -/
  · intros
    /-
      case h₁
      R : Type u_1
      inst✝⁷ : CommSemiring R
      m : Type u_3
      n : Type u_4
      inst✝⁶ : Fintype n
      inst✝⁵ : Fintype m
      inst✝⁴ : DecidableEq n
      M₁ : Type u_5
      M₂ : Type u_6
      inst✝³ : AddCommMonoid M₁
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M₁
      inst✝ : Module R M₂
      v₁ : Basis n R M₁
      v₂ : Basis m R M₂
      M : Matrix m n R
      i : n
      j : m
      _hj : Membership.mem Finset.univ j
      a✝ : Not (Membership.mem Finset.univ i)
      ⊢ Eq (HMul.hMul (M j i) ((Finsupp.single i 1) i)) 0
    -/
    have := Finset.mem_univ i
    /-
      case h₁
      R : Type u_1
      inst✝⁷ : CommSemiring R
      m : Type u_3
      n : Type u_4
      inst✝⁶ : Fintype n
      inst✝⁵ : Fintype m
      inst✝⁴ : DecidableEq n
      M₁ : Type u_5
      M₂ : Type u_6
      inst✝³ : AddCommMonoid M₁
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M₁
      inst✝ : Module R M₂
      v₁ : Basis n R M₁
      v₂ : Basis m R M₂
      M : Matrix m n R
      i : n
      j : m
      _hj : Membership.mem Finset.univ j
      a✝ : Not (Membership.mem Finset.univ i)
      this : Membership.mem Finset.univ i
      ⊢ Eq (HMul.hMul (M j i) ((Finsupp.single i 1) i)) 0
    -/
    contradiction
    /-
      🎉 no goals
    -/


theorem LinearMap.toMatrix_comp [Finite l] [DecidableEq m] (f : M₂ →ₗ[R] M₃) (g : M₁ →ₗ[R] M₂) :
    LinearMap.toMatrix v₁ v₃ (f.comp g) =
    LinearMap.toMatrix v₂ v₃ f * LinearMap.toMatrix v₁ v₂ g := by
  simp_rw [LinearMap.toMatrix, LinearEquiv.trans_apply, LinearEquiv.arrowCongr_comp _ v₂.equivFun,
    LinearMap.toMatrix'_comp]


theorem LinearMap.toMatrix_mul (f g : M₁ →ₗ[R] M₁) :
    LinearMap.toMatrix v₁ v₁ (f * g) = LinearMap.toMatrix v₁ v₁ f * LinearMap.toMatrix v₁ v₁ g := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    n : Type u_4
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    M₁ : Type u_5
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    v₁ : Basis n R M₁
    f g : LinearMap (RingHom.id R) M₁ M₁
    ⊢ Eq ((LinearMap.toMatrix v₁ v₁) (HMul.hMul f g)) (HMul.hMul ((LinearMap.toMat …
  -/
  rw [LinearMap.mul_eq_comp, LinearMap.toMatrix_comp v₁ v₁ v₁ f g]
  /-
    🎉 no goals
  -/


lemma LinearMap.toMatrix_pow (f : M₁ →ₗ[R] M₁) (k : ℕ) :
    (toMatrix v₁ v₁ f) ^ k = toMatrix v₁ v₁ (f ^ k) := by
  induction k with
  | zero => simp
  | succ k ih => rw [pow_succ, pow_succ, ih, ← toMatrix_mul]


theorem Matrix.toLin_mul [Finite l] [DecidableEq m] (A : Matrix l m R) (B : Matrix m n R) :
    Matrix.toLin v₁ v₃ (A * B) = (Matrix.toLin v₂ v₃ A).comp (Matrix.toLin v₁ v₂ B) := by
  /-
    R : Type u_1
    inst✝¹¹ : CommSemiring R
    l : Type u_2
    m : Type u_3
    n : Type u_4
    inst✝¹⁰ : Fintype n
    inst✝⁹ : Fintype m
    inst✝⁸ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : AddCommMonoid M₂
    inst✝⁵ : Module R M₁
    inst✝⁴ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    M₃ : Type u_7
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M₃
    v₃ : Basis l R M₃
    inst✝¹ : Finite l
    inst✝ : DecidableEq m
    A : Matrix l m R
    B : Matrix m n R
    ⊢ Eq ((Matrix.toLin v₁ v₃) (HMul.hMul A B)) (((Matrix.toLin v₂ v₃) A).comp ((M …
  -/
  apply (LinearMap.toMatrix v₁ v₃).injective
  /-
    case a
    R : Type u_1
    inst✝¹¹ : CommSemiring R
    l : Type u_2
    m : Type u_3
    n : Type u_4
    inst✝¹⁰ : Fintype n
    inst✝⁹ : Fintype m
    inst✝⁸ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : AddCommMonoid M₂
    inst✝⁵ : Module R M₁
    inst✝⁴ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    M₃ : Type u_7
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M₃
    v₃ : Basis l R M₃
    inst✝¹ : Finite l
    inst✝ : DecidableEq m
    A : Matrix l m R
    B : Matrix m n R
    ⊢ Eq ((LinearMap.toMatrix v₁ v₃) ((Matrix.toLin v₁ v₃) (HMul.hMul A B))) ((Lin …
  -/
  haveI : DecidableEq l := fun _ _ ↦ Classical.propDecidable _
  /-
    case a
    R : Type u_1
    inst✝¹¹ : CommSemiring R
    l : Type u_2
    m : Type u_3
    n : Type u_4
    inst✝¹⁰ : Fintype n
    inst✝⁹ : Fintype m
    inst✝⁸ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : AddCommMonoid M₂
    inst✝⁵ : Module R M₁
    inst✝⁴ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    M₃ : Type u_7
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M₃
    v₃ : Basis l R M₃
    inst✝¹ : Finite l
    inst✝ : DecidableEq m
    A : Matrix l m R
    B : Matrix m n R
    this : DecidableEq l
    ⊢ Eq ((LinearMap.toMatrix v₁ v₃) ((Matrix.toLin v₁ v₃) (HMul.hMul A B))) ((Lin …
  -/
  rw [LinearMap.toMatrix_comp v₁ v₂ v₃]
  /-
    case a
    R : Type u_1
    inst✝¹¹ : CommSemiring R
    l : Type u_2
    m : Type u_3
    n : Type u_4
    inst✝¹⁰ : Fintype n
    inst✝⁹ : Fintype m
    inst✝⁸ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : AddCommMonoid M₂
    inst✝⁵ : Module R M₁
    inst✝⁴ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    M₃ : Type u_7
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M₃
    v₃ : Basis l R M₃
    inst✝¹ : Finite l
    inst✝ : DecidableEq m
    A : Matrix l m R
    B : Matrix m n R
    this : DecidableEq l
    ⊢ Eq ((LinearMap.toMatrix v₁ v₃) ((Matrix.toLin v₁ v₃) (HMul.hMul A B))) (HMul …
  -/
  repeat' rw [LinearMap.toMatrix_toLin]
  /-
    🎉 no goals
  -/


/-- Shortcut lemma for `Matrix.toLin_mul` and `LinearMap.comp_apply`. -/
theorem Matrix.toLin_mul_apply [Finite l] [DecidableEq m] (A : Matrix l m R) (B : Matrix m n R)
    (x) : Matrix.toLin v₁ v₃ (A * B) x = (Matrix.toLin v₂ v₃ A) (Matrix.toLin v₁ v₂ B x) := by
  /-
    R : Type u_1
    inst✝¹¹ : CommSemiring R
    l : Type u_2
    m : Type u_3
    n : Type u_4
    inst✝¹⁰ : Fintype n
    inst✝⁹ : Fintype m
    inst✝⁸ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : AddCommMonoid M₂
    inst✝⁵ : Module R M₁
    inst✝⁴ : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    M₃ : Type u_7
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M₃
    v₃ : Basis l R M₃
    inst✝¹ : Finite l
    inst✝ : DecidableEq m
    A : Matrix l m R
    B : Matrix m n R
    x : M₁
    ⊢ Eq (((Matrix.toLin v₁ v₃) (HMul.hMul A B)) x) (((Matrix.toLin v₂ v₃) A) (((M …
  -/
  rw [Matrix.toLin_mul v₁ v₂, LinearMap.comp_apply]
  /-
    🎉 no goals
  -/


/-- If `M` and `M` are each other's inverse matrices, `Matrix.toLin M` and `Matrix.toLin M'`
form a linear equivalence. -/
@[simps]
def Matrix.toLinOfInv [DecidableEq m] {M : Matrix m n R} {M' : Matrix n m R} (hMM' : M * M' = 1)
    (hM'M : M' * M = 1) : M₁ ≃ₗ[R] M₂ :=
  { Matrix.toLin v₁ v₂ M with
    toFun := Matrix.toLin v₁ v₂ M
    invFun := Matrix.toLin v₂ v₁ M'
                           /-
                             R : Type u_1
                             inst✝¹⁰ : CommSemiring R
                             l : Type u_2
                             m : Type u_3
                             n : Type u_4
                             inst✝⁹ : Fintype n
                             inst✝⁸ : Fintype m
                             inst✝⁷ : DecidableEq n
                             M₁ : Type u_5
                             M₂ : Type u_6
                             inst✝⁶ : AddCommMonoid M₁
                             inst✝⁵ : AddCommMonoid M₂
                             inst✝⁴ : Module R M₁
                             inst✝³ : Module R M₂
                             v₁ : Basis n R M₁
                             v₂ : Basis m R M₂
                             M₃ : Type u_7
                             inst✝² : AddCommMonoid M₃
                             inst✝¹ : Module R M₃
                             v₃ : Basis l R M₃
                             inst✝ : DecidableEq m
                             M : Matrix m n R
                             M' : Matrix n m R
                             hMM' : Eq (HMul.hMul M M') 1
                             hM'M : Eq (HMul.hMul M' M) 1
                             x : M₁
                             ⊢ Eq (((Matrix.toLin v₂ v₁) M') ({ toFun := ⇑((Matrix.toLin v₁ v₂) M), map_add …
                           -/
    left_inv := fun x ↦ by rw [← Matrix.toLin_mul_apply, hM'M, Matrix.toLin_one, id_apply]
                           /-
                             🎉 no goals
                           -/
    right_inv := fun x ↦ by
      /-
        R : Type u_1
        inst✝¹⁰ : CommSemiring R
        l : Type u_2
        m : Type u_3
        n : Type u_4
        inst✝⁹ : Fintype n
        inst✝⁸ : Fintype m
        inst✝⁷ : DecidableEq n
        M₁ : Type u_5
        M₂ : Type u_6
        inst✝⁶ : AddCommMonoid M₁
        inst✝⁵ : AddCommMonoid M₂
        inst✝⁴ : Module R M₁
        inst✝³ : Module R M₂
        v₁ : Basis n R M₁
        v₂ : Basis m R M₂
        M₃ : Type u_7
        inst✝² : AddCommMonoid M₃
        inst✝¹ : Module R M₃
        v₃ : Basis l R M₃
        inst✝ : DecidableEq m
        M : Matrix m n R
        M' : Matrix n m R
        hMM' : Eq (HMul.hMul M M') 1
        hM'M : Eq (HMul.hMul M' M) 1
        x : M₂
        ⊢ Eq ({ toFun := ⇑((Matrix.toLin v₁ v₂) M), map_add' := ⋯, map_smul' := ⋯ }.to …
      -/
      simp only
      /-
        R : Type u_1
        inst✝¹⁰ : CommSemiring R
        l : Type u_2
        m : Type u_3
        n : Type u_4
        inst✝⁹ : Fintype n
        inst✝⁸ : Fintype m
        inst✝⁷ : DecidableEq n
        M₁ : Type u_5
        M₂ : Type u_6
        inst✝⁶ : AddCommMonoid M₁
        inst✝⁵ : AddCommMonoid M₂
        inst✝⁴ : Module R M₁
        inst✝³ : Module R M₂
        v₁ : Basis n R M₁
        v₂ : Basis m R M₂
        M₃ : Type u_7
        inst✝² : AddCommMonoid M₃
        inst✝¹ : Module R M₃
        v₃ : Basis l R M₃
        inst✝ : DecidableEq m
        M : Matrix m n R
        M' : Matrix n m R
        hMM' : Eq (HMul.hMul M M') 1
        hM'M : Eq (HMul.hMul M' M) 1
        x : M₂
        ⊢ Eq (((Matrix.toLin v₁ v₂) M) (((Matrix.toLin v₂ v₁) M') x)) x
      -/
      rw [← Matrix.toLin_mul_apply, hMM', Matrix.toLin_one, id_apply] }
      /-
        🎉 no goals
      -/


/-- Given a basis of a module `M₁` over a commutative ring `R`, we get an algebra
equivalence between linear maps `M₁ →ₗ M₁` and square matrices over `R` indexed by the basis. -/
def LinearMap.toMatrixAlgEquiv : (M₁ →ₗ[R] M₁) ≃ₐ[R] Matrix n n R :=
  AlgEquiv.ofLinearEquiv
    (LinearMap.toMatrix v₁ v₁) (LinearMap.toMatrix_one v₁) (LinearMap.toMatrix_mul v₁)


/-- Given a basis of a module `M₁` over a commutative ring `R`, we get an algebra
equivalence between square matrices over `R` indexed by the basis and linear maps `M₁ →ₗ M₁`. -/
def Matrix.toLinAlgEquiv : Matrix n n R ≃ₐ[R] M₁ →ₗ[R] M₁ :=
  (LinearMap.toMatrixAlgEquiv v₁).symm


@[simp]
theorem LinearMap.toMatrixAlgEquiv_symm :
    (LinearMap.toMatrixAlgEquiv v₁).symm = Matrix.toLinAlgEquiv v₁ :=
  rfl


@[simp]
theorem Matrix.toLinAlgEquiv_symm :
    (Matrix.toLinAlgEquiv v₁).symm = LinearMap.toMatrixAlgEquiv v₁ :=
  rfl


@[simp]
theorem Matrix.toLinAlgEquiv_toMatrixAlgEquiv (f : M₁ →ₗ[R] M₁) :
    Matrix.toLinAlgEquiv v₁ (LinearMap.toMatrixAlgEquiv v₁ f) = f := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    n : Type u_4
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    M₁ : Type u_5
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    v₁ : Basis n R M₁
    f : LinearMap (RingHom.id R) M₁ M₁
    ⊢ Eq ((Matrix.toLinAlgEquiv v₁) ((LinearMap.toMatrixAlgEquiv v₁) f)) f
  -/
  rw [← Matrix.toLinAlgEquiv_symm, AlgEquiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem LinearMap.toMatrixAlgEquiv_toLinAlgEquiv (M : Matrix n n R) :
    LinearMap.toMatrixAlgEquiv v₁ (Matrix.toLinAlgEquiv v₁ M) = M := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    n : Type u_4
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    M₁ : Type u_5
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    v₁ : Basis n R M₁
    M : Matrix n n R
    ⊢ Eq ((LinearMap.toMatrixAlgEquiv v₁) ((Matrix.toLinAlgEquiv v₁) M)) M
  -/
  rw [← Matrix.toLinAlgEquiv_symm, AlgEquiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


theorem LinearMap.toMatrixAlgEquiv_apply (f : M₁ →ₗ[R] M₁) (i j : n) :
    LinearMap.toMatrixAlgEquiv v₁ f i j = v₁.repr (f (v₁ j)) i := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    n : Type u_4
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    M₁ : Type u_5
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    v₁ : Basis n R M₁
    f : LinearMap (RingHom.id R) M₁ M₁
    i j : n
    ⊢ Eq ((LinearMap.toMatrixAlgEquiv v₁) f i j) ((v₁.repr (f (v₁ j))) i)
  -/
  simp [LinearMap.toMatrixAlgEquiv, LinearMap.toMatrix_apply]
  /-
    🎉 no goals
  -/


theorem LinearMap.toMatrixAlgEquiv_transpose_apply (f : M₁ →ₗ[R] M₁) (j : n) :
    (LinearMap.toMatrixAlgEquiv v₁ f)ᵀ j = v₁.repr (f (v₁ j)) :=
  funext fun i ↦ f.toMatrix_apply _ _ i j


theorem LinearMap.toMatrixAlgEquiv_apply' (f : M₁ →ₗ[R] M₁) (i j : n) :
    LinearMap.toMatrixAlgEquiv v₁ f i j = v₁.repr (f (v₁ j)) i :=
  LinearMap.toMatrixAlgEquiv_apply v₁ f i j


theorem LinearMap.toMatrixAlgEquiv_transpose_apply' (f : M₁ →ₗ[R] M₁) (j : n) :
    (LinearMap.toMatrixAlgEquiv v₁ f)ᵀ j = v₁.repr (f (v₁ j)) :=
  LinearMap.toMatrixAlgEquiv_transpose_apply v₁ f j


theorem Matrix.toLinAlgEquiv_apply (M : Matrix n n R) (v : M₁) :
    Matrix.toLinAlgEquiv v₁ M v = ∑ j, (M *ᵥ v₁.repr v) j • v₁ j :=
  show v₁.equivFun.symm (Matrix.toLinAlgEquiv' M (v₁.repr v)) = _ by
    /-
      R : Type u_1
      inst✝⁴ : CommSemiring R
      n : Type u_4
      inst✝³ : Fintype n
      inst✝² : DecidableEq n
      M₁ : Type u_5
      inst✝¹ : AddCommMonoid M₁
      inst✝ : Module R M₁
      v₁ : Basis n R M₁
      M : Matrix n n R
      v : M₁
      ⊢ Eq (v₁.equivFun.symm ((Matrix.toLinAlgEquiv' M) ⇑(v₁.repr v))) (Finset.univ. …
    -/
    rw [Matrix.toLinAlgEquiv'_apply, v₁.equivFun_symm_apply]
    /-
      🎉 no goals
    -/


@[simp]
theorem Matrix.toLinAlgEquiv_self (M : Matrix n n R) (i : n) :
    Matrix.toLinAlgEquiv v₁ M (v₁ i) = ∑ j, M j i • v₁ j :=
  Matrix.toLin_self _ _ _ _


theorem LinearMap.toMatrixAlgEquiv_id : LinearMap.toMatrixAlgEquiv v₁ id = 1 := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    n : Type u_4
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    M₁ : Type u_5
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    v₁ : Basis n R M₁
    ⊢ Eq ((LinearMap.toMatrixAlgEquiv v₁) LinearMap.id) 1
  -/
  simp_rw [LinearMap.toMatrixAlgEquiv, AlgEquiv.ofLinearEquiv_apply, LinearMap.toMatrix_id]
  /-
    🎉 no goals
  -/

-- Porting note: the simpNF linter rejects this, as `simp` already simplifies the lhs
-- to `(1 : M₁ →ₗ[R] M₁)`.
-- @[simp]

theorem Matrix.toLinAlgEquiv_one : Matrix.toLinAlgEquiv v₁ 1 = LinearMap.id := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    n : Type u_4
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    M₁ : Type u_5
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    v₁ : Basis n R M₁
    ⊢ Eq ((Matrix.toLinAlgEquiv v₁) 1) LinearMap.id
  -/
  rw [← LinearMap.toMatrixAlgEquiv_id v₁, Matrix.toLinAlgEquiv_toMatrixAlgEquiv]
  /-
    🎉 no goals
  -/


theorem LinearMap.toMatrixAlgEquiv_reindexRange [DecidableEq M₁] (f : M₁ →ₗ[R] M₁) (k i : n) :
    LinearMap.toMatrixAlgEquiv v₁.reindexRange f
        ⟨v₁ k, Set.mem_range_self k⟩ ⟨v₁ i, Set.mem_range_self i⟩ =
      LinearMap.toMatrixAlgEquiv v₁ f k i := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    n : Type u_4
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    M₁ : Type u_5
    inst✝² : AddCommMonoid M₁
    inst✝¹ : Module R M₁
    v₁ : Basis n R M₁
    inst✝ : DecidableEq M₁
    f : LinearMap (RingHom.id R) M₁ M₁
    k i : n
    ⊢ Eq ((LinearMap.toMatrixAlgEquiv v₁.reindexRange) f ⟨v₁ k, ⋯⟩ ⟨v₁ i, ⋯⟩) ((Li …
  -/
  simp_rw [LinearMap.toMatrixAlgEquiv_apply, Basis.reindexRange_self, Basis.reindexRange_repr]
  /-
    🎉 no goals
  -/


theorem LinearMap.toMatrixAlgEquiv_comp (f g : M₁ →ₗ[R] M₁) :
    LinearMap.toMatrixAlgEquiv v₁ (f.comp g) =
      LinearMap.toMatrixAlgEquiv v₁ f * LinearMap.toMatrixAlgEquiv v₁ g := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    n : Type u_4
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    M₁ : Type u_5
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    v₁ : Basis n R M₁
    f g : LinearMap (RingHom.id R) M₁ M₁
    ⊢ Eq ((LinearMap.toMatrixAlgEquiv v₁) (f.comp g)) (HMul.hMul ((LinearMap.toMat …
  -/
  simp [LinearMap.toMatrixAlgEquiv, LinearMap.toMatrix_comp v₁ v₁ v₁ f g]
  /-
    🎉 no goals
  -/


theorem LinearMap.toMatrixAlgEquiv_mul (f g : M₁ →ₗ[R] M₁) :
    LinearMap.toMatrixAlgEquiv v₁ (f * g) =
      LinearMap.toMatrixAlgEquiv v₁ f * LinearMap.toMatrixAlgEquiv v₁ g := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    n : Type u_4
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    M₁ : Type u_5
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    v₁ : Basis n R M₁
    f g : LinearMap (RingHom.id R) M₁ M₁
    ⊢ Eq ((LinearMap.toMatrixAlgEquiv v₁) (HMul.hMul f g)) (HMul.hMul ((LinearMap. …
  -/
  rw [LinearMap.mul_eq_comp, LinearMap.toMatrixAlgEquiv_comp v₁ f g]
  /-
    🎉 no goals
  -/


theorem Matrix.toLinAlgEquiv_mul (A B : Matrix n n R) :
    Matrix.toLinAlgEquiv v₁ (A * B) =
      (Matrix.toLinAlgEquiv v₁ A).comp (Matrix.toLinAlgEquiv v₁ B) := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    n : Type u_4
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    M₁ : Type u_5
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    v₁ : Basis n R M₁
    A B : Matrix n n R
    ⊢ Eq ((Matrix.toLinAlgEquiv v₁) (HMul.hMul A B)) (((Matrix.toLinAlgEquiv v₁) A …
  -/
  convert Matrix.toLin_mul v₁ v₁ v₁ A B
  /-
    🎉 no goals
  -/


@[simp]
theorem Matrix.toLin_finTwoProd_apply (a b c d : R) (x : R × R) :
    Matrix.toLin (Basis.finTwoProd R) (Basis.finTwoProd R) !![a, b; c, d] x =
      (a * x.fst + b * x.snd, c * x.fst + d * x.snd) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    a b c d : R
    x : Prod R R
    ⊢ Eq (((Matrix.toLin (Basis.finTwoProd R) (Basis.finTwoProd R)) (Matrix.of (Ma …
  -/
  simp [Matrix.toLin_apply, Matrix.mulVec, dotProduct]
  /-
    🎉 no goals
  -/


theorem Matrix.toLin_finTwoProd (a b c d : R) :
    Matrix.toLin (Basis.finTwoProd R) (Basis.finTwoProd R) !![a, b; c, d] =
      (a • LinearMap.fst R R R + b • LinearMap.snd R R R).prod
        (c • LinearMap.fst R R R + d • LinearMap.snd R R R) :=
  LinearMap.ext <| Matrix.toLin_finTwoProd_apply _ _ _ _


@[simp]
theorem toMatrix_distrib_mul_action_toLinearMap (x : R) :
    LinearMap.toMatrix v₁ v₁ (DistribMulAction.toLinearMap R M₁ x) =
    Matrix.diagonal fun _ ↦ x := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    n : Type u_4
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    M₁ : Type u_5
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    v₁ : Basis n R M₁
    x : R
    ⊢ Eq ((LinearMap.toMatrix v₁ v₁) (DistribMulAction.toLinearMap R M₁ x)) (Matri …
  -/
  ext
  rw [LinearMap.toMatrix_apply, DistribMulAction.toLinearMap_apply, LinearEquiv.map_smul,
    Basis.repr_self, Finsupp.smul_single_one, Finsupp.single_eq_pi_single, Matrix.diagonal_apply,
    Pi.single_apply]


lemma LinearMap.toMatrix_prodMap [DecidableEq m] [DecidableEq (n ⊕ m)]
    (φ₁ : Module.End R M₁) (φ₂ : Module.End R M₂) :
    toMatrix (v₁.prod v₂) (v₁.prod v₂) (φ₁.prodMap φ₂) =
      Matrix.fromBlocks (toMatrix v₁ v₁ φ₁) 0 0 (toMatrix v₂ v₂ φ₂) := by
  /-
    R : Type u_1
    inst✝⁹ : CommSemiring R
    m : Type u_3
    n : Type u_4
    inst✝⁸ : Fintype n
    inst✝⁷ : Fintype m
    inst✝⁶ : DecidableEq n
    M₁ : Type u_5
    M₂ : Type u_6
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R M₁
    inst✝² : Module R M₂
    v₁ : Basis n R M₁
    v₂ : Basis m R M₂
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq (Sum n m)
    φ₁ : Module.End R M₁
    φ₂ : Module.End R M₂
    ⊢ Eq ((LinearMap.toMatrix (v₁.prod v₂) (v₁.prod v₂)) (LinearMap.prodMap φ₁ φ₂) …
  -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
  ext (i|i) (j|j) <;> simp [toMatrix]
                      /-
                        🎉 no goals
                      -/


theorem toMatrix_lmul' (x : S) (i j) :
    LinearMap.toMatrix b b (lmul R S x) i j = b.repr (x * b j) i := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring S
    inst✝² : Algebra R S
    m : Type u_3
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    b : Basis m R S
    x : S
    i j : m
    ⊢ Eq ((LinearMap.toMatrix b b) ((Algebra.lmul R S) x) i j) ((b.repr (HMul.hMul …
  -/
  simp only [LinearMap.toMatrix_apply', coe_lmul_eq_mul, LinearMap.mul_apply']
  /-
    🎉 no goals
  -/


@[simp]
theorem toMatrix_lsmul (x : R) :
    LinearMap.toMatrix b b (Algebra.lsmul R R S x) = Matrix.diagonal fun _ ↦ x :=
  toMatrix_distrib_mul_action_toLinearMap b x


/-- `leftMulMatrix b x` is the matrix corresponding to the linear map `fun y ↦ x * y`.

`leftMulMatrix_eq_repr_mul` gives a formula for the entries of `leftMulMatrix`.

This definition is useful for doing (more) explicit computations with `LinearMap.mulLeft`,
such as the trace form or norm map for algebras.
-/
noncomputable def leftMulMatrix : S →ₐ[R] Matrix m m R where
  toFun x := LinearMap.toMatrix b b (Algebra.lmul R S x)
  map_zero' := by
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      m : Type u_3
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      b : Basis m R S
      ⊢ Eq ((↑{ toFun := fun x => (LinearMap.toMatrix b b) ((Algebra.lmul R S) x), m …
    -/
    dsimp only  -- porting node: needed due to new-style structures
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      m : Type u_3
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      b : Basis m R S
      ⊢ Eq ((LinearMap.toMatrix b b) ((Algebra.lmul R S) 0)) 0
    -/
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      m : Type u_3
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      b : Basis m R S
      ⊢ Eq ((fun x => (LinearMap.toMatrix b b) ((Algebra.lmul R S) x)) 1) 1
    -/
    rw [_root_.map_zero, LinearEquiv.map_zero]
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      m : Type u_3
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      b : Basis m R S
      ⊢ Eq ((LinearMap.toMatrix b b) ((Algebra.lmul R S) 1)) 1
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  map_one' := by
    dsimp only  -- porting node: needed due to new-style structures
    rw [_root_.map_one, LinearMap.toMatrix_one]
  map_add' x y := by
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      m : Type u_3
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      b : Basis m R S
      x y : S
      ⊢ Eq ({ toFun := fun x => (LinearMap.toMatrix b b) ((Algebra.lmul R S) x), map …
    -/
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      m : Type u_3
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      b : Basis m R S
      x y : S
      ⊢ Eq ((↑{ toFun := fun x => (LinearMap.toMatrix b b) ((Algebra.lmul R S) x), m …
    -/
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      m : Type u_3
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      b : Basis m R S
      x y : S
      ⊢ Eq ((LinearMap.toMatrix b b) ((Algebra.lmul R S) (HMul.hMul x y))) (HMul.hMu …
    -/
    dsimp only  -- porting node: needed due to new-style structures
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      m : Type u_3
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      b : Basis m R S
      x y : S
      ⊢ Eq ((LinearMap.toMatrix b b) ((Algebra.lmul R S) (HAdd.hAdd x y))) (HAdd.hAd …
    -/
    rw [map_add, LinearEquiv.map_add]
    /-
      🎉 no goals
    -/
  map_mul' x y := by
    dsimp only  -- porting node: needed due to new-style structures
    rw [_root_.map_mul, LinearMap.toMatrix_mul]
  commutes' r := by
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      m : Type u_3
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      b : Basis m R S
      r : R
      ⊢ Eq ((↑↑{ toFun := fun x => (LinearMap.toMatrix b b) ((Algebra.lmul R S) x),  …
    -/
    dsimp only  -- porting node: needed due to new-style structures
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      m : Type u_3
      inst✝¹ : Fintype m
      inst✝ : DecidableEq m
      b : Basis m R S
      r : R
      ⊢ Eq ((LinearMap.toMatrix b b) ((Algebra.lmul R S) ((algebraMap R S) r))) ((al …
    -/
    ext
    rw [lmul_algebraMap, toMatrix_lsmul, algebraMap_eq_diagonal, Pi.algebraMap_def,
      Algebra.id.map_eq_self]


theorem leftMulMatrix_apply (x : S) : leftMulMatrix b x = LinearMap.toMatrix b b (lmul R S x) :=
  rfl


theorem leftMulMatrix_eq_repr_mul (x : S) (i j) : leftMulMatrix b x i j = b.repr (x * b j) i := by
  -- This is defeq to just `toMatrix_lmul' b x i j`,
  -- but the unfolding goes a lot faster with this explicit `rw`.
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring S
    inst✝² : Algebra R S
    m : Type u_3
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    b : Basis m R S
    x : S
    i j : m
    ⊢ Eq ((Algebra.leftMulMatrix b) x i j) ((b.repr (HMul.hMul x (b j))) i)
  -/
  rw [leftMulMatrix_apply, toMatrix_lmul' b x i j]
  /-
    🎉 no goals
  -/


theorem leftMulMatrix_mulVec_repr (x y : S) :
    leftMulMatrix b x *ᵥ b.repr y = b.repr (x * y) :=
  (LinearMap.mulLeft R x).toMatrix_mulVec_repr b b y


@[simp]
theorem toMatrix_lmul_eq (x : S) :
    LinearMap.toMatrix b b (LinearMap.mulLeft R x) = leftMulMatrix b x :=
  rfl


theorem leftMulMatrix_injective : Function.Injective (leftMulMatrix b) := fun x x' h ↦
  calc
    x = Algebra.lmul R S x 1 := (mul_one x).symm
                                    /-
                                      R : Type u_1
                                      S : Type u_2
                                      inst✝⁴ : CommSemiring R
                                      inst✝³ : Semiring S
                                      inst✝² : Algebra R S
                                      m : Type u_3
                                      inst✝¹ : Fintype m
                                      inst✝ : DecidableEq m
                                      b : Basis m R S
                                      x x' : S
                                      h : Eq ((Algebra.leftMulMatrix b) x) ((Algebra.leftMulMatrix b) x')
                                      ⊢ Eq (((Algebra.lmul R S) x) 1) (((Algebra.lmul R S) x') 1)
                                    -/
    _ = Algebra.lmul R S x' 1 := by rw [(LinearMap.toMatrix b b).injective h]
                                    /-
                                      🎉 no goals
                                    -/
    _ = x' := mul_one x'


@[simp]
theorem smul_leftMulMatrix {G} [Group G] [DistribMulAction G S]
    [SMulCommClass G R S] [SMulCommClass G S S] (g : G) (x) :
    leftMulMatrix (g • b) x = leftMulMatrix b x := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Semiring S
    inst✝⁶ : Algebra R S
    m : Type u_3
    inst✝⁵ : Fintype m
    inst✝⁴ : DecidableEq m
    b : Basis m R S
    G : Type u_4
    inst✝³ : Group G
    inst✝² : DistribMulAction G S
    inst✝¹ : SMulCommClass G R S
    inst✝ : SMulCommClass G S S
    g : G
    x : S
    ⊢ Eq ((Algebra.leftMulMatrix (HSMul.hSMul g b)) x) ((Algebra.leftMulMatrix b) x)
  -/
  ext
  simp_rw [leftMulMatrix_apply, LinearMap.toMatrix_apply, coe_lmul_eq_mul, LinearMap.mul_apply',
    Basis.repr_smul, Basis.smul_apply, LinearEquiv.trans_apply,
    DistribMulAction.toLinearEquiv_symm_apply, mul_smul_comm, inv_smul_smul]


lemma _root_.LinearMap.restrictScalars_toMatrix (f : M →ₗ[A] M) :
    (f.restrictScalars R).toMatrix (bA.smulTower' bM) (bA.smulTower' bM) =
      ((f.toMatrix bM bM).map (leftMulMatrix bA)).comp _ _ _ _ _ := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    m : Type u_3
    inst✝⁹ : Fintype m
    inst✝⁸ : DecidableEq m
    A : Type u_4
    M : Type u_5
    n : Type u_6
    inst✝⁷ : Fintype n
    inst✝⁶ : DecidableEq n
    inst✝⁵ : CommSemiring A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module A M
    inst✝¹ : Algebra R A
    inst✝ : IsScalarTower R A M
    bA : Basis m R A
    bM : Basis n A M
    f : LinearMap (RingHom.id A) M M
    ⊢ Eq ((LinearMap.toMatrix (bA.smulTower' bM) (bA.smulTower' bM)) (↑R f)) ((Mat …
  -/
  ext; simp [toMatrix, Basis.repr, Algebra.leftMulMatrix_apply,
    Basis.smulTower'_repr, Basis.smulTower'_apply, mul_comm]


theorem smulTower_leftMulMatrix (x) (ik jk) :
    leftMulMatrix (b.smulTower c) x ik jk =
      leftMulMatrix b (leftMulMatrix c x ik.2 jk.2) ik.1 jk.1 := by
  simp only [leftMulMatrix_apply, LinearMap.toMatrix_apply, mul_comm, Basis.smulTower_apply,
    Basis.smulTower_repr, Finsupp.smul_apply, id.smul_eq_mul, LinearEquiv.map_smul, mul_smul_comm,
    coe_lmul_eq_mul, LinearMap.mul_apply']


theorem smulTower_leftMulMatrix_algebraMap (x : S) :
    leftMulMatrix (b.smulTower c) (algebraMap _ _ x) = blockDiagonal fun _ ↦ leftMulMatrix b x := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Semiring T
    inst✝⁷ : Algebra R S
    inst✝⁶ : Algebra S T
    inst✝⁵ : Algebra R T
    inst✝⁴ : IsScalarTower R S T
    m : Type u_4
    n : Type u_5
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    b : Basis m R S
    c : Basis n S T
    x : S
    ⊢ Eq ((Algebra.leftMulMatrix (b.smulTower c)) ((algebraMap S T) x)) (Matrix.bl …
  -/
  ext ⟨i, k⟩ ⟨j, k'⟩
  /-
    case a.mk.mk
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Semiring T
    inst✝⁷ : Algebra R S
    inst✝⁶ : Algebra S T
    inst✝⁵ : Algebra R T
    inst✝⁴ : IsScalarTower R S T
    m : Type u_4
    n : Type u_5
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    b : Basis m R S
    c : Basis n S T
    x : S
    i : m
    k : n
    j : m
    k' : n
    ⊢ Eq ((Algebra.leftMulMatrix (b.smulTower c)) ((algebraMap S T) x) { fst := i, …
  -/
  rw [smulTower_leftMulMatrix, AlgHom.commutes, blockDiagonal_apply, algebraMap_matrix_apply]
  /-
    case a.mk.mk
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Semiring T
    inst✝⁷ : Algebra R S
    inst✝⁶ : Algebra S T
    inst✝⁵ : Algebra R T
    inst✝⁴ : IsScalarTower R S T
    m : Type u_4
    n : Type u_5
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    b : Basis m R S
    c : Basis n S T
    x : S
    i : m
    k : n
    j : m
    k' : n
    ⊢ Eq ((Algebra.leftMulMatrix b) (ite (Eq { fst := i, snd := k }.2 { fst := j,  …
  -/
                                          /-
                                            🎉 no goals
                                          -/
  split_ifs with h <;> simp only at h <;> simp [h]
                                          /-
                                            🎉 no goals
                                          -/


theorem smulTower_leftMulMatrix_algebraMap_eq (x : S) (i j k) :
    leftMulMatrix (b.smulTower c) (algebraMap _ _ x) (i, k) (j, k) = leftMulMatrix b x i j := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Semiring T
    inst✝⁷ : Algebra R S
    inst✝⁶ : Algebra S T
    inst✝⁵ : Algebra R T
    inst✝⁴ : IsScalarTower R S T
    m : Type u_4
    n : Type u_5
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    b : Basis m R S
    c : Basis n S T
    x : S
    i j : m
    k : n
    ⊢ Eq ((Algebra.leftMulMatrix (b.smulTower c)) ((algebraMap S T) x) { fst := i, …
  -/
  rw [smulTower_leftMulMatrix_algebraMap, blockDiagonal_apply_eq]
  /-
    🎉 no goals
  -/


theorem smulTower_leftMulMatrix_algebraMap_ne (x : S) (i j) {k k'} (h : k ≠ k') :
    leftMulMatrix (b.smulTower c) (algebraMap _ _ x) (i, k) (j, k') = 0 := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Semiring T
    inst✝⁷ : Algebra R S
    inst✝⁶ : Algebra S T
    inst✝⁵ : Algebra R T
    inst✝⁴ : IsScalarTower R S T
    m : Type u_4
    n : Type u_5
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    b : Basis m R S
    c : Basis n S T
    x : S
    i j : m
    k k' : n
    h : Ne k k'
    ⊢ Eq ((Algebra.leftMulMatrix (b.smulTower c)) ((algebraMap S T) x) { fst := i, …
  -/
  rw [smulTower_leftMulMatrix_algebraMap, blockDiagonal_apply_ne _ _ _ h]
  /-
    🎉 no goals
  -/


/-- The natural equivalence between linear endomorphisms of finite free modules and square matrices
is compatible with the algebra structures. -/
def algEquivMatrix' [Fintype n] : Module.End R (n → R) ≃ₐ[R] Matrix n n R :=
  { LinearMap.toMatrix' with
    map_mul' := LinearMap.toMatrix'_comp
    commutes' := LinearMap.toMatrix'_algebraMap }


/-- A linear equivalence of two modules induces an equivalence of algebras of their
endomorphisms. -/
def LinearEquiv.algConj (e : M₁ ≃ₗ[R] M₂) : Module.End R M₁ ≃ₐ[R] Module.End R M₂ :=
  { e.conj with
                             /-
                               R : Type u_1
                               inst✝⁷ : CommSemiring R
                               n : Type u_2
                               inst✝⁶ : DecidableEq n
                               M : Type u_3
                               M₁ : Type u_4
                               M₂ : Type u_5
                               inst✝⁵ : AddCommMonoid M
                               inst✝⁴ : Module R M
                               inst✝³ : AddCommMonoid M₁
                               inst✝² : Module R M₁
                               inst✝¹ : AddCommMonoid M₂
                               inst✝ : Module R M₂
                               e : LinearEquiv (RingHom.id R) M₁ M₂
                               f g : Module.End R M₁
                               ⊢ Eq ({ toFun := (↑__src✝).toFun, invFun := __src✝.invFun, left_inv := ⋯, righ …
                             -/
    map_mul' := fun f g ↦ by apply e.arrowCongr_comp
                             /-
                               🎉 no goals
                             -/
    commutes' := fun r ↦ by
      /-
        R : Type u_1
        inst✝⁷ : CommSemiring R
        n : Type u_2
        inst✝⁶ : DecidableEq n
        M : Type u_3
        M₁ : Type u_4
        M₂ : Type u_5
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : Module R M
        inst✝³ : AddCommMonoid M₁
        inst✝² : Module R M₁
        inst✝¹ : AddCommMonoid M₂
        inst✝ : Module R M₂
        e : LinearEquiv (RingHom.id R) M₁ M₂
        r : R
        ⊢ Eq ({ toFun := (↑__src✝).toFun, invFun := __src✝.invFun, left_inv := ⋯, righ …
      -/
      change e.conj _ = _
      simp only [Algebra.algebraMap_eq_smul_one, LinearEquiv.map_smul,
        one_eq_id, LinearEquiv.conj_id] }


/-- A basis of a module induces an equivalence of algebras from the endomorphisms of the module to
square matrices. -/
def algEquivMatrix [Fintype n] (h : Basis n R M) : Module.End R M ≃ₐ[R] Matrix n n R :=
  h.equivFun.algConj.trans algEquivMatrix'


/-- The standard basis of the space linear maps between two modules
induced by a basis of the domain and codomain.

If `M₁` and `M₂` are modules with basis `b₁` and `b₂` respectively indexed
by finite types `ι₁` and `ι₂`,
then `Basis.linearMap b₁ b₂` is the basis of `M₁ →ₗ[R] M₂` indexed by `ι₂ × ι₁`
where `(i, j)` indexes the linear map that sends `b j` to `b i`
and sends all other basis vectors to `0`. -/
@[simps! (config := .lemmasOnly) repr_apply repr_symm_apply]
noncomputable
def linearMap (b₁ : Basis ι₁ R M₁) (b₂ : Basis ι₂ R M₂) :
    Basis (ι₂ × ι₁) R (M₁ →ₗ[R] M₂) :=
  (Matrix.stdBasis R ι₂ ι₁).map (LinearMap.toMatrix b₁ b₂).symm


lemma linearMap_apply (ij : ι₂ × ι₁) :
    (b₁.linearMap b₂ ij) = (Matrix.toLin b₁ b₂) (Matrix.stdBasis R ι₂ ι₁ ij) := by
  /-
    R : Type u_1
    M₁ : Type u_3
    M₂ : Type u_4
    ι₁ : Type u_6
    ι₂ : Type u_7
    inst✝⁷ : CommSemiring R
    inst✝⁶ : AddCommMonoid M₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    inst✝² : Fintype ι₁
    inst✝¹ : Fintype ι₂
    inst✝ : DecidableEq ι₁
    b₁ : Basis ι₁ R M₁
    b₂ : Basis ι₂ R M₂
    ij : Prod ι₂ ι₁
    ⊢ Eq ((b₁.linearMap b₂) ij) ((Matrix.toLin b₁ b₂) ((Matrix.stdBasis R ι₂ ι₁) i …
  -/
  simp [linearMap]
  /-
    🎉 no goals
  -/


lemma linearMap_apply_apply (ij : ι₂ × ι₁) (k : ι₁) :
    (b₁.linearMap b₂ ij) (b₁ k) = if ij.2 = k then b₂ ij.1 else 0 := by
  /-
    R : Type u_1
    M₁ : Type u_3
    M₂ : Type u_4
    ι₁ : Type u_6
    ι₂ : Type u_7
    inst✝⁷ : CommSemiring R
    inst✝⁶ : AddCommMonoid M₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    inst✝² : Fintype ι₁
    inst✝¹ : Fintype ι₂
    inst✝ : DecidableEq ι₁
    b₁ : Basis ι₁ R M₁
    b₂ : Basis ι₂ R M₂
    ij : Prod ι₂ ι₁
    k : ι₁
    ⊢ Eq (((b₁.linearMap b₂) ij) (b₁ k)) (ite (Eq ij.2 k) (b₂ ij.1) 0)
  -/
  have := Classical.decEq ι₂
  /-
    R : Type u_1
    M₁ : Type u_3
    M₂ : Type u_4
    ι₁ : Type u_6
    ι₂ : Type u_7
    inst✝⁷ : CommSemiring R
    inst✝⁶ : AddCommMonoid M₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    inst✝² : Fintype ι₁
    inst✝¹ : Fintype ι₂
    inst✝ : DecidableEq ι₁
    b₁ : Basis ι₁ R M₁
    b₂ : Basis ι₂ R M₂
    ij : Prod ι₂ ι₁
    k : ι₁
    this : DecidableEq ι₂
    ⊢ Eq (((b₁.linearMap b₂) ij) (b₁ k)) (ite (Eq ij.2 k) (b₂ ij.1) 0)
  -/
  rw [linearMap_apply, Matrix.stdBasis_eq_stdBasisMatrix, Matrix.toLin_self]
  /-
    R : Type u_1
    M₁ : Type u_3
    M₂ : Type u_4
    ι₁ : Type u_6
    ι₂ : Type u_7
    inst✝⁷ : CommSemiring R
    inst✝⁶ : AddCommMonoid M₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    inst✝² : Fintype ι₁
    inst✝¹ : Fintype ι₂
    inst✝ : DecidableEq ι₁
    b₁ : Basis ι₁ R M₁
    b₂ : Basis ι₂ R M₂
    ij : Prod ι₂ ι₁
    k : ι₁
    this : DecidableEq ι₂
    ⊢ Eq (Finset.univ.sum fun j => HSMul.hSMul (Matrix.stdBasisMatrix ij.1 ij.2 1  …
  -/
  dsimp only [Matrix.stdBasisMatrix, of_apply]
  /-
    R : Type u_1
    M₁ : Type u_3
    M₂ : Type u_4
    ι₁ : Type u_6
    ι₂ : Type u_7
    inst✝⁷ : CommSemiring R
    inst✝⁶ : AddCommMonoid M₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    inst✝² : Fintype ι₁
    inst✝¹ : Fintype ι₂
    inst✝ : DecidableEq ι₁
    b₁ : Basis ι₁ R M₁
    b₂ : Basis ι₂ R M₂
    ij : Prod ι₂ ι₁
    k : ι₁
    this : DecidableEq ι₂
    ⊢ Eq (Finset.univ.sum fun j => HSMul.hSMul (ite (And (Eq ij.1 j) (Eq ij.2 k))  …
  -/
  simp_rw [ite_smul, one_smul, zero_smul, ite_and, Finset.sum_ite_eq, Finset.mem_univ, if_true]
  /-
    🎉 no goals
  -/


/-- The standard basis of the endomorphism algebra of a module
induced by a basis of the module.

If `M` is a module with basis `b` indexed by a finite type `ι`,
then `Basis.end b` is the basis of `Module.End R M` indexed by `ι × ι`
where `(i, j)` indexes the linear map that sends `b j` to `b i`
and sends all other basis vectors to `0`. -/
@[simps! (config := .lemmasOnly) repr_apply repr_symm_apply]
noncomputable
abbrev _root_.Basis.end (b : Basis ι R M) : Basis (ι × ι) R (Module.End R M) :=
  b.linearMap b


lemma end_apply (ij : ι × ι) : (b.end ij) = (Matrix.toLin b b) (Matrix.stdBasis R ι ι ij) :=
  linearMap_apply b b ij


lemma end_apply_apply (ij : ι × ι) (k : ι) : (b.end ij) (b k) = if ij.2 = k then b ij.1 else 0 :=
  linearMap_apply_apply b b ij k


/--
Let `M` be an `A`-module. Every `A`-linear map `Mⁿ → Mⁿ` corresponds to a `n×n`-matrix whose entries
are `A`-linear maps `M → M`. In another word, we have`End(Mⁿ) ≅ Matₙₓₙ(End(M))` defined by:
`(f : Mⁿ → Mⁿ) ↦ (x ↦ f (0, ..., x at j-th position, ..., 0) i)ᵢⱼ` and
`m : Matₙₓₙ(End(M)) ↦ (v ↦ ∑ⱼ mᵢⱼ(vⱼ))`.

See also `LinearMap.toMatrix'`
-/
@[simp]
def endVecRingEquivMatrixEnd :
    Module.End A (ι → M) ≃+* Matrix ι ι (Module.End A M) where
  toFun f i j :=
  { toFun := fun x ↦ f (Pi.single j x) i
                             /-
                               ι : Type u_1
                               inst✝⁸ : Fintype ι
                               inst✝⁷ : DecidableEq ι
                               R : Type u_2
                               inst✝⁶ : CommSemiring R
                               A : Type u_3
                               inst✝⁵ : Semiring A
                               inst✝⁴ : Algebra R A
                               M : Type u_4
                               inst✝³ : AddCommMonoid M
                               inst✝² : Module R M
                               inst✝¹ : Module A M
                               inst✝ : IsScalarTower R A M
                               f : Module.End A (ι → M)
                               i j : ι
                               x y : M
                               ⊢ Eq ((fun x => f (Pi.single j x) i) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x => f  …
                             -/
    map_add' := fun x y ↦ by simp [Pi.single_add]
                             /-
                               🎉 no goals
                             -/
                              /-
                                ι : Type u_1
                                inst✝⁸ : Fintype ι
                                inst✝⁷ : DecidableEq ι
                                R : Type u_2
                                inst✝⁶ : CommSemiring R
                                A : Type u_3
                                inst✝⁵ : Semiring A
                                inst✝⁴ : Algebra R A
                                M : Type u_4
                                inst✝³ : AddCommMonoid M
                                inst✝² : Module R M
                                inst✝¹ : Module A M
                                inst✝ : IsScalarTower R A M
                                f : Module.End A (ι → M)
                                i j : ι
                                x : A
                                y : M
                                ⊢ Eq ({ toFun := fun x => f (Pi.single j x) i, map_add' := ⋯ }.toFun (HSMul.hS …
                              -/
    map_smul' := fun x y ↦ by simp [Pi.single_smul] }
                              /-
                                🎉 no goals
                              -/
  invFun m :=
  { toFun := fun x i ↦ ∑ j, m i j (x j)
                   /-
                     ι : Type u_1
                     inst✝⁸ : Fintype ι
                     inst✝⁷ : DecidableEq ι
                     R : Type u_2
                     inst✝⁶ : CommSemiring R
                     A : Type u_3
                     inst✝⁵ : Semiring A
                     inst✝⁴ : Algebra R A
                     M : Type u_4
                     inst✝³ : AddCommMonoid M
                     inst✝² : Module R M
                     inst✝¹ : Module A M
                     inst✝ : IsScalarTower R A M
                     m : Matrix ι ι (Module.End A M)
                     ⊢ ∀ (x y : ι → M), Eq ((fun x i => Finset.univ.sum fun j => (m i j) (x j)) (HA …
                   -/
    map_add' := by intros; ext; simp [Finset.sum_add_distrib]
                                /-
                                  🎉 no goals
                                -/
                    /-
                      ι : Type u_1
                      inst✝⁸ : Fintype ι
                      inst✝⁷ : DecidableEq ι
                      R : Type u_2
                      inst✝⁶ : CommSemiring R
                      A : Type u_3
                      inst✝⁵ : Semiring A
                      inst✝⁴ : Algebra R A
                      M : Type u_4
                      inst✝³ : AddCommMonoid M
                      inst✝² : Module R M
                      inst✝¹ : Module A M
                      inst✝ : IsScalarTower R A M
                      m : Matrix ι ι (Module.End A M)
                      ⊢ ∀ (m_1 : A) (x : ι → M), Eq ({ toFun := fun x i => Finset.univ.sum fun j =>  …
                    -/
    map_smul' := by intros; ext; simp [Finset.smul_sum] }
                                 /-
                                   🎉 no goals
                                 -/
  left_inv f := by
    /-
      ι : Type u_1
      inst✝⁸ : Fintype ι
      inst✝⁷ : DecidableEq ι
      R : Type u_2
      inst✝⁶ : CommSemiring R
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      f : Module.End A (ι → M)
      ⊢ Eq ((fun m => { toFun := fun x i => Finset.univ.sum fun j => (m i j) (x j),  …
    -/
    ext i x j
    /-
      case h.h.h
      ι : Type u_1
      inst✝⁸ : Fintype ι
      inst✝⁷ : DecidableEq ι
      R : Type u_2
      inst✝⁶ : CommSemiring R
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      f : Module.End A (ι → M)
      i : ι
      x : M
      j : ι
      ⊢ Eq ((((fun m => { toFun := fun x i => Finset.univ.sum fun j => (m i j) (x j) …
    -/
    simp only [LinearMap.coe_mk, AddHom.coe_mk, coe_comp, coe_single, Function.comp_apply]
    /-
      case h.h.h
      ι : Type u_1
      inst✝⁸ : Fintype ι
      inst✝⁷ : DecidableEq ι
      R : Type u_2
      inst✝⁶ : CommSemiring R
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      f : Module.End A (ι → M)
      i : ι
      x : M
      j : ι
      ⊢ Eq (Finset.univ.sum fun x_1 => f (Pi.single x_1 (Pi.single i x x_1)) j) (f ( …
    -/
    rw [← Fintype.sum_apply, ← map_sum]
    /-
      case h.h.h
      ι : Type u_1
      inst✝⁸ : Fintype ι
      inst✝⁷ : DecidableEq ι
      R : Type u_2
      inst✝⁶ : CommSemiring R
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      f : Module.End A (ι → M)
      i : ι
      x : M
      j : ι
      ⊢ Eq (f (Finset.univ.sum fun x_1 => Pi.single x_1 (Pi.single i x x_1)) j) (f ( …
    -/
    exact congr_arg₂ _ (by aesop) rfl
    /-
      🎉 no goals
    -/
                    /-
                      ι : Type u_1
                      inst✝⁸ : Fintype ι
                      inst✝⁷ : DecidableEq ι
                      R : Type u_2
                      inst✝⁶ : CommSemiring R
                      A : Type u_3
                      inst✝⁵ : Semiring A
                      inst✝⁴ : Algebra R A
                      M : Type u_4
                      inst✝³ : AddCommMonoid M
                      inst✝² : Module R M
                      inst✝¹ : Module A M
                      inst✝ : IsScalarTower R A M
                      m : Matrix ι ι (Module.End A M)
                      ⊢ Eq ((fun f i j => { toFun := fun x => f (Pi.single j x) i, map_add' := ⋯, ma …
                    -/
  right_inv m := by ext; simp [Pi.single_apply, apply_ite]
                         /-
                           🎉 no goals
                         -/
  map_mul' f g := by
    /-
      ι : Type u_1
      inst✝⁸ : Fintype ι
      inst✝⁷ : DecidableEq ι
      R : Type u_2
      inst✝⁶ : CommSemiring R
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      f g : Module.End A (ι → M)
      ⊢ Eq ({ toFun := fun f i j => { toFun := fun x => f (Pi.single j x) i, map_add …
    -/
    ext
    simp only [LinearMap.mul_apply, LinearMap.coe_mk, AddHom.coe_mk, Matrix.mul_apply, coeFn_sum,
      Finset.sum_apply]
    /-
      case a.h
      ι : Type u_1
      inst✝⁸ : Fintype ι
      inst✝⁷ : DecidableEq ι
      R : Type u_2
      inst✝⁶ : CommSemiring R
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      f g : Module.End A (ι → M)
      i✝ j✝ : ι
      x✝ : M
      ⊢ Eq (f (g (Pi.single j✝ x✝)) i✝) (Finset.univ.sum fun x => f (Pi.single x (g  …
    -/
    rw [← Fintype.sum_apply, ← map_sum]
    /-
      case a.h
      ι : Type u_1
      inst✝⁸ : Fintype ι
      inst✝⁷ : DecidableEq ι
      R : Type u_2
      inst✝⁶ : CommSemiring R
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      f g : Module.End A (ι → M)
      i✝ j✝ : ι
      x✝ : M
      ⊢ Eq (f (g (Pi.single j✝ x✝)) i✝) (f (Finset.univ.sum fun x => Pi.single x (g  …
    -/
    exact congr_arg₂ _ (by aesop) rfl
    /-
      🎉 no goals
    -/
                     /-
                       ι : Type u_1
                       inst✝⁸ : Fintype ι
                       inst✝⁷ : DecidableEq ι
                       R : Type u_2
                       inst✝⁶ : CommSemiring R
                       A : Type u_3
                       inst✝⁵ : Semiring A
                       inst✝⁴ : Algebra R A
                       M : Type u_4
                       inst✝³ : AddCommMonoid M
                       inst✝² : Module R M
                       inst✝¹ : Module A M
                       inst✝ : IsScalarTower R A M
                       f g : Module.End A (ι → M)
                       ⊢ Eq ({ toFun := fun f i j => { toFun := fun x => f (Pi.single j x) i, map_add …
                     -/
  map_add' f g := by ext; simp
                          /-
                            🎉 no goals
                          -/


/--
Let `M` be an `A`-module. Every `A`-linear map `Mⁿ → Mⁿ` corresponds to a `n×n`-matrix whose entries
are `R`-linear maps `M → M`. In another word, we have`End(Mⁿ) ≅ Matₙₓₙ(End(M))` defined by:
`(f : Mⁿ → Mⁿ) ↦ (x ↦ f (0, ..., x at j-th position, ..., 0) i)ᵢⱼ` and
`m : Matₙₓₙ(End(M)) ↦ (v ↦ ∑ⱼ mᵢⱼ(vⱼ))`.

See also `LinearMap.toMatrix'`
-/
@[simps!]
def endVecAlgEquivMatrixEnd :
    Module.End A (ι → M) ≃ₐ[R] Matrix ι ι (Module.End A M) where
  __ := endVecRingEquivMatrixEnd ι A M
  commutes' r := by
    /-
      ι : Type u_1
      inst✝⁸ : Fintype ι
      inst✝⁷ : DecidableEq ι
      R : Type u_2
      inst✝⁶ : CommSemiring R
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      r : R
      ⊢ Eq (__spread✝⁻⁰.toFun ((algebraMap R (Module.End A (ι → M))) r)) ((algebraMa …
    -/
    ext
    simp only [endVecRingEquivMatrixEnd, RingEquiv.toEquiv_eq_coe, Module.algebraMap_end_eq_smul_id,
      Equiv.toFun_as_coe, EquivLike.coe_coe, RingEquiv.coe_mk, Equiv.coe_fn_mk,
      LinearMap.smul_apply, id_coe, id_eq, Pi.smul_apply, Pi.single_apply, smul_ite, smul_zero,
      LinearMap.coe_mk, AddHom.coe_mk, algebraMap_matrix_apply]
    /-
      case a.h
      ι : Type u_1
      inst✝⁸ : Fintype ι
      inst✝⁷ : DecidableEq ι
      R : Type u_2
      inst✝⁶ : CommSemiring R
      A : Type u_3
      inst✝⁵ : Semiring A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      r : R
      i✝ j✝ : ι
      x✝ : M
      ⊢ Eq (ite (Eq i✝ j✝) (HSMul.hSMul r x✝) 0) ((ite (Eq i✝ j✝) (HSMul.hSMul r Lin …
    -/
                  /-
                    🎉 no goals
                  -/
    split_ifs <;> rfl
                  /-
                    🎉 no goals
                  -/


