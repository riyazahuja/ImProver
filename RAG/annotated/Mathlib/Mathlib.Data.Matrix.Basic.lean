instance decidableEq [DecidableEq α] [Fintype m] [Fintype n] : DecidableEq (Matrix m n α) :=
  Fintype.decidablePiFintype


instance {n m} [Fintype m] [DecidableEq m] [Fintype n] [DecidableEq n] (α) [Fintype α] :
    Fintype (Matrix m n α) := inferInstanceAs (Fintype (m → n → α))


instance {n m} [Finite m] [Finite n] (α) [Finite α] :
    Finite (Matrix m n α) := inferInstanceAs (Finite (m → n → α))


/-- This is `Matrix.of` bundled as a linear equivalence. -/
def ofLinearEquiv [Semiring R] [AddCommMonoid α] [Module R α] : (m → n → α) ≃ₗ[R] Matrix m n α where
  __ := ofAddEquiv
  map_smul' _ _ := rfl


@[simp] lemma coe_ofLinearEquiv [Semiring R] [AddCommMonoid α] [Module R α] :
    ⇑(ofLinearEquiv _ : (m → n → α) ≃ₗ[R] Matrix m n α) = of := rfl

@[simp] lemma coe_ofLinearEquiv_symm [Semiring R] [AddCommMonoid α] [Module R α] :
    ⇑((ofLinearEquiv _).symm : Matrix m n α ≃ₗ[R] (m → n → α)) = of.symm := rfl


theorem sum_apply [AddCommMonoid α] (i : m) (j : n) (s : Finset β) (g : β → Matrix m n α) :
    (∑ c ∈ s, g c) i j = ∑ c ∈ s, g c i j :=
  (congr_fun (s.sum_apply i g) j).trans (s.sum_apply j _)


/-- `Matrix.diagonal` as an `AddMonoidHom`. -/
@[simps]
def diagonalAddMonoidHom [AddZeroClass α] : (n → α) →+ Matrix n n α where
  toFun := diagonal
  map_zero' := diagonal_zero
  map_add' x y := (diagonal_add x y).symm


/-- `Matrix.diagonal` as a `LinearMap`. -/
@[simps]
def diagonalLinearMap [Semiring R] [AddCommMonoid α] [Module R α] : (n → α) →ₗ[R] Matrix n n α :=
  { diagonalAddMonoidHom n α with map_smul' := diagonal_smul }


lemma zero_le_one_elem [Preorder α] [ZeroLEOneClass α] (i j : n) :
    0 ≤ (1 : Matrix n n α) i j := by
  /-
    n : Type u_3
    α : Type v
    inst✝⁴ : DecidableEq n
    inst✝³ : Zero α
    inst✝² : One α
    inst✝¹ : Preorder α
    inst✝ : ZeroLEOneClass α
    i j : n
    ⊢ LE.le 0 (1 i j)
  -/
  by_cases hi : i = j
    /-
      case pos
      n : Type u_3
      α : Type v
      inst✝⁴ : DecidableEq n
      inst✝³ : Zero α
      inst✝² : One α
      inst✝¹ : Preorder α
      inst✝ : ZeroLEOneClass α
      i j : n
      hi : Eq i j
      ⊢ LE.le 0 (1 i j)
    -/
  · subst hi
    /-
      case pos
      n : Type u_3
      α : Type v
      inst✝⁴ : DecidableEq n
      inst✝³ : Zero α
      inst✝² : One α
      inst✝¹ : Preorder α
      inst✝ : ZeroLEOneClass α
      i : n
      ⊢ LE.le 0 (1 i i)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Type u_3
      α : Type v
      inst✝⁴ : DecidableEq n
      inst✝³ : Zero α
      inst✝² : One α
      inst✝¹ : Preorder α
      inst✝ : ZeroLEOneClass α
      i j : n
      hi : Not (Eq i j)
      ⊢ LE.le 0 (1 i j)
    -/
  · simp [hi]
    /-
      🎉 no goals
    -/


lemma zero_le_one_row [Preorder α] [ZeroLEOneClass α] (i : n) :
    0 ≤ (1 : Matrix n n α) i :=
  zero_le_one_elem i


/-- `Matrix.diag` as an `AddMonoidHom`. -/
@[simps]
def diagAddMonoidHom [AddZeroClass α] : Matrix n n α →+ n → α where
  toFun := diag
  map_zero' := diag_zero
  map_add' := diag_add


/-- `Matrix.diag` as a `LinearMap`. -/
@[simps]
def diagLinearMap [Semiring R] [AddCommMonoid α] [Module R α] : Matrix n n α →ₗ[R] n → α :=
  { diagAddMonoidHom n α with map_smul' := diag_smul }


@[simp]
theorem diag_list_sum [AddMonoid α] (l : List (Matrix n n α)) : diag l.sum = (l.map diag).sum :=
  map_list_sum (diagAddMonoidHom n α) l


@[simp]
theorem diag_multiset_sum [AddCommMonoid α] (s : Multiset (Matrix n n α)) :
    diag s.sum = (s.map diag).sum :=
  map_multiset_sum (diagAddMonoidHom n α) s


@[simp]
theorem diag_sum {ι} [AddCommMonoid α] (s : Finset ι) (f : ι → Matrix n n α) :
    diag (∑ i ∈ s, f i) = ∑ i ∈ s, diag (f i) :=
  map_sum (diagAddMonoidHom n α) f s


/-- `Matrix.diagonal` as a `RingHom`. -/
@[simps]
def diagonalRingHom [Fintype n] [DecidableEq n] : (n → α) →+* Matrix n n α :=
  { diagonalAddMonoidHom n α with
    toFun := diagonal
    map_one' := diagonal_one
    map_mul' := fun _ _ => (diagonal_mul_diagonal' _ _).symm }


theorem diagonal_pow [Fintype n] [DecidableEq n] (v : n → α) (k : ℕ) :
    diagonal v ^ k = diagonal (v ^ k) :=
  (map_pow (diagonalRingHom n α) v k).symm


/-- The ring homomorphism `α →+* Matrix n n α`
sending `a` to the diagonal matrix with `a` on the diagonal.
-/
def scalar (n : Type u) [DecidableEq n] [Fintype n] : α →+* Matrix n n α :=
  (diagonalRingHom n α).comp <| Pi.constRingHom n α


@[simp]
theorem scalar_apply (a : α) : scalar n a = diagonal fun _ => a :=
  rfl


theorem scalar_inj [Nonempty n] {r s : α} : scalar n r = scalar n s ↔ r = s :=
  (diagonal_injective.comp Function.const_injective).eq_iff


theorem scalar_commute_iff {r : α} {M : Matrix n n α} :
    Commute (scalar n r) M ↔ r • M = MulOpposite.op r • M := by
  /-
    n : Type u_3
    α : Type v
    inst✝² : Semiring α
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    r : α
    M : Matrix n n α
    ⊢ Iff (Commute ((Matrix.scalar n) r) M) (Eq (HSMul.hSMul r M) (HSMul.hSMul (Mu …
  -/
  simp_rw [Commute, SemiconjBy, scalar_apply, ← smul_eq_diagonal_mul, ← op_smul_eq_mul_diagonal]
  /-
    🎉 no goals
  -/


theorem scalar_commute (r : α) (hr : ∀ r', Commute r r') (M : Matrix n n α) :
    Commute (scalar n r) M := scalar_commute_iff.2 <| ext fun _ _ => hr _


instance instAlgebra : Algebra R (Matrix n n α) where
  toRingHom := (Matrix.scalar n).comp (algebraMap R α)
  commutes' _ _ := scalar_commute _ (fun _ => Algebra.commutes _ _) _
                      /-
                        l : Type u_1
                        m : Type u_2
                        n : Type u_3
                        o : Type u_4
                        m' : o → Type u_5
                        n' : o → Type u_6
                        R : Type u_7
                        S : Type u_8
                        α : Type v
                        β : Type w
                        γ : Type u_9
                        inst✝⁶ : Fintype n
                        inst✝⁵ : DecidableEq n
                        inst✝⁴ : CommSemiring R
                        inst✝³ : Semiring α
                        inst✝² : Semiring β
                        inst✝¹ : Algebra R α
                        inst✝ : Algebra R β
                        r : R
                        x : Matrix n n α
                        ⊢ Eq (HSMul.hSMul r x) (HMul.hMul (((Matrix.scalar n).comp (algebraMap R α)) r …
                      -/
  smul_def' r x := by ext; simp [Matrix.scalar, Algebra.smul_def r]
                           /-
                             🎉 no goals
                           -/


theorem algebraMap_matrix_apply {r : R} {i j : n} :
    algebraMap R (Matrix n n α) r i j = if i = j then algebraMap R α r else 0 := by
  /-
    n : Type u_3
    R : Type u_7
    α : Type v
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : CommSemiring R
    inst✝¹ : Semiring α
    inst✝ : Algebra R α
    r : R
    i j : n
    ⊢ Eq ((algebraMap R (Matrix n n α)) r i j) (ite (Eq i j) ((algebraMap R α) r) 0)
  -/
  dsimp [algebraMap, Algebra.toRingHom, Matrix.scalar]
  /-
    n : Type u_3
    R : Type u_7
    α : Type v
    inst✝⁴ : Fintype n
    inst✝³ : DecidableEq n
    inst✝² : CommSemiring R
    inst✝¹ : Semiring α
    inst✝ : Algebra R α
    r : R
    i j : n
    ⊢ Eq (Matrix.diagonal ((Pi.constRingHom n α) (Algebra.toRingHom r)) i j) (ite  …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h, Matrix.one_apply_ne]
                       /-
                         🎉 no goals
                       -/


theorem algebraMap_eq_diagonal (r : R) :
    algebraMap R (Matrix n n α) r = diagonal (algebraMap R (n → α) r) := rfl


theorem algebraMap_eq_diagonalRingHom :
    algebraMap R (Matrix n n α) = (diagonalRingHom n α).comp (algebraMap R _) := rfl


@[simp]
theorem map_algebraMap (r : R) (f : α → β) (hf : f 0 = 0)
    (hf₂ : f (algebraMap R α r) = algebraMap R β r) :
    (algebraMap R (Matrix n n α) r).map f = algebraMap R (Matrix n n β) r := by
  /-
    n : Type u_3
    R : Type u_7
    α : Type v
    β : Type w
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq n
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring α
    inst✝² : Semiring β
    inst✝¹ : Algebra R α
    inst✝ : Algebra R β
    r : R
    f : α → β
    hf : Eq (f 0) 0
    hf₂ : Eq (f ((algebraMap R α) r)) ((algebraMap R β) r)
    ⊢ Eq (((algebraMap R (Matrix n n α)) r).map f) ((algebraMap R (Matrix n n β)) r)
  -/
  rw [algebraMap_eq_diagonal, algebraMap_eq_diagonal, diagonal_map hf]
  -- Porting note: (congr) the remaining proof was
  -- ```
  -- congr 1
  -- simp only [hf₂, Pi.algebraMap_apply]
  -- ```
  -- But some `congr 1` doesn't quite work.
  /-
    n : Type u_3
    R : Type u_7
    α : Type v
    β : Type w
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq n
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring α
    inst✝² : Semiring β
    inst✝¹ : Algebra R α
    inst✝ : Algebra R β
    r : R
    f : α → β
    hf : Eq (f 0) 0
    hf₂ : Eq (f ((algebraMap R α) r)) ((algebraMap R β) r)
    ⊢ Eq (Matrix.diagonal fun m => f ((algebraMap R (n → α)) r m)) (Matrix.diagona …
  -/
  simp only [Pi.algebraMap_apply, diagonal_eq_diagonal_iff]
  /-
    n : Type u_3
    R : Type u_7
    α : Type v
    β : Type w
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq n
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring α
    inst✝² : Semiring β
    inst✝¹ : Algebra R α
    inst✝ : Algebra R β
    r : R
    f : α → β
    hf : Eq (f 0) 0
    hf₂ : Eq (f ((algebraMap R α) r)) ((algebraMap R β) r)
    ⊢ n → Eq (f ((algebraMap R α) r)) ((algebraMap R β) r)
  -/
  intro
  /-
    n : Type u_3
    R : Type u_7
    α : Type v
    β : Type w
    inst✝⁶ : Fintype n
    inst✝⁵ : DecidableEq n
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring α
    inst✝² : Semiring β
    inst✝¹ : Algebra R α
    inst✝ : Algebra R β
    r : R
    f : α → β
    hf : Eq (f 0) 0
    hf₂ : Eq (f ((algebraMap R α) r)) ((algebraMap R β) r)
    i✝ : n
    ⊢ Eq (f ((algebraMap R α) r)) ((algebraMap R β) r)
  -/
  rw [hf₂]
  /-
    🎉 no goals
  -/


/-- `Matrix.diagonal` as an `AlgHom`. -/
@[simps]
def diagonalAlgHom : (n → α) →ₐ[R] Matrix n n α :=
  { diagonalRingHom n α with
    toFun := diagonal
    commutes' := fun r => (algebraMap_eq_diagonal r).symm }


variable (R α) in
/-- Extracting entries from a matrix as an additive homomorphism.  -/
@[simps]
def entryAddHom (i : m) (j : n) : AddHom (Matrix m n α) α where
  toFun M := M i j
  map_add' _ _ := rfl

-- It is necessary to spell out the name of the coercion explicitly on the RHS
-- for unification to succeed

lemma entryAddHom_eq_comp {i : m} {j : n} :
    entryAddHom α i j =
      ((Pi.evalAddHom (fun _ => α) j).comp (Pi.evalAddHom _ i)).comp
        (AddHomClass.toAddHom ofAddEquiv.symm) :=
  rfl


variable (R α) in
/--
Extracting entries from a matrix as an additive monoid homomorphism. Note this cannot be upgraded to
a ring homomorphism, as it does not respect multiplication.
-/
@[simps]
def entryAddMonoidHom (i : m) (j : n) : Matrix m n α →+ α where
  toFun M := M i j
  map_add' _ _ := rfl
  map_zero' := rfl

-- It is necessary to spell out the name of the coercion explicitly on the RHS
-- for unification to succeed

lemma entryAddMonoidHom_eq_comp {i : m} {j : n} :
    entryAddMonoidHom α i j =
      ((Pi.evalAddMonoidHom (fun _ => α) j).comp (Pi.evalAddMonoidHom _ i)).comp
        (AddMonoidHomClass.toAddMonoidHom ofAddEquiv.symm) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝ : AddZeroClass α
    i : m
    j : n
    ⊢ Eq (Matrix.entryAddMonoidHom α i j) (((Pi.evalAddMonoidHom (fun x => α) j).c …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] lemma evalAddMonoidHom_comp_diagAddMonoidHom (i : m) :
    (Pi.evalAddMonoidHom _ i).comp (diagAddMonoidHom m α) = entryAddMonoidHom α i i := by
  /-
    m : Type u_2
    α : Type v
    inst✝ : AddZeroClass α
    i : m
    ⊢ Eq ((Pi.evalAddMonoidHom (fun i => α) i).comp (Matrix.diagAddMonoidHom m α)) …
  -/
  simp [AddMonoidHom.ext_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma entryAddMonoidHom_toAddHom {i : m} {j : n} :
  (entryAddMonoidHom α i j : AddHom _ _) = entryAddHom α i j := rfl


variable (R α) in
/--
Extracting entries from a matrix as a linear map. Note this cannot be upgraded to an algebra
homomorphism, as it does not respect multiplication.
-/
@[simps]
def entryLinearMap (i : m) (j : n) :
    Matrix m n α →ₗ[R] α where
  toFun M := M i j
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

-- It is necessary to spell out the name of the coercion explicitly on the RHS
-- for unification to succeed

lemma entryLinearMap_eq_comp {i : m} {j : n} :
    entryLinearMap R α i j =
      LinearMap.proj j ∘ₗ LinearMap.proj i ∘ₗ (ofLinearEquiv R).symm.toLinearMap := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_7
    α : Type v
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid α
    inst✝ : Module R α
    i : m
    j : n
    ⊢ Eq (Matrix.entryLinearMap R α i j) ((LinearMap.proj j).comp ((LinearMap.proj …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] lemma proj_comp_diagLinearMap (i : m) :
    LinearMap.proj i ∘ₗ diagLinearMap m R α = entryLinearMap R α i i := by
  /-
    m : Type u_2
    R : Type u_7
    α : Type v
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid α
    inst✝ : Module R α
    i : m
    ⊢ Eq ((LinearMap.proj i).comp (Matrix.diagLinearMap m R α)) (Matrix.entryLinea …
  -/
  simp [LinearMap.ext_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma entryLinearMap_toAddMonoidHom {i : m} {j : n} :
    (entryLinearMap R α i j : _ →+ _) = entryAddMonoidHom α i j := rfl


@[simp] lemma entryLinearMap_toAddHom {i : m} {j : n} :
    (entryLinearMap R α i j : AddHom _ _) = entryAddHom α i j := rfl


/-- The `Equiv` between spaces of matrices induced by an `Equiv` between their
coefficients. This is `Matrix.map` as an `Equiv`. -/
@[simps apply]
def mapMatrix (f : α ≃ β) : Matrix m n α ≃ Matrix m n β where
  toFun M := M.map f
  invFun M := M.map f.symm
  left_inv _ := Matrix.ext fun _ _ => f.symm_apply_apply _
  right_inv _ := Matrix.ext fun _ _ => f.apply_symm_apply _


@[simp]
theorem mapMatrix_refl : (Equiv.refl α).mapMatrix = Equiv.refl (Matrix m n α) :=
  rfl


@[simp]
theorem mapMatrix_symm (f : α ≃ β) : f.mapMatrix.symm = (f.symm.mapMatrix : Matrix m n β ≃ _) :=
  rfl


@[simp]
theorem mapMatrix_trans (f : α ≃ β) (g : β ≃ γ) :
    f.mapMatrix.trans g.mapMatrix = ((f.trans g).mapMatrix : Matrix m n α ≃ _) :=
  rfl


/-- The `AddMonoidHom` between spaces of matrices induced by an `AddMonoidHom` between their
coefficients. This is `Matrix.map` as an `AddMonoidHom`. -/
@[simps]
def mapMatrix (f : α →+ β) : Matrix m n α →+ Matrix m n β where
  toFun M := M.map f
  map_zero' := Matrix.map_zero f f.map_zero
  map_add' := Matrix.map_add f f.map_add


@[simp]
theorem mapMatrix_id : (AddMonoidHom.id α).mapMatrix = AddMonoidHom.id (Matrix m n α) :=
  rfl


@[simp]
theorem mapMatrix_comp (f : β →+ γ) (g : α →+ β) :
    f.mapMatrix.comp g.mapMatrix = ((f.comp g).mapMatrix : Matrix m n α →+ _) :=
  rfl


@[simp] lemma entryAddMonoidHom_comp_mapMatrix (f : α →+ β) (i : m) (j : n) :
    (entryAddMonoidHom β i j).comp f.mapMatrix = f.comp (entryAddMonoidHom α i j) := rfl


/-- The `AddEquiv` between spaces of matrices induced by an `AddEquiv` between their
coefficients. This is `Matrix.map` as an `AddEquiv`. -/
@[simps apply]
def mapMatrix (f : α ≃+ β) : Matrix m n α ≃+ Matrix m n β :=
  { f.toEquiv.mapMatrix with
    toFun := fun M => M.map f
    invFun := fun M => M.map f.symm
    map_add' := Matrix.map_add f (map_add f) }


@[simp]
theorem mapMatrix_refl : (AddEquiv.refl α).mapMatrix = AddEquiv.refl (Matrix m n α) :=
  rfl


@[simp]
theorem mapMatrix_symm (f : α ≃+ β) : f.mapMatrix.symm = (f.symm.mapMatrix : Matrix m n β ≃+ _) :=
  rfl


@[simp]
theorem mapMatrix_trans (f : α ≃+ β) (g : β ≃+ γ) :
    f.mapMatrix.trans g.mapMatrix = ((f.trans g).mapMatrix : Matrix m n α ≃+ _) :=
  rfl


@[simp] lemma entryAddHom_comp_mapMatrix (f : α ≃+ β) (i : m) (j : n) :
    (entryAddHom β i j).comp (AddHomClass.toAddHom f.mapMatrix) =
      (f : AddHom α β).comp (entryAddHom _ i j) := rfl


/-- The `LinearMap` between spaces of matrices induced by a `LinearMap` between their
coefficients. This is `Matrix.map` as a `LinearMap`. -/
@[simps]
def mapMatrix (f : α →ₗ[R] β) : Matrix m n α →ₗ[R] Matrix m n β where
  toFun M := M.map f
  map_add' := Matrix.map_add f f.map_add
  map_smul' r := Matrix.map_smul f r (f.map_smul r)


@[simp]
theorem mapMatrix_id : LinearMap.id.mapMatrix = (LinearMap.id : Matrix m n α →ₗ[R] _) :=
  rfl


@[simp]
theorem mapMatrix_comp (f : β →ₗ[R] γ) (g : α →ₗ[R] β) :
    f.mapMatrix.comp g.mapMatrix = ((f.comp g).mapMatrix : Matrix m n α →ₗ[R] _) :=
  rfl


@[simp] lemma entryLinearMap_comp_mapMatrix (f : α →ₗ[R] β) (i : m) (j : n) :
    entryLinearMap R _ i j ∘ₗ f.mapMatrix = f ∘ₗ entryLinearMap R _ i j := rfl


/-- The `LinearEquiv` between spaces of matrices induced by a `LinearEquiv` between their
coefficients. This is `Matrix.map` as a `LinearEquiv`. -/
@[simps apply]
def mapMatrix (f : α ≃ₗ[R] β) : Matrix m n α ≃ₗ[R] Matrix m n β :=
  { f.toEquiv.mapMatrix,
    f.toLinearMap.mapMatrix with
    toFun := fun M => M.map f
    invFun := fun M => M.map f.symm }


@[simp]
theorem mapMatrix_refl : (LinearEquiv.refl R α).mapMatrix = LinearEquiv.refl R (Matrix m n α) :=
  rfl


@[simp]
theorem mapMatrix_symm (f : α ≃ₗ[R] β) :
    f.mapMatrix.symm = (f.symm.mapMatrix : Matrix m n β ≃ₗ[R] _) :=
  rfl


@[simp]
theorem mapMatrix_trans (f : α ≃ₗ[R] β) (g : β ≃ₗ[R] γ) :
    f.mapMatrix.trans g.mapMatrix = ((f.trans g).mapMatrix : Matrix m n α ≃ₗ[R] _) :=
  rfl


@[simp] lemma mapMatrix_toLinearMap (f : α ≃ₗ[R] β) :
    (f.mapMatrix : _ ≃ₗ[R] Matrix m n β).toLinearMap = f.toLinearMap.mapMatrix := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_7
    α : Type v
    β : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid α
    inst✝² : AddCommMonoid β
    inst✝¹ : Module R α
    inst✝ : Module R β
    f : LinearEquiv (RingHom.id R) α β
    ⊢ Eq (↑f.mapMatrix) (↑f).mapMatrix
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] lemma entryLinearMap_comp_mapMatrix (f : α ≃ₗ[R] β) (i : m) (j : n) :
    entryLinearMap R _ i j ∘ₗ f.mapMatrix.toLinearMap =
      f.toLinearMap ∘ₗ entryLinearMap R _ i j := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_7
    α : Type v
    β : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid α
    inst✝² : AddCommMonoid β
    inst✝¹ : Module R α
    inst✝ : Module R β
    f : LinearEquiv (RingHom.id R) α β
    i : m
    j : n
    ⊢ Eq ((Matrix.entryLinearMap R β i j).comp ↑f.mapMatrix) ((↑f).comp (Matrix.en …
  -/
  simp only [mapMatrix_toLinearMap, LinearMap.entryLinearMap_comp_mapMatrix]
  /-
    🎉 no goals
  -/


/-- The `RingHom` between spaces of square matrices induced by a `RingHom` between their
coefficients. This is `Matrix.map` as a `RingHom`. -/
@[simps]
def mapMatrix (f : α →+* β) : Matrix m m α →+* Matrix m m β :=
  { f.toAddMonoidHom.mapMatrix with
    toFun := fun M => M.map f
                   /-
                     l : Type u_1
                     m : Type u_2
                     n : Type u_3
                     o : Type u_4
                     m' : o → Type u_5
                     n' : o → Type u_6
                     R : Type u_7
                     S : Type u_8
                     α : Type v
                     β : Type w
                     γ : Type u_9
                     inst✝⁴ : Fintype m
                     inst✝³ : DecidableEq m
                     inst✝² : NonAssocSemiring α
                     inst✝¹ : NonAssocSemiring β
                     inst✝ : NonAssocSemiring γ
                     f : RingHom α β
                     ⊢ Eq ((fun M => M.map ⇑f) 1) 1
                   -/
    map_one' := by simp
                   /-
                     🎉 no goals
                   -/
    map_mul' := fun _ _ => Matrix.map_mul }


@[simp]
theorem mapMatrix_id : (RingHom.id α).mapMatrix = RingHom.id (Matrix m m α) :=
  rfl


@[simp]
theorem mapMatrix_comp (f : β →+* γ) (g : α →+* β) :
    f.mapMatrix.comp g.mapMatrix = ((f.comp g).mapMatrix : Matrix m m α →+* _) :=
  rfl


/-- The `RingEquiv` between spaces of square matrices induced by a `RingEquiv` between their
coefficients. This is `Matrix.map` as a `RingEquiv`. -/
@[simps apply]
def mapMatrix (f : α ≃+* β) : Matrix m m α ≃+* Matrix m m β :=
  { f.toRingHom.mapMatrix,
    f.toAddEquiv.mapMatrix with
    toFun := fun M => M.map f
    invFun := fun M => M.map f.symm }


@[simp]
theorem mapMatrix_refl : (RingEquiv.refl α).mapMatrix = RingEquiv.refl (Matrix m m α) :=
  rfl


@[simp]
theorem mapMatrix_symm (f : α ≃+* β) : f.mapMatrix.symm = (f.symm.mapMatrix : Matrix m m β ≃+* _) :=
  rfl


@[simp]
theorem mapMatrix_trans (f : α ≃+* β) (g : β ≃+* γ) :
    f.mapMatrix.trans g.mapMatrix = ((f.trans g).mapMatrix : Matrix m m α ≃+* _) :=
  rfl


open MulOpposite in
/--
For any ring `R`, we have ring isomorphism `Matₙₓₙ(Rᵒᵖ) ≅ (Matₙₓₙ(R))ᵒᵖ` given by transpose.
-/
@[simps apply symm_apply]
def mopMatrix : Matrix m m αᵐᵒᵖ ≃+* (Matrix m m α)ᵐᵒᵖ where
  toFun M := op (M.transpose.map unop)
  invFun M := M.unop.transpose.map op
                   /-
                     l : Type u_1
                     m : Type u_2
                     n : Type u_3
                     o : Type u_4
                     m' : o → Type u_5
                     n' : o → Type u_6
                     R : Type u_7
                     S : Type u_8
                     α : Type v
                     β : Type w
                     γ : Type u_9
                     inst✝⁴ : Fintype m
                     inst✝³ : DecidableEq m
                     inst✝² : NonAssocSemiring α
                     inst✝¹ : NonAssocSemiring β
                     inst✝ : NonAssocSemiring γ
                     x✝ : Matrix m m (MulOpposite α)
                     ⊢ Eq ((fun M => (MulOpposite.unop M).transpose.map MulOpposite.op) ((fun M =>  …
                   -/
  left_inv _ := by aesop
                   /-
                     🎉 no goals
                   -/
                    /-
                      l : Type u_1
                      m : Type u_2
                      n : Type u_3
                      o : Type u_4
                      m' : o → Type u_5
                      n' : o → Type u_6
                      R : Type u_7
                      S : Type u_8
                      α : Type v
                      β : Type w
                      γ : Type u_9
                      inst✝⁴ : Fintype m
                      inst✝³ : DecidableEq m
                      inst✝² : NonAssocSemiring α
                      inst✝¹ : NonAssocSemiring β
                      inst✝ : NonAssocSemiring γ
                      x✝ : MulOpposite (Matrix m m α)
                      ⊢ Eq ((fun M => MulOpposite.op (M.transpose.map MulOpposite.unop)) ((fun M =>  …
                    -/
  right_inv _ := by aesop
                    /-
                      🎉 no goals
                    -/
                                       /-
                                         l : Type u_1
                                         m : Type u_2
                                         n : Type u_3
                                         o : Type u_4
                                         m' : o → Type u_5
                                         n' : o → Type u_6
                                         R : Type u_7
                                         S : Type u_8
                                         α : Type v
                                         β : Type w
                                         γ : Type u_9
                                         inst✝⁴ : Fintype m
                                         inst✝³ : DecidableEq m
                                         inst✝² : NonAssocSemiring α
                                         inst✝¹ : NonAssocSemiring β
                                         inst✝ : NonAssocSemiring γ
                                         x✝¹ x✝ : Matrix m m (MulOpposite α)
                                         ⊢ Eq (MulOpposite.unop ({ toFun := fun M => MulOpposite.op (M.transpose.map Mu …
                                       -/
  map_mul' _ _ := unop_injective <| by ext; simp [transpose, mul_apply]
                                            /-
                                              🎉 no goals
                                            -/
                     /-
                       l : Type u_1
                       m : Type u_2
                       n : Type u_3
                       o : Type u_4
                       m' : o → Type u_5
                       n' : o → Type u_6
                       R : Type u_7
                       S : Type u_8
                       α : Type v
                       β : Type w
                       γ : Type u_9
                       inst✝⁴ : Fintype m
                       inst✝³ : DecidableEq m
                       inst✝² : NonAssocSemiring α
                       inst✝¹ : NonAssocSemiring β
                       inst✝ : NonAssocSemiring γ
                       x✝¹ x✝ : Matrix m m (MulOpposite α)
                       ⊢ Eq ({ toFun := fun M => MulOpposite.op (M.transpose.map MulOpposite.unop), i …
                     -/
  map_add' _ _ := by aesop
                     /-
                       🎉 no goals
                     -/


/-- The `AlgHom` between spaces of square matrices induced by an `AlgHom` between their
coefficients. This is `Matrix.map` as an `AlgHom`. -/
@[simps]
def mapMatrix (f : α →ₐ[R] β) : Matrix m m α →ₐ[R] Matrix m m β :=
  { f.toRingHom.mapMatrix with
    toFun := fun M => M.map f
    commutes' := fun r => Matrix.map_algebraMap r f (map_zero _) (f.commutes r) }


@[simp]
theorem mapMatrix_id : (AlgHom.id R α).mapMatrix = AlgHom.id R (Matrix m m α) :=
  rfl


@[simp]
theorem mapMatrix_comp (f : β →ₐ[R] γ) (g : α →ₐ[R] β) :
    f.mapMatrix.comp g.mapMatrix = ((f.comp g).mapMatrix : Matrix m m α →ₐ[R] _) :=
  rfl


/-- The `AlgEquiv` between spaces of square matrices induced by an `AlgEquiv` between their
coefficients. This is `Matrix.map` as an `AlgEquiv`. -/
@[simps apply]
def mapMatrix (f : α ≃ₐ[R] β) : Matrix m m α ≃ₐ[R] Matrix m m β :=
  { f.toAlgHom.mapMatrix,
    f.toRingEquiv.mapMatrix with
    toFun := fun M => M.map f
    invFun := fun M => M.map f.symm }


@[simp]
theorem mapMatrix_refl : AlgEquiv.refl.mapMatrix = (AlgEquiv.refl : Matrix m m α ≃ₐ[R] _) :=
  rfl


@[simp]
theorem mapMatrix_symm (f : α ≃ₐ[R] β) :
    f.mapMatrix.symm = (f.symm.mapMatrix : Matrix m m β ≃ₐ[R] _) :=
  rfl


@[simp]
theorem mapMatrix_trans (f : α ≃ₐ[R] β) (g : β ≃ₐ[R] γ) :
    f.mapMatrix.trans g.mapMatrix = ((f.trans g).mapMatrix : Matrix m m α ≃ₐ[R] _) :=
  rfl


/-- `Matrix.transpose` as an `AddEquiv` -/
@[simps apply]
def transposeAddEquiv [Add α] : Matrix m n α ≃+ Matrix n m α where
  toFun := transpose
  invFun := transpose
  left_inv := transpose_transpose
  right_inv := transpose_transpose
  map_add' := transpose_add


@[simp]
theorem transposeAddEquiv_symm [Add α] : (transposeAddEquiv m n α).symm = transposeAddEquiv n m α :=
  rfl


theorem transpose_list_sum [AddMonoid α] (l : List (Matrix m n α)) :
    l.sumᵀ = (l.map transpose).sum :=
  map_list_sum (transposeAddEquiv m n α) l


theorem transpose_multiset_sum [AddCommMonoid α] (s : Multiset (Matrix m n α)) :
    s.sumᵀ = (s.map transpose).sum :=
  (transposeAddEquiv m n α).toAddMonoidHom.map_multiset_sum s


theorem transpose_sum [AddCommMonoid α] {ι : Type*} (s : Finset ι) (M : ι → Matrix m n α) :
    (∑ i ∈ s, M i)ᵀ = ∑ i ∈ s, (M i)ᵀ :=
  map_sum (transposeAddEquiv m n α) _ s


/-- `Matrix.transpose` as a `LinearMap` -/
@[simps apply]
def transposeLinearEquiv [Semiring R] [AddCommMonoid α] [Module R α] :
    Matrix m n α ≃ₗ[R] Matrix n m α :=
  { transposeAddEquiv m n α with map_smul' := transpose_smul }


@[simp]
theorem transposeLinearEquiv_symm [Semiring R] [AddCommMonoid α] [Module R α] :
    (transposeLinearEquiv m n R α).symm = transposeLinearEquiv n m R α :=
  rfl


/-- `Matrix.transpose` as a `RingEquiv` to the opposite ring -/
@[simps]
def transposeRingEquiv [AddCommMonoid α] [CommSemigroup α] [Fintype m] :
    Matrix m m α ≃+* (Matrix m m α)ᵐᵒᵖ :=
  { (transposeAddEquiv m m α).trans MulOpposite.opAddEquiv with
    toFun := fun M => MulOpposite.op Mᵀ
    invFun := fun M => M.unopᵀ
    map_mul' := fun M N =>
      (congr_arg MulOpposite.op (transpose_mul M N)).trans (MulOpposite.op_mul _ _)
    left_inv := fun M => transpose_transpose M
    right_inv := fun M => MulOpposite.unop_injective <| transpose_transpose M.unop }


@[simp]
theorem transpose_pow [CommSemiring α] [Fintype m] [DecidableEq m] (M : Matrix m m α) (k : ℕ) :
    (M ^ k)ᵀ = Mᵀ ^ k :=
  MulOpposite.op_injective <| map_pow (transposeRingEquiv m α) M k


theorem transpose_list_prod [CommSemiring α] [Fintype m] [DecidableEq m] (l : List (Matrix m m α)) :
    l.prodᵀ = (l.map transpose).reverse.prod :=
  (transposeRingEquiv m α).unop_map_list_prod l


/-- `Matrix.transpose` as an `AlgEquiv` to the opposite ring -/
@[simps]
def transposeAlgEquiv [CommSemiring R] [CommSemiring α] [Fintype m] [DecidableEq m] [Algebra R α] :
    Matrix m m α ≃ₐ[R] (Matrix m m α)ᵐᵒᵖ :=
  { (transposeAddEquiv m m α).trans MulOpposite.opAddEquiv,
    transposeRingEquiv m α with
    toFun := fun M => MulOpposite.op Mᵀ
    commutes' := fun r => by
      /-
        l : Type u_1
        m : Type u_2
        n : Type u_3
        o : Type u_4
        m' : o → Type u_5
        n' : o → Type u_6
        R : Type u_7
        S : Type u_8
        α : Type v
        β : Type w
        γ : Type u_9
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring α
        inst✝² : Fintype m
        inst✝¹ : DecidableEq m
        inst✝ : Algebra R α
        r : R
        ⊢ Eq ({ toFun := fun M => MulOpposite.op M.transpose, invFun := __src✝¹.invFun …
      -/
      simp only [algebraMap_eq_diagonal, diagonal_transpose, MulOpposite.algebraMap_apply] }
      /-
        🎉 no goals
      -/


