/-- `∀` with better defeq for `∀ x : Matrix (Fin m) (Fin n) α, P x`. -/
def Forall : ∀ {m n} (_ : Matrix (Fin m) (Fin n) α → Prop), Prop
  | 0, _, P => P (of ![])
  | _ + 1, _, P => FinVec.Forall fun r => Forall fun A => P (of (Matrix.vecCons r A))


/-- This can be used to prove
```lean
example (P : Matrix (Fin 2) (Fin 3) α → Prop) :
  (∀ x, P x) ↔ ∀ a b c d e f, P !![a, b, c; d, e, f] :=
(forall_iff _).symm
```
-/
theorem forall_iff : ∀ {m n} (P : Matrix (Fin m) (Fin n) α → Prop), Forall P ↔ ∀ x, P x
  | 0, _, _ => Iff.symm Fin.forall_fin_zero_pi
  | m + 1, n, P => by
    /-
      α : Type u_1
      m n : Nat
      P : Matrix (Fin (HAdd.hAdd m 1)) (Fin n) α → Prop
      ⊢ Iff (Matrix.Forall P) (∀ (x : Matrix (Fin (HAdd.hAdd m 1)) (Fin n) α), P x)
    -/
    simp only [Forall, FinVec.forall_iff, forall_iff]
    /-
      α : Type u_1
      m n : Nat
      P : Matrix (Fin (HAdd.hAdd m 1)) (Fin n) α → Prop
      ⊢ Iff (∀ (x : Fin n → α) (x_1 : Matrix (Fin m) (Fin n) α), P (Matrix.of (Matri …
    -/
    exact Iff.symm Fin.forall_fin_succ_pi
    /-
      🎉 no goals
    -/


/-- `∃` with better defeq for `∃ x : Matrix (Fin m) (Fin n) α, P x`. -/
def Exists : ∀ {m n} (_ : Matrix (Fin m) (Fin n) α → Prop), Prop
  | 0, _, P => P (of ![])
  | _ + 1, _, P => FinVec.Exists fun r => Exists fun A => P (of (Matrix.vecCons r A))


/-- This can be used to prove
```lean
example (P : Matrix (Fin 2) (Fin 3) α → Prop) :
  (∃ x, P x) ↔ ∃ a b c d e f, P !![a, b, c; d, e, f] :=
(exists_iff _).symm
```
-/
theorem exists_iff : ∀ {m n} (P : Matrix (Fin m) (Fin n) α → Prop), Exists P ↔ ∃ x, P x
  | 0, _, _ => Iff.symm Fin.exists_fin_zero_pi
  | m + 1, n, P => by
    /-
      α : Type u_1
      m n : Nat
      P : Matrix (Fin (HAdd.hAdd m 1)) (Fin n) α → Prop
      ⊢ Iff (Matrix.Exists P) (_root_.Exists fun x => P x)
    -/
    simp only [Exists, FinVec.exists_iff, exists_iff]
    /-
      α : Type u_1
      m n : Nat
      P : Matrix (Fin (HAdd.hAdd m 1)) (Fin n) α → Prop
      ⊢ Iff (_root_.Exists fun r => _root_.Exists fun A => P (Matrix.of (Matrix.vecC …
    -/
    exact Iff.symm Fin.exists_fin_succ_pi
    /-
      🎉 no goals
    -/


/-- `Matrix.transpose` with better defeq for `Fin` -/
def transposeᵣ : ∀ {m n}, Matrix (Fin m) (Fin n) α → Matrix (Fin n) (Fin m) α
  | _, 0, _ => of ![]
  | _, _ + 1, A =>
    of <| vecCons (FinVec.map (fun v : Fin _ → α => v 0) A) (transposeᵣ (A.submatrix id Fin.succ))


/-- This can be used to prove
```lean
example (a b c d : α) : transpose !![a, b; c, d] = !![a, c; b, d] := (transposeᵣ_eq _).symm
```
-/
@[simp]
theorem transposeᵣ_eq : ∀ {m n} (A : Matrix (Fin m) (Fin n) α), transposeᵣ A = transpose A
  | _, 0, _ => Subsingleton.elim _ _
  | m, n + 1, A =>
    Matrix.ext fun i j => by
      /-
        α : Type u_1
        m n : Nat
        A : Matrix (Fin m) (Fin (HAdd.hAdd n 1)) α
        i : Fin (HAdd.hAdd n 1)
        j : Fin m
        ⊢ Eq (A.transposeᵣ i j) (A.transpose i j)
      -/
      simp_rw [transposeᵣ, transposeᵣ_eq]
      /-
        α : Type u_1
        m n : Nat
        A : Matrix (Fin m) (Fin (HAdd.hAdd n 1)) α
        i : Fin (HAdd.hAdd n 1)
        j : Fin m
        ⊢ Eq (Matrix.of (Matrix.vecCons (FinVec.map (fun v => v 0) A) (A.submatrix id  …
      -/
      refine i.cases ?_ fun i => ?_
        /-
          case refine_1
          α : Type u_1
          m n : Nat
          A : Matrix (Fin m) (Fin (HAdd.hAdd n 1)) α
          i : Fin (HAdd.hAdd n 1)
          j : Fin m
          ⊢ Eq (Matrix.of (Matrix.vecCons (FinVec.map (fun v => v 0) A) (A.submatrix id  …
        -/
      · dsimp
        /-
          case refine_1
          α : Type u_1
          m n : Nat
          A : Matrix (Fin m) (Fin (HAdd.hAdd n 1)) α
          i : Fin (HAdd.hAdd n 1)
          j : Fin m
          ⊢ Eq (FinVec.map (fun v => v 0) A j) (A j 0)
        -/
        rw [FinVec.map_eq, Function.comp_apply]
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          α : Type u_1
          m n : Nat
          A : Matrix (Fin m) (Fin (HAdd.hAdd n 1)) α
          i✝ : Fin (HAdd.hAdd n 1)
          j : Fin m
          i : Fin n
          ⊢ Eq (Matrix.of (Matrix.vecCons (FinVec.map (fun v => v 0) A) (A.submatrix id  …
        -/
      · simp only [of_apply, Matrix.cons_val_succ]
        /-
          case refine_2
          α : Type u_1
          m n : Nat
          A : Matrix (Fin m) (Fin (HAdd.hAdd n 1)) α
          i✝ : Fin (HAdd.hAdd n 1)
          j : Fin m
          i : Fin n
          ⊢ Eq ((A.submatrix id Fin.succ).transpose i j) (A.transpose i.succ j)
        -/
        rfl
        /-
          🎉 no goals
        -/


/-- `dotProduct` with better defeq for `Fin` -/
def dotProductᵣ [Mul α] [Add α] [Zero α] {m} (a b : Fin m → α) : α :=
  FinVec.sum <| FinVec.seq (FinVec.map (· * ·) a) b


/-- This can be used to prove
```lean
example (a b c d : α) [Mul α] [AddCommMonoid α] :
  dot_product ![a, b] ![c, d] = a * c + b * d :=
(dot_productᵣ_eq _ _).symm
```
-/
@[simp]
theorem dotProductᵣ_eq [Mul α] [AddCommMonoid α] {m} (a b : Fin m → α) :
    dotProductᵣ a b = dotProduct a b := by
  simp_rw [dotProductᵣ, dotProduct, FinVec.sum_eq, FinVec.seq_eq, FinVec.map_eq,
      Function.comp_apply]


/-- `Matrix.mul` with better defeq for `Fin` -/
def mulᵣ [Mul α] [Add α] [Zero α] (A : Matrix (Fin l) (Fin m) α) (B : Matrix (Fin m) (Fin n) α) :
    Matrix (Fin l) (Fin n) α :=
  of <| FinVec.map (fun v₁ => FinVec.map (fun v₂ => dotProductᵣ v₁ v₂) Bᵀ) A


/-- This can be used to prove
```lean
example [AddCommMonoid α] [Mul α] (a₁₁ a₁₂ a₂₁ a₂₂ b₁₁ b₁₂ b₂₁ b₂₂ : α) :
  !![a₁₁, a₁₂;
     a₂₁, a₂₂] * !![b₁₁, b₁₂;
                    b₂₁, b₂₂] =
  !![a₁₁*b₁₁ + a₁₂*b₂₁, a₁₁*b₁₂ + a₁₂*b₂₂;
     a₂₁*b₁₁ + a₂₂*b₂₁, a₂₁*b₁₂ + a₂₂*b₂₂] :=
(mulᵣ_eq _ _).symm
```
-/
@[simp]
theorem mulᵣ_eq [Mul α] [AddCommMonoid α] (A : Matrix (Fin l) (Fin m) α)
    (B : Matrix (Fin m) (Fin n) α) : mulᵣ A B = A * B := by
  /-
    l m n : Nat
    α : Type u_1
    inst✝¹ : Mul α
    inst✝ : AddCommMonoid α
    A : Matrix (Fin l) (Fin m) α
    B : Matrix (Fin m) (Fin n) α
    ⊢ Eq (A.mulᵣ B) (HMul.hMul A B)
  -/
  simp [mulᵣ, Function.comp, Matrix.transpose]
  /-
    l m n : Nat
    α : Type u_1
    inst✝¹ : Mul α
    inst✝ : AddCommMonoid α
    A : Matrix (Fin l) (Fin m) α
    B : Matrix (Fin m) (Fin n) α
    ⊢ Eq (Matrix.of (Function.comp (fun v₁ => Function.comp (fun v₂ => dotProduct  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `Matrix.mulVec` with better defeq for `Fin` -/
def mulVecᵣ [Mul α] [Add α] [Zero α] (A : Matrix (Fin l) (Fin m) α) (v : Fin m → α) : Fin l → α :=
  FinVec.map (fun a => dotProductᵣ a v) A


/-- This can be used to prove
```lean
example [NonUnitalNonAssocSemiring α] (a₁₁ a₁₂ a₂₁ a₂₂ b₁ b₂ : α) :
  !![a₁₁, a₁₂;
     a₂₁, a₂₂] *ᵥ ![b₁, b₂] = ![a₁₁*b₁ + a₁₂*b₂, a₂₁*b₁ + a₂₂*b₂] :=
(mulVecᵣ_eq _ _).symm
```
-/
@[simp]
theorem mulVecᵣ_eq [NonUnitalNonAssocSemiring α] (A : Matrix (Fin l) (Fin m) α) (v : Fin m → α) :
    mulVecᵣ A v = A *ᵥ v := by
  /-
    l m : Nat
    α : Type u_1
    inst✝ : NonUnitalNonAssocSemiring α
    A : Matrix (Fin l) (Fin m) α
    v : Fin m → α
    ⊢ Eq (A.mulVecᵣ v) (A.mulVec v)
  -/
  simp [mulVecᵣ, Function.comp]
  /-
    l m : Nat
    α : Type u_1
    inst✝ : NonUnitalNonAssocSemiring α
    A : Matrix (Fin l) (Fin m) α
    v : Fin m → α
    ⊢ Eq (Function.comp (fun a => dotProduct a v) A) (A.mulVec v)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `Matrix.vecMul` with better defeq for `Fin` -/
def vecMulᵣ [Mul α] [Add α] [Zero α] (v : Fin l → α) (A : Matrix (Fin l) (Fin m) α) : Fin m → α :=
  FinVec.map (fun a => dotProductᵣ v a) Aᵀ


/-- This can be used to prove
```lean
example [NonUnitalNonAssocSemiring α] (a₁₁ a₁₂ a₂₁ a₂₂ b₁ b₂ : α) :
  ![b₁, b₂] ᵥ* !![a₁₁, a₁₂;
                       a₂₁, a₂₂] = ![b₁*a₁₁ + b₂*a₂₁, b₁*a₁₂ + b₂*a₂₂] :=
(vecMulᵣ_eq _ _).symm
```
-/
@[simp]
theorem vecMulᵣ_eq [NonUnitalNonAssocSemiring α] (v : Fin l → α) (A : Matrix (Fin l) (Fin m) α) :
    vecMulᵣ v A = v ᵥ* A := by
  /-
    l m : Nat
    α : Type u_1
    inst✝ : NonUnitalNonAssocSemiring α
    v : Fin l → α
    A : Matrix (Fin l) (Fin m) α
    ⊢ Eq (Matrix.vecMulᵣ v A) (Matrix.vecMul v A)
  -/
  simp [vecMulᵣ, Function.comp]
  /-
    l m : Nat
    α : Type u_1
    inst✝ : NonUnitalNonAssocSemiring α
    v : Fin l → α
    A : Matrix (Fin l) (Fin m) α
    ⊢ Eq (Function.comp (fun a => dotProduct v a) A.transpose) (Matrix.vecMul v A)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Expand `A` to `!![A 0 0, ...; ..., A m n]` -/
def etaExpand {m n} (A : Matrix (Fin m) (Fin n) α) : Matrix (Fin m) (Fin n) α :=
  Matrix.of (FinVec.etaExpand fun i => FinVec.etaExpand fun j => A i j)


/-- This can be used to prove
```lean
example (A : Matrix (Fin 2) (Fin 2) α) :
  A = !![A 0 0, A 0 1;
         A 1 0, A 1 1] :=
(etaExpand_eq _).symm
```
-/
theorem etaExpand_eq {m n} (A : Matrix (Fin m) (Fin n) α) : etaExpand A = A := by
  /-
    α : Type u_1
    m n : Nat
    A : Matrix (Fin m) (Fin n) α
    ⊢ Eq A.etaExpand A
  -/
  simp_rw [etaExpand, FinVec.etaExpand_eq, Matrix.of]
  -- This to be in the above `simp_rw` before https://github.com/leanprover/lean4/pull/2644
  /-
    α : Type u_1
    m n : Nat
    A : Matrix (Fin m) (Fin n) α
    ⊢ Eq ((Equiv.refl (Fin m → Fin n → α)) fun i j => A i j) A
  -/
  erw [Equiv.refl_apply]
  /-
    🎉 no goals
  -/


