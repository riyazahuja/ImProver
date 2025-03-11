/-- A *Coxeter matrix* is a symmetric matrix of natural numbers whose diagonal entries are equal to
1 and whose off-diagonal entries are not equal to 1. -/
@[ext]
structure CoxeterMatrix (B : Type*) where
  /-- The underlying matrix of the Coxeter matrix. -/
  M : Matrix B B ℕ
  isSymm : M.IsSymm := by decide
  diagonal i : M i i = 1 := by decide
  off_diagonal i i' : i ≠ i' → M i i' ≠ 1 := by decide


/-- A Coxeter matrix can be coerced to a matrix. -/
instance : CoeFun (CoxeterMatrix B) fun _ ↦ (Matrix B B ℕ) := ⟨M⟩


theorem symmetric (i i' : B) : M i i' = M i' i := M.isSymm.apply i' i


/-- The Coxeter matrix formed by reindexing via the bijection `e : B ≃ B'`. -/
protected def reindex : CoxeterMatrix B' where
  M := Matrix.reindex e e M
  isSymm := M.isSymm.submatrix _
  diagonal i := M.diagonal (e.symm i)
  off_diagonal i i' h := M.off_diagonal (e.symm i) (e.symm i') (e.symm.injective.ne h)


theorem reindex_apply (i i' : B') : M.reindex e i i' = M (e.symm i) (e.symm i') := rfl


/-- The Coxeter matrix of type Aₙ.

The corresponding Coxeter-Dynkin diagram is:
```
    o --- o --- o ⬝ ⬝ ⬝ ⬝ o --- o
```
-/
def Aₙ : CoxeterMatrix (Fin n) where
  M := Matrix.of fun i j : Fin n ↦
    if i = j then 1
      else (if (j : ℕ) + 1 = i ∨ (i : ℕ) + 1 = j then 3 else 2)
               /-
                 B : Type u_1
                 B' : Type u_2
                 e : Equiv B B'
                 M : CoxeterMatrix B
                 n : Nat
                 ⊢ (Matrix.of fun i j => ite (Eq i j) 1 (ite (Or (Eq (HAdd.hAdd (↑j) 1) ↑i) (Eq …
               -/
  isSymm := by unfold Matrix.IsSymm; aesop
                                     /-
                                       🎉 no goals
                                     -/
                 /-
                   B : Type u_1
                   B' : Type u_2
                   e : Equiv B B'
                   M : CoxeterMatrix B
                   n : Nat
                   ⊢ ∀ (i : Fin n), Eq (Matrix.of (fun i j => ite (Eq i j) 1 (ite (Or (Eq (HAdd.h …
                 -/
  diagonal := by simp
                 /-
                   🎉 no goals
                 -/
                     /-
                       B : Type u_1
                       B' : Type u_2
                       e : Equiv B B'
                       M : CoxeterMatrix B
                       n : Nat
                       ⊢ ∀ (i i' : Fin n), Ne i i' → Ne (Matrix.of (fun i j => ite (Eq i j) 1 (ite (O …
                     -/
  off_diagonal := by aesop
                     /-
                       🎉 no goals
                     -/


/-- The Coxeter matrix of type Bₙ.

The corresponding Coxeter-Dynkin diagram is:
```
       4
    o --- o --- o ⬝ ⬝ ⬝ ⬝ o --- o
```
-/
def Bₙ : CoxeterMatrix (Fin n) where
  M := Matrix.of fun i j : Fin n ↦
    if i = j then 1
      else (if i = n - 1 ∧ j = n - 2 ∨ j = n - 1 ∧ i = n - 2 then 4
        else (if (j : ℕ) + 1 = i ∨ (i : ℕ) + 1 = j then 3 else 2))
               /-
                 B : Type u_1
                 B' : Type u_2
                 e : Equiv B B'
                 M : CoxeterMatrix B
                 n : Nat
                 ⊢ (Matrix.of fun i j => ite (Eq i j) 1 (ite (Or (And (Eq (↑i) (HSub.hSub n 1)) …
               -/
  isSymm := by unfold Matrix.IsSymm; aesop
                                     /-
                                       🎉 no goals
                                     -/
                 /-
                   B : Type u_1
                   B' : Type u_2
                   e : Equiv B B'
                   M : CoxeterMatrix B
                   n : Nat
                   ⊢ ∀ (i : Fin n), Eq (Matrix.of (fun i j => ite (Eq i j) 1 (ite (Or (And (Eq (↑ …
                 -/
  diagonal := by simp
                 /-
                   🎉 no goals
                 -/
                     /-
                       B : Type u_1
                       B' : Type u_2
                       e : Equiv B B'
                       M : CoxeterMatrix B
                       n : Nat
                       ⊢ ∀ (i i' : Fin n), Ne i i' → Ne (Matrix.of (fun i j => ite (Eq i j) 1 (ite (O …
                     -/
  off_diagonal := by aesop
                     /-
                       🎉 no goals
                     -/


/-- The Coxeter matrix of type Dₙ.

The corresponding Coxeter-Dynkin diagram is:
```
    o
     \
      o --- o ⬝ ⬝ ⬝ ⬝ o --- o
     /
    o
```
-/
def Dₙ : CoxeterMatrix (Fin n) where
  M := Matrix.of fun i j : Fin n ↦
    if i = j then 1
      else (if i = n - 1 ∧ j = n - 3 ∨ j = n - 1 ∧ i = n - 3 then 3
        else (if (j : ℕ) + 1 = i ∨ (i : ℕ) + 1 = j then 3 else 2))
               /-
                 B : Type u_1
                 B' : Type u_2
                 e : Equiv B B'
                 M : CoxeterMatrix B
                 n : Nat
                 ⊢ (Matrix.of fun i j => ite (Eq i j) 1 (ite (Or (And (Eq (↑i) (HSub.hSub n 1)) …
               -/
  isSymm := by unfold Matrix.IsSymm; aesop
                                     /-
                                       🎉 no goals
                                     -/
                 /-
                   B : Type u_1
                   B' : Type u_2
                   e : Equiv B B'
                   M : CoxeterMatrix B
                   n : Nat
                   ⊢ ∀ (i : Fin n), Eq (Matrix.of (fun i j => ite (Eq i j) 1 (ite (Or (And (Eq (↑ …
                 -/
  diagonal := by simp
                 /-
                   🎉 no goals
                 -/
                     /-
                       B : Type u_1
                       B' : Type u_2
                       e : Equiv B B'
                       M : CoxeterMatrix B
                       n : Nat
                       ⊢ ∀ (i i' : Fin n), Ne i i' → Ne (Matrix.of (fun i j => ite (Eq i j) 1 (ite (O …
                     -/
  off_diagonal := by aesop
                     /-
                       🎉 no goals
                     -/


/-- The Coxeter matrix of type I₂(m).

The corresponding Coxeter-Dynkin diagram is:
```
     m + 2
    o --- o
```
-/
def I₂ₘ (m : ℕ) : CoxeterMatrix (Fin 2) where
  M := Matrix.of fun i j => if i = j then 1 else m + 2
               /-
                 B : Type u_1
                 B' : Type u_2
                 e : Equiv B B'
                 M : CoxeterMatrix B
                 n m : Nat
                 ⊢ (Matrix.of fun i j => ite (Eq i j) 1 (HAdd.hAdd m 2)).IsSymm
               -/
  isSymm := by unfold Matrix.IsSymm; aesop
                                     /-
                                       🎉 no goals
                                     -/
                 /-
                   B : Type u_1
                   B' : Type u_2
                   e : Equiv B B'
                   M : CoxeterMatrix B
                   n m : Nat
                   ⊢ ∀ (i : Fin 2), Eq (Matrix.of (fun i j => ite (Eq i j) 1 (HAdd.hAdd m 2)) i i …
                 -/
  diagonal := by simp
                 /-
                   🎉 no goals
                 -/
                     /-
                       B : Type u_1
                       B' : Type u_2
                       e : Equiv B B'
                       M : CoxeterMatrix B
                       n m : Nat
                       ⊢ ∀ (i i' : Fin 2), Ne i i' → Ne (Matrix.of (fun i j => ite (Eq i j) 1 (HAdd.h …
                     -/
  off_diagonal := by simp
                     /-
                       🎉 no goals
                     -/


/-- The Coxeter matrix of type E₆.

The corresponding Coxeter-Dynkin diagram is:
```
                o
                |
    o --- o --- o --- o --- o
```
-/
def E₆ : CoxeterMatrix (Fin 6) where
  M := !![1, 2, 3, 2, 2, 2;
          2, 1, 2, 3, 2, 2;
          3, 2, 1, 3, 2, 2;
          2, 3, 3, 1, 3, 2;
          2, 2, 2, 3, 1, 3;
          2, 2, 2, 2, 3, 1]


/-- The Coxeter matrix of type E₇.

The corresponding Coxeter-Dynkin diagram is:
```
                o
                |
    o --- o --- o --- o --- o --- o
```
-/
def E₇ : CoxeterMatrix (Fin 7) where
  M := !![1, 2, 3, 2, 2, 2, 2;
          2, 1, 2, 3, 2, 2, 2;
          3, 2, 1, 3, 2, 2, 2;
          2, 3, 3, 1, 3, 2, 2;
          2, 2, 2, 3, 1, 3, 2;
          2, 2, 2, 2, 3, 1, 3;
          2, 2, 2, 2, 2, 3, 1]


/-- The Coxeter matrix of type E₈.

The corresponding Coxeter-Dynkin diagram is:
```
                o
                |
    o --- o --- o --- o --- o --- o --- o
```
-/
def E₈ : CoxeterMatrix (Fin 8) where
  M := !![1, 2, 3, 2, 2, 2, 2, 2;
          2, 1, 2, 3, 2, 2, 2, 2;
          3, 2, 1, 3, 2, 2, 2, 2;
          2, 3, 3, 1, 3, 2, 2, 2;
          2, 2, 2, 3, 1, 3, 2, 2;
          2, 2, 2, 2, 3, 1, 3, 2;
          2, 2, 2, 2, 2, 3, 1, 3;
          2, 2, 2, 2, 2, 2, 3, 1]


/-- The Coxeter matrix of type F₄.

The corresponding Coxeter-Dynkin diagram is:
```
             4
    o --- o --- o --- o
```
-/
def F₄ : CoxeterMatrix (Fin 4) where
  M := !![1, 3, 2, 2;
          3, 1, 4, 2;
          2, 4, 1, 3;
          2, 2, 3, 1]


/-- The Coxeter matrix of type G₂.

The corresponding Coxeter-Dynkin diagram is:
```
       6
    o --- o
```
-/
def G₂ : CoxeterMatrix (Fin 2) where
  M := !![1, 6;
          6, 1]


/-- The Coxeter matrix of type H₃.

The corresponding Coxeter-Dynkin diagram is:
```
       5
    o --- o --- o
```
-/
def H₃ : CoxeterMatrix (Fin 3) where
  M := !![1, 3, 2;
          3, 1, 5;
          2, 5, 1]


/-- The Coxeter matrix of type H₄.

The corresponding Coxeter-Dynkin diagram is:
```
       5
    o --- o --- o --- o
```
-/
def H₄ : CoxeterMatrix (Fin 4) where
  M := !![1, 3, 2, 2;
          3, 1, 3, 2;
          2, 3, 1, 5;
          2, 2, 5, 1]


