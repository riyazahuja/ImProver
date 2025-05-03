theorem exp_diagonal (v : m → 𝔸) : exp 𝕂 (diagonal v) = diagonal (exp 𝕂 v) := by
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁷ : Fintype m
    inst✝⁶ : DecidableEq m
    inst✝⁵ : Field 𝕂
    inst✝⁴ : Ring 𝔸
    inst✝³ : TopologicalSpace 𝔸
    inst✝² : TopologicalRing 𝔸
    inst✝¹ : Algebra 𝕂 𝔸
    inst✝ : T2Space 𝔸
    v : m → 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂 (Matrix.diagonal v)) (Matrix.diagonal (NormedSpace.exp …
  -/
  simp_rw [exp_eq_tsum, diagonal_pow, ← diagonal_smul, ← diagonal_tsum]
  /-
    🎉 no goals
  -/


theorem exp_blockDiagonal (v : m → Matrix n n 𝔸) :
    exp 𝕂 (blockDiagonal v) = blockDiagonal (exp 𝕂 v) := by
  /-
    𝕂 : Type u_1
    m : Type u_2
    n : Type u_3
    𝔸 : Type u_5
    inst✝⁹ : Fintype m
    inst✝⁸ : DecidableEq m
    inst✝⁷ : Fintype n
    inst✝⁶ : DecidableEq n
    inst✝⁵ : Field 𝕂
    inst✝⁴ : Ring 𝔸
    inst✝³ : TopologicalSpace 𝔸
    inst✝² : TopologicalRing 𝔸
    inst✝¹ : Algebra 𝕂 𝔸
    inst✝ : T2Space 𝔸
    v : m → Matrix n n 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂 (Matrix.blockDiagonal v)) (Matrix.blockDiagonal (Norme …
  -/
  simp_rw [exp_eq_tsum, ← blockDiagonal_pow, ← blockDiagonal_smul, ← blockDiagonal_tsum]
  /-
    🎉 no goals
  -/


theorem exp_blockDiagonal' (v : ∀ i, Matrix (n' i) (n' i) 𝔸) :
    exp 𝕂 (blockDiagonal' v) = blockDiagonal' (exp 𝕂 v) := by
  /-
    𝕂 : Type u_1
    m : Type u_2
    n' : m → Type u_4
    𝔸 : Type u_5
    inst✝⁹ : Fintype m
    inst✝⁸ : DecidableEq m
    inst✝⁷ : (i : m) → Fintype (n' i)
    inst✝⁶ : (i : m) → DecidableEq (n' i)
    inst✝⁵ : Field 𝕂
    inst✝⁴ : Ring 𝔸
    inst✝³ : TopologicalSpace 𝔸
    inst✝² : TopologicalRing 𝔸
    inst✝¹ : Algebra 𝕂 𝔸
    inst✝ : T2Space 𝔸
    v : (i : m) → Matrix (n' i) (n' i) 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂 (Matrix.blockDiagonal' v)) (Matrix.blockDiagonal' (Nor …
  -/
  simp_rw [exp_eq_tsum, ← blockDiagonal'_pow, ← blockDiagonal'_smul, ← blockDiagonal'_tsum]
  /-
    🎉 no goals
  -/


theorem exp_conjTranspose [StarRing 𝔸] [ContinuousStar 𝔸] (A : Matrix m m 𝔸) :
    exp 𝕂 Aᴴ = (exp 𝕂 A)ᴴ :=
  (star_exp A).symm


theorem IsHermitian.exp [StarRing 𝔸] [ContinuousStar 𝔸] {A : Matrix m m 𝔸} (h : A.IsHermitian) :
    (exp 𝕂 A).IsHermitian :=
  (exp_conjTranspose _ _).symm.trans <| congr_arg _ h


theorem exp_transpose (A : Matrix m m 𝔸) : exp 𝕂 Aᵀ = (exp 𝕂 A)ᵀ := by
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁷ : Fintype m
    inst✝⁶ : DecidableEq m
    inst✝⁵ : Field 𝕂
    inst✝⁴ : CommRing 𝔸
    inst✝³ : TopologicalSpace 𝔸
    inst✝² : TopologicalRing 𝔸
    inst✝¹ : Algebra 𝕂 𝔸
    inst✝ : T2Space 𝔸
    A : Matrix m m 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂 A.transpose) (NormedSpace.exp 𝕂 A).transpose
  -/
  simp_rw [exp_eq_tsum, transpose_tsum, transpose_smul, transpose_pow]
  /-
    🎉 no goals
  -/


theorem IsSymm.exp {A : Matrix m m 𝔸} (h : A.IsSymm) : (exp 𝕂 A).IsSymm :=
  (exp_transpose _ _).symm.trans <| congr_arg _ h


nonrec theorem exp_add_of_commute (A B : Matrix m m 𝔸) (h : Commute A B) :
    exp 𝕂 (A + B) = exp 𝕂 A * exp 𝕂 B := by
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    A B : Matrix m m 𝔸
    h : Commute A B
    ⊢ Eq (NormedSpace.exp 𝕂 (HAdd.hAdd A B)) (HMul.hMul (NormedSpace.exp 𝕂 A) (Nor …
  -/
  letI : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    A B : Matrix m m 𝔸
    h : Commute A B
    this : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    ⊢ Eq (NormedSpace.exp 𝕂 (HAdd.hAdd A B)) (HMul.hMul (NormedSpace.exp 𝕂 A) (Nor …
  -/
  letI : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    A B : Matrix m m 𝔸
    h : Commute A B
    this✝ : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    this : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
    ⊢ Eq (NormedSpace.exp 𝕂 (HAdd.hAdd A B)) (HMul.hMul (NormedSpace.exp 𝕂 A) (Nor …
  -/
  letI : NormedAlgebra 𝕂 (Matrix m m 𝔸) := Matrix.linftyOpNormedAlgebra
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    A B : Matrix m m 𝔸
    h : Commute A B
    this✝¹ : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    this✝ : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
    this : NormedAlgebra 𝕂 (Matrix m m 𝔸) := Matrix.linftyOpNormedAlgebra
    ⊢ Eq (NormedSpace.exp 𝕂 (HAdd.hAdd A B)) (HMul.hMul (NormedSpace.exp 𝕂 A) (Nor …
  -/
  exact exp_add_of_commute h
  /-
    🎉 no goals
  -/


nonrec theorem exp_sum_of_commute {ι} (s : Finset ι) (f : ι → Matrix m m 𝔸)
    (h : (s : Set ι).Pairwise (Commute on f)) :
    exp 𝕂 (∑ i ∈ s, f i) =
      s.noncommProd (fun i => exp 𝕂 (f i)) fun _ hi _ hj _ => (h.of_refl hi hj).exp 𝕂 := by
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    ι : Type u_6
    s : Finset ι
    f : ι → Matrix m m 𝔸
    h : (↑s).Pairwise (Function.onFun Commute f)
    ⊢ Eq (NormedSpace.exp 𝕂 (s.sum fun i => f i)) (s.noncommProd (fun i => NormedS …
  -/
  letI : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    ι : Type u_6
    s : Finset ι
    f : ι → Matrix m m 𝔸
    h : (↑s).Pairwise (Function.onFun Commute f)
    this : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    ⊢ Eq (NormedSpace.exp 𝕂 (s.sum fun i => f i)) (s.noncommProd (fun i => NormedS …
  -/
  letI : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    ι : Type u_6
    s : Finset ι
    f : ι → Matrix m m 𝔸
    h : (↑s).Pairwise (Function.onFun Commute f)
    this✝ : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    this : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
    ⊢ Eq (NormedSpace.exp 𝕂 (s.sum fun i => f i)) (s.noncommProd (fun i => NormedS …
  -/
  letI : NormedAlgebra 𝕂 (Matrix m m 𝔸) := Matrix.linftyOpNormedAlgebra
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    ι : Type u_6
    s : Finset ι
    f : ι → Matrix m m 𝔸
    h : (↑s).Pairwise (Function.onFun Commute f)
    this✝¹ : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    this✝ : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
    this : NormedAlgebra 𝕂 (Matrix m m 𝔸) := Matrix.linftyOpNormedAlgebra
    ⊢ Eq (NormedSpace.exp 𝕂 (s.sum fun i => f i)) (s.noncommProd (fun i => NormedS …
  -/
  exact exp_sum_of_commute s f h
  /-
    🎉 no goals
  -/


nonrec theorem exp_nsmul (n : ℕ) (A : Matrix m m 𝔸) : exp 𝕂 (n • A) = exp 𝕂 A ^ n := by
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    n : Nat
    A : Matrix m m 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂 (HSMul.hSMul n A)) (HPow.hPow (NormedSpace.exp 𝕂 A) n)
  -/
  letI : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    n : Nat
    A : Matrix m m 𝔸
    this : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    ⊢ Eq (NormedSpace.exp 𝕂 (HSMul.hSMul n A)) (HPow.hPow (NormedSpace.exp 𝕂 A) n)
  -/
  letI : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    n : Nat
    A : Matrix m m 𝔸
    this✝ : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    this : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
    ⊢ Eq (NormedSpace.exp 𝕂 (HSMul.hSMul n A)) (HPow.hPow (NormedSpace.exp 𝕂 A) n)
  -/
  letI : NormedAlgebra 𝕂 (Matrix m m 𝔸) := Matrix.linftyOpNormedAlgebra
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    n : Nat
    A : Matrix m m 𝔸
    this✝¹ : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    this✝ : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
    this : NormedAlgebra 𝕂 (Matrix m m 𝔸) := Matrix.linftyOpNormedAlgebra
    ⊢ Eq (NormedSpace.exp 𝕂 (HSMul.hSMul n A)) (HPow.hPow (NormedSpace.exp 𝕂 A) n)
  -/
  exact exp_nsmul n A
  /-
    🎉 no goals
  -/


nonrec theorem isUnit_exp (A : Matrix m m 𝔸) : IsUnit (exp 𝕂 A) := by
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    A : Matrix m m 𝔸
    ⊢ IsUnit (NormedSpace.exp 𝕂 A)
  -/
  letI : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    A : Matrix m m 𝔸
    this : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    ⊢ IsUnit (NormedSpace.exp 𝕂 A)
  -/
  letI : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    A : Matrix m m 𝔸
    this✝ : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    this : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
    ⊢ IsUnit (NormedSpace.exp 𝕂 A)
  -/
  letI : NormedAlgebra 𝕂 (Matrix m m 𝔸) := Matrix.linftyOpNormedAlgebra
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    A : Matrix m m 𝔸
    this✝¹ : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    this✝ : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
    this : NormedAlgebra 𝕂 (Matrix m m 𝔸) := Matrix.linftyOpNormedAlgebra
    ⊢ IsUnit (NormedSpace.exp 𝕂 A)
  -/
  exact isUnit_exp _ A
  /-
    🎉 no goals
  -/


nonrec theorem exp_units_conj (U : (Matrix m m 𝔸)ˣ) (A : Matrix m m 𝔸) :
    exp 𝕂 (U * A * U⁻¹) = U * exp 𝕂 A * U⁻¹ := by
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    U : Units (Matrix m m 𝔸)
    A : Matrix m m 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂 (HMul.hMul (HMul.hMul (↑U) A) ↑(Inv.inv U))) (HMul.hMu …
  -/
  letI : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    U : Units (Matrix m m 𝔸)
    A : Matrix m m 𝔸
    this : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    ⊢ Eq (NormedSpace.exp 𝕂 (HMul.hMul (HMul.hMul (↑U) A) ↑(Inv.inv U))) (HMul.hMu …
  -/
  letI : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    U : Units (Matrix m m 𝔸)
    A : Matrix m m 𝔸
    this✝ : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    this : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
    ⊢ Eq (NormedSpace.exp 𝕂 (HMul.hMul (HMul.hMul (↑U) A) ↑(Inv.inv U))) (HMul.hMu …
  -/
  letI : NormedAlgebra 𝕂 (Matrix m m 𝔸) := Matrix.linftyOpNormedAlgebra
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    U : Units (Matrix m m 𝔸)
    A : Matrix m m 𝔸
    this✝¹ : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    this✝ : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
    this : NormedAlgebra 𝕂 (Matrix m m 𝔸) := Matrix.linftyOpNormedAlgebra
    ⊢ Eq (NormedSpace.exp 𝕂 (HMul.hMul (HMul.hMul (↑U) A) ↑(Inv.inv U))) (HMul.hMu …
  -/
  exact exp_units_conj _ U A
  /-
    🎉 no goals
  -/


theorem exp_units_conj' (U : (Matrix m m 𝔸)ˣ) (A : Matrix m m 𝔸) :
    exp 𝕂 (U⁻¹ * A * U) = U⁻¹ * exp 𝕂 A * U :=
  exp_units_conj 𝕂 U⁻¹ A


theorem exp_neg (A : Matrix m m 𝔸) : exp 𝕂 (-A) = (exp 𝕂 A)⁻¹ := by
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedCommRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    A : Matrix m m 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂 (Neg.neg A)) (Inv.inv (NormedSpace.exp 𝕂 A))
  -/
  rw [nonsing_inv_eq_ring_inverse]
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedCommRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    A : Matrix m m 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂 (Neg.neg A)) (Ring.inverse (NormedSpace.exp 𝕂 A))
  -/
  letI : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedCommRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    A : Matrix m m 𝔸
    this : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    ⊢ Eq (NormedSpace.exp 𝕂 (Neg.neg A)) (Ring.inverse (NormedSpace.exp 𝕂 A))
  -/
  letI : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedCommRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    A : Matrix m m 𝔸
    this✝ : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    this : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
    ⊢ Eq (NormedSpace.exp 𝕂 (Neg.neg A)) (Ring.inverse (NormedSpace.exp 𝕂 A))
  -/
  letI : NormedAlgebra 𝕂 (Matrix m m 𝔸) := Matrix.linftyOpNormedAlgebra
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedCommRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    A : Matrix m m 𝔸
    this✝¹ : SeminormedRing (Matrix m m 𝔸) := Matrix.linftyOpSemiNormedRing
    this✝ : NormedRing (Matrix m m 𝔸) := Matrix.linftyOpNormedRing
    this : NormedAlgebra 𝕂 (Matrix m m 𝔸) := Matrix.linftyOpNormedAlgebra
    ⊢ Eq (NormedSpace.exp 𝕂 (Neg.neg A)) (Ring.inverse (NormedSpace.exp 𝕂 A))
  -/
  exact (Ring.inverse_exp _ A).symm
  /-
    🎉 no goals
  -/


theorem exp_zsmul (z : ℤ) (A : Matrix m m 𝔸) : exp 𝕂 (z • A) = exp 𝕂 A ^ z := by
  /-
    𝕂 : Type u_1
    m : Type u_2
    𝔸 : Type u_5
    inst✝⁵ : RCLike 𝕂
    inst✝⁴ : Fintype m
    inst✝³ : DecidableEq m
    inst✝² : NormedCommRing 𝔸
    inst✝¹ : NormedAlgebra 𝕂 𝔸
    inst✝ : CompleteSpace 𝔸
    z : Int
    A : Matrix m m 𝔸
    ⊢ Eq (NormedSpace.exp 𝕂 (HSMul.hSMul z A)) (HPow.hPow (NormedSpace.exp 𝕂 A) z)
  -/
  obtain ⟨n, rfl | rfl⟩ := z.eq_nat_or_neg
    /-
      case intro.inl
      𝕂 : Type u_1
      m : Type u_2
      𝔸 : Type u_5
      inst✝⁵ : RCLike 𝕂
      inst✝⁴ : Fintype m
      inst✝³ : DecidableEq m
      inst✝² : NormedCommRing 𝔸
      inst✝¹ : NormedAlgebra 𝕂 𝔸
      inst✝ : CompleteSpace 𝔸
      A : Matrix m m 𝔸
      n : Nat
      ⊢ Eq (NormedSpace.exp 𝕂 (HSMul.hSMul (↑n) A)) (HPow.hPow (NormedSpace.exp 𝕂 A) …
    -/
  · rw [zpow_natCast, natCast_zsmul, exp_nsmul]
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      𝕂 : Type u_1
      m : Type u_2
      𝔸 : Type u_5
      inst✝⁵ : RCLike 𝕂
      inst✝⁴ : Fintype m
      inst✝³ : DecidableEq m
      inst✝² : NormedCommRing 𝔸
      inst✝¹ : NormedAlgebra 𝕂 𝔸
      inst✝ : CompleteSpace 𝔸
      A : Matrix m m 𝔸
      n : Nat
      ⊢ Eq (NormedSpace.exp 𝕂 (HSMul.hSMul (Neg.neg ↑n) A)) (HPow.hPow (NormedSpace. …
    -/
  · have : IsUnit (exp 𝕂 A).det := (Matrix.isUnit_iff_isUnit_det _).mp (isUnit_exp _ _)
    /-
      case intro.inr
      𝕂 : Type u_1
      m : Type u_2
      𝔸 : Type u_5
      inst✝⁵ : RCLike 𝕂
      inst✝⁴ : Fintype m
      inst✝³ : DecidableEq m
      inst✝² : NormedCommRing 𝔸
      inst✝¹ : NormedAlgebra 𝕂 𝔸
      inst✝ : CompleteSpace 𝔸
      A : Matrix m m 𝔸
      n : Nat
      this : IsUnit (NormedSpace.exp 𝕂 A).det
      ⊢ Eq (NormedSpace.exp 𝕂 (HSMul.hSMul (Neg.neg ↑n) A)) (HPow.hPow (NormedSpace. …
    -/
    rw [Matrix.zpow_neg this, zpow_natCast, neg_smul, exp_neg, natCast_zsmul, exp_nsmul]
    /-
      🎉 no goals
    -/


theorem exp_conj (U : Matrix m m 𝔸) (A : Matrix m m 𝔸) (hy : IsUnit U) :
    exp 𝕂 (U * A * U⁻¹) = U * exp 𝕂 A * U⁻¹ :=
  let ⟨u, hu⟩ := hy
          /-
            𝕂 : Type u_1
            m : Type u_2
            𝔸 : Type u_5
            inst✝⁵ : RCLike 𝕂
            inst✝⁴ : Fintype m
            inst✝³ : DecidableEq m
            inst✝² : NormedCommRing 𝔸
            inst✝¹ : NormedAlgebra 𝕂 𝔸
            inst✝ : CompleteSpace 𝔸
            U A : Matrix m m 𝔸
            hy : IsUnit U
            u : Units (Matrix m m 𝔸)
            hu : Eq (↑u) U
            ⊢ Eq (NormedSpace.exp 𝕂 (HMul.hMul (HMul.hMul (↑u) A) (Inv.inv ↑u))) (HMul.hMu …
          -/
  hu ▸ by simpa only [Matrix.coe_units_inv] using exp_units_conj 𝕂 u A
          /-
            🎉 no goals
          -/


theorem exp_conj' (U : Matrix m m 𝔸) (A : Matrix m m 𝔸) (hy : IsUnit U) :
    exp 𝕂 (U⁻¹ * A * U) = U⁻¹ * exp 𝕂 A * U :=
  let ⟨u, hu⟩ := hy
          /-
            𝕂 : Type u_1
            m : Type u_2
            𝔸 : Type u_5
            inst✝⁵ : RCLike 𝕂
            inst✝⁴ : Fintype m
            inst✝³ : DecidableEq m
            inst✝² : NormedCommRing 𝔸
            inst✝¹ : NormedAlgebra 𝕂 𝔸
            inst✝ : CompleteSpace 𝔸
            U A : Matrix m m 𝔸
            hy : IsUnit U
            u : Units (Matrix m m 𝔸)
            hu : Eq (↑u) U
            ⊢ Eq (NormedSpace.exp 𝕂 (HMul.hMul (HMul.hMul (Inv.inv ↑u) A) ↑u)) (HMul.hMul  …
          -/
  hu ▸ by simpa only [Matrix.coe_units_inv] using exp_units_conj' 𝕂 u A
          /-
            🎉 no goals
          -/


