/-- A quaternion basis contains the information both sufficient and necessary to construct an
`R`-algebra homomorphism from `ℍ[R,c₁,c₂]` to `A`; or equivalently, a surjective
`R`-algebra homomorphism from `ℍ[R,c₁,c₂]` to an `R`-subalgebra of `A`.

Note that for definitional convenience, `k` is provided as a field even though `i_mul_j` fully
determines it. -/
structure Basis {R : Type*} (A : Type*) [CommRing R] [Ring A] [Algebra R A] (c₁ c₂ : R) where
  (i j k : A)
  i_mul_i : i * i = c₁ • (1 : A)
  j_mul_j : j * j = c₂ • (1 : A)
  i_mul_j : i * j = k
  j_mul_i : j * i = -k


/-- Since `k` is redundant, it is not necessary to show `q₁.k = q₂.k` when showing `q₁ = q₂`. -/
@[ext]
protected theorem ext ⦃q₁ q₂ : Basis A c₁ c₂⦄ (hi : q₁.i = q₂.i) (hj : q₁.j = q₂.j) : q₁ = q₂ := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q₁ q₂ : QuaternionAlgebra.Basis A c₁ c₂
    hi : Eq q₁.i q₂.i
    hj : Eq q₁.j q₂.j
    ⊢ Eq q₁ q₂
  -/
  cases q₁; rename_i q₁_i_mul_j _
  /-
    case mk
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q₂ : QuaternionAlgebra.Basis A c₁ c₂
    i✝ j✝ k✝ : A
    i_mul_i✝ : Eq (HMul.hMul i✝ i✝) (HSMul.hSMul c₁ 1)
    j_mul_j✝ : Eq (HMul.hMul j✝ j✝) (HSMul.hSMul c₂ 1)
    q₁_i_mul_j : Eq (HMul.hMul i✝ j✝) k✝
    j_mul_i✝ : Eq (HMul.hMul j✝ i✝) (Neg.neg k✝)
    hi : Eq { i := i✝, j := j✝, k := k✝, i_mul_i := i_mul_i✝, j_mul_j := j_mul_j✝, …
    hj : Eq { i := i✝, j := j✝, k := k✝, i_mul_i := i_mul_i✝, j_mul_j := j_mul_j✝, …
    ⊢ Eq { i := i✝, j := j✝, k := k✝, i_mul_i := i_mul_i✝, j_mul_j := j_mul_j✝, i_ …
  -/
  cases q₂; rename_i q₂_i_mul_j _
  /-
    case mk.mk
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    i✝¹ j✝¹ k✝¹ : A
    i_mul_i✝¹ : Eq (HMul.hMul i✝¹ i✝¹) (HSMul.hSMul c₁ 1)
    j_mul_j✝¹ : Eq (HMul.hMul j✝¹ j✝¹) (HSMul.hSMul c₂ 1)
    q₁_i_mul_j : Eq (HMul.hMul i✝¹ j✝¹) k✝¹
    j_mul_i✝¹ : Eq (HMul.hMul j✝¹ i✝¹) (Neg.neg k✝¹)
    i✝ j✝ k✝ : A
    i_mul_i✝ : Eq (HMul.hMul i✝ i✝) (HSMul.hSMul c₁ 1)
    j_mul_j✝ : Eq (HMul.hMul j✝ j✝) (HSMul.hSMul c₂ 1)
    q₂_i_mul_j : Eq (HMul.hMul i✝ j✝) k✝
    j_mul_i✝ : Eq (HMul.hMul j✝ i✝) (Neg.neg k✝)
    hi : Eq { i := i✝¹, j := j✝¹, k := k✝¹, i_mul_i := i_mul_i✝¹, j_mul_j := j_mul …
    hj : Eq { i := i✝¹, j := j✝¹, k := k✝¹, i_mul_i := i_mul_i✝¹, j_mul_j := j_mul …
    ⊢ Eq { i := i✝¹, j := j✝¹, k := k✝¹, i_mul_i := i_mul_i✝¹, j_mul_j := j_mul_j✝ …
  -/
  congr
  /-
    case mk.mk.e_k
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    i✝¹ j✝¹ k✝¹ : A
    i_mul_i✝¹ : Eq (HMul.hMul i✝¹ i✝¹) (HSMul.hSMul c₁ 1)
    j_mul_j✝¹ : Eq (HMul.hMul j✝¹ j✝¹) (HSMul.hSMul c₂ 1)
    q₁_i_mul_j : Eq (HMul.hMul i✝¹ j✝¹) k✝¹
    j_mul_i✝¹ : Eq (HMul.hMul j✝¹ i✝¹) (Neg.neg k✝¹)
    i✝ j✝ k✝ : A
    i_mul_i✝ : Eq (HMul.hMul i✝ i✝) (HSMul.hSMul c₁ 1)
    j_mul_j✝ : Eq (HMul.hMul j✝ j✝) (HSMul.hSMul c₂ 1)
    q₂_i_mul_j : Eq (HMul.hMul i✝ j✝) k✝
    j_mul_i✝ : Eq (HMul.hMul j✝ i✝) (Neg.neg k✝)
    hi : Eq { i := i✝¹, j := j✝¹, k := k✝¹, i_mul_i := i_mul_i✝¹, j_mul_j := j_mul …
    hj : Eq { i := i✝¹, j := j✝¹, k := k✝¹, i_mul_i := i_mul_i✝¹, j_mul_j := j_mul …
    ⊢ Eq k✝¹ k✝
  -/
  rw [← q₁_i_mul_j, ← q₂_i_mul_j]
  /-
    case mk.mk.e_k
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    i✝¹ j✝¹ k✝¹ : A
    i_mul_i✝¹ : Eq (HMul.hMul i✝¹ i✝¹) (HSMul.hSMul c₁ 1)
    j_mul_j✝¹ : Eq (HMul.hMul j✝¹ j✝¹) (HSMul.hSMul c₂ 1)
    q₁_i_mul_j : Eq (HMul.hMul i✝¹ j✝¹) k✝¹
    j_mul_i✝¹ : Eq (HMul.hMul j✝¹ i✝¹) (Neg.neg k✝¹)
    i✝ j✝ k✝ : A
    i_mul_i✝ : Eq (HMul.hMul i✝ i✝) (HSMul.hSMul c₁ 1)
    j_mul_j✝ : Eq (HMul.hMul j✝ j✝) (HSMul.hSMul c₂ 1)
    q₂_i_mul_j : Eq (HMul.hMul i✝ j✝) k✝
    j_mul_i✝ : Eq (HMul.hMul j✝ i✝) (Neg.neg k✝)
    hi : Eq { i := i✝¹, j := j✝¹, k := k✝¹, i_mul_i := i_mul_i✝¹, j_mul_j := j_mul …
    hj : Eq { i := i✝¹, j := j✝¹, k := k✝¹, i_mul_i := i_mul_i✝¹, j_mul_j := j_mul …
    ⊢ Eq (HMul.hMul i✝¹ j✝¹) (HMul.hMul i✝ j✝)
  -/
  congr
  /-
    🎉 no goals
  -/


/-- There is a natural quaternionic basis for the `QuaternionAlgebra`. -/
@[simps i j k]
protected def self : Basis ℍ[R,c₁,c₂] c₁ c₂ where
  i := ⟨0, 1, 0, 0⟩
                /-
                  R : Type u_1
                  A : Type u_2
                  B : Type u_3
                  inst✝⁴ : CommRing R
                  inst✝³ : Ring A
                  inst✝² : Ring B
                  inst✝¹ : Algebra R A
                  inst✝ : Algebra R B
                  c₁ c₂ : R
                  ⊢ Eq (HMul.hMul { re := 0, imI := 1, imJ := 0, imK := 0 } { re := 0, imI := 1, …
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
  i_mul_i := by ext <;> simp
                        /-
                          🎉 no goals
                        -/
  j := ⟨0, 0, 1, 0⟩
                /-
                  R : Type u_1
                  A : Type u_2
                  B : Type u_3
                  inst✝⁴ : CommRing R
                  inst✝³ : Ring A
                  inst✝² : Ring B
                  inst✝¹ : Algebra R A
                  inst✝ : Algebra R B
                  c₁ c₂ : R
                  ⊢ Eq (HMul.hMul { re := 0, imI := 0, imJ := 1, imK := 0 } { re := 0, imI := 0, …
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
  j_mul_j := by ext <;> simp
                        /-
                          🎉 no goals
                        -/
  k := ⟨0, 0, 0, 1⟩
                /-
                  R : Type u_1
                  A : Type u_2
                  B : Type u_3
                  inst✝⁴ : CommRing R
                  inst✝³ : Ring A
                  inst✝² : Ring B
                  inst✝¹ : Algebra R A
                  inst✝ : Algebra R B
                  c₁ c₂ : R
                  ⊢ Eq (HMul.hMul { re := 0, imI := 1, imJ := 0, imK := 0 } { re := 0, imI := 0, …
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
  i_mul_j := by ext <;> simp
                        /-
                          🎉 no goals
                        -/
                /-
                  R : Type u_1
                  A : Type u_2
                  B : Type u_3
                  inst✝⁴ : CommRing R
                  inst✝³ : Ring A
                  inst✝² : Ring B
                  inst✝¹ : Algebra R A
                  inst✝ : Algebra R B
                  c₁ c₂ : R
                  ⊢ Eq (HMul.hMul { re := 0, imI := 0, imJ := 1, imK := 0 } { re := 0, imI := 1, …
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
  j_mul_i := by ext <;> simp
                        /-
                          🎉 no goals
                        -/


instance : Inhabited (Basis ℍ[R,c₁,c₂] c₁ c₂) :=
  ⟨Basis.self R⟩


@[simp]
theorem i_mul_k : q.i * q.k = c₁ • q.j := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    ⊢ Eq (HMul.hMul q.i q.k) (HSMul.hSMul c₁ q.j)
  -/
  rw [← i_mul_j, ← mul_assoc, i_mul_i, smul_mul_assoc, one_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem k_mul_i : q.k * q.i = -c₁ • q.j := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    ⊢ Eq (HMul.hMul q.k q.i) (HSMul.hSMul (Neg.neg c₁) q.j)
  -/
  rw [← i_mul_j, mul_assoc, j_mul_i, mul_neg, i_mul_k, neg_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem k_mul_j : q.k * q.j = c₂ • q.i := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    ⊢ Eq (HMul.hMul q.k q.j) (HSMul.hSMul c₂ q.i)
  -/
  rw [← i_mul_j, mul_assoc, j_mul_j, mul_smul_comm, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem j_mul_k : q.j * q.k = -c₂ • q.i := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    ⊢ Eq (HMul.hMul q.j q.k) (HSMul.hSMul (Neg.neg c₂) q.i)
  -/
  rw [← i_mul_j, ← mul_assoc, j_mul_i, neg_mul, k_mul_j, neg_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem k_mul_k : q.k * q.k = -((c₁ * c₂) • (1 : A)) := by
  rw [← i_mul_j, mul_assoc, ← mul_assoc q.j _ _, j_mul_i, ← i_mul_j, ← mul_assoc, mul_neg, ←
    mul_assoc, i_mul_i, smul_mul_assoc, one_mul, neg_mul, smul_mul_assoc, j_mul_j, smul_smul]


/-- Intermediate result used to define `QuaternionAlgebra.Basis.liftHom`. -/
def lift (x : ℍ[R,c₁,c₂]) : A :=
  algebraMap R _ x.re + x.imI • q.i + x.imJ • q.j + x.imK • q.k


                                                      /-
                                                        R : Type u_1
                                                        A : Type u_2
                                                        inst✝² : CommRing R
                                                        inst✝¹ : Ring A
                                                        inst✝ : Algebra R A
                                                        c₁ c₂ : R
                                                        q : QuaternionAlgebra.Basis A c₁ c₂
                                                        ⊢ Eq (q.lift 0) 0
                                                      -/
theorem lift_zero : q.lift (0 : ℍ[R,c₁,c₂]) = 0 := by simp [lift]
                                                      /-
                                                        🎉 no goals
                                                      -/


                                                     /-
                                                       R : Type u_1
                                                       A : Type u_2
                                                       inst✝² : CommRing R
                                                       inst✝¹ : Ring A
                                                       inst✝ : Algebra R A
                                                       c₁ c₂ : R
                                                       q : QuaternionAlgebra.Basis A c₁ c₂
                                                       ⊢ Eq (q.lift 1) 1
                                                     -/
theorem lift_one : q.lift (1 : ℍ[R,c₁,c₂]) = 1 := by simp [lift]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem lift_add (x y : ℍ[R,c₁,c₂]) : q.lift (x + y) = q.lift x + q.lift y := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    x y : QuaternionAlgebra R c₁ c₂
    ⊢ Eq (q.lift (HAdd.hAdd x y)) (HAdd.hAdd (q.lift x) (q.lift y))
  -/
  simp only [lift, add_re, map_add, add_imI, add_smul, add_imJ, add_imK]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    x y : QuaternionAlgebra R c₁ c₂
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ((algebraMap R A) x.re) ((alg …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


theorem lift_mul (x y : ℍ[R,c₁,c₂]) : q.lift (x * y) = q.lift x * q.lift y := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    x y : QuaternionAlgebra R c₁ c₂
    ⊢ Eq (q.lift (HMul.hMul x y)) (HMul.hMul (q.lift x) (q.lift y))
  -/
  simp only [lift, Algebra.algebraMap_eq_smul_one]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    x y : QuaternionAlgebra R c₁ c₂
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (HMul.hMul x y).re 1) (HSMu …
  -/
  simp_rw [add_mul, mul_add, smul_mul_assoc, mul_smul_comm, one_mul, mul_one, smul_smul]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    x y : QuaternionAlgebra R c₁ c₂
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (HMul.hMul x y).re 1) (HSMu …
  -/
  simp only [i_mul_i, j_mul_j, i_mul_j, j_mul_i, i_mul_k, k_mul_i, k_mul_j, j_mul_k, k_mul_k]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    x y : QuaternionAlgebra R c₁ c₂
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (HMul.hMul x y).re 1) (HSMu …
  -/
  simp only [smul_smul, smul_neg, sub_eq_add_neg, add_smul, ← add_assoc, mul_neg, neg_smul]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    x y : QuaternionAlgebra R c₁ c₂
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (HMul.hMul x y).re 1) (HSMu …
  -/
  simp only [mul_right_comm _ _ (c₁ * c₂), mul_comm _ (c₁ * c₂)]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    x y : QuaternionAlgebra R c₁ c₂
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (HMul.hMul x y).re 1) (HSMu …
  -/
  simp only [mul_comm _ c₁, mul_right_comm _ _ c₁]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    x y : QuaternionAlgebra R c₁ c₂
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (HMul.hMul x y).re 1) (HSMu …
  -/
  simp only [mul_comm _ c₂, mul_right_comm _ _ c₂]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    x y : QuaternionAlgebra R c₁ c₂
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (HMul.hMul x y).re 1) (HSMu …
  -/
  simp only [← mul_comm c₁ c₂, ← mul_assoc]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    x y : QuaternionAlgebra R c₁ c₂
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (HMul.hMul x y).re 1) (HSMu …
  -/
  simp only [mul_re, sub_eq_add_neg, add_smul, neg_smul, mul_imI, ← add_assoc, mul_imJ, mul_imK]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    x y : QuaternionAlgebra R c₁ c₂
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.h …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


theorem lift_smul (r : R) (x : ℍ[R,c₁,c₂]) : q.lift (r • x) = r • q.lift x := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    c₁ c₂ : R
    q : QuaternionAlgebra.Basis A c₁ c₂
    r : R
    x : QuaternionAlgebra R c₁ c₂
    ⊢ Eq (q.lift (HSMul.hSMul r x)) (HSMul.hSMul r (q.lift x))
  -/
  simp [lift, mul_smul, ← Algebra.smul_def]
  /-
    🎉 no goals
  -/


/-- A `QuaternionAlgebra.Basis` implies an `AlgHom` from the quaternions. -/
@[simps!]
def liftHom : ℍ[R,c₁,c₂] →ₐ[R] A :=
  AlgHom.mk'
    { toFun := q.lift
      map_zero' := q.lift_zero
      map_one' := q.lift_one
      map_add' := q.lift_add
      map_mul' := q.lift_mul } q.lift_smul


/-- Transform a `QuaternionAlgebra.Basis` through an `AlgHom`. -/
@[simps i j k]
def compHom (F : A →ₐ[R] B) : Basis B c₁ c₂ where
  i := F q.i
                /-
                  R : Type u_1
                  A : Type u_2
                  B : Type u_3
                  inst✝⁴ : CommRing R
                  inst✝³ : Ring A
                  inst✝² : Ring B
                  inst✝¹ : Algebra R A
                  inst✝ : Algebra R B
                  c₁ c₂ : R
                  q : QuaternionAlgebra.Basis A c₁ c₂
                  F : AlgHom R A B
                  ⊢ Eq (HMul.hMul (F q.i) (F q.i)) (HSMul.hSMul c₁ 1)
                -/
  i_mul_i := by rw [← map_mul, q.i_mul_i, map_smul, map_one]
                /-
                  🎉 no goals
                -/
  j := F q.j
                /-
                  R : Type u_1
                  A : Type u_2
                  B : Type u_3
                  inst✝⁴ : CommRing R
                  inst✝³ : Ring A
                  inst✝² : Ring B
                  inst✝¹ : Algebra R A
                  inst✝ : Algebra R B
                  c₁ c₂ : R
                  q : QuaternionAlgebra.Basis A c₁ c₂
                  F : AlgHom R A B
                  ⊢ Eq (HMul.hMul (F q.j) (F q.j)) (HSMul.hSMul c₂ 1)
                -/
  j_mul_j := by rw [← map_mul, q.j_mul_j, map_smul, map_one]
                /-
                  🎉 no goals
                -/
  k := F q.k
                /-
                  R : Type u_1
                  A : Type u_2
                  B : Type u_3
                  inst✝⁴ : CommRing R
                  inst✝³ : Ring A
                  inst✝² : Ring B
                  inst✝¹ : Algebra R A
                  inst✝ : Algebra R B
                  c₁ c₂ : R
                  q : QuaternionAlgebra.Basis A c₁ c₂
                  F : AlgHom R A B
                  ⊢ Eq (HMul.hMul (F q.i) (F q.j)) (F q.k)
                -/
  i_mul_j := by rw [← map_mul, q.i_mul_j]
                /-
                  🎉 no goals
                -/
                /-
                  R : Type u_1
                  A : Type u_2
                  B : Type u_3
                  inst✝⁴ : CommRing R
                  inst✝³ : Ring A
                  inst✝² : Ring B
                  inst✝¹ : Algebra R A
                  inst✝ : Algebra R B
                  c₁ c₂ : R
                  q : QuaternionAlgebra.Basis A c₁ c₂
                  F : AlgHom R A B
                  ⊢ Eq (HMul.hMul (F q.j) (F q.i)) (Neg.neg (F q.k))
                -/
  j_mul_i := by rw [← map_mul, q.j_mul_i, map_neg]
                /-
                  🎉 no goals
                -/


/-- A quaternionic basis on `A` is equivalent to a map from the quaternion algebra to `A`. -/
@[simps]
def lift : Basis A c₁ c₂ ≃ (ℍ[R,c₁,c₂] →ₐ[R] A) where
  toFun := Basis.liftHom
  invFun := (Basis.self R).compHom
                   /-
                     R : Type u_1
                     A : Type u_2
                     B : Type u_3
                     inst✝⁴ : CommRing R
                     inst✝³ : Ring A
                     inst✝² : Ring B
                     inst✝¹ : Algebra R A
                     inst✝ : Algebra R B
                     c₁ c₂ : R
                     q : QuaternionAlgebra.Basis A c₁ c₂
                     ⊢ Eq ((QuaternionAlgebra.Basis.self R).compHom q.liftHom) q
                   -/
                           /-
                             🎉 no goals
                           -/
  left_inv q := by ext <;> simp [Basis.lift]
                           /-
                             🎉 no goals
                           -/
  right_inv F := by
    /-
      R : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : Ring A
      inst✝² : Ring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      c₁ c₂ : R
      F : AlgHom R (QuaternionAlgebra R c₁ c₂) A
      ⊢ Eq ((QuaternionAlgebra.Basis.self R).compHom F).liftHom F
    -/
    ext
    /-
      case H
      R : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : Ring A
      inst✝² : Ring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      c₁ c₂ : R
      F : AlgHom R (QuaternionAlgebra R c₁ c₂) A
      x✝ : QuaternionAlgebra R c₁ c₂
      ⊢ Eq (((QuaternionAlgebra.Basis.self R).compHom F).liftHom x✝) (F x✝)
    -/
    dsimp [Basis.lift]
    /-
      case H
      R : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : Ring A
      inst✝² : Ring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      c₁ c₂ : R
      F : AlgHom R (QuaternionAlgebra R c₁ c₂) A
      x✝ : QuaternionAlgebra R c₁ c₂
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ((algebraMap R A) x✝.re) (HSMul.hSMul x✝ …
    -/
    rw [← F.commutes]
    simp only [← F.commutes, ← map_smul, ← map_add, mk_add_mk, smul_mk, smul_zero,
      algebraMap_eq]
    /-
      case H
      R : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : Ring A
      inst✝² : Ring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      c₁ c₂ : R
      F : AlgHom R (QuaternionAlgebra R c₁ c₂) A
      x✝ : QuaternionAlgebra R c₁ c₂
      ⊢ Eq (F { re := HAdd.hAdd (HAdd.hAdd (HAdd.hAdd x✝.re 0) 0) 0, imI := HAdd.hAd …
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
    congr <;> simp
              /-
                🎉 no goals
              -/


/-- Two `R`-algebra morphisms from a quaternion algebra are equal if they agree on `i` and `j`. -/
@[ext]
theorem hom_ext ⦃f g : ℍ[R,c₁,c₂] →ₐ[R] A⦄
    (hi : f (Basis.self R).i = g (Basis.self R).i) (hj : f (Basis.self R).j = g (Basis.self R).j) :
    f = g :=
  lift.symm.injective <| Basis.ext hi hj


/-- Two `R`-algebra morphisms from the quaternions are equal if they agree on `i` and `j`. -/
@[ext]
theorem hom_ext ⦃f g : ℍ[R] →ₐ[R] A⦄
    (hi : f (Basis.self R).i = g (Basis.self R).i) (hj : f (Basis.self R).j = g (Basis.self R).j) :
    f = g :=
  QuaternionAlgebra.hom_ext hi hj


