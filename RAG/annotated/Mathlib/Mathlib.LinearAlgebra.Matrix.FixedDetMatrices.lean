/--The subtype of matrices with fixed determinant `m`. -/
def FixedDetMatrix (m : R) := { A : Matrix n n R // A.det = m }


/--Extensionality theorem for `FixedDetMatrix` with respect to the underlying matrix, not
entriwise. -/
lemma ext' {m : R} {A B : FixedDetMatrix n R m} (h : A.1 = B.1) : A = B := by
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommRing R
    m : R
    A B : FixedDetMatrix n R m
    h : Eq ↑A ↑B
    ⊢ Eq A B
  -/
  cases A; cases B
  /-
    case mk.mk
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommRing R
    m : R
    val✝¹ : Matrix n n R
    property✝¹ : Eq val✝¹.det m
    val✝ : Matrix n n R
    property✝ : Eq val✝.det m
    h : Eq ↑⟨val✝¹, property✝¹⟩ ↑⟨val✝, property✝⟩
    ⊢ Eq ⟨val✝¹, property✝¹⟩ ⟨val✝, property✝⟩
  -/
  congr
  /-
    🎉 no goals
  -/


@[ext]
lemma ext {m : R} {A B : FixedDetMatrix n R m} (h : ∀ i j , A.1 i j = B.1 i j) : A = B := by
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommRing R
    m : R
    A B : FixedDetMatrix n R m
    h : ∀ (i j : n), Eq (↑A i j) (↑B i j)
    ⊢ Eq A B
  -/
  apply ext'
  /-
    case h
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommRing R
    m : R
    A B : FixedDetMatrix n R m
    h : ∀ (i j : n), Eq (↑A i j) (↑B i j)
    ⊢ Eq ↑A ↑B
  -/
  ext i j
  /-
    case h.a
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommRing R
    m : R
    A B : FixedDetMatrix n R m
    h : ∀ (i j : n), Eq (↑A i j) (↑B i j)
    i j : n
    ⊢ Eq (↑A i j) (↑B i j)
  -/
  apply h
  /-
    🎉 no goals
  -/


instance (m : R) : SMul (SpecialLinearGroup n R) (FixedDetMatrix n R m) where
                           /-
                             n : Type u_1
                             inst✝² : DecidableEq n
                             inst✝¹ : Fintype n
                             R : Type u_2
                             inst✝ : CommRing R
                             m : R
                             g : Matrix.SpecialLinearGroup n R
                             A : FixedDetMatrix n R m
                             ⊢ Eq (HMul.hMul ↑g ↑A).det m
                           -/
  smul g A := ⟨g * A.1, by simp only [det_mul, SpecialLinearGroup.det_coe, A.2, one_mul]⟩
                           /-
                             🎉 no goals
                           -/


lemma smul_def (m : R) (g : SpecialLinearGroup n R) (A : (FixedDetMatrix n R m)) :
                         /-
                           n : Type u_1
                           inst✝² : DecidableEq n
                           inst✝¹ : Fintype n
                           R : Type u_2
                           inst✝ : CommRing R
                           m : R
                           g : Matrix.SpecialLinearGroup n R
                           A : FixedDetMatrix n R m
                           ⊢ Eq (HMul.hMul ↑g ↑A).det m
                         -/
    g • A = ⟨g * A.1, by simp only [det_mul, SpecialLinearGroup.det_coe, A.2, one_mul]⟩ :=
                         /-
                           🎉 no goals
                         -/
  rfl


instance (m : R) : MulAction (SpecialLinearGroup n R) (FixedDetMatrix n R m) where
                   /-
                     n : Type u_1
                     inst✝² : DecidableEq n
                     inst✝¹ : Fintype n
                     R : Type u_2
                     inst✝ : CommRing R
                     m : R
                     b : FixedDetMatrix n R m
                     ⊢ Eq (HSMul.hSMul 1 b) b
                   -/
  one_smul b := by rw [smul_def]; simp only [coe_one, one_mul, Subtype.coe_eta]
                                  /-
                                    🎉 no goals
                                  -/
                       /-
                         n : Type u_1
                         inst✝² : DecidableEq n
                         inst✝¹ : Fintype n
                         R : Type u_2
                         inst✝ : CommRing R
                         m : R
                         x y : Matrix.SpecialLinearGroup n R
                         b : FixedDetMatrix n R m
                         ⊢ Eq (HSMul.hSMul (HMul.hMul x y) b) (HSMul.hSMul x (HSMul.hSMul y b))
                       -/
  mul_smul x y b := by simp_rw [smul_def, ← mul_assoc, coe_mul]
                       /-
                         🎉 no goals
                       -/


lemma smul_coe (m : R) (g : SpecialLinearGroup n R) (A : FixedDetMatrix n R m) :
    (g • A).1 = g * A.1 := by
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommRing R
    m : R
    g : Matrix.SpecialLinearGroup n R
    A : FixedDetMatrix n R m
    ⊢ Eq (↑(HSMul.hSMul g A)) (HMul.hMul ↑g ↑A)
  -/
  rw [smul_def]
  /-
    🎉 no goals
  -/


local notation:1024 "Δ" m : 1024 => (FixedDetMatrix (Fin 2) ℤ m)


/--Set of representatives for the orbits under `S` and `T`. -/
def reps (m : ℤ) : Set (Δ m) :=
  {A : Δ m | (A.1 1 0) = 0 ∧ 0 < A.1 0 0 ∧ 0 ≤ A.1 0 1 ∧ |(A.1 0 1)| < |(A.1 1 1)|}


/--Reduction step for matrices in `Δ m` which moves the matrices towards `reps`.-/
def reduceStep (A : Δ m) : Δ m := S • (T ^ (-(A.1 0 0 / A.1 1 0))) • A


private lemma reduce_aux {A : Δ m} (h : (A.1 1 0) ≠ 0) :
    |((reduceStep A).1 1 0)| < |(A.1 1 0)| := by
  suffices ((reduceStep A).1 1 0) = A.1 0 0 % A.1 1 0 by
    rw [this, abs_eq_self.mpr (Int.emod_nonneg (A.1 0 0) h)]
    exact Int.emod_lt (A.1 0 0) h
  /-
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    h : Ne (↑A 1 0) 0
    ⊢ Eq (↑(FixedDetMatrices.reduceStep A) 1 0) (HMod.hMod (↑A 0 0) (↑A 1 0))
  -/
  simp_rw [Int.emod_def, sub_eq_add_neg, reduceStep, smul_coe, coe_T_zpow, S]
  /-
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    h : Ne (↑A 1 0) 0
    ⊢ Eq (HMul.hMul (Matrix.of (Matrix.vecCons (Matrix.vecCons 0 (Matrix.vecCons ( …
  -/
  norm_num [vecMul, vecHead, vecTail, mul_comm]
  /-
    🎉 no goals
  -/


/--Reduction lemma for integral FixedDetMatrices. -/
@[elab_as_elim]
def reduce_rec {C : Δ m → Sort*}
    (base : ∀ A : Δ m, (A.1 1 0) = 0 → C A)
    (step : ∀ A : Δ m, (A.1 1 0) ≠ 0 → C (reduceStep A) → C A) :
    ∀ A, C A := fun A => by
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommRing R
    m : Int
    C : FixedDetMatrix (Fin 2) Int m → Sort u_3
    base : (A : FixedDetMatrix (Fin 2) Int m) → Eq (↑A 1 0) 0 → C A
    step : (A : FixedDetMatrix (Fin 2) Int m) → Ne (↑A 1 0) 0 → C (FixedDetMatrice …
    A : FixedDetMatrix (Fin 2) Int m
    ⊢ C A
  -/
  by_cases h : (A.1 1 0) = 0
    /-
      case pos
      n : Type u_1
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      R : Type u_2
      inst✝ : CommRing R
      m : Int
      C : FixedDetMatrix (Fin 2) Int m → Sort u_3
      base : (A : FixedDetMatrix (Fin 2) Int m) → Eq (↑A 1 0) 0 → C A
      step : (A : FixedDetMatrix (Fin 2) Int m) → Ne (↑A 1 0) 0 → C (FixedDetMatrice …
      A : FixedDetMatrix (Fin 2) Int m
      h : Eq (↑A 1 0) 0
      ⊢ C A
    -/
  · exact base _ h
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Type u_1
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      R : Type u_2
      inst✝ : CommRing R
      m : Int
      C : FixedDetMatrix (Fin 2) Int m → Sort u_3
      base : (A : FixedDetMatrix (Fin 2) Int m) → Eq (↑A 1 0) 0 → C A
      step : (A : FixedDetMatrix (Fin 2) Int m) → Ne (↑A 1 0) 0 → C (FixedDetMatrice …
      A : FixedDetMatrix (Fin 2) Int m
      h : Not (Eq (↑A 1 0) 0)
      ⊢ C A
    -/
  · exact step A h (reduce_rec base step (reduceStep A))
    /-
      🎉 no goals
    -/
  termination_by A => Int.natAbs (A.1 1 0)
  decreasing_by
    zify
    exact reduce_aux h


/--Map from `Δ m → Δ m` which reduces a FixedDetMatrix towards a representative element in reps. -/
def reduce : Δ m → Δ m := fun A ↦
  if (A.1 1 0) = 0 then
    if 0 < A.1 0 0 then (T ^ (-(A.1 0 1 / A.1 1 1))) • A else
      (T ^ (-(-A.1 0 1 / -A.1 1 1))) • (S • (S • A)) --the -/- don't cancel with ℤ divs.
  else
    reduce (reduceStep A)
  termination_by b => Int.natAbs (b.1 1 0)
  decreasing_by
    next a h =>
    zify
    exact reduce_aux h


lemma reduce_of_pos {A : Δ m} (hc : (A.1 1 0) = 0) (ha : 0 < A.1 0 0) :
    reduce A = (T ^ (-(A.1 0 1 / A.1 1 1))) • A := by
  /-
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    hc : Eq (↑A 1 0) 0
    ha : LT.lt 0 (↑A 0 0)
    ⊢ Eq (FixedDetMatrices.reduce A) (HSMul.hSMul (HPow.hPow ModularGroup.T (Neg.n …
  -/
  rw [reduce]
  /-
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    hc : Eq (↑A 1 0) 0
    ha : LT.lt 0 (↑A 0 0)
    ⊢ Eq (ite (Eq (↑A 1 0) 0) (ite (LT.lt 0 (↑A 0 0)) (HSMul.hSMul (HPow.hPow Modu …
  -/
  simp only [zpow_neg, Int.ediv_neg, neg_neg] at *
  /-
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    hc : Eq (↑A 1 0) 0
    ha : LT.lt 0 (↑A 0 0)
    ⊢ Eq (ite (Eq (↑A 1 0) 0) (ite (LT.lt 0 (↑A 0 0)) (HSMul.hSMul (Inv.inv (HPow. …
  -/
  simp_rw [if_pos hc, if_pos ha]
  /-
    🎉 no goals
  -/


lemma reduce_of_not_pos {A : Δ m} (hc : (A.1 1 0) = 0) (ha : ¬ 0 < A.1 0 0) :
    reduce A = (T ^ (-(-A.1 0 1 / -A.1 1 1))) • (S • (S • A)) := by
  /-
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    hc : Eq (↑A 1 0) 0
    ha : Not (LT.lt 0 (↑A 0 0))
    ⊢ Eq (FixedDetMatrices.reduce A) (HSMul.hSMul (HPow.hPow ModularGroup.T (Neg.n …
  -/
  rw [reduce]
  /-
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    hc : Eq (↑A 1 0) 0
    ha : Not (LT.lt 0 (↑A 0 0))
    ⊢ Eq (ite (Eq (↑A 1 0) 0) (ite (LT.lt 0 (↑A 0 0)) (HSMul.hSMul (HPow.hPow Modu …
  -/
  simp only [abs_eq_zero, zpow_neg, Int.ediv_neg, neg_neg] at *
  /-
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    hc : Eq (↑A 1 0) 0
    ha : Not (LT.lt 0 (↑A 0 0))
    ⊢ Eq (ite (Eq (↑A 1 0) 0) (ite (LT.lt 0 (↑A 0 0)) (HSMul.hSMul (Inv.inv (HPow. …
  -/
  simp_rw [if_pos hc, if_neg ha]
  /-
    🎉 no goals
  -/


@[simp]
lemma reduce_reduceStep {A : Δ m} (hc : (A.1 1 0) ≠ 0) :
    reduce (reduceStep A) = reduce A := by
  /-
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    hc : Ne (↑A 1 0) 0
    ⊢ Eq (FixedDetMatrices.reduce (FixedDetMatrices.reduceStep A)) (FixedDetMatric …
  -/
  symm
  /-
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    hc : Ne (↑A 1 0) 0
    ⊢ Eq (FixedDetMatrices.reduce A) (FixedDetMatrices.reduce (FixedDetMatrices.re …
  -/
  rw [reduce, if_neg hc]
  /-
    🎉 no goals
  -/


private lemma A_c_eq_zero {A : Δ m} (ha : A.1 1 0 = 0) : A.1 0 0 * A.1 1 1 = m := by
  /-
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    ha : Eq (↑A 1 0) 0
    ⊢ Eq (HMul.hMul (↑A 0 0) (↑A 1 1)) m
  -/
  simpa only [det_fin_two, ha, mul_zero, sub_zero] using A.2
  /-
    🎉 no goals
  -/


private lemma A_d_ne_zero {A : Δ m} (ha : A.1 1 0 = 0) (hm : m ≠ 0) : A.1 1 1 ≠ 0 :=
  right_ne_zero_of_mul (A_c_eq_zero (ha) ▸ hm)


private lemma A_a_ne_zero {A : Δ m} (ha : A.1 1 0 = 0) (hm : m ≠ 0) : A.1 0 0 ≠ 0 :=
  left_ne_zero_of_mul (A_c_eq_zero ha ▸ hm)


/--An auxiliary result bounding the size of the entries of the representatives in `reps`. -/
lemma reps_entries_le_m' {A : Δ m} (h : A ∈ reps m) (i j : Fin 2) :
    A.1 i j ∈ Finset.Icc (-|m|) |m| := by
  /-
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    h : Membership.mem (FixedDetMatrices.reps m) A
    i j : Fin 2
    ⊢ Membership.mem (Finset.Icc (Neg.neg (abs m)) (abs m)) (↑A i j)
  -/
  suffices |A.1 i j| ≤ |m| from Finset.mem_Icc.mpr <| abs_le.mp this
  /-
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    h : Membership.mem (FixedDetMatrices.reps m) A
    i j : Fin 2
    ⊢ LE.le (abs (↑A i j)) (abs m)
  -/
  obtain ⟨h10, h00, h01, h11⟩ := h
  /-
    case intro.intro.intro
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    i j : Fin 2
    h10 : Eq (↑A 1 0) 0
    h00 : LT.lt 0 (↑A 0 0)
    h01 : LE.le 0 (↑A 0 1)
    h11 : LT.lt (abs (↑A 0 1)) (abs (↑A 1 1))
    ⊢ LE.le (abs (↑A i j)) (abs m)
  -/
  have h1 : 0 < |A.1 1 1| := (abs_nonneg _).trans_lt h11
  /-
    case intro.intro.intro
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    i j : Fin 2
    h10 : Eq (↑A 1 0) 0
    h00 : LT.lt 0 (↑A 0 0)
    h01 : LE.le 0 (↑A 0 1)
    h11 : LT.lt (abs (↑A 0 1)) (abs (↑A 1 1))
    h1 : LT.lt 0 (abs (↑A 1 1))
    ⊢ LE.le (abs (↑A i j)) (abs m)
  -/
  have h2 : 0 < |A.1 0 0| := abs_pos.mpr h00.ne'
  /-
    case intro.intro.intro
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    i j : Fin 2
    h10 : Eq (↑A 1 0) 0
    h00 : LT.lt 0 (↑A 0 0)
    h01 : LE.le 0 (↑A 0 1)
    h11 : LT.lt (abs (↑A 0 1)) (abs (↑A 1 1))
    h1 : LT.lt 0 (abs (↑A 1 1))
    h2 : LT.lt 0 (abs (↑A 0 0))
    ⊢ LE.le (abs (↑A i j)) (abs m)
  -/
  fin_cases i <;> fin_cases j
    /-
      case intro.intro.intro.«0».«0»
      m : Int
      A : FixedDetMatrix (Fin 2) Int m
      h10 : Eq (↑A 1 0) 0
      h00 : LT.lt 0 (↑A 0 0)
      h01 : LE.le 0 (↑A 0 1)
      h11 : LT.lt (abs (↑A 0 1)) (abs (↑A 1 1))
      h1 : LT.lt 0 (abs (↑A 1 1))
      h2 : LT.lt 0 (abs (↑A 0 0))
      ⊢ LE.le (abs (↑A ((fun i => i) ⟨0, ⋯⟩) ((fun i => i) ⟨0, ⋯⟩))) (abs m)
    -/
  · simpa only [← abs_mul, A_c_eq_zero h10] using (le_mul_iff_one_le_right h2).mpr h1
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.«0».«1»
      m : Int
      A : FixedDetMatrix (Fin 2) Int m
      h10 : Eq (↑A 1 0) 0
      h00 : LT.lt 0 (↑A 0 0)
      h01 : LE.le 0 (↑A 0 1)
      h11 : LT.lt (abs (↑A 0 1)) (abs (↑A 1 1))
      h1 : LT.lt 0 (abs (↑A 1 1))
      h2 : LT.lt 0 (abs (↑A 0 0))
      ⊢ LE.le (abs (↑A ((fun i => i) ⟨0, ⋯⟩) ((fun i => i) ⟨1, ⋯⟩))) (abs m)
    -/
  · simpa only [← abs_mul, A_c_eq_zero h10] using h11.le.trans (le_mul_of_one_le_left h1.le h2)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.«1».«0»
      m : Int
      A : FixedDetMatrix (Fin 2) Int m
      h10 : Eq (↑A 1 0) 0
      h00 : LT.lt 0 (↑A 0 0)
      h01 : LE.le 0 (↑A 0 1)
      h11 : LT.lt (abs (↑A 0 1)) (abs (↑A 1 1))
      h1 : LT.lt 0 (abs (↑A 1 1))
      h2 : LT.lt 0 (abs (↑A 0 0))
      ⊢ LE.le (abs (↑A ((fun i => i) ⟨1, ⋯⟩) ((fun i => i) ⟨0, ⋯⟩))) (abs m)
    -/
  · simp_all
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.«1».«1»
      m : Int
      A : FixedDetMatrix (Fin 2) Int m
      h10 : Eq (↑A 1 0) 0
      h00 : LT.lt 0 (↑A 0 0)
      h01 : LE.le 0 (↑A 0 1)
      h11 : LT.lt (abs (↑A 0 1)) (abs (↑A 1 1))
      h1 : LT.lt 0 (abs (↑A 1 1))
      h2 : LT.lt 0 (abs (↑A 0 0))
      ⊢ LE.le (abs (↑A ((fun i => i) ⟨1, ⋯⟩) ((fun i => i) ⟨1, ⋯⟩))) (abs m)
    -/
  · simpa only [← abs_mul, A_c_eq_zero h10] using (le_mul_iff_one_le_left h1).mpr h2
    /-
      🎉 no goals
    -/


@[simp]
lemma reps_zero_empty : reps 0 = ∅ := by
  /-
    ⊢ Eq (FixedDetMatrices.reps 0) EmptyCollection.emptyCollection
  -/
  rw [reps, Set.eq_empty_iff_forall_not_mem]
  /-
    ⊢ ∀ (x : FixedDetMatrix (Fin 2) Int 0), Not (Membership.mem (setOf fun A => An …
  -/
  rintro A ⟨h₁, h₂, -, h₄⟩
  /-
    case intro.intro.intro
    A : FixedDetMatrix (Fin 2) Int 0
    h₁ : Eq (↑A 1 0) 0
    h₂ : LT.lt 0 (↑A 0 0)
    h₄ : LT.lt (abs (↑A 0 1)) (abs (↑A 1 1))
    ⊢ False
  -/
  suffices |A.1 0 1| < 0 by linarith [abs_nonneg (A.1 0 1)]
  /-
    case intro.intro.intro
    A : FixedDetMatrix (Fin 2) Int 0
    h₁ : Eq (↑A 1 0) 0
    h₂ : LT.lt 0 (↑A 0 0)
    h₄ : LT.lt (abs (↑A 0 1)) (abs (↑A 1 1))
    ⊢ LT.lt (abs (↑A 0 1)) 0
  -/
  have := A_c_eq_zero h₁
  /-
    case intro.intro.intro
    A : FixedDetMatrix (Fin 2) Int 0
    h₁ : Eq (↑A 1 0) 0
    h₂ : LT.lt 0 (↑A 0 0)
    h₄ : LT.lt (abs (↑A 0 1)) (abs (↑A 1 1))
    this : Eq (HMul.hMul (↑A 0 0) (↑A 1 1)) 0
    ⊢ LT.lt (abs (↑A 0 1)) 0
  -/
  simp_all [h₂.ne']
  /-
    🎉 no goals
  -/


noncomputable instance repsFintype (k : ℤ) : Fintype (reps k) := by
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommRing R
    m k : Int
    ⊢ Fintype ↑(FixedDetMatrices.reps k)
  -/
  let H := Finset.Icc (-|k|) |k|
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommRing R
    m k : Int
    H : Finset Int := Finset.Icc (Neg.neg (abs k)) (abs k)
    ⊢ Fintype ↑(FixedDetMatrices.reps k)
  -/
  let H4 := Fin 2 → Fin 2 → H
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommRing R
    m k : Int
    H : Finset Int := Finset.Icc (Neg.neg (abs k)) (abs k)
    H4 : Type := Fin 2 → Fin 2 → Subtype fun x => Membership.mem H x
    ⊢ Fintype ↑(FixedDetMatrices.reps k)
  -/
  apply Fintype.ofInjective (β := H4) (f := fun M i j ↦ ⟨M.1.1 i j, reps_entries_le_m' M.2 i j⟩)
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommRing R
    m k : Int
    H : Finset Int := Finset.Icc (Neg.neg (abs k)) (abs k)
    H4 : Type := Fin 2 → Fin 2 → Subtype fun x => Membership.mem H x
    ⊢ Function.Injective fun M i j => ⟨↑↑M i j, ⋯⟩
  -/
  intro M N h
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommRing R
    m k : Int
    H : Finset Int := Finset.Icc (Neg.neg (abs k)) (abs k)
    H4 : Type := Fin 2 → Fin 2 → Subtype fun x => Membership.mem H x
    M N : ↑(FixedDetMatrices.reps k)
    h : Eq ((fun M i j => ⟨↑↑M i j, ⋯⟩) M) ((fun M i j => ⟨↑↑M i j, ⋯⟩) N)
    ⊢ Eq M N
  -/
  ext i j
  /-
    case a.h
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type u_2
    inst✝ : CommRing R
    m k : Int
    H : Finset Int := Finset.Icc (Neg.neg (abs k)) (abs k)
    H4 : Type := Fin 2 → Fin 2 → Subtype fun x => Membership.mem H x
    M N : ↑(FixedDetMatrices.reps k)
    h : Eq ((fun M i j => ⟨↑↑M i j, ⋯⟩) M) ((fun M i j => ⟨↑↑M i j, ⋯⟩) N)
    i j : Fin 2
    ⊢ Eq (↑↑M i j) (↑↑N i j)
  -/
  simpa only [Subtype.mk.injEq] using congrFun₂ h i j
  /-
    🎉 no goals
  -/


@[simp]
lemma S_smul_four (A : Δ m) : S • S • S • S • A = A := by
  /-
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    ⊢ Eq (HSMul.hSMul ModularGroup.S (HSMul.hSMul ModularGroup.S (HSMul.hSMul Modu …
  -/
  simp only [smul_def, ← mul_assoc, S_mul_S_eq, neg_mul, one_mul, mul_neg, neg_neg, Subtype.coe_eta]
  /-
    🎉 no goals
  -/


@[simp]
lemma T_S_rel_smul (A : Δ m) : S • S • S • T • S • T • S • A = T⁻¹ • A := by
  /-
    m : Int
    A : FixedDetMatrix (Fin 2) Int m
    ⊢ Eq (HSMul.hSMul ModularGroup.S (HSMul.hSMul ModularGroup.S (HSMul.hSMul Modu …
  -/
  simp_rw [← T_S_rel, ← smul_assoc]
  /-
    🎉 no goals
  -/


lemma reduce_mem_reps {m : ℤ} (hm : m ≠ 0) (A : Δ m) : reduce A ∈ reps m := by
  induction A using reduce_rec with
  | step A h1 h2 => simpa only [reduce_reduceStep h1] using h2
  | base A h =>
    have hd := A_d_ne_zero h hm
    by_cases h1 : 0 < A.1 0 0
    · simp only [reduce_of_pos h h1]
      have h2 := Int.emod_def (A.1 0 1) (A.1 1 1)
      have h4 := Int.ediv_mul_le (A.1 0 1) hd
      set n : ℤ := A.1 0 1 / A.1 1 1
      have h3 := Int.emod_lt (A.1 0 1) hd
      rw [← abs_eq_self.mpr <| Int.emod_nonneg _ hd] at h3
      simp only [smul_def, Fin.isValue, coe_T_zpow]
      suffices A.1 1 0 = 0 ∧ n * A.1 1 0 < A.1 0 0 ∧
          n * A.1 1 1 ≤ A.1 0 1 ∧ |A.1 0 1 + -(n * A.1 1 1)| < |A.1 1 1| by
        simpa only [reps, Fin.isValue, cons_mul, Nat.succ_eq_add_one, Nat.reduceAdd, empty_mul,
          Equiv.symm_apply_apply, Set.mem_setOf_eq, of_apply, cons_val', vecMul, cons_dotProduct,
          vecHead, one_mul, vecTail, Function.comp_apply, Fin.succ_zero_eq_one, neg_mul,
          dotProduct_empty, add_zero, zero_mul, zero_add, empty_val', cons_val_fin_one,
          cons_val_one, cons_val_zero, lt_add_neg_iff_add_lt, le_add_neg_iff_add_le]
      simp_all only [h, mul_comm n, zero_mul, ← sub_eq_add_neg, ← h2,
        Fin.isValue, h1, h3, and_true, true_and]
    · simp only [reps, Fin.isValue, reduce_of_not_pos h h1, Int.ediv_neg, neg_neg, smul_def, ←
        mul_assoc, S_mul_S_eq, neg_mul, one_mul, coe_T_zpow, mul_neg, cons_mul, Nat.succ_eq_add_one,
        Nat.reduceAdd, empty_mul, Equiv.symm_apply_apply, neg_of, neg_cons, neg_empty,
        Set.mem_setOf_eq, of_apply, cons_val', Pi.neg_apply, vecMul, cons_dotProduct, vecHead,
        vecTail, Function.comp_apply, Fin.succ_zero_eq_one, h, mul_zero, dotProduct_empty, add_zero,
        zero_mul, neg_zero, empty_val', cons_val_fin_one, cons_val_one, cons_val_zero, lt_neg,
        neg_add_rev, zero_add, le_add_neg_iff_add_le, ← le_neg, abs_neg, true_and]
      refine ⟨?_, Int.ediv_mul_le _ hd, ?_⟩
      · simp only [Int.lt_iff_le_and_ne]
        exact ⟨not_lt.mp h1, A_a_ne_zero h hm⟩
      · rw [mul_comm, add_comm, ← Int.sub_eq_add_neg, ← Int.emod_def,
         abs_eq_self.mpr <| Int.emod_nonneg _ hd]
        exact Int.emod_lt _ hd


private lemma prop_red_S (hS : ∀ B, C B → C (S • B)) (B) : C (S • B) ↔ C B := by
  /-
    m : Int
    C : FixedDetMatrix (Fin 2) Int m → Prop
    hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
    B : FixedDetMatrix (Fin 2) Int m
    ⊢ Iff (C (HSMul.hSMul ModularGroup.S B)) (C B)
  -/
  refine ⟨?_, hS _⟩
  /-
    m : Int
    C : FixedDetMatrix (Fin 2) Int m → Prop
    hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
    B : FixedDetMatrix (Fin 2) Int m
    ⊢ C (HSMul.hSMul ModularGroup.S B) → C B
  -/
  intro ih
  /-
    m : Int
    C : FixedDetMatrix (Fin 2) Int m → Prop
    hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
    B : FixedDetMatrix (Fin 2) Int m
    ih : C (HSMul.hSMul ModularGroup.S B)
    ⊢ C B
  -/
  rw [← (S_smul_four B)]
  /-
    m : Int
    C : FixedDetMatrix (Fin 2) Int m → Prop
    hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
    B : FixedDetMatrix (Fin 2) Int m
    ih : C (HSMul.hSMul ModularGroup.S B)
    ⊢ C (HSMul.hSMul ModularGroup.S (HSMul.hSMul ModularGroup.S (HSMul.hSMul Modul …
  -/
  solve_by_elim
  /-
    🎉 no goals
  -/


private lemma prop_red_T (hS : ∀ B, C B → C (S • B)) (hT : ∀ B, C B → C (T • B)) (B) :
    C (T • B) ↔ C B := by
  /-
    m : Int
    C : FixedDetMatrix (Fin 2) Int m → Prop
    hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
    hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
    B : FixedDetMatrix (Fin 2) Int m
    ⊢ Iff (C (HSMul.hSMul ModularGroup.T B)) (C B)
  -/
  refine ⟨?_, hT _⟩
  /-
    m : Int
    C : FixedDetMatrix (Fin 2) Int m → Prop
    hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
    hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
    B : FixedDetMatrix (Fin 2) Int m
    ⊢ C (HSMul.hSMul ModularGroup.T B) → C B
  -/
  intro ih
  /-
    m : Int
    C : FixedDetMatrix (Fin 2) Int m → Prop
    hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
    hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
    B : FixedDetMatrix (Fin 2) Int m
    ih : C (HSMul.hSMul ModularGroup.T B)
    ⊢ C B
  -/
  rw [show B = T⁻¹ • T • B by simp, ← T_S_rel_smul]
  /-
    m : Int
    C : FixedDetMatrix (Fin 2) Int m → Prop
    hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
    hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
    B : FixedDetMatrix (Fin 2) Int m
    ih : C (HSMul.hSMul ModularGroup.T B)
    ⊢ C (HSMul.hSMul ModularGroup.S (HSMul.hSMul ModularGroup.S (HSMul.hSMul Modul …
  -/
  solve_by_elim (config := {maxDepth := 10})
  /-
    🎉 no goals
  -/


private lemma prop_red_T_pow (hS : ∀ B, C B → C (S • B)) (hT : ∀ B, C B → C (T • B)) :
     ∀ B (n : ℤ), C (T^n • B) ↔ C B := by
  /-
    m : Int
    C : FixedDetMatrix (Fin 2) Int m → Prop
    hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
    hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
    ⊢ ∀ (B : FixedDetMatrix (Fin 2) Int m) (n : Int), Iff (C (HSMul.hSMul (HPow.hP …
  -/
  intro B n
  /-
    m : Int
    C : FixedDetMatrix (Fin 2) Int m → Prop
    hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
    hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
    B : FixedDetMatrix (Fin 2) Int m
    n : Int
    ⊢ Iff (C (HSMul.hSMul (HPow.hPow ModularGroup.T n) B)) (C B)
  -/
  induction' n using Int.induction_on with n hn m hm
    /-
      case hz
      m : Int
      C : FixedDetMatrix (Fin 2) Int m → Prop
      hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
      hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
      B : FixedDetMatrix (Fin 2) Int m
      ⊢ Iff (C (HSMul.hSMul (HPow.hPow ModularGroup.T 0) B)) (C B)
    -/
  · simp only [zpow_zero, one_smul, imp_self]
    /-
      🎉 no goals
    -/
    /-
      case hp
      m : Int
      C : FixedDetMatrix (Fin 2) Int m → Prop
      hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
      hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
      B : FixedDetMatrix (Fin 2) Int m
      n : Nat
      hn : Iff (C (HSMul.hSMul (HPow.hPow ModularGroup.T ↑n) B)) (C B)
      ⊢ Iff (C (HSMul.hSMul (HPow.hPow ModularGroup.T (HAdd.hAdd (↑n) 1)) B)) (C B)
    -/
  · simpa only [add_comm (n:ℤ), zpow_add _ 1, ← smul_eq_mul, zpow_one, smul_assoc, prop_red_T hS hT]
    /-
      🎉 no goals
    -/
    /-
      case hn
      m✝ : Int
      C : FixedDetMatrix (Fin 2) Int m✝ → Prop
      hS : ∀ (B : FixedDetMatrix (Fin 2) Int m✝), C B → C (HSMul.hSMul ModularGroup. …
      hT : ∀ (B : FixedDetMatrix (Fin 2) Int m✝), C B → C (HSMul.hSMul ModularGroup. …
      B : FixedDetMatrix (Fin 2) Int m✝
      m : Nat
      hm : Iff (C (HSMul.hSMul (HPow.hPow ModularGroup.T (Neg.neg ↑m)) B)) (C B)
      ⊢ Iff (C (HSMul.hSMul (HPow.hPow ModularGroup.T (HSub.hSub (Neg.neg ↑m) 1)) B) …
    -/
  · rwa [sub_eq_neg_add, zpow_add, zpow_neg_one, ← prop_red_T hS hT, mul_smul, smul_inv_smul]
    /-
      🎉 no goals
    -/


@[elab_as_elim]
theorem induction_on {C : Δ m → Prop} {A : Δ m} (hm : m ≠ 0)
    (h0 : ∀ A : Δ m, A.1 1 0 = 0 → 0 < A.1 0 0 → 0 ≤ A.1 0 1 → |(A.1 0 1)| < |(A.1 1 1)| → C A)
    (hS : ∀ B, C B → C (S • B)) (hT : ∀ B, C B → C (T • B)) : C A := by
  have h_reduce : C (reduce A) := by
    rcases reduce_mem_reps hm A with ⟨H1, H2, H3, H4⟩
    exact h0 _ H1 H2 H3 H4
  /-
    m : Int
    C : FixedDetMatrix (Fin 2) Int m → Prop
    A : FixedDetMatrix (Fin 2) Int m
    hm : Ne m 0
    h0 : ∀ (A : FixedDetMatrix (Fin 2) Int m), Eq (↑A 1 0) 0 → LT.lt 0 (↑A 0 0) →  …
    hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
    hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
    h_reduce : C (FixedDetMatrices.reduce A)
    ⊢ C A
  -/
  suffices ∀ A : Δ m, C (reduce A) → C A from this _ h_reduce
  /-
    m : Int
    C : FixedDetMatrix (Fin 2) Int m → Prop
    A : FixedDetMatrix (Fin 2) Int m
    hm : Ne m 0
    h0 : ∀ (A : FixedDetMatrix (Fin 2) Int m), Eq (↑A 1 0) 0 → LT.lt 0 (↑A 0 0) →  …
    hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
    hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
    h_reduce : C (FixedDetMatrices.reduce A)
    ⊢ ∀ (A : FixedDetMatrix (Fin 2) Int m), C (FixedDetMatrices.reduce A) → C A
  -/
  apply reduce_rec
    /-
      case base
      m : Int
      C : FixedDetMatrix (Fin 2) Int m → Prop
      A : FixedDetMatrix (Fin 2) Int m
      hm : Ne m 0
      h0 : ∀ (A : FixedDetMatrix (Fin 2) Int m), Eq (↑A 1 0) 0 → LT.lt 0 (↑A 0 0) →  …
      hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
      hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
      h_reduce : C (FixedDetMatrices.reduce A)
      ⊢ ∀ (A : FixedDetMatrix (Fin 2) Int m), Eq (↑A 1 0) 0 → C (FixedDetMatrices.re …
    -/
  · intro A h
    /-
      case base
      m : Int
      C : FixedDetMatrix (Fin 2) Int m → Prop
      A✝ : FixedDetMatrix (Fin 2) Int m
      hm : Ne m 0
      h0 : ∀ (A : FixedDetMatrix (Fin 2) Int m), Eq (↑A 1 0) 0 → LT.lt 0 (↑A 0 0) →  …
      hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
      hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
      h_reduce : C (FixedDetMatrices.reduce A✝)
      A : FixedDetMatrix (Fin 2) Int m
      h : Eq (↑A 1 0) 0
      ⊢ C (FixedDetMatrices.reduce A) → C A
    -/
    by_cases h1 : 0 < A.1 0 0
      /-
        case pos
        m : Int
        C : FixedDetMatrix (Fin 2) Int m → Prop
        A✝ : FixedDetMatrix (Fin 2) Int m
        hm : Ne m 0
        h0 : ∀ (A : FixedDetMatrix (Fin 2) Int m), Eq (↑A 1 0) 0 → LT.lt 0 (↑A 0 0) →  …
        hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
        hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
        h_reduce : C (FixedDetMatrices.reduce A✝)
        A : FixedDetMatrix (Fin 2) Int m
        h : Eq (↑A 1 0) 0
        h1 : LT.lt 0 (↑A 0 0)
        ⊢ C (FixedDetMatrices.reduce A) → C A
      -/
    · simp only [reduce_of_pos h h1, prop_red_T_pow hS hT, imp_self]
      /-
        🎉 no goals
      -/
      /-
        case neg
        m : Int
        C : FixedDetMatrix (Fin 2) Int m → Prop
        A✝ : FixedDetMatrix (Fin 2) Int m
        hm : Ne m 0
        h0 : ∀ (A : FixedDetMatrix (Fin 2) Int m), Eq (↑A 1 0) 0 → LT.lt 0 (↑A 0 0) →  …
        hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
        hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
        h_reduce : C (FixedDetMatrices.reduce A✝)
        A : FixedDetMatrix (Fin 2) Int m
        h : Eq (↑A 1 0) 0
        h1 : Not (LT.lt 0 (↑A 0 0))
        ⊢ C (FixedDetMatrices.reduce A) → C A
      -/
    · simp only [reduce_of_not_pos h h1, prop_red_T_pow hS hT, prop_red_S hS, imp_self]
      /-
        🎉 no goals
      -/
  /-
    case step
    m : Int
    C : FixedDetMatrix (Fin 2) Int m → Prop
    A : FixedDetMatrix (Fin 2) Int m
    hm : Ne m 0
    h0 : ∀ (A : FixedDetMatrix (Fin 2) Int m), Eq (↑A 1 0) 0 → LT.lt 0 (↑A 0 0) →  …
    hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
    hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
    h_reduce : C (FixedDetMatrices.reduce A)
    ⊢ ∀ (A : FixedDetMatrix (Fin 2) Int m), Ne (↑A 1 0) 0 → (C (FixedDetMatrices.r …
  -/
  intro A hc ih hA
  /-
    case step
    m : Int
    C : FixedDetMatrix (Fin 2) Int m → Prop
    A✝ : FixedDetMatrix (Fin 2) Int m
    hm : Ne m 0
    h0 : ∀ (A : FixedDetMatrix (Fin 2) Int m), Eq (↑A 1 0) 0 → LT.lt 0 (↑A 0 0) →  …
    hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
    hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
    h_reduce : C (FixedDetMatrices.reduce A✝)
    A : FixedDetMatrix (Fin 2) Int m
    hc : Ne (↑A 1 0) 0
    ih : C (FixedDetMatrices.reduce (FixedDetMatrices.reduceStep A)) → C (FixedDet …
    hA : C (FixedDetMatrices.reduce A)
    ⊢ C A
  -/
  rw [← reduce_reduceStep hc] at hA
  /-
    case step
    m : Int
    C : FixedDetMatrix (Fin 2) Int m → Prop
    A✝ : FixedDetMatrix (Fin 2) Int m
    hm : Ne m 0
    h0 : ∀ (A : FixedDetMatrix (Fin 2) Int m), Eq (↑A 1 0) 0 → LT.lt 0 (↑A 0 0) →  …
    hS : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.S …
    hT : ∀ (B : FixedDetMatrix (Fin 2) Int m), C B → C (HSMul.hSMul ModularGroup.T …
    h_reduce : C (FixedDetMatrices.reduce A✝)
    A : FixedDetMatrix (Fin 2) Int m
    hc : Ne (↑A 1 0) 0
    ih : C (FixedDetMatrices.reduce (FixedDetMatrices.reduceStep A)) → C (FixedDet …
    hA : C (FixedDetMatrices.reduce (FixedDetMatrices.reduceStep A))
    ⊢ C A
  -/
  simpa only [reduceStep, prop_red_S hS, prop_red_T_pow hS hT] using ih hA
  /-
    🎉 no goals
  -/


lemma reps_one_id (A : FixedDetMatrix (Fin 2) ℤ 1) (a1 : A.1 1 0 = 0) (a4 : 0 < A.1 0 0)
    (a6 : |A.1 0 1| < |(A.1 1 1)|) : A = (1 : SL(2, ℤ)) := by
  /-
    A : FixedDetMatrix (Fin 2) Int 1
    a1 : Eq (↑A 1 0) 0
    a4 : LT.lt 0 (↑A 0 0)
    a6 : LT.lt (abs (↑A 0 1)) (abs (↑A 1 1))
    ⊢ Eq A 1
  -/
  have := Int.mul_eq_one_iff_eq_one_or_neg_one.mp (A_c_eq_zero a1)
  /-
    A : FixedDetMatrix (Fin 2) Int 1
    a1 : Eq (↑A 1 0) 0
    a4 : LT.lt 0 (↑A 0 0)
    a6 : LT.lt (abs (↑A 0 1)) (abs (↑A 1 1))
    this : Or (And (Eq (↑A 0 0) 1) (Eq (↑A 1 1) 1)) (And (Eq (↑A 0 0) (-1)) (Eq (↑ …
    ⊢ Eq A 1
  -/
  ext i j
  /-
    case h
    A : FixedDetMatrix (Fin 2) Int 1
    a1 : Eq (↑A 1 0) 0
    a4 : LT.lt 0 (↑A 0 0)
    a6 : LT.lt (abs (↑A 0 1)) (abs (↑A 1 1))
    this : Or (And (Eq (↑A 0 0) 1) (Eq (↑A 1 1) 1)) (And (Eq (↑A 0 0) (-1)) (Eq (↑ …
    i j : Fin 2
    ⊢ Eq (↑A i j) (↑1 i j)
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
  fin_cases i <;> fin_cases j <;> aesop
                                  /-
                                    🎉 no goals
                                  -/


/-- `SL(2, ℤ)` is generated by `S` and `T`. -/
lemma SpecialLinearGroup.SL2Z_generators : closure {S, T} = ⊤ := by
  /-
    ⊢ Eq (Subgroup.closure (Insert.insert ModularGroup.S (Singleton.singleton Modu …
  -/
  rw [eq_top_iff']
  /-
    ⊢ ∀ (x : Matrix.SpecialLinearGroup (Fin 2) Int), Membership.mem (Subgroup.clos …
  -/
  intro A
  induction A using (induction_on one_ne_zero) with
  | h0 A a1 a4 _ a6 =>
    rw [reps_one_id A a1 a4 a6]
    exact one_mem _
  | hS B hb =>
    exact mul_mem (subset_closure (Set.mem_insert S {T})) hb
  | hT B hb =>
    exact mul_mem (subset_closure (Set.mem_insert_of_mem S rfl)) hb


