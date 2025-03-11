/-- `Nat.superFactorial n` is the superfactorial of `n`. -/
def superFactorial : ℕ → ℕ
  | 0 => 1
  | succ n => factorial n.succ * superFactorial n


/-- `sf` notation for superfactorial -/
scoped notation "sf" n:60 => Nat.superFactorial n


@[simp]
theorem superFactorial_zero : sf 0 = 1 :=
  rfl


theorem superFactorial_succ (n : ℕ) : (sf n.succ) = (n + 1)! * sf n :=
  rfl


@[simp]
theorem superFactorial_one : sf 1 = 1 :=
  rfl


@[simp]
theorem superFactorial_two : sf 2 = 2 :=
  rfl


@[simp]
theorem prod_Icc_factorial : ∀ n : ℕ, ∏ x ∈ Icc 1 n, x ! = sf n
  | 0 => rfl
  | n + 1 => by
    rw [← Ico_succ_right 1 n.succ, prod_Ico_succ_top <| Nat.succ_le_succ <| Nat.zero_le n,
    Nat.factorial_succ, Ico_succ_right 1 n, prod_Icc_factorial n, superFactorial, factorial,
    Nat.succ_eq_add_one, mul_comm]


@[simp]
theorem prod_range_factorial_succ (n : ℕ) : ∏ x ∈ range n, (x + 1)! = sf n :=
  (prod_Icc_factorial n) ▸ range_eq_Ico ▸ Finset.prod_Ico_add' _ _ _ _


@[simp]
theorem prod_range_succ_factorial : ∀ n : ℕ, ∏ x ∈ range (n + 1), x ! = sf n
  | 0 => rfl
  | n + 1 => by
    /-
      n : Nat
      ⊢ Eq ((Finset.range (HAdd.hAdd (HAdd.hAdd n 1) 1)).prod fun x => x.factorial)  …
    -/
    rw [prod_range_succ, prod_range_succ_factorial n, mul_comm, superFactorial]
    /-
      🎉 no goals
    -/


theorem det_vandermonde_id_eq_superFactorial (n : ℕ) :
    (Matrix.vandermonde (fun (i : Fin (n + 1)) ↦ (i : R))).det = Nat.superFactorial n := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    ⊢ Eq (Matrix.vandermonde fun i => ↑↑i).det ↑n.superFactorial
  -/
  induction' n with n hn
    /-
      case zero
      R : Type u_1
      inst✝ : CommRing R
      ⊢ Eq (Matrix.vandermonde fun i => ↑↑i).det ↑(Nat.superFactorial 0)
    -/
  · simp [Matrix.det_vandermonde]
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      hn : Eq (Matrix.vandermonde fun i => ↑↑i).det ↑n.superFactorial
      ⊢ Eq (Matrix.vandermonde fun i => ↑↑i).det ↑(HAdd.hAdd n 1).superFactorial
    -/
  · rw [Nat.superFactorial, Matrix.det_vandermonde, Fin.prod_univ_succAbove _ 0]
    /-
      case succ
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      hn : Eq (Matrix.vandermonde fun i => ↑↑i).det ↑n.superFactorial
      ⊢ Eq (HMul.hMul ((Finset.Ioi 0).prod fun j => HSub.hSub ↑↑j ↑↑0) (Finset.univ. …
    -/
    push_cast
    /-
      case succ
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      hn : Eq (Matrix.vandermonde fun i => ↑↑i).det ↑n.superFactorial
      ⊢ Eq (HMul.hMul ((Finset.Ioi 0).prod fun j => HSub.hSub ↑↑j ↑↑0) (Finset.univ. …
    -/
    congr
      /-
        case succ.e_a
        R : Type u_1
        inst✝ : CommRing R
        n : Nat
        hn : Eq (Matrix.vandermonde fun i => ↑↑i).det ↑n.superFactorial
        ⊢ Eq ((Finset.Ioi 0).prod fun j => HSub.hSub ↑↑j ↑↑0) ↑n.succ.factorial
      -/
    · simp only [Fin.val_zero, Nat.cast_zero, sub_zero]
      /-
        case succ.e_a
        R : Type u_1
        inst✝ : CommRing R
        n : Nat
        hn : Eq (Matrix.vandermonde fun i => ↑↑i).det ↑n.superFactorial
        ⊢ Eq ((Finset.Ioi 0).prod fun x => ↑↑x) ↑n.succ.factorial
      -/
      norm_cast
      /-
        case succ.e_a
        R : Type u_1
        inst✝ : CommRing R
        n : Nat
        hn : Eq (Matrix.vandermonde fun i => ↑↑i).det ↑n.superFactorial
        ⊢ Eq ↑((Finset.Ioi 0).prod fun x => ↑x) ↑n.succ.factorial
      -/
      simp [Fin.prod_univ_eq_prod_range (fun i ↦ (↑i + 1)) (n + 1)]
      /-
        🎉 no goals
      -/
      /-
        case succ.e_a
        R : Type u_1
        inst✝ : CommRing R
        n : Nat
        hn : Eq (Matrix.vandermonde fun i => ↑↑i).det ↑n.superFactorial
        ⊢ Eq (Finset.univ.prod fun i => (Finset.Ioi (Fin.succAbove 0 i)).prod fun j => …
      -/
    · rw [Matrix.det_vandermonde] at hn
      /-
        case succ.e_a
        R : Type u_1
        inst✝ : CommRing R
        n : Nat
        hn : Eq (Finset.univ.prod fun i => (Finset.Ioi i).prod fun j => HSub.hSub ↑↑j  …
        ⊢ Eq (Finset.univ.prod fun i => (Finset.Ioi (Fin.succAbove 0 i)).prod fun j => …
      -/
      simp [hn]
      /-
        🎉 no goals
      -/


theorem superFactorial_two_mul : ∀ n : ℕ,
    sf (2 * n) = (∏ i ∈ range n, (2 * i + 1) !) ^ 2 * 2 ^ n * n !
  | 0 => rfl
  | (n + 1) => by
    simp only [prod_range_succ, mul_pow, mul_add, mul_one, superFactorial_succ,
      superFactorial_two_mul n, factorial_succ]
    /-
      n : Nat
      ⊢ Eq (HMul.hMul (HMul.hMul (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 n) 1) 1) (HMul.h …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem superFactorial_four_mul (n : ℕ) :
    sf (4 * n) = ((∏ i ∈ range (2 * n), (2 * i + 1) !) * 2 ^ n) ^ 2 * (2 * n) ! :=
  calc
    sf (4 * n) = (∏ i ∈ range (2 * n), (2 * i + 1) !) ^ 2 * 2 ^ (2 * n) * (2 * n) ! := by
      /-
        n : Nat
        ⊢ Eq (HMul.hMul 4 n).superFactorial (HMul.hMul (HMul.hMul (HPow.hPow ((Finset. …
      -/
      rw [← superFactorial_two_mul, ← mul_assoc, Nat.mul_two]
      /-
        🎉 no goals
      -/
    _ = ((∏ i ∈ range (2 * n), (2 * i + 1) !) * 2 ^ n) ^ 2 * (2 * n) ! := by
      /-
        n : Nat
        ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow ((Finset.range (HMul.hMul 2 n)).prod fun …
      -/
      rw [pow_mul', mul_pow]
      /-
        🎉 no goals
      -/


private theorem matrixOf_eval_descPochhammer_eq_mul_matrixOf_choose {n : ℕ} (v : Fin n → ℕ) :
    (Matrix.of (fun (i j : Fin n) => (descPochhammer ℤ j).eval (v i : ℤ))).det =
    (∏ i : Fin n, Nat.factorial i) *
      (Matrix.of (fun (i j : Fin n) => (Nat.choose (v i) (j : ℕ) : ℤ))).det := by
  /-
    n : Nat
    v : Fin n → Nat
    ⊢ Eq (Matrix.of fun i j => Polynomial.eval (↑(v i)) (descPochhammer Int ↑j)).d …
  -/
  convert Matrix.det_mul_row (fun (i : Fin n) => ((Nat.factorial (i : ℕ)) : ℤ)) _
    /-
      case h.e'_2.h.e'_6.h.e'_6.h.h
      n : Nat
      v : Fin n → Nat
      x✝¹ x✝ : Fin n
      ⊢ Eq (Polynomial.eval (↑(v x✝¹)) (descPochhammer Int ↑x✝)) (HMul.hMul (↑(↑x✝). …
    -/
  · rw [Matrix.of_apply, descPochhammer_eval_eq_descFactorial ℤ _ _]
    /-
      case h.e'_2.h.e'_6.h.e'_6.h.h
      n : Nat
      v : Fin n → Nat
      x✝¹ x✝ : Fin n
      ⊢ Eq (↑((v x✝¹).descFactorial ↑x✝)) (HMul.hMul ↑(↑x✝).factorial ↑((v x✝¹).choo …
    -/
    congr
    /-
      case h.e'_2.h.e'_6.h.e'_6.h.h.e_a
      n : Nat
      v : Fin n → Nat
      x✝¹ x✝ : Fin n
      ⊢ Eq ((v x✝¹).descFactorial ↑x✝) (HMul.hMul (↑x✝).factorial ((v x✝¹).choose ↑x …
    -/
    exact Nat.descFactorial_eq_factorial_mul_choose _ _
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_5
      n : Nat
      v : Fin n → Nat
      ⊢ Eq (↑(Finset.univ.prod fun i => (↑i).factorial)) (Finset.univ.prod fun i =>  …
    -/
  · rw [Nat.cast_prod]
    /-
      🎉 no goals
    -/


theorem superFactorial_dvd_vandermonde_det {n : ℕ} (v : Fin (n + 1) → ℤ) :
    ↑(Nat.superFactorial n) ∣ (Matrix.vandermonde v).det := by
  /-
    n : Nat
    v : Fin (HAdd.hAdd n 1) → Int
    ⊢ Dvd.dvd (↑n.superFactorial) (Matrix.vandermonde v).det
  -/
  let m := inf' univ ⟨0, mem_univ _⟩ v
  /-
    n : Nat
    v : Fin (HAdd.hAdd n 1) → Int
    m : Int := Finset.univ.inf' ⋯ v
    ⊢ Dvd.dvd (↑n.superFactorial) (Matrix.vandermonde v).det
  -/
  let w' := fun i ↦ (v i - m).toNat
  /-
    n : Nat
    v : Fin (HAdd.hAdd n 1) → Int
    m : Int := Finset.univ.inf' ⋯ v
    w' : Fin (HAdd.hAdd n 1) → Nat := fun i => (HSub.hSub (v i) m).toNat
    ⊢ Dvd.dvd (↑n.superFactorial) (Matrix.vandermonde v).det
  -/
  have hw' : ∀ i, (w' i : ℤ) = v i - m := fun i ↦ Int.toNat_sub_of_le (inf'_le _ (mem_univ _))
  have h := Matrix.det_eval_matrixOfPolynomials_eq_det_vandermonde (fun i ↦ ↑(w' i))
      (fun i => descPochhammer ℤ i)
      (fun i => descPochhammer_natDegree ℤ i)
      (fun i => monic_descPochhammer ℤ i)
  /-
    n : Nat
    v : Fin (HAdd.hAdd n 1) → Int
    m : Int := Finset.univ.inf' ⋯ v
    w' : Fin (HAdd.hAdd n 1) → Nat := fun i => (HSub.hSub (v i) m).toNat
    hw' : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (↑(w' i)) (HSub.hSub (v i) m)
    h : Eq (Matrix.vandermonde fun i => ↑(w' i)).det (Matrix.of fun i j => Polynom …
    ⊢ Dvd.dvd (↑n.superFactorial) (Matrix.vandermonde v).det
  -/
  conv_lhs at h => simp only [hw', Matrix.det_vandermonde_sub]
  /-
    n : Nat
    v : Fin (HAdd.hAdd n 1) → Int
    m : Int := Finset.univ.inf' ⋯ v
    w' : Fin (HAdd.hAdd n 1) → Nat := fun i => (HSub.hSub (v i) m).toNat
    hw' : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (↑(w' i)) (HSub.hSub (v i) m)
    h : Eq (Matrix.vandermonde v).det (Matrix.of fun i j => Polynomial.eval ((fun  …
    ⊢ Dvd.dvd (↑n.superFactorial) (Matrix.vandermonde v).det
  -/
  use (Matrix.of (fun (i j : Fin (n + 1)) => (Nat.choose (w' i) (j : ℕ) : ℤ))).det
  /-
    case h
    n : Nat
    v : Fin (HAdd.hAdd n 1) → Int
    m : Int := Finset.univ.inf' ⋯ v
    w' : Fin (HAdd.hAdd n 1) → Nat := fun i => (HSub.hSub (v i) m).toNat
    hw' : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (↑(w' i)) (HSub.hSub (v i) m)
    h : Eq (Matrix.vandermonde v).det (Matrix.of fun i j => Polynomial.eval ((fun  …
    ⊢ Eq (Matrix.vandermonde v).det (HMul.hMul (↑n.superFactorial) (Matrix.of fun  …
  -/
  simp [h, matrixOf_eval_descPochhammer_eq_mul_matrixOf_choose w', Fin.prod_univ_eq_prod_range]
  /-
    🎉 no goals
  -/


