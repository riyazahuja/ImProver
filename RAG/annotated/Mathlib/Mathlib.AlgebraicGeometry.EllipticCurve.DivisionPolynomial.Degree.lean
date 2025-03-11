lemma natDegree_Ψ₂Sq_le : W.Ψ₂Sq.natDegree ≤ 3 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ LE.le W.Ψ₂Sq.natDegree 3
  -/
  rw [Ψ₂Sq]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C 4) (HPow.hPo …
  -/
  compute_degree
  /-
    🎉 no goals
  -/


@[simp]
lemma coeff_Ψ₂Sq : W.Ψ₂Sq.coeff 3 = 4 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ Eq (W.Ψ₂Sq.coeff 3) 4
  -/
  rw [Ψ₂Sq]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ Eq ((HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C 4) (HPow.hPow  …
  -/
  compute_degree!
  /-
    🎉 no goals
  -/


lemma coeff_Ψ₂Sq_ne_zero (h : (4 : R) ≠ 0) : W.Ψ₂Sq.coeff 3 ≠ 0 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    h : Ne 4 0
    ⊢ Ne (W.Ψ₂Sq.coeff 3) 0
  -/
  rwa [coeff_Ψ₂Sq]
  /-
    🎉 no goals
  -/


@[simp]
lemma natDegree_Ψ₂Sq (h : (4 : R) ≠ 0) : W.Ψ₂Sq.natDegree = 3 :=
  natDegree_eq_of_le_of_coeff_ne_zero W.natDegree_Ψ₂Sq_le <| W.coeff_Ψ₂Sq_ne_zero h


lemma natDegree_Ψ₂Sq_pos (h : (4 : R) ≠ 0) : 0 < W.Ψ₂Sq.natDegree :=
  W.natDegree_Ψ₂Sq h ▸ three_pos


@[simp]
lemma leadingCoeff_Ψ₂Sq (h : (4 : R) ≠ 0) : W.Ψ₂Sq.leadingCoeff = 4 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    h : Ne 4 0
    ⊢ Eq W.Ψ₂Sq.leadingCoeff 4
  -/
  rw [leadingCoeff, W.natDegree_Ψ₂Sq h, coeff_Ψ₂Sq]
  /-
    🎉 no goals
  -/


lemma Ψ₂Sq_ne_zero (h : (4 : R) ≠ 0) : W.Ψ₂Sq ≠ 0 :=
  ne_zero_of_natDegree_gt <| W.natDegree_Ψ₂Sq_pos h


lemma natDegree_Ψ₃_le : W.Ψ₃.natDegree ≤ 4 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ LE.le W.Ψ₃.natDegree 4
  -/
  rw [Ψ₃]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HPow.hPow Po …
  -/
  compute_degree
  /-
    🎉 no goals
  -/


@[simp]
lemma coeff_Ψ₃ : W.Ψ₃.coeff 4 = 3 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ Eq (W.Ψ₃.coeff 4) 3
  -/
  rw [Ψ₃]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ Eq ((HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul 3 (HPow.hPow Poly …
  -/
  compute_degree!
  /-
    🎉 no goals
  -/


lemma coeff_Ψ₃_ne_zero (h : (3 : R) ≠ 0) : W.Ψ₃.coeff 4 ≠ 0 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    h : Ne 3 0
    ⊢ Ne (W.Ψ₃.coeff 4) 0
  -/
  rwa [coeff_Ψ₃]
  /-
    🎉 no goals
  -/


@[simp]
lemma natDegree_Ψ₃ (h : (3 : R) ≠ 0) : W.Ψ₃.natDegree = 4 :=
  natDegree_eq_of_le_of_coeff_ne_zero W.natDegree_Ψ₃_le <| W.coeff_Ψ₃_ne_zero h


lemma natDegree_Ψ₃_pos (h : (3 : R) ≠ 0) : 0 < W.Ψ₃.natDegree :=
  W.natDegree_Ψ₃ h ▸ four_pos


@[simp]
lemma leadingCoeff_Ψ₃ (h : (3 : R) ≠ 0) : W.Ψ₃.leadingCoeff = 3 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    h : Ne 3 0
    ⊢ Eq W.Ψ₃.leadingCoeff 3
  -/
  rw [leadingCoeff, W.natDegree_Ψ₃ h, coeff_Ψ₃]
  /-
    🎉 no goals
  -/


lemma Ψ₃_ne_zero (h : (3 : R) ≠ 0) : W.Ψ₃ ≠ 0 :=
  ne_zero_of_natDegree_gt <| W.natDegree_Ψ₃_pos h


lemma natDegree_preΨ₄_le : W.preΨ₄.natDegree ≤ 6 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ LE.le W.preΨ₄.natDegree 6
  -/
  rw [preΨ₄]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMu …
  -/
  compute_degree
  /-
    🎉 no goals
  -/


@[simp]
lemma coeff_preΨ₄ : W.preΨ₄.coeff 6 = 2 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ Eq (W.preΨ₄.coeff 6) 2
  -/
  rw [preΨ₄]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ Eq ((HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul. …
  -/
  compute_degree!
  /-
    🎉 no goals
  -/


lemma coeff_preΨ₄_ne_zero (h : (2 : R) ≠ 0) : W.preΨ₄.coeff 6 ≠ 0 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    h : Ne 2 0
    ⊢ Ne (W.preΨ₄.coeff 6) 0
  -/
  rwa [coeff_preΨ₄]
  /-
    🎉 no goals
  -/


@[simp]
lemma natDegree_preΨ₄ (h : (2 : R) ≠ 0) : W.preΨ₄.natDegree = 6 :=
  natDegree_eq_of_le_of_coeff_ne_zero W.natDegree_preΨ₄_le <| W.coeff_preΨ₄_ne_zero h


lemma natDegree_preΨ₄_pos (h : (2 : R) ≠ 0) : 0 < W.preΨ₄.natDegree := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    h : Ne 2 0
    ⊢ LT.lt 0 W.preΨ₄.natDegree
  -/
  linarith only [W.natDegree_preΨ₄ h]
  /-
    🎉 no goals
  -/


@[simp]
lemma leadingCoeff_preΨ₄ (h : (2 : R) ≠ 0) : W.preΨ₄.leadingCoeff = 2 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    h : Ne 2 0
    ⊢ Eq W.preΨ₄.leadingCoeff 2
  -/
  rw [leadingCoeff, W.natDegree_preΨ₄ h, coeff_preΨ₄]
  /-
    🎉 no goals
  -/


lemma preΨ₄_ne_zero (h : (2 : R) ≠ 0) : W.preΨ₄ ≠ 0 :=
  ne_zero_of_natDegree_gt <| W.natDegree_preΨ₄_pos h


private def expDegree (n : ℕ) : ℕ :=
  (n ^ 2 - if Even n then 4 else 1) / 2


private lemma expDegree_cast {n : ℕ} (hn : n ≠ 0) :
    2 * (expDegree n : ℤ) = n ^ 2 - if Even n then 4 else 1 := by
  /-
    n : Nat
    hn : Ne n 0
    ⊢ Eq (HMul.hMul 2 ↑(WeierstrassCurve.expDegree n)) (HSub.hSub (HPow.hPow (↑n)  …
  -/
  rcases n.even_or_odd' with ⟨n, rfl | rfl⟩
    /-
      case intro.inl
      n : Nat
      hn : Ne (HMul.hMul 2 n) 0
      ⊢ Eq (HMul.hMul 2 ↑(WeierstrassCurve.expDegree (HMul.hMul 2 n))) (HSub.hSub (H …
    -/
  · rcases n with _ | n
      /-
        case intro.inl.zero
        hn : Ne (HMul.hMul 2 0) 0
        ⊢ Eq (HMul.hMul 2 ↑(WeierstrassCurve.expDegree (HMul.hMul 2 0))) (HSub.hSub (H …
      -/
    · contradiction
      /-
        🎉 no goals
      -/
    push_cast [expDegree, show (2 * (n + 1)) ^ 2 = 2 * (2 * n * (n + 2)) + 4 by ring1, even_two_mul,
      Nat.add_sub_cancel, Nat.mul_div_cancel_left _ two_pos]
    /-
      case intro.inl.succ
      n : Nat
      hn : Ne (HMul.hMul 2 (HAdd.hAdd n 1)) 0
      ⊢ Eq (HMul.hMul 2 (HMul.hMul (HMul.hMul 2 ↑n) (HAdd.hAdd (↑n) 2))) (HSub.hSub  …
    -/
    ring1
    /-
      🎉 no goals
    -/
  · push_cast [expDegree, show (2 * n + 1) ^ 2 = 2 * (2 * n * (n + 1)) + 1 by ring1,
      n.not_even_two_mul_add_one, Nat.add_sub_cancel, Nat.mul_div_cancel_left _ two_pos]
    /-
      case intro.inr
      n : Nat
      hn : Ne (HAdd.hAdd (HMul.hMul 2 n) 1) 0
      ⊢ Eq (HMul.hMul 2 (HMul.hMul (HMul.hMul 2 ↑n) (HAdd.hAdd (↑n) 1))) (HSub.hSub  …
    -/
    ring1
    /-
      🎉 no goals
    -/


private lemma expDegree_rec (m : ℕ) :
    (expDegree (2 * (m + 3)) = 2 * expDegree (m + 2) + expDegree (m + 3) + expDegree (m + 5) ∧
    expDegree (2 * (m + 3)) = expDegree (m + 1) + expDegree (m + 3) + 2 * expDegree (m + 4)) ∧
    (expDegree (2 * (m + 2) + 1) =
      expDegree (m + 4) + 3 * expDegree (m + 2) + (if Even m then 2 * 3 else 0) ∧
    expDegree (2 * (m + 2) + 1) =
      expDegree (m + 1) + 3 * expDegree (m + 3) + (if Even m then 0 else 2 * 3)) := by
  push_cast [← @Nat.cast_inj ℤ, ← mul_left_cancel_iff_of_pos (b := (expDegree _ : ℤ)) two_pos,
    mul_add, mul_left_comm (2 : ℤ)]
  /-
    m : Nat
    ⊢ And (And (Eq (HMul.hMul 2 ↑(WeierstrassCurve.expDegree (HAdd.hAdd (HMul.hMul …
  -/
  repeat rw [expDegree_cast <| by omega]
  /-
    m : Nat
    ⊢ And (And (Eq (HSub.hSub (HPow.hPow (↑(HAdd.hAdd (HMul.hMul 2 m) 6)) 2) (ite  …
  -/
  push_cast [Nat.even_add_one, ite_not, even_two_mul]
  /-
    m : Nat
    ⊢ And (And (Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul 2 ↑m) 6) 2) 4) (HAd …
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
                                                /-
                                                  🎉 no goals
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
  constructor <;> constructor <;> split_ifs <;> ring1
                                                /-
                                                  🎉 no goals
                                                -/


private def expCoeff (n : ℕ) : ℤ :=
  if Even n then n / 2 else n


private lemma expCoeff_cast (n : ℕ) : (expCoeff n : ℚ) = if Even n then (n / 2 : ℚ) else n := by
  /-
    n : Nat
    ⊢ Eq (↑(WeierstrassCurve.expCoeff n)) (ite (Even n) (HDiv.hDiv (↑n) 2) ↑n)
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  rcases n.even_or_odd' with ⟨n, rfl | rfl⟩ <;> simp [expCoeff, n.not_even_two_mul_add_one]
                                                /-
                                                  🎉 no goals
                                                -/


private lemma expCoeff_rec (m : ℕ) :
    (expCoeff (2 * (m + 3)) =
      expCoeff (m + 2) ^ 2 * expCoeff (m + 3) * expCoeff (m + 5) -
        expCoeff (m + 1) * expCoeff (m + 3) * expCoeff (m + 4) ^ 2) ∧
    (expCoeff (2 * (m + 2) + 1) =
      expCoeff (m + 4) * expCoeff (m + 2) ^ 3 * (if Even m then 4 ^ 2 else 1) -
        expCoeff (m + 1) * expCoeff (m + 3) ^ 3 * (if Even m then 1 else 4 ^ 2)) := by
  push_cast [← @Int.cast_inj ℚ, expCoeff_cast, even_two_mul, m.not_even_two_mul_add_one,
    Nat.even_add_one, ite_not]
  /-
    m : Nat
    ⊢ And (Eq (HDiv.hDiv (HMul.hMul 2 (HAdd.hAdd (↑m) 3)) 2) (HSub.hSub (HMul.hMul …
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
  constructor <;> split_ifs <;> ring1
                                /-
                                  🎉 no goals
                                -/


private lemma natDegree_coeff_preΨ' (n : ℕ) :
    (W.preΨ' n).natDegree ≤ expDegree n ∧ (W.preΨ' n).coeff (expDegree n) = expCoeff n := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    ⊢ And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree n)) (Eq ((W.pre …
  -/
  let dm {m n p q} : _ → _ → (p * q : R[X]).natDegree ≤ m + n := natDegree_mul_le_of_le
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
    ⊢ And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree n)) (Eq ((W.pre …
  -/
  let dp {m n p} : _ → (p ^ n : R[X]).natDegree ≤ n * m := natDegree_pow_le_of_le n
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
    dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
    ⊢ And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree n)) (Eq ((W.pre …
  -/
  let cm {m n p q} : _ → _ → (p * q : R[X]).coeff (m + n) = _ := coeff_mul_of_natDegree_le
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
    dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
    cm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
    ⊢ And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree n)) (Eq ((W.pre …
  -/
  let cp {m n p} : _ → (p ^ m : R[X]).coeff (m * n) = _ := coeff_pow_of_natDegree_le
  induction n using normEDSRec with
  | zero => simpa only [preΨ'_zero] using ⟨by rfl, Int.cast_zero.symm⟩
  | one => simpa only [preΨ'_one] using ⟨natDegree_one.le, coeff_one_zero.trans Int.cast_one.symm⟩
  | two => simpa only [preΨ'_two] using ⟨natDegree_one.le, coeff_one_zero.trans Int.cast_one.symm⟩
  | three => simpa only [preΨ'_three] using ⟨W.natDegree_Ψ₃_le, W.coeff_Ψ₃ ▸ Int.cast_three.symm⟩
  | four => simpa only [preΨ'_four] using ⟨W.natDegree_preΨ₄_le, W.coeff_preΨ₄ ▸ Int.cast_two.symm⟩
  | even m h₁ h₂ h₃ h₄ h₅ =>
    constructor
    · nth_rw 1 [preΨ'_even, ← max_self <| expDegree _, (expDegree_rec m).1.1, (expDegree_rec m).1.2]
      exact natDegree_sub_le_of_le (dm (dm (dp h₂.1) h₃.1) h₅.1) (dm (dm h₁.1 h₃.1) (dp h₄.1))
    · nth_rw 1 [preΨ'_even, coeff_sub, (expDegree_rec m).1.1, cm (dm (dp h₂.1) h₃.1) h₅.1,
        cm (dp h₂.1) h₃.1, cp h₂.1, h₂.2, h₃.2, h₅.2, (expDegree_rec m).1.2,
        cm (dm h₁.1 h₃.1) (dp h₄.1), cm h₁.1 h₃.1, h₁.2, cp h₄.1, h₃.2, h₄.2, (expCoeff_rec m).1]
      norm_cast
  | odd m h₁ h₂ h₃ h₄ =>
    rw [preΨ'_odd]
    constructor
    · nth_rw 1 [← max_self <| expDegree _, (expDegree_rec m).2.1, (expDegree_rec m).2.2]
      refine natDegree_sub_le_of_le (dm (dm h₄.1 (dp h₂.1)) ?_) (dm (dm h₁.1 (dp h₃.1)) ?_)
      all_goals split_ifs <;>
        simp only [apply_ite natDegree, natDegree_one.le, dp W.natDegree_Ψ₂Sq_le]
    · nth_rw 1 [coeff_sub, (expDegree_rec m).2.1, cm (dm h₄.1 (dp h₂.1)), cm h₄.1 (dp h₂.1),
        h₄.2, cp h₂.1, h₂.2, apply_ite₂ coeff, cp W.natDegree_Ψ₂Sq_le, coeff_Ψ₂Sq, coeff_one_zero,
        (expDegree_rec m).2.2, cm (dm h₁.1 (dp h₃.1)), cm h₁.1 (dp h₃.1), h₁.2, cp h₃.1, h₃.2,
        apply_ite₂ coeff, cp W.natDegree_Ψ₂Sq_le, coeff_one_zero, coeff_Ψ₂Sq, (expCoeff_rec m).2]
      · norm_cast
      all_goals split_ifs <;>
        simp only [apply_ite natDegree, natDegree_one.le, dp W.natDegree_Ψ₂Sq_le]


lemma natDegree_preΨ'_le (n : ℕ) : (W.preΨ' n).natDegree ≤ (n ^ 2 - if Even n then 4 else 1) / 2 :=
  (W.natDegree_coeff_preΨ' n).left


@[simp]
lemma coeff_preΨ' (n : ℕ) : (W.preΨ' n).coeff ((n ^ 2 - if Even n then 4 else 1) / 2) =
    if Even n then n / 2 else n := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    ⊢ Eq ((W.preΨ' n).coeff (HDiv.hDiv (HSub.hSub (HPow.hPow n 2) (ite (Even n) 4  …
  -/
  convert (W.natDegree_coeff_preΨ' n).right using 1
  /-
    case h.e'_3
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    ⊢ Eq ↑(ite (Even n) (HDiv.hDiv n 2) n) ↑(WeierstrassCurve.expCoeff n)
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  rcases n.even_or_odd' with ⟨n, rfl | rfl⟩ <;> simp [expCoeff, n.not_even_two_mul_add_one]
                                                /-
                                                  🎉 no goals
                                                -/


lemma coeff_preΨ'_ne_zero {n : ℕ} (h : (n : R) ≠ 0) :
    (W.preΨ' n).coeff ((n ^ 2 - if Even n then 4 else 1) / 2) ≠ 0 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    h : Ne (↑n) 0
    ⊢ Ne ((W.preΨ' n).coeff (HDiv.hDiv (HSub.hSub (HPow.hPow n 2) (ite (Even n) 4  …
  -/
  rcases n.even_or_odd' with ⟨n, rfl | rfl⟩
    /-
      case intro.inl
      R : Type u
      inst✝ : CommRing R
      W : WeierstrassCurve R
      n : Nat
      h : Ne (↑(HMul.hMul 2 n)) 0
      ⊢ Ne ((W.preΨ' (HMul.hMul 2 n)).coeff (HDiv.hDiv (HSub.hSub (HPow.hPow (HMul.h …
    -/
  · rw [coeff_preΨ', if_pos <| even_two_mul n, n.mul_div_cancel_left two_pos]
    /-
      case intro.inl
      R : Type u
      inst✝ : CommRing R
      W : WeierstrassCurve R
      n : Nat
      h : Ne (↑(HMul.hMul 2 n)) 0
      ⊢ Ne (↑n) 0
    -/
    exact right_ne_zero_of_mul <| by rwa [← Nat.cast_mul]
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      R : Type u
      inst✝ : CommRing R
      W : WeierstrassCurve R
      n : Nat
      h : Ne (↑(HAdd.hAdd (HMul.hMul 2 n) 1)) 0
      ⊢ Ne ((W.preΨ' (HAdd.hAdd (HMul.hMul 2 n) 1)).coeff (HDiv.hDiv (HSub.hSub (HPo …
    -/
  · rwa [coeff_preΨ', if_neg n.not_even_two_mul_add_one]
    /-
      🎉 no goals
    -/


@[simp]
lemma natDegree_preΨ' {n : ℕ} (h : (n : R) ≠ 0) :
    (W.preΨ' n).natDegree = (n ^ 2 - if Even n then 4 else 1) / 2 :=
  natDegree_eq_of_le_of_coeff_ne_zero (W.natDegree_preΨ'_le n) <| W.coeff_preΨ'_ne_zero h


lemma natDegree_preΨ'_pos {n : ℕ} (hn : 2 < n) (h : (n : R) ≠ 0) : 0 < (W.preΨ' n).natDegree := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    hn : LT.lt 2 n
    h : Ne (↑n) 0
    ⊢ LT.lt 0 (W.preΨ' n).natDegree
  -/
  simp only [W.natDegree_preΨ' h, Nat.div_pos_iff, zero_lt_two, true_and]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    hn : LT.lt 2 n
    h : Ne (↑n) 0
    ⊢ LE.le 2 (HSub.hSub (HPow.hPow n 2) (ite (Even n) 4 1))
  -/
  split_ifs <;>
    /-
      case pos
      R : Type u
      inst✝ : CommRing R
      W : WeierstrassCurve R
      n : Nat
      hn : LT.lt 2 n
      h : Ne (↑n) 0
      h✝ : Even n
      ⊢ LE.le 2 (HSub.hSub (HPow.hPow n 2) 4)
    -/
    /-
      🎉 no goals
    -/
    exact Nat.AtLeastTwo.prop.trans <| Nat.sub_le_sub_right (Nat.pow_le_pow_of_le_left hn 2) _
    /-
      🎉 no goals
    -/


@[simp]
lemma leadingCoeff_preΨ' {n : ℕ} (h : (n : R) ≠ 0) :
    (W.preΨ' n).leadingCoeff = if Even n then n / 2 else n := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    h : Ne (↑n) 0
    ⊢ Eq (W.preΨ' n).leadingCoeff ↑(ite (Even n) (HDiv.hDiv n 2) n)
  -/
  rw [leadingCoeff, W.natDegree_preΨ' h, coeff_preΨ']
  /-
    🎉 no goals
  -/


lemma preΨ'_ne_zero [Nontrivial R] {n : ℕ} (h : (n : R) ≠ 0) : W.preΨ' n ≠ 0 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : Nontrivial R
    n : Nat
    h : Ne (↑n) 0
    ⊢ Ne (W.preΨ' n) 0
  -/
  by_cases hn : 2 < n
    /-
      case pos
      R : Type u
      inst✝¹ : CommRing R
      W : WeierstrassCurve R
      inst✝ : Nontrivial R
      n : Nat
      h : Ne (↑n) 0
      hn : LT.lt 2 n
      ⊢ Ne (W.preΨ' n) 0
    -/
  · exact ne_zero_of_natDegree_gt <| W.natDegree_preΨ'_pos hn h
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝¹ : CommRing R
      W : WeierstrassCurve R
      inst✝ : Nontrivial R
      n : Nat
      h : Ne (↑n) 0
      hn : Not (LT.lt 2 n)
      ⊢ Ne (W.preΨ' n) 0
    -/
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
  · rcases n with _ | _ | _ <;> aesop
                                /-
                                  🎉 no goals
                                -/


lemma natDegree_preΨ_le (n : ℤ) : (W.preΨ n).natDegree ≤
    (n.natAbs ^ 2 - if Even n then 4 else 1) / 2 := by
  induction n using Int.negInduction with
  | nat n => exact_mod_cast W.preΨ_ofNat n ▸ W.natDegree_preΨ'_le n
  | neg ih => simp only [preΨ_neg, natDegree_neg, Int.natAbs_neg, even_neg, ih]


@[simp]
lemma coeff_preΨ (n : ℤ) : (W.preΨ n).coeff ((n.natAbs ^ 2 - if Even n then 4 else 1) / 2) =
    if Even n then n / 2 else n := by
  induction n using Int.negInduction with
  | nat n => exact_mod_cast W.preΨ_ofNat n ▸ W.coeff_preΨ' n
  | neg ih n =>
    simp only [preΨ_neg, coeff_neg, Int.natAbs_neg, even_neg]
    rcases ih n, n.even_or_odd' with ⟨ih, ⟨n, rfl | rfl⟩⟩ <;>
      push_cast [even_two_mul, Int.not_even_two_mul_add_one, Int.neg_ediv_of_dvd ⟨n, rfl⟩] at * <;>
      rw [ih]


lemma coeff_preΨ_ne_zero {n : ℤ} (h : (n : R) ≠ 0) :
    (W.preΨ n).coeff ((n.natAbs ^ 2 - if Even n then 4 else 1) / 2) ≠ 0 := by
  induction n using Int.negInduction with
  | nat n => simpa only [preΨ_ofNat, Int.even_coe_nat]
      using W.coeff_preΨ'_ne_zero <| by exact_mod_cast h
  | neg ih n => simpa only [preΨ_neg, coeff_neg, neg_ne_zero, Int.natAbs_neg, even_neg]
        using ih n <| neg_ne_zero.mp <| by exact_mod_cast h


@[simp]
lemma natDegree_preΨ {n : ℤ} (h : (n : R) ≠ 0) :
    (W.preΨ n).natDegree = (n.natAbs ^ 2 - if Even n then 4 else 1) / 2 :=
  natDegree_eq_of_le_of_coeff_ne_zero (W.natDegree_preΨ_le n) <| W.coeff_preΨ_ne_zero h


lemma natDegree_preΨ_pos {n : ℤ} (hn : 2 < n.natAbs) (h : (n : R) ≠ 0) :
    0 < (W.preΨ n).natDegree := by
  induction n using Int.negInduction with
  | nat n => simpa only [preΨ_ofNat] using W.natDegree_preΨ'_pos hn <| by exact_mod_cast h
  | neg ih n => simpa only [preΨ_neg, natDegree_neg]
        using ih n (by rwa [← Int.natAbs_neg]) <| neg_ne_zero.mp <| by exact_mod_cast h


@[simp]
lemma leadingCoeff_preΨ {n : ℤ} (h : (n : R) ≠ 0) :
    (W.preΨ n).leadingCoeff = if Even n then n / 2 else n := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Int
    h : Ne (↑n) 0
    ⊢ Eq (W.preΨ n).leadingCoeff ↑(ite (Even n) (HDiv.hDiv n 2) n)
  -/
  rw [leadingCoeff, W.natDegree_preΨ h, coeff_preΨ]
  /-
    🎉 no goals
  -/


lemma preΨ_ne_zero [Nontrivial R] {n : ℤ} (h : (n : R) ≠ 0) : W.preΨ n ≠ 0 := by
  induction n using Int.negInduction with
  | nat n => simpa only [preΨ_ofNat] using W.preΨ'_ne_zero <| by exact_mod_cast h
  | neg ih n => simpa only [preΨ_neg, neg_ne_zero]
        using ih n <| neg_ne_zero.mp <| by exact_mod_cast h


private lemma natDegree_coeff_ΨSq_ofNat (n : ℕ) :
    (W.ΨSq n).natDegree ≤ n ^ 2 - 1 ∧ (W.ΨSq n).coeff (n ^ 2 - 1) = (n ^ 2 : ℤ) := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    ⊢ And (LE.le (W.ΨSq ↑n).natDegree (HSub.hSub (HPow.hPow n 2) 1)) (Eq ((W.ΨSq ↑ …
  -/
  let dp {m n p} : _ → (p ^ n : R[X]).natDegree ≤ n * m := natDegree_pow_le_of_le n
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
    ⊢ And (LE.le (W.ΨSq ↑n).natDegree (HSub.hSub (HPow.hPow n 2) 1)) (Eq ((W.ΨSq ↑ …
  -/
  let h {n} := W.natDegree_coeff_preΨ' n
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
    h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
    ⊢ And (LE.le (W.ΨSq ↑n).natDegree (HSub.hSub (HPow.hPow n 2) 1)) (Eq ((W.ΨSq ↑ …
  -/
  rcases n with _ | n
    /-
      case zero
      R : Type u
      inst✝ : CommRing R
      W : WeierstrassCurve R
      dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
      h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
      ⊢ And (LE.le (W.ΨSq ↑0).natDegree (HSub.hSub (HPow.hPow 0 2) 1)) (Eq ((W.ΨSq ↑ …
    -/
  · simp
    /-
      🎉 no goals
    -/
  have hd : (n + 1) ^ 2 - 1 = 2 * expDegree (n + 1) + if Even (n + 1) then 3 else 0 := by
    push_cast [← @Nat.cast_inj ℤ, add_sq, expDegree_cast (by omega : n + 1 ≠ 0)]
    split_ifs <;> ring1
  have hc : (n + 1 : ℕ) ^ 2 = expCoeff (n + 1) ^ 2 * if Even (n + 1) then 4 else 1 := by
    push_cast [← @Int.cast_inj ℚ, expCoeff_cast]
    split_ifs <;> ring1
  /-
    case succ
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
    h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
    n : Nat
    hd : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd n 1) 2) 1) (HAdd.hAdd (HMul.hMul 2 (W …
    hc : Eq (HPow.hPow (↑(HAdd.hAdd n 1)) 2) (HMul.hMul (HPow.hPow (WeierstrassCur …
    ⊢ And (LE.le (W.ΨSq ↑(HAdd.hAdd n 1)).natDegree (HSub.hSub (HPow.hPow (HAdd.hA …
  -/
  rw [ΨSq_ofNat, hd]
  /-
    case succ
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
    h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
    n : Nat
    hd : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd n 1) 2) 1) (HAdd.hAdd (HMul.hMul 2 (W …
    hc : Eq (HPow.hPow (↑(HAdd.hAdd n 1)) 2) (HMul.hMul (HPow.hPow (WeierstrassCur …
    ⊢ And (LE.le (HMul.hMul (HPow.hPow (W.preΨ' (HAdd.hAdd n 1)) 2) (ite (Even (HA …
  -/
  constructor
    /-
      case succ.left
      R : Type u
      inst✝ : CommRing R
      W : WeierstrassCurve R
      dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
      h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
      n : Nat
      hd : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd n 1) 2) 1) (HAdd.hAdd (HMul.hMul 2 (W …
      hc : Eq (HPow.hPow (↑(HAdd.hAdd n 1)) 2) (HMul.hMul (HPow.hPow (WeierstrassCur …
      ⊢ LE.le (HMul.hMul (HPow.hPow (W.preΨ' (HAdd.hAdd n 1)) 2) (ite (Even (HAdd.hA …
    -/
  · refine natDegree_mul_le_of_le (dp h.1) ?_
    /-
      case succ.left
      R : Type u
      inst✝ : CommRing R
      W : WeierstrassCurve R
      dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
      h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
      n : Nat
      hd : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd n 1) 2) 1) (HAdd.hAdd (HMul.hMul 2 (W …
      hc : Eq (HPow.hPow (↑(HAdd.hAdd n 1)) 2) (HMul.hMul (HPow.hPow (WeierstrassCur …
      ⊢ LE.le (ite (Even (HAdd.hAdd n 1)) W.Ψ₂Sq 1).natDegree (ite (Even (HAdd.hAdd  …
    -/
                  /-
                    🎉 no goals
                  -/
    split_ifs <;> simp only [apply_ite natDegree, natDegree_one.le, W.natDegree_Ψ₂Sq_le]
                  /-
                    🎉 no goals
                  -/
  · rw [coeff_mul_of_natDegree_le (dp h.1), coeff_pow_of_natDegree_le h.1, h.2, apply_ite₂ coeff,
      coeff_Ψ₂Sq, coeff_one_zero, hc]
      /-
        case succ.right
        R : Type u
        inst✝ : CommRing R
        W : WeierstrassCurve R
        dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
        h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
        n : Nat
        hd : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd n 1) 2) 1) (HAdd.hAdd (HMul.hMul 2 (W …
        hc : Eq (HPow.hPow (↑(HAdd.hAdd n 1)) 2) (HMul.hMul (HPow.hPow (WeierstrassCur …
        ⊢ Eq (HMul.hMul (HPow.hPow (↑(WeierstrassCurve.expCoeff (HAdd.hAdd n 1))) 2) ( …
      -/
    · norm_cast
      /-
        🎉 no goals
      -/
    /-
      case succ.right
      R : Type u
      inst✝ : CommRing R
      W : WeierstrassCurve R
      dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
      h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
      n : Nat
      hd : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd n 1) 2) 1) (HAdd.hAdd (HMul.hMul 2 (W …
      hc : Eq (HPow.hPow (↑(HAdd.hAdd n 1)) 2) (HMul.hMul (HPow.hPow (WeierstrassCur …
      ⊢ LE.le (ite (Even (HAdd.hAdd n 1)) W.Ψ₂Sq 1).natDegree (ite (Even (HAdd.hAdd  …
    -/
                  /-
                    🎉 no goals
                  -/
    split_ifs <;> simp only [apply_ite natDegree, natDegree_one.le, W.natDegree_Ψ₂Sq_le]
                  /-
                    🎉 no goals
                  -/


lemma natDegree_ΨSq_le (n : ℤ) : (W.ΨSq n).natDegree ≤ n.natAbs ^ 2 - 1 := by
  induction n using Int.negInduction with
  | nat n => exact (W.natDegree_coeff_ΨSq_ofNat n).left
  | neg ih => simp only [ΨSq_neg, Int.natAbs_neg, ih]


@[simp]
lemma coeff_ΨSq (n : ℤ) : (W.ΨSq n).coeff (n.natAbs ^ 2 - 1) = n ^ 2 := by
  induction n using Int.negInduction with
  | nat n => exact_mod_cast (W.natDegree_coeff_ΨSq_ofNat n).right
  | neg ih => simp_rw [ΨSq_neg, Int.natAbs_neg, ← Int.cast_pow, neg_sq, Int.cast_pow, ih]


lemma coeff_ΨSq_ne_zero [NoZeroDivisors R] {n : ℤ} (h : (n : R) ≠ 0) :
    (W.ΨSq n).coeff (n.natAbs ^ 2 - 1) ≠ 0 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : NoZeroDivisors R
    n : Int
    h : Ne (↑n) 0
    ⊢ Ne ((W.ΨSq n).coeff (HSub.hSub (HPow.hPow n.natAbs 2) 1)) 0
  -/
  rwa [coeff_ΨSq, pow_ne_zero_iff two_ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
lemma natDegree_ΨSq [NoZeroDivisors R] {n : ℤ} (h : (n : R) ≠ 0) :
    (W.ΨSq n).natDegree = n.natAbs ^ 2 - 1 :=
  natDegree_eq_of_le_of_coeff_ne_zero (W.natDegree_ΨSq_le n) <| W.coeff_ΨSq_ne_zero h


lemma natDegree_ΨSq_pos [NoZeroDivisors R] {n : ℤ} (hn : 1 < n.natAbs) (h : (n : R) ≠ 0) :
    0 < (W.ΨSq n).natDegree := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : NoZeroDivisors R
    n : Int
    hn : LT.lt 1 n.natAbs
    h : Ne (↑n) 0
    ⊢ LT.lt 0 (W.ΨSq n).natDegree
  -/
  rwa [W.natDegree_ΨSq h, Nat.sub_pos_iff_lt, Nat.one_lt_pow_iff two_ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
lemma leadingCoeff_ΨSq [NoZeroDivisors R] {n : ℤ} (h : (n : R) ≠ 0) :
    (W.ΨSq n).leadingCoeff = n ^ 2 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : NoZeroDivisors R
    n : Int
    h : Ne (↑n) 0
    ⊢ Eq (W.ΨSq n).leadingCoeff (HPow.hPow (↑n) 2)
  -/
  rw [leadingCoeff, W.natDegree_ΨSq h, coeff_ΨSq]
  /-
    🎉 no goals
  -/


lemma ΨSq_ne_zero [NoZeroDivisors R] {n : ℤ} (h : (n : R) ≠ 0) : W.ΨSq n ≠ 0 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : NoZeroDivisors R
    n : Int
    h : Ne (↑n) 0
    ⊢ Ne (W.ΨSq n) 0
  -/
  by_cases hn : 1 < n.natAbs
    /-
      case pos
      R : Type u
      inst✝¹ : CommRing R
      W : WeierstrassCurve R
      inst✝ : NoZeroDivisors R
      n : Int
      h : Ne (↑n) 0
      hn : LT.lt 1 n.natAbs
      ⊢ Ne (W.ΨSq n) 0
    -/
  · exact ne_zero_of_natDegree_gt <| W.natDegree_ΨSq_pos hn h
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝¹ : CommRing R
      W : WeierstrassCurve R
      inst✝ : NoZeroDivisors R
      n : Int
      h : Ne (↑n) 0
      hn : Not (LT.lt 1 n.natAbs)
      ⊢ Ne (W.ΨSq n) 0
    -/
  · rcases hm : n.natAbs with _ | m
      /-
        case neg.zero
        R : Type u
        inst✝¹ : CommRing R
        W : WeierstrassCurve R
        inst✝ : NoZeroDivisors R
        n : Int
        h : Ne (↑n) 0
        hn : Not (LT.lt 1 n.natAbs)
        hm : Eq n.natAbs 0
        ⊢ Ne (W.ΨSq n) 0
      -/
    · push_cast [Int.natAbs_eq_zero.mp hm, ne_self_iff_false] at h
      /-
        🎉 no goals
      -/
      /-
        case neg.succ
        R : Type u
        inst✝¹ : CommRing R
        W : WeierstrassCurve R
        inst✝ : NoZeroDivisors R
        n : Int
        h : Ne (↑n) 0
        hn : Not (LT.lt 1 n.natAbs)
        m : Nat
        hm : Eq n.natAbs (HAdd.hAdd m 1)
        ⊢ Ne (W.ΨSq n) 0
      -/
    · rcases Int.natAbs_eq_iff.mp hm with rfl | rfl <;>
        /-
          case neg.succ.inl
          R : Type u
          inst✝¹ : CommRing R
          W : WeierstrassCurve R
          inst✝ : NoZeroDivisors R
          m : Nat
          h : Ne (↑↑(HAdd.hAdd m 1)) 0
          hn : Not (LT.lt 1 (↑(HAdd.hAdd m 1)).natAbs)
          hm : Eq (↑(HAdd.hAdd m 1)).natAbs (HAdd.hAdd m 1)
          ⊢ Ne (W.ΨSq ↑(HAdd.hAdd m 1)) 0
        -/
        rw [hm, Nat.lt_add_left_iff_pos, Nat.not_lt_eq, Nat.le_zero] at hn <;>
        /-
          case neg.succ.inl
          R : Type u
          inst✝¹ : CommRing R
          W : WeierstrassCurve R
          inst✝ : NoZeroDivisors R
          m : Nat
          h : Ne (↑↑(HAdd.hAdd m 1)) 0
          hn : Eq m 0
          hm : Eq (↑(HAdd.hAdd m 1)).natAbs (HAdd.hAdd m 1)
          ⊢ Ne (W.ΨSq ↑(HAdd.hAdd m 1)) 0
        -/
        push_cast [hn, ΨSq_neg, ΨSq_one] <;>
        /-
          case neg.succ.inl
          R : Type u
          inst✝¹ : CommRing R
          W : WeierstrassCurve R
          inst✝ : NoZeroDivisors R
          m : Nat
          h : Ne (↑↑(HAdd.hAdd m 1)) 0
          hn : Eq m 0
          hm : Eq (↑(HAdd.hAdd m 1)).natAbs (HAdd.hAdd m 1)
          ⊢ Ne 1 0
        -/
        /-
          🎉 no goals
        -/
        exact fun h' => h <| C_injective <| by push_cast [hn, C_neg, C_1, h', neg_zero, C_0]; rfl
        /-
          🎉 no goals
        -/


private lemma natDegree_coeff_Φ_ofNat (n : ℕ) :
    (W.Φ n).natDegree ≤ n ^ 2 ∧ (W.Φ n).coeff (n ^ 2) = 1 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    ⊢ And (LE.le (W.Φ ↑n).natDegree (HPow.hPow n 2)) (Eq ((W.Φ ↑n).coeff (HPow.hPo …
  -/
  let dm {m n p q} : _ → _ → (p * q : R[X]).natDegree ≤ m + n := natDegree_mul_le_of_le
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
    ⊢ And (LE.le (W.Φ ↑n).natDegree (HPow.hPow n 2)) (Eq ((W.Φ ↑n).coeff (HPow.hPo …
  -/
  let dp {m n p} : _ → (p ^ n : R[X]).natDegree ≤ n * m := natDegree_pow_le_of_le n
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
    dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
    ⊢ And (LE.le (W.Φ ↑n).natDegree (HPow.hPow n 2)) (Eq ((W.Φ ↑n).coeff (HPow.hPo …
  -/
  let cm {m n p q} : _ → _ → (p * q : R[X]).coeff (m + n) = _ := coeff_mul_of_natDegree_le
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
    dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
    cm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
    ⊢ And (LE.le (W.Φ ↑n).natDegree (HPow.hPow n 2)) (Eq ((W.Φ ↑n).coeff (HPow.hPo …
  -/
  let h {n} := W.natDegree_coeff_preΨ' n
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    n : Nat
    dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
    dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
    cm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
    h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
    ⊢ And (LE.le (W.Φ ↑n).natDegree (HPow.hPow n 2)) (Eq ((W.Φ ↑n).coeff (HPow.hPo …
  -/
  rcases n with _ | _ | n
    /-
      case zero
      R : Type u
      inst✝ : CommRing R
      W : WeierstrassCurve R
      dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
      dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
      cm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
      h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
      ⊢ And (LE.le (W.Φ ↑0).natDegree (HPow.hPow 0 2)) (Eq ((W.Φ ↑0).coeff (HPow.hPo …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ.zero
      R : Type u
      inst✝ : CommRing R
      W : WeierstrassCurve R
      dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
      dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
      cm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
      h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
      ⊢ And (LE.le (W.Φ ↑(HAdd.hAdd 0 1)).natDegree (HPow.hPow (HAdd.hAdd 0 1) 2)) ( …
    -/
  · simp [natDegree_X_le]
    /-
      🎉 no goals
    -/
  have hd : (n + 1 + 1) ^ 2 = 1 + 2 * expDegree (n + 2) + if Even (n + 1) then 0 else 3 := by
    push_cast [← @Nat.cast_inj ℤ, expDegree_cast (by omega : n + 2 ≠ 0), Nat.even_add_one, ite_not]
    split_ifs <;> ring1
  have hd' : (n + 1 + 1) ^ 2 =
      expDegree (n + 3) + expDegree (n + 1) + if Even (n + 1) then 3 else 0 := by
    push_cast [← @Nat.cast_inj ℤ, ← mul_left_cancel_iff_of_pos (b := (_ ^ 2 : ℤ)) two_pos, mul_add,
      expDegree_cast (by omega : n + 3 ≠ 0), expDegree_cast (by omega : n + 1 ≠ 0),
      Nat.even_add_one, ite_not]
    split_ifs <;> ring1
  have hc : (1 : ℤ) = 1 * expCoeff (n + 2) ^ 2 * (if Even (n + 1) then 1 else 4) -
      expCoeff (n + 3) * expCoeff (n + 1) * (if Even (n + 1) then 4 else 1) := by
    push_cast [← @Int.cast_inj ℚ, expCoeff_cast, Nat.even_add_one, ite_not]
    split_ifs <;> ring1
  /-
    case succ.succ
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
    dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
    cm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
    h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
    n : Nat
    hd : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd n 1) 1) 2) (HAdd.hAdd (HAdd.hAdd 1 (H …
    hd' : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd n 1) 1) 2) (HAdd.hAdd (HAdd.hAdd (We …
    hc : Eq 1 (HSub.hSub (HMul.hMul (HMul.hMul 1 (HPow.hPow (WeierstrassCurve.expC …
    ⊢ And (LE.le (W.Φ ↑(HAdd.hAdd (HAdd.hAdd n 1) 1)).natDegree (HPow.hPow (HAdd.h …
  -/
  rw [Nat.cast_add, Nat.cast_one, Φ_ofNat]
  /-
    case succ.succ
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
    dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
    cm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
    h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
    n : Nat
    hd : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd n 1) 1) 2) (HAdd.hAdd (HAdd.hAdd 1 (H …
    hd' : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd n 1) 1) 2) (HAdd.hAdd (HAdd.hAdd (We …
    hc : Eq 1 (HSub.hSub (HMul.hMul (HMul.hMul 1 (HPow.hPow (WeierstrassCurve.expC …
    ⊢ And (LE.le (HSub.hSub (HMul.hMul (HMul.hMul Polynomial.X (HPow.hPow (W.preΨ' …
  -/
  constructor
    /-
      case succ.succ.left
      R : Type u
      inst✝ : CommRing R
      W : WeierstrassCurve R
      dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
      dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
      cm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
      h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
      n : Nat
      hd : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd n 1) 1) 2) (HAdd.hAdd (HAdd.hAdd 1 (H …
      hd' : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd n 1) 1) 2) (HAdd.hAdd (HAdd.hAdd (We …
      hc : Eq 1 (HSub.hSub (HMul.hMul (HMul.hMul 1 (HPow.hPow (WeierstrassCurve.expC …
      ⊢ LE.le (HSub.hSub (HMul.hMul (HMul.hMul Polynomial.X (HPow.hPow (W.preΨ' (HAd …
    -/
  · nth_rw 1 [← max_self <| (_ + _) ^ 2, hd, hd']
    /-
      case succ.succ.left
      R : Type u
      inst✝ : CommRing R
      W : WeierstrassCurve R
      dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
      dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
      cm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
      h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
      n : Nat
      hd : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd n 1) 1) 2) (HAdd.hAdd (HAdd.hAdd 1 (H …
      hd' : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd n 1) 1) 2) (HAdd.hAdd (HAdd.hAdd (We …
      hc : Eq 1 (HSub.hSub (HMul.hMul (HMul.hMul 1 (HPow.hPow (WeierstrassCurve.expC …
      ⊢ LE.le (HSub.hSub (HMul.hMul (HMul.hMul Polynomial.X (HPow.hPow (W.preΨ' (HAd …
    -/
    refine natDegree_sub_le_of_le (dm (dm natDegree_X_le (dp h.1)) ?_) (dm (dm h.1 h.1) ?_)
    /-
      case succ.succ.left.refine_1
      R : Type u
      inst✝ : CommRing R
      W : WeierstrassCurve R
      dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
      dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
      cm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
      h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
      n : Nat
      hd : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd n 1) 1) 2) (HAdd.hAdd (HAdd.hAdd 1 (H …
      hd' : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd n 1) 1) 2) (HAdd.hAdd (HAdd.hAdd (We …
      hc : Eq 1 (HSub.hSub (HMul.hMul (HMul.hMul 1 (HPow.hPow (WeierstrassCurve.expC …
      ⊢ LE.le (ite (Even (HAdd.hAdd n 1)) 1 W.Ψ₂Sq).natDegree (ite (Even (HAdd.hAdd  …
    -/
    all_goals split_ifs <;> simp only [apply_ite natDegree, natDegree_one.le, W.natDegree_Ψ₂Sq_le]
    /-
      🎉 no goals
    -/
  · nth_rw 1 [coeff_sub, hd, hd', cm (dm natDegree_X_le (dp h.1)), cm natDegree_X_le (dp h.1),
      coeff_X_one, coeff_pow_of_natDegree_le h.1, h.2, apply_ite₂ coeff, coeff_one_zero, coeff_Ψ₂Sq,
      cm (dm h.1 h.1), cm h.1 h.1, h.2, h.2, apply_ite₂ coeff, coeff_one_zero, coeff_Ψ₂Sq]
    /-
      case succ.succ.right
      R : Type u
      inst✝ : CommRing R
      W : WeierstrassCurve R
      dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
      dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
      cm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
      h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
      n : Nat
      hd : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd n 1) 1) 2) (HAdd.hAdd (HAdd.hAdd 1 (H …
      hd' : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd n 1) 1) 2) (HAdd.hAdd (HAdd.hAdd (We …
      hc : Eq 1 (HSub.hSub (HMul.hMul (HMul.hMul 1 (HPow.hPow (WeierstrassCurve.expC …
      ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul 1 (HPow.hPow (↑(WeierstrassCurve.expCoef …
    -/
    conv_rhs => rw [← Int.cast_one, hc]
      /-
        case succ.succ.right
        R : Type u
        inst✝ : CommRing R
        W : WeierstrassCurve R
        dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
        dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
        cm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
        h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
        n : Nat
        hd : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd n 1) 1) 2) (HAdd.hAdd (HAdd.hAdd 1 (H …
        hd' : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd n 1) 1) 2) (HAdd.hAdd (HAdd.hAdd (We …
        hc : Eq 1 (HSub.hSub (HMul.hMul (HMul.hMul 1 (HPow.hPow (WeierstrassCurve.expC …
        ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul 1 (HPow.hPow (↑(WeierstrassCurve.expCoef …
      -/
    · norm_cast
      /-
        🎉 no goals
      -/
    /-
      case succ.succ.right
      R : Type u
      inst✝ : CommRing R
      W : WeierstrassCurve R
      dm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
      dp : ∀ {m n : Nat} {p : Polynomial R}, LE.le p.natDegree m → LE.le (HPow.hPow  …
      cm : ∀ {m n : Nat} {p q : Polynomial R}, LE.le p.natDegree m → LE.le q.natDegr …
      h : ∀ {n : Nat}, And (LE.le (W.preΨ' n).natDegree (WeierstrassCurve.expDegree  …
      n : Nat
      hd : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd n 1) 1) 2) (HAdd.hAdd (HAdd.hAdd 1 (H …
      hd' : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd n 1) 1) 2) (HAdd.hAdd (HAdd.hAdd (We …
      hc : Eq 1 (HSub.hSub (HMul.hMul (HMul.hMul 1 (HPow.hPow (WeierstrassCurve.expC …
      ⊢ LE.le (ite (Even (HAdd.hAdd n 1)) W.Ψ₂Sq 1).natDegree (ite (Even (HAdd.hAdd  …
    -/
    all_goals split_ifs <;> simp only [apply_ite natDegree, natDegree_one.le, W.natDegree_Ψ₂Sq_le]
    /-
      🎉 no goals
    -/


lemma natDegree_Φ_le (n : ℤ) : (W.Φ n).natDegree ≤ n.natAbs ^ 2 := by
  induction n using Int.negInduction with
  | nat n => exact (W.natDegree_coeff_Φ_ofNat n).left
  | neg ih => simp only [Φ_neg, Int.natAbs_neg, ih]


@[simp]
lemma coeff_Φ (n : ℤ) : (W.Φ n).coeff (n.natAbs ^ 2) = 1 := by
  induction n using Int.negInduction with
  | nat n => exact (W.natDegree_coeff_Φ_ofNat n).right
  | neg ih => simp only [Φ_neg, Int.natAbs_neg, ih]


lemma coeff_Φ_ne_zero [Nontrivial R] (n : ℤ) : (W.Φ n).coeff (n.natAbs ^ 2) ≠ 0 :=
  W.coeff_Φ n ▸ one_ne_zero


@[simp]
lemma natDegree_Φ [Nontrivial R] (n : ℤ) : (W.Φ n).natDegree = n.natAbs ^ 2 :=
  natDegree_eq_of_le_of_coeff_ne_zero (W.natDegree_Φ_le n) <| W.coeff_Φ_ne_zero n


lemma natDegree_Φ_pos [Nontrivial R] {n : ℤ} (hn : n ≠ 0) : 0 < (W.Φ n).natDegree := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : Nontrivial R
    n : Int
    hn : Ne n 0
    ⊢ LT.lt 0 (W.Φ n).natDegree
  -/
  rwa [natDegree_Φ, pow_pos_iff two_ne_zero, Int.natAbs_pos]
  /-
    🎉 no goals
  -/


@[simp]
lemma leadingCoeff_Φ [Nontrivial R] (n : ℤ) : (W.Φ n).leadingCoeff = 1 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : Nontrivial R
    n : Int
    ⊢ Eq (W.Φ n).leadingCoeff 1
  -/
  rw [leadingCoeff, natDegree_Φ, coeff_Φ]
  /-
    🎉 no goals
  -/


lemma Φ_ne_zero [Nontrivial R] (n : ℤ) : W.Φ n ≠ 0 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : Nontrivial R
    n : Int
    ⊢ Ne (W.Φ n) 0
  -/
  by_cases hn : n = 0
    /-
      case pos
      R : Type u
      inst✝¹ : CommRing R
      W : WeierstrassCurve R
      inst✝ : Nontrivial R
      n : Int
      hn : Eq n 0
      ⊢ Ne (W.Φ n) 0
    -/
  · simpa only [hn, Φ_zero] using one_ne_zero
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝¹ : CommRing R
      W : WeierstrassCurve R
      inst✝ : Nontrivial R
      n : Int
      hn : Not (Eq n 0)
      ⊢ Ne (W.Φ n) 0
    -/
  · exact ne_zero_of_natDegree_gt <| W.natDegree_Φ_pos hn
    /-
      🎉 no goals
    -/


