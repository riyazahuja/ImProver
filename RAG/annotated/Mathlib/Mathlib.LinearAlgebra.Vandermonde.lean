/-- `vandermonde v` is the square matrix with `i`th row equal to `1, v i, v i ^ 2, v i ^ 3, ...`.
-/
def vandermonde {n : ℕ} (v : Fin n → R) : Matrix (Fin n) (Fin n) R := .of fun i j => v i ^ (j : ℕ)


@[simp]
theorem vandermonde_apply {n : ℕ} (v : Fin n → R) (i j) : vandermonde v i j = v i ^ (j : ℕ) :=
  rfl


@[simp]
theorem vandermonde_cons {n : ℕ} (v0 : R) (v : Fin n → R) :
    vandermonde (Fin.cons v0 v : Fin n.succ → R) =
      Fin.cons (fun (j : Fin n.succ) => v0 ^ (j : ℕ)) fun i => Fin.cons 1
      fun j => v i * vandermonde v i j := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v0 : R
    v : Fin n → R
    ⊢ Eq (Matrix.vandermonde (Fin.cons v0 v)) (Fin.cons (fun j => HPow.hPow v0 ↑j) …
  -/
  ext i j
  /-
    case a
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v0 : R
    v : Fin n → R
    i j : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Matrix.vandermonde (Fin.cons v0 v) i j) (Fin.cons (fun j => HPow.hPow v0 …
  -/
  refine Fin.cases (by simp) (fun i => ?_) i
  /-
    case a
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v0 : R
    v : Fin n → R
    i✝ j : Fin (HAdd.hAdd n 1)
    i : Fin n
    ⊢ Eq (Matrix.vandermonde (Fin.cons v0 v) i.succ j) (Fin.cons (fun j => HPow.hP …
  -/
  refine Fin.cases (by simp) (fun j => ?_) j
  /-
    case a
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v0 : R
    v : Fin n → R
    i✝ j✝ : Fin (HAdd.hAdd n 1)
    i j : Fin n
    ⊢ Eq (Matrix.vandermonde (Fin.cons v0 v) i.succ j.succ) (Fin.cons (fun j => HP …
  -/
  simp [pow_succ']
  /-
    🎉 no goals
  -/


theorem vandermonde_succ {n : ℕ} (v : Fin n.succ → R) :
    vandermonde v = .of
      Fin.cons (fun (j : Fin n.succ) => v 0 ^ (j : ℕ)) fun i =>
        Fin.cons 1 fun j => v i.succ * vandermonde (Fin.tail v) i j := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v : Fin n.succ → R
    ⊢ Eq (Matrix.vandermonde v) (Matrix.of Fin.cons (fun j => HPow.hPow (v 0) ↑j)  …
  -/
  conv_lhs => rw [← Fin.cons_self_tail v, vandermonde_cons]
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v : Fin n.succ → R
    ⊢ Eq (Fin.cons (fun j => HPow.hPow (v 0) ↑j) fun i => Fin.cons 1 fun j => HMul …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem vandermonde_mul_vandermonde_transpose {n : ℕ} (v w : Fin n → R) (i j) :
    (vandermonde v * (vandermonde w)ᵀ) i j = ∑ k : Fin n, (v i * w j) ^ (k : ℕ) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v w : Fin n → R
    i j : Fin n
    ⊢ Eq (HMul.hMul (Matrix.vandermonde v) (Matrix.vandermonde w).transpose i j) ( …
  -/
  simp only [vandermonde_apply, Matrix.mul_apply, Matrix.transpose_apply, mul_pow]
  /-
    🎉 no goals
  -/


theorem vandermonde_transpose_mul_vandermonde {n : ℕ} (v : Fin n → R) (i j) :
    ((vandermonde v)ᵀ * vandermonde v) i j = ∑ k : Fin n, v k ^ (i + j : ℕ) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v : Fin n → R
    i j : Fin n
    ⊢ Eq (HMul.hMul (Matrix.vandermonde v).transpose (Matrix.vandermonde v) i j) ( …
  -/
  simp only [vandermonde_apply, Matrix.mul_apply, Matrix.transpose_apply, pow_add]
  /-
    🎉 no goals
  -/


theorem det_vandermonde {n : ℕ} (v : Fin n → R) :
    det (vandermonde v) = ∏ i : Fin n, ∏ j ∈ Ioi i, (v j - v i) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v : Fin n → R
    ⊢ Eq (Matrix.vandermonde v).det (Finset.univ.prod fun i => (Finset.Ioi i).prod …
  -/
  unfold vandermonde
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v : Fin n → R
    ⊢ Eq (Matrix.of fun i j => HPow.hPow (v i) ↑j).det (Finset.univ.prod fun i =>  …
  -/
  induction' n with n ih
    /-
      case zero
      R : Type u_1
      inst✝ : CommRing R
      v : Fin 0 → R
      ⊢ Eq (Matrix.of fun i j => HPow.hPow (v i) ↑j).det (Finset.univ.prod fun i =>  …
    -/
  · exact det_eq_one_of_card_eq_zero (Fintype.card_fin 0)
    /-
      🎉 no goals
    -/
  calc
    det (of fun i j : Fin n.succ => v i ^ (j : ℕ)) =
        det
          (of fun i j : Fin n.succ =>
            Matrix.vecCons (v 0 ^ (j : ℕ)) (fun i => v (Fin.succ i) ^ (j : ℕ) - v 0 ^ (j : ℕ)) i) :=
      det_eq_of_forall_row_eq_smul_add_const (Matrix.vecCons 0 1) 0 (Fin.cons_zero _ _) ?_
    _ =
        det
          (of fun i j : Fin n =>
            Matrix.vecCons (v 0 ^ (j.succ : ℕ))
              (fun i : Fin n => v (Fin.succ i) ^ (j.succ : ℕ) - v 0 ^ (j.succ : ℕ))
              (Fin.succAbove 0 i)) := by
      simp_rw [det_succ_column_zero, Fin.sum_univ_succ, of_apply, Matrix.cons_val_zero, submatrix,
        of_apply, Matrix.cons_val_succ, Fin.val_zero, pow_zero, one_mul, sub_self,
        mul_zero, zero_mul, Finset.sum_const_zero, add_zero]
    _ =
        det
          (of fun i j : Fin n =>
              (v (Fin.succ i) - v 0) *
                ∑ k ∈ Finset.range (j + 1 : ℕ), v i.succ ^ k * v 0 ^ (j - k : ℕ) :
            Matrix _ _ R) := by
      congr
      ext i j
      rw [Fin.succAbove_zero, Matrix.cons_val_succ, Fin.val_succ, mul_comm]
      exact (geom_sum₂_mul (v i.succ) (v 0) (j + 1 : ℕ)).symm
    _ =
        (∏ i ∈ Finset.univ, (v (Fin.succ i) - v 0)) *
          det fun i j : Fin n =>
            ∑ k ∈ Finset.range (j + 1 : ℕ), v i.succ ^ k * v 0 ^ (j - k : ℕ) :=
      (det_mul_column (fun i => v (Fin.succ i) - v 0) _)
    _ = (∏ i ∈ Finset.univ, (v (Fin.succ i) - v 0)) *
    det (of fun i j : Fin n => v (Fin.succ i) ^ (j : ℕ)) := congr_arg _ ?_
    _ = ∏ i : Fin n.succ, ∏ j ∈ Ioi i, (v j - v i) := by
      simp_rw [Fin.prod_univ_succ, Fin.prod_Ioi_zero, Fin.prod_Ioi_succ]
      have h : (of fun i j : Fin n ↦ v i.succ ^ (j : ℕ)).det =
          ∏ x : Fin n, ∏ y ∈ Ioi x, (v y.succ - v x.succ) := by
        simpa using ih (v ∘ Fin.succ)
      rw [h]

    /-
      case succ.calc_1
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      ih : ∀ (v : Fin n → R), Eq (Matrix.of fun i j => HPow.hPow (v i) ↑j).det (Fins …
      v : Fin (HAdd.hAdd n 1) → R
      ⊢ ∀ (i j : Fin n.succ), Eq (Matrix.of (fun i j => HPow.hPow (v i) ↑j) i j) (HA …
    -/
  · intro i j
    /-
      case succ.calc_1
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      ih : ∀ (v : Fin n → R), Eq (Matrix.of fun i j => HPow.hPow (v i) ↑j).det (Fins …
      v : Fin (HAdd.hAdd n 1) → R
      i j : Fin n.succ
      ⊢ Eq (Matrix.of (fun i j => HPow.hPow (v i) ↑j) i j) (HAdd.hAdd (Matrix.of (fu …
    -/
    simp_rw [of_apply]
    /-
      case succ.calc_1
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      ih : ∀ (v : Fin n → R), Eq (Matrix.of fun i j => HPow.hPow (v i) ↑j).det (Fins …
      v : Fin (HAdd.hAdd n 1) → R
      i j : Fin n.succ
      ⊢ Eq (HPow.hPow (v i) ↑j) (HAdd.hAdd (Matrix.vecCons (HPow.hPow (v 0) ↑j) (fun …
    -/
    rw [Matrix.cons_val_zero]
    /-
      case succ.calc_1
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      ih : ∀ (v : Fin n → R), Eq (Matrix.of fun i j => HPow.hPow (v i) ↑j).det (Fins …
      v : Fin (HAdd.hAdd n 1) → R
      i j : Fin n.succ
      ⊢ Eq (HPow.hPow (v i) ↑j) (HAdd.hAdd (Matrix.vecCons (HPow.hPow (v 0) ↑j) (fun …
    -/
    refine Fin.cases ?_ (fun i => ?_) i
      /-
        case succ.calc_1.refine_1
        R : Type u_1
        inst✝ : CommRing R
        n : Nat
        ih : ∀ (v : Fin n → R), Eq (Matrix.of fun i j => HPow.hPow (v i) ↑j).det (Fins …
        v : Fin (HAdd.hAdd n 1) → R
        i j : Fin n.succ
        ⊢ Eq (HPow.hPow (v 0) ↑j) (HAdd.hAdd (Matrix.vecCons (HPow.hPow (v 0) ↑j) (fun …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case succ.calc_1.refine_2
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      ih : ∀ (v : Fin n → R), Eq (Matrix.of fun i j => HPow.hPow (v i) ↑j).det (Fins …
      v : Fin (HAdd.hAdd n 1) → R
      i✝ j : Fin n.succ
      i : Fin n
      ⊢ Eq (HPow.hPow (v i.succ) ↑j) (HAdd.hAdd (Matrix.vecCons (HPow.hPow (v 0) ↑j) …
    -/
    rw [Matrix.cons_val_succ, Matrix.cons_val_succ, Pi.one_apply]
    /-
      case succ.calc_1.refine_2
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      ih : ∀ (v : Fin n → R), Eq (Matrix.of fun i j => HPow.hPow (v i) ↑j).det (Fins …
      v : Fin (HAdd.hAdd n 1) → R
      i✝ j : Fin n.succ
      i : Fin n
      ⊢ Eq (HPow.hPow (v i.succ) ↑j) (HAdd.hAdd (HSub.hSub (HPow.hPow (v i.succ) ↑j) …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case succ.calc_2
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      ih : ∀ (v : Fin n → R), Eq (Matrix.of fun i j => HPow.hPow (v i) ↑j).det (Fins …
      v : Fin (HAdd.hAdd n 1) → R
      ⊢ Eq (Matrix.det fun i j => (Finset.range (HAdd.hAdd (↑j) 1)).sum fun k => HMu …
    -/
  · cases n
    · rw [det_eq_one_of_card_eq_zero (Fintype.card_fin 0),
      det_eq_one_of_card_eq_zero (Fintype.card_fin 0)]
    /-
      case succ.calc_2.succ
      R : Type u_1
      inst✝ : CommRing R
      n✝ : Nat
      ih : ∀ (v : Fin (HAdd.hAdd n✝ 1) → R), Eq (Matrix.of fun i j => HPow.hPow (v i …
      v : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) → R
      ⊢ Eq (Matrix.det fun i j => (Finset.range (HAdd.hAdd (↑j) 1)).sum fun k => HMu …
    -/
    apply det_eq_of_forall_col_eq_smul_add_pred fun _ => v 0
      /-
        case succ.calc_2.succ.A_zero
        R : Type u_1
        inst✝ : CommRing R
        n✝ : Nat
        ih : ∀ (v : Fin (HAdd.hAdd n✝ 1) → R), Eq (Matrix.of fun i j => HPow.hPow (v i …
        v : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) → R
        ⊢ ∀ (i : Fin (HAdd.hAdd n✝ 1)), Eq ((Finset.range (HAdd.hAdd (↑0) 1)).sum fun  …
      -/
    · intro j
      /-
        case succ.calc_2.succ.A_zero
        R : Type u_1
        inst✝ : CommRing R
        n✝ : Nat
        ih : ∀ (v : Fin (HAdd.hAdd n✝ 1) → R), Eq (Matrix.of fun i j => HPow.hPow (v i …
        v : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) → R
        j : Fin (HAdd.hAdd n✝ 1)
        ⊢ Eq ((Finset.range (HAdd.hAdd (↑0) 1)).sum fun k => HMul.hMul (HPow.hPow (v j …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case succ.calc_2.succ.A_succ
        R : Type u_1
        inst✝ : CommRing R
        n✝ : Nat
        ih : ∀ (v : Fin (HAdd.hAdd n✝ 1) → R), Eq (Matrix.of fun i j => HPow.hPow (v i …
        v : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) → R
        ⊢ ∀ (i : Fin (HAdd.hAdd n✝ 1)) (j : Fin n✝), Eq ((Finset.range (HAdd.hAdd (↑j. …
      -/
    · intro i j
      /-
        case succ.calc_2.succ.A_succ
        R : Type u_1
        inst✝ : CommRing R
        n✝ : Nat
        ih : ∀ (v : Fin (HAdd.hAdd n✝ 1) → R), Eq (Matrix.of fun i j => HPow.hPow (v i …
        v : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) → R
        i : Fin (HAdd.hAdd n✝ 1)
        j : Fin n✝
        ⊢ Eq ((Finset.range (HAdd.hAdd (↑j.succ) 1)).sum fun k => HMul.hMul (HPow.hPow …
      -/
      simp only [smul_eq_mul, Pi.add_apply, Fin.val_succ, Fin.coe_castSucc, Pi.smul_apply]
      /-
        case succ.calc_2.succ.A_succ
        R : Type u_1
        inst✝ : CommRing R
        n✝ : Nat
        ih : ∀ (v : Fin (HAdd.hAdd n✝ 1) → R), Eq (Matrix.of fun i j => HPow.hPow (v i …
        v : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) → R
        i : Fin (HAdd.hAdd n✝ 1)
        j : Fin n✝
        ⊢ Eq ((Finset.range (HAdd.hAdd (HAdd.hAdd (↑j) 1) 1)).sum fun x => HMul.hMul ( …
      -/
      rw [Finset.sum_range_succ, add_comm, tsub_self, pow_zero, mul_one, Finset.mul_sum]
      /-
        case succ.calc_2.succ.A_succ
        R : Type u_1
        inst✝ : CommRing R
        n✝ : Nat
        ih : ∀ (v : Fin (HAdd.hAdd n✝ 1) → R), Eq (Matrix.of fun i j => HPow.hPow (v i …
        v : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) → R
        i : Fin (HAdd.hAdd n✝ 1)
        j : Fin n✝
        ⊢ Eq (HAdd.hAdd (HPow.hPow (v i.succ) (HAdd.hAdd (↑j) 1)) ((Finset.range (HAdd …
      -/
      congr 1
      /-
        case succ.calc_2.succ.A_succ.e_a
        R : Type u_1
        inst✝ : CommRing R
        n✝ : Nat
        ih : ∀ (v : Fin (HAdd.hAdd n✝ 1) → R), Eq (Matrix.of fun i j => HPow.hPow (v i …
        v : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) → R
        i : Fin (HAdd.hAdd n✝ 1)
        j : Fin n✝
        ⊢ Eq ((Finset.range (HAdd.hAdd (↑j) 1)).sum fun x => HMul.hMul (HPow.hPow (v i …
      -/
      refine Finset.sum_congr rfl fun i' hi' => ?_
      /-
        case succ.calc_2.succ.A_succ.e_a
        R : Type u_1
        inst✝ : CommRing R
        n✝ : Nat
        ih : ∀ (v : Fin (HAdd.hAdd n✝ 1) → R), Eq (Matrix.of fun i j => HPow.hPow (v i …
        v : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) → R
        i : Fin (HAdd.hAdd n✝ 1)
        j : Fin n✝
        i' : Nat
        hi' : Membership.mem (Finset.range (HAdd.hAdd (↑j) 1)) i'
        ⊢ Eq (HMul.hMul (HPow.hPow (v i.succ) i') (HPow.hPow (v 0) (HSub.hSub (HAdd.hA …
      -/
      rw [mul_left_comm (v 0), Nat.succ_sub, pow_succ']
      /-
        case succ.calc_2.succ.A_succ.e_a
        R : Type u_1
        inst✝ : CommRing R
        n✝ : Nat
        ih : ∀ (v : Fin (HAdd.hAdd n✝ 1) → R), Eq (Matrix.of fun i j => HPow.hPow (v i …
        v : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) → R
        i : Fin (HAdd.hAdd n✝ 1)
        j : Fin n✝
        i' : Nat
        hi' : Membership.mem (Finset.range (HAdd.hAdd (↑j) 1)) i'
        ⊢ LE.le i' ↑j
      -/
      exact Nat.lt_succ_iff.mp (Finset.mem_range.mp hi')
      /-
        🎉 no goals
      -/


theorem det_vandermonde_eq_zero_iff [IsDomain R] {n : ℕ} {v : Fin n → R} :
    det (vandermonde v) = 0 ↔ ∃ i j : Fin n, v i = v j ∧ i ≠ j := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    v : Fin n → R
    ⊢ Iff (Eq (Matrix.vandermonde v).det 0) (Exists fun i => Exists fun j => And ( …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      v : Fin n → R
      ⊢ Eq (Matrix.vandermonde v).det 0 → Exists fun i => Exists fun j => And (Eq (v …
    -/
  · simp only [det_vandermonde v, Finset.prod_eq_zero_iff, sub_eq_zero, forall_exists_index]
    /-
      case mp
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      v : Fin n → R
      ⊢ ∀ (x : Fin n), And (Membership.mem Finset.univ x) (Exists fun a => And (Memb …
    -/
    rintro i ⟨_, j, h₁, h₂⟩
    /-
      case mp.intro.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      v : Fin n → R
      i : Fin n
      left✝ : Membership.mem Finset.univ i
      j : Fin n
      h₁ : Membership.mem (Finset.Ioi i) j
      h₂ : Eq (v j) (v i)
      ⊢ Exists fun i => Exists fun j => And (Eq (v i) (v j)) (Ne i j)
    -/
    exact ⟨j, i, h₂, (mem_Ioi.mp h₁).ne'⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      v : Fin n → R
      ⊢ (Exists fun i => Exists fun j => And (Eq (v i) (v j)) (Ne i j)) → Eq (Matrix …
    -/
  · simp only [Ne, forall_exists_index, and_imp]
    /-
      case mpr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      v : Fin n → R
      ⊢ ∀ (x x_1 : Fin n), Eq (v x) (v x_1) → Not (Eq x x_1) → Eq (Matrix.vandermond …
    -/
    refine fun i j h₁ h₂ => Matrix.det_zero_of_row_eq h₂ (funext fun k => ?_)
    /-
      case mpr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      v : Fin n → R
      i j : Fin n
      h₁ : Eq (v i) (v j)
      h₂ : Not (Eq i j)
      k : Fin n
      ⊢ Eq (Matrix.vandermonde v i k) (Matrix.vandermonde v j k)
    -/
    rw [vandermonde_apply, vandermonde_apply, h₁]
    /-
      🎉 no goals
    -/


theorem det_vandermonde_ne_zero_iff [IsDomain R] {n : ℕ} {v : Fin n → R} :
    det (vandermonde v) ≠ 0 ↔ Function.Injective v := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    v : Fin n → R
    ⊢ Iff (Ne (Matrix.vandermonde v).det 0) (Function.Injective v)
  -/
  unfold Function.Injective
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    v : Fin n → R
    ⊢ Iff (Ne (Matrix.vandermonde v).det 0) (∀ ⦃a₁ a₂ : Fin n⦄, Eq (v a₁) (v a₂) → …
  -/
  simp only [det_vandermonde_eq_zero_iff, Ne, not_exists, not_and, Classical.not_not]
  /-
    🎉 no goals
  -/


@[simp]
theorem det_vandermonde_add {n : ℕ} (v : Fin n → R) (a : R) :
    (Matrix.vandermonde fun i ↦ v i + a).det = (Matrix.vandermonde v).det := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v : Fin n → R
    a : R
    ⊢ Eq (Matrix.vandermonde fun i => HAdd.hAdd (v i) a).det (Matrix.vandermonde v …
  -/
  simp [Matrix.det_vandermonde]
  /-
    🎉 no goals
  -/


@[simp]
theorem det_vandermonde_sub {n : ℕ} (v : Fin n → R) (a : R) :
    (Matrix.vandermonde fun i ↦ v i - a).det = (Matrix.vandermonde v).det := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v : Fin n → R
    a : R
    ⊢ Eq (Matrix.vandermonde fun i => HSub.hSub (v i) a).det (Matrix.vandermonde v …
  -/
  rw [← det_vandermonde_add v (- a)]
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v : Fin n → R
    a : R
    ⊢ Eq (Matrix.vandermonde fun i => HSub.hSub (v i) a).det (Matrix.vandermonde f …
  -/
  simp only [← sub_eq_add_neg]
  /-
    🎉 no goals
  -/


theorem eq_zero_of_forall_index_sum_pow_mul_eq_zero {R : Type*} [CommRing R] [IsDomain R] {n : ℕ}
    {f v : Fin n → R} (hf : Function.Injective f)
    (hfv : ∀ j, (∑ i : Fin n, f j ^ (i : ℕ) * v i) = 0) : v = 0 :=
  eq_zero_of_mulVec_eq_zero (det_vandermonde_ne_zero_iff.mpr hf) (funext hfv)


theorem eq_zero_of_forall_index_sum_mul_pow_eq_zero {R : Type*} [CommRing R] [IsDomain R] {n : ℕ}
    {f v : Fin n → R} (hf : Function.Injective f) (hfv : ∀ j, (∑ i, v i * f j ^ (i : ℕ)) = 0) :
    v = 0 := by
  /-
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    f v : Fin n → R
    hf : Function.Injective f
    hfv : ∀ (j : Fin n), Eq (Finset.univ.sum fun i => HMul.hMul (v i) (HPow.hPow ( …
    ⊢ Eq v 0
  -/
  apply eq_zero_of_forall_index_sum_pow_mul_eq_zero hf
  /-
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    f v : Fin n → R
    hf : Function.Injective f
    hfv : ∀ (j : Fin n), Eq (Finset.univ.sum fun i => HMul.hMul (v i) (HPow.hPow ( …
    ⊢ ∀ (j : Fin n), Eq (Finset.univ.sum fun i => HMul.hMul (HPow.hPow (f j) ↑i) ( …
  -/
  simp_rw [mul_comm]
  /-
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    f v : Fin n → R
    hf : Function.Injective f
    hfv : ∀ (j : Fin n), Eq (Finset.univ.sum fun i => HMul.hMul (v i) (HPow.hPow ( …
    ⊢ ∀ (j : Fin n), Eq (Finset.univ.sum fun x => HMul.hMul (v x) (HPow.hPow (f j) …
  -/
  exact hfv
  /-
    🎉 no goals
  -/


theorem eq_zero_of_forall_pow_sum_mul_pow_eq_zero {R : Type*} [CommRing R] [IsDomain R] {n : ℕ}
    {f v : Fin n → R} (hf : Function.Injective f)
    (hfv : ∀ i : Fin n, (∑ j : Fin n, v j * f j ^ (i : ℕ)) = 0) : v = 0 :=
  eq_zero_of_vecMul_eq_zero (det_vandermonde_ne_zero_iff.mpr hf) (funext hfv)


theorem eval_matrixOfPolynomials_eq_vandermonde_mul_matrixOfPolynomials {n : ℕ}
    (v : Fin n → R) (p : Fin n → R[X]) (h_deg : ∀ i, (p i).natDegree ≤ i) :
    Matrix.of (fun i j => ((p j).eval (v i))) =
    (Matrix.vandermonde v) * (Matrix.of (fun (i j : Fin n) => (p j).coeff i)) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v : Fin n → R
    p : Fin n → Polynomial R
    h_deg : ∀ (i : Fin n), LE.le (p i).natDegree ↑i
    ⊢ Eq (Matrix.of fun i j => Polynomial.eval (v i) (p j)) (HMul.hMul (Matrix.van …
  -/
  ext i j
  /-
    case a
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v : Fin n → R
    p : Fin n → Polynomial R
    h_deg : ∀ (i : Fin n), LE.le (p i).natDegree ↑i
    i j : Fin n
    ⊢ Eq (Matrix.of (fun i j => Polynomial.eval (v i) (p j)) i j) (HMul.hMul (Matr …
  -/
  rw [Matrix.mul_apply, eval, Matrix.of_apply, eval₂]
  /-
    case a
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v : Fin n → R
    p : Fin n → Polynomial R
    h_deg : ∀ (i : Fin n), LE.le (p i).natDegree ↑i
    i j : Fin n
    ⊢ Eq ((p j).sum fun e a => HMul.hMul ((RingHom.id R) a) (HPow.hPow (v i) e)) ( …
  -/
  simp only [eq_intCast, Int.cast_id, Matrix.vandermonde]
  /-
    case a
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v : Fin n → R
    p : Fin n → Polynomial R
    h_deg : ∀ (i : Fin n), LE.le (p i).natDegree ↑i
    i j : Fin n
    ⊢ Eq ((p j).sum fun e a => HMul.hMul ((RingHom.id R) a) (HPow.hPow (v i) e)) ( …
  -/
  have : (p j).support ⊆ range n := supp_subset_range <| Nat.lt_of_le_of_lt (h_deg j) <| Fin.prop j
  /-
    case a
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v : Fin n → R
    p : Fin n → Polynomial R
    h_deg : ∀ (i : Fin n), LE.le (p i).natDegree ↑i
    i j : Fin n
    this : HasSubset.Subset (p j).support (Finset.range n)
    ⊢ Eq ((p j).sum fun e a => HMul.hMul ((RingHom.id R) a) (HPow.hPow (v i) e)) ( …
  -/
  rw [sum_eq_of_subset _ (fun j => zero_mul ((v i) ^ j)) this, ← Fin.sum_univ_eq_sum_range]
  /-
    case a
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v : Fin n → R
    p : Fin n → Polynomial R
    h_deg : ∀ (i : Fin n), LE.le (p i).natDegree ↑i
    i j : Fin n
    this : HasSubset.Subset (p j).support (Finset.range n)
    ⊢ Eq (Finset.univ.sum fun i_1 => HMul.hMul ((RingHom.id R) ((p j).coeff ↑i_1)) …
  -/
  congr
  /-
    case a.e_f
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v : Fin n → R
    p : Fin n → Polynomial R
    h_deg : ∀ (i : Fin n), LE.le (p i).natDegree ↑i
    i j : Fin n
    this : HasSubset.Subset (p j).support (Finset.range n)
    ⊢ Eq (fun i_1 => HMul.hMul ((RingHom.id R) ((p j).coeff ↑i_1)) (HPow.hPow (v i …
  -/
  ext k
  /-
    case a.e_f.h
    R : Type u_1
    inst✝ : CommRing R
    n : Nat
    v : Fin n → R
    p : Fin n → Polynomial R
    h_deg : ∀ (i : Fin n), LE.le (p i).natDegree ↑i
    i j : Fin n
    this : HasSubset.Subset (p j).support (Finset.range n)
    k : Fin n
    ⊢ Eq (HMul.hMul ((RingHom.id R) ((p j).coeff ↑k)) (HPow.hPow (v i) ↑k)) (HMul. …
  -/
  rw [mul_comm, Matrix.of_apply, RingHom.id_apply, of_apply]
  /-
    🎉 no goals
  -/


theorem det_eval_matrixOfPolynomials_eq_det_vandermonde {n : ℕ} (v : Fin n → R) (p : Fin n → R[X])
    (h_deg : ∀ i, (p i).natDegree = i) (h_monic : ∀ i, Monic <| p i) :
    (Matrix.vandermonde v).det = (Matrix.of (fun i j => ((p j).eval (v i)))).det := by
  rw [Matrix.eval_matrixOfPolynomials_eq_vandermonde_mul_matrixOfPolynomials v p (fun i ↦
      Nat.le_of_eq (h_deg i)), Matrix.det_mul,
      Matrix.det_matrixOfPolynomials p h_deg h_monic, mul_one]


