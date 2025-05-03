local notation3 "m" => Fintype.card α

local notation3 "n" => Fintype.card β

local notation3 "e" => m / ((n : ℝ) - m) -- exponent

local notation3 "B" => Nat.floor (((n : ℝ) * max 1 ‖A‖) ^ e)
-- B' is the vector with all components = B

local notation3 "B'" => fun _ : β => (B : ℤ)
-- T is the box [0 B]^n

local notation3 "T" =>  Finset.Icc 0 B'

local notation3 "P" => fun i : α => ∑ j : β, B * posPart (A i j)

local notation3 "N" => fun i : α => ∑ j : β, B * (- negPart (A i j))
-- S is the box where the image of T goes

local notation3 "S" => Finset.Icc N P


private lemma image_T_subset_S [DecidableEq α] [DecidableEq β] (v) (hv : v ∈ T) : A *ᵥ v ∈ S := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Fintype α
    inst✝² : Fintype β
    A : Matrix α β Int
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    v : β → Int
    hv : Membership.mem (Finset.Icc 0 fun x => ↑(Nat.floor (HPow.hPow (HMul.hMul ( …
    ⊢ Membership.mem (Finset.Icc (fun i => Finset.univ.sum fun j => HMul.hMul (↑(N …
  -/
  rw [mem_Icc] at hv ⊢
  have mulVec_def : A.mulVec v =
      fun i ↦ Finset.sum univ fun j : β ↦ A i j * v j := rfl
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Fintype α
    inst✝² : Fintype β
    A : Matrix α β Int
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    v : β → Int
    hv : And (LE.le 0 v) (LE.le v fun x => ↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fi …
    mulVec_def : Eq (A.mulVec v) fun i => Finset.univ.sum fun j => HMul.hMul (A i  …
    ⊢ And (LE.le (fun i => Finset.univ.sum fun j => HMul.hMul (↑(Nat.floor (HPow.h …
  -/
  rw [mulVec_def]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Fintype α
    inst✝² : Fintype β
    A : Matrix α β Int
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    v : β → Int
    hv : And (LE.le 0 v) (LE.le v fun x => ↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fi …
    mulVec_def : Eq (A.mulVec v) fun i => Finset.univ.sum fun j => HMul.hMul (A i  …
    ⊢ And (LE.le (fun i => Finset.univ.sum fun j => HMul.hMul (↑(Nat.floor (HPow.h …
  -/
  refine ⟨fun i ↦ ?_, fun i ↦ ?_⟩
  all_goals
    simp only [mul_neg]
    gcongr ∑ _ : α, ?_ with j _ -- Get rid of sums
    rw [← mul_comm (v j)] -- Move A i j to the right of the products
    rcases le_total 0 (A i j) with hsign | hsign-- We have to distinguish cases: we have now 4 goals
    /-
      case refine_1.h.inl
      α : Type u_1
      β : Type u_2
      inst✝³ : Fintype α
      inst✝² : Fintype β
      A : Matrix α β Int
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      v : β → Int
      hv : And (LE.le 0 v) (LE.le v fun x => ↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fi …
      mulVec_def : Eq (A.mulVec v) fun i => Finset.univ.sum fun j => HMul.hMul (A i  …
      i : α
      j : β
      a✝ : Membership.mem Finset.univ j
      hsign : LE.le 0 (A i j)
      ⊢ LE.le (Neg.neg (HMul.hMul (↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fintype.card …
    -/
  · rw [negPart_eq_zero.2 hsign]
    /-
      case refine_1.h.inl
      α : Type u_1
      β : Type u_2
      inst✝³ : Fintype α
      inst✝² : Fintype β
      A : Matrix α β Int
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      v : β → Int
      hv : And (LE.le 0 v) (LE.le v fun x => ↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fi …
      mulVec_def : Eq (A.mulVec v) fun i => Finset.univ.sum fun j => HMul.hMul (A i  …
      i : α
      j : β
      a✝ : Membership.mem Finset.univ j
      hsign : LE.le 0 (A i j)
      ⊢ LE.le (Neg.neg (HMul.hMul (↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fintype.card …
    -/
    exact mul_nonneg (hv.1 j) hsign
    /-
      🎉 no goals
    -/
    /-
      case refine_1.h.inr
      α : Type u_1
      β : Type u_2
      inst✝³ : Fintype α
      inst✝² : Fintype β
      A : Matrix α β Int
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      v : β → Int
      hv : And (LE.le 0 v) (LE.le v fun x => ↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fi …
      mulVec_def : Eq (A.mulVec v) fun i => Finset.univ.sum fun j => HMul.hMul (A i  …
      i : α
      j : β
      a✝ : Membership.mem Finset.univ j
      hsign : LE.le (A i j) 0
      ⊢ LE.le (Neg.neg (HMul.hMul (↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fintype.card …
    -/
  · rw [negPart_eq_neg.2 hsign]
    /-
      case refine_1.h.inr
      α : Type u_1
      β : Type u_2
      inst✝³ : Fintype α
      inst✝² : Fintype β
      A : Matrix α β Int
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      v : β → Int
      hv : And (LE.le 0 v) (LE.le v fun x => ↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fi …
      mulVec_def : Eq (A.mulVec v) fun i => Finset.univ.sum fun j => HMul.hMul (A i  …
      i : α
      j : β
      a✝ : Membership.mem Finset.univ j
      hsign : LE.le (A i j) 0
      ⊢ LE.le (Neg.neg (HMul.hMul (↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fintype.card …
    -/
    simp only [mul_neg, neg_neg]
    /-
      case refine_1.h.inr
      α : Type u_1
      β : Type u_2
      inst✝³ : Fintype α
      inst✝² : Fintype β
      A : Matrix α β Int
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      v : β → Int
      hv : And (LE.le 0 v) (LE.le v fun x => ↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fi …
      mulVec_def : Eq (A.mulVec v) fun i => Finset.univ.sum fun j => HMul.hMul (A i  …
      i : α
      j : β
      a✝ : Membership.mem Finset.univ j
      hsign : LE.le (A i j) 0
      ⊢ LE.le (HMul.hMul (↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fintype.card β)) (Max …
    -/
    exact mul_le_mul_of_nonpos_right (hv.2 j) hsign
    /-
      🎉 no goals
    -/
    /-
      case refine_2.h.inl
      α : Type u_1
      β : Type u_2
      inst✝³ : Fintype α
      inst✝² : Fintype β
      A : Matrix α β Int
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      v : β → Int
      hv : And (LE.le 0 v) (LE.le v fun x => ↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fi …
      mulVec_def : Eq (A.mulVec v) fun i => Finset.univ.sum fun j => HMul.hMul (A i  …
      i : α
      j : β
      a✝ : Membership.mem Finset.univ j
      hsign : LE.le 0 (A i j)
      ⊢ LE.le (HMul.hMul (v j) (A i j)) (HMul.hMul (↑(Nat.floor (HPow.hPow (HMul.hMu …
    -/
  · rw [posPart_eq_self.2 hsign]
    /-
      case refine_2.h.inl
      α : Type u_1
      β : Type u_2
      inst✝³ : Fintype α
      inst✝² : Fintype β
      A : Matrix α β Int
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      v : β → Int
      hv : And (LE.le 0 v) (LE.le v fun x => ↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fi …
      mulVec_def : Eq (A.mulVec v) fun i => Finset.univ.sum fun j => HMul.hMul (A i  …
      i : α
      j : β
      a✝ : Membership.mem Finset.univ j
      hsign : LE.le 0 (A i j)
      ⊢ LE.le (HMul.hMul (v j) (A i j)) (HMul.hMul (↑(Nat.floor (HPow.hPow (HMul.hMu …
    -/
    exact mul_le_mul_of_nonneg_right (hv.2 j) hsign
    /-
      🎉 no goals
    -/
    /-
      case refine_2.h.inr
      α : Type u_1
      β : Type u_2
      inst✝³ : Fintype α
      inst✝² : Fintype β
      A : Matrix α β Int
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      v : β → Int
      hv : And (LE.le 0 v) (LE.le v fun x => ↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fi …
      mulVec_def : Eq (A.mulVec v) fun i => Finset.univ.sum fun j => HMul.hMul (A i  …
      i : α
      j : β
      a✝ : Membership.mem Finset.univ j
      hsign : LE.le (A i j) 0
      ⊢ LE.le (HMul.hMul (v j) (A i j)) (HMul.hMul (↑(Nat.floor (HPow.hPow (HMul.hMu …
    -/
  · rw [posPart_eq_zero.2 hsign]
    /-
      case refine_2.h.inr
      α : Type u_1
      β : Type u_2
      inst✝³ : Fintype α
      inst✝² : Fintype β
      A : Matrix α β Int
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      v : β → Int
      hv : And (LE.le 0 v) (LE.le v fun x => ↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fi …
      mulVec_def : Eq (A.mulVec v) fun i => Finset.univ.sum fun j => HMul.hMul (A i  …
      i : α
      j : β
      a✝ : Membership.mem Finset.univ j
      hsign : LE.le (A i j) 0
      ⊢ LE.le (HMul.hMul (v j) (A i j)) (HMul.hMul (↑(Nat.floor (HPow.hPow (HMul.hMu …
    -/
    exact mul_nonpos_of_nonneg_of_nonpos (hv.1 j) hsign
    /-
      🎉 no goals
    -/

-- # Preparation for Step 2


private lemma card_T_eq [DecidableEq β] : #T = (B + 1) ^ n := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Fintype α
    inst✝¹ : Fintype β
    A : Matrix α β Int
    inst✝ : DecidableEq β
    ⊢ Eq (Finset.Icc 0 fun x => ↑(Nat.floor (HPow.hPow (HMul.hMul (↑(Fintype.card  …
  -/
  rw [Pi.card_Icc 0 B']
  simp only [Pi.zero_apply, card_Icc, sub_zero, toNat_ofNat_add_one, prod_const, card_univ,
    add_pos_iff, zero_lt_one, or_true]

-- This lemma is necessary to be able to apply the formula #(Icc a b) = b + 1 - a

private lemma N_le_P_add_one (i : α) : N i ≤ P i + 1 := by
  calc N i
  _ ≤ 0 := by
    apply Finset.sum_nonpos
    intro j _
    simp only [mul_neg, Left.neg_nonpos_iff]
    exact mul_nonneg (Nat.cast_nonneg B) (negPart_nonneg (A i j))
  _ ≤ P i + 1 := by
    apply le_trans (Finset.sum_nonneg _) (Int.le_add_one (le_refl P i))
    intro j _
    exact mul_nonneg (Nat.cast_nonneg B) (posPart_nonneg (A i j))


private lemma card_S_eq [DecidableEq α] : #(Finset.Icc N P) = ∏ i : α, (P i - N i + 1) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Fintype α
    inst✝¹ : Fintype β
    A : Matrix α β Int
    inst✝ : DecidableEq α
    ⊢ Eq (↑(Finset.Icc (fun i => Finset.univ.sum fun j => HMul.hMul (↑(Nat.floor ( …
  -/
  rw [Pi.card_Icc N P, Nat.cast_prod]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Fintype α
    inst✝¹ : Fintype β
    A : Matrix α β Int
    inst✝ : DecidableEq α
    ⊢ Eq (Finset.univ.prod fun i => ↑(Finset.Icc (Finset.univ.sum fun j => HMul.hM …
  -/
  congr
  /-
    case e_f
    α : Type u_1
    β : Type u_2
    inst✝² : Fintype α
    inst✝¹ : Fintype β
    A : Matrix α β Int
    inst✝ : DecidableEq α
    ⊢ Eq (fun i => ↑(Finset.Icc (Finset.univ.sum fun j => HMul.hMul (↑(Nat.floor ( …
  -/
  ext i
  /-
    case e_f.h
    α : Type u_1
    β : Type u_2
    inst✝² : Fintype α
    inst✝¹ : Fintype β
    A : Matrix α β Int
    inst✝ : DecidableEq α
    i : α
    ⊢ Eq (↑(Finset.Icc (Finset.univ.sum fun j => HMul.hMul (↑(Nat.floor (HPow.hPow …
  -/
  rw [Int.card_Icc_of_le (N i) (P i) (N_le_P_add_one A i)]
  /-
    case e_f.h
    α : Type u_1
    β : Type u_2
    inst✝² : Fintype α
    inst✝¹ : Fintype β
    A : Matrix α β Int
    inst✝ : DecidableEq α
    i : α
    ⊢ Eq (HSub.hSub (HAdd.hAdd ((fun i => Finset.univ.sum fun j => HMul.hMul (↑(Na …
  -/
  exact add_sub_right_comm (P i) 1 (N i)
  /-
    🎉 no goals
  -/


/-- The sup norm of a non-zero integer matrix is at least one  -/
lemma one_le_norm_A_of_ne_zero (hA : A ≠ 0) : 1 ≤ ‖A‖ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    A : Matrix α β Int
    hA : Ne A 0
    ⊢ LE.le 1 (Norm.norm A)
  -/
  by_contra! h
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    A : Matrix α β Int
    hA : Ne A 0
    h : LT.lt (Norm.norm A) 1
    ⊢ False
  -/
  apply hA
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    A : Matrix α β Int
    hA : Ne A 0
    h : LT.lt (Norm.norm A) 1
    ⊢ Eq A 0
  -/
  ext i j
  /-
    case a
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    A : Matrix α β Int
    hA : Ne A 0
    h : LT.lt (Norm.norm A) 1
    i : α
    j : β
    ⊢ Eq (A i j) (0 i j)
  -/
  simp only [zero_apply]
  /-
    case a
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    A : Matrix α β Int
    hA : Ne A 0
    h : LT.lt (Norm.norm A) 1
    i : α
    j : β
    ⊢ Eq (A i j) 0
  -/
  rw [norm_lt_iff Real.zero_lt_one] at h
  /-
    case a
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    A : Matrix α β Int
    hA : Ne A 0
    h : ∀ (i : α) (j : β), LT.lt (Norm.norm (A i j)) 1
    i : α
    j : β
    ⊢ Eq (A i j) 0
  -/
  specialize h i j
  /-
    case a
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    A : Matrix α β Int
    hA : Ne A 0
    i : α
    j : β
    h : LT.lt (Norm.norm (A i j)) 1
    ⊢ Eq (A i j) 0
  -/
  rw [Int.norm_eq_abs] at h
  /-
    case a
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    A : Matrix α β Int
    hA : Ne A 0
    i : α
    j : β
    h : LT.lt (abs ↑(A i j)) 1
    ⊢ Eq (A i j) 0
  -/
  norm_cast at h
  /-
    case a
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    A : Matrix α β Int
    hA : Ne A 0
    i : α
    j : β
    h : LT.lt (abs (A i j)) 1
    ⊢ Eq (A i j) 0
  -/
  exact Int.abs_lt_one_iff.1 h
  /-
    🎉 no goals
  -/

-- # Step 2: #S < #T


private lemma card_S_lt_card_T [DecidableEq α] [DecidableEq β]
    (hn : Fintype.card α < Fintype.card β) (hm : 0 < Fintype.card α) :
    #S < #T := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Fintype α
    inst✝² : Fintype β
    A : Matrix α β Int
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    hn : LT.lt (Fintype.card α) (Fintype.card β)
    hm : LT.lt 0 (Fintype.card α)
    ⊢ LT.lt (Finset.Icc (fun i => Finset.univ.sum fun j => HMul.hMul (↑(Nat.floor  …
  -/
  zify -- This is necessary to use card_S_eq
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Fintype α
    inst✝² : Fintype β
    A : Matrix α β Int
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    hn : LT.lt (Fintype.card α) (Fintype.card β)
    hm : LT.lt 0 (Fintype.card α)
    ⊢ LT.lt ↑(Finset.Icc (fun i => Finset.univ.sum fun j => HMul.hMul (↑(Nat.floor …
  -/
  rw [card_T_eq A, card_S_eq]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Fintype α
    inst✝² : Fintype β
    A : Matrix α β Int
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    hn : LT.lt (Fintype.card α) (Fintype.card β)
    hm : LT.lt 0 (Fintype.card α)
    ⊢ LT.lt (Finset.univ.prod fun i => HAdd.hAdd (HSub.hSub ((fun i => Finset.univ …
  -/
  rify -- This is necessary because ‖A‖ is a real number
  calc
  ∏ x : α, (∑ x_1 : β, ↑B * ↑(A x x_1)⁺ - ∑ x_1 : β, ↑B * -↑(A x x_1)⁻ + 1)
    ≤ ∏ x : α, (n * max 1 ‖A‖ * B + 1) := by
      refine Finset.prod_le_prod (fun i _ ↦ ?_) (fun i _ ↦ ?_)
      · have h := N_le_P_add_one A i
        rify at h
        linarith only [h]
      · simp only [mul_neg, sum_neg_distrib, sub_neg_eq_add, add_le_add_iff_right]
        have h1 : n * max 1 ‖A‖ * B = ∑ _ : β, max 1 ‖A‖ * B := by
          simp only [sum_const, card_univ, nsmul_eq_mul]
          ring
        simp_rw [h1, ← Finset.sum_add_distrib, ← mul_add, mul_comm (max 1 ‖A‖), ← Int.cast_add]
        gcongr with j _
        rw [posPart_add_negPart (A i j), Int.cast_abs]
        exact le_trans (norm_entry_le_entrywise_sup_norm A) (le_max_right ..)
  _  = (n * max 1 ‖A‖ * B + 1) ^ m := by simp only [prod_const, card_univ]
  _  ≤ (n * max 1 ‖A‖) ^ m * (B + 1) ^ m := by
        rw [← mul_pow, mul_add, mul_one]
        gcongr
        have H : 1 ≤ (n : ℝ) := mod_cast (hm.trans hn)
        exact one_le_mul_of_one_le_of_one_le H <| le_max_left ..
  _ = ((n * max 1 ‖A‖) ^ (m / ((n : ℝ) - m))) ^ ((n : ℝ) - m)  * (B + 1) ^ m := by
        congr 1
        rw [← rpow_mul (mul_nonneg (Nat.cast_nonneg' n) (le_trans zero_le_one (le_max_left ..))),
          ← Real.rpow_natCast, div_mul_cancel₀]
        exact sub_ne_zero_of_ne (mod_cast hn.ne')
  _ < (B + 1) ^ ((n : ℝ) - m) * (B + 1) ^ m := by
        gcongr
        · exact sub_pos.mpr (mod_cast hn)
        · exact Nat.lt_floor_add_one ((n * max 1 ‖A‖) ^ e)
  _ = (B + 1) ^ n := by
        rw [← rpow_natCast, ← rpow_add (Nat.cast_add_one_pos B), ← rpow_natCast, sub_add_cancel]


theorem exists_ne_zero_int_vec_norm_le
    (hn : Fintype.card α < Fintype.card β) (hm : 0 < Fintype.card α) : ∃ t : β → ℤ, t ≠ 0 ∧
    A *ᵥ t = 0 ∧ ‖t‖ ≤ (n * max 1 ‖A‖) ^ ((m : ℝ) / (n - m)) := by
  classical
  -- Pigeonhole
  rcases Finset.exists_ne_map_eq_of_card_lt_of_maps_to
    (card_S_lt_card_T A hn hm) (image_T_subset_S A)
    with ⟨x, hxT, y, hyT, hneq, hfeq⟩
  -- Proofs that x - y ≠ 0 and x - y is a solution
  refine ⟨x - y, sub_ne_zero.mpr hneq, by simp only [mulVec_sub, sub_eq_zero, hfeq], ?_⟩
  -- Inequality
  have n_mul_norm_A_pow_e_nonneg : 0 ≤ (n * max 1 ‖A‖) ^ e := by positivity
  rw [← norm_col (ι := Unit), norm_le_iff n_mul_norm_A_pow_e_nonneg]
  intro i j
  simp only [col_apply, Pi.sub_apply]
  rw [Int.norm_eq_abs, ← Int.cast_abs]
  refine le_trans ?_ (Nat.floor_le n_mul_norm_A_pow_e_nonneg)
  norm_cast
  rw [abs_le]
  rw [Finset.mem_Icc] at hxT hyT
  constructor
  · simp only [neg_le_sub_iff_le_add]
    apply le_trans (hyT.2 i)
    norm_cast
    simp only [le_add_iff_nonneg_left]
    exact hxT.1 i
  · simp only [tsub_le_iff_right]
    apply le_trans (hxT.2 i)
    norm_cast
    simp only [le_add_iff_nonneg_right]
    exact hyT.1 i



theorem exists_ne_zero_int_vec_norm_le'
    (hn : Fintype.card α < Fintype.card β) (hm : 0 < Fintype.card α) (hA : A ≠ 0) :
    ∃ t : β → ℤ, t ≠ 0 ∧
    A *ᵥ t = 0 ∧ ‖t‖ ≤ (n * ‖A‖) ^ ((m : ℝ) / (n - m)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    A : Matrix α β Int
    hn : LT.lt (Fintype.card α) (Fintype.card β)
    hm : LT.lt 0 (Fintype.card α)
    hA : Ne A 0
    ⊢ Exists fun t => And (Ne t 0) (And (Eq (A.mulVec t) 0) (LE.le (Norm.norm t) ( …
  -/
  have := exists_ne_zero_int_vec_norm_le A hn hm
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    A : Matrix α β Int
    hn : LT.lt (Fintype.card α) (Fintype.card β)
    hm : LT.lt 0 (Fintype.card α)
    hA : Ne A 0
    this : Exists fun t => And (Ne t 0) (And (Eq (A.mulVec t) 0) (LE.le (Norm.norm …
    ⊢ Exists fun t => And (Ne t 0) (And (Eq (A.mulVec t) 0) (LE.le (Norm.norm t) ( …
  -/
  rwa [max_eq_right] at this
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    A : Matrix α β Int
    hn : LT.lt (Fintype.card α) (Fintype.card β)
    hm : LT.lt 0 (Fintype.card α)
    hA : Ne A 0
    this : Exists fun t => And (Ne t 0) (And (Eq (A.mulVec t) 0) (LE.le (Norm.norm …
    ⊢ LE.le 1 (Norm.norm A)
  -/
  exact Int.Matrix.one_le_norm_A_of_ne_zero _ hA
  /-
    🎉 no goals
  -/


