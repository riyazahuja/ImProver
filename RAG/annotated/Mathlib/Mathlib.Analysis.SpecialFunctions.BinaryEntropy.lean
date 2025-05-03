/-- The [binary entropy function](https://en.wikipedia.org/wiki/Binary_entropy_function)
`binEntropy p := - p * log p - (1-p) * log (1 - p)`
is the Shannon entropy of a Bernoulli random variable with success probability `p`. -/
@[pp_nodot] noncomputable def binEntropy (p : ℝ) : ℝ := p * log p⁻¹ + (1 - p) * log (1 - p)⁻¹


                                                       /-
                                                         ⊢ Eq (Real.binEntropy 0) 0
                                                       -/
@[simp] lemma binEntropy_zero : binEntropy 0 = 0 := by simp [binEntropy]
                                                       /-
                                                         🎉 no goals
                                                       -/

                                                      /-
                                                        ⊢ Eq (Real.binEntropy 1) 0
                                                      -/
@[simp] lemma binEntropy_one : binEntropy 1 = 0 := by simp [binEntropy]
                                                      /-
                                                        🎉 no goals
                                                      -/

                                                                /-
                                                                  ⊢ Eq (Real.binEntropy (Inv.inv 2)) (Real.log 2)
                                                                -/
@[simp] lemma binEntropy_two_inv : binEntropy 2⁻¹ = log 2 := by norm_num [binEntropy]; simp; ring
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


lemma binEntropy_eq_negMulLog_add_negMulLog_one_sub (p : ℝ) :
                                                         /-
                                                           p : Real
                                                           ⊢ Eq (Real.binEntropy p) (HAdd.hAdd p.negMulLog (HSub.hSub 1 p).negMulLog)
                                                         -/
    binEntropy p = negMulLog p + negMulLog (1 - p) := by simp [binEntropy, negMulLog, ← neg_mul]
                                                         /-
                                                           🎉 no goals
                                                         -/


lemma binEntropy_eq_negMulLog_add_negMulLog_one_sub' :
    binEntropy = fun p ↦ negMulLog p + negMulLog (1 - p) :=
  funext binEntropy_eq_negMulLog_add_negMulLog_one_sub


/-- `binEntropy` is symmetric about 1/2. -/
@[simp] lemma binEntropy_one_sub (p : ℝ) : binEntropy (1 - p) = binEntropy p := by
  /-
    p : Real
    ⊢ Eq (Real.binEntropy (HSub.hSub 1 p)) (Real.binEntropy p)
  -/
  simp [binEntropy, add_comm]
  /-
    🎉 no goals
  -/


/-- `binEntropy` is symmetric about 1/2. -/
lemma binEntropy_two_inv_add (p : ℝ) : binEntropy (2⁻¹ + p) = binEntropy (2⁻¹ - p) := by
  /-
    p : Real
    ⊢ Eq (Real.binEntropy (HAdd.hAdd (Inv.inv 2) p)) (Real.binEntropy (HSub.hSub ( …
  -/
  rw [← binEntropy_one_sub]; ring_nf
                             /-
                               🎉 no goals
                             -/


lemma binEntropy_pos (hp₀ : 0 < p) (hp₁ : p < 1) : 0 < binEntropy p := by
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    ⊢ LT.lt 0 (Real.binEntropy p)
  -/
  unfold binEntropy
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    ⊢ LT.lt 0 (HAdd.hAdd (HMul.hMul p (Real.log (Inv.inv p))) (HMul.hMul (HSub.hSu …
  -/
  have : 0 < 1 - p := sub_pos.2 hp₁
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    this : LT.lt 0 (HSub.hSub 1 p)
    ⊢ LT.lt 0 (HAdd.hAdd (HMul.hMul p (Real.log (Inv.inv p))) (HMul.hMul (HSub.hSu …
  -/
  have : 0 < log p⁻¹ := log_pos <| (one_lt_inv₀ hp₀).2 hp₁
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    this✝ : LT.lt 0 (HSub.hSub 1 p)
    this : LT.lt 0 (Real.log (Inv.inv p))
    ⊢ LT.lt 0 (HAdd.hAdd (HMul.hMul p (Real.log (Inv.inv p))) (HMul.hMul (HSub.hSu …
  -/
  have : 0 < log (1 - p)⁻¹ := log_pos <| (one_lt_inv₀ ‹_›).2 (sub_lt_self _ hp₀)
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    this✝¹ : LT.lt 0 (HSub.hSub 1 p)
    this✝ : LT.lt 0 (Real.log (Inv.inv p))
    this : LT.lt 0 (Real.log (Inv.inv (HSub.hSub 1 p)))
    ⊢ LT.lt 0 (HAdd.hAdd (HMul.hMul p (Real.log (Inv.inv p))) (HMul.hMul (HSub.hSu …
  -/
  positivity
  /-
    🎉 no goals
  -/


lemma binEntropy_nonneg (hp₀ : 0 ≤ p) (hp₁ : p ≤ 1) : 0 ≤ binEntropy p := by
  /-
    p : Real
    hp₀ : LE.le 0 p
    hp₁ : LE.le p 1
    ⊢ LE.le 0 (Real.binEntropy p)
  -/
  obtain rfl | hp₀ := hp₀.eq_or_lt
    /-
      case inl
      hp₀ : LE.le 0 0
      hp₁ : LE.le 0 1
      ⊢ LE.le 0 (Real.binEntropy 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    p : Real
    hp₀✝ : LE.le 0 p
    hp₁ : LE.le p 1
    hp₀ : LT.lt 0 p
    ⊢ LE.le 0 (Real.binEntropy p)
  -/
  obtain rfl | hp₁ := hp₁.eq_or_lt
    /-
      case inr.inl
      hp₀✝ : LE.le 0 1
      hp₁ : LE.le 1 1
      hp₀ : LT.lt 0 1
      ⊢ LE.le 0 (Real.binEntropy 1)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    p : Real
    hp₀✝ : LE.le 0 p
    hp₁✝ : LE.le p 1
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    ⊢ LE.le 0 (Real.binEntropy p)
  -/
  exact (binEntropy_pos hp₀ hp₁).le
  /-
    🎉 no goals
  -/


/-- Outside the usual range of `binEntropy`, it is negative. This is due to `log p = log |p|`. -/
lemma binEntropy_neg_of_neg (hp : p < 0) : binEntropy p < 0 := by
  /-
    p : Real
    hp : LT.lt p 0
    ⊢ LT.lt (Real.binEntropy p) 0
  -/
  rw [binEntropy, log_inv, log_inv]
  /-
    p : Real
    hp : LT.lt p 0
    ⊢ LT.lt (HAdd.hAdd (HMul.hMul p (Neg.neg (Real.log p))) (HMul.hMul (HSub.hSub  …
  -/
  suffices -p * log p < (1 - p) * log (1 - p) by linarith
  /-
    p : Real
    hp : LT.lt p 0
    ⊢ LT.lt (HMul.hMul (Neg.neg p) (Real.log p)) (HMul.hMul (HSub.hSub 1 p) (Real. …
  -/
  by_cases hp' : p < -1
  · have : log p < log (1 - p) := by
      rw [← log_neg_eq_log]
      exact log_lt_log (Left.neg_pos_iff.mpr hp) (by linarith)
    /-
      case pos
      p : Real
      hp : LT.lt p 0
      hp' : LT.lt p (-1)
      this : LT.lt (Real.log p) (Real.log (HSub.hSub 1 p))
      ⊢ LT.lt (HMul.hMul (Neg.neg p) (Real.log p)) (HMul.hMul (HSub.hSub 1 p) (Real. …
    -/
    nlinarith [log_pos_of_lt_neg_one hp']
    /-
      🎉 no goals
    -/
  · have : -p * log p ≤ 0 := by
      wlog h : -1 < p
      · simp only [show p = -1 by linarith, log_neg_eq_log, log_one, le_refl, mul_zero]
      · nlinarith [log_neg_of_lt_zero hp h]
    /-
      case neg
      p : Real
      hp : LT.lt p 0
      hp' : Not (LT.lt p (-1))
      this : LE.le (HMul.hMul (Neg.neg p) (Real.log p)) 0
      ⊢ LT.lt (HMul.hMul (Neg.neg p) (Real.log p)) (HMul.hMul (HSub.hSub 1 p) (Real. …
    -/
    nlinarith [(log_pos (by linarith) : 0 < log (1 - p))]
    /-
      🎉 no goals
    -/


/-- Outside the usual range of `binEntropy`, it is negative. This is due to `log p = log |p|`. -/
lemma binEntropy_nonpos_of_nonpos (hp : p ≤ 0) : binEntropy p ≤ 0 := by
  /-
    p : Real
    hp : LE.le p 0
    ⊢ LE.le (Real.binEntropy p) 0
  -/
  obtain rfl | hp := hp.eq_or_lt
    /-
      case inl
      hp : LE.le 0 0
      ⊢ LE.le (Real.binEntropy 0) 0
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      p : Real
      hp✝ : LE.le p 0
      hp : LT.lt p 0
      ⊢ LE.le (Real.binEntropy p) 0
    -/
  · exact (binEntropy_neg_of_neg hp).le
    /-
      🎉 no goals
    -/


/-- Outside the usual range of `binEntropy`, it is negative. This is due to `log p = log |p|` -/
lemma binEntropy_neg_of_one_lt (hp : 1 < p) : binEntropy p < 0 := by
  /-
    p : Real
    hp : LT.lt 1 p
    ⊢ LT.lt (Real.binEntropy p) 0
  -/
  rw [← binEntropy_one_sub]; exact binEntropy_neg_of_neg (sub_neg.2 hp)
                             /-
                               🎉 no goals
                             -/


/-- Outside the usual range of `binEntropy`, it is negative. This is due to `log p = log |p|` -/
lemma binEntropy_nonpos_of_one_le (hp : 1 ≤ p) : binEntropy p ≤ 0 := by
  /-
    p : Real
    hp : LE.le 1 p
    ⊢ LE.le (Real.binEntropy p) 0
  -/
  rw [← binEntropy_one_sub]; exact binEntropy_nonpos_of_nonpos (sub_nonpos.2 hp)
                             /-
                               🎉 no goals
                             -/


lemma binEntropy_eq_zero : binEntropy p = 0 ↔ p = 0 ∨ p = 1 := by
  /-
    p : Real
    ⊢ Iff (Eq (Real.binEntropy p) 0) (Or (Eq p 0) (Eq p 1))
  -/
  refine ⟨fun h ↦ ?_, by rintro (rfl | rfl) <;> simp⟩
  /-
    p : Real
    h : Eq (Real.binEntropy p) 0
    ⊢ Or (Eq p 0) (Eq p 1)
  -/
  contrapose! h
  /-
    p : Real
    h : And (Ne p 0) (Ne p 1)
    ⊢ Ne (Real.binEntropy p) 0
  -/
  obtain hp₀ | hp₀ := h.1.lt_or_lt
    /-
      case inl
      p : Real
      h : And (Ne p 0) (Ne p 1)
      hp₀ : LT.lt p 0
      ⊢ Ne (Real.binEntropy p) 0
    -/
  · exact (binEntropy_neg_of_neg hp₀).ne
    /-
      🎉 no goals
    -/
  /-
    case inr
    p : Real
    h : And (Ne p 0) (Ne p 1)
    hp₀ : LT.lt 0 p
    ⊢ Ne (Real.binEntropy p) 0
  -/
  obtain hp₁ | hp₁ := h.2.lt_or_lt.symm
    /-
      case inr.inl
      p : Real
      h : And (Ne p 0) (Ne p 1)
      hp₀ : LT.lt 0 p
      hp₁ : LT.lt 1 p
      ⊢ Ne (Real.binEntropy p) 0
    -/
  · exact (binEntropy_neg_of_one_lt hp₁).ne
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      p : Real
      h : And (Ne p 0) (Ne p 1)
      hp₀ : LT.lt 0 p
      hp₁ : LT.lt p 1
      ⊢ Ne (Real.binEntropy p) 0
    -/
  · exact (binEntropy_pos hp₀ hp₁).ne'
    /-
      🎉 no goals
    -/


/-- For probability `p ≠ 0.5`, `binEntropy p < log 2`. -/
lemma binEntropy_lt_log_two : binEntropy p < log 2 ↔ p ≠ 2⁻¹ := by
  /-
    p : Real
    ⊢ Iff (LT.lt (Real.binEntropy p) (Real.log 2)) (Ne p (Inv.inv 2))
  -/
  refine ⟨?_, fun h ↦ ?_⟩
    /-
      case refine_1
      p : Real
      ⊢ LT.lt (Real.binEntropy p) (Real.log 2) → Ne p (Inv.inv 2)
    -/
  · rintro h rfl
    /-
      case refine_1
      h : LT.lt (Real.binEntropy (Inv.inv 2)) (Real.log 2)
      ⊢ False
    -/
    simp at h
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    p : Real
    h : Ne p (Inv.inv 2)
    ⊢ LT.lt (Real.binEntropy p) (Real.log 2)
  -/
  wlog hp : p < 2⁻¹
  · have hp : 1 - p < 2⁻¹ := by
      rw [sub_lt_comm]; norm_num at *; linarith (config := { splitNe := true })
    /-
      case refine_2.inr
      p : Real
      h : Ne p (Inv.inv 2)
      this : ∀ {p : Real}, Ne p (Inv.inv 2) → LT.lt p (Inv.inv 2) → LT.lt (Real.binE …
      hp✝ : Not (LT.lt p (Inv.inv 2))
      hp : LT.lt (HSub.hSub 1 p) (Inv.inv 2)
      ⊢ LT.lt (Real.binEntropy p) (Real.log 2)
    -/
    rw [← binEntropy_one_sub]
    /-
      case refine_2.inr
      p : Real
      h : Ne p (Inv.inv 2)
      this : ∀ {p : Real}, Ne p (Inv.inv 2) → LT.lt p (Inv.inv 2) → LT.lt (Real.binE …
      hp✝ : Not (LT.lt p (Inv.inv 2))
      hp : LT.lt (HSub.hSub 1 p) (Inv.inv 2)
      ⊢ LT.lt (Real.binEntropy (HSub.hSub 1 p)) (Real.log 2)
    -/
    exact this hp.ne hp
    /-
      🎉 no goals
    -/
  /-
    p✝ p : Real
    h : Ne p (Inv.inv 2)
    hp : LT.lt p (Inv.inv 2)
    ⊢ LT.lt (Real.binEntropy p) (Real.log 2)
  -/
  obtain hp₀ | hp₀ := le_or_lt p 0
    /-
      case inl
      p✝ p : Real
      h : Ne p (Inv.inv 2)
      hp : LT.lt p (Inv.inv 2)
      hp₀ : LE.le p 0
      ⊢ LT.lt (Real.binEntropy p) (Real.log 2)
    -/
  · exact (binEntropy_nonpos_of_nonpos hp₀).trans_lt <| log_pos <| by norm_num
    /-
      🎉 no goals
    -/
  /-
    case inr
    p✝ p : Real
    h : Ne p (Inv.inv 2)
    hp : LT.lt p (Inv.inv 2)
    hp₀ : LT.lt 0 p
    ⊢ LT.lt (Real.binEntropy p) (Real.log 2)
  -/
  have hp₁ : 0 < 1 - p := sub_pos.2 <| hp.trans <| by norm_num
  calc
  _ < log (p * p⁻¹ + (1 - p) * (1 - p)⁻¹) :=
    strictConcaveOn_log_Ioi.2 (inv_pos.2 hp₀) (inv_pos.2 hp₁)
      (by simpa [eq_sub_iff_add_eq, ← two_mul, mul_comm, mul_eq_one_iff_eq_inv₀]) hp₀ hp₁ (by simp)
  _ = log 2 := by rw [mul_inv_cancel₀, mul_inv_cancel₀, one_add_one_eq_two] <;> positivity


lemma binEntropy_le_log_two : binEntropy p ≤ log 2 := by
  /-
    p : Real
    ⊢ LE.le (Real.binEntropy p) (Real.log 2)
  -/
  obtain rfl | hp := eq_or_ne p 2⁻¹
    /-
      case inl
      ⊢ LE.le (Real.binEntropy (Inv.inv 2)) (Real.log 2)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      p : Real
      hp : Ne p (Inv.inv 2)
      ⊢ LE.le (Real.binEntropy p) (Real.log 2)
    -/
  · exact (binEntropy_lt_log_two.2 hp).le
    /-
      🎉 no goals
    -/


lemma binEntropy_eq_log_two : binEntropy p = log 2 ↔ p = 2⁻¹ := by
  /-
    p : Real
    ⊢ Iff (Eq (Real.binEntropy p) (Real.log 2)) (Eq p (Inv.inv 2))
  -/
  rw [binEntropy_le_log_two.eq_iff_not_lt, binEntropy_lt_log_two, not_ne_iff]
  /-
    🎉 no goals
  -/


/-- Binary entropy is continuous everywhere.
This is due to definition of `Real.log` for negative numbers. -/
@[fun_prop] lemma binEntropy_continuous : Continuous binEntropy := by
  /-
    ⊢ Continuous Real.binEntropy
  -/
  rw [binEntropy_eq_negMulLog_add_negMulLog_one_sub']; fun_prop
                                                       /-
                                                         🎉 no goals
                                                       -/


@[fun_prop] lemma differentiableAt_binEntropy (hp₀ : p ≠ 0) (hp₁ : p ≠ 1) :
    DifferentiableAt ℝ binEntropy p := by
  /-
    p : Real
    hp₀ : Ne p 0
    hp₁ : Ne p 1
    ⊢ DifferentiableAt Real Real.binEntropy p
  -/
  rw [ne_comm, ← sub_ne_zero] at hp₁
  /-
    p : Real
    hp₀ : Ne p 0
    hp₁ : Ne (HSub.hSub 1 p) 0
    ⊢ DifferentiableAt Real Real.binEntropy p
  -/
  unfold binEntropy
  /-
    p : Real
    hp₀ : Ne p 0
    hp₁ : Ne (HSub.hSub 1 p) 0
    ⊢ DifferentiableAt Real (fun p => HAdd.hAdd (HMul.hMul p (Real.log (Inv.inv p) …
  -/
  simp only [log_inv, mul_neg]
  /-
    p : Real
    hp₀ : Ne p 0
    hp₁ : Ne (HSub.hSub 1 p) 0
    ⊢ DifferentiableAt Real (fun p => HAdd.hAdd (Neg.neg (HMul.hMul p (Real.log p) …
  -/
  fun_prop (disch := assumption)
  /-
    🎉 no goals
  -/


set_option push_neg.use_distrib true in
lemma differentiableAt_binEntropy_iff_ne_zero_one :
    DifferentiableAt ℝ binEntropy p ↔ p ≠ 0 ∧ p ≠ 1 := by
  /-
    p : Real
    ⊢ Iff (DifferentiableAt Real Real.binEntropy p) (And (Ne p 0) (Ne p 1))
  -/
  refine ⟨fun h ↦ ⟨?_, ?_⟩, fun h ↦ differentiableAt_binEntropy h.1 h.2⟩
        /-
          case refine_1
          p : Real
          h : DifferentiableAt Real Real.binEntropy p
          ⊢ Ne p 0
        -/
    <;> rintro rfl <;> unfold binEntropy at h
    /-
      case refine_1
      h : DifferentiableAt Real (fun p => HAdd.hAdd (HMul.hMul p (Real.log (Inv.inv  …
      ⊢ False
    -/
  · rw [DifferentiableAt.add_iff_left] at h
      /-
        case refine_1
        h : DifferentiableAt Real (fun p => HMul.hMul p (Real.log (Inv.inv p))) 0
        ⊢ False
      -/
    · simp [log_inv, mul_neg, ← neg_mul, ← negMulLog_def, differentiableAt_negMulLog_iff] at h
      /-
        🎉 no goals
      -/
      /-
        case refine_1
        h : DifferentiableAt Real (fun p => HAdd.hAdd (HMul.hMul p (Real.log (Inv.inv  …
        ⊢ DifferentiableAt Real (fun p => HMul.hMul (HSub.hSub 1 p) (Real.log (Inv.inv …
      -/
    · fun_prop (disch := simp)
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      h : DifferentiableAt Real (fun p => HAdd.hAdd (HMul.hMul p (Real.log (Inv.inv  …
      ⊢ False
    -/
  · rw [DifferentiableAt.add_iff_right, differentiableAt_iff_comp_const_sub (b := 1)] at h
      /-
        case refine_2
        h : DifferentiableAt Real (fun x => HMul.hMul (HSub.hSub 1 (HSub.hSub 1 x)) (R …
        ⊢ False
      -/
    · simp [log_inv, mul_neg, ← neg_mul, ← negMulLog_def, differentiableAt_negMulLog_iff] at h
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        h : DifferentiableAt Real (fun p => HAdd.hAdd (HMul.hMul p (Real.log (Inv.inv  …
        ⊢ DifferentiableAt Real (fun p => HMul.hMul p (Real.log (Inv.inv p))) 1
      -/
    · fun_prop (disch := simp)
      /-
        🎉 no goals
      -/


set_option push_neg.use_distrib true in
/-- Binary entropy has derivative `log (1 - p) - log p`.
It's not differentiable at `0` or `1` but the junk values of `deriv` and `log` coincide there. -/
lemma deriv_binEntropy (p : ℝ) : deriv binEntropy p = log (1 - p) - log p := by
  /-
    p : Real
    ⊢ Eq (deriv Real.binEntropy p) (HSub.hSub (Real.log (HSub.hSub 1 p)) (Real.log …
  -/
  by_cases hp : p ≠ 0 ∧ p ≠ 1
    /-
      case pos
      p : Real
      hp : And (Ne p 0) (Ne p 1)
      ⊢ Eq (deriv Real.binEntropy p) (HSub.hSub (Real.log (HSub.hSub 1 p)) (Real.log …
    -/
  · obtain ⟨hp₀, hp₁⟩ := hp
    /-
      case pos.intro
      p : Real
      hp₀ : Ne p 0
      hp₁ : Ne p 1
      ⊢ Eq (deriv Real.binEntropy p) (HSub.hSub (Real.log (HSub.hSub 1 p)) (Real.log …
    -/
    rw [ne_comm, ← sub_ne_zero] at hp₁
    rw [binEntropy_eq_negMulLog_add_negMulLog_one_sub', deriv_add, deriv_comp_const_sub,
      deriv_negMulLog hp₀, deriv_negMulLog hp₁]
      /-
        case pos.intro
        p : Real
        hp₀ : Ne p 0
        hp₁ : Ne (HSub.hSub 1 p) 0
        ⊢ Eq (HAdd.hAdd (HSub.hSub (Neg.neg (Real.log p)) 1) (Neg.neg (HSub.hSub (Neg. …
      -/
    · ring
      /-
        🎉 no goals
      -/
    /-
      case pos.intro.hf
      p : Real
      hp₀ : Ne p 0
      hp₁ : Ne (HSub.hSub 1 p) 0
      ⊢ DifferentiableAt Real Real.negMulLog p
    -/
    all_goals fun_prop (disch := assumption)
    /-
      🎉 no goals
    -/
  -- pathological case where `deriv = 0` since `binEntropy` is not differentiable there
    /-
      case neg
      p : Real
      hp : Not (And (Ne p 0) (Ne p 1))
      ⊢ Eq (deriv Real.binEntropy p) (HSub.hSub (Real.log (HSub.hSub 1 p)) (Real.log …
    -/
  · rw [deriv_zero_of_not_differentiableAt (differentiableAt_binEntropy_iff_ne_zero_one.not.2 hp)]
    /-
      case neg
      p : Real
      hp : Not (And (Ne p 0) (Ne p 1))
      ⊢ Eq 0 (HSub.hSub (Real.log (HSub.hSub 1 p)) (Real.log p))
    -/
    push_neg at hp
    /-
      case neg
      p : Real
      hp : Or (Eq p 0) (Eq p 1)
      ⊢ Eq 0 (HSub.hSub (Real.log (HSub.hSub 1 p)) (Real.log p))
    -/
                               /-
                                 🎉 no goals
                               -/
    obtain rfl | rfl := hp <;> simp
                               /-
                                 🎉 no goals
                               -/


/-- Shannon q-ary Entropy function (measured in Nats, i.e., using natural logs).

It's the Shannon entropy of a random variable with possible outcomes {1, ..., q}
where outcome `1` has probability `1 - p` and all other outcomes are equally likely.

The usual domain of definition is p ∈ [0,1], i.e., input is a probability.

This is a generalization of the binary entropy function `binEntropy`. -/
@[pp_nodot] noncomputable def qaryEntropy (q : ℕ) (p : ℝ) : ℝ := p * log (q - 1 : ℤ) + binEntropy p


                                                                   /-
                                                                     q : Nat
                                                                     ⊢ Eq (Real.qaryEntropy q 0) 0
                                                                   -/
@[simp] lemma qaryEntropy_zero (q : ℕ) : qaryEntropy q 0 = 0 := by simp [qaryEntropy]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/

                                                                                /-
                                                                                  q : Nat
                                                                                  ⊢ Eq (Real.qaryEntropy q 1) (Real.log ↑(HSub.hSub (↑q) 1))
                                                                                -/
@[simp] lemma qaryEntropy_one (q : ℕ) : qaryEntropy q 1 = log (q - 1 : ℤ) := by simp [qaryEntropy]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/

                                                                 /-
                                                                   ⊢ Eq (Real.qaryEntropy 2) Real.binEntropy
                                                                 -/
@[simp] lemma qaryEntropy_two : qaryEntropy 2 = binEntropy := by ext; simp [qaryEntropy]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


lemma qaryEntropy_pos (hp₀ : 0 < p) (hp₁ : p < 1) : 0 < qaryEntropy q p := by
  /-
    q : Nat
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    ⊢ LT.lt 0 (Real.qaryEntropy q p)
  -/
  unfold qaryEntropy
  /-
    q : Nat
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    ⊢ LT.lt 0 (HAdd.hAdd (HMul.hMul p (Real.log ↑(HSub.hSub (↑q) 1))) (Real.binEnt …
  -/
  have := binEntropy_pos hp₀ hp₁
  /-
    q : Nat
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    this : LT.lt 0 (Real.binEntropy p)
    ⊢ LT.lt 0 (HAdd.hAdd (HMul.hMul p (Real.log ↑(HSub.hSub (↑q) 1))) (Real.binEnt …
  -/
  positivity
  /-
    🎉 no goals
  -/


lemma qaryEntropy_nonneg (hp₀ : 0 ≤ p) (hp₁ : p ≤ 1) : 0 ≤ qaryEntropy q p := by
  /-
    q : Nat
    p : Real
    hp₀ : LE.le 0 p
    hp₁ : LE.le p 1
    ⊢ LE.le 0 (Real.qaryEntropy q p)
  -/
  obtain rfl | hp₀ := hp₀.eq_or_lt
    /-
      case inl
      q : Nat
      hp₀ : LE.le 0 0
      hp₁ : LE.le 0 1
      ⊢ LE.le 0 (Real.qaryEntropy q 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    q : Nat
    p : Real
    hp₀✝ : LE.le 0 p
    hp₁ : LE.le p 1
    hp₀ : LT.lt 0 p
    ⊢ LE.le 0 (Real.qaryEntropy q p)
  -/
  obtain rfl | hp₁ := hp₁.eq_or_lt
    /-
      case inr.inl
      q : Nat
      hp₀✝ : LE.le 0 1
      hp₁ : LE.le 1 1
      hp₀ : LT.lt 0 1
      ⊢ LE.le 0 (Real.qaryEntropy q 1)
    -/
  · simpa [qaryEntropy, -Int.cast_sub] using log_intCast_nonneg _
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    q : Nat
    p : Real
    hp₀✝ : LE.le 0 p
    hp₁✝ : LE.le p 1
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    ⊢ LE.le 0 (Real.qaryEntropy q p)
  -/
  exact (qaryEntropy_pos hp₀ hp₁).le
  /-
    🎉 no goals
  -/


/-- Outside the usual range of `qaryEntropy`, it is negative. This is due to `log p = log |p|`. -/
lemma qaryEntropy_neg_of_neg (hp : p < 0) : qaryEntropy q p < 0 :=
  add_neg_of_nonpos_of_neg (mul_nonpos_of_nonpos_of_nonneg hp.le (log_intCast_nonneg _))
    (binEntropy_neg_of_neg hp)


/-- Outside the usual range of `qaryEntropy`, it is negative. This is due to `log p = log |p|`. -/
lemma qaryEntropy_nonpos_of_nonpos (hp : p ≤ 0) : qaryEntropy q p ≤ 0 :=
  add_nonpos (mul_nonpos_of_nonpos_of_nonneg hp (log_intCast_nonneg _))
    (binEntropy_nonpos_of_nonpos hp)


/-- The q-ary entropy function is continuous everywhere.
This is due to definition of `Real.log` for negative numbers. -/
@[fun_prop] lemma qaryEntropy_continuous : Continuous (qaryEntropy q) := by
  /-
    q : Nat
    ⊢ Continuous (Real.qaryEntropy q)
  -/
  unfold qaryEntropy; fun_prop
                      /-
                        🎉 no goals
                      -/


@[fun_prop] lemma differentiableAt_qaryEntropy (hp₀ : p ≠ 0) (hp₁ : p ≠ 1) :
                                               /-
                                                 q : Nat
                                                 p : Real
                                                 hp₀ : Ne p 0
                                                 hp₁ : Ne p 1
                                                 ⊢ DifferentiableAt Real (Real.qaryEntropy q) p
                                               -/
    DifferentiableAt ℝ (qaryEntropy q) p := by unfold qaryEntropy; fun_prop (disch := assumption)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


lemma deriv_qaryEntropy (hp₀ : p ≠ 0) (hp₁ : p ≠ 1) :
    deriv (qaryEntropy q) p = log (q - 1) + log (1 - p) - log p := by
  /-
    q : Nat
    p : Real
    hp₀ : Ne p 0
    hp₁ : Ne p 1
    ⊢ Eq (deriv (Real.qaryEntropy q) p) (HSub.hSub (HAdd.hAdd (Real.log (HSub.hSub …
  -/
  unfold qaryEntropy
  /-
    q : Nat
    p : Real
    hp₀ : Ne p 0
    hp₁ : Ne p 1
    ⊢ Eq (deriv (fun p => HAdd.hAdd (HMul.hMul p (Real.log ↑(HSub.hSub (↑q) 1))) ( …
  -/
  rw [deriv_add]
  · simp only [Int.cast_sub, Int.cast_natCast, Int.cast_one, differentiableAt_id', deriv_mul_const,
      deriv_id'', one_mul, deriv_binEntropy, add_sub_assoc]
  /-
    case hf
    q : Nat
    p : Real
    hp₀ : Ne p 0
    hp₁ : Ne p 1
    ⊢ DifferentiableAt Real (fun p => HMul.hMul p (Real.log ↑(HSub.hSub (↑q) 1))) p
  -/
  all_goals fun_prop (disch := assumption)
  /-
    🎉 no goals
  -/


/-- Binary entropy has derivative `log (1 - p) - log p`. -/
lemma hasDerivAt_binEntropy (hp₀ : p ≠ 0) (hp₁ : p ≠ 1) :
    HasDerivAt binEntropy (log (1 - p) - log p) p :=
  deriv_binEntropy _ ▸ (differentiableAt_binEntropy hp₀ hp₁).hasDerivAt


lemma hasDerivAt_qaryEntropy (hp₀ : p ≠ 0) (hp₁ : p ≠ 1) :
    HasDerivAt (qaryEntropy q) (log (q - 1) + log (1 - p) - log p) p :=
  deriv_qaryEntropy hp₀ hp₁ ▸ (differentiableAt_qaryEntropy hp₀ hp₁).hasDerivAt


private lemma tendsto_log_one_sub_sub_log_nhdsGT_atAtop :
    Tendsto (fun p ↦ log (1 - p) - log p) (𝓝[>] 0) atTop := by
  /-
    ⊢ Filter.Tendsto (fun p => HSub.hSub (Real.log (HSub.hSub 1 p)) (Real.log p))  …
  -/
  apply Filter.tendsto_atTop_add_left_of_le' (𝓝[>] 0) (log (1/2) : ℝ)
    /-
      case hf
      ⊢ Filter.Eventually (fun x => LE.le (Real.log (1 / 2)) (Real.log (HSub.hSub 1  …
    -/
  · have h₁ : (0 : ℝ) < 1 / 2 := by norm_num
    /-
      case hf
      h₁ : LT.lt 0 (1 / 2)
      ⊢ Filter.Eventually (fun x => LE.le (Real.log (1 / 2)) (Real.log (HSub.hSub 1  …
    -/
    filter_upwards [Ioc_mem_nhdsGT h₁] with p hx
    /-
      case h
      h₁ : LT.lt 0 (1 / 2)
      p : Real
      hx : Membership.mem (Set.Ioc 0 (1 / 2)) p
      ⊢ LE.le (Real.log (1 / 2)) (Real.log (HSub.hSub 1 p))
    -/
    gcongr
    /-
      case h.hxy
      h₁ : LT.lt 0 (1 / 2)
      p : Real
      hx : Membership.mem (Set.Ioc 0 (1 / 2)) p
      ⊢ LE.le (1 / 2) (HSub.hSub 1 p)
    -/
    linarith [hx.2]
    /-
      🎉 no goals
    -/
    /-
      case hg
      ⊢ Filter.Tendsto (fun x => Neg.neg (Real.log x)) (nhdsWithin 0 (Set.Ioi 0)) Fi …
    -/
  · apply tendsto_neg_atTop_iff.mpr tendsto_log_nhdsWithin_zero_right
    /-
      🎉 no goals
    -/


private lemma tendsto_log_one_sub_sub_log_nhdsLT_one_atBot :
    Tendsto (fun p ↦ log (1 - p) - log p) (𝓝[<] 1) atBot := by
  /-
    ⊢ Filter.Tendsto (fun p => HSub.hSub (Real.log (HSub.hSub 1 p)) (Real.log p))  …
  -/
  apply Filter.tendsto_atBot_add_right_of_ge' (𝓝[<] 1) (-log (1 - 2⁻¹))
    /-
      case hf
      ⊢ Filter.Tendsto (fun x => Real.log (HSub.hSub 1 x)) (nhdsWithin 1 (Set.Iio 1) …
    -/
  · have : Tendsto log (𝓝[>] 0) atBot := Real.tendsto_log_nhdsWithin_zero_right
    /-
      case hf
      this : Filter.Tendsto Real.log (nhdsWithin 0 (Set.Ioi 0)) Filter.atBot
      ⊢ Filter.Tendsto (fun x => Real.log (HSub.hSub 1 x)) (nhdsWithin 1 (Set.Iio 1) …
    -/
    apply Tendsto.comp (f := (1 - ·)) (g := log) this
    /-
      case hf
      this : Filter.Tendsto Real.log (nhdsWithin 0 (Set.Ioi 0)) Filter.atBot
      ⊢ Filter.Tendsto (fun x => HSub.hSub 1 x) (nhdsWithin 1 (Set.Iio 1)) (nhdsWith …
    -/
    have contF : Continuous ((1 : ℝ) - ·) := continuous_sub_left 1
    have : MapsTo ((1 : ℝ) - ·) (Iio 1) (Ioi 0) := by
      intro p hx
      simp_all only [mem_Iio, mem_Ioi, sub_pos]
    /-
      case hf
      this✝ : Filter.Tendsto Real.log (nhdsWithin 0 (Set.Ioi 0)) Filter.atBot
      contF : Continuous fun x => HSub.hSub 1 x
      this : Set.MapsTo (fun x => HSub.hSub 1 x) (Set.Iio 1) (Set.Ioi 0)
      ⊢ Filter.Tendsto (fun x => HSub.hSub 1 x) (nhdsWithin 1 (Set.Iio 1)) (nhdsWith …
    -/
    convert ContinuousWithinAt.tendsto_nhdsWithin (x :=(1 : ℝ)) contF.continuousWithinAt this
    /-
      case h.e'_5.h.e'_3
      this✝ : Filter.Tendsto Real.log (nhdsWithin 0 (Set.Ioi 0)) Filter.atBot
      contF : Continuous fun x => HSub.hSub 1 x
      this : Set.MapsTo (fun x => HSub.hSub 1 x) (Set.Iio 1) (Set.Ioi 0)
      ⊢ Eq 0 (HSub.hSub 1 1)
    -/
    exact Eq.symm (sub_eq_zero_of_eq rfl)
    /-
      🎉 no goals
    -/
    /-
      case hg
      ⊢ Filter.Eventually (fun x => LE.le (Neg.neg (Real.log x)) (Neg.neg (Real.log  …
    -/
  · have h₁ : (1 : ℝ) - (2 : ℝ)⁻¹ < 1 := by norm_num
    /-
      case hg
      h₁ : LT.lt (HSub.hSub 1 (Inv.inv 2)) 1
      ⊢ Filter.Eventually (fun x => LE.le (Neg.neg (Real.log x)) (Neg.neg (Real.log  …
    -/
    filter_upwards [Ico_mem_nhdsLT h₁] with p hx
    /-
      case h
      h₁ : LT.lt (HSub.hSub 1 (Inv.inv 2)) 1
      p : Real
      hx : Membership.mem (Set.Ico (HSub.hSub 1 (Inv.inv 2)) 1) p
      ⊢ LE.le (Neg.neg (Real.log p)) (Neg.neg (Real.log (HSub.hSub 1 (Inv.inv 2))))
    -/
    gcongr
    /-
      case h.a.hxy
      h₁ : LT.lt (HSub.hSub 1 (Inv.inv 2)) 1
      p : Real
      hx : Membership.mem (Set.Ico (HSub.hSub 1 (Inv.inv 2)) 1) p
      ⊢ LE.le (HSub.hSub 1 (Inv.inv 2)) p
    -/
    exact hx.1
    /-
      🎉 no goals
    -/


lemma not_continuousAt_deriv_qaryEntropy_one :
    ¬ContinuousAt (deriv (qaryEntropy q)) 1 := by
  have tendstoBot : Tendsto (fun p ↦ log (q - 1) + log (1 - p) - log p) (𝓝[<] 1) atBot := by
    have : (fun p ↦ log (q - 1) + log (1 - p) - log p)
      = (fun p ↦ log (q - 1) + (log (1 - p) - log p)) := by
      ext
      ring
    rw [this]
    apply tendsto_atBot_add_const_left
    exact tendsto_log_one_sub_sub_log_nhdsLT_one_atBot
  /-
    q : Nat
    tendstoBot : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
    ⊢ Not (ContinuousAt (deriv (Real.qaryEntropy q)) 1)
  -/
  apply not_continuousAt_of_tendsto (Filter.Tendsto.congr' _ tendstoBot) nhdsWithin_le_nhds
    /-
      q : Nat
      tendstoBot : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
      ⊢ Disjoint (nhds (deriv (Real.qaryEntropy q) 1)) Filter.atBot
    -/
  · simp only [disjoint_nhds_atBot_iff, not_isBot, not_false_eq_true]
    /-
      🎉 no goals
    -/
  /-
    q : Nat
    tendstoBot : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
    ⊢ (nhdsWithin 1 (Set.Iio 1)).EventuallyEq (fun p => HSub.hSub (HAdd.hAdd (Real …
  -/
  filter_upwards [Ioo_mem_nhdsLT (show 1 - 2⁻¹ < (1 : ℝ) by norm_num)]
  /-
    case h
    q : Nat
    tendstoBot : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
    ⊢ ∀ (a : Real), Membership.mem (Set.Ioo (HSub.hSub 1 (Inv.inv 2)) 1) a → Eq (H …
  -/
  intros
  /-
    case h
    q : Nat
    tendstoBot : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
    a✝¹ : Real
    a✝ : Membership.mem (Set.Ioo (HSub.hSub 1 (Inv.inv 2)) 1) a✝¹
    ⊢ Eq (HSub.hSub (HAdd.hAdd (Real.log (HSub.hSub (↑q) 1)) (Real.log (HSub.hSub  …
  -/
  apply (deriv_qaryEntropy _ _).symm
    /-
      q : Nat
      tendstoBot : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
      a✝¹ : Real
      a✝ : Membership.mem (Set.Ioo (HSub.hSub 1 (Inv.inv 2)) 1) a✝¹
      ⊢ Ne a✝¹ 0
    -/
  · simp_all only [mem_Ioo, ne_eq]
    /-
      q : Nat
      tendstoBot : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
      a✝¹ : Real
      a✝ : And (LT.lt (HSub.hSub 1 (Inv.inv 2)) a✝¹) (LT.lt a✝¹ 1)
      ⊢ Not (Eq a✝¹ 0)
    -/
    linarith [show (1 : ℝ) = 2⁻¹ + 2⁻¹ by norm_num]
    /-
      🎉 no goals
    -/
    /-
      q : Nat
      tendstoBot : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
      a✝¹ : Real
      a✝ : Membership.mem (Set.Ioo (HSub.hSub 1 (Inv.inv 2)) 1) a✝¹
      ⊢ Ne a✝¹ 1
    -/
  · simp_all only [mem_Ioo, ne_eq]
    /-
      q : Nat
      tendstoBot : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
      a✝¹ : Real
      a✝ : And (LT.lt (HSub.hSub 1 (Inv.inv 2)) a✝¹) (LT.lt a✝¹ 1)
      ⊢ Not (Eq a✝¹ 1)
    -/
    linarith [two_inv_lt_one (α := ℝ)]
    /-
      🎉 no goals
    -/


lemma not_continuousAt_deriv_qaryEntropy_zero :
    ¬ContinuousAt (deriv (qaryEntropy q)) 0 := by
  have tendstoTop : Tendsto (fun p ↦ log (q - 1) + log (1 - p) - log p) (𝓝[>] 0) atTop := by
    have : (fun p ↦ log (q - 1) + log (1 - p) - log p)
        = (fun p ↦ log (q - 1) + (log (1 - p) - log p)) := by ext; ring
    rw [this]
    exact tendsto_atTop_add_const_left _ _ tendsto_log_one_sub_sub_log_nhdsGT_atAtop
  /-
    q : Nat
    tendstoTop : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
    ⊢ Not (ContinuousAt (deriv (Real.qaryEntropy q)) 0)
  -/
  apply not_continuousAt_of_tendsto (Filter.Tendsto.congr' _ tendstoTop) nhdsWithin_le_nhds
    /-
      q : Nat
      tendstoTop : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
      ⊢ Disjoint (nhds (deriv (Real.qaryEntropy q) 0)) Filter.atTop
    -/
  · simp only [disjoint_nhds_atTop_iff, not_isTop, not_false_eq_true]
    /-
      🎉 no goals
    -/
  /-
    q : Nat
    tendstoTop : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
    ⊢ (nhdsWithin 0 (Set.Ioi 0)).EventuallyEq (fun p => HSub.hSub (HAdd.hAdd (Real …
  -/
  filter_upwards [Ioo_mem_nhdsGT (show (0 : ℝ) < 2⁻¹ by norm_num)]
  /-
    case h
    q : Nat
    tendstoTop : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
    ⊢ ∀ (a : Real), Membership.mem (Set.Ioo 0 (Inv.inv 2)) a → Eq (HSub.hSub (HAdd …
  -/
  intros
  /-
    case h
    q : Nat
    tendstoTop : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
    a✝¹ : Real
    a✝ : Membership.mem (Set.Ioo 0 (Inv.inv 2)) a✝¹
    ⊢ Eq (HSub.hSub (HAdd.hAdd (Real.log (HSub.hSub (↑q) 1)) (Real.log (HSub.hSub  …
  -/
  apply (deriv_qaryEntropy _ _).symm
    /-
      q : Nat
      tendstoTop : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
      a✝¹ : Real
      a✝ : Membership.mem (Set.Ioo 0 (Inv.inv 2)) a✝¹
      ⊢ Ne a✝¹ 0
    -/
  · simp_all only [zero_add, mem_Ioo, ne_eq]
    /-
      q : Nat
      tendstoTop : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
      a✝¹ : Real
      a✝ : And (LT.lt 0 a✝¹) (LT.lt a✝¹ (Inv.inv 2))
      ⊢ Not (Eq a✝¹ 0)
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      q : Nat
      tendstoTop : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
      a✝¹ : Real
      a✝ : Membership.mem (Set.Ioo 0 (Inv.inv 2)) a✝¹
      ⊢ Ne a✝¹ 1
    -/
  · simp_all only [zero_add, mem_Ioo, ne_eq]
    /-
      q : Nat
      tendstoTop : Filter.Tendsto (fun p => HSub.hSub (HAdd.hAdd (Real.log (HSub.hSu …
      a✝¹ : Real
      a✝ : And (LT.lt 0 a✝¹) (LT.lt a✝¹ (Inv.inv 2))
      ⊢ Not (Eq a✝¹ 1)
    -/
    linarith [two_inv_lt_one (α := ℝ)]
    /-
      🎉 no goals
    -/


/-- Second derivative of q-ary entropy. -/
lemma deriv2_qaryEntropy :
    deriv^[2] (qaryEntropy q) p = -1 / (p * (1 - p)) := by
  /-
    q : Nat
    p : Real
    ⊢ Eq (Nat.iterate deriv 2 (Real.qaryEntropy q) p) (HDiv.hDiv (-1) (HMul.hMul p …
  -/
  simp only [Function.iterate_succ, Function.iterate_zero, Function.id_comp, Function.comp_apply]
  /-
    q : Nat
    p : Real
    ⊢ Eq (deriv (deriv (Real.qaryEntropy q)) p) (HDiv.hDiv (-1) (HMul.hMul p (HSub …
  -/
  by_cases is_x_where_nondiff : p ≠ 0 ∧ p ≠ 1  -- normal case
    /-
      case pos
      q : Nat
      p : Real
      is_x_where_nondiff : And (Ne p 0) (Ne p 1)
      ⊢ Eq (deriv (deriv (Real.qaryEntropy q)) p) (HDiv.hDiv (-1) (HMul.hMul p (HSub …
    -/
  · obtain ⟨xne0, xne1⟩ := is_x_where_nondiff
    suffices ∀ᶠ y in (𝓝 p),
        deriv (fun p ↦ (qaryEntropy q) p) y = log (q - 1) + log (1 - y) - log y by
      refine (Filter.EventuallyEq.deriv_eq this).trans ?_
      rw [deriv_sub ?_ (differentiableAt_log xne0)]
      · rw [deriv.log differentiableAt_id' xne0]
        simp only [deriv_id'', one_div]
        · have {q : ℝ} (p : ℝ) : DifferentiableAt ℝ (fun p => q - p) p := by fun_prop
          have d_oneminus (p : ℝ) : deriv (fun (y : ℝ) ↦ 1 - y) p = -1 := by
            rw [deriv_const_sub 1, deriv_id'']
          field_simp [sub_ne_zero_of_ne xne1.symm, this, d_oneminus]
          ring
      · apply DifferentiableAt.add
        simp only [ne_eq, differentiableAt_const]
        exact DifferentiableAt.log (by fun_prop) (sub_ne_zero.mpr xne1.symm)
    filter_upwards [eventually_ne_nhds xne0, eventually_ne_nhds xne1]
      with y xne0 h2 using deriv_qaryEntropy xne0 h2
  -- Pathological case where we use junk value (because function not differentiable)
    /-
      case neg
      q : Nat
      p : Real
      is_x_where_nondiff : Not (And (Ne p 0) (Ne p 1))
      ⊢ Eq (deriv (deriv (Real.qaryEntropy q)) p) (HDiv.hDiv (-1) (HMul.hMul p (HSub …
    -/
  · have : p = 0 ∨ p = 1 := Decidable.or_iff_not_and_not.mpr is_x_where_nondiff
    /-
      case neg
      q : Nat
      p : Real
      is_x_where_nondiff : Not (And (Ne p 0) (Ne p 1))
      this : Or (Eq p 0) (Eq p 1)
      ⊢ Eq (deriv (deriv (Real.qaryEntropy q)) p) (HDiv.hDiv (-1) (HMul.hMul p (HSub …
    -/
    rw [deriv_zero_of_not_differentiableAt]
      /-
        case neg
        q : Nat
        p : Real
        is_x_where_nondiff : Not (And (Ne p 0) (Ne p 1))
        this : Or (Eq p 0) (Eq p 1)
        ⊢ Eq 0 (HDiv.hDiv (-1) (HMul.hMul p (HSub.hSub 1 p)))
      -/
    · simp_all only [ne_eq, not_and, Decidable.not_not]
      /-
        case neg
        q : Nat
        p : Real
        is_x_where_nondiff : Not (Eq p 0) → Eq p 1
        this : Or (Eq p 0) (Eq p 1)
        ⊢ Eq 0 (HDiv.hDiv (-1) (HMul.hMul p (HSub.hSub 1 p)))
      -/
      cases this <;> simp_all only [
        mul_zero, one_ne_zero, zero_ne_one, sub_zero, mul_one, div_zero, sub_self]
      /-
        case neg
        q : Nat
        p : Real
        is_x_where_nondiff : Not (And (Ne p 0) (Ne p 1))
        this : Or (Eq p 0) (Eq p 1)
        ⊢ Not (DifferentiableAt Real (deriv (Real.qaryEntropy q)) p)
      -/
    · intro h
      /-
        case neg
        q : Nat
        p : Real
        is_x_where_nondiff : Not (And (Ne p 0) (Ne p 1))
        this : Or (Eq p 0) (Eq p 1)
        h : DifferentiableAt Real (deriv (Real.qaryEntropy q)) p
        ⊢ False
      -/
      have contAt := h.continuousAt
      /-
        case neg
        q : Nat
        p : Real
        is_x_where_nondiff : Not (And (Ne p 0) (Ne p 1))
        this : Or (Eq p 0) (Eq p 1)
        h : DifferentiableAt Real (deriv (Real.qaryEntropy q)) p
        contAt : ContinuousAt (deriv (Real.qaryEntropy q)) p
        ⊢ False
      -/
      cases this <;> simp_all [
        not_continuousAt_deriv_qaryEntropy_zero, not_continuousAt_deriv_qaryEntropy_one, contAt]


lemma deriv2_binEntropy : deriv^[2] binEntropy p = -1 / (p * (1 - p)) :=
  qaryEntropy_two ▸ deriv2_qaryEntropy


/-- Qary entropy is strictly increasing in the interval [0, 1 - q⁻¹]. -/
lemma qaryEntropy_strictMonoOn (qLe2 : 2 ≤ q) :
    StrictMonoOn (qaryEntropy q) (Icc 0 (1 - 1/q)) := by
  /-
    q : Nat
    qLe2 : LE.le 2 q
    ⊢ StrictMonoOn (Real.qaryEntropy q) (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q)))
  -/
  intro p1 hp1 p2 hp2 p1le2
  /-
    q : Nat
    qLe2 : LE.le 2 q
    p1 : Real
    hp1 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p1
    p2 : Real
    hp2 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p2
    p1le2 : LT.lt p1 p2
    ⊢ LT.lt (Real.qaryEntropy q p1) (Real.qaryEntropy q p2)
  -/
  apply strictMonoOn_of_deriv_pos (convex_Icc 0 (1 - 1/(q : ℝ))) _ _ hp1 hp2 p1le2
    /-
      q : Nat
      qLe2 : LE.le 2 q
      p1 : Real
      hp1 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p1
      p2 : Real
      hp2 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p2
      p1le2 : LT.lt p1 p2
      ⊢ ContinuousOn (Real.qaryEntropy q) (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q)))
    -/
  · exact qaryEntropy_continuous.continuousOn
    /-
      🎉 no goals
    -/
    /-
      q : Nat
      qLe2 : LE.le 2 q
      p1 : Real
      hp1 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p1
      p2 : Real
      hp2 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p2
      p1le2 : LT.lt p1 p2
      ⊢ ∀ (x : Real), Membership.mem (interior (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1  …
    -/
  · intro p hp
    /-
      q : Nat
      qLe2 : LE.le 2 q
      p1 : Real
      hp1 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p1
      p2 : Real
      hp2 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p2
      p1le2 : LT.lt p1 p2
      p : Real
      hp : Membership.mem (interior (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q)))) p
      ⊢ LT.lt 0 (deriv (Real.qaryEntropy q) p)
    -/
    have : 2 ≤ (q : ℝ) := Nat.ofNat_le_cast.mpr qLe2
    /-
      q : Nat
      qLe2 : LE.le 2 q
      p1 : Real
      hp1 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p1
      p2 : Real
      hp2 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p2
      p1le2 : LT.lt p1 p2
      p : Real
      hp : Membership.mem (interior (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q)))) p
      this : LE.le 2 ↑q
      ⊢ LT.lt 0 (deriv (Real.qaryEntropy q) p)
    -/
    have zero_le_qinv : 0 < (q : ℝ)⁻¹ := by positivity
    have : 0 < 1 - p := by
      simp only [sub_pos, hp.2]
      have p_lt_1_minus_qinv : p < 1 - (q : ℝ)⁻¹ := by
        simp_all only [inv_pos, interior_Icc, mem_Ioo, one_div]
      linarith
    /-
      q : Nat
      qLe2 : LE.le 2 q
      p1 : Real
      hp1 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p1
      p2 : Real
      hp2 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p2
      p1le2 : LT.lt p1 p2
      p : Real
      hp : Membership.mem (interior (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q)))) p
      this✝ : LE.le 2 ↑q
      zero_le_qinv : LT.lt 0 (Inv.inv ↑q)
      this : LT.lt 0 (HSub.hSub 1 p)
      ⊢ LT.lt 0 (deriv (Real.qaryEntropy q) p)
    -/
    simp only [one_div, interior_Icc, mem_Ioo] at hp
    /-
      q : Nat
      qLe2 : LE.le 2 q
      p1 : Real
      hp1 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p1
      p2 : Real
      hp2 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p2
      p1le2 : LT.lt p1 p2
      p : Real
      this✝ : LE.le 2 ↑q
      zero_le_qinv : LT.lt 0 (Inv.inv ↑q)
      this : LT.lt 0 (HSub.hSub 1 p)
      hp : And (LT.lt 0 p) (LT.lt p (HSub.hSub 1 (Inv.inv ↑q)))
      ⊢ LT.lt 0 (deriv (Real.qaryEntropy q) p)
    -/
    rw [deriv_qaryEntropy (by linarith)]
      /-
        q : Nat
        qLe2 : LE.le 2 q
        p1 : Real
        hp1 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p1
        p2 : Real
        hp2 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p2
        p1le2 : LT.lt p1 p2
        p : Real
        this✝ : LE.le 2 ↑q
        zero_le_qinv : LT.lt 0 (Inv.inv ↑q)
        this : LT.lt 0 (HSub.hSub 1 p)
        hp : And (LT.lt 0 p) (LT.lt p (HSub.hSub 1 (Inv.inv ↑q)))
        ⊢ LT.lt 0 (HSub.hSub (HAdd.hAdd (Real.log (HSub.hSub (↑q) 1)) (Real.log (HSub. …
      -/
    · field_simp
      /-
        q : Nat
        qLe2 : LE.le 2 q
        p1 : Real
        hp1 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p1
        p2 : Real
        hp2 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p2
        p1le2 : LT.lt p1 p2
        p : Real
        this✝ : LE.le 2 ↑q
        zero_le_qinv : LT.lt 0 (Inv.inv ↑q)
        this : LT.lt 0 (HSub.hSub 1 p)
        hp : And (LT.lt 0 p) (LT.lt p (HSub.hSub 1 (Inv.inv ↑q)))
        ⊢ LT.lt (Real.log p) (HAdd.hAdd (Real.log (HSub.hSub (↑q) 1)) (Real.log (HSub. …
      -/
      rw [← log_mul (by linarith) (by linarith)]
      /-
        q : Nat
        qLe2 : LE.le 2 q
        p1 : Real
        hp1 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p1
        p2 : Real
        hp2 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p2
        p1le2 : LT.lt p1 p2
        p : Real
        this✝ : LE.le 2 ↑q
        zero_le_qinv : LT.lt 0 (Inv.inv ↑q)
        this : LT.lt 0 (HSub.hSub 1 p)
        hp : And (LT.lt 0 p) (LT.lt p (HSub.hSub 1 (Inv.inv ↑q)))
        ⊢ LT.lt (Real.log p) (Real.log (HMul.hMul (HSub.hSub (↑q) 1) (HSub.hSub 1 p)))
      -/
      apply Real.strictMonoOn_log (mem_Ioi.mpr hp.1)
        /-
          case x
          q : Nat
          qLe2 : LE.le 2 q
          p1 : Real
          hp1 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p1
          p2 : Real
          hp2 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p2
          p1le2 : LT.lt p1 p2
          p : Real
          this✝ : LE.le 2 ↑q
          zero_le_qinv : LT.lt 0 (Inv.inv ↑q)
          this : LT.lt 0 (HSub.hSub 1 p)
          hp : And (LT.lt 0 p) (LT.lt p (HSub.hSub 1 (Inv.inv ↑q)))
          ⊢ Membership.mem (Set.Ioi 0) (HMul.hMul (HSub.hSub (↑q) 1) (HSub.hSub 1 p))
        -/
      · simp_all only [mem_Ioi, mul_pos_iff_of_pos_left, show 0 < (q : ℝ) - 1 by linarith]
        /-
          🎉 no goals
        -/
        /-
          case a
          q : Nat
          qLe2 : LE.le 2 q
          p1 : Real
          hp1 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p1
          p2 : Real
          hp2 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p2
          p1le2 : LT.lt p1 p2
          p : Real
          this✝ : LE.le 2 ↑q
          zero_le_qinv : LT.lt 0 (Inv.inv ↑q)
          this : LT.lt 0 (HSub.hSub 1 p)
          hp : And (LT.lt 0 p) (LT.lt p (HSub.hSub 1 (Inv.inv ↑q)))
          ⊢ LT.lt p (HMul.hMul (HSub.hSub (↑q) 1) (HSub.hSub 1 p))
        -/
      · have qpos : 0 < (q : ℝ) := by positivity
        have : q * p < q - 1 := by
          convert (mul_lt_mul_left qpos).2 hp.2 using 1
          simp only [mul_sub, mul_one, isUnit_iff_ne_zero, ne_eq, ne_of_gt qpos, not_false_eq_true,
            IsUnit.mul_inv_cancel]
        /-
          case a
          q : Nat
          qLe2 : LE.le 2 q
          p1 : Real
          hp1 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p1
          p2 : Real
          hp2 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p2
          p1le2 : LT.lt p1 p2
          p : Real
          this✝¹ : LE.le 2 ↑q
          zero_le_qinv : LT.lt 0 (Inv.inv ↑q)
          this✝ : LT.lt 0 (HSub.hSub 1 p)
          hp : And (LT.lt 0 p) (LT.lt p (HSub.hSub 1 (Inv.inv ↑q)))
          qpos : LT.lt 0 ↑q
          this : LT.lt (HMul.hMul (↑q) p) (HSub.hSub (↑q) 1)
          ⊢ LT.lt p (HMul.hMul (HSub.hSub (↑q) 1) (HSub.hSub 1 p))
        -/
        linarith
        /-
          🎉 no goals
        -/
    /-
      q : Nat
      qLe2 : LE.le 2 q
      p1 : Real
      hp1 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p1
      p2 : Real
      hp2 : Membership.mem (Set.Icc 0 (HSub.hSub 1 (HDiv.hDiv 1 ↑q))) p2
      p1le2 : LT.lt p1 p2
      p : Real
      this✝ : LE.le 2 ↑q
      zero_le_qinv : LT.lt 0 (Inv.inv ↑q)
      this : LT.lt 0 (HSub.hSub 1 p)
      hp : And (LT.lt 0 p) (LT.lt p (HSub.hSub 1 (Inv.inv ↑q)))
      ⊢ Ne p 1
    -/
    exact (ne_of_gt (lt_add_neg_iff_lt.mp this : p < 1)).symm
    /-
      🎉 no goals
    -/


/-- Qary entropy is strictly decreasing in the interval [1 - q⁻¹, 1]. -/
lemma qaryEntropy_strictAntiOn (qLe2 : 2 ≤ q) :
    StrictAntiOn (qaryEntropy q) (Icc (1 - 1/q) 1) := by
  /-
    q : Nat
    qLe2 : LE.le 2 q
    ⊢ StrictAntiOn (Real.qaryEntropy q) (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1)
  -/
  intro p1 hp1 p2 hp2 p1le2
  /-
    q : Nat
    qLe2 : LE.le 2 q
    p1 : Real
    hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
    p2 : Real
    hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
    p1le2 : LT.lt p1 p2
    ⊢ LT.lt (Real.qaryEntropy q p2) (Real.qaryEntropy q p1)
  -/
  apply strictAntiOn_of_deriv_neg (convex_Icc (1 - 1/(q : ℝ)) 1) _ _ hp1 hp2 p1le2
    /-
      q : Nat
      qLe2 : LE.le 2 q
      p1 : Real
      hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
      p2 : Real
      hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
      p1le2 : LT.lt p1 p2
      ⊢ ContinuousOn (Real.qaryEntropy q) (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1)
    -/
  · exact qaryEntropy_continuous.continuousOn
    /-
      🎉 no goals
    -/
    /-
      q : Nat
      qLe2 : LE.le 2 q
      p1 : Real
      hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
      p2 : Real
      hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
      p1le2 : LT.lt p1 p2
      ⊢ ∀ (x : Real), Membership.mem (interior (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q …
    -/
  · intro p hp
    /-
      q : Nat
      qLe2 : LE.le 2 q
      p1 : Real
      hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
      p2 : Real
      hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
      p1le2 : LT.lt p1 p2
      p : Real
      hp : Membership.mem (interior (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1)) p
      ⊢ LT.lt (deriv (Real.qaryEntropy q) p) 0
    -/
    have : 2 ≤ (q : ℝ) := Nat.ofNat_le_cast.mpr qLe2
    /-
      q : Nat
      qLe2 : LE.le 2 q
      p1 : Real
      hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
      p2 : Real
      hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
      p1le2 : LT.lt p1 p2
      p : Real
      hp : Membership.mem (interior (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1)) p
      this : LE.le 2 ↑q
      ⊢ LT.lt (deriv (Real.qaryEntropy q) p) 0
    -/
    have qinv_lt_1 : (q : ℝ)⁻¹ < 1 := inv_lt_one_of_one_lt₀ (by linarith)
    /-
      q : Nat
      qLe2 : LE.le 2 q
      p1 : Real
      hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
      p2 : Real
      hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
      p1le2 : LT.lt p1 p2
      p : Real
      hp : Membership.mem (interior (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1)) p
      this : LE.le 2 ↑q
      qinv_lt_1 : LT.lt (Inv.inv ↑q) 1
      ⊢ LT.lt (deriv (Real.qaryEntropy q) p) 0
    -/
    have zero_lt_1_sub_p : 0 < 1 - p := by simp_all only [sub_pos, hp.2, interior_Icc, mem_Ioo]
    /-
      q : Nat
      qLe2 : LE.le 2 q
      p1 : Real
      hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
      p2 : Real
      hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
      p1le2 : LT.lt p1 p2
      p : Real
      hp : Membership.mem (interior (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1)) p
      this : LE.le 2 ↑q
      qinv_lt_1 : LT.lt (Inv.inv ↑q) 1
      zero_lt_1_sub_p : LT.lt 0 (HSub.hSub 1 p)
      ⊢ LT.lt (deriv (Real.qaryEntropy q) p) 0
    -/
    simp only [one_div, interior_Icc, mem_Ioo] at hp
    /-
      q : Nat
      qLe2 : LE.le 2 q
      p1 : Real
      hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
      p2 : Real
      hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
      p1le2 : LT.lt p1 p2
      p : Real
      this : LE.le 2 ↑q
      qinv_lt_1 : LT.lt (Inv.inv ↑q) 1
      zero_lt_1_sub_p : LT.lt 0 (HSub.hSub 1 p)
      hp : And (LT.lt (HSub.hSub 1 (Inv.inv ↑q)) p) (LT.lt p 1)
      ⊢ LT.lt (deriv (Real.qaryEntropy q) p) 0
    -/
    rw [deriv_qaryEntropy (by linarith)]
      /-
        q : Nat
        qLe2 : LE.le 2 q
        p1 : Real
        hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
        p2 : Real
        hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
        p1le2 : LT.lt p1 p2
        p : Real
        this : LE.le 2 ↑q
        qinv_lt_1 : LT.lt (Inv.inv ↑q) 1
        zero_lt_1_sub_p : LT.lt 0 (HSub.hSub 1 p)
        hp : And (LT.lt (HSub.hSub 1 (Inv.inv ↑q)) p) (LT.lt p 1)
        ⊢ LT.lt (HSub.hSub (HAdd.hAdd (Real.log (HSub.hSub (↑q) 1)) (Real.log (HSub.hS …
      -/
    · field_simp
      /-
        q : Nat
        qLe2 : LE.le 2 q
        p1 : Real
        hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
        p2 : Real
        hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
        p1le2 : LT.lt p1 p2
        p : Real
        this : LE.le 2 ↑q
        qinv_lt_1 : LT.lt (Inv.inv ↑q) 1
        zero_lt_1_sub_p : LT.lt 0 (HSub.hSub 1 p)
        hp : And (LT.lt (HSub.hSub 1 (Inv.inv ↑q)) p) (LT.lt p 1)
        ⊢ LT.lt (HAdd.hAdd (Real.log (HSub.hSub (↑q) 1)) (Real.log (HSub.hSub 1 p))) ( …
      -/
      rw [← log_mul (by linarith) (by linarith)]
      /-
        q : Nat
        qLe2 : LE.le 2 q
        p1 : Real
        hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
        p2 : Real
        hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
        p1le2 : LT.lt p1 p2
        p : Real
        this : LE.le 2 ↑q
        qinv_lt_1 : LT.lt (Inv.inv ↑q) 1
        zero_lt_1_sub_p : LT.lt 0 (HSub.hSub 1 p)
        hp : And (LT.lt (HSub.hSub 1 (Inv.inv ↑q)) p) (LT.lt p 1)
        ⊢ LT.lt (Real.log (HMul.hMul (HSub.hSub (↑q) 1) (HSub.hSub 1 p))) (Real.log p)
      -/
      apply Real.strictMonoOn_log (mem_Ioi.mpr (show 0 < (↑q - 1) * (1 - p) by nlinarith))
        /-
          case x
          q : Nat
          qLe2 : LE.le 2 q
          p1 : Real
          hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
          p2 : Real
          hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
          p1le2 : LT.lt p1 p2
          p : Real
          this : LE.le 2 ↑q
          qinv_lt_1 : LT.lt (Inv.inv ↑q) 1
          zero_lt_1_sub_p : LT.lt 0 (HSub.hSub 1 p)
          hp : And (LT.lt (HSub.hSub 1 (Inv.inv ↑q)) p) (LT.lt p 1)
          ⊢ Membership.mem (Set.Ioi 0) p
        -/
      · simp_all only [mem_Ioi, mul_pos_iff_of_pos_left]
        /-
          case x
          q : Nat
          qLe2 : LE.le 2 q
          p1 : Real
          hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
          p2 : Real
          hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
          p1le2 : LT.lt p1 p2
          p : Real
          this : LE.le 2 ↑q
          qinv_lt_1 : LT.lt (Inv.inv ↑q) 1
          zero_lt_1_sub_p : LT.lt 0 (HSub.hSub 1 p)
          hp : And (LT.lt (HSub.hSub 1 (Inv.inv ↑q)) p) (LT.lt p 1)
          ⊢ LT.lt 0 p
        -/
        linarith
        /-
          🎉 no goals
        -/
        /-
          case a
          q : Nat
          qLe2 : LE.le 2 q
          p1 : Real
          hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
          p2 : Real
          hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
          p1le2 : LT.lt p1 p2
          p : Real
          this : LE.le 2 ↑q
          qinv_lt_1 : LT.lt (Inv.inv ↑q) 1
          zero_lt_1_sub_p : LT.lt 0 (HSub.hSub 1 p)
          hp : And (LT.lt (HSub.hSub 1 (Inv.inv ↑q)) p) (LT.lt p 1)
          ⊢ LT.lt (HMul.hMul (HSub.hSub (↑q) 1) (HSub.hSub 1 p)) p
        -/
      · have qpos : 0 < (q : ℝ) := by positivity
        /-
          case a
          q : Nat
          qLe2 : LE.le 2 q
          p1 : Real
          hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
          p2 : Real
          hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
          p1le2 : LT.lt p1 p2
          p : Real
          this : LE.le 2 ↑q
          qinv_lt_1 : LT.lt (Inv.inv ↑q) 1
          zero_lt_1_sub_p : LT.lt 0 (HSub.hSub 1 p)
          hp : And (LT.lt (HSub.hSub 1 (Inv.inv ↑q)) p) (LT.lt p 1)
          qpos : LT.lt 0 ↑q
          ⊢ LT.lt (HMul.hMul (HSub.hSub (↑q) 1) (HSub.hSub 1 p)) p
        -/
        ring_nf
        /-
          case a
          q : Nat
          qLe2 : LE.le 2 q
          p1 : Real
          hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
          p2 : Real
          hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
          p1le2 : LT.lt p1 p2
          p : Real
          this : LE.le 2 ↑q
          qinv_lt_1 : LT.lt (Inv.inv ↑q) 1
          zero_lt_1_sub_p : LT.lt 0 (HSub.hSub 1 p)
          hp : And (LT.lt (HSub.hSub 1 (Inv.inv ↑q)) p) (LT.lt p 1)
          qpos : LT.lt 0 ↑q
          ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd (-1) (HSub.hSub (↑q) (HMul.hMul (↑q) p))) p) p
        -/
        simp only [add_lt_iff_neg_right, neg_add_lt_iff_lt_add, add_zero, gt_iff_lt]
        have : (q : ℝ) - 1 < p * q := by
          have tmp := mul_lt_mul_of_pos_right hp.1 qpos
          simp at tmp
          have : (q : ℝ) ≠ 0 := (ne_of_lt qpos).symm
          have asdfasfd : (1 - (q : ℝ)⁻¹) * ↑q = q - 1 := by calc (1 - (q : ℝ)⁻¹) * ↑q
            _ = q - (q : ℝ)⁻¹ * (q : ℝ) := by ring
            _ = q - 1 := by simp_all only [ne_eq, isUnit_iff_ne_zero, Rat.cast_eq_zero,
              not_false_eq_true, IsUnit.inv_mul_cancel]
          rwa [asdfasfd] at tmp
        /-
          case a
          q : Nat
          qLe2 : LE.le 2 q
          p1 : Real
          hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
          p2 : Real
          hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
          p1le2 : LT.lt p1 p2
          p : Real
          this✝ : LE.le 2 ↑q
          qinv_lt_1 : LT.lt (Inv.inv ↑q) 1
          zero_lt_1_sub_p : LT.lt 0 (HSub.hSub 1 p)
          hp : And (LT.lt (HSub.hSub 1 (Inv.inv ↑q)) p) (LT.lt p 1)
          qpos : LT.lt 0 ↑q
          this : LT.lt (HSub.hSub (↑q) 1) (HMul.hMul p ↑q)
          ⊢ LT.lt (HSub.hSub (↑q) (HMul.hMul (↑q) p)) 1
        -/
        nlinarith
        /-
          🎉 no goals
        -/
    /-
      q : Nat
      qLe2 : LE.le 2 q
      p1 : Real
      hp1 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p1
      p2 : Real
      hp2 : Membership.mem (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑q)) 1) p2
      p1le2 : LT.lt p1 p2
      p : Real
      this : LE.le 2 ↑q
      qinv_lt_1 : LT.lt (Inv.inv ↑q) 1
      zero_lt_1_sub_p : LT.lt 0 (HSub.hSub 1 p)
      hp : And (LT.lt (HSub.hSub 1 (Inv.inv ↑q)) p) (LT.lt p 1)
      ⊢ Ne p 1
    -/
    exact (ne_of_gt (lt_add_neg_iff_lt.mp zero_lt_1_sub_p : p < 1)).symm
    /-
      🎉 no goals
    -/


/-- Binary entropy is strictly increasing in interval [0, 1/2]. -/
lemma binEntropy_strictMonoOn : StrictMonoOn binEntropy (Icc 0 2⁻¹) := by
  /-
    ⊢ StrictMonoOn Real.binEntropy (Set.Icc 0 (Inv.inv 2))
  -/
  rw [show Icc (0 : ℝ) 2⁻¹ = Icc 0 (1 - 1/2) by norm_num, ← qaryEntropy_two]
  /-
    ⊢ StrictMonoOn (Real.qaryEntropy 2) (Set.Icc 0 (HSub.hSub 1 (1 / 2)))
  -/
  exact qaryEntropy_strictMonoOn (by rfl)
  /-
    🎉 no goals
  -/


/-- Binary entropy is strictly decreasing in interval [1/2, 1]. -/
lemma binEntropy_strictAntiOn : StrictAntiOn binEntropy (Icc 2⁻¹ 1) := by
  /-
    ⊢ StrictAntiOn Real.binEntropy (Set.Icc (Inv.inv 2) 1)
  -/
  rw [show (Icc (2⁻¹ : ℝ) 1) = Icc (1/2) 1 by norm_num, ← qaryEntropy_two]
  /-
    ⊢ StrictAntiOn (Real.qaryEntropy 2) (Set.Icc (1 / 2) 1)
  -/
  convert qaryEntropy_strictAntiOn (by rfl) using 1
  /-
    case h.e'_6
    ⊢ Eq (Set.Icc (1 / 2) 1) (Set.Icc (HSub.hSub 1 (HDiv.hDiv 1 ↑2)) 1)
  -/
  norm_num
  /-
    🎉 no goals
  -/


lemma strictConcaveOn_qaryEntropy : StrictConcaveOn ℝ (Icc 0 1) (qaryEntropy q) := by
  /-
    q : Nat
    ⊢ StrictConcaveOn Real (Set.Icc 0 1) (Real.qaryEntropy q)
  -/
  apply strictConcaveOn_of_deriv2_neg (convex_Icc 0 1) qaryEntropy_continuous.continuousOn
  /-
    q : Nat
    ⊢ ∀ (x : Real), Membership.mem (interior (Set.Icc 0 1)) x → LT.lt (Nat.iterate …
  -/
  intro p hp
  /-
    q : Nat
    p : Real
    hp : Membership.mem (interior (Set.Icc 0 1)) p
    ⊢ LT.lt (Nat.iterate deriv 2 (Real.qaryEntropy q) p) 0
  -/
  rw [deriv2_qaryEntropy]
    /-
      q : Nat
      p : Real
      hp : Membership.mem (interior (Set.Icc 0 1)) p
      ⊢ LT.lt (HDiv.hDiv (-1) (HMul.hMul p (HSub.hSub 1 p))) 0
    -/
  · simp_all only [interior_Icc, mem_Ioo]
    /-
      q : Nat
      p : Real
      hp : And (LT.lt 0 p) (LT.lt p 1)
      ⊢ LT.lt (HDiv.hDiv (-1) (HMul.hMul p (HSub.hSub 1 p))) 0
    -/
    apply div_neg_of_neg_of_pos
      /-
        case ha
        q : Nat
        p : Real
        hp : And (LT.lt 0 p) (LT.lt p 1)
        ⊢ LT.lt (-1) 0
      -/
    · norm_num [show 0 < log 2 by positivity]
      /-
        🎉 no goals
      -/
      /-
        case hb
        q : Nat
        p : Real
        hp : And (LT.lt 0 p) (LT.lt p 1)
        ⊢ LT.lt 0 (HMul.hMul p (HSub.hSub 1 p))
      -/
    · simp_all only [gt_iff_lt, mul_pos_iff_of_pos_left, sub_pos, hp]
      /-
        🎉 no goals
      -/


lemma strictConcave_binEntropy : StrictConcaveOn ℝ (Icc 0 1) binEntropy :=
  qaryEntropy_two ▸ strictConcaveOn_qaryEntropy


