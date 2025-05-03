lemma isUltrametricDist_of_forall_norm_add_one_le_max_norm_one
    (h : ∀ x : R, ‖x + 1‖ ≤ max ‖x‖ 1) : IsUltrametricDist R := by
  /-
    R : Type u_1
    inst✝ : NormedDivisionRing R
    h : ∀ (x : R), LE.le (Norm.norm (HAdd.hAdd x 1)) (Max.max (Norm.norm x) 1)
    ⊢ IsUltrametricDist R
  -/
  refine isUltrametricDist_of_forall_norm_add_le_max_norm (fun x y ↦ ?_)
  /-
    R : Type u_1
    inst✝ : NormedDivisionRing R
    h : ∀ (x : R), LE.le (Norm.norm (HAdd.hAdd x 1)) (Max.max (Norm.norm x) 1)
    x y : R
    ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (Max.max (Norm.norm x) (Norm.norm y))
  -/
  rcases eq_or_ne y 0 with rfl | hy
    /-
      case inl
      R : Type u_1
      inst✝ : NormedDivisionRing R
      h : ∀ (x : R), LE.le (Norm.norm (HAdd.hAdd x 1)) (Max.max (Norm.norm x) 1)
      x : R
      ⊢ LE.le (Norm.norm (HAdd.hAdd x 0)) (Max.max (Norm.norm x) (Norm.norm 0))
    -/
  · simpa only [add_zero] using le_max_left _ _
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝ : NormedDivisionRing R
      h : ∀ (x : R), LE.le (Norm.norm (HAdd.hAdd x 1)) (Max.max (Norm.norm x) 1)
      x y : R
      hy : Ne y 0
      ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (Max.max (Norm.norm x) (Norm.norm y))
    -/
  · have p : 0 < ‖y‖ := norm_pos_iff.mpr hy
    simpa only [div_add_one hy, norm_div, div_le_iff₀ p, max_mul_of_nonneg _ _ p.le, one_mul,
      div_mul_cancel₀ _ p.ne'] using h (x / y)


lemma isUltrametricDist_of_forall_norm_add_one_of_norm_le_one
    (h : ∀ x : R, ‖x‖ ≤ 1 → ‖x + 1‖ ≤ 1) : IsUltrametricDist R := by
  /-
    R : Type u_1
    inst✝ : NormedDivisionRing R
    h : ∀ (x : R), LE.le (Norm.norm x) 1 → LE.le (Norm.norm (HAdd.hAdd x 1)) 1
    ⊢ IsUltrametricDist R
  -/
  refine isUltrametricDist_of_forall_norm_add_one_le_max_norm_one fun x ↦ ?_
  /-
    R : Type u_1
    inst✝ : NormedDivisionRing R
    h : ∀ (x : R), LE.le (Norm.norm x) 1 → LE.le (Norm.norm (HAdd.hAdd x 1)) 1
    x : R
    ⊢ LE.le (Norm.norm (HAdd.hAdd x 1)) (Max.max (Norm.norm x) 1)
  -/
  rcases le_or_lt ‖x‖ 1 with H|H
    /-
      case inl
      R : Type u_1
      inst✝ : NormedDivisionRing R
      h : ∀ (x : R), LE.le (Norm.norm x) 1 → LE.le (Norm.norm (HAdd.hAdd x 1)) 1
      x : R
      H : LE.le (Norm.norm x) 1
      ⊢ LE.le (Norm.norm (HAdd.hAdd x 1)) (Max.max (Norm.norm x) 1)
    -/
  · exact (h _ H).trans (le_max_right _ _)
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝ : NormedDivisionRing R
      h : ∀ (x : R), LE.le (Norm.norm x) 1 → LE.le (Norm.norm (HAdd.hAdd x 1)) 1
      x : R
      H : LT.lt 1 (Norm.norm x)
      ⊢ LE.le (Norm.norm (HAdd.hAdd x 1)) (Max.max (Norm.norm x) 1)
    -/
  · suffices ‖x + 1‖ ≤ ‖x‖ from this.trans (le_max_left _ _)
    rw [← div_le_one (by positivity), ← norm_div, add_div,
      div_self (by simpa using H.trans' zero_lt_one), add_comm]
    /-
      case inr
      R : Type u_1
      inst✝ : NormedDivisionRing R
      h : ∀ (x : R), LE.le (Norm.norm x) 1 → LE.le (Norm.norm (HAdd.hAdd x 1)) 1
      x : R
      H : LT.lt 1 (Norm.norm x)
      ⊢ LE.le (Norm.norm (HAdd.hAdd (HDiv.hDiv 1 x) 1)) 1
    -/
    apply h
    /-
      case inr.a
      R : Type u_1
      inst✝ : NormedDivisionRing R
      h : ∀ (x : R), LE.le (Norm.norm x) 1 → LE.le (Norm.norm (HAdd.hAdd x 1)) 1
      x : R
      H : LT.lt 1 (Norm.norm x)
      ⊢ LE.le (Norm.norm (HDiv.hDiv 1 x)) 1
    -/
    simp [inv_le_one_iff₀, H.le]
    /-
      🎉 no goals
    -/


lemma isUltrametricDist_of_forall_norm_sub_one_of_norm_le_one
    (h : ∀ x : R, ‖x‖ ≤ 1 → ‖x - 1‖ ≤ 1) : IsUltrametricDist R := by
  have (x : R) (hx : ‖x‖ ≤ 1) : ‖x + 1‖ ≤ 1 := by
    simpa only [← neg_add', norm_neg] using h (-x) (norm_neg x ▸ hx)
  /-
    R : Type u_1
    inst✝ : NormedDivisionRing R
    h : ∀ (x : R), LE.le (Norm.norm x) 1 → LE.le (Norm.norm (HSub.hSub x 1)) 1
    this : ∀ (x : R), LE.le (Norm.norm x) 1 → LE.le (Norm.norm (HAdd.hAdd x 1)) 1
    ⊢ IsUltrametricDist R
  -/
  exact isUltrametricDist_of_forall_norm_add_one_of_norm_le_one this
  /-
    🎉 no goals
  -/


/-- This technical lemma is used in the proof of
`isUltrametricDist_of_forall_norm_natCast_le_one`. -/
lemma isUltrametricDist_of_forall_pow_norm_le_nsmul_pow_max_one_norm
    (h : ∀ (x : R) (m : ℕ), ‖x + 1‖ ^ m ≤ (m + 1) • max 1 (‖x‖ ^ m)) :
    IsUltrametricDist R := by
  -- it will suffice to prove that `‖x + 1‖ ≤ max 1 ‖x‖`
  /-
    R : Type u_1
    inst✝ : NormedDivisionRing R
    h : ∀ (x : R) (m : Nat), LE.le (HPow.hPow (Norm.norm (HAdd.hAdd x 1)) m) (HSMu …
    ⊢ IsUltrametricDist R
  -/
  refine isUltrametricDist_of_forall_norm_add_one_le_max_norm_one fun x ↦ ?_
  -- Morally, we want to deduce this from the hypothesis `h` by taking an `m`-th root and showing
  -- that `(m + 1) ^ (1 / m)` gets arbitrarily close to 1, although we will formalise this in a way
  -- that avoids explicitly mentioning `m`-th roots.
  -- First note it suffices to show that `‖x + 1‖ ≤ a` for all `a : ℝ` with `max ‖x‖ 1 < a`.
  /-
    R : Type u_1
    inst✝ : NormedDivisionRing R
    h : ∀ (x : R) (m : Nat), LE.le (HPow.hPow (Norm.norm (HAdd.hAdd x 1)) m) (HSMu …
    x : R
    ⊢ LE.le (Norm.norm (HAdd.hAdd x 1)) (Max.max (Norm.norm x) 1)
  -/
  rw [max_comm]
  /-
    R : Type u_1
    inst✝ : NormedDivisionRing R
    h : ∀ (x : R) (m : Nat), LE.le (HPow.hPow (Norm.norm (HAdd.hAdd x 1)) m) (HSMu …
    x : R
    ⊢ LE.le (Norm.norm (HAdd.hAdd x 1)) (Max.max 1 (Norm.norm x))
  -/
  refine le_of_forall_le_of_dense fun a ha ↦ ?_
  /-
    R : Type u_1
    inst✝ : NormedDivisionRing R
    h : ∀ (x : R) (m : Nat), LE.le (HPow.hPow (Norm.norm (HAdd.hAdd x 1)) m) (HSMu …
    x : R
    a : Real
    ha : LT.lt (Max.max 1 (Norm.norm x)) a
    ⊢ LE.le (Norm.norm (HAdd.hAdd x 1)) a
  -/
  have ha' : 1 < a := (max_lt_iff.mp ha).left
  -- `max 1 ‖x‖ < a`, so there must be some `m : ℕ` such that `m + 1 < (a / max 1 ‖x‖) ^ m`
  -- by the virtue of exponential growth being faster than linear growth
  obtain ⟨m, hm⟩ : ∃ m : ℕ, ((m + 1) : ℕ) < (a / (max 1 ‖x‖)) ^ m := by
    apply_mod_cast Real.exists_natCast_add_one_lt_pow_of_one_lt
    rwa [one_lt_div (by positivity)]
  -- and we rearrange again to get `(m + 1) • max 1 ‖x‖ ^ m < a ^ m`
  /-
    case intro
    R : Type u_1
    inst✝ : NormedDivisionRing R
    h : ∀ (x : R) (m : Nat), LE.le (HPow.hPow (Norm.norm (HAdd.hAdd x 1)) m) (HSMu …
    x : R
    a : Real
    ha : LT.lt (Max.max 1 (Norm.norm x)) a
    ha' : LT.lt 1 a
    m : Nat
    hm : LT.lt (↑(HAdd.hAdd m 1)) (HPow.hPow (HDiv.hDiv a (Max.max 1 (Norm.norm x) …
    ⊢ LE.le (Norm.norm (HAdd.hAdd x 1)) a
  -/
  rw [div_pow, lt_div_iff₀ (by positivity), ← nsmul_eq_mul] at hm
  -- which squeezes down to get our `‖x + 1‖ ≤ a` using our to-be-proven hypothesis of
  -- `‖x + 1‖ ^ m ≤ (m + 1) • max 1 ‖x‖ ^ m`, so we're done
  -- we can distribute powers into the right term of `max`
  have hp : max 1 ‖x‖ ^ m = max 1 (‖x‖ ^ m) := by
    rw [pow_left_monotoneOn.map_max (by simp [zero_le_one]) (norm_nonneg x), one_pow]
  /-
    case intro
    R : Type u_1
    inst✝ : NormedDivisionRing R
    h : ∀ (x : R) (m : Nat), LE.le (HPow.hPow (Norm.norm (HAdd.hAdd x 1)) m) (HSMu …
    x : R
    a : Real
    ha : LT.lt (Max.max 1 (Norm.norm x)) a
    ha' : LT.lt 1 a
    m : Nat
    hm : LT.lt (HSMul.hSMul (HAdd.hAdd m 1) (HPow.hPow (Max.max 1 (Norm.norm x)) m …
    hp : Eq (HPow.hPow (Max.max 1 (Norm.norm x)) m) (Max.max 1 (HPow.hPow (Norm.no …
    ⊢ LE.le (Norm.norm (HAdd.hAdd x 1)) a
  -/
  rw [hp] at hm
  /-
    case intro
    R : Type u_1
    inst✝ : NormedDivisionRing R
    h : ∀ (x : R) (m : Nat), LE.le (HPow.hPow (Norm.norm (HAdd.hAdd x 1)) m) (HSMu …
    x : R
    a : Real
    ha : LT.lt (Max.max 1 (Norm.norm x)) a
    ha' : LT.lt 1 a
    m : Nat
    hm : LT.lt (HSMul.hSMul (HAdd.hAdd m 1) (Max.max 1 (HPow.hPow (Norm.norm x) m) …
    hp : Eq (HPow.hPow (Max.max 1 (Norm.norm x)) m) (Max.max 1 (HPow.hPow (Norm.no …
    ⊢ LE.le (Norm.norm (HAdd.hAdd x 1)) a
  -/
  refine le_of_pow_le_pow_left₀ (fun h ↦ ?_) (zero_lt_one.trans ha').le ((h _ _).trans hm.le)
  /-
    case intro
    R : Type u_1
    inst✝ : NormedDivisionRing R
    h✝ : ∀ (x : R) (m : Nat), LE.le (HPow.hPow (Norm.norm (HAdd.hAdd x 1)) m) (HSM …
    x : R
    a : Real
    ha : LT.lt (Max.max 1 (Norm.norm x)) a
    ha' : LT.lt 1 a
    m : Nat
    hm : LT.lt (HSMul.hSMul (HAdd.hAdd m 1) (Max.max 1 (HPow.hPow (Norm.norm x) m) …
    hp : Eq (HPow.hPow (Max.max 1 (Norm.norm x)) m) (Max.max 1 (HPow.hPow (Norm.no …
    h : Eq m 0
    ⊢ False
  -/
  simp only [h, zero_add, pow_zero, max_self, one_smul, lt_self_iff_false] at hm
  /-
    🎉 no goals
  -/


/-- To prove that a normed division ring is nonarchimedean, it suffices to prove that the norm
of the image of any natural is less than or equal to one. -/
lemma isUltrametricDist_of_forall_norm_natCast_le_one
    (h : ∀ n : ℕ, ‖(n : R)‖ ≤ 1) : IsUltrametricDist R := by
  -- from a previous lemma, suffices to prove that for all `m`, we have
  -- `‖x + 1‖ ^ m ≤ (m + 1) • max 1 ‖x‖ ^ m`
  /-
    R : Type u_1
    inst✝ : NormedDivisionRing R
    h : ∀ (n : Nat), LE.le (Norm.norm ↑n) 1
    ⊢ IsUltrametricDist R
  -/
  refine isUltrametricDist_of_forall_pow_norm_le_nsmul_pow_max_one_norm (fun x m ↦ ?_)
  -- we first use our hypothesis about the norm of naturals to have that multiplication by
  -- naturals keeps the norm small
  replace h (x : R) (n : ℕ) : ‖n • x‖ ≤ ‖x‖ := by
    rw [nsmul_eq_mul, norm_mul]
    rcases (norm_nonneg x).eq_or_lt with hx | hx
    · simp only [← hx, mul_zero, le_refl]
    · simpa only [mul_le_iff_le_one_left hx] using h _
  -- we expand the LHS using the binomial theorem, and apply the hypothesis to bound each term by
  -- a power of ‖x‖
  /-
    R : Type u_1
    inst✝ : NormedDivisionRing R
    x : R
    m : Nat
    h : ∀ (x : R) (n : Nat), LE.le (Norm.norm (HSMul.hSMul n x)) (Norm.norm x)
    ⊢ LE.le (HPow.hPow (Norm.norm (HAdd.hAdd x 1)) m) (HSMul.hSMul (HAdd.hAdd m 1) …
  -/
  transitivity ∑ k ∈ Finset.range (m + 1), ‖x‖ ^ k
  · simpa only [← norm_pow, (Commute.one_right x).add_pow, one_pow, mul_one, nsmul_eq_mul,
      Nat.cast_comm] using (norm_sum_le _ _).trans (Finset.sum_le_sum fun _ _ ↦ h _ _)
  -- the nature of the norm means that one of `1` and `‖x‖ ^ m` is the largest of the two, so the
  -- other terms in the binomial expansion are bounded by the max of these, and the number of terms
  -- in the sum is precisely `m + 1`
  /-
    R : Type u_1
    inst✝ : NormedDivisionRing R
    x : R
    m : Nat
    h : ∀ (x : R) (n : Nat), LE.le (Norm.norm (HSMul.hSMul n x)) (Norm.norm x)
    ⊢ LE.le ((Finset.range (HAdd.hAdd m 1)).sum fun k => HPow.hPow (Norm.norm x) k …
  -/
  rw [← Finset.card_range (m + 1), ← Finset.sum_const, Finset.card_range]
  /-
    R : Type u_1
    inst✝ : NormedDivisionRing R
    x : R
    m : Nat
    h : ∀ (x : R) (n : Nat), LE.le (Norm.norm (HSMul.hSMul n x)) (Norm.norm x)
    ⊢ LE.le ((Finset.range (HAdd.hAdd m 1)).sum fun k => HPow.hPow (Norm.norm x) k …
  -/
  rcases max_cases 1 (‖x‖ ^ m) with (⟨hm, hx⟩|⟨hm, hx⟩) <;> rw [hm] <;>
  -- which we show by comparing the terms in the sum one by one
  /-
    case inl.intro
    R : Type u_1
    inst✝ : NormedDivisionRing R
    x : R
    m : Nat
    h : ∀ (x : R) (n : Nat), LE.le (Norm.norm (HSMul.hSMul n x)) (Norm.norm x)
    hm : Eq (Max.max 1 (HPow.hPow (Norm.norm x) m)) 1
    hx : LE.le (HPow.hPow (Norm.norm x) m) 1
    ⊢ LE.le ((Finset.range (HAdd.hAdd m 1)).sum fun k => HPow.hPow (Norm.norm x) k …
  -/
  refine Finset.sum_le_sum fun i hi ↦ ?_
    /-
      case inl.intro
      R : Type u_1
      inst✝ : NormedDivisionRing R
      x : R
      m : Nat
      h : ∀ (x : R) (n : Nat), LE.le (Norm.norm (HSMul.hSMul n x)) (Norm.norm x)
      hm : Eq (Max.max 1 (HPow.hPow (Norm.norm x) m)) 1
      hx : LE.le (HPow.hPow (Norm.norm x) m) 1
      i : Nat
      hi : Membership.mem (Finset.range (HAdd.hAdd m 1)) i
      ⊢ LE.le (HPow.hPow (Norm.norm x) i) 1
    -/
  · rcases eq_or_ne m 0 with rfl | hm
    · simp only [pow_zero, le_refl,
        show i = 0 by simpa only [zero_add, Finset.range_one, Finset.mem_singleton] using hi]
      /-
        case inl.intro.inr
        R : Type u_1
        inst✝ : NormedDivisionRing R
        x : R
        m : Nat
        h : ∀ (x : R) (n : Nat), LE.le (Norm.norm (HSMul.hSMul n x)) (Norm.norm x)
        hm✝ : Eq (Max.max 1 (HPow.hPow (Norm.norm x) m)) 1
        hx : LE.le (HPow.hPow (Norm.norm x) m) 1
        i : Nat
        hi : Membership.mem (Finset.range (HAdd.hAdd m 1)) i
        hm : Ne m 0
        ⊢ LE.le (HPow.hPow (Norm.norm x) i) 1
      -/
    · rw [pow_le_one_iff_of_nonneg (norm_nonneg _) hm] at hx
      /-
        case inl.intro.inr
        R : Type u_1
        inst✝ : NormedDivisionRing R
        x : R
        m : Nat
        h : ∀ (x : R) (n : Nat), LE.le (Norm.norm (HSMul.hSMul n x)) (Norm.norm x)
        hm✝ : Eq (Max.max 1 (HPow.hPow (Norm.norm x) m)) 1
        hx : LE.le (Norm.norm x) 1
        i : Nat
        hi : Membership.mem (Finset.range (HAdd.hAdd m 1)) i
        hm : Ne m 0
        ⊢ LE.le (HPow.hPow (Norm.norm x) i) 1
      -/
      exact pow_le_one₀ (norm_nonneg _) hx
      /-
        🎉 no goals
      -/
    /-
      case inr.intro
      R : Type u_1
      inst✝ : NormedDivisionRing R
      x : R
      m : Nat
      h : ∀ (x : R) (n : Nat), LE.le (Norm.norm (HSMul.hSMul n x)) (Norm.norm x)
      hm : Eq (Max.max 1 (HPow.hPow (Norm.norm x) m)) (HPow.hPow (Norm.norm x) m)
      hx : LT.lt 1 (HPow.hPow (Norm.norm x) m)
      i : Nat
      hi : Membership.mem (Finset.range (HAdd.hAdd m 1)) i
      ⊢ LE.le (HPow.hPow (Norm.norm x) i) (HPow.hPow (Norm.norm x) m)
    -/
  · refine pow_le_pow_right₀ ?_ (by simpa only [Finset.mem_range, Nat.lt_succ] using hi)
    /-
      case inr.intro
      R : Type u_1
      inst✝ : NormedDivisionRing R
      x : R
      m : Nat
      h : ∀ (x : R) (n : Nat), LE.le (Norm.norm (HSMul.hSMul n x)) (Norm.norm x)
      hm : Eq (Max.max 1 (HPow.hPow (Norm.norm x) m)) (HPow.hPow (Norm.norm x) m)
      hx : LT.lt 1 (HPow.hPow (Norm.norm x) m)
      i : Nat
      hi : Membership.mem (Finset.range (HAdd.hAdd m 1)) i
      ⊢ LE.le 1 (Norm.norm x)
    -/
    contrapose! hx
    /-
      case inr.intro
      R : Type u_1
      inst✝ : NormedDivisionRing R
      x : R
      m : Nat
      h : ∀ (x : R) (n : Nat), LE.le (Norm.norm (HSMul.hSMul n x)) (Norm.norm x)
      hm : Eq (Max.max 1 (HPow.hPow (Norm.norm x) m)) (HPow.hPow (Norm.norm x) m)
      i : Nat
      hi : Membership.mem (Finset.range (HAdd.hAdd m 1)) i
      hx : LT.lt (Norm.norm x) 1
      ⊢ LE.le (HPow.hPow (Norm.norm x) m) 1
    -/
    exact pow_le_one₀ (norm_nonneg _) hx.le
    /-
      🎉 no goals
    -/


theorem isUltrametricDist_iff_forall_norm_natCast_le_one {R : Type*}
    [NormedDivisionRing R] : IsUltrametricDist R ↔ ∀ n : ℕ, ‖(n : R)‖ ≤ 1 :=
  ⟨fun _ => IsUltrametricDist.norm_natCast_le_one R,
      IsUltrametricDist.isUltrametricDist_of_forall_norm_natCast_le_one⟩

