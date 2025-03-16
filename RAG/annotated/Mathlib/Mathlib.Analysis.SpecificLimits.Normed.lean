theorem isLittleO_pow_pow_of_lt_left {r₁ r₂ : ℝ} (h₁ : 0 ≤ r₁) (h₂ : r₁ < r₂) :
    (fun n : ℕ ↦ r₁ ^ n) =o[atTop] fun n ↦ r₂ ^ n :=
  have H : 0 < r₂ := h₁.trans_lt h₂
  (isLittleO_of_tendsto fun _ hn ↦ False.elim <| H.ne' <| pow_eq_zero hn) <|
    (tendsto_pow_atTop_nhds_zero_of_lt_one
      (div_nonneg h₁ (h₁.trans h₂.le)) ((div_lt_one H).2 h₂)).congr fun _ ↦ div_pow _ _ _


theorem isBigO_pow_pow_of_le_left {r₁ r₂ : ℝ} (h₁ : 0 ≤ r₁) (h₂ : r₁ ≤ r₂) :
    (fun n : ℕ ↦ r₁ ^ n) =O[atTop] fun n ↦ r₂ ^ n :=
  h₂.eq_or_lt.elim (fun h ↦ h ▸ isBigO_refl _ _) fun h ↦ (isLittleO_pow_pow_of_lt_left h₁ h).isBigO


theorem isLittleO_pow_pow_of_abs_lt_left {r₁ r₂ : ℝ} (h : |r₁| < |r₂|) :
    (fun n : ℕ ↦ r₁ ^ n) =o[atTop] fun n ↦ r₂ ^ n := by
  /-
    r₁ r₂ : Real
    h : LT.lt (abs r₁) (abs r₂)
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun n => HPow.hPow r₁ n) fun n => HPow.h …
  -/
  refine (IsLittleO.of_norm_left ?_).of_norm_right
  /-
    r₁ r₂ : Real
    h : LT.lt (abs r₁) (abs r₂)
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => Norm.norm (HPow.hPow r₁ x)) fun …
  -/
  exact (isLittleO_pow_pow_of_lt_left (abs_nonneg r₁) h).congr (pow_abs r₁) (pow_abs r₂)
  /-
    🎉 no goals
  -/


open List in
/-- Various statements equivalent to the fact that `f n` grows exponentially slower than `R ^ n`.

* 0: $f n = o(a ^ n)$ for some $-R < a < R$;
* 1: $f n = o(a ^ n)$ for some $0 < a < R$;
* 2: $f n = O(a ^ n)$ for some $-R < a < R$;
* 3: $f n = O(a ^ n)$ for some $0 < a < R$;
* 4: there exist `a < R` and `C` such that one of `C` and `R` is positive and $|f n| ≤ Ca^n$
     for all `n`;
* 5: there exists `0 < a < R` and a positive `C` such that $|f n| ≤ Ca^n$ for all `n`;
* 6: there exists `a < R` such that $|f n| ≤ a ^ n$ for sufficiently large `n`;
* 7: there exists `0 < a < R` such that $|f n| ≤ a ^ n$ for sufficiently large `n`.

NB: For backwards compatibility, if you add more items to the list, please append them at the end of
the list. -/
theorem TFAE_exists_lt_isLittleO_pow (f : ℕ → ℝ) (R : ℝ) :
    TFAE
      [∃ a ∈ Ioo (-R) R, f =o[atTop] (a ^ ·), ∃ a ∈ Ioo 0 R, f =o[atTop] (a ^ ·),
        ∃ a ∈ Ioo (-R) R, f =O[atTop] (a ^ ·), ∃ a ∈ Ioo 0 R, f =O[atTop] (a ^ ·),
        ∃ a < R, ∃ C : ℝ, (0 < C ∨ 0 < R) ∧ ∀ n, |f n| ≤ C * a ^ n,
        ∃ a ∈ Ioo 0 R, ∃ C > 0, ∀ n, |f n| ≤ C * a ^ n, ∃ a < R, ∀ᶠ n in atTop, |f n| ≤ a ^ n,
        ∃ a ∈ Ioo 0 R, ∀ᶠ n in atTop, |f n| ≤ a ^ n] := by
  have A : Ico 0 R ⊆ Ioo (-R) R :=
    fun x hx ↦ ⟨(neg_lt_zero.2 (hx.1.trans_lt hx.2)).trans_le hx.1, hx.2⟩
  /-
    f : Nat → Real
    R : Real
    A : HasSubset.Subset (Set.Ico 0 R) (Set.Ioo (Neg.neg R) R)
    ⊢ (List.cons (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a) ( …
  -/
  have B : Ioo 0 R ⊆ Ioo (-R) R := Subset.trans Ioo_subset_Ico_self A
  -- First we prove that 1-4 are equivalent using 2 → 3 → 4, 1 → 3, and 2 → 1
  /-
    f : Nat → Real
    R : Real
    A : HasSubset.Subset (Set.Ico 0 R) (Set.Ioo (Neg.neg R) R)
    B : HasSubset.Subset (Set.Ioo 0 R) (Set.Ioo (Neg.neg R) R)
    ⊢ (List.cons (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a) ( …
  -/
  tfae_have 1 → 3 := fun ⟨a, ha, H⟩ ↦ ⟨a, ha, H.isBigO⟩
  /-
    f : Nat → Real
    R : Real
    A : HasSubset.Subset (Set.Ico 0 R) (Set.Ioo (Neg.neg R) R)
    B : HasSubset.Subset (Set.Ioo 0 R) (Set.Ioo (Neg.neg R) R)
    tfae_1_to_3 : (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a)  …
    ⊢ (List.cons (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a) ( …
  -/
  tfae_have 2 → 1 := fun ⟨a, ha, H⟩ ↦ ⟨a, B ha, H⟩
  tfae_have 3 → 2
  | ⟨a, ha, H⟩ => by
    rcases exists_between (abs_lt.2 ha) with ⟨b, hab, hbR⟩
    exact ⟨b, ⟨(abs_nonneg a).trans_lt hab, hbR⟩,
      H.trans_isLittleO (isLittleO_pow_pow_of_abs_lt_left (hab.trans_le (le_abs_self b)))⟩
  /-
    f : Nat → Real
    R : Real
    A : HasSubset.Subset (Set.Ico 0 R) (Set.Ioo (Neg.neg R) R)
    B : HasSubset.Subset (Set.Ioo 0 R) (Set.Ioo (Neg.neg R) R)
    tfae_1_to_3 : (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a)  …
    tfae_2_to_1 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    tfae_3_to_2 : (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a)  …
    ⊢ (List.cons (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a) ( …
  -/
  tfae_have 2 → 4 := fun ⟨a, ha, H⟩ ↦ ⟨a, ha, H.isBigO⟩
  /-
    f : Nat → Real
    R : Real
    A : HasSubset.Subset (Set.Ico 0 R) (Set.Ioo (Neg.neg R) R)
    B : HasSubset.Subset (Set.Ioo 0 R) (Set.Ioo (Neg.neg R) R)
    tfae_1_to_3 : (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a)  …
    tfae_2_to_1 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    tfae_3_to_2 : (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a)  …
    tfae_2_to_4 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    ⊢ (List.cons (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a) ( …
  -/
  tfae_have 4 → 3 := fun ⟨a, ha, H⟩ ↦ ⟨a, B ha, H⟩
  -- Add 5 and 6 using 4 → 6 → 5 → 3
  tfae_have 4 → 6
  | ⟨a, ha, H⟩ => by
    rcases bound_of_isBigO_nat_atTop H with ⟨C, hC₀, hC⟩
    refine ⟨a, ha, C, hC₀, fun n ↦ ?_⟩
    simpa only [Real.norm_eq_abs, abs_pow, abs_of_nonneg ha.1.le] using hC (pow_ne_zero n ha.1.ne')
  /-
    f : Nat → Real
    R : Real
    A : HasSubset.Subset (Set.Ico 0 R) (Set.Ioo (Neg.neg R) R)
    B : HasSubset.Subset (Set.Ioo 0 R) (Set.Ioo (Neg.neg R) R)
    tfae_1_to_3 : (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a)  …
    tfae_2_to_1 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    tfae_3_to_2 : (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a)  …
    tfae_2_to_4 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    tfae_4_to_3 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    tfae_4_to_6 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    ⊢ (List.cons (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a) ( …
  -/
  tfae_have 6 → 5 := fun ⟨a, ha, C, H₀, H⟩ ↦ ⟨a, ha.2, C, Or.inl H₀, H⟩
  tfae_have 5 → 3
  | ⟨a, ha, C, h₀, H⟩ => by
    rcases sign_cases_of_C_mul_pow_nonneg fun n ↦ (abs_nonneg _).trans (H n) with (rfl | ⟨hC₀, ha₀⟩)
    · obtain rfl : f = 0 := by
        ext n
        simpa using H n
      simp only [lt_irrefl, false_or] at h₀
      exact ⟨0, ⟨neg_lt_zero.2 h₀, h₀⟩, isBigO_zero _ _⟩
    exact ⟨a, A ⟨ha₀, ha⟩,
      isBigO_of_le' _ fun n ↦ (H n).trans <| mul_le_mul_of_nonneg_left (le_abs_self _) hC₀.le⟩
  -- Add 7 and 8 using 2 → 8 → 7 → 3
  tfae_have 2 → 8
  | ⟨a, ha, H⟩ => by
    refine ⟨a, ha, (H.def zero_lt_one).mono fun n hn ↦ ?_⟩
    rwa [Real.norm_eq_abs, Real.norm_eq_abs, one_mul, abs_pow, abs_of_pos ha.1] at hn
  /-
    f : Nat → Real
    R : Real
    A : HasSubset.Subset (Set.Ico 0 R) (Set.Ioo (Neg.neg R) R)
    B : HasSubset.Subset (Set.Ioo 0 R) (Set.Ioo (Neg.neg R) R)
    tfae_1_to_3 : (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a)  …
    tfae_2_to_1 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    tfae_3_to_2 : (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a)  …
    tfae_2_to_4 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    tfae_4_to_3 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    tfae_4_to_6 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    tfae_6_to_5 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Exists fu …
    tfae_5_to_3 : (Exists fun a => And (LT.lt a R) (Exists fun C => And (Or (LT.lt …
    tfae_2_to_8 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    ⊢ (List.cons (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a) ( …
  -/
  tfae_have 8 → 7 := fun ⟨a, ha, H⟩ ↦ ⟨a, ha.2, H⟩
  tfae_have 7 → 3
  | ⟨a, ha, H⟩ => by
    have : 0 ≤ a := nonneg_of_eventually_pow_nonneg (H.mono fun n ↦ (abs_nonneg _).trans)
    refine ⟨a, A ⟨this, ha⟩, IsBigO.of_bound 1 ?_⟩
    simpa only [Real.norm_eq_abs, one_mul, abs_pow, abs_of_nonneg this]
  /-
    f : Nat → Real
    R : Real
    A : HasSubset.Subset (Set.Ico 0 R) (Set.Ioo (Neg.neg R) R)
    B : HasSubset.Subset (Set.Ioo 0 R) (Set.Ioo (Neg.neg R) R)
    tfae_1_to_3 : (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a)  …
    tfae_2_to_1 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    tfae_3_to_2 : (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a)  …
    tfae_2_to_4 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    tfae_4_to_3 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    tfae_4_to_6 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    tfae_6_to_5 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Exists fu …
    tfae_5_to_3 : (Exists fun a => And (LT.lt a R) (Exists fun C => And (Or (LT.lt …
    tfae_2_to_8 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Asymptoti …
    tfae_8_to_7 : (Exists fun a => And (Membership.mem (Set.Ioo 0 R) a) (Filter.Ev …
    tfae_7_to_3 : (Exists fun a => And (LT.lt a R) (Filter.Eventually (fun n => LE …
    ⊢ (List.cons (Exists fun a => And (Membership.mem (Set.Ioo (Neg.neg R) R) a) ( …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


/-- For any natural `k` and a real `r > 1` we have `n ^ k = o(r ^ n)` as `n → ∞`. -/
theorem isLittleO_pow_const_const_pow_of_one_lt {R : Type*} [NormedRing R] (k : ℕ) {r : ℝ}
    (hr : 1 < r) : (fun n ↦ (n : R) ^ k : ℕ → R) =o[atTop] fun n ↦ r ^ n := by
  have : Tendsto (fun x : ℝ ↦ x ^ k) (𝓝[>] 1) (𝓝 1) :=
    ((continuous_id.pow k).tendsto' (1 : ℝ) 1 (one_pow _)).mono_left inf_le_left
  obtain ⟨r' : ℝ, hr' : r' ^ k < r, h1 : 1 < r'⟩ :=
    ((this.eventually (gt_mem_nhds hr)).and self_mem_nhdsWithin).exists
  /-
    case intro.intro
    R : Type u_2
    inst✝ : NormedRing R
    k : Nat
    r : Real
    hr : LT.lt 1 r
    this : Filter.Tendsto (fun x => HPow.hPow x k) (nhdsWithin 1 (Set.Ioi 1)) (nhd …
    r' : Real
    hr' : LT.lt (HPow.hPow r' k) r
    h1 : LT.lt 1 r'
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun n => HPow.hPow (↑n) k) fun n => HPow …
  -/
  have h0 : 0 ≤ r' := zero_le_one.trans h1.le
  suffices (fun n ↦ (n : R) ^ k : ℕ → R) =O[atTop] fun n : ℕ ↦ (r' ^ k) ^ n from
    this.trans_isLittleO (isLittleO_pow_pow_of_lt_left (pow_nonneg h0 _) hr')
  /-
    case intro.intro
    R : Type u_2
    inst✝ : NormedRing R
    k : Nat
    r : Real
    hr : LT.lt 1 r
    this : Filter.Tendsto (fun x => HPow.hPow x k) (nhdsWithin 1 (Set.Ioi 1)) (nhd …
    r' : Real
    hr' : LT.lt (HPow.hPow r' k) r
    h1 : LT.lt 1 r'
    h0 : LE.le 0 r'
    ⊢ Asymptotics.IsBigO Filter.atTop (fun n => HPow.hPow (↑n) k) fun n => HPow.hP …
  -/
  conv in (r' ^ _) ^ _ => rw [← pow_mul, mul_comm, pow_mul]
  suffices ∀ n : ℕ, ‖(n : R)‖ ≤ (r' - 1)⁻¹ * ‖(1 : R)‖ * ‖r' ^ n‖ from
    (isBigO_of_le' _ this).pow _
  /-
    case intro.intro
    R : Type u_2
    inst✝ : NormedRing R
    k : Nat
    r : Real
    hr : LT.lt 1 r
    this : Filter.Tendsto (fun x => HPow.hPow x k) (nhdsWithin 1 (Set.Ioi 1)) (nhd …
    r' : Real
    hr' : LT.lt (HPow.hPow r' k) r
    h1 : LT.lt 1 r'
    h0 : LE.le 0 r'
    ⊢ ∀ (n : Nat), LE.le (Norm.norm ↑n) (HMul.hMul (HMul.hMul (Inv.inv (HSub.hSub  …
  -/
  intro n
  /-
    case intro.intro
    R : Type u_2
    inst✝ : NormedRing R
    k : Nat
    r : Real
    hr : LT.lt 1 r
    this : Filter.Tendsto (fun x => HPow.hPow x k) (nhdsWithin 1 (Set.Ioi 1)) (nhd …
    r' : Real
    hr' : LT.lt (HPow.hPow r' k) r
    h1 : LT.lt 1 r'
    h0 : LE.le 0 r'
    n : Nat
    ⊢ LE.le (Norm.norm ↑n) (HMul.hMul (HMul.hMul (Inv.inv (HSub.hSub r' 1)) (Norm. …
  -/
  rw [mul_right_comm]
  /-
    case intro.intro
    R : Type u_2
    inst✝ : NormedRing R
    k : Nat
    r : Real
    hr : LT.lt 1 r
    this : Filter.Tendsto (fun x => HPow.hPow x k) (nhdsWithin 1 (Set.Ioi 1)) (nhd …
    r' : Real
    hr' : LT.lt (HPow.hPow r' k) r
    h1 : LT.lt 1 r'
    h0 : LE.le 0 r'
    n : Nat
    ⊢ LE.le (Norm.norm ↑n) (HMul.hMul (HMul.hMul (Inv.inv (HSub.hSub r' 1)) (Norm. …
  -/
  refine n.norm_cast_le.trans (mul_le_mul_of_nonneg_right ?_ (norm_nonneg _))
  /-
    case intro.intro
    R : Type u_2
    inst✝ : NormedRing R
    k : Nat
    r : Real
    hr : LT.lt 1 r
    this : Filter.Tendsto (fun x => HPow.hPow x k) (nhdsWithin 1 (Set.Ioi 1)) (nhd …
    r' : Real
    hr' : LT.lt (HPow.hPow r' k) r
    h1 : LT.lt 1 r'
    h0 : LE.le 0 r'
    n : Nat
    ⊢ LE.le (↑n) (HMul.hMul (Inv.inv (HSub.hSub r' 1)) (Norm.norm (HPow.hPow r' n)))
  -/
  simpa [_root_.div_eq_inv_mul, Real.norm_eq_abs, abs_of_nonneg h0] using n.cast_le_pow_div_sub h1
  /-
    🎉 no goals
  -/


/-- For a real `r > 1` we have `n = o(r ^ n)` as `n → ∞`. -/
theorem isLittleO_coe_const_pow_of_one_lt {R : Type*} [NormedRing R] {r : ℝ} (hr : 1 < r) :
    ((↑) : ℕ → R) =o[atTop] fun n ↦ r ^ n := by
  /-
    R : Type u_2
    inst✝ : NormedRing R
    r : Real
    hr : LT.lt 1 r
    ⊢ Asymptotics.IsLittleO Filter.atTop Nat.cast fun n => HPow.hPow r n
  -/
  simpa only [pow_one] using @isLittleO_pow_const_const_pow_of_one_lt R _ 1 _ hr
  /-
    🎉 no goals
  -/


/-- If `‖r₁‖ < r₂`, then for any natural `k` we have `n ^ k r₁ ^ n = o (r₂ ^ n)` as `n → ∞`. -/
theorem isLittleO_pow_const_mul_const_pow_const_pow_of_norm_lt {R : Type*} [NormedRing R] (k : ℕ)
    {r₁ : R} {r₂ : ℝ} (h : ‖r₁‖ < r₂) :
    (fun n ↦ (n : R) ^ k * r₁ ^ n : ℕ → R) =o[atTop] fun n ↦ r₂ ^ n := by
  /-
    R : Type u_2
    inst✝ : NormedRing R
    k : Nat
    r₁ : R
    r₂ : Real
    h : LT.lt (Norm.norm r₁) r₂
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun n => HMul.hMul (HPow.hPow (↑n) k) (H …
  -/
  by_cases h0 : r₁ = 0
    /-
      case pos
      R : Type u_2
      inst✝ : NormedRing R
      k : Nat
      r₁ : R
      r₂ : Real
      h : LT.lt (Norm.norm r₁) r₂
      h0 : Eq r₁ 0
      ⊢ Asymptotics.IsLittleO Filter.atTop (fun n => HMul.hMul (HPow.hPow (↑n) k) (H …
    -/
  · refine (isLittleO_zero _ _).congr' (mem_atTop_sets.2 <| ⟨1, fun n hn ↦ ?_⟩) EventuallyEq.rfl
    /-
      case pos
      R : Type u_2
      inst✝ : NormedRing R
      k : Nat
      r₁ : R
      r₂ : Real
      h : LT.lt (Norm.norm r₁) r₂
      h0 : Eq r₁ 0
      n : Nat
      hn : GE.ge n 1
      ⊢ Membership.mem (setOf fun x => (fun x => Eq ((fun _x => 0) x) ((fun n => HMu …
    -/
    simp [zero_pow (one_le_iff_ne_zero.1 hn), h0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_2
    inst✝ : NormedRing R
    k : Nat
    r₁ : R
    r₂ : Real
    h : LT.lt (Norm.norm r₁) r₂
    h0 : Not (Eq r₁ 0)
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun n => HMul.hMul (HPow.hPow (↑n) k) (H …
  -/
  rw [← Ne, ← norm_pos_iff] at h0
  have A : (fun n ↦ (n : R) ^ k : ℕ → R) =o[atTop] fun n ↦ (r₂ / ‖r₁‖) ^ n :=
    isLittleO_pow_const_const_pow_of_one_lt k ((one_lt_div h0).2 h)
  suffices (fun n ↦ r₁ ^ n) =O[atTop] fun n ↦ ‖r₁‖ ^ n by
    simpa [div_mul_cancel₀ _ (pow_pos h0 _).ne', div_pow] using A.mul_isBigO this
  /-
    case neg
    R : Type u_2
    inst✝ : NormedRing R
    k : Nat
    r₁ : R
    r₂ : Real
    h : LT.lt (Norm.norm r₁) r₂
    h0 : LT.lt 0 (Norm.norm r₁)
    A : Asymptotics.IsLittleO Filter.atTop (fun n => HPow.hPow (↑n) k) fun n => HP …
    ⊢ Asymptotics.IsBigO Filter.atTop (fun n => HPow.hPow r₁ n) fun n => HPow.hPow …
  -/
  exact IsBigO.of_bound 1 (by simpa using eventually_norm_pow_le r₁)
  /-
    🎉 no goals
  -/


theorem tendsto_pow_const_div_const_pow_of_one_lt (k : ℕ) {r : ℝ} (hr : 1 < r) :
    Tendsto (fun n ↦ (n : ℝ) ^ k / r ^ n : ℕ → ℝ) atTop (𝓝 0) :=
  (isLittleO_pow_const_const_pow_of_one_lt k hr).tendsto_div_nhds_zero


/-- If `|r| < 1`, then `n ^ k r ^ n` tends to zero for any natural `k`. -/
theorem tendsto_pow_const_mul_const_pow_of_abs_lt_one (k : ℕ) {r : ℝ} (hr : |r| < 1) :
    Tendsto (fun n ↦ (n : ℝ) ^ k * r ^ n : ℕ → ℝ) atTop (𝓝 0) := by
  /-
    k : Nat
    r : Real
    hr : LT.lt (abs r) 1
    ⊢ Filter.Tendsto (fun n => HMul.hMul (HPow.hPow (↑n) k) (HPow.hPow r n)) Filte …
  -/
  by_cases h0 : r = 0
  · exact tendsto_const_nhds.congr'
      (mem_atTop_sets.2 ⟨1, fun n hn ↦ by simp [zero_lt_one.trans_le hn |>.ne', h0]⟩)
  /-
    case neg
    k : Nat
    r : Real
    hr : LT.lt (abs r) 1
    h0 : Not (Eq r 0)
    ⊢ Filter.Tendsto (fun n => HMul.hMul (HPow.hPow (↑n) k) (HPow.hPow r n)) Filte …
  -/
  have hr' : 1 < |r|⁻¹ := (one_lt_inv₀ (abs_pos.2 h0)).2 hr
  /-
    case neg
    k : Nat
    r : Real
    hr : LT.lt (abs r) 1
    h0 : Not (Eq r 0)
    hr' : LT.lt 1 (Inv.inv (abs r))
    ⊢ Filter.Tendsto (fun n => HMul.hMul (HPow.hPow (↑n) k) (HPow.hPow r n)) Filte …
  -/
  rw [tendsto_zero_iff_norm_tendsto_zero]
  /-
    case neg
    k : Nat
    r : Real
    hr : LT.lt (abs r) 1
    h0 : Not (Eq r 0)
    hr' : LT.lt 1 (Inv.inv (abs r))
    ⊢ Filter.Tendsto (fun x => Norm.norm (HMul.hMul (HPow.hPow (↑x) k) (HPow.hPow  …
  -/
  simpa [div_eq_mul_inv] using tendsto_pow_const_div_const_pow_of_one_lt k hr'
  /-
    🎉 no goals
  -/


/--For `k ≠ 0` and a constant `r` the function `r / n ^ k` tends to zero. -/
lemma tendsto_const_div_pow (r : ℝ) (k : ℕ) (hk : k ≠ 0) :
    Tendsto (fun n : ℕ => r / n ^ k) atTop (𝓝 0) := by
  simpa using Filter.Tendsto.const_div_atTop (tendsto_natCast_atTop_atTop (R := ℝ).comp
    (tendsto_pow_atTop hk) ) r


/-- If `0 ≤ r < 1`, then `n ^ k r ^ n` tends to zero for any natural `k`.
This is a specialized version of `tendsto_pow_const_mul_const_pow_of_abs_lt_one`, singled out
for ease of application. -/
theorem tendsto_pow_const_mul_const_pow_of_lt_one (k : ℕ) {r : ℝ} (hr : 0 ≤ r) (h'r : r < 1) :
    Tendsto (fun n ↦ (n : ℝ) ^ k * r ^ n : ℕ → ℝ) atTop (𝓝 0) :=
  tendsto_pow_const_mul_const_pow_of_abs_lt_one k (abs_lt.2 ⟨neg_one_lt_zero.trans_le hr, h'r⟩)


/-- If `|r| < 1`, then `n * r ^ n` tends to zero. -/
theorem tendsto_self_mul_const_pow_of_abs_lt_one {r : ℝ} (hr : |r| < 1) :
    Tendsto (fun n ↦ n * r ^ n : ℕ → ℝ) atTop (𝓝 0) := by
  /-
    r : Real
    hr : LT.lt (abs r) 1
    ⊢ Filter.Tendsto (fun n => HMul.hMul (↑n) (HPow.hPow r n)) Filter.atTop (nhds 0)
  -/
  simpa only [pow_one] using tendsto_pow_const_mul_const_pow_of_abs_lt_one 1 hr
  /-
    🎉 no goals
  -/


/-- If `0 ≤ r < 1`, then `n * r ^ n` tends to zero. This is a specialized version of
`tendsto_self_mul_const_pow_of_abs_lt_one`, singled out for ease of application. -/
theorem tendsto_self_mul_const_pow_of_lt_one {r : ℝ} (hr : 0 ≤ r) (h'r : r < 1) :
    Tendsto (fun n ↦ n * r ^ n : ℕ → ℝ) atTop (𝓝 0) := by
  /-
    r : Real
    hr : LE.le 0 r
    h'r : LT.lt r 1
    ⊢ Filter.Tendsto (fun n => HMul.hMul (↑n) (HPow.hPow r n)) Filter.atTop (nhds 0)
  -/
  simpa only [pow_one] using tendsto_pow_const_mul_const_pow_of_lt_one 1 hr h'r
  /-
    🎉 no goals
  -/


/-- In a normed ring, the powers of an element x with `‖x‖ < 1` tend to zero. -/
theorem tendsto_pow_atTop_nhds_zero_of_norm_lt_one {R : Type*} [NormedRing R] {x : R}
    (h : ‖x‖ < 1) :
    Tendsto (fun n : ℕ ↦ x ^ n) atTop (𝓝 0) := by
  /-
    R : Type u_2
    inst✝ : NormedRing R
    x : R
    h : LT.lt (Norm.norm x) 1
    ⊢ Filter.Tendsto (fun n => HPow.hPow x n) Filter.atTop (nhds 0)
  -/
  apply squeeze_zero_norm' (eventually_norm_pow_le x)
  /-
    R : Type u_2
    inst✝ : NormedRing R
    x : R
    h : LT.lt (Norm.norm x) 1
    ⊢ Filter.Tendsto (HPow.hPow (Norm.norm x)) Filter.atTop (nhds 0)
  -/
  exact tendsto_pow_atTop_nhds_zero_of_lt_one (norm_nonneg _) h
  /-
    🎉 no goals
  -/


theorem tendsto_pow_atTop_nhds_zero_of_abs_lt_one {r : ℝ} (h : |r| < 1) :
    Tendsto (fun n : ℕ ↦ r ^ n) atTop (𝓝 0) :=
  tendsto_pow_atTop_nhds_zero_of_norm_lt_one h


/-- A normed ring has summable geometric series if, for all `ξ` of norm `< 1`, the geometric series
`∑ ξ ^ n` converges. This holds both in complete normed rings and in normed fields, providing a
convenient abstraction of these two classes to avoid repeating the same proofs. -/
class HasSummableGeomSeries (K : Type*) [NormedRing K] : Prop where
  summable_geometric_of_norm_lt_one : ∀ (ξ : K), ‖ξ‖ < 1 → Summable (fun n ↦ ξ ^ n)


lemma summable_geometric_of_norm_lt_one {K : Type*} [NormedRing K] [HasSummableGeomSeries K]
    {x : K} (h : ‖x‖ < 1) : Summable (fun n ↦ x ^ n) :=
  HasSummableGeomSeries.summable_geometric_of_norm_lt_one x h


instance {R : Type*} [NormedRing R] [CompleteSpace R] : HasSummableGeomSeries R := by
  /-
    α : Type u_1
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : CompleteSpace R
    ⊢ HasSummableGeomSeries R
  -/
  constructor
  /-
    case summable_geometric_of_norm_lt_one
    α : Type u_1
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : CompleteSpace R
    ⊢ ∀ (ξ : R), LT.lt (Norm.norm ξ) 1 → Summable fun n => HPow.hPow ξ n
  -/
  intro x hx
  /-
    case summable_geometric_of_norm_lt_one
    α : Type u_1
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : CompleteSpace R
    x : R
    hx : LT.lt (Norm.norm x) 1
    ⊢ Summable fun n => HPow.hPow x n
  -/
  have h1 : Summable fun n : ℕ ↦ ‖x‖ ^ n := summable_geometric_of_lt_one (norm_nonneg _) hx
  /-
    case summable_geometric_of_norm_lt_one
    α : Type u_1
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : CompleteSpace R
    x : R
    hx : LT.lt (Norm.norm x) 1
    h1 : Summable fun n => HPow.hPow (Norm.norm x) n
    ⊢ Summable fun n => HPow.hPow x n
  -/
  exact h1.of_norm_bounded_eventually_nat _ (eventually_norm_pow_le x)
  /-
    🎉 no goals
  -/


/-- Bound for the sum of a geometric series in a normed ring. This formula does not assume that the
normed ring satisfies the axiom `‖1‖ = 1`. -/
theorem tsum_geometric_le_of_norm_lt_one (x : R) (h : ‖x‖ < 1) :
    ‖∑' n : ℕ, x ^ n‖ ≤ ‖(1 : R)‖ - 1 + (1 - ‖x‖)⁻¹ := by
  /-
    R : Type u_2
    inst✝ : NormedRing R
    x : R
    h : LT.lt (Norm.norm x) 1
    ⊢ LE.le (Norm.norm (tsum fun n => HPow.hPow x n)) (HAdd.hAdd (HSub.hSub (Norm. …
  -/
  by_cases hx : Summable (fun n ↦ x ^ n)
    /-
      case pos
      R : Type u_2
      inst✝ : NormedRing R
      x : R
      h : LT.lt (Norm.norm x) 1
      hx : Summable fun n => HPow.hPow x n
      ⊢ LE.le (Norm.norm (tsum fun n => HPow.hPow x n)) (HAdd.hAdd (HSub.hSub (Norm. …
    -/
  · rw [tsum_eq_zero_add hx]
    /-
      case pos
      R : Type u_2
      inst✝ : NormedRing R
      x : R
      h : LT.lt (Norm.norm x) 1
      hx : Summable fun n => HPow.hPow x n
      ⊢ LE.le (Norm.norm (HAdd.hAdd (HPow.hPow x 0) (tsum fun b => HPow.hPow x (HAdd …
    -/
    simp only [_root_.pow_zero]
    /-
      case pos
      R : Type u_2
      inst✝ : NormedRing R
      x : R
      h : LT.lt (Norm.norm x) 1
      hx : Summable fun n => HPow.hPow x n
      ⊢ LE.le (Norm.norm (HAdd.hAdd 1 (tsum fun b => HPow.hPow x (HAdd.hAdd b 1))))  …
    -/
    refine le_trans (norm_add_le _ _) ?_
    have : ‖∑' b : ℕ, (fun n ↦ x ^ (n + 1)) b‖ ≤ (1 - ‖x‖)⁻¹ - 1 := by
      refine tsum_of_norm_bounded ?_ fun b ↦ norm_pow_le' _ (Nat.succ_pos b)
      convert (hasSum_nat_add_iff' 1).mpr (hasSum_geometric_of_lt_one (norm_nonneg x) h)
      simp
    /-
      case pos
      R : Type u_2
      inst✝ : NormedRing R
      x : R
      h : LT.lt (Norm.norm x) 1
      hx : Summable fun n => HPow.hPow x n
      this : LE.le (Norm.norm (tsum fun b => (fun n => HPow.hPow x (HAdd.hAdd n 1))  …
      ⊢ LE.le (HAdd.hAdd (Norm.norm 1) (Norm.norm (tsum fun b => HPow.hPow x (HAdd.h …
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_2
      inst✝ : NormedRing R
      x : R
      h : LT.lt (Norm.norm x) 1
      hx : Not (Summable fun n => HPow.hPow x n)
      ⊢ LE.le (Norm.norm (tsum fun n => HPow.hPow x n)) (HAdd.hAdd (HSub.hSub (Norm. …
    -/
  · simp [tsum_eq_zero_of_not_summable hx]
    /-
      case neg
      R : Type u_2
      inst✝ : NormedRing R
      x : R
      h : LT.lt (Norm.norm x) 1
      hx : Not (Summable fun n => HPow.hPow x n)
      ⊢ LE.le 0 (HAdd.hAdd (HSub.hSub (Norm.norm 1) 1) (Inv.inv (HSub.hSub 1 (Norm.n …
    -/
    nontriviality R
    /-
      R : Type u_2
      inst✝ : NormedRing R
      x : R
      h : LT.lt (Norm.norm x) 1
      hx : Not (Summable fun n => HPow.hPow x n)
      a✝ : Nontrivial R
      ⊢ LE.le 0 (HAdd.hAdd (HSub.hSub (Norm.norm 1) 1) (Inv.inv (HSub.hSub 1 (Norm.n …
    -/
    have : 1 ≤ ‖(1 : R)‖ := one_le_norm_one R
    /-
      R : Type u_2
      inst✝ : NormedRing R
      x : R
      h : LT.lt (Norm.norm x) 1
      hx : Not (Summable fun n => HPow.hPow x n)
      a✝ : Nontrivial R
      this : LE.le 1 (Norm.norm 1)
      ⊢ LE.le 0 (HAdd.hAdd (HSub.hSub (Norm.norm 1) 1) (Inv.inv (HSub.hSub 1 (Norm.n …
    -/
    have : 0 ≤ (1 - ‖x‖) ⁻¹ := inv_nonneg.2 (by linarith)
    /-
      R : Type u_2
      inst✝ : NormedRing R
      x : R
      h : LT.lt (Norm.norm x) 1
      hx : Not (Summable fun n => HPow.hPow x n)
      a✝ : Nontrivial R
      this✝ : LE.le 1 (Norm.norm 1)
      this : LE.le 0 (Inv.inv (HSub.hSub 1 (Norm.norm x)))
      ⊢ LE.le 0 (HAdd.hAdd (HSub.hSub (Norm.norm 1) 1) (Inv.inv (HSub.hSub 1 (Norm.n …
    -/
    linarith
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias NormedRing.tsum_geometric_of_norm_lt_one := tsum_geometric_le_of_norm_lt_one


theorem geom_series_mul_neg (x : R) (h : ‖x‖ < 1) : (∑' i : ℕ, x ^ i) * (1 - x) = 1 := by
  /-
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    ⊢ Eq (HMul.hMul (tsum fun i => HPow.hPow x i) (HSub.hSub 1 x)) 1
  -/
  have := (summable_geometric_of_norm_lt_one h).hasSum.mul_right (1 - x)
  /-
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    this : HasSum (fun i => HMul.hMul (HPow.hPow x i) (HSub.hSub 1 x)) (HMul.hMul  …
    ⊢ Eq (HMul.hMul (tsum fun i => HPow.hPow x i) (HSub.hSub 1 x)) 1
  -/
  refine tendsto_nhds_unique this.tendsto_sum_nat ?_
  have : Tendsto (fun n : ℕ ↦ 1 - x ^ n) atTop (𝓝 1) := by
    simpa using tendsto_const_nhds.sub (tendsto_pow_atTop_nhds_zero_of_norm_lt_one h)
  /-
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    this✝ : HasSum (fun i => HMul.hMul (HPow.hPow x i) (HSub.hSub 1 x)) (HMul.hMul …
    this : Filter.Tendsto (fun n => HSub.hSub 1 (HPow.hPow x n)) Filter.atTop (nhd …
    ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun i => HMul.hMul (HPow.hPow  …
  -/
  convert← this
  /-
    case h.e'_3.h
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    this✝ : HasSum (fun i => HMul.hMul (HPow.hPow x i) (HSub.hSub 1 x)) (HMul.hMul …
    this : Filter.Tendsto (fun n => HSub.hSub 1 (HPow.hPow x n)) Filter.atTop (nhd …
    x✝ : Nat
    ⊢ Eq (HSub.hSub 1 (HPow.hPow x x✝)) ((Finset.range x✝).sum fun i => HMul.hMul  …
  -/
  rw [← geom_sum_mul_neg, Finset.sum_mul]
  /-
    🎉 no goals
  -/


theorem mul_neg_geom_series (x : R) (h : ‖x‖ < 1) : (1 - x) * ∑' i : ℕ, x ^ i = 1 := by
  /-
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    ⊢ Eq (HMul.hMul (HSub.hSub 1 x) (tsum fun i => HPow.hPow x i)) 1
  -/
  have := (summable_geometric_of_norm_lt_one h).hasSum.mul_left (1 - x)
  /-
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    this : HasSum (fun i => HMul.hMul (HSub.hSub 1 x) (HPow.hPow x i)) (HMul.hMul  …
    ⊢ Eq (HMul.hMul (HSub.hSub 1 x) (tsum fun i => HPow.hPow x i)) 1
  -/
  refine tendsto_nhds_unique this.tendsto_sum_nat ?_
  have : Tendsto (fun n : ℕ ↦ 1 - x ^ n) atTop (𝓝 1) := by
    simpa using tendsto_const_nhds.sub (tendsto_pow_atTop_nhds_zero_of_norm_lt_one h)
  /-
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    this✝ : HasSum (fun i => HMul.hMul (HSub.hSub 1 x) (HPow.hPow x i)) (HMul.hMul …
    this : Filter.Tendsto (fun n => HSub.hSub 1 (HPow.hPow x n)) Filter.atTop (nhd …
    ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun i => HMul.hMul (HSub.hSub  …
  -/
  convert← this
  /-
    case h.e'_3.h
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    this✝ : HasSum (fun i => HMul.hMul (HSub.hSub 1 x) (HPow.hPow x i)) (HMul.hMul …
    this : Filter.Tendsto (fun n => HSub.hSub 1 (HPow.hPow x n)) Filter.atTop (nhd …
    x✝ : Nat
    ⊢ Eq (HSub.hSub 1 (HPow.hPow x x✝)) ((Finset.range x✝).sum fun i => HMul.hMul  …
  -/
  rw [← mul_neg_geom_sum, Finset.mul_sum]
  /-
    🎉 no goals
  -/


theorem geom_series_succ (x : R) (h : ‖x‖ < 1) : ∑' i : ℕ, x ^ (i + 1) = ∑' i : ℕ, x ^ i - 1 := by
  rw [eq_sub_iff_add_eq, tsum_eq_zero_add (summable_geometric_of_norm_lt_one h),
    pow_zero, add_comm]


theorem geom_series_mul_shift (x : R) (h : ‖x‖ < 1) :
    x * ∑' i : ℕ, x ^ i = ∑' i : ℕ, x ^ (i + 1) := by
  /-
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    ⊢ Eq (HMul.hMul x (tsum fun i => HPow.hPow x i)) (tsum fun i => HPow.hPow x (H …
  -/
  simp_rw [← (summable_geometric_of_norm_lt_one h).tsum_mul_left, ← _root_.pow_succ']
  /-
    🎉 no goals
  -/


theorem geom_series_mul_one_add (x : R) (h : ‖x‖ < 1) :
    (1 + x) * ∑' i : ℕ, x ^ i = 2 * ∑' i : ℕ, x ^ i - 1 := by
  /-
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    ⊢ Eq (HMul.hMul (HAdd.hAdd 1 x) (tsum fun i => HPow.hPow x i)) (HSub.hSub (HMu …
  -/
  rw [add_mul, one_mul, geom_series_mul_shift x h, geom_series_succ x h, two_mul, add_sub_assoc]
  /-
    🎉 no goals
  -/


/-- In a normed ring with summable geometric series, a perturbation of `1` by an element `t`
of distance less than `1` from `1` is a unit.  Here we construct its `Units` structure. -/
@[simps val]
def Units.oneSub (t : R) (h : ‖t‖ < 1) : Rˣ where
  val := 1 - t
  inv := ∑' n : ℕ, t ^ n
  val_inv := mul_neg_geom_series t h
  inv_val := geom_series_mul_neg t h


theorem geom_series_eq_inverse (x : R) (h : ‖x‖ < 1) :
    ∑' i, x ^ i = Ring.inverse (1 - x) := by
  /-
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    ⊢ Eq (tsum fun i => HPow.hPow x i) (Ring.inverse (HSub.hSub 1 x))
  -/
  change (Units.oneSub x h) ⁻¹ = Ring.inverse (1 - x)
  /-
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    ⊢ Eq (↑(Inv.inv (Units.oneSub x h))) (Ring.inverse (HSub.hSub 1 x))
  -/
  rw [← Ring.inverse_unit]
  /-
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    ⊢ Eq (Ring.inverse ↑(Units.oneSub x h)) (Ring.inverse (HSub.hSub 1 x))
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem hasSum_geom_series_inverse (x : R) (h : ‖x‖ < 1) :
    HasSum (fun i ↦ x ^ i) (Ring.inverse (1 - x)) := by
  /-
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    ⊢ HasSum (fun i => HPow.hPow x i) (Ring.inverse (HSub.hSub 1 x))
  -/
  convert (summable_geometric_of_norm_lt_one h).hasSum
  /-
    case h.e'_6
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    ⊢ Eq (Ring.inverse (HSub.hSub 1 x)) (tsum fun b => HPow.hPow x b)
  -/
  exact (geom_series_eq_inverse x h).symm
  /-
    🎉 no goals
  -/


lemma isUnit_one_sub_of_norm_lt_one {x : R} (h : ‖x‖ < 1) : IsUnit (1 - x) :=
  ⟨Units.oneSub x h, rfl⟩


@[deprecated (since := "2024-07-27")]
alias NormedRing.summable_geometric_of_norm_lt_one := summable_geometric_of_norm_lt_one


theorem hasSum_geometric_of_norm_lt_one (h : ‖ξ‖ < 1) : HasSum (fun n : ℕ ↦ ξ ^ n) (1 - ξ)⁻¹ := by
  have xi_ne_one : ξ ≠ 1 := by
    contrapose! h
    simp [h]
  have A : Tendsto (fun n ↦ (ξ ^ n - 1) * (ξ - 1)⁻¹) atTop (𝓝 ((0 - 1) * (ξ - 1)⁻¹)) :=
    ((tendsto_pow_atTop_nhds_zero_of_norm_lt_one h).sub tendsto_const_nhds).mul tendsto_const_nhds
  /-
    K : Type u_2
    inst✝ : NormedDivisionRing K
    ξ : K
    h : LT.lt (Norm.norm ξ) 1
    xi_ne_one : Ne ξ 1
    A : Filter.Tendsto (fun n => HMul.hMul (HSub.hSub (HPow.hPow ξ n) 1) (Inv.inv  …
    ⊢ HasSum (fun n => HPow.hPow ξ n) (Inv.inv (HSub.hSub 1 ξ))
  -/
  rw [hasSum_iff_tendsto_nat_of_summable_norm]
    /-
      K : Type u_2
      inst✝ : NormedDivisionRing K
      ξ : K
      h : LT.lt (Norm.norm ξ) 1
      xi_ne_one : Ne ξ 1
      A : Filter.Tendsto (fun n => HMul.hMul (HSub.hSub (HPow.hPow ξ n) 1) (Inv.inv  …
      ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun i => HPow.hPow ξ i) Filter …
    -/
  · simpa [geom_sum_eq, xi_ne_one, neg_inv, div_eq_mul_inv] using A
    /-
      🎉 no goals
    -/
    /-
      K : Type u_2
      inst✝ : NormedDivisionRing K
      ξ : K
      h : LT.lt (Norm.norm ξ) 1
      xi_ne_one : Ne ξ 1
      A : Filter.Tendsto (fun n => HMul.hMul (HSub.hSub (HPow.hPow ξ n) 1) (Inv.inv  …
      ⊢ Summable fun i => Norm.norm (HPow.hPow ξ i)
    -/
  · simp [norm_pow, summable_geometric_of_lt_one (norm_nonneg _) h]
    /-
      🎉 no goals
    -/


instance : HasSummableGeomSeries K :=
  ⟨fun _ h ↦ (hasSum_geometric_of_norm_lt_one h).summable⟩


theorem tsum_geometric_of_norm_lt_one (h : ‖ξ‖ < 1) : ∑' n : ℕ, ξ ^ n = (1 - ξ)⁻¹ :=
  (hasSum_geometric_of_norm_lt_one h).tsum_eq


theorem hasSum_geometric_of_abs_lt_one {r : ℝ} (h : |r| < 1) :
    HasSum (fun n : ℕ ↦ r ^ n) (1 - r)⁻¹ :=
  hasSum_geometric_of_norm_lt_one h


theorem summable_geometric_of_abs_lt_one {r : ℝ} (h : |r| < 1) : Summable fun n : ℕ ↦ r ^ n :=
  summable_geometric_of_norm_lt_one h


theorem tsum_geometric_of_abs_lt_one {r : ℝ} (h : |r| < 1) : ∑' n : ℕ, r ^ n = (1 - r)⁻¹ :=
  tsum_geometric_of_norm_lt_one h


/-- A geometric series in a normed field is summable iff the norm of the common ratio is less than
one. -/
@[simp]
theorem summable_geometric_iff_norm_lt_one : (Summable fun n : ℕ ↦ ξ ^ n) ↔ ‖ξ‖ < 1 := by
  /-
    K : Type u_2
    inst✝ : NormedDivisionRing K
    ξ : K
    ⊢ Iff (Summable fun n => HPow.hPow ξ n) (LT.lt (Norm.norm ξ) 1)
  -/
  refine ⟨fun h ↦ ?_, summable_geometric_of_norm_lt_one⟩
  obtain ⟨k : ℕ, hk : dist (ξ ^ k) 0 < 1⟩ :=
    (h.tendsto_cofinite_zero.eventually (ball_mem_nhds _ zero_lt_one)).exists
  /-
    case intro
    K : Type u_2
    inst✝ : NormedDivisionRing K
    ξ : K
    h : Summable fun n => HPow.hPow ξ n
    k : Nat
    hk : LT.lt (Dist.dist (HPow.hPow ξ k) 0) 1
    ⊢ LT.lt (Norm.norm ξ) 1
  -/
  simp only [norm_pow, dist_zero_right] at hk
  /-
    case intro
    K : Type u_2
    inst✝ : NormedDivisionRing K
    ξ : K
    h : Summable fun n => HPow.hPow ξ n
    k : Nat
    hk : LT.lt (HPow.hPow (Norm.norm ξ) k) 1
    ⊢ LT.lt (Norm.norm ξ) 1
  -/
  rw [← one_pow k] at hk
  /-
    case intro
    K : Type u_2
    inst✝ : NormedDivisionRing K
    ξ : K
    h : Summable fun n => HPow.hPow ξ n
    k : Nat
    hk : LT.lt (HPow.hPow (Norm.norm ξ) k) (HPow.hPow 1 k)
    ⊢ LT.lt (Norm.norm ξ) 1
  -/
  exact lt_of_pow_lt_pow_left₀ _ zero_le_one hk
  /-
    🎉 no goals
  -/


theorem summable_norm_mul_geometric_of_norm_lt_one {k : ℕ} {r : R}
    (hr : ‖r‖ < 1) {u : ℕ → ℕ} (hu : (fun n ↦ (u n : ℝ)) =O[atTop] (fun n ↦ (↑(n ^ k) : ℝ))) :
    Summable fun n : ℕ ↦ ‖(u n * r ^ n : R)‖ := by
  /-
    R : Type u_2
    inst✝ : NormedRing R
    k : Nat
    r : R
    hr : LT.lt (Norm.norm r) 1
    u : Nat → Nat
    hu : Asymptotics.IsBigO Filter.atTop (fun n => ↑(u n)) fun n => ↑(HPow.hPow n k)
    ⊢ Summable fun n => Norm.norm (HMul.hMul (↑(u n)) (HPow.hPow r n))
  -/
  rcases exists_between hr with ⟨r', hrr', h⟩
  /-
    case intro.intro
    R : Type u_2
    inst✝ : NormedRing R
    k : Nat
    r : R
    hr : LT.lt (Norm.norm r) 1
    u : Nat → Nat
    hu : Asymptotics.IsBigO Filter.atTop (fun n => ↑(u n)) fun n => ↑(HPow.hPow n k)
    r' : Real
    hrr' : LT.lt (Norm.norm r) r'
    h : LT.lt r' 1
    ⊢ Summable fun n => Norm.norm (HMul.hMul (↑(u n)) (HPow.hPow r n))
  -/
  rw [← norm_norm] at hrr'
  /-
    case intro.intro
    R : Type u_2
    inst✝ : NormedRing R
    k : Nat
    r : R
    hr : LT.lt (Norm.norm r) 1
    u : Nat → Nat
    hu : Asymptotics.IsBigO Filter.atTop (fun n => ↑(u n)) fun n => ↑(HPow.hPow n k)
    r' : Real
    hrr' : LT.lt (Norm.norm (Norm.norm r)) r'
    h : LT.lt r' 1
    ⊢ Summable fun n => Norm.norm (HMul.hMul (↑(u n)) (HPow.hPow r n))
  -/
  apply summable_of_isBigO_nat (summable_geometric_of_lt_one ((norm_nonneg _).trans hrr'.le) h)
  calc
  fun n ↦ ‖↑(u n) * r ^ n‖
  _ =O[atTop] fun n ↦ u n * ‖r‖ ^ n := by
      apply (IsBigOWith.of_bound (c := ‖(1 : R)‖) ?_).isBigO
      filter_upwards [eventually_norm_pow_le r] with n hn
      simp only [norm_norm, norm_mul, Real.norm_eq_abs, abs_cast, norm_pow, abs_norm]
      apply (norm_mul_le _ _).trans
      have : ‖(u n : R)‖ * ‖r ^ n‖ ≤ (u n * ‖(1 : R)‖) * ‖r‖ ^ n := by
        gcongr; exact norm_cast_le (u n)
      exact this.trans (le_of_eq (by ring))
  _ =O[atTop] fun n ↦ ↑(n ^ k) * ‖r‖ ^ n := hu.mul (isBigO_refl _ _)
  _ =O[atTop] fun n ↦ r' ^ n := by
      simp only [cast_pow]
      exact (isLittleO_pow_const_mul_const_pow_const_pow_of_norm_lt k hrr').isBigO


theorem summable_norm_pow_mul_geometric_of_norm_lt_one (k : ℕ) {r : R}
    (hr : ‖r‖ < 1) : Summable fun n : ℕ ↦ ‖((n : R) ^ k * r ^ n : R)‖ := by
  /-
    R : Type u_2
    inst✝ : NormedRing R
    k : Nat
    r : R
    hr : LT.lt (Norm.norm r) 1
    ⊢ Summable fun n => Norm.norm (HMul.hMul (HPow.hPow (↑n) k) (HPow.hPow r n))
  -/
  simp only [← cast_pow]
  exact summable_norm_mul_geometric_of_norm_lt_one (k := k) (u := fun n ↦ n ^ k) hr
    (isBigO_refl _ _)


theorem summable_norm_geometric_of_norm_lt_one {r : R}
    (hr : ‖r‖ < 1) : Summable fun n : ℕ ↦ ‖(r ^ n : R)‖ := by
  /-
    R : Type u_2
    inst✝ : NormedRing R
    r : R
    hr : LT.lt (Norm.norm r) 1
    ⊢ Summable fun n => Norm.norm (HPow.hPow r n)
  -/
  simpa using summable_norm_pow_mul_geometric_of_norm_lt_one 0 hr
  /-
    🎉 no goals
  -/


lemma hasSum_choose_mul_geometric_of_norm_lt_one'
    (k : ℕ) {r : R} (hr : ‖r‖ < 1) :
    HasSum (fun n ↦ (n + k).choose k * r ^ n) (Ring.inverse (1 - r) ^ (k + 1)) := by
  induction k with
  | zero => simpa using hasSum_geom_series_inverse r hr
  | succ k ih =>
      have I1 : Summable (fun (n : ℕ) ↦ ‖(n + k).choose k * r ^ n‖) := by
        apply summable_norm_mul_geometric_of_norm_lt_one (k := k) hr
        apply isBigO_iff.2 ⟨2 ^ k, ?_⟩
        filter_upwards [Ioi_mem_atTop k] with n (hn : k < n)
        simp only [Real.norm_eq_abs, abs_cast, cast_pow, norm_pow]
        norm_cast
        calc (n + k).choose k
          _ ≤ (2 * n).choose k := choose_le_choose k (by omega)
          _ ≤ (2 * n) ^ k := Nat.choose_le_pow _ _
          _ = 2 ^ k * n ^ k := Nat.mul_pow 2 n k
      convert hasSum_sum_range_mul_of_summable_norm' I1 ih.summable
        (summable_norm_geometric_of_norm_lt_one hr) (summable_geometric_of_norm_lt_one hr) with n
      · have : ∑ i ∈ Finset.range (n + 1), ↑((i + k).choose k) * r ^ i * r ^ (n - i) =
            ∑ i ∈ Finset.range (n + 1), ↑((i + k).choose k) * r ^ n := by
          apply Finset.sum_congr rfl (fun i hi ↦ ?_)
          simp only [Finset.mem_range] at hi
          rw [mul_assoc, ← pow_add, show i + (n - i) = n by omega]
        simp [this, ← sum_mul, ← Nat.cast_sum, sum_range_add_choose n k, add_assoc]
      · rw [ih.tsum_eq, (hasSum_geom_series_inverse r hr).tsum_eq, pow_succ]


lemma summable_choose_mul_geometric_of_norm_lt_one (k : ℕ) {r : R} (hr : ‖r‖ < 1) :
    Summable (fun n ↦ (n + k).choose k * r ^ n) :=
  (hasSum_choose_mul_geometric_of_norm_lt_one' k hr).summable


lemma tsum_choose_mul_geometric_of_norm_lt_one' (k : ℕ) {r : R} (hr : ‖r‖ < 1) :
    ∑' n, (n + k).choose k * r ^ n = (Ring.inverse (1 - r)) ^ (k + 1) :=
  (hasSum_choose_mul_geometric_of_norm_lt_one' k hr).tsum_eq


lemma hasSum_choose_mul_geometric_of_norm_lt_one
    (k : ℕ) {r : 𝕜} (hr : ‖r‖ < 1) :
    HasSum (fun n ↦ (n + k).choose k * r ^ n) (1 / (1 - r) ^ (k + 1)) := by
  /-
    𝕜 : Type u_3
    inst✝ : NormedDivisionRing 𝕜
    k : Nat
    r : 𝕜
    hr : LT.lt (Norm.norm r) 1
    ⊢ HasSum (fun n => HMul.hMul (↑((HAdd.hAdd n k).choose k)) (HPow.hPow r n)) (H …
  -/
  convert hasSum_choose_mul_geometric_of_norm_lt_one' k hr
  /-
    case h.e'_6
    𝕜 : Type u_3
    inst✝ : NormedDivisionRing 𝕜
    k : Nat
    r : 𝕜
    hr : LT.lt (Norm.norm r) 1
    ⊢ Eq (HDiv.hDiv 1 (HPow.hPow (HSub.hSub 1 r) (HAdd.hAdd k 1))) (HPow.hPow (Rin …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma tsum_choose_mul_geometric_of_norm_lt_one (k : ℕ) {r : 𝕜} (hr : ‖r‖ < 1) :
    ∑' n, (n + k).choose k * r ^ n = 1/ (1 - r) ^ (k + 1) :=
  (hasSum_choose_mul_geometric_of_norm_lt_one k hr).tsum_eq


lemma summable_descFactorial_mul_geometric_of_norm_lt_one (k : ℕ) {r : R} (hr : ‖r‖ < 1) :
    Summable (fun n ↦ (n + k).descFactorial k * r ^ n) := by
  convert (summable_choose_mul_geometric_of_norm_lt_one k hr).mul_left (k.factorial : R)
    using 2 with n
  /-
    case h.e'_5.h
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    k : Nat
    r : R
    hr : LT.lt (Norm.norm r) 1
    n : Nat
    ⊢ Eq (HMul.hMul (↑((HAdd.hAdd n k).descFactorial k)) (HPow.hPow r n)) (HMul.hM …
  -/
  simp [← mul_assoc, descFactorial_eq_factorial_mul_choose (n + k) k]
  /-
    🎉 no goals
  -/


open Polynomial in
theorem summable_pow_mul_geometric_of_norm_lt_one (k : ℕ) {r : R} (hr : ‖r‖ < 1) :
    Summable (fun n ↦ (n : R) ^ k * r ^ n : ℕ → R) := by
  /-
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    k : Nat
    r : R
    hr : LT.lt (Norm.norm r) 1
    ⊢ Summable fun n => HMul.hMul (HPow.hPow (↑n) k) (HPow.hPow r n)
  -/
  refine Nat.strong_induction_on k fun k hk => ?_
  obtain ⟨a, ha⟩ : ∃ (a : ℕ → ℕ), ∀ n, (n + k).descFactorial k
      = n ^ k + ∑ i ∈ range k, a i * n ^ i := by
    let P : Polynomial ℕ := (ascPochhammer ℕ k).comp (Polynomial.X + C 1)
    refine ⟨fun i ↦ P.coeff i, fun n ↦ ?_⟩
    have mP : Monic P := Monic.comp_X_add_C (monic_ascPochhammer ℕ k) _
    have dP : P.natDegree = k := by
      simp only [P, natDegree_comp, ascPochhammer_natDegree, mul_one, natDegree_X_add_C]
    have A : (n + k).descFactorial k = P.eval n := by
      have : n + 1 + k - 1 = n + k := by omega
      simp [P, ascPochhammer_nat_eq_descFactorial, this]
    conv_lhs => rw [A, mP.as_sum, dP]
    simp [eval_finset_sum]
  have : Summable (fun n ↦ (n + k).descFactorial k * r ^ n
      - ∑ i ∈ range k, a i * n ^ (i : ℕ) * r ^ n) := by
    apply (summable_descFactorial_mul_geometric_of_norm_lt_one k hr).sub
    apply summable_sum (fun i hi ↦ ?_)
    simp_rw [mul_assoc]
    simp only [Finset.mem_range] at hi
    exact (hk _ hi).mul_left _
  /-
    case intro
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    k✝ : Nat
    r : R
    hr : LT.lt (Norm.norm r) 1
    k : Nat
    hk : ∀ (m : Nat), LT.lt m k → Summable fun n => HMul.hMul (HPow.hPow (↑n) m) ( …
    a : Nat → Nat
    ha : ∀ (n : Nat), Eq ((HAdd.hAdd n k).descFactorial k) (HAdd.hAdd (HPow.hPow n …
    this : Summable fun n => HSub.hSub (HMul.hMul (↑((HAdd.hAdd n k).descFactorial …
    ⊢ Summable fun n => HMul.hMul (HPow.hPow (↑n) k) (HPow.hPow r n)
  -/
  convert this using 1
  /-
    case h.e'_5
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    k✝ : Nat
    r : R
    hr : LT.lt (Norm.norm r) 1
    k : Nat
    hk : ∀ (m : Nat), LT.lt m k → Summable fun n => HMul.hMul (HPow.hPow (↑n) m) ( …
    a : Nat → Nat
    ha : ∀ (n : Nat), Eq ((HAdd.hAdd n k).descFactorial k) (HAdd.hAdd (HPow.hPow n …
    this : Summable fun n => HSub.hSub (HMul.hMul (↑((HAdd.hAdd n k).descFactorial …
    ⊢ Eq (fun n => HMul.hMul (HPow.hPow (↑n) k) (HPow.hPow r n)) fun n => HSub.hSu …
  -/
  ext n
  /-
    case h.e'_5.h
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    k✝ : Nat
    r : R
    hr : LT.lt (Norm.norm r) 1
    k : Nat
    hk : ∀ (m : Nat), LT.lt m k → Summable fun n => HMul.hMul (HPow.hPow (↑n) m) ( …
    a : Nat → Nat
    ha : ∀ (n : Nat), Eq ((HAdd.hAdd n k).descFactorial k) (HAdd.hAdd (HPow.hPow n …
    this : Summable fun n => HSub.hSub (HMul.hMul (↑((HAdd.hAdd n k).descFactorial …
    n : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow (↑n) k) (HPow.hPow r n)) (HSub.hSub (HMul.hMul (↑(( …
  -/
  simp [ha n, add_mul, sum_mul]
  /-
    🎉 no goals
  -/


/-- If `‖r‖ < 1`, then `∑' n : ℕ, n * r ^ n = r / (1 - r) ^ 2`, `HasSum` version in a general ring
with summable geometric series. For a version in a field, using division instead of `Ring.inverse`,
see `hasSum_coe_mul_geometric_of_norm_lt_one`. -/
theorem hasSum_coe_mul_geometric_of_norm_lt_one'
    {x : R} (h : ‖x‖ < 1) :
    HasSum (fun n ↦ n * x ^ n : ℕ → R) (x * (Ring.inverse (1 - x)) ^ 2) := by
  have A : HasSum (fun (n : ℕ) ↦ (n + 1) * x ^ n) (Ring.inverse (1 - x) ^ 2) := by
    convert hasSum_choose_mul_geometric_of_norm_lt_one' 1 h with n
    simp
  /-
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    A : HasSum (fun n => HMul.hMul (HAdd.hAdd (↑n) 1) (HPow.hPow x n)) (HPow.hPow  …
    ⊢ HasSum (fun n => HMul.hMul (↑n) (HPow.hPow x n)) (HMul.hMul x (HPow.hPow (Ri …
  -/
  have B : HasSum (fun (n : ℕ) ↦ x ^ n) (Ring.inverse (1 - x)) := hasSum_geom_series_inverse x h
  /-
    R : Type u_2
    inst✝¹ : NormedRing R
    inst✝ : HasSummableGeomSeries R
    x : R
    h : LT.lt (Norm.norm x) 1
    A : HasSum (fun n => HMul.hMul (HAdd.hAdd (↑n) 1) (HPow.hPow x n)) (HPow.hPow  …
    B : HasSum (fun n => HPow.hPow x n) (Ring.inverse (HSub.hSub 1 x))
    ⊢ HasSum (fun n => HMul.hMul (↑n) (HPow.hPow x n)) (HMul.hMul x (HPow.hPow (Ri …
  -/
  convert A.sub B using 1
    /-
      case h.e'_5
      R : Type u_2
      inst✝¹ : NormedRing R
      inst✝ : HasSummableGeomSeries R
      x : R
      h : LT.lt (Norm.norm x) 1
      A : HasSum (fun n => HMul.hMul (HAdd.hAdd (↑n) 1) (HPow.hPow x n)) (HPow.hPow  …
      B : HasSum (fun n => HPow.hPow x n) (Ring.inverse (HSub.hSub 1 x))
      ⊢ Eq (fun n => HMul.hMul (↑n) (HPow.hPow x n)) fun b => HSub.hSub (HMul.hMul ( …
    -/
  · ext n
    /-
      case h.e'_5.h
      R : Type u_2
      inst✝¹ : NormedRing R
      inst✝ : HasSummableGeomSeries R
      x : R
      h : LT.lt (Norm.norm x) 1
      A : HasSum (fun n => HMul.hMul (HAdd.hAdd (↑n) 1) (HPow.hPow x n)) (HPow.hPow  …
      B : HasSum (fun n => HPow.hPow x n) (Ring.inverse (HSub.hSub 1 x))
      n : Nat
      ⊢ Eq (HMul.hMul (↑n) (HPow.hPow x n)) (HSub.hSub (HMul.hMul (HAdd.hAdd (↑n) 1) …
    -/
    simp [add_mul]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_6
      R : Type u_2
      inst✝¹ : NormedRing R
      inst✝ : HasSummableGeomSeries R
      x : R
      h : LT.lt (Norm.norm x) 1
      A : HasSum (fun n => HMul.hMul (HAdd.hAdd (↑n) 1) (HPow.hPow x n)) (HPow.hPow  …
      B : HasSum (fun n => HPow.hPow x n) (Ring.inverse (HSub.hSub 1 x))
      ⊢ Eq (HMul.hMul x (HPow.hPow (Ring.inverse (HSub.hSub 1 x)) 2)) (HSub.hSub (HP …
    -/
  · symm
    calc Ring.inverse (1 - x) ^ 2 - Ring.inverse (1 - x)
    _ = Ring.inverse (1 - x) ^ 2 - ((1 - x) * Ring.inverse (1 - x)) * Ring.inverse (1 - x) := by
      simp [Ring.mul_inverse_cancel (1 - x) (isUnit_one_sub_of_norm_lt_one h)]
    _ = x * Ring.inverse (1 - x) ^ 2 := by noncomm_ring


/-- If `‖r‖ < 1`, then `∑' n : ℕ, n * r ^ n = r / (1 - r) ^ 2`, version in a general ring with
summable geometric series. For a version in a field, using division instead of `Ring.inverse`,
see `tsum_coe_mul_geometric_of_norm_lt_one`. -/
theorem tsum_coe_mul_geometric_of_norm_lt_one'
    {r : 𝕜} (hr : ‖r‖ < 1) : (∑' n : ℕ, n * r ^ n : 𝕜) = r * Ring.inverse (1 - r) ^ 2 :=
  (hasSum_coe_mul_geometric_of_norm_lt_one' hr).tsum_eq


/-- If `‖r‖ < 1`, then `∑' n : ℕ, n * r ^ n = r / (1 - r) ^ 2`, `HasSum` version. -/
theorem hasSum_coe_mul_geometric_of_norm_lt_one {r : 𝕜} (hr : ‖r‖ < 1) :
    HasSum (fun n ↦ n * r ^ n : ℕ → 𝕜) (r / (1 - r) ^ 2) := by
  /-
    𝕜 : Type u_3
    inst✝ : NormedDivisionRing 𝕜
    r : 𝕜
    hr : LT.lt (Norm.norm r) 1
    ⊢ HasSum (fun n => HMul.hMul (↑n) (HPow.hPow r n)) (HDiv.hDiv r (HPow.hPow (HS …
  -/
  convert hasSum_coe_mul_geometric_of_norm_lt_one' hr using 1
  /-
    case h.e'_6
    𝕜 : Type u_3
    inst✝ : NormedDivisionRing 𝕜
    r : 𝕜
    hr : LT.lt (Norm.norm r) 1
    ⊢ Eq (HDiv.hDiv r (HPow.hPow (HSub.hSub 1 r) 2)) (HMul.hMul r (HPow.hPow (Ring …
  -/
  simp [div_eq_mul_inv]
  /-
    🎉 no goals
  -/


/-- If `‖r‖ < 1`, then `∑' n : ℕ, n * r ^ n = r / (1 - r) ^ 2`. -/
theorem tsum_coe_mul_geometric_of_norm_lt_one {r : 𝕜} (hr : ‖r‖ < 1) :
    (∑' n : ℕ, n * r ^ n : 𝕜) = r / (1 - r) ^ 2 :=
  (hasSum_coe_mul_geometric_of_norm_lt_one hr).tsum_eq


nonrec theorem SeminormedAddCommGroup.cauchySeq_of_le_geometric {C : ℝ} {r : ℝ} (hr : r < 1)
    {u : ℕ → α} (h : ∀ n, ‖u n - u (n + 1)‖ ≤ C * r ^ n) : CauchySeq u :=
                                       /-
                                         α : Type u_1
                                         inst✝ : SeminormedAddCommGroup α
                                         C r : Real
                                         hr : LT.lt r 1
                                         u : Nat → α
                                         h : ∀ (n : Nat), LE.le (Norm.norm (HSub.hSub (u n) (u (HAdd.hAdd n 1)))) (HMul …
                                         ⊢ ∀ (n : Nat), LE.le (Dist.dist (u n) (u (HAdd.hAdd n 1))) (HMul.hMul C (HPow. …
                                       -/
  cauchySeq_of_le_geometric r C hr (by simpa [dist_eq_norm] using h)
                                       /-
                                         🎉 no goals
                                       -/


theorem dist_partial_sum_le_of_le_geometric (hf : ∀ n, ‖f n‖ ≤ C * r ^ n) (n : ℕ) :
    dist (∑ i ∈ range n, f i) (∑ i ∈ range (n + 1), f i) ≤ C * r ^ n := by
  /-
    α : Type u_1
    inst✝ : SeminormedAddCommGroup α
    r C : Real
    f : Nat → α
    hf : ∀ (n : Nat), LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow r n))
    n : Nat
    ⊢ LE.le (Dist.dist ((Finset.range n).sum fun i => f i) ((Finset.range (HAdd.hA …
  -/
  rw [sum_range_succ, dist_eq_norm, ← norm_neg, neg_sub, add_sub_cancel_left]
  /-
    α : Type u_1
    inst✝ : SeminormedAddCommGroup α
    r C : Real
    f : Nat → α
    hf : ∀ (n : Nat), LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow r n))
    n : Nat
    ⊢ LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow r n))
  -/
  exact hf n
  /-
    🎉 no goals
  -/


/-- If `‖f n‖ ≤ C * r ^ n` for all `n : ℕ` and some `r < 1`, then the partial sums of `f` form a
Cauchy sequence. This lemma does not assume `0 ≤ r` or `0 ≤ C`. -/
theorem cauchySeq_finset_of_geometric_bound (hr : r < 1) (hf : ∀ n, ‖f n‖ ≤ C * r ^ n) :
    CauchySeq fun s : Finset ℕ ↦ ∑ x ∈ s, f x :=
  cauchySeq_finset_of_norm_bounded _
    (aux_hasSum_of_le_geometric hr (dist_partial_sum_le_of_le_geometric hf)).summable hf


/-- If `‖f n‖ ≤ C * r ^ n` for all `n : ℕ` and some `r < 1`, then the partial sums of `f` are within
distance `C * r ^ n / (1 - r)` of the sum of the series. This lemma does not assume `0 ≤ r` or
`0 ≤ C`. -/
theorem norm_sub_le_of_geometric_bound_of_hasSum (hr : r < 1) (hf : ∀ n, ‖f n‖ ≤ C * r ^ n) {a : α}
    (ha : HasSum f a) (n : ℕ) : ‖(∑ x ∈ Finset.range n, f x) - a‖ ≤ C * r ^ n / (1 - r) := by
  /-
    α : Type u_1
    inst✝ : SeminormedAddCommGroup α
    r C : Real
    f : Nat → α
    hr : LT.lt r 1
    hf : ∀ (n : Nat), LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow r n))
    a : α
    ha : HasSum f a
    n : Nat
    ⊢ LE.le (Norm.norm (HSub.hSub ((Finset.range n).sum fun x => f x) a)) (HDiv.hD …
  -/
  rw [← dist_eq_norm]
  /-
    α : Type u_1
    inst✝ : SeminormedAddCommGroup α
    r C : Real
    f : Nat → α
    hr : LT.lt r 1
    hf : ∀ (n : Nat), LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow r n))
    a : α
    ha : HasSum f a
    n : Nat
    ⊢ LE.le (Dist.dist ((Finset.range n).sum fun x => f x) a) (HDiv.hDiv (HMul.hMu …
  -/
  apply dist_le_of_le_geometric_of_tendsto r C hr (dist_partial_sum_le_of_le_geometric hf)
  /-
    case ha
    α : Type u_1
    inst✝ : SeminormedAddCommGroup α
    r C : Real
    f : Nat → α
    hr : LT.lt r 1
    hf : ∀ (n : Nat), LE.le (Norm.norm (f n)) (HMul.hMul C (HPow.hPow r n))
    a : α
    ha : HasSum f a
    n : Nat
    ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun i => f i) Filter.atTop (nh …
  -/
  exact ha.tendsto_sum_nat
  /-
    🎉 no goals
  -/


@[simp]
theorem dist_partial_sum (u : ℕ → α) (n : ℕ) :
    dist (∑ k ∈ range (n + 1), u k) (∑ k ∈ range n, u k) = ‖u n‖ := by
  /-
    α : Type u_1
    inst✝ : SeminormedAddCommGroup α
    u : Nat → α
    n : Nat
    ⊢ Eq (Dist.dist ((Finset.range (HAdd.hAdd n 1)).sum fun k => u k) ((Finset.ran …
  -/
  simp [dist_eq_norm, sum_range_succ]
  /-
    🎉 no goals
  -/


@[simp]
theorem dist_partial_sum' (u : ℕ → α) (n : ℕ) :
    dist (∑ k ∈ range n, u k) (∑ k ∈ range (n + 1), u k) = ‖u n‖ := by
  /-
    α : Type u_1
    inst✝ : SeminormedAddCommGroup α
    u : Nat → α
    n : Nat
    ⊢ Eq (Dist.dist ((Finset.range n).sum fun k => u k) ((Finset.range (HAdd.hAdd  …
  -/
  simp [dist_eq_norm', sum_range_succ]
  /-
    🎉 no goals
  -/


theorem cauchy_series_of_le_geometric {C : ℝ} {u : ℕ → α} {r : ℝ} (hr : r < 1)
    (h : ∀ n, ‖u n‖ ≤ C * r ^ n) : CauchySeq fun n ↦ ∑ k ∈ range n, u k :=
                                       /-
                                         α : Type u_1
                                         inst✝ : SeminormedAddCommGroup α
                                         C : Real
                                         u : Nat → α
                                         r : Real
                                         hr : LT.lt r 1
                                         h : ∀ (n : Nat), LE.le (Norm.norm (u n)) (HMul.hMul C (HPow.hPow r n))
                                         ⊢ ∀ (n : Nat), LE.le (Dist.dist ((Finset.range n).sum fun k => u k) ((Finset.r …
                                       -/
  cauchySeq_of_le_geometric r C hr (by simp [h])
                                       /-
                                         🎉 no goals
                                       -/


theorem NormedAddCommGroup.cauchy_series_of_le_geometric' {C : ℝ} {u : ℕ → α} {r : ℝ} (hr : r < 1)
    (h : ∀ n, ‖u n‖ ≤ C * r ^ n) : CauchySeq fun n ↦ ∑ k ∈ range (n + 1), u k :=
  (cauchy_series_of_le_geometric hr h).comp_tendsto <| tendsto_add_atTop_nat 1


theorem NormedAddCommGroup.cauchy_series_of_le_geometric'' {C : ℝ} {u : ℕ → α} {N : ℕ} {r : ℝ}
    (hr₀ : 0 < r) (hr₁ : r < 1) (h : ∀ n ≥ N, ‖u n‖ ≤ C * r ^ n) :
    CauchySeq fun n ↦ ∑ k ∈ range (n + 1), u k := by
  /-
    α : Type u_1
    inst✝ : SeminormedAddCommGroup α
    C : Real
    u : Nat → α
    N : Nat
    r : Real
    hr₀ : LT.lt 0 r
    hr₁ : LT.lt r 1
    h : ∀ (n : Nat), GE.ge n N → LE.le (Norm.norm (u n)) (HMul.hMul C (HPow.hPow r …
    ⊢ CauchySeq fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => u k
  -/
  set v : ℕ → α := fun n ↦ if n < N then 0 else u n
  have hC : 0 ≤ C :=
    (mul_nonneg_iff_of_pos_right <| pow_pos hr₀ N).mp ((norm_nonneg _).trans <| h N <| le_refl N)
  have : ∀ n ≥ N, u n = v n := by
    intro n hn
    simp [v, hn, if_neg (not_lt.mpr hn)]
  apply cauchySeq_sum_of_eventually_eq this
    (NormedAddCommGroup.cauchy_series_of_le_geometric' hr₁ _)
    /-
      α : Type u_1
      inst✝ : SeminormedAddCommGroup α
      C : Real
      u : Nat → α
      N : Nat
      r : Real
      hr₀ : LT.lt 0 r
      hr₁ : LT.lt r 1
      h : ∀ (n : Nat), GE.ge n N → LE.le (Norm.norm (u n)) (HMul.hMul C (HPow.hPow r …
      v : Nat → α := fun n => ite (LT.lt n N) 0 (u n)
      hC : LE.le 0 C
      this : ∀ (n : Nat), GE.ge n N → Eq (u n) (v n)
      ⊢ Real
    -/
  · exact C
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    inst✝ : SeminormedAddCommGroup α
    C : Real
    u : Nat → α
    N : Nat
    r : Real
    hr₀ : LT.lt 0 r
    hr₁ : LT.lt r 1
    h : ∀ (n : Nat), GE.ge n N → LE.le (Norm.norm (u n)) (HMul.hMul C (HPow.hPow r …
    v : Nat → α := fun n => ite (LT.lt n N) 0 (u n)
    hC : LE.le 0 C
    this : ∀ (n : Nat), GE.ge n N → Eq (u n) (v n)
    ⊢ ∀ (n : Nat), LE.le (Norm.norm (v n)) (HMul.hMul C (HPow.hPow r n))
  -/
  intro n
  /-
    α : Type u_1
    inst✝ : SeminormedAddCommGroup α
    C : Real
    u : Nat → α
    N : Nat
    r : Real
    hr₀ : LT.lt 0 r
    hr₁ : LT.lt r 1
    h : ∀ (n : Nat), GE.ge n N → LE.le (Norm.norm (u n)) (HMul.hMul C (HPow.hPow r …
    v : Nat → α := fun n => ite (LT.lt n N) 0 (u n)
    hC : LE.le 0 C
    this : ∀ (n : Nat), GE.ge n N → Eq (u n) (v n)
    n : Nat
    ⊢ LE.le (Norm.norm (v n)) (HMul.hMul C (HPow.hPow r n))
  -/
  simp only [v]
  /-
    α : Type u_1
    inst✝ : SeminormedAddCommGroup α
    C : Real
    u : Nat → α
    N : Nat
    r : Real
    hr₀ : LT.lt 0 r
    hr₁ : LT.lt r 1
    h : ∀ (n : Nat), GE.ge n N → LE.le (Norm.norm (u n)) (HMul.hMul C (HPow.hPow r …
    v : Nat → α := fun n => ite (LT.lt n N) 0 (u n)
    hC : LE.le 0 C
    this : ∀ (n : Nat), GE.ge n N → Eq (u n) (v n)
    n : Nat
    ⊢ LE.le (Norm.norm (ite (LT.lt n N) 0 (u n))) (HMul.hMul C (HPow.hPow r n))
  -/
  split_ifs with H
    /-
      case pos
      α : Type u_1
      inst✝ : SeminormedAddCommGroup α
      C : Real
      u : Nat → α
      N : Nat
      r : Real
      hr₀ : LT.lt 0 r
      hr₁ : LT.lt r 1
      h : ∀ (n : Nat), GE.ge n N → LE.le (Norm.norm (u n)) (HMul.hMul C (HPow.hPow r …
      v : Nat → α := fun n => ite (LT.lt n N) 0 (u n)
      hC : LE.le 0 C
      this : ∀ (n : Nat), GE.ge n N → Eq (u n) (v n)
      n : Nat
      H : LT.lt n N
      ⊢ LE.le (Norm.norm 0) (HMul.hMul C (HPow.hPow r n))
    -/
  · rw [norm_zero]
    /-
      case pos
      α : Type u_1
      inst✝ : SeminormedAddCommGroup α
      C : Real
      u : Nat → α
      N : Nat
      r : Real
      hr₀ : LT.lt 0 r
      hr₁ : LT.lt r 1
      h : ∀ (n : Nat), GE.ge n N → LE.le (Norm.norm (u n)) (HMul.hMul C (HPow.hPow r …
      v : Nat → α := fun n => ite (LT.lt n N) 0 (u n)
      hC : LE.le 0 C
      this : ∀ (n : Nat), GE.ge n N → Eq (u n) (v n)
      n : Nat
      H : LT.lt n N
      ⊢ LE.le 0 (HMul.hMul C (HPow.hPow r n))
    -/
    exact mul_nonneg hC (pow_nonneg hr₀.le _)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : SeminormedAddCommGroup α
      C : Real
      u : Nat → α
      N : Nat
      r : Real
      hr₀ : LT.lt 0 r
      hr₁ : LT.lt r 1
      h : ∀ (n : Nat), GE.ge n N → LE.le (Norm.norm (u n)) (HMul.hMul C (HPow.hPow r …
      v : Nat → α := fun n => ite (LT.lt n N) 0 (u n)
      hC : LE.le 0 C
      this : ∀ (n : Nat), GE.ge n N → Eq (u n) (v n)
      n : Nat
      H : Not (LT.lt n N)
      ⊢ LE.le (Norm.norm (u n)) (HMul.hMul C (HPow.hPow r n))
    -/
  · push_neg at H
    /-
      case neg
      α : Type u_1
      inst✝ : SeminormedAddCommGroup α
      C : Real
      u : Nat → α
      N : Nat
      r : Real
      hr₀ : LT.lt 0 r
      hr₁ : LT.lt r 1
      h : ∀ (n : Nat), GE.ge n N → LE.le (Norm.norm (u n)) (HMul.hMul C (HPow.hPow r …
      v : Nat → α := fun n => ite (LT.lt n N) 0 (u n)
      hC : LE.le 0 C
      this : ∀ (n : Nat), GE.ge n N → Eq (u n) (v n)
      n : Nat
      H : LE.le N n
      ⊢ LE.le (Norm.norm (u n)) (HMul.hMul C (HPow.hPow r n))
    -/
    exact h _ H
    /-
      🎉 no goals
    -/


/-- The term norms of any convergent series are bounded by a constant. -/
lemma exists_norm_le_of_cauchySeq (h : CauchySeq fun n ↦ ∑ k ∈ range n, f k) :
    ∃ C, ∀ n, ‖f n‖ ≤ C := by
  /-
    α : Type u_1
    inst✝ : SeminormedAddCommGroup α
    f : Nat → α
    h : CauchySeq fun n => (Finset.range n).sum fun k => f k
    ⊢ Exists fun C => ∀ (n : Nat), LE.le (Norm.norm (f n)) C
  -/
  obtain ⟨b, ⟨_, key, _⟩⟩ := cauchySeq_iff_le_tendsto_0.mp h
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : SeminormedAddCommGroup α
    f : Nat → α
    h : CauchySeq fun n => (Finset.range n).sum fun k => f k
    b : Nat → Real
    left✝ : ∀ (n : Nat), LE.le 0 (b n)
    key : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist ((Finset.range …
    right✝ : Filter.Tendsto b Filter.atTop (nhds 0)
    ⊢ Exists fun C => ∀ (n : Nat), LE.le (Norm.norm (f n)) C
  -/
  refine ⟨b 0, fun n ↦ ?_⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : SeminormedAddCommGroup α
    f : Nat → α
    h : CauchySeq fun n => (Finset.range n).sum fun k => f k
    b : Nat → Real
    left✝ : ∀ (n : Nat), LE.le 0 (b n)
    key : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist ((Finset.range …
    right✝ : Filter.Tendsto b Filter.atTop (nhds 0)
    n : Nat
    ⊢ LE.le (Norm.norm (f n)) (b 0)
  -/
  simpa only [dist_partial_sum'] using key n (n + 1) 0 (_root_.zero_le _) (_root_.zero_le _)
  /-
    🎉 no goals
  -/


theorem summable_of_ratio_norm_eventually_le {α : Type*} [SeminormedAddCommGroup α]
    [CompleteSpace α] {f : ℕ → α} {r : ℝ} (hr₁ : r < 1)
    (h : ∀ᶠ n in atTop, ‖f (n + 1)‖ ≤ r * ‖f n‖) : Summable f := by
  /-
    α : Type u_2
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : CompleteSpace α
    f : Nat → α
    r : Real
    hr₁ : LT.lt r 1
    h : Filter.Eventually (fun n => LE.le (Norm.norm (f (HAdd.hAdd n 1))) (HMul.hM …
    ⊢ Summable f
  -/
  by_cases hr₀ : 0 ≤ r
    /-
      case pos
      α : Type u_2
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : CompleteSpace α
      f : Nat → α
      r : Real
      hr₁ : LT.lt r 1
      h : Filter.Eventually (fun n => LE.le (Norm.norm (f (HAdd.hAdd n 1))) (HMul.hM …
      hr₀ : LE.le 0 r
      ⊢ Summable f
    -/
  · rw [eventually_atTop] at h
    /-
      case pos
      α : Type u_2
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : CompleteSpace α
      f : Nat → α
      r : Real
      hr₁ : LT.lt r 1
      h : Exists fun a => ∀ (b : Nat), GE.ge b a → LE.le (Norm.norm (f (HAdd.hAdd b  …
      hr₀ : LE.le 0 r
      ⊢ Summable f
    -/
    rcases h with ⟨N, hN⟩
    /-
      case pos.intro
      α : Type u_2
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : CompleteSpace α
      f : Nat → α
      r : Real
      hr₁ : LT.lt r 1
      hr₀ : LE.le 0 r
      N : Nat
      hN : ∀ (b : Nat), GE.ge b N → LE.le (Norm.norm (f (HAdd.hAdd b 1))) (HMul.hMul …
      ⊢ Summable f
    -/
    rw [← @summable_nat_add_iff α _ _ _ _ N]
    refine .of_norm_bounded (fun n ↦ ‖f N‖ * r ^ n)
      (Summable.mul_left _ <| summable_geometric_of_lt_one hr₀ hr₁) fun n ↦ ?_
    /-
      case pos.intro
      α : Type u_2
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : CompleteSpace α
      f : Nat → α
      r : Real
      hr₁ : LT.lt r 1
      hr₀ : LE.le 0 r
      N : Nat
      hN : ∀ (b : Nat), GE.ge b N → LE.le (Norm.norm (f (HAdd.hAdd b 1))) (HMul.hMul …
      n : Nat
      ⊢ LE.le (Norm.norm (f (HAdd.hAdd n N))) ((fun n => HMul.hMul (Norm.norm (f N)) …
    -/
    simp only
    /-
      case pos.intro
      α : Type u_2
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : CompleteSpace α
      f : Nat → α
      r : Real
      hr₁ : LT.lt r 1
      hr₀ : LE.le 0 r
      N : Nat
      hN : ∀ (b : Nat), GE.ge b N → LE.le (Norm.norm (f (HAdd.hAdd b 1))) (HMul.hMul …
      n : Nat
      ⊢ LE.le (Norm.norm (f (HAdd.hAdd n N))) (HMul.hMul (Norm.norm (f N)) (HPow.hPo …
    -/
    conv_rhs => rw [mul_comm, ← zero_add N]
    /-
      case pos.intro
      α : Type u_2
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : CompleteSpace α
      f : Nat → α
      r : Real
      hr₁ : LT.lt r 1
      hr₀ : LE.le 0 r
      N : Nat
      hN : ∀ (b : Nat), GE.ge b N → LE.le (Norm.norm (f (HAdd.hAdd b 1))) (HMul.hMul …
      n : Nat
      ⊢ LE.le (Norm.norm (f (HAdd.hAdd n N))) (HMul.hMul (HPow.hPow r n) (Norm.norm  …
    -/
    refine le_geom (u := fun n ↦ ‖f (n + N)‖) hr₀ n fun i _ ↦ ?_
    /-
      case pos.intro
      α : Type u_2
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : CompleteSpace α
      f : Nat → α
      r : Real
      hr₁ : LT.lt r 1
      hr₀ : LE.le 0 r
      N : Nat
      hN : ∀ (b : Nat), GE.ge b N → LE.le (Norm.norm (f (HAdd.hAdd b 1))) (HMul.hMul …
      n i : Nat
      x✝ : LT.lt i n
      ⊢ LE.le ((fun n => Norm.norm (f (HAdd.hAdd n N))) (HAdd.hAdd i 1)) (HMul.hMul  …
    -/
    convert hN (i + N) (N.le_add_left i) using 3
    /-
      case h.e'_3.h.e'_3.h.e'_1
      α : Type u_2
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : CompleteSpace α
      f : Nat → α
      r : Real
      hr₁ : LT.lt r 1
      hr₀ : LE.le 0 r
      N : Nat
      hN : ∀ (b : Nat), GE.ge b N → LE.le (Norm.norm (f (HAdd.hAdd b 1))) (HMul.hMul …
      n i : Nat
      x✝ : LT.lt i n
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd i 1) N) (HAdd.hAdd (HAdd.hAdd i N) 1)
    -/
    ac_rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : CompleteSpace α
      f : Nat → α
      r : Real
      hr₁ : LT.lt r 1
      h : Filter.Eventually (fun n => LE.le (Norm.norm (f (HAdd.hAdd n 1))) (HMul.hM …
      hr₀ : Not (LE.le 0 r)
      ⊢ Summable f
    -/
  · push_neg at hr₀
    /-
      case neg
      α : Type u_2
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : CompleteSpace α
      f : Nat → α
      r : Real
      hr₁ : LT.lt r 1
      h : Filter.Eventually (fun n => LE.le (Norm.norm (f (HAdd.hAdd n 1))) (HMul.hM …
      hr₀ : LT.lt r 0
      ⊢ Summable f
    -/
    refine .of_norm_bounded_eventually_nat 0 summable_zero ?_
    /-
      case neg
      α : Type u_2
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : CompleteSpace α
      f : Nat → α
      r : Real
      hr₁ : LT.lt r 1
      h : Filter.Eventually (fun n => LE.le (Norm.norm (f (HAdd.hAdd n 1))) (HMul.hM …
      hr₀ : LT.lt r 0
      ⊢ Filter.Eventually (fun i => LE.le (Norm.norm (f i)) (0 i)) Filter.atTop
    -/
    filter_upwards [h] with _ hn
    /-
      case h
      α : Type u_2
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : CompleteSpace α
      f : Nat → α
      r : Real
      hr₁ : LT.lt r 1
      h : Filter.Eventually (fun n => LE.le (Norm.norm (f (HAdd.hAdd n 1))) (HMul.hM …
      hr₀ : LT.lt r 0
      a✝ : Nat
      hn : LE.le (Norm.norm (f (HAdd.hAdd a✝ 1))) (HMul.hMul r (Norm.norm (f a✝)))
      ⊢ LE.le (Norm.norm (f a✝)) (0 a✝)
    -/
    by_contra! h
    /-
      case h
      α : Type u_2
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : CompleteSpace α
      f : Nat → α
      r : Real
      hr₁ : LT.lt r 1
      h✝ : Filter.Eventually (fun n => LE.le (Norm.norm (f (HAdd.hAdd n 1))) (HMul.h …
      hr₀ : LT.lt r 0
      a✝ : Nat
      hn : LE.le (Norm.norm (f (HAdd.hAdd a✝ 1))) (HMul.hMul r (Norm.norm (f a✝)))
      h : LT.lt (0 a✝) (Norm.norm (f a✝))
      ⊢ False
    -/
    exact not_lt.mpr (norm_nonneg _) (lt_of_le_of_lt hn <| mul_neg_of_neg_of_pos hr₀ h)
    /-
      🎉 no goals
    -/


theorem summable_of_ratio_test_tendsto_lt_one {α : Type*} [NormedAddCommGroup α] [CompleteSpace α]
    {f : ℕ → α} {l : ℝ} (hl₁ : l < 1) (hf : ∀ᶠ n in atTop, f n ≠ 0)
    (h : Tendsto (fun n ↦ ‖f (n + 1)‖ / ‖f n‖) atTop (𝓝 l)) : Summable f := by
  /-
    α : Type u_2
    inst✝¹ : NormedAddCommGroup α
    inst✝ : CompleteSpace α
    f : Nat → α
    l : Real
    hl₁ : LT.lt l 1
    hf : Filter.Eventually (fun n => Ne (f n) 0) Filter.atTop
    h : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (f (HAdd.hAdd n 1))) (Norm.n …
    ⊢ Summable f
  -/
  rcases exists_between hl₁ with ⟨r, hr₀, hr₁⟩
  /-
    case intro.intro
    α : Type u_2
    inst✝¹ : NormedAddCommGroup α
    inst✝ : CompleteSpace α
    f : Nat → α
    l : Real
    hl₁ : LT.lt l 1
    hf : Filter.Eventually (fun n => Ne (f n) 0) Filter.atTop
    h : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (f (HAdd.hAdd n 1))) (Norm.n …
    r : Real
    hr₀ : LT.lt l r
    hr₁ : LT.lt r 1
    ⊢ Summable f
  -/
  refine summable_of_ratio_norm_eventually_le hr₁ ?_
  /-
    case intro.intro
    α : Type u_2
    inst✝¹ : NormedAddCommGroup α
    inst✝ : CompleteSpace α
    f : Nat → α
    l : Real
    hl₁ : LT.lt l 1
    hf : Filter.Eventually (fun n => Ne (f n) 0) Filter.atTop
    h : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (f (HAdd.hAdd n 1))) (Norm.n …
    r : Real
    hr₀ : LT.lt l r
    hr₁ : LT.lt r 1
    ⊢ Filter.Eventually (fun n => LE.le (Norm.norm (f (HAdd.hAdd n 1))) (HMul.hMul …
  -/
  filter_upwards [h.eventually_le_const hr₀, hf] with _ _ h₁
  /-
    case h
    α : Type u_2
    inst✝¹ : NormedAddCommGroup α
    inst✝ : CompleteSpace α
    f : Nat → α
    l : Real
    hl₁ : LT.lt l 1
    hf : Filter.Eventually (fun n => Ne (f n) 0) Filter.atTop
    h : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (f (HAdd.hAdd n 1))) (Norm.n …
    r : Real
    hr₀ : LT.lt l r
    hr₁ : LT.lt r 1
    a✝¹ : Nat
    a✝ : LE.le (HDiv.hDiv (Norm.norm (f (HAdd.hAdd a✝¹ 1))) (Norm.norm (f a✝¹))) r
    h₁ : Ne (f a✝¹) 0
    ⊢ LE.le (Norm.norm (f (HAdd.hAdd a✝¹ 1))) (HMul.hMul r (Norm.norm (f a✝¹)))
  -/
  rwa [← div_le_iff₀ (norm_pos_iff.mpr h₁)]
  /-
    🎉 no goals
  -/


theorem not_summable_of_ratio_norm_eventually_ge {α : Type*} [SeminormedAddCommGroup α] {f : ℕ → α}
    {r : ℝ} (hr : 1 < r) (hf : ∃ᶠ n in atTop, ‖f n‖ ≠ 0)
    (h : ∀ᶠ n in atTop, r * ‖f n‖ ≤ ‖f (n + 1)‖) : ¬Summable f := by
  /-
    α : Type u_2
    inst✝ : SeminormedAddCommGroup α
    f : Nat → α
    r : Real
    hr : LT.lt 1 r
    hf : Filter.Frequently (fun n => Ne (Norm.norm (f n)) 0) Filter.atTop
    h : Filter.Eventually (fun n => LE.le (HMul.hMul r (Norm.norm (f n))) (Norm.no …
    ⊢ Not (Summable f)
  -/
  rw [eventually_atTop] at h
  /-
    α : Type u_2
    inst✝ : SeminormedAddCommGroup α
    f : Nat → α
    r : Real
    hr : LT.lt 1 r
    hf : Filter.Frequently (fun n => Ne (Norm.norm (f n)) 0) Filter.atTop
    h : Exists fun a => ∀ (b : Nat), GE.ge b a → LE.le (HMul.hMul r (Norm.norm (f  …
    ⊢ Not (Summable f)
  -/
  rcases h with ⟨N₀, hN₀⟩
  /-
    case intro
    α : Type u_2
    inst✝ : SeminormedAddCommGroup α
    f : Nat → α
    r : Real
    hr : LT.lt 1 r
    hf : Filter.Frequently (fun n => Ne (Norm.norm (f n)) 0) Filter.atTop
    N₀ : Nat
    hN₀ : ∀ (b : Nat), GE.ge b N₀ → LE.le (HMul.hMul r (Norm.norm (f b))) (Norm.no …
    ⊢ Not (Summable f)
  -/
  rw [frequently_atTop] at hf
  /-
    case intro
    α : Type u_2
    inst✝ : SeminormedAddCommGroup α
    f : Nat → α
    r : Real
    hr : LT.lt 1 r
    hf : ∀ (a : Nat), Exists fun b => And (GE.ge b a) (Ne (Norm.norm (f b)) 0)
    N₀ : Nat
    hN₀ : ∀ (b : Nat), GE.ge b N₀ → LE.le (HMul.hMul r (Norm.norm (f b))) (Norm.no …
    ⊢ Not (Summable f)
  -/
  rcases hf N₀ with ⟨N, hNN₀ : N₀ ≤ N, hN⟩
  /-
    case intro.intro.intro
    α : Type u_2
    inst✝ : SeminormedAddCommGroup α
    f : Nat → α
    r : Real
    hr : LT.lt 1 r
    hf : ∀ (a : Nat), Exists fun b => And (GE.ge b a) (Ne (Norm.norm (f b)) 0)
    N₀ : Nat
    hN₀ : ∀ (b : Nat), GE.ge b N₀ → LE.le (HMul.hMul r (Norm.norm (f b))) (Norm.no …
    N : Nat
    hNN₀ : LE.le N₀ N
    hN : Ne (Norm.norm (f N)) 0
    ⊢ Not (Summable f)
  -/
  rw [← @summable_nat_add_iff α _ _ _ _ N]
  refine mt Summable.tendsto_atTop_zero
    fun h' ↦ not_tendsto_atTop_of_tendsto_nhds (tendsto_norm_zero.comp h') ?_
  /-
    case intro.intro.intro
    α : Type u_2
    inst✝ : SeminormedAddCommGroup α
    f : Nat → α
    r : Real
    hr : LT.lt 1 r
    hf : ∀ (a : Nat), Exists fun b => And (GE.ge b a) (Ne (Norm.norm (f b)) 0)
    N₀ : Nat
    hN₀ : ∀ (b : Nat), GE.ge b N₀ → LE.le (HMul.hMul r (Norm.norm (f b))) (Norm.no …
    N : Nat
    hNN₀ : LE.le N₀ N
    hN : Ne (Norm.norm (f N)) 0
    h' : Filter.Tendsto (fun n => f (HAdd.hAdd n N)) Filter.atTop (nhds 0)
    ⊢ Filter.Tendsto (Function.comp (fun a => Norm.norm a) fun n => f (HAdd.hAdd n …
  -/
  convert tendsto_atTop_of_geom_le _ hr _
    /-
      case intro.intro.intro.convert_2
      α : Type u_2
      inst✝ : SeminormedAddCommGroup α
      f : Nat → α
      r : Real
      hr : LT.lt 1 r
      hf : ∀ (a : Nat), Exists fun b => And (GE.ge b a) (Ne (Norm.norm (f b)) 0)
      N₀ : Nat
      hN₀ : ∀ (b : Nat), GE.ge b N₀ → LE.le (HMul.hMul r (Norm.norm (f b))) (Norm.no …
      N : Nat
      hNN₀ : LE.le N₀ N
      hN : Ne (Norm.norm (f N)) 0
      h' : Filter.Tendsto (fun n => f (HAdd.hAdd n N)) Filter.atTop (nhds 0)
      ⊢ LT.lt 0 (Function.comp (fun a => Norm.norm a) (fun n => f (HAdd.hAdd n N)) 0)
    -/
  · refine lt_of_le_of_ne (norm_nonneg _) ?_
    /-
      case intro.intro.intro.convert_2
      α : Type u_2
      inst✝ : SeminormedAddCommGroup α
      f : Nat → α
      r : Real
      hr : LT.lt 1 r
      hf : ∀ (a : Nat), Exists fun b => And (GE.ge b a) (Ne (Norm.norm (f b)) 0)
      N₀ : Nat
      hN₀ : ∀ (b : Nat), GE.ge b N₀ → LE.le (HMul.hMul r (Norm.norm (f b))) (Norm.no …
      N : Nat
      hNN₀ : LE.le N₀ N
      hN : Ne (Norm.norm (f N)) 0
      h' : Filter.Tendsto (fun n => f (HAdd.hAdd n N)) Filter.atTop (nhds 0)
      ⊢ Ne 0 (Function.comp (fun a => Norm.norm a) (fun n => f (HAdd.hAdd n N)) 0)
    -/
    intro h''
    /-
      case intro.intro.intro.convert_2
      α : Type u_2
      inst✝ : SeminormedAddCommGroup α
      f : Nat → α
      r : Real
      hr : LT.lt 1 r
      hf : ∀ (a : Nat), Exists fun b => And (GE.ge b a) (Ne (Norm.norm (f b)) 0)
      N₀ : Nat
      hN₀ : ∀ (b : Nat), GE.ge b N₀ → LE.le (HMul.hMul r (Norm.norm (f b))) (Norm.no …
      N : Nat
      hNN₀ : LE.le N₀ N
      hN : Ne (Norm.norm (f N)) 0
      h' : Filter.Tendsto (fun n => f (HAdd.hAdd n N)) Filter.atTop (nhds 0)
      h'' : Eq 0 (Function.comp (fun a => Norm.norm a) (fun n => f (HAdd.hAdd n N)) 0)
      ⊢ False
    -/
    specialize hN₀ N hNN₀
    /-
      case intro.intro.intro.convert_2
      α : Type u_2
      inst✝ : SeminormedAddCommGroup α
      f : Nat → α
      r : Real
      hr : LT.lt 1 r
      hf : ∀ (a : Nat), Exists fun b => And (GE.ge b a) (Ne (Norm.norm (f b)) 0)
      N₀ N : Nat
      hNN₀ : LE.le N₀ N
      hN : Ne (Norm.norm (f N)) 0
      h' : Filter.Tendsto (fun n => f (HAdd.hAdd n N)) Filter.atTop (nhds 0)
      h'' : Eq 0 (Function.comp (fun a => Norm.norm a) (fun n => f (HAdd.hAdd n N)) 0)
      hN₀ : LE.le (HMul.hMul r (Norm.norm (f N))) (Norm.norm (f (HAdd.hAdd N 1)))
      ⊢ False
    -/
    simp only [comp_apply, zero_add] at h''
    /-
      case intro.intro.intro.convert_2
      α : Type u_2
      inst✝ : SeminormedAddCommGroup α
      f : Nat → α
      r : Real
      hr : LT.lt 1 r
      hf : ∀ (a : Nat), Exists fun b => And (GE.ge b a) (Ne (Norm.norm (f b)) 0)
      N₀ N : Nat
      hNN₀ : LE.le N₀ N
      hN : Ne (Norm.norm (f N)) 0
      h' : Filter.Tendsto (fun n => f (HAdd.hAdd n N)) Filter.atTop (nhds 0)
      hN₀ : LE.le (HMul.hMul r (Norm.norm (f N))) (Norm.norm (f (HAdd.hAdd N 1)))
      h'' : Eq 0 (Norm.norm (f N))
      ⊢ False
    -/
    exact hN h''.symm
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.convert_3
      α : Type u_2
      inst✝ : SeminormedAddCommGroup α
      f : Nat → α
      r : Real
      hr : LT.lt 1 r
      hf : ∀ (a : Nat), Exists fun b => And (GE.ge b a) (Ne (Norm.norm (f b)) 0)
      N₀ : Nat
      hN₀ : ∀ (b : Nat), GE.ge b N₀ → LE.le (HMul.hMul r (Norm.norm (f b))) (Norm.no …
      N : Nat
      hNN₀ : LE.le N₀ N
      hN : Ne (Norm.norm (f N)) 0
      h' : Filter.Tendsto (fun n => f (HAdd.hAdd n N)) Filter.atTop (nhds 0)
      ⊢ ∀ (n : Nat), LE.le (HMul.hMul r (Function.comp (fun a => Norm.norm a) (fun n …
    -/
  · intro i
    /-
      case intro.intro.intro.convert_3
      α : Type u_2
      inst✝ : SeminormedAddCommGroup α
      f : Nat → α
      r : Real
      hr : LT.lt 1 r
      hf : ∀ (a : Nat), Exists fun b => And (GE.ge b a) (Ne (Norm.norm (f b)) 0)
      N₀ : Nat
      hN₀ : ∀ (b : Nat), GE.ge b N₀ → LE.le (HMul.hMul r (Norm.norm (f b))) (Norm.no …
      N : Nat
      hNN₀ : LE.le N₀ N
      hN : Ne (Norm.norm (f N)) 0
      h' : Filter.Tendsto (fun n => f (HAdd.hAdd n N)) Filter.atTop (nhds 0)
      i : Nat
      ⊢ LE.le (HMul.hMul r (Function.comp (fun a => Norm.norm a) (fun n => f (HAdd.h …
    -/
    dsimp only [comp_apply]
    /-
      case intro.intro.intro.convert_3
      α : Type u_2
      inst✝ : SeminormedAddCommGroup α
      f : Nat → α
      r : Real
      hr : LT.lt 1 r
      hf : ∀ (a : Nat), Exists fun b => And (GE.ge b a) (Ne (Norm.norm (f b)) 0)
      N₀ : Nat
      hN₀ : ∀ (b : Nat), GE.ge b N₀ → LE.le (HMul.hMul r (Norm.norm (f b))) (Norm.no …
      N : Nat
      hNN₀ : LE.le N₀ N
      hN : Ne (Norm.norm (f N)) 0
      h' : Filter.Tendsto (fun n => f (HAdd.hAdd n N)) Filter.atTop (nhds 0)
      i : Nat
      ⊢ LE.le (HMul.hMul r (Norm.norm (f (HAdd.hAdd i N)))) (Norm.norm (f (HAdd.hAdd …
    -/
    convert hN₀ (i + N) (hNN₀.trans (N.le_add_left i)) using 3
    /-
      case h.e'_4.h.e'_3.h.e'_1
      α : Type u_2
      inst✝ : SeminormedAddCommGroup α
      f : Nat → α
      r : Real
      hr : LT.lt 1 r
      hf : ∀ (a : Nat), Exists fun b => And (GE.ge b a) (Ne (Norm.norm (f b)) 0)
      N₀ : Nat
      hN₀ : ∀ (b : Nat), GE.ge b N₀ → LE.le (HMul.hMul r (Norm.norm (f b))) (Norm.no …
      N : Nat
      hNN₀ : LE.le N₀ N
      hN : Ne (Norm.norm (f N)) 0
      h' : Filter.Tendsto (fun n => f (HAdd.hAdd n N)) Filter.atTop (nhds 0)
      i : Nat
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd i 1) N) (HAdd.hAdd (HAdd.hAdd i N) 1)
    -/
    ac_rfl
    /-
      🎉 no goals
    -/


theorem not_summable_of_ratio_test_tendsto_gt_one {α : Type*} [SeminormedAddCommGroup α]
    {f : ℕ → α} {l : ℝ} (hl : 1 < l) (h : Tendsto (fun n ↦ ‖f (n + 1)‖ / ‖f n‖) atTop (𝓝 l)) :
    ¬Summable f := by
  have key : ∀ᶠ n in atTop, ‖f n‖ ≠ 0 := by
    filter_upwards [h.eventually_const_le hl] with _ hn hc
    rw [hc, _root_.div_zero] at hn
    linarith
  /-
    α : Type u_2
    inst✝ : SeminormedAddCommGroup α
    f : Nat → α
    l : Real
    hl : LT.lt 1 l
    h : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (f (HAdd.hAdd n 1))) (Norm.n …
    key : Filter.Eventually (fun n => Ne (Norm.norm (f n)) 0) Filter.atTop
    ⊢ Not (Summable f)
  -/
  rcases exists_between hl with ⟨r, hr₀, hr₁⟩
  /-
    case intro.intro
    α : Type u_2
    inst✝ : SeminormedAddCommGroup α
    f : Nat → α
    l : Real
    hl : LT.lt 1 l
    h : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (f (HAdd.hAdd n 1))) (Norm.n …
    key : Filter.Eventually (fun n => Ne (Norm.norm (f n)) 0) Filter.atTop
    r : Real
    hr₀ : LT.lt 1 r
    hr₁ : LT.lt r l
    ⊢ Not (Summable f)
  -/
  refine not_summable_of_ratio_norm_eventually_ge hr₀ key.frequently ?_
  /-
    case intro.intro
    α : Type u_2
    inst✝ : SeminormedAddCommGroup α
    f : Nat → α
    l : Real
    hl : LT.lt 1 l
    h : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (f (HAdd.hAdd n 1))) (Norm.n …
    key : Filter.Eventually (fun n => Ne (Norm.norm (f n)) 0) Filter.atTop
    r : Real
    hr₀ : LT.lt 1 r
    hr₁ : LT.lt r l
    ⊢ Filter.Eventually (fun n => LE.le (HMul.hMul r (Norm.norm (f n))) (Norm.norm …
  -/
  filter_upwards [h.eventually_const_le hr₁, key] with _ _ h₁
  /-
    case h
    α : Type u_2
    inst✝ : SeminormedAddCommGroup α
    f : Nat → α
    l : Real
    hl : LT.lt 1 l
    h : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (f (HAdd.hAdd n 1))) (Norm.n …
    key : Filter.Eventually (fun n => Ne (Norm.norm (f n)) 0) Filter.atTop
    r : Real
    hr₀ : LT.lt 1 r
    hr₁ : LT.lt r l
    a✝¹ : Nat
    a✝ : LE.le r (HDiv.hDiv (Norm.norm (f (HAdd.hAdd a✝¹ 1))) (Norm.norm (f a✝¹)))
    h₁ : Ne (Norm.norm (f a✝¹)) 0
    ⊢ LE.le (HMul.hMul r (Norm.norm (f a✝¹))) (Norm.norm (f (HAdd.hAdd a✝¹ 1)))
  -/
  rwa [← le_div_iff₀ (lt_of_le_of_ne (norm_nonneg _) h₁.symm)]
  /-
    🎉 no goals
  -/


/-- If a power series converges at `w`, it converges absolutely at all `z` of smaller norm. -/
theorem summable_powerSeries_of_norm_lt {w z : α}
    (h : CauchySeq fun n ↦ ∑ i ∈ range n, f i * w ^ i) (hz : ‖z‖ < ‖w‖) :
    Summable fun n ↦ f n * z ^ n := by
  /-
    α : Type u_1
    inst✝¹ : NormedDivisionRing α
    inst✝ : CompleteSpace α
    f : Nat → α
    w z : α
    h : CauchySeq fun n => (Finset.range n).sum fun i => HMul.hMul (f i) (HPow.hPo …
    hz : LT.lt (Norm.norm z) (Norm.norm w)
    ⊢ Summable fun n => HMul.hMul (f n) (HPow.hPow z n)
  -/
  have hw : 0 < ‖w‖ := (norm_nonneg z).trans_lt hz
  /-
    α : Type u_1
    inst✝¹ : NormedDivisionRing α
    inst✝ : CompleteSpace α
    f : Nat → α
    w z : α
    h : CauchySeq fun n => (Finset.range n).sum fun i => HMul.hMul (f i) (HPow.hPo …
    hz : LT.lt (Norm.norm z) (Norm.norm w)
    hw : LT.lt 0 (Norm.norm w)
    ⊢ Summable fun n => HMul.hMul (f n) (HPow.hPow z n)
  -/
  obtain ⟨C, hC⟩ := exists_norm_le_of_cauchySeq h
  /-
    case intro
    α : Type u_1
    inst✝¹ : NormedDivisionRing α
    inst✝ : CompleteSpace α
    f : Nat → α
    w z : α
    h : CauchySeq fun n => (Finset.range n).sum fun i => HMul.hMul (f i) (HPow.hPo …
    hz : LT.lt (Norm.norm z) (Norm.norm w)
    hw : LT.lt 0 (Norm.norm w)
    C : Real
    hC : ∀ (n : Nat), LE.le (Norm.norm (HMul.hMul (f n) (HPow.hPow w n))) C
    ⊢ Summable fun n => HMul.hMul (f n) (HPow.hPow z n)
  -/
  rw [summable_iff_cauchySeq_finset]
  refine cauchySeq_finset_of_geometric_bound (r := ‖z‖ / ‖w‖) (C := C) ((div_lt_one hw).mpr hz)
    (fun n ↦ ?_)
  /-
    case intro
    α : Type u_1
    inst✝¹ : NormedDivisionRing α
    inst✝ : CompleteSpace α
    f : Nat → α
    w z : α
    h : CauchySeq fun n => (Finset.range n).sum fun i => HMul.hMul (f i) (HPow.hPo …
    hz : LT.lt (Norm.norm z) (Norm.norm w)
    hw : LT.lt 0 (Norm.norm w)
    C : Real
    hC : ∀ (n : Nat), LE.le (Norm.norm (HMul.hMul (f n) (HPow.hPow w n))) C
    n : Nat
    ⊢ LE.le (Norm.norm (HMul.hMul (f n) (HPow.hPow z n))) (HMul.hMul C (HPow.hPow  …
  -/
  rw [norm_mul, norm_pow, div_pow, ← mul_comm_div]
  /-
    case intro
    α : Type u_1
    inst✝¹ : NormedDivisionRing α
    inst✝ : CompleteSpace α
    f : Nat → α
    w z : α
    h : CauchySeq fun n => (Finset.range n).sum fun i => HMul.hMul (f i) (HPow.hPo …
    hz : LT.lt (Norm.norm z) (Norm.norm w)
    hw : LT.lt 0 (Norm.norm w)
    C : Real
    hC : ∀ (n : Nat), LE.le (Norm.norm (HMul.hMul (f n) (HPow.hPow w n))) C
    n : Nat
    ⊢ LE.le (HMul.hMul (Norm.norm (f n)) (HPow.hPow (Norm.norm z) n)) (HMul.hMul ( …
  -/
  conv at hC => enter [n]; rw [norm_mul, norm_pow, ← _root_.le_div_iff₀ (by positivity)]
  /-
    case intro
    α : Type u_1
    inst✝¹ : NormedDivisionRing α
    inst✝ : CompleteSpace α
    f : Nat → α
    w z : α
    h : CauchySeq fun n => (Finset.range n).sum fun i => HMul.hMul (f i) (HPow.hPo …
    hz : LT.lt (Norm.norm z) (Norm.norm w)
    hw : LT.lt 0 (Norm.norm w)
    C : Real
    hC : ∀ (n : Nat), LE.le (Norm.norm (f n)) (HDiv.hDiv C (HPow.hPow (Norm.norm w …
    n : Nat
    ⊢ LE.le (HMul.hMul (Norm.norm (f n)) (HPow.hPow (Norm.norm z) n)) (HMul.hMul ( …
  -/
  exact mul_le_mul_of_nonneg_right (hC n) (pow_nonneg (norm_nonneg z) n)
  /-
    🎉 no goals
  -/


/-- If a power series converges at 1, it converges absolutely at all `z` of smaller norm. -/
theorem summable_powerSeries_of_norm_lt_one {z : α}
    (h : CauchySeq fun n ↦ ∑ i ∈ range n, f i) (hz : ‖z‖ < 1) :
    Summable fun n ↦ f n * z ^ n :=
                                               /-
                                                 α : Type u_1
                                                 inst✝¹ : NormedDivisionRing α
                                                 inst✝ : CompleteSpace α
                                                 f : Nat → α
                                                 z : α
                                                 h : CauchySeq fun n => (Finset.range n).sum fun i => f i
                                                 hz : LT.lt (Norm.norm z) 1
                                                 ⊢ CauchySeq fun n => (Finset.range n).sum fun i => HMul.hMul (f i) (HPow.hPow  …
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
  summable_powerSeries_of_norm_lt (w := 1) (by simp [h]) (by simp [hz])
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- **Dirichlet's test** for monotone sequences. -/
theorem Monotone.cauchySeq_series_mul_of_tendsto_zero_of_bounded (hfa : Monotone f)
    (hf0 : Tendsto f atTop (𝓝 0)) (hgb : ∀ n, ‖∑ i ∈ range n, z i‖ ≤ b) :
    CauchySeq fun n ↦ ∑ i ∈ range n, f i • z i := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    b : Real
    f : Nat → Real
    z : Nat → E
    hfa : Monotone f
    hf0 : Filter.Tendsto f Filter.atTop (nhds 0)
    hgb : ∀ (n : Nat), LE.le (Norm.norm ((Finset.range n).sum fun i => z i)) b
    ⊢ CauchySeq fun n => (Finset.range n).sum fun i => HSMul.hSMul (f i) (z i)
  -/
  rw [← cauchySeq_shift 1]
  simp_rw [Finset.sum_range_by_parts _ _ (Nat.succ _), sub_eq_add_neg, Nat.succ_sub_succ_eq_sub,
    tsub_zero]
  apply (NormedField.tendsto_zero_smul_of_tendsto_zero_of_bounded hf0
    ⟨b, eventually_map.mpr <| Eventually.of_forall fun n ↦ hgb <| n + 1⟩).cauchySeq.add
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    b : Real
    f : Nat → Real
    z : Nat → E
    hfa : Monotone f
    hf0 : Filter.Tendsto f Filter.atTop (nhds 0)
    hgb : ∀ (n : Nat), LE.le (Norm.norm ((Finset.range n).sum fun i => z i)) b
    ⊢ CauchySeq fun n => Neg.neg ((Finset.range n).sum fun x => HSMul.hSMul (HAdd. …
  -/
  refine CauchySeq.neg ?_
  refine cauchySeq_range_of_norm_bounded _ ?_
    (fun n ↦ ?_ : ∀ n, ‖(f (n + 1) + -f n) • (Finset.range (n + 1)).sum z‖ ≤ b * |f (n + 1) - f n|)
    /-
      case refine_1
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      b : Real
      f : Nat → Real
      z : Nat → E
      hfa : Monotone f
      hf0 : Filter.Tendsto f Filter.atTop (nhds 0)
      hgb : ∀ (n : Nat), LE.le (Norm.norm ((Finset.range n).sum fun i => z i)) b
      ⊢ CauchySeq fun n => (Finset.range n).sum fun i => HMul.hMul b (abs (HSub.hSub …
    -/
  · simp_rw [abs_of_nonneg (sub_nonneg_of_le (hfa (Nat.le_succ _))), ← mul_sum]
    /-
      case refine_1
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      b : Real
      f : Nat → Real
      z : Nat → E
      hfa : Monotone f
      hf0 : Filter.Tendsto f Filter.atTop (nhds 0)
      hgb : ∀ (n : Nat), LE.le (Norm.norm ((Finset.range n).sum fun i => z i)) b
      ⊢ CauchySeq fun n => HMul.hMul b ((Finset.range n).sum fun i => HSub.hSub (f i …
    -/
    apply Real.uniformContinuous_const_mul.comp_cauchySeq
    /-
      case refine_1
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      b : Real
      f : Nat → Real
      z : Nat → E
      hfa : Monotone f
      hf0 : Filter.Tendsto f Filter.atTop (nhds 0)
      hgb : ∀ (n : Nat), LE.le (Norm.norm ((Finset.range n).sum fun i => z i)) b
      ⊢ CauchySeq fun n => (Finset.range n).sum fun i => HSub.hSub (f i.succ) (f i)
    -/
    simp_rw [sum_range_sub, sub_eq_add_neg]
    /-
      case refine_1
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      b : Real
      f : Nat → Real
      z : Nat → E
      hfa : Monotone f
      hf0 : Filter.Tendsto f Filter.atTop (nhds 0)
      hgb : ∀ (n : Nat), LE.le (Norm.norm ((Finset.range n).sum fun i => z i)) b
      ⊢ CauchySeq fun n => HAdd.hAdd (f n) (Neg.neg (f 0))
    -/
    exact (Tendsto.cauchySeq hf0).add_const
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      b : Real
      f : Nat → Real
      z : Nat → E
      hfa : Monotone f
      hf0 : Filter.Tendsto f Filter.atTop (nhds 0)
      hgb : ∀ (n : Nat), LE.le (Norm.norm ((Finset.range n).sum fun i => z i)) b
      n : Nat
      ⊢ LE.le (Norm.norm (HSMul.hSMul (HAdd.hAdd (f (HAdd.hAdd n 1)) (Neg.neg (f n)) …
    -/
  · rw [norm_smul, mul_comm]
    /-
      case refine_2
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      b : Real
      f : Nat → Real
      z : Nat → E
      hfa : Monotone f
      hf0 : Filter.Tendsto f Filter.atTop (nhds 0)
      hgb : ∀ (n : Nat), LE.le (Norm.norm ((Finset.range n).sum fun i => z i)) b
      n : Nat
      ⊢ LE.le (HMul.hMul (Norm.norm ((Finset.range (HAdd.hAdd n 1)).sum z)) (Norm.no …
    -/
    exact mul_le_mul_of_nonneg_right (hgb _) (abs_nonneg _)
    /-
      🎉 no goals
    -/


/-- **Dirichlet's test** for antitone sequences. -/
theorem Antitone.cauchySeq_series_mul_of_tendsto_zero_of_bounded (hfa : Antitone f)
    (hf0 : Tendsto f atTop (𝓝 0)) (hzb : ∀ n, ‖∑ i ∈ range n, z i‖ ≤ b) :
    CauchySeq fun n ↦ ∑ i ∈ range n, f i • z i := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    b : Real
    f : Nat → Real
    z : Nat → E
    hfa : Antitone f
    hf0 : Filter.Tendsto f Filter.atTop (nhds 0)
    hzb : ∀ (n : Nat), LE.le (Norm.norm ((Finset.range n).sum fun i => z i)) b
    ⊢ CauchySeq fun n => (Finset.range n).sum fun i => HSMul.hSMul (f i) (z i)
  -/
  have hfa' : Monotone fun n ↦ -f n := fun _ _ hab ↦ neg_le_neg <| hfa hab
  have hf0' : Tendsto (fun n ↦ -f n) atTop (𝓝 0) := by
    convert hf0.neg
    norm_num
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    b : Real
    f : Nat → Real
    z : Nat → E
    hfa : Antitone f
    hf0 : Filter.Tendsto f Filter.atTop (nhds 0)
    hzb : ∀ (n : Nat), LE.le (Norm.norm ((Finset.range n).sum fun i => z i)) b
    hfa' : Monotone fun n => Neg.neg (f n)
    hf0' : Filter.Tendsto (fun n => Neg.neg (f n)) Filter.atTop (nhds 0)
    ⊢ CauchySeq fun n => (Finset.range n).sum fun i => HSMul.hSMul (f i) (z i)
  -/
  convert (hfa'.cauchySeq_series_mul_of_tendsto_zero_of_bounded hf0' hzb).neg
  /-
    case h.e'_5.h
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    b : Real
    f : Nat → Real
    z : Nat → E
    hfa : Antitone f
    hf0 : Filter.Tendsto f Filter.atTop (nhds 0)
    hzb : ∀ (n : Nat), LE.le (Norm.norm ((Finset.range n).sum fun i => z i)) b
    hfa' : Monotone fun n => Neg.neg (f n)
    hf0' : Filter.Tendsto (fun n => Neg.neg (f n)) Filter.atTop (nhds 0)
    x✝ : Nat
    ⊢ Eq ((Finset.range x✝).sum fun i => HSMul.hSMul (f i) (z i)) (Neg.neg (fun n  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem norm_sum_neg_one_pow_le (n : ℕ) : ‖∑ i ∈ range n, (-1 : ℝ) ^ i‖ ≤ 1 := by
  /-
    n : Nat
    ⊢ LE.le (Norm.norm ((Finset.range n).sum fun i => HPow.hPow (-1) i)) 1
  -/
  rw [neg_one_geom_sum]
  /-
    n : Nat
    ⊢ LE.le (Norm.norm (ite (Even n) 0 1)) 1
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> norm_num
                /-
                  🎉 no goals
                -/


/-- The **alternating series test** for monotone sequences.
See also `Monotone.tendsto_alternating_series_of_tendsto_zero`. -/
theorem Monotone.cauchySeq_alternating_series_of_tendsto_zero (hfa : Monotone f)
    (hf0 : Tendsto f atTop (𝓝 0)) : CauchySeq fun n ↦ ∑ i ∈ range n, (-1) ^ i * f i := by
  /-
    f : Nat → Real
    hfa : Monotone f
    hf0 : Filter.Tendsto f Filter.atTop (nhds 0)
    ⊢ CauchySeq fun n => (Finset.range n).sum fun i => HMul.hMul (HPow.hPow (-1) i …
  -/
  simp_rw [mul_comm]
  /-
    f : Nat → Real
    hfa : Monotone f
    hf0 : Filter.Tendsto f Filter.atTop (nhds 0)
    ⊢ CauchySeq fun n => (Finset.range n).sum fun x => HMul.hMul (f x) (HPow.hPow  …
  -/
  exact hfa.cauchySeq_series_mul_of_tendsto_zero_of_bounded hf0 norm_sum_neg_one_pow_le
  /-
    🎉 no goals
  -/


/-- The **alternating series test** for monotone sequences. -/
theorem Monotone.tendsto_alternating_series_of_tendsto_zero (hfa : Monotone f)
    (hf0 : Tendsto f atTop (𝓝 0)) :
    ∃ l, Tendsto (fun n ↦ ∑ i ∈ range n, (-1) ^ i * f i) atTop (𝓝 l) :=
  cauchySeq_tendsto_of_complete <| hfa.cauchySeq_alternating_series_of_tendsto_zero hf0


/-- The **alternating series test** for antitone sequences.
See also `Antitone.tendsto_alternating_series_of_tendsto_zero`. -/
theorem Antitone.cauchySeq_alternating_series_of_tendsto_zero (hfa : Antitone f)
    (hf0 : Tendsto f atTop (𝓝 0)) : CauchySeq fun n ↦ ∑ i ∈ range n, (-1) ^ i * f i := by
  /-
    f : Nat → Real
    hfa : Antitone f
    hf0 : Filter.Tendsto f Filter.atTop (nhds 0)
    ⊢ CauchySeq fun n => (Finset.range n).sum fun i => HMul.hMul (HPow.hPow (-1) i …
  -/
  simp_rw [mul_comm]
  /-
    f : Nat → Real
    hfa : Antitone f
    hf0 : Filter.Tendsto f Filter.atTop (nhds 0)
    ⊢ CauchySeq fun n => (Finset.range n).sum fun x => HMul.hMul (f x) (HPow.hPow  …
  -/
  exact hfa.cauchySeq_series_mul_of_tendsto_zero_of_bounded hf0 norm_sum_neg_one_pow_le
  /-
    🎉 no goals
  -/


/-- The **alternating series test** for antitone sequences. -/
theorem Antitone.tendsto_alternating_series_of_tendsto_zero (hfa : Antitone f)
    (hf0 : Tendsto f atTop (𝓝 0)) :
    ∃ l, Tendsto (fun n ↦ ∑ i ∈ range n, (-1) ^ i * f i) atTop (𝓝 l) :=
  cauchySeq_tendsto_of_complete <| hfa.cauchySeq_alternating_series_of_tendsto_zero hf0


/-- Partial sums of an alternating monotone series with an even number of terms provide
upper bounds on the limit. -/
theorem Monotone.tendsto_le_alternating_series
    (hfl : Tendsto (fun n ↦ ∑ i ∈ range n, (-1) ^ i * f i) atTop (𝓝 l))
    (hfm : Monotone f) (k : ℕ) : l ≤ ∑ i ∈ range (2 * k), (-1) ^ i * f i := by
  have ha : Antitone (fun n ↦ ∑ i ∈ range (2 * n), (-1) ^ i * f i) := by
    refine antitone_nat_of_succ_le (fun n ↦ ?_)
    rw [show 2 * (n + 1) = 2 * n + 1 + 1 by ring, sum_range_succ, sum_range_succ]
    simp_rw [_root_.pow_succ', show (-1 : E) ^ (2 * n) = 1 by simp, neg_one_mul, one_mul,
      ← sub_eq_add_neg, sub_le_iff_le_add]
    gcongr
    exact hfm (by omega)
  /-
    E : Type u_2
    inst✝² : OrderedRing E
    inst✝¹ : TopologicalSpace E
    inst✝ : OrderClosedTopology E
    l : E
    f : Nat → E
    hfl : Filter.Tendsto (fun n => (Finset.range n).sum fun i => HMul.hMul (HPow.h …
    hfm : Monotone f
    k : Nat
    ha : Antitone fun n => (Finset.range (HMul.hMul 2 n)).sum fun i => HMul.hMul ( …
    ⊢ LE.le l ((Finset.range (HMul.hMul 2 k)).sum fun i => HMul.hMul (HPow.hPow (- …
  -/
  exact ha.le_of_tendsto (hfl.comp (tendsto_atTop_mono (fun n ↦ by dsimp; omega) tendsto_id)) _
  /-
    🎉 no goals
  -/


/-- Partial sums of an alternating monotone series with an odd number of terms provide
lower bounds on the limit. -/
theorem Monotone.alternating_series_le_tendsto
    (hfl : Tendsto (fun n ↦ ∑ i ∈ range n, (-1) ^ i * f i) atTop (𝓝 l))
    (hfm : Monotone f) (k : ℕ) : ∑ i ∈ range (2 * k + 1), (-1) ^ i * f i ≤ l := by
  have hm : Monotone (fun n ↦ ∑ i ∈ range (2 * n + 1), (-1) ^ i * f i) := by
    refine monotone_nat_of_le_succ (fun n ↦ ?_)
    rw [show 2 * (n + 1) = 2 * n + 1 + 1 by ring,
      sum_range_succ _ (2 * n + 1 + 1), sum_range_succ _ (2 * n + 1)]
    simp_rw [_root_.pow_succ', show (-1 : E) ^ (2 * n) = 1 by simp, neg_one_mul, neg_neg, one_mul,
      ← sub_eq_add_neg, sub_add_eq_add_sub, le_sub_iff_add_le]
    gcongr
    exact hfm (by omega)
  /-
    E : Type u_2
    inst✝² : OrderedRing E
    inst✝¹ : TopologicalSpace E
    inst✝ : OrderClosedTopology E
    l : E
    f : Nat → E
    hfl : Filter.Tendsto (fun n => (Finset.range n).sum fun i => HMul.hMul (HPow.h …
    hfm : Monotone f
    k : Nat
    hm : Monotone fun n => (Finset.range (HAdd.hAdd (HMul.hMul 2 n) 1)).sum fun i  …
    ⊢ LE.le ((Finset.range (HAdd.hAdd (HMul.hMul 2 k) 1)).sum fun i => HMul.hMul ( …
  -/
  exact hm.ge_of_tendsto (hfl.comp (tendsto_atTop_mono (fun n ↦ by dsimp; omega) tendsto_id)) _
  /-
    🎉 no goals
  -/


/-- Partial sums of an alternating antitone series with an even number of terms provide
lower bounds on the limit. -/
theorem Antitone.alternating_series_le_tendsto
    (hfl : Tendsto (fun n ↦ ∑ i ∈ range n, (-1) ^ i * f i) atTop (𝓝 l))
    (hfa : Antitone f) (k : ℕ) : ∑ i ∈ range (2 * k), (-1) ^ i * f i ≤ l := by
  have hm : Monotone (fun n ↦ ∑ i ∈ range (2 * n), (-1) ^ i * f i) := by
    refine monotone_nat_of_le_succ (fun n ↦ ?_)
    rw [show 2 * (n + 1) = 2 * n + 1 + 1 by ring, sum_range_succ, sum_range_succ]
    simp_rw [_root_.pow_succ', show (-1 : E) ^ (2 * n) = 1 by simp, neg_one_mul, one_mul,
      ← sub_eq_add_neg, le_sub_iff_add_le]
    gcongr
    exact hfa (by omega)
  /-
    E : Type u_2
    inst✝² : OrderedRing E
    inst✝¹ : TopologicalSpace E
    inst✝ : OrderClosedTopology E
    l : E
    f : Nat → E
    hfl : Filter.Tendsto (fun n => (Finset.range n).sum fun i => HMul.hMul (HPow.h …
    hfa : Antitone f
    k : Nat
    hm : Monotone fun n => (Finset.range (HMul.hMul 2 n)).sum fun i => HMul.hMul ( …
    ⊢ LE.le ((Finset.range (HMul.hMul 2 k)).sum fun i => HMul.hMul (HPow.hPow (-1) …
  -/
  exact hm.ge_of_tendsto (hfl.comp (tendsto_atTop_mono (fun n ↦ by dsimp; omega) tendsto_id)) _
  /-
    🎉 no goals
  -/


/-- Partial sums of an alternating antitone series with an odd number of terms provide
upper bounds on the limit. -/
theorem Antitone.tendsto_le_alternating_series
    (hfl : Tendsto (fun n ↦ ∑ i ∈ range n, (-1) ^ i * f i) atTop (𝓝 l))
    (hfa : Antitone f) (k : ℕ) : l ≤ ∑ i ∈ range (2 * k + 1), (-1) ^ i * f i := by
  have ha : Antitone (fun n ↦ ∑ i ∈ range (2 * n + 1), (-1) ^ i * f i) := by
    refine antitone_nat_of_succ_le (fun n ↦ ?_)
    rw [show 2 * (n + 1) = 2 * n + 1 + 1 by ring, sum_range_succ, sum_range_succ]
    simp_rw [_root_.pow_succ', show (-1 : E) ^ (2 * n) = 1 by simp, neg_one_mul, neg_neg, one_mul,
      ← sub_eq_add_neg, sub_add_eq_add_sub, sub_le_iff_le_add]
    gcongr
    exact hfa (by omega)
  /-
    E : Type u_2
    inst✝² : OrderedRing E
    inst✝¹ : TopologicalSpace E
    inst✝ : OrderClosedTopology E
    l : E
    f : Nat → E
    hfl : Filter.Tendsto (fun n => (Finset.range n).sum fun i => HMul.hMul (HPow.h …
    hfa : Antitone f
    k : Nat
    ha : Antitone fun n => (Finset.range (HAdd.hAdd (HMul.hMul 2 n) 1)).sum fun i  …
    ⊢ LE.le l ((Finset.range (HAdd.hAdd (HMul.hMul 2 k) 1)).sum fun i => HMul.hMul …
  -/
  exact ha.le_of_tendsto (hfl.comp (tendsto_atTop_mono (fun n ↦ by dsimp; omega) tendsto_id)) _
  /-
    🎉 no goals
  -/


/-- The series `∑' n, x ^ n / n!` is summable of any `x : ℝ`. See also `expSeries_div_summable`
for a version that also works in `ℂ`, and `NormedSpace.expSeries_summable'` for a version
that works in any normed algebra over `ℝ` or `ℂ`. -/
theorem Real.summable_pow_div_factorial (x : ℝ) : Summable (fun n ↦ x ^ n / n ! : ℕ → ℝ) := by
  -- We start with trivial estimates
  /-
    x : Real
    ⊢ Summable fun n => HDiv.hDiv (HPow.hPow x n) ↑n.factorial
  -/
  have A : (0 : ℝ) < ⌊‖x‖⌋₊ + 1 := zero_lt_one.trans_le (by simp)
  /-
    x : Real
    A : LT.lt 0 (HAdd.hAdd (↑(Nat.floor (Norm.norm x))) 1)
    ⊢ Summable fun n => HDiv.hDiv (HPow.hPow x n) ↑n.factorial
  -/
  have B : ‖x‖ / (⌊‖x‖⌋₊ + 1) < 1 := (div_lt_one A).2 (Nat.lt_floor_add_one _)
  -- Then we apply the ratio test. The estimate works for `n ≥ ⌊‖x‖⌋₊`.
  suffices ∀ n ≥ ⌊‖x‖⌋₊, ‖x ^ (n + 1) / (n + 1)!‖ ≤ ‖x‖ / (⌊‖x‖⌋₊ + 1) * ‖x ^ n / ↑n !‖ from
    summable_of_ratio_norm_eventually_le B (eventually_atTop.2 ⟨⌊‖x‖⌋₊, this⟩)
  -- Finally, we prove the upper estimate
  /-
    x : Real
    A : LT.lt 0 (HAdd.hAdd (↑(Nat.floor (Norm.norm x))) 1)
    B : LT.lt (HDiv.hDiv (Norm.norm x) (HAdd.hAdd (↑(Nat.floor (Norm.norm x))) 1)) 1
    ⊢ ∀ (n : Nat), GE.ge n (Nat.floor (Norm.norm x)) → LE.le (Norm.norm (HDiv.hDiv …
  -/
  intro n hn
  calc
    ‖x ^ (n + 1) / (n + 1)!‖ = ‖x‖ / (n + 1) * ‖x ^ n / (n !)‖ := by
      rw [_root_.pow_succ', Nat.factorial_succ, Nat.cast_mul, ← _root_.div_mul_div_comm, norm_mul,
        norm_div, Real.norm_natCast, Nat.cast_succ]
    _ ≤ ‖x‖ / (⌊‖x‖⌋₊ + 1) * ‖x ^ n / (n !)‖ := by gcongr


@[deprecated "`Real.tendsto_pow_div_factorial_atTop` has been deprecated, use
`FloorSemiring.tendsto_pow_div_factorial_atTop` instead" (since := "2024-10-05")]
theorem Real.tendsto_pow_div_factorial_atTop (x : ℝ) :
    Tendsto (fun n ↦ x ^ n / n ! : ℕ → ℝ) atTop (𝓝 0) :=
  (Real.summable_pow_div_factorial x).tendsto_atTop_zero

