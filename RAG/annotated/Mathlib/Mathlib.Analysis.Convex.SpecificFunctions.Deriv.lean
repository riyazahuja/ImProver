/-- `x^n`, `n : ℕ` is strictly convex on `[0, +∞)` for all `n` greater than `2`. -/
theorem strictConvexOn_pow {n : ℕ} (hn : 2 ≤ n) : StrictConvexOn ℝ (Ici 0) fun x : ℝ => x ^ n := by
  /-
    n : Nat
    hn : LE.le 2 n
    ⊢ StrictConvexOn Real (Set.Ici 0) fun x => HPow.hPow x n
  -/
  apply StrictMonoOn.strictConvexOn_of_deriv (convex_Ici _) (continuousOn_pow _)
  /-
    n : Nat
    hn : LE.le 2 n
    ⊢ StrictMonoOn (deriv fun x => HPow.hPow x n) (interior (Set.Ici 0))
  -/
  rw [deriv_pow', interior_Ici]
  exact fun x (hx : 0 < x) y _ hxy => mul_lt_mul_of_pos_left
    (pow_lt_pow_left₀ hxy hx.le <| Nat.sub_ne_zero_of_lt hn) (by positivity)


/-- `x^n`, `n : ℕ` is strictly convex on the whole real line whenever `n ≠ 0` is even. -/
theorem Even.strictConvexOn_pow {n : ℕ} (hn : Even n) (h : n ≠ 0) :
    StrictConvexOn ℝ Set.univ fun x : ℝ => x ^ n := by
  /-
    n : Nat
    hn : Even n
    h : Ne n 0
    ⊢ StrictConvexOn Real Set.univ fun x => HPow.hPow x n
  -/
  apply StrictMono.strictConvexOn_univ_of_deriv (continuous_pow n)
  /-
    n : Nat
    hn : Even n
    h : Ne n 0
    ⊢ StrictMono (deriv fun a => HPow.hPow a n)
  -/
  rw [deriv_pow']
  /-
    n : Nat
    hn : Even n
    h : Ne n 0
    ⊢ StrictMono fun x => HMul.hMul (↑n) (HPow.hPow x (HSub.hSub n 1))
  -/
  replace h := Nat.pos_of_ne_zero h
  exact StrictMono.const_mul (Odd.strictMono_pow <| Nat.Even.sub_odd h hn <| Nat.odd_iff.2 rfl)
    (Nat.cast_pos.2 h)


theorem Finset.prod_nonneg_of_card_nonpos_even {α β : Type*} [LinearOrderedCommRing β] {f : α → β}
    [DecidablePred fun x => f x ≤ 0] {s : Finset α} (h0 : Even (s.filter fun x => f x ≤ 0).card) :
    0 ≤ ∏ x ∈ s, f x :=
  calc
    0 ≤ ∏ x ∈ s, (if f x ≤ 0 then (-1 : β) else 1) * f x :=
      Finset.prod_nonneg fun x _ => by
        /-
          α : Type u_1
          β : Type u_2
          inst✝¹ : LinearOrderedCommRing β
          f : α → β
          inst✝ : DecidablePred fun x => LE.le (f x) 0
          s : Finset α
          h0 : Even (Finset.filter (fun x => LE.le (f x) 0) s).card
          x : α
          x✝ : Membership.mem s x
          ⊢ LE.le 0 (HMul.hMul (ite (LE.le (f x) 0) (-1) 1) (f x))
        -/
        split_ifs with hx
          /-
            case pos
            α : Type u_1
            β : Type u_2
            inst✝¹ : LinearOrderedCommRing β
            f : α → β
            inst✝ : DecidablePred fun x => LE.le (f x) 0
            s : Finset α
            h0 : Even (Finset.filter (fun x => LE.le (f x) 0) s).card
            x : α
            x✝ : Membership.mem s x
            hx : LE.le (f x) 0
            ⊢ LE.le 0 (HMul.hMul (-1) (f x))
          -/
        · simp [hx]
          /-
            🎉 no goals
          -/
        /-
          case neg
          α : Type u_1
          β : Type u_2
          inst✝¹ : LinearOrderedCommRing β
          f : α → β
          inst✝ : DecidablePred fun x => LE.le (f x) 0
          s : Finset α
          h0 : Even (Finset.filter (fun x => LE.le (f x) 0) s).card
          x : α
          x✝ : Membership.mem s x
          hx : Not (LE.le (f x) 0)
          ⊢ LE.le 0 (HMul.hMul 1 (f x))
        -/
        simp? at hx ⊢ says simp only [not_le, one_mul] at hx ⊢
        /-
          case neg
          α : Type u_1
          β : Type u_2
          inst✝¹ : LinearOrderedCommRing β
          f : α → β
          inst✝ : DecidablePred fun x => LE.le (f x) 0
          s : Finset α
          h0 : Even (Finset.filter (fun x => LE.le (f x) 0) s).card
          x : α
          x✝ : Membership.mem s x
          hx : LT.lt 0 (f x)
          ⊢ LE.le 0 (f x)
        -/
        exact le_of_lt hx
        /-
          🎉 no goals
        -/
    _ = _ := by
      rw [Finset.prod_mul_distrib, Finset.prod_ite, Finset.prod_const_one, mul_one,
        Finset.prod_const, neg_one_pow_eq_pow_mod_two, Nat.even_iff.1 h0, pow_zero, one_mul]


theorem int_prod_range_nonneg (m : ℤ) (n : ℕ) (hn : Even n) :
    0 ≤ ∏ k ∈ Finset.range n, (m - k) := by
  /-
    m : Int
    n : Nat
    hn : Even n
    ⊢ LE.le 0 ((Finset.range n).prod fun k => HSub.hSub m ↑k)
  -/
  rcases hn with ⟨n, rfl⟩
  induction n with
  | zero => simp
  | succ n ihn =>
    rw [← two_mul] at ihn
    rw [← two_mul, mul_add, mul_one, ← one_add_one_eq_two, ← add_assoc,
      Finset.prod_range_succ, Finset.prod_range_succ, mul_assoc]
    refine mul_nonneg ihn ?_; generalize (1 + 1) * n = k
    rcases le_or_lt m k with hmk | hmk
    · have : m ≤ k + 1 := hmk.trans (lt_add_one (k : ℤ)).le
      convert mul_nonneg_of_nonpos_of_nonpos (sub_nonpos_of_le hmk) _
      convert sub_nonpos_of_le this
    · exact mul_nonneg (sub_nonneg_of_le hmk.le) (sub_nonneg_of_le hmk)


theorem int_prod_range_pos {m : ℤ} {n : ℕ} (hn : Even n) (hm : m ∉ Ico (0 : ℤ) n) :
    0 < ∏ k ∈ Finset.range n, (m - k) := by
  /-
    m : Int
    n : Nat
    hn : Even n
    hm : Not (Membership.mem (Set.Ico 0 ↑n) m)
    ⊢ LT.lt 0 ((Finset.range n).prod fun k => HSub.hSub m ↑k)
  -/
  refine (int_prod_range_nonneg m n hn).lt_of_ne fun h => hm ?_
  /-
    m : Int
    n : Nat
    hn : Even n
    hm : Not (Membership.mem (Set.Ico 0 ↑n) m)
    h : Eq 0 ((Finset.range n).prod fun k => HSub.hSub m ↑k)
    ⊢ Membership.mem (Set.Ico 0 ↑n) m
  -/
  rw [eq_comm, Finset.prod_eq_zero_iff] at h
  /-
    m : Int
    n : Nat
    hn : Even n
    hm : Not (Membership.mem (Set.Ico 0 ↑n) m)
    h : Exists fun a => And (Membership.mem (Finset.range n) a) (Eq (HSub.hSub m ↑ …
    ⊢ Membership.mem (Set.Ico 0 ↑n) m
  -/
  obtain ⟨a, ha, h⟩ := h
  /-
    case intro.intro
    m : Int
    n : Nat
    hn : Even n
    hm : Not (Membership.mem (Set.Ico 0 ↑n) m)
    a : Nat
    ha : Membership.mem (Finset.range n) a
    h : Eq (HSub.hSub m ↑a) 0
    ⊢ Membership.mem (Set.Ico 0 ↑n) m
  -/
  rw [sub_eq_zero.1 h]
  /-
    case intro.intro
    m : Int
    n : Nat
    hn : Even n
    hm : Not (Membership.mem (Set.Ico 0 ↑n) m)
    a : Nat
    ha : Membership.mem (Finset.range n) a
    h : Eq (HSub.hSub m ↑a) 0
    ⊢ Membership.mem (Set.Ico 0 ↑n) ↑a
  -/
  exact ⟨Int.ofNat_zero_le _, Int.ofNat_lt.2 <| Finset.mem_range.1 ha⟩
  /-
    🎉 no goals
  -/


/-- `x^m`, `m : ℤ` is convex on `(0, +∞)` for all `m` except `0` and `1`. -/
theorem strictConvexOn_zpow {m : ℤ} (hm₀ : m ≠ 0) (hm₁ : m ≠ 1) :
    StrictConvexOn ℝ (Ioi 0) fun x : ℝ => x ^ m := by
  /-
    m : Int
    hm₀ : Ne m 0
    hm₁ : Ne m 1
    ⊢ StrictConvexOn Real (Set.Ioi 0) fun x => HPow.hPow x m
  -/
  apply strictConvexOn_of_deriv2_pos' (convex_Ioi 0)
    /-
      case hf
      m : Int
      hm₀ : Ne m 0
      hm₁ : Ne m 1
      ⊢ ContinuousOn (fun x => HPow.hPow x m) (Set.Ioi 0)
    -/
  · exact (continuousOn_zpow₀ m).mono fun x hx => ne_of_gt hx
    /-
      🎉 no goals
    -/
  /-
    case hf''
    m : Int
    hm₀ : Ne m 0
    hm₁ : Ne m 1
    ⊢ ∀ (x : Real), Membership.mem (Set.Ioi 0) x → LT.lt 0 (Nat.iterate deriv 2 (f …
  -/
  intro x hx
  /-
    case hf''
    m : Int
    hm₀ : Ne m 0
    hm₁ : Ne m 1
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ LT.lt 0 (Nat.iterate deriv 2 (fun x => HPow.hPow x m) x)
  -/
  rw [mem_Ioi] at hx
  /-
    case hf''
    m : Int
    hm₀ : Ne m 0
    hm₁ : Ne m 1
    x : Real
    hx : LT.lt 0 x
    ⊢ LT.lt 0 (Nat.iterate deriv 2 (fun x => HPow.hPow x m) x)
  -/
  rw [iter_deriv_zpow]
  /-
    case hf''
    m : Int
    hm₀ : Ne m 0
    hm₁ : Ne m 1
    x : Real
    hx : LT.lt 0 x
    ⊢ LT.lt 0 (HMul.hMul ((Finset.range 2).prod fun i => HSub.hSub ↑m ↑i) (HPow.hP …
  -/
  refine mul_pos ?_ (zpow_pos hx _)
  /-
    case hf''
    m : Int
    hm₀ : Ne m 0
    hm₁ : Ne m 1
    x : Real
    hx : LT.lt 0 x
    ⊢ LT.lt 0 ((Finset.range 2).prod fun i => HSub.hSub ↑m ↑i)
  -/
  norm_cast
  /-
    case hf''
    m : Int
    hm₀ : Ne m 0
    hm₁ : Ne m 1
    x : Real
    hx : LT.lt 0 x
    ⊢ LT.lt 0 ((Finset.range 2).prod fun i => HSub.hSub m ↑i)
  -/
  refine int_prod_range_pos (by decide) fun hm => ?_
  /-
    case hf''
    m : Int
    hm₀ : Ne m 0
    hm₁ : Ne m 1
    x : Real
    hx : LT.lt 0 x
    hm : Membership.mem (Set.Ico 0 ↑2) m
    ⊢ False
  -/
  rw [← Finset.coe_Ico] at hm
  /-
    case hf''
    m : Int
    hm₀ : Ne m 0
    hm₁ : Ne m 1
    x : Real
    hx : LT.lt 0 x
    hm : Membership.mem (↑(Finset.Ico 0 ↑2)) m
    ⊢ False
  -/
  norm_cast at hm
  /-
    case hf''
    m : Int
    hm₀ : Ne m 0
    hm₁ : Ne m 1
    x : Real
    hx : LT.lt 0 x
    hm : Membership.mem (Finset.Ico 0 2) m
    ⊢ False
  -/
                   /-
                     🎉 no goals
                   -/
  fin_cases hm <;> simp_all
                   /-
                     🎉 no goals
                   -/


theorem hasDerivAt_sqrt_mul_log {x : ℝ} (hx : x ≠ 0) :
    HasDerivAt (fun x => √x * log x) ((2 + log x) / (2 * √x)) x := by
  /-
    x : Real
    hx : Ne x 0
    ⊢ HasDerivAt (fun x => HMul.hMul x.sqrt (Real.log x)) (HDiv.hDiv (HAdd.hAdd 2  …
  -/
  convert (hasDerivAt_sqrt hx).mul (hasDerivAt_log hx) using 1
  rw [add_div, div_mul_cancel_left₀ two_ne_zero, ← div_eq_mul_inv, sqrt_div_self', add_comm,
    one_div, one_div, ← div_eq_inv_mul]


theorem deriv_sqrt_mul_log (x : ℝ) :
    deriv (fun x => √x * log x) x = (2 + log x) / (2 * √x) := by
  /-
    x : Real
    ⊢ Eq (deriv (fun x => HMul.hMul x.sqrt (Real.log x)) x) (HDiv.hDiv (HAdd.hAdd  …
  -/
  cases' lt_or_le 0 x with hx hx
    /-
      case inl
      x : Real
      hx : LT.lt 0 x
      ⊢ Eq (deriv (fun x => HMul.hMul x.sqrt (Real.log x)) x) (HDiv.hDiv (HAdd.hAdd  …
    -/
  · exact (hasDerivAt_sqrt_mul_log hx.ne').deriv
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : Real
      hx : LE.le x 0
      ⊢ Eq (deriv (fun x => HMul.hMul x.sqrt (Real.log x)) x) (HDiv.hDiv (HAdd.hAdd  …
    -/
  · rw [sqrt_eq_zero_of_nonpos hx, mul_zero, div_zero]
    /-
      case inr
      x : Real
      hx : LE.le x 0
      ⊢ Eq (deriv (fun x => HMul.hMul x.sqrt (Real.log x)) x) 0
    -/
    refine HasDerivWithinAt.deriv_eq_zero ?_ (uniqueDiffOn_Iic 0 x hx)
    /-
      case inr
      x : Real
      hx : LE.le x 0
      ⊢ HasDerivWithinAt (fun x => HMul.hMul x.sqrt (Real.log x)) 0 (Set.Iic 0) x
    -/
    refine (hasDerivWithinAt_const x _ 0).congr_of_mem (fun x hx => ?_) hx
    /-
      case inr
      x✝ : Real
      hx✝ : LE.le x✝ 0
      x : Real
      hx : Membership.mem (Set.Iic 0) x
      ⊢ Eq (HMul.hMul x.sqrt (Real.log x)) 0
    -/
    rw [sqrt_eq_zero_of_nonpos hx, zero_mul]
    /-
      🎉 no goals
    -/


theorem deriv_sqrt_mul_log' :
    (deriv fun x => √x * log x) = fun x => (2 + log x) / (2 * √x) :=
  funext deriv_sqrt_mul_log


theorem deriv2_sqrt_mul_log (x : ℝ) :
    deriv^[2] (fun x => √x * log x) x = -log x / (4 * √x ^ 3) := by
  /-
    x : Real
    ⊢ Eq (Nat.iterate deriv 2 (fun x => HMul.hMul x.sqrt (Real.log x)) x) (HDiv.hD …
  -/
  simp only [Nat.iterate, deriv_sqrt_mul_log']
  /-
    x : Real
    ⊢ Eq (deriv (fun x => HDiv.hDiv (HAdd.hAdd 2 (Real.log x)) (HMul.hMul 2 x.sqrt …
  -/
  rcases le_or_lt x 0 with hx | hx
    /-
      case inl
      x : Real
      hx : LE.le x 0
      ⊢ Eq (deriv (fun x => HDiv.hDiv (HAdd.hAdd 2 (Real.log x)) (HMul.hMul 2 x.sqrt …
    -/
  · rw [sqrt_eq_zero_of_nonpos hx, zero_pow three_ne_zero, mul_zero, div_zero]
    /-
      case inl
      x : Real
      hx : LE.le x 0
      ⊢ Eq (deriv (fun x => HDiv.hDiv (HAdd.hAdd 2 (Real.log x)) (HMul.hMul 2 x.sqrt …
    -/
    refine HasDerivWithinAt.deriv_eq_zero ?_ (uniqueDiffOn_Iic 0 x hx)
    /-
      case inl
      x : Real
      hx : LE.le x 0
      ⊢ HasDerivWithinAt (fun x => HDiv.hDiv (HAdd.hAdd 2 (Real.log x)) (HMul.hMul 2 …
    -/
    refine (hasDerivWithinAt_const _ _ 0).congr_of_mem (fun x hx => ?_) hx
    /-
      case inl
      x✝ : Real
      hx✝ : LE.le x✝ 0
      x : Real
      hx : Membership.mem (Set.Iic 0) x
      ⊢ Eq (HDiv.hDiv (HAdd.hAdd 2 (Real.log x)) (HMul.hMul 2 x.sqrt)) 0
    -/
    rw [sqrt_eq_zero_of_nonpos hx, mul_zero, div_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : Real
      hx : LT.lt 0 x
      ⊢ Eq (deriv (fun x => HDiv.hDiv (HAdd.hAdd 2 (Real.log x)) (HMul.hMul 2 x.sqrt …
    -/
  · have h₀ : √x ≠ 0 := sqrt_ne_zero'.2 hx
    convert (((hasDerivAt_log hx.ne').const_add 2).div ((hasDerivAt_sqrt hx.ne').const_mul 2) <|
      mul_ne_zero two_ne_zero h₀).deriv using 1
    /-
      case h.e'_3
      x : Real
      hx : LT.lt 0 x
      h₀ : Ne x.sqrt 0
      ⊢ Eq (HDiv.hDiv (Neg.neg (Real.log x)) (HMul.hMul 4 (HPow.hPow x.sqrt 3))) (HD …
    -/
    nth_rw 3 [← mul_self_sqrt hx.le]
    /-
      case h.e'_3
      x : Real
      hx : LT.lt 0 x
      h₀ : Ne x.sqrt 0
      ⊢ Eq (HDiv.hDiv (Neg.neg (Real.log x)) (HMul.hMul 4 (HPow.hPow x.sqrt 3))) (HD …
    -/
    generalize √x = sqx at h₀ -- else field_simp rewrites sqrt x * sqrt x back to x
    /-
      case h.e'_3
      x : Real
      hx : LT.lt 0 x
      sqx : Real
      h₀ : Ne sqx 0
      ⊢ Eq (HDiv.hDiv (Neg.neg (Real.log x)) (HMul.hMul 4 (HPow.hPow sqx 3))) (HDiv. …
    -/
    field_simp
    /-
      case h.e'_3
      x : Real
      hx : LT.lt 0 x
      sqx : Real
      h₀ : Ne sqx 0
      ⊢ Eq (Neg.neg (HMul.hMul (Real.log x) (HMul.hMul (HMul.hMul (HMul.hMul sqx sqx …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem strictConcaveOn_sqrt_mul_log_Ioi :
    StrictConcaveOn ℝ (Set.Ioi 1) fun x => √x * log x := by
  /-
    ⊢ StrictConcaveOn Real (Set.Ioi 1) fun x => HMul.hMul x.sqrt (Real.log x)
  -/
  apply strictConcaveOn_of_deriv2_neg' (convex_Ioi 1) _ fun x hx => ?_
  · exact continuous_sqrt.continuousOn.mul
      (continuousOn_log.mono fun x hx => ne_of_gt (zero_lt_one.trans hx))
    /-
      x : Real
      hx : Membership.mem (Set.Ioi 1) x
      ⊢ LT.lt (Nat.iterate deriv 2 (fun x => HMul.hMul x.sqrt (Real.log x)) x) 0
    -/
  · rw [deriv2_sqrt_mul_log x]
    exact div_neg_of_neg_of_pos (neg_neg_of_pos (log_pos hx))
      (mul_pos four_pos (pow_pos (sqrt_pos.mpr (zero_lt_one.trans hx)) 3))


theorem strictConcaveOn_sin_Icc : StrictConcaveOn ℝ (Icc 0 π) sin := by
  /-
    ⊢ StrictConcaveOn Real (Set.Icc 0 Real.pi) Real.sin
  -/
  apply strictConcaveOn_of_deriv2_neg (convex_Icc _ _) continuousOn_sin fun x hx => ?_
  /-
    x : Real
    hx : Membership.mem (interior (Set.Icc 0 Real.pi)) x
    ⊢ LT.lt (Nat.iterate deriv 2 Real.sin x) 0
  -/
  rw [interior_Icc] at hx
  /-
    x : Real
    hx : Membership.mem (Set.Ioo 0 Real.pi) x
    ⊢ LT.lt (Nat.iterate deriv 2 Real.sin x) 0
  -/
  simp [sin_pos_of_mem_Ioo hx]
  /-
    🎉 no goals
  -/


theorem strictConcaveOn_cos_Icc : StrictConcaveOn ℝ (Icc (-(π / 2)) (π / 2)) cos := by
  /-
    ⊢ StrictConcaveOn Real (Set.Icc (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Rea …
  -/
  apply strictConcaveOn_of_deriv2_neg (convex_Icc _ _) continuousOn_cos fun x hx => ?_
  /-
    x : Real
    hx : Membership.mem (interior (Set.Icc (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.h …
    ⊢ LT.lt (Nat.iterate deriv 2 Real.cos x) 0
  -/
  rw [interior_Icc] at hx
  /-
    x : Real
    hx : Membership.mem (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    ⊢ LT.lt (Nat.iterate deriv 2 Real.cos x) 0
  -/
  simp [cos_pos_of_mem_Ioo hx]
  /-
    🎉 no goals
  -/

