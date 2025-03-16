/-- For a ring seminorm `f` on `R` and `c ∈ R`, the sequence given by `(f (x * c^n))/((f c)^n)`. -/
def seminormFromConst_seq (x : R) : ℕ → ℝ := fun n ↦ f (x * c ^ n) / f c ^ n


lemma seminormFromConst_seq_def (x : R) :
    seminormFromConst_seq c f x = fun n ↦ f (x * c ^ n) / f c ^ n := rfl


/-- The terms in the sequence `seminormFromConst_seq c f x` are nonnegative. -/
theorem seminormFromConst_seq_nonneg (x : R) : 0 ≤ seminormFromConst_seq c f x :=
  fun n ↦ div_nonneg (apply_nonneg f (x * c ^ n)) (pow_nonneg (apply_nonneg f c) n)


/-- The image of `seminormFromConst_seq c f x` is bounded below by zero. -/
theorem seminormFromConst_bddBelow (x : R) :
    BddBelow (Set.range (seminormFromConst_seq c f x)) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    x : R
    ⊢ BddBelow (Set.range (seminormFromConst_seq c f x))
  -/
  use 0
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    x : R
    ⊢ Membership.mem (lowerBounds (Set.range (seminormFromConst_seq c f x))) 0
  -/
  rintro r ⟨n, rfl⟩
  /-
    case h.intro
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    x : R
    n : Nat
    ⊢ LE.le 0 (seminormFromConst_seq c f x n)
  -/
  exact seminormFromConst_seq_nonneg c f x n
  /-
    🎉 no goals
  -/


/-- `seminormFromConst_seq c f 0` is the constant sequence zero. -/
theorem seminormFromConst_seq_zero (hf : f 0 = 0) : seminormFromConst_seq c f 0 = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf : Eq (f 0) 0
    ⊢ Eq (seminormFromConst_seq c f 0) 0
  -/
  rw [seminormFromConst_seq_def]
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf : Eq (f 0) 0
    ⊢ Eq (fun n => HDiv.hDiv (f (HMul.hMul 0 (HPow.hPow c n))) (HPow.hPow (f c) n) …
  -/
  ext n
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf : Eq (f 0) 0
    n : Nat
    ⊢ Eq (HDiv.hDiv (f (HMul.hMul 0 (HPow.hPow c n))) (HPow.hPow (f c) n)) (0 n)
  -/
  rw [zero_mul, hf, zero_div, Pi.zero_apply]
  /-
    🎉 no goals
  -/


/-- If `1 ≤ n`, then `seminormFromConst_seq c f 1 n = 1`. -/
theorem seminormFromConst_seq_one (n : ℕ) (hn : 1 ≤ n) : seminormFromConst_seq c f 1 n = 1 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    n : Nat
    hn : LE.le 1 n
    ⊢ Eq (seminormFromConst_seq c f 1 n) 1
  -/
  simp only [seminormFromConst_seq]
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    n : Nat
    hn : LE.le 1 n
    ⊢ Eq (HDiv.hDiv (f (HMul.hMul 1 (HPow.hPow c n))) (HPow.hPow (f c) n)) 1
  -/
  rw [one_mul, hpm _ hn, div_self (pow_ne_zero n hc)]
  /-
    🎉 no goals
  -/


/-- `seminormFromConst_seq c f x` is antitone. -/
theorem seminormFromConst_seq_antitone (x : R) : Antitone (seminormFromConst_seq c f x) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    ⊢ Antitone (seminormFromConst_seq c f x)
  -/
  intro m n hmn
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    m n : Nat
    hmn : LE.le m n
    ⊢ LE.le (seminormFromConst_seq c f x n) (seminormFromConst_seq c f x m)
  -/
  simp only [seminormFromConst_seq]
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    m n : Nat
    hmn : LE.le m n
    ⊢ LE.le (HDiv.hDiv (f (HMul.hMul x (HPow.hPow c n))) (HPow.hPow (f c) n)) (HDi …
  -/
  nth_rw 1 [← Nat.add_sub_of_le hmn]
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    m n : Nat
    hmn : LE.le m n
    ⊢ LE.le (HDiv.hDiv (f (HMul.hMul x (HPow.hPow c (HAdd.hAdd m (HSub.hSub n m))) …
  -/
  rw [pow_add, ← mul_assoc]
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    m n : Nat
    hmn : LE.le m n
    ⊢ LE.le (HDiv.hDiv (f (HMul.hMul (HMul.hMul x (HPow.hPow c m)) (HPow.hPow c (H …
  -/
  have hc_pos : 0 < f c := lt_of_le_of_ne (apply_nonneg f _) hc.symm
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    m n : Nat
    hmn : LE.le m n
    hc_pos : LT.lt 0 (f c)
    ⊢ LE.le (HDiv.hDiv (f (HMul.hMul (HMul.hMul x (HPow.hPow c m)) (HPow.hPow c (H …
  -/
  apply le_trans ((div_le_div_iff_of_pos_right (pow_pos hc_pos _)).mpr (map_mul_le_mul f _ _))
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    m n : Nat
    hmn : LE.le m n
    hc_pos : LT.lt 0 (f c)
    ⊢ LE.le (HDiv.hDiv (HMul.hMul (f (HMul.hMul x (HPow.hPow c m))) (f (HPow.hPow  …
  -/
  by_cases heq : m = n
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      x : R
      m n : Nat
      hmn : LE.le m n
      hc_pos : LT.lt 0 (f c)
      heq : Eq m n
      ⊢ LE.le (HDiv.hDiv (HMul.hMul (f (HMul.hMul x (HPow.hPow c m))) (f (HPow.hPow  …
    -/
  · have hnm : n - m = 0 := by rw [heq, Nat.sub_self n]
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      x : R
      m n : Nat
      hmn : LE.le m n
      hc_pos : LT.lt 0 (f c)
      heq : Eq m n
      hnm : Eq (HSub.hSub n m) 0
      ⊢ LE.le (HDiv.hDiv (HMul.hMul (f (HMul.hMul x (HPow.hPow c m))) (f (HPow.hPow  …
    -/
    rw [hnm, heq, div_le_div_iff_of_pos_right (pow_pos hc_pos _), pow_zero]
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      x : R
      m n : Nat
      hmn : LE.le m n
      hc_pos : LT.lt 0 (f c)
      heq : Eq m n
      hnm : Eq (HSub.hSub n m) 0
      ⊢ LE.le (HMul.hMul (f (HMul.hMul x (HPow.hPow c n))) (f 1)) (f (HMul.hMul x (H …
    -/
    conv_rhs => rw [← mul_one (f (x * c ^ n))]
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      x : R
      m n : Nat
      hmn : LE.le m n
      hc_pos : LT.lt 0 (f c)
      heq : Eq m n
      hnm : Eq (HSub.hSub n m) 0
      ⊢ LE.le (HMul.hMul (f (HMul.hMul x (HPow.hPow c n))) (f 1)) (HMul.hMul (f (HMu …
    -/
    exact mul_le_mul_of_nonneg_left hf1 (apply_nonneg f _)
    /-
      🎉 no goals
    -/
  · have h1 : 1 ≤ n - m := by
      rw [Nat.one_le_iff_ne_zero, ne_eq, Nat.sub_eq_zero_iff_le, not_le]
      exact lt_of_le_of_ne hmn heq
    rw [hpm c h1, mul_div_assoc, div_eq_mul_inv, pow_sub₀ _ hc hmn, mul_assoc, mul_comm (f c ^ m)⁻¹,
      ← mul_assoc (f c ^ n), mul_inv_cancel₀ (pow_ne_zero n hc), one_mul, div_eq_mul_inv]


/-- The real-valued function sending `x ∈ R` to the limit of `(f (x * c^n))/((f c)^n)`. -/
def seminormFromConst' (x : R) : ℝ :=
  (Real.tendsto_of_bddBelow_antitone (seminormFromConst_bddBelow c f x)
    (seminormFromConst_seq_antitone hf1 hc hpm x)).choose


/-- We prove that `seminormFromConst' hf1 hc hpm x` is the limit of the sequence
  `seminormFromConst_seq c f x` as `n` tends to infinity. -/
theorem seminormFromConst_isLimit (x : R) :
    Tendsto (seminormFromConst_seq c f x) atTop (𝓝 (seminormFromConst' hf1 hc hpm x)) :=
  (Real.tendsto_of_bddBelow_antitone (seminormFromConst_bddBelow c f x)
      (seminormFromConst_seq_antitone hf1 hc hpm x)).choose_spec


/-- `seminormFromConst' hf1 hc hpm 1 = 1`. -/
theorem seminormFromConst_one : seminormFromConst' hf1 hc hpm 1 = 1 := by
  apply tendsto_nhds_unique_of_eventuallyEq (seminormFromConst_isLimit hf1 hc hpm 1)
    tendsto_const_nhds
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    ⊢ Filter.atTop.EventuallyEq (seminormFromConst_seq c f 1) fun x => 1
  -/
  simp only [EventuallyEq, eventually_atTop, ge_iff_le]
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    ⊢ Exists fun a => ∀ (b : Nat), LE.le a b → Eq (seminormFromConst_seq c f 1 b) 1
  -/
  exact ⟨1, seminormFromConst_seq_one hc hpm⟩
  /-
    🎉 no goals
  -/


/-- The function `seminormFromConst` is a `RingSeminorm` on `R`. -/
def seminormFromConst : RingSeminorm R where
  toFun     := seminormFromConst' hf1 hc hpm
  map_zero' := tendsto_nhds_unique (seminormFromConst_isLimit hf1 hc hpm 0)
        /-
          R : Type ?u.19783
          inst✝ : CommRing R
          c : R
          f : RingSeminorm R
          hf1 : LE.le (f 1) 1
          hc : Ne (f c) 0
          hpm : IsPowMul ⇑f
          ⊢ Filter.Tendsto (seminormFromConst_seq c f 0) Filter.atTop (nhds 0)
        -/
    (by simpa [seminormFromConst_seq_zero c (map_zero _)] using tendsto_const_nhds)
        /-
          🎉 no goals
        -/
  add_le' x y := by
    apply le_of_tendsto_of_tendsto' (seminormFromConst_isLimit hf1 hc hpm (x + y))
      ((seminormFromConst_isLimit hf1 hc hpm x).add (seminormFromConst_isLimit hf1 hc hpm y))
    /-
      R : Type ?u.19783
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      x y : R
      ⊢ ∀ (x_1 : Nat), LE.le (seminormFromConst_seq c f (HAdd.hAdd x y) x_1) (HAdd.h …
    -/
    intro n
    have h_add : f ((x + y) * c ^ n) ≤ f (x * c ^ n) + f (y * c ^ n) := by
      simp only [add_mul, map_add_le_add f _ _]
    /-
      R : Type ?u.19783
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      x y : R
      n : Nat
      h_add : LE.le (f (HMul.hMul (HAdd.hAdd x y) (HPow.hPow c n))) (HAdd.hAdd (f (H …
      ⊢ LE.le (seminormFromConst_seq c f (HAdd.hAdd x y) n) (HAdd.hAdd (seminormFrom …
    -/
    simp only [seminormFromConst_seq, div_add_div_same]
    /-
      R : Type ?u.19783
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      x y : R
      n : Nat
      h_add : LE.le (f (HMul.hMul (HAdd.hAdd x y) (HPow.hPow c n))) (HAdd.hAdd (f (H …
      ⊢ LE.le (HDiv.hDiv (f (HMul.hMul (HAdd.hAdd x y) (HPow.hPow c n))) (HPow.hPow  …
    -/
    gcongr
    /-
      🎉 no goals
    -/
  neg' x := by
    apply tendsto_nhds_unique_of_eventuallyEq (seminormFromConst_isLimit hf1 hc hpm (-x))
      (seminormFromConst_isLimit hf1 hc hpm x)
    /-
      R : Type ?u.19783
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      x : R
      ⊢ Filter.atTop.EventuallyEq (seminormFromConst_seq c f (Neg.neg x)) (seminormF …
    -/
    simp only [EventuallyEq, eventually_atTop]
    /-
      R : Type ?u.19783
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      x : R
      ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → Eq (seminormFromConst_seq c f (Neg. …
    -/
    use 0
    /-
      case h
      R : Type ?u.19783
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      x : R
      ⊢ ∀ (b : Nat), GE.ge b 0 → Eq (seminormFromConst_seq c f (Neg.neg x) b) (semin …
    -/
    simp only [seminormFromConst_seq, neg_mul, map_neg_eq_map, zero_le, implies_true]
    /-
      🎉 no goals
    -/
  mul_le' x y := by
    have hlim : Tendsto (fun n ↦ seminormFromConst_seq c f (x * y) (2 * n)) atTop
        (𝓝 (seminormFromConst' hf1 hc hpm (x * y))) := by
      apply (seminormFromConst_isLimit hf1 hc hpm (x * y)).comp
        (tendsto_atTop_atTop_of_monotone (fun _ _ hnm ↦ by
          simp only [mul_le_mul_left, Nat.succ_pos', hnm]) _)
      · rintro n; use n; omega
    refine le_of_tendsto_of_tendsto' hlim ((seminormFromConst_isLimit hf1 hc hpm x).mul
      (seminormFromConst_isLimit hf1 hc hpm y)) (fun n ↦ ?_)
    /-
      R : Type ?u.19783
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      x y : R
      hlim : Filter.Tendsto (fun n => seminormFromConst_seq c f (HMul.hMul x y) (HMu …
      n : Nat
      ⊢ LE.le (seminormFromConst_seq c f (HMul.hMul x y) (HMul.hMul 2 n)) (HMul.hMul …
    -/
    simp only [seminormFromConst_seq]
    rw [div_mul_div_comm, ← pow_add, two_mul,
      div_le_div_iff_of_pos_right (pow_pos (lt_of_le_of_ne (apply_nonneg f _) hc.symm) _), pow_add,
      ← mul_assoc, mul_comm (x * y), ← mul_assoc, mul_assoc, mul_comm (c ^ n)]
    /-
      R : Type ?u.19783
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      x y : R
      hlim : Filter.Tendsto (fun n => seminormFromConst_seq c f (HMul.hMul x y) (HMu …
      n : Nat
      ⊢ LE.le (f (HMul.hMul (HMul.hMul x (HPow.hPow c n)) (HMul.hMul y (HPow.hPow c  …
    -/
    exact map_mul_le_mul f (x * c ^ n) (y * c ^ n)
    /-
      🎉 no goals
    -/


theorem seminormFromConst_def (x : R) :
    seminormFromConst hf1 hc hpm x = seminormFromConst' hf1 hc hpm x :=
  rfl


/-- `seminormFromConst' hf1 hc hpm 1 ≤ 1`. -/
theorem seminormFromConst_one_le : seminormFromConst' hf1 hc hpm 1 ≤ 1 :=
  le_of_eq (seminormFromConst_one hf1 hc hpm)


/-- The function `seminormFromConst' hf1 hc hpm` is nonarchimedean. -/
theorem seminormFromConst_isNonarchimedean (hna : IsNonarchimedean f) :
    IsNonarchimedean (seminormFromConst' hf1 hc hpm) := fun x y ↦ by
  apply le_of_tendsto_of_tendsto' (seminormFromConst_isLimit hf1 hc hpm (x + y))
    ((seminormFromConst_isLimit hf1 hc hpm x).max (seminormFromConst_isLimit hf1 hc hpm y))
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    hna : IsNonarchimedean ⇑f
    x y : R
    ⊢ ∀ (x_1 : Nat), LE.le (seminormFromConst_seq c f (HAdd.hAdd x y) x_1) (Max.ma …
  -/
  intro n
  have hmax : f ((x + y) * c ^ n) ≤ max (f (x * c ^ n)) (f (y * c ^ n)) := by
    simp only [add_mul, hna _ _]
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    hna : IsNonarchimedean ⇑f
    x y : R
    n : Nat
    hmax : LE.le (f (HMul.hMul (HAdd.hAdd x y) (HPow.hPow c n))) (Max.max (f (HMul …
    ⊢ LE.le (seminormFromConst_seq c f (HAdd.hAdd x y) n) (Max.max (seminormFromCo …
  -/
  rw [le_max_iff] at hmax ⊢
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    hna : IsNonarchimedean ⇑f
    x y : R
    n : Nat
    hmax : Or (LE.le (f (HMul.hMul (HAdd.hAdd x y) (HPow.hPow c n))) (f (HMul.hMul …
    ⊢ Or (LE.le (seminormFromConst_seq c f (HAdd.hAdd x y) n) (seminormFromConst_s …
  -/
  unfold seminormFromConst_seq
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    hna : IsNonarchimedean ⇑f
    x y : R
    n : Nat
    hmax : Or (LE.le (f (HMul.hMul (HAdd.hAdd x y) (HPow.hPow c n))) (f (HMul.hMul …
    ⊢ Or (LE.le (HDiv.hDiv (f (HMul.hMul (HAdd.hAdd x y) (HPow.hPow c n))) (HPow.h …
  -/
                               /-
                                 🎉 no goals
                               -/
  apply hmax.imp <;> intro <;> gcongr
                               /-
                                 🎉 no goals
                               -/


/-- The function `seminormFromConst' hf1 hc hpm` is power-multiplicative. -/
theorem seminormFromConst_isPowMul : IsPowMul (seminormFromConst' hf1 hc hpm) := fun x m hm ↦ by
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    m : Nat
    hm : LE.le 1 m
    ⊢ Eq (seminormFromConst' hf1 hc hpm (HPow.hPow x m)) (HPow.hPow (seminormFromC …
  -/
  simp only [seminormFromConst']
  have hlim : Tendsto (fun n ↦ seminormFromConst_seq c f (x ^ m) (m * n)) atTop
      (𝓝 (seminormFromConst' hf1 hc hpm (x ^ m))) := by
    apply (seminormFromConst_isLimit hf1 hc hpm (x ^ m)).comp
      (tendsto_atTop_atTop_of_monotone (fun _ _ hnk ↦ mul_le_mul_left' hnk m) _)
    rintro n; use n; exact le_mul_of_one_le_left' hm
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    m : Nat
    hm : LE.le 1 m
    hlim : Filter.Tendsto (fun n => seminormFromConst_seq c f (HPow.hPow x m) (HMu …
    ⊢ Eq ⋯.choose (HPow.hPow ⋯.choose m)
  -/
  apply tendsto_nhds_unique hlim
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    m : Nat
    hm : LE.le 1 m
    hlim : Filter.Tendsto (fun n => seminormFromConst_seq c f (HPow.hPow x m) (HMu …
    ⊢ Filter.Tendsto (fun n => seminormFromConst_seq c f (HPow.hPow x m) (HMul.hMu …
  -/
  convert (seminormFromConst_isLimit hf1 hc hpm x).pow m using 1
  /-
    case h.e'_3
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    m : Nat
    hm : LE.le 1 m
    hlim : Filter.Tendsto (fun n => seminormFromConst_seq c f (HPow.hPow x m) (HMu …
    ⊢ Eq (fun n => seminormFromConst_seq c f (HPow.hPow x m) (HMul.hMul m n)) fun  …
  -/
  ext n
  /-
    case h.e'_3.h
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    m : Nat
    hm : LE.le 1 m
    hlim : Filter.Tendsto (fun n => seminormFromConst_seq c f (HPow.hPow x m) (HMu …
    n : Nat
    ⊢ Eq (seminormFromConst_seq c f (HPow.hPow x m) (HMul.hMul m n)) (HPow.hPow (s …
  -/
  simp only [seminormFromConst_seq, div_pow, ← hpm _ hm, ← pow_mul, mul_pow, mul_comm m n]
  /-
    🎉 no goals
  -/


/-- The function `seminormFromConst' hf1 hc hpm` is bounded above by `x`. -/
theorem seminormFromConst_le_seminorm (x : R) : seminormFromConst' hf1 hc hpm x ≤ f x := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    ⊢ LE.le (seminormFromConst' hf1 hc hpm x) (f x)
  -/
  apply le_of_tendsto (seminormFromConst_isLimit hf1 hc hpm x)
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    ⊢ Filter.Eventually (fun c_1 => LE.le (seminormFromConst_seq c f x c_1) (f x)) …
  -/
  simp only [eventually_atTop, ge_iff_le]
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    ⊢ Exists fun a => ∀ (b : Nat), LE.le a b → LE.le (seminormFromConst_seq c f x  …
  -/
  use 1
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    ⊢ ∀ (b : Nat), LE.le 1 b → LE.le (seminormFromConst_seq c f x b) (f x)
  -/
  intro n hn
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    n : Nat
    hn : LE.le 1 n
    ⊢ LE.le (seminormFromConst_seq c f x n) (f x)
  -/
  rw [seminormFromConst_seq, div_le_iff₀ (by positivity), ← hpm c hn]
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    n : Nat
    hn : LE.le 1 n
    ⊢ LE.le (f (HMul.hMul x (HPow.hPow c n))) (HMul.hMul (f x) (f (HPow.hPow c n)))
  -/
  exact map_mul_le_mul ..
  /-
    🎉 no goals
  -/


/-- If `x : R` is multiplicative for `f`, then `seminormFromConst' hf1 hc hpm x = f x`. -/
theorem seminormFromConst_apply_of_isMul {x : R} (hx : ∀ y : R, f (x * y) = f x * f y) :
    seminormFromConst' hf1 hc hpm x = f x :=
  have hlim : Tendsto (seminormFromConst_seq c f x) atTop (𝓝 (f x)) := by
    have hseq : seminormFromConst_seq c f x = fun _n ↦ f x := by
      ext n
      by_cases hn : n = 0
      · simp only [seminormFromConst_seq, hn, pow_zero, mul_one, div_one]
      · simp only [seminormFromConst_seq, hx (c ^ n), hpm _ (Nat.one_le_iff_ne_zero.mpr hn),
          mul_div_assoc, div_self (pow_ne_zero n hc), mul_one]
    /-
      R : Type u_1
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      x : R
      hx : ∀ (y : R), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      hseq : Eq (seminormFromConst_seq c f x) fun _n => f x
      ⊢ Filter.Tendsto (seminormFromConst_seq c f x) Filter.atTop (nhds (f x))
    -/
    rw [hseq]
    /-
      R : Type u_1
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      x : R
      hx : ∀ (y : R), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      hseq : Eq (seminormFromConst_seq c f x) fun _n => f x
      ⊢ Filter.Tendsto (fun _n => f x) Filter.atTop (nhds (f x))
    -/
    exact tendsto_const_nhds
    /-
      🎉 no goals
    -/
  tendsto_nhds_unique (seminormFromConst_isLimit hf1 hc hpm x) hlim


/-- If `x : R` is multiplicative for `f`, then it is multiplicative for
  `seminormFromConst' hf1 hc hpm`. -/
theorem seminormFromConst_isMul_of_isMul {x : R} (hx : ∀ y : R, f (x * y) = f x * f y) (y : R) :
    seminormFromConst' hf1 hc hpm (x * y) =
      seminormFromConst' hf1 hc hpm x * seminormFromConst' hf1 hc hpm y :=
  have hlim : Tendsto (seminormFromConst_seq c f (x * y)) atTop
      (𝓝 (seminormFromConst' hf1 hc hpm x * seminormFromConst' hf1 hc hpm y)) := by
    /-
      R : Type u_1
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      x : R
      hx : ∀ (y : R), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      y : R
      ⊢ Filter.Tendsto (seminormFromConst_seq c f (HMul.hMul x y)) Filter.atTop (nhd …
    -/
    rw [seminormFromConst_apply_of_isMul hf1 hc hpm hx]
    have hseq : seminormFromConst_seq c f (x * y) =
        fun n ↦ f x * seminormFromConst_seq c f y n := by
      ext n
      simp only [seminormFromConst_seq, mul_assoc, hx, mul_div_assoc]
    /-
      R : Type u_1
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      x : R
      hx : ∀ (y : R), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      y : R
      hseq : Eq (seminormFromConst_seq c f (HMul.hMul x y)) fun n => HMul.hMul (f x) …
      ⊢ Filter.Tendsto (seminormFromConst_seq c f (HMul.hMul x y)) Filter.atTop (nhd …
    -/
    simpa [hseq] using (seminormFromConst_isLimit hf1 hc hpm y).const_mul _
    /-
      🎉 no goals
    -/
  tendsto_nhds_unique (seminormFromConst_isLimit hf1 hc hpm (x * y)) hlim


/-- `seminormFromConst' hf1 hc hpm c = f c`. -/
theorem seminormFromConst_apply_c : seminormFromConst' hf1 hc hpm c = f c :=
  have hlim : Tendsto (seminormFromConst_seq c f c) atTop (𝓝 (f c)) := by
    have hseq : seminormFromConst_seq c f c = fun _n ↦ f c := by
      ext n
      simp only [seminormFromConst_seq]
      rw [mul_comm, ← pow_succ, hpm _ le_add_self, pow_succ, mul_comm,  mul_div_assoc,
        div_self (pow_ne_zero n hc), mul_one]
    /-
      R : Type u_1
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      hseq : Eq (seminormFromConst_seq c f c) fun _n => f c
      ⊢ Filter.Tendsto (seminormFromConst_seq c f c) Filter.atTop (nhds (f c))
    -/
    rw [hseq]
    /-
      R : Type u_1
      inst✝ : CommRing R
      c : R
      f : RingSeminorm R
      hf1 : LE.le (f 1) 1
      hc : Ne (f c) 0
      hpm : IsPowMul ⇑f
      hseq : Eq (seminormFromConst_seq c f c) fun _n => f c
      ⊢ Filter.Tendsto (fun _n => f c) Filter.atTop (nhds (f c))
    -/
    exact tendsto_const_nhds
    /-
      🎉 no goals
    -/
  tendsto_nhds_unique (seminormFromConst_isLimit hf1 hc hpm c) hlim


/-- For every `x : R`, `seminormFromConst' hf1 hc hpm (c * x)` equals the product
  `seminormFromConst' hf1 hc hpm c * SeminormFromConst' hf1 hc hpm x`. -/
theorem seminormFromConst_const_mul (x : R) :
    seminormFromConst' hf1 hc hpm (c * x) =
      seminormFromConst' hf1 hc hpm c * seminormFromConst' hf1 hc hpm x := by
  have hlim : Tendsto (fun n ↦ seminormFromConst_seq c f x (n + 1)) atTop
      (𝓝 (seminormFromConst' hf1 hc hpm x)) := by
    apply (seminormFromConst_isLimit hf1 hc hpm x).comp
      (tendsto_atTop_atTop_of_monotone (fun _ _ hnm ↦ add_le_add_right hnm 1) _)
    rintro n; use n; omega
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    hlim : Filter.Tendsto (fun n => seminormFromConst_seq c f x (HAdd.hAdd n 1)) F …
    ⊢ Eq (seminormFromConst' hf1 hc hpm (HMul.hMul c x)) (HMul.hMul (seminormFromC …
  -/
  rw [seminormFromConst_apply_c hf1 hc hpm]
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    hlim : Filter.Tendsto (fun n => seminormFromConst_seq c f x (HAdd.hAdd n 1)) F …
    ⊢ Eq (seminormFromConst' hf1 hc hpm (HMul.hMul c x)) (HMul.hMul (f c) (seminor …
  -/
  apply tendsto_nhds_unique (seminormFromConst_isLimit hf1 hc hpm (c * x))
  have hterm : seminormFromConst_seq c f (c * x) =
      fun n ↦ f c * seminormFromConst_seq c f x (n + 1) := by
    simp only [seminormFromConst_seq_def]
    ext n
    ring_nf
    rw [mul_assoc _ (f c), mul_inv_cancel₀ hc, mul_one]
  /-
    R : Type u_1
    inst✝ : CommRing R
    c : R
    f : RingSeminorm R
    hf1 : LE.le (f 1) 1
    hc : Ne (f c) 0
    hpm : IsPowMul ⇑f
    x : R
    hlim : Filter.Tendsto (fun n => seminormFromConst_seq c f x (HAdd.hAdd n 1)) F …
    hterm : Eq (seminormFromConst_seq c f (HMul.hMul c x)) fun n => HMul.hMul (f c …
    ⊢ Filter.Tendsto (seminormFromConst_seq c f (HMul.hMul c x)) Filter.atTop (nhd …
  -/
  simpa [hterm] using tendsto_const_nhds.mul hlim
  /-
    🎉 no goals
  -/


/-- If `K` is a field, the function `seminormFromConst` is a `RingNorm` on `K`. -/
def normFromConst {k : K} {g : RingSeminorm K} (hg1 : g 1 ≤ 1) (hg_k : g k ≠ 0)
    (hg_pm : IsPowMul g) : RingNorm K :=
  (seminormFromConst hg1 hg_k hg_pm).toRingNorm (RingSeminorm.ne_zero_iff.mpr
             /-
               K : Type ?u.88181
               inst✝ : Field K
               k : K
               g : RingSeminorm K
               hg1 : LE.le (g 1) 1
               hg_k : Ne (g k) 0
               hg_pm : IsPowMul ⇑g
               ⊢ Ne ((seminormFromConst hg1 hg_k hg_pm) k) 0
             -/
      ⟨k, by simpa [seminormFromConst_def, seminormFromConst_apply_c] using hg_k⟩)
             /-
               🎉 no goals
             -/


theorem seminormFromConstRingNormOfField_def {k : K} {g : RingSeminorm K} (hg1 : g 1 ≤ 1)
    (hg_k : g k ≠ 0) (hg_pm : IsPowMul g) (x : K) :
    normFromConst hg1 hg_k hg_pm x = seminormFromConst' hg1 hg_k hg_pm x := rfl


