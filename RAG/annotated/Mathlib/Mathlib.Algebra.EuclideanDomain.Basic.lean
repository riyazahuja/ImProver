/-- The well founded relation in a Euclidean Domain satisfying `a % b ≺ b` for `b ≠ 0`  -/
local infixl:50 " ≺ " => EuclideanDomain.R

-- See note [lower instance priority]

instance (priority := 100) toMulDivCancelClass : MulDivCancelClass R where
  mul_div_cancel a b hb := by
    /-
      R : Type u
      inst✝ : EuclideanDomain R
      a b : R
      hb : Ne b 0
      ⊢ Eq (HDiv.hDiv (HMul.hMul a b) b) a
    -/
    refine (eq_of_sub_eq_zero ?_).symm
    /-
      R : Type u
      inst✝ : EuclideanDomain R
      a b : R
      hb : Ne b 0
      ⊢ Eq (HSub.hSub a (HDiv.hDiv (HMul.hMul a b) b)) 0
    -/
    by_contra h
    /-
      R : Type u
      inst✝ : EuclideanDomain R
      a b : R
      hb : Ne b 0
      h : Not (Eq (HSub.hSub a (HDiv.hDiv (HMul.hMul a b) b)) 0)
      ⊢ False
    -/
    have := mul_right_not_lt b h
    /-
      R : Type u
      inst✝ : EuclideanDomain R
      a b : R
      hb : Ne b 0
      h : Not (Eq (HSub.hSub a (HDiv.hDiv (HMul.hMul a b) b)) 0)
      this : Not (EuclideanDomain.r (HMul.hMul (HSub.hSub a (HDiv.hDiv (HMul.hMul a  …
      ⊢ False
    -/
    rw [sub_mul, mul_comm (_ / _), sub_eq_iff_eq_add'.2 (div_add_mod (a * b) b).symm] at this
    /-
      R : Type u
      inst✝ : EuclideanDomain R
      a b : R
      hb : Ne b 0
      h : Not (Eq (HSub.hSub a (HDiv.hDiv (HMul.hMul a b) b)) 0)
      this : Not (EuclideanDomain.r (HMod.hMod (HMul.hMul a b) b) b)
      ⊢ False
    -/
    exact this (mod_lt _ hb)
    /-
      🎉 no goals
    -/


@[simp]
theorem mod_eq_zero {a b : R} : a % b = 0 ↔ b ∣ a :=
  ⟨fun h => by
    /-
      R : Type u
      inst✝ : EuclideanDomain R
      a b : R
      h : Eq (HMod.hMod a b) 0
      ⊢ Dvd.dvd b a
    -/
    rw [← div_add_mod a b, h, add_zero]
    /-
      R : Type u
      inst✝ : EuclideanDomain R
      a b : R
      h : Eq (HMod.hMod a b) 0
      ⊢ Dvd.dvd b (HMul.hMul b (HDiv.hDiv a b))
    -/
    exact dvd_mul_right _ _, fun ⟨c, e⟩ => by
    /-
      🎉 no goals
    -/
    /-
      R : Type u
      inst✝ : EuclideanDomain R
      a b : R
      x✝ : Dvd.dvd b a
      c : R
      e : Eq a (HMul.hMul b c)
      ⊢ Eq (HMod.hMod a b) 0
    -/
    rw [e, ← add_left_cancel_iff, div_add_mod, add_zero]
    /-
      R : Type u
      inst✝ : EuclideanDomain R
      a b : R
      x✝ : Dvd.dvd b a
      c : R
      e : Eq a (HMul.hMul b c)
      ⊢ Eq (HMul.hMul b c) (HMul.hMul b (HDiv.hDiv (HMul.hMul b c) b))
    -/
    haveI := Classical.dec
    /-
      R : Type u
      inst✝ : EuclideanDomain R
      a b : R
      x✝ : Dvd.dvd b a
      c : R
      e : Eq a (HMul.hMul b c)
      this : (p : Prop) → Decidable p
      ⊢ Eq (HMul.hMul b c) (HMul.hMul b (HDiv.hDiv (HMul.hMul b c) b))
    -/
    by_cases b0 : b = 0
      /-
        case pos
        R : Type u
        inst✝ : EuclideanDomain R
        a b : R
        x✝ : Dvd.dvd b a
        c : R
        e : Eq a (HMul.hMul b c)
        this : (p : Prop) → Decidable p
        b0 : Eq b 0
        ⊢ Eq (HMul.hMul b c) (HMul.hMul b (HDiv.hDiv (HMul.hMul b c) b))
      -/
    · simp only [b0, zero_mul]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u
        inst✝ : EuclideanDomain R
        a b : R
        x✝ : Dvd.dvd b a
        c : R
        e : Eq a (HMul.hMul b c)
        this : (p : Prop) → Decidable p
        b0 : Not (Eq b 0)
        ⊢ Eq (HMul.hMul b c) (HMul.hMul b (HDiv.hDiv (HMul.hMul b c) b))
      -/
    · rw [mul_div_cancel_left₀ _ b0]⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem mod_self (a : R) : a % a = 0 :=
  mod_eq_zero.2 dvd_rfl


theorem dvd_mod_iff {a b c : R} (h : c ∣ b) : c ∣ a % b ↔ c ∣ a := by
  /-
    R : Type u
    inst✝ : EuclideanDomain R
    a b c : R
    h : Dvd.dvd c b
    ⊢ Iff (Dvd.dvd c (HMod.hMod a b)) (Dvd.dvd c a)
  -/
  rw [← dvd_add_right (h.mul_right _), div_add_mod]
  /-
    🎉 no goals
  -/


@[simp]
theorem mod_one (a : R) : a % 1 = 0 :=
  mod_eq_zero.2 (one_dvd _)


@[simp]
theorem zero_mod (b : R) : 0 % b = 0 :=
  mod_eq_zero.2 (dvd_zero _)


@[simp]
theorem zero_div {a : R} : 0 / a = 0 :=
  by_cases (fun a0 : a = 0 => a0.symm ▸ div_zero 0) fun a0 => by
    /-
      R : Type u
      inst✝ : EuclideanDomain R
      a : R
      a0 : Not (Eq a 0)
      ⊢ Eq (HDiv.hDiv 0 a) 0
    -/
    simpa only [zero_mul] using mul_div_cancel_right₀ 0 a0
    /-
      🎉 no goals
    -/


@[simp]
theorem div_self {a : R} (a0 : a ≠ 0) : a / a = 1 := by
  /-
    R : Type u
    inst✝ : EuclideanDomain R
    a : R
    a0 : Ne a 0
    ⊢ Eq (HDiv.hDiv a a) 1
  -/
  simpa only [one_mul] using mul_div_cancel_right₀ 1 a0
  /-
    🎉 no goals
  -/


theorem eq_div_of_mul_eq_left {a b c : R} (hb : b ≠ 0) (h : a * b = c) : a = c / b := by
  /-
    R : Type u
    inst✝ : EuclideanDomain R
    a b c : R
    hb : Ne b 0
    h : Eq (HMul.hMul a b) c
    ⊢ Eq a (HDiv.hDiv c b)
  -/
  rw [← h, mul_div_cancel_right₀ _ hb]
  /-
    🎉 no goals
  -/


theorem eq_div_of_mul_eq_right {a b c : R} (ha : a ≠ 0) (h : a * b = c) : b = c / a := by
  /-
    R : Type u
    inst✝ : EuclideanDomain R
    a b c : R
    ha : Ne a 0
    h : Eq (HMul.hMul a b) c
    ⊢ Eq b (HDiv.hDiv c a)
  -/
  rw [← h, mul_div_cancel_left₀ _ ha]
  /-
    🎉 no goals
  -/


theorem mul_div_assoc (x : R) {y z : R} (h : z ∣ y) : x * y / z = x * (y / z) := by
  /-
    R : Type u
    inst✝ : EuclideanDomain R
    x y z : R
    h : Dvd.dvd z y
    ⊢ Eq (HDiv.hDiv (HMul.hMul x y) z) (HMul.hMul x (HDiv.hDiv y z))
  -/
  by_cases hz : z = 0
    /-
      case pos
      R : Type u
      inst✝ : EuclideanDomain R
      x y z : R
      h : Dvd.dvd z y
      hz : Eq z 0
      ⊢ Eq (HDiv.hDiv (HMul.hMul x y) z) (HMul.hMul x (HDiv.hDiv y z))
    -/
  · subst hz
    /-
      case pos
      R : Type u
      inst✝ : EuclideanDomain R
      x y : R
      h : Dvd.dvd 0 y
      ⊢ Eq (HDiv.hDiv (HMul.hMul x y) 0) (HMul.hMul x (HDiv.hDiv y 0))
    -/
    rw [div_zero, div_zero, mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝ : EuclideanDomain R
    x y z : R
    h : Dvd.dvd z y
    hz : Not (Eq z 0)
    ⊢ Eq (HDiv.hDiv (HMul.hMul x y) z) (HMul.hMul x (HDiv.hDiv y z))
  -/
  rcases h with ⟨p, rfl⟩
  /-
    case neg.intro
    R : Type u
    inst✝ : EuclideanDomain R
    x z : R
    hz : Not (Eq z 0)
    p : R
    ⊢ Eq (HDiv.hDiv (HMul.hMul x (HMul.hMul z p)) z) (HMul.hMul x (HDiv.hDiv (HMul …
  -/
  rw [mul_div_cancel_left₀ _ hz, mul_left_comm, mul_div_cancel_left₀ _ hz]
  /-
    🎉 no goals
  -/


protected theorem mul_div_cancel' {a b : R} (hb : b ≠ 0) (hab : b ∣ a) : b * (a / b) = a := by
  /-
    R : Type u
    inst✝ : EuclideanDomain R
    a b : R
    hb : Ne b 0
    hab : Dvd.dvd b a
    ⊢ Eq (HMul.hMul b (HDiv.hDiv a b)) a
  -/
  rw [← mul_div_assoc _ hab, mul_div_cancel_left₀ _ hb]
  /-
    🎉 no goals
  -/

-- This generalizes `Int.div_one`, see note [simp-normal form]

@[simp]
theorem div_one (p : R) : p / 1 = p :=
  (EuclideanDomain.eq_div_of_mul_eq_left (one_ne_zero' R) (mul_one p)).symm


theorem div_dvd_of_dvd {p q : R} (hpq : q ∣ p) : p / q ∣ p := by
  /-
    R : Type u
    inst✝ : EuclideanDomain R
    p q : R
    hpq : Dvd.dvd q p
    ⊢ Dvd.dvd (HDiv.hDiv p q) p
  -/
  by_cases hq : q = 0
    /-
      case pos
      R : Type u
      inst✝ : EuclideanDomain R
      p q : R
      hpq : Dvd.dvd q p
      hq : Eq q 0
      ⊢ Dvd.dvd (HDiv.hDiv p q) p
    -/
  · rw [hq, zero_dvd_iff] at hpq
    /-
      case pos
      R : Type u
      inst✝ : EuclideanDomain R
      p q : R
      hpq : Eq p 0
      hq : Eq q 0
      ⊢ Dvd.dvd (HDiv.hDiv p q) p
    -/
    rw [hpq]
    /-
      case pos
      R : Type u
      inst✝ : EuclideanDomain R
      p q : R
      hpq : Eq p 0
      hq : Eq q 0
      ⊢ Dvd.dvd (HDiv.hDiv 0 q) 0
    -/
    exact dvd_zero _
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝ : EuclideanDomain R
    p q : R
    hpq : Dvd.dvd q p
    hq : Not (Eq q 0)
    ⊢ Dvd.dvd (HDiv.hDiv p q) p
  -/
  use q
  /-
    case h
    R : Type u
    inst✝ : EuclideanDomain R
    p q : R
    hpq : Dvd.dvd q p
    hq : Not (Eq q 0)
    ⊢ Eq p (HMul.hMul (HDiv.hDiv p q) q)
  -/
  rw [mul_comm, ← EuclideanDomain.mul_div_assoc _ hpq, mul_comm, mul_div_cancel_right₀ _ hq]
  /-
    🎉 no goals
  -/


theorem dvd_div_of_mul_dvd {a b c : R} (h : a * b ∣ c) : b ∣ c / a := by
  /-
    R : Type u
    inst✝ : EuclideanDomain R
    a b c : R
    h : Dvd.dvd (HMul.hMul a b) c
    ⊢ Dvd.dvd b (HDiv.hDiv c a)
  -/
  rcases eq_or_ne a 0 with (rfl | ha)
    /-
      case inl
      R : Type u
      inst✝ : EuclideanDomain R
      b c : R
      h : Dvd.dvd (HMul.hMul 0 b) c
      ⊢ Dvd.dvd b (HDiv.hDiv c 0)
    -/
  · simp only [div_zero, dvd_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    inst✝ : EuclideanDomain R
    a b c : R
    h : Dvd.dvd (HMul.hMul a b) c
    ha : Ne a 0
    ⊢ Dvd.dvd b (HDiv.hDiv c a)
  -/
  rcases h with ⟨d, rfl⟩
  /-
    case inr.intro
    R : Type u
    inst✝ : EuclideanDomain R
    a b : R
    ha : Ne a 0
    d : R
    ⊢ Dvd.dvd b (HDiv.hDiv (HMul.hMul (HMul.hMul a b) d) a)
  -/
  refine ⟨d, ?_⟩
  /-
    case inr.intro
    R : Type u
    inst✝ : EuclideanDomain R
    a b : R
    ha : Ne a 0
    d : R
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul a b) d) a) (HMul.hMul b d)
  -/
  rw [mul_assoc, mul_div_cancel_left₀ _ ha]
  /-
    🎉 no goals
  -/


@[simp]
theorem gcd_zero_right (a : R) : gcd a 0 = a := by
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    a : R
    ⊢ Eq (EuclideanDomain.gcd a 0) a
  -/
  rw [gcd]
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    a : R
    ⊢ Eq (dite (Eq a 0) (fun a0 => 0) fun a0 => letFun ⋯ fun x => EuclideanDomain. …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp only [h, zero_mod, gcd_zero_left]
                       /-
                         🎉 no goals
                       -/


theorem gcd_val (a b : R) : gcd a b = gcd (b % a) a := by
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    a b : R
    ⊢ Eq (EuclideanDomain.gcd a b) (EuclideanDomain.gcd (HMod.hMod b a) a)
  -/
  rw [gcd]
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    a b : R
    ⊢ Eq (dite (Eq a 0) (fun a0 => b) fun a0 => letFun ⋯ fun x => EuclideanDomain. …
  -/
  split_ifs with h <;> [simp only [h, mod_zero, gcd_zero_right]; rfl]
  /-
    🎉 no goals
  -/


theorem gcd_dvd (a b : R) : gcd a b ∣ a ∧ gcd a b ∣ b :=
  GCD.induction a b
    (fun b => by
      /-
        R : Type u
        inst✝¹ : EuclideanDomain R
        inst✝ : DecidableEq R
        a b✝ b : R
        ⊢ And (Dvd.dvd (EuclideanDomain.gcd 0 b) 0) (Dvd.dvd (EuclideanDomain.gcd 0 b) …
      -/
      rw [gcd_zero_left]
      /-
        R : Type u
        inst✝¹ : EuclideanDomain R
        inst✝ : DecidableEq R
        a b✝ b : R
        ⊢ And (Dvd.dvd b 0) (Dvd.dvd b b)
      -/
      exact ⟨dvd_zero _, dvd_rfl⟩)
      /-
        🎉 no goals
      -/
    fun a b _ ⟨IH₁, IH₂⟩ => by
    /-
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      a✝ b✝ a b : R
      x✝¹ : Ne a 0
      x✝ : And (Dvd.dvd (EuclideanDomain.gcd (HMod.hMod b a) a) (HMod.hMod b a)) (Dv …
      IH₁ : Dvd.dvd (EuclideanDomain.gcd (HMod.hMod b a) a) (HMod.hMod b a)
      IH₂ : Dvd.dvd (EuclideanDomain.gcd (HMod.hMod b a) a) a
      ⊢ And (Dvd.dvd (EuclideanDomain.gcd a b) a) (Dvd.dvd (EuclideanDomain.gcd a b) …
    -/
    rw [gcd_val]
    /-
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      a✝ b✝ a b : R
      x✝¹ : Ne a 0
      x✝ : And (Dvd.dvd (EuclideanDomain.gcd (HMod.hMod b a) a) (HMod.hMod b a)) (Dv …
      IH₁ : Dvd.dvd (EuclideanDomain.gcd (HMod.hMod b a) a) (HMod.hMod b a)
      IH₂ : Dvd.dvd (EuclideanDomain.gcd (HMod.hMod b a) a) a
      ⊢ And (Dvd.dvd (EuclideanDomain.gcd (HMod.hMod b a) a) a) (Dvd.dvd (EuclideanD …
    -/
    exact ⟨IH₂, (dvd_mod_iff IH₂).1 IH₁⟩
    /-
      🎉 no goals
    -/


theorem gcd_dvd_left (a b : R) : gcd a b ∣ a :=
  (gcd_dvd a b).left


theorem gcd_dvd_right (a b : R) : gcd a b ∣ b :=
  (gcd_dvd a b).right


protected theorem gcd_eq_zero_iff {a b : R} : gcd a b = 0 ↔ a = 0 ∧ b = 0 :=
               /-
                 R : Type u
                 inst✝¹ : EuclideanDomain R
                 inst✝ : DecidableEq R
                 a b : R
                 h : Eq (EuclideanDomain.gcd a b) 0
                 ⊢ And (Eq a 0) (Eq b 0)
               -/
  ⟨fun h => by simpa [h] using gcd_dvd a b, by
               /-
                 🎉 no goals
               -/
    /-
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      a b : R
      ⊢ And (Eq a 0) (Eq b 0) → Eq (EuclideanDomain.gcd a b) 0
    -/
    rintro ⟨rfl, rfl⟩
    /-
      case intro
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      ⊢ Eq (EuclideanDomain.gcd 0 0) 0
    -/
    exact gcd_zero_right _⟩
    /-
      🎉 no goals
    -/


theorem dvd_gcd {a b c : R} : c ∣ a → c ∣ b → c ∣ gcd a b :=
                                     /-
                                       R : Type u
                                       inst✝¹ : EuclideanDomain R
                                       inst✝ : DecidableEq R
                                       a b c x✝¹ : R
                                       x✝ : Dvd.dvd c 0
                                       H : Dvd.dvd c x✝¹
                                       ⊢ Dvd.dvd c (EuclideanDomain.gcd 0 x✝¹)
                                     -/
  GCD.induction a b (fun _ _ H => by simpa only [gcd_zero_left] using H) fun a b _ IH ca cb => by
                                     /-
                                       🎉 no goals
                                     -/
    /-
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      a✝ b✝ c a b : R
      x✝ : Ne a 0
      IH : Dvd.dvd c (HMod.hMod b a) → Dvd.dvd c a → Dvd.dvd c (EuclideanDomain.gcd  …
      ca : Dvd.dvd c a
      cb : Dvd.dvd c b
      ⊢ Dvd.dvd c (EuclideanDomain.gcd a b)
    -/
    rw [gcd_val]
    /-
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      a✝ b✝ c a b : R
      x✝ : Ne a 0
      IH : Dvd.dvd c (HMod.hMod b a) → Dvd.dvd c a → Dvd.dvd c (EuclideanDomain.gcd  …
      ca : Dvd.dvd c a
      cb : Dvd.dvd c b
      ⊢ Dvd.dvd c (EuclideanDomain.gcd (HMod.hMod b a) a)
    -/
    exact IH ((dvd_mod_iff ca).2 cb) ca
    /-
      🎉 no goals
    -/


theorem gcd_eq_left {a b : R} : gcd a b = a ↔ a ∣ b :=
  ⟨fun h => by
    /-
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      a b : R
      h : Eq (EuclideanDomain.gcd a b) a
      ⊢ Dvd.dvd a b
    -/
    rw [← h]
    /-
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      a b : R
      h : Eq (EuclideanDomain.gcd a b) a
      ⊢ Dvd.dvd (EuclideanDomain.gcd a b) b
    -/
    /-
      🎉 no goals
    -/
    apply gcd_dvd_right, fun h => by rw [gcd_val, mod_eq_zero.2 h, gcd_zero_left]⟩
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem gcd_one_left (a : R) : gcd 1 a = 1 :=
  gcd_eq_left.2 (one_dvd _)


@[simp]
theorem gcd_self (a : R) : gcd a a = a :=
  gcd_eq_left.2 dvd_rfl


@[simp]
theorem xgcdAux_fst (x y : R) : ∀ s t s' t', (xgcdAux x s t y s' t').1 = gcd x y :=
  GCD.induction x y
    (by
      /-
        R : Type u
        inst✝¹ : EuclideanDomain R
        inst✝ : DecidableEq R
        x y : R
        ⊢ ∀ (x s t s' t' : R), Eq (EuclideanDomain.xgcdAux 0 s t x s' t').1 (Euclidean …
      -/
      intros
      /-
        R : Type u
        inst✝¹ : EuclideanDomain R
        inst✝ : DecidableEq R
        x y x✝ s✝ t✝ s'✝ t'✝ : R
        ⊢ Eq (EuclideanDomain.xgcdAux 0 s✝ t✝ x✝ s'✝ t'✝).1 (EuclideanDomain.gcd 0 x✝)
      -/
      rw [xgcd_zero_left, gcd_zero_left])
      /-
        🎉 no goals
      -/
    fun x y h IH s t s' t' => by
    /-
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x✝ y✝ x y : R
      h : Ne x 0
      IH : ∀ (s t s' t' : R), Eq (EuclideanDomain.xgcdAux (HMod.hMod y x) s t x s' t …
      s t s' t' : R
      ⊢ Eq (EuclideanDomain.xgcdAux x s t y s' t').1 (EuclideanDomain.gcd x y)
    -/
    simp only [xgcdAux_rec h, if_neg h, IH]
    /-
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x✝ y✝ x y : R
      h : Ne x 0
      IH : ∀ (s t s' t' : R), Eq (EuclideanDomain.xgcdAux (HMod.hMod y x) s t x s' t …
      s t s' t' : R
      ⊢ Eq (EuclideanDomain.gcd (HMod.hMod y x) x) (EuclideanDomain.gcd x y)
    -/
    rw [← gcd_val]
    /-
      🎉 no goals
    -/


theorem xgcdAux_val (x y : R) : xgcdAux x 1 0 y 0 1 = (gcd x y, xgcd x y) := by
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    x y : R
    ⊢ Eq (EuclideanDomain.xgcdAux x 1 0 y 0 1) { fst := EuclideanDomain.gcd x y, s …
  -/
  rw [xgcd, ← xgcdAux_fst x y 1 0 0 1]
  /-
    🎉 no goals
  -/


private def P (a b : R) : R × R × R → Prop
  | (r, s, t) => (r : R) = a * s + b * t


theorem xgcdAux_P (a b : R) {r r' : R} {s t s' t'} (p : P a b (r, s, t))
    (p' : P a b (r', s', t')) : P a b (xgcdAux r s t r' s' t') := by
  induction r, r' using GCD.induction generalizing s t s' t' with
  | H0 n => simpa only [xgcd_zero_left]
  | H1 _ _ h IH =>
    rw [xgcdAux_rec h]
    refine IH ?_ p
    unfold P at p p' ⊢
    dsimp
    rw [mul_sub, mul_sub, add_sub, sub_add_eq_add_sub, ← p', sub_sub, mul_comm _ s, ← mul_assoc,
      mul_comm _ t, ← mul_assoc, ← add_mul, ← p, mod_eq_sub_mul_div]


/-- An explicit version of **Bézout's lemma** for Euclidean domains. -/
theorem gcd_eq_gcd_ab (a b : R) : (gcd a b : R) = a * gcdA a b + b * gcdB a b := by
  have :=
    @xgcdAux_P _ _ _ a b a b 1 0 0 1 (by dsimp [P]; rw [mul_one, mul_zero, add_zero])
      (by dsimp [P]; rw [mul_one, mul_zero, zero_add])
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    a b : R
    this : EuclideanDomain.P a b (EuclideanDomain.xgcdAux a 1 0 b 0 1)
    ⊢ Eq (EuclideanDomain.gcd a b) (HAdd.hAdd (HMul.hMul a (EuclideanDomain.gcdA a …
  -/
  rwa [xgcdAux_val, xgcd_val] at this
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

instance (priority := 70) (R : Type*) [e : EuclideanDomain R] : NoZeroDivisors R :=
  haveI := Classical.decEq R
  { eq_zero_or_eq_zero_of_mul_eq_zero := fun {a b} h =>
                                                /-
                                                  R✝ : Type u
                                                  inst✝¹ : EuclideanDomain R✝
                                                  inst✝ : DecidableEq R✝
                                                  R : Type u_1
                                                  e : EuclideanDomain R
                                                  this : DecidableEq R
                                                  a b : R
                                                  h : Eq (HMul.hMul a b) 0
                                                  h0 : And (Not (Eq a 0)) (Not (Eq b 0))
                                                  ⊢ Eq a 0
                                                -/
      or_iff_not_and_not.2 fun h0 => h0.1 <| by rw [← mul_div_cancel_right₀ a h0.2, h, zero_div] }
                                                /-
                                                  🎉 no goals
                                                -/

-- see Note [lower instance priority]

instance (priority := 70) (R : Type*) [e : EuclideanDomain R] : IsDomain R :=
  { e, NoZeroDivisors.to_isDomain R with }


theorem dvd_lcm_left (x y : R) : x ∣ lcm x y :=
  by_cases
    (fun hxy : gcd x y = 0 => by
      /-
        R : Type u
        inst✝¹ : EuclideanDomain R
        inst✝ : DecidableEq R
        x y : R
        hxy : Eq (EuclideanDomain.gcd x y) 0
        ⊢ Dvd.dvd x (EuclideanDomain.lcm x y)
      -/
      rw [lcm, hxy, div_zero]
      /-
        R : Type u
        inst✝¹ : EuclideanDomain R
        inst✝ : DecidableEq R
        x y : R
        hxy : Eq (EuclideanDomain.gcd x y) 0
        ⊢ Dvd.dvd x 0
      -/
      exact dvd_zero _)
      /-
        🎉 no goals
      -/
    fun hxy =>
    let ⟨z, hz⟩ := (gcd_dvd x y).2
                                                   /-
                                                     R : Type u
                                                     inst✝¹ : EuclideanDomain R
                                                     inst✝ : DecidableEq R
                                                     x y : R
                                                     hxy : Not (Eq (EuclideanDomain.gcd x y) 0)
                                                     z : R
                                                     hz : Eq y (HMul.hMul (EuclideanDomain.gcd x y) z)
                                                     ⊢ Eq (HMul.hMul (HMul.hMul x z) (EuclideanDomain.gcd x y)) (HMul.hMul x y)
                                                   -/
    ⟨z, Eq.symm <| eq_div_of_mul_eq_left hxy <| by rw [mul_right_comm, mul_assoc, ← hz]⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem dvd_lcm_right (x y : R) : y ∣ lcm x y :=
  by_cases
    (fun hxy : gcd x y = 0 => by
      /-
        R : Type u
        inst✝¹ : EuclideanDomain R
        inst✝ : DecidableEq R
        x y : R
        hxy : Eq (EuclideanDomain.gcd x y) 0
        ⊢ Dvd.dvd y (EuclideanDomain.lcm x y)
      -/
      rw [lcm, hxy, div_zero]
      /-
        R : Type u
        inst✝¹ : EuclideanDomain R
        inst✝ : DecidableEq R
        x y : R
        hxy : Eq (EuclideanDomain.gcd x y) 0
        ⊢ Dvd.dvd y 0
      -/
      exact dvd_zero _)
      /-
        🎉 no goals
      -/
    fun hxy =>
    let ⟨z, hz⟩ := (gcd_dvd x y).1
                                                    /-
                                                      R : Type u
                                                      inst✝¹ : EuclideanDomain R
                                                      inst✝ : DecidableEq R
                                                      x y : R
                                                      hxy : Not (Eq (EuclideanDomain.gcd x y) 0)
                                                      z : R
                                                      hz : Eq x (HMul.hMul (EuclideanDomain.gcd x y) z)
                                                      ⊢ Eq (HMul.hMul (EuclideanDomain.gcd x y) (HMul.hMul y z)) (HMul.hMul x y)
                                                    -/
    ⟨z, Eq.symm <| eq_div_of_mul_eq_right hxy <| by rw [← mul_assoc, mul_right_comm, ← hz]⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem lcm_dvd {x y z : R} (hxz : x ∣ z) (hyz : y ∣ z) : lcm x y ∣ z := by
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    x y z : R
    hxz : Dvd.dvd x z
    hyz : Dvd.dvd y z
    ⊢ Dvd.dvd (EuclideanDomain.lcm x y) z
  -/
  rw [lcm]
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    x y z : R
    hxz : Dvd.dvd x z
    hyz : Dvd.dvd y z
    ⊢ Dvd.dvd (HDiv.hDiv (HMul.hMul x y) (EuclideanDomain.gcd x y)) z
  -/
  by_cases hxy : gcd x y = 0
    /-
      case pos
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y z : R
      hxz : Dvd.dvd x z
      hyz : Dvd.dvd y z
      hxy : Eq (EuclideanDomain.gcd x y) 0
      ⊢ Dvd.dvd (HDiv.hDiv (HMul.hMul x y) (EuclideanDomain.gcd x y)) z
    -/
  · rw [hxy, div_zero]
    /-
      case pos
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y z : R
      hxz : Dvd.dvd x z
      hyz : Dvd.dvd y z
      hxy : Eq (EuclideanDomain.gcd x y) 0
      ⊢ Dvd.dvd 0 z
    -/
    rw [EuclideanDomain.gcd_eq_zero_iff] at hxy
    /-
      case pos
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y z : R
      hxz : Dvd.dvd x z
      hyz : Dvd.dvd y z
      hxy : And (Eq x 0) (Eq y 0)
      ⊢ Dvd.dvd 0 z
    -/
    rwa [hxy.1] at hxz
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    x y z : R
    hxz : Dvd.dvd x z
    hyz : Dvd.dvd y z
    hxy : Not (Eq (EuclideanDomain.gcd x y) 0)
    ⊢ Dvd.dvd (HDiv.hDiv (HMul.hMul x y) (EuclideanDomain.gcd x y)) z
  -/
  rcases gcd_dvd x y with ⟨⟨r, hr⟩, ⟨s, hs⟩⟩
  suffices x * y ∣ z * gcd x y by
    obtain ⟨p, hp⟩ := this
    use p
    generalize gcd x y = g at hxy hs hp ⊢
    subst hs
    rw [mul_left_comm, mul_div_cancel_left₀ _ hxy, ← mul_left_inj' hxy, hp]
    rw [← mul_assoc]
    simp only [mul_right_comm]
  /-
    case neg.intro.intro.intro
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    x y z : R
    hxz : Dvd.dvd x z
    hyz : Dvd.dvd y z
    hxy : Not (Eq (EuclideanDomain.gcd x y) 0)
    r : R
    hr : Eq x (HMul.hMul (EuclideanDomain.gcd x y) r)
    s : R
    hs : Eq y (HMul.hMul (EuclideanDomain.gcd x y) s)
    ⊢ Dvd.dvd (HMul.hMul x y) (HMul.hMul z (EuclideanDomain.gcd x y))
  -/
  rw [gcd_eq_gcd_ab, mul_add]
  /-
    case neg.intro.intro.intro
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    x y z : R
    hxz : Dvd.dvd x z
    hyz : Dvd.dvd y z
    hxy : Not (Eq (EuclideanDomain.gcd x y) 0)
    r : R
    hr : Eq x (HMul.hMul (EuclideanDomain.gcd x y) r)
    s : R
    hs : Eq y (HMul.hMul (EuclideanDomain.gcd x y) s)
    ⊢ Dvd.dvd (HMul.hMul x y) (HAdd.hAdd (HMul.hMul z (HMul.hMul x (EuclideanDomai …
  -/
  apply dvd_add
    /-
      case neg.intro.intro.intro.h₁
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y z : R
      hxz : Dvd.dvd x z
      hyz : Dvd.dvd y z
      hxy : Not (Eq (EuclideanDomain.gcd x y) 0)
      r : R
      hr : Eq x (HMul.hMul (EuclideanDomain.gcd x y) r)
      s : R
      hs : Eq y (HMul.hMul (EuclideanDomain.gcd x y) s)
      ⊢ Dvd.dvd (HMul.hMul x y) (HMul.hMul z (HMul.hMul x (EuclideanDomain.gcdA x y)))
    -/
  · rw [mul_left_comm]
    /-
      case neg.intro.intro.intro.h₁
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y z : R
      hxz : Dvd.dvd x z
      hyz : Dvd.dvd y z
      hxy : Not (Eq (EuclideanDomain.gcd x y) 0)
      r : R
      hr : Eq x (HMul.hMul (EuclideanDomain.gcd x y) r)
      s : R
      hs : Eq y (HMul.hMul (EuclideanDomain.gcd x y) s)
      ⊢ Dvd.dvd (HMul.hMul x y) (HMul.hMul x (HMul.hMul z (EuclideanDomain.gcdA x y)))
    -/
    exact mul_dvd_mul_left _ (hyz.mul_right _)
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.intro.intro.h₂
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y z : R
      hxz : Dvd.dvd x z
      hyz : Dvd.dvd y z
      hxy : Not (Eq (EuclideanDomain.gcd x y) 0)
      r : R
      hr : Eq x (HMul.hMul (EuclideanDomain.gcd x y) r)
      s : R
      hs : Eq y (HMul.hMul (EuclideanDomain.gcd x y) s)
      ⊢ Dvd.dvd (HMul.hMul x y) (HMul.hMul z (HMul.hMul y (EuclideanDomain.gcdB x y)))
    -/
  · rw [mul_left_comm, mul_comm]
    /-
      case neg.intro.intro.intro.h₂
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y z : R
      hxz : Dvd.dvd x z
      hyz : Dvd.dvd y z
      hxy : Not (Eq (EuclideanDomain.gcd x y) 0)
      r : R
      hr : Eq x (HMul.hMul (EuclideanDomain.gcd x y) r)
      s : R
      hs : Eq y (HMul.hMul (EuclideanDomain.gcd x y) s)
      ⊢ Dvd.dvd (HMul.hMul y x) (HMul.hMul y (HMul.hMul z (EuclideanDomain.gcdB x y)))
    -/
    exact mul_dvd_mul_left _ (hxz.mul_right _)
    /-
      🎉 no goals
    -/


@[simp]
theorem lcm_dvd_iff {x y z : R} : lcm x y ∣ z ↔ x ∣ z ∧ y ∣ z :=
  ⟨fun hz => ⟨(dvd_lcm_left _ _).trans hz, (dvd_lcm_right _ _).trans hz⟩, fun ⟨hxz, hyz⟩ =>
    lcm_dvd hxz hyz⟩


@[simp]
                                                  /-
                                                    R : Type u
                                                    inst✝¹ : EuclideanDomain R
                                                    inst✝ : DecidableEq R
                                                    x : R
                                                    ⊢ Eq (EuclideanDomain.lcm 0 x) 0
                                                  -/
theorem lcm_zero_left (x : R) : lcm 0 x = 0 := by rw [lcm, zero_mul, zero_div]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
                                                   /-
                                                     R : Type u
                                                     inst✝¹ : EuclideanDomain R
                                                     inst✝ : DecidableEq R
                                                     x : R
                                                     ⊢ Eq (EuclideanDomain.lcm x 0) 0
                                                   -/
theorem lcm_zero_right (x : R) : lcm x 0 = 0 := by rw [lcm, mul_zero, zero_div]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem lcm_eq_zero_iff {x y : R} : lcm x y = 0 ↔ x = 0 ∨ y = 0 := by
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    x y : R
    ⊢ Iff (Eq (EuclideanDomain.lcm x y) 0) (Or (Eq x 0) (Eq y 0))
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y : R
      ⊢ Eq (EuclideanDomain.lcm x y) 0 → Or (Eq x 0) (Eq y 0)
    -/
  · intro hxy
    /-
      case mp
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y : R
      hxy : Eq (EuclideanDomain.lcm x y) 0
      ⊢ Or (Eq x 0) (Eq y 0)
    -/
    rw [lcm, mul_div_assoc _ (gcd_dvd_right _ _), mul_eq_zero] at hxy
    /-
      case mp
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y : R
      hxy : Or (Eq x 0) (Eq (HDiv.hDiv y (EuclideanDomain.gcd x y)) 0)
      ⊢ Or (Eq x 0) (Eq y 0)
    -/
    apply Or.imp_right _ hxy
    /-
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y : R
      hxy : Or (Eq x 0) (Eq (HDiv.hDiv y (EuclideanDomain.gcd x y)) 0)
      ⊢ Eq (HDiv.hDiv y (EuclideanDomain.gcd x y)) 0 → Eq y 0
    -/
    intro hy
    /-
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y : R
      hxy : Or (Eq x 0) (Eq (HDiv.hDiv y (EuclideanDomain.gcd x y)) 0)
      hy : Eq (HDiv.hDiv y (EuclideanDomain.gcd x y)) 0
      ⊢ Eq y 0
    -/
    by_cases hgxy : gcd x y = 0
      /-
        case pos
        R : Type u
        inst✝¹ : EuclideanDomain R
        inst✝ : DecidableEq R
        x y : R
        hxy : Or (Eq x 0) (Eq (HDiv.hDiv y (EuclideanDomain.gcd x y)) 0)
        hy : Eq (HDiv.hDiv y (EuclideanDomain.gcd x y)) 0
        hgxy : Eq (EuclideanDomain.gcd x y) 0
        ⊢ Eq y 0
      -/
    · rw [EuclideanDomain.gcd_eq_zero_iff] at hgxy
      /-
        case pos
        R : Type u
        inst✝¹ : EuclideanDomain R
        inst✝ : DecidableEq R
        x y : R
        hxy : Or (Eq x 0) (Eq (HDiv.hDiv y (EuclideanDomain.gcd x y)) 0)
        hy : Eq (HDiv.hDiv y (EuclideanDomain.gcd x y)) 0
        hgxy : And (Eq x 0) (Eq y 0)
        ⊢ Eq y 0
      -/
      exact hgxy.2
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u
        inst✝¹ : EuclideanDomain R
        inst✝ : DecidableEq R
        x y : R
        hxy : Or (Eq x 0) (Eq (HDiv.hDiv y (EuclideanDomain.gcd x y)) 0)
        hy : Eq (HDiv.hDiv y (EuclideanDomain.gcd x y)) 0
        hgxy : Not (Eq (EuclideanDomain.gcd x y) 0)
        ⊢ Eq y 0
      -/
    · rcases gcd_dvd x y with ⟨⟨r, hr⟩, ⟨s, hs⟩⟩
      /-
        case neg.intro.intro.intro
        R : Type u
        inst✝¹ : EuclideanDomain R
        inst✝ : DecidableEq R
        x y : R
        hxy : Or (Eq x 0) (Eq (HDiv.hDiv y (EuclideanDomain.gcd x y)) 0)
        hy : Eq (HDiv.hDiv y (EuclideanDomain.gcd x y)) 0
        hgxy : Not (Eq (EuclideanDomain.gcd x y) 0)
        r : R
        hr : Eq x (HMul.hMul (EuclideanDomain.gcd x y) r)
        s : R
        hs : Eq y (HMul.hMul (EuclideanDomain.gcd x y) s)
        ⊢ Eq y 0
      -/
      generalize gcd x y = g at hr hs hy hgxy ⊢
      /-
        case neg.intro.intro.intro
        R : Type u
        inst✝¹ : EuclideanDomain R
        inst✝ : DecidableEq R
        x y : R
        hxy : Or (Eq x 0) (Eq (HDiv.hDiv y (EuclideanDomain.gcd x y)) 0)
        r s g : R
        hr : Eq x (HMul.hMul g r)
        hs : Eq y (HMul.hMul g s)
        hy : Eq (HDiv.hDiv y g) 0
        hgxy : Not (Eq g 0)
        ⊢ Eq y 0
      -/
      subst hs
      /-
        case neg.intro.intro.intro
        R : Type u
        inst✝¹ : EuclideanDomain R
        inst✝ : DecidableEq R
        x r s g : R
        hr : Eq x (HMul.hMul g r)
        hgxy : Not (Eq g 0)
        hxy : Or (Eq x 0) (Eq (HDiv.hDiv (HMul.hMul g s) (EuclideanDomain.gcd x (HMul. …
        hy : Eq (HDiv.hDiv (HMul.hMul g s) g) 0
        ⊢ Eq (HMul.hMul g s) 0
      -/
      rw [mul_div_cancel_left₀ _ hgxy] at hy
      /-
        case neg.intro.intro.intro
        R : Type u
        inst✝¹ : EuclideanDomain R
        inst✝ : DecidableEq R
        x r s g : R
        hr : Eq x (HMul.hMul g r)
        hgxy : Not (Eq g 0)
        hxy : Or (Eq x 0) (Eq (HDiv.hDiv (HMul.hMul g s) (EuclideanDomain.gcd x (HMul. …
        hy : Eq s 0
        ⊢ Eq (HMul.hMul g s) 0
      -/
      rw [hy, mul_zero]
      /-
        🎉 no goals
      -/
  /-
    case mpr
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    x y : R
    ⊢ Or (Eq x 0) (Eq y 0) → Eq (EuclideanDomain.lcm x y) 0
  -/
  rintro (hx | hy)
    /-
      case mpr.inl
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y : R
      hx : Eq x 0
      ⊢ Eq (EuclideanDomain.lcm x y) 0
    -/
  · rw [hx, lcm_zero_left]
    /-
      🎉 no goals
    -/
    /-
      case mpr.inr
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y : R
      hy : Eq y 0
      ⊢ Eq (EuclideanDomain.lcm x y) 0
    -/
  · rw [hy, lcm_zero_right]
    /-
      🎉 no goals
    -/


@[simp]
theorem gcd_mul_lcm (x y : R) : gcd x y * lcm x y = x * y := by
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    x y : R
    ⊢ Eq (HMul.hMul (EuclideanDomain.gcd x y) (EuclideanDomain.lcm x y)) (HMul.hMu …
  -/
  rw [lcm]; by_cases h : gcd x y = 0
    /-
      case pos
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y : R
      h : Eq (EuclideanDomain.gcd x y) 0
      ⊢ Eq (HMul.hMul (EuclideanDomain.gcd x y) (HDiv.hDiv (HMul.hMul x y) (Euclidea …
    -/
  · rw [h, zero_mul]
    /-
      case pos
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y : R
      h : Eq (EuclideanDomain.gcd x y) 0
      ⊢ Eq 0 (HMul.hMul x y)
    -/
    rw [EuclideanDomain.gcd_eq_zero_iff] at h
    /-
      case pos
      R : Type u
      inst✝¹ : EuclideanDomain R
      inst✝ : DecidableEq R
      x y : R
      h : And (Eq x 0) (Eq y 0)
      ⊢ Eq 0 (HMul.hMul x y)
    -/
    rw [h.1, zero_mul]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    x y : R
    h : Not (Eq (EuclideanDomain.gcd x y) 0)
    ⊢ Eq (HMul.hMul (EuclideanDomain.gcd x y) (HDiv.hDiv (HMul.hMul x y) (Euclidea …
  -/
  rcases gcd_dvd x y with ⟨⟨r, hr⟩, ⟨s, hs⟩⟩
  /-
    case neg.intro.intro.intro
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    x y : R
    h : Not (Eq (EuclideanDomain.gcd x y) 0)
    r : R
    hr : Eq x (HMul.hMul (EuclideanDomain.gcd x y) r)
    s : R
    hs : Eq y (HMul.hMul (EuclideanDomain.gcd x y) s)
    ⊢ Eq (HMul.hMul (EuclideanDomain.gcd x y) (HDiv.hDiv (HMul.hMul x y) (Euclidea …
  -/
  generalize gcd x y = g at h hr ⊢; subst hr
  /-
    case neg.intro.intro.intro
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    y r s g : R
    h : Not (Eq g 0)
    hs : Eq y (HMul.hMul (EuclideanDomain.gcd (HMul.hMul g r) y) s)
    ⊢ Eq (HMul.hMul g (HDiv.hDiv (HMul.hMul (HMul.hMul g r) y) g)) (HMul.hMul (HMu …
  -/
  rw [mul_assoc, mul_div_cancel_left₀ _ h]
  /-
    🎉 no goals
  -/


theorem mul_div_mul_cancel {a b c : R} (ha : a ≠ 0) (hcb : c ∣ b) : a * b / (a * c) = b / c := by
  /-
    R : Type u
    inst✝ : EuclideanDomain R
    a b c : R
    ha : Ne a 0
    hcb : Dvd.dvd c b
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b) (HMul.hMul a c)) (HDiv.hDiv b c)
  -/
  by_cases hc : c = 0; · simp [hc]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    R : Type u
    inst✝ : EuclideanDomain R
    a b c : R
    ha : Ne a 0
    hcb : Dvd.dvd c b
    hc : Not (Eq c 0)
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b) (HMul.hMul a c)) (HDiv.hDiv b c)
  -/
  refine eq_div_of_mul_eq_right hc (mul_left_cancel₀ ha ?_)
  rw [← mul_assoc, ← mul_div_assoc _ (mul_dvd_mul_left a hcb),
    mul_div_cancel_left₀ _ (mul_ne_zero ha hc)]


theorem mul_div_mul_comm_of_dvd_dvd {a b c d : R} (hac : c ∣ a) (hbd : d ∣ b) :
    a * b / (c * d) = a / c * (b / d) := by
  /-
    R : Type u
    inst✝ : EuclideanDomain R
    a b c d : R
    hac : Dvd.dvd c a
    hbd : Dvd.dvd d b
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b) (HMul.hMul c d)) (HMul.hMul (HDiv.hDiv a c) (H …
  -/
  rcases eq_or_ne c 0 with (rfl | hc0); · simp
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case inr
    R : Type u
    inst✝ : EuclideanDomain R
    a b c d : R
    hac : Dvd.dvd c a
    hbd : Dvd.dvd d b
    hc0 : Ne c 0
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b) (HMul.hMul c d)) (HMul.hMul (HDiv.hDiv a c) (H …
  -/
  rcases eq_or_ne d 0 with (rfl | hd0); · simp
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case inr.inr
    R : Type u
    inst✝ : EuclideanDomain R
    a b c d : R
    hac : Dvd.dvd c a
    hbd : Dvd.dvd d b
    hc0 : Ne c 0
    hd0 : Ne d 0
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b) (HMul.hMul c d)) (HMul.hMul (HDiv.hDiv a c) (H …
  -/
  obtain ⟨k1, rfl⟩ := hac
  /-
    case inr.inr.intro
    R : Type u
    inst✝ : EuclideanDomain R
    b c d : R
    hbd : Dvd.dvd d b
    hc0 : Ne c 0
    hd0 : Ne d 0
    k1 : R
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul c k1) b) (HMul.hMul c d)) (HMul.hMul (HD …
  -/
  obtain ⟨k2, rfl⟩ := hbd
  rw [mul_div_cancel_left₀ _ hc0, mul_div_cancel_left₀ _ hd0, mul_mul_mul_comm,
    mul_div_cancel_left₀ _ (mul_ne_zero hc0 hd0)]


