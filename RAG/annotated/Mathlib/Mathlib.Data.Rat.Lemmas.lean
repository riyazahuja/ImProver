theorem num_dvd (a) {b : ℤ} (b0 : b ≠ 0) : (a /. b).num ∣ a := by
  /-
    a b : Int
    b0 : Ne b 0
    ⊢ Dvd.dvd (Rat.divInt a b).num a
  -/
  cases' e : a /. b with n d h c
  /-
    case mk'
    a b : Int
    b0 : Ne b 0
    n : Int
    d : Nat
    h : Ne d 0
    c : n.natAbs.Coprime d
    e : Eq (Rat.divInt a b) { num := n, den := d, den_nz := h, reduced := c }
    ⊢ Dvd.dvd { num := n, den := d, den_nz := h, reduced := c }.num a
  -/
  rw [Rat.mk'_eq_divInt, divInt_eq_iff b0 (mod_cast h)] at e
  refine Int.natAbs_dvd.1 <| Int.dvd_natAbs.1 <| Int.natCast_dvd_natCast.2 <|
    c.dvd_of_dvd_mul_right ?_
  /-
    case mk'
    a b : Int
    b0 : Ne b 0
    n : Int
    d : Nat
    h : Ne d 0
    c : n.natAbs.Coprime d
    e : Eq (HMul.hMul a ↑d) (HMul.hMul n b)
    ⊢ Dvd.dvd n.natAbs (HMul.hMul a.natAbs d)
  -/
  have := congr_arg Int.natAbs e
  /-
    case mk'
    a b : Int
    b0 : Ne b 0
    n : Int
    d : Nat
    h : Ne d 0
    c : n.natAbs.Coprime d
    e : Eq (HMul.hMul a ↑d) (HMul.hMul n b)
    this : Eq (HMul.hMul a ↑d).natAbs (HMul.hMul n b).natAbs
    ⊢ Dvd.dvd n.natAbs (HMul.hMul a.natAbs d)
  -/
  simp only [Int.natAbs_mul, Int.natAbs_ofNat] at this; simp [this]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem den_dvd (a b : ℤ) : ((a /. b).den : ℤ) ∣ b := by
  /-
    a b : Int
    ⊢ Dvd.dvd (↑(Rat.divInt a b).den) b
  -/
  by_cases b0 : b = 0; · simp [b0]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    a b : Int
    b0 : Not (Eq b 0)
    ⊢ Dvd.dvd (↑(Rat.divInt a b).den) b
  -/
  cases' e : a /. b with n d h c
  /-
    case neg.mk'
    a b : Int
    b0 : Not (Eq b 0)
    n : Int
    d : Nat
    h : Ne d 0
    c : n.natAbs.Coprime d
    e : Eq (Rat.divInt a b) { num := n, den := d, den_nz := h, reduced := c }
    ⊢ Dvd.dvd (↑{ num := n, den := d, den_nz := h, reduced := c }.den) b
  -/
  rw [mk'_eq_divInt, divInt_eq_iff b0 (ne_of_gt (Int.natCast_pos.2 (Nat.pos_of_ne_zero h)))] at e
  /-
    case neg.mk'
    a b : Int
    b0 : Not (Eq b 0)
    n : Int
    d : Nat
    h : Ne d 0
    c : n.natAbs.Coprime d
    e : Eq (HMul.hMul a ↑d) (HMul.hMul n b)
    ⊢ Dvd.dvd (↑{ num := n, den := d, den_nz := h, reduced := c }.den) b
  -/
  refine Int.dvd_natAbs.1 <| Int.natCast_dvd_natCast.2 <| c.symm.dvd_of_dvd_mul_left ?_
  /-
    case neg.mk'
    a b : Int
    b0 : Not (Eq b 0)
    n : Int
    d : Nat
    h : Ne d 0
    c : n.natAbs.Coprime d
    e : Eq (HMul.hMul a ↑d) (HMul.hMul n b)
    ⊢ Dvd.dvd d (HMul.hMul n.natAbs b.natAbs)
  -/
  rw [← Int.natAbs_mul, ← Int.natCast_dvd_natCast, Int.dvd_natAbs, ← e]; simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem num_den_mk {q : ℚ} {n d : ℤ} (hd : d ≠ 0) (qdf : q = n /. d) :
    ∃ c : ℤ, n = c * q.num ∧ d = c * q.den := by
  /-
    q : Rat
    n d : Int
    hd : Ne d 0
    qdf : Eq q (Rat.divInt n d)
    ⊢ Exists fun c => And (Eq n (HMul.hMul c q.num)) (Eq d (HMul.hMul c ↑q.den))
  -/
  obtain rfl | hn := eq_or_ne n 0
    /-
      case inl
      q : Rat
      d : Int
      hd : Ne d 0
      qdf : Eq q (Rat.divInt 0 d)
      ⊢ Exists fun c => And (Eq 0 (HMul.hMul c q.num)) (Eq d (HMul.hMul c ↑q.den))
    -/
  · simp [qdf]
    /-
      🎉 no goals
    -/
  have : q.num * d = n * ↑q.den := by
    refine (divInt_eq_iff ?_ hd).mp ?_
    · exact Int.natCast_ne_zero.mpr (Rat.den_nz _)
    · rwa [num_divInt_den]
  have hqdn : q.num ∣ n := by
    rw [qdf]
    exact Rat.num_dvd _ hd
  /-
    case inr
    q : Rat
    n d : Int
    hd : Ne d 0
    qdf : Eq q (Rat.divInt n d)
    hn : Ne n 0
    this : Eq (HMul.hMul q.num d) (HMul.hMul n ↑q.den)
    hqdn : Dvd.dvd q.num n
    ⊢ Exists fun c => And (Eq n (HMul.hMul c q.num)) (Eq d (HMul.hMul c ↑q.den))
  -/
  refine ⟨n / q.num, ?_, ?_⟩
    /-
      case inr.refine_1
      q : Rat
      n d : Int
      hd : Ne d 0
      qdf : Eq q (Rat.divInt n d)
      hn : Ne n 0
      this : Eq (HMul.hMul q.num d) (HMul.hMul n ↑q.den)
      hqdn : Dvd.dvd q.num n
      ⊢ Eq n (HMul.hMul (HDiv.hDiv n q.num) q.num)
    -/
  · rw [Int.ediv_mul_cancel hqdn]
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_2
      q : Rat
      n d : Int
      hd : Ne d 0
      qdf : Eq q (Rat.divInt n d)
      hn : Ne n 0
      this : Eq (HMul.hMul q.num d) (HMul.hMul n ↑q.den)
      hqdn : Dvd.dvd q.num n
      ⊢ Eq d (HMul.hMul (HDiv.hDiv n q.num) ↑q.den)
    -/
  · refine Int.eq_mul_div_of_mul_eq_mul_of_dvd_left ?_ hqdn this
    /-
      case inr.refine_2
      q : Rat
      n d : Int
      hd : Ne d 0
      qdf : Eq q (Rat.divInt n d)
      hn : Ne n 0
      this : Eq (HMul.hMul q.num d) (HMul.hMul n ↑q.den)
      hqdn : Dvd.dvd q.num n
      ⊢ Ne q.num 0
    -/
    rw [qdf]
    /-
      case inr.refine_2
      q : Rat
      n d : Int
      hd : Ne d 0
      qdf : Eq q (Rat.divInt n d)
      hn : Ne n 0
      this : Eq (HMul.hMul q.num d) (HMul.hMul n ↑q.den)
      hqdn : Dvd.dvd q.num n
      ⊢ Ne (Rat.divInt n d).num 0
    -/
    exact Rat.num_ne_zero.2 ((divInt_ne_zero hd).mpr hn)
    /-
      🎉 no goals
    -/


theorem num_mk (n d : ℤ) : (n /. d).num = d.sign * n / n.gcd d := by
  have (m : ℕ) : Int.natAbs (m + 1) = m + 1 := by
    rw [← Nat.cast_one, ← Nat.cast_add, Int.natAbs_cast]
  /-
    n d : Int
    this : ∀ (m : Nat), Eq (HAdd.hAdd (↑m) 1).natAbs (HAdd.hAdd m 1)
    ⊢ Eq (Rat.divInt n d).num (HDiv.hDiv (HMul.hMul d.sign n) ↑(n.gcd d))
  -/
  rcases d with ((_ | _) | _) <;>
  /-
    case ofNat.zero
    n : Int
    this : ∀ (m : Nat), Eq (HAdd.hAdd (↑m) 1).natAbs (HAdd.hAdd m 1)
    ⊢ Eq (Rat.divInt n (Int.ofNat 0)).num (HDiv.hDiv (HMul.hMul (Int.ofNat 0).sign …
  -/
  rw [← Int.tdiv_eq_ediv_of_dvd] <;>
  simp [divInt, mkRat, Rat.normalize, Nat.succPNat, Int.sign, Int.gcd,
    Int.zero_ediv, Int.ofNat_dvd_left, Nat.gcd_dvd_left, this]


theorem den_mk (n d : ℤ) : (n /. d).den = if d = 0 then 1 else d.natAbs / n.gcd d := by
  have (m : ℕ) : Int.natAbs (m + 1) = m + 1 := by
    rw [← Nat.cast_one, ← Nat.cast_add, Int.natAbs_cast]
  /-
    n d : Int
    this : ∀ (m : Nat), Eq (HAdd.hAdd (↑m) 1).natAbs (HAdd.hAdd m 1)
    ⊢ Eq (Rat.divInt n d).den (ite (Eq d 0) 1 (HDiv.hDiv d.natAbs (n.gcd d)))
  -/
  rcases d with ((_ | _) | _) <;>
    simp [divInt, mkRat, Rat.normalize, Nat.succPNat, Int.sign, Int.gcd,
      if_neg (Nat.cast_add_one_ne_zero _), this]


theorem add_den_dvd_lcm (q₁ q₂ : ℚ) : (q₁ + q₂).den ∣ q₁.den.lcm q₂.den := by
  rw [add_def, normalize_eq, Nat.div_dvd_iff_dvd_mul (Nat.gcd_dvd_right _ _)
    (Nat.gcd_ne_zero_right (by simp)), ← Nat.gcd_mul_lcm,
    mul_dvd_mul_iff_right (Nat.lcm_ne_zero (by simp) (by simp)), Nat.dvd_gcd_iff]
  /-
    q₁ q₂ : Rat
    ⊢ And (Dvd.dvd (q₁.den.gcd q₂.den) (HAdd.hAdd (HMul.hMul q₁.num ↑q₂.den) (HMul …
  -/
  refine ⟨?_, dvd_mul_right _ _⟩
  /-
    q₁ q₂ : Rat
    ⊢ Dvd.dvd (q₁.den.gcd q₂.den) (HAdd.hAdd (HMul.hMul q₁.num ↑q₂.den) (HMul.hMul …
  -/
  rw [← Int.natCast_dvd_natCast, Int.dvd_natAbs]
  apply Int.dvd_add
    <;> apply dvd_mul_of_dvd_right <;> rw [Int.natCast_dvd_natCast]
    <;> [exact Nat.gcd_dvd_right _ _; exact Nat.gcd_dvd_left _ _]


theorem add_den_dvd (q₁ q₂ : ℚ) : (q₁ + q₂).den ∣ q₁.den * q₂.den := by
  /-
    q₁ q₂ : Rat
    ⊢ Dvd.dvd (HAdd.hAdd q₁ q₂).den (HMul.hMul q₁.den q₂.den)
  -/
  rw [add_def, normalize_eq]
  /-
    q₁ q₂ : Rat
    ⊢ Dvd.dvd { num := HDiv.hDiv (HAdd.hAdd (HMul.hMul q₁.num ↑q₂.den) (HMul.hMul  …
  -/
  apply Nat.div_dvd_of_dvd
  /-
    case h
    q₁ q₂ : Rat
    ⊢ Dvd.dvd ((HAdd.hAdd (HMul.hMul q₁.num ↑q₂.den) (HMul.hMul q₂.num ↑q₁.den)).n …
  -/
  apply Nat.gcd_dvd_right
  /-
    🎉 no goals
  -/


theorem mul_den_dvd (q₁ q₂ : ℚ) : (q₁ * q₂).den ∣ q₁.den * q₂.den := by
  /-
    q₁ q₂ : Rat
    ⊢ Dvd.dvd (HMul.hMul q₁ q₂).den (HMul.hMul q₁.den q₂.den)
  -/
  rw [mul_def, normalize_eq]
  /-
    q₁ q₂ : Rat
    ⊢ Dvd.dvd { num := HDiv.hDiv (HMul.hMul q₁.num q₂.num) ↑((HMul.hMul q₁.num q₂. …
  -/
  apply Nat.div_dvd_of_dvd
  /-
    case h
    q₁ q₂ : Rat
    ⊢ Dvd.dvd ((HMul.hMul q₁.num q₂.num).natAbs.gcd (HMul.hMul q₁.den q₂.den)) (HM …
  -/
  apply Nat.gcd_dvd_right
  /-
    🎉 no goals
  -/


theorem mul_num (q₁ q₂ : ℚ) :
    (q₁ * q₂).num = q₁.num * q₂.num / Nat.gcd (q₁.num * q₂.num).natAbs (q₁.den * q₂.den) := by
  /-
    q₁ q₂ : Rat
    ⊢ Eq (HMul.hMul q₁ q₂).num (HDiv.hDiv (HMul.hMul q₁.num q₂.num) ↑((HMul.hMul q …
  -/
  rw [mul_def, normalize_eq]
  /-
    🎉 no goals
  -/


theorem mul_den (q₁ q₂ : ℚ) :
    (q₁ * q₂).den =
      q₁.den * q₂.den / Nat.gcd (q₁.num * q₂.num).natAbs (q₁.den * q₂.den) := by
  /-
    q₁ q₂ : Rat
    ⊢ Eq (HMul.hMul q₁ q₂).den (HDiv.hDiv (HMul.hMul q₁.den q₂.den) ((HMul.hMul q₁ …
  -/
  rw [mul_def, normalize_eq]
  /-
    🎉 no goals
  -/


theorem mul_self_num (q : ℚ) : (q * q).num = q.num * q.num := by
  /-
    q : Rat
    ⊢ Eq (HMul.hMul q q).num (HMul.hMul q.num q.num)
  -/
  rw [mul_num, Int.natAbs_mul, Nat.Coprime.gcd_eq_one, Int.ofNat_one, Int.ediv_one]
  /-
    q : Rat
    ⊢ (HMul.hMul q.num.natAbs q.num.natAbs).Coprime (HMul.hMul q.den q.den)
  -/
  exact (q.reduced.mul_right q.reduced).mul (q.reduced.mul_right q.reduced)
  /-
    🎉 no goals
  -/


theorem mul_self_den (q : ℚ) : (q * q).den = q.den * q.den := by
  /-
    q : Rat
    ⊢ Eq (HMul.hMul q q).den (HMul.hMul q.den q.den)
  -/
  rw [Rat.mul_den, Int.natAbs_mul, Nat.Coprime.gcd_eq_one, Nat.div_one]
  /-
    q : Rat
    ⊢ (HMul.hMul q.num.natAbs q.num.natAbs).Coprime (HMul.hMul q.den q.den)
  -/
  exact (q.reduced.mul_right q.reduced).mul (q.reduced.mul_right q.reduced)
  /-
    🎉 no goals
  -/


theorem add_num_den (q r : ℚ) :
    q + r = (q.num * r.den + q.den * r.num : ℤ) /. (↑q.den * ↑r.den : ℤ) := by
  /-
    q r : Rat
    ⊢ Eq (HAdd.hAdd q r) (Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMu …
  -/
  have hqd : (q.den : ℤ) ≠ 0 := Int.natCast_ne_zero_iff_pos.2 q.den_pos
  /-
    q r : Rat
    hqd : Ne (↑q.den) 0
    ⊢ Eq (HAdd.hAdd q r) (Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMu …
  -/
  have hrd : (r.den : ℤ) ≠ 0 := Int.natCast_ne_zero_iff_pos.2 r.den_pos
  /-
    q r : Rat
    hqd : Ne (↑q.den) 0
    hrd : Ne (↑r.den) 0
    ⊢ Eq (HAdd.hAdd q r) (Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMu …
  -/
  conv_lhs => rw [← num_divInt_den q, ← num_divInt_den r, divInt_add_divInt _ _ hqd hrd]
  /-
    q r : Rat
    hqd : Ne (↑q.den) 0
    hrd : Ne (↑r.den) 0
    ⊢ Eq (Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q.den)) …
  -/
  rw [mul_comm r.num q.den]
  /-
    🎉 no goals
  -/



theorem isSquare_iff {q : ℚ} : IsSquare q ↔ IsSquare q.num ∧ IsSquare q.den := by
  /-
    q : Rat
    ⊢ Iff (IsSquare q) (And (IsSquare q.num) (IsSquare q.den))
  -/
  constructor
    /-
      case mp
      q : Rat
      ⊢ IsSquare q → And (IsSquare q.num) (IsSquare q.den)
    -/
  · rintro ⟨qr, rfl⟩
    /-
      case mp.intro
      qr : Rat
      ⊢ And (IsSquare (HMul.hMul qr qr).num) (IsSquare (HMul.hMul qr qr).den)
    -/
    rw [Rat.mul_self_num, mul_self_den]
    /-
      case mp.intro
      qr : Rat
      ⊢ And (IsSquare (HMul.hMul qr.num qr.num)) (IsSquare (HMul.hMul qr.den qr.den))
    -/
    simp only [IsSquare.mul_self, and_self]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      q : Rat
      ⊢ And (IsSquare q.num) (IsSquare q.den) → IsSquare q
    -/
  · rintro ⟨⟨nr, hnr⟩, ⟨dr, hdr⟩⟩
    /-
      case mpr.intro.intro.intro
      q : Rat
      nr : Int
      hnr : Eq q.num (HMul.hMul nr nr)
      dr : Nat
      hdr : Eq q.den (HMul.hMul dr dr)
      ⊢ IsSquare q
    -/
    refine ⟨nr / dr, ?_⟩
    /-
      case mpr.intro.intro.intro
      q : Rat
      nr : Int
      hnr : Eq q.num (HMul.hMul nr nr)
      dr : Nat
      hdr : Eq q.den (HMul.hMul dr dr)
      ⊢ Eq q (HMul.hMul (HDiv.hDiv ↑nr ↑dr) (HDiv.hDiv ↑nr ↑dr))
    -/
    rw [div_mul_div_comm, ← Int.cast_mul, ← Nat.cast_mul, ← hnr, ← hdr, num_div_den]
    /-
      🎉 no goals
    -/


@[norm_cast, simp]
theorem isSquare_natCast_iff {n : ℕ} : IsSquare (n : ℚ) ↔ IsSquare n := by
  /-
    n : Nat
    ⊢ Iff (IsSquare ↑n) (IsSquare n)
  -/
  simp_rw [isSquare_iff, num_natCast, den_natCast, IsSquare.one, and_true, Int.isSquare_natCast_iff]
  /-
    🎉 no goals
  -/


@[norm_cast, simp]
theorem isSquare_intCast_iff {z : ℤ} : IsSquare (z : ℚ) ↔ IsSquare z := by
  /-
    z : Int
    ⊢ Iff (IsSquare ↑z) (IsSquare z)
  -/
  simp_rw [isSquare_iff, intCast_num, intCast_den, IsSquare.one, and_true]
  /-
    🎉 no goals
  -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem isSquare_ofNat_iff {n : ℕ} :
    IsSquare (no_index (OfNat.ofNat n) : ℚ) ↔ IsSquare (OfNat.ofNat n : ℕ) :=
  isSquare_natCast_iff


theorem exists_eq_mul_div_num_and_eq_mul_div_den (n : ℤ) {d : ℤ} (d_ne_zero : d ≠ 0) :
    ∃ c : ℤ, n = c * ((n : ℚ) / d).num ∧ (d : ℤ) = c * ((n : ℚ) / d).den :=
                                             /-
                                               n d : Int
                                               d_ne_zero : Ne d 0
                                               ⊢ Eq (HDiv.hDiv ↑n ↑d) (Rat.divInt n d)
                                             -/
  haveI : (n : ℚ) / d = Rat.divInt n d := by rw [← Rat.divInt_eq_div]
                                             /-
                                               🎉 no goals
                                             -/
  Rat.num_den_mk d_ne_zero this


theorem mul_num_den' (q r : ℚ) :
    (q * r).num * q.den * r.den = q.num * r.num * (q * r).den := by
  /-
    q r : Rat
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul q r).num ↑q.den) ↑r.den) (HMul.hMul (HMu …
  -/
  let s := q.num * r.num /. (q.den * r.den : ℤ)
  /-
    q r : Rat
    s : Rat := Rat.divInt (HMul.hMul q.num r.num) (HMul.hMul ↑q.den ↑r.den)
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul q r).num ↑q.den) ↑r.den) (HMul.hMul (HMu …
  -/
  have hs : (q.den * r.den : ℤ) ≠ 0 := Int.natCast_ne_zero_iff_pos.mpr (Nat.mul_pos q.pos r.pos)
  obtain ⟨c, ⟨c_mul_num, c_mul_den⟩⟩ :=
    exists_eq_mul_div_num_and_eq_mul_div_den (q.num * r.num) hs
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HMul.hMul q.num r.num) (HMul.hMul ↑q.den ↑r.den)
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HMul.hMul q.num r.num) (HMul.hMul c (HDiv.hDiv ↑(HMul.hMul q.n …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HMul.hMul  …
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul q r).num ↑q.den) ↑r.den) (HMul.hMul (HMu …
  -/
  rw [c_mul_num, mul_assoc, mul_comm]
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HMul.hMul q.num r.num) (HMul.hMul ↑q.den ↑r.den)
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HMul.hMul q.num r.num) (HMul.hMul c (HDiv.hDiv ↑(HMul.hMul q.n …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HMul.hMul  …
    ⊢ Eq (HMul.hMul (HMul.hMul ↑q.den ↑r.den) (HMul.hMul q r).num) (HMul.hMul (HMu …
  -/
  nth_rw 1 [c_mul_den]
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HMul.hMul q.num r.num) (HMul.hMul ↑q.den ↑r.den)
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HMul.hMul q.num r.num) (HMul.hMul c (HDiv.hDiv ↑(HMul.hMul q.n …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HMul.hMul  …
    ⊢ Eq (HMul.hMul (HMul.hMul c ↑(HDiv.hDiv ↑(HMul.hMul q.num r.num) ↑(HMul.hMul  …
  -/
  rw [Int.mul_assoc, Int.mul_assoc, mul_eq_mul_left_iff, or_iff_not_imp_right]
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HMul.hMul q.num r.num) (HMul.hMul ↑q.den ↑r.den)
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HMul.hMul q.num r.num) (HMul.hMul c (HDiv.hDiv ↑(HMul.hMul q.n …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HMul.hMul  …
    ⊢ Not (Eq c 0) → Eq (HMul.hMul (↑(HDiv.hDiv ↑(HMul.hMul q.num r.num) ↑(HMul.hM …
  -/
  intro
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HMul.hMul q.num r.num) (HMul.hMul ↑q.den ↑r.den)
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HMul.hMul q.num r.num) (HMul.hMul c (HDiv.hDiv ↑(HMul.hMul q.n …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HMul.hMul  …
    a✝ : Not (Eq c 0)
    ⊢ Eq (HMul.hMul (↑(HDiv.hDiv ↑(HMul.hMul q.num r.num) ↑(HMul.hMul ↑q.den ↑r.de …
  -/
  have h : _ = s := divInt_mul_divInt q.num r.num (mod_cast q.den_ne_zero) (mod_cast r.den_ne_zero)
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HMul.hMul q.num r.num) (HMul.hMul ↑q.den ↑r.den)
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HMul.hMul q.num r.num) (HMul.hMul c (HDiv.hDiv ↑(HMul.hMul q.n …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HMul.hMul  …
    a✝ : Not (Eq c 0)
    h : Eq (HMul.hMul (Rat.divInt q.num ↑q.den) (Rat.divInt r.num ↑r.den)) s
    ⊢ Eq (HMul.hMul (↑(HDiv.hDiv ↑(HMul.hMul q.num r.num) ↑(HMul.hMul ↑q.den ↑r.de …
  -/
  rw [num_divInt_den, num_divInt_den] at h
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HMul.hMul q.num r.num) (HMul.hMul ↑q.den ↑r.den)
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HMul.hMul q.num r.num) (HMul.hMul c (HDiv.hDiv ↑(HMul.hMul q.n …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HMul.hMul  …
    a✝ : Not (Eq c 0)
    h : Eq (HMul.hMul q r) s
    ⊢ Eq (HMul.hMul (↑(HDiv.hDiv ↑(HMul.hMul q.num r.num) ↑(HMul.hMul ↑q.den ↑r.de …
  -/
  rw [h, mul_comm, ← Rat.eq_iff_mul_eq_mul, ← divInt_eq_div]
  /-
    🎉 no goals
  -/


theorem add_num_den' (q r : ℚ) :
    (q + r).num * q.den * r.den = (q.num * r.den + r.num * q.den) * (q + r).den := by
  /-
    q r : Rat
    ⊢ Eq (HMul.hMul (HMul.hMul (HAdd.hAdd q r).num ↑q.den) ↑r.den) (HMul.hMul (HAd …
  -/
  let s := divInt (q.num * r.den + r.num * q.den) (q.den * r.den : ℤ)
  /-
    q r : Rat
    s : Rat := Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q. …
    ⊢ Eq (HMul.hMul (HMul.hMul (HAdd.hAdd q r).num ↑q.den) ↑r.den) (HMul.hMul (HAd …
  -/
  have hs : (q.den * r.den : ℤ) ≠ 0 := Int.natCast_ne_zero_iff_pos.mpr (Nat.mul_pos q.pos r.pos)
  obtain ⟨c, ⟨c_mul_num, c_mul_den⟩⟩ :=
    exists_eq_mul_div_num_and_eq_mul_div_den (q.num * r.den + r.num * q.den) hs
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q. …
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q.den)) ( …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HAdd.hAdd  …
    ⊢ Eq (HMul.hMul (HMul.hMul (HAdd.hAdd q r).num ↑q.den) ↑r.den) (HMul.hMul (HAd …
  -/
  rw [c_mul_num, mul_assoc, mul_comm]
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q. …
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q.den)) ( …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HAdd.hAdd  …
    ⊢ Eq (HMul.hMul (HMul.hMul ↑q.den ↑r.den) (HAdd.hAdd q r).num) (HMul.hMul (HMu …
  -/
  nth_rw 1 [c_mul_den]
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q. …
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q.den)) ( …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HAdd.hAdd  …
    ⊢ Eq (HMul.hMul (HMul.hMul c ↑(HDiv.hDiv ↑(HAdd.hAdd (HMul.hMul q.num ↑r.den)  …
  -/
  repeat rw [Int.mul_assoc]
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q. …
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q.den)) ( …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HAdd.hAdd  …
    ⊢ Eq (HMul.hMul c (HMul.hMul (↑(HDiv.hDiv ↑(HAdd.hAdd (HMul.hMul q.num ↑r.den) …
  -/
  apply mul_eq_mul_left_iff.2
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q. …
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q.den)) ( …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HAdd.hAdd  …
    ⊢ Or (Eq (HMul.hMul (↑(HDiv.hDiv ↑(HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hM …
  -/
  rw [or_iff_not_imp_right]
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q. …
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q.den)) ( …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HAdd.hAdd  …
    ⊢ Not (Eq c 0) → Eq (HMul.hMul (↑(HDiv.hDiv ↑(HAdd.hAdd (HMul.hMul q.num ↑r.de …
  -/
  intro
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q. …
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q.den)) ( …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HAdd.hAdd  …
    a✝ : Not (Eq c 0)
    ⊢ Eq (HMul.hMul (↑(HDiv.hDiv ↑(HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r …
  -/
  have h : _ = s := divInt_add_divInt q.num r.num (mod_cast q.den_ne_zero) (mod_cast r.den_ne_zero)
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q. …
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q.den)) ( …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HAdd.hAdd  …
    a✝ : Not (Eq c 0)
    h : Eq (HAdd.hAdd (Rat.divInt q.num ↑q.den) (Rat.divInt r.num ↑r.den)) s
    ⊢ Eq (HMul.hMul (↑(HDiv.hDiv ↑(HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r …
  -/
  rw [num_divInt_den, num_divInt_den] at h
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q. …
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q.den)) ( …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HAdd.hAdd  …
    a✝ : Not (Eq c 0)
    h : Eq (HAdd.hAdd q r) s
    ⊢ Eq (HMul.hMul (↑(HDiv.hDiv ↑(HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r …
  -/
  rw [h]
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q. …
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q.den)) ( …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HAdd.hAdd  …
    a✝ : Not (Eq c 0)
    h : Eq (HAdd.hAdd q r) s
    ⊢ Eq (HMul.hMul (↑(HDiv.hDiv ↑(HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r …
  -/
  rw [mul_comm]
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q. …
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q.den)) ( …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HAdd.hAdd  …
    a✝ : Not (Eq c 0)
    h : Eq (HAdd.hAdd q r) s
    ⊢ Eq (HMul.hMul s.num ↑(HDiv.hDiv ↑(HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.h …
  -/
  apply Rat.eq_iff_mul_eq_mul.mp
  /-
    case intro.intro
    q r : Rat
    s : Rat := Rat.divInt (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q. …
    hs : Ne (HMul.hMul ↑q.den ↑r.den) 0
    c : Int
    c_mul_num : Eq (HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q.den)) ( …
    c_mul_den : Eq (HMul.hMul ↑q.den ↑r.den) (HMul.hMul c ↑(HDiv.hDiv ↑(HAdd.hAdd  …
    a✝ : Not (Eq c 0)
    h : Eq (HAdd.hAdd q r) s
    ⊢ Eq s (HDiv.hDiv ↑(HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul r.num ↑q.den …
  -/
  rw [← divInt_eq_div]
  /-
    🎉 no goals
  -/


theorem substr_num_den' (q r : ℚ) :
    (q - r).num * q.den * r.den = (q.num * r.den - r.num * q.den) * (q - r).den := by
  rw [sub_eq_add_neg, sub_eq_add_neg, ← neg_mul, ← num_neg_eq_neg_num, ← den_neg_eq_den r,
    add_num_den' q (-r)]


protected theorem inv_neg (q : ℚ) : (-q)⁻¹ = -q⁻¹ := by
  /-
    q : Rat
    ⊢ Eq (Inv.inv (Neg.neg q)) (Neg.neg (Inv.inv q))
  -/
  rw [← num_divInt_den q]
  /-
    q : Rat
    ⊢ Eq (Inv.inv (Neg.neg (Rat.divInt q.num ↑q.den))) (Neg.neg (Inv.inv (Rat.divI …
  -/
  simp only [Rat.neg_divInt, Rat.inv_divInt', eq_self_iff_true, Rat.divInt_neg]
  /-
    🎉 no goals
  -/


theorem num_div_eq_of_coprime {a b : ℤ} (hb0 : 0 < b) (h : Nat.Coprime a.natAbs b.natAbs) :
    (a / b : ℚ).num = a := by
  -- Porting note: was `lift b to ℕ using le_of_lt hb0`
  rw [← Int.natAbs_of_nonneg hb0.le, ← Rat.divInt_eq_div,
    ← mk_eq_divInt _ _ (Int.natAbs_ne_zero.mpr hb0.ne') h]


theorem den_div_eq_of_coprime {a b : ℤ} (hb0 : 0 < b) (h : Nat.Coprime a.natAbs b.natAbs) :
    ((a / b : ℚ).den : ℤ) = b := by
  -- Porting note: was `lift b to ℕ using le_of_lt hb0`
  rw [← Int.natAbs_of_nonneg hb0.le, ← Rat.divInt_eq_div,
    ← mk_eq_divInt _ _ (Int.natAbs_ne_zero.mpr hb0.ne') h]


theorem div_int_inj {a b c d : ℤ} (hb0 : 0 < b) (hd0 : 0 < d) (h1 : Nat.Coprime a.natAbs b.natAbs)
    (h2 : Nat.Coprime c.natAbs d.natAbs) (h : (a : ℚ) / b = (c : ℚ) / d) : a = c ∧ b = d := by
  /-
    a b c d : Int
    hb0 : LT.lt 0 b
    hd0 : LT.lt 0 d
    h1 : a.natAbs.Coprime b.natAbs
    h2 : c.natAbs.Coprime d.natAbs
    h : Eq (HDiv.hDiv ↑a ↑b) (HDiv.hDiv ↑c ↑d)
    ⊢ And (Eq a c) (Eq b d)
  -/
  apply And.intro
    /-
      case left
      a b c d : Int
      hb0 : LT.lt 0 b
      hd0 : LT.lt 0 d
      h1 : a.natAbs.Coprime b.natAbs
      h2 : c.natAbs.Coprime d.natAbs
      h : Eq (HDiv.hDiv ↑a ↑b) (HDiv.hDiv ↑c ↑d)
      ⊢ Eq a c
    -/
  · rw [← num_div_eq_of_coprime hb0 h1, h, num_div_eq_of_coprime hd0 h2]
    /-
      🎉 no goals
    -/
    /-
      case right
      a b c d : Int
      hb0 : LT.lt 0 b
      hd0 : LT.lt 0 d
      h1 : a.natAbs.Coprime b.natAbs
      h2 : c.natAbs.Coprime d.natAbs
      h : Eq (HDiv.hDiv ↑a ↑b) (HDiv.hDiv ↑c ↑d)
      ⊢ Eq b d
    -/
  · rw [← den_div_eq_of_coprime hb0 h1, h, den_div_eq_of_coprime hd0 h2]
    /-
      🎉 no goals
    -/


@[norm_cast]
theorem intCast_div_self (n : ℤ) : ((n / n : ℤ) : ℚ) = n / n := by
  /-
    n : Int
    ⊢ Eq (↑(HDiv.hDiv n n)) (HDiv.hDiv ↑n ↑n)
  -/
  by_cases hn : n = 0
    /-
      case pos
      n : Int
      hn : Eq n 0
      ⊢ Eq (↑(HDiv.hDiv n n)) (HDiv.hDiv ↑n ↑n)
    -/
  · subst hn
    /-
      case pos
      ⊢ Eq (↑(0 / 0)) (HDiv.hDiv ↑0 ↑0)
    -/
    simp only [Int.cast_zero, Int.zero_tdiv, zero_div, Int.ediv_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Int
      hn : Not (Eq n 0)
      ⊢ Eq (↑(HDiv.hDiv n n)) (HDiv.hDiv ↑n ↑n)
    -/
  · have : (n : ℚ) ≠ 0 := by rwa [← coe_int_inj] at hn
    /-
      case neg
      n : Int
      hn : Not (Eq n 0)
      this : Ne (↑n) 0
      ⊢ Eq (↑(HDiv.hDiv n n)) (HDiv.hDiv ↑n ↑n)
    -/
    simp only [Int.ediv_self hn, Int.cast_one, Ne, not_false_iff, div_self this]
    /-
      🎉 no goals
    -/


@[norm_cast]
theorem natCast_div_self (n : ℕ) : ((n / n : ℕ) : ℚ) = n / n :=
  intCast_div_self n


theorem intCast_div (a b : ℤ) (h : b ∣ a) : ((a / b : ℤ) : ℚ) = a / b := by
  /-
    a b : Int
    h : Dvd.dvd b a
    ⊢ Eq (↑(HDiv.hDiv a b)) (HDiv.hDiv ↑a ↑b)
  -/
  rcases h with ⟨c, rfl⟩
  rw [mul_comm b, Int.mul_ediv_assoc c (dvd_refl b), Int.cast_mul,
    intCast_div_self, Int.cast_mul, mul_div_assoc]


theorem natCast_div (a b : ℕ) (h : b ∣ a) : ((a / b : ℕ) : ℚ) = a / b :=
  intCast_div a b (Int.ofNat_dvd.mpr h)


theorem den_div_intCast_eq_one_iff (m n : ℤ) (hn : n ≠ 0) : ((m : ℚ) / n).den = 1 ↔ n ∣ m := by
  /-
    m n : Int
    hn : Ne n 0
    ⊢ Iff (Eq (HDiv.hDiv ↑m ↑n).den 1) (Dvd.dvd n m)
  -/
  replace hn : (n : ℚ) ≠ 0 := num_ne_zero.mp hn
  /-
    m n : Int
    hn : Ne (↑n) 0
    ⊢ Iff (Eq (HDiv.hDiv ↑m ↑n).den 1) (Dvd.dvd n m)
  -/
  constructor
    /-
      case mp
      m n : Int
      hn : Ne (↑n) 0
      ⊢ Eq (HDiv.hDiv ↑m ↑n).den 1 → Dvd.dvd n m
    -/
  · rw [Rat.den_eq_one_iff, eq_div_iff hn]
    /-
      case mp
      m n : Int
      hn : Ne (↑n) 0
      ⊢ Eq (HMul.hMul ↑(HDiv.hDiv ↑m ↑n).num ↑n) ↑m → Dvd.dvd n m
    -/
    exact mod_cast (Dvd.intro_left _)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      m n : Int
      hn : Ne (↑n) 0
      ⊢ Dvd.dvd n m → Eq (HDiv.hDiv ↑m ↑n).den 1
    -/
  · exact (intCast_div _ _ · ▸ rfl)
    /-
      🎉 no goals
    -/


theorem den_div_natCast_eq_one_iff (m n : ℕ) (hn : n ≠ 0) : ((m : ℚ) / n).den = 1 ↔ n ∣ m :=
  (den_div_intCast_eq_one_iff m n (Int.ofNat_ne_zero.mpr hn)).trans Int.ofNat_dvd


@[deprecated (since := "2024-05-11")] alias den_div_cast_eq_one_iff := den_div_intCast_eq_one_iff


theorem inv_intCast_num_of_pos {a : ℤ} (ha0 : 0 < a) : (a : ℚ)⁻¹.num = 1 := by
  /-
    a : Int
    ha0 : LT.lt 0 a
    ⊢ Eq (Inv.inv ↑a).num 1
  -/
  rw [← ofInt_eq_cast, ofInt, mk_eq_divInt, Rat.inv_divInt', divInt_eq_div, Nat.cast_one]
  /-
    a : Int
    ha0 : LT.lt 0 a
    ⊢ Eq (HDiv.hDiv ↑1 ↑a).num 1
  -/
  apply num_div_eq_of_coprime ha0
  /-
    a : Int
    ha0 : LT.lt 0 a
    ⊢ (Int.natAbs 1).Coprime a.natAbs
  -/
  rw [Int.natAbs_one]
  /-
    a : Int
    ha0 : LT.lt 0 a
    ⊢ Nat.Coprime 1 a.natAbs
  -/
  exact Nat.coprime_one_left _
  /-
    🎉 no goals
  -/


theorem inv_natCast_num_of_pos {a : ℕ} (ha0 : 0 < a) : (a : ℚ)⁻¹.num = 1 :=
  inv_intCast_num_of_pos (mod_cast ha0 : 0 < (a : ℤ))


theorem inv_intCast_den_of_pos {a : ℤ} (ha0 : 0 < a) : ((a : ℚ)⁻¹.den : ℤ) = a := by
  /-
    a : Int
    ha0 : LT.lt 0 a
    ⊢ Eq (↑(Inv.inv ↑a).den) a
  -/
  rw [← ofInt_eq_cast, ofInt, mk_eq_divInt, Rat.inv_divInt', divInt_eq_div, Nat.cast_one]
  /-
    a : Int
    ha0 : LT.lt 0 a
    ⊢ Eq (↑(HDiv.hDiv ↑1 ↑a).den) a
  -/
  apply den_div_eq_of_coprime ha0
  /-
    a : Int
    ha0 : LT.lt 0 a
    ⊢ (Int.natAbs 1).Coprime a.natAbs
  -/
  rw [Int.natAbs_one]
  /-
    a : Int
    ha0 : LT.lt 0 a
    ⊢ Nat.Coprime 1 a.natAbs
  -/
  exact Nat.coprime_one_left _
  /-
    🎉 no goals
  -/


theorem inv_natCast_den_of_pos {a : ℕ} (ha0 : 0 < a) : (a : ℚ)⁻¹.den = a := by
  /-
    a : Nat
    ha0 : LT.lt 0 a
    ⊢ Eq (Inv.inv ↑a).den a
  -/
  rw [← Int.ofNat_inj, ← Int.cast_natCast a, inv_intCast_den_of_pos]
  /-
    a : Nat
    ha0 : LT.lt 0 a
    ⊢ LT.lt 0 ↑a
  -/
  rwa [Int.natCast_pos]
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_intCast_num (a : ℤ) : (a : ℚ)⁻¹.num = Int.sign a := by
  /-
    a : Int
    ⊢ Eq (Inv.inv ↑a).num a.sign
  -/
  rcases lt_trichotomy a 0 with lt | rfl | gt
    /-
      case inl
      a : Int
      lt : LT.lt a 0
      ⊢ Eq (Inv.inv ↑a).num a.sign
    -/
  · obtain ⟨a, rfl⟩ : ∃ b, -b = a := ⟨-a, a.neg_neg⟩
    /-
      case inl.intro
      a : Int
      lt : LT.lt (Neg.neg a) 0
      ⊢ Eq (Inv.inv ↑(Neg.neg a)).num (Neg.neg a).sign
    -/
    simp at lt
    /-
      case inl.intro
      a : Int
      lt : LT.lt 0 a
      ⊢ Eq (Inv.inv ↑(Neg.neg a)).num (Neg.neg a).sign
    -/
    simp [Rat.inv_neg, inv_intCast_num_of_pos lt, Int.sign_eq_one_iff_pos.mpr lt]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      ⊢ Eq (Inv.inv ↑0).num (Int.sign 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      a : Int
      gt : LT.lt 0 a
      ⊢ Eq (Inv.inv ↑a).num a.sign
    -/
  · simp [inv_intCast_num_of_pos gt, Int.sign_eq_one_iff_pos.mpr gt]
    /-
      🎉 no goals
    -/


@[simp]
theorem inv_natCast_num (a : ℕ) : (a : ℚ)⁻¹.num = Int.sign a :=
  inv_intCast_num a


@[simp]
theorem inv_ofNat_num (a : ℕ) [a.AtLeastTwo] : (no_index (OfNat.ofNat a : ℚ))⁻¹.num = 1 :=
  inv_natCast_num_of_pos (Nat.pos_of_neZero a)


@[simp]
theorem inv_intCast_den (a : ℤ) : (a : ℚ)⁻¹.den = if a = 0 then 1 else a.natAbs := by
  /-
    a : Int
    ⊢ Eq (Inv.inv ↑a).den (ite (Eq a 0) 1 a.natAbs)
  -/
  rw [← Int.ofNat_inj]
  /-
    a : Int
    ⊢ Eq ↑(Inv.inv ↑a).den ↑(ite (Eq a 0) 1 a.natAbs)
  -/
  rcases lt_trichotomy a 0 with lt | rfl | gt
    /-
      case inl
      a : Int
      lt : LT.lt a 0
      ⊢ Eq ↑(Inv.inv ↑a).den ↑(ite (Eq a 0) 1 a.natAbs)
    -/
  · obtain ⟨a, rfl⟩ : ∃ b, -b = a := ⟨-a, a.neg_neg⟩
    /-
      case inl.intro
      a : Int
      lt : LT.lt (Neg.neg a) 0
      ⊢ Eq ↑(Inv.inv ↑(Neg.neg a)).den ↑(ite (Eq (Neg.neg a) 0) 1 (Neg.neg a).natAbs)
    -/
    simp at lt
    /-
      case inl.intro
      a : Int
      lt : LT.lt 0 a
      ⊢ Eq ↑(Inv.inv ↑(Neg.neg a)).den ↑(ite (Eq (Neg.neg a) 0) 1 (Neg.neg a).natAbs)
    -/
    rw [if_neg (by omega)]
    /-
      case inl.intro
      a : Int
      lt : LT.lt 0 a
      ⊢ Eq ↑(Inv.inv ↑(Neg.neg a)).den ↑(Neg.neg a).natAbs
    -/
    simp only [Int.cast_neg, Rat.inv_neg, neg_den, inv_intCast_den_of_pos lt, Int.natAbs_neg]
    /-
      case inl.intro
      a : Int
      lt : LT.lt 0 a
      ⊢ Eq a ↑a.natAbs
    -/
    exact Int.eq_natAbs_of_zero_le (by omega)
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      ⊢ Eq ↑(Inv.inv ↑0).den ↑(ite (Eq 0 0) 1 (Int.natAbs 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      a : Int
      gt : LT.lt 0 a
      ⊢ Eq ↑(Inv.inv ↑a).den ↑(ite (Eq a 0) 1 a.natAbs)
    -/
  · rw [if_neg (by omega)]
    /-
      case inr.inr
      a : Int
      gt : LT.lt 0 a
      ⊢ Eq ↑(Inv.inv ↑a).den ↑a.natAbs
    -/
    simp only [inv_intCast_den_of_pos gt]
    /-
      case inr.inr
      a : Int
      gt : LT.lt 0 a
      ⊢ Eq a ↑a.natAbs
    -/
    exact Int.eq_natAbs_of_zero_le (by omega)
    /-
      🎉 no goals
    -/


@[simp]
theorem inv_natCast_den (a : ℕ) : (a : ℚ)⁻¹.den = if a = 0 then 1 else a := by
  /-
    a : Nat
    ⊢ Eq (Inv.inv ↑a).den (ite (Eq a 0) 1 a)
  -/
  simpa [-inv_intCast_den, ofInt_eq_cast] using inv_intCast_den a
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-05")] alias coe_int_div_self := intCast_div_self

@[deprecated (since := "2024-04-05")] alias coe_nat_div_self := natCast_div_self

@[deprecated (since := "2024-04-05")] alias coe_int_div := intCast_div

@[deprecated (since := "2024-04-05")] alias coe_nat_div := natCast_div

@[deprecated (since := "2024-04-05")] alias inv_coe_int_num_of_pos := inv_intCast_num_of_pos

@[deprecated (since := "2024-04-05")] alias inv_coe_nat_num_of_pos := inv_natCast_num_of_pos

@[deprecated (since := "2024-04-05")] alias inv_coe_int_den_of_pos := inv_intCast_den_of_pos

@[deprecated (since := "2024-04-05")] alias inv_coe_nat_den_of_pos := inv_natCast_den_of_pos

@[deprecated (since := "2024-04-05")] alias inv_coe_int_num := inv_intCast_num

@[deprecated (since := "2024-04-05")] alias inv_coe_nat_num := inv_natCast_num

@[deprecated (since := "2024-04-05")] alias inv_coe_int_den := inv_intCast_den

@[deprecated (since := "2024-04-05")] alias inv_coe_nat_den := inv_natCast_den


@[simp]
theorem inv_ofNat_den (a : ℕ) [a.AtLeastTwo] :
    (no_index (OfNat.ofNat a : ℚ))⁻¹.den = OfNat.ofNat a :=
  inv_natCast_den_of_pos (Nat.pos_of_neZero a)


protected theorem «forall» {p : ℚ → Prop} : (∀ r, p r) ↔ ∀ a b : ℤ, p (a / b) :=
  ⟨fun h _ _ => h _,
   fun h q => by
    /-
      p : Rat → Prop
      h : ∀ (a b : Int), p (HDiv.hDiv ↑a ↑b)
      q : Rat
      ⊢ p q
    -/
    have := h q.num q.den
    /-
      p : Rat → Prop
      h : ∀ (a b : Int), p (HDiv.hDiv ↑a ↑b)
      q : Rat
      this : p (HDiv.hDiv ↑q.num ↑↑q.den)
      ⊢ p q
    -/
    rwa [Int.cast_natCast, num_div_den q] at this⟩
    /-
      🎉 no goals
    -/


protected theorem «exists» {p : ℚ → Prop} : (∃ r, p r) ↔ ∃ a b : ℤ, p (a / b) :=
                                    /-
                                      p : Rat → Prop
                                      x✝ : Exists fun r => p r
                                      r : Rat
                                      hr : p r
                                      ⊢ p (HDiv.hDiv ↑r.num ↑↑r.den)
                                    -/
  ⟨fun ⟨r, hr⟩ => ⟨r.num, r.den, by convert hr; convert num_div_den r⟩, fun ⟨_, _, h⟩ => ⟨_, h⟩⟩
                                                /-
                                                  🎉 no goals
                                                -/


/-- Denominator as `ℕ+`. -/
def pnatDen (x : ℚ) : ℕ+ :=
  ⟨x.den, x.pos⟩


@[simp]
theorem coe_pnatDen (x : ℚ) : (x.pnatDen : ℕ) = x.den :=
  rfl


theorem pnatDen_eq_iff_den_eq {x : ℚ} {n : ℕ+} : x.pnatDen = n ↔ x.den = ↑n :=
  Subtype.ext_iff


@[simp]
theorem pnatDen_one : (1 : ℚ).pnatDen = 1 :=
  rfl


@[simp]
theorem pnatDen_zero : (0 : ℚ).pnatDen = 1 :=
  rfl


