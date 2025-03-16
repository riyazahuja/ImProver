theorem isPrimitiveRoot_exp_of_coprime (i n : ℕ) (h0 : n ≠ 0) (hi : i.Coprime n) :
    IsPrimitiveRoot (exp (2 * π * I * (i / n))) n := by
  /-
    i n : Nat
    h0 : Ne n 0
    hi : i.Coprime n
    ⊢ IsPrimitiveRoot (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Co …
  -/
  rw [IsPrimitiveRoot.iff_def]
  /-
    i n : Nat
    h0 : Ne n 0
    hi : i.Coprime n
    ⊢ And (Eq (HPow.hPow (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) …
  -/
  simp only [← exp_nat_mul, exp_eq_one_iff]
  /-
    i n : Nat
    h0 : Ne n 0
    hi : i.Coprime n
    ⊢ And (Exists fun n_1 => Eq (HMul.hMul (↑n) (HMul.hMul (HMul.hMul (HMul.hMul 2 …
  -/
  have hn0 : (n : ℂ) ≠ 0 := mod_cast h0
  /-
    i n : Nat
    h0 : Ne n 0
    hi : i.Coprime n
    hn0 : Ne (↑n) 0
    ⊢ And (Exists fun n_1 => Eq (HMul.hMul (↑n) (HMul.hMul (HMul.hMul (HMul.hMul 2 …
  -/
  constructor
    /-
      case left
      i n : Nat
      h0 : Ne n 0
      hi : i.Coprime n
      hn0 : Ne (↑n) 0
      ⊢ Exists fun n_1 => Eq (HMul.hMul (↑n) (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Rea …
    -/
  · use i
    /-
      case h
      i n : Nat
      h0 : Ne n 0
      hi : i.Coprime n
      hn0 : Ne (↑n) 0
      ⊢ Eq (HMul.hMul (↑n) (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) ( …
    -/
    field_simp [hn0, mul_comm (i : ℂ), mul_comm (n : ℂ)]
    /-
      🎉 no goals
    -/
  · simp only [hn0, mul_right_comm _ _ ↑n, mul_left_inj' two_pi_I_ne_zero, Ne, not_false_iff,
      mul_comm _ (i : ℂ), ← mul_assoc _ (i : ℂ), exists_imp, field_simps]
    /-
      case right
      i n : Nat
      h0 : Ne n 0
      hi : i.Coprime n
      hn0 : Ne (↑n) 0
      ⊢ ∀ (l : Nat) (x : Int), Eq (HMul.hMul (HMul.hMul ↑i ↑l) (HMul.hMul (HMul.hMul …
    -/
    norm_cast
    /-
      case right
      i n : Nat
      h0 : Ne n 0
      hi : i.Coprime n
      hn0 : Ne (↑n) 0
      ⊢ ∀ (l : Nat) (x : Int), Eq (HMul.hMul (↑(HMul.hMul i l)) (HMul.hMul (↑(HMul.h …
    -/
    rintro l k hk
    /-
      case right
      i n : Nat
      h0 : Ne n 0
      hi : i.Coprime n
      hn0 : Ne (↑n) 0
      l : Nat
      k : Int
      hk : Eq (HMul.hMul (↑(HMul.hMul i l)) (HMul.hMul (↑(HMul.hMul 2 Real.pi)) Comp …
      ⊢ Dvd.dvd n l
    -/
    conv_rhs at hk => rw [mul_comm, ← mul_assoc]
    /-
      case right
      i n : Nat
      h0 : Ne n 0
      hi : i.Coprime n
      hn0 : Ne (↑n) 0
      l : Nat
      k : Int
      hk : Eq (HMul.hMul (↑(HMul.hMul i l)) (HMul.hMul (↑(HMul.hMul 2 Real.pi)) Comp …
      ⊢ Dvd.dvd n l
    -/
    have hz : 2 * ↑π * I ≠ 0 := by simp [pi_pos.ne.symm, I_ne_zero]
    /-
      case right
      i n : Nat
      h0 : Ne n 0
      hi : i.Coprime n
      hn0 : Ne (↑n) 0
      l : Nat
      k : Int
      hk : Eq (HMul.hMul (↑(HMul.hMul i l)) (HMul.hMul (↑(HMul.hMul 2 Real.pi)) Comp …
      hz : Ne (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) 0
      ⊢ Dvd.dvd n l
    -/
    field_simp [hz] at hk
    /-
      case right
      i n : Nat
      h0 : Ne n 0
      hi : i.Coprime n
      hn0 : Ne (↑n) 0
      l : Nat
      k : Int
      hz : Ne (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) 0
      hk : Eq (HMul.hMul ↑i ↑l) (HMul.hMul ↑n ↑k)
      ⊢ Dvd.dvd n l
    -/
    norm_cast at hk
    /-
      case right
      i n : Nat
      h0 : Ne n 0
      hi : i.Coprime n
      hn0 : Ne (↑n) 0
      l : Nat
      k : Int
      hz : Ne (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) 0
      hk : Eq (↑(HMul.hMul i l)) (HMul.hMul (↑n) k)
      ⊢ Dvd.dvd n l
    -/
    have : n ∣ i * l := by rw [← Int.natCast_dvd_natCast, hk, mul_comm]; apply dvd_mul_left
    /-
      case right
      i n : Nat
      h0 : Ne n 0
      hi : i.Coprime n
      hn0 : Ne (↑n) 0
      l : Nat
      k : Int
      hz : Ne (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) 0
      hk : Eq (↑(HMul.hMul i l)) (HMul.hMul (↑n) k)
      this : Dvd.dvd n (HMul.hMul i l)
      ⊢ Dvd.dvd n l
    -/
    exact hi.symm.dvd_of_dvd_mul_left this
    /-
      🎉 no goals
    -/


theorem isPrimitiveRoot_exp (n : ℕ) (h0 : n ≠ 0) : IsPrimitiveRoot (exp (2 * π * I / n)) n := by
  simpa only [Nat.cast_one, one_div] using
    isPrimitiveRoot_exp_of_coprime 1 n h0 n.coprime_one_left


theorem isPrimitiveRoot_iff (ζ : ℂ) (n : ℕ) (hn : n ≠ 0) :
    IsPrimitiveRoot ζ n ↔ ∃ i < n, ∃ _ : i.Coprime n, exp (2 * π * I * (i / n)) = ζ := by
  /-
    ζ : Complex
    n : Nat
    hn : Ne n 0
    ⊢ Iff (IsPrimitiveRoot ζ n) (Exists fun i => And (LT.lt i n) (Exists fun x =>  …
  -/
  have hn0 : (n : ℂ) ≠ 0 := mod_cast hn
  /-
    ζ : Complex
    n : Nat
    hn : Ne n 0
    hn0 : Ne (↑n) 0
    ⊢ Iff (IsPrimitiveRoot ζ n) (Exists fun i => And (LT.lt i n) (Exists fun x =>  …
  -/
  constructor; swap
    /-
      case mpr
      ζ : Complex
      n : Nat
      hn : Ne n 0
      hn0 : Ne (↑n) 0
      ⊢ (Exists fun i => And (LT.lt i n) (Exists fun x => Eq (Complex.exp (HMul.hMul …
    -/
  · rintro ⟨i, -, hi, rfl⟩; exact isPrimitiveRoot_exp_of_coprime i n hn hi
                            /-
                              🎉 no goals
                            -/
  /-
    case mp
    ζ : Complex
    n : Nat
    hn : Ne n 0
    hn0 : Ne (↑n) 0
    ⊢ IsPrimitiveRoot ζ n → Exists fun i => And (LT.lt i n) (Exists fun x => Eq (C …
  -/
  intro h
  /-
    case mp
    ζ : Complex
    n : Nat
    hn : Ne n 0
    hn0 : Ne (↑n) 0
    h : IsPrimitiveRoot ζ n
    ⊢ Exists fun i => And (LT.lt i n) (Exists fun x => Eq (Complex.exp (HMul.hMul  …
  -/
  have : NeZero n := ⟨hn⟩
  obtain ⟨i, hi, rfl⟩ :=
    (isPrimitiveRoot_exp n hn).eq_pow_of_pow_eq_one h.pow_eq_one
  /-
    case mp.intro.intro
    n : Nat
    hn : Ne n 0
    hn0 : Ne (↑n) 0
    this : NeZero n
    i : Nat
    hi : LT.lt i n
    h : IsPrimitiveRoot (HPow.hPow (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 …
    ⊢ Exists fun i_1 => And (LT.lt i_1 n) (Exists fun x => Eq (Complex.exp (HMul.h …
  -/
  refine ⟨i, hi, ((isPrimitiveRoot_exp n hn).pow_iff_coprime (Nat.pos_of_ne_zero hn) i).mp h, ?_⟩
  /-
    case mp.intro.intro
    n : Nat
    hn : Ne n 0
    hn0 : Ne (↑n) 0
    this : NeZero n
    i : Nat
    hi : LT.lt i n
    h : IsPrimitiveRoot (HPow.hPow (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 …
    ⊢ Eq (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) (HDi …
  -/
  rw [← exp_nat_mul]
  /-
    case mp.intro.intro
    n : Nat
    hn : Ne n 0
    hn0 : Ne (↑n) 0
    this : NeZero n
    i : Nat
    hi : LT.lt i n
    h : IsPrimitiveRoot (HPow.hPow (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 …
    ⊢ Eq (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) (HDi …
  -/
  congr 1
  /-
    case mp.intro.intro.e_z
    n : Nat
    hn : Ne n 0
    hn0 : Ne (↑n) 0
    this : NeZero n
    i : Nat
    hi : LT.lt i n
    h : IsPrimitiveRoot (HPow.hPow (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 …
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) (HDiv.hDiv ↑i ↑n) …
  -/
  field_simp [hn0, mul_comm (i : ℂ)]
  /-
    🎉 no goals
  -/


/-- The complex `n`-th roots of unity are exactly the
complex numbers of the form `exp (2 * Real.pi * Complex.I * (i / n))` for some `i < n`. -/
nonrec theorem mem_rootsOfUnity (n : ℕ) [NeZero n] (x : Units ℂ) :
    x ∈ rootsOfUnity n ℂ ↔ ∃ i < n, exp (2 * π * I * (i / n)) = x := by
  /-
    n : Nat
    inst✝ : NeZero n
    x : Units Complex
    ⊢ Iff (Membership.mem (rootsOfUnity n Complex) x) (Exists fun i => And (LT.lt  …
  -/
  rw [mem_rootsOfUnity, Units.ext_iff, Units.val_pow_eq_pow_val, Units.val_one]
  /-
    n : Nat
    inst✝ : NeZero n
    x : Units Complex
    ⊢ Iff (Eq (HPow.hPow (↑x) n) 1) (Exists fun i => And (LT.lt i n) (Eq (Complex. …
  -/
  have hn0 : (n : ℂ) ≠ 0 := mod_cast NeZero.out
  /-
    n : Nat
    inst✝ : NeZero n
    x : Units Complex
    hn0 : Ne (↑n) 0
    ⊢ Iff (Eq (HPow.hPow (↑x) n) 1) (Exists fun i => And (LT.lt i n) (Eq (Complex. …
  -/
  constructor
    /-
      case mp
      n : Nat
      inst✝ : NeZero n
      x : Units Complex
      hn0 : Ne (↑n) 0
      ⊢ Eq (HPow.hPow (↑x) n) 1 → Exists fun i => And (LT.lt i n) (Eq (Complex.exp ( …
    -/
  · intro h
    obtain ⟨i, hi, H⟩ : ∃ i < (n : ℕ), exp (2 * π * I / n) ^ i = x := by
      simpa only using (isPrimitiveRoot_exp n NeZero.out).eq_pow_of_pow_eq_one h
    /-
      case mp.intro.intro
      n : Nat
      inst✝ : NeZero n
      x : Units Complex
      hn0 : Ne (↑n) 0
      h : Eq (HPow.hPow (↑x) n) 1
      i : Nat
      hi : LT.lt i n
      H : Eq (HPow.hPow (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Co …
      ⊢ Exists fun i => And (LT.lt i n) (Eq (Complex.exp (HMul.hMul (HMul.hMul (HMul …
    -/
    refine ⟨i, hi, ?_⟩
    /-
      case mp.intro.intro
      n : Nat
      inst✝ : NeZero n
      x : Units Complex
      hn0 : Ne (↑n) 0
      h : Eq (HPow.hPow (↑x) n) 1
      i : Nat
      hi : LT.lt i n
      H : Eq (HPow.hPow (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Co …
      ⊢ Eq (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) (HDi …
    -/
    rw [← H, ← exp_nat_mul]
    /-
      case mp.intro.intro
      n : Nat
      inst✝ : NeZero n
      x : Units Complex
      hn0 : Ne (↑n) 0
      h : Eq (HPow.hPow (↑x) n) 1
      i : Nat
      hi : LT.lt i n
      H : Eq (HPow.hPow (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Co …
      ⊢ Eq (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) (HDi …
    -/
    congr 1
    /-
      case mp.intro.intro.e_z
      n : Nat
      inst✝ : NeZero n
      x : Units Complex
      hn0 : Ne (↑n) 0
      h : Eq (HPow.hPow (↑x) n) 1
      i : Nat
      hi : LT.lt i n
      H : Eq (HPow.hPow (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Co …
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) (HDiv.hDiv ↑i ↑n) …
    -/
    field_simp [hn0, mul_comm (i : ℂ)]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      inst✝ : NeZero n
      x : Units Complex
      hn0 : Ne (↑n) 0
      ⊢ (Exists fun i => And (LT.lt i n) (Eq (Complex.exp (HMul.hMul (HMul.hMul (HMu …
    -/
  · rintro ⟨i, _, H⟩
    /-
      case mpr.intro.intro
      n : Nat
      inst✝ : NeZero n
      x : Units Complex
      hn0 : Ne (↑n) 0
      i : Nat
      left✝ : LT.lt i n
      H : Eq (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) (H …
      ⊢ Eq (HPow.hPow (↑x) n) 1
    -/
    rw [← H, ← exp_nat_mul, exp_eq_one_iff]
    /-
      case mpr.intro.intro
      n : Nat
      inst✝ : NeZero n
      x : Units Complex
      hn0 : Ne (↑n) 0
      i : Nat
      left✝ : LT.lt i n
      H : Eq (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) (H …
      ⊢ Exists fun n_1 => Eq (HMul.hMul (↑n) (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Rea …
    -/
    use i
    /-
      case h
      n : Nat
      inst✝ : NeZero n
      x : Units Complex
      hn0 : Ne (↑n) 0
      i : Nat
      left✝ : LT.lt i n
      H : Eq (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) (H …
      ⊢ Eq (HMul.hMul (↑n) (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) ( …
    -/
    field_simp [hn0, mul_comm ((n : ℕ) : ℂ), mul_comm (i : ℂ)]
    /-
      🎉 no goals
    -/


theorem card_rootsOfUnity (n : ℕ) [NeZero n] : Fintype.card (rootsOfUnity n ℂ) = n :=
  (isPrimitiveRoot_exp n NeZero.out).card_rootsOfUnity


theorem card_primitiveRoots (k : ℕ) : (primitiveRoots k ℂ).card = φ k := by
  /-
    k : Nat
    ⊢ Eq (primitiveRoots k Complex).card k.totient
  -/
  by_cases h : k = 0
    /-
      case pos
      k : Nat
      h : Eq k 0
      ⊢ Eq (primitiveRoots k Complex).card k.totient
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  /-
    case neg
    k : Nat
    h : Not (Eq k 0)
    ⊢ Eq (primitiveRoots k Complex).card k.totient
  -/
  exact (isPrimitiveRoot_exp k h).card_primitiveRoots
  /-
    🎉 no goals
  -/


theorem IsPrimitiveRoot.norm'_eq_one {ζ : ℂ} {n : ℕ} (h : IsPrimitiveRoot ζ n) (hn : n ≠ 0) :
    ‖ζ‖ = 1 :=
  Complex.norm_eq_one_of_pow_eq_one h.pow_eq_one hn


theorem IsPrimitiveRoot.nnnorm_eq_one {ζ : ℂ} {n : ℕ} (h : IsPrimitiveRoot ζ n) (hn : n ≠ 0) :
    ‖ζ‖₊ = 1 :=
  Subtype.ext <| h.norm'_eq_one hn


theorem IsPrimitiveRoot.arg_ext {n m : ℕ} {ζ μ : ℂ} (hζ : IsPrimitiveRoot ζ n)
    (hμ : IsPrimitiveRoot μ m) (hn : n ≠ 0) (hm : m ≠ 0) (h : ζ.arg = μ.arg) : ζ = μ :=
  Complex.ext_abs_arg ((hζ.norm'_eq_one hn).trans (hμ.norm'_eq_one hm).symm) h


theorem IsPrimitiveRoot.arg_eq_zero_iff {n : ℕ} {ζ : ℂ} (hζ : IsPrimitiveRoot ζ n) (hn : n ≠ 0) :
    ζ.arg = 0 ↔ ζ = 1 :=
  ⟨fun h => hζ.arg_ext IsPrimitiveRoot.one hn one_ne_zero (h.trans Complex.arg_one.symm), fun h =>
    h.symm ▸ Complex.arg_one⟩


theorem IsPrimitiveRoot.arg_eq_pi_iff {n : ℕ} {ζ : ℂ} (hζ : IsPrimitiveRoot ζ n) (hn : n ≠ 0) :
    ζ.arg = Real.pi ↔ ζ = -1 :=
  ⟨fun h =>
    hζ.arg_ext (IsPrimitiveRoot.neg_one 0 two_ne_zero.symm) hn two_ne_zero
      (h.trans Complex.arg_neg_one.symm),
    fun h => h.symm ▸ Complex.arg_neg_one⟩


theorem IsPrimitiveRoot.arg {n : ℕ} {ζ : ℂ} (h : IsPrimitiveRoot ζ n) (hn : n ≠ 0) :
    ∃ i : ℤ, ζ.arg = i / n * (2 * Real.pi) ∧ IsCoprime i n ∧ i.natAbs < n := by
  /-
    n : Nat
    ζ : Complex
    h : IsPrimitiveRoot ζ n
    hn : Ne n 0
    ⊢ Exists fun i => And (Eq ζ.arg (HMul.hMul (HDiv.hDiv ↑i ↑n) (HMul.hMul 2 Real …
  -/
  rw [Complex.isPrimitiveRoot_iff _ _ hn] at h
  /-
    n : Nat
    ζ : Complex
    h : Exists fun i => And (LT.lt i n) (Exists fun x => Eq (Complex.exp (HMul.hMu …
    hn : Ne n 0
    ⊢ Exists fun i => And (Eq ζ.arg (HMul.hMul (HDiv.hDiv ↑i ↑n) (HMul.hMul 2 Real …
  -/
  obtain ⟨i, h, hin, rfl⟩ := h
  /-
    case intro.intro.intro
    n : Nat
    hn : Ne n 0
    i : Nat
    h : LT.lt i n
    hin : i.Coprime n
    ⊢ Exists fun i_1 => And (Eq (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑R …
  -/
  rw [mul_comm, ← mul_assoc, Complex.exp_mul_I]
  /-
    case intro.intro.intro
    n : Nat
    hn : Ne n 0
    i : Nat
    h : LT.lt i n
    hin : i.Coprime n
    ⊢ Exists fun i_1 => And (Eq (HAdd.hAdd (Complex.cos (HMul.hMul (HDiv.hDiv ↑i ↑ …
  -/
  refine ⟨if i * 2 ≤ n then i else i - n, ?_, ?_, ?_⟩
  on_goal 2 =>
    replace hin := Nat.isCoprime_iff_coprime.mpr hin
    split_ifs
    · exact hin
    · convert hin.add_mul_left_left (-1) using 1
      rw [mul_neg_one, sub_eq_add_neg]
  on_goal 2 =>
    split_ifs with h₂
    · exact mod_cast h
    suffices (i - n : ℤ).natAbs = n - i by
      rw [this]
      apply tsub_lt_self hn.bot_lt
      contrapose! h₂
      rw [Nat.eq_zero_of_le_zero h₂, zero_mul]
      exact zero_le _
    rw [← Int.natAbs_neg, neg_sub, Int.natAbs_eq_iff]
    exact Or.inl (Int.ofNat_sub h.le).symm
  /-
    case intro.intro.intro.refine_1
    n : Nat
    hn : Ne n 0
    i : Nat
    h : LT.lt i n
    hin : i.Coprime n
    ⊢ Eq (HAdd.hAdd (Complex.cos (HMul.hMul (HDiv.hDiv ↑i ↑n) (HMul.hMul 2 ↑Real.p …
  -/
  split_ifs with h₂
    /-
      case pos
      n : Nat
      hn : Ne n 0
      i : Nat
      h : LT.lt i n
      hin : i.Coprime n
      h₂ : LE.le (HMul.hMul i 2) n
      ⊢ Eq (HAdd.hAdd (Complex.cos (HMul.hMul (HDiv.hDiv ↑i ↑n) (HMul.hMul 2 ↑Real.p …
    -/
  · convert Complex.arg_cos_add_sin_mul_I _
      /-
        case h.e'_2.h.e'_1.h.e'_5.h.e'_1
        n : Nat
        hn : Ne n 0
        i : Nat
        h : LT.lt i n
        hin : i.Coprime n
        h₂ : LE.le (HMul.hMul i 2) n
        ⊢ Eq (HMul.hMul (HDiv.hDiv ↑i ↑n) (HMul.hMul 2 ↑Real.pi)) ↑(HMul.hMul (HDiv.hD …
      -/
    · push_cast; rfl
                 /-
                   🎉 no goals
                 -/
      /-
        case h.e'_2.h.e'_1.h.e'_6.h.e'_5.h.e'_1
        n : Nat
        hn : Ne n 0
        i : Nat
        h : LT.lt i n
        hin : i.Coprime n
        h₂ : LE.le (HMul.hMul i 2) n
        ⊢ Eq (HMul.hMul (HDiv.hDiv ↑i ↑n) (HMul.hMul 2 ↑Real.pi)) ↑(HMul.hMul (HDiv.hD …
      -/
    · push_cast; rfl
                 /-
                   🎉 no goals
                 -/
    /-
      case pos.convert_2
      n : Nat
      hn : Ne n 0
      i : Nat
      h : LT.lt i n
      hin : i.Coprime n
      h₂ : LE.le (HMul.hMul i 2) n
      ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HMul.hMul (HDiv.hDiv ↑↑i …
    -/
    field_simp [hn]
    /-
      case pos.convert_2
      n : Nat
      hn : Ne n 0
      i : Nat
      h : LT.lt i n
      hin : i.Coprime n
      h₂ : LE.le (HMul.hMul i 2) n
      ⊢ And (LT.lt (Neg.neg Real.pi) (HDiv.hDiv (HMul.hMul (↑i) (HMul.hMul 2 Real.pi …
    -/
    refine ⟨(neg_lt_neg Real.pi_pos).trans_le ?_, ?_⟩
      /-
        case pos.convert_2.refine_1
        n : Nat
        hn : Ne n 0
        i : Nat
        h : LT.lt i n
        hin : i.Coprime n
        h₂ : LE.le (HMul.hMul i 2) n
        ⊢ LE.le (-0) (HDiv.hDiv (HMul.hMul (↑i) (HMul.hMul 2 Real.pi)) ↑n)
      -/
    · rw [neg_zero]
      exact mul_nonneg (mul_nonneg i.cast_nonneg <| by simp [Real.pi_pos.le])
        (by rw [inv_nonneg]; simp only [Nat.cast_nonneg])
    /-
      case pos.convert_2.refine_2
      n : Nat
      hn : Ne n 0
      i : Nat
      h : LT.lt i n
      hin : i.Coprime n
      h₂ : LE.le (HMul.hMul i 2) n
      ⊢ LE.le (HDiv.hDiv (HMul.hMul (↑i) (HMul.hMul 2 Real.pi)) ↑n) Real.pi
    -/
    rw [← mul_rotate', mul_div_assoc]
    /-
      case pos.convert_2.refine_2
      n : Nat
      hn : Ne n 0
      i : Nat
      h : LT.lt i n
      hin : i.Coprime n
      h₂ : LE.le (HMul.hMul i 2) n
      ⊢ LE.le (HMul.hMul Real.pi (HDiv.hDiv (HMul.hMul (↑i) 2) ↑n)) Real.pi
    -/
    rw [← mul_one n] at h₂
    exact mul_le_of_le_one_right Real.pi_pos.le
      ((div_le_iff₀' <| mod_cast pos_of_gt h).mpr <| mod_cast h₂)
  /-
    case neg
    n : Nat
    hn : Ne n 0
    i : Nat
    h : LT.lt i n
    hin : i.Coprime n
    h₂ : Not (LE.le (HMul.hMul i 2) n)
    ⊢ Eq (HAdd.hAdd (Complex.cos (HMul.hMul (HDiv.hDiv ↑i ↑n) (HMul.hMul 2 ↑Real.p …
  -/
  rw [← Complex.cos_sub_two_pi, ← Complex.sin_sub_two_pi]
  /-
    case neg
    n : Nat
    hn : Ne n 0
    i : Nat
    h : LT.lt i n
    hin : i.Coprime n
    h₂ : Not (LE.le (HMul.hMul i 2) n)
    ⊢ Eq (HAdd.hAdd (Complex.cos (HSub.hSub (HMul.hMul (HDiv.hDiv ↑i ↑n) (HMul.hMu …
  -/
  convert Complex.arg_cos_add_sin_mul_I _
    /-
      case h.e'_2.h.e'_1.h.e'_5.h.e'_1
      n : Nat
      hn : Ne n 0
      i : Nat
      h : LT.lt i n
      hin : i.Coprime n
      h₂ : Not (LE.le (HMul.hMul i 2) n)
      ⊢ Eq (HSub.hSub (HMul.hMul (HDiv.hDiv ↑i ↑n) (HMul.hMul 2 ↑Real.pi)) (HMul.hMu …
    -/
  · push_cast
    /-
      case h.e'_2.h.e'_1.h.e'_5.h.e'_1
      n : Nat
      hn : Ne n 0
      i : Nat
      h : LT.lt i n
      hin : i.Coprime n
      h₂ : Not (LE.le (HMul.hMul i 2) n)
      ⊢ Eq (HSub.hSub (HMul.hMul (HDiv.hDiv ↑i ↑n) (HMul.hMul 2 ↑Real.pi)) (HMul.hMu …
    -/
    rw [← sub_one_mul, sub_div, div_self]
    /-
      case h.e'_2.h.e'_1.h.e'_5.h.e'_1
      n : Nat
      hn : Ne n 0
      i : Nat
      h : LT.lt i n
      hin : i.Coprime n
      h₂ : Not (LE.le (HMul.hMul i 2) n)
      ⊢ Ne (↑n) 0
    -/
    exact mod_cast hn
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.e'_1.h.e'_6.h.e'_5.h.e'_1
      n : Nat
      hn : Ne n 0
      i : Nat
      h : LT.lt i n
      hin : i.Coprime n
      h₂ : Not (LE.le (HMul.hMul i 2) n)
      ⊢ Eq (HSub.hSub (HMul.hMul (HDiv.hDiv ↑i ↑n) (HMul.hMul 2 ↑Real.pi)) (HMul.hMu …
    -/
  · push_cast
    /-
      case h.e'_2.h.e'_1.h.e'_6.h.e'_5.h.e'_1
      n : Nat
      hn : Ne n 0
      i : Nat
      h : LT.lt i n
      hin : i.Coprime n
      h₂ : Not (LE.le (HMul.hMul i 2) n)
      ⊢ Eq (HSub.hSub (HMul.hMul (HDiv.hDiv ↑i ↑n) (HMul.hMul 2 ↑Real.pi)) (HMul.hMu …
    -/
    rw [← sub_one_mul, sub_div, div_self]
    /-
      case h.e'_2.h.e'_1.h.e'_6.h.e'_5.h.e'_1
      n : Nat
      hn : Ne n 0
      i : Nat
      h : LT.lt i n
      hin : i.Coprime n
      h₂ : Not (LE.le (HMul.hMul i 2) n)
      ⊢ Ne (↑n) 0
    -/
    exact mod_cast hn
    /-
      🎉 no goals
    -/
  /-
    case neg.convert_2
    n : Nat
    hn : Ne n 0
    i : Nat
    h : LT.lt i n
    hin : i.Coprime n
    h₂ : Not (LE.le (HMul.hMul i 2) n)
    ⊢ Membership.mem (Set.Ioc (Neg.neg Real.pi) Real.pi) (HMul.hMul (HDiv.hDiv ↑(H …
  -/
  field_simp [hn]
  /-
    case neg.convert_2
    n : Nat
    hn : Ne n 0
    i : Nat
    h : LT.lt i n
    hin : i.Coprime n
    h₂ : Not (LE.le (HMul.hMul i 2) n)
    ⊢ And (LT.lt (Neg.neg Real.pi) (HDiv.hDiv (HMul.hMul (HSub.hSub ↑i ↑n) (HMul.h …
  -/
  refine ⟨?_, le_trans ?_ Real.pi_pos.le⟩
  on_goal 2 =>
    rw [mul_div_assoc]
    exact mul_nonpos_of_nonpos_of_nonneg (sub_nonpos.mpr <| mod_cast h.le)
      (div_nonneg (by simp [Real.pi_pos.le]) <| by simp)
  rw [← mul_rotate', mul_div_assoc, neg_lt, ← mul_neg, mul_lt_iff_lt_one_right Real.pi_pos, ←
    neg_div, ← neg_mul, neg_sub, div_lt_iff₀, one_mul, sub_mul, sub_lt_comm, ← mul_sub_one]
    /-
      case neg.convert_2.refine_1
      n : Nat
      hn : Ne n 0
      i : Nat
      h : LT.lt i n
      hin : i.Coprime n
      h₂ : Not (LE.le (HMul.hMul i 2) n)
      ⊢ LT.lt (HMul.hMul (↑n) (HSub.hSub 2 1)) (HMul.hMul (↑i) 2)
    -/
  · norm_num
    /-
      case neg.convert_2.refine_1
      n : Nat
      hn : Ne n 0
      i : Nat
      h : LT.lt i n
      hin : i.Coprime n
      h₂ : Not (LE.le (HMul.hMul i 2) n)
      ⊢ LT.lt (↑n) (HMul.hMul (↑i) 2)
    -/
    exact mod_cast not_le.mp h₂
    /-
      🎉 no goals
    -/
    /-
      case neg.convert_2.refine_1
      n : Nat
      hn : Ne n 0
      i : Nat
      h : LT.lt i n
      hin : i.Coprime n
      h₂ : Not (LE.le (HMul.hMul i 2) n)
      ⊢ LT.lt 0 ↑n
    -/
  · exact Nat.cast_pos.mpr hn.bot_lt
    /-
      🎉 no goals
    -/


lemma Complex.norm_eq_one_of_mem_rootsOfUnity {ζ : ℂˣ} {n : ℕ} [NeZero n]
    (hζ : ζ ∈ rootsOfUnity n ℂ) :
    ‖(ζ : ℂ)‖ = 1 := by
  /-
    ζ : Units Complex
    n : Nat
    inst✝ : NeZero n
    hζ : Membership.mem (rootsOfUnity n Complex) ζ
    ⊢ Eq (Norm.norm ↑ζ) 1
  -/
  refine norm_eq_one_of_pow_eq_one ?_ <| NeZero.ne n
  /-
    ζ : Units Complex
    n : Nat
    inst✝ : NeZero n
    hζ : Membership.mem (rootsOfUnity n Complex) ζ
    ⊢ Eq (HPow.hPow (↑ζ) n) 1
  -/
  norm_cast
  /-
    ζ : Units Complex
    n : Nat
    inst✝ : NeZero n
    hζ : Membership.mem (rootsOfUnity n Complex) ζ
    ⊢ Eq (↑(HPow.hPow ζ n)) 1
  -/
  rw [_root_.mem_rootsOfUnity] at hζ
  /-
    ζ : Units Complex
    n : Nat
    inst✝ : NeZero n
    hζ : Eq (HPow.hPow ζ n) 1
    ⊢ Eq (↑(HPow.hPow ζ n)) 1
  -/
  rw [hζ, Units.val_one]
  /-
    🎉 no goals
  -/

