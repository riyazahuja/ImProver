/-- If `K` is a `n`-th cyclotomic extension of `ℚ`, where `2 < n`, then there are no real places
of `K`. -/
theorem nrRealPlaces_eq_zero [IsCyclotomicExtension {n} ℚ K]
    (hn : 2 < n) :
    haveI := IsCyclotomicExtension.numberField {n} ℚ K
    nrRealPlaces K = 0 := by
  /-
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hn : LT.lt 2 n
    ⊢ Eq (NumberField.InfinitePlace.nrRealPlaces K) 0
  -/
  have := IsCyclotomicExtension.numberField {n} ℚ K
  /-
    n : PNat
    K : Type u
    inst✝² : Field K
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hn : LT.lt 2 n
    this : NumberField K
    ⊢ Eq (NumberField.InfinitePlace.nrRealPlaces K) 0
  -/
  apply (IsCyclotomicExtension.zeta_spec n ℚ K).nrRealPlaces_eq_zero_of_two_lt hn
  /-
    🎉 no goals
  -/


/-- If `K` is a `n`-th cyclotomic extension of `ℚ`, then there are `φ n / n` complex places
of `K`. Note that this uses `1 / 2 = 0` in the cases `n = 1, 2`. -/
theorem nrComplexPlaces_eq_totient_div_two [h : IsCyclotomicExtension {n} ℚ K] :
    haveI := IsCyclotomicExtension.numberField {n} ℚ K
    nrComplexPlaces K = φ n / 2 := by
  /-
    n : PNat
    K : Type u
    inst✝¹ : Field K
    inst✝ : CharZero K
    h : IsCyclotomicExtension (Singleton.singleton n) Rat K
    ⊢ Eq (NumberField.InfinitePlace.nrComplexPlaces K) (HDiv.hDiv (↑n).totient 2)
  -/
  have := IsCyclotomicExtension.numberField {n} ℚ K
  /-
    n : PNat
    K : Type u
    inst✝¹ : Field K
    inst✝ : CharZero K
    h : IsCyclotomicExtension (Singleton.singleton n) Rat K
    this : NumberField K
    ⊢ Eq (NumberField.InfinitePlace.nrComplexPlaces K) (HDiv.hDiv (↑n).totient 2)
  -/
  by_cases hn : 2 < n
    /-
      case pos
      n : PNat
      K : Type u
      inst✝¹ : Field K
      inst✝ : CharZero K
      h : IsCyclotomicExtension (Singleton.singleton n) Rat K
      this : NumberField K
      hn : LT.lt 2 n
      ⊢ Eq (NumberField.InfinitePlace.nrComplexPlaces K) (HDiv.hDiv (↑n).totient 2)
    -/
  · obtain ⟨k, hk : φ n = k + k⟩ := totient_even hn
    /-
      case pos.intro
      n : PNat
      K : Type u
      inst✝¹ : Field K
      inst✝ : CharZero K
      h : IsCyclotomicExtension (Singleton.singleton n) Rat K
      this : NumberField K
      hn : LT.lt 2 n
      k : Nat
      hk : Eq (↑n).totient (HAdd.hAdd k k)
      ⊢ Eq (NumberField.InfinitePlace.nrComplexPlaces K) (HDiv.hDiv (↑n).totient 2)
    -/
    have key := card_add_two_mul_card_eq_rank K
    rw [nrRealPlaces_eq_zero K hn, zero_add, IsCyclotomicExtension.finrank (n := n) K
      (cyclotomic.irreducible_rat n.pos), hk, ← two_mul, Nat.mul_right_inj (by norm_num)] at key
    /-
      case pos.intro
      n : PNat
      K : Type u
      inst✝¹ : Field K
      inst✝ : CharZero K
      h : IsCyclotomicExtension (Singleton.singleton n) Rat K
      this : NumberField K
      hn : LT.lt 2 n
      k : Nat
      hk : Eq (↑n).totient (HAdd.hAdd k k)
      key : Eq (NumberField.InfinitePlace.nrComplexPlaces K) k
      ⊢ Eq (NumberField.InfinitePlace.nrComplexPlaces K) (HDiv.hDiv (↑n).totient 2)
    -/
    simp [hk, key, ← two_mul]
    /-
      🎉 no goals
    -/
  · have : φ n = 1 := by
      by_cases h1 : 1 < n.1
      · convert totient_two
        exact (eq_of_le_of_not_lt (succ_le_of_lt h1) hn).symm
      · convert totient_one
        rw [← PNat.one_coe, PNat.coe_inj]
        exact eq_of_le_of_not_lt (not_lt.mp h1) (PNat.not_lt_one _)
    /-
      case neg
      n : PNat
      K : Type u
      inst✝¹ : Field K
      inst✝ : CharZero K
      h : IsCyclotomicExtension (Singleton.singleton n) Rat K
      this✝ : NumberField K
      hn : Not (LT.lt 2 n)
      this : Eq (↑n).totient 1
      ⊢ Eq (NumberField.InfinitePlace.nrComplexPlaces K) (HDiv.hDiv (↑n).totient 2)
    -/
    rw [this]
    /-
      case neg
      n : PNat
      K : Type u
      inst✝¹ : Field K
      inst✝ : CharZero K
      h : IsCyclotomicExtension (Singleton.singleton n) Rat K
      this✝ : NumberField K
      hn : Not (LT.lt 2 n)
      this : Eq (↑n).totient 1
      ⊢ Eq (NumberField.InfinitePlace.nrComplexPlaces K) (1 / 2)
    -/
    apply nrComplexPlaces_eq_zero_of_finrank_eq_one
    /-
      case neg.h
      n : PNat
      K : Type u
      inst✝¹ : Field K
      inst✝ : CharZero K
      h : IsCyclotomicExtension (Singleton.singleton n) Rat K
      this✝ : NumberField K
      hn : Not (LT.lt 2 n)
      this : Eq (↑n).totient 1
      ⊢ Eq (Module.finrank Rat K) 1
    -/
    rw [IsCyclotomicExtension.finrank K (cyclotomic.irreducible_rat n.pos), this]
    /-
      🎉 no goals
    -/



