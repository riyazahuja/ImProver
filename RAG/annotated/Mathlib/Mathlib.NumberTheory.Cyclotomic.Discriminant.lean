/-- The discriminant of the power basis given by a primitive root of unity `ζ` is the same as the
discriminant of the power basis given by `ζ - 1`. -/
theorem discr_zeta_eq_discr_zeta_sub_one (hζ : IsPrimitiveRoot ζ n) :
    discr ℚ (hζ.powerBasis ℚ).basis = discr ℚ (hζ.subOnePowerBasis ℚ).basis := by
  /-
    n : PNat
    K : Type u
    inst✝¹ : Field K
    inst✝ : CharZero K
    ζ : K
    ce : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hζ : IsPrimitiveRoot ζ ↑n
    ⊢ Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.powerBasis Rat hζ).basis) (Algebra.d …
  -/
  haveI : NumberField K := @NumberField.mk _ _ _ (IsCyclotomicExtension.finiteDimensional {n} ℚ K)
  /-
    n : PNat
    K : Type u
    inst✝¹ : Field K
    inst✝ : CharZero K
    ζ : K
    ce : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hζ : IsPrimitiveRoot ζ ↑n
    this : NumberField K
    ⊢ Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.powerBasis Rat hζ).basis) (Algebra.d …
  -/
  have H₁ : (aeval (hζ.powerBasis ℚ).gen) (X - 1 : ℤ[X]) = (hζ.subOnePowerBasis ℚ).gen := by simp
  /-
    n : PNat
    K : Type u
    inst✝¹ : Field K
    inst✝ : CharZero K
    ζ : K
    ce : IsCyclotomicExtension (Singleton.singleton n) Rat K
    hζ : IsPrimitiveRoot ζ ↑n
    this : NumberField K
    H₁ : Eq ((Polynomial.aeval (IsPrimitiveRoot.powerBasis Rat hζ).gen) (HSub.hSub …
    ⊢ Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.powerBasis Rat hζ).basis) (Algebra.d …
  -/
  have H₂ : (aeval (hζ.subOnePowerBasis ℚ).gen) (X + 1 : ℤ[X]) = (hζ.powerBasis ℚ).gen := by simp
  refine discr_eq_discr_of_toMatrix_coeff_isIntegral _ (fun i j => toMatrix_isIntegral H₁ ?_ ?_ _ _)
    fun i j => toMatrix_isIntegral H₂ ?_ ?_ _ _
    /-
      case refine_1
      n : PNat
      K : Type u
      inst✝¹ : Field K
      inst✝ : CharZero K
      ζ : K
      ce : IsCyclotomicExtension (Singleton.singleton n) Rat K
      hζ : IsPrimitiveRoot ζ ↑n
      this : NumberField K
      H₁ : Eq ((Polynomial.aeval (IsPrimitiveRoot.powerBasis Rat hζ).gen) (HSub.hSub …
      H₂ : Eq ((Polynomial.aeval (IsPrimitiveRoot.subOnePowerBasis Rat hζ).gen) (HAd …
      i : Fin (IsPrimitiveRoot.powerBasis Rat hζ).dim
      j : Fin (IsPrimitiveRoot.subOnePowerBasis Rat hζ).dim
      ⊢ IsIntegral Int (IsPrimitiveRoot.powerBasis Rat hζ).gen
    -/
  · exact hζ.isIntegral n.pos
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : PNat
      K : Type u
      inst✝¹ : Field K
      inst✝ : CharZero K
      ζ : K
      ce : IsCyclotomicExtension (Singleton.singleton n) Rat K
      hζ : IsPrimitiveRoot ζ ↑n
      this : NumberField K
      H₁ : Eq ((Polynomial.aeval (IsPrimitiveRoot.powerBasis Rat hζ).gen) (HSub.hSub …
      H₂ : Eq ((Polynomial.aeval (IsPrimitiveRoot.subOnePowerBasis Rat hζ).gen) (HAd …
      i : Fin (IsPrimitiveRoot.powerBasis Rat hζ).dim
      j : Fin (IsPrimitiveRoot.subOnePowerBasis Rat hζ).dim
      ⊢ Eq (minpoly Rat (IsPrimitiveRoot.powerBasis Rat hζ).gen) (Polynomial.map (al …
    -/
  · refine minpoly.isIntegrallyClosed_eq_field_fractions' (K := ℚ) (hζ.isIntegral n.pos)
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      n : PNat
      K : Type u
      inst✝¹ : Field K
      inst✝ : CharZero K
      ζ : K
      ce : IsCyclotomicExtension (Singleton.singleton n) Rat K
      hζ : IsPrimitiveRoot ζ ↑n
      this : NumberField K
      H₁ : Eq ((Polynomial.aeval (IsPrimitiveRoot.powerBasis Rat hζ).gen) (HSub.hSub …
      H₂ : Eq ((Polynomial.aeval (IsPrimitiveRoot.subOnePowerBasis Rat hζ).gen) (HAd …
      i : Fin (IsPrimitiveRoot.subOnePowerBasis Rat hζ).dim
      j : Fin (IsPrimitiveRoot.powerBasis Rat hζ).dim
      ⊢ IsIntegral Int (IsPrimitiveRoot.subOnePowerBasis Rat hζ).gen
    -/
  · exact (hζ.isIntegral n.pos).sub isIntegral_one
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      n : PNat
      K : Type u
      inst✝¹ : Field K
      inst✝ : CharZero K
      ζ : K
      ce : IsCyclotomicExtension (Singleton.singleton n) Rat K
      hζ : IsPrimitiveRoot ζ ↑n
      this : NumberField K
      H₁ : Eq ((Polynomial.aeval (IsPrimitiveRoot.powerBasis Rat hζ).gen) (HSub.hSub …
      H₂ : Eq ((Polynomial.aeval (IsPrimitiveRoot.subOnePowerBasis Rat hζ).gen) (HAd …
      i : Fin (IsPrimitiveRoot.subOnePowerBasis Rat hζ).dim
      j : Fin (IsPrimitiveRoot.powerBasis Rat hζ).dim
      ⊢ Eq (minpoly Rat (IsPrimitiveRoot.subOnePowerBasis Rat hζ).gen) (Polynomial.m …
    -/
  · refine minpoly.isIntegrallyClosed_eq_field_fractions' (K := ℚ) ?_
    /-
      case refine_4
      n : PNat
      K : Type u
      inst✝¹ : Field K
      inst✝ : CharZero K
      ζ : K
      ce : IsCyclotomicExtension (Singleton.singleton n) Rat K
      hζ : IsPrimitiveRoot ζ ↑n
      this : NumberField K
      H₁ : Eq ((Polynomial.aeval (IsPrimitiveRoot.powerBasis Rat hζ).gen) (HSub.hSub …
      H₂ : Eq ((Polynomial.aeval (IsPrimitiveRoot.subOnePowerBasis Rat hζ).gen) (HAd …
      i : Fin (IsPrimitiveRoot.subOnePowerBasis Rat hζ).dim
      j : Fin (IsPrimitiveRoot.powerBasis Rat hζ).dim
      ⊢ IsIntegral Int (IsPrimitiveRoot.subOnePowerBasis Rat hζ).gen
    -/
    exact (hζ.isIntegral n.pos).sub isIntegral_one
    /-
      🎉 no goals
    -/


/-- If `p` is a prime and `IsCyclotomicExtension {p ^ (k + 1)} K L`, then the discriminant of
`hζ.powerBasis K` is `(-1) ^ ((p ^ (k + 1).totient) / 2) * p ^ (p ^ k * ((p - 1) * (k + 1) - 1))`
if `Irreducible (cyclotomic (p ^ (k + 1)) K))`, and `p ^ (k + 1) ≠ 2`. -/
theorem discr_prime_pow_ne_two [IsCyclotomicExtension {p ^ (k + 1)} K L] [hp : Fact (p : ℕ).Prime]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1))) (hirr : Irreducible (cyclotomic (↑(p ^ (k + 1)) : ℕ) K))
    (hk : p ^ (k + 1) ≠ 2) : discr K (hζ.powerBasis K).basis =
      (-1) ^ ((p ^ (k + 1) : ℕ).totient / 2) * p ^ ((p : ℕ) ^ k * ((p - 1) * (k + 1) - 1)) := by
  /-
    p : PNat
    k : Nat
    K : Type u
    L : Type v
    ζ : L
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hp : Fact (Nat.Prime ↑p)
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
  -/
  haveI hne := IsCyclotomicExtension.neZero' (p ^ (k + 1)) K L
  -- Porting note: these two instances are not automatically synthesised and must be constructed
  /-
    p : PNat
    k : Nat
    K : Type u
    L : Type v
    ζ : L
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hp : Fact (Nat.Prime ↑p)
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
    ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
  -/
  haveI mf : Module.Finite K L := finiteDimensional {p ^ (k + 1)} K L
  /-
    p : PNat
    k : Nat
    K : Type u
    L : Type v
    ζ : L
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hp : Fact (Nat.Prime ↑p)
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
    mf : Module.Finite K L
    ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
  -/
  haveI se : Algebra.IsSeparable K L := (isGalois (p ^ (k + 1)) K L).to_isSeparable
  rw [discr_powerBasis_eq_norm, finrank L hirr, hζ.powerBasis_gen _, ←
    hζ.minpoly_eq_cyclotomic_of_irreducible hirr, PNat.pow_coe,
    totient_prime_pow hp.out (succ_pos k), Nat.add_one_sub_one]
  /-
    p : PNat
    k : Nat
    K : Type u
    L : Type v
    ζ : L
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hp : Fact (Nat.Prime ↑p)
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
    mf : Module.Finite K L
    se : Algebra.IsSeparable K L
    ⊢ Eq (HMul.hMul (HPow.hPow (-1) (HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑ …
  -/
  have coe_two : ((2 : ℕ+) : ℕ) = 2 := rfl
  have hp2 : p = 2 → k ≠ 0 := by
    rintro rfl rfl
    exact absurd rfl hk
  /-
    p : PNat
    k : Nat
    K : Type u
    L : Type v
    ζ : L
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hp : Fact (Nat.Prime ↑p)
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
    mf : Module.Finite K L
    se : Algebra.IsSeparable K L
    coe_two : Eq (↑2) 2
    hp2 : Eq p 2 → Ne k 0
    ⊢ Eq (HMul.hMul (HPow.hPow (-1) (HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑ …
  -/
  congr 1
    /-
      case e_a
      p : PNat
      k : Nat
      K : Type u
      L : Type v
      ζ : L
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
      hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
      hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
      mf : Module.Finite K L
      se : Algebra.IsSeparable K L
      coe_two : Eq (↑2) 2
      hp2 : Eq p 2 → Ne k 0
      ⊢ Eq (HPow.hPow (-1) (HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑p) k) (HSub …
    -/
  · rcases eq_or_ne p 2 with (rfl | hp2)
      /-
        case e_a.inl
        k : Nat
        K : Type u
        L : Type v
        ζ : L
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        mf : Module.Finite K L
        se : Algebra.IsSeparable K L
        coe_two : Eq (↑2) 2
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
        hp : Fact (Nat.Prime ↑2)
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd k 1))) K)
        hk : Ne (HPow.hPow 2 (HAdd.hAdd k 1)) 2
        hne : NeZero ↑↑(HPow.hPow 2 (HAdd.hAdd k 1))
        hp2 : Eq 2 2 → Ne k 0
        ⊢ Eq (HPow.hPow (-1) (HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑2) k) (HSub …
      -/
    · rcases Nat.exists_eq_succ_of_ne_zero (hp2 rfl) with ⟨k, rfl⟩
      /-
        case e_a.inl.intro
        K : Type u
        L : Type v
        ζ : L
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        mf : Module.Finite K L
        se : Algebra.IsSeparable K L
        coe_two : Eq (↑2) 2
        hp : Fact (Nat.Prime ↑2)
        k : Nat
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k.s …
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k.succ 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd k.succ 1)) …
        hk : Ne (HPow.hPow 2 (HAdd.hAdd k.succ 1)) 2
        hne : NeZero ↑↑(HPow.hPow 2 (HAdd.hAdd k.succ 1))
        hp2 : Eq 2 2 → Ne k.succ 0
        ⊢ Eq (HPow.hPow (-1) (HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑2) k.succ)  …
      -/
      rw [coe_two, succ_sub_succ_eq_sub, tsub_zero, mul_one]; simp only [_root_.pow_succ']
      /-
        case e_a.inl.intro
        K : Type u
        L : Type v
        ζ : L
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        mf : Module.Finite K L
        se : Algebra.IsSeparable K L
        coe_two : Eq (↑2) 2
        hp : Fact (Nat.Prime ↑2)
        k : Nat
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k.s …
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k.succ 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd k.succ 1)) …
        hk : Ne (HPow.hPow 2 (HAdd.hAdd k.succ 1)) 2
        hne : NeZero ↑↑(HPow.hPow 2 (HAdd.hAdd k.succ 1))
        hp2 : Eq 2 2 → Ne k.succ 0
        ⊢ Eq (HPow.hPow (-1) (HDiv.hDiv (HMul.hMul (HMul.hMul 2 (HPow.hPow 2 k)) (HSub …
      -/
      rw [mul_assoc, Nat.mul_div_cancel_left _ zero_lt_two, Nat.mul_div_cancel_left _ zero_lt_two]
      /-
        case e_a.inl.intro
        K : Type u
        L : Type v
        ζ : L
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        mf : Module.Finite K L
        se : Algebra.IsSeparable K L
        coe_two : Eq (↑2) 2
        hp : Fact (Nat.Prime ↑2)
        k : Nat
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k.s …
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k.succ 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd k.succ 1)) …
        hk : Ne (HPow.hPow 2 (HAdd.hAdd k.succ 1)) 2
        hne : NeZero ↑↑(HPow.hPow 2 (HAdd.hAdd k.succ 1))
        hp2 : Eq 2 2 → Ne k.succ 0
        ⊢ Eq (HPow.hPow (-1) (HMul.hMul (HPow.hPow 2 k) (HSub.hSub (HMul.hMul 2 (HPow. …
      -/
      cases k
        /-
          case e_a.inl.intro.zero
          K : Type u
          L : Type v
          ζ : L
          inst✝³ : Field K
          inst✝² : Field L
          inst✝¹ : Algebra K L
          mf : Module.Finite K L
          se : Algebra.IsSeparable K L
          coe_two : Eq (↑2) 2
          hp : Fact (Nat.Prime ↑2)
          inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd (Na …
          hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd (Nat.succ 0) 1))
          hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd (Nat.succ  …
          hk : Ne (HPow.hPow 2 (HAdd.hAdd (Nat.succ 0) 1)) 2
          hne : NeZero ↑↑(HPow.hPow 2 (HAdd.hAdd (Nat.succ 0) 1))
          hp2 : Eq 2 2 → Ne (Nat.succ 0) 0
          ⊢ Eq (HPow.hPow (-1) (HMul.hMul (HPow.hPow 2 0) (HSub.hSub (HMul.hMul 2 (HPow. …
        -/
      · simp
        /-
          🎉 no goals
        -/
      · simp_rw [_root_.pow_succ', (even_two.mul_right _).neg_one_pow,
          ((even_two.mul_right _).mul_right _).neg_one_pow]
      /-
        case e_a.inr
        p : PNat
        k : Nat
        K : Type u
        L : Type v
        ζ : L
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
        hp : Fact (Nat.Prime ↑p)
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
        hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
        hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
        mf : Module.Finite K L
        se : Algebra.IsSeparable K L
        coe_two : Eq (↑2) 2
        hp2✝ : Eq p 2 → Ne k 0
        hp2 : Ne p 2
        ⊢ Eq (HPow.hPow (-1) (HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑p) k) (HSub …
      -/
    · replace hp2 : (p : ℕ) ≠ 2 := by rwa [Ne, ← coe_two, PNat.coe_inj]
      /-
        case e_a.inr
        p : PNat
        k : Nat
        K : Type u
        L : Type v
        ζ : L
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
        hp : Fact (Nat.Prime ↑p)
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
        hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
        hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
        mf : Module.Finite K L
        se : Algebra.IsSeparable K L
        coe_two : Eq (↑2) 2
        hp2✝ : Eq p 2 → Ne k 0
        hp2 : Ne (↑p) 2
        ⊢ Eq (HPow.hPow (-1) (HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑p) k) (HSub …
      -/
      have hpo : Odd (p : ℕ) := hp.out.odd_of_ne_two hp2
      /-
        case e_a.inr
        p : PNat
        k : Nat
        K : Type u
        L : Type v
        ζ : L
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
        hp : Fact (Nat.Prime ↑p)
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
        hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
        hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
        mf : Module.Finite K L
        se : Algebra.IsSeparable K L
        coe_two : Eq (↑2) 2
        hp2✝ : Eq p 2 → Ne k 0
        hp2 : Ne (↑p) 2
        hpo : Odd ↑p
        ⊢ Eq (HPow.hPow (-1) (HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑p) k) (HSub …
      -/
      obtain ⟨a, ha⟩ := (hp.out.even_sub_one hp2).two_dvd
      rw [ha, mul_left_comm, mul_assoc, Nat.mul_div_cancel_left _ two_pos,
        Nat.mul_div_cancel_left _ two_pos, mul_right_comm, pow_mul, (hpo.pow.mul _).neg_one_pow,
        pow_mul, hpo.pow.neg_one_pow]
      /-
        p : PNat
        k : Nat
        K : Type u
        L : Type v
        ζ : L
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
        hp : Fact (Nat.Prime ↑p)
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
        hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
        hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
        mf : Module.Finite K L
        se : Algebra.IsSeparable K L
        coe_two : Eq (↑2) 2
        hp2✝ : Eq p 2 → Ne k 0
        hp2 : Ne (↑p) 2
        hpo : Odd ↑p
        a : Nat
        ha : Eq (HSub.hSub (↑p) 1) (HMul.hMul 2 a)
        ⊢ Odd (HSub.hSub (HMul.hMul 2 (HMul.hMul (HPow.hPow (↑p) k) a)) 1)
      -/
      refine Nat.Even.sub_odd ?_ (even_two_mul _) odd_one
      /-
        p : PNat
        k : Nat
        K : Type u
        L : Type v
        ζ : L
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
        hp : Fact (Nat.Prime ↑p)
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
        hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
        hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
        mf : Module.Finite K L
        se : Algebra.IsSeparable K L
        coe_two : Eq (↑2) 2
        hp2✝ : Eq p 2 → Ne k 0
        hp2 : Ne (↑p) 2
        hpo : Odd ↑p
        a : Nat
        ha : Eq (HSub.hSub (↑p) 1) (HMul.hMul 2 a)
        ⊢ LE.le 1 (HMul.hMul 2 (HMul.hMul (HPow.hPow (↑p) k) a))
      -/
      rw [mul_left_comm, ← ha]
      /-
        p : PNat
        k : Nat
        K : Type u
        L : Type v
        ζ : L
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
        hp : Fact (Nat.Prime ↑p)
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
        hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
        hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
        mf : Module.Finite K L
        se : Algebra.IsSeparable K L
        coe_two : Eq (↑2) 2
        hp2✝ : Eq p 2 → Ne k 0
        hp2 : Ne (↑p) 2
        hpo : Odd ↑p
        a : Nat
        ha : Eq (HSub.hSub (↑p) 1) (HMul.hMul 2 a)
        ⊢ LE.le 1 (HMul.hMul (HPow.hPow (↑p) k) (HSub.hSub (↑p) 1))
      -/
      exact one_le_mul (one_le_pow _ _ hp.1.pos) (succ_le_iff.2 <| tsub_pos_of_lt hp.1.one_lt)
      /-
        🎉 no goals
      -/
    /-
      case e_a
      p : PNat
      k : Nat
      K : Type u
      L : Type v
      ζ : L
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
      hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
      hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
      mf : Module.Finite K L
      se : Algebra.IsSeparable K L
      coe_two : Eq (↑2) 2
      hp2 : Eq p 2 → Ne k 0
      ⊢ Eq ((Algebra.norm K) ((Polynomial.aeval ζ) (Polynomial.derivative (Polynomia …
    -/
  · have H := congr_arg (@derivative K _) (cyclotomic_prime_pow_mul_X_pow_sub_one K p k)
    rw [derivative_mul, derivative_sub, derivative_one, sub_zero, derivative_X_pow, C_eq_natCast,
      derivative_sub, derivative_one, sub_zero, derivative_X_pow, C_eq_natCast, ← PNat.pow_coe,
      hζ.minpoly_eq_cyclotomic_of_irreducible hirr] at H
    /-
      case e_a
      p : PNat
      k : Nat
      K : Type u
      L : Type v
      ζ : L
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
      hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
      hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
      mf : Module.Finite K L
      se : Algebra.IsSeparable K L
      coe_two : Eq (↑2) 2
      hp2 : Eq p 2 → Ne k 0
      H : Eq (HAdd.hAdd (HMul.hMul (Polynomial.derivative (minpoly K ζ)) (HSub.hSub  …
      ⊢ Eq ((Algebra.norm K) ((Polynomial.aeval ζ) (Polynomial.derivative (Polynomia …
    -/
    replace H := congr_arg (fun P => aeval ζ P) H
    simp only [aeval_add, aeval_mul, minpoly.aeval, zero_mul, add_zero, aeval_natCast,
      _root_.map_sub, aeval_one, aeval_X_pow] at H
    /-
      case e_a
      p : PNat
      k : Nat
      K : Type u
      L : Type v
      ζ : L
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
      hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
      hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
      mf : Module.Finite K L
      se : Algebra.IsSeparable K L
      coe_two : Eq (↑2) 2
      hp2 : Eq p 2 → Ne k 0
      H : Eq (HMul.hMul ((Polynomial.aeval ζ) (Polynomial.derivative (minpoly K ζ))) …
      ⊢ Eq ((Algebra.norm K) ((Polynomial.aeval ζ) (Polynomial.derivative (Polynomia …
    -/
    replace H := congr_arg (Algebra.norm K) H
    have hnorm : (norm K) (ζ ^ (p : ℕ) ^ k - 1) = (p : K) ^ (p : ℕ) ^ k := by
      by_cases hp : p = 2
      · exact mod_cast hζ.norm_pow_sub_one_eq_prime_pow_of_ne_zero hirr le_rfl (hp2 hp)
      · exact mod_cast hζ.norm_pow_sub_one_of_prime_ne_two hirr le_rfl hp
    rw [MonoidHom.map_mul, hnorm, MonoidHom.map_mul, ← map_natCast (algebraMap K L),
      Algebra.norm_algebraMap, finrank L hirr] at H
    conv_rhs at H => -- Porting note: need to drill down to successfully rewrite the totient
      enter [1, 2]
      rw [PNat.pow_coe, ← succ_eq_add_one, totient_prime_pow hp.out (succ_pos k), Nat.sub_one,
        Nat.pred_succ]
    rw [← hζ.minpoly_eq_cyclotomic_of_irreducible hirr, map_pow, hζ.norm_eq_one hk hirr, one_pow,
      mul_one, PNat.pow_coe, cast_pow, ← pow_mul, ← mul_assoc, mul_comm (k + 1), mul_assoc] at H
    /-
      case e_a
      p : PNat
      k : Nat
      K : Type u
      L : Type v
      ζ : L
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
      hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
      hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
      mf : Module.Finite K L
      se : Algebra.IsSeparable K L
      coe_two : Eq (↑2) 2
      hp2 : Eq p 2 → Ne k 0
      H : Eq (HMul.hMul ((Algebra.norm K) ((Polynomial.aeval ζ) (Polynomial.derivati …
      hnorm : Eq ((Algebra.norm K) (HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) k)) 1)) ( …
      ⊢ Eq ((Algebra.norm K) ((Polynomial.aeval ζ) (Polynomial.derivative (Polynomia …
    -/
    have := mul_pos (succ_pos k) (tsub_pos_of_lt hp.out.one_lt)
    /-
      case e_a
      p : PNat
      k : Nat
      K : Type u
      L : Type v
      ζ : L
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
      hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
      hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
      mf : Module.Finite K L
      se : Algebra.IsSeparable K L
      coe_two : Eq (↑2) 2
      hp2 : Eq p 2 → Ne k 0
      H : Eq (HMul.hMul ((Algebra.norm K) ((Polynomial.aeval ζ) (Polynomial.derivati …
      hnorm : Eq ((Algebra.norm K) (HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) k)) 1)) ( …
      this : LT.lt 0 (HMul.hMul k.succ (HSub.hSub (↑p) 1))
      ⊢ Eq ((Algebra.norm K) ((Polynomial.aeval ζ) (Polynomial.derivative (Polynomia …
    -/
    rw [← succ_pred_eq_of_pos this, mul_succ, pow_add _ _ ((p : ℕ) ^ k)] at H
    /-
      case e_a
      p : PNat
      k : Nat
      K : Type u
      L : Type v
      ζ : L
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
      hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
      hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
      mf : Module.Finite K L
      se : Algebra.IsSeparable K L
      coe_two : Eq (↑2) 2
      hp2 : Eq p 2 → Ne k 0
      H : Eq (HMul.hMul ((Algebra.norm K) ((Polynomial.aeval ζ) (Polynomial.derivati …
      hnorm : Eq ((Algebra.norm K) (HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) k)) 1)) ( …
      this : LT.lt 0 (HMul.hMul k.succ (HSub.hSub (↑p) 1))
      ⊢ Eq ((Algebra.norm K) ((Polynomial.aeval ζ) (Polynomial.derivative (Polynomia …
    -/
    replace H := (mul_left_inj' fun h => ?_).1 H
      /-
        case e_a.refine_2
        p : PNat
        k : Nat
        K : Type u
        L : Type v
        ζ : L
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
        hp : Fact (Nat.Prime ↑p)
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
        hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
        hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
        mf : Module.Finite K L
        se : Algebra.IsSeparable K L
        coe_two : Eq (↑2) 2
        hp2 : Eq p 2 → Ne k 0
        hnorm : Eq ((Algebra.norm K) (HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) k)) 1)) ( …
        this : LT.lt 0 (HMul.hMul k.succ (HSub.hSub (↑p) 1))
        H : Eq ((Algebra.norm K) ((Polynomial.aeval ζ) (Polynomial.derivative (Polynom …
        ⊢ Eq ((Algebra.norm K) ((Polynomial.aeval ζ) (Polynomial.derivative (Polynomia …
      -/
    · simp only [H, mul_comm _ (k + 1)]; norm_cast
                                         /-
                                           🎉 no goals
                                         -/
    · -- Porting note: was `replace h := pow_eq_zero h; rw [coe_coe] at h; simpa using hne.1`
      /-
        case e_a.refine_1
        p : PNat
        k : Nat
        K : Type u
        L : Type v
        ζ : L
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
        hp : Fact (Nat.Prime ↑p)
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
        hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
        hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
        mf : Module.Finite K L
        se : Algebra.IsSeparable K L
        coe_two : Eq (↑2) 2
        hp2 : Eq p 2 → Ne k 0
        H : Eq (HMul.hMul ((Algebra.norm K) ((Polynomial.aeval ζ) (Polynomial.derivati …
        hnorm : Eq ((Algebra.norm K) (HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) k)) 1)) ( …
        this : LT.lt 0 (HMul.hMul k.succ (HSub.hSub (↑p) 1))
        h : Eq (HPow.hPow (↑↑p) (HPow.hPow (↑p) k)) 0
        ⊢ False
      -/
      have := hne.1
      /-
        case e_a.refine_1
        p : PNat
        k : Nat
        K : Type u
        L : Type v
        ζ : L
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
        hp : Fact (Nat.Prime ↑p)
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
        hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
        hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
        mf : Module.Finite K L
        se : Algebra.IsSeparable K L
        coe_two : Eq (↑2) 2
        hp2 : Eq p 2 → Ne k 0
        H : Eq (HMul.hMul ((Algebra.norm K) ((Polynomial.aeval ζ) (Polynomial.derivati …
        hnorm : Eq ((Algebra.norm K) (HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) k)) 1)) ( …
        this✝ : LT.lt 0 (HMul.hMul k.succ (HSub.hSub (↑p) 1))
        h : Eq (HPow.hPow (↑↑p) (HPow.hPow (↑p) k)) 0
        this : Ne (↑↑(HPow.hPow p (HAdd.hAdd k 1))) 0
        ⊢ False
      -/
      rw [PNat.pow_coe, Nat.cast_pow, Ne, pow_eq_zero_iff (by omega)] at this
      /-
        case e_a.refine_1
        p : PNat
        k : Nat
        K : Type u
        L : Type v
        ζ : L
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
        hp : Fact (Nat.Prime ↑p)
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
        hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
        hne : NeZero ↑↑(HPow.hPow p (HAdd.hAdd k 1))
        mf : Module.Finite K L
        se : Algebra.IsSeparable K L
        coe_two : Eq (↑2) 2
        hp2 : Eq p 2 → Ne k 0
        H : Eq (HMul.hMul ((Algebra.norm K) ((Polynomial.aeval ζ) (Polynomial.derivati …
        hnorm : Eq ((Algebra.norm K) (HSub.hSub (HPow.hPow ζ (HPow.hPow (↑p) k)) 1)) ( …
        this✝ : LT.lt 0 (HMul.hMul k.succ (HSub.hSub (↑p) 1))
        h : Eq (HPow.hPow (↑↑p) (HPow.hPow (↑p) k)) 0
        this : Not (Eq (↑↑p) 0)
        ⊢ False
      -/
      exact absurd (pow_eq_zero h) this
      /-
        🎉 no goals
      -/


/-- If `p` is a prime and `IsCyclotomicExtension {p ^ (k + 1)} K L`, then the discriminant of
`hζ.powerBasis K` is `(-1) ^ (p ^ k * (p - 1) / 2) * p ^ (p ^ k * ((p - 1) * (k + 1) - 1))`
if `Irreducible (cyclotomic (p ^ (k + 1)) K))`, and `p ^ (k + 1) ≠ 2`. -/
theorem discr_prime_pow_ne_two' [IsCyclotomicExtension {p ^ (k + 1)} K L] [hp : Fact (p : ℕ).Prime]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1))) (hirr : Irreducible (cyclotomic (↑(p ^ (k + 1)) : ℕ) K))
    (hk : p ^ (k + 1) ≠ 2) : discr K (hζ.powerBasis K).basis =
      (-1) ^ ((p : ℕ) ^ k * (p - 1) / 2) * p ^ ((p : ℕ) ^ k * ((p - 1) * (k + 1) - 1)) := by
  /-
    p : PNat
    k : Nat
    K : Type u
    L : Type v
    ζ : L
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hp : Fact (Nat.Prime ↑p)
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
    hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
  -/
  simpa [totient_prime_pow hp.out (succ_pos k)] using discr_prime_pow_ne_two hζ hirr hk
  /-
    🎉 no goals
  -/


/-- If `p` is a prime and `IsCyclotomicExtension {p ^ k} K L`, then the discriminant of
`hζ.powerBasis K` is `(-1) ^ ((p ^ k).totient / 2) * p ^ (p ^ (k - 1) * ((p - 1) * k - 1))`
if `Irreducible (cyclotomic (p ^ k) K))`. Beware that in the cases `p ^ k = 1` and `p ^ k = 2`
the formula uses `1 / 2 = 0` and `0 - 1 = 0`. It is useful only to have a uniform result.
See also `IsCyclotomicExtension.discr_prime_pow_eq_unit_mul_pow`. -/
theorem discr_prime_pow [hcycl : IsCyclotomicExtension {p ^ k} K L] [hp : Fact (p : ℕ).Prime]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ k)) (hirr : Irreducible (cyclotomic (↑(p ^ k) : ℕ) K)) :
    discr K (hζ.powerBasis K).basis =
      (-1) ^ ((p ^ k : ℕ).totient / 2) * p ^ ((p : ℕ) ^ (k - 1) * ((p - 1) * k - 1)) := by
  /-
    p : PNat
    k : Nat
    K : Type u
    L : Type v
    ζ : L
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) K L
    hp : Fact (Nat.Prime ↑p)
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p k)) K)
    ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
  -/
  cases' k with k k
  · simp only [coe_basis, _root_.pow_zero, powerBasis_gen _ hζ, totient_one, mul_zero, mul_one,
      show 1 / 2 = 0 by rfl, discr, traceMatrix]
    /-
      case zero
      p : PNat
      K : Type u
      L : Type v
      ζ : L
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      hp : Fact (Nat.Prime ↑p)
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 0)) K L
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p 0)
      hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p 0)) K)
      ⊢ Eq (Matrix.of fun i j => ((Algebra.traceForm K L) (HPow.hPow ζ ↑i)) (HPow.hP …
    -/
    have hζone : ζ = 1 := by simpa using hζ
    rw [hζ.powerBasis_dim _, hζone, ← (algebraMap K L).map_one,
      minpoly.eq_X_sub_C_of_algebraMap_inj _ (algebraMap K L).injective, natDegree_X_sub_C]
    /-
      case zero
      p : PNat
      K : Type u
      L : Type v
      ζ : L
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      hp : Fact (Nat.Prime ↑p)
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 0)) K L
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p 0)
      hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p 0)) K)
      hζone : Eq ζ 1
      ⊢ Eq (Matrix.of fun i j => ((Algebra.traceForm K L) (HPow.hPow ((algebraMap K  …
    -/
    simp only [traceMatrix, map_one, one_pow, Matrix.det_unique, traceForm_apply, mul_one]
    /-
      case zero
      p : PNat
      K : Type u
      L : Type v
      ζ : L
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      hp : Fact (Nat.Prime ↑p)
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 0)) K L
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p 0)
      hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p 0)) K)
      hζone : Eq ζ 1
      ⊢ Eq (Matrix.of (fun i j => (Algebra.trace K L) 1) Inhabited.default Inhabited …
    -/
    rw [← (algebraMap K L).map_one, trace_algebraMap, finrank _ hirr]
    /-
      case zero
      p : PNat
      K : Type u
      L : Type v
      ζ : L
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      hp : Fact (Nat.Prime ↑p)
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 0)) K L
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p 0)
      hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p 0)) K)
      hζone : Eq ζ 1
      ⊢ Eq (Matrix.of (fun i j => HSMul.hSMul (↑(HPow.hPow p 0)).totient 1) Inhabite …
    -/
    norm_num
    /-
      🎉 no goals
    -/
    /-
      case succ
      p : PNat
      K : Type u
      L : Type v
      ζ : L
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      hp : Fact (Nat.Prime ↑p)
      k : Nat
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
      ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
    -/
  · by_cases hk : p ^ (k + 1) = 2
      /-
        case pos
        p : PNat
        K : Type u
        L : Type v
        ζ : L
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        hp : Fact (Nat.Prime ↑p)
        k : Nat
        hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
        hk : Eq (HPow.hPow p (HAdd.hAdd k 1)) 2
        ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
      -/
    · have coe_two : 2 = ((2 : ℕ+) : ℕ) := rfl
      have hp : p = 2 := by
        rw [← PNat.coe_inj, PNat.pow_coe, ← pow_one 2] at hk
        replace hk :=
          eq_of_prime_pow_eq (prime_iff.1 hp.out) (prime_iff.1 Nat.prime_two) (succ_pos _) hk
        rwa [coe_two, PNat.coe_inj] at hk
      /-
        case pos
        p : PNat
        K : Type u
        L : Type v
        ζ : L
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        hp✝ : Fact (Nat.Prime ↑p)
        k : Nat
        hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
        hk : Eq (HPow.hPow p (HAdd.hAdd k 1)) 2
        coe_two : Eq 2 ↑2
        hp : Eq p 2
        ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
      -/
      subst hp
      /-
        case pos
        K : Type u
        L : Type v
        ζ : L
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        k : Nat
        coe_two : Eq 2 ↑2
        hp : Fact (Nat.Prime ↑2)
        hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd k 1))) K)
        hk : Eq (HPow.hPow 2 (HAdd.hAdd k 1)) 2
        ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
      -/
      rw [← PNat.coe_inj, PNat.pow_coe] at hk
      /-
        case pos
        K : Type u
        L : Type v
        ζ : L
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        k : Nat
        coe_two : Eq 2 ↑2
        hp : Fact (Nat.Prime ↑2)
        hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd k 1))) K)
        hk : Eq (HPow.hPow (↑2) (HAdd.hAdd k 1)) ↑2
        ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
      -/
      nth_rw 2 [← pow_one 2] at hk
      /-
        case pos
        K : Type u
        L : Type v
        ζ : L
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        k : Nat
        coe_two : Eq 2 ↑2
        hp : Fact (Nat.Prime ↑2)
        hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd k 1))) K)
        hk : Eq (HPow.hPow (↑2) (HAdd.hAdd k 1)) ↑(HPow.hPow 2 1)
        ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
      -/
      replace hk := Nat.pow_right_injective rfl.le hk
      /-
        case pos
        K : Type u
        L : Type v
        ζ : L
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        k : Nat
        coe_two : Eq 2 ↑2
        hp : Fact (Nat.Prime ↑2)
        hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd k 1))) K)
        hk : Eq (HAdd.hAdd k 1) 1
        ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
      -/
      rw [add_left_eq_self] at hk
      /-
        case pos
        K : Type u
        L : Type v
        ζ : L
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        k : Nat
        coe_two : Eq 2 ↑2
        hp : Fact (Nat.Prime ↑2)
        hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd k 1))) K)
        hk : Eq k 0
        ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
      -/
      subst hk
      /-
        case pos
        K : Type u
        L : Type v
        ζ : L
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        coe_two : Eq 2 ↑2
        hp : Fact (Nat.Prime ↑2)
        hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd 0 1 …
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd 0 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd 0 1))) K)
        ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
      -/
      rw [pow_one] at hζ hcycl
      have : natDegree (minpoly K ζ) = 1 := by
        rw [hζ.eq_neg_one_of_two_right, show (-1 : L) = algebraMap K L (-1) by simp,
          minpoly.eq_X_sub_C_of_algebraMap_inj _ (NoZeroSMulDivisors.algebraMap_injective K L)]
        exact natDegree_X_sub_C (-1)
      /-
        case pos
        K : Type u
        L : Type v
        ζ : L
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        coe_two : Eq 2 ↑2
        hp : Fact (Nat.Prime ↑2)
        hcycl✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd 0  …
        hcycl : IsCyclotomicExtension (Singleton.singleton 2) K L
        hζ✝ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd 0 1))
        hζ : IsPrimitiveRoot ζ ↑2
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd 0 1))) K)
        this : Eq (minpoly K ζ).natDegree 1
        ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ✝).basis) (HMul.hMul (H …
      -/
      rcases Fin.equiv_iff_eq.2 this with ⟨e⟩
      /-
        case pos.intro
        K : Type u
        L : Type v
        ζ : L
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        coe_two : Eq 2 ↑2
        hp : Fact (Nat.Prime ↑2)
        hcycl✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd 0  …
        hcycl : IsCyclotomicExtension (Singleton.singleton 2) K L
        hζ✝ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd 0 1))
        hζ : IsPrimitiveRoot ζ ↑2
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd 0 1))) K)
        this : Eq (minpoly K ζ).natDegree 1
        e : Equiv (Fin (minpoly K ζ).natDegree) (Fin 1)
        ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ✝).basis) (HMul.hMul (H …
      -/
      rw [← Algebra.discr_reindex K (hζ.powerBasis K).basis e, coe_basis, powerBasis_gen]; norm_num
      /-
        case pos.intro
        K : Type u
        L : Type v
        ζ : L
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        coe_two : Eq 2 ↑2
        hp : Fact (Nat.Prime ↑2)
        hcycl✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd 0  …
        hcycl : IsCyclotomicExtension (Singleton.singleton 2) K L
        hζ✝ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd 0 1))
        hζ : IsPrimitiveRoot ζ ↑2
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd 0 1))) K)
        this : Eq (minpoly K ζ).natDegree 1
        e : Equiv (Fin (minpoly K ζ).natDegree) (Fin 1)
        ⊢ Eq (Algebra.discr K (Function.comp (fun i => HPow.hPow ζ ↑i) ⇑e.symm)) 1
      -/
      simp_rw [hζ.eq_neg_one_of_two_right, show (-1 : L) = algebraMap K L (-1) by simp]
      /-
        case pos.intro
        K : Type u
        L : Type v
        ζ : L
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        coe_two : Eq 2 ↑2
        hp : Fact (Nat.Prime ↑2)
        hcycl✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd 0  …
        hcycl : IsCyclotomicExtension (Singleton.singleton 2) K L
        hζ✝ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd 0 1))
        hζ : IsPrimitiveRoot ζ ↑2
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd 0 1))) K)
        this : Eq (minpoly K ζ).natDegree 1
        e : Equiv (Fin (minpoly K ζ).natDegree) (Fin 1)
        ⊢ Eq (Algebra.discr K (Function.comp (fun i => HPow.hPow ((algebraMap K L) (-1 …
      -/
      convert_to (discr K fun i : Fin 1 ↦ (algebraMap K L) (-1) ^ ↑i) = _
        /-
          case h.e'_2
          K : Type u
          L : Type v
          ζ : L
          inst✝² : Field K
          inst✝¹ : Field L
          inst✝ : Algebra K L
          coe_two : Eq 2 ↑2
          hp : Fact (Nat.Prime ↑2)
          hcycl✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd 0  …
          hcycl : IsCyclotomicExtension (Singleton.singleton 2) K L
          hζ✝ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd 0 1))
          hζ : IsPrimitiveRoot ζ ↑2
          hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd 0 1))) K)
          this : Eq (minpoly K ζ).natDegree 1
          e : Equiv (Fin (minpoly K ζ).natDegree) (Fin 1)
          ⊢ Eq (Algebra.discr K (Function.comp (fun i => HPow.hPow ((algebraMap K L) (-1 …
        -/
      · congr
        /-
          case h.e'_2.e_b
          K : Type u
          L : Type v
          ζ : L
          inst✝² : Field K
          inst✝¹ : Field L
          inst✝ : Algebra K L
          coe_two : Eq 2 ↑2
          hp : Fact (Nat.Prime ↑2)
          hcycl✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd 0  …
          hcycl : IsCyclotomicExtension (Singleton.singleton 2) K L
          hζ✝ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd 0 1))
          hζ : IsPrimitiveRoot ζ ↑2
          hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd 0 1))) K)
          this : Eq (minpoly K ζ).natDegree 1
          e : Equiv (Fin (minpoly K ζ).natDegree) (Fin 1)
          ⊢ Eq (Function.comp (fun i => HPow.hPow ((algebraMap K L) (-1)) ↑i) ⇑e.symm) f …
        -/
        ext i
        /-
          case h.e'_2.e_b.h
          K : Type u
          L : Type v
          ζ : L
          inst✝² : Field K
          inst✝¹ : Field L
          inst✝ : Algebra K L
          coe_two : Eq 2 ↑2
          hp : Fact (Nat.Prime ↑2)
          hcycl✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd 0  …
          hcycl : IsCyclotomicExtension (Singleton.singleton 2) K L
          hζ✝ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd 0 1))
          hζ : IsPrimitiveRoot ζ ↑2
          hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd 0 1))) K)
          this : Eq (minpoly K ζ).natDegree 1
          e : Equiv (Fin (minpoly K ζ).natDegree) (Fin 1)
          i : Fin 1
          ⊢ Eq (Function.comp (fun i => HPow.hPow ((algebraMap K L) (-1)) ↑i) (⇑e.symm)  …
        -/
        simp only [map_neg, map_one, Function.comp_apply, Fin.val_eq_zero, _root_.pow_zero]
        /-
          case h.e'_2.e_b.h
          K : Type u
          L : Type v
          ζ : L
          inst✝² : Field K
          inst✝¹ : Field L
          inst✝ : Algebra K L
          coe_two : Eq 2 ↑2
          hp : Fact (Nat.Prime ↑2)
          hcycl✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd 0  …
          hcycl : IsCyclotomicExtension (Singleton.singleton 2) K L
          hζ✝ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd 0 1))
          hζ : IsPrimitiveRoot ζ ↑2
          hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd 0 1))) K)
          this : Eq (minpoly K ζ).natDegree 1
          e : Equiv (Fin (minpoly K ζ).natDegree) (Fin 1)
          i : Fin 1
          ⊢ Eq (HPow.hPow (-1) ↑(e.symm i)) 1
        -/
        suffices (e.symm i : ℕ) = 0 by simp [this]
        /-
          case h.e'_2.e_b.h
          K : Type u
          L : Type v
          ζ : L
          inst✝² : Field K
          inst✝¹ : Field L
          inst✝ : Algebra K L
          coe_two : Eq 2 ↑2
          hp : Fact (Nat.Prime ↑2)
          hcycl✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd 0  …
          hcycl : IsCyclotomicExtension (Singleton.singleton 2) K L
          hζ✝ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd 0 1))
          hζ : IsPrimitiveRoot ζ ↑2
          hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd 0 1))) K)
          this : Eq (minpoly K ζ).natDegree 1
          e : Equiv (Fin (minpoly K ζ).natDegree) (Fin 1)
          i : Fin 1
          ⊢ Eq (↑(e.symm i)) 0
        -/
        rw [← Nat.lt_one_iff]
        /-
          case h.e'_2.e_b.h
          K : Type u
          L : Type v
          ζ : L
          inst✝² : Field K
          inst✝¹ : Field L
          inst✝ : Algebra K L
          coe_two : Eq 2 ↑2
          hp : Fact (Nat.Prime ↑2)
          hcycl✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd 0  …
          hcycl : IsCyclotomicExtension (Singleton.singleton 2) K L
          hζ✝ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd 0 1))
          hζ : IsPrimitiveRoot ζ ↑2
          hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd 0 1))) K)
          this : Eq (minpoly K ζ).natDegree 1
          e : Equiv (Fin (minpoly K ζ).natDegree) (Fin 1)
          i : Fin 1
          ⊢ LT.lt (↑(e.symm i)) 1
        -/
        convert (e.symm i).2
        /-
          case h.e'_4
          K : Type u
          L : Type v
          ζ : L
          inst✝² : Field K
          inst✝¹ : Field L
          inst✝ : Algebra K L
          coe_two : Eq 2 ↑2
          hp : Fact (Nat.Prime ↑2)
          hcycl✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd 0  …
          hcycl : IsCyclotomicExtension (Singleton.singleton 2) K L
          hζ✝ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd 0 1))
          hζ : IsPrimitiveRoot ζ ↑2
          hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd 0 1))) K)
          this : Eq (minpoly K ζ).natDegree 1
          e : Equiv (Fin (minpoly K ζ).natDegree) (Fin 1)
          i : Fin 1
          ⊢ Eq 1 (minpoly K ζ).natDegree
        -/
        rw [this]
        /-
          🎉 no goals
        -/
      · simp only [discr, traceMatrix_apply, Matrix.det_unique, Fin.default_eq_zero, Fin.val_zero,
          _root_.pow_zero, traceForm_apply, mul_one]
        /-
          case pos.intro.convert_2
          K : Type u
          L : Type v
          ζ : L
          inst✝² : Field K
          inst✝¹ : Field L
          inst✝ : Algebra K L
          coe_two : Eq 2 ↑2
          hp : Fact (Nat.Prime ↑2)
          hcycl✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd 0  …
          hcycl : IsCyclotomicExtension (Singleton.singleton 2) K L
          hζ✝ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd 0 1))
          hζ : IsPrimitiveRoot ζ ↑2
          hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow 2 (HAdd.hAdd 0 1))) K)
          this : Eq (minpoly K ζ).natDegree 1
          e : Equiv (Fin (minpoly K ζ).natDegree) (Fin 1)
          ⊢ Eq ((Algebra.trace K L) 1) 1
        -/
        rw [← (algebraMap K L).map_one, trace_algebraMap, finrank _ hirr]; norm_num
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
      /-
        case neg
        p : PNat
        K : Type u
        L : Type v
        ζ : L
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        hp : Fact (Nat.Prime ↑p)
        k : Nat
        hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
        hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p (HAdd.hAdd k 1))) K)
        hk : Not (Eq (HPow.hPow p (HAdd.hAdd k 1)) 2)
        ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
      -/
    · exact discr_prime_pow_ne_two hζ hirr hk
      /-
        🎉 no goals
      -/


/-- If `p` is a prime and `IsCyclotomicExtension {p ^ k} K L`, then there are `u : ℤˣ` and
`n : ℕ` such that the discriminant of `hζ.powerBasis K` is `u * p ^ n`. Often this is enough and
less cumbersome to use than `IsCyclotomicExtension.discr_prime_pow`. -/
theorem discr_prime_pow_eq_unit_mul_pow [IsCyclotomicExtension {p ^ k} K L]
    [hp : Fact (p : ℕ).Prime] (hζ : IsPrimitiveRoot ζ ↑(p ^ k))
    (hirr : Irreducible (cyclotomic (↑(p ^ k) : ℕ) K)) :
    ∃ (u : ℤˣ) (n : ℕ), discr K (hζ.powerBasis K).basis = u * p ^ n := by
  /-
    p : PNat
    k : Nat
    K : Type u
    L : Type v
    ζ : L
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) K L
    hp : Fact (Nat.Prime ↑p)
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p k)) K)
    ⊢ Exists fun u => Exists fun n => Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerB …
  -/
  rw [discr_prime_pow hζ hirr]
  /-
    p : PNat
    k : Nat
    K : Type u
    L : Type v
    ζ : L
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) K L
    hp : Fact (Nat.Prime ↑p)
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p k)) K)
    ⊢ Exists fun u => Exists fun n => Eq (HMul.hMul (HPow.hPow (-1) (HDiv.hDiv (HP …
  -/
  by_cases heven : Even ((p ^ k : ℕ).totient / 2)
    /-
      case pos
      p : PNat
      k : Nat
      K : Type u
      L : Type v
      ζ : L
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) K L
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
      hirr : Irreducible (Polynomial.cyclotomic (↑(HPow.hPow p k)) K)
      heven : Even (HDiv.hDiv (HPow.hPow (↑p) k).totient 2)
      ⊢ Exists fun u => Exists fun n => Eq (HMul.hMul (HPow.hPow (-1) (HDiv.hDiv (HP …
    -/
  · exact ⟨1, (p : ℕ) ^ (k - 1) * ((p - 1) * k - 1), by rw [heven.neg_one_pow]; norm_num⟩
    /-
      🎉 no goals
    -/
  · exact ⟨-1, (p : ℕ) ^ (k - 1) * ((p - 1) * k - 1), by
      rw [(not_even_iff_odd.1 heven).neg_one_pow]; norm_num⟩


/-- If `p` is an odd prime and `IsCyclotomicExtension {p} K L`, then
`discr K (hζ.powerBasis K).basis = (-1) ^ ((p - 1) / 2) * p ^ (p - 2)` if
`Irreducible (cyclotomic p K)`. -/
theorem discr_odd_prime [IsCyclotomicExtension {p} K L] [hp : Fact (p : ℕ).Prime]
    (hζ : IsPrimitiveRoot ζ p) (hirr : Irreducible (cyclotomic p K)) (hodd : p ≠ 2) :
    discr K (hζ.powerBasis K).basis = (-1) ^ (((p : ℕ) - 1) / 2) * p ^ ((p : ℕ) - 2) := by
  have : IsCyclotomicExtension {p ^ (0 + 1)} K L := by
    rw [zero_add, pow_one]
    infer_instance
  /-
    p : PNat
    K : Type u
    L : Type v
    ζ : L
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton p) K L
    hp : Fact (Nat.Prime ↑p)
    hζ : IsPrimitiveRoot ζ ↑p
    hirr : Irreducible (Polynomial.cyclotomic (↑p) K)
    hodd : Ne p 2
    this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
    ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
  -/
  have hζ' : IsPrimitiveRoot ζ (p ^ (0 + 1):) := by simpa using hζ
  /-
    p : PNat
    K : Type u
    L : Type v
    ζ : L
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension (Singleton.singleton p) K L
    hp : Fact (Nat.Prime ↑p)
    hζ : IsPrimitiveRoot ζ ↑p
    hirr : Irreducible (Polynomial.cyclotomic (↑p) K)
    hodd : Ne p 2
    this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
    hζ' : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd 0 1))
    ⊢ Eq (Algebra.discr K ⇑(IsPrimitiveRoot.powerBasis K hζ).basis) (HMul.hMul (HP …
  -/
  convert discr_prime_pow_ne_two hζ' (by simpa [hirr]) (by simp [hodd]) using 2
    /-
      case h.e'_3.h.e'_5
      p : PNat
      K : Type u
      L : Type v
      ζ : L
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsCyclotomicExtension (Singleton.singleton p) K L
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot ζ ↑p
      hirr : Irreducible (Polynomial.cyclotomic (↑p) K)
      hodd : Ne p 2
      this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
      hζ' : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd 0 1))
      ⊢ Eq (HPow.hPow (-1) (HDiv.hDiv (HSub.hSub (↑p) 1) 2)) (HPow.hPow (-1) (HDiv.h …
    -/
  · rw [zero_add, pow_one, totient_prime hp.out]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_6
      p : PNat
      K : Type u
      L : Type v
      ζ : L
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : IsCyclotomicExtension (Singleton.singleton p) K L
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot ζ ↑p
      hirr : Irreducible (Polynomial.cyclotomic (↑p) K)
      hodd : Ne p 2
      this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
      hζ' : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd 0 1))
      ⊢ Eq (HPow.hPow (↑↑p) (HSub.hSub (↑p) 2)) (HPow.hPow (↑↑p) (HMul.hMul (HPow.hP …
    -/
  · rw [_root_.pow_zero, one_mul, zero_add, mul_one, Nat.sub_sub]
    /-
      🎉 no goals
    -/


