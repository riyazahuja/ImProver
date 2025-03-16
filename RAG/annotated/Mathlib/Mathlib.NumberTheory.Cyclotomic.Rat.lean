/-- The discriminant of the power basis given by `ζ - 1`. -/
theorem discr_prime_pow_ne_two' [IsCyclotomicExtension {p ^ (k + 1)} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1))) (hk : p ^ (k + 1) ≠ 2) :
    discr ℚ (hζ.subOnePowerBasis ℚ).basis =
      (-1) ^ ((p ^ (k + 1) : ℕ).totient / 2) * p ^ ((p : ℕ) ^ k * ((p - 1) * (k + 1) - 1)) := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    ⊢ Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis) (HMu …
  -/
  rw [← discr_prime_pow_ne_two hζ (cyclotomic.irreducible_rat (p ^ (k + 1)).pos) hk]
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hk : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    ⊢ Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis) (Alg …
  -/
  exact hζ.discr_zeta_eq_discr_zeta_sub_one.symm
  /-
    🎉 no goals
  -/


theorem discr_odd_prime' [IsCyclotomicExtension {p} ℚ K] (hζ : IsPrimitiveRoot ζ p) (hodd : p ≠ 2) :
    discr ℚ (hζ.subOnePowerBasis ℚ).basis = (-1) ^ (((p : ℕ) - 1) / 2) * p ^ ((p : ℕ) - 2) := by
  /-
    p : PNat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    hodd : Ne p 2
    ⊢ Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis) (HMu …
  -/
  rw [← discr_odd_prime hζ (cyclotomic.irreducible_rat hp.out.pos) hodd]
  /-
    p : PNat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    hodd : Ne p 2
    ⊢ Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis) (Alg …
  -/
  exact hζ.discr_zeta_eq_discr_zeta_sub_one.symm
  /-
    🎉 no goals
  -/


/-- The discriminant of the power basis given by `ζ - 1`. Beware that in the cases `p ^ k = 1` and
`p ^ k = 2` the formula uses `1 / 2 = 0` and `0 - 1 = 0`. It is useful only to have a uniform
result. See also `IsCyclotomicExtension.Rat.discr_prime_pow_eq_unit_mul_pow'`. -/
theorem discr_prime_pow' [IsCyclotomicExtension {p ^ k} ℚ K] (hζ : IsPrimitiveRoot ζ ↑(p ^ k)) :
    discr ℚ (hζ.subOnePowerBasis ℚ).basis =
      (-1) ^ ((p ^ k : ℕ).totient / 2) * p ^ ((p : ℕ) ^ (k - 1) * ((p - 1) * k - 1)) := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    ⊢ Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis) (HMu …
  -/
  rw [← discr_prime_pow hζ (cyclotomic.irreducible_rat (p ^ k).pos)]
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    ⊢ Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis) (Alg …
  -/
  exact hζ.discr_zeta_eq_discr_zeta_sub_one.symm
  /-
    🎉 no goals
  -/


/-- If `p` is a prime and `IsCyclotomicExtension {p ^ k} K L`, then there are `u : ℤˣ` and
`n : ℕ` such that the discriminant of the power basis given by `ζ - 1` is `u * p ^ n`. Often this is
enough and less cumbersome to use than `IsCyclotomicExtension.Rat.discr_prime_pow'`. -/
theorem discr_prime_pow_eq_unit_mul_pow' [IsCyclotomicExtension {p ^ k} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ k)) :
    ∃ (u : ℤˣ) (n : ℕ), discr ℚ (hζ.subOnePowerBasis ℚ).basis = u * p ^ n := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    ⊢ Exists fun u => Exists fun n => Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subO …
  -/
  rw [hζ.discr_zeta_eq_discr_zeta_sub_one.symm]
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    ⊢ Exists fun u => Exists fun n => Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.powe …
  -/
  exact discr_prime_pow_eq_unit_mul_pow hζ (cyclotomic.irreducible_rat (p ^ k).pos)
  /-
    🎉 no goals
  -/


/-- If `K` is a `p ^ k`-th cyclotomic extension of `ℚ`, then `(adjoin ℤ {ζ})` is the
integral closure of `ℤ` in `K`. -/
theorem isIntegralClosure_adjoin_singleton_of_prime_pow [hcycl : IsCyclotomicExtension {p ^ k} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ k)) : IsIntegralClosure (adjoin ℤ ({ζ} : Set K)) ℤ K := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    ⊢ IsIntegralClosure (Subtype fun x => Membership.mem (Algebra.adjoin Int (Sing …
  -/
  refine ⟨Subtype.val_injective, @fun x => ⟨fun h => ⟨⟨x, ?_⟩, rfl⟩, ?_⟩⟩
  /-
    case refine_1
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    x : K
    h : IsIntegral Int x
    ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ζ)) x
  -/
  swap
    /-
      case refine_2
      p : PNat
      k : Nat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
      x : K
      ⊢ (Exists fun y => Eq ((algebraMap (Subtype fun x => Membership.mem (Algebra.a …
    -/
  · rintro ⟨y, rfl⟩
    exact
      IsIntegral.algebraMap
        ((le_integralClosure_iff_isIntegral.1
          (adjoin_le_integralClosure (hζ.isIntegral (p ^ k).pos))).isIntegral _)
  /-
    case refine_1
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    x : K
    h : IsIntegral Int x
    ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ζ)) x
  -/
  let B := hζ.subOnePowerBasis ℚ
  /-
    case refine_1
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    x : K
    h : IsIntegral Int x
    B : PowerBasis Rat K := IsPrimitiveRoot.subOnePowerBasis Rat hζ
    ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ζ)) x
  -/
  have hint : IsIntegral ℤ B.gen := (hζ.isIntegral (p ^ k).pos).sub isIntegral_one
-- Porting note: the following `haveI` was not needed because the locale `cyclotomic` set it
-- as instances.
  /-
    case refine_1
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    x : K
    h : IsIntegral Int x
    B : PowerBasis Rat K := IsPrimitiveRoot.subOnePowerBasis Rat hζ
    hint : IsIntegral Int B.gen
    ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ζ)) x
  -/
  letI := IsCyclotomicExtension.finiteDimensional {p ^ k} ℚ K
  /-
    case refine_1
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    x : K
    h : IsIntegral Int x
    B : PowerBasis Rat K := IsPrimitiveRoot.subOnePowerBasis Rat hζ
    hint : IsIntegral Int B.gen
    this : FiniteDimensional Rat K := IsCyclotomicExtension.finiteDimensional (Sin …
    ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ζ)) x
  -/
  have H := discr_mul_isIntegral_mem_adjoin ℚ hint h
  /-
    case refine_1
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    x : K
    h : IsIntegral Int x
    B : PowerBasis Rat K := IsPrimitiveRoot.subOnePowerBasis Rat hζ
    hint : IsIntegral Int B.gen
    this : FiniteDimensional Rat K := IsCyclotomicExtension.finiteDimensional (Sin …
    H : Membership.mem (Algebra.adjoin Int (Singleton.singleton B.gen)) (HSMul.hSM …
    ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ζ)) x
  -/
  obtain ⟨u, n, hun⟩ := discr_prime_pow_eq_unit_mul_pow' hζ
  /-
    case refine_1.intro.intro
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    x : K
    h : IsIntegral Int x
    B : PowerBasis Rat K := IsPrimitiveRoot.subOnePowerBasis Rat hζ
    hint : IsIntegral Int B.gen
    this : FiniteDimensional Rat K := IsCyclotomicExtension.finiteDimensional (Sin …
    H : Membership.mem (Algebra.adjoin Int (Singleton.singleton B.gen)) (HSMul.hSM …
    u : Units Int
    n : Nat
    hun : Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis)  …
    ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ζ)) x
  -/
  rw [hun] at H
  /-
    case refine_1.intro.intro
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    x : K
    h : IsIntegral Int x
    B : PowerBasis Rat K := IsPrimitiveRoot.subOnePowerBasis Rat hζ
    hint : IsIntegral Int B.gen
    this : FiniteDimensional Rat K := IsCyclotomicExtension.finiteDimensional (Sin …
    u : Units Int
    n : Nat
    H : Membership.mem (Algebra.adjoin Int (Singleton.singleton B.gen)) (HSMul.hSM …
    hun : Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis)  …
    ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ζ)) x
  -/
  replace H := Subalgebra.smul_mem _ H u.inv
-- Porting note: the proof is slightly different because of coercions.
  rw [← smul_assoc, ← smul_mul_assoc, Units.inv_eq_val_inv, zsmul_eq_mul, ← Int.cast_mul,
    Units.inv_mul, Int.cast_one, one_mul, smul_def, map_pow] at H
  /-
    case refine_1.intro.intro
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    x : K
    h : IsIntegral Int x
    B : PowerBasis Rat K := IsPrimitiveRoot.subOnePowerBasis Rat hζ
    hint : IsIntegral Int B.gen
    this : FiniteDimensional Rat K := IsCyclotomicExtension.finiteDimensional (Sin …
    u : Units Int
    n : Nat
    hun : Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis)  …
    H : Membership.mem (Algebra.adjoin Int (Singleton.singleton B.gen)) (HMul.hMul …
    ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ζ)) x
  -/
  cases k
    /-
      case refine_1.intro.intro.zero
      p : PNat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      x : K
      h : IsIntegral Int x
      u : Units Int
      n : Nat
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 0)) Rat K
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p 0)
      B : PowerBasis Rat K := IsPrimitiveRoot.subOnePowerBasis Rat hζ
      hint : IsIntegral Int B.gen
      this : FiniteDimensional Rat K := IsCyclotomicExtension.finiteDimensional (Sin …
      hun : Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis)  …
      H : Membership.mem (Algebra.adjoin Int (Singleton.singleton B.gen)) (HMul.hMul …
      ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ζ)) x
    -/
  · haveI : IsCyclotomicExtension {1} ℚ K := by simpa using hcycl
    have : x ∈ (⊥ : Subalgebra ℚ K) := by
      rw [singleton_one ℚ K]
      exact mem_top
    /-
      case refine_1.intro.intro.zero
      p : PNat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      x : K
      h : IsIntegral Int x
      u : Units Int
      n : Nat
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 0)) Rat K
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p 0)
      B : PowerBasis Rat K := IsPrimitiveRoot.subOnePowerBasis Rat hζ
      hint : IsIntegral Int B.gen
      this✝¹ : FiniteDimensional Rat K := IsCyclotomicExtension.finiteDimensional (S …
      hun : Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis)  …
      H : Membership.mem (Algebra.adjoin Int (Singleton.singleton B.gen)) (HMul.hMul …
      this✝ : IsCyclotomicExtension (Singleton.singleton 1) Rat K
      this : Membership.mem Bot.bot x
      ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ζ)) x
    -/
    obtain ⟨y, rfl⟩ := mem_bot.1 this
    /-
      case refine_1.intro.intro.zero.intro
      p : PNat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      u : Units Int
      n : Nat
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 0)) Rat K
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p 0)
      B : PowerBasis Rat K := IsPrimitiveRoot.subOnePowerBasis Rat hζ
      hint : IsIntegral Int B.gen
      this✝¹ : FiniteDimensional Rat K := IsCyclotomicExtension.finiteDimensional (S …
      hun : Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis)  …
      this✝ : IsCyclotomicExtension (Singleton.singleton 1) Rat K
      y : Rat
      h : IsIntegral Int ((algebraMap Rat K) y)
      H : Membership.mem (Algebra.adjoin Int (Singleton.singleton B.gen)) (HMul.hMul …
      this : Membership.mem Bot.bot ((algebraMap Rat K) y)
      ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ζ)) ((algebraMap Rat …
    -/
    replace h := (isIntegral_algebraMap_iff (algebraMap ℚ K).injective).1 h
    /-
      case refine_1.intro.intro.zero.intro
      p : PNat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      u : Units Int
      n : Nat
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 0)) Rat K
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p 0)
      B : PowerBasis Rat K := IsPrimitiveRoot.subOnePowerBasis Rat hζ
      hint : IsIntegral Int B.gen
      this✝¹ : FiniteDimensional Rat K := IsCyclotomicExtension.finiteDimensional (S …
      hun : Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis)  …
      this✝ : IsCyclotomicExtension (Singleton.singleton 1) Rat K
      y : Rat
      H : Membership.mem (Algebra.adjoin Int (Singleton.singleton B.gen)) (HMul.hMul …
      this : Membership.mem Bot.bot ((algebraMap Rat K) y)
      h : IsIntegral Int y
      ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ζ)) ((algebraMap Rat …
    -/
    obtain ⟨z, hz⟩ := IsIntegrallyClosed.isIntegral_iff.1 h
    /-
      case refine_1.intro.intro.zero.intro.intro
      p : PNat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      u : Units Int
      n : Nat
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 0)) Rat K
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p 0)
      B : PowerBasis Rat K := IsPrimitiveRoot.subOnePowerBasis Rat hζ
      hint : IsIntegral Int B.gen
      this✝¹ : FiniteDimensional Rat K := IsCyclotomicExtension.finiteDimensional (S …
      hun : Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis)  …
      this✝ : IsCyclotomicExtension (Singleton.singleton 1) Rat K
      y : Rat
      H : Membership.mem (Algebra.adjoin Int (Singleton.singleton B.gen)) (HMul.hMul …
      this : Membership.mem Bot.bot ((algebraMap Rat K) y)
      h : IsIntegral Int y
      z : Int
      hz : Eq ((algebraMap Int Rat) z) y
      ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ζ)) ((algebraMap Rat …
    -/
    rw [← hz, ← IsScalarTower.algebraMap_apply]
    /-
      case refine_1.intro.intro.zero.intro.intro
      p : PNat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      u : Units Int
      n : Nat
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 0)) Rat K
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p 0)
      B : PowerBasis Rat K := IsPrimitiveRoot.subOnePowerBasis Rat hζ
      hint : IsIntegral Int B.gen
      this✝¹ : FiniteDimensional Rat K := IsCyclotomicExtension.finiteDimensional (S …
      hun : Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis)  …
      this✝ : IsCyclotomicExtension (Singleton.singleton 1) Rat K
      y : Rat
      H : Membership.mem (Algebra.adjoin Int (Singleton.singleton B.gen)) (HMul.hMul …
      this : Membership.mem Bot.bot ((algebraMap Rat K) y)
      h : IsIntegral Int y
      z : Int
      hz : Eq ((algebraMap Int Rat) z) y
      ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ζ)) ((algebraMap Int …
    -/
    exact Subalgebra.algebraMap_mem _ _
    /-
      🎉 no goals
    -/
  · have hmin : (minpoly ℤ B.gen).IsEisensteinAt (Submodule.span ℤ {((p : ℕ) : ℤ)}) := by
      have h₁ := minpoly.isIntegrallyClosed_eq_field_fractions' ℚ hint
      have h₂ := hζ.minpoly_sub_one_eq_cyclotomic_comp (cyclotomic.irreducible_rat (p ^ _).pos)
      rw [IsPrimitiveRoot.subOnePowerBasis_gen] at h₁
      rw [h₁, ← map_cyclotomic_int, show Int.castRingHom ℚ = algebraMap ℤ ℚ by rfl,
        show X + 1 = map (algebraMap ℤ ℚ) (X + 1) by simp, ← map_comp] at h₂
      rw [IsPrimitiveRoot.subOnePowerBasis_gen,
        map_injective (algebraMap ℤ ℚ) (algebraMap ℤ ℚ).injective_int h₂]
      exact cyclotomic_prime_pow_comp_X_add_one_isEisensteinAt p _
    refine
      adjoin_le ?_
        (mem_adjoin_of_smul_prime_pow_smul_of_minpoly_isEisensteinAt (n := n)
          (Nat.prime_iff_prime_int.1 hp.out) hint h (by simpa using H) hmin)
    /-
      case refine_1.intro.intro.succ
      p : PNat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      x : K
      h : IsIntegral Int x
      u : Units Int
      n n✝ : Nat
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd n✝  …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd n✝ 1))
      B : PowerBasis Rat K := IsPrimitiveRoot.subOnePowerBasis Rat hζ
      hint : IsIntegral Int B.gen
      this : FiniteDimensional Rat K := IsCyclotomicExtension.finiteDimensional (Sin …
      hun : Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis)  …
      H : Membership.mem (Algebra.adjoin Int (Singleton.singleton B.gen)) (HMul.hMul …
      hmin : (minpoly Int B.gen).IsEisensteinAt (Submodule.span Int (Singleton.singl …
      ⊢ HasSubset.Subset (Singleton.singleton B.gen) ↑(Algebra.adjoin Int (Singleton …
    -/
    simp only [Set.singleton_subset_iff, SetLike.mem_coe]
    /-
      case refine_1.intro.intro.succ
      p : PNat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      x : K
      h : IsIntegral Int x
      u : Units Int
      n n✝ : Nat
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd n✝  …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd n✝ 1))
      B : PowerBasis Rat K := IsPrimitiveRoot.subOnePowerBasis Rat hζ
      hint : IsIntegral Int B.gen
      this : FiniteDimensional Rat K := IsCyclotomicExtension.finiteDimensional (Sin …
      hun : Eq (Algebra.discr Rat ⇑(IsPrimitiveRoot.subOnePowerBasis Rat hζ).basis)  …
      H : Membership.mem (Algebra.adjoin Int (Singleton.singleton B.gen)) (HMul.hMul …
      hmin : (minpoly Int B.gen).IsEisensteinAt (Submodule.span Int (Singleton.singl …
      ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ζ)) B.gen
    -/
    exact Subalgebra.sub_mem _ (self_mem_adjoin_singleton ℤ _) (Subalgebra.one_mem _)
    /-
      🎉 no goals
    -/


theorem isIntegralClosure_adjoin_singleton_of_prime [hcycl : IsCyclotomicExtension {p} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑p) : IsIntegralClosure (adjoin ℤ ({ζ} : Set K)) ℤ K := by
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    ⊢ IsIntegralClosure (Subtype fun x => Membership.mem (Algebra.adjoin Int (Sing …
  -/
  rw [← pow_one p] at hζ hcycl
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 1)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p 1)
    ⊢ IsIntegralClosure (Subtype fun x => Membership.mem (Algebra.adjoin Int (Sing …
  -/
  exact isIntegralClosure_adjoin_singleton_of_prime_pow hζ
  /-
    🎉 no goals
  -/


/-- The integral closure of `ℤ` inside `CyclotomicField (p ^ k) ℚ` is
`CyclotomicRing (p ^ k) ℤ ℚ`. -/
theorem cyclotomicRing_isIntegralClosure_of_prime_pow :
    IsIntegralClosure (CyclotomicRing (p ^ k) ℤ ℚ) ℤ (CyclotomicField (p ^ k) ℚ) := by
  /-
    p : PNat
    k : Nat
    hp : Fact (Nat.Prime ↑p)
    ⊢ IsIntegralClosure (CyclotomicRing (HPow.hPow p k) Int Rat) Int (CyclotomicFi …
  -/
  have hζ := zeta_spec (p ^ k) ℚ (CyclotomicField (p ^ k) ℚ)
  /-
    p : PNat
    k : Nat
    hp : Fact (Nat.Prime ↑p)
    hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat (Cyclotom …
    ⊢ IsIntegralClosure (CyclotomicRing (HPow.hPow p k) Int Rat) Int (CyclotomicFi …
  -/
  refine ⟨IsFractionRing.injective _ _, @fun x => ⟨fun h => ⟨⟨x, ?_⟩, rfl⟩, ?_⟩⟩
-- Porting note: having `.isIntegral_iff` inside the definition of `this` causes an error.
    /-
      case refine_1
      p : PNat
      k : Nat
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat (Cyclotom …
      x : CyclotomicField (HPow.hPow p k) Rat
      h : IsIntegral Int x
      ⊢ Membership.mem (Algebra.adjoin Int (setOf fun b => Eq (HPow.hPow b ↑(HPow.hP …
    -/
  · have := isIntegralClosure_adjoin_singleton_of_prime_pow hζ
    /-
      case refine_1
      p : PNat
      k : Nat
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat (Cyclotom …
      x : CyclotomicField (HPow.hPow p k) Rat
      h : IsIntegral Int x
      this : IsIntegralClosure (Subtype fun x => Membership.mem (Algebra.adjoin Int  …
      ⊢ Membership.mem (Algebra.adjoin Int (setOf fun b => Eq (HPow.hPow b ↑(HPow.hP …
    -/
    obtain ⟨y, rfl⟩ := this.isIntegral_iff.1 h
    /-
      case refine_1.intro
      p : PNat
      k : Nat
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat (Cyclotom …
      this : IsIntegralClosure (Subtype fun x => Membership.mem (Algebra.adjoin Int  …
      y : Subtype fun x => Membership.mem (Algebra.adjoin Int (Singleton.singleton ( …
      h : IsIntegral Int ((algebraMap (Subtype fun x => Membership.mem (Algebra.adjo …
      ⊢ Membership.mem (Algebra.adjoin Int (setOf fun b => Eq (HPow.hPow b ↑(HPow.hP …
    -/
    refine adjoin_mono ?_ y.2
    /-
      case refine_1.intro
      p : PNat
      k : Nat
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat (Cyclotom …
      this : IsIntegralClosure (Subtype fun x => Membership.mem (Algebra.adjoin Int  …
      y : Subtype fun x => Membership.mem (Algebra.adjoin Int (Singleton.singleton ( …
      h : IsIntegral Int ((algebraMap (Subtype fun x => Membership.mem (Algebra.adjo …
      ⊢ HasSubset.Subset (Singleton.singleton (IsCyclotomicExtension.zeta (HPow.hPow …
    -/
    simp only [PNat.pow_coe, Set.singleton_subset_iff, Set.mem_setOf_eq]
    /-
      case refine_1.intro
      p : PNat
      k : Nat
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat (Cyclotom …
      this : IsIntegralClosure (Subtype fun x => Membership.mem (Algebra.adjoin Int  …
      y : Subtype fun x => Membership.mem (Algebra.adjoin Int (Singleton.singleton ( …
      h : IsIntegral Int ((algebraMap (Subtype fun x => Membership.mem (Algebra.adjo …
      ⊢ Eq (HPow.hPow (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat (CyclotomicFie …
    -/
    exact hζ.pow_eq_one
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      p : PNat
      k : Nat
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat (Cyclotom …
      x : CyclotomicField (HPow.hPow p k) Rat
      ⊢ (Exists fun y => Eq ((algebraMap (CyclotomicRing (HPow.hPow p k) Int Rat) (C …
    -/
  · rintro ⟨y, rfl⟩
    /-
      case refine_2.intro
      p : PNat
      k : Nat
      hp : Fact (Nat.Prime ↑p)
      hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat (Cyclotom …
      y : CyclotomicRing (HPow.hPow p k) Int Rat
      ⊢ IsIntegral Int ((algebraMap (CyclotomicRing (HPow.hPow p k) Int Rat) (Cyclot …
    -/
    exact IsIntegral.algebraMap ((IsCyclotomicExtension.integral {p ^ k} ℤ _).isIntegral _)
    /-
      🎉 no goals
    -/


theorem cyclotomicRing_isIntegralClosure_of_prime :
    IsIntegralClosure (CyclotomicRing p ℤ ℚ) ℤ (CyclotomicField p ℚ) := by
  /-
    p : PNat
    hp : Fact (Nat.Prime ↑p)
    ⊢ IsIntegralClosure (CyclotomicRing p Int Rat) Int (CyclotomicField p Rat)
  -/
  rw [← pow_one p]
  /-
    p : PNat
    hp : Fact (Nat.Prime ↑p)
    ⊢ IsIntegralClosure (CyclotomicRing (HPow.hPow p 1) Int Rat) Int (CyclotomicFi …
  -/
  exact cyclotomicRing_isIntegralClosure_of_prime_pow
  /-
    🎉 no goals
  -/


/-- The algebra isomorphism `adjoin ℤ {ζ} ≃ₐ[ℤ] (𝓞 K)`, where `ζ` is a primitive `p ^ k`-th root of
unity and `K` is a `p ^ k`-th cyclotomic extension of `ℚ`. -/
@[simps!]
noncomputable def _root_.IsPrimitiveRoot.adjoinEquivRingOfIntegers
    [IsCyclotomicExtension {p ^ k} ℚ K] (hζ : IsPrimitiveRoot ζ ↑(p ^ k)) :
    adjoin ℤ ({ζ} : Set K) ≃ₐ[ℤ] 𝓞 K :=
  let _ := isIntegralClosure_adjoin_singleton_of_prime_pow hζ
  IsIntegralClosure.equiv ℤ (adjoin ℤ ({ζ} : Set K)) K (𝓞 K)


/-- The ring of integers of a `p ^ k`-th cyclotomic extension of `ℚ` is a cyclotomic extension. -/
instance IsCyclotomicExtension.ringOfIntegers [IsCyclotomicExtension {p ^ k} ℚ K] :
    IsCyclotomicExtension {p ^ k} ℤ (𝓞 K) :=
  let _ := (zeta_spec (p ^ k) ℚ K).adjoin_isCyclotomicExtension ℤ
  IsCyclotomicExtension.equiv _ ℤ _ (zeta_spec (p ^ k) ℚ K).adjoinEquivRingOfIntegers


/-- The integral `PowerBasis` of `𝓞 K` given by a primitive root of unity, where `K` is a `p ^ k`
cyclotomic extension of `ℚ`. -/
noncomputable def integralPowerBasis [IsCyclotomicExtension {p ^ k} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ k)) : PowerBasis ℤ (𝓞 K) :=
  (Algebra.adjoin.powerBasis' (hζ.isIntegral (p ^ k).pos)).map hζ.adjoinEquivRingOfIntegers


/-- Abbreviation to see a primitive root of unity as a member of the ring of integers. -/
abbrev toInteger {k : ℕ+} (hζ : IsPrimitiveRoot ζ k) : 𝓞 K := ⟨ζ, hζ.isIntegral k.pos⟩


lemma coe_toInteger {k : ℕ+} (hζ : IsPrimitiveRoot ζ k) : hζ.toInteger.1 = ζ := rfl


/-- `𝓞 K ⧸ Ideal.span {ζ - 1}` is finite. -/
lemma finite_quotient_toInteger_sub_one [NumberField K] {k : ℕ+} (hk : 1 < k)
    (hζ : IsPrimitiveRoot ζ k) : Finite (𝓞 K ⧸ Ideal.span {hζ.toInteger - 1}) := by
  /-
    K : Type u
    inst✝¹ : Field K
    ζ : K
    inst✝ : NumberField K
    k : PNat
    hk : LT.lt 1 k
    hζ : IsPrimitiveRoot ζ ↑k
    ⊢ Finite (HasQuotient.Quotient (NumberField.RingOfIntegers K) (Ideal.span (Sin …
  -/
  refine (finite_iff_nonempty_fintype _).2 ⟨?_⟩
  /-
    K : Type u
    inst✝¹ : Field K
    ζ : K
    inst✝ : NumberField K
    k : PNat
    hk : LT.lt 1 k
    hζ : IsPrimitiveRoot ζ ↑k
    ⊢ Fintype (HasQuotient.Quotient (NumberField.RingOfIntegers K) (Ideal.span (Si …
  -/
  refine Ideal.fintypeQuotientOfFreeOfNeBot _ (fun h ↦ ?_)
  /-
    K : Type u
    inst✝¹ : Field K
    ζ : K
    inst✝ : NumberField K
    k : PNat
    hk : LT.lt 1 k
    hζ : IsPrimitiveRoot ζ ↑k
    h : Eq (Ideal.span (Singleton.singleton (HSub.hSub hζ.toInteger 1))) Bot.bot
    ⊢ False
  -/
  simp only [Ideal.span_singleton_eq_bot, sub_eq_zero, ← Subtype.coe_inj] at h
  /-
    K : Type u
    inst✝¹ : Field K
    ζ : K
    inst✝ : NumberField K
    k : PNat
    hk : LT.lt 1 k
    hζ : IsPrimitiveRoot ζ ↑k
    h : Eq hζ.toInteger 1
    ⊢ False
  -/
  exact hζ.ne_one hk (RingOfIntegers.ext_iff.1 h)
  /-
    🎉 no goals
  -/


/-- We have that `𝓞 K ⧸ Ideal.span {ζ - 1}` has cardinality equal to the norm of `ζ - 1`.

See the results below to compute this norm in various cases. -/
lemma card_quotient_toInteger_sub_one [NumberField K] {k : ℕ+} (hk : 1 < k)
    (hζ : IsPrimitiveRoot ζ k) :
    Nat.card (𝓞 K ⧸ Ideal.span {hζ.toInteger - 1}) =
      (Algebra.norm ℤ (hζ.toInteger - 1)).natAbs := by
  /-
    K : Type u
    inst✝¹ : Field K
    ζ : K
    inst✝ : NumberField K
    k : PNat
    hk : LT.lt 1 k
    hζ : IsPrimitiveRoot ζ ↑k
    ⊢ Eq (Nat.card (HasQuotient.Quotient (NumberField.RingOfIntegers K) (Ideal.spa …
  -/
  have := hζ.finite_quotient_toInteger_sub_one hk
  /-
    K : Type u
    inst✝¹ : Field K
    ζ : K
    inst✝ : NumberField K
    k : PNat
    hk : LT.lt 1 k
    hζ : IsPrimitiveRoot ζ ↑k
    this : Finite (HasQuotient.Quotient (NumberField.RingOfIntegers K) (Ideal.span …
    ⊢ Eq (Nat.card (HasQuotient.Quotient (NumberField.RingOfIntegers K) (Ideal.spa …
  -/
  let _ := Fintype.ofFinite (𝓞 K ⧸ Ideal.span {hζ.toInteger - 1})
  /-
    K : Type u
    inst✝¹ : Field K
    ζ : K
    inst✝ : NumberField K
    k : PNat
    hk : LT.lt 1 k
    hζ : IsPrimitiveRoot ζ ↑k
    this : Finite (HasQuotient.Quotient (NumberField.RingOfIntegers K) (Ideal.span …
    x✝ : Fintype (HasQuotient.Quotient (NumberField.RingOfIntegers K) (Ideal.span  …
    ⊢ Eq (Nat.card (HasQuotient.Quotient (NumberField.RingOfIntegers K) (Ideal.spa …
  -/
  rw [← Submodule.cardQuot_apply, ← Ideal.absNorm_apply, Ideal.absNorm_span_singleton]
  /-
    🎉 no goals
  -/


lemma toInteger_isPrimitiveRoot {k : ℕ+} (hζ : IsPrimitiveRoot ζ k) :
    IsPrimitiveRoot hζ.toInteger k :=
                                          /-
                                            K : Type u
                                            inst✝ : Field K
                                            ζ : K
                                            k : PNat
                                            hζ : IsPrimitiveRoot ζ ↑k
                                            ⊢ IsPrimitiveRoot ((algebraMap (NumberField.RingOfIntegers K) K) hζ.toInteger) …
                                          -/
  IsPrimitiveRoot.of_map_of_injective (by exact hζ) RingOfIntegers.coe_injective
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
theorem integralPowerBasis_gen [hcycl : IsCyclotomicExtension {p ^ k} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ k)) :
    hζ.integralPowerBasis.gen = hζ.toInteger :=
  Subtype.ext <| show algebraMap _ K hζ.integralPowerBasis.gen = _ by
    /-
      p : PNat
      k : Nat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
      ⊢ Eq ((algebraMap (NumberField.RingOfIntegers K) K) hζ.integralPowerBasis.gen) …
    -/
    rw [integralPowerBasis, PowerBasis.map_gen, adjoin.powerBasis'_gen]
    /-
      p : PNat
      k : Nat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
      ⊢ Eq ((algebraMap (NumberField.RingOfIntegers K) K) (hζ.adjoinEquivRingOfInteg …
    -/
    simp only [adjoinEquivRingOfIntegers_apply, IsIntegralClosure.algebraMap_lift]
    /-
      p : PNat
      k : Nat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
      ⊢ Eq ((algebraMap (Subtype fun x => Membership.mem (Algebra.adjoin Int (Single …
    -/
    rfl
    /-
      🎉 no goals
    -/


set_option linter.unusedVariables false in
@[simp]
theorem integralPowerBasis_dim [hcycl : IsCyclotomicExtension {p ^ k} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ k)) : hζ.integralPowerBasis.dim = φ (p ^ k) := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    ⊢ Eq hζ.integralPowerBasis.dim (HPow.hPow (↑p) k).totient
  -/
  simp [integralPowerBasis, ← cyclotomic_eq_minpoly hζ, natDegree_cyclotomic]
  /-
    🎉 no goals
  -/


/-- The algebra isomorphism `adjoin ℤ {ζ} ≃ₐ[ℤ] (𝓞 K)`, where `ζ` is a primitive `p`-th root of
unity and `K` is a `p`-th cyclotomic extension of `ℚ`. -/
@[simps!]
noncomputable def _root_.IsPrimitiveRoot.adjoinEquivRingOfIntegers'
    [hcycl : IsCyclotomicExtension {p} ℚ K] (hζ : IsPrimitiveRoot ζ p) :
    adjoin ℤ ({ζ} : Set K) ≃ₐ[ℤ] 𝓞 K :=
                                                 /-
                                                   p : PNat
                                                   k : Nat
                                                   K : Type u
                                                   inst✝¹ : Field K
                                                   ζ : K
                                                   hp : Fact (Nat.Prime ↑p)
                                                   inst✝ : CharZero K
                                                   hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
                                                   hζ : IsPrimitiveRoot ζ ↑p
                                                   ⊢ IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 1)) Rat K
                                                 -/
  have : IsCyclotomicExtension {p ^ 1} ℚ K := by convert hcycl; rw [pow_one]
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                           /-
                                                             p : PNat
                                                             k : Nat
                                                             K : Type u
                                                             inst✝¹ : Field K
                                                             ζ : K
                                                             hp : Fact (Nat.Prime ↑p)
                                                             inst✝ : CharZero K
                                                             hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
                                                             hζ : IsPrimitiveRoot ζ ↑p
                                                             this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 1)) Rat K
                                                             ⊢ IsPrimitiveRoot ζ ↑(HPow.hPow p 1)
                                                           -/
  adjoinEquivRingOfIntegers (p := p) (k := 1) (ζ := ζ) (by rwa [pow_one])
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- The ring of integers of a `p`-th cyclotomic extension of `ℚ` is a cyclotomic extension. -/
instance _root_.IsCyclotomicExtension.ring_of_integers' [IsCyclotomicExtension {p} ℚ K] :
    IsCyclotomicExtension {p} ℤ (𝓞 K) :=
  let _ := (zeta_spec p ℚ K).adjoin_isCyclotomicExtension ℤ
  IsCyclotomicExtension.equiv _ ℤ _ (zeta_spec p ℚ K).adjoinEquivRingOfIntegers'


/-- The integral `PowerBasis` of `𝓞 K` given by a primitive root of unity, where `K` is a `p`-th
cyclotomic extension of `ℚ`. -/
noncomputable def integralPowerBasis' [hcycl : IsCyclotomicExtension {p} ℚ K]
    (hζ : IsPrimitiveRoot ζ p) : PowerBasis ℤ (𝓞 K) :=
                                                 /-
                                                   p : PNat
                                                   k : Nat
                                                   K : Type u
                                                   inst✝¹ : Field K
                                                   ζ : K
                                                   hp : Fact (Nat.Prime ↑p)
                                                   inst✝ : CharZero K
                                                   hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
                                                   hζ : IsPrimitiveRoot ζ ↑p
                                                   ⊢ IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 1)) Rat K
                                                 -/
  have : IsCyclotomicExtension {p ^ 1} ℚ K := by convert hcycl; rw [pow_one]
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                    /-
                                                      p : PNat
                                                      k : Nat
                                                      K : Type u
                                                      inst✝¹ : Field K
                                                      ζ : K
                                                      hp : Fact (Nat.Prime ↑p)
                                                      inst✝ : CharZero K
                                                      hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
                                                      hζ : IsPrimitiveRoot ζ ↑p
                                                      this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 1)) Rat K
                                                      ⊢ IsPrimitiveRoot ζ ↑(HPow.hPow p 1)
                                                    -/
  integralPowerBasis (p := p) (k := 1) (ζ := ζ) (by rwa [pow_one])
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem integralPowerBasis'_gen [hcycl : IsCyclotomicExtension {p} ℚ K] (hζ : IsPrimitiveRoot ζ p) :
    hζ.integralPowerBasis'.gen = hζ.toInteger :=
                                      /-
                                        p : PNat
                                        K : Type u
                                        inst✝¹ : Field K
                                        ζ : K
                                        hp : Fact (Nat.Prime ↑p)
                                        inst✝ : CharZero K
                                        hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
                                        hζ : IsPrimitiveRoot ζ ↑p
                                        ⊢ IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 1)) Rat K
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
  integralPowerBasis_gen (hcycl := by rwa [pow_one]) (by rwa [pow_one])
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
theorem power_basis_int'_dim [hcycl : IsCyclotomicExtension {p} ℚ K] (hζ : IsPrimitiveRoot ζ p) :
    hζ.integralPowerBasis'.dim = φ p := by
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    ⊢ Eq hζ.integralPowerBasis'.dim (↑p).totient
  -/
  erw [integralPowerBasis_dim (hcycl := by rwa [pow_one]) (by rwa [pow_one]), pow_one]
  /-
    🎉 no goals
  -/



/-- The integral `PowerBasis` of `𝓞 K` given by `ζ - 1`, where `K` is a `p ^ k` cyclotomic
extension of `ℚ`. -/
noncomputable def subOneIntegralPowerBasis [IsCyclotomicExtension {p ^ k} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ k)) : PowerBasis ℤ (𝓞 K) :=
  PowerBasis.ofGenMemAdjoin' hζ.integralPowerBasis (RingOfIntegers.isIntegral _)
    (by
      /-
        p : PNat
        k : Nat
        K : Type u
        inst✝² : Field K
        ζ : K
        hp : Fact (Nat.Prime ↑p)
        inst✝¹ : CharZero K
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
        ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton ?m.134380)) hζ.integ …
      -/
      simp only [integralPowerBasis_gen, toInteger]
      convert Subalgebra.add_mem _ (self_mem_adjoin_singleton ℤ (⟨ζ - 1, _⟩ : 𝓞 K))
        (Subalgebra.one_mem _)
        /-
          case h.e'_5.h.h.e'_3
          p : PNat
          k : Nat
          K : Type u
          inst✝² : Field K
          ζ : K
          hp : Fact (Nat.Prime ↑p)
          inst✝¹ : CharZero K
          inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
          hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
          e_1✝ : Eq (NumberField.RingOfIntegers K) (Subtype fun x => Membership.mem (int …
          ⊢ Eq ζ ↑(HAdd.hAdd ⟨HSub.hSub ζ 1, ?m.140095⟩ 1)
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          p : PNat
          k : Nat
          K : Type u
          inst✝² : Field K
          ζ : K
          hp : Fact (Nat.Prime ↑p)
          inst✝¹ : CharZero K
          inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
          hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
          ⊢ Membership.mem (integralClosure Int K) (HSub.hSub ζ 1)
        -/
      · exact Subalgebra.sub_mem _ (hζ.isIntegral (by simp)) (Subalgebra.one_mem _))
        /-
          🎉 no goals
        -/


@[simp]
theorem subOneIntegralPowerBasis_gen [IsCyclotomicExtension {p ^ k} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ k)) :
    hζ.subOneIntegralPowerBasis.gen =
      ⟨ζ - 1, Subalgebra.sub_mem _ (hζ.isIntegral (p ^ k).pos) (Subalgebra.one_mem _)⟩ := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p k)
    ⊢ Eq hζ.subOneIntegralPowerBasis.gen ⟨HSub.hSub ζ 1, ⋯⟩
  -/
  simp [subOneIntegralPowerBasis]
  /-
    🎉 no goals
  -/


/-- The integral `PowerBasis` of `𝓞 K` given by `ζ - 1`, where `K` is a `p`-th cyclotomic
extension of `ℚ`. -/
noncomputable def subOneIntegralPowerBasis' [IsCyclotomicExtension {p} ℚ K]
    (hζ : IsPrimitiveRoot ζ p) : PowerBasis ℤ (𝓞 K) :=
                                                 /-
                                                   p : PNat
                                                   k : Nat
                                                   K : Type u
                                                   inst✝² : Field K
                                                   ζ : K
                                                   hp : Fact (Nat.Prime ↑p)
                                                   inst✝¹ : CharZero K
                                                   inst✝ : IsCyclotomicExtension (Singleton.singleton p) Rat K
                                                   hζ : IsPrimitiveRoot ζ ↑p
                                                   ⊢ IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 1)) Rat K
                                                 -/
  have : IsCyclotomicExtension {p ^ 1} ℚ K := by rwa [pow_one]
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                          /-
                                                            p : PNat
                                                            k : Nat
                                                            K : Type u
                                                            inst✝² : Field K
                                                            ζ : K
                                                            hp : Fact (Nat.Prime ↑p)
                                                            inst✝¹ : CharZero K
                                                            inst✝ : IsCyclotomicExtension (Singleton.singleton p) Rat K
                                                            hζ : IsPrimitiveRoot ζ ↑p
                                                            this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 1)) Rat K
                                                            ⊢ IsPrimitiveRoot ζ ↑(HPow.hPow p 1)
                                                          -/
  subOneIntegralPowerBasis (p := p) (k := 1) (ζ := ζ) (by rwa [pow_one])
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp, nolint unusedHavesSuffices]
theorem subOneIntegralPowerBasis'_gen [IsCyclotomicExtension {p} ℚ K]
    (hζ : IsPrimitiveRoot ζ p) :
    hζ.subOneIntegralPowerBasis'.gen = hζ.toInteger - 1 :=
  -- The `unusedHavesSuffices` linter incorrectly thinks this `have` is unnecessary.
                                                 /-
                                                   p : PNat
                                                   K : Type u
                                                   inst✝² : Field K
                                                   ζ : K
                                                   hp : Fact (Nat.Prime ↑p)
                                                   inst✝¹ : CharZero K
                                                   inst✝ : IsCyclotomicExtension (Singleton.singleton p) Rat K
                                                   hζ : IsPrimitiveRoot ζ ↑p
                                                   ⊢ IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 1)) Rat K
                                                 -/
  have : IsCyclotomicExtension {p ^ 1} ℚ K := by rwa [pow_one]
                                                 /-
                                                   🎉 no goals
                                                 -/
                                   /-
                                     p : PNat
                                     K : Type u
                                     inst✝² : Field K
                                     ζ : K
                                     hp : Fact (Nat.Prime ↑p)
                                     inst✝¹ : CharZero K
                                     inst✝ : IsCyclotomicExtension (Singleton.singleton p) Rat K
                                     hζ : IsPrimitiveRoot ζ ↑p
                                     this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p 1)) Rat K
                                     ⊢ IsPrimitiveRoot ζ ↑(HPow.hPow p 1)
                                   -/
  subOneIntegralPowerBasis_gen (by rwa [pow_one])
                                   /-
                                     🎉 no goals
                                   -/


/-- `ζ - 1` is prime if `p ≠ 2` and `ζ` is a primitive `p ^ (k + 1)`-th root of unity.
  See `zeta_sub_one_prime` for a general statement. -/
theorem zeta_sub_one_prime_of_ne_two [IsCyclotomicExtension {p ^ (k + 1)} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1))) (hodd : p ≠ 2) :
    Prime (hζ.toInteger - 1) := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hodd : Ne p 2
    ⊢ Prime (HSub.hSub hζ.toInteger 1)
  -/
  letI := IsCyclotomicExtension.numberField {p ^ (k + 1)} ℚ K
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hodd : Ne p 2
    this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
    ⊢ Prime (HSub.hSub hζ.toInteger 1)
  -/
  refine Ideal.prime_of_irreducible_absNorm_span (fun h ↦ ?_) ?_
    /-
      case refine_1
      p : PNat
      k : Nat
      K : Type u
      inst✝² : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hodd : Ne p 2
      this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
      h : Eq (HSub.hSub hζ.toInteger 1) 0
      ⊢ False
    -/
  · apply hζ.pow_ne_one_of_pos_of_lt zero_lt_one (one_lt_pow₀ hp.out.one_lt (by simp))
    /-
      case refine_1
      p : PNat
      k : Nat
      K : Type u
      inst✝² : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hodd : Ne p 2
      this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
      h : Eq (HSub.hSub hζ.toInteger 1) 0
      ⊢ Eq (HPow.hPow ζ 1) 1
    -/
    rw [sub_eq_zero] at h
    /-
      case refine_1
      p : PNat
      k : Nat
      K : Type u
      inst✝² : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hodd : Ne p 2
      this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
      h : Eq hζ.toInteger 1
      ⊢ Eq (HPow.hPow ζ 1) 1
    -/
    simpa using congrArg (algebraMap _ K) h
    /-
      🎉 no goals
    -/
  rw [Nat.irreducible_iff_prime, Ideal.absNorm_span_singleton, ← Nat.prime_iff,
    ← Int.prime_iff_natAbs_prime]
  /-
    case refine_2
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hodd : Ne p 2
    this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
    ⊢ Prime ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))
  -/
  convert Nat.prime_iff_prime_int.1 hp.out
  /-
    case h.e'_3
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hodd : Ne p 2
    this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
    ⊢ Eq ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1)) ↑↑p
  -/
  apply RingHom.injective_int (algebraMap ℤ ℚ)
  /-
    case h.e'_3.a
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hodd : Ne p 2
    this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
    ⊢ Eq ((algebraMap Int Rat) ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))) (( …
  -/
  rw [← Algebra.norm_localization (Sₘ := K) ℤ (nonZeroDivisors ℤ)]
  simp only [PNat.pow_coe, id.map_eq_id, RingHomCompTriple.comp_eq, RingHom.coe_coe,
    Subalgebra.coe_val, algebraMap_int_eq, map_natCast]
  /-
    case h.e'_3.a
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hodd : Ne p 2
    this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
    ⊢ Eq ((Algebra.norm Rat) ((algebraMap (NumberField.RingOfIntegers K) K) (HSub. …
  -/
  exact hζ.norm_sub_one_of_prime_ne_two (Polynomial.cyclotomic.irreducible_rat (PNat.pos _)) hodd
  /-
    🎉 no goals
  -/


/-- `ζ - 1` is prime if `ζ` is a primitive `2 ^ (k + 1)`-th root of unity.
  See `zeta_sub_one_prime` for a general statement. -/
theorem zeta_sub_one_prime_of_two_pow [IsCyclotomicExtension {(2 : ℕ+) ^ (k + 1)} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑((2 : ℕ+) ^ (k + 1))) :
    Prime (hζ.toInteger - 1) := by
  /-
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k 1))
    ⊢ Prime (HSub.hSub hζ.toInteger 1)
  -/
  letI := IsCyclotomicExtension.numberField {(2 : ℕ+) ^ (k + 1)} ℚ K
  /-
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k 1))
    this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
    ⊢ Prime (HSub.hSub hζ.toInteger 1)
  -/
  refine Ideal.prime_of_irreducible_absNorm_span (fun h ↦ ?_) ?_
    /-
      case refine_1
      k : Nat
      K : Type u
      inst✝² : Field K
      ζ : K
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k 1))
      this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
      h : Eq (HSub.hSub hζ.toInteger 1) 0
      ⊢ False
    -/
  · apply hζ.pow_ne_one_of_pos_of_lt zero_lt_one (one_lt_pow₀ (by decide) (by simp))
    /-
      case refine_1
      k : Nat
      K : Type u
      inst✝² : Field K
      ζ : K
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k 1))
      this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
      h : Eq (HSub.hSub hζ.toInteger 1) 0
      ⊢ Eq (HPow.hPow ζ 1) 1
    -/
    rw [sub_eq_zero] at h
    /-
      case refine_1
      k : Nat
      K : Type u
      inst✝² : Field K
      ζ : K
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k 1))
      this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
      h : Eq hζ.toInteger 1
      ⊢ Eq (HPow.hPow ζ 1) 1
    -/
    simpa using congrArg (algebraMap _ K) h
    /-
      🎉 no goals
    -/
  rw [Nat.irreducible_iff_prime, Ideal.absNorm_span_singleton, ← Nat.prime_iff,
    ← Int.prime_iff_natAbs_prime]
  /-
    case refine_2
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k 1))
    this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
    ⊢ Prime ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))
  -/
  cases k
    /-
      case refine_2.zero
      K : Type u
      inst✝² : Field K
      ζ : K
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd 0 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd 0 1))
      this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
      ⊢ Prime ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))
    -/
  · convert Prime.neg Int.prime_two
    /-
      case h.e'_3
      K : Type u
      inst✝² : Field K
      ζ : K
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd 0 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd 0 1))
      this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
      ⊢ Eq ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1)) (-2)
    -/
    apply RingHom.injective_int (algebraMap ℤ ℚ)
    /-
      case h.e'_3.a
      K : Type u
      inst✝² : Field K
      ζ : K
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd 0 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd 0 1))
      this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
      ⊢ Eq ((algebraMap Int Rat) ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))) (( …
    -/
    rw [← Algebra.norm_localization (Sₘ := K) ℤ (nonZeroDivisors ℤ)]
    simp only [PNat.pow_coe, id.map_eq_id, RingHomCompTriple.comp_eq, RingHom.coe_coe,
      Subalgebra.coe_val, algebraMap_int_eq, map_neg, map_ofNat]
    simpa only [zero_add, pow_one, AddSubgroupClass.coe_sub, OneMemClass.coe_one,
        pow_zero]
      using hζ.norm_pow_sub_one_two (cyclotomic.irreducible_rat
        (by simp only [zero_add, pow_one, Nat.ofNat_pos]))
  /-
    case refine_2.succ
    K : Type u
    inst✝² : Field K
    ζ : K
    inst✝¹ : CharZero K
    n✝ : Nat
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd (HA …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd (HAdd.hAdd n✝ 1) 1))
    this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
    ⊢ Prime ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))
  -/
  convert Int.prime_two
  /-
    case h.e'_3
    K : Type u
    inst✝² : Field K
    ζ : K
    inst✝¹ : CharZero K
    n✝ : Nat
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd (HA …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd (HAdd.hAdd n✝ 1) 1))
    this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
    ⊢ Eq ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1)) 2
  -/
  apply RingHom.injective_int (algebraMap ℤ ℚ)
  /-
    case h.e'_3.a
    K : Type u
    inst✝² : Field K
    ζ : K
    inst✝¹ : CharZero K
    n✝ : Nat
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd (HA …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd (HAdd.hAdd n✝ 1) 1))
    this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
    ⊢ Eq ((algebraMap Int Rat) ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))) (( …
  -/
  rw [← Algebra.norm_localization (Sₘ := K) ℤ (nonZeroDivisors ℤ)]
  simp only [PNat.pow_coe, id.map_eq_id, RingHomCompTriple.comp_eq, RingHom.coe_coe,
    Subalgebra.coe_val, algebraMap_int_eq, map_natCast]
  /-
    case h.e'_3.a
    K : Type u
    inst✝² : Field K
    ζ : K
    inst✝¹ : CharZero K
    n✝ : Nat
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd (HA …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd (HAdd.hAdd n✝ 1) 1))
    this : NumberField K := IsCyclotomicExtension.numberField (Singleton.singleton …
    ⊢ Eq ((Algebra.norm Rat) ((algebraMap (NumberField.RingOfIntegers K) K) (HSub. …
  -/
  exact hζ.norm_sub_one_two Nat.AtLeastTwo.prop (cyclotomic.irreducible_rat (by simp))
  /-
    🎉 no goals
  -/


/-- `ζ - 1` is prime if `ζ` is a primitive `p ^ (k + 1)`-th root of unity. -/
theorem zeta_sub_one_prime [IsCyclotomicExtension {p ^ (k + 1)} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1))) : Prime (hζ.toInteger - 1) := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    ⊢ Prime (HSub.hSub hζ.toInteger 1)
  -/
  by_cases htwo : p = 2
    /-
      case pos
      p : PNat
      k : Nat
      K : Type u
      inst✝² : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      htwo : Eq p 2
      ⊢ Prime (HSub.hSub hζ.toInteger 1)
    -/
  · subst htwo
    /-
      case pos
      k : Nat
      K : Type u
      inst✝² : Field K
      ζ : K
      inst✝¹ : CharZero K
      hp : Fact (Nat.Prime ↑2)
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k 1))
      ⊢ Prime (HSub.hSub hζ.toInteger 1)
    -/
    apply hζ.zeta_sub_one_prime_of_two_pow
    /-
      🎉 no goals
    -/
    /-
      case neg
      p : PNat
      k : Nat
      K : Type u
      inst✝² : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      htwo : Not (Eq p 2)
      ⊢ Prime (HSub.hSub hζ.toInteger 1)
    -/
  · apply hζ.zeta_sub_one_prime_of_ne_two htwo
    /-
      🎉 no goals
    -/


/-- `ζ - 1` is prime if `ζ` is a primitive `p`-th root of unity. -/
theorem zeta_sub_one_prime' [h : IsCyclotomicExtension {p} ℚ K] (hζ : IsPrimitiveRoot ζ p) :
    Prime ((hζ.toInteger - 1)) := by
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    h : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    ⊢ Prime (HSub.hSub hζ.toInteger 1)
  -/
  convert zeta_sub_one_prime (k := 0) (by simpa only [zero_add, pow_one])
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    h : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    ⊢ IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1))) Ra …
  -/
  simpa only [zero_add, pow_one]
  /-
    🎉 no goals
  -/


theorem subOneIntegralPowerBasis_gen_prime [IsCyclotomicExtension {p ^ (k + 1)} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1))) :
    Prime hζ.subOneIntegralPowerBasis.gen := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    ⊢ Prime hζ.subOneIntegralPowerBasis.gen
  -/
  simpa only [subOneIntegralPowerBasis_gen] using hζ.zeta_sub_one_prime
  /-
    🎉 no goals
  -/


theorem subOneIntegralPowerBasis'_gen_prime [IsCyclotomicExtension {p} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑p) :
    Prime hζ.subOneIntegralPowerBasis'.gen := by
  /-
    p : PNat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    ⊢ Prime hζ.subOneIntegralPowerBasis'.gen
  -/
  simpa only [subOneIntegralPowerBasis'_gen] using hζ.zeta_sub_one_prime'
  /-
    🎉 no goals
  -/


/-- The norm, relative to `ℤ`, of `ζ ^ p ^ s - 1` in a `p ^ (k + 1)`-th cyclotomic extension of `ℚ`
is p ^ p ^ s` if `s ≤ k` and `p ^ (k - s + 1) ≠ 2`. -/
lemma norm_toInteger_pow_sub_one_of_prime_pow_ne_two [IsCyclotomicExtension {p ^ (k + 1)} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1))) {s : ℕ} (hs : s ≤ k) (htwo : p ^ (k - s + 1) ≠ 2) :
    Algebra.norm ℤ (hζ.toInteger ^ (p : ℕ) ^ s - 1) = p ^ (p : ℕ) ^ s := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    s : Nat
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    ⊢ Eq ((Algebra.norm Int) (HSub.hSub (HPow.hPow hζ.toInteger (HPow.hPow (↑p) s) …
  -/
  have : NumberField K := IsCyclotomicExtension.numberField {p ^ (k + 1)} ℚ K
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    s : Nat
    hs : LE.le s k
    htwo : Ne (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    this : NumberField K
    ⊢ Eq ((Algebra.norm Int) (HSub.hSub (HPow.hPow hζ.toInteger (HPow.hPow (↑p) s) …
  -/
  rw [Algebra.norm_eq_iff ℤ (Sₘ := K) (Rₘ := ℚ) rfl.le]
  simp [hζ.norm_pow_sub_one_of_prime_pow_ne_two
          (cyclotomic.irreducible_rat (by simp only [PNat.pow_coe, gt_iff_lt, PNat.pos, pow_pos]))
          hs htwo]


/-- The norm, relative to `ℤ`, of `ζ ^ 2 ^ k - 1` in a `2 ^ (k + 1)`-th cyclotomic extension of `ℚ`
is `(-2) ^ 2 ^ k`. -/
lemma norm_toInteger_pow_sub_one_of_two [IsCyclotomicExtension {2 ^ (k + 1)} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑((2 : ℕ+) ^ (k + 1))) :
    Algebra.norm ℤ (hζ.toInteger ^ 2 ^ k - 1) = (-2) ^ (2 : ℕ) ^ k := by
  /-
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k 1))
    ⊢ Eq ((Algebra.norm Int) (HSub.hSub (HPow.hPow hζ.toInteger (HPow.hPow 2 k)) 1 …
  -/
  have : NumberField K := IsCyclotomicExtension.numberField {2 ^ (k + 1)} ℚ K
  /-
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k 1))
    this : NumberField K
    ⊢ Eq ((Algebra.norm Int) (HSub.hSub (HPow.hPow hζ.toInteger (HPow.hPow 2 k)) 1 …
  -/
  rw [Algebra.norm_eq_iff ℤ (Sₘ := K) (Rₘ := ℚ) rfl.le]
  /-
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow 2 (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow 2 (HAdd.hAdd k 1))
    this : NumberField K
    ⊢ Eq ((Algebra.norm Rat) ((algebraMap (NumberField.RingOfIntegers K) K) (HSub. …
  -/
  simp [hζ.norm_pow_sub_one_two (cyclotomic.irreducible_rat (pow_pos (by decide) _))]
  /-
    🎉 no goals
  -/


/-- The norm, relative to `ℤ`, of `ζ ^ p ^ s - 1` in a `p ^ (k + 1)`-th cyclotomic extension of `ℚ`
is `p ^ p ^ s` if `s ≤ k` and `p ≠ 2`. -/
lemma norm_toInteger_pow_sub_one_of_prime_ne_two [IsCyclotomicExtension {p ^ (k + 1)} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1))) {s : ℕ} (hs : s ≤ k) (hodd : p ≠ 2) :
    Algebra.norm ℤ (hζ.toInteger ^ (p : ℕ) ^ s - 1) = p ^ (p : ℕ) ^ s := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    s : Nat
    hs : LE.le s k
    hodd : Ne p 2
    ⊢ Eq ((Algebra.norm Int) (HSub.hSub (HPow.hPow hζ.toInteger (HPow.hPow (↑p) s) …
  -/
  refine hζ.norm_toInteger_pow_sub_one_of_prime_pow_ne_two hs (fun h ↦ hodd ?_)
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    s : Nat
    hs : LE.le s k
    hodd : Ne p 2
    h : Eq (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    ⊢ Eq p 2
  -/
  suffices h : (p : ℕ) = 2 from PNat.coe_injective h
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    s : Nat
    hs : LE.le s k
    hodd : Ne p 2
    h : Eq (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    ⊢ Eq (↑p) 2
  -/
  apply eq_of_prime_pow_eq hp.out.prime Nat.prime_two.prime (k - s).succ_pos
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    s : Nat
    hs : LE.le s k
    hodd : Ne p 2
    h : Eq (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    ⊢ Eq (HPow.hPow (↑p) (HSub.hSub k s).succ) (HPow.hPow 2 ?m.261420)
  -/
  rw [pow_one]
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    s : Nat
    hs : LE.le s k
    hodd : Ne p 2
    h : Eq (HPow.hPow p (HAdd.hAdd (HSub.hSub k s) 1)) 2
    ⊢ Eq (HPow.hPow (↑p) (HSub.hSub k s).succ) 2
  -/
  exact congr_arg Subtype.val h
  /-
    🎉 no goals
  -/


/-- The norm, relative to `ℤ`, of `ζ - 1` in a `p ^ (k + 1)`-th cyclotomic extension of `ℚ` is
`p` if `p ≠ 2`. -/
lemma norm_toInteger_sub_one_of_prime_ne_two [IsCyclotomicExtension {p ^ (k + 1)} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1))) (hodd : p ≠ 2) :
    Algebra.norm ℤ (hζ.toInteger - 1) = p := by
  simpa only [pow_zero, pow_one] using
    hζ.norm_toInteger_pow_sub_one_of_prime_ne_two (Nat.zero_le _) hodd


/-- The norm, relative to `ℤ`, of `ζ - 1` in a `p`-th cyclotomic extension of `ℚ` is `p` if
`p ≠ 2`. -/
lemma norm_toInteger_sub_one_of_prime_ne_two' [hcycl : IsCyclotomicExtension {p} ℚ K]
    (hζ : IsPrimitiveRoot ζ p) (h : p ≠ 2) : Algebra.norm ℤ (hζ.toInteger - 1) = p := by
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    h : Ne p 2
    ⊢ Eq ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1)) ↑↑p
  -/
  have : IsCyclotomicExtension {p ^ (0 + 1)} ℚ K := by simpa using hcycl
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    h : Ne p 2
    this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
    ⊢ Eq ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1)) ↑↑p
  -/
  replace hζ : IsPrimitiveRoot ζ (p ^ (0 + 1)) := by simpa using hζ
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ✝ : IsPrimitiveRoot ζ ↑p
    h : Ne p 2
    this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
    hζ : IsPrimitiveRoot ζ (HPow.hPow (↑p) (HAdd.hAdd 0 1))
    ⊢ Eq ((Algebra.norm Int) (HSub.hSub hζ✝.toInteger 1)) ↑↑p
  -/
  exact hζ.norm_toInteger_sub_one_of_prime_ne_two h
  /-
    🎉 no goals
  -/


/-- The norm, relative to `ℤ`, of `ζ - 1` in a `p ^ (k + 1)`-th cyclotomic extension of `ℚ` is
a prime if `p ^ (k  + 1) ≠ 2`. -/
lemma prime_norm_toInteger_sub_one_of_prime_pow_ne_two [IsCyclotomicExtension {p ^ (k + 1)} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1))) (htwo : p ^ (k + 1) ≠ 2) :
    Prime (Algebra.norm ℤ (hζ.toInteger - 1)) := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    ⊢ Prime ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))
  -/
  have := hζ.norm_toInteger_pow_sub_one_of_prime_pow_ne_two (zero_le _) htwo
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    this : Eq ((Algebra.norm Int) (HSub.hSub (HPow.hPow hζ.toInteger (HPow.hPow (↑ …
    ⊢ Prime ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))
  -/
  simp only [pow_zero, pow_one] at this
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    this : Eq ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1)) ↑↑p
    ⊢ Prime ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))
  -/
  rw [this]
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    this : Eq ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1)) ↑↑p
    ⊢ Prime ↑↑p
  -/
  exact Nat.prime_iff_prime_int.1 hp.out
  /-
    🎉 no goals
  -/


/-- The norm, relative to `ℤ`, of `ζ - 1` in a `p ^ (k + 1)`-th cyclotomic extension of `ℚ` is
a prime if `p ≠ 2`. -/
lemma prime_norm_toInteger_sub_one_of_prime_ne_two [hcycl : IsCyclotomicExtension {p ^ (k + 1)} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1))) (hodd : p ≠ 2) :
    Prime (Algebra.norm ℤ (hζ.toInteger - 1)) := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hodd : Ne p 2
    ⊢ Prime ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))
  -/
  have := hζ.norm_toInteger_sub_one_of_prime_ne_two hodd
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hodd : Ne p 2
    this : Eq ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1)) ↑↑p
    ⊢ Prime ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))
  -/
  simp only [pow_zero, pow_one] at this
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hodd : Ne p 2
    this : Eq ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1)) ↑↑p
    ⊢ Prime ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))
  -/
  rw [this]
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hodd : Ne p 2
    this : Eq ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1)) ↑↑p
    ⊢ Prime ↑↑p
  -/
  exact Nat.prime_iff_prime_int.1 hp.out
  /-
    🎉 no goals
  -/


/-- The norm, relative to `ℤ`, of `ζ - 1` in a `p`-th cyclotomic extension of `ℚ` is a prime if
`p ≠ 2`. -/
lemma prime_norm_toInteger_sub_one_of_prime_ne_two' [hcycl : IsCyclotomicExtension {p} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑p) (hodd : p ≠ 2) :
    Prime (Algebra.norm ℤ (hζ.toInteger - 1)) := by
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    hodd : Ne p 2
    ⊢ Prime ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))
  -/
  have : IsCyclotomicExtension {p ^ (0 + 1)} ℚ K := by simpa using hcycl
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    hodd : Ne p 2
    this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
    ⊢ Prime ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))
  -/
  replace hζ : IsPrimitiveRoot ζ (p ^ (0 + 1)) := by simpa using hζ
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ✝ : IsPrimitiveRoot ζ ↑p
    hodd : Ne p 2
    this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
    hζ : IsPrimitiveRoot ζ (HPow.hPow (↑p) (HAdd.hAdd 0 1))
    ⊢ Prime ((Algebra.norm Int) (HSub.hSub hζ✝.toInteger 1))
  -/
  exact hζ.prime_norm_toInteger_sub_one_of_prime_ne_two hodd
  /-
    🎉 no goals
  -/


/-- In a `p ^ (k + 1)`-th cyclotomic extension of `ℚ `, we have that `ζ` is not congruent to an
  integer modulo `p` if `p ^ (k  + 1) ≠ 2`. -/
theorem not_exists_int_prime_dvd_sub_of_prime_pow_ne_two
    [hcycl : IsCyclotomicExtension {p ^ (k + 1)} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1))) (htwo : p ^ (k + 1) ≠ 2) :
    ¬(∃ n : ℤ, (p : 𝓞 K) ∣ (hζ.toInteger - n : 𝓞 K)) := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    ⊢ Not (Exists fun n => Dvd.dvd (↑↑p) (HSub.hSub hζ.toInteger ↑n))
  -/
  intro ⟨n, x, h⟩
  -- Let `pB` be the power basis of `𝓞 K` given by powers of `ζ`.
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    n : Int
    x : NumberField.RingOfIntegers K
    h : Eq (HSub.hSub hζ.toInteger ↑n) (HMul.hMul (↑↑p) x)
    ⊢ False
  -/
  let pB := hζ.integralPowerBasis
  have hdim : pB.dim = ↑p ^ k * (↑p - 1) := by
    simp [integralPowerBasis_dim, pB, Nat.totient_prime_pow hp.1 (Nat.zero_lt_succ k)]
  replace hdim : 1 < pB.dim := by
    rw [Nat.one_lt_iff_ne_zero_and_ne_one, hdim]
    refine ⟨by simp only [ne_eq, mul_eq_zero, pow_eq_zero_iff', PNat.ne_zero, false_and, false_or,
      Nat.sub_eq_zero_iff_le, not_le, Nat.Prime.one_lt hp.out], ne_of_gt ?_⟩
    by_cases hk : k = 0
    · simp only [hk, zero_add, pow_one, pow_zero, one_mul, Nat.lt_sub_iff_add_lt,
        Nat.reduceAdd] at htwo ⊢
      exact htwo.symm.lt_of_le hp.1.two_le
    · exact one_lt_mul_of_lt_of_le (one_lt_pow₀ hp.1.one_lt hk)
        (have := Nat.Prime.two_le hp.out; by omega)
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    n : Int
    x : NumberField.RingOfIntegers K
    h : Eq (HSub.hSub hζ.toInteger ↑n) (HMul.hMul (↑↑p) x)
    pB : PowerBasis Int (NumberField.RingOfIntegers K) := hζ.integralPowerBasis
    hdim : LT.lt 1 pB.dim
    ⊢ False
  -/
  rw [sub_eq_iff_eq_add] at h
  -- We are assuming that `ζ = n + p * x` for some integer `n` and `x : 𝓞 K`. Looking at the
  -- coordinates in the base `pB`, we obtain that `1` is a multiple of `p`, contradiction.
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    n : Int
    x : NumberField.RingOfIntegers K
    h : Eq hζ.toInteger (HAdd.hAdd (HMul.hMul (↑↑p) x) ↑n)
    pB : PowerBasis Int (NumberField.RingOfIntegers K) := hζ.integralPowerBasis
    hdim : LT.lt 1 pB.dim
    ⊢ False
  -/
  replace h := pB.basis.ext_elem_iff.1 h ⟨1, hdim⟩
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    n : Int
    x : NumberField.RingOfIntegers K
    pB : PowerBasis Int (NumberField.RingOfIntegers K) := hζ.integralPowerBasis
    hdim : LT.lt 1 pB.dim
    h : Eq ((pB.basis.repr hζ.toInteger) ⟨1, hdim⟩) ((pB.basis.repr (HAdd.hAdd (HM …
    ⊢ False
  -/
  have := pB.basis_eq_pow ⟨1, hdim⟩
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    n : Int
    x : NumberField.RingOfIntegers K
    pB : PowerBasis Int (NumberField.RingOfIntegers K) := hζ.integralPowerBasis
    hdim : LT.lt 1 pB.dim
    h : Eq ((pB.basis.repr hζ.toInteger) ⟨1, hdim⟩) ((pB.basis.repr (HAdd.hAdd (HM …
    this : Eq (pB.basis ⟨1, hdim⟩) (HPow.hPow pB.gen ↑⟨1, hdim⟩)
    ⊢ False
  -/
  rw [hζ.integralPowerBasis_gen] at this
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    n : Int
    x : NumberField.RingOfIntegers K
    pB : PowerBasis Int (NumberField.RingOfIntegers K) := hζ.integralPowerBasis
    hdim : LT.lt 1 pB.dim
    h : Eq ((pB.basis.repr hζ.toInteger) ⟨1, hdim⟩) ((pB.basis.repr (HAdd.hAdd (HM …
    this : Eq (pB.basis ⟨1, hdim⟩) (HPow.hPow hζ.toInteger ↑⟨1, hdim⟩)
    ⊢ False
  -/
  simp only [PowerBasis.coe_basis, pow_one] at this
  rw [← this, show pB.gen = pB.gen ^ (⟨1, hdim⟩ : Fin pB.dim).1 by simp, ← pB.basis_eq_pow,
    pB.basis.repr_self_apply] at h
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    n : Int
    x : NumberField.RingOfIntegers K
    pB : PowerBasis Int (NumberField.RingOfIntegers K) := hζ.integralPowerBasis
    hdim : LT.lt 1 pB.dim
    h : Eq (ite (Eq ⟨1, hdim⟩ ⟨1, hdim⟩) 1 0) ((pB.basis.repr (HAdd.hAdd (HMul.hMu …
    this : Eq pB.gen hζ.toInteger
    ⊢ False
  -/
  simp only [↓reduceIte, map_add, Finsupp.coe_add, Pi.add_apply] at h
  rw [show (p : 𝓞 K) * x = (p : ℤ) • x by simp, ← pB.basis.coord_apply,
    LinearMap.map_smul, ← zsmul_one, ← pB.basis.coord_apply, LinearMap.map_smul,
    show 1 = pB.gen ^ (⟨0, by omega⟩ : Fin pB.dim).1 by simp, ← pB.basis_eq_pow,
    pB.basis.coord_apply, pB.basis.coord_apply, pB.basis.repr_self_apply] at h
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    n : Int
    x : NumberField.RingOfIntegers K
    pB : PowerBasis Int (NumberField.RingOfIntegers K) := hζ.integralPowerBasis
    hdim : LT.lt 1 pB.dim
    this : Eq pB.gen hζ.toInteger
    h : Eq 1 (HAdd.hAdd (HSMul.hSMul (↑↑p) ((pB.basis.repr x) ⟨1, hdim⟩)) (HSMul.h …
    ⊢ False
  -/
  simp only [smul_eq_mul, Fin.mk.injEq, zero_ne_one, ↓reduceIte, mul_zero, add_zero] at h
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Ne (HPow.hPow p (HAdd.hAdd k 1)) 2
    n : Int
    x : NumberField.RingOfIntegers K
    pB : PowerBasis Int (NumberField.RingOfIntegers K) := hζ.integralPowerBasis
    hdim : LT.lt 1 pB.dim
    this : Eq pB.gen hζ.toInteger
    h : Eq 1 (HMul.hMul (↑↑p) ((pB.basis.repr x) ⟨1, hdim⟩))
    ⊢ False
  -/
  exact (Int.prime_iff_natAbs_prime.2 (by simp [hp.1])).not_dvd_one ⟨_, h⟩
  /-
    🎉 no goals
  -/


/-- In a `p ^ (k + 1)`-th cyclotomic extension of `ℚ `, we have that `ζ` is not congruent to an
  integer modulo `p` if `p ≠ 2`. -/
theorem not_exists_int_prime_dvd_sub_of_prime_ne_two
    [hcycl : IsCyclotomicExtension {p ^ (k + 1)} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1))) (hodd : p ≠ 2) :
    ¬(∃ n : ℤ, (p : 𝓞 K) ∣ (hζ.toInteger - n : 𝓞 K)) := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hodd : Ne p 2
    ⊢ Not (Exists fun n => Dvd.dvd (↑↑p) (HSub.hSub hζ.toInteger ↑n))
  -/
  refine not_exists_int_prime_dvd_sub_of_prime_pow_ne_two hζ (fun h ↦ ?_)
  simp_all only [(@Nat.Prime.pow_eq_iff 2 p (k+1) Nat.prime_two).mp (by assumption_mod_cast),
    pow_one, ne_eq]


/-- In a `p`-th cyclotomic extension of `ℚ `, we have that `ζ` is not congruent to an
  integer modulo `p` if `p ≠ 2`. -/
theorem not_exists_int_prime_dvd_sub_of_prime_ne_two'
    [hcycl : IsCyclotomicExtension {p} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑p) (hodd : p ≠ 2) :
    ¬(∃ n : ℤ, (p : 𝓞 K) ∣ (hζ.toInteger - n : 𝓞 K)) := by
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    hodd : Ne p 2
    ⊢ Not (Exists fun n => Dvd.dvd (↑↑p) (HSub.hSub hζ.toInteger ↑n))
  -/
  have : IsCyclotomicExtension {p ^ (0 + 1)} ℚ K := by simpa using hcycl
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    hodd : Ne p 2
    this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
    ⊢ Not (Exists fun n => Dvd.dvd (↑↑p) (HSub.hSub hζ.toInteger ↑n))
  -/
  replace hζ : IsPrimitiveRoot ζ (p ^ (0 + 1)) := by simpa using hζ
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ✝ : IsPrimitiveRoot ζ ↑p
    hodd : Ne p 2
    this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
    hζ : IsPrimitiveRoot ζ (HPow.hPow (↑p) (HAdd.hAdd 0 1))
    ⊢ Not (Exists fun n => Dvd.dvd (↑↑p) (HSub.hSub hζ✝.toInteger ↑n))
  -/
  exact not_exists_int_prime_dvd_sub_of_prime_ne_two hζ hodd
  /-
    🎉 no goals
  -/


theorem finite_quotient_span_sub_one [hcycl : IsCyclotomicExtension {p ^ (k + 1)} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1))) :
    Finite (𝓞 K ⧸ Ideal.span {hζ.toInteger - 1}) := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    ⊢ Finite (HasQuotient.Quotient (NumberField.RingOfIntegers K) (Ideal.span (Sin …
  -/
  have : NumberField K := IsCyclotomicExtension.numberField {p ^ (k + 1)} ℚ K
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    this : NumberField K
    ⊢ Finite (HasQuotient.Quotient (NumberField.RingOfIntegers K) (Ideal.span (Sin …
  -/
  refine Fintype.finite <| Ideal.fintypeQuotientOfFreeOfNeBot _ (fun h ↦ ?_)
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    this : NumberField K
    h : Eq (Ideal.span (Singleton.singleton (HSub.hSub hζ.toInteger 1))) Bot.bot
    ⊢ False
  -/
  simp only [Ideal.span_singleton_eq_bot, sub_eq_zero, ← Subtype.coe_inj] at h
  exact hζ.ne_one (one_lt_pow₀ hp.1.one_lt (Nat.zero_ne_add_one k).symm)
    (RingOfIntegers.ext_iff.1 h)


theorem finite_quotient_span_sub_one' [hcycl : IsCyclotomicExtension {p} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑p) :
    Finite (𝓞 K ⧸ Ideal.span {hζ.toInteger - 1}) := by
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    ⊢ Finite (HasQuotient.Quotient (NumberField.RingOfIntegers K) (Ideal.span (Sin …
  -/
  have : IsCyclotomicExtension {p ^ (0 + 1)} ℚ K := by simpa using hcycl
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
    ⊢ Finite (HasQuotient.Quotient (NumberField.RingOfIntegers K) (Ideal.span (Sin …
  -/
  replace hζ : IsPrimitiveRoot ζ (p ^ (0 + 1)) := by simpa using hζ
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ✝ : IsPrimitiveRoot ζ ↑p
    this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
    hζ : IsPrimitiveRoot ζ (HPow.hPow (↑p) (HAdd.hAdd 0 1))
    ⊢ Finite (HasQuotient.Quotient (NumberField.RingOfIntegers K) (Ideal.span (Sin …
  -/
  exact hζ.finite_quotient_span_sub_one
  /-
    🎉 no goals
  -/


/-- In a `p ^ (k + 1)`-th cyclotomic extension of `ℚ`, we have that
  `ζ - 1` divides `p` in `𝓞 K`. -/
lemma toInteger_sub_one_dvd_prime [hcycl : IsCyclotomicExtension {p ^ (k + 1)} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1))) : ((hζ.toInteger - 1)) ∣ p := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) ↑↑p
  -/
  by_cases htwo : p ^ (k + 1) = 2
    /-
      case pos
      p : PNat
      k : Nat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      htwo : Eq (HPow.hPow p (HAdd.hAdd k 1)) 2
      ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) ↑↑p
    -/
  · replace htwo : (p : ℕ) ^ (k + 1) = 2 := by exact_mod_cast htwo
    /-
      case pos
      p : PNat
      k : Nat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      htwo : Eq (HPow.hPow (↑p) (HAdd.hAdd k 1)) 2
      ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) ↑↑p
    -/
    have ⟨hp2, hk⟩ := (Nat.Prime.pow_eq_iff Nat.prime_two).1 htwo
    /-
      case pos
      p : PNat
      k : Nat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      htwo : Eq (HPow.hPow (↑p) (HAdd.hAdd k 1)) 2
      hp2 : Eq (↑p) 2
      hk : Eq (HAdd.hAdd k 1) 1
      ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) ↑↑p
    -/
    simp only [add_left_eq_self] at hk
    have hζ' : ζ = -1 := by
      refine IsPrimitiveRoot.eq_neg_one_of_two_right ?_
      rwa [hk, zero_add, pow_one, hp2] at hζ
    replace hζ' : hζ.toInteger = -1 := by
      ext
      exact hζ'
    /-
      case pos
      p : PNat
      k : Nat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      htwo : Eq (HPow.hPow (↑p) (HAdd.hAdd k 1)) 2
      hp2 : Eq (↑p) 2
      hk : Eq k 0
      hζ' : Eq hζ.toInteger (-1)
      ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) ↑↑p
    -/
    rw [hζ', hp2]
    /-
      case pos
      p : PNat
      k : Nat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      htwo : Eq (HPow.hPow (↑p) (HAdd.hAdd k 1)) 2
      hp2 : Eq (↑p) 2
      hk : Eq k 0
      hζ' : Eq hζ.toInteger (-1)
      ⊢ Dvd.dvd (HSub.hSub (-1) 1) ↑2
    -/
    exact ⟨-1, by ring⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Not (Eq (HPow.hPow p (HAdd.hAdd k 1)) 2)
    ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) ↑↑p
  -/
  suffices (hζ.toInteger - 1) ∣ (p : ℤ) by simpa
  /-
    case neg
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Not (Eq (HPow.hPow p (HAdd.hAdd k 1)) 2)
    ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) ↑↑↑p
  -/
  have := IsCyclotomicExtension.numberField {p ^ (k + 1)} ℚ K
  /-
    case neg
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Not (Eq (HPow.hPow p (HAdd.hAdd k 1)) 2)
    this : NumberField K
    ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) ↑↑↑p
  -/
  have H := hζ.norm_toInteger_pow_sub_one_of_prime_pow_ne_two (zero_le _) htwo
  /-
    case neg
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Not (Eq (HPow.hPow p (HAdd.hAdd k 1)) 2)
    this : NumberField K
    H : Eq ((Algebra.norm Int) (HSub.hSub (HPow.hPow hζ.toInteger (HPow.hPow (↑p)  …
    ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) ↑↑↑p
  -/
  rw [pow_zero, pow_one] at H
  /-
    case neg
    p : PNat
    k : Nat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    htwo : Not (Eq (HPow.hPow p (HAdd.hAdd k 1)) 2)
    this : NumberField K
    H : Eq ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1)) (HPow.hPow (↑↑p) 1)
    ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) ↑↑↑p
  -/
  rw [← Ideal.norm_dvd_iff, H]
    /-
      case neg
      p : PNat
      k : Nat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      htwo : Not (Eq (HPow.hPow p (HAdd.hAdd k 1)) 2)
      this : NumberField K
      H : Eq ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1)) (HPow.hPow (↑↑p) 1)
      ⊢ Dvd.dvd (HPow.hPow (↑↑p) 1) ↑↑p
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case neg.hx
      p : PNat
      k : Nat
      K : Type u
      inst✝¹ : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝ : CharZero K
      hcycl : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      htwo : Not (Eq (HPow.hPow p (HAdd.hAdd k 1)) 2)
      this : NumberField K
      H : Eq ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1)) (HPow.hPow (↑↑p) 1)
      ⊢ Prime ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))
    -/
  · exact prime_norm_toInteger_sub_one_of_prime_pow_ne_two hζ htwo
    /-
      🎉 no goals
    -/


/-- In a `p`-th cyclotomic extension of `ℚ`, we have that `ζ - 1` divides `p` in `𝓞 K`. -/
lemma toInteger_sub_one_dvd_prime' [hcycl : IsCyclotomicExtension {p} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑p) : ((hζ.toInteger - 1)) ∣ p := by
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) ↑↑p
  -/
  have : IsCyclotomicExtension {p ^ (0 + 1)} ℚ K := by simpa using hcycl
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ : IsPrimitiveRoot ζ ↑p
    this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
    ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) ↑↑p
  -/
  replace hζ : IsPrimitiveRoot ζ (p ^ (0 + 1)) := by simpa using hζ
  /-
    p : PNat
    K : Type u
    inst✝¹ : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝ : CharZero K
    hcycl : IsCyclotomicExtension (Singleton.singleton p) Rat K
    hζ✝ : IsPrimitiveRoot ζ ↑p
    this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
    hζ : IsPrimitiveRoot ζ (HPow.hPow (↑p) (HAdd.hAdd 0 1))
    ⊢ Dvd.dvd (HSub.hSub hζ✝.toInteger 1) ↑↑p
  -/
  exact toInteger_sub_one_dvd_prime hζ
  /-
    🎉 no goals
  -/


/-- We have that `hζ.toInteger - 1` does not divide `2`. -/
lemma toInteger_sub_one_not_dvd_two [IsCyclotomicExtension {p ^ (k + 1)} ℚ K]
    (hζ : IsPrimitiveRoot ζ ↑(p ^ (k + 1))) (hodd : p ≠ 2) : ¬ hζ.toInteger - 1 ∣ 2 := fun h ↦ by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hodd : Ne p 2
    h : Dvd.dvd (HSub.hSub hζ.toInteger 1) 2
    ⊢ False
  -/
  have : NumberField K := IsCyclotomicExtension.numberField {p ^ (k + 1)} ℚ K
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hodd : Ne p 2
    h : Dvd.dvd (HSub.hSub hζ.toInteger 1) 2
    this : NumberField K
    ⊢ False
  -/
  replace h : hζ.toInteger - 1 ∣ ↑(2 : ℤ) := by simp [h]
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    ζ : K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
    hodd : Ne p 2
    this : NumberField K
    h : Dvd.dvd (HSub.hSub hζ.toInteger 1) ↑2
    ⊢ False
  -/
  rw [← Ideal.norm_dvd_iff, hζ.norm_toInteger_sub_one_of_prime_ne_two hodd] at h
    /-
      p : PNat
      k : Nat
      K : Type u
      inst✝² : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hodd : Ne p 2
      this : NumberField K
      h : Dvd.dvd (↑↑p) 2
      ⊢ False
    -/
  · refine hodd <| PNat.coe_inj.1 <| (prime_dvd_prime_iff_eq ?_ ?_).1 ?_
      /-
        case refine_1
        p : PNat
        k : Nat
        K : Type u
        inst✝² : Field K
        ζ : K
        hp : Fact (Nat.Prime ↑p)
        inst✝¹ : CharZero K
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
        hodd : Ne p 2
        this : NumberField K
        h : Dvd.dvd (↑↑p) 2
        ⊢ Prime ↑p
      -/
    · exact Nat.prime_iff.1 hp.1
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        p : PNat
        k : Nat
        K : Type u
        inst✝² : Field K
        ζ : K
        hp : Fact (Nat.Prime ↑p)
        inst✝¹ : CharZero K
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
        hodd : Ne p 2
        this : NumberField K
        h : Dvd.dvd (↑↑p) 2
        ⊢ Prime ↑2
      -/
    · exact Nat.prime_iff.1 Nat.prime_two
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        p : PNat
        k : Nat
        K : Type u
        inst✝² : Field K
        ζ : K
        hp : Fact (Nat.Prime ↑p)
        inst✝¹ : CharZero K
        inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
        hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
        hodd : Ne p 2
        this : NumberField K
        h : Dvd.dvd (↑↑p) 2
        ⊢ Dvd.dvd ↑p ↑2
      -/
    · exact Int.ofNat_dvd.mp h
      /-
        🎉 no goals
      -/
    /-
      case hx
      p : PNat
      k : Nat
      K : Type u
      inst✝² : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hodd : Ne p 2
      this : NumberField K
      h : Dvd.dvd (HSub.hSub hζ.toInteger 1) ↑2
      ⊢ Prime ((Algebra.norm Int) (HSub.hSub hζ.toInteger 1))
    -/
  · rw [hζ.norm_toInteger_sub_one_of_prime_ne_two hodd]
    /-
      case hx
      p : PNat
      k : Nat
      K : Type u
      inst✝² : Field K
      ζ : K
      hp : Fact (Nat.Prime ↑p)
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
      hζ : IsPrimitiveRoot ζ ↑(HPow.hPow p (HAdd.hAdd k 1))
      hodd : Ne p 2
      this : NumberField K
      h : Dvd.dvd (HSub.hSub hζ.toInteger 1) ↑2
      ⊢ Prime ↑↑p
    -/
    exact Nat.prime_iff_prime_int.1 hp.1
    /-
      🎉 no goals
    -/


/-- We compute the absolute discriminant of a `p ^ k`-th cyclotomic field.
  Beware that in the cases `p ^ k = 1` and `p ^ k = 2` the formula uses `1 / 2 = 0` and `0 - 1 = 0`.
  See also the results below. -/
theorem absdiscr_prime_pow [IsCyclotomicExtension {p ^ k} ℚ K] :
    haveI : NumberField K := IsCyclotomicExtension.numberField {p ^ k} ℚ K
    NumberField.discr K =
    (-1) ^ ((p ^ k : ℕ).totient / 2) * p ^ ((p : ℕ) ^ (k - 1) * ((p - 1) * k - 1)) := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    ⊢ Eq (NumberField.discr K) (HMul.hMul (HPow.hPow (-1) (HDiv.hDiv (HPow.hPow (↑ …
  -/
  have hζ := IsCyclotomicExtension.zeta_spec (p ^ k) ℚ K
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat K) ↑(HPow …
    ⊢ Eq (NumberField.discr K) (HMul.hMul (HPow.hPow (-1) (HDiv.hDiv (HPow.hPow (↑ …
  -/
  have : NumberField K := IsCyclotomicExtension.numberField {p ^ k} ℚ K
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat K) ↑(HPow …
    this : NumberField K
    ⊢ Eq (NumberField.discr K) (HMul.hMul (HPow.hPow (-1) (HDiv.hDiv (HPow.hPow (↑ …
  -/
  let pB₁ := integralPowerBasis hζ
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat K) ↑(HPow …
    this : NumberField K
    pB₁ : PowerBasis Int (NumberField.RingOfIntegers K) := hζ.integralPowerBasis
    ⊢ Eq (NumberField.discr K) (HMul.hMul (HPow.hPow (-1) (HDiv.hDiv (HPow.hPow (↑ …
  -/
  apply (algebraMap ℤ ℚ).injective_int
  /-
    case a
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat K) ↑(HPow …
    this : NumberField K
    pB₁ : PowerBasis Int (NumberField.RingOfIntegers K) := hζ.integralPowerBasis
    ⊢ Eq ((algebraMap Int Rat) (NumberField.discr K)) ((algebraMap Int Rat) (HMul. …
  -/
  rw [← NumberField.discr_eq_discr _ pB₁.basis, ← Algebra.discr_localizationLocalization ℤ ℤ⁰ K]
  /-
    case a
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
    hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat K) ↑(HPow …
    this : NumberField K
    pB₁ : PowerBasis Int (NumberField.RingOfIntegers K) := hζ.integralPowerBasis
    ⊢ Eq (Algebra.discr Rat ⇑(Basis.localizationLocalization Rat (nonZeroDivisors  …
  -/
  convert IsCyclotomicExtension.discr_prime_pow hζ (cyclotomic.irreducible_rat (p ^ k).2) using 1
  · have : pB₁.dim = (IsPrimitiveRoot.powerBasis ℚ hζ).dim := by
      rw [← PowerBasis.finrank, ← PowerBasis.finrank]
      exact RingOfIntegers.rank K
    /-
      case h.e'_2
      p : PNat
      k : Nat
      K : Type u
      inst✝² : Field K
      hp : Fact (Nat.Prime ↑p)
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
      hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat K) ↑(HPow …
      this✝ : NumberField K
      pB₁ : PowerBasis Int (NumberField.RingOfIntegers K) := hζ.integralPowerBasis
      this : Eq pB₁.dim (IsPrimitiveRoot.powerBasis Rat hζ).dim
      ⊢ Eq (Algebra.discr Rat ⇑(Basis.localizationLocalization Rat (nonZeroDivisors  …
    -/
    rw [← Algebra.discr_reindex _ _ (finCongr this)]
    /-
      case h.e'_2
      p : PNat
      k : Nat
      K : Type u
      inst✝² : Field K
      hp : Fact (Nat.Prime ↑p)
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
      hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat K) ↑(HPow …
      this✝ : NumberField K
      pB₁ : PowerBasis Int (NumberField.RingOfIntegers K) := hζ.integralPowerBasis
      this : Eq pB₁.dim (IsPrimitiveRoot.powerBasis Rat hζ).dim
      ⊢ Eq (Algebra.discr Rat (Function.comp ⇑(Basis.localizationLocalization Rat (n …
    -/
    congr 1
    /-
      case h.e'_2.e_b
      p : PNat
      k : Nat
      K : Type u
      inst✝² : Field K
      hp : Fact (Nat.Prime ↑p)
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
      hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat K) ↑(HPow …
      this✝ : NumberField K
      pB₁ : PowerBasis Int (NumberField.RingOfIntegers K) := hζ.integralPowerBasis
      this : Eq pB₁.dim (IsPrimitiveRoot.powerBasis Rat hζ).dim
      ⊢ Eq (Function.comp ⇑(Basis.localizationLocalization Rat (nonZeroDivisors Int) …
    -/
    ext i
    simp_rw [Function.comp_apply, Basis.localizationLocalization_apply, powerBasis_dim,
      PowerBasis.coe_basis, pB₁, integralPowerBasis_gen]
    /-
      case h.e'_2.e_b.h
      p : PNat
      k : Nat
      K : Type u
      inst✝² : Field K
      hp : Fact (Nat.Prime ↑p)
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
      hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat K) ↑(HPow …
      this✝ : NumberField K
      pB₁ : PowerBasis Int (NumberField.RingOfIntegers K) := hζ.integralPowerBasis
      this : Eq pB₁.dim (IsPrimitiveRoot.powerBasis Rat hζ).dim
      i : Fin (IsPrimitiveRoot.powerBasis Rat hζ).dim
      ⊢ Eq ((algebraMap (NumberField.RingOfIntegers K) K) (HPow.hPow hζ.toInteger ↑( …
    -/
    convert ← ((IsPrimitiveRoot.powerBasis ℚ hζ).basis_eq_pow i).symm using 1
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      p : PNat
      k : Nat
      K : Type u
      inst✝² : Field K
      hp : Fact (Nat.Prime ↑p)
      inst✝¹ : CharZero K
      inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p k)) Rat K
      hζ : IsPrimitiveRoot (IsCyclotomicExtension.zeta (HPow.hPow p k) Rat K) ↑(HPow …
      this : NumberField K
      pB₁ : PowerBasis Int (NumberField.RingOfIntegers K) := hζ.integralPowerBasis
      ⊢ Eq ((algebraMap Int Rat) (HMul.hMul (HPow.hPow (-1) (HDiv.hDiv (HPow.hPow (↑ …
    -/
  · simp_rw [algebraMap_int_eq, map_mul, map_pow, map_neg, map_one, map_natCast]
    /-
      🎉 no goals
    -/


open Nat in
/-- We compute the absolute discriminant of a `p ^ (k + 1)`-th cyclotomic field.
  Beware that in the case `p ^ k = 2` the formula uses `1 / 2 = 0`. See also the results below. -/
theorem absdiscr_prime_pow_succ [IsCyclotomicExtension {p ^ (k + 1)} ℚ K] :
    haveI : NumberField K := IsCyclotomicExtension.numberField {p ^ (k + 1)} ℚ K
    NumberField.discr K =
    (-1) ^ ((p : ℕ) ^ k * (p - 1) / 2) * p ^ ((p : ℕ) ^ k * ((p - 1) * (k + 1) - 1)) := by
  /-
    p : PNat
    k : Nat
    K : Type u
    inst✝² : Field K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd k 1 …
    ⊢ Eq (NumberField.discr K) (HMul.hMul (HPow.hPow (-1) (HDiv.hDiv (HMul.hMul (H …
  -/
  simpa [totient_prime_pow hp.out (succ_pos k)] using absdiscr_prime_pow p (k + 1) K
  /-
    🎉 no goals
  -/


/-- We compute the absolute discriminant of a `p`-th cyclotomic field where `p` is prime. -/
theorem absdiscr_prime [IsCyclotomicExtension {p} ℚ K] :
    haveI : NumberField K := IsCyclotomicExtension.numberField {p} ℚ K
    NumberField.discr K = (-1) ^ (((p : ℕ) - 1) / 2) * p ^ ((p : ℕ) - 2) := by
  have : IsCyclotomicExtension {p ^ (0 + 1)} ℚ K := by
    rw [zero_add, pow_one]
    infer_instance
  /-
    p : PNat
    K : Type u
    inst✝² : Field K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton p) Rat K
    this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
    ⊢ Eq (NumberField.discr K) (HMul.hMul (HPow.hPow (-1) (HDiv.hDiv (HSub.hSub (↑ …
  -/
  rw [absdiscr_prime_pow_succ p 0 K]
  simp only [Int.reduceNeg, pow_zero, one_mul, zero_add, mul_one, mul_eq_mul_left_iff, gt_iff_lt,
    Nat.cast_pos, PNat.pos, pow_eq_zero_iff', neg_eq_zero, one_ne_zero, ne_eq, false_and, or_false]
  /-
    p : PNat
    K : Type u
    inst✝² : Field K
    hp : Fact (Nat.Prime ↑p)
    inst✝¹ : CharZero K
    inst✝ : IsCyclotomicExtension (Singleton.singleton p) Rat K
    this : IsCyclotomicExtension (Singleton.singleton (HPow.hPow p (HAdd.hAdd 0 1) …
    ⊢ Eq (HPow.hPow (↑↑p) (HSub.hSub (HSub.hSub (↑p) 1) 1)) (HPow.hPow (↑↑p) (HSub …
  -/
  rfl
  /-
    🎉 no goals
  -/


