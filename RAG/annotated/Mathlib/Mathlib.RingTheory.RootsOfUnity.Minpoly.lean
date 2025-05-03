/-- `μ` is integral over `ℤ`. -/
-- Porting note: `hpos` was in the `variable` line, with an `omit` in mathlib3 just after this
-- declaration. For some reason, in Lean4, `hpos` gets included also in the declarations below,
-- even if it is not used in the proof.
theorem isIntegral (hpos : 0 < n) : IsIntegral ℤ μ := by
  /-
    n : Nat
    K : Type u_1
    inst✝ : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    hpos : LT.lt 0 n
    ⊢ IsIntegral Int μ
  -/
  use X ^ n - 1
  /-
    case h
    n : Nat
    K : Type u_1
    inst✝ : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    hpos : LT.lt 0 n
    ⊢ And (HSub.hSub (HPow.hPow Polynomial.X n) 1).Monic (Eq (Polynomial.eval₂ (al …
  -/
  constructor
    /-
      case h.left
      n : Nat
      K : Type u_1
      inst✝ : CommRing K
      μ : K
      h : IsPrimitiveRoot μ n
      hpos : LT.lt 0 n
      ⊢ (HSub.hSub (HPow.hPow Polynomial.X n) 1).Monic
    -/
  · exact monic_X_pow_sub_C 1 (ne_of_lt hpos).symm
    /-
      🎉 no goals
    -/
  · simp only [((IsPrimitiveRoot.iff_def μ n).mp h).left, eval₂_one, eval₂_X_pow, eval₂_sub,
      sub_self]


/-- The minimal polynomial of a root of unity `μ` divides `X ^ n - 1`. -/
theorem minpoly_dvd_x_pow_sub_one : minpoly ℤ μ ∣ X ^ n - 1 := by
  /-
    n : Nat
    K : Type u_1
    inst✝² : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    ⊢ Dvd.dvd (minpoly Int μ) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
  -/
  rcases n.eq_zero_or_pos with (rfl | h0)
    /-
      case inl
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      h : IsPrimitiveRoot μ 0
      ⊢ Dvd.dvd (minpoly Int μ) (HSub.hSub (HPow.hPow Polynomial.X 0) 1)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Nat
    K : Type u_1
    inst✝² : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    h0 : GT.gt n 0
    ⊢ Dvd.dvd (minpoly Int μ) (HSub.hSub (HPow.hPow Polynomial.X n) 1)
  -/
  apply minpoly.isIntegrallyClosed_dvd (isIntegral h h0)
  simp only [((IsPrimitiveRoot.iff_def μ n).mp h).left, aeval_X_pow, eq_intCast, Int.cast_one,
    aeval_one, map_sub, sub_self]


/-- The reduction modulo `p` of the minimal polynomial of a root of unity `μ` is separable. -/
theorem separable_minpoly_mod {p : ℕ} [Fact p.Prime] (hdiv : ¬p ∣ n) :
    Separable (map (Int.castRingHom (ZMod p)) (minpoly ℤ μ)) := by
  have hdvd : map (Int.castRingHom (ZMod p)) (minpoly ℤ μ) ∣ X ^ n - 1 := by
    convert RingHom.map_dvd (mapRingHom (Int.castRingHom (ZMod p)))
        (minpoly_dvd_x_pow_sub_one h)
    simp only [map_sub, map_pow, coe_mapRingHom, map_X, map_one]
  /-
    n : Nat
    K : Type u_1
    inst✝³ : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝² : IsDomain K
    inst✝¹ : CharZero K
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hdiv : Not (Dvd.dvd p n)
    hdvd : Dvd.dvd (Polynomial.map (Int.castRingHom (ZMod p)) (minpoly Int μ)) (HS …
    ⊢ (Polynomial.map (Int.castRingHom (ZMod p)) (minpoly Int μ)).Separable
  -/
  refine Separable.of_dvd (separable_X_pow_sub_C 1 ?_ one_ne_zero) hdvd
  /-
    n : Nat
    K : Type u_1
    inst✝³ : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝² : IsDomain K
    inst✝¹ : CharZero K
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hdiv : Not (Dvd.dvd p n)
    hdvd : Dvd.dvd (Polynomial.map (Int.castRingHom (ZMod p)) (minpoly Int μ)) (HS …
    ⊢ Ne (↑n) 0
  -/
  by_contra hzero
  /-
    n : Nat
    K : Type u_1
    inst✝³ : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝² : IsDomain K
    inst✝¹ : CharZero K
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hdiv : Not (Dvd.dvd p n)
    hdvd : Dvd.dvd (Polynomial.map (Int.castRingHom (ZMod p)) (minpoly Int μ)) (HS …
    hzero : Eq (↑n) 0
    ⊢ False
  -/
  exact hdiv ((ZMod.natCast_zmod_eq_zero_iff_dvd n p).1 hzero)
  /-
    🎉 no goals
  -/


/-- The reduction modulo `p` of the minimal polynomial of a root of unity `μ` is squarefree. -/
theorem squarefree_minpoly_mod {p : ℕ} [Fact p.Prime] (hdiv : ¬p ∣ n) :
    Squarefree (map (Int.castRingHom (ZMod p)) (minpoly ℤ μ)) :=
  (separable_minpoly_mod h hdiv).squarefree


/-- Let `P` be the minimal polynomial of a root of unity `μ` and `Q` be the minimal polynomial of
`μ ^ p`, where `p` is a natural number that does not divide `n`. Then `P` divides `expand ℤ p Q`. -/
theorem minpoly_dvd_expand {p : ℕ} (hdiv : ¬p ∣ n) :
    minpoly ℤ μ ∣ expand ℤ p (minpoly ℤ (μ ^ p)) := by
  /-
    n : Nat
    K : Type u_1
    inst✝² : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    p : Nat
    hdiv : Not (Dvd.dvd p n)
    ⊢ Dvd.dvd (minpoly Int μ) ((Polynomial.expand Int p) (minpoly Int (HPow.hPow μ …
  -/
  rcases n.eq_zero_or_pos with (rfl | hpos)
    /-
      case inl
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      p : Nat
      h : IsPrimitiveRoot μ 0
      hdiv : Not (Dvd.dvd p 0)
      ⊢ Dvd.dvd (minpoly Int μ) ((Polynomial.expand Int p) (minpoly Int (HPow.hPow μ …
    -/
  · simp_all
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Nat
    K : Type u_1
    inst✝² : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    p : Nat
    hdiv : Not (Dvd.dvd p n)
    hpos : GT.gt n 0
    ⊢ Dvd.dvd (minpoly Int μ) ((Polynomial.expand Int p) (minpoly Int (HPow.hPow μ …
  -/
  letI : IsIntegrallyClosed ℤ := GCDMonoid.toIsIntegrallyClosed
  /-
    case inr
    n : Nat
    K : Type u_1
    inst✝² : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    p : Nat
    hdiv : Not (Dvd.dvd p n)
    hpos : GT.gt n 0
    this : IsIntegrallyClosed Int := GCDMonoid.toIsIntegrallyClosed
    ⊢ Dvd.dvd (minpoly Int μ) ((Polynomial.expand Int p) (minpoly Int (HPow.hPow μ …
  -/
  refine minpoly.isIntegrallyClosed_dvd (h.isIntegral hpos) ?_
  rw [aeval_def, coe_expand, ← comp, eval₂_eq_eval_map, map_comp, Polynomial.map_pow, map_X,
    eval_comp, eval_pow, eval_X, ← eval₂_eq_eval_map, ← aeval_def]
  /-
    case inr
    n : Nat
    K : Type u_1
    inst✝² : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    p : Nat
    hdiv : Not (Dvd.dvd p n)
    hpos : GT.gt n 0
    this : IsIntegrallyClosed Int := GCDMonoid.toIsIntegrallyClosed
    ⊢ Eq ((Polynomial.aeval (HPow.hPow μ p)) (minpoly Int (HPow.hPow μ p))) 0
  -/
  exact minpoly.aeval _ _
  /-
    🎉 no goals
  -/


/-- Let `P` be the minimal polynomial of a root of unity `μ` and `Q` be the minimal polynomial of
`μ ^ p`, where `p` is a prime that does not divide `n`. Then `P` divides `Q ^ p` modulo `p`. -/
theorem minpoly_dvd_pow_mod {p : ℕ} [hprime : Fact p.Prime] (hdiv : ¬p ∣ n) :
    map (Int.castRingHom (ZMod p)) (minpoly ℤ μ) ∣
      map (Int.castRingHom (ZMod p)) (minpoly ℤ (μ ^ p)) ^ p := by
  /-
    n : Nat
    K : Type u_1
    inst✝² : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    p : Nat
    hprime : Fact (Nat.Prime p)
    hdiv : Not (Dvd.dvd p n)
    ⊢ Dvd.dvd (Polynomial.map (Int.castRingHom (ZMod p)) (minpoly Int μ)) (HPow.hP …
  -/
  set Q := minpoly ℤ (μ ^ p)
  have hfrob :
    map (Int.castRingHom (ZMod p)) Q ^ p = map (Int.castRingHom (ZMod p)) (expand ℤ p Q) := by
    rw [← ZMod.expand_card, map_expand]
  /-
    n : Nat
    K : Type u_1
    inst✝² : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    p : Nat
    hprime : Fact (Nat.Prime p)
    hdiv : Not (Dvd.dvd p n)
    Q : Polynomial Int := minpoly Int (HPow.hPow μ p)
    hfrob : Eq (HPow.hPow (Polynomial.map (Int.castRingHom (ZMod p)) Q) p) (Polyno …
    ⊢ Dvd.dvd (Polynomial.map (Int.castRingHom (ZMod p)) (minpoly Int μ)) (HPow.hP …
  -/
  rw [hfrob]
  /-
    n : Nat
    K : Type u_1
    inst✝² : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    p : Nat
    hprime : Fact (Nat.Prime p)
    hdiv : Not (Dvd.dvd p n)
    Q : Polynomial Int := minpoly Int (HPow.hPow μ p)
    hfrob : Eq (HPow.hPow (Polynomial.map (Int.castRingHom (ZMod p)) Q) p) (Polyno …
    ⊢ Dvd.dvd (Polynomial.map (Int.castRingHom (ZMod p)) (minpoly Int μ)) (Polynom …
  -/
  apply RingHom.map_dvd (mapRingHom (Int.castRingHom (ZMod p)))
  /-
    n : Nat
    K : Type u_1
    inst✝² : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    p : Nat
    hprime : Fact (Nat.Prime p)
    hdiv : Not (Dvd.dvd p n)
    Q : Polynomial Int := minpoly Int (HPow.hPow μ p)
    hfrob : Eq (HPow.hPow (Polynomial.map (Int.castRingHom (ZMod p)) Q) p) (Polyno …
    ⊢ Dvd.dvd (minpoly Int μ) ((Polynomial.expand Int p) Q)
  -/
  exact minpoly_dvd_expand h hdiv
  /-
    🎉 no goals
  -/


/-- Let `P` be the minimal polynomial of a root of unity `μ` and `Q` be the minimal polynomial of
`μ ^ p`, where `p` is a prime that does not divide `n`. Then `P` divides `Q` modulo `p`. -/
theorem minpoly_dvd_mod_p {p : ℕ} [Fact p.Prime] (hdiv : ¬p ∣ n) :
    map (Int.castRingHom (ZMod p)) (minpoly ℤ μ) ∣
      map (Int.castRingHom (ZMod p)) (minpoly ℤ (μ ^ p)) :=
  (squarefree_minpoly_mod h hdiv).isRadical _ _ (minpoly_dvd_pow_mod h hdiv)


/-- If `p` is a prime that does not divide `n`,
then the minimal polynomials of a primitive `n`-th root of unity `μ`
and of `μ ^ p` are the same. -/
theorem minpoly_eq_pow {p : ℕ} [hprime : Fact p.Prime] (hdiv : ¬p ∣ n) :
    minpoly ℤ μ = minpoly ℤ (μ ^ p) := by
  classical
  by_cases hn : n = 0
  · simp_all
  have hpos := Nat.pos_of_ne_zero hn
  by_contra hdiff
  set P := minpoly ℤ μ
  set Q := minpoly ℤ (μ ^ p)
  have Pmonic : P.Monic := minpoly.monic (h.isIntegral hpos)
  have Qmonic : Q.Monic := minpoly.monic ((h.pow_of_prime hprime.1 hdiv).isIntegral hpos)
  have Pirr : Irreducible P := minpoly.irreducible (h.isIntegral hpos)
  have Qirr : Irreducible Q := minpoly.irreducible ((h.pow_of_prime hprime.1 hdiv).isIntegral hpos)
  have PQprim : IsPrimitive (P * Q) := Pmonic.isPrimitive.mul Qmonic.isPrimitive
  have prod : P * Q ∣ X ^ n - 1 := by
    rw [IsPrimitive.Int.dvd_iff_map_cast_dvd_map_cast (P * Q) (X ^ n - 1) PQprim
        (monic_X_pow_sub_C (1 : ℤ) (ne_of_gt hpos)).isPrimitive,
      Polynomial.map_mul]
    refine IsCoprime.mul_dvd ?_ ?_ ?_
    · have aux := IsPrimitive.Int.irreducible_iff_irreducible_map_cast Pmonic.isPrimitive
      refine (dvd_or_coprime _ _ (aux.1 Pirr)).resolve_left ?_
      rw [map_dvd_map (Int.castRingHom ℚ) Int.cast_injective Pmonic]
      intro hdiv
      refine hdiff (eq_of_monic_of_associated Pmonic Qmonic ?_)
      exact associated_of_dvd_dvd hdiv (Pirr.dvd_symm Qirr hdiv)
    · apply (map_dvd_map (Int.castRingHom ℚ) Int.cast_injective Pmonic).2
      exact minpoly_dvd_x_pow_sub_one h
    · apply (map_dvd_map (Int.castRingHom ℚ) Int.cast_injective Qmonic).2
      exact minpoly_dvd_x_pow_sub_one (pow_of_prime h hprime.1 hdiv)
  replace prod := RingHom.map_dvd (mapRingHom (Int.castRingHom (ZMod p))) prod
  rw [coe_mapRingHom, Polynomial.map_mul, Polynomial.map_sub, Polynomial.map_one,
    Polynomial.map_pow, map_X] at prod
  obtain ⟨R, hR⟩ := minpoly_dvd_mod_p h hdiv
  rw [hR, ← mul_assoc, ← Polynomial.map_mul, ← sq, Polynomial.map_pow] at prod
  have habs : map (Int.castRingHom (ZMod p)) P ^ 2 ∣ map (Int.castRingHom (ZMod p)) P ^ 2 * R := by
    use R
  replace habs :=
    lt_of_lt_of_le (Nat.cast_lt.2 one_lt_two)
      (le_emultiplicity_of_pow_dvd (dvd_trans habs prod))
  have hfree : Squarefree (X ^ n - 1 : (ZMod p)[X]) :=
    (separable_X_pow_sub_C 1 (fun h => hdiv <| (ZMod.natCast_zmod_eq_zero_iff_dvd n p).1 h)
        one_ne_zero).squarefree
  cases'
    (squarefree_iff_emultiplicity_le_one (X ^ n - 1)).1 hfree
      (map (Int.castRingHom (ZMod p)) P) with
    hle hunit
  · rw [Nat.cast_one] at habs; exact hle.not_lt habs
  · replace hunit := degree_eq_zero_of_isUnit hunit
    rw [degree_map_eq_of_leadingCoeff_ne_zero (Int.castRingHom (ZMod p)) _] at hunit
    · exact (minpoly.degree_pos (isIntegral h hpos)).ne' hunit
    simp only [Pmonic, eq_intCast, Monic.leadingCoeff, Int.cast_one, Ne, not_false_iff,
      one_ne_zero]


/-- If `m : ℕ` is coprime with `n`,
then the minimal polynomials of a primitive `n`-th root of unity `μ`
and of `μ ^ m` are the same. -/
theorem minpoly_eq_pow_coprime {m : ℕ} (hcop : Nat.Coprime m n) :
    minpoly ℤ μ = minpoly ℤ (μ ^ m) := by
  /-
    n : Nat
    K : Type u_1
    inst✝² : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    m : Nat
    hcop : m.Coprime n
    ⊢ Eq (minpoly Int μ) (minpoly Int (HPow.hPow μ m))
  -/
  revert n hcop
  /-
    K : Type u_1
    inst✝² : CommRing K
    μ : K
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    m : Nat
    ⊢ ∀ {n : Nat}, IsPrimitiveRoot μ n → m.Coprime n → Eq (minpoly Int μ) (minpoly …
  -/
  refine UniqueFactorizationMonoid.induction_on_prime m ?_ ?_ ?_
    /-
      case refine_1
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      m n✝ : Nat
      ⊢ IsPrimitiveRoot μ n✝ → Nat.Coprime 0 n✝ → Eq (minpoly Int μ) (minpoly Int (H …
    -/
  · intro h hn
    /-
      case refine_1
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      m n✝ : Nat
      h : IsPrimitiveRoot μ n✝
      hn : Nat.Coprime 0 n✝
      ⊢ Eq (minpoly Int μ) (minpoly Int (HPow.hPow μ 0))
    -/
    congr
    /-
      case refine_1.e_x
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      m n✝ : Nat
      h : IsPrimitiveRoot μ n✝
      hn : Nat.Coprime 0 n✝
      ⊢ Eq μ (HPow.hPow μ 0)
    -/
    simpa [(Nat.coprime_zero_left _).mp hn] using h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      m n✝ : Nat
      ⊢ ∀ (x : Nat), IsUnit x → IsPrimitiveRoot μ n✝ → x.Coprime n✝ → Eq (minpoly In …
    -/
  · intro u hunit _ _
    /-
      case refine_2
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      m n✝ u : Nat
      hunit : IsUnit u
      h✝ : IsPrimitiveRoot μ n✝
      hcop✝ : u.Coprime n✝
      ⊢ Eq (minpoly Int μ) (minpoly Int (HPow.hPow μ u))
    -/
    congr
    /-
      case refine_2.e_x
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      m n✝ u : Nat
      hunit : IsUnit u
      h✝ : IsPrimitiveRoot μ n✝
      hcop✝ : u.Coprime n✝
      ⊢ Eq μ (HPow.hPow μ u)
    -/
    simp [Nat.isUnit_iff.mp hunit]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      m n✝ : Nat
      ⊢ ∀ (a p : Nat), Ne a 0 → Prime p → (IsPrimitiveRoot μ n✝ → a.Coprime n✝ → Eq  …
    -/
  · intro a p _ hprime
    /-
      case refine_3
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      m n✝ a p : Nat
      a✝ : Ne a 0
      hprime : Prime p
      ⊢ (IsPrimitiveRoot μ n✝ → a.Coprime n✝ → Eq (minpoly Int μ) (minpoly Int (HPow …
    -/
    intro hind h hcop
    /-
      case refine_3
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      m n✝ a p : Nat
      a✝ : Ne a 0
      hprime : Prime p
      hind : IsPrimitiveRoot μ n✝ → a.Coprime n✝ → Eq (minpoly Int μ) (minpoly Int ( …
      h : IsPrimitiveRoot μ n✝
      hcop : (HMul.hMul p a).Coprime n✝
      ⊢ Eq (minpoly Int μ) (minpoly Int (HPow.hPow μ (HMul.hMul p a)))
    -/
    rw [hind h (Nat.Coprime.coprime_mul_left hcop)]; clear hind
    /-
      case refine_3
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      m n✝ a p : Nat
      a✝ : Ne a 0
      hprime : Prime p
      h : IsPrimitiveRoot μ n✝
      hcop : (HMul.hMul p a).Coprime n✝
      ⊢ Eq (minpoly Int (HPow.hPow μ a)) (minpoly Int (HPow.hPow μ (HMul.hMul p a)))
    -/
    replace hprime := hprime.nat_prime
    /-
      case refine_3
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      m n✝ a p : Nat
      a✝ : Ne a 0
      h : IsPrimitiveRoot μ n✝
      hcop : (HMul.hMul p a).Coprime n✝
      hprime : Nat.Prime p
      ⊢ Eq (minpoly Int (HPow.hPow μ a)) (minpoly Int (HPow.hPow μ (HMul.hMul p a)))
    -/
    have hdiv := (Nat.Prime.coprime_iff_not_dvd hprime).1 (Nat.Coprime.coprime_mul_right hcop)
    /-
      case refine_3
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      m n✝ a p : Nat
      a✝ : Ne a 0
      h : IsPrimitiveRoot μ n✝
      hcop : (HMul.hMul p a).Coprime n✝
      hprime : Nat.Prime p
      hdiv : Not (Dvd.dvd p n✝)
      ⊢ Eq (minpoly Int (HPow.hPow μ a)) (minpoly Int (HPow.hPow μ (HMul.hMul p a)))
    -/
    haveI := Fact.mk hprime
    /-
      case refine_3
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      m n✝ a p : Nat
      a✝ : Ne a 0
      h : IsPrimitiveRoot μ n✝
      hcop : (HMul.hMul p a).Coprime n✝
      hprime : Nat.Prime p
      hdiv : Not (Dvd.dvd p n✝)
      this : Fact (Nat.Prime p)
      ⊢ Eq (minpoly Int (HPow.hPow μ a)) (minpoly Int (HPow.hPow μ (HMul.hMul p a)))
    -/
    rw [minpoly_eq_pow (h.pow_of_coprime a (Nat.Coprime.coprime_mul_left hcop)) hdiv]
    /-
      case refine_3
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      m n✝ a p : Nat
      a✝ : Ne a 0
      h : IsPrimitiveRoot μ n✝
      hcop : (HMul.hMul p a).Coprime n✝
      hprime : Nat.Prime p
      hdiv : Not (Dvd.dvd p n✝)
      this : Fact (Nat.Prime p)
      ⊢ Eq (minpoly Int (HPow.hPow (HPow.hPow μ a) p)) (minpoly Int (HPow.hPow μ (HM …
    -/
    congr 1
    /-
      case refine_3.e_x
      K : Type u_1
      inst✝² : CommRing K
      μ : K
      inst✝¹ : IsDomain K
      inst✝ : CharZero K
      m n✝ a p : Nat
      a✝ : Ne a 0
      h : IsPrimitiveRoot μ n✝
      hcop : (HMul.hMul p a).Coprime n✝
      hprime : Nat.Prime p
      hdiv : Not (Dvd.dvd p n✝)
      this : Fact (Nat.Prime p)
      ⊢ Eq (HPow.hPow (HPow.hPow μ a) p) (HPow.hPow μ (HMul.hMul p a))
    -/
    ring
    /-
      🎉 no goals
    -/


/-- If `m : ℕ` is coprime with `n`,
then the minimal polynomial of a primitive `n`-th root of unity `μ`
has `μ ^ m` as root. -/
theorem pow_isRoot_minpoly {m : ℕ} (hcop : Nat.Coprime m n) :
    IsRoot (map (Int.castRingHom K) (minpoly ℤ μ)) (μ ^ m) := by
  /-
    n : Nat
    K : Type u_1
    inst✝² : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    m : Nat
    hcop : m.Coprime n
    ⊢ (Polynomial.map (Int.castRingHom K) (minpoly Int μ)).IsRoot (HPow.hPow μ m)
  -/
  simp only [minpoly_eq_pow_coprime h hcop, IsRoot.def, eval_map]
  /-
    n : Nat
    K : Type u_1
    inst✝² : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝¹ : IsDomain K
    inst✝ : CharZero K
    m : Nat
    hcop : m.Coprime n
    ⊢ Eq (Polynomial.eval₂ (Int.castRingHom K) (HPow.hPow μ m) (minpoly Int (HPow. …
  -/
  exact minpoly.aeval ℤ (μ ^ m)
  /-
    🎉 no goals
  -/


/-- `primitiveRoots n K` is a subset of the roots of the minimal polynomial of a primitive
`n`-th root of unity `μ`. -/
theorem is_roots_of_minpoly [DecidableEq K] :
    primitiveRoots n K ⊆ (map (Int.castRingHom K) (minpoly ℤ μ)).roots.toFinset := by
  /-
    n : Nat
    K : Type u_1
    inst✝³ : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝² : IsDomain K
    inst✝¹ : CharZero K
    inst✝ : DecidableEq K
    ⊢ HasSubset.Subset (primitiveRoots n K) (Polynomial.map (Int.castRingHom K) (m …
  -/
  by_cases hn : n = 0; · simp_all
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    n : Nat
    K : Type u_1
    inst✝³ : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝² : IsDomain K
    inst✝¹ : CharZero K
    inst✝ : DecidableEq K
    hn : Not (Eq n 0)
    ⊢ HasSubset.Subset (primitiveRoots n K) (Polynomial.map (Int.castRingHom K) (m …
  -/
  have : NeZero n := ⟨hn⟩
  /-
    case neg
    n : Nat
    K : Type u_1
    inst✝³ : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝² : IsDomain K
    inst✝¹ : CharZero K
    inst✝ : DecidableEq K
    hn : Not (Eq n 0)
    this : NeZero n
    ⊢ HasSubset.Subset (primitiveRoots n K) (Polynomial.map (Int.castRingHom K) (m …
  -/
  have hpos := Nat.pos_of_ne_zero hn
  /-
    case neg
    n : Nat
    K : Type u_1
    inst✝³ : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝² : IsDomain K
    inst✝¹ : CharZero K
    inst✝ : DecidableEq K
    hn : Not (Eq n 0)
    this : NeZero n
    hpos : LT.lt 0 n
    ⊢ HasSubset.Subset (primitiveRoots n K) (Polynomial.map (Int.castRingHom K) (m …
  -/
  intro x hx
  /-
    case neg
    n : Nat
    K : Type u_1
    inst✝³ : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝² : IsDomain K
    inst✝¹ : CharZero K
    inst✝ : DecidableEq K
    hn : Not (Eq n 0)
    this : NeZero n
    hpos : LT.lt 0 n
    x : K
    hx : Membership.mem (primitiveRoots n K) x
    ⊢ Membership.mem (Polynomial.map (Int.castRingHom K) (minpoly Int μ)).roots.to …
  -/
  obtain ⟨m, _, hcop, rfl⟩ := (isPrimitiveRoot_iff h).1 ((mem_primitiveRoots hpos).1 hx)
  /-
    case neg.intro.intro.intro
    n : Nat
    K : Type u_1
    inst✝³ : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝² : IsDomain K
    inst✝¹ : CharZero K
    inst✝ : DecidableEq K
    hn : Not (Eq n 0)
    this : NeZero n
    hpos : LT.lt 0 n
    m : Nat
    left✝ : LT.lt m n
    hcop : m.Coprime n
    hx : Membership.mem (primitiveRoots n K) (HPow.hPow μ m)
    ⊢ Membership.mem (Polynomial.map (Int.castRingHom K) (minpoly Int μ)).roots.to …
  -/
  simp only [Multiset.mem_toFinset, mem_roots]
  /-
    case neg.intro.intro.intro
    n : Nat
    K : Type u_1
    inst✝³ : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝² : IsDomain K
    inst✝¹ : CharZero K
    inst✝ : DecidableEq K
    hn : Not (Eq n 0)
    this : NeZero n
    hpos : LT.lt 0 n
    m : Nat
    left✝ : LT.lt m n
    hcop : m.Coprime n
    hx : Membership.mem (primitiveRoots n K) (HPow.hPow μ m)
    ⊢ Membership.mem (Polynomial.map (Int.castRingHom K) (minpoly Int μ)).roots (H …
  -/
  convert pow_isRoot_minpoly h hcop using 0
  /-
    case a
    n : Nat
    K : Type u_1
    inst✝³ : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝² : IsDomain K
    inst✝¹ : CharZero K
    inst✝ : DecidableEq K
    hn : Not (Eq n 0)
    this : NeZero n
    hpos : LT.lt 0 n
    m : Nat
    left✝ : LT.lt m n
    hcop : m.Coprime n
    hx : Membership.mem (primitiveRoots n K) (HPow.hPow μ m)
    ⊢ Iff (Membership.mem (Polynomial.map (Int.castRingHom K) (minpoly Int μ)).roo …
  -/
  rw [← mem_roots]
  /-
    case a
    n : Nat
    K : Type u_1
    inst✝³ : CommRing K
    μ : K
    h : IsPrimitiveRoot μ n
    inst✝² : IsDomain K
    inst✝¹ : CharZero K
    inst✝ : DecidableEq K
    hn : Not (Eq n 0)
    this : NeZero n
    hpos : LT.lt 0 n
    m : Nat
    left✝ : LT.lt m n
    hcop : m.Coprime n
    hx : Membership.mem (primitiveRoots n K) (HPow.hPow μ m)
    ⊢ Ne (Polynomial.map (Int.castRingHom K) (minpoly Int μ)) 0
  -/
  exact map_monic_ne_zero <| minpoly.monic <| isIntegral h hpos
  /-
    🎉 no goals
  -/


/-- The degree of the minimal polynomial of `μ` is at least `totient n`. -/
theorem totient_le_degree_minpoly : Nat.totient n ≤ (minpoly ℤ μ).natDegree := by
  classical
  let P : ℤ[X] := minpoly ℤ μ
  -- minimal polynomial of `μ`
  let P_K : K[X] := map (Int.castRingHom K) P
  -- minimal polynomial of `μ` sent to `K[X]`
  calc
    n.totient = (primitiveRoots n K).card := h.card_primitiveRoots.symm
    _ ≤ P_K.roots.toFinset.card := Finset.card_le_card (is_roots_of_minpoly h)
    _ ≤ Multiset.card P_K.roots := Multiset.toFinset_card_le _
    _ ≤ P_K.natDegree := card_roots' _
    _ ≤ P.natDegree := natDegree_map_le


