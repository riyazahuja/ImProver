/-- Statement of Fermat's Last Theorem over a given semiring with a specific exponent. -/
def FermatLastTheoremWith (α : Type*) [Semiring α] (n : ℕ) : Prop :=
  ∀ a b c : α, a ≠ 0 → b ≠ 0 → c ≠ 0 → a ^ n + b ^ n ≠ c ^ n


/-- Statement of Fermat's Last Theorem over the naturals for a given exponent. -/
def FermatLastTheoremFor (n : ℕ) : Prop := FermatLastTheoremWith ℕ n


/-- Statement of Fermat's Last Theorem: `a ^ n + b ^ n = c ^ n` has no nontrivial natural solution
when `n ≥ 3`.

This is now a theorem of Wiles and Taylor--Wiles; see
https://github.com/ImperialCollegeLondon/FLT for an ongoing Lean formalisation of
a proof. -/
def FermatLastTheorem : Prop := ∀ n ≥ 3, FermatLastTheoremFor n


lemma fermatLastTheoremFor_zero : FermatLastTheoremFor 0 :=
                       /-
                         x✝⁵ x✝⁴ x✝³ : Nat
                         x✝² : Ne x✝⁵ 0
                         x✝¹ : Ne x✝⁴ 0
                         x✝ : Ne x✝³ 0
                         ⊢ Ne (HAdd.hAdd (HPow.hPow x✝⁵ 0) (HPow.hPow x✝⁴ 0)) (HPow.hPow x✝³ 0)
                       -/
  fun _ _ _ _ _ _ ↦ by norm_num
                       /-
                         🎉 no goals
                       -/


lemma not_fermatLastTheoremFor_one : ¬ FermatLastTheoremFor 1 :=
                      /-
                        h : FermatLastTheoremFor 1
                        ⊢ Ne 1 0
                      -/
                      /-
                        🎉 no goals
                      -/
                                    /-
                                      🎉 no goals
                                    -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  fun h ↦ h 1 1 2 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
                                                                /-
                                                                  🎉 no goals
                                                                -/


lemma not_fermatLastTheoremFor_two : ¬ FermatLastTheoremFor 2 :=
                      /-
                        h : FermatLastTheoremFor 2
                        ⊢ Ne 3 0
                      -/
                      /-
                        🎉 no goals
                      -/
                                    /-
                                      🎉 no goals
                                    -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  fun h ↦ h 3 4 5 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
                                                                /-
                                                                  🎉 no goals
                                                                -/


lemma FermatLastTheoremWith.mono (hmn : m ∣ n) (hm : FermatLastTheoremWith α m) :
    FermatLastTheoremWith α n := by
  /-
    α : Type u_1
    inst✝¹ : Semiring α
    inst✝ : NoZeroDivisors α
    m n : Nat
    hmn : Dvd.dvd m n
    hm : FermatLastTheoremWith α m
    ⊢ FermatLastTheoremWith α n
  -/
  rintro a b c ha hb hc
  /-
    α : Type u_1
    inst✝¹ : Semiring α
    inst✝ : NoZeroDivisors α
    m n : Nat
    hmn : Dvd.dvd m n
    hm : FermatLastTheoremWith α m
    a b c : α
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    ⊢ Ne (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n)
  -/
  obtain ⟨k, rfl⟩ := hmn
  /-
    case intro
    α : Type u_1
    inst✝¹ : Semiring α
    inst✝ : NoZeroDivisors α
    m : Nat
    hm : FermatLastTheoremWith α m
    a b c : α
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    k : Nat
    ⊢ Ne (HAdd.hAdd (HPow.hPow a (HMul.hMul m k)) (HPow.hPow b (HMul.hMul m k))) ( …
  -/
  simp_rw [pow_mul']
  /-
    case intro
    α : Type u_1
    inst✝¹ : Semiring α
    inst✝ : NoZeroDivisors α
    m : Nat
    hm : FermatLastTheoremWith α m
    a b c : α
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    k : Nat
    ⊢ Ne (HAdd.hAdd (HPow.hPow (HPow.hPow a k) m) (HPow.hPow (HPow.hPow b k) m)) ( …
  -/
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 🎉 no goals
                               -/
  refine hm _ _ _ ?_ ?_ ?_ <;> exact pow_ne_zero _ ‹_›
                               /-
                                 🎉 no goals
                               -/


lemma FermatLastTheoremFor.mono (hmn : m ∣ n) (hm : FermatLastTheoremFor m) :
    FermatLastTheoremFor n := by
  /-
    m n : Nat
    hmn : Dvd.dvd m n
    hm : FermatLastTheoremFor m
    ⊢ FermatLastTheoremFor n
  -/
  exact FermatLastTheoremWith.mono hmn hm
  /-
    🎉 no goals
  -/


lemma fermatLastTheoremWith_nat_int_rat_tfae (n : ℕ) :
    TFAE [FermatLastTheoremWith ℕ n, FermatLastTheoremWith ℤ n, FermatLastTheoremWith ℚ n] := by
  tfae_have 1 → 2
  | h, a, b, c, ha, hb, hc, habc => by
    obtain hn | hn := n.even_or_odd
    · refine h a.natAbs b.natAbs c.natAbs (by positivity) (by positivity) (by positivity)
        (Int.natCast_inj.1 ?_)
      push_cast
      simp only [hn.pow_abs, habc]
    obtain ha | ha := ha.lt_or_lt <;> obtain hb | hb := hb.lt_or_lt <;>
      obtain hc | hc := hc.lt_or_lt
    · refine h a.natAbs b.natAbs c.natAbs (by positivity) (by positivity) (by positivity)
        (Int.natCast_inj.1 ?_)
      push_cast
      simp only [abs_of_neg, neg_pow a, neg_pow b, neg_pow c, ← mul_add, habc, *]
    · exact (by positivity : 0 < c ^ n).not_lt <| habc.symm.trans_lt <| add_neg (hn.pow_neg ha) <|
        hn.pow_neg hb
    · refine h b.natAbs c.natAbs a.natAbs (by positivity) (by positivity) (by positivity)
        (Int.natCast_inj.1 ?_)
      push_cast
      simp only [abs_of_pos, abs_of_neg, hn.neg_pow, habc, add_neg_eq_iff_eq_add,
        eq_neg_add_iff_add_eq, *]
    · refine h a.natAbs c.natAbs b.natAbs (by positivity) (by positivity) (by positivity)
        (Int.natCast_inj.1 ?_)
      push_cast
      simp only [abs_of_pos, abs_of_neg, hn.neg_pow, habc, neg_add_eq_iff_eq_add,
        eq_neg_add_iff_add_eq, *]
    · refine h c.natAbs a.natAbs b.natAbs (by positivity) (by positivity) (by positivity)
        (Int.natCast_inj.1 ?_)
      push_cast
      simp only [abs_of_pos, abs_of_neg, hn.neg_pow, habc, neg_add_eq_iff_eq_add,
        eq_add_neg_iff_add_eq, *]
    · refine h c.natAbs b.natAbs a.natAbs (by positivity) (by positivity) (by positivity)
        (Int.natCast_inj.1 ?_)
      push_cast
      simp only [abs_of_pos, abs_of_neg, hn.neg_pow, habc, add_neg_eq_iff_eq_add,
        eq_add_neg_iff_add_eq, *]
    · exact (by positivity : 0 < a ^ n + b ^ n).not_lt <| habc.trans_lt <| hn.pow_neg hc
    · refine h a.natAbs b.natAbs c.natAbs (by positivity) (by positivity) (by positivity)
        (Int.natCast_inj.1 ?_)
      push_cast
      simp only [abs_of_pos, habc, *]
  tfae_have 2 → 3
  | h, a, b, c, ha, hb, hc, habc => by
    rw [← Rat.num_ne_zero] at ha hb hc
    refine h (a.num * b.den * c.den) (a.den * b.num * c.den) (a.den * b.den * c.num)
      (by positivity) (by positivity) (by positivity) ?_
    have : (a.den * b.den * c.den : ℚ) ^ n ≠ 0 := by positivity
    refine Int.cast_injective <| (div_left_inj' this).1 ?_
    push_cast
    simp only [add_div, ← div_pow, mul_div_mul_comm, div_self (by positivity : (a.den : ℚ) ≠ 0),
      div_self (by positivity : (b.den : ℚ) ≠ 0), div_self (by positivity : (c.den : ℚ) ≠ 0),
      one_mul, mul_one, Rat.num_div_den, habc]
  tfae_have 3 → 1
  | h, a, b, c => mod_cast h a b c
  /-
    n : Nat
    tfae_1_to_2 : FermatLastTheoremWith Nat n → FermatLastTheoremWith Int n
    tfae_2_to_3 : FermatLastTheoremWith Int n → FermatLastTheoremWith Rat n
    tfae_3_to_1 : FermatLastTheoremWith Rat n → FermatLastTheoremWith Nat n
    ⊢ (List.cons (FermatLastTheoremWith Nat n) (List.cons (FermatLastTheoremWith I …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


lemma fermatLastTheoremFor_iff_nat {n : ℕ} : FermatLastTheoremFor n ↔ FermatLastTheoremWith ℕ n :=
  Iff.rfl


lemma fermatLastTheoremFor_iff_int {n : ℕ} : FermatLastTheoremFor n ↔ FermatLastTheoremWith ℤ n :=
  /-
    n : Nat
    ⊢ Eq ((List.cons (FermatLastTheoremWith Nat n) (List.cons (FermatLastTheoremWi …
  -/
  /-
    🎉 no goals
  -/
  (fermatLastTheoremWith_nat_int_rat_tfae n).out 0 1
  /-
    🎉 no goals
  -/


lemma fermatLastTheoremFor_iff_rat {n : ℕ} : FermatLastTheoremFor n ↔ FermatLastTheoremWith ℚ n :=
  /-
    n : Nat
    ⊢ Eq ((List.cons (FermatLastTheoremWith Nat n) (List.cons (FermatLastTheoremWi …
  -/
  /-
    🎉 no goals
  -/
  (fermatLastTheoremWith_nat_int_rat_tfae n).out 0 2
  /-
    🎉 no goals
  -/


/--
A relaxed variant of Fermat's Last Theorem over a given commutative semiring with a specific
exponent, allowing nonzero solutions of units and their common multiples.

1. The variant `FermatLastTheoremWith' α` is weaker than `FermatLastTheoremWith α` in general.
   In particular, it holds trivially for `[Field α]`.
2. This variant is equivalent to the original `FermatLastTheoremWith α` for `α = ℕ` or `ℤ`.
   In general, they are equivalent if there is no solutions of units to the Fermat equation.
3. For a polynomial ring `α = k[X]`, the original `FermatLastTheoremWith α` is false but the weaker
   variant `FermatLastTheoremWith' α` is true. This polynomial variant of Fermat's Last Theorem
   can be shown elementarily using Mason--Stothers theorem.
-/
def FermatLastTheoremWith' (α : Type*) [CommSemiring α] (n : ℕ) : Prop :=
  ∀ a b c : α, a ≠ 0 → b ≠ 0 → c ≠ 0 → a ^ n + b ^ n = c ^ n →
    ∃ d a' b' c', (a = a' * d ∧ b = b' * d ∧ c = c' * d) ∧ (IsUnit a' ∧ IsUnit b' ∧ IsUnit c')


lemma FermatLastTheoremWith.fermatLastTheoremWith' {α : Type*} [CommSemiring α] {n : ℕ}
    (h : FermatLastTheoremWith α n) : FermatLastTheoremWith' α n :=
                         /-
                           α : Type u_2
                           inst✝ : CommSemiring α
                           n : Nat
                           h : FermatLastTheoremWith α n
                           a b c : α
                           x✝³ : Ne a 0
                           x✝² : Ne b 0
                           x✝¹ : Ne c 0
                           x✝ : Eq (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n)
                           ⊢ Exists fun d => Exists fun a' => Exists fun b' => Exists fun c' => And (And  …
                         -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  fun a b c _ _ _ _ ↦ by exfalso; apply h a b c <;> assumption
                                                    /-
                                                      🎉 no goals
                                                    -/


lemma fermatLastTheoremWith'_of_field (α : Type*) [Field α] (n : ℕ) : FermatLastTheoremWith' α n :=
  fun a b c ha hb hc _ ↦
    ⟨1, a, b, c,
     ⟨(mul_one a).symm, (mul_one b).symm, (mul_one c).symm⟩,
     ⟨ha.isUnit, hb.isUnit, hc.isUnit⟩⟩


lemma FermatLastTheoremWith'.fermatLastTheoremWith {α : Type*} [CommSemiring α] [IsDomain α]
    {n : ℕ} (h : FermatLastTheoremWith' α n)
    (hn : ∀ a b c : α, IsUnit a → IsUnit b → IsUnit c → a ^ n + b ^ n ≠ c ^ n) :
    FermatLastTheoremWith α n := by
  /-
    α : Type u_2
    inst✝¹ : CommSemiring α
    inst✝ : IsDomain α
    n : Nat
    h : FermatLastTheoremWith' α n
    hn : ∀ (a b c : α), IsUnit a → IsUnit b → IsUnit c → Ne (HAdd.hAdd (HPow.hPow  …
    ⊢ FermatLastTheoremWith α n
  -/
  intro a b c ha hb hc heq
  /-
    α : Type u_2
    inst✝¹ : CommSemiring α
    inst✝ : IsDomain α
    n : Nat
    h : FermatLastTheoremWith' α n
    hn : ∀ (a b c : α), IsUnit a → IsUnit b → IsUnit c → Ne (HAdd.hAdd (HPow.hPow  …
    a b c : α
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    heq : Eq (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n)
    ⊢ False
  -/
  rcases h a b c ha hb hc heq with ⟨d, a', b', c', ⟨rfl, rfl, rfl⟩, ⟨ua, ub, uc⟩⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_2
    inst✝¹ : CommSemiring α
    inst✝ : IsDomain α
    n : Nat
    h : FermatLastTheoremWith' α n
    hn : ∀ (a b c : α), IsUnit a → IsUnit b → IsUnit c → Ne (HAdd.hAdd (HPow.hPow  …
    d a' b' c' : α
    ha : Ne (HMul.hMul a' d) 0
    hb : Ne (HMul.hMul b' d) 0
    hc : Ne (HMul.hMul c' d) 0
    heq : Eq (HAdd.hAdd (HPow.hPow (HMul.hMul a' d) n) (HPow.hPow (HMul.hMul b' d) …
    ua : IsUnit a'
    ub : IsUnit b'
    uc : IsUnit c'
    ⊢ False
  -/
  rw [mul_pow, mul_pow, mul_pow, ← add_mul] at heq
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_2
    inst✝¹ : CommSemiring α
    inst✝ : IsDomain α
    n : Nat
    h : FermatLastTheoremWith' α n
    hn : ∀ (a b c : α), IsUnit a → IsUnit b → IsUnit c → Ne (HAdd.hAdd (HPow.hPow  …
    d a' b' c' : α
    ha : Ne (HMul.hMul a' d) 0
    hb : Ne (HMul.hMul b' d) 0
    hc : Ne (HMul.hMul c' d) 0
    heq : Eq (HMul.hMul (HAdd.hAdd (HPow.hPow a' n) (HPow.hPow b' n)) (HPow.hPow d …
    ua : IsUnit a'
    ub : IsUnit b'
    uc : IsUnit c'
    ⊢ False
  -/
  exact hn _ _ _ ua ub uc <| mul_right_cancel₀ (pow_ne_zero _ (right_ne_zero_of_mul ha)) heq
  /-
    🎉 no goals
  -/


lemma fermatLastTheoremWith'_iff_fermatLastTheoremWith {α : Type*} [CommSemiring α] [IsDomain α]
    {n : ℕ} (hn : ∀ a b c : α, IsUnit a → IsUnit b → IsUnit c → a ^ n + b ^ n ≠ c ^ n) :
    FermatLastTheoremWith' α n ↔ FermatLastTheoremWith α n :=
  Iff.intro (fun h ↦ h.fermatLastTheoremWith hn) (fun h ↦ h.fermatLastTheoremWith')


lemma fermatLastTheoremWith'_nat_int_tfae (n : ℕ) :
    TFAE [FermatLastTheoremFor n, FermatLastTheoremWith' ℕ n, FermatLastTheoremWith' ℤ n] := by
  tfae_have 2 ↔ 1 := by
    apply fermatLastTheoremWith'_iff_fermatLastTheoremWith
    simp only [Nat.isUnit_iff]
    intro _ _ _ ha hb hc
    rw [ha, hb, hc]
    simp only [one_pow, Nat.reduceAdd, ne_eq, OfNat.ofNat_ne_one, not_false_eq_true]
  tfae_have 3 ↔ 1 := by
    rw [fermatLastTheoremFor_iff_int]
    apply fermatLastTheoremWith'_iff_fermatLastTheoremWith
    intro a b c ha hb hc
    by_cases hn : n = 0
    · subst hn
      simp only [pow_zero, Int.reduceAdd, ne_eq, OfNat.ofNat_ne_one, not_false_eq_true]
    · rw [← isUnit_pow_iff hn, Int.isUnit_iff] at ha hb hc
      -- case division
      rcases ha with ha | ha <;> rcases hb with hb | hb <;> rcases hc with hc | hc <;>
        rw [ha, hb, hc] <;> decide
  /-
    n : Nat
    tfae_2_iff_1 : Iff (FermatLastTheoremWith' Nat n) (FermatLastTheoremFor n)
    tfae_3_iff_1 : Iff (FermatLastTheoremWith' Int n) (FermatLastTheoremFor n)
    ⊢ (List.cons (FermatLastTheoremFor n) (List.cons (FermatLastTheoremWith' Nat n …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


open Finset in
/-- To prove Fermat Last Theorem in any semiring that is a `NormalizedGCDMonoid` one can assume
that the `gcd` of `{a, b, c}` is `1`. -/
lemma fermatLastTheoremWith_of_fermatLastTheoremWith_coprime {n : ℕ} {R : Type*} [CommSemiring R]
    [IsDomain R] [DecidableEq R] [NormalizedGCDMonoid R]
    (hn : ∀ a b c : R, a ≠ 0 → b ≠ 0 → c ≠ 0 → ({a, b, c} : Finset R).gcd id = 1 →
      a ^ n + b ^ n ≠ c ^ n) :
    FermatLastTheoremWith R n := by
  /-
    n : Nat
    R : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsDomain R
    inst✝¹ : DecidableEq R
    inst✝ : NormalizedGCDMonoid R
    hn : ∀ (a b c : R), Ne a 0 → Ne b 0 → Ne c 0 → Eq ((Insert.insert a (Insert.in …
    ⊢ FermatLastTheoremWith R n
  -/
  intro a b c ha hb hc habc
  /-
    n : Nat
    R : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsDomain R
    inst✝¹ : DecidableEq R
    inst✝ : NormalizedGCDMonoid R
    hn : ∀ (a b c : R), Ne a 0 → Ne b 0 → Ne c 0 → Eq ((Insert.insert a (Insert.in …
    a b c : R
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    habc : Eq (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n)
    ⊢ False
  -/
  let s : Finset R := {a, b, c}; let d := s.gcd id
  /-
    n : Nat
    R : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsDomain R
    inst✝¹ : DecidableEq R
    inst✝ : NormalizedGCDMonoid R
    hn : ∀ (a b c : R), Ne a 0 → Ne b 0 → Ne c 0 → Eq ((Insert.insert a (Insert.in …
    a b c : R
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    habc : Eq (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n)
    s : Finset R := Insert.insert a (Insert.insert b (Singleton.singleton c))
    d : R := s.gcd id
    ⊢ False
  -/
  obtain ⟨A, hA⟩ : d ∣ a := gcd_dvd (by simp [s])
  /-
    case intro
    n : Nat
    R : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsDomain R
    inst✝¹ : DecidableEq R
    inst✝ : NormalizedGCDMonoid R
    hn : ∀ (a b c : R), Ne a 0 → Ne b 0 → Ne c 0 → Eq ((Insert.insert a (Insert.in …
    a b c : R
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    habc : Eq (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n)
    s : Finset R := Insert.insert a (Insert.insert b (Singleton.singleton c))
    d : R := s.gcd id
    A : R
    hA : Eq a (HMul.hMul d A)
    ⊢ False
  -/
  obtain ⟨B, hB⟩ : d ∣ b := gcd_dvd (by simp [s])
  /-
    case intro.intro
    n : Nat
    R : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsDomain R
    inst✝¹ : DecidableEq R
    inst✝ : NormalizedGCDMonoid R
    hn : ∀ (a b c : R), Ne a 0 → Ne b 0 → Ne c 0 → Eq ((Insert.insert a (Insert.in …
    a b c : R
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    habc : Eq (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n)
    s : Finset R := Insert.insert a (Insert.insert b (Singleton.singleton c))
    d : R := s.gcd id
    A : R
    hA : Eq a (HMul.hMul d A)
    B : R
    hB : Eq b (HMul.hMul d B)
    ⊢ False
  -/
  obtain ⟨C, hC⟩ : d ∣ c := gcd_dvd (by simp [s])
  /-
    case intro.intro.intro
    n : Nat
    R : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsDomain R
    inst✝¹ : DecidableEq R
    inst✝ : NormalizedGCDMonoid R
    hn : ∀ (a b c : R), Ne a 0 → Ne b 0 → Ne c 0 → Eq ((Insert.insert a (Insert.in …
    a b c : R
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    habc : Eq (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n)
    s : Finset R := Insert.insert a (Insert.insert b (Singleton.singleton c))
    d : R := s.gcd id
    A : R
    hA : Eq a (HMul.hMul d A)
    B : R
    hB : Eq b (HMul.hMul d B)
    C : R
    hC : Eq c (HMul.hMul d C)
    ⊢ False
  -/
  simp only [hA, hB, hC, mul_ne_zero_iff, mul_pow] at ha hb hc habc
  /-
    case intro.intro.intro
    n : Nat
    R : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsDomain R
    inst✝¹ : DecidableEq R
    inst✝ : NormalizedGCDMonoid R
    hn : ∀ (a b c : R), Ne a 0 → Ne b 0 → Ne c 0 → Eq ((Insert.insert a (Insert.in …
    a b c : R
    s : Finset R := Insert.insert a (Insert.insert b (Singleton.singleton c))
    d : R := s.gcd id
    A : R
    hA : Eq a (HMul.hMul d A)
    B : R
    hB : Eq b (HMul.hMul d B)
    C : R
    hC : Eq c (HMul.hMul d C)
    ha : And (Ne d 0) (Ne A 0)
    hb : And (Ne d 0) (Ne B 0)
    hc : And (Ne d 0) (Ne C 0)
    habc : Eq (HAdd.hAdd (HMul.hMul (HPow.hPow d n) (HPow.hPow A n)) (HMul.hMul (H …
    ⊢ False
  -/
  rw [← mul_add, mul_right_inj' (pow_ne_zero n ha.1)] at habc
  /-
    case intro.intro.intro
    n : Nat
    R : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsDomain R
    inst✝¹ : DecidableEq R
    inst✝ : NormalizedGCDMonoid R
    hn : ∀ (a b c : R), Ne a 0 → Ne b 0 → Ne c 0 → Eq ((Insert.insert a (Insert.in …
    a b c : R
    s : Finset R := Insert.insert a (Insert.insert b (Singleton.singleton c))
    d : R := s.gcd id
    A : R
    hA : Eq a (HMul.hMul d A)
    B : R
    hB : Eq b (HMul.hMul d B)
    C : R
    hC : Eq c (HMul.hMul d C)
    ha : And (Ne d 0) (Ne A 0)
    hb : And (Ne d 0) (Ne B 0)
    hc : And (Ne d 0) (Ne C 0)
    habc : Eq (HAdd.hAdd (HPow.hPow A n) (HPow.hPow B n)) (HPow.hPow C n)
    ⊢ False
  -/
  refine hn A B C ha.2 hb.2 hc.2 ?_ habc
  /-
    case intro.intro.intro
    n : Nat
    R : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsDomain R
    inst✝¹ : DecidableEq R
    inst✝ : NormalizedGCDMonoid R
    hn : ∀ (a b c : R), Ne a 0 → Ne b 0 → Ne c 0 → Eq ((Insert.insert a (Insert.in …
    a b c : R
    s : Finset R := Insert.insert a (Insert.insert b (Singleton.singleton c))
    d : R := s.gcd id
    A : R
    hA : Eq a (HMul.hMul d A)
    B : R
    hB : Eq b (HMul.hMul d B)
    C : R
    hC : Eq c (HMul.hMul d C)
    ha : And (Ne d 0) (Ne A 0)
    hb : And (Ne d 0) (Ne B 0)
    hc : And (Ne d 0) (Ne C 0)
    habc : Eq (HAdd.hAdd (HPow.hPow A n) (HPow.hPow B n)) (HPow.hPow C n)
    ⊢ Eq ((Insert.insert A (Insert.insert B (Singleton.singleton C))).gcd id) 1
  -/
  rw [← Finset.normalize_gcd, normalize_eq_one]
  /-
    case intro.intro.intro
    n : Nat
    R : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsDomain R
    inst✝¹ : DecidableEq R
    inst✝ : NormalizedGCDMonoid R
    hn : ∀ (a b c : R), Ne a 0 → Ne b 0 → Ne c 0 → Eq ((Insert.insert a (Insert.in …
    a b c : R
    s : Finset R := Insert.insert a (Insert.insert b (Singleton.singleton c))
    d : R := s.gcd id
    A : R
    hA : Eq a (HMul.hMul d A)
    B : R
    hB : Eq b (HMul.hMul d B)
    C : R
    hC : Eq c (HMul.hMul d C)
    ha : And (Ne d 0) (Ne A 0)
    hb : And (Ne d 0) (Ne B 0)
    hc : And (Ne d 0) (Ne C 0)
    habc : Eq (HAdd.hAdd (HPow.hPow A n) (HPow.hPow B n)) (HPow.hPow C n)
    ⊢ IsUnit ((Insert.insert A (Insert.insert B (Singleton.singleton C))).gcd id)
  -/
  obtain ⟨u, hu⟩ := normalize_associated d
  /-
    case intro.intro.intro.intro
    n : Nat
    R : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsDomain R
    inst✝¹ : DecidableEq R
    inst✝ : NormalizedGCDMonoid R
    hn : ∀ (a b c : R), Ne a 0 → Ne b 0 → Ne c 0 → Eq ((Insert.insert a (Insert.in …
    a b c : R
    s : Finset R := Insert.insert a (Insert.insert b (Singleton.singleton c))
    d : R := s.gcd id
    A : R
    hA : Eq a (HMul.hMul d A)
    B : R
    hB : Eq b (HMul.hMul d B)
    C : R
    hC : Eq c (HMul.hMul d C)
    ha : And (Ne d 0) (Ne A 0)
    hb : And (Ne d 0) (Ne B 0)
    hc : And (Ne d 0) (Ne C 0)
    habc : Eq (HAdd.hAdd (HPow.hPow A n) (HPow.hPow B n)) (HPow.hPow C n)
    u : Units R
    hu : Eq (HMul.hMul (normalize d) ↑u) d
    ⊢ IsUnit ((Insert.insert A (Insert.insert B (Singleton.singleton C))).gcd id)
  -/
  refine ⟨u, mul_left_cancel₀ (mt normalize_eq_zero.mp ha.1) (hu.symm ▸ ?_)⟩
  rw [← Finset.gcd_mul_left, gcd_eq_gcd_image, image_insert, image_insert, image_singleton,
      id_eq, id_eq, id_eq, ← hA, ← hB, ← hC]


lemma dvd_c_of_prime_of_dvd_a_of_dvd_b_of_FLT {n : ℕ} {p : ℤ} (hp : Prime p) {a b c : ℤ}
    (hpa : p ∣ a) (hpb : p ∣ b) (HF : a ^ n + b ^ n + c ^ n = 0) : p ∣ c := by
  /-
    n : Nat
    p : Int
    hp : Prime p
    a b c : Int
    hpa : Dvd.dvd p a
    hpb : Dvd.dvd p b
    HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n) …
    ⊢ Dvd.dvd p c
  -/
  rcases eq_or_ne n 0 with rfl | hn
    /-
      case inl
      p : Int
      hp : Prime p
      a b c : Int
      hpa : Dvd.dvd p a
      hpb : Dvd.dvd p b
      HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 0) (HPow.hPow b 0)) (HPow.hPow c 0) …
      ⊢ Dvd.dvd p c
    -/
  · simp at HF
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Nat
    p : Int
    hp : Prime p
    a b c : Int
    hpa : Dvd.dvd p a
    hpb : Dvd.dvd p b
    HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n) …
    hn : Ne n 0
    ⊢ Dvd.dvd p c
  -/
  refine hp.dvd_of_dvd_pow (n := n) (dvd_neg.1 ?_)
  /-
    case inr
    n : Nat
    p : Int
    hp : Prime p
    a b c : Int
    hpa : Dvd.dvd p a
    hpb : Dvd.dvd p b
    HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n) …
    hn : Ne n 0
    ⊢ Dvd.dvd p (Neg.neg (HPow.hPow c n))
  -/
  rw [add_eq_zero_iff_eq_neg] at HF
  /-
    case inr
    n : Nat
    p : Int
    hp : Prime p
    a b c : Int
    hpa : Dvd.dvd p a
    hpb : Dvd.dvd p b
    HF : Eq (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (Neg.neg (HPow.hPow c n))
    hn : Ne n 0
    ⊢ Dvd.dvd p (Neg.neg (HPow.hPow c n))
  -/
  exact HF.symm ▸ dvd_add (dvd_pow hpa hn) (dvd_pow hpb hn)
  /-
    🎉 no goals
  -/


lemma isCoprime_of_gcd_eq_one_of_FLT {n : ℕ} {a b c : ℤ} (Hgcd : Finset.gcd {a, b, c} id = 1)
    (HF : a ^ n + b ^ n + c ^ n = 0) : IsCoprime a b := by
  /-
    n : Nat
    a b c : Int
    Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
    HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n) …
    ⊢ IsCoprime a b
  -/
  rcases eq_or_ne n 0 with rfl | hn
    /-
      case inl
      a b c : Int
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 0) (HPow.hPow b 0)) (HPow.hPow c 0) …
      ⊢ IsCoprime a b
    -/
  · simp only [pow_zero, Int.reduceAdd, OfNat.ofNat_ne_zero] at HF
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Nat
    a b c : Int
    Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
    HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n) …
    hn : Ne n 0
    ⊢ IsCoprime a b
  -/
  refine isCoprime_of_prime_dvd  ?_ <| (fun p hp hpa hpb ↦ hp.not_dvd_one ?_)
    /-
      case inr.refine_1
      n : Nat
      a b c : Int
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n) …
      hn : Ne n 0
      ⊢ Not (And (Eq a 0) (Eq b 0))
    -/
  · rintro ⟨rfl, rfl⟩
    simp only [ne_eq, hn, not_false_eq_true, zero_pow, add_zero, zero_add, pow_eq_zero_iff]
      at HF
    simp only [HF, Finset.mem_singleton, Finset.insert_eq_of_mem, Finset.gcd_singleton, id_eq,
      map_zero, zero_ne_one] at Hgcd
    /-
      case inr.refine_2
      n : Nat
      a b c : Int
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n) …
      hn : Ne n 0
      p : Int
      hp : Prime p
      hpa : Dvd.dvd p a
      hpb : Dvd.dvd p b
      ⊢ Dvd.dvd p 1
    -/
  · rw [← Hgcd]
    /-
      case inr.refine_2
      n : Nat
      a b c : Int
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n) …
      hn : Ne n 0
      p : Int
      hp : Prime p
      hpa : Dvd.dvd p a
      hpb : Dvd.dvd p b
      ⊢ Dvd.dvd p ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id)
    -/
    refine Finset.dvd_gcd_iff.mpr fun x hx ↦ ?_
    /-
      case inr.refine_2
      n : Nat
      a b c : Int
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n) …
      hn : Ne n 0
      p : Int
      hp : Prime p
      hpa : Dvd.dvd p a
      hpb : Dvd.dvd p b
      x : Int
      hx : Membership.mem (Insert.insert a (Insert.insert b (Singleton.singleton c)) …
      ⊢ Dvd.dvd p (id x)
    -/
    simp only [Finset.mem_insert, Finset.mem_singleton] at hx
    /-
      case inr.refine_2
      n : Nat
      a b c : Int
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) (HPow.hPow c n) …
      hn : Ne n 0
      p : Int
      hp : Prime p
      hpa : Dvd.dvd p a
      hpb : Dvd.dvd p b
      x : Int
      hx : Or (Eq x a) (Or (Eq x b) (Eq x c))
      ⊢ Dvd.dvd p (id x)
    -/
    rcases hx with hx | hx | hx <;> simp only [id_eq, hx, hpa, hpb,
      dvd_c_of_prime_of_dvd_a_of_dvd_b_of_FLT hp hpa hpb HF]

