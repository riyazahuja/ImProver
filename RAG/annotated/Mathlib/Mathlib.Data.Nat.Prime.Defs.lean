/-- `Nat.Prime p` means that `p` is a prime number, that is, a natural number
  at least 2 whose only divisors are `p` and `1`.
  The theorem `Nat.prime_def` witnesses this description of a prime number. -/
@[pp_nodot]
def Prime (p : ℕ) :=
  Irreducible p


theorem irreducible_iff_nat_prime (a : ℕ) : Irreducible a ↔ Nat.Prime a :=
  Iff.rfl


@[aesop safe destruct] theorem not_prime_zero : ¬Prime 0
  | h => h.ne_zero rfl


@[aesop safe destruct] theorem not_prime_one : ¬Prime 1
  | h => h.ne_one rfl


theorem Prime.ne_zero {n : ℕ} (h : Prime n) : n ≠ 0 :=
  Irreducible.ne_zero h


theorem Prime.pos {p : ℕ} (pp : Prime p) : 0 < p :=
  Nat.pos_of_ne_zero pp.ne_zero


theorem Prime.two_le : ∀ {p : ℕ}, Prime p → 2 ≤ p
  | 0, h => (not_prime_zero h).elim
  | 1, h => (not_prime_one h).elim
  | _ + 2, _ => le_add_left 2 _


theorem Prime.one_lt {p : ℕ} : Prime p → 1 < p :=
  Prime.two_le


lemma Prime.one_le {p : ℕ} (hp : p.Prime) : 1 ≤ p := hp.one_lt.le


instance Prime.one_lt' (p : ℕ) [hp : Fact p.Prime] : Fact (1 < p) :=
  ⟨hp.1.one_lt⟩


theorem Prime.ne_one {p : ℕ} (hp : p.Prime) : p ≠ 1 :=
  hp.one_lt.ne'


theorem Prime.eq_one_or_self_of_dvd {p : ℕ} (pp : p.Prime) (m : ℕ) (hm : m ∣ p) :
    m = 1 ∨ m = p := by
  /-
    p : Nat
    pp : Nat.Prime p
    m : Nat
    hm : Dvd.dvd m p
    ⊢ Or (Eq m 1) (Eq m p)
  -/
  obtain ⟨n, hn⟩ := hm
  /-
    case intro
    p : Nat
    pp : Nat.Prime p
    m n : Nat
    hn : Eq p (HMul.hMul m n)
    ⊢ Or (Eq m 1) (Eq m p)
  -/
  have := pp.isUnit_or_isUnit hn
  /-
    case intro
    p : Nat
    pp : Nat.Prime p
    m n : Nat
    hn : Eq p (HMul.hMul m n)
    this : Or (IsUnit m) (IsUnit n)
    ⊢ Or (Eq m 1) (Eq m p)
  -/
  rw [Nat.isUnit_iff, Nat.isUnit_iff] at this
  /-
    case intro
    p : Nat
    pp : Nat.Prime p
    m n : Nat
    hn : Eq p (HMul.hMul m n)
    this : Or (Eq m 1) (Eq n 1)
    ⊢ Or (Eq m 1) (Eq m p)
  -/
  apply Or.imp_right _ this
  /-
    p : Nat
    pp : Nat.Prime p
    m n : Nat
    hn : Eq p (HMul.hMul m n)
    this : Or (Eq m 1) (Eq n 1)
    ⊢ Eq n 1 → Eq m p
  -/
  rintro rfl
  /-
    p : Nat
    pp : Nat.Prime p
    m : Nat
    hn : Eq p (HMul.hMul m 1)
    this : Or (Eq m 1) (Eq 1 1)
    ⊢ Eq m p
  -/
  rw [hn, mul_one]
  /-
    🎉 no goals
  -/


@[inherit_doc Nat.Prime]
theorem prime_def {p : ℕ} : Prime p ↔ 2 ≤ p ∧ ∀ m, m ∣ p → m = 1 ∨ m = p := by
  /-
    p : Nat
    ⊢ Iff (Nat.Prime p) (And (LE.le 2 p) (∀ (m : Nat), Dvd.dvd m p → Or (Eq m 1) ( …
  -/
  refine ⟨fun h => ⟨h.two_le, h.eq_one_or_self_of_dvd⟩, fun h => ?_⟩
  /-
    p : Nat
    h : And (LE.le 2 p) (∀ (m : Nat), Dvd.dvd m p → Or (Eq m 1) (Eq m p))
    ⊢ Nat.Prime p
  -/
  have h1 := Nat.one_lt_two.trans_le h.1
  /-
    p : Nat
    h : And (LE.le 2 p) (∀ (m : Nat), Dvd.dvd m p → Or (Eq m 1) (Eq m p))
    h1 : LT.lt 1 p
    ⊢ Nat.Prime p
  -/
  refine ⟨mt Nat.isUnit_iff.mp h1.ne', fun a b hab => ?_⟩
  /-
    p : Nat
    h : And (LE.le 2 p) (∀ (m : Nat), Dvd.dvd m p → Or (Eq m 1) (Eq m p))
    h1 : LT.lt 1 p
    a b : Nat
    hab : Eq p (HMul.hMul a b)
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
  simp only [Nat.isUnit_iff]
  /-
    p : Nat
    h : And (LE.le 2 p) (∀ (m : Nat), Dvd.dvd m p → Or (Eq m 1) (Eq m p))
    h1 : LT.lt 1 p
    a b : Nat
    hab : Eq p (HMul.hMul a b)
    ⊢ Or (Eq a 1) (Eq b 1)
  -/
  apply Or.imp_right _ (h.2 a _)
    /-
      p : Nat
      h : And (LE.le 2 p) (∀ (m : Nat), Dvd.dvd m p → Or (Eq m 1) (Eq m p))
      h1 : LT.lt 1 p
      a b : Nat
      hab : Eq p (HMul.hMul a b)
      ⊢ Eq a p → Eq b 1
    -/
  · rintro rfl
    /-
      a b : Nat
      h : And (LE.le 2 a) (∀ (m : Nat), Dvd.dvd m a → Or (Eq m 1) (Eq m a))
      h1 : LT.lt 1 a
      hab : Eq a (HMul.hMul a b)
      ⊢ Eq b 1
    -/
    rw [← mul_right_inj' (not_eq_zero_of_lt h1), ← hab, mul_one]
    /-
      🎉 no goals
    -/
    /-
      p : Nat
      h : And (LE.le 2 p) (∀ (m : Nat), Dvd.dvd m p → Or (Eq m 1) (Eq m p))
      h1 : LT.lt 1 p
      a b : Nat
      hab : Eq p (HMul.hMul a b)
      ⊢ Dvd.dvd a p
    -/
  · rw [hab]
    /-
      p : Nat
      h : And (LE.le 2 p) (∀ (m : Nat), Dvd.dvd m p → Or (Eq m 1) (Eq m p))
      h1 : LT.lt 1 p
      a b : Nat
      hab : Eq p (HMul.hMul a b)
      ⊢ Dvd.dvd a (HMul.hMul a b)
    -/
    exact dvd_mul_right _ _
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-19")]
alias prime_def_lt'' := prime_def


theorem prime_def_lt {p : ℕ} : Prime p ↔ 2 ≤ p ∧ ∀ m < p, m ∣ p → m = 1 :=
  prime_def.trans <|
    and_congr_right fun p2 =>
      forall_congr' fun _ =>
        ⟨fun h l d => (h d).resolve_right (ne_of_lt l), fun h d =>
          (le_of_dvd (le_of_succ_le p2) d).lt_or_eq_dec.imp_left fun l => h l d⟩


theorem prime_def_lt' {p : ℕ} : Prime p ↔ 2 ≤ p ∧ ∀ m, 2 ≤ m → m < p → ¬m ∣ p :=
  prime_def_lt.trans <|
    and_congr_right fun p2 =>
      forall_congr' fun m =>
                                                            /-
                                                              p : Nat
                                                              p2 : LE.le 2 p
                                                              m : Nat
                                                              h : LT.lt m p → Dvd.dvd m p → Eq m 1
                                                              m2 : LE.le 2 m
                                                              l : LT.lt m p
                                                              d : Dvd.dvd m p
                                                              ⊢ LT.lt 1 2
                                                            -/
        ⟨fun h m2 l d => not_lt_of_ge m2 ((h l d).symm ▸ by decide), fun h l d => by
                                                            /-
                                                              🎉 no goals
                                                            -/
          /-
            p : Nat
            p2 : LE.le 2 p
            m : Nat
            h : LE.le 2 m → LT.lt m p → Not (Dvd.dvd m p)
            l : LT.lt m p
            d : Dvd.dvd m p
            ⊢ Eq m 1
          -/
          rcases m with (_ | _ | m)
            /-
              case zero
              p : Nat
              p2 : LE.le 2 p
              h : LE.le 2 0 → LT.lt 0 p → Not (Dvd.dvd 0 p)
              l : LT.lt 0 p
              d : Dvd.dvd 0 p
              ⊢ Eq 0 1
            -/
          · rw [eq_zero_of_zero_dvd d] at p2
            /-
              case zero
              p : Nat
              p2 : LE.le 2 0
              h : LE.le 2 0 → LT.lt 0 p → Not (Dvd.dvd 0 p)
              l : LT.lt 0 p
              d : Dvd.dvd 0 p
              ⊢ Eq 0 1
            -/
            revert p2
            /-
              case zero
              p : Nat
              h : LE.le 2 0 → LT.lt 0 p → Not (Dvd.dvd 0 p)
              l : LT.lt 0 p
              d : Dvd.dvd 0 p
              ⊢ LE.le 2 0 → Eq 0 1
            -/
            decide
            /-
              🎉 no goals
            -/
            /-
              case succ.zero
              p : Nat
              p2 : LE.le 2 p
              h : LE.le 2 (HAdd.hAdd 0 1) → LT.lt (HAdd.hAdd 0 1) p → Not (Dvd.dvd (HAdd.hAd …
              l : LT.lt (HAdd.hAdd 0 1) p
              d : Dvd.dvd (HAdd.hAdd 0 1) p
              ⊢ Eq (HAdd.hAdd 0 1) 1
            -/
          · rfl
            /-
              🎉 no goals
            -/
            /-
              case succ.succ
              p : Nat
              p2 : LE.le 2 p
              m : Nat
              h : LE.le 2 (HAdd.hAdd (HAdd.hAdd m 1) 1) → LT.lt (HAdd.hAdd (HAdd.hAdd m 1) 1 …
              l : LT.lt (HAdd.hAdd (HAdd.hAdd m 1) 1) p
              d : Dvd.dvd (HAdd.hAdd (HAdd.hAdd m 1) 1) p
              ⊢ Eq (HAdd.hAdd (HAdd.hAdd m 1) 1) 1
            -/
          · exact (h (le_add_left 2 m) l).elim d⟩
            /-
              🎉 no goals
            -/


theorem prime_def_le_sqrt {p : ℕ} : Prime p ↔ 2 ≤ p ∧ ∀ m, 2 ≤ m → m ≤ sqrt p → ¬m ∣ p :=
  prime_def_lt'.trans <|
    and_congr_right fun p2 =>
      ⟨fun a m m2 l => a m m2 <| lt_of_le_of_lt l <| sqrt_lt_self p2, fun a =>
        have : ∀ {m k : ℕ}, m ≤ k → 1 < m → p ≠ m * k := fun {m k} mk m1 e =>
          a m m1 (le_sqrt.2 (e.symm ▸ Nat.mul_le_mul_left m mk)) ⟨k, e⟩
        fun m m2 l ⟨k, e⟩ => by
        /-
          p : Nat
          p2 : LE.le 2 p
          a : ∀ (m : Nat), LE.le 2 m → LE.le m p.sqrt → Not (Dvd.dvd m p)
          this : ∀ {m k : Nat}, LE.le m k → LT.lt 1 m → Ne p (HMul.hMul m k)
          m : Nat
          m2 : LE.le 2 m
          l : LT.lt m p
          x✝ : Dvd.dvd m p
          k : Nat
          e : Eq p (HMul.hMul m k)
          ⊢ False
        -/
        rcases le_total m k with mk | km
          /-
            case inl
            p : Nat
            p2 : LE.le 2 p
            a : ∀ (m : Nat), LE.le 2 m → LE.le m p.sqrt → Not (Dvd.dvd m p)
            this : ∀ {m k : Nat}, LE.le m k → LT.lt 1 m → Ne p (HMul.hMul m k)
            m : Nat
            m2 : LE.le 2 m
            l : LT.lt m p
            x✝ : Dvd.dvd m p
            k : Nat
            e : Eq p (HMul.hMul m k)
            mk : LE.le m k
            ⊢ False
          -/
        · exact this mk m2 e
          /-
            🎉 no goals
          -/
          /-
            case inr
            p : Nat
            p2 : LE.le 2 p
            a : ∀ (m : Nat), LE.le 2 m → LE.le m p.sqrt → Not (Dvd.dvd m p)
            this : ∀ {m k : Nat}, LE.le m k → LT.lt 1 m → Ne p (HMul.hMul m k)
            m : Nat
            m2 : LE.le 2 m
            l : LT.lt m p
            x✝ : Dvd.dvd m p
            k : Nat
            e : Eq p (HMul.hMul m k)
            km : LE.le k m
            ⊢ False
          -/
        · rw [mul_comm] at e
          /-
            case inr
            p : Nat
            p2 : LE.le 2 p
            a : ∀ (m : Nat), LE.le 2 m → LE.le m p.sqrt → Not (Dvd.dvd m p)
            this : ∀ {m k : Nat}, LE.le m k → LT.lt 1 m → Ne p (HMul.hMul m k)
            m : Nat
            m2 : LE.le 2 m
            l : LT.lt m p
            x✝ : Dvd.dvd m p
            k : Nat
            e : Eq p (HMul.hMul k m)
            km : LE.le k m
            ⊢ False
          -/
          refine this km (Nat.lt_of_mul_lt_mul_right (a := m) ?_) e
          /-
            case inr
            p : Nat
            p2 : LE.le 2 p
            a : ∀ (m : Nat), LE.le 2 m → LE.le m p.sqrt → Not (Dvd.dvd m p)
            this : ∀ {m k : Nat}, LE.le m k → LT.lt 1 m → Ne p (HMul.hMul m k)
            m : Nat
            m2 : LE.le 2 m
            l : LT.lt m p
            x✝ : Dvd.dvd m p
            k : Nat
            e : Eq p (HMul.hMul k m)
            km : LE.le k m
            ⊢ LT.lt (HMul.hMul 1 m) (HMul.hMul k m)
          -/
          rwa [one_mul, ← e]⟩
          /-
            🎉 no goals
          -/


theorem prime_of_coprime (n : ℕ) (h1 : 1 < n) (h : ∀ m < n, m ≠ 0 → n.Coprime m) : Prime n := by
  /-
    n : Nat
    h1 : LT.lt 1 n
    h : ∀ (m : Nat), LT.lt m n → Ne m 0 → n.Coprime m
    ⊢ Nat.Prime n
  -/
  refine prime_def_lt.mpr ⟨h1, fun m mlt mdvd => ?_⟩
  have hm : m ≠ 0 := by
    rintro rfl
    rw [zero_dvd_iff] at mdvd
    exact mlt.ne' mdvd
  /-
    n : Nat
    h1 : LT.lt 1 n
    h : ∀ (m : Nat), LT.lt m n → Ne m 0 → n.Coprime m
    m : Nat
    mlt : LT.lt m n
    mdvd : Dvd.dvd m n
    hm : Ne m 0
    ⊢ Eq m 1
  -/
  exact (h m mlt hm).symm.eq_one_of_dvd mdvd
  /-
    🎉 no goals
  -/


/--
This instance is set up to work in the kernel (`by decide`) for small values.

Below (`decidablePrime'`) we will define a faster variant to be used by the compiler
(e.g. in `#eval` or `by native_decide`).

If you need to prove that a particular number is prime, in any case
you should not use `by decide`, but rather `by norm_num`, which is
much faster.
-/
instance decidablePrime (p : ℕ) : Decidable (Prime p) :=
  decidable_of_iff' _ prime_def_lt'


                                  /-
                                    ⊢ Nat.Prime 2
                                  -/
theorem prime_two : Prime 2 := by decide
                                  /-
                                    🎉 no goals
                                  -/


                                    /-
                                      ⊢ Nat.Prime 3
                                    -/
theorem prime_three : Prime 3 := by decide
                                    /-
                                      🎉 no goals
                                    -/


                                   /-
                                     ⊢ Nat.Prime 5
                                   -/
theorem prime_five : Prime 5 := by decide
                                   /-
                                     🎉 no goals
                                   -/


theorem dvd_prime {p m : ℕ} (pp : Prime p) : m ∣ p ↔ m = 1 ∨ m = p :=
  ⟨fun d => pp.eq_one_or_self_of_dvd m d, fun h =>
    h.elim (fun e => e.symm ▸ one_dvd _) fun e => e.symm ▸ dvd_rfl⟩


theorem dvd_prime_two_le {p m : ℕ} (pp : Prime p) (H : 2 ≤ m) : m ∣ p ↔ m = p :=
  (dvd_prime pp).trans <| or_iff_right_of_imp <| Not.elim <| ne_of_gt H


theorem prime_dvd_prime_iff_eq {p q : ℕ} (pp : p.Prime) (qp : q.Prime) : p ∣ q ↔ p = q :=
  dvd_prime_two_le qp (Prime.two_le pp)


theorem Prime.not_dvd_one {p : ℕ} (pp : Prime p) : ¬p ∣ 1 :=
  Irreducible.not_dvd_one pp


theorem minFac_lemma (n k : ℕ) (h : ¬n < k * k) : sqrt n - k < sqrt n + 2 - k :=
                                                                                       /-
                                                                                         n k : Nat
                                                                                         h : Not (LT.lt n (HMul.hMul k k))
                                                                                         ⊢ LT.lt 0 2
                                                                                       -/
  (Nat.sub_lt_sub_right <| le_sqrt.2 <| le_of_not_gt h) <| Nat.lt_add_of_pos_right (by decide)
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


/--
If `n < k * k`, then `minFacAux n k = n`, if `k | n`, then `minFacAux n k = k`.
Otherwise, `minFacAux n k = minFacAux n (k+2)` using well-founded recursion.
If `n` is odd and `1 < n`, then `minFacAux n 3` is the smallest prime factor of `n`.

By default this well-founded recursion would be irreducible.
This prevents use `decide` to resolve `Nat.prime n` for small values of `n`,
so we mark this as `@[semireducible]`.

In future, we may want to remove this annotation and instead use `norm_num` instead of `decide`
in these situations.
-/
@[semireducible] def minFacAux (n : ℕ) : ℕ → ℕ
  | k =>
    if n < k * k then n
    else
      if k ∣ n then k
      else
        minFacAux n (k + 2)
termination_by k => sqrt n + 2 - k
/-
  n a✝ : Nat
  k : Nat := a✝
  h✝¹ : Not (LT.lt n (HMul.hMul k k))
  h✝ : Not (Dvd.dvd k n)
  ⊢ LT.lt (HSub.hSub (HAdd.hAdd n.sqrt 2) (HAdd.hAdd a✝ 2)) (HSub.hSub (HAdd.hAd …
-/
decreasing_by simp_wf; apply minFac_lemma n k; assumption
/-
  🎉 no goals
-/


/-- Returns the smallest prime factor of `n ≠ 1`. -/
def minFac (n : ℕ) : ℕ :=
  if 2 ∣ n then 2 else minFacAux n 3


@[simp]
theorem minFac_zero : minFac 0 = 2 :=
  rfl


@[simp]
theorem minFac_one : minFac 1 = 1 := by
  /-
    ⊢ Eq (Nat.minFac 1) 1
  -/
  simp [minFac, minFacAux]
  /-
    🎉 no goals
  -/


@[simp]
theorem minFac_two : minFac 2 = 2 := by
  /-
    ⊢ Eq (Nat.minFac 2) 2
  -/
  simp [minFac, minFacAux]
  /-
    🎉 no goals
  -/


theorem minFac_eq (n : ℕ) : minFac n = if 2 ∣ n then 2 else minFacAux n 3 := rfl


private def minFacProp (n k : ℕ) :=
  2 ≤ k ∧ k ∣ n ∧ ∀ m, 2 ≤ m → m ∣ n → k ≤ m


theorem minFacAux_has_prop {n : ℕ} (n2 : 2 ≤ n) :
    ∀ k i, k = 2 * i + 3 → (∀ m, 2 ≤ m → m ∣ n → k ≤ m) → minFacProp n (minFacAux n k)
  | k => fun i e a => by
    /-
      n : Nat
      n2 : LE.le 2 n
      x✝ : Nat
      k : Nat := x✝
      i : Nat
      e : Eq x✝ (HAdd.hAdd (HMul.hMul 2 i) 3)
      a : ∀ (m : Nat), LE.le 2 m → Dvd.dvd m n → LE.le x✝ m
      ⊢ Nat.minFacProp n (n.minFacAux x✝)
    -/
    rw [minFacAux]
    /-
      n : Nat
      n2 : LE.le 2 n
      x✝ : Nat
      k : Nat := x✝
      i : Nat
      e : Eq x✝ (HAdd.hAdd (HMul.hMul 2 i) 3)
      a : ∀ (m : Nat), LE.le 2 m → Dvd.dvd m n → LE.le x✝ m
      ⊢ Nat.minFacProp n (ite (LT.lt n (HMul.hMul x✝ x✝)) n (ite (Dvd.dvd x✝ n) x✝ ( …
    -/
    by_cases h : n < k * k
    · have pp : Prime n :=
        prime_def_le_sqrt.2
          ⟨n2, fun m m2 l d => not_lt_of_ge l <| lt_of_lt_of_le (sqrt_lt.2 h) (a m m2 d)⟩
      simpa only [k, h] using
        ⟨n2, dvd_rfl, fun m m2 d => le_of_eq ((dvd_prime_two_le pp m2).1 d).symm⟩
    have k2 : 2 ≤ k := by
      subst e
      apply Nat.le_add_left
    /-
      case neg
      n : Nat
      n2 : LE.le 2 n
      x✝ : Nat
      k : Nat := x✝
      i : Nat
      e : Eq x✝ (HAdd.hAdd (HMul.hMul 2 i) 3)
      a : ∀ (m : Nat), LE.le 2 m → Dvd.dvd m n → LE.le x✝ m
      h : Not (LT.lt n (HMul.hMul k k))
      k2 : LE.le 2 k
      ⊢ Nat.minFacProp n (ite (LT.lt n (HMul.hMul x✝ x✝)) n (ite (Dvd.dvd x✝ n) x✝ ( …
    -/
    simp only [k, h, ↓reduceIte]
    /-
      case neg
      n : Nat
      n2 : LE.le 2 n
      x✝ : Nat
      k : Nat := x✝
      i : Nat
      e : Eq x✝ (HAdd.hAdd (HMul.hMul 2 i) 3)
      a : ∀ (m : Nat), LE.le 2 m → Dvd.dvd m n → LE.le x✝ m
      h : Not (LT.lt n (HMul.hMul k k))
      k2 : LE.le 2 k
      ⊢ Nat.minFacProp n (ite (Dvd.dvd x✝ n) x✝ (n.minFacAux (HAdd.hAdd x✝ 2)))
    -/
    by_cases dk : k ∣ n <;> simp only [k, dk, ↓reduceIte]
      /-
        case pos
        n : Nat
        n2 : LE.le 2 n
        x✝ : Nat
        k : Nat := x✝
        i : Nat
        e : Eq x✝ (HAdd.hAdd (HMul.hMul 2 i) 3)
        a : ∀ (m : Nat), LE.le 2 m → Dvd.dvd m n → LE.le x✝ m
        h : Not (LT.lt n (HMul.hMul k k))
        k2 : LE.le 2 k
        dk : Dvd.dvd k n
        ⊢ Nat.minFacProp n x✝
      -/
    · exact ⟨k2, dk, a⟩
      /-
        🎉 no goals
      -/
    · refine
        have := minFac_lemma n k h
        minFacAux_has_prop n2 (k + 2) (i + 1) (by simp [k, e, Nat.left_distrib, add_right_comm])
          fun m m2 d => ?_
      /-
        case neg
        n : Nat
        n2 : LE.le 2 n
        x✝ : Nat
        k : Nat := x✝
        i : Nat
        e : Eq x✝ (HAdd.hAdd (HMul.hMul 2 i) 3)
        a : ∀ (m : Nat), LE.le 2 m → Dvd.dvd m n → LE.le x✝ m
        h : Not (LT.lt n (HMul.hMul k k))
        k2 : LE.le 2 k
        dk : Not (Dvd.dvd k n)
        this : LT.lt (HSub.hSub n.sqrt k) (HSub.hSub (HAdd.hAdd n.sqrt 2) k)
        m : Nat
        m2 : LE.le 2 m
        d : Dvd.dvd m n
        ⊢ LE.le (HAdd.hAdd k 2) m
      -/
      rcases Nat.eq_or_lt_of_le (a m m2 d) with me | ml
        /-
          case neg.inl
          n : Nat
          n2 : LE.le 2 n
          x✝ : Nat
          k : Nat := x✝
          i : Nat
          e : Eq x✝ (HAdd.hAdd (HMul.hMul 2 i) 3)
          a : ∀ (m : Nat), LE.le 2 m → Dvd.dvd m n → LE.le x✝ m
          h : Not (LT.lt n (HMul.hMul k k))
          k2 : LE.le 2 k
          dk : Not (Dvd.dvd k n)
          this : LT.lt (HSub.hSub n.sqrt k) (HSub.hSub (HAdd.hAdd n.sqrt 2) k)
          m : Nat
          m2 : LE.le 2 m
          d : Dvd.dvd m n
          me : Eq x✝ m
          ⊢ LE.le (HAdd.hAdd k 2) m
        -/
      · subst me
        /-
          case neg.inl
          n : Nat
          n2 : LE.le 2 n
          x✝ : Nat
          k : Nat := x✝
          i : Nat
          e : Eq x✝ (HAdd.hAdd (HMul.hMul 2 i) 3)
          a : ∀ (m : Nat), LE.le 2 m → Dvd.dvd m n → LE.le x✝ m
          h : Not (LT.lt n (HMul.hMul k k))
          k2 : LE.le 2 k
          dk : Not (Dvd.dvd k n)
          this : LT.lt (HSub.hSub n.sqrt k) (HSub.hSub (HAdd.hAdd n.sqrt 2) k)
          m2 : LE.le 2 x✝
          d : Dvd.dvd x✝ n
          ⊢ LE.le (HAdd.hAdd k 2) x✝
        -/
        contradiction
        /-
          🎉 no goals
        -/
      /-
        case neg.inr
        n : Nat
        n2 : LE.le 2 n
        x✝ : Nat
        k : Nat := x✝
        i : Nat
        e : Eq x✝ (HAdd.hAdd (HMul.hMul 2 i) 3)
        a : ∀ (m : Nat), LE.le 2 m → Dvd.dvd m n → LE.le x✝ m
        h : Not (LT.lt n (HMul.hMul k k))
        k2 : LE.le 2 k
        dk : Not (Dvd.dvd k n)
        this : LT.lt (HSub.hSub n.sqrt k) (HSub.hSub (HAdd.hAdd n.sqrt 2) k)
        m : Nat
        m2 : LE.le 2 m
        d : Dvd.dvd m n
        ml : LT.lt x✝ m
        ⊢ LE.le (HAdd.hAdd k 2) m
      -/
      apply (Nat.eq_or_lt_of_le ml).resolve_left
      /-
        case neg.inr
        n : Nat
        n2 : LE.le 2 n
        x✝ : Nat
        k : Nat := x✝
        i : Nat
        e : Eq x✝ (HAdd.hAdd (HMul.hMul 2 i) 3)
        a : ∀ (m : Nat), LE.le 2 m → Dvd.dvd m n → LE.le x✝ m
        h : Not (LT.lt n (HMul.hMul k k))
        k2 : LE.le 2 k
        dk : Not (Dvd.dvd k n)
        this : LT.lt (HSub.hSub n.sqrt k) (HSub.hSub (HAdd.hAdd n.sqrt 2) k)
        m : Nat
        m2 : LE.le 2 m
        d : Dvd.dvd m n
        ml : LT.lt x✝ m
        ⊢ Not (Eq x✝.succ m)
      -/
      intro me
      /-
        case neg.inr
        n : Nat
        n2 : LE.le 2 n
        x✝ : Nat
        k : Nat := x✝
        i : Nat
        e : Eq x✝ (HAdd.hAdd (HMul.hMul 2 i) 3)
        a : ∀ (m : Nat), LE.le 2 m → Dvd.dvd m n → LE.le x✝ m
        h : Not (LT.lt n (HMul.hMul k k))
        k2 : LE.le 2 k
        dk : Not (Dvd.dvd k n)
        this : LT.lt (HSub.hSub n.sqrt k) (HSub.hSub (HAdd.hAdd n.sqrt 2) k)
        m : Nat
        m2 : LE.le 2 m
        d : Dvd.dvd m n
        ml : LT.lt x✝ m
        me : Eq x✝.succ m
        ⊢ False
      -/
      rw [← me, e] at d
      /-
        case neg.inr
        n : Nat
        n2 : LE.le 2 n
        x✝ : Nat
        k : Nat := x✝
        i : Nat
        e : Eq x✝ (HAdd.hAdd (HMul.hMul 2 i) 3)
        a : ∀ (m : Nat), LE.le 2 m → Dvd.dvd m n → LE.le x✝ m
        h : Not (LT.lt n (HMul.hMul k k))
        k2 : LE.le 2 k
        dk : Not (Dvd.dvd k n)
        this : LT.lt (HSub.hSub n.sqrt k) (HSub.hSub (HAdd.hAdd n.sqrt 2) k)
        m : Nat
        m2 : LE.le 2 m
        d : Dvd.dvd (HAdd.hAdd (HMul.hMul 2 i) 3).succ n
        ml : LT.lt x✝ m
        me : Eq x✝.succ m
        ⊢ False
      -/
      have d' : 2 * (i + 2) ∣ n := d
      /-
        case neg.inr
        n : Nat
        n2 : LE.le 2 n
        x✝ : Nat
        k : Nat := x✝
        i : Nat
        e : Eq x✝ (HAdd.hAdd (HMul.hMul 2 i) 3)
        a : ∀ (m : Nat), LE.le 2 m → Dvd.dvd m n → LE.le x✝ m
        h : Not (LT.lt n (HMul.hMul k k))
        k2 : LE.le 2 k
        dk : Not (Dvd.dvd k n)
        this : LT.lt (HSub.hSub n.sqrt k) (HSub.hSub (HAdd.hAdd n.sqrt 2) k)
        m : Nat
        m2 : LE.le 2 m
        d : Dvd.dvd (HAdd.hAdd (HMul.hMul 2 i) 3).succ n
        ml : LT.lt x✝ m
        me : Eq x✝.succ m
        d' : Dvd.dvd (HMul.hMul 2 (HAdd.hAdd i 2)) n
        ⊢ False
      -/
      have := a _ le_rfl (dvd_of_mul_right_dvd d')
      /-
        case neg.inr
        n : Nat
        n2 : LE.le 2 n
        x✝ : Nat
        k : Nat := x✝
        i : Nat
        e : Eq x✝ (HAdd.hAdd (HMul.hMul 2 i) 3)
        a : ∀ (m : Nat), LE.le 2 m → Dvd.dvd m n → LE.le x✝ m
        h : Not (LT.lt n (HMul.hMul k k))
        k2 : LE.le 2 k
        dk : Not (Dvd.dvd k n)
        this✝ : LT.lt (HSub.hSub n.sqrt k) (HSub.hSub (HAdd.hAdd n.sqrt 2) k)
        m : Nat
        m2 : LE.le 2 m
        d : Dvd.dvd (HAdd.hAdd (HMul.hMul 2 i) 3).succ n
        ml : LT.lt x✝ m
        me : Eq x✝.succ m
        d' : Dvd.dvd (HMul.hMul 2 (HAdd.hAdd i 2)) n
        this : LE.le x✝ 2
        ⊢ False
      -/
      rw [e] at this
      /-
        case neg.inr
        n : Nat
        n2 : LE.le 2 n
        x✝ : Nat
        k : Nat := x✝
        i : Nat
        e : Eq x✝ (HAdd.hAdd (HMul.hMul 2 i) 3)
        a : ∀ (m : Nat), LE.le 2 m → Dvd.dvd m n → LE.le x✝ m
        h : Not (LT.lt n (HMul.hMul k k))
        k2 : LE.le 2 k
        dk : Not (Dvd.dvd k n)
        this✝ : LT.lt (HSub.hSub n.sqrt k) (HSub.hSub (HAdd.hAdd n.sqrt 2) k)
        m : Nat
        m2 : LE.le 2 m
        d : Dvd.dvd (HAdd.hAdd (HMul.hMul 2 i) 3).succ n
        ml : LT.lt x✝ m
        me : Eq x✝.succ m
        d' : Dvd.dvd (HMul.hMul 2 (HAdd.hAdd i 2)) n
        this : LE.le (HAdd.hAdd (HMul.hMul 2 i) 3) 2
        ⊢ False
      -/
      exact absurd this (by contradiction)
      /-
        🎉 no goals
      -/
  termination_by k => sqrt n + 2 - k


theorem minFac_has_prop {n : ℕ} (n1 : n ≠ 1) : minFacProp n (minFac n) := by
  /-
    n : Nat
    n1 : Ne n 1
    ⊢ Nat.minFacProp n n.minFac
  -/
  by_cases n0 : n = 0
    /-
      case pos
      n : Nat
      n1 : Ne n 1
      n0 : Eq n 0
      ⊢ Nat.minFacProp n n.minFac
    -/
  · simp [n0, minFacProp, GE.ge]
    /-
      🎉 no goals
    -/
  have n2 : 2 ≤ n := by
    revert n0 n1
    rcases n with (_ | _ | _) <;> simp [succ_le_succ]
  /-
    case neg
    n : Nat
    n1 : Ne n 1
    n0 : Not (Eq n 0)
    n2 : LE.le 2 n
    ⊢ Nat.minFacProp n n.minFac
  -/
  simp only [minFac_eq, Nat.isUnit_iff]
  /-
    case neg
    n : Nat
    n1 : Ne n 1
    n0 : Not (Eq n 0)
    n2 : LE.le 2 n
    ⊢ Nat.minFacProp n (ite (Dvd.dvd 2 n) 2 (n.minFacAux 3))
  -/
  by_cases d2 : 2 ∣ n <;> simp only [d2, ↓reduceIte]
    /-
      case pos
      n : Nat
      n1 : Ne n 1
      n0 : Not (Eq n 0)
      n2 : LE.le 2 n
      d2 : Dvd.dvd 2 n
      ⊢ Nat.minFacProp n 2
    -/
  · exact ⟨le_rfl, d2, fun k k2 _ => k2⟩
    /-
      🎉 no goals
    -/
  · refine
      minFacAux_has_prop n2 3 0 rfl fun m m2 d => (Nat.eq_or_lt_of_le m2).resolve_left (mt ?_ d2)
    /-
      case neg
      n : Nat
      n1 : Ne n 1
      n0 : Not (Eq n 0)
      n2 : LE.le 2 n
      d2 : Not (Dvd.dvd 2 n)
      m : Nat
      m2 : LE.le 2 m
      d : Dvd.dvd m n
      ⊢ Eq 2 m → Dvd.dvd 2 n
    -/
    exact fun e => e.symm ▸ d
    /-
      🎉 no goals
    -/


theorem minFac_dvd (n : ℕ) : minFac n ∣ n :=
                        /-
                          n : Nat
                          n1 : Eq n 1
                          ⊢ Dvd.dvd n.minFac n
                        -/
  if n1 : n = 1 then by simp [n1] else (minFac_has_prop n1).2.1
                        /-
                          🎉 no goals
                        -/


theorem minFac_prime {n : ℕ} (n1 : n ≠ 1) : Prime (minFac n) :=
  let ⟨f2, fd, a⟩ := minFac_has_prop n1
  prime_def_lt'.2 ⟨f2, fun m m2 l d => not_le_of_gt l (a m m2 (d.trans fd))⟩


theorem minFac_le_of_dvd {n : ℕ} : ∀ {m : ℕ}, 2 ≤ m → m ∣ n → minFac n ≤ m := by
  /-
    n : Nat
    ⊢ ∀ {m : Nat}, LE.le 2 m → Dvd.dvd m n → LE.le n.minFac m
  -/
  by_cases n1 : n = 1
    /-
      case pos
      n m✝ : Nat
      n1 : Eq n 1
      ⊢ LE.le 2 m✝ → Dvd.dvd m✝ n → LE.le n.minFac m✝
    -/
  · exact fun m2 _ => n1.symm ▸ le_trans (by simp) m2
    /-
      🎉 no goals
    -/
    /-
      case neg
      n m✝ : Nat
      n1 : Not (Eq n 1)
      ⊢ LE.le 2 m✝ → Dvd.dvd m✝ n → LE.le n.minFac m✝
    -/
  · apply (minFac_has_prop n1).2.2
    /-
      🎉 no goals
    -/


theorem minFac_pos (n : ℕ) : 0 < minFac n := by
  /-
    n : Nat
    ⊢ LT.lt 0 n.minFac
  -/
  by_cases n1 : n = 1
    /-
      case pos
      n : Nat
      n1 : Eq n 1
      ⊢ LT.lt 0 n.minFac
    -/
  · simp [n1]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      n1 : Not (Eq n 1)
      ⊢ LT.lt 0 n.minFac
    -/
  · exact (minFac_prime n1).pos
    /-
      🎉 no goals
    -/


theorem minFac_le {n : ℕ} (H : 0 < n) : minFac n ≤ n :=
  le_of_dvd H (minFac_dvd n)


theorem le_minFac {m n : ℕ} : n = 1 ∨ m ≤ minFac n ↔ ∀ p, Prime p → p ∣ n → m ≤ p :=
  ⟨fun h p pp d =>
               /-
                 m n : Nat
                 h : Or (Eq n 1) (LE.le m n.minFac)
                 p : Nat
                 pp : Nat.Prime p
                 d : Dvd.dvd p n
                 ⊢ Eq n 1 → LE.le m p
               -/
    h.elim (by rintro rfl; cases pp.not_dvd_one d) fun h =>
                           /-
                             🎉 no goals
                           -/
      le_trans h <| minFac_le_of_dvd pp.two_le d,
    fun H => or_iff_not_imp_left.2 fun n1 => H _ (minFac_prime n1) (minFac_dvd _)⟩


theorem le_minFac' {m n : ℕ} : n = 1 ∨ m ≤ minFac n ↔ ∀ p, 2 ≤ p → p ∣ n → m ≤ p :=
  ⟨fun h p (pp : 1 < p) d =>
               /-
                 m n : Nat
                 h : Or (Eq n 1) (LE.le m n.minFac)
                 p : Nat
                 pp : LT.lt 1 p
                 d : Dvd.dvd p n
                 ⊢ Eq n 1 → LE.le m p
               -/
    h.elim (by rintro rfl; cases not_le_of_lt pp (le_of_dvd (by decide) d)) fun h =>
                           /-
                             🎉 no goals
                           -/
      le_trans h <| minFac_le_of_dvd pp d,
    fun H => le_minFac.2 fun p pp d => H p pp.two_le d⟩


theorem prime_def_minFac {p : ℕ} : Prime p ↔ 2 ≤ p ∧ minFac p = p :=
  ⟨fun pp =>
    ⟨pp.two_le,
      let ⟨f2, fd, _⟩ := minFac_has_prop <| ne_of_gt pp.one_lt
      ((dvd_prime pp).1 fd).resolve_left (ne_of_gt f2)⟩,
    fun ⟨p2, e⟩ => e ▸ minFac_prime (ne_of_gt p2)⟩


@[simp]
theorem Prime.minFac_eq {p : ℕ} (hp : Prime p) : minFac p = p :=
  (prime_def_minFac.1 hp).2


/--
This definition is faster in the virtual machine than `decidablePrime`,
but slower in the kernel.
-/
def decidablePrime' (p : ℕ) : Decidable (Prime p) :=
  decidable_of_iff' _ prime_def_minFac


@[csimp] theorem decidablePrime_csimp :
    @decidablePrime = @decidablePrime' := by
  /-
    ⊢ Eq Nat.decidablePrime Nat.decidablePrime'
  -/
  funext; apply Subsingleton.elim
          /-
            🎉 no goals
          -/


theorem not_prime_iff_minFac_lt {n : ℕ} (n2 : 2 ≤ n) : ¬Prime n ↔ minFac n < n :=
  (not_congr <| prime_def_minFac.trans <| and_iff_right n2).trans <|
    (lt_iff_le_and_ne.trans <| and_iff_right <| minFac_le <| le_of_succ_le n2).symm


theorem minFac_le_div {n : ℕ} (pos : 0 < n) (np : ¬Prime n) : minFac n ≤ n / minFac n :=
  match minFac_dvd n with
                                /-
                                  n : Nat
                                  pos : LT.lt 0 n
                                  np : Not (Nat.Prime n)
                                  h0 : Eq n (HMul.hMul n.minFac 0)
                                  ⊢ Not (LT.lt 0 n)
                                -/
  | ⟨0, h0⟩ => absurd pos <| by rw [h0, mul_zero]; decide
                                                   /-
                                                     🎉 no goals
                                                   -/
  | ⟨1, h1⟩ => by
    /-
      n : Nat
      pos : LT.lt 0 n
      np : Not (Nat.Prime n)
      h1 : Eq n (HMul.hMul n.minFac 1)
      ⊢ LE.le n.minFac (HDiv.hDiv n n.minFac)
    -/
    rw [mul_one] at h1
    rw [prime_def_minFac, not_and_or, ← h1, eq_self_iff_true, _root_.not_true, _root_.or_false,
      not_le] at np
    /-
      n : Nat
      pos : LT.lt 0 n
      np : LT.lt n 2
      h1 : Eq n n.minFac
      ⊢ LE.le n.minFac (HDiv.hDiv n n.minFac)
    -/
    rw [le_antisymm (le_of_lt_succ np) (succ_le_of_lt pos), minFac_one, Nat.div_one]
    /-
      🎉 no goals
    -/
  | ⟨x + 2, hx⟩ => by
    conv_rhs =>
      congr
      rw [hx]
    /-
      n : Nat
      pos : LT.lt 0 n
      np : Not (Nat.Prime n)
      x : Nat
      hx : Eq n (HMul.hMul n.minFac (HAdd.hAdd x 2))
      ⊢ LE.le n.minFac (HDiv.hDiv (HMul.hMul n.minFac (HAdd.hAdd x 2)) n.minFac)
    -/
    rw [Nat.mul_div_cancel_left _ (minFac_pos _)]
    /-
      n : Nat
      pos : LT.lt 0 n
      np : Not (Nat.Prime n)
      x : Nat
      hx : Eq n (HMul.hMul n.minFac (HAdd.hAdd x 2))
      ⊢ LE.le n.minFac (HAdd.hAdd x 2)
    -/
    exact minFac_le_of_dvd (le_add_left 2 x) ⟨minFac n, by rwa [mul_comm]⟩
    /-
      🎉 no goals
    -/


/-- The square of the smallest prime factor of a composite number `n` is at most `n`.
-/
theorem minFac_sq_le_self {n : ℕ} (w : 0 < n) (h : ¬Prime n) : minFac n ^ 2 ≤ n :=
  have t : minFac n ≤ n / minFac n := minFac_le_div w h
  calc
    minFac n ^ 2 = minFac n * minFac n := sq (minFac n)
    _ ≤ n / minFac n * minFac n := Nat.mul_le_mul_right (minFac n) t
    _ ≤ n := div_mul_le_self n (minFac n)


@[simp]
theorem minFac_eq_one_iff {n : ℕ} : minFac n = 1 ↔ n = 1 := by
  /-
    n : Nat
    ⊢ Iff (Eq n.minFac 1) (Eq n 1)
  -/
  constructor
    /-
      case mp
      n : Nat
      ⊢ Eq n.minFac 1 → Eq n 1
    -/
  · intro h
    /-
      case mp
      n : Nat
      h : Eq n.minFac 1
      ⊢ Eq n 1
    -/
    by_contra hn
    /-
      case mp
      n : Nat
      h : Eq n.minFac 1
      hn : Not (Eq n 1)
      ⊢ False
    -/
    have := minFac_prime hn
    /-
      case mp
      n : Nat
      h : Eq n.minFac 1
      hn : Not (Eq n 1)
      this : Nat.Prime n.minFac
      ⊢ False
    -/
    rw [h] at this
    /-
      case mp
      n : Nat
      h : Eq n.minFac 1
      hn : Not (Eq n 1)
      this : Nat.Prime 1
      ⊢ False
    -/
    exact not_prime_one this
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      ⊢ Eq n 1 → Eq n.minFac 1
    -/
  · rintro rfl
    /-
      case mpr
      ⊢ Eq (Nat.minFac 1) 1
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem minFac_eq_two_iff (n : ℕ) : minFac n = 2 ↔ 2 ∣ n := by
  /-
    n : Nat
    ⊢ Iff (Eq n.minFac 2) (Dvd.dvd 2 n)
  -/
  constructor
    /-
      case mp
      n : Nat
      ⊢ Eq n.minFac 2 → Dvd.dvd 2 n
    -/
  · intro h
    /-
      case mp
      n : Nat
      h : Eq n.minFac 2
      ⊢ Dvd.dvd 2 n
    -/
    rw [← h]
    /-
      case mp
      n : Nat
      h : Eq n.minFac 2
      ⊢ Dvd.dvd n.minFac n
    -/
    exact minFac_dvd n
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      ⊢ Dvd.dvd 2 n → Eq n.minFac 2
    -/
  · intro h
    /-
      case mpr
      n : Nat
      h : Dvd.dvd 2 n
      ⊢ Eq n.minFac 2
    -/
    have ub := minFac_le_of_dvd (le_refl 2) h
    /-
      case mpr
      n : Nat
      h : Dvd.dvd 2 n
      ub : LE.le n.minFac 2
      ⊢ Eq n.minFac 2
    -/
    have lb := minFac_pos n
    /-
      case mpr
      n : Nat
      h : Dvd.dvd 2 n
      ub : LE.le n.minFac 2
      lb : LT.lt 0 n.minFac
      ⊢ Eq n.minFac 2
    -/
    refine ub.eq_or_lt.resolve_right fun h' => ?_
    /-
      case mpr
      n : Nat
      h : Dvd.dvd 2 n
      ub : LE.le n.minFac 2
      lb : LT.lt 0 n.minFac
      h' : LT.lt n.minFac 2
      ⊢ False
    -/
    have := le_antisymm (Nat.succ_le_of_lt lb) (Nat.lt_succ_iff.mp h')
    /-
      case mpr
      n : Nat
      h : Dvd.dvd 2 n
      ub : LE.le n.minFac 2
      lb : LT.lt 0 n.minFac
      h' : LT.lt n.minFac 2
      this : Eq (Nat.succ 0) n.minFac
      ⊢ False
    -/
    rw [eq_comm, Nat.minFac_eq_one_iff] at this
    /-
      case mpr
      n : Nat
      h : Dvd.dvd 2 n
      ub : LE.le n.minFac 2
      lb : LT.lt 0 n.minFac
      h' : LT.lt n.minFac 2
      this : Eq n 1
      ⊢ False
    -/
    subst this
    /-
      case mpr
      h : Dvd.dvd 2 1
      ub : LE.le (Nat.minFac 1) 2
      lb : LT.lt 0 (Nat.minFac 1)
      h' : LT.lt (Nat.minFac 1) 2
      ⊢ False
    -/
    exact not_lt_of_le (le_of_dvd lb h) h'
    /-
      🎉 no goals
    -/


theorem factors_lemma {k} : (k + 2) / minFac (k + 2) < k + 2 :=
  div_lt_self (Nat.zero_lt_succ _) (minFac_prime (by
      /-
        k : Nat
        ⊢ Ne (HAdd.hAdd k 2) 1
      -/
      apply Nat.ne_of_gt
      /-
        case h
        k : Nat
        ⊢ LT.lt 1 (HAdd.hAdd k 2)
      -/
      apply Nat.succ_lt_succ
      /-
        case h.a
        k : Nat
        ⊢ LT.lt 0 (HAdd.hAdd k 1)
      -/
      apply Nat.zero_lt_succ
      /-
        🎉 no goals
      -/
      )).one_lt


theorem exists_prime_and_dvd {n : ℕ} (hn : n ≠ 1) : ∃ p, Prime p ∧ p ∣ n :=
  ⟨minFac n, minFac_prime hn, minFac_dvd _⟩


theorem coprime_of_dvd {m n : ℕ} (H : ∀ k, Prime k → k ∣ m → ¬k ∣ n) : Coprime m n := by
  /-
    m n : Nat
    H : ∀ (k : Nat), Nat.Prime k → Dvd.dvd k m → Not (Dvd.dvd k n)
    ⊢ m.Coprime n
  -/
  rw [coprime_iff_gcd_eq_one]
  /-
    m n : Nat
    H : ∀ (k : Nat), Nat.Prime k → Dvd.dvd k m → Not (Dvd.dvd k n)
    ⊢ Eq (m.gcd n) 1
  -/
  by_contra g2
  /-
    m n : Nat
    H : ∀ (k : Nat), Nat.Prime k → Dvd.dvd k m → Not (Dvd.dvd k n)
    g2 : Not (Eq (m.gcd n) 1)
    ⊢ False
  -/
  obtain ⟨p, hp, hpdvd⟩ := exists_prime_and_dvd g2
  /-
    case intro.intro
    m n : Nat
    H : ∀ (k : Nat), Nat.Prime k → Dvd.dvd k m → Not (Dvd.dvd k n)
    g2 : Not (Eq (m.gcd n) 1)
    p : Nat
    hp : Nat.Prime p
    hpdvd : Dvd.dvd p (m.gcd n)
    ⊢ False
  -/
  apply H p hp <;> apply dvd_trans hpdvd
    /-
      case intro.intro.a
      m n : Nat
      H : ∀ (k : Nat), Nat.Prime k → Dvd.dvd k m → Not (Dvd.dvd k n)
      g2 : Not (Eq (m.gcd n) 1)
      p : Nat
      hp : Nat.Prime p
      hpdvd : Dvd.dvd p (m.gcd n)
      ⊢ Dvd.dvd (m.gcd n) m
    -/
  · exact gcd_dvd_left _ _
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.a
      m n : Nat
      H : ∀ (k : Nat), Nat.Prime k → Dvd.dvd k m → Not (Dvd.dvd k n)
      g2 : Not (Eq (m.gcd n) 1)
      p : Nat
      hp : Nat.Prime p
      hpdvd : Dvd.dvd p (m.gcd n)
      ⊢ Dvd.dvd (m.gcd n) n
    -/
  · exact gcd_dvd_right _ _
    /-
      🎉 no goals
    -/


theorem Prime.coprime_iff_not_dvd {p n : ℕ} (pp : Prime p) : Coprime p n ↔ ¬p ∣ n :=
                                                            /-
                                                              p n : Nat
                                                              pp : Nat.Prime p
                                                              co : p.Coprime n
                                                              d : Dvd.dvd p n
                                                              ⊢ Dvd.dvd p (HMul.hMul n 1)
                                                            -/
  ⟨fun co d => pp.not_dvd_one <| co.dvd_of_dvd_mul_left (by simp [d]), fun nd =>
                                                            /-
                                                              🎉 no goals
                                                            -/
    coprime_of_dvd fun _ m2 mp => ((prime_dvd_prime_iff_eq m2 pp).1 mp).symm ▸ nd⟩


theorem Prime.dvd_mul {p m n : ℕ} (pp : Prime p) : p ∣ m * n ↔ p ∣ m ∨ p ∣ n :=
  ⟨fun H => or_iff_not_imp_left.2 fun h => (pp.coprime_iff_not_dvd.2 h).dvd_of_dvd_mul_left H,
    Or.rec (fun h : p ∣ m => h.mul_right _) fun h : p ∣ n => h.mul_left _⟩


theorem prime_iff {p : ℕ} : p.Prime ↔ _root_.Prime p :=
  ⟨fun h => ⟨h.ne_zero, h.not_unit, fun _ _ => h.dvd_mul.mp⟩, Prime.irreducible⟩


alias ⟨Prime.prime, _root_.Prime.nat_prime⟩ := prime_iff


theorem irreducible_iff_prime {p : ℕ} : Irreducible p ↔ _root_.Prime p :=
  prime_iff


/-- The type of prime numbers -/
def Primes :=
  { p : ℕ // p.Prime }
  deriving DecidableEq


instance : Repr Nat.Primes :=
  ⟨fun p _ => repr p.val⟩


instance inhabitedPrimes : Inhabited Primes :=
  ⟨⟨2, prime_two⟩⟩


instance coeNat : Coe Nat.Primes ℕ :=
  ⟨Subtype.val⟩

-- Porting note: change in signature to match change in coercion

theorem coe_nat_injective : Function.Injective (fun (a : Nat.Primes) ↦ (a : ℕ)) :=
  Subtype.coe_injective


theorem coe_nat_inj (p q : Nat.Primes) : (p : ℕ) = (q : ℕ) ↔ p = q :=
  Subtype.ext_iff.symm


instance monoid.primePow {α : Type*} [Monoid α] : Pow α Primes :=
  ⟨fun x p => x ^ (p : ℕ)⟩


instance fact_prime_two : Fact (Prime 2) :=
  ⟨prime_two⟩


instance fact_prime_three : Fact (Prime 3) :=
  ⟨prime_three⟩


