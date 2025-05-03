/-- **Fermat's theorem on the sum of two squares**. Every prime not congruent to 3 mod 4 is the sum
of two squares. Also known as **Fermat's Christmas theorem**. -/
theorem Nat.Prime.sq_add_sq {p : ℕ} [Fact p.Prime] (hp : p % 4 ≠ 3) :
    ∃ a b : ℕ, a ^ 2 + b ^ 2 = p := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hp : Ne (HMod.hMod p 4) 3
    ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2 …
  -/
  apply sq_add_sq_of_nat_prime_of_not_irreducible p
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hp : Ne (HMod.hMod p 4) 3
    ⊢ Not (Irreducible ↑p)
  -/
  rwa [_root_.irreducible_iff_prime, prime_iff_mod_four_eq_three_of_nat_prime p]
  /-
    🎉 no goals
  -/


/-- The set of sums of two squares is closed under multiplication in any commutative ring.
See also `sq_add_sq_mul_sq_add_sq`. -/
theorem sq_add_sq_mul {R} [CommRing R] {a b x y u v : R} (ha : a = x ^ 2 + y ^ 2)
    (hb : b = u ^ 2 + v ^ 2) : ∃ r s : R, a * b = r ^ 2 + s ^ 2 :=
                                    /-
                                      R : Type u_1
                                      inst✝ : CommRing R
                                      a b x y u v : R
                                      ha : Eq a (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
                                      hb : Eq b (HAdd.hAdd (HPow.hPow u 2) (HPow.hPow v 2))
                                      ⊢ Eq (HMul.hMul a b) (HAdd.hAdd (HPow.hPow (HSub.hSub (HMul.hMul x u) (HMul.hM …
                                    -/
  ⟨x * u - y * v, x * v + y * u, by rw [ha, hb]; ring⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- The set of natural numbers that are sums of two squares is closed under multiplication. -/
theorem Nat.sq_add_sq_mul {a b x y u v : ℕ} (ha : a = x ^ 2 + y ^ 2) (hb : b = u ^ 2 + v ^ 2) :
    ∃ r s : ℕ, a * b = r ^ 2 + s ^ 2 := by
  /-
    a b x y u v : Nat
    ha : Eq a (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
    hb : Eq b (HAdd.hAdd (HPow.hPow u 2) (HPow.hPow v 2))
    ⊢ Exists fun r => Exists fun s => Eq (HMul.hMul a b) (HAdd.hAdd (HPow.hPow r 2 …
  -/
  zify at ha hb ⊢
  /-
    a b x y u v : Nat
    ha : Eq (↑a) (HAdd.hAdd (HPow.hPow (↑x) 2) (HPow.hPow (↑y) 2))
    hb : Eq (↑b) (HAdd.hAdd (HPow.hPow (↑u) 2) (HPow.hPow (↑v) 2))
    ⊢ Exists fun r => Exists fun s => Eq (HMul.hMul ↑a ↑b) (HAdd.hAdd (HPow.hPow ( …
  -/
  obtain ⟨r, s, h⟩ := _root_.sq_add_sq_mul ha hb
  /-
    case intro.intro
    a b x y u v : Nat
    ha : Eq (↑a) (HAdd.hAdd (HPow.hPow (↑x) 2) (HPow.hPow (↑y) 2))
    hb : Eq (↑b) (HAdd.hAdd (HPow.hPow (↑u) 2) (HPow.hPow (↑v) 2))
    r s : Int
    h : Eq (HMul.hMul ↑a ↑b) (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2))
    ⊢ Exists fun r => Exists fun s => Eq (HMul.hMul ↑a ↑b) (HAdd.hAdd (HPow.hPow ( …
  -/
  refine ⟨r.natAbs, s.natAbs, ?_⟩
  /-
    case intro.intro
    a b x y u v : Nat
    ha : Eq (↑a) (HAdd.hAdd (HPow.hPow (↑x) 2) (HPow.hPow (↑y) 2))
    hb : Eq (↑b) (HAdd.hAdd (HPow.hPow (↑u) 2) (HPow.hPow (↑v) 2))
    r s : Int
    h : Eq (HMul.hMul ↑a ↑b) (HAdd.hAdd (HPow.hPow r 2) (HPow.hPow s 2))
    ⊢ Eq (HMul.hMul ↑a ↑b) (HAdd.hAdd (HPow.hPow (↑r.natAbs) 2) (HPow.hPow (↑s.nat …
  -/
  simpa only [Int.natCast_natAbs, sq_abs]
  /-
    🎉 no goals
  -/


/-- If `-1` is a square modulo `n` and `m` divides `n`, then `-1` is also a square modulo `m`. -/
theorem ZMod.isSquare_neg_one_of_dvd {m n : ℕ} (hd : m ∣ n) (hs : IsSquare (-1 : ZMod n)) :
    IsSquare (-1 : ZMod m) := by
  /-
    m n : Nat
    hd : Dvd.dvd m n
    hs : IsSquare (-1)
    ⊢ IsSquare (-1)
  -/
  let f : ZMod n →+* ZMod m := ZMod.castHom hd _
  /-
    m n : Nat
    hd : Dvd.dvd m n
    hs : IsSquare (-1)
    f : RingHom (ZMod n) (ZMod m) := ZMod.castHom hd (ZMod m)
    ⊢ IsSquare (-1)
  -/
  rw [← RingHom.map_one f, ← RingHom.map_neg]
  /-
    m n : Nat
    hd : Dvd.dvd m n
    hs : IsSquare (-1)
    f : RingHom (ZMod n) (ZMod m) := ZMod.castHom hd (ZMod m)
    ⊢ IsSquare (f (-1))
  -/
  exact hs.map f
  /-
    🎉 no goals
  -/


/-- If `-1` is a square modulo coprime natural numbers `m` and `n`, then `-1` is also
a square modulo `m*n`. -/
theorem ZMod.isSquare_neg_one_mul {m n : ℕ} (hc : m.Coprime n) (hm : IsSquare (-1 : ZMod m))
    (hn : IsSquare (-1 : ZMod n)) : IsSquare (-1 : ZMod (m * n)) := by
  have : IsSquare (-1 : ZMod m × ZMod n) := by
    rw [show (-1 : ZMod m × ZMod n) = ((-1 : ZMod m), (-1 : ZMod n)) from rfl]
    obtain ⟨x, hx⟩ := hm
    obtain ⟨y, hy⟩ := hn
    rw [hx, hy]
    exact ⟨(x, y), rfl⟩
  /-
    m n : Nat
    hc : m.Coprime n
    hm : IsSquare (-1)
    hn : IsSquare (-1)
    this : IsSquare (-1)
    ⊢ IsSquare (-1)
  -/
  simpa only [RingEquiv.map_neg_one] using this.map (ZMod.chineseRemainder hc).symm
  /-
    🎉 no goals
  -/


/-- If a prime `p` divides `n` such that `-1` is a square modulo `n`, then `p % 4 ≠ 3`. -/
theorem Nat.Prime.mod_four_ne_three_of_dvd_isSquare_neg_one {p n : ℕ} (hpp : p.Prime) (hp : p ∣ n)
    (hs : IsSquare (-1 : ZMod n)) : p % 4 ≠ 3 := by
  /-
    p n : Nat
    hpp : Nat.Prime p
    hp : Dvd.dvd p n
    hs : IsSquare (-1)
    ⊢ Ne (HMod.hMod p 4) 3
  -/
  obtain ⟨y, h⟩ := ZMod.isSquare_neg_one_of_dvd hp hs
  /-
    case intro
    p n : Nat
    hpp : Nat.Prime p
    hp : Dvd.dvd p n
    hs : IsSquare (-1)
    y : ZMod p
    h : Eq (-1) (HMul.hMul y y)
    ⊢ Ne (HMod.hMod p 4) 3
  -/
  rw [← sq, eq_comm, show (-1 : ZMod p) = -1 ^ 2 by ring] at h
  /-
    case intro
    p n : Nat
    hpp : Nat.Prime p
    hp : Dvd.dvd p n
    hs : IsSquare (-1)
    y : ZMod p
    h : Eq (HPow.hPow y 2) (Neg.neg (HPow.hPow 1 2))
    ⊢ Ne (HMod.hMod p 4) 3
  -/
  haveI : Fact p.Prime := ⟨hpp⟩
  /-
    case intro
    p n : Nat
    hpp : Nat.Prime p
    hp : Dvd.dvd p n
    hs : IsSquare (-1)
    y : ZMod p
    h : Eq (HPow.hPow y 2) (Neg.neg (HPow.hPow 1 2))
    this : Fact (Nat.Prime p)
    ⊢ Ne (HMod.hMod p 4) 3
  -/
  exact ZMod.mod_four_ne_three_of_sq_eq_neg_sq' one_ne_zero h
  /-
    🎉 no goals
  -/


/-- If `n` is a squarefree natural number, then `-1` is a square modulo `n` if and only if
`n` is not divisible by a prime `q` such that `q % 4 = 3`. -/
theorem ZMod.isSquare_neg_one_iff {n : ℕ} (hn : Squarefree n) :
    IsSquare (-1 : ZMod n) ↔ ∀ {q : ℕ}, q.Prime → q ∣ n → q % 4 ≠ 3 := by
  /-
    n : Nat
    hn : Squarefree n
    ⊢ Iff (IsSquare (-1)) (∀ {q : Nat}, Nat.Prime q → Dvd.dvd q n → Ne (HMod.hMod  …
  -/
  refine ⟨fun H q hqp hqd => hqp.mod_four_ne_three_of_dvd_isSquare_neg_one hqd H, fun H => ?_⟩
  /-
    n : Nat
    hn : Squarefree n
    H : ∀ {q : Nat}, Nat.Prime q → Dvd.dvd q n → Ne (HMod.hMod q 4) 3
    ⊢ IsSquare (-1)
  -/
  induction' n using induction_on_primes with p n hpp ih
    /-
      case h₀
      hn : Squarefree 0
      H : ∀ {q : Nat}, Nat.Prime q → Dvd.dvd q 0 → Ne (HMod.hMod q 4) 3
      ⊢ IsSquare (-1)
    -/
  · exact False.elim (hn.ne_zero rfl)
    /-
      🎉 no goals
    -/
    /-
      case h₁
      hn : Squarefree 1
      H : ∀ {q : Nat}, Nat.Prime q → Dvd.dvd q 1 → Ne (HMod.hMod q 4) 3
      ⊢ IsSquare (-1)
    -/
  · exact ⟨0, by simp only [mul_zero, eq_iff_true_of_subsingleton]⟩
    /-
      🎉 no goals
    -/
    /-
      case h
      p n : Nat
      hpp : Nat.Prime p
      ih : Squarefree n → (∀ {q : Nat}, Nat.Prime q → Dvd.dvd q n → Ne (HMod.hMod q  …
      hn : Squarefree (HMul.hMul p n)
      H : ∀ {q : Nat}, Nat.Prime q → Dvd.dvd q (HMul.hMul p n) → Ne (HMod.hMod q 4) 3
      ⊢ IsSquare (-1)
    -/
  · haveI : Fact p.Prime := ⟨hpp⟩
    have hcp : p.Coprime n := by
      by_contra hc
      exact hpp.not_unit (hn p <| mul_dvd_mul_left p <| hpp.dvd_iff_not_coprime.mpr hc)
    /-
      case h
      p n : Nat
      hpp : Nat.Prime p
      ih : Squarefree n → (∀ {q : Nat}, Nat.Prime q → Dvd.dvd q n → Ne (HMod.hMod q  …
      hn : Squarefree (HMul.hMul p n)
      H : ∀ {q : Nat}, Nat.Prime q → Dvd.dvd q (HMul.hMul p n) → Ne (HMod.hMod q 4) 3
      this : Fact (Nat.Prime p)
      hcp : p.Coprime n
      ⊢ IsSquare (-1)
    -/
    have hp₁ := ZMod.exists_sq_eq_neg_one_iff.mpr (H hpp (dvd_mul_right p n))
    exact ZMod.isSquare_neg_one_mul hcp hp₁
      (ih hn.of_mul_right fun hqp hqd => H hqp <| dvd_mul_of_dvd_right hqd _)


/-- If `n` is a squarefree natural number, then `-1` is a square modulo `n` if and only if
`n` has no divisor `q` that is `≡ 3 mod 4`. -/
theorem ZMod.isSquare_neg_one_iff' {n : ℕ} (hn : Squarefree n) :
    IsSquare (-1 : ZMod n) ↔ ∀ {q : ℕ}, q ∣ n → q % 4 ≠ 3 := by
  /-
    n : Nat
    hn : Squarefree n
    ⊢ Iff (IsSquare (-1)) (∀ {q : Nat}, Dvd.dvd q n → Ne (HMod.hMod q 4) 3)
  -/
  have help : ∀ a b : ZMod 4, a ≠ 3 → b ≠ 3 → a * b ≠ 3 := by decide
  /-
    n : Nat
    hn : Squarefree n
    help : ∀ (a b : ZMod 4), Ne a 3 → Ne b 3 → Ne (HMul.hMul a b) 3
    ⊢ Iff (IsSquare (-1)) (∀ {q : Nat}, Dvd.dvd q n → Ne (HMod.hMod q 4) 3)
  -/
  rw [ZMod.isSquare_neg_one_iff hn]
  /-
    n : Nat
    hn : Squarefree n
    help : ∀ (a b : ZMod 4), Ne a 3 → Ne b 3 → Ne (HMul.hMul a b) 3
    ⊢ Iff (∀ {q : Nat}, Nat.Prime q → Dvd.dvd q n → Ne (HMod.hMod q 4) 3) (∀ {q :  …
  -/
  refine ⟨?_, fun H q _ => H⟩
  /-
    n : Nat
    hn : Squarefree n
    help : ∀ (a b : ZMod 4), Ne a 3 → Ne b 3 → Ne (HMul.hMul a b) 3
    ⊢ (∀ {q : Nat}, Nat.Prime q → Dvd.dvd q n → Ne (HMod.hMod q 4) 3) → ∀ {q : Nat …
  -/
  intro H
  /-
    n : Nat
    hn : Squarefree n
    help : ∀ (a b : ZMod 4), Ne a 3 → Ne b 3 → Ne (HMul.hMul a b) 3
    H : ∀ {q : Nat}, Nat.Prime q → Dvd.dvd q n → Ne (HMod.hMod q 4) 3
    ⊢ ∀ {q : Nat}, Dvd.dvd q n → Ne (HMod.hMod q 4) 3
  -/
  refine @induction_on_primes _ ?_ ?_ (fun p q hp hq hpq => ?_)
    /-
      case refine_1
      n : Nat
      hn : Squarefree n
      help : ∀ (a b : ZMod 4), Ne a 3 → Ne b 3 → Ne (HMul.hMul a b) 3
      H : ∀ {q : Nat}, Nat.Prime q → Dvd.dvd q n → Ne (HMod.hMod q 4) 3
      ⊢ Dvd.dvd 0 n → Ne (HMod.hMod 0 4) 3
    -/
  · exact fun _ => by norm_num
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Nat
      hn : Squarefree n
      help : ∀ (a b : ZMod 4), Ne a 3 → Ne b 3 → Ne (HMul.hMul a b) 3
      H : ∀ {q : Nat}, Nat.Prime q → Dvd.dvd q n → Ne (HMod.hMod q 4) 3
      ⊢ Dvd.dvd 1 n → Ne (HMod.hMod 1 4) 3
    -/
  · exact fun _ => by norm_num
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      n : Nat
      hn : Squarefree n
      help : ∀ (a b : ZMod 4), Ne a 3 → Ne b 3 → Ne (HMul.hMul a b) 3
      H : ∀ {q : Nat}, Nat.Prime q → Dvd.dvd q n → Ne (HMod.hMod q 4) 3
      p q : Nat
      hp : Nat.Prime p
      hq : Dvd.dvd q n → Ne (HMod.hMod q 4) 3
      hpq : Dvd.dvd (HMul.hMul p q) n
      ⊢ Ne (HMod.hMod (HMul.hMul p q) 4) 3
    -/
  · replace hp := H hp (dvd_of_mul_right_dvd hpq)
    /-
      case refine_3
      n : Nat
      hn : Squarefree n
      help : ∀ (a b : ZMod 4), Ne a 3 → Ne b 3 → Ne (HMul.hMul a b) 3
      H : ∀ {q : Nat}, Nat.Prime q → Dvd.dvd q n → Ne (HMod.hMod q 4) 3
      p q : Nat
      hq : Dvd.dvd q n → Ne (HMod.hMod q 4) 3
      hpq : Dvd.dvd (HMul.hMul p q) n
      hp : Ne (HMod.hMod p 4) 3
      ⊢ Ne (HMod.hMod (HMul.hMul p q) 4) 3
    -/
    replace hq := hq (dvd_of_mul_left_dvd hpq)
    /-
      case refine_3
      n : Nat
      hn : Squarefree n
      help : ∀ (a b : ZMod 4), Ne a 3 → Ne b 3 → Ne (HMul.hMul a b) 3
      H : ∀ {q : Nat}, Nat.Prime q → Dvd.dvd q n → Ne (HMod.hMod q 4) 3
      p q : Nat
      hpq : Dvd.dvd (HMul.hMul p q) n
      hp : Ne (HMod.hMod p 4) 3
      hq : Ne (HMod.hMod q 4) 3
      ⊢ Ne (HMod.hMod (HMul.hMul p q) 4) 3
    -/
    rw [show 3 = 3 % 4 by norm_num, Ne, ← ZMod.natCast_eq_natCast_iff'] at hp hq ⊢
    /-
      case refine_3
      n : Nat
      hn : Squarefree n
      help : ∀ (a b : ZMod 4), Ne a 3 → Ne b 3 → Ne (HMul.hMul a b) 3
      H : ∀ {q : Nat}, Nat.Prime q → Dvd.dvd q n → Ne (HMod.hMod q 4) 3
      p q : Nat
      hpq : Dvd.dvd (HMul.hMul p q) n
      hp : Not (Eq ↑p ↑3)
      hq : Not (Eq ↑q ↑3)
      ⊢ Not (Eq ↑(HMul.hMul p q) ↑3)
    -/
    rw [Nat.cast_mul]
    /-
      case refine_3
      n : Nat
      hn : Squarefree n
      help : ∀ (a b : ZMod 4), Ne a 3 → Ne b 3 → Ne (HMul.hMul a b) 3
      H : ∀ {q : Nat}, Nat.Prime q → Dvd.dvd q n → Ne (HMod.hMod q 4) 3
      p q : Nat
      hpq : Dvd.dvd (HMul.hMul p q) n
      hp : Not (Eq ↑p ↑3)
      hq : Not (Eq ↑q ↑3)
      ⊢ Not (Eq (HMul.hMul ↑p ↑q) ↑3)
    -/
    exact help p q hp hq
    /-
      🎉 no goals
    -/


/-- If `-1` is a square modulo the natural number `n`, then `n` is a sum of two squares. -/
theorem Nat.eq_sq_add_sq_of_isSquare_mod_neg_one {n : ℕ} (h : IsSquare (-1 : ZMod n)) :
    ∃ x y : ℕ, n = x ^ 2 + y ^ 2 := by
  /-
    n : Nat
    h : IsSquare (-1)
    ⊢ Exists fun x => Exists fun y => Eq n (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y …
  -/
  induction' n using induction_on_primes with p n hpp ih
    /-
      case h₀
      h : IsSquare (-1)
      ⊢ Exists fun x => Exists fun y => Eq 0 (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y …
    -/
  · exact ⟨0, 0, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h₁
      h : IsSquare (-1)
      ⊢ Exists fun x => Exists fun y => Eq 1 (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y …
    -/
  · exact ⟨0, 1, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h
      p n : Nat
      hpp : Nat.Prime p
      ih : IsSquare (-1) → Exists fun x => Exists fun y => Eq n (HAdd.hAdd (HPow.hPo …
      h : IsSquare (-1)
      ⊢ Exists fun x => Exists fun y => Eq (HMul.hMul p n) (HAdd.hAdd (HPow.hPow x 2 …
    -/
  · haveI : Fact p.Prime := ⟨hpp⟩
    /-
      case h
      p n : Nat
      hpp : Nat.Prime p
      ih : IsSquare (-1) → Exists fun x => Exists fun y => Eq n (HAdd.hAdd (HPow.hPo …
      h : IsSquare (-1)
      this : Fact (Nat.Prime p)
      ⊢ Exists fun x => Exists fun y => Eq (HMul.hMul p n) (HAdd.hAdd (HPow.hPow x 2 …
    -/
    have hp : IsSquare (-1 : ZMod p) := ZMod.isSquare_neg_one_of_dvd ⟨n, rfl⟩ h
    /-
      case h
      p n : Nat
      hpp : Nat.Prime p
      ih : IsSquare (-1) → Exists fun x => Exists fun y => Eq n (HAdd.hAdd (HPow.hPo …
      h : IsSquare (-1)
      this : Fact (Nat.Prime p)
      hp : IsSquare (-1)
      ⊢ Exists fun x => Exists fun y => Eq (HMul.hMul p n) (HAdd.hAdd (HPow.hPow x 2 …
    -/
    obtain ⟨u, v, huv⟩ := Nat.Prime.sq_add_sq (ZMod.exists_sq_eq_neg_one_iff.mp hp)
    /-
      case h.intro.intro
      p n : Nat
      hpp : Nat.Prime p
      ih : IsSquare (-1) → Exists fun x => Exists fun y => Eq n (HAdd.hAdd (HPow.hPo …
      h : IsSquare (-1)
      this : Fact (Nat.Prime p)
      hp : IsSquare (-1)
      u v : Nat
      huv : Eq (HAdd.hAdd (HPow.hPow u 2) (HPow.hPow v 2)) p
      ⊢ Exists fun x => Exists fun y => Eq (HMul.hMul p n) (HAdd.hAdd (HPow.hPow x 2 …
    -/
    obtain ⟨x, y, hxy⟩ := ih (ZMod.isSquare_neg_one_of_dvd ⟨p, mul_comm _ _⟩ h)
    /-
      case h.intro.intro.intro.intro
      p n : Nat
      hpp : Nat.Prime p
      ih : IsSquare (-1) → Exists fun x => Exists fun y => Eq n (HAdd.hAdd (HPow.hPo …
      h : IsSquare (-1)
      this : Fact (Nat.Prime p)
      hp : IsSquare (-1)
      u v : Nat
      huv : Eq (HAdd.hAdd (HPow.hPow u 2) (HPow.hPow v 2)) p
      x y : Nat
      hxy : Eq n (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
      ⊢ Exists fun x => Exists fun y => Eq (HMul.hMul p n) (HAdd.hAdd (HPow.hPow x 2 …
    -/
    exact Nat.sq_add_sq_mul huv.symm hxy
    /-
      🎉 no goals
    -/


/-- If the integer `n` is a sum of two squares of coprime integers,
then `-1` is a square modulo `n`. -/
theorem ZMod.isSquare_neg_one_of_eq_sq_add_sq_of_isCoprime {n x y : ℤ} (h : n = x ^ 2 + y ^ 2)
    (hc : IsCoprime x y) : IsSquare (-1 : ZMod n.natAbs) := by
  obtain ⟨u, v, huv⟩ : IsCoprime x n := by
    have hc2 : IsCoprime (x ^ 2) (y ^ 2) := hc.pow
    rw [show y ^ 2 = n + -1 * x ^ 2 by omega] at hc2
    exact (IsCoprime.pow_left_iff zero_lt_two).mp hc2.of_add_mul_right_right
  have H : u * y * (u * y) - -1 = n * (-v ^ 2 * n + u ^ 2 + 2 * v) := by
    linear_combination -u ^ 2 * h + (n * v - u * x - 1) * huv
  /-
    case intro.intro
    n x y : Int
    h : Eq n (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
    hc : IsCoprime x y
    u v : Int
    huv : Eq (HAdd.hAdd (HMul.hMul u x) (HMul.hMul v n)) 1
    H : Eq (HSub.hSub (HMul.hMul (HMul.hMul u y) (HMul.hMul u y)) (-1)) (HMul.hMul …
    ⊢ IsSquare (-1)
  -/
  refine ⟨u * y, ?_⟩
  /-
    case intro.intro
    n x y : Int
    h : Eq n (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
    hc : IsCoprime x y
    u v : Int
    huv : Eq (HAdd.hAdd (HMul.hMul u x) (HMul.hMul v n)) 1
    H : Eq (HSub.hSub (HMul.hMul (HMul.hMul u y) (HMul.hMul u y)) (-1)) (HMul.hMul …
    ⊢ Eq (-1) (HMul.hMul (HMul.hMul ↑u ↑y) (HMul.hMul ↑u ↑y))
  -/
  conv_rhs => tactic => norm_cast
  /-
    case intro.intro
    n x y : Int
    h : Eq n (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
    hc : IsCoprime x y
    u v : Int
    huv : Eq (HAdd.hAdd (HMul.hMul u x) (HMul.hMul v n)) 1
    H : Eq (HSub.hSub (HMul.hMul (HMul.hMul u y) (HMul.hMul u y)) (-1)) (HMul.hMul …
    ⊢ Eq (-1) ↑(HMul.hMul (HMul.hMul u y) (HMul.hMul u y))
  -/
  rw [(by norm_cast : (-1 : ZMod n.natAbs) = (-1 : ℤ))]
  /-
    case intro.intro
    n x y : Int
    h : Eq n (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
    hc : IsCoprime x y
    u v : Int
    huv : Eq (HAdd.hAdd (HMul.hMul u x) (HMul.hMul v n)) 1
    H : Eq (HSub.hSub (HMul.hMul (HMul.hMul u y) (HMul.hMul u y)) (-1)) (HMul.hMul …
    ⊢ Eq ↑(-1) ↑(HMul.hMul (HMul.hMul u y) (HMul.hMul u y))
  -/
  exact (ZMod.intCast_eq_intCast_iff_dvd_sub _ _ _).mpr (Int.natAbs_dvd.mpr ⟨_, H⟩)
  /-
    🎉 no goals
  -/


/-- If the natural number `n` is a sum of two squares of coprime natural numbers, then
`-1` is a square modulo `n`. -/
theorem ZMod.isSquare_neg_one_of_eq_sq_add_sq_of_coprime {n x y : ℕ} (h : n = x ^ 2 + y ^ 2)
    (hc : x.Coprime y) : IsSquare (-1 : ZMod n) := by
  /-
    n x y : Nat
    h : Eq n (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
    hc : x.Coprime y
    ⊢ IsSquare (-1)
  -/
  zify at h
  /-
    n x y : Nat
    hc : x.Coprime y
    h : Eq (↑n) (HAdd.hAdd (HPow.hPow (↑x) 2) (HPow.hPow (↑y) 2))
    ⊢ IsSquare (-1)
  -/
  exact ZMod.isSquare_neg_one_of_eq_sq_add_sq_of_isCoprime h hc.isCoprime
  /-
    🎉 no goals
  -/


/-- A natural number `n` is a sum of two squares if and only if `n = a^2 * b` with natural
numbers `a` and `b` such that `-1` is a square modulo `b`. -/
theorem Nat.eq_sq_add_sq_iff_eq_sq_mul {n : ℕ} :
    (∃ x y : ℕ, n = x ^ 2 + y ^ 2) ↔ ∃ a b : ℕ, n = a ^ 2 * b ∧ IsSquare (-1 : ZMod b) := by
  /-
    n : Nat
    ⊢ Iff (Exists fun x => Exists fun y => Eq n (HAdd.hAdd (HPow.hPow x 2) (HPow.h …
  -/
  constructor
    /-
      case mp
      n : Nat
      ⊢ (Exists fun x => Exists fun y => Eq n (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow  …
    -/
  · rintro ⟨x, y, h⟩
    /-
      case mp.intro.intro
      n x y : Nat
      h : Eq n (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
      ⊢ Exists fun a => Exists fun b => And (Eq n (HMul.hMul (HPow.hPow a 2) b)) (Is …
    -/
    by_cases hxy : x = 0 ∧ y = 0
    · exact ⟨0, 1, by rw [h, hxy.1, hxy.2, zero_pow two_ne_zero, add_zero, zero_mul],
        ⟨0, by rw [zero_mul, neg_eq_zero, Fin.one_eq_zero_iff]⟩⟩
      /-
        case neg
        n x y : Nat
        h : Eq n (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
        hxy : Not (And (Eq x 0) (Eq y 0))
        ⊢ Exists fun a => Exists fun b => And (Eq n (HMul.hMul (HPow.hPow a 2) b)) (Is …
      -/
    · have hg := Nat.pos_of_ne_zero (mt Nat.gcd_eq_zero_iff.mp hxy)
      /-
        case neg
        n x y : Nat
        h : Eq n (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
        hxy : Not (And (Eq x 0) (Eq y 0))
        hg : LT.lt 0 (x.gcd y)
        ⊢ Exists fun a => Exists fun b => And (Eq n (HMul.hMul (HPow.hPow a 2) b)) (Is …
      -/
      obtain ⟨g, x₁, y₁, _, h₂, h₃, h₄⟩ := Nat.exists_coprime' hg
      exact ⟨g, x₁ ^ 2 + y₁ ^ 2, by rw [h, h₃, h₄]; ring,
        ZMod.isSquare_neg_one_of_eq_sq_add_sq_of_coprime rfl h₂⟩
    /-
      case mpr
      n : Nat
      ⊢ (Exists fun a => Exists fun b => And (Eq n (HMul.hMul (HPow.hPow a 2) b)) (I …
    -/
  · rintro ⟨a, b, h₁, h₂⟩
    /-
      case mpr.intro.intro.intro
      n a b : Nat
      h₁ : Eq n (HMul.hMul (HPow.hPow a 2) b)
      h₂ : IsSquare (-1)
      ⊢ Exists fun x => Exists fun y => Eq n (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y …
    -/
    obtain ⟨x', y', h⟩ := Nat.eq_sq_add_sq_of_isSquare_mod_neg_one h₂
    /-
      case mpr.intro.intro.intro.intro.intro
      n a b : Nat
      h₁ : Eq n (HMul.hMul (HPow.hPow a 2) b)
      h₂ : IsSquare (-1)
      x' y' : Nat
      h : Eq b (HAdd.hAdd (HPow.hPow x' 2) (HPow.hPow y' 2))
      ⊢ Exists fun x => Exists fun y => Eq n (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y …
    -/
    exact ⟨a * x', a * y', by rw [h₁, h]; ring⟩
    /-
      🎉 no goals
    -/


/-- A (positive) natural number `n` is a sum of two squares if and only if the exponent of
every prime `q` such that `q % 4 = 3` in the prime factorization of `n` is even.
(The assumption `0 < n` is not present, since for `n = 0`, both sides are satisfied;
the right hand side holds, since `padicValNat q 0 = 0` by definition.) -/
theorem Nat.eq_sq_add_sq_iff {n : ℕ} :
    (∃ x y : ℕ, n = x ^ 2 + y ^ 2) ↔ ∀ {q : ℕ}, q.Prime → q % 4 = 3 → Even (padicValNat q n) := by
  /-
    n : Nat
    ⊢ Iff (Exists fun x => Exists fun y => Eq n (HAdd.hAdd (HPow.hPow x 2) (HPow.h …
  -/
  rcases n.eq_zero_or_pos with (rfl | hn₀)
    /-
      case inl
      ⊢ Iff (Exists fun x => Exists fun y => Eq 0 (HAdd.hAdd (HPow.hPow x 2) (HPow.h …
    -/
  · exact ⟨fun _ q _ _ => (@padicValNat.zero q).symm ▸ even_zero, fun _ => ⟨0, 0, rfl⟩⟩
    /-
      🎉 no goals
    -/
  -- now `0 < n`
  /-
    case inr
    n : Nat
    hn₀ : GT.gt n 0
    ⊢ Iff (Exists fun x => Exists fun y => Eq n (HAdd.hAdd (HPow.hPow x 2) (HPow.h …
  -/
  rw [Nat.eq_sq_add_sq_iff_eq_sq_mul]
  /-
    case inr
    n : Nat
    hn₀ : GT.gt n 0
    ⊢ Iff (Exists fun a => Exists fun b => And (Eq n (HMul.hMul (HPow.hPow a 2) b) …
  -/
  refine ⟨fun H q hq h => ?_, fun H => ?_⟩
    /-
      case inr.refine_1
      n : Nat
      hn₀ : GT.gt n 0
      H : Exists fun a => Exists fun b => And (Eq n (HMul.hMul (HPow.hPow a 2) b)) ( …
      q : Nat
      hq : Nat.Prime q
      h : Eq (HMod.hMod q 4) 3
      ⊢ Even (padicValNat q n)
    -/
  · obtain ⟨a, b, h₁, h₂⟩ := H
    have hqb := padicValNat.eq_zero_of_not_dvd fun hf =>
      (hq.mod_four_ne_three_of_dvd_isSquare_neg_one hf h₂) h
    /-
      case inr.refine_1.intro.intro.intro
      n : Nat
      hn₀ : GT.gt n 0
      q : Nat
      hq : Nat.Prime q
      h : Eq (HMod.hMod q 4) 3
      a b : Nat
      h₁ : Eq n (HMul.hMul (HPow.hPow a 2) b)
      h₂ : IsSquare (-1)
      hqb : Eq (padicValNat q b) 0
      ⊢ Even (padicValNat q n)
    -/
    have hab : a ^ 2 * b ≠ 0 := h₁ ▸ hn₀.ne'
    /-
      case inr.refine_1.intro.intro.intro
      n : Nat
      hn₀ : GT.gt n 0
      q : Nat
      hq : Nat.Prime q
      h : Eq (HMod.hMod q 4) 3
      a b : Nat
      h₁ : Eq n (HMul.hMul (HPow.hPow a 2) b)
      h₂ : IsSquare (-1)
      hqb : Eq (padicValNat q b) 0
      hab : Ne (HMul.hMul (HPow.hPow a 2) b) 0
      ⊢ Even (padicValNat q n)
    -/
    have ha₂ := left_ne_zero_of_mul hab
    /-
      case inr.refine_1.intro.intro.intro
      n : Nat
      hn₀ : GT.gt n 0
      q : Nat
      hq : Nat.Prime q
      h : Eq (HMod.hMod q 4) 3
      a b : Nat
      h₁ : Eq n (HMul.hMul (HPow.hPow a 2) b)
      h₂ : IsSquare (-1)
      hqb : Eq (padicValNat q b) 0
      hab : Ne (HMul.hMul (HPow.hPow a 2) b) 0
      ha₂ : Ne (HPow.hPow a 2) 0
      ⊢ Even (padicValNat q n)
    -/
    have ha := mt sq_eq_zero_iff.mpr ha₂
    /-
      case inr.refine_1.intro.intro.intro
      n : Nat
      hn₀ : GT.gt n 0
      q : Nat
      hq : Nat.Prime q
      h : Eq (HMod.hMod q 4) 3
      a b : Nat
      h₁ : Eq n (HMul.hMul (HPow.hPow a 2) b)
      h₂ : IsSquare (-1)
      hqb : Eq (padicValNat q b) 0
      hab : Ne (HMul.hMul (HPow.hPow a 2) b) 0
      ha₂ : Ne (HPow.hPow a 2) 0
      ha : Not (Eq a 0)
      ⊢ Even (padicValNat q n)
    -/
    have hb := right_ne_zero_of_mul hab
    /-
      case inr.refine_1.intro.intro.intro
      n : Nat
      hn₀ : GT.gt n 0
      q : Nat
      hq : Nat.Prime q
      h : Eq (HMod.hMod q 4) 3
      a b : Nat
      h₁ : Eq n (HMul.hMul (HPow.hPow a 2) b)
      h₂ : IsSquare (-1)
      hqb : Eq (padicValNat q b) 0
      hab : Ne (HMul.hMul (HPow.hPow a 2) b) 0
      ha₂ : Ne (HPow.hPow a 2) 0
      ha : Not (Eq a 0)
      hb : Ne b 0
      ⊢ Even (padicValNat q n)
    -/
    haveI hqi : Fact q.Prime := ⟨hq⟩
    /-
      case inr.refine_1.intro.intro.intro
      n : Nat
      hn₀ : GT.gt n 0
      q : Nat
      hq : Nat.Prime q
      h : Eq (HMod.hMod q 4) 3
      a b : Nat
      h₁ : Eq n (HMul.hMul (HPow.hPow a 2) b)
      h₂ : IsSquare (-1)
      hqb : Eq (padicValNat q b) 0
      hab : Ne (HMul.hMul (HPow.hPow a 2) b) 0
      ha₂ : Ne (HPow.hPow a 2) 0
      ha : Not (Eq a 0)
      hb : Ne b 0
      hqi : Fact (Nat.Prime q)
      ⊢ Even (padicValNat q n)
    -/
    simp_rw [h₁, padicValNat.mul ha₂ hb, padicValNat.pow 2 ha, hqb, add_zero]
    /-
      case inr.refine_1.intro.intro.intro
      n : Nat
      hn₀ : GT.gt n 0
      q : Nat
      hq : Nat.Prime q
      h : Eq (HMod.hMod q 4) 3
      a b : Nat
      h₁ : Eq n (HMul.hMul (HPow.hPow a 2) b)
      h₂ : IsSquare (-1)
      hqb : Eq (padicValNat q b) 0
      hab : Ne (HMul.hMul (HPow.hPow a 2) b) 0
      ha₂ : Ne (HPow.hPow a 2) 0
      ha : Not (Eq a 0)
      hb : Ne b 0
      hqi : Fact (Nat.Prime q)
      ⊢ Even (HMul.hMul 2 (padicValNat q a))
    -/
    exact even_two_mul _
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_2
      n : Nat
      hn₀ : GT.gt n 0
      H : ∀ {q : Nat}, Nat.Prime q → Eq (HMod.hMod q 4) 3 → Even (padicValNat q n)
      ⊢ Exists fun a => Exists fun b => And (Eq n (HMul.hMul (HPow.hPow a 2) b)) (Is …
    -/
  · obtain ⟨b, a, hb₀, ha₀, hab, hb⟩ := Nat.sq_mul_squarefree_of_pos hn₀
    /-
      case inr.refine_2.intro.intro.intro.intro.intro
      n : Nat
      hn₀ : GT.gt n 0
      H : ∀ {q : Nat}, Nat.Prime q → Eq (HMod.hMod q 4) 3 → Even (padicValNat q n)
      b a : Nat
      hb₀ : LT.lt 0 b
      ha₀ : LT.lt 0 a
      hab : Eq (HMul.hMul (HPow.hPow a 2) b) n
      hb : Squarefree b
      ⊢ Exists fun a => Exists fun b => And (Eq n (HMul.hMul (HPow.hPow a 2) b)) (Is …
    -/
    refine ⟨a, b, hab.symm, (ZMod.isSquare_neg_one_iff hb).mpr fun {q} hqp hqb hq4 => ?_⟩
    /-
      case inr.refine_2.intro.intro.intro.intro.intro
      n : Nat
      hn₀ : GT.gt n 0
      H : ∀ {q : Nat}, Nat.Prime q → Eq (HMod.hMod q 4) 3 → Even (padicValNat q n)
      b a : Nat
      hb₀ : LT.lt 0 b
      ha₀ : LT.lt 0 a
      hab : Eq (HMul.hMul (HPow.hPow a 2) b) n
      hb : Squarefree b
      q : Nat
      hqp : Nat.Prime q
      hqb : Dvd.dvd q b
      hq4 : Eq (HMod.hMod q 4) 3
      ⊢ False
    -/
    refine Nat.not_even_iff_odd.2 ?_ (H hqp hq4)
    have hqb' : padicValNat q b = 1 :=
      b.factorization_def hqp ▸ le_antisymm (hb.natFactorization_le_one _)
        ((hqp.dvd_iff_one_le_factorization hb₀.ne').mp hqb)
    /-
      case inr.refine_2.intro.intro.intro.intro.intro
      n : Nat
      hn₀ : GT.gt n 0
      H : ∀ {q : Nat}, Nat.Prime q → Eq (HMod.hMod q 4) 3 → Even (padicValNat q n)
      b a : Nat
      hb₀ : LT.lt 0 b
      ha₀ : LT.lt 0 a
      hab : Eq (HMul.hMul (HPow.hPow a 2) b) n
      hb : Squarefree b
      q : Nat
      hqp : Nat.Prime q
      hqb : Dvd.dvd q b
      hq4 : Eq (HMod.hMod q 4) 3
      hqb' : Eq (padicValNat q b) 1
      ⊢ Odd (padicValNat q n)
    -/
    haveI hqi : Fact q.Prime := ⟨hqp⟩
    simp_rw [← hab, padicValNat.mul (pow_ne_zero 2 ha₀.ne') hb₀.ne', hqb',
      padicValNat.pow 2 ha₀.ne']
    /-
      case inr.refine_2.intro.intro.intro.intro.intro
      n : Nat
      hn₀ : GT.gt n 0
      H : ∀ {q : Nat}, Nat.Prime q → Eq (HMod.hMod q 4) 3 → Even (padicValNat q n)
      b a : Nat
      hb₀ : LT.lt 0 b
      ha₀ : LT.lt 0 a
      hab : Eq (HMul.hMul (HPow.hPow a 2) b) n
      hb : Squarefree b
      q : Nat
      hqp : Nat.Prime q
      hqb : Dvd.dvd q b
      hq4 : Eq (HMod.hMod q 4) 3
      hqb' : Eq (padicValNat q b) 1
      hqi : Fact (Nat.Prime q)
      ⊢ Odd (HAdd.hAdd (HMul.hMul 2 (padicValNat q a)) 1)
    -/
    exact odd_two_mul_add_one _
    /-
      🎉 no goals
    -/


