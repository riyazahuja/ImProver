/-- Given `P 0, P 1` and a way to extend `P a` to `P (p ^ n * a)` for prime `p` not dividing `a`,
we can define `P` for all natural numbers. -/
@[elab_as_elim]
def recOnPrimePow {P : ℕ → Sort*} (h0 : P 0) (h1 : P 1)
    (h : ∀ a p n : ℕ, p.Prime → ¬p ∣ a → 0 < n → P a → P (p ^ n * a)) : ∀ a : ℕ, P a := fun a =>
  Nat.strongRecOn' a fun n =>
    match n with
    | 0 => fun _ => h0
    | 1 => fun _ => h1
    | k + 2 => fun hk => by
      /-
        a✝ b m n✝ p : Nat
        P : Nat → Sort u_1
        h0 : P 0
        h1 : P 1
        h : (a p n : Nat) → Nat.Prime p → Not (Dvd.dvd p a) → LT.lt 0 n → P a → P (HMu …
        a n k : Nat
        hk : (m : Nat) → LT.lt m (HAdd.hAdd k 2) → P m
        ⊢ P (HAdd.hAdd k 2)
      -/
      letI p := (k + 2).minFac
      /-
        a✝ b m n✝ p✝ : Nat
        P : Nat → Sort u_1
        h0 : P 0
        h1 : P 1
        h : (a p n : Nat) → Nat.Prime p → Not (Dvd.dvd p a) → LT.lt 0 n → P a → P (HMu …
        a n k : Nat
        hk : (m : Nat) → LT.lt m (HAdd.hAdd k 2) → P m
        p : Nat := (HAdd.hAdd k 2).minFac
        ⊢ P (HAdd.hAdd k 2)
      -/
      haveI hp : Prime p := minFac_prime (succ_succ_ne_one k)
      /-
        a✝ b m n✝ p✝ : Nat
        P : Nat → Sort u_1
        h0 : P 0
        h1 : P 1
        h : (a p n : Nat) → Nat.Prime p → Not (Dvd.dvd p a) → LT.lt 0 n → P a → P (HMu …
        a n k : Nat
        hk : (m : Nat) → LT.lt m (HAdd.hAdd k 2) → P m
        p : Nat := (HAdd.hAdd k 2).minFac
        hp : Nat.Prime p
        ⊢ P (HAdd.hAdd k 2)
      -/
      letI t := (k + 2).factorization p
      /-
        a✝ b m n✝ p✝ : Nat
        P : Nat → Sort u_1
        h0 : P 0
        h1 : P 1
        h : (a p n : Nat) → Nat.Prime p → Not (Dvd.dvd p a) → LT.lt 0 n → P a → P (HMu …
        a n k : Nat
        hk : (m : Nat) → LT.lt m (HAdd.hAdd k 2) → P m
        p : Nat := (HAdd.hAdd k 2).minFac
        hp : Nat.Prime p
        t : Nat := (HAdd.hAdd k 2).factorization p
        ⊢ P (HAdd.hAdd k 2)
      -/
      haveI hpt : p ^ t ∣ k + 2 := ordProj_dvd _ _
      /-
        a✝ b m n✝ p✝ : Nat
        P : Nat → Sort u_1
        h0 : P 0
        h1 : P 1
        h : (a p n : Nat) → Nat.Prime p → Not (Dvd.dvd p a) → LT.lt 0 n → P a → P (HMu …
        a n k : Nat
        hk : (m : Nat) → LT.lt m (HAdd.hAdd k 2) → P m
        p : Nat := (HAdd.hAdd k 2).minFac
        hp : Nat.Prime p
        t : Nat := (HAdd.hAdd k 2).factorization p
        hpt : Dvd.dvd (HPow.hPow p t) (HAdd.hAdd k 2)
        ⊢ P (HAdd.hAdd k 2)
      -/
      haveI htp : 0 < t := hp.factorization_pos_of_dvd (k + 1).succ_ne_zero (k + 2).minFac_dvd
      /-
        a✝ b m n✝ p✝ : Nat
        P : Nat → Sort u_1
        h0 : P 0
        h1 : P 1
        h : (a p n : Nat) → Nat.Prime p → Not (Dvd.dvd p a) → LT.lt 0 n → P a → P (HMu …
        a n k : Nat
        hk : (m : Nat) → LT.lt m (HAdd.hAdd k 2) → P m
        p : Nat := (HAdd.hAdd k 2).minFac
        hp : Nat.Prime p
        t : Nat := (HAdd.hAdd k 2).factorization p
        hpt : Dvd.dvd (HPow.hPow p t) (HAdd.hAdd k 2)
        htp : LT.lt 0 t
        ⊢ P (HAdd.hAdd k 2)
      -/
      convert h ((k + 2) / p ^ t) p t hp _ htp (hk _ (Nat.div_lt_of_lt_mul _)) using 1
        /-
          case h.e'_1
          a✝ b m n✝ p✝ : Nat
          P : Nat → Sort u_1
          h0 : P 0
          h1 : P 1
          h : (a p n : Nat) → Nat.Prime p → Not (Dvd.dvd p a) → LT.lt 0 n → P a → P (HMu …
          a n k : Nat
          hk : (m : Nat) → LT.lt m (HAdd.hAdd k 2) → P m
          p : Nat := (HAdd.hAdd k 2).minFac
          hp : Nat.Prime p
          t : Nat := (HAdd.hAdd k 2).factorization p
          hpt : Dvd.dvd (HPow.hPow p t) (HAdd.hAdd k 2)
          htp : LT.lt 0 t
          ⊢ Eq (HAdd.hAdd k 2) (HMul.hMul (HPow.hPow p t) (HDiv.hDiv (HAdd.hAdd k 2) (HP …
        -/
      · rw [Nat.mul_div_cancel' hpt]
        /-
          🎉 no goals
        -/
        /-
          case convert_1
          a✝ b m n✝ p✝ : Nat
          P : Nat → Sort u_1
          h0 : P 0
          h1 : P 1
          h : (a p n : Nat) → Nat.Prime p → Not (Dvd.dvd p a) → LT.lt 0 n → P a → P (HMu …
          a n k : Nat
          hk : (m : Nat) → LT.lt m (HAdd.hAdd k 2) → P m
          p : Nat := (HAdd.hAdd k 2).minFac
          hp : Nat.Prime p
          t : Nat := (HAdd.hAdd k 2).factorization p
          hpt : Dvd.dvd (HPow.hPow p t) (HAdd.hAdd k 2)
          htp : LT.lt 0 t
          ⊢ Not (Dvd.dvd p (HDiv.hDiv (HAdd.hAdd k 2) (HPow.hPow p t)))
        -/
      · rw [Nat.dvd_div_iff_mul_dvd hpt, ← Nat.pow_succ]
        /-
          case convert_1
          a✝ b m n✝ p✝ : Nat
          P : Nat → Sort u_1
          h0 : P 0
          h1 : P 1
          h : (a p n : Nat) → Nat.Prime p → Not (Dvd.dvd p a) → LT.lt 0 n → P a → P (HMu …
          a n k : Nat
          hk : (m : Nat) → LT.lt m (HAdd.hAdd k 2) → P m
          p : Nat := (HAdd.hAdd k 2).minFac
          hp : Nat.Prime p
          t : Nat := (HAdd.hAdd k 2).factorization p
          hpt : Dvd.dvd (HPow.hPow p t) (HAdd.hAdd k 2)
          htp : LT.lt 0 t
          ⊢ Not (Dvd.dvd (HPow.hPow p t.succ) (HAdd.hAdd k 2))
        -/
        exact pow_succ_factorization_not_dvd (k + 1).succ_ne_zero hp
        /-
          🎉 no goals
        -/
        /-
          case convert_2
          a✝ b m n✝ p✝ : Nat
          P : Nat → Sort u_1
          h0 : P 0
          h1 : P 1
          h : (a p n : Nat) → Nat.Prime p → Not (Dvd.dvd p a) → LT.lt 0 n → P a → P (HMu …
          a n k : Nat
          hk : (m : Nat) → LT.lt m (HAdd.hAdd k 2) → P m
          p : Nat := (HAdd.hAdd k 2).minFac
          hp : Nat.Prime p
          t : Nat := (HAdd.hAdd k 2).factorization p
          hpt : Dvd.dvd (HPow.hPow p t) (HAdd.hAdd k 2)
          htp : LT.lt 0 t
          ⊢ LT.lt (HAdd.hAdd k 2) (HMul.hMul (HPow.hPow p t) (HAdd.hAdd k 2))
        -/
      · simp [lt_mul_iff_one_lt_left Nat.succ_pos', one_lt_pow_iff htp.ne', hp.one_lt]
        /-
          🎉 no goals
        -/


/-- Given `P 0`, `P 1`, and `P (p ^ n)` for positive prime powers, and a way to extend `P a` and
`P b` to `P (a * b)` when `a, b` are positive coprime, we can define `P` for all natural numbers. -/
@[elab_as_elim]
def recOnPosPrimePosCoprime {P : ℕ → Sort*} (hp : ∀ p n : ℕ, Prime p → 0 < n → P (p ^ n))
    (h0 : P 0) (h1 : P 1) (h : ∀ a b, 1 < a → 1 < b → Coprime a b → P a → P b → P (a * b)) :
    ∀ a, P a :=
  recOnPrimePow h0 h1 <| by
    /-
      a b m n p : Nat
      P : Nat → Sort u_1
      hp : (p n : Nat) → Nat.Prime p → LT.lt 0 n → P (HPow.hPow p n)
      h0 : P 0
      h1 : P 1
      h : (a b : Nat) → LT.lt 1 a → LT.lt 1 b → a.Coprime b → P a → P b → P (HMul.hM …
      ⊢ (a p n : Nat) → Nat.Prime p → Not (Dvd.dvd p a) → LT.lt 0 n → P a → P (HMul. …
    -/
    intro a p n hp' hpa hn hPa
    /-
      a✝ b m n✝ p✝ : Nat
      P : Nat → Sort u_1
      hp : (p n : Nat) → Nat.Prime p → LT.lt 0 n → P (HPow.hPow p n)
      h0 : P 0
      h1 : P 1
      h : (a b : Nat) → LT.lt 1 a → LT.lt 1 b → a.Coprime b → P a → P b → P (HMul.hM …
      a p n : Nat
      hp' : Nat.Prime p
      hpa : Not (Dvd.dvd p a)
      hn : LT.lt 0 n
      hPa : P a
      ⊢ P (HMul.hMul (HPow.hPow p n) a)
    -/
    by_cases ha1 : a = 1
      /-
        case pos
        a✝ b m n✝ p✝ : Nat
        P : Nat → Sort u_1
        hp : (p n : Nat) → Nat.Prime p → LT.lt 0 n → P (HPow.hPow p n)
        h0 : P 0
        h1 : P 1
        h : (a b : Nat) → LT.lt 1 a → LT.lt 1 b → a.Coprime b → P a → P b → P (HMul.hM …
        a p n : Nat
        hp' : Nat.Prime p
        hpa : Not (Dvd.dvd p a)
        hn : LT.lt 0 n
        hPa : P a
        ha1 : Eq a 1
        ⊢ P (HMul.hMul (HPow.hPow p n) a)
      -/
    · rw [ha1, mul_one]
      /-
        case pos
        a✝ b m n✝ p✝ : Nat
        P : Nat → Sort u_1
        hp : (p n : Nat) → Nat.Prime p → LT.lt 0 n → P (HPow.hPow p n)
        h0 : P 0
        h1 : P 1
        h : (a b : Nat) → LT.lt 1 a → LT.lt 1 b → a.Coprime b → P a → P b → P (HMul.hM …
        a p n : Nat
        hp' : Nat.Prime p
        hpa : Not (Dvd.dvd p a)
        hn : LT.lt 0 n
        hPa : P a
        ha1 : Eq a 1
        ⊢ P (HPow.hPow p n)
      -/
      exact hp p n hp' hn
      /-
        🎉 no goals
      -/
    /-
      case neg
      a✝ b m n✝ p✝ : Nat
      P : Nat → Sort u_1
      hp : (p n : Nat) → Nat.Prime p → LT.lt 0 n → P (HPow.hPow p n)
      h0 : P 0
      h1 : P 1
      h : (a b : Nat) → LT.lt 1 a → LT.lt 1 b → a.Coprime b → P a → P b → P (HMul.hM …
      a p n : Nat
      hp' : Nat.Prime p
      hpa : Not (Dvd.dvd p a)
      hn : LT.lt 0 n
      hPa : P a
      ha1 : Not (Eq a 1)
      ⊢ P (HMul.hMul (HPow.hPow p n) a)
    -/
    refine h (p ^ n) a (hp'.one_lt.trans_le (le_self_pow hn.ne' _)) ?_ ?_ (hp _ _ hp' hn) hPa
      /-
        case neg.refine_1
        a✝ b m n✝ p✝ : Nat
        P : Nat → Sort u_1
        hp : (p n : Nat) → Nat.Prime p → LT.lt 0 n → P (HPow.hPow p n)
        h0 : P 0
        h1 : P 1
        h : (a b : Nat) → LT.lt 1 a → LT.lt 1 b → a.Coprime b → P a → P b → P (HMul.hM …
        a p n : Nat
        hp' : Nat.Prime p
        hpa : Not (Dvd.dvd p a)
        hn : LT.lt 0 n
        hPa : P a
        ha1 : Not (Eq a 1)
        ⊢ LT.lt 1 a
      -/
    · contrapose! hpa
      /-
        case neg.refine_1
        a✝ b m n✝ p✝ : Nat
        P : Nat → Sort u_1
        hp : (p n : Nat) → Nat.Prime p → LT.lt 0 n → P (HPow.hPow p n)
        h0 : P 0
        h1 : P 1
        h : (a b : Nat) → LT.lt 1 a → LT.lt 1 b → a.Coprime b → P a → P b → P (HMul.hM …
        a p n : Nat
        hp' : Nat.Prime p
        hn : LT.lt 0 n
        hPa : P a
        ha1 : Not (Eq a 1)
        hpa : LE.le a 1
        ⊢ Dvd.dvd p a
      -/
      simp [lt_one_iff.1 (lt_of_le_of_ne hpa ha1)]
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2
        a✝ b m n✝ p✝ : Nat
        P : Nat → Sort u_1
        hp : (p n : Nat) → Nat.Prime p → LT.lt 0 n → P (HPow.hPow p n)
        h0 : P 0
        h1 : P 1
        h : (a b : Nat) → LT.lt 1 a → LT.lt 1 b → a.Coprime b → P a → P b → P (HMul.hM …
        a p n : Nat
        hp' : Nat.Prime p
        hpa : Not (Dvd.dvd p a)
        hn : LT.lt 0 n
        hPa : P a
        ha1 : Not (Eq a 1)
        ⊢ (HPow.hPow p n).Coprime a
      -/
    · simpa [hn, Prime.coprime_iff_not_dvd hp']
      /-
        🎉 no goals
      -/


/-- Given `P 0`, `P (p ^ n)` for all prime powers, and a way to extend `P a` and `P b` to
`P (a * b)` when `a, b` are positive coprime, we can define `P` for all natural numbers. -/
@[elab_as_elim]
def recOnPrimeCoprime {P : ℕ → Sort*} (h0 : P 0) (hp : ∀ p n : ℕ, Prime p → P (p ^ n))
    (h : ∀ a b, 1 < a → 1 < b → Coprime a b → P a → P b → P (a * b)) : ∀ a, P a :=
  recOnPosPrimePosCoprime (fun p n h _ => hp p n h) h0 (hp 2 0 prime_two) h


/-- Given `P 0`, `P 1`, `P p` for all primes, and a way to extend `P a` and `P b` to
`P (a * b)`, we can define `P` for all natural numbers. -/
@[elab_as_elim]
def recOnMul {P : ℕ → Sort*} (h0 : P 0) (h1 : P 1) (hp : ∀ p, Prime p → P p)
    (h : ∀ a b, P a → P b → P (a * b)) : ∀ a, P a :=
  let rec
    /-- The predicate holds on prime powers -/
    hp'' (p n : ℕ) (hp' : Prime p) : P (p ^ n) :=
    match n with
    | 0 => h1
    | n + 1 => h _ _ (hp'' p n hp') (hp p hp')
  recOnPrimeCoprime h0 hp'' fun a b _ _ _ => h a b


lemma _root_.induction_on_primes {P : ℕ → Prop} (h₀ : P 0) (h₁ : P 1)
    (h : ∀ p a : ℕ, p.Prime → P a → P (p * a)) : ∀ n, P n := by
  /-
    P : Nat → Prop
    h₀ : P 0
    h₁ : P 1
    h : ∀ (p a : Nat), Nat.Prime p → P a → P (HMul.hMul p a)
    ⊢ ∀ (n : Nat), P n
  -/
  refine recOnPrimePow h₀ h₁ ?_
  /-
    P : Nat → Prop
    h₀ : P 0
    h₁ : P 1
    h : ∀ (p a : Nat), Nat.Prime p → P a → P (HMul.hMul p a)
    ⊢ ∀ (a p n : Nat), Nat.Prime p → Not (Dvd.dvd p a) → LT.lt 0 n → P a → P (HMul …
  -/
  rintro a p n hp - - ha
  /-
    P : Nat → Prop
    h₀ : P 0
    h₁ : P 1
    h : ∀ (p a : Nat), Nat.Prime p → P a → P (HMul.hMul p a)
    a p n : Nat
    hp : Nat.Prime p
    ha : P a
    ⊢ P (HMul.hMul (HPow.hPow p n) a)
  -/
  induction' n with n ih
    /-
      case zero
      P : Nat → Prop
      h₀ : P 0
      h₁ : P 1
      h : ∀ (p a : Nat), Nat.Prime p → P a → P (HMul.hMul p a)
      a p : Nat
      hp : Nat.Prime p
      ha : P a
      ⊢ P (HMul.hMul (HPow.hPow p 0) a)
    -/
  · simpa using ha
    /-
      🎉 no goals
    -/
    /-
      case succ
      P : Nat → Prop
      h₀ : P 0
      h₁ : P 1
      h : ∀ (p a : Nat), Nat.Prime p → P a → P (HMul.hMul p a)
      a p : Nat
      hp : Nat.Prime p
      ha : P a
      n : Nat
      ih : P (HMul.hMul (HPow.hPow p n) a)
      ⊢ P (HMul.hMul (HPow.hPow p (HAdd.hAdd n 1)) a)
    -/
  · rw [pow_succ', mul_assoc]
    /-
      case succ
      P : Nat → Prop
      h₀ : P 0
      h₁ : P 1
      h : ∀ (p a : Nat), Nat.Prime p → P a → P (HMul.hMul p a)
      a p : Nat
      hp : Nat.Prime p
      ha : P a
      n : Nat
      ih : P (HMul.hMul (HPow.hPow p n) a)
      ⊢ P (HMul.hMul p (HMul.hMul (HPow.hPow p n) a))
    -/
    exact h _ _ hp ih
    /-
      🎉 no goals
    -/


lemma prime_composite_induction {P : ℕ → Prop} (zero : P 0) (one : P 1)
    (prime : ∀ p : ℕ, p.Prime → P p) (composite : ∀ a, 2 ≤ a → P a → ∀ b, 2 ≤ b → P b → P (a * b))
    (n : ℕ) : P n := by
  /-
    P : Nat → Prop
    zero : P 0
    one : P 1
    prime : ∀ (p : Nat), Nat.Prime p → P p
    composite : ∀ (a : Nat), LE.le 2 a → P a → ∀ (b : Nat), LE.le 2 b → P b → P (H …
    n : Nat
    ⊢ P n
  -/
  refine induction_on_primes zero one ?_ _
  /-
    P : Nat → Prop
    zero : P 0
    one : P 1
    prime : ∀ (p : Nat), Nat.Prime p → P p
    composite : ∀ (a : Nat), LE.le 2 a → P a → ∀ (b : Nat), LE.le 2 b → P b → P (H …
    n : Nat
    ⊢ ∀ (p a : Nat), Nat.Prime p → P a → P (HMul.hMul p a)
  -/
  rintro p (_ | _ | a) hp ha
    /-
      case zero
      P : Nat → Prop
      zero : P 0
      one : P 1
      prime : ∀ (p : Nat), Nat.Prime p → P p
      composite : ∀ (a : Nat), LE.le 2 a → P a → ∀ (b : Nat), LE.le 2 b → P b → P (H …
      n p : Nat
      hp : Nat.Prime p
      ha : P 0
      ⊢ P (HMul.hMul p 0)
    -/
  · simpa
    /-
      🎉 no goals
    -/
    /-
      case succ.zero
      P : Nat → Prop
      zero : P 0
      one : P 1
      prime : ∀ (p : Nat), Nat.Prime p → P p
      composite : ∀ (a : Nat), LE.le 2 a → P a → ∀ (b : Nat), LE.le 2 b → P b → P (H …
      n p : Nat
      hp : Nat.Prime p
      ha : P (HAdd.hAdd 0 1)
      ⊢ P (HMul.hMul p (HAdd.hAdd 0 1))
    -/
  · simpa using prime _ hp
    /-
      🎉 no goals
    -/
    /-
      case succ.succ
      P : Nat → Prop
      zero : P 0
      one : P 1
      prime : ∀ (p : Nat), Nat.Prime p → P p
      composite : ∀ (a : Nat), LE.le 2 a → P a → ∀ (b : Nat), LE.le 2 b → P b → P (H …
      n p a : Nat
      hp : Nat.Prime p
      ha : P (HAdd.hAdd (HAdd.hAdd a 1) 1)
      ⊢ P (HMul.hMul p (HAdd.hAdd (HAdd.hAdd a 1) 1))
    -/
  · exact composite _ hp.two_le (prime _ hp) _ a.one_lt_succ_succ ha
    /-
      🎉 no goals
    -/


/-- For any multiplicative function `f` with `f 1 = 1` and any `n ≠ 0`,
we can evaluate `f n` by evaluating `f` at `p ^ k` over the factorization of `n` -/
theorem multiplicative_factorization {β : Type*} [CommMonoid β] (f : ℕ → β)
    (h_mult : ∀ x y : ℕ, Coprime x y → f (x * y) = f x * f y) (hf : f 1 = 1) :
    ∀ {n : ℕ}, n ≠ 0 → f n = n.factorization.prod fun p k => f (p ^ k) := by
  /-
    β : Type u_1
    inst✝ : CommMonoid β
    f : Nat → β
    h_mult : ∀ (x y : Nat), x.Coprime y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x)  …
    hf : Eq (f 1) 1
    ⊢ ∀ {n : Nat}, Ne n 0 → Eq (f n) (n.factorization.prod fun p k => f (HPow.hPow …
  -/
  apply Nat.recOnPosPrimePosCoprime
    /-
      case hp
      β : Type u_1
      inst✝ : CommMonoid β
      f : Nat → β
      h_mult : ∀ (x y : Nat), x.Coprime y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x)  …
      hf : Eq (f 1) 1
      ⊢ ∀ (p n : Nat), Nat.Prime p → LT.lt 0 n → Ne (HPow.hPow p n) 0 → Eq (f (HPow. …
    -/
  · rintro p k hp - -
    -- Porting note: replaced `simp` with `rw`
    /-
      case hp
      β : Type u_1
      inst✝ : CommMonoid β
      f : Nat → β
      h_mult : ∀ (x y : Nat), x.Coprime y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x)  …
      hf : Eq (f 1) 1
      p k : Nat
      hp : Nat.Prime p
      ⊢ Eq (f (HPow.hPow p k)) ((HPow.hPow p k).factorization.prod fun p k => f (HPo …
    -/
    rw [Prime.factorization_pow hp, Finsupp.prod_single_index _]
    /-
      β : Type u_1
      inst✝ : CommMonoid β
      f : Nat → β
      h_mult : ∀ (x y : Nat), x.Coprime y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x)  …
      hf : Eq (f 1) 1
      p k : Nat
      hp : Nat.Prime p
      ⊢ Eq (f (HPow.hPow p 0)) 1
    -/
    rwa [pow_zero]
    /-
      🎉 no goals
    -/
    /-
      case h0
      β : Type u_1
      inst✝ : CommMonoid β
      f : Nat → β
      h_mult : ∀ (x y : Nat), x.Coprime y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x)  …
      hf : Eq (f 1) 1
      ⊢ Ne 0 0 → Eq (f 0) ((Nat.factorization 0).prod fun p k => f (HPow.hPow p k))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h1
      β : Type u_1
      inst✝ : CommMonoid β
      f : Nat → β
      h_mult : ∀ (x y : Nat), x.Coprime y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x)  …
      hf : Eq (f 1) 1
      ⊢ Ne 1 0 → Eq (f 1) ((Nat.factorization 1).prod fun p k => f (HPow.hPow p k))
    -/
  · rintro -
    /-
      case h1
      β : Type u_1
      inst✝ : CommMonoid β
      f : Nat → β
      h_mult : ∀ (x y : Nat), x.Coprime y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x)  …
      hf : Eq (f 1) 1
      ⊢ Eq (f 1) ((Nat.factorization 1).prod fun p k => f (HPow.hPow p k))
    -/
    rw [factorization_one, hf]
    /-
      case h1
      β : Type u_1
      inst✝ : CommMonoid β
      f : Nat → β
      h_mult : ∀ (x y : Nat), x.Coprime y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x)  …
      hf : Eq (f 1) 1
      ⊢ Eq 1 (Finsupp.prod 0 fun p k => f (HPow.hPow p k))
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case h
      β : Type u_1
      inst✝ : CommMonoid β
      f : Nat → β
      h_mult : ∀ (x y : Nat), x.Coprime y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x)  …
      hf : Eq (f 1) 1
      ⊢ ∀ (a b : Nat), LT.lt 1 a → LT.lt 1 b → a.Coprime b → (Ne a 0 → Eq (f a) (a.f …
    -/
  · intro a b _ _ hab ha hb hab_pos
    rw [h_mult a b hab, ha (left_ne_zero_of_mul hab_pos), hb (right_ne_zero_of_mul hab_pos),
      factorization_mul_of_coprime hab, ← prod_add_index_of_disjoint]
    /-
      case h.hd
      β : Type u_1
      inst✝ : CommMonoid β
      f : Nat → β
      h_mult : ∀ (x y : Nat), x.Coprime y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x)  …
      hf : Eq (f 1) 1
      a b : Nat
      a✝¹ : LT.lt 1 a
      a✝ : LT.lt 1 b
      hab : a.Coprime b
      ha : Ne a 0 → Eq (f a) (a.factorization.prod fun p k => f (HPow.hPow p k))
      hb : Ne b 0 → Eq (f b) (b.factorization.prod fun p k => f (HPow.hPow p k))
      hab_pos : Ne (HMul.hMul a b) 0
      ⊢ Disjoint a.factorization.support b.factorization.support
    -/
    exact hab.disjoint_primeFactors
    /-
      🎉 no goals
    -/


/-- For any multiplicative function `f` with `f 1 = 1` and `f 0 = 1`,
we can evaluate `f n` by evaluating `f` at `p ^ k` over the factorization of `n` -/
theorem multiplicative_factorization' {β : Type*} [CommMonoid β] (f : ℕ → β)
    (h_mult : ∀ x y : ℕ, Coprime x y → f (x * y) = f x * f y) (hf0 : f 0 = 1) (hf1 : f 1 = 1) :
    f n = n.factorization.prod fun p k => f (p ^ k) := by
  /-
    n : Nat
    β : Type u_1
    inst✝ : CommMonoid β
    f : Nat → β
    h_mult : ∀ (x y : Nat), x.Coprime y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x)  …
    hf0 : Eq (f 0) 1
    hf1 : Eq (f 1) 1
    ⊢ Eq (f n) (n.factorization.prod fun p k => f (HPow.hPow p k))
  -/
  obtain rfl | hn := eq_or_ne n 0
    /-
      case inl
      β : Type u_1
      inst✝ : CommMonoid β
      f : Nat → β
      h_mult : ∀ (x y : Nat), x.Coprime y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x)  …
      hf0 : Eq (f 0) 1
      hf1 : Eq (f 1) 1
      ⊢ Eq (f 0) ((Nat.factorization 0).prod fun p k => f (HPow.hPow p k))
    -/
  · simpa
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      β : Type u_1
      inst✝ : CommMonoid β
      f : Nat → β
      h_mult : ∀ (x y : Nat), x.Coprime y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x)  …
      hf0 : Eq (f 0) 1
      hf1 : Eq (f 1) 1
      hn : Ne n 0
      ⊢ Eq (f n) (n.factorization.prod fun p k => f (HPow.hPow p k))
    -/
  · exact multiplicative_factorization _ h_mult hf1 hn
    /-
      🎉 no goals
    -/


