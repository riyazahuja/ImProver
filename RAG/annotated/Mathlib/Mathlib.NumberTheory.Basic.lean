theorem dvd_sub_pow_of_dvd_sub {R : Type*} [CommRing R] {p : ℕ} {a b : R} (h : (p : R) ∣ a - b)
    (k : ℕ) : (p ^ (k + 1) : R) ∣ a ^ p ^ k - b ^ p ^ k := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    p : Nat
    a b : R
    h : Dvd.dvd (↑p) (HSub.hSub a b)
    k : Nat
    ⊢ Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd k 1)) (HSub.hSub (HPow.hPow a (HPow.hPow  …
  -/
  induction' k with k ih
    /-
      case zero
      R : Type u_1
      inst✝ : CommRing R
      p : Nat
      a b : R
      h : Dvd.dvd (↑p) (HSub.hSub a b)
      ⊢ Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd 0 1)) (HSub.hSub (HPow.hPow a (HPow.hPow  …
    -/
  · rwa [pow_one, pow_zero, pow_one, pow_one]
    /-
      🎉 no goals
    -/
  /-
    case succ
    R : Type u_1
    inst✝ : CommRing R
    p : Nat
    a b : R
    h : Dvd.dvd (↑p) (HSub.hSub a b)
    k : Nat
    ih : Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd k 1)) (HSub.hSub (HPow.hPow a (HPow.hP …
    ⊢ Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd (HAdd.hAdd k 1) 1)) (HSub.hSub (HPow.hPow …
  -/
  rw [pow_succ p k, pow_mul, pow_mul, ← geom_sum₂_mul, pow_succ']
  /-
    case succ
    R : Type u_1
    inst✝ : CommRing R
    p : Nat
    a b : R
    h : Dvd.dvd (↑p) (HSub.hSub a b)
    k : Nat
    ih : Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd k 1)) (HSub.hSub (HPow.hPow a (HPow.hP …
    ⊢ Dvd.dvd (HMul.hMul (↑p) (HPow.hPow (↑p) (HAdd.hAdd k 1))) (HMul.hMul ((Finse …
  -/
  refine mul_dvd_mul ?_ ih
  /-
    case succ
    R : Type u_1
    inst✝ : CommRing R
    p : Nat
    a b : R
    h : Dvd.dvd (↑p) (HSub.hSub a b)
    k : Nat
    ih : Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd k 1)) (HSub.hSub (HPow.hPow a (HPow.hP …
    ⊢ Dvd.dvd (↑p) ((Finset.range p).sum fun i => HMul.hMul (HPow.hPow (HPow.hPow  …
  -/
  let f : R →+* R ⧸ span {(p : R)} := mk (span {(p : R)})
  /-
    case succ
    R : Type u_1
    inst✝ : CommRing R
    p : Nat
    a b : R
    h : Dvd.dvd (↑p) (HSub.hSub a b)
    k : Nat
    ih : Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd k 1)) (HSub.hSub (HPow.hPow a (HPow.hP …
    f : RingHom R (HasQuotient.Quotient R (Ideal.span (Singleton.singleton ↑p))) : …
    ⊢ Dvd.dvd (↑p) ((Finset.range p).sum fun i => HMul.hMul (HPow.hPow (HPow.hPow  …
  -/
  have hf : ∀ r : R, (p : R) ∣ r ↔ f r = 0 := fun r ↦ by rw [eq_zero_iff_mem, mem_span_singleton]
  /-
    case succ
    R : Type u_1
    inst✝ : CommRing R
    p : Nat
    a b : R
    h : Dvd.dvd (↑p) (HSub.hSub a b)
    k : Nat
    ih : Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd k 1)) (HSub.hSub (HPow.hPow a (HPow.hP …
    f : RingHom R (HasQuotient.Quotient R (Ideal.span (Singleton.singleton ↑p))) : …
    hf : ∀ (r : R), Iff (Dvd.dvd (↑p) r) (Eq (f r) 0)
    ⊢ Dvd.dvd (↑p) ((Finset.range p).sum fun i => HMul.hMul (HPow.hPow (HPow.hPow  …
  -/
  rw [hf, map_sub, sub_eq_zero] at h
  /-
    case succ
    R : Type u_1
    inst✝ : CommRing R
    p : Nat
    a b : R
    k : Nat
    ih : Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd k 1)) (HSub.hSub (HPow.hPow a (HPow.hP …
    f : RingHom R (HasQuotient.Quotient R (Ideal.span (Singleton.singleton ↑p))) : …
    h : Eq (f a) (f b)
    hf : ∀ (r : R), Iff (Dvd.dvd (↑p) r) (Eq (f r) 0)
    ⊢ Dvd.dvd (↑p) ((Finset.range p).sum fun i => HMul.hMul (HPow.hPow (HPow.hPow  …
  -/
  rw [hf, RingHom.map_geom_sum₂, map_pow, map_pow, h, geom_sum₂_self, mul_eq_zero_of_left]
  /-
    case succ.h
    R : Type u_1
    inst✝ : CommRing R
    p : Nat
    a b : R
    k : Nat
    ih : Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd k 1)) (HSub.hSub (HPow.hPow a (HPow.hP …
    f : RingHom R (HasQuotient.Quotient R (Ideal.span (Singleton.singleton ↑p))) : …
    h : Eq (f a) (f b)
    hf : ∀ (r : R), Iff (Dvd.dvd (↑p) r) (Eq (f r) 0)
    ⊢ Eq (↑p) 0
  -/
  rw [← map_natCast f, eq_zero_iff_mem, mem_span_singleton]
  /-
    🎉 no goals
  -/


