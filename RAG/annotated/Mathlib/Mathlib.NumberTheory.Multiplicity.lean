theorem dvd_geom_sum₂_iff_of_dvd_sub {x y p : R} (h : p ∣ x - y) :
    (p ∣ ∑ i ∈ range n, x ^ i * y ^ (n - 1 - i)) ↔ p ∣ n * y ^ (n - 1) := by
  /-
    R : Type u_1
    n : Nat
    inst✝ : CommRing R
    x y p : R
    h : Dvd.dvd p (HSub.hSub x y)
    ⊢ Iff (Dvd.dvd p ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow x i) (HPo …
  -/
  rw [← mem_span_singleton, ← Ideal.Quotient.eq] at h
  simp only [← mem_span_singleton, ← eq_zero_iff_mem, RingHom.map_geom_sum₂, h, geom_sum₂_self,
    _root_.map_mul, map_pow, map_natCast]


theorem dvd_geom_sum₂_iff_of_dvd_sub' {x y p : R} (h : p ∣ x - y) :
    (p ∣ ∑ i ∈ range n, x ^ i * y ^ (n - 1 - i)) ↔ p ∣ n * x ^ (n - 1) := by
  /-
    R : Type u_1
    n : Nat
    inst✝ : CommRing R
    x y p : R
    h : Dvd.dvd p (HSub.hSub x y)
    ⊢ Iff (Dvd.dvd p ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow x i) (HPo …
  -/
  rw [geom_sum₂_comm, dvd_geom_sum₂_iff_of_dvd_sub]; simpa using h.neg_right
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem dvd_geom_sum₂_self {x y : R} (h : ↑n ∣ x - y) :
    ↑n ∣ ∑ i ∈ range n, x ^ i * y ^ (n - 1 - i) :=
  (dvd_geom_sum₂_iff_of_dvd_sub h).mpr (dvd_mul_right _ _)


theorem sq_dvd_add_pow_sub_sub (p x : R) (n : ℕ) :
    p ^ 2 ∣ (x + p) ^ n - x ^ (n - 1) * p * n - x ^ n := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    p x : R
    n : Nat
    ⊢ Dvd.dvd (HPow.hPow p 2) (HSub.hSub (HSub.hSub (HPow.hPow (HAdd.hAdd x p) n)  …
  -/
  cases' n with n n
    /-
      case zero
      R : Type u_1
      inst✝ : CommRing R
      p x : R
      ⊢ Dvd.dvd (HPow.hPow p 2) (HSub.hSub (HSub.hSub (HPow.hPow (HAdd.hAdd x p) 0)  …
    -/
  · simp only [pow_zero, Nat.cast_zero, sub_zero, sub_self, dvd_zero, mul_zero]
    /-
      🎉 no goals
    -/
  · simp only [Nat.succ_sub_succ_eq_sub, tsub_zero, Nat.cast_succ, add_pow, Finset.sum_range_succ,
      Nat.choose_self, Nat.succ_sub _, tsub_self, pow_one, Nat.choose_succ_self_right, pow_zero,
      mul_one, Nat.cast_zero, zero_add, Nat.succ_eq_add_one, add_tsub_cancel_left]
    suffices p ^ 2 ∣ ∑ i ∈ range n, x ^ i * p ^ (n + 1 - i) * ↑((n + 1).choose i) by
      convert this; abel
    /-
      case succ
      R : Type u_1
      inst✝ : CommRing R
      p x : R
      n : Nat
      ⊢ Dvd.dvd (HPow.hPow p 2) ((Finset.range n).sum fun i => HMul.hMul (HMul.hMul  …
    -/
    apply Finset.dvd_sum
    /-
      case succ.h
      R : Type u_1
      inst✝ : CommRing R
      p x : R
      n : Nat
      ⊢ ∀ (i : Nat), Membership.mem (Finset.range n) i → Dvd.dvd (HPow.hPow p 2) (HM …
    -/
    intro y hy
    calc
      p ^ 2 ∣ p ^ (n + 1 - y) :=
        pow_dvd_pow p (le_tsub_of_add_le_left (by linarith [Finset.mem_range.mp hy]))
      _ ∣ x ^ y * p ^ (n + 1 - y) * ↑((n + 1).choose y) :=
        dvd_mul_of_dvd_left (dvd_mul_left _ _) _


theorem not_dvd_geom_sum₂ {p : R} (hp : Prime p) (hxy : p ∣ x - y) (hx : ¬p ∣ x) (hn : ¬p ∣ n) :
    ¬p ∣ ∑ i ∈ range n, x ^ i * y ^ (n - 1 - i) := fun h =>
  hx <|
    hp.dvd_of_dvd_pow <| (hp.dvd_or_dvd <| (dvd_geom_sum₂_iff_of_dvd_sub' hxy).mp h).resolve_left hn


theorem odd_sq_dvd_geom_sum₂_sub (hp : Odd p) :
    (p : R) ^ 2 ∣ (∑ i ∈ range p, (a + p * b) ^ i * a ^ (p - 1 - i)) - p * a ^ (p - 1) := by
  have h1 : ∀ (i : ℕ),
      (p : R) ^ 2 ∣ (a + ↑p * b) ^ i - (a ^ (i - 1) * (↑p * b) * i + a ^ i) := by
    intro i
    calc
      ↑p ^ 2 ∣ (↑p * b) ^ 2 := by simp only [mul_pow, dvd_mul_right]
      _ ∣ (a + ↑p * b) ^ i - (a ^ (i - 1) * (↑p * b) * ↑i + a ^ i) := by
        simp only [sq_dvd_add_pow_sub_sub (↑p * b) a i, ← sub_sub]
  /-
    R : Type u_1
    inst✝ : CommRing R
    a b : R
    p : Nat
    hp : Odd p
    h1 : ∀ (i : Nat), Dvd.dvd (HPow.hPow (↑p) 2) (HSub.hSub (HPow.hPow (HAdd.hAdd  …
    ⊢ Dvd.dvd (HPow.hPow (↑p) 2) (HSub.hSub ((Finset.range p).sum fun i => HMul.hM …
  -/
  simp_rw [← mem_span_singleton, ← Ideal.Quotient.eq] at *
  /-
    R : Type u_1
    inst✝ : CommRing R
    a b : R
    p : Nat
    hp : Odd p
    h1 : ∀ (i : Nat), Eq ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton (HPo …
    ⊢ Eq ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton (HPow.hPow (↑p) 2))) …
  -/
  let s : R := (p : R)^2
  calc
    (Ideal.Quotient.mk (span {s})) (∑ i ∈ range p, (a + (p : R) * b) ^ i * a ^ (p - 1 - i)) =
        ∑ i ∈ Finset.range p,
        mk (span {s}) ((a ^ (i - 1) * (↑p * b) * ↑i + a ^ i) * a ^ (p - 1 - i)) := by
      simp_rw [s, RingHom.map_geom_sum₂, ← map_pow, h1, ← _root_.map_mul]
    _ =
        mk (span {s})
            (∑ x ∈ Finset.range p, a ^ (x - 1) * (a ^ (p - 1 - x) * (↑p * (b * ↑x)))) +
          mk (span {s}) (∑ x ∈ Finset.range p, a ^ (x + (p - 1 - x))) := by
      ring_nf
      simp only [← pow_add, map_add, Finset.sum_add_distrib, ← map_sum]
      congr
      simp [pow_add a, mul_assoc]
    _ =
        mk (span {s})
            (∑ x ∈ Finset.range p, a ^ (x - 1) * (a ^ (p - 1 - x) * (↑p * (b * ↑x)))) +
          mk (span {s}) (∑ _x ∈ Finset.range p, a ^ (p - 1)) := by
      rw [add_right_inj]
      have : ∀ (x : ℕ), (hx : x ∈ range p) → a ^ (x + (p - 1 - x)) = a ^ (p - 1) := by
        intro x hx
        rw [← Nat.add_sub_assoc _ x, Nat.add_sub_cancel_left]
        exact Nat.le_sub_one_of_lt (Finset.mem_range.mp hx)
      rw [Finset.sum_congr rfl this]
    _ =
        mk (span {s})
            (∑ x ∈ Finset.range p, a ^ (x - 1) * (a ^ (p - 1 - x) * (↑p * (b * ↑x)))) +
          mk (span {s}) (↑p * a ^ (p - 1)) := by
      simp only [add_right_inj, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    _ =
        mk (span {s}) (↑p * b * ∑ x ∈ Finset.range p, a ^ (p - 2) * x) +
          mk (span {s}) (↑p * a ^ (p - 1)) := by
      simp only [Finset.mul_sum, ← mul_assoc, ← pow_add]
      rw [Finset.sum_congr rfl]
      rintro (⟨⟩ | ⟨x⟩) hx
      · rw [Nat.cast_zero, mul_zero, mul_zero]
      · have : x.succ - 1 + (p - 1 - x.succ) = p - 2 := by
          rw [← Nat.add_sub_assoc (Nat.le_sub_one_of_lt (Finset.mem_range.mp hx))]
          exact congr_arg Nat.pred (Nat.add_sub_cancel_left _ _)
        rw [this]
        ring1
    _ = mk (span {s}) (↑p * a ^ (p - 1)) := by
      have : Finset.sum (range p) (fun (x : ℕ) ↦ (x : R)) =
          ((Finset.sum (range p) (fun (x : ℕ) ↦ (x : ℕ)))) := by simp only [Nat.cast_sum]
      simp only [add_left_eq_self, ← Finset.mul_sum, this]
      norm_cast
      simp only [Finset.sum_range_id]
      norm_cast
      simp only [Nat.cast_mul, _root_.map_mul,
          Nat.mul_div_assoc p (even_iff_two_dvd.mp (Nat.Odd.sub_odd hp odd_one))]
      ring_nf
      rw [mul_assoc, mul_assoc]
      refine mul_eq_zero_of_left ?_ _
      refine Ideal.Quotient.eq_zero_iff_mem.mpr ?_
      simp [s, mem_span_singleton]


theorem emultiplicity_pow_sub_pow_of_prime {p : R} (hp : Prime p) {x y : R}
    (hxy : p ∣ x - y) (hx : ¬p ∣ x) {n : ℕ} (hn : ¬p ∣ n) :
    emultiplicity p (x ^ n - y ^ n) = emultiplicity p (x - y) := by
  rw [← geom_sum₂_mul, emultiplicity_mul hp,
    emultiplicity_eq_zero.2 (not_dvd_geom_sum₂ hp hxy hx hn), zero_add]


@[deprecated (since := "2024-11-30")]
alias multiplicity.pow_sub_pow_of_prime := emultiplicity_pow_sub_pow_of_prime


theorem emultiplicity_geom_sum₂_eq_one :
    emultiplicity (↑p) (∑ i ∈ range p, x ^ i * y ^ (p - 1 - i)) = 1 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    x y : R
    p : Nat
    inst✝ : IsDomain R
    hp : Prime ↑p
    hp1 : Odd p
    hxy : Dvd.dvd (↑p) (HSub.hSub x y)
    hx : Not (Dvd.dvd (↑p) x)
    ⊢ Eq (emultiplicity (↑p) ((Finset.range p).sum fun i => HMul.hMul (HPow.hPow x …
  -/
  rw [← Nat.cast_one]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    x y : R
    p : Nat
    inst✝ : IsDomain R
    hp : Prime ↑p
    hp1 : Odd p
    hxy : Dvd.dvd (↑p) (HSub.hSub x y)
    hx : Not (Dvd.dvd (↑p) x)
    ⊢ Eq (emultiplicity (↑p) ((Finset.range p).sum fun i => HMul.hMul (HPow.hPow x …
  -/
  refine emultiplicity_eq_coe.2 ⟨?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝¹ : CommRing R
      x y : R
      p : Nat
      inst✝ : IsDomain R
      hp : Prime ↑p
      hp1 : Odd p
      hxy : Dvd.dvd (↑p) (HSub.hSub x y)
      hx : Not (Dvd.dvd (↑p) x)
      ⊢ Dvd.dvd (HPow.hPow (↑p) 1) ((Finset.range p).sum fun i => HMul.hMul (HPow.hP …
    -/
  · rw [pow_one]
    /-
      case refine_1
      R : Type u_1
      inst✝¹ : CommRing R
      x y : R
      p : Nat
      inst✝ : IsDomain R
      hp : Prime ↑p
      hp1 : Odd p
      hxy : Dvd.dvd (↑p) (HSub.hSub x y)
      hx : Not (Dvd.dvd (↑p) x)
      ⊢ Dvd.dvd (↑p) ((Finset.range p).sum fun i => HMul.hMul (HPow.hPow x i) (HPow. …
    -/
    exact dvd_geom_sum₂_self hxy
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    R : Type u_1
    inst✝¹ : CommRing R
    x y : R
    p : Nat
    inst✝ : IsDomain R
    hp : Prime ↑p
    hp1 : Odd p
    hxy : Dvd.dvd (↑p) (HSub.hSub x y)
    hx : Not (Dvd.dvd (↑p) x)
    ⊢ Not (Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd 1 1)) ((Finset.range p).sum fun i => …
  -/
  rw [dvd_iff_dvd_of_dvd_sub hxy] at hx
  /-
    case refine_2
    R : Type u_1
    inst✝¹ : CommRing R
    x y : R
    p : Nat
    inst✝ : IsDomain R
    hp : Prime ↑p
    hp1 : Odd p
    hxy : Dvd.dvd (↑p) (HSub.hSub x y)
    hx : Not (Dvd.dvd (↑p) y)
    ⊢ Not (Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd 1 1)) ((Finset.range p).sum fun i => …
  -/
  cases' hxy with k hk
  /-
    case refine_2.intro
    R : Type u_1
    inst✝¹ : CommRing R
    x y : R
    p : Nat
    inst✝ : IsDomain R
    hp : Prime ↑p
    hp1 : Odd p
    hx : Not (Dvd.dvd (↑p) y)
    k : R
    hk : Eq (HSub.hSub x y) (HMul.hMul (↑p) k)
    ⊢ Not (Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd 1 1)) ((Finset.range p).sum fun i => …
  -/
  rw [one_add_one_eq_two, eq_add_of_sub_eq' hk]
  /-
    case refine_2.intro
    R : Type u_1
    inst✝¹ : CommRing R
    x y : R
    p : Nat
    inst✝ : IsDomain R
    hp : Prime ↑p
    hp1 : Odd p
    hx : Not (Dvd.dvd (↑p) y)
    k : R
    hk : Eq (HSub.hSub x y) (HMul.hMul (↑p) k)
    ⊢ Not (Dvd.dvd (HPow.hPow (↑p) 2) ((Finset.range p).sum fun i => HMul.hMul (HP …
  -/
  refine mt (dvd_iff_dvd_of_dvd_sub (@odd_sq_dvd_geom_sum₂_sub _ _ y k _ hp1)).mp ?_
  /-
    case refine_2.intro
    R : Type u_1
    inst✝¹ : CommRing R
    x y : R
    p : Nat
    inst✝ : IsDomain R
    hp : Prime ↑p
    hp1 : Odd p
    hx : Not (Dvd.dvd (↑p) y)
    k : R
    hk : Eq (HSub.hSub x y) (HMul.hMul (↑p) k)
    ⊢ Not (Dvd.dvd (HPow.hPow (↑p) 2) (HMul.hMul (↑p) (HPow.hPow y (HSub.hSub p 1) …
  -/
  rw [pow_two, mul_dvd_mul_iff_left hp.ne_zero]
  /-
    case refine_2.intro
    R : Type u_1
    inst✝¹ : CommRing R
    x y : R
    p : Nat
    inst✝ : IsDomain R
    hp : Prime ↑p
    hp1 : Odd p
    hx : Not (Dvd.dvd (↑p) y)
    k : R
    hk : Eq (HSub.hSub x y) (HMul.hMul (↑p) k)
    ⊢ Not (Dvd.dvd (↑p) (HPow.hPow y (HSub.hSub p 1)))
  -/
  exact mt hp.dvd_of_dvd_pow hx
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.geom_sum₂_eq_one := emultiplicity_geom_sum₂_eq_one


theorem emultiplicity_pow_prime_sub_pow_prime :
    emultiplicity (↑p) (x ^ p - y ^ p) = emultiplicity (↑p) (x - y) + 1 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    x y : R
    p : Nat
    inst✝ : IsDomain R
    hp : Prime ↑p
    hp1 : Odd p
    hxy : Dvd.dvd (↑p) (HSub.hSub x y)
    hx : Not (Dvd.dvd (↑p) x)
    ⊢ Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow x p) (HPow.hPow y p))) (HAdd.hA …
  -/
  rw [← geom_sum₂_mul, emultiplicity_mul hp, emultiplicity_geom_sum₂_eq_one hp hp1 hxy hx, add_comm]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.pow_prime_sub_pow_prime := emultiplicity_pow_prime_sub_pow_prime


theorem emultiplicity_pow_prime_pow_sub_pow_prime_pow (a : ℕ) :
    emultiplicity (↑p) (x ^ p ^ a - y ^ p ^ a) = emultiplicity (↑p) (x - y) + a := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    x y : R
    p : Nat
    inst✝ : IsDomain R
    hp : Prime ↑p
    hp1 : Odd p
    hxy : Dvd.dvd (↑p) (HSub.hSub x y)
    hx : Not (Dvd.dvd (↑p) x)
    a : Nat
    ⊢ Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow x (HPow.hPow p a)) (HPow.hPow y …
  -/
  induction' a with a h_ind
    /-
      case zero
      R : Type u_1
      inst✝¹ : CommRing R
      x y : R
      p : Nat
      inst✝ : IsDomain R
      hp : Prime ↑p
      hp1 : Odd p
      hxy : Dvd.dvd (↑p) (HSub.hSub x y)
      hx : Not (Dvd.dvd (↑p) x)
      ⊢ Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow x (HPow.hPow p 0)) (HPow.hPow y …
    -/
  · rw [Nat.cast_zero, add_zero, pow_zero, pow_one, pow_one]
    /-
      🎉 no goals
    -/
  /-
    case succ
    R : Type u_1
    inst✝¹ : CommRing R
    x y : R
    p : Nat
    inst✝ : IsDomain R
    hp : Prime ↑p
    hp1 : Odd p
    hxy : Dvd.dvd (↑p) (HSub.hSub x y)
    hx : Not (Dvd.dvd (↑p) x)
    a : Nat
    h_ind : Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow x (HPow.hPow p a)) (HPow. …
    ⊢ Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow x (HPow.hPow p (HAdd.hAdd a 1)) …
  -/
  rw [Nat.cast_add, Nat.cast_one, ← add_assoc, ← h_ind, pow_succ, pow_mul, pow_mul]
  /-
    case succ
    R : Type u_1
    inst✝¹ : CommRing R
    x y : R
    p : Nat
    inst✝ : IsDomain R
    hp : Prime ↑p
    hp1 : Odd p
    hxy : Dvd.dvd (↑p) (HSub.hSub x y)
    hx : Not (Dvd.dvd (↑p) x)
    a : Nat
    h_ind : Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow x (HPow.hPow p a)) (HPow. …
    ⊢ Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow (HPow.hPow x (HPow.hPow p a)) p …
  -/
  apply emultiplicity_pow_prime_sub_pow_prime hp hp1
    /-
      case succ.hxy
      R : Type u_1
      inst✝¹ : CommRing R
      x y : R
      p : Nat
      inst✝ : IsDomain R
      hp : Prime ↑p
      hp1 : Odd p
      hxy : Dvd.dvd (↑p) (HSub.hSub x y)
      hx : Not (Dvd.dvd (↑p) x)
      a : Nat
      h_ind : Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow x (HPow.hPow p a)) (HPow. …
      ⊢ Dvd.dvd (↑p) (HSub.hSub (HPow.hPow x (HPow.hPow p a)) (HPow.hPow y (HPow.hPo …
    -/
  · rw [← geom_sum₂_mul]
    /-
      case succ.hxy
      R : Type u_1
      inst✝¹ : CommRing R
      x y : R
      p : Nat
      inst✝ : IsDomain R
      hp : Prime ↑p
      hp1 : Odd p
      hxy : Dvd.dvd (↑p) (HSub.hSub x y)
      hx : Not (Dvd.dvd (↑p) x)
      a : Nat
      h_ind : Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow x (HPow.hPow p a)) (HPow. …
      ⊢ Dvd.dvd (↑p) (HMul.hMul ((Finset.range (HPow.hPow p a)).sum fun i => HMul.hM …
    -/
    exact dvd_mul_of_dvd_right hxy _
    /-
      🎉 no goals
    -/
    /-
      case succ.hx
      R : Type u_1
      inst✝¹ : CommRing R
      x y : R
      p : Nat
      inst✝ : IsDomain R
      hp : Prime ↑p
      hp1 : Odd p
      hxy : Dvd.dvd (↑p) (HSub.hSub x y)
      hx : Not (Dvd.dvd (↑p) x)
      a : Nat
      h_ind : Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow x (HPow.hPow p a)) (HPow. …
      ⊢ Not (Dvd.dvd (↑p) (HPow.hPow x (HPow.hPow p a)))
    -/
  · exact fun h => hx (hp.dvd_of_dvd_pow h)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.pow_prime_pow_sub_pow_prime_pow := emultiplicity_pow_prime_pow_sub_pow_prime_pow


/-- **Lifting the exponent lemma** for odd primes. -/
theorem Int.emultiplicity_pow_sub_pow {x y : ℤ} (hxy : ↑p ∣ x - y) (hx : ¬↑p ∣ x) (n : ℕ) :
    emultiplicity (↑p) (x ^ n - y ^ n) = emultiplicity (↑p) (x - y) + emultiplicity p n := by
  /-
    p : Nat
    hp : Nat.Prime p
    hp1 : Odd p
    x y : Int
    hxy : Dvd.dvd (↑p) (HSub.hSub x y)
    hx : Not (Dvd.dvd (↑p) x)
    n : Nat
    ⊢ Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))) (HAdd.hA …
  -/
  cases' n with n
    /-
      case zero
      p : Nat
      hp : Nat.Prime p
      hp1 : Odd p
      x y : Int
      hxy : Dvd.dvd (↑p) (HSub.hSub x y)
      hx : Not (Dvd.dvd (↑p) x)
      ⊢ Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow x 0) (HPow.hPow y 0))) (HAdd.hA …
    -/
  · simp only [emultiplicity_zero, add_top, pow_zero, sub_self]
    /-
      🎉 no goals
    -/
  /-
    case succ
    p : Nat
    hp : Nat.Prime p
    hp1 : Odd p
    x y : Int
    hxy : Dvd.dvd (↑p) (HSub.hSub x y)
    hx : Not (Dvd.dvd (↑p) x)
    n : Nat
    ⊢ Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow x (HAdd.hAdd n 1)) (HPow.hPow y …
  -/
  have h : FiniteMultiplicity _ _ := Nat.finiteMultiplicity_iff.mpr ⟨hp.ne_one, n.succ_pos⟩
  /-
    case succ
    p : Nat
    hp : Nat.Prime p
    hp1 : Odd p
    x y : Int
    hxy : Dvd.dvd (↑p) (HSub.hSub x y)
    hx : Not (Dvd.dvd (↑p) x)
    n : Nat
    h : FiniteMultiplicity p n.succ
    ⊢ Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow x (HAdd.hAdd n 1)) (HPow.hPow y …
  -/
  simp only [Nat.succ_eq_add_one] at h
  /-
    case succ
    p : Nat
    hp : Nat.Prime p
    hp1 : Odd p
    x y : Int
    hxy : Dvd.dvd (↑p) (HSub.hSub x y)
    hx : Not (Dvd.dvd (↑p) x)
    n : Nat
    h : FiniteMultiplicity p (HAdd.hAdd n 1)
    ⊢ Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow x (HAdd.hAdd n 1)) (HPow.hPow y …
  -/
  rcases emultiplicity_eq_coe.mp h.emultiplicity_eq_multiplicity with ⟨⟨k, hk⟩, hpn⟩
  /-
    case succ.intro.intro
    p : Nat
    hp : Nat.Prime p
    hp1 : Odd p
    x y : Int
    hxy : Dvd.dvd (↑p) (HSub.hSub x y)
    hx : Not (Dvd.dvd (↑p) x)
    n : Nat
    h : FiniteMultiplicity p (HAdd.hAdd n 1)
    hpn : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (multiplicity p (HAdd.hAdd n 1)) 1) …
    k : Nat
    hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow p (multiplicity p (HAdd.hAdd n 1 …
    ⊢ Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow x (HAdd.hAdd n 1)) (HPow.hPow y …
  -/
  conv_lhs => rw [hk, pow_mul, pow_mul]
  /-
    case succ.intro.intro
    p : Nat
    hp : Nat.Prime p
    hp1 : Odd p
    x y : Int
    hxy : Dvd.dvd (↑p) (HSub.hSub x y)
    hx : Not (Dvd.dvd (↑p) x)
    n : Nat
    h : FiniteMultiplicity p (HAdd.hAdd n 1)
    hpn : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (multiplicity p (HAdd.hAdd n 1)) 1) …
    k : Nat
    hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow p (multiplicity p (HAdd.hAdd n 1 …
    ⊢ Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow (HPow.hPow x (HPow.hPow p (mult …
  -/
  rw [Nat.prime_iff_prime_int] at hp
  rw [emultiplicity_pow_sub_pow_of_prime hp,
    emultiplicity_pow_prime_pow_sub_pow_prime_pow hp hp1 hxy hx, h.emultiplicity_eq_multiplicity]
    /-
      case succ.intro.intro.hxy
      p : Nat
      hp : Prime ↑p
      hp1 : Odd p
      x y : Int
      hxy : Dvd.dvd (↑p) (HSub.hSub x y)
      hx : Not (Dvd.dvd (↑p) x)
      n : Nat
      h : FiniteMultiplicity p (HAdd.hAdd n 1)
      hpn : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (multiplicity p (HAdd.hAdd n 1)) 1) …
      k : Nat
      hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow p (multiplicity p (HAdd.hAdd n 1 …
      ⊢ Dvd.dvd (↑p) (HSub.hSub (HPow.hPow x (HPow.hPow p (multiplicity p (HAdd.hAdd …
    -/
  · rw [← geom_sum₂_mul]
    /-
      case succ.intro.intro.hxy
      p : Nat
      hp : Prime ↑p
      hp1 : Odd p
      x y : Int
      hxy : Dvd.dvd (↑p) (HSub.hSub x y)
      hx : Not (Dvd.dvd (↑p) x)
      n : Nat
      h : FiniteMultiplicity p (HAdd.hAdd n 1)
      hpn : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (multiplicity p (HAdd.hAdd n 1)) 1) …
      k : Nat
      hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow p (multiplicity p (HAdd.hAdd n 1 …
      ⊢ Dvd.dvd (↑p) (HMul.hMul ((Finset.range (HPow.hPow p (multiplicity p (HAdd.hA …
    -/
    exact dvd_mul_of_dvd_right hxy _
    /-
      🎉 no goals
    -/
    /-
      case succ.intro.intro.hx
      p : Nat
      hp : Prime ↑p
      hp1 : Odd p
      x y : Int
      hxy : Dvd.dvd (↑p) (HSub.hSub x y)
      hx : Not (Dvd.dvd (↑p) x)
      n : Nat
      h : FiniteMultiplicity p (HAdd.hAdd n 1)
      hpn : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (multiplicity p (HAdd.hAdd n 1)) 1) …
      k : Nat
      hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow p (multiplicity p (HAdd.hAdd n 1 …
      ⊢ Not (Dvd.dvd (↑p) (HPow.hPow x (HPow.hPow p (multiplicity p (HAdd.hAdd n 1)) …
    -/
  · exact fun h => hx (hp.dvd_of_dvd_pow h)
    /-
      🎉 no goals
    -/
    /-
      case succ.intro.intro.hn
      p : Nat
      hp : Prime ↑p
      hp1 : Odd p
      x y : Int
      hxy : Dvd.dvd (↑p) (HSub.hSub x y)
      hx : Not (Dvd.dvd (↑p) x)
      n : Nat
      h : FiniteMultiplicity p (HAdd.hAdd n 1)
      hpn : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (multiplicity p (HAdd.hAdd n 1)) 1) …
      k : Nat
      hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow p (multiplicity p (HAdd.hAdd n 1 …
      ⊢ Not (Dvd.dvd ↑p ↑k)
    -/
  · rw [Int.natCast_dvd_natCast]
    /-
      case succ.intro.intro.hn
      p : Nat
      hp : Prime ↑p
      hp1 : Odd p
      x y : Int
      hxy : Dvd.dvd (↑p) (HSub.hSub x y)
      hx : Not (Dvd.dvd (↑p) x)
      n : Nat
      h : FiniteMultiplicity p (HAdd.hAdd n 1)
      hpn : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (multiplicity p (HAdd.hAdd n 1)) 1) …
      k : Nat
      hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow p (multiplicity p (HAdd.hAdd n 1 …
      ⊢ Not (Dvd.dvd p k)
    -/
    rintro ⟨c, rfl⟩
    /-
      case succ.intro.intro.hn.intro
      p : Nat
      hp : Prime ↑p
      hp1 : Odd p
      x y : Int
      hxy : Dvd.dvd (↑p) (HSub.hSub x y)
      hx : Not (Dvd.dvd (↑p) x)
      n : Nat
      h : FiniteMultiplicity p (HAdd.hAdd n 1)
      hpn : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (multiplicity p (HAdd.hAdd n 1)) 1) …
      c : Nat
      hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow p (multiplicity p (HAdd.hAdd n 1 …
      ⊢ False
    -/
    refine hpn ⟨c, ?_⟩
    /-
      case succ.intro.intro.hn.intro
      p : Nat
      hp : Prime ↑p
      hp1 : Odd p
      x y : Int
      hxy : Dvd.dvd (↑p) (HSub.hSub x y)
      hx : Not (Dvd.dvd (↑p) x)
      n : Nat
      h : FiniteMultiplicity p (HAdd.hAdd n 1)
      hpn : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (multiplicity p (HAdd.hAdd n 1)) 1) …
      c : Nat
      hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow p (multiplicity p (HAdd.hAdd n 1 …
      ⊢ Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow p (HAdd.hAdd (multiplicity p (HAdd. …
    -/
    rwa [pow_succ, mul_assoc]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Int.pow_sub_pow := Int.emultiplicity_pow_sub_pow


theorem Int.emultiplicity_pow_add_pow {x y : ℤ} (hxy : ↑p ∣ x + y) (hx : ¬↑p ∣ x)
    {n : ℕ} (hn : Odd n) :
    emultiplicity (↑p) (x ^ n + y ^ n) = emultiplicity (↑p) (x + y) + emultiplicity p n := by
  /-
    p : Nat
    hp : Nat.Prime p
    hp1 : Odd p
    x y : Int
    hxy : Dvd.dvd (↑p) (HAdd.hAdd x y)
    hx : Not (Dvd.dvd (↑p) x)
    n : Nat
    hn : Odd n
    ⊢ Eq (emultiplicity (↑p) (HAdd.hAdd (HPow.hPow x n) (HPow.hPow y n))) (HAdd.hA …
  -/
  rw [← sub_neg_eq_add] at hxy
  /-
    p : Nat
    hp : Nat.Prime p
    hp1 : Odd p
    x y : Int
    hxy : Dvd.dvd (↑p) (HSub.hSub x (Neg.neg y))
    hx : Not (Dvd.dvd (↑p) x)
    n : Nat
    hn : Odd n
    ⊢ Eq (emultiplicity (↑p) (HAdd.hAdd (HPow.hPow x n) (HPow.hPow y n))) (HAdd.hA …
  -/
  rw [← sub_neg_eq_add, ← sub_neg_eq_add, ← Odd.neg_pow hn]
  /-
    p : Nat
    hp : Nat.Prime p
    hp1 : Odd p
    x y : Int
    hxy : Dvd.dvd (↑p) (HSub.hSub x (Neg.neg y))
    hx : Not (Dvd.dvd (↑p) x)
    n : Nat
    hn : Odd n
    ⊢ Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow x n) (HPow.hPow (Neg.neg y) n)) …
  -/
  exact Int.emultiplicity_pow_sub_pow hp hp1 hxy hx n
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Int.pow_add_pow := Int.emultiplicity_pow_add_pow


theorem Nat.emultiplicity_pow_sub_pow {x y : ℕ} (hxy : p ∣ x - y) (hx : ¬p ∣ x) (n : ℕ) :
    emultiplicity p (x ^ n - y ^ n) = emultiplicity p (x - y) + emultiplicity p n := by
  /-
    p : Nat
    hp : Nat.Prime p
    hp1 : Odd p
    x y : Nat
    hxy : Dvd.dvd p (HSub.hSub x y)
    hx : Not (Dvd.dvd p x)
    n : Nat
    ⊢ Eq (emultiplicity p (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))) (HAdd.hAdd  …
  -/
  obtain hyx | hyx := le_total y x
    /-
      case inl
      p : Nat
      hp : Nat.Prime p
      hp1 : Odd p
      x y : Nat
      hxy : Dvd.dvd p (HSub.hSub x y)
      hx : Not (Dvd.dvd p x)
      n : Nat
      hyx : LE.le y x
      ⊢ Eq (emultiplicity p (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))) (HAdd.hAdd  …
    -/
  · iterate 2 rw [← Int.natCast_emultiplicity]
    /-
      case inl
      p : Nat
      hp : Nat.Prime p
      hp1 : Odd p
      x y : Nat
      hxy : Dvd.dvd p (HSub.hSub x y)
      hx : Not (Dvd.dvd p x)
      n : Nat
      hyx : LE.le y x
      ⊢ Eq (emultiplicity ↑p ↑(HSub.hSub (HPow.hPow x n) (HPow.hPow y n))) (HAdd.hAd …
    -/
    rw [Int.ofNat_sub (Nat.pow_le_pow_left hyx n)]
    /-
      case inl
      p : Nat
      hp : Nat.Prime p
      hp1 : Odd p
      x y : Nat
      hxy : Dvd.dvd p (HSub.hSub x y)
      hx : Not (Dvd.dvd p x)
      n : Nat
      hyx : LE.le y x
      ⊢ Eq (emultiplicity (↑p) (HSub.hSub ↑(HPow.hPow x n) ↑(HPow.hPow y n))) (HAdd. …
    -/
    rw [← Int.natCast_dvd_natCast] at hxy hx
    /-
      case inl
      p : Nat
      hp : Nat.Prime p
      hp1 : Odd p
      x y : Nat
      hxy : Dvd.dvd ↑p ↑(HSub.hSub x y)
      hx : Not (Dvd.dvd ↑p ↑x)
      n : Nat
      hyx : LE.le y x
      ⊢ Eq (emultiplicity (↑p) (HSub.hSub ↑(HPow.hPow x n) ↑(HPow.hPow y n))) (HAdd. …
    -/
    rw [Int.natCast_sub hyx] at *
    /-
      case inl
      p : Nat
      hp : Nat.Prime p
      hp1 : Odd p
      x y : Nat
      hxy : Dvd.dvd (↑p) (HSub.hSub ↑x ↑y)
      hx : Not (Dvd.dvd ↑p ↑x)
      n : Nat
      hyx : LE.le y x
      ⊢ Eq (emultiplicity (↑p) (HSub.hSub ↑(HPow.hPow x n) ↑(HPow.hPow y n))) (HAdd. …
    -/
    push_cast at *
    /-
      case inl
      p : Nat
      hp : Nat.Prime p
      hp1 : Odd p
      x y : Nat
      hxy : Dvd.dvd (↑p) (HSub.hSub ↑x ↑y)
      hx : Not (Dvd.dvd ↑p ↑x)
      n : Nat
      hyx : LE.le y x
      ⊢ Eq (emultiplicity (↑p) (HSub.hSub (HPow.hPow (↑x) n) (HPow.hPow (↑y) n))) (H …
    -/
    exact Int.emultiplicity_pow_sub_pow hp hp1 hxy hx n
    /-
      🎉 no goals
    -/
  · simp only [Nat.sub_eq_zero_iff_le.mpr (Nat.pow_le_pow_left hyx n), emultiplicity_zero,
    Nat.sub_eq_zero_iff_le.mpr hyx, top_add]


@[deprecated (since := "2024-11-30")]
alias multiplicity.Nat.pow_sub_pow := Nat.emultiplicity_pow_sub_pow


theorem Nat.emultiplicity_pow_add_pow {x y : ℕ} (hxy : p ∣ x + y) (hx : ¬p ∣ x)
    {n : ℕ} (hn : Odd n) :
    emultiplicity p (x ^ n + y ^ n) = emultiplicity p (x + y) + emultiplicity p n := by
  /-
    p : Nat
    hp : Nat.Prime p
    hp1 : Odd p
    x y : Nat
    hxy : Dvd.dvd p (HAdd.hAdd x y)
    hx : Not (Dvd.dvd p x)
    n : Nat
    hn : Odd n
    ⊢ Eq (emultiplicity p (HAdd.hAdd (HPow.hPow x n) (HPow.hPow y n))) (HAdd.hAdd  …
  -/
  iterate 2 rw [← Int.natCast_emultiplicity]
  /-
    p : Nat
    hp : Nat.Prime p
    hp1 : Odd p
    x y : Nat
    hxy : Dvd.dvd p (HAdd.hAdd x y)
    hx : Not (Dvd.dvd p x)
    n : Nat
    hn : Odd n
    ⊢ Eq (emultiplicity ↑p ↑(HAdd.hAdd (HPow.hPow x n) (HPow.hPow y n))) (HAdd.hAd …
  -/
  rw [← Int.natCast_dvd_natCast] at hxy hx
  /-
    p : Nat
    hp : Nat.Prime p
    hp1 : Odd p
    x y : Nat
    hxy : Dvd.dvd ↑p ↑(HAdd.hAdd x y)
    hx : Not (Dvd.dvd ↑p ↑x)
    n : Nat
    hn : Odd n
    ⊢ Eq (emultiplicity ↑p ↑(HAdd.hAdd (HPow.hPow x n) (HPow.hPow y n))) (HAdd.hAd …
  -/
  push_cast at *
  /-
    p : Nat
    hp : Nat.Prime p
    hp1 : Odd p
    x y : Nat
    hx : Not (Dvd.dvd ↑p ↑x)
    n : Nat
    hn : Odd n
    hxy : Dvd.dvd (↑p) (HAdd.hAdd ↑x ↑y)
    ⊢ Eq (emultiplicity (↑p) (HAdd.hAdd (HPow.hPow (↑x) n) (HPow.hPow (↑y) n))) (H …
  -/
  exact Int.emultiplicity_pow_add_pow hp hp1 hxy hx hn
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.Nat.pow_add_pow := Nat.emultiplicity_pow_add_pow


theorem pow_two_pow_sub_pow_two_pow [CommRing R] {x y : R} (n : ℕ) :
    x ^ 2 ^ n - y ^ 2 ^ n = (∏ i ∈ Finset.range n, (x ^ 2 ^ i + y ^ 2 ^ i)) * (x - y) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    x y : R
    n : Nat
    ⊢ Eq (HSub.hSub (HPow.hPow x (HPow.hPow 2 n)) (HPow.hPow y (HPow.hPow 2 n))) ( …
  -/
  induction' n with d hd
    /-
      case zero
      R : Type u_1
      inst✝ : CommRing R
      x y : R
      ⊢ Eq (HSub.hSub (HPow.hPow x (HPow.hPow 2 0)) (HPow.hPow y (HPow.hPow 2 0))) ( …
    -/
  · simp only [pow_zero, pow_one, range_zero, prod_empty, one_mul]
    /-
      🎉 no goals
    -/
  · suffices x ^ 2 ^ d.succ - y ^ 2 ^ d.succ = (x ^ 2 ^ d + y ^ 2 ^ d) * (x ^ 2 ^ d - y ^ 2 ^ d) by
      rw [this, hd, Finset.prod_range_succ, ← mul_assoc, mul_comm (x ^ 2 ^ d + y ^ 2 ^ d)]
    /-
      case succ
      R : Type u_1
      inst✝ : CommRing R
      x y : R
      d : Nat
      hd : Eq (HSub.hSub (HPow.hPow x (HPow.hPow 2 d)) (HPow.hPow y (HPow.hPow 2 d)) …
      ⊢ Eq (HSub.hSub (HPow.hPow x (HPow.hPow 2 d.succ)) (HPow.hPow y (HPow.hPow 2 d …
    -/
    rw [Nat.succ_eq_add_one]
    /-
      case succ
      R : Type u_1
      inst✝ : CommRing R
      x y : R
      d : Nat
      hd : Eq (HSub.hSub (HPow.hPow x (HPow.hPow 2 d)) (HPow.hPow y (HPow.hPow 2 d)) …
      ⊢ Eq (HSub.hSub (HPow.hPow x (HPow.hPow 2 (HAdd.hAdd d 1))) (HPow.hPow y (HPow …
    -/
    ring
    /-
      🎉 no goals
    -/

-- Porting note: simplified proof because `fin_cases` was not available in that case

theorem Int.sq_mod_four_eq_one_of_odd {x : ℤ} : Odd x → x ^ 2 % 4 = 1 := by
  /-
    x : Int
    ⊢ Odd x → Eq (HMod.hMod (HPow.hPow x 2) 4) 1
  -/
  intro hx
  /-
    x : Int
    hx : Odd x
    ⊢ Eq (HMod.hMod (HPow.hPow x 2) 4) 1
  -/
  unfold Odd at hx
  /-
    x : Int
    hx : Exists fun k => Eq x (HAdd.hAdd (HMul.hMul 2 k) 1)
    ⊢ Eq (HMod.hMod (HPow.hPow x 2) 4) 1
  -/
  rcases hx with ⟨_, rfl⟩
  /-
    case intro
    w✝ : Int
    ⊢ Eq (HMod.hMod (HPow.hPow (HAdd.hAdd (HMul.hMul 2 w✝) 1) 2) 4) 1
  -/
  ring_nf
  /-
    case intro
    w✝ : Int
    ⊢ Eq (HMod.hMod (HAdd.hAdd (HAdd.hAdd 1 (HMul.hMul w✝ 4)) (HMul.hMul (HPow.hPo …
  -/
  rw [add_assoc, ← add_mul, Int.add_mul_emod_self]
  /-
    case intro
    w✝ : Int
    ⊢ Eq (HMod.hMod 1 4) 1
  -/
  decide
  /-
    🎉 no goals
  -/


theorem Int.two_pow_two_pow_add_two_pow_two_pow {x y : ℤ} (hx : ¬2 ∣ x) (hxy : 4 ∣ x - y) (i : ℕ) :
    emultiplicity 2 (x ^ 2 ^ i + y ^ 2 ^ i) = ↑(1 : ℕ) := by
  /-
    x y : Int
    hx : Not (Dvd.dvd 2 x)
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    i : Nat
    ⊢ Eq (emultiplicity 2 (HAdd.hAdd (HPow.hPow x (HPow.hPow 2 i)) (HPow.hPow y (H …
  -/
  have hx_odd : Odd x := by rwa [← Int.not_even_iff_odd, even_iff_two_dvd]
  /-
    x y : Int
    hx : Not (Dvd.dvd 2 x)
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    i : Nat
    hx_odd : Odd x
    ⊢ Eq (emultiplicity 2 (HAdd.hAdd (HPow.hPow x (HPow.hPow 2 i)) (HPow.hPow y (H …
  -/
  have hxy_even : Even (x - y) := even_iff_two_dvd.mpr (dvd_trans (by decide) hxy)
  /-
    x y : Int
    hx : Not (Dvd.dvd 2 x)
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    i : Nat
    hx_odd : Odd x
    hxy_even : Even (HSub.hSub x y)
    ⊢ Eq (emultiplicity 2 (HAdd.hAdd (HPow.hPow x (HPow.hPow 2 i)) (HPow.hPow y (H …
  -/
  have hy_odd : Odd y := by simpa using hx_odd.sub_even hxy_even
  /-
    x y : Int
    hx : Not (Dvd.dvd 2 x)
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    i : Nat
    hx_odd : Odd x
    hxy_even : Even (HSub.hSub x y)
    hy_odd : Odd y
    ⊢ Eq (emultiplicity 2 (HAdd.hAdd (HPow.hPow x (HPow.hPow 2 i)) (HPow.hPow y (H …
  -/
  refine emultiplicity_eq_coe.mpr ⟨?_, ?_⟩
    /-
      case refine_1
      x y : Int
      hx : Not (Dvd.dvd 2 x)
      hxy : Dvd.dvd 4 (HSub.hSub x y)
      i : Nat
      hx_odd : Odd x
      hxy_even : Even (HSub.hSub x y)
      hy_odd : Odd y
      ⊢ Dvd.dvd (HPow.hPow 2 1) (HAdd.hAdd (HPow.hPow x (HPow.hPow 2 i)) (HPow.hPow  …
    -/
  · rw [pow_one, ← even_iff_two_dvd]
    /-
      case refine_1
      x y : Int
      hx : Not (Dvd.dvd 2 x)
      hxy : Dvd.dvd 4 (HSub.hSub x y)
      i : Nat
      hx_odd : Odd x
      hxy_even : Even (HSub.hSub x y)
      hy_odd : Odd y
      ⊢ Even (HAdd.hAdd (HPow.hPow x (HPow.hPow 2 i)) (HPow.hPow y (HPow.hPow 2 i)))
    -/
    exact hx_odd.pow.add_odd hy_odd.pow
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    x y : Int
    hx : Not (Dvd.dvd 2 x)
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    i : Nat
    hx_odd : Odd x
    hxy_even : Even (HSub.hSub x y)
    hy_odd : Odd y
    ⊢ Not (Dvd.dvd (HPow.hPow 2 (HAdd.hAdd 1 1)) (HAdd.hAdd (HPow.hPow x (HPow.hPo …
  -/
  cases' i with i
    /-
      case refine_2.zero
      x y : Int
      hx : Not (Dvd.dvd 2 x)
      hxy : Dvd.dvd 4 (HSub.hSub x y)
      hx_odd : Odd x
      hxy_even : Even (HSub.hSub x y)
      hy_odd : Odd y
      ⊢ Not (Dvd.dvd (HPow.hPow 2 (HAdd.hAdd 1 1)) (HAdd.hAdd (HPow.hPow x (HPow.hPo …
    -/
  · intro hxy'
    have : 2 * 2 ∣ 2 * x := by
      have := dvd_add hxy hxy'
      norm_num at *
      rw [two_mul]
      exact this
    /-
      case refine_2.zero
      x y : Int
      hx : Not (Dvd.dvd 2 x)
      hxy : Dvd.dvd 4 (HSub.hSub x y)
      hx_odd : Odd x
      hxy_even : Even (HSub.hSub x y)
      hy_odd : Odd y
      hxy' : Dvd.dvd (HPow.hPow 2 (HAdd.hAdd 1 1)) (HAdd.hAdd (HPow.hPow x (HPow.hPo …
      this : Dvd.dvd (HMul.hMul 2 2) (HMul.hMul 2 x)
      ⊢ False
    -/
    have : 2 ∣ x := (mul_dvd_mul_iff_left (by norm_num)).mp this
    /-
      case refine_2.zero
      x y : Int
      hx : Not (Dvd.dvd 2 x)
      hxy : Dvd.dvd 4 (HSub.hSub x y)
      hx_odd : Odd x
      hxy_even : Even (HSub.hSub x y)
      hy_odd : Odd y
      hxy' : Dvd.dvd (HPow.hPow 2 (HAdd.hAdd 1 1)) (HAdd.hAdd (HPow.hPow x (HPow.hPo …
      this✝ : Dvd.dvd (HMul.hMul 2 2) (HMul.hMul 2 x)
      this : Dvd.dvd 2 x
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/
  suffices ∀ x : ℤ, Odd x → x ^ 2 ^ (i + 1) % 4 = 1 by
    rw [show (2 ^ (1 + 1) : ℤ) = 4 by norm_num, Int.dvd_iff_emod_eq_zero, Int.add_emod,
      this _ hx_odd, this _ hy_odd]
    decide
  /-
    case refine_2.succ
    x y : Int
    hx : Not (Dvd.dvd 2 x)
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    hx_odd : Odd x
    hxy_even : Even (HSub.hSub x y)
    hy_odd : Odd y
    i : Nat
    ⊢ ∀ (x : Int), Odd x → Eq (HMod.hMod (HPow.hPow x (HPow.hPow 2 (HAdd.hAdd i 1) …
  -/
  intro x hx
  /-
    case refine_2.succ
    x✝ y : Int
    hx✝ : Not (Dvd.dvd 2 x✝)
    hxy : Dvd.dvd 4 (HSub.hSub x✝ y)
    hx_odd : Odd x✝
    hxy_even : Even (HSub.hSub x✝ y)
    hy_odd : Odd y
    i : Nat
    x : Int
    hx : Odd x
    ⊢ Eq (HMod.hMod (HPow.hPow x (HPow.hPow 2 (HAdd.hAdd i 1))) 4) 1
  -/
  rw [pow_succ', mul_comm, pow_mul, Int.sq_mod_four_eq_one_of_odd hx.pow]
  /-
    🎉 no goals
  -/


theorem Int.two_pow_two_pow_sub_pow_two_pow {x y : ℤ} (n : ℕ) (hxy : 4 ∣ x - y) (hx : ¬2 ∣ x) :
    emultiplicity 2 (x ^ 2 ^ n - y ^ 2 ^ n) = emultiplicity 2 (x - y) + n := by
  simp only [pow_two_pow_sub_pow_two_pow n, emultiplicity_mul Int.prime_two,
    Finset.emultiplicity_prod Int.prime_two, add_comm, Nat.cast_one, Finset.sum_const,
    Finset.card_range, nsmul_one, Int.two_pow_two_pow_add_two_pow_two_pow hx hxy]


theorem Int.two_pow_sub_pow' {x y : ℤ} (n : ℕ) (hxy : 4 ∣ x - y) (hx : ¬2 ∣ x) :
    emultiplicity 2 (x ^ n - y ^ n) = emultiplicity 2 (x - y) + emultiplicity (2 : ℤ) n := by
  /-
    x y : Int
    n : Nat
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    ⊢ Eq (emultiplicity 2 (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))) (HAdd.hAdd  …
  -/
  have hx_odd : Odd x := by rwa [← Int.not_even_iff_odd, even_iff_two_dvd]
  /-
    x y : Int
    n : Nat
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    hx_odd : Odd x
    ⊢ Eq (emultiplicity 2 (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))) (HAdd.hAdd  …
  -/
  have hxy_even : Even (x - y) := even_iff_two_dvd.mpr (dvd_trans (by decide) hxy)
  /-
    x y : Int
    n : Nat
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    hx_odd : Odd x
    hxy_even : Even (HSub.hSub x y)
    ⊢ Eq (emultiplicity 2 (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))) (HAdd.hAdd  …
  -/
  have hy_odd : Odd y := by simpa using hx_odd.sub_even hxy_even
  /-
    x y : Int
    n : Nat
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    hx_odd : Odd x
    hxy_even : Even (HSub.hSub x y)
    hy_odd : Odd y
    ⊢ Eq (emultiplicity 2 (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))) (HAdd.hAdd  …
  -/
  cases' n with n
    /-
      case zero
      x y : Int
      hxy : Dvd.dvd 4 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      hx_odd : Odd x
      hxy_even : Even (HSub.hSub x y)
      hy_odd : Odd y
      ⊢ Eq (emultiplicity 2 (HSub.hSub (HPow.hPow x 0) (HPow.hPow y 0))) (HAdd.hAdd  …
    -/
  · simp only [pow_zero, sub_self, emultiplicity_zero, Int.ofNat_zero, add_top]
    /-
      🎉 no goals
    -/
  /-
    case succ
    x y : Int
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    hx_odd : Odd x
    hxy_even : Even (HSub.hSub x y)
    hy_odd : Odd y
    n : Nat
    ⊢ Eq (emultiplicity 2 (HSub.hSub (HPow.hPow x (HAdd.hAdd n 1)) (HPow.hPow y (H …
  -/
  have h : FiniteMultiplicity 2 n.succ := Nat.finiteMultiplicity_iff.mpr ⟨by norm_num, n.succ_pos⟩
  /-
    case succ
    x y : Int
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    hx_odd : Odd x
    hxy_even : Even (HSub.hSub x y)
    hy_odd : Odd y
    n : Nat
    h : FiniteMultiplicity 2 n.succ
    ⊢ Eq (emultiplicity 2 (HSub.hSub (HPow.hPow x (HAdd.hAdd n 1)) (HPow.hPow y (H …
  -/
  simp only [Nat.succ_eq_add_one] at h
  /-
    case succ
    x y : Int
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    hx_odd : Odd x
    hxy_even : Even (HSub.hSub x y)
    hy_odd : Odd y
    n : Nat
    h : FiniteMultiplicity 2 (HAdd.hAdd n 1)
    ⊢ Eq (emultiplicity 2 (HSub.hSub (HPow.hPow x (HAdd.hAdd n 1)) (HPow.hPow y (H …
  -/
  rcases emultiplicity_eq_coe.mp h.emultiplicity_eq_multiplicity with ⟨⟨k, hk⟩, hpn⟩
  rw [hk, pow_mul, pow_mul, emultiplicity_pow_sub_pow_of_prime,
    Int.two_pow_two_pow_sub_pow_two_pow _ hxy hx, ← hk]
    /-
      case succ.intro.intro
      x y : Int
      hxy : Dvd.dvd 4 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      hx_odd : Odd x
      hxy_even : Even (HSub.hSub x y)
      hy_odd : Odd y
      n : Nat
      h : FiniteMultiplicity 2 (HAdd.hAdd n 1)
      hpn : Not (Dvd.dvd (HPow.hPow 2 (HAdd.hAdd (multiplicity 2 (HAdd.hAdd n 1)) 1) …
      k : Nat
      hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow 2 (multiplicity 2 (HAdd.hAdd n 1 …
      ⊢ Eq (HAdd.hAdd (emultiplicity 2 (HSub.hSub x y)) ↑(multiplicity 2 (HAdd.hAdd  …
    -/
  · norm_cast
    /-
      case succ.intro.intro
      x y : Int
      hxy : Dvd.dvd 4 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      hx_odd : Odd x
      hxy_even : Even (HSub.hSub x y)
      hy_odd : Odd y
      n : Nat
      h : FiniteMultiplicity 2 (HAdd.hAdd n 1)
      hpn : Not (Dvd.dvd (HPow.hPow 2 (HAdd.hAdd (multiplicity 2 (HAdd.hAdd n 1)) 1) …
      k : Nat
      hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow 2 (multiplicity 2 (HAdd.hAdd n 1 …
      ⊢ Eq (HAdd.hAdd (emultiplicity 2 (HSub.hSub x y)) ↑(multiplicity 2 (HAdd.hAdd  …
    -/
    rw [h.emultiplicity_eq_multiplicity]
    /-
      🎉 no goals
    -/
    /-
      case succ.intro.intro.hp
      x y : Int
      hxy : Dvd.dvd 4 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      hx_odd : Odd x
      hxy_even : Even (HSub.hSub x y)
      hy_odd : Odd y
      n : Nat
      h : FiniteMultiplicity 2 (HAdd.hAdd n 1)
      hpn : Not (Dvd.dvd (HPow.hPow 2 (HAdd.hAdd (multiplicity 2 (HAdd.hAdd n 1)) 1) …
      k : Nat
      hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow 2 (multiplicity 2 (HAdd.hAdd n 1 …
      ⊢ Prime 2
    -/
  · exact Int.prime_two
    /-
      🎉 no goals
    -/
    /-
      case succ.intro.intro.hxy
      x y : Int
      hxy : Dvd.dvd 4 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      hx_odd : Odd x
      hxy_even : Even (HSub.hSub x y)
      hy_odd : Odd y
      n : Nat
      h : FiniteMultiplicity 2 (HAdd.hAdd n 1)
      hpn : Not (Dvd.dvd (HPow.hPow 2 (HAdd.hAdd (multiplicity 2 (HAdd.hAdd n 1)) 1) …
      k : Nat
      hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow 2 (multiplicity 2 (HAdd.hAdd n 1 …
      ⊢ Dvd.dvd 2 (HSub.hSub (HPow.hPow x (HPow.hPow 2 (multiplicity 2 (HAdd.hAdd n  …
    -/
  · simpa only [even_iff_two_dvd] using hx_odd.pow.sub_odd hy_odd.pow
    /-
      🎉 no goals
    -/
    /-
      case succ.intro.intro.hx
      x y : Int
      hxy : Dvd.dvd 4 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      hx_odd : Odd x
      hxy_even : Even (HSub.hSub x y)
      hy_odd : Odd y
      n : Nat
      h : FiniteMultiplicity 2 (HAdd.hAdd n 1)
      hpn : Not (Dvd.dvd (HPow.hPow 2 (HAdd.hAdd (multiplicity 2 (HAdd.hAdd n 1)) 1) …
      k : Nat
      hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow 2 (multiplicity 2 (HAdd.hAdd n 1 …
      ⊢ Not (Dvd.dvd 2 (HPow.hPow x (HPow.hPow 2 (multiplicity 2 (HAdd.hAdd n 1)))))
    -/
  · simpa only [even_iff_two_dvd, ← Int.not_even_iff_odd] using hx_odd.pow
    /-
      🎉 no goals
    -/
  /-
    case succ.intro.intro.hn
    x y : Int
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    hx_odd : Odd x
    hxy_even : Even (HSub.hSub x y)
    hy_odd : Odd y
    n : Nat
    h : FiniteMultiplicity 2 (HAdd.hAdd n 1)
    hpn : Not (Dvd.dvd (HPow.hPow 2 (HAdd.hAdd (multiplicity 2 (HAdd.hAdd n 1)) 1) …
    k : Nat
    hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow 2 (multiplicity 2 (HAdd.hAdd n 1 …
    ⊢ Not (Dvd.dvd 2 ↑k)
  -/
  erw [Int.natCast_dvd_natCast]
  -- `erw` to deal with `2 : ℤ` vs `(2 : ℕ) : ℤ`
  /-
    case succ.intro.intro.hn
    x y : Int
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    hx_odd : Odd x
    hxy_even : Even (HSub.hSub x y)
    hy_odd : Odd y
    n : Nat
    h : FiniteMultiplicity 2 (HAdd.hAdd n 1)
    hpn : Not (Dvd.dvd (HPow.hPow 2 (HAdd.hAdd (multiplicity 2 (HAdd.hAdd n 1)) 1) …
    k : Nat
    hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow 2 (multiplicity 2 (HAdd.hAdd n 1 …
    ⊢ Not (Dvd.dvd 2 k)
  -/
  contrapose! hpn
  /-
    case succ.intro.intro.hn
    x y : Int
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    hx_odd : Odd x
    hxy_even : Even (HSub.hSub x y)
    hy_odd : Odd y
    n : Nat
    h : FiniteMultiplicity 2 (HAdd.hAdd n 1)
    k : Nat
    hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow 2 (multiplicity 2 (HAdd.hAdd n 1 …
    hpn : Dvd.dvd 2 k
    ⊢ Dvd.dvd (HPow.hPow 2 (HAdd.hAdd (multiplicity 2 (HAdd.hAdd n 1)) 1)) (HAdd.h …
  -/
  rw [pow_succ]
  /-
    case succ.intro.intro.hn
    x y : Int
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    hx_odd : Odd x
    hxy_even : Even (HSub.hSub x y)
    hy_odd : Odd y
    n : Nat
    h : FiniteMultiplicity 2 (HAdd.hAdd n 1)
    k : Nat
    hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow 2 (multiplicity 2 (HAdd.hAdd n 1 …
    hpn : Dvd.dvd 2 k
    ⊢ Dvd.dvd (HMul.hMul (HPow.hPow 2 (multiplicity 2 (HAdd.hAdd n 1))) 2) (HAdd.h …
  -/
  conv_rhs => rw [hk]
  /-
    case succ.intro.intro.hn
    x y : Int
    hxy : Dvd.dvd 4 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    hx_odd : Odd x
    hxy_even : Even (HSub.hSub x y)
    hy_odd : Odd y
    n : Nat
    h : FiniteMultiplicity 2 (HAdd.hAdd n 1)
    k : Nat
    hk : Eq (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow 2 (multiplicity 2 (HAdd.hAdd n 1 …
    hpn : Dvd.dvd 2 k
    ⊢ Dvd.dvd (HMul.hMul (HPow.hPow 2 (multiplicity 2 (HAdd.hAdd n 1))) 2) (HMul.h …
  -/
  exact mul_dvd_mul_left _ hpn
  /-
    🎉 no goals
  -/


/-- **Lifting the exponent lemma** for `p = 2` -/
theorem Int.two_pow_sub_pow {x y : ℤ} {n : ℕ} (hxy : 2 ∣ x - y) (hx : ¬2 ∣ x) (hn : Even n) :
    emultiplicity 2 (x ^ n - y ^ n) + 1 =
      emultiplicity 2 (x + y) + emultiplicity 2 (x - y) + emultiplicity (2 : ℤ) n := by
  have hy : Odd y := by
    rw [← even_iff_two_dvd, Int.not_even_iff_odd] at hx
    replace hxy := (@even_neg _ _ (x - y)).mpr (even_iff_two_dvd.mpr hxy)
    convert Even.add_odd hxy hx
    abel
  /-
    x y : Int
    n : Nat
    hxy : Dvd.dvd 2 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    hn : Even n
    hy : Odd y
    ⊢ Eq (HAdd.hAdd (emultiplicity 2 (HSub.hSub (HPow.hPow x n) (HPow.hPow y n)))  …
  -/
  cases' hn with d hd
  /-
    case intro
    x y : Int
    n : Nat
    hxy : Dvd.dvd 2 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    hy : Odd y
    d : Nat
    hd : Eq n (HAdd.hAdd d d)
    ⊢ Eq (HAdd.hAdd (emultiplicity 2 (HSub.hSub (HPow.hPow x n) (HPow.hPow y n)))  …
  -/
  subst hd
  /-
    case intro
    x y : Int
    hxy : Dvd.dvd 2 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    hy : Odd y
    d : Nat
    ⊢ Eq (HAdd.hAdd (emultiplicity 2 (HSub.hSub (HPow.hPow x (HAdd.hAdd d d)) (HPo …
  -/
  simp only [← two_mul, pow_mul]
  have hxy4 : 4 ∣ x ^ 2 - y ^ 2 := by
    rw [Int.dvd_iff_emod_eq_zero, Int.sub_emod, Int.sq_mod_four_eq_one_of_odd _,
      Int.sq_mod_four_eq_one_of_odd hy]
    · norm_num
    · simp only [← Int.not_even_iff_odd, even_iff_two_dvd, hx, not_false_iff]
  rw [Int.two_pow_sub_pow' d hxy4 _, sq_sub_sq, ← Int.ofNat_mul_out,
    emultiplicity_mul Int.prime_two, emultiplicity_mul Int.prime_two]
    /-
      case intro
      x y : Int
      hxy : Dvd.dvd 2 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      hy : Odd y
      d : Nat
      hxy4 : Dvd.dvd 4 (HSub.hSub (HPow.hPow x 2) (HPow.hPow y 2))
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (emultiplicity 2 (HAdd.hAdd x y)) (emult …
    -/
  · suffices emultiplicity (2 : ℤ) ↑(2 : ℕ) = 1 by rw [this, add_comm 1, ← add_assoc]
    /-
      case intro
      x y : Int
      hxy : Dvd.dvd 2 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      hy : Odd y
      d : Nat
      hxy4 : Dvd.dvd 4 (HSub.hSub (HPow.hPow x 2) (HPow.hPow y 2))
      ⊢ Eq (emultiplicity 2 ↑2) 1
    -/
    norm_cast
    /-
      case intro
      x y : Int
      hxy : Dvd.dvd 2 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      hy : Odd y
      d : Nat
      hxy4 : Dvd.dvd 4 (HSub.hSub (HPow.hPow x 2) (HPow.hPow y 2))
      ⊢ Eq (emultiplicity 2 2) 1
    -/
    rw [FiniteMultiplicity.emultiplicity_self]
    /-
      case intro
      x y : Int
      hxy : Dvd.dvd 2 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      hy : Odd y
      d : Nat
      hxy4 : Dvd.dvd 4 (HSub.hSub (HPow.hPow x 2) (HPow.hPow y 2))
      ⊢ FiniteMultiplicity 2 2
    -/
    rw [Nat.finiteMultiplicity_iff]
    /-
      case intro
      x y : Int
      hxy : Dvd.dvd 2 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      hy : Odd y
      d : Nat
      hxy4 : Dvd.dvd 4 (HSub.hSub (HPow.hPow x 2) (HPow.hPow y 2))
      ⊢ And (Ne 2 1) (LT.lt 0 2)
    -/
    decide
    /-
      🎉 no goals
    -/
    /-
      x y : Int
      hxy : Dvd.dvd 2 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      hy : Odd y
      d : Nat
      hxy4 : Dvd.dvd 4 (HSub.hSub (HPow.hPow x 2) (HPow.hPow y 2))
      ⊢ Not (Dvd.dvd 2 (HPow.hPow x 2))
    -/
  · rw [← even_iff_two_dvd, Int.not_even_iff_odd]
    /-
      x y : Int
      hxy : Dvd.dvd 2 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      hy : Odd y
      d : Nat
      hxy4 : Dvd.dvd 4 (HSub.hSub (HPow.hPow x 2) (HPow.hPow y 2))
      ⊢ Odd (HPow.hPow x 2)
    -/
    apply Odd.pow
    /-
      case ha
      x y : Int
      hxy : Dvd.dvd 2 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      hy : Odd y
      d : Nat
      hxy4 : Dvd.dvd 4 (HSub.hSub (HPow.hPow x 2) (HPow.hPow y 2))
      ⊢ Odd x
    -/
    simp only [← Int.not_even_iff_odd, even_iff_two_dvd, hx, not_false_iff]
    /-
      🎉 no goals
    -/


theorem Nat.two_pow_sub_pow {x y : ℕ} (hxy : 2 ∣ x - y) (hx : ¬2 ∣ x) {n : ℕ} (hn : Even n) :
    emultiplicity 2 (x ^ n - y ^ n) + 1 =
      emultiplicity 2 (x + y) + emultiplicity 2 (x - y) + emultiplicity 2 n := by
  /-
    x y : Nat
    hxy : Dvd.dvd 2 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    n : Nat
    hn : Even n
    ⊢ Eq (HAdd.hAdd (emultiplicity 2 (HSub.hSub (HPow.hPow x n) (HPow.hPow y n)))  …
  -/
  obtain hyx | hyx := le_total y x
    /-
      case inl
      x y : Nat
      hxy : Dvd.dvd 2 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      n : Nat
      hn : Even n
      hyx : LE.le y x
      ⊢ Eq (HAdd.hAdd (emultiplicity 2 (HSub.hSub (HPow.hPow x n) (HPow.hPow y n)))  …
    -/
  · iterate 3 rw [← Int.natCast_emultiplicity]
    simp only [Int.ofNat_sub hyx, Int.ofNat_sub (pow_le_pow_left' hyx _), Int.ofNat_add,
      Int.natCast_pow]
    /-
      case inl
      x y : Nat
      hxy : Dvd.dvd 2 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      n : Nat
      hn : Even n
      hyx : LE.le y x
      ⊢ Eq (HAdd.hAdd (emultiplicity (↑2) (HSub.hSub (HPow.hPow (↑x) n) (HPow.hPow ( …
    -/
    rw [← Int.natCast_dvd_natCast] at hx
    /-
      case inl
      x y : Nat
      hxy : Dvd.dvd 2 (HSub.hSub x y)
      hx : Not (Dvd.dvd ↑2 ↑x)
      n : Nat
      hn : Even n
      hyx : LE.le y x
      ⊢ Eq (HAdd.hAdd (emultiplicity (↑2) (HSub.hSub (HPow.hPow (↑x) n) (HPow.hPow ( …
    -/
    rw [← Int.natCast_dvd_natCast, Int.ofNat_sub hyx] at hxy
    /-
      case inl
      x y : Nat
      hxy : Dvd.dvd (↑2) (HSub.hSub ↑x ↑y)
      hx : Not (Dvd.dvd ↑2 ↑x)
      n : Nat
      hn : Even n
      hyx : LE.le y x
      ⊢ Eq (HAdd.hAdd (emultiplicity (↑2) (HSub.hSub (HPow.hPow (↑x) n) (HPow.hPow ( …
    -/
    convert Int.two_pow_sub_pow hxy hx hn using 2
    /-
      case h.e'_3.h.e'_6
      x y : Nat
      hxy : Dvd.dvd (↑2) (HSub.hSub ↑x ↑y)
      hx : Not (Dvd.dvd ↑2 ↑x)
      n : Nat
      hn : Even n
      hyx : LE.le y x
      ⊢ Eq (emultiplicity 2 n) (emultiplicity 2 ↑n)
    -/
    rw [← Int.natCast_emultiplicity]
    /-
      case h.e'_3.h.e'_6
      x y : Nat
      hxy : Dvd.dvd (↑2) (HSub.hSub ↑x ↑y)
      hx : Not (Dvd.dvd ↑2 ↑x)
      n : Nat
      hn : Even n
      hyx : LE.le y x
      ⊢ Eq (emultiplicity ↑2 ↑n) (emultiplicity 2 ↑n)
    -/
    rfl
    /-
      🎉 no goals
    -/
  · simp only [Nat.sub_eq_zero_iff_le.mpr hyx,
      Nat.sub_eq_zero_iff_le.mpr (pow_le_pow_left' hyx n), emultiplicity_zero,
      top_add, add_top]


theorem pow_two_sub_pow (hyx : y < x) (hxy : 2 ∣ x - y) (hx : ¬2 ∣ x) {n : ℕ} (hn : n ≠ 0)
    (hneven : Even n) :
    padicValNat 2 (x ^ n - y ^ n) + 1 =
      padicValNat 2 (x + y) + padicValNat 2 (x - y) + padicValNat 2 n := by
  /-
    x y : Nat
    hyx : LT.lt y x
    hxy : Dvd.dvd 2 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    n : Nat
    hn : Ne n 0
    hneven : Even n
    ⊢ Eq (HAdd.hAdd (padicValNat 2 (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))) 1) …
  -/
  simp only [← Nat.cast_inj (R := ℕ∞), Nat.cast_add]
  /-
    x y : Nat
    hyx : LT.lt y x
    hxy : Dvd.dvd 2 (HSub.hSub x y)
    hx : Not (Dvd.dvd 2 x)
    n : Nat
    hn : Ne n 0
    hneven : Even n
    ⊢ Eq (HAdd.hAdd ↑(padicValNat 2 (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))) ↑ …
  -/
  iterate 4 rw [padicValNat_eq_emultiplicity]
    /-
      x y : Nat
      hyx : LT.lt y x
      hxy : Dvd.dvd 2 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      n : Nat
      hn : Ne n 0
      hneven : Even n
      ⊢ Eq (HAdd.hAdd (emultiplicity 2 (HSub.hSub (HPow.hPow x n) (HPow.hPow y n)))  …
    -/
  · exact Nat.two_pow_sub_pow hxy hx hneven
    /-
      🎉 no goals
    -/
    /-
      x y : Nat
      hyx : LT.lt y x
      hxy : Dvd.dvd 2 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      n : Nat
      hn : Ne n 0
      hneven : Even n
      ⊢ LT.lt 0 n
    -/
  · exact hn.bot_lt
    /-
      🎉 no goals
    -/
    /-
      x y : Nat
      hyx : LT.lt y x
      hxy : Dvd.dvd 2 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      n : Nat
      hn : Ne n 0
      hneven : Even n
      ⊢ LT.lt 0 (HSub.hSub x y)
    -/
  · exact Nat.sub_pos_of_lt hyx
    /-
      🎉 no goals
    -/
    /-
      x y : Nat
      hyx : LT.lt y x
      hxy : Dvd.dvd 2 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      n : Nat
      hn : Ne n 0
      hneven : Even n
      ⊢ LT.lt 0 (HAdd.hAdd x y)
    -/
  · omega
    /-
      🎉 no goals
    -/
    /-
      x y : Nat
      hyx : LT.lt y x
      hxy : Dvd.dvd 2 (HSub.hSub x y)
      hx : Not (Dvd.dvd 2 x)
      n : Nat
      hn : Ne n 0
      hneven : Even n
      ⊢ LT.lt 0 (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))
    -/
  · simp only [tsub_pos_iff_lt, Nat.pow_lt_pow_left hyx hn]
    /-
      🎉 no goals
    -/


theorem pow_sub_pow (hyx : y < x) (hxy : p ∣ x - y) (hx : ¬p ∣ x) {n : ℕ} (hn : n ≠ 0) :
    padicValNat p (x ^ n - y ^ n) = padicValNat p (x - y) + padicValNat p n := by
  /-
    x y p : Nat
    hp : Fact (Nat.Prime p)
    hp1 : Odd p
    hyx : LT.lt y x
    hxy : Dvd.dvd p (HSub.hSub x y)
    hx : Not (Dvd.dvd p x)
    n : Nat
    hn : Ne n 0
    ⊢ Eq (padicValNat p (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))) (HAdd.hAdd (p …
  -/
  rw [← Nat.cast_inj (R := ℕ∞), Nat.cast_add]
  /-
    x y p : Nat
    hp : Fact (Nat.Prime p)
    hp1 : Odd p
    hyx : LT.lt y x
    hxy : Dvd.dvd p (HSub.hSub x y)
    hx : Not (Dvd.dvd p x)
    n : Nat
    hn : Ne n 0
    ⊢ Eq (↑(padicValNat p (HSub.hSub (HPow.hPow x n) (HPow.hPow y n)))) (HAdd.hAdd …
  -/
  iterate 3 rw [padicValNat_eq_emultiplicity]
    /-
      x y p : Nat
      hp : Fact (Nat.Prime p)
      hp1 : Odd p
      hyx : LT.lt y x
      hxy : Dvd.dvd p (HSub.hSub x y)
      hx : Not (Dvd.dvd p x)
      n : Nat
      hn : Ne n 0
      ⊢ Eq (emultiplicity p (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))) (HAdd.hAdd  …
    -/
  · exact Nat.emultiplicity_pow_sub_pow hp.out hp1 hxy hx n
    /-
      🎉 no goals
    -/
    /-
      x y p : Nat
      hp : Fact (Nat.Prime p)
      hp1 : Odd p
      hyx : LT.lt y x
      hxy : Dvd.dvd p (HSub.hSub x y)
      hx : Not (Dvd.dvd p x)
      n : Nat
      hn : Ne n 0
      ⊢ LT.lt 0 n
    -/
  · exact hn.bot_lt
    /-
      🎉 no goals
    -/
    /-
      x y p : Nat
      hp : Fact (Nat.Prime p)
      hp1 : Odd p
      hyx : LT.lt y x
      hxy : Dvd.dvd p (HSub.hSub x y)
      hx : Not (Dvd.dvd p x)
      n : Nat
      hn : Ne n 0
      ⊢ LT.lt 0 (HSub.hSub x y)
    -/
  · exact Nat.sub_pos_of_lt hyx
    /-
      🎉 no goals
    -/
    /-
      x y p : Nat
      hp : Fact (Nat.Prime p)
      hp1 : Odd p
      hyx : LT.lt y x
      hxy : Dvd.dvd p (HSub.hSub x y)
      hx : Not (Dvd.dvd p x)
      n : Nat
      hn : Ne n 0
      ⊢ LT.lt 0 (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))
    -/
  · exact Nat.sub_pos_of_lt (Nat.pow_lt_pow_left hyx hn)
    /-
      🎉 no goals
    -/


theorem pow_add_pow (hxy : p ∣ x + y) (hx : ¬p ∣ x) {n : ℕ} (hn : Odd n) :
    padicValNat p (x ^ n + y ^ n) = padicValNat p (x + y) + padicValNat p n := by
  /-
    x y p : Nat
    hp : Fact (Nat.Prime p)
    hp1 : Odd p
    hxy : Dvd.dvd p (HAdd.hAdd x y)
    hx : Not (Dvd.dvd p x)
    n : Nat
    hn : Odd n
    ⊢ Eq (padicValNat p (HAdd.hAdd (HPow.hPow x n) (HPow.hPow y n))) (HAdd.hAdd (p …
  -/
  cases' y with y
    /-
      case zero
      x p : Nat
      hp : Fact (Nat.Prime p)
      hp1 : Odd p
      hx : Not (Dvd.dvd p x)
      n : Nat
      hn : Odd n
      hxy : Dvd.dvd p (HAdd.hAdd x 0)
      ⊢ Eq (padicValNat p (HAdd.hAdd (HPow.hPow x n) (HPow.hPow 0 n))) (HAdd.hAdd (p …
    -/
  · contradiction
    /-
      🎉 no goals
    -/
  /-
    case succ
    x p : Nat
    hp : Fact (Nat.Prime p)
    hp1 : Odd p
    hx : Not (Dvd.dvd p x)
    n : Nat
    hn : Odd n
    y : Nat
    hxy : Dvd.dvd p (HAdd.hAdd x (HAdd.hAdd y 1))
    ⊢ Eq (padicValNat p (HAdd.hAdd (HPow.hPow x n) (HPow.hPow (HAdd.hAdd y 1) n))) …
  -/
  rw [← Nat.cast_inj (R := ℕ∞), Nat.cast_add]
  /-
    case succ
    x p : Nat
    hp : Fact (Nat.Prime p)
    hp1 : Odd p
    hx : Not (Dvd.dvd p x)
    n : Nat
    hn : Odd n
    y : Nat
    hxy : Dvd.dvd p (HAdd.hAdd x (HAdd.hAdd y 1))
    ⊢ Eq (↑(padicValNat p (HAdd.hAdd (HPow.hPow x n) (HPow.hPow (HAdd.hAdd y 1) n) …
  -/
  iterate 3 rw [padicValNat_eq_emultiplicity]
    /-
      case succ
      x p : Nat
      hp : Fact (Nat.Prime p)
      hp1 : Odd p
      hx : Not (Dvd.dvd p x)
      n : Nat
      hn : Odd n
      y : Nat
      hxy : Dvd.dvd p (HAdd.hAdd x (HAdd.hAdd y 1))
      ⊢ Eq (emultiplicity p (HAdd.hAdd (HPow.hPow x n) (HPow.hPow (HAdd.hAdd y 1) n) …
    -/
  · exact Nat.emultiplicity_pow_add_pow hp.out hp1 hxy hx hn
    /-
      🎉 no goals
    -/
    /-
      case succ
      x p : Nat
      hp : Fact (Nat.Prime p)
      hp1 : Odd p
      hx : Not (Dvd.dvd p x)
      n : Nat
      hn : Odd n
      y : Nat
      hxy : Dvd.dvd p (HAdd.hAdd x (HAdd.hAdd y 1))
      ⊢ LT.lt 0 n
    -/
  · exact Odd.pos hn
    /-
      🎉 no goals
    -/
    /-
      case succ
      x p : Nat
      hp : Fact (Nat.Prime p)
      hp1 : Odd p
      hx : Not (Dvd.dvd p x)
      n : Nat
      hn : Odd n
      y : Nat
      hxy : Dvd.dvd p (HAdd.hAdd x (HAdd.hAdd y 1))
      ⊢ LT.lt 0 (HAdd.hAdd x (HAdd.hAdd y 1))
    -/
  · simp only [add_pos_iff, Nat.succ_pos', or_true]
    /-
      🎉 no goals
    -/
    /-
      case succ
      x p : Nat
      hp : Fact (Nat.Prime p)
      hp1 : Odd p
      hx : Not (Dvd.dvd p x)
      n : Nat
      hn : Odd n
      y : Nat
      hxy : Dvd.dvd p (HAdd.hAdd x (HAdd.hAdd y 1))
      ⊢ LT.lt 0 (HAdd.hAdd (HPow.hPow x n) (HPow.hPow (HAdd.hAdd y 1) n))
    -/
  · exact Nat.lt_add_left _ (pow_pos y.succ_pos _)
    /-
      🎉 no goals
    -/


