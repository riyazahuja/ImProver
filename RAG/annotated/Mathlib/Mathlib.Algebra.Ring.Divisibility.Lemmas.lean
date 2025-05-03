lemma dvd_smul_of_dvd {M : Type*} [SMul M R] [Semigroup R] [SMulCommClass M R R] {x y : R}
    (m : M) (h : x ∣ y) : x ∣ m • y :=
                               /-
                                 R : Type u_1
                                 M : Type u_2
                                 inst✝² : SMul M R
                                 inst✝¹ : Semigroup R
                                 inst✝ : SMulCommClass M R R
                                 x y : R
                                 m : M
                                 h : Dvd.dvd x y
                                 k : R
                                 hk : Eq y (HMul.hMul x k)
                                 ⊢ Eq (HSMul.hSMul m y) (HMul.hMul x (HSMul.hSMul m k))
                               -/
  let ⟨k, hk⟩ := h; ⟨m • k, by rw [mul_smul_comm, ← hk]⟩
                               /-
                                 🎉 no goals
                               -/


lemma dvd_nsmul_of_dvd [NonUnitalSemiring R] {x y : R} (n : ℕ) (h : x ∣ y) : x ∣ n • y :=
  dvd_smul_of_dvd n h


lemma dvd_zsmul_of_dvd [NonUnitalRing R] {x y : R} (z : ℤ) (h : x ∣ y) : x ∣ z • y :=
  dvd_smul_of_dvd z h


lemma pow_dvd_add_pow_of_pow_eq_zero_right (hp : n + m ≤ p + 1) (h_comm : Commute x y)
    (hy : y ^ n = 0) : x ^ m ∣ (x + y) ^ p := by
  /-
    R : Type u_1
    x y : R
    n m p : Nat
    inst✝ : Semiring R
    hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
    h_comm : Commute x y
    hy : Eq (HPow.hPow y n) 0
    ⊢ Dvd.dvd (HPow.hPow x m) (HPow.hPow (HAdd.hAdd x y) p)
  -/
  rw [h_comm.add_pow']
  /-
    R : Type u_1
    x y : R
    n m p : Nat
    inst✝ : Semiring R
    hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
    h_comm : Commute x y
    hy : Eq (HPow.hPow y n) 0
    ⊢ Dvd.dvd (HPow.hPow x m) ((Finset.HasAntidiagonal.antidiagonal p).sum fun m = …
  -/
  refine Finset.dvd_sum fun ⟨i, j⟩ hij ↦ ?_
  /-
    R : Type u_1
    x y : R
    n m p : Nat
    inst✝ : Semiring R
    hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
    h_comm : Commute x y
    hy : Eq (HPow.hPow y n) 0
    x✝ : Prod Nat Nat
    i j : Nat
    hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal p) { fst := i, snd : …
    ⊢ Dvd.dvd (HPow.hPow x m) (HSMul.hSMul (p.choose { fst := i, snd := j }.1) (HM …
  -/
  replace hij : i + j = p := by simpa using hij
  /-
    R : Type u_1
    x y : R
    n m p : Nat
    inst✝ : Semiring R
    hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
    h_comm : Commute x y
    hy : Eq (HPow.hPow y n) 0
    x✝ : Prod Nat Nat
    i j : Nat
    hij : Eq (HAdd.hAdd i j) p
    ⊢ Dvd.dvd (HPow.hPow x m) (HSMul.hSMul (p.choose { fst := i, snd := j }.1) (HM …
  -/
  apply dvd_nsmul_of_dvd
  /-
    case h
    R : Type u_1
    x y : R
    n m p : Nat
    inst✝ : Semiring R
    hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
    h_comm : Commute x y
    hy : Eq (HPow.hPow y n) 0
    x✝ : Prod Nat Nat
    i j : Nat
    hij : Eq (HAdd.hAdd i j) p
    ⊢ Dvd.dvd (HPow.hPow x m) (HMul.hMul (HPow.hPow x { fst := i, snd := j }.1) (H …
  -/
  rcases le_or_lt m i with (hi : m ≤ i) | (hi : i + 1 ≤ m)
    /-
      case h.inl
      R : Type u_1
      x y : R
      n m p : Nat
      inst✝ : Semiring R
      hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
      h_comm : Commute x y
      hy : Eq (HPow.hPow y n) 0
      x✝ : Prod Nat Nat
      i j : Nat
      hij : Eq (HAdd.hAdd i j) p
      hi : LE.le m i
      ⊢ Dvd.dvd (HPow.hPow x m) (HMul.hMul (HPow.hPow x { fst := i, snd := j }.1) (H …
    -/
  · exact dvd_mul_of_dvd_left (pow_dvd_pow x hi) _
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      R : Type u_1
      x y : R
      n m p : Nat
      inst✝ : Semiring R
      hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
      h_comm : Commute x y
      hy : Eq (HPow.hPow y n) 0
      x✝ : Prod Nat Nat
      i j : Nat
      hij : Eq (HAdd.hAdd i j) p
      hi : LE.le (HAdd.hAdd i 1) m
      ⊢ Dvd.dvd (HPow.hPow x m) (HMul.hMul (HPow.hPow x { fst := i, snd := j }.1) (H …
    -/
  · simp [pow_eq_zero_of_le (by omega : n ≤ j) hy]
    /-
      🎉 no goals
    -/


lemma pow_dvd_add_pow_of_pow_eq_zero_left (hp : n + m ≤ p + 1) (h_comm : Commute x y)
    (hx : x ^ n = 0) : y ^ m ∣ (x + y) ^ p :=
  add_comm x y ▸ h_comm.symm.pow_dvd_add_pow_of_pow_eq_zero_right hp hx


lemma pow_dvd_pow_of_sub_pow_eq_zero (hp : n + m ≤ p + 1) (h_comm : Commute x y)
    (h : (x - y) ^ n = 0) : x ^ m ∣ y ^ p := by
  /-
    R : Type u_1
    x y : R
    n m p : Nat
    inst✝ : Ring R
    hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
    h_comm : Commute x y
    h : Eq (HPow.hPow (HSub.hSub x y) n) 0
    ⊢ Dvd.dvd (HPow.hPow x m) (HPow.hPow y p)
  -/
  rw [← sub_add_cancel y x]
  /-
    R : Type u_1
    x y : R
    n m p : Nat
    inst✝ : Ring R
    hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
    h_comm : Commute x y
    h : Eq (HPow.hPow (HSub.hSub x y) n) 0
    ⊢ Dvd.dvd (HPow.hPow x m) (HPow.hPow (HAdd.hAdd (HSub.hSub y x) x) p)
  -/
  apply (h_comm.symm.sub_left rfl).pow_dvd_add_pow_of_pow_eq_zero_left hp _
  /-
    R : Type u_1
    x y : R
    n m p : Nat
    inst✝ : Ring R
    hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
    h_comm : Commute x y
    h : Eq (HPow.hPow (HSub.hSub x y) n) 0
    ⊢ Eq (HPow.hPow (HSub.hSub y x) n) 0
  -/
  rw [← neg_sub x y, neg_pow, h, mul_zero]
  /-
    🎉 no goals
  -/


lemma pow_dvd_pow_of_add_pow_eq_zero (hp : n + m ≤ p + 1) (h_comm : Commute x y)
    (h : (x + y) ^ n = 0) : x ^ m ∣ y ^ p := by
  /-
    R : Type u_1
    x y : R
    n m p : Nat
    inst✝ : Ring R
    hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
    h_comm : Commute x y
    h : Eq (HPow.hPow (HAdd.hAdd x y) n) 0
    ⊢ Dvd.dvd (HPow.hPow x m) (HPow.hPow y p)
  -/
  rw [← neg_neg y, neg_pow']
  /-
    R : Type u_1
    x y : R
    n m p : Nat
    inst✝ : Ring R
    hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
    h_comm : Commute x y
    h : Eq (HPow.hPow (HAdd.hAdd x y) n) 0
    ⊢ Dvd.dvd (HPow.hPow x m) (HMul.hMul (HPow.hPow (Neg.neg y) p) (HPow.hPow (-1) …
  -/
  apply dvd_mul_of_dvd_left
  /-
    case h
    R : Type u_1
    x y : R
    n m p : Nat
    inst✝ : Ring R
    hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
    h_comm : Commute x y
    h : Eq (HPow.hPow (HAdd.hAdd x y) n) 0
    ⊢ Dvd.dvd (HPow.hPow x m) (HPow.hPow (Neg.neg y) p)
  -/
  apply h_comm.neg_right.pow_dvd_pow_of_sub_pow_eq_zero hp
  /-
    case h
    R : Type u_1
    x y : R
    n m p : Nat
    inst✝ : Ring R
    hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
    h_comm : Commute x y
    h : Eq (HPow.hPow (HAdd.hAdd x y) n) 0
    ⊢ Eq (HPow.hPow (HSub.hSub x (Neg.neg y)) n) 0
  -/
  simpa
  /-
    🎉 no goals
  -/


lemma pow_dvd_sub_pow_of_pow_eq_zero_right (hp : n + m ≤ p + 1) (h_comm : Commute x y)
    (hy : y ^ n = 0) : x ^ m ∣ (x - y) ^ p :=
                                                               /-
                                                                 R : Type u_1
                                                                 x y : R
                                                                 n m p : Nat
                                                                 inst✝ : Ring R
                                                                 hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
                                                                 h_comm : Commute x y
                                                                 hy : Eq (HPow.hPow y n) 0
                                                                 ⊢ Eq (HPow.hPow (HSub.hSub x (HSub.hSub x y)) n) 0
                                                               -/
  (sub_right rfl h_comm).pow_dvd_pow_of_sub_pow_eq_zero hp (by simpa)
                                                               /-
                                                                 🎉 no goals
                                                               -/


lemma pow_dvd_sub_pow_of_pow_eq_zero_left (hp : n + m ≤ p + 1) (h_comm : Commute x y)
    (hx : x ^ n = 0) : y ^ m ∣ (x - y) ^ p := by
  /-
    R : Type u_1
    x y : R
    n m p : Nat
    inst✝ : Ring R
    hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
    h_comm : Commute x y
    hx : Eq (HPow.hPow x n) 0
    ⊢ Dvd.dvd (HPow.hPow y m) (HPow.hPow (HSub.hSub x y) p)
  -/
  rw [← neg_sub y x, neg_pow']
  /-
    R : Type u_1
    x y : R
    n m p : Nat
    inst✝ : Ring R
    hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
    h_comm : Commute x y
    hx : Eq (HPow.hPow x n) 0
    ⊢ Dvd.dvd (HPow.hPow y m) (HMul.hMul (HPow.hPow (HSub.hSub y x) p) (HPow.hPow  …
  -/
  apply dvd_mul_of_dvd_left
  /-
    case h
    R : Type u_1
    x y : R
    n m p : Nat
    inst✝ : Ring R
    hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
    h_comm : Commute x y
    hx : Eq (HPow.hPow x n) 0
    ⊢ Dvd.dvd (HPow.hPow y m) (HPow.hPow (HSub.hSub y x) p)
  -/
  exact h_comm.symm.pow_dvd_sub_pow_of_pow_eq_zero_right hp hx
  /-
    🎉 no goals
  -/


lemma add_pow_dvd_pow_of_pow_eq_zero_right (hp : n + m ≤ p + 1) (h_comm : Commute x y)
    (hx : x ^ n = 0) : (x + y) ^ m ∣ y ^ p :=
                                                              /-
                                                                R : Type u_1
                                                                x y : R
                                                                n m p : Nat
                                                                inst✝ : Ring R
                                                                hp : LE.le (HAdd.hAdd n m) (HAdd.hAdd p 1)
                                                                h_comm : Commute x y
                                                                hx : Eq (HPow.hPow x n) 0
                                                                ⊢ Eq (HPow.hPow (HSub.hSub (HAdd.hAdd x y) y) n) 0
                                                              -/
  (h_comm.add_left rfl).pow_dvd_pow_of_sub_pow_eq_zero hp (by simpa)
                                                              /-
                                                                🎉 no goals
                                                              -/


lemma add_pow_dvd_pow_of_pow_eq_zero_left (hp : n + m ≤ p + 1) (h_comm : Commute x y)
    (hy : y ^ n = 0) : (x + y) ^ m ∣ x ^ p :=
  add_comm x y ▸ h_comm.symm.add_pow_dvd_pow_of_pow_eq_zero_right hp hy


lemma dvd_mul_sub_mul_mul_left_of_dvd {p a b c d x y : R}
    (h1 : p ∣ a * x + b * y) (h2 : p ∣ c * x + d * y) : p ∣ (a * d - b * c) * x := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    p a b c d x y : R
    h1 : Dvd.dvd p (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y))
    h2 : Dvd.dvd p (HAdd.hAdd (HMul.hMul c x) (HMul.hMul d y))
    ⊢ Dvd.dvd p (HMul.hMul (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) x)
  -/
  obtain ⟨k1, hk1⟩ := h1
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    p a b c d x y : R
    h2 : Dvd.dvd p (HAdd.hAdd (HMul.hMul c x) (HMul.hMul d y))
    k1 : R
    hk1 : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y)) (HMul.hMul p k1)
    ⊢ Dvd.dvd p (HMul.hMul (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) x)
  -/
  obtain ⟨k2, hk2⟩ := h2
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    p a b c d x y k1 : R
    hk1 : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y)) (HMul.hMul p k1)
    k2 : R
    hk2 : Eq (HAdd.hAdd (HMul.hMul c x) (HMul.hMul d y)) (HMul.hMul p k2)
    ⊢ Dvd.dvd p (HMul.hMul (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) x)
  -/
  refine ⟨d * k1 - b * k2, ?_⟩
  rw [show (a * d - b * c) * x = a * x * d - c * x * b by ring, eq_sub_of_add_eq hk1,
    eq_sub_of_add_eq hk2]
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    p a b c d x y k1 : R
    hk1 : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y)) (HMul.hMul p k1)
    k2 : R
    hk2 : Eq (HAdd.hAdd (HMul.hMul c x) (HMul.hMul d y)) (HMul.hMul p k2)
    ⊢ Eq (HSub.hSub (HMul.hMul (HSub.hSub (HMul.hMul p k1) (HMul.hMul b y)) d) (HM …
  -/
  ring
  /-
    🎉 no goals
  -/


lemma dvd_mul_sub_mul_mul_right_of_dvd {p a b c d x y : R}
    (h1 : p ∣ a * x + b * y) (h2 : p ∣ c * x + d * y) : p ∣ (a * d - b * c) * y :=
  (mul_comm a _ ▸ mul_comm c _ ▸ dvd_mul_sub_mul_mul_left_of_dvd
    (add_comm (c * x) _ ▸ h2) (add_comm (a * x) _ ▸ h1))


lemma dvd_mul_sub_mul_mul_gcd_of_dvd {p a b c d x y : R} [IsDomain R] [GCDMonoid R]
    (h1 : p ∣ a * x + b * y) (h2 : p ∣ c * x + d * y) : p ∣ (a * d - b * c) * gcd x y := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    p a b c d x y : R
    inst✝¹ : IsDomain R
    inst✝ : GCDMonoid R
    h1 : Dvd.dvd p (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y))
    h2 : Dvd.dvd p (HAdd.hAdd (HMul.hMul c x) (HMul.hMul d y))
    ⊢ Dvd.dvd p (HMul.hMul (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) (GCDMonoid. …
  -/
  rw [← (gcd_mul_left' (a*d - b*c) x y).dvd_iff_dvd_right]
  exact (dvd_gcd_iff _ _ _).2 ⟨dvd_mul_sub_mul_mul_left_of_dvd h1 h2,
    dvd_mul_sub_mul_mul_right_of_dvd h1 h2⟩


