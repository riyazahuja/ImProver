/-- Shorthand for a trinomial -/
noncomputable def trinomial :=
  C u * X ^ k + C v * X ^ m + C w * X ^ n


theorem trinomial_def : trinomial k m n u v w = C u * X ^ k + C v * X ^ m + C w * X ^ n :=
  rfl


theorem trinomial_leading_coeff' (hkm : k < m) (hmn : m < n) :
    (trinomial k m n u v w).coeff n = w := by
  rw [trinomial_def, coeff_add, coeff_add, coeff_C_mul_X_pow, coeff_C_mul_X_pow, coeff_C_mul_X_pow,
    if_neg (hkm.trans hmn).ne', if_neg hmn.ne', if_pos rfl, zero_add, zero_add]


theorem trinomial_middle_coeff (hkm : k < m) (hmn : m < n) :
    (trinomial k m n u v w).coeff m = v := by
  rw [trinomial_def, coeff_add, coeff_add, coeff_C_mul_X_pow, coeff_C_mul_X_pow, coeff_C_mul_X_pow,
    if_neg hkm.ne', if_pos rfl, if_neg hmn.ne, zero_add, add_zero]


theorem trinomial_trailing_coeff' (hkm : k < m) (hmn : m < n) :
    (trinomial k m n u v w).coeff k = u := by
  rw [trinomial_def, coeff_add, coeff_add, coeff_C_mul_X_pow, coeff_C_mul_X_pow, coeff_C_mul_X_pow,
    if_pos rfl, if_neg hkm.ne, if_neg (hkm.trans hmn).ne, add_zero, add_zero]


theorem trinomial_natDegree (hkm : k < m) (hmn : m < n) (hw : w ≠ 0) :
    (trinomial k m n u v w).natDegree = n := by
  refine
    natDegree_eq_of_degree_eq_some
      ((Finset.sup_le fun i h => ?_).antisymm <|
        le_degree_of_ne_zero <| by rwa [trinomial_leading_coeff' hkm hmn])
  /-
    R : Type u_1
    inst✝ : Semiring R
    k m n : Nat
    u v w : R
    hkm : LT.lt k m
    hmn : LT.lt m n
    hw : Ne w 0
    i : Nat
    h : Membership.mem (Polynomial.trinomial k m n u v w).support i
    ⊢ LE.le ↑i ↑n
  -/
  replace h := support_trinomial' k m n u v w h
  /-
    R : Type u_1
    inst✝ : Semiring R
    k m n : Nat
    u v w : R
    hkm : LT.lt k m
    hmn : LT.lt m n
    hw : Ne w 0
    i : Nat
    h : Membership.mem (Insert.insert k (Insert.insert m (Singleton.singleton n))) i
    ⊢ LE.le ↑i ↑n
  -/
  rw [mem_insert, mem_insert, mem_singleton] at h
  /-
    R : Type u_1
    inst✝ : Semiring R
    k m n : Nat
    u v w : R
    hkm : LT.lt k m
    hmn : LT.lt m n
    hw : Ne w 0
    i : Nat
    h : Or (Eq i k) (Or (Eq i m) (Eq i n))
    ⊢ LE.le ↑i ↑n
  -/
  rcases h with (rfl | rfl | rfl)
    /-
      case inl
      R : Type u_1
      inst✝ : Semiring R
      m n : Nat
      u v w : R
      hmn : LT.lt m n
      hw : Ne w 0
      i : Nat
      hkm : LT.lt i m
      ⊢ LE.le ↑i ↑n
    -/
  · exact WithBot.coe_le_coe.mpr (hkm.trans hmn).le
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      R : Type u_1
      inst✝ : Semiring R
      k n : Nat
      u v w : R
      hw : Ne w 0
      i : Nat
      hkm : LT.lt k i
      hmn : LT.lt i n
      ⊢ LE.le ↑i ↑n
    -/
  · exact WithBot.coe_le_coe.mpr hmn.le
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      R : Type u_1
      inst✝ : Semiring R
      k m : Nat
      u v w : R
      hkm : LT.lt k m
      hw : Ne w 0
      i : Nat
      hmn : LT.lt m i
      ⊢ LE.le ↑i ↑i
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/


theorem trinomial_natTrailingDegree (hkm : k < m) (hmn : m < n) (hu : u ≠ 0) :
    (trinomial k m n u v w).natTrailingDegree = k := by
  refine
    natTrailingDegree_eq_of_trailingDegree_eq_some
      ((Finset.le_inf fun i h => ?_).antisymm <|
          trailingDegree_le_of_ne_zero <| by rwa [trinomial_trailing_coeff' hkm hmn]).symm
  /-
    R : Type u_1
    inst✝ : Semiring R
    k m n : Nat
    u v w : R
    hkm : LT.lt k m
    hmn : LT.lt m n
    hu : Ne u 0
    i : Nat
    h : Membership.mem (Polynomial.trinomial k m n u v w).support i
    ⊢ LE.le ↑k ↑i
  -/
  replace h := support_trinomial' k m n u v w h
  /-
    R : Type u_1
    inst✝ : Semiring R
    k m n : Nat
    u v w : R
    hkm : LT.lt k m
    hmn : LT.lt m n
    hu : Ne u 0
    i : Nat
    h : Membership.mem (Insert.insert k (Insert.insert m (Singleton.singleton n))) i
    ⊢ LE.le ↑k ↑i
  -/
  rw [mem_insert, mem_insert, mem_singleton] at h
  /-
    R : Type u_1
    inst✝ : Semiring R
    k m n : Nat
    u v w : R
    hkm : LT.lt k m
    hmn : LT.lt m n
    hu : Ne u 0
    i : Nat
    h : Or (Eq i k) (Or (Eq i m) (Eq i n))
    ⊢ LE.le ↑k ↑i
  -/
  rcases h with (rfl | rfl | rfl)
    /-
      case inl
      R : Type u_1
      inst✝ : Semiring R
      m n : Nat
      u v w : R
      hmn : LT.lt m n
      hu : Ne u 0
      i : Nat
      hkm : LT.lt i m
      ⊢ LE.le ↑i ↑i
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      R : Type u_1
      inst✝ : Semiring R
      k n : Nat
      u v w : R
      hu : Ne u 0
      i : Nat
      hkm : LT.lt k i
      hmn : LT.lt i n
      ⊢ LE.le ↑k ↑i
    -/
  · exact WithTop.coe_le_coe.mpr hkm.le
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      R : Type u_1
      inst✝ : Semiring R
      k m : Nat
      u v w : R
      hkm : LT.lt k m
      hu : Ne u 0
      i : Nat
      hmn : LT.lt m i
      ⊢ LE.le ↑k ↑i
    -/
  · exact WithTop.coe_le_coe.mpr (hkm.trans hmn).le
    /-
      🎉 no goals
    -/


theorem trinomial_leadingCoeff (hkm : k < m) (hmn : m < n) (hw : w ≠ 0) :
    (trinomial k m n u v w).leadingCoeff = w := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    k m n : Nat
    u v w : R
    hkm : LT.lt k m
    hmn : LT.lt m n
    hw : Ne w 0
    ⊢ Eq (Polynomial.trinomial k m n u v w).leadingCoeff w
  -/
  rw [leadingCoeff, trinomial_natDegree hkm hmn hw, trinomial_leading_coeff' hkm hmn]
  /-
    🎉 no goals
  -/


theorem trinomial_trailingCoeff (hkm : k < m) (hmn : m < n) (hu : u ≠ 0) :
    (trinomial k m n u v w).trailingCoeff = u := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    k m n : Nat
    u v w : R
    hkm : LT.lt k m
    hmn : LT.lt m n
    hu : Ne u 0
    ⊢ Eq (Polynomial.trinomial k m n u v w).trailingCoeff u
  -/
  rw [trailingCoeff, trinomial_natTrailingDegree hkm hmn hu, trinomial_trailing_coeff' hkm hmn]
  /-
    🎉 no goals
  -/


theorem trinomial_monic (hkm : k < m) (hmn : m < n) : (trinomial k m n u v 1).Monic := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    k m n : Nat
    u v : R
    hkm : LT.lt k m
    hmn : LT.lt m n
    ⊢ (Polynomial.trinomial k m n u v 1).Monic
  -/
  nontriviality R
  /-
    R : Type u_1
    inst✝ : Semiring R
    k m n : Nat
    u v : R
    hkm : LT.lt k m
    hmn : LT.lt m n
    a✝ : Nontrivial R
    ⊢ (Polynomial.trinomial k m n u v 1).Monic
  -/
  exact trinomial_leadingCoeff hkm hmn one_ne_zero
  /-
    🎉 no goals
  -/


theorem trinomial_mirror (hkm : k < m) (hmn : m < n) (hu : u ≠ 0) (hw : w ≠ 0) :
    (trinomial k m n u v w).mirror = trinomial k (n - m + k) n w v u := by
  rw [mirror, trinomial_natTrailingDegree hkm hmn hu, reverse, trinomial_natDegree hkm hmn hw,
    trinomial_def, reflect_add, reflect_add, reflect_C_mul_X_pow, reflect_C_mul_X_pow,
    reflect_C_mul_X_pow, revAt_le (hkm.trans hmn).le, revAt_le hmn.le, revAt_le le_rfl, add_mul,
    add_mul, mul_assoc, mul_assoc, mul_assoc, ← pow_add, ← pow_add, ← pow_add,
    Nat.sub_add_cancel (hkm.trans hmn).le, Nat.sub_self, zero_add, add_comm, add_comm (C u * X ^ n),
    ← add_assoc, ← trinomial_def]


theorem trinomial_support (hkm : k < m) (hmn : m < n) (hu : u ≠ 0) (hv : v ≠ 0) (hw : w ≠ 0) :
    (trinomial k m n u v w).support = {k, m, n} :=
  support_trinomial hkm hmn hu hv hw


/-- A unit trinomial is a trinomial with unit coefficients. -/
def IsUnitTrinomial :=
  ∃ (k m n : ℕ) (_ : k < m) (_ : m < n) (u v w : Units ℤ), p = trinomial k m n (u : ℤ) v w


theorem not_isUnit (hp : p.IsUnitTrinomial) : ¬IsUnit p := by
  /-
    p : Polynomial Int
    hp : p.IsUnitTrinomial
    ⊢ Not (IsUnit p)
  -/
  obtain ⟨k, m, n, hkm, hmn, u, v, w, rfl⟩ := hp
  exact fun h =>
    ne_zero_of_lt hmn
      ((trinomial_natDegree hkm hmn w.ne_zero).symm.trans
        (natDegree_eq_of_degree_eq_some (degree_eq_zero_of_isUnit h)))


theorem card_support_eq_three (hp : p.IsUnitTrinomial) : #p.support = 3 := by
  /-
    p : Polynomial Int
    hp : p.IsUnitTrinomial
    ⊢ Eq p.support.card 3
  -/
  obtain ⟨k, m, n, hkm, hmn, u, v, w, rfl⟩ := hp
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    u v w : Units Int
    ⊢ Eq (Polynomial.trinomial k m n ↑u ↑v ↑w).support.card 3
  -/
  exact card_support_trinomial hkm hmn u.ne_zero v.ne_zero w.ne_zero
  /-
    🎉 no goals
  -/


theorem ne_zero (hp : p.IsUnitTrinomial) : p ≠ 0 := by
  /-
    p : Polynomial Int
    hp : p.IsUnitTrinomial
    ⊢ Ne p 0
  -/
  rintro rfl
  /-
    hp : Polynomial.IsUnitTrinomial 0
    ⊢ False
  -/
  simpa using hp.card_support_eq_three
  /-
    🎉 no goals
  -/


theorem coeff_isUnit (hp : p.IsUnitTrinomial) {k : ℕ} (hk : k ∈ p.support) :
    IsUnit (p.coeff k) := by
  /-
    p : Polynomial Int
    hp : p.IsUnitTrinomial
    k : Nat
    hk : Membership.mem p.support k
    ⊢ IsUnit (p.coeff k)
  -/
  obtain ⟨k, m, n, hkm, hmn, u, v, w, rfl⟩ := hp
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    k✝ k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    u v w : Units Int
    hk : Membership.mem (Polynomial.trinomial k m n ↑u ↑v ↑w).support k✝
    ⊢ IsUnit ((Polynomial.trinomial k m n ↑u ↑v ↑w).coeff k✝)
  -/
  have := support_trinomial' k m n (u : ℤ) v w hk
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    k✝ k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    u v w : Units Int
    hk : Membership.mem (Polynomial.trinomial k m n ↑u ↑v ↑w).support k✝
    this : Membership.mem (Insert.insert k (Insert.insert m (Singleton.singleton n …
    ⊢ IsUnit ((Polynomial.trinomial k m n ↑u ↑v ↑w).coeff k✝)
  -/
  rw [mem_insert, mem_insert, mem_singleton] at this
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    k✝ k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    u v w : Units Int
    hk : Membership.mem (Polynomial.trinomial k m n ↑u ↑v ↑w).support k✝
    this : Or (Eq k✝ k) (Or (Eq k✝ m) (Eq k✝ n))
    ⊢ IsUnit ((Polynomial.trinomial k m n ↑u ↑v ↑w).coeff k✝)
  -/
  rcases this with (rfl | rfl | rfl)
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.inl
      k m n : Nat
      hmn : LT.lt m n
      u v w : Units Int
      hkm : LT.lt k m
      hk : Membership.mem (Polynomial.trinomial k m n ↑u ↑v ↑w).support k
      ⊢ IsUnit ((Polynomial.trinomial k m n ↑u ↑v ↑w).coeff k)
    -/
  · refine ⟨u, by rw [trinomial_trailing_coeff' hkm hmn]⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.inr.inl
      k✝ k n : Nat
      u v w : Units Int
      hkm : LT.lt k k✝
      hmn : LT.lt k✝ n
      hk : Membership.mem (Polynomial.trinomial k k✝ n ↑u ↑v ↑w).support k✝
      ⊢ IsUnit ((Polynomial.trinomial k k✝ n ↑u ↑v ↑w).coeff k✝)
    -/
  · refine ⟨v, by rw [trinomial_middle_coeff hkm hmn]⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.inr.inr
      k✝ k m : Nat
      hkm : LT.lt k m
      u v w : Units Int
      hmn : LT.lt m k✝
      hk : Membership.mem (Polynomial.trinomial k m k✝ ↑u ↑v ↑w).support k✝
      ⊢ IsUnit ((Polynomial.trinomial k m k✝ ↑u ↑v ↑w).coeff k✝)
    -/
  · refine ⟨w, by rw [trinomial_leading_coeff' hkm hmn]⟩
    /-
      🎉 no goals
    -/


theorem leadingCoeff_isUnit (hp : p.IsUnitTrinomial) : IsUnit p.leadingCoeff :=
  hp.coeff_isUnit (natDegree_mem_support_of_nonzero hp.ne_zero)


theorem trailingCoeff_isUnit (hp : p.IsUnitTrinomial) : IsUnit p.trailingCoeff :=
  hp.coeff_isUnit (natTrailingDegree_mem_support_of_nonzero hp.ne_zero)


theorem isUnitTrinomial_iff :
    p.IsUnitTrinomial ↔ #p.support = 3 ∧ ∀ k ∈ p.support, IsUnit (p.coeff k) := by
  /-
    p : Polynomial Int
    ⊢ Iff p.IsUnitTrinomial (And (Eq p.support.card 3) (∀ (k : Nat), Membership.me …
  -/
  refine ⟨fun hp => ⟨hp.card_support_eq_three, fun k => hp.coeff_isUnit⟩, fun hp => ?_⟩
  /-
    p : Polynomial Int
    hp : And (Eq p.support.card 3) (∀ (k : Nat), Membership.mem p.support k → IsUn …
    ⊢ p.IsUnitTrinomial
  -/
  obtain ⟨k, m, n, hkm, hmn, x, y, z, hx, hy, hz, rfl⟩ := card_support_eq_three.mp hp.1
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    x y z : Int
    hx : Ne x 0
    hy : Ne y 0
    hz : Ne z 0
    hp : And (Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Poly …
    ⊢ (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X k) …
  -/
  rw [support_trinomial hkm hmn hx hy hz] at hp
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    x y z : Int
    hx : Ne x 0
    hy : Ne y 0
    hz : Ne z 0
    hp : And (Eq (Insert.insert k (Insert.insert m (Singleton.singleton n))).card  …
    ⊢ (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X k) …
  -/
  replace hx := hp.2 k (mem_insert_self k {m, n})
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    x y z : Int
    hy : Ne y 0
    hz : Ne z 0
    hp : And (Eq (Insert.insert k (Insert.insert m (Singleton.singleton n))).card  …
    hx : IsUnit ((HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Poly …
    ⊢ (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X k) …
  -/
  replace hy := hp.2 m (mem_insert_of_mem (mem_insert_self m {n}))
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    x y z : Int
    hz : Ne z 0
    hp : And (Eq (Insert.insert k (Insert.insert m (Singleton.singleton n))).card  …
    hx : IsUnit ((HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Poly …
    hy : IsUnit ((HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Poly …
    ⊢ (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X k) …
  -/
  replace hz := hp.2 n (mem_insert_of_mem (mem_insert_of_mem (mem_singleton_self n)))
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    x y z : Int
    hp : And (Eq (Insert.insert k (Insert.insert m (Singleton.singleton n))).card  …
    hx : IsUnit ((HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Poly …
    hy : IsUnit ((HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Poly …
    hz : IsUnit ((HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Poly …
    ⊢ (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X k) …
  -/
  simp_rw [coeff_add, coeff_C_mul, coeff_X_pow_self, mul_one, coeff_X_pow] at hx hy hz
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    x y z : Int
    hp : And (Eq (Insert.insert k (Insert.insert m (Singleton.singleton n))).card  …
    hx : IsUnit (HAdd.hAdd (HAdd.hAdd x (HMul.hMul y (ite (Eq k m) 1 0))) (HMul.hM …
    hy : IsUnit (HAdd.hAdd (HAdd.hAdd (HMul.hMul x (ite (Eq m k) 1 0)) y) (HMul.hM …
    hz : IsUnit (HAdd.hAdd (HAdd.hAdd (HMul.hMul x (ite (Eq n k) 1 0)) (HMul.hMul  …
    ⊢ (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X k) …
  -/
  rw [if_neg hkm.ne, if_neg (hkm.trans hmn).ne] at hx
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    x y z : Int
    hp : And (Eq (Insert.insert k (Insert.insert m (Singleton.singleton n))).card  …
    hx : IsUnit (HAdd.hAdd (HAdd.hAdd x (HMul.hMul y 0)) (HMul.hMul z 0))
    hy : IsUnit (HAdd.hAdd (HAdd.hAdd (HMul.hMul x (ite (Eq m k) 1 0)) y) (HMul.hM …
    hz : IsUnit (HAdd.hAdd (HAdd.hAdd (HMul.hMul x (ite (Eq n k) 1 0)) (HMul.hMul  …
    ⊢ (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X k) …
  -/
  rw [if_neg hkm.ne', if_neg hmn.ne] at hy
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    x y z : Int
    hp : And (Eq (Insert.insert k (Insert.insert m (Singleton.singleton n))).card  …
    hx : IsUnit (HAdd.hAdd (HAdd.hAdd x (HMul.hMul y 0)) (HMul.hMul z 0))
    hy : IsUnit (HAdd.hAdd (HAdd.hAdd (HMul.hMul x 0) y) (HMul.hMul z 0))
    hz : IsUnit (HAdd.hAdd (HAdd.hAdd (HMul.hMul x (ite (Eq n k) 1 0)) (HMul.hMul  …
    ⊢ (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X k) …
  -/
  rw [if_neg (hkm.trans hmn).ne', if_neg hmn.ne'] at hz
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    x y z : Int
    hp : And (Eq (Insert.insert k (Insert.insert m (Singleton.singleton n))).card  …
    hx : IsUnit (HAdd.hAdd (HAdd.hAdd x (HMul.hMul y 0)) (HMul.hMul z 0))
    hy : IsUnit (HAdd.hAdd (HAdd.hAdd (HMul.hMul x 0) y) (HMul.hMul z 0))
    hz : IsUnit (HAdd.hAdd (HAdd.hAdd (HMul.hMul x 0) (HMul.hMul y 0)) z)
    ⊢ (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X k) …
  -/
  simp_rw [mul_zero, zero_add, add_zero] at hx hy hz
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    x y z : Int
    hp : And (Eq (Insert.insert k (Insert.insert m (Singleton.singleton n))).card  …
    hz : IsUnit z
    hx : IsUnit x
    hy : IsUnit y
    ⊢ (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X k) …
  -/
  exact ⟨k, m, n, hkm, hmn, hx.unit, hy.unit, hz.unit, rfl⟩
  /-
    🎉 no goals
  -/


theorem isUnitTrinomial_iff' :
    p.IsUnitTrinomial ↔
      (p * p.mirror).coeff (((p * p.mirror).natDegree + (p * p.mirror).natTrailingDegree) / 2) =
        3 := by
  rw [natDegree_mul_mirror, natTrailingDegree_mul_mirror, ← mul_add,
    Nat.mul_div_right _ zero_lt_two, coeff_mul_mirror]
  /-
    p : Polynomial Int
    ⊢ Iff p.IsUnitTrinomial (Eq (p.sum fun x x => HPow.hPow x 2) 3)
  -/
  refine ⟨?_, fun hp => ?_⟩
    /-
      case refine_1
      p : Polynomial Int
      ⊢ p.IsUnitTrinomial → Eq (p.sum fun x x => HPow.hPow x 2) 3
    -/
  · rintro ⟨k, m, n, hkm, hmn, u, v, w, rfl⟩
    rw [sum_def, trinomial_support hkm hmn u.ne_zero v.ne_zero w.ne_zero,
      sum_insert (mt mem_insert.mp (not_or_intro hkm.ne (mt mem_singleton.mp (hkm.trans hmn).ne))),
      sum_insert (mt mem_singleton.mp hmn.ne), sum_singleton, trinomial_leading_coeff' hkm hmn,
      trinomial_middle_coeff hkm hmn, trinomial_trailing_coeff' hkm hmn]
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      ⊢ Eq (HAdd.hAdd (HPow.hPow (↑u) 2) (HAdd.hAdd (HPow.hPow (↑v) 2) (HPow.hPow (↑ …
    -/
    simp_rw [← Units.val_pow_eq_pow_val, Int.units_sq, Units.val_one]
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      ⊢ Eq (HAdd.hAdd 1 (HAdd.hAdd 1 1)) 3
    -/
    decide
    /-
      🎉 no goals
    -/
  · have key : ∀ k ∈ p.support, p.coeff k ^ 2 = 1 := fun k hk =>
      Int.sq_eq_one_of_sq_le_three
        ((single_le_sum (fun k _ => sq_nonneg (p.coeff k)) hk).trans hp.le) (mem_support_iff.mp hk)
    /-
      case refine_2
      p : Polynomial Int
      hp : Eq (p.sum fun x x => HPow.hPow x 2) 3
      key : ∀ (k : Nat), Membership.mem p.support k → Eq (HPow.hPow (p.coeff k) 2) 1
      ⊢ p.IsUnitTrinomial
    -/
    refine isUnitTrinomial_iff.mpr ⟨?_, fun k hk => isUnit_ofPowEqOne (key k hk) two_ne_zero⟩
    /-
      case refine_2
      p : Polynomial Int
      hp : Eq (p.sum fun x x => HPow.hPow x 2) 3
      key : ∀ (k : Nat), Membership.mem p.support k → Eq (HPow.hPow (p.coeff k) 2) 1
      ⊢ Eq p.support.card 3
    -/
    rw [sum_def, sum_congr rfl key, sum_const, Nat.smul_one_eq_cast] at hp
    /-
      case refine_2
      p : Polynomial Int
      hp : Eq (↑p.support.card) 3
      key : ∀ (k : Nat), Membership.mem p.support k → Eq (HPow.hPow (p.coeff k) 2) 1
      ⊢ Eq p.support.card 3
    -/
    exact Nat.cast_injective hp
    /-
      🎉 no goals
    -/


theorem isUnitTrinomial_iff'' (h : p * p.mirror = q * q.mirror) :
    p.IsUnitTrinomial ↔ q.IsUnitTrinomial := by
  /-
    p q : Polynomial Int
    h : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
    ⊢ Iff p.IsUnitTrinomial q.IsUnitTrinomial
  -/
  rw [isUnitTrinomial_iff', isUnitTrinomial_iff', h]
  /-
    🎉 no goals
  -/


theorem irreducible_aux1 {k m n : ℕ} (hkm : k < m) (hmn : m < n) (u v w : Units ℤ)
    (hp : p = trinomial k m n (u : ℤ) v w) :
    C (v : ℤ) * (C (u : ℤ) * X ^ (m + n) + C (w : ℤ) * X ^ (n - m + k + n)) =
      ⟨Finsupp.filter (· ∈ Set.Ioo (k + n) (n + n)) (p * p.mirror).toFinsupp⟩ := by
  /-
    p : Polynomial Int
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    u v w : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    ⊢ Eq (HMul.hMul (Polynomial.C ↑v) (HAdd.hAdd (HMul.hMul (Polynomial.C ↑u) (HPo …
  -/
  have key : n - m + k < n := by rwa [← lt_tsub_iff_right, tsub_lt_tsub_iff_left_of_le hmn.le]
  /-
    p : Polynomial Int
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    u v w : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
    ⊢ Eq (HMul.hMul (Polynomial.C ↑v) (HAdd.hAdd (HMul.hMul (Polynomial.C ↑u) (HPo …
  -/
  rw [hp, trinomial_mirror hkm hmn u.ne_zero w.ne_zero]
  simp_rw [trinomial_def, C_mul_X_pow_eq_monomial, add_mul, mul_add, monomial_mul_monomial,
    toFinsupp_add, toFinsupp_monomial]
  -- Porting note: added next line (less powerful `simp`).
  rw [Finsupp.filter_add, Finsupp.filter_add, Finsupp.filter_add, Finsupp.filter_add,
    Finsupp.filter_add, Finsupp.filter_add, Finsupp.filter_add, Finsupp.filter_add]
  rw [Finsupp.filter_single_of_neg, Finsupp.filter_single_of_neg, Finsupp.filter_single_of_neg,
    Finsupp.filter_single_of_neg, Finsupp.filter_single_of_neg, Finsupp.filter_single_of_pos,
    Finsupp.filter_single_of_neg, Finsupp.filter_single_of_pos, Finsupp.filter_single_of_neg]
    /-
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C ↑v) ((Polynomial.monomial (HAdd.hAdd  …
    -/
  · simp only [add_zero, zero_add, ofFinsupp_add, ofFinsupp_single]
    -- Porting note: added next two lines (less powerful `simp`).
    /-
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C ↑v) ((Polynomial.monomial (HAdd.hAdd  …
    -/
    rw [ofFinsupp_add]
    /-
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C ↑v) ((Polynomial.monomial (HAdd.hAdd  …
    -/
    simp only [ofFinsupp_single]
    /-
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C ↑v) ((Polynomial.monomial (HAdd.hAdd  …
    -/
    rw [C_mul_monomial, C_mul_monomial, mul_comm (v : ℤ) w, add_comm (n - m + k) n]
    /-
      🎉 no goals
    -/
    /-
      case h
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      ⊢ Not (Membership.mem (Set.Ioo (HAdd.hAdd k n) (HAdd.hAdd n n)) (HAdd.hAdd n n))
    -/
  · exact fun h => h.2.ne rfl
    /-
      🎉 no goals
    -/
    /-
      case h
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      ⊢ Membership.mem (Set.Ioo (HAdd.hAdd k n) (HAdd.hAdd n n)) (HAdd.hAdd n (HAdd. …
    -/
  · refine ⟨?_, add_lt_add_left key n⟩
    /-
      case h
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      ⊢ LT.lt (HAdd.hAdd k n) (HAdd.hAdd n (HAdd.hAdd (HSub.hSub n m) k))
    -/
    rwa [add_comm, add_lt_add_iff_left, lt_add_iff_pos_left, tsub_pos_iff_lt]
    /-
      🎉 no goals
    -/
    /-
      case h
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      ⊢ Not (Membership.mem (Set.Ioo (HAdd.hAdd k n) (HAdd.hAdd n n)) (HAdd.hAdd n k))
    -/
  · exact fun h => h.1.ne (add_comm k n)
    /-
      🎉 no goals
    -/
    /-
      case h
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      ⊢ Membership.mem (Set.Ioo (HAdd.hAdd k n) (HAdd.hAdd n n)) (HAdd.hAdd m n)
    -/
  · exact ⟨add_lt_add_right hkm n, add_lt_add_right hmn n⟩
    /-
      🎉 no goals
    -/
    /-
      case h
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      ⊢ Not (Membership.mem (Set.Ioo (HAdd.hAdd k n) (HAdd.hAdd n n)) (HAdd.hAdd m ( …
    -/
  · rw [← add_assoc, add_tsub_cancel_of_le hmn.le, add_comm]
    /-
      case h
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      ⊢ Not (Membership.mem (Set.Ioo (HAdd.hAdd n k) (HAdd.hAdd n n)) (HAdd.hAdd n k))
    -/
    exact fun h => h.1.ne rfl
    /-
      🎉 no goals
    -/
    /-
      case h
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      ⊢ Not (Membership.mem (Set.Ioo (HAdd.hAdd k n) (HAdd.hAdd n n)) (HAdd.hAdd m k))
    -/
  · intro h
    /-
      case h
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      h : Membership.mem (Set.Ioo (HAdd.hAdd k n) (HAdd.hAdd n n)) (HAdd.hAdd m k)
      ⊢ False
    -/
    have := h.1
    /-
      case h
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      h : Membership.mem (Set.Ioo (HAdd.hAdd k n) (HAdd.hAdd n n)) (HAdd.hAdd m k)
      this : LT.lt (HAdd.hAdd k n) (HAdd.hAdd m k)
      ⊢ False
    -/
    rw [add_comm, add_lt_add_iff_right] at this
    /-
      case h
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      h : Membership.mem (Set.Ioo (HAdd.hAdd k n) (HAdd.hAdd n n)) (HAdd.hAdd m k)
      this : LT.lt n m
      ⊢ False
    -/
    exact asymm this hmn
    /-
      🎉 no goals
    -/
    /-
      case h
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      ⊢ Not (Membership.mem (Set.Ioo (HAdd.hAdd k n) (HAdd.hAdd n n)) (HAdd.hAdd k n))
    -/
  · exact fun h => h.1.ne rfl
    /-
      🎉 no goals
    -/
    /-
      case h
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      ⊢ Not (Membership.mem (Set.Ioo (HAdd.hAdd k n) (HAdd.hAdd n n)) (HAdd.hAdd k ( …
    -/
  · exact fun h => asymm ((add_lt_add_iff_left k).mp h.1) key
    /-
      🎉 no goals
    -/
    /-
      case h
      p : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      key : LT.lt (HAdd.hAdd (HSub.hSub n m) k) n
      ⊢ Not (Membership.mem (Set.Ioo (HAdd.hAdd k n) (HAdd.hAdd n n)) (HAdd.hAdd k k))
    -/
  · exact fun h => asymm ((add_lt_add_iff_left k).mp h.1) (hkm.trans hmn)
    /-
      🎉 no goals
    -/


theorem irreducible_aux2 {k m m' n : ℕ} (hkm : k < m) (hmn : m < n) (hkm' : k < m') (hmn' : m' < n)
    (u v w : Units ℤ) (hp : p = trinomial k m n (u : ℤ) v w) (hq : q = trinomial k m' n (u : ℤ) v w)
    (h : p * p.mirror = q * q.mirror) : q = p ∨ q = p.mirror := by
  /-
    p q : Polynomial Int
    k m m' n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    hkm' : LT.lt k m'
    hmn' : LT.lt m' n
    u v w : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    hq : Eq q (Polynomial.trinomial k m' n ↑u ↑v ↑w)
    h : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
    ⊢ Or (Eq q p) (Eq q p.mirror)
  -/
  let f : ℤ[X] → ℤ[X] := fun p => ⟨Finsupp.filter (· ∈ Set.Ioo (k + n) (n + n)) p.toFinsupp⟩
  /-
    p q : Polynomial Int
    k m m' n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    hkm' : LT.lt k m'
    hmn' : LT.lt m' n
    u v w : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    hq : Eq q (Polynomial.trinomial k m' n ↑u ↑v ↑w)
    h : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
    f : Polynomial Int → Polynomial Int := fun p => { toFinsupp := Finsupp.filter  …
    ⊢ Or (Eq q p) (Eq q p.mirror)
  -/
  replace h := congr_arg f h
  /-
    p q : Polynomial Int
    k m m' n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    hkm' : LT.lt k m'
    hmn' : LT.lt m' n
    u v w : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    hq : Eq q (Polynomial.trinomial k m' n ↑u ↑v ↑w)
    f : Polynomial Int → Polynomial Int := fun p => { toFinsupp := Finsupp.filter  …
    h : Eq (f (HMul.hMul p p.mirror)) (f (HMul.hMul q q.mirror))
    ⊢ Or (Eq q p) (Eq q p.mirror)
  -/
  replace h := (irreducible_aux1 hkm hmn u v w hp).trans h
  /-
    p q : Polynomial Int
    k m m' n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    hkm' : LT.lt k m'
    hmn' : LT.lt m' n
    u v w : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    hq : Eq q (Polynomial.trinomial k m' n ↑u ↑v ↑w)
    f : Polynomial Int → Polynomial Int := fun p => { toFinsupp := Finsupp.filter  …
    h : Eq (HMul.hMul (Polynomial.C ↑v) (HAdd.hAdd (HMul.hMul (Polynomial.C ↑u) (H …
    ⊢ Or (Eq q p) (Eq q p.mirror)
  -/
  replace h := h.trans (irreducible_aux1 hkm' hmn' u v w hq).symm
  /-
    p q : Polynomial Int
    k m m' n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    hkm' : LT.lt k m'
    hmn' : LT.lt m' n
    u v w : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    hq : Eq q (Polynomial.trinomial k m' n ↑u ↑v ↑w)
    f : Polynomial Int → Polynomial Int := fun p => { toFinsupp := Finsupp.filter  …
    h : Eq (HMul.hMul (Polynomial.C ↑v) (HAdd.hAdd (HMul.hMul (Polynomial.C ↑u) (H …
    ⊢ Or (Eq q p) (Eq q p.mirror)
  -/
  rw [(isUnit_C.mpr v.isUnit).mul_right_inj] at h
  /-
    p q : Polynomial Int
    k m m' n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    hkm' : LT.lt k m'
    hmn' : LT.lt m' n
    u v w : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    hq : Eq q (Polynomial.trinomial k m' n ↑u ↑v ↑w)
    f : Polynomial Int → Polynomial Int := fun p => { toFinsupp := Finsupp.filter  …
    h : Eq (HAdd.hAdd (HMul.hMul (Polynomial.C ↑u) (HPow.hPow Polynomial.X (HAdd.h …
    ⊢ Or (Eq q p) (Eq q p.mirror)
  -/
  rw [binomial_eq_binomial u.ne_zero w.ne_zero] at h
  /-
    p q : Polynomial Int
    k m m' n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    hkm' : LT.lt k m'
    hmn' : LT.lt m' n
    u v w : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    hq : Eq q (Polynomial.trinomial k m' n ↑u ↑v ↑w)
    f : Polynomial Int → Polynomial Int := fun p => { toFinsupp := Finsupp.filter  …
    h : Or (And (Eq (HAdd.hAdd m n) (HAdd.hAdd m' n)) (Eq (HAdd.hAdd (HAdd.hAdd (H …
    ⊢ Or (Eq q p) (Eq q p.mirror)
  -/
  simp only [add_left_inj, Units.eq_iff] at h
  /-
    p q : Polynomial Int
    k m m' n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    hkm' : LT.lt k m'
    hmn' : LT.lt m' n
    u v w : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    hq : Eq q (Polynomial.trinomial k m' n ↑u ↑v ↑w)
    f : Polynomial Int → Polynomial Int := fun p => { toFinsupp := Finsupp.filter  …
    h : Or (And (Eq m m') (Eq (HSub.hSub n m) (HSub.hSub n m'))) (Or (And (Eq u w) …
    ⊢ Or (Eq q p) (Eq q p.mirror)
  -/
  rcases h with (⟨rfl, -⟩ | ⟨rfl, rfl, h⟩ | ⟨-, hm, hm'⟩)
    /-
      case inl.intro
      p q : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      f : Polynomial Int → Polynomial Int := fun p => { toFinsupp := Finsupp.filter  …
      hkm' : LT.lt k m
      hmn' : LT.lt m n
      hq : Eq q (Polynomial.trinomial k m n ↑u ↑v ↑w)
      ⊢ Or (Eq q p) (Eq q p.mirror)
    -/
  · exact Or.inl (hq.trans hp.symm)
    /-
      🎉 no goals
    -/
    /-
      case inr.inl.intro.intro
      p q : Polynomial Int
      k m' n : Nat
      hkm' : LT.lt k m'
      hmn' : LT.lt m' n
      u v : Units Int
      f : Polynomial Int → Polynomial Int := fun p => { toFinsupp := Finsupp.filter  …
      hq : Eq q (Polynomial.trinomial k m' n ↑u ↑v ↑u)
      hkm : LT.lt k (HAdd.hAdd (HSub.hSub n m') k)
      hmn : LT.lt (HAdd.hAdd (HSub.hSub n m') k) n
      hp : Eq p (Polynomial.trinomial k (HAdd.hAdd (HSub.hSub n m') k) n ↑u ↑v ↑u)
      h : Eq (HAdd.hAdd (HSub.hSub n (HAdd.hAdd (HSub.hSub n m') k)) k) m'
      ⊢ Or (Eq q p) (Eq q p.mirror)
    -/
  · refine Or.inr ?_
    /-
      case inr.inl.intro.intro
      p q : Polynomial Int
      k m' n : Nat
      hkm' : LT.lt k m'
      hmn' : LT.lt m' n
      u v : Units Int
      f : Polynomial Int → Polynomial Int := fun p => { toFinsupp := Finsupp.filter  …
      hq : Eq q (Polynomial.trinomial k m' n ↑u ↑v ↑u)
      hkm : LT.lt k (HAdd.hAdd (HSub.hSub n m') k)
      hmn : LT.lt (HAdd.hAdd (HSub.hSub n m') k) n
      hp : Eq p (Polynomial.trinomial k (HAdd.hAdd (HSub.hSub n m') k) n ↑u ↑v ↑u)
      h : Eq (HAdd.hAdd (HSub.hSub n (HAdd.hAdd (HSub.hSub n m') k)) k) m'
      ⊢ Eq q p.mirror
    -/
    rw [← trinomial_mirror hkm' hmn' u.ne_zero u.ne_zero, eq_comm, mirror_eq_iff] at hp
    /-
      case inr.inl.intro.intro
      p q : Polynomial Int
      k m' n : Nat
      hkm' : LT.lt k m'
      hmn' : LT.lt m' n
      u v : Units Int
      f : Polynomial Int → Polynomial Int := fun p => { toFinsupp := Finsupp.filter  …
      hq : Eq q (Polynomial.trinomial k m' n ↑u ↑v ↑u)
      hkm : LT.lt k (HAdd.hAdd (HSub.hSub n m') k)
      hmn : LT.lt (HAdd.hAdd (HSub.hSub n m') k) n
      hp : Eq (Polynomial.trinomial k m' n ↑u ↑v ↑u) p.mirror
      h : Eq (HAdd.hAdd (HSub.hSub n (HAdd.hAdd (HSub.hSub n m') k)) k) m'
      ⊢ Eq q p.mirror
    -/
    exact hq.trans hp
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.intro.intro
      p q : Polynomial Int
      k m m' n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      hkm' : LT.lt k m'
      hmn' : LT.lt m' n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      hq : Eq q (Polynomial.trinomial k m' n ↑u ↑v ↑w)
      f : Polynomial Int → Polynomial Int := fun p => { toFinsupp := Finsupp.filter  …
      hm : Eq m (HAdd.hAdd (HSub.hSub n m) k)
      hm' : Eq m' (HAdd.hAdd (HSub.hSub n m') k)
      ⊢ Or (Eq q p) (Eq q p.mirror)
    -/
  · obtain rfl : m = m' := by omega
    /-
      case inr.inr.intro.intro
      p q : Polynomial Int
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      f : Polynomial Int → Polynomial Int := fun p => { toFinsupp := Finsupp.filter  …
      hm : Eq m (HAdd.hAdd (HSub.hSub n m) k)
      hkm' : LT.lt k m
      hmn' : LT.lt m n
      hq : Eq q (Polynomial.trinomial k m n ↑u ↑v ↑w)
      hm' : Eq m (HAdd.hAdd (HSub.hSub n m) k)
      ⊢ Or (Eq q p) (Eq q p.mirror)
    -/
    exact Or.inl (hq.trans hp.symm)
    /-
      🎉 no goals
    -/


theorem irreducible_aux3 {k m m' n : ℕ} (hkm : k < m) (hmn : m < n) (hkm' : k < m') (hmn' : m' < n)
    (u v w x z : Units ℤ) (hp : p = trinomial k m n (u : ℤ) v w)
    (hq : q = trinomial k m' n (x : ℤ) v z) (h : p * p.mirror = q * q.mirror) :
    q = p ∨ q = p.mirror := by
  /-
    p q : Polynomial Int
    k m m' n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    hkm' : LT.lt k m'
    hmn' : LT.lt m' n
    u v w x z : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    hq : Eq q (Polynomial.trinomial k m' n ↑x ↑v ↑z)
    h : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
    ⊢ Or (Eq q p) (Eq q p.mirror)
  -/
  have hmul := congr_arg leadingCoeff h
  rw [leadingCoeff_mul, leadingCoeff_mul, mirror_leadingCoeff, mirror_leadingCoeff, hp, hq,
    trinomial_leadingCoeff hkm hmn w.ne_zero, trinomial_leadingCoeff hkm' hmn' z.ne_zero,
    trinomial_trailingCoeff hkm hmn u.ne_zero, trinomial_trailingCoeff hkm' hmn' x.ne_zero]
    at hmul
  /-
    p q : Polynomial Int
    k m m' n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    hkm' : LT.lt k m'
    hmn' : LT.lt m' n
    u v w x z : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    hq : Eq q (Polynomial.trinomial k m' n ↑x ↑v ↑z)
    h : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
    hmul : Eq (HMul.hMul ↑w ↑u) (HMul.hMul ↑z ↑x)
    ⊢ Or (Eq q p) (Eq q p.mirror)
  -/
  have hadd := congr_arg (eval 1) h
  /-
    p q : Polynomial Int
    k m m' n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    hkm' : LT.lt k m'
    hmn' : LT.lt m' n
    u v w x z : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    hq : Eq q (Polynomial.trinomial k m' n ↑x ↑v ↑z)
    h : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
    hmul : Eq (HMul.hMul ↑w ↑u) (HMul.hMul ↑z ↑x)
    hadd : Eq (Polynomial.eval 1 (HMul.hMul p p.mirror)) (Polynomial.eval 1 (HMul. …
    ⊢ Or (Eq q p) (Eq q p.mirror)
  -/
  rw [eval_mul, eval_mul, mirror_eval_one, mirror_eval_one, ← sq, ← sq, hp, hq] at hadd
  /-
    p q : Polynomial Int
    k m m' n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    hkm' : LT.lt k m'
    hmn' : LT.lt m' n
    u v w x z : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    hq : Eq q (Polynomial.trinomial k m' n ↑x ↑v ↑z)
    h : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
    hmul : Eq (HMul.hMul ↑w ↑u) (HMul.hMul ↑z ↑x)
    hadd : Eq (HPow.hPow (Polynomial.eval 1 (Polynomial.trinomial k m n ↑u ↑v ↑w)) …
    ⊢ Or (Eq q p) (Eq q p.mirror)
  -/
  simp only [eval_add, eval_C_mul, eval_pow, eval_X, one_pow, mul_one, trinomial_def] at hadd
  /-
    p q : Polynomial Int
    k m m' n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    hkm' : LT.lt k m'
    hmn' : LT.lt m' n
    u v w x z : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    hq : Eq q (Polynomial.trinomial k m' n ↑x ↑v ↑z)
    h : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
    hmul : Eq (HMul.hMul ↑w ↑u) (HMul.hMul ↑z ↑x)
    hadd : Eq (HPow.hPow (HAdd.hAdd (HAdd.hAdd ↑u ↑v) ↑w) 2) (HPow.hPow (HAdd.hAdd …
    ⊢ Or (Eq q p) (Eq q p.mirror)
  -/
  rw [add_assoc, add_assoc, add_comm (u : ℤ), add_comm (x : ℤ), add_assoc, add_assoc] at hadd
  /-
    p q : Polynomial Int
    k m m' n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    hkm' : LT.lt k m'
    hmn' : LT.lt m' n
    u v w x z : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    hq : Eq q (Polynomial.trinomial k m' n ↑x ↑v ↑z)
    h : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
    hmul : Eq (HMul.hMul ↑w ↑u) (HMul.hMul ↑z ↑x)
    hadd : Eq (HPow.hPow (HAdd.hAdd (↑v) (HAdd.hAdd ↑w ↑u)) 2) (HPow.hPow (HAdd.hA …
    ⊢ Or (Eq q p) (Eq q p.mirror)
  -/
  simp only [add_sq', add_assoc, add_right_inj, ← Units.val_pow_eq_pow_val, Int.units_sq] at hadd
  rw [mul_assoc, hmul, ← mul_assoc, add_right_inj,
    mul_right_inj' (show 2 * (v : ℤ) ≠ 0 from mul_ne_zero two_ne_zero v.ne_zero)] at hadd
  replace hadd :=
    (Int.isUnit_add_isUnit_eq_isUnit_add_isUnit w.isUnit u.isUnit z.isUnit x.isUnit).mp hadd
  /-
    p q : Polynomial Int
    k m m' n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    hkm' : LT.lt k m'
    hmn' : LT.lt m' n
    u v w x z : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    hq : Eq q (Polynomial.trinomial k m' n ↑x ↑v ↑z)
    h : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
    hmul : Eq (HMul.hMul ↑w ↑u) (HMul.hMul ↑z ↑x)
    hadd : Or (And (Eq ↑w ↑z) (Eq ↑u ↑x)) (And (Eq ↑w ↑x) (Eq ↑u ↑z))
    ⊢ Or (Eq q p) (Eq q p.mirror)
  -/
  simp only [Units.eq_iff] at hadd
  /-
    p q : Polynomial Int
    k m m' n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    hkm' : LT.lt k m'
    hmn' : LT.lt m' n
    u v w x z : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    hq : Eq q (Polynomial.trinomial k m' n ↑x ↑v ↑z)
    h : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
    hmul : Eq (HMul.hMul ↑w ↑u) (HMul.hMul ↑z ↑x)
    hadd : Or (And (Eq w z) (Eq u x)) (And (Eq w x) (Eq u z))
    ⊢ Or (Eq q p) (Eq q p.mirror)
  -/
  rcases hadd with (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩)
    /-
      case inl.intro
      p q : Polynomial Int
      k m m' n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      hkm' : LT.lt k m'
      hmn' : LT.lt m' n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      h : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
      hq : Eq q (Polynomial.trinomial k m' n ↑u ↑v ↑w)
      hmul : Eq (HMul.hMul ↑w ↑u) (HMul.hMul ↑w ↑u)
      ⊢ Or (Eq q p) (Eq q p.mirror)
    -/
  · exact irreducible_aux2 hkm hmn hkm' hmn' u v w hp hq h
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      p q : Polynomial Int
      k m m' n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      hkm' : LT.lt k m'
      hmn' : LT.lt m' n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      h : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
      hq : Eq q (Polynomial.trinomial k m' n ↑w ↑v ↑u)
      hmul : Eq (HMul.hMul ↑w ↑u) (HMul.hMul ↑u ↑w)
      ⊢ Or (Eq q p) (Eq q p.mirror)
    -/
  · rw [← mirror_inj, trinomial_mirror hkm' hmn' w.ne_zero u.ne_zero] at hq
    /-
      case inr.intro
      p q : Polynomial Int
      k m m' n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      hkm' : LT.lt k m'
      hmn' : LT.lt m' n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      h : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
      hq : Eq q.mirror (Polynomial.trinomial k (HAdd.hAdd (HSub.hSub n m') k) n ↑u ↑ …
      hmul : Eq (HMul.hMul ↑w ↑u) (HMul.hMul ↑u ↑w)
      ⊢ Or (Eq q p) (Eq q p.mirror)
    -/
    rw [mul_comm q, ← q.mirror_mirror, q.mirror.mirror_mirror] at h
    /-
      case inr.intro
      p q : Polynomial Int
      k m m' n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      hkm' : LT.lt k m'
      hmn' : LT.lt m' n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      h : Eq (HMul.hMul p p.mirror) (HMul.hMul q.mirror q.mirror.mirror)
      hq : Eq q.mirror (Polynomial.trinomial k (HAdd.hAdd (HSub.hSub n m') k) n ↑u ↑ …
      hmul : Eq (HMul.hMul ↑w ↑u) (HMul.hMul ↑u ↑w)
      ⊢ Or (Eq q p) (Eq q p.mirror)
    -/
    rw [← mirror_inj, or_comm, ← mirror_eq_iff]
    exact
      irreducible_aux2 hkm hmn (lt_add_of_pos_left k (tsub_pos_of_lt hmn'))
        (lt_tsub_iff_right.mp ((tsub_lt_tsub_iff_left_of_le hmn'.le).mpr hkm')) u v w hp hq h


theorem irreducible_of_coprime (hp : p.IsUnitTrinomial)
    (h : IsRelPrime p p.mirror) : Irreducible p := by
  /-
    p : Polynomial Int
    hp : p.IsUnitTrinomial
    h : IsRelPrime p p.mirror
    ⊢ Irreducible p
  -/
  refine irreducible_of_mirror hp.not_isUnit (fun q hpq => ?_) h
  /-
    p : Polynomial Int
    hp : p.IsUnitTrinomial
    h : IsRelPrime p p.mirror
    q : Polynomial Int
    hpq : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
    ⊢ Or (Eq q p) (Or (Eq q (Neg.neg p)) (Or (Eq q p.mirror) (Eq q (Neg.neg p.mirr …
  -/
  have hq : IsUnitTrinomial q := (isUnitTrinomial_iff'' hpq).mp hp
  /-
    p : Polynomial Int
    hp : p.IsUnitTrinomial
    h : IsRelPrime p p.mirror
    q : Polynomial Int
    hpq : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
    hq : q.IsUnitTrinomial
    ⊢ Or (Eq q p) (Or (Eq q (Neg.neg p)) (Or (Eq q p.mirror) (Eq q (Neg.neg p.mirr …
  -/
  obtain ⟨k, m, n, hkm, hmn, u, v, w, hp⟩ := hp
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    p : Polynomial Int
    h : IsRelPrime p p.mirror
    q : Polynomial Int
    hpq : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
    hq : q.IsUnitTrinomial
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    u v w : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    ⊢ Or (Eq q p) (Or (Eq q (Neg.neg p)) (Or (Eq q p.mirror) (Eq q (Neg.neg p.mirr …
  -/
  obtain ⟨k', m', n', hkm', hmn', x, y, z, hq⟩ := hq
  have hk : k = k' := by
    rw [← mul_right_inj' (show 2 ≠ 0 from two_ne_zero), ←
      trinomial_natTrailingDegree hkm hmn u.ne_zero, ← hp, ← natTrailingDegree_mul_mirror, hpq,
      natTrailingDegree_mul_mirror, hq, trinomial_natTrailingDegree hkm' hmn' x.ne_zero]
  have hn : n = n' := by
    rw [← mul_right_inj' (show 2 ≠ 0 from two_ne_zero), ← trinomial_natDegree hkm hmn w.ne_zero, ←
      hp, ← natDegree_mul_mirror, hpq, natDegree_mul_mirror, hq,
      trinomial_natDegree hkm' hmn' z.ne_zero]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    p : Polynomial Int
    h : IsRelPrime p p.mirror
    q : Polynomial Int
    hpq : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    u v w : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    k' m' n' : Nat
    hkm' : LT.lt k' m'
    hmn' : LT.lt m' n'
    x y z : Units Int
    hq : Eq q (Polynomial.trinomial k' m' n' ↑x ↑y ↑z)
    hk : Eq k k'
    hn : Eq n n'
    ⊢ Or (Eq q p) (Or (Eq q (Neg.neg p)) (Or (Eq q p.mirror) (Eq q (Neg.neg p.mirr …
  -/
  subst hk
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    p : Polynomial Int
    h : IsRelPrime p p.mirror
    q : Polynomial Int
    hpq : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    u v w : Units Int
    hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
    m' n' : Nat
    hmn' : LT.lt m' n'
    x y z : Units Int
    hn : Eq n n'
    hkm' : LT.lt k m'
    hq : Eq q (Polynomial.trinomial k m' n' ↑x ↑y ↑z)
    ⊢ Or (Eq q p) (Or (Eq q (Neg.neg p)) (Or (Eq q p.mirror) (Eq q (Neg.neg p.mirr …
  -/
  subst hn
  rcases eq_or_eq_neg_of_sq_eq_sq (y : ℤ) (v : ℤ)
      ((Int.isUnit_sq y.isUnit).trans (Int.isUnit_sq v.isUnit).symm) with
    (h1 | h1)
  · -- Porting note: `rw [h1] at *` rewrites at `h1`
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      p : Polynomial Int
      h : IsRelPrime p p.mirror
      q : Polynomial Int
      hpq : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      m' : Nat
      x y z : Units Int
      hkm' : LT.lt k m'
      hmn' : LT.lt m' n
      hq : Eq q (Polynomial.trinomial k m' n ↑x ↑y ↑z)
      h1 : Eq ↑y ↑v
      ⊢ Or (Eq q p) (Or (Eq q (Neg.neg p)) (Or (Eq q p.mirror) (Eq q (Neg.neg p.mirr …
    -/
    rw [h1] at hq
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      p : Polynomial Int
      h : IsRelPrime p p.mirror
      q : Polynomial Int
      hpq : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      m' : Nat
      x y z : Units Int
      hkm' : LT.lt k m'
      hmn' : LT.lt m' n
      hq : Eq q (Polynomial.trinomial k m' n ↑x ↑v ↑z)
      h1 : Eq ↑y ↑v
      ⊢ Or (Eq q p) (Or (Eq q (Neg.neg p)) (Or (Eq q p.mirror) (Eq q (Neg.neg p.mirr …
    -/
    rcases irreducible_aux3 hkm hmn hkm' hmn' u v w x z hp hq hpq with (h2 | h2)
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
        p : Polynomial Int
        h : IsRelPrime p p.mirror
        q : Polynomial Int
        hpq : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
        k m n : Nat
        hkm : LT.lt k m
        hmn : LT.lt m n
        u v w : Units Int
        hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
        m' : Nat
        x y z : Units Int
        hkm' : LT.lt k m'
        hmn' : LT.lt m' n
        hq : Eq q (Polynomial.trinomial k m' n ↑x ↑v ↑z)
        h1 : Eq ↑y ↑v
        h2 : Eq q p
        ⊢ Or (Eq q p) (Or (Eq q (Neg.neg p)) (Or (Eq q p.mirror) (Eq q (Neg.neg p.mirr …
      -/
    · exact Or.inl h2
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
        p : Polynomial Int
        h : IsRelPrime p p.mirror
        q : Polynomial Int
        hpq : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
        k m n : Nat
        hkm : LT.lt k m
        hmn : LT.lt m n
        u v w : Units Int
        hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
        m' : Nat
        x y z : Units Int
        hkm' : LT.lt k m'
        hmn' : LT.lt m' n
        hq : Eq q (Polynomial.trinomial k m' n ↑x ↑v ↑z)
        h1 : Eq ↑y ↑v
        h2 : Eq q p.mirror
        ⊢ Or (Eq q p) (Or (Eq q (Neg.neg p)) (Or (Eq q p.mirror) (Eq q (Neg.neg p.mirr …
      -/
    · exact Or.inr (Or.inr (Or.inl h2))
      /-
        🎉 no goals
      -/
  · -- Porting note: `rw [h1] at *` rewrites at `h1`
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      p : Polynomial Int
      h : IsRelPrime p p.mirror
      q : Polynomial Int
      hpq : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      m' : Nat
      x y z : Units Int
      hkm' : LT.lt k m'
      hmn' : LT.lt m' n
      hq : Eq q (Polynomial.trinomial k m' n ↑x ↑y ↑z)
      h1 : Eq (↑y) (Neg.neg ↑v)
      ⊢ Or (Eq q p) (Or (Eq q (Neg.neg p)) (Or (Eq q p.mirror) (Eq q (Neg.neg p.mirr …
    -/
    rw [h1] at hq
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      p : Polynomial Int
      h : IsRelPrime p p.mirror
      q : Polynomial Int
      hpq : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq p (Polynomial.trinomial k m n ↑u ↑v ↑w)
      m' : Nat
      x y z : Units Int
      hkm' : LT.lt k m'
      hmn' : LT.lt m' n
      hq : Eq q (Polynomial.trinomial k m' n (↑x) (Neg.neg ↑v) ↑z)
      h1 : Eq (↑y) (Neg.neg ↑v)
      ⊢ Or (Eq q p) (Or (Eq q (Neg.neg p)) (Or (Eq q p.mirror) (Eq q (Neg.neg p.mirr …
    -/
    rw [trinomial_def] at hp
    rw [← neg_inj, neg_add, neg_add, ← neg_mul, ← neg_mul, ← neg_mul, ← C_neg, ← C_neg, ← C_neg]
      at hp
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      p : Polynomial Int
      h : IsRelPrime p p.mirror
      q : Polynomial Int
      hpq : Eq (HMul.hMul p p.mirror) (HMul.hMul q q.mirror)
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq (Neg.neg p) (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C (Neg.neg ↑u …
      m' : Nat
      x y z : Units Int
      hkm' : LT.lt k m'
      hmn' : LT.lt m' n
      hq : Eq q (Polynomial.trinomial k m' n (↑x) (Neg.neg ↑v) ↑z)
      h1 : Eq (↑y) (Neg.neg ↑v)
      ⊢ Or (Eq q p) (Or (Eq q (Neg.neg p)) (Or (Eq q p.mirror) (Eq q (Neg.neg p.mirr …
    -/
    rw [← neg_mul_neg, ← mirror_neg] at hpq
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      p : Polynomial Int
      h : IsRelPrime p p.mirror
      q : Polynomial Int
      hpq : Eq (HMul.hMul (Neg.neg p) (Neg.neg p).mirror) (HMul.hMul q q.mirror)
      k m n : Nat
      hkm : LT.lt k m
      hmn : LT.lt m n
      u v w : Units Int
      hp : Eq (Neg.neg p) (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C (Neg.neg ↑u …
      m' : Nat
      x y z : Units Int
      hkm' : LT.lt k m'
      hmn' : LT.lt m' n
      hq : Eq q (Polynomial.trinomial k m' n (↑x) (Neg.neg ↑v) ↑z)
      h1 : Eq (↑y) (Neg.neg ↑v)
      ⊢ Or (Eq q p) (Or (Eq q (Neg.neg p)) (Or (Eq q p.mirror) (Eq q (Neg.neg p.mirr …
    -/
    rcases irreducible_aux3 hkm hmn hkm' hmn' (-u) (-v) (-w) x z hp hq hpq with (rfl | rfl)
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
        p : Polynomial Int
        h : IsRelPrime p p.mirror
        k m n : Nat
        hkm : LT.lt k m
        hmn : LT.lt m n
        u v w : Units Int
        hp : Eq (Neg.neg p) (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C (Neg.neg ↑u …
        m' : Nat
        x y z : Units Int
        hkm' : LT.lt k m'
        hmn' : LT.lt m' n
        h1 : Eq (↑y) (Neg.neg ↑v)
        hpq : Eq (HMul.hMul (Neg.neg p) (Neg.neg p).mirror) (HMul.hMul (Neg.neg p) (Ne …
        hq : Eq (Neg.neg p) (Polynomial.trinomial k m' n (↑x) (Neg.neg ↑v) ↑z)
        ⊢ Or (Eq (Neg.neg p) p) (Or (Eq (Neg.neg p) (Neg.neg p)) (Or (Eq (Neg.neg p) p …
      -/
    · exact Or.inr (Or.inl rfl)
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
        p : Polynomial Int
        h : IsRelPrime p p.mirror
        k m n : Nat
        hkm : LT.lt k m
        hmn : LT.lt m n
        u v w : Units Int
        hp : Eq (Neg.neg p) (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C (Neg.neg ↑u …
        m' : Nat
        x y z : Units Int
        hkm' : LT.lt k m'
        hmn' : LT.lt m' n
        h1 : Eq (↑y) (Neg.neg ↑v)
        hpq : Eq (HMul.hMul (Neg.neg p) (Neg.neg p).mirror) (HMul.hMul (Neg.neg p).mir …
        hq : Eq (Neg.neg p).mirror (Polynomial.trinomial k m' n (↑x) (Neg.neg ↑v) ↑z)
        ⊢ Or (Eq (Neg.neg p).mirror p) (Or (Eq (Neg.neg p).mirror (Neg.neg p)) (Or (Eq …
      -/
    · exact Or.inr (Or.inr (Or.inr p.mirror_neg))
      /-
        🎉 no goals
      -/


/-- A unit trinomial is irreducible if it is coprime with its mirror -/
theorem irreducible_of_isCoprime (hp : p.IsUnitTrinomial) (h : IsCoprime p p.mirror) :
    Irreducible p :=
  irreducible_of_coprime hp fun _ => h.isUnit_of_dvd'


