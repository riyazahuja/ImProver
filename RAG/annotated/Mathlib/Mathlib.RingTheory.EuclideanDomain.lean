theorem gcd_ne_zero_of_left (hp : p ≠ 0) : GCDMonoid.gcd p q ≠ 0 := fun h =>
  hp <| eq_zero_of_zero_dvd (h ▸ gcd_dvd_left p q)


theorem gcd_ne_zero_of_right (hp : q ≠ 0) : GCDMonoid.gcd p q ≠ 0 := fun h =>
  hp <| eq_zero_of_zero_dvd (h ▸ gcd_dvd_right p q)


theorem left_div_gcd_ne_zero {p q : R} (hp : p ≠ 0) : p / GCDMonoid.gcd p q ≠ 0 := by
  /-
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    inst✝ : GCDMonoid R
    p q : R
    hp : Ne p 0
    ⊢ Ne (HDiv.hDiv p (GCDMonoid.gcd p q)) 0
  -/
  obtain ⟨r, hr⟩ := GCDMonoid.gcd_dvd_left p q
  /-
    case intro
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    inst✝ : GCDMonoid R
    p q : R
    hp : Ne p 0
    r : R
    hr : Eq p (HMul.hMul (GCDMonoid.gcd p q) r)
    ⊢ Ne (HDiv.hDiv p (GCDMonoid.gcd p q)) 0
  -/
  obtain ⟨pq0, r0⟩ : GCDMonoid.gcd p q ≠ 0 ∧ r ≠ 0 := mul_ne_zero_iff.mp (hr ▸ hp)
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    inst✝ : GCDMonoid R
    p q : R
    hp : Ne p 0
    r : R
    hr : Eq p (HMul.hMul (GCDMonoid.gcd p q) r)
    pq0 : Ne (GCDMonoid.gcd p q) 0
    r0 : Ne r 0
    ⊢ Ne (HDiv.hDiv p (GCDMonoid.gcd p q)) 0
  -/
  nth_rw 1 [hr]
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    inst✝ : GCDMonoid R
    p q : R
    hp : Ne p 0
    r : R
    hr : Eq p (HMul.hMul (GCDMonoid.gcd p q) r)
    pq0 : Ne (GCDMonoid.gcd p q) 0
    r0 : Ne r 0
    ⊢ Ne (HDiv.hDiv (HMul.hMul (GCDMonoid.gcd p q) r) (GCDMonoid.gcd p q)) 0
  -/
  rw [mul_comm, mul_div_cancel_right₀ _ pq0]
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    inst✝ : GCDMonoid R
    p q : R
    hp : Ne p 0
    r : R
    hr : Eq p (HMul.hMul (GCDMonoid.gcd p q) r)
    pq0 : Ne (GCDMonoid.gcd p q) 0
    r0 : Ne r 0
    ⊢ Ne r 0
  -/
  exact r0
  /-
    🎉 no goals
  -/


theorem right_div_gcd_ne_zero {p q : R} (hq : q ≠ 0) : q / GCDMonoid.gcd p q ≠ 0 := by
  /-
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    inst✝ : GCDMonoid R
    p q : R
    hq : Ne q 0
    ⊢ Ne (HDiv.hDiv q (GCDMonoid.gcd p q)) 0
  -/
  obtain ⟨r, hr⟩ := GCDMonoid.gcd_dvd_right p q
  /-
    case intro
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    inst✝ : GCDMonoid R
    p q : R
    hq : Ne q 0
    r : R
    hr : Eq q (HMul.hMul (GCDMonoid.gcd p q) r)
    ⊢ Ne (HDiv.hDiv q (GCDMonoid.gcd p q)) 0
  -/
  obtain ⟨pq0, r0⟩ : GCDMonoid.gcd p q ≠ 0 ∧ r ≠ 0 := mul_ne_zero_iff.mp (hr ▸ hq)
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    inst✝ : GCDMonoid R
    p q : R
    hq : Ne q 0
    r : R
    hr : Eq q (HMul.hMul (GCDMonoid.gcd p q) r)
    pq0 : Ne (GCDMonoid.gcd p q) 0
    r0 : Ne r 0
    ⊢ Ne (HDiv.hDiv q (GCDMonoid.gcd p q)) 0
  -/
  nth_rw 1 [hr]
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    inst✝ : GCDMonoid R
    p q : R
    hq : Ne q 0
    r : R
    hr : Eq q (HMul.hMul (GCDMonoid.gcd p q) r)
    pq0 : Ne (GCDMonoid.gcd p q) 0
    r0 : Ne r 0
    ⊢ Ne (HDiv.hDiv (HMul.hMul (GCDMonoid.gcd p q) r) (GCDMonoid.gcd p q)) 0
  -/
  rw [mul_comm, mul_div_cancel_right₀ _ pq0]
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : EuclideanDomain R
    inst✝ : GCDMonoid R
    p q : R
    hq : Ne q 0
    r : R
    hr : Eq q (HMul.hMul (GCDMonoid.gcd p q) r)
    pq0 : Ne (GCDMonoid.gcd p q) 0
    r0 : Ne r 0
    ⊢ Ne r 0
  -/
  exact r0
  /-
    🎉 no goals
  -/


theorem isCoprime_div_gcd_div_gcd (hq : q ≠ 0) :
    IsCoprime (p / GCDMonoid.gcd p q) (q / GCDMonoid.gcd p q) :=
  (gcd_isUnit_iff _ _).1 <|
    isUnit_gcd_of_eq_mul_gcd
        (EuclideanDomain.mul_div_cancel' (gcd_ne_zero_of_right hq) <| gcd_dvd_left _ _).symm
        (EuclideanDomain.mul_div_cancel' (gcd_ne_zero_of_right hq) <| gcd_dvd_right _ _).symm <|
      gcd_ne_zero_of_right hq


/-- Create a `GCDMonoid` whose `GCDMonoid.gcd` matches `EuclideanDomain.gcd`. -/
-- Porting note: added `DecidableEq R`
def gcdMonoid (R) [EuclideanDomain R] [DecidableEq R] : GCDMonoid R where
  gcd := gcd
  lcm := lcm
  gcd_dvd_left := gcd_dvd_left
  gcd_dvd_right := gcd_dvd_right
  dvd_gcd := dvd_gcd
                        /-
                          R : Type ?u.8034
                          inst✝¹ : EuclideanDomain R
                          inst✝ : DecidableEq R
                          a b : R
                          ⊢ Associated (HMul.hMul (EuclideanDomain.gcd a b) (EuclideanDomain.lcm a b)) ( …
                        -/
  gcd_mul_lcm a b := by rw [EuclideanDomain.gcd_mul_lcm]
                        /-
                          🎉 no goals
                        -/
  lcm_zero_left := lcm_zero_left
  lcm_zero_right := lcm_zero_right


theorem span_gcd [DecidableEq α] (x y : α) :
    span ({gcd x y} : Set α) = span ({x, y} : Set α) :=
  letI := EuclideanDomain.gcdMonoid α
  _root_.span_gcd x y


theorem gcd_isUnit_iff [DecidableEq α] {x y : α} : IsUnit (gcd x y) ↔ IsCoprime x y :=
  letI := EuclideanDomain.gcdMonoid α
  _root_.gcd_isUnit_iff x y

-- this should be proved for UFDs surely?

theorem isCoprime_of_dvd {x y : α} (nonzero : ¬(x = 0 ∧ y = 0))
    (H : ∀ z ∈ nonunits α, z ≠ 0 → z ∣ x → ¬z ∣ y) : IsCoprime x y :=
  letI := Classical.decEq α
  letI := EuclideanDomain.gcdMonoid α
  _root_.isCoprime_of_dvd x y nonzero H

-- this should be proved for UFDs surely?

theorem dvd_or_coprime (x y : α) (h : Irreducible x) :
    x ∣ y ∨ IsCoprime x y :=
  letI := Classical.decEq α
  letI := EuclideanDomain.gcdMonoid α
  _root_.dvd_or_coprime x y h


