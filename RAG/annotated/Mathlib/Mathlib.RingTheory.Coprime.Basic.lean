/-- The proposition that `x` and `y` are coprime, defined to be the existence of `a` and `b` such
that `a * x + b * y = 1`. Note that elements with no common divisors are not necessarily coprime,
e.g., the multivariate polynomials `x₁` and `x₂` are not coprime. -/
def IsCoprime : Prop :=
  ∃ a b, a * x + b * y = 1


@[symm]
theorem IsCoprime.symm (H : IsCoprime x y) : IsCoprime y x :=
  let ⟨a, b, H⟩ := H
            /-
              R : Type u
              inst✝ : CommSemiring R
              x y : R
              H✝ : IsCoprime x y
              a b : R
              H : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y)) 1
              ⊢ Eq (HAdd.hAdd (HMul.hMul b y) (HMul.hMul a x)) 1
            -/
  ⟨b, a, by rw [add_comm, H]⟩
            /-
              🎉 no goals
            -/


theorem isCoprime_comm : IsCoprime x y ↔ IsCoprime y x :=
  ⟨IsCoprime.symm, IsCoprime.symm⟩


theorem isCoprime_self : IsCoprime x x ↔ IsUnit x :=
                                                         /-
                                                           R : Type u
                                                           inst✝ : CommSemiring R
                                                           x : R
                                                           x✝ : IsCoprime x x
                                                           a b : R
                                                           h : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b x)) 1
                                                           ⊢ Eq (HMul.hMul x (HAdd.hAdd a b)) 1
                                                         -/
  ⟨fun ⟨a, b, h⟩ => isUnit_of_mul_eq_one x (a + b) <| by rwa [mul_comm, add_mul], fun h =>
                                                         /-
                                                           🎉 no goals
                                                         -/
    let ⟨b, hb⟩ := isUnit_iff_exists_inv'.1 h
              /-
                R : Type u
                inst✝ : CommSemiring R
                x : R
                h : IsUnit x
                b : R
                hb : Eq (HMul.hMul b x) 1
                ⊢ Eq (HAdd.hAdd (HMul.hMul b x) (HMul.hMul 0 x)) 1
              -/
    ⟨b, 0, by rwa [zero_mul, add_zero]⟩⟩
              /-
                🎉 no goals
              -/


theorem isCoprime_zero_left : IsCoprime 0 x ↔ IsUnit x :=
                                                   /-
                                                     R : Type u
                                                     inst✝ : CommSemiring R
                                                     x : R
                                                     x✝ : IsCoprime 0 x
                                                     a b : R
                                                     H : Eq (HAdd.hAdd (HMul.hMul a 0) (HMul.hMul b x)) 1
                                                     ⊢ Eq (HMul.hMul x b) 1
                                                   -/
  ⟨fun ⟨a, b, H⟩ => isUnit_of_mul_eq_one x b <| by rwa [mul_zero, zero_add, mul_comm] at H, fun H =>
                                                   /-
                                                     🎉 no goals
                                                   -/
    let ⟨b, hb⟩ := isUnit_iff_exists_inv'.1 H
              /-
                R : Type u
                inst✝ : CommSemiring R
                x : R
                H : IsUnit x
                b : R
                hb : Eq (HMul.hMul b x) 1
                ⊢ Eq (HAdd.hAdd (HMul.hMul 1 0) (HMul.hMul b x)) 1
              -/
    ⟨1, b, by rwa [one_mul, zero_add]⟩⟩
              /-
                🎉 no goals
              -/


theorem isCoprime_zero_right : IsCoprime x 0 ↔ IsUnit x :=
  isCoprime_comm.trans isCoprime_zero_left


theorem not_isCoprime_zero_zero [Nontrivial R] : ¬IsCoprime (0 : R) 0 :=
  mt isCoprime_zero_right.mp not_isUnit_zero


lemma IsCoprime.intCast {R : Type*} [CommRing R] {a b : ℤ} (h : IsCoprime a b) :
    IsCoprime (a : R) (b : R) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    a b : Int
    h : IsCoprime a b
    ⊢ IsCoprime ↑a ↑b
  -/
  rcases h with ⟨u, v, H⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    a b u v : Int
    H : Eq (HAdd.hAdd (HMul.hMul u a) (HMul.hMul v b)) 1
    ⊢ IsCoprime ↑a ↑b
  -/
  use u, v
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    a b u v : Int
    H : Eq (HAdd.hAdd (HMul.hMul u a) (HMul.hMul v b)) 1
    ⊢ Eq (HAdd.hAdd (HMul.hMul ↑u ↑a) (HMul.hMul ↑v ↑b)) 1
  -/
  rw_mod_cast [H]
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    a b u v : Int
    H : Eq (HAdd.hAdd (HMul.hMul u a) (HMul.hMul v b)) 1
    ⊢ Eq (↑1) 1
  -/
  exact Int.cast_one
  /-
    🎉 no goals
  -/


/-- If a 2-vector `p` satisfies `IsCoprime (p 0) (p 1)`, then `p ≠ 0`. -/
theorem IsCoprime.ne_zero [Nontrivial R] {p : Fin 2 → R} (h : IsCoprime (p 0) (p 1)) : p ≠ 0 := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    p : Fin 2 → R
    h : IsCoprime (p 0) (p 1)
    ⊢ Ne p 0
  -/
  rintro rfl
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    h : IsCoprime (0 0) (0 1)
    ⊢ False
  -/
  exact not_isCoprime_zero_zero h
  /-
    🎉 no goals
  -/


theorem IsCoprime.ne_zero_or_ne_zero [Nontrivial R] (h : IsCoprime x y) : x ≠ 0 ∨ y ≠ 0 := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    x y : R
    inst✝ : Nontrivial R
    h : IsCoprime x y
    ⊢ Or (Ne x 0) (Ne y 0)
  -/
  apply not_or_of_imp
  /-
    case a
    R : Type u
    inst✝¹ : CommSemiring R
    x y : R
    inst✝ : Nontrivial R
    h : IsCoprime x y
    ⊢ Eq x 0 → Ne y 0
  -/
  rintro rfl rfl
  /-
    case a
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    h : IsCoprime 0 0
    ⊢ False
  -/
  exact not_isCoprime_zero_zero h
  /-
    🎉 no goals
  -/


theorem isCoprime_one_left : IsCoprime 1 x :=
            /-
              R : Type u
              inst✝ : CommSemiring R
              x : R
              ⊢ Eq (HAdd.hAdd (HMul.hMul 1 1) (HMul.hMul 0 x)) 1
            -/
  ⟨1, 0, by rw [one_mul, zero_mul, add_zero]⟩
            /-
              🎉 no goals
            -/


theorem isCoprime_one_right : IsCoprime x 1 :=
            /-
              R : Type u
              inst✝ : CommSemiring R
              x : R
              ⊢ Eq (HAdd.hAdd (HMul.hMul 0 x) (HMul.hMul 1 1)) 1
            -/
  ⟨0, 1, by rw [one_mul, zero_mul, zero_add]⟩
            /-
              🎉 no goals
            -/


theorem IsCoprime.dvd_of_dvd_mul_right (H1 : IsCoprime x z) (H2 : x ∣ y * z) : x ∣ y := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H1 : IsCoprime x z
    H2 : Dvd.dvd x (HMul.hMul y z)
    ⊢ Dvd.dvd x y
  -/
  let ⟨a, b, H⟩ := H1
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H1 : IsCoprime x z
    H2 : Dvd.dvd x (HMul.hMul y z)
    a b : R
    H : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b z)) 1
    ⊢ Dvd.dvd x y
  -/
  rw [← mul_one y, ← H, mul_add, ← mul_assoc, mul_left_comm]
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H1 : IsCoprime x z
    H2 : Dvd.dvd x (HMul.hMul y z)
    a b : R
    H : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b z)) 1
    ⊢ Dvd.dvd x (HAdd.hAdd (HMul.hMul (HMul.hMul y a) x) (HMul.hMul b (HMul.hMul y …
  -/
  exact dvd_add (dvd_mul_left _ _) (H2.mul_left _)
  /-
    🎉 no goals
  -/


theorem IsCoprime.dvd_of_dvd_mul_left (H1 : IsCoprime x y) (H2 : x ∣ y * z) : x ∣ z := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H1 : IsCoprime x y
    H2 : Dvd.dvd x (HMul.hMul y z)
    ⊢ Dvd.dvd x z
  -/
  let ⟨a, b, H⟩ := H1
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H1 : IsCoprime x y
    H2 : Dvd.dvd x (HMul.hMul y z)
    a b : R
    H : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y)) 1
    ⊢ Dvd.dvd x z
  -/
  rw [← one_mul z, ← H, add_mul, mul_right_comm, mul_assoc b]
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H1 : IsCoprime x y
    H2 : Dvd.dvd x (HMul.hMul y z)
    a b : R
    H : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y)) 1
    ⊢ Dvd.dvd x (HAdd.hAdd (HMul.hMul (HMul.hMul a z) x) (HMul.hMul b (HMul.hMul y …
  -/
  exact dvd_add (dvd_mul_left _ _) (H2.mul_left _)
  /-
    🎉 no goals
  -/


theorem IsCoprime.mul_left (H1 : IsCoprime x z) (H2 : IsCoprime y z) : IsCoprime (x * y) z :=
  let ⟨a, b, h1⟩ := H1
  let ⟨c, d, h2⟩ := H2
  ⟨a * c, a * x * d + b * c * y + b * d * z,
    calc a * c * (x * y) + (a * x * d + b * c * y + b * d * z) * z
                                                  /-
                                                    R : Type u
                                                    inst✝ : CommSemiring R
                                                    x y z : R
                                                    H1 : IsCoprime x z
                                                    H2 : IsCoprime y z
                                                    a b : R
                                                    h1 : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b z)) 1
                                                    c d : R
                                                    h2 : Eq (HAdd.hAdd (HMul.hMul c y) (HMul.hMul d z)) 1
                                                    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul a c) (HMul.hMul x y)) (HMul.hMul (HAdd.h …
                                                  -/
      _ = (a * x + b * z) * (c * y + d * z) := by ring
                                                  /-
                                                    🎉 no goals
                                                  -/
                  /-
                    R : Type u
                    inst✝ : CommSemiring R
                    x y z : R
                    H1 : IsCoprime x z
                    H2 : IsCoprime y z
                    a b : R
                    h1 : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b z)) 1
                    c d : R
                    h2 : Eq (HAdd.hAdd (HMul.hMul c y) (HMul.hMul d z)) 1
                    ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b z)) (HAdd.hAdd (HMul.h …
                  -/
      _ = 1 := by rw [h1, h2, mul_one]
                  /-
                    🎉 no goals
                  -/
      ⟩


theorem IsCoprime.mul_right (H1 : IsCoprime x y) (H2 : IsCoprime x z) : IsCoprime x (y * z) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H1 : IsCoprime x y
    H2 : IsCoprime x z
    ⊢ IsCoprime x (HMul.hMul y z)
  -/
  rw [isCoprime_comm] at H1 H2 ⊢
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H1 : IsCoprime y x
    H2 : IsCoprime z x
    ⊢ IsCoprime (HMul.hMul y z) x
  -/
  exact H1.mul_left H2
  /-
    🎉 no goals
  -/


theorem IsCoprime.mul_dvd (H : IsCoprime x y) (H1 : x ∣ z) (H2 : y ∣ z) : x * y ∣ z := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H : IsCoprime x y
    H1 : Dvd.dvd x z
    H2 : Dvd.dvd y z
    ⊢ Dvd.dvd (HMul.hMul x y) z
  -/
  obtain ⟨a, b, h⟩ := H
  /-
    case intro.intro
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H1 : Dvd.dvd x z
    H2 : Dvd.dvd y z
    a b : R
    h : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y)) 1
    ⊢ Dvd.dvd (HMul.hMul x y) z
  -/
  rw [← mul_one z, ← h, mul_add]
  /-
    case intro.intro
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H1 : Dvd.dvd x z
    H2 : Dvd.dvd y z
    a b : R
    h : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y)) 1
    ⊢ Dvd.dvd (HMul.hMul x y) (HAdd.hAdd (HMul.hMul z (HMul.hMul a x)) (HMul.hMul  …
  -/
  apply dvd_add
    /-
      case intro.intro.h₁
      R : Type u
      inst✝ : CommSemiring R
      x y z : R
      H1 : Dvd.dvd x z
      H2 : Dvd.dvd y z
      a b : R
      h : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y)) 1
      ⊢ Dvd.dvd (HMul.hMul x y) (HMul.hMul z (HMul.hMul a x))
    -/
  · rw [mul_comm z, mul_assoc]
    /-
      case intro.intro.h₁
      R : Type u
      inst✝ : CommSemiring R
      x y z : R
      H1 : Dvd.dvd x z
      H2 : Dvd.dvd y z
      a b : R
      h : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y)) 1
      ⊢ Dvd.dvd (HMul.hMul x y) (HMul.hMul a (HMul.hMul x z))
    -/
    exact (mul_dvd_mul_left _ H2).mul_left _
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.h₂
      R : Type u
      inst✝ : CommSemiring R
      x y z : R
      H1 : Dvd.dvd x z
      H2 : Dvd.dvd y z
      a b : R
      h : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y)) 1
      ⊢ Dvd.dvd (HMul.hMul x y) (HMul.hMul z (HMul.hMul b y))
    -/
  · rw [mul_comm b, ← mul_assoc]
    /-
      case intro.intro.h₂
      R : Type u
      inst✝ : CommSemiring R
      x y z : R
      H1 : Dvd.dvd x z
      H2 : Dvd.dvd y z
      a b : R
      h : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y)) 1
      ⊢ Dvd.dvd (HMul.hMul x y) (HMul.hMul (HMul.hMul z y) b)
    -/
    exact (mul_dvd_mul_right H1 _).mul_right _
    /-
      🎉 no goals
    -/


theorem IsCoprime.of_mul_left_left (H : IsCoprime (x * y) z) : IsCoprime x z :=
  let ⟨a, b, h⟩ := H
                /-
                  R : Type u
                  inst✝ : CommSemiring R
                  x y z : R
                  H : IsCoprime (HMul.hMul x y) z
                  a b : R
                  h : Eq (HAdd.hAdd (HMul.hMul a (HMul.hMul x y)) (HMul.hMul b z)) 1
                  ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul a y) x) (HMul.hMul b z)) 1
                -/
  ⟨a * y, b, by rwa [mul_right_comm, mul_assoc]⟩
                /-
                  🎉 no goals
                -/


theorem IsCoprime.of_mul_left_right (H : IsCoprime (x * y) z) : IsCoprime y z := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H : IsCoprime (HMul.hMul x y) z
    ⊢ IsCoprime y z
  -/
  rw [mul_comm] at H
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H : IsCoprime (HMul.hMul y x) z
    ⊢ IsCoprime y z
  -/
  exact H.of_mul_left_left
  /-
    🎉 no goals
  -/


theorem IsCoprime.of_mul_right_left (H : IsCoprime x (y * z)) : IsCoprime x y := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H : IsCoprime x (HMul.hMul y z)
    ⊢ IsCoprime x y
  -/
  rw [isCoprime_comm] at H ⊢
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H : IsCoprime (HMul.hMul y z) x
    ⊢ IsCoprime y x
  -/
  exact H.of_mul_left_left
  /-
    🎉 no goals
  -/


theorem IsCoprime.of_mul_right_right (H : IsCoprime x (y * z)) : IsCoprime x z := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H : IsCoprime x (HMul.hMul y z)
    ⊢ IsCoprime x z
  -/
  rw [mul_comm] at H
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    H : IsCoprime x (HMul.hMul z y)
    ⊢ IsCoprime x z
  -/
  exact H.of_mul_right_left
  /-
    🎉 no goals
  -/


theorem IsCoprime.mul_left_iff : IsCoprime (x * y) z ↔ IsCoprime x z ∧ IsCoprime y z :=
  ⟨fun H => ⟨H.of_mul_left_left, H.of_mul_left_right⟩, fun ⟨H1, H2⟩ => H1.mul_left H2⟩


theorem IsCoprime.mul_right_iff : IsCoprime x (y * z) ↔ IsCoprime x y ∧ IsCoprime x z := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    ⊢ Iff (IsCoprime x (HMul.hMul y z)) (And (IsCoprime x y) (IsCoprime x z))
  -/
  rw [isCoprime_comm, IsCoprime.mul_left_iff, isCoprime_comm, @isCoprime_comm _ _ z]
  /-
    🎉 no goals
  -/


theorem IsCoprime.of_isCoprime_of_dvd_left (h : IsCoprime y z) (hdvd : x ∣ y) : IsCoprime x z := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsCoprime y z
    hdvd : Dvd.dvd x y
    ⊢ IsCoprime x z
  -/
  obtain ⟨d, rfl⟩ := hdvd
  /-
    case intro
    R : Type u
    inst✝ : CommSemiring R
    x z d : R
    h : IsCoprime (HMul.hMul x d) z
    ⊢ IsCoprime x z
  -/
  exact IsCoprime.of_mul_left_left h
  /-
    🎉 no goals
  -/


theorem IsCoprime.of_isCoprime_of_dvd_right (h : IsCoprime z y) (hdvd : x ∣ y) : IsCoprime z x :=
  (h.symm.of_isCoprime_of_dvd_left hdvd).symm


theorem IsCoprime.isUnit_of_dvd (H : IsCoprime x y) (d : x ∣ y) : IsUnit x :=
  let ⟨k, hk⟩ := d
  isCoprime_self.1 <| IsCoprime.of_mul_right_left <| show IsCoprime x (x * k) from hk ▸ H


theorem IsCoprime.isUnit_of_dvd' {a b x : R} (h : IsCoprime a b) (ha : x ∣ a) (hb : x ∣ b) :
    IsUnit x :=
  (h.of_isCoprime_of_dvd_left ha).isUnit_of_dvd hb


theorem IsCoprime.isRelPrime {a b : R} (h : IsCoprime a b) : IsRelPrime a b :=
  fun _ ↦ h.isUnit_of_dvd'


theorem IsCoprime.map (H : IsCoprime x y) {S : Type v} [CommSemiring S] (f : R →+* S) :
    IsCoprime (f x) (f y) :=
  let ⟨a, b, h⟩ := H
                /-
                  R : Type u
                  inst✝¹ : CommSemiring R
                  x y : R
                  H : IsCoprime x y
                  S : Type v
                  inst✝ : CommSemiring S
                  f : RingHom R S
                  a b : R
                  h : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y)) 1
                  ⊢ Eq (HAdd.hAdd (HMul.hMul (f a) (f x)) (HMul.hMul (f b) (f y))) 1
                -/
  ⟨f a, f b, by rw [← f.map_mul, ← f.map_mul, ← f.map_add, h, f.map_one]⟩
                /-
                  🎉 no goals
                -/


theorem IsCoprime.of_add_mul_left_left (h : IsCoprime (x + y * z) y) : IsCoprime x y :=
  let ⟨a, b, H⟩ := h
  ⟨a, a * z + b, by
    simpa only [add_mul, mul_add, add_assoc, add_comm, add_left_comm, mul_assoc, mul_comm,
      mul_left_comm] using H⟩


theorem IsCoprime.of_add_mul_right_left (h : IsCoprime (x + z * y) y) : IsCoprime x y := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsCoprime (HAdd.hAdd x (HMul.hMul z y)) y
    ⊢ IsCoprime x y
  -/
  rw [mul_comm] at h
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsCoprime (HAdd.hAdd x (HMul.hMul y z)) y
    ⊢ IsCoprime x y
  -/
  exact h.of_add_mul_left_left
  /-
    🎉 no goals
  -/


theorem IsCoprime.of_add_mul_left_right (h : IsCoprime x (y + x * z)) : IsCoprime x y := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsCoprime x (HAdd.hAdd y (HMul.hMul x z))
    ⊢ IsCoprime x y
  -/
  rw [isCoprime_comm] at h ⊢
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsCoprime (HAdd.hAdd y (HMul.hMul x z)) x
    ⊢ IsCoprime y x
  -/
  exact h.of_add_mul_left_left
  /-
    🎉 no goals
  -/


theorem IsCoprime.of_add_mul_right_right (h : IsCoprime x (y + z * x)) : IsCoprime x y := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsCoprime x (HAdd.hAdd y (HMul.hMul z x))
    ⊢ IsCoprime x y
  -/
  rw [mul_comm] at h
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsCoprime x (HAdd.hAdd y (HMul.hMul x z))
    ⊢ IsCoprime x y
  -/
  exact h.of_add_mul_left_right
  /-
    🎉 no goals
  -/


theorem IsCoprime.of_mul_add_left_left (h : IsCoprime (y * z + x) y) : IsCoprime x y := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsCoprime (HAdd.hAdd (HMul.hMul y z) x) y
    ⊢ IsCoprime x y
  -/
  rw [add_comm] at h
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsCoprime (HAdd.hAdd x (HMul.hMul y z)) y
    ⊢ IsCoprime x y
  -/
  exact h.of_add_mul_left_left
  /-
    🎉 no goals
  -/


theorem IsCoprime.of_mul_add_right_left (h : IsCoprime (z * y + x) y) : IsCoprime x y := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsCoprime (HAdd.hAdd (HMul.hMul z y) x) y
    ⊢ IsCoprime x y
  -/
  rw [add_comm] at h
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsCoprime (HAdd.hAdd x (HMul.hMul z y)) y
    ⊢ IsCoprime x y
  -/
  exact h.of_add_mul_right_left
  /-
    🎉 no goals
  -/


theorem IsCoprime.of_mul_add_left_right (h : IsCoprime x (x * z + y)) : IsCoprime x y := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsCoprime x (HAdd.hAdd (HMul.hMul x z) y)
    ⊢ IsCoprime x y
  -/
  rw [add_comm] at h
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsCoprime x (HAdd.hAdd y (HMul.hMul x z))
    ⊢ IsCoprime x y
  -/
  exact h.of_add_mul_left_right
  /-
    🎉 no goals
  -/


theorem IsCoprime.of_mul_add_right_right (h : IsCoprime x (z * x + y)) : IsCoprime x y := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsCoprime x (HAdd.hAdd (HMul.hMul z x) y)
    ⊢ IsCoprime x y
  -/
  rw [add_comm] at h
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsCoprime x (HAdd.hAdd y (HMul.hMul z x))
    ⊢ IsCoprime x y
  -/
  exact h.of_add_mul_right_right
  /-
    🎉 no goals
  -/


theorem IsRelPrime.of_add_mul_left_left (h : IsRelPrime (x + y * z) y) : IsRelPrime x y :=
  fun _ hx hy ↦ h (dvd_add hx <| dvd_mul_of_dvd_left hy z) hy


theorem IsRelPrime.of_add_mul_right_left (h : IsRelPrime (x + z * y) y) : IsRelPrime x y :=
  (mul_comm z y ▸ h).of_add_mul_left_left


theorem IsRelPrime.of_add_mul_left_right (h : IsRelPrime x (y + x * z)) : IsRelPrime x y := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsRelPrime x (HAdd.hAdd y (HMul.hMul x z))
    ⊢ IsRelPrime x y
  -/
  rw [isRelPrime_comm] at h ⊢
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y z : R
    h : IsRelPrime (HAdd.hAdd y (HMul.hMul x z)) x
    ⊢ IsRelPrime y x
  -/
  exact h.of_add_mul_left_left
  /-
    🎉 no goals
  -/


theorem IsRelPrime.of_add_mul_right_right (h : IsRelPrime x (y + z * x)) : IsRelPrime x y :=
  (mul_comm z x ▸ h).of_add_mul_left_right


theorem IsRelPrime.of_mul_add_left_left (h : IsRelPrime (y * z + x) y) : IsRelPrime x y :=
  (add_comm _ x ▸ h).of_add_mul_left_left


theorem IsRelPrime.of_mul_add_right_left (h : IsRelPrime (z * y + x) y) : IsRelPrime x y :=
  (add_comm _ x ▸ h).of_add_mul_right_left


theorem IsRelPrime.of_mul_add_left_right (h : IsRelPrime x (x * z + y)) : IsRelPrime x y :=
  (add_comm _ y ▸ h).of_add_mul_left_right


theorem IsRelPrime.of_mul_add_right_right (h : IsRelPrime x (z * x + y)) : IsRelPrime x y :=
  (add_comm _ y ▸ h).of_add_mul_right_right


theorem isCoprime_group_smul_left : IsCoprime (x • y) z ↔ IsCoprime y z :=
                                  /-
                                    R : Type u_1
                                    G : Type u_2
                                    inst✝⁴ : CommSemiring R
                                    inst✝³ : Group G
                                    inst✝² : MulAction G R
                                    inst✝¹ : SMulCommClass G R R
                                    inst✝ : IsScalarTower G R R
                                    x : G
                                    y z : R
                                    x✝ : IsCoprime (HSMul.hSMul x y) z
                                    a b : R
                                    h : Eq (HAdd.hAdd (HMul.hMul a (HSMul.hSMul x y)) (HMul.hMul b z)) 1
                                    ⊢ Eq (HAdd.hAdd (HMul.hMul (HSMul.hSMul x a) y) (HMul.hMul b z)) 1
                                  -/
  ⟨fun ⟨a, b, h⟩ => ⟨x • a, b, by rwa [smul_mul_assoc, ← mul_smul_comm]⟩, fun ⟨a, b, h⟩ =>
                                  /-
                                    🎉 no goals
                                  -/
                    /-
                      R : Type u_1
                      G : Type u_2
                      inst✝⁴ : CommSemiring R
                      inst✝³ : Group G
                      inst✝² : MulAction G R
                      inst✝¹ : SMulCommClass G R R
                      inst✝ : IsScalarTower G R R
                      x : G
                      y z : R
                      x✝ : IsCoprime y z
                      a b : R
                      h : Eq (HAdd.hAdd (HMul.hMul a y) (HMul.hMul b z)) 1
                      ⊢ Eq (HAdd.hAdd (HMul.hMul (HSMul.hSMul (Inv.inv x) a) (HSMul.hSMul x y)) (HMu …
                    -/
    ⟨x⁻¹ • a, b, by rwa [smul_mul_smul_comm, inv_mul_cancel, one_smul]⟩⟩
                    /-
                      🎉 no goals
                    -/


theorem isCoprime_group_smul_right : IsCoprime y (x • z) ↔ IsCoprime y z :=
  isCoprime_comm.trans <| (isCoprime_group_smul_left x z y).trans isCoprime_comm


theorem isCoprime_group_smul : IsCoprime (x • y) (x • z) ↔ IsCoprime y z :=
  (isCoprime_group_smul_left x y (x • z)).trans (isCoprime_group_smul_right x y z)


theorem isCoprime_mul_unit_left_left (hu : IsUnit x) (y z : R) :
    IsCoprime (x * y) z ↔ IsCoprime y z :=
  let ⟨u, hu⟩ := hu
  hu ▸ isCoprime_group_smul_left u y z


theorem isCoprime_mul_unit_left_right (hu : IsUnit x) (y z : R) :
    IsCoprime y (x * z) ↔ IsCoprime y z :=
  let ⟨u, hu⟩ := hu
  hu ▸ isCoprime_group_smul_right u y z


theorem isCoprime_mul_unit_right_left (hu : IsUnit x) (y z : R) :
    IsCoprime (y * x) z ↔ IsCoprime y z :=
  mul_comm x y ▸ isCoprime_mul_unit_left_left hu y z


theorem isCoprime_mul_unit_right_right (hu : IsUnit x) (y z : R) :
    IsCoprime y (z * x) ↔ IsCoprime y z :=
  mul_comm x z ▸ isCoprime_mul_unit_left_right hu y z


theorem isCoprime_mul_units_left (hu : IsUnit u) (hv : IsUnit v) (y z : R) :
    IsCoprime (u * y) (v * z) ↔ IsCoprime y z :=
  Iff.trans
    (isCoprime_mul_unit_left_left hu _ _)
    (isCoprime_mul_unit_left_right hv _ _)


theorem isCoprime_mul_units_right (hu : IsUnit u) (hv : IsUnit v) (y z : R) :
    IsCoprime (y * u) (z * v) ↔ IsCoprime y z :=
  Iff.trans
    (isCoprime_mul_unit_right_left hu _ _)
    (isCoprime_mul_unit_right_right hv _ _)


theorem isCoprime_mul_unit_left (hu : IsUnit x) (y z : R) :
    IsCoprime (x * y) (x * z) ↔ IsCoprime y z :=
  isCoprime_mul_units_left hu hu _ _


theorem isCoprime_mul_unit_right (hu : IsUnit x) (y z : R) :
    IsCoprime (y * x) (z * x) ↔ IsCoprime y z :=
  isCoprime_mul_units_right hu hu _ _


theorem add_mul_left_left {x y : R} (h : IsCoprime x y) (z : R) : IsCoprime (x + y * z) y :=
                                           /-
                                             R : Type u
                                             inst✝ : CommRing R
                                             x y : R
                                             h : IsCoprime x y
                                             z : R
                                             ⊢ IsCoprime (HAdd.hAdd (HAdd.hAdd x (HMul.hMul y z)) (HMul.hMul y (Neg.neg z)) …
                                           -/
  @of_add_mul_left_left R _ _ _ (-z) <| by simpa only [mul_neg, add_neg_cancel_right] using h
                                           /-
                                             🎉 no goals
                                           -/


theorem add_mul_right_left {x y : R} (h : IsCoprime x y) (z : R) : IsCoprime (x + z * y) y := by
  /-
    R : Type u
    inst✝ : CommRing R
    x y : R
    h : IsCoprime x y
    z : R
    ⊢ IsCoprime (HAdd.hAdd x (HMul.hMul z y)) y
  -/
  rw [mul_comm]
  /-
    R : Type u
    inst✝ : CommRing R
    x y : R
    h : IsCoprime x y
    z : R
    ⊢ IsCoprime (HAdd.hAdd x (HMul.hMul y z)) y
  -/
  exact h.add_mul_left_left z
  /-
    🎉 no goals
  -/


theorem add_mul_left_right {x y : R} (h : IsCoprime x y) (z : R) : IsCoprime x (y + x * z) := by
  /-
    R : Type u
    inst✝ : CommRing R
    x y : R
    h : IsCoprime x y
    z : R
    ⊢ IsCoprime x (HAdd.hAdd y (HMul.hMul x z))
  -/
  rw [isCoprime_comm]
  /-
    R : Type u
    inst✝ : CommRing R
    x y : R
    h : IsCoprime x y
    z : R
    ⊢ IsCoprime (HAdd.hAdd y (HMul.hMul x z)) x
  -/
  exact h.symm.add_mul_left_left z
  /-
    🎉 no goals
  -/


theorem add_mul_right_right {x y : R} (h : IsCoprime x y) (z : R) : IsCoprime x (y + z * x) := by
  /-
    R : Type u
    inst✝ : CommRing R
    x y : R
    h : IsCoprime x y
    z : R
    ⊢ IsCoprime x (HAdd.hAdd y (HMul.hMul z x))
  -/
  rw [isCoprime_comm]
  /-
    R : Type u
    inst✝ : CommRing R
    x y : R
    h : IsCoprime x y
    z : R
    ⊢ IsCoprime (HAdd.hAdd y (HMul.hMul z x)) x
  -/
  exact h.symm.add_mul_right_left z
  /-
    🎉 no goals
  -/


theorem mul_add_left_left {x y : R} (h : IsCoprime x y) (z : R) : IsCoprime (y * z + x) y := by
  /-
    R : Type u
    inst✝ : CommRing R
    x y : R
    h : IsCoprime x y
    z : R
    ⊢ IsCoprime (HAdd.hAdd (HMul.hMul y z) x) y
  -/
  rw [add_comm]
  /-
    R : Type u
    inst✝ : CommRing R
    x y : R
    h : IsCoprime x y
    z : R
    ⊢ IsCoprime (HAdd.hAdd x (HMul.hMul y z)) y
  -/
  exact h.add_mul_left_left z
  /-
    🎉 no goals
  -/


theorem mul_add_right_left {x y : R} (h : IsCoprime x y) (z : R) : IsCoprime (z * y + x) y := by
  /-
    R : Type u
    inst✝ : CommRing R
    x y : R
    h : IsCoprime x y
    z : R
    ⊢ IsCoprime (HAdd.hAdd (HMul.hMul z y) x) y
  -/
  rw [add_comm]
  /-
    R : Type u
    inst✝ : CommRing R
    x y : R
    h : IsCoprime x y
    z : R
    ⊢ IsCoprime (HAdd.hAdd x (HMul.hMul z y)) y
  -/
  exact h.add_mul_right_left z
  /-
    🎉 no goals
  -/


theorem mul_add_left_right {x y : R} (h : IsCoprime x y) (z : R) : IsCoprime x (x * z + y) := by
  /-
    R : Type u
    inst✝ : CommRing R
    x y : R
    h : IsCoprime x y
    z : R
    ⊢ IsCoprime x (HAdd.hAdd (HMul.hMul x z) y)
  -/
  rw [add_comm]
  /-
    R : Type u
    inst✝ : CommRing R
    x y : R
    h : IsCoprime x y
    z : R
    ⊢ IsCoprime x (HAdd.hAdd y (HMul.hMul x z))
  -/
  exact h.add_mul_left_right z
  /-
    🎉 no goals
  -/


theorem mul_add_right_right {x y : R} (h : IsCoprime x y) (z : R) : IsCoprime x (z * x + y) := by
  /-
    R : Type u
    inst✝ : CommRing R
    x y : R
    h : IsCoprime x y
    z : R
    ⊢ IsCoprime x (HAdd.hAdd (HMul.hMul z x) y)
  -/
  rw [add_comm]
  /-
    R : Type u
    inst✝ : CommRing R
    x y : R
    h : IsCoprime x y
    z : R
    ⊢ IsCoprime x (HAdd.hAdd y (HMul.hMul z x))
  -/
  exact h.add_mul_right_right z
  /-
    🎉 no goals
  -/


theorem add_mul_left_left_iff {x y z : R} : IsCoprime (x + y * z) y ↔ IsCoprime x y :=
  ⟨of_add_mul_left_left, fun h => h.add_mul_left_left z⟩


theorem add_mul_right_left_iff {x y z : R} : IsCoprime (x + z * y) y ↔ IsCoprime x y :=
  ⟨of_add_mul_right_left, fun h => h.add_mul_right_left z⟩


theorem add_mul_left_right_iff {x y z : R} : IsCoprime x (y + x * z) ↔ IsCoprime x y :=
  ⟨of_add_mul_left_right, fun h => h.add_mul_left_right z⟩


theorem add_mul_right_right_iff {x y z : R} : IsCoprime x (y + z * x) ↔ IsCoprime x y :=
  ⟨of_add_mul_right_right, fun h => h.add_mul_right_right z⟩


theorem mul_add_left_left_iff {x y z : R} : IsCoprime (y * z + x) y ↔ IsCoprime x y :=
  ⟨of_mul_add_left_left, fun h => h.mul_add_left_left z⟩


theorem mul_add_right_left_iff {x y z : R} : IsCoprime (z * y + x) y ↔ IsCoprime x y :=
  ⟨of_mul_add_right_left, fun h => h.mul_add_right_left z⟩


theorem mul_add_left_right_iff {x y z : R} : IsCoprime x (x * z + y) ↔ IsCoprime x y :=
  ⟨of_mul_add_left_right, fun h => h.mul_add_left_right z⟩


theorem mul_add_right_right_iff {x y z : R} : IsCoprime x (z * x + y) ↔ IsCoprime x y :=
  ⟨of_mul_add_right_right, fun h => h.mul_add_right_right z⟩


theorem neg_left {x y : R} (h : IsCoprime x y) : IsCoprime (-x) y := by
  /-
    R : Type u
    inst✝ : CommRing R
    x y : R
    h : IsCoprime x y
    ⊢ IsCoprime (Neg.neg x) y
  -/
  obtain ⟨a, b, h⟩ := h
  /-
    case intro.intro
    R : Type u
    inst✝ : CommRing R
    x y a b : R
    h : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y)) 1
    ⊢ IsCoprime (Neg.neg x) y
  -/
  use -a, b
  /-
    case h
    R : Type u
    inst✝ : CommRing R
    x y a b : R
    h : Eq (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y)) 1
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Neg.neg a) (Neg.neg x)) (HMul.hMul b y)) 1
  -/
  rwa [neg_mul_neg]
  /-
    🎉 no goals
  -/


theorem neg_left_iff (x y : R) : IsCoprime (-x) y ↔ IsCoprime x y :=
  ⟨fun h => neg_neg x ▸ h.neg_left, neg_left⟩


theorem neg_right {x y : R} (h : IsCoprime x y) : IsCoprime x (-y) :=
  h.symm.neg_left.symm


theorem neg_right_iff (x y : R) : IsCoprime x (-y) ↔ IsCoprime x y :=
  ⟨fun h => neg_neg y ▸ h.neg_right, neg_right⟩


theorem neg_neg {x y : R} (h : IsCoprime x y) : IsCoprime (-x) (-y) :=
  h.neg_left.neg_right


theorem neg_neg_iff (x y : R) : IsCoprime (-x) (-y) ↔ IsCoprime x y :=
  (neg_left_iff _ _).trans (neg_right_iff _ _)


theorem sq_add_sq_ne_zero {R : Type*} [LinearOrderedCommRing R] {a b : R} (h : IsCoprime a b) :
    a ^ 2 + b ^ 2 ≠ 0 := by
  /-
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    a b : R
    h : IsCoprime a b
    ⊢ Ne (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2)) 0
  -/
  intro h'
  /-
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    a b : R
    h : IsCoprime a b
    h' : Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2)) 0
    ⊢ False
  -/
  obtain ⟨ha, hb⟩ := (add_eq_zero_iff_of_nonneg (sq_nonneg _) (sq_nonneg _)).mp h'
  /-
    case intro
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    a b : R
    h : IsCoprime a b
    h' : Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2)) 0
    ha : Eq (HPow.hPow a 2) 0
    hb : Eq (HPow.hPow b 2) 0
    ⊢ False
  -/
  obtain rfl := pow_eq_zero ha
  /-
    case intro
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    b : R
    hb : Eq (HPow.hPow b 2) 0
    h : IsCoprime 0 b
    h' : Eq (HAdd.hAdd (HPow.hPow 0 2) (HPow.hPow b 2)) 0
    ha : Eq (HPow.hPow 0 2) 0
    ⊢ False
  -/
  obtain rfl := pow_eq_zero hb
  /-
    case intro
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    ha hb : Eq (HPow.hPow 0 2) 0
    h : IsCoprime 0 0
    h' : Eq (HAdd.hAdd (HPow.hPow 0 2) (HPow.hPow 0 2)) 0
    ⊢ False
  -/
  exact not_isCoprime_zero_zero h
  /-
    🎉 no goals
  -/


theorem add_mul_left_left (h : IsRelPrime x y) (z : R) : IsRelPrime (x + y * z) y :=
                                           /-
                                             R : Type u_1
                                             inst✝ : CommRing R
                                             x y : R
                                             h : IsRelPrime x y
                                             z : R
                                             ⊢ IsRelPrime (HAdd.hAdd (HAdd.hAdd x (HMul.hMul y z)) (HMul.hMul y (Neg.neg z) …
                                           -/
  @of_add_mul_left_left R _ _ _ (-z) <| by simpa only [mul_neg, add_neg_cancel_right] using h
                                           /-
                                             🎉 no goals
                                           -/


theorem add_mul_right_left (h : IsRelPrime x y) (z : R) : IsRelPrime (x + z * y) y :=
  mul_comm z y ▸ h.add_mul_left_left z


theorem add_mul_left_right (h : IsRelPrime x y) (z : R) : IsRelPrime x (y + x * z) :=
  (h.symm.add_mul_left_left z).symm


theorem add_mul_right_right (h : IsRelPrime x y) (z : R) : IsRelPrime x (y + z * x) :=
  (h.symm.add_mul_right_left z).symm


theorem mul_add_left_left (h : IsRelPrime x y) (z : R) : IsRelPrime (y * z + x) y :=
  add_comm x _ ▸ h.add_mul_left_left z


theorem mul_add_right_left (h : IsRelPrime x y) (z : R) : IsRelPrime (z * y + x) y :=
  add_comm x _ ▸ h.add_mul_right_left z


theorem mul_add_left_right (h : IsRelPrime x y) (z : R) : IsRelPrime x (x * z + y) :=
  add_comm y _ ▸ h.add_mul_left_right z


theorem mul_add_right_right (h : IsRelPrime x y) (z : R) : IsRelPrime x (z * x + y) :=
  add_comm y _ ▸ h.add_mul_right_right z


theorem add_mul_left_left_iff : IsRelPrime (x + y * z) y ↔ IsRelPrime x y :=
  ⟨of_add_mul_left_left, fun h ↦ h.add_mul_left_left z⟩


theorem add_mul_right_left_iff : IsRelPrime (x + z * y) y ↔ IsRelPrime x y :=
  ⟨of_add_mul_right_left, fun h ↦ h.add_mul_right_left z⟩


theorem add_mul_left_right_iff : IsRelPrime x (y + x * z) ↔ IsRelPrime x y :=
  ⟨of_add_mul_left_right, fun h ↦ h.add_mul_left_right z⟩


theorem add_mul_right_right_iff : IsRelPrime x (y + z * x) ↔ IsRelPrime x y :=
  ⟨of_add_mul_right_right, fun h ↦ h.add_mul_right_right z⟩


theorem mul_add_left_left_iff {x y z : R} : IsRelPrime (y * z + x) y ↔ IsRelPrime x y :=
  ⟨of_mul_add_left_left, fun h ↦ h.mul_add_left_left z⟩


theorem mul_add_right_left_iff {x y z : R} : IsRelPrime (z * y + x) y ↔ IsRelPrime x y :=
  ⟨of_mul_add_right_left, fun h ↦ h.mul_add_right_left z⟩


theorem mul_add_left_right_iff {x y z : R} : IsRelPrime x (x * z + y) ↔ IsRelPrime x y :=
  ⟨of_mul_add_left_right, fun h ↦ h.mul_add_left_right z⟩


theorem mul_add_right_right_iff {x y z : R} : IsRelPrime x (z * x + y) ↔ IsRelPrime x y :=
  ⟨of_mul_add_right_right, fun h ↦ h.mul_add_right_right z⟩


theorem neg_left (h : IsRelPrime x y) : IsRelPrime (-x) y := fun _ ↦ (h <| dvd_neg.mp ·)

theorem neg_right (h : IsRelPrime x y) : IsRelPrime x (-y) := h.symm.neg_left.symm

protected theorem neg_neg (h : IsRelPrime x y) : IsRelPrime (-x) (-y) := h.neg_left.neg_right


theorem neg_left_iff (x y : R) : IsRelPrime (-x) y ↔ IsRelPrime x y :=
  ⟨fun h ↦ neg_neg x ▸ h.neg_left, neg_left⟩


theorem neg_right_iff (x y : R) : IsRelPrime x (-y) ↔ IsRelPrime x y :=
  ⟨fun h ↦ neg_neg y ▸ h.neg_right, neg_right⟩


theorem neg_neg_iff (x y : R) : IsRelPrime (-x) (-y) ↔ IsRelPrime x y :=
  (neg_left_iff _ _).trans (neg_right_iff _ _)


