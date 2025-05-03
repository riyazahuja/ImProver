theorem IsLocalization.surj_of_gcd_domain [GCDMonoid R] (M : Submonoid R) [IsLocalization M A]
    (z : A) : ∃ a b : R, IsUnit (gcd a b) ∧ z * algebraMap R A b = algebraMap R A a := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : GCDMonoid R
    M : Submonoid R
    inst✝ : IsLocalization M A
    z : A
    ⊢ Exists fun a => Exists fun b => And (IsUnit (GCDMonoid.gcd a b)) (Eq (HMul.h …
  -/
  obtain ⟨x, ⟨y, hy⟩, rfl⟩ := IsLocalization.mk'_surjective M z
  /-
    case intro.intro.mk
    R : Type u_1
    A : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : GCDMonoid R
    M : Submonoid R
    inst✝ : IsLocalization M A
    x y : R
    hy : Membership.mem M y
    ⊢ Exists fun a => Exists fun b => And (IsUnit (GCDMonoid.gcd a b)) (Eq (HMul.h …
  -/
  obtain ⟨x', y', hx', hy', hu⟩ := extract_gcd x y
  /-
    case intro.intro.mk.intro.intro.intro.intro
    R : Type u_1
    A : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : GCDMonoid R
    M : Submonoid R
    inst✝ : IsLocalization M A
    x y : R
    hy : Membership.mem M y
    x' y' : R
    hx' : Eq x (HMul.hMul (GCDMonoid.gcd x y) x')
    hy' : Eq y (HMul.hMul (GCDMonoid.gcd x y) y')
    hu : IsUnit (GCDMonoid.gcd x' y')
    ⊢ Exists fun a => Exists fun b => And (IsUnit (GCDMonoid.gcd a b)) (Eq (HMul.h …
  -/
  use x', y', hu
  /-
    case right
    R : Type u_1
    A : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : GCDMonoid R
    M : Submonoid R
    inst✝ : IsLocalization M A
    x y : R
    hy : Membership.mem M y
    x' y' : R
    hx' : Eq x (HMul.hMul (GCDMonoid.gcd x y) x')
    hy' : Eq y (HMul.hMul (GCDMonoid.gcd x y) y')
    hu : IsUnit (GCDMonoid.gcd x' y')
    ⊢ Eq (HMul.hMul (IsLocalization.mk' A x ⟨y, hy⟩) ((algebraMap R A) y')) ((alge …
  -/
  rw [mul_comm, IsLocalization.mul_mk'_eq_mk'_of_mul]
  /-
    case right
    R : Type u_1
    A : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : GCDMonoid R
    M : Submonoid R
    inst✝ : IsLocalization M A
    x y : R
    hy : Membership.mem M y
    x' y' : R
    hx' : Eq x (HMul.hMul (GCDMonoid.gcd x y) x')
    hy' : Eq y (HMul.hMul (GCDMonoid.gcd x y) y')
    hu : IsUnit (GCDMonoid.gcd x' y')
    ⊢ Eq (IsLocalization.mk' A (HMul.hMul y' x) ⟨y, hy⟩) ((algebraMap R A) x')
  -/
  convert IsLocalization.mk'_mul_cancel_left (M := M) (S := A) _ _ using 2
  /-
    case h.e'_2.h.e'_8
    R : Type u_1
    A : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : GCDMonoid R
    M : Submonoid R
    inst✝ : IsLocalization M A
    x y : R
    hy : Membership.mem M y
    x' y' : R
    hx' : Eq x (HMul.hMul (GCDMonoid.gcd x y) x')
    hy' : Eq y (HMul.hMul (GCDMonoid.gcd x y) y')
    hu : IsUnit (GCDMonoid.gcd x' y')
    ⊢ Eq (HMul.hMul y' x) (HMul.hMul (↑⟨y, hy⟩) x')
  -/
  rw [Subtype.coe_mk, hy', ← mul_comm y', mul_assoc]; conv_lhs => rw [hx']
                                                      /-
                                                        🎉 no goals
                                                      -/


instance (priority := 100) GCDMonoid.toIsIntegrallyClosed
    [h : Nonempty (GCDMonoid R)] : IsIntegrallyClosed R :=
  (isIntegrallyClosed_iff (FractionRing R)).mpr fun {X} ⟨p, hp₁, hp₂⟩ => by
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      h : Nonempty (GCDMonoid R)
      X : FractionRing R
      x✝ : IsIntegral R X
      p : Polynomial R
      hp₁ : p.Monic
      hp₂ : Eq (Polynomial.eval₂ (algebraMap R (FractionRing R)) X p) 0
      ⊢ Exists fun y => Eq ((algebraMap R (FractionRing R)) y) X
    -/
    cases h
    /-
      case intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      X : FractionRing R
      x✝ : IsIntegral R X
      p : Polynomial R
      hp₁ : p.Monic
      hp₂ : Eq (Polynomial.eval₂ (algebraMap R (FractionRing R)) X p) 0
      val✝ : GCDMonoid R
      ⊢ Exists fun y => Eq ((algebraMap R (FractionRing R)) y) X
    -/
    obtain ⟨x, y, hg, he⟩ := IsLocalization.surj_of_gcd_domain (nonZeroDivisors R) X
    have :=
      Polynomial.dvd_pow_natDegree_of_eval₂_eq_zero (IsFractionRing.injective R <| FractionRing R)
        hp₁ y x _ hp₂ (by rw [mul_comm, he])
    have : IsUnit y := by
      rw [isUnit_iff_dvd_one, ← one_pow]
      exact
        (dvd_gcd this <| dvd_refl y).trans
          (gcd_pow_left_dvd_pow_gcd.trans <| pow_dvd_pow_of_dvd (isUnit_iff_dvd_one.1 hg) _)
    /-
      case intro.intro.intro.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      X : FractionRing R
      x✝ : IsIntegral R X
      p : Polynomial R
      hp₁ : p.Monic
      hp₂ : Eq (Polynomial.eval₂ (algebraMap R (FractionRing R)) X p) 0
      val✝ : GCDMonoid R
      x y : R
      hg : IsUnit (GCDMonoid.gcd x y)
      he : Eq (HMul.hMul X ((algebraMap R (FractionRing R)) y)) ((algebraMap R (Frac …
      this✝ : Dvd.dvd y (HPow.hPow x p.natDegree)
      this : IsUnit y
      ⊢ Exists fun y => Eq ((algebraMap R (FractionRing R)) y) X
    -/
    use x * (this.unit⁻¹ : _)
    /-
      case h
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      X : FractionRing R
      x✝ : IsIntegral R X
      p : Polynomial R
      hp₁ : p.Monic
      hp₂ : Eq (Polynomial.eval₂ (algebraMap R (FractionRing R)) X p) 0
      val✝ : GCDMonoid R
      x y : R
      hg : IsUnit (GCDMonoid.gcd x y)
      he : Eq (HMul.hMul X ((algebraMap R (FractionRing R)) y)) ((algebraMap R (Frac …
      this✝ : Dvd.dvd y (HPow.hPow x p.natDegree)
      this : IsUnit y
      ⊢ Eq ((algebraMap R (FractionRing R)) (HMul.hMul x ↑(Inv.inv this.unit))) X
    -/
    erw [map_mul, ← Units.coe_map_inv, eq_comm, Units.eq_mul_inv_iff_mul_eq]
    /-
      case h
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      X : FractionRing R
      x✝ : IsIntegral R X
      p : Polynomial R
      hp₁ : p.Monic
      hp₂ : Eq (Polynomial.eval₂ (algebraMap R (FractionRing R)) X p) 0
      val✝ : GCDMonoid R
      x y : R
      hg : IsUnit (GCDMonoid.gcd x y)
      he : Eq (HMul.hMul X ((algebraMap R (FractionRing R)) y)) ((algebraMap R (Frac …
      this✝ : Dvd.dvd y (HPow.hPow x p.natDegree)
      this : IsUnit y
      ⊢ Eq (HMul.hMul X ↑((Units.map ↑(algebraMap R (FractionRing R))) this.unit)) ( …
    -/
    exact he
    /-
      🎉 no goals
    -/

