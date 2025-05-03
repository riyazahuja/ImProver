theorem exists_reduced_fraction (x : K) :
    ∃ (a : A) (b : nonZeroDivisors A), IsRelPrime a b ∧ mk' K a b = x := by
  /-
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    ⊢ Exists fun a => Exists fun b => And (IsRelPrime a ↑b) (Eq (IsLocalization.mk …
  -/
  obtain ⟨⟨b, b_nonzero⟩, a, hab⟩ := exists_integer_multiple (nonZeroDivisors A) x
  obtain ⟨a', b', c', no_factor, rfl, rfl⟩ :=
    UniqueFactorizationMonoid.exists_reduced_factors' a b
      (mem_nonZeroDivisors_iff_ne_zero.mp b_nonzero)
  /-
    case intro.mk.intro.intro.intro.intro.intro.intro
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    a' b' c' : A
    no_factor : IsRelPrime a' b'
    b_nonzero : Membership.mem (nonZeroDivisors A) (HMul.hMul c' b')
    hab : Eq ((algebraMap A K) (HMul.hMul c' a')) (HSMul.hSMul (↑⟨HMul.hMul c' b', …
    ⊢ Exists fun a => Exists fun b => And (IsRelPrime a ↑b) (Eq (IsLocalization.mk …
  -/
  obtain ⟨_, b'_nonzero⟩ := mul_mem_nonZeroDivisors.mp b_nonzero
  /-
    case intro.mk.intro.intro.intro.intro.intro.intro.intro
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    a' b' c' : A
    no_factor : IsRelPrime a' b'
    b_nonzero : Membership.mem (nonZeroDivisors A) (HMul.hMul c' b')
    hab : Eq ((algebraMap A K) (HMul.hMul c' a')) (HSMul.hSMul (↑⟨HMul.hMul c' b', …
    left✝ : Membership.mem (nonZeroDivisors A) c'
    b'_nonzero : Membership.mem (nonZeroDivisors A) b'
    ⊢ Exists fun a => Exists fun b => And (IsRelPrime a ↑b) (Eq (IsLocalization.mk …
  -/
  refine ⟨a', ⟨b', b'_nonzero⟩, no_factor, ?_⟩
  /-
    case intro.mk.intro.intro.intro.intro.intro.intro.intro
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    a' b' c' : A
    no_factor : IsRelPrime a' b'
    b_nonzero : Membership.mem (nonZeroDivisors A) (HMul.hMul c' b')
    hab : Eq ((algebraMap A K) (HMul.hMul c' a')) (HSMul.hSMul (↑⟨HMul.hMul c' b', …
    left✝ : Membership.mem (nonZeroDivisors A) c'
    b'_nonzero : Membership.mem (nonZeroDivisors A) b'
    ⊢ Eq (IsLocalization.mk' K a' ⟨b', b'_nonzero⟩) x
  -/
  refine mul_left_cancel₀ (IsFractionRing.to_map_ne_zero_of_mem_nonZeroDivisors b_nonzero) ?_
  /-
    case intro.mk.intro.intro.intro.intro.intro.intro.intro
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    a' b' c' : A
    no_factor : IsRelPrime a' b'
    b_nonzero : Membership.mem (nonZeroDivisors A) (HMul.hMul c' b')
    hab : Eq ((algebraMap A K) (HMul.hMul c' a')) (HSMul.hSMul (↑⟨HMul.hMul c' b', …
    left✝ : Membership.mem (nonZeroDivisors A) c'
    b'_nonzero : Membership.mem (nonZeroDivisors A) b'
    ⊢ Eq (HMul.hMul ((algebraMap A K) (HMul.hMul c' b')) (IsLocalization.mk' K a'  …
  -/
  simp only [Subtype.coe_mk, RingHom.map_mul, Algebra.smul_def] at *
  /-
    case intro.mk.intro.intro.intro.intro.intro.intro.intro
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    a' b' c' : A
    no_factor : IsRelPrime a' b'
    b_nonzero : Membership.mem (nonZeroDivisors A) (HMul.hMul c' b')
    left✝ : Membership.mem (nonZeroDivisors A) c'
    b'_nonzero : Membership.mem (nonZeroDivisors A) b'
    hab : Eq (HMul.hMul ((algebraMap A K) c') ((algebraMap A K) a')) (HMul.hMul (H …
    ⊢ Eq (HMul.hMul (HMul.hMul ((algebraMap A K) c') ((algebraMap A K) b')) (IsLoc …
  -/
  rw [← hab, mul_assoc, mk'_spec' _ a' ⟨b', b'_nonzero⟩]
  /-
    🎉 no goals
  -/


/-- `f.num x` is the numerator of `x : f.codomain` as a reduced fraction. -/
noncomputable def num (x : K) : A :=
  Classical.choose (exists_reduced_fraction A x)


/-- `f.den x` is the denominator of `x : f.codomain` as a reduced fraction. -/
noncomputable def den (x : K) : nonZeroDivisors A :=
  Classical.choose (Classical.choose_spec (exists_reduced_fraction A x))


theorem num_den_reduced (x : K) : IsRelPrime (num A x) (den A x) :=
  (Classical.choose_spec (Classical.choose_spec (exists_reduced_fraction A x))).1

-- @[simp] -- Porting note: LHS reduces to give the simp lemma below

theorem mk'_num_den (x : K) : mk' K (num A x) (den A x) = x :=
  (Classical.choose_spec (Classical.choose_spec (exists_reduced_fraction A x))).2


@[simp]
theorem mk'_num_den' (x : K) : algebraMap A K (num A x) / algebraMap A K (den A x) = x := by
  /-
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    ⊢ Eq (HDiv.hDiv ((algebraMap A K) (IsFractionRing.num A x)) ((algebraMap A K)  …
  -/
  rw [← mk'_eq_div]
  /-
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    ⊢ Eq (IsLocalization.mk' K (IsFractionRing.num A x) (IsFractionRing.den A x)) x
  -/
  apply mk'_num_den
  /-
    🎉 no goals
  -/


theorem num_mul_den_eq_num_iff_eq {x y : K} :
    x * algebraMap A K (den A y) = algebraMap A K (num A y) ↔ x = y :=
               /-
                 A : Type u_1
                 inst✝⁵ : CommRing A
                 inst✝⁴ : IsDomain A
                 inst✝³ : UniqueFactorizationMonoid A
                 K : Type u_2
                 inst✝² : Field K
                 inst✝¹ : Algebra A K
                 inst✝ : IsFractionRing A K
                 x y : K
                 h : Eq (HMul.hMul x ((algebraMap A K) ↑(IsFractionRing.den A y))) ((algebraMap …
                 ⊢ Eq x y
               -/
  ⟨fun h => by simpa only [mk'_num_den] using eq_mk'_iff_mul_eq.mpr h, fun h ↦
               /-
                 🎉 no goals
               -/
                             /-
                               A : Type u_1
                               inst✝⁵ : CommRing A
                               inst✝⁴ : IsDomain A
                               inst✝³ : UniqueFactorizationMonoid A
                               K : Type u_2
                               inst✝² : Field K
                               inst✝¹ : Algebra A K
                               inst✝ : IsFractionRing A K
                               x y : K
                               h : Eq x y
                               ⊢ Eq x (IsLocalization.mk' K (IsFractionRing.num A y) (IsFractionRing.den A y))
                             -/
    eq_mk'_iff_mul_eq.mp (by rw [h, mk'_num_den])⟩
                             /-
                               🎉 no goals
                             -/


theorem num_mul_den_eq_num_iff_eq' {x y : K} :
    y * algebraMap A K (den A x) = algebraMap A K (num A x) ↔ x = y :=
              /-
                A : Type u_1
                inst✝⁵ : CommRing A
                inst✝⁴ : IsDomain A
                inst✝³ : UniqueFactorizationMonoid A
                K : Type u_2
                inst✝² : Field K
                inst✝¹ : Algebra A K
                inst✝ : IsFractionRing A K
                x y : K
                h : Eq (HMul.hMul y ((algebraMap A K) ↑(IsFractionRing.den A x))) ((algebraMap …
                ⊢ Eq x y
              -/
  ⟨fun h ↦ by simpa only [eq_comm, mk'_num_den] using eq_mk'_iff_mul_eq.mpr h, fun h ↦
              /-
                🎉 no goals
              -/
                             /-
                               A : Type u_1
                               inst✝⁵ : CommRing A
                               inst✝⁴ : IsDomain A
                               inst✝³ : UniqueFactorizationMonoid A
                               K : Type u_2
                               inst✝² : Field K
                               inst✝¹ : Algebra A K
                               inst✝ : IsFractionRing A K
                               x y : K
                               h : Eq x y
                               ⊢ Eq y (IsLocalization.mk' K (IsFractionRing.num A x) (IsFractionRing.den A x))
                             -/
    eq_mk'_iff_mul_eq.mp (by rw [h, mk'_num_den])⟩
                             /-
                               🎉 no goals
                             -/


theorem num_mul_den_eq_num_mul_den_iff_eq {x y : K} :
    num A y * den A x = num A x * den A y ↔ x = y :=
              /-
                A : Type u_1
                inst✝⁵ : CommRing A
                inst✝⁴ : IsDomain A
                inst✝³ : UniqueFactorizationMonoid A
                K : Type u_2
                inst✝² : Field K
                inst✝¹ : Algebra A K
                inst✝ : IsFractionRing A K
                x y : K
                h : Eq (HMul.hMul (IsFractionRing.num A y) ↑(IsFractionRing.den A x)) (HMul.hM …
                ⊢ Eq x y
              -/
              /-
                🎉 no goals
              -/
  ⟨fun h ↦ by simpa only [mk'_num_den] using mk'_eq_of_eq' (S := K) h, fun h ↦ by rw [h]⟩
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem eq_zero_of_num_eq_zero {x : K} (h : num A x = 0) : x = 0 :=
                                               /-
                                                 A : Type u_1
                                                 inst✝⁵ : CommRing A
                                                 inst✝⁴ : IsDomain A
                                                 inst✝³ : UniqueFactorizationMonoid A
                                                 K : Type u_2
                                                 inst✝² : Field K
                                                 inst✝¹ : Algebra A K
                                                 inst✝ : IsFractionRing A K
                                                 x : K
                                                 h : Eq (IsFractionRing.num A x) 0
                                                 ⊢ Eq (HMul.hMul 0 ((algebraMap A K) ↑(IsFractionRing.den A x))) ((algebraMap A …
                                               -/
  (num_mul_den_eq_num_iff_eq' (A := A)).mp (by rw [zero_mul, h, RingHom.map_zero])
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
lemma num_zero : IsFractionRing.num A (0 : K) = 0 := by
  /-
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    ⊢ Eq (IsFractionRing.num A 0) 0
  -/
  have := mk'_num_den' A (0 : K)
  /-
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    this : Eq (HDiv.hDiv ((algebraMap A K) (IsFractionRing.num A 0)) ((algebraMap  …
    ⊢ Eq (IsFractionRing.num A 0) 0
  -/
  simp only [div_eq_zero_iff] at this
  /-
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    this : Or (Eq ((algebraMap A K) (IsFractionRing.num A 0)) 0) (Eq ((algebraMap  …
    ⊢ Eq (IsFractionRing.num A 0) 0
  -/
  rcases this with h | h
    /-
      case inl
      A : Type u_1
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      h : Eq ((algebraMap A K) (IsFractionRing.num A 0)) 0
      ⊢ Eq (IsFractionRing.num A 0) 0
    -/
  · exact NoZeroSMulDivisors.algebraMap_injective A K (by convert h; simp)
    /-
      🎉 no goals
    -/
    /-
      case inr
      A : Type u_1
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      h : Eq ((algebraMap A K) ↑(IsFractionRing.den A 0)) 0
      ⊢ Eq (IsFractionRing.num A 0) 0
    -/
  · replace h : algebraMap A K (den A (0 : K)) = algebraMap A K 0 := by convert h; simp
    /-
      case inr
      A : Type u_1
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      h : Eq ((algebraMap A K) ↑(IsFractionRing.den A 0)) ((algebraMap A K) 0)
      ⊢ Eq (IsFractionRing.num A 0) 0
    -/
    absurd NoZeroSMulDivisors.algebraMap_injective A K h
    /-
      case inr
      A : Type u_1
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      h : Eq ((algebraMap A K) ↑(IsFractionRing.den A 0)) ((algebraMap A K) 0)
      ⊢ Not (Eq (↑(IsFractionRing.den A 0)) 0)
    -/
    apply nonZeroDivisors.coe_ne_zero
    /-
      🎉 no goals
    -/


@[simp]
lemma num_eq_zero (x : K) : IsFractionRing.num A x = 0 ↔ x = 0 :=
  ⟨eq_zero_of_num_eq_zero, fun h ↦ h ▸ num_zero⟩


theorem isInteger_of_isUnit_den {x : K} (h : IsUnit (den A x : A)) : IsInteger A x := by
  /-
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    h : IsUnit ↑(IsFractionRing.den A x)
    ⊢ IsLocalization.IsInteger A x
  -/
  cases' h with d hd
  have d_ne_zero : algebraMap A K (den A x) ≠ 0 :=
    IsFractionRing.to_map_ne_zero_of_mem_nonZeroDivisors (den A x).2
  /-
    case intro
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    d : Units A
    hd : Eq ↑d ↑(IsFractionRing.den A x)
    d_ne_zero : Ne ((algebraMap A K) ↑(IsFractionRing.den A x)) 0
    ⊢ IsLocalization.IsInteger A x
  -/
  use ↑d⁻¹ * num A x
  /-
    case h
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    d : Units A
    hd : Eq ↑d ↑(IsFractionRing.den A x)
    d_ne_zero : Ne ((algebraMap A K) ↑(IsFractionRing.den A x)) 0
    ⊢ Eq ((algebraMap A K) (HMul.hMul (↑(Inv.inv d)) (IsFractionRing.num A x))) x
  -/
  refine _root_.trans ?_ (mk'_num_den A x)
  /-
    case h
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    d : Units A
    hd : Eq ↑d ↑(IsFractionRing.den A x)
    d_ne_zero : Ne ((algebraMap A K) ↑(IsFractionRing.den A x)) 0
    ⊢ Eq ((algebraMap A K) (HMul.hMul (↑(Inv.inv d)) (IsFractionRing.num A x))) (I …
  -/
  rw [map_mul, map_units_inv, hd]
  /-
    case h
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    d : Units A
    hd : Eq ↑d ↑(IsFractionRing.den A x)
    d_ne_zero : Ne ((algebraMap A K) ↑(IsFractionRing.den A x)) 0
    ⊢ Eq (HMul.hMul (Inv.inv ((algebraMap A K) ↑(IsFractionRing.den A x))) ((algeb …
  -/
  apply mul_left_cancel₀ d_ne_zero
  /-
    case h
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    d : Units A
    hd : Eq ↑d ↑(IsFractionRing.den A x)
    d_ne_zero : Ne ((algebraMap A K) ↑(IsFractionRing.den A x)) 0
    ⊢ Eq (HMul.hMul ((algebraMap A K) ↑(IsFractionRing.den A x)) (HMul.hMul (Inv.i …
  -/
  rw [← mul_assoc, mul_inv_cancel₀ d_ne_zero, one_mul, mk'_spec']
  /-
    🎉 no goals
  -/


theorem isUnit_den_iff (x : K) : IsUnit (den A x : A) ↔ IsLocalization.IsInteger A x where
  mp := isInteger_of_isUnit_den
  mpr h := by
    /-
      A : Type u_1
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      x : K
      h : IsLocalization.IsInteger A x
      ⊢ IsUnit ↑(IsFractionRing.den A x)
    -/
    have ⟨v, h⟩ := h
    /-
      A : Type u_1
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      x : K
      h✝ : IsLocalization.IsInteger A x
      v : A
      h : Eq ((algebraMap A K) v) x
      ⊢ IsUnit ↑(IsFractionRing.den A x)
    -/
    apply IsRelPrime.isUnit_of_dvd (num_den_reduced A x).symm
    /-
      A : Type u_1
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      x : K
      h✝ : IsLocalization.IsInteger A x
      v : A
      h : Eq ((algebraMap A K) v) x
      ⊢ Dvd.dvd (↑(IsFractionRing.den A x)) (IsFractionRing.num A x)
    -/
    use v
    /-
      case h
      A : Type u_1
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      x : K
      h✝ : IsLocalization.IsInteger A x
      v : A
      h : Eq ((algebraMap A K) v) x
      ⊢ Eq (IsFractionRing.num A x) (HMul.hMul (↑(IsFractionRing.den A x)) v)
    -/
    apply_fun algebraMap A K
      /-
        case h
        A : Type u_1
        inst✝⁵ : CommRing A
        inst✝⁴ : IsDomain A
        inst✝³ : UniqueFactorizationMonoid A
        K : Type u_2
        inst✝² : Field K
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        x : K
        h✝ : IsLocalization.IsInteger A x
        v : A
        h : Eq ((algebraMap A K) v) x
        ⊢ Eq ((algebraMap A K) (IsFractionRing.num A x)) ((algebraMap A K) (HMul.hMul  …
      -/
    · simp only [map_mul, h]
      /-
        case h
        A : Type u_1
        inst✝⁵ : CommRing A
        inst✝⁴ : IsDomain A
        inst✝³ : UniqueFactorizationMonoid A
        K : Type u_2
        inst✝² : Field K
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        x : K
        h✝ : IsLocalization.IsInteger A x
        v : A
        h : Eq ((algebraMap A K) v) x
        ⊢ Eq ((algebraMap A K) (IsFractionRing.num A x)) (HMul.hMul ((algebraMap A K)  …
      -/
      rw [mul_comm, ← div_eq_iff]
        /-
          case h
          A : Type u_1
          inst✝⁵ : CommRing A
          inst✝⁴ : IsDomain A
          inst✝³ : UniqueFactorizationMonoid A
          K : Type u_2
          inst✝² : Field K
          inst✝¹ : Algebra A K
          inst✝ : IsFractionRing A K
          x : K
          h✝ : IsLocalization.IsInteger A x
          v : A
          h : Eq ((algebraMap A K) v) x
          ⊢ Eq (HDiv.hDiv ((algebraMap A K) (IsFractionRing.num A x)) ((algebraMap A K)  …
        -/
      · simp only [mk'_num_den']
        /-
          🎉 no goals
        -/
      /-
        case h
        A : Type u_1
        inst✝⁵ : CommRing A
        inst✝⁴ : IsDomain A
        inst✝³ : UniqueFactorizationMonoid A
        K : Type u_2
        inst✝² : Field K
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        x : K
        h✝ : IsLocalization.IsInteger A x
        v : A
        h : Eq ((algebraMap A K) v) x
        ⊢ Ne ((algebraMap A K) ↑(IsFractionRing.den A x)) 0
      -/
      intro h
      /-
        case h
        A : Type u_1
        inst✝⁵ : CommRing A
        inst✝⁴ : IsDomain A
        inst✝³ : UniqueFactorizationMonoid A
        K : Type u_2
        inst✝² : Field K
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        x : K
        h✝¹ : IsLocalization.IsInteger A x
        v : A
        h✝ : Eq ((algebraMap A K) v) x
        h : Eq ((algebraMap A K) ↑(IsFractionRing.den A x)) 0
        ⊢ False
      -/
      replace h : algebraMap A K (den A x : A) = algebraMap A K 0 := by convert h; simp
      /-
        case h
        A : Type u_1
        inst✝⁵ : CommRing A
        inst✝⁴ : IsDomain A
        inst✝³ : UniqueFactorizationMonoid A
        K : Type u_2
        inst✝² : Field K
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        x : K
        h✝¹ : IsLocalization.IsInteger A x
        v : A
        h✝ : Eq ((algebraMap A K) v) x
        h : Eq ((algebraMap A K) ↑(IsFractionRing.den A x)) ((algebraMap A K) 0)
        ⊢ False
      -/
      exact nonZeroDivisors.coe_ne_zero _ <| NoZeroSMulDivisors.algebraMap_injective A K h
      /-
        🎉 no goals
      -/
    /-
      case h.inj
      A : Type u_1
      inst✝⁵ : CommRing A
      inst✝⁴ : IsDomain A
      inst✝³ : UniqueFactorizationMonoid A
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      x : K
      h✝ : IsLocalization.IsInteger A x
      v : A
      h : Eq ((algebraMap A K) v) x
      ⊢ Function.Injective ⇑(algebraMap A K)
    -/
    exact NoZeroSMulDivisors.algebraMap_injective A K
    /-
      🎉 no goals
    -/


theorem isUnit_den_zero : IsUnit (den A (0 : K) : A) := by
  /-
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    ⊢ IsUnit ↑(IsFractionRing.den A 0)
  -/
  simp [isUnit_den_iff, IsLocalization.isInteger_zero]
  /-
    🎉 no goals
  -/


@[deprecated isUnit_den_zero (since := "2024-07-11")]
theorem isUnit_den_of_num_eq_zero {x : K} (h : num A x = 0) : IsUnit (den A x : A) :=
  eq_zero_of_num_eq_zero h ▸ isUnit_den_zero


lemma associated_den_num_inv (x : K) (hx : x ≠ 0) : Associated (den A x : A) (num A x⁻¹) :=
  associated_of_dvd_dvd
    (IsRelPrime.dvd_of_dvd_mul_right (IsFractionRing.num_den_reduced A x).symm <|
      dvd_of_mul_left_dvd (a := (den A x⁻¹ : A)) <| dvd_of_eq <|
      NoZeroSMulDivisors.algebraMap_injective A K <| Eq.symm <| eq_of_div_eq_one
          /-
            A : Type u_1
            inst✝⁵ : CommRing A
            inst✝⁴ : IsDomain A
            inst✝³ : UniqueFactorizationMonoid A
            K : Type u_2
            inst✝² : Field K
            inst✝¹ : Algebra A K
            inst✝ : IsFractionRing A K
            x : K
            hx : Ne x 0
            ⊢ Eq (HDiv.hDiv ((algebraMap A K) (HMul.hMul (IsFractionRing.num A (Inv.inv x) …
          -/
      (by simp [mul_div_mul_comm, hx]))
          /-
            🎉 no goals
          -/
    (IsRelPrime.dvd_of_dvd_mul_right (IsFractionRing.num_den_reduced A x⁻¹) <|
      dvd_of_mul_left_dvd (a := (num A x : A)) <| dvd_of_eq <|
      NoZeroSMulDivisors.algebraMap_injective A K <| eq_of_div_eq_one
          /-
            A : Type u_1
            inst✝⁵ : CommRing A
            inst✝⁴ : IsDomain A
            inst✝³ : UniqueFactorizationMonoid A
            K : Type u_2
            inst✝² : Field K
            inst✝¹ : Algebra A K
            inst✝ : IsFractionRing A K
            x : K
            hx : Ne x 0
            ⊢ Eq (HDiv.hDiv ((algebraMap A K) (HMul.hMul (IsFractionRing.num A x) (IsFract …
          -/
      (by simp [mul_div_mul_comm, hx]))
          /-
            🎉 no goals
          -/


lemma associated_num_den_inv (x : K) (hx : x ≠ 0) : Associated (num A x : A) (den A x⁻¹) := by
  have : Associated (num A x⁻¹⁻¹ : A) (den A x⁻¹) :=
    (associated_den_num_inv x⁻¹ (inv_ne_zero hx)).symm
  /-
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    hx : Ne x 0
    this : Associated (IsFractionRing.num A (Inv.inv (Inv.inv x))) ↑(IsFractionRin …
    ⊢ Associated (IsFractionRing.num A x) ↑(IsFractionRing.den A (Inv.inv x))
  -/
  rw [inv_inv] at this
  /-
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : IsDomain A
    inst✝³ : UniqueFactorizationMonoid A
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    x : K
    hx : Ne x 0
    this : Associated (IsFractionRing.num A x) ↑(IsFractionRing.den A (Inv.inv x))
    ⊢ Associated (IsFractionRing.num A x) ↑(IsFractionRing.den A (Inv.inv x))
  -/
  exact this
  /-
    🎉 no goals
  -/


