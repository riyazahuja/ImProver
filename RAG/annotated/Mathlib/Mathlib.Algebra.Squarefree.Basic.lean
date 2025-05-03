/-- An element of a monoid is squarefree if the only squares that
  divide it are the squares of units. -/
def Squarefree [Monoid R] (r : R) : Prop :=
  ∀ x : R, x * x ∣ r → IsUnit x


theorem IsRelPrime.of_squarefree_mul [CommMonoid R] {m n : R} (h : Squarefree (m * n)) :
    IsRelPrime m n := fun c hca hcb ↦ h c (mul_dvd_mul hca hcb)


@[simp]
theorem IsUnit.squarefree [CommMonoid R] {x : R} (h : IsUnit x) : Squarefree x := fun _ hdvd =>
  isUnit_of_mul_isUnit_left (isUnit_of_dvd_unit hdvd h)


theorem squarefree_one [CommMonoid R] : Squarefree (1 : R) :=
  isUnit_one.squarefree


@[simp]
theorem not_squarefree_zero [MonoidWithZero R] [Nontrivial R] : ¬Squarefree (0 : R) := by
  /-
    R : Type u_1
    inst✝¹ : MonoidWithZero R
    inst✝ : Nontrivial R
    ⊢ Not (Squarefree 0)
  -/
  erw [not_forall]
  /-
    R : Type u_1
    inst✝¹ : MonoidWithZero R
    inst✝ : Nontrivial R
    ⊢ Exists fun x => Not (Dvd.dvd (HMul.hMul x x) 0 → IsUnit x)
  -/
  exact ⟨0, by simp⟩
  /-
    🎉 no goals
  -/


theorem Squarefree.ne_zero [MonoidWithZero R] [Nontrivial R] {m : R} (hm : Squarefree (m : R)) :
    m ≠ 0 := by
  /-
    R : Type u_1
    inst✝¹ : MonoidWithZero R
    inst✝ : Nontrivial R
    m : R
    hm : Squarefree m
    ⊢ Ne m 0
  -/
  rintro rfl
  /-
    R : Type u_1
    inst✝¹ : MonoidWithZero R
    inst✝ : Nontrivial R
    hm : Squarefree 0
    ⊢ False
  -/
  exact not_squarefree_zero hm
  /-
    🎉 no goals
  -/


@[simp]
theorem Irreducible.squarefree [CommMonoid R] {x : R} (h : Irreducible x) : Squarefree x := by
  /-
    R : Type u_1
    inst✝ : CommMonoid R
    x : R
    h : Irreducible x
    ⊢ Squarefree x
  -/
  rintro y ⟨z, hz⟩
  /-
    case intro
    R : Type u_1
    inst✝ : CommMonoid R
    x : R
    h : Irreducible x
    y z : R
    hz : Eq x (HMul.hMul (HMul.hMul y y) z)
    ⊢ IsUnit y
  -/
  rw [mul_assoc] at hz
  /-
    case intro
    R : Type u_1
    inst✝ : CommMonoid R
    x : R
    h : Irreducible x
    y z : R
    hz : Eq x (HMul.hMul y (HMul.hMul y z))
    ⊢ IsUnit y
  -/
  rcases h.isUnit_or_isUnit hz with (hu | hu)
    /-
      case intro.inl
      R : Type u_1
      inst✝ : CommMonoid R
      x : R
      h : Irreducible x
      y z : R
      hz : Eq x (HMul.hMul y (HMul.hMul y z))
      hu : IsUnit y
      ⊢ IsUnit y
    -/
  · exact hu
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      R : Type u_1
      inst✝ : CommMonoid R
      x : R
      h : Irreducible x
      y z : R
      hz : Eq x (HMul.hMul y (HMul.hMul y z))
      hu : IsUnit (HMul.hMul y z)
      ⊢ IsUnit y
    -/
  · apply isUnit_of_mul_isUnit_left hu
    /-
      🎉 no goals
    -/


@[simp]
theorem Prime.squarefree [CancelCommMonoidWithZero R] {x : R} (h : Prime x) : Squarefree x :=
  h.irreducible.squarefree


theorem Squarefree.of_mul_left [Monoid R] {m n : R} (hmn : Squarefree (m * n)) : Squarefree m :=
  fun p hp => hmn p (dvd_mul_of_dvd_left hp n)


theorem Squarefree.of_mul_right [CommMonoid R] {m n : R} (hmn : Squarefree (m * n)) :
    Squarefree n := fun p hp => hmn p (dvd_mul_of_dvd_right hp m)


theorem Squarefree.squarefree_of_dvd [Monoid R] {x y : R} (hdvd : x ∣ y) (hsq : Squarefree y) :
    Squarefree x := fun _ h => hsq _ (h.trans hdvd)


theorem Squarefree.eq_zero_or_one_of_pow_of_not_isUnit [Monoid R] {x : R} {n : ℕ}
    (h : Squarefree (x ^ n)) (h' : ¬ IsUnit x) :
    n = 0 ∨ n = 1 := by
  /-
    R : Type u_1
    inst✝ : Monoid R
    x : R
    n : Nat
    h : Squarefree (HPow.hPow x n)
    h' : Not (IsUnit x)
    ⊢ Or (Eq n 0) (Eq n 1)
  -/
  contrapose! h'
  /-
    R : Type u_1
    inst✝ : Monoid R
    x : R
    n : Nat
    h : Squarefree (HPow.hPow x n)
    h' : And (Ne n 0) (Ne n 1)
    ⊢ IsUnit x
  -/
  replace h' : 2 ≤ n := by omega
  /-
    R : Type u_1
    inst✝ : Monoid R
    x : R
    n : Nat
    h : Squarefree (HPow.hPow x n)
    h' : LE.le 2 n
    ⊢ IsUnit x
  -/
  have : x * x ∣ x ^ n := by rw [← sq]; exact pow_dvd_pow x h'
  /-
    R : Type u_1
    inst✝ : Monoid R
    x : R
    n : Nat
    h : Squarefree (HPow.hPow x n)
    h' : LE.le 2 n
    this : Dvd.dvd (HMul.hMul x x) (HPow.hPow x n)
    ⊢ IsUnit x
  -/
  exact h.squarefree_of_dvd this x (refl _)
  /-
    🎉 no goals
  -/


theorem Squarefree.pow_dvd_of_pow_dvd [Monoid R] {x y : R} {n : ℕ}
    (hx : Squarefree y) (h : x ^ n ∣ y) : x ^ n ∣ x := by
  /-
    R : Type u_1
    inst✝ : Monoid R
    x y : R
    n : Nat
    hx : Squarefree y
    h : Dvd.dvd (HPow.hPow x n) y
    ⊢ Dvd.dvd (HPow.hPow x n) x
  -/
  by_cases hu : IsUnit x
    /-
      case pos
      R : Type u_1
      inst✝ : Monoid R
      x y : R
      n : Nat
      hx : Squarefree y
      h : Dvd.dvd (HPow.hPow x n) y
      hu : IsUnit x
      ⊢ Dvd.dvd (HPow.hPow x n) x
    -/
  · exact (hu.pow n).dvd
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Monoid R
      x y : R
      n : Nat
      hx : Squarefree y
      h : Dvd.dvd (HPow.hPow x n) y
      hu : Not (IsUnit x)
      ⊢ Dvd.dvd (HPow.hPow x n) x
    -/
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/
  · rcases (hx.squarefree_of_dvd h).eq_zero_or_one_of_pow_of_not_isUnit hu with rfl | rfl <;> simp
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


theorem Squarefree.gcd_right (a : α) {b : α} (hb : Squarefree b) : Squarefree (gcd a b) :=
  hb.squarefree_of_dvd (gcd_dvd_right _ _)


theorem Squarefree.gcd_left {a : α} (b : α) (ha : Squarefree a) : Squarefree (gcd a b) :=
  ha.squarefree_of_dvd (gcd_dvd_left _ _)


theorem squarefree_iff_emultiplicity_le_one [CommMonoid R] (r : R) :
    Squarefree r ↔ ∀ x : R, emultiplicity x r ≤ 1 ∨ IsUnit x := by
  /-
    R : Type u_1
    inst✝ : CommMonoid R
    r : R
    ⊢ Iff (Squarefree r) (∀ (x : R), Or (LE.le (emultiplicity x r) 1) (IsUnit x))
  -/
  refine forall_congr' fun a => ?_
  /-
    R : Type u_1
    inst✝ : CommMonoid R
    r a : R
    ⊢ Iff (Dvd.dvd (HMul.hMul a a) r → IsUnit a) (Or (LE.le (emultiplicity a r) 1) …
  -/
  rw [← sq, pow_dvd_iff_le_emultiplicity, or_iff_not_imp_left, not_le, imp_congr _ Iff.rfl]
  /-
    R : Type u_1
    inst✝ : CommMonoid R
    r a : R
    ⊢ Iff (LE.le (↑2) (emultiplicity a r)) (LT.lt 1 (emultiplicity a r))
  -/
  norm_cast
  /-
    R : Type u_1
    inst✝ : CommMonoid R
    r a : R
    ⊢ Iff (LE.le 2 (emultiplicity a r)) (LT.lt 1 (emultiplicity a r))
  -/
  rw [← one_add_one_eq_two]
  /-
    R : Type u_1
    inst✝ : CommMonoid R
    r a : R
    ⊢ Iff (LE.le (HAdd.hAdd 1 1) (emultiplicity a r)) (LT.lt 1 (emultiplicity a r))
  -/
  exact Order.add_one_le_iff_of_not_isMax (by simp)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.squarefree_iff_emultiplicity_le_one := squarefree_iff_emultiplicity_le_one


theorem squarefree_iff_no_irreducibles {x : R} (hx₀ : x ≠ 0) :
    Squarefree x ↔ ∀ p, Irreducible p → ¬ (p * p ∣ x) := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    inst✝ : WfDvdMonoid R
    x : R
    hx₀ : Ne x 0
    ⊢ Iff (Squarefree x) (∀ (p : R), Irreducible p → Not (Dvd.dvd (HMul.hMul p p)  …
  -/
  refine ⟨fun h p hp hp' ↦ hp.not_unit (h p hp'), fun h d hd ↦ by_contra fun hdu ↦ ?_⟩
  /-
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    inst✝ : WfDvdMonoid R
    x : R
    hx₀ : Ne x 0
    h : ∀ (p : R), Irreducible p → Not (Dvd.dvd (HMul.hMul p p) x)
    d : R
    hd : Dvd.dvd (HMul.hMul d d) x
    hdu : Not (IsUnit d)
    ⊢ False
  -/
  have hd₀ : d ≠ 0 := ne_zero_of_dvd_ne_zero (ne_zero_of_dvd_ne_zero hx₀ hd) (dvd_mul_left d d)
  /-
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    inst✝ : WfDvdMonoid R
    x : R
    hx₀ : Ne x 0
    h : ∀ (p : R), Irreducible p → Not (Dvd.dvd (HMul.hMul p p) x)
    d : R
    hd : Dvd.dvd (HMul.hMul d d) x
    hdu : Not (IsUnit d)
    hd₀ : Ne d 0
    ⊢ False
  -/
  obtain ⟨p, irr, dvd⟩ := WfDvdMonoid.exists_irreducible_factor hdu hd₀
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    inst✝ : WfDvdMonoid R
    x : R
    hx₀ : Ne x 0
    h : ∀ (p : R), Irreducible p → Not (Dvd.dvd (HMul.hMul p p) x)
    d : R
    hd : Dvd.dvd (HMul.hMul d d) x
    hdu : Not (IsUnit d)
    hd₀ : Ne d 0
    p : R
    irr : Irreducible p
    dvd : Dvd.dvd p d
    ⊢ False
  -/
  exact h p irr ((mul_dvd_mul dvd dvd).trans hd)
  /-
    🎉 no goals
  -/


theorem irreducible_sq_not_dvd_iff_eq_zero_and_no_irreducibles_or_squarefree (r : R) :
    (∀ x : R, Irreducible x → ¬x * x ∣ r) ↔ (r = 0 ∧ ∀ x : R, ¬Irreducible x) ∨ Squarefree r := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    inst✝ : WfDvdMonoid R
    r : R
    ⊢ Iff (∀ (x : R), Irreducible x → Not (Dvd.dvd (HMul.hMul x x) r)) (Or (And (E …
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝¹ : CommMonoidWithZero R
      inst✝ : WfDvdMonoid R
      r : R
      h : ∀ (x : R), Irreducible x → Not (Dvd.dvd (HMul.hMul x x) r)
      ⊢ Or (And (Eq r 0) (∀ (x : R), Not (Irreducible x))) (Squarefree r)
    -/
  · rcases eq_or_ne r 0 with (rfl | hr)
      /-
        case refine_1.inl
        R : Type u_1
        inst✝¹ : CommMonoidWithZero R
        inst✝ : WfDvdMonoid R
        h : ∀ (x : R), Irreducible x → Not (Dvd.dvd (HMul.hMul x x) 0)
        ⊢ Or (And (Eq 0 0) (∀ (x : R), Not (Irreducible x))) (Squarefree 0)
      -/
    · exact .inl (by simpa using h)
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr
        R : Type u_1
        inst✝¹ : CommMonoidWithZero R
        inst✝ : WfDvdMonoid R
        r : R
        h : ∀ (x : R), Irreducible x → Not (Dvd.dvd (HMul.hMul x x) r)
        hr : Ne r 0
        ⊢ Or (And (Eq r 0) (∀ (x : R), Not (Irreducible x))) (Squarefree r)
      -/
    · exact .inr ((squarefree_iff_no_irreducibles hr).mpr h)
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      R : Type u_1
      inst✝¹ : CommMonoidWithZero R
      inst✝ : WfDvdMonoid R
      r : R
      ⊢ Or (And (Eq r 0) (∀ (x : R), Not (Irreducible x))) (Squarefree r) → ∀ (x : R …
    -/
  · rintro (⟨rfl, h⟩ | h)
      /-
        case refine_2.inl.intro
        R : Type u_1
        inst✝¹ : CommMonoidWithZero R
        inst✝ : WfDvdMonoid R
        h : ∀ (x : R), Not (Irreducible x)
        ⊢ ∀ (x : R), Irreducible x → Not (Dvd.dvd (HMul.hMul x x) 0)
      -/
    · simpa using h
      /-
        🎉 no goals
      -/
    /-
      case refine_2.inr
      R : Type u_1
      inst✝¹ : CommMonoidWithZero R
      inst✝ : WfDvdMonoid R
      r : R
      h : Squarefree r
      ⊢ ∀ (x : R), Irreducible x → Not (Dvd.dvd (HMul.hMul x x) r)
    -/
    intro x hx t
    /-
      case refine_2.inr
      R : Type u_1
      inst✝¹ : CommMonoidWithZero R
      inst✝ : WfDvdMonoid R
      r : R
      h : Squarefree r
      x : R
      hx : Irreducible x
      t : Dvd.dvd (HMul.hMul x x) r
      ⊢ False
    -/
    exact hx.not_unit (h x t)
    /-
      🎉 no goals
    -/


theorem squarefree_iff_irreducible_sq_not_dvd_of_ne_zero {r : R} (hr : r ≠ 0) :
    Squarefree r ↔ ∀ x : R, Irreducible x → ¬x * x ∣ r := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    inst✝ : WfDvdMonoid R
    r : R
    hr : Ne r 0
    ⊢ Iff (Squarefree r) (∀ (x : R), Irreducible x → Not (Dvd.dvd (HMul.hMul x x)  …
  -/
  simpa [hr] using (irreducible_sq_not_dvd_iff_eq_zero_and_no_irreducibles_or_squarefree r).symm
  /-
    🎉 no goals
  -/


theorem squarefree_iff_irreducible_sq_not_dvd_of_exists_irreducible {r : R}
    (hr : ∃ x : R, Irreducible x) : Squarefree r ↔ ∀ x : R, Irreducible x → ¬x * x ∣ r := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    inst✝ : WfDvdMonoid R
    r : R
    hr : Exists fun x => Irreducible x
    ⊢ Iff (Squarefree r) (∀ (x : R), Irreducible x → Not (Dvd.dvd (HMul.hMul x x)  …
  -/
  rw [irreducible_sq_not_dvd_iff_eq_zero_and_no_irreducibles_or_squarefree, ← not_exists]
  /-
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    inst✝ : WfDvdMonoid R
    r : R
    hr : Exists fun x => Irreducible x
    ⊢ Iff (Squarefree r) (Or (And (Eq r 0) (Not (Exists fun x => Irreducible x)))  …
  -/
  simp only [hr, not_true, false_or, and_false]
  /-
    🎉 no goals
  -/


theorem Squarefree.isRadical {x : R} (hx : Squarefree x) : IsRadical x :=
  (isRadical_iff_pow_one_lt 2 one_lt_two).2 fun y hy ↦ by
    /-
      R : Type u_1
      inst✝¹ : CommMonoidWithZero R
      inst✝ : DecompositionMonoid R
      x : R
      hx : Squarefree x
      y : R
      hy : Dvd.dvd x (HPow.hPow y 2)
      ⊢ Dvd.dvd x y
    -/
    obtain ⟨a, b, ha, hb, rfl⟩ := exists_dvd_and_dvd_of_dvd_mul (sq y ▸ hy)
    /-
      case intro.intro.intro.intro
      R : Type u_1
      inst✝¹ : CommMonoidWithZero R
      inst✝ : DecompositionMonoid R
      y a b : R
      ha : Dvd.dvd a y
      hb : Dvd.dvd b y
      hx : Squarefree (HMul.hMul a b)
      hy : Dvd.dvd (HMul.hMul a b) (HPow.hPow y 2)
      ⊢ Dvd.dvd (HMul.hMul a b) y
    -/
    exact (IsRelPrime.of_squarefree_mul hx).mul_dvd ha hb
    /-
      🎉 no goals
    -/


theorem Squarefree.dvd_pow_iff_dvd {x y : R} {n : ℕ} (hsq : Squarefree x) (h0 : n ≠ 0) :
    x ∣ y ^ n ↔ x ∣ y := ⟨hsq.isRadical n y, (·.pow h0)⟩


@[deprecated (since := "2024-02-12")]
alias UniqueFactorizationMonoid.dvd_pow_iff_dvd_of_squarefree := Squarefree.dvd_pow_iff_dvd


theorem IsRadical.squarefree (h0 : x ≠ 0) (h : IsRadical x) : Squarefree x := by
  /-
    R : Type u_1
    inst✝ : CancelCommMonoidWithZero R
    x : R
    h0 : Ne x 0
    h : IsRadical x
    ⊢ Squarefree x
  -/
  rintro z ⟨w, rfl⟩
  /-
    case intro
    R : Type u_1
    inst✝ : CancelCommMonoidWithZero R
    z w : R
    h0 : Ne (HMul.hMul (HMul.hMul z z) w) 0
    h : IsRadical (HMul.hMul (HMul.hMul z z) w)
    ⊢ IsUnit z
  -/
  specialize h 2 (z * w) ⟨w, by simp_rw [pow_two, mul_left_comm, ← mul_assoc]⟩
  /-
    case intro
    R : Type u_1
    inst✝ : CancelCommMonoidWithZero R
    z w : R
    h0 : Ne (HMul.hMul (HMul.hMul z z) w) 0
    h : Dvd.dvd (HMul.hMul (HMul.hMul z z) w) (HMul.hMul z w)
    ⊢ IsUnit z
  -/
  rwa [← one_mul (z * w), mul_assoc, mul_dvd_mul_iff_right, ← isUnit_iff_dvd_one] at h
  /-
    case intro
    R : Type u_1
    inst✝ : CancelCommMonoidWithZero R
    z w : R
    h0 : Ne (HMul.hMul (HMul.hMul z z) w) 0
    h : Dvd.dvd (HMul.hMul z (HMul.hMul z w)) (HMul.hMul 1 (HMul.hMul z w))
    ⊢ Ne (HMul.hMul z w) 0
  -/
  rw [mul_assoc, mul_ne_zero_iff] at h0; exact h0.2
                                         /-
                                           🎉 no goals
                                         -/


theorem pow_dvd_of_squarefree_of_pow_succ_dvd_mul_right {k : ℕ}
    (hx : Squarefree x) (hp : Prime p) (h : p ^ (k + 1) ∣ x * y) :
    p ^ k ∣ y := by
  /-
    R : Type u_1
    inst✝ : CancelCommMonoidWithZero R
    x y p : R
    k : Nat
    hx : Squarefree x
    hp : Prime p
    h : Dvd.dvd (HPow.hPow p (HAdd.hAdd k 1)) (HMul.hMul x y)
    ⊢ Dvd.dvd (HPow.hPow p k) y
  -/
  by_cases hxp : p ∣ x
    /-
      case pos
      R : Type u_1
      inst✝ : CancelCommMonoidWithZero R
      x y p : R
      k : Nat
      hx : Squarefree x
      hp : Prime p
      h : Dvd.dvd (HPow.hPow p (HAdd.hAdd k 1)) (HMul.hMul x y)
      hxp : Dvd.dvd p x
      ⊢ Dvd.dvd (HPow.hPow p k) y
    -/
  · obtain ⟨x', rfl⟩ := hxp
    /-
      case pos.intro
      R : Type u_1
      inst✝ : CancelCommMonoidWithZero R
      y p : R
      k : Nat
      hp : Prime p
      x' : R
      hx : Squarefree (HMul.hMul p x')
      h : Dvd.dvd (HPow.hPow p (HAdd.hAdd k 1)) (HMul.hMul (HMul.hMul p x') y)
      ⊢ Dvd.dvd (HPow.hPow p k) y
    -/
    have hx' : ¬ p ∣ x' := fun contra ↦ hp.not_unit <| hx p (mul_dvd_mul_left p contra)
    replace h : p ^ k ∣ x' * y := by
      rw [pow_succ', mul_assoc] at h
      exact (mul_dvd_mul_iff_left hp.ne_zero).mp h
    /-
      case pos.intro
      R : Type u_1
      inst✝ : CancelCommMonoidWithZero R
      y p : R
      k : Nat
      hp : Prime p
      x' : R
      hx : Squarefree (HMul.hMul p x')
      hx' : Not (Dvd.dvd p x')
      h : Dvd.dvd (HPow.hPow p k) (HMul.hMul x' y)
      ⊢ Dvd.dvd (HPow.hPow p k) y
    -/
    exact hp.pow_dvd_of_dvd_mul_left _ hx' h
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : CancelCommMonoidWithZero R
      x y p : R
      k : Nat
      hx : Squarefree x
      hp : Prime p
      h : Dvd.dvd (HPow.hPow p (HAdd.hAdd k 1)) (HMul.hMul x y)
      hxp : Not (Dvd.dvd p x)
      ⊢ Dvd.dvd (HPow.hPow p k) y
    -/
  · exact (pow_dvd_pow _ k.le_succ).trans (hp.pow_dvd_of_dvd_mul_left _ hxp h)
    /-
      🎉 no goals
    -/


theorem pow_dvd_of_squarefree_of_pow_succ_dvd_mul_left {k : ℕ}
    (hy : Squarefree y) (hp : Prime p) (h : p ^ (k + 1) ∣ x * y) :
    p ^ k ∣ x := by
  /-
    R : Type u_1
    inst✝ : CancelCommMonoidWithZero R
    x y p : R
    k : Nat
    hy : Squarefree y
    hp : Prime p
    h : Dvd.dvd (HPow.hPow p (HAdd.hAdd k 1)) (HMul.hMul x y)
    ⊢ Dvd.dvd (HPow.hPow p k) x
  -/
  rw [mul_comm] at h
  /-
    R : Type u_1
    inst✝ : CancelCommMonoidWithZero R
    x y p : R
    k : Nat
    hy : Squarefree y
    hp : Prime p
    h : Dvd.dvd (HPow.hPow p (HAdd.hAdd k 1)) (HMul.hMul y x)
    ⊢ Dvd.dvd (HPow.hPow p k) x
  -/
  exact pow_dvd_of_squarefree_of_pow_succ_dvd_mul_right hy hp h
  /-
    🎉 no goals
  -/


theorem dvd_of_squarefree_of_mul_dvd_mul_right (hx : Squarefree x) (h : d * d ∣ x * y) : d ∣ y := by
  /-
    R : Type u_1
    inst✝¹ : CancelCommMonoidWithZero R
    x y d : R
    inst✝ : DecompositionMonoid R
    hx : Squarefree x
    h : Dvd.dvd (HMul.hMul d d) (HMul.hMul x y)
    ⊢ Dvd.dvd d y
  -/
  nontriviality R
  /-
    R : Type u_1
    inst✝¹ : CancelCommMonoidWithZero R
    x y d : R
    inst✝ : DecompositionMonoid R
    hx : Squarefree x
    h : Dvd.dvd (HMul.hMul d d) (HMul.hMul x y)
    a✝ : Nontrivial R
    ⊢ Dvd.dvd d y
  -/
  obtain ⟨a, b, ha, hb, eq⟩ := exists_dvd_and_dvd_of_dvd_mul h
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝¹ : CancelCommMonoidWithZero R
    x y d : R
    inst✝ : DecompositionMonoid R
    hx : Squarefree x
    h : Dvd.dvd (HMul.hMul d d) (HMul.hMul x y)
    a✝ : Nontrivial R
    a b : R
    ha : Dvd.dvd a x
    hb : Dvd.dvd b y
    eq : Eq (HMul.hMul d d) (HMul.hMul a b)
    ⊢ Dvd.dvd d y
  -/
  replace ha : Squarefree a := hx.squarefree_of_dvd ha
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝¹ : CancelCommMonoidWithZero R
    x y d : R
    inst✝ : DecompositionMonoid R
    hx : Squarefree x
    h : Dvd.dvd (HMul.hMul d d) (HMul.hMul x y)
    a✝ : Nontrivial R
    a b : R
    hb : Dvd.dvd b y
    eq : Eq (HMul.hMul d d) (HMul.hMul a b)
    ha : Squarefree a
    ⊢ Dvd.dvd d y
  -/
  obtain ⟨c, hc⟩ : a ∣ d := ha.isRadical 2 d ⟨b, by rw [sq, eq]⟩
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    inst✝¹ : CancelCommMonoidWithZero R
    x y d : R
    inst✝ : DecompositionMonoid R
    hx : Squarefree x
    h : Dvd.dvd (HMul.hMul d d) (HMul.hMul x y)
    a✝ : Nontrivial R
    a b : R
    hb : Dvd.dvd b y
    eq : Eq (HMul.hMul d d) (HMul.hMul a b)
    ha : Squarefree a
    c : R
    hc : Eq d (HMul.hMul a c)
    ⊢ Dvd.dvd d y
  -/
  rw [hc, mul_assoc, (mul_right_injective₀ ha.ne_zero).eq_iff] at eq
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    inst✝¹ : CancelCommMonoidWithZero R
    x y d : R
    inst✝ : DecompositionMonoid R
    hx : Squarefree x
    h : Dvd.dvd (HMul.hMul d d) (HMul.hMul x y)
    a✝ : Nontrivial R
    a b : R
    hb : Dvd.dvd b y
    ha : Squarefree a
    c : R
    eq : Eq (HMul.hMul c (HMul.hMul a c)) b
    hc : Eq d (HMul.hMul a c)
    ⊢ Dvd.dvd d y
  -/
  exact dvd_trans ⟨c, by rw [hc, ← eq, mul_comm]⟩ hb
  /-
    🎉 no goals
  -/


theorem dvd_of_squarefree_of_mul_dvd_mul_left (hy : Squarefree y) (h : d * d ∣ x * y) : d ∣ x :=
  dvd_of_squarefree_of_mul_dvd_mul_right hy (mul_comm x y ▸ h)


/-- `x * y` is square-free iff `x` and `y` have no common factors and are themselves square-free. -/
theorem squarefree_mul_iff : Squarefree (x * y) ↔ IsRelPrime x y ∧ Squarefree x ∧ Squarefree y :=
  ⟨fun h ↦ ⟨IsRelPrime.of_squarefree_mul h, h.of_mul_left, h.of_mul_right⟩,
    fun ⟨hp, sqx, sqy⟩ _ dvd ↦ hp (sqy.dvd_of_squarefree_of_mul_dvd_mul_left dvd)
      (sqx.dvd_of_squarefree_of_mul_dvd_mul_right dvd)⟩


theorem isRadical_iff_squarefree_or_zero : IsRadical x ↔ Squarefree x ∨ x = 0 :=
  ⟨fun hx ↦ (em <| x = 0).elim .inr fun h ↦ .inl <| hx.squarefree h,
    Or.rec Squarefree.isRadical <| by
      /-
        R : Type u_1
        inst✝¹ : CancelCommMonoidWithZero R
        x : R
        inst✝ : DecompositionMonoid R
        ⊢ Eq x 0 → IsRadical x
      -/
      rintro rfl
      /-
        R : Type u_1
        inst✝¹ : CancelCommMonoidWithZero R
        inst✝ : DecompositionMonoid R
        ⊢ IsRadical 0
      -/
      rw [zero_isRadical_iff]
      /-
        R : Type u_1
        inst✝¹ : CancelCommMonoidWithZero R
        inst✝ : DecompositionMonoid R
        ⊢ IsReduced R
      -/
      infer_instance⟩
      /-
        🎉 no goals
      -/


theorem isRadical_iff_squarefree_of_ne_zero (h : x ≠ 0) : IsRadical x ↔ Squarefree x :=
  ⟨IsRadical.squarefree h, Squarefree.isRadical⟩


lemma _root_.exists_squarefree_dvd_pow_of_ne_zero {x : R} (hx : x ≠ 0) :
    ∃ (y : R) (n : ℕ), Squarefree y ∧ y ∣ x ∧ x ∣ y ^ n := by
  /-
    R : Type u_1
    inst✝¹ : CancelCommMonoidWithZero R
    inst✝ : UniqueFactorizationMonoid R
    x : R
    hx : Ne x 0
    ⊢ Exists fun y => Exists fun n => And (Squarefree y) (And (Dvd.dvd y x) (Dvd.d …
  -/
  induction' x using WfDvdMonoid.induction_on_irreducible with u hu z p hz hp ih
    /-
      case h0
      R : Type u_1
      inst✝¹ : CancelCommMonoidWithZero R
      inst✝ : UniqueFactorizationMonoid R
      hx : Ne 0 0
      ⊢ Exists fun y => Exists fun n => And (Squarefree y) (And (Dvd.dvd y 0) (Dvd.d …
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case hu
      R : Type u_1
      inst✝¹ : CancelCommMonoidWithZero R
      inst✝ : UniqueFactorizationMonoid R
      u : R
      hu : IsUnit u
      hx : Ne u 0
      ⊢ Exists fun y => Exists fun n => And (Squarefree y) (And (Dvd.dvd y u) (Dvd.d …
    -/
  · exact ⟨1, 0, squarefree_one, one_dvd u, hu.dvd⟩
    /-
      🎉 no goals
    -/
    /-
      case hi
      R : Type u_1
      inst✝¹ : CancelCommMonoidWithZero R
      inst✝ : UniqueFactorizationMonoid R
      z p : R
      hz : Ne z 0
      hp : Irreducible p
      ih : Ne z 0 → Exists fun y => Exists fun n => And (Squarefree y) (And (Dvd.dvd …
      hx : Ne (HMul.hMul p z) 0
      ⊢ Exists fun y => Exists fun n => And (Squarefree y) (And (Dvd.dvd y (HMul.hMu …
    -/
  · obtain ⟨y, n, hy, hyx, hy'⟩ := ih hz
    /-
      case hi.intro.intro.intro.intro
      R : Type u_1
      inst✝¹ : CancelCommMonoidWithZero R
      inst✝ : UniqueFactorizationMonoid R
      z p : R
      hz : Ne z 0
      hp : Irreducible p
      ih : Ne z 0 → Exists fun y => Exists fun n => And (Squarefree y) (And (Dvd.dvd …
      hx : Ne (HMul.hMul p z) 0
      y : R
      n : Nat
      hy : Squarefree y
      hyx : Dvd.dvd y z
      hy' : Dvd.dvd z (HPow.hPow y n)
      ⊢ Exists fun y => Exists fun n => And (Squarefree y) (And (Dvd.dvd y (HMul.hMu …
    -/
    rcases n.eq_zero_or_pos with rfl | hn
      /-
        case hi.intro.intro.intro.intro.inl
        R : Type u_1
        inst✝¹ : CancelCommMonoidWithZero R
        inst✝ : UniqueFactorizationMonoid R
        z p : R
        hz : Ne z 0
        hp : Irreducible p
        ih : Ne z 0 → Exists fun y => Exists fun n => And (Squarefree y) (And (Dvd.dvd …
        hx : Ne (HMul.hMul p z) 0
        y : R
        hy : Squarefree y
        hyx : Dvd.dvd y z
        hy' : Dvd.dvd z (HPow.hPow y 0)
        ⊢ Exists fun y => Exists fun n => And (Squarefree y) (And (Dvd.dvd y (HMul.hMu …
      -/
    · exact ⟨p, 1, hp.squarefree, dvd_mul_right p z, by simp [isUnit_of_dvd_one (pow_zero y ▸ hy')]⟩
      /-
        🎉 no goals
      -/
    /-
      case hi.intro.intro.intro.intro.inr
      R : Type u_1
      inst✝¹ : CancelCommMonoidWithZero R
      inst✝ : UniqueFactorizationMonoid R
      z p : R
      hz : Ne z 0
      hp : Irreducible p
      ih : Ne z 0 → Exists fun y => Exists fun n => And (Squarefree y) (And (Dvd.dvd …
      hx : Ne (HMul.hMul p z) 0
      y : R
      n : Nat
      hy : Squarefree y
      hyx : Dvd.dvd y z
      hy' : Dvd.dvd z (HPow.hPow y n)
      hn : GT.gt n 0
      ⊢ Exists fun y => Exists fun n => And (Squarefree y) (And (Dvd.dvd y (HMul.hMu …
    -/
    by_cases hp' : p ∣ y
    · exact ⟨y, n + 1, hy, dvd_mul_of_dvd_right hyx _,
        mul_comm p z ▸ pow_succ y n ▸ mul_dvd_mul hy' hp'⟩
    · suffices Squarefree (p * y) from ⟨p * y, n, this,
        mul_dvd_mul_left p hyx, mul_pow p y n ▸ mul_dvd_mul (dvd_pow_self p hn.ne') hy'⟩
      /-
        case neg
        R : Type u_1
        inst✝¹ : CancelCommMonoidWithZero R
        inst✝ : UniqueFactorizationMonoid R
        z p : R
        hz : Ne z 0
        hp : Irreducible p
        ih : Ne z 0 → Exists fun y => Exists fun n => And (Squarefree y) (And (Dvd.dvd …
        hx : Ne (HMul.hMul p z) 0
        y : R
        n : Nat
        hy : Squarefree y
        hyx : Dvd.dvd y z
        hy' : Dvd.dvd z (HPow.hPow y n)
        hn : GT.gt n 0
        hp' : Not (Dvd.dvd p y)
        ⊢ Squarefree (HMul.hMul p y)
      -/
      exact squarefree_mul_iff.mpr ⟨hp.isRelPrime_iff_not_dvd.mpr hp', hp.squarefree, hy⟩
      /-
        🎉 no goals
      -/


theorem squarefree_iff_nodup_normalizedFactors [NormalizationMonoid R] {x : R}
    (x0 : x ≠ 0) : Squarefree x ↔ Multiset.Nodup (normalizedFactors x) := by
  classical
  rw [squarefree_iff_emultiplicity_le_one, Multiset.nodup_iff_count_le_one]
  haveI := nontrivial_of_ne x 0 x0
  constructor <;> intro h a
  · by_cases hmem : a ∈ normalizedFactors x
    · have ha := irreducible_of_normalized_factor _ hmem
      rcases h a with (h | h)
      · rw [← normalize_normalized_factor _ hmem]
        rw [emultiplicity_eq_count_normalizedFactors ha x0] at h
        assumption_mod_cast
      · have := ha.1
        contradiction
    · simp [Multiset.count_eq_zero_of_not_mem hmem]
  · rw [or_iff_not_imp_right]
    intro hu
    rcases eq_or_ne a 0 with rfl | h0
    · simp [x0]
    rcases WfDvdMonoid.exists_irreducible_factor hu h0 with ⟨b, hib, hdvd⟩
    apply le_trans (emultiplicity_le_emultiplicity_of_dvd_left hdvd)
    rw [emultiplicity_eq_count_normalizedFactors hib x0]
    exact_mod_cast h (normalize b)


@[simp]
theorem squarefree_natAbs {n : ℤ} : Squarefree n.natAbs ↔ Squarefree n := by
  simp_rw [Squarefree, natAbs_surjective.forall, ← natAbs_mul, natAbs_dvd_natAbs,
    isUnit_iff_natAbs_eq, Nat.isUnit_iff]


@[simp]
theorem squarefree_natCast {n : ℕ} : Squarefree (n : ℤ) ↔ Squarefree n := by
  /-
    n : Nat
    ⊢ Iff (Squarefree ↑n) (Squarefree n)
  -/
  rw [← squarefree_natAbs, natAbs_ofNat]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-05")] alias squarefree_coe_nat := squarefree_natCast


