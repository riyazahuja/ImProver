/-- An element `p` of a commutative monoid with zero (e.g., a ring) is called *prime*,
if it's not zero, not a unit, and `p ∣ a * b → p ∣ a ∨ p ∣ b` for all `a`, `b`. -/
def Prime (p : M) : Prop :=
  p ≠ 0 ∧ ¬IsUnit p ∧ ∀ a b, p ∣ a * b → p ∣ a ∨ p ∣ b


theorem ne_zero : p ≠ 0 :=
  hp.1


theorem not_unit : ¬IsUnit p :=
  hp.2.1


theorem not_dvd_one : ¬p ∣ 1 :=
  mt (isUnit_of_dvd_one ·) hp.not_unit


theorem ne_one : p ≠ 1 := fun h => hp.2.1 (h.symm ▸ isUnit_one)


theorem dvd_or_dvd {a b : M} (h : p ∣ a * b) : p ∣ a ∨ p ∣ b :=
  hp.2.2 a b h


theorem dvd_mul {a b : M} : p ∣ a * b ↔ p ∣ a ∨ p ∣ b :=
  ⟨hp.dvd_or_dvd, (Or.elim · (dvd_mul_of_dvd_left · _) (dvd_mul_of_dvd_right · _))⟩


theorem isPrimal : IsPrimal p := fun _a _b dvd ↦ (hp.dvd_or_dvd dvd).elim
  (fun h ↦ ⟨p, 1, h, one_dvd _, (mul_one p).symm⟩) fun h ↦ ⟨1, p, one_dvd _, h, (one_mul p).symm⟩


theorem not_dvd_mul {a b : M} (ha : ¬ p ∣ a) (hb : ¬ p ∣ b) : ¬ p ∣ a * b :=
  hp.dvd_mul.not.mpr <| not_or.mpr ⟨ha, hb⟩


theorem dvd_of_dvd_pow {a : M} {n : ℕ} (h : p ∣ a ^ n) : p ∣ a := by
  induction n with
  | zero =>
    rw [pow_zero] at h
    have := isUnit_of_dvd_one h
    have := not_unit hp
    contradiction
  | succ n ih =>
    rw [pow_succ'] at h
    rcases dvd_or_dvd hp h with dvd_a | dvd_pow
    · assumption
    · exact ih dvd_pow


theorem dvd_pow_iff_dvd {a : M} {n : ℕ} (hn : n ≠ 0) : p ∣ a ^ n ↔ p ∣ a :=
  ⟨hp.dvd_of_dvd_pow, (dvd_pow · hn)⟩


@[simp]
theorem not_prime_zero : ¬Prime (0 : M) := fun h => h.ne_zero rfl


@[simp]
theorem not_prime_one : ¬Prime (1 : M) := fun h => h.not_unit isUnit_one


/-- `Irreducible p` states that `p` is non-unit and only factors into units.

We explicitly avoid stating that `p` is non-zero, this would require a semiring. Assuming only a
monoid allows us to reuse irreducible for associated elements.
-/
structure Irreducible [Monoid M] (p : M) : Prop where
  /-- `p` is not a unit -/
  not_unit : ¬IsUnit p
  /-- if `p` factors then one factor is a unit -/
  isUnit_or_isUnit' : ∀ a b, p = a * b → IsUnit a ∨ IsUnit b


theorem not_dvd_one [CommMonoid M] {p : M} (hp : Irreducible p) : ¬p ∣ 1 :=
  mt (isUnit_of_dvd_one ·) hp.not_unit


theorem isUnit_or_isUnit [Monoid M] {p : M} (hp : Irreducible p) {a b : M} (h : p = a * b) :
    IsUnit a ∨ IsUnit b :=
  hp.isUnit_or_isUnit' a b h


theorem irreducible_iff [Monoid M] {p : M} :
    Irreducible p ↔ ¬IsUnit p ∧ ∀ a b, p = a * b → IsUnit a ∨ IsUnit b :=
  ⟨fun h => ⟨h.1, h.2⟩, fun h => ⟨h.1, h.2⟩⟩


@[simp]
                                                                    /-
                                                                      M : Type u_1
                                                                      inst✝ : Monoid M
                                                                      ⊢ Not (Irreducible 1)
                                                                    -/
theorem not_irreducible_one [Monoid M] : ¬Irreducible (1 : M) := by simp [irreducible_iff]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem Irreducible.ne_one [Monoid M] : ∀ {p : M}, Irreducible p → p ≠ 1
  | _, hp, rfl => not_irreducible_one hp


@[simp]
theorem not_irreducible_zero [MonoidWithZero M] : ¬Irreducible (0 : M)
  | ⟨hn0, h⟩ =>
    have : IsUnit (0 : M) ∨ IsUnit (0 : M) := h 0 0 (mul_zero 0).symm
    this.elim hn0 hn0


theorem Irreducible.ne_zero [MonoidWithZero M] : ∀ {p : M}, Irreducible p → p ≠ 0
  | _, hp, rfl => not_irreducible_zero hp


theorem of_irreducible_mul {M} [Monoid M] {x y : M} : Irreducible (x * y) → IsUnit x ∨ IsUnit y
  | ⟨_, h⟩ => h _ _ rfl


theorem irreducible_or_factor {M} [Monoid M] (x : M) (h : ¬IsUnit x) :
    Irreducible x ∨ ∃ a b, ¬IsUnit a ∧ ¬IsUnit b ∧ a * b = x := by
  /-
    M : Type u_2
    inst✝ : Monoid M
    x : M
    h : Not (IsUnit x)
    ⊢ Or (Irreducible x) (Exists fun a => Exists fun b => And (Not (IsUnit a)) (An …
  -/
  haveI := Classical.dec
  /-
    M : Type u_2
    inst✝ : Monoid M
    x : M
    h : Not (IsUnit x)
    this : (p : Prop) → Decidable p
    ⊢ Or (Irreducible x) (Exists fun a => Exists fun b => And (Not (IsUnit a)) (An …
  -/
  refine or_iff_not_imp_right.2 fun H => ?_
  simp? [h, irreducible_iff] at H ⊢ says
    simp only [exists_and_left, not_exists, not_and, irreducible_iff, h, not_false_eq_true,
      true_and] at H ⊢
  /-
    M : Type u_2
    inst✝ : Monoid M
    x : M
    h : Not (IsUnit x)
    this : (p : Prop) → Decidable p
    H : ∀ (x_1 : M), Not (IsUnit x_1) → ∀ (x_2 : M), Not (IsUnit x_2) → Not (Eq (H …
    ⊢ ∀ (a b : M), Eq x (HMul.hMul a b) → Or (IsUnit a) (IsUnit b)
  -/
  refine fun a b h => by_contradiction fun o => ?_
  /-
    M : Type u_2
    inst✝ : Monoid M
    x : M
    h✝ : Not (IsUnit x)
    this : (p : Prop) → Decidable p
    H : ∀ (x_1 : M), Not (IsUnit x_1) → ∀ (x_2 : M), Not (IsUnit x_2) → Not (Eq (H …
    a b : M
    h : Eq x (HMul.hMul a b)
    o : Not (Or (IsUnit a) (IsUnit b))
    ⊢ False
  -/
  simp? [not_or] at o says simp only [not_or] at o
  /-
    M : Type u_2
    inst✝ : Monoid M
    x : M
    h✝ : Not (IsUnit x)
    this : (p : Prop) → Decidable p
    H : ∀ (x_1 : M), Not (IsUnit x_1) → ∀ (x_2 : M), Not (IsUnit x_2) → Not (Eq (H …
    a b : M
    h : Eq x (HMul.hMul a b)
    o : And (Not (IsUnit a)) (Not (IsUnit b))
    ⊢ False
  -/
  exact H _ o.1 _ o.2 h.symm
  /-
    🎉 no goals
  -/


/-- If `p` and `q` are irreducible, then `p ∣ q` implies `q ∣ p`. -/
theorem Irreducible.dvd_symm [Monoid M] {p q : M} (hp : Irreducible p) (hq : Irreducible q) :
    p ∣ q → q ∣ p := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    p q : M
    hp : Irreducible p
    hq : Irreducible q
    ⊢ Dvd.dvd p q → Dvd.dvd q p
  -/
  rintro ⟨q', rfl⟩
  /-
    case intro
    M : Type u_1
    inst✝ : Monoid M
    p : M
    hp : Irreducible p
    q' : M
    hq : Irreducible (HMul.hMul p q')
    ⊢ Dvd.dvd (HMul.hMul p q') p
  -/
  rw [IsUnit.mul_right_dvd (Or.resolve_left (of_irreducible_mul hq) hp.not_unit)]
  /-
    🎉 no goals
  -/


theorem Irreducible.dvd_comm [Monoid M] {p q : M} (hp : Irreducible p) (hq : Irreducible q) :
    p ∣ q ↔ q ∣ p :=
  ⟨hp.dvd_symm hq, hq.dvd_symm hp⟩


theorem Irreducible.prime_of_isPrimal {a : M}
    (irr : Irreducible a) (primal : IsPrimal a) : Prime a :=
  ⟨irr.ne_zero, irr.not_unit, fun a b dvd ↦ by
    /-
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      a✝ : M
      irr : Irreducible a✝
      primal : IsPrimal a✝
      a b : M
      dvd : Dvd.dvd a✝ (HMul.hMul a b)
      ⊢ Or (Dvd.dvd a✝ a) (Dvd.dvd a✝ b)
    -/
    obtain ⟨d₁, d₂, h₁, h₂, rfl⟩ := primal dvd
    /-
      case intro.intro.intro.intro
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      a b d₁ d₂ : M
      h₁ : Dvd.dvd d₁ a
      h₂ : Dvd.dvd d₂ b
      irr : Irreducible (HMul.hMul d₁ d₂)
      primal : IsPrimal (HMul.hMul d₁ d₂)
      dvd : Dvd.dvd (HMul.hMul d₁ d₂) (HMul.hMul a b)
      ⊢ Or (Dvd.dvd (HMul.hMul d₁ d₂) a) (Dvd.dvd (HMul.hMul d₁ d₂) b)
    -/
    exact (of_irreducible_mul irr).symm.imp (·.mul_right_dvd.mpr h₁) (·.mul_left_dvd.mpr h₂)⟩
    /-
      🎉 no goals
    -/


theorem Irreducible.prime [DecompositionMonoid M] {a : M} (irr : Irreducible a) : Prime a :=
  irr.prime_of_isPrimal (DecompositionMonoid.primal a)


protected theorem Prime.irreducible (hp : Prime p) : Irreducible p :=
  ⟨hp.not_unit, fun a b ↦ by
    /-
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : M
      hp : Prime p
      a b : M
      ⊢ Eq p (HMul.hMul a b) → Or (IsUnit a) (IsUnit b)
    -/
    rintro rfl
    exact (hp.dvd_or_dvd dvd_rfl).symm.imp
      (isUnit_of_dvd_one <| (mul_dvd_mul_iff_right <| right_ne_zero_of_mul hp.ne_zero).mp <|
        dvd_mul_of_dvd_right · _)
      (isUnit_of_dvd_one <| (mul_dvd_mul_iff_left <| left_ne_zero_of_mul hp.ne_zero).mp <|
        dvd_mul_of_dvd_left · _)⟩


theorem irreducible_iff_prime [DecompositionMonoid M] {a : M} : Irreducible a ↔ Prime a :=
  ⟨Irreducible.prime, Prime.irreducible⟩


