/-- An element of the unit group of a nonzero monoid with zero represented as an element
    of the monoid is nonzero. -/
@[simp]
theorem ne_zero [Nontrivial M₀] (u : M₀ˣ) : (u : M₀) ≠ 0 :=
  left_ne_zero_of_mul_eq_one u.mul_inv

-- We can't use `mul_eq_zero` + `Units.ne_zero` in the next two lemmas because we don't assume
-- `Nonzero M₀`.

@[simp]
theorem mul_left_eq_zero (u : M₀ˣ) {a : M₀} : a * u = 0 ↔ a = 0 :=
               /-
                 M₀ : Type u_2
                 inst✝ : MonoidWithZero M₀
                 u : Units M₀
                 a : M₀
                 h : Eq (HMul.hMul a ↑u) 0
                 ⊢ Eq a 0
               -/
  ⟨fun h => by simpa using mul_eq_zero_of_left h ↑u⁻¹, fun h => mul_eq_zero_of_left h u⟩
               /-
                 🎉 no goals
               -/


@[simp]
theorem mul_right_eq_zero (u : M₀ˣ) {a : M₀} : ↑u * a = 0 ↔ a = 0 :=
               /-
                 M₀ : Type u_2
                 inst✝ : MonoidWithZero M₀
                 u : Units M₀
                 a : M₀
                 h : Eq (HMul.hMul (↑u) a) 0
                 ⊢ Eq a 0
               -/
  ⟨fun h => by simpa using mul_eq_zero_of_right (↑u⁻¹) h, mul_eq_zero_of_right (u : M₀)⟩
               /-
                 🎉 no goals
               -/


theorem ne_zero [Nontrivial M₀] {a : M₀} (ha : IsUnit a) : a ≠ 0 :=
  let ⟨u, hu⟩ := ha
  hu ▸ u.ne_zero


theorem mul_right_eq_zero {a b : M₀} (ha : IsUnit a) : a * b = 0 ↔ b = 0 :=
  let ⟨u, hu⟩ := ha
  hu ▸ u.mul_right_eq_zero


theorem mul_left_eq_zero {a b : M₀} (hb : IsUnit b) : a * b = 0 ↔ a = 0 :=
  let ⟨u, hu⟩ := hb
  hu ▸ u.mul_left_eq_zero


@[simp]
theorem isUnit_zero_iff : IsUnit (0 : M₀) ↔ (0 : M₀) = 1 :=
                                                /-
                                                  M₀ : Type u_2
                                                  inst✝ : MonoidWithZero M₀
                                                  x✝ : IsUnit 0
                                                  a : M₀
                                                  a0 : Eq (HMul.hMul 0 a) 1
                                                  inv_val✝ : Eq (HMul.hMul a 0) 1
                                                  ⊢ Eq 0 1
                                                -/
  ⟨fun ⟨⟨_, a, (a0 : 0 * a = 1), _⟩, rfl⟩ => by rwa [zero_mul] at a0, fun h =>
                                                /-
                                                  🎉 no goals
                                                -/
    @isUnit_of_subsingleton _ _ (subsingleton_of_zero_eq_one h) 0⟩


theorem not_isUnit_zero [Nontrivial M₀] : ¬IsUnit (0 : M₀) :=
  mt isUnit_zero_iff.1 zero_ne_one


open Classical in
/-- Introduce a function `inverse` on a monoid with zero `M₀`, which sends `x` to `x⁻¹` if `x` is
invertible and to `0` otherwise.  This definition is somewhat ad hoc, but one needs a fully (rather
than partially) defined inverse function for some purposes, including for calculus.

Note that while this is in the `Ring` namespace for brevity, it requires the weaker assumption
`MonoidWithZero M₀` instead of `Ring M₀`. -/
noncomputable def inverse : M₀ → M₀ := fun x => if h : IsUnit x then ((h.unit⁻¹ : M₀ˣ) : M₀) else 0


/-- By definition, if `x` is invertible then `inverse x = x⁻¹`. -/
@[simp]
theorem inverse_unit (u : M₀ˣ) : inverse (u : M₀) = (u⁻¹ : M₀ˣ) := by
  /-
    M₀ : Type u_2
    inst✝ : MonoidWithZero M₀
    u : Units M₀
    ⊢ Eq (Ring.inverse ↑u) ↑(Inv.inv u)
  -/
  rw [inverse, dif_pos u.isUnit, IsUnit.unit_of_val_units]
  /-
    🎉 no goals
  -/


theorem IsUnit.ringInverse {x : M₀} (h : IsUnit x) : IsUnit (inverse x) :=
  match h with
  | ⟨u, hu⟩ => hu ▸ ⟨u⁻¹, (inverse_unit u).symm⟩


theorem inverse_of_isUnit {x : M₀} (h : IsUnit x) : inverse x = ((h.unit⁻¹ : M₀ˣ) : M₀) := dif_pos h


/-- By definition, if `x` is not invertible then `inverse x = 0`. -/
@[simp]
theorem inverse_non_unit (x : M₀) (h : ¬IsUnit x) : inverse x = 0 :=
  dif_neg h


theorem mul_inverse_cancel (x : M₀) (h : IsUnit x) : x * inverse x = 1 := by
  /-
    M₀ : Type u_2
    inst✝ : MonoidWithZero M₀
    x : M₀
    h : IsUnit x
    ⊢ Eq (HMul.hMul x (Ring.inverse x)) 1
  -/
  rcases h with ⟨u, rfl⟩
  /-
    case intro
    M₀ : Type u_2
    inst✝ : MonoidWithZero M₀
    u : Units M₀
    ⊢ Eq (HMul.hMul (↑u) (Ring.inverse ↑u)) 1
  -/
  rw [inverse_unit, Units.mul_inv]
  /-
    🎉 no goals
  -/


theorem inverse_mul_cancel (x : M₀) (h : IsUnit x) : inverse x * x = 1 := by
  /-
    M₀ : Type u_2
    inst✝ : MonoidWithZero M₀
    x : M₀
    h : IsUnit x
    ⊢ Eq (HMul.hMul (Ring.inverse x) x) 1
  -/
  rcases h with ⟨u, rfl⟩
  /-
    case intro
    M₀ : Type u_2
    inst✝ : MonoidWithZero M₀
    u : Units M₀
    ⊢ Eq (HMul.hMul (Ring.inverse ↑u) ↑u) 1
  -/
  rw [inverse_unit, Units.inv_mul]
  /-
    🎉 no goals
  -/


theorem mul_inverse_cancel_right (x y : M₀) (h : IsUnit x) : y * x * inverse x = y := by
  /-
    M₀ : Type u_2
    inst✝ : MonoidWithZero M₀
    x y : M₀
    h : IsUnit x
    ⊢ Eq (HMul.hMul (HMul.hMul y x) (Ring.inverse x)) y
  -/
  rw [mul_assoc, mul_inverse_cancel x h, mul_one]
  /-
    🎉 no goals
  -/


theorem inverse_mul_cancel_right (x y : M₀) (h : IsUnit x) : y * inverse x * x = y := by
  /-
    M₀ : Type u_2
    inst✝ : MonoidWithZero M₀
    x y : M₀
    h : IsUnit x
    ⊢ Eq (HMul.hMul (HMul.hMul y (Ring.inverse x)) x) y
  -/
  rw [mul_assoc, inverse_mul_cancel x h, mul_one]
  /-
    🎉 no goals
  -/


theorem mul_inverse_cancel_left (x y : M₀) (h : IsUnit x) : x * (inverse x * y) = y := by
  /-
    M₀ : Type u_2
    inst✝ : MonoidWithZero M₀
    x y : M₀
    h : IsUnit x
    ⊢ Eq (HMul.hMul x (HMul.hMul (Ring.inverse x) y)) y
  -/
  rw [← mul_assoc, mul_inverse_cancel x h, one_mul]
  /-
    🎉 no goals
  -/


theorem inverse_mul_cancel_left (x y : M₀) (h : IsUnit x) : inverse x * (x * y) = y := by
  /-
    M₀ : Type u_2
    inst✝ : MonoidWithZero M₀
    x y : M₀
    h : IsUnit x
    ⊢ Eq (HMul.hMul (Ring.inverse x) (HMul.hMul x y)) y
  -/
  rw [← mul_assoc, inverse_mul_cancel x h, one_mul]
  /-
    🎉 no goals
  -/


theorem inverse_mul_eq_iff_eq_mul (x y z : M₀) (h : IsUnit x) : inverse x * y = z ↔ y = x * z :=
                /-
                  M₀ : Type u_2
                  inst✝ : MonoidWithZero M₀
                  x y z : M₀
                  h : IsUnit x
                  h1 : Eq (HMul.hMul (Ring.inverse x) y) z
                  ⊢ Eq y (HMul.hMul x z)
                -/
  ⟨fun h1 => by rw [← h1, mul_inverse_cancel_left _ _ h],
                /-
                  🎉 no goals
                -/
               /-
                 M₀ : Type u_2
                 inst✝ : MonoidWithZero M₀
                 x y z : M₀
                 h : IsUnit x
                 h1 : Eq y (HMul.hMul x z)
                 ⊢ Eq (HMul.hMul (Ring.inverse x) y) z
               -/
  fun h1 => by rw [h1, inverse_mul_cancel_left _ _ h]⟩
               /-
                 🎉 no goals
               -/


theorem eq_mul_inverse_iff_mul_eq (x y z : M₀) (h : IsUnit z) : x = y * inverse z ↔ x * z = y :=
                /-
                  M₀ : Type u_2
                  inst✝ : MonoidWithZero M₀
                  x y z : M₀
                  h : IsUnit z
                  h1 : Eq x (HMul.hMul y (Ring.inverse z))
                  ⊢ Eq (HMul.hMul x z) y
                -/
  ⟨fun h1 => by rw [h1, inverse_mul_cancel_right _ _ h],
                /-
                  🎉 no goals
                -/
               /-
                 M₀ : Type u_2
                 inst✝ : MonoidWithZero M₀
                 x y z : M₀
                 h : IsUnit z
                 h1 : Eq (HMul.hMul x z) y
                 ⊢ Eq x (HMul.hMul y (Ring.inverse z))
               -/
  fun h1 => by rw [← h1, mul_inverse_cancel_right _ _ h]⟩
               /-
                 🎉 no goals
               -/


@[simp]
theorem inverse_one : inverse (1 : M₀) = 1 :=
  inverse_unit 1


@[simp]
theorem inverse_zero : inverse (0 : M₀) = 0 := by
  /-
    M₀ : Type u_2
    inst✝ : MonoidWithZero M₀
    ⊢ Eq (Ring.inverse 0) 0
  -/
  nontriviality
  /-
    M₀ : Type u_2
    inst✝ : MonoidWithZero M₀
    a✝ : Nontrivial M₀
    ⊢ Eq (Ring.inverse 0) 0
  -/
  exact inverse_non_unit _ not_isUnit_zero
  /-
    🎉 no goals
  -/


theorem IsUnit.ring_inverse {a : M₀} : IsUnit a → IsUnit (Ring.inverse a)
  | ⟨u, hu⟩ => hu ▸ ⟨u⁻¹, (Ring.inverse_unit u).symm⟩


@[simp]
theorem isUnit_ring_inverse {a : M₀} : IsUnit (Ring.inverse a) ↔ IsUnit a :=
  ⟨fun h => by
    /-
      M₀ : Type u_2
      inst✝ : MonoidWithZero M₀
      a : M₀
      h : IsUnit (Ring.inverse a)
      ⊢ IsUnit a
    -/
    cases subsingleton_or_nontrivial M₀
      /-
        case inl
        M₀ : Type u_2
        inst✝ : MonoidWithZero M₀
        a : M₀
        h : IsUnit (Ring.inverse a)
        h✝ : Subsingleton M₀
        ⊢ IsUnit a
      -/
    · convert h
      /-
        🎉 no goals
      -/
      /-
        case inr
        M₀ : Type u_2
        inst✝ : MonoidWithZero M₀
        a : M₀
        h : IsUnit (Ring.inverse a)
        h✝ : Nontrivial M₀
        ⊢ IsUnit a
      -/
    · contrapose h
      /-
        case inr
        M₀ : Type u_2
        inst✝ : MonoidWithZero M₀
        a : M₀
        h✝ : Nontrivial M₀
        h : Not (IsUnit a)
        ⊢ Not (IsUnit (Ring.inverse a))
      -/
      rw [Ring.inverse_non_unit _ h]
      /-
        case inr
        M₀ : Type u_2
        inst✝ : MonoidWithZero M₀
        a : M₀
        h✝ : Nontrivial M₀
        h : Not (IsUnit a)
        ⊢ Not (IsUnit 0)
      -/
      exact not_isUnit_zero
      /-
        🎉 no goals
      -/
      ,
    IsUnit.ring_inverse⟩


/-- Embed a non-zero element of a `GroupWithZero` into the unit group.
  By combining this function with the operations on units,
  or the `/ₚ` operation, it is possible to write a division
  as a partial function with three arguments. -/
def mk0 (a : G₀) (ha : a ≠ 0) : G₀ˣ :=
  ⟨a, a⁻¹, mul_inv_cancel₀ ha, inv_mul_cancel₀ ha⟩


@[simp]
theorem mk0_one (h := one_ne_zero) : mk0 (1 : G₀) h = 1 := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    h : optParam (Ne 1 0) ⋯
    ⊢ Eq (Units.mk0 1 h) 1
  -/
  ext
  /-
    case a
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    h : optParam (Ne 1 0) ⋯
    ⊢ Eq ↑(Units.mk0 1 h) ↑1
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem val_mk0 {a : G₀} (h : a ≠ 0) : (mk0 a h : G₀) = a :=
  rfl


@[simp]
theorem mk0_val (u : G₀ˣ) (h : (u : G₀) ≠ 0) : mk0 (u : G₀) h = u :=
  Units.ext rfl


theorem mul_inv' (u : G₀ˣ) : u * (u : G₀)⁻¹ = 1 :=
  mul_inv_cancel₀ u.ne_zero


theorem inv_mul' (u : G₀ˣ) : (u⁻¹ : G₀) * u = 1 :=
  inv_mul_cancel₀ u.ne_zero


@[simp]
theorem mk0_inj {a b : G₀} (ha : a ≠ 0) (hb : b ≠ 0) : Units.mk0 a ha = Units.mk0 b hb ↔ a = b :=
               /-
                 G₀ : Type u_3
                 inst✝ : GroupWithZero G₀
                 a b : G₀
                 ha : Ne a 0
                 hb : Ne b 0
                 h : Eq (Units.mk0 a ha) (Units.mk0 b hb)
                 ⊢ Eq a b
               -/
  ⟨fun h => by injection h, fun h => Units.ext h⟩
               /-
                 🎉 no goals
               -/


/-- In a group with zero, an existential over a unit can be rewritten in terms of `Units.mk0`. -/
theorem exists0 {p : G₀ˣ → Prop} : (∃ g : G₀ˣ, p g) ↔ ∃ (g : G₀) (hg : g ≠ 0), p (Units.mk0 g hg) :=
  ⟨fun ⟨g, pg⟩ => ⟨g, g.ne_zero, (g.mk0_val g.ne_zero).symm ▸ pg⟩,
  fun ⟨g, hg, pg⟩ => ⟨Units.mk0 g hg, pg⟩⟩


/-- An alternative version of `Units.exists0`. This one is useful if Lean cannot
figure out `p` when using `Units.exists0` from right to left. -/
theorem exists0' {p : ∀ g : G₀, g ≠ 0 → Prop} :
    (∃ (g : G₀) (hg : g ≠ 0), p g hg) ↔ ∃ g : G₀ˣ, p g g.ne_zero :=
                /-
                  G₀ : Type u_3
                  inst✝ : GroupWithZero G₀
                  p : (g : G₀) → Ne g 0 → Prop
                  ⊢ Iff (Exists fun g => Exists fun hg => p g hg) (Exists fun g => Exists fun hg …
                -/
  Iff.trans (by simp_rw [val_mk0]) exists0.symm
                /-
                  🎉 no goals
                -/
  -- Porting note: had to add the `rfl`


@[simp]
theorem exists_iff_ne_zero {p : G₀ → Prop} : (∃ u : G₀ˣ, p u) ↔ ∃ x ≠ 0, p x := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    p : G₀ → Prop
    ⊢ Iff (Exists fun u => p ↑u) (Exists fun x => And (Ne x 0) (p x))
  -/
  simp [exists0]
  /-
    🎉 no goals
  -/


theorem _root_.GroupWithZero.eq_zero_or_unit (a : G₀) : a = 0 ∨ ∃ u : G₀ˣ, a = u := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a : G₀
    ⊢ Or (Eq a 0) (Exists fun u => Eq a ↑u)
  -/
  simpa using em _
  /-
    🎉 no goals
  -/


theorem IsUnit.mk0 (x : G₀) (hx : x ≠ 0) : IsUnit x :=
  (Units.mk0 x hx).isUnit


@[simp]
theorem isUnit_iff_ne_zero : IsUnit a ↔ a ≠ 0 :=
                                                      /-
                                                        G₀ : Type u_3
                                                        inst✝ : GroupWithZero G₀
                                                        a : G₀
                                                        ⊢ Iff (Exists fun x => And (Ne x 0) (Eq x a)) (Ne a 0)
                                                      -/
  (Units.exists_iff_ne_zero (p := (· = a))).trans (by simp)
                                                      /-
                                                        🎉 no goals
                                                      -/


alias ⟨_, Ne.isUnit⟩ := isUnit_iff_ne_zero

-- Porting note: can't add this attribute?
-- https://github.com/leanprover-community/mathlib4/issues/740
-- attribute [protected] Ne.is_unit

-- see Note [lower instance priority]

instance (priority := 10) GroupWithZero.noZeroDivisors : NoZeroDivisors G₀ :=
  { (‹_› : GroupWithZero G₀) with
    eq_zero_or_eq_zero_of_mul_eq_zero := @fun a b h => by
      /-
        α : Type u_1
        M₀ : Type u_2
        G₀ : Type u_3
        inst✝¹ : MonoidWithZero M₀
        inst✝ : GroupWithZero G₀
        a✝ b✝ c : G₀
        m n : Nat
        a b : G₀
        h : Eq (HMul.hMul a b) 0
        ⊢ Or (Eq a 0) (Eq b 0)
      -/
      contrapose! h
      /-
        α : Type u_1
        M₀ : Type u_2
        G₀ : Type u_3
        inst✝¹ : MonoidWithZero M₀
        inst✝ : GroupWithZero G₀
        a✝ b✝ c : G₀
        m n : Nat
        a b : G₀
        h : And (Ne a 0) (Ne b 0)
        ⊢ Ne (HMul.hMul a b) 0
      -/
      exact (Units.mk0 a h.1 * Units.mk0 b h.2).ne_zero }
      /-
        🎉 no goals
      -/

-- Can't be put next to the other `mk0` lemmas because it depends on the
-- `NoZeroDivisors` instance, which depends on `mk0`.

@[simp]
theorem Units.mk0_mul (x y : G₀) (hxy) :
    Units.mk0 (x * y) hxy =
      Units.mk0 x (mul_ne_zero_iff.mp hxy).1 * Units.mk0 y (mul_ne_zero_iff.mp hxy).2 := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    x y : G₀
    hxy : Ne (HMul.hMul x y) 0
    ⊢ Eq (Units.mk0 (HMul.hMul x y) hxy) (HMul.hMul (Units.mk0 x ⋯) (Units.mk0 y ⋯))
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


theorem div_ne_zero (ha : a ≠ 0) (hb : b ≠ 0) : a / b ≠ 0 := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a b : G₀
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Ne (HDiv.hDiv a b) 0
  -/
  rw [div_eq_mul_inv]
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a b : G₀
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Ne (HMul.hMul a (Inv.inv b)) 0
  -/
  exact mul_ne_zero ha (inv_ne_zero hb)
  /-
    🎉 no goals
  -/


@[simp]
                                                          /-
                                                            G₀ : Type u_3
                                                            inst✝ : GroupWithZero G₀
                                                            a b : G₀
                                                            ⊢ Iff (Eq (HDiv.hDiv a b) 0) (Or (Eq a 0) (Eq b 0))
                                                          -/
theorem div_eq_zero_iff : a / b = 0 ↔ a = 0 ∨ b = 0 := by simp [div_eq_mul_inv]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem div_ne_zero_iff : a / b ≠ 0 ↔ a ≠ 0 ∧ b ≠ 0 :=
  div_eq_zero_iff.not.trans not_or


@[simp] lemma div_self (h : a ≠ 0) : a / a = 1 := h.isUnit.div_self


lemma eq_mul_inv_iff_mul_eq₀ (hc : c ≠ 0) : a = b * c⁻¹ ↔ a * c = b :=
  hc.isUnit.eq_mul_inv_iff_mul_eq


lemma eq_inv_mul_iff_mul_eq₀ (hb : b ≠ 0) : a = b⁻¹ * c ↔ b * a = c :=
  hb.isUnit.eq_inv_mul_iff_mul_eq


lemma inv_mul_eq_iff_eq_mul₀ (ha : a ≠ 0) : a⁻¹ * b = c ↔ b = a * c :=
  ha.isUnit.inv_mul_eq_iff_eq_mul


lemma mul_inv_eq_iff_eq_mul₀ (hb : b ≠ 0) : a * b⁻¹ = c ↔ a = c * b :=
  hb.isUnit.mul_inv_eq_iff_eq_mul


lemma mul_inv_eq_one₀ (hb : b ≠ 0) : a * b⁻¹ = 1 ↔ a = b := hb.isUnit.mul_inv_eq_one


lemma inv_mul_eq_one₀ (ha : a ≠ 0) : a⁻¹ * b = 1 ↔ a = b := ha.isUnit.inv_mul_eq_one


lemma mul_eq_one_iff_eq_inv₀ (hb : b ≠ 0) : a * b = 1 ↔ a = b⁻¹ := hb.isUnit.mul_eq_one_iff_eq_inv


lemma mul_eq_one_iff_inv_eq₀ (ha : a ≠ 0) : a * b = 1 ↔ a⁻¹ = b := ha.isUnit.mul_eq_one_iff_inv_eq


/-- A variant of `eq_mul_inv_iff_mul_eq₀` that moves the nonzero hypothesis to another variable. -/
lemma mul_eq_of_eq_mul_inv₀ (ha : a ≠ 0) (h : a = c * b⁻¹) : a * b = c := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a b c : G₀
    ha : Ne a 0
    h : Eq a (HMul.hMul c (Inv.inv b))
    ⊢ Eq (HMul.hMul a b) c
  -/
  rwa [← eq_mul_inv_iff_mul_eq₀]; rintro rfl; simp [ha] at h
                                              /-
                                                🎉 no goals
                                              -/


/-- A variant of `eq_inv_mul_iff_mul_eq₀` that moves the nonzero hypothesis to another variable. -/
lemma mul_eq_of_eq_inv_mul₀ (hb : b ≠ 0) (h : b = a⁻¹ * c) : a * b = c := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a b c : G₀
    hb : Ne b 0
    h : Eq b (HMul.hMul (Inv.inv a) c)
    ⊢ Eq (HMul.hMul a b) c
  -/
  rwa [← eq_inv_mul_iff_mul_eq₀]; rintro rfl; simp [hb] at h
                                              /-
                                                🎉 no goals
                                              -/


/-- A variant of `inv_mul_eq_iff_eq_mul₀` that moves the nonzero hypothesis to another variable. -/
lemma eq_mul_of_inv_mul_eq₀ (hc : c ≠ 0) (h : b⁻¹ * a = c) : a = b * c := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a b c : G₀
    hc : Ne c 0
    h : Eq (HMul.hMul (Inv.inv b) a) c
    ⊢ Eq a (HMul.hMul b c)
  -/
  rwa [← inv_mul_eq_iff_eq_mul₀]; rintro rfl; simp [hc.symm] at h
                                              /-
                                                🎉 no goals
                                              -/


/-- A variant of `mul_inv_eq_iff_eq_mul₀` that moves the nonzero hypothesis to another variable. -/
lemma eq_mul_of_mul_inv_eq₀ (hb : b ≠ 0) (h : a * c⁻¹ = b) : a = b * c := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a b c : G₀
    hb : Ne b 0
    h : Eq (HMul.hMul a (Inv.inv c)) b
    ⊢ Eq a (HMul.hMul b c)
  -/
  rwa [← mul_inv_eq_iff_eq_mul₀]; rintro rfl; simp [hb.symm] at h
                                              /-
                                                🎉 no goals
                                              -/


@[simp] lemma div_mul_cancel₀ (a : G₀) (h : b ≠ 0) : a / b * b = a := h.isUnit.div_mul_cancel _


lemma mul_one_div_cancel (h : a ≠ 0) : a * (1 / a) = 1 := h.isUnit.mul_one_div_cancel


lemma one_div_mul_cancel (h : a ≠ 0) : 1 / a * a = 1 := h.isUnit.one_div_mul_cancel


lemma div_left_inj' (hc : c ≠ 0) : a / c = b / c ↔ a = b := hc.isUnit.div_left_inj


@[field_simps] lemma div_eq_iff (hb : b ≠ 0) : a / b = c ↔ a = c * b := hb.isUnit.div_eq_iff


@[field_simps] lemma eq_div_iff (hb : b ≠ 0) : c = a / b ↔ c * b = a := hb.isUnit.eq_div_iff

-- TODO: Swap RHS around

lemma div_eq_iff_mul_eq (hb : b ≠ 0) : a / b = c ↔ c * b = a := hb.isUnit.div_eq_iff.trans eq_comm


lemma eq_div_iff_mul_eq (hc : c ≠ 0) : a = b / c ↔ a * c = b := hc.isUnit.eq_div_iff


lemma div_eq_of_eq_mul (hb : b ≠ 0) : a = c * b → a / b = c := hb.isUnit.div_eq_of_eq_mul


lemma eq_div_of_mul_eq (hc : c ≠ 0) : a * c = b → a = b / c := hc.isUnit.eq_div_of_mul_eq


lemma div_eq_one_iff_eq (hb : b ≠ 0) : a / b = 1 ↔ a = b := hb.isUnit.div_eq_one_iff_eq


lemma div_mul_cancel_right₀ (hb : b ≠ 0) (a : G₀) : b / (a * b) = a⁻¹ :=
  hb.isUnit.div_mul_cancel_right _


set_option linter.deprecated false in
@[deprecated div_mul_cancel_right₀ (since := "2024-03-20")]
lemma div_mul_left (hb : b ≠ 0) : b / (a * b) = 1 / a := hb.isUnit.div_mul_left


lemma mul_div_mul_right (a b : G₀) (hc : c ≠ 0) : a * c / (b * c) = a / b :=
  hc.isUnit.mul_div_mul_right _ _

-- TODO: Duplicate of `mul_inv_cancel_right₀`

lemma mul_mul_div (a : G₀) (hb : b ≠ 0) : a = a * b * (1 / b) := (hb.isUnit.mul_mul_div _).symm


lemma div_div_div_cancel_right₀ (hc : c ≠ 0) (a b : G₀) : a / c / (b / c) = a / b := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    c : G₀
    hc : Ne c 0
    a b : G₀
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv a c) (HDiv.hDiv b c)) (HDiv.hDiv a b)
  -/
  rw [div_div_eq_mul_div, div_mul_cancel₀ _ hc]
  /-
    🎉 no goals
  -/


lemma div_mul_div_cancel₀ (hb : b ≠ 0) : a / b * (b / c) = a / c := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a b c : G₀
    hb : Ne b 0
    ⊢ Eq (HMul.hMul (HDiv.hDiv a b) (HDiv.hDiv b c)) (HDiv.hDiv a c)
  -/
  rw [← mul_div_assoc, div_mul_cancel₀ _ hb]
  /-
    🎉 no goals
  -/


lemma div_mul_cancel_of_imp (h : b = 0 → a = 0) : a / b * b = a := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a b : G₀
    h : Eq b 0 → Eq a 0
    ⊢ Eq (HMul.hMul (HDiv.hDiv a b) b) a
  -/
                                       /-
                                         🎉 no goals
                                       -/
  obtain rfl | hb := eq_or_ne b 0 <;>  simp [*]
                                       /-
                                         🎉 no goals
                                       -/


lemma mul_div_cancel_of_imp (h : b = 0 → a = 0) : a * b / b = a := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a b : G₀
    h : Eq b 0 → Eq a 0
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b) b) a
  -/
                                       /-
                                         🎉 no goals
                                       -/
  obtain rfl | hb := eq_or_ne b 0 <;>  simp [*]
                                       /-
                                         🎉 no goals
                                       -/


@[simp] lemma divp_mk0 (a : G₀) (hb : b ≠ 0) : a /ₚ Units.mk0 b hb = a / b := divp_eq_div _ _


lemma pow_sub₀ (a : G₀) (ha : a ≠ 0) (h : n ≤ m) : a ^ (m - n) = a ^ m * (a ^ n)⁻¹ := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    m n : Nat
    a : G₀
    ha : Ne a 0
    h : LE.le n m
    ⊢ Eq (HPow.hPow a (HSub.hSub m n)) (HMul.hMul (HPow.hPow a m) (Inv.inv (HPow.h …
  -/
  have h1 : m - n + n = m := Nat.sub_add_cancel h
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    m n : Nat
    a : G₀
    ha : Ne a 0
    h : LE.le n m
    h1 : Eq (HAdd.hAdd (HSub.hSub m n) n) m
    ⊢ Eq (HPow.hPow a (HSub.hSub m n)) (HMul.hMul (HPow.hPow a m) (Inv.inv (HPow.h …
  -/
  have h2 : a ^ (m - n) * a ^ n = a ^ m := by rw [← pow_add, h1]
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    m n : Nat
    a : G₀
    ha : Ne a 0
    h : LE.le n m
    h1 : Eq (HAdd.hAdd (HSub.hSub m n) n) m
    h2 : Eq (HMul.hMul (HPow.hPow a (HSub.hSub m n)) (HPow.hPow a n)) (HPow.hPow a …
    ⊢ Eq (HPow.hPow a (HSub.hSub m n)) (HMul.hMul (HPow.hPow a m) (Inv.inv (HPow.h …
  -/
  simpa only [div_eq_mul_inv] using eq_div_of_mul_eq (pow_ne_zero _ ha) h2
  /-
    🎉 no goals
  -/


lemma pow_sub_of_lt (a : G₀) (h : n < m) : a ^ (m - n) = a ^ m * (a ^ n)⁻¹ := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    m n : Nat
    a : G₀
    h : LT.lt n m
    ⊢ Eq (HPow.hPow a (HSub.hSub m n)) (HMul.hMul (HPow.hPow a m) (Inv.inv (HPow.h …
  -/
  obtain rfl | ha := eq_or_ne a 0
    /-
      case inl
      G₀ : Type u_3
      inst✝ : GroupWithZero G₀
      m n : Nat
      h : LT.lt n m
      ⊢ Eq (HPow.hPow 0 (HSub.hSub m n)) (HMul.hMul (HPow.hPow 0 m) (Inv.inv (HPow.h …
    -/
  · rw [zero_pow (Nat.ne_of_gt <| Nat.sub_pos_of_lt h), zero_pow (by omega), zero_mul]
    /-
      🎉 no goals
    -/
    /-
      case inr
      G₀ : Type u_3
      inst✝ : GroupWithZero G₀
      m n : Nat
      a : G₀
      h : LT.lt n m
      ha : Ne a 0
      ⊢ Eq (HPow.hPow a (HSub.hSub m n)) (HMul.hMul (HPow.hPow a m) (Inv.inv (HPow.h …
    -/
  · exact pow_sub₀ _ ha <| Nat.le_of_lt h
    /-
      🎉 no goals
    -/


lemma inv_pow_sub₀ (ha : a ≠ 0) (h : n ≤ m) : a⁻¹ ^ (m - n) = (a ^ m)⁻¹ * a ^ n := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a : G₀
    m n : Nat
    ha : Ne a 0
    h : LE.le n m
    ⊢ Eq (HPow.hPow (Inv.inv a) (HSub.hSub m n)) (HMul.hMul (Inv.inv (HPow.hPow a  …
  -/
  rw [pow_sub₀ _ (inv_ne_zero ha) h, inv_pow, inv_pow, inv_inv]
  /-
    🎉 no goals
  -/


lemma inv_pow_sub_of_lt (a : G₀) (h : n < m) : a⁻¹ ^ (m - n) = (a ^ m)⁻¹ * a ^ n := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    m n : Nat
    a : G₀
    h : LT.lt n m
    ⊢ Eq (HPow.hPow (Inv.inv a) (HSub.hSub m n)) (HMul.hMul (Inv.inv (HPow.hPow a  …
  -/
  rw [pow_sub_of_lt a⁻¹ h, inv_pow, inv_pow, inv_inv]
  /-
    🎉 no goals
  -/


lemma zpow_sub₀ (ha : a ≠ 0) (m n : ℤ) : a ^ (m - n) = a ^ m / a ^ n := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a : G₀
    ha : Ne a 0
    m n : Int
    ⊢ Eq (HPow.hPow a (HSub.hSub m n)) (HDiv.hDiv (HPow.hPow a m) (HPow.hPow a n))
  -/
  rw [Int.sub_eq_add_neg, zpow_add₀ ha, zpow_neg, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


lemma zpow_natCast_sub_natCast₀ (ha : a ≠ 0) (m n : ℕ) : a ^ (m - n : ℤ) = a ^ m / a ^ n := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a : G₀
    ha : Ne a 0
    m n : Nat
    ⊢ Eq (HPow.hPow a (HSub.hSub ↑m ↑n)) (HDiv.hDiv (HPow.hPow a m) (HPow.hPow a n))
  -/
  simpa using zpow_sub₀ ha m n
  /-
    🎉 no goals
  -/


lemma zpow_natCast_sub_one₀ (ha : a ≠ 0) (n : ℕ) : a ^ (n - 1 : ℤ) = a ^ n / a := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a : G₀
    ha : Ne a 0
    n : Nat
    ⊢ Eq (HPow.hPow a (HSub.hSub (↑n) 1)) (HDiv.hDiv (HPow.hPow a n) a)
  -/
  simpa using zpow_sub₀ ha n 1
  /-
    🎉 no goals
  -/


lemma zpow_one_sub_natCast₀ (ha : a ≠ 0) (n : ℕ) : a ^ (1 - n : ℤ) = a / a ^ n := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a : G₀
    ha : Ne a 0
    n : Nat
    ⊢ Eq (HPow.hPow a (HSub.hSub 1 ↑n)) (HDiv.hDiv a (HPow.hPow a n))
  -/
  simpa using zpow_sub₀ ha 1 n
  /-
    🎉 no goals
  -/


lemma zpow_ne_zero {a : G₀} : ∀ n : ℤ, a ≠ 0 → a ^ n ≠ 0
                  /-
                    G₀ : Type u_3
                    inst✝ : GroupWithZero G₀
                    a : G₀
                    a✝ : Nat
                    ⊢ Ne a 0 → Ne (HPow.hPow a ↑a✝) 0
                  -/
  | (_ : ℕ) => by rw [zpow_natCast]; exact pow_ne_zero _
                                     /-
                                       🎉 no goals
                                     -/
                              /-
                                G₀ : Type u_3
                                inst✝ : GroupWithZero G₀
                                a : G₀
                                n : Nat
                                ha : Ne a 0
                                ⊢ Ne (HPow.hPow a (Int.negSucc n)) 0
                              -/
  | .negSucc n => fun ha ↦ by rw [zpow_negSucc]; exact inv_ne_zero (pow_ne_zero _ ha)
                                                 /-
                                                   🎉 no goals
                                                 -/


lemma eq_zero_of_zpow_eq_zero {n : ℤ} : a ^ n = 0 → a = 0 := not_imp_not.1 (zpow_ne_zero _)


@[deprecated (since := "2024-05-07")] alias zpow_ne_zero_of_ne_zero := zpow_ne_zero

@[deprecated (since := "2024-05-07")] alias zpow_eq_zero := eq_zero_of_zpow_eq_zero


lemma zpow_eq_zero_iff {n : ℤ} (hn : n ≠ 0) : a ^ n = 0 ↔ a = 0 :=
  ⟨eq_zero_of_zpow_eq_zero, fun ha => ha.symm ▸ zero_zpow _ hn⟩


lemma zpow_ne_zero_iff {n : ℤ} (hn : n ≠ 0) : a ^ n ≠ 0 ↔ a ≠ 0 := (zpow_eq_zero_iff hn).ne


lemma zpow_neg_mul_zpow_self (n : ℤ) (ha : a ≠ 0) : a ^ (-n) * a ^ n = 1 := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a : G₀
    n : Int
    ha : Ne a 0
    ⊢ Eq (HMul.hMul (HPow.hPow a (Neg.neg n)) (HPow.hPow a n)) 1
  -/
  rw [zpow_neg]; exact inv_mul_cancel₀ (zpow_ne_zero n ha)
                 /-
                   🎉 no goals
                 -/


theorem Ring.inverse_eq_inv (a : G₀) : Ring.inverse a = a⁻¹ := by
  /-
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    a : G₀
    ⊢ Eq (Ring.inverse a) (Inv.inv a)
  -/
  obtain rfl | ha := eq_or_ne a 0
    /-
      case inl
      G₀ : Type u_3
      inst✝ : GroupWithZero G₀
      ⊢ Eq (Ring.inverse 0) (Inv.inv 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      G₀ : Type u_3
      inst✝ : GroupWithZero G₀
      a : G₀
      ha : Ne a 0
      ⊢ Eq (Ring.inverse a) (Inv.inv a)
    -/
  · exact Ring.inverse_unit (Units.mk0 a ha)
    /-
      🎉 no goals
    -/


@[simp]
theorem Ring.inverse_eq_inv' : (Ring.inverse : G₀ → G₀) = Inv.inv :=
  funext Ring.inverse_eq_inv


instance (priority := 10) CommGroupWithZero.toCancelCommMonoidWithZero :
    CancelCommMonoidWithZero G₀ :=
  { GroupWithZero.toCancelMonoidWithZero,
    CommGroupWithZero.toCommMonoidWithZero with }

-- See note [lower instance priority]

instance (priority := 100) CommGroupWithZero.toDivisionCommMonoid :
    DivisionCommMonoid G₀ where
  __ := ‹CommGroupWithZero G₀›
  __ := GroupWithZero.toDivisionMonoid


lemma div_mul_cancel_left₀ (ha : a ≠ 0) (b : G₀) : a / (a * b) = b⁻¹ :=
  ha.isUnit.div_mul_cancel_left _


@[deprecated div_mul_cancel_left₀ (since := "2024-03-22")]
lemma div_mul_right (b : G₀) (ha : a ≠ 0) : a / (a * b) = 1 / b := by
  /-
    G₀ : Type u_3
    inst✝ : CommGroupWithZero G₀
    a b : G₀
    ha : Ne a 0
    ⊢ Eq (HDiv.hDiv a (HMul.hMul a b)) (HDiv.hDiv 1 b)
  -/
  simp [div_mul_cancel_left₀ ha]
  /-
    🎉 no goals
  -/


lemma mul_div_cancel_left_of_imp (h : a = 0 → b = 0) : a * b / a = b := by
  /-
    G₀ : Type u_3
    inst✝ : CommGroupWithZero G₀
    a b : G₀
    h : Eq a 0 → Eq b 0
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b) a) b
  -/
  rw [mul_comm, mul_div_cancel_of_imp h]
  /-
    🎉 no goals
  -/


lemma mul_div_cancel_of_imp' (h : b = 0 → a = 0) : b * (a / b) = a := by
  /-
    G₀ : Type u_3
    inst✝ : CommGroupWithZero G₀
    a b : G₀
    h : Eq b 0 → Eq a 0
    ⊢ Eq (HMul.hMul b (HDiv.hDiv a b)) a
  -/
  rw [mul_comm, div_mul_cancel_of_imp h]
  /-
    🎉 no goals
  -/


lemma mul_div_cancel₀ (a : G₀) (hb : b ≠ 0) : b * (a / b) = a :=
  hb.isUnit.mul_div_cancel _


lemma mul_div_mul_left (a b : G₀) (hc : c ≠ 0) : c * a / (c * b) = a / b :=
  hc.isUnit.mul_div_mul_left _ _


lemma mul_eq_mul_of_div_eq_div (a c : G₀) (hb : b ≠ 0) (hd : d ≠ 0)
    (h : a / b = c / d) : a * d = c * b := by
  /-
    G₀ : Type u_3
    inst✝ : CommGroupWithZero G₀
    b d a c : G₀
    hb : Ne b 0
    hd : Ne d 0
    h : Eq (HDiv.hDiv a b) (HDiv.hDiv c d)
    ⊢ Eq (HMul.hMul a d) (HMul.hMul c b)
  -/
  rw [← mul_one a, ← div_self hb, ← mul_comm_div, h, div_mul_eq_mul_div, div_mul_cancel₀ _ hd]
  /-
    🎉 no goals
  -/


@[field_simps] lemma div_eq_div_iff (hb : b ≠ 0) (hd : d ≠ 0) : a / b = c / d ↔ a * d = c * b :=
  hb.isUnit.div_eq_div_iff hd.isUnit


/-- The `CommGroupWithZero` version of `div_eq_div_iff_div_eq_div`. -/
lemma div_eq_div_iff_div_eq_div' (hb : b ≠ 0) (hc : c ≠ 0) : a / b = c / d ↔ a / c = b / d := by
  /-
    G₀ : Type u_3
    inst✝ : CommGroupWithZero G₀
    a b c d : G₀
    hb : Ne b 0
    hc : Ne c 0
    ⊢ Iff (Eq (HDiv.hDiv a b) (HDiv.hDiv c d)) (Eq (HDiv.hDiv a c) (HDiv.hDiv b d))
  -/
  conv_lhs => rw [← mul_left_inj' hb, div_mul_cancel₀ _ hb]
  /-
    G₀ : Type u_3
    inst✝ : CommGroupWithZero G₀
    a b c d : G₀
    hb : Ne b 0
    hc : Ne c 0
    ⊢ Iff (Eq a (HMul.hMul (HDiv.hDiv c d) b)) (Eq (HDiv.hDiv a c) (HDiv.hDiv b d))
  -/
  conv_rhs => rw [← mul_left_inj' hc, div_mul_cancel₀ _ hc]
  /-
    G₀ : Type u_3
    inst✝ : CommGroupWithZero G₀
    a b c d : G₀
    hb : Ne b 0
    hc : Ne c 0
    ⊢ Iff (Eq a (HMul.hMul (HDiv.hDiv c d) b)) (Eq a (HMul.hMul (HDiv.hDiv b d) c))
  -/
  rw [mul_comm _ c, div_mul_eq_mul_div, mul_div_assoc]
  /-
    🎉 no goals
  -/


@[simp] lemma div_div_cancel₀ (ha : a ≠ 0) : a / (a / b) = b := ha.isUnit.div_div_cancel


@[deprecated (since := "2024-11-25")] alias div_div_cancel' := div_div_cancel₀


lemma div_div_cancel_left' (ha : a ≠ 0) : a / b / a = b⁻¹ := ha.isUnit.div_div_cancel_left


lemma div_helper (b : G₀) (h : a ≠ 0) : 1 / (a * b) * a = 1 / b := by
  /-
    G₀ : Type u_3
    inst✝ : CommGroupWithZero G₀
    a b : G₀
    h : Ne a 0
    ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HMul.hMul a b)) a) (HDiv.hDiv 1 b)
  -/
  rw [div_mul_eq_mul_div, one_mul, div_mul_cancel_left₀ h, one_div]
  /-
    🎉 no goals
  -/


lemma div_div_div_cancel_left' (a b : G₀) (hc : c ≠ 0) : c / a / (c / b) = b / a := by
  /-
    G₀ : Type u_3
    inst✝ : CommGroupWithZero G₀
    c a b : G₀
    hc : Ne c 0
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv c a) (HDiv.hDiv c b)) (HDiv.hDiv b a)
  -/
  rw [div_div_div_eq, mul_comm, mul_div_mul_right _ _ hc]
  /-
    🎉 no goals
  -/


@[simp] lemma div_mul_div_cancel₀' (ha : a ≠ 0) (b c : G₀) : a / b * (c / a) = c / b := by
  /-
    G₀ : Type u_3
    inst✝ : CommGroupWithZero G₀
    a : G₀
    ha : Ne a 0
    b c : G₀
    ⊢ Eq (HMul.hMul (HDiv.hDiv a b) (HDiv.hDiv c a)) (HDiv.hDiv c b)
  -/
  rw [mul_comm, div_mul_div_cancel₀ ha]
  /-
    🎉 no goals
  -/


open Classical in
/-- Constructs a `GroupWithZero` structure on a `MonoidWithZero`
  consisting only of units and 0. -/
noncomputable def groupWithZeroOfIsUnitOrEqZero [hM : MonoidWithZero M]
    (h : ∀ a : M, IsUnit a ∨ a = 0) : GroupWithZero M :=
  { hM with
    inv := fun a => if h0 : a = 0 then 0 else ↑((h a).resolve_right h0).unit⁻¹,
    inv_zero := dif_pos rfl,
    mul_inv_cancel := fun a h0 => by
      /-
        α : Type u_1
        M₀ : Type u_2
        G₀ : Type u_3
        inst✝¹ : MonoidWithZero M₀
        M : Type u_4
        inst✝ : Nontrivial M
        hM : MonoidWithZero M
        h : ∀ (a : M), Or (IsUnit a) (Eq a 0)
        a : M
        h0 : Ne a 0
        ⊢ Eq (HMul.hMul a (Inv.inv a)) 1
      -/
      change (a * if h0 : a = 0 then 0 else ↑((h a).resolve_right h0).unit⁻¹) = 1
      /-
        α : Type u_1
        M₀ : Type u_2
        G₀ : Type u_3
        inst✝¹ : MonoidWithZero M₀
        M : Type u_4
        inst✝ : Nontrivial M
        hM : MonoidWithZero M
        h : ∀ (a : M), Or (IsUnit a) (Eq a 0)
        a : M
        h0 : Ne a 0
        ⊢ Eq (HMul.hMul a (dite (Eq a 0) (fun h0 => 0) fun h0 => ↑(Inv.inv ⋯.unit))) 1
      -/
      rw [dif_neg h0, Units.mul_inv_eq_iff_eq_mul, one_mul, IsUnit.unit_spec],
      /-
        🎉 no goals
      -/
    exists_pair_ne := Nontrivial.exists_pair_ne }


/-- Constructs a `CommGroupWithZero` structure on a `CommMonoidWithZero`
  consisting only of units and 0. -/
noncomputable def commGroupWithZeroOfIsUnitOrEqZero [hM : CommMonoidWithZero M]
    (h : ∀ a : M, IsUnit a ∨ a = 0) : CommGroupWithZero M :=
  { groupWithZeroOfIsUnitOrEqZero h, hM with }


@[deprecated (since := "2024-03-20")] alias mul_div_cancel' := mul_div_cancel₀

