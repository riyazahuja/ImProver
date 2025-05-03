/-- Elements of the unit group of a monoid represented as elements of the monoid
    divide any element of the monoid. -/
theorem coe_dvd : ↑u ∣ a :=
                /-
                  α : Type u_1
                  inst✝ : Monoid α
                  a : α
                  u : Units α
                  ⊢ Eq a (HMul.hMul (↑u) (HMul.hMul (↑(Inv.inv u)) a))
                -/
  ⟨↑u⁻¹ * a, by simp⟩
                /-
                  🎉 no goals
                -/


/-- In a monoid, an element `a` divides an element `b` iff `a` divides all
    associates of `b`. -/
theorem dvd_mul_right : a ∣ b * u ↔ a ∣ b :=
                                         /-
                                           α : Type u_1
                                           inst✝ : Monoid α
                                           a b : α
                                           u : Units α
                                           x✝ : Dvd.dvd a (HMul.hMul b ↑u)
                                           c : α
                                           Eq : _root_.Eq (HMul.hMul b ↑u) (HMul.hMul a c)
                                           ⊢ _root_.Eq b (HMul.hMul a (HMul.hMul c ↑(Inv.inv u)))
                                         -/
  Iff.intro (fun ⟨c, Eq⟩ ↦ ⟨c * ↑u⁻¹, by rw [← mul_assoc, ← Eq, Units.mul_inv_cancel_right]⟩)
                                         /-
                                           🎉 no goals
                                         -/
    fun ⟨_, Eq⟩ ↦ Eq.symm ▸ (_root_.dvd_mul_right _ _).mul_right _


/-- In a monoid, an element `a` divides an element `b` iff all associates of `a` divide `b`. -/
theorem mul_right_dvd : a * u ∣ b ↔ a ∣ b :=
  Iff.intro (fun ⟨c, Eq⟩ => ⟨↑u * c, Eq.trans (mul_assoc _ _ _)⟩) fun h =>
                                    /-
                                      α : Type u_1
                                      inst✝ : Monoid α
                                      a b : α
                                      u : Units α
                                      h : Dvd.dvd a b
                                      ⊢ Eq (HMul.hMul (HMul.hMul a ↑u) ↑(Inv.inv u)) a
                                    -/
    dvd_trans (Dvd.intro (↑u⁻¹) (by rw [mul_assoc, u.mul_inv, mul_one])) h
                                    /-
                                      🎉 no goals
                                    -/


/-- In a commutative monoid, an element `a` divides an element `b` iff `a` divides all left
    associates of `b`. -/
theorem dvd_mul_left : a ∣ u * b ↔ a ∣ b := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    a b : α
    u : Units α
    ⊢ Iff (Dvd.dvd a (HMul.hMul (↑u) b)) (Dvd.dvd a b)
  -/
  rw [mul_comm]
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    a b : α
    u : Units α
    ⊢ Iff (Dvd.dvd a (HMul.hMul b ↑u)) (Dvd.dvd a b)
  -/
  apply dvd_mul_right
  /-
    🎉 no goals
  -/


/-- In a commutative monoid, an element `a` divides an element `b` iff all
  left associates of `a` divide `b`. -/
theorem mul_left_dvd : ↑u * a ∣ b ↔ a ∣ b := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    a b : α
    u : Units α
    ⊢ Iff (Dvd.dvd (HMul.hMul (↑u) a) b) (Dvd.dvd a b)
  -/
  rw [mul_comm]
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    a b : α
    u : Units α
    ⊢ Iff (Dvd.dvd (HMul.hMul a ↑u) b) (Dvd.dvd a b)
  -/
  apply mul_right_dvd
  /-
    🎉 no goals
  -/


/-- Units of a monoid divide any element of the monoid. -/
@[simp]
theorem dvd (hu : IsUnit u) : u ∣ a := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a u : α
    hu : IsUnit u
    ⊢ Dvd.dvd u a
  -/
  rcases hu with ⟨u, rfl⟩
  /-
    case intro
    α : Type u_1
    inst✝ : Monoid α
    a : α
    u : Units α
    ⊢ Dvd.dvd (↑u) a
  -/
  apply Units.coe_dvd
  /-
    🎉 no goals
  -/


@[simp]
theorem dvd_mul_right (hu : IsUnit u) : a ∣ b * u ↔ a ∣ b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b u : α
    hu : IsUnit u
    ⊢ Iff (Dvd.dvd a (HMul.hMul b u)) (Dvd.dvd a b)
  -/
  rcases hu with ⟨u, rfl⟩
  /-
    case intro
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    u : Units α
    ⊢ Iff (Dvd.dvd a (HMul.hMul b ↑u)) (Dvd.dvd a b)
  -/
  apply Units.dvd_mul_right
  /-
    🎉 no goals
  -/


/-- In a monoid, an element a divides an element b iff all associates of `a` divide `b`. -/
@[simp]
theorem mul_right_dvd (hu : IsUnit u) : a * u ∣ b ↔ a ∣ b := by
  /-
    α : Type u_1
    inst✝ : Monoid α
    a b u : α
    hu : IsUnit u
    ⊢ Iff (Dvd.dvd (HMul.hMul a u) b) (Dvd.dvd a b)
  -/
  rcases hu with ⟨u, rfl⟩
  /-
    case intro
    α : Type u_1
    inst✝ : Monoid α
    a b : α
    u : Units α
    ⊢ Iff (Dvd.dvd (HMul.hMul a ↑u) b) (Dvd.dvd a b)
  -/
  apply Units.mul_right_dvd
  /-
    🎉 no goals
  -/


theorem isPrimal (hu : IsUnit u) : IsPrimal u :=
  fun _ _ _ ↦ ⟨u, 1, hu.dvd, one_dvd _, (mul_one u).symm⟩


/-- In a commutative monoid, an element `a` divides an element `b` iff `a` divides all left
    associates of `b`. -/
@[simp]
theorem dvd_mul_left (hu : IsUnit u) : a ∣ u * b ↔ a ∣ b := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    a b u : α
    hu : IsUnit u
    ⊢ Iff (Dvd.dvd a (HMul.hMul u b)) (Dvd.dvd a b)
  -/
  rcases hu with ⟨u, rfl⟩
  /-
    case intro
    α : Type u_1
    inst✝ : CommMonoid α
    a b : α
    u : Units α
    ⊢ Iff (Dvd.dvd a (HMul.hMul (↑u) b)) (Dvd.dvd a b)
  -/
  apply Units.dvd_mul_left
  /-
    🎉 no goals
  -/


/-- In a commutative monoid, an element `a` divides an element `b` iff all
  left associates of `a` divide `b`. -/
@[simp]
theorem mul_left_dvd (hu : IsUnit u) : u * a ∣ b ↔ a ∣ b := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    a b u : α
    hu : IsUnit u
    ⊢ Iff (Dvd.dvd (HMul.hMul u a) b) (Dvd.dvd a b)
  -/
  rcases hu with ⟨u, rfl⟩
  /-
    case intro
    α : Type u_1
    inst✝ : CommMonoid α
    a b : α
    u : Units α
    ⊢ Iff (Dvd.dvd (HMul.hMul (↑u) a) b) (Dvd.dvd a b)
  -/
  apply Units.mul_left_dvd
  /-
    🎉 no goals
  -/


theorem isUnit_iff_dvd_one {x : α} : IsUnit x ↔ x ∣ 1 :=
                                                /-
                                                  α : Type u_1
                                                  inst✝ : CommMonoid α
                                                  x : α
                                                  x✝ : Dvd.dvd x 1
                                                  y : α
                                                  h : Eq 1 (HMul.hMul x y)
                                                  ⊢ Eq (HMul.hMul y x) 1
                                                -/
  ⟨IsUnit.dvd, fun ⟨y, h⟩ => ⟨⟨x, y, h.symm, by rw [h, mul_comm]⟩, rfl⟩⟩
                                                /-
                                                  🎉 no goals
                                                -/


theorem isUnit_iff_forall_dvd {x : α} : IsUnit x ↔ ∀ y, x ∣ y :=
  isUnit_iff_dvd_one.trans ⟨fun h _ => h.trans (one_dvd _), fun h => h _⟩


theorem isUnit_of_dvd_unit {x y : α} (xy : x ∣ y) (hu : IsUnit y) : IsUnit x :=
  isUnit_iff_dvd_one.2 <| xy.trans <| isUnit_iff_dvd_one.1 hu


theorem isUnit_of_dvd_one {a : α} (h : a ∣ 1) : IsUnit (a : α) :=
  isUnit_iff_dvd_one.mpr h


theorem not_isUnit_of_not_isUnit_dvd {a b : α} (ha : ¬IsUnit a) (hb : a ∣ b) : ¬IsUnit b :=
  mt (isUnit_of_dvd_unit hb) ha


/-- `x` and `y` are relatively prime if every common divisor is a unit. -/
def IsRelPrime [Monoid α] (x y : α) : Prop := ∀ ⦃d⦄, d ∣ x → d ∣ y → IsUnit d


@[symm] theorem IsRelPrime.symm (H : IsRelPrime x y) : IsRelPrime y x := fun _ hx hy ↦ H hy hx


theorem isRelPrime_comm : IsRelPrime x y ↔ IsRelPrime y x :=
  ⟨IsRelPrime.symm, IsRelPrime.symm⟩


theorem isRelPrime_self : IsRelPrime x x ↔ IsUnit x :=
  ⟨(· dvd_rfl dvd_rfl), fun hu _ _ dvd ↦ isUnit_of_dvd_unit dvd hu⟩


theorem IsUnit.isRelPrime_left (h : IsUnit x) : IsRelPrime x y :=
  fun _ hx _ ↦ isUnit_of_dvd_unit hx h

theorem IsUnit.isRelPrime_right (h : IsUnit y) : IsRelPrime x y := h.isRelPrime_left.symm

theorem isRelPrime_one_left : IsRelPrime 1 x := isUnit_one.isRelPrime_left

theorem isRelPrime_one_right : IsRelPrime x 1 := isUnit_one.isRelPrime_right


theorem IsRelPrime.of_mul_left_left (H : IsRelPrime (x * y) z) : IsRelPrime x z :=
  fun _ hx ↦ H (dvd_mul_of_dvd_left hx _)


theorem IsRelPrime.of_mul_left_right (H : IsRelPrime (x * y) z) : IsRelPrime y z :=
  (mul_comm x y ▸ H).of_mul_left_left


theorem IsRelPrime.of_mul_right_left (H : IsRelPrime x (y * z)) : IsRelPrime x y := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    x y z : α
    H : IsRelPrime x (HMul.hMul y z)
    ⊢ IsRelPrime x y
  -/
  rw [isRelPrime_comm] at H ⊢
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    x y z : α
    H : IsRelPrime (HMul.hMul y z) x
    ⊢ IsRelPrime y x
  -/
  exact H.of_mul_left_left
  /-
    🎉 no goals
  -/


theorem IsRelPrime.of_mul_right_right (H : IsRelPrime x (y * z)) : IsRelPrime x z :=
  (mul_comm y z ▸ H).of_mul_right_left


theorem IsRelPrime.of_dvd_left (h : IsRelPrime y z) (dvd : x ∣ y) : IsRelPrime x z := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    x y z : α
    h : IsRelPrime y z
    dvd : Dvd.dvd x y
    ⊢ IsRelPrime x z
  -/
  obtain ⟨d, rfl⟩ := dvd; exact IsRelPrime.of_mul_left_left h
                          /-
                            🎉 no goals
                          -/


theorem IsRelPrime.of_dvd_right (h : IsRelPrime z y) (dvd : x ∣ y) : IsRelPrime z x :=
  (h.symm.of_dvd_left dvd).symm


theorem IsRelPrime.isUnit_of_dvd (H : IsRelPrime x y) (d : x ∣ y) : IsUnit x := H dvd_rfl d


theorem isRelPrime_mul_unit_left_left : IsRelPrime (x * y) z ↔ IsRelPrime y z :=
  ⟨IsRelPrime.of_mul_left_right, fun H _ h ↦ H (hu.dvd_mul_left.mp h)⟩


theorem isRelPrime_mul_unit_left_right : IsRelPrime y (x * z) ↔ IsRelPrime y z := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    x y z : α
    hu : IsUnit x
    ⊢ Iff (IsRelPrime y (HMul.hMul x z)) (IsRelPrime y z)
  -/
  rw [isRelPrime_comm, isRelPrime_mul_unit_left_left hu, isRelPrime_comm]
  /-
    🎉 no goals
  -/


theorem isRelPrime_mul_unit_left : IsRelPrime (x * y) (x * z) ↔ IsRelPrime y z := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    x y z : α
    hu : IsUnit x
    ⊢ Iff (IsRelPrime (HMul.hMul x y) (HMul.hMul x z)) (IsRelPrime y z)
  -/
  rw [isRelPrime_mul_unit_left_left hu, isRelPrime_mul_unit_left_right hu]
  /-
    🎉 no goals
  -/


theorem isRelPrime_mul_unit_right_left : IsRelPrime (y * x) z ↔ IsRelPrime y z := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    x y z : α
    hu : IsUnit x
    ⊢ Iff (IsRelPrime (HMul.hMul y x) z) (IsRelPrime y z)
  -/
  rw [mul_comm, isRelPrime_mul_unit_left_left hu]
  /-
    🎉 no goals
  -/


theorem isRelPrime_mul_unit_right_right : IsRelPrime y (z * x) ↔ IsRelPrime y z := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    x y z : α
    hu : IsUnit x
    ⊢ Iff (IsRelPrime y (HMul.hMul z x)) (IsRelPrime y z)
  -/
  rw [mul_comm, isRelPrime_mul_unit_left_right hu]
  /-
    🎉 no goals
  -/


theorem isRelPrime_mul_unit_right : IsRelPrime (y * x) (z * x) ↔ IsRelPrime y z := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    x y z : α
    hu : IsUnit x
    ⊢ Iff (IsRelPrime (HMul.hMul y x) (HMul.hMul z x)) (IsRelPrime y z)
  -/
  rw [isRelPrime_mul_unit_right_left hu, isRelPrime_mul_unit_right_right hu]
  /-
    🎉 no goals
  -/


theorem IsRelPrime.dvd_of_dvd_mul_right_of_isPrimal (H1 : IsRelPrime x z) (H2 : x ∣ y * z)
    (h : IsPrimal x) : x ∣ y := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    x y z : α
    H1 : IsRelPrime x z
    H2 : Dvd.dvd x (HMul.hMul y z)
    h : IsPrimal x
    ⊢ Dvd.dvd x y
  -/
  obtain ⟨a, b, ha, hb, rfl⟩ := h H2
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝ : CommMonoid α
    y z a b : α
    ha : Dvd.dvd a y
    hb : Dvd.dvd b z
    H1 : IsRelPrime (HMul.hMul a b) z
    H2 : Dvd.dvd (HMul.hMul a b) (HMul.hMul y z)
    h : IsPrimal (HMul.hMul a b)
    ⊢ Dvd.dvd (HMul.hMul a b) y
  -/
  exact (H1.of_mul_left_right.isUnit_of_dvd hb).mul_right_dvd.mpr ha
  /-
    🎉 no goals
  -/


theorem IsRelPrime.dvd_of_dvd_mul_left_of_isPrimal (H1 : IsRelPrime x y) (H2 : x ∣ y * z)
    (h : IsPrimal x) : x ∣ z :=
  H1.dvd_of_dvd_mul_right_of_isPrimal (mul_comm y z ▸ H2) h


theorem IsRelPrime.mul_dvd_of_right_isPrimal (H : IsRelPrime x y) (H1 : x ∣ z) (H2 : y ∣ z)
    (hy : IsPrimal y) : x * y ∣ z := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    x y z : α
    H : IsRelPrime x y
    H1 : Dvd.dvd x z
    H2 : Dvd.dvd y z
    hy : IsPrimal y
    ⊢ Dvd.dvd (HMul.hMul x y) z
  -/
  obtain ⟨w, rfl⟩ := H1
  /-
    case intro
    α : Type u_1
    inst✝ : CommMonoid α
    x y : α
    H : IsRelPrime x y
    hy : IsPrimal y
    w : α
    H2 : Dvd.dvd y (HMul.hMul x w)
    ⊢ Dvd.dvd (HMul.hMul x y) (HMul.hMul x w)
  -/
  exact mul_dvd_mul_left x (H.symm.dvd_of_dvd_mul_left_of_isPrimal H2 hy)
  /-
    🎉 no goals
  -/


theorem IsRelPrime.mul_dvd_of_left_isPrimal (H : IsRelPrime x y) (H1 : x ∣ z) (H2 : y ∣ z)
    (hx : IsPrimal x) : x * y ∣ z := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    x y z : α
    H : IsRelPrime x y
    H1 : Dvd.dvd x z
    H2 : Dvd.dvd y z
    hx : IsPrimal x
    ⊢ Dvd.dvd (HMul.hMul x y) z
  -/
  rw [mul_comm]; exact H.symm.mul_dvd_of_right_isPrimal H2 H1 hx
                 /-
                   🎉 no goals
                 -/


theorem IsRelPrime.dvd_of_dvd_mul_right (H1 : IsRelPrime x z) (H2 : x ∣ y * z) : x ∣ y :=
  H1.dvd_of_dvd_mul_right_of_isPrimal H2 (DecompositionMonoid.primal x)


theorem IsRelPrime.dvd_of_dvd_mul_left (H1 : IsRelPrime x y) (H2 : x ∣ y * z) : x ∣ z :=
  H1.dvd_of_dvd_mul_right (mul_comm y z ▸ H2)


theorem IsRelPrime.mul_left (H1 : IsRelPrime x z) (H2 : IsRelPrime y z) : IsRelPrime (x * y) z :=
  fun _ h hz ↦ by
    /-
      α : Type u_1
      inst✝¹ : CommMonoid α
      x y z : α
      inst✝ : DecompositionMonoid α
      H1 : IsRelPrime x z
      H2 : IsRelPrime y z
      x✝ : α
      h : Dvd.dvd x✝ (HMul.hMul x y)
      hz : Dvd.dvd x✝ z
      ⊢ IsUnit x✝
    -/
    obtain ⟨a, b, ha, hb, rfl⟩ := exists_dvd_and_dvd_of_dvd_mul h
    /-
      case intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : CommMonoid α
      x y z : α
      inst✝ : DecompositionMonoid α
      H1 : IsRelPrime x z
      H2 : IsRelPrime y z
      a b : α
      ha : Dvd.dvd a x
      hb : Dvd.dvd b y
      h : Dvd.dvd (HMul.hMul a b) (HMul.hMul x y)
      hz : Dvd.dvd (HMul.hMul a b) z
      ⊢ IsUnit (HMul.hMul a b)
    -/
    exact (H1 ha <| (dvd_mul_right a b).trans hz).mul (H2 hb <| (dvd_mul_left b a).trans hz)
    /-
      🎉 no goals
    -/


theorem IsRelPrime.mul_right (H1 : IsRelPrime x y) (H2 : IsRelPrime x z) :
    IsRelPrime x (y * z) := by
  /-
    α : Type u_1
    inst✝¹ : CommMonoid α
    x y z : α
    inst✝ : DecompositionMonoid α
    H1 : IsRelPrime x y
    H2 : IsRelPrime x z
    ⊢ IsRelPrime x (HMul.hMul y z)
  -/
  rw [isRelPrime_comm] at H1 H2 ⊢; exact H1.mul_left H2
                                   /-
                                     🎉 no goals
                                   -/


theorem IsRelPrime.mul_left_iff : IsRelPrime (x * y) z ↔ IsRelPrime x z ∧ IsRelPrime y z :=
  ⟨fun H ↦ ⟨H.of_mul_left_left, H.of_mul_left_right⟩, fun ⟨H1, H2⟩ ↦ H1.mul_left H2⟩


theorem IsRelPrime.mul_right_iff : IsRelPrime x (y * z) ↔ IsRelPrime x y ∧ IsRelPrime x z :=
  ⟨fun H ↦ ⟨H.of_mul_right_left, H.of_mul_right_right⟩, fun ⟨H1, H2⟩ ↦ H1.mul_right H2⟩


theorem IsRelPrime.mul_dvd (H : IsRelPrime x y) (H1 : x ∣ z) (H2 : y ∣ z) : x * y ∣ z :=
  H.mul_dvd_of_left_isPrimal H1 H2 (DecompositionMonoid.primal x)


