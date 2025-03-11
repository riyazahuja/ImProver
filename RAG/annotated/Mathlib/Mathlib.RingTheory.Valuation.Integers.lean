/-- The ring of integers under a given valuation is the subring of elements with valuation ≤ 1. -/
def integer : Subring R where
  carrier := { x | v x ≤ 1 }
  one_mem' := le_of_eq v.map_one
                             /-
                               R : Type u
                               Γ₀ : Type v
                               inst✝¹ : Ring R
                               inst✝ : LinearOrderedCommGroupWithZero Γ₀
                               v : Valuation R Γ₀
                               x y : R
                               hx : Membership.mem (setOf fun x => LE.le (v x) 1) x
                               hy : Membership.mem (setOf fun x => LE.le (v x) 1) y
                               ⊢ Membership.mem (setOf fun x => LE.le (v x) 1) (HMul.hMul x y)
                             -/
  mul_mem' {x y} hx hy := by simp only [Set.mem_setOf_eq, _root_.map_mul, mul_le_one' hx hy]
                             /-
                               🎉 no goals
                             -/
                  /-
                    R : Type u
                    Γ₀ : Type v
                    inst✝¹ : Ring R
                    inst✝ : LinearOrderedCommGroupWithZero Γ₀
                    v : Valuation R Γ₀
                    ⊢ Membership.mem { carrier := setOf fun x => LE.le (v x) 1, mul_mem' := ⋯, one …
                  -/
  zero_mem' := by simp only [Set.mem_setOf_eq, _root_.map_zero, zero_le']
                  /-
                    🎉 no goals
                  -/
  add_mem' {x y} hx hy := le_trans (v.map_add x y) (max_le hx hy)
                        /-
                          R : Type u
                          Γ₀ : Type v
                          inst✝¹ : Ring R
                          inst✝ : LinearOrderedCommGroupWithZero Γ₀
                          v : Valuation R Γ₀
                          x : R
                          hx : Membership.mem { carrier := setOf fun x => LE.le (v x) 1, mul_mem' := ⋯,  …
                          ⊢ Membership.mem { carrier := setOf fun x => LE.le (v x) 1, mul_mem' := ⋯, one …
                        -/
  neg_mem' {x} hx := by simp only [Set.mem_setOf_eq] at hx; simpa only [Set.mem_setOf_eq, map_neg]
                                                            /-
                                                              🎉 no goals
                                                            -/


                                                              /-
                                                                R : Type u
                                                                Γ₀ : Type v
                                                                inst✝¹ : Ring R
                                                                inst✝ : LinearOrderedCommGroupWithZero Γ₀
                                                                v : Valuation R Γ₀
                                                                r : R
                                                                ⊢ Iff (Membership.mem v.integer r) (LE.le (v r) 1)
                                                              -/
lemma mem_integer_iff (r : R) : r ∈ v.integer ↔ v r ≤ 1 := by rfl
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- Given a valuation v : R → Γ₀ and a ring homomorphism O →+* R, we say that O is the integers of v
if f is injective, and its range is exactly `v.integer`. -/
structure Integers : Prop where
  hom_inj : Function.Injective (algebraMap O R)
  map_le_one : ∀ x, v (algebraMap O R x) ≤ 1
  exists_of_le_one : ∀ ⦃r⦄, v r ≤ 1 → ∃ x, algebraMap O R x = r

-- typeclass shortcut

instance : Algebra v.integer R :=
  Algebra.ofSubring v.integer


theorem integer.integers : v.Integers v.integer :=
  { hom_inj := Subtype.coe_injective
    map_le_one := fun r => r.2
    exists_of_le_one := fun r hr => ⟨⟨r, hr⟩, rfl⟩ }


theorem one_of_isUnit' {x : O} (hx : IsUnit x) (H : ∀ x, v (algebraMap O R x) ≤ 1) :
    v (algebraMap O R x) = 1 :=
  let ⟨u, hu⟩ := hx
  le_antisymm (H _) <| by
    rw [← v.map_one, ← (algebraMap O R).map_one, ← u.mul_inv, ← mul_one (v (algebraMap O R x)), hu,
      (algebraMap O R).map_mul, v.map_mul]
    /-
      R : Type u
      Γ₀ : Type v
      inst✝³ : CommRing R
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation R Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O R
      x : O
      hx : IsUnit x
      H : ∀ (x : O), LE.le (v ((algebraMap O R) x)) 1
      u : Units O
      hu : Eq (↑u) x
      ⊢ LE.le (HMul.hMul (v ((algebraMap O R) x)) (v ((algebraMap O R) ↑(Inv.inv u)) …
    -/
    exact mul_le_mul_left' (H (u⁻¹ : Units O)) _
    /-
      🎉 no goals
    -/


theorem one_of_isUnit (hv : Integers v O) {x : O} (hx : IsUnit x) : v (algebraMap O R x) = 1 :=
  one_of_isUnit' hx hv.map_le_one


/--
Let `O` be the integers of the valuation `v` on some commutative ring `R`. For every element `x` in
`O`, `x` is a unit in `O` if and only if the image of `x` in `R` is a unit and has valuation 1.
-/
theorem isUnit_of_one (hv : Integers v O) {x : O} (hx : IsUnit (algebraMap O R x))
    (hvx : v (algebraMap O R x) = 1) : IsUnit x :=
  let ⟨u, hu⟩ := hx
  have h1 : v u ≤ 1 := hu.symm ▸ hv.2 x
  have h2 : v (u⁻¹ : Rˣ) ≤ 1 := by
    /-
      R : Type u
      Γ₀ : Type v
      inst✝³ : CommRing R
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation R Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O R
      hv : v.Integers O
      x : O
      hx : IsUnit ((algebraMap O R) x)
      hvx : Eq (v ((algebraMap O R) x)) 1
      u : Units R
      hu : Eq (↑u) ((algebraMap O R) x)
      h1 : LE.le (v ↑u) 1
      ⊢ LE.le (v ↑(Inv.inv u)) 1
    -/
    rw [← one_mul (v _), ← hvx, ← v.map_mul, ← hu, u.mul_inv, hu, hvx, v.map_one]
    /-
      🎉 no goals
    -/
  let ⟨r1, hr1⟩ := hv.3 h1
  let ⟨r2, hr2⟩ := hv.3 h2
                       /-
                         R : Type u
                         Γ₀ : Type v
                         inst✝³ : CommRing R
                         inst✝² : LinearOrderedCommGroupWithZero Γ₀
                         v : Valuation R Γ₀
                         O : Type w
                         inst✝¹ : CommRing O
                         inst✝ : Algebra O R
                         hv : v.Integers O
                         x : O
                         hx : IsUnit ((algebraMap O R) x)
                         hvx : Eq (v ((algebraMap O R) x)) 1
                         u : Units R
                         hu : Eq (↑u) ((algebraMap O R) x)
                         h1 : LE.le (v ↑u) 1
                         h2 : LE.le (v ↑(Inv.inv u)) 1
                         r1 : O
                         hr1 : Eq ((algebraMap O R) r1) ↑u
                         r2 : O
                         hr2 : Eq ((algebraMap O R) r2) ↑(Inv.inv u)
                         ⊢ Eq ((algebraMap O R) (HMul.hMul r1 r2)) ((algebraMap O R) 1)
                       -/
  ⟨⟨r1, r2, hv.1 <| by rw [RingHom.map_mul, RingHom.map_one, hr1, hr2, Units.mul_inv],
                       /-
                         🎉 no goals
                       -/
                 /-
                   R : Type u
                   Γ₀ : Type v
                   inst✝³ : CommRing R
                   inst✝² : LinearOrderedCommGroupWithZero Γ₀
                   v : Valuation R Γ₀
                   O : Type w
                   inst✝¹ : CommRing O
                   inst✝ : Algebra O R
                   hv : v.Integers O
                   x : O
                   hx : IsUnit ((algebraMap O R) x)
                   hvx : Eq (v ((algebraMap O R) x)) 1
                   u : Units R
                   hu : Eq (↑u) ((algebraMap O R) x)
                   h1 : LE.le (v ↑u) 1
                   h2 : LE.le (v ↑(Inv.inv u)) 1
                   r1 : O
                   hr1 : Eq ((algebraMap O R) r1) ↑u
                   r2 : O
                   hr2 : Eq ((algebraMap O R) r2) ↑(Inv.inv u)
                   ⊢ Eq ((algebraMap O R) (HMul.hMul r2 r1)) ((algebraMap O R) 1)
                 -/
      hv.1 <| by rw [RingHom.map_mul, RingHom.map_one, hr1, hr2, Units.inv_mul]⟩,
                 /-
                   🎉 no goals
                 -/
    hv.1 <| hr1.trans hu⟩


theorem le_of_dvd (hv : Integers v O) {x y : O} (h : x ∣ y) :
    v (algebraMap O R y) ≤ v (algebraMap O R x) := by
  /-
    R : Type u
    Γ₀ : Type v
    inst✝³ : CommRing R
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O R
    hv : v.Integers O
    x y : O
    h : Dvd.dvd x y
    ⊢ LE.le (v ((algebraMap O R) y)) (v ((algebraMap O R) x))
  -/
  let ⟨z, hz⟩ := h
  /-
    R : Type u
    Γ₀ : Type v
    inst✝³ : CommRing R
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O R
    hv : v.Integers O
    x y : O
    h : Dvd.dvd x y
    z : O
    hz : Eq y (HMul.hMul x z)
    ⊢ LE.le (v ((algebraMap O R) y)) (v ((algebraMap O R) x))
  -/
  rw [← mul_one (v (algebraMap O R x)), hz, RingHom.map_mul, v.map_mul]
  /-
    R : Type u
    Γ₀ : Type v
    inst✝³ : CommRing R
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O R
    hv : v.Integers O
    x y : O
    h : Dvd.dvd x y
    z : O
    hz : Eq y (HMul.hMul x z)
    ⊢ LE.le (HMul.hMul (v ((algebraMap O R) x)) (v ((algebraMap O R) z))) (HMul.hM …
  -/
  exact mul_le_mul_left' (hv.2 z) _
  /-
    🎉 no goals
  -/


theorem dvd_of_le (hv : Integers v O) {x y : O}
    (h : v (algebraMap O F x) ≤ v (algebraMap O F y)) : y ∣ x :=
  by_cases
    (fun hy : algebraMap O F y = 0 =>
      have hx : x = 0 :=
        hv.1 <|
          (algebraMap O F).map_zero.symm ▸ (v.zero_iff.1 <| le_zero_iff.1 (v.map_zero ▸ hy ▸ h))
      hx.symm ▸ dvd_zero y)
    fun hy : algebraMap O F y ≠ 0 =>
    have : v ((algebraMap O F y)⁻¹ * algebraMap O F x) ≤ 1 := by
      /-
        F : Type u
        Γ₀ : Type v
        inst✝³ : Field F
        inst✝² : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation F Γ₀
        O : Type w
        inst✝¹ : CommRing O
        inst✝ : Algebra O F
        hv : v.Integers O
        x y : O
        h : LE.le (v ((algebraMap O F) x)) (v ((algebraMap O F) y))
        hy : Ne ((algebraMap O F) y) 0
        ⊢ LE.le (v (HMul.hMul (Inv.inv ((algebraMap O F) y)) ((algebraMap O F) x))) 1
      -/
      rw [← v.map_one, ← inv_mul_cancel₀ hy, v.map_mul, v.map_mul]
      /-
        F : Type u
        Γ₀ : Type v
        inst✝³ : Field F
        inst✝² : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation F Γ₀
        O : Type w
        inst✝¹ : CommRing O
        inst✝ : Algebra O F
        hv : v.Integers O
        x y : O
        h : LE.le (v ((algebraMap O F) x)) (v ((algebraMap O F) y))
        hy : Ne ((algebraMap O F) y) 0
        ⊢ LE.le (HMul.hMul (v (Inv.inv ((algebraMap O F) y))) (v ((algebraMap O F) x)) …
      -/
      exact mul_le_mul_left' h _
      /-
        🎉 no goals
      -/
    let ⟨z, hz⟩ := hv.3 this
    ⟨z, hv.1 <| ((algebraMap O F).map_mul y z).symm ▸ hz.symm ▸ (mul_inv_cancel_left₀ hy _).symm⟩


theorem dvd_iff_le (hv : Integers v O) {x y : O} :
    x ∣ y ↔ v (algebraMap O F y) ≤ v (algebraMap O F x) :=
  ⟨hv.le_of_dvd, hv.dvd_of_le⟩


theorem le_iff_dvd (hv : Integers v O) {x y : O} :
    v (algebraMap O F x) ≤ v (algebraMap O F y) ↔ y ∣ x :=
  ⟨hv.dvd_of_le, hv.le_of_dvd⟩


/--
This is the special case of `Valuation.Integers.isUnit_of_one` when the valuation is defined
over a field. Let `v` be a valuation on some field `F` and `O` be its integers. For every element
`x` in `O`, `x` is a unit in `O` if and only if the image of `x` in `F` has valuation 1.
-/
theorem isUnit_of_one' (hv : Integers v O) {x : O} (hvx : v (algebraMap O F x) = 1) : IsUnit x := by
  /-
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x : O
    hvx : Eq (v ((algebraMap O F) x)) 1
    ⊢ IsUnit x
  -/
  refine isUnit_of_one hv (IsUnit.mk0 _ ?_) hvx
  /-
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x : O
    hvx : Eq (v ((algebraMap O F) x)) 1
    ⊢ Ne ((algebraMap O F) x) 0
  -/
  simp only [← v.ne_zero_iff, hvx, ne_eq, one_ne_zero, not_false_eq_true]
  /-
    🎉 no goals
  -/


lemma isUnit_iff_valuation_eq_one (hv : Integers v O) {x : O} :
    IsUnit x ↔ v (algebraMap O F x) = 1 :=
  ⟨hv.one_of_isUnit, hv.isUnit_of_one'⟩


lemma valuation_unit (hv : Integers v O) (x : Oˣ) :
    v (algebraMap O F x) = 1 := by
  /-
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x : Units O
    ⊢ Eq (v ((algebraMap O F) ↑x)) 1
  -/
  simp [← hv.isUnit_iff_valuation_eq_one]
  /-
    🎉 no goals
  -/


lemma valuation_pos_iff_ne_zero (hv : Integers v O) {x : O} :
    0 < v (algebraMap O F x) ↔ x ≠ 0 := by
  /-
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x : O
    ⊢ Iff (LT.lt 0 (v ((algebraMap O F) x))) (Ne x 0)
  -/
  rw [← not_le]
  /-
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x : O
    ⊢ Iff (Not (LE.le (v ((algebraMap O F) x)) 0)) (Ne x 0)
  -/
  refine not_congr ?_
  /-
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x : O
    ⊢ Iff (LE.le (v ((algebraMap O F) x)) 0) (Eq x 0)
  -/
  simp [map_eq_zero_iff _ hv.hom_inj]
  /-
    🎉 no goals
  -/


theorem dvdNotUnit_iff_lt (hv : Integers v O) {x y : O} :
    DvdNotUnit x y ↔ v (algebraMap O F y) < v (algebraMap O F x) := by
  /-
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x y : O
    ⊢ Iff (DvdNotUnit x y) (LT.lt (v ((algebraMap O F) y)) (v ((algebraMap O F) x)))
  -/
  rw [lt_iff_le_not_le, hv.le_iff_dvd, hv.le_iff_dvd]
  /-
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x y : O
    ⊢ Iff (DvdNotUnit x y) (And (Dvd.dvd x y) (Not (Dvd.dvd y x)))
  -/
  refine ⟨?_, And.elim dvdNotUnit_of_dvd_of_not_dvd⟩
  /-
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x y : O
    ⊢ DvdNotUnit x y → And (Dvd.dvd x y) (Not (Dvd.dvd y x))
  -/
  rintro ⟨hx0, d, hdu, rfl⟩
  /-
    case intro.intro.intro
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x : O
    hx0 : Ne x 0
    d : O
    hdu : Not (IsUnit d)
    ⊢ And (Dvd.dvd x (HMul.hMul x d)) (Not (Dvd.dvd (HMul.hMul x d) x))
  -/
  refine ⟨⟨d, rfl⟩, ?_⟩
  /-
    case intro.intro.intro
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x : O
    hx0 : Ne x 0
    d : O
    hdu : Not (IsUnit d)
    ⊢ Not (Dvd.dvd (HMul.hMul x d) x)
  -/
  rw [hv.isUnit_iff_valuation_eq_one, ← ne_eq, ne_iff_lt_iff_le.mpr (hv.map_le_one d)] at hdu
  /-
    case intro.intro.intro
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x : O
    hx0 : Ne x 0
    d : O
    hdu : LT.lt (v ((algebraMap O F) d)) 1
    ⊢ Not (Dvd.dvd (HMul.hMul x d) x)
  -/
  rw [dvd_iff_le hv]
  /-
    case intro.intro.intro
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x : O
    hx0 : Ne x 0
    d : O
    hdu : LT.lt (v ((algebraMap O F) d)) 1
    ⊢ Not (LE.le (v ((algebraMap O F) x)) (v ((algebraMap O F) (HMul.hMul x d))))
  -/
  simp only [_root_.map_mul, not_le]
  /-
    case intro.intro.intro
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x : O
    hx0 : Ne x 0
    d : O
    hdu : LT.lt (v ((algebraMap O F) d)) 1
    ⊢ LT.lt (HMul.hMul (v ((algebraMap O F) x)) (v ((algebraMap O F) d))) (v ((alg …
  -/
  contrapose! hdu
  /-
    case intro.intro.intro
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x : O
    hx0 : Ne x 0
    d : O
    hdu : LE.le (v ((algebraMap O F) x)) (HMul.hMul (v ((algebraMap O F) x)) (v (( …
    ⊢ LE.le 1 (v ((algebraMap O F) d))
  -/
  refine one_le_of_le_mul_left₀ ?_ hdu
  /-
    case intro.intro.intro
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x : O
    hx0 : Ne x 0
    d : O
    hdu : LE.le (v ((algebraMap O F) x)) (HMul.hMul (v ((algebraMap O F) x)) (v (( …
    ⊢ LT.lt 0 (v ((algebraMap O F) x))
  -/
  simp [hv.valuation_pos_iff_ne_zero, hx0]
  /-
    🎉 no goals
  -/


theorem eq_algebraMap_or_inv_eq_algebraMap (hv : Integers v O) (x : F) :
    ∃ a : O, x = algebraMap O F a ∨ x⁻¹ = algebraMap O F a := by
  /-
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x : F
    ⊢ Exists fun a => Or (Eq x ((algebraMap O F) a)) (Eq (Inv.inv x) ((algebraMap  …
  -/
  rcases val_le_one_or_val_inv_le_one v x with h | h <;>
  /-
    case inl
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x : F
    h : LE.le (v x) 1
    ⊢ Exists fun a => Or (Eq x ((algebraMap O F) a)) (Eq (Inv.inv x) ((algebraMap  …
  -/
  obtain ⟨a, ha⟩ := exists_of_le_one hv h
  /-
    case inl.intro
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    x : F
    h : LE.le (v x) 1
    a : O
    ha : Eq ((algebraMap O F) a) x
    ⊢ Exists fun a => Or (Eq x ((algebraMap O F) a)) (Eq (Inv.inv x) ((algebraMap  …
  -/
  exacts [⟨a, Or.inl ha.symm⟩, ⟨a, Or.inr ha.symm⟩]
  /-
    🎉 no goals
  -/


lemma isPrincipal_iff_exists_isGreatest (hv : Integers v O) {I : Ideal O} :
    I.IsPrincipal ↔ ∃ x, IsGreatest (v ∘ algebraMap O F '' I) x := by
  /-
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    I : Ideal O
    ⊢ Iff (Submodule.IsPrincipal I) (Exists fun x => IsGreatest (Set.image (Functi …
  -/
  constructor <;> rintro ⟨x, hx⟩
    /-
      case mp.mk.intro
      F : Type u
      Γ₀ : Type v
      inst✝³ : Field F
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation F Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O F
      hv : v.Integers O
      I : Ideal O
      x : O
      hx : Eq I (Submodule.span O (Singleton.singleton x))
      ⊢ Exists fun x => IsGreatest (Set.image (Function.comp ⇑v ⇑(algebraMap O F)) ↑ …
    -/
  · refine ⟨(v ∘ algebraMap O F) x, ?_, ?_⟩
      /-
        case mp.mk.intro.refine_1
        F : Type u
        Γ₀ : Type v
        inst✝³ : Field F
        inst✝² : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation F Γ₀
        O : Type w
        inst✝¹ : CommRing O
        inst✝ : Algebra O F
        hv : v.Integers O
        I : Ideal O
        x : O
        hx : Eq I (Submodule.span O (Singleton.singleton x))
        ⊢ Membership.mem (Set.image (Function.comp ⇑v ⇑(algebraMap O F)) ↑I) (Function …
      -/
    · refine Set.mem_image_of_mem _ ?_
      /-
        case mp.mk.intro.refine_1
        F : Type u
        Γ₀ : Type v
        inst✝³ : Field F
        inst✝² : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation F Γ₀
        O : Type w
        inst✝¹ : CommRing O
        inst✝ : Algebra O F
        hv : v.Integers O
        I : Ideal O
        x : O
        hx : Eq I (Submodule.span O (Singleton.singleton x))
        ⊢ Membership.mem (↑I) x
      -/
      simp [hx, Ideal.mem_span_singleton_self]
      /-
        🎉 no goals
      -/
      /-
        case mp.mk.intro.refine_2
        F : Type u
        Γ₀ : Type v
        inst✝³ : Field F
        inst✝² : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation F Γ₀
        O : Type w
        inst✝¹ : CommRing O
        inst✝ : Algebra O F
        hv : v.Integers O
        I : Ideal O
        x : O
        hx : Eq I (Submodule.span O (Singleton.singleton x))
        ⊢ Membership.mem (upperBounds (Set.image (Function.comp ⇑v ⇑(algebraMap O F))  …
      -/
    · intro y hy
      simp only [Function.comp_apply, hx, Ideal.submodule_span_eq, Set.mem_image,
        SetLike.mem_coe, Ideal.mem_span_singleton] at hy
      /-
        case mp.mk.intro.refine_2
        F : Type u
        Γ₀ : Type v
        inst✝³ : Field F
        inst✝² : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation F Γ₀
        O : Type w
        inst✝¹ : CommRing O
        inst✝ : Algebra O F
        hv : v.Integers O
        I : Ideal O
        x : O
        hx : Eq I (Submodule.span O (Singleton.singleton x))
        y : Γ₀
        hy : Exists fun x_1 => And (Dvd.dvd x x_1) (Eq (v ((algebraMap O F) x_1)) y)
        ⊢ LE.le y (Function.comp (⇑v) (⇑(algebraMap O F)) x)
      -/
      obtain ⟨y, hy, rfl⟩ := hy
      /-
        case mp.mk.intro.refine_2.intro.intro
        F : Type u
        Γ₀ : Type v
        inst✝³ : Field F
        inst✝² : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation F Γ₀
        O : Type w
        inst✝¹ : CommRing O
        inst✝ : Algebra O F
        hv : v.Integers O
        I : Ideal O
        x : O
        hx : Eq I (Submodule.span O (Singleton.singleton x))
        y : O
        hy : Dvd.dvd x y
        ⊢ LE.le (v ((algebraMap O F) y)) (Function.comp (⇑v) (⇑(algebraMap O F)) x)
      -/
      exact le_of_dvd hv hy
      /-
        🎉 no goals
      -/
    /-
      case mpr.intro
      F : Type u
      Γ₀ : Type v
      inst✝³ : Field F
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation F Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O F
      hv : v.Integers O
      I : Ideal O
      x : Γ₀
      hx : IsGreatest (Set.image (Function.comp ⇑v ⇑(algebraMap O F)) ↑I) x
      ⊢ Submodule.IsPrincipal I
    -/
  · obtain ⟨a, ha, rfl⟩ : ∃ a ∈ I, (v ∘ algebraMap O F) a = x := by simpa using hx.left
    /-
      case mpr.intro.intro.intro
      F : Type u
      Γ₀ : Type v
      inst✝³ : Field F
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation F Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O F
      hv : v.Integers O
      I : Ideal O
      a : O
      ha : Membership.mem I a
      hx : IsGreatest (Set.image (Function.comp ⇑v ⇑(algebraMap O F)) ↑I) (Function. …
      ⊢ Submodule.IsPrincipal I
    -/
    refine ⟨a, ?_⟩
    /-
      case mpr.intro.intro.intro
      F : Type u
      Γ₀ : Type v
      inst✝³ : Field F
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation F Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O F
      hv : v.Integers O
      I : Ideal O
      a : O
      ha : Membership.mem I a
      hx : IsGreatest (Set.image (Function.comp ⇑v ⇑(algebraMap O F)) ↑I) (Function. …
      ⊢ Eq I (Submodule.span O (Singleton.singleton a))
    -/
    ext b
    /-
      case mpr.intro.intro.intro.h
      F : Type u
      Γ₀ : Type v
      inst✝³ : Field F
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation F Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O F
      hv : v.Integers O
      I : Ideal O
      a : O
      ha : Membership.mem I a
      hx : IsGreatest (Set.image (Function.comp ⇑v ⇑(algebraMap O F)) ↑I) (Function. …
      b : O
      ⊢ Iff (Membership.mem I b) (Membership.mem (Submodule.span O (Singleton.single …
    -/
    simp only [Ideal.submodule_span_eq, Ideal.mem_span_singleton]
    /-
      case mpr.intro.intro.intro.h
      F : Type u
      Γ₀ : Type v
      inst✝³ : Field F
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation F Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O F
      hv : v.Integers O
      I : Ideal O
      a : O
      ha : Membership.mem I a
      hx : IsGreatest (Set.image (Function.comp ⇑v ⇑(algebraMap O F)) ↑I) (Function. …
      b : O
      ⊢ Iff (Membership.mem I b) (Dvd.dvd a b)
    -/
    exact ⟨fun hb ↦ dvd_of_le hv (hx.2 <| mem_image_of_mem _ hb), fun hb ↦ I.mem_of_dvd hb ha⟩
    /-
      🎉 no goals
    -/


lemma isPrincipal_iff_exists_eq_setOf_valuation_le (hv : Integers v O) {I : Ideal O} :
    I.IsPrincipal ↔ ∃ x, (I : Set O) = {y | v (algebraMap O F y) ≤ v (algebraMap O F x)} := by
  /-
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    I : Ideal O
    ⊢ Iff (Submodule.IsPrincipal I) (Exists fun x => Eq (↑I) (setOf fun y => LE.le …
  -/
  rw [isPrincipal_iff_exists_isGreatest hv]
  /-
    F : Type u
    Γ₀ : Type v
    inst✝³ : Field F
    inst✝² : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝¹ : CommRing O
    inst✝ : Algebra O F
    hv : v.Integers O
    I : Ideal O
    ⊢ Iff (Exists fun x => IsGreatest (Set.image (Function.comp ⇑v ⇑(algebraMap O  …
  -/
  constructor <;> rintro ⟨x, hx⟩
    /-
      case mp.intro
      F : Type u
      Γ₀ : Type v
      inst✝³ : Field F
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation F Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O F
      hv : v.Integers O
      I : Ideal O
      x : Γ₀
      hx : IsGreatest (Set.image (Function.comp ⇑v ⇑(algebraMap O F)) ↑I) x
      ⊢ Exists fun x => Eq (↑I) (setOf fun y => LE.le (v ((algebraMap O F) y)) (v (( …
    -/
  · obtain ⟨a, ha, rfl⟩ : ∃ a ∈ I, (v ∘ algebraMap O F) a = x := by simpa using hx.left
    /-
      case mp.intro.intro.intro
      F : Type u
      Γ₀ : Type v
      inst✝³ : Field F
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation F Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O F
      hv : v.Integers O
      I : Ideal O
      a : O
      ha : Membership.mem I a
      hx : IsGreatest (Set.image (Function.comp ⇑v ⇑(algebraMap O F)) ↑I) (Function. …
      ⊢ Exists fun x => Eq (↑I) (setOf fun y => LE.le (v ((algebraMap O F) y)) (v (( …
    -/
    refine ⟨a, ?_⟩
    /-
      case mp.intro.intro.intro
      F : Type u
      Γ₀ : Type v
      inst✝³ : Field F
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation F Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O F
      hv : v.Integers O
      I : Ideal O
      a : O
      ha : Membership.mem I a
      hx : IsGreatest (Set.image (Function.comp ⇑v ⇑(algebraMap O F)) ↑I) (Function. …
      ⊢ Eq (↑I) (setOf fun y => LE.le (v ((algebraMap O F) y)) (v ((algebraMap O F)  …
    -/
    ext b
    /-
      case mp.intro.intro.intro.h
      F : Type u
      Γ₀ : Type v
      inst✝³ : Field F
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation F Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O F
      hv : v.Integers O
      I : Ideal O
      a : O
      ha : Membership.mem I a
      hx : IsGreatest (Set.image (Function.comp ⇑v ⇑(algebraMap O F)) ↑I) (Function. …
      b : O
      ⊢ Iff (Membership.mem (↑I) b) (Membership.mem (setOf fun y => LE.le (v ((algeb …
    -/
    simp only [SetLike.mem_coe, mem_setOf_eq]
    /-
      case mp.intro.intro.intro.h
      F : Type u
      Γ₀ : Type v
      inst✝³ : Field F
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation F Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O F
      hv : v.Integers O
      I : Ideal O
      a : O
      ha : Membership.mem I a
      hx : IsGreatest (Set.image (Function.comp ⇑v ⇑(algebraMap O F)) ↑I) (Function. …
      b : O
      ⊢ Iff (Membership.mem I b) (LE.le (v ((algebraMap O F) b)) (v ((algebraMap O F …
    -/
    constructor <;> intro h
      /-
        case mp.intro.intro.intro.h.mp
        F : Type u
        Γ₀ : Type v
        inst✝³ : Field F
        inst✝² : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation F Γ₀
        O : Type w
        inst✝¹ : CommRing O
        inst✝ : Algebra O F
        hv : v.Integers O
        I : Ideal O
        a : O
        ha : Membership.mem I a
        hx : IsGreatest (Set.image (Function.comp ⇑v ⇑(algebraMap O F)) ↑I) (Function. …
        b : O
        h : Membership.mem I b
        ⊢ LE.le (v ((algebraMap O F) b)) (v ((algebraMap O F) a))
      -/
    · exact hx.right (Set.mem_image_of_mem _ h)
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.h.mpr
        F : Type u
        Γ₀ : Type v
        inst✝³ : Field F
        inst✝² : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation F Γ₀
        O : Type w
        inst✝¹ : CommRing O
        inst✝ : Algebra O F
        hv : v.Integers O
        I : Ideal O
        a : O
        ha : Membership.mem I a
        hx : IsGreatest (Set.image (Function.comp ⇑v ⇑(algebraMap O F)) ↑I) (Function. …
        b : O
        h : LE.le (v ((algebraMap O F) b)) (v ((algebraMap O F) a))
        ⊢ Membership.mem I b
      -/
    · rw [le_iff_dvd hv] at h
      /-
        case mp.intro.intro.intro.h.mpr
        F : Type u
        Γ₀ : Type v
        inst✝³ : Field F
        inst✝² : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation F Γ₀
        O : Type w
        inst✝¹ : CommRing O
        inst✝ : Algebra O F
        hv : v.Integers O
        I : Ideal O
        a : O
        ha : Membership.mem I a
        hx : IsGreatest (Set.image (Function.comp ⇑v ⇑(algebraMap O F)) ↑I) (Function. …
        b : O
        h : Dvd.dvd a b
        ⊢ Membership.mem I b
      -/
      exact Ideal.mem_of_dvd I h ha
      /-
        🎉 no goals
      -/
    /-
      case mpr.intro
      F : Type u
      Γ₀ : Type v
      inst✝³ : Field F
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation F Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O F
      hv : v.Integers O
      I : Ideal O
      x : O
      hx : Eq (↑I) (setOf fun y => LE.le (v ((algebraMap O F) y)) (v ((algebraMap O  …
      ⊢ Exists fun x => IsGreatest (Set.image (Function.comp ⇑v ⇑(algebraMap O F)) ↑ …
    -/
  · refine ⟨v (algebraMap O F x), Set.mem_image_of_mem _ ?_, ?_⟩
      /-
        case mpr.intro.refine_1
        F : Type u
        Γ₀ : Type v
        inst✝³ : Field F
        inst✝² : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation F Γ₀
        O : Type w
        inst✝¹ : CommRing O
        inst✝ : Algebra O F
        hv : v.Integers O
        I : Ideal O
        x : O
        hx : Eq (↑I) (setOf fun y => LE.le (v ((algebraMap O F) y)) (v ((algebraMap O  …
        ⊢ Membership.mem (↑I) x
      -/
    · simp [hx]
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.refine_2
        F : Type u
        Γ₀ : Type v
        inst✝³ : Field F
        inst✝² : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation F Γ₀
        O : Type w
        inst✝¹ : CommRing O
        inst✝ : Algebra O F
        hv : v.Integers O
        I : Ideal O
        x : O
        hx : Eq (↑I) (setOf fun y => LE.le (v ((algebraMap O F) y)) (v ((algebraMap O  …
        ⊢ Membership.mem (upperBounds (Set.image (Function.comp ⇑v ⇑(algebraMap O F))  …
      -/
    · simp [hx, mem_upperBounds]
      /-
        🎉 no goals
      -/


lemma not_denselyOrdered_of_isPrincipalIdealRing [IsPrincipalIdealRing O] (hv : Integers v O) :
    ¬ DenselyOrdered (range v) := by
  /-
    F : Type u
    Γ₀ : Type v
    inst✝⁴ : Field F
    inst✝³ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝² : CommRing O
    inst✝¹ : Algebra O F
    inst✝ : IsPrincipalIdealRing O
    hv : v.Integers O
    ⊢ Not (DenselyOrdered ↑(Set.range ⇑v))
  -/
  intro H
  -- nonunits as an ideal isn't defined here, nor shown to be equivalent to `v x < 1`
  set I : Ideal O := {
    carrier := v ∘ algebraMap O F ⁻¹' Iio (1 : Γ₀)
    add_mem' := fun {a b} ha hb ↦ by simpa using map_add_lt v ha hb
    zero_mem' := by simp
    smul_mem' := by
      intro c x
      simp only [mem_preimage, Function.comp_apply, mem_Iio, smul_eq_mul, _root_.map_mul]
      intro hx
      exact Right.mul_lt_one_of_le_of_lt (hv.map_le_one c) hx
  }
  obtain ⟨x, hx₁, hx⟩ :
    ∃ x, v (algebraMap O F x) < 1 ∧
      v (algebraMap O F x) ∈ upperBounds (Iio 1 ∩ range (v ∘ algebraMap O F)) := by
    simpa [I, IsGreatest, hv.isPrincipal_iff_exists_isGreatest, ← image_preimage_eq_inter_range]
      using IsPrincipalIdealRing.principal I
  obtain ⟨y, hy, hy₁⟩ : ∃ y, v (algebraMap O F x) < v y ∧ v y < 1 := by
    simpa only [Subtype.exists, Subtype.mk_lt_mk, exists_range_iff, exists_prop]
      using H.dense ⟨v (algebraMap O F x), mem_range_self _⟩ ⟨1, 1, v.map_one⟩ hx₁
  /-
    case intro.intro.intro.intro
    F : Type u
    Γ₀ : Type v
    inst✝⁴ : Field F
    inst✝³ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝² : CommRing O
    inst✝¹ : Algebra O F
    inst✝ : IsPrincipalIdealRing O
    hv : v.Integers O
    H : DenselyOrdered ↑(Set.range ⇑v)
    I : Ideal O := { carrier := Set.preimage (Function.comp ⇑v ⇑(algebraMap O F))  …
    x : O
    hx₁ : LT.lt (v ((algebraMap O F) x)) 1
    hx : Membership.mem (upperBounds (Inter.inter (Set.Iio 1) (Set.range (Function …
    y : F
    hy : LT.lt (v ((algebraMap O F) x)) (v y)
    hy₁ : LT.lt (v y) 1
    ⊢ False
  -/
  obtain ⟨z, rfl⟩ := hv.exists_of_le_one hy₁.le
  /-
    case intro.intro.intro.intro.intro
    F : Type u
    Γ₀ : Type v
    inst✝⁴ : Field F
    inst✝³ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation F Γ₀
    O : Type w
    inst✝² : CommRing O
    inst✝¹ : Algebra O F
    inst✝ : IsPrincipalIdealRing O
    hv : v.Integers O
    H : DenselyOrdered ↑(Set.range ⇑v)
    I : Ideal O := { carrier := Set.preimage (Function.comp ⇑v ⇑(algebraMap O F))  …
    x : O
    hx₁ : LT.lt (v ((algebraMap O F) x)) 1
    hx : Membership.mem (upperBounds (Inter.inter (Set.Iio 1) (Set.range (Function …
    z : O
    hy : LT.lt (v ((algebraMap O F) x)) (v ((algebraMap O F) z))
    hy₁ : LT.lt (v ((algebraMap O F) z)) 1
    ⊢ False
  -/
  exact hy.not_le <| hx ⟨hy₁, mem_range_self _⟩
  /-
    🎉 no goals
  -/

-- TODO: isPrincipalIdealRing_iff_not_denselyOrdered when MulArchimedean


