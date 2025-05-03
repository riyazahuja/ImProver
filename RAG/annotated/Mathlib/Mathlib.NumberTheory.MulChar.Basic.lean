/-- Define a structure for multiplicative characters.
A multiplicative character from a commutative monoid `R` to a commutative monoid with zero `R'`
is a homomorphism of (multiplicative) monoids that sends non-units to zero. -/
structure MulChar extends MonoidHom R R' where
  map_nonunit' : ∀ a : R, ¬IsUnit a → toFun a = 0


instance MulChar.instFunLike : FunLike (MulChar R R') R R' :=
  ⟨fun χ => χ.toFun,
                      /-
                        R : Type u_1
                        inst✝¹ : CommMonoid R
                        R' : Type u_2
                        inst✝ : CommMonoidWithZero R'
                        χ₀ χ₁ : MulChar R R'
                        h : Eq ((fun χ => (↑χ.toMonoidHom).toFun) χ₀) ((fun χ => (↑χ.toMonoidHom).toFu …
                        ⊢ Eq χ₀ χ₁
                      -/
    fun χ₀ χ₁ h => by cases χ₀; cases χ₁; congr; apply MonoidHom.ext (fun _ => congr_fun h _)⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- This is the corresponding extension of `MonoidHomClass`. -/
class MulCharClass (F : Type*) (R R' : outParam Type*) [CommMonoid R]
  [CommMonoidWithZero R'] [FunLike F R R'] extends MonoidHomClass F R R' : Prop where
  map_nonunit : ∀ (χ : F) {a : R} (_ : ¬IsUnit a), χ a = 0


variable (R R') in
/-- The trivial multiplicative character. It takes the value `0` on non-units and
the value `1` on units. -/
@[simps]
noncomputable def trivial : MulChar R R' where
              /-
                R : Type u_1
                inst✝¹ : CommMonoid R
                R' : Type u_2
                inst✝ : CommMonoidWithZero R'
                ⊢ R → R'
              -/
  toFun := by classical exact fun x => if IsUnit x then 1 else 0
              /-
                🎉 no goals
              -/
  map_nonunit' := by
    /-
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      ⊢ ∀ (a : R), Not (IsUnit a) → Eq ((↑{ toFun := fun x => ite (IsUnit x) 1 0, ma …
    -/
    intro a ha
                 /-
                   R : Type u_1
                   inst✝¹ : CommMonoid R
                   R' : Type u_2
                   inst✝ : CommMonoidWithZero R'
                   ⊢ Eq (ite (IsUnit 1) 1 0) 1
                 -/
    /-
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      a : R
      ha : Not (IsUnit a)
      ⊢ Eq ((↑{ toFun := fun x => ite (IsUnit x) 1 0, map_one' := ⋯, map_mul' := ⋯ } …
    -/
                 /-
                   🎉 no goals
                 -/
    simp only [ha, if_false]
    /-
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      ⊢ ∀ (x y : R), Eq ({ toFun := fun x => ite (IsUnit x) 1 0, map_one' := ⋯ }.toF …
    -/
    /-
      🎉 no goals
    -/
  map_one' := by simp only [isUnit_one, if_true]
  map_mul' := by
    intro x y
    classical
      simp only [IsUnit.mul_iff, boole_mul]
      split_ifs <;> tauto


@[simp]
theorem coe_mk (f : R →* R') (hf) : (MulChar.mk f hf : R → R') = f :=
  rfl


/-- Extensionality. See `ext` below for the version that will actually be used. -/
theorem ext' {χ χ' : MulChar R R'} (h : ∀ a, χ a = χ' a) : χ = χ' := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ χ' : MulChar R R'
    h : ∀ (a : R), Eq (χ a) (χ' a)
    ⊢ Eq χ χ'
  -/
  cases χ
  /-
    case mk
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ' : MulChar R R'
    toMonoidHom✝ : MonoidHom R R'
    map_nonunit'✝ : ∀ (a : R), Not (IsUnit a) → Eq ((↑toMonoidHom✝).toFun a) 0
    h : ∀ (a : R), Eq ({ toMonoidHom := toMonoidHom✝, map_nonunit' := map_nonunit' …
    ⊢ Eq { toMonoidHom := toMonoidHom✝, map_nonunit' := map_nonunit'✝ } χ'
  -/
  cases χ'
  /-
    case mk.mk
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    toMonoidHom✝¹ : MonoidHom R R'
    map_nonunit'✝¹ : ∀ (a : R), Not (IsUnit a) → Eq ((↑toMonoidHom✝¹).toFun a) 0
    toMonoidHom✝ : MonoidHom R R'
    map_nonunit'✝ : ∀ (a : R), Not (IsUnit a) → Eq ((↑toMonoidHom✝).toFun a) 0
    h : ∀ (a : R), Eq ({ toMonoidHom := toMonoidHom✝¹, map_nonunit' := map_nonunit …
    ⊢ Eq { toMonoidHom := toMonoidHom✝¹, map_nonunit' := map_nonunit'✝¹ } { toMono …
  -/
  congr
  /-
    case mk.mk.e_toMonoidHom
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    toMonoidHom✝¹ : MonoidHom R R'
    map_nonunit'✝¹ : ∀ (a : R), Not (IsUnit a) → Eq ((↑toMonoidHom✝¹).toFun a) 0
    toMonoidHom✝ : MonoidHom R R'
    map_nonunit'✝ : ∀ (a : R), Not (IsUnit a) → Eq ((↑toMonoidHom✝).toFun a) 0
    h : ∀ (a : R), Eq ({ toMonoidHom := toMonoidHom✝¹, map_nonunit' := map_nonunit …
    ⊢ Eq toMonoidHom✝¹ toMonoidHom✝
  -/
  exact MonoidHom.ext h
  /-
    🎉 no goals
  -/


instance : MulCharClass (MulChar R R') R R' where
  map_mul χ := χ.map_mul'
  map_one χ := χ.map_one'
  map_nonunit χ := χ.map_nonunit' _


theorem map_nonunit (χ : MulChar R R') {a : R} (ha : ¬IsUnit a) : χ a = 0 :=
  χ.map_nonunit' a ha


/-- Extensionality. Since `MulChar`s always take the value zero on non-units, it is sufficient
to compare the values on units. -/
@[ext]
theorem ext {χ χ' : MulChar R R'} (h : ∀ a : Rˣ, χ a = χ' a) : χ = χ' := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ χ' : MulChar R R'
    h : ∀ (a : Units R), Eq (χ ↑a) (χ' ↑a)
    ⊢ Eq χ χ'
  -/
  apply ext'
  /-
    case h
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ χ' : MulChar R R'
    h : ∀ (a : Units R), Eq (χ ↑a) (χ' ↑a)
    ⊢ ∀ (a : R), Eq (χ a) (χ' a)
  -/
  intro a
  /-
    case h
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ χ' : MulChar R R'
    h : ∀ (a : Units R), Eq (χ ↑a) (χ' ↑a)
    a : R
    ⊢ Eq (χ a) (χ' a)
  -/
  by_cases ha : IsUnit a
    /-
      case pos
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      χ χ' : MulChar R R'
      h : ∀ (a : Units R), Eq (χ ↑a) (χ' ↑a)
      a : R
      ha : IsUnit a
      ⊢ Eq (χ a) (χ' a)
    -/
  · exact h ha.unit
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      χ χ' : MulChar R R'
      h : ∀ (a : Units R), Eq (χ ↑a) (χ' ↑a)
      a : R
      ha : Not (IsUnit a)
      ⊢ Eq (χ a) (χ' a)
    -/
  · rw [map_nonunit χ ha, map_nonunit χ' ha]
    /-
      🎉 no goals
    -/


/-- Turn a `MulChar` into a homomorphism between the unit groups. -/
def toUnitHom (χ : MulChar R R') : Rˣ →* R'ˣ :=
  Units.map χ


theorem coe_toUnitHom (χ : MulChar R R') (a : Rˣ) : ↑(χ.toUnitHom a) = χ a :=
  rfl


/-- Turn a homomorphism between unit groups into a `MulChar`. -/
noncomputable def ofUnitHom (f : Rˣ →* R'ˣ) : MulChar R R' where
              /-
                R : Type u_1
                inst✝¹ : CommMonoid R
                R' : Type u_2
                inst✝ : CommMonoidWithZero R'
                f : MonoidHom (Units R) (Units R')
                ⊢ R → R'
              -/
  toFun := by classical exact fun x => if hx : IsUnit x then f hx.unit else 0
              /-
                🎉 no goals
              -/
  map_one' := by
    /-
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      f : MonoidHom (Units R) (Units R')
      ⊢ Eq (dite (IsUnit 1) (fun hx => ↑(f hx.unit)) fun hx => 0) 1
    -/
    have h1 : (isUnit_one.unit : Rˣ) = 1 := Units.eq_iff.mp rfl
    /-
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      f : MonoidHom (Units R) (Units R')
      h1 : Eq ⋯.unit 1
      ⊢ Eq (dite (IsUnit 1) (fun hx => ↑(f hx.unit)) fun hx => 0) 1
    -/
    simp only [h1, dif_pos, Units.val_eq_one, map_one, isUnit_one]
    /-
      🎉 no goals
    -/
  map_mul' := by
    classical
      intro x y
      by_cases hx : IsUnit x
      · simp only [hx, IsUnit.mul_iff, true_and, dif_pos]
        by_cases hy : IsUnit y
        · simp only [hy, dif_pos]
          have hm : (IsUnit.mul_iff.mpr ⟨hx, hy⟩).unit = hx.unit * hy.unit := Units.eq_iff.mp rfl
          rw [hm, map_mul]
          norm_cast
        · simp only [hy, not_false_iff, dif_neg, mul_zero]
      · simp only [hx, IsUnit.mul_iff, false_and, not_false_iff, dif_neg, zero_mul]
  map_nonunit' := by
    /-
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      f : MonoidHom (Units R) (Units R')
      ⊢ ∀ (a : R), Not (IsUnit a) → Eq ((↑{ toFun := fun x => dite (IsUnit x) (fun h …
    -/
    intro a ha
    /-
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      f : MonoidHom (Units R) (Units R')
      a : R
      ha : Not (IsUnit a)
      ⊢ Eq ((↑{ toFun := fun x => dite (IsUnit x) (fun hx => ↑(f hx.unit)) fun hx => …
    -/
    simp only [ha, not_false_iff, dif_neg]
    /-
      🎉 no goals
    -/


                                                                            /-
                                                                              R : Type u_1
                                                                              inst✝¹ : CommMonoid R
                                                                              R' : Type u_2
                                                                              inst✝ : CommMonoidWithZero R'
                                                                              f : MonoidHom (Units R) (Units R')
                                                                              a : Units R
                                                                              ⊢ Eq ((MulChar.ofUnitHom f) ↑a) ↑(f a)
                                                                            -/
theorem ofUnitHom_coe (f : Rˣ →* R'ˣ) (a : Rˣ) : ofUnitHom f ↑a = f a := by simp [ofUnitHom]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- The equivalence between multiplicative characters and homomorphisms of unit groups. -/
noncomputable def equivToUnitHom : MulChar R R' ≃ (Rˣ →* R'ˣ) where
  toFun := toUnitHom
  invFun := ofUnitHom
  left_inv := by
    /-
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      ⊢ Function.LeftInverse MulChar.ofUnitHom MulChar.toUnitHom
    -/
    intro χ
    /-
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      χ : MulChar R R'
      ⊢ Eq (MulChar.ofUnitHom χ.toUnitHom) χ
    -/
    ext x
    /-
      case h
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      χ : MulChar R R'
      x : Units R
      ⊢ Eq ((MulChar.ofUnitHom χ.toUnitHom) ↑x) (χ ↑x)
    -/
    rw [ofUnitHom_coe, coe_toUnitHom]
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      ⊢ Function.RightInverse MulChar.ofUnitHom MulChar.toUnitHom
    -/
    intro f
    /-
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      f : MonoidHom (Units R) (Units R')
      ⊢ Eq (MulChar.ofUnitHom f).toUnitHom f
    -/
    ext x
    /-
      case h.a
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      f : MonoidHom (Units R) (Units R')
      x : Units R
      ⊢ Eq ↑((MulChar.ofUnitHom f).toUnitHom x) ↑(f x)
    -/
    simp only [coe_toUnitHom, ofUnitHom_coe]
    /-
      🎉 no goals
    -/


@[simp]
theorem toUnitHom_eq (χ : MulChar R R') : toUnitHom χ = equivToUnitHom χ :=
  rfl


@[simp]
theorem ofUnitHom_eq (χ : Rˣ →* R'ˣ) : ofUnitHom χ = equivToUnitHom.symm χ :=
  rfl


@[simp]
theorem coe_equivToUnitHom (χ : MulChar R R') (a : Rˣ) : ↑(equivToUnitHom χ a) = χ a :=
  coe_toUnitHom χ a


@[simp]
theorem equivToUnitHom_symm_coe (f : Rˣ →* R'ˣ) (a : Rˣ) : equivToUnitHom.symm f ↑a = f a :=
  ofUnitHom_coe f a


@[simp]
lemma coe_toMonoidHom (χ : MulChar R R')
    (x : R) : χ.toMonoidHom x = χ x := rfl


protected theorem map_one (χ : MulChar R R') : χ (1 : R) = 1 :=
  χ.map_one'


/-- If the domain has a zero (and is nontrivial), then `χ 0 = 0`. -/
protected theorem map_zero {R : Type*} [CommMonoidWithZero R] [Nontrivial R] (χ : MulChar R R') :
                        /-
                          R' : Type u_2
                          inst✝² : CommMonoidWithZero R'
                          R : Type u_3
                          inst✝¹ : CommMonoidWithZero R
                          inst✝ : Nontrivial R
                          χ : MulChar R R'
                          ⊢ Eq (χ 0) 0
                        -/
    χ (0 : R) = 0 := by rw [map_nonunit χ not_isUnit_zero]
                        /-
                          🎉 no goals
                        -/


/-- We can convert a multiplicative character into a homomorphism of monoids with zero when
the source has a zero and another element. -/
@[coe, simps]
def toMonoidWithZeroHom {R : Type*} [CommMonoidWithZero R] [Nontrivial R] (χ : MulChar R R') :
    R →*₀ R' where
  toFun := χ.toFun
  map_zero' := χ.map_zero
  map_one' := χ.map_one'
  map_mul' := χ.map_mul'


/-- If the domain is a ring `R`, then `χ (ringChar R) = 0`. -/
theorem map_ringChar {R : Type*} [CommRing R] [Nontrivial R] (χ : MulChar R R') :
                             /-
                               R' : Type u_2
                               inst✝² : CommMonoidWithZero R'
                               R : Type u_3
                               inst✝¹ : CommRing R
                               inst✝ : Nontrivial R
                               χ : MulChar R R'
                               ⊢ Eq (χ ↑(ringChar R)) 0
                             -/
    χ (ringChar R) = 0 := by rw [ringChar.Nat.cast_ringChar, χ.map_zero]
                             /-
                               🎉 no goals
                             -/


noncomputable instance hasOne : One (MulChar R R') :=
  ⟨trivial R R'⟩


noncomputable instance inhabited : Inhabited (MulChar R R') :=
  ⟨1⟩


/-- Evaluation of the trivial character -/
@[simp]
                                                                /-
                                                                  R : Type u_1
                                                                  inst✝¹ : CommMonoid R
                                                                  R' : Type u_2
                                                                  inst✝ : CommMonoidWithZero R'
                                                                  a : Units R
                                                                  ⊢ Eq (1 ↑a) 1
                                                                -/
theorem one_apply_coe (a : Rˣ) : (1 : MulChar R R') a = 1 := by classical exact dif_pos a.isUnit
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- Evaluation of the trivial character -/
lemma one_apply {x : R} (hx : IsUnit x) : (1 : MulChar R R') x = 1 := one_apply_coe hx.unit


/-- Multiplication of multiplicative characters. (This needs the target to be commutative.) -/
def mul (χ χ' : MulChar R R') : MulChar R R' :=
  { χ.toMonoidHom * χ'.toMonoidHom with
    toFun := χ * χ'
                                   /-
                                     R : Type u_1
                                     inst✝¹ : CommMonoid R
                                     R' : Type u_2
                                     inst✝ : CommMonoidWithZero R'
                                     χ χ' : MulChar R R'
                                     a : R
                                     ha : Not (IsUnit a)
                                     ⊢ Eq ((↑{ toFun := HMul.hMul ⇑χ ⇑χ', map_one' := ⋯, map_mul' := ⋯ }).toFun a) 0
                                   -/
    map_nonunit' := fun a ha => by simp only [map_nonunit χ ha, zero_mul, Pi.mul_apply] }
                                   /-
                                     🎉 no goals
                                   -/


instance hasMul : Mul (MulChar R R') :=
  ⟨mul⟩


theorem mul_apply (χ χ' : MulChar R R') (a : R) : (χ * χ') a = χ a * χ' a :=
  rfl


@[simp]
theorem coeToFun_mul (χ χ' : MulChar R R') : ⇑(χ * χ') = χ * χ' :=
  rfl


protected theorem one_mul (χ : MulChar R R') : (1 : MulChar R R') * χ = χ := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ : MulChar R R'
    ⊢ Eq (HMul.hMul 1 χ) χ
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ : MulChar R R'
    a✝ : Units R
    ⊢ Eq ((HMul.hMul 1 χ) ↑a✝) (χ ↑a✝)
  -/
  simp only [one_mul, Pi.mul_apply, MulChar.coeToFun_mul, MulChar.one_apply_coe]
  /-
    🎉 no goals
  -/


protected theorem mul_one (χ : MulChar R R') : χ * 1 = χ := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ : MulChar R R'
    ⊢ Eq (HMul.hMul χ 1) χ
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ : MulChar R R'
    a✝ : Units R
    ⊢ Eq ((HMul.hMul χ 1) ↑a✝) (χ ↑a✝)
  -/
  simp only [mul_one, Pi.mul_apply, MulChar.coeToFun_mul, MulChar.one_apply_coe]
  /-
    🎉 no goals
  -/


/-- The inverse of a multiplicative character. We define it as `inverse ∘ χ`. -/
noncomputable def inv (χ : MulChar R R') : MulChar R R' :=
  { MonoidWithZero.inverse.toMonoidHom.comp χ.toMonoidHom with
    toFun := fun a => MonoidWithZero.inverse (χ a)
                                   /-
                                     R : Type u_1
                                     inst✝¹ : CommMonoid R
                                     R' : Type u_2
                                     inst✝ : CommMonoidWithZero R'
                                     χ : MulChar R R'
                                     a : R
                                     ha : Not (IsUnit a)
                                     ⊢ Eq ((↑{ toFun := fun a => MonoidWithZero.inverse (χ a), map_one' := ⋯, map_m …
                                   -/
    map_nonunit' := fun a ha => by simp [map_nonunit _ ha] }
                                   /-
                                     🎉 no goals
                                   -/


noncomputable instance hasInv : Inv (MulChar R R') :=
  ⟨inv⟩


/-- The inverse of a multiplicative character `χ`, applied to `a`, is the inverse of `χ a`. -/
theorem inv_apply_eq_inv (χ : MulChar R R') (a : R) : χ⁻¹ a = Ring.inverse (χ a) :=
  Eq.refl <| inv χ a


/-- The inverse of a multiplicative character `χ`, applied to `a`, is the inverse of `χ a`.
Variant when the target is a field -/
theorem inv_apply_eq_inv' {R' : Type*} [Field R'] (χ : MulChar R R') (a : R) : χ⁻¹ a = (χ a)⁻¹ :=
  (inv_apply_eq_inv χ a).trans <| Ring.inverse_eq_inv (χ a)


/-- When the domain has a zero, then the inverse of a multiplicative character `χ`,
applied to `a`, is `χ` applied to the inverse of `a`. -/
theorem inv_apply {R : Type*} [CommMonoidWithZero R] (χ : MulChar R R') (a : R) :
    χ⁻¹ a = χ (Ring.inverse a) := by
  /-
    R' : Type u_2
    inst✝¹ : CommMonoidWithZero R'
    R : Type u_3
    inst✝ : CommMonoidWithZero R
    χ : MulChar R R'
    a : R
    ⊢ Eq ((Inv.inv χ) a) (χ (Ring.inverse a))
  -/
  by_cases ha : IsUnit a
    /-
      case pos
      R' : Type u_2
      inst✝¹ : CommMonoidWithZero R'
      R : Type u_3
      inst✝ : CommMonoidWithZero R
      χ : MulChar R R'
      a : R
      ha : IsUnit a
      ⊢ Eq ((Inv.inv χ) a) (χ (Ring.inverse a))
    -/
  · rw [inv_apply_eq_inv]
    /-
      case pos
      R' : Type u_2
      inst✝¹ : CommMonoidWithZero R'
      R : Type u_3
      inst✝ : CommMonoidWithZero R
      χ : MulChar R R'
      a : R
      ha : IsUnit a
      ⊢ Eq (Ring.inverse (χ a)) (χ (Ring.inverse a))
    -/
    have h := IsUnit.map χ ha
    /-
      case pos
      R' : Type u_2
      inst✝¹ : CommMonoidWithZero R'
      R : Type u_3
      inst✝ : CommMonoidWithZero R
      χ : MulChar R R'
      a : R
      ha : IsUnit a
      h : IsUnit (χ a)
      ⊢ Eq (Ring.inverse (χ a)) (χ (Ring.inverse a))
    -/
    apply_fun (χ a * ·) using IsUnit.mul_right_injective h
    /-
      case pos
      R' : Type u_2
      inst✝¹ : CommMonoidWithZero R'
      R : Type u_3
      inst✝ : CommMonoidWithZero R
      χ : MulChar R R'
      a : R
      ha : IsUnit a
      h : IsUnit (χ a)
      ⊢ Eq ((fun x => HMul.hMul (χ a) x) (Ring.inverse (χ a))) ((fun x => HMul.hMul  …
    -/
    dsimp only
    /-
      case pos
      R' : Type u_2
      inst✝¹ : CommMonoidWithZero R'
      R : Type u_3
      inst✝ : CommMonoidWithZero R
      χ : MulChar R R'
      a : R
      ha : IsUnit a
      h : IsUnit (χ a)
      ⊢ Eq (HMul.hMul (χ a) (Ring.inverse (χ a))) (HMul.hMul (χ a) (χ (Ring.inverse  …
    -/
    rw [Ring.mul_inverse_cancel _ h, ← map_mul, Ring.mul_inverse_cancel _ ha, map_one]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R' : Type u_2
      inst✝¹ : CommMonoidWithZero R'
      R : Type u_3
      inst✝ : CommMonoidWithZero R
      χ : MulChar R R'
      a : R
      ha : Not (IsUnit a)
      ⊢ Eq ((Inv.inv χ) a) (χ (Ring.inverse a))
    -/
  · revert ha
    /-
      case neg
      R' : Type u_2
      inst✝¹ : CommMonoidWithZero R'
      R : Type u_3
      inst✝ : CommMonoidWithZero R
      χ : MulChar R R'
      a : R
      ⊢ Not (IsUnit a) → Eq ((Inv.inv χ) a) (χ (Ring.inverse a))
    -/
    nontriviality R
    /-
      R' : Type u_2
      inst✝¹ : CommMonoidWithZero R'
      R : Type u_3
      inst✝ : CommMonoidWithZero R
      χ : MulChar R R'
      a : R
      a✝ : Nontrivial R
      ⊢ Not (IsUnit a) → Eq ((Inv.inv χ) a) (χ (Ring.inverse a))
    -/
    intro ha
    -- `nontriviality R` by itself doesn't do it
    /-
      R' : Type u_2
      inst✝¹ : CommMonoidWithZero R'
      R : Type u_3
      inst✝ : CommMonoidWithZero R
      χ : MulChar R R'
      a : R
      a✝ : Nontrivial R
      ha : Not (IsUnit a)
      ⊢ Eq ((Inv.inv χ) a) (χ (Ring.inverse a))
    -/
    rw [map_nonunit _ ha, Ring.inverse_non_unit a ha, MulChar.map_zero χ]
    /-
      🎉 no goals
    -/


/-- When the domain has a zero, then the inverse of a multiplicative character `χ`,
applied to `a`, is `χ` applied to the inverse of `a`. -/
theorem inv_apply' {R : Type*} [Field R] (χ : MulChar R R') (a : R) : χ⁻¹ a = χ a⁻¹ :=
  (inv_apply χ a).trans <| congr_arg _ (Ring.inverse_eq_inv a)


/-- The product of a character with its inverse is the trivial character. -/
theorem inv_mul (χ : MulChar R R') : χ⁻¹ * χ = 1 := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ : MulChar R R'
    ⊢ Eq (HMul.hMul (Inv.inv χ) χ) 1
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ : MulChar R R'
    x : Units R
    ⊢ Eq ((HMul.hMul (Inv.inv χ) χ) ↑x) (1 ↑x)
  -/
  rw [coeToFun_mul, Pi.mul_apply, inv_apply_eq_inv]
  /-
    case h
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ : MulChar R R'
    x : Units R
    ⊢ Eq (HMul.hMul (Ring.inverse (χ ↑x)) (χ ↑x)) (1 ↑x)
  -/
  simp only [Ring.inverse_mul_cancel _ (IsUnit.map χ x.isUnit)]
  /-
    case h
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ : MulChar R R'
    x : Units R
    ⊢ Eq 1 (1 ↑x)
  -/
  rw [one_apply_coe]
  /-
    🎉 no goals
  -/


/-- The commutative group structure on `MulChar R R'`. -/
noncomputable instance commGroup : CommGroup (MulChar R R') :=
  { one := 1
    mul := (· * ·)
    inv := Inv.inv
    inv_mul_cancel := inv_mul
    mul_assoc := by
      /-
        R : Type u_1
        inst✝¹ : CommMonoid R
        R' : Type u_2
        inst✝ : CommMonoidWithZero R'
        ⊢ ∀ (a b c : MulChar R R'), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMu …
      -/
      intro χ₁ χ₂ χ₃
      /-
        R : Type u_1
        inst✝¹ : CommMonoid R
        R' : Type u_2
        inst✝ : CommMonoidWithZero R'
        χ₁ χ₂ χ₃ : MulChar R R'
        ⊢ Eq (HMul.hMul (HMul.hMul χ₁ χ₂) χ₃) (HMul.hMul χ₁ (HMul.hMul χ₂ χ₃))
      -/
      ext a
      /-
        case h
        R : Type u_1
        inst✝¹ : CommMonoid R
        R' : Type u_2
        inst✝ : CommMonoidWithZero R'
        χ₁ χ₂ χ₃ : MulChar R R'
        a : Units R
        ⊢ Eq ((HMul.hMul (HMul.hMul χ₁ χ₂) χ₃) ↑a) ((HMul.hMul χ₁ (HMul.hMul χ₂ χ₃)) ↑a)
      -/
      simp only [mul_assoc, Pi.mul_apply, MulChar.coeToFun_mul]
      /-
        🎉 no goals
      -/
    mul_comm := by
      /-
        R : Type u_1
        inst✝¹ : CommMonoid R
        R' : Type u_2
        inst✝ : CommMonoidWithZero R'
        ⊢ ∀ (a b : MulChar R R'), Eq (HMul.hMul a b) (HMul.hMul b a)
      -/
      intro χ₁ χ₂
      /-
        R : Type u_1
        inst✝¹ : CommMonoid R
        R' : Type u_2
        inst✝ : CommMonoidWithZero R'
        χ₁ χ₂ : MulChar R R'
        ⊢ Eq (HMul.hMul χ₁ χ₂) (HMul.hMul χ₂ χ₁)
      -/
      ext a
      /-
        case h
        R : Type u_1
        inst✝¹ : CommMonoid R
        R' : Type u_2
        inst✝ : CommMonoidWithZero R'
        χ₁ χ₂ : MulChar R R'
        a : Units R
        ⊢ Eq ((HMul.hMul χ₁ χ₂) ↑a) ((HMul.hMul χ₂ χ₁) ↑a)
      -/
      simp only [mul_comm, Pi.mul_apply, MulChar.coeToFun_mul]
      /-
        🎉 no goals
      -/
    one_mul := MulChar.one_mul
    mul_one := MulChar.mul_one }


/-- If `a` is a unit and `n : ℕ`, then `(χ ^ n) a = (χ a) ^ n`. -/
theorem pow_apply_coe (χ : MulChar R R') (n : ℕ) (a : Rˣ) : (χ ^ n) a = χ a ^ n := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ : MulChar R R'
    n : Nat
    a : Units R
    ⊢ Eq ((HPow.hPow χ n) ↑a) (HPow.hPow (χ ↑a) n)
  -/
  induction' n with n ih
    /-
      case zero
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      χ : MulChar R R'
      a : Units R
      ⊢ Eq ((HPow.hPow χ 0) ↑a) (HPow.hPow (χ ↑a) 0)
    -/
  · rw [pow_zero, pow_zero, one_apply_coe]
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      χ : MulChar R R'
      a : Units R
      n : Nat
      ih : Eq ((HPow.hPow χ n) ↑a) (HPow.hPow (χ ↑a) n)
      ⊢ Eq ((HPow.hPow χ (HAdd.hAdd n 1)) ↑a) (HPow.hPow (χ ↑a) (HAdd.hAdd n 1))
    -/
  · rw [pow_succ, pow_succ, mul_apply, ih]
    /-
      🎉 no goals
    -/


/-- If `n` is positive, then `(χ ^ n) a = (χ a) ^ n`. -/
theorem pow_apply' (χ : MulChar R R') {n : ℕ} (hn : n ≠ 0) (a : R) : (χ ^ n) a = χ a ^ n := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ : MulChar R R'
    n : Nat
    hn : Ne n 0
    a : R
    ⊢ Eq ((HPow.hPow χ n) a) (HPow.hPow (χ a) n)
  -/
  by_cases ha : IsUnit a
    /-
      case pos
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      χ : MulChar R R'
      n : Nat
      hn : Ne n 0
      a : R
      ha : IsUnit a
      ⊢ Eq ((HPow.hPow χ n) a) (HPow.hPow (χ a) n)
    -/
  · exact pow_apply_coe χ n ha.unit
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommMonoidWithZero R'
      χ : MulChar R R'
      n : Nat
      hn : Ne n 0
      a : R
      ha : Not (IsUnit a)
      ⊢ Eq ((HPow.hPow χ n) a) (HPow.hPow (χ a) n)
    -/
  · rw [map_nonunit (χ ^ n) ha, map_nonunit χ ha, zero_pow hn]
    /-
      🎉 no goals
    -/


lemma equivToUnitHom_mul_apply (χ₁ χ₂ : MulChar R R') (a : Rˣ) :
    equivToUnitHom (χ₁ * χ₂) a = equivToUnitHom χ₁ a * equivToUnitHom χ₂ a := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ₁ χ₂ : MulChar R R'
    a : Units R
    ⊢ Eq ((MulChar.equivToUnitHom (HMul.hMul χ₁ χ₂)) a) (HMul.hMul ((MulChar.equiv …
  -/
  apply_fun ((↑) : R'ˣ → R') using Units.ext
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ₁ χ₂ : MulChar R R'
    a : Units R
    ⊢ Eq ↑((MulChar.equivToUnitHom (HMul.hMul χ₁ χ₂)) a) ↑(HMul.hMul ((MulChar.equ …
  -/
  push_cast
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ₁ χ₂ : MulChar R R'
    a : Units R
    ⊢ Eq (↑((MulChar.equivToUnitHom (HMul.hMul χ₁ χ₂)) a)) (HMul.hMul ↑((MulChar.e …
  -/
  simp_rw [coe_equivToUnitHom, coeToFun_mul, Pi.mul_apply]
  /-
    🎉 no goals
  -/


/-- The equivalence between multiplicative characters and homomorphisms of unit groups
as a multiplicative equivalence. -/
noncomputable
def mulEquivToUnitHom : MulChar R R' ≃* (Rˣ →* R'ˣ) :=
  { equivToUnitHom with
    map_mul' := by
      /-
        R : Type u_1
        inst✝¹ : CommMonoid R
        R' : Type u_2
        inst✝ : CommMonoidWithZero R'
        ⊢ ∀ (x y : MulChar R R'), Eq (__src✝.toFun (HMul.hMul x y)) (HMul.hMul (__src✝ …
      -/
      intro χ ψ
      /-
        R : Type u_1
        inst✝¹ : CommMonoid R
        R' : Type u_2
        inst✝ : CommMonoidWithZero R'
        χ ψ : MulChar R R'
        ⊢ Eq (__src✝.toFun (HMul.hMul χ ψ)) (HMul.hMul (__src✝.toFun χ) (__src✝.toFun  …
      -/
      ext
      simp only [Equiv.toFun_as_coe, coe_equivToUnitHom, coeToFun_mul, Pi.mul_apply,
        MonoidHom.mul_apply, Units.val_mul]
  }


lemma eq_one_iff {χ : MulChar R R'} : χ = 1 ↔ ∀ a : Rˣ, χ a = 1 := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ : MulChar R R'
    ⊢ Iff (Eq χ 1) (∀ (a : Units R), Eq (χ ↑a) 1)
  -/
  simp only [MulChar.ext_iff, one_apply_coe]
  /-
    🎉 no goals
  -/


lemma ne_one_iff {χ : MulChar R R'} : χ ≠ 1 ↔ ∃ a : Rˣ, χ a ≠ 1 := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ : MulChar R R'
    ⊢ Iff (Ne χ 1) (Exists fun a => Ne (χ ↑a) 1)
  -/
  simp only [Ne, eq_one_iff, not_forall]
  /-
    🎉 no goals
  -/


/-- A multiplicative character is *nontrivial* if it takes a value `≠ 1` on a unit. -/
@[deprecated "No deprecation message was provided." (since := "2024-06-16")]
def IsNontrivial (χ : MulChar R R') : Prop :=
  ∃ a : Rˣ, χ a ≠ 1


set_option linter.deprecated false in
/-- A multiplicative character is nontrivial iff it is not the trivial character. -/
@[deprecated "No deprecation message was provided." (since := "2024-06-16")]
theorem isNontrivial_iff (χ : MulChar R R') : χ.IsNontrivial ↔ χ ≠ 1 := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommMonoidWithZero R'
    χ : MulChar R R'
    ⊢ Iff χ.IsNontrivial (Ne χ 1)
  -/
  simp only [IsNontrivial, Ne, MulChar.ext_iff, not_forall, one_apply_coe]
  /-
    🎉 no goals
  -/


/-- A multiplicative character is *quadratic* if it takes only the values `0`, `1`, `-1`. -/
def IsQuadratic (χ : MulChar R R') : Prop :=
  ∀ a, χ a = 0 ∨ χ a = 1 ∨ χ a = -1


/-- If two values of quadratic characters with target `ℤ` agree after coercion into a ring
of characteristic not `2`, then they agree in `ℤ`. -/
theorem IsQuadratic.eq_of_eq_coe {χ : MulChar R ℤ} (hχ : IsQuadratic χ) {χ' : MulChar R' ℤ}
    (hχ' : IsQuadratic χ') [Nontrivial R''] (hR'' : ringChar R'' ≠ 2) {a : R} {a' : R'}
    (h : (χ a : R'') = χ' a') : χ a = χ' a' :=
  Int.cast_injOn_of_ringChar_ne_two hR'' (hχ a) (hχ' a') h


/-- We can post-compose a multiplicative character with a ring homomorphism. -/
@[simps]
def ringHomComp (χ : MulChar R R') (f : R' →+* R'') : MulChar R R'' :=
  { f.toMonoidHom.comp χ.toMonoidHom with
    toFun := fun a => f (χ a)
                                   /-
                                     R : Type u_1
                                     inst✝² : CommMonoid R
                                     R' : Type u_2
                                     inst✝¹ : CommRing R'
                                     R'' : Type u_3
                                     inst✝ : CommRing R''
                                     χ : MulChar R R'
                                     f : RingHom R' R''
                                     a : R
                                     ha : Not (IsUnit a)
                                     ⊢ Eq ((↑{ toFun := fun a => f (χ a), map_one' := ⋯, map_mul' := ⋯ }).toFun a) 0
                                   -/
    map_nonunit' := fun a ha => by simp only [map_nonunit χ ha, map_zero] }
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
lemma ringHomComp_one (f : R' →+* R'') : (1 : MulChar R R').ringHomComp f = 1 := by
  /-
    R : Type u_1
    inst✝² : CommMonoid R
    R' : Type u_2
    inst✝¹ : CommRing R'
    R'' : Type u_3
    inst✝ : CommRing R''
    f : RingHom R' R''
    ⊢ Eq (MulChar.ringHomComp 1 f) 1
  -/
  ext1
  /-
    case h
    R : Type u_1
    inst✝² : CommMonoid R
    R' : Type u_2
    inst✝¹ : CommRing R'
    R'' : Type u_3
    inst✝ : CommRing R''
    f : RingHom R' R''
    a✝ : Units R
    ⊢ Eq ((MulChar.ringHomComp 1 f) ↑a✝) (1 ↑a✝)
  -/
  simp only [MulChar.ringHomComp_apply, MulChar.one_apply_coe, map_one]
  /-
    🎉 no goals
  -/


lemma ringHomComp_inv {R : Type*} [CommRing R] (χ : MulChar R R') (f : R' →+* R'') :
    (χ.ringHomComp f)⁻¹ = χ⁻¹.ringHomComp f := by
  /-
    R' : Type u_2
    inst✝² : CommRing R'
    R'' : Type u_3
    inst✝¹ : CommRing R''
    R : Type u_4
    inst✝ : CommRing R
    χ : MulChar R R'
    f : RingHom R' R''
    ⊢ Eq (Inv.inv (χ.ringHomComp f)) ((Inv.inv χ).ringHomComp f)
  -/
  ext1
  /-
    case h
    R' : Type u_2
    inst✝² : CommRing R'
    R'' : Type u_3
    inst✝¹ : CommRing R''
    R : Type u_4
    inst✝ : CommRing R
    χ : MulChar R R'
    f : RingHom R' R''
    a✝ : Units R
    ⊢ Eq ((Inv.inv (χ.ringHomComp f)) ↑a✝) (((Inv.inv χ).ringHomComp f) ↑a✝)
  -/
  simp only [inv_apply, Ring.inverse_unit, ringHomComp_apply]
  /-
    🎉 no goals
  -/


lemma ringHomComp_mul (χ φ : MulChar R R') (f : R' →+* R'') :
    (χ * φ).ringHomComp f = χ.ringHomComp f * φ.ringHomComp f := by
  /-
    R : Type u_1
    inst✝² : CommMonoid R
    R' : Type u_2
    inst✝¹ : CommRing R'
    R'' : Type u_3
    inst✝ : CommRing R''
    χ φ : MulChar R R'
    f : RingHom R' R''
    ⊢ Eq ((HMul.hMul χ φ).ringHomComp f) (HMul.hMul (χ.ringHomComp f) (φ.ringHomCo …
  -/
  ext1
  /-
    case h
    R : Type u_1
    inst✝² : CommMonoid R
    R' : Type u_2
    inst✝¹ : CommRing R'
    R'' : Type u_3
    inst✝ : CommRing R''
    χ φ : MulChar R R'
    f : RingHom R' R''
    a✝ : Units R
    ⊢ Eq (((HMul.hMul χ φ).ringHomComp f) ↑a✝) ((HMul.hMul (χ.ringHomComp f) (φ.ri …
  -/
  simp only [ringHomComp_apply, coeToFun_mul, Pi.mul_apply, map_mul]
  /-
    🎉 no goals
  -/


lemma ringHomComp_pow (χ : MulChar R R') (f : R' →+* R'') (n : ℕ) :
    χ.ringHomComp f ^ n = (χ ^ n).ringHomComp f := by
  induction n with
  | zero => simp only [pow_zero, ringHomComp_one]
  | succ n ih => simp only [pow_succ, ih, ringHomComp_mul]


lemma injective_ringHomComp {f : R' →+* R''} (hf : Function.Injective f) :
    Function.Injective (ringHomComp (R := R) · f) := by
  simpa
    only [Function.Injective, MulChar.ext_iff, ringHomComp, coe_mk, MonoidHom.coe_mk, OneHom.coe_mk]
    using fun χ χ' h a ↦ hf (h a)


lemma ringHomComp_eq_one_iff {f : R' →+* R''} (hf : Function.Injective f) {χ : MulChar R R'} :
    χ.ringHomComp f = 1 ↔ χ = 1 := by
  /-
    R : Type u_1
    inst✝² : CommMonoid R
    R' : Type u_2
    inst✝¹ : CommRing R'
    R'' : Type u_3
    inst✝ : CommRing R''
    f : RingHom R' R''
    hf : Function.Injective ⇑f
    χ : MulChar R R'
    ⊢ Iff (Eq (χ.ringHomComp f) 1) (Eq χ 1)
  -/
  conv_lhs => rw [← (show (1 : MulChar R R').ringHomComp f = 1 by ext; simp)]
  /-
    R : Type u_1
    inst✝² : CommMonoid R
    R' : Type u_2
    inst✝¹ : CommRing R'
    R'' : Type u_3
    inst✝ : CommRing R''
    f : RingHom R' R''
    hf : Function.Injective ⇑f
    χ : MulChar R R'
    ⊢ Iff (Eq (χ.ringHomComp f) (MulChar.ringHomComp 1 f)) (Eq χ 1)
  -/
  exact (injective_ringHomComp hf).eq_iff
  /-
    🎉 no goals
  -/


lemma ringHomComp_ne_one_iff {f : R' →+* R''} (hf : Function.Injective f) {χ : MulChar R R'} :
    χ.ringHomComp f ≠ 1 ↔ χ ≠ 1 :=
  (ringHomComp_eq_one_iff hf).not


set_option linter.deprecated false in
/-- Composition with an injective ring homomorphism preserves nontriviality. -/
@[deprecated ringHomComp_ne_one_iff (since := "2024-06-16")]
theorem IsNontrivial.comp {χ : MulChar R R'} (hχ : χ.IsNontrivial) {f : R' →+* R''}
    (hf : Function.Injective f) : (χ.ringHomComp f).IsNontrivial := by
  /-
    R : Type u_1
    inst✝² : CommMonoid R
    R' : Type u_2
    inst✝¹ : CommRing R'
    R'' : Type u_3
    inst✝ : CommRing R''
    χ : MulChar R R'
    hχ : χ.IsNontrivial
    f : RingHom R' R''
    hf : Function.Injective ⇑f
    ⊢ (χ.ringHomComp f).IsNontrivial
  -/
  obtain ⟨a, ha⟩ := hχ
  /-
    case intro
    R : Type u_1
    inst✝² : CommMonoid R
    R' : Type u_2
    inst✝¹ : CommRing R'
    R'' : Type u_3
    inst✝ : CommRing R''
    χ : MulChar R R'
    f : RingHom R' R''
    hf : Function.Injective ⇑f
    a : Units R
    ha : Ne (χ ↑a) 1
    ⊢ (χ.ringHomComp f).IsNontrivial
  -/
  use a
  /-
    case h
    R : Type u_1
    inst✝² : CommMonoid R
    R' : Type u_2
    inst✝¹ : CommRing R'
    R'' : Type u_3
    inst✝ : CommRing R''
    χ : MulChar R R'
    f : RingHom R' R''
    hf : Function.Injective ⇑f
    a : Units R
    ha : Ne (χ ↑a) 1
    ⊢ Ne ((χ.ringHomComp f) ↑a) 1
  -/
  simp_rw [ringHomComp_apply, ← RingHom.map_one f]
  /-
    case h
    R : Type u_1
    inst✝² : CommMonoid R
    R' : Type u_2
    inst✝¹ : CommRing R'
    R'' : Type u_3
    inst✝ : CommRing R''
    χ : MulChar R R'
    f : RingHom R' R''
    hf : Function.Injective ⇑f
    a : Units R
    ha : Ne (χ ↑a) 1
    ⊢ Ne (f (χ ↑a)) (f 1)
  -/
  exact fun h => ha (hf h)
  /-
    🎉 no goals
  -/


/-- Composition with a ring homomorphism preserves the property of being a quadratic character. -/
theorem IsQuadratic.comp {χ : MulChar R R'} (hχ : χ.IsQuadratic) (f : R' →+* R'') :
    (χ.ringHomComp f).IsQuadratic := by
  /-
    R : Type u_1
    inst✝² : CommMonoid R
    R' : Type u_2
    inst✝¹ : CommRing R'
    R'' : Type u_3
    inst✝ : CommRing R''
    χ : MulChar R R'
    hχ : χ.IsQuadratic
    f : RingHom R' R''
    ⊢ (χ.ringHomComp f).IsQuadratic
  -/
  intro a
  /-
    R : Type u_1
    inst✝² : CommMonoid R
    R' : Type u_2
    inst✝¹ : CommRing R'
    R'' : Type u_3
    inst✝ : CommRing R''
    χ : MulChar R R'
    hχ : χ.IsQuadratic
    f : RingHom R' R''
    a : R
    ⊢ Or (Eq ((χ.ringHomComp f) a) 0) (Or (Eq ((χ.ringHomComp f) a) 1) (Eq ((χ.rin …
  -/
                                      /-
                                        🎉 no goals
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
  rcases hχ a with (ha | ha | ha) <;> simp [ha]
                                      /-
                                        🎉 no goals
                                      -/


/-- The inverse of a quadratic character is itself. →  -/
theorem IsQuadratic.inv {χ : MulChar R R'} (hχ : χ.IsQuadratic) : χ⁻¹ = χ := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommRing R'
    χ : MulChar R R'
    hχ : χ.IsQuadratic
    ⊢ Eq (Inv.inv χ) χ
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommRing R'
    χ : MulChar R R'
    hχ : χ.IsQuadratic
    x : Units R
    ⊢ Eq ((Inv.inv χ) ↑x) (χ ↑x)
  -/
  rw [inv_apply_eq_inv]
  /-
    case h
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommRing R'
    χ : MulChar R R'
    hχ : χ.IsQuadratic
    x : Units R
    ⊢ Eq (Ring.inverse (χ ↑x)) (χ ↑x)
  -/
  rcases hχ x with (h₀ | h₁ | h₂)
    /-
      case h.inl
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommRing R'
      χ : MulChar R R'
      hχ : χ.IsQuadratic
      x : Units R
      h₀ : Eq (χ ↑x) 0
      ⊢ Eq (Ring.inverse (χ ↑x)) (χ ↑x)
    -/
  · rw [h₀, Ring.inverse_zero]
    /-
      🎉 no goals
    -/
    /-
      case h.inr.inl
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommRing R'
      χ : MulChar R R'
      hχ : χ.IsQuadratic
      x : Units R
      h₁ : Eq (χ ↑x) 1
      ⊢ Eq (Ring.inverse (χ ↑x)) (χ ↑x)
    -/
  · rw [h₁, Ring.inverse_one]
    /-
      🎉 no goals
    -/
  · -- Porting note: was `by norm_cast`
    /-
      case h.inr.inr
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommRing R'
      χ : MulChar R R'
      hχ : χ.IsQuadratic
      x : Units R
      h₂ : Eq (χ ↑x) (-1)
      ⊢ Eq (Ring.inverse (χ ↑x)) (χ ↑x)
    -/
    have : (-1 : R') = (-1 : R'ˣ) := by rw [Units.val_neg, Units.val_one]
    /-
      case h.inr.inr
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommRing R'
      χ : MulChar R R'
      hχ : χ.IsQuadratic
      x : Units R
      h₂ : Eq (χ ↑x) (-1)
      this : Eq (-1) ↑(-1)
      ⊢ Eq (Ring.inverse (χ ↑x)) (χ ↑x)
    -/
    rw [h₂, this, Ring.inverse_unit (-1 : R'ˣ)]
    /-
      case h.inr.inr
      R : Type u_1
      inst✝¹ : CommMonoid R
      R' : Type u_2
      inst✝ : CommRing R'
      χ : MulChar R R'
      hχ : χ.IsQuadratic
      x : Units R
      h₂ : Eq (χ ↑x) (-1)
      this : Eq (-1) ↑(-1)
      ⊢ Eq ↑(Inv.inv (-1)) ↑(-1)
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The square of a quadratic character is the trivial character. -/
theorem IsQuadratic.sq_eq_one {χ : MulChar R R'} (hχ : χ.IsQuadratic) : χ ^ 2 = 1 := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommRing R'
    χ : MulChar R R'
    hχ : χ.IsQuadratic
    ⊢ Eq (HPow.hPow χ 2) 1
  -/
  rw [← inv_mul_cancel χ, pow_two, hχ.inv]
  /-
    🎉 no goals
  -/


/-- The `p`th power of a quadratic character is itself, when `p` is the (prime) characteristic
of the target ring. -/
theorem IsQuadratic.pow_char {χ : MulChar R R'} (hχ : χ.IsQuadratic) (p : ℕ) [hp : Fact p.Prime]
    [CharP R' p] : χ ^ p = χ := by
  /-
    R : Type u_1
    inst✝² : CommMonoid R
    R' : Type u_2
    inst✝¹ : CommRing R'
    χ : MulChar R R'
    hχ : χ.IsQuadratic
    p : Nat
    hp : Fact (Nat.Prime p)
    inst✝ : CharP R' p
    ⊢ Eq (HPow.hPow χ p) χ
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝² : CommMonoid R
    R' : Type u_2
    inst✝¹ : CommRing R'
    χ : MulChar R R'
    hχ : χ.IsQuadratic
    p : Nat
    hp : Fact (Nat.Prime p)
    inst✝ : CharP R' p
    x : Units R
    ⊢ Eq ((HPow.hPow χ p) ↑x) (χ ↑x)
  -/
  rw [pow_apply_coe]
  /-
    case h
    R : Type u_1
    inst✝² : CommMonoid R
    R' : Type u_2
    inst✝¹ : CommRing R'
    χ : MulChar R R'
    hχ : χ.IsQuadratic
    p : Nat
    hp : Fact (Nat.Prime p)
    inst✝ : CharP R' p
    x : Units R
    ⊢ Eq (HPow.hPow (χ ↑x) p) (χ ↑x)
  -/
  rcases hχ x with (hx | hx | hx) <;> rw [hx]
    /-
      case h.inl
      R : Type u_1
      inst✝² : CommMonoid R
      R' : Type u_2
      inst✝¹ : CommRing R'
      χ : MulChar R R'
      hχ : χ.IsQuadratic
      p : Nat
      hp : Fact (Nat.Prime p)
      inst✝ : CharP R' p
      x : Units R
      hx : Eq (χ ↑x) 0
      ⊢ Eq (HPow.hPow 0 p) 0
    -/
  · rw [zero_pow (@Fact.out p.Prime).ne_zero]
    /-
      🎉 no goals
    -/
    /-
      case h.inr.inl
      R : Type u_1
      inst✝² : CommMonoid R
      R' : Type u_2
      inst✝¹ : CommRing R'
      χ : MulChar R R'
      hχ : χ.IsQuadratic
      p : Nat
      hp : Fact (Nat.Prime p)
      inst✝ : CharP R' p
      x : Units R
      hx : Eq (χ ↑x) 1
      ⊢ Eq (HPow.hPow 1 p) 1
    -/
  · rw [one_pow]
    /-
      🎉 no goals
    -/
    /-
      case h.inr.inr
      R : Type u_1
      inst✝² : CommMonoid R
      R' : Type u_2
      inst✝¹ : CommRing R'
      χ : MulChar R R'
      hχ : χ.IsQuadratic
      p : Nat
      hp : Fact (Nat.Prime p)
      inst✝ : CharP R' p
      x : Units R
      hx : Eq (χ ↑x) (-1)
      ⊢ Eq (HPow.hPow (-1) p) (-1)
    -/
  · exact neg_one_pow_char R' p
    /-
      🎉 no goals
    -/


/-- The `n`th power of a quadratic character is the trivial character, when `n` is even. -/
theorem IsQuadratic.pow_even {χ : MulChar R R'} (hχ : χ.IsQuadratic) {n : ℕ} (hn : Even n) :
    χ ^ n = 1 := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommRing R'
    χ : MulChar R R'
    hχ : χ.IsQuadratic
    n : Nat
    hn : Even n
    ⊢ Eq (HPow.hPow χ n) 1
  -/
  obtain ⟨n, rfl⟩ := even_iff_two_dvd.mp hn
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommRing R'
    χ : MulChar R R'
    hχ : χ.IsQuadratic
    n : Nat
    hn : Even (HMul.hMul 2 n)
    ⊢ Eq (HPow.hPow χ (HMul.hMul 2 n)) 1
  -/
  rw [pow_mul, hχ.sq_eq_one, one_pow]
  /-
    🎉 no goals
  -/


/-- The `n`th power of a quadratic character is itself, when `n` is odd. -/
theorem IsQuadratic.pow_odd {χ : MulChar R R'} (hχ : χ.IsQuadratic) {n : ℕ} (hn : Odd n) :
    χ ^ n = χ := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommRing R'
    χ : MulChar R R'
    hχ : χ.IsQuadratic
    n : Nat
    hn : Odd n
    ⊢ Eq (HPow.hPow χ n) χ
  -/
  obtain ⟨n, rfl⟩ := hn
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommMonoid R
    R' : Type u_2
    inst✝ : CommRing R'
    χ : MulChar R R'
    hχ : χ.IsQuadratic
    n : Nat
    ⊢ Eq (HPow.hPow χ (HAdd.hAdd (HMul.hMul 2 n) 1)) χ
  -/
  rw [pow_add, pow_one, hχ.pow_even (even_two_mul _), one_mul]
  /-
    🎉 no goals
  -/


/-- A multiplicative character `χ` into an integral domain is quadratic
if and only if `χ^2 = 1`. -/
lemma isQuadratic_iff_sq_eq_one {M R : Type*} [CommMonoid M] [CommRing R] [NoZeroDivisors R]
    [Nontrivial R] {χ : MulChar M R} :
    IsQuadratic χ ↔ χ ^ 2 = 1:= by
  /-
    M : Type u_4
    R : Type u_5
    inst✝³ : CommMonoid M
    inst✝² : CommRing R
    inst✝¹ : NoZeroDivisors R
    inst✝ : Nontrivial R
    χ : MulChar M R
    ⊢ Iff χ.IsQuadratic (Eq (HPow.hPow χ 2) 1)
  -/
  refine ⟨fun h ↦ ext (fun x ↦ ?_), fun h x ↦ ?_⟩
    /-
      case refine_1
      M : Type u_4
      R : Type u_5
      inst✝³ : CommMonoid M
      inst✝² : CommRing R
      inst✝¹ : NoZeroDivisors R
      inst✝ : Nontrivial R
      χ : MulChar M R
      h : χ.IsQuadratic
      x : Units M
      ⊢ Eq ((HPow.hPow χ 2) ↑x) (1 ↑x)
    -/
  · rw [one_apply_coe, χ.pow_apply_coe]
    /-
      case refine_1
      M : Type u_4
      R : Type u_5
      inst✝³ : CommMonoid M
      inst✝² : CommRing R
      inst✝¹ : NoZeroDivisors R
      inst✝ : Nontrivial R
      χ : MulChar M R
      h : χ.IsQuadratic
      x : Units M
      ⊢ Eq (HPow.hPow (χ ↑x) 2) 1
    -/
    rcases h x with H | H | H
      /-
        case refine_1.inl
        M : Type u_4
        R : Type u_5
        inst✝³ : CommMonoid M
        inst✝² : CommRing R
        inst✝¹ : NoZeroDivisors R
        inst✝ : Nontrivial R
        χ : MulChar M R
        h : χ.IsQuadratic
        x : Units M
        H : Eq (χ ↑x) 0
        ⊢ Eq (HPow.hPow (χ ↑x) 2) 1
      -/
    · exact (not_isUnit_zero <| H ▸ IsUnit.map χ <| x.isUnit).elim
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr.inl
        M : Type u_4
        R : Type u_5
        inst✝³ : CommMonoid M
        inst✝² : CommRing R
        inst✝¹ : NoZeroDivisors R
        inst✝ : Nontrivial R
        χ : MulChar M R
        h : χ.IsQuadratic
        x : Units M
        H : Eq (χ ↑x) 1
        ⊢ Eq (HPow.hPow (χ ↑x) 2) 1
      -/
    · simp only [H, one_pow]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr.inr
        M : Type u_4
        R : Type u_5
        inst✝³ : CommMonoid M
        inst✝² : CommRing R
        inst✝¹ : NoZeroDivisors R
        inst✝ : Nontrivial R
        χ : MulChar M R
        h : χ.IsQuadratic
        x : Units M
        H : Eq (χ ↑x) (-1)
        ⊢ Eq (HPow.hPow (χ ↑x) 2) 1
      -/
    · simp only [H, even_two, Even.neg_pow, one_pow]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      M : Type u_4
      R : Type u_5
      inst✝³ : CommMonoid M
      inst✝² : CommRing R
      inst✝¹ : NoZeroDivisors R
      inst✝ : Nontrivial R
      χ : MulChar M R
      h : Eq (HPow.hPow χ 2) 1
      x : M
      ⊢ Or (Eq (χ x) 0) (Or (Eq (χ x) 1) (Eq (χ x) (-1)))
    -/
  · by_cases hx : IsUnit x
      /-
        case pos
        M : Type u_4
        R : Type u_5
        inst✝³ : CommMonoid M
        inst✝² : CommRing R
        inst✝¹ : NoZeroDivisors R
        inst✝ : Nontrivial R
        χ : MulChar M R
        h : Eq (HPow.hPow χ 2) 1
        x : M
        hx : IsUnit x
        ⊢ Or (Eq (χ x) 0) (Or (Eq (χ x) 1) (Eq (χ x) (-1)))
      -/
    · refine .inr <| sq_eq_one_iff.mp ?_
      /-
        case pos
        M : Type u_4
        R : Type u_5
        inst✝³ : CommMonoid M
        inst✝² : CommRing R
        inst✝¹ : NoZeroDivisors R
        inst✝ : Nontrivial R
        χ : MulChar M R
        h : Eq (HPow.hPow χ 2) 1
        x : M
        hx : IsUnit x
        ⊢ Eq (HPow.hPow (χ x) 2) 1
      -/
      rw [← χ.pow_apply' two_ne_zero, h, MulChar.one_apply hx]
      /-
        🎉 no goals
      -/
      /-
        case neg
        M : Type u_4
        R : Type u_5
        inst✝³ : CommMonoid M
        inst✝² : CommRing R
        inst✝¹ : NoZeroDivisors R
        inst✝ : Nontrivial R
        χ : MulChar M R
        h : Eq (HPow.hPow χ 2) 1
        x : M
        hx : Not (IsUnit x)
        ⊢ Or (Eq (χ x) 0) (Or (Eq (χ x) 1) (Eq (χ x) (-1)))
      -/
    · exact .inl <| map_nonunit χ hx
      /-
        🎉 no goals
      -/


/-- If `χ` is a multiplicative character on a commutative monoid `M` with finitely many units,
then `χ ^ #Mˣ = 1`. -/
protected lemma pow_card_eq_one [Fintype Mˣ] (χ : MulChar M R) : χ ^ (Fintype.card Mˣ) = 1 := by
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    R : Type u_2
    inst✝¹ : CommMonoidWithZero R
    inst✝ : Fintype (Units M)
    χ : MulChar M R
    ⊢ Eq (HPow.hPow χ (Fintype.card (Units M))) 1
  -/
  ext1
  rw [pow_apply_coe, ← map_pow, one_apply_coe, ← Units.val_pow_eq_pow_val, pow_card_eq_one,
    Units.val_eq_one.mpr rfl, map_one]


/-- A multiplicative character on a commutative monoid with finitely many units
has finite (= positive) order. -/
lemma orderOf_pos [Finite Mˣ] (χ : MulChar M R) : 0 < orderOf χ := by
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    R : Type u_2
    inst✝¹ : CommMonoidWithZero R
    inst✝ : Finite (Units M)
    χ : MulChar M R
    ⊢ LT.lt 0 (orderOf χ)
  -/
  cases nonempty_fintype Mˣ
  /-
    case intro
    M : Type u_1
    inst✝² : CommMonoid M
    R : Type u_2
    inst✝¹ : CommMonoidWithZero R
    inst✝ : Finite (Units M)
    χ : MulChar M R
    val✝ : Fintype (Units M)
    ⊢ LT.lt 0 (orderOf χ)
  -/
  apply IsOfFinOrder.orderOf_pos
  /-
    case intro.h
    M : Type u_1
    inst✝² : CommMonoid M
    R : Type u_2
    inst✝¹ : CommMonoidWithZero R
    inst✝ : Finite (Units M)
    χ : MulChar M R
    val✝ : Fintype (Units M)
    ⊢ IsOfFinOrder χ
  -/
  exact isOfFinOrder_iff_pow_eq_one.2 ⟨_, Fintype.card_pos, χ.pow_card_eq_one⟩
  /-
    🎉 no goals
  -/


/-- The sum over all values of a nontrivial multiplicative character on a finite ring is zero
(when the target is a domain). -/
theorem sum_eq_zero_of_ne_one [IsDomain R'] {χ : MulChar R R'} (hχ : χ ≠ 1) : ∑ a, χ a = 0 := by
  /-
    R : Type u_1
    inst✝³ : CommMonoid R
    inst✝² : Fintype R
    R' : Type u_2
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    χ : MulChar R R'
    hχ : Ne χ 1
    ⊢ Eq (Finset.univ.sum fun a => χ a) 0
  -/
  rcases ne_one_iff.mp hχ with ⟨b, hb⟩
  /-
    case intro
    R : Type u_1
    inst✝³ : CommMonoid R
    inst✝² : Fintype R
    R' : Type u_2
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    χ : MulChar R R'
    hχ : Ne χ 1
    b : Units R
    hb : Ne (χ ↑b) 1
    ⊢ Eq (Finset.univ.sum fun a => χ a) 0
  -/
  refine eq_zero_of_mul_eq_self_left hb ?_
  /-
    case intro
    R : Type u_1
    inst✝³ : CommMonoid R
    inst✝² : Fintype R
    R' : Type u_2
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    χ : MulChar R R'
    hχ : Ne χ 1
    b : Units R
    hb : Ne (χ ↑b) 1
    ⊢ Eq (HMul.hMul (χ ↑b) (Finset.univ.sum fun a => χ a)) (Finset.univ.sum fun a  …
  -/
  simpa only [Finset.mul_sum, ← map_mul] using b.mulLeft_bijective.sum_comp _
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-06-16")]
lemma IsNontrivial.sum_eq_zero [IsDomain R'] {χ : MulChar R R'} (hχ : χ.IsNontrivial) :
    ∑ a, χ a = 0 :=
  sum_eq_zero_of_ne_one ((isNontrivial_iff _).mp hχ)


/-- The sum over all values of the trivial multiplicative character on a finite ring is
the cardinality of its unit group. -/
theorem sum_one_eq_card_units [DecidableEq R] :
    (∑ a, (1 : MulChar R R') a) = Fintype.card Rˣ := by
  calc
    (∑ a, (1 : MulChar R R') a) = ∑ a : R, if IsUnit a then 1 else 0 :=
      Finset.sum_congr rfl fun a _ => ?_
    _ = ((Finset.univ : Finset R).filter IsUnit).card := Finset.sum_boole _ _
    _ = (Finset.univ.map ⟨((↑) : Rˣ → R), Units.ext⟩).card := ?_
    _ = Fintype.card Rˣ := congr_arg _ (Finset.card_map _)
    /-
      case calc_1
      R : Type u_1
      inst✝³ : CommMonoid R
      inst✝² : Fintype R
      R' : Type u_2
      inst✝¹ : CommRing R'
      inst✝ : DecidableEq R
      a : R
      x✝ : Membership.mem Finset.univ a
      ⊢ Eq (1 a) (ite (IsUnit a) 1 0)
    -/
  · split_ifs with h
      /-
        case pos
        R : Type u_1
        inst✝³ : CommMonoid R
        inst✝² : Fintype R
        R' : Type u_2
        inst✝¹ : CommRing R'
        inst✝ : DecidableEq R
        a : R
        x✝ : Membership.mem Finset.univ a
        h : IsUnit a
        ⊢ Eq (1 a) 1
      -/
    · exact one_apply_coe h.unit
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝³ : CommMonoid R
        inst✝² : Fintype R
        R' : Type u_2
        inst✝¹ : CommRing R'
        inst✝ : DecidableEq R
        a : R
        x✝ : Membership.mem Finset.univ a
        h : Not (IsUnit a)
        ⊢ Eq (1 a) 0
      -/
    · exact map_nonunit _ h
      /-
        🎉 no goals
      -/
    /-
      case calc_2
      R : Type u_1
      inst✝³ : CommMonoid R
      inst✝² : Fintype R
      R' : Type u_2
      inst✝¹ : CommRing R'
      inst✝ : DecidableEq R
      ⊢ Eq ↑(Finset.filter IsUnit Finset.univ).card ↑(Finset.map { toFun := Units.va …
    -/
  · congr
    /-
      case calc_2.e_a.e_s
      R : Type u_1
      inst✝³ : CommMonoid R
      inst✝² : Fintype R
      R' : Type u_2
      inst✝¹ : CommRing R'
      inst✝ : DecidableEq R
      ⊢ Eq (Finset.filter IsUnit Finset.univ) (Finset.map { toFun := Units.val, inj' …
    -/
    ext a
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_map,
      Function.Embedding.coeFn_mk, exists_true_left, IsUnit]


/-- If `χ` is of odd order, then `χ(-1) = 1` -/
lemma val_neg_one_eq_one_of_odd_order {χ : MulChar R R'} {n : ℕ} (hn : Odd n) (hχ : χ ^ n = 1) :
    χ (-1) = 1 := by
  /-
    R : Type u_1
    R' : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommMonoidWithZero R'
    χ : MulChar R R'
    n : Nat
    hn : Odd n
    hχ : Eq (HPow.hPow χ n) 1
    ⊢ Eq (χ (-1)) 1
  -/
  rw [← hn.neg_one_pow, map_pow, ← χ.pow_apply' (Nat.ne_of_odd_add hn), hχ]
  /-
    R : Type u_1
    R' : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommMonoidWithZero R'
    χ : MulChar R R'
    n : Nat
    hn : Odd n
    hχ : Eq (HPow.hPow χ n) 1
    ⊢ Eq (1 (-1)) 1
  -/
  exact MulChar.one_apply_coe (-1)
  /-
    🎉 no goals
  -/


