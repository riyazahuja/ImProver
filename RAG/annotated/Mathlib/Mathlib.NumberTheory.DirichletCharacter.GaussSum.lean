lemma gaussSum_aux_of_mulShift (χ : DirichletCharacter R N) {d : ℕ}
    (hd : d ∣ N) (he : e.mulShift d = 1) {u : (ZMod N)ˣ} (hu : ZMod.unitsMap hd u = 1) :
    χ u * gaussSum χ e = gaussSum χ e := by
  /-
    N : Nat
    inst✝¹ : NeZero N
    R : Type u_1
    inst✝ : CommRing R
    e : AddChar (ZMod N) R
    χ : DirichletCharacter R N
    d : Nat
    hd : Dvd.dvd d N
    he : Eq (e.mulShift ↑d) 1
    u : Units (ZMod N)
    hu : Eq ((ZMod.unitsMap hd) u) 1
    ⊢ Eq (HMul.hMul (χ ↑u) (gaussSum χ e)) (gaussSum χ e)
  -/
  suffices e.mulShift u = e by conv_lhs => rw [← this, gaussSum_mulShift]
  /-
    N : Nat
    inst✝¹ : NeZero N
    R : Type u_1
    inst✝ : CommRing R
    e : AddChar (ZMod N) R
    χ : DirichletCharacter R N
    d : Nat
    hd : Dvd.dvd d N
    he : Eq (e.mulShift ↑d) 1
    u : Units (ZMod N)
    hu : Eq ((ZMod.unitsMap hd) u) 1
    ⊢ Eq (e.mulShift ↑u) e
  -/
  rw [(by ring : u.val = (u - 1) + 1), ← mulShift_mul, mulShift_one, mul_left_eq_self]
  /-
    N : Nat
    inst✝¹ : NeZero N
    R : Type u_1
    inst✝ : CommRing R
    e : AddChar (ZMod N) R
    χ : DirichletCharacter R N
    d : Nat
    hd : Dvd.dvd d N
    he : Eq (e.mulShift ↑d) 1
    u : Units (ZMod N)
    hu : Eq ((ZMod.unitsMap hd) u) 1
    ⊢ Eq (e.mulShift (HSub.hSub (↑u) 1)) 1
  -/
  rsuffices ⟨a, ha⟩ : (d : ℤ) ∣ (u.val.val - 1 : ℤ)
  · have : u.val - 1 = ↑(u.val.val - 1 : ℤ) := by simp only [ZMod.natCast_val, Int.cast_sub,
      ZMod.intCast_cast, ZMod.cast_id', id_eq, Int.cast_one]
    /-
      case intro
      N : Nat
      inst✝¹ : NeZero N
      R : Type u_1
      inst✝ : CommRing R
      e : AddChar (ZMod N) R
      χ : DirichletCharacter R N
      d : Nat
      hd : Dvd.dvd d N
      he : Eq (e.mulShift ↑d) 1
      u : Units (ZMod N)
      hu : Eq ((ZMod.unitsMap hd) u) 1
      a : Int
      ha : Eq (HSub.hSub (↑(↑u).val) 1) (HMul.hMul (↑d) a)
      this : Eq (HSub.hSub (↑u) 1) ↑(HSub.hSub (↑(↑u).val) 1)
      ⊢ Eq (e.mulShift (HSub.hSub (↑u) 1)) 1
    -/
    rw [this, ha]
    /-
      case intro
      N : Nat
      inst✝¹ : NeZero N
      R : Type u_1
      inst✝ : CommRing R
      e : AddChar (ZMod N) R
      χ : DirichletCharacter R N
      d : Nat
      hd : Dvd.dvd d N
      he : Eq (e.mulShift ↑d) 1
      u : Units (ZMod N)
      hu : Eq ((ZMod.unitsMap hd) u) 1
      a : Int
      ha : Eq (HSub.hSub (↑(↑u).val) 1) (HMul.hMul (↑d) a)
      this : Eq (HSub.hSub (↑u) 1) ↑(HSub.hSub (↑(↑u).val) 1)
      ⊢ Eq (e.mulShift ↑(HMul.hMul (↑d) a)) 1
    -/
    ext1 y
    simpa only [Int.cast_mul, Int.cast_natCast, mulShift_apply, mul_assoc, one_apply]
      using DFunLike.ext_iff.mp he (a * y)
  /-
    N : Nat
    inst✝¹ : NeZero N
    R : Type u_1
    inst✝ : CommRing R
    e : AddChar (ZMod N) R
    χ : DirichletCharacter R N
    d : Nat
    hd : Dvd.dvd d N
    he : Eq (e.mulShift ↑d) 1
    u : Units (ZMod N)
    hu : Eq ((ZMod.unitsMap hd) u) 1
    ⊢ Dvd.dvd (↑d) (HSub.hSub (↑(↑u).val) 1)
  -/
  rw [← Units.eq_iff, Units.val_one, ZMod.unitsMap_def, Units.coe_map] at hu
  /-
    N : Nat
    inst✝¹ : NeZero N
    R : Type u_1
    inst✝ : CommRing R
    e : AddChar (ZMod N) R
    χ : DirichletCharacter R N
    d : Nat
    hd : Dvd.dvd d N
    he : Eq (e.mulShift ↑d) 1
    u : Units (ZMod N)
    hu : Eq (↑(ZMod.castHom hd (ZMod d)) ↑u) 1
    ⊢ Dvd.dvd (↑d) (HSub.hSub (↑(↑u).val) 1)
  -/
  have : ZMod.castHom hd (ZMod d) u.val = ((u.val.val : ℤ) : ZMod d) := by simp
  rwa [MonoidHom.coe_coe, this, ← Int.cast_one, eq_comm,
    ZMod.intCast_eq_intCast_iff_dvd_sub] at hu


/-- If `gaussSum χ e ≠ 0`, and `d` is such that `e.mulShift d = 1`, then `χ` must factor through
`d`. (This will be used to show that Gauss sums vanish when `χ` is primitive and `e` is not.) -/
lemma factorsThrough_of_gaussSum_ne_zero [IsDomain R] {χ : DirichletCharacter R N} {d : ℕ}
    (hd : d ∣ N) (he : e.mulShift d = 1) (h_ne : gaussSum χ e ≠ 0) :
    χ.FactorsThrough d := by
  /-
    N : Nat
    inst✝² : NeZero N
    R : Type u_1
    inst✝¹ : CommRing R
    e : AddChar (ZMod N) R
    inst✝ : IsDomain R
    χ : DirichletCharacter R N
    d : Nat
    hd : Dvd.dvd d N
    he : Eq (e.mulShift ↑d) 1
    h_ne : Ne (gaussSum χ e) 0
    ⊢ χ.FactorsThrough d
  -/
  rw [DirichletCharacter.factorsThrough_iff_ker_unitsMap hd]
  /-
    N : Nat
    inst✝² : NeZero N
    R : Type u_1
    inst✝¹ : CommRing R
    e : AddChar (ZMod N) R
    inst✝ : IsDomain R
    χ : DirichletCharacter R N
    d : Nat
    hd : Dvd.dvd d N
    he : Eq (e.mulShift ↑d) 1
    h_ne : Ne (gaussSum χ e) 0
    ⊢ LE.le (ZMod.unitsMap hd).ker (MulChar.toUnitHom χ).ker
  -/
  intro u hu
  /-
    N : Nat
    inst✝² : NeZero N
    R : Type u_1
    inst✝¹ : CommRing R
    e : AddChar (ZMod N) R
    inst✝ : IsDomain R
    χ : DirichletCharacter R N
    d : Nat
    hd : Dvd.dvd d N
    he : Eq (e.mulShift ↑d) 1
    h_ne : Ne (gaussSum χ e) 0
    u : Units (ZMod N)
    hu : Membership.mem (ZMod.unitsMap hd).ker u
    ⊢ Membership.mem (MulChar.toUnitHom χ).ker u
  -/
  rw [MonoidHom.mem_ker, ← Units.eq_iff, MulChar.coe_toUnitHom]
  simpa only [Units.val_one, ne_eq, h_ne, not_false_eq_true, mul_eq_right₀] using
    gaussSum_aux_of_mulShift e χ hd he hu


/-- If `χ` is primitive, but `e` is not, then `gaussSum χ e = 0`. -/
lemma gaussSum_eq_zero_of_isPrimitive_of_not_isPrimitive [IsDomain R]
    {χ : DirichletCharacter R N} (hχ : IsPrimitive χ) (he : ¬IsPrimitive e) :
    gaussSum χ e = 0 := by
  /-
    N : Nat
    inst✝² : NeZero N
    R : Type u_1
    inst✝¹ : CommRing R
    e : AddChar (ZMod N) R
    inst✝ : IsDomain R
    χ : DirichletCharacter R N
    hχ : χ.IsPrimitive
    he : Not e.IsPrimitive
    ⊢ Eq (gaussSum χ e) 0
  -/
  contrapose! hχ
  /-
    N : Nat
    inst✝² : NeZero N
    R : Type u_1
    inst✝¹ : CommRing R
    e : AddChar (ZMod N) R
    inst✝ : IsDomain R
    χ : DirichletCharacter R N
    he : Not e.IsPrimitive
    hχ : Ne (gaussSum χ e) 0
    ⊢ Not χ.IsPrimitive
  -/
  rcases e.exists_divisor_of_not_isPrimitive he with ⟨d, hd₁, hd₂, hed⟩
  /-
    case intro.intro.intro
    N : Nat
    inst✝² : NeZero N
    R : Type u_1
    inst✝¹ : CommRing R
    e : AddChar (ZMod N) R
    inst✝ : IsDomain R
    χ : DirichletCharacter R N
    he : Not e.IsPrimitive
    hχ : Ne (gaussSum χ e) 0
    d : Nat
    hd₁ : Dvd.dvd d N
    hd₂ : LT.lt d N
    hed : Eq (e.mulShift ↑d) 1
    ⊢ Not χ.IsPrimitive
  -/
  have : χ.conductor ≤ d := Nat.sInf_le <| factorsThrough_of_gaussSum_ne_zero e hd₁ hed hχ
  /-
    case intro.intro.intro
    N : Nat
    inst✝² : NeZero N
    R : Type u_1
    inst✝¹ : CommRing R
    e : AddChar (ZMod N) R
    inst✝ : IsDomain R
    χ : DirichletCharacter R N
    he : Not e.IsPrimitive
    hχ : Ne (gaussSum χ e) 0
    d : Nat
    hd₁ : Dvd.dvd d N
    hd₂ : LT.lt d N
    hed : Eq (e.mulShift ↑d) 1
    this : LE.le χ.conductor d
    ⊢ Not χ.IsPrimitive
  -/
  exact (this.trans_lt hd₂).ne
  /-
    🎉 no goals
  -/


/-- If `χ` is a primitive character, then the function `a ↦ gaussSum χ (e.mulShift a)`, for any
fixed additive character `e`, is a constant multiple of `χ⁻¹`. -/
lemma gaussSum_mulShift_of_isPrimitive [IsDomain R] {χ : DirichletCharacter R N}
    (hχ : IsPrimitive χ) (a : ZMod N) :
    gaussSum χ (e.mulShift a) = χ⁻¹ a * gaussSum χ e := by
  /-
    N : Nat
    inst✝² : NeZero N
    R : Type u_1
    inst✝¹ : CommRing R
    e : AddChar (ZMod N) R
    inst✝ : IsDomain R
    χ : DirichletCharacter R N
    hχ : χ.IsPrimitive
    a : ZMod N
    ⊢ Eq (gaussSum χ (e.mulShift a)) (HMul.hMul ((Inv.inv χ) a) (gaussSum χ e))
  -/
  by_cases ha : IsUnit a
    /-
      case pos
      N : Nat
      inst✝² : NeZero N
      R : Type u_1
      inst✝¹ : CommRing R
      e : AddChar (ZMod N) R
      inst✝ : IsDomain R
      χ : DirichletCharacter R N
      hχ : χ.IsPrimitive
      a : ZMod N
      ha : IsUnit a
      ⊢ Eq (gaussSum χ (e.mulShift a)) (HMul.hMul ((Inv.inv χ) a) (gaussSum χ e))
    -/
  · conv_rhs => rw [← gaussSum_mulShift χ e ha.unit]
    /-
      case pos
      N : Nat
      inst✝² : NeZero N
      R : Type u_1
      inst✝¹ : CommRing R
      e : AddChar (ZMod N) R
      inst✝ : IsDomain R
      χ : DirichletCharacter R N
      hχ : χ.IsPrimitive
      a : ZMod N
      ha : IsUnit a
      ⊢ Eq (gaussSum χ (e.mulShift a)) (HMul.hMul ((Inv.inv χ) a) (HMul.hMul (χ ↑ha. …
    -/
    rw [IsUnit.unit_spec, MulChar.inv_apply_eq_inv, Ring.inverse_mul_cancel_left _ _ (ha.map χ)]
    /-
      🎉 no goals
    -/
    /-
      case neg
      N : Nat
      inst✝² : NeZero N
      R : Type u_1
      inst✝¹ : CommRing R
      e : AddChar (ZMod N) R
      inst✝ : IsDomain R
      χ : DirichletCharacter R N
      hχ : χ.IsPrimitive
      a : ZMod N
      ha : Not (IsUnit a)
      ⊢ Eq (gaussSum χ (e.mulShift a)) (HMul.hMul ((Inv.inv χ) a) (gaussSum χ e))
    -/
  · rw [MulChar.map_nonunit _ ha, zero_mul]
    /-
      case neg
      N : Nat
      inst✝² : NeZero N
      R : Type u_1
      inst✝¹ : CommRing R
      e : AddChar (ZMod N) R
      inst✝ : IsDomain R
      χ : DirichletCharacter R N
      hχ : χ.IsPrimitive
      a : ZMod N
      ha : Not (IsUnit a)
      ⊢ Eq (gaussSum χ (e.mulShift a)) 0
    -/
    exact gaussSum_eq_zero_of_isPrimitive_of_not_isPrimitive _ hχ (not_isPrimitive_mulShift e ha)
    /-
      🎉 no goals
    -/

