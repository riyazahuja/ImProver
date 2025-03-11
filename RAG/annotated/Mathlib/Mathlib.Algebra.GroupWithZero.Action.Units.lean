@[simp] lemma inv_smul_smul₀ (ha : a ≠ 0) (x : β) : a⁻¹ • a • x = x :=
  inv_smul_smul (Units.mk0 a ha) x


@[simp]
lemma smul_inv_smul₀ (ha : a ≠ 0) (x : β) : a • a⁻¹ • x = x := smul_inv_smul (Units.mk0 a ha) x


lemma inv_smul_eq_iff₀ (ha : a ≠ 0) {x y : β} : a⁻¹ • x = y ↔ x = a • y :=
  inv_smul_eq_iff (g := Units.mk0 a ha)


lemma eq_inv_smul_iff₀ (ha : a ≠ 0) {x y : β} : x = a⁻¹ • y ↔ a • x = y :=
  eq_inv_smul_iff (g := Units.mk0 a ha)


@[simp]
lemma Commute.smul_right_iff₀ [Mul β] [SMulCommClass α β β] [IsScalarTower α β β] {x y : β}
    (ha : a ≠ 0) : Commute x (a • y) ↔ Commute x y := Commute.smul_right_iff (g := Units.mk0 a ha)


@[simp]
lemma Commute.smul_left_iff₀ [Mul β] [SMulCommClass α β β] [IsScalarTower α β β] {x y : β}
    (ha : a ≠ 0) : Commute (a • x) y ↔ Commute x y := Commute.smul_left_iff (g := Units.mk0 a ha)


/-- Right scalar multiplication as an order isomorphism. -/
@[simps] def Equiv.smulRight (ha : a ≠ 0) : β ≃ β where
  toFun b := a • b
  invFun b := a⁻¹ • b
  left_inv := inv_smul_smul₀ ha
  right_inv := smul_inv_smul₀ ha


@[to_additive]
instance [Monoid M] [SMul M α] : SMul Mˣ α where smul m a := (m : M) • a


instance instSMulZeroClass [Monoid M] [Zero α] [SMulZeroClass M α] : SMulZeroClass Mˣ α where
  smul := (· • ·)
  smul_zero m := smul_zero (m : M)


instance instDistribSMulUnits [Monoid M] [AddZeroClass α] [DistribSMul M α] :
    DistribSMul Mˣ α where smul_add m := smul_add (m : M)


instance instDistribMulAction [Monoid M] [AddMonoid α] [DistribMulAction M α] :
    DistribMulAction Mˣ α where
  __ := instDistribSMulUnits
  one_smul := fun b => one_smul M b
  mul_smul := fun x y b => mul_smul (x : M) y b


instance instMulDistribMulAction [Monoid M] [Monoid α] [MulDistribMulAction M α] :
    MulDistribMulAction Mˣ α where
  smul_mul m := smul_mul' (m : M)
  smul_one m := smul_one (m : M)


/-- A stronger form of `Units.mul_action'`. -/
instance mulDistribMulAction' [Group G] [Monoid M] [MulDistribMulAction G M] [SMulCommClass G M M]
    [IsScalarTower G M M] : MulDistribMulAction G Mˣ :=
  { Units.mulAction' with
    smul := (· • ·),
    smul_one := fun _ => Units.ext <| smul_one _,
    smul_mul := fun _ _ _ => Units.ext <| smul_mul' _ _ _ }


@[simp] lemma IsUnit.smul_eq_zero (hu : IsUnit u) : u • x = 0 ↔ x = 0 := smul_eq_zero_iff_eq hu.unit


