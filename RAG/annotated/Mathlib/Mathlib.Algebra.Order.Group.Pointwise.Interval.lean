@[to_additive Icc_add_Icc_subset]
theorem Icc_mul_Icc_subset' (a b c d : α) : Icc a b * Icc c d ⊆ Icc (a * c) (b * d) := by
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : Preorder α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightMono α
    a b c d : α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Icc a b) (Set.Icc c d)) (Set.Icc (HMul.hMul …
  -/
  rintro x ⟨y, ⟨hya, hyb⟩, z, ⟨hzc, hzd⟩, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : Preorder α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightMono α
    a b c d y : α
    hya : LE.le a y
    hyb : LE.le y b
    z : α
    hzc : LE.le c z
    hzd : LE.le z d
    ⊢ Membership.mem (Set.Icc (HMul.hMul a c) (HMul.hMul b d)) ((fun x1 x2 => HMul …
  -/
  exact ⟨mul_le_mul' hya hzc, mul_le_mul' hyb hzd⟩
  /-
    🎉 no goals
  -/


@[to_additive Iic_add_Iic_subset]
theorem Iic_mul_Iic_subset' (a b : α) : Iic a * Iic b ⊆ Iic (a * b) := by
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : Preorder α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightMono α
    a b : α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Iic a) (Set.Iic b)) (Set.Iic (HMul.hMul a b))
  -/
  rintro x ⟨y, hya, z, hzb, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : Preorder α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightMono α
    a b y : α
    hya : Membership.mem (Set.Iic a) y
    z : α
    hzb : Membership.mem (Set.Iic b) z
    ⊢ Membership.mem (Set.Iic (HMul.hMul a b)) ((fun x1 x2 => HMul.hMul x1 x2) y z)
  -/
  exact mul_le_mul' hya hzb
  /-
    🎉 no goals
  -/


@[to_additive Ici_add_Ici_subset]
theorem Ici_mul_Ici_subset' (a b : α) : Ici a * Ici b ⊆ Ici (a * b) := by
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : Preorder α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightMono α
    a b : α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Ici a) (Set.Ici b)) (Set.Ici (HMul.hMul a b))
  -/
  rintro x ⟨y, hya, z, hzb, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : Preorder α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightMono α
    a b y : α
    hya : Membership.mem (Set.Ici a) y
    z : α
    hzb : Membership.mem (Set.Ici b) z
    ⊢ Membership.mem (Set.Ici (HMul.hMul a b)) ((fun x1 x2 => HMul.hMul x1 x2) y z)
  -/
  exact mul_le_mul' hya hzb
  /-
    🎉 no goals
  -/


@[to_additive Icc_add_Ico_subset]
theorem Icc_mul_Ico_subset' (a b c d : α) : Icc a b * Ico c d ⊆ Ico (a * c) (b * d) := by
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b c d : α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Icc a b) (Set.Ico c d)) (Set.Ico (HMul.hMul …
  -/
  have := mulLeftMono_of_mulLeftStrictMono α
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b c d : α
    this : MulLeftMono α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Icc a b) (Set.Ico c d)) (Set.Ico (HMul.hMul …
  -/
  have := mulRightMono_of_mulRightStrictMono α
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b c d : α
    this✝ : MulLeftMono α
    this : MulRightMono α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Icc a b) (Set.Ico c d)) (Set.Ico (HMul.hMul …
  -/
  rintro x ⟨y, ⟨hya, hyb⟩, z, ⟨hzc, hzd⟩, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b c d : α
    this✝ : MulLeftMono α
    this : MulRightMono α
    y : α
    hya : LE.le a y
    hyb : LE.le y b
    z : α
    hzc : LE.le c z
    hzd : LT.lt z d
    ⊢ Membership.mem (Set.Ico (HMul.hMul a c) (HMul.hMul b d)) ((fun x1 x2 => HMul …
  -/
  exact ⟨mul_le_mul' hya hzc, mul_lt_mul_of_le_of_lt hyb hzd⟩
  /-
    🎉 no goals
  -/


@[to_additive Ico_add_Icc_subset]
theorem Ico_mul_Icc_subset' (a b c d : α) : Ico a b * Icc c d ⊆ Ico (a * c) (b * d) := by
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b c d : α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Ico a b) (Set.Icc c d)) (Set.Ico (HMul.hMul …
  -/
  have := mulLeftMono_of_mulLeftStrictMono α
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b c d : α
    this : MulLeftMono α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Ico a b) (Set.Icc c d)) (Set.Ico (HMul.hMul …
  -/
  have := mulRightMono_of_mulRightStrictMono α
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b c d : α
    this✝ : MulLeftMono α
    this : MulRightMono α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Ico a b) (Set.Icc c d)) (Set.Ico (HMul.hMul …
  -/
  rintro x ⟨y, ⟨hya, hyb⟩, z, ⟨hzc, hzd⟩, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b c d : α
    this✝ : MulLeftMono α
    this : MulRightMono α
    y : α
    hya : LE.le a y
    hyb : LT.lt y b
    z : α
    hzc : LE.le c z
    hzd : LE.le z d
    ⊢ Membership.mem (Set.Ico (HMul.hMul a c) (HMul.hMul b d)) ((fun x1 x2 => HMul …
  -/
  exact ⟨mul_le_mul' hya hzc, mul_lt_mul_of_lt_of_le hyb hzd⟩
  /-
    🎉 no goals
  -/


@[to_additive Ioc_add_Ico_subset]
theorem Ioc_mul_Ico_subset' (a b c d : α) : Ioc a b * Ico c d ⊆ Ioo (a * c) (b * d) := by
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b c d : α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Ioc a b) (Set.Ico c d)) (Set.Ioo (HMul.hMul …
  -/
  have := mulLeftMono_of_mulLeftStrictMono α
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b c d : α
    this : MulLeftMono α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Ioc a b) (Set.Ico c d)) (Set.Ioo (HMul.hMul …
  -/
  have := mulRightMono_of_mulRightStrictMono α
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b c d : α
    this✝ : MulLeftMono α
    this : MulRightMono α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Ioc a b) (Set.Ico c d)) (Set.Ioo (HMul.hMul …
  -/
  rintro x ⟨y, ⟨hya, hyb⟩, z, ⟨hzc, hzd⟩, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b c d : α
    this✝ : MulLeftMono α
    this : MulRightMono α
    y : α
    hya : LT.lt a y
    hyb : LE.le y b
    z : α
    hzc : LE.le c z
    hzd : LT.lt z d
    ⊢ Membership.mem (Set.Ioo (HMul.hMul a c) (HMul.hMul b d)) ((fun x1 x2 => HMul …
  -/
  exact ⟨mul_lt_mul_of_lt_of_le hya hzc, mul_lt_mul_of_le_of_lt hyb hzd⟩
  /-
    🎉 no goals
  -/


@[to_additive Ico_add_Ioc_subset]
theorem Ico_mul_Ioc_subset' (a b c d : α) : Ico a b * Ioc c d ⊆ Ioo (a * c) (b * d) := by
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b c d : α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Ico a b) (Set.Ioc c d)) (Set.Ioo (HMul.hMul …
  -/
  have := mulLeftMono_of_mulLeftStrictMono α
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b c d : α
    this : MulLeftMono α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Ico a b) (Set.Ioc c d)) (Set.Ioo (HMul.hMul …
  -/
  have := mulRightMono_of_mulRightStrictMono α
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b c d : α
    this✝ : MulLeftMono α
    this : MulRightMono α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Ico a b) (Set.Ioc c d)) (Set.Ioo (HMul.hMul …
  -/
  rintro x ⟨y, ⟨hya, hyb⟩, z, ⟨hzc, hzd⟩, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b c d : α
    this✝ : MulLeftMono α
    this : MulRightMono α
    y : α
    hya : LE.le a y
    hyb : LT.lt y b
    z : α
    hzc : LT.lt c z
    hzd : LE.le z d
    ⊢ Membership.mem (Set.Ioo (HMul.hMul a c) (HMul.hMul b d)) ((fun x1 x2 => HMul …
  -/
  exact ⟨mul_lt_mul_of_le_of_lt hya hzc, mul_lt_mul_of_lt_of_le hyb hzd⟩
  /-
    🎉 no goals
  -/


@[to_additive Iic_add_Iio_subset]
theorem Iic_mul_Iio_subset' (a b : α) : Iic a * Iio b ⊆ Iio (a * b) := by
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b : α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Iic a) (Set.Iio b)) (Set.Iio (HMul.hMul a b))
  -/
  have := mulRightMono_of_mulRightStrictMono α
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b : α
    this : MulRightMono α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Iic a) (Set.Iio b)) (Set.Iio (HMul.hMul a b))
  -/
  rintro x ⟨y, hya, z, hzb, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b : α
    this : MulRightMono α
    y : α
    hya : Membership.mem (Set.Iic a) y
    z : α
    hzb : Membership.mem (Set.Iio b) z
    ⊢ Membership.mem (Set.Iio (HMul.hMul a b)) ((fun x1 x2 => HMul.hMul x1 x2) y z)
  -/
  exact mul_lt_mul_of_le_of_lt hya hzb
  /-
    🎉 no goals
  -/


@[to_additive Iio_add_Iic_subset]
theorem Iio_mul_Iic_subset' (a b : α) : Iio a * Iic b ⊆ Iio (a * b) := by
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b : α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Iio a) (Set.Iic b)) (Set.Iio (HMul.hMul a b))
  -/
  have := mulLeftMono_of_mulLeftStrictMono α
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b : α
    this : MulLeftMono α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Iio a) (Set.Iic b)) (Set.Iio (HMul.hMul a b))
  -/
  rintro x ⟨y, hya, z, hzb, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b : α
    this : MulLeftMono α
    y : α
    hya : Membership.mem (Set.Iio a) y
    z : α
    hzb : Membership.mem (Set.Iic b) z
    ⊢ Membership.mem (Set.Iio (HMul.hMul a b)) ((fun x1 x2 => HMul.hMul x1 x2) y z)
  -/
  exact mul_lt_mul_of_lt_of_le hya hzb
  /-
    🎉 no goals
  -/


@[to_additive Ioi_add_Ici_subset]
theorem Ioi_mul_Ici_subset' (a b : α) : Ioi a * Ici b ⊆ Ioi (a * b) := by
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b : α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Ioi a) (Set.Ici b)) (Set.Ioi (HMul.hMul a b))
  -/
  have := mulLeftMono_of_mulLeftStrictMono α
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b : α
    this : MulLeftMono α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Ioi a) (Set.Ici b)) (Set.Ioi (HMul.hMul a b))
  -/
  rintro x ⟨y, hya, z, hzb, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b : α
    this : MulLeftMono α
    y : α
    hya : Membership.mem (Set.Ioi a) y
    z : α
    hzb : Membership.mem (Set.Ici b) z
    ⊢ Membership.mem (Set.Ioi (HMul.hMul a b)) ((fun x1 x2 => HMul.hMul x1 x2) y z)
  -/
  exact mul_lt_mul_of_lt_of_le hya hzb
  /-
    🎉 no goals
  -/


@[to_additive Ici_add_Ioi_subset]
theorem Ici_mul_Ioi_subset' (a b : α) : Ici a * Ioi b ⊆ Ioi (a * b) := by
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b : α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Ici a) (Set.Ioi b)) (Set.Ioi (HMul.hMul a b))
  -/
  have := mulRightMono_of_mulRightStrictMono α
  /-
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b : α
    this : MulRightMono α
    ⊢ HasSubset.Subset (HMul.hMul (Set.Ici a) (Set.Ioi b)) (Set.Ioi (HMul.hMul a b))
  -/
  rintro x ⟨y, hya, z, hzb, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝³ : Mul α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a b : α
    this : MulRightMono α
    y : α
    hya : Membership.mem (Set.Ici a) y
    z : α
    hzb : Membership.mem (Set.Ioi b) z
    ⊢ Membership.mem (Set.Ioi (HMul.hMul a b)) ((fun x1 x2 => HMul.hMul x1 x2) y z)
  -/
  exact mul_lt_mul_of_le_of_lt hya hzb
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma smul_Icc (a b c : α) : a • Icc b c = Icc (a * b) (a * c) := by
  /-
    α : Type u_1
    inst✝² : LinearOrderedCommMonoid α
    inst✝¹ : MulLeftReflectLE α
    inst✝ : ExistsMulOfLE α
    a b c : α
    ⊢ Eq (HSMul.hSMul a (Set.Icc b c)) (Set.Icc (HMul.hMul a b) (HMul.hMul a c))
  -/
  ext x
  /-
    case h
    α : Type u_1
    inst✝² : LinearOrderedCommMonoid α
    inst✝¹ : MulLeftReflectLE α
    inst✝ : ExistsMulOfLE α
    a b c x : α
    ⊢ Iff (Membership.mem (HSMul.hSMul a (Set.Icc b c)) x) (Membership.mem (Set.Ic …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      inst✝² : LinearOrderedCommMonoid α
      inst✝¹ : MulLeftReflectLE α
      inst✝ : ExistsMulOfLE α
      a b c x : α
      ⊢ Membership.mem (HSMul.hSMul a (Set.Icc b c)) x → Membership.mem (Set.Icc (HM …
    -/
  · rintro ⟨y, ⟨hby, hyc⟩, rfl⟩
    /-
      case h.mp.intro.intro.intro
      α : Type u_1
      inst✝² : LinearOrderedCommMonoid α
      inst✝¹ : MulLeftReflectLE α
      inst✝ : ExistsMulOfLE α
      a b c y : α
      hby : LE.le b y
      hyc : LE.le y c
      ⊢ Membership.mem (Set.Icc (HMul.hMul a b) (HMul.hMul a c)) ((fun x => HSMul.hS …
    -/
    exact ⟨mul_le_mul_left' hby _, mul_le_mul_left' hyc _⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      inst✝² : LinearOrderedCommMonoid α
      inst✝¹ : MulLeftReflectLE α
      inst✝ : ExistsMulOfLE α
      a b c x : α
      ⊢ Membership.mem (Set.Icc (HMul.hMul a b) (HMul.hMul a c)) x → Membership.mem  …
    -/
  · rintro ⟨habx, hxac⟩
    /-
      case h.mpr.intro
      α : Type u_1
      inst✝² : LinearOrderedCommMonoid α
      inst✝¹ : MulLeftReflectLE α
      inst✝ : ExistsMulOfLE α
      a b c x : α
      habx : LE.le (HMul.hMul a b) x
      hxac : LE.le x (HMul.hMul a c)
      ⊢ Membership.mem (HSMul.hSMul a (Set.Icc b c)) x
    -/
    obtain ⟨y, hy, rfl⟩ := exists_one_le_mul_of_le habx
    /-
      case h.mpr.intro.intro.intro
      α : Type u_1
      inst✝² : LinearOrderedCommMonoid α
      inst✝¹ : MulLeftReflectLE α
      inst✝ : ExistsMulOfLE α
      a b c y : α
      hy : LE.le 1 y
      habx : LE.le (HMul.hMul a b) (HMul.hMul (HMul.hMul a b) y)
      hxac : LE.le (HMul.hMul (HMul.hMul a b) y) (HMul.hMul a c)
      ⊢ Membership.mem (HSMul.hSMul a (Set.Icc b c)) (HMul.hMul (HMul.hMul a b) y)
    -/
    refine ⟨b * y, ⟨le_mul_of_one_le_right' hy, ?_⟩, (mul_assoc ..).symm⟩
    /-
      case h.mpr.intro.intro.intro
      α : Type u_1
      inst✝² : LinearOrderedCommMonoid α
      inst✝¹ : MulLeftReflectLE α
      inst✝ : ExistsMulOfLE α
      a b c y : α
      hy : LE.le 1 y
      habx : LE.le (HMul.hMul a b) (HMul.hMul (HMul.hMul a b) y)
      hxac : LE.le (HMul.hMul (HMul.hMul a b) y) (HMul.hMul a c)
      ⊢ LE.le (HMul.hMul b y) c
    -/
    rwa [mul_assoc, mul_le_mul_iff_left] at hxac
    /-
      🎉 no goals
    -/


@[to_additive]
lemma Icc_mul_Icc (hab : a ≤ b) (hcd : c ≤ d) : Icc a b * Icc c d = Icc (a * c) (b * d) := by
  /-
    α : Type u_1
    inst✝² : LinearOrderedCommMonoid α
    inst✝¹ : MulLeftReflectLE α
    inst✝ : ExistsMulOfLE α
    a b c d : α
    hab : LE.le a b
    hcd : LE.le c d
    ⊢ Eq (HMul.hMul (Set.Icc a b) (Set.Icc c d)) (Set.Icc (HMul.hMul a c) (HMul.hM …
  -/
  refine (Icc_mul_Icc_subset' _ _ _ _).antisymm fun x ⟨hacx, hxbd⟩ ↦ ?_
  /-
    α : Type u_1
    inst✝² : LinearOrderedCommMonoid α
    inst✝¹ : MulLeftReflectLE α
    inst✝ : ExistsMulOfLE α
    a b c d : α
    hab : LE.le a b
    hcd : LE.le c d
    x : α
    x✝ : Membership.mem (Set.Icc (HMul.hMul a c) (HMul.hMul b d)) x
    hacx : LE.le (HMul.hMul a c) x
    hxbd : LE.le x (HMul.hMul b d)
    ⊢ Membership.mem (HMul.hMul (Set.Icc a b) (Set.Icc c d)) x
  -/
  obtain hxbc | hbcx := le_total x (b * c)
    /-
      case inl
      α : Type u_1
      inst✝² : LinearOrderedCommMonoid α
      inst✝¹ : MulLeftReflectLE α
      inst✝ : ExistsMulOfLE α
      a b c d : α
      hab : LE.le a b
      hcd : LE.le c d
      x : α
      x✝ : Membership.mem (Set.Icc (HMul.hMul a c) (HMul.hMul b d)) x
      hacx : LE.le (HMul.hMul a c) x
      hxbd : LE.le x (HMul.hMul b d)
      hxbc : LE.le x (HMul.hMul b c)
      ⊢ Membership.mem (HMul.hMul (Set.Icc a b) (Set.Icc c d)) x
    -/
  · obtain ⟨y, hy, rfl⟩ := exists_one_le_mul_of_le hacx
    /-
      case inl.intro.intro
      α : Type u_1
      inst✝² : LinearOrderedCommMonoid α
      inst✝¹ : MulLeftReflectLE α
      inst✝ : ExistsMulOfLE α
      a b c d : α
      hab : LE.le a b
      hcd : LE.le c d
      y : α
      hy : LE.le 1 y
      x✝ : Membership.mem (Set.Icc (HMul.hMul a c) (HMul.hMul b d)) (HMul.hMul (HMul …
      hacx : LE.le (HMul.hMul a c) (HMul.hMul (HMul.hMul a c) y)
      hxbd : LE.le (HMul.hMul (HMul.hMul a c) y) (HMul.hMul b d)
      hxbc : LE.le (HMul.hMul (HMul.hMul a c) y) (HMul.hMul b c)
      ⊢ Membership.mem (HMul.hMul (Set.Icc a b) (Set.Icc c d)) (HMul.hMul (HMul.hMul …
    -/
    refine ⟨a * y, ⟨le_mul_of_one_le_right' hy, ?_⟩, c, left_mem_Icc.2 hcd, mul_right_comm ..⟩
    /-
      case inl.intro.intro
      α : Type u_1
      inst✝² : LinearOrderedCommMonoid α
      inst✝¹ : MulLeftReflectLE α
      inst✝ : ExistsMulOfLE α
      a b c d : α
      hab : LE.le a b
      hcd : LE.le c d
      y : α
      hy : LE.le 1 y
      x✝ : Membership.mem (Set.Icc (HMul.hMul a c) (HMul.hMul b d)) (HMul.hMul (HMul …
      hacx : LE.le (HMul.hMul a c) (HMul.hMul (HMul.hMul a c) y)
      hxbd : LE.le (HMul.hMul (HMul.hMul a c) y) (HMul.hMul b d)
      hxbc : LE.le (HMul.hMul (HMul.hMul a c) y) (HMul.hMul b c)
      ⊢ LE.le (HMul.hMul a y) b
    -/
    rwa [mul_right_comm, mul_le_mul_iff_right] at hxbc
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝² : LinearOrderedCommMonoid α
      inst✝¹ : MulLeftReflectLE α
      inst✝ : ExistsMulOfLE α
      a b c d : α
      hab : LE.le a b
      hcd : LE.le c d
      x : α
      x✝ : Membership.mem (Set.Icc (HMul.hMul a c) (HMul.hMul b d)) x
      hacx : LE.le (HMul.hMul a c) x
      hxbd : LE.le x (HMul.hMul b d)
      hbcx : LE.le (HMul.hMul b c) x
      ⊢ Membership.mem (HMul.hMul (Set.Icc a b) (Set.Icc c d)) x
    -/
  · obtain ⟨y, hy, rfl⟩ := exists_one_le_mul_of_le hbcx
    /-
      case inr.intro.intro
      α : Type u_1
      inst✝² : LinearOrderedCommMonoid α
      inst✝¹ : MulLeftReflectLE α
      inst✝ : ExistsMulOfLE α
      a b c d : α
      hab : LE.le a b
      hcd : LE.le c d
      y : α
      hy : LE.le 1 y
      x✝ : Membership.mem (Set.Icc (HMul.hMul a c) (HMul.hMul b d)) (HMul.hMul (HMul …
      hacx : LE.le (HMul.hMul a c) (HMul.hMul (HMul.hMul b c) y)
      hxbd : LE.le (HMul.hMul (HMul.hMul b c) y) (HMul.hMul b d)
      hbcx : LE.le (HMul.hMul b c) (HMul.hMul (HMul.hMul b c) y)
      ⊢ Membership.mem (HMul.hMul (Set.Icc a b) (Set.Icc c d)) (HMul.hMul (HMul.hMul …
    -/
    refine ⟨b, right_mem_Icc.2 hab, c * y, ⟨le_mul_of_one_le_right' hy, ?_⟩, (mul_assoc ..).symm⟩
    /-
      case inr.intro.intro
      α : Type u_1
      inst✝² : LinearOrderedCommMonoid α
      inst✝¹ : MulLeftReflectLE α
      inst✝ : ExistsMulOfLE α
      a b c d : α
      hab : LE.le a b
      hcd : LE.le c d
      y : α
      hy : LE.le 1 y
      x✝ : Membership.mem (Set.Icc (HMul.hMul a c) (HMul.hMul b d)) (HMul.hMul (HMul …
      hacx : LE.le (HMul.hMul a c) (HMul.hMul (HMul.hMul b c) y)
      hxbd : LE.le (HMul.hMul (HMul.hMul b c) y) (HMul.hMul b d)
      hbcx : LE.le (HMul.hMul b c) (HMul.hMul (HMul.hMul b c) y)
      ⊢ LE.le (HMul.hMul c y) d
    -/
    rwa [mul_assoc, mul_le_mul_iff_left] at hxbd
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)] lemma inv_Ici (a : α) : (Ici a)⁻¹ = Iic a⁻¹ := ext fun _x ↦ le_inv'

@[to_additive (attr := simp)] lemma inv_Iic (a : α) : (Iic a)⁻¹ = Ici a⁻¹ := ext fun _x ↦ inv_le'

@[to_additive (attr := simp)] lemma inv_Ioi (a : α) : (Ioi a)⁻¹ = Iio a⁻¹ := ext fun _x ↦ lt_inv'

@[to_additive (attr := simp)] lemma inv_Iio (a : α) : (Iio a)⁻¹ = Ioi a⁻¹ := ext fun _x ↦ inv_lt'


@[to_additive (attr := simp)]
                                                          /-
                                                            α : Type u_1
                                                            inst✝ : OrderedCommGroup α
                                                            a b : α
                                                            ⊢ Eq (Inv.inv (Set.Icc a b)) (Set.Icc (Inv.inv b) (Inv.inv a))
                                                          -/
lemma inv_Icc (a b : α) : (Icc a b)⁻¹ = Icc b⁻¹ a⁻¹ := by simp [← Ici_inter_Iic, inter_comm]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[to_additive (attr := simp)]
lemma inv_Ico (a b : α) : (Ico a b)⁻¹ = Ioc b⁻¹ a⁻¹ := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    ⊢ Eq (Inv.inv (Set.Ico a b)) (Set.Ioc (Inv.inv b) (Inv.inv a))
  -/
  simp [← Ici_inter_Iio, ← Ioi_inter_Iic, inter_comm]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma inv_Ioc (a b : α) : (Ioc a b)⁻¹ = Ico b⁻¹ a⁻¹ := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    ⊢ Eq (Inv.inv (Set.Ioc a b)) (Set.Ico (Inv.inv b) (Inv.inv a))
  -/
  simp [← Ioi_inter_Iic, ← Ici_inter_Iio, inter_comm]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
                                                          /-
                                                            α : Type u_1
                                                            inst✝ : OrderedCommGroup α
                                                            a b : α
                                                            ⊢ Eq (Inv.inv (Set.Ioo a b)) (Set.Ioo (Inv.inv b) (Inv.inv a))
                                                          -/
lemma inv_Ioo (a b : α) : (Ioo a b)⁻¹ = Ioo b⁻¹ a⁻¹ := by simp [← Ioi_inter_Iio, inter_comm]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[deprecated (since := "2024-11-23")] alias preimage_neg_Ici := neg_Ici

@[deprecated (since := "2024-11-23")] alias preimage_neg_Iic := neg_Iic

@[deprecated (since := "2024-11-23")] alias preimage_neg_Ioi := neg_Ioi

@[deprecated (since := "2024-11-23")] alias preimage_neg_Iio := neg_Iio

@[deprecated (since := "2024-11-23")] alias preimage_neg_Icc := neg_Icc

@[deprecated (since := "2024-11-23")] alias preimage_neg_Ico := neg_Ico

@[deprecated (since := "2024-11-23")] alias preimage_neg_Ioc := neg_Ioc

@[deprecated (since := "2024-11-23")] alias preimage_neg_Ioo := neg_Ioo


@[simp]
theorem preimage_const_add_Ici : (fun x => a + x) ⁻¹' Ici b = Ici (b - a) :=
  ext fun _x => sub_le_iff_le_add'.symm


@[simp]
theorem preimage_const_add_Ioi : (fun x => a + x) ⁻¹' Ioi b = Ioi (b - a) :=
  ext fun _x => sub_lt_iff_lt_add'.symm


@[simp]
theorem preimage_const_add_Iic : (fun x => a + x) ⁻¹' Iic b = Iic (b - a) :=
  ext fun _x => le_sub_iff_add_le'.symm


@[simp]
theorem preimage_const_add_Iio : (fun x => a + x) ⁻¹' Iio b = Iio (b - a) :=
  ext fun _x => lt_sub_iff_add_lt'.symm


@[simp]
theorem preimage_const_add_Icc : (fun x => a + x) ⁻¹' Icc b c = Icc (b - a) (c - a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HAdd.hAdd a x) (Set.Icc b c)) (Set.Icc (HSub.hSub …
  -/
  simp [← Ici_inter_Iic]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_const_add_Ico : (fun x => a + x) ⁻¹' Ico b c = Ico (b - a) (c - a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HAdd.hAdd a x) (Set.Ico b c)) (Set.Ico (HSub.hSub …
  -/
  simp [← Ici_inter_Iio]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_const_add_Ioc : (fun x => a + x) ⁻¹' Ioc b c = Ioc (b - a) (c - a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HAdd.hAdd a x) (Set.Ioc b c)) (Set.Ioc (HSub.hSub …
  -/
  simp [← Ioi_inter_Iic]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_const_add_Ioo : (fun x => a + x) ⁻¹' Ioo b c = Ioo (b - a) (c - a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HAdd.hAdd a x) (Set.Ioo b c)) (Set.Ioo (HSub.hSub …
  -/
  simp [← Ioi_inter_Iio]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_add_const_Ici : (fun x => x + a) ⁻¹' Ici b = Ici (b - a) :=
  ext fun _x => sub_le_iff_le_add.symm


@[simp]
theorem preimage_add_const_Ioi : (fun x => x + a) ⁻¹' Ioi b = Ioi (b - a) :=
  ext fun _x => sub_lt_iff_lt_add.symm


@[simp]
theorem preimage_add_const_Iic : (fun x => x + a) ⁻¹' Iic b = Iic (b - a) :=
  ext fun _x => le_sub_iff_add_le.symm


@[simp]
theorem preimage_add_const_Iio : (fun x => x + a) ⁻¹' Iio b = Iio (b - a) :=
  ext fun _x => lt_sub_iff_add_lt.symm


@[simp]
theorem preimage_add_const_Icc : (fun x => x + a) ⁻¹' Icc b c = Icc (b - a) (c - a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HAdd.hAdd x a) (Set.Icc b c)) (Set.Icc (HSub.hSub …
  -/
  simp [← Ici_inter_Iic]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_add_const_Ico : (fun x => x + a) ⁻¹' Ico b c = Ico (b - a) (c - a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HAdd.hAdd x a) (Set.Ico b c)) (Set.Ico (HSub.hSub …
  -/
  simp [← Ici_inter_Iio]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_add_const_Ioc : (fun x => x + a) ⁻¹' Ioc b c = Ioc (b - a) (c - a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HAdd.hAdd x a) (Set.Ioc b c)) (Set.Ioc (HSub.hSub …
  -/
  simp [← Ioi_inter_Iic]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_add_const_Ioo : (fun x => x + a) ⁻¹' Ioo b c = Ioo (b - a) (c - a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HAdd.hAdd x a) (Set.Ioo b c)) (Set.Ioo (HSub.hSub …
  -/
  simp [← Ioi_inter_Iio]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_sub_const_Ici : (fun x => x - a) ⁻¹' Ici b = Ici (b + a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b : α
    ⊢ Eq (Set.preimage (fun x => HSub.hSub x a) (Set.Ici b)) (Set.Ici (HAdd.hAdd b …
  -/
  simp [sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_sub_const_Ioi : (fun x => x - a) ⁻¹' Ioi b = Ioi (b + a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b : α
    ⊢ Eq (Set.preimage (fun x => HSub.hSub x a) (Set.Ioi b)) (Set.Ioi (HAdd.hAdd b …
  -/
  simp [sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_sub_const_Iic : (fun x => x - a) ⁻¹' Iic b = Iic (b + a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b : α
    ⊢ Eq (Set.preimage (fun x => HSub.hSub x a) (Set.Iic b)) (Set.Iic (HAdd.hAdd b …
  -/
  simp [sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_sub_const_Iio : (fun x => x - a) ⁻¹' Iio b = Iio (b + a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b : α
    ⊢ Eq (Set.preimage (fun x => HSub.hSub x a) (Set.Iio b)) (Set.Iio (HAdd.hAdd b …
  -/
  simp [sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_sub_const_Icc : (fun x => x - a) ⁻¹' Icc b c = Icc (b + a) (c + a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HSub.hSub x a) (Set.Icc b c)) (Set.Icc (HAdd.hAdd …
  -/
  simp [sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_sub_const_Ico : (fun x => x - a) ⁻¹' Ico b c = Ico (b + a) (c + a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HSub.hSub x a) (Set.Ico b c)) (Set.Ico (HAdd.hAdd …
  -/
  simp [sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_sub_const_Ioc : (fun x => x - a) ⁻¹' Ioc b c = Ioc (b + a) (c + a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HSub.hSub x a) (Set.Ioc b c)) (Set.Ioc (HAdd.hAdd …
  -/
  simp [sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_sub_const_Ioo : (fun x => x - a) ⁻¹' Ioo b c = Ioo (b + a) (c + a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HSub.hSub x a) (Set.Ioo b c)) (Set.Ioo (HAdd.hAdd …
  -/
  simp [sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_const_sub_Ici : (fun x => a - x) ⁻¹' Ici b = Iic (a - b) :=
  ext fun _x => le_sub_comm


@[simp]
theorem preimage_const_sub_Iic : (fun x => a - x) ⁻¹' Iic b = Ici (a - b) :=
  ext fun _x => sub_le_comm


@[simp]
theorem preimage_const_sub_Ioi : (fun x => a - x) ⁻¹' Ioi b = Iio (a - b) :=
  ext fun _x => lt_sub_comm


@[simp]
theorem preimage_const_sub_Iio : (fun x => a - x) ⁻¹' Iio b = Ioi (a - b) :=
  ext fun _x => sub_lt_comm


@[simp]
theorem preimage_const_sub_Icc : (fun x => a - x) ⁻¹' Icc b c = Icc (a - c) (a - b) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HSub.hSub a x) (Set.Icc b c)) (Set.Icc (HSub.hSub …
  -/
  simp [← Ici_inter_Iic, inter_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_const_sub_Ico : (fun x => a - x) ⁻¹' Ico b c = Ioc (a - c) (a - b) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HSub.hSub a x) (Set.Ico b c)) (Set.Ioc (HSub.hSub …
  -/
  simp [← Ioi_inter_Iic, ← Ici_inter_Iio, inter_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_const_sub_Ioc : (fun x => a - x) ⁻¹' Ioc b c = Ico (a - c) (a - b) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HSub.hSub a x) (Set.Ioc b c)) (Set.Ico (HSub.hSub …
  -/
  simp [← Ioi_inter_Iic, ← Ici_inter_Iio, inter_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_const_sub_Ioo : (fun x => a - x) ⁻¹' Ioo b c = Ioo (a - c) (a - b) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HSub.hSub a x) (Set.Ioo b c)) (Set.Ioo (HSub.hSub …
  -/
  simp [← Ioi_inter_Iio, inter_comm]
  /-
    🎉 no goals
  -/


                                                                            /-
                                                                              α : Type u_1
                                                                              inst✝ : OrderedAddCommGroup α
                                                                              a b : α
                                                                              ⊢ Eq (Set.image (fun x => HAdd.hAdd a x) (Set.Iic b)) (Set.Iic (HAdd.hAdd a b))
                                                                            -/
theorem image_const_add_Iic : (fun x => a + x) '' Iic b = Iic (a + b) := by simp [add_comm]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/

-- simp can prove this modulo `add_comm`

                                                                            /-
                                                                              α : Type u_1
                                                                              inst✝ : OrderedAddCommGroup α
                                                                              a b : α
                                                                              ⊢ Eq (Set.image (fun x => HAdd.hAdd a x) (Set.Iio b)) (Set.Iio (HAdd.hAdd a b))
                                                                            -/
theorem image_const_add_Iio : (fun x => a + x) '' Iio b = Iio (a + b) := by simp [add_comm]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


                                                                            /-
                                                                              α : Type u_1
                                                                              inst✝ : OrderedAddCommGroup α
                                                                              a b : α
                                                                              ⊢ Eq (Set.image (fun x => HAdd.hAdd x a) (Set.Iic b)) (Set.Iic (HAdd.hAdd b a))
                                                                            -/
theorem image_add_const_Iic : (fun x => x + a) '' Iic b = Iic (b + a) := by simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


                                                                            /-
                                                                              α : Type u_1
                                                                              inst✝ : OrderedAddCommGroup α
                                                                              a b : α
                                                                              ⊢ Eq (Set.image (fun x => HAdd.hAdd x a) (Set.Iio b)) (Set.Iio (HAdd.hAdd b a))
                                                                            -/
theorem image_add_const_Iio : (fun x => x + a) '' Iio b = Iio (b + a) := by simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


                                                          /-
                                                            α : Type u_1
                                                            inst✝ : OrderedAddCommGroup α
                                                            a : α
                                                            ⊢ Eq (Set.image Neg.neg (Set.Ici a)) (Set.Iic (Neg.neg a))
                                                          -/
theorem image_neg_Ici : Neg.neg '' Ici a = Iic (-a) := by simp
                                                          /-
                                                            🎉 no goals
                                                          -/


                                                          /-
                                                            α : Type u_1
                                                            inst✝ : OrderedAddCommGroup α
                                                            a : α
                                                            ⊢ Eq (Set.image Neg.neg (Set.Iic a)) (Set.Ici (Neg.neg a))
                                                          -/
theorem image_neg_Iic : Neg.neg '' Iic a = Ici (-a) := by simp
                                                          /-
                                                            🎉 no goals
                                                          -/


                                                          /-
                                                            α : Type u_1
                                                            inst✝ : OrderedAddCommGroup α
                                                            a : α
                                                            ⊢ Eq (Set.image Neg.neg (Set.Ioi a)) (Set.Iio (Neg.neg a))
                                                          -/
theorem image_neg_Ioi : Neg.neg '' Ioi a = Iio (-a) := by simp
                                                          /-
                                                            🎉 no goals
                                                          -/


                                                          /-
                                                            α : Type u_1
                                                            inst✝ : OrderedAddCommGroup α
                                                            a : α
                                                            ⊢ Eq (Set.image Neg.neg (Set.Iio a)) (Set.Ioi (Neg.neg a))
                                                          -/
theorem image_neg_Iio : Neg.neg '' Iio a = Ioi (-a) := by simp
                                                          /-
                                                            🎉 no goals
                                                          -/


                                                                 /-
                                                                   α : Type u_1
                                                                   inst✝ : OrderedAddCommGroup α
                                                                   a b : α
                                                                   ⊢ Eq (Set.image Neg.neg (Set.Icc a b)) (Set.Icc (Neg.neg b) (Neg.neg a))
                                                                 -/
theorem image_neg_Icc : Neg.neg '' Icc a b = Icc (-b) (-a) := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


                                                                 /-
                                                                   α : Type u_1
                                                                   inst✝ : OrderedAddCommGroup α
                                                                   a b : α
                                                                   ⊢ Eq (Set.image Neg.neg (Set.Ico a b)) (Set.Ioc (Neg.neg b) (Neg.neg a))
                                                                 -/
theorem image_neg_Ico : Neg.neg '' Ico a b = Ioc (-b) (-a) := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


                                                                 /-
                                                                   α : Type u_1
                                                                   inst✝ : OrderedAddCommGroup α
                                                                   a b : α
                                                                   ⊢ Eq (Set.image Neg.neg (Set.Ioc a b)) (Set.Ico (Neg.neg b) (Neg.neg a))
                                                                 -/
theorem image_neg_Ioc : Neg.neg '' Ioc a b = Ico (-b) (-a) := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


                                                                 /-
                                                                   α : Type u_1
                                                                   inst✝ : OrderedAddCommGroup α
                                                                   a b : α
                                                                   ⊢ Eq (Set.image Neg.neg (Set.Ioo a b)) (Set.Ioo (Neg.neg b) (Neg.neg a))
                                                                 -/
theorem image_neg_Ioo : Neg.neg '' Ioo a b = Ioo (-b) (-a) := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem image_const_sub_Ici : (fun x => a - x) '' Ici b = Iic (a - b) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b : α
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.Ici b)) (Set.Iic (HSub.hSub a b))
  -/
  have := image_comp (fun x => a + x) fun x => -x; dsimp [Function.comp_def] at this
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b : α
    this : ∀ (a_1 : Set α), Eq (Set.image (fun x => HAdd.hAdd a (Neg.neg x)) a_1)  …
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.Ici b)) (Set.Iic (HSub.hSub a b))
  -/
  simp [sub_eq_add_neg, this, add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_const_sub_Iic : (fun x => a - x) '' Iic b = Ici (a - b) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b : α
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.Iic b)) (Set.Ici (HSub.hSub a b))
  -/
  have := image_comp (fun x => a + x) fun x => -x; dsimp [Function.comp_def] at this
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b : α
    this : ∀ (a_1 : Set α), Eq (Set.image (fun x => HAdd.hAdd a (Neg.neg x)) a_1)  …
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.Iic b)) (Set.Ici (HSub.hSub a b))
  -/
  simp [sub_eq_add_neg, this, add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_const_sub_Ioi : (fun x => a - x) '' Ioi b = Iio (a - b) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b : α
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.Ioi b)) (Set.Iio (HSub.hSub a b))
  -/
  have := image_comp (fun x => a + x) fun x => -x; dsimp [Function.comp_def] at this
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b : α
    this : ∀ (a_1 : Set α), Eq (Set.image (fun x => HAdd.hAdd a (Neg.neg x)) a_1)  …
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.Ioi b)) (Set.Iio (HSub.hSub a b))
  -/
  simp [sub_eq_add_neg, this, add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_const_sub_Iio : (fun x => a - x) '' Iio b = Ioi (a - b) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b : α
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.Iio b)) (Set.Ioi (HSub.hSub a b))
  -/
  have := image_comp (fun x => a + x) fun x => -x; dsimp [Function.comp_def] at this
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b : α
    this : ∀ (a_1 : Set α), Eq (Set.image (fun x => HAdd.hAdd a (Neg.neg x)) a_1)  …
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.Iio b)) (Set.Ioi (HSub.hSub a b))
  -/
  simp [sub_eq_add_neg, this, add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_const_sub_Icc : (fun x => a - x) '' Icc b c = Icc (a - c) (a - b) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.Icc b c)) (Set.Icc (HSub.hSub a  …
  -/
  have := image_comp (fun x => a + x) fun x => -x; dsimp [Function.comp_def] at this
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    this : ∀ (a_1 : Set α), Eq (Set.image (fun x => HAdd.hAdd a (Neg.neg x)) a_1)  …
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.Icc b c)) (Set.Icc (HSub.hSub a  …
  -/
  simp [sub_eq_add_neg, this, add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_const_sub_Ico : (fun x => a - x) '' Ico b c = Ioc (a - c) (a - b) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.Ico b c)) (Set.Ioc (HSub.hSub a  …
  -/
  have := image_comp (fun x => a + x) fun x => -x; dsimp [Function.comp_def] at this
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    this : ∀ (a_1 : Set α), Eq (Set.image (fun x => HAdd.hAdd a (Neg.neg x)) a_1)  …
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.Ico b c)) (Set.Ioc (HSub.hSub a  …
  -/
  simp [sub_eq_add_neg, this, add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_const_sub_Ioc : (fun x => a - x) '' Ioc b c = Ico (a - c) (a - b) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.Ioc b c)) (Set.Ico (HSub.hSub a  …
  -/
  have := image_comp (fun x => a + x) fun x => -x; dsimp [Function.comp_def] at this
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    this : ∀ (a_1 : Set α), Eq (Set.image (fun x => HAdd.hAdd a (Neg.neg x)) a_1)  …
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.Ioc b c)) (Set.Ico (HSub.hSub a  …
  -/
  simp [sub_eq_add_neg, this, add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_const_sub_Ioo : (fun x => a - x) '' Ioo b c = Ioo (a - c) (a - b) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.Ioo b c)) (Set.Ioo (HSub.hSub a  …
  -/
  have := image_comp (fun x => a + x) fun x => -x; dsimp [Function.comp_def] at this
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    this : ∀ (a_1 : Set α), Eq (Set.image (fun x => HAdd.hAdd a (Neg.neg x)) a_1)  …
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.Ioo b c)) (Set.Ioo (HSub.hSub a  …
  -/
  simp [sub_eq_add_neg, this, add_comm]
  /-
    🎉 no goals
  -/


@[simp]
                                                                            /-
                                                                              α : Type u_1
                                                                              inst✝ : OrderedAddCommGroup α
                                                                              a b : α
                                                                              ⊢ Eq (Set.image (fun x => HSub.hSub x a) (Set.Ici b)) (Set.Ici (HSub.hSub b a))
                                                                            -/
theorem image_sub_const_Ici : (fun x => x - a) '' Ici b = Ici (b - a) := by simp [sub_eq_neg_add]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp]
                                                                            /-
                                                                              α : Type u_1
                                                                              inst✝ : OrderedAddCommGroup α
                                                                              a b : α
                                                                              ⊢ Eq (Set.image (fun x => HSub.hSub x a) (Set.Iic b)) (Set.Iic (HSub.hSub b a))
                                                                            -/
theorem image_sub_const_Iic : (fun x => x - a) '' Iic b = Iic (b - a) := by simp [sub_eq_neg_add]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp]
                                                                            /-
                                                                              α : Type u_1
                                                                              inst✝ : OrderedAddCommGroup α
                                                                              a b : α
                                                                              ⊢ Eq (Set.image (fun x => HSub.hSub x a) (Set.Ioi b)) (Set.Ioi (HSub.hSub b a))
                                                                            -/
theorem image_sub_const_Ioi : (fun x => x - a) '' Ioi b = Ioi (b - a) := by simp [sub_eq_neg_add]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp]
                                                                            /-
                                                                              α : Type u_1
                                                                              inst✝ : OrderedAddCommGroup α
                                                                              a b : α
                                                                              ⊢ Eq (Set.image (fun x => HSub.hSub x a) (Set.Iio b)) (Set.Iio (HSub.hSub b a))
                                                                            -/
theorem image_sub_const_Iio : (fun x => x - a) '' Iio b = Iio (b - a) := by simp [sub_eq_neg_add]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp]
theorem image_sub_const_Icc : (fun x => x - a) '' Icc b c = Icc (b - a) (c - a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.image (fun x => HSub.hSub x a) (Set.Icc b c)) (Set.Icc (HSub.hSub b  …
  -/
  simp [sub_eq_neg_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_sub_const_Ico : (fun x => x - a) '' Ico b c = Ico (b - a) (c - a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.image (fun x => HSub.hSub x a) (Set.Ico b c)) (Set.Ico (HSub.hSub b  …
  -/
  simp [sub_eq_neg_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_sub_const_Ioc : (fun x => x - a) '' Ioc b c = Ioc (b - a) (c - a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.image (fun x => HSub.hSub x a) (Set.Ioc b c)) (Set.Ioc (HSub.hSub b  …
  -/
  simp [sub_eq_neg_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_sub_const_Ioo : (fun x => x - a) '' Ioo b c = Ioo (b - a) (c - a) := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.image (fun x => HSub.hSub x a) (Set.Ioo b c)) (Set.Ioo (HSub.hSub b  …
  -/
  simp [sub_eq_neg_add]
  /-
    🎉 no goals
  -/


theorem Iic_add_bij : BijOn (· + a) (Iic b) (Iic (b + a)) :=
  image_add_const_Iic a b ▸ (add_left_injective _).injOn.bijOn_image


theorem Iio_add_bij : BijOn (· + a) (Iio b) (Iio (b + a)) :=
  image_add_const_Iio a b ▸ (add_left_injective _).injOn.bijOn_image


@[to_additive (attr := simp)]
lemma inv_uIcc (a b : α) : [[a, b]]⁻¹ = [[a⁻¹, b⁻¹]] := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroup α
    a b : α
    ⊢ Eq (Inv.inv (Set.uIcc a b)) (Set.uIcc (Inv.inv a) (Inv.inv b))
  -/
  simp only [uIcc, inv_Icc, inv_sup, inv_inf]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_const_add_uIcc : (fun x => a + x) ⁻¹' [[b, c]] = [[b - a, c - a]] := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HAdd.hAdd a x) (Set.uIcc b c)) (Set.uIcc (HSub.hS …
  -/
  simp only [← Icc_min_max, preimage_const_add_Icc, min_sub_sub_right, max_sub_sub_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_add_const_uIcc : (fun x => x + a) ⁻¹' [[b, c]] = [[b - a, c - a]] := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HAdd.hAdd x a) (Set.uIcc b c)) (Set.uIcc (HSub.hS …
  -/
  simpa only [add_comm] using preimage_const_add_uIcc a b c
  /-
    🎉 no goals
  -/


@[deprecated neg_uIcc (since := "2024-11-23")]
theorem preimage_neg_uIcc : -[[a, b]] = [[-a, -b]] := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b : α
    ⊢ Eq (Neg.neg (Set.uIcc a b)) (Set.uIcc (Neg.neg a) (Neg.neg b))
  -/
  simp only [← Icc_min_max, neg_Icc, min_neg_neg, max_neg_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_sub_const_uIcc : (fun x => x - a) ⁻¹' [[b, c]] = [[b + a, c + a]] := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HSub.hSub x a) (Set.uIcc b c)) (Set.uIcc (HAdd.hA …
  -/
  simp [sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_const_sub_uIcc : (fun x => a - x) ⁻¹' [[b, c]] = [[a - b, a - c]] := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.preimage (fun x => HSub.hSub a x) (Set.uIcc b c)) (Set.uIcc (HSub.hS …
  -/
  simp_rw [← Icc_min_max, preimage_const_sub_Icc]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.Icc (HSub.hSub a (Max.max b c)) (HSub.hSub a (Min.min b c))) (Set.Ic …
  -/
  simp only [sub_eq_add_neg, min_add_add_left, max_add_add_left, min_neg_neg, max_neg_neg]
  /-
    🎉 no goals
  -/

-- simp can prove this modulo `add_comm`

                                                                                     /-
                                                                                       α : Type u_1
                                                                                       inst✝ : LinearOrderedAddCommGroup α
                                                                                       a b c : α
                                                                                       ⊢ Eq (Set.image (fun x => HAdd.hAdd a x) (Set.uIcc b c)) (Set.uIcc (HAdd.hAdd  …
                                                                                     -/
theorem image_const_add_uIcc : (fun x => a + x) '' [[b, c]] = [[a + b, a + c]] := by simp [add_comm]
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


                                                                                     /-
                                                                                       α : Type u_1
                                                                                       inst✝ : LinearOrderedAddCommGroup α
                                                                                       a b c : α
                                                                                       ⊢ Eq (Set.image (fun x => HAdd.hAdd x a) (Set.uIcc b c)) (Set.uIcc (HAdd.hAdd  …
                                                                                     -/
theorem image_add_const_uIcc : (fun x => x + a) '' [[b, c]] = [[b + a, c + a]] := by simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp]
theorem image_const_sub_uIcc : (fun x => a - x) '' [[b, c]] = [[a - b, a - c]] := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.uIcc b c)) (Set.uIcc (HSub.hSub  …
  -/
  have := image_comp (fun x => a + x) fun x => -x; dsimp [Function.comp_def] at this
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c : α
    this : ∀ (a_1 : Set α), Eq (Set.image (fun x => HAdd.hAdd a (Neg.neg x)) a_1)  …
    ⊢ Eq (Set.image (fun x => HSub.hSub a x) (Set.uIcc b c)) (Set.uIcc (HSub.hSub  …
  -/
  simp [sub_eq_add_neg, this, add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_sub_const_uIcc : (fun x => x - a) '' [[b, c]] = [[b - a, c - a]] := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c : α
    ⊢ Eq (Set.image (fun x => HSub.hSub x a) (Set.uIcc b c)) (Set.uIcc (HSub.hSub  …
  -/
  simp [sub_eq_add_neg, add_comm]
  /-
    🎉 no goals
  -/


                                                                /-
                                                                  α : Type u_1
                                                                  inst✝ : LinearOrderedAddCommGroup α
                                                                  a b : α
                                                                  ⊢ Eq (Set.image Neg.neg (Set.uIcc a b)) (Set.uIcc (Neg.neg a) (Neg.neg b))
                                                                -/
theorem image_neg_uIcc : Neg.neg '' [[a, b]] = [[-a, -b]] := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- If `[c, d]` is a subinterval of `[a, b]`, then the distance between `c` and `d` is less than or
equal to that of `a` and `b` -/
theorem abs_sub_le_of_uIcc_subset_uIcc (h : [[c, d]] ⊆ [[a, b]]) : |d - c| ≤ |b - a| := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c d : α
    h : HasSubset.Subset (Set.uIcc c d) (Set.uIcc a b)
    ⊢ LE.le (abs (HSub.hSub d c)) (abs (HSub.hSub b a))
  -/
  rw [← max_sub_min_eq_abs, ← max_sub_min_eq_abs]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c d : α
    h : HasSubset.Subset (Set.uIcc c d) (Set.uIcc a b)
    ⊢ LE.le (HSub.hSub (Max.max c d) (Min.min c d)) (HSub.hSub (Max.max a b) (Min. …
  -/
  rw [uIcc_subset_uIcc_iff_le] at h
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c d : α
    h : And (LE.le (Min.min a b) (Min.min c d)) (LE.le (Max.max c d) (Max.max a b))
    ⊢ LE.le (HSub.hSub (Max.max c d) (Min.min c d)) (HSub.hSub (Max.max a b) (Min. …
  -/
  exact sub_le_sub h.2 h.1
  /-
    🎉 no goals
  -/


/-- If `c ∈ [a, b]`, then the distance between `a` and `c` is less than or equal to
that of `a` and `b`  -/
theorem abs_sub_left_of_mem_uIcc (h : c ∈ [[a, b]]) : |c - a| ≤ |b - a| :=
  abs_sub_le_of_uIcc_subset_uIcc <| uIcc_subset_uIcc_left h


/-- If `x ∈ [a, b]`, then the distance between `c` and `b` is less than or equal to
that of `a` and `b`  -/
theorem abs_sub_right_of_mem_uIcc (h : c ∈ [[a, b]]) : |b - c| ≤ |b - a| :=
  abs_sub_le_of_uIcc_subset_uIcc <| uIcc_subset_uIcc_right h


@[simp]
theorem preimage_mul_const_Iio (a : α) {c : α} (h : 0 < c) :
    (fun x => x * c) ⁻¹' Iio a = Iio (a / c) :=
  ext fun _x => (lt_div_iff₀ h).symm


@[simp]
theorem preimage_mul_const_Ioi (a : α) {c : α} (h : 0 < c) :
    (fun x => x * c) ⁻¹' Ioi a = Ioi (a / c) :=
  ext fun _x => (div_lt_iff₀ h).symm


@[simp]
theorem preimage_mul_const_Iic (a : α) {c : α} (h : 0 < c) :
    (fun x => x * c) ⁻¹' Iic a = Iic (a / c) :=
  ext fun _x => (le_div_iff₀ h).symm


@[simp]
theorem preimage_mul_const_Ici (a : α) {c : α} (h : 0 < c) :
    (fun x => x * c) ⁻¹' Ici a = Ici (a / c) :=
  ext fun _x => (div_le_iff₀ h).symm


@[simp]
theorem preimage_mul_const_Ioo (a b : α) {c : α} (h : 0 < c) :
                                                             /-
                                                               α : Type u_1
                                                               inst✝ : LinearOrderedField α
                                                               a b c : α
                                                               h : LT.lt 0 c
                                                               ⊢ Eq (Set.preimage (fun x => HMul.hMul x c) (Set.Ioo a b)) (Set.Ioo (HDiv.hDiv …
                                                             -/
    (fun x => x * c) ⁻¹' Ioo a b = Ioo (a / c) (b / c) := by simp [← Ioi_inter_Iio, h]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem preimage_mul_const_Ioc (a b : α) {c : α} (h : 0 < c) :
                                                             /-
                                                               α : Type u_1
                                                               inst✝ : LinearOrderedField α
                                                               a b c : α
                                                               h : LT.lt 0 c
                                                               ⊢ Eq (Set.preimage (fun x => HMul.hMul x c) (Set.Ioc a b)) (Set.Ioc (HDiv.hDiv …
                                                             -/
    (fun x => x * c) ⁻¹' Ioc a b = Ioc (a / c) (b / c) := by simp [← Ioi_inter_Iic, h]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem preimage_mul_const_Ico (a b : α) {c : α} (h : 0 < c) :
                                                             /-
                                                               α : Type u_1
                                                               inst✝ : LinearOrderedField α
                                                               a b c : α
                                                               h : LT.lt 0 c
                                                               ⊢ Eq (Set.preimage (fun x => HMul.hMul x c) (Set.Ico a b)) (Set.Ico (HDiv.hDiv …
                                                             -/
    (fun x => x * c) ⁻¹' Ico a b = Ico (a / c) (b / c) := by simp [← Ici_inter_Iio, h]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem preimage_mul_const_Icc (a b : α) {c : α} (h : 0 < c) :
                                                             /-
                                                               α : Type u_1
                                                               inst✝ : LinearOrderedField α
                                                               a b c : α
                                                               h : LT.lt 0 c
                                                               ⊢ Eq (Set.preimage (fun x => HMul.hMul x c) (Set.Icc a b)) (Set.Icc (HDiv.hDiv …
                                                             -/
    (fun x => x * c) ⁻¹' Icc a b = Icc (a / c) (b / c) := by simp [← Ici_inter_Iic, h]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem preimage_mul_const_Iio_of_neg (a : α) {c : α} (h : c < 0) :
    (fun x => x * c) ⁻¹' Iio a = Ioi (a / c) :=
  ext fun _x => (div_lt_iff_of_neg h).symm


@[simp]
theorem preimage_mul_const_Ioi_of_neg (a : α) {c : α} (h : c < 0) :
    (fun x => x * c) ⁻¹' Ioi a = Iio (a / c) :=
  ext fun _x => (lt_div_iff_of_neg h).symm


@[simp]
theorem preimage_mul_const_Iic_of_neg (a : α) {c : α} (h : c < 0) :
    (fun x => x * c) ⁻¹' Iic a = Ici (a / c) :=
  ext fun _x => (div_le_iff_of_neg h).symm


@[simp]
theorem preimage_mul_const_Ici_of_neg (a : α) {c : α} (h : c < 0) :
    (fun x => x * c) ⁻¹' Ici a = Iic (a / c) :=
  ext fun _x => (le_div_iff_of_neg h).symm


@[simp]
theorem preimage_mul_const_Ioo_of_neg (a b : α) {c : α} (h : c < 0) :
                                                             /-
                                                               α : Type u_1
                                                               inst✝ : LinearOrderedField α
                                                               a b c : α
                                                               h : LT.lt c 0
                                                               ⊢ Eq (Set.preimage (fun x => HMul.hMul x c) (Set.Ioo a b)) (Set.Ioo (HDiv.hDiv …
                                                             -/
    (fun x => x * c) ⁻¹' Ioo a b = Ioo (b / c) (a / c) := by simp [← Ioi_inter_Iio, h, inter_comm]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem preimage_mul_const_Ioc_of_neg (a b : α) {c : α} (h : c < 0) :
    (fun x => x * c) ⁻¹' Ioc a b = Ico (b / c) (a / c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : α
    h : LT.lt c 0
    ⊢ Eq (Set.preimage (fun x => HMul.hMul x c) (Set.Ioc a b)) (Set.Ico (HDiv.hDiv …
  -/
  simp [← Ioi_inter_Iic, ← Ici_inter_Iio, h, inter_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_mul_const_Ico_of_neg (a b : α) {c : α} (h : c < 0) :
    (fun x => x * c) ⁻¹' Ico a b = Ioc (b / c) (a / c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : α
    h : LT.lt c 0
    ⊢ Eq (Set.preimage (fun x => HMul.hMul x c) (Set.Ico a b)) (Set.Ioc (HDiv.hDiv …
  -/
  simp [← Ici_inter_Iio, ← Ioi_inter_Iic, h, inter_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_mul_const_Icc_of_neg (a b : α) {c : α} (h : c < 0) :
                                                             /-
                                                               α : Type u_1
                                                               inst✝ : LinearOrderedField α
                                                               a b c : α
                                                               h : LT.lt c 0
                                                               ⊢ Eq (Set.preimage (fun x => HMul.hMul x c) (Set.Icc a b)) (Set.Icc (HDiv.hDiv …
                                                             -/
    (fun x => x * c) ⁻¹' Icc a b = Icc (b / c) (a / c) := by simp [← Ici_inter_Iic, h, inter_comm]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem preimage_const_mul_Iio (a : α) {c : α} (h : 0 < c) : (c * ·) ⁻¹' Iio a = Iio (a / c) :=
  ext fun _x => (lt_div_iff₀' h).symm


@[simp]
theorem preimage_const_mul_Ioi (a : α) {c : α} (h : 0 < c) : (c * ·) ⁻¹' Ioi a = Ioi (a / c) :=
  ext fun _x => (div_lt_iff₀' h).symm


@[simp]
theorem preimage_const_mul_Iic (a : α) {c : α} (h : 0 < c) : (c * ·) ⁻¹' Iic a = Iic (a / c) :=
  ext fun _x => (le_div_iff₀' h).symm


@[simp]
theorem preimage_const_mul_Ici (a : α) {c : α} (h : 0 < c) : (c * ·) ⁻¹' Ici a = Ici (a / c) :=
  ext fun _x => (div_le_iff₀' h).symm


@[simp]
theorem preimage_const_mul_Ioo (a b : α) {c : α} (h : 0 < c) :
                                                    /-
                                                      α : Type u_1
                                                      inst✝ : LinearOrderedField α
                                                      a b c : α
                                                      h : LT.lt 0 c
                                                      ⊢ Eq (Set.preimage (fun x => HMul.hMul c x) (Set.Ioo a b)) (Set.Ioo (HDiv.hDiv …
                                                    -/
    (c * ·) ⁻¹' Ioo a b = Ioo (a / c) (b / c) := by simp [← Ioi_inter_Iio, h]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem preimage_const_mul_Ioc (a b : α) {c : α} (h : 0 < c) :
                                                    /-
                                                      α : Type u_1
                                                      inst✝ : LinearOrderedField α
                                                      a b c : α
                                                      h : LT.lt 0 c
                                                      ⊢ Eq (Set.preimage (fun x => HMul.hMul c x) (Set.Ioc a b)) (Set.Ioc (HDiv.hDiv …
                                                    -/
    (c * ·) ⁻¹' Ioc a b = Ioc (a / c) (b / c) := by simp [← Ioi_inter_Iic, h]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem preimage_const_mul_Ico (a b : α) {c : α} (h : 0 < c) :
                                                    /-
                                                      α : Type u_1
                                                      inst✝ : LinearOrderedField α
                                                      a b c : α
                                                      h : LT.lt 0 c
                                                      ⊢ Eq (Set.preimage (fun x => HMul.hMul c x) (Set.Ico a b)) (Set.Ico (HDiv.hDiv …
                                                    -/
    (c * ·) ⁻¹' Ico a b = Ico (a / c) (b / c) := by simp [← Ici_inter_Iio, h]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem preimage_const_mul_Icc (a b : α) {c : α} (h : 0 < c) :
                                                    /-
                                                      α : Type u_1
                                                      inst✝ : LinearOrderedField α
                                                      a b c : α
                                                      h : LT.lt 0 c
                                                      ⊢ Eq (Set.preimage (fun x => HMul.hMul c x) (Set.Icc a b)) (Set.Icc (HDiv.hDiv …
                                                    -/
    (c * ·) ⁻¹' Icc a b = Icc (a / c) (b / c) := by simp [← Ici_inter_Iic, h]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem preimage_const_mul_Iio_of_neg (a : α) {c : α} (h : c < 0) :
    (c * ·) ⁻¹' Iio a = Ioi (a / c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a c : α
    h : LT.lt c 0
    ⊢ Eq (Set.preimage (fun x => HMul.hMul c x) (Set.Iio a)) (Set.Ioi (HDiv.hDiv a …
  -/
  simpa only [mul_comm] using preimage_mul_const_Iio_of_neg a h
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_const_mul_Ioi_of_neg (a : α) {c : α} (h : c < 0) :
    (c * ·) ⁻¹' Ioi a = Iio (a / c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a c : α
    h : LT.lt c 0
    ⊢ Eq (Set.preimage (fun x => HMul.hMul c x) (Set.Ioi a)) (Set.Iio (HDiv.hDiv a …
  -/
  simpa only [mul_comm] using preimage_mul_const_Ioi_of_neg a h
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_const_mul_Iic_of_neg (a : α) {c : α} (h : c < 0) :
    (c * ·) ⁻¹' Iic a = Ici (a / c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a c : α
    h : LT.lt c 0
    ⊢ Eq (Set.preimage (fun x => HMul.hMul c x) (Set.Iic a)) (Set.Ici (HDiv.hDiv a …
  -/
  simpa only [mul_comm] using preimage_mul_const_Iic_of_neg a h
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_const_mul_Ici_of_neg (a : α) {c : α} (h : c < 0) :
    (c * ·) ⁻¹' Ici a = Iic (a / c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a c : α
    h : LT.lt c 0
    ⊢ Eq (Set.preimage (fun x => HMul.hMul c x) (Set.Ici a)) (Set.Iic (HDiv.hDiv a …
  -/
  simpa only [mul_comm] using preimage_mul_const_Ici_of_neg a h
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_const_mul_Ioo_of_neg (a b : α) {c : α} (h : c < 0) :
    (c * ·) ⁻¹' Ioo a b = Ioo (b / c) (a / c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : α
    h : LT.lt c 0
    ⊢ Eq (Set.preimage (fun x => HMul.hMul c x) (Set.Ioo a b)) (Set.Ioo (HDiv.hDiv …
  -/
  simpa only [mul_comm] using preimage_mul_const_Ioo_of_neg a b h
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_const_mul_Ioc_of_neg (a b : α) {c : α} (h : c < 0) :
    (c * ·) ⁻¹' Ioc a b = Ico (b / c) (a / c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : α
    h : LT.lt c 0
    ⊢ Eq (Set.preimage (fun x => HMul.hMul c x) (Set.Ioc a b)) (Set.Ico (HDiv.hDiv …
  -/
  simpa only [mul_comm] using preimage_mul_const_Ioc_of_neg a b h
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_const_mul_Ico_of_neg (a b : α) {c : α} (h : c < 0) :
    (c * ·) ⁻¹' Ico a b = Ioc (b / c) (a / c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : α
    h : LT.lt c 0
    ⊢ Eq (Set.preimage (fun x => HMul.hMul c x) (Set.Ico a b)) (Set.Ioc (HDiv.hDiv …
  -/
  simpa only [mul_comm] using preimage_mul_const_Ico_of_neg a b h
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_const_mul_Icc_of_neg (a b : α) {c : α} (h : c < 0) :
    (c * ·) ⁻¹' Icc a b = Icc (b / c) (a / c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : α
    h : LT.lt c 0
    ⊢ Eq (Set.preimage (fun x => HMul.hMul c x) (Set.Icc a b)) (Set.Icc (HDiv.hDiv …
  -/
  simpa only [mul_comm] using preimage_mul_const_Icc_of_neg a b h
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_mul_const_uIcc (ha : a ≠ 0) (b c : α) :
    (· * a) ⁻¹' [[b, c]] = [[b / a, c / a]] :=
  (lt_or_gt_of_ne ha).elim
    (fun h => by
      /-
        α : Type u_1
        inst✝ : LinearOrderedField α
        a : α
        ha : Ne a 0
        b c : α
        h : LT.lt a 0
        ⊢ Eq (Set.preimage (fun x => HMul.hMul x a) (Set.uIcc b c)) (Set.uIcc (HDiv.hD …
      -/
      simp [← Icc_min_max, h, h.le, min_div_div_right_of_nonpos, max_div_div_right_of_nonpos])
      /-
        🎉 no goals
      -/
                         /-
                           α : Type u_1
                           inst✝ : LinearOrderedField α
                           a : α
                           ha✝ : Ne a 0
                           b c : α
                           ha : LT.lt 0 a
                           ⊢ Eq (Set.preimage (fun x => HMul.hMul x a) (Set.uIcc b c)) (Set.uIcc (HDiv.hD …
                         -/
    fun ha : 0 < a => by simp [← Icc_min_max, ha, ha.le, min_div_div_right, max_div_div_right]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem preimage_const_mul_uIcc (ha : a ≠ 0) (b c : α) :
    (a * ·) ⁻¹' [[b, c]] = [[b / a, c / a]] := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a : α
    ha : Ne a 0
    b c : α
    ⊢ Eq (Set.preimage (fun x => HMul.hMul a x) (Set.uIcc b c)) (Set.uIcc (HDiv.hD …
  -/
  simp only [← preimage_mul_const_uIcc ha, mul_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_div_const_uIcc (ha : a ≠ 0) (b c : α) :
    (fun x => x / a) ⁻¹' [[b, c]] = [[b * a, c * a]] := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a : α
    ha : Ne a 0
    b c : α
    ⊢ Eq (Set.preimage (fun x => HDiv.hDiv x a) (Set.uIcc b c)) (Set.uIcc (HMul.hM …
  -/
  simp only [div_eq_mul_inv, preimage_mul_const_uIcc (inv_ne_zero ha), inv_inv]
  /-
    🎉 no goals
  -/


lemma preimage_const_mul_Ioi_or_Iio (hb : a ≠ 0) {U V : Set α}
    (hU : U ∈ {s | ∃ a, s = Ioi a ∨ s = Iio a}) (hV : V = HMul.hMul a ⁻¹' U) :
    V ∈ {s | ∃ a, s = Ioi a ∨ s = Iio a} := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a : α
    hb : Ne a 0
    U V : Set α
    hU : Membership.mem (setOf fun s => Exists fun a => Or (Eq s (Set.Ioi a)) (Eq  …
    hV : Eq V (Set.preimage (HMul.hMul a) U)
    ⊢ Membership.mem (setOf fun s => Exists fun a => Or (Eq s (Set.Ioi a)) (Eq s ( …
  -/
  obtain ⟨aU, (haU | haU)⟩ := hU <;>
  /-
    case intro.inl
    α : Type u_1
    inst✝ : LinearOrderedField α
    a : α
    hb : Ne a 0
    U V : Set α
    hV : Eq V (Set.preimage (HMul.hMul a) U)
    aU : α
    haU : Eq U (Set.Ioi aU)
    ⊢ Membership.mem (setOf fun s => Exists fun a => Or (Eq s (Set.Ioi a)) (Eq s ( …
  -/
  simp only [hV, haU, mem_setOf_eq] <;>
  /-
    case intro.inl
    α : Type u_1
    inst✝ : LinearOrderedField α
    a : α
    hb : Ne a 0
    U V : Set α
    hV : Eq V (Set.preimage (HMul.hMul a) U)
    aU : α
    haU : Eq U (Set.Ioi aU)
    ⊢ Exists fun a_1 => Or (Eq (Set.preimage (HMul.hMul a) (Set.Ioi aU)) (Set.Ioi  …
  -/
  use a⁻¹ * aU <;>
  /-
    case h
    α : Type u_1
    inst✝ : LinearOrderedField α
    a : α
    hb : Ne a 0
    U V : Set α
    hV : Eq V (Set.preimage (HMul.hMul a) U)
    aU : α
    haU : Eq U (Set.Ioi aU)
    ⊢ Or (Eq (Set.preimage (HMul.hMul a) (Set.Ioi aU)) (Set.Ioi (HMul.hMul (Inv.in …
  -/
  rcases lt_or_gt_of_ne hb with (hb | hb)
    /-
      case h.inl
      α : Type u_1
      inst✝ : LinearOrderedField α
      a : α
      hb✝ : Ne a 0
      U V : Set α
      hV : Eq V (Set.preimage (HMul.hMul a) U)
      aU : α
      haU : Eq U (Set.Ioi aU)
      hb : LT.lt a 0
      ⊢ Or (Eq (Set.preimage (HMul.hMul a) (Set.Ioi aU)) (Set.Ioi (HMul.hMul (Inv.in …
    -/
  · right; rw [Set.preimage_const_mul_Ioi_of_neg _ hb, div_eq_inv_mul]
           /-
             🎉 no goals
           -/
    /-
      case h.inr
      α : Type u_1
      inst✝ : LinearOrderedField α
      a : α
      hb✝ : Ne a 0
      U V : Set α
      hV : Eq V (Set.preimage (HMul.hMul a) U)
      aU : α
      haU : Eq U (Set.Ioi aU)
      hb : GT.gt a 0
      ⊢ Or (Eq (Set.preimage (HMul.hMul a) (Set.Ioi aU)) (Set.Ioi (HMul.hMul (Inv.in …
    -/
  · left; rw [Set.preimage_const_mul_Ioi _ hb, div_eq_inv_mul]
          /-
            🎉 no goals
          -/
    /-
      case h.inl
      α : Type u_1
      inst✝ : LinearOrderedField α
      a : α
      hb✝ : Ne a 0
      U V : Set α
      hV : Eq V (Set.preimage (HMul.hMul a) U)
      aU : α
      haU : Eq U (Set.Iio aU)
      hb : LT.lt a 0
      ⊢ Or (Eq (Set.preimage (HMul.hMul a) (Set.Iio aU)) (Set.Ioi (HMul.hMul (Inv.in …
    -/
  · left; rw [Set.preimage_const_mul_Iio_of_neg _ hb, div_eq_inv_mul]
          /-
            🎉 no goals
          -/
    /-
      case h.inr
      α : Type u_1
      inst✝ : LinearOrderedField α
      a : α
      hb✝ : Ne a 0
      U V : Set α
      hV : Eq V (Set.preimage (HMul.hMul a) U)
      aU : α
      haU : Eq U (Set.Iio aU)
      hb : GT.gt a 0
      ⊢ Or (Eq (Set.preimage (HMul.hMul a) (Set.Iio aU)) (Set.Ioi (HMul.hMul (Inv.in …
    -/
  · right; rw [Set.preimage_const_mul_Iio _ hb, div_eq_inv_mul]
           /-
             🎉 no goals
           -/


@[simp]
theorem image_mul_const_uIcc (a b c : α) : (· * a) '' [[b, c]] = [[b * a, c * a]] :=
                        /-
                          α : Type u_1
                          inst✝ : LinearOrderedField α
                          a b c : α
                          ha : Eq a 0
                          ⊢ Eq (Set.image (fun x => HMul.hMul x a) (Set.uIcc b c)) (Set.uIcc (HMul.hMul  …
                        -/
  if ha : a = 0 then by simp [ha]
                        /-
                          🎉 no goals
                        -/
  else calc
    (fun x => x * a) '' [[b, c]] = (· * a⁻¹) ⁻¹' [[b, c]] :=
      (Units.mk0 a ha).mulRight.image_eq_preimage _
                                            /-
                                              α : Type u_1
                                              inst✝ : LinearOrderedField α
                                              a b c : α
                                              ha : Not (Eq a 0)
                                              ⊢ Eq (Set.preimage (fun x => HMul.hMul x (Inv.inv a)) (Set.uIcc b c)) (Set.pre …
                                            -/
    _ = (fun x => x / a) ⁻¹' [[b, c]] := by simp only [div_eq_mul_inv]
                                            /-
                                              🎉 no goals
                                            -/
    _ = [[b * a, c * a]] := preimage_div_const_uIcc ha _ _


@[simp]
theorem image_const_mul_uIcc (a b c : α) : (a * ·) '' [[b, c]] = [[a * b, a * c]] := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : α
    ⊢ Eq (Set.image (fun x => HMul.hMul a x) (Set.uIcc b c)) (Set.uIcc (HMul.hMul  …
  -/
  simpa only [mul_comm] using image_mul_const_uIcc a b c
  /-
    🎉 no goals
  -/


@[simp]
theorem image_div_const_uIcc (a b c : α) : (fun x => x / a) '' [[b, c]] = [[b / a, c / a]] := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : α
    ⊢ Eq (Set.image (fun x => HDiv.hDiv x a) (Set.uIcc b c)) (Set.uIcc (HDiv.hDiv  …
  -/
  simp only [div_eq_mul_inv, image_mul_const_uIcc]
  /-
    🎉 no goals
  -/


theorem image_mul_right_Icc' (a b : α) {c : α} (h : 0 < c) :
    (fun x => x * c) '' Icc a b = Icc (a * c) (b * c) :=
                                                               /-
                                                                 α : Type u_1
                                                                 inst✝ : LinearOrderedField α
                                                                 a b c : α
                                                                 h : LT.lt 0 c
                                                                 ⊢ Eq (Set.preimage (⇑(Equiv.symm (Units.mk0 c ⋯).mulRight)) (Set.Icc a b)) (Se …
                                                               -/
  ((Units.mk0 c h.ne').mulRight.image_eq_preimage _).trans (by simp [h, division_def])
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem image_mul_right_Icc {a b c : α} (hab : a ≤ b) (hc : 0 ≤ c) :
    (fun x => x * c) '' Icc a b = Icc (a * c) (b * c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : α
    hab : LE.le a b
    hc : LE.le 0 c
    ⊢ Eq (Set.image (fun x => HMul.hMul x c) (Set.Icc a b)) (Set.Icc (HMul.hMul a  …
  -/
  cases eq_or_lt_of_le hc
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b c : α
      hab : LE.le a b
      hc : LE.le 0 c
      h✝ : Eq 0 c
      ⊢ Eq (Set.image (fun x => HMul.hMul x c) (Set.Icc a b)) (Set.Icc (HMul.hMul a  …
    -/
  · subst c
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedField α
      a b : α
      hab : LE.le a b
      hc : LE.le 0 0
      ⊢ Eq (Set.image (fun x => HMul.hMul x 0) (Set.Icc a b)) (Set.Icc (HMul.hMul a  …
    -/
    simp [(nonempty_Icc.2 hab).image_const]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : α
    hab : LE.le a b
    hc : LE.le 0 c
    h✝ : LT.lt 0 c
    ⊢ Eq (Set.image (fun x => HMul.hMul x c) (Set.Icc a b)) (Set.Icc (HMul.hMul a  …
  -/
  exact image_mul_right_Icc' a b ‹0 < c›
  /-
    🎉 no goals
  -/


theorem image_mul_left_Icc' {a : α} (h : 0 < a) (b c : α) :
    (a * ·) '' Icc b c = Icc (a * b) (a * c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a : α
    h : LT.lt 0 a
    b c : α
    ⊢ Eq (Set.image (fun x => HMul.hMul a x) (Set.Icc b c)) (Set.Icc (HMul.hMul a  …
  -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  convert image_mul_right_Icc' b c h using 1 <;> simp only [mul_comm _ a]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem image_mul_left_Icc {a b c : α} (ha : 0 ≤ a) (hbc : b ≤ c) :
    (a * ·) '' Icc b c = Icc (a * b) (a * c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a b c : α
    ha : LE.le 0 a
    hbc : LE.le b c
    ⊢ Eq (Set.image (fun x => HMul.hMul a x) (Set.Icc b c)) (Set.Icc (HMul.hMul a  …
  -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  convert image_mul_right_Icc hbc ha using 1 <;> simp only [mul_comm _ a]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem image_mul_right_Ioo (a b : α) {c : α} (h : 0 < c) :
    (fun x => x * c) '' Ioo a b = Ioo (a * c) (b * c) :=
                                                               /-
                                                                 α : Type u_1
                                                                 inst✝ : LinearOrderedField α
                                                                 a b c : α
                                                                 h : LT.lt 0 c
                                                                 ⊢ Eq (Set.preimage (⇑(Equiv.symm (Units.mk0 c ⋯).mulRight)) (Set.Ioo a b)) (Se …
                                                               -/
  ((Units.mk0 c h.ne').mulRight.image_eq_preimage _).trans (by simp [h, division_def])
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem image_mul_left_Ioo {a : α} (h : 0 < a) (b c : α) :
    (a * ·) '' Ioo b c = Ioo (a * b) (a * c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a : α
    h : LT.lt 0 a
    b c : α
    ⊢ Eq (Set.image (fun x => HMul.hMul a x) (Set.Ioo b c)) (Set.Ioo (HMul.hMul a  …
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  convert image_mul_right_Ioo b c h using 1 <;> simp only [mul_comm _ a]
                                                /-
                                                  🎉 no goals
                                                -/


theorem image_mul_right_Ico (a b : α) {c : α} (h : 0 < c) :
    (fun x => x * c) '' Ico a b = Ico (a * c) (b * c) :=
                                                               /-
                                                                 α : Type u_1
                                                                 inst✝ : LinearOrderedField α
                                                                 a b c : α
                                                                 h : LT.lt 0 c
                                                                 ⊢ Eq (Set.preimage (⇑(Equiv.symm (Units.mk0 c ⋯).mulRight)) (Set.Ico a b)) (Se …
                                                               -/
  ((Units.mk0 c h.ne').mulRight.image_eq_preimage _).trans (by simp [h, division_def])
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem image_mul_left_Ico {a : α} (h : 0 < a) (b c : α) :
    (a * ·) '' Ico b c = Ico (a * b) (a * c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a : α
    h : LT.lt 0 a
    b c : α
    ⊢ Eq (Set.image (fun x => HMul.hMul a x) (Set.Ico b c)) (Set.Ico (HMul.hMul a  …
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  convert image_mul_right_Ico b c h using 1 <;> simp only [mul_comm _ a]
                                                /-
                                                  🎉 no goals
                                                -/


theorem image_mul_right_Ioc (a b : α) {c : α} (h : 0 < c) :
    (fun x => x * c) '' Ioc a b = Ioc (a * c) (b * c) :=
                                                               /-
                                                                 α : Type u_1
                                                                 inst✝ : LinearOrderedField α
                                                                 a b c : α
                                                                 h : LT.lt 0 c
                                                                 ⊢ Eq (Set.preimage (⇑(Equiv.symm (Units.mk0 c ⋯).mulRight)) (Set.Ioc a b)) (Se …
                                                               -/
  ((Units.mk0 c h.ne').mulRight.image_eq_preimage _).trans (by simp [h, division_def])
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem image_mul_left_Ioc {a : α} (h : 0 < a) (b c : α) :
    (a * ·) '' Ioc b c = Ioc (a * b) (a * c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a : α
    h : LT.lt 0 a
    b c : α
    ⊢ Eq (Set.image (fun x => HMul.hMul a x) (Set.Ioc b c)) (Set.Ioc (HMul.hMul a  …
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  convert image_mul_right_Ioc b c h using 1 <;> simp only [mul_comm _ a]
                                                /-
                                                  🎉 no goals
                                                -/


/-- The (pre)image under `inv` of `Ioo 0 a` is `Ioi a⁻¹`. -/
theorem inv_Ioo_0_left {a : α} (ha : 0 < a) : (Ioo 0 a)⁻¹ = Ioi a⁻¹ := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a : α
    ha : LT.lt 0 a
    ⊢ Eq (Inv.inv (Set.Ioo 0 a)) (Set.Ioi (Inv.inv a))
  -/
  ext x
  exact
    ⟨fun h => inv_inv x ▸ (inv_lt_inv₀ ha h.1).2 h.2, fun h =>
      ⟨inv_pos.2 <| (inv_pos.2 ha).trans h,
        inv_inv a ▸ (inv_lt_inv₀ ((inv_pos.2 ha).trans h)
          (inv_pos.2 ha)).2 h⟩⟩


theorem inv_Ioi₀ {a : α} (ha : 0 < a) : (Ioi a)⁻¹ = Ioo 0 a⁻¹ := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a : α
    ha : LT.lt 0 a
    ⊢ Eq (Inv.inv (Set.Ioi a)) (Set.Ioo 0 (Inv.inv a))
  -/
  rw [inv_eq_iff_eq_inv, inv_Ioo_0_left (inv_pos.2 ha), inv_inv]
  /-
    🎉 no goals
  -/


theorem image_const_mul_Ioi_zero {k : Type*} [LinearOrderedField k] {x : k} (hx : 0 < x) :
    (fun y => x * y) '' Ioi (0 : k) = Ioi 0 := by
  erw [(Units.mk0 x hx.ne').mulLeft.image_eq_preimage,
    preimage_const_mul_Ioi 0 (inv_pos.mpr hx), zero_div]


@[simp]
theorem image_affine_Icc' {a : α} (h : 0 < a) (b c d : α) :
    (a * · + b) '' Icc c d = Icc (a * c + b) (a * d + b) := by
  suffices (· + b) '' ((a * ·) '' Icc c d) = Icc (a * c + b) (a * d + b) by
    rwa [Set.image_image] at this
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a : α
    h : LT.lt 0 a
    b c d : α
    ⊢ Eq (Set.image (fun x => HAdd.hAdd x b) (Set.image (fun x => HMul.hMul a x) ( …
  -/
  rw [image_mul_left_Icc' h, image_add_const_Icc]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_affine_Ico {a : α} (h : 0 < a) (b c d : α) :
    (a * · + b) '' Ico c d = Ico (a * c + b) (a * d + b) := by
  suffices (· + b) '' ((a * ·) '' Ico c d) = Ico (a * c + b) (a * d + b) by
    rwa [Set.image_image] at this
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a : α
    h : LT.lt 0 a
    b c d : α
    ⊢ Eq (Set.image (fun x => HAdd.hAdd x b) (Set.image (fun x => HMul.hMul a x) ( …
  -/
  rw [image_mul_left_Ico h, image_add_const_Ico]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_affine_Ioc {a : α} (h : 0 < a) (b c d : α) :
    (a * · + b) '' Ioc c d = Ioc (a * c + b) (a * d + b) := by
  suffices (· + b) '' ((a * ·) '' Ioc c d) = Ioc (a * c + b) (a * d + b) by
    rwa [Set.image_image] at this
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a : α
    h : LT.lt 0 a
    b c d : α
    ⊢ Eq (Set.image (fun x => HAdd.hAdd x b) (Set.image (fun x => HMul.hMul a x) ( …
  -/
  rw [image_mul_left_Ioc h, image_add_const_Ioc]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_affine_Ioo {a : α} (h : 0 < a) (b c d : α) :
    (a * · + b) '' Ioo c d = Ioo (a * c + b) (a * d + b) := by
  suffices (· + b) '' ((a * ·) '' Ioo c d) = Ioo (a * c + b) (a * d + b) by
    rwa [Set.image_image] at this
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    a : α
    h : LT.lt 0 a
    b c d : α
    ⊢ Eq (Set.image (fun x => HAdd.hAdd x b) (Set.image (fun x => HMul.hMul a x) ( …
  -/
  rw [image_mul_left_Ioo h, image_add_const_Ioo]
  /-
    🎉 no goals
  -/


