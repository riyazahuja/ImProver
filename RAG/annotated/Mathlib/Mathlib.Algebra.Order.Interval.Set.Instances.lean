instance zero : Zero (Icc (0 : α) 1) where zero := ⟨0, left_mem_Icc.2 zero_le_one⟩


instance one : One (Icc (0 : α) 1) where one := ⟨1, right_mem_Icc.2 zero_le_one⟩


@[simp, norm_cast]
theorem coe_zero : ↑(0 : Icc (0 : α) 1) = (0 : α) :=
  rfl


@[simp, norm_cast]
theorem coe_one : ↑(1 : Icc (0 : α) 1) = (1 : α) :=
  rfl


@[simp]
theorem mk_zero (h : (0 : α) ∈ Icc (0 : α) 1) : (⟨0, h⟩ : Icc (0 : α) 1) = 0 :=
  rfl


@[simp]
theorem mk_one (h : (1 : α) ∈ Icc (0 : α) 1) : (⟨1, h⟩ : Icc (0 : α) 1) = 1 :=
  rfl


@[simp, norm_cast]
theorem coe_eq_zero {x : Icc (0 : α) 1} : (x : α) = 0 ↔ x = 0 := by
  /-
    α : Type u_1
    inst✝ : OrderedSemiring α
    x : ↑(Set.Icc 0 1)
    ⊢ Iff (Eq (↑x) 0) (Eq x 0)
  -/
  symm
  /-
    α : Type u_1
    inst✝ : OrderedSemiring α
    x : ↑(Set.Icc 0 1)
    ⊢ Iff (Eq x 0) (Eq (↑x) 0)
  -/
  exact Subtype.ext_iff
  /-
    🎉 no goals
  -/


theorem coe_ne_zero {x : Icc (0 : α) 1} : (x : α) ≠ 0 ↔ x ≠ 0 :=
  not_iff_not.mpr coe_eq_zero


@[simp, norm_cast]
theorem coe_eq_one {x : Icc (0 : α) 1} : (x : α) = 1 ↔ x = 1 := by
  /-
    α : Type u_1
    inst✝ : OrderedSemiring α
    x : ↑(Set.Icc 0 1)
    ⊢ Iff (Eq (↑x) 1) (Eq x 1)
  -/
  symm
  /-
    α : Type u_1
    inst✝ : OrderedSemiring α
    x : ↑(Set.Icc 0 1)
    ⊢ Iff (Eq x 1) (Eq (↑x) 1)
  -/
  exact Subtype.ext_iff
  /-
    🎉 no goals
  -/


theorem coe_ne_one {x : Icc (0 : α) 1} : (x : α) ≠ 1 ↔ x ≠ 1 :=
  not_iff_not.mpr coe_eq_one


theorem coe_nonneg (x : Icc (0 : α) 1) : 0 ≤ (x : α) :=
  x.2.1


theorem coe_le_one (x : Icc (0 : α) 1) : (x : α) ≤ 1 :=
  x.2.2


/-- like `coe_nonneg`, but with the inequality in `Icc (0:α) 1`. -/
theorem nonneg {t : Icc (0 : α) 1} : 0 ≤ t :=
  t.2.1


/-- like `coe_le_one`, but with the inequality in `Icc (0:α) 1`. -/
theorem le_one {t : Icc (0 : α) 1} : t ≤ 1 :=
  t.2.2


instance mul : Mul (Icc (0 : α) 1) where
  mul p q := ⟨p * q, ⟨mul_nonneg p.2.1 q.2.1, mul_le_one₀ p.2.2 q.2.1 q.2.2⟩⟩


instance pow : Pow (Icc (0 : α) 1) ℕ where
  pow p n := ⟨p.1 ^ n, ⟨pow_nonneg p.2.1 n, pow_le_one₀ p.2.1 p.2.2⟩⟩


@[simp, norm_cast]
theorem coe_mul (x y : Icc (0 : α) 1) : ↑(x * y) = (x * y : α) :=
  rfl


@[simp, norm_cast]
theorem coe_pow (x : Icc (0 : α) 1) (n : ℕ) : ↑(x ^ n) = ((x : α) ^ n) :=
  rfl


theorem mul_le_left {x y : Icc (0 : α) 1} : x * y ≤ x :=
  (mul_le_mul_of_nonneg_left y.2.2 x.2.1).trans_eq (mul_one _)


theorem mul_le_right {x y : Icc (0 : α) 1} : x * y ≤ y :=
  (mul_le_mul_of_nonneg_right x.2.2 y.2.1).trans_eq (one_mul _)


instance monoidWithZero : MonoidWithZero (Icc (0 : α) 1) :=
  Subtype.coe_injective.monoidWithZero _ coe_zero coe_one coe_mul coe_pow


instance commMonoidWithZero {α : Type*} [OrderedCommSemiring α] :
    CommMonoidWithZero (Icc (0 : α) 1) :=
  Subtype.coe_injective.commMonoidWithZero _ coe_zero coe_one coe_mul coe_pow


instance cancelMonoidWithZero {α : Type*} [OrderedRing α] [NoZeroDivisors α] :
    CancelMonoidWithZero (Icc (0 : α) 1) :=
  @Function.Injective.cancelMonoidWithZero α _ NoZeroDivisors.toCancelMonoidWithZero _ _ _ _
    (fun v => v.val) Subtype.coe_injective coe_zero coe_one coe_mul coe_pow


instance cancelCommMonoidWithZero {α : Type*} [OrderedCommRing α] [NoZeroDivisors α] :
    CancelCommMonoidWithZero (Icc (0 : α) 1) :=
  @Function.Injective.cancelCommMonoidWithZero α _ NoZeroDivisors.toCancelCommMonoidWithZero _ _ _ _
    (fun v => v.val) Subtype.coe_injective coe_zero coe_one coe_mul coe_pow


theorem one_sub_mem {t : β} (ht : t ∈ Icc (0 : β) 1) : 1 - t ∈ Icc (0 : β) 1 := by
  /-
    β : Type u_2
    inst✝ : OrderedRing β
    t : β
    ht : Membership.mem (Set.Icc 0 1) t
    ⊢ Membership.mem (Set.Icc 0 1) (HSub.hSub 1 t)
  -/
  rw [mem_Icc] at *
  /-
    β : Type u_2
    inst✝ : OrderedRing β
    t : β
    ht : And (LE.le 0 t) (LE.le t 1)
    ⊢ And (LE.le 0 (HSub.hSub 1 t)) (LE.le (HSub.hSub 1 t) 1)
  -/
  exact ⟨sub_nonneg.2 ht.2, (sub_le_self_iff _).2 ht.1⟩
  /-
    🎉 no goals
  -/


theorem mem_iff_one_sub_mem {t : β} : t ∈ Icc (0 : β) 1 ↔ 1 - t ∈ Icc (0 : β) 1 :=
  ⟨one_sub_mem, fun h => sub_sub_cancel 1 t ▸ one_sub_mem h⟩


                                                                   /-
                                                                     β : Type u_2
                                                                     inst✝ : OrderedRing β
                                                                     x : ↑(Set.Icc 0 1)
                                                                     ⊢ LE.le 0 (HSub.hSub 1 ↑x)
                                                                   -/
theorem one_sub_nonneg (x : Icc (0 : β) 1) : 0 ≤ 1 - (x : β) := by simpa using x.2.2
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


                                                                   /-
                                                                     β : Type u_2
                                                                     inst✝ : OrderedRing β
                                                                     x : ↑(Set.Icc 0 1)
                                                                     ⊢ LE.le (HSub.hSub 1 ↑x) 1
                                                                   -/
theorem one_sub_le_one (x : Icc (0 : β) 1) : 1 - (x : β) ≤ 1 := by simpa using x.2.1
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


instance zero [Nontrivial α] : Zero (Ico (0 : α) 1) where zero := ⟨0, left_mem_Ico.2 zero_lt_one⟩


@[simp, norm_cast]
theorem coe_zero [Nontrivial α] : ↑(0 : Ico (0 : α) 1) = (0 : α) :=
  rfl


@[simp]
theorem mk_zero [Nontrivial α] (h : (0 : α) ∈ Ico (0 : α) 1) : (⟨0, h⟩ : Ico (0 : α) 1) = 0 :=
  rfl


@[simp, norm_cast]
theorem coe_eq_zero [Nontrivial α] {x : Ico (0 : α) 1} : (x : α) = 0 ↔ x = 0 := by
  /-
    α : Type u_1
    inst✝¹ : OrderedSemiring α
    inst✝ : Nontrivial α
    x : ↑(Set.Ico 0 1)
    ⊢ Iff (Eq (↑x) 0) (Eq x 0)
  -/
  symm
  /-
    α : Type u_1
    inst✝¹ : OrderedSemiring α
    inst✝ : Nontrivial α
    x : ↑(Set.Ico 0 1)
    ⊢ Iff (Eq x 0) (Eq (↑x) 0)
  -/
  exact Subtype.ext_iff
  /-
    🎉 no goals
  -/


theorem coe_ne_zero [Nontrivial α] {x : Ico (0 : α) 1} : (x : α) ≠ 0 ↔ x ≠ 0 :=
  not_iff_not.mpr coe_eq_zero


theorem coe_nonneg (x : Ico (0 : α) 1) : 0 ≤ (x : α) :=
  x.2.1


theorem coe_lt_one (x : Ico (0 : α) 1) : (x : α) < 1 :=
  x.2.2


/-- like `coe_nonneg`, but with the inequality in `Ico (0:α) 1`. -/
theorem nonneg [Nontrivial α] {t : Ico (0 : α) 1} : 0 ≤ t :=
  t.2.1


instance mul : Mul (Ico (0 : α) 1) where
  mul p q :=
    ⟨p * q, ⟨mul_nonneg p.2.1 q.2.1, mul_lt_one_of_nonneg_of_lt_one_right p.2.2.le q.2.1 q.2.2⟩⟩


@[simp, norm_cast]
theorem coe_mul (x y : Ico (0 : α) 1) : ↑(x * y) = (x * y : α) :=
  rfl


instance semigroup : Semigroup (Ico (0 : α) 1) :=
  Subtype.coe_injective.semigroup _ coe_mul


instance commSemigroup {α : Type*} [OrderedCommSemiring α] : CommSemigroup (Ico (0 : α) 1) :=
  Subtype.coe_injective.commSemigroup _ coe_mul


instance one : One (Ioc (0 : α) 1) where one := ⟨1, ⟨zero_lt_one, le_refl 1⟩⟩


@[simp, norm_cast]
theorem coe_one : ↑(1 : Ioc (0 : α) 1) = (1 : α) :=
  rfl


@[simp]
theorem mk_one (h : (1 : α) ∈ Ioc (0 : α) 1) : (⟨1, h⟩ : Ioc (0 : α) 1) = 1 :=
  rfl


@[simp, norm_cast]
theorem coe_eq_one {x : Ioc (0 : α) 1} : (x : α) = 1 ↔ x = 1 := by
  /-
    α : Type u_1
    inst✝ : StrictOrderedSemiring α
    x : ↑(Set.Ioc 0 1)
    ⊢ Iff (Eq (↑x) 1) (Eq x 1)
  -/
  symm
  /-
    α : Type u_1
    inst✝ : StrictOrderedSemiring α
    x : ↑(Set.Ioc 0 1)
    ⊢ Iff (Eq x 1) (Eq (↑x) 1)
  -/
  exact Subtype.ext_iff
  /-
    🎉 no goals
  -/


theorem coe_ne_one {x : Ioc (0 : α) 1} : (x : α) ≠ 1 ↔ x ≠ 1 :=
  not_iff_not.mpr coe_eq_one


theorem coe_pos (x : Ioc (0 : α) 1) : 0 < (x : α) :=
  x.2.1


theorem coe_le_one (x : Ioc (0 : α) 1) : (x : α) ≤ 1 :=
  x.2.2


/-- like `coe_le_one`, but with the inequality in `Ioc (0:α) 1`. -/
theorem le_one {t : Ioc (0 : α) 1} : t ≤ 1 :=
  t.2.2


instance mul : Mul (Ioc (0 : α) 1) where
  mul p q := ⟨p.1 * q.1, ⟨mul_pos p.2.1 q.2.1, mul_le_one₀ p.2.2 (le_of_lt q.2.1) q.2.2⟩⟩


instance pow : Pow (Ioc (0 : α) 1) ℕ where
  pow p n := ⟨p.1 ^ n, ⟨pow_pos p.2.1 n, pow_le_one₀ (le_of_lt p.2.1) p.2.2⟩⟩


@[simp, norm_cast]
theorem coe_mul (x y : Ioc (0 : α) 1) : ↑(x * y) = (x * y : α) :=
  rfl


@[simp, norm_cast]
theorem coe_pow (x : Ioc (0 : α) 1) (n : ℕ) : ↑(x ^ n) = ((x : α) ^ n) :=
  rfl


instance semigroup : Semigroup (Ioc (0 : α) 1) :=
  Subtype.coe_injective.semigroup _ coe_mul


instance monoid : Monoid (Ioc (0 : α) 1) :=
  Subtype.coe_injective.monoid _ coe_one coe_mul coe_pow


instance commSemigroup {α : Type*} [StrictOrderedCommSemiring α] : CommSemigroup (Ioc (0 : α) 1) :=
  Subtype.coe_injective.commSemigroup _ coe_mul


instance commMonoid {α : Type*} [StrictOrderedCommSemiring α] :
    CommMonoid (Ioc (0 : α) 1) :=
  Subtype.coe_injective.commMonoid _ coe_one coe_mul coe_pow


instance cancelMonoid {α : Type*} [StrictOrderedRing α] [IsDomain α] :
    CancelMonoid (Ioc (0 : α) 1) :=
  { Set.Ioc.monoid with
    mul_left_cancel := fun a _ _ h =>
      Subtype.ext <| mul_left_cancel₀ a.prop.1.ne' <| (congr_arg Subtype.val h : _)
    mul_right_cancel := fun _ b _ h =>
      Subtype.ext <| mul_right_cancel₀ b.prop.1.ne' <| (congr_arg Subtype.val h : _) }


instance cancelCommMonoid {α : Type*} [StrictOrderedCommRing α] [IsDomain α] :
    CancelCommMonoid (Ioc (0 : α) 1) :=
  { Set.Ioc.cancelMonoid, Set.Ioc.commMonoid with }


theorem pos (x : Ioo (0 : α) 1) : 0 < (x : α) :=
  x.2.1


theorem lt_one (x : Ioo (0 : α) 1) : (x : α) < 1 :=
  x.2.2


instance mul : Mul (Ioo (0 : α) 1) where
  mul p q :=
    ⟨p.1 * q.1, ⟨mul_pos p.2.1 q.2.1, mul_lt_one_of_nonneg_of_lt_one_right p.2.2.le q.2.1.le q.2.2⟩⟩


@[simp, norm_cast]
theorem coe_mul (x y : Ioo (0 : α) 1) : ↑(x * y) = (x * y : α) :=
  rfl


instance semigroup : Semigroup (Ioo (0 : α) 1) :=
  Subtype.coe_injective.semigroup _ coe_mul


instance commSemigroup {α : Type*} [StrictOrderedCommSemiring α] : CommSemigroup (Ioo (0 : α) 1) :=
  Subtype.coe_injective.commSemigroup _ coe_mul


theorem one_sub_mem {t : β} (ht : t ∈ Ioo (0 : β) 1) : 1 - t ∈ Ioo (0 : β) 1 := by
  /-
    β : Type u_2
    inst✝ : OrderedRing β
    t : β
    ht : Membership.mem (Set.Ioo 0 1) t
    ⊢ Membership.mem (Set.Ioo 0 1) (HSub.hSub 1 t)
  -/
  rw [mem_Ioo] at *
  /-
    β : Type u_2
    inst✝ : OrderedRing β
    t : β
    ht : And (LT.lt 0 t) (LT.lt t 1)
    ⊢ And (LT.lt 0 (HSub.hSub 1 t)) (LT.lt (HSub.hSub 1 t) 1)
  -/
  refine ⟨sub_pos.2 ht.2, ?_⟩
  /-
    β : Type u_2
    inst✝ : OrderedRing β
    t : β
    ht : And (LT.lt 0 t) (LT.lt t 1)
    ⊢ LT.lt (HSub.hSub 1 t) 1
  -/
  exact lt_of_le_of_ne ((sub_le_self_iff 1).2 ht.1.le) (mt sub_eq_self.mp ht.1.ne')
  /-
    🎉 no goals
  -/


theorem mem_iff_one_sub_mem {t : β} : t ∈ Ioo (0 : β) 1 ↔ 1 - t ∈ Ioo (0 : β) 1 :=
  ⟨one_sub_mem, fun h => sub_sub_cancel 1 t ▸ one_sub_mem h⟩


                                                                  /-
                                                                    β : Type u_2
                                                                    inst✝ : OrderedRing β
                                                                    x : ↑(Set.Ioo 0 1)
                                                                    ⊢ LT.lt 0 (HSub.hSub 1 ↑x)
                                                                  -/
theorem one_minus_pos (x : Ioo (0 : β) 1) : 0 < 1 - (x : β) := by simpa using x.2.2
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


                                                                     /-
                                                                       β : Type u_2
                                                                       inst✝ : OrderedRing β
                                                                       x : ↑(Set.Ioo 0 1)
                                                                       ⊢ LT.lt (HSub.hSub 1 ↑x) 1
                                                                     -/
theorem one_minus_lt_one (x : Ioo (0 : β) 1) : 1 - (x : β) < 1 := by simpa using x.2.1
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


