@[to_additive (attr := simp)]
theorem max_one_div_max_inv_one_eq_self (a : α) : max a 1 / max a⁻¹ 1 = a := by
  /-
    α : Type u_1
    inst✝² : Group α
    inst✝¹ : LinearOrder α
    inst✝ : MulLeftMono α
    a : α
    ⊢ Eq (HDiv.hDiv (Max.max a 1) (Max.max (Inv.inv a) 1)) a
  -/
                                       /-
                                         🎉 no goals
                                       -/
  rcases le_total a 1 with (h | h) <;> simp [h]
                                       /-
                                         🎉 no goals
                                       -/


alias max_zero_sub_eq_self := max_zero_sub_max_neg_zero_eq_self


@[to_additive]
lemma max_inv_one (a : α) : max a⁻¹ 1 = a⁻¹ * max a 1 := by
  /-
    α : Type u_1
    inst✝² : Group α
    inst✝¹ : LinearOrder α
    inst✝ : MulLeftMono α
    a : α
    ⊢ Eq (Max.max (Inv.inv a) 1) (HMul.hMul (Inv.inv a) (Max.max a 1))
  -/
  rw [eq_inv_mul_iff_mul_eq, ← eq_div_iff_mul_eq', max_one_div_max_inv_one_eq_self]
  /-
    🎉 no goals
  -/


@[to_additive min_neg_neg]
theorem min_inv_inv' (a b : α) : min a⁻¹ b⁻¹ = (max a b)⁻¹ :=
  Eq.symm <| (@Monotone.map_max α αᵒᵈ _ _ Inv.inv a b) fun _ _ =>
    inv_le_inv_iff.mpr


@[to_additive max_neg_neg]
theorem max_inv_inv' (a b : α) : max a⁻¹ b⁻¹ = (min a b)⁻¹ :=
  Eq.symm <| (@Monotone.map_min α αᵒᵈ _ _ Inv.inv a b) fun _ _ =>
    inv_le_inv_iff.mpr


@[to_additive min_sub_sub_right]
theorem min_div_div_right' (a b c : α) : min (a / c) (b / c) = min a b / c := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroup α
    a b c : α
    ⊢ Eq (Min.min (HDiv.hDiv a c) (HDiv.hDiv b c)) (HDiv.hDiv (Min.min a b) c)
  -/
  simpa only [div_eq_mul_inv] using min_mul_mul_right a b c⁻¹
  /-
    🎉 no goals
  -/


@[to_additive max_sub_sub_right]
theorem max_div_div_right' (a b c : α) : max (a / c) (b / c) = max a b / c := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroup α
    a b c : α
    ⊢ Eq (Max.max (HDiv.hDiv a c) (HDiv.hDiv b c)) (HDiv.hDiv (Max.max a b) c)
  -/
  simpa only [div_eq_mul_inv] using max_mul_mul_right a b c⁻¹
  /-
    🎉 no goals
  -/


@[to_additive min_sub_sub_left]
theorem min_div_div_left' (a b c : α) : min (a / b) (a / c) = a / max b c := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroup α
    a b c : α
    ⊢ Eq (Min.min (HDiv.hDiv a b) (HDiv.hDiv a c)) (HDiv.hDiv a (Max.max b c))
  -/
  simp only [div_eq_mul_inv, min_mul_mul_left, min_inv_inv']
  /-
    🎉 no goals
  -/


@[to_additive max_sub_sub_left]
theorem max_div_div_left' (a b c : α) : max (a / b) (a / c) = a / min b c := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroup α
    a b c : α
    ⊢ Eq (Max.max (HDiv.hDiv a b) (HDiv.hDiv a c)) (HDiv.hDiv a (Min.min b c))
  -/
  simp only [div_eq_mul_inv, max_mul_mul_left, max_inv_inv']
  /-
    🎉 no goals
  -/


theorem max_sub_max_le_max (a b c d : α) : max a b - max c d ≤ max (a - c) (b - d) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c d : α
    ⊢ LE.le (HSub.hSub (Max.max a b) (Max.max c d)) (Max.max (HSub.hSub a c) (HSub …
  -/
  simp only [sub_le_iff_le_add, max_le_iff]; constructor
  · calc
    a = a - c + c := (sub_add_cancel a c).symm
    _ ≤ max (a - c) (b - d) + max c d := add_le_add (le_max_left _ _) (le_max_left _ _)
  · calc
    b = b - d + d := (sub_add_cancel b d).symm
    _ ≤ max (a - c) (b - d) + max c d := add_le_add (le_max_right _ _) (le_max_right _ _)


theorem abs_max_sub_max_le_max (a b c d : α) : |max a b - max c d| ≤ max |a - c| |b - d| := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    a b c d : α
    ⊢ LE.le (abs (HSub.hSub (Max.max a b) (Max.max c d))) (Max.max (abs (HSub.hSub …
  -/
  refine abs_sub_le_iff.2 ⟨?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      a b c d : α
      ⊢ LE.le (HSub.hSub (Max.max a b) (Max.max c d)) (Max.max (abs (HSub.hSub a c)) …
    -/
  · exact (max_sub_max_le_max _ _ _ _).trans (max_le_max (le_abs_self _) (le_abs_self _))
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      a b c d : α
      ⊢ LE.le (HSub.hSub (Max.max c d) (Max.max a b)) (Max.max (abs (HSub.hSub a c)) …
    -/
  · rw [abs_sub_comm a c, abs_sub_comm b d]
    /-
      case refine_2
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      a b c d : α
      ⊢ LE.le (HSub.hSub (Max.max c d) (Max.max a b)) (Max.max (abs (HSub.hSub c a)) …
    -/
    exact (max_sub_max_le_max _ _ _ _).trans (max_le_max (le_abs_self _) (le_abs_self _))
    /-
      🎉 no goals
    -/


theorem abs_min_sub_min_le_max (a b c d : α) : |min a b - min c d| ≤ max |a - c| |b - d| := by
  simpa only [max_neg_neg, neg_sub_neg, abs_sub_comm] using
    abs_max_sub_max_le_max (-a) (-b) (-c) (-d)


theorem abs_max_sub_max_le_abs (a b c : α) : |max a c - max b c| ≤ |a - b| := by
  simpa only [sub_self, abs_zero, max_eq_left (abs_nonneg (a - b))]
    using abs_max_sub_max_le_max a c b c


