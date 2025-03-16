@[to_additive]
theorem le_mul_ciInf [MulLeftMono α] {a : α} {g : α} {h : ι → α}
    (H : ∀ j, a ≤ g * h j) : a ≤ g * iInf h :=
  inv_mul_le_iff_le_mul.mp <| le_ciInf fun _ => inv_mul_le_iff_le_mul.mpr <| H _


@[to_additive]
theorem mul_ciSup_le [MulLeftMono α] {a : α} {g : α} {h : ι → α}
    (H : ∀ j, g * h j ≤ a) : g * iSup h ≤ a :=
  le_mul_ciInf (α := αᵒᵈ) H


@[to_additive]
theorem le_ciInf_mul [MulRightMono α] {a : α} {g : ι → α}
    {h : α} (H : ∀ i, a ≤ g i * h) : a ≤ iInf g * h :=
  mul_inv_le_iff_le_mul.mp <| le_ciInf fun _ => mul_inv_le_iff_le_mul.mpr <| H _


@[to_additive]
theorem ciSup_mul_le [MulRightMono α] {a : α} {g : ι → α}
    {h : α} (H : ∀ i, g i * h ≤ a) : iSup g * h ≤ a :=
  le_ciInf_mul (α := αᵒᵈ) H


@[to_additive]
theorem le_ciInf_mul_ciInf [MulLeftMono α] [MulRightMono α] {a : α} {g : ι → α} {h : ι' → α}
    (H : ∀ i j, a ≤ g i * h j) : a ≤ iInf g * iInf h :=
  le_ciInf_mul fun _ => le_mul_ciInf <| H _


@[to_additive]
theorem ciSup_mul_ciSup_le [MulLeftMono α] [MulRightMono α] {a : α} {g : ι → α} {h : ι' → α}
    (H : ∀ i j, g i * h j ≤ a) : iSup g * iSup h ≤ a :=
  ciSup_mul_le fun _ => mul_ciSup_le <| H _


