@[to_additive Icc_add_Icc_subset]
theorem Icc_mul_Icc_subset' [LocallyFiniteOrder α] (a b c d : α) :
    Icc a b * Icc c d ⊆ Icc (a * c) (b * d) :=
                             /-
                               α : Type u_1
                               inst✝⁵ : Mul α
                               inst✝⁴ : Preorder α
                               inst✝³ : DecidableEq α
                               inst✝² : MulLeftMono α
                               inst✝¹ : MulRightMono α
                               inst✝ : LocallyFiniteOrder α
                               a b c d : α
                               ⊢ HasSubset.Subset ↑(HMul.hMul (Finset.Icc a b) (Finset.Icc c d)) ↑(Finset.Icc …
                             -/
  Finset.coe_subset.mp <| by simpa using Set.Icc_mul_Icc_subset' _ _ _ _
                             /-
                               🎉 no goals
                             -/


@[to_additive Iic_add_Iic_subset]
theorem Iic_mul_Iic_subset' [LocallyFiniteOrderBot α] (a b : α) : Iic a * Iic b ⊆ Iic (a * b) :=
                             /-
                               α : Type u_1
                               inst✝⁵ : Mul α
                               inst✝⁴ : Preorder α
                               inst✝³ : DecidableEq α
                               inst✝² : MulLeftMono α
                               inst✝¹ : MulRightMono α
                               inst✝ : LocallyFiniteOrderBot α
                               a b : α
                               ⊢ HasSubset.Subset ↑(HMul.hMul (Finset.Iic a) (Finset.Iic b)) ↑(Finset.Iic (HM …
                             -/
  Finset.coe_subset.mp <| by simpa using Set.Iic_mul_Iic_subset' _ _
                             /-
                               🎉 no goals
                             -/


@[to_additive Ici_add_Ici_subset]
theorem Ici_mul_Ici_subset' [LocallyFiniteOrderTop α] (a b : α) : Ici a * Ici b ⊆ Ici (a * b) :=
                             /-
                               α : Type u_1
                               inst✝⁵ : Mul α
                               inst✝⁴ : Preorder α
                               inst✝³ : DecidableEq α
                               inst✝² : MulLeftMono α
                               inst✝¹ : MulRightMono α
                               inst✝ : LocallyFiniteOrderTop α
                               a b : α
                               ⊢ HasSubset.Subset ↑(HMul.hMul (Finset.Ici a) (Finset.Ici b)) ↑(Finset.Ici (HM …
                             -/
  Finset.coe_subset.mp <| by simpa using Set.Ici_mul_Ici_subset' _ _
                             /-
                               🎉 no goals
                             -/


@[to_additive Icc_add_Ico_subset]
theorem Icc_mul_Ico_subset' [LocallyFiniteOrder α] (a b c d : α) :
    Icc a b * Ico c d ⊆ Ico (a * c) (b * d) :=
                             /-
                               α : Type u_1
                               inst✝⁵ : Mul α
                               inst✝⁴ : PartialOrder α
                               inst✝³ : DecidableEq α
                               inst✝² : MulLeftStrictMono α
                               inst✝¹ : MulRightStrictMono α
                               inst✝ : LocallyFiniteOrder α
                               a b c d : α
                               ⊢ HasSubset.Subset ↑(HMul.hMul (Finset.Icc a b) (Finset.Ico c d)) ↑(Finset.Ico …
                             -/
  Finset.coe_subset.mp <| by simpa using Set.Icc_mul_Ico_subset' _ _ _ _
                             /-
                               🎉 no goals
                             -/


@[to_additive Ico_add_Icc_subset]
theorem Ico_mul_Icc_subset' [LocallyFiniteOrder α] (a b c d : α) :
    Ico a b * Icc c d ⊆ Ico (a * c) (b * d) :=
                             /-
                               α : Type u_1
                               inst✝⁵ : Mul α
                               inst✝⁴ : PartialOrder α
                               inst✝³ : DecidableEq α
                               inst✝² : MulLeftStrictMono α
                               inst✝¹ : MulRightStrictMono α
                               inst✝ : LocallyFiniteOrder α
                               a b c d : α
                               ⊢ HasSubset.Subset ↑(HMul.hMul (Finset.Ico a b) (Finset.Icc c d)) ↑(Finset.Ico …
                             -/
  Finset.coe_subset.mp <| by simpa using Set.Ico_mul_Icc_subset' _ _ _ _
                             /-
                               🎉 no goals
                             -/


@[to_additive Ioc_add_Ico_subset]
theorem Ioc_mul_Ico_subset' [LocallyFiniteOrder α] (a b c d : α) :
    Ioc a b * Ico c d ⊆ Ioo (a * c) (b * d) :=
                             /-
                               α : Type u_1
                               inst✝⁵ : Mul α
                               inst✝⁴ : PartialOrder α
                               inst✝³ : DecidableEq α
                               inst✝² : MulLeftStrictMono α
                               inst✝¹ : MulRightStrictMono α
                               inst✝ : LocallyFiniteOrder α
                               a b c d : α
                               ⊢ HasSubset.Subset ↑(HMul.hMul (Finset.Ioc a b) (Finset.Ico c d)) ↑(Finset.Ioo …
                             -/
  Finset.coe_subset.mp <| by simpa using Set.Ioc_mul_Ico_subset' _ _ _ _
                             /-
                               🎉 no goals
                             -/


@[to_additive Ico_add_Ioc_subset]
theorem Ico_mul_Ioc_subset' [LocallyFiniteOrder α] (a b c d : α) :
    Ico a b * Ioc c d ⊆ Ioo (a * c) (b * d) :=
                             /-
                               α : Type u_1
                               inst✝⁵ : Mul α
                               inst✝⁴ : PartialOrder α
                               inst✝³ : DecidableEq α
                               inst✝² : MulLeftStrictMono α
                               inst✝¹ : MulRightStrictMono α
                               inst✝ : LocallyFiniteOrder α
                               a b c d : α
                               ⊢ HasSubset.Subset ↑(HMul.hMul (Finset.Ico a b) (Finset.Ioc c d)) ↑(Finset.Ioo …
                             -/
  Finset.coe_subset.mp <| by simpa using Set.Ico_mul_Ioc_subset' _ _ _ _
                             /-
                               🎉 no goals
                             -/


@[to_additive Iic_add_Iio_subset]
theorem Iic_mul_Iio_subset' [LocallyFiniteOrderBot α] (a b : α) : Iic a * Iio b ⊆ Iio (a * b) :=
                             /-
                               α : Type u_1
                               inst✝⁵ : Mul α
                               inst✝⁴ : PartialOrder α
                               inst✝³ : DecidableEq α
                               inst✝² : MulLeftStrictMono α
                               inst✝¹ : MulRightStrictMono α
                               inst✝ : LocallyFiniteOrderBot α
                               a b : α
                               ⊢ HasSubset.Subset ↑(HMul.hMul (Finset.Iic a) (Finset.Iio b)) ↑(Finset.Iio (HM …
                             -/
  Finset.coe_subset.mp <| by simpa using Set.Iic_mul_Iio_subset' _ _
                             /-
                               🎉 no goals
                             -/


@[to_additive Iio_add_Iic_subset]
theorem Iio_mul_Iic_subset' [LocallyFiniteOrderBot α] (a b : α) : Iio a * Iic b ⊆ Iio (a * b) :=
                             /-
                               α : Type u_1
                               inst✝⁵ : Mul α
                               inst✝⁴ : PartialOrder α
                               inst✝³ : DecidableEq α
                               inst✝² : MulLeftStrictMono α
                               inst✝¹ : MulRightStrictMono α
                               inst✝ : LocallyFiniteOrderBot α
                               a b : α
                               ⊢ HasSubset.Subset ↑(HMul.hMul (Finset.Iio a) (Finset.Iic b)) ↑(Finset.Iio (HM …
                             -/
  Finset.coe_subset.mp <| by simpa using Set.Iio_mul_Iic_subset' _ _
                             /-
                               🎉 no goals
                             -/


@[to_additive Ioi_add_Ici_subset]
theorem Ioi_mul_Ici_subset' [LocallyFiniteOrderTop α] (a b : α) : Ioi a * Ici b ⊆ Ioi (a * b) :=
                             /-
                               α : Type u_1
                               inst✝⁵ : Mul α
                               inst✝⁴ : PartialOrder α
                               inst✝³ : DecidableEq α
                               inst✝² : MulLeftStrictMono α
                               inst✝¹ : MulRightStrictMono α
                               inst✝ : LocallyFiniteOrderTop α
                               a b : α
                               ⊢ HasSubset.Subset ↑(HMul.hMul (Finset.Ioi a) (Finset.Ici b)) ↑(Finset.Ioi (HM …
                             -/
  Finset.coe_subset.mp <| by simpa using Set.Ioi_mul_Ici_subset' _ _
                             /-
                               🎉 no goals
                             -/


@[to_additive Ici_add_Ioi_subset]
theorem Ici_mul_Ioi_subset' [LocallyFiniteOrderTop α] (a b : α) : Ici a * Ioi b ⊆ Ioi (a * b) :=
                             /-
                               α : Type u_1
                               inst✝⁵ : Mul α
                               inst✝⁴ : PartialOrder α
                               inst✝³ : DecidableEq α
                               inst✝² : MulLeftStrictMono α
                               inst✝¹ : MulRightStrictMono α
                               inst✝ : LocallyFiniteOrderTop α
                               a b : α
                               ⊢ HasSubset.Subset ↑(HMul.hMul (Finset.Ici a) (Finset.Ioi b)) ↑(Finset.Ioi (HM …
                             -/
  Finset.coe_subset.mp <| by simpa using Set.Ici_mul_Ioi_subset' _ _
                             /-
                               🎉 no goals
                             -/


