instance [OrderedSemiring α] [OrderedSemiring β] : OrderedSemiring (α × β) :=
  { inferInstanceAs (Semiring (α × β)), inferInstanceAs (OrderedAddCommMonoid (α × β)) with
    zero_le_one := ⟨zero_le_one, zero_le_one⟩
    mul_le_mul_of_nonneg_left := fun _ _ _ hab hc =>
      ⟨mul_le_mul_of_nonneg_left hab.1 hc.1, mul_le_mul_of_nonneg_left hab.2 hc.2⟩
    mul_le_mul_of_nonneg_right := fun _ _ _ hab hc =>
      ⟨mul_le_mul_of_nonneg_right hab.1 hc.1, mul_le_mul_of_nonneg_right hab.2 hc.2⟩ }


instance [OrderedCommSemiring α] [OrderedCommSemiring β] : OrderedCommSemiring (α × β) :=
  { inferInstanceAs (OrderedSemiring (α × β)), inferInstanceAs (CommSemiring (α × β)) with }

-- Porting note: compile fails with `inferInstanceAs (OrderedSemiring (α × β))`

instance [OrderedRing α] [OrderedRing β] : OrderedRing (α × β) :=
  { inferInstanceAs (Ring (α × β)), inferInstanceAs (OrderedAddCommGroup (α × β)) with
    zero_le_one := ⟨zero_le_one, zero_le_one⟩
    mul_nonneg := fun _ _ ha hb => ⟨mul_nonneg ha.1 hb.1, mul_nonneg ha.2 hb.2⟩ }


instance [OrderedCommRing α] [OrderedCommRing β] : OrderedCommRing (α × β) :=
  { inferInstanceAs (OrderedRing (α × β)), inferInstanceAs (CommRing (α × β)) with }

