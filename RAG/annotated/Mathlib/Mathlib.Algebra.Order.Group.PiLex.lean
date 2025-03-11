@[to_additive]
instance orderedCancelCommMonoid [∀ i, OrderedCancelCommMonoid (α i)] :
    OrderedCancelCommMonoid (Lex (∀ i, α i)) where
  mul_le_mul_left _ _ hxy z :=
    hxy.elim (fun hxyz => hxyz ▸ le_rfl) fun ⟨i, hi⟩ =>
      Or.inr ⟨i, fun j hji => congr_arg (z j * ·) (hi.1 j hji), mul_lt_mul_left' hi.2 _⟩
  le_of_mul_le_mul_left _ _ _ hxyz :=
    hxyz.elim (fun h => (mul_left_cancel h).le) fun ⟨i, hi⟩ =>
      Or.inr ⟨i, fun j hj => (mul_left_cancel <| hi.1 j hj), lt_of_mul_lt_mul_left' hi.2⟩


@[to_additive]
instance orderedCommGroup [∀ i, OrderedCommGroup (α i)] : OrderedCommGroup (Lex (∀ i, α i)) where
  mul_le_mul_left _ _ := mul_le_mul_left'


@[to_additive]
noncomputable instance linearOrderedCancelCommMonoid [WellFoundedLT ι]
    [∀ i, LinearOrderedCancelCommMonoid (α i)] :
    LinearOrderedCancelCommMonoid (Lex (∀ i, α i)) where
  __ : LinearOrder (Lex (∀ i, α i)) := inferInstance
  __ : OrderedCancelCommMonoid (Lex (∀ i, α i)) := inferInstance


@[to_additive]
noncomputable instance linearOrderedCommGroup [WellFoundedLT ι]
    [∀ i, LinearOrderedCommGroup (α i)] :
    LinearOrderedCommGroup (Lex (∀ i, α i)) where
  __ : LinearOrder (Lex (∀ i, α i)) := inferInstance
  mul_le_mul_left _ _ := mul_le_mul_left'


