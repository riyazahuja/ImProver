@[to_additive]
instance instOrderedCommMonoid [OrderedCommMonoid β] : OrderedCommMonoid (Germ l β) where
  mul_le_mul_left f g := inductionOn₂ f g fun _ _ H h ↦ inductionOn h fun _ ↦ H.mono
    fun _ H ↦ mul_le_mul_left' H _


@[to_additive]
instance instOrderedCancelCommMonoid [OrderedCancelCommMonoid β] :
    OrderedCancelCommMonoid (Germ l β) where
  le_of_mul_le_mul_left f g h := inductionOn₃ f g h fun _ _ _ H ↦ H.mono
    fun _ ↦ le_of_mul_le_mul_left'


@[to_additive]
instance instCanonicallyOrderedCommMonoid [CanonicallyOrderedCommMonoid β] :
    CanonicallyOrderedCommMonoid (Germ l β) where
  __ := instExistsMulOfLE
  le_self_mul x y := inductionOn₂ x y fun _ _ ↦ Eventually.of_forall fun _ ↦ le_self_mul


