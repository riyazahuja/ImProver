instance Multiplicative.orderedCommGroup [OrderedAddCommGroup α] :
    OrderedCommGroup (Multiplicative α) :=
  { Multiplicative.commGroup, Multiplicative.orderedCommMonoid with }


instance Additive.orderedAddCommGroup [OrderedCommGroup α] :
    OrderedAddCommGroup (Additive α) :=
  { Additive.addCommGroup, Additive.orderedAddCommMonoid with }


instance Multiplicative.linearOrderedCommGroup [LinearOrderedAddCommGroup α] :
    LinearOrderedCommGroup (Multiplicative α) :=
  { Multiplicative.linearOrder, Multiplicative.orderedCommGroup with }


instance Additive.linearOrderedAddCommGroup [LinearOrderedCommGroup α] :
    LinearOrderedAddCommGroup (Additive α) :=
  { Additive.linearOrder, Additive.orderedAddCommGroup with }

