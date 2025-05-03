deriving instance CanonicallyOrderedCommSemiring for NNRat

deriving instance CanonicallyLinearOrderedAddCommMonoid for NNRat

-- TODO: `deriving instance OrderedSub for NNRat` doesn't work yet, so we add the instance manually

instance NNRat.instOrderedSub : OrderedSub ℚ≥0 := Nonneg.orderedSub

