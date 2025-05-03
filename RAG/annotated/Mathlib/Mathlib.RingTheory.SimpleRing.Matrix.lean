instance matrix [IsSimpleRing A] : IsSimpleRing (Matrix ι ι A) where
  simple := TwoSidedIdeal.orderIsoMatricesOver (Nonempty.some ‹_›) (Nonempty.some ‹_›)
    |>.symm.isSimpleOrder


