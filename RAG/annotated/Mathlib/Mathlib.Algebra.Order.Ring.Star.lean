private lemma mul_le_mul_of_nonneg_left {R : Type*} [CommSemiring R] [PartialOrder R]
    [StarRing R] [StarOrderedRing R] {a b c : R} (hab : a ≤ b) (hc : 0 ≤ c) : c * a ≤ c * b := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    inst✝² : PartialOrder R
    inst✝¹ : StarRing R
    inst✝ : StarOrderedRing R
    a b c : R
    hab : LE.le a b
    hc : LE.le 0 c
    ⊢ LE.le (HMul.hMul c a) (HMul.hMul c b)
  -/
  rw [StarOrderedRing.nonneg_iff] at hc
  induction hc using AddSubmonoid.closure_induction with
  | mem _ h =>
    obtain ⟨x, rfl⟩ := h
    simp_rw [mul_assoc, mul_comm x, ← mul_assoc]
    exact conjugate_le_conjugate hab x
  | one => simp
  | mul x hx y hy =>
    simp only [← nonneg_iff, add_mul] at hx hy ⊢
    apply add_le_add <;> aesop


/-- A commutative star-ordered semiring is an ordered semiring.

This is not registered as an instance because it introduces a type class loop between `CommSemiring`
and `OrderedCommSemiring`, and it seem loops still cause issues sometimes.

See note [reducible non-instances]. -/
abbrev toOrderedCommSemiring (R : Type*) [CommSemiring R] [PartialOrder R]
    [StarRing R] [StarOrderedRing R] : OrderedCommSemiring R where
  add_le_add_left _ _ := add_le_add_left
  zero_le_one := zero_le_one
  mul_comm := mul_comm
  mul_le_mul_of_nonneg_left _ _ _ := mul_le_mul_of_nonneg_left
                                         /-
                                           R : Type u_1
                                           inst✝³ : CommSemiring R
                                           inst✝² : PartialOrder R
                                           inst✝¹ : StarRing R
                                           inst✝ : StarOrderedRing R
                                           a b c : R
                                           ⊢ LE.le a b → LE.le 0 c → LE.le (HMul.hMul a c) (HMul.hMul b c)
                                         -/
  mul_le_mul_of_nonneg_right a b c := by simpa only [mul_comm _ c] using mul_le_mul_of_nonneg_left
                                         /-
                                           🎉 no goals
                                         -/


/-- A commutative star-ordered ring is an ordered ring.

This is not registered as an instance because it introduces a type class loop between `CommSemiring`
and `OrderedCommSemiring`, and it seem loops still cause issues sometimes.

See note [reducible non-instances]. -/
abbrev toOrderedCommRing (R : Type*) [CommRing R] [PartialOrder R]
    [StarRing R] [StarOrderedRing R] : OrderedCommRing R where
  add_le_add_left _ _ := add_le_add_left
  zero_le_one := zero_le_one
  mul_comm := mul_comm
  mul_nonneg _ _ := let _ := toOrderedCommSemiring R; mul_nonneg


