instance subsemiring (R : Type u) [Semiring R] (p : ℕ) [CharP R p] (S : Subsemiring R) :
    CharP S p :=
  ⟨fun x =>
    Iff.symm <|
      (CharP.cast_eq_zero_iff R p x).symm.trans
                                                        /-
                                                          R : Type u
                                                          inst✝¹ : Semiring R
                                                          p : Nat
                                                          inst✝ : CharP R p
                                                          S : Subsemiring R
                                                          x : Nat
                                                          h : Eq (↑x) 0
                                                          ⊢ Eq (S.subtype ↑x) 0
                                                        -/
        ⟨fun h => Subtype.eq <| show S.subtype x = 0 by rw [map_natCast, h], fun h =>
                                                        /-
                                                          🎉 no goals
                                                        -/
                                       /-
                                         R : Type u
                                         inst✝¹ : Semiring R
                                         p : Nat
                                         inst✝ : CharP R p
                                         S : Subsemiring R
                                         x : Nat
                                         h : Eq (↑x) 0
                                         ⊢ Eq (S.subtype ↑x) 0
                                       -/
          map_natCast S.subtype x ▸ by rw [h, RingHom.map_zero]⟩⟩
                                       /-
                                         🎉 no goals
                                       -/


instance subring (R : Type u) [Ring R] (p : ℕ) [CharP R p] (S : Subring R) : CharP S p :=
  ⟨fun x =>
    Iff.symm <|
      (CharP.cast_eq_zero_iff R p x).symm.trans
                                                        /-
                                                          R : Type u
                                                          inst✝¹ : Ring R
                                                          p : Nat
                                                          inst✝ : CharP R p
                                                          S : Subring R
                                                          x : Nat
                                                          h : Eq (↑x) 0
                                                          ⊢ Eq (S.subtype ↑x) 0
                                                        -/
        ⟨fun h => Subtype.eq <| show S.subtype x = 0 by rw [map_natCast, h], fun h =>
                                                        /-
                                                          🎉 no goals
                                                        -/
                                       /-
                                         R : Type u
                                         inst✝¹ : Ring R
                                         p : Nat
                                         inst✝ : CharP R p
                                         S : Subring R
                                         x : Nat
                                         h : Eq (↑x) 0
                                         ⊢ Eq (S.subtype ↑x) 0
                                       -/
          map_natCast S.subtype x ▸ by rw [h, RingHom.map_zero]⟩⟩
                                       /-
                                         🎉 no goals
                                       -/


instance subring' (R : Type u) [CommRing R] (p : ℕ) [CharP R p] (S : Subring R) : CharP S p :=
  CharP.subring R p S


/-- The characteristic of a division ring is equal to the characteristic
  of its center-/
theorem charP_center_iff {R : Type u} [Ring R] {p : ℕ} :
    CharP (Subring.center R) p ↔ CharP R p :=
  (algebraMap (Subring.center R) R).charP_iff Subtype.val_injective p


theorem expChar_center_iff {R : Type u} [Ring R] {p : ℕ} :
    ExpChar (Subring.center R) p ↔ ExpChar R p :=
  (algebraMap (Subring.center R) R).expChar_iff Subtype.val_injective p


