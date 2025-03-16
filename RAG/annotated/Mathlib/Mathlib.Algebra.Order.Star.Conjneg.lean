@[simp] lemma conjneg_nonneg : 0 ≤ conjneg f ↔ 0 ≤ f :=
                                    /-
                                      G : Type u_1
                                      R : Type u_2
                                      inst✝³ : AddGroup G
                                      inst✝² : OrderedCommSemiring R
                                      inst✝¹ : StarRing R
                                      inst✝ : StarOrderedRing R
                                      f : G → R
                                      ⊢ ∀ (b : G), Iff (LE.le (0 ((Equiv.symm (Equiv.neg G)) b)) (conjneg f ((Equiv. …
                                    -/
  (Equiv.neg _).forall_congr' <| by simp [starRingEnd_apply]
                                    /-
                                      🎉 no goals
                                    -/


@[simp] lemma conjneg_pos : 0 < conjneg f ↔ 0 < f := by
  /-
    G : Type u_1
    R : Type u_2
    inst✝³ : AddGroup G
    inst✝² : OrderedCommSemiring R
    inst✝¹ : StarRing R
    inst✝ : StarOrderedRing R
    f : G → R
    ⊢ Iff (LT.lt 0 (conjneg f)) (LT.lt 0 f)
  -/
  simp_rw [lt_iff_le_and_ne, ne_comm, conjneg_nonneg, conjneg_ne_zero]
  /-
    🎉 no goals
  -/


@[simp] lemma conjneg_nonpos : conjneg f ≤ 0 ↔ f ≤ 0 := by
  /-
    G : Type u_1
    R : Type u_2
    inst✝³ : AddGroup G
    inst✝² : OrderedCommRing R
    inst✝¹ : StarRing R
    inst✝ : StarOrderedRing R
    f : G → R
    ⊢ Iff (LE.le (conjneg f) 0) (LE.le f 0)
  -/
  simp_rw [← neg_nonneg, ← conjneg_neg, conjneg_nonneg]
  /-
    🎉 no goals
  -/


@[simp] lemma conjneg_neg' : conjneg f < 0 ↔ f < 0 := by
  /-
    G : Type u_1
    R : Type u_2
    inst✝³ : AddGroup G
    inst✝² : OrderedCommRing R
    inst✝¹ : StarRing R
    inst✝ : StarOrderedRing R
    f : G → R
    ⊢ Iff (LT.lt (conjneg f) 0) (LT.lt f 0)
  -/
  simp_rw [← neg_pos, ← conjneg_neg, conjneg_pos]
  /-
    🎉 no goals
  -/


