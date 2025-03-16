theorem algebraMap_monotone : Monotone (algebraMap R A) := fun a b h => by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : OrderedCommRing R
    inst✝² : OrderedRing A
    inst✝¹ : Algebra R A
    inst✝ : OrderedSMul R A
    a b : R
    h : LE.le a b
    ⊢ LE.le ((algebraMap R A) a) ((algebraMap R A) b)
  -/
  rw [Algebra.algebraMap_eq_smul_one, Algebra.algebraMap_eq_smul_one, ← sub_nonneg, ← sub_smul]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : OrderedCommRing R
    inst✝² : OrderedRing A
    inst✝¹ : Algebra R A
    inst✝ : OrderedSMul R A
    a b : R
    h : LE.le a b
    ⊢ LE.le 0 (HSMul.hSMul (HSub.hSub b a) 1)
  -/
  trans (b - a) • (0 : A)
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : OrderedCommRing R
      inst✝² : OrderedRing A
      inst✝¹ : Algebra R A
      inst✝ : OrderedSMul R A
      a b : R
      h : LE.le a b
      ⊢ LE.le 0 (HSMul.hSMul (HSub.hSub b a) 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : OrderedCommRing R
      inst✝² : OrderedRing A
      inst✝¹ : Algebra R A
      inst✝ : OrderedSMul R A
      a b : R
      h : LE.le a b
      ⊢ LE.le (HSMul.hSMul (HSub.hSub b a) 0) (HSMul.hSMul (HSub.hSub b a) 1)
    -/
  · exact smul_le_smul_of_nonneg_left zero_le_one (sub_nonneg.mpr h)
    /-
      🎉 no goals
    -/


