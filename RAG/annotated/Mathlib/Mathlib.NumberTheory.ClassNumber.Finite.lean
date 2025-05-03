/-- If `b` is an `R`-basis of `S` of cardinality `n`, then `normBound abv b` is an integer
such that for every `R`-integral element `a : S` with coordinates `≤ y`,
we have algebra.norm a ≤ norm_bound abv b * y ^ n`. (See also `norm_le` and `norm_lt`). -/
noncomputable def normBound : ℤ :=
  let n := Fintype.card ι
  let i : ι := Nonempty.some bS.index_nonempty
  let m : ℤ :=
    Finset.max'
      (Finset.univ.image fun ijk : ι × ι × ι =>
        abv (Algebra.leftMulMatrix bS (bS ijk.1) ijk.2.1 ijk.2.2))
      ⟨_, Finset.mem_image.mpr ⟨⟨i, i, i⟩, Finset.mem_univ _, rfl⟩⟩
  Nat.factorial n • (n • m) ^ n


theorem normBound_pos : 0 < normBound abv bS := by
  obtain ⟨i, j, k, hijk⟩ : ∃ i j k, Algebra.leftMulMatrix bS (bS i) j k ≠ 0 := by
    by_contra! h
    obtain ⟨i⟩ := bS.index_nonempty
    apply bS.ne_zero i
    apply
      (injective_iff_map_eq_zero (Algebra.leftMulMatrix bS)).mp (Algebra.leftMulMatrix_injective bS)
    ext j k
    simp [h, DMatrix.zero_apply]
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁵ : EuclideanDomain R
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    inst✝² : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    bS : Basis ι R S
    i j k : ι
    hijk : Ne ((Algebra.leftMulMatrix bS) (bS i) j k) 0
    ⊢ LT.lt 0 (ClassGroup.normBound abv bS)
  -/
  simp only [normBound, Algebra.smul_def, eq_natCast]
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁵ : EuclideanDomain R
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    inst✝² : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    bS : Basis ι R S
    i j k : ι
    hijk : Ne ((Algebra.leftMulMatrix bS) (bS i) j k) 0
    ⊢ LT.lt 0 (HMul.hMul (↑(Fintype.card ι).factorial) (HPow.hPow (HMul.hMul (↑(Fi …
  -/
  apply mul_pos (Int.natCast_pos.mpr (Nat.factorial_pos _))
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁵ : EuclideanDomain R
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    inst✝² : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    bS : Basis ι R S
    i j k : ι
    hijk : Ne ((Algebra.leftMulMatrix bS) (bS i) j k) 0
    ⊢ LT.lt 0 (HPow.hPow (HMul.hMul (↑(Fintype.card ι)) ((Finset.image (fun ijk => …
  -/
  refine pow_pos (mul_pos (Int.natCast_pos.mpr (Fintype.card_pos_iff.mpr ⟨i⟩)) ?_) _
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁵ : EuclideanDomain R
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    inst✝² : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    bS : Basis ι R S
    i j k : ι
    hijk : Ne ((Algebra.leftMulMatrix bS) (bS i) j k) 0
    ⊢ LT.lt 0 ((Finset.image (fun ijk => abv ((Algebra.leftMulMatrix bS) (bS ijk.1 …
  -/
  refine lt_of_lt_of_le (abv.pos hijk) (Finset.le_max' _ _ ?_)
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁵ : EuclideanDomain R
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    inst✝² : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    bS : Basis ι R S
    i j k : ι
    hijk : Ne ((Algebra.leftMulMatrix bS) (bS i) j k) 0
    ⊢ Membership.mem (Finset.image (fun ijk => abv ((Algebra.leftMulMatrix bS) (bS …
  -/
  exact Finset.mem_image.mpr ⟨⟨i, j, k⟩, Finset.mem_univ _, rfl⟩
  /-
    🎉 no goals
  -/


/-- If the `R`-integral element `a : S` has coordinates `≤ y` with respect to some basis `b`,
its norm is less than `normBound abv b * y ^ dim S`. -/
theorem norm_le (a : S) {y : ℤ} (hy : ∀ k, abv (bS.repr a k) ≤ y) :
    abv (Algebra.norm R a) ≤ normBound abv bS * y ^ Fintype.card ι := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁵ : EuclideanDomain R
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    inst✝² : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    bS : Basis ι R S
    a : S
    y : Int
    hy : ∀ (k : ι), LE.le (abv ((bS.repr a) k)) y
    ⊢ LE.le (abv ((Algebra.norm R) a)) (HMul.hMul (ClassGroup.normBound abv bS) (H …
  -/
  conv_lhs => rw [← bS.sum_repr a]
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁵ : EuclideanDomain R
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    inst✝² : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    bS : Basis ι R S
    a : S
    y : Int
    hy : ∀ (k : ι), LE.le (abv ((bS.repr a) k)) y
    ⊢ LE.le (abv ((Algebra.norm R) (Finset.univ.sum fun i => HSMul.hSMul ((bS.repr …
  -/
  rw [Algebra.norm_apply, ← LinearMap.det_toMatrix bS]
  simp only [Algebra.norm_apply, map_sum, map_smul, map_sum, map_smul, Algebra.toMatrix_lmul_eq,
    normBound, smul_mul_assoc, ← mul_pow]
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁵ : EuclideanDomain R
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    inst✝² : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    bS : Basis ι R S
    a : S
    y : Int
    hy : ∀ (k : ι), LE.le (abv ((bS.repr a) k)) y
    ⊢ LE.le (abv (Finset.univ.sum fun x => HSMul.hSMul ((bS.repr a) x) ((LinearMap …
  -/
  convert Matrix.det_sum_smul_le Finset.univ _ hy using 3
    /-
      case h.e'_4.h.e'_6.h.e'_5
      R : Type u_1
      S : Type u_2
      inst✝⁵ : EuclideanDomain R
      inst✝⁴ : CommRing S
      inst✝³ : IsDomain S
      inst✝² : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      bS : Basis ι R S
      a : S
      y : Int
      hy : ∀ (k : ι), LE.le (abv ((bS.repr a) k)) y
      ⊢ Eq (HSMul.hSMul (Fintype.card ι) (HMul.hMul ((Finset.image (fun ijk => abv ( …
    -/
  · rw [Finset.card_univ, smul_mul_assoc, mul_comm]
    /-
      🎉 no goals
    -/
    /-
      case convert_6
      R : Type u_1
      S : Type u_2
      inst✝⁵ : EuclideanDomain R
      inst✝⁴ : CommRing S
      inst✝³ : IsDomain S
      inst✝² : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      bS : Basis ι R S
      a : S
      y : Int
      hy : ∀ (k : ι), LE.le (abv ((bS.repr a) k)) y
      ⊢ ∀ (k i j : ι), LE.le (abv ((LinearMap.toMatrix bS bS) ((Algebra.lmul R S) (b …
    -/
  · intro i j k
    /-
      case convert_6
      R : Type u_1
      S : Type u_2
      inst✝⁵ : EuclideanDomain R
      inst✝⁴ : CommRing S
      inst✝³ : IsDomain S
      inst✝² : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      bS : Basis ι R S
      a : S
      y : Int
      hy : ∀ (k : ι), LE.le (abv ((bS.repr a) k)) y
      i j k : ι
      ⊢ LE.le (abv ((LinearMap.toMatrix bS bS) ((Algebra.lmul R S) (bS i)) j k)) ((F …
    -/
    apply Finset.le_max'
    /-
      case convert_6.H2
      R : Type u_1
      S : Type u_2
      inst✝⁵ : EuclideanDomain R
      inst✝⁴ : CommRing S
      inst✝³ : IsDomain S
      inst✝² : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      bS : Basis ι R S
      a : S
      y : Int
      hy : ∀ (k : ι), LE.le (abv ((bS.repr a) k)) y
      i j k : ι
      ⊢ Membership.mem (Finset.image (fun ijk => abv ((Algebra.leftMulMatrix bS) (bS …
    -/
    exact Finset.mem_image.mpr ⟨⟨i, j, k⟩, Finset.mem_univ _, rfl⟩
    /-
      🎉 no goals
    -/


/-- If the `R`-integral element `a : S` has coordinates `< y` with respect to some basis `b`,
its norm is strictly less than `normBound abv b * y ^ dim S`. -/
theorem norm_lt {T : Type*} [LinearOrderedRing T] (a : S) {y : T}
    (hy : ∀ k, (abv (bS.repr a k) : T) < y) :
    (abv (Algebra.norm R a) : T) < normBound abv bS * y ^ Fintype.card ι := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : EuclideanDomain R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain S
    inst✝³ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    bS : Basis ι R S
    T : Type u_6
    inst✝ : LinearOrderedRing T
    a : S
    y : T
    hy : ∀ (k : ι), LT.lt (↑(abv ((bS.repr a) k))) y
    ⊢ LT.lt (↑(abv ((Algebra.norm R) a))) (HMul.hMul (↑(ClassGroup.normBound abv b …
  -/
  obtain ⟨i⟩ := bS.index_nonempty
  have him : (Finset.univ.image fun k => abv (bS.repr a k)).Nonempty :=
    ⟨_, Finset.mem_image.mpr ⟨i, Finset.mem_univ _, rfl⟩⟩
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝⁶ : EuclideanDomain R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain S
    inst✝³ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    bS : Basis ι R S
    T : Type u_6
    inst✝ : LinearOrderedRing T
    a : S
    y : T
    hy : ∀ (k : ι), LT.lt (↑(abv ((bS.repr a) k))) y
    i : ι
    him : (Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).Nonempty
    ⊢ LT.lt (↑(abv ((Algebra.norm R) a))) (HMul.hMul (↑(ClassGroup.normBound abv b …
  -/
  set y' : ℤ := Finset.max' _ him with y'_def
  have hy' : ∀ k, abv (bS.repr a k) ≤ y' := by
    intro k
    exact @Finset.le_max' ℤ _ _ _ (Finset.mem_image.mpr ⟨k, Finset.mem_univ _, rfl⟩)
  have : (y' : T) < y := by
    rw [y'_def, ←
      Finset.max'_image (show Monotone (_ : ℤ → T) from fun x y h => Int.cast_le.mpr h)]
    apply (Finset.max'_lt_iff _ (him.image _)).mpr
    simp only [Finset.mem_image, exists_prop]
    rintro _ ⟨x, ⟨k, -, rfl⟩, rfl⟩
    exact hy k
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝⁶ : EuclideanDomain R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain S
    inst✝³ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    bS : Basis ι R S
    T : Type u_6
    inst✝ : LinearOrderedRing T
    a : S
    y : T
    hy : ∀ (k : ι), LT.lt (↑(abv ((bS.repr a) k))) y
    i : ι
    him : (Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).Nonempty
    y' : Int := (Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).max' him
    y'_def : Eq y' ((Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).max' …
    hy' : ∀ (k : ι), LE.le (abv ((bS.repr a) k)) y'
    this : LT.lt (↑y') y
    ⊢ LT.lt (↑(abv ((Algebra.norm R) a))) (HMul.hMul (↑(ClassGroup.normBound abv b …
  -/
  have y'_nonneg : 0 ≤ y' := le_trans (abv.nonneg _) (hy' i)
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝⁶ : EuclideanDomain R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain S
    inst✝³ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    bS : Basis ι R S
    T : Type u_6
    inst✝ : LinearOrderedRing T
    a : S
    y : T
    hy : ∀ (k : ι), LT.lt (↑(abv ((bS.repr a) k))) y
    i : ι
    him : (Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).Nonempty
    y' : Int := (Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).max' him
    y'_def : Eq y' ((Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).max' …
    hy' : ∀ (k : ι), LE.le (abv ((bS.repr a) k)) y'
    this : LT.lt (↑y') y
    y'_nonneg : LE.le 0 y'
    ⊢ LT.lt (↑(abv ((Algebra.norm R) a))) (HMul.hMul (↑(ClassGroup.normBound abv b …
  -/
  apply (Int.cast_le.mpr (norm_le abv bS a hy')).trans_lt
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝⁶ : EuclideanDomain R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain S
    inst✝³ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    bS : Basis ι R S
    T : Type u_6
    inst✝ : LinearOrderedRing T
    a : S
    y : T
    hy : ∀ (k : ι), LT.lt (↑(abv ((bS.repr a) k))) y
    i : ι
    him : (Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).Nonempty
    y' : Int := (Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).max' him
    y'_def : Eq y' ((Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).max' …
    hy' : ∀ (k : ι), LE.le (abv ((bS.repr a) k)) y'
    this : LT.lt (↑y') y
    y'_nonneg : LE.le 0 y'
    ⊢ LT.lt (↑(HMul.hMul (ClassGroup.normBound abv bS) (HPow.hPow y' (Fintype.card …
  -/
  simp only [Int.cast_mul, Int.cast_pow]
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝⁶ : EuclideanDomain R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain S
    inst✝³ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    bS : Basis ι R S
    T : Type u_6
    inst✝ : LinearOrderedRing T
    a : S
    y : T
    hy : ∀ (k : ι), LT.lt (↑(abv ((bS.repr a) k))) y
    i : ι
    him : (Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).Nonempty
    y' : Int := (Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).max' him
    y'_def : Eq y' ((Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).max' …
    hy' : ∀ (k : ι), LE.le (abv ((bS.repr a) k)) y'
    this : LT.lt (↑y') y
    y'_nonneg : LE.le 0 y'
    ⊢ LT.lt (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (↑y') (Fintype. …
  -/
  apply mul_lt_mul' le_rfl
    /-
      case intro.h₂
      R : Type u_1
      S : Type u_2
      inst✝⁶ : EuclideanDomain R
      inst✝⁵ : CommRing S
      inst✝⁴ : IsDomain S
      inst✝³ : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝² : DecidableEq ι
      inst✝¹ : Fintype ι
      bS : Basis ι R S
      T : Type u_6
      inst✝ : LinearOrderedRing T
      a : S
      y : T
      hy : ∀ (k : ι), LT.lt (↑(abv ((bS.repr a) k))) y
      i : ι
      him : (Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).Nonempty
      y' : Int := (Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).max' him
      y'_def : Eq y' ((Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).max' …
      hy' : ∀ (k : ι), LE.le (abv ((bS.repr a) k)) y'
      this : LT.lt (↑y') y
      y'_nonneg : LE.le 0 y'
      ⊢ LT.lt (HPow.hPow (↑y') (Fintype.card ι)) (HPow.hPow y (Fintype.card ι))
    -/
  · exact pow_lt_pow_left₀ this (Int.cast_nonneg.mpr y'_nonneg) (@Fintype.card_ne_zero _ _ ⟨i⟩)
    /-
      🎉 no goals
    -/
    /-
      case intro.c0
      R : Type u_1
      S : Type u_2
      inst✝⁶ : EuclideanDomain R
      inst✝⁵ : CommRing S
      inst✝⁴ : IsDomain S
      inst✝³ : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝² : DecidableEq ι
      inst✝¹ : Fintype ι
      bS : Basis ι R S
      T : Type u_6
      inst✝ : LinearOrderedRing T
      a : S
      y : T
      hy : ∀ (k : ι), LT.lt (↑(abv ((bS.repr a) k))) y
      i : ι
      him : (Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).Nonempty
      y' : Int := (Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).max' him
      y'_def : Eq y' ((Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).max' …
      hy' : ∀ (k : ι), LE.le (abv ((bS.repr a) k)) y'
      this : LT.lt (↑y') y
      y'_nonneg : LE.le 0 y'
      ⊢ LE.le 0 (HPow.hPow (↑y') (Fintype.card ι))
    -/
  · exact pow_nonneg (Int.cast_nonneg.mpr y'_nonneg) _
    /-
      🎉 no goals
    -/
    /-
      case intro.b0
      R : Type u_1
      S : Type u_2
      inst✝⁶ : EuclideanDomain R
      inst✝⁵ : CommRing S
      inst✝⁴ : IsDomain S
      inst✝³ : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝² : DecidableEq ι
      inst✝¹ : Fintype ι
      bS : Basis ι R S
      T : Type u_6
      inst✝ : LinearOrderedRing T
      a : S
      y : T
      hy : ∀ (k : ι), LT.lt (↑(abv ((bS.repr a) k))) y
      i : ι
      him : (Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).Nonempty
      y' : Int := (Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).max' him
      y'_def : Eq y' ((Finset.image (fun k => abv ((bS.repr a) k)) Finset.univ).max' …
      hy' : ∀ (k : ι), LE.le (abv ((bS.repr a) k)) y'
      this : LT.lt (↑y') y
      y'_nonneg : LE.le 0 y'
      ⊢ LT.lt 0 ↑(ClassGroup.normBound abv bS)
    -/
  · exact Int.cast_pos.mpr (normBound_pos abv bS)
    /-
      🎉 no goals
    -/



/-- A nonzero ideal has an element of minimal norm. -/
theorem exists_min (I : (Ideal S)⁰) :
    ∃ b ∈ (I : Ideal S),
      b ≠ 0 ∧ ∀ c ∈ (I : Ideal S), abv (Algebra.norm R c) < abv (Algebra.norm R b) → c =
      (0 : S) := by
  obtain ⟨_, ⟨b, b_mem, b_ne_zero, rfl⟩, min⟩ := @Int.exists_least_of_bdd
      (fun a => ∃ b ∈ (I : Ideal S), b ≠ (0 : S) ∧ abv (Algebra.norm R b) = a)
    (by
      use 0
      rintro _ ⟨b, _, _, rfl⟩
      apply abv.nonneg)
    (by
      obtain ⟨b, b_mem, b_ne_zero⟩ := (I : Ideal S).ne_bot_iff.mp (nonZeroDivisors.coe_ne_zero I)
      exact ⟨_, ⟨b, b_mem, b_ne_zero, rfl⟩⟩)
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝³ : EuclideanDomain R
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra R S
    abv : AbsoluteValue R Int
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal S)) x
    b : S
    b_mem : Membership.mem (↑I) b
    b_ne_zero : Ne b 0
    min : ∀ (z : Int), (Exists fun b => And (Membership.mem (↑I) b) (And (Ne b 0)  …
    ⊢ Exists fun b => And (Membership.mem (↑I) b) (And (Ne b 0) (∀ (c : S), Member …
  -/
  refine ⟨b, b_mem, b_ne_zero, ?_⟩
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝³ : EuclideanDomain R
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra R S
    abv : AbsoluteValue R Int
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal S)) x
    b : S
    b_mem : Membership.mem (↑I) b
    b_ne_zero : Ne b 0
    min : ∀ (z : Int), (Exists fun b => And (Membership.mem (↑I) b) (And (Ne b 0)  …
    ⊢ ∀ (c : S), Membership.mem (↑I) c → LT.lt (abv ((Algebra.norm R) c)) (abv ((A …
  -/
  intro c hc lt
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝³ : EuclideanDomain R
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra R S
    abv : AbsoluteValue R Int
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal S)) x
    b : S
    b_mem : Membership.mem (↑I) b
    b_ne_zero : Ne b 0
    min : ∀ (z : Int), (Exists fun b => And (Membership.mem (↑I) b) (And (Ne b 0)  …
    c : S
    hc : Membership.mem (↑I) c
    lt : LT.lt (abv ((Algebra.norm R) c)) (abv ((Algebra.norm R) b))
    ⊢ Eq c 0
  -/
  contrapose! lt with c_ne_zero
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝³ : EuclideanDomain R
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra R S
    abv : AbsoluteValue R Int
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal S)) x
    b : S
    b_mem : Membership.mem (↑I) b
    b_ne_zero : Ne b 0
    min : ∀ (z : Int), (Exists fun b => And (Membership.mem (↑I) b) (And (Ne b 0)  …
    c : S
    hc : Membership.mem (↑I) c
    c_ne_zero : Ne c 0
    ⊢ LE.le (abv ((Algebra.norm R) b)) (abv ((Algebra.norm R) c))
  -/
  exact min _ ⟨c, hc, c_ne_zero, rfl⟩
  /-
    🎉 no goals
  -/


/-- If we have a large enough set of elements in `R^ι`, then there will be a pair
whose remainders are close together. We'll show that all sets of cardinality
at least `cardM bS adm` elements satisfy this condition.

The value of `cardM` is not at all optimal: for specific choices of `R`,
the minimum cardinality can be exponentially smaller.
-/
noncomputable def cardM : ℕ :=
  adm.card (normBound abv bS ^ (-1 / Fintype.card ι : ℝ)) ^ Fintype.card ι


/-- In the following results, we need a large set of distinct elements of `R`. -/
noncomputable def distinctElems : Fin (cardM bS adm).succ ↪ R :=
  Fin.valEmbedding.trans (Infinite.natEmbedding R)


/-- `finsetApprox` is a finite set such that each fractional ideal in the integral closure
contains an element close to `finsetApprox`. -/
noncomputable def finsetApprox : Finset R :=
  (Finset.univ.image fun xy : _ × _ => distinctElems bS adm xy.1 - distinctElems bS adm xy.2).erase
    0


theorem finsetApprox.zero_not_mem : (0 : R) ∉ finsetApprox bS adm :=
  Finset.not_mem_erase _ _


@[simp]
theorem mem_finsetApprox {x : R} :
    x ∈ finsetApprox bS adm ↔ ∃ i j, i ≠ j ∧ distinctElems bS adm i - distinctElems bS adm j =
    x := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    x : R
    ⊢ Iff (Membership.mem (ClassGroup.finsetApprox bS adm) x) (Exists fun i => Exi …
  -/
  simp only [finsetApprox, Finset.mem_erase, Finset.mem_image]
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    x : R
    ⊢ Iff (And (Ne x 0) (Exists fun a => And (Membership.mem Finset.univ a) (Eq (H …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝⁷ : EuclideanDomain R
      inst✝⁶ : CommRing S
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝³ : DecidableEq ι
      inst✝² : Fintype ι
      bS : Basis ι R S
      adm : abv.IsAdmissible
      inst✝¹ : Infinite R
      inst✝ : DecidableEq R
      x : R
      ⊢ And (Ne x 0) (Exists fun a => And (Membership.mem Finset.univ a) (Eq (HSub.h …
    -/
  · rintro ⟨hx, ⟨i, j⟩, _, rfl⟩
    /-
      case mp.intro.intro.mk.intro
      R : Type u_1
      S : Type u_2
      inst✝⁷ : EuclideanDomain R
      inst✝⁶ : CommRing S
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝³ : DecidableEq ι
      inst✝² : Fintype ι
      bS : Basis ι R S
      adm : abv.IsAdmissible
      inst✝¹ : Infinite R
      inst✝ : DecidableEq R
      i j : Fin (ClassGroup.cardM bS adm).succ
      left✝ : Membership.mem Finset.univ { fst := i, snd := j }
      hx : Ne (HSub.hSub ((ClassGroup.distinctElems bS adm) { fst := i, snd := j }.1 …
      ⊢ Exists fun i_1 => Exists fun j_1 => And (Ne i_1 j_1) (Eq (HSub.hSub ((ClassG …
    -/
    refine ⟨i, j, ?_, rfl⟩
    /-
      case mp.intro.intro.mk.intro
      R : Type u_1
      S : Type u_2
      inst✝⁷ : EuclideanDomain R
      inst✝⁶ : CommRing S
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝³ : DecidableEq ι
      inst✝² : Fintype ι
      bS : Basis ι R S
      adm : abv.IsAdmissible
      inst✝¹ : Infinite R
      inst✝ : DecidableEq R
      i j : Fin (ClassGroup.cardM bS adm).succ
      left✝ : Membership.mem Finset.univ { fst := i, snd := j }
      hx : Ne (HSub.hSub ((ClassGroup.distinctElems bS adm) { fst := i, snd := j }.1 …
      ⊢ Ne i j
    -/
    rintro rfl
    /-
      case mp.intro.intro.mk.intro
      R : Type u_1
      S : Type u_2
      inst✝⁷ : EuclideanDomain R
      inst✝⁶ : CommRing S
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝³ : DecidableEq ι
      inst✝² : Fintype ι
      bS : Basis ι R S
      adm : abv.IsAdmissible
      inst✝¹ : Infinite R
      inst✝ : DecidableEq R
      i : Fin (ClassGroup.cardM bS adm).succ
      left✝ : Membership.mem Finset.univ { fst := i, snd := i }
      hx : Ne (HSub.hSub ((ClassGroup.distinctElems bS adm) { fst := i, snd := i }.1 …
      ⊢ False
    -/
    simp at hx
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      S : Type u_2
      inst✝⁷ : EuclideanDomain R
      inst✝⁶ : CommRing S
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝³ : DecidableEq ι
      inst✝² : Fintype ι
      bS : Basis ι R S
      adm : abv.IsAdmissible
      inst✝¹ : Infinite R
      inst✝ : DecidableEq R
      x : R
      ⊢ (Exists fun i => Exists fun j => And (Ne i j) (Eq (HSub.hSub ((ClassGroup.di …
    -/
  · rintro ⟨i, j, hij, rfl⟩
    /-
      case mpr.intro.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝⁷ : EuclideanDomain R
      inst✝⁶ : CommRing S
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝³ : DecidableEq ι
      inst✝² : Fintype ι
      bS : Basis ι R S
      adm : abv.IsAdmissible
      inst✝¹ : Infinite R
      inst✝ : DecidableEq R
      i j : Fin (ClassGroup.cardM bS adm).succ
      hij : Ne i j
      ⊢ And (Ne (HSub.hSub ((ClassGroup.distinctElems bS adm) i) ((ClassGroup.distin …
    -/
    refine ⟨?_, ⟨i, j⟩, Finset.mem_univ _, rfl⟩
    /-
      case mpr.intro.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝⁷ : EuclideanDomain R
      inst✝⁶ : CommRing S
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝³ : DecidableEq ι
      inst✝² : Fintype ι
      bS : Basis ι R S
      adm : abv.IsAdmissible
      inst✝¹ : Infinite R
      inst✝ : DecidableEq R
      i j : Fin (ClassGroup.cardM bS adm).succ
      hij : Ne i j
      ⊢ Ne (HSub.hSub ((ClassGroup.distinctElems bS adm) i) ((ClassGroup.distinctEle …
    -/
    rw [Ne, sub_eq_zero]
    /-
      case mpr.intro.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝⁷ : EuclideanDomain R
      inst✝⁶ : CommRing S
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝³ : DecidableEq ι
      inst✝² : Fintype ι
      bS : Basis ι R S
      adm : abv.IsAdmissible
      inst✝¹ : Infinite R
      inst✝ : DecidableEq R
      i j : Fin (ClassGroup.cardM bS adm).succ
      hij : Ne i j
      ⊢ Not (Eq ((ClassGroup.distinctElems bS adm) i) ((ClassGroup.distinctElems bS  …
    -/
    exact fun h => hij ((distinctElems bS adm).injective h)
    /-
      🎉 no goals
    -/


/-- We can approximate `a / b : L` with `q / r`, where `r` has finitely many options for `L`. -/
theorem exists_mem_finsetApprox (a : S) {b} (hb : b ≠ (0 : R)) :
    ∃ q : S,
      ∃ r ∈ finsetApprox bS adm, abv (Algebra.norm R (r • a - b • q)) <
      abv (Algebra.norm R (algebraMap R S b)) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    a : S
    b : R
    hb : Ne b 0
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  have dim_pos := Fintype.card_pos_iff.mpr bS.index_nonempty
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    a : S
    b : R
    hb : Ne b 0
    dim_pos : LT.lt 0 (Fintype.card ι)
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  set ε : ℝ := normBound abv bS ^ (-1 / Fintype.card ι : ℝ) with ε_eq
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    a : S
    b : R
    hb : Ne b 0
    dim_pos : LT.lt 0 (Fintype.card ι)
    ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
    ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  have hε : 0 < ε := Real.rpow_pos_of_pos (Int.cast_pos.mpr (normBound_pos abv bS)) _
  have ε_le : (normBound abv bS : ℝ) * (abv b • ε) ^ (Fintype.card ι : ℝ)
                ≤ abv b ^ (Fintype.card ι : ℝ) := by
    have := normBound_pos abv bS
    have := abv.nonneg b
    rw [ε_eq, Algebra.smul_def, eq_intCast, mul_rpow, ← rpow_mul, div_mul_cancel₀, rpow_neg_one,
      mul_left_comm, mul_inv_cancel₀, mul_one, rpow_natCast] <;>
      try norm_cast; omega
    · exact Iff.mpr Int.cast_nonneg this
    · linarith
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    a : S
    b : R
    hb : Ne b 0
    dim_pos : LT.lt 0 (Fintype.card ι)
    ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
    ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
    hε : LT.lt 0 ε
    ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  set μ : Fin (cardM bS adm).succ ↪ R := distinctElems bS adm with hμ
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    a : S
    b : R
    hb : Ne b 0
    dim_pos : LT.lt 0 (Fintype.card ι)
    ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
    ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
    hε : LT.lt 0 ε
    ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
    μ : Function.Embedding (Fin (ClassGroup.cardM bS adm).succ) R := ClassGroup.di …
    hμ : Eq μ (ClassGroup.distinctElems bS adm)
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  let s : ι →₀ R := bS.repr a
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    a : S
    b : R
    hb : Ne b 0
    dim_pos : LT.lt 0 (Fintype.card ι)
    ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
    ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
    hε : LT.lt 0 ε
    ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
    μ : Function.Embedding (Fin (ClassGroup.cardM bS adm).succ) R := ClassGroup.di …
    hμ : Eq μ (ClassGroup.distinctElems bS adm)
    s : Finsupp ι R := bS.repr a
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  have s_eq : ∀ i, s i = bS.repr a i := fun i => rfl
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    a : S
    b : R
    hb : Ne b 0
    dim_pos : LT.lt 0 (Fintype.card ι)
    ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
    ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
    hε : LT.lt 0 ε
    ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
    μ : Function.Embedding (Fin (ClassGroup.cardM bS adm).succ) R := ClassGroup.di …
    hμ : Eq μ (ClassGroup.distinctElems bS adm)
    s : Finsupp ι R := bS.repr a
    s_eq : ∀ (i : ι), Eq (s i) ((bS.repr a) i)
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  let qs : Fin (cardM bS adm).succ → ι → R := fun j i => μ j * s i / b
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    a : S
    b : R
    hb : Ne b 0
    dim_pos : LT.lt 0 (Fintype.card ι)
    ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
    ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
    hε : LT.lt 0 ε
    ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
    μ : Function.Embedding (Fin (ClassGroup.cardM bS adm).succ) R := ClassGroup.di …
    hμ : Eq μ (ClassGroup.distinctElems bS adm)
    s : Finsupp ι R := bS.repr a
    s_eq : ∀ (i : ι), Eq (s i) ((bS.repr a) i)
    qs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HDiv.hDiv (HMul. …
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  let rs : Fin (cardM bS adm).succ → ι → R := fun j i => μ j * s i % b
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    a : S
    b : R
    hb : Ne b 0
    dim_pos : LT.lt 0 (Fintype.card ι)
    ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
    ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
    hε : LT.lt 0 ε
    ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
    μ : Function.Embedding (Fin (ClassGroup.cardM bS adm).succ) R := ClassGroup.di …
    hμ : Eq μ (ClassGroup.distinctElems bS adm)
    s : Finsupp ι R := bS.repr a
    s_eq : ∀ (i : ι), Eq (s i) ((bS.repr a) i)
    qs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HDiv.hDiv (HMul. …
    rs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HMod.hMod (HMul. …
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  have r_eq : ∀ j i, rs j i = μ j * s i % b := fun i j => rfl
  have μ_eq : ∀ i j, μ j * s i = b * qs j i + rs j i := by
    intro i j
    rw [r_eq, EuclideanDomain.div_add_mod]
  have μ_mul_a_eq : ∀ j, μ j • a = b • ∑ i, qs j i • bS i + ∑ i, rs j i • bS i := by
    intro j
    rw [← bS.sum_repr a]
    simp only [μ, qs, rs, Finset.smul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
-- Porting note `← hμ, ← r_eq` and the final `← μ_eq` were not needed.
    rw [← hμ, ← r_eq, ← s_eq, ← mul_smul, μ_eq, add_smul, mul_smul, ← μ_eq]
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    a : S
    b : R
    hb : Ne b 0
    dim_pos : LT.lt 0 (Fintype.card ι)
    ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
    ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
    hε : LT.lt 0 ε
    ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
    μ : Function.Embedding (Fin (ClassGroup.cardM bS adm).succ) R := ClassGroup.di …
    hμ : Eq μ (ClassGroup.distinctElems bS adm)
    s : Finsupp ι R := bS.repr a
    s_eq : ∀ (i : ι), Eq (s i) ((bS.repr a) i)
    qs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HDiv.hDiv (HMul. …
    rs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HMod.hMod (HMul. …
    r_eq : ∀ (j : Fin (ClassGroup.cardM bS adm).succ) (i : ι), Eq (rs j i) (HMod.h …
    μ_eq : ∀ (i : ι) (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HMul.hMul (μ j) …
    μ_mul_a_eq : ∀ (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HSMul.hSMul (μ j) …
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  obtain ⟨j, k, j_ne_k, hjk⟩ := adm.exists_approx hε hb fun j i => μ j * s i
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    a : S
    b : R
    hb : Ne b 0
    dim_pos : LT.lt 0 (Fintype.card ι)
    ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
    ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
    hε : LT.lt 0 ε
    ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
    μ : Function.Embedding (Fin (ClassGroup.cardM bS adm).succ) R := ClassGroup.di …
    hμ : Eq μ (ClassGroup.distinctElems bS adm)
    s : Finsupp ι R := bS.repr a
    s_eq : ∀ (i : ι), Eq (s i) ((bS.repr a) i)
    qs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HDiv.hDiv (HMul. …
    rs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HMod.hMod (HMul. …
    r_eq : ∀ (j : Fin (ClassGroup.cardM bS adm).succ) (i : ι), Eq (rs j i) (HMod.h …
    μ_eq : ∀ (i : ι) (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HMul.hMul (μ j) …
    μ_mul_a_eq : ∀ (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HSMul.hSMul (μ j) …
    j k : Fin (HPow.hPow (adm.card ε) (Fintype.card ι)).succ
    j_ne_k : Ne j k
    hjk : ∀ (k_1 : ι), LT.lt (↑(abv (HSub.hSub (HMod.hMod (HMul.hMul (μ k) (s k_1) …
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  have hjk' : ∀ i, (abv (rs k i - rs j i) : ℝ) < abv b • ε := by simpa only [r_eq] using hjk
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    a : S
    b : R
    hb : Ne b 0
    dim_pos : LT.lt 0 (Fintype.card ι)
    ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
    ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
    hε : LT.lt 0 ε
    ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
    μ : Function.Embedding (Fin (ClassGroup.cardM bS adm).succ) R := ClassGroup.di …
    hμ : Eq μ (ClassGroup.distinctElems bS adm)
    s : Finsupp ι R := bS.repr a
    s_eq : ∀ (i : ι), Eq (s i) ((bS.repr a) i)
    qs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HDiv.hDiv (HMul. …
    rs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HMod.hMod (HMul. …
    r_eq : ∀ (j : Fin (ClassGroup.cardM bS adm).succ) (i : ι), Eq (rs j i) (HMod.h …
    μ_eq : ∀ (i : ι) (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HMul.hMul (μ j) …
    μ_mul_a_eq : ∀ (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HSMul.hSMul (μ j) …
    j k : Fin (HPow.hPow (adm.card ε) (Fintype.card ι)).succ
    j_ne_k : Ne j k
    hjk : ∀ (k_1 : ι), LT.lt (↑(abv (HSub.hSub (HMod.hMod (HMul.hMul (μ k) (s k_1) …
    hjk' : ∀ (i : ι), LT.lt (↑(abv (HSub.hSub (rs k i) (rs j i)))) (HSMul.hSMul (a …
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  let q := ∑ i, (qs k i - qs j i) • bS i
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    a : S
    b : R
    hb : Ne b 0
    dim_pos : LT.lt 0 (Fintype.card ι)
    ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
    ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
    hε : LT.lt 0 ε
    ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
    μ : Function.Embedding (Fin (ClassGroup.cardM bS adm).succ) R := ClassGroup.di …
    hμ : Eq μ (ClassGroup.distinctElems bS adm)
    s : Finsupp ι R := bS.repr a
    s_eq : ∀ (i : ι), Eq (s i) ((bS.repr a) i)
    qs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HDiv.hDiv (HMul. …
    rs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HMod.hMod (HMul. …
    r_eq : ∀ (j : Fin (ClassGroup.cardM bS adm).succ) (i : ι), Eq (rs j i) (HMod.h …
    μ_eq : ∀ (i : ι) (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HMul.hMul (μ j) …
    μ_mul_a_eq : ∀ (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HSMul.hSMul (μ j) …
    j k : Fin (HPow.hPow (adm.card ε) (Fintype.card ι)).succ
    j_ne_k : Ne j k
    hjk : ∀ (k_1 : ι), LT.lt (↑(abv (HSub.hSub (HMod.hMod (HMul.hMul (μ k) (s k_1) …
    hjk' : ∀ (i : ι), LT.lt (↑(abv (HSub.hSub (rs k i) (rs j i)))) (HSMul.hSMul (a …
    q : S := Finset.univ.sum fun i => HSMul.hSMul (HSub.hSub (qs k i) (qs j i)) (b …
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  set r := μ k - μ j with r_eq
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    a : S
    b : R
    hb : Ne b 0
    dim_pos : LT.lt 0 (Fintype.card ι)
    ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
    ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
    hε : LT.lt 0 ε
    ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
    μ : Function.Embedding (Fin (ClassGroup.cardM bS adm).succ) R := ClassGroup.di …
    hμ : Eq μ (ClassGroup.distinctElems bS adm)
    s : Finsupp ι R := bS.repr a
    s_eq : ∀ (i : ι), Eq (s i) ((bS.repr a) i)
    qs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HDiv.hDiv (HMul. …
    rs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HMod.hMod (HMul. …
    r_eq✝ : ∀ (j : Fin (ClassGroup.cardM bS adm).succ) (i : ι), Eq (rs j i) (HMod. …
    μ_eq : ∀ (i : ι) (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HMul.hMul (μ j) …
    μ_mul_a_eq : ∀ (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HSMul.hSMul (μ j) …
    j k : Fin (HPow.hPow (adm.card ε) (Fintype.card ι)).succ
    j_ne_k : Ne j k
    hjk : ∀ (k_1 : ι), LT.lt (↑(abv (HSub.hSub (HMod.hMod (HMul.hMul (μ k) (s k_1) …
    hjk' : ∀ (i : ι), LT.lt (↑(abv (HSub.hSub (rs k i) (rs j i)))) (HSMul.hSMul (a …
    q : S := Finset.univ.sum fun i => HSMul.hSMul (HSub.hSub (qs k i) (qs j i)) (b …
    r : R := HSub.hSub (μ k) (μ j)
    r_eq : Eq r (HSub.hSub (μ k) (μ j))
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  refine ⟨q, r, (mem_finsetApprox bS adm).mpr ?_, ?_⟩
    /-
      case intro.intro.intro.refine_1
      R : Type u_1
      S : Type u_2
      inst✝⁷ : EuclideanDomain R
      inst✝⁶ : CommRing S
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝³ : DecidableEq ι
      inst✝² : Fintype ι
      bS : Basis ι R S
      adm : abv.IsAdmissible
      inst✝¹ : Infinite R
      inst✝ : DecidableEq R
      a : S
      b : R
      hb : Ne b 0
      dim_pos : LT.lt 0 (Fintype.card ι)
      ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
      ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
      hε : LT.lt 0 ε
      ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
      μ : Function.Embedding (Fin (ClassGroup.cardM bS adm).succ) R := ClassGroup.di …
      hμ : Eq μ (ClassGroup.distinctElems bS adm)
      s : Finsupp ι R := bS.repr a
      s_eq : ∀ (i : ι), Eq (s i) ((bS.repr a) i)
      qs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HDiv.hDiv (HMul. …
      rs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HMod.hMod (HMul. …
      r_eq✝ : ∀ (j : Fin (ClassGroup.cardM bS adm).succ) (i : ι), Eq (rs j i) (HMod. …
      μ_eq : ∀ (i : ι) (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HMul.hMul (μ j) …
      μ_mul_a_eq : ∀ (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HSMul.hSMul (μ j) …
      j k : Fin (HPow.hPow (adm.card ε) (Fintype.card ι)).succ
      j_ne_k : Ne j k
      hjk : ∀ (k_1 : ι), LT.lt (↑(abv (HSub.hSub (HMod.hMod (HMul.hMul (μ k) (s k_1) …
      hjk' : ∀ (i : ι), LT.lt (↑(abv (HSub.hSub (rs k i) (rs j i)))) (HSMul.hSMul (a …
      q : S := Finset.univ.sum fun i => HSMul.hSMul (HSub.hSub (qs k i) (qs j i)) (b …
      r : R := HSub.hSub (μ k) (μ j)
      r_eq : Eq r (HSub.hSub (μ k) (μ j))
      ⊢ Exists fun i => Exists fun j => And (Ne i j) (Eq (HSub.hSub ((ClassGroup.dis …
    -/
  · exact ⟨k, j, j_ne_k.symm, rfl⟩
    /-
      🎉 no goals
    -/
  have : r • a - b • q = ∑ x : ι, (rs k x • bS x - rs j x • bS x) := by
    simp only [q, r_eq, sub_smul, μ_mul_a_eq, Finset.smul_sum, ← Finset.sum_add_distrib,
      ← Finset.sum_sub_distrib, smul_sub]
    refine Finset.sum_congr rfl fun x _ => ?_
    ring
  /-
    case intro.intro.intro.refine_2
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    a : S
    b : R
    hb : Ne b 0
    dim_pos : LT.lt 0 (Fintype.card ι)
    ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
    ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
    hε : LT.lt 0 ε
    ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
    μ : Function.Embedding (Fin (ClassGroup.cardM bS adm).succ) R := ClassGroup.di …
    hμ : Eq μ (ClassGroup.distinctElems bS adm)
    s : Finsupp ι R := bS.repr a
    s_eq : ∀ (i : ι), Eq (s i) ((bS.repr a) i)
    qs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HDiv.hDiv (HMul. …
    rs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HMod.hMod (HMul. …
    r_eq✝ : ∀ (j : Fin (ClassGroup.cardM bS adm).succ) (i : ι), Eq (rs j i) (HMod. …
    μ_eq : ∀ (i : ι) (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HMul.hMul (μ j) …
    μ_mul_a_eq : ∀ (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HSMul.hSMul (μ j) …
    j k : Fin (HPow.hPow (adm.card ε) (Fintype.card ι)).succ
    j_ne_k : Ne j k
    hjk : ∀ (k_1 : ι), LT.lt (↑(abv (HSub.hSub (HMod.hMod (HMul.hMul (μ k) (s k_1) …
    hjk' : ∀ (i : ι), LT.lt (↑(abv (HSub.hSub (rs k i) (rs j i)))) (HSMul.hSMul (a …
    q : S := Finset.univ.sum fun i => HSMul.hSMul (HSub.hSub (qs k i) (qs j i)) (b …
    r : R := HSub.hSub (μ k) (μ j)
    r_eq : Eq r (HSub.hSub (μ k) (μ j))
    this : Eq (HSub.hSub (HSMul.hSMul r a) (HSMul.hSMul b q)) (Finset.univ.sum fun …
    ⊢ LT.lt (abv ((Algebra.norm R) (HSub.hSub (HSMul.hSMul r a) (HSMul.hSMul b q)) …
  -/
  rw [this, Algebra.norm_algebraMap_of_basis bS, abv.map_pow]
  /-
    case intro.intro.intro.refine_2
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    a : S
    b : R
    hb : Ne b 0
    dim_pos : LT.lt 0 (Fintype.card ι)
    ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
    ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
    hε : LT.lt 0 ε
    ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
    μ : Function.Embedding (Fin (ClassGroup.cardM bS adm).succ) R := ClassGroup.di …
    hμ : Eq μ (ClassGroup.distinctElems bS adm)
    s : Finsupp ι R := bS.repr a
    s_eq : ∀ (i : ι), Eq (s i) ((bS.repr a) i)
    qs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HDiv.hDiv (HMul. …
    rs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HMod.hMod (HMul. …
    r_eq✝ : ∀ (j : Fin (ClassGroup.cardM bS adm).succ) (i : ι), Eq (rs j i) (HMod. …
    μ_eq : ∀ (i : ι) (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HMul.hMul (μ j) …
    μ_mul_a_eq : ∀ (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HSMul.hSMul (μ j) …
    j k : Fin (HPow.hPow (adm.card ε) (Fintype.card ι)).succ
    j_ne_k : Ne j k
    hjk : ∀ (k_1 : ι), LT.lt (↑(abv (HSub.hSub (HMod.hMod (HMul.hMul (μ k) (s k_1) …
    hjk' : ∀ (i : ι), LT.lt (↑(abv (HSub.hSub (rs k i) (rs j i)))) (HSMul.hSMul (a …
    q : S := Finset.univ.sum fun i => HSMul.hSMul (HSub.hSub (qs k i) (qs j i)) (b …
    r : R := HSub.hSub (μ k) (μ j)
    r_eq : Eq r (HSub.hSub (μ k) (μ j))
    this : Eq (HSub.hSub (HSMul.hSMul r a) (HSMul.hSMul b q)) (Finset.univ.sum fun …
    ⊢ LT.lt (abv ((Algebra.norm R) (Finset.univ.sum fun x => HSub.hSub (HSMul.hSMu …
  -/
  refine Int.cast_lt.mp ((norm_lt abv bS _ fun i => lt_of_le_of_lt ?_ (hjk' i)).trans_le ?_)
    /-
      case intro.intro.intro.refine_2.refine_1
      R : Type u_1
      S : Type u_2
      inst✝⁷ : EuclideanDomain R
      inst✝⁶ : CommRing S
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝³ : DecidableEq ι
      inst✝² : Fintype ι
      bS : Basis ι R S
      adm : abv.IsAdmissible
      inst✝¹ : Infinite R
      inst✝ : DecidableEq R
      a : S
      b : R
      hb : Ne b 0
      dim_pos : LT.lt 0 (Fintype.card ι)
      ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
      ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
      hε : LT.lt 0 ε
      ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
      μ : Function.Embedding (Fin (ClassGroup.cardM bS adm).succ) R := ClassGroup.di …
      hμ : Eq μ (ClassGroup.distinctElems bS adm)
      s : Finsupp ι R := bS.repr a
      s_eq : ∀ (i : ι), Eq (s i) ((bS.repr a) i)
      qs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HDiv.hDiv (HMul. …
      rs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HMod.hMod (HMul. …
      r_eq✝ : ∀ (j : Fin (ClassGroup.cardM bS adm).succ) (i : ι), Eq (rs j i) (HMod. …
      μ_eq : ∀ (i : ι) (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HMul.hMul (μ j) …
      μ_mul_a_eq : ∀ (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HSMul.hSMul (μ j) …
      j k : Fin (HPow.hPow (adm.card ε) (Fintype.card ι)).succ
      j_ne_k : Ne j k
      hjk : ∀ (k_1 : ι), LT.lt (↑(abv (HSub.hSub (HMod.hMod (HMul.hMul (μ k) (s k_1) …
      hjk' : ∀ (i : ι), LT.lt (↑(abv (HSub.hSub (rs k i) (rs j i)))) (HSMul.hSMul (a …
      q : S := Finset.univ.sum fun i => HSMul.hSMul (HSub.hSub (qs k i) (qs j i)) (b …
      r : R := HSub.hSub (μ k) (μ j)
      r_eq : Eq r (HSub.hSub (μ k) (μ j))
      this : Eq (HSub.hSub (HSMul.hSMul r a) (HSMul.hSMul b q)) (Finset.univ.sum fun …
      i : ι
      ⊢ LE.le ↑(abv ((bS.repr (Finset.univ.sum fun x => HSub.hSub (HSMul.hSMul (rs k …
    -/
  · apply le_of_eq
    /-
      case intro.intro.intro.refine_2.refine_1.hab
      R : Type u_1
      S : Type u_2
      inst✝⁷ : EuclideanDomain R
      inst✝⁶ : CommRing S
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝³ : DecidableEq ι
      inst✝² : Fintype ι
      bS : Basis ι R S
      adm : abv.IsAdmissible
      inst✝¹ : Infinite R
      inst✝ : DecidableEq R
      a : S
      b : R
      hb : Ne b 0
      dim_pos : LT.lt 0 (Fintype.card ι)
      ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
      ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
      hε : LT.lt 0 ε
      ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
      μ : Function.Embedding (Fin (ClassGroup.cardM bS adm).succ) R := ClassGroup.di …
      hμ : Eq μ (ClassGroup.distinctElems bS adm)
      s : Finsupp ι R := bS.repr a
      s_eq : ∀ (i : ι), Eq (s i) ((bS.repr a) i)
      qs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HDiv.hDiv (HMul. …
      rs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HMod.hMod (HMul. …
      r_eq✝ : ∀ (j : Fin (ClassGroup.cardM bS adm).succ) (i : ι), Eq (rs j i) (HMod. …
      μ_eq : ∀ (i : ι) (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HMul.hMul (μ j) …
      μ_mul_a_eq : ∀ (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HSMul.hSMul (μ j) …
      j k : Fin (HPow.hPow (adm.card ε) (Fintype.card ι)).succ
      j_ne_k : Ne j k
      hjk : ∀ (k_1 : ι), LT.lt (↑(abv (HSub.hSub (HMod.hMod (HMul.hMul (μ k) (s k_1) …
      hjk' : ∀ (i : ι), LT.lt (↑(abv (HSub.hSub (rs k i) (rs j i)))) (HSMul.hSMul (a …
      q : S := Finset.univ.sum fun i => HSMul.hSMul (HSub.hSub (qs k i) (qs j i)) (b …
      r : R := HSub.hSub (μ k) (μ j)
      r_eq : Eq r (HSub.hSub (μ k) (μ j))
      this : Eq (HSub.hSub (HSMul.hSMul r a) (HSMul.hSMul b q)) (Finset.univ.sum fun …
      i : ι
      ⊢ Eq ↑(abv ((bS.repr (Finset.univ.sum fun x => HSub.hSub (HSMul.hSMul (rs k x) …
    -/
    congr
    simp_rw [map_sum, map_sub, map_smul, Finset.sum_apply',
      Finsupp.sub_apply, Finsupp.smul_apply, Finset.sum_sub_distrib, Basis.repr_self_apply,
      smul_eq_mul, mul_boole, Finset.sum_ite_eq', Finset.mem_univ, if_true]
    /-
      case intro.intro.intro.refine_2.refine_2
      R : Type u_1
      S : Type u_2
      inst✝⁷ : EuclideanDomain R
      inst✝⁶ : CommRing S
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R S
      abv : AbsoluteValue R Int
      ι : Type u_5
      inst✝³ : DecidableEq ι
      inst✝² : Fintype ι
      bS : Basis ι R S
      adm : abv.IsAdmissible
      inst✝¹ : Infinite R
      inst✝ : DecidableEq R
      a : S
      b : R
      hb : Ne b 0
      dim_pos : LT.lt 0 (Fintype.card ι)
      ε : Real := HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Finty …
      ε_eq : Eq ε (HPow.hPow (↑(ClassGroup.normBound abv bS)) (HDiv.hDiv (-1) ↑(Fint …
      hε : LT.lt 0 ε
      ε_le : LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSM …
      μ : Function.Embedding (Fin (ClassGroup.cardM bS adm).succ) R := ClassGroup.di …
      hμ : Eq μ (ClassGroup.distinctElems bS adm)
      s : Finsupp ι R := bS.repr a
      s_eq : ∀ (i : ι), Eq (s i) ((bS.repr a) i)
      qs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HDiv.hDiv (HMul. …
      rs : Fin (ClassGroup.cardM bS adm).succ → ι → R := fun j i => HMod.hMod (HMul. …
      r_eq✝ : ∀ (j : Fin (ClassGroup.cardM bS adm).succ) (i : ι), Eq (rs j i) (HMod. …
      μ_eq : ∀ (i : ι) (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HMul.hMul (μ j) …
      μ_mul_a_eq : ∀ (j : Fin (ClassGroup.cardM bS adm).succ), Eq (HSMul.hSMul (μ j) …
      j k : Fin (HPow.hPow (adm.card ε) (Fintype.card ι)).succ
      j_ne_k : Ne j k
      hjk : ∀ (k_1 : ι), LT.lt (↑(abv (HSub.hSub (HMod.hMod (HMul.hMul (μ k) (s k_1) …
      hjk' : ∀ (i : ι), LT.lt (↑(abv (HSub.hSub (rs k i) (rs j i)))) (HSMul.hSMul (a …
      q : S := Finset.univ.sum fun i => HSMul.hSMul (HSub.hSub (qs k i) (qs j i)) (b …
      r : R := HSub.hSub (μ k) (μ j)
      r_eq : Eq r (HSub.hSub (μ k) (μ j))
      this : Eq (HSub.hSub (HSMul.hSMul r a) (HSMul.hSMul b q)) (Finset.univ.sum fun …
      ⊢ LE.le (HMul.hMul (↑(ClassGroup.normBound abv bS)) (HPow.hPow (HSMul.hSMul (a …
    -/
  · exact mod_cast ε_le
    /-
      🎉 no goals
    -/


/-- We can approximate `a / b : L` with `q / r`, where `r` has finitely many options for `L`. -/
theorem exists_mem_finset_approx' [Algebra.IsAlgebraic R S] (a : S) {b : S} (hb : b ≠ 0) :
    ∃ q : S,
      ∃ r ∈ finsetApprox bS adm, abv (Algebra.norm R (r • a - q * b)) < abv (Algebra.norm R b) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁸ : EuclideanDomain R
    inst✝⁷ : CommRing S
    inst✝⁶ : IsDomain S
    inst✝⁵ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝² : Infinite R
    inst✝¹ : DecidableEq R
    inst✝ : Algebra.IsAlgebraic R S
    a b : S
    hb : Ne b 0
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  obtain ⟨a', b', hb', h⟩ := Algebra.IsAlgebraic.exists_smul_eq_mul R a hb
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁸ : EuclideanDomain R
    inst✝⁷ : CommRing S
    inst✝⁶ : IsDomain S
    inst✝⁵ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝² : Infinite R
    inst✝¹ : DecidableEq R
    inst✝ : Algebra.IsAlgebraic R S
    a b : S
    hb : Ne b 0
    a' : S
    b' : R
    hb' : Ne b' 0
    h : Eq (HSMul.hSMul b' a) (HMul.hMul b a')
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  obtain ⟨q, r, hr, hqr⟩ := exists_mem_finsetApprox bS adm a' hb'
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁸ : EuclideanDomain R
    inst✝⁷ : CommRing S
    inst✝⁶ : IsDomain S
    inst✝⁵ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝² : Infinite R
    inst✝¹ : DecidableEq R
    inst✝ : Algebra.IsAlgebraic R S
    a b : S
    hb : Ne b 0
    a' : S
    b' : R
    hb' : Ne b' 0
    h : Eq (HSMul.hSMul b' a) (HMul.hMul b a')
    q : S
    r : R
    hr : Membership.mem (ClassGroup.finsetApprox bS adm) r
    hqr : LT.lt (abv ((Algebra.norm R) (HSub.hSub (HSMul.hSMul r a') (HSMul.hSMul  …
    ⊢ Exists fun q => Exists fun r => And (Membership.mem (ClassGroup.finsetApprox …
  -/
  refine ⟨q, r, hr, ?_⟩
  refine
    lt_of_mul_lt_mul_left ?_ (show 0 ≤ abv (Algebra.norm R (algebraMap R S b')) from abv.nonneg _)
  refine
    lt_of_le_of_lt (le_of_eq ?_)
      (mul_lt_mul hqr le_rfl (abv.pos ((Algebra.norm_ne_zero_iff_of_basis bS).mpr hb))
        (abv.nonneg _))
  rw [← abv.map_mul, ← MonoidHom.map_mul, ← abv.map_mul, ← MonoidHom.map_mul, ← Algebra.smul_def,
    smul_sub b', sub_mul, smul_comm, h, mul_comm b a', Algebra.smul_mul_assoc r a' b,
    Algebra.smul_mul_assoc b' q b]


theorem prod_finsetApprox_ne_zero : algebraMap R S (∏ m ∈ finsetApprox bS adm, m) ≠ 0 := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    ⊢ Ne ((algebraMap R S) ((ClassGroup.finsetApprox bS adm).prod fun m => m)) 0
  -/
  refine mt ((injective_iff_map_eq_zero _).mp bS.algebraMap_injective _) ?_
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    ⊢ Not (Eq ((ClassGroup.finsetApprox bS adm).prod fun m => m) 0)
  -/
  simp only [Finset.prod_eq_zero_iff, not_exists]
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    ⊢ ∀ (x : R), Not (And (Membership.mem (ClassGroup.finsetApprox bS adm) x) (Eq  …
  -/
  rintro x ⟨hx, rfl⟩
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝⁷ : EuclideanDomain R
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝¹ : Infinite R
    inst✝ : DecidableEq R
    hx : Membership.mem (ClassGroup.finsetApprox bS adm) 0
    ⊢ False
  -/
  exact finsetApprox.zero_not_mem bS adm hx
  /-
    🎉 no goals
  -/


theorem ne_bot_of_prod_finsetApprox_mem (J : Ideal S)
    (h : algebraMap _ _ (∏ m ∈ finsetApprox bS adm, m) ∈ J) : J ≠ ⊥ :=
  (Submodule.ne_bot_iff _).mpr ⟨_, h, prod_finsetApprox_ne_zero _ _⟩


/-- Each class in the class group contains an ideal `J`
such that `M := Π m ∈ finsetApprox` is in `J`. -/
theorem exists_mk0_eq_mk0 [IsDedekindDomain S] [Algebra.IsAlgebraic R S] (I : (Ideal S)⁰) :
    ∃ J : (Ideal S)⁰,
      ClassGroup.mk0 I = ClassGroup.mk0 J ∧
        algebraMap _ _ (∏ m ∈ finsetApprox bS adm, m) ∈ (J : Ideal S) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁹ : EuclideanDomain R
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝³ : Infinite R
    inst✝² : DecidableEq R
    inst✝¹ : IsDedekindDomain S
    inst✝ : Algebra.IsAlgebraic R S
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal S)) x
    ⊢ Exists fun J => And (Eq (ClassGroup.mk0 I) (ClassGroup.mk0 J)) (Membership.m …
  -/
  set M := ∏ m ∈ finsetApprox bS adm, m
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁹ : EuclideanDomain R
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝³ : Infinite R
    inst✝² : DecidableEq R
    inst✝¹ : IsDedekindDomain S
    inst✝ : Algebra.IsAlgebraic R S
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal S)) x
    M : R := (ClassGroup.finsetApprox bS adm).prod fun m => m
    ⊢ Exists fun J => And (Eq (ClassGroup.mk0 I) (ClassGroup.mk0 J)) (Membership.m …
  -/
  have hM : algebraMap R S M ≠ 0 := prod_finsetApprox_ne_zero bS adm
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁹ : EuclideanDomain R
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝³ : Infinite R
    inst✝² : DecidableEq R
    inst✝¹ : IsDedekindDomain S
    inst✝ : Algebra.IsAlgebraic R S
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal S)) x
    M : R := (ClassGroup.finsetApprox bS adm).prod fun m => m
    hM : Ne ((algebraMap R S) M) 0
    ⊢ Exists fun J => And (Eq (ClassGroup.mk0 I) (ClassGroup.mk0 J)) (Membership.m …
  -/
  obtain ⟨b, b_mem, b_ne_zero, b_min⟩ := exists_min abv I
  suffices Ideal.span {b} ∣ Ideal.span {algebraMap _ _ M} * I.1 by
    obtain ⟨J, hJ⟩ := this
    refine ⟨⟨J, ?_⟩, ?_, ?_⟩
    · rw [mem_nonZeroDivisors_iff_ne_zero]
      rintro rfl
      rw [Ideal.zero_eq_bot, Ideal.mul_bot] at hJ
      exact hM (Ideal.span_singleton_eq_bot.mp (I.2 _ hJ))
    · rw [ClassGroup.mk0_eq_mk0_iff]
      exact ⟨algebraMap _ _ M, b, hM, b_ne_zero, hJ⟩
    rw [← SetLike.mem_coe, ← Set.singleton_subset_iff, ← Ideal.span_le, ← Ideal.dvd_iff_le]
    apply (mul_dvd_mul_iff_left _).mp _
    swap; · exact mt Ideal.span_singleton_eq_bot.mp b_ne_zero
    rw [Subtype.coe_mk, Ideal.dvd_iff_le, ← hJ, mul_comm]
    apply Ideal.mul_mono le_rfl
    rw [Ideal.span_le, Set.singleton_subset_iff]
    exact b_mem
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁹ : EuclideanDomain R
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝³ : Infinite R
    inst✝² : DecidableEq R
    inst✝¹ : IsDedekindDomain S
    inst✝ : Algebra.IsAlgebraic R S
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal S)) x
    M : R := (ClassGroup.finsetApprox bS adm).prod fun m => m
    hM : Ne ((algebraMap R S) M) 0
    b : S
    b_mem : Membership.mem (↑I) b
    b_ne_zero : Ne b 0
    b_min : ∀ (c : S), Membership.mem (↑I) c → LT.lt (abv ((Algebra.norm R) c)) (a …
    ⊢ Dvd.dvd (Ideal.span (Singleton.singleton b)) (HMul.hMul (Ideal.span (Singlet …
  -/
  rw [Ideal.dvd_iff_le, Ideal.mul_le]
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁹ : EuclideanDomain R
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝³ : Infinite R
    inst✝² : DecidableEq R
    inst✝¹ : IsDedekindDomain S
    inst✝ : Algebra.IsAlgebraic R S
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal S)) x
    M : R := (ClassGroup.finsetApprox bS adm).prod fun m => m
    hM : Ne ((algebraMap R S) M) 0
    b : S
    b_mem : Membership.mem (↑I) b
    b_ne_zero : Ne b 0
    b_min : ∀ (c : S), Membership.mem (↑I) c → LT.lt (abv ((Algebra.norm R) c)) (a …
    ⊢ ∀ (r : S), Membership.mem (Ideal.span (Singleton.singleton ((algebraMap R S) …
  -/
  intro r' hr' a ha
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁹ : EuclideanDomain R
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝³ : Infinite R
    inst✝² : DecidableEq R
    inst✝¹ : IsDedekindDomain S
    inst✝ : Algebra.IsAlgebraic R S
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal S)) x
    M : R := (ClassGroup.finsetApprox bS adm).prod fun m => m
    hM : Ne ((algebraMap R S) M) 0
    b : S
    b_mem : Membership.mem (↑I) b
    b_ne_zero : Ne b 0
    b_min : ∀ (c : S), Membership.mem (↑I) c → LT.lt (abv ((Algebra.norm R) c)) (a …
    r' : S
    hr' : Membership.mem (Ideal.span (Singleton.singleton ((algebraMap R S) M))) r'
    a : S
    ha : Membership.mem (↑I) a
    ⊢ Membership.mem (Ideal.span (Singleton.singleton b)) (HMul.hMul r' a)
  -/
  rw [Ideal.mem_span_singleton] at hr' ⊢
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁹ : EuclideanDomain R
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝³ : Infinite R
    inst✝² : DecidableEq R
    inst✝¹ : IsDedekindDomain S
    inst✝ : Algebra.IsAlgebraic R S
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal S)) x
    M : R := (ClassGroup.finsetApprox bS adm).prod fun m => m
    hM : Ne ((algebraMap R S) M) 0
    b : S
    b_mem : Membership.mem (↑I) b
    b_ne_zero : Ne b 0
    b_min : ∀ (c : S), Membership.mem (↑I) c → LT.lt (abv ((Algebra.norm R) c)) (a …
    r' : S
    hr' : Dvd.dvd ((algebraMap R S) M) r'
    a : S
    ha : Membership.mem (↑I) a
    ⊢ Dvd.dvd b (HMul.hMul r' a)
  -/
  obtain ⟨q, r, r_mem, lt⟩ := exists_mem_finset_approx' bS adm a b_ne_zero
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁹ : EuclideanDomain R
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝³ : Infinite R
    inst✝² : DecidableEq R
    inst✝¹ : IsDedekindDomain S
    inst✝ : Algebra.IsAlgebraic R S
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal S)) x
    M : R := (ClassGroup.finsetApprox bS adm).prod fun m => m
    hM : Ne ((algebraMap R S) M) 0
    b : S
    b_mem : Membership.mem (↑I) b
    b_ne_zero : Ne b 0
    b_min : ∀ (c : S), Membership.mem (↑I) c → LT.lt (abv ((Algebra.norm R) c)) (a …
    r' : S
    hr' : Dvd.dvd ((algebraMap R S) M) r'
    a : S
    ha : Membership.mem (↑I) a
    q : S
    r : R
    r_mem : Membership.mem (ClassGroup.finsetApprox bS adm) r
    lt : LT.lt (abv ((Algebra.norm R) (HSub.hSub (HSMul.hSMul r a) (HMul.hMul q b) …
    ⊢ Dvd.dvd b (HMul.hMul r' a)
  -/
  apply @dvd_of_mul_left_dvd _ _ q
  /-
    case intro.intro.intro.intro.intro.intro.h
    R : Type u_1
    S : Type u_2
    inst✝⁹ : EuclideanDomain R
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝³ : Infinite R
    inst✝² : DecidableEq R
    inst✝¹ : IsDedekindDomain S
    inst✝ : Algebra.IsAlgebraic R S
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal S)) x
    M : R := (ClassGroup.finsetApprox bS adm).prod fun m => m
    hM : Ne ((algebraMap R S) M) 0
    b : S
    b_mem : Membership.mem (↑I) b
    b_ne_zero : Ne b 0
    b_min : ∀ (c : S), Membership.mem (↑I) c → LT.lt (abv ((Algebra.norm R) c)) (a …
    r' : S
    hr' : Dvd.dvd ((algebraMap R S) M) r'
    a : S
    ha : Membership.mem (↑I) a
    q : S
    r : R
    r_mem : Membership.mem (ClassGroup.finsetApprox bS adm) r
    lt : LT.lt (abv ((Algebra.norm R) (HSub.hSub (HSMul.hSMul r a) (HMul.hMul q b) …
    ⊢ Dvd.dvd (HMul.hMul q b) (HMul.hMul r' a)
  -/
  simp only [Algebra.smul_def] at lt
  rw [←
    sub_eq_zero.mp (b_min _ (I.1.sub_mem (I.1.mul_mem_left _ ha) (I.1.mul_mem_left _ b_mem)) lt)]
  /-
    case intro.intro.intro.intro.intro.intro.h
    R : Type u_1
    S : Type u_2
    inst✝⁹ : EuclideanDomain R
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝³ : Infinite R
    inst✝² : DecidableEq R
    inst✝¹ : IsDedekindDomain S
    inst✝ : Algebra.IsAlgebraic R S
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal S)) x
    M : R := (ClassGroup.finsetApprox bS adm).prod fun m => m
    hM : Ne ((algebraMap R S) M) 0
    b : S
    b_mem : Membership.mem (↑I) b
    b_ne_zero : Ne b 0
    b_min : ∀ (c : S), Membership.mem (↑I) c → LT.lt (abv ((Algebra.norm R) c)) (a …
    r' : S
    hr' : Dvd.dvd ((algebraMap R S) M) r'
    a : S
    ha : Membership.mem (↑I) a
    q : S
    r : R
    r_mem : Membership.mem (ClassGroup.finsetApprox bS adm) r
    lt : LT.lt (abv ((Algebra.norm R) (HSub.hSub (HMul.hMul ((algebraMap R S) r) a …
    ⊢ Dvd.dvd (HMul.hMul ((algebraMap R S) r) a) (HMul.hMul r' a)
  -/
  refine mul_dvd_mul_right (dvd_trans (RingHom.map_dvd _ ?_) hr') _
  /-
    case intro.intro.intro.intro.intro.intro.h
    R : Type u_1
    S : Type u_2
    inst✝⁹ : EuclideanDomain R
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝³ : Infinite R
    inst✝² : DecidableEq R
    inst✝¹ : IsDedekindDomain S
    inst✝ : Algebra.IsAlgebraic R S
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal S)) x
    M : R := (ClassGroup.finsetApprox bS adm).prod fun m => m
    hM : Ne ((algebraMap R S) M) 0
    b : S
    b_mem : Membership.mem (↑I) b
    b_ne_zero : Ne b 0
    b_min : ∀ (c : S), Membership.mem (↑I) c → LT.lt (abv ((Algebra.norm R) c)) (a …
    r' : S
    hr' : Dvd.dvd ((algebraMap R S) M) r'
    a : S
    ha : Membership.mem (↑I) a
    q : S
    r : R
    r_mem : Membership.mem (ClassGroup.finsetApprox bS adm) r
    lt : LT.lt (abv ((Algebra.norm R) (HSub.hSub (HMul.hMul ((algebraMap R S) r) a …
    ⊢ Dvd.dvd r M
  -/
  exact Multiset.dvd_prod (Multiset.mem_map.mpr ⟨_, r_mem, rfl⟩)
  /-
    🎉 no goals
  -/


/-- `ClassGroup.mkMMem` is a specialization of `ClassGroup.mk0` to (the finite set of)
ideals that contain `M := ∏ m ∈ finsetApprox L f abs, m`.
By showing this function is surjective, we prove that the class group is finite. -/
noncomputable def mkMMem [IsDedekindDomain S]
    (J : { J : Ideal S // algebraMap _ _ (∏ m ∈ finsetApprox bS adm, m) ∈ J }) : ClassGroup S :=
  ClassGroup.mk0
    ⟨J.1, mem_nonZeroDivisors_iff_ne_zero.mpr (ne_bot_of_prod_finsetApprox_mem bS adm J.1 J.2)⟩


theorem mkMMem_surjective [IsDedekindDomain S] [Algebra.IsAlgebraic R S] :
    Function.Surjective (ClassGroup.mkMMem bS adm) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁹ : EuclideanDomain R
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝³ : Infinite R
    inst✝² : DecidableEq R
    inst✝¹ : IsDedekindDomain S
    inst✝ : Algebra.IsAlgebraic R S
    ⊢ Function.Surjective (ClassGroup.mkMMem bS adm)
  -/
  intro I'
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁹ : EuclideanDomain R
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝³ : Infinite R
    inst✝² : DecidableEq R
    inst✝¹ : IsDedekindDomain S
    inst✝ : Algebra.IsAlgebraic R S
    I' : ClassGroup S
    ⊢ Exists fun a => Eq (ClassGroup.mkMMem bS adm a) I'
  -/
  obtain ⟨⟨I, hI⟩, rfl⟩ := ClassGroup.mk0_surjective I'
  /-
    case intro.mk
    R : Type u_1
    S : Type u_2
    inst✝⁹ : EuclideanDomain R
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝³ : Infinite R
    inst✝² : DecidableEq R
    inst✝¹ : IsDedekindDomain S
    inst✝ : Algebra.IsAlgebraic R S
    I : Ideal S
    hI : Membership.mem (nonZeroDivisors (Ideal S)) I
    ⊢ Exists fun a => Eq (ClassGroup.mkMMem bS adm a) (ClassGroup.mk0 ⟨I, hI⟩)
  -/
  obtain ⟨J, mk0_eq_mk0, J_dvd⟩ := exists_mk0_eq_mk0 bS adm ⟨I, hI⟩
  /-
    case intro.mk.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁹ : EuclideanDomain R
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝³ : Infinite R
    inst✝² : DecidableEq R
    inst✝¹ : IsDedekindDomain S
    inst✝ : Algebra.IsAlgebraic R S
    I : Ideal S
    hI : Membership.mem (nonZeroDivisors (Ideal S)) I
    J : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal S)) x
    mk0_eq_mk0 : Eq (ClassGroup.mk0 ⟨I, hI⟩) (ClassGroup.mk0 J)
    J_dvd : Membership.mem (↑J) ((algebraMap R S) ((ClassGroup.finsetApprox bS adm …
    ⊢ Exists fun a => Eq (ClassGroup.mkMMem bS adm a) (ClassGroup.mk0 ⟨I, hI⟩)
  -/
  exact ⟨⟨J, J_dvd⟩, mk0_eq_mk0.symm⟩
  /-
    🎉 no goals
  -/


open Classical in
/-- The **class number theorem**: the class group of an integral closure `S` of `R` in an
algebraic extension `L` is finite if there is an admissible absolute value.

See also `ClassGroup.fintypeOfAdmissibleOfFinite` where `L` is a finite
extension of `K = Frac(R)`, supplying most of the required assumptions automatically.
-/
noncomputable def fintypeOfAdmissibleOfAlgebraic [IsDedekindDomain S]
    [Algebra.IsAlgebraic R S] : Fintype (ClassGroup S) :=
  @Fintype.ofSurjective _ _ _
    (@Fintype.ofEquiv _
      { J // J ∣ Ideal.span ({algebraMap R S (∏ m ∈ finsetApprox bS adm, m)} : Set S) }
      (UniqueFactorizationMonoid.fintypeSubtypeDvd _
        (by
          /-
            R : Type u_1
            S : Type u_2
            K : Type u_3
            L : Type u_4
            inst✝¹⁸ : EuclideanDomain R
            inst✝¹⁷ : CommRing S
            inst✝¹⁶ : IsDomain S
            inst✝¹⁵ : Field K
            inst✝¹⁴ : Field L
            inst✝¹³ : Algebra R K
            inst✝¹² : IsFractionRing R K
            inst✝¹¹ : Algebra K L
            inst✝¹⁰ : FiniteDimensional K L
            inst✝⁹ : Algebra.IsSeparable K L
            algRL : Algebra R L
            inst✝⁸ : IsScalarTower R K L
            inst✝⁷ : Algebra R S
            inst✝⁶ : Algebra S L
            ist : IsScalarTower R S L
            abv : AbsoluteValue R Int
            ι : Type u_5
            inst✝⁵ : DecidableEq ι
            inst✝⁴ : Fintype ι
            bS : Basis ι R S
            adm : abv.IsAdmissible
            inst✝³ : Infinite R
            inst✝² : DecidableEq R
            inst✝¹ : IsDedekindDomain S
            inst✝ : Algebra.IsAlgebraic R S
            ⊢ Ne (Ideal.span (Singleton.singleton ((algebraMap R S) ((ClassGroup.finsetApp …
          -/
          rw [Ne, Ideal.zero_eq_bot, Ideal.span_singleton_eq_bot]
          /-
            R : Type u_1
            S : Type u_2
            K : Type u_3
            L : Type u_4
            inst✝¹⁸ : EuclideanDomain R
            inst✝¹⁷ : CommRing S
            inst✝¹⁶ : IsDomain S
            inst✝¹⁵ : Field K
            inst✝¹⁴ : Field L
            inst✝¹³ : Algebra R K
            inst✝¹² : IsFractionRing R K
            inst✝¹¹ : Algebra K L
            inst✝¹⁰ : FiniteDimensional K L
            inst✝⁹ : Algebra.IsSeparable K L
            algRL : Algebra R L
            inst✝⁸ : IsScalarTower R K L
            inst✝⁷ : Algebra R S
            inst✝⁶ : Algebra S L
            ist : IsScalarTower R S L
            abv : AbsoluteValue R Int
            ι : Type u_5
            inst✝⁵ : DecidableEq ι
            inst✝⁴ : Fintype ι
            bS : Basis ι R S
            adm : abv.IsAdmissible
            inst✝³ : Infinite R
            inst✝² : DecidableEq R
            inst✝¹ : IsDedekindDomain S
            inst✝ : Algebra.IsAlgebraic R S
            ⊢ Not (Eq ((algebraMap R S) ((ClassGroup.finsetApprox bS adm).prod fun m => m) …
          -/
          exact prod_finsetApprox_ne_zero bS adm))
          /-
            🎉 no goals
          -/
      ((Equiv.refl _).subtypeEquiv fun I =>
        Ideal.dvd_iff_le.trans (by
          /-
            R : Type u_1
            S : Type u_2
            K : Type u_3
            L : Type u_4
            inst✝¹⁸ : EuclideanDomain R
            inst✝¹⁷ : CommRing S
            inst✝¹⁶ : IsDomain S
            inst✝¹⁵ : Field K
            inst✝¹⁴ : Field L
            inst✝¹³ : Algebra R K
            inst✝¹² : IsFractionRing R K
            inst✝¹¹ : Algebra K L
            inst✝¹⁰ : FiniteDimensional K L
            inst✝⁹ : Algebra.IsSeparable K L
            algRL : Algebra R L
            inst✝⁸ : IsScalarTower R K L
            inst✝⁷ : Algebra R S
            inst✝⁶ : Algebra S L
            ist : IsScalarTower R S L
            abv : AbsoluteValue R Int
            ι : Type u_5
            inst✝⁵ : DecidableEq ι
            inst✝⁴ : Fintype ι
            bS : Basis ι R S
            adm : abv.IsAdmissible
            inst✝³ : Infinite R
            inst✝² : DecidableEq R
            inst✝¹ : IsDedekindDomain S
            inst✝ : Algebra.IsAlgebraic R S
            I : Ideal S
            ⊢ Iff (LE.le (Ideal.span (Singleton.singleton ((algebraMap R S) ((ClassGroup.f …
          -/
          rw [Equiv.refl_apply, Ideal.span_le, Set.singleton_subset_iff]; rfl)))
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    (ClassGroup.mkMMem bS adm) (ClassGroup.mkMMem_surjective bS adm)


/-- The main theorem: the class group of an integral closure `S` of `R` in a
finite extension `L` of `K = Frac(R)` is finite if there is an admissible
absolute value.

See also `ClassGroup.fintypeOfAdmissibleOfAlgebraic` where `L` is an
algebraic extension of `R`, that includes some extra assumptions.
-/
noncomputable def fintypeOfAdmissibleOfFinite [IsIntegralClosure S R L] :
    Fintype (ClassGroup S) := by
  /-
    R : Type u_1
    S : Type u_2
    K : Type u_3
    L : Type u_4
    inst✝¹⁷ : EuclideanDomain R
    inst✝¹⁶ : CommRing S
    inst✝¹⁵ : IsDomain S
    inst✝¹⁴ : Field K
    inst✝¹³ : Field L
    inst✝¹² : Algebra R K
    inst✝¹¹ : IsFractionRing R K
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : FiniteDimensional K L
    inst✝⁸ : Algebra.IsSeparable K L
    algRL : Algebra R L
    inst✝⁷ : IsScalarTower R K L
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra S L
    ist : IsScalarTower R S L
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝² : Infinite R
    inst✝¹ : DecidableEq R
    inst✝ : IsIntegralClosure S R L
    ⊢ Fintype (ClassGroup S)
  -/
  letI := Classical.decEq L
  /-
    R : Type u_1
    S : Type u_2
    K : Type u_3
    L : Type u_4
    inst✝¹⁷ : EuclideanDomain R
    inst✝¹⁶ : CommRing S
    inst✝¹⁵ : IsDomain S
    inst✝¹⁴ : Field K
    inst✝¹³ : Field L
    inst✝¹² : Algebra R K
    inst✝¹¹ : IsFractionRing R K
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : FiniteDimensional K L
    inst✝⁸ : Algebra.IsSeparable K L
    algRL : Algebra R L
    inst✝⁷ : IsScalarTower R K L
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra S L
    ist : IsScalarTower R S L
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝² : Infinite R
    inst✝¹ : DecidableEq R
    inst✝ : IsIntegralClosure S R L
    this : DecidableEq L := Classical.decEq L
    ⊢ Fintype (ClassGroup S)
  -/
  letI := IsIntegralClosure.isFractionRing_of_finite_extension R K L S
  /-
    R : Type u_1
    S : Type u_2
    K : Type u_3
    L : Type u_4
    inst✝¹⁷ : EuclideanDomain R
    inst✝¹⁶ : CommRing S
    inst✝¹⁵ : IsDomain S
    inst✝¹⁴ : Field K
    inst✝¹³ : Field L
    inst✝¹² : Algebra R K
    inst✝¹¹ : IsFractionRing R K
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : FiniteDimensional K L
    inst✝⁸ : Algebra.IsSeparable K L
    algRL : Algebra R L
    inst✝⁷ : IsScalarTower R K L
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra S L
    ist : IsScalarTower R S L
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝² : Infinite R
    inst✝¹ : DecidableEq R
    inst✝ : IsIntegralClosure S R L
    this✝ : DecidableEq L := Classical.decEq L
    this : IsFractionRing S L := IsIntegralClosure.isFractionRing_of_finite_extens …
    ⊢ Fintype (ClassGroup S)
  -/
  letI := IsIntegralClosure.isDedekindDomain R K L S
  /-
    R : Type u_1
    S : Type u_2
    K : Type u_3
    L : Type u_4
    inst✝¹⁷ : EuclideanDomain R
    inst✝¹⁶ : CommRing S
    inst✝¹⁵ : IsDomain S
    inst✝¹⁴ : Field K
    inst✝¹³ : Field L
    inst✝¹² : Algebra R K
    inst✝¹¹ : IsFractionRing R K
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : FiniteDimensional K L
    inst✝⁸ : Algebra.IsSeparable K L
    algRL : Algebra R L
    inst✝⁷ : IsScalarTower R K L
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra S L
    ist : IsScalarTower R S L
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝² : Infinite R
    inst✝¹ : DecidableEq R
    inst✝ : IsIntegralClosure S R L
    this✝¹ : DecidableEq L := Classical.decEq L
    this✝ : IsFractionRing S L := IsIntegralClosure.isFractionRing_of_finite_exten …
    this : IsDedekindDomain S := IsIntegralClosure.isDedekindDomain R K L S
    ⊢ Fintype (ClassGroup S)
  -/
  choose s b hb_int using FiniteDimensional.exists_is_basis_integral R K L
-- Porting note: `this` and `f` below where solved at the end rather than being defined at first.
  have : LinearIndependent R ((Algebra.traceForm K L).dualBasis
      (traceForm_nondegenerate K L) b) := by
    apply (Basis.linearIndependent _).restrict_scalars
    simp only [Algebra.smul_def, mul_one]
    apply IsFractionRing.injective
  obtain ⟨n, b⟩ :=
    Submodule.basisOfPidOfLESpan this (IsIntegralClosure.range_le_span_dualBasis S b hb_int)
  let f : (S ⧸ LinearMap.ker (LinearMap.restrictScalars R (Algebra.linearMap S L))) ≃ₗ[R] S := by
    rw [LinearMap.ker_eq_bot.mpr]
    · exact Submodule.quotEquivOfEqBot _ rfl
    · exact IsIntegralClosure.algebraMap_injective _ R _
  /-
    case mk
    R : Type u_1
    S : Type u_2
    K : Type u_3
    L : Type u_4
    inst✝¹⁷ : EuclideanDomain R
    inst✝¹⁶ : CommRing S
    inst✝¹⁵ : IsDomain S
    inst✝¹⁴ : Field K
    inst✝¹³ : Field L
    inst✝¹² : Algebra R K
    inst✝¹¹ : IsFractionRing R K
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : FiniteDimensional K L
    inst✝⁸ : Algebra.IsSeparable K L
    algRL : Algebra R L
    inst✝⁷ : IsScalarTower R K L
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra S L
    ist : IsScalarTower R S L
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    bS : Basis ι R S
    adm : abv.IsAdmissible
    inst✝² : Infinite R
    inst✝¹ : DecidableEq R
    inst✝ : IsIntegralClosure S R L
    this✝² : DecidableEq L := Classical.decEq L
    this✝¹ : IsFractionRing S L := IsIntegralClosure.isFractionRing_of_finite_exte …
    this✝ : IsDedekindDomain S := IsIntegralClosure.isDedekindDomain R K L S
    s : Finset L
    b✝ : Basis (Subtype fun x => Membership.mem s x) K L
    hb_int : ∀ (x : Subtype fun x => Membership.mem s x), IsIntegral R (b✝ x)
    this : LinearIndependent R ⇑((Algebra.traceForm K L).dualBasis ⋯ b✝)
    n : Nat
    b : Basis (Fin n) R (Subtype fun x => Membership.mem (LinearMap.range (↑R (Alg …
    f : LinearEquiv (RingHom.id R) (HasQuotient.Quotient S (LinearMap.ker (↑R (Alg …
    ⊢ Fintype (ClassGroup S)
  -/
  let bS := b.map ((LinearMap.quotKerEquivRange _).symm ≪≫ₗ f)
  /-
    case mk
    R : Type u_1
    S : Type u_2
    K : Type u_3
    L : Type u_4
    inst✝¹⁷ : EuclideanDomain R
    inst✝¹⁶ : CommRing S
    inst✝¹⁵ : IsDomain S
    inst✝¹⁴ : Field K
    inst✝¹³ : Field L
    inst✝¹² : Algebra R K
    inst✝¹¹ : IsFractionRing R K
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : FiniteDimensional K L
    inst✝⁸ : Algebra.IsSeparable K L
    algRL : Algebra R L
    inst✝⁷ : IsScalarTower R K L
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra S L
    ist : IsScalarTower R S L
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    bS✝ : Basis ι R S
    adm : abv.IsAdmissible
    inst✝² : Infinite R
    inst✝¹ : DecidableEq R
    inst✝ : IsIntegralClosure S R L
    this✝² : DecidableEq L := Classical.decEq L
    this✝¹ : IsFractionRing S L := IsIntegralClosure.isFractionRing_of_finite_exte …
    this✝ : IsDedekindDomain S := IsIntegralClosure.isDedekindDomain R K L S
    s : Finset L
    b✝ : Basis (Subtype fun x => Membership.mem s x) K L
    hb_int : ∀ (x : Subtype fun x => Membership.mem s x), IsIntegral R (b✝ x)
    this : LinearIndependent R ⇑((Algebra.traceForm K L).dualBasis ⋯ b✝)
    n : Nat
    b : Basis (Fin n) R (Subtype fun x => Membership.mem (LinearMap.range (↑R (Alg …
    f : LinearEquiv (RingHom.id R) (HasQuotient.Quotient S (LinearMap.ker (↑R (Alg …
    bS : Basis (Fin n) R S := b.map ((↑R (Algebra.linearMap S L)).quotKerEquivRang …
    ⊢ Fintype (ClassGroup S)
  -/
  have : Algebra.IsIntegral R S := IsIntegralClosure.isIntegral_algebra R L
  /-
    case mk
    R : Type u_1
    S : Type u_2
    K : Type u_3
    L : Type u_4
    inst✝¹⁷ : EuclideanDomain R
    inst✝¹⁶ : CommRing S
    inst✝¹⁵ : IsDomain S
    inst✝¹⁴ : Field K
    inst✝¹³ : Field L
    inst✝¹² : Algebra R K
    inst✝¹¹ : IsFractionRing R K
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : FiniteDimensional K L
    inst✝⁸ : Algebra.IsSeparable K L
    algRL : Algebra R L
    inst✝⁷ : IsScalarTower R K L
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra S L
    ist : IsScalarTower R S L
    abv : AbsoluteValue R Int
    ι : Type u_5
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    bS✝ : Basis ι R S
    adm : abv.IsAdmissible
    inst✝² : Infinite R
    inst✝¹ : DecidableEq R
    inst✝ : IsIntegralClosure S R L
    this✝³ : DecidableEq L := Classical.decEq L
    this✝² : IsFractionRing S L := IsIntegralClosure.isFractionRing_of_finite_exte …
    this✝¹ : IsDedekindDomain S := IsIntegralClosure.isDedekindDomain R K L S
    s : Finset L
    b✝ : Basis (Subtype fun x => Membership.mem s x) K L
    hb_int : ∀ (x : Subtype fun x => Membership.mem s x), IsIntegral R (b✝ x)
    this✝ : LinearIndependent R ⇑((Algebra.traceForm K L).dualBasis ⋯ b✝)
    n : Nat
    b : Basis (Fin n) R (Subtype fun x => Membership.mem (LinearMap.range (↑R (Alg …
    f : LinearEquiv (RingHom.id R) (HasQuotient.Quotient S (LinearMap.ker (↑R (Alg …
    bS : Basis (Fin n) R S := b.map ((↑R (Algebra.linearMap S L)).quotKerEquivRang …
    this : Algebra.IsIntegral R S
    ⊢ Fintype (ClassGroup S)
  -/
  exact fintypeOfAdmissibleOfAlgebraic bS adm
  /-
    🎉 no goals
  -/


