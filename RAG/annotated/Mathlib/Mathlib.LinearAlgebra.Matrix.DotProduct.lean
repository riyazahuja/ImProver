set_option linter.deprecated false in
@[simp, deprecated Matrix.dotProduct_single (since := "2024-08-09")]
theorem dotProduct_stdBasis_eq_mul [DecidableEq n] (v : n → R) (c : R) (i : n) :
    dotProduct v (LinearMap.stdBasis R (fun _ => R) i c) = v i * c :=
  dotProduct_single ..


set_option linter.deprecated false in
@[deprecated Matrix.dotProduct_single_one (since := "2024-08-09")]
theorem dotProduct_stdBasis_one [DecidableEq n] (v : n → R) (i : n) :
    dotProduct v (LinearMap.stdBasis R (fun _ => R) i 1) = v i :=
  dotProduct_single_one ..


theorem dotProduct_eq (v w : n → R) (h : ∀ u, dotProduct v u = dotProduct w u) : v = w := by
  /-
    n : Type u_2
    R : Type u_4
    inst✝¹ : Semiring R
    inst✝ : Fintype n
    v w : n → R
    h : ∀ (u : n → R), Eq (dotProduct v u) (dotProduct w u)
    ⊢ Eq v w
  -/
  funext x
  /-
    case h
    n : Type u_2
    R : Type u_4
    inst✝¹ : Semiring R
    inst✝ : Fintype n
    v w : n → R
    h : ∀ (u : n → R), Eq (dotProduct v u) (dotProduct w u)
    x : n
    ⊢ Eq (v x) (w x)
  -/
  classical rw [← dotProduct_single_one v x, ← dotProduct_single_one w x, h]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.dotProduct_eq := dotProduct_eq


theorem dotProduct_eq_iff {v w : n → R} : (∀ u, dotProduct v u = dotProduct w u) ↔ v = w :=
  ⟨fun h => dotProduct_eq v w h, fun h _ => h ▸ rfl⟩


@[deprecated (since := "2024-12-12")] protected alias Matrix.dotProduct_eq_iff := dotProduct_eq_iff


theorem dotProduct_eq_zero (v : n → R) (h : ∀ w, dotProduct v w = 0) : v = 0 :=
  dotProduct_eq _ _ fun u => (h u).symm ▸ (zero_dotProduct u).symm


@[deprecated (since := "2024-12-12")]
protected alias Matrix.dotProduct_eq_zero := dotProduct_eq_zero


theorem dotProduct_eq_zero_iff {v : n → R} : (∀ w, dotProduct v w = 0) ↔ v = 0 :=
  ⟨fun h => dotProduct_eq_zero v h, fun h w => h.symm ▸ zero_dotProduct w⟩


@[deprecated (since := "2024-12-12")]
protected alias Matrix.dotProduct_eq_zero_iff := dotProduct_eq_zero_iff


lemma dotProduct_nonneg_of_nonneg {v w : n → R} (hv : 0 ≤ v) (hw : 0 ≤ w) : 0 ≤ dotProduct v w :=
  Finset.sum_nonneg (fun i _ => mul_nonneg (hv i) (hw i))


@[deprecated (since := "2024-12-12")]
protected alias Matrix.dotProduct_nonneg_of_nonneg := dotProduct_nonneg_of_nonneg


lemma dotProduct_le_dotProduct_of_nonneg_right {u v w : n → R} (huv : u ≤ v) (hw : 0 ≤ w) :
    dotProduct u w ≤ dotProduct v w :=
  Finset.sum_le_sum (fun i _ => mul_le_mul_of_nonneg_right (huv i) (hw i))


@[deprecated (since := "2024-12-12")]
protected alias Matrix.dotProduct_le_dotProduct_of_nonneg_right :=
  dotProduct_le_dotProduct_of_nonneg_right


lemma dotProduct_le_dotProduct_of_nonneg_left {u v w : n → R} (huv : u ≤ v) (hw : 0 ≤ w) :
    dotProduct w u ≤ dotProduct w v :=
  Finset.sum_le_sum (fun i _ => mul_le_mul_of_nonneg_left (huv i) (hw i))


@[deprecated (since := "2024-12-12")]
protected alias Matrix.dotProduct_le_dotProduct_of_nonneg_left :=
  dotProduct_le_dotProduct_of_nonneg_left


@[simp]
theorem dotProduct_self_eq_zero [LinearOrderedRing R] {v : n → R} : dotProduct v v = 0 ↔ v = 0 :=
  (Finset.sum_eq_zero_iff_of_nonneg fun i _ => mul_self_nonneg (v i)).trans <| by
    /-
      n : Type u_2
      R : Type u_4
      inst✝¹ : Fintype n
      inst✝ : LinearOrderedRing R
      v : n → R
      ⊢ Iff (∀ (i : n), Membership.mem Finset.univ i → Eq (HMul.hMul (v i) (v i)) 0) …
    -/
    simp [funext_iff]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-12")]
protected alias Matrix.dotProduct_self_eq_zero := dotProduct_self_eq_zero


/-- Note that this applies to `ℂ` via `RCLike.toStarOrderedRing`. -/
@[simp]
theorem dotProduct_star_self_nonneg (v : n → R) : 0 ≤ dotProduct (star v) v :=
  Fintype.sum_nonneg fun _ => star_mul_self_nonneg _


@[deprecated (since := "2024-12-12")]
protected alias Matrix.dotProduct_star_self_nonneg := dotProduct_star_self_nonneg


/-- Note that this applies to `ℂ` via `RCLike.toStarOrderedRing`. -/
@[simp]
theorem dotProduct_self_star_nonneg (v : n → R) : 0 ≤ dotProduct v (star v) :=
  Fintype.sum_nonneg fun _ => mul_star_self_nonneg _


@[deprecated (since := "2024-12-12")]
protected alias Matrix.dotProduct_self_star_nonneg := dotProduct_self_star_nonneg


/-- Note that this applies to `ℂ` via `RCLike.toStarOrderedRing`. -/
@[simp]
theorem dotProduct_star_self_eq_zero {v : n → R} : dotProduct (star v) v = 0 ↔ v = 0 :=
  (Fintype.sum_eq_zero_iff_of_nonneg fun _ => star_mul_self_nonneg _).trans <|
       /-
         n : Type u_2
         R : Type u_4
         inst✝⁵ : Fintype n
         inst✝⁴ : PartialOrder R
         inst✝³ : NonUnitalRing R
         inst✝² : StarRing R
         inst✝¹ : StarOrderedRing R
         inst✝ : NoZeroDivisors R
         v : n → R
         ⊢ Iff (Eq (fun i => HMul.hMul (Star.star v i) (v i)) 0) (Eq v 0)
       -/
    by simp [funext_iff, mul_eq_zero]
       /-
         🎉 no goals
       -/


@[deprecated (since := "2024-12-12")]
protected alias Matrix.dotProduct_star_self_eq_zero := dotProduct_star_self_eq_zero


/-- Note that this applies to `ℂ` via `RCLike.toStarOrderedRing`. -/
@[simp]
theorem dotProduct_self_star_eq_zero {v : n → R} : dotProduct v (star v) = 0 ↔ v = 0 :=
  (Fintype.sum_eq_zero_iff_of_nonneg fun _ => mul_star_self_nonneg _).trans <|
       /-
         n : Type u_2
         R : Type u_4
         inst✝⁵ : Fintype n
         inst✝⁴ : PartialOrder R
         inst✝³ : NonUnitalRing R
         inst✝² : StarRing R
         inst✝¹ : StarOrderedRing R
         inst✝ : NoZeroDivisors R
         v : n → R
         ⊢ Iff (Eq (fun i => HMul.hMul (v i) (Star.star v i)) 0) (Eq v 0)
       -/
    by simp [funext_iff, mul_eq_zero]
       /-
         🎉 no goals
       -/


@[deprecated (since := "2024-12-12")]
protected alias Matrix.dotProduct_self_star_eq_zero := dotProduct_self_star_eq_zero


@[simp]
lemma conjTranspose_mul_self_eq_zero {n} {A : Matrix m n R} : Aᴴ * A = 0 ↔ A = 0 :=
  ⟨fun h => Matrix.ext fun i j =>
    (congr_fun <| dotProduct_star_self_eq_zero.1 <| Matrix.ext_iff.2 h j j) i,
  fun h => h ▸ Matrix.mul_zero _⟩


@[simp]
lemma self_mul_conjTranspose_eq_zero {m} {A : Matrix m n R} : A * Aᴴ = 0 ↔ A = 0 :=
  ⟨fun h => Matrix.ext fun i j =>
    (congr_fun <| dotProduct_self_star_eq_zero.1 <| Matrix.ext_iff.2 h i i) j,
  fun h => h ▸ Matrix.zero_mul _⟩


lemma conjTranspose_mul_self_mul_eq_zero {p} (A : Matrix m n R) (B : Matrix n p R) :
    (Aᴴ * A) * B = 0 ↔ A * B = 0 := by
  /-
    m : Type u_1
    n : Type u_2
    R : Type u_4
    inst✝⁶ : Fintype m
    inst✝⁵ : Fintype n
    inst✝⁴ : PartialOrder R
    inst✝³ : NonUnitalRing R
    inst✝² : StarRing R
    inst✝¹ : StarOrderedRing R
    inst✝ : NoZeroDivisors R
    p : Type u_5
    A : Matrix m n R
    B : Matrix n p R
    ⊢ Iff (Eq (HMul.hMul (HMul.hMul A.conjTranspose A) B) 0) (Eq (HMul.hMul A B) 0)
  -/
  refine ⟨fun h => ?_, fun h => by simp only [Matrix.mul_assoc, h, Matrix.mul_zero]⟩
  /-
    m : Type u_1
    n : Type u_2
    R : Type u_4
    inst✝⁶ : Fintype m
    inst✝⁵ : Fintype n
    inst✝⁴ : PartialOrder R
    inst✝³ : NonUnitalRing R
    inst✝² : StarRing R
    inst✝¹ : StarOrderedRing R
    inst✝ : NoZeroDivisors R
    p : Type u_5
    A : Matrix m n R
    B : Matrix n p R
    h : Eq (HMul.hMul (HMul.hMul A.conjTranspose A) B) 0
    ⊢ Eq (HMul.hMul A B) 0
  -/
  apply_fun (Bᴴ * ·) at h
  rwa [Matrix.mul_zero, Matrix.mul_assoc, ← Matrix.mul_assoc, ← conjTranspose_mul,
    conjTranspose_mul_self_eq_zero] at h


lemma self_mul_conjTranspose_mul_eq_zero {p} (A : Matrix m n R) (B : Matrix m p R) :
    (A * Aᴴ) * B = 0 ↔ Aᴴ * B = 0 := by
  /-
    m : Type u_1
    n : Type u_2
    R : Type u_4
    inst✝⁶ : Fintype m
    inst✝⁵ : Fintype n
    inst✝⁴ : PartialOrder R
    inst✝³ : NonUnitalRing R
    inst✝² : StarRing R
    inst✝¹ : StarOrderedRing R
    inst✝ : NoZeroDivisors R
    p : Type u_5
    A : Matrix m n R
    B : Matrix m p R
    ⊢ Iff (Eq (HMul.hMul (HMul.hMul A A.conjTranspose) B) 0) (Eq (HMul.hMul A.conj …
  -/
  simpa only [conjTranspose_conjTranspose] using conjTranspose_mul_self_mul_eq_zero Aᴴ _
  /-
    🎉 no goals
  -/


lemma mul_self_mul_conjTranspose_eq_zero {p} (A : Matrix m n R) (B : Matrix p m R) :
    B * (A * Aᴴ) = 0 ↔ B * A = 0 := by
  rw [← conjTranspose_eq_zero, conjTranspose_mul, conjTranspose_mul, conjTranspose_conjTranspose,
    self_mul_conjTranspose_mul_eq_zero, ← conjTranspose_mul, conjTranspose_eq_zero]


lemma mul_conjTranspose_mul_self_eq_zero {p} (A : Matrix m n R) (B : Matrix p n R) :
    B * (Aᴴ * A) = 0 ↔ B * Aᴴ = 0 := by
  /-
    m : Type u_1
    n : Type u_2
    R : Type u_4
    inst✝⁶ : Fintype m
    inst✝⁵ : Fintype n
    inst✝⁴ : PartialOrder R
    inst✝³ : NonUnitalRing R
    inst✝² : StarRing R
    inst✝¹ : StarOrderedRing R
    inst✝ : NoZeroDivisors R
    p : Type u_5
    A : Matrix m n R
    B : Matrix p n R
    ⊢ Iff (Eq (HMul.hMul B (HMul.hMul A.conjTranspose A)) 0) (Eq (HMul.hMul B A.co …
  -/
  simpa only [conjTranspose_conjTranspose] using mul_self_mul_conjTranspose_eq_zero Aᴴ _
  /-
    🎉 no goals
  -/


lemma conjTranspose_mul_self_mulVec_eq_zero (A : Matrix m n R) (v : n → R) :
    (Aᴴ * A) *ᵥ v = 0 ↔ A *ᵥ v = 0 := by
  simpa only [← Matrix.col_mulVec, col_eq_zero] using
    conjTranspose_mul_self_mul_eq_zero A (col (Fin 1) v)


lemma self_mul_conjTranspose_mulVec_eq_zero (A : Matrix m n R) (v : m → R) :
    (A * Aᴴ) *ᵥ v = 0 ↔ Aᴴ *ᵥ v = 0 := by
  /-
    m : Type u_1
    n : Type u_2
    R : Type u_4
    inst✝⁶ : Fintype m
    inst✝⁵ : Fintype n
    inst✝⁴ : PartialOrder R
    inst✝³ : NonUnitalRing R
    inst✝² : StarRing R
    inst✝¹ : StarOrderedRing R
    inst✝ : NoZeroDivisors R
    A : Matrix m n R
    v : m → R
    ⊢ Iff (Eq ((HMul.hMul A A.conjTranspose).mulVec v) 0) (Eq (A.conjTranspose.mul …
  -/
  simpa only [conjTranspose_conjTranspose] using conjTranspose_mul_self_mulVec_eq_zero Aᴴ _
  /-
    🎉 no goals
  -/


lemma vecMul_conjTranspose_mul_self_eq_zero (A : Matrix m n R) (v : n → R) :
    v ᵥ* (Aᴴ * A) = 0 ↔ v ᵥ* Aᴴ = 0 := by
  simpa only [← Matrix.row_vecMul, row_eq_zero] using
    mul_conjTranspose_mul_self_eq_zero A (row (Fin 1) v)


lemma vecMul_self_mul_conjTranspose_eq_zero (A : Matrix m n R) (v : m → R) :
    v ᵥ* (A * Aᴴ) = 0 ↔ v ᵥ* A = 0 := by
  /-
    m : Type u_1
    n : Type u_2
    R : Type u_4
    inst✝⁶ : Fintype m
    inst✝⁵ : Fintype n
    inst✝⁴ : PartialOrder R
    inst✝³ : NonUnitalRing R
    inst✝² : StarRing R
    inst✝¹ : StarOrderedRing R
    inst✝ : NoZeroDivisors R
    A : Matrix m n R
    v : m → R
    ⊢ Iff (Eq (Matrix.vecMul v (HMul.hMul A A.conjTranspose)) 0) (Eq (Matrix.vecMu …
  -/
  simpa only [conjTranspose_conjTranspose] using vecMul_conjTranspose_mul_self_eq_zero Aᴴ _
  /-
    🎉 no goals
  -/


/-- Note that this applies to `ℂ` via `RCLike.toStarOrderedRing`. -/
@[simp]
theorem dotProduct_star_self_pos_iff {v : n → R} :
    0 < dotProduct (star v) v ↔ v ≠ 0 := by
  /-
    n : Type u_2
    R : Type u_4
    inst✝⁵ : Fintype n
    inst✝⁴ : PartialOrder R
    inst✝³ : NonUnitalRing R
    inst✝² : StarRing R
    inst✝¹ : StarOrderedRing R
    inst✝ : NoZeroDivisors R
    v : n → R
    ⊢ Iff (LT.lt 0 (dotProduct (Star.star v) v)) (Ne v 0)
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      n : Type u_2
      R : Type u_4
      inst✝⁵ : Fintype n
      inst✝⁴ : PartialOrder R
      inst✝³ : NonUnitalRing R
      inst✝² : StarRing R
      inst✝¹ : StarOrderedRing R
      inst✝ : NoZeroDivisors R
      v : n → R
      h✝ : Subsingleton R
      ⊢ Iff (LT.lt 0 (dotProduct (Star.star v) v)) (Ne v 0)
    -/
  · obtain rfl : v = 0 := Subsingleton.elim _ _
    /-
      case inl
      n : Type u_2
      R : Type u_4
      inst✝⁵ : Fintype n
      inst✝⁴ : PartialOrder R
      inst✝³ : NonUnitalRing R
      inst✝² : StarRing R
      inst✝¹ : StarOrderedRing R
      inst✝ : NoZeroDivisors R
      h✝ : Subsingleton R
      ⊢ Iff (LT.lt 0 (dotProduct (Star.star 0) 0)) (Ne 0 0)
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Type u_2
    R : Type u_4
    inst✝⁵ : Fintype n
    inst✝⁴ : PartialOrder R
    inst✝³ : NonUnitalRing R
    inst✝² : StarRing R
    inst✝¹ : StarOrderedRing R
    inst✝ : NoZeroDivisors R
    v : n → R
    h✝ : Nontrivial R
    ⊢ Iff (LT.lt 0 (dotProduct (Star.star v) v)) (Ne v 0)
  -/
  refine (Fintype.sum_pos_iff_of_nonneg fun i => star_mul_self_nonneg _).trans ?_
  /-
    case inr
    n : Type u_2
    R : Type u_4
    inst✝⁵ : Fintype n
    inst✝⁴ : PartialOrder R
    inst✝³ : NonUnitalRing R
    inst✝² : StarRing R
    inst✝¹ : StarOrderedRing R
    inst✝ : NoZeroDivisors R
    v : n → R
    h✝ : Nontrivial R
    ⊢ Iff (LT.lt 0 fun i => HMul.hMul (Star.star v i) (v i)) (Ne v 0)
  -/
  simp_rw [Pi.lt_def, Function.ne_iff, Pi.zero_apply]
  /-
    case inr
    n : Type u_2
    R : Type u_4
    inst✝⁵ : Fintype n
    inst✝⁴ : PartialOrder R
    inst✝³ : NonUnitalRing R
    inst✝² : StarRing R
    inst✝¹ : StarOrderedRing R
    inst✝ : NoZeroDivisors R
    v : n → R
    h✝ : Nontrivial R
    ⊢ Iff (And (LE.le 0 fun i => HMul.hMul (Star.star v i) (v i)) (Exists fun i => …
  -/
  refine (and_iff_right fun i => star_mul_self_nonneg (v i)).trans <| exists_congr fun i => ?_
  /-
    case inr
    n : Type u_2
    R : Type u_4
    inst✝⁵ : Fintype n
    inst✝⁴ : PartialOrder R
    inst✝³ : NonUnitalRing R
    inst✝² : StarRing R
    inst✝¹ : StarOrderedRing R
    inst✝ : NoZeroDivisors R
    v : n → R
    h✝ : Nontrivial R
    i : n
    ⊢ Iff (LT.lt 0 (HMul.hMul (Star.star v i) (v i))) (Ne (v i) 0)
  -/
  constructor
    /-
      case inr.mp
      n : Type u_2
      R : Type u_4
      inst✝⁵ : Fintype n
      inst✝⁴ : PartialOrder R
      inst✝³ : NonUnitalRing R
      inst✝² : StarRing R
      inst✝¹ : StarOrderedRing R
      inst✝ : NoZeroDivisors R
      v : n → R
      h✝ : Nontrivial R
      i : n
      ⊢ LT.lt 0 (HMul.hMul (Star.star v i) (v i)) → Ne (v i) 0
    -/
  · rintro h hv
    /-
      case inr.mp
      n : Type u_2
      R : Type u_4
      inst✝⁵ : Fintype n
      inst✝⁴ : PartialOrder R
      inst✝³ : NonUnitalRing R
      inst✝² : StarRing R
      inst✝¹ : StarOrderedRing R
      inst✝ : NoZeroDivisors R
      v : n → R
      h✝ : Nontrivial R
      i : n
      h : LT.lt 0 (HMul.hMul (Star.star v i) (v i))
      hv : Eq (v i) 0
      ⊢ False
    -/
    simp [hv] at h
    /-
      🎉 no goals
    -/
    /-
      case inr.mpr
      n : Type u_2
      R : Type u_4
      inst✝⁵ : Fintype n
      inst✝⁴ : PartialOrder R
      inst✝³ : NonUnitalRing R
      inst✝² : StarRing R
      inst✝¹ : StarOrderedRing R
      inst✝ : NoZeroDivisors R
      v : n → R
      h✝ : Nontrivial R
      i : n
      ⊢ Ne (v i) 0 → LT.lt 0 (HMul.hMul (Star.star v i) (v i))
    -/
  · exact (star_mul_self_pos <| isRegular_of_ne_zero ·)
    /-
      🎉 no goals
    -/


/-- Note that this applies to `ℂ` via `RCLike.toStarOrderedRing`. -/
@[simp]
theorem dotProduct_self_star_pos_iff {v : n → R} : 0 < dotProduct v (star v) ↔ v ≠ 0 := by
  /-
    n : Type u_2
    R : Type u_4
    inst✝⁵ : Fintype n
    inst✝⁴ : PartialOrder R
    inst✝³ : NonUnitalRing R
    inst✝² : StarRing R
    inst✝¹ : StarOrderedRing R
    inst✝ : NoZeroDivisors R
    v : n → R
    ⊢ Iff (LT.lt 0 (dotProduct v (Star.star v))) (Ne v 0)
  -/
  simpa using dotProduct_star_self_pos_iff (v := star v)
  /-
    🎉 no goals
  -/


