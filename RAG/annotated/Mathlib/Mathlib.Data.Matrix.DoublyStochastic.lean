/--
A square matrix is doubly stochastic iff all entries are nonnegative, and left or right
multiplication by the vector of all 1s gives the vector of all 1s.
-/
def doublyStochastic (R n : Type*) [Fintype n] [DecidableEq n] [OrderedSemiring R] :
    Submonoid (Matrix n n R) where
  carrier := {M | (∀ i j, 0 ≤ M i j) ∧ M *ᵥ 1 = 1 ∧ 1 ᵥ* M = 1 }
  mul_mem' {M N} hM hN := by
    /-
      R✝ : Type u_1
      n✝ : Type u_2
      inst✝⁵ : Fintype n✝
      inst✝⁴ : DecidableEq n✝
      inst✝³ : OrderedSemiring R✝
      M✝ : Matrix n✝ n✝ R✝
      R : Type u_3
      n : Type u_4
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : OrderedSemiring R
      M N : Matrix n n R
      hM : Membership.mem (setOf fun M => And (∀ (i j : n), LE.le 0 (M i j)) (And (E …
      hN : Membership.mem (setOf fun M => And (∀ (i j : n), LE.le 0 (M i j)) (And (E …
      ⊢ Membership.mem (setOf fun M => And (∀ (i j : n), LE.le 0 (M i j)) (And (Eq ( …
    -/
    refine ⟨fun i j => sum_nonneg fun i _ => mul_nonneg (hM.1 _ _) (hN.1 _ _), ?_, ?_⟩
    /-
      case refine_1
      R✝ : Type u_1
      n✝ : Type u_2
      inst✝⁵ : Fintype n✝
      inst✝⁴ : DecidableEq n✝
      inst✝³ : OrderedSemiring R✝
      M✝ : Matrix n✝ n✝ R✝
      R : Type u_3
      n : Type u_4
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : OrderedSemiring R
      M N : Matrix n n R
      hM : Membership.mem (setOf fun M => And (∀ (i j : n), LE.le 0 (M i j)) (And (E …
      hN : Membership.mem (setOf fun M => And (∀ (i j : n), LE.le 0 (M i j)) (And (E …
      ⊢ Eq ((HMul.hMul M N).mulVec 1) 1
    -/
    next => rw [← mulVec_mulVec, hN.2.1, hM.2.1]
    /-
      case refine_2
      R✝ : Type u_1
      n✝ : Type u_2
      inst✝⁵ : Fintype n✝
      inst✝⁴ : DecidableEq n✝
      inst✝³ : OrderedSemiring R✝
      M✝ : Matrix n✝ n✝ R✝
      R : Type u_3
      n : Type u_4
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : OrderedSemiring R
      M N : Matrix n n R
      hM : Membership.mem (setOf fun M => And (∀ (i j : n), LE.le 0 (M i j)) (And (E …
      hN : Membership.mem (setOf fun M => And (∀ (i j : n), LE.le 0 (M i j)) (And (E …
      ⊢ Eq (Matrix.vecMul 1 (HMul.hMul M N)) 1
    -/
    next => rw [← vecMul_vecMul, hM.2.2, hN.2.2]
    /-
      🎉 no goals
    -/
                 /-
                   R✝ : Type u_1
                   n✝ : Type u_2
                   inst✝⁵ : Fintype n✝
                   inst✝⁴ : DecidableEq n✝
                   inst✝³ : OrderedSemiring R✝
                   M : Matrix n✝ n✝ R✝
                   R : Type u_3
                   n : Type u_4
                   inst✝² : Fintype n
                   inst✝¹ : DecidableEq n
                   inst✝ : OrderedSemiring R
                   ⊢ Membership.mem { carrier := setOf fun M => And (∀ (i j : n), LE.le 0 (M i j) …
                 -/
  one_mem' := by simp [zero_le_one_elem]
                 /-
                   🎉 no goals
                 -/


lemma mem_doublyStochastic :
    M ∈ doublyStochastic R n ↔ (∀ i j, 0 ≤ M i j) ∧ M *ᵥ 1 = 1 ∧ 1 ᵥ* M = 1 :=
  Iff.rfl


lemma mem_doublyStochastic_iff_sum :
    M ∈ doublyStochastic R n ↔
      (∀ i j, 0 ≤ M i j) ∧ (∀ i, ∑ j, M i j = 1) ∧ ∀ j, ∑ i, M i j = 1 := by
  /-
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : OrderedSemiring R
    M : Matrix n n R
    ⊢ Iff (Membership.mem (doublyStochastic R n) M) (And (∀ (i j : n), LE.le 0 (M  …
  -/
  simp [funext_iff, doublyStochastic, mulVec, vecMul, dotProduct]
  /-
    🎉 no goals
  -/


/-- Every entry of a doubly stochastic matrix is nonnegative. -/
lemma nonneg_of_mem_doublyStochastic (hM : M ∈ doublyStochastic R n) {i j : n} : 0 ≤ M i j :=
  hM.1 _ _


/-- Each row sum of a doubly stochastic matrix is 1. -/
lemma sum_row_of_mem_doublyStochastic (hM : M ∈ doublyStochastic R n) (i : n) : ∑ j, M i j = 1 :=
  (mem_doublyStochastic_iff_sum.1 hM).2.1 _


/-- Each column sum of a doubly stochastic matrix is 1. -/
lemma sum_col_of_mem_doublyStochastic (hM : M ∈ doublyStochastic R n) (j : n) : ∑ i, M i j = 1 :=
  (mem_doublyStochastic_iff_sum.1 hM).2.2 _


/-- A doubly stochastic matrix multiplied with the all-ones column vector is 1. -/
lemma mulVec_one_of_mem_doublyStochastic (hM : M ∈ doublyStochastic R n) : M *ᵥ 1 = 1 :=
  (mem_doublyStochastic.1 hM).2.1


/-- The all-ones row vector multiplied with a doubly stochastic matrix is 1. -/
lemma one_vecMul_of_mem_doublyStochastic (hM : M ∈ doublyStochastic R n) : 1 ᵥ* M = 1 :=
  (mem_doublyStochastic.1 hM).2.2


/-- Every entry of a doubly stochastic matrix is less than or equal to 1. -/
lemma le_one_of_mem_doublyStochastic (hM : M ∈ doublyStochastic R n) {i j : n} :
    M i j ≤ 1 := by
  /-
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : OrderedSemiring R
    M : Matrix n n R
    hM : Membership.mem (doublyStochastic R n) M
    i j : n
    ⊢ LE.le (M i j) 1
  -/
  rw [← sum_row_of_mem_doublyStochastic hM i]
  /-
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : OrderedSemiring R
    M : Matrix n n R
    hM : Membership.mem (doublyStochastic R n) M
    i j : n
    ⊢ LE.le (M i j) (Finset.univ.sum fun j => M i j)
  -/
  exact single_le_sum (fun k _ => hM.1 _ k) (mem_univ j)
  /-
    🎉 no goals
  -/


/-- The set of doubly stochastic matrices is convex. -/
lemma convex_doublyStochastic : Convex R (doublyStochastic R n : Set (Matrix n n R)) := by
  /-
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : OrderedSemiring R
    ⊢ Convex R ↑(doublyStochastic R n)
  -/
  intro x hx y hy a b ha hb h
  /-
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : OrderedSemiring R
    x : Matrix n n R
    hx : Membership.mem (↑(doublyStochastic R n)) x
    y : Matrix n n R
    hy : Membership.mem (↑(doublyStochastic R n)) y
    a b : R
    ha : LE.le 0 a
    hb : LE.le 0 b
    h : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (↑(doublyStochastic R n)) (HAdd.hAdd (HSMul.hSMul a x) (HSMul …
  -/
  simp only [SetLike.mem_coe, mem_doublyStochastic_iff_sum] at hx hy ⊢
  /-
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : OrderedSemiring R
    x y : Matrix n n R
    a b : R
    ha : LE.le 0 a
    hb : LE.le 0 b
    h : Eq (HAdd.hAdd a b) 1
    hx : And (∀ (i j : n), LE.le 0 (x i j)) (And (∀ (i : n), Eq (Finset.univ.sum f …
    hy : And (∀ (i j : n), LE.le 0 (y i j)) (And (∀ (i : n), Eq (Finset.univ.sum f …
    ⊢ And (∀ (i j : n), LE.le 0 (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y) i j …
  -/
  simp [add_nonneg, ha, hb, mul_nonneg, hx, hy, sum_add_distrib, ← mul_sum, h]
  /-
    🎉 no goals
  -/


/-- Any permutation matrix is doubly stochastic. -/
lemma permMatrix_mem_doublyStochastic {σ : Equiv.Perm n} :
    σ.permMatrix R ∈ doublyStochastic R n := by
  /-
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : OrderedSemiring R
    σ : Equiv.Perm n
    ⊢ Membership.mem (doublyStochastic R n) (Equiv.Perm.permMatrix R σ)
  -/
  rw [mem_doublyStochastic_iff_sum]
  /-
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : OrderedSemiring R
    σ : Equiv.Perm n
    ⊢ And (∀ (i j : n), LE.le 0 (Equiv.Perm.permMatrix R σ i j)) (And (∀ (i : n),  …
  -/
  refine ⟨fun i j => ?g1, ?g2, ?g3⟩
  /-
    case g1
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : OrderedSemiring R
    σ : Equiv.Perm n
    i j : n
    ⊢ LE.le 0 (Equiv.Perm.permMatrix R σ i j)
  -/
  case g1 => aesop
  /-
    case g2
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : OrderedSemiring R
    σ : Equiv.Perm n
    ⊢ ∀ (i : n), Eq (Finset.univ.sum fun j => Equiv.Perm.permMatrix R σ i j) 1
  -/
  case g2 => simp [Equiv.toPEquiv_apply]
  /-
    case g3
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : OrderedSemiring R
    σ : Equiv.Perm n
    ⊢ ∀ (j : n), Eq (Finset.univ.sum fun i => Equiv.Perm.permMatrix R σ i j) 1
  -/
  case g3 => simp [Equiv.toPEquiv_apply, ← Equiv.eq_symm_apply]
  /-
    🎉 no goals
  -/


/--
A matrix is `s` times a doubly stochastic matrix iff all entries are nonnegative, and all row and
column sums are equal to `s`.

This lemma is useful for the proof of Birkhoff's theorem - in particular because it allows scaling
by nonnegative factors rather than positive ones only.
-/
lemma exists_mem_doublyStochastic_eq_smul_iff {M : Matrix n n R} {s : R} (hs : 0 ≤ s) :
    (∃ M' ∈ doublyStochastic R n, M = s • M') ↔
      (∀ i j, 0 ≤ M i j) ∧ (∀ i, ∑ j, M i j = s) ∧ (∀ j, ∑ i, M i j = s) := by
  classical
  constructor
  case mp =>
    rintro ⟨M', hM', rfl⟩
    rw [mem_doublyStochastic_iff_sum] at hM'
    simp only [smul_apply, smul_eq_mul, ← mul_sum]
    exact ⟨fun i j => mul_nonneg hs (hM'.1 _ _), by simp [hM']⟩
  rcases eq_or_lt_of_le hs with rfl | hs
  case inl =>
    simp only [zero_smul, exists_and_right, and_imp]
    intro h₁ h₂ _
    refine ⟨⟨1, Submonoid.one_mem _⟩, ?_⟩
    ext i j
    specialize h₂ i
    rw [sum_eq_zero_iff_of_nonneg (by simp [h₁ i])] at h₂
    exact h₂ _ (by simp)
  rintro ⟨hM₁, hM₂, hM₃⟩
  exact ⟨s⁻¹ • M, by simp [mem_doublyStochastic_iff_sum, ← mul_sum, hs.ne', inv_mul_cancel₀, *]⟩


