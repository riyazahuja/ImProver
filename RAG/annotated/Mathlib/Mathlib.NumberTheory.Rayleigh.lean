/-- In the Beatty sequence for real number `r`, the `k`th term is `⌊k * r⌋`. -/
noncomputable def beattySeq (r : ℝ) : ℤ → ℤ :=
  fun k ↦ ⌊k * r⌋


/-- In this variant of the Beatty sequence for `r`, the `k`th term is `⌈k * r⌉ - 1`. -/
noncomputable def beattySeq' (r : ℝ) : ℤ → ℤ :=
  fun k ↦ ⌈k * r⌉ - 1


/-- Let `r > 1` and `1/r + 1/s = 1`. Then `B_r` and `B'_s` are disjoint (i.e. no collision exists).
-/
private theorem no_collision (hrs : r.IsConjExponent s) :
    Disjoint {beattySeq r k | k} {beattySeq' s k | k} := by
  /-
    r s : Real
    hrs : r.IsConjExponent s
    ⊢ Disjoint (setOf fun x => Exists fun k => Eq (beattySeq r k) x) (setOf fun x  …
  -/
  rw [Set.disjoint_left]
  /-
    r s : Real
    hrs : r.IsConjExponent s
    ⊢ ∀ ⦃a : Int⦄, Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq r  …
  -/
  intro j ⟨k, h₁⟩ ⟨m, h₂⟩
  /-
    r s : Real
    hrs : r.IsConjExponent s
    j k : Int
    h₁ : Eq (beattySeq r k) j
    m : Int
    h₂ : Eq (beattySeq' s m) j
    ⊢ False
  -/
  rw [beattySeq, Int.floor_eq_iff, ← div_le_iff₀ hrs.pos, ← lt_div_iff₀ hrs.pos] at h₁
  rw [beattySeq', sub_eq_iff_eq_add, Int.ceil_eq_iff, Int.cast_add, Int.cast_one,
    add_sub_cancel_right, ← div_lt_iff₀ hrs.symm.pos, ← le_div_iff₀ hrs.symm.pos] at h₂
  /-
    r s : Real
    hrs : r.IsConjExponent s
    j k : Int
    h₁ : And (LE.le (HDiv.hDiv (↑j) r) ↑k) (LT.lt (↑k) (HDiv.hDiv (HAdd.hAdd (↑j)  …
    m : Int
    h₂ : And (LT.lt (HDiv.hDiv (↑j) s) ↑m) (LE.le (↑m) (HDiv.hDiv (HAdd.hAdd (↑j)  …
    ⊢ False
  -/
  have h₃ := add_lt_add_of_le_of_lt h₁.1 h₂.1
  /-
    r s : Real
    hrs : r.IsConjExponent s
    j k : Int
    h₁ : And (LE.le (HDiv.hDiv (↑j) r) ↑k) (LT.lt (↑k) (HDiv.hDiv (HAdd.hAdd (↑j)  …
    m : Int
    h₂ : And (LT.lt (HDiv.hDiv (↑j) s) ↑m) (LE.le (↑m) (HDiv.hDiv (HAdd.hAdd (↑j)  …
    h₃ : LT.lt (HAdd.hAdd (HDiv.hDiv (↑j) r) (HDiv.hDiv (↑j) s)) (HAdd.hAdd ↑k ↑m)
    ⊢ False
  -/
  have h₄ := add_lt_add_of_lt_of_le h₁.2 h₂.2
  /-
    r s : Real
    hrs : r.IsConjExponent s
    j k : Int
    h₁ : And (LE.le (HDiv.hDiv (↑j) r) ↑k) (LT.lt (↑k) (HDiv.hDiv (HAdd.hAdd (↑j)  …
    m : Int
    h₂ : And (LT.lt (HDiv.hDiv (↑j) s) ↑m) (LE.le (↑m) (HDiv.hDiv (HAdd.hAdd (↑j)  …
    h₃ : LT.lt (HAdd.hAdd (HDiv.hDiv (↑j) r) (HDiv.hDiv (↑j) s)) (HAdd.hAdd ↑k ↑m)
    h₄ : LT.lt (HAdd.hAdd ↑k ↑m) (HAdd.hAdd (HDiv.hDiv (HAdd.hAdd (↑j) 1) r) (HDiv …
    ⊢ False
  -/
  simp_rw [div_eq_inv_mul, ← right_distrib, hrs.inv_add_inv_conj, one_mul] at h₃ h₄
  /-
    r s : Real
    hrs : r.IsConjExponent s
    j k : Int
    h₁ : And (LE.le (HDiv.hDiv (↑j) r) ↑k) (LT.lt (↑k) (HDiv.hDiv (HAdd.hAdd (↑j)  …
    m : Int
    h₂ : And (LT.lt (HDiv.hDiv (↑j) s) ↑m) (LE.le (↑m) (HDiv.hDiv (HAdd.hAdd (↑j)  …
    h₃ : LT.lt (↑j) (HAdd.hAdd ↑k ↑m)
    h₄ : LT.lt (HAdd.hAdd ↑k ↑m) (HAdd.hAdd (↑j) 1)
    ⊢ False
  -/
  rw [← Int.cast_one] at h₄
  /-
    r s : Real
    hrs : r.IsConjExponent s
    j k : Int
    h₁ : And (LE.le (HDiv.hDiv (↑j) r) ↑k) (LT.lt (↑k) (HDiv.hDiv (HAdd.hAdd (↑j)  …
    m : Int
    h₂ : And (LT.lt (HDiv.hDiv (↑j) s) ↑m) (LE.le (↑m) (HDiv.hDiv (HAdd.hAdd (↑j)  …
    h₃ : LT.lt (↑j) (HAdd.hAdd ↑k ↑m)
    h₄ : LT.lt (HAdd.hAdd ↑k ↑m) (HAdd.hAdd ↑j ↑1)
    ⊢ False
  -/
  simp_rw [← Int.cast_add, Int.cast_lt, Int.lt_add_one_iff] at h₃ h₄
  /-
    r s : Real
    hrs : r.IsConjExponent s
    j k : Int
    h₁ : And (LE.le (HDiv.hDiv (↑j) r) ↑k) (LT.lt (↑k) (HDiv.hDiv (HAdd.hAdd (↑j)  …
    m : Int
    h₂ : And (LT.lt (HDiv.hDiv (↑j) s) ↑m) (LE.le (↑m) (HDiv.hDiv (HAdd.hAdd (↑j)  …
    h₃ : LT.lt j (HAdd.hAdd k m)
    h₄ : LE.le (HAdd.hAdd k m) j
    ⊢ False
  -/
  exact h₄.not_lt h₃
  /-
    🎉 no goals
  -/


/-- Let `r > 1` and `1/r + 1/s = 1`. Suppose there is an integer `j` where `B_r` and `B'_s` both
jump over `j` (i.e. an anti-collision). Then this leads to a contradiction. -/
private theorem no_anticollision (hrs : r.IsConjExponent s) :
    ¬∃ j k m : ℤ, k < j / r ∧ (j + 1) / r ≤ k + 1 ∧ m ≤ j / s ∧ (j + 1) / s < m + 1 := by
  /-
    r s : Real
    hrs : r.IsConjExponent s
    ⊢ Not (Exists fun j => Exists fun k => Exists fun m => And (LT.lt (↑k) (HDiv.h …
  -/
  intro ⟨j, k, m, h₁₁, h₁₂, h₂₁, h₂₂⟩
  /-
    r s : Real
    hrs : r.IsConjExponent s
    j k m : Int
    h₁₁ : LT.lt (↑k) (HDiv.hDiv (↑j) r)
    h₁₂ : LE.le (HDiv.hDiv (HAdd.hAdd (↑j) 1) r) (HAdd.hAdd (↑k) 1)
    h₂₁ : LE.le (↑m) (HDiv.hDiv (↑j) s)
    h₂₂ : LT.lt (HDiv.hDiv (HAdd.hAdd (↑j) 1) s) (HAdd.hAdd (↑m) 1)
    ⊢ False
  -/
  have h₃ := add_lt_add_of_lt_of_le h₁₁ h₂₁
  /-
    r s : Real
    hrs : r.IsConjExponent s
    j k m : Int
    h₁₁ : LT.lt (↑k) (HDiv.hDiv (↑j) r)
    h₁₂ : LE.le (HDiv.hDiv (HAdd.hAdd (↑j) 1) r) (HAdd.hAdd (↑k) 1)
    h₂₁ : LE.le (↑m) (HDiv.hDiv (↑j) s)
    h₂₂ : LT.lt (HDiv.hDiv (HAdd.hAdd (↑j) 1) s) (HAdd.hAdd (↑m) 1)
    h₃ : LT.lt (HAdd.hAdd ↑k ↑m) (HAdd.hAdd (HDiv.hDiv (↑j) r) (HDiv.hDiv (↑j) s))
    ⊢ False
  -/
  have h₄ := add_lt_add_of_le_of_lt h₁₂ h₂₂
  /-
    r s : Real
    hrs : r.IsConjExponent s
    j k m : Int
    h₁₁ : LT.lt (↑k) (HDiv.hDiv (↑j) r)
    h₁₂ : LE.le (HDiv.hDiv (HAdd.hAdd (↑j) 1) r) (HAdd.hAdd (↑k) 1)
    h₂₁ : LE.le (↑m) (HDiv.hDiv (↑j) s)
    h₂₂ : LT.lt (HDiv.hDiv (HAdd.hAdd (↑j) 1) s) (HAdd.hAdd (↑m) 1)
    h₃ : LT.lt (HAdd.hAdd ↑k ↑m) (HAdd.hAdd (HDiv.hDiv (↑j) r) (HDiv.hDiv (↑j) s))
    h₄ : LT.lt (HAdd.hAdd (HDiv.hDiv (HAdd.hAdd (↑j) 1) r) (HDiv.hDiv (HAdd.hAdd ( …
    ⊢ False
  -/
  simp_rw [div_eq_inv_mul, ← right_distrib, hrs.inv_add_inv_conj, one_mul] at h₃ h₄
  /-
    r s : Real
    hrs : r.IsConjExponent s
    j k m : Int
    h₁₁ : LT.lt (↑k) (HDiv.hDiv (↑j) r)
    h₁₂ : LE.le (HDiv.hDiv (HAdd.hAdd (↑j) 1) r) (HAdd.hAdd (↑k) 1)
    h₂₁ : LE.le (↑m) (HDiv.hDiv (↑j) s)
    h₂₂ : LT.lt (HDiv.hDiv (HAdd.hAdd (↑j) 1) s) (HAdd.hAdd (↑m) 1)
    h₃ : LT.lt (HAdd.hAdd ↑k ↑m) ↑j
    h₄ : LT.lt (HAdd.hAdd (↑j) 1) (HAdd.hAdd (HAdd.hAdd (↑k) 1) (HAdd.hAdd (↑m) 1))
    ⊢ False
  -/
  rw [← Int.cast_one, ← add_assoc, add_lt_add_iff_right, add_right_comm] at h₄
  /-
    r s : Real
    hrs : r.IsConjExponent s
    j k m : Int
    h₁₁ : LT.lt (↑k) (HDiv.hDiv (↑j) r)
    h₁₂ : LE.le (HDiv.hDiv (HAdd.hAdd (↑j) 1) r) (HAdd.hAdd (↑k) 1)
    h₂₁ : LE.le (↑m) (HDiv.hDiv (↑j) s)
    h₂₂ : LT.lt (HDiv.hDiv (HAdd.hAdd (↑j) 1) s) (HAdd.hAdd (↑m) 1)
    h₃ : LT.lt (HAdd.hAdd ↑k ↑m) ↑j
    h₄ : LT.lt (↑j) (HAdd.hAdd (HAdd.hAdd ↑k ↑m) ↑1)
    ⊢ False
  -/
  simp_rw [← Int.cast_add, Int.cast_lt, Int.lt_add_one_iff] at h₃ h₄
  /-
    r s : Real
    hrs : r.IsConjExponent s
    j k m : Int
    h₁₁ : LT.lt (↑k) (HDiv.hDiv (↑j) r)
    h₁₂ : LE.le (HDiv.hDiv (HAdd.hAdd (↑j) 1) r) (HAdd.hAdd (↑k) 1)
    h₂₁ : LE.le (↑m) (HDiv.hDiv (↑j) s)
    h₂₂ : LT.lt (HDiv.hDiv (HAdd.hAdd (↑j) 1) s) (HAdd.hAdd (↑m) 1)
    h₃ : LT.lt (HAdd.hAdd k m) j
    h₄ : LE.le j (HAdd.hAdd k m)
    ⊢ False
  -/
  exact h₄.not_lt h₃
  /-
    🎉 no goals
  -/


/-- Let `0 < r ∈ ℝ` and `j ∈ ℤ`. Then either `j ∈ B_r` or `B_r` jumps over `j`. -/
private theorem hit_or_miss (h : r > 0) :
    j ∈ {beattySeq r k | k} ∨ ∃ k : ℤ, k < j / r ∧ (j + 1) / r ≤ k + 1 := by
  -- for both cases, the candidate is `k = ⌈(j + 1) / r⌉ - 1`
  /-
    r : Real
    j : Int
    h : GT.gt r 0
    ⊢ Or (Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq r k) x) j)  …
  -/
  cases lt_or_ge ((⌈(j + 1) / r⌉ - 1) * r) j
    /-
      case inl
      r : Real
      j : Int
      h : GT.gt r 0
      h✝ : LT.lt (HMul.hMul (HSub.hSub (↑(Int.ceil (HDiv.hDiv (HAdd.hAdd (↑j) 1) r)) …
      ⊢ Or (Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq r k) x) j)  …
    -/
  · refine Or.inr ⟨⌈(j + 1) / r⌉ - 1, ?_⟩
    /-
      case inl
      r : Real
      j : Int
      h : GT.gt r 0
      h✝ : LT.lt (HMul.hMul (HSub.hSub (↑(Int.ceil (HDiv.hDiv (HAdd.hAdd (↑j) 1) r)) …
      ⊢ And (LT.lt (↑(HSub.hSub (Int.ceil (HDiv.hDiv (HAdd.hAdd (↑j) 1) r)) 1)) (HDi …
    -/
    rw [Int.cast_sub, Int.cast_one, lt_div_iff₀ h, sub_add_cancel]
    /-
      case inl
      r : Real
      j : Int
      h : GT.gt r 0
      h✝ : LT.lt (HMul.hMul (HSub.hSub (↑(Int.ceil (HDiv.hDiv (HAdd.hAdd (↑j) 1) r)) …
      ⊢ And (LT.lt (HMul.hMul (HSub.hSub (↑(Int.ceil (HDiv.hDiv (HAdd.hAdd (↑j) 1) r …
    -/
    exact ⟨‹_›, Int.le_ceil _⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      r : Real
      j : Int
      h : GT.gt r 0
      h✝ : GE.ge (HMul.hMul (HSub.hSub (↑(Int.ceil (HDiv.hDiv (HAdd.hAdd (↑j) 1) r)) …
      ⊢ Or (Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq r k) x) j)  …
    -/
  · refine Or.inl ⟨⌈(j + 1) / r⌉ - 1, ?_⟩
    /-
      case inr
      r : Real
      j : Int
      h : GT.gt r 0
      h✝ : GE.ge (HMul.hMul (HSub.hSub (↑(Int.ceil (HDiv.hDiv (HAdd.hAdd (↑j) 1) r)) …
      ⊢ Eq (beattySeq r (HSub.hSub (Int.ceil (HDiv.hDiv (HAdd.hAdd (↑j) 1) r)) 1)) j
    -/
    rw [beattySeq, Int.floor_eq_iff, Int.cast_sub, Int.cast_one, ← lt_div_iff₀ h, sub_lt_iff_lt_add]
    /-
      case inr
      r : Real
      j : Int
      h : GT.gt r 0
      h✝ : GE.ge (HMul.hMul (HSub.hSub (↑(Int.ceil (HDiv.hDiv (HAdd.hAdd (↑j) 1) r)) …
      ⊢ And (LE.le (↑j) (HMul.hMul (HSub.hSub (↑(Int.ceil (HDiv.hDiv (HAdd.hAdd (↑j) …
    -/
    exact ⟨‹_›, Int.ceil_lt_add_one _⟩
    /-
      🎉 no goals
    -/


/-- Let `0 < r ∈ ℝ` and `j ∈ ℤ`. Then either `j ∈ B'_r` or `B'_r` jumps over `j`. -/
private theorem hit_or_miss' (h : r > 0) :
    j ∈ {beattySeq' r k | k} ∨ ∃ k : ℤ, k ≤ j / r ∧ (j + 1) / r < k + 1 := by
  -- for both cases, the candidate is `k = ⌊(j + 1) / r⌋`
  /-
    r : Real
    j : Int
    h : GT.gt r 0
    ⊢ Or (Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq' r k) x) j) …
  -/
  cases le_or_gt (⌊(j + 1) / r⌋ * r) j
    /-
      case inl
      r : Real
      j : Int
      h : GT.gt r 0
      h✝ : LE.le (HMul.hMul (↑(Int.floor (HDiv.hDiv (HAdd.hAdd (↑j) 1) r))) r) ↑j
      ⊢ Or (Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq' r k) x) j) …
    -/
  · exact Or.inr ⟨⌊(j + 1) / r⌋, (le_div_iff₀ h).2 ‹_›, Int.lt_floor_add_one _⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      r : Real
      j : Int
      h : GT.gt r 0
      h✝ : GT.gt (HMul.hMul (↑(Int.floor (HDiv.hDiv (HAdd.hAdd (↑j) 1) r))) r) ↑j
      ⊢ Or (Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq' r k) x) j) …
    -/
  · refine Or.inl ⟨⌊(j + 1) / r⌋, ?_⟩
    /-
      case inr
      r : Real
      j : Int
      h : GT.gt r 0
      h✝ : GT.gt (HMul.hMul (↑(Int.floor (HDiv.hDiv (HAdd.hAdd (↑j) 1) r))) r) ↑j
      ⊢ Eq (beattySeq' r (Int.floor (HDiv.hDiv (HAdd.hAdd (↑j) 1) r))) j
    -/
    rw [beattySeq', sub_eq_iff_eq_add, Int.ceil_eq_iff, Int.cast_add, Int.cast_one]
    /-
      case inr
      r : Real
      j : Int
      h : GT.gt r 0
      h✝ : GT.gt (HMul.hMul (↑(Int.floor (HDiv.hDiv (HAdd.hAdd (↑j) 1) r))) r) ↑j
      ⊢ And (LT.lt (HSub.hSub (HAdd.hAdd (↑j) 1) 1) (HMul.hMul (↑(Int.floor (HDiv.hD …
    -/
    constructor
      /-
        case inr.left
        r : Real
        j : Int
        h : GT.gt r 0
        h✝ : GT.gt (HMul.hMul (↑(Int.floor (HDiv.hDiv (HAdd.hAdd (↑j) 1) r))) r) ↑j
        ⊢ LT.lt (HSub.hSub (HAdd.hAdd (↑j) 1) 1) (HMul.hMul (↑(Int.floor (HDiv.hDiv (H …
      -/
    · rwa [add_sub_cancel_right]
      /-
        🎉 no goals
      -/
    /-
      case inr.right
      r : Real
      j : Int
      h : GT.gt r 0
      h✝ : GT.gt (HMul.hMul (↑(Int.floor (HDiv.hDiv (HAdd.hAdd (↑j) 1) r))) r) ↑j
      ⊢ LE.le (HMul.hMul (↑(Int.floor (HDiv.hDiv (HAdd.hAdd (↑j) 1) r))) r) (HAdd.hA …
    -/
    exact sub_nonneg.1 (Int.sub_floor_div_mul_nonneg (j + 1 : ℝ) h)
    /-
      🎉 no goals
    -/


/-- Generalization of Rayleigh's theorem on Beatty sequences. Let `r` be a real number greater
than 1, and `1/r + 1/s = 1`. Then the complement of `B_r` is `B'_s`. -/
theorem compl_beattySeq {r s : ℝ} (hrs : r.IsConjExponent s) :
    {beattySeq r k | k}ᶜ = {beattySeq' s k | k} := by
  /-
    r s : Real
    hrs : r.IsConjExponent s
    ⊢ Eq (HasCompl.compl (setOf fun x => Exists fun k => Eq (beattySeq r k) x)) (s …
  -/
  ext j
  /-
    case h
    r s : Real
    hrs : r.IsConjExponent s
    j : Int
    ⊢ Iff (Membership.mem (HasCompl.compl (setOf fun x => Exists fun k => Eq (beat …
  -/
  by_cases h₁ : j ∈ {beattySeq r k | k} <;> by_cases h₂ : j ∈ {beattySeq' s k | k}
    /-
      case pos
      r s : Real
      hrs : r.IsConjExponent s
      j : Int
      h₁ : Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq r k) x) j
      h₂ : Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq' s k) x) j
      ⊢ Iff (Membership.mem (HasCompl.compl (setOf fun x => Exists fun k => Eq (beat …
    -/
  · exact (Set.not_disjoint_iff.2 ⟨j, h₁, h₂⟩ (Beatty.no_collision hrs)).elim
    /-
      🎉 no goals
    -/
    /-
      case neg
      r s : Real
      hrs : r.IsConjExponent s
      j : Int
      h₁ : Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq r k) x) j
      h₂ : Not (Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq' s k) x …
      ⊢ Iff (Membership.mem (HasCompl.compl (setOf fun x => Exists fun k => Eq (beat …
    -/
  · simp only [Set.mem_compl_iff, h₁, h₂, not_true_eq_false]
    /-
      🎉 no goals
    -/
    /-
      case pos
      r s : Real
      hrs : r.IsConjExponent s
      j : Int
      h₁ : Not (Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq r k) x) …
      h₂ : Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq' s k) x) j
      ⊢ Iff (Membership.mem (HasCompl.compl (setOf fun x => Exists fun k => Eq (beat …
    -/
  · simp only [Set.mem_compl_iff, h₁, h₂, not_false_eq_true]
    /-
      🎉 no goals
    -/
    /-
      case neg
      r s : Real
      hrs : r.IsConjExponent s
      j : Int
      h₁ : Not (Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq r k) x) …
      h₂ : Not (Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq' s k) x …
      ⊢ Iff (Membership.mem (HasCompl.compl (setOf fun x => Exists fun k => Eq (beat …
    -/
  · have ⟨k, h₁₁, h₁₂⟩ := (Beatty.hit_or_miss hrs.pos).resolve_left h₁
    /-
      case neg
      r s : Real
      hrs : r.IsConjExponent s
      j : Int
      h₁ : Not (Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq r k) x) …
      h₂ : Not (Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq' s k) x …
      k : Int
      h₁₁ : LT.lt (↑k) (HDiv.hDiv (↑j) r)
      h₁₂ : LE.le (HDiv.hDiv (HAdd.hAdd (↑j) 1) r) (HAdd.hAdd (↑k) 1)
      ⊢ Iff (Membership.mem (HasCompl.compl (setOf fun x => Exists fun k => Eq (beat …
    -/
    have ⟨m, h₂₁, h₂₂⟩ := (Beatty.hit_or_miss' hrs.symm.pos).resolve_left h₂
    /-
      case neg
      r s : Real
      hrs : r.IsConjExponent s
      j : Int
      h₁ : Not (Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq r k) x) …
      h₂ : Not (Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq' s k) x …
      k : Int
      h₁₁ : LT.lt (↑k) (HDiv.hDiv (↑j) r)
      h₁₂ : LE.le (HDiv.hDiv (HAdd.hAdd (↑j) 1) r) (HAdd.hAdd (↑k) 1)
      m : Int
      h₂₁ : LE.le (↑m) (HDiv.hDiv (↑j) s)
      h₂₂ : LT.lt (HDiv.hDiv (HAdd.hAdd (↑j) 1) s) (HAdd.hAdd (↑m) 1)
      ⊢ Iff (Membership.mem (HasCompl.compl (setOf fun x => Exists fun k => Eq (beat …
    -/
    exact (Beatty.no_anticollision hrs ⟨j, k, m, h₁₁, h₁₂, h₂₁, h₂₂⟩).elim
    /-
      🎉 no goals
    -/


theorem compl_beattySeq' {r s : ℝ} (hrs : r.IsConjExponent s) :
    {beattySeq' r k | k}ᶜ = {beattySeq s k | k} := by
  /-
    r s : Real
    hrs : r.IsConjExponent s
    ⊢ Eq (HasCompl.compl (setOf fun x => Exists fun k => Eq (beattySeq' r k) x)) ( …
  -/
  rw [← compl_beattySeq hrs.symm, compl_compl]
  /-
    🎉 no goals
  -/


/-- Generalization of Rayleigh's theorem on Beatty sequences. Let `r` be a real number greater
than 1, and `1/r + 1/s = 1`. Then `B⁺_r` and `B⁺'_s` partition the positive integers. -/
theorem beattySeq_symmDiff_beattySeq'_pos {r s : ℝ} (hrs : r.IsConjExponent s) :
    {beattySeq r k | k > 0} ∆ {beattySeq' s k | k > 0} = {n | 0 < n} := by
  /-
    r s : Real
    hrs : r.IsConjExponent s
    ⊢ Eq (symmDiff (setOf fun x => Exists fun k => And (GT.gt k 0) (Eq (beattySeq  …
  -/
  apply Set.eq_of_subset_of_subset
    /-
      case a
      r s : Real
      hrs : r.IsConjExponent s
      ⊢ HasSubset.Subset (symmDiff (setOf fun x => Exists fun k => And (GT.gt k 0) ( …
    -/
  · rintro j (⟨⟨k, hk, hjk⟩, -⟩ | ⟨⟨k, hk, hjk⟩, -⟩)
      /-
        case a.inl.intro.intro.intro
        r s : Real
        hrs : r.IsConjExponent s
        j k : Int
        hk : GT.gt k 0
        hjk : Eq (beattySeq r k) j
        ⊢ Membership.mem (setOf fun n => LT.lt 0 n) j
      -/
    · rw [Set.mem_setOf_eq, ← hjk, beattySeq, Int.floor_pos]
      /-
        case a.inl.intro.intro.intro
        r s : Real
        hrs : r.IsConjExponent s
        j k : Int
        hk : GT.gt k 0
        hjk : Eq (beattySeq r k) j
        ⊢ LE.le 1 (HMul.hMul (↑k) r)
      -/
      exact one_le_mul_of_one_le_of_one_le (by norm_cast) hrs.one_lt.le
      /-
        🎉 no goals
      -/
      /-
        case a.inr.intro.intro.intro
        r s : Real
        hrs : r.IsConjExponent s
        j k : Int
        hk : GT.gt k 0
        hjk : Eq (beattySeq' s k) j
        ⊢ Membership.mem (setOf fun n => LT.lt 0 n) j
      -/
    · rw [Set.mem_setOf_eq, ← hjk, beattySeq', sub_pos, Int.lt_ceil, Int.cast_one]
      /-
        case a.inr.intro.intro.intro
        r s : Real
        hrs : r.IsConjExponent s
        j k : Int
        hk : GT.gt k 0
        hjk : Eq (beattySeq' s k) j
        ⊢ LT.lt 1 (HMul.hMul (↑k) s)
      -/
      exact one_lt_mul_of_le_of_lt (by norm_cast) hrs.symm.one_lt
      /-
        🎉 no goals
      -/
  /-
    case a
    r s : Real
    hrs : r.IsConjExponent s
    ⊢ HasSubset.Subset (setOf fun n => LT.lt 0 n) (symmDiff (setOf fun x => Exists …
  -/
  intro j (hj : 0 < j)
  have hb₁ : ∀ s ≥ 0, j ∈ {beattySeq s k | k > 0} ↔ j ∈ {beattySeq s k | k} := by
    intro _ hs
    refine ⟨fun ⟨k, _, hk⟩ ↦ ⟨k, hk⟩, fun ⟨k, hk⟩ ↦ ⟨k, ?_, hk⟩⟩
    rw [← hk, beattySeq, Int.floor_pos] at hj
    exact_mod_cast pos_of_mul_pos_left (zero_lt_one.trans_le hj) hs
  have hb₂ : ∀ s ≥ 0, j ∈ {beattySeq' s k | k > 0} ↔ j ∈ {beattySeq' s k | k} := by
    intro _ hs
    refine ⟨fun ⟨k, _, hk⟩ ↦ ⟨k, hk⟩, fun ⟨k, hk⟩ ↦ ⟨k, ?_, hk⟩⟩
    rw [← hk, beattySeq', sub_pos, Int.lt_ceil, Int.cast_one] at hj
    exact_mod_cast pos_of_mul_pos_left (zero_lt_one.trans hj) hs
  rw [Set.mem_symmDiff, hb₁ _ hrs.nonneg, hb₂ _ hrs.symm.nonneg, ← compl_beattySeq hrs,
    Set.not_mem_compl_iff, Set.mem_compl_iff, and_self, and_self]
  /-
    case a
    r s : Real
    hrs : r.IsConjExponent s
    j : Int
    hj : LT.lt 0 j
    hb₁ : ∀ (s : Real), GE.ge s 0 → Iff (Membership.mem (setOf fun x => Exists fun …
    hb₂ : ∀ (s : Real), GE.ge s 0 → Iff (Membership.mem (setOf fun x => Exists fun …
    ⊢ Or (Membership.mem (setOf fun x => Exists fun k => Eq (beattySeq r k) x) j)  …
  -/
  exact or_not
  /-
    🎉 no goals
  -/


theorem beattySeq'_symmDiff_beattySeq_pos {r s : ℝ} (hrs : r.IsConjExponent s) :
    {beattySeq' r k | k > 0} ∆ {beattySeq s k | k > 0} = {n | 0 < n} := by
  /-
    r s : Real
    hrs : r.IsConjExponent s
    ⊢ Eq (symmDiff (setOf fun x => Exists fun k => And (GT.gt k 0) (Eq (beattySeq' …
  -/
  rw [symmDiff_comm, beattySeq_symmDiff_beattySeq'_pos hrs.symm]
  /-
    🎉 no goals
  -/


/-- Let `r` be an irrational number. Then `B⁺_r` and `B⁺'_r` are equal. -/
theorem Irrational.beattySeq'_pos_eq {r : ℝ} (hr : Irrational r) :
    {beattySeq' r k | k > 0} = {beattySeq r k | k > 0} := by
  /-
    r : Real
    hr : Irrational r
    ⊢ Eq (setOf fun x => Exists fun k => And (GT.gt k 0) (Eq (beattySeq' r k) x))  …
  -/
  dsimp only [beattySeq, beattySeq']
  /-
    r : Real
    hr : Irrational r
    ⊢ Eq (setOf fun x => Exists fun k => And (GT.gt k 0) (Eq (HSub.hSub (Int.ceil  …
  -/
  congr! 4; rename_i k; rw [and_congr_right_iff]; intro hk; congr!
  /-
    case h.e'_2.h.h.e'_2.h.a.a.h.e'_2
    r : Real
    hr : Irrational r
    x✝ k : Int
    hk : GT.gt k 0
    ⊢ Eq (HSub.hSub (Int.ceil (HMul.hMul (↑k) r)) 1) (Int.floor (HMul.hMul (↑k) r))
  -/
  rw [sub_eq_iff_eq_add, Int.ceil_eq_iff, Int.cast_add, Int.cast_one, add_sub_cancel_right]
  /-
    case h.e'_2.h.h.e'_2.h.a.a.h.e'_2
    r : Real
    hr : Irrational r
    x✝ k : Int
    hk : GT.gt k 0
    ⊢ And (LT.lt (↑(Int.floor (HMul.hMul (↑k) r))) (HMul.hMul (↑k) r)) (LE.le (HMu …
  -/
  refine ⟨(Int.floor_le _).lt_of_ne fun h ↦ ?_, (Int.lt_floor_add_one _).le⟩
  /-
    case h.e'_2.h.h.e'_2.h.a.a.h.e'_2
    r : Real
    hr : Irrational r
    x✝ k : Int
    hk : GT.gt k 0
    h : Eq (↑(Int.floor (HMul.hMul (↑k) r))) (HMul.hMul (↑k) r)
    ⊢ False
  -/
  exact (hr.int_mul hk.ne').ne_int ⌊k * r⌋ h.symm
  /-
    🎉 no goals
  -/


/-- **Rayleigh's theorem** on Beatty sequences. Let `r` be an irrational number greater than 1, and
`1/r + 1/s = 1`. Then `B⁺_r` and `B⁺_s` partition the positive integers. -/
theorem Irrational.beattySeq_symmDiff_beattySeq_pos {r s : ℝ}
    (hrs : r.IsConjExponent s) (hr : Irrational r) :
    {beattySeq r k | k > 0} ∆ {beattySeq s k | k > 0} = {n | 0 < n} := by
  /-
    r s : Real
    hrs : r.IsConjExponent s
    hr : Irrational r
    ⊢ Eq (symmDiff (setOf fun x => Exists fun k => And (GT.gt k 0) (Eq (beattySeq  …
  -/
  rw [← hr.beattySeq'_pos_eq, beattySeq'_symmDiff_beattySeq_pos hrs]
  /-
    🎉 no goals
  -/

