local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


/-- An orthonormal set of vectors in an `InnerProductSpace` -/
def Orthonormal (v : ι → E) : Prop :=
  (∀ i, ‖v i‖ = 1) ∧ Pairwise fun i j => ⟪v i, v j⟫ = 0


/-- `if ... then ... else` characterization of an indexed set of vectors being orthonormal.  (Inner
product equals Kronecker delta.) -/
theorem orthonormal_iff_ite [DecidableEq ι] {v : ι → E} :
    Orthonormal 𝕜 v ↔ ∀ i j, ⟪v i, v j⟫ = if i = j then (1 : 𝕜) else (0 : 𝕜) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    ι : Type u_4
    inst✝ : DecidableEq ι
    v : ι → E
    ⊢ Iff (Orthonormal 𝕜 v) (∀ (i j : ι), Eq (Inner.inner (v i) (v j)) (ite (Eq i  …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : RCLike 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      ι : Type u_4
      inst✝ : DecidableEq ι
      v : ι → E
      ⊢ Orthonormal 𝕜 v → ∀ (i j : ι), Eq (Inner.inner (v i) (v j)) (ite (Eq i j) 1 0)
    -/
  · intro hv i j
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : RCLike 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      ι : Type u_4
      inst✝ : DecidableEq ι
      v : ι → E
      hv : Orthonormal 𝕜 v
      i j : ι
      ⊢ Eq (Inner.inner (v i) (v j)) (ite (Eq i j) 1 0)
    -/
    split_ifs with h
      /-
        case pos
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : SeminormedAddCommGroup E
        inst✝¹ : InnerProductSpace 𝕜 E
        ι : Type u_4
        inst✝ : DecidableEq ι
        v : ι → E
        hv : Orthonormal 𝕜 v
        i j : ι
        h : Eq i j
        ⊢ Eq (Inner.inner (v i) (v j)) 1
      -/
    · simp [h, inner_self_eq_norm_sq_to_K, hv.1]
      /-
        🎉 no goals
      -/
      /-
        case neg
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : SeminormedAddCommGroup E
        inst✝¹ : InnerProductSpace 𝕜 E
        ι : Type u_4
        inst✝ : DecidableEq ι
        v : ι → E
        hv : Orthonormal 𝕜 v
        i j : ι
        h : Not (Eq i j)
        ⊢ Eq (Inner.inner (v i) (v j)) 0
      -/
    · exact hv.2 h
      /-
        🎉 no goals
      -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : RCLike 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      ι : Type u_4
      inst✝ : DecidableEq ι
      v : ι → E
      ⊢ (∀ (i j : ι), Eq (Inner.inner (v i) (v j)) (ite (Eq i j) 1 0)) → Orthonormal …
    -/
  · intro h
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : RCLike 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      ι : Type u_4
      inst✝ : DecidableEq ι
      v : ι → E
      h : ∀ (i j : ι), Eq (Inner.inner (v i) (v j)) (ite (Eq i j) 1 0)
      ⊢ Orthonormal 𝕜 v
    -/
    constructor
      /-
        case mpr.left
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : SeminormedAddCommGroup E
        inst✝¹ : InnerProductSpace 𝕜 E
        ι : Type u_4
        inst✝ : DecidableEq ι
        v : ι → E
        h : ∀ (i j : ι), Eq (Inner.inner (v i) (v j)) (ite (Eq i j) 1 0)
        ⊢ ∀ (i : ι), Eq (Norm.norm (v i)) 1
      -/
    · intro i
      /-
        case mpr.left
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : SeminormedAddCommGroup E
        inst✝¹ : InnerProductSpace 𝕜 E
        ι : Type u_4
        inst✝ : DecidableEq ι
        v : ι → E
        h : ∀ (i j : ι), Eq (Inner.inner (v i) (v j)) (ite (Eq i j) 1 0)
        i : ι
        ⊢ Eq (Norm.norm (v i)) 1
      -/
      have h' : ‖v i‖ ^ 2 = 1 ^ 2 := by simp [@norm_sq_eq_inner 𝕜, h i i]
      /-
        case mpr.left
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : SeminormedAddCommGroup E
        inst✝¹ : InnerProductSpace 𝕜 E
        ι : Type u_4
        inst✝ : DecidableEq ι
        v : ι → E
        h : ∀ (i j : ι), Eq (Inner.inner (v i) (v j)) (ite (Eq i j) 1 0)
        i : ι
        h' : Eq (HPow.hPow (Norm.norm (v i)) 2) (HPow.hPow 1 2)
        ⊢ Eq (Norm.norm (v i)) 1
      -/
      have h₁ : 0 ≤ ‖v i‖ := norm_nonneg _
      /-
        case mpr.left
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : SeminormedAddCommGroup E
        inst✝¹ : InnerProductSpace 𝕜 E
        ι : Type u_4
        inst✝ : DecidableEq ι
        v : ι → E
        h : ∀ (i j : ι), Eq (Inner.inner (v i) (v j)) (ite (Eq i j) 1 0)
        i : ι
        h' : Eq (HPow.hPow (Norm.norm (v i)) 2) (HPow.hPow 1 2)
        h₁ : LE.le 0 (Norm.norm (v i))
        ⊢ Eq (Norm.norm (v i)) 1
      -/
      have h₂ : (0 : ℝ) ≤ 1 := zero_le_one
      /-
        case mpr.left
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : SeminormedAddCommGroup E
        inst✝¹ : InnerProductSpace 𝕜 E
        ι : Type u_4
        inst✝ : DecidableEq ι
        v : ι → E
        h : ∀ (i j : ι), Eq (Inner.inner (v i) (v j)) (ite (Eq i j) 1 0)
        i : ι
        h' : Eq (HPow.hPow (Norm.norm (v i)) 2) (HPow.hPow 1 2)
        h₁ : LE.le 0 (Norm.norm (v i))
        h₂ : LE.le 0 1
        ⊢ Eq (Norm.norm (v i)) 1
      -/
      rwa [sq_eq_sq₀ h₁ h₂] at h'
      /-
        🎉 no goals
      -/
      /-
        case mpr.right
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : SeminormedAddCommGroup E
        inst✝¹ : InnerProductSpace 𝕜 E
        ι : Type u_4
        inst✝ : DecidableEq ι
        v : ι → E
        h : ∀ (i j : ι), Eq (Inner.inner (v i) (v j)) (ite (Eq i j) 1 0)
        ⊢ Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
      -/
    · intro i j hij
      /-
        case mpr.right
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : SeminormedAddCommGroup E
        inst✝¹ : InnerProductSpace 𝕜 E
        ι : Type u_4
        inst✝ : DecidableEq ι
        v : ι → E
        h : ∀ (i j : ι), Eq (Inner.inner (v i) (v j)) (ite (Eq i j) 1 0)
        i j : ι
        hij : Ne i j
        ⊢ Eq (Inner.inner (v i) (v j)) 0
      -/
      simpa [hij] using h i j
      /-
        🎉 no goals
      -/


/-- `if ... then ... else` characterization of a set of vectors being orthonormal.  (Inner product
equals Kronecker delta.) -/
theorem orthonormal_subtype_iff_ite [DecidableEq E] {s : Set E} :
    Orthonormal 𝕜 (Subtype.val : s → E) ↔ ∀ v ∈ s, ∀ w ∈ s, ⟪v, w⟫ = if v = w then 1 else 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : DecidableEq E
    s : Set E
    ⊢ Iff (Orthonormal 𝕜 Subtype.val) (∀ (v : E), Membership.mem s v → ∀ (w : E),  …
  -/
  rw [orthonormal_iff_ite]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : DecidableEq E
    s : Set E
    ⊢ Iff (∀ (i j : Subtype fun x => Membership.mem s x), Eq (Inner.inner ↑i ↑j) ( …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : RCLike 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : DecidableEq E
      s : Set E
      ⊢ (∀ (i j : Subtype fun x => Membership.mem s x), Eq (Inner.inner ↑i ↑j) (ite  …
    -/
  · intro h v hv w hw
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : RCLike 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : DecidableEq E
      s : Set E
      h : ∀ (i j : Subtype fun x => Membership.mem s x), Eq (Inner.inner ↑i ↑j) (ite …
      v : E
      hv : Membership.mem s v
      w : E
      hw : Membership.mem s w
      ⊢ Eq (Inner.inner v w) (ite (Eq v w) 1 0)
    -/
    convert h ⟨v, hv⟩ ⟨w, hw⟩ using 1
    /-
      case h.e'_3
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : RCLike 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : DecidableEq E
      s : Set E
      h : ∀ (i j : Subtype fun x => Membership.mem s x), Eq (Inner.inner ↑i ↑j) (ite …
      v : E
      hv : Membership.mem s v
      w : E
      hw : Membership.mem s w
      ⊢ Eq (ite (Eq v w) 1 0) (ite (Eq ⟨v, hv⟩ ⟨w, hw⟩) 1 0)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : RCLike 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : DecidableEq E
      s : Set E
      ⊢ (∀ (v : E), Membership.mem s v → ∀ (w : E), Membership.mem s w → Eq (Inner.i …
    -/
  · rintro h ⟨v, hv⟩ ⟨w, hw⟩
    /-
      case mpr.mk.mk
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : RCLike 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : DecidableEq E
      s : Set E
      h : ∀ (v : E), Membership.mem s v → ∀ (w : E), Membership.mem s w → Eq (Inner. …
      v : E
      hv : Membership.mem s v
      w : E
      hw : Membership.mem s w
      ⊢ Eq (Inner.inner ↑⟨v, hv⟩ ↑⟨w, hw⟩) (ite (Eq ⟨v, hv⟩ ⟨w, hw⟩) 1 0)
    -/
    convert h v hv w hw using 1
    /-
      case h.e'_3
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : RCLike 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : DecidableEq E
      s : Set E
      h : ∀ (v : E), Membership.mem s v → ∀ (w : E), Membership.mem s w → Eq (Inner. …
      v : E
      hv : Membership.mem s v
      w : E
      hw : Membership.mem s w
      ⊢ Eq (ite (Eq ⟨v, hv⟩ ⟨w, hw⟩) 1 0) (ite (Eq v w) 1 0)
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The inner product of a linear combination of a set of orthonormal vectors with one of those
vectors picks out the coefficient of that vector. -/
theorem Orthonormal.inner_right_finsupp {v : ι → E} (hv : Orthonormal 𝕜 v) (l : ι →₀ 𝕜) (i : ι) :
    ⟪v i, linearCombination 𝕜 v l⟫ = l i := by
  classical
  simpa [linearCombination_apply, Finsupp.inner_sum, orthonormal_iff_ite.mp hv] using Eq.symm


/-- The inner product of a linear combination of a set of orthonormal vectors with one of those
vectors picks out the coefficient of that vector. -/
theorem Orthonormal.inner_right_sum {v : ι → E} (hv : Orthonormal 𝕜 v) (l : ι → 𝕜) {s : Finset ι}
    {i : ι} (hi : i ∈ s) : ⟪v i, ∑ i ∈ s, l i • v i⟫ = l i := by
  classical
  simp [inner_sum, inner_smul_right, orthonormal_iff_ite.mp hv, hi]


/-- The inner product of a linear combination of a set of orthonormal vectors with one of those
vectors picks out the coefficient of that vector. -/
theorem Orthonormal.inner_right_fintype [Fintype ι] {v : ι → E} (hv : Orthonormal 𝕜 v) (l : ι → 𝕜)
    (i : ι) : ⟪v i, ∑ i : ι, l i • v i⟫ = l i :=
  hv.inner_right_sum l (Finset.mem_univ _)


/-- The inner product of a linear combination of a set of orthonormal vectors with one of those
vectors picks out the coefficient of that vector. -/
theorem Orthonormal.inner_left_finsupp {v : ι → E} (hv : Orthonormal 𝕜 v) (l : ι →₀ 𝕜) (i : ι) :
                                                      /-
                                                        𝕜 : Type u_1
                                                        E : Type u_2
                                                        inst✝² : RCLike 𝕜
                                                        inst✝¹ : SeminormedAddCommGroup E
                                                        inst✝ : InnerProductSpace 𝕜 E
                                                        ι : Type u_4
                                                        v : ι → E
                                                        hv : Orthonormal 𝕜 v
                                                        l : Finsupp ι 𝕜
                                                        i : ι
                                                        ⊢ Eq (Inner.inner ((Finsupp.linearCombination 𝕜 v) l) (v i)) ((starRingEnd 𝕜)  …
                                                      -/
    ⟪linearCombination 𝕜 v l, v i⟫ = conj (l i) := by rw [← inner_conj_symm, hv.inner_right_finsupp]
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- The inner product of a linear combination of a set of orthonormal vectors with one of those
vectors picks out the coefficient of that vector. -/
theorem Orthonormal.inner_left_sum {v : ι → E} (hv : Orthonormal 𝕜 v) (l : ι → 𝕜) {s : Finset ι}
    {i : ι} (hi : i ∈ s) : ⟪∑ i ∈ s, l i • v i, v i⟫ = conj (l i) := by
  classical
  simp only [sum_inner, inner_smul_left, orthonormal_iff_ite.mp hv, hi, mul_boole,
    Finset.sum_ite_eq', if_true]


/-- The inner product of a linear combination of a set of orthonormal vectors with one of those
vectors picks out the coefficient of that vector. -/
theorem Orthonormal.inner_left_fintype [Fintype ι] {v : ι → E} (hv : Orthonormal 𝕜 v) (l : ι → 𝕜)
    (i : ι) : ⟪∑ i : ι, l i • v i, v i⟫ = conj (l i) :=
  hv.inner_left_sum l (Finset.mem_univ _)


/-- The inner product of two linear combinations of a set of orthonormal vectors, expressed as
a sum over the first `Finsupp`. -/
theorem Orthonormal.inner_finsupp_eq_sum_left {v : ι → E} (hv : Orthonormal 𝕜 v) (l₁ l₂ : ι →₀ 𝕜) :
    ⟪linearCombination 𝕜 v l₁, linearCombination 𝕜 v l₂⟫ = l₁.sum fun i y => conj y * l₂ i := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Orthonormal 𝕜 v
    l₁ l₂ : Finsupp ι 𝕜
    ⊢ Eq (Inner.inner ((Finsupp.linearCombination 𝕜 v) l₁) ((Finsupp.linearCombina …
  -/
  simp only [l₁.linearCombination_apply _, Finsupp.sum_inner, hv.inner_right_finsupp, smul_eq_mul]
  /-
    🎉 no goals
  -/


/-- The inner product of two linear combinations of a set of orthonormal vectors, expressed as
a sum over the second `Finsupp`. -/
theorem Orthonormal.inner_finsupp_eq_sum_right {v : ι → E} (hv : Orthonormal 𝕜 v) (l₁ l₂ : ι →₀ 𝕜) :
    ⟪linearCombination 𝕜 v l₁, linearCombination 𝕜 v l₂⟫ = l₂.sum fun i y => conj (l₁ i) * y := by
  simp only [l₂.linearCombination_apply _, Finsupp.inner_sum, hv.inner_left_finsupp, mul_comm,
             smul_eq_mul]


/-- The inner product of two linear combinations of a set of orthonormal vectors, expressed as
a sum. -/
protected theorem Orthonormal.inner_sum {v : ι → E} (hv : Orthonormal 𝕜 v) (l₁ l₂ : ι → 𝕜)
    (s : Finset ι) : ⟪∑ i ∈ s, l₁ i • v i, ∑ i ∈ s, l₂ i • v i⟫ = ∑ i ∈ s, conj (l₁ i) * l₂ i := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Orthonormal 𝕜 v
    l₁ l₂ : ι → 𝕜
    s : Finset ι
    ⊢ Eq (Inner.inner (s.sum fun i => HSMul.hSMul (l₁ i) (v i)) (s.sum fun i => HS …
  -/
  simp_rw [sum_inner, inner_smul_left]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Orthonormal 𝕜 v
    l₁ l₂ : ι → 𝕜
    s : Finset ι
    ⊢ Eq (s.sum fun x => HMul.hMul ((starRingEnd 𝕜) (l₁ x)) (Inner.inner (v x) (s. …
  -/
  refine Finset.sum_congr rfl fun i hi => ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Orthonormal 𝕜 v
    l₁ l₂ : ι → 𝕜
    s : Finset ι
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (HMul.hMul ((starRingEnd 𝕜) (l₁ i)) (Inner.inner (v i) (s.sum fun i => HS …
  -/
  rw [hv.inner_right_sum l₂ hi]
  /-
    🎉 no goals
  -/


/--
The double sum of weighted inner products of pairs of vectors from an orthonormal sequence is the
sum of the weights.
-/
theorem Orthonormal.inner_left_right_finset {s : Finset ι} {v : ι → E} (hv : Orthonormal 𝕜 v)
    {a : ι → ι → 𝕜} : (∑ i ∈ s, ∑ j ∈ s, a i j • ⟪v j, v i⟫) = ∑ k ∈ s, a k k := by
  classical
  simp [orthonormal_iff_ite.mp hv, Finset.sum_ite_of_true]


/-- An orthonormal set is linearly independent. -/
theorem Orthonormal.linearIndependent {v : ι → E} (hv : Orthonormal 𝕜 v) :
    LinearIndependent 𝕜 v := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Orthonormal 𝕜 v
    ⊢ LinearIndependent 𝕜 v
  -/
  rw [linearIndependent_iff]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Orthonormal 𝕜 v
    ⊢ ∀ (l : Finsupp ι 𝕜), Eq ((Finsupp.linearCombination 𝕜 v) l) 0 → Eq l 0
  -/
  intro l hl
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Orthonormal 𝕜 v
    l : Finsupp ι 𝕜
    hl : Eq ((Finsupp.linearCombination 𝕜 v) l) 0
    ⊢ Eq l 0
  -/
  ext i
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Orthonormal 𝕜 v
    l : Finsupp ι 𝕜
    hl : Eq ((Finsupp.linearCombination 𝕜 v) l) 0
    i : ι
    ⊢ Eq (l i) (0 i)
  -/
  have key : ⟪v i, Finsupp.linearCombination 𝕜 v l⟫ = ⟪v i, 0⟫ := by rw [hl]
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Orthonormal 𝕜 v
    l : Finsupp ι 𝕜
    hl : Eq ((Finsupp.linearCombination 𝕜 v) l) 0
    i : ι
    key : Eq (Inner.inner (v i) ((Finsupp.linearCombination 𝕜 v) l)) (Inner.inner  …
    ⊢ Eq (l i) (0 i)
  -/
  simpa only [hv.inner_right_finsupp, inner_zero_right] using key
  /-
    🎉 no goals
  -/


/-- A subfamily of an orthonormal family (i.e., a composition with an injective map) is an
orthonormal family. -/
theorem Orthonormal.comp {ι' : Type*} {v : ι → E} (hv : Orthonormal 𝕜 v) (f : ι' → ι)
    (hf : Function.Injective f) : Orthonormal 𝕜 (v ∘ f) := by
  classical
  rw [orthonormal_iff_ite] at hv ⊢
  intro i j
  convert hv (f i) (f j) using 1
  simp [hf.eq_iff]


/-- An injective family `v : ι → E` is orthonormal if and only if `Subtype.val : (range v) → E` is
orthonormal. -/
theorem orthonormal_subtype_range {v : ι → E} (hv : Function.Injective v) :
    Orthonormal 𝕜 (Subtype.val : Set.range v → E) ↔ Orthonormal 𝕜 v := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Function.Injective v
    ⊢ Iff (Orthonormal 𝕜 Subtype.val) (Orthonormal 𝕜 v)
  -/
  let f : ι ≃ Set.range v := Equiv.ofInjective v hv
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Function.Injective v
    f : Equiv ι ↑(Set.range v) := Equiv.ofInjective v hv
    ⊢ Iff (Orthonormal 𝕜 Subtype.val) (Orthonormal 𝕜 v)
  -/
  refine ⟨fun h => h.comp f f.injective, fun h => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Function.Injective v
    f : Equiv ι ↑(Set.range v) := Equiv.ofInjective v hv
    h : Orthonormal 𝕜 v
    ⊢ Orthonormal 𝕜 Subtype.val
  -/
  rw [← Equiv.self_comp_ofInjective_symm hv]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Function.Injective v
    f : Equiv ι ↑(Set.range v) := Equiv.ofInjective v hv
    h : Orthonormal 𝕜 v
    ⊢ Orthonormal 𝕜 (Function.comp v ⇑(Equiv.ofInjective v hv).symm)
  -/
  exact h.comp f.symm f.symm.injective
  /-
    🎉 no goals
  -/


/-- If `v : ι → E` is an orthonormal family, then `Subtype.val : (range v) → E` is an orthonormal
family. -/
theorem Orthonormal.toSubtypeRange {v : ι → E} (hv : Orthonormal 𝕜 v) :
    Orthonormal 𝕜 (Subtype.val : Set.range v → E) :=
  (orthonormal_subtype_range hv.linearIndependent.injective).2 hv


/-- A linear combination of some subset of an orthonormal set is orthogonal to other members of the
set. -/
theorem Orthonormal.inner_finsupp_eq_zero {v : ι → E} (hv : Orthonormal 𝕜 v) {s : Set ι} {i : ι}
    (hi : i ∉ s) {l : ι →₀ 𝕜} (hl : l ∈ Finsupp.supported 𝕜 𝕜 s) :
    ⟪Finsupp.linearCombination 𝕜 v l, v i⟫ = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Orthonormal 𝕜 v
    s : Set ι
    i : ι
    hi : Not (Membership.mem s i)
    l : Finsupp ι 𝕜
    hl : Membership.mem (Finsupp.supported 𝕜 𝕜 s) l
    ⊢ Eq (Inner.inner ((Finsupp.linearCombination 𝕜 v) l) (v i)) 0
  -/
  rw [Finsupp.mem_supported'] at hl
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Orthonormal 𝕜 v
    s : Set ι
    i : ι
    hi : Not (Membership.mem s i)
    l : Finsupp ι 𝕜
    hl : ∀ (x : ι), Not (Membership.mem s x) → Eq (l x) 0
    ⊢ Eq (Inner.inner ((Finsupp.linearCombination 𝕜 v) l) (v i)) 0
  -/
  simp only [hv.inner_left_finsupp, hl i hi, map_zero]
  /-
    🎉 no goals
  -/


/-- Given an orthonormal family, a second family of vectors is orthonormal if every vector equals
the corresponding vector in the original family or its negation. -/
theorem Orthonormal.orthonormal_of_forall_eq_or_eq_neg {v w : ι → E} (hv : Orthonormal 𝕜 v)
    (hw : ∀ i, w i = v i ∨ w i = -v i) : Orthonormal 𝕜 w := by
  classical
  rw [orthonormal_iff_ite] at *
  intro i j
  cases' hw i with hi hi <;> cases' hw j with hj hj <;>
    replace hv := hv i j <;> split_ifs at hv ⊢ with h <;>
    simpa only [hi, hj, h, inner_neg_right, inner_neg_left, neg_neg, eq_self_iff_true,
      neg_eq_zero] using hv

/- The material that follows, culminating in the existence of a maximal orthonormal subset, is
adapted from the corresponding development of the theory of linearly independents sets.  See
`exists_linearIndependent` in particular. -/

theorem orthonormal_empty : Orthonormal 𝕜 (fun x => x : (∅ : Set E) → E) := by
  classical
  simp [orthonormal_subtype_iff_ite]


theorem orthonormal_iUnion_of_directed {η : Type*} {s : η → Set E} (hs : Directed (· ⊆ ·) s)
    (h : ∀ i, Orthonormal 𝕜 (fun x => x : s i → E)) :
    Orthonormal 𝕜 (fun x => x : (⋃ i, s i) → E) := by
  classical
  rw [orthonormal_subtype_iff_ite]
  rintro x ⟨_, ⟨i, rfl⟩, hxi⟩ y ⟨_, ⟨j, rfl⟩, hyj⟩
  obtain ⟨k, hik, hjk⟩ := hs i j
  have h_orth : Orthonormal 𝕜 (fun x => x : s k → E) := h k
  rw [orthonormal_subtype_iff_ite] at h_orth
  exact h_orth x (hik hxi) y (hjk hyj)


theorem orthonormal_sUnion_of_directed {s : Set (Set E)} (hs : DirectedOn (· ⊆ ·) s)
    (h : ∀ a ∈ s, Orthonormal 𝕜 (fun x => ((x : a) : E))) :
    Orthonormal 𝕜 (fun x => x : ⋃₀ s → E) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    s : Set (Set E)
    hs : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) s
    h : ∀ (a : Set E), Membership.mem s a → Orthonormal 𝕜 fun x => ↑x
    ⊢ Orthonormal 𝕜 fun x => ↑x
  -/
  rw [Set.sUnion_eq_iUnion]; exact orthonormal_iUnion_of_directed hs.directed_val (by simpa using h)
                             /-
                               🎉 no goals
                             -/


/-- Given an orthonormal set `v` of vectors in `E`, there exists a maximal orthonormal set
containing it. -/
theorem exists_maximal_orthonormal {s : Set E} (hs : Orthonormal 𝕜 (Subtype.val : s → E)) :
    ∃ w ⊇ s, Orthonormal 𝕜 (Subtype.val : w → E) ∧
      ∀ u ⊇ w, Orthonormal 𝕜 (Subtype.val : u → E) → u = w := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    s : Set E
    hs : Orthonormal 𝕜 Subtype.val
    ⊢ Exists fun w => And (Superset w s) (And (Orthonormal 𝕜 Subtype.val) (∀ (u :  …
  -/
  have := zorn_subset_nonempty { b | Orthonormal 𝕜 (Subtype.val : b → E) } ?_ _ hs
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      s : Set E
      hs : Orthonormal 𝕜 Subtype.val
      this : Exists fun m => And (HasSubset.Subset s m) (Maximal (fun x => Membershi …
      ⊢ Exists fun w => And (Superset w s) (And (Orthonormal 𝕜 Subtype.val) (∀ (u :  …
    -/
  · obtain ⟨b, hb⟩ := this
    /-
      case refine_2.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      s : Set E
      hs : Orthonormal 𝕜 Subtype.val
      b : Set E
      hb : And (HasSubset.Subset s b) (Maximal (fun x => Membership.mem (setOf fun b …
      ⊢ Exists fun w => And (Superset w s) (And (Orthonormal 𝕜 Subtype.val) (∀ (u :  …
    -/
    exact ⟨b, hb.1, hb.2.1, fun u hus hu => hb.2.eq_of_ge hu hus⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      s : Set E
      hs : Orthonormal 𝕜 Subtype.val
      ⊢ ∀ (c : Set (Set E)), HasSubset.Subset c (setOf fun b => Orthonormal 𝕜 Subtyp …
    -/
  · refine fun c hc cc _c0 => ⟨⋃₀ c, ?_, ?_⟩
      /-
        case refine_1.refine_1
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : RCLike 𝕜
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : InnerProductSpace 𝕜 E
        s : Set E
        hs : Orthonormal 𝕜 Subtype.val
        c : Set (Set E)
        hc : HasSubset.Subset c (setOf fun b => Orthonormal 𝕜 Subtype.val)
        cc : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) c
        _c0 : c.Nonempty
        ⊢ Membership.mem (setOf fun b => Orthonormal 𝕜 Subtype.val) c.sUnion
      -/
    · exact orthonormal_sUnion_of_directed cc.directedOn fun x xc => hc xc
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : RCLike 𝕜
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : InnerProductSpace 𝕜 E
        s : Set E
        hs : Orthonormal 𝕜 Subtype.val
        c : Set (Set E)
        hc : HasSubset.Subset c (setOf fun b => Orthonormal 𝕜 Subtype.val)
        cc : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) c
        _c0 : c.Nonempty
        ⊢ ∀ (s : Set E), Membership.mem c s → HasSubset.Subset s c.sUnion
      -/
    · exact fun _ => Set.subset_sUnion_of_mem
      /-
        🎉 no goals
      -/


/-- A family of orthonormal vectors with the correct cardinality forms a basis. -/
def basisOfOrthonormalOfCardEqFinrank [Fintype ι] [Nonempty ι] {v : ι → E} (hv : Orthonormal 𝕜 v)
    (card_eq : Fintype.card ι = finrank 𝕜 E) : Basis ι 𝕜 E :=
  basisOfLinearIndependentOfCardEqFinrank hv.linearIndependent card_eq


@[simp]
theorem coe_basisOfOrthonormalOfCardEqFinrank [Fintype ι] [Nonempty ι] {v : ι → E}
    (hv : Orthonormal 𝕜 v) (card_eq : Fintype.card ι = finrank 𝕜 E) :
    (basisOfOrthonormalOfCardEqFinrank hv card_eq : ι → E) = v :=
  coe_basisOfLinearIndependentOfCardEqFinrank _ _


theorem Orthonormal.ne_zero {v : ι → E} (hv : Orthonormal 𝕜 v) (i : ι) : v i ≠ 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Orthonormal 𝕜 v
    i : ι
    ⊢ Ne (v i) 0
  -/
  refine ne_of_apply_ne norm ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Orthonormal 𝕜 v
    i : ι
    ⊢ Ne (Norm.norm (v i)) (Norm.norm 0)
  -/
  rw [hv.1 i, norm_zero]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hv : Orthonormal 𝕜 v
    i : ι
    ⊢ Ne 1 0
  -/
  norm_num
  /-
    🎉 no goals
  -/


/-- A linear isometry preserves the property of being orthonormal. -/
theorem LinearIsometry.orthonormal_comp_iff {v : ι → E} (f : E →ₗᵢ[𝕜] E') :
    Orthonormal 𝕜 (f ∘ v) ↔ Orthonormal 𝕜 v := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    ι : Type u_4
    E' : Type u_7
    inst✝¹ : SeminormedAddCommGroup E'
    inst✝ : InnerProductSpace 𝕜 E'
    v : ι → E
    f : LinearIsometry (RingHom.id 𝕜) E E'
    ⊢ Iff (Orthonormal 𝕜 (Function.comp (⇑f) v)) (Orthonormal 𝕜 v)
  -/
  classical simp_rw [orthonormal_iff_ite, Function.comp_apply, LinearIsometry.inner_map_map]
  /-
    🎉 no goals
  -/


/-- A linear isometry preserves the property of being orthonormal. -/
theorem Orthonormal.comp_linearIsometry {v : ι → E} (hv : Orthonormal 𝕜 v) (f : E →ₗᵢ[𝕜] E') :
                                /-
                                  𝕜 : Type u_1
                                  E : Type u_2
                                  inst✝⁴ : RCLike 𝕜
                                  inst✝³ : SeminormedAddCommGroup E
                                  inst✝² : InnerProductSpace 𝕜 E
                                  ι : Type u_4
                                  E' : Type u_7
                                  inst✝¹ : SeminormedAddCommGroup E'
                                  inst✝ : InnerProductSpace 𝕜 E'
                                  v : ι → E
                                  hv : Orthonormal 𝕜 v
                                  f : LinearIsometry (RingHom.id 𝕜) E E'
                                  ⊢ Orthonormal 𝕜 (Function.comp (⇑f) v)
                                -/
    Orthonormal 𝕜 (f ∘ v) := by rwa [f.orthonormal_comp_iff]
                                /-
                                  🎉 no goals
                                -/


/-- A linear isometric equivalence preserves the property of being orthonormal. -/
theorem Orthonormal.comp_linearIsometryEquiv {v : ι → E} (hv : Orthonormal 𝕜 v) (f : E ≃ₗᵢ[𝕜] E') :
    Orthonormal 𝕜 (f ∘ v) :=
  hv.comp_linearIsometry f.toLinearIsometry


/-- A linear isometric equivalence, applied with `Basis.map`, preserves the property of being
orthonormal. -/
theorem Orthonormal.mapLinearIsometryEquiv {v : Basis ι 𝕜 E} (hv : Orthonormal 𝕜 v)
    (f : E ≃ₗᵢ[𝕜] E') : Orthonormal 𝕜 (v.map f.toLinearEquiv) :=
  hv.comp_linearIsometryEquiv f


/-- A linear map that sends an orthonormal basis to orthonormal vectors is a linear isometry. -/
def LinearMap.isometryOfOrthonormal (f : E →ₗ[𝕜] E') {v : Basis ι 𝕜 E} (hv : Orthonormal 𝕜 v)
    (hf : Orthonormal 𝕜 (f ∘ v)) : E →ₗᵢ[𝕜] E' :=
  f.isometryOfInner fun x y => by
    classical rw [← v.linearCombination_repr x, ← v.linearCombination_repr y,
      Finsupp.apply_linearCombination, Finsupp.apply_linearCombination,
      hv.inner_finsupp_eq_sum_left, hf.inner_finsupp_eq_sum_left]


@[simp]
theorem LinearMap.coe_isometryOfOrthonormal (f : E →ₗ[𝕜] E') {v : Basis ι 𝕜 E}
    (hv : Orthonormal 𝕜 v) (hf : Orthonormal 𝕜 (f ∘ v)) : ⇑(f.isometryOfOrthonormal hv hf) = f :=
  rfl


@[simp]
theorem LinearMap.isometryOfOrthonormal_toLinearMap (f : E →ₗ[𝕜] E') {v : Basis ι 𝕜 E}
    (hv : Orthonormal 𝕜 v) (hf : Orthonormal 𝕜 (f ∘ v)) :
    (f.isometryOfOrthonormal hv hf).toLinearMap = f :=
  rfl


/-- A linear equivalence that sends an orthonormal basis to orthonormal vectors is a linear
isometric equivalence. -/
def LinearEquiv.isometryOfOrthonormal (f : E ≃ₗ[𝕜] E') {v : Basis ι 𝕜 E} (hv : Orthonormal 𝕜 v)
    (hf : Orthonormal 𝕜 (f ∘ v)) : E ≃ₗᵢ[𝕜] E' :=
  f.isometryOfInner fun x y => by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : SeminormedAddCommGroup E
      inst✝⁶ : InnerProductSpace 𝕜 E
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : InnerProductSpace Real F
      ι : Type u_4
      ι' : Type u_5
      ι'' : Type u_6
      E' : Type u_7
      inst✝³ : SeminormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      E'' : Type u_8
      inst✝¹ : SeminormedAddCommGroup E''
      inst✝ : InnerProductSpace 𝕜 E''
      f : LinearEquiv (RingHom.id 𝕜) E E'
      v : Basis ι 𝕜 E
      hv : Orthonormal 𝕜 ⇑v
      hf : Orthonormal 𝕜 (Function.comp ⇑f ⇑v)
      x y : E
      ⊢ Eq (Inner.inner (f x) (f y)) (Inner.inner x y)
    -/
    rw [← LinearEquiv.coe_coe] at hf
    classical rw [← v.linearCombination_repr x, ← v.linearCombination_repr y,
      ← LinearEquiv.coe_coe f, Finsupp.apply_linearCombination,
      Finsupp.apply_linearCombination, hv.inner_finsupp_eq_sum_left, hf.inner_finsupp_eq_sum_left]


@[simp]
theorem LinearEquiv.coe_isometryOfOrthonormal (f : E ≃ₗ[𝕜] E') {v : Basis ι 𝕜 E}
    (hv : Orthonormal 𝕜 v) (hf : Orthonormal 𝕜 (f ∘ v)) : ⇑(f.isometryOfOrthonormal hv hf) = f :=
  rfl


@[simp]
theorem LinearEquiv.isometryOfOrthonormal_toLinearEquiv (f : E ≃ₗ[𝕜] E') {v : Basis ι 𝕜 E}
    (hv : Orthonormal 𝕜 v) (hf : Orthonormal 𝕜 (f ∘ v)) :
    (f.isometryOfOrthonormal hv hf).toLinearEquiv = f :=
  rfl


/-- A linear isometric equivalence that sends an orthonormal basis to a given orthonormal basis. -/
def Orthonormal.equiv {v : Basis ι 𝕜 E} (hv : Orthonormal 𝕜 v) {v' : Basis ι' 𝕜 E'}
    (hv' : Orthonormal 𝕜 v') (e : ι ≃ ι') : E ≃ₗᵢ[𝕜] E' :=
  (v.equiv v' e).isometryOfOrthonormal hv
    (by
      have h : v.equiv v' e ∘ v = v' ∘ e := by
        ext i
        simp
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁸ : RCLike 𝕜
        inst✝⁷ : SeminormedAddCommGroup E
        inst✝⁶ : InnerProductSpace 𝕜 E
        inst✝⁵ : SeminormedAddCommGroup F
        inst✝⁴ : InnerProductSpace Real F
        ι : Type u_4
        ι' : Type u_5
        ι'' : Type u_6
        E' : Type u_7
        inst✝³ : SeminormedAddCommGroup E'
        inst✝² : InnerProductSpace 𝕜 E'
        E'' : Type u_8
        inst✝¹ : SeminormedAddCommGroup E''
        inst✝ : InnerProductSpace 𝕜 E''
        v : Basis ι 𝕜 E
        hv : Orthonormal 𝕜 ⇑v
        v' : Basis ι' 𝕜 E'
        hv' : Orthonormal 𝕜 ⇑v'
        e : Equiv ι ι'
        h : Eq (Function.comp ⇑(v.equiv v' e) ⇑v) (Function.comp ⇑v' ⇑e)
        ⊢ Orthonormal 𝕜 (Function.comp ⇑(v.equiv v' e) ⇑v)
      -/
      rw [h]
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁸ : RCLike 𝕜
        inst✝⁷ : SeminormedAddCommGroup E
        inst✝⁶ : InnerProductSpace 𝕜 E
        inst✝⁵ : SeminormedAddCommGroup F
        inst✝⁴ : InnerProductSpace Real F
        ι : Type u_4
        ι' : Type u_5
        ι'' : Type u_6
        E' : Type u_7
        inst✝³ : SeminormedAddCommGroup E'
        inst✝² : InnerProductSpace 𝕜 E'
        E'' : Type u_8
        inst✝¹ : SeminormedAddCommGroup E''
        inst✝ : InnerProductSpace 𝕜 E''
        v : Basis ι 𝕜 E
        hv : Orthonormal 𝕜 ⇑v
        v' : Basis ι' 𝕜 E'
        hv' : Orthonormal 𝕜 ⇑v'
        e : Equiv ι ι'
        h : Eq (Function.comp ⇑(v.equiv v' e) ⇑v) (Function.comp ⇑v' ⇑e)
        ⊢ Orthonormal 𝕜 (Function.comp ⇑v' ⇑e)
      -/
      classical exact hv'.comp _ e.injective)
      /-
        🎉 no goals
      -/


@[simp]
theorem Orthonormal.equiv_toLinearEquiv {v : Basis ι 𝕜 E} (hv : Orthonormal 𝕜 v)
    {v' : Basis ι' 𝕜 E'} (hv' : Orthonormal 𝕜 v') (e : ι ≃ ι') :
    (hv.equiv hv' e).toLinearEquiv = v.equiv v' e :=
  rfl


@[simp]
theorem Orthonormal.equiv_apply {ι' : Type*} {v : Basis ι 𝕜 E} (hv : Orthonormal 𝕜 v)
    {v' : Basis ι' 𝕜 E'} (hv' : Orthonormal 𝕜 v') (e : ι ≃ ι') (i : ι) :
    hv.equiv hv' e (v i) = v' (e i) :=
  Basis.equiv_apply _ _ _ _


@[simp]
theorem Orthonormal.equiv_trans {v : Basis ι 𝕜 E} (hv : Orthonormal 𝕜 v) {v' : Basis ι' 𝕜 E'}
    (hv' : Orthonormal 𝕜 v') (e : ι ≃ ι') {v'' : Basis ι'' 𝕜 E''} (hv'' : Orthonormal 𝕜 v'')
    (e' : ι' ≃ ι'') : (hv.equiv hv' e).trans (hv'.equiv hv'' e') = hv.equiv hv'' (e.trans e') :=
  v.ext_linearIsometryEquiv fun i => by
    simp only [LinearIsometryEquiv.trans_apply, Orthonormal.equiv_apply, e.coe_trans,
      Function.comp_apply]


theorem Orthonormal.map_equiv {v : Basis ι 𝕜 E} (hv : Orthonormal 𝕜 v) {v' : Basis ι' 𝕜 E'}
    (hv' : Orthonormal 𝕜 v') (e : ι ≃ ι') :
    v.map (hv.equiv hv' e).toLinearEquiv = v'.reindex e.symm :=
  v.map_equiv _ _


@[simp]
theorem Orthonormal.equiv_refl {v : Basis ι 𝕜 E} (hv : Orthonormal 𝕜 v) :
    hv.equiv hv (Equiv.refl ι) = LinearIsometryEquiv.refl 𝕜 E :=
  v.ext_linearIsometryEquiv fun i => by
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      ι : Type u_4
      v : Basis ι 𝕜 E
      hv : Orthonormal 𝕜 ⇑v
      i : ι
      ⊢ Eq ((hv.equiv hv (Equiv.refl ι)) (v i)) ((LinearIsometryEquiv.refl 𝕜 E) (v i))
    -/
    simp only [Orthonormal.equiv_apply, Equiv.coe_refl, id, LinearIsometryEquiv.coe_refl]
    /-
      🎉 no goals
    -/


@[simp]
theorem Orthonormal.equiv_symm {v : Basis ι 𝕜 E} (hv : Orthonormal 𝕜 v) {v' : Basis ι' 𝕜 E'}
    (hv' : Orthonormal 𝕜 v') (e : ι ≃ ι') : (hv.equiv hv' e).symm = hv'.equiv hv e.symm :=
  v'.ext_linearIsometryEquiv fun i =>
    (hv.equiv hv' e).injective <| by
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁴ : RCLike 𝕜
        inst✝³ : SeminormedAddCommGroup E
        inst✝² : InnerProductSpace 𝕜 E
        ι : Type u_4
        ι' : Type u_5
        E' : Type u_6
        inst✝¹ : SeminormedAddCommGroup E'
        inst✝ : InnerProductSpace 𝕜 E'
        v : Basis ι 𝕜 E
        hv : Orthonormal 𝕜 ⇑v
        v' : Basis ι' 𝕜 E'
        hv' : Orthonormal 𝕜 ⇑v'
        e : Equiv ι ι'
        i : ι'
        ⊢ Eq ((hv.equiv hv' e) ((hv.equiv hv' e).symm (v' i))) ((hv.equiv hv' e) ((hv' …
      -/
      simp only [LinearIsometryEquiv.apply_symm_apply, Orthonormal.equiv_apply, e.apply_symm_apply]
      /-
        🎉 no goals
      -/


/-- Bessel's inequality for finite sums. -/
theorem Orthonormal.sum_inner_products_le {s : Finset ι} (hv : Orthonormal 𝕜 v) :
    ∑ i ∈ s, ‖⟪v i, x⟫‖ ^ 2 ≤ ‖x‖ ^ 2 := by
  have h₂ :
    (∑ i ∈ s, ∑ j ∈ s, ⟪v i, x⟫ * ⟪x, v j⟫ * ⟪v j, v i⟫) = (∑ k ∈ s, ⟪v k, x⟫ * ⟪x, v k⟫ : 𝕜) := by
    classical exact hv.inner_left_right_finset
  have h₃ : ∀ z : 𝕜, re (z * conj z) = ‖z‖ ^ 2 := by
    intro z
    simp only [mul_conj, normSq_eq_def']
    norm_cast
  suffices hbf : ‖x - ∑ i ∈ s, ⟪v i, x⟫ • v i‖ ^ 2 = ‖x‖ ^ 2 - ∑ i ∈ s, ‖⟪v i, x⟫‖ ^ 2 by
    rw [← sub_nonneg, ← hbf]
    simp only [norm_nonneg, pow_nonneg]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    x : E
    v : ι → E
    s : Finset ι
    hv : Orthonormal 𝕜 v
    h₂ : Eq (s.sum fun i => s.sum fun j => HMul.hMul (HMul.hMul (Inner.inner (v i) …
    h₃ : ∀ (z : 𝕜), Eq (RCLike.re (HMul.hMul z ((starRingEnd 𝕜) z))) (HPow.hPow (N …
    ⊢ Eq (HPow.hPow (Norm.norm (HSub.hSub x (s.sum fun i => HSMul.hSMul (Inner.inn …
  -/
  rw [@norm_sub_sq 𝕜, sub_add]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    x : E
    v : ι → E
    s : Finset ι
    hv : Orthonormal 𝕜 v
    h₂ : Eq (s.sum fun i => s.sum fun j => HMul.hMul (HMul.hMul (Inner.inner (v i) …
    h₃ : ∀ (z : 𝕜), Eq (RCLike.re (HMul.hMul z ((starRingEnd 𝕜) z))) (HPow.hPow (N …
    ⊢ Eq (HSub.hSub (HPow.hPow (Norm.norm x) 2) (HSub.hSub (HMul.hMul 2 (RCLike.re …
  -/
  simp only [@InnerProductSpace.norm_sq_eq_inner 𝕜 E, inner_sum, sum_inner]
  simp only [inner_smul_right, two_mul, inner_smul_left, inner_conj_symm, ← mul_assoc, h₂,
    add_sub_cancel_right, sub_right_inj]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    x : E
    v : ι → E
    s : Finset ι
    hv : Orthonormal 𝕜 v
    h₂ : Eq (s.sum fun i => s.sum fun j => HMul.hMul (HMul.hMul (Inner.inner (v i) …
    h₃ : ∀ (z : 𝕜), Eq (RCLike.re (HMul.hMul z ((starRingEnd 𝕜) z))) (HPow.hPow (N …
    ⊢ Eq (RCLike.re (s.sum fun x_1 => HMul.hMul (Inner.inner (v x_1) x) (Inner.inn …
  -/
  simp only [map_sum, ← inner_conj_symm x, ← h₃]
  /-
    🎉 no goals
  -/


/-- Bessel's inequality. -/
theorem Orthonormal.tsum_inner_products_le (hv : Orthonormal 𝕜 v) :
    ∑' i, ‖⟪v i, x⟫‖ ^ 2 ≤ ‖x‖ ^ 2 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    x : E
    v : ι → E
    hv : Orthonormal 𝕜 v
    ⊢ LE.le (tsum fun i => HPow.hPow (Norm.norm (Inner.inner (v i) x)) 2) (HPow.hP …
  -/
  refine tsum_le_of_sum_le' ?_ fun s => hv.sum_inner_products_le x
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    x : E
    v : ι → E
    hv : Orthonormal 𝕜 v
    ⊢ LE.le 0 (HPow.hPow (Norm.norm x) 2)
  -/
  simp only [norm_nonneg, pow_nonneg]
  /-
    🎉 no goals
  -/


/-- The sum defined in Bessel's inequality is summable. -/
theorem Orthonormal.inner_products_summable (hv : Orthonormal 𝕜 v) :
    Summable fun i => ‖⟪v i, x⟫‖ ^ 2 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    x : E
    v : ι → E
    hv : Orthonormal 𝕜 v
    ⊢ Summable fun i => HPow.hPow (Norm.norm (Inner.inner (v i) x)) 2
  -/
  use ⨆ s : Finset ι, ∑ i ∈ s, ‖⟪v i, x⟫‖ ^ 2
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    x : E
    v : ι → E
    hv : Orthonormal 𝕜 v
    ⊢ HasSum (fun i => HPow.hPow (Norm.norm (Inner.inner (v i) x)) 2) (iSup fun s  …
  -/
  apply hasSum_of_isLUB_of_nonneg
    /-
      case h.h
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      ι : Type u_4
      x : E
      v : ι → E
      hv : Orthonormal 𝕜 v
      ⊢ ∀ (i : ι), LE.le 0 (HPow.hPow (Norm.norm (Inner.inner (v i) x)) 2)
    -/
  · intro b
    /-
      case h.h
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      ι : Type u_4
      x : E
      v : ι → E
      hv : Orthonormal 𝕜 v
      b : ι
      ⊢ LE.le 0 (HPow.hPow (Norm.norm (Inner.inner (v b) x)) 2)
    -/
    simp only [norm_nonneg, pow_nonneg]
    /-
      🎉 no goals
    -/
    /-
      case h.hf
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      ι : Type u_4
      x : E
      v : ι → E
      hv : Orthonormal 𝕜 v
      ⊢ IsLUB (Set.range fun s => s.sum fun i => HPow.hPow (Norm.norm (Inner.inner ( …
    -/
  · refine isLUB_ciSup ?_
    /-
      case h.hf
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      ι : Type u_4
      x : E
      v : ι → E
      hv : Orthonormal 𝕜 v
      ⊢ BddAbove (Set.range fun s => s.sum fun i => HPow.hPow (Norm.norm (Inner.inne …
    -/
    use ‖x‖ ^ 2
    /-
      case h
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      ι : Type u_4
      x : E
      v : ι → E
      hv : Orthonormal 𝕜 v
      ⊢ Membership.mem (upperBounds (Set.range fun s => s.sum fun i => HPow.hPow (No …
    -/
    rintro y ⟨s, rfl⟩
    /-
      case h.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      ι : Type u_4
      x : E
      v : ι → E
      hv : Orthonormal 𝕜 v
      s : Finset ι
      ⊢ LE.le ((fun s => s.sum fun i => HPow.hPow (Norm.norm (Inner.inner (v i) x))  …
    -/
    exact hv.sum_inner_products_le x
    /-
      🎉 no goals
    -/


