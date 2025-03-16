theorem univ_fin2 : (univ : Finset (Fin 2)) = {0, 1} := by
  /-
    ⊢ Eq Finset.univ (Insert.insert 0 (Singleton.singleton 1))
  -/
  ext x
  /-
    case h
    x : Fin 2
    ⊢ Iff (Membership.mem Finset.univ x) (Membership.mem (Insert.insert 0 (Singlet …
  -/
                  /-
                    🎉 no goals
                  -/
  fin_cases x <;> simp
                  /-
                    🎉 no goals
                  -/


/-- A weighted sum of the results of subtracting a base point from the
given points, as a linear map on the weights.  The main cases of
interest are where the sum of the weights is 0, in which case the sum
is independent of the choice of base point, and where the sum of the
weights is 1, in which case the sum added to the base point is
independent of the choice of base point. -/
def weightedVSubOfPoint (p : ι → P) (b : P) : (ι → k) →ₗ[k] V :=
  ∑ i ∈ s, (LinearMap.proj i : (ι → k) →ₗ[k] k).smulRight (p i -ᵥ b)


@[simp]
theorem weightedVSubOfPoint_apply (w : ι → k) (p : ι → P) (b : P) :
    s.weightedVSubOfPoint p b w = ∑ i ∈ s, w i • (p i -ᵥ b) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    b : P
    ⊢ Eq ((s.weightedVSubOfPoint p b) w) (s.sum fun i => HSMul.hSMul (w i) (VSub.v …
  -/
  simp [weightedVSubOfPoint, LinearMap.sum_apply]
  /-
    🎉 no goals
  -/


/-- The value of `weightedVSubOfPoint`, where the given points are equal. -/
@[simp (high)]
theorem weightedVSubOfPoint_apply_const (w : ι → k) (p : P) (b : P) :
    s.weightedVSubOfPoint (fun _ => p) b w = (∑ i ∈ s, w i) • (p -ᵥ b) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p b : P
    ⊢ Eq ((s.weightedVSubOfPoint (fun x => p) b) w) (HSMul.hSMul (s.sum fun i => w …
  -/
  rw [weightedVSubOfPoint_apply, sum_smul]
  /-
    🎉 no goals
  -/


lemma weightedVSubOfPoint_vadd (s : Finset ι) (w : ι → k) (p : ι → P) (b : P) (v : V) :
    s.weightedVSubOfPoint (v +ᵥ p) b w = s.weightedVSubOfPoint p (-v +ᵥ b) w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    b : P
    v : V
    ⊢ Eq ((s.weightedVSubOfPoint (HVAdd.hVAdd v p) b) w) ((s.weightedVSubOfPoint p …
  -/
  simp [vadd_vsub_assoc, vsub_vadd_eq_vsub_sub, add_comm]
  /-
    🎉 no goals
  -/


lemma weightedVSubOfPoint_smul {G : Type*} [Group G] [DistribMulAction G V] [SMulCommClass G k V]
    (s : Finset ι) (w : ι → k) (p : ι → V) (b : V) (a : G) :
    s.weightedVSubOfPoint (a • p) b w = a • s.weightedVSubOfPoint p (a⁻¹ • b) w := by
  /-
    k : Type u_1
    V : Type u_2
    inst✝⁵ : Ring k
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module k V
    ι : Type u_4
    G : Type u_6
    inst✝² : Group G
    inst✝¹ : DistribMulAction G V
    inst✝ : SMulCommClass G k V
    s : Finset ι
    w : ι → k
    p : ι → V
    b : V
    a : G
    ⊢ Eq ((s.weightedVSubOfPoint (HSMul.hSMul a p) b) w) (HSMul.hSMul a ((s.weight …
  -/
  simp [smul_sum, smul_sub, smul_comm a (w _)]
  /-
    🎉 no goals
  -/


/-- `weightedVSubOfPoint` gives equal results for two families of weights and two families of
points that are equal on `s`. -/
theorem weightedVSubOfPoint_congr {w₁ w₂ : ι → k} (hw : ∀ i ∈ s, w₁ i = w₂ i) {p₁ p₂ : ι → P}
    (hp : ∀ i ∈ s, p₁ i = p₂ i) (b : P) :
    s.weightedVSubOfPoint p₁ b w₁ = s.weightedVSubOfPoint p₂ b w₂ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w₁ w₂ : ι → k
    hw : ∀ (i : ι), Membership.mem s i → Eq (w₁ i) (w₂ i)
    p₁ p₂ : ι → P
    hp : ∀ (i : ι), Membership.mem s i → Eq (p₁ i) (p₂ i)
    b : P
    ⊢ Eq ((s.weightedVSubOfPoint p₁ b) w₁) ((s.weightedVSubOfPoint p₂ b) w₂)
  -/
  simp_rw [weightedVSubOfPoint_apply]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w₁ w₂ : ι → k
    hw : ∀ (i : ι), Membership.mem s i → Eq (w₁ i) (w₂ i)
    p₁ p₂ : ι → P
    hp : ∀ (i : ι), Membership.mem s i → Eq (p₁ i) (p₂ i)
    b : P
    ⊢ Eq (s.sum fun i => HSMul.hSMul (w₁ i) (VSub.vsub (p₁ i) b)) (s.sum fun i =>  …
  -/
  refine sum_congr rfl fun i hi => ?_
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w₁ w₂ : ι → k
    hw : ∀ (i : ι), Membership.mem s i → Eq (w₁ i) (w₂ i)
    p₁ p₂ : ι → P
    hp : ∀ (i : ι), Membership.mem s i → Eq (p₁ i) (p₂ i)
    b : P
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (HSMul.hSMul (w₁ i) (VSub.vsub (p₁ i) b)) (HSMul.hSMul (w₂ i) (VSub.vsub  …
  -/
  rw [hw i hi, hp i hi]
  /-
    🎉 no goals
  -/


/-- Given a family of points, if we use a member of the family as a base point, the
`weightedVSubOfPoint` does not depend on the value of the weights at this point. -/
theorem weightedVSubOfPoint_eq_of_weights_eq (p : ι → P) (j : ι) (w₁ w₂ : ι → k)
    (hw : ∀ i, i ≠ j → w₁ i = w₂ i) :
    s.weightedVSubOfPoint p (p j) w₁ = s.weightedVSubOfPoint p (p j) w₂ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    p : ι → P
    j : ι
    w₁ w₂ : ι → k
    hw : ∀ (i : ι), Ne i j → Eq (w₁ i) (w₂ i)
    ⊢ Eq ((s.weightedVSubOfPoint p (p j)) w₁) ((s.weightedVSubOfPoint p (p j)) w₂)
  -/
  simp only [Finset.weightedVSubOfPoint_apply]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    p : ι → P
    j : ι
    w₁ w₂ : ι → k
    hw : ∀ (i : ι), Ne i j → Eq (w₁ i) (w₂ i)
    ⊢ Eq (s.sum fun i => HSMul.hSMul (w₁ i) (VSub.vsub (p i) (p j))) (s.sum fun i  …
  -/
  congr
  /-
    case e_f
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    p : ι → P
    j : ι
    w₁ w₂ : ι → k
    hw : ∀ (i : ι), Ne i j → Eq (w₁ i) (w₂ i)
    ⊢ Eq (fun i => HSMul.hSMul (w₁ i) (VSub.vsub (p i) (p j))) fun i => HSMul.hSMu …
  -/
  ext i
  /-
    case e_f.h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    p : ι → P
    j : ι
    w₁ w₂ : ι → k
    hw : ∀ (i : ι), Ne i j → Eq (w₁ i) (w₂ i)
    i : ι
    ⊢ Eq (HSMul.hSMul (w₁ i) (VSub.vsub (p i) (p j))) (HSMul.hSMul (w₂ i) (VSub.vs …
  -/
  rcases eq_or_ne i j with h | h
    /-
      case e_f.h.inl
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝² : Ring k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      S : AddTorsor V P
      ι : Type u_4
      s : Finset ι
      p : ι → P
      j : ι
      w₁ w₂ : ι → k
      hw : ∀ (i : ι), Ne i j → Eq (w₁ i) (w₂ i)
      i : ι
      h : Eq i j
      ⊢ Eq (HSMul.hSMul (w₁ i) (VSub.vsub (p i) (p j))) (HSMul.hSMul (w₂ i) (VSub.vs …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case e_f.h.inr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝² : Ring k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      S : AddTorsor V P
      ι : Type u_4
      s : Finset ι
      p : ι → P
      j : ι
      w₁ w₂ : ι → k
      hw : ∀ (i : ι), Ne i j → Eq (w₁ i) (w₂ i)
      i : ι
      h : Ne i j
      ⊢ Eq (HSMul.hSMul (w₁ i) (VSub.vsub (p i) (p j))) (HSMul.hSMul (w₂ i) (VSub.vs …
    -/
  · simp [hw i h]
    /-
      🎉 no goals
    -/


/-- The weighted sum is independent of the base point when the sum of
the weights is 0. -/
theorem weightedVSubOfPoint_eq_of_sum_eq_zero (w : ι → k) (p : ι → P) (h : ∑ i ∈ s, w i = 0)
    (b₁ b₂ : P) : s.weightedVSubOfPoint p b₁ w = s.weightedVSubOfPoint p b₂ w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    h : Eq (s.sum fun i => w i) 0
    b₁ b₂ : P
    ⊢ Eq ((s.weightedVSubOfPoint p b₁) w) ((s.weightedVSubOfPoint p b₂) w)
  -/
  apply eq_of_sub_eq_zero
  /-
    case h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    h : Eq (s.sum fun i => w i) 0
    b₁ b₂ : P
    ⊢ Eq (HSub.hSub ((s.weightedVSubOfPoint p b₁) w) ((s.weightedVSubOfPoint p b₂) …
  -/
  rw [weightedVSubOfPoint_apply, weightedVSubOfPoint_apply, ← sum_sub_distrib]
  conv_lhs =>
    congr
    · skip
    · ext
      rw [← smul_sub, vsub_sub_vsub_cancel_left]
  /-
    case h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    h : Eq (s.sum fun i => w i) 0
    b₁ b₂ : P
    ⊢ Eq (s.sum fun x => HSMul.hSMul (w x) (VSub.vsub b₂ b₁)) 0
  -/
  rw [← sum_smul, h, zero_smul]
  /-
    🎉 no goals
  -/


/-- The weighted sum, added to the base point, is independent of the
base point when the sum of the weights is 1. -/
theorem weightedVSubOfPoint_vadd_eq_of_sum_eq_one (w : ι → k) (p : ι → P) (h : ∑ i ∈ s, w i = 1)
    (b₁ b₂ : P) : s.weightedVSubOfPoint p b₁ w +ᵥ b₁ = s.weightedVSubOfPoint p b₂ w +ᵥ b₂ := by
  rw [weightedVSubOfPoint_apply, weightedVSubOfPoint_apply, ← @vsub_eq_zero_iff_eq V,
    vadd_vsub_assoc, vsub_vadd_eq_vsub_sub, ← add_sub_assoc, add_comm, add_sub_assoc, ←
    sum_sub_distrib]
  conv_lhs =>
    congr
    · skip
    · congr
      · skip
      · ext
        rw [← smul_sub, vsub_sub_vsub_cancel_left]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    h : Eq (s.sum fun i => w i) 1
    b₁ b₂ : P
    ⊢ Eq (HAdd.hAdd (VSub.vsub b₁ b₂) (s.sum fun x => HSMul.hSMul (w x) (VSub.vsub …
  -/
  rw [← sum_smul, h, one_smul, vsub_add_vsub_cancel, vsub_self]
  /-
    🎉 no goals
  -/


/-- The weighted sum is unaffected by removing the base point, if
present, from the set of points. -/
@[simp (high)]
theorem weightedVSubOfPoint_erase [DecidableEq ι] (w : ι → k) (p : ι → P) (i : ι) :
    (s.erase i).weightedVSubOfPoint p (p i) w = s.weightedVSubOfPoint p (p i) w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    w : ι → k
    p : ι → P
    i : ι
    ⊢ Eq (((s.erase i).weightedVSubOfPoint p (p i)) w) ((s.weightedVSubOfPoint p ( …
  -/
  rw [weightedVSubOfPoint_apply, weightedVSubOfPoint_apply]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    w : ι → k
    p : ι → P
    i : ι
    ⊢ Eq ((s.erase i).sum fun i_1 => HSMul.hSMul (w i_1) (VSub.vsub (p i_1) (p i)) …
  -/
  apply sum_erase
  /-
    case h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    w : ι → k
    p : ι → P
    i : ι
    ⊢ Eq (HSMul.hSMul (w i) (VSub.vsub (p i) (p i))) 0
  -/
  rw [vsub_self, smul_zero]
  /-
    🎉 no goals
  -/


/-- The weighted sum is unaffected by adding the base point, whether
or not present, to the set of points. -/
@[simp (high)]
theorem weightedVSubOfPoint_insert [DecidableEq ι] (w : ι → k) (p : ι → P) (i : ι) :
    (insert i s).weightedVSubOfPoint p (p i) w = s.weightedVSubOfPoint p (p i) w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    w : ι → k
    p : ι → P
    i : ι
    ⊢ Eq (((Insert.insert i s).weightedVSubOfPoint p (p i)) w) ((s.weightedVSubOfP …
  -/
  rw [weightedVSubOfPoint_apply, weightedVSubOfPoint_apply]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    w : ι → k
    p : ι → P
    i : ι
    ⊢ Eq ((Insert.insert i s).sum fun i_1 => HSMul.hSMul (w i_1) (VSub.vsub (p i_1 …
  -/
  apply sum_insert_zero
  /-
    case h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    w : ι → k
    p : ι → P
    i : ι
    ⊢ Eq (HSMul.hSMul (w i) (VSub.vsub (p i) (p i))) 0
  -/
  rw [vsub_self, smul_zero]
  /-
    🎉 no goals
  -/


/-- The weighted sum is unaffected by changing the weights to the
corresponding indicator function and adding points to the set. -/
theorem weightedVSubOfPoint_indicator_subset (w : ι → k) (p : ι → P) (b : P) {s₁ s₂ : Finset ι}
    (h : s₁ ⊆ s₂) :
    s₁.weightedVSubOfPoint p b w = s₂.weightedVSubOfPoint p b (Set.indicator (↑s₁) w) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    w : ι → k
    p : ι → P
    b : P
    s₁ s₂ : Finset ι
    h : HasSubset.Subset s₁ s₂
    ⊢ Eq ((s₁.weightedVSubOfPoint p b) w) ((s₂.weightedVSubOfPoint p b) ((↑s₁).ind …
  -/
  rw [weightedVSubOfPoint_apply, weightedVSubOfPoint_apply]
  exact Eq.symm <|
    sum_indicator_subset_of_eq_zero w (fun i wi => wi • (p i -ᵥ b : V)) h fun i => zero_smul k _


/-- A weighted sum, over the image of an embedding, equals a weighted
sum with the same points and weights over the original
`Finset`. -/
theorem weightedVSubOfPoint_map (e : ι₂ ↪ ι) (w : ι → k) (p : ι → P) (b : P) :
    (s₂.map e).weightedVSubOfPoint p b w = s₂.weightedVSubOfPoint (p ∘ e) b (w ∘ e) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    ι₂ : Type u_5
    s₂ : Finset ι₂
    e : Function.Embedding ι₂ ι
    w : ι → k
    p : ι → P
    b : P
    ⊢ Eq (((Finset.map e s₂).weightedVSubOfPoint p b) w) ((s₂.weightedVSubOfPoint  …
  -/
  simp_rw [weightedVSubOfPoint_apply]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    ι₂ : Type u_5
    s₂ : Finset ι₂
    e : Function.Embedding ι₂ ι
    w : ι → k
    p : ι → P
    b : P
    ⊢ Eq ((Finset.map e s₂).sum fun i => HSMul.hSMul (w i) (VSub.vsub (p i) b)) (s …
  -/
  exact Finset.sum_map _ _ _
  /-
    🎉 no goals
  -/


/-- A weighted sum of pairwise subtractions, expressed as a subtraction of two
`weightedVSubOfPoint` expressions. -/
theorem sum_smul_vsub_eq_weightedVSubOfPoint_sub (w : ι → k) (p₁ p₂ : ι → P) (b : P) :
    (∑ i ∈ s, w i • (p₁ i -ᵥ p₂ i)) =
      s.weightedVSubOfPoint p₁ b w - s.weightedVSubOfPoint p₂ b w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p₁ p₂ : ι → P
    b : P
    ⊢ Eq (s.sum fun i => HSMul.hSMul (w i) (VSub.vsub (p₁ i) (p₂ i))) (HSub.hSub ( …
  -/
  simp_rw [weightedVSubOfPoint_apply, ← sum_sub_distrib, ← smul_sub, vsub_sub_vsub_cancel_right]
  /-
    🎉 no goals
  -/


/-- A weighted sum of pairwise subtractions, where the point on the right is constant,
expressed as a subtraction involving a `weightedVSubOfPoint` expression. -/
theorem sum_smul_vsub_const_eq_weightedVSubOfPoint_sub (w : ι → k) (p₁ : ι → P) (p₂ b : P) :
    (∑ i ∈ s, w i • (p₁ i -ᵥ p₂)) = s.weightedVSubOfPoint p₁ b w - (∑ i ∈ s, w i) • (p₂ -ᵥ b) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p₁ : ι → P
    p₂ b : P
    ⊢ Eq (s.sum fun i => HSMul.hSMul (w i) (VSub.vsub (p₁ i) p₂)) (HSub.hSub ((s.w …
  -/
  rw [sum_smul_vsub_eq_weightedVSubOfPoint_sub, weightedVSubOfPoint_apply_const]
  /-
    🎉 no goals
  -/


/-- A weighted sum of pairwise subtractions, where the point on the left is constant,
expressed as a subtraction involving a `weightedVSubOfPoint` expression. -/
theorem sum_smul_const_vsub_eq_sub_weightedVSubOfPoint (w : ι → k) (p₂ : ι → P) (p₁ b : P) :
    (∑ i ∈ s, w i • (p₁ -ᵥ p₂ i)) = (∑ i ∈ s, w i) • (p₁ -ᵥ b) - s.weightedVSubOfPoint p₂ b w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p₂ : ι → P
    p₁ b : P
    ⊢ Eq (s.sum fun i => HSMul.hSMul (w i) (VSub.vsub p₁ (p₂ i))) (HSub.hSub (HSMu …
  -/
  rw [sum_smul_vsub_eq_weightedVSubOfPoint_sub, weightedVSubOfPoint_apply_const]
  /-
    🎉 no goals
  -/


/-- A weighted sum may be split into such sums over two subsets. -/
theorem weightedVSubOfPoint_sdiff [DecidableEq ι] {s₂ : Finset ι} (h : s₂ ⊆ s) (w : ι → k)
    (p : ι → P) (b : P) :
    (s \ s₂).weightedVSubOfPoint p b w + s₂.weightedVSubOfPoint p b w =
      s.weightedVSubOfPoint p b w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    s₂ : Finset ι
    h : HasSubset.Subset s₂ s
    w : ι → k
    p : ι → P
    b : P
    ⊢ Eq (HAdd.hAdd (((SDiff.sdiff s s₂).weightedVSubOfPoint p b) w) ((s₂.weighted …
  -/
  simp_rw [weightedVSubOfPoint_apply, sum_sdiff h]
  /-
    🎉 no goals
  -/


/-- A weighted sum may be split into a subtraction of such sums over two subsets. -/
theorem weightedVSubOfPoint_sdiff_sub [DecidableEq ι] {s₂ : Finset ι} (h : s₂ ⊆ s) (w : ι → k)
    (p : ι → P) (b : P) :
    (s \ s₂).weightedVSubOfPoint p b w - s₂.weightedVSubOfPoint p b (-w) =
      s.weightedVSubOfPoint p b w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    s₂ : Finset ι
    h : HasSubset.Subset s₂ s
    w : ι → k
    p : ι → P
    b : P
    ⊢ Eq (HSub.hSub (((SDiff.sdiff s s₂).weightedVSubOfPoint p b) w) ((s₂.weighted …
  -/
  rw [map_neg, sub_neg_eq_add, s.weightedVSubOfPoint_sdiff h]
  /-
    🎉 no goals
  -/


/-- A weighted sum over `s.subtype pred` equals one over `{x ∈ s | pred x}`. -/
theorem weightedVSubOfPoint_subtype_eq_filter (w : ι → k) (p : ι → P) (b : P) (pred : ι → Prop)
    [DecidablePred pred] :
    ((s.subtype pred).weightedVSubOfPoint (fun i => p i) b fun i => w i) =
      {x ∈ s | pred x}.weightedVSubOfPoint p b w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    b : P
    pred : ι → Prop
    inst✝ : DecidablePred pred
    ⊢ Eq (((Finset.subtype pred s).weightedVSubOfPoint (fun i => p ↑i) b) fun i => …
  -/
  rw [weightedVSubOfPoint_apply, weightedVSubOfPoint_apply, ← sum_subtype_eq_sum_filter]
  /-
    🎉 no goals
  -/


/-- A weighted sum over `{x ∈ s | pred x}` equals one over `s` if all the weights at indices in `s`
not satisfying `pred` are zero. -/
theorem weightedVSubOfPoint_filter_of_ne (w : ι → k) (p : ι → P) (b : P) {pred : ι → Prop}
    [DecidablePred pred] (h : ∀ i ∈ s, w i ≠ 0 → pred i) :
    {x ∈ s | pred x}.weightedVSubOfPoint p b w = s.weightedVSubOfPoint p b w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    b : P
    pred : ι → Prop
    inst✝ : DecidablePred pred
    h : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → pred i
    ⊢ Eq (((Finset.filter (fun x => pred x) s).weightedVSubOfPoint p b) w) ((s.wei …
  -/
  rw [weightedVSubOfPoint_apply, weightedVSubOfPoint_apply, sum_filter_of_ne]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    b : P
    pred : ι → Prop
    inst✝ : DecidablePred pred
    h : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → pred i
    ⊢ ∀ (x : ι), Membership.mem s x → Ne (HSMul.hSMul (w x) (VSub.vsub (p x) b)) 0 …
  -/
  intro i hi hne
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    b : P
    pred : ι → Prop
    inst✝ : DecidablePred pred
    h : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → pred i
    i : ι
    hi : Membership.mem s i
    hne : Ne (HSMul.hSMul (w i) (VSub.vsub (p i) b)) 0
    ⊢ pred i
  -/
  refine h i hi ?_
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    b : P
    pred : ι → Prop
    inst✝ : DecidablePred pred
    h : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → pred i
    i : ι
    hi : Membership.mem s i
    hne : Ne (HSMul.hSMul (w i) (VSub.vsub (p i) b)) 0
    ⊢ Ne (w i) 0
  -/
  intro hw
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    b : P
    pred : ι → Prop
    inst✝ : DecidablePred pred
    h : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → pred i
    i : ι
    hi : Membership.mem s i
    hne : Ne (HSMul.hSMul (w i) (VSub.vsub (p i) b)) 0
    hw : Eq (w i) 0
    ⊢ False
  -/
  simp [hw] at hne
  /-
    🎉 no goals
  -/


/-- A constant multiplier of the weights in `weightedVSubOfPoint` may be moved outside the
sum. -/
theorem weightedVSubOfPoint_const_smul (w : ι → k) (p : ι → P) (b : P) (c : k) :
    s.weightedVSubOfPoint p b (c • w) = c • s.weightedVSubOfPoint p b w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    b : P
    c : k
    ⊢ Eq ((s.weightedVSubOfPoint p b) (HSMul.hSMul c w)) (HSMul.hSMul c ((s.weight …
  -/
  simp_rw [weightedVSubOfPoint_apply, smul_sum, Pi.smul_apply, smul_smul, smul_eq_mul]
  /-
    🎉 no goals
  -/


/-- A weighted sum of the results of subtracting a default base point
from the given points, as a linear map on the weights.  This is
intended to be used when the sum of the weights is 0; that condition
is specified as a hypothesis on those lemmas that require it. -/
def weightedVSub (p : ι → P) : (ι → k) →ₗ[k] V :=
  s.weightedVSubOfPoint p (Classical.choice S.nonempty)


/-- Applying `weightedVSub` with given weights.  This is for the case
where a result involving a default base point is OK (for example, when
that base point will cancel out later); a more typical use case for
`weightedVSub` would involve selecting a preferred base point with
`weightedVSub_eq_weightedVSubOfPoint_of_sum_eq_zero` and then
using `weightedVSubOfPoint_apply`. -/
theorem weightedVSub_apply (w : ι → k) (p : ι → P) :
    s.weightedVSub p w = ∑ i ∈ s, w i • (p i -ᵥ Classical.choice S.nonempty) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    ⊢ Eq ((s.weightedVSub p) w) (s.sum fun i => HSMul.hSMul (w i) (VSub.vsub (p i) …
  -/
  simp [weightedVSub, LinearMap.sum_apply]
  /-
    🎉 no goals
  -/


/-- `weightedVSub` gives the sum of the results of subtracting any
base point, when the sum of the weights is 0. -/
theorem weightedVSub_eq_weightedVSubOfPoint_of_sum_eq_zero (w : ι → k) (p : ι → P)
    (h : ∑ i ∈ s, w i = 0) (b : P) : s.weightedVSub p w = s.weightedVSubOfPoint p b w :=
  s.weightedVSubOfPoint_eq_of_sum_eq_zero w p h _ _


/-- The value of `weightedVSub`, where the given points are equal and the sum of the weights
is 0. -/
@[simp]
theorem weightedVSub_apply_const (w : ι → k) (p : P) (h : ∑ i ∈ s, w i = 0) :
    s.weightedVSub (fun _ => p) w = 0 := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : P
    h : Eq (s.sum fun i => w i) 0
    ⊢ Eq ((s.weightedVSub fun x => p) w) 0
  -/
  rw [weightedVSub, weightedVSubOfPoint_apply_const, h, zero_smul]
  /-
    🎉 no goals
  -/


/-- The `weightedVSub` for an empty set is 0. -/
@[simp]
theorem weightedVSub_empty (w : ι → k) (p : ι → P) : (∅ : Finset ι).weightedVSub p w = (0 : V) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    w : ι → k
    p : ι → P
    ⊢ Eq ((EmptyCollection.emptyCollection.weightedVSub p) w) 0
  -/
  simp [weightedVSub_apply]
  /-
    🎉 no goals
  -/


lemma weightedVSub_vadd {s : Finset ι} {w : ι → k} (h : ∑ i ∈ s, w i = 0) (p : ι → P) (v : V) :
    s.weightedVSub (v +ᵥ p) w = s.weightedVSub p w := by
  rw [weightedVSub, weightedVSubOfPoint_vadd,
    weightedVSub_eq_weightedVSubOfPoint_of_sum_eq_zero _ _ _ h]


lemma weightedVSub_smul {G : Type*} [Group G] [DistribMulAction G V] [SMulCommClass G k V]
    {s : Finset ι} {w : ι → k} (h : ∑ i ∈ s, w i = 0) (p : ι → V) (a : G) :
    s.weightedVSub (a • p) w = a • s.weightedVSub p w := by
  rw [weightedVSub, weightedVSubOfPoint_smul,
    weightedVSub_eq_weightedVSubOfPoint_of_sum_eq_zero _ _ _ h]


/-- `weightedVSub` gives equal results for two families of weights and two families of points
that are equal on `s`. -/
theorem weightedVSub_congr {w₁ w₂ : ι → k} (hw : ∀ i ∈ s, w₁ i = w₂ i) {p₁ p₂ : ι → P}
    (hp : ∀ i ∈ s, p₁ i = p₂ i) : s.weightedVSub p₁ w₁ = s.weightedVSub p₂ w₂ :=
  s.weightedVSubOfPoint_congr hw hp _


/-- The weighted sum is unaffected by changing the weights to the
corresponding indicator function and adding points to the set. -/
theorem weightedVSub_indicator_subset (w : ι → k) (p : ι → P) {s₁ s₂ : Finset ι} (h : s₁ ⊆ s₂) :
    s₁.weightedVSub p w = s₂.weightedVSub p (Set.indicator (↑s₁) w) :=
  weightedVSubOfPoint_indicator_subset _ _ _ h


/-- A weighted subtraction, over the image of an embedding, equals a
weighted subtraction with the same points and weights over the
original `Finset`. -/
theorem weightedVSub_map (e : ι₂ ↪ ι) (w : ι → k) (p : ι → P) :
    (s₂.map e).weightedVSub p w = s₂.weightedVSub (p ∘ e) (w ∘ e) :=
  s₂.weightedVSubOfPoint_map _ _ _ _


/-- A weighted sum of pairwise subtractions, expressed as a subtraction of two `weightedVSub`
expressions. -/
theorem sum_smul_vsub_eq_weightedVSub_sub (w : ι → k) (p₁ p₂ : ι → P) :
    (∑ i ∈ s, w i • (p₁ i -ᵥ p₂ i)) = s.weightedVSub p₁ w - s.weightedVSub p₂ w :=
  s.sum_smul_vsub_eq_weightedVSubOfPoint_sub _ _ _ _


/-- A weighted sum of pairwise subtractions, where the point on the right is constant and the
sum of the weights is 0. -/
theorem sum_smul_vsub_const_eq_weightedVSub (w : ι → k) (p₁ : ι → P) (p₂ : P)
    (h : ∑ i ∈ s, w i = 0) : (∑ i ∈ s, w i • (p₁ i -ᵥ p₂)) = s.weightedVSub p₁ w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p₁ : ι → P
    p₂ : P
    h : Eq (s.sum fun i => w i) 0
    ⊢ Eq (s.sum fun i => HSMul.hSMul (w i) (VSub.vsub (p₁ i) p₂)) ((s.weightedVSub …
  -/
  rw [sum_smul_vsub_eq_weightedVSub_sub, s.weightedVSub_apply_const _ _ h, sub_zero]
  /-
    🎉 no goals
  -/


/-- A weighted sum of pairwise subtractions, where the point on the left is constant and the
sum of the weights is 0. -/
theorem sum_smul_const_vsub_eq_neg_weightedVSub (w : ι → k) (p₂ : ι → P) (p₁ : P)
    (h : ∑ i ∈ s, w i = 0) : (∑ i ∈ s, w i • (p₁ -ᵥ p₂ i)) = -s.weightedVSub p₂ w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p₂ : ι → P
    p₁ : P
    h : Eq (s.sum fun i => w i) 0
    ⊢ Eq (s.sum fun i => HSMul.hSMul (w i) (VSub.vsub p₁ (p₂ i))) (Neg.neg ((s.wei …
  -/
  rw [sum_smul_vsub_eq_weightedVSub_sub, s.weightedVSub_apply_const _ _ h, zero_sub]
  /-
    🎉 no goals
  -/


/-- A weighted sum may be split into such sums over two subsets. -/
theorem weightedVSub_sdiff [DecidableEq ι] {s₂ : Finset ι} (h : s₂ ⊆ s) (w : ι → k) (p : ι → P) :
    (s \ s₂).weightedVSub p w + s₂.weightedVSub p w = s.weightedVSub p w :=
  s.weightedVSubOfPoint_sdiff h _ _ _


/-- A weighted sum may be split into a subtraction of such sums over two subsets. -/
theorem weightedVSub_sdiff_sub [DecidableEq ι] {s₂ : Finset ι} (h : s₂ ⊆ s) (w : ι → k)
    (p : ι → P) : (s \ s₂).weightedVSub p w - s₂.weightedVSub p (-w) = s.weightedVSub p w :=
  s.weightedVSubOfPoint_sdiff_sub h _ _ _


/-- A weighted sum over `s.subtype pred` equals one over `{x ∈ s | pred x}`. -/
theorem weightedVSub_subtype_eq_filter (w : ι → k) (p : ι → P) (pred : ι → Prop)
    [DecidablePred pred] :
    ((s.subtype pred).weightedVSub (fun i => p i) fun i => w i) =
      {x ∈ s | pred x}.weightedVSub p w :=
  s.weightedVSubOfPoint_subtype_eq_filter _ _ _ _


/-- A weighted sum over `{x ∈ s | pred x}` equals one over `s` if all the weights at indices in `s`
not satisfying `pred` are zero. -/
theorem weightedVSub_filter_of_ne (w : ι → k) (p : ι → P) {pred : ι → Prop} [DecidablePred pred]
    (h : ∀ i ∈ s, w i ≠ 0 → pred i) : {x ∈ s | pred x}.weightedVSub p w = s.weightedVSub p w :=
  s.weightedVSubOfPoint_filter_of_ne _ _ _ h


/-- A constant multiplier of the weights in `weightedVSub_of` may be moved outside the sum. -/
theorem weightedVSub_const_smul (w : ι → k) (p : ι → P) (c : k) :
    s.weightedVSub p (c • w) = c • s.weightedVSub p w :=
  s.weightedVSubOfPoint_const_smul _ _ _ _


instance : AffineSpace (ι → k) (ι → k) := Pi.instAddTorsor


/-- A weighted sum of the results of subtracting a default base point
from the given points, added to that base point, as an affine map on
the weights.  This is intended to be used when the sum of the weights
is 1, in which case it is an affine combination (barycenter) of the
points with the given weights; that condition is specified as a
hypothesis on those lemmas that require it. -/
def affineCombination (p : ι → P) : (ι → k) →ᵃ[k] P where
  toFun w := s.weightedVSubOfPoint p (Classical.choice S.nonempty) w +ᵥ Classical.choice S.nonempty
  linear := s.weightedVSub p
                        /-
                          k : Type u_1
                          V : Type u_2
                          P : Type u_3
                          inst✝² : Ring k
                          inst✝¹ : AddCommGroup V
                          inst✝ : Module k V
                          S : AddTorsor V P
                          ι : Type u_4
                          s : Finset ι
                          ι₂ : Type u_5
                          s₂ : Finset ι₂
                          p : ι → P
                          w₁ w₂ : ι → k
                          ⊢ Eq ((fun w => HVAdd.hVAdd ((s.weightedVSubOfPoint p (Classical.choice ⋯)) w) …
                        -/
  map_vadd' w₁ w₂ := by simp_rw [vadd_vadd, weightedVSub, vadd_eq_add, LinearMap.map_add]
                        /-
                          🎉 no goals
                        -/


/-- The linear map corresponding to `affineCombination` is
`weightedVSub`. -/
@[simp]
theorem affineCombination_linear (p : ι → P) :
    (s.affineCombination k p).linear = s.weightedVSub p :=
  rfl


/-- Applying `affineCombination` with given weights.  This is for the
case where a result involving a default base point is OK (for example,
when that base point will cancel out later); a more typical use case
for `affineCombination` would involve selecting a preferred base
point with
`affineCombination_eq_weightedVSubOfPoint_vadd_of_sum_eq_one` and
then using `weightedVSubOfPoint_apply`. -/
theorem affineCombination_apply (w : ι → k) (p : ι → P) :
    (s.affineCombination k p) w =
      s.weightedVSubOfPoint p (Classical.choice S.nonempty) w +ᵥ Classical.choice S.nonempty :=
  rfl


/-- The value of `affineCombination`, where the given points are equal. -/
@[simp]
theorem affineCombination_apply_const (w : ι → k) (p : P) (h : ∑ i ∈ s, w i = 1) :
    s.affineCombination k (fun _ => p) w = p := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : P
    h : Eq (s.sum fun i => w i) 1
    ⊢ Eq ((Finset.affineCombination k s fun x => p) w) p
  -/
  rw [affineCombination_apply, s.weightedVSubOfPoint_apply_const, h, one_smul, vsub_vadd]
  /-
    🎉 no goals
  -/


/-- `affineCombination` gives equal results for two families of weights and two families of
points that are equal on `s`. -/
theorem affineCombination_congr {w₁ w₂ : ι → k} (hw : ∀ i ∈ s, w₁ i = w₂ i) {p₁ p₂ : ι → P}
    (hp : ∀ i ∈ s, p₁ i = p₂ i) : s.affineCombination k p₁ w₁ = s.affineCombination k p₂ w₂ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w₁ w₂ : ι → k
    hw : ∀ (i : ι), Membership.mem s i → Eq (w₁ i) (w₂ i)
    p₁ p₂ : ι → P
    hp : ∀ (i : ι), Membership.mem s i → Eq (p₁ i) (p₂ i)
    ⊢ Eq ((Finset.affineCombination k s p₁) w₁) ((Finset.affineCombination k s p₂) …
  -/
  simp_rw [affineCombination_apply, s.weightedVSubOfPoint_congr hw hp]
  /-
    🎉 no goals
  -/


/-- `affineCombination` gives the sum with any base point, when the
sum of the weights is 1. -/
theorem affineCombination_eq_weightedVSubOfPoint_vadd_of_sum_eq_one (w : ι → k) (p : ι → P)
    (h : ∑ i ∈ s, w i = 1) (b : P) :
    s.affineCombination k p w = s.weightedVSubOfPoint p b w +ᵥ b :=
  s.weightedVSubOfPoint_vadd_eq_of_sum_eq_one w p h _ _


/-- Adding a `weightedVSub` to an `affineCombination`. -/
theorem weightedVSub_vadd_affineCombination (w₁ w₂ : ι → k) (p : ι → P) :
    s.weightedVSub p w₁ +ᵥ s.affineCombination k p w₂ = s.affineCombination k p (w₁ + w₂) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w₁ w₂ : ι → k
    p : ι → P
    ⊢ Eq (HVAdd.hVAdd ((s.weightedVSub p) w₁) ((Finset.affineCombination k s p) w₂ …
  -/
  rw [← vadd_eq_add, AffineMap.map_vadd, affineCombination_linear]
  /-
    🎉 no goals
  -/


/-- Subtracting two `affineCombination`s. -/
theorem affineCombination_vsub (w₁ w₂ : ι → k) (p : ι → P) :
    s.affineCombination k p w₁ -ᵥ s.affineCombination k p w₂ = s.weightedVSub p (w₁ - w₂) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w₁ w₂ : ι → k
    p : ι → P
    ⊢ Eq (VSub.vsub ((Finset.affineCombination k s p) w₁) ((Finset.affineCombinati …
  -/
  rw [← AffineMap.linearMap_vsub, affineCombination_linear, vsub_eq_sub]
  /-
    🎉 no goals
  -/


theorem attach_affineCombination_of_injective [DecidableEq P] (s : Finset P) (w : P → k) (f : s → P)
    (hf : Function.Injective f) :
    s.attach.affineCombination k f (w ∘ f) = (image f univ).affineCombination k id w := by
  simp only [affineCombination, weightedVSubOfPoint_apply, id, vadd_right_cancel_iff,
    Function.comp_apply, AffineMap.coe_mk]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    inst✝ : DecidableEq P
    s : Finset P
    w : P → k
    f : (Subtype fun x => Membership.mem s x) → P
    hf : Function.Injective f
    ⊢ Eq (s.attach.sum fun x => HSMul.hSMul (w (f x)) (VSub.vsub (f x) (Classical. …
  -/
  let g₁ : s → V := fun i => w (f i) • (f i -ᵥ Classical.choice S.nonempty)
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    inst✝ : DecidableEq P
    s : Finset P
    w : P → k
    f : (Subtype fun x => Membership.mem s x) → P
    hf : Function.Injective f
    g₁ : (Subtype fun x => Membership.mem s x) → V := fun i => HSMul.hSMul (w (f i …
    ⊢ Eq (s.attach.sum fun x => HSMul.hSMul (w (f x)) (VSub.vsub (f x) (Classical. …
  -/
  let g₂ : P → V := fun i => w i • (i -ᵥ Classical.choice S.nonempty)
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    inst✝ : DecidableEq P
    s : Finset P
    w : P → k
    f : (Subtype fun x => Membership.mem s x) → P
    hf : Function.Injective f
    g₁ : (Subtype fun x => Membership.mem s x) → V := fun i => HSMul.hSMul (w (f i …
    g₂ : P → V := fun i => HSMul.hSMul (w i) (VSub.vsub i (Classical.choice ⋯))
    ⊢ Eq (s.attach.sum fun x => HSMul.hSMul (w (f x)) (VSub.vsub (f x) (Classical. …
  -/
  change univ.sum g₁ = (image f univ).sum g₂
  have hgf : g₁ = g₂ ∘ f := by
    ext
    simp [g₁, g₂]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    inst✝ : DecidableEq P
    s : Finset P
    w : P → k
    f : (Subtype fun x => Membership.mem s x) → P
    hf : Function.Injective f
    g₁ : (Subtype fun x => Membership.mem s x) → V := fun i => HSMul.hSMul (w (f i …
    g₂ : P → V := fun i => HSMul.hSMul (w i) (VSub.vsub i (Classical.choice ⋯))
    hgf : Eq g₁ (Function.comp g₂ f)
    ⊢ Eq (Finset.univ.sum g₁) ((Finset.image f Finset.univ).sum g₂)
  -/
  rw [hgf, sum_image]
    /-
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      S : AddTorsor V P
      inst✝ : DecidableEq P
      s : Finset P
      w : P → k
      f : (Subtype fun x => Membership.mem s x) → P
      hf : Function.Injective f
      g₁ : (Subtype fun x => Membership.mem s x) → V := fun i => HSMul.hSMul (w (f i …
      g₂ : P → V := fun i => HSMul.hSMul (w i) (VSub.vsub i (Classical.choice ⋯))
      hgf : Eq g₁ (Function.comp g₂ f)
      ⊢ Eq (Finset.univ.sum (Function.comp g₂ f)) (Finset.univ.sum fun x => HSMul.hS …
    -/
  · simp only [g₁, g₂,Function.comp_apply]
    /-
      🎉 no goals
    -/
    /-
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      S : AddTorsor V P
      inst✝ : DecidableEq P
      s : Finset P
      w : P → k
      f : (Subtype fun x => Membership.mem s x) → P
      hf : Function.Injective f
      g₁ : (Subtype fun x => Membership.mem s x) → V := fun i => HSMul.hSMul (w (f i …
      g₂ : P → V := fun i => HSMul.hSMul (w i) (VSub.vsub i (Classical.choice ⋯))
      hgf : Eq g₁ (Function.comp g₂ f)
      ⊢ ∀ (x : Subtype fun x => Membership.mem s x), Membership.mem Finset.univ x →  …
    -/
  · exact fun _ _ _ _ hxy => hf hxy
    /-
      🎉 no goals
    -/


theorem attach_affineCombination_coe (s : Finset P) (w : P → k) :
    s.attach.affineCombination k ((↑) : s → P) (w ∘ (↑)) = s.affineCombination k id w := by
  classical rw [attach_affineCombination_of_injective s w ((↑) : s → P) Subtype.coe_injective,
      univ_eq_attach, attach_image_val]


/-- Viewing a module as an affine space modelled on itself, a `weightedVSub` is just a linear
combination. -/
@[simp]
theorem weightedVSub_eq_linear_combination {ι} (s : Finset ι) {w : ι → k} {p : ι → V}
    (hw : s.sum w = 0) : s.weightedVSub p w = ∑ i ∈ s, w i • p i := by
  /-
    k : Type u_1
    V : Type u_2
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    ι : Type u_6
    s : Finset ι
    w : ι → k
    p : ι → V
    hw : Eq (s.sum w) 0
    ⊢ Eq ((s.weightedVSub p) w) (s.sum fun i => HSMul.hSMul (w i) (p i))
  -/
  simp [s.weightedVSub_apply, vsub_eq_sub, smul_sub, ← Finset.sum_smul, hw]
  /-
    🎉 no goals
  -/


/-- Viewing a module as an affine space modelled on itself, affine combinations are just linear
combinations. -/
@[simp]
theorem affineCombination_eq_linear_combination (s : Finset ι) (p : ι → V) (w : ι → k)
    (hw : ∑ i ∈ s, w i = 1) : s.affineCombination k p w = ∑ i ∈ s, w i • p i := by
  /-
    k : Type u_1
    V : Type u_2
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    ι : Type u_4
    s : Finset ι
    p : ι → V
    w : ι → k
    hw : Eq (s.sum fun i => w i) 1
    ⊢ Eq ((Finset.affineCombination k s p) w) (s.sum fun i => HSMul.hSMul (w i) (p …
  -/
  simp [s.affineCombination_eq_weightedVSubOfPoint_vadd_of_sum_eq_one w p hw 0]
  /-
    🎉 no goals
  -/


/-- An `affineCombination` equals a point if that point is in the set
and has weight 1 and the other points in the set have weight 0. -/
@[simp]
theorem affineCombination_of_eq_one_of_eq_zero (w : ι → k) (p : ι → P) {i : ι} (his : i ∈ s)
    (hwi : w i = 1) (hw0 : ∀ i2 ∈ s, i2 ≠ i → w i2 = 0) : s.affineCombination k p w = p i := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    i : ι
    his : Membership.mem s i
    hwi : Eq (w i) 1
    hw0 : ∀ (i2 : ι), Membership.mem s i2 → Ne i2 i → Eq (w i2) 0
    ⊢ Eq ((Finset.affineCombination k s p) w) (p i)
  -/
  have h1 : ∑ i ∈ s, w i = 1 := hwi ▸ sum_eq_single i hw0 fun h => False.elim (h his)
  rw [s.affineCombination_eq_weightedVSubOfPoint_vadd_of_sum_eq_one w p h1 (p i),
    weightedVSubOfPoint_apply]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    i : ι
    his : Membership.mem s i
    hwi : Eq (w i) 1
    hw0 : ∀ (i2 : ι), Membership.mem s i2 → Ne i2 i → Eq (w i2) 0
    h1 : Eq (s.sum fun i => w i) 1
    ⊢ Eq (HVAdd.hVAdd (s.sum fun i_1 => HSMul.hSMul (w i_1) (VSub.vsub (p i_1) (p  …
  -/
  convert zero_vadd V (p i)
  /-
    case h.e'_2.h.e'_5
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    i : ι
    his : Membership.mem s i
    hwi : Eq (w i) 1
    hw0 : ∀ (i2 : ι), Membership.mem s i2 → Ne i2 i → Eq (w i2) 0
    h1 : Eq (s.sum fun i => w i) 1
    ⊢ Eq (s.sum fun i_1 => HSMul.hSMul (w i_1) (VSub.vsub (p i_1) (p i))) 0
  -/
  refine sum_eq_zero ?_
  /-
    case h.e'_2.h.e'_5
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    i : ι
    his : Membership.mem s i
    hwi : Eq (w i) 1
    hw0 : ∀ (i2 : ι), Membership.mem s i2 → Ne i2 i → Eq (w i2) 0
    h1 : Eq (s.sum fun i => w i) 1
    ⊢ ∀ (x : ι), Membership.mem s x → Eq (HSMul.hSMul (w x) (VSub.vsub (p x) (p i) …
  -/
  intro i2 hi2
  /-
    case h.e'_2.h.e'_5
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    i : ι
    his : Membership.mem s i
    hwi : Eq (w i) 1
    hw0 : ∀ (i2 : ι), Membership.mem s i2 → Ne i2 i → Eq (w i2) 0
    h1 : Eq (s.sum fun i => w i) 1
    i2 : ι
    hi2 : Membership.mem s i2
    ⊢ Eq (HSMul.hSMul (w i2) (VSub.vsub (p i2) (p i))) 0
  -/
  by_cases h : i2 = i
    /-
      case pos
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝² : Ring k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      S : AddTorsor V P
      ι : Type u_4
      s : Finset ι
      w : ι → k
      p : ι → P
      i : ι
      his : Membership.mem s i
      hwi : Eq (w i) 1
      hw0 : ∀ (i2 : ι), Membership.mem s i2 → Ne i2 i → Eq (w i2) 0
      h1 : Eq (s.sum fun i => w i) 1
      i2 : ι
      hi2 : Membership.mem s i2
      h : Eq i2 i
      ⊢ Eq (HSMul.hSMul (w i2) (VSub.vsub (p i2) (p i))) 0
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝² : Ring k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      S : AddTorsor V P
      ι : Type u_4
      s : Finset ι
      w : ι → k
      p : ι → P
      i : ι
      his : Membership.mem s i
      hwi : Eq (w i) 1
      hw0 : ∀ (i2 : ι), Membership.mem s i2 → Ne i2 i → Eq (w i2) 0
      h1 : Eq (s.sum fun i => w i) 1
      i2 : ι
      hi2 : Membership.mem s i2
      h : Not (Eq i2 i)
      ⊢ Eq (HSMul.hSMul (w i2) (VSub.vsub (p i2) (p i))) 0
    -/
  · simp [hw0 i2 hi2 h]
    /-
      🎉 no goals
    -/


/-- An affine combination is unaffected by changing the weights to the
corresponding indicator function and adding points to the set. -/
theorem affineCombination_indicator_subset (w : ι → k) (p : ι → P) {s₁ s₂ : Finset ι}
    (h : s₁ ⊆ s₂) :
    s₁.affineCombination k p w = s₂.affineCombination k p (Set.indicator (↑s₁) w) := by
  rw [affineCombination_apply, affineCombination_apply,
    weightedVSubOfPoint_indicator_subset _ _ _ h]


/-- An affine combination, over the image of an embedding, equals an
affine combination with the same points and weights over the original
`Finset`. -/
theorem affineCombination_map (e : ι₂ ↪ ι) (w : ι → k) (p : ι → P) :
    (s₂.map e).affineCombination k p w = s₂.affineCombination k (p ∘ e) (w ∘ e) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    ι₂ : Type u_5
    s₂ : Finset ι₂
    e : Function.Embedding ι₂ ι
    w : ι → k
    p : ι → P
    ⊢ Eq ((Finset.affineCombination k (Finset.map e s₂) p) w) ((Finset.affineCombi …
  -/
  simp_rw [affineCombination_apply, weightedVSubOfPoint_map]
  /-
    🎉 no goals
  -/


/-- A weighted sum of pairwise subtractions, expressed as a subtraction of two `affineCombination`
expressions. -/
theorem sum_smul_vsub_eq_affineCombination_vsub (w : ι → k) (p₁ p₂ : ι → P) :
    (∑ i ∈ s, w i • (p₁ i -ᵥ p₂ i)) =
      s.affineCombination k p₁ w -ᵥ s.affineCombination k p₂ w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p₁ p₂ : ι → P
    ⊢ Eq (s.sum fun i => HSMul.hSMul (w i) (VSub.vsub (p₁ i) (p₂ i))) (VSub.vsub ( …
  -/
  simp_rw [affineCombination_apply, vadd_vsub_vadd_cancel_right]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p₁ p₂ : ι → P
    ⊢ Eq (s.sum fun i => HSMul.hSMul (w i) (VSub.vsub (p₁ i) (p₂ i))) (HSub.hSub ( …
  -/
  exact s.sum_smul_vsub_eq_weightedVSubOfPoint_sub _ _ _ _
  /-
    🎉 no goals
  -/


/-- A weighted sum of pairwise subtractions, where the point on the right is constant and the
sum of the weights is 1. -/
theorem sum_smul_vsub_const_eq_affineCombination_vsub (w : ι → k) (p₁ : ι → P) (p₂ : P)
    (h : ∑ i ∈ s, w i = 1) : (∑ i ∈ s, w i • (p₁ i -ᵥ p₂)) = s.affineCombination k p₁ w -ᵥ p₂ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p₁ : ι → P
    p₂ : P
    h : Eq (s.sum fun i => w i) 1
    ⊢ Eq (s.sum fun i => HSMul.hSMul (w i) (VSub.vsub (p₁ i) p₂)) (VSub.vsub ((Fin …
  -/
  rw [sum_smul_vsub_eq_affineCombination_vsub, affineCombination_apply_const _ _ _ h]
  /-
    🎉 no goals
  -/


/-- A weighted sum of pairwise subtractions, where the point on the left is constant and the
sum of the weights is 1. -/
theorem sum_smul_const_vsub_eq_vsub_affineCombination (w : ι → k) (p₂ : ι → P) (p₁ : P)
    (h : ∑ i ∈ s, w i = 1) : (∑ i ∈ s, w i • (p₁ -ᵥ p₂ i)) = p₁ -ᵥ s.affineCombination k p₂ w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p₂ : ι → P
    p₁ : P
    h : Eq (s.sum fun i => w i) 1
    ⊢ Eq (s.sum fun i => HSMul.hSMul (w i) (VSub.vsub p₁ (p₂ i))) (VSub.vsub p₁ (( …
  -/
  rw [sum_smul_vsub_eq_affineCombination_vsub, affineCombination_apply_const _ _ _ h]
  /-
    🎉 no goals
  -/


/-- A weighted sum may be split into a subtraction of affine combinations over two subsets. -/
theorem affineCombination_sdiff_sub [DecidableEq ι] {s₂ : Finset ι} (h : s₂ ⊆ s) (w : ι → k)
    (p : ι → P) :
    (s \ s₂).affineCombination k p w -ᵥ s₂.affineCombination k p (-w) = s.weightedVSub p w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    s₂ : Finset ι
    h : HasSubset.Subset s₂ s
    w : ι → k
    p : ι → P
    ⊢ Eq (VSub.vsub ((Finset.affineCombination k (SDiff.sdiff s s₂) p) w) ((Finset …
  -/
  simp_rw [affineCombination_apply, vadd_vsub_vadd_cancel_right]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    s₂ : Finset ι
    h : HasSubset.Subset s₂ s
    w : ι → k
    p : ι → P
    ⊢ Eq (HSub.hSub (((SDiff.sdiff s s₂).weightedVSubOfPoint p (Classical.choice ⋯ …
  -/
  exact s.weightedVSub_sdiff_sub h _ _
  /-
    🎉 no goals
  -/


/-- If a weighted sum is zero and one of the weights is `-1`, the corresponding point is
the affine combination of the other points with the given weights. -/
theorem affineCombination_eq_of_weightedVSub_eq_zero_of_eq_neg_one {w : ι → k} {p : ι → P}
    (hw : s.weightedVSub p w = (0 : V)) {i : ι} [DecidablePred (· ≠ i)] (his : i ∈ s)
    (hwi : w i = -1) : {x ∈ s | x ≠ i}.affineCombination k p w = p i := by
  classical
    rw [← @vsub_eq_zero_iff_eq V, ← hw,
      ← s.affineCombination_sdiff_sub (singleton_subset_iff.2 his), sdiff_singleton_eq_erase,
      ← filter_ne']
    congr
    refine (affineCombination_of_eq_one_of_eq_zero _ _ _ (mem_singleton_self _) ?_ ?_).symm
    · simp [hwi]
    · simp


/-- An affine combination over `s.subtype pred` equals one over `{x ∈ s | pred x}`. -/
theorem affineCombination_subtype_eq_filter (w : ι → k) (p : ι → P) (pred : ι → Prop)
    [DecidablePred pred] :
    ((s.subtype pred).affineCombination k (fun i => p i) fun i => w i) =
      {x ∈ s | pred x}.affineCombination k p w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    w : ι → k
    p : ι → P
    pred : ι → Prop
    inst✝ : DecidablePred pred
    ⊢ Eq ((Finset.affineCombination k (Finset.subtype pred s) fun i => p ↑i) fun i …
  -/
  rw [affineCombination_apply, affineCombination_apply, weightedVSubOfPoint_subtype_eq_filter]
  /-
    🎉 no goals
  -/


/-- An affine combination over `{x ∈ s | pred x}` equals one over `s` if all the weights at indices
in `s` not satisfying `pred` are zero. -/
theorem affineCombination_filter_of_ne (w : ι → k) (p : ι → P) {pred : ι → Prop}
    [DecidablePred pred] (h : ∀ i ∈ s, w i ≠ 0 → pred i) :
    {x ∈ s | pred x}.affineCombination k p w = s.affineCombination k p w := by
  rw [affineCombination_apply, affineCombination_apply,
    s.weightedVSubOfPoint_filter_of_ne _ _ _ h]


/-- Suppose an indexed family of points is given, along with a subset
of the index type.  A vector can be expressed as
`weightedVSubOfPoint` using a `Finset` lying within that subset and
with a given sum of weights if and only if it can be expressed as
`weightedVSubOfPoint` with that sum of weights for the
corresponding indexed family whose index type is the subtype
corresponding to that subset. -/
theorem eq_weightedVSubOfPoint_subset_iff_eq_weightedVSubOfPoint_subtype {v : V} {x : k} {s : Set ι}
    {p : ι → P} {b : P} :
    (∃ fs : Finset ι, ↑fs ⊆ s ∧ ∃ w : ι → k, ∑ i ∈ fs, w i = x ∧
        v = fs.weightedVSubOfPoint p b w) ↔
      ∃ (fs : Finset s) (w : s → k), ∑ i ∈ fs, w i = x ∧
        v = fs.weightedVSubOfPoint (fun i : s => p i) b w := by
  classical
    simp_rw [weightedVSubOfPoint_apply]
    constructor
    · rintro ⟨fs, hfs, w, rfl, rfl⟩
      exact ⟨fs.subtype s, fun i => w i, sum_subtype_of_mem _ hfs, (sum_subtype_of_mem _ hfs).symm⟩
    · rintro ⟨fs, w, rfl, rfl⟩
      refine
          ⟨fs.map (Function.Embedding.subtype _), map_subtype_subset _, fun i =>
            if h : i ∈ s then w ⟨i, h⟩ else 0, ?_, ?_⟩ <;>
        simp


/-- Suppose an indexed family of points is given, along with a subset
of the index type.  A vector can be expressed as `weightedVSub` using
a `Finset` lying within that subset and with sum of weights 0 if and
only if it can be expressed as `weightedVSub` with sum of weights 0
for the corresponding indexed family whose index type is the subtype
corresponding to that subset. -/
theorem eq_weightedVSub_subset_iff_eq_weightedVSub_subtype {v : V} {s : Set ι} {p : ι → P} :
    (∃ fs : Finset ι, ↑fs ⊆ s ∧ ∃ w : ι → k, ∑ i ∈ fs, w i = 0 ∧
        v = fs.weightedVSub p w) ↔
      ∃ (fs : Finset s) (w : s → k), ∑ i ∈ fs, w i = 0 ∧
        v = fs.weightedVSub (fun i : s => p i) w :=
  eq_weightedVSubOfPoint_subset_iff_eq_weightedVSubOfPoint_subtype


/-- Suppose an indexed family of points is given, along with a subset
of the index type.  A point can be expressed as an
`affineCombination` using a `Finset` lying within that subset and
with sum of weights 1 if and only if it can be expressed an
`affineCombination` with sum of weights 1 for the corresponding
indexed family whose index type is the subtype corresponding to that
subset. -/
theorem eq_affineCombination_subset_iff_eq_affineCombination_subtype {p0 : P} {s : Set ι}
    {p : ι → P} :
    (∃ fs : Finset ι, ↑fs ⊆ s ∧ ∃ w : ι → k, ∑ i ∈ fs, w i = 1 ∧
        p0 = fs.affineCombination k p w) ↔
      ∃ (fs : Finset s) (w : s → k), ∑ i ∈ fs, w i = 1 ∧
        p0 = fs.affineCombination k (fun i : s => p i) w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    p0 : P
    s : Set ι
    p : ι → P
    ⊢ Iff (Exists fun fs => And (HasSubset.Subset (↑fs) s) (Exists fun w => And (E …
  -/
  simp_rw [affineCombination_apply, eq_vadd_iff_vsub_eq]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    p0 : P
    s : Set ι
    p : ι → P
    ⊢ Iff (Exists fun fs => And (HasSubset.Subset (↑fs) s) (Exists fun w => And (E …
  -/
  exact eq_weightedVSubOfPoint_subset_iff_eq_weightedVSubOfPoint_subtype
  /-
    🎉 no goals
  -/


/-- Affine maps commute with affine combinations. -/
theorem map_affineCombination {V₂ P₂ : Type*} [AddCommGroup V₂] [Module k V₂] [AffineSpace V₂ P₂]
    (p : ι → P) (w : ι → k) (hw : s.sum w = 1) (f : P →ᵃ[k] P₂) :
    f (s.affineCombination k p w) = s.affineCombination k (f ∘ p) w := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁵ : Ring k
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    V₂ : Type u_6
    P₂ : Type u_7
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    p : ι → P
    w : ι → k
    hw : Eq (s.sum w) 1
    f : AffineMap k P P₂
    ⊢ Eq (f ((Finset.affineCombination k s p) w)) ((Finset.affineCombination k s ( …
  -/
  have b := Classical.choice (inferInstance : AffineSpace V P).nonempty
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁵ : Ring k
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    V₂ : Type u_6
    P₂ : Type u_7
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    p : ι → P
    w : ι → k
    hw : Eq (s.sum w) 1
    f : AffineMap k P P₂
    b : P
    ⊢ Eq (f ((Finset.affineCombination k s p) w)) ((Finset.affineCombination k s ( …
  -/
  have b₂ := Classical.choice (inferInstance : AffineSpace V₂ P₂).nonempty
  rw [s.affineCombination_eq_weightedVSubOfPoint_vadd_of_sum_eq_one w p hw b,
    s.affineCombination_eq_weightedVSubOfPoint_vadd_of_sum_eq_one w (f ∘ p) hw b₂, ←
    s.weightedVSubOfPoint_vadd_eq_of_sum_eq_one w (f ∘ p) hw (f b) b₂]
  simp only [weightedVSubOfPoint_apply, RingHom.id_apply, AffineMap.map_vadd,
    LinearMap.map_smulₛₗ, AffineMap.linearMap_vsub, map_sum, Function.comp_apply]


/-- Weights for expressing a single point as an affine combination. -/
def affineCombinationSingleWeights [DecidableEq ι] (i : ι) : ι → k :=
  Function.update (Function.const ι 0) i 1


@[simp]
theorem affineCombinationSingleWeights_apply_self [DecidableEq ι] (i : ι) :
                                                   /-
                                                     k : Type u_1
                                                     inst✝¹ : Ring k
                                                     ι : Type u_4
                                                     inst✝ : DecidableEq ι
                                                     i : ι
                                                     ⊢ Eq (Finset.affineCombinationSingleWeights k i i) 1
                                                   -/
    affineCombinationSingleWeights k i i = 1 := by simp [affineCombinationSingleWeights]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem affineCombinationSingleWeights_apply_of_ne [DecidableEq ι] {i j : ι} (h : j ≠ i) :
                                                   /-
                                                     k : Type u_1
                                                     inst✝¹ : Ring k
                                                     ι : Type u_4
                                                     inst✝ : DecidableEq ι
                                                     i j : ι
                                                     h : Ne j i
                                                     ⊢ Eq (Finset.affineCombinationSingleWeights k i j) 0
                                                   -/
    affineCombinationSingleWeights k i j = 0 := by simp [affineCombinationSingleWeights, h]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem sum_affineCombinationSingleWeights [DecidableEq ι] {i : ι} (h : i ∈ s) :
    ∑ j ∈ s, affineCombinationSingleWeights k i j = 1 := by
  /-
    k : Type u_1
    inst✝¹ : Ring k
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    i : ι
    h : Membership.mem s i
    ⊢ Eq (s.sum fun j => Finset.affineCombinationSingleWeights k i j) 1
  -/
  rw [← affineCombinationSingleWeights_apply_self k i]
  /-
    k : Type u_1
    inst✝¹ : Ring k
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    i : ι
    h : Membership.mem s i
    ⊢ Eq (s.sum fun j => Finset.affineCombinationSingleWeights k i j) (Finset.affi …
  -/
  exact sum_eq_single_of_mem i h fun j _ hj => affineCombinationSingleWeights_apply_of_ne k hj
  /-
    🎉 no goals
  -/


/-- Weights for expressing the subtraction of two points as a `weightedVSub`. -/
def weightedVSubVSubWeights [DecidableEq ι] (i j : ι) : ι → k :=
  affineCombinationSingleWeights k i - affineCombinationSingleWeights k j


@[simp]
theorem weightedVSubVSubWeights_self [DecidableEq ι] (i : ι) :
                                            /-
                                              k : Type u_1
                                              inst✝¹ : Ring k
                                              ι : Type u_4
                                              inst✝ : DecidableEq ι
                                              i : ι
                                              ⊢ Eq (Finset.weightedVSubVSubWeights k i i) 0
                                            -/
    weightedVSubVSubWeights k i i = 0 := by simp [weightedVSubVSubWeights]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem weightedVSubVSubWeights_apply_left [DecidableEq ι] {i j : ι} (h : i ≠ j) :
                                              /-
                                                k : Type u_1
                                                inst✝¹ : Ring k
                                                ι : Type u_4
                                                inst✝ : DecidableEq ι
                                                i j : ι
                                                h : Ne i j
                                                ⊢ Eq (Finset.weightedVSubVSubWeights k i j i) 1
                                              -/
    weightedVSubVSubWeights k i j i = 1 := by simp [weightedVSubVSubWeights, h]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem weightedVSubVSubWeights_apply_right [DecidableEq ι] {i j : ι} (h : i ≠ j) :
                                               /-
                                                 k : Type u_1
                                                 inst✝¹ : Ring k
                                                 ι : Type u_4
                                                 inst✝ : DecidableEq ι
                                                 i j : ι
                                                 h : Ne i j
                                                 ⊢ Eq (Finset.weightedVSubVSubWeights k i j j) (-1)
                                               -/
    weightedVSubVSubWeights k i j j = -1 := by simp [weightedVSubVSubWeights, h.symm]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem weightedVSubVSubWeights_apply_of_ne [DecidableEq ι] {i j t : ι} (hi : t ≠ i) (hj : t ≠ j) :
                                              /-
                                                k : Type u_1
                                                inst✝¹ : Ring k
                                                ι : Type u_4
                                                inst✝ : DecidableEq ι
                                                i j t : ι
                                                hi : Ne t i
                                                hj : Ne t j
                                                ⊢ Eq (Finset.weightedVSubVSubWeights k i j t) 0
                                              -/
    weightedVSubVSubWeights k i j t = 0 := by simp [weightedVSubVSubWeights, hi, hj]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem sum_weightedVSubVSubWeights [DecidableEq ι] {i j : ι} (hi : i ∈ s) (hj : j ∈ s) :
    ∑ t ∈ s, weightedVSubVSubWeights k i j t = 0 := by
  /-
    k : Type u_1
    inst✝¹ : Ring k
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    i j : ι
    hi : Membership.mem s i
    hj : Membership.mem s j
    ⊢ Eq (s.sum fun t => Finset.weightedVSubVSubWeights k i j t) 0
  -/
  simp_rw [weightedVSubVSubWeights, Pi.sub_apply, sum_sub_distrib]
  /-
    k : Type u_1
    inst✝¹ : Ring k
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    i j : ι
    hi : Membership.mem s i
    hj : Membership.mem s j
    ⊢ Eq (HSub.hSub (s.sum fun x => Finset.affineCombinationSingleWeights k i x) ( …
  -/
  simp [hi, hj]
  /-
    🎉 no goals
  -/


/-- Weights for expressing `lineMap` as an affine combination. -/
def affineCombinationLineMapWeights [DecidableEq ι] (i j : ι) (c : k) : ι → k :=
  c • weightedVSubVSubWeights k j i + affineCombinationSingleWeights k i


@[simp]
theorem affineCombinationLineMapWeights_self [DecidableEq ι] (i : ι) (c : k) :
    affineCombinationLineMapWeights i i c = affineCombinationSingleWeights k i := by
  /-
    k : Type u_1
    inst✝¹ : Ring k
    ι : Type u_4
    inst✝ : DecidableEq ι
    i : ι
    c : k
    ⊢ Eq (Finset.affineCombinationLineMapWeights i i c) (Finset.affineCombinationS …
  -/
  simp [affineCombinationLineMapWeights]
  /-
    🎉 no goals
  -/


@[simp]
theorem affineCombinationLineMapWeights_apply_left [DecidableEq ι] {i j : ι} (h : i ≠ j) (c : k) :
    affineCombinationLineMapWeights i j c i = 1 - c := by
  /-
    k : Type u_1
    inst✝¹ : Ring k
    ι : Type u_4
    inst✝ : DecidableEq ι
    i j : ι
    h : Ne i j
    c : k
    ⊢ Eq (Finset.affineCombinationLineMapWeights i j c i) (HSub.hSub 1 c)
  -/
  simp [affineCombinationLineMapWeights, h.symm, sub_eq_neg_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem affineCombinationLineMapWeights_apply_right [DecidableEq ι] {i j : ι} (h : i ≠ j) (c : k) :
    affineCombinationLineMapWeights i j c j = c := by
  /-
    k : Type u_1
    inst✝¹ : Ring k
    ι : Type u_4
    inst✝ : DecidableEq ι
    i j : ι
    h : Ne i j
    c : k
    ⊢ Eq (Finset.affineCombinationLineMapWeights i j c j) c
  -/
  simp [affineCombinationLineMapWeights, h.symm]
  /-
    🎉 no goals
  -/


@[simp]
theorem affineCombinationLineMapWeights_apply_of_ne [DecidableEq ι] {i j t : ι} (hi : t ≠ i)
    (hj : t ≠ j) (c : k) : affineCombinationLineMapWeights i j c t = 0 := by
  /-
    k : Type u_1
    inst✝¹ : Ring k
    ι : Type u_4
    inst✝ : DecidableEq ι
    i j t : ι
    hi : Ne t i
    hj : Ne t j
    c : k
    ⊢ Eq (Finset.affineCombinationLineMapWeights i j c t) 0
  -/
  simp [affineCombinationLineMapWeights, hi, hj]
  /-
    🎉 no goals
  -/


@[simp]
theorem sum_affineCombinationLineMapWeights [DecidableEq ι] {i j : ι} (hi : i ∈ s) (hj : j ∈ s)
    (c : k) : ∑ t ∈ s, affineCombinationLineMapWeights i j c t = 1 := by
  /-
    k : Type u_1
    inst✝¹ : Ring k
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    i j : ι
    hi : Membership.mem s i
    hj : Membership.mem s j
    c : k
    ⊢ Eq (s.sum fun t => Finset.affineCombinationLineMapWeights i j c t) 1
  -/
  simp_rw [affineCombinationLineMapWeights, Pi.add_apply, sum_add_distrib]
  /-
    k : Type u_1
    inst✝¹ : Ring k
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    i j : ι
    hi : Membership.mem s i
    hj : Membership.mem s j
    c : k
    ⊢ Eq (HAdd.hAdd (s.sum fun x => HSMul.hSMul c (Finset.weightedVSubVSubWeights  …
  -/
  simp [hi, hj, ← mul_sum]
  /-
    🎉 no goals
  -/


/-- An affine combination with `affineCombinationSingleWeights` gives the specified point. -/
@[simp]
theorem affineCombination_affineCombinationSingleWeights [DecidableEq ι] (p : ι → P) {i : ι}
    (hi : i ∈ s) : s.affineCombination k p (affineCombinationSingleWeights k i) = p i := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    p : ι → P
    i : ι
    hi : Membership.mem s i
    ⊢ Eq ((Finset.affineCombination k s p) (Finset.affineCombinationSingleWeights  …
  -/
  refine s.affineCombination_of_eq_one_of_eq_zero _ _ hi (by simp) ?_
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    p : ι → P
    i : ι
    hi : Membership.mem s i
    ⊢ ∀ (i2 : ι), Membership.mem s i2 → Ne i2 i → Eq (Finset.affineCombinationSing …
  -/
  rintro j - hj
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    inst✝ : DecidableEq ι
    p : ι → P
    i : ι
    hi : Membership.mem s i
    j : ι
    hj : Ne j i
    ⊢ Eq (Finset.affineCombinationSingleWeights k i j) 0
  -/
  simp [hj]
  /-
    🎉 no goals
  -/


/-- A weighted subtraction with `weightedVSubVSubWeights` gives the result of subtracting the
specified points. -/
@[simp]
theorem weightedVSub_weightedVSubVSubWeights [DecidableEq ι] (p : ι → P) {i j : ι} (hi : i ∈ s)
    (hj : j ∈ s) : s.weightedVSub p (weightedVSubVSubWeights k i j) = p i -ᵥ p j := by
  rw [weightedVSubVSubWeights, ← affineCombination_vsub,
    s.affineCombination_affineCombinationSingleWeights k p hi,
    s.affineCombination_affineCombinationSingleWeights k p hj]


/-- An affine combination with `affineCombinationLineMapWeights` gives the result of
`line_map`. -/
@[simp]
theorem affineCombination_affineCombinationLineMapWeights [DecidableEq ι] (p : ι → P) {i j : ι}
    (hi : i ∈ s) (hj : j ∈ s) (c : k) :
    s.affineCombination k p (affineCombinationLineMapWeights i j c) =
      AffineMap.lineMap (p i) (p j) c := by
  rw [affineCombinationLineMapWeights, ← weightedVSub_vadd_affineCombination,
    weightedVSub_const_smul, s.affineCombination_affineCombinationSingleWeights k p hi,
    s.weightedVSub_weightedVSubVSubWeights k p hj hi, AffineMap.lineMap_apply]


/-- The weights for the centroid of some points. -/
def centroidWeights : ι → k :=
  Function.const ι (#s : k)⁻¹


/-- `centroidWeights` at any point. -/
@[simp]
theorem centroidWeights_apply (i : ι) : s.centroidWeights k i = (#s : k)⁻¹ :=
  rfl


/-- `centroidWeights` equals a constant function. -/
theorem centroidWeights_eq_const : s.centroidWeights k = Function.const ι (#s : k)⁻¹ :=
  rfl


/-- The weights in the centroid sum to 1, if the number of points,
converted to `k`, is not zero. -/
theorem sum_centroidWeights_eq_one_of_cast_card_ne_zero (h : (#s : k) ≠ 0) :
                                             /-
                                               k : Type u_1
                                               inst✝ : DivisionRing k
                                               ι : Type u_4
                                               s : Finset ι
                                               h : Ne (↑s.card) 0
                                               ⊢ Eq (s.sum fun i => Finset.centroidWeights k s i) 1
                                             -/
    ∑ i ∈ s, s.centroidWeights k i = 1 := by simp [h]
                                             /-
                                               🎉 no goals
                                             -/


/-- In the characteristic zero case, the weights in the centroid sum
to 1 if the number of points is not zero. -/
theorem sum_centroidWeights_eq_one_of_card_ne_zero [CharZero k] (h : #s ≠ 0) :
    ∑ i ∈ s, s.centroidWeights k i = 1 := by
  -- Porting note: `simp` cannot find `mul_inv_cancel` and does not use `norm_cast`
  /-
    k : Type u_1
    inst✝¹ : DivisionRing k
    ι : Type u_4
    s : Finset ι
    inst✝ : CharZero k
    h : Ne s.card 0
    ⊢ Eq (s.sum fun i => Finset.centroidWeights k s i) 1
  -/
  simp only [centroidWeights_apply, sum_const, nsmul_eq_mul, ne_eq, Nat.cast_eq_zero, card_eq_zero]
  /-
    k : Type u_1
    inst✝¹ : DivisionRing k
    ι : Type u_4
    s : Finset ι
    inst✝ : CharZero k
    h : Ne s.card 0
    ⊢ Eq (HMul.hMul (↑s.card) (Inv.inv ↑s.card)) 1
  -/
  refine mul_inv_cancel₀ ?_
  /-
    k : Type u_1
    inst✝¹ : DivisionRing k
    ι : Type u_4
    s : Finset ι
    inst✝ : CharZero k
    h : Ne s.card 0
    ⊢ Ne (↑s.card) 0
  -/
  norm_cast
  /-
    🎉 no goals
  -/


/-- In the characteristic zero case, the weights in the centroid sum
to 1 if the set is nonempty. -/
theorem sum_centroidWeights_eq_one_of_nonempty [CharZero k] (h : s.Nonempty) :
    ∑ i ∈ s, s.centroidWeights k i = 1 :=
  s.sum_centroidWeights_eq_one_of_card_ne_zero k (ne_of_gt (card_pos.2 h))


/-- In the characteristic zero case, the weights in the centroid sum
to 1 if the number of points is `n + 1`. -/
theorem sum_centroidWeights_eq_one_of_card_eq_add_one [CharZero k] {n : ℕ} (h : #s = n + 1) :
    ∑ i ∈ s, s.centroidWeights k i = 1 :=
  s.sum_centroidWeights_eq_one_of_card_ne_zero k (h.symm ▸ Nat.succ_ne_zero n)


/-- The centroid of some points.  Although defined for any `s`, this
is intended to be used in the case where the number of points,
converted to `k`, is not zero. -/
def centroid (p : ι → P) : P :=
  s.affineCombination k p (s.centroidWeights k)


/-- The definition of the centroid. -/
theorem centroid_def (p : ι → P) : s.centroid k p = s.affineCombination k p (s.centroidWeights k) :=
  rfl


theorem centroid_univ (s : Finset P) : univ.centroid k ((↑) : s → P) = s.centroid k id := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Finset P
    ⊢ Eq (Finset.centroid k Finset.univ Subtype.val) (Finset.centroid k s id)
  -/
  rw [centroid, centroid, ← s.attach_affineCombination_coe]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Finset P
    ⊢ Eq ((Finset.affineCombination k Finset.univ Subtype.val) (Finset.centroidWei …
  -/
  congr
  /-
    case h.e_6.h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Finset P
    ⊢ Eq (Finset.centroidWeights k Finset.univ) (Function.comp (Finset.centroidWei …
  -/
  ext
  /-
    case h.e_6.h.h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Finset P
    x✝ : Subtype fun x => Membership.mem s x
    ⊢ Eq (Finset.centroidWeights k Finset.univ x✝) (Function.comp (Finset.centroid …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The centroid of a single point. -/
@[simp]
theorem centroid_singleton (p : ι → P) (i : ι) : ({i} : Finset ι).centroid k p = p i := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ι : Type u_4
    p : ι → P
    i : ι
    ⊢ Eq (Finset.centroid k (Singleton.singleton i) p) (p i)
  -/
  simp [centroid_def, affineCombination_apply]
  /-
    🎉 no goals
  -/


/-- The centroid of two points, expressed directly as adding a vector
to a point. -/
theorem centroid_pair [DecidableEq ι] [Invertible (2 : k)] (p : ι → P) (i₁ i₂ : ι) :
    ({i₁, i₂} : Finset ι).centroid k p = (2⁻¹ : k) • (p i₂ -ᵥ p i₁) +ᵥ p i₁ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁵ : DivisionRing k
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module k V
    inst✝² : AddTorsor V P
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Invertible 2
    p : ι → P
    i₁ i₂ : ι
    ⊢ Eq (Finset.centroid k (Insert.insert i₁ (Singleton.singleton i₂)) p) (HVAdd. …
  -/
  by_cases h : i₁ = i₂
    /-
      case pos
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝⁵ : DivisionRing k
      inst✝⁴ : AddCommGroup V
      inst✝³ : Module k V
      inst✝² : AddTorsor V P
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Invertible 2
      p : ι → P
      i₁ i₂ : ι
      h : Eq i₁ i₂
      ⊢ Eq (Finset.centroid k (Insert.insert i₁ (Singleton.singleton i₂)) p) (HVAdd. …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  · have hc : (#{i₁, i₂} : k) ≠ 0 := by
      rw [card_insert_of_not_mem (not_mem_singleton.2 h), card_singleton]
      norm_num
      exact Invertible.ne_zero _
    rw [centroid_def,
      affineCombination_eq_weightedVSubOfPoint_vadd_of_sum_eq_one _ _ _
        (sum_centroidWeights_eq_one_of_cast_card_ne_zero _ hc) (p i₁)]
    /-
      case neg
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝⁵ : DivisionRing k
      inst✝⁴ : AddCommGroup V
      inst✝³ : Module k V
      inst✝² : AddTorsor V P
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Invertible 2
      p : ι → P
      i₁ i₂ : ι
      h : Not (Eq i₁ i₂)
      hc : Ne (↑(Insert.insert i₁ (Singleton.singleton i₂)).card) 0
      ⊢ Eq (HVAdd.hVAdd (((Insert.insert i₁ (Singleton.singleton i₂)).weightedVSubOf …
    -/
    simp [h, one_add_one_eq_two]
    /-
      🎉 no goals
    -/


/-- The centroid of two points indexed by `Fin 2`, expressed directly
as adding a vector to the first point. -/
theorem centroid_pair_fin [Invertible (2 : k)] (p : Fin 2 → P) :
    univ.centroid k p = (2⁻¹ : k) • (p 1 -ᵥ p 0) +ᵥ p 0 := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    inst✝ : Invertible 2
    p : Fin 2 → P
    ⊢ Eq (Finset.centroid k Finset.univ p) (HVAdd.hVAdd (HSMul.hSMul (Inv.inv 2) ( …
  -/
  rw [univ_fin2]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    inst✝ : Invertible 2
    p : Fin 2 → P
    ⊢ Eq (Finset.centroid k (Insert.insert 0 (Singleton.singleton 1)) p) (HVAdd.hV …
  -/
  convert centroid_pair k p 0 1
  /-
    🎉 no goals
  -/


/-- A centroid, over the image of an embedding, equals a centroid with
the same points and weights over the original `Finset`. -/
theorem centroid_map (e : ι₂ ↪ ι) (p : ι → P) :
    (s₂.map e).centroid k p = s₂.centroid k (p ∘ e) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : DivisionRing k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ι : Type u_4
    ι₂ : Type u_5
    s₂ : Finset ι₂
    e : Function.Embedding ι₂ ι
    p : ι → P
    ⊢ Eq (Finset.centroid k (Finset.map e s₂) p) (Finset.centroid k s₂ (Function.c …
  -/
  simp [centroid_def, affineCombination_map, centroidWeights]
  /-
    🎉 no goals
  -/


/-- `centroidWeights` gives the weights for the centroid as a
constant function, which is suitable when summing over the points
whose centroid is being taken.  This function gives the weights in a
form suitable for summing over a larger set of points, as an indicator
function that is zero outside the set whose centroid is being taken.
In the case of a `Fintype`, the sum may be over `univ`. -/
def centroidWeightsIndicator : ι → k :=
  Set.indicator (↑s) (s.centroidWeights k)


/-- The definition of `centroidWeightsIndicator`. -/
theorem centroidWeightsIndicator_def :
    s.centroidWeightsIndicator k = Set.indicator (↑s) (s.centroidWeights k) :=
  rfl


/-- The sum of the weights for the centroid indexed by a `Fintype`. -/
theorem sum_centroidWeightsIndicator [Fintype ι] :
    ∑ i, s.centroidWeightsIndicator k i = ∑ i ∈ s, s.centroidWeights k i :=
  sum_indicator_subset _ (subset_univ _)


/-- In the characteristic zero case, the weights in the centroid
indexed by a `Fintype` sum to 1 if the number of points is not
zero. -/
theorem sum_centroidWeightsIndicator_eq_one_of_card_ne_zero [CharZero k] [Fintype ι]
    (h : #s ≠ 0) : ∑ i, s.centroidWeightsIndicator k i = 1 := by
  /-
    k : Type u_1
    inst✝² : DivisionRing k
    ι : Type u_4
    s : Finset ι
    inst✝¹ : CharZero k
    inst✝ : Fintype ι
    h : Ne s.card 0
    ⊢ Eq (Finset.univ.sum fun i => Finset.centroidWeightsIndicator k s i) 1
  -/
  rw [sum_centroidWeightsIndicator]
  /-
    k : Type u_1
    inst✝² : DivisionRing k
    ι : Type u_4
    s : Finset ι
    inst✝¹ : CharZero k
    inst✝ : Fintype ι
    h : Ne s.card 0
    ⊢ Eq (s.sum fun i => Finset.centroidWeights k s i) 1
  -/
  exact s.sum_centroidWeights_eq_one_of_card_ne_zero k h
  /-
    🎉 no goals
  -/


/-- In the characteristic zero case, the weights in the centroid
indexed by a `Fintype` sum to 1 if the set is nonempty. -/
theorem sum_centroidWeightsIndicator_eq_one_of_nonempty [CharZero k] [Fintype ι] (h : s.Nonempty) :
    ∑ i, s.centroidWeightsIndicator k i = 1 := by
  /-
    k : Type u_1
    inst✝² : DivisionRing k
    ι : Type u_4
    s : Finset ι
    inst✝¹ : CharZero k
    inst✝ : Fintype ι
    h : s.Nonempty
    ⊢ Eq (Finset.univ.sum fun i => Finset.centroidWeightsIndicator k s i) 1
  -/
  rw [sum_centroidWeightsIndicator]
  /-
    k : Type u_1
    inst✝² : DivisionRing k
    ι : Type u_4
    s : Finset ι
    inst✝¹ : CharZero k
    inst✝ : Fintype ι
    h : s.Nonempty
    ⊢ Eq (s.sum fun i => Finset.centroidWeights k s i) 1
  -/
  exact s.sum_centroidWeights_eq_one_of_nonempty k h
  /-
    🎉 no goals
  -/


/-- In the characteristic zero case, the weights in the centroid
indexed by a `Fintype` sum to 1 if the number of points is `n + 1`. -/
theorem sum_centroidWeightsIndicator_eq_one_of_card_eq_add_one [CharZero k] [Fintype ι] {n : ℕ}
    (h : #s = n + 1) : ∑ i, s.centroidWeightsIndicator k i = 1 := by
  /-
    k : Type u_1
    inst✝² : DivisionRing k
    ι : Type u_4
    s : Finset ι
    inst✝¹ : CharZero k
    inst✝ : Fintype ι
    n : Nat
    h : Eq s.card (HAdd.hAdd n 1)
    ⊢ Eq (Finset.univ.sum fun i => Finset.centroidWeightsIndicator k s i) 1
  -/
  rw [sum_centroidWeightsIndicator]
  /-
    k : Type u_1
    inst✝² : DivisionRing k
    ι : Type u_4
    s : Finset ι
    inst✝¹ : CharZero k
    inst✝ : Fintype ι
    n : Nat
    h : Eq s.card (HAdd.hAdd n 1)
    ⊢ Eq (s.sum fun i => Finset.centroidWeights k s i) 1
  -/
  exact s.sum_centroidWeights_eq_one_of_card_eq_add_one k h
  /-
    🎉 no goals
  -/


/-- The centroid as an affine combination over a `Fintype`. -/
theorem centroid_eq_affineCombination_fintype [Fintype ι] (p : ι → P) :
    s.centroid k p = univ.affineCombination k p (s.centroidWeightsIndicator k) :=
  affineCombination_indicator_subset _ _ (subset_univ _)


/-- An indexed family of points that is injective on the given
`Finset` has the same centroid as the image of that `Finset`.  This is
stated in terms of a set equal to the image to provide control of
definitional equality for the index type used for the centroid of the
image. -/
theorem centroid_eq_centroid_image_of_inj_on {p : ι → P}
    (hi : ∀ i ∈ s, ∀ j ∈ s, p i = p j → i = j) {ps : Set P} [Fintype ps]
    (hps : ps = p '' ↑s) : s.centroid k p = (univ : Finset ps).centroid k fun x => (x : P) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    p : ι → P
    hi : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Eq (p i)  …
    ps : Set P
    inst✝ : Fintype ↑ps
    hps : Eq ps (Set.image p ↑s)
    ⊢ Eq (Finset.centroid k s p) (Finset.centroid k Finset.univ fun x => ↑x)
  -/
  let f : p '' ↑s → ι := fun x => x.property.choose
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    p : ι → P
    hi : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Eq (p i)  …
    ps : Set P
    inst✝ : Fintype ↑ps
    hps : Eq ps (Set.image p ↑s)
    f : ↑(Set.image p ↑s) → ι := fun x => Exists.choose ⋯
    ⊢ Eq (Finset.centroid k s p) (Finset.centroid k Finset.univ fun x => ↑x)
  -/
  have hf : ∀ x, f x ∈ s ∧ p (f x) = x := fun x => x.property.choose_spec
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    p : ι → P
    hi : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Eq (p i)  …
    ps : Set P
    inst✝ : Fintype ↑ps
    hps : Eq ps (Set.image p ↑s)
    f : ↑(Set.image p ↑s) → ι := fun x => Exists.choose ⋯
    hf : ∀ (x : ↑(Set.image p ↑s)), And (Membership.mem s (f x)) (Eq (p (f x)) ↑x)
    ⊢ Eq (Finset.centroid k s p) (Finset.centroid k Finset.univ fun x => ↑x)
  -/
  let f' : ps → ι := fun x => f ⟨x, hps ▸ x.property⟩
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    p : ι → P
    hi : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Eq (p i)  …
    ps : Set P
    inst✝ : Fintype ↑ps
    hps : Eq ps (Set.image p ↑s)
    f : ↑(Set.image p ↑s) → ι := fun x => Exists.choose ⋯
    hf : ∀ (x : ↑(Set.image p ↑s)), And (Membership.mem s (f x)) (Eq (p (f x)) ↑x)
    f' : ↑ps → ι := fun x => f ⟨↑x, ⋯⟩
    ⊢ Eq (Finset.centroid k s p) (Finset.centroid k Finset.univ fun x => ↑x)
  -/
  have hf' : ∀ x, f' x ∈ s ∧ p (f' x) = x := fun x => hf ⟨x, hps ▸ x.property⟩
  have hf'i : Function.Injective f' := by
    intro x y h
    rw [Subtype.ext_iff, ← (hf' x).2, ← (hf' y).2, h]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    p : ι → P
    hi : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Eq (p i)  …
    ps : Set P
    inst✝ : Fintype ↑ps
    hps : Eq ps (Set.image p ↑s)
    f : ↑(Set.image p ↑s) → ι := fun x => Exists.choose ⋯
    hf : ∀ (x : ↑(Set.image p ↑s)), And (Membership.mem s (f x)) (Eq (p (f x)) ↑x)
    f' : ↑ps → ι := fun x => f ⟨↑x, ⋯⟩
    hf' : ∀ (x : ↑ps), And (Membership.mem s (f' x)) (Eq (p (f' x)) ↑x)
    hf'i : Function.Injective f'
    ⊢ Eq (Finset.centroid k s p) (Finset.centroid k Finset.univ fun x => ↑x)
  -/
  let f'e : ps ↪ ι := ⟨f', hf'i⟩
  have hu : Finset.univ.map f'e = s := by
    ext x
    rw [mem_map]
    constructor
    · rintro ⟨i, _, rfl⟩
      exact (hf' i).1
    · intro hx
      use ⟨p x, hps.symm ▸ Set.mem_image_of_mem _ hx⟩, mem_univ _
      refine hi _ (hf' _).1 _ hx ?_
      rw [(hf' _).2]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    p : ι → P
    hi : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Eq (p i)  …
    ps : Set P
    inst✝ : Fintype ↑ps
    hps : Eq ps (Set.image p ↑s)
    f : ↑(Set.image p ↑s) → ι := fun x => Exists.choose ⋯
    hf : ∀ (x : ↑(Set.image p ↑s)), And (Membership.mem s (f x)) (Eq (p (f x)) ↑x)
    f' : ↑ps → ι := fun x => f ⟨↑x, ⋯⟩
    hf' : ∀ (x : ↑ps), And (Membership.mem s (f' x)) (Eq (p (f' x)) ↑x)
    hf'i : Function.Injective f'
    f'e : Function.Embedding (↑ps) ι := { toFun := f', inj' := hf'i }
    hu : Eq (Finset.map f'e Finset.univ) s
    ⊢ Eq (Finset.centroid k s p) (Finset.centroid k Finset.univ fun x => ↑x)
  -/
  rw [← hu, centroid_map]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    p : ι → P
    hi : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Eq (p i)  …
    ps : Set P
    inst✝ : Fintype ↑ps
    hps : Eq ps (Set.image p ↑s)
    f : ↑(Set.image p ↑s) → ι := fun x => Exists.choose ⋯
    hf : ∀ (x : ↑(Set.image p ↑s)), And (Membership.mem s (f x)) (Eq (p (f x)) ↑x)
    f' : ↑ps → ι := fun x => f ⟨↑x, ⋯⟩
    hf' : ∀ (x : ↑ps), And (Membership.mem s (f' x)) (Eq (p (f' x)) ↑x)
    hf'i : Function.Injective f'
    f'e : Function.Embedding (↑ps) ι := { toFun := f', inj' := hf'i }
    hu : Eq (Finset.map f'e Finset.univ) s
    ⊢ Eq (Finset.centroid k Finset.univ (Function.comp p ⇑f'e)) (Finset.centroid k …
  -/
  congr with x
  /-
    case e_p.h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    p : ι → P
    hi : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Eq (p i)  …
    ps : Set P
    inst✝ : Fintype ↑ps
    hps : Eq ps (Set.image p ↑s)
    f : ↑(Set.image p ↑s) → ι := fun x => Exists.choose ⋯
    hf : ∀ (x : ↑(Set.image p ↑s)), And (Membership.mem s (f x)) (Eq (p (f x)) ↑x)
    f' : ↑ps → ι := fun x => f ⟨↑x, ⋯⟩
    hf' : ∀ (x : ↑ps), And (Membership.mem s (f' x)) (Eq (p (f' x)) ↑x)
    hf'i : Function.Injective f'
    f'e : Function.Embedding (↑ps) ι := { toFun := f', inj' := hf'i }
    hu : Eq (Finset.map f'e Finset.univ) s
    x : ↑ps
    ⊢ Eq (Function.comp p (⇑f'e) x) ↑x
  -/
  change p (f' x) = ↑x
  /-
    case e_p.h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : DivisionRing k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    ι : Type u_4
    s : Finset ι
    p : ι → P
    hi : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Eq (p i)  …
    ps : Set P
    inst✝ : Fintype ↑ps
    hps : Eq ps (Set.image p ↑s)
    f : ↑(Set.image p ↑s) → ι := fun x => Exists.choose ⋯
    hf : ∀ (x : ↑(Set.image p ↑s)), And (Membership.mem s (f x)) (Eq (p (f x)) ↑x)
    f' : ↑ps → ι := fun x => f ⟨↑x, ⋯⟩
    hf' : ∀ (x : ↑ps), And (Membership.mem s (f' x)) (Eq (p (f' x)) ↑x)
    hf'i : Function.Injective f'
    f'e : Function.Embedding (↑ps) ι := { toFun := f', inj' := hf'i }
    hu : Eq (Finset.map f'e Finset.univ) s
    x : ↑ps
    ⊢ Eq (p (f' x)) ↑x
  -/
  rw [(hf' x).2]
  /-
    🎉 no goals
  -/


/-- Two indexed families of points that are injective on the given
`Finset`s and with the same points in the image of those `Finset`s
have the same centroid. -/
theorem centroid_eq_of_inj_on_of_image_eq {p : ι → P}
    (hi : ∀ i ∈ s, ∀ j ∈ s, p i = p j → i = j) {p₂ : ι₂ → P}
    (hi₂ : ∀ i ∈ s₂, ∀ j ∈ s₂, p₂ i = p₂ j → i = j) (he : p '' ↑s = p₂ '' ↑s₂) :
    s.centroid k p = s₂.centroid k p₂ := by
  classical rw [s.centroid_eq_centroid_image_of_inj_on k hi rfl,
      s₂.centroid_eq_centroid_image_of_inj_on k hi₂ he]


/-- A `weightedVSub` with sum of weights 0 is in the `vectorSpan` of
an indexed family. -/
theorem weightedVSub_mem_vectorSpan {s : Finset ι} {w : ι → k} (h : ∑ i ∈ s, w i = 0)
    (p : ι → P) : s.weightedVSub p w ∈ vectorSpan k (Set.range p) := by
  classical
    rcases isEmpty_or_nonempty ι with (hι | ⟨⟨i0⟩⟩)
    · simp [Finset.eq_empty_of_isEmpty s]
    · rw [vectorSpan_range_eq_span_range_vsub_right k p i0, ← Set.image_univ,
        Finsupp.mem_span_image_iff_linearCombination,
        Finset.weightedVSub_eq_weightedVSubOfPoint_of_sum_eq_zero s w p h (p i0),
        Finset.weightedVSubOfPoint_apply]
      let w' := Set.indicator (↑s) w
      have hwx : ∀ i, w' i ≠ 0 → i ∈ s := fun i => Set.mem_of_indicator_ne_zero
      use Finsupp.onFinset s w' hwx, Set.subset_univ _
      rw [Finsupp.linearCombination_apply, Finsupp.onFinset_sum hwx]
      · apply Finset.sum_congr rfl
        intro i hi
        simp [w', Set.indicator_apply, if_pos hi]
      · exact fun _ => zero_smul k _


/-- An `affineCombination` with sum of weights 1 is in the
`affineSpan` of an indexed family, if the underlying ring is
nontrivial. -/
theorem affineCombination_mem_affineSpan [Nontrivial k] {s : Finset ι} {w : ι → k}
    (h : ∑ i ∈ s, w i = 1) (p : ι → P) :
    s.affineCombination k p w ∈ affineSpan k (Set.range p) := by
  classical
    have hnz : ∑ i ∈ s, w i ≠ 0 := h.symm ▸ one_ne_zero
    have hn : s.Nonempty := Finset.nonempty_of_sum_ne_zero hnz
    cases' hn with i1 hi1
    let w1 : ι → k := Function.update (Function.const ι 0) i1 1
    have hw1 : ∑ i ∈ s, w1 i = 1 := by
      simp only [w1, Function.const_zero, Finset.sum_update_of_mem hi1, Pi.zero_apply,
          Finset.sum_const_zero, add_zero]
    have hw1s : s.affineCombination k p w1 = p i1 :=
      s.affineCombination_of_eq_one_of_eq_zero w1 p hi1 (Function.update_self ..) fun _ _ hne =>
        Function.update_of_ne hne ..
    have hv : s.affineCombination k p w -ᵥ p i1 ∈ (affineSpan k (Set.range p)).direction := by
      rw [direction_affineSpan, ← hw1s, Finset.affineCombination_vsub]
      apply weightedVSub_mem_vectorSpan
      simp [Pi.sub_apply, h, hw1]
    rw [← vsub_vadd (s.affineCombination k p w) (p i1)]
    exact AffineSubspace.vadd_mem_of_mem_direction hv (mem_affineSpan k (Set.mem_range_self _))


/-- A vector is in the `vectorSpan` of an indexed family if and only
if it is a `weightedVSub` with sum of weights 0. -/
theorem mem_vectorSpan_iff_eq_weightedVSub {v : V} {p : ι → P} :
    v ∈ vectorSpan k (Set.range p) ↔
      ∃ (s : Finset ι) (w : ι → k), ∑ i ∈ s, w i = 0 ∧ v = s.weightedVSub p w := by
  classical
    constructor
    · rcases isEmpty_or_nonempty ι with (hι | ⟨⟨i0⟩⟩)
      swap
      · rw [vectorSpan_range_eq_span_range_vsub_right k p i0, ← Set.image_univ,
          Finsupp.mem_span_image_iff_linearCombination]
        rintro ⟨l, _, hv⟩
        use insert i0 l.support
        set w :=
          (l : ι → k) - Function.update (Function.const ι 0 : ι → k) i0 (∑ i ∈ l.support, l i) with
          hwdef
        use w
        have hw : ∑ i ∈ insert i0 l.support, w i = 0 := by
          rw [hwdef]
          simp_rw [Pi.sub_apply, Finset.sum_sub_distrib,
            Finset.sum_update_of_mem (Finset.mem_insert_self _ _),
            Finset.sum_insert_of_eq_zero_if_not_mem Finsupp.not_mem_support_iff.1]
          simp only [Finsupp.mem_support_iff, ne_eq, Finset.mem_insert, true_or, not_true,
            Function.const_apply, Finset.sum_const_zero, add_zero, sub_self]
        use hw
        have hz : w i0 • (p i0 -ᵥ p i0 : V) = 0 := (vsub_self (p i0)).symm ▸ smul_zero _
        change (fun i => w i • (p i -ᵥ p i0 : V)) i0 = 0 at hz
        rw [Finset.weightedVSub_eq_weightedVSubOfPoint_of_sum_eq_zero _ w p hw (p i0),
          Finset.weightedVSubOfPoint_apply, ← hv, Finsupp.linearCombination_apply,
          @Finset.sum_insert_zero _ _ l.support i0 _ _ _ hz]
        change (∑ i ∈ l.support, l i • _) = _
        congr with i
        by_cases h : i = i0
        · simp [h]
        · simp [hwdef, h]
      · rw [Set.range_eq_empty, vectorSpan_empty, Submodule.mem_bot]
        rintro rfl
        use ∅
        simp
    · rintro ⟨s, w, hw, rfl⟩
      exact weightedVSub_mem_vectorSpan hw p


/-- A point in the `affineSpan` of an indexed family is an
`affineCombination` with sum of weights 1. See also
`eq_affineCombination_of_mem_affineSpan_of_fintype`. -/
theorem eq_affineCombination_of_mem_affineSpan {p1 : P} {p : ι → P}
    (h : p1 ∈ affineSpan k (Set.range p)) :
    ∃ (s : Finset ι) (w : ι → k), ∑ i ∈ s, w i = 1 ∧ p1 = s.affineCombination k p w := by
  classical
    have hn : (affineSpan k (Set.range p) : Set P).Nonempty := ⟨p1, h⟩
    rw [affineSpan_nonempty, Set.range_nonempty_iff_nonempty] at hn
    cases' hn with i0
    have h0 : p i0 ∈ affineSpan k (Set.range p) := mem_affineSpan k (Set.mem_range_self i0)
    have hd : p1 -ᵥ p i0 ∈ (affineSpan k (Set.range p)).direction :=
      AffineSubspace.vsub_mem_direction h h0
    rw [direction_affineSpan, mem_vectorSpan_iff_eq_weightedVSub] at hd
    rcases hd with ⟨s, w, h, hs⟩
    let s' := insert i0 s
    let w' := Set.indicator (↑s) w
    have h' : ∑ i ∈ s', w' i = 0 := by
      rw [← h, Finset.sum_indicator_subset _ (Finset.subset_insert i0 s)]
    have hs' : s'.weightedVSub p w' = p1 -ᵥ p i0 := by
      rw [hs]
      exact (Finset.weightedVSub_indicator_subset _ _ (Finset.subset_insert i0 s)).symm
    let w0 : ι → k := Function.update (Function.const ι 0) i0 1
    have hw0 : ∑ i ∈ s', w0 i = 1 := by
      rw [Finset.sum_update_of_mem (Finset.mem_insert_self _ _)]
      simp only [Finset.mem_insert, true_or, not_true, Function.const_apply, Finset.sum_const_zero,
        add_zero]
    have hw0s : s'.affineCombination k p w0 = p i0 :=
      s'.affineCombination_of_eq_one_of_eq_zero w0 p (Finset.mem_insert_self _ _)
        (Function.update_self ..) fun _ _ hne => Function.update_of_ne hne _ _
    refine ⟨s', w0 + w', ?_, ?_⟩
    · simp [Pi.add_apply, Finset.sum_add_distrib, hw0, h']
    · rw [add_comm, ← Finset.weightedVSub_vadd_affineCombination, hw0s, hs', vsub_vadd]


theorem eq_affineCombination_of_mem_affineSpan_of_fintype [Fintype ι] {p1 : P} {p : ι → P}
    (h : p1 ∈ affineSpan k (Set.range p)) :
    ∃ w : ι → k, ∑ i, w i = 1 ∧ p1 = Finset.univ.affineCombination k p w := by
  classical
    obtain ⟨s, w, hw, rfl⟩ := eq_affineCombination_of_mem_affineSpan h
    refine
      ⟨(s : Set ι).indicator w, ?_, Finset.affineCombination_indicator_subset w p s.subset_univ⟩
    simp only [Finset.mem_coe, Set.indicator_apply, ← hw]
    rw [Fintype.sum_extend_by_zero s w]


/-- A point is in the `affineSpan` of an indexed family if and only
if it is an `affineCombination` with sum of weights 1, provided the
underlying ring is nontrivial. -/
theorem mem_affineSpan_iff_eq_affineCombination [Nontrivial k] {p1 : P} {p : ι → P} :
    p1 ∈ affineSpan k (Set.range p) ↔
      ∃ (s : Finset ι) (w : ι → k), ∑ i ∈ s, w i = 1 ∧ p1 = s.affineCombination k p w := by
  /-
    ι : Type u_1
    k : Type u_2
    V : Type u_3
    P : Type u_4
    inst✝⁴ : Ring k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    inst✝ : Nontrivial k
    p1 : P
    p : ι → P
    ⊢ Iff (Membership.mem (affineSpan k (Set.range p)) p1) (Exists fun s => Exists …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      k : Type u_2
      V : Type u_3
      P : Type u_4
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      inst✝ : Nontrivial k
      p1 : P
      p : ι → P
      ⊢ Membership.mem (affineSpan k (Set.range p)) p1 → Exists fun s => Exists fun  …
    -/
  · exact eq_affineCombination_of_mem_affineSpan
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      k : Type u_2
      V : Type u_3
      P : Type u_4
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      inst✝ : Nontrivial k
      p1 : P
      p : ι → P
      ⊢ (Exists fun s => Exists fun w => And (Eq (s.sum fun i => w i) 1) (Eq p1 ((Fi …
    -/
  · rintro ⟨s, w, hw, rfl⟩
    /-
      case mpr.intro.intro.intro
      ι : Type u_1
      k : Type u_2
      V : Type u_3
      P : Type u_4
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      inst✝ : Nontrivial k
      p : ι → P
      s : Finset ι
      w : ι → k
      hw : Eq (s.sum fun i => w i) 1
      ⊢ Membership.mem (affineSpan k (Set.range p)) ((Finset.affineCombination k s p …
    -/
    exact affineCombination_mem_affineSpan hw p
    /-
      🎉 no goals
    -/


/-- Given a family of points together with a chosen base point in that family, membership of the
affine span of this family corresponds to an identity in terms of `weightedVSubOfPoint`, with
weights that are not required to sum to 1. -/
theorem mem_affineSpan_iff_eq_weightedVSubOfPoint_vadd [Nontrivial k] (p : ι → P) (j : ι) (q : P) :
    q ∈ affineSpan k (Set.range p) ↔
      ∃ (s : Finset ι) (w : ι → k), q = s.weightedVSubOfPoint p (p j) w +ᵥ p j := by
  /-
    ι : Type u_1
    k : Type u_2
    V : Type u_3
    P : Type u_4
    inst✝⁴ : Ring k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    inst✝ : Nontrivial k
    p : ι → P
    j : ι
    q : P
    ⊢ Iff (Membership.mem (affineSpan k (Set.range p)) q) (Exists fun s => Exists  …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      k : Type u_2
      V : Type u_3
      P : Type u_4
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      inst✝ : Nontrivial k
      p : ι → P
      j : ι
      q : P
      ⊢ Membership.mem (affineSpan k (Set.range p)) q → Exists fun s => Exists fun w …
    -/
  · intro hq
    /-
      case mp
      ι : Type u_1
      k : Type u_2
      V : Type u_3
      P : Type u_4
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      inst✝ : Nontrivial k
      p : ι → P
      j : ι
      q : P
      hq : Membership.mem (affineSpan k (Set.range p)) q
      ⊢ Exists fun s => Exists fun w => Eq q (HVAdd.hVAdd ((s.weightedVSubOfPoint p  …
    -/
    obtain ⟨s, w, hw, rfl⟩ := eq_affineCombination_of_mem_affineSpan hq
    /-
      case mp.intro.intro.intro
      ι : Type u_1
      k : Type u_2
      V : Type u_3
      P : Type u_4
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      inst✝ : Nontrivial k
      p : ι → P
      j : ι
      s : Finset ι
      w : ι → k
      hw : Eq (s.sum fun i => w i) 1
      hq : Membership.mem (affineSpan k (Set.range p)) ((Finset.affineCombination k  …
      ⊢ Exists fun s_1 => Exists fun w_1 => Eq ((Finset.affineCombination k s p) w)  …
    -/
    exact ⟨s, w, s.affineCombination_eq_weightedVSubOfPoint_vadd_of_sum_eq_one w p hw (p j)⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      k : Type u_2
      V : Type u_3
      P : Type u_4
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      inst✝ : Nontrivial k
      p : ι → P
      j : ι
      q : P
      ⊢ (Exists fun s => Exists fun w => Eq q (HVAdd.hVAdd ((s.weightedVSubOfPoint p …
    -/
  · rintro ⟨s, w, rfl⟩
    classical
      let w' : ι → k := Function.update w j (1 - (s \ {j}).sum w)
      have h₁ : (insert j s).sum w' = 1 := by
        by_cases hj : j ∈ s
        · simp [w', Finset.sum_update_of_mem hj, Finset.insert_eq_of_mem hj]
        · simp [w', Finset.sum_insert hj, Finset.sum_update_of_not_mem hj, hj]
      have hww : ∀ i, i ≠ j → w i = w' i := by
        intro i hij
        simp [w', hij]
      rw [s.weightedVSubOfPoint_eq_of_weights_eq p j w w' hww, ←
        s.weightedVSubOfPoint_insert w' p j, ←
        (insert j s).affineCombination_eq_weightedVSubOfPoint_vadd_of_sum_eq_one w' p h₁ (p j)]
      exact affineCombination_mem_affineSpan h₁ p


/-- Given a set of points, together with a chosen base point in this set, if we affinely transport
all other members of the set along the line joining them to this base point, the affine span is
unchanged. -/
theorem affineSpan_eq_affineSpan_lineMap_units [Nontrivial k] {s : Set P} {p : P} (hp : p ∈ s)
    (w : s → Units k) :
    affineSpan k (Set.range fun q : s => AffineMap.lineMap p ↑q (w q : k)) = affineSpan k s := by
  /-
    k : Type u_2
    V : Type u_3
    P : Type u_4
    inst✝⁴ : Ring k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    inst✝ : Nontrivial k
    s : Set P
    p : P
    hp : Membership.mem s p
    w : ↑s → Units k
    ⊢ Eq (affineSpan k (Set.range fun q => (AffineMap.lineMap p ↑q) ↑(w q))) (affi …
  -/
  have : s = Set.range ((↑) : s → P) := by simp
  conv_rhs =>
    rw [this]

  apply le_antisymm
    <;> intro q hq
    <;> erw [mem_affineSpan_iff_eq_weightedVSubOfPoint_vadd k V _ (⟨p, hp⟩ : s) q] at hq ⊢
    <;> obtain ⟨t, μ, rfl⟩ := hq
    <;> use t
    <;> [use fun x => μ x * ↑(w x); use fun x => μ x * ↑(w x)⁻¹]
        /-
          case h
          k : Type u_2
          V : Type u_3
          P : Type u_4
          inst✝⁴ : Ring k
          inst✝³ : AddCommGroup V
          inst✝² : Module k V
          inst✝¹ : AddTorsor V P
          inst✝ : Nontrivial k
          s : Set P
          p : P
          hp : Membership.mem s p
          w : ↑s → Units k
          this : Eq s (Set.range Subtype.val)
          t : Finset (Subtype fun x => Membership.mem s x)
          μ : (Subtype fun x => Membership.mem s x) → k
          ⊢ Eq (HVAdd.hVAdd ((t.weightedVSubOfPoint (fun q => (AffineMap.lineMap p ↑q) ↑ …
        -/
        /-
          🎉 no goals
        -/
    <;> simp [smul_smul]
        /-
          🎉 no goals
        -/


/-- The centroid lies in the affine span if the number of points,
converted to `k`, is not zero. -/
theorem centroid_mem_affineSpan_of_cast_card_ne_zero {s : Finset ι} (p : ι → P)
    (h : (#s : k) ≠ 0) : s.centroid k p ∈ affineSpan k (range p) :=
  affineCombination_mem_affineSpan (s.sum_centroidWeights_eq_one_of_cast_card_ne_zero h) p


/-- In the characteristic zero case, the centroid lies in the affine
span if the number of points is not zero. -/
theorem centroid_mem_affineSpan_of_card_ne_zero [CharZero k] {s : Finset ι} (p : ι → P)
    (h : #s ≠ 0) : s.centroid k p ∈ affineSpan k (range p) :=
  affineCombination_mem_affineSpan (s.sum_centroidWeights_eq_one_of_card_ne_zero k h) p


/-- In the characteristic zero case, the centroid lies in the affine
span if the set is nonempty. -/
theorem centroid_mem_affineSpan_of_nonempty [CharZero k] {s : Finset ι} (p : ι → P)
    (h : s.Nonempty) : s.centroid k p ∈ affineSpan k (range p) :=
  affineCombination_mem_affineSpan (s.sum_centroidWeights_eq_one_of_nonempty k h) p


/-- In the characteristic zero case, the centroid lies in the affine
span if the number of points is `n + 1`. -/
theorem centroid_mem_affineSpan_of_card_eq_add_one [CharZero k] {s : Finset ι} (p : ι → P) {n : ℕ}
    (h : #s = n + 1) : s.centroid k p ∈ affineSpan k (range p) :=
  affineCombination_mem_affineSpan (s.sum_centroidWeights_eq_one_of_card_eq_add_one k h) p


/-- A weighted sum, as an affine map on the points involved. -/
def weightedVSubOfPoint (w : ι → k) : (ι → P) × P →ᵃ[k] V where
  toFun p := s.weightedVSubOfPoint p.fst p.snd w
  linear := ∑ i ∈ s, w i • ((LinearMap.proj i).comp (LinearMap.fst _ _ _) - LinearMap.snd _ _ _)
  map_vadd' := by
    /-
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : CommRing k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      ι : Type u_4
      s : Finset ι
      w : ι → k
      ⊢ ∀ (p : Prod (ι → P) P) (v : Prod (ι → V) V), Eq ((fun p => (s.weightedVSubOf …
    -/
    rintro ⟨p, b⟩ ⟨v, b'⟩
    -- Porting note: needed to give `Prod.mk_vadd_mk` a hint
    simp [LinearMap.sum_apply, Finset.weightedVSubOfPoint, vsub_vadd_eq_vsub_sub,
     vadd_vsub_assoc,
     add_sub, ← sub_add_eq_add_sub, smul_add, Finset.sum_add_distrib, Prod.mk_vadd_mk v]


