@[to_additive]
lemma norm_mul_le_max (x y : S) :
    ‖x * y‖ ≤ max ‖x‖ ‖y‖ := by
  simpa only [le_max_iff, dist_eq_norm_div, div_inv_eq_mul, div_one, one_mul] using
    dist_triangle_max x 1 y⁻¹


@[to_additive]
lemma isUltrametricDist_of_forall_norm_mul_le_max_norm
    (h : ∀ x y : S', ‖x * y‖ ≤ max ‖x‖ ‖y‖) : IsUltrametricDist S' where
  dist_triangle_max x y z := by
    /-
      S' : Type u_2
      inst✝ : SeminormedGroup S'
      h : ∀ (x y : S'), LE.le (Norm.norm (HMul.hMul x y)) (Max.max (Norm.norm x) (No …
      x y z : S'
      ⊢ LE.le (Dist.dist x z) (Max.max (Dist.dist x y) (Dist.dist y z))
    -/
    simpa only [dist_eq_norm_div, le_max_iff, div_mul_div_cancel] using h (x / y) (y / z)
    /-
      🎉 no goals
    -/


lemma isUltrametricDist_of_isNonarchimedean_norm {S' : Type*} [SeminormedAddGroup S']
    (h : IsNonarchimedean (norm : S' → ℝ)) : IsUltrametricDist S' :=
  isUltrametricDist_of_forall_norm_add_le_max_norm h


lemma isNonarchimedean_norm {R} [SeminormedAddCommGroup R] [IsUltrametricDist R] :
    IsNonarchimedean (‖·‖ : R → ℝ) := by
  /-
    R : Type u_4
    inst✝¹ : SeminormedAddCommGroup R
    inst✝ : IsUltrametricDist R
    ⊢ IsNonarchimedean fun x => Norm.norm x
  -/
  intro x y
  /-
    R : Type u_4
    inst✝¹ : SeminormedAddCommGroup R
    inst✝ : IsUltrametricDist R
    x y : R
    ⊢ LE.le ((fun x => Norm.norm x) (HAdd.hAdd x y)) (Max.max ((fun x => Norm.norm …
  -/
  convert dist_triangle_max 0 x (x + y) using 1
    /-
      case h.e'_3
      R : Type u_4
      inst✝¹ : SeminormedAddCommGroup R
      inst✝ : IsUltrametricDist R
      x y : R
      ⊢ Eq ((fun x => Norm.norm x) (HAdd.hAdd x y)) (Dist.dist 0 (HAdd.hAdd x y))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4
      R : Type u_4
      inst✝¹ : SeminormedAddCommGroup R
      inst✝ : IsUltrametricDist R
      x y : R
      ⊢ Eq (Max.max ((fun x => Norm.norm x) x) ((fun x => Norm.norm x) y)) (Max.max  …
    -/
              /-
                🎉 no goals
              -/
  · congr <;> simp [SeminormedAddGroup.dist_eq]
              /-
                🎉 no goals
              -/


lemma isUltrametricDist_iff_isNonarchimedean_norm {R} [SeminormedAddCommGroup R] :
    IsUltrametricDist R ↔ IsNonarchimedean (‖·‖ : R → ℝ) :=
  ⟨fun h => h.isNonarchimedean_norm, IsUltrametricDist.isUltrametricDist_of_isNonarchimedean_norm⟩


@[to_additive]
lemma nnnorm_mul_le_max (x y : S) :
    ‖x * y‖₊ ≤ max ‖x‖₊ ‖y‖₊ :=
  norm_mul_le_max _ _


@[to_additive]
lemma isUltrametricDist_of_forall_nnnorm_mul_le_max_nnnorm
    (h : ∀ x y : S', ‖x * y‖₊ ≤ max ‖x‖₊ ‖y‖₊) : IsUltrametricDist S' :=
  isUltrametricDist_of_forall_norm_mul_le_max_norm h


lemma isUltrametricDist_of_isNonarchimedean_nnnorm {S' : Type*} [SeminormedAddGroup S']
    (h : IsNonarchimedean ((↑) ∘ (nnnorm : S' → ℝ≥0))) : IsUltrametricDist S' :=
  isUltrametricDist_of_forall_nnnorm_add_le_max_nnnorm h


lemma isNonarchimedean_nnnorm {R} [SeminormedAddCommGroup R] [IsUltrametricDist R] :
    IsNonarchimedean (‖·‖₊ : R → ℝ) := by
  /-
    R : Type u_4
    inst✝¹ : SeminormedAddCommGroup R
    inst✝ : IsUltrametricDist R
    ⊢ IsNonarchimedean fun x => ↑(NNNorm.nnnorm x)
  -/
  intro x y
  /-
    R : Type u_4
    inst✝¹ : SeminormedAddCommGroup R
    inst✝ : IsUltrametricDist R
    x y : R
    ⊢ LE.le ((fun x => ↑(NNNorm.nnnorm x)) (HAdd.hAdd x y)) (Max.max ((fun x => ↑( …
  -/
  convert dist_triangle_max 0 x (x + y) using 1
    /-
      case h.e'_3
      R : Type u_4
      inst✝¹ : SeminormedAddCommGroup R
      inst✝ : IsUltrametricDist R
      x y : R
      ⊢ Eq ((fun x => ↑(NNNorm.nnnorm x)) (HAdd.hAdd x y)) (Dist.dist 0 (HAdd.hAdd x …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4
      R : Type u_4
      inst✝¹ : SeminormedAddCommGroup R
      inst✝ : IsUltrametricDist R
      x y : R
      ⊢ Eq (Max.max ((fun x => ↑(NNNorm.nnnorm x)) x) ((fun x => ↑(NNNorm.nnnorm x)) …
    -/
              /-
                🎉 no goals
              -/
  · congr <;> simp [SeminormedAddGroup.dist_eq]
              /-
                🎉 no goals
              -/


lemma isUltrametricDist_iff_isNonarchimedean_nnnorm {R} [SeminormedAddCommGroup R] :
    IsUltrametricDist R ↔ IsNonarchimedean (‖·‖₊ : R → ℝ) :=
  ⟨fun h => h.isNonarchimedean_norm, IsUltrametricDist.isUltrametricDist_of_isNonarchimedean_norm⟩


/-- All triangles are isosceles in an ultrametric normed group. -/
@[to_additive "All triangles are isosceles in an ultrametric normed additive group."]
lemma norm_mul_eq_max_of_norm_ne_norm
    {x y : S} (h : ‖x‖ ≠ ‖y‖) : ‖x * y‖ = max ‖x‖ ‖y‖ := by
  /-
    S : Type u_1
    inst✝¹ : SeminormedGroup S
    inst✝ : IsUltrametricDist S
    x y : S
    h : Ne (Norm.norm x) (Norm.norm y)
    ⊢ Eq (Norm.norm (HMul.hMul x y)) (Max.max (Norm.norm x) (Norm.norm y))
  -/
  rw [← div_inv_eq_mul, ← dist_eq_norm_div, dist_eq_max_of_dist_ne_dist _ 1 _ (by simp [h])]
  /-
    S : Type u_1
    inst✝¹ : SeminormedGroup S
    inst✝ : IsUltrametricDist S
    x y : S
    h : Ne (Norm.norm x) (Norm.norm y)
    ⊢ Eq (Max.max (Dist.dist x 1) (Dist.dist 1 (Inv.inv y))) (Max.max (Norm.norm x …
  -/
  simp only [dist_one_right, dist_one_left, norm_inv']
  /-
    🎉 no goals
  -/


@[to_additive]
lemma norm_eq_of_mul_norm_lt_max {x y : S} (h : ‖x * y‖ < max ‖x‖ ‖y‖) :
    ‖x‖ = ‖y‖ :=
  not_ne_iff.mp (h.ne ∘ norm_mul_eq_max_of_norm_ne_norm)


/-- All triangles are isosceles in an ultrametric normed group. -/
@[to_additive "All triangles are isosceles in an ultrametric normed additive group."]
lemma nnnorm_mul_eq_max_of_nnnorm_ne_nnnorm
    {x y : S} (h : ‖x‖₊ ≠ ‖y‖₊) : ‖x * y‖₊ = max ‖x‖₊ ‖y‖₊ := by
  simpa only [← NNReal.coe_inj, NNReal.coe_max] using
    norm_mul_eq_max_of_norm_ne_norm (NNReal.coe_injective.ne h)


@[to_additive]
lemma nnnorm_eq_of_mul_nnnorm_lt_max {x y : S} (h : ‖x * y‖₊ < max ‖x‖₊ ‖y‖₊) :
    ‖x‖₊ = ‖y‖₊ :=
  not_ne_iff.mp (h.ne ∘ nnnorm_mul_eq_max_of_nnnorm_ne_nnnorm)


/-- All triangles are isosceles in an ultrametric normed group. -/
@[to_additive "All triangles are isosceles in an ultrametric normed additive group."]
lemma norm_div_eq_max_of_norm_div_ne_norm_div (x y z : S) (h : ‖x / y‖ ≠ ‖y / z‖) :
    ‖x / z‖ = max ‖x / y‖ ‖y / z‖ := by
  /-
    S : Type u_1
    inst✝¹ : SeminormedGroup S
    inst✝ : IsUltrametricDist S
    x y z : S
    h : Ne (Norm.norm (HDiv.hDiv x y)) (Norm.norm (HDiv.hDiv y z))
    ⊢ Eq (Norm.norm (HDiv.hDiv x z)) (Max.max (Norm.norm (HDiv.hDiv x y)) (Norm.no …
  -/
  simpa only [div_mul_div_cancel] using norm_mul_eq_max_of_norm_ne_norm h
  /-
    🎉 no goals
  -/


/-- All triangles are isosceles in an ultrametric normed group. -/
@[to_additive "All triangles are isosceles in an ultrametric normed additive group."]
lemma nnnorm_div_eq_max_of_nnnorm_div_ne_nnnorm_div (x y z : S) (h : ‖x / y‖₊ ≠ ‖y / z‖₊) :
    ‖x / z‖₊ = max ‖x / y‖₊ ‖y / z‖₊ := by
  simpa only [← NNReal.coe_inj, NNReal.coe_max] using
    norm_div_eq_max_of_norm_div_ne_norm_div _ _ _ (NNReal.coe_injective.ne h)


@[to_additive]
lemma nnnorm_pow_le (x : S) (n : ℕ) :
    ‖x ^ n‖₊ ≤ ‖x‖₊ := by
  induction n with
  | zero => simp
  | succ n hn => simpa [pow_add, hn] using nnnorm_mul_le_max (x ^ n) x


@[to_additive]
lemma norm_pow_le (x : S) (n : ℕ) :
    ‖x ^ n‖ ≤ ‖x‖ :=
  nnnorm_pow_le x n


@[to_additive]
lemma nnnorm_zpow_le (x : S) (z : ℤ) :
    ‖x ^ z‖₊ ≤ ‖x‖₊ := by
  /-
    S : Type u_1
    inst✝¹ : SeminormedGroup S
    inst✝ : IsUltrametricDist S
    x : S
    z : Int
    ⊢ LE.le (NNNorm.nnnorm (HPow.hPow x z)) (NNNorm.nnnorm x)
  -/
  cases z <;>
  /-
    case ofNat
    S : Type u_1
    inst✝¹ : SeminormedGroup S
    inst✝ : IsUltrametricDist S
    x : S
    a✝ : Nat
    ⊢ LE.le (NNNorm.nnnorm (HPow.hPow x (Int.ofNat a✝))) (NNNorm.nnnorm x)
  -/
  /-
    🎉 no goals
  -/
  simpa using nnnorm_pow_le _ _
  /-
    🎉 no goals
  -/


@[to_additive]
lemma norm_zpow_le (x : S) (z : ℤ) :
    ‖x ^ z‖ ≤ ‖x‖ :=
  nnnorm_zpow_le x z


/--
In a group with an ultrametric norm, open balls around 1 of positive radius are open subgroups.
-/
@[to_additive "In an additive group with an ultrametric norm, open balls around 0 of
positive radius are open subgroups."]
def ball_openSubgroup {r : ℝ} (hr : 0 < r) : OpenSubgroup S where
  carrier := Metric.ball (1 : S) r
  mul_mem' {x} {y} hx hy := by
    /-
      S : Type u_1
      S' : Type u_2
      ι : Type u_3
      inst✝² : SeminormedGroup S
      inst✝¹ : SeminormedGroup S'
      inst✝ : IsUltrametricDist S
      r : Real
      hr : LT.lt 0 r
      x y : S
      hx : Membership.mem (Metric.ball 1 r) x
      hy : Membership.mem (Metric.ball 1 r) y
      ⊢ Membership.mem (Metric.ball 1 r) (HMul.hMul x y)
    -/
    simp only [Metric.mem_ball, dist_eq_norm_div, div_one] at hx hy ⊢
    /-
      S : Type u_1
      S' : Type u_2
      ι : Type u_3
      inst✝² : SeminormedGroup S
      inst✝¹ : SeminormedGroup S'
      inst✝ : IsUltrametricDist S
      r : Real
      hr : LT.lt 0 r
      x y : S
      hx : LT.lt (Norm.norm x) r
      hy : LT.lt (Norm.norm y) r
      ⊢ LT.lt (Norm.norm (HMul.hMul x y)) r
    -/
    exact (norm_mul_le_max x y).trans_lt (max_lt hx hy)
    /-
      🎉 no goals
    -/
  one_mem' := Metric.mem_ball_self hr
                 /-
                   S : Type u_1
                   S' : Type u_2
                   ι : Type u_3
                   inst✝² : SeminormedGroup S
                   inst✝¹ : SeminormedGroup S'
                   inst✝ : IsUltrametricDist S
                   r : Real
                   hr : LT.lt 0 r
                   ⊢ ∀ {x : S}, Membership.mem { carrier := Metric.ball 1 r, mul_mem' := ⋯, one_m …
                 -/
  inv_mem' := by simp only [Metric.mem_ball, dist_one_right, norm_inv', imp_self, implies_true]
                 /-
                   🎉 no goals
                 -/
  isOpen' := Metric.isOpen_ball


/--
In a group with an ultrametric norm, closed balls around 1 of positive radius are open subgroups.
-/
@[to_additive "In an additive group with an ultrametric norm, closed balls around 0 of positive
radius are open subgroups."]
def closedBall_openSubgroup {r : ℝ} (hr : 0 < r) : OpenSubgroup S where
  carrier := Metric.closedBall (1 : S) r
  mul_mem' {x} {y} hx hy := by
    /-
      S : Type u_1
      S' : Type u_2
      ι : Type u_3
      inst✝² : SeminormedGroup S
      inst✝¹ : SeminormedGroup S'
      inst✝ : IsUltrametricDist S
      r : Real
      hr : LT.lt 0 r
      x y : S
      hx : Membership.mem (Metric.closedBall 1 r) x
      hy : Membership.mem (Metric.closedBall 1 r) y
      ⊢ Membership.mem (Metric.closedBall 1 r) (HMul.hMul x y)
    -/
    simp only [Metric.mem_closedBall, dist_eq_norm_div, div_one] at hx hy ⊢
    /-
      S : Type u_1
      S' : Type u_2
      ι : Type u_3
      inst✝² : SeminormedGroup S
      inst✝¹ : SeminormedGroup S'
      inst✝ : IsUltrametricDist S
      r : Real
      hr : LT.lt 0 r
      x y : S
      hx : LE.le (Norm.norm x) r
      hy : LE.le (Norm.norm y) r
      ⊢ LE.le (Norm.norm (HMul.hMul x y)) r
    -/
    exact (norm_mul_le_max x y).trans (max_le hx hy)
    /-
      🎉 no goals
    -/
  one_mem' := Metric.mem_closedBall_self hr.le
                 /-
                   S : Type u_1
                   S' : Type u_2
                   ι : Type u_3
                   inst✝² : SeminormedGroup S
                   inst✝¹ : SeminormedGroup S'
                   inst✝ : IsUltrametricDist S
                   r : Real
                   hr : LT.lt 0 r
                   ⊢ ∀ {x : S}, Membership.mem { carrier := Metric.closedBall 1 r, mul_mem' := ⋯, …
                 -/
  inv_mem' := by simp only [mem_closedBall, dist_one_right, norm_inv', imp_self, implies_true]
                 /-
                   🎉 no goals
                 -/
  isOpen' := IsUltrametricDist.isOpen_closedBall _ hr.ne'


/-- A commutative group with an ultrametric group seminorm is nonarchimedean (as a topological
group, i.e. every neighborhood of 1 contains an open subgroup). -/
@[to_additive "A commutative additive group with an ultrametric group seminorm is nonarchimedean
(as a topological group, i.e. every neighborhood of 0 contains an open subgroup)."]
instance nonarchimedeanGroup : NonarchimedeanGroup M where
  is_nonarchimedean := by simpa only [Metric.mem_nhds_iff]
    using fun U ⟨ε, hεp, hεU⟩ ↦ ⟨ball_openSubgroup M hεp, hεU⟩


/-- Nonarchimedean norm of a product is less than or equal the norm of any term in the product.
This version is phrased using `Finset.sup'` and `Finset.Nonempty` due to `Finset.sup`
operating over an `OrderBot`, which `ℝ` is not.
-/
@[to_additive "Nonarchimedean norm of a sum is less than or equal the norm of any term in the sum.
This version is phrased using `Finset.sup'` and `Finset.Nonempty` due to `Finset.sup`
operating over an `OrderBot`, which `ℝ` is not. "]
lemma _root_.Finset.Nonempty.norm_prod_le_sup'_norm {s : Finset ι} (hs : s.Nonempty) (f : ι → M) :
    ‖∏ i ∈ s, f i‖ ≤ s.sup' hs (‖f ·‖) := by
  /-
    M : Type u_1
    ι : Type u_2
    inst✝¹ : SeminormedCommGroup M
    inst✝ : IsUltrametricDist M
    s : Finset ι
    hs : s.Nonempty
    f : ι → M
    ⊢ LE.le (Norm.norm (s.prod fun i => f i)) (s.sup' hs fun x => Norm.norm (f x))
  -/
  simp only [Finset.le_sup'_iff]
  induction hs using Finset.Nonempty.cons_induction with
  | singleton j => simp only [Finset.mem_singleton, Finset.prod_singleton, exists_eq_left, le_refl]
  | cons j t hj _ IH =>
      simp only [Finset.prod_cons, Finset.mem_cons, exists_eq_or_imp]
      refine (le_total ‖∏ i ∈ t, f i‖ ‖f j‖).imp ?_ ?_ <;> intro h
      · exact (norm_mul_le_max _ _).trans (max_eq_left h).le
      · exact ⟨_, IH.choose_spec.left, (norm_mul_le_max _ _).trans <|
          ((max_eq_right h).le.trans IH.choose_spec.right)⟩


/-- Nonarchimedean norm of a product is less than or equal to the largest norm of a term in the
product. -/
@[to_additive "Nonarchimedean norm of a sum is less than or equal to the largest norm of a term in
the sum."]
lemma _root_.Finset.nnnorm_prod_le_sup_nnnorm (s : Finset ι) (f : ι → M) :
    ‖∏ i ∈ s, f i‖₊ ≤ s.sup (‖f ·‖₊) := by
  /-
    M : Type u_1
    ι : Type u_2
    inst✝¹ : SeminormedCommGroup M
    inst✝ : IsUltrametricDist M
    s : Finset ι
    f : ι → M
    ⊢ LE.le (NNNorm.nnnorm (s.prod fun i => f i)) (s.sup fun x => NNNorm.nnnorm (f …
  -/
  rcases s.eq_empty_or_nonempty with rfl|hs
    /-
      case inl
      M : Type u_1
      ι : Type u_2
      inst✝¹ : SeminormedCommGroup M
      inst✝ : IsUltrametricDist M
      f : ι → M
      ⊢ LE.le (NNNorm.nnnorm (EmptyCollection.emptyCollection.prod fun i => f i)) (E …
    -/
  · simp only [Finset.prod_empty, nnnorm_one', Finset.sup_empty, bot_eq_zero', le_refl]
    /-
      🎉 no goals
    -/
  · simpa only [← Finset.sup'_eq_sup hs, Finset.le_sup'_iff, coe_le_coe, coe_nnnorm']
      using hs.norm_prod_le_sup'_norm f


/--
Generalised ultrametric triangle inequality for finite products in commutative groups with
an ultrametric norm.
-/
@[to_additive "Generalised ultrametric triangle inequality for finite sums in additive commutative
groups with an ultrametric norm."]
lemma nnnorm_prod_le_of_forall_le {s : Finset ι} {f : ι → M} {C : ℝ≥0}
    (hC : ∀ i ∈ s, ‖f i‖₊ ≤ C) : ‖∏ i ∈ s, f i‖₊ ≤ C :=
  (s.nnnorm_prod_le_sup_nnnorm f).trans <| Finset.sup_le hC


/--
Generalised ultrametric triangle inequality for nonempty finite products in commutative groups with
an ultrametric norm.
-/
@[to_additive "Generalised ultrametric triangle inequality for nonempty finite sums in additive
commutative groups with an ultrametric norm."]
lemma norm_prod_le_of_forall_le_of_nonempty {s : Finset ι} (hs : s.Nonempty) {f : ι → M} {C : ℝ}
    (hC : ∀ i ∈ s, ‖f i‖ ≤ C) : ‖∏ i ∈ s, f i‖ ≤ C :=
  (hs.norm_prod_le_sup'_norm f).trans (Finset.sup'_le hs _ hC)


/--
Generalised ultrametric triangle inequality for finite products in commutative groups with
an ultrametric norm.
-/
@[to_additive "Generalised ultrametric triangle inequality for finite sums in additive commutative
groups with an ultrametric norm."]
lemma norm_prod_le_of_forall_le_of_nonneg {s : Finset ι} {f : ι → M} {C : ℝ}
    (h_nonneg : 0 ≤ C) (hC : ∀ i ∈ s, ‖f i‖ ≤ C) : ‖∏ i ∈ s, f i‖ ≤ C := by
  /-
    M : Type u_1
    ι : Type u_2
    inst✝¹ : SeminormedCommGroup M
    inst✝ : IsUltrametricDist M
    s : Finset ι
    f : ι → M
    C : Real
    h_nonneg : LE.le 0 C
    hC : ∀ (i : ι), Membership.mem s i → LE.le (Norm.norm (f i)) C
    ⊢ LE.le (Norm.norm (s.prod fun i => f i)) C
  -/
  lift C to NNReal using h_nonneg
  /-
    case intro
    M : Type u_1
    ι : Type u_2
    inst✝¹ : SeminormedCommGroup M
    inst✝ : IsUltrametricDist M
    s : Finset ι
    f : ι → M
    C : NNReal
    hC : ∀ (i : ι), Membership.mem s i → LE.le (Norm.norm (f i)) ↑C
    ⊢ LE.le (Norm.norm (s.prod fun i => f i)) ↑C
  -/
  exact nnnorm_prod_le_of_forall_le hC
  /-
    🎉 no goals
  -/


/--
Given a function `f : ι → M` and a nonempty finite set `t ⊆ ι`, we can always find `i ∈ t` such that
`‖∏ j in t, f j‖ ≤ ‖f i‖`.
-/
@[to_additive "Given a function `f : ι → M` and a nonempty finite set `t ⊆ ι`, we can always find
`i ∈ t` such that `‖∑ j in t, f j‖ ≤ ‖f i‖`."]
theorem exists_norm_finset_prod_le_of_nonempty {t : Finset ι} (ht : t.Nonempty) (f : ι → M) :
    ∃ i ∈ t, ‖∏ j in t, f j‖ ≤ ‖f i‖ :=
  match t.exists_mem_eq_sup' ht (‖f ·‖) with
  |⟨j, hj, hj'⟩ => ⟨j, hj, (ht.norm_prod_le_sup'_norm f).trans (le_of_eq hj')⟩


/--
Given a function `f : ι → M` and a finite set `t ⊆ ι`, we can always find `i : ι`, belonging to `t`
if `t` is nonempty, such that `‖∏ j in t, f j‖ ≤ ‖f i‖`.
-/
@[to_additive "Given a function `f : ι → M` and a finite set `t ⊆ ι`, we can always find `i : ι`,
belonging to `t` if `t` is nonempty, such that `‖∑ j in t, f j‖ ≤ ‖f i‖`."]
theorem exists_norm_finset_prod_le (t : Finset ι) [Nonempty ι] (f : ι → M) :
    ∃ i : ι, (t.Nonempty → i ∈ t) ∧ ‖∏ j in t, f j‖ ≤ ‖f i‖ := by
  /-
    M : Type u_1
    ι : Type u_2
    inst✝² : SeminormedCommGroup M
    inst✝¹ : IsUltrametricDist M
    t : Finset ι
    inst✝ : Nonempty ι
    f : ι → M
    ⊢ Exists fun i => And (t.Nonempty → Membership.mem t i) (LE.le (Norm.norm (t.p …
  -/
  rcases t.eq_empty_or_nonempty with rfl | ht
    /-
      case inl
      M : Type u_1
      ι : Type u_2
      inst✝² : SeminormedCommGroup M
      inst✝¹ : IsUltrametricDist M
      inst✝ : Nonempty ι
      f : ι → M
      ⊢ Exists fun i => And (EmptyCollection.emptyCollection.Nonempty → Membership.m …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    M : Type u_1
    ι : Type u_2
    inst✝² : SeminormedCommGroup M
    inst✝¹ : IsUltrametricDist M
    t : Finset ι
    inst✝ : Nonempty ι
    f : ι → M
    ht : t.Nonempty
    ⊢ Exists fun i => And (t.Nonempty → Membership.mem t i) (LE.le (Norm.norm (t.p …
  -/
  exact (fun ⟨i, h, h'⟩ => ⟨i, fun _ ↦ h, h'⟩) <| exists_norm_finset_prod_le_of_nonempty ht f
  /-
    🎉 no goals
  -/


/--
Given a function `f : ι → M` and a multiset `t : Multiset ι`, we can always find `i : ι`, belonging
to `t` if `t` is nonempty, such that `‖(s.map f).prod‖ ≤ ‖f i‖`.
-/
@[to_additive "Given a function `f : ι → M` and a multiset `t : Multiset ι`, we can always find
`i : ι`, belonging to `t` if `t` is nonempty, such that `‖(s.map f).sum‖ ≤ ‖f i‖`."]
theorem exists_norm_multiset_prod_le (s : Multiset ι) [Nonempty ι] {f : ι → M} :
    ∃ i : ι, (s ≠ 0 → i ∈ s) ∧ ‖(s.map f).prod‖ ≤ ‖f i‖ := by
  /-
    M : Type u_1
    ι : Type u_2
    inst✝² : SeminormedCommGroup M
    inst✝¹ : IsUltrametricDist M
    s : Multiset ι
    inst✝ : Nonempty ι
    f : ι → M
    ⊢ Exists fun i => And (Ne s 0 → Membership.mem s i) (LE.le (Norm.norm (Multise …
  -/
  inhabit ι
  induction s using Multiset.induction_on with
  | empty => simp
  | @cons a t hM =>
      obtain ⟨M, hMs, hM⟩ := hM
      by_cases hMa : ‖f M‖ ≤ ‖f a‖
      · refine ⟨a, by simp, ?_⟩
        · rw [Multiset.map_cons, Multiset.prod_cons]
          exact le_trans (norm_mul_le_max _ _) (max_le (le_refl _) (le_trans hM hMa))
      · rw [not_le] at hMa
        rcases eq_or_ne t 0 with rfl|ht
        · exact ⟨a, by simp, by simp⟩
        · refine ⟨M, ?_, ?_⟩
          · simp [hMs ht]
          rw [Multiset.map_cons, Multiset.prod_cons]
          exact le_trans (norm_mul_le_max _ _) (max_le hMa.le hM)


@[to_additive]
lemma norm_tprod_le (f : ι → M) : ‖∏' i, f i‖ ≤ ⨆ i, ‖f i‖ := by
  /-
    M : Type u_1
    ι : Type u_2
    inst✝¹ : SeminormedCommGroup M
    inst✝ : IsUltrametricDist M
    f : ι → M
    ⊢ LE.le (Norm.norm (tprod fun i => f i)) (iSup fun i => Norm.norm (f i))
  -/
  rcases isEmpty_or_nonempty ι with hι | hι
  · -- Silly case #1 : the index type is empty
    /-
      case inl
      M : Type u_1
      ι : Type u_2
      inst✝¹ : SeminormedCommGroup M
      inst✝ : IsUltrametricDist M
      f : ι → M
      hι : IsEmpty ι
      ⊢ LE.le (Norm.norm (tprod fun i => f i)) (iSup fun i => Norm.norm (f i))
    -/
    simp only [tprod_empty, norm_one', Real.iSup_of_isEmpty, le_refl]
    /-
      🎉 no goals
    -/
  /-
    case inr
    M : Type u_1
    ι : Type u_2
    inst✝¹ : SeminormedCommGroup M
    inst✝ : IsUltrametricDist M
    f : ι → M
    hι : Nonempty ι
    ⊢ LE.le (Norm.norm (tprod fun i => f i)) (iSup fun i => Norm.norm (f i))
  -/
  by_cases h : Multipliable f; swap
  · -- Silly case #2 : the product is divergent
    /-
      case neg
      M : Type u_1
      ι : Type u_2
      inst✝¹ : SeminormedCommGroup M
      inst✝ : IsUltrametricDist M
      f : ι → M
      hι : Nonempty ι
      h : Not (Multipliable f)
      ⊢ LE.le (Norm.norm (tprod fun i => f i)) (iSup fun i => Norm.norm (f i))
    -/
    rw [tprod_eq_one_of_not_multipliable h, norm_one']
    /-
      case neg
      M : Type u_1
      ι : Type u_2
      inst✝¹ : SeminormedCommGroup M
      inst✝ : IsUltrametricDist M
      f : ι → M
      hι : Nonempty ι
      h : Not (Multipliable f)
      ⊢ LE.le 0 (iSup fun i => Norm.norm (f i))
    -/
    by_cases h_bd : BddAbove (Set.range fun i ↦ ‖f i‖)
      /-
        case pos
        M : Type u_1
        ι : Type u_2
        inst✝¹ : SeminormedCommGroup M
        inst✝ : IsUltrametricDist M
        f : ι → M
        hι : Nonempty ι
        h : Not (Multipliable f)
        h_bd : BddAbove (Set.range fun i => Norm.norm (f i))
        ⊢ LE.le 0 (iSup fun i => Norm.norm (f i))
      -/
    · exact le_ciSup_of_le h_bd hι.some (norm_nonneg' _)
      /-
        🎉 no goals
      -/
      /-
        case neg
        M : Type u_1
        ι : Type u_2
        inst✝¹ : SeminormedCommGroup M
        inst✝ : IsUltrametricDist M
        f : ι → M
        hι : Nonempty ι
        h : Not (Multipliable f)
        h_bd : Not (BddAbove (Set.range fun i => Norm.norm (f i)))
        ⊢ LE.le 0 (iSup fun i => Norm.norm (f i))
      -/
    · rw [Real.iSup_of_not_bddAbove h_bd]
      /-
        🎉 no goals
      -/
  -- now the interesting case
  have h_bd : BddAbove (Set.range fun i ↦ ‖f i‖) :=
    h.tendsto_cofinite_one.norm'.bddAbove_range_of_cofinite
  /-
    case pos
    M : Type u_1
    ι : Type u_2
    inst✝¹ : SeminormedCommGroup M
    inst✝ : IsUltrametricDist M
    f : ι → M
    hι : Nonempty ι
    h : Multipliable f
    h_bd : BddAbove (Set.range fun i => Norm.norm (f i))
    ⊢ LE.le (Norm.norm (tprod fun i => f i)) (iSup fun i => Norm.norm (f i))
  -/
  refine le_of_tendsto' h.hasProd.norm' (fun s ↦ norm_prod_le_of_forall_le_of_nonneg ?_ ?_)
    /-
      case pos.refine_1
      M : Type u_1
      ι : Type u_2
      inst✝¹ : SeminormedCommGroup M
      inst✝ : IsUltrametricDist M
      f : ι → M
      hι : Nonempty ι
      h : Multipliable f
      h_bd : BddAbove (Set.range fun i => Norm.norm (f i))
      s : Finset ι
      ⊢ LE.le 0 (iSup fun i => Norm.norm (f i))
    -/
  · exact le_ciSup_of_le h_bd hι.some (norm_nonneg' _)
    /-
      🎉 no goals
    -/
    /-
      case pos.refine_2
      M : Type u_1
      ι : Type u_2
      inst✝¹ : SeminormedCommGroup M
      inst✝ : IsUltrametricDist M
      f : ι → M
      hι : Nonempty ι
      h : Multipliable f
      h_bd : BddAbove (Set.range fun i => Norm.norm (f i))
      s : Finset ι
      ⊢ ∀ (i : ι), Membership.mem s i → LE.le (Norm.norm (f i)) (iSup fun i => Norm. …
    -/
  · exact fun i _ ↦ le_ciSup h_bd i
    /-
      🎉 no goals
    -/


@[to_additive]
lemma nnnorm_tprod_le (f : ι → M) : ‖∏' i, f i‖₊ ≤ ⨆ i, ‖f i‖₊ := by
  /-
    M : Type u_1
    ι : Type u_2
    inst✝¹ : SeminormedCommGroup M
    inst✝ : IsUltrametricDist M
    f : ι → M
    ⊢ LE.le (NNNorm.nnnorm (tprod fun i => f i)) (iSup fun i => NNNorm.nnnorm (f i))
  -/
  simpa only [← NNReal.coe_le_coe, coe_nnnorm', coe_iSup] using norm_tprod_le f
  /-
    🎉 no goals
  -/


@[to_additive]
lemma norm_tprod_le_of_forall_le [Nonempty ι] {f : ι → M} {C : ℝ} (h : ∀ i, ‖f i‖ ≤ C) :
    ‖∏' i, f i‖ ≤ C :=
  (norm_tprod_le f).trans (ciSup_le h)


@[to_additive]
lemma norm_tprod_le_of_forall_le_of_nonneg {f : ι → M} {C : ℝ} (hC : 0 ≤ C) (h : ∀ i, ‖f i‖ ≤ C) :
    ‖∏' i, f i‖ ≤ C := by
  /-
    M : Type u_1
    ι : Type u_2
    inst✝¹ : SeminormedCommGroup M
    inst✝ : IsUltrametricDist M
    f : ι → M
    C : Real
    hC : LE.le 0 C
    h : ∀ (i : ι), LE.le (Norm.norm (f i)) C
    ⊢ LE.le (Norm.norm (tprod fun i => f i)) C
  -/
  rcases isEmpty_or_nonempty ι
    /-
      case inl
      M : Type u_1
      ι : Type u_2
      inst✝¹ : SeminormedCommGroup M
      inst✝ : IsUltrametricDist M
      f : ι → M
      C : Real
      hC : LE.le 0 C
      h : ∀ (i : ι), LE.le (Norm.norm (f i)) C
      h✝ : IsEmpty ι
      ⊢ LE.le (Norm.norm (tprod fun i => f i)) C
    -/
  · simpa only [tprod_empty, norm_one'] using hC
    /-
      🎉 no goals
    -/
    /-
      case inr
      M : Type u_1
      ι : Type u_2
      inst✝¹ : SeminormedCommGroup M
      inst✝ : IsUltrametricDist M
      f : ι → M
      C : Real
      hC : LE.le 0 C
      h : ∀ (i : ι), LE.le (Norm.norm (f i)) C
      h✝ : Nonempty ι
      ⊢ LE.le (Norm.norm (tprod fun i => f i)) C
    -/
  · exact norm_tprod_le_of_forall_le h
    /-
      🎉 no goals
    -/


@[to_additive]
lemma nnnorm_tprod_le_of_forall_le {f : ι → M} {C : ℝ≥0} (h : ∀ i, ‖f i‖₊ ≤ C) : ‖∏' i, f i‖₊ ≤ C :=
  (nnnorm_tprod_le f).trans (ciSup_le' h)


