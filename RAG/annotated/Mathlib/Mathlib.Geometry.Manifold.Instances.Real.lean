/-- The half-space in `ℝ^n`, used to model manifolds with boundary. We only define it when
`1 ≤ n`, as the definition only makes sense in this case.
-/
def EuclideanHalfSpace (n : ℕ) [NeZero n] : Type :=
  { x : EuclideanSpace ℝ (Fin n) // 0 ≤ x 0 }


/--
The quadrant in `ℝ^n`, used to model manifolds with corners, made of all vectors with nonnegative
coordinates.
-/
def EuclideanQuadrant (n : ℕ) : Type :=
  { x : EuclideanSpace ℝ (Fin n) // ∀ i : Fin n, 0 ≤ x i }


instance [NeZero n] : TopologicalSpace (EuclideanHalfSpace n) :=
  instTopologicalSpaceSubtype


instance : TopologicalSpace (EuclideanQuadrant n) :=
  instTopologicalSpaceSubtype


instance [NeZero n] : Inhabited (EuclideanHalfSpace n) :=
  ⟨⟨0, le_rfl⟩⟩


instance : Inhabited (EuclideanQuadrant n) :=
  ⟨⟨0, fun _ => le_rfl⟩⟩


@[ext]
theorem EuclideanQuadrant.ext (x y : EuclideanQuadrant n) (h : x.1 = y.1) : x = y :=
  Subtype.eq h


@[ext]
theorem EuclideanHalfSpace.ext [NeZero n] (x y : EuclideanHalfSpace n)
    (h : x.1 = y.1) : x = y :=
  Subtype.eq h


theorem EuclideanHalfSpace.convex [NeZero n] :
    Convex ℝ { x : EuclideanSpace ℝ (Fin n) | 0 ≤ x 0 } :=
                               /-
                                 n : Nat
                                 inst✝ : NeZero n
                                 x✝⁶ : EuclideanSpace Real (Fin n)
                                 hx : Membership.mem (setOf fun x => LE.le 0 (x 0)) x✝⁶
                                 x✝⁵ : EuclideanSpace Real (Fin n)
                                 hy : Membership.mem (setOf fun x => LE.le 0 (x 0)) x✝⁵
                                 x✝⁴ x✝³ : Real
                                 x✝² : LE.le 0 x✝⁴
                                 x✝¹ : LE.le 0 x✝³
                                 x✝ : Eq (HAdd.hAdd x✝⁴ x✝³) 1
                                 ⊢ Membership.mem (setOf fun x => LE.le 0 (x 0)) (HAdd.hAdd (HSMul.hSMul x✝⁴ x✝ …
                               -/
  fun _ hx _ hy _ _ _ _ _ ↦ by dsimp at hx hy ⊢; positivity
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem EuclideanQuadrant.convex :
    Convex ℝ { x : EuclideanSpace ℝ (Fin n) | ∀ i, 0 ≤ x i } :=
                                 /-
                                   n : Nat
                                   x✝⁶ : EuclideanSpace Real (Fin n)
                                   hx : Membership.mem (setOf fun x => ∀ (i : Fin n), LE.le 0 (x i)) x✝⁶
                                   x✝⁵ : EuclideanSpace Real (Fin n)
                                   hy : Membership.mem (setOf fun x => ∀ (i : Fin n), LE.le 0 (x i)) x✝⁵
                                   x✝⁴ x✝³ : Real
                                   x✝² : LE.le 0 x✝⁴
                                   x✝¹ : LE.le 0 x✝³
                                   x✝ : Eq (HAdd.hAdd x✝⁴ x✝³) 1
                                   i : Fin n
                                   ⊢ LE.le 0 (HAdd.hAdd (HSMul.hSMul x✝⁴ x✝⁶) (HSMul.hSMul x✝³ x✝⁵) i)
                                 -/
  fun _ hx _ hy _ _ _ _ _ i ↦ by dsimp at hx hy ⊢; specialize hx i; specialize hy i; positivity
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


instance EuclideanHalfSpace.pathConnectedSpace [NeZero n] :
    PathConnectedSpace (EuclideanHalfSpace n) :=
                                                                             /-
                                                                               n : Nat
                                                                               inst✝ : NeZero n
                                                                               ⊢ Membership.mem (setOf fun x => LE.le 0 (x 0)) 0
                                                                             -/
  isPathConnected_iff_pathConnectedSpace.mp <| convex.isPathConnected ⟨0, by simp⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


instance EuclideanQuadrant.pathConnectedSpace : PathConnectedSpace (EuclideanQuadrant n) :=
                                                                             /-
                                                                               n : Nat
                                                                               ⊢ Membership.mem (setOf fun x => ∀ (i : Fin n), LE.le 0 (x i)) 0
                                                                             -/
  isPathConnected_iff_pathConnectedSpace.mp <| convex.isPathConnected ⟨0, by simp⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


instance [NeZero n] : LocPathConnectedSpace (EuclideanHalfSpace n) :=
  EuclideanHalfSpace.convex.locPathConnectedSpace


instance : LocPathConnectedSpace (EuclideanQuadrant n) :=
  EuclideanQuadrant.convex.locPathConnectedSpace


theorem range_euclideanHalfSpace (n : ℕ) [NeZero n] :
    (range fun x : EuclideanHalfSpace n => x.val) = { y | 0 ≤ y 0 } :=
  Subtype.range_val

@[deprecated (since := "2024-04-05")] alias range_half_space := range_euclideanHalfSpace


open ENNReal in
@[simp]
theorem interior_halfSpace {n : ℕ} (p : ℝ≥0∞) (a : ℝ) (i : Fin n) :
    interior { y : PiLp p (fun _ : Fin n ↦ ℝ) | a ≤ y i } = { y | a < y i } := by
  /-
    n : Nat
    p : ENNReal
    a : Real
    i : Fin n
    ⊢ Eq (interior (setOf fun y => LE.le a (y i))) (setOf fun y => LT.lt a (y i))
  -/
  let f : (Π _ : Fin n, ℝ) →L[ℝ] ℝ := ContinuousLinearMap.proj i
  /-
    n : Nat
    p : ENNReal
    a : Real
    i : Fin n
    f : ContinuousLinearMap (RingHom.id Real) (Fin n → Real) Real := ContinuousLin …
    ⊢ Eq (interior (setOf fun y => LE.le a (y i))) (setOf fun y => LT.lt a (y i))
  -/
  change interior (f ⁻¹' Ici a) = f ⁻¹' Ioi a
  /-
    n : Nat
    p : ENNReal
    a : Real
    i : Fin n
    f : ContinuousLinearMap (RingHom.id Real) (Fin n → Real) Real := ContinuousLin …
    ⊢ Eq (interior (Set.preimage (⇑f) (Set.Ici a))) (Set.preimage (⇑f) (Set.Ioi a))
  -/
  rw [f.interior_preimage, interior_Ici]
  /-
    case hsurj
    n : Nat
    p : ENNReal
    a : Real
    i : Fin n
    f : ContinuousLinearMap (RingHom.id Real) (Fin n → Real) Real := ContinuousLin …
    ⊢ Function.Surjective ⇑f
  -/
  apply Function.surjective_eval
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-12")] alias interior_halfspace := interior_halfSpace


open ENNReal in
@[simp]
theorem closure_halfSpace {n : ℕ} (p : ℝ≥0∞) (a : ℝ) (i : Fin n) :
    closure { y : PiLp p (fun _ : Fin n ↦ ℝ) | a ≤ y i } = { y | a ≤ y i } := by
  /-
    n : Nat
    p : ENNReal
    a : Real
    i : Fin n
    ⊢ Eq (closure (setOf fun y => LE.le a (y i))) (setOf fun y => LE.le a (y i))
  -/
  let f : (Π _ : Fin n, ℝ) →L[ℝ] ℝ := ContinuousLinearMap.proj i
  /-
    n : Nat
    p : ENNReal
    a : Real
    i : Fin n
    f : ContinuousLinearMap (RingHom.id Real) (Fin n → Real) Real := ContinuousLin …
    ⊢ Eq (closure (setOf fun y => LE.le a (y i))) (setOf fun y => LE.le a (y i))
  -/
  change closure (f ⁻¹' Ici a) = f ⁻¹' Ici a
  /-
    n : Nat
    p : ENNReal
    a : Real
    i : Fin n
    f : ContinuousLinearMap (RingHom.id Real) (Fin n → Real) Real := ContinuousLin …
    ⊢ Eq (closure (Set.preimage (⇑f) (Set.Ici a))) (Set.preimage (⇑f) (Set.Ici a))
  -/
  rw [f.closure_preimage, closure_Ici]
  /-
    case hsurj
    n : Nat
    p : ENNReal
    a : Real
    i : Fin n
    f : ContinuousLinearMap (RingHom.id Real) (Fin n → Real) Real := ContinuousLin …
    ⊢ Function.Surjective ⇑f
  -/
  apply Function.surjective_eval
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-12")] alias closure_halfspace := closure_halfSpace


open ENNReal in
@[simp]
theorem closure_open_halfSpace {n : ℕ} (p : ℝ≥0∞) (a : ℝ) (i : Fin n) :
    closure { y : PiLp p (fun _ : Fin n ↦ ℝ) | a < y i } = { y | a ≤ y i } := by
  /-
    n : Nat
    p : ENNReal
    a : Real
    i : Fin n
    ⊢ Eq (closure (setOf fun y => LT.lt a (y i))) (setOf fun y => LE.le a (y i))
  -/
  let f : (Π _ : Fin n, ℝ) →L[ℝ] ℝ := ContinuousLinearMap.proj i
  /-
    n : Nat
    p : ENNReal
    a : Real
    i : Fin n
    f : ContinuousLinearMap (RingHom.id Real) (Fin n → Real) Real := ContinuousLin …
    ⊢ Eq (closure (setOf fun y => LT.lt a (y i))) (setOf fun y => LE.le a (y i))
  -/
  change closure (f ⁻¹' Ioi a) = f ⁻¹' Ici a
  /-
    n : Nat
    p : ENNReal
    a : Real
    i : Fin n
    f : ContinuousLinearMap (RingHom.id Real) (Fin n → Real) Real := ContinuousLin …
    ⊢ Eq (closure (Set.preimage (⇑f) (Set.Ioi a))) (Set.preimage (⇑f) (Set.Ici a))
  -/
  rw [f.closure_preimage, closure_Ioi]
  /-
    case hsurj
    n : Nat
    p : ENNReal
    a : Real
    i : Fin n
    f : ContinuousLinearMap (RingHom.id Real) (Fin n → Real) Real := ContinuousLin …
    ⊢ Function.Surjective ⇑f
  -/
  apply Function.surjective_eval
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-12")] alias closure_open_halfspace := closure_open_halfSpace


open ENNReal in
@[simp]
theorem frontier_halfSpace {n : ℕ} (p : ℝ≥0∞) (a : ℝ) (i : Fin n) :
    frontier { y : PiLp p (fun _ : Fin n ↦ ℝ) | a ≤ y i } = { y | a = y i } := by
  /-
    n : Nat
    p : ENNReal
    a : Real
    i : Fin n
    ⊢ Eq (frontier (setOf fun y => LE.le a (y i))) (setOf fun y => Eq a (y i))
  -/
  rw [frontier, closure_halfSpace, interior_halfSpace]
  /-
    n : Nat
    p : ENNReal
    a : Real
    i : Fin n
    ⊢ Eq (SDiff.sdiff (setOf fun y => LE.le a (y i)) (setOf fun y => LT.lt a (y i) …
  -/
  ext y
  /-
    case h
    n : Nat
    p : ENNReal
    a : Real
    i : Fin n
    y : PiLp p fun x => Real
    ⊢ Iff (Membership.mem (SDiff.sdiff (setOf fun y => LE.le a (y i)) (setOf fun y …
  -/
  simpa only [mem_diff, mem_setOf_eq, not_lt] using antisymm_iff
  /-
    🎉 no goals
  -/

@[deprecated (since := "2024-11-12")] alias frontier_halfspace := frontier_halfSpace


theorem range_euclideanQuadrant (n : ℕ) :
    (range fun x : EuclideanQuadrant n => x.val) = { y | ∀ i : Fin n, 0 ≤ y i } :=
  Subtype.range_val

@[deprecated (since := "2024-04-05")] alias range_quadrant := range_euclideanQuadrant


/--
Definition of the model with corners `(EuclideanSpace ℝ (Fin n), EuclideanHalfSpace n)`, used as
a model for manifolds with boundary. In the locale `Manifold`, use the shortcut `𝓡∂ n`.
-/
def modelWithCornersEuclideanHalfSpace (n : ℕ) [NeZero n] :
    ModelWithCorners ℝ (EuclideanSpace ℝ (Fin n)) (EuclideanHalfSpace n) where
  toFun := Subtype.val
                                            /-
                                              n : Nat
                                              inst✝ : NeZero n
                                              x : EuclideanSpace Real (Fin n)
                                              ⊢ LE.le 0 (Function.update x 0 (Max.max (x 0) 0) 0)
                                            -/
  invFun x := ⟨update x 0 (max (x 0) 0), by simp [le_refl]⟩
                                            /-
                                              🎉 no goals
                                            -/
  source := univ
  target := { x | 0 ≤ x 0 }
  map_source' x _ := x.property
  map_target' _ _ := mem_univ _
  left_inv' := fun ⟨xval, xprop⟩ _ => by
    /-
      n : Nat
      inst✝ : NeZero n
      x✝¹ : EuclideanHalfSpace n
      xval : EuclideanSpace Real (Fin n)
      xprop : LE.le 0 (xval 0)
      x✝ : Membership.mem Set.univ ⟨xval, xprop⟩
      ⊢ Eq ((fun x => ⟨Function.update x 0 (Max.max (x 0) 0), ⋯⟩) ↑⟨xval, xprop⟩) ⟨x …
    -/
    rw [Subtype.mk_eq_mk, update_eq_iff]
    /-
      n : Nat
      inst✝ : NeZero n
      x✝¹ : EuclideanHalfSpace n
      xval : EuclideanSpace Real (Fin n)
      xprop : LE.le 0 (xval 0)
      x✝ : Membership.mem Set.univ ⟨xval, xprop⟩
      ⊢ And (Eq (Max.max (↑⟨xval, xprop⟩ 0) 0) (xval 0)) (∀ (x : Fin n), Ne x 0 → Eq …
    -/
    exact ⟨max_eq_left xprop, fun i _ => rfl⟩
    /-
      🎉 no goals
    -/
  right_inv' _ hx := update_eq_iff.2 ⟨max_eq_left hx, fun _ _ => rfl⟩
  source_eq := rfl
  uniqueDiffOn' := by
    have : UniqueDiffOn ℝ _ :=
      UniqueDiffOn.pi (Fin n) (fun _ => ℝ) _ _ fun i (_ : i ∈ ({0} : Set (Fin n))) =>
        uniqueDiffOn_Ici 0
    /-
      n : Nat
      inst✝ : NeZero n
      this : UniqueDiffOn Real ((Singleton.singleton 0).pi fun i => Set.Ici 0)
      ⊢ UniqueDiffOn Real { toFun := Subtype.val, invFun := fun x => ⟨Function.updat …
    -/
    simpa only [singleton_pi] using this
    /-
      🎉 no goals
    -/
                                       /-
                                         n : Nat
                                         inst✝ : NeZero n
                                         ⊢ HasSubset.Subset { toFun := Subtype.val, invFun := fun x => ⟨Function.update …
                                       -/
  target_subset_closure_interior := by simp
                                       /-
                                         🎉 no goals
                                       -/
  continuous_toFun := continuous_subtype_val
  continuous_invFun := by
    /-
      n : Nat
      inst✝ : NeZero n
      ⊢ Continuous { toFun := Subtype.val, invFun := fun x => ⟨Function.update x 0 ( …
    -/
    exact (continuous_id.update 0 <| (continuous_apply 0).max continuous_const).subtype_mk _
    /-
      🎉 no goals
    -/


/--
Definition of the model with corners `(EuclideanSpace ℝ (Fin n), EuclideanQuadrant n)`, used as a
model for manifolds with corners -/
def modelWithCornersEuclideanQuadrant (n : ℕ) :
    ModelWithCorners ℝ (EuclideanSpace ℝ (Fin n)) (EuclideanQuadrant n) where
  toFun := Subtype.val
                                                 /-
                                                   n : Nat
                                                   x : EuclideanSpace Real (Fin n)
                                                   i : Fin n
                                                   ⊢ LE.le 0 ((fun i => Max.max (x i) 0) i)
                                                 -/
  invFun x := ⟨fun i => max (x i) 0, fun i => by simp only [le_refl, or_true, le_max_iff]⟩
                                                 /-
                                                   🎉 no goals
                                                 -/
  source := univ
  target := { x | ∀ i, 0 ≤ x i }
  map_source' x _ := x.property
  map_target' _ _ := mem_univ _
                      /-
                        n : Nat
                        x : EuclideanQuadrant n
                        x✝ : Membership.mem Set.univ x
                        ⊢ Eq ((fun x => ⟨fun i => Max.max (x i) 0, ⋯⟩) ↑x) x
                      -/
  left_inv' x _ := by ext i; simp only [Subtype.coe_mk, x.2 i, max_eq_left]
                             /-
                               🎉 no goals
                             -/
                        /-
                          n : Nat
                          x : EuclideanSpace Real (Fin n)
                          hx : Membership.mem (setOf fun x => ∀ (i : Fin n), LE.le 0 (x i)) x
                          ⊢ Eq (↑((fun x => ⟨fun i => Max.max (x i) 0, ⋯⟩) x)) x
                        -/
  right_inv' x hx := by ext1 i; simp only [hx i, max_eq_left]
                                /-
                                  🎉 no goals
                                -/
  source_eq := rfl
  uniqueDiffOn' := by
    have this : UniqueDiffOn ℝ _ :=
      UniqueDiffOn.univ_pi (Fin n) (fun _ => ℝ) _ fun _ => uniqueDiffOn_Ici 0
    /-
      n : Nat
      this : UniqueDiffOn Real (Set.univ.pi fun x => Set.Ici 0)
      ⊢ UniqueDiffOn Real { toFun := Subtype.val, invFun := fun x => ⟨fun i => Max.m …
    -/
    simpa only [pi_univ_Ici] using this
    /-
      🎉 no goals
    -/
  target_subset_closure_interior := by
    have : {x : EuclideanSpace ℝ (Fin n) | ∀ (i : Fin n), 0 ≤ x i}
      = Set.pi univ (fun i ↦ Ici 0) := by aesop
    /-
      n : Nat
      this : Eq (setOf fun x => ∀ (i : Fin n), LE.le 0 (x i)) (Set.univ.pi fun i =>  …
      ⊢ HasSubset.Subset { toFun := Subtype.val, invFun := fun x => ⟨fun i => Max.ma …
    -/
    simp only [this, interior_pi_set finite_univ]
    /-
      n : Nat
      this : Eq (setOf fun x => ∀ (i : Fin n), LE.le 0 (x i)) (Set.univ.pi fun i =>  …
      ⊢ HasSubset.Subset (Set.univ.pi fun i => Set.Ici 0) (closure (Set.univ.pi fun  …
    -/
    rw [closure_pi_set]
    /-
      n : Nat
      this : Eq (setOf fun x => ∀ (i : Fin n), LE.le 0 (x i)) (Set.univ.pi fun i =>  …
      ⊢ HasSubset.Subset (Set.univ.pi fun i => Set.Ici 0) (Set.univ.pi fun i => clos …
    -/
    simp
    /-
      🎉 no goals
    -/
  continuous_toFun := continuous_subtype_val
  continuous_invFun := Continuous.subtype_mk
    (continuous_pi fun i => (continuous_id.max continuous_const).comp (continuous_apply i)) _


/-- The model space used to define `n`-dimensional real manifolds without boundary. -/
scoped[Manifold]
  notation3 "𝓡 " n =>
    (modelWithCornersSelf ℝ (EuclideanSpace ℝ (Fin n)) :
      ModelWithCorners ℝ (EuclideanSpace ℝ (Fin n)) (EuclideanSpace ℝ (Fin n)))


/-- The model space used to define `n`-dimensional real manifolds with boundary. -/
scoped[Manifold]
  notation3 "𝓡∂ " n =>
    (modelWithCornersEuclideanHalfSpace n :
      ModelWithCorners ℝ (EuclideanSpace ℝ (Fin n)) (EuclideanHalfSpace n))


lemma range_modelWithCornersEuclideanHalfSpace (n : ℕ) [NeZero n] :
    range (𝓡∂ n) = { y | 0 ≤ y 0 } := range_euclideanHalfSpace n


lemma interior_range_modelWithCornersEuclideanHalfSpace (n : ℕ) [NeZero n] :
    interior (range (𝓡∂ n)) = { y | 0 < y 0 } := by
  calc interior (range (𝓡∂ n))
    _ = interior ({ y | 0 ≤ y 0}) := by
      congr!
      apply range_euclideanHalfSpace
    _ = { y | 0 < y 0 } := interior_halfSpace _ _ _


lemma frontier_range_modelWithCornersEuclideanHalfSpace (n : ℕ) [NeZero n] :
    frontier (range (𝓡∂ n)) = { y | 0 = y 0 } := by
  calc frontier (range (𝓡∂ n))
    _ = frontier ({ y | 0 ≤ y 0 }) := by
      congr!
      apply range_euclideanHalfSpace
    _ = { y | 0 = y 0 } := frontier_halfSpace 2 _ _


/-- The left chart for the topological space `[x, y]`, defined on `[x,y)` and sending `x` to `0` in
`EuclideanHalfSpace 1`.
-/
def IccLeftChart (x y : ℝ) [h : Fact (x < y)] :
    PartialHomeomorph (Icc x y) (EuclideanHalfSpace 1) where
  source := { z : Icc x y | z.val < y }
  target := { z : EuclideanHalfSpace 1 | z.val 0 < y - x }
  toFun := fun z : Icc x y => ⟨fun _ => z.val - x, sub_nonneg.mpr z.property.1⟩
                                       /-
                                         x y : Real
                                         h : Fact (LT.lt x y)
                                         z : EuclideanHalfSpace 1
                                         ⊢ Membership.mem (Set.Icc x y) (Min.min (HAdd.hAdd (↑z 0) x) y)
                                       -/
  invFun z := ⟨min (z.val 0 + x) y, by simp [le_refl, z.prop, le_of_lt h.out]⟩
                                       /-
                                         🎉 no goals
                                       -/
                    /-
                      x y : Real
                      h : Fact (LT.lt x y)
                      ⊢ ∀ ⦃x_1 : ↑(Set.Icc x y)⦄, Membership.mem (setOf fun z => LT.lt (↑z) y) x_1 → …
                    -/
  map_source' := by simp only [imp_self, sub_lt_sub_iff_right, mem_setOf_eq, forall_true_iff]
                    /-
                      🎉 no goals
                    -/
  map_target' := by
    /-
      x y : Real
      h : Fact (LT.lt x y)
      ⊢ ∀ ⦃x_1 : EuclideanHalfSpace 1⦄, Membership.mem (setOf fun z => LT.lt (↑z 0)  …
    -/
    simp only [min_lt_iff, mem_setOf_eq]; intro z hz; left
    /-
      case h
      x y : Real
      h : Fact (LT.lt x y)
      z : EuclideanHalfSpace 1
      hz : LT.lt (↑z 0) (HSub.hSub y x)
      ⊢ LT.lt (HAdd.hAdd (↑z 0) x) y
    -/
    linarith
    /-
      🎉 no goals
    -/
  left_inv' := by
    /-
      x y : Real
      h : Fact (LT.lt x y)
      ⊢ ∀ ⦃x_1 : ↑(Set.Icc x y)⦄, Membership.mem (setOf fun z => LT.lt (↑z) y) x_1 → …
    -/
    rintro ⟨z, hz⟩ h'z
    /-
      case mk
      x y : Real
      h : Fact (LT.lt x y)
      z : Real
      hz : Membership.mem (Set.Icc x y) z
      h'z : Membership.mem (setOf fun z => LT.lt (↑z) y) ⟨z, hz⟩
      ⊢ Eq ((fun z => ⟨Min.min (HAdd.hAdd (↑z 0) x) y, ⋯⟩) ((fun z => ⟨fun x_1 => HS …
    -/
    simp only [mem_setOf_eq, mem_Icc] at hz h'z
    /-
      case mk
      x y : Real
      h : Fact (LT.lt x y)
      z : Real
      hz✝ : Membership.mem (Set.Icc x y) z
      h'z : LT.lt z y
      hz : And (LE.le x z) (LE.le z y)
      ⊢ Eq ((fun z => ⟨Min.min (HAdd.hAdd (↑z 0) x) y, ⋯⟩) ((fun z => ⟨fun x_1 => HS …
    -/
    simp only [hz, min_eq_left, sub_add_cancel]
    /-
      🎉 no goals
    -/
  right_inv' := by
    /-
      x y : Real
      h : Fact (LT.lt x y)
      ⊢ ∀ ⦃x_1 : EuclideanHalfSpace 1⦄, Membership.mem (setOf fun z => LT.lt (↑z 0)  …
    -/
    rintro ⟨z, hz⟩ h'z
    /-
      case mk
      x y : Real
      h : Fact (LT.lt x y)
      z : EuclideanSpace Real (Fin 1)
      hz : LE.le 0 (z 0)
      h'z : Membership.mem (setOf fun z => LT.lt (↑z 0) (HSub.hSub y x)) ⟨z, hz⟩
      ⊢ Eq ((fun z => ⟨fun x_1 => HSub.hSub (↑z) x, ⋯⟩) ((fun z => ⟨Min.min (HAdd.hA …
    -/
    rw [Subtype.mk_eq_mk]
    /-
      case mk
      x y : Real
      h : Fact (LT.lt x y)
      z : EuclideanSpace Real (Fin 1)
      hz : LE.le 0 (z 0)
      h'z : Membership.mem (setOf fun z => LT.lt (↑z 0) (HSub.hSub y x)) ⟨z, hz⟩
      ⊢ Eq (fun x_1 => HSub.hSub (↑((fun z => ⟨Min.min (HAdd.hAdd (↑z 0) x) y, ⋯⟩) ⟨ …
    -/
    funext i
    /-
      case mk.h
      x y : Real
      h : Fact (LT.lt x y)
      z : EuclideanSpace Real (Fin 1)
      hz : LE.le 0 (z 0)
      h'z : Membership.mem (setOf fun z => LT.lt (↑z 0) (HSub.hSub y x)) ⟨z, hz⟩
      i : Fin 1
      ⊢ Eq (HSub.hSub (↑((fun z => ⟨Min.min (HAdd.hAdd (↑z 0) x) y, ⋯⟩) ⟨z, hz⟩)) x) …
    -/
    dsimp at hz h'z
    /-
      case mk.h
      x y : Real
      h : Fact (LT.lt x y)
      z : EuclideanSpace Real (Fin 1)
      hz : LE.le 0 (z 0)
      h'z : LT.lt (z 0) (HSub.hSub y x)
      i : Fin 1
      ⊢ Eq (HSub.hSub (↑((fun z => ⟨Min.min (HAdd.hAdd (↑z 0) x) y, ⋯⟩) ⟨z, hz⟩)) x) …
    -/
    have A : x + z 0 ≤ y := by linarith
    /-
      case mk.h
      x y : Real
      h : Fact (LT.lt x y)
      z : EuclideanSpace Real (Fin 1)
      hz : LE.le 0 (z 0)
      h'z : LT.lt (z 0) (HSub.hSub y x)
      i : Fin 1
      A : LE.le (HAdd.hAdd x (z 0)) y
      ⊢ Eq (HSub.hSub (↑((fun z => ⟨Min.min (HAdd.hAdd (↑z 0) x) y, ⋯⟩) ⟨z, hz⟩)) x) …
    -/
    rw [Subsingleton.elim i 0]
    /-
      case mk.h
      x y : Real
      h : Fact (LT.lt x y)
      z : EuclideanSpace Real (Fin 1)
      hz : LE.le 0 (z 0)
      h'z : LT.lt (z 0) (HSub.hSub y x)
      i : Fin 1
      A : LE.le (HAdd.hAdd x (z 0)) y
      ⊢ Eq (HSub.hSub (↑((fun z => ⟨Min.min (HAdd.hAdd (↑z 0) x) y, ⋯⟩) ⟨z, hz⟩)) x) …
    -/
    simp only [A, add_comm, add_sub_cancel_left, min_eq_left]
    /-
      🎉 no goals
    -/
  open_source :=
    haveI : IsOpen { z : ℝ | z < y } := isOpen_Iio
    this.preimage continuous_subtype_val
  open_target := by
    /-
      x y : Real
      h : Fact (LT.lt x y)
      ⊢ IsOpen { toFun := fun z => ⟨fun x_1 => HSub.hSub (↑z) x, ⋯⟩, invFun := fun z …
    -/
    have : IsOpen { z : ℝ | z < y - x } := isOpen_Iio
    have : IsOpen { z : EuclideanSpace ℝ (Fin 1) | z 0 < y - x } :=
      this.preimage (@continuous_apply (Fin 1) (fun _ => ℝ) _ 0)
    /-
      x y : Real
      h : Fact (LT.lt x y)
      this✝ : IsOpen (setOf fun z => LT.lt z (HSub.hSub y x))
      this : IsOpen (setOf fun z => LT.lt (z 0) (HSub.hSub y x))
      ⊢ IsOpen { toFun := fun z => ⟨fun x_1 => HSub.hSub (↑z) x, ⋯⟩, invFun := fun z …
    -/
    exact this.preimage continuous_subtype_val
    /-
      🎉 no goals
    -/
  continuousOn_toFun := by
    /-
      x y : Real
      h : Fact (LT.lt x y)
      ⊢ ContinuousOn ↑{ toFun := fun z => ⟨fun x_1 => HSub.hSub (↑z) x, ⋯⟩, invFun : …
    -/
    apply Continuous.continuousOn
    /-
      case h
      x y : Real
      h : Fact (LT.lt x y)
      ⊢ Continuous ↑{ toFun := fun z => ⟨fun x_1 => HSub.hSub (↑z) x, ⋯⟩, invFun :=  …
    -/
    apply Continuous.subtype_mk
    have : Continuous fun (z : ℝ) (_ : Fin 1) => z - x :=
      Continuous.sub (continuous_pi fun _ => continuous_id) continuous_const
    /-
      case h.h
      x y : Real
      h : Fact (LT.lt x y)
      this : Continuous fun z x_1 => HSub.hSub z x
      ⊢ Continuous fun x_1 x_2 => HSub.hSub (↑x_1) x
    -/
    exact this.comp continuous_subtype_val
    /-
      🎉 no goals
    -/
  continuousOn_invFun := by
    /-
      x y : Real
      h : Fact (LT.lt x y)
      ⊢ ContinuousOn { toFun := fun z => ⟨fun x_1 => HSub.hSub (↑z) x, ⋯⟩, invFun := …
    -/
    apply Continuous.continuousOn
    /-
      case h
      x y : Real
      h : Fact (LT.lt x y)
      ⊢ Continuous { toFun := fun z => ⟨fun x_1 => HSub.hSub (↑z) x, ⋯⟩, invFun := f …
    -/
    apply Continuous.subtype_mk
    have A : Continuous fun z : ℝ => min (z + x) y :=
      (continuous_id.add continuous_const).min continuous_const
    /-
      case h.h
      x y : Real
      h : Fact (LT.lt x y)
      A : Continuous fun z => Min.min (HAdd.hAdd z x) y
      ⊢ Continuous fun x_1 => Min.min (HAdd.hAdd (↑x_1 0) x) y
    -/
    have B : Continuous fun z : EuclideanSpace ℝ (Fin 1) => z 0 := continuous_apply 0
    /-
      case h.h
      x y : Real
      h : Fact (LT.lt x y)
      A : Continuous fun z => Min.min (HAdd.hAdd z x) y
      B : Continuous fun z => z 0
      ⊢ Continuous fun x_1 => Min.min (HAdd.hAdd (↑x_1 0) x) y
    -/
    exact (A.comp B).comp continuous_subtype_val
    /-
      🎉 no goals
    -/


/-- The right chart for the topological space `[x, y]`, defined on `(x,y]` and sending `y` to `0` in
`EuclideanHalfSpace 1`.
-/
def IccRightChart (x y : ℝ) [h : Fact (x < y)] :
    PartialHomeomorph (Icc x y) (EuclideanHalfSpace 1) where
  source := { z : Icc x y | x < z.val }
  target := { z : EuclideanHalfSpace 1 | z.val 0 < y - x }
  toFun z := ⟨fun _ => y - z.val, sub_nonneg.mpr z.property.2⟩
  invFun z :=
                             /-
                               x y : Real
                               h : Fact (LT.lt x y)
                               z : EuclideanHalfSpace 1
                               ⊢ Membership.mem (Set.Icc x y) (Max.max (HSub.hSub y (↑z 0)) x)
                             -/
    ⟨max (y - z.val 0) x, by simp [le_refl, z.prop, le_of_lt h.out, sub_eq_add_neg]⟩
                             /-
                               🎉 no goals
                             -/
                    /-
                      x y : Real
                      h : Fact (LT.lt x y)
                      ⊢ ∀ ⦃x_1 : ↑(Set.Icc x y)⦄, Membership.mem (setOf fun z => LT.lt x ↑z) x_1 → M …
                    -/
  map_source' := by simp only [imp_self, mem_setOf_eq, sub_lt_sub_iff_left, forall_true_iff]
                    /-
                      🎉 no goals
                    -/
  map_target' := by
    /-
      x y : Real
      h : Fact (LT.lt x y)
      ⊢ ∀ ⦃x_1 : EuclideanHalfSpace 1⦄, Membership.mem (setOf fun z => LT.lt (↑z 0)  …
    -/
    simp only [lt_max_iff, mem_setOf_eq]; intro z hz; left
    /-
      case h
      x y : Real
      h : Fact (LT.lt x y)
      z : EuclideanHalfSpace 1
      hz : LT.lt (↑z 0) (HSub.hSub y x)
      ⊢ LT.lt x (HSub.hSub y (↑z 0))
    -/
    linarith
    /-
      🎉 no goals
    -/
  left_inv' := by
    /-
      x y : Real
      h : Fact (LT.lt x y)
      ⊢ ∀ ⦃x_1 : ↑(Set.Icc x y)⦄, Membership.mem (setOf fun z => LT.lt x ↑z) x_1 → E …
    -/
    rintro ⟨z, hz⟩ h'z
    /-
      case mk
      x y : Real
      h : Fact (LT.lt x y)
      z : Real
      hz : Membership.mem (Set.Icc x y) z
      h'z : Membership.mem (setOf fun z => LT.lt x ↑z) ⟨z, hz⟩
      ⊢ Eq ((fun z => ⟨Max.max (HSub.hSub y (↑z 0)) x, ⋯⟩) ((fun z => ⟨fun x_1 => HS …
    -/
    simp only [mem_setOf_eq, mem_Icc] at hz h'z
    /-
      case mk
      x y : Real
      h : Fact (LT.lt x y)
      z : Real
      hz✝ : Membership.mem (Set.Icc x y) z
      h'z : LT.lt x z
      hz : And (LE.le x z) (LE.le z y)
      ⊢ Eq ((fun z => ⟨Max.max (HSub.hSub y (↑z 0)) x, ⋯⟩) ((fun z => ⟨fun x_1 => HS …
    -/
    simp only [hz, sub_eq_add_neg, max_eq_left, add_add_neg_cancel'_right, neg_add_rev, neg_neg]
    /-
      🎉 no goals
    -/
  right_inv' := by
    /-
      x y : Real
      h : Fact (LT.lt x y)
      ⊢ ∀ ⦃x_1 : EuclideanHalfSpace 1⦄, Membership.mem (setOf fun z => LT.lt (↑z 0)  …
    -/
    rintro ⟨z, hz⟩ h'z
    /-
      case mk
      x y : Real
      h : Fact (LT.lt x y)
      z : EuclideanSpace Real (Fin 1)
      hz : LE.le 0 (z 0)
      h'z : Membership.mem (setOf fun z => LT.lt (↑z 0) (HSub.hSub y x)) ⟨z, hz⟩
      ⊢ Eq ((fun z => ⟨fun x_1 => HSub.hSub y ↑z, ⋯⟩) ((fun z => ⟨Max.max (HSub.hSub …
    -/
    rw [Subtype.mk_eq_mk]
    /-
      case mk
      x y : Real
      h : Fact (LT.lt x y)
      z : EuclideanSpace Real (Fin 1)
      hz : LE.le 0 (z 0)
      h'z : Membership.mem (setOf fun z => LT.lt (↑z 0) (HSub.hSub y x)) ⟨z, hz⟩
      ⊢ Eq (fun x_1 => HSub.hSub y ↑((fun z => ⟨Max.max (HSub.hSub y (↑z 0)) x, ⋯⟩)  …
    -/
    funext i
    /-
      case mk.h
      x y : Real
      h : Fact (LT.lt x y)
      z : EuclideanSpace Real (Fin 1)
      hz : LE.le 0 (z 0)
      h'z : Membership.mem (setOf fun z => LT.lt (↑z 0) (HSub.hSub y x)) ⟨z, hz⟩
      i : Fin 1
      ⊢ Eq (HSub.hSub y ↑((fun z => ⟨Max.max (HSub.hSub y (↑z 0)) x, ⋯⟩) ⟨z, hz⟩)) ( …
    -/
    dsimp at hz h'z
    /-
      case mk.h
      x y : Real
      h : Fact (LT.lt x y)
      z : EuclideanSpace Real (Fin 1)
      hz : LE.le 0 (z 0)
      h'z : LT.lt (z 0) (HSub.hSub y x)
      i : Fin 1
      ⊢ Eq (HSub.hSub y ↑((fun z => ⟨Max.max (HSub.hSub y (↑z 0)) x, ⋯⟩) ⟨z, hz⟩)) ( …
    -/
    have A : x ≤ y - z 0 := by linarith
    /-
      case mk.h
      x y : Real
      h : Fact (LT.lt x y)
      z : EuclideanSpace Real (Fin 1)
      hz : LE.le 0 (z 0)
      h'z : LT.lt (z 0) (HSub.hSub y x)
      i : Fin 1
      A : LE.le x (HSub.hSub y (z 0))
      ⊢ Eq (HSub.hSub y ↑((fun z => ⟨Max.max (HSub.hSub y (↑z 0)) x, ⋯⟩) ⟨z, hz⟩)) ( …
    -/
    rw [Subsingleton.elim i 0]
    /-
      case mk.h
      x y : Real
      h : Fact (LT.lt x y)
      z : EuclideanSpace Real (Fin 1)
      hz : LE.le 0 (z 0)
      h'z : LT.lt (z 0) (HSub.hSub y x)
      i : Fin 1
      A : LE.le x (HSub.hSub y (z 0))
      ⊢ Eq (HSub.hSub y ↑((fun z => ⟨Max.max (HSub.hSub y (↑z 0)) x, ⋯⟩) ⟨z, hz⟩)) ( …
    -/
    simp only [A, sub_sub_cancel, max_eq_left]
    /-
      🎉 no goals
    -/
  open_source :=
    haveI : IsOpen { z : ℝ | x < z } := isOpen_Ioi
    this.preimage continuous_subtype_val
  open_target := by
    /-
      x y : Real
      h : Fact (LT.lt x y)
      ⊢ IsOpen { toFun := fun z => ⟨fun x_1 => HSub.hSub y ↑z, ⋯⟩, invFun := fun z = …
    -/
    have : IsOpen { z : ℝ | z < y - x } := isOpen_Iio
    have : IsOpen { z : EuclideanSpace ℝ (Fin 1) | z 0 < y - x } :=
      this.preimage (@continuous_apply (Fin 1) (fun _ => ℝ) _ 0)
    /-
      x y : Real
      h : Fact (LT.lt x y)
      this✝ : IsOpen (setOf fun z => LT.lt z (HSub.hSub y x))
      this : IsOpen (setOf fun z => LT.lt (z 0) (HSub.hSub y x))
      ⊢ IsOpen { toFun := fun z => ⟨fun x_1 => HSub.hSub y ↑z, ⋯⟩, invFun := fun z = …
    -/
    exact this.preimage continuous_subtype_val
    /-
      🎉 no goals
    -/
  continuousOn_toFun := by
    /-
      x y : Real
      h : Fact (LT.lt x y)
      ⊢ ContinuousOn ↑{ toFun := fun z => ⟨fun x_1 => HSub.hSub y ↑z, ⋯⟩, invFun :=  …
    -/
    apply Continuous.continuousOn
    /-
      case h
      x y : Real
      h : Fact (LT.lt x y)
      ⊢ Continuous ↑{ toFun := fun z => ⟨fun x_1 => HSub.hSub y ↑z, ⋯⟩, invFun := fu …
    -/
    apply Continuous.subtype_mk
    have : Continuous fun (z : ℝ) (_ : Fin 1) => y - z :=
      continuous_const.sub (continuous_pi fun _ => continuous_id)
    /-
      case h.h
      x y : Real
      h : Fact (LT.lt x y)
      this : Continuous fun z x => HSub.hSub y z
      ⊢ Continuous fun x_1 x_2 => HSub.hSub y ↑x_1
    -/
    exact this.comp continuous_subtype_val
    /-
      🎉 no goals
    -/
  continuousOn_invFun := by
    /-
      x y : Real
      h : Fact (LT.lt x y)
      ⊢ ContinuousOn { toFun := fun z => ⟨fun x_1 => HSub.hSub y ↑z, ⋯⟩, invFun := f …
    -/
    apply Continuous.continuousOn
    /-
      case h
      x y : Real
      h : Fact (LT.lt x y)
      ⊢ Continuous { toFun := fun z => ⟨fun x_1 => HSub.hSub y ↑z, ⋯⟩, invFun := fun …
    -/
    apply Continuous.subtype_mk
    have A : Continuous fun z : ℝ => max (y - z) x :=
      (continuous_const.sub continuous_id).max continuous_const
    /-
      case h.h
      x y : Real
      h : Fact (LT.lt x y)
      A : Continuous fun z => Max.max (HSub.hSub y z) x
      ⊢ Continuous fun x_1 => Max.max (HSub.hSub y (↑x_1 0)) x
    -/
    have B : Continuous fun z : EuclideanSpace ℝ (Fin 1) => z 0 := continuous_apply 0
    /-
      case h.h
      x y : Real
      h : Fact (LT.lt x y)
      A : Continuous fun z => Max.max (HSub.hSub y z) x
      B : Continuous fun z => z 0
      ⊢ Continuous fun x_1 => Max.max (HSub.hSub y (↑x_1 0)) x
    -/
    exact (A.comp B).comp continuous_subtype_val
    /-
      🎉 no goals
    -/


/-- Charted space structure on `[x, y]`, using only two charts taking values in
`EuclideanHalfSpace 1`.
-/
instance IccChartedSpace (x y : ℝ) [h : Fact (x < y)] :
    ChartedSpace (EuclideanHalfSpace 1) (Icc x y) where
  atlas := {IccLeftChart x y, IccRightChart x y}
  chartAt z := if z.val < y then IccLeftChart x y else IccRightChart x y
  mem_chart_source z := by
    /-
      x y : Real
      h : Fact (LT.lt x y)
      z : ↑(Set.Icc x y)
      ⊢ Membership.mem ((fun z => ite (LT.lt (↑z) y) (IccLeftChart x y) (IccRightCha …
    -/
    by_cases h' : z.val < y
      /-
        case pos
        x y : Real
        h : Fact (LT.lt x y)
        z : ↑(Set.Icc x y)
        h' : LT.lt (↑z) y
        ⊢ Membership.mem ((fun z => ite (LT.lt (↑z) y) (IccLeftChart x y) (IccRightCha …
      -/
    · simp only [h', if_true]
      /-
        case pos
        x y : Real
        h : Fact (LT.lt x y)
        z : ↑(Set.Icc x y)
        h' : LT.lt (↑z) y
        ⊢ Membership.mem (IccLeftChart x y).source z
      -/
      exact h'
      /-
        🎉 no goals
      -/
      /-
        case neg
        x y : Real
        h : Fact (LT.lt x y)
        z : ↑(Set.Icc x y)
        h' : Not (LT.lt (↑z) y)
        ⊢ Membership.mem ((fun z => ite (LT.lt (↑z) y) (IccLeftChart x y) (IccRightCha …
      -/
    · simp only [h', if_false]
      /-
        case neg
        x y : Real
        h : Fact (LT.lt x y)
        z : ↑(Set.Icc x y)
        h' : Not (LT.lt (↑z) y)
        ⊢ Membership.mem (IccRightChart x y).source z
      -/
      apply lt_of_lt_of_le h.out
      /-
        case neg
        x y : Real
        h : Fact (LT.lt x y)
        z : ↑(Set.Icc x y)
        h' : Not (LT.lt (↑z) y)
        ⊢ LE.le y ↑z
      -/
      simpa only [not_lt] using h'
      /-
        🎉 no goals
      -/
                          /-
                            x y : Real
                            h : Fact (LT.lt x y)
                            z : ↑(Set.Icc x y)
                            ⊢ Membership.mem (Insert.insert (IccLeftChart x y) (Singleton.singleton (IccRi …
                          -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  chart_mem_atlas z := by by_cases h' : (z : ℝ) < y <;> simp [h']
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- The manifold structure on `[x, y]` is smooth.
-/
instance Icc_smoothManifoldWithCorners (x y : ℝ) [Fact (x < y)] :
    SmoothManifoldWithCorners (𝓡∂ 1) (Icc x y) := by
  have M : ContDiff ℝ ∞ (show EuclideanSpace ℝ (Fin 1) → EuclideanSpace ℝ (Fin 1)
      from fun z i => -z i + (y - x)) :=
    contDiff_id.neg.add contDiff_const
  /-
    x y : Real
    inst✝ : Fact (LT.lt x y)
    M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
    ⊢ SmoothManifoldWithCorners (modelWithCornersEuclideanHalfSpace 1) ↑(Set.Icc x …
  -/
  apply smoothManifoldWithCorners_of_contDiffOn
  /-
    case h
    x y : Real
    inst✝ : Fact (LT.lt x y)
    M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
    ⊢ ∀ (e e' : PartialHomeomorph (↑(Set.Icc x y)) (EuclideanHalfSpace 1)), Member …
  -/
  intro e e' he he'
  /-
    case h
    x y : Real
    inst✝ : Fact (LT.lt x y)
    M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
    e e' : PartialHomeomorph (↑(Set.Icc x y)) (EuclideanHalfSpace 1)
    he : Membership.mem (atlas (EuclideanHalfSpace 1) ↑(Set.Icc x y)) e
    he' : Membership.mem (atlas (EuclideanHalfSpace 1) ↑(Set.Icc x y)) e'
    ⊢ ContDiffOn Real (↑Top.top) (Function.comp (↑(modelWithCornersEuclideanHalfSp …
  -/
  simp only [atlas, mem_singleton_iff, mem_insert_iff] at he he'
  /- We need to check that any composition of two charts gives a `C^∞` function. Each chart can be
  either the left chart or the right chart, leaving 4 possibilities that we handle successively. -/
  /-
    case h
    x y : Real
    inst✝ : Fact (LT.lt x y)
    M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
    e e' : PartialHomeomorph (↑(Set.Icc x y)) (EuclideanHalfSpace 1)
    he : Membership.mem ChartedSpace.atlas e
    he' : Membership.mem ChartedSpace.atlas e'
    ⊢ ContDiffOn Real (↑Top.top) (Function.comp (↑(modelWithCornersEuclideanHalfSp …
  -/
  rcases he with (rfl | rfl) <;> rcases he' with (rfl | rfl)
  · -- `e = left chart`, `e' = left chart`
    /-
      case h.inl.inl
      x y : Real
      inst✝ : Fact (LT.lt x y)
      M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
      ⊢ ContDiffOn Real (↑Top.top) (Function.comp (↑(modelWithCornersEuclideanHalfSp …
    -/
    exact (mem_groupoid_of_pregroupoid.mpr (symm_trans_mem_contDiffGroupoid _)).1
    /-
      🎉 no goals
    -/
  · -- `e = left chart`, `e' = right chart`
    /-
      case h.inl.inr
      x y : Real
      inst✝ : Fact (LT.lt x y)
      M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
      ⊢ ContDiffOn Real (↑Top.top) (Function.comp (↑(modelWithCornersEuclideanHalfSp …
    -/
    apply M.contDiffOn.congr
    /-
      case h.inl.inr
      x y : Real
      inst✝ : Fact (LT.lt x y)
      M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
      ⊢ ∀ (x_1 : EuclideanSpace Real (Fin 1)), Membership.mem (Inter.inter (Set.prei …
    -/
    rintro _ ⟨⟨hz₁, hz₂⟩, ⟨⟨z, hz₀⟩, rfl⟩⟩
    simp only [modelWithCornersEuclideanHalfSpace, IccLeftChart, IccRightChart, update_self,
      max_eq_left, hz₀, lt_sub_iff_add_lt, mfld_simps] at hz₁ hz₂
    /-
      case h.inl.inr.intro.intro.intro.mk
      x y : Real
      inst✝ : Fact (LT.lt x y)
      M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
      z : EuclideanSpace Real (Fin 1)
      hz₀ : LE.le 0 (z 0)
      hz₁ : LT.lt (HAdd.hAdd (z 0) x) y
      hz₂ : LT.lt x (Min.min (HAdd.hAdd (z 0) x) y)
      ⊢ Eq (Function.comp (↑(modelWithCornersEuclideanHalfSpace 1)) (Function.comp ↑ …
    -/
    rw [min_eq_left hz₁.le, lt_add_iff_pos_left] at hz₂
    /-
      case h.inl.inr.intro.intro.intro.mk
      x y : Real
      inst✝ : Fact (LT.lt x y)
      M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
      z : EuclideanSpace Real (Fin 1)
      hz₀ : LE.le 0 (z 0)
      hz₁ : LT.lt (HAdd.hAdd (z 0) x) y
      hz₂ : LT.lt 0 (z 0)
      ⊢ Eq (Function.comp (↑(modelWithCornersEuclideanHalfSpace 1)) (Function.comp ↑ …
    -/
    ext i
    /-
      case h.inl.inr.intro.intro.intro.mk.h
      x y : Real
      inst✝ : Fact (LT.lt x y)
      M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
      z : EuclideanSpace Real (Fin 1)
      hz₀ : LE.le 0 (z 0)
      hz₁ : LT.lt (HAdd.hAdd (z 0) x) y
      hz₂ : LT.lt 0 (z 0)
      i : Fin 1
      ⊢ Eq (Function.comp (↑(modelWithCornersEuclideanHalfSpace 1)) (Function.comp ↑ …
    -/
    rw [Subsingleton.elim i 0]
    simp only [modelWithCornersEuclideanHalfSpace, IccLeftChart, IccRightChart, *, PiLp.add_apply,
      PiLp.neg_apply, max_eq_left, min_eq_left hz₁.le, update_self, mfld_simps]
    /-
      case h.inl.inr.intro.intro.intro.mk.h
      x y : Real
      inst✝ : Fact (LT.lt x y)
      M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
      z : EuclideanSpace Real (Fin 1)
      hz₀ : LE.le 0 (z 0)
      hz₁ : LT.lt (HAdd.hAdd (z 0) x) y
      hz₂ : LT.lt 0 (z 0)
      i : Fin 1
      ⊢ Eq (HSub.hSub y (HAdd.hAdd (z 0) x)) (HAdd.hAdd (Neg.neg (z 0)) (HSub.hSub y …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/
  · -- `e = right chart`, `e' = left chart`
    /-
      case h.inr.inl
      x y : Real
      inst✝ : Fact (LT.lt x y)
      M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
      ⊢ ContDiffOn Real (↑Top.top) (Function.comp (↑(modelWithCornersEuclideanHalfSp …
    -/
    apply M.contDiffOn.congr
    /-
      case h.inr.inl
      x y : Real
      inst✝ : Fact (LT.lt x y)
      M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
      ⊢ ∀ (x_1 : EuclideanSpace Real (Fin 1)), Membership.mem (Inter.inter (Set.prei …
    -/
    rintro _ ⟨⟨hz₁, hz₂⟩, ⟨z, hz₀⟩, rfl⟩
    simp only [modelWithCornersEuclideanHalfSpace, IccLeftChart, IccRightChart, max_lt_iff,
      update_self, max_eq_left hz₀, mfld_simps] at hz₁ hz₂
    /-
      case h.inr.inl.intro.intro.intro.mk
      x y : Real
      inst✝ : Fact (LT.lt x y)
      M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
      z : EuclideanSpace Real (Fin 1)
      hz₀ : LE.le 0 (z 0)
      hz₁ : LT.lt (z 0) (HSub.hSub y x)
      hz₂ : And (LT.lt (HSub.hSub y (z 0)) y) (LT.lt x y)
      ⊢ Eq (Function.comp (↑(modelWithCornersEuclideanHalfSpace 1)) (Function.comp ↑ …
    -/
    rw [lt_sub_comm] at hz₁
    /-
      case h.inr.inl.intro.intro.intro.mk
      x y : Real
      inst✝ : Fact (LT.lt x y)
      M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
      z : EuclideanSpace Real (Fin 1)
      hz₀ : LE.le 0 (z 0)
      hz₁ : LT.lt x (HSub.hSub y (z 0))
      hz₂ : And (LT.lt (HSub.hSub y (z 0)) y) (LT.lt x y)
      ⊢ Eq (Function.comp (↑(modelWithCornersEuclideanHalfSpace 1)) (Function.comp ↑ …
    -/
    ext i
    /-
      case h.inr.inl.intro.intro.intro.mk.h
      x y : Real
      inst✝ : Fact (LT.lt x y)
      M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
      z : EuclideanSpace Real (Fin 1)
      hz₀ : LE.le 0 (z 0)
      hz₁ : LT.lt x (HSub.hSub y (z 0))
      hz₂ : And (LT.lt (HSub.hSub y (z 0)) y) (LT.lt x y)
      i : Fin 1
      ⊢ Eq (Function.comp (↑(modelWithCornersEuclideanHalfSpace 1)) (Function.comp ↑ …
    -/
    rw [Subsingleton.elim i 0]
    simp only [modelWithCornersEuclideanHalfSpace, IccLeftChart, IccRightChart, PiLp.add_apply,
      PiLp.neg_apply, update_self, max_eq_left, hz₀, hz₁.le, mfld_simps]
    /-
      case h.inr.inl.intro.intro.intro.mk.h
      x y : Real
      inst✝ : Fact (LT.lt x y)
      M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
      z : EuclideanSpace Real (Fin 1)
      hz₀ : LE.le 0 (z 0)
      hz₁ : LT.lt x (HSub.hSub y (z 0))
      hz₂ : And (LT.lt (HSub.hSub y (z 0)) y) (LT.lt x y)
      i : Fin 1
      ⊢ Eq (HSub.hSub (HSub.hSub y (z 0)) x) (HAdd.hAdd (Neg.neg (z 0)) (HSub.hSub y …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/
  ·-- `e = right chart`, `e' = right chart`
    /-
      case h.inr.inr
      x y : Real
      inst✝ : Fact (LT.lt x y)
      M : ContDiff Real (↑Top.top) (letFun (fun z i => HAdd.hAdd (Neg.neg (z i)) (HS …
      ⊢ ContDiffOn Real (↑Top.top) (Function.comp (↑(modelWithCornersEuclideanHalfSp …
    -/
    exact (mem_groupoid_of_pregroupoid.mpr (symm_trans_mem_contDiffGroupoid _)).1
    /-
      🎉 no goals
    -/


                                                                     /-
                                                                       ⊢ ChartedSpace (EuclideanHalfSpace 1) ↑(Set.Icc 0 1)
                                                                     -/
instance : ChartedSpace (EuclideanHalfSpace 1) (Icc (0 : ℝ) 1) := by infer_instance
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                                                  /-
                                                                    ⊢ SmoothManifoldWithCorners (modelWithCornersEuclideanHalfSpace 1) ↑(Set.Icc 0 …
                                                                  -/
instance : SmoothManifoldWithCorners (𝓡∂ 1) (Icc (0 : ℝ) 1) := by infer_instance
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


