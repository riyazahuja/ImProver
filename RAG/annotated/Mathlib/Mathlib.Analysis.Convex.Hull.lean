/-- The convex hull of a set `s` is the minimal convex set that includes `s`. -/
@[simps! isClosed]
def convexHull : ClosureOperator (Set E) := .ofCompletePred (Convex 𝕜) fun _ ↦ convex_sInter


theorem subset_convexHull : s ⊆ convexHull 𝕜 s :=
  (convexHull 𝕜).le_closure s


theorem convex_convexHull : Convex 𝕜 (convexHull 𝕜 s) := (convexHull 𝕜).isClosed_closure s


theorem convexHull_eq_iInter : convexHull 𝕜 s = ⋂ (t : Set E) (_ : s ⊆ t) (_ : Convex 𝕜 t), t := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ Eq ((convexHull 𝕜) s) (Set.iInter fun t => Set.iInter fun x => Set.iInter fu …
  -/
  simp [convexHull, iInter_subtype, iInter_and]
  /-
    🎉 no goals
  -/


theorem mem_convexHull_iff : x ∈ convexHull 𝕜 s ↔ ∀ t, s ⊆ t → Convex 𝕜 t → x ∈ t := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    ⊢ Iff (Membership.mem ((convexHull 𝕜) s) x) (∀ (t : Set E), HasSubset.Subset s …
  -/
  simp_rw [convexHull_eq_iInter, mem_iInter]
  /-
    🎉 no goals
  -/


theorem convexHull_min : s ⊆ t → Convex 𝕜 t → convexHull 𝕜 s ⊆ t := (convexHull 𝕜).closure_min


theorem Convex.convexHull_subset_iff (ht : Convex 𝕜 t) : convexHull 𝕜 s ⊆ t ↔ s ⊆ t :=
  (show (convexHull 𝕜).IsClosed t from ht).closure_le_iff


@[mono, gcongr]
theorem convexHull_mono (hst : s ⊆ t) : convexHull 𝕜 s ⊆ convexHull 𝕜 t :=
  (convexHull 𝕜).monotone hst


lemma convexHull_eq_self : convexHull 𝕜 s = s ↔ Convex 𝕜 s := (convexHull 𝕜).isClosed_iff.symm


alias ⟨_, Convex.convexHull_eq⟩ := convexHull_eq_self


@[simp]
theorem convexHull_univ : convexHull 𝕜 (univ : Set E) = univ :=
  ClosureOperator.closure_top (convexHull 𝕜)


@[simp]
theorem convexHull_empty : convexHull 𝕜 (∅ : Set E) = ∅ :=
  convex_empty.convexHull_eq


@[simp]
theorem convexHull_empty_iff : convexHull 𝕜 s = ∅ ↔ s = ∅ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ Iff (Eq ((convexHull 𝕜) s) EmptyCollection.emptyCollection) (Eq s EmptyColle …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      s : Set E
      ⊢ Eq ((convexHull 𝕜) s) EmptyCollection.emptyCollection → Eq s EmptyCollection …
    -/
  · intro h
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      s : Set E
      h : Eq ((convexHull 𝕜) s) EmptyCollection.emptyCollection
      ⊢ Eq s EmptyCollection.emptyCollection
    -/
    rw [← Set.subset_empty_iff, ← h]
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      s : Set E
      h : Eq ((convexHull 𝕜) s) EmptyCollection.emptyCollection
      ⊢ HasSubset.Subset s ((convexHull 𝕜) s)
    -/
    exact subset_convexHull 𝕜 _
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      s : Set E
      ⊢ Eq s EmptyCollection.emptyCollection → Eq ((convexHull 𝕜) s) EmptyCollection …
    -/
  · rintro rfl
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      ⊢ Eq ((convexHull 𝕜) EmptyCollection.emptyCollection) EmptyCollection.emptyCol …
    -/
    exact convexHull_empty
    /-
      🎉 no goals
    -/


@[simp]
theorem convexHull_nonempty_iff : (convexHull 𝕜 s).Nonempty ↔ s.Nonempty := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ Iff ((convexHull 𝕜) s).Nonempty s.Nonempty
  -/
  rw [nonempty_iff_ne_empty, nonempty_iff_ne_empty, Ne, Ne]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ Iff (Not (Eq ((convexHull 𝕜) s) EmptyCollection.emptyCollection)) (Not (Eq s …
  -/
  exact not_congr convexHull_empty_iff
  /-
    🎉 no goals
  -/


protected alias ⟨_, Set.Nonempty.convexHull⟩ := convexHull_nonempty_iff


theorem segment_subset_convexHull (hx : x ∈ s) (hy : y ∈ s) : segment 𝕜 x y ⊆ convexHull 𝕜 s :=
  (convex_convexHull _ _).segment_subset (subset_convexHull _ _ hx) (subset_convexHull _ _ hy)


@[simp]
theorem convexHull_singleton (x : E) : convexHull 𝕜 ({x} : Set E) = {x} :=
  (convex_singleton x).convexHull_eq


@[simp]
theorem convexHull_zero : convexHull 𝕜 (0 : Set E) = 0 :=
  convexHull_singleton 0


@[simp]
theorem convexHull_pair (x y : E) : convexHull 𝕜 {x, y} = segment 𝕜 x y := by
  refine (convexHull_min ?_ <| convex_segment _ _).antisymm
    (segment_subset_convexHull (mem_insert _ _) <| subset_insert _ _ <| mem_singleton _)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x y : E
    ⊢ HasSubset.Subset (Insert.insert x (Singleton.singleton y)) (segment 𝕜 x y)
  -/
  rw [insert_subset_iff, singleton_subset_iff]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x y : E
    ⊢ And (Membership.mem (segment 𝕜 x y) x) (Membership.mem (segment 𝕜 x y) y)
  -/
  exact ⟨left_mem_segment _ _ _, right_mem_segment _ _ _⟩
  /-
    🎉 no goals
  -/


theorem convexHull_convexHull_union_left (s t : Set E) :
    convexHull 𝕜 (convexHull 𝕜 s ∪ t) = convexHull 𝕜 (s ∪ t) :=
  ClosureOperator.closure_sup_closure_left _ _ _


theorem convexHull_convexHull_union_right (s t : Set E) :
    convexHull 𝕜 (s ∪ convexHull 𝕜 t) = convexHull 𝕜 (s ∪ t) :=
  ClosureOperator.closure_sup_closure_right _ _ _


theorem Convex.convex_remove_iff_not_mem_convexHull_remove {s : Set E} (hs : Convex 𝕜 s) (x : E) :
    Convex 𝕜 (s \ {x}) ↔ x ∉ convexHull 𝕜 (s \ {x}) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    x : E
    ⊢ Iff (Convex 𝕜 (SDiff.sdiff s (Singleton.singleton x))) (Not (Membership.mem  …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      s : Set E
      hs : Convex 𝕜 s
      x : E
      ⊢ Convex 𝕜 (SDiff.sdiff s (Singleton.singleton x)) → Not (Membership.mem ((con …
    -/
  · rintro hsx hx
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      s : Set E
      hs : Convex 𝕜 s
      x : E
      hsx : Convex 𝕜 (SDiff.sdiff s (Singleton.singleton x))
      hx : Membership.mem ((convexHull 𝕜) (SDiff.sdiff s (Singleton.singleton x))) x
      ⊢ False
    -/
    rw [hsx.convexHull_eq] at hx
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      s : Set E
      hs : Convex 𝕜 s
      x : E
      hsx : Convex 𝕜 (SDiff.sdiff s (Singleton.singleton x))
      hx : Membership.mem (SDiff.sdiff s (Singleton.singleton x)) x
      ⊢ False
    -/
    exact hx.2 (mem_singleton _)
    /-
      🎉 no goals
    -/
  /-
    case mpr
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    x : E
    ⊢ Not (Membership.mem ((convexHull 𝕜) (SDiff.sdiff s (Singleton.singleton x))) …
  -/
  rintro hx
  suffices h : s \ {x} = convexHull 𝕜 (s \ {x}) by
    rw [h]
    exact convex_convexHull 𝕜 _
  exact
    Subset.antisymm (subset_convexHull 𝕜 _) fun y hy =>
      ⟨convexHull_min diff_subset hs hy, by
        rintro (rfl : y = x)
        exact hx hy⟩


theorem IsLinearMap.image_convexHull {f : E → F} (hf : IsLinearMap 𝕜 f) (s : Set E) :
    f '' convexHull 𝕜 s = convexHull 𝕜 (f '' s) :=
  Set.Subset.antisymm
    (image_subset_iff.2 <|
      convexHull_min (image_subset_iff.1 <| subset_convexHull 𝕜 _)
        ((convex_convexHull 𝕜 _).is_linear_preimage hf))
    (convexHull_min (image_subset _ (subset_convexHull 𝕜 s)) <|
      (convex_convexHull 𝕜 s).is_linear_image hf)


theorem LinearMap.image_convexHull (f : E →ₗ[𝕜] F) (s : Set E) :
    f '' convexHull 𝕜 s = convexHull 𝕜 (f '' s) :=
  f.isLinear.image_convexHull s


theorem convexHull_add_subset {s t : Set E} :
    convexHull 𝕜 (s + t) ⊆ convexHull 𝕜 s + convexHull 𝕜 t :=
  convexHull_min (add_subset_add (subset_convexHull _ _) (subset_convexHull _ _))
    (Convex.add (convex_convexHull 𝕜 s) (convex_convexHull 𝕜 t))


theorem convexHull_smul (a : 𝕜) (s : Set E) : convexHull 𝕜 (a • s) = a • convexHull 𝕜 s :=
  (LinearMap.lsmul _ _ a).image_convexHull _ |>.symm


theorem AffineMap.image_convexHull (f : E →ᵃ[𝕜] F) (s : Set E) :
    f '' convexHull 𝕜 s = convexHull 𝕜 (f '' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    f : AffineMap 𝕜 E F
    s : Set E
    ⊢ Eq (Set.image (⇑f) ((convexHull 𝕜) s)) ((convexHull 𝕜) (Set.image (⇑f) s))
  -/
  apply Set.Subset.antisymm
    /-
      case h₁
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : OrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 F
      f : AffineMap 𝕜 E F
      s : Set E
      ⊢ HasSubset.Subset (Set.image (⇑f) ((convexHull 𝕜) s)) ((convexHull 𝕜) (Set.im …
    -/
  · rw [Set.image_subset_iff]
    /-
      case h₁
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : OrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 F
      f : AffineMap 𝕜 E F
      s : Set E
      ⊢ HasSubset.Subset ((convexHull 𝕜) s) (Set.preimage (⇑f) ((convexHull 𝕜) (Set. …
    -/
    refine convexHull_min ?_ ((convex_convexHull 𝕜 (f '' s)).affine_preimage f)
    /-
      case h₁
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : OrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 F
      f : AffineMap 𝕜 E F
      s : Set E
      ⊢ HasSubset.Subset s (Set.preimage (⇑f) ((convexHull 𝕜) (Set.image (⇑f) s)))
    -/
    rw [← Set.image_subset_iff]
    /-
      case h₁
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : OrderedRing 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 F
      f : AffineMap 𝕜 E F
      s : Set E
      ⊢ HasSubset.Subset (Set.image (⇑f) s) ((convexHull 𝕜) (Set.image (⇑f) s))
    -/
    exact subset_convexHull 𝕜 (f '' s)
    /-
      🎉 no goals
    -/
  · exact convexHull_min (Set.image_subset _ (subset_convexHull 𝕜 s))
      ((convex_convexHull 𝕜 s).affine_image f)


theorem convexHull_subset_affineSpan (s : Set E) : convexHull 𝕜 s ⊆ (affineSpan 𝕜 s : Set E) :=
  convexHull_min (subset_affineSpan 𝕜 s) (affineSpan 𝕜 s).convex


@[simp]
theorem affineSpan_convexHull (s : Set E) : affineSpan 𝕜 (convexHull 𝕜 s) = affineSpan 𝕜 s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ Eq (affineSpan 𝕜 ((convexHull 𝕜) s)) (affineSpan 𝕜 s)
  -/
  refine le_antisymm ?_ (affineSpan_mono 𝕜 (subset_convexHull 𝕜 s))
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ LE.le (affineSpan 𝕜 ((convexHull 𝕜) s)) (affineSpan 𝕜 s)
  -/
  rw [affineSpan_le]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ HasSubset.Subset ((convexHull 𝕜) s) ↑(affineSpan 𝕜 s)
  -/
  exact convexHull_subset_affineSpan s
  /-
    🎉 no goals
  -/


theorem convexHull_neg (s : Set E) : convexHull 𝕜 (-s) = -convexHull 𝕜 s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ Eq ((convexHull 𝕜) (Neg.neg s)) (Neg.neg ((convexHull 𝕜) s))
  -/
  simp_rw [← image_neg_eq_neg]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ Eq ((convexHull 𝕜) (Set.image (fun x => Neg.neg x) s)) (Set.image (fun x =>  …
  -/
  exact AffineMap.image_convexHull (-1) _ |>.symm
  /-
    🎉 no goals
  -/


lemma convexHull_vadd (x : E) (s : Set E) : convexHull 𝕜 (x +ᵥ s) = x +ᵥ convexHull 𝕜 s :=
  (AffineEquiv.constVAdd 𝕜 _ x).toAffineMap.image_convexHull s |>.symm


