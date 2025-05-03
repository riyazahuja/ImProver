/-- The join of two sets is the union of the segments joining them. This can be interpreted as the
topological join, but within the original space. -/
def convexJoin (s t : Set E) : Set E :=
  ⋃ (x ∈ s) (y ∈ t), segment 𝕜 x y


theorem mem_convexJoin : x ∈ convexJoin 𝕜 s t ↔ ∃ a ∈ s, ∃ b ∈ t, x ∈ segment 𝕜 a b := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s t : Set E
    x : E
    ⊢ Iff (Membership.mem (convexJoin 𝕜 s t) x) (Exists fun a => And (Membership.m …
  -/
  simp [convexJoin]
  /-
    🎉 no goals
  -/


theorem convexJoin_comm (s t : Set E) : convexJoin 𝕜 s t = convexJoin 𝕜 t s :=
                               /-
                                 𝕜 : Type u_2
                                 E : Type u_3
                                 inst✝² : OrderedSemiring 𝕜
                                 inst✝¹ : AddCommMonoid E
                                 inst✝ : Module 𝕜 E
                                 s t : Set E
                                 ⊢ Eq (Set.iUnion fun i₂ => Set.iUnion fun j₂ => Set.iUnion fun i₁ => Set.iUnio …
                               -/
  (iUnion₂_comm _).trans <| by simp_rw [convexJoin, segment_symm]
                               /-
                                 🎉 no goals
                               -/


theorem convexJoin_mono (hs : s₁ ⊆ s₂) (ht : t₁ ⊆ t₂) : convexJoin 𝕜 s₁ t₁ ⊆ convexJoin 𝕜 s₂ t₂ :=
  biUnion_mono hs fun _ _ => biUnion_subset_biUnion_left ht


theorem convexJoin_mono_left (hs : s₁ ⊆ s₂) : convexJoin 𝕜 s₁ t ⊆ convexJoin 𝕜 s₂ t :=
  convexJoin_mono hs Subset.rfl


theorem convexJoin_mono_right (ht : t₁ ⊆ t₂) : convexJoin 𝕜 s t₁ ⊆ convexJoin 𝕜 s t₂ :=
  convexJoin_mono Subset.rfl ht


@[simp]
                                                                       /-
                                                                         𝕜 : Type u_2
                                                                         E : Type u_3
                                                                         inst✝² : OrderedSemiring 𝕜
                                                                         inst✝¹ : AddCommMonoid E
                                                                         inst✝ : Module 𝕜 E
                                                                         t : Set E
                                                                         ⊢ Eq (convexJoin 𝕜 EmptyCollection.emptyCollection t) EmptyCollection.emptyCol …
                                                                       -/
theorem convexJoin_empty_left (t : Set E) : convexJoin 𝕜 ∅ t = ∅ := by simp [convexJoin]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
                                                                        /-
                                                                          𝕜 : Type u_2
                                                                          E : Type u_3
                                                                          inst✝² : OrderedSemiring 𝕜
                                                                          inst✝¹ : AddCommMonoid E
                                                                          inst✝ : Module 𝕜 E
                                                                          s : Set E
                                                                          ⊢ Eq (convexJoin 𝕜 s EmptyCollection.emptyCollection) EmptyCollection.emptyCol …
                                                                        -/
theorem convexJoin_empty_right (s : Set E) : convexJoin 𝕜 s ∅ = ∅ := by simp [convexJoin]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
theorem convexJoin_singleton_left (t : Set E) (x : E) :
                                                      /-
                                                        𝕜 : Type u_2
                                                        E : Type u_3
                                                        inst✝² : OrderedSemiring 𝕜
                                                        inst✝¹ : AddCommMonoid E
                                                        inst✝ : Module 𝕜 E
                                                        t : Set E
                                                        x : E
                                                        ⊢ Eq (convexJoin 𝕜 (Singleton.singleton x) t) (Set.iUnion fun y => Set.iUnion  …
                                                      -/
    convexJoin 𝕜 {x} t = ⋃ y ∈ t, segment 𝕜 x y := by simp [convexJoin]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem convexJoin_singleton_right (s : Set E) (y : E) :
                                                      /-
                                                        𝕜 : Type u_2
                                                        E : Type u_3
                                                        inst✝² : OrderedSemiring 𝕜
                                                        inst✝¹ : AddCommMonoid E
                                                        inst✝ : Module 𝕜 E
                                                        s : Set E
                                                        y : E
                                                        ⊢ Eq (convexJoin 𝕜 s (Singleton.singleton y)) (Set.iUnion fun x => Set.iUnion  …
                                                      -/
    convexJoin 𝕜 s {y} = ⋃ x ∈ s, segment 𝕜 x y := by simp [convexJoin]
                                                      /-
                                                        🎉 no goals
                                                      -/


                                                                                   /-
                                                                                     𝕜 : Type u_2
                                                                                     E : Type u_3
                                                                                     inst✝² : OrderedSemiring 𝕜
                                                                                     inst✝¹ : AddCommMonoid E
                                                                                     inst✝ : Module 𝕜 E
                                                                                     y x : E
                                                                                     ⊢ Eq (convexJoin 𝕜 (Singleton.singleton x) (Singleton.singleton y)) (segment 𝕜 …
                                                                                   -/
theorem convexJoin_singletons (x : E) : convexJoin 𝕜 {x} {y} = segment 𝕜 x y := by simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[simp]
theorem convexJoin_union_left (s₁ s₂ t : Set E) :
    convexJoin 𝕜 (s₁ ∪ s₂) t = convexJoin 𝕜 s₁ t ∪ convexJoin 𝕜 s₂ t := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s₁ s₂ t : Set E
    ⊢ Eq (convexJoin 𝕜 (Union.union s₁ s₂) t) (Union.union (convexJoin 𝕜 s₁ t) (co …
  -/
  simp_rw [convexJoin, mem_union, iUnion_or, iUnion_union_distrib]
  /-
    🎉 no goals
  -/


@[simp]
theorem convexJoin_union_right (s t₁ t₂ : Set E) :
    convexJoin 𝕜 s (t₁ ∪ t₂) = convexJoin 𝕜 s t₁ ∪ convexJoin 𝕜 s t₂ := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s t₁ t₂ : Set E
    ⊢ Eq (convexJoin 𝕜 s (Union.union t₁ t₂)) (Union.union (convexJoin 𝕜 s t₁) (co …
  -/
  simp_rw [convexJoin_comm s, convexJoin_union_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem convexJoin_iUnion_left (s : ι → Set E) (t : Set E) :
    convexJoin 𝕜 (⋃ i, s i) t = ⋃ i, convexJoin 𝕜 (s i) t := by
  /-
    ι : Sort u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : ι → Set E
    t : Set E
    ⊢ Eq (convexJoin 𝕜 (Set.iUnion fun i => s i) t) (Set.iUnion fun i => convexJoi …
  -/
  simp_rw [convexJoin, mem_iUnion, iUnion_exists]
  /-
    ι : Sort u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : ι → Set E
    t : Set E
    ⊢ Eq (Set.iUnion fun x => Set.iUnion fun i => Set.iUnion fun h => Set.iUnion f …
  -/
  exact iUnion_comm _
  /-
    🎉 no goals
  -/


@[simp]
theorem convexJoin_iUnion_right (s : Set E) (t : ι → Set E) :
    convexJoin 𝕜 s (⋃ i, t i) = ⋃ i, convexJoin 𝕜 s (t i) := by
  /-
    ι : Sort u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    t : ι → Set E
    ⊢ Eq (convexJoin 𝕜 s (Set.iUnion fun i => t i)) (Set.iUnion fun i => convexJoi …
  -/
  simp_rw [convexJoin_comm s, convexJoin_iUnion_left]
  /-
    🎉 no goals
  -/


theorem segment_subset_convexJoin (hx : x ∈ s) (hy : y ∈ t) : segment 𝕜 x y ⊆ convexJoin 𝕜 s t :=
  subset_iUnion₂_of_subset x hx <| subset_iUnion₂ (s := fun y _ ↦ segment 𝕜 x y) y hy


theorem subset_convexJoin_left (h : t.Nonempty) : s ⊆ convexJoin 𝕜 s t := fun _x hx =>
  let ⟨_y, hy⟩ := h
  segment_subset_convexJoin hx hy <| left_mem_segment _ _ _


theorem subset_convexJoin_right (h : s.Nonempty) : t ⊆ convexJoin 𝕜 s t :=
  convexJoin_comm (𝕜 := 𝕜) t s ▸ subset_convexJoin_left h


theorem convexJoin_subset (hs : s ⊆ u) (ht : t ⊆ u) (hu : Convex 𝕜 u) : convexJoin 𝕜 s t ⊆ u :=
  iUnion₂_subset fun _x hx => iUnion₂_subset fun _y hy => hu.segment_subset (hs hx) (ht hy)


theorem convexJoin_subset_convexHull (s t : Set E) : convexJoin 𝕜 s t ⊆ convexHull 𝕜 (s ∪ t) :=
  convexJoin_subset (subset_union_left.trans <| subset_convexHull _ _)
      (subset_union_right.trans <| subset_convexHull _ _) <|
    convex_convexHull _ _


theorem convexJoin_assoc_aux (s t u : Set E) :
    convexJoin 𝕜 (convexJoin 𝕜 s t) u ⊆ convexJoin 𝕜 s (convexJoin 𝕜 t u) := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t u : Set E
    ⊢ HasSubset.Subset (convexJoin 𝕜 (convexJoin 𝕜 s t) u) (convexJoin 𝕜 s (convex …
  -/
  simp_rw [subset_def, mem_convexJoin]
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t u : Set E
    ⊢ ∀ (x : E), (Exists fun a => And (Exists fun a_1 => And (Membership.mem s a_1 …
  -/
  rintro _ ⟨z, ⟨x, hx, y, hy, a₁, b₁, ha₁, hb₁, hab₁, rfl⟩, z, hz, a₂, b₂, ha₂, hb₂, hab₂, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t u : Set E
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem t y
    a₁ b₁ : 𝕜
    ha₁ : LE.le 0 a₁
    hb₁ : LE.le 0 b₁
    hab₁ : Eq (HAdd.hAdd a₁ b₁) 1
    z : E
    hz : Membership.mem u z
    a₂ b₂ : 𝕜
    ha₂ : LE.le 0 a₂
    hb₂ : LE.le 0 b₂
    hab₂ : Eq (HAdd.hAdd a₂ b₂) 1
    ⊢ Exists fun a => And (Membership.mem s a) (Exists fun b => And (Exists fun a  …
  -/
  obtain rfl | hb₂ := hb₂.eq_or_lt
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      𝕜 : Type u_2
      E : Type u_3
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s t u : Set E
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem t y
      a₁ b₁ : 𝕜
      ha₁ : LE.le 0 a₁
      hb₁ : LE.le 0 b₁
      hab₁ : Eq (HAdd.hAdd a₁ b₁) 1
      z : E
      hz : Membership.mem u z
      a₂ : 𝕜
      ha₂ : LE.le 0 a₂
      hb₂ : LE.le 0 0
      hab₂ : Eq (HAdd.hAdd a₂ 0) 1
      ⊢ Exists fun a => And (Membership.mem s a) (Exists fun b => And (Exists fun a  …
    -/
  · refine ⟨x, hx, y, ⟨y, hy, z, hz, left_mem_segment 𝕜 _ _⟩, a₁, b₁, ha₁, hb₁, hab₁, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      𝕜 : Type u_2
      E : Type u_3
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s t u : Set E
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem t y
      a₁ b₁ : 𝕜
      ha₁ : LE.le 0 a₁
      hb₁ : LE.le 0 b₁
      hab₁ : Eq (HAdd.hAdd a₁ b₁) 1
      z : E
      hz : Membership.mem u z
      a₂ : 𝕜
      ha₂ : LE.le 0 a₂
      hb₂ : LE.le 0 0
      hab₂ : Eq (HAdd.hAdd a₂ 0) 1
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul a₁ x) (HSMul.hSMul b₁ y)) (HAdd.hAdd (HSMul.hSMul …
    -/
    linear_combination (norm := module) -hab₂ • (a₁ • x + b₁ • y)
    /-
      🎉 no goals
    -/
  refine
    ⟨x, hx, (a₂ * b₁ / (a₂ * b₁ + b₂)) • y + (b₂ / (a₂ * b₁ + b₂)) • z,
      ⟨y, hy, z, hz, _, _, by positivity, by positivity, by field_simp, rfl⟩,
      a₂ * a₁, a₂ * b₁ + b₂, by positivity, by positivity, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      𝕜 : Type u_2
      E : Type u_3
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s t u : Set E
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem t y
      a₁ b₁ : 𝕜
      ha₁ : LE.le 0 a₁
      hb₁ : LE.le 0 b₁
      hab₁ : Eq (HAdd.hAdd a₁ b₁) 1
      z : E
      hz : Membership.mem u z
      a₂ b₂ : 𝕜
      ha₂ : LE.le 0 a₂
      hb₂✝ : LE.le 0 b₂
      hab₂ : Eq (HAdd.hAdd a₂ b₂) 1
      hb₂ : LT.lt 0 b₂
      ⊢ Eq (HAdd.hAdd (HMul.hMul a₂ a₁) (HAdd.hAdd (HMul.hMul a₂ b₁) b₂)) 1
    -/
  · linear_combination a₂ * hab₁ + hab₂
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      𝕜 : Type u_2
      E : Type u_3
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s t u : Set E
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem t y
      a₁ b₁ : 𝕜
      ha₁ : LE.le 0 a₁
      hb₁ : LE.le 0 b₁
      hab₁ : Eq (HAdd.hAdd a₁ b₁) 1
      z : E
      hz : Membership.mem u z
      a₂ b₂ : 𝕜
      ha₂ : LE.le 0 a₂
      hb₂✝ : LE.le 0 b₂
      hab₂ : Eq (HAdd.hAdd a₂ b₂) 1
      hb₂ : LT.lt 0 b₂
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HMul.hMul a₂ a₁) x) (HSMul.hSMul (HAdd.hAdd (HMu …
    -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
  · match_scalars <;> field_simp
                      /-
                        🎉 no goals
                      -/


theorem convexJoin_assoc (s t u : Set E) :
    convexJoin 𝕜 (convexJoin 𝕜 s t) u = convexJoin 𝕜 s (convexJoin 𝕜 t u) := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t u : Set E
    ⊢ Eq (convexJoin 𝕜 (convexJoin 𝕜 s t) u) (convexJoin 𝕜 s (convexJoin 𝕜 t u))
  -/
  refine (convexJoin_assoc_aux _ _ _).antisymm ?_
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t u : Set E
    ⊢ HasSubset.Subset (convexJoin 𝕜 s (convexJoin 𝕜 t u)) (convexJoin 𝕜 (convexJo …
  -/
  simp_rw [convexJoin_comm s, convexJoin_comm _ u]
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t u : Set E
    ⊢ HasSubset.Subset (convexJoin 𝕜 (convexJoin 𝕜 u t) s) (convexJoin 𝕜 u (convex …
  -/
  exact convexJoin_assoc_aux _ _ _
  /-
    🎉 no goals
  -/


theorem convexJoin_left_comm (s t u : Set E) :
    convexJoin 𝕜 s (convexJoin 𝕜 t u) = convexJoin 𝕜 t (convexJoin 𝕜 s u) := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t u : Set E
    ⊢ Eq (convexJoin 𝕜 s (convexJoin 𝕜 t u)) (convexJoin 𝕜 t (convexJoin 𝕜 s u))
  -/
  simp_rw [← convexJoin_assoc, convexJoin_comm]
  /-
    🎉 no goals
  -/


theorem convexJoin_right_comm (s t u : Set E) :
    convexJoin 𝕜 (convexJoin 𝕜 s t) u = convexJoin 𝕜 (convexJoin 𝕜 s u) t := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t u : Set E
    ⊢ Eq (convexJoin 𝕜 (convexJoin 𝕜 s t) u) (convexJoin 𝕜 (convexJoin 𝕜 s u) t)
  -/
  simp_rw [convexJoin_assoc, convexJoin_comm]
  /-
    🎉 no goals
  -/


theorem convexJoin_convexJoin_convexJoin_comm (s t u v : Set E) :
    convexJoin 𝕜 (convexJoin 𝕜 s t) (convexJoin 𝕜 u v) =
      convexJoin 𝕜 (convexJoin 𝕜 s u) (convexJoin 𝕜 t v) := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t u v : Set E
    ⊢ Eq (convexJoin 𝕜 (convexJoin 𝕜 s t) (convexJoin 𝕜 u v)) (convexJoin 𝕜 (conve …
  -/
  simp_rw [← convexJoin_assoc, convexJoin_right_comm]
  /-
    🎉 no goals
  -/

-- Porting note: moved 3 lemmas from below to golf

protected theorem Convex.convexJoin (hs : Convex 𝕜 s) (ht : Convex 𝕜 t) :
    Convex 𝕜 (convexJoin 𝕜 s t) := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Convex 𝕜 s
    ht : Convex 𝕜 t
    ⊢ Convex 𝕜 (convexJoin 𝕜 s t)
  -/
  simp only [Convex, StarConvex, convexJoin, mem_iUnion]
  rintro _ ⟨x₁, hx₁, y₁, hy₁, a₁, b₁, ha₁, hb₁, hab₁, rfl⟩
    _ ⟨x₂, hx₂, y₂, hy₂, a₂, b₂, ha₂, hb₂, hab₂, rfl⟩ p q hp hq hpq
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Convex 𝕜 s
    ht : Convex 𝕜 t
    x₁ : E
    hx₁ : Membership.mem s x₁
    y₁ : E
    hy₁ : Membership.mem t y₁
    a₁ b₁ : 𝕜
    ha₁ : LE.le 0 a₁
    hb₁ : LE.le 0 b₁
    hab₁ : Eq (HAdd.hAdd a₁ b₁) 1
    x₂ : E
    hx₂ : Membership.mem s x₂
    y₂ : E
    hy₂ : Membership.mem t y₂
    a₂ b₂ : 𝕜
    ha₂ : LE.le 0 a₂
    hb₂ : LE.le 0 b₂
    hab₂ : Eq (HAdd.hAdd a₂ b₂) 1
    p q : 𝕜
    hp : LE.le 0 p
    hq : LE.le 0 q
    hpq : Eq (HAdd.hAdd p q) 1
    ⊢ Exists fun i => Exists fun h => Exists fun i_1 => Exists fun i_2 => Membersh …
  -/
  rcases hs.exists_mem_add_smul_eq hx₁ hx₂ (mul_nonneg hp ha₁) (mul_nonneg hq ha₂) with ⟨x, hxs, hx⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Convex 𝕜 s
    ht : Convex 𝕜 t
    x₁ : E
    hx₁ : Membership.mem s x₁
    y₁ : E
    hy₁ : Membership.mem t y₁
    a₁ b₁ : 𝕜
    ha₁ : LE.le 0 a₁
    hb₁ : LE.le 0 b₁
    hab₁ : Eq (HAdd.hAdd a₁ b₁) 1
    x₂ : E
    hx₂ : Membership.mem s x₂
    y₂ : E
    hy₂ : Membership.mem t y₂
    a₂ b₂ : 𝕜
    ha₂ : LE.le 0 a₂
    hb₂ : LE.le 0 b₂
    hab₂ : Eq (HAdd.hAdd a₂ b₂) 1
    p q : 𝕜
    hp : LE.le 0 p
    hq : LE.le 0 q
    hpq : Eq (HAdd.hAdd p q) 1
    x : E
    hxs : Membership.mem s x
    hx : Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul p a₁) (HMul.hMul q a₂)) x) (HAdd.hA …
    ⊢ Exists fun i => Exists fun h => Exists fun i_1 => Exists fun i_2 => Membersh …
  -/
  rcases ht.exists_mem_add_smul_eq hy₁ hy₂ (mul_nonneg hp hb₁) (mul_nonneg hq hb₂) with ⟨y, hyt, hy⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Convex 𝕜 s
    ht : Convex 𝕜 t
    x₁ : E
    hx₁ : Membership.mem s x₁
    y₁ : E
    hy₁ : Membership.mem t y₁
    a₁ b₁ : 𝕜
    ha₁ : LE.le 0 a₁
    hb₁ : LE.le 0 b₁
    hab₁ : Eq (HAdd.hAdd a₁ b₁) 1
    x₂ : E
    hx₂ : Membership.mem s x₂
    y₂ : E
    hy₂ : Membership.mem t y₂
    a₂ b₂ : 𝕜
    ha₂ : LE.le 0 a₂
    hb₂ : LE.le 0 b₂
    hab₂ : Eq (HAdd.hAdd a₂ b₂) 1
    p q : 𝕜
    hp : LE.le 0 p
    hq : LE.le 0 q
    hpq : Eq (HAdd.hAdd p q) 1
    x : E
    hxs : Membership.mem s x
    hx : Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul p a₁) (HMul.hMul q a₂)) x) (HAdd.hA …
    y : E
    hyt : Membership.mem t y
    hy : Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul p b₁) (HMul.hMul q b₂)) y) (HAdd.hA …
    ⊢ Exists fun i => Exists fun h => Exists fun i_1 => Exists fun i_2 => Membersh …
  -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  refine ⟨_, hxs, _, hyt, p * a₁ + q * a₂, p * b₁ + q * b₂, ?_, ?_, ?_, ?_⟩ <;> try positivity
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      𝕜 : Type u_2
      E : Type u_3
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s t : Set E
      hs : Convex 𝕜 s
      ht : Convex 𝕜 t
      x₁ : E
      hx₁ : Membership.mem s x₁
      y₁ : E
      hy₁ : Membership.mem t y₁
      a₁ b₁ : 𝕜
      ha₁ : LE.le 0 a₁
      hb₁ : LE.le 0 b₁
      hab₁ : Eq (HAdd.hAdd a₁ b₁) 1
      x₂ : E
      hx₂ : Membership.mem s x₂
      y₂ : E
      hy₂ : Membership.mem t y₂
      a₂ b₂ : 𝕜
      ha₂ : LE.le 0 a₂
      hb₂ : LE.le 0 b₂
      hab₂ : Eq (HAdd.hAdd a₂ b₂) 1
      p q : 𝕜
      hp : LE.le 0 p
      hq : LE.le 0 q
      hpq : Eq (HAdd.hAdd p q) 1
      x : E
      hxs : Membership.mem s x
      hx : Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul p a₁) (HMul.hMul q a₂)) x) (HAdd.hA …
      y : E
      hyt : Membership.mem t y
      hy : Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul p b₁) (HMul.hMul q b₂)) y) (HAdd.hA …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul p a₁) (HMul.hMul q a₂)) (HAdd.hAdd (HMul …
    -/
  · linear_combination p * hab₁ + q * hab₂ + hpq
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      𝕜 : Type u_2
      E : Type u_3
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s t : Set E
      hs : Convex 𝕜 s
      ht : Convex 𝕜 t
      x₁ : E
      hx₁ : Membership.mem s x₁
      y₁ : E
      hy₁ : Membership.mem t y₁
      a₁ b₁ : 𝕜
      ha₁ : LE.le 0 a₁
      hb₁ : LE.le 0 b₁
      hab₁ : Eq (HAdd.hAdd a₁ b₁) 1
      x₂ : E
      hx₂ : Membership.mem s x₂
      y₂ : E
      hy₂ : Membership.mem t y₂
      a₂ b₂ : 𝕜
      ha₂ : LE.le 0 a₂
      hb₂ : LE.le 0 b₂
      hab₂ : Eq (HAdd.hAdd a₂ b₂) 1
      p q : 𝕜
      hp : LE.le 0 p
      hq : LE.le 0 q
      hpq : Eq (HAdd.hAdd p q) 1
      x : E
      hxs : Membership.mem s x
      hx : Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul p a₁) (HMul.hMul q a₂)) x) (HAdd.hA …
      y : E
      hyt : Membership.mem t y
      hy : Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul p b₁) (HMul.hMul q b₂)) y) (HAdd.hA …
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (HMul.hMul p a₁) (HMul.hMul q a₂)) x)  …
    -/
  · rw [hx, hy]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      𝕜 : Type u_2
      E : Type u_3
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s t : Set E
      hs : Convex 𝕜 s
      ht : Convex 𝕜 t
      x₁ : E
      hx₁ : Membership.mem s x₁
      y₁ : E
      hy₁ : Membership.mem t y₁
      a₁ b₁ : 𝕜
      ha₁ : LE.le 0 a₁
      hb₁ : LE.le 0 b₁
      hab₁ : Eq (HAdd.hAdd a₁ b₁) 1
      x₂ : E
      hx₂ : Membership.mem s x₂
      y₂ : E
      hy₂ : Membership.mem t y₂
      a₂ b₂ : 𝕜
      ha₂ : LE.le 0 a₂
      hb₂ : LE.le 0 b₂
      hab₂ : Eq (HAdd.hAdd a₂ b₂) 1
      p q : 𝕜
      hp : LE.le 0 p
      hq : LE.le 0 q
      hpq : Eq (HAdd.hAdd p q) 1
      x : E
      hxs : Membership.mem s x
      hx : Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul p a₁) (HMul.hMul q a₂)) x) (HAdd.hA …
      y : E
      hyt : Membership.mem t y
      hy : Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul p b₁) (HMul.hMul q b₂)) y) (HAdd.hA …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (HMul.hMul p a₁) x₁) (HSMul.hSMul (HMu …
    -/
    module
    /-
      🎉 no goals
    -/


protected theorem Convex.convexHull_union (hs : Convex 𝕜 s) (ht : Convex 𝕜 t) (hs₀ : s.Nonempty)
    (ht₀ : t.Nonempty) : convexHull 𝕜 (s ∪ t) = convexJoin 𝕜 s t :=
  (convexHull_min (union_subset (subset_convexJoin_left ht₀) <| subset_convexJoin_right hs₀) <|
        hs.convexJoin ht).antisymm <|
    convexJoin_subset_convexHull _ _


theorem convexHull_union (hs : s.Nonempty) (ht : t.Nonempty) :
    convexHull 𝕜 (s ∪ t) = convexJoin 𝕜 (convexHull 𝕜 s) (convexHull 𝕜 t) := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : s.Nonempty
    ht : t.Nonempty
    ⊢ Eq ((convexHull 𝕜) (Union.union s t)) (convexJoin 𝕜 ((convexHull 𝕜) s) ((con …
  -/
  rw [← convexHull_convexHull_union_left, ← convexHull_convexHull_union_right]
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : s.Nonempty
    ht : t.Nonempty
    ⊢ Eq ((convexHull 𝕜) (Union.union ((convexHull 𝕜) s) ((convexHull 𝕜) t))) (con …
  -/
  exact (convex_convexHull 𝕜 s).convexHull_union (convex_convexHull 𝕜 t) hs.convexHull ht.convexHull
  /-
    🎉 no goals
  -/


theorem convexHull_insert (hs : s.Nonempty) :
    convexHull 𝕜 (insert x s) = convexJoin 𝕜 {x} (convexHull 𝕜 s) := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    hs : s.Nonempty
    ⊢ Eq ((convexHull 𝕜) (Insert.insert x s)) (convexJoin 𝕜 (Singleton.singleton x …
  -/
  rw [insert_eq, convexHull_union (singleton_nonempty _) hs, convexHull_singleton]
  /-
    🎉 no goals
  -/


theorem convexJoin_segments (a b c d : E) :
    convexJoin 𝕜 (segment 𝕜 a b) (segment 𝕜 c d) = convexHull 𝕜 {a, b, c, d} := by
  simp_rw [← convexHull_pair, convexHull_insert (insert_nonempty _ _),
    convexHull_insert (singleton_nonempty _), convexJoin_assoc,
    convexHull_singleton]


theorem convexJoin_segment_singleton (a b c : E) :
    convexJoin 𝕜 (segment 𝕜 a b) {c} = convexHull 𝕜 {a, b, c} := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    a b c : E
    ⊢ Eq (convexJoin 𝕜 (segment 𝕜 a b) (Singleton.singleton c)) ((convexHull 𝕜) (I …
  -/
  rw [← pair_eq_singleton, ← convexJoin_segments, segment_same, pair_eq_singleton]
  /-
    🎉 no goals
  -/


theorem convexJoin_singleton_segment (a b c : E) :
    convexJoin 𝕜 {a} (segment 𝕜 b c) = convexHull 𝕜 {a, b, c} := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    a b c : E
    ⊢ Eq (convexJoin 𝕜 (Singleton.singleton a) (segment 𝕜 b c)) ((convexHull 𝕜) (I …
  -/
  rw [← segment_same 𝕜, convexJoin_segments, insert_idem]
  /-
    🎉 no goals
  -/

-- Porting note: moved 3 lemmas up to golf


