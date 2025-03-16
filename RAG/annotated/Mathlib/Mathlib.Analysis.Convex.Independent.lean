/-- An indexed family is said to be convex independent if every point only belongs to convex hulls
of sets containing it. -/
def ConvexIndependent (p : ι → E) : Prop :=
  ∀ (s : Set ι) (x : ι), p x ∈ convexHull 𝕜 (p '' s) → x ∈ s


/-- A family with at most one point is convex independent. -/
theorem Subsingleton.convexIndependent [Subsingleton ι] (p : ι → E) : ConvexIndependent 𝕜 p := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Subsingleton ι
    p : ι → E
    ⊢ ConvexIndependent 𝕜 p
  -/
  intro s x hx
  /-
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Subsingleton ι
    p : ι → E
    s : Set ι
    x : ι
    hx : Membership.mem ((convexHull 𝕜) (Set.image p s)) (p x)
    ⊢ Membership.mem s x
  -/
  have : (convexHull 𝕜 (p '' s)).Nonempty := ⟨p x, hx⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Subsingleton ι
    p : ι → E
    s : Set ι
    x : ι
    hx : Membership.mem ((convexHull 𝕜) (Set.image p s)) (p x)
    this : ((convexHull 𝕜) (Set.image p s)).Nonempty
    ⊢ Membership.mem s x
  -/
  rw [convexHull_nonempty_iff, Set.image_nonempty] at this
  /-
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Subsingleton ι
    p : ι → E
    s : Set ι
    x : ι
    hx : Membership.mem ((convexHull 𝕜) (Set.image p s)) (p x)
    this : s.Nonempty
    ⊢ Membership.mem s x
  -/
  rwa [Subsingleton.mem_iff_nonempty]
  /-
    🎉 no goals
  -/


/-- A convex independent family is injective. -/
protected theorem ConvexIndependent.injective {p : ι → E} (hc : ConvexIndependent 𝕜 p) :
    Function.Injective p := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    hc : ConvexIndependent 𝕜 p
    ⊢ Function.Injective p
  -/
  refine fun i j hij => hc {j} i ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    hc : ConvexIndependent 𝕜 p
    i j : ι
    hij : Eq (p i) (p j)
    ⊢ Membership.mem ((convexHull 𝕜) (Set.image p (Singleton.singleton j))) (p i)
  -/
  rw [hij, Set.image_singleton, convexHull_singleton]
  /-
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    hc : ConvexIndependent 𝕜 p
    i j : ι
    hij : Eq (p i) (p j)
    ⊢ Membership.mem (Singleton.singleton (p j)) (p j)
  -/
  exact Set.mem_singleton _
  /-
    🎉 no goals
  -/


/-- If a family is convex independent, so is any subfamily given by composition of an embedding into
index type with the original family. -/
theorem ConvexIndependent.comp_embedding {ι' : Type*} (f : ι' ↪ ι) {p : ι → E}
    (hc : ConvexIndependent 𝕜 p) : ConvexIndependent 𝕜 (p ∘ f) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    ι' : Type u_4
    f : Function.Embedding ι' ι
    p : ι → E
    hc : ConvexIndependent 𝕜 p
    ⊢ ConvexIndependent 𝕜 (Function.comp p ⇑f)
  -/
  intro s x hx
  /-
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    ι' : Type u_4
    f : Function.Embedding ι' ι
    p : ι → E
    hc : ConvexIndependent 𝕜 p
    s : Set ι'
    x : ι'
    hx : Membership.mem ((convexHull 𝕜) (Set.image (Function.comp p ⇑f) s)) (Funct …
    ⊢ Membership.mem s x
  -/
  rw [← f.injective.mem_set_image]
  /-
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    ι' : Type u_4
    f : Function.Embedding ι' ι
    p : ι → E
    hc : ConvexIndependent 𝕜 p
    s : Set ι'
    x : ι'
    hx : Membership.mem ((convexHull 𝕜) (Set.image (Function.comp p ⇑f) s)) (Funct …
    ⊢ Membership.mem (Set.image (⇑f) s) (f x)
  -/
  exact hc _ _ (by rwa [Set.image_image])
  /-
    🎉 no goals
  -/


/-- If a family is convex independent, so is any subfamily indexed by a subtype of the index type.
-/
protected theorem ConvexIndependent.subtype {p : ι → E} (hc : ConvexIndependent 𝕜 p) (s : Set ι) :
    ConvexIndependent 𝕜 fun i : s => p i :=
  hc.comp_embedding (Embedding.subtype _)


/-- If an indexed family of points is convex independent, so is the corresponding set of points. -/
protected theorem ConvexIndependent.range {p : ι → E} (hc : ConvexIndependent 𝕜 p) :
    ConvexIndependent 𝕜 ((↑) : Set.range p → E) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    hc : ConvexIndependent 𝕜 p
    ⊢ ConvexIndependent 𝕜 Subtype.val
  -/
  let f : Set.range p → ι := fun x => x.property.choose
  /-
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    hc : ConvexIndependent 𝕜 p
    f : ↑(Set.range p) → ι := fun x => Exists.choose ⋯
    ⊢ ConvexIndependent 𝕜 Subtype.val
  -/
  have hf : ∀ x, p (f x) = x := fun x => x.property.choose_spec
  /-
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    hc : ConvexIndependent 𝕜 p
    f : ↑(Set.range p) → ι := fun x => Exists.choose ⋯
    hf : ∀ (x : ↑(Set.range p)), Eq (p (f x)) ↑x
    ⊢ ConvexIndependent 𝕜 Subtype.val
  -/
  let fe : Set.range p ↪ ι := ⟨f, fun x₁ x₂ he => Subtype.ext (hf x₁ ▸ hf x₂ ▸ he ▸ rfl)⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    hc : ConvexIndependent 𝕜 p
    f : ↑(Set.range p) → ι := fun x => Exists.choose ⋯
    hf : ∀ (x : ↑(Set.range p)), Eq (p (f x)) ↑x
    fe : Function.Embedding (↑(Set.range p)) ι := { toFun := f, inj' := ⋯ }
    ⊢ ConvexIndependent 𝕜 Subtype.val
  -/
  convert hc.comp_embedding fe
  /-
    case h.e'_7
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    hc : ConvexIndependent 𝕜 p
    f : ↑(Set.range p) → ι := fun x => Exists.choose ⋯
    hf : ∀ (x : ↑(Set.range p)), Eq (p (f x)) ↑x
    fe : Function.Embedding (↑(Set.range p)) ι := { toFun := f, inj' := ⋯ }
    ⊢ Eq Subtype.val (Function.comp p ⇑fe)
  -/
  ext
  /-
    case h.e'_7.h
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    hc : ConvexIndependent 𝕜 p
    f : ↑(Set.range p) → ι := fun x => Exists.choose ⋯
    hf : ∀ (x : ↑(Set.range p)), Eq (p (f x)) ↑x
    fe : Function.Embedding (↑(Set.range p)) ι := { toFun := f, inj' := ⋯ }
    x✝ : Subtype fun x => Membership.mem (Set.range p) x
    ⊢ Eq (↑x✝) (Function.comp p (⇑fe) x✝)
  -/
  rw [Embedding.coeFn_mk, comp_apply, hf]
  /-
    🎉 no goals
  -/


/-- A subset of a convex independent set of points is convex independent as well. -/
protected theorem ConvexIndependent.mono {s t : Set E} (hc : ConvexIndependent 𝕜 ((↑) : t → E))
    (hs : s ⊆ t) : ConvexIndependent 𝕜 ((↑) : s → E) :=
  hc.comp_embedding (s.embeddingOfSubset t hs)


/-- The range of an injective indexed family of points is convex independent iff that family is. -/
theorem Function.Injective.convexIndependent_iff_set {p : ι → E} (hi : Function.Injective p) :
    ConvexIndependent 𝕜 ((↑) : Set.range p → E) ↔ ConvexIndependent 𝕜 p :=
  ⟨fun hc =>
    hc.comp_embedding
      (⟨fun i => ⟨p i, Set.mem_range_self _⟩, fun _ _ h => hi (Subtype.mk_eq_mk.1 h)⟩ :
        ι ↪ Set.range p),
    ConvexIndependent.range⟩


/-- If a family is convex independent, a point in the family is in the convex hull of some of the
points given by a subset of the index type if and only if the point's index is in this subset. -/
@[simp]
protected theorem ConvexIndependent.mem_convexHull_iff {p : ι → E} (hc : ConvexIndependent 𝕜 p)
    (s : Set ι) (i : ι) : p i ∈ convexHull 𝕜 (p '' s) ↔ i ∈ s :=
  ⟨hc _ _, fun hi => subset_convexHull 𝕜 _ (Set.mem_image_of_mem p hi)⟩


/-- If a family is convex independent, a point in the family is not in the convex hull of the other
points. See `convexIndependent_set_iff_not_mem_convexHull_diff` for the `Set` version. -/
theorem convexIndependent_iff_not_mem_convexHull_diff {p : ι → E} :
    ConvexIndependent 𝕜 p ↔ ∀ i s, p i ∉ convexHull 𝕜 (p '' (s \ {i})) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    ⊢ Iff (ConvexIndependent 𝕜 p) (∀ (i : ι) (s : Set ι), Not (Membership.mem ((co …
  -/
  refine ⟨fun hc i s h => ?_, fun h s i hi => ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      ι : Type u_3
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : ι → E
      hc : ConvexIndependent 𝕜 p
      i : ι
      s : Set ι
      h : Membership.mem ((convexHull 𝕜) (Set.image p (SDiff.sdiff s (Singleton.sing …
      ⊢ False
    -/
  · rw [hc.mem_convexHull_iff] at h
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      ι : Type u_3
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : ι → E
      hc : ConvexIndependent 𝕜 p
      i : ι
      s : Set ι
      h : Membership.mem (SDiff.sdiff s (Singleton.singleton i)) i
      ⊢ False
    -/
    exact h.2 (Set.mem_singleton _)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      ι : Type u_3
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : ι → E
      h : ∀ (i : ι) (s : Set ι), Not (Membership.mem ((convexHull 𝕜) (Set.image p (S …
      s : Set ι
      i : ι
      hi : Membership.mem ((convexHull 𝕜) (Set.image p s)) (p i)
      ⊢ Membership.mem s i
    -/
  · by_contra H
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      ι : Type u_3
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : ι → E
      h : ∀ (i : ι) (s : Set ι), Not (Membership.mem ((convexHull 𝕜) (Set.image p (S …
      s : Set ι
      i : ι
      hi : Membership.mem ((convexHull 𝕜) (Set.image p s)) (p i)
      H : Not (Membership.mem s i)
      ⊢ False
    -/
    refine h i s ?_
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      ι : Type u_3
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : ι → E
      h : ∀ (i : ι) (s : Set ι), Not (Membership.mem ((convexHull 𝕜) (Set.image p (S …
      s : Set ι
      i : ι
      hi : Membership.mem ((convexHull 𝕜) (Set.image p s)) (p i)
      H : Not (Membership.mem s i)
      ⊢ Membership.mem ((convexHull 𝕜) (Set.image p (SDiff.sdiff s (Singleton.single …
    -/
    rw [Set.diff_singleton_eq_self H]
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      ι : Type u_3
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : ι → E
      h : ∀ (i : ι) (s : Set ι), Not (Membership.mem ((convexHull 𝕜) (Set.image p (S …
      s : Set ι
      i : ι
      hi : Membership.mem ((convexHull 𝕜) (Set.image p s)) (p i)
      H : Not (Membership.mem s i)
      ⊢ Membership.mem ((convexHull 𝕜) (Set.image p s)) (p i)
    -/
    exact hi
    /-
      🎉 no goals
    -/


theorem convexIndependent_set_iff_inter_convexHull_subset {s : Set E} :
    ConvexIndependent 𝕜 ((↑) : s → E) ↔ ∀ t, t ⊆ s → s ∩ convexHull 𝕜 t ⊆ t := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ Iff (ConvexIndependent 𝕜 Subtype.val) (∀ (t : Set E), HasSubset.Subset t s → …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      ⊢ ConvexIndependent 𝕜 Subtype.val → ∀ (t : Set E), HasSubset.Subset t s → HasS …
    -/
  · rintro hc t h x ⟨hxs, hxt⟩
    /-
      case mp.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hc : ConvexIndependent 𝕜 Subtype.val
      t : Set E
      h : HasSubset.Subset t s
      x : E
      hxs : Membership.mem s x
      hxt : Membership.mem ((convexHull 𝕜) t) x
      ⊢ Membership.mem t x
    -/
    refine hc { x | ↑x ∈ t } ⟨x, hxs⟩ ?_
    /-
      case mp.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hc : ConvexIndependent 𝕜 Subtype.val
      t : Set E
      h : HasSubset.Subset t s
      x : E
      hxs : Membership.mem s x
      hxt : Membership.mem ((convexHull 𝕜) t) x
      ⊢ Membership.mem ((convexHull 𝕜) (Set.image Subtype.val (setOf fun x => Member …
    -/
    rw [Subtype.coe_image_of_subset h]
    /-
      case mp.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hc : ConvexIndependent 𝕜 Subtype.val
      t : Set E
      h : HasSubset.Subset t s
      x : E
      hxs : Membership.mem s x
      hxt : Membership.mem ((convexHull 𝕜) t) x
      ⊢ Membership.mem ((convexHull 𝕜) t) ↑⟨x, hxs⟩
    -/
    exact hxt
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      ⊢ (∀ (t : Set E), HasSubset.Subset t s → HasSubset.Subset (Inter.inter s ((con …
    -/
  · intro hc t x h
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hc : ∀ (t : Set E), HasSubset.Subset t s → HasSubset.Subset (Inter.inter s ((c …
      t : Set (Subtype fun x => Membership.mem s x)
      x : Subtype fun x => Membership.mem s x
      h : Membership.mem ((convexHull 𝕜) (Set.image Subtype.val t)) ↑x
      ⊢ Membership.mem t x
    -/
    rw [← Subtype.coe_injective.mem_set_image]
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hc : ∀ (t : Set E), HasSubset.Subset t s → HasSubset.Subset (Inter.inter s ((c …
      t : Set (Subtype fun x => Membership.mem s x)
      x : Subtype fun x => Membership.mem s x
      h : Membership.mem ((convexHull 𝕜) (Set.image Subtype.val t)) ↑x
      ⊢ Membership.mem (Set.image (fun a => ↑a) t) ↑x
    -/
    exact hc (t.image ((↑) : s → E)) (Subtype.coe_image_subset s t) ⟨x.prop, h⟩
    /-
      🎉 no goals
    -/


/-- If a set is convex independent, a point in the set is not in the convex hull of the other
points. See `convexIndependent_iff_not_mem_convexHull_diff` for the indexed family version. -/
theorem convexIndependent_set_iff_not_mem_convexHull_diff {s : Set E} :
    ConvexIndependent 𝕜 ((↑) : s → E) ↔ ∀ x ∈ s, x ∉ convexHull 𝕜 (s \ {x}) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ Iff (ConvexIndependent 𝕜 Subtype.val) (∀ (x : E), Membership.mem s x → Not ( …
  -/
  rw [convexIndependent_set_iff_inter_convexHull_subset]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ Iff (∀ (t : Set E), HasSubset.Subset t s → HasSubset.Subset (Inter.inter s ( …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      ⊢ (∀ (t : Set E), HasSubset.Subset t s → HasSubset.Subset (Inter.inter s ((con …
    -/
  · rintro hs x hxs hx
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hs : ∀ (t : Set E), HasSubset.Subset t s → HasSubset.Subset (Inter.inter s ((c …
      x : E
      hxs : Membership.mem s x
      hx : Membership.mem ((convexHull 𝕜) (SDiff.sdiff s (Singleton.singleton x))) x
      ⊢ False
    -/
    exact (hs _ Set.diff_subset ⟨hxs, hx⟩).2 (Set.mem_singleton _)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      ⊢ (∀ (x : E), Membership.mem s x → Not (Membership.mem ((convexHull 𝕜) (SDiff. …
    -/
  · rintro hs t ht x ⟨hxs, hxt⟩
    /-
      case mpr.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hs : ∀ (x : E), Membership.mem s x → Not (Membership.mem ((convexHull 𝕜) (SDif …
      t : Set E
      ht : HasSubset.Subset t s
      x : E
      hxs : Membership.mem s x
      hxt : Membership.mem ((convexHull 𝕜) t) x
      ⊢ Membership.mem t x
    -/
    by_contra h
    /-
      case mpr.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hs : ∀ (x : E), Membership.mem s x → Not (Membership.mem ((convexHull 𝕜) (SDif …
      t : Set E
      ht : HasSubset.Subset t s
      x : E
      hxs : Membership.mem s x
      hxt : Membership.mem ((convexHull 𝕜) t) x
      h : Not (Membership.mem t x)
      ⊢ False
    -/
    exact hs _ hxs (convexHull_mono (Set.subset_diff_singleton ht h) hxt)
    /-
      🎉 no goals
    -/


open scoped Classical in
/-- To check convex independence, one only has to check finsets thanks to Carathéodory's theorem. -/
theorem convexIndependent_iff_finset {p : ι → E} :
    ConvexIndependent 𝕜 p ↔
      ∀ (s : Finset ι) (x : ι), p x ∈ convexHull 𝕜 (s.image p : Set E) → x ∈ s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    ⊢ Iff (ConvexIndependent 𝕜 p) (∀ (s : Finset ι) (x : ι), Membership.mem ((conv …
  -/
  refine ⟨fun hc s x hx => hc s x ?_, fun h s x hx => ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      ι : Type u_3
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : ι → E
      hc : ConvexIndependent 𝕜 p
      s : Finset ι
      x : ι
      hx : Membership.mem ((convexHull 𝕜) ↑(Finset.image p s)) (p x)
      ⊢ Membership.mem ((convexHull 𝕜) (Set.image p ↑s)) (p x)
    -/
  · rwa [Finset.coe_image] at hx
    /-
      🎉 no goals
    -/
  have hp : Injective p := by
    rintro a b hab
    rw [← mem_singleton]
    refine h {b} a ?_
    rw [hab, image_singleton, coe_singleton, convexHull_singleton]
    exact Set.mem_singleton _
  /-
    case refine_2
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    h : ∀ (s : Finset ι) (x : ι), Membership.mem ((convexHull 𝕜) ↑(Finset.image p  …
    s : Set ι
    x : ι
    hx : Membership.mem ((convexHull 𝕜) (Set.image p s)) (p x)
    hp : Function.Injective p
    ⊢ Membership.mem s x
  -/
  rw [convexHull_eq_union_convexHull_finite_subsets] at hx
  /-
    case refine_2
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    h : ∀ (s : Finset ι) (x : ι), Membership.mem ((convexHull 𝕜) ↑(Finset.image p  …
    s : Set ι
    x : ι
    hx : Membership.mem (Set.iUnion fun t => Set.iUnion fun x => (convexHull 𝕜) ↑t …
    hp : Function.Injective p
    ⊢ Membership.mem s x
  -/
  simp_rw [Set.mem_iUnion] at hx
  /-
    case refine_2
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    h : ∀ (s : Finset ι) (x : ι), Membership.mem ((convexHull 𝕜) ↑(Finset.image p  …
    s : Set ι
    x : ι
    hp : Function.Injective p
    hx : Exists fun i => Exists fun i_1 => Membership.mem ((convexHull 𝕜) ↑i) (p x)
    ⊢ Membership.mem s x
  -/
  obtain ⟨t, ht, hx⟩ := hx
  /-
    case refine_2.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    h : ∀ (s : Finset ι) (x : ι), Membership.mem ((convexHull 𝕜) ↑(Finset.image p  …
    s : Set ι
    x : ι
    hp : Function.Injective p
    t : Finset E
    ht : HasSubset.Subset (↑t) (Set.image p s)
    hx : Membership.mem ((convexHull 𝕜) ↑t) (p x)
    ⊢ Membership.mem s x
  -/
  rw [← hp.mem_set_image]
  /-
    case refine_2.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    h : ∀ (s : Finset ι) (x : ι), Membership.mem ((convexHull 𝕜) ↑(Finset.image p  …
    s : Set ι
    x : ι
    hp : Function.Injective p
    t : Finset E
    ht : HasSubset.Subset (↑t) (Set.image p s)
    hx : Membership.mem ((convexHull 𝕜) ↑t) (p x)
    ⊢ Membership.mem (Set.image p s) (p x)
  -/
  refine ht ?_
  /-
    case refine_2.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    h : ∀ (s : Finset ι) (x : ι), Membership.mem ((convexHull 𝕜) ↑(Finset.image p  …
    s : Set ι
    x : ι
    hp : Function.Injective p
    t : Finset E
    ht : HasSubset.Subset (↑t) (Set.image p s)
    hx : Membership.mem ((convexHull 𝕜) ↑t) (p x)
    ⊢ Membership.mem (↑t) (p x)
  -/
  suffices x ∈ t.preimage p hp.injOn by rwa [mem_preimage, ← mem_coe] at this
  /-
    case refine_2.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    h : ∀ (s : Finset ι) (x : ι), Membership.mem ((convexHull 𝕜) ↑(Finset.image p  …
    s : Set ι
    x : ι
    hp : Function.Injective p
    t : Finset E
    ht : HasSubset.Subset (↑t) (Set.image p s)
    hx : Membership.mem ((convexHull 𝕜) ↑t) (p x)
    ⊢ Membership.mem (t.preimage p ⋯) x
  -/
  refine h _ x ?_
  /-
    case refine_2.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    h : ∀ (s : Finset ι) (x : ι), Membership.mem ((convexHull 𝕜) ↑(Finset.image p  …
    s : Set ι
    x : ι
    hp : Function.Injective p
    t : Finset E
    ht : HasSubset.Subset (↑t) (Set.image p s)
    hx : Membership.mem ((convexHull 𝕜) ↑t) (p x)
    ⊢ Membership.mem ((convexHull 𝕜) ↑(Finset.image p (t.preimage p ⋯))) (p x)
  -/
  rwa [t.image_preimage p hp.injOn, filter_true_of_mem]
  /-
    case refine_2.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    ι : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → E
    h : ∀ (s : Finset ι) (x : ι), Membership.mem ((convexHull 𝕜) ↑(Finset.image p  …
    s : Set ι
    x : ι
    hp : Function.Injective p
    t : Finset E
    ht : HasSubset.Subset (↑t) (Set.image p s)
    hx : Membership.mem ((convexHull 𝕜) ↑t) (p x)
    ⊢ ∀ (x : E), Membership.mem t x → Membership.mem (Set.range p) x
  -/
  exact fun y hy => s.image_subset_range p (ht <| mem_coe.2 hy)
  /-
    🎉 no goals
  -/


theorem Convex.convexIndependent_extremePoints (hs : Convex 𝕜 s) :
    ConvexIndependent 𝕜 ((↑) : s.extremePoints 𝕜 → E) :=
  convexIndependent_set_iff_not_mem_convexHull_diff.2 fun _ hx h =>
    (extremePoints_convexHull_subset
          (inter_extremePoints_subset_extremePoints_of_subset
            (convexHull_min (Set.diff_subset.trans extremePoints_subset) hs) ⟨h, hx⟩)).2
      (Set.mem_singleton _)


