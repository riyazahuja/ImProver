/-- A set is strictly convex if the open segment between any two distinct points lies is in its
interior. This basically means "convex and not flat on the boundary". -/
def StrictConvex : Prop :=
  s.Pairwise fun x y => ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b → a + b = 1 → a • x + b • y ∈ interior s


theorem strictConvex_iff_openSegment_subset :
    StrictConvex 𝕜 s ↔ s.Pairwise fun x y => openSegment 𝕜 x y ⊆ interior s :=
  forall₅_congr fun _ _ _ _ _ => (openSegment_subset_iff 𝕜).symm


theorem StrictConvex.openSegment_subset (hs : StrictConvex 𝕜 s) (hx : x ∈ s) (hy : y ∈ s)
    (h : x ≠ y) : openSegment 𝕜 x y ⊆ interior s :=
  strictConvex_iff_openSegment_subset.1 hs hx hy h


theorem strictConvex_empty : StrictConvex 𝕜 (∅ : Set E) :=
  pairwise_empty _


theorem strictConvex_univ : StrictConvex 𝕜 (univ : Set E) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    ⊢ StrictConvex 𝕜 Set.univ
  -/
  intro x _ y _ _ a b _ _ _
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    x : E
    a✝⁵ : Membership.mem Set.univ x
    y : E
    a✝⁴ : Membership.mem Set.univ y
    a✝³ : Ne x y
    a b : 𝕜
    a✝² : LT.lt 0 a
    a✝¹ : LT.lt 0 b
    a✝ : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (interior Set.univ) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul …
  -/
  rw [interior_univ]
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    x : E
    a✝⁵ : Membership.mem Set.univ x
    y : E
    a✝⁴ : Membership.mem Set.univ y
    a✝³ : Ne x y
    a b : 𝕜
    a✝² : LT.lt 0 a
    a✝¹ : LT.lt 0 b
    a✝ : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem Set.univ (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
  -/
  exact mem_univ _
  /-
    🎉 no goals
  -/


protected nonrec theorem StrictConvex.eq (hs : StrictConvex 𝕜 s) (hx : x ∈ s) (hy : y ∈ s)
    (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) (h : a • x + b • y ∉ interior s) : x = y :=
  hs.eq hx hy fun H => h <| H ha hb hab


protected theorem StrictConvex.inter {t : Set E} (hs : StrictConvex 𝕜 s) (ht : StrictConvex 𝕜 t) :
    StrictConvex 𝕜 (s ∩ t) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    s t : Set E
    hs : StrictConvex 𝕜 s
    ht : StrictConvex 𝕜 t
    ⊢ StrictConvex 𝕜 (Inter.inter s t)
  -/
  intro x hx y hy hxy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    s t : Set E
    hs : StrictConvex 𝕜 s
    ht : StrictConvex 𝕜 t
    x : E
    hx : Membership.mem (Inter.inter s t) x
    y : E
    hy : Membership.mem (Inter.inter s t) y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (interior (Inter.inter s t)) (HAdd.hAdd (HSMul.hSMul a x) (HS …
  -/
  rw [interior_inter]
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    s t : Set E
    hs : StrictConvex 𝕜 s
    ht : StrictConvex 𝕜 t
    x : E
    hx : Membership.mem (Inter.inter s t) x
    y : E
    hy : Membership.mem (Inter.inter s t) y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Inter.inter (interior s) (interior t)) (HAdd.hAdd (HSMul.hSM …
  -/
  exact ⟨hs hx.1 hy.1 hxy ha hb hab, ht hx.2 hy.2 hxy ha hb hab⟩
  /-
    🎉 no goals
  -/


theorem Directed.strictConvex_iUnion {ι : Sort*} {s : ι → Set E} (hdir : Directed (· ⊆ ·) s)
    (hs : ∀ ⦃i : ι⦄, StrictConvex 𝕜 (s i)) : StrictConvex 𝕜 (⋃ i, s i) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    ι : Sort u_6
    s : ι → Set E
    hdir : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    hs : ∀ ⦃i : ι⦄, StrictConvex 𝕜 (s i)
    ⊢ StrictConvex 𝕜 (Set.iUnion fun i => s i)
  -/
  rintro x hx y hy hxy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    ι : Sort u_6
    s : ι → Set E
    hdir : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    hs : ∀ ⦃i : ι⦄, StrictConvex 𝕜 (s i)
    x : E
    hx : Membership.mem (Set.iUnion fun i => s i) x
    y : E
    hy : Membership.mem (Set.iUnion fun i => s i) y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (interior (Set.iUnion fun i => s i)) (HAdd.hAdd (HSMul.hSMul  …
  -/
  rw [mem_iUnion] at hx hy
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    ι : Sort u_6
    s : ι → Set E
    hdir : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    hs : ∀ ⦃i : ι⦄, StrictConvex 𝕜 (s i)
    x : E
    hx : Exists fun i => Membership.mem (s i) x
    y : E
    hy : Exists fun i => Membership.mem (s i) y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (interior (Set.iUnion fun i => s i)) (HAdd.hAdd (HSMul.hSMul  …
  -/
  obtain ⟨i, hx⟩ := hx
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    ι : Sort u_6
    s : ι → Set E
    hdir : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    hs : ∀ ⦃i : ι⦄, StrictConvex 𝕜 (s i)
    x y : E
    hy : Exists fun i => Membership.mem (s i) y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    i : ι
    hx : Membership.mem (s i) x
    ⊢ Membership.mem (interior (Set.iUnion fun i => s i)) (HAdd.hAdd (HSMul.hSMul  …
  -/
  obtain ⟨j, hy⟩ := hy
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    ι : Sort u_6
    s : ι → Set E
    hdir : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    hs : ∀ ⦃i : ι⦄, StrictConvex 𝕜 (s i)
    x y : E
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    i : ι
    hx : Membership.mem (s i) x
    j : ι
    hy : Membership.mem (s j) y
    ⊢ Membership.mem (interior (Set.iUnion fun i => s i)) (HAdd.hAdd (HSMul.hSMul  …
  -/
  obtain ⟨k, hik, hjk⟩ := hdir i j
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    ι : Sort u_6
    s : ι → Set E
    hdir : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    hs : ∀ ⦃i : ι⦄, StrictConvex 𝕜 (s i)
    x y : E
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    i : ι
    hx : Membership.mem (s i) x
    j : ι
    hy : Membership.mem (s j) y
    k : ι
    hik : HasSubset.Subset (s i) (s k)
    hjk : HasSubset.Subset (s j) (s k)
    ⊢ Membership.mem (interior (Set.iUnion fun i => s i)) (HAdd.hAdd (HSMul.hSMul  …
  -/
  exact interior_mono (subset_iUnion s k) (hs (hik hx) (hjk hy) hxy ha hb hab)
  /-
    🎉 no goals
  -/


theorem DirectedOn.strictConvex_sUnion {S : Set (Set E)} (hdir : DirectedOn (· ⊆ ·) S)
    (hS : ∀ s ∈ S, StrictConvex 𝕜 s) : StrictConvex 𝕜 (⋃₀ S) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    S : Set (Set E)
    hdir : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) S
    hS : ∀ (s : Set E), Membership.mem S s → StrictConvex 𝕜 s
    ⊢ StrictConvex 𝕜 S.sUnion
  -/
  rw [sUnion_eq_iUnion]
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    S : Set (Set E)
    hdir : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) S
    hS : ∀ (s : Set E), Membership.mem S s → StrictConvex 𝕜 s
    ⊢ StrictConvex 𝕜 (Set.iUnion fun i => ↑i)
  -/
  exact (directedOn_iff_directed.1 hdir).strictConvex_iUnion fun s => hS _ s.2
  /-
    🎉 no goals
  -/


protected theorem StrictConvex.convex (hs : StrictConvex 𝕜 s) : Convex 𝕜 s :=
  convex_iff_pairwise_pos.2 fun _ hx _ hy hxy _ _ ha hb hab =>
    interior_subset <| hs hx hy hxy ha hb hab


/-- An open convex set is strictly convex. -/
protected theorem Convex.strictConvex_of_isOpen (h : IsOpen s) (hs : Convex 𝕜 s) :
    StrictConvex 𝕜 s :=
  fun _ hx _ hy _ _ _ ha hb hab => h.interior_eq.symm ▸ hs hx hy ha.le hb.le hab


theorem IsOpen.strictConvex_iff (h : IsOpen s) : StrictConvex 𝕜 s ↔ Convex 𝕜 s :=
  ⟨StrictConvex.convex, Convex.strictConvex_of_isOpen h⟩


theorem strictConvex_singleton (c : E) : StrictConvex 𝕜 ({c} : Set E) :=
  pairwise_singleton _ _


theorem Set.Subsingleton.strictConvex (hs : s.Subsingleton) : StrictConvex 𝕜 s :=
  hs.pairwise _


theorem StrictConvex.linear_image [Semiring 𝕝] [Module 𝕝 E] [Module 𝕝 F]
    [LinearMap.CompatibleSMul E F 𝕜 𝕝] (hs : StrictConvex 𝕜 s) (f : E →ₗ[𝕝] F) (hf : IsOpenMap f) :
    StrictConvex 𝕜 (f '' s) := by
  /-
    𝕜 : Type u_1
    𝕝 : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : OrderedSemiring 𝕜
    inst✝⁹ : TopologicalSpace E
    inst✝⁸ : TopologicalSpace F
    inst✝⁷ : AddCommMonoid E
    inst✝⁶ : AddCommMonoid F
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : Module 𝕜 F
    s : Set E
    inst✝³ : Semiring 𝕝
    inst✝² : Module 𝕝 E
    inst✝¹ : Module 𝕝 F
    inst✝ : LinearMap.CompatibleSMul E F 𝕜 𝕝
    hs : StrictConvex 𝕜 s
    f : LinearMap (RingHom.id 𝕝) E F
    hf : IsOpenMap ⇑f
    ⊢ StrictConvex 𝕜 (Set.image (⇑f) s)
  -/
  rintro _ ⟨x, hx, rfl⟩ _ ⟨y, hy, rfl⟩ hxy a b ha hb hab
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    𝕝 : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : OrderedSemiring 𝕜
    inst✝⁹ : TopologicalSpace E
    inst✝⁸ : TopologicalSpace F
    inst✝⁷ : AddCommMonoid E
    inst✝⁶ : AddCommMonoid F
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : Module 𝕜 F
    s : Set E
    inst✝³ : Semiring 𝕝
    inst✝² : Module 𝕝 E
    inst✝¹ : Module 𝕝 F
    inst✝ : LinearMap.CompatibleSMul E F 𝕜 𝕝
    hs : StrictConvex 𝕜 s
    f : LinearMap (RingHom.id 𝕝) E F
    hf : IsOpenMap ⇑f
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    hxy : Ne (f x) (f y)
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (interior (Set.image (⇑f) s)) (HAdd.hAdd (HSMul.hSMul a (f x) …
  -/
  refine hf.image_interior_subset _ ⟨a • x + b • y, hs hx hy (ne_of_apply_ne _ hxy) ha hb hab, ?_⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    𝕝 : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : OrderedSemiring 𝕜
    inst✝⁹ : TopologicalSpace E
    inst✝⁸ : TopologicalSpace F
    inst✝⁷ : AddCommMonoid E
    inst✝⁶ : AddCommMonoid F
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : Module 𝕜 F
    s : Set E
    inst✝³ : Semiring 𝕝
    inst✝² : Module 𝕝 E
    inst✝¹ : Module 𝕝 F
    inst✝ : LinearMap.CompatibleSMul E F 𝕜 𝕝
    hs : StrictConvex 𝕜 s
    f : LinearMap (RingHom.id 𝕝) E F
    hf : IsOpenMap ⇑f
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    hxy : Ne (f x) (f y)
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Eq (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul.hSM …
  -/
  rw [map_add, f.map_smul_of_tower a, f.map_smul_of_tower b]
  /-
    🎉 no goals
  -/


theorem StrictConvex.is_linear_image (hs : StrictConvex 𝕜 s) {f : E → F} (h : IsLinearMap 𝕜 f)
    (hf : IsOpenMap f) : StrictConvex 𝕜 (f '' s) :=
  hs.linear_image (h.mk' f) hf


theorem StrictConvex.linear_preimage {s : Set F} (hs : StrictConvex 𝕜 s) (f : E →ₗ[𝕜] F)
    (hf : Continuous f) (hfinj : Injective f) : StrictConvex 𝕜 (s.preimage f) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁶ : OrderedSemiring 𝕜
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set F
    hs : StrictConvex 𝕜 s
    f : LinearMap (RingHom.id 𝕜) E F
    hf : Continuous ⇑f
    hfinj : Function.Injective ⇑f
    ⊢ StrictConvex 𝕜 (Set.preimage (⇑f) s)
  -/
  intro x hx y hy hxy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁶ : OrderedSemiring 𝕜
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set F
    hs : StrictConvex 𝕜 s
    f : LinearMap (RingHom.id 𝕜) E F
    hf : Continuous ⇑f
    hfinj : Function.Injective ⇑f
    x : E
    hx : Membership.mem (Set.preimage (⇑f) s) x
    y : E
    hy : Membership.mem (Set.preimage (⇑f) s) y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (interior (Set.preimage (⇑f) s)) (HAdd.hAdd (HSMul.hSMul a x) …
  -/
  refine preimage_interior_subset_interior_preimage hf ?_
  /-
    𝕜 : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁶ : OrderedSemiring 𝕜
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set F
    hs : StrictConvex 𝕜 s
    f : LinearMap (RingHom.id 𝕜) E F
    hf : Continuous ⇑f
    hfinj : Function.Injective ⇑f
    x : E
    hx : Membership.mem (Set.preimage (⇑f) s) x
    y : E
    hy : Membership.mem (Set.preimage (⇑f) s) y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.preimage (⇑f) (interior s)) (HAdd.hAdd (HSMul.hSMul a x) …
  -/
  rw [mem_preimage, f.map_add, f.map_smul, f.map_smul]
  /-
    𝕜 : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁶ : OrderedSemiring 𝕜
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set F
    hs : StrictConvex 𝕜 s
    f : LinearMap (RingHom.id 𝕜) E F
    hf : Continuous ⇑f
    hfinj : Function.Injective ⇑f
    x : E
    hx : Membership.mem (Set.preimage (⇑f) s) x
    y : E
    hy : Membership.mem (Set.preimage (⇑f) s) y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (interior s) (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b  …
  -/
  exact hs hx hy (hfinj.ne hxy) ha hb hab
  /-
    🎉 no goals
  -/


theorem StrictConvex.is_linear_preimage {s : Set F} (hs : StrictConvex 𝕜 s) {f : E → F}
    (h : IsLinearMap 𝕜 f) (hf : Continuous f) (hfinj : Injective f) :
    StrictConvex 𝕜 (s.preimage f) :=
  hs.linear_preimage (h.mk' f) hf hfinj


protected theorem Set.OrdConnected.strictConvex {s : Set β} (hs : OrdConnected s) :
    StrictConvex 𝕜 s := by
  /-
    𝕜 : Type u_1
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : TopologicalSpace β
    inst✝³ : LinearOrderedCancelAddCommMonoid β
    inst✝² : OrderTopology β
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set β
    hs : s.OrdConnected
    ⊢ StrictConvex 𝕜 s
  -/
  refine strictConvex_iff_openSegment_subset.2 fun x hx y hy hxy => ?_
  /-
    𝕜 : Type u_1
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : TopologicalSpace β
    inst✝³ : LinearOrderedCancelAddCommMonoid β
    inst✝² : OrderTopology β
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set β
    hs : s.OrdConnected
    x : β
    hx : Membership.mem s x
    y : β
    hy : Membership.mem s y
    hxy : Ne x y
    ⊢ (fun x y => HasSubset.Subset (openSegment 𝕜 x y) (interior s)) x y
  -/
  cases' hxy.lt_or_lt with hlt hlt <;> [skip; rw [openSegment_symm]] <;>
    exact
      (openSegment_subset_Ioo hlt).trans
        (isOpen_Ioo.subset_interior_iff.2 <| Ioo_subset_Icc_self.trans <| hs.out ‹_› ‹_›)


theorem strictConvex_Iic (r : β) : StrictConvex 𝕜 (Iic r) :=
  ordConnected_Iic.strictConvex


theorem strictConvex_Ici (r : β) : StrictConvex 𝕜 (Ici r) :=
  ordConnected_Ici.strictConvex


theorem strictConvex_Iio (r : β) : StrictConvex 𝕜 (Iio r) :=
  ordConnected_Iio.strictConvex


theorem strictConvex_Ioi (r : β) : StrictConvex 𝕜 (Ioi r) :=
  ordConnected_Ioi.strictConvex


theorem strictConvex_Icc (r s : β) : StrictConvex 𝕜 (Icc r s) :=
  ordConnected_Icc.strictConvex


theorem strictConvex_Ioo (r s : β) : StrictConvex 𝕜 (Ioo r s) :=
  ordConnected_Ioo.strictConvex


theorem strictConvex_Ico (r s : β) : StrictConvex 𝕜 (Ico r s) :=
  ordConnected_Ico.strictConvex


theorem strictConvex_Ioc (r s : β) : StrictConvex 𝕜 (Ioc r s) :=
  ordConnected_Ioc.strictConvex


theorem strictConvex_uIcc (r s : β) : StrictConvex 𝕜 (uIcc r s) :=
  strictConvex_Icc _ _


theorem strictConvex_uIoc (r s : β) : StrictConvex 𝕜 (uIoc r s) :=
  strictConvex_Ioc _ _


/-- The translation of a strictly convex set is also strictly convex. -/
theorem StrictConvex.preimage_add_right (hs : StrictConvex 𝕜 s) (z : E) :
    StrictConvex 𝕜 ((fun x => z + x) ⁻¹' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : TopologicalSpace E
    inst✝² : AddCancelCommMonoid E
    inst✝¹ : ContinuousAdd E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : StrictConvex 𝕜 s
    z : E
    ⊢ StrictConvex 𝕜 (Set.preimage (fun x => HAdd.hAdd z x) s)
  -/
  intro x hx y hy hxy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : TopologicalSpace E
    inst✝² : AddCancelCommMonoid E
    inst✝¹ : ContinuousAdd E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : StrictConvex 𝕜 s
    z x : E
    hx : Membership.mem (Set.preimage (fun x => HAdd.hAdd z x) s) x
    y : E
    hy : Membership.mem (Set.preimage (fun x => HAdd.hAdd z x) s) y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (interior (Set.preimage (fun x => HAdd.hAdd z x) s)) (HAdd.hA …
  -/
  refine preimage_interior_subset_interior_preimage (continuous_add_left _) ?_
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : TopologicalSpace E
    inst✝² : AddCancelCommMonoid E
    inst✝¹ : ContinuousAdd E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : StrictConvex 𝕜 s
    z x : E
    hx : Membership.mem (Set.preimage (fun x => HAdd.hAdd z x) s) x
    y : E
    hy : Membership.mem (Set.preimage (fun x => HAdd.hAdd z x) s) y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.preimage (fun b => HAdd.hAdd z b) (interior s)) (HAdd.hA …
  -/
  have h := hs hx hy ((add_right_injective _).ne hxy) ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : TopologicalSpace E
    inst✝² : AddCancelCommMonoid E
    inst✝¹ : ContinuousAdd E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : StrictConvex 𝕜 s
    z x : E
    hx : Membership.mem (Set.preimage (fun x => HAdd.hAdd z x) s) x
    y : E
    hy : Membership.mem (Set.preimage (fun x => HAdd.hAdd z x) s) y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    h : Membership.mem (interior s) (HAdd.hAdd (HSMul.hSMul a ((fun x => HAdd.hAdd …
    ⊢ Membership.mem (Set.preimage (fun b => HAdd.hAdd z b) (interior s)) (HAdd.hA …
  -/
  rwa [smul_add, smul_add, add_add_add_comm, ← _root_.add_smul, hab, one_smul] at h
  /-
    🎉 no goals
  -/


/-- The translation of a strictly convex set is also strictly convex. -/
theorem StrictConvex.preimage_add_left (hs : StrictConvex 𝕜 s) (z : E) :
    StrictConvex 𝕜 ((fun x => x + z) ⁻¹' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : TopologicalSpace E
    inst✝² : AddCancelCommMonoid E
    inst✝¹ : ContinuousAdd E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : StrictConvex 𝕜 s
    z : E
    ⊢ StrictConvex 𝕜 (Set.preimage (fun x => HAdd.hAdd x z) s)
  -/
  simpa only [add_comm] using hs.preimage_add_right z
  /-
    🎉 no goals
  -/


theorem StrictConvex.add (hs : StrictConvex 𝕜 s) (ht : StrictConvex 𝕜 t) :
    StrictConvex 𝕜 (s + t) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : TopologicalSpace E
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousAdd E
    s t : Set E
    hs : StrictConvex 𝕜 s
    ht : StrictConvex 𝕜 t
    ⊢ StrictConvex 𝕜 (HAdd.hAdd s t)
  -/
  rintro _ ⟨v, hv, w, hw, rfl⟩ _ ⟨x, hx, y, hy, rfl⟩ h a b ha hb hab
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : TopologicalSpace E
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousAdd E
    s t : Set E
    hs : StrictConvex 𝕜 s
    ht : StrictConvex 𝕜 t
    v : E
    hv : Membership.mem s v
    w : E
    hw : Membership.mem t w
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem t y
    h : Ne ((fun x1 x2 => HAdd.hAdd x1 x2) v w) ((fun x1 x2 => HAdd.hAdd x1 x2) x y)
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (interior (HAdd.hAdd s t)) (HAdd.hAdd (HSMul.hSMul a ((fun x1 …
  -/
  rw [smul_add, smul_add, add_add_add_comm]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : TopologicalSpace E
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousAdd E
    s t : Set E
    hs : StrictConvex 𝕜 s
    ht : StrictConvex 𝕜 t
    v : E
    hv : Membership.mem s v
    w : E
    hw : Membership.mem t w
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem t y
    h : Ne ((fun x1 x2 => HAdd.hAdd x1 x2) v w) ((fun x1 x2 => HAdd.hAdd x1 x2) x y)
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (interior (HAdd.hAdd s t)) (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul …
  -/
  obtain rfl | hvx := eq_or_ne v x
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.inl
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : TopologicalSpace E
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousAdd E
      s t : Set E
      hs : StrictConvex 𝕜 s
      ht : StrictConvex 𝕜 t
      v : E
      hv : Membership.mem s v
      w : E
      hw : Membership.mem t w
      y : E
      hy : Membership.mem t y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hx : Membership.mem s v
      h : Ne ((fun x1 x2 => HAdd.hAdd x1 x2) v w) ((fun x1 x2 => HAdd.hAdd x1 x2) v y)
      ⊢ Membership.mem (interior (HAdd.hAdd s t)) (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul …
    -/
  · refine interior_mono (add_subset_add (singleton_subset_iff.2 hv) Subset.rfl) ?_
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.inl
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : TopologicalSpace E
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : ContinuousAdd E
      s t : Set E
      hs : StrictConvex 𝕜 s
      ht : StrictConvex 𝕜 t
      v : E
      hv : Membership.mem s v
      w : E
      hw : Membership.mem t w
      y : E
      hy : Membership.mem t y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hx : Membership.mem s v
      h : Ne ((fun x1 x2 => HAdd.hAdd x1 x2) v w) ((fun x1 x2 => HAdd.hAdd x1 x2) v y)
      ⊢ Membership.mem (interior (HAdd.hAdd (Singleton.singleton v) t)) (HAdd.hAdd ( …
    -/
    rw [Convex.combo_self hab, singleton_add]
    exact
      (isOpenMap_add_left _).image_interior_subset _
        (mem_image_of_mem _ <| ht hw hy (ne_of_apply_ne _ h) ha hb hab)
  exact
    subset_interior_add_left
      (add_mem_add (hs hv hx hvx ha hb hab) <| ht.convex hw hy ha.le hb.le hab)


theorem StrictConvex.add_left (hs : StrictConvex 𝕜 s) (z : E) :
    StrictConvex 𝕜 ((fun x => z + x) '' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : TopologicalSpace E
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousAdd E
    s : Set E
    hs : StrictConvex 𝕜 s
    z : E
    ⊢ StrictConvex 𝕜 (Set.image (fun x => HAdd.hAdd z x) s)
  -/
  simpa only [singleton_add] using (strictConvex_singleton z).add hs
  /-
    🎉 no goals
  -/


theorem StrictConvex.add_right (hs : StrictConvex 𝕜 s) (z : E) :
                                                 /-
                                                   𝕜 : Type u_1
                                                   E : Type u_3
                                                   inst✝⁴ : OrderedSemiring 𝕜
                                                   inst✝³ : TopologicalSpace E
                                                   inst✝² : AddCommGroup E
                                                   inst✝¹ : Module 𝕜 E
                                                   inst✝ : ContinuousAdd E
                                                   s : Set E
                                                   hs : StrictConvex 𝕜 s
                                                   z : E
                                                   ⊢ StrictConvex 𝕜 (Set.image (fun x => HAdd.hAdd x z) s)
                                                 -/
    StrictConvex 𝕜 ((fun x => x + z) '' s) := by simpa only [add_comm] using hs.add_left z
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- The translation of a strictly convex set is also strictly convex. -/
theorem StrictConvex.vadd (hs : StrictConvex 𝕜 s) (x : E) : StrictConvex 𝕜 (x +ᵥ s) :=
  hs.add_left x


theorem StrictConvex.smul (hs : StrictConvex 𝕜 s) (c : 𝕝) : StrictConvex 𝕜 (c • s) := by
  /-
    𝕜 : Type u_1
    𝕝 : Type u_2
    E : Type u_3
    inst✝⁷ : OrderedSemiring 𝕜
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : LinearOrderedField 𝕝
    inst✝² : Module 𝕝 E
    inst✝¹ : ContinuousConstSMul 𝕝 E
    inst✝ : LinearMap.CompatibleSMul E E 𝕜 𝕝
    s : Set E
    hs : StrictConvex 𝕜 s
    c : 𝕝
    ⊢ StrictConvex 𝕜 (HSMul.hSMul c s)
  -/
  obtain rfl | hc := eq_or_ne c 0
    /-
      case inl
      𝕜 : Type u_1
      𝕝 : Type u_2
      E : Type u_3
      inst✝⁷ : OrderedSemiring 𝕜
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module 𝕜 E
      inst✝³ : LinearOrderedField 𝕝
      inst✝² : Module 𝕝 E
      inst✝¹ : ContinuousConstSMul 𝕝 E
      inst✝ : LinearMap.CompatibleSMul E E 𝕜 𝕝
      s : Set E
      hs : StrictConvex 𝕜 s
      ⊢ StrictConvex 𝕜 (HSMul.hSMul 0 s)
    -/
  · exact (subsingleton_zero_smul_set _).strictConvex
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      𝕝 : Type u_2
      E : Type u_3
      inst✝⁷ : OrderedSemiring 𝕜
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module 𝕜 E
      inst✝³ : LinearOrderedField 𝕝
      inst✝² : Module 𝕝 E
      inst✝¹ : ContinuousConstSMul 𝕝 E
      inst✝ : LinearMap.CompatibleSMul E E 𝕜 𝕝
      s : Set E
      hs : StrictConvex 𝕜 s
      c : 𝕝
      hc : Ne c 0
      ⊢ StrictConvex 𝕜 (HSMul.hSMul c s)
    -/
  · exact hs.linear_image (LinearMap.lsmul _ _ c) (isOpenMap_smul₀ hc)
    /-
      🎉 no goals
    -/


theorem StrictConvex.affinity [ContinuousAdd E] (hs : StrictConvex 𝕜 s) (z : E) (c : 𝕝) :
    StrictConvex 𝕜 (z +ᵥ c • s) :=
  (hs.smul c).vadd z


theorem StrictConvex.preimage_smul (hs : StrictConvex 𝕜 s) (c : 𝕜) :
    StrictConvex 𝕜 ((fun z => c • z) ⁻¹' s) := by
  classical
    obtain rfl | hc := eq_or_ne c 0
    · simp_rw [zero_smul, preimage_const]
      split_ifs
      · exact strictConvex_univ
      · exact strictConvex_empty
    refine hs.linear_preimage (LinearMap.lsmul _ _ c) ?_ (smul_right_injective E hc)
    unfold LinearMap.lsmul LinearMap.mk₂ LinearMap.mk₂' LinearMap.mk₂'ₛₗ
    exact continuous_const_smul _


theorem StrictConvex.eq_of_openSegment_subset_frontier [Nontrivial 𝕜] [DenselyOrdered 𝕜]
    (hs : StrictConvex 𝕜 s) (hx : x ∈ s) (hy : y ∈ s) (h : openSegment 𝕜 x y ⊆ frontier s) :
    x = y := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁵ : OrderedRing 𝕜
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    s : Set E
    x y : E
    inst✝¹ : Nontrivial 𝕜
    inst✝ : DenselyOrdered 𝕜
    hs : StrictConvex 𝕜 s
    hx : Membership.mem s x
    hy : Membership.mem s y
    h : HasSubset.Subset (openSegment 𝕜 x y) (frontier s)
    ⊢ Eq x y
  -/
  obtain ⟨a, ha₀, ha₁⟩ := DenselyOrdered.dense (0 : 𝕜) 1 zero_lt_one
  classical
    by_contra hxy
    exact
      (h ⟨a, 1 - a, ha₀, sub_pos_of_lt ha₁, add_sub_cancel _ _, rfl⟩).2
        (hs hx hy hxy ha₀ (sub_pos_of_lt ha₁) <| add_sub_cancel _ _)


theorem StrictConvex.add_smul_mem (hs : StrictConvex 𝕜 s) (hx : x ∈ s) (hxy : x + y ∈ s)
    (hy : y ≠ 0) {t : 𝕜} (ht₀ : 0 < t) (ht₁ : t < 1) : x + t • y ∈ interior s := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedRing 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x y : E
    hs : StrictConvex 𝕜 s
    hx : Membership.mem s x
    hxy : Membership.mem s (HAdd.hAdd x y)
    hy : Ne y 0
    t : 𝕜
    ht₀ : LT.lt 0 t
    ht₁ : LT.lt t 1
    ⊢ Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul t y))
  -/
  have h : x + t • y = (1 - t) • x + t • (x + y) := by match_scalars <;> field_simp
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedRing 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x y : E
    hs : StrictConvex 𝕜 s
    hx : Membership.mem s x
    hxy : Membership.mem s (HAdd.hAdd x y)
    hy : Ne y 0
    t : 𝕜
    ht₀ : LT.lt 0 t
    ht₁ : LT.lt t 1
    h : Eq (HAdd.hAdd x (HSMul.hSMul t y)) (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 t) …
    ⊢ Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul t y))
  -/
  rw [h]
  exact hs hx hxy (fun h => hy <| add_left_cancel (a := x) (by rw [← h, add_zero]))
    (sub_pos_of_lt ht₁) ht₀ (sub_add_cancel 1 t)


theorem StrictConvex.smul_mem_of_zero_mem (hs : StrictConvex 𝕜 s) (zero_mem : (0 : E) ∈ s)
    (hx : x ∈ s) (hx₀ : x ≠ 0) {t : 𝕜} (ht₀ : 0 < t) (ht₁ : t < 1) : t • x ∈ interior s := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedRing 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    hs : StrictConvex 𝕜 s
    zero_mem : Membership.mem s 0
    hx : Membership.mem s x
    hx₀ : Ne x 0
    t : 𝕜
    ht₀ : LT.lt 0 t
    ht₁ : LT.lt t 1
    ⊢ Membership.mem (interior s) (HSMul.hSMul t x)
  -/
  simpa using hs.add_smul_mem zero_mem (by simpa using hx) hx₀ ht₀ ht₁
  /-
    🎉 no goals
  -/


theorem StrictConvex.add_smul_sub_mem (h : StrictConvex 𝕜 s) (hx : x ∈ s) (hy : y ∈ s) (hxy : x ≠ y)
    {t : 𝕜} (ht₀ : 0 < t) (ht₁ : t < 1) : x + t • (y - x) ∈ interior s := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedRing 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x y : E
    h : StrictConvex 𝕜 s
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxy : Ne x y
    t : 𝕜
    ht₀ : LT.lt 0 t
    ht₁ : LT.lt t 1
    ⊢ Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul t (HSub.hSub y x)))
  -/
  apply h.openSegment_subset hx hy hxy
  /-
    case a
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedRing 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x y : E
    h : StrictConvex 𝕜 s
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxy : Ne x y
    t : 𝕜
    ht₀ : LT.lt 0 t
    ht₁ : LT.lt t 1
    ⊢ Membership.mem (openSegment 𝕜 x y) (HAdd.hAdd x (HSMul.hSMul t (HSub.hSub y  …
  -/
  rw [openSegment_eq_image']
  /-
    case a
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : OrderedRing 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x y : E
    h : StrictConvex 𝕜 s
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxy : Ne x y
    t : 𝕜
    ht₀ : LT.lt 0 t
    ht₁ : LT.lt t 1
    ⊢ Membership.mem (Set.image (fun θ => HAdd.hAdd x (HSMul.hSMul θ (HSub.hSub y  …
  -/
  exact mem_image_of_mem _ ⟨ht₀, ht₁⟩
  /-
    🎉 no goals
  -/


/-- The preimage of a strictly convex set under an affine map is strictly convex. -/
theorem StrictConvex.affine_preimage {s : Set F} (hs : StrictConvex 𝕜 s) {f : E →ᵃ[𝕜] F}
    (hf : Continuous f) (hfinj : Injective f) : StrictConvex 𝕜 (f ⁻¹' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁶ : OrderedRing 𝕜
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set F
    hs : StrictConvex 𝕜 s
    f : AffineMap 𝕜 E F
    hf : Continuous ⇑f
    hfinj : Function.Injective ⇑f
    ⊢ StrictConvex 𝕜 (Set.preimage (⇑f) s)
  -/
  intro x hx y hy hxy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁶ : OrderedRing 𝕜
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set F
    hs : StrictConvex 𝕜 s
    f : AffineMap 𝕜 E F
    hf : Continuous ⇑f
    hfinj : Function.Injective ⇑f
    x : E
    hx : Membership.mem (Set.preimage (⇑f) s) x
    y : E
    hy : Membership.mem (Set.preimage (⇑f) s) y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (interior (Set.preimage (⇑f) s)) (HAdd.hAdd (HSMul.hSMul a x) …
  -/
  refine preimage_interior_subset_interior_preimage hf ?_
  /-
    𝕜 : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁶ : OrderedRing 𝕜
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set F
    hs : StrictConvex 𝕜 s
    f : AffineMap 𝕜 E F
    hf : Continuous ⇑f
    hfinj : Function.Injective ⇑f
    x : E
    hx : Membership.mem (Set.preimage (⇑f) s) x
    y : E
    hy : Membership.mem (Set.preimage (⇑f) s) y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.preimage (⇑f) (interior s)) (HAdd.hAdd (HSMul.hSMul a x) …
  -/
  rw [mem_preimage, Convex.combo_affine_apply hab]
  /-
    𝕜 : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁶ : OrderedRing 𝕜
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set F
    hs : StrictConvex 𝕜 s
    f : AffineMap 𝕜 E F
    hf : Continuous ⇑f
    hfinj : Function.Injective ⇑f
    x : E
    hx : Membership.mem (Set.preimage (⇑f) s) x
    y : E
    hy : Membership.mem (Set.preimage (⇑f) s) y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (interior s) (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b  …
  -/
  exact hs hx hy (hfinj.ne hxy) ha hb hab
  /-
    🎉 no goals
  -/


/-- The image of a strictly convex set under an affine map is strictly convex. -/
theorem StrictConvex.affine_image (hs : StrictConvex 𝕜 s) {f : E →ᵃ[𝕜] F} (hf : IsOpenMap f) :
    StrictConvex 𝕜 (f '' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁶ : OrderedRing 𝕜
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set E
    hs : StrictConvex 𝕜 s
    f : AffineMap 𝕜 E F
    hf : IsOpenMap ⇑f
    ⊢ StrictConvex 𝕜 (Set.image (⇑f) s)
  -/
  rintro _ ⟨x, hx, rfl⟩ _ ⟨y, hy, rfl⟩ hxy a b ha hb hab
  exact
    hf.image_interior_subset _
      ⟨a • x + b • y, ⟨hs hx hy (ne_of_apply_ne _ hxy) ha hb hab, Convex.combo_affine_apply hab⟩⟩


theorem StrictConvex.neg (hs : StrictConvex 𝕜 s) : StrictConvex 𝕜 (-s) :=
  hs.is_linear_preimage IsLinearMap.isLinearMap_neg continuous_id.neg neg_injective


theorem StrictConvex.sub (hs : StrictConvex 𝕜 s) (ht : StrictConvex 𝕜 t) : StrictConvex 𝕜 (s - t) :=
  (sub_eq_add_neg s t).symm ▸ hs.add ht.neg


/-- Alternative definition of set strict convexity, using division. -/
theorem strictConvex_iff_div :
    StrictConvex 𝕜 s ↔
      s.Pairwise fun x y =>
        ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b → (a / (a + b)) • x + (b / (a + b)) • y ∈ interior s :=
                                                   /-
                                                     𝕜 : Type u_1
                                                     E : Type u_3
                                                     inst✝³ : LinearOrderedField 𝕜
                                                     inst✝² : TopologicalSpace E
                                                     inst✝¹ : AddCommGroup E
                                                     inst✝ : Module 𝕜 E
                                                     s : Set E
                                                     h : StrictConvex 𝕜 s
                                                     x : E
                                                     hx : Membership.mem s x
                                                     y : E
                                                     hy : Membership.mem s y
                                                     hxy : Ne x y
                                                     a b : 𝕜
                                                     ha : LT.lt 0 a
                                                     hb : LT.lt 0 b
                                                     ⊢ LT.lt 0 (HDiv.hDiv a (HAdd.hAdd a b))
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  ⟨fun h x hx y hy hxy a b ha hb ↦ h hx hy hxy (by positivity) (by positivity) (by field_simp),
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
    fun h x hx y hy hxy a b ha hb hab ↦ by
    /-
      𝕜 : Type u_1
      E : Type u_3
      inst✝³ : LinearOrderedField 𝕜
      inst✝² : TopologicalSpace E
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      h : s.Pairwise fun x y => ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b → Membership.mem  …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hxy : Ne x y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Membership.mem (interior s) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
    -/
                                  /-
                                    🎉 no goals
                                  -/
    convert h hx hy hxy ha hb <;> rw [hab, div_one]⟩
                                  /-
                                    🎉 no goals
                                  -/


theorem StrictConvex.mem_smul_of_zero_mem (hs : StrictConvex 𝕜 s) (zero_mem : (0 : E) ∈ s)
    (hx : x ∈ s) (hx₀ : x ≠ 0) {t : 𝕜} (ht : 1 < t) : x ∈ t • interior s := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    hs : StrictConvex 𝕜 s
    zero_mem : Membership.mem s 0
    hx : Membership.mem s x
    hx₀ : Ne x 0
    t : 𝕜
    ht : LT.lt 1 t
    ⊢ Membership.mem (HSMul.hSMul t (interior s)) x
  -/
  rw [mem_smul_set_iff_inv_smul_mem₀ (by positivity)]
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : TopologicalSpace E
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    hs : StrictConvex 𝕜 s
    zero_mem : Membership.mem s 0
    hx : Membership.mem s x
    hx₀ : Ne x 0
    t : 𝕜
    ht : LT.lt 1 t
    ⊢ Membership.mem (interior s) (HSMul.hSMul (Inv.inv t) x)
  -/
  exact hs.smul_mem_of_zero_mem zero_mem hx hx₀ (by positivity) (inv_lt_one_of_one_lt₀ ht)
  /-
    🎉 no goals
  -/


/-- A set in a linear ordered field is strictly convex if and only if it is convex. -/
@[simp]
theorem strictConvex_iff_convex : StrictConvex 𝕜 s ↔ Convex 𝕜 s :=
  ⟨StrictConvex.convex, fun hs => hs.ordConnected.strictConvex⟩


theorem strictConvex_iff_ordConnected : StrictConvex 𝕜 s ↔ s.OrdConnected :=
  strictConvex_iff_convex.trans convex_iff_ordConnected


alias ⟨StrictConvex.ordConnected, _⟩ := strictConvex_iff_ordConnected


