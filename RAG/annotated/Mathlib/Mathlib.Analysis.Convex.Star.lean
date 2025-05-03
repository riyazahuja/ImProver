/-- Star-convexity of sets. `s` is star-convex at `x` if every segment from `x` to a point in `s` is
contained in `s`. -/
def StarConvex : Prop :=
  ∀ ⦃y : E⦄, y ∈ s → ∀ ⦃a b : 𝕜⦄, 0 ≤ a → 0 ≤ b → a + b = 1 → a • x + b • y ∈ s


theorem starConvex_iff_segment_subset : StarConvex 𝕜 x s ↔ ∀ ⦃y⦄, y ∈ s → [x -[𝕜] y] ⊆ s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    x : E
    s : Set E
    ⊢ Iff (StarConvex 𝕜 x s) (∀ ⦃y : E⦄, Membership.mem s y → HasSubset.Subset (se …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : SMul 𝕜 E
      x : E
      s : Set E
      ⊢ StarConvex 𝕜 x s → ∀ ⦃y : E⦄, Membership.mem s y → HasSubset.Subset (segment …
    -/
  · rintro h y hy z ⟨a, b, ha, hb, hab, rfl⟩
    /-
      case mp.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : SMul 𝕜 E
      x : E
      s : Set E
      h : StarConvex 𝕜 x s
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
    -/
    exact h hy ha hb hab
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : SMul 𝕜 E
      x : E
      s : Set E
      ⊢ (∀ ⦃y : E⦄, Membership.mem s y → HasSubset.Subset (segment 𝕜 x y) s) → StarC …
    -/
  · rintro h y hy a b ha hb hab
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : SMul 𝕜 E
      x : E
      s : Set E
      h : ∀ ⦃y : E⦄, Membership.mem s y → HasSubset.Subset (segment 𝕜 x y) s
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
    -/
    exact h hy ⟨a, b, ha, hb, hab, rfl⟩
    /-
      🎉 no goals
    -/


theorem StarConvex.segment_subset (h : StarConvex 𝕜 x s) {y : E} (hy : y ∈ s) : [x -[𝕜] y] ⊆ s :=
  starConvex_iff_segment_subset.1 h hy


theorem StarConvex.openSegment_subset (h : StarConvex 𝕜 x s) {y : E} (hy : y ∈ s) :
    openSegment 𝕜 x y ⊆ s :=
  (openSegment_subset_segment 𝕜 x y).trans (h.segment_subset hy)


/-- Alternative definition of star-convexity, in terms of pointwise set operations. -/
theorem starConvex_iff_pointwise_add_subset :
    StarConvex 𝕜 x s ↔ ∀ ⦃a b : 𝕜⦄, 0 ≤ a → 0 ≤ b → a + b = 1 → a • {x} + b • s ⊆ s := by
  refine
    ⟨?_, fun h y hy a b ha hb hab =>
      h ha hb hab (add_mem_add (smul_mem_smul_set <| mem_singleton _) ⟨_, hy, rfl⟩)⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    x : E
    s : Set E
    ⊢ StarConvex 𝕜 x s → ∀ ⦃a b : 𝕜⦄, LE.le 0 a → LE.le 0 b → Eq (HAdd.hAdd a b) 1 …
  -/
  rintro hA a b ha hb hab w ⟨au, ⟨u, rfl : u = x, rfl⟩, bv, ⟨v, hv, rfl⟩, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    s : Set E
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    u : E
    hA : StarConvex 𝕜 u s
    v : E
    hv : Membership.mem s v
    ⊢ Membership.mem s ((fun x1 x2 => HAdd.hAdd x1 x2) ((fun x => HSMul.hSMul a x) …
  -/
  exact hA hv ha hb hab
  /-
    🎉 no goals
  -/


theorem starConvex_empty (x : E) : StarConvex 𝕜 x ∅ := fun _ hy => hy.elim


theorem starConvex_univ (x : E) : StarConvex 𝕜 x univ := fun _ _ _ _ _ _ _ => trivial


theorem StarConvex.inter (hs : StarConvex 𝕜 x s) (ht : StarConvex 𝕜 x t) : StarConvex 𝕜 x (s ∩ t) :=
  fun _ hy _ _ ha hb hab => ⟨hs hy.left ha hb hab, ht hy.right ha hb hab⟩


theorem starConvex_sInter {S : Set (Set E)} (h : ∀ s ∈ S, StarConvex 𝕜 x s) :
    StarConvex 𝕜 x (⋂₀ S) := fun _ hy _ _ ha hb hab s hs => h s hs (hy s hs) ha hb hab


theorem starConvex_iInter {ι : Sort*} {s : ι → Set E} (h : ∀ i, StarConvex 𝕜 x (s i)) :
    StarConvex 𝕜 x (⋂ i, s i) :=
  sInter_range s ▸ starConvex_sInter <| forall_mem_range.2 h


theorem StarConvex.union (hs : StarConvex 𝕜 x s) (ht : StarConvex 𝕜 x t) :
    StarConvex 𝕜 x (s ∪ t) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    x : E
    s t : Set E
    hs : StarConvex 𝕜 x s
    ht : StarConvex 𝕜 x t
    ⊢ StarConvex 𝕜 x (Union.union s t)
  -/
  rintro y (hy | hy) a b ha hb hab
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : SMul 𝕜 E
      x : E
      s t : Set E
      hs : StarConvex 𝕜 x s
      ht : StarConvex 𝕜 x t
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Membership.mem (Union.union s t) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b …
    -/
  · exact Or.inl (hs hy ha hb hab)
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : SMul 𝕜 E
      x : E
      s t : Set E
      hs : StarConvex 𝕜 x s
      ht : StarConvex 𝕜 x t
      y : E
      hy : Membership.mem t y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Membership.mem (Union.union s t) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b …
    -/
  · exact Or.inr (ht hy ha hb hab)
    /-
      🎉 no goals
    -/


theorem starConvex_iUnion {ι : Sort*} {s : ι → Set E} (hs : ∀ i, StarConvex 𝕜 x (s i)) :
    StarConvex 𝕜 x (⋃ i, s i) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    x : E
    ι : Sort u_4
    s : ι → Set E
    hs : ∀ (i : ι), StarConvex 𝕜 x (s i)
    ⊢ StarConvex 𝕜 x (Set.iUnion fun i => s i)
  -/
  rintro y hy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    x : E
    ι : Sort u_4
    s : ι → Set E
    hs : ∀ (i : ι), StarConvex 𝕜 x (s i)
    y : E
    hy : Membership.mem (Set.iUnion fun i => s i) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.iUnion fun i => s i) (HAdd.hAdd (HSMul.hSMul a x) (HSMul …
  -/
  rw [mem_iUnion] at hy ⊢
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    x : E
    ι : Sort u_4
    s : ι → Set E
    hs : ∀ (i : ι), StarConvex 𝕜 x (s i)
    y : E
    hy : Exists fun i => Membership.mem (s i) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Exists fun i => Membership.mem (s i) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSM …
  -/
  obtain ⟨i, hy⟩ := hy
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    x : E
    ι : Sort u_4
    s : ι → Set E
    hs : ∀ (i : ι), StarConvex 𝕜 x (s i)
    y : E
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    i : ι
    hy : Membership.mem (s i) y
    ⊢ Exists fun i => Membership.mem (s i) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSM …
  -/
  exact ⟨i, hs i hy ha hb hab⟩
  /-
    🎉 no goals
  -/


theorem starConvex_sUnion {S : Set (Set E)} (hS : ∀ s ∈ S, StarConvex 𝕜 x s) :
    StarConvex 𝕜 x (⋃₀ S) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    x : E
    S : Set (Set E)
    hS : ∀ (s : Set E), Membership.mem S s → StarConvex 𝕜 x s
    ⊢ StarConvex 𝕜 x S.sUnion
  -/
  rw [sUnion_eq_iUnion]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    x : E
    S : Set (Set E)
    hS : ∀ (s : Set E), Membership.mem S s → StarConvex 𝕜 x s
    ⊢ StarConvex 𝕜 x (Set.iUnion fun i => ↑i)
  -/
  exact starConvex_iUnion fun s => hS _ s.2
  /-
    🎉 no goals
  -/


theorem StarConvex.prod {y : F} {s : Set E} {t : Set F} (hs : StarConvex 𝕜 x s)
    (ht : StarConvex 𝕜 y t) : StarConvex 𝕜 (x, y) (s ×ˢ t) := fun _ hy _ _ ha hb hab =>
  ⟨hs hy.1 ha hb hab, ht hy.2 ha hb hab⟩


theorem starConvex_pi {ι : Type*} {E : ι → Type*} [∀ i, AddCommMonoid (E i)] [∀ i, SMul 𝕜 (E i)]
    {x : ∀ i, E i} {s : Set ι} {t : ∀ i, Set (E i)} (ht : ∀ ⦃i⦄, i ∈ s → StarConvex 𝕜 (x i) (t i)) :
    StarConvex 𝕜 x (s.pi t) := fun _ hy _ _ ha hb hab i hi => ht hi (hy i hi) ha hb hab


theorem StarConvex.mem (hs : StarConvex 𝕜 x s) (h : s.Nonempty) : x ∈ s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    h : s.Nonempty
    ⊢ Membership.mem s x
  -/
  obtain ⟨y, hy⟩ := h
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    y : E
    hy : Membership.mem s y
    ⊢ Membership.mem s x
  -/
  convert hs hy zero_le_one le_rfl (add_zero 1)
  /-
    case h.e'_5
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    y : E
    hy : Membership.mem s y
    ⊢ Eq x (HAdd.hAdd (HSMul.hSMul 1 x) (HSMul.hSMul 0 y))
  -/
  rw [one_smul, zero_smul, add_zero]
  /-
    🎉 no goals
  -/


theorem starConvex_iff_forall_pos (hx : x ∈ s) : StarConvex 𝕜 x s ↔
    ∀ ⦃y⦄, y ∈ s → ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b → a + b = 1 → a • x + b • y ∈ s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hx : Membership.mem s x
    ⊢ Iff (StarConvex 𝕜 x s) (∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 …
  -/
  refine ⟨fun h y hy a b ha hb hab => h hy ha.le hb.le hab, ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hx : Membership.mem s x
    ⊢ (∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b → Eq (HA …
  -/
  intro h y hy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hx : Membership.mem s x
    h : ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b → Eq (H …
    y : E
    hy : Membership.mem s y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
  -/
  obtain rfl | ha := ha.eq_or_lt
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      x : E
      s : Set E
      hx : Membership.mem s x
      h : ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b → Eq (H …
      y : E
      hy : Membership.mem s y
      b : 𝕜
      hb : LE.le 0 b
      ha : LE.le 0 0
      hab : Eq (HAdd.hAdd 0 b) 1
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul 0 x) (HSMul.hSMul b y))
    -/
  · rw [zero_add] at hab
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      x : E
      s : Set E
      hx : Membership.mem s x
      h : ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b → Eq (H …
      y : E
      hy : Membership.mem s y
      b : 𝕜
      hb : LE.le 0 b
      ha : LE.le 0 0
      hab : Eq b 1
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul 0 x) (HSMul.hSMul b y))
    -/
    rwa [hab, one_smul, zero_smul, zero_add]
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hx : Membership.mem s x
    h : ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b → Eq (H …
    y : E
    hy : Membership.mem s y
    a b : 𝕜
    ha✝ : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ha : LT.lt 0 a
    ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
  -/
  obtain rfl | hb := hb.eq_or_lt
    /-
      case inr.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      x : E
      s : Set E
      hx : Membership.mem s x
      h : ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b → Eq (H …
      y : E
      hy : Membership.mem s y
      a : 𝕜
      ha✝ : LE.le 0 a
      ha : LT.lt 0 a
      hb : LE.le 0 0
      hab : Eq (HAdd.hAdd a 0) 1
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul 0 y))
    -/
  · rw [add_zero] at hab
    /-
      case inr.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      x : E
      s : Set E
      hx : Membership.mem s x
      h : ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b → Eq (H …
      y : E
      hy : Membership.mem s y
      a : 𝕜
      ha✝ : LE.le 0 a
      ha : LT.lt 0 a
      hb : LE.le 0 0
      hab : Eq a 1
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul 0 y))
    -/
    rwa [hab, one_smul, zero_smul, add_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hx : Membership.mem s x
    h : ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b → Eq (H …
    y : E
    hy : Membership.mem s y
    a b : 𝕜
    ha✝ : LE.le 0 a
    hb✝ : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
  -/
  exact h hy ha hb hab
  /-
    🎉 no goals
  -/


theorem starConvex_iff_forall_ne_pos (hx : x ∈ s) :
    StarConvex 𝕜 x s ↔
      ∀ ⦃y⦄, y ∈ s → x ≠ y → ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b → a + b = 1 → a • x + b • y ∈ s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hx : Membership.mem s x
    ⊢ Iff (StarConvex 𝕜 x s) (∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ ⦃a b : 𝕜⦄ …
  -/
  refine ⟨fun h y hy _ a b ha hb hab => h hy ha.le hb.le hab, ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hx : Membership.mem s x
    ⊢ (∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b …
  -/
  intro h y hy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hx : Membership.mem s x
    h : ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0  …
    y : E
    hy : Membership.mem s y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
  -/
  obtain rfl | ha' := ha.eq_or_lt
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      x : E
      s : Set E
      hx : Membership.mem s x
      h : ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0  …
      y : E
      hy : Membership.mem s y
      b : 𝕜
      hb : LE.le 0 b
      ha : LE.le 0 0
      hab : Eq (HAdd.hAdd 0 b) 1
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul 0 x) (HSMul.hSMul b y))
    -/
  · rw [zero_add] at hab
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      x : E
      s : Set E
      hx : Membership.mem s x
      h : ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0  …
      y : E
      hy : Membership.mem s y
      b : 𝕜
      hb : LE.le 0 b
      ha : LE.le 0 0
      hab : Eq b 1
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul 0 x) (HSMul.hSMul b y))
    -/
    rwa [hab, zero_smul, one_smul, zero_add]
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hx : Membership.mem s x
    h : ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0  …
    y : E
    hy : Membership.mem s y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ha' : LT.lt 0 a
    ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
  -/
  obtain rfl | hb' := hb.eq_or_lt
    /-
      case inr.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      x : E
      s : Set E
      hx : Membership.mem s x
      h : ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0  …
      y : E
      hy : Membership.mem s y
      a : 𝕜
      ha : LE.le 0 a
      ha' : LT.lt 0 a
      hb : LE.le 0 0
      hab : Eq (HAdd.hAdd a 0) 1
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul 0 y))
    -/
  · rw [add_zero] at hab
    /-
      case inr.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      x : E
      s : Set E
      hx : Membership.mem s x
      h : ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0  …
      y : E
      hy : Membership.mem s y
      a : 𝕜
      ha : LE.le 0 a
      ha' : LT.lt 0 a
      hb : LE.le 0 0
      hab : Eq a 1
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul 0 y))
    -/
    rwa [hab, zero_smul, one_smul, add_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hx : Membership.mem s x
    h : ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0  …
    y : E
    hy : Membership.mem s y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ha' : LT.lt 0 a
    hb' : LT.lt 0 b
    ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
  -/
  obtain rfl | hxy := eq_or_ne x y
    /-
      case inr.inr.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      x : E
      s : Set E
      hx : Membership.mem s x
      h : ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0  …
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ha' : LT.lt 0 a
      hb' : LT.lt 0 b
      hy : Membership.mem s x
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b x))
    -/
  · rwa [Convex.combo_self hab]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr.inr
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hx : Membership.mem s x
    h : ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0  …
    y : E
    hy : Membership.mem s y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ha' : LT.lt 0 a
    hb' : LT.lt 0 b
    hxy : Ne x y
    ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
  -/
  exact h hy hxy ha' hb' hab
  /-
    🎉 no goals
  -/


theorem starConvex_iff_openSegment_subset (hx : x ∈ s) :
    StarConvex 𝕜 x s ↔ ∀ ⦃y⦄, y ∈ s → openSegment 𝕜 x y ⊆ s :=
  starConvex_iff_segment_subset.trans <|
    forall₂_congr fun _ hy => (openSegment_subset_iff_segment_subset hx hy).symm


theorem starConvex_singleton (x : E) : StarConvex 𝕜 x {x} := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    ⊢ StarConvex 𝕜 x (Singleton.singleton x)
  -/
  rintro y (rfl : y = x) a b _ _ hab
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    y : E
    a b : 𝕜
    a✝¹ : LE.le 0 a
    a✝ : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Singleton.singleton y) (HAdd.hAdd (HSMul.hSMul a y) (HSMul.h …
  -/
  exact Convex.combo_self hab _
  /-
    🎉 no goals
  -/


theorem StarConvex.linear_image (hs : StarConvex 𝕜 x s) (f : E →ₗ[𝕜] F) :
    StarConvex 𝕜 (f x) (f '' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    f : LinearMap (RingHom.id 𝕜) E F
    ⊢ StarConvex 𝕜 (f x) (Set.image (⇑f) s)
  -/
  rintro _ ⟨y, hy, rfl⟩ a b ha hb hab
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    f : LinearMap (RingHom.id 𝕜) E F
    y : E
    hy : Membership.mem s y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.image (⇑f) s) (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hS …
  -/
  exact ⟨a • x + b • y, hs hy ha hb hab, by rw [f.map_add, f.map_smul, f.map_smul]⟩
  /-
    🎉 no goals
  -/


theorem StarConvex.is_linear_image (hs : StarConvex 𝕜 x s) {f : E → F} (hf : IsLinearMap 𝕜 f) :
    StarConvex 𝕜 (f x) (f '' s) :=
  hs.linear_image <| hf.mk' f


theorem StarConvex.linear_preimage {s : Set F} (f : E →ₗ[𝕜] F) (hs : StarConvex 𝕜 (f x) s) :
    StarConvex 𝕜 x (f ⁻¹' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    s : Set F
    f : LinearMap (RingHom.id 𝕜) E F
    hs : StarConvex 𝕜 (f x) s
    ⊢ StarConvex 𝕜 x (Set.preimage (⇑f) s)
  -/
  intro y hy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    s : Set F
    f : LinearMap (RingHom.id 𝕜) E F
    hs : StarConvex 𝕜 (f x) s
    y : E
    hy : Membership.mem (Set.preimage (⇑f) s) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.preimage (⇑f) s) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSM …
  -/
  rw [mem_preimage, f.map_add, f.map_smul, f.map_smul]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    s : Set F
    f : LinearMap (RingHom.id 𝕜) E F
    hs : StarConvex 𝕜 (f x) s
    y : E
    hy : Membership.mem (Set.preimage (⇑f) s) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y)))
  -/
  exact hs hy ha hb hab
  /-
    🎉 no goals
  -/


theorem StarConvex.is_linear_preimage {s : Set F} {f : E → F} (hs : StarConvex 𝕜 (f x) s)
    (hf : IsLinearMap 𝕜 f) : StarConvex 𝕜 x (preimage f s) :=
  hs.linear_preimage <| hf.mk' f


theorem StarConvex.add {t : Set E} (hs : StarConvex 𝕜 x s) (ht : StarConvex 𝕜 y t) :
    StarConvex 𝕜 (x + y) (s + t) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x y : E
    s t : Set E
    hs : StarConvex 𝕜 x s
    ht : StarConvex 𝕜 y t
    ⊢ StarConvex 𝕜 (HAdd.hAdd x y) (HAdd.hAdd s t)
  -/
  rw [← add_image_prod]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x y : E
    s t : Set E
    hs : StarConvex 𝕜 x s
    ht : StarConvex 𝕜 y t
    ⊢ StarConvex 𝕜 (HAdd.hAdd x y) (Set.image (fun x => HAdd.hAdd x.1 x.2) (SProd. …
  -/
  exact (hs.prod ht).is_linear_image IsLinearMap.isLinearMap_add
  /-
    🎉 no goals
  -/


theorem StarConvex.add_left (hs : StarConvex 𝕜 x s) (z : E) :
    StarConvex 𝕜 (z + x) ((fun x => z + x) '' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    z : E
    ⊢ StarConvex 𝕜 (HAdd.hAdd z x) (Set.image (fun x => HAdd.hAdd z x) s)
  -/
  intro y hy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    z y : E
    hy : Membership.mem (Set.image (fun x => HAdd.hAdd z x) s) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.image (fun x => HAdd.hAdd z x) s) (HAdd.hAdd (HSMul.hSMu …
  -/
  obtain ⟨y', hy', rfl⟩ := hy
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    z : E
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    y' : E
    hy' : Membership.mem s y'
    ⊢ Membership.mem (Set.image (fun x => HAdd.hAdd z x) s) (HAdd.hAdd (HSMul.hSMu …
  -/
  refine ⟨a • x + b • y', hs hy' ha hb hab, ?_⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    z : E
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    y' : E
    hy' : Membership.mem s y'
    ⊢ Eq ((fun x => HAdd.hAdd z x) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y') …
  -/
                    /-
                      🎉 no goals
                    -/
                    /-
                      🎉 no goals
                    -/
  match_scalars <;> simp [hab]
                    /-
                      🎉 no goals
                    -/


theorem StarConvex.add_right (hs : StarConvex 𝕜 x s) (z : E) :
    StarConvex 𝕜 (x + z) ((fun x => x + z) '' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    z : E
    ⊢ StarConvex 𝕜 (HAdd.hAdd x z) (Set.image (fun x => HAdd.hAdd x z) s)
  -/
  intro y hy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    z y : E
    hy : Membership.mem (Set.image (fun x => HAdd.hAdd x z) s) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.image (fun x => HAdd.hAdd x z) s) (HAdd.hAdd (HSMul.hSMu …
  -/
  obtain ⟨y', hy', rfl⟩ := hy
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    z : E
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    y' : E
    hy' : Membership.mem s y'
    ⊢ Membership.mem (Set.image (fun x => HAdd.hAdd x z) s) (HAdd.hAdd (HSMul.hSMu …
  -/
  refine ⟨a • x + b • y', hs hy' ha hb hab, ?_⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    z : E
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    y' : E
    hy' : Membership.mem s y'
    ⊢ Eq ((fun x => HAdd.hAdd x z) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y') …
  -/
                    /-
                      🎉 no goals
                    -/
                    /-
                      🎉 no goals
                    -/
  match_scalars <;> simp [hab]
                    /-
                      🎉 no goals
                    -/


/-- The translation of a star-convex set is also star-convex. -/
theorem StarConvex.preimage_add_right (hs : StarConvex 𝕜 (z + x) s) :
    StarConvex 𝕜 x ((fun x => z + x) ⁻¹' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x z : E
    s : Set E
    hs : StarConvex 𝕜 (HAdd.hAdd z x) s
    ⊢ StarConvex 𝕜 x (Set.preimage (fun x => HAdd.hAdd z x) s)
  -/
  intro y hy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x z : E
    s : Set E
    hs : StarConvex 𝕜 (HAdd.hAdd z x) s
    y : E
    hy : Membership.mem (Set.preimage (fun x => HAdd.hAdd z x) s) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.preimage (fun x => HAdd.hAdd z x) s) (HAdd.hAdd (HSMul.h …
  -/
  have h := hs hy ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x z : E
    s : Set E
    hs : StarConvex 𝕜 (HAdd.hAdd z x) s
    y : E
    hy : Membership.mem (Set.preimage (fun x => HAdd.hAdd z x) s) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    h : Membership.mem s (HAdd.hAdd (HSMul.hSMul a (HAdd.hAdd z x)) (HSMul.hSMul b …
    ⊢ Membership.mem (Set.preimage (fun x => HAdd.hAdd z x) s) (HAdd.hAdd (HSMul.h …
  -/
  rwa [smul_add, smul_add, add_add_add_comm, ← add_smul, hab, one_smul] at h
  /-
    🎉 no goals
  -/


/-- The translation of a star-convex set is also star-convex. -/
theorem StarConvex.preimage_add_left (hs : StarConvex 𝕜 (x + z) s) :
    StarConvex 𝕜 x ((fun x => x + z) ⁻¹' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x z : E
    s : Set E
    hs : StarConvex 𝕜 (HAdd.hAdd x z) s
    ⊢ StarConvex 𝕜 x (Set.preimage (fun x => HAdd.hAdd x z) s)
  -/
  rw [add_comm] at hs
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x z : E
    s : Set E
    hs : StarConvex 𝕜 (HAdd.hAdd z x) s
    ⊢ StarConvex 𝕜 x (Set.preimage (fun x => HAdd.hAdd x z) s)
  -/
  simpa only [add_comm] using hs.preimage_add_right
  /-
    🎉 no goals
  -/


theorem StarConvex.sub' {s : Set (E × E)} (hs : StarConvex 𝕜 (x, y) s) :
    StarConvex 𝕜 (x - y) ((fun x : E × E => x.1 - x.2) '' s) :=
  hs.is_linear_image IsLinearMap.isLinearMap_sub


theorem StarConvex.smul (hs : StarConvex 𝕜 x s) (c : 𝕜) : StarConvex 𝕜 (c • x) (c • s) :=
  hs.linear_image <| LinearMap.lsmul _ _ c


theorem StarConvex.preimage_smul {c : 𝕜} (hs : StarConvex 𝕜 (c • x) s) :
    StarConvex 𝕜 x ((fun z => c • z) ⁻¹' s) :=
  hs.linear_preimage (LinearMap.lsmul _ _ c)


theorem StarConvex.affinity (hs : StarConvex 𝕜 x s) (z : E) (c : 𝕜) :
    StarConvex 𝕜 (z + c • x) ((fun x => z + c • x) '' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedCommSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    z : E
    c : 𝕜
    ⊢ StarConvex 𝕜 (HAdd.hAdd z (HSMul.hSMul c x)) (Set.image (fun x => HAdd.hAdd  …
  -/
  have h := (hs.smul c).add_left z
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedCommSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    z : E
    c : 𝕜
    h : StarConvex 𝕜 (HAdd.hAdd z (HSMul.hSMul c x)) (Set.image (fun x => HAdd.hAd …
    ⊢ StarConvex 𝕜 (HAdd.hAdd z (HSMul.hSMul c x)) (Set.image (fun x => HAdd.hAdd  …
  -/
  rwa [← image_smul, image_image] at h
  /-
    🎉 no goals
  -/


theorem starConvex_zero_iff :
    StarConvex 𝕜 0 s ↔ ∀ ⦃x : E⦄, x ∈ s → ∀ ⦃a : 𝕜⦄, 0 ≤ a → a ≤ 1 → a • x ∈ s := by
  refine
    forall_congr' fun x => forall_congr' fun _ => ⟨fun h a ha₀ ha₁ => ?_, fun h a b ha hb hab => ?_⟩
  · simpa only [sub_add_cancel, eq_self_iff_true, forall_true_left, zero_add, smul_zero] using
      h (sub_nonneg_of_le ha₁) ha₀
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedRing 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : SMulWithZero 𝕜 E
      s : Set E
      x : E
      x✝ : Membership.mem s x
      h : ∀ ⦃a : 𝕜⦄, LE.le 0 a → LE.le a 1 → Membership.mem s (HSMul.hSMul a x)
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a 0) (HSMul.hSMul b x))
    -/
  · rw [smul_zero, zero_add]
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedRing 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : SMulWithZero 𝕜 E
      s : Set E
      x : E
      x✝ : Membership.mem s x
      h : ∀ ⦃a : 𝕜⦄, LE.le 0 a → LE.le a 1 → Membership.mem s (HSMul.hSMul a x)
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Membership.mem s (HSMul.hSMul b x)
    -/
    exact h hb (by rw [← hab]; exact le_add_of_nonneg_left ha)
    /-
      🎉 no goals
    -/


theorem StarConvex.add_smul_mem (hs : StarConvex 𝕜 x s) (hy : x + y ∈ s) {t : 𝕜} (ht₀ : 0 ≤ t)
    (ht₁ : t ≤ 1) : x + t • y ∈ s := by
  have h : x + t • y = (1 - t) • x + t • (x + y) := by
    rw [smul_add, ← add_assoc, ← add_smul, sub_add_cancel, one_smul]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    s : Set E
    hs : StarConvex 𝕜 x s
    hy : Membership.mem s (HAdd.hAdd x y)
    t : 𝕜
    ht₀ : LE.le 0 t
    ht₁ : LE.le t 1
    h : Eq (HAdd.hAdd x (HSMul.hSMul t y)) (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 t) …
    ⊢ Membership.mem s (HAdd.hAdd x (HSMul.hSMul t y))
  -/
  rw [h]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    s : Set E
    hs : StarConvex 𝕜 x s
    hy : Membership.mem s (HAdd.hAdd x y)
    t : 𝕜
    ht₀ : LE.le 0 t
    ht₁ : LE.le t 1
    h : Eq (HAdd.hAdd x (HSMul.hSMul t y)) (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 t) …
    ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 t) x) (HSMul.hSMul t ( …
  -/
  exact hs hy (sub_nonneg_of_le ht₁) ht₀ (sub_add_cancel _ _)
  /-
    🎉 no goals
  -/


theorem StarConvex.smul_mem (hs : StarConvex 𝕜 0 s) (hx : x ∈ s) {t : 𝕜} (ht₀ : 0 ≤ t)
                                    /-
                                      𝕜 : Type u_1
                                      E : Type u_2
                                      inst✝² : OrderedRing 𝕜
                                      inst✝¹ : AddCommGroup E
                                      inst✝ : Module 𝕜 E
                                      x : E
                                      s : Set E
                                      hs : StarConvex 𝕜 0 s
                                      hx : Membership.mem s x
                                      t : 𝕜
                                      ht₀ : LE.le 0 t
                                      ht₁ : LE.le t 1
                                      ⊢ Membership.mem s (HSMul.hSMul t x)
                                    -/
    (ht₁ : t ≤ 1) : t • x ∈ s := by simpa using hs.add_smul_mem (by simpa using hx) ht₀ ht₁
                                    /-
                                      🎉 no goals
                                    -/


theorem StarConvex.add_smul_sub_mem (hs : StarConvex 𝕜 x s) (hy : y ∈ s) {t : 𝕜} (ht₀ : 0 ≤ t)
    (ht₁ : t ≤ 1) : x + t • (y - x) ∈ s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    s : Set E
    hs : StarConvex 𝕜 x s
    hy : Membership.mem s y
    t : 𝕜
    ht₀ : LE.le 0 t
    ht₁ : LE.le t 1
    ⊢ Membership.mem s (HAdd.hAdd x (HSMul.hSMul t (HSub.hSub y x)))
  -/
  apply hs.segment_subset hy
  /-
    case a
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    s : Set E
    hs : StarConvex 𝕜 x s
    hy : Membership.mem s y
    t : 𝕜
    ht₀ : LE.le 0 t
    ht₁ : LE.le t 1
    ⊢ Membership.mem (segment 𝕜 x y) (HAdd.hAdd x (HSMul.hSMul t (HSub.hSub y x)))
  -/
  rw [segment_eq_image']
  /-
    case a
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    s : Set E
    hs : StarConvex 𝕜 x s
    hy : Membership.mem s y
    t : 𝕜
    ht₀ : LE.le 0 t
    ht₁ : LE.le t 1
    ⊢ Membership.mem (Set.image (fun θ => HAdd.hAdd x (HSMul.hSMul θ (HSub.hSub y  …
  -/
  exact mem_image_of_mem _ ⟨ht₀, ht₁⟩
  /-
    🎉 no goals
  -/


/-- The preimage of a star-convex set under an affine map is star-convex. -/
theorem StarConvex.affine_preimage (f : E →ᵃ[𝕜] F) {s : Set F} (hs : StarConvex 𝕜 (f x) s) :
    StarConvex 𝕜 x (f ⁻¹' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    f : AffineMap 𝕜 E F
    s : Set F
    hs : StarConvex 𝕜 (f x) s
    ⊢ StarConvex 𝕜 x (Set.preimage (⇑f) s)
  -/
  intro y hy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    f : AffineMap 𝕜 E F
    s : Set F
    hs : StarConvex 𝕜 (f x) s
    y : E
    hy : Membership.mem (Set.preimage (⇑f) s) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.preimage (⇑f) s) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSM …
  -/
  rw [mem_preimage, Convex.combo_affine_apply hab]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    f : AffineMap 𝕜 E F
    s : Set F
    hs : StarConvex 𝕜 (f x) s
    y : E
    hy : Membership.mem (Set.preimage (⇑f) s) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y)))
  -/
  exact hs hy ha hb hab
  /-
    🎉 no goals
  -/


/-- The image of a star-convex set under an affine map is star-convex. -/
theorem StarConvex.affine_image (f : E →ᵃ[𝕜] F) {s : Set E} (hs : StarConvex 𝕜 x s) :
    StarConvex 𝕜 (f x) (f '' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    f : AffineMap 𝕜 E F
    s : Set E
    hs : StarConvex 𝕜 x s
    ⊢ StarConvex 𝕜 (f x) (Set.image (⇑f) s)
  -/
  rintro y ⟨y', ⟨hy', hy'f⟩⟩ a b ha hb hab
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    f : AffineMap 𝕜 E F
    s : Set E
    hs : StarConvex 𝕜 x s
    y : F
    y' : E
    hy' : Membership.mem s y'
    hy'f : Eq (f y') y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.image (⇑f) s) (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hS …
  -/
  refine ⟨a • x + b • y', ⟨hs hy' ha hb hab, ?_⟩⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x : E
    f : AffineMap 𝕜 E F
    s : Set E
    hs : StarConvex 𝕜 x s
    y : F
    y' : E
    hy' : Membership.mem s y'
    hy'f : Eq (f y') y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Eq (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y'))) (HAdd.hAdd (HSMul.hS …
  -/
  rw [Convex.combo_affine_apply hab, hy'f]
  /-
    🎉 no goals
  -/


theorem StarConvex.neg (hs : StarConvex 𝕜 x s) : StarConvex 𝕜 (-x) (-s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    ⊢ StarConvex 𝕜 (Neg.neg x) (Neg.neg s)
  -/
  rw [← image_neg_eq_neg]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 x s
    ⊢ StarConvex 𝕜 (Neg.neg x) (Set.image (fun x => Neg.neg x) s)
  -/
  exact hs.is_linear_image IsLinearMap.isLinearMap_neg
  /-
    🎉 no goals
  -/


theorem StarConvex.sub (hs : StarConvex 𝕜 x s) (ht : StarConvex 𝕜 y t) :
    StarConvex 𝕜 (x - y) (s - t) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    s t : Set E
    hs : StarConvex 𝕜 x s
    ht : StarConvex 𝕜 y t
    ⊢ StarConvex 𝕜 (HSub.hSub x y) (HSub.hSub s t)
  -/
  simp_rw [sub_eq_add_neg]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x y : E
    s t : Set E
    hs : StarConvex 𝕜 x s
    ht : StarConvex 𝕜 y t
    ⊢ StarConvex 𝕜 (HAdd.hAdd x (Neg.neg y)) (HAdd.hAdd s (Neg.neg t))
  -/
  exact hs.add ht.neg
  /-
    🎉 no goals
  -/


/-- If `x < y`, then `(Set.Iic x)ᶜ` is star convex at `y`. -/
lemma starConvex_compl_Iic (h : x < y) : StarConvex 𝕜 y (Iic x)ᶜ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedRing 𝕜
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : OrderedSMul 𝕜 E
    x y : E
    h : LT.lt x y
    ⊢ StarConvex 𝕜 y (HasCompl.compl (Set.Iic x))
  -/
  refine (starConvex_iff_forall_pos <| by simp [h.not_le]).mpr fun z hz a b ha hb hab ↦ ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedRing 𝕜
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : OrderedSMul 𝕜 E
    x y : E
    h : LT.lt x y
    z : E
    hz : Membership.mem (HasCompl.compl (Set.Iic x)) z
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (HasCompl.compl (Set.Iic x)) (HAdd.hAdd (HSMul.hSMul a y) (HS …
  -/
  rw [mem_compl_iff, mem_Iic] at hz ⊢
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedRing 𝕜
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : OrderedSMul 𝕜 E
    x y : E
    h : LT.lt x y
    z : E
    hz : Not (LE.le z x)
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Not (LE.le (HAdd.hAdd (HSMul.hSMul a y) (HSMul.hSMul b z)) x)
  -/
  contrapose! hz
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedRing 𝕜
    inst✝² : OrderedAddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : OrderedSMul 𝕜 E
    x y : E
    h : LT.lt x y
    z : E
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hz : LE.le (HAdd.hAdd (HSMul.hSMul a y) (HSMul.hSMul b z)) x
    ⊢ LE.le z x
  -/
  refine (lt_of_smul_lt_smul_of_nonneg_left ?_ hb.le).le
  calc
    b • z ≤ (a + b) • x - a • y := by rwa [le_sub_iff_add_le', hab, one_smul]
    _ < b • x := by
      rw [add_smul, sub_lt_iff_lt_add']
      gcongr


/-- If `x < y`, then `(Set.Ici y)ᶜ` is star convex at `x`. -/
lemma starConvex_compl_Ici (h : x < y) : StarConvex 𝕜 x (Ici y)ᶜ :=
  starConvex_compl_Iic (E := Eᵒᵈ) h


/-- Alternative definition of star-convexity, using division. -/
theorem starConvex_iff_div : StarConvex 𝕜 x s ↔ ∀ ⦃y⦄, y ∈ s →
    ∀ ⦃a b : 𝕜⦄, 0 ≤ a → 0 ≤ b → 0 < a + b → (a / (a + b)) • x + (b / (a + b)) • y ∈ s :=
  ⟨fun h y hy a b ha hb hab => by
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x : E
      s : Set E
      h : StarConvex 𝕜 x s
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : LT.lt 0 (HAdd.hAdd a b)
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv a (HAdd.hAdd a b)) x) (H …
    -/
    apply h hy
      /-
        case a
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : Module 𝕜 E
        x : E
        s : Set E
        h : StarConvex 𝕜 x s
        y : E
        hy : Membership.mem s y
        a b : 𝕜
        ha : LE.le 0 a
        hb : LE.le 0 b
        hab : LT.lt 0 (HAdd.hAdd a b)
        ⊢ LE.le 0 (HDiv.hDiv a (HAdd.hAdd a b))
      -/
    · positivity
      /-
        🎉 no goals
      -/
      /-
        case a
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : Module 𝕜 E
        x : E
        s : Set E
        h : StarConvex 𝕜 x s
        y : E
        hy : Membership.mem s y
        a b : 𝕜
        ha : LE.le 0 a
        hb : LE.le 0 b
        hab : LT.lt 0 (HAdd.hAdd a b)
        ⊢ LE.le 0 (HDiv.hDiv b (HAdd.hAdd a b))
      -/
    · positivity
      /-
        🎉 no goals
      -/
      /-
        case a
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : Module 𝕜 E
        x : E
        s : Set E
        h : StarConvex 𝕜 x s
        y : E
        hy : Membership.mem s y
        a b : 𝕜
        ha : LE.le 0 a
        hb : LE.le 0 b
        hab : LT.lt 0 (HAdd.hAdd a b)
        ⊢ Eq (HAdd.hAdd (HDiv.hDiv a (HAdd.hAdd a b)) (HDiv.hDiv b (HAdd.hAdd a b))) 1
      -/
    · rw [← add_div]
      /-
        case a
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : Module 𝕜 E
        x : E
        s : Set E
        h : StarConvex 𝕜 x s
        y : E
        hy : Membership.mem s y
        a b : 𝕜
        ha : LE.le 0 a
        hb : LE.le 0 b
        hab : LT.lt 0 (HAdd.hAdd a b)
        ⊢ Eq (HDiv.hDiv (HAdd.hAdd a b) (HAdd.hAdd a b)) 1
      -/
      exact div_self hab.ne',
      /-
        🎉 no goals
      -/
  fun h y hy a b ha hb hab => by
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x : E
      s : Set E
      h : ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜⦄, LE.le 0 a → LE.le 0 b → LT.lt …
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
    -/
    have h' := h hy ha hb
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x : E
      s : Set E
      h : ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜⦄, LE.le 0 a → LE.le 0 b → LT.lt …
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      h' : LT.lt 0 (HAdd.hAdd a b) → Membership.mem s (HAdd.hAdd (HSMul.hSMul (HDiv. …
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
    -/
    rw [hab, div_one, div_one] at h'
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      x : E
      s : Set E
      h : ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜⦄, LE.le 0 a → LE.le 0 b → LT.lt …
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      h' : LT.lt 0 1 → Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b  …
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
    -/
    exact h' zero_lt_one⟩
    /-
      🎉 no goals
    -/


theorem StarConvex.mem_smul (hs : StarConvex 𝕜 0 s) (hx : x ∈ s) {t : 𝕜} (ht : 1 ≤ t) :
    x ∈ t • s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 0 s
    hx : Membership.mem s x
    t : 𝕜
    ht : LE.le 1 t
    ⊢ Membership.mem (HSMul.hSMul t s) x
  -/
  rw [mem_smul_set_iff_inv_smul_mem₀ (zero_lt_one.trans_le ht).ne']
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x : E
    s : Set E
    hs : StarConvex 𝕜 0 s
    hx : Membership.mem s x
    t : 𝕜
    ht : LE.le 1 t
    ⊢ Membership.mem s (HSMul.hSMul (Inv.inv t) x)
  -/
  exact hs.smul_mem hx (by positivity) (inv_le_one_of_one_le₀ ht)
  /-
    🎉 no goals
  -/


/-- If `s` is an order-connected set in an ordered module over an ordered semiring
and all elements of `s` are comparable with `x ∈ s`, then `s` is `StarConvex` at `x`. -/
theorem Set.OrdConnected.starConvex [OrderedSemiring 𝕜] [OrderedAddCommMonoid E] [Module 𝕜 E]
    [OrderedSMul 𝕜 E] {x : E} {s : Set E} (hs : s.OrdConnected) (hx : x ∈ s)
    (h : ∀ y ∈ s, x ≤ y ∨ y ≤ x) : StarConvex 𝕜 x s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : OrderedAddCommMonoid E
    inst✝¹ : Module 𝕜 E
    inst✝ : OrderedSMul 𝕜 E
    x : E
    s : Set E
    hs : s.OrdConnected
    hx : Membership.mem s x
    h : ∀ (y : E), Membership.mem s y → Or (LE.le x y) (LE.le y x)
    ⊢ StarConvex 𝕜 x s
  -/
  intro y hy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : OrderedAddCommMonoid E
    inst✝¹ : Module 𝕜 E
    inst✝ : OrderedSMul 𝕜 E
    x : E
    s : Set E
    hs : s.OrdConnected
    hx : Membership.mem s x
    h : ∀ (y : E), Membership.mem s y → Or (LE.le x y) (LE.le y x)
    y : E
    hy : Membership.mem s y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
  -/
  obtain hxy | hyx := h _ hy
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : OrderedSemiring 𝕜
      inst✝² : OrderedAddCommMonoid E
      inst✝¹ : Module 𝕜 E
      inst✝ : OrderedSMul 𝕜 E
      x : E
      s : Set E
      hs : s.OrdConnected
      hx : Membership.mem s x
      h : ∀ (y : E), Membership.mem s y → Or (LE.le x y) (LE.le y x)
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hxy : LE.le x y
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
    -/
  · refine hs.out hx hy (mem_Icc.2 ⟨?_, ?_⟩)
    · calc
        x = a • x + b • x := (Convex.combo_self hab _).symm
        _ ≤ a • x + b • y := by gcongr
    calc
      a • x + b • y ≤ a • y + b • y := by gcongr
      _ = y := Convex.combo_self hab _
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : OrderedSemiring 𝕜
      inst✝² : OrderedAddCommMonoid E
      inst✝¹ : Module 𝕜 E
      inst✝ : OrderedSMul 𝕜 E
      x : E
      s : Set E
      hs : s.OrdConnected
      hx : Membership.mem s x
      h : ∀ (y : E), Membership.mem s y → Or (LE.le x y) (LE.le y x)
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hyx : LE.le y x
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
    -/
  · refine hs.out hy hx (mem_Icc.2 ⟨?_, ?_⟩)
    · calc
        y = a • y + b • y := (Convex.combo_self hab _).symm
        _ ≤ a • x + b • y := by gcongr
    calc
      a • x + b • y ≤ a • x + b • x := by gcongr
      _ = x := Convex.combo_self hab _


theorem starConvex_iff_ordConnected [LinearOrderedField 𝕜] {x : 𝕜} {s : Set 𝕜} (hx : x ∈ s) :
    StarConvex 𝕜 x s ↔ s.OrdConnected := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    ⊢ Iff (StarConvex 𝕜 x s) s.OrdConnected
  -/
  simp_rw [ordConnected_iff_uIcc_subset_left hx, starConvex_iff_segment_subset, segment_eq_uIcc]
  /-
    🎉 no goals
  -/


alias ⟨StarConvex.ordConnected, _⟩ := starConvex_iff_ordConnected


