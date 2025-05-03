/-- Convexity of sets. -/
def Convex : Prop :=
  ∀ ⦃x : E⦄, x ∈ s → StarConvex 𝕜 x s


theorem Convex.starConvex (hs : Convex 𝕜 s) (hx : x ∈ s) : StarConvex 𝕜 x s :=
  hs hx


theorem convex_iff_segment_subset : Convex 𝕜 s ↔ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → [x -[𝕜] y] ⊆ s :=
  forall₂_congr fun _ _ => starConvex_iff_segment_subset


theorem Convex.segment_subset (h : Convex 𝕜 s) {x y : E} (hx : x ∈ s) (hy : y ∈ s) :
    [x -[𝕜] y] ⊆ s :=
  convex_iff_segment_subset.1 h hx hy


theorem Convex.openSegment_subset (h : Convex 𝕜 s) {x y : E} (hx : x ∈ s) (hy : y ∈ s) :
    openSegment 𝕜 x y ⊆ s :=
  (openSegment_subset_segment 𝕜 x y).trans (h.segment_subset hx hy)


/-- Alternative definition of set convexity, in terms of pointwise set operations. -/
theorem convex_iff_pointwise_add_subset :
    Convex 𝕜 s ↔ ∀ ⦃a b : 𝕜⦄, 0 ≤ a → 0 ≤ b → a + b = 1 → a • s + b • s ⊆ s :=
  Iff.intro
    (by
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommMonoid E
        inst✝ : SMul 𝕜 E
        s : Set E
        ⊢ Convex 𝕜 s → ∀ ⦃a b : 𝕜⦄, LE.le 0 a → LE.le 0 b → Eq (HAdd.hAdd a b) 1 → Has …
      -/
      rintro hA a b ha hb hab w ⟨au, ⟨u, hu, rfl⟩, bv, ⟨v, hv, rfl⟩, rfl⟩
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommMonoid E
        inst✝ : SMul 𝕜 E
        s : Set E
        hA : Convex 𝕜 s
        a b : 𝕜
        ha : LE.le 0 a
        hb : LE.le 0 b
        hab : Eq (HAdd.hAdd a b) 1
        u : E
        hu : Membership.mem s u
        v : E
        hv : Membership.mem s v
        ⊢ Membership.mem s ((fun x1 x2 => HAdd.hAdd x1 x2) ((fun x => HSMul.hSMul a x) …
      -/
      exact hA hu hv ha hb hab)
      /-
        🎉 no goals
      -/
    fun h _ hx _ hy _ _ ha hb hab => (h ha hb hab) (Set.add_mem_add ⟨_, hx, rfl⟩ ⟨_, hy, rfl⟩)


alias ⟨Convex.set_combo_subset, _⟩ := convex_iff_pointwise_add_subset


theorem convex_empty : Convex 𝕜 (∅ : Set E) := fun _ => False.elim


theorem convex_univ : Convex 𝕜 (Set.univ : Set E) := fun _ _ => starConvex_univ _


theorem Convex.inter {t : Set E} (hs : Convex 𝕜 s) (ht : Convex 𝕜 t) : Convex 𝕜 (s ∩ t) :=
  fun _ hx => (hs hx.1).inter (ht hx.2)


theorem convex_sInter {S : Set (Set E)} (h : ∀ s ∈ S, Convex 𝕜 s) : Convex 𝕜 (⋂₀ S) := fun _ hx =>
  starConvex_sInter fun _ hs => h _ hs <| hx _ hs


theorem convex_iInter {ι : Sort*} {s : ι → Set E} (h : ∀ i, Convex 𝕜 (s i)) :
    Convex 𝕜 (⋂ i, s i) :=
  sInter_range s ▸ convex_sInter <| forall_mem_range.2 h


theorem convex_iInter₂ {ι : Sort*} {κ : ι → Sort*} {s : ∀ i, κ i → Set E}
    (h : ∀ i j, Convex 𝕜 (s i j)) : Convex 𝕜 (⋂ (i) (j), s i j) :=
  convex_iInter fun i => convex_iInter <| h i


theorem Convex.prod {s : Set E} {t : Set F} (hs : Convex 𝕜 s) (ht : Convex 𝕜 t) :
    Convex 𝕜 (s ×ˢ t) := fun _ hx => (hs hx.1).prod (ht hx.2)


theorem convex_pi {ι : Type*} {E : ι → Type*} [∀ i, AddCommMonoid (E i)] [∀ i, SMul 𝕜 (E i)]
    {s : Set ι} {t : ∀ i, Set (E i)} (ht : ∀ ⦃i⦄, i ∈ s → Convex 𝕜 (t i)) : Convex 𝕜 (s.pi t) :=
  fun _ hx => starConvex_pi fun _ hi => ht hi <| hx _ hi


theorem Directed.convex_iUnion {ι : Sort*} {s : ι → Set E} (hdir : Directed (· ⊆ ·) s)
    (hc : ∀ ⦃i : ι⦄, Convex 𝕜 (s i)) : Convex 𝕜 (⋃ i, s i) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    ι : Sort u_5
    s : ι → Set E
    hdir : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    hc : ∀ ⦃i : ι⦄, Convex 𝕜 (s i)
    ⊢ Convex 𝕜 (Set.iUnion fun i => s i)
  -/
  rintro x hx y hy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    ι : Sort u_5
    s : ι → Set E
    hdir : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    hc : ∀ ⦃i : ι⦄, Convex 𝕜 (s i)
    x : E
    hx : Membership.mem (Set.iUnion fun i => s i) x
    y : E
    hy : Membership.mem (Set.iUnion fun i => s i) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.iUnion fun i => s i) (HAdd.hAdd (HSMul.hSMul a x) (HSMul …
  -/
  rw [mem_iUnion] at hx hy ⊢
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    ι : Sort u_5
    s : ι → Set E
    hdir : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    hc : ∀ ⦃i : ι⦄, Convex 𝕜 (s i)
    x : E
    hx : Exists fun i => Membership.mem (s i) x
    y : E
    hy : Exists fun i => Membership.mem (s i) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Exists fun i => Membership.mem (s i) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSM …
  -/
  obtain ⟨i, hx⟩ := hx
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    ι : Sort u_5
    s : ι → Set E
    hdir : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    hc : ∀ ⦃i : ι⦄, Convex 𝕜 (s i)
    x y : E
    hy : Exists fun i => Membership.mem (s i) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    i : ι
    hx : Membership.mem (s i) x
    ⊢ Exists fun i => Membership.mem (s i) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSM …
  -/
  obtain ⟨j, hy⟩ := hy
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    ι : Sort u_5
    s : ι → Set E
    hdir : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    hc : ∀ ⦃i : ι⦄, Convex 𝕜 (s i)
    x y : E
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    i : ι
    hx : Membership.mem (s i) x
    j : ι
    hy : Membership.mem (s j) y
    ⊢ Exists fun i => Membership.mem (s i) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSM …
  -/
  obtain ⟨k, hik, hjk⟩ := hdir i j
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    ι : Sort u_5
    s : ι → Set E
    hdir : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    hc : ∀ ⦃i : ι⦄, Convex 𝕜 (s i)
    x y : E
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    i : ι
    hx : Membership.mem (s i) x
    j : ι
    hy : Membership.mem (s j) y
    k : ι
    hik : HasSubset.Subset (s i) (s k)
    hjk : HasSubset.Subset (s j) (s k)
    ⊢ Exists fun i => Membership.mem (s i) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSM …
  -/
  exact ⟨k, hc (hik hx) (hjk hy) ha hb hab⟩
  /-
    🎉 no goals
  -/


theorem DirectedOn.convex_sUnion {c : Set (Set E)} (hdir : DirectedOn (· ⊆ ·) c)
    (hc : ∀ ⦃A : Set E⦄, A ∈ c → Convex 𝕜 A) : Convex 𝕜 (⋃₀ c) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    c : Set (Set E)
    hdir : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) c
    hc : ∀ ⦃A : Set E⦄, Membership.mem c A → Convex 𝕜 A
    ⊢ Convex 𝕜 c.sUnion
  -/
  rw [sUnion_eq_iUnion]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    c : Set (Set E)
    hdir : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) c
    hc : ∀ ⦃A : Set E⦄, Membership.mem c A → Convex 𝕜 A
    ⊢ Convex 𝕜 (Set.iUnion fun i => ↑i)
  -/
  exact (directedOn_iff_directed.1 hdir).convex_iUnion fun A => hc A.2
  /-
    🎉 no goals
  -/


theorem convex_iff_openSegment_subset :
    Convex 𝕜 s ↔ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → openSegment 𝕜 x y ⊆ s :=
  forall₂_congr fun _ => starConvex_iff_openSegment_subset


theorem convex_iff_forall_pos :
    Convex 𝕜 s ↔
      ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b → a + b = 1 → a • x + b • y ∈ s :=
  forall₂_congr fun _ => starConvex_iff_forall_pos


theorem convex_iff_pairwise_pos : Convex 𝕜 s ↔
    s.Pairwise fun x y => ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b → a + b = 1 → a • x + b • y ∈ s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ Iff (Convex 𝕜 s) (s.Pairwise fun x y => ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b → …
  -/
  refine convex_iff_forall_pos.trans ⟨fun h x hx y hy _ => h hx hy, ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ (s.Pairwise fun x y => ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b → Eq (HAdd.hAdd a  …
  -/
  intro h x hx y hy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    h : s.Pairwise fun x y => ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b → Eq (HAdd.hAdd a …
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
  -/
  obtain rfl | hxy := eq_or_ne x y
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      s : Set E
      h : s.Pairwise fun x y => ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b → Eq (HAdd.hAdd a …
      x : E
      hx : Membership.mem s x
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hy : Membership.mem s x
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b x))
    -/
  · rwa [Convex.combo_self hab]
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
      s : Set E
      h : s.Pairwise fun x y => ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b → Eq (HAdd.hAdd a …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hxy : Ne x y
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
    -/
  · exact h hx hy hxy ha hb hab
    /-
      🎉 no goals
    -/


theorem Convex.starConvex_iff (hs : Convex 𝕜 s) (h : s.Nonempty) : StarConvex 𝕜 x s ↔ x ∈ s :=
  ⟨fun hxs => hxs.mem h, hs.starConvex⟩


protected theorem Set.Subsingleton.convex {s : Set E} (h : s.Subsingleton) : Convex 𝕜 s :=
  convex_iff_pairwise_pos.mpr (h.pairwise _)


theorem convex_singleton (c : E) : Convex 𝕜 ({c} : Set E) :=
  subsingleton_singleton.convex


theorem convex_zero : Convex 𝕜 (0 : Set E) :=
  convex_singleton _


theorem convex_segment (x y : E) : Convex 𝕜 [x -[𝕜] y] := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    x y : E
    ⊢ Convex 𝕜 (segment 𝕜 x y)
  -/
  rintro p ⟨ap, bp, hap, hbp, habp, rfl⟩ q ⟨aq, bq, haq, hbq, habq, rfl⟩ a b ha hb hab
  refine
    ⟨a * ap + b * aq, a * bp + b * bq, add_nonneg (mul_nonneg ha hap) (mul_nonneg hb haq),
      add_nonneg (mul_nonneg ha hbp) (mul_nonneg hb hbq), ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      x y : E
      ap bp : 𝕜
      hap : LE.le 0 ap
      hbp : LE.le 0 bp
      habp : Eq (HAdd.hAdd ap bp) 1
      aq bq : 𝕜
      haq : LE.le 0 aq
      hbq : LE.le 0 bq
      habq : Eq (HAdd.hAdd aq bq) 1
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a ap) (HMul.hMul b aq)) (HAdd.hAdd (HMul …
    -/
  · rw [add_add_add_comm, ← mul_add, ← mul_add, habp, habq, mul_one, mul_one, hab]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      x y : E
      ap bp : 𝕜
      hap : LE.le 0 ap
      hbp : LE.le 0 bp
      habp : Eq (HAdd.hAdd ap bp) 1
      aq bq : 𝕜
      haq : LE.le 0 aq
      hbq : LE.le 0 bq
      habq : Eq (HAdd.hAdd aq bq) 1
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (HMul.hMul a ap) (HMul.hMul b aq)) x)  …
    -/
                      /-
                        🎉 no goals
                      -/
  · match_scalars <;> noncomm_ring
                      /-
                        🎉 no goals
                      -/


theorem Convex.linear_image (hs : Convex 𝕜 s) (f : E →ₗ[𝕜] F) : Convex 𝕜 (f '' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set E
    hs : Convex 𝕜 s
    f : LinearMap (RingHom.id 𝕜) E F
    ⊢ Convex 𝕜 (Set.image (⇑f) s)
  -/
  rintro _ ⟨x, hx, rfl⟩ _ ⟨y, hy, rfl⟩ a b ha hb hab
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set E
    hs : Convex 𝕜 s
    f : LinearMap (RingHom.id 𝕜) E F
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.image (⇑f) s) (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hS …
  -/
  exact ⟨a • x + b • y, hs hx hy ha hb hab, by rw [f.map_add, f.map_smul, f.map_smul]⟩
  /-
    🎉 no goals
  -/


theorem Convex.is_linear_image (hs : Convex 𝕜 s) {f : E → F} (hf : IsLinearMap 𝕜 f) :
    Convex 𝕜 (f '' s) :=
  hs.linear_image <| hf.mk' f


theorem Convex.linear_preimage {s : Set F} (hs : Convex 𝕜 s) (f : E →ₗ[𝕜] F) :
    Convex 𝕜 (f ⁻¹' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set F
    hs : Convex 𝕜 s
    f : LinearMap (RingHom.id 𝕜) E F
    ⊢ Convex 𝕜 (Set.preimage (⇑f) s)
  -/
  intro x hx y hy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set F
    hs : Convex 𝕜 s
    f : LinearMap (RingHom.id 𝕜) E F
    x : E
    hx : Membership.mem (Set.preimage (⇑f) s) x
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
    s : Set F
    hs : Convex 𝕜 s
    f : LinearMap (RingHom.id 𝕜) E F
    x : E
    hx : Membership.mem (Set.preimage (⇑f) s) x
    y : E
    hy : Membership.mem (Set.preimage (⇑f) s) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y)))
  -/
  exact hs hx hy ha hb hab
  /-
    🎉 no goals
  -/


theorem Convex.is_linear_preimage {s : Set F} (hs : Convex 𝕜 s) {f : E → F} (hf : IsLinearMap 𝕜 f) :
    Convex 𝕜 (f ⁻¹' s) :=
  hs.linear_preimage <| hf.mk' f


theorem Convex.add {t : Set E} (hs : Convex 𝕜 s) (ht : Convex 𝕜 t) : Convex 𝕜 (s + t) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Convex 𝕜 s
    ht : Convex 𝕜 t
    ⊢ Convex 𝕜 (HAdd.hAdd s t)
  -/
  rw [← add_image_prod]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Convex 𝕜 s
    ht : Convex 𝕜 t
    ⊢ Convex 𝕜 (Set.image (fun x => HAdd.hAdd x.1 x.2) (SProd.sprod s t))
  -/
  exact (hs.prod ht).is_linear_image IsLinearMap.isLinearMap_add
  /-
    🎉 no goals
  -/


/-- The convex sets form an additive submonoid under pointwise addition. -/
def convexAddSubmonoid : AddSubmonoid (Set E) where
  carrier := {s : Set E | Convex 𝕜 s}
  zero_mem' := convex_zero
  add_mem' := Convex.add


@[simp, norm_cast]
theorem coe_convexAddSubmonoid : ↑(convexAddSubmonoid 𝕜 E) = {s : Set E | Convex 𝕜 s} :=
  rfl


@[simp]
theorem mem_convexAddSubmonoid {s : Set E} : s ∈ convexAddSubmonoid 𝕜 E ↔ Convex 𝕜 s :=
  Iff.rfl


theorem convex_list_sum {l : List (Set E)} (h : ∀ i ∈ l, Convex 𝕜 i) : Convex 𝕜 l.sum :=
  (convexAddSubmonoid 𝕜 E).list_sum_mem h


theorem convex_multiset_sum {s : Multiset (Set E)} (h : ∀ i ∈ s, Convex 𝕜 i) : Convex 𝕜 s.sum :=
  (convexAddSubmonoid 𝕜 E).multiset_sum_mem _ h


theorem convex_sum {ι} {s : Finset ι} (t : ι → Set E) (h : ∀ i ∈ s, Convex 𝕜 (t i)) :
    Convex 𝕜 (∑ i ∈ s, t i) :=
  (convexAddSubmonoid 𝕜 E).sum_mem h


theorem Convex.vadd (hs : Convex 𝕜 s) (z : E) : Convex 𝕜 (z +ᵥ s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    z : E
    ⊢ Convex 𝕜 (HVAdd.hVAdd z s)
  -/
  simp_rw [← image_vadd, vadd_eq_add, ← singleton_add]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    z : E
    ⊢ Convex 𝕜 (HAdd.hAdd (Singleton.singleton z) s)
  -/
  exact (convex_singleton _).add hs
  /-
    🎉 no goals
  -/


theorem Convex.translate (hs : Convex 𝕜 s) (z : E) : Convex 𝕜 ((fun x => z + x) '' s) :=
  hs.vadd _


/-- The translation of a convex set is also convex. -/
theorem Convex.translate_preimage_right (hs : Convex 𝕜 s) (z : E) :
    Convex 𝕜 ((fun x => z + x) ⁻¹' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    z : E
    ⊢ Convex 𝕜 (Set.preimage (fun x => HAdd.hAdd z x) s)
  -/
  intro x hx y hy a b ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    z x : E
    hx : Membership.mem (Set.preimage (fun x => HAdd.hAdd z x) s) x
    y : E
    hy : Membership.mem (Set.preimage (fun x => HAdd.hAdd z x) s) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.preimage (fun x => HAdd.hAdd z x) s) (HAdd.hAdd (HSMul.h …
  -/
  have h := hs hx hy ha hb hab
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    z x : E
    hx : Membership.mem (Set.preimage (fun x => HAdd.hAdd z x) s) x
    y : E
    hy : Membership.mem (Set.preimage (fun x => HAdd.hAdd z x) s) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    h : Membership.mem s (HAdd.hAdd (HSMul.hSMul a ((fun x => HAdd.hAdd z x) x)) ( …
    ⊢ Membership.mem (Set.preimage (fun x => HAdd.hAdd z x) s) (HAdd.hAdd (HSMul.h …
  -/
  rwa [smul_add, smul_add, add_add_add_comm, ← add_smul, hab, one_smul] at h
  /-
    🎉 no goals
  -/


/-- The translation of a convex set is also convex. -/
theorem Convex.translate_preimage_left (hs : Convex 𝕜 s) (z : E) :
    Convex 𝕜 ((fun x => x + z) ⁻¹' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    z : E
    ⊢ Convex 𝕜 (Set.preimage (fun x => HAdd.hAdd x z) s)
  -/
  simpa only [add_comm] using hs.translate_preimage_right z
  /-
    🎉 no goals
  -/


theorem convex_Iic (r : β) : Convex 𝕜 (Iic r) := fun x hx y hy a b ha hb hab =>
  calc
    a • x + b • y ≤ a • r + b • r :=
      add_le_add (smul_le_smul_of_nonneg_left hx ha) (smul_le_smul_of_nonneg_left hy hb)
    _ = r := Convex.combo_self hab _


theorem convex_Ici (r : β) : Convex 𝕜 (Ici r) :=
  @convex_Iic 𝕜 βᵒᵈ _ _ _ _ r


theorem convex_Icc (r s : β) : Convex 𝕜 (Icc r s) :=
  Ici_inter_Iic.subst ((convex_Ici r).inter <| convex_Iic s)


theorem convex_halfSpace_le {f : E → β} (h : IsLinearMap 𝕜 f) (r : β) : Convex 𝕜 { w | f w ≤ r } :=
  (convex_Iic r).is_linear_preimage h

@[deprecated (since := "2024-11-12")] alias convex_halfspace_le := convex_halfSpace_le


theorem convex_halfSpace_ge {f : E → β} (h : IsLinearMap 𝕜 f) (r : β) : Convex 𝕜 { w | r ≤ f w } :=
  (convex_Ici r).is_linear_preimage h

@[deprecated (since := "2024-11-12")] alias convex_halfspace_ge := convex_halfSpace_ge


theorem convex_hyperplane {f : E → β} (h : IsLinearMap 𝕜 f) (r : β) : Convex 𝕜 { w | f w = r } := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : Module 𝕜 E
    inst✝² : OrderedAddCommMonoid β
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    f : E → β
    h : IsLinearMap 𝕜 f
    r : β
    ⊢ Convex 𝕜 (setOf fun w => Eq (f w) r)
  -/
  simp_rw [le_antisymm_iff]
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_4
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : Module 𝕜 E
    inst✝² : OrderedAddCommMonoid β
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    f : E → β
    h : IsLinearMap 𝕜 f
    r : β
    ⊢ Convex 𝕜 (setOf fun w => And (LE.le (f w) r) (LE.le r (f w)))
  -/
  exact (convex_halfSpace_le h r).inter (convex_halfSpace_ge h r)
  /-
    🎉 no goals
  -/


theorem convex_Iio (r : β) : Convex 𝕜 (Iio r) := by
  /-
    𝕜 : Type u_1
    β : Type u_4
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : OrderedCancelAddCommMonoid β
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    r : β
    ⊢ Convex 𝕜 (Set.Iio r)
  -/
  intro x hx y hy a b ha hb hab
  /-
    𝕜 : Type u_1
    β : Type u_4
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : OrderedCancelAddCommMonoid β
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    r x : β
    hx : Membership.mem (Set.Iio r) x
    y : β
    hy : Membership.mem (Set.Iio r) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Set.Iio r) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
  -/
  obtain rfl | ha' := ha.eq_or_lt
    /-
      case inl
      𝕜 : Type u_1
      β : Type u_4
      inst✝³ : OrderedSemiring 𝕜
      inst✝² : OrderedCancelAddCommMonoid β
      inst✝¹ : Module 𝕜 β
      inst✝ : OrderedSMul 𝕜 β
      r x : β
      hx : Membership.mem (Set.Iio r) x
      y : β
      hy : Membership.mem (Set.Iio r) y
      b : 𝕜
      hb : LE.le 0 b
      ha : LE.le 0 0
      hab : Eq (HAdd.hAdd 0 b) 1
      ⊢ Membership.mem (Set.Iio r) (HAdd.hAdd (HSMul.hSMul 0 x) (HSMul.hSMul b y))
    -/
  · rw [zero_add] at hab
    /-
      case inl
      𝕜 : Type u_1
      β : Type u_4
      inst✝³ : OrderedSemiring 𝕜
      inst✝² : OrderedCancelAddCommMonoid β
      inst✝¹ : Module 𝕜 β
      inst✝ : OrderedSMul 𝕜 β
      r x : β
      hx : Membership.mem (Set.Iio r) x
      y : β
      hy : Membership.mem (Set.Iio r) y
      b : 𝕜
      hb : LE.le 0 b
      ha : LE.le 0 0
      hab : Eq b 1
      ⊢ Membership.mem (Set.Iio r) (HAdd.hAdd (HSMul.hSMul 0 x) (HSMul.hSMul b y))
    -/
    rwa [zero_smul, zero_add, hab, one_smul]
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    β : Type u_4
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : OrderedCancelAddCommMonoid β
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    r x : β
    hx : Membership.mem (Set.Iio r) x
    y : β
    hy : Membership.mem (Set.Iio r) y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ha' : LT.lt 0 a
    ⊢ Membership.mem (Set.Iio r) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
  -/
  rw [mem_Iio] at hx hy
  calc
    a • x + b • y < a • r + b • r := add_lt_add_of_lt_of_le
        (smul_lt_smul_of_pos_left hx ha') (smul_le_smul_of_nonneg_left hy.le hb)
    _ = r := Convex.combo_self hab _


theorem convex_Ioi (r : β) : Convex 𝕜 (Ioi r) :=
  convex_Iio (β := βᵒᵈ) r


theorem convex_Ioo (r s : β) : Convex 𝕜 (Ioo r s) :=
  Ioi_inter_Iio.subst ((convex_Ioi r).inter <| convex_Iio s)


theorem convex_Ico (r s : β) : Convex 𝕜 (Ico r s) :=
  Ici_inter_Iio.subst ((convex_Ici r).inter <| convex_Iio s)


theorem convex_Ioc (r s : β) : Convex 𝕜 (Ioc r s) :=
  Ioi_inter_Iic.subst ((convex_Ioi r).inter <| convex_Iic s)


theorem convex_halfSpace_lt {f : E → β} (h : IsLinearMap 𝕜 f) (r : β) : Convex 𝕜 { w | f w < r } :=
  (convex_Iio r).is_linear_preimage h

@[deprecated (since := "2024-11-12")] alias convex_halfspace_lt := convex_halfSpace_lt


theorem convex_halfSpace_gt {f : E → β} (h : IsLinearMap 𝕜 f) (r : β) : Convex 𝕜 { w | r < f w } :=
  (convex_Ioi r).is_linear_preimage h

@[deprecated (since := "2024-11-12")] alias convex_halfspace_gt := convex_halfSpace_gt


theorem convex_uIcc (r s : β) : Convex 𝕜 (uIcc r s) :=
  convex_Icc _ _


theorem MonotoneOn.convex_le (hf : MonotoneOn f s) (hs : Convex 𝕜 s) (r : β) :
    Convex 𝕜 ({ x ∈ s | f x ≤ r }) := fun x hx y hy _ _ ha hb hab =>
  ⟨hs hx.1 hy.1 ha hb hab,
    (hf (hs hx.1 hy.1 ha hb hab) (max_rec' s hx.1 hy.1) (Convex.combo_le_max x y ha hb hab)).trans
      (max_rec' { x | f x ≤ r } hx.2 hy.2)⟩


theorem MonotoneOn.convex_lt (hf : MonotoneOn f s) (hs : Convex 𝕜 s) (r : β) :
    Convex 𝕜 ({ x ∈ s | f x < r }) := fun x hx y hy _ _ ha hb hab =>
  ⟨hs hx.1 hy.1 ha hb hab,
    (hf (hs hx.1 hy.1 ha hb hab) (max_rec' s hx.1 hy.1)
          (Convex.combo_le_max x y ha hb hab)).trans_lt
      (max_rec' { x | f x < r } hx.2 hy.2)⟩


theorem MonotoneOn.convex_ge (hf : MonotoneOn f s) (hs : Convex 𝕜 s) (r : β) :
    Convex 𝕜 ({ x ∈ s | r ≤ f x }) :=
  MonotoneOn.convex_le (E := Eᵒᵈ) (β := βᵒᵈ) hf.dual hs r


theorem MonotoneOn.convex_gt (hf : MonotoneOn f s) (hs : Convex 𝕜 s) (r : β) :
    Convex 𝕜 ({ x ∈ s | r < f x }) :=
  MonotoneOn.convex_lt (E := Eᵒᵈ) (β := βᵒᵈ) hf.dual hs r


theorem AntitoneOn.convex_le (hf : AntitoneOn f s) (hs : Convex 𝕜 s) (r : β) :
    Convex 𝕜 ({ x ∈ s | f x ≤ r }) :=
  MonotoneOn.convex_ge (β := βᵒᵈ) hf hs r


theorem AntitoneOn.convex_lt (hf : AntitoneOn f s) (hs : Convex 𝕜 s) (r : β) :
    Convex 𝕜 ({ x ∈ s | f x < r }) :=
  MonotoneOn.convex_gt (β := βᵒᵈ) hf hs r


theorem AntitoneOn.convex_ge (hf : AntitoneOn f s) (hs : Convex 𝕜 s) (r : β) :
    Convex 𝕜 ({ x ∈ s | r ≤ f x }) :=
  MonotoneOn.convex_le (β := βᵒᵈ) hf hs r


theorem AntitoneOn.convex_gt (hf : AntitoneOn f s) (hs : Convex 𝕜 s) (r : β) :
    Convex 𝕜 ({ x ∈ s | r < f x }) :=
  MonotoneOn.convex_lt (β := βᵒᵈ)  hf hs r


theorem Monotone.convex_le (hf : Monotone f) (r : β) : Convex 𝕜 { x | f x ≤ r } :=
  Set.sep_univ.subst ((hf.monotoneOn univ).convex_le convex_univ r)


theorem Monotone.convex_lt (hf : Monotone f) (r : β) : Convex 𝕜 { x | f x ≤ r } :=
  Set.sep_univ.subst ((hf.monotoneOn univ).convex_le convex_univ r)


theorem Monotone.convex_ge (hf : Monotone f) (r : β) : Convex 𝕜 { x | r ≤ f x } :=
  Set.sep_univ.subst ((hf.monotoneOn univ).convex_ge convex_univ r)


theorem Monotone.convex_gt (hf : Monotone f) (r : β) : Convex 𝕜 { x | f x ≤ r } :=
  Set.sep_univ.subst ((hf.monotoneOn univ).convex_le convex_univ r)


theorem Antitone.convex_le (hf : Antitone f) (r : β) : Convex 𝕜 { x | f x ≤ r } :=
  Set.sep_univ.subst ((hf.antitoneOn univ).convex_le convex_univ r)


theorem Antitone.convex_lt (hf : Antitone f) (r : β) : Convex 𝕜 { x | f x < r } :=
  Set.sep_univ.subst ((hf.antitoneOn univ).convex_lt convex_univ r)


theorem Antitone.convex_ge (hf : Antitone f) (r : β) : Convex 𝕜 { x | r ≤ f x } :=
  Set.sep_univ.subst ((hf.antitoneOn univ).convex_ge convex_univ r)


theorem Antitone.convex_gt (hf : Antitone f) (r : β) : Convex 𝕜 { x | r < f x } :=
  Set.sep_univ.subst ((hf.antitoneOn univ).convex_gt convex_univ r)


theorem Convex.smul (hs : Convex 𝕜 s) (c : 𝕜) : Convex 𝕜 (c • s) :=
  hs.linear_image (LinearMap.lsmul _ _ c)


theorem Convex.smul_preimage (hs : Convex 𝕜 s) (c : 𝕜) : Convex 𝕜 ((fun z => c • z) ⁻¹' s) :=
  hs.linear_preimage (LinearMap.lsmul _ _ c)


theorem Convex.affinity (hs : Convex 𝕜 s) (z : E) (c : 𝕜) :
    Convex 𝕜 ((fun x => z + c • x) '' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedCommSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    z : E
    c : 𝕜
    ⊢ Convex 𝕜 (Set.image (fun x => HAdd.hAdd z (HSMul.hSMul c x)) s)
  -/
  simpa only [← image_smul, ← image_vadd, image_image] using (hs.smul c).vadd z
  /-
    🎉 no goals
  -/


theorem convex_openSegment (a b : E) : Convex 𝕜 (openSegment 𝕜 a b) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : StrictOrderedCommSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    a b : E
    ⊢ Convex 𝕜 (openSegment 𝕜 a b)
  -/
  rw [convex_iff_openSegment_subset]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : StrictOrderedCommSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    a b : E
    ⊢ ∀ ⦃x : E⦄, Membership.mem (openSegment 𝕜 a b) x → ∀ ⦃y : E⦄, Membership.mem  …
  -/
  rintro p ⟨ap, bp, hap, hbp, habp, rfl⟩ q ⟨aq, bq, haq, hbq, habq, rfl⟩ z ⟨a, b, ha, hb, hab, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : StrictOrderedCommSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    a✝ b✝ : E
    ap bp : 𝕜
    hap : LT.lt 0 ap
    hbp : LT.lt 0 bp
    habp : Eq (HAdd.hAdd ap bp) 1
    aq bq : 𝕜
    haq : LT.lt 0 aq
    hbq : LT.lt 0 bq
    habq : Eq (HAdd.hAdd aq bq) 1
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (openSegment 𝕜 a✝ b✝) (HAdd.hAdd (HSMul.hSMul a (HAdd.hAdd (H …
  -/
  refine ⟨a * ap + b * aq, a * bp + b * bq, by positivity, by positivity, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : StrictOrderedCommSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      a✝ b✝ : E
      ap bp : 𝕜
      hap : LT.lt 0 ap
      hbp : LT.lt 0 bp
      habp : Eq (HAdd.hAdd ap bp) 1
      aq bq : 𝕜
      haq : LT.lt 0 aq
      hbq : LT.lt 0 bq
      habq : Eq (HAdd.hAdd aq bq) 1
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a ap) (HMul.hMul b aq)) (HAdd.hAdd (HMul …
    -/
  · linear_combination (norm := noncomm_ring) a * habp + b * habq + hab
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : StrictOrderedCommSemiring 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      a✝ b✝ : E
      ap bp : 𝕜
      hap : LT.lt 0 ap
      hbp : LT.lt 0 bp
      habp : Eq (HAdd.hAdd ap bp) 1
      aq bq : 𝕜
      haq : LT.lt 0 aq
      hbq : LT.lt 0 bq
      habq : Eq (HAdd.hAdd aq bq) 1
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (HMul.hMul a ap) (HMul.hMul b aq)) a✝) …
    -/
  · module
    /-
      🎉 no goals
    -/


@[simp]
theorem convex_vadd (a : E) : Convex 𝕜 (a +ᵥ s) ↔ Convex 𝕜 s :=
              /-
                𝕜 : Type u_1
                E : Type u_2
                inst✝² : OrderedRing 𝕜
                inst✝¹ : AddCommGroup E
                inst✝ : Module 𝕜 E
                s : Set E
                a : E
                h : Convex 𝕜 (HVAdd.hVAdd a s)
                ⊢ Convex 𝕜 s
              -/
  ⟨fun h ↦ by simpa using h.vadd (-a), fun h ↦ h.vadd _⟩
              /-
                🎉 no goals
              -/


theorem Convex.add_smul_mem (hs : Convex 𝕜 s) {x y : E} (hx : x ∈ s) (hy : x + y ∈ s) {t : 𝕜}
    (ht : t ∈ Icc (0 : 𝕜) 1) : x + t • y ∈ s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    x y : E
    hx : Membership.mem s x
    hy : Membership.mem s (HAdd.hAdd x y)
    t : 𝕜
    ht : Membership.mem (Set.Icc 0 1) t
    ⊢ Membership.mem s (HAdd.hAdd x (HSMul.hSMul t y))
  -/
  have h : x + t • y = (1 - t) • x + t • (x + y) := by match_scalars <;> noncomm_ring
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    x y : E
    hx : Membership.mem s x
    hy : Membership.mem s (HAdd.hAdd x y)
    t : 𝕜
    ht : Membership.mem (Set.Icc 0 1) t
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
    s : Set E
    hs : Convex 𝕜 s
    x y : E
    hx : Membership.mem s x
    hy : Membership.mem s (HAdd.hAdd x y)
    t : 𝕜
    ht : Membership.mem (Set.Icc 0 1) t
    h : Eq (HAdd.hAdd x (HSMul.hSMul t y)) (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 t) …
    ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 t) x) (HSMul.hSMul t ( …
  -/
  exact hs hx hy (sub_nonneg_of_le ht.2) ht.1 (sub_add_cancel _ _)
  /-
    🎉 no goals
  -/


theorem Convex.smul_mem_of_zero_mem (hs : Convex 𝕜 s) {x : E} (zero_mem : (0 : E) ∈ s) (hx : x ∈ s)
    {t : 𝕜} (ht : t ∈ Icc (0 : 𝕜) 1) : t • x ∈ s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    x : E
    zero_mem : Membership.mem s 0
    hx : Membership.mem s x
    t : 𝕜
    ht : Membership.mem (Set.Icc 0 1) t
    ⊢ Membership.mem s (HSMul.hSMul t x)
  -/
  simpa using hs.add_smul_mem zero_mem (by simpa using hx) ht
  /-
    🎉 no goals
  -/


theorem Convex.mapsTo_lineMap (h : Convex 𝕜 s) {x y : E} (hx : x ∈ s) (hy : y ∈ s) :
    MapsTo (AffineMap.lineMap x y) (Icc (0 : 𝕜) 1) s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    h : Convex 𝕜 s
    x y : E
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ Set.MapsTo (⇑(AffineMap.lineMap x y)) (Set.Icc 0 1) s
  -/
  simpa only [mapsTo', segment_eq_image_lineMap] using h.segment_subset hx hy
  /-
    🎉 no goals
  -/


theorem Convex.lineMap_mem (h : Convex 𝕜 s) {x y : E} (hx : x ∈ s) (hy : y ∈ s) {t : 𝕜}
    (ht : t ∈ Icc 0 1) : AffineMap.lineMap x y t ∈ s :=
  h.mapsTo_lineMap hx hy ht


theorem Convex.add_smul_sub_mem (h : Convex 𝕜 s) {x y : E} (hx : x ∈ s) (hy : y ∈ s) {t : 𝕜}
    (ht : t ∈ Icc (0 : 𝕜) 1) : x + t • (y - x) ∈ s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    h : Convex 𝕜 s
    x y : E
    hx : Membership.mem s x
    hy : Membership.mem s y
    t : 𝕜
    ht : Membership.mem (Set.Icc 0 1) t
    ⊢ Membership.mem s (HAdd.hAdd x (HSMul.hSMul t (HSub.hSub y x)))
  -/
  rw [add_comm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    h : Convex 𝕜 s
    x y : E
    hx : Membership.mem s x
    hy : Membership.mem s y
    t : 𝕜
    ht : Membership.mem (Set.Icc 0 1) t
    ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul t (HSub.hSub y x)) x)
  -/
  exact h.lineMap_mem hx hy ht
  /-
    🎉 no goals
  -/


/-- Affine subspaces are convex. -/
theorem AffineSubspace.convex (Q : AffineSubspace 𝕜 E) : Convex 𝕜 (Q : Set E) :=
                                 /-
                                   𝕜 : Type u_1
                                   E : Type u_2
                                   inst✝² : OrderedRing 𝕜
                                   inst✝¹ : AddCommGroup E
                                   inst✝ : Module 𝕜 E
                                   Q : AffineSubspace 𝕜 E
                                   x : E
                                   hx : Membership.mem (↑Q) x
                                   y : E
                                   hy : Membership.mem (↑Q) y
                                   a b : 𝕜
                                   x✝¹ : LE.le 0 a
                                   x✝ : LE.le 0 b
                                   hab : Eq (HAdd.hAdd a b) 1
                                   ⊢ Membership.mem (↑Q) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
                                 -/
  fun x hx y hy a b _ _ hab ↦ by simpa [Convex.combo_eq_smul_sub_add hab] using Q.2 _ hy hx hx
                                 /-
                                   🎉 no goals
                                 -/


/-- The preimage of a convex set under an affine map is convex. -/
theorem Convex.affine_preimage (f : E →ᵃ[𝕜] F) {s : Set F} (hs : Convex 𝕜 s) : Convex 𝕜 (f ⁻¹' s) :=
  fun _ hx => (hs hx).affine_preimage _


/-- The image of a convex set under an affine map is convex. -/
theorem Convex.affine_image (f : E →ᵃ[𝕜] F) (hs : Convex 𝕜 s) : Convex 𝕜 (f '' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : OrderedRing 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    s : Set E
    f : AffineMap 𝕜 E F
    hs : Convex 𝕜 s
    ⊢ Convex 𝕜 (Set.image (⇑f) s)
  -/
  rintro _ ⟨x, hx, rfl⟩
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
    s : Set E
    f : AffineMap 𝕜 E F
    hs : Convex 𝕜 s
    x : E
    hx : Membership.mem s x
    ⊢ StarConvex 𝕜 (f x) (Set.image (⇑f) s)
  -/
  exact (hs hx).affine_image _
  /-
    🎉 no goals
  -/


theorem Convex.neg (hs : Convex 𝕜 s) : Convex 𝕜 (-s) :=
  hs.is_linear_preimage IsLinearMap.isLinearMap_neg


theorem Convex.sub (hs : Convex 𝕜 s) (ht : Convex 𝕜 t) : Convex 𝕜 (s - t) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Convex 𝕜 s
    ht : Convex 𝕜 t
    ⊢ Convex 𝕜 (HSub.hSub s t)
  -/
  rw [sub_eq_add_neg]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s t : Set E
    hs : Convex 𝕜 s
    ht : Convex 𝕜 t
    ⊢ Convex 𝕜 (HAdd.hAdd s (Neg.neg t))
  -/
  exact hs.add ht.neg
  /-
    🎉 no goals
  -/


theorem Convex_subadditive_le [SMul 𝕜 E] {f : E → 𝕜} (hf1 : ∀ x y, f (x + y) ≤ (f x) + (f y))
    (hf2 : ∀ ⦃c⦄ x, 0 ≤ c → f (c • x) ≤ c * f x) (B : 𝕜) :
    Convex 𝕜 { x | f x ≤ B } := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedRing 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    f : E → 𝕜
    hf1 : ∀ (x y : E), LE.le (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    hf2 : ∀ ⦃c : 𝕜⦄ (x : E), LE.le 0 c → LE.le (f (HSMul.hSMul c x)) (HMul.hMul c  …
    B : 𝕜
    ⊢ Convex 𝕜 (setOf fun x => LE.le (f x) B)
  -/
  rw [convex_iff_segment_subset]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedRing 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    f : E → 𝕜
    hf1 : ∀ (x y : E), LE.le (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    hf2 : ∀ ⦃c : 𝕜⦄ (x : E), LE.le 0 c → LE.le (f (HSMul.hSMul c x)) (HMul.hMul c  …
    B : 𝕜
    ⊢ ∀ ⦃x : E⦄, Membership.mem (setOf fun x => LE.le (f x) B) x → ∀ ⦃y : E⦄, Memb …
  -/
  rintro x hx y hy z ⟨a, b, ha, hb, hs, rfl⟩
  calc
    _ ≤ a • (f x) + b • (f y) := le_trans (hf1 _ _) (add_le_add (hf2 x ha) (hf2 y hb))
    _ ≤ a • B + b • B :=
        add_le_add (smul_le_smul_of_nonneg_left hx ha) (smul_le_smul_of_nonneg_left hy hb)
    _ ≤ B := by rw [← add_smul, hs, one_smul]


/-- Alternative definition of set convexity, using division. -/
theorem convex_iff_div :
    Convex 𝕜 s ↔ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s →
      ∀ ⦃a b : 𝕜⦄, 0 ≤ a → 0 ≤ b → 0 < a + b → (a / (a + b)) • x + (b / (a + b)) • y ∈ s :=
  forall₂_congr fun _ _ => starConvex_iff_div


theorem Convex.mem_smul_of_zero_mem (h : Convex 𝕜 s) {x : E} (zero_mem : (0 : E) ∈ s) (hx : x ∈ s)
    {t : 𝕜} (ht : 1 ≤ t) : x ∈ t • s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    h : Convex 𝕜 s
    x : E
    zero_mem : Membership.mem s 0
    hx : Membership.mem s x
    t : 𝕜
    ht : LE.le 1 t
    ⊢ Membership.mem (HSMul.hSMul t s) x
  -/
  rw [mem_smul_set_iff_inv_smul_mem₀ (zero_lt_one.trans_le ht).ne']
  exact h.smul_mem_of_zero_mem zero_mem hx
    ⟨inv_nonneg.2 (zero_le_one.trans ht), inv_le_one_of_one_le₀ ht⟩


theorem Convex.exists_mem_add_smul_eq (h : Convex 𝕜 s) {x y : E} {p q : 𝕜} (hx : x ∈ s) (hy : y ∈ s)
    (hp : 0 ≤ p) (hq : 0 ≤ q) : ∃ z ∈ s, (p + q) • z = p • x + q • y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    h : Convex 𝕜 s
    x y : E
    p q : 𝕜
    hx : Membership.mem s x
    hy : Membership.mem s y
    hp : LE.le 0 p
    hq : LE.le 0 q
    ⊢ Exists fun z => And (Membership.mem s z) (Eq (HSMul.hSMul (HAdd.hAdd p q) z) …
  -/
  rcases _root_.em (p = 0 ∧ q = 0) with (⟨rfl, rfl⟩ | hpq)
    /-
      case inl.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      h : Convex 𝕜 s
      x y : E
      hx : Membership.mem s x
      hy : Membership.mem s y
      hp hq : LE.le 0 0
      ⊢ Exists fun z => And (Membership.mem s z) (Eq (HSMul.hSMul (HAdd.hAdd 0 0) z) …
    -/
  · use x, hx
    /-
      case right
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      h : Convex 𝕜 s
      x y : E
      hx : Membership.mem s x
      hy : Membership.mem s y
      hp hq : LE.le 0 0
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd 0 0) x) (HAdd.hAdd (HSMul.hSMul 0 x) (HSMul.hSMul …
    -/
    simp
    /-
      🎉 no goals
    -/
  · replace hpq : 0 < p + q :=
      (add_nonneg hp hq).lt_of_ne' (mt (add_eq_zero_iff_of_nonneg hp hq).1 hpq)
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      h : Convex 𝕜 s
      x y : E
      p q : 𝕜
      hx : Membership.mem s x
      hy : Membership.mem s y
      hp : LE.le 0 p
      hq : LE.le 0 q
      hpq : LT.lt 0 (HAdd.hAdd p q)
      ⊢ Exists fun z => And (Membership.mem s z) (Eq (HSMul.hSMul (HAdd.hAdd p q) z) …
    -/
    refine ⟨_, convex_iff_div.1 h hx hy hp hq hpq, ?_⟩
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      h : Convex 𝕜 s
      x y : E
      p q : 𝕜
      hx : Membership.mem s x
      hy : Membership.mem s y
      hp : LE.le 0 p
      hq : LE.le 0 q
      hpq : LT.lt 0 (HAdd.hAdd p q)
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd p q) (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv p (HAdd.h …
    -/
                      /-
                        🎉 no goals
                      -/
    match_scalars <;> field_simp
                      /-
                        🎉 no goals
                      -/


theorem Convex.add_smul (h_conv : Convex 𝕜 s) {p q : 𝕜} (hp : 0 ≤ p) (hq : 0 ≤ q) :
    (p + q) • s = p • s + q • s := (add_smul_subset _ _ _).antisymm <| by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    h_conv : Convex 𝕜 s
    p q : 𝕜
    hp : LE.le 0 p
    hq : LE.le 0 q
    ⊢ HasSubset.Subset (HAdd.hAdd (HSMul.hSMul p s) (HSMul.hSMul q s)) (HSMul.hSMu …
  -/
  rintro _ ⟨_, ⟨v₁, h₁, rfl⟩, _, ⟨v₂, h₂, rfl⟩, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    h_conv : Convex 𝕜 s
    p q : 𝕜
    hp : LE.le 0 p
    hq : LE.le 0 q
    v₁ : E
    h₁ : Membership.mem s v₁
    v₂ : E
    h₂ : Membership.mem s v₂
    ⊢ Membership.mem (HSMul.hSMul (HAdd.hAdd p q) s) ((fun x1 x2 => HAdd.hAdd x1 x …
  -/
  exact h_conv.exists_mem_add_smul_eq h₁ h₂ hp hq
  /-
    🎉 no goals
  -/


theorem Set.OrdConnected.convex_of_chain [OrderedSemiring 𝕜] [OrderedAddCommMonoid E] [Module 𝕜 E]
    [OrderedSMul 𝕜 E] {s : Set E} (hs : s.OrdConnected) (h : IsChain (· ≤ ·) s) : Convex 𝕜 s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : OrderedAddCommMonoid E
    inst✝¹ : Module 𝕜 E
    inst✝ : OrderedSMul 𝕜 E
    s : Set E
    hs : s.OrdConnected
    h : IsChain (fun x1 x2 => LE.le x1 x2) s
    ⊢ Convex 𝕜 s
  -/
  refine convex_iff_segment_subset.mpr fun x hx y hy => ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : OrderedAddCommMonoid E
    inst✝¹ : Module 𝕜 E
    inst✝ : OrderedSMul 𝕜 E
    s : Set E
    hs : s.OrdConnected
    h : IsChain (fun x1 x2 => LE.le x1 x2) s
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    ⊢ HasSubset.Subset (segment 𝕜 x y) s
  -/
  obtain hxy | hyx := h.total hx hy
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : OrderedSemiring 𝕜
      inst✝² : OrderedAddCommMonoid E
      inst✝¹ : Module 𝕜 E
      inst✝ : OrderedSMul 𝕜 E
      s : Set E
      hs : s.OrdConnected
      h : IsChain (fun x1 x2 => LE.le x1 x2) s
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hxy : LE.le x y
      ⊢ HasSubset.Subset (segment 𝕜 x y) s
    -/
  · exact (segment_subset_Icc hxy).trans (hs.out hx hy)
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : OrderedSemiring 𝕜
      inst✝² : OrderedAddCommMonoid E
      inst✝¹ : Module 𝕜 E
      inst✝ : OrderedSMul 𝕜 E
      s : Set E
      hs : s.OrdConnected
      h : IsChain (fun x1 x2 => LE.le x1 x2) s
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hyx : LE.le y x
      ⊢ HasSubset.Subset (segment 𝕜 x y) s
    -/
  · rw [segment_symm]
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : OrderedSemiring 𝕜
      inst✝² : OrderedAddCommMonoid E
      inst✝¹ : Module 𝕜 E
      inst✝ : OrderedSMul 𝕜 E
      s : Set E
      hs : s.OrdConnected
      h : IsChain (fun x1 x2 => LE.le x1 x2) s
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hyx : LE.le y x
      ⊢ HasSubset.Subset (segment 𝕜 y x) s
    -/
    exact (segment_subset_Icc hyx).trans (hs.out hy hx)
    /-
      🎉 no goals
    -/


theorem Set.OrdConnected.convex [OrderedSemiring 𝕜] [LinearOrderedAddCommMonoid E] [Module 𝕜 E]
    [OrderedSMul 𝕜 E] {s : Set E} (hs : s.OrdConnected) : Convex 𝕜 s :=
  hs.convex_of_chain <| isChain_of_trichotomous s


theorem convex_iff_ordConnected [LinearOrderedField 𝕜] {s : Set 𝕜} :
    Convex 𝕜 s ↔ s.OrdConnected := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    ⊢ Iff (Convex 𝕜 s) s.OrdConnected
  -/
  simp_rw [convex_iff_segment_subset, segment_eq_uIcc, ordConnected_iff_uIcc_subset]
  /-
    🎉 no goals
  -/


alias ⟨Convex.ordConnected, _⟩ := convex_iff_ordConnected


protected theorem convex (K : Submodule 𝕜 E) : Convex 𝕜 (↑K : Set E) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    K : Submodule 𝕜 E
    ⊢ Convex 𝕜 ↑K
  -/
  repeat' intro
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    K : Submodule 𝕜 E
    x✝ : E
    a✝⁵ : Membership.mem (↑K) x✝
    y✝ : E
    a✝⁴ : Membership.mem (↑K) y✝
    a✝³ b✝ : 𝕜
    a✝² : LE.le 0 a✝³
    a✝¹ : LE.le 0 b✝
    a✝ : Eq (HAdd.hAdd a✝³ b✝) 1
    ⊢ Membership.mem (↑K) (HAdd.hAdd (HSMul.hSMul a✝³ x✝) (HSMul.hSMul b✝ y✝))
  -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  refine add_mem (smul_mem _ _ ?_) (smul_mem _ _ ?_) <;> assumption
                                                         /-
                                                           🎉 no goals
                                                         -/


protected theorem starConvex (K : Submodule 𝕜 E) : StarConvex 𝕜 (0 : E) K :=
  K.convex K.zero_mem


/-- The standard simplex in the space of functions `ι → 𝕜` is the set of vectors with non-negative
coordinates with total sum `1`. This is the free object in the category of convex spaces. -/
def stdSimplex : Set (ι → 𝕜) :=
  { f | (∀ x, 0 ≤ f x) ∧ ∑ x, f x = 1 }


theorem stdSimplex_eq_inter : stdSimplex 𝕜 ι = (⋂ x, { f | 0 ≤ f x }) ∩ { f | ∑ x, f x = 1 } := by
  /-
    𝕜 : Type u_1
    ι : Type u_5
    inst✝¹ : OrderedSemiring 𝕜
    inst✝ : Fintype ι
    ⊢ Eq (stdSimplex 𝕜 ι) (Inter.inter (Set.iInter fun x => setOf fun f => LE.le 0 …
  -/
  ext f
  /-
    case h
    𝕜 : Type u_1
    ι : Type u_5
    inst✝¹ : OrderedSemiring 𝕜
    inst✝ : Fintype ι
    f : ι → 𝕜
    ⊢ Iff (Membership.mem (stdSimplex 𝕜 ι) f) (Membership.mem (Inter.inter (Set.iI …
  -/
  simp only [stdSimplex, Set.mem_inter_iff, Set.mem_iInter, Set.mem_setOf_eq]
  /-
    🎉 no goals
  -/


theorem convex_stdSimplex : Convex 𝕜 (stdSimplex 𝕜 ι) := by
  /-
    𝕜 : Type u_1
    ι : Type u_5
    inst✝¹ : OrderedSemiring 𝕜
    inst✝ : Fintype ι
    ⊢ Convex 𝕜 (stdSimplex 𝕜 ι)
  -/
  refine fun f hf g hg a b ha hb hab => ⟨fun x => ?_, ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      ι : Type u_5
      inst✝¹ : OrderedSemiring 𝕜
      inst✝ : Fintype ι
      f : ι → 𝕜
      hf : Membership.mem (stdSimplex 𝕜 ι) f
      g : ι → 𝕜
      hg : Membership.mem (stdSimplex 𝕜 ι) g
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      x : ι
      ⊢ LE.le 0 (HAdd.hAdd (HSMul.hSMul a f) (HSMul.hSMul b g) x)
    -/
  · apply_rules [add_nonneg, mul_nonneg, hf.1, hg.1]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      ι : Type u_5
      inst✝¹ : OrderedSemiring 𝕜
      inst✝ : Fintype ι
      f : ι → 𝕜
      hf : Membership.mem (stdSimplex 𝕜 ι) f
      g : ι → 𝕜
      hg : Membership.mem (stdSimplex 𝕜 ι) g
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Eq (Finset.univ.sum fun x => HAdd.hAdd (HSMul.hSMul a f) (HSMul.hSMul b g) x …
    -/
  · erw [Finset.sum_add_distrib]
    /-
      case refine_2
      𝕜 : Type u_1
      ι : Type u_5
      inst✝¹ : OrderedSemiring 𝕜
      inst✝ : Fintype ι
      f : ι → 𝕜
      hf : Membership.mem (stdSimplex 𝕜 ι) f
      g : ι → 𝕜
      hg : Membership.mem (stdSimplex 𝕜 ι) g
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Eq (HAdd.hAdd (Finset.univ.sum fun x => HSMul.hSMul a f x) (Finset.univ.sum  …
    -/
    simp only [Pi.smul_apply] -- Porting note: `erw` failed to rewrite with `← Finset.smul_sum`
    rw [← Finset.smul_sum, ← Finset.smul_sum, hf.2, hg.2, smul_eq_mul,
      smul_eq_mul, mul_one, mul_one]
    /-
      case refine_2
      𝕜 : Type u_1
      ι : Type u_5
      inst✝¹ : OrderedSemiring 𝕜
      inst✝ : Fintype ι
      f : ι → 𝕜
      hf : Membership.mem (stdSimplex 𝕜 ι) f
      g : ι → 𝕜
      hg : Membership.mem (stdSimplex 𝕜 ι) g
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Eq (HAdd.hAdd a b) 1
    -/
    exact hab
    /-
      🎉 no goals
    -/


@[nontriviality] lemma stdSimplex_of_subsingleton [Subsingleton 𝕜] : stdSimplex 𝕜 ι = univ :=
  eq_univ_of_forall fun _ ↦ ⟨fun _ ↦ (Subsingleton.elim _ _).le, Subsingleton.elim _ _⟩


/-- The standard simplex in the zero-dimensional space is empty. -/
lemma stdSimplex_of_isEmpty_index [IsEmpty ι] [Nontrivial 𝕜] : stdSimplex 𝕜 ι = ∅ :=
                                   /-
                                     𝕜 : Type u_1
                                     ι : Type u_5
                                     inst✝³ : OrderedSemiring 𝕜
                                     inst✝² : Fintype ι
                                     inst✝¹ : IsEmpty ι
                                     inst✝ : Nontrivial 𝕜
                                     ⊢ ∀ (x : ι → 𝕜), Not (Membership.mem (stdSimplex 𝕜 ι) x)
                                   -/
  eq_empty_of_forall_not_mem <| by rintro f ⟨-, hf⟩; simp at hf
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma stdSimplex_unique [Nonempty ι] [Subsingleton ι] : stdSimplex 𝕜 ι = {fun _ ↦ 1} := by
  /-
    𝕜 : Type u_1
    ι : Type u_5
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : Fintype ι
    inst✝¹ : Nonempty ι
    inst✝ : Subsingleton ι
    ⊢ Eq (stdSimplex 𝕜 ι) (Singleton.singleton fun x => 1)
  -/
  cases nonempty_unique ι
  /-
    case intro
    𝕜 : Type u_1
    ι : Type u_5
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : Fintype ι
    inst✝¹ : Nonempty ι
    inst✝ : Subsingleton ι
    val✝ : Unique ι
    ⊢ Eq (stdSimplex 𝕜 ι) (Singleton.singleton fun x => 1)
  -/
  refine eq_singleton_iff_unique_mem.2 ⟨⟨fun _ ↦ zero_le_one, Fintype.sum_unique _⟩, ?_⟩
  /-
    case intro
    𝕜 : Type u_1
    ι : Type u_5
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : Fintype ι
    inst✝¹ : Nonempty ι
    inst✝ : Subsingleton ι
    val✝ : Unique ι
    ⊢ ∀ (x : ι → 𝕜), Membership.mem (stdSimplex 𝕜 ι) x → Eq x fun x => 1
  -/
  rintro f ⟨-, hf⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    ι : Type u_5
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : Fintype ι
    inst✝¹ : Nonempty ι
    inst✝ : Subsingleton ι
    val✝ : Unique ι
    f : ι → 𝕜
    hf : Eq (Finset.univ.sum fun x => f x) 1
    ⊢ Eq f fun x => 1
  -/
  rw [Fintype.sum_unique] at hf
  /-
    case intro.intro
    𝕜 : Type u_1
    ι : Type u_5
    inst✝³ : OrderedSemiring 𝕜
    inst✝² : Fintype ι
    inst✝¹ : Nonempty ι
    inst✝ : Subsingleton ι
    val✝ : Unique ι
    f : ι → 𝕜
    hf : Eq (f Inhabited.default) 1
    ⊢ Eq f fun x => 1
  -/
  exact funext (Unique.forall_iff.2 hf)
  /-
    🎉 no goals
  -/


theorem single_mem_stdSimplex (i : ι) : Pi.single i 1 ∈ stdSimplex 𝕜 ι :=
                                                       /-
                                                         𝕜 : Type u_1
                                                         ι : Type u_5
                                                         inst✝² : OrderedSemiring 𝕜
                                                         inst✝¹ : Fintype ι
                                                         inst✝ : DecidableEq ι
                                                         i : ι
                                                         ⊢ Eq (Finset.univ.sum fun x => Pi.single i 1 x) 1
                                                       -/
  ⟨le_update_iff.2 ⟨zero_le_one, fun _ _ ↦ le_rfl⟩, by simp⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem ite_eq_mem_stdSimplex (i : ι) : (if i = · then (1 : 𝕜) else 0) ∈ stdSimplex 𝕜 ι := by
  /-
    𝕜 : Type u_1
    ι : Type u_5
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    i : ι
    ⊢ Membership.mem (stdSimplex 𝕜 ι) fun x => ite (Eq i x) 1 0
  -/
  simpa only [@eq_comm _ i, ← Pi.single_apply] using single_mem_stdSimplex 𝕜 i
  /-
    🎉 no goals
  -/


/-- The edges are contained in the simplex. -/
lemma segment_single_subset_stdSimplex (i j : ι) :
    ([Pi.single i 1 -[𝕜] Pi.single j 1] : Set (ι → 𝕜)) ⊆ stdSimplex 𝕜 ι :=
  (convex_stdSimplex 𝕜 ι).segment_subset (single_mem_stdSimplex _ _) (single_mem_stdSimplex _ _)


lemma stdSimplex_fin_two :
    stdSimplex 𝕜 (Fin 2) = ([Pi.single 0 1 -[𝕜] Pi.single 1 1] : Set (Fin 2 → 𝕜)) := by
  /-
    𝕜 : Type u_1
    inst✝ : OrderedSemiring 𝕜
    ⊢ Eq (stdSimplex 𝕜 (Fin 2)) (segment 𝕜 (Pi.single 0 1) (Pi.single 1 1))
  -/
  refine Subset.antisymm ?_ (segment_single_subset_stdSimplex 𝕜 (0 : Fin 2) 1)
  /-
    𝕜 : Type u_1
    inst✝ : OrderedSemiring 𝕜
    ⊢ HasSubset.Subset (stdSimplex 𝕜 (Fin 2)) (segment 𝕜 (Pi.single 0 1) (Pi.singl …
  -/
  rintro f ⟨hf₀, hf₁⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝ : OrderedSemiring 𝕜
    f : Fin 2 → 𝕜
    hf₀ : ∀ (x : Fin 2), LE.le 0 (f x)
    hf₁ : Eq (Finset.univ.sum fun x => f x) 1
    ⊢ Membership.mem (segment 𝕜 (Pi.single 0 1) (Pi.single 1 1)) f
  -/
  rw [Fin.sum_univ_two] at hf₁
  /-
    case intro
    𝕜 : Type u_1
    inst✝ : OrderedSemiring 𝕜
    f : Fin 2 → 𝕜
    hf₀ : ∀ (x : Fin 2), LE.le 0 (f x)
    hf₁ : Eq (HAdd.hAdd (f 0) (f 1)) 1
    ⊢ Membership.mem (segment 𝕜 (Pi.single 0 1) (Pi.single 1 1)) f
  -/
  refine ⟨f 0, f 1, hf₀ 0, hf₀ 1, hf₁, funext <| Fin.forall_fin_two.2 ?_⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝ : OrderedSemiring 𝕜
    f : Fin 2 → 𝕜
    hf₀ : ∀ (x : Fin 2), LE.le 0 (f x)
    hf₁ : Eq (HAdd.hAdd (f 0) (f 1)) 1
    ⊢ And (Eq (HAdd.hAdd (HSMul.hSMul (f 0) (Pi.single 0 1)) (HSMul.hSMul (f 1) (P …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The standard one-dimensional simplex in `Fin 2 → 𝕜` is equivalent to the unit interval. -/
@[simps (config := .asFn)]
def stdSimplexEquivIcc : stdSimplex 𝕜 (Fin 2) ≃ Icc (0 : 𝕜) 1 where
  toFun f := ⟨f.1 0, f.2.1 _, f.2.2 ▸
    Finset.single_le_sum (fun i _ ↦ f.2.1 i) (Finset.mem_univ _)⟩
  invFun x := ⟨![x, 1 - x], Fin.forall_fin_two.2 ⟨x.2.1, sub_nonneg.2 x.2.2⟩,
    calc
      ∑ i : Fin 2, ![(x : 𝕜), 1 - x] i = x + (1 - x) := Fin.sum_univ_two _
      _ = 1 := add_sub_cancel _ _⟩
  left_inv f := Subtype.eq <| funext <| Fin.forall_fin_two.2 <| .intro rfl <|
      calc
                                                      /-
                                                        𝕜 : Type u_1
                                                        E : Type u_2
                                                        F : Type u_3
                                                        β : Type u_4
                                                        inst✝ : OrderedRing 𝕜
                                                        f : ↑(stdSimplex 𝕜 (Fin 2))
                                                        ⊢ Eq (HSub.hSub 1 (↑f 0)) (HSub.hSub (HAdd.hAdd (↑f 0) (↑f 1)) (↑f 0))
                                                      -/
        (1 : 𝕜) - f.1 0 = f.1 0 + f.1 1 - f.1 0 := by rw [← Fin.sum_univ_two f.1, f.2.2]
                                                      /-
                                                        🎉 no goals
                                                      -/
        _ = f.1 1 := add_sub_cancel_left _ _
  right_inv _ := Subtype.eq rfl


