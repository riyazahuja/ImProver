/-- Convexity of functions -/
def ConvexOn : Prop :=
  Convex 𝕜 s ∧ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → ∀ ⦃a b : 𝕜⦄, 0 ≤ a → 0 ≤ b → a + b = 1 →
    f (a • x + b • y) ≤ a • f x + b • f y


/-- Concavity of functions -/
def ConcaveOn : Prop :=
  Convex 𝕜 s ∧ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → ∀ ⦃a b : 𝕜⦄, 0 ≤ a → 0 ≤ b → a + b = 1 →
    a • f x + b • f y ≤ f (a • x + b • y)


/-- Strict convexity of functions -/
def StrictConvexOn : Prop :=
  Convex 𝕜 s ∧ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → x ≠ y → ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b → a + b = 1 →
    f (a • x + b • y) < a • f x + b • f y


/-- Strict concavity of functions -/
def StrictConcaveOn : Prop :=
  Convex 𝕜 s ∧ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → x ≠ y → ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b → a + b = 1 →
    a • f x + b • f y < f (a • x + b • y)


theorem ConvexOn.dual (hf : ConvexOn 𝕜 s f) : ConcaveOn 𝕜 s (toDual ∘ f) := hf


theorem ConcaveOn.dual (hf : ConcaveOn 𝕜 s f) : ConvexOn 𝕜 s (toDual ∘ f) := hf


theorem StrictConvexOn.dual (hf : StrictConvexOn 𝕜 s f) : StrictConcaveOn 𝕜 s (toDual ∘ f) := hf


theorem StrictConcaveOn.dual (hf : StrictConcaveOn 𝕜 s f) : StrictConvexOn 𝕜 s (toDual ∘ f) := hf


theorem convexOn_id {s : Set β} (hs : Convex 𝕜 s) : ConvexOn 𝕜 s _root_.id :=
  ⟨hs, by
    /-
      𝕜 : Type u_1
      β : Type u_5
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : OrderedAddCommMonoid β
      inst✝ : SMul 𝕜 β
      s : Set β
      hs : Convex 𝕜 s
      ⊢ ∀ ⦃x : β⦄, Membership.mem s x → ∀ ⦃y : β⦄, Membership.mem s y → ∀ ⦃a b : 𝕜⦄, …
    -/
    intros
    /-
      𝕜 : Type u_1
      β : Type u_5
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : OrderedAddCommMonoid β
      inst✝ : SMul 𝕜 β
      s : Set β
      hs : Convex 𝕜 s
      x✝ : β
      a✝⁵ : Membership.mem s x✝
      y✝ : β
      a✝⁴ : Membership.mem s y✝
      a✝³ b✝ : 𝕜
      a✝² : LE.le 0 a✝³
      a✝¹ : LE.le 0 b✝
      a✝ : Eq (HAdd.hAdd a✝³ b✝) 1
      ⊢ LE.le (id (HAdd.hAdd (HSMul.hSMul a✝³ x✝) (HSMul.hSMul b✝ y✝))) (HAdd.hAdd ( …
    -/
    rfl⟩
    /-
      🎉 no goals
    -/


theorem concaveOn_id {s : Set β} (hs : Convex 𝕜 s) : ConcaveOn 𝕜 s _root_.id :=
  ⟨hs, by
    /-
      𝕜 : Type u_1
      β : Type u_5
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : OrderedAddCommMonoid β
      inst✝ : SMul 𝕜 β
      s : Set β
      hs : Convex 𝕜 s
      ⊢ ∀ ⦃x : β⦄, Membership.mem s x → ∀ ⦃y : β⦄, Membership.mem s y → ∀ ⦃a b : 𝕜⦄, …
    -/
    intros
    /-
      𝕜 : Type u_1
      β : Type u_5
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : OrderedAddCommMonoid β
      inst✝ : SMul 𝕜 β
      s : Set β
      hs : Convex 𝕜 s
      x✝ : β
      a✝⁵ : Membership.mem s x✝
      y✝ : β
      a✝⁴ : Membership.mem s y✝
      a✝³ b✝ : 𝕜
      a✝² : LE.le 0 a✝³
      a✝¹ : LE.le 0 b✝
      a✝ : Eq (HAdd.hAdd a✝³ b✝) 1
      ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a✝³ (id x✝)) (HSMul.hSMul b✝ (id y✝))) (id (HA …
    -/
    rfl⟩
    /-
      🎉 no goals
    -/


theorem ConvexOn.congr (hf : ConvexOn 𝕜 s f) (hfg : EqOn f g s) : ConvexOn 𝕜 s g :=
  ⟨hf.1, fun x hx y hy a b ha hb hab => by
    /-
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : SMul 𝕜 E
      inst✝ : SMul 𝕜 β
      s : Set E
      f g : E → β
      hf : ConvexOn 𝕜 s f
      hfg : Set.EqOn f g s
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ LE.le (g (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul. …
    -/
    simpa only [← hfg hx, ← hfg hy, ← hfg (hf.1 hx hy ha hb hab)] using hf.2 hx hy ha hb hab⟩
    /-
      🎉 no goals
    -/


theorem ConcaveOn.congr (hf : ConcaveOn 𝕜 s f) (hfg : EqOn f g s) : ConcaveOn 𝕜 s g :=
  ⟨hf.1, fun x hx y hy a b ha hb hab => by
    /-
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : SMul 𝕜 E
      inst✝ : SMul 𝕜 β
      s : Set E
      f g : E → β
      hf : ConcaveOn 𝕜 s f
      hfg : Set.EqOn f g s
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a (g x)) (HSMul.hSMul b (g y))) (g (HAdd.hAdd  …
    -/
    simpa only [← hfg hx, ← hfg hy, ← hfg (hf.1 hx hy ha hb hab)] using hf.2 hx hy ha hb hab⟩
    /-
      🎉 no goals
    -/


theorem StrictConvexOn.congr (hf : StrictConvexOn 𝕜 s f) (hfg : EqOn f g s) :
    StrictConvexOn 𝕜 s g :=
  ⟨hf.1, fun x hx y hy hxy a b ha hb hab => by
    simpa only [← hfg hx, ← hfg hy, ← hfg (hf.1 hx hy ha.le hb.le hab)] using
      hf.2 hx hy hxy ha hb hab⟩


theorem StrictConcaveOn.congr (hf : StrictConcaveOn 𝕜 s f) (hfg : EqOn f g s) :
    StrictConcaveOn 𝕜 s g :=
  ⟨hf.1, fun x hx y hy hxy a b ha hb hab => by
    simpa only [← hfg hx, ← hfg hy, ← hfg (hf.1 hx hy ha.le hb.le hab)] using
      hf.2 hx hy hxy ha hb hab⟩


theorem ConvexOn.subset {t : Set E} (hf : ConvexOn 𝕜 t f) (hst : s ⊆ t) (hs : Convex 𝕜 s) :
    ConvexOn 𝕜 s f :=
  ⟨hs, fun _ hx _ hy => hf.2 (hst hx) (hst hy)⟩


theorem ConcaveOn.subset {t : Set E} (hf : ConcaveOn 𝕜 t f) (hst : s ⊆ t) (hs : Convex 𝕜 s) :
    ConcaveOn 𝕜 s f :=
  ⟨hs, fun _ hx _ hy => hf.2 (hst hx) (hst hy)⟩


theorem StrictConvexOn.subset {t : Set E} (hf : StrictConvexOn 𝕜 t f) (hst : s ⊆ t)
    (hs : Convex 𝕜 s) : StrictConvexOn 𝕜 s f :=
  ⟨hs, fun _ hx _ hy => hf.2 (hst hx) (hst hy)⟩


theorem StrictConcaveOn.subset {t : Set E} (hf : StrictConcaveOn 𝕜 t f) (hst : s ⊆ t)
    (hs : Convex 𝕜 s) : StrictConcaveOn 𝕜 s f :=
  ⟨hs, fun _ hx _ hy => hf.2 (hst hx) (hst hy)⟩


theorem ConvexOn.comp (hg : ConvexOn 𝕜 (f '' s) g) (hf : ConvexOn 𝕜 s f)
    (hg' : MonotoneOn g (f '' s)) : ConvexOn 𝕜 s (g ∘ f) :=
  ⟨hf.1, fun _ hx _ hy _ _ ha hb hab =>
    (hg' (mem_image_of_mem f <| hf.1 hx hy ha hb hab)
            (hg.1 (mem_image_of_mem f hx) (mem_image_of_mem f hy) ha hb hab) <|
          hf.2 hx hy ha hb hab).trans <|
      hg.2 (mem_image_of_mem f hx) (mem_image_of_mem f hy) ha hb hab⟩


theorem ConcaveOn.comp (hg : ConcaveOn 𝕜 (f '' s) g) (hf : ConcaveOn 𝕜 s f)
    (hg' : MonotoneOn g (f '' s)) : ConcaveOn 𝕜 s (g ∘ f) :=
  ⟨hf.1, fun _ hx _ hy _ _ ha hb hab =>
    (hg.2 (mem_image_of_mem f hx) (mem_image_of_mem f hy) ha hb hab).trans <|
      hg' (hg.1 (mem_image_of_mem f hx) (mem_image_of_mem f hy) ha hb hab)
          (mem_image_of_mem f <| hf.1 hx hy ha hb hab) <|
        hf.2 hx hy ha hb hab⟩


theorem ConvexOn.comp_concaveOn (hg : ConvexOn 𝕜 (f '' s) g) (hf : ConcaveOn 𝕜 s f)
    (hg' : AntitoneOn g (f '' s)) : ConvexOn 𝕜 s (g ∘ f) :=
  hg.dual.comp hf hg'


theorem ConcaveOn.comp_convexOn (hg : ConcaveOn 𝕜 (f '' s) g) (hf : ConvexOn 𝕜 s f)
    (hg' : AntitoneOn g (f '' s)) : ConcaveOn 𝕜 s (g ∘ f) :=
  hg.dual.comp hf hg'


theorem StrictConvexOn.comp (hg : StrictConvexOn 𝕜 (f '' s) g) (hf : StrictConvexOn 𝕜 s f)
    (hg' : StrictMonoOn g (f '' s)) (hf' : s.InjOn f) : StrictConvexOn 𝕜 s (g ∘ f) :=
  ⟨hf.1, fun _ hx _ hy hxy _ _ ha hb hab =>
    (hg' (mem_image_of_mem f <| hf.1 hx hy ha.le hb.le hab)
            (hg.1 (mem_image_of_mem f hx) (mem_image_of_mem f hy) ha.le hb.le hab) <|
          hf.2 hx hy hxy ha hb hab).trans <|
      hg.2 (mem_image_of_mem f hx) (mem_image_of_mem f hy) (mt (hf' hx hy) hxy) ha hb hab⟩


theorem StrictConcaveOn.comp (hg : StrictConcaveOn 𝕜 (f '' s) g) (hf : StrictConcaveOn 𝕜 s f)
    (hg' : StrictMonoOn g (f '' s)) (hf' : s.InjOn f) : StrictConcaveOn 𝕜 s (g ∘ f) :=
  ⟨hf.1, fun _ hx _ hy hxy _ _ ha hb hab =>
    (hg.2 (mem_image_of_mem f hx) (mem_image_of_mem f hy) (mt (hf' hx hy) hxy) ha hb hab).trans <|
      hg' (hg.1 (mem_image_of_mem f hx) (mem_image_of_mem f hy) ha.le hb.le hab)
          (mem_image_of_mem f <| hf.1 hx hy ha.le hb.le hab) <|
        hf.2 hx hy hxy ha hb hab⟩


theorem StrictConvexOn.comp_strictConcaveOn (hg : StrictConvexOn 𝕜 (f '' s) g)
    (hf : StrictConcaveOn 𝕜 s f) (hg' : StrictAntiOn g (f '' s)) (hf' : s.InjOn f) :
    StrictConvexOn 𝕜 s (g ∘ f) :=
  hg.dual.comp hf hg' hf'


theorem StrictConcaveOn.comp_strictConvexOn (hg : StrictConcaveOn 𝕜 (f '' s) g)
    (hf : StrictConvexOn 𝕜 s f) (hg' : StrictAntiOn g (f '' s)) (hf' : s.InjOn f) :
    StrictConcaveOn 𝕜 s (g ∘ f) :=
  hg.dual.comp hf hg' hf'


theorem ConvexOn.add (hf : ConvexOn 𝕜 s f) (hg : ConvexOn 𝕜 s g) : ConvexOn 𝕜 s (f + g) :=
  ⟨hf.1, fun x hx y hy a b ha hb hab =>
    calc
      f (a • x + b • y) + g (a • x + b • y) ≤ a • f x + b • f y + (a • g x + b • g y) :=
        add_le_add (hf.2 hx hy ha hb hab) (hg.2 hx hy ha hb hab)
                                                  /-
                                                    𝕜 : Type u_1
                                                    E : Type u_2
                                                    β : Type u_5
                                                    inst✝⁴ : OrderedSemiring 𝕜
                                                    inst✝³ : AddCommMonoid E
                                                    inst✝² : OrderedAddCommMonoid β
                                                    inst✝¹ : SMul 𝕜 E
                                                    inst✝ : DistribMulAction 𝕜 β
                                                    s : Set E
                                                    f g : E → β
                                                    hf : ConvexOn 𝕜 s f
                                                    hg : ConvexOn 𝕜 s g
                                                    x : E
                                                    hx : Membership.mem s x
                                                    y : E
                                                    hy : Membership.mem s y
                                                    a b : 𝕜
                                                    ha : LE.le 0 a
                                                    hb : LE.le 0 b
                                                    hab : Eq (HAdd.hAdd a b) 1
                                                    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y))) (HAdd. …
                                                  -/
      _ = a • (f x + g x) + b • (f y + g y) := by rw [smul_add, smul_add, add_add_add_comm]
                                                  /-
                                                    🎉 no goals
                                                  -/
      ⟩


theorem ConcaveOn.add (hf : ConcaveOn 𝕜 s f) (hg : ConcaveOn 𝕜 s g) : ConcaveOn 𝕜 s (f + g) :=
  hf.dual.add hg


theorem convexOn_const (c : β) (hs : Convex 𝕜 s) : ConvexOn 𝕜 s fun _ : E => c :=
  ⟨hs, fun _ _ _ _ _ _ _ _ hab => (Convex.combo_self hab c).ge⟩


theorem concaveOn_const (c : β) (hs : Convex 𝕜 s) : ConcaveOn 𝕜 s fun _ => c :=
  convexOn_const (β := βᵒᵈ) _ hs


theorem ConvexOn.add_const (hf : ConvexOn 𝕜 s f) (b : β) :
    ConvexOn 𝕜 s (f + fun _ => b) :=
  hf.add (convexOn_const _ hf.1)


theorem ConcaveOn.add_const (hf : ConcaveOn 𝕜 s f) (b : β) :
    ConcaveOn 𝕜 s (f + fun _ => b) :=
  hf.add (concaveOn_const _ hf.1)


theorem convexOn_of_convex_epigraph (h : Convex 𝕜 { p : E × β | p.1 ∈ s ∧ f p.1 ≤ p.2 }) :
    ConvexOn 𝕜 s f :=
  ⟨fun x hx y hy a b ha hb hab => (@h (x, f x) ⟨hx, le_rfl⟩ (y, f y) ⟨hy, le_rfl⟩ a b ha hb hab).1,
    fun x hx y hy a b ha hb hab => (@h (x, f x) ⟨hx, le_rfl⟩ (y, f y) ⟨hy, le_rfl⟩ a b ha hb hab).2⟩


theorem concaveOn_of_convex_hypograph (h : Convex 𝕜 { p : E × β | p.1 ∈ s ∧ p.2 ≤ f p.1 }) :
    ConcaveOn 𝕜 s f :=
  convexOn_of_convex_epigraph (β := βᵒᵈ) h


theorem ConvexOn.convex_le (hf : ConvexOn 𝕜 s f) (r : β) : Convex 𝕜 ({ x ∈ s | f x ≤ r }) :=
  fun x hx y hy a b ha hb hab =>
  ⟨hf.1 hx.1 hy.1 ha hb hab,
    calc
      f (a • x + b • y) ≤ a • f x + b • f y := hf.2 hx.1 hy.1 ha hb hab
      _ ≤ a • r + b • r := by
        /-
          𝕜 : Type u_1
          E : Type u_2
          β : Type u_5
          inst✝⁵ : OrderedSemiring 𝕜
          inst✝⁴ : AddCommMonoid E
          inst✝³ : OrderedAddCommMonoid β
          inst✝² : SMul 𝕜 E
          inst✝¹ : Module 𝕜 β
          inst✝ : OrderedSMul 𝕜 β
          s : Set E
          f : E → β
          hf : ConvexOn 𝕜 s f
          r : β
          x : E
          hx : Membership.mem (setOf fun x => And (Membership.mem s x) (LE.le (f x) r)) x
          y : E
          hy : Membership.mem (setOf fun x => And (Membership.mem s x) (LE.le (f x) r)) y
          a b : 𝕜
          ha : LE.le 0 a
          hb : LE.le 0 b
          hab : Eq (HAdd.hAdd a b) 1
          ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y))) (HAdd.hAdd (HS …
        -/
        gcongr
          /-
            case h₁.hb
            𝕜 : Type u_1
            E : Type u_2
            β : Type u_5
            inst✝⁵ : OrderedSemiring 𝕜
            inst✝⁴ : AddCommMonoid E
            inst✝³ : OrderedAddCommMonoid β
            inst✝² : SMul 𝕜 E
            inst✝¹ : Module 𝕜 β
            inst✝ : OrderedSMul 𝕜 β
            s : Set E
            f : E → β
            hf : ConvexOn 𝕜 s f
            r : β
            x : E
            hx : Membership.mem (setOf fun x => And (Membership.mem s x) (LE.le (f x) r)) x
            y : E
            hy : Membership.mem (setOf fun x => And (Membership.mem s x) (LE.le (f x) r)) y
            a b : 𝕜
            ha : LE.le 0 a
            hb : LE.le 0 b
            hab : Eq (HAdd.hAdd a b) 1
            ⊢ LE.le (f x) r
          -/
        · exact hx.2
          /-
            🎉 no goals
          -/
          /-
            case h₂.hb
            𝕜 : Type u_1
            E : Type u_2
            β : Type u_5
            inst✝⁵ : OrderedSemiring 𝕜
            inst✝⁴ : AddCommMonoid E
            inst✝³ : OrderedAddCommMonoid β
            inst✝² : SMul 𝕜 E
            inst✝¹ : Module 𝕜 β
            inst✝ : OrderedSMul 𝕜 β
            s : Set E
            f : E → β
            hf : ConvexOn 𝕜 s f
            r : β
            x : E
            hx : Membership.mem (setOf fun x => And (Membership.mem s x) (LE.le (f x) r)) x
            y : E
            hy : Membership.mem (setOf fun x => And (Membership.mem s x) (LE.le (f x) r)) y
            a b : 𝕜
            ha : LE.le 0 a
            hb : LE.le 0 b
            hab : Eq (HAdd.hAdd a b) 1
            ⊢ LE.le (f y) r
          -/
        · exact hy.2
          /-
            🎉 no goals
          -/
      _ = r := Convex.combo_self hab r
      ⟩


theorem ConcaveOn.convex_ge (hf : ConcaveOn 𝕜 s f) (r : β) : Convex 𝕜 ({ x ∈ s | r ≤ f x }) :=
  hf.dual.convex_le r


theorem ConvexOn.convex_epigraph (hf : ConvexOn 𝕜 s f) :
    Convex 𝕜 { p : E × β | p.1 ∈ s ∧ f p.1 ≤ p.2 } := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : OrderedAddCommMonoid β
    inst✝² : SMul 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    ⊢ Convex 𝕜 (setOf fun p => And (Membership.mem s p.1) (LE.le (f p.1) p.2))
  -/
  rintro ⟨x, r⟩ ⟨hx, hr⟩ ⟨y, t⟩ ⟨hy, ht⟩ a b ha hb hab
  /-
    case mk.intro.mk.intro
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : OrderedAddCommMonoid β
    inst✝² : SMul 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    x : E
    r : β
    hx : Membership.mem s { fst := x, snd := r }.1
    hr : LE.le (f { fst := x, snd := r }.1) { fst := x, snd := r }.2
    y : E
    t : β
    hy : Membership.mem s { fst := y, snd := t }.1
    ht : LE.le (f { fst := y, snd := t }.1) { fst := y, snd := t }.2
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (setOf fun p => And (Membership.mem s p.1) (LE.le (f p.1) p.2 …
  -/
  refine ⟨hf.1 hx hy ha hb hab, ?_⟩
  calc
    f (a • x + b • y) ≤ a • f x + b • f y := hf.2 hx hy ha hb hab
    _ ≤ a • r + b • t := by gcongr


theorem ConcaveOn.convex_hypograph (hf : ConcaveOn 𝕜 s f) :
    Convex 𝕜 { p : E × β | p.1 ∈ s ∧ p.2 ≤ f p.1 } :=
  hf.dual.convex_epigraph


theorem convexOn_iff_convex_epigraph :
    ConvexOn 𝕜 s f ↔ Convex 𝕜 { p : E × β | p.1 ∈ s ∧ f p.1 ≤ p.2 } :=
  ⟨ConvexOn.convex_epigraph, convexOn_of_convex_epigraph⟩


theorem concaveOn_iff_convex_hypograph :
    ConcaveOn 𝕜 s f ↔ Convex 𝕜 { p : E × β | p.1 ∈ s ∧ p.2 ≤ f p.1 } :=
  convexOn_iff_convex_epigraph (β := βᵒᵈ)


/-- Right translation preserves convexity. -/
theorem ConvexOn.translate_right (hf : ConvexOn 𝕜 s f) (c : E) :
    ConvexOn 𝕜 ((fun z => c + z) ⁻¹' s) (f ∘ fun z => c + z) :=
  ⟨hf.1.translate_preimage_right _, fun x hx y hy a b ha hb hab =>
    calc
      f (c + (a • x + b • y)) = f (a • (c + x) + b • (c + y)) := by
        /-
          𝕜 : Type u_1
          E : Type u_2
          β : Type u_5
          inst✝⁴ : OrderedSemiring 𝕜
          inst✝³ : AddCommMonoid E
          inst✝² : OrderedAddCommMonoid β
          inst✝¹ : Module 𝕜 E
          inst✝ : SMul 𝕜 β
          s : Set E
          f : E → β
          hf : ConvexOn 𝕜 s f
          c x : E
          hx : Membership.mem (Set.preimage (fun z => HAdd.hAdd c z) s) x
          y : E
          hy : Membership.mem (Set.preimage (fun z => HAdd.hAdd c z) s) y
          a b : 𝕜
          ha : LE.le 0 a
          hb : LE.le 0 b
          hab : Eq (HAdd.hAdd a b) 1
          ⊢ Eq (f (HAdd.hAdd c (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)))) (f (HAd …
        -/
        rw [smul_add, smul_add, add_add_add_comm, Convex.combo_self hab]
        /-
          🎉 no goals
        -/
      _ ≤ a • f (c + x) + b • f (c + y) := hf.2 hx hy ha hb hab
      ⟩


/-- Right translation preserves concavity. -/
theorem ConcaveOn.translate_right (hf : ConcaveOn 𝕜 s f) (c : E) :
    ConcaveOn 𝕜 ((fun z => c + z) ⁻¹' s) (f ∘ fun z => c + z) :=
  hf.dual.translate_right _


/-- Left translation preserves convexity. -/
theorem ConvexOn.translate_left (hf : ConvexOn 𝕜 s f) (c : E) :
    ConvexOn 𝕜 ((fun z => c + z) ⁻¹' s) (f ∘ fun z => z + c) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : OrderedAddCommMonoid β
    inst✝¹ : Module 𝕜 E
    inst✝ : SMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    c : E
    ⊢ ConvexOn 𝕜 (Set.preimage (fun z => HAdd.hAdd c z) s) (Function.comp f fun z  …
  -/
  simpa only [add_comm c] using hf.translate_right c
  /-
    🎉 no goals
  -/


/-- Left translation preserves concavity. -/
theorem ConcaveOn.translate_left (hf : ConcaveOn 𝕜 s f) (c : E) :
    ConcaveOn 𝕜 ((fun z => c + z) ⁻¹' s) (f ∘ fun z => z + c) :=
  hf.dual.translate_left _


theorem convexOn_iff_forall_pos {s : Set E} {f : E → β} :
    ConvexOn 𝕜 s f ↔ Convex 𝕜 s ∧ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b →
      a + b = 1 → f (a • x + b • y) ≤ a • f x + b • f y := by
  refine and_congr_right'
    ⟨fun h x hx y hy a b ha hb hab => h hx hy ha.le hb.le hab, fun h x hx y hy a b ha hb hab => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : OrderedAddCommMonoid β
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 β
    s : Set E
    f : E → β
    h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul. …
  -/
  obtain rfl | ha' := ha.eq_or_lt
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      b : 𝕜
      hb : LE.le 0 b
      ha : LE.le 0 0
      hab : Eq (HAdd.hAdd 0 b) 1
      ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul 0 x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul. …
    -/
  · rw [zero_add] at hab
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      b : 𝕜
      hb : LE.le 0 b
      ha : LE.le 0 0
      hab : Eq b 1
      ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul 0 x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul. …
    -/
    subst b
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      ha : LE.le 0 0
      hb : LE.le 0 1
      ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul 0 x) (HSMul.hSMul 1 y))) (HAdd.hAdd (HSMul. …
    -/
    simp_rw [zero_smul, zero_add, one_smul, le_rfl]
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : OrderedAddCommMonoid β
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 β
    s : Set E
    f : E → β
    h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ha' : LT.lt 0 a
    ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul. …
  -/
  obtain rfl | hb' := hb.eq_or_lt
    /-
      case inr.inl
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      a : 𝕜
      ha : LE.le 0 a
      ha' : LT.lt 0 a
      hb : LE.le 0 0
      hab : Eq (HAdd.hAdd a 0) 1
      ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul 0 y))) (HAdd.hAdd (HSMul. …
    -/
  · rw [add_zero] at hab
    /-
      case inr.inl
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      a : 𝕜
      ha : LE.le 0 a
      ha' : LT.lt 0 a
      hb : LE.le 0 0
      hab : Eq a 1
      ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul 0 y))) (HAdd.hAdd (HSMul. …
    -/
    subst a
    /-
      case inr.inl
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hb : LE.le 0 0
      ha : LE.le 0 1
      ha' : LT.lt 0 1
      ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul 1 x) (HSMul.hSMul 0 y))) (HAdd.hAdd (HSMul. …
    -/
    simp_rw [zero_smul, add_zero, one_smul, le_rfl]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : OrderedAddCommMonoid β
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 β
    s : Set E
    f : E → β
    h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ha' : LT.lt 0 a
    hb' : LT.lt 0 b
    ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul. …
  -/
  exact h hx hy ha' hb' hab
  /-
    🎉 no goals
  -/


theorem concaveOn_iff_forall_pos {s : Set E} {f : E → β} :
    ConcaveOn 𝕜 s f ↔
      Convex 𝕜 s ∧ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b → a + b = 1 →
        a • f x + b • f y ≤ f (a • x + b • y) :=
  convexOn_iff_forall_pos (β := βᵒᵈ)


theorem convexOn_iff_pairwise_pos {s : Set E} {f : E → β} :
    ConvexOn 𝕜 s f ↔
      Convex 𝕜 s ∧
        s.Pairwise fun x y =>
          ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b → a + b = 1 → f (a • x + b • y) ≤ a • f x + b • f y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : OrderedAddCommMonoid β
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 β
    s : Set E
    f : E → β
    ⊢ Iff (ConvexOn 𝕜 s f) (And (Convex 𝕜 s) (s.Pairwise fun x y => ∀ ⦃a b : 𝕜⦄, L …
  -/
  rw [convexOn_iff_forall_pos]
  refine
    and_congr_right'
      ⟨fun h x hx y hy _ a b ha hb hab => h hx hy ha hb hab, fun h x hx y hy a b ha hb hab => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : OrderedAddCommMonoid β
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 β
    s : Set E
    f : E → β
    h : s.Pairwise fun x y => ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b → Eq (HAdd.hAdd a …
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul. …
  -/
  obtain rfl | hxy := eq_or_ne x y
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : Module 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      h : s.Pairwise fun x y => ∀ ⦃a b : 𝕜⦄, LT.lt 0 a → LT.lt 0 b → Eq (HAdd.hAdd a …
      x : E
      hx : Membership.mem s x
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hy : Membership.mem s x
      ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b x))) (HAdd.hAdd (HSMul. …
    -/
  · rw [Convex.combo_self hab, Convex.combo_self hab]
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : OrderedAddCommMonoid β
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 β
    s : Set E
    f : E → β
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
    ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul. …
  -/
  exact h hx hy hxy ha hb hab
  /-
    🎉 no goals
  -/


theorem concaveOn_iff_pairwise_pos {s : Set E} {f : E → β} :
    ConcaveOn 𝕜 s f ↔
      Convex 𝕜 s ∧
        s.Pairwise fun x y =>
          ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b → a + b = 1 → a • f x + b • f y ≤ f (a • x + b • y) :=
  convexOn_iff_pairwise_pos (β := βᵒᵈ)


/-- A linear map is convex. -/
theorem LinearMap.convexOn (f : E →ₗ[𝕜] β) {s : Set E} (hs : Convex 𝕜 s) : ConvexOn 𝕜 s f :=
                                   /-
                                     𝕜 : Type u_1
                                     E : Type u_2
                                     β : Type u_5
                                     inst✝⁴ : OrderedSemiring 𝕜
                                     inst✝³ : AddCommMonoid E
                                     inst✝² : OrderedAddCommMonoid β
                                     inst✝¹ : Module 𝕜 E
                                     inst✝ : Module 𝕜 β
                                     f : LinearMap (RingHom.id 𝕜) E β
                                     s : Set E
                                     hs : Convex 𝕜 s
                                     x✝⁸ : E
                                     x✝⁷ : Membership.mem s x✝⁸
                                     x✝⁶ : E
                                     x✝⁵ : Membership.mem s x✝⁶
                                     x✝⁴ x✝³ : 𝕜
                                     x✝² : LE.le 0 x✝⁴
                                     x✝¹ : LE.le 0 x✝³
                                     x✝ : Eq (HAdd.hAdd x✝⁴ x✝³) 1
                                     ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul x✝⁴ x✝⁸) (HSMul.hSMul x✝³ x✝⁶))) (HAdd.hAdd …
                                   -/
  ⟨hs, fun _ _ _ _ _ _ _ _ _ => by rw [f.map_add, f.map_smul, f.map_smul]⟩
                                   /-
                                     🎉 no goals
                                   -/


/-- A linear map is concave. -/
theorem LinearMap.concaveOn (f : E →ₗ[𝕜] β) {s : Set E} (hs : Convex 𝕜 s) : ConcaveOn 𝕜 s f :=
                                   /-
                                     𝕜 : Type u_1
                                     E : Type u_2
                                     β : Type u_5
                                     inst✝⁴ : OrderedSemiring 𝕜
                                     inst✝³ : AddCommMonoid E
                                     inst✝² : OrderedAddCommMonoid β
                                     inst✝¹ : Module 𝕜 E
                                     inst✝ : Module 𝕜 β
                                     f : LinearMap (RingHom.id 𝕜) E β
                                     s : Set E
                                     hs : Convex 𝕜 s
                                     x✝⁸ : E
                                     x✝⁷ : Membership.mem s x✝⁸
                                     x✝⁶ : E
                                     x✝⁵ : Membership.mem s x✝⁶
                                     x✝⁴ x✝³ : 𝕜
                                     x✝² : LE.le 0 x✝⁴
                                     x✝¹ : LE.le 0 x✝³
                                     x✝ : Eq (HAdd.hAdd x✝⁴ x✝³) 1
                                     ⊢ LE.le (HAdd.hAdd (HSMul.hSMul x✝⁴ (f x✝⁸)) (HSMul.hSMul x✝³ (f x✝⁶))) (f (HA …
                                   -/
  ⟨hs, fun _ _ _ _ _ _ _ _ _ => by rw [f.map_add, f.map_smul, f.map_smul]⟩
                                   /-
                                     🎉 no goals
                                   -/


theorem StrictConvexOn.convexOn {s : Set E} {f : E → β} (hf : StrictConvexOn 𝕜 s f) :
    ConvexOn 𝕜 s f :=
  convexOn_iff_pairwise_pos.mpr
    ⟨hf.1, fun _ hx _ hy hxy _ _ ha hb hab => (hf.2 hx hy hxy ha hb hab).le⟩


theorem StrictConcaveOn.concaveOn {s : Set E} {f : E → β} (hf : StrictConcaveOn 𝕜 s f) :
    ConcaveOn 𝕜 s f :=
  hf.dual.convexOn


theorem StrictConvexOn.convex_lt (hf : StrictConvexOn 𝕜 s f) (r : β) :
    Convex 𝕜 ({ x ∈ s | f x < r }) :=
  convex_iff_pairwise_pos.2 fun x hx y hy hxy a b ha hb hab =>
    ⟨hf.1 hx.1 hy.1 ha.le hb.le hab,
      calc
        f (a • x + b • y) < a • f x + b • f y := hf.2 hx.1 hy.1 hxy ha hb hab
        _ ≤ a • r + b • r := by
          /-
            𝕜 : Type u_1
            E : Type u_2
            β : Type u_5
            inst✝⁵ : OrderedSemiring 𝕜
            inst✝⁴ : AddCommMonoid E
            inst✝³ : OrderedAddCommMonoid β
            inst✝² : Module 𝕜 E
            inst✝¹ : Module 𝕜 β
            inst✝ : OrderedSMul 𝕜 β
            s : Set E
            f : E → β
            hf : StrictConvexOn 𝕜 s f
            r : β
            x : E
            hx : Membership.mem (setOf fun x => And (Membership.mem s x) (LT.lt (f x) r)) x
            y : E
            hy : Membership.mem (setOf fun x => And (Membership.mem s x) (LT.lt (f x) r)) y
            hxy : Ne x y
            a b : 𝕜
            ha : LT.lt 0 a
            hb : LT.lt 0 b
            hab : Eq (HAdd.hAdd a b) 1
            ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y))) (HAdd.hAdd (HS …
          -/
          gcongr
            /-
              case h₁.hb
              𝕜 : Type u_1
              E : Type u_2
              β : Type u_5
              inst✝⁵ : OrderedSemiring 𝕜
              inst✝⁴ : AddCommMonoid E
              inst✝³ : OrderedAddCommMonoid β
              inst✝² : Module 𝕜 E
              inst✝¹ : Module 𝕜 β
              inst✝ : OrderedSMul 𝕜 β
              s : Set E
              f : E → β
              hf : StrictConvexOn 𝕜 s f
              r : β
              x : E
              hx : Membership.mem (setOf fun x => And (Membership.mem s x) (LT.lt (f x) r)) x
              y : E
              hy : Membership.mem (setOf fun x => And (Membership.mem s x) (LT.lt (f x) r)) y
              hxy : Ne x y
              a b : 𝕜
              ha : LT.lt 0 a
              hb : LT.lt 0 b
              hab : Eq (HAdd.hAdd a b) 1
              ⊢ LE.le (f x) r
            -/
          · exact hx.2.le
            /-
              🎉 no goals
            -/
            /-
              case h₂.hb
              𝕜 : Type u_1
              E : Type u_2
              β : Type u_5
              inst✝⁵ : OrderedSemiring 𝕜
              inst✝⁴ : AddCommMonoid E
              inst✝³ : OrderedAddCommMonoid β
              inst✝² : Module 𝕜 E
              inst✝¹ : Module 𝕜 β
              inst✝ : OrderedSMul 𝕜 β
              s : Set E
              f : E → β
              hf : StrictConvexOn 𝕜 s f
              r : β
              x : E
              hx : Membership.mem (setOf fun x => And (Membership.mem s x) (LT.lt (f x) r)) x
              y : E
              hy : Membership.mem (setOf fun x => And (Membership.mem s x) (LT.lt (f x) r)) y
              hxy : Ne x y
              a b : 𝕜
              ha : LT.lt 0 a
              hb : LT.lt 0 b
              hab : Eq (HAdd.hAdd a b) 1
              ⊢ LE.le (f y) r
            -/
          · exact hy.2.le
            /-
              🎉 no goals
            -/
        _ = r := Convex.combo_self hab r
        ⟩


theorem StrictConcaveOn.convex_gt (hf : StrictConcaveOn 𝕜 s f) (r : β) :
    Convex 𝕜 ({ x ∈ s | r < f x }) :=
  hf.dual.convex_lt r


/-- For a function on a convex set in a linearly ordered space (where the order and the algebraic
structures aren't necessarily compatible), in order to prove that it is convex, it suffices to
verify the inequality `f (a • x + b • y) ≤ a • f x + b • f y` only for `x < y` and positive `a`,
`b`. The main use case is `E = 𝕜` however one can apply it, e.g., to `𝕜^n` with lexicographic order.
-/
theorem LinearOrder.convexOn_of_lt (hs : Convex 𝕜 s)
    (hf : ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → x < y → ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b → a + b = 1 →
      f (a • x + b • y) ≤ a • f x + b • f y) :
    ConvexOn 𝕜 s f := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : OrderedAddCommMonoid β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : LinearOrder E
    s : Set E
    f : E → β
    hs : Convex 𝕜 s
    hf : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → LT.lt x y …
    ⊢ ConvexOn 𝕜 s f
  -/
  refine convexOn_iff_pairwise_pos.2 ⟨hs, fun x hx y hy hxy a b ha hb hab => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : OrderedAddCommMonoid β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : LinearOrder E
    s : Set E
    f : E → β
    hs : Convex 𝕜 s
    hf : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → LT.lt x y …
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul. …
  -/
  wlog h : x < y
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁵ : OrderedSemiring 𝕜
      inst✝⁴ : AddCommMonoid E
      inst✝³ : OrderedAddCommMonoid β
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 β
      inst✝ : LinearOrder E
      s : Set E
      f : E → β
      hs : Convex 𝕜 s
      hf : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → LT.lt x y …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hxy : Ne x y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      this : ∀ {𝕜 : Type u_1} {E : Type u_2} {β : Type u_5} [inst : OrderedSemiring  …
      h : Not (LT.lt x y)
      ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul. …
    -/
  · rw [add_comm (a • x), add_comm (a • f x)]
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁵ : OrderedSemiring 𝕜
      inst✝⁴ : AddCommMonoid E
      inst✝³ : OrderedAddCommMonoid β
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 β
      inst✝ : LinearOrder E
      s : Set E
      f : E → β
      hs : Convex 𝕜 s
      hf : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → LT.lt x y …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hxy : Ne x y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      this : ∀ {𝕜 : Type u_1} {E : Type u_2} {β : Type u_5} [inst : OrderedSemiring  …
      h : Not (LT.lt x y)
      ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul b y) (HSMul.hSMul a x))) (HAdd.hAdd (HSMul. …
    -/
    rw [add_comm] at hab
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁵ : OrderedSemiring 𝕜
      inst✝⁴ : AddCommMonoid E
      inst✝³ : OrderedAddCommMonoid β
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 β
      inst✝ : LinearOrder E
      s : Set E
      f : E → β
      hs : Convex 𝕜 s
      hf : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → LT.lt x y …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hxy : Ne x y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd b a) 1
      this : ∀ {𝕜 : Type u_1} {E : Type u_2} {β : Type u_5} [inst : OrderedSemiring  …
      h : Not (LT.lt x y)
      ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul b y) (HSMul.hSMul a x))) (HAdd.hAdd (HSMul. …
    -/
    exact this hs hf y hy x hx hxy.symm b a hb ha hab (hxy.lt_or_lt.resolve_left h)
    /-
      🎉 no goals
    -/
  /-
    𝕜✝ : Type u_1
    E✝ : Type u_2
    β✝ : Type u_5
    inst✝¹¹ : OrderedSemiring 𝕜✝
    inst✝¹⁰ : AddCommMonoid E✝
    inst✝⁹ : OrderedAddCommMonoid β✝
    inst✝⁸ : Module 𝕜✝ E✝
    inst✝⁷ : Module 𝕜✝ β✝
    inst✝⁶ : LinearOrder E✝
    s✝ : Set E✝
    f✝ : E✝ → β✝
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : OrderedAddCommMonoid β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : LinearOrder E
    s : Set E
    f : E → β
    hs : Convex 𝕜 s
    hf : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → LT.lt x y …
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    h : LT.lt x y
    ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul. …
  -/
  exact hf hx hy h ha hb hab
  /-
    🎉 no goals
  -/


/-- For a function on a convex set in a linearly ordered space (where the order and the algebraic
structures aren't necessarily compatible), in order to prove that it is concave it suffices to
verify the inequality `a • f x + b • f y ≤ f (a • x + b • y)` for `x < y` and positive `a`, `b`. The
main use case is `E = ℝ` however one can apply it, e.g., to `ℝ^n` with lexicographic order. -/
theorem LinearOrder.concaveOn_of_lt (hs : Convex 𝕜 s)
    (hf : ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → x < y → ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b → a + b = 1 →
      a • f x + b • f y ≤ f (a • x + b • y)) :
    ConcaveOn 𝕜 s f :=
  LinearOrder.convexOn_of_lt (β := βᵒᵈ) hs hf


/-- For a function on a convex set in a linearly ordered space (where the order and the algebraic
structures aren't necessarily compatible), in order to prove that it is strictly convex, it suffices
to verify the inequality `f (a • x + b • y) < a • f x + b • f y` for `x < y` and positive `a`, `b`.
The main use case is `E = 𝕜` however one can apply it, e.g., to `𝕜^n` with lexicographic order. -/
theorem LinearOrder.strictConvexOn_of_lt (hs : Convex 𝕜 s)
    (hf : ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → x < y → ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b → a + b = 1 →
      f (a • x + b • y) < a • f x + b • f y) :
    StrictConvexOn 𝕜 s f := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : OrderedAddCommMonoid β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : LinearOrder E
    s : Set E
    f : E → β
    hs : Convex 𝕜 s
    hf : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → LT.lt x y …
    ⊢ StrictConvexOn 𝕜 s f
  -/
  refine ⟨hs, fun x hx y hy hxy a b ha hb hab => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : OrderedAddCommMonoid β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : LinearOrder E
    s : Set E
    f : E → β
    hs : Convex 𝕜 s
    hf : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → LT.lt x y …
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LT.lt (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul. …
  -/
  wlog h : x < y
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁵ : OrderedSemiring 𝕜
      inst✝⁴ : AddCommMonoid E
      inst✝³ : OrderedAddCommMonoid β
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 β
      inst✝ : LinearOrder E
      s : Set E
      f : E → β
      hs : Convex 𝕜 s
      hf : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → LT.lt x y …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hxy : Ne x y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      this : ∀ {𝕜 : Type u_1} {E : Type u_2} {β : Type u_5} [inst : OrderedSemiring  …
      h : Not (LT.lt x y)
      ⊢ LT.lt (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul. …
    -/
  · rw [add_comm (a • x), add_comm (a • f x)]
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁵ : OrderedSemiring 𝕜
      inst✝⁴ : AddCommMonoid E
      inst✝³ : OrderedAddCommMonoid β
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 β
      inst✝ : LinearOrder E
      s : Set E
      f : E → β
      hs : Convex 𝕜 s
      hf : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → LT.lt x y …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hxy : Ne x y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      this : ∀ {𝕜 : Type u_1} {E : Type u_2} {β : Type u_5} [inst : OrderedSemiring  …
      h : Not (LT.lt x y)
      ⊢ LT.lt (f (HAdd.hAdd (HSMul.hSMul b y) (HSMul.hSMul a x))) (HAdd.hAdd (HSMul. …
    -/
    rw [add_comm] at hab
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁵ : OrderedSemiring 𝕜
      inst✝⁴ : AddCommMonoid E
      inst✝³ : OrderedAddCommMonoid β
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 β
      inst✝ : LinearOrder E
      s : Set E
      f : E → β
      hs : Convex 𝕜 s
      hf : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → LT.lt x y …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hxy : Ne x y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd b a) 1
      this : ∀ {𝕜 : Type u_1} {E : Type u_2} {β : Type u_5} [inst : OrderedSemiring  …
      h : Not (LT.lt x y)
      ⊢ LT.lt (f (HAdd.hAdd (HSMul.hSMul b y) (HSMul.hSMul a x))) (HAdd.hAdd (HSMul. …
    -/
    exact this hs hf y hy x hx hxy.symm b a hb ha hab (hxy.lt_or_lt.resolve_left h)
    /-
      🎉 no goals
    -/
  /-
    𝕜✝ : Type u_1
    E✝ : Type u_2
    β✝ : Type u_5
    inst✝¹¹ : OrderedSemiring 𝕜✝
    inst✝¹⁰ : AddCommMonoid E✝
    inst✝⁹ : OrderedAddCommMonoid β✝
    inst✝⁸ : Module 𝕜✝ E✝
    inst✝⁷ : Module 𝕜✝ β✝
    inst✝⁶ : LinearOrder E✝
    s✝ : Set E✝
    f✝ : E✝ → β✝
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : OrderedAddCommMonoid β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : LinearOrder E
    s : Set E
    f : E → β
    hs : Convex 𝕜 s
    hf : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → LT.lt x y …
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    h : LT.lt x y
    ⊢ LT.lt (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul. …
  -/
  exact hf hx hy h ha hb hab
  /-
    🎉 no goals
  -/


/-- For a function on a convex set in a linearly ordered space (where the order and the algebraic
structures aren't necessarily compatible), in order to prove that it is strictly concave it suffices
to verify the inequality `a • f x + b • f y < f (a • x + b • y)` for `x < y` and positive `a`, `b`.
The main use case is `E = 𝕜` however one can apply it, e.g., to `𝕜^n` with lexicographic order. -/
theorem LinearOrder.strictConcaveOn_of_lt (hs : Convex 𝕜 s)
    (hf : ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → x < y → ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b → a + b = 1 →
      a • f x + b • f y < f (a • x + b • y)) :
    StrictConcaveOn 𝕜 s f :=
  LinearOrder.strictConvexOn_of_lt (β := βᵒᵈ) hs hf


/-- If `g` is convex on `s`, so is `(f ∘ g)` on `f ⁻¹' s` for a linear `f`. -/
theorem ConvexOn.comp_linearMap {f : F → β} {s : Set F} (hf : ConvexOn 𝕜 s f) (g : E →ₗ[𝕜] F) :
    ConvexOn 𝕜 (g ⁻¹' s) (f ∘ g) :=
  ⟨hf.1.linear_preimage _, fun x hx y hy a b ha hb hab =>
    calc
                                                          /-
                                                            𝕜 : Type u_1
                                                            E : Type u_2
                                                            F : Type u_3
                                                            β : Type u_5
                                                            inst✝⁶ : OrderedSemiring 𝕜
                                                            inst✝⁵ : AddCommMonoid E
                                                            inst✝⁴ : AddCommMonoid F
                                                            inst✝³ : OrderedAddCommMonoid β
                                                            inst✝² : Module 𝕜 E
                                                            inst✝¹ : Module 𝕜 F
                                                            inst✝ : SMul 𝕜 β
                                                            f : F → β
                                                            s : Set F
                                                            hf : ConvexOn 𝕜 s f
                                                            g : LinearMap (RingHom.id 𝕜) E F
                                                            x : E
                                                            hx : Membership.mem (Set.preimage (⇑g) s) x
                                                            y : E
                                                            hy : Membership.mem (Set.preimage (⇑g) s) y
                                                            a b : 𝕜
                                                            ha : LE.le 0 a
                                                            hb : LE.le 0 b
                                                            hab : Eq (HAdd.hAdd a b) 1
                                                            ⊢ Eq (f (g (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)))) (f (HAdd.hAdd (HS …
                                                          -/
      f (g (a • x + b • y)) = f (a • g x + b • g y) := by rw [g.map_add, g.map_smul, g.map_smul]
                                                          /-
                                                            🎉 no goals
                                                          -/
      _ ≤ a • f (g x) + b • f (g y) := hf.2 hx hy ha hb hab⟩


/-- If `g` is concave on `s`, so is `(g ∘ f)` on `f ⁻¹' s` for a linear `f`. -/
theorem ConcaveOn.comp_linearMap {f : F → β} {s : Set F} (hf : ConcaveOn 𝕜 s f) (g : E →ₗ[𝕜] F) :
    ConcaveOn 𝕜 (g ⁻¹' s) (f ∘ g) :=
  hf.dual.comp_linearMap g


theorem StrictConvexOn.add_convexOn (hf : StrictConvexOn 𝕜 s f) (hg : ConvexOn 𝕜 s g) :
    StrictConvexOn 𝕜 s (f + g) :=
  ⟨hf.1, fun x hx y hy hxy a b ha hb hab =>
    calc
      f (a • x + b • y) + g (a • x + b • y) < a • f x + b • f y + (a • g x + b • g y) :=
        add_lt_add_of_lt_of_le (hf.2 hx hy hxy ha hb hab) (hg.2 hx hy ha.le hb.le hab)
                                                  /-
                                                    𝕜 : Type u_1
                                                    E : Type u_2
                                                    β : Type u_5
                                                    inst✝⁴ : OrderedSemiring 𝕜
                                                    inst✝³ : AddCommMonoid E
                                                    inst✝² : OrderedCancelAddCommMonoid β
                                                    inst✝¹ : SMul 𝕜 E
                                                    inst✝ : DistribMulAction 𝕜 β
                                                    s : Set E
                                                    f g : E → β
                                                    hf : StrictConvexOn 𝕜 s f
                                                    hg : ConvexOn 𝕜 s g
                                                    x : E
                                                    hx : Membership.mem s x
                                                    y : E
                                                    hy : Membership.mem s y
                                                    hxy : Ne x y
                                                    a b : 𝕜
                                                    ha : LT.lt 0 a
                                                    hb : LT.lt 0 b
                                                    hab : Eq (HAdd.hAdd a b) 1
                                                    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y))) (HAdd. …
                                                  -/
      _ = a • (f x + g x) + b • (f y + g y) := by rw [smul_add, smul_add, add_add_add_comm]⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem ConvexOn.add_strictConvexOn (hf : ConvexOn 𝕜 s f) (hg : StrictConvexOn 𝕜 s g) :
    StrictConvexOn 𝕜 s (f + g) :=
  add_comm g f ▸ hg.add_convexOn hf


theorem StrictConvexOn.add (hf : StrictConvexOn 𝕜 s f) (hg : StrictConvexOn 𝕜 s g) :
    StrictConvexOn 𝕜 s (f + g) :=
  ⟨hf.1, fun x hx y hy hxy a b ha hb hab =>
    calc
      f (a • x + b • y) + g (a • x + b • y) < a • f x + b • f y + (a • g x + b • g y) :=
        add_lt_add (hf.2 hx hy hxy ha hb hab) (hg.2 hx hy hxy ha hb hab)
                                                  /-
                                                    𝕜 : Type u_1
                                                    E : Type u_2
                                                    β : Type u_5
                                                    inst✝⁴ : OrderedSemiring 𝕜
                                                    inst✝³ : AddCommMonoid E
                                                    inst✝² : OrderedCancelAddCommMonoid β
                                                    inst✝¹ : SMul 𝕜 E
                                                    inst✝ : DistribMulAction 𝕜 β
                                                    s : Set E
                                                    f g : E → β
                                                    hf : StrictConvexOn 𝕜 s f
                                                    hg : StrictConvexOn 𝕜 s g
                                                    x : E
                                                    hx : Membership.mem s x
                                                    y : E
                                                    hy : Membership.mem s y
                                                    hxy : Ne x y
                                                    a b : 𝕜
                                                    ha : LT.lt 0 a
                                                    hb : LT.lt 0 b
                                                    hab : Eq (HAdd.hAdd a b) 1
                                                    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y))) (HAdd. …
                                                  -/
      _ = a • (f x + g x) + b • (f y + g y) := by rw [smul_add, smul_add, add_add_add_comm]⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem StrictConcaveOn.add_concaveOn (hf : StrictConcaveOn 𝕜 s f) (hg : ConcaveOn 𝕜 s g) :
    StrictConcaveOn 𝕜 s (f + g) :=
  hf.dual.add_convexOn hg.dual


theorem ConcaveOn.add_strictConcaveOn (hf : ConcaveOn 𝕜 s f) (hg : StrictConcaveOn 𝕜 s g) :
    StrictConcaveOn 𝕜 s (f + g) :=
  hf.dual.add_strictConvexOn hg.dual


theorem StrictConcaveOn.add (hf : StrictConcaveOn 𝕜 s f) (hg : StrictConcaveOn 𝕜 s g) :
    StrictConcaveOn 𝕜 s (f + g) :=
  hf.dual.add hg


theorem StrictConvexOn.add_const {γ : Type*} {f : E → γ} [OrderedCancelAddCommMonoid γ]
    [Module 𝕜 γ] (hf : StrictConvexOn 𝕜 s f) (b : γ) : StrictConvexOn 𝕜 s (f + fun _ => b) :=
  hf.add_convexOn (convexOn_const _ hf.1)


theorem StrictConcaveOn.add_const {γ : Type*} {f : E → γ} [OrderedCancelAddCommMonoid γ]
    [Module 𝕜 γ] (hf : StrictConcaveOn 𝕜 s f) (b : γ) : StrictConcaveOn 𝕜 s (f + fun _ => b) :=
  hf.add_concaveOn (concaveOn_const _ hf.1)


theorem ConvexOn.convex_lt (hf : ConvexOn 𝕜 s f) (r : β) : Convex 𝕜 ({ x ∈ s | f x < r }) :=
  convex_iff_forall_pos.2 fun x hx y hy a b ha hb hab =>
    ⟨hf.1 hx.1 hy.1 ha.le hb.le hab,
      calc
        f (a • x + b • y) ≤ a • f x + b • f y := hf.2 hx.1 hy.1 ha.le hb.le hab
        _ < a • r + b • r :=
          (add_lt_add_of_lt_of_le (smul_lt_smul_of_pos_left hx.2 ha)
            (smul_le_smul_of_nonneg_left hy.2.le hb.le))
        _ = r := Convex.combo_self hab _⟩


theorem ConcaveOn.convex_gt (hf : ConcaveOn 𝕜 s f) (r : β) : Convex 𝕜 ({ x ∈ s | r < f x }) :=
  hf.dual.convex_lt r


theorem ConvexOn.openSegment_subset_strict_epigraph (hf : ConvexOn 𝕜 s f) (p q : E × β)
    (hp : p.1 ∈ s ∧ f p.1 < p.2) (hq : q.1 ∈ s ∧ f q.1 ≤ q.2) :
    openSegment 𝕜 p q ⊆ { p : E × β | p.1 ∈ s ∧ f p.1 < p.2 } := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : OrderedCancelAddCommMonoid β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    p q : Prod E β
    hp : And (Membership.mem s p.1) (LT.lt (f p.1) p.2)
    hq : And (Membership.mem s q.1) (LE.le (f q.1) q.2)
    ⊢ HasSubset.Subset (openSegment 𝕜 p q) (setOf fun p => And (Membership.mem s p …
  -/
  rintro _ ⟨a, b, ha, hb, hab, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : OrderedCancelAddCommMonoid β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    p q : Prod E β
    hp : And (Membership.mem s p.1) (LT.lt (f p.1) p.2)
    hq : And (Membership.mem s q.1) (LE.le (f q.1) q.2)
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (setOf fun p => And (Membership.mem s p.1) (LT.lt (f p.1) p.2 …
  -/
  refine ⟨hf.1 hp.1 hq.1 ha.le hb.le hab, ?_⟩
  calc
    f (a • p.1 + b • q.1) ≤ a • f p.1 + b • f q.1 := hf.2 hp.1 hq.1 ha.le hb.le hab
    _ < a • p.2 + b • q.2 := add_lt_add_of_lt_of_le
       (smul_lt_smul_of_pos_left hp.2 ha) (smul_le_smul_of_nonneg_left hq.2 hb.le)


theorem ConcaveOn.openSegment_subset_strict_hypograph (hf : ConcaveOn 𝕜 s f) (p q : E × β)
    (hp : p.1 ∈ s ∧ p.2 < f p.1) (hq : q.1 ∈ s ∧ q.2 ≤ f q.1) :
    openSegment 𝕜 p q ⊆ { p : E × β | p.1 ∈ s ∧ p.2 < f p.1 } :=
  hf.dual.openSegment_subset_strict_epigraph p q hp hq


theorem ConvexOn.convex_strict_epigraph (hf : ConvexOn 𝕜 s f) :
    Convex 𝕜 { p : E × β | p.1 ∈ s ∧ f p.1 < p.2 } :=
  convex_iff_openSegment_subset.mpr fun p hp q hq =>
    hf.openSegment_subset_strict_epigraph p q hp ⟨hq.1, hq.2.le⟩


theorem ConcaveOn.convex_strict_hypograph (hf : ConcaveOn 𝕜 s f) :
    Convex 𝕜 { p : E × β | p.1 ∈ s ∧ p.2 < f p.1 } :=
  hf.dual.convex_strict_epigraph


/-- The pointwise maximum of convex functions is convex. -/
theorem ConvexOn.sup (hf : ConvexOn 𝕜 s f) (hg : ConvexOn 𝕜 s g) : ConvexOn 𝕜 s (f ⊔ g) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : LinearOrderedAddCommMonoid β
    inst✝² : SMul 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f g : E → β
    hf : ConvexOn 𝕜 s f
    hg : ConvexOn 𝕜 s g
    ⊢ ConvexOn 𝕜 s (Max.max f g)
  -/
  refine ⟨hf.left, fun x hx y hy a b ha hb hab => sup_le ?_ ?_⟩
  · calc
      f (a • x + b • y) ≤ a • f x + b • f y := hf.right hx hy ha hb hab
      _ ≤ a • (f x ⊔ g x) + b • (f y ⊔ g y) := by gcongr <;> apply le_sup_left
  · calc
      g (a • x + b • y) ≤ a • g x + b • g y := hg.right hx hy ha hb hab
      _ ≤ a • (f x ⊔ g x) + b • (f y ⊔ g y) := by gcongr <;> apply le_sup_right


/-- The pointwise minimum of concave functions is concave. -/
theorem ConcaveOn.inf (hf : ConcaveOn 𝕜 s f) (hg : ConcaveOn 𝕜 s g) : ConcaveOn 𝕜 s (f ⊓ g) :=
  hf.dual.sup hg


/-- The pointwise maximum of strictly convex functions is strictly convex. -/
theorem StrictConvexOn.sup (hf : StrictConvexOn 𝕜 s f) (hg : StrictConvexOn 𝕜 s g) :
    StrictConvexOn 𝕜 s (f ⊔ g) :=
  ⟨hf.left, fun x hx y hy hxy a b ha hb hab =>
    max_lt
      (calc
        f (a • x + b • y) < a • f x + b • f y := hf.2 hx hy hxy ha hb hab
                                                    /-
                                                      𝕜 : Type u_1
                                                      E : Type u_2
                                                      β : Type u_5
                                                      inst✝⁵ : OrderedSemiring 𝕜
                                                      inst✝⁴ : AddCommMonoid E
                                                      inst✝³ : LinearOrderedAddCommMonoid β
                                                      inst✝² : SMul 𝕜 E
                                                      inst✝¹ : Module 𝕜 β
                                                      inst✝ : OrderedSMul 𝕜 β
                                                      s : Set E
                                                      f g : E → β
                                                      hf : StrictConvexOn 𝕜 s f
                                                      hg : StrictConvexOn 𝕜 s g
                                                      x : E
                                                      hx : Membership.mem s x
                                                      y : E
                                                      hy : Membership.mem s y
                                                      hxy : Ne x y
                                                      a b : 𝕜
                                                      ha : LT.lt 0 a
                                                      hb : LT.lt 0 b
                                                      hab : Eq (HAdd.hAdd a b) 1
                                                      ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y))) (HAdd.hAdd (HS …
                                                    -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
        _ ≤ a • (f x ⊔ g x) + b • (f y ⊔ g y) := by gcongr <;> apply le_sup_left)
                                                               /-
                                                                 🎉 no goals
                                                               -/
      (calc
        g (a • x + b • y) < a • g x + b • g y := hg.2 hx hy hxy ha hb hab
                                                    /-
                                                      𝕜 : Type u_1
                                                      E : Type u_2
                                                      β : Type u_5
                                                      inst✝⁵ : OrderedSemiring 𝕜
                                                      inst✝⁴ : AddCommMonoid E
                                                      inst✝³ : LinearOrderedAddCommMonoid β
                                                      inst✝² : SMul 𝕜 E
                                                      inst✝¹ : Module 𝕜 β
                                                      inst✝ : OrderedSMul 𝕜 β
                                                      s : Set E
                                                      f g : E → β
                                                      hf : StrictConvexOn 𝕜 s f
                                                      hg : StrictConvexOn 𝕜 s g
                                                      x : E
                                                      hx : Membership.mem s x
                                                      y : E
                                                      hy : Membership.mem s y
                                                      hxy : Ne x y
                                                      a b : 𝕜
                                                      ha : LT.lt 0 a
                                                      hb : LT.lt 0 b
                                                      hab : Eq (HAdd.hAdd a b) 1
                                                      ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a (g x)) (HSMul.hSMul b (g y))) (HAdd.hAdd (HS …
                                                    -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
        _ ≤ a • (f x ⊔ g x) + b • (f y ⊔ g y) := by gcongr <;> apply le_sup_right)⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- The pointwise minimum of strictly concave functions is strictly concave. -/
theorem StrictConcaveOn.inf (hf : StrictConcaveOn 𝕜 s f) (hg : StrictConcaveOn 𝕜 s g) :
    StrictConcaveOn 𝕜 s (f ⊓ g) :=
  hf.dual.sup hg


/-- A convex function on a segment is upper-bounded by the max of its endpoints. -/
theorem ConvexOn.le_on_segment' (hf : ConvexOn 𝕜 s f) {x y : E} (hx : x ∈ s) (hy : y ∈ s) {a b : 𝕜}
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) : f (a • x + b • y) ≤ max (f x) (f y) :=
  calc
    f (a • x + b • y) ≤ a • f x + b • f y := hf.2 hx hy ha hb hab
    _ ≤ a • max (f x) (f y) + b • max (f x) (f y) := by
      /-
        𝕜 : Type u_1
        E : Type u_2
        β : Type u_5
        inst✝⁵ : OrderedSemiring 𝕜
        inst✝⁴ : AddCommMonoid E
        inst✝³ : LinearOrderedAddCommMonoid β
        inst✝² : SMul 𝕜 E
        inst✝¹ : Module 𝕜 β
        inst✝ : OrderedSMul 𝕜 β
        s : Set E
        f : E → β
        hf : ConvexOn 𝕜 s f
        x y : E
        hx : Membership.mem s x
        hy : Membership.mem s y
        a b : 𝕜
        ha : LE.le 0 a
        hb : LE.le 0 b
        hab : Eq (HAdd.hAdd a b) 1
        ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y))) (HAdd.hAdd (HS …
      -/
      gcongr
        /-
          case h₁.hb
          𝕜 : Type u_1
          E : Type u_2
          β : Type u_5
          inst✝⁵ : OrderedSemiring 𝕜
          inst✝⁴ : AddCommMonoid E
          inst✝³ : LinearOrderedAddCommMonoid β
          inst✝² : SMul 𝕜 E
          inst✝¹ : Module 𝕜 β
          inst✝ : OrderedSMul 𝕜 β
          s : Set E
          f : E → β
          hf : ConvexOn 𝕜 s f
          x y : E
          hx : Membership.mem s x
          hy : Membership.mem s y
          a b : 𝕜
          ha : LE.le 0 a
          hb : LE.le 0 b
          hab : Eq (HAdd.hAdd a b) 1
          ⊢ LE.le (f x) (Max.max (f x) (f y))
        -/
      · apply le_max_left
        /-
          🎉 no goals
        -/
        /-
          case h₂.hb
          𝕜 : Type u_1
          E : Type u_2
          β : Type u_5
          inst✝⁵ : OrderedSemiring 𝕜
          inst✝⁴ : AddCommMonoid E
          inst✝³ : LinearOrderedAddCommMonoid β
          inst✝² : SMul 𝕜 E
          inst✝¹ : Module 𝕜 β
          inst✝ : OrderedSMul 𝕜 β
          s : Set E
          f : E → β
          hf : ConvexOn 𝕜 s f
          x y : E
          hx : Membership.mem s x
          hy : Membership.mem s y
          a b : 𝕜
          ha : LE.le 0 a
          hb : LE.le 0 b
          hab : Eq (HAdd.hAdd a b) 1
          ⊢ LE.le (f y) (Max.max (f x) (f y))
        -/
      · apply le_max_right
        /-
          🎉 no goals
        -/
    _ = max (f x) (f y) := Convex.combo_self hab _


/-- A concave function on a segment is lower-bounded by the min of its endpoints. -/
theorem ConcaveOn.ge_on_segment' (hf : ConcaveOn 𝕜 s f) {x y : E} (hx : x ∈ s) (hy : y ∈ s)
    {a b : 𝕜} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) : min (f x) (f y) ≤ f (a • x + b • y) :=
  hf.dual.le_on_segment' hx hy ha hb hab


/-- A convex function on a segment is upper-bounded by the max of its endpoints. -/
theorem ConvexOn.le_on_segment (hf : ConvexOn 𝕜 s f) {x y z : E} (hx : x ∈ s) (hy : y ∈ s)
    (hz : z ∈ [x -[𝕜] y]) : f z ≤ max (f x) (f y) :=
  let ⟨_, _, ha, hb, hab, hz⟩ := hz
  hz ▸ hf.le_on_segment' hx hy ha hb hab


/-- A concave function on a segment is lower-bounded by the min of its endpoints. -/
theorem ConcaveOn.ge_on_segment (hf : ConcaveOn 𝕜 s f) {x y z : E} (hx : x ∈ s) (hy : y ∈ s)
    (hz : z ∈ [x -[𝕜] y]) : min (f x) (f y) ≤ f z :=
  hf.dual.le_on_segment hx hy hz


/-- A strictly convex function on an open segment is strictly upper-bounded by the max of its
endpoints. -/
theorem StrictConvexOn.lt_on_open_segment' (hf : StrictConvexOn 𝕜 s f) {x y : E} (hx : x ∈ s)
    (hy : y ∈ s) (hxy : x ≠ y) {a b : 𝕜} (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
    f (a • x + b • y) < max (f x) (f y) :=
  calc
    f (a • x + b • y) < a • f x + b • f y := hf.2 hx hy hxy ha hb hab
    _ ≤ a • max (f x) (f y) + b • max (f x) (f y) := by
      /-
        𝕜 : Type u_1
        E : Type u_2
        β : Type u_5
        inst✝⁵ : OrderedSemiring 𝕜
        inst✝⁴ : AddCommMonoid E
        inst✝³ : LinearOrderedAddCommMonoid β
        inst✝² : SMul 𝕜 E
        inst✝¹ : Module 𝕜 β
        inst✝ : OrderedSMul 𝕜 β
        s : Set E
        f : E → β
        hf : StrictConvexOn 𝕜 s f
        x y : E
        hx : Membership.mem s x
        hy : Membership.mem s y
        hxy : Ne x y
        a b : 𝕜
        ha : LT.lt 0 a
        hb : LT.lt 0 b
        hab : Eq (HAdd.hAdd a b) 1
        ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y))) (HAdd.hAdd (HS …
      -/
      gcongr
        /-
          case h₁.hb
          𝕜 : Type u_1
          E : Type u_2
          β : Type u_5
          inst✝⁵ : OrderedSemiring 𝕜
          inst✝⁴ : AddCommMonoid E
          inst✝³ : LinearOrderedAddCommMonoid β
          inst✝² : SMul 𝕜 E
          inst✝¹ : Module 𝕜 β
          inst✝ : OrderedSMul 𝕜 β
          s : Set E
          f : E → β
          hf : StrictConvexOn 𝕜 s f
          x y : E
          hx : Membership.mem s x
          hy : Membership.mem s y
          hxy : Ne x y
          a b : 𝕜
          ha : LT.lt 0 a
          hb : LT.lt 0 b
          hab : Eq (HAdd.hAdd a b) 1
          ⊢ LE.le (f x) (Max.max (f x) (f y))
        -/
      · apply le_max_left
        /-
          🎉 no goals
        -/
        /-
          case h₂.hb
          𝕜 : Type u_1
          E : Type u_2
          β : Type u_5
          inst✝⁵ : OrderedSemiring 𝕜
          inst✝⁴ : AddCommMonoid E
          inst✝³ : LinearOrderedAddCommMonoid β
          inst✝² : SMul 𝕜 E
          inst✝¹ : Module 𝕜 β
          inst✝ : OrderedSMul 𝕜 β
          s : Set E
          f : E → β
          hf : StrictConvexOn 𝕜 s f
          x y : E
          hx : Membership.mem s x
          hy : Membership.mem s y
          hxy : Ne x y
          a b : 𝕜
          ha : LT.lt 0 a
          hb : LT.lt 0 b
          hab : Eq (HAdd.hAdd a b) 1
          ⊢ LE.le (f y) (Max.max (f x) (f y))
        -/
      · apply le_max_right
        /-
          🎉 no goals
        -/
    _ = max (f x) (f y) := Convex.combo_self hab _


/-- A strictly concave function on an open segment is strictly lower-bounded by the min of its
endpoints. -/
theorem StrictConcaveOn.lt_on_open_segment' (hf : StrictConcaveOn 𝕜 s f) {x y : E} (hx : x ∈ s)
    (hy : y ∈ s) (hxy : x ≠ y) {a b : 𝕜} (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
    min (f x) (f y) < f (a • x + b • y) :=
  hf.dual.lt_on_open_segment' hx hy hxy ha hb hab


/-- A strictly convex function on an open segment is strictly upper-bounded by the max of its
endpoints. -/
theorem StrictConvexOn.lt_on_openSegment (hf : StrictConvexOn 𝕜 s f) {x y z : E} (hx : x ∈ s)
    (hy : y ∈ s) (hxy : x ≠ y) (hz : z ∈ openSegment 𝕜 x y) : f z < max (f x) (f y) :=
  let ⟨_, _, ha, hb, hab, hz⟩ := hz
  hz ▸ hf.lt_on_open_segment' hx hy hxy ha hb hab


/-- A strictly concave function on an open segment is strictly lower-bounded by the min of its
endpoints. -/
theorem StrictConcaveOn.lt_on_openSegment (hf : StrictConcaveOn 𝕜 s f) {x y z : E} (hx : x ∈ s)
    (hy : y ∈ s) (hxy : x ≠ y) (hz : z ∈ openSegment 𝕜 x y) : min (f x) (f y) < f z :=
  hf.dual.lt_on_openSegment hx hy hxy hz


theorem ConvexOn.le_left_of_right_le' (hf : ConvexOn 𝕜 s f) {x y : E} (hx : x ∈ s) (hy : y ∈ s)
    {a b : 𝕜} (ha : 0 < a) (hb : 0 ≤ b) (hab : a + b = 1) (hfy : f y ≤ f (a • x + b • y)) :
    f (a • x + b • y) ≤ f x :=
  le_of_not_lt fun h ↦ lt_irrefl (f (a • x + b • y)) <|
    calc
      f (a • x + b • y) ≤ a • f x + b • f y := hf.2 hx hy ha.le hb hab
      _ < a • f (a • x + b • y) + b • f (a • x + b • y) := add_lt_add_of_lt_of_le
          (smul_lt_smul_of_pos_left h ha) (smul_le_smul_of_nonneg_left hfy hb)
      _ = f (a • x + b • y) := Convex.combo_self hab _


theorem ConcaveOn.left_le_of_le_right' (hf : ConcaveOn 𝕜 s f) {x y : E} (hx : x ∈ s) (hy : y ∈ s)
    {a b : 𝕜} (ha : 0 < a) (hb : 0 ≤ b) (hab : a + b = 1) (hfy : f (a • x + b • y) ≤ f y) :
    f x ≤ f (a • x + b • y) :=
  hf.dual.le_left_of_right_le' hx hy ha hb hab hfy


theorem ConvexOn.le_right_of_left_le' (hf : ConvexOn 𝕜 s f) {x y : E} {a b : 𝕜} (hx : x ∈ s)
    (hy : y ∈ s) (ha : 0 ≤ a) (hb : 0 < b) (hab : a + b = 1) (hfx : f x ≤ f (a • x + b • y)) :
    f (a • x + b • y) ≤ f y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : LinearOrderedCancelAddCommMonoid β
    inst✝² : SMul 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    x y : E
    a b : 𝕜
    hx : Membership.mem s x
    hy : Membership.mem s y
    ha : LE.le 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hfx : LE.le (f x) (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)))
    ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (f y)
  -/
  rw [add_comm] at hab hfx ⊢
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : LinearOrderedCancelAddCommMonoid β
    inst✝² : SMul 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    x y : E
    a b : 𝕜
    hx : Membership.mem s x
    hy : Membership.mem s y
    ha : LE.le 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd b a) 1
    hfx : LE.le (f x) (f (HAdd.hAdd (HSMul.hSMul b y) (HSMul.hSMul a x)))
    ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul b y) (HSMul.hSMul a x))) (f y)
  -/
  exact hf.le_left_of_right_le' hy hx hb ha hab hfx
  /-
    🎉 no goals
  -/


theorem ConcaveOn.right_le_of_le_left' (hf : ConcaveOn 𝕜 s f) {x y : E} {a b : 𝕜} (hx : x ∈ s)
    (hy : y ∈ s) (ha : 0 ≤ a) (hb : 0 < b) (hab : a + b = 1) (hfx : f (a • x + b • y) ≤ f x) :
    f y ≤ f (a • x + b • y) :=
  hf.dual.le_right_of_left_le' hx hy ha hb hab hfx


theorem ConvexOn.le_left_of_right_le (hf : ConvexOn 𝕜 s f) {x y z : E} (hx : x ∈ s) (hy : y ∈ s)
    (hz : z ∈ openSegment 𝕜 x y) (hyz : f y ≤ f z) : f z ≤ f x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : LinearOrderedCancelAddCommMonoid β
    inst✝² : SMul 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    x y z : E
    hx : Membership.mem s x
    hy : Membership.mem s y
    hz : Membership.mem (openSegment 𝕜 x y) z
    hyz : LE.le (f y) (f z)
    ⊢ LE.le (f z) (f x)
  -/
  obtain ⟨a, b, ha, hb, hab, rfl⟩ := hz
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : LinearOrderedCancelAddCommMonoid β
    inst✝² : SMul 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    x y : E
    hx : Membership.mem s x
    hy : Membership.mem s y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hyz : LE.le (f y) (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)))
    ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (f x)
  -/
  exact hf.le_left_of_right_le' hx hy ha hb.le hab hyz
  /-
    🎉 no goals
  -/


theorem ConcaveOn.left_le_of_le_right (hf : ConcaveOn 𝕜 s f) {x y z : E} (hx : x ∈ s) (hy : y ∈ s)
    (hz : z ∈ openSegment 𝕜 x y) (hyz : f z ≤ f y) : f x ≤ f z :=
  hf.dual.le_left_of_right_le hx hy hz hyz


theorem ConvexOn.le_right_of_left_le (hf : ConvexOn 𝕜 s f) {x y z : E} (hx : x ∈ s) (hy : y ∈ s)
    (hz : z ∈ openSegment 𝕜 x y) (hxz : f x ≤ f z) : f z ≤ f y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : LinearOrderedCancelAddCommMonoid β
    inst✝² : SMul 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    x y z : E
    hx : Membership.mem s x
    hy : Membership.mem s y
    hz : Membership.mem (openSegment 𝕜 x y) z
    hxz : LE.le (f x) (f z)
    ⊢ LE.le (f z) (f y)
  -/
  obtain ⟨a, b, ha, hb, hab, rfl⟩ := hz
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : LinearOrderedCancelAddCommMonoid β
    inst✝² : SMul 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    x y : E
    hx : Membership.mem s x
    hy : Membership.mem s y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hxz : LE.le (f x) (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)))
    ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (f y)
  -/
  exact hf.le_right_of_left_le' hx hy ha.le hb hab hxz
  /-
    🎉 no goals
  -/


theorem ConcaveOn.right_le_of_le_left (hf : ConcaveOn 𝕜 s f) {x y z : E} (hx : x ∈ s) (hy : y ∈ s)
    (hz : z ∈ openSegment 𝕜 x y) (hxz : f z ≤ f x) : f y ≤ f z :=
  hf.dual.le_right_of_left_le hx hy hz hxz


theorem ConvexOn.lt_left_of_right_lt' (hf : ConvexOn 𝕜 s f) {x y : E} (hx : x ∈ s) (hy : y ∈ s)
    {a b : 𝕜} (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) (hfy : f y < f (a • x + b • y)) :
    f (a • x + b • y) < f x :=
  not_le.1 fun h ↦ lt_irrefl (f (a • x + b • y)) <|
    calc
      f (a • x + b • y) ≤ a • f x + b • f y := hf.2 hx hy ha.le hb.le hab
      _ < a • f (a • x + b • y) + b • f (a • x + b • y) := add_lt_add_of_le_of_lt
          (smul_le_smul_of_nonneg_left h ha.le) (smul_lt_smul_of_pos_left hfy hb)
      _ = f (a • x + b • y) := Convex.combo_self hab _


theorem ConcaveOn.left_lt_of_lt_right' (hf : ConcaveOn 𝕜 s f) {x y : E} (hx : x ∈ s) (hy : y ∈ s)
    {a b : 𝕜} (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) (hfy : f (a • x + b • y) < f y) :
    f x < f (a • x + b • y) :=
  hf.dual.lt_left_of_right_lt' hx hy ha hb hab hfy


theorem ConvexOn.lt_right_of_left_lt' (hf : ConvexOn 𝕜 s f) {x y : E} {a b : 𝕜} (hx : x ∈ s)
    (hy : y ∈ s) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) (hfx : f x < f (a • x + b • y)) :
    f (a • x + b • y) < f y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : LinearOrderedCancelAddCommMonoid β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    x y : E
    a b : 𝕜
    hx : Membership.mem s x
    hy : Membership.mem s y
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hfx : LT.lt (f x) (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)))
    ⊢ LT.lt (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (f y)
  -/
  rw [add_comm] at hab hfx ⊢
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : LinearOrderedCancelAddCommMonoid β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    x y : E
    a b : 𝕜
    hx : Membership.mem s x
    hy : Membership.mem s y
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd b a) 1
    hfx : LT.lt (f x) (f (HAdd.hAdd (HSMul.hSMul b y) (HSMul.hSMul a x)))
    ⊢ LT.lt (f (HAdd.hAdd (HSMul.hSMul b y) (HSMul.hSMul a x))) (f y)
  -/
  exact hf.lt_left_of_right_lt' hy hx hb ha hab hfx
  /-
    🎉 no goals
  -/


theorem ConcaveOn.lt_right_of_left_lt' (hf : ConcaveOn 𝕜 s f) {x y : E} {a b : 𝕜} (hx : x ∈ s)
    (hy : y ∈ s) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) (hfx : f (a • x + b • y) < f x) :
    f y < f (a • x + b • y) :=
  hf.dual.lt_right_of_left_lt' hx hy ha hb hab hfx


theorem ConvexOn.lt_left_of_right_lt (hf : ConvexOn 𝕜 s f) {x y z : E} (hx : x ∈ s) (hy : y ∈ s)
    (hz : z ∈ openSegment 𝕜 x y) (hyz : f y < f z) : f z < f x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : LinearOrderedCancelAddCommMonoid β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    x y z : E
    hx : Membership.mem s x
    hy : Membership.mem s y
    hz : Membership.mem (openSegment 𝕜 x y) z
    hyz : LT.lt (f y) (f z)
    ⊢ LT.lt (f z) (f x)
  -/
  obtain ⟨a, b, ha, hb, hab, rfl⟩ := hz
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : LinearOrderedCancelAddCommMonoid β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    x y : E
    hx : Membership.mem s x
    hy : Membership.mem s y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hyz : LT.lt (f y) (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)))
    ⊢ LT.lt (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (f x)
  -/
  exact hf.lt_left_of_right_lt' hx hy ha hb hab hyz
  /-
    🎉 no goals
  -/


theorem ConcaveOn.left_lt_of_lt_right (hf : ConcaveOn 𝕜 s f) {x y z : E} (hx : x ∈ s) (hy : y ∈ s)
    (hz : z ∈ openSegment 𝕜 x y) (hyz : f z < f y) : f x < f z :=
  hf.dual.lt_left_of_right_lt hx hy hz hyz


theorem ConvexOn.lt_right_of_left_lt (hf : ConvexOn 𝕜 s f) {x y z : E} (hx : x ∈ s) (hy : y ∈ s)
    (hz : z ∈ openSegment 𝕜 x y) (hxz : f x < f z) : f z < f y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : LinearOrderedCancelAddCommMonoid β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    x y z : E
    hx : Membership.mem s x
    hy : Membership.mem s y
    hz : Membership.mem (openSegment 𝕜 x y) z
    hxz : LT.lt (f x) (f z)
    ⊢ LT.lt (f z) (f y)
  -/
  obtain ⟨a, b, ha, hb, hab, rfl⟩ := hz
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : OrderedSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : LinearOrderedCancelAddCommMonoid β
    inst✝² : Module 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    s : Set E
    f : E → β
    hf : ConvexOn 𝕜 s f
    x y : E
    hx : Membership.mem s x
    hy : Membership.mem s y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hxz : LT.lt (f x) (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)))
    ⊢ LT.lt (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (f y)
  -/
  exact hf.lt_right_of_left_lt' hx hy ha hb hab hxz
  /-
    🎉 no goals
  -/


theorem ConcaveOn.lt_right_of_left_lt (hf : ConcaveOn 𝕜 s f) {x y z : E} (hx : x ∈ s) (hy : y ∈ s)
    (hz : z ∈ openSegment 𝕜 x y) (hxz : f z < f x) : f y < f z :=
  hf.dual.lt_right_of_left_lt hx hy hz hxz


/-- A function `-f` is convex iff `f` is concave. -/
@[simp]
theorem neg_convexOn_iff : ConvexOn 𝕜 s (-f) ↔ ConcaveOn 𝕜 s f := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : SMul 𝕜 E
    inst✝ : Module 𝕜 β
    s : Set E
    f : E → β
    ⊢ Iff (ConvexOn 𝕜 s (Neg.neg f)) (ConcaveOn 𝕜 s f)
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : SMul 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      ⊢ ConvexOn 𝕜 s (Neg.neg f) → ConcaveOn 𝕜 s f
    -/
  · rintro ⟨hconv, h⟩
    /-
      case mp.intro
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : SMul 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      hconv : Convex 𝕜 s
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
      ⊢ ConcaveOn 𝕜 s f
    -/
    refine ⟨hconv, fun x hx y hy a b ha hb hab => ?_⟩
    simp? [neg_apply, neg_le, add_comm] at h says
      simp only [Pi.neg_apply, smul_neg, le_add_neg_iff_add_le, add_comm,
        add_neg_le_iff_le_add] at h
    /-
      case mp.intro
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : SMul 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      hconv : Convex 𝕜 s
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
      ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y))) (f (HAdd.hAdd  …
    -/
    exact h hx hy ha hb hab
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : SMul 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      ⊢ ConcaveOn 𝕜 s f → ConvexOn 𝕜 s (Neg.neg f)
    -/
  · rintro ⟨hconv, h⟩
    /-
      case mpr.intro
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : SMul 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      hconv : Convex 𝕜 s
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
      ⊢ ConvexOn 𝕜 s (Neg.neg f)
    -/
    refine ⟨hconv, fun x hx y hy a b ha hb hab => ?_⟩
    /-
      case mpr.intro
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : SMul 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      hconv : Convex 𝕜 s
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ LE.le (Neg.neg f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd …
    -/
    rw [← neg_le_neg_iff]
    /-
      case mpr.intro
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : SMul 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      hconv : Convex 𝕜 s
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ LE.le (Neg.neg (HAdd.hAdd (HSMul.hSMul a (Neg.neg f x)) (HSMul.hSMul b (Neg. …
    -/
    simp_rw [neg_add, Pi.neg_apply, smul_neg, neg_neg]
    /-
      case mpr.intro
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : SMul 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      hconv : Convex 𝕜 s
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y))) (f (HAdd.hAdd  …
    -/
    exact h hx hy ha hb hab
    /-
      🎉 no goals
    -/


/-- A function `-f` is concave iff `f` is convex. -/
@[simp]
theorem neg_concaveOn_iff : ConcaveOn 𝕜 s (-f) ↔ ConvexOn 𝕜 s f := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : SMul 𝕜 E
    inst✝ : Module 𝕜 β
    s : Set E
    f : E → β
    ⊢ Iff (ConcaveOn 𝕜 s (Neg.neg f)) (ConvexOn 𝕜 s f)
  -/
  rw [← neg_convexOn_iff, neg_neg f]
  /-
    🎉 no goals
  -/


/-- A function `-f` is strictly convex iff `f` is strictly concave. -/
@[simp]
theorem neg_strictConvexOn_iff : StrictConvexOn 𝕜 s (-f) ↔ StrictConcaveOn 𝕜 s f := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : SMul 𝕜 E
    inst✝ : Module 𝕜 β
    s : Set E
    f : E → β
    ⊢ Iff (StrictConvexOn 𝕜 s (Neg.neg f)) (StrictConcaveOn 𝕜 s f)
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : SMul 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      ⊢ StrictConvexOn 𝕜 s (Neg.neg f) → StrictConcaveOn 𝕜 s f
    -/
  · rintro ⟨hconv, h⟩
    /-
      case mp.intro
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : SMul 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      hconv : Convex 𝕜 s
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ …
      ⊢ StrictConcaveOn 𝕜 s f
    -/
    refine ⟨hconv, fun x hx y hy hxy a b ha hb hab => ?_⟩
    simp only [ne_eq, Pi.neg_apply, smul_neg, lt_add_neg_iff_add_lt, add_comm,
      add_neg_lt_iff_lt_add] at h
    /-
      case mp.intro
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : SMul 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      hconv : Convex 𝕜 s
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hxy : Ne x y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → Not (Eq x  …
      ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y))) (f (HAdd.hAdd  …
    -/
    exact h hx hy hxy ha hb hab
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : SMul 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      ⊢ StrictConcaveOn 𝕜 s f → StrictConvexOn 𝕜 s (Neg.neg f)
    -/
  · rintro ⟨hconv, h⟩
    /-
      case mpr.intro
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : SMul 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      hconv : Convex 𝕜 s
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ …
      ⊢ StrictConvexOn 𝕜 s (Neg.neg f)
    -/
    refine ⟨hconv, fun x hx y hy hxy a b ha hb hab => ?_⟩
    /-
      case mpr.intro
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : SMul 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      hconv : Convex 𝕜 s
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hxy : Ne x y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ LT.lt (Neg.neg f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd …
    -/
    rw [← neg_lt_neg_iff]
    /-
      case mpr.intro
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : SMul 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      hconv : Convex 𝕜 s
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hxy : Ne x y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ LT.lt (Neg.neg (HAdd.hAdd (HSMul.hSMul a (Neg.neg f x)) (HSMul.hSMul b (Neg. …
    -/
    simp_rw [neg_add, Pi.neg_apply, smul_neg, neg_neg]
    /-
      case mpr.intro
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : OrderedSemiring 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : SMul 𝕜 E
      inst✝ : Module 𝕜 β
      s : Set E
      f : E → β
      hconv : Convex 𝕜 s
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hxy : Ne x y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y))) (f (HAdd.hAdd  …
    -/
    exact h hx hy hxy ha hb hab
    /-
      🎉 no goals
    -/


/-- A function `-f` is strictly concave iff `f` is strictly convex. -/
@[simp]
theorem neg_strictConcaveOn_iff : StrictConcaveOn 𝕜 s (-f) ↔ StrictConvexOn 𝕜 s f := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : SMul 𝕜 E
    inst✝ : Module 𝕜 β
    s : Set E
    f : E → β
    ⊢ Iff (StrictConcaveOn 𝕜 s (Neg.neg f)) (StrictConvexOn 𝕜 s f)
  -/
  rw [← neg_strictConvexOn_iff, neg_neg f]
  /-
    🎉 no goals
  -/


alias ⟨_, ConcaveOn.neg⟩ := neg_convexOn_iff


alias ⟨_, ConvexOn.neg⟩ := neg_concaveOn_iff


alias ⟨_, StrictConcaveOn.neg⟩ := neg_strictConvexOn_iff


alias ⟨_, StrictConvexOn.neg⟩ := neg_strictConcaveOn_iff


theorem ConvexOn.sub (hf : ConvexOn 𝕜 s f) (hg : ConcaveOn 𝕜 s g) : ConvexOn 𝕜 s (f - g) :=
  (sub_eq_add_neg f g).symm ▸ hf.add hg.neg


theorem ConcaveOn.sub (hf : ConcaveOn 𝕜 s f) (hg : ConvexOn 𝕜 s g) : ConcaveOn 𝕜 s (f - g) :=
  (sub_eq_add_neg f g).symm ▸ hf.add hg.neg


theorem StrictConvexOn.sub (hf : StrictConvexOn 𝕜 s f) (hg : StrictConcaveOn 𝕜 s g) :
    StrictConvexOn 𝕜 s (f - g) :=
  (sub_eq_add_neg f g).symm ▸ hf.add hg.neg


theorem StrictConcaveOn.sub (hf : StrictConcaveOn 𝕜 s f) (hg : StrictConvexOn 𝕜 s g) :
    StrictConcaveOn 𝕜 s (f - g) :=
  (sub_eq_add_neg f g).symm ▸ hf.add hg.neg


theorem ConvexOn.sub_strictConcaveOn (hf : ConvexOn 𝕜 s f) (hg : StrictConcaveOn 𝕜 s g) :
    StrictConvexOn 𝕜 s (f - g) :=
  (sub_eq_add_neg f g).symm ▸ hf.add_strictConvexOn hg.neg


theorem ConcaveOn.sub_strictConvexOn (hf : ConcaveOn 𝕜 s f) (hg : StrictConvexOn 𝕜 s g) :
    StrictConcaveOn 𝕜 s (f - g) :=
  (sub_eq_add_neg f g).symm ▸ hf.add_strictConcaveOn hg.neg


theorem StrictConvexOn.sub_concaveOn (hf : StrictConvexOn 𝕜 s f) (hg : ConcaveOn 𝕜 s g) :
    StrictConvexOn 𝕜 s (f - g) :=
  (sub_eq_add_neg f g).symm ▸ hf.add_convexOn hg.neg


theorem StrictConcaveOn.sub_convexOn (hf : StrictConcaveOn 𝕜 s f) (hg : ConvexOn 𝕜 s g) :
    StrictConcaveOn 𝕜 s (f - g) :=
  (sub_eq_add_neg f g).symm ▸ hf.add_concaveOn hg.neg


/-- Right translation preserves strict convexity. -/
theorem StrictConvexOn.translate_right (hf : StrictConvexOn 𝕜 s f) (c : E) :
    StrictConvexOn 𝕜 ((fun z => c + z) ⁻¹' s) (f ∘ fun z => c + z) :=
  ⟨hf.1.translate_preimage_right _, fun x hx y hy hxy a b ha hb hab =>
    calc
      f (c + (a • x + b • y)) = f (a • (c + x) + b • (c + y)) := by
        /-
          𝕜 : Type u_1
          E : Type u_2
          β : Type u_5
          inst✝⁴ : OrderedSemiring 𝕜
          inst✝³ : AddCancelCommMonoid E
          inst✝² : OrderedAddCommMonoid β
          inst✝¹ : Module 𝕜 E
          inst✝ : SMul 𝕜 β
          s : Set E
          f : E → β
          hf : StrictConvexOn 𝕜 s f
          c x : E
          hx : Membership.mem (Set.preimage (fun z => HAdd.hAdd c z) s) x
          y : E
          hy : Membership.mem (Set.preimage (fun z => HAdd.hAdd c z) s) y
          hxy : Ne x y
          a b : 𝕜
          ha : LT.lt 0 a
          hb : LT.lt 0 b
          hab : Eq (HAdd.hAdd a b) 1
          ⊢ Eq (f (HAdd.hAdd c (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)))) (f (HAd …
        -/
        rw [smul_add, smul_add, add_add_add_comm, Convex.combo_self hab]
        /-
          🎉 no goals
        -/
      _ < a • f (c + x) + b • f (c + y) := hf.2 hx hy ((add_right_injective c).ne hxy) ha hb hab⟩


/-- Right translation preserves strict concavity. -/
theorem StrictConcaveOn.translate_right (hf : StrictConcaveOn 𝕜 s f) (c : E) :
    StrictConcaveOn 𝕜 ((fun z => c + z) ⁻¹' s) (f ∘ fun z => c + z) :=
  hf.dual.translate_right _


/-- Left translation preserves strict convexity. -/
theorem StrictConvexOn.translate_left (hf : StrictConvexOn 𝕜 s f) (c : E) :
    StrictConvexOn 𝕜 ((fun z => c + z) ⁻¹' s) (f ∘ fun z => z + c) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCancelCommMonoid E
    inst✝² : OrderedAddCommMonoid β
    inst✝¹ : Module 𝕜 E
    inst✝ : SMul 𝕜 β
    s : Set E
    f : E → β
    hf : StrictConvexOn 𝕜 s f
    c : E
    ⊢ StrictConvexOn 𝕜 (Set.preimage (fun z => HAdd.hAdd c z) s) (Function.comp f  …
  -/
  simpa only [add_comm] using hf.translate_right c
  /-
    🎉 no goals
  -/


/-- Left translation preserves strict concavity. -/
theorem StrictConcaveOn.translate_left (hf : StrictConcaveOn 𝕜 s f) (c : E) :
    StrictConcaveOn 𝕜 ((fun z => c + z) ⁻¹' s) (f ∘ fun z => z + c) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : AddCancelCommMonoid E
    inst✝² : OrderedAddCommMonoid β
    inst✝¹ : Module 𝕜 E
    inst✝ : SMul 𝕜 β
    s : Set E
    f : E → β
    hf : StrictConcaveOn 𝕜 s f
    c : E
    ⊢ StrictConcaveOn 𝕜 (Set.preimage (fun z => HAdd.hAdd c z) s) (Function.comp f …
  -/
  simpa only [add_comm] using hf.translate_right c
  /-
    🎉 no goals
  -/


theorem ConvexOn.smul {c : 𝕜} (hc : 0 ≤ c) (hf : ConvexOn 𝕜 s f) : ConvexOn 𝕜 s fun x => c • f x :=
  ⟨hf.1, fun x hx y hy a b ha hb hab =>
    calc
      c • f (a • x + b • y) ≤ c • (a • f x + b • f y) :=
        smul_le_smul_of_nonneg_left (hf.2 hx hy ha hb hab) hc
                                          /-
                                            𝕜 : Type u_1
                                            E : Type u_2
                                            β : Type u_5
                                            inst✝⁵ : OrderedCommSemiring 𝕜
                                            inst✝⁴ : AddCommMonoid E
                                            inst✝³ : OrderedAddCommMonoid β
                                            inst✝² : SMul 𝕜 E
                                            inst✝¹ : Module 𝕜 β
                                            inst✝ : OrderedSMul 𝕜 β
                                            s : Set E
                                            f : E → β
                                            c : 𝕜
                                            hc : LE.le 0 c
                                            hf : ConvexOn 𝕜 s f
                                            x : E
                                            hx : Membership.mem s x
                                            y : E
                                            hy : Membership.mem s y
                                            a b : 𝕜
                                            ha : LE.le 0 a
                                            hb : LE.le 0 b
                                            hab : Eq (HAdd.hAdd a b) 1
                                            ⊢ Eq (HSMul.hSMul c (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y)))) ( …
                                          -/
      _ = a • c • f x + b • c • f y := by rw [smul_add, smul_comm c, smul_comm c]⟩
                                          /-
                                            🎉 no goals
                                          -/


theorem ConcaveOn.smul {c : 𝕜} (hc : 0 ≤ c) (hf : ConcaveOn 𝕜 s f) :
    ConcaveOn 𝕜 s fun x => c • f x :=
  hf.dual.smul hc


/-- If a function is convex on `s`, it remains convex when precomposed by an affine map. -/
theorem ConvexOn.comp_affineMap {f : F → β} (g : E →ᵃ[𝕜] F) {s : Set F} (hf : ConvexOn 𝕜 s f) :
    ConvexOn 𝕜 (g ⁻¹' s) (f ∘ g) :=
  ⟨hf.1.affine_preimage _, fun x hx y hy a b ha hb hab =>
    calc
      (f ∘ g) (a • x + b • y) = f (g (a • x + b • y)) := rfl
                                      /-
                                        𝕜 : Type u_1
                                        E : Type u_2
                                        F : Type u_3
                                        β : Type u_5
                                        inst✝⁶ : LinearOrderedField 𝕜
                                        inst✝⁵ : AddCommGroup E
                                        inst✝⁴ : AddCommGroup F
                                        inst✝³ : OrderedAddCommMonoid β
                                        inst✝² : Module 𝕜 E
                                        inst✝¹ : Module 𝕜 F
                                        inst✝ : SMul 𝕜 β
                                        f : F → β
                                        g : AffineMap 𝕜 E F
                                        s : Set F
                                        hf : ConvexOn 𝕜 s f
                                        x : E
                                        hx : Membership.mem (Set.preimage (⇑g) s) x
                                        y : E
                                        hy : Membership.mem (Set.preimage (⇑g) s) y
                                        a b : 𝕜
                                        ha : LE.le 0 a
                                        hb : LE.le 0 b
                                        hab : Eq (HAdd.hAdd a b) 1
                                        ⊢ Eq (f (g (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y)))) (f (HAdd.hAdd (HS …
                                      -/
      _ = f (a • g x + b • g y) := by rw [Convex.combo_affine_apply hab]
                                      /-
                                        🎉 no goals
                                      -/
      _ ≤ a • f (g x) + b • f (g y) := hf.2 hx hy ha hb hab⟩


/-- If a function is concave on `s`, it remains concave when precomposed by an affine map. -/
theorem ConcaveOn.comp_affineMap {f : F → β} (g : E →ᵃ[𝕜] F) {s : Set F} (hf : ConcaveOn 𝕜 s f) :
    ConcaveOn 𝕜 (g ⁻¹' s) (f ∘ g) :=
  hf.dual.comp_affineMap g


theorem convexOn_iff_div {f : E → β} :
    ConvexOn 𝕜 s f ↔
      Convex 𝕜 s ∧ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → ∀ ⦃a b : 𝕜⦄, 0 ≤ a → 0 ≤ b → 0 < a + b →
        f ((a / (a + b)) • x + (b / (a + b)) • y) ≤ (a / (a + b)) • f x + (b / (a + b)) • f y :=
  and_congr Iff.rfl ⟨by
    /-
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : LinearOrderedField 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : SMul 𝕜 E
      inst✝ : SMul 𝕜 β
      s : Set E
      f : E → β
      ⊢ (∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜⦄ …
    -/
    intro h x hx y hy a b ha hb hab
    /-
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : LinearOrderedField 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : SMul 𝕜 E
      inst✝ : SMul 𝕜 β
      s : Set E
      f : E → β
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : LT.lt 0 (HAdd.hAdd a b)
      ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv a (HAdd.hAdd a b)) x) (HSMul.hSM …
    -/
    apply h hx hy (div_nonneg ha hab.le) (div_nonneg hb hab.le)
    /-
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : LinearOrderedField 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : SMul 𝕜 E
      inst✝ : SMul 𝕜 β
      s : Set E
      f : E → β
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : LT.lt 0 (HAdd.hAdd a b)
      ⊢ Eq (HAdd.hAdd (HDiv.hDiv a (HAdd.hAdd a b)) (HDiv.hDiv b (HAdd.hAdd a b))) 1
    -/
    rw [← add_div, div_self hab.ne'], by
    /-
      🎉 no goals
    -/
    /-
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : LinearOrderedField 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : SMul 𝕜 E
      inst✝ : SMul 𝕜 β
      s : Set E
      f : E → β
      ⊢ (∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜⦄ …
    -/
    intro h x hx y hy a b ha hb hab
    /-
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : LinearOrderedField 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : SMul 𝕜 E
      inst✝ : SMul 𝕜 β
      s : Set E
      f : E → β
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → ∀ ⦃a b : 𝕜 …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      a b : 𝕜
      ha : LE.le 0 a
      hb : LE.le 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul. …
    -/
    simpa [hab, zero_lt_one] using h hx hy ha hb⟩
    /-
      🎉 no goals
    -/


theorem concaveOn_iff_div {f : E → β} :
    ConcaveOn 𝕜 s f ↔
      Convex 𝕜 s ∧ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → ∀ ⦃a b : 𝕜⦄, 0 ≤ a → 0 ≤ b → 0 < a + b →
        (a / (a + b)) • f x + (b / (a + b)) • f y ≤ f ((a / (a + b)) • x + (b / (a + b)) • y) :=
  convexOn_iff_div (β := βᵒᵈ)


theorem strictConvexOn_iff_div {f : E → β} :
    StrictConvexOn 𝕜 s f ↔
      Convex 𝕜 s ∧ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → x ≠ y → ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b →
        f ((a / (a + b)) • x + (b / (a + b)) • y) < (a / (a + b)) • f x + (b / (a + b)) • f y :=
  and_congr Iff.rfl ⟨by
    /-
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : LinearOrderedField 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : SMul 𝕜 E
      inst✝ : SMul 𝕜 β
      s : Set E
      f : E → β
      ⊢ (∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀  …
    -/
    intro h x hx y hy hxy a b ha hb
    /-
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : LinearOrderedField 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : SMul 𝕜 E
      inst✝ : SMul 𝕜 β
      s : Set E
      f : E → β
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hxy : Ne x y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      ⊢ LT.lt (f (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv a (HAdd.hAdd a b)) x) (HSMul.hSM …
    -/
    have hab := add_pos ha hb
    /-
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : LinearOrderedField 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : SMul 𝕜 E
      inst✝ : SMul 𝕜 β
      s : Set E
      f : E → β
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hxy : Ne x y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : LT.lt 0 (HAdd.hAdd a b)
      ⊢ LT.lt (f (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv a (HAdd.hAdd a b)) x) (HSMul.hSM …
    -/
    apply h hx hy hxy (div_pos ha hab) (div_pos hb hab)
    /-
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : LinearOrderedField 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : SMul 𝕜 E
      inst✝ : SMul 𝕜 β
      s : Set E
      f : E → β
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hxy : Ne x y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : LT.lt 0 (HAdd.hAdd a b)
      ⊢ Eq (HAdd.hAdd (HDiv.hDiv a (HAdd.hAdd a b)) (HDiv.hDiv b (HAdd.hAdd a b))) 1
    -/
    rw [← add_div, div_self hab.ne'], by
    /-
      🎉 no goals
    -/
    /-
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : LinearOrderedField 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : SMul 𝕜 E
      inst✝ : SMul 𝕜 β
      s : Set E
      f : E → β
      ⊢ (∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀  …
    -/
    intro h x hx y hy hxy a b ha hb hab
    /-
      𝕜 : Type u_1
      E : Type u_2
      β : Type u_5
      inst✝⁴ : LinearOrderedField 𝕜
      inst✝³ : AddCommMonoid E
      inst✝² : OrderedAddCommMonoid β
      inst✝¹ : SMul 𝕜 E
      inst✝ : SMul 𝕜 β
      s : Set E
      f : E → β
      h : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → Ne x y → ∀ …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hxy : Ne x y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ LT.lt (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul. …
    -/
    simpa [hab, zero_lt_one] using h hx hy hxy ha hb⟩
    /-
      🎉 no goals
    -/


theorem strictConcaveOn_iff_div {f : E → β} :
    StrictConcaveOn 𝕜 s f ↔
      Convex 𝕜 s ∧ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → x ≠ y → ∀ ⦃a b : 𝕜⦄, 0 < a → 0 < b →
        (a / (a + b)) • f x + (b / (a + b)) • f y < f ((a / (a + b)) • x + (b / (a + b)) • y) :=
  strictConvexOn_iff_div (β := βᵒᵈ)


theorem OrderIso.strictConvexOn_symm (f : α ≃o β) (hf : StrictConcaveOn 𝕜 univ f) :
    StrictConvexOn 𝕜 univ f.symm := by
  /-
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : StrictConcaveOn 𝕜 _root_.Set.univ ⇑f
    ⊢ StrictConvexOn 𝕜 _root_.Set.univ ⇑f.symm
  -/
  refine ⟨convex_univ, fun x _ y _ hxy a b ha hb hab => ?_⟩
  /-
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : StrictConcaveOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LT.lt (f.symm (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (H …
  -/
  obtain ⟨x', hx''⟩ := f.surjective.exists.mp ⟨x, rfl⟩
  /-
    case intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : StrictConcaveOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    ⊢ LT.lt (f.symm (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (H …
  -/
  obtain ⟨y', hy''⟩ := f.surjective.exists.mp ⟨y, rfl⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : StrictConcaveOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    y' : α
    hy'' : Eq y (f y')
    ⊢ LT.lt (f.symm (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (H …
  -/
  have hxy' : x' ≠ y' := by rw [← f.injective.ne_iff, ← hx'', ← hy'']; exact hxy
  /-
    case intro.intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : StrictConcaveOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    y' : α
    hy'' : Eq y (f y')
    hxy' : Ne x' y'
    ⊢ LT.lt (f.symm (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (H …
  -/
  simp only [hx'', hy'', OrderIso.symm_apply_apply, gt_iff_lt]
  /-
    case intro.intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : StrictConcaveOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    y' : α
    hy'' : Eq y (f y')
    hxy' : Ne x' y'
    ⊢ LT.lt (f.symm (HAdd.hAdd (HSMul.hSMul a (f x')) (HSMul.hSMul b (f y')))) (HA …
  -/
  rw [← f.lt_iff_lt, OrderIso.apply_symm_apply]
  /-
    case intro.intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : StrictConcaveOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    y' : α
    hy'' : Eq y (f y')
    hxy' : Ne x' y'
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a (f x')) (HSMul.hSMul b (f y'))) (f (HAdd.hAd …
  -/
  exact hf.2 (by simp : x' ∈ univ) (by simp : y' ∈ univ) hxy' ha hb hab
  /-
    🎉 no goals
  -/


theorem OrderIso.convexOn_symm (f : α ≃o β) (hf : ConcaveOn 𝕜 univ f) :
    ConvexOn 𝕜 univ f.symm := by
  /-
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : ConcaveOn 𝕜 _root_.Set.univ ⇑f
    ⊢ ConvexOn 𝕜 _root_.Set.univ ⇑f.symm
  -/
  refine ⟨convex_univ, fun x _ y _ a b ha hb hab => ?_⟩
  /-
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : ConcaveOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LE.le (f.symm (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (H …
  -/
  obtain ⟨x', hx''⟩ := f.surjective.exists.mp ⟨x, rfl⟩
  /-
    case intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : ConcaveOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    ⊢ LE.le (f.symm (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (H …
  -/
  obtain ⟨y', hy''⟩ := f.surjective.exists.mp ⟨y, rfl⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : ConcaveOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    y' : α
    hy'' : Eq y (f y')
    ⊢ LE.le (f.symm (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (H …
  -/
  simp only [hx'', hy'', OrderIso.symm_apply_apply, gt_iff_lt]
  /-
    case intro.intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : ConcaveOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    y' : α
    hy'' : Eq y (f y')
    ⊢ LE.le (f.symm (HAdd.hAdd (HSMul.hSMul a (f x')) (HSMul.hSMul b (f y')))) (HA …
  -/
  rw [← f.le_iff_le, OrderIso.apply_symm_apply]
  /-
    case intro.intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : ConcaveOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    y' : α
    hy'' : Eq y (f y')
    ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a (f x')) (HSMul.hSMul b (f y'))) (f (HAdd.hAd …
  -/
  exact hf.2 (by simp : x' ∈ univ) (by simp : y' ∈ univ) ha hb hab
  /-
    🎉 no goals
  -/


theorem OrderIso.strictConcaveOn_symm (f : α ≃o β) (hf : StrictConvexOn 𝕜 univ f) :
    StrictConcaveOn 𝕜 univ f.symm := by
  /-
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : StrictConvexOn 𝕜 _root_.Set.univ ⇑f
    ⊢ StrictConcaveOn 𝕜 _root_.Set.univ ⇑f.symm
  -/
  refine ⟨convex_univ, fun x _ y _ hxy a b ha hb hab => ?_⟩
  /-
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : StrictConvexOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a (f.symm x)) (HSMul.hSMul b (f.symm y))) (f.s …
  -/
  obtain ⟨x', hx''⟩ := f.surjective.exists.mp ⟨x, rfl⟩
  /-
    case intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : StrictConvexOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a (f.symm x)) (HSMul.hSMul b (f.symm y))) (f.s …
  -/
  obtain ⟨y', hy''⟩ := f.surjective.exists.mp ⟨y, rfl⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : StrictConvexOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    y' : α
    hy'' : Eq y (f y')
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a (f.symm x)) (HSMul.hSMul b (f.symm y))) (f.s …
  -/
  have hxy' : x' ≠ y' := by rw [← f.injective.ne_iff, ← hx'', ← hy'']; exact hxy
  /-
    case intro.intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : StrictConvexOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    y' : α
    hy'' : Eq y (f y')
    hxy' : Ne x' y'
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a (f.symm x)) (HSMul.hSMul b (f.symm y))) (f.s …
  -/
  simp only [hx'', hy'', OrderIso.symm_apply_apply, gt_iff_lt]
  /-
    case intro.intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : StrictConvexOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    y' : α
    hy'' : Eq y (f y')
    hxy' : Ne x' y'
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a x') (HSMul.hSMul b y')) (f.symm (HAdd.hAdd ( …
  -/
  rw [← f.lt_iff_lt, OrderIso.apply_symm_apply]
  /-
    case intro.intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : StrictConvexOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    hxy : Ne x y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    y' : α
    hy'' : Eq y (f y')
    hxy' : Ne x' y'
    ⊢ LT.lt (f (HAdd.hAdd (HSMul.hSMul a x') (HSMul.hSMul b y'))) (HAdd.hAdd (HSMu …
  -/
  exact hf.2 (by simp : x' ∈ univ) (by simp : y' ∈ univ) hxy' ha hb hab
  /-
    🎉 no goals
  -/


theorem OrderIso.concaveOn_symm (f : α ≃o β) (hf : ConvexOn 𝕜 univ f) :
    ConcaveOn 𝕜 univ f.symm := by
  /-
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : ConvexOn 𝕜 _root_.Set.univ ⇑f
    ⊢ ConcaveOn 𝕜 _root_.Set.univ ⇑f.symm
  -/
  refine ⟨convex_univ, fun x _ y _ a b ha hb hab => ?_⟩
  /-
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : ConvexOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a (f.symm x)) (HSMul.hSMul b (f.symm y))) (f.s …
  -/
  obtain ⟨x', hx''⟩ := f.surjective.exists.mp ⟨x, rfl⟩
  /-
    case intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : ConvexOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a (f.symm x)) (HSMul.hSMul b (f.symm y))) (f.s …
  -/
  obtain ⟨y', hy''⟩ := f.surjective.exists.mp ⟨y, rfl⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : ConvexOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    y' : α
    hy'' : Eq y (f y')
    ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a (f.symm x)) (HSMul.hSMul b (f.symm y))) (f.s …
  -/
  simp only [hx'', hy'', OrderIso.symm_apply_apply, gt_iff_lt]
  /-
    case intro.intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : ConvexOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    y' : α
    hy'' : Eq y (f y')
    ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a x') (HSMul.hSMul b y')) (f.symm (HAdd.hAdd ( …
  -/
  rw [← f.le_iff_le, OrderIso.apply_symm_apply]
  /-
    case intro.intro
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝⁴ : OrderedSemiring 𝕜
    inst✝³ : OrderedAddCommMonoid α
    inst✝² : SMul 𝕜 α
    inst✝¹ : OrderedAddCommMonoid β
    inst✝ : SMul 𝕜 β
    f : OrderIso α β
    hf : ConvexOn 𝕜 _root_.Set.univ ⇑f
    x : β
    x✝¹ : Membership.mem _root_.Set.univ x
    y : β
    x✝ : Membership.mem _root_.Set.univ y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : α
    hx'' : Eq x (f x')
    y' : α
    hy'' : Eq y (f y')
    ⊢ LE.le (f (HAdd.hAdd (HSMul.hSMul a x') (HSMul.hSMul b y'))) (HAdd.hAdd (HSMu …
  -/
  exact hf.2 (by simp : x' ∈ univ) (by simp : y' ∈ univ) ha hb hab
  /-
    🎉 no goals
  -/


/-- A strictly convex function admits at most one global minimum. -/
lemma StrictConvexOn.eq_of_isMinOn (hf : StrictConvexOn 𝕜 s f) (hfx : IsMinOn f s x)
    (hfy : IsMinOn f s y) (hx : x ∈ s) (hy : y ∈ s) : x = y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : OrderedAddCommMonoid β
    inst✝³ : AddCommMonoid E
    inst✝² : SMul 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    f : E → β
    s : Set E
    x y : E
    hf : StrictConvexOn 𝕜 s f
    hfx : IsMinOn f s x
    hfy : IsMinOn f s y
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ Eq x y
  -/
  by_contra hxy
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : OrderedAddCommMonoid β
    inst✝³ : AddCommMonoid E
    inst✝² : SMul 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    f : E → β
    s : Set E
    x y : E
    hf : StrictConvexOn 𝕜 s f
    hfx : IsMinOn f s x
    hfy : IsMinOn f s y
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxy : Not (Eq x y)
    ⊢ False
  -/
  let z := (2 : 𝕜)⁻¹ • x + (2 : 𝕜)⁻¹ • y
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : OrderedAddCommMonoid β
    inst✝³ : AddCommMonoid E
    inst✝² : SMul 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    f : E → β
    s : Set E
    x y : E
    hf : StrictConvexOn 𝕜 s f
    hfx : IsMinOn f s x
    hfy : IsMinOn f s y
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxy : Not (Eq x y)
    z : E := HAdd.hAdd (HSMul.hSMul (Inv.inv 2) x) (HSMul.hSMul (Inv.inv 2) y)
    ⊢ False
  -/
  have hz : z ∈ s := hf.1 hx hy (by norm_num) (by norm_num) <| by norm_num
  /-
    𝕜 : Type u_1
    E : Type u_2
    β : Type u_5
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : OrderedAddCommMonoid β
    inst✝³ : AddCommMonoid E
    inst✝² : SMul 𝕜 E
    inst✝¹ : Module 𝕜 β
    inst✝ : OrderedSMul 𝕜 β
    f : E → β
    s : Set E
    x y : E
    hf : StrictConvexOn 𝕜 s f
    hfx : IsMinOn f s x
    hfy : IsMinOn f s y
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxy : Not (Eq x y)
    z : E := HAdd.hAdd (HSMul.hSMul (Inv.inv 2) x) (HSMul.hSMul (Inv.inv 2) y)
    hz : Membership.mem s z
    ⊢ False
  -/
  refine lt_irrefl (f z) ?_
  calc
    f z < _ := hf.2 hx hy hxy (by norm_num) (by norm_num) <| by norm_num
    _ ≤ (2 : 𝕜)⁻¹ • f z + (2 : 𝕜)⁻¹ • f z := by gcongr; exacts [hfx hz, hfy hz]
    _ = f z := by rw [← _root_.add_smul]; norm_num


/-- A strictly concave function admits at most one global maximum. -/
lemma StrictConcaveOn.eq_of_isMaxOn (hf : StrictConcaveOn 𝕜 s f) (hfx : IsMaxOn f s x)
    (hfy : IsMaxOn f s y) (hx : x ∈ s) (hy : y ∈ s) : x = y :=
  hf.dual.eq_of_isMinOn hfx hfy hx hy


theorem ConvexOn.le_right_of_left_le'' (hf : ConvexOn 𝕜 s f) (hx : x ∈ s) (hz : z ∈ s) (hxy : x < y)
    (hyz : y ≤ z) (h : f x ≤ f y) : f y ≤ f z :=
  hyz.eq_or_lt.elim (fun hyz => (congr_arg f hyz).le) fun hyz =>
    hf.le_right_of_left_le hx hz (Ioo_subset_openSegment ⟨hxy, hyz⟩) h


theorem ConvexOn.le_left_of_right_le'' (hf : ConvexOn 𝕜 s f) (hx : x ∈ s) (hz : z ∈ s) (hxy : x ≤ y)
    (hyz : y < z) (h : f z ≤ f y) : f y ≤ f x :=
  hxy.eq_or_lt.elim (fun hxy => (congr_arg f hxy).ge) fun hxy =>
    hf.le_left_of_right_le hx hz (Ioo_subset_openSegment ⟨hxy, hyz⟩) h


theorem ConcaveOn.right_le_of_le_left'' (hf : ConcaveOn 𝕜 s f) (hx : x ∈ s) (hz : z ∈ s)
    (hxy : x < y) (hyz : y ≤ z) (h : f y ≤ f x) : f z ≤ f y :=
  hf.dual.le_right_of_left_le'' hx hz hxy hyz h


theorem ConcaveOn.left_le_of_le_right'' (hf : ConcaveOn 𝕜 s f) (hx : x ∈ s) (hz : z ∈ s)
    (hxy : x ≤ y) (hyz : y < z) (h : f y ≤ f z) : f x ≤ f y :=
  hf.dual.le_left_of_right_le'' hx hz hxy hyz h


