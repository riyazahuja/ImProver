instance : MorphismProperty.RespectsIso (topologically Function.Injective) :=
  topologically_respectsIso _ (fun e ↦ e.injective) (fun _ _ hf hg ↦ hg.comp hf)


instance injective_isLocalAtTarget : IsLocalAtTarget (topologically Function.Injective) := by
  refine topologically_isLocalAtTarget _ (fun _ s _ _ h ↦ h.restrictPreimage s)
    fun f ι U H _ hf x₁ x₂ e ↦ ?_
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    α✝ β✝ : Type u_1
    inst✝¹ : TopologicalSpace α✝
    inst✝ : TopologicalSpace β✝
    f : α✝ → β✝
    ι : Type u_1
    U : ι → TopologicalSpace.Opens β✝
    H : Eq (iSup U) Top.top
    x✝ : Continuous f
    hf : ∀ (i : ι), Function.Injective ((U i).carrier.restrictPreimage f)
    x₁ x₂ : α✝
    e : Eq (f x₁) (f x₂)
    ⊢ Eq x₁ x₂
  -/
  obtain ⟨i, hxi⟩ : ∃ i, f x₁ ∈ U i := by simpa using congr(f x₁ ∈ $H)
  /-
    case intro
    X Y Z : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    α✝ β✝ : Type u_1
    inst✝¹ : TopologicalSpace α✝
    inst✝ : TopologicalSpace β✝
    f : α✝ → β✝
    ι : Type u_1
    U : ι → TopologicalSpace.Opens β✝
    H : Eq (iSup U) Top.top
    x✝ : Continuous f
    hf : ∀ (i : ι), Function.Injective ((U i).carrier.restrictPreimage f)
    x₁ x₂ : α✝
    e : Eq (f x₁) (f x₂)
    i : ι
    hxi : Membership.mem (U i) (f x₁)
    ⊢ Eq x₁ x₂
  -/
  exact congr(($(@hf i ⟨x₁, hxi⟩ ⟨x₂, show f x₂ ∈ U i from e ▸ hxi⟩ (Subtype.ext e))).1)
  /-
    🎉 no goals
  -/


/-- A morphism of schemes is surjective if the underlying map is. -/
@[mk_iff]
class Surjective : Prop where
  surj : Function.Surjective f.base


lemma surjective_eq_topologically :
                                                          /-
                                                            ⊢ Eq (@AlgebraicGeometry.Surjective) (AlgebraicGeometry.topologically fun {α β …
                                                          -/
    @Surjective = topologically Function.Surjective := by ext; exact surjective_iff _
                                                               /-
                                                                 🎉 no goals
                                                               -/


lemma Scheme.Hom.surjective (f : X.Hom Y) [Surjective f] : Function.Surjective f.base :=
  Surjective.surj


instance (priority := 100) [IsIso f] : Surjective f := ⟨f.homeomorph.surjective⟩


instance [Surjective f] [Surjective g] : Surjective (f ≫ g) := ⟨g.surjective.comp f.surjective⟩


lemma Surjective.of_comp [Surjective (f ≫ g)] : Surjective g where
  surj := Function.Surjective.of_comp (g := f.base) (f ≫ g).surjective


lemma Surjective.comp_iff [Surjective f] : Surjective (f ≫ g) ↔ Surjective g :=
  ⟨fun _ ↦ of_comp f g, fun _ ↦ inferInstance⟩


instance : MorphismProperty.RespectsIso @Surjective :=
  surjective_eq_topologically ▸ topologically_respectsIso _ (fun e ↦ e.surjective)
    (fun _ _ hf hg ↦ hg.comp hf)


instance surjective_isLocalAtTarget : IsLocalAtTarget @Surjective := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ AlgebraicGeometry.IsLocalAtTarget @AlgebraicGeometry.Surjective
  -/
  have : MorphismProperty.RespectsIso @Surjective := inferInstance
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    this : CategoryTheory.MorphismProperty.RespectsIso @AlgebraicGeometry.Surjective
    ⊢ AlgebraicGeometry.IsLocalAtTarget @AlgebraicGeometry.Surjective
  -/
  rw [surjective_eq_topologically] at this ⊢
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    this : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topolo …
    ⊢ AlgebraicGeometry.IsLocalAtTarget (AlgebraicGeometry.topologically fun {α β} …
  -/
  refine topologically_isLocalAtTarget _ (fun _ s _ _ h ↦ h.restrictPreimage s) ?_
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    this : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topolo …
    ⊢ ∀ {α β : Type u_1} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace β] …
  -/
  intro α β _ _ f ι U H _ hf x
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    this : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topolo …
    α β : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_1
    U : ι → TopologicalSpace.Opens β
    H : Eq (iSup U) Top.top
    x✝ : Continuous f
    hf : ∀ (i : ι), Function.Surjective ((U i).carrier.restrictPreimage f)
    x : β
    ⊢ Exists fun a => Eq (f a) x
  -/
  obtain ⟨i, hxi⟩ : ∃ i, x ∈ U i := by simpa using congr(x ∈ $H)
  /-
    case intro
    X Y Z : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    this : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topolo …
    α β : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_1
    U : ι → TopologicalSpace.Opens β
    H : Eq (iSup U) Top.top
    x✝ : Continuous f
    hf : ∀ (i : ι), Function.Surjective ((U i).carrier.restrictPreimage f)
    x : β
    i : ι
    hxi : Membership.mem (U i) x
    ⊢ Exists fun a => Eq (f a) x
  -/
  obtain ⟨⟨y, _⟩, hy⟩ := hf i ⟨x, hxi⟩
  /-
    case intro.intro.mk
    X Y Z : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    this : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topolo …
    α β : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_1
    U : ι → TopologicalSpace.Opens β
    H : Eq (iSup U) Top.top
    x✝ : Continuous f
    hf : ∀ (i : ι), Function.Surjective ((U i).carrier.restrictPreimage f)
    x : β
    i : ι
    hxi : Membership.mem (U i) x
    y : α
    property✝ : Membership.mem (Set.preimage f (U i).carrier) y
    hy : Eq ((U i).carrier.restrictPreimage f ⟨y, property✝⟩) ⟨x, hxi⟩
    ⊢ Exists fun a => Eq (f a) x
  -/
  exact ⟨y, congr(($hy).1)⟩
  /-
    🎉 no goals
  -/


@[simp]
lemma range_eq_univ [Surjective f] : Set.range f.base = Set.univ := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.Surjective f
    ⊢ Eq (Set.range ⇑f.base) Set.univ
  -/
  simpa [Set.range_eq_univ] using f.surjective
  /-
    🎉 no goals
  -/


lemma range_eq_range_of_surjective {S : Scheme.{u}} (f : X ⟶ S) (g : Y ⟶ S) (e : X ⟶ Y)
    [Surjective e] (hge : e ≫ g = f) : Set.range f.base = Set.range g.base := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    e : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.Surjective e
    hge : Eq (CategoryTheory.CategoryStruct.comp e g) f
    ⊢ Eq (Set.range ⇑f.base) (Set.range ⇑g.base)
  -/
  rw [← hge]
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    e : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.Surjective e
    hge : Eq (CategoryTheory.CategoryStruct.comp e g) f
    ⊢ Eq (Set.range ⇑(CategoryTheory.CategoryStruct.comp e g).base) (Set.range ⇑g. …
  -/
  simp [Set.range_comp]
  /-
    🎉 no goals
  -/


lemma mem_range_iff_of_surjective {S : Scheme.{u}} (f : X ⟶ S) (g : Y ⟶ S) (e : X ⟶ Y)
    [Surjective e] (hge : e ≫ g = f) (s : S) : s ∈ Set.range f.base ↔ s ∈ Set.range g.base := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    e : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.Surjective e
    hge : Eq (CategoryTheory.CategoryStruct.comp e g) f
    s : ↑↑S.toPresheafedSpace
    ⊢ Iff (Membership.mem (Set.range ⇑f.base) s) (Membership.mem (Set.range ⇑g.bas …
  -/
  rw [range_eq_range_of_surjective f g e hge]
  /-
    🎉 no goals
  -/

instance injective_isStableUnderComposition :
    MorphismProperty.IsStableUnderComposition (topologically (Function.Injective ·)) where
  comp_mem _ _ hf hg := hg.comp hf


instance : (topologically IsOpenMap).RespectsIso :=
  topologically_respectsIso _ (fun e ↦ e.isOpenMap) (fun _ _ hf hg ↦ hg.comp hf)


instance isOpenMap_isLocalAtTarget : IsLocalAtTarget (topologically IsOpenMap) :=
  topologically_isLocalAtTarget' _ fun _ _ _ hU _ ↦ isOpenMap_iff_isOpenMap_of_iSup_eq_top hU


instance : (topologically IsClosedMap).RespectsIso :=
  topologically_respectsIso _ (fun e ↦ e.isClosedMap) (fun _ _ hf hg ↦ hg.comp hf)


instance isClosedMap_isLocalAtTarget : IsLocalAtTarget (topologically IsClosedMap) :=
  topologically_isLocalAtTarget' _ fun _ _ _ hU _ ↦ isClosedMap_iff_isClosedMap_of_iSup_eq_top hU


instance : (topologically IsEmbedding).RespectsIso :=
  topologically_respectsIso _ (fun e ↦ e.isEmbedding) (fun _ _ hf hg ↦ hg.comp hf)


instance isEmbedding_isLocalAtTarget : IsLocalAtTarget (topologically IsEmbedding) :=
  topologically_isLocalAtTarget' _ fun _ _ _ ↦ isEmbedding_iff_of_iSup_eq_top


instance : (topologically IsOpenEmbedding).RespectsIso :=
  topologically_respectsIso _ (fun e ↦ e.isOpenEmbedding) (fun _ _ hf hg ↦ hg.comp hf)


instance isOpenEmbedding_isLocalAtTarget : IsLocalAtTarget (topologically IsOpenEmbedding) :=
  topologically_isLocalAtTarget' _ fun _ _ _ ↦ isOpenEmbedding_iff_isOpenEmbedding_of_iSup_eq_top


instance : (topologically IsClosedEmbedding).RespectsIso :=
  topologically_respectsIso _ (fun e ↦ e.isClosedEmbedding) (fun _ _ hf hg ↦ hg.comp hf)


instance isClosedEmbedding_isLocalAtTarget : IsLocalAtTarget (topologically IsClosedEmbedding) :=
  topologically_isLocalAtTarget' _
    fun _ _ _ ↦ isClosedEmbedding_iff_isClosedEmbedding_of_iSup_eq_top


/-- A morphism of schemes is dominant if the underlying map has dense range. -/
@[mk_iff]
class IsDominant : Prop where
  denseRange : DenseRange f.base


lemma dominant_eq_topologically :
                                                 /-
                                                   ⊢ Eq (@AlgebraicGeometry.IsDominant) (AlgebraicGeometry.topologically fun {α β …
                                                 -/
    @IsDominant = topologically DenseRange := by ext; exact isDominant_iff _
                                                      /-
                                                        🎉 no goals
                                                      -/


lemma Scheme.Hom.denseRange (f : X.Hom Y) [IsDominant f] : DenseRange f.base :=
  IsDominant.denseRange


instance (priority := 100) [Surjective f] : IsDominant f := ⟨f.surjective.denseRange⟩


instance [IsDominant f] [IsDominant g] : IsDominant (f ≫ g) :=
  ⟨g.denseRange.comp f.denseRange g.base.2⟩


instance : MorphismProperty.IsMultiplicative @IsDominant where
  id_mem := fun _ ↦ inferInstance
  comp_mem := fun _ _ _ _ ↦ inferInstance


lemma IsDominant.of_comp [H : IsDominant (f ≫ g)] : IsDominant g := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsDominant (CategoryTheory.CategoryStruct.comp f g)
    ⊢ AlgebraicGeometry.IsDominant g
  -/
  rw [isDominant_iff, denseRange_iff_closure_range, ← Set.univ_subset_iff] at H ⊢
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    H : HasSubset.Subset Set.univ (closure (Set.range ⇑(CategoryTheory.CategoryStr …
    ⊢ HasSubset.Subset Set.univ (closure (Set.range ⇑g.base))
  -/
  exact H.trans (closure_mono (Set.range_comp_subset_range f.base g.base))
  /-
    🎉 no goals
  -/


lemma IsDominant.comp_iff [IsDominant f] : IsDominant (f ≫ g) ↔ IsDominant g :=
  ⟨fun _ ↦ of_comp f g, fun _ ↦ inferInstance⟩


instance IsDominant.respectsIso : MorphismProperty.RespectsIso @IsDominant :=
  MorphismProperty.respectsIso_of_isStableUnderComposition fun _ _ f (_ : IsIso f) ↦ inferInstance


instance IsDominant.isLocalAtTarget : IsLocalAtTarget @IsDominant :=
  have : MorphismProperty.RespectsIso (topologically DenseRange) :=
    dominant_eq_topologically ▸ IsDominant.respectsIso
  dominant_eq_topologically ▸ topologically_isLocalAtTarget' DenseRange
    fun _ _ _ hU _ ↦ denseRange_iff_denseRange_of_iSup_eq_top hU


lemma surjective_of_isDominant_of_isClosed_range (f : X ⟶ Y) [IsDominant f]
    (hf : IsClosed (Set.range f.base)) :
    Surjective f :=
      /-
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        inst✝ : AlgebraicGeometry.IsDominant f
        hf : IsClosed (Set.range ⇑f.base)
        ⊢ Function.Surjective ⇑f.base
      -/
  ⟨by rw [← Set.range_eq_univ, ← hf.closure_eq, f.denseRange.closure_range]⟩
      /-
        🎉 no goals
      -/


lemma IsDominant.of_comp_of_isOpenImmersion
    (f : X ⟶ Y) (g : Y ⟶ Z) [H : IsDominant (f ≫ g)] [IsOpenImmersion g] :
    IsDominant f := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsDominant (CategoryTheory.CategoryStruct.comp f g)
    inst✝ : AlgebraicGeometry.IsOpenImmersion g
    ⊢ AlgebraicGeometry.IsDominant f
  -/
  rw [isDominant_iff, DenseRange] at H ⊢
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    H : Dense (Set.range ⇑(CategoryTheory.CategoryStruct.comp f g).base)
    inst✝ : AlgebraicGeometry.IsOpenImmersion g
    ⊢ Dense (Set.range ⇑f.base)
  -/
  simp only [Scheme.comp_coeBase, TopCat.coe_comp, Set.range_comp] at H
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.IsOpenImmersion g
    H : Dense (Set.image (⇑g.base) (Set.range ⇑f.base))
    ⊢ Dense (Set.range ⇑f.base)
  -/
  convert H.preimage g.isOpenEmbedding.isOpenMap using 1
  /-
    case h.e'_3
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.IsOpenImmersion g
    H : Dense (Set.image (⇑g.base) (Set.range ⇑f.base))
    ⊢ Eq (Set.range ⇑f.base) (Set.preimage (⇑g.base) (Set.image (⇑g.base) (Set.ran …
  -/
  rw [Set.preimage_image_eq _ g.isOpenEmbedding.injective]
  /-
    🎉 no goals
  -/


instance specializingMap_respectsIso : (topologically @SpecializingMap).RespectsIso := by
  /-
    ⊢ (AlgebraicGeometry.topologically @SpecializingMap).RespectsIso
  -/
  apply topologically_respectsIso
    /-
      case hP₁
      ⊢ ∀ {α β : Type u_1} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace β] …
    -/
  · introv
    /-
      case hP₁
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : Homeomorph α β
      ⊢ SpecializingMap ⇑f
    -/
    exact f.isClosedMap.specializingMap
    /-
      🎉 no goals
    -/
    /-
      case hP₂
      ⊢ ∀ {α β γ : Type u_1} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
    -/
  · introv hf hg
    /-
      case hP₂
      α β γ : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace γ
      f : α → β
      g : β → γ
      hf : SpecializingMap f
      hg : SpecializingMap g
      ⊢ SpecializingMap (Function.comp g f)
    -/
    exact hf.comp hg
    /-
      🎉 no goals
    -/


instance specializingMap_isLocalAtTarget : IsLocalAtTarget (topologically @SpecializingMap) := by
  /-
    ⊢ AlgebraicGeometry.IsLocalAtTarget (AlgebraicGeometry.topologically @Speciali …
  -/
  apply topologically_isLocalAtTarget
    /-
      case hP₂
      ⊢ ∀ {α β : Type u_1} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace β] …
    -/
  · introv _ _ hf
    /-
      case hP₂
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set β
      x✝¹ : Continuous f
      x✝ : IsOpen s
      hf : SpecializingMap f
      ⊢ SpecializingMap (s.restrictPreimage f)
    -/
    rw [specializingMap_iff_closure_singleton_subset] at hf ⊢
    /-
      case hP₂
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set β
      x✝¹ : Continuous f
      x✝ : IsOpen s
      hf : ∀ (x : α), HasSubset.Subset (closure (Singleton.singleton (f x))) (Set.im …
      ⊢ ∀ (x : ↑(Set.preimage f s)), HasSubset.Subset (closure (Singleton.singleton  …
    -/
    intro ⟨x, hx⟩ ⟨y, hy⟩ hcl
    /-
      case hP₂
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set β
      x✝¹ : Continuous f
      x✝ : IsOpen s
      hf : ∀ (x : α), HasSubset.Subset (closure (Singleton.singleton (f x))) (Set.im …
      x : α
      hx : Membership.mem (Set.preimage f s) x
      y : β
      hy : Membership.mem s y
      hcl : Membership.mem (closure (Singleton.singleton (s.restrictPreimage f ⟨x, h …
      ⊢ Membership.mem (Set.image (s.restrictPreimage f) (closure (Singleton.singlet …
    -/
    simp only [closure_subtype, Set.restrictPreimage_mk, Set.image_singleton] at hcl
    /-
      case hP₂
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set β
      x✝¹ : Continuous f
      x✝ : IsOpen s
      hf : ∀ (x : α), HasSubset.Subset (closure (Singleton.singleton (f x))) (Set.im …
      x : α
      hx : Membership.mem (Set.preimage f s) x
      y : β
      hy : Membership.mem s y
      hcl : Membership.mem (closure (Singleton.singleton (f x))) y
      ⊢ Membership.mem (Set.image (s.restrictPreimage f) (closure (Singleton.singlet …
    -/
    obtain ⟨a, ha, hay⟩ := hf x hcl
    /-
      case hP₂.intro.intro
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set β
      x✝¹ : Continuous f
      x✝ : IsOpen s
      hf : ∀ (x : α), HasSubset.Subset (closure (Singleton.singleton (f x))) (Set.im …
      x : α
      hx : Membership.mem (Set.preimage f s) x
      y : β
      hy : Membership.mem s y
      hcl : Membership.mem (closure (Singleton.singleton (f x))) y
      a : α
      ha : Membership.mem (closure (Singleton.singleton x)) a
      hay : Eq (f a) y
      ⊢ Membership.mem (Set.image (s.restrictPreimage f) (closure (Singleton.singlet …
    -/
    rw [← specializes_iff_mem_closure] at hcl
    /-
      case hP₂.intro.intro
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set β
      x✝¹ : Continuous f
      x✝ : IsOpen s
      hf : ∀ (x : α), HasSubset.Subset (closure (Singleton.singleton (f x))) (Set.im …
      x : α
      hx : Membership.mem (Set.preimage f s) x
      y : β
      hy : Membership.mem s y
      hcl : Specializes (f x) y
      a : α
      ha : Membership.mem (closure (Singleton.singleton x)) a
      hay : Eq (f a) y
      ⊢ Membership.mem (Set.image (s.restrictPreimage f) (closure (Singleton.singlet …
    -/
    exact ⟨⟨a, by simp [hay, hy]⟩, by simpa [closure_subtype], by simpa⟩
    /-
      🎉 no goals
    -/
    /-
      case hP₃
      ⊢ ∀ {α β : Type u_1} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace β] …
    -/
  · introv hU _ hsp
    /-
      case hP₃
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_1
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      x✝ : Continuous f
      hsp : ∀ (i : ι), SpecializingMap ((U i).carrier.restrictPreimage f)
      ⊢ SpecializingMap f
    -/
    simp_rw [specializingMap_iff_closure_singleton_subset] at hsp ⊢
    /-
      case hP₃
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_1
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      x✝ : Continuous f
      hsp : ∀ (i : ι) (x : ↑(Set.preimage f (U i).carrier)), HasSubset.Subset (closu …
      ⊢ ∀ (x : α), HasSubset.Subset (closure (Singleton.singleton (f x))) (Set.image …
    -/
    intro x y hy
    /-
      case hP₃
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_1
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      x✝ : Continuous f
      hsp : ∀ (i : ι) (x : ↑(Set.preimage f (U i).carrier)), HasSubset.Subset (closu …
      x : α
      y : β
      hy : Membership.mem (closure (Singleton.singleton (f x))) y
      ⊢ Membership.mem (Set.image f (closure (Singleton.singleton x))) y
    -/
    have : ∃ i, y ∈ U i := Opens.mem_iSup.mp (hU ▸ Opens.mem_top _)
    /-
      case hP₃
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_1
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      x✝ : Continuous f
      hsp : ∀ (i : ι) (x : ↑(Set.preimage f (U i).carrier)), HasSubset.Subset (closu …
      x : α
      y : β
      hy : Membership.mem (closure (Singleton.singleton (f x))) y
      this : Exists fun i => Membership.mem (U i) y
      ⊢ Membership.mem (Set.image f (closure (Singleton.singleton x))) y
    -/
    obtain ⟨i, hi⟩ := this
    /-
      case hP₃.intro
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_1
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      x✝ : Continuous f
      hsp : ∀ (i : ι) (x : ↑(Set.preimage f (U i).carrier)), HasSubset.Subset (closu …
      x : α
      y : β
      hy : Membership.mem (closure (Singleton.singleton (f x))) y
      i : ι
      hi : Membership.mem (U i) y
      ⊢ Membership.mem (Set.image f (closure (Singleton.singleton x))) y
    -/
    rw [← specializes_iff_mem_closure] at hy
    /-
      case hP₃.intro
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_1
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      x✝ : Continuous f
      hsp : ∀ (i : ι) (x : ↑(Set.preimage f (U i).carrier)), HasSubset.Subset (closu …
      x : α
      y : β
      hy : Specializes (f x) y
      i : ι
      hi : Membership.mem (U i) y
      ⊢ Membership.mem (Set.image f (closure (Singleton.singleton x))) y
    -/
    have hfx : f x ∈ U i := (U i).2.stableUnderGeneralization hy hi
    have hy : (⟨y, hi⟩ : U i) ∈ closure {⟨f x, hfx⟩} := by
      simp only [closure_subtype, Set.image_singleton]
      rwa [← specializes_iff_mem_closure]
    /-
      case hP₃.intro
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_1
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      x✝ : Continuous f
      hsp : ∀ (i : ι) (x : ↑(Set.preimage f (U i).carrier)), HasSubset.Subset (closu …
      x : α
      y : β
      hy✝ : Specializes (f x) y
      i : ι
      hi : Membership.mem (U i) y
      hfx : Membership.mem (U i) (f x)
      hy : Membership.mem (closure (Singleton.singleton ⟨f x, hfx⟩)) ⟨y, hi⟩
      ⊢ Membership.mem (Set.image f (closure (Singleton.singleton x))) y
    -/
    obtain ⟨a, ha, hay⟩ := hsp i ⟨x, hfx⟩ hy
    /-
      case hP₃.intro.intro.intro
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_1
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      x✝ : Continuous f
      hsp : ∀ (i : ι) (x : ↑(Set.preimage f (U i).carrier)), HasSubset.Subset (closu …
      x : α
      y : β
      hy✝ : Specializes (f x) y
      i : ι
      hi : Membership.mem (U i) y
      hfx : Membership.mem (U i) (f x)
      hy : Membership.mem (closure (Singleton.singleton ⟨f x, hfx⟩)) ⟨y, hi⟩
      a : ↑(Set.preimage f (U i).carrier)
      ha : Membership.mem (closure (Singleton.singleton ⟨x, hfx⟩)) a
      hay : Eq ((U i).carrier.restrictPreimage f a) ⟨y, hi⟩
      ⊢ Membership.mem (Set.image f (closure (Singleton.singleton x))) y
    -/
    rw [closure_subtype] at ha
    /-
      case hP₃.intro.intro.intro
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_1
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      x✝ : Continuous f
      hsp : ∀ (i : ι) (x : ↑(Set.preimage f (U i).carrier)), HasSubset.Subset (closu …
      x : α
      y : β
      hy✝ : Specializes (f x) y
      i : ι
      hi : Membership.mem (U i) y
      hfx : Membership.mem (U i) (f x)
      hy : Membership.mem (closure (Singleton.singleton ⟨f x, hfx⟩)) ⟨y, hi⟩
      a : ↑(Set.preimage f (U i).carrier)
      ha : Membership.mem (closure (Set.image Subtype.val (Singleton.singleton ⟨x, h …
      hay : Eq ((U i).carrier.restrictPreimage f a) ⟨y, hi⟩
      ⊢ Membership.mem (Set.image f (closure (Singleton.singleton x))) y
    -/
    simp only [Opens.carrier_eq_coe, Set.image_singleton] at ha
    /-
      case hP₃.intro.intro.intro
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_1
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      x✝ : Continuous f
      hsp : ∀ (i : ι) (x : ↑(Set.preimage f (U i).carrier)), HasSubset.Subset (closu …
      x : α
      y : β
      hy✝ : Specializes (f x) y
      i : ι
      hi : Membership.mem (U i) y
      hfx : Membership.mem (U i) (f x)
      hy : Membership.mem (closure (Singleton.singleton ⟨f x, hfx⟩)) ⟨y, hi⟩
      a : ↑(Set.preimage f (U i).carrier)
      hay : Eq ((U i).carrier.restrictPreimage f a) ⟨y, hi⟩
      ha : Membership.mem (closure (Singleton.singleton x)) ↑a
      ⊢ Membership.mem (Set.image f (closure (Singleton.singleton x))) y
    -/
    apply_fun Subtype.val at hay
    /-
      case hP₃.intro.intro.intro
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_1
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      x✝ : Continuous f
      hsp : ∀ (i : ι) (x : ↑(Set.preimage f (U i).carrier)), HasSubset.Subset (closu …
      x : α
      y : β
      hy✝ : Specializes (f x) y
      i : ι
      hi : Membership.mem (U i) y
      hfx : Membership.mem (U i) (f x)
      hy : Membership.mem (closure (Singleton.singleton ⟨f x, hfx⟩)) ⟨y, hi⟩
      a : ↑(Set.preimage f (U i).carrier)
      ha : Membership.mem (closure (Singleton.singleton x)) ↑a
      hay : Eq ↑((U i).carrier.restrictPreimage f a) ↑⟨y, hi⟩
      ⊢ Membership.mem (Set.image f (closure (Singleton.singleton x))) y
    -/
    simp only [Opens.carrier_eq_coe, Set.restrictPreimage_coe] at hay
    /-
      case hP₃.intro.intro.intro
      α β : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_1
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      x✝ : Continuous f
      hsp : ∀ (i : ι) (x : ↑(Set.preimage f (U i).carrier)), HasSubset.Subset (closu …
      x : α
      y : β
      hy✝ : Specializes (f x) y
      i : ι
      hi : Membership.mem (U i) y
      hfx : Membership.mem (U i) (f x)
      hy : Membership.mem (closure (Singleton.singleton ⟨f x, hfx⟩)) ⟨y, hi⟩
      a : ↑(Set.preimage f (U i).carrier)
      ha : Membership.mem (closure (Singleton.singleton x)) ↑a
      hay : Eq (f ↑a) y
      ⊢ Membership.mem (Set.image f (closure (Singleton.singleton x))) y
    -/
    use a.val, ha, hay
    /-
      🎉 no goals
    -/


