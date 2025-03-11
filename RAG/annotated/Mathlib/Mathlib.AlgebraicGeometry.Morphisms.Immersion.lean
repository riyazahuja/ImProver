/-- A morphism of schemes `f : X ⟶ Y` is an immersion if
1. the underlying map of topological spaces is an embedding
2. the range of the map is locally closed
3. the induced morphisms of stalks are all surjective. -/
@[mk_iff]
class IsImmersion (f : X ⟶ Y) extends IsPreimmersion f : Prop where
  isLocallyClosed_range : IsLocallyClosed (Set.range f.base)


lemma Scheme.Hom.isLocallyClosed_range (f : X.Hom Y) [IsImmersion f] :
    IsLocallyClosed (Set.range f.base) :=
  IsImmersion.isLocallyClosed_range


/--
Given an immersion `f : X ⟶ Y`, this is the biggest open set `U ⊆ Y` containing the image of `X`
such that `X` is closed in `U`.
-/
def Scheme.Hom.coborderRange (f : X.Hom Y) [IsImmersion f] : Y.Opens :=
  ⟨coborder (Set.range f.base), f.isLocallyClosed_range.isOpen_coborder⟩


/--
The first part of the factorization of an immersion `f : X ⟶ Y` to a closed immersion
`f.liftCoborder : X ⟶ f.coborderRange` and a dominant open immersion `f.coborderRange.ι`.
-/
noncomputable
def Scheme.Hom.liftCoborder (f : X.Hom Y) [IsImmersion f] : X ⟶ f.coborderRange :=
                                               /-
                                                 X Y : AlgebraicGeometry.Scheme
                                                 f✝ : Quiver.Hom X Y
                                                 f : X.Hom Y
                                                 inst✝ : AlgebraicGeometry.IsImmersion f
                                                 ⊢ HasSubset.Subset (Set.range ⇑f.base) (Set.range ⇑f.coborderRange.ι.base)
                                               -/
  IsOpenImmersion.lift f.coborderRange.ι f (by simpa using subset_coborder)
                                               /-
                                                 🎉 no goals
                                               -/


/--
Any (locally-closed) immersion can be factored into
a closed immersion followed by a (dominant) open immersion.
-/
@[reassoc (attr := simp)]
lemma Scheme.Hom.liftCoborder_ι (f : X.Hom Y) [IsImmersion f] :
    f.liftCoborder ≫ f.coborderRange.ι = f :=
  IsOpenImmersion.lift_fac _ _ _


instance [IsImmersion f] : IsClosedImmersion f.liftCoborder := by
  have : IsPreimmersion (f.liftCoborder ≫ f.coborderRange.ι) := by
    simp only [Scheme.Hom.liftCoborder_ι]; infer_instance
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsImmersion f
    this : AlgebraicGeometry.IsPreimmersion (CategoryTheory.CategoryStruct.comp (A …
    ⊢ AlgebraicGeometry.IsClosedImmersion (AlgebraicGeometry.Scheme.Hom.liftCobord …
  -/
  have : IsPreimmersion f.liftCoborder := .of_comp f.liftCoborder f.coborderRange.ι
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsImmersion f
    this✝ : AlgebraicGeometry.IsPreimmersion (CategoryTheory.CategoryStruct.comp ( …
    this : AlgebraicGeometry.IsPreimmersion (AlgebraicGeometry.Scheme.Hom.liftCobo …
    ⊢ AlgebraicGeometry.IsClosedImmersion (AlgebraicGeometry.Scheme.Hom.liftCobord …
  -/
  refine .of_isPreimmersion _ ?_
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsImmersion f
    this✝ : AlgebraicGeometry.IsPreimmersion (CategoryTheory.CategoryStruct.comp ( …
    this : AlgebraicGeometry.IsPreimmersion (AlgebraicGeometry.Scheme.Hom.liftCobo …
    ⊢ IsClosed (Set.range ⇑(AlgebraicGeometry.Scheme.Hom.liftCoborder f).base)
  -/
  convert isClosed_preimage_val_coborder
  /-
    case h.e'_3
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsImmersion f
    this✝ : AlgebraicGeometry.IsPreimmersion (CategoryTheory.CategoryStruct.comp ( …
    this : AlgebraicGeometry.IsPreimmersion (AlgebraicGeometry.Scheme.Hom.liftCobo …
    ⊢ Eq (Set.range ⇑(AlgebraicGeometry.Scheme.Hom.liftCoborder f).base) (Set.prei …
  -/
  apply Set.image_injective.mpr f.coborderRange.ι.isEmbedding.injective
  /-
    case h.e'_3.a
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsImmersion f
    this✝ : AlgebraicGeometry.IsPreimmersion (CategoryTheory.CategoryStruct.comp ( …
    this : AlgebraicGeometry.IsPreimmersion (AlgebraicGeometry.Scheme.Hom.liftCobo …
    ⊢ Eq (Set.image (⇑(AlgebraicGeometry.Scheme.Hom.coborderRange f).ι.base) (Set. …
  -/
  rw [← Set.range_comp, ← TopCat.coe_comp, ← Scheme.comp_base, f.liftCoborder_ι]
  /-
    case h.e'_3.a
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsImmersion f
    this✝ : AlgebraicGeometry.IsPreimmersion (CategoryTheory.CategoryStruct.comp ( …
    this : AlgebraicGeometry.IsPreimmersion (AlgebraicGeometry.Scheme.Hom.liftCobo …
    ⊢ Eq (Set.range ⇑f.base) (Set.image (⇑(AlgebraicGeometry.Scheme.Hom.coborderRa …
  -/
  exact (Set.image_preimage_eq_of_subset (by simpa using subset_coborder)).symm
  /-
    🎉 no goals
  -/


instance [IsImmersion f] : IsDominant f.coborderRange.ι := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsImmersion f
    ⊢ AlgebraicGeometry.IsDominant (AlgebraicGeometry.Scheme.Hom.coborderRange f).ι
  -/
  rw [isDominant_iff, DenseRange, Scheme.Opens.range_ι]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsImmersion f
    ⊢ Dense ↑(AlgebraicGeometry.Scheme.Hom.coborderRange f)
  -/
  exact dense_coborder
  /-
    🎉 no goals
  -/


lemma isImmersion_eq_inf : @IsImmersion = (@IsPreimmersion ⊓
    topologically fun {_ _} _ _ f ↦ IsLocallyClosed (Set.range f) : MorphismProperty Scheme) := by
  /-
    ⊢ Eq (@AlgebraicGeometry.IsImmersion) (Min.min (@AlgebraicGeometry.IsPreimmers …
  -/
  ext; exact isImmersion_iff _
       /-
         🎉 no goals
       -/


instance : IsLocalAtTarget @IsImmersion := by
  suffices IsLocalAtTarget (topologically fun {X Y} _ _ f ↦ IsLocallyClosed (Set.range f)) from
    isImmersion_eq_inf ▸ inferInstance
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ AlgebraicGeometry.IsLocalAtTarget (AlgebraicGeometry.topologically fun {X Y} …
  -/
  apply (config := { allowSynthFailures := true }) topologically_isLocalAtTarget'
    /-
      case inst
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topological …
    -/
  · refine { precomp := ?_, postcomp := ?_ }
      /-
        case inst.refine_1
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        ⊢ ∀ {X Y Z : AlgebraicGeometry.Scheme} (i : Quiver.Hom X Y), CategoryTheory.Mo …
      -/
    · intro X Y Z i hi f hf
      /-
        case inst.refine_1
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        X Y Z : AlgebraicGeometry.Scheme
        i : Quiver.Hom X Y
        hi : CategoryTheory.MorphismProperty.isomorphisms AlgebraicGeometry.Scheme i
        f : Quiver.Hom Y Z
        hf : AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topologi …
        ⊢ AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topological …
      -/
      replace hi : IsIso i := hi
      /-
        case inst.refine_1
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        X Y Z : AlgebraicGeometry.Scheme
        i : Quiver.Hom X Y
        f : Quiver.Hom Y Z
        hf : AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topologi …
        hi : CategoryTheory.IsIso i
        ⊢ AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topological …
      -/
      show IsLocallyClosed _
      simpa only [Scheme.comp_coeBase, TopCat.coe_comp, Set.range_comp,
        Set.range_eq_univ.mpr i.surjective, Set.image_univ]
      /-
        case inst.refine_2
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        ⊢ ∀ {X Y Z : AlgebraicGeometry.Scheme} (i : Quiver.Hom Y Z), CategoryTheory.Mo …
      -/
    · intro X Y Z i hi f hf
      /-
        case inst.refine_2
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        X Y Z : AlgebraicGeometry.Scheme
        i : Quiver.Hom Y Z
        hi : CategoryTheory.MorphismProperty.isomorphisms AlgebraicGeometry.Scheme i
        f : Quiver.Hom X Y
        hf : AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topologi …
        ⊢ AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topological …
      -/
      replace hi : IsIso i := hi
      /-
        case inst.refine_2
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        X Y Z : AlgebraicGeometry.Scheme
        i : Quiver.Hom Y Z
        f : Quiver.Hom X Y
        hf : AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topologi …
        hi : CategoryTheory.IsIso i
        ⊢ AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topological …
      -/
      show IsLocallyClosed _
      /-
        case inst.refine_2
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        X Y Z : AlgebraicGeometry.Scheme
        i : Quiver.Hom Y Z
        f : Quiver.Hom X Y
        hf : AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topologi …
        hi : CategoryTheory.IsIso i
        ⊢ IsLocallyClosed (Set.range ⇑(CategoryTheory.CategoryStruct.comp f i).base)
      -/
      simp only [Scheme.comp_coeBase, TopCat.coe_comp, Set.range_comp]
      /-
        case inst.refine_2
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        X Y Z : AlgebraicGeometry.Scheme
        i : Quiver.Hom Y Z
        f : Quiver.Hom X Y
        hf : AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topologi …
        hi : CategoryTheory.IsIso i
        ⊢ IsLocallyClosed (Set.image (⇑i.base) (Set.range ⇑f.base))
      -/
      refine hf.image i.homeomorph.isInducing ?_
      /-
        case inst.refine_2
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        X Y Z : AlgebraicGeometry.Scheme
        i : Quiver.Hom Y Z
        f : Quiver.Hom X Y
        hf : AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topologi …
        hi : CategoryTheory.IsIso i
        ⊢ IsLocallyClosed (Set.range ⇑i.base)
      -/
      rw [Set.range_eq_univ.mpr i.surjective]
      /-
        case inst.refine_2
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        X Y Z : AlgebraicGeometry.Scheme
        i : Quiver.Hom Y Z
        f : Quiver.Hom X Y
        hf : AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topologi …
        hi : CategoryTheory.IsIso i
        ⊢ IsLocallyClosed Set.univ
      -/
      exact isOpen_univ.isLocallyClosed
      /-
        🎉 no goals
      -/
    /-
      case hP
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ ∀ {α β : Type u_1} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace β] …
    -/
  · simp_rw [Set.range_restrictPreimage]
    /-
      case hP
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ ∀ {α β : Type u_1} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace β] …
    -/
    exact fun _ _ _ e _ ↦ isLocallyClosed_iff_coe_preimage_of_iSup_eq_top e _
    /-
      🎉 no goals
    -/


instance (priority := 900) {X Y : Scheme} (f : X ⟶ Y) [IsOpenImmersion f] : IsImmersion f where
  isLocallyClosed_range := f.isOpenEmbedding.2.isLocallyClosed


instance (priority := 900) {X Y : Scheme} (f : X ⟶ Y) [IsClosedImmersion f] : IsImmersion f where
  isLocallyClosed_range := f.isClosedEmbedding.2.isLocallyClosed


instance : MorphismProperty.IsMultiplicative @IsImmersion where
  id_mem _ := inferInstance
  comp_mem {X Y Z} f g hf hg := by
    /-
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : AlgebraicGeometry.IsImmersion f
      hg : AlgebraicGeometry.IsImmersion g
      ⊢ AlgebraicGeometry.IsImmersion (CategoryTheory.CategoryStruct.comp f g)
    -/
    refine { __ := inferInstanceAs (IsPreimmersion (f ≫ g)), isLocallyClosed_range := ?_ }
    /-
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : AlgebraicGeometry.IsImmersion f
      hg : AlgebraicGeometry.IsImmersion g
      ⊢ IsLocallyClosed (Set.range ⇑(CategoryTheory.CategoryStruct.comp f g).base)
    -/
    simp only [Scheme.comp_coeBase, TopCat.coe_comp, Set.range_comp]
    /-
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : AlgebraicGeometry.IsImmersion f
      hg : AlgebraicGeometry.IsImmersion g
      ⊢ IsLocallyClosed (Set.image (⇑g.base) (Set.range ⇑f.base))
    -/
    exact f.isLocallyClosed_range.image g.isEmbedding.isInducing g.isLocallyClosed_range
    /-
      🎉 no goals
    -/


instance comp {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) [IsImmersion f]
    [IsImmersion g] : IsImmersion (f ≫ g) :=
  MorphismProperty.IsStableUnderComposition.comp_mem f g inferInstance inferInstance


variable {f} in
/--
A morphism is a (locally-closed) immersion if and only if it can be factored into
a closed immersion followed by an open immersion.
-/
lemma isImmersion_iff_exists : IsImmersion f ↔ ∃ (Z : Scheme) (g₁ : X ⟶ Z) (g₂ : Z ⟶ Y),
    IsClosedImmersion g₁ ∧ IsOpenImmersion g₂ ∧ g₁ ≫ g₂ = f :=
  ⟨fun _ ↦ ⟨_, f.liftCoborder, f.coborderRange.ι, inferInstance, inferInstance, f.liftCoborder_ι⟩,
    fun ⟨_, _, _, _, _, e⟩ ↦ e ▸ inferInstance⟩


theorem of_comp {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) [IsImmersion g]
    [IsImmersion (f ≫ g)] : IsImmersion f where
  __ := IsPreimmersion.of_comp f g
  isLocallyClosed_range := by
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.IsImmersion g
      inst✝ : AlgebraicGeometry.IsImmersion (CategoryTheory.CategoryStruct.comp f g)
      ⊢ IsLocallyClosed (Set.range ⇑f.base)
    -/
    rw [← Set.preimage_image_eq (Set.range _) g.isEmbedding.injective]
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.IsImmersion g
      inst✝ : AlgebraicGeometry.IsImmersion (CategoryTheory.CategoryStruct.comp f g)
      ⊢ IsLocallyClosed (Set.preimage (⇑g.base) (Set.image (⇑g.base) (Set.range ⇑f.b …
    -/
    have := (f ≫ g).isLocallyClosed_range.preimage g.base.2
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.IsImmersion g
      inst✝ : AlgebraicGeometry.IsImmersion (CategoryTheory.CategoryStruct.comp f g)
      this : IsLocallyClosed (Set.preimage g.base.toFun (Set.range ⇑(CategoryTheory. …
      ⊢ IsLocallyClosed (Set.preimage (⇑g.base) (Set.image (⇑g.base) (Set.range ⇑f.b …
    -/
    simpa only [Scheme.comp_coeBase, TopCat.coe_comp, Set.range_comp] using this
    /-
      🎉 no goals
    -/


theorem comp_iff {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) [IsImmersion g] :
    IsImmersion (f ≫ g) ↔ IsImmersion f :=
  ⟨fun _ ↦ of_comp f g, fun _ ↦ inferInstance⟩


instance isStableUnderBaseChange : MorphismProperty.IsStableUnderBaseChange @IsImmersion where
  of_isPullback := by
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ ∀ {X Y Y' S : AlgebraicGeometry.Scheme} {f : Quiver.Hom X S} {g : Quiver.Hom …
    -/
    intros X Y Y' S f g f' g' H hg
    /-
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      X Y Y' S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      f' : Quiver.Hom Y' Y
      g' : Quiver.Hom Y' X
      H : CategoryTheory.IsPullback f' g' g f
      hg : AlgebraicGeometry.IsImmersion g
      ⊢ AlgebraicGeometry.IsImmersion g'
    -/
    let Z := Limits.pullback f g.coborderRange.ι
    /-
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      X Y Y' S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      f' : Quiver.Hom Y' Y
      g' : Quiver.Hom Y' X
      H : CategoryTheory.IsPullback f' g' g f
      hg : AlgebraicGeometry.IsImmersion g
      Z : AlgebraicGeometry.Scheme := CategoryTheory.Limits.pullback f (AlgebraicGeo …
      ⊢ AlgebraicGeometry.IsImmersion g'
    -/
    let e : Y' ⟶ Z := Limits.pullback.lift g' (f' ≫ g.liftCoborder) (by simpa using H.w.symm)
    have : IsClosedImmersion e := by
      have := (IsPullback.paste_horiz_iff (.of_hasPullback f g.coborderRange.ι)
        (show e ≫ Limits.pullback.snd _ _ = _ from Limits.pullback.lift_snd _ _ _)).mp ?_
      · exact MorphismProperty.of_isPullback this.flip inferInstance
      · simpa [e] using H.flip
    rw [← Limits.pullback.lift_fst (f := f) (g := g.coborderRange.ι) g' (f' ≫ g.liftCoborder)
      (by simpa using H.w.symm)]
    /-
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      X Y Y' S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      f' : Quiver.Hom Y' Y
      g' : Quiver.Hom Y' X
      H : CategoryTheory.IsPullback f' g' g f
      hg : AlgebraicGeometry.IsImmersion g
      Z : AlgebraicGeometry.Scheme := CategoryTheory.Limits.pullback f (AlgebraicGeo …
      e : Quiver.Hom Y' Z := CategoryTheory.Limits.pullback.lift g' (CategoryTheory. …
      this : AlgebraicGeometry.IsClosedImmersion e
      ⊢ AlgebraicGeometry.IsImmersion (CategoryTheory.CategoryStruct.comp (CategoryT …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


open Limits Scheme.Pullback in
/-- The diagonal morphism is always an immersion. -/
@[stacks 01KJ]
instance : IsImmersion (pullback.diagonal f) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ AlgebraicGeometry.IsImmersion (CategoryTheory.Limits.pullback.diagonal f)
  -/
  let 𝒰 := Y.affineCover
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover := Y.affineCover
    ⊢ AlgebraicGeometry.IsImmersion (CategoryTheory.Limits.pullback.diagonal f)
  -/
  let 𝒱 (i) := (pullback f (𝒰.map i)).affineCover
  have H : pullback.diagonal f ⁻¹ᵁ diagonalCoverDiagonalRange f 𝒰 𝒱 = ⊤ :=
    top_le_iff.mp fun _ _ ↦ range_diagonal_subset_diagonalCoverDiagonalRange _ _ _ ⟨_, rfl⟩
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover := Y.affineCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover := fun  …
    H : Eq ((TopologicalSpace.Opens.map (CategoryTheory.Limits.pullback.diagonal f …
    ⊢ AlgebraicGeometry.IsImmersion (CategoryTheory.Limits.pullback.diagonal f)
  -/
  have := isClosedImmersion_diagonal_restrict_diagonalCoverDiagonalRange f 𝒰 𝒱
  have : IsImmersion ((pullback.diagonal f ∣_
    diagonalCoverDiagonalRange f 𝒰 𝒱) ≫ Scheme.Opens.ι _) := inferInstance
  rwa [morphismRestrict_ι, H, ← Scheme.topIso_hom,
    MorphismProperty.cancel_left_of_respectsIso (P := @IsImmersion)] at this


