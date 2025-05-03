/-- The image of a bicone under a functor. -/
@[simps]
def mapBicone {f : J → C} (b : Bicone f) : Bicone (F.obj ∘ f) where
  pt := F.obj b.pt
  π j := F.map (b.π j)
  ι j := F.map (b.ι j)
  ι_π j j' := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
      F : CategoryTheory.Functor C D
      inst✝ : F.PreservesZeroMorphisms
      J : Type w₁
      f : J → C
      b : CategoryTheory.Limits.Bicone f
      j j' : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun j => F.map (b.ι j)) j) ((fun j  …
    -/
    rw [← F.map_comp]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
      F : CategoryTheory.Functor C D
      inst✝ : F.PreservesZeroMorphisms
      J : Type w₁
      f : J → C
      b : CategoryTheory.Limits.Bicone f
      j j' : J
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (b.ι j) (b.π j'))) (dite (Eq j …
    -/
    split_ifs with h
      /-
        case pos
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
        F : CategoryTheory.Functor C D
        inst✝ : F.PreservesZeroMorphisms
        J : Type w₁
        f : J → C
        b : CategoryTheory.Limits.Bicone f
        j j' : J
        h : Eq j j'
        ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (b.ι j) (b.π j'))) (CategoryTh …
      -/
    · subst h
      /-
        case pos
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
        F : CategoryTheory.Functor C D
        inst✝ : F.PreservesZeroMorphisms
        J : Type w₁
        f : J → C
        b : CategoryTheory.Limits.Bicone f
        j : J
        ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (b.ι j) (b.π j))) (CategoryThe …
      -/
      simp only [bicone_ι_π_self, CategoryTheory.Functor.map_id, eqToHom_refl]; dsimp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
      /-
        case neg
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
        F : CategoryTheory.Functor C D
        inst✝ : F.PreservesZeroMorphisms
        J : Type w₁
        f : J → C
        b : CategoryTheory.Limits.Bicone f
        j j' : J
        h : Not (Eq j j')
        ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (b.ι j) (b.π j'))) 0
      -/
    · rw [bicone_ι_π_ne _ h, F.map_zero]
      /-
        🎉 no goals
      -/


theorem mapBicone_whisker {K : Type w₂} {g : K ≃ J} {f : J → C} (c : Bicone f) :
    F.mapBicone (c.whisker g) = (F.mapBicone c).whisker g :=
  rfl


/-- The image of a binary bicone under a functor. -/
@[simps!]
def mapBinaryBicone {X Y : C} (b : BinaryBicone X Y) : BinaryBicone (F.obj X) (F.obj Y) :=
  (BinaryBicones.functoriality _ _ F).obj b


/-- A functor `F` preserves biproducts of `f` if `F` maps every bilimit bicone over `f` to a
    bilimit bicone over `F.obj ∘ f`. -/
class PreservesBiproduct (f : J → C) (F : C ⥤ D) [PreservesZeroMorphisms F] : Prop where
  preserves : ∀ {b : Bicone f}, b.IsBilimit → Nonempty (F.mapBicone b).IsBilimit


/-- A functor `F` preserves biproducts of `f` if `F` maps every bilimit bicone over `f` to a
    bilimit bicone over `F.obj ∘ f`. -/
def isBilimitOfPreserves {f : J → C} (F : C ⥤ D) [PreservesZeroMorphisms F] [PreservesBiproduct f F]
    {b : Bicone f} (hb : b.IsBilimit) : (F.mapBicone b).IsBilimit :=
  (PreservesBiproduct.preserves hb).some


/-- A functor `F` preserves biproducts of shape `J` if it preserves biproducts of `f` for every
    `f : J → C`. -/
class PreservesBiproductsOfShape (F : C ⥤ D) [PreservesZeroMorphisms F] : Prop where
  preserves : ∀ {f : J → C}, PreservesBiproduct f F


/-- A functor `F` preserves finite biproducts if it preserves biproducts of shape `J` whenever
    `J` is a fintype. -/
class PreservesFiniteBiproducts (F : C ⥤ D) [PreservesZeroMorphisms F] : Prop where
  preserves : ∀ {J : Type} [Fintype J], PreservesBiproductsOfShape J F


/-- A functor `F` preserves biproducts if it preserves biproducts of any shape `J` of size `w`.
    The usual notion of preservation of biproducts is recovered by choosing `w` to be the universe
    of the morphisms of `C`. -/
class PreservesBiproducts (F : C ⥤ D) [PreservesZeroMorphisms F] : Prop where
  preserves : ∀ {J : Type w₁}, PreservesBiproductsOfShape J F


/-- Preserving biproducts at a bigger universe level implies preserving biproducts at a
smaller universe level. -/
lemma preservesBiproducts_shrink (F : C ⥤ D) [PreservesZeroMorphisms F]
    [PreservesBiproducts.{max w₁ w₂} F] : PreservesBiproducts.{w₁} F :=
  ⟨fun {_} =>
    ⟨fun {_} =>
      ⟨fun {b} ib =>
        ⟨((F.mapBicone b).whiskerIsBilimitIff _).toFun
          (isBilimitOfPreserves F ((b.whiskerIsBilimitIff Equiv.ulift.{w₂}).invFun ib))⟩⟩⟩⟩


instance (priority := 100) preservesFiniteBiproductsOfPreservesBiproducts (F : C ⥤ D)
    [PreservesZeroMorphisms F] [PreservesBiproducts.{w₁} F] : PreservesFiniteBiproducts F where
                        /-
                          C : Type u₁
                          inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
                          D : Type u₂
                          inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
                          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                          inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
                          F : CategoryTheory.Functor C D
                          inst✝¹ : F.PreservesZeroMorphisms
                          inst✝ : CategoryTheory.Limits.PreservesBiproducts F
                          J : Type
                          x✝ : Fintype J
                          ⊢ CategoryTheory.Limits.PreservesBiproductsOfShape J F
                        -/
  preserves {J} _ := by letI := preservesBiproducts_shrink.{0} F; infer_instance
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- A functor `F` preserves binary biproducts of `X` and `Y` if `F` maps every bilimit bicone over
    `X` and `Y` to a bilimit bicone over `F.obj X` and `F.obj Y`. -/
class PreservesBinaryBiproduct (X Y : C) (F : C ⥤ D) [PreservesZeroMorphisms F] : Prop where
  preserves : ∀ {b : BinaryBicone X Y}, b.IsBilimit → Nonempty ((F.mapBinaryBicone b).IsBilimit)


/-- A functor `F` preserves binary biproducts of `X` and `Y` if `F` maps every bilimit bicone over
    `X` and `Y` to a bilimit bicone over `F.obj X` and `F.obj Y`. -/
def isBinaryBilimitOfPreserves {X Y : C} (F : C ⥤ D) [PreservesZeroMorphisms F]
    [PreservesBinaryBiproduct X Y F] {b : BinaryBicone X Y} (hb : b.IsBilimit) :
    (F.mapBinaryBicone b).IsBilimit :=
  (PreservesBinaryBiproduct.preserves hb).some


/-- A functor `F` preserves binary biproducts if it preserves the binary biproduct of `X` and `Y`
    for all `X` and `Y`. -/
class PreservesBinaryBiproducts (F : C ⥤ D) [PreservesZeroMorphisms F] : Prop where
  preserves : ∀ {X Y : C}, PreservesBinaryBiproduct X Y F := by infer_instance


/-- A functor that preserves biproducts of a pair preserves binary biproducts. -/
lemma preservesBinaryBiproduct_of_preservesBiproduct (F : C ⥤ D)
    [PreservesZeroMorphisms F] (X Y : C) [PreservesBiproduct (pairFunction X Y) F] :
    PreservesBinaryBiproduct X Y F where
  preserves {b} hb := ⟨{
      isLimit :=
        IsLimit.ofIsoLimit
            ((IsLimit.postcomposeHomEquiv (diagramIsoPair _) _).symm
              (isBilimitOfPreserves F (b.toBiconeIsBilimit.symm hb)).isLimit) <|
          Cones.ext (Iso.refl _) fun j => by
            /-
              C : Type u₁
              inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
              inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
              inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
              F : CategoryTheory.Functor C D
              inst✝¹ : F.PreservesZeroMorphisms
              X Y : C
              inst✝ : CategoryTheory.Limits.PreservesBiproduct (CategoryTheory.Limits.pairFu …
              b : CategoryTheory.Limits.BinaryBicone X Y
              hb : b.IsBilimit
              j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
              ⊢ Eq (((CategoryTheory.Limits.Cones.postcompose (CategoryTheory.Limits.diagram …
            -/
                                   /-
                                     🎉 no goals
                                   -/
            rcases j with ⟨⟨⟩⟩ <;> simp
                                   /-
                                     🎉 no goals
                                   -/
      isColimit :=
        IsColimit.ofIsoColimit
            ((IsColimit.precomposeInvEquiv (diagramIsoPair _) _).symm
              (isBilimitOfPreserves F (b.toBiconeIsBilimit.symm hb)).isColimit) <|
          Cocones.ext (Iso.refl _) fun j => by
            /-
              C : Type u₁
              inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
              inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
              inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
              F : CategoryTheory.Functor C D
              inst✝¹ : F.PreservesZeroMorphisms
              X Y : C
              inst✝ : CategoryTheory.Limits.PreservesBiproduct (CategoryTheory.Limits.pairFu …
              b : CategoryTheory.Limits.BinaryBicone X Y
              hb : b.IsBilimit
              j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.Cocones.prec …
            -/
                                   /-
                                     🎉 no goals
                                   -/
            rcases j with ⟨⟨⟩⟩ <;> simp }⟩
                                   /-
                                     🎉 no goals
                                   -/


/-- A functor that preserves biproducts of a pair preserves binary biproducts. -/
lemma preservesBinaryBiproducts_of_preservesBiproducts (F : C ⥤ D) [PreservesZeroMorphisms F]
    [PreservesBiproductsOfShape WalkingPair F] : PreservesBinaryBiproducts F where
  preserves {X} Y := preservesBinaryBiproduct_of_preservesBiproduct F X Y


/-- As for products, any functor between categories with biproducts gives rise to a morphism
    `F.obj (⨁ f) ⟶ ⨁ (F.obj ∘ f)`. -/
def biproductComparison : F.obj (⨁ f) ⟶ ⨁ F.obj ∘ f :=
  biproduct.lift fun j => F.map (biproduct.π f j)


@[reassoc (attr := simp)]
theorem biproductComparison_π (j : J) :
    biproductComparison F f ≫ biproduct.π _ j = F.map (biproduct.π f j) :=
  biproduct.lift_π _ _


/-- As for coproducts, any functor between categories with biproducts gives rise to a morphism
    `⨁ (F.obj ∘ f) ⟶ F.obj (⨁ f)` -/
def biproductComparison' : ⨁ F.obj ∘ f ⟶ F.obj (⨁ f) :=
  biproduct.desc fun j => F.map (biproduct.ι f j)


@[reassoc (attr := simp)]
theorem ι_biproductComparison' (j : J) :
    biproduct.ι _ j ≫ biproductComparison' F f = F.map (biproduct.ι f j) :=
  biproduct.ι_desc _ _


/-- The composition in the opposite direction is equal to the identity if and only if `F` preserves
    the biproduct, see `preservesBiproduct_of_monoBiproductComparison`. -/
@[reassoc (attr := simp)]
theorem biproductComparison'_comp_biproductComparison :
    biproductComparison' F f ≫ biproductComparison F f = 𝟙 (⨁ F.obj ∘ f) := by
  classical
    ext
    simp [biproduct.ι_π, ← Functor.map_comp, eqToHom_map]


/-- `biproduct_comparison F f` is a split epimorphism. -/
@[simps]
def splitEpiBiproductComparison : SplitEpi (biproductComparison F f) where
  section_ := biproductComparison' F f
           /-
             C : Type u₁
             inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
             D : Type u₂
             inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
             inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
             inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
             J : Type w₁
             F : CategoryTheory.Functor C D
             f : J → C
             inst✝² : CategoryTheory.Limits.HasBiproduct f
             inst✝¹ : CategoryTheory.Limits.HasBiproduct (Function.comp F.obj f)
             inst✝ : F.PreservesZeroMorphisms
             ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.biproductComparison' f) (F.biprodu …
           -/
  id := by aesop
           /-
             🎉 no goals
           -/


instance : IsSplitEpi (biproductComparison F f) :=
  IsSplitEpi.mk' (splitEpiBiproductComparison F f)


/-- `biproduct_comparison' F f` is a split monomorphism. -/
@[simps]
def splitMonoBiproductComparison' : SplitMono (biproductComparison' F f) where
  retraction := biproductComparison F f
           /-
             C : Type u₁
             inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
             D : Type u₂
             inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
             inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
             inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
             J : Type w₁
             F : CategoryTheory.Functor C D
             f : J → C
             inst✝² : CategoryTheory.Limits.HasBiproduct f
             inst✝¹ : CategoryTheory.Limits.HasBiproduct (Function.comp F.obj f)
             inst✝ : F.PreservesZeroMorphisms
             ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.biproductComparison' f) (F.biprodu …
           -/
  id := by aesop
           /-
             🎉 no goals
           -/


instance : IsSplitMono (biproductComparison' F f) :=
  IsSplitMono.mk' (splitMonoBiproductComparison' F f)


instance hasBiproduct_of_preserves : HasBiproduct (F.obj ∘ f) :=
  HasBiproduct.mk
    { bicone := F.mapBicone (biproduct.bicone f)
      isBilimit := isBilimitOfPreserves _ (biproduct.isBilimit _) }


/-- If `F` preserves a biproduct, we get a definitionally nice isomorphism
    `F.obj (⨁ f) ≅ ⨁ (F.obj ∘ f)`. -/
@[simp]
def mapBiproduct : F.obj (⨁ f) ≅ ⨁ F.obj ∘ f :=
  biproduct.uniqueUpToIso _ (isBilimitOfPreserves _ (biproduct.isBilimit _))


theorem mapBiproduct_hom :
    haveI : HasBiproduct fun j => F.obj (f j) := hasBiproduct_of_preserves F f
    (mapBiproduct F f).hom = biproduct.lift fun j => F.map (biproduct.π f j) := rfl


theorem mapBiproduct_inv :
    haveI : HasBiproduct fun j => F.obj (f j) := hasBiproduct_of_preserves F f
    (mapBiproduct F f).inv = biproduct.desc fun j => F.map (biproduct.ι f j) := rfl


/-- As for products, any functor between categories with binary biproducts gives rise to a
    morphism `F.obj (X ⊞ Y) ⟶ F.obj X ⊞ F.obj Y`. -/
def biprodComparison : F.obj (X ⊞ Y) ⟶ F.obj X ⊞ F.obj Y :=
  biprod.lift (F.map biprod.fst) (F.map biprod.snd)


@[reassoc (attr := simp)]
theorem biprodComparison_fst : biprodComparison F X Y ≫ biprod.fst = F.map biprod.fst :=
  biprod.lift_fst _ _


@[reassoc (attr := simp)]
theorem biprodComparison_snd : biprodComparison F X Y ≫ biprod.snd = F.map biprod.snd :=
  biprod.lift_snd _ _


/-- As for coproducts, any functor between categories with binary biproducts gives rise to a
    morphism `F.obj X ⊞ F.obj Y ⟶ F.obj (X ⊞ Y)`. -/
def biprodComparison' : F.obj X ⊞ F.obj Y ⟶ F.obj (X ⊞ Y) :=
  biprod.desc (F.map biprod.inl) (F.map biprod.inr)


@[reassoc (attr := simp)]
theorem inl_biprodComparison' : biprod.inl ≫ biprodComparison' F X Y = F.map biprod.inl :=
  biprod.inl_desc _ _


@[reassoc (attr := simp)]
theorem inr_biprodComparison' : biprod.inr ≫ biprodComparison' F X Y = F.map biprod.inr :=
  biprod.inr_desc _ _


/-- The composition in the opposite direction is equal to the identity if and only if `F` preserves
    the biproduct, see `preservesBinaryBiproduct_of_monoBiprodComparison`. -/
@[reassoc (attr := simp)]
theorem biprodComparison'_comp_biprodComparison :
    biprodComparison' F X Y ≫ biprodComparison F X Y = 𝟙 (F.obj X ⊞ F.obj Y) := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    X Y : C
    inst✝² : CategoryTheory.Limits.HasBinaryBiproduct X Y
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct (F.obj X) (F.obj Y)
    inst✝ : F.PreservesZeroMorphisms
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.biprodComparison' X Y) (F.biprodCo …
  -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
  ext <;> simp [← Functor.map_comp]
          /-
            🎉 no goals
          -/


/-- `biprodComparison F X Y` is a split epi. -/
@[simps]
def splitEpiBiprodComparison : SplitEpi (biprodComparison F X Y) where
  section_ := biprodComparison' F X Y
           /-
             C : Type u₁
             inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
             D : Type u₂
             inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
             inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
             inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
             F : CategoryTheory.Functor C D
             X Y : C
             inst✝² : CategoryTheory.Limits.HasBinaryBiproduct X Y
             inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct (F.obj X) (F.obj Y)
             inst✝ : F.PreservesZeroMorphisms
             ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.biprodComparison' X Y) (F.biprodCo …
           -/
  id := by aesop
           /-
             🎉 no goals
           -/


instance : IsSplitEpi (biprodComparison F X Y) :=
  IsSplitEpi.mk' (splitEpiBiprodComparison F X Y)


/-- `biprodComparison' F X Y` is a split mono. -/
@[simps]
def splitMonoBiprodComparison' : SplitMono (biprodComparison' F X Y) where
  retraction := biprodComparison F X Y
           /-
             C : Type u₁
             inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
             D : Type u₂
             inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
             inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
             inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
             F : CategoryTheory.Functor C D
             X Y : C
             inst✝² : CategoryTheory.Limits.HasBinaryBiproduct X Y
             inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct (F.obj X) (F.obj Y)
             inst✝ : F.PreservesZeroMorphisms
             ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.biprodComparison' X Y) (F.biprodCo …
           -/
  id := by aesop
           /-
             🎉 no goals
           -/


instance : IsSplitMono (biprodComparison' F X Y) :=
  IsSplitMono.mk' (splitMonoBiprodComparison' F X Y)


instance hasBinaryBiproduct_of_preserves : HasBinaryBiproduct (F.obj X) (F.obj Y) :=
  HasBinaryBiproduct.mk
    { bicone := F.mapBinaryBicone (BinaryBiproduct.bicone X Y)
      isBilimit := isBinaryBilimitOfPreserves F (BinaryBiproduct.isBilimit _ _) }


/-- If `F` preserves a binary biproduct, we get a definitionally nice isomorphism
    `F.obj (X ⊞ Y) ≅ F.obj X ⊞ F.obj Y`. -/
@[simp]
def mapBiprod : F.obj (X ⊞ Y) ≅ F.obj X ⊞ F.obj Y :=
  biprod.uniqueUpToIso _ _ (isBinaryBilimitOfPreserves F (BinaryBiproduct.isBilimit _ _))


theorem mapBiprod_hom : (mapBiprod F X Y).hom = biprod.lift (F.map biprod.fst) (F.map biprod.snd) :=
  rfl


theorem mapBiprod_inv : (mapBiprod F X Y).inv = biprod.desc (F.map biprod.inl) (F.map biprod.inr) :=
  rfl


theorem biproduct.map_lift_mapBiprod (g : ∀ j, W ⟶ f j) :
    -- Porting note: twice we need haveI to tell Lean about hasBiproduct_of_preserves F f
    haveI : HasBiproduct fun j => F.obj (f j) := hasBiproduct_of_preserves F f
    F.map (biproduct.lift g) ≫ (F.mapBiproduct f).hom = biproduct.lift fun j => F.map (g j) := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    J : Type w₁
    f : J → C
    inst✝¹ : CategoryTheory.Limits.HasBiproduct f
    inst✝ : CategoryTheory.Limits.PreservesBiproduct f F
    W : C
    g : (j : J) → Quiver.Hom W (f j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.biprodu …
  -/
  ext j
  /-
    case w
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    J : Type w₁
    f : J → C
    inst✝¹ : CategoryTheory.Limits.HasBiproduct f
    inst✝ : CategoryTheory.Limits.PreservesBiproduct f F
    W : C
    g : (j : J) → Quiver.Hom W (f j)
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp only [Function.comp_def]
  /-
    case w
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    J : Type w₁
    f : J → C
    inst✝¹ : CategoryTheory.Limits.HasBiproduct f
    inst✝ : CategoryTheory.Limits.PreservesBiproduct f F
    W : C
    g : (j : J) → Quiver.Hom W (f j)
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  haveI : HasBiproduct fun j => F.obj (f j) := hasBiproduct_of_preserves F f
  /-
    case w
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    J : Type w₁
    f : J → C
    inst✝¹ : CategoryTheory.Limits.HasBiproduct f
    inst✝ : CategoryTheory.Limits.PreservesBiproduct f F
    W : C
    g : (j : J) → Quiver.Hom W (f j)
    j : J
    this : CategoryTheory.Limits.HasBiproduct fun j => F.obj (f j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [mapBiproduct_hom, Category.assoc, biproduct.lift_π, ← F.map_comp]
  /-
    🎉 no goals
  -/


theorem biproduct.mapBiproduct_inv_map_desc (g : ∀ j, f j ⟶ W) :
    -- Porting note: twice we need haveI to tell Lean about hasBiproduct_of_preserves F f
    haveI : HasBiproduct fun j => F.obj (f j) := hasBiproduct_of_preserves F f
    (F.mapBiproduct f).inv ≫ F.map (biproduct.desc g) = biproduct.desc fun j => F.map (g j) := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    J : Type w₁
    f : J → C
    inst✝¹ : CategoryTheory.Limits.HasBiproduct f
    inst✝ : CategoryTheory.Limits.PreservesBiproduct f F
    W : C
    g : (j : J) → Quiver.Hom (f j) W
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.mapBiproduct f).inv (F.map (Catego …
  -/
  ext j
  /-
    case w
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    J : Type w₁
    f : J → C
    inst✝¹ : CategoryTheory.Limits.HasBiproduct f
    inst✝ : CategoryTheory.Limits.PreservesBiproduct f F
    W : C
    g : (j : J) → Quiver.Hom (f j) W
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (F …
  -/
  dsimp only [Function.comp_def]
  /-
    case w
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    J : Type w₁
    f : J → C
    inst✝¹ : CategoryTheory.Limits.HasBiproduct f
    inst✝ : CategoryTheory.Limits.PreservesBiproduct f F
    W : C
    g : (j : J) → Quiver.Hom (f j) W
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (f …
  -/
  haveI : HasBiproduct fun j => F.obj (f j) := hasBiproduct_of_preserves F f
  /-
    case w
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    J : Type w₁
    f : J → C
    inst✝¹ : CategoryTheory.Limits.HasBiproduct f
    inst✝ : CategoryTheory.Limits.PreservesBiproduct f F
    W : C
    g : (j : J) → Quiver.Hom (f j) W
    j : J
    this : CategoryTheory.Limits.HasBiproduct fun j => F.obj (f j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (f …
  -/
  simp only [mapBiproduct_inv, ← Category.assoc, biproduct.ι_desc ,← F.map_comp]
  /-
    🎉 no goals
  -/


theorem biproduct.mapBiproduct_hom_desc (g : ∀ j, f j ⟶ W) :
    ((F.mapBiproduct f).hom ≫ biproduct.desc fun j => F.map (g j)) = F.map (biproduct.desc g) := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    J : Type w₁
    f : J → C
    inst✝¹ : CategoryTheory.Limits.HasBiproduct f
    inst✝ : CategoryTheory.Limits.PreservesBiproduct f F
    W : C
    g : (j : J) → Quiver.Hom (f j) W
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.mapBiproduct f).hom (CategoryTheor …
  -/
  rw [← biproduct.mapBiproduct_inv_map_desc, Iso.hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


theorem biprod.map_lift_mapBiprod (f : W ⟶ X) (g : W ⟶ Y) :
    F.map (biprod.lift f g) ≫ (F.mapBiprod X Y).hom = biprod.lift (F.map f) (F.map g) := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X Y
    inst✝ : CategoryTheory.Limits.PreservesBinaryBiproduct X Y F
    W : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.biprod. …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp [mapBiprod, ← F.map_comp]
          /-
            🎉 no goals
          -/


theorem biprod.lift_mapBiprod (f : W ⟶ X) (g : W ⟶ Y) :
    biprod.lift (F.map f) (F.map g) ≫ (F.mapBiprod X Y).inv = F.map (biprod.lift f g) := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X Y
    inst✝ : CategoryTheory.Limits.PreservesBinaryBiproduct X Y F
    W : C
    f : Quiver.Hom W X
    g : Quiver.Hom W Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.lift (F …
  -/
  rw [← biprod.map_lift_mapBiprod, Category.assoc, Iso.hom_inv_id, Category.comp_id]
  /-
    🎉 no goals
  -/


theorem biprod.mapBiprod_inv_map_desc (f : X ⟶ W) (g : Y ⟶ W) :
    (F.mapBiprod X Y).inv ≫ F.map (biprod.desc f g) = biprod.desc (F.map f) (F.map g) := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X Y
    inst✝ : CategoryTheory.Limits.PreservesBinaryBiproduct X Y F
    W : C
    f : Quiver.Hom X W
    g : Quiver.Hom Y W
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.mapBiprod X Y).inv (F.map (Categor …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp [mapBiprod, ← F.map_comp]
          /-
            🎉 no goals
          -/


theorem biprod.mapBiprod_hom_desc (f : X ⟶ W) (g : Y ⟶ W) :
    (F.mapBiprod X Y).hom ≫ biprod.desc (F.map f) (F.map g) = F.map (biprod.desc f g) := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X Y
    inst✝ : CategoryTheory.Limits.PreservesBinaryBiproduct X Y F
    W : C
    f : Quiver.Hom X W
    g : Quiver.Hom Y W
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.mapBiprod X Y).hom (CategoryTheory …
  -/
  rw [← biprod.mapBiprod_inv_map_desc, Iso.hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


