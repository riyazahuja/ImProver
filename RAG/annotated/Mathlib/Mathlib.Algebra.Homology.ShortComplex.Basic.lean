/-- A short complex in a category `C` with zero morphisms is the datum
of two composable morphisms `f : X₁ ⟶ X₂` and `g : X₂ ⟶ X₃` such that
`f ≫ g = 0`. -/
structure ShortComplex [HasZeroMorphisms C] where
  /-- the first (left) object of a `ShortComplex` -/
  {X₁ : C}
  /-- the second (middle) object of a `ShortComplex` -/
  {X₂ : C}
  /-- the third (right) object of a `ShortComplex` -/
  {X₃ : C}
  /-- the first morphism of a `ShortComplex` -/
  f : X₁ ⟶ X₂
  /-- the second morphism of a `ShortComplex` -/
  g : X₂ ⟶ X₃
  /-- the composition of the two given morphisms is zero -/
  zero : f ≫ g = 0


attribute [reassoc (attr := simp)] ShortComplex.zero


/-- Morphisms of short complexes are the commutative diagrams of the obvious shape. -/
@[ext]
structure Hom (S₁ S₂ : ShortComplex C) where
  /-- the morphism on the left objects -/
  τ₁ : S₁.X₁ ⟶ S₂.X₁
  /-- the morphism on the middle objects -/
  τ₂ : S₁.X₂ ⟶ S₂.X₂
  /-- the morphism on the right objects -/
  τ₃ : S₁.X₃ ⟶ S₂.X₃
  /-- the left commutative square of a morphism in `ShortComplex` -/
  comm₁₂ : τ₁ ≫ S₂.f = S₁.f ≫ τ₂ := by aesop_cat
  /-- the right commutative square of a morphism in `ShortComplex` -/
  comm₂₃ : τ₂ ≫ S₂.g = S₁.g ≫ τ₃ := by aesop_cat


attribute [reassoc] Hom.comm₁₂ Hom.comm₂₃

/-- The identity morphism of a short complex. -/
@[simps]
def Hom.id : Hom S S where
  τ₁ := 𝟙 _
  τ₂ := 𝟙 _
  τ₃ := 𝟙 _


/-- The composition of morphisms of short complexes. -/
@[simps]
def Hom.comp (φ₁₂ : Hom S₁ S₂) (φ₂₃ : Hom S₂ S₃) : Hom S₁ S₃ where
  τ₁ := φ₁₂.τ₁ ≫ φ₂₃.τ₁
  τ₂ := φ₁₂.τ₂ ≫ φ₂₃.τ₂
  τ₃ := φ₁₂.τ₃ ≫ φ₂₃.τ₃


instance : Category (ShortComplex C) where
  Hom := Hom
  id := Hom.id
  comp := Hom.comp


@[ext]
lemma hom_ext (f g : S₁ ⟶ S₂) (h₁ : f.τ₁ = g.τ₁) (h₂ : f.τ₂ = g.τ₂) (h₃ : f.τ₃ = g.τ₃) : f = g :=
  Hom.ext h₁ h₂ h₃


/-- A constructor for morphisms in `ShortComplex C` when the commutativity conditions
are not obvious. -/
@[simps]
def homMk {S₁ S₂ : ShortComplex C} (τ₁ : S₁.X₁ ⟶ S₂.X₁) (τ₂ : S₁.X₂ ⟶ S₂.X₂)
    (τ₃ : S₁.X₃ ⟶ S₂.X₃) (comm₁₂ : τ₁ ≫ S₂.f = S₁.f ≫ τ₂)
    (comm₂₃ : τ₂ ≫ S₂.g = S₁.g ≫ τ₃) : S₁ ⟶ S₂ := ⟨τ₁, τ₂, τ₃, comm₁₂, comm₂₃⟩


@[simp] lemma id_τ₁ : Hom.τ₁ (𝟙 S) = 𝟙 _ := rfl

@[simp] lemma id_τ₂ : Hom.τ₂ (𝟙 S) = 𝟙 _ := rfl

@[simp] lemma id_τ₃ : Hom.τ₃ (𝟙 S) = 𝟙 _ := rfl

@[reassoc] lemma comp_τ₁ (φ₁₂ : S₁ ⟶ S₂) (φ₂₃ : S₂ ⟶ S₃) :
    (φ₁₂ ≫ φ₂₃).τ₁ = φ₁₂.τ₁ ≫ φ₂₃.τ₁ := rfl

@[reassoc] lemma comp_τ₂ (φ₁₂ : S₁ ⟶ S₂) (φ₂₃ : S₂ ⟶ S₃) :
    (φ₁₂ ≫ φ₂₃).τ₂ = φ₁₂.τ₂ ≫ φ₂₃.τ₂ := rfl

@[reassoc] lemma comp_τ₃ (φ₁₂ : S₁ ⟶ S₂) (φ₂₃ : S₂ ⟶ S₃) :
    (φ₁₂ ≫ φ₂₃).τ₃ = φ₁₂.τ₃ ≫ φ₂₃.τ₃ := rfl


instance : Zero (S₁ ⟶ S₂) := ⟨{ τ₁ := 0, τ₂ := 0, τ₃ := 0 }⟩


@[simp] lemma zero_τ₁ : Hom.τ₁ (0 : S₁ ⟶ S₂) = 0 := rfl

@[simp] lemma zero_τ₂ : Hom.τ₂ (0 : S₁ ⟶ S₂) = 0 := rfl

@[simp] lemma zero_τ₃ : Hom.τ₃ (0 : S₁ ⟶ S₂) = 0 := rfl


instance : HasZeroMorphisms (ShortComplex C) where


/-- The first projection functor `ShortComplex C ⥤ C`. -/
@[simps]
def π₁ : ShortComplex C ⥤ C where
  obj S := S.X₁
  map f := f.τ₁


/-- The second projection functor `ShortComplex C ⥤ C`. -/
@[simps]
def π₂ : ShortComplex C ⥤ C where
  obj S := S.X₂
  map f := f.τ₂


/-- The third projection functor `ShortComplex C ⥤ C`. -/
@[simps]
def π₃ : ShortComplex C ⥤ C where
  obj S := S.X₃
  map f := f.τ₃


instance preservesZeroMorphisms_π₁ : Functor.PreservesZeroMorphisms (π₁ : _ ⥤ C) where

instance preservesZeroMorphisms_π₂ : Functor.PreservesZeroMorphisms (π₂ : _ ⥤ C) where

instance preservesZeroMorphisms_π₃ : Functor.PreservesZeroMorphisms (π₃ : _ ⥤ C) where


instance (f : S₁ ⟶ S₂) [IsIso f] : IsIso f.τ₁ := (inferInstance : IsIso (π₁.mapIso (asIso f)).hom)

instance (f : S₁ ⟶ S₂) [IsIso f] : IsIso f.τ₂ := (inferInstance : IsIso (π₂.mapIso (asIso f)).hom)

instance (f : S₁ ⟶ S₂) [IsIso f] : IsIso f.τ₃ := (inferInstance : IsIso (π₃.mapIso (asIso f)).hom)


/-- The natural transformation `π₁ ⟶ π₂` induced by `S.f` for all `S : ShortComplex C`. -/
@[simps] def π₁Toπ₂ : (π₁ : _ ⥤ C) ⟶ π₂ where
  app S := S.f


/-- The natural transformation `π₂ ⟶ π₃` induced by `S.g` for all `S : ShortComplex C`. -/
@[simps] def π₂Toπ₃ : (π₂ : _ ⥤ C) ⟶ π₃ where
  app S := S.g


@[reassoc (attr := simp)]
                                                                         /-
                                                                           C : Type u_1
                                                                           inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
                                                                           inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.ShortComplex.π₁Toπ₂ Ca …
                                                                         -/
lemma π₁Toπ₂_comp_π₂Toπ₃ : (π₁Toπ₂ : (_ : _ ⥤ C) ⟶ _) ≫ π₂Toπ₃ = 0 := by aesop_cat
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- The short complex in `D` obtained by applying a functor `F : C ⥤ D` to a
short complex in `C`, assuming that `F` preserves zero morphisms. -/
@[simps]
def map (F : C ⥤ D) [F.PreservesZeroMorphisms] : ShortComplex D :=
                                              /-
                                                C : Type u_1
                                                D : Type u_2
                                                inst✝⁴ : CategoryTheory.Category.{?u.35998, u_1} C
                                                inst✝³ : CategoryTheory.Category.{?u.36002, u_2} D
                                                inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                                S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                                inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
                                                F : CategoryTheory.Functor C D
                                                inst✝ : F.PreservesZeroMorphisms
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map S.f) (F.map S.g)) 0
                                              -/
  ShortComplex.mk (F.map S.f) (F.map S.g) (by rw [← F.map_comp, S.zero, F.map_zero])
                                              /-
                                                🎉 no goals
                                              -/


/-- The morphism of short complexes `S.map F ⟶ S.map G` induced by
a natural transformation `F ⟶ G`. -/
@[simps]
def mapNatTrans {F G : C ⥤ D} [F.PreservesZeroMorphisms] [G.PreservesZeroMorphisms] (τ : F ⟶ G) :
    S.map F ⟶ S.map G where
  τ₁ := τ.app _
  τ₂ := τ.app _
  τ₃ := τ.app _


/-- The isomorphism of short complexes `S.map F ≅ S.map G` induced by
a natural isomorphism `F ≅ G`. -/
@[simps]
def mapNatIso {F G : C ⥤ D} [F.PreservesZeroMorphisms] [G.PreservesZeroMorphisms] (τ : F ≅ G) :
    S.map F ≅ S.map G where
  hom := S.mapNatTrans τ.hom
  inv := S.mapNatTrans τ.inv


/-- The functor `ShortComplex C ⥤ ShortComplex D` induced by a functor `C ⥤ D` which
preserves zero morphisms. -/
@[simps]
def _root_.CategoryTheory.Functor.mapShortComplex (F : C ⥤ D) [F.PreservesZeroMorphisms] :
    ShortComplex C ⥤ ShortComplex D where
  obj S := S.map F
  map φ :=
    { τ₁ := F.map φ.τ₁
      τ₂ := F.map φ.τ₂
      τ₃ := F.map φ.τ₃
      comm₁₂ := by
        /-
          C : Type u_1
          D : Type u_2
          inst✝⁴ : CategoryTheory.Category.{?u.43874, u_1} C
          inst✝³ : CategoryTheory.Category.{?u.43878, u_2} D
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C D
          inst✝ : F.PreservesZeroMorphisms
          X✝ Y✝ : CategoryTheory.ShortComplex C
          φ : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ.τ₁) ((fun S => S.map F) Y✝). …
        -/
        dsimp
        /-
          C : Type u_1
          D : Type u_2
          inst✝⁴ : CategoryTheory.Category.{?u.43874, u_1} C
          inst✝³ : CategoryTheory.Category.{?u.43878, u_2} D
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C D
          inst✝ : F.PreservesZeroMorphisms
          X✝ Y✝ : CategoryTheory.ShortComplex C
          φ : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ.τ₁) (F.map Y✝.f)) (CategoryT …
        -/
        simp only [← F.map_comp, φ.comm₁₂]
        /-
          🎉 no goals
        -/
      comm₂₃ := by
        /-
          C : Type u_1
          D : Type u_2
          inst✝⁴ : CategoryTheory.Category.{?u.43874, u_1} C
          inst✝³ : CategoryTheory.Category.{?u.43878, u_2} D
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C D
          inst✝ : F.PreservesZeroMorphisms
          X✝ Y✝ : CategoryTheory.ShortComplex C
          φ : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ.τ₂) ((fun S => S.map F) Y✝). …
        -/
        dsimp
        /-
          C : Type u_1
          D : Type u_2
          inst✝⁴ : CategoryTheory.Category.{?u.43874, u_1} C
          inst✝³ : CategoryTheory.Category.{?u.43878, u_2} D
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C D
          inst✝ : F.PreservesZeroMorphisms
          X✝ Y✝ : CategoryTheory.ShortComplex C
          φ : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ.τ₂) (F.map Y✝.g)) (CategoryT …
        -/
        simp only [← F.map_comp, φ.comm₂₃] }
        /-
          🎉 no goals
        -/


/-- A constructor for isomorphisms in the category `ShortComplex C`-/
@[simps]
def isoMk (e₁ : S₁.X₁ ≅ S₂.X₁) (e₂ : S₁.X₂ ≅ S₂.X₂) (e₃ : S₁.X₃ ≅ S₂.X₃)
    (comm₁₂ : e₁.hom ≫ S₂.f = S₁.f ≫ e₂.hom := by aesop_cat)
    (comm₂₃ : e₂.hom ≫ S₂.g = S₁.g ≫ e₃.hom := by aesop_cat) :
    S₁ ≅ S₂ where
  hom := ⟨e₁.hom, e₂.hom, e₃.hom, comm₁₂, comm₂₃⟩
  inv := homMk e₁.inv e₂.inv e₃.inv
    (by rw [← cancel_mono e₂.hom, assoc, assoc, e₂.inv_hom_id, comp_id,
          ← comm₁₂, e₁.inv_hom_id_assoc])
    (by rw [← cancel_mono e₃.hom, assoc, assoc, e₃.inv_hom_id, comp_id,
          ← comm₂₃, e₂.inv_hom_id_assoc])


lemma isIso_of_isIso (f : S₁ ⟶ S₂) [IsIso f.τ₁] [IsIso f.τ₂] [IsIso f.τ₃] : IsIso f :=
   /-
     C : Type u_1
     inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
     inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
     S₁ S₂ : CategoryTheory.ShortComplex C
     f : Quiver.Hom S₁ S₂
     inst✝² : CategoryTheory.IsIso f.τ₁
     inst✝¹ : CategoryTheory.IsIso f.τ₂
     inst✝ : CategoryTheory.IsIso f.τ₃
     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.asIso f.τ₁).hom S₂.f) …
   -/
   /-
     🎉 no goals
   -/
  (isoMk (asIso f.τ₁) (asIso f.τ₂) (asIso f.τ₃)).isIso_hom
   /-
     🎉 no goals
   -/


/-- The opposite `ShortComplex` in `Cᵒᵖ` associated to a short complex in `C`. -/
@[simps]
def op : ShortComplex Cᵒᵖ :=
                       /-
                         C : Type u_1
                         D : Type u_2
                         inst✝³ : CategoryTheory.Category.{?u.61803, u_1} C
                         inst✝² : CategoryTheory.Category.{?u.61807, u_2} D
                         inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                         S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                         inst✝ : CategoryTheory.Limits.HasZeroMorphisms D
                         ⊢ Eq (CategoryTheory.CategoryStruct.comp S.g.op S.f.op) 0
                       -/
  mk S.g.op S.f.op (by simp only [← op_comp, S.zero]; rfl)
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- The opposite morphism in `ShortComplex Cᵒᵖ` associated to a morphism in `ShortComplex C` -/
@[simps]
def opMap (φ : S₁ ⟶ S₂) : S₂.op ⟶ S₁.op where
  τ₁ := φ.τ₃.op
  τ₂ := φ.τ₂.op
  τ₃ := φ.τ₁.op
  comm₁₂ := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.62475, u_1} C
      inst✝² : CategoryTheory.Category.{?u.62479, u_2} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms D
      φ : Quiver.Hom S₁ S₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.τ₃.op S₁.op.f) (CategoryTheory.Cate …
    -/
    dsimp
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.62475, u_1} C
      inst✝² : CategoryTheory.Category.{?u.62479, u_2} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms D
      φ : Quiver.Hom S₁ S₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.τ₃.op S₁.g.op) (CategoryTheory.Cate …
    -/
    simp only [← op_comp, φ.comm₂₃]
    /-
      🎉 no goals
    -/
  comm₂₃ := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.62475, u_1} C
      inst✝² : CategoryTheory.Category.{?u.62479, u_2} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms D
      φ : Quiver.Hom S₁ S₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.τ₂.op S₁.op.g) (CategoryTheory.Cate …
    -/
    dsimp
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.62475, u_1} C
      inst✝² : CategoryTheory.Category.{?u.62479, u_2} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms D
      φ : Quiver.Hom S₁ S₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.τ₂.op S₁.f.op) (CategoryTheory.Cate …
    -/
    simp only [← op_comp, φ.comm₁₂]
    /-
      🎉 no goals
    -/


@[simp]
lemma opMap_id : opMap (𝟙 S) = 𝟙 S.op := rfl


/-- The `ShortComplex` in `C` associated to a short complex in `Cᵒᵖ`. -/
@[simps]
def unop (S : ShortComplex Cᵒᵖ) : ShortComplex C :=
                           /-
                             C : Type u_1
                             D : Type u_2
                             inst✝³ : CategoryTheory.Category.{?u.63913, u_1} C
                             inst✝² : CategoryTheory.Category.{?u.63917, u_2} D
                             inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                             S✝ S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                             inst✝ : CategoryTheory.Limits.HasZeroMorphisms D
                             S : CategoryTheory.ShortComplex (Opposite C)
                             ⊢ Eq (CategoryTheory.CategoryStruct.comp S.g.unop S.f.unop) 0
                           -/
  mk S.g.unop S.f.unop (by simp only [← unop_comp, S.zero]; rfl)
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- The morphism in `ShortComplex C` associated to a morphism in `ShortComplex Cᵒᵖ` -/
@[simps]
def unopMap {S₁ S₂ : ShortComplex Cᵒᵖ} (φ : S₁ ⟶ S₂) : S₂.unop ⟶ S₁.unop where
  τ₁ := φ.τ₃.unop
  τ₂ := φ.τ₂.unop
  τ₃ := φ.τ₁.unop
  comm₁₂ := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.64657, u_1} C
      inst✝² : CategoryTheory.Category.{?u.64661, u_2} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁✝ S₂✝ S₃ : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms D
      S₁ S₂ : CategoryTheory.ShortComplex (Opposite C)
      φ : Quiver.Hom S₁ S₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.τ₃.unop S₁.unop.f) (CategoryTheory. …
    -/
    dsimp
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.64657, u_1} C
      inst✝² : CategoryTheory.Category.{?u.64661, u_2} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁✝ S₂✝ S₃ : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms D
      S₁ S₂ : CategoryTheory.ShortComplex (Opposite C)
      φ : Quiver.Hom S₁ S₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.τ₃.unop S₁.g.unop) (CategoryTheory. …
    -/
    simp only [← unop_comp, φ.comm₂₃]
    /-
      🎉 no goals
    -/
  comm₂₃ := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.64657, u_1} C
      inst✝² : CategoryTheory.Category.{?u.64661, u_2} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁✝ S₂✝ S₃ : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms D
      S₁ S₂ : CategoryTheory.ShortComplex (Opposite C)
      φ : Quiver.Hom S₁ S₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.τ₂.unop S₁.unop.g) (CategoryTheory. …
    -/
    dsimp
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.64657, u_1} C
      inst✝² : CategoryTheory.Category.{?u.64661, u_2} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁✝ S₂✝ S₃ : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms D
      S₁ S₂ : CategoryTheory.ShortComplex (Opposite C)
      φ : Quiver.Hom S₁ S₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.τ₂.unop S₁.f.unop) (CategoryTheory. …
    -/
    simp only [← unop_comp, φ.comm₁₂]
    /-
      🎉 no goals
    -/


@[simp]
lemma unopMap_id (S : ShortComplex Cᵒᵖ) : unopMap (𝟙 S) = 𝟙 S.unop := rfl


/-- The obvious functor `(ShortComplex C)ᵒᵖ ⥤ ShortComplex Cᵒᵖ`. -/
@[simps]
def opFunctor : (ShortComplex C)ᵒᵖ ⥤ ShortComplex Cᵒᵖ where
  obj S := (Opposite.unop S).op
  map φ := opMap φ.unop


/-- The obvious functor `ShortComplex Cᵒᵖ ⥤ (ShortComplex C)ᵒᵖ`. -/
@[simps]
def unopFunctor : ShortComplex Cᵒᵖ ⥤ (ShortComplex C)ᵒᵖ where
  obj S := Opposite.op (S.unop)
  map φ := (unopMap φ).op


/-- The obvious equivalence of categories `(ShortComplex C)ᵒᵖ ≌ ShortComplex Cᵒᵖ`. -/
@[simps]
def opEquiv : (ShortComplex C)ᵒᵖ ≌ ShortComplex Cᵒᵖ where
  functor := opFunctor C
  inverse := unopFunctor C
  unitIso := Iso.refl _
  counitIso := Iso.refl _


/-- The canonical isomorphism `S.unop.op ≅ S` for a short complex `S` in `Cᵒᵖ` -/
abbrev unopOp (S : ShortComplex Cᵒᵖ) : S.unop.op ≅ S := (opEquiv C).counitIso.app S


/-- The canonical isomorphism `S.op.unop ≅ S` for a short complex `S` -/
abbrev opUnop (S : ShortComplex C) : S.op.unop ≅ S :=
  Iso.unop ((opEquiv C).unitIso.app (Opposite.op S))


