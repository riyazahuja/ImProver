/-- A functor `F : C ⥤ D` is full if for each `X Y : C`, `F.map` is surjective.

See <https://stacks.math.columbia.edu/tag/001C>.
-/
class Full (F : C ⥤ D) : Prop where
  map_surjective {X Y : C} : Function.Surjective (F.map (X := X) (Y := Y))


/-- A functor `F : C ⥤ D` is faithful if for each `X Y : C`, `F.map` is injective.

See <https://stacks.math.columbia.edu/tag/001C>.
-/
class Faithful (F : C ⥤ D) : Prop where
  /-- `F.map` is injective for each `X Y : C`. -/
  map_injective : ∀ {X Y : C}, Function.Injective (F.map : (X ⟶ Y) → (F.obj X ⟶ F.obj Y)) := by
    aesop_cat


theorem map_injective (F : C ⥤ D) [Faithful F] :
    Function.Injective <| (F.map : (X ⟶ Y) → (F.obj X ⟶ F.obj Y)) :=
  Faithful.map_injective


lemma map_injective_iff (F : C ⥤ D) [Faithful F] {X Y : C} (f g : X ⟶ Y) :
    F.map f = F.map g ↔ f = g :=
                                           /-
                                             C : Type u₁
                                             inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                             D : Type u₂
                                             inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                             F : CategoryTheory.Functor C D
                                             inst✝ : F.Faithful
                                             X Y : C
                                             f g : Quiver.Hom X Y
                                             h : Eq f g
                                             ⊢ Eq (F.map f) (F.map g)
                                           -/
  ⟨fun h => F.map_injective h, fun h => by rw [h]⟩
                                           /-
                                             🎉 no goals
                                           -/


theorem mapIso_injective (F : C ⥤ D) [Faithful F] :
    Function.Injective <| (F.mapIso : (X ≅ Y) → (F.obj X ≅ F.obj Y))  := fun _ _ h =>
  Iso.ext (map_injective F (congr_arg Iso.hom h : _))


theorem map_surjective (F : C ⥤ D) [Full F] :
    Function.Surjective (F.map : (X ⟶ Y) → (F.obj X ⟶ F.obj Y)) :=
  Full.map_surjective


/-- The choice of a preimage of a morphism under a full functor. -/
noncomputable def preimage (F : C ⥤ D) [Full F] (f : F.obj X ⟶ F.obj Y) : X ⟶ Y :=
  (F.map_surjective f).choose


@[simp]
theorem map_preimage (F : C ⥤ D) [Full F] {X Y : C} (f : F.obj X ⟶ F.obj Y) :
    F.map (preimage F f) = f :=
  (F.map_surjective f).choose_spec


@[simp]
theorem preimage_id : F.preimage (𝟙 (F.obj X)) = 𝟙 X :=
                      /-
                        C : Type u₁
                        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                        D : Type u₂
                        inst✝² : CategoryTheory.Category.{v₂, u₂} D
                        F : CategoryTheory.Functor C D
                        X : C
                        inst✝¹ : F.Full
                        inst✝ : F.Faithful
                        ⊢ Eq (F.map (F.preimage (CategoryTheory.CategoryStruct.id (F.obj X)))) (F.map  …
                      -/
  F.map_injective (by simp)
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem preimage_comp (f : F.obj X ⟶ F.obj Y) (g : F.obj Y ⟶ F.obj Z) :
    F.preimage (f ≫ g) = F.preimage f ≫ F.preimage g :=
                      /-
                        C : Type u₁
                        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                        D : Type u₂
                        inst✝² : CategoryTheory.Category.{v₂, u₂} D
                        F : CategoryTheory.Functor C D
                        X Y Z : C
                        inst✝¹ : F.Full
                        inst✝ : F.Faithful
                        f : Quiver.Hom (F.obj X) (F.obj Y)
                        g : Quiver.Hom (F.obj Y) (F.obj Z)
                        ⊢ Eq (F.map (F.preimage (CategoryTheory.CategoryStruct.comp f g))) (F.map (Cat …
                      -/
  F.map_injective (by simp)
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem preimage_map (f : X ⟶ Y) : F.preimage (F.map f) = f :=
                      /-
                        C : Type u₁
                        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                        D : Type u₂
                        inst✝² : CategoryTheory.Category.{v₂, u₂} D
                        F : CategoryTheory.Functor C D
                        X Y : C
                        inst✝¹ : F.Full
                        inst✝ : F.Faithful
                        f : Quiver.Hom X Y
                        ⊢ Eq (F.map (F.preimage (F.map f))) (F.map f)
                      -/
  F.map_injective (by simp)
                      /-
                        🎉 no goals
                      -/


/-- If `F : C ⥤ D` is fully faithful, every isomorphism `F.obj X ≅ F.obj Y` has a preimage. -/
@[simps]
noncomputable def preimageIso (f : F.obj X ≅ F.obj Y) :
    X ≅ Y where
  hom := F.preimage f.hom
  inv := F.preimage f.inv
                                    /-
                                      C : Type u₁
                                      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                      D : Type u₂
                                      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                                      E : Type u_1
                                      inst✝² : CategoryTheory.Category.{?u.4427, u_1} E
                                      X✝ Y✝ : C
                                      F : CategoryTheory.Functor C D
                                      X Y Z : C
                                      inst✝¹ : F.Full
                                      inst✝ : F.Faithful
                                      f : CategoryTheory.Iso (F.obj X) (F.obj Y)
                                      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (F.preimage f.hom) (F.preimage …
                                    -/
  hom_inv_id := F.map_injective (by simp)
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      C : Type u₁
                                      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                      D : Type u₂
                                      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                                      E : Type u_1
                                      inst✝² : CategoryTheory.Category.{?u.4427, u_1} E
                                      X✝ Y✝ : C
                                      F : CategoryTheory.Functor C D
                                      X Y Z : C
                                      inst✝¹ : F.Full
                                      inst✝ : F.Faithful
                                      f : CategoryTheory.Iso (F.obj X) (F.obj Y)
                                      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (F.preimage f.inv) (F.preimage …
                                    -/
  inv_hom_id := F.map_injective (by simp)
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem preimageIso_mapIso (f : X ≅ Y) : F.preimageIso (F.mapIso f) = f := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    f : CategoryTheory.Iso X Y
    ⊢ Eq (F.preimageIso (F.mapIso f)) f
  -/
  ext
  /-
    case w
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    f : CategoryTheory.Iso X Y
    ⊢ Eq (F.preimageIso (F.mapIso f)).hom f.hom
  -/
  simp
  /-
    🎉 no goals
  -/


variable (F) in
/-- Structure containing the data of inverse map `(F.obj X ⟶ F.obj Y) ⟶ (X ⟶ Y)` of `F.map`
in order to express that `F` is a fully faithful functor. -/
structure FullyFaithful where
  /-- The inverse map `(F.obj X ⟶ F.obj Y) ⟶ (X ⟶ Y)` of `F.map`. -/
  preimage {X Y : C} (f : F.obj X ⟶ F.obj Y) : X ⟶ Y
  map_preimage {X Y : C} (f : F.obj X ⟶ F.obj Y) : F.map (preimage f) = f := by aesop_cat
  preimage_map {X Y : C} (f : X ⟶ Y) : preimage (F.map f) = f := by aesop_cat


variable (F) in
/-- A `FullyFaithful` structure can be obtained from the assumption the `F` is both
full and faithful. -/
noncomputable def ofFullyFaithful [F.Full] [F.Faithful] :
    F.FullyFaithful where
  preimage := F.preimage


variable (C) in
/-- The identity functor is fully faithful. -/
@[simps]
def id : (𝟭 C).FullyFaithful where
  preimage f := f


/-- The equivalence `(X ⟶ Y) ≃ (F.obj X ⟶ F.obj Y)` given by `h : F.FullyFaithful`. -/
@[simps]
def homEquiv {X Y : C} : (X ⟶ Y) ≃ (F.obj X ⟶ F.obj Y) where
  toFun := F.map
  invFun := hF.preimage
                   /-
                     C : Type u₁
                     inst✝² : CategoryTheory.Category.{v₁, u₁} C
                     D : Type u₂
                     inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                     E : Type u_1
                     inst✝ : CategoryTheory.Category.{?u.8661, u_1} E
                     X✝¹ Y✝¹ : C
                     F : CategoryTheory.Functor C D
                     X✝ Y✝ Z : C
                     hF : F.FullyFaithful
                     X Y : C
                     x✝ : Quiver.Hom X Y
                     ⊢ Eq (hF.preimage (F.map x✝)) x✝
                   -/
  left_inv _ := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      C : Type u₁
                      inst✝² : CategoryTheory.Category.{v₁, u₁} C
                      D : Type u₂
                      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                      E : Type u_1
                      inst✝ : CategoryTheory.Category.{?u.8661, u_1} E
                      X✝¹ Y✝¹ : C
                      F : CategoryTheory.Functor C D
                      X✝ Y✝ Z : C
                      hF : F.FullyFaithful
                      X Y : C
                      x✝ : Quiver.Hom (F.obj X) (F.obj Y)
                      ⊢ Eq (F.map (hF.preimage x✝)) x✝
                    -/
  right_inv _ := by simp
                    /-
                      🎉 no goals
                    -/


lemma map_injective {X Y : C} {f g : X ⟶ Y} (h : F.map f = F.map g) : f = g :=
  hF.homEquiv.injective h


lemma map_surjective {X Y : C} :
    Function.Surjective (F.map : (X ⟶ Y) → (F.obj X ⟶ F.obj Y)) :=
  hF.homEquiv.surjective


lemma map_bijective (X Y : C) :
    Function.Bijective (F.map : (X ⟶ Y) → (F.obj X ⟶ F.obj Y)) :=
  hF.homEquiv.bijective


lemma full : F.Full where
  map_surjective := hF.map_surjective


lemma faithful : F.Faithful where
  map_injective := hF.map_injective


instance : Subsingleton F.FullyFaithful where
  allEq h₁ h₂ := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u_1
      inst✝ : CategoryTheory.Category.{?u.10373, u_1} E
      X✝ Y✝ : C
      F : CategoryTheory.Functor C D
      X Y Z : C
      hF h₁ h₂ : F.FullyFaithful
      ⊢ Eq h₁ h₂
    -/
    have := h₁.faithful
    cases h₁ with | mk f₁ hf₁ _ => cases h₂ with | mk f₂ hf₂ _ =>
    simp only [Functor.FullyFaithful.mk.injEq]
    ext
    apply F.map_injective
    rw [hf₁, hf₂]


/-- The unique isomorphism `X ≅ Y` which induces an isomorphism `F.obj X ≅ F.obj Y`
when `hF : F.FullyFaithful`. -/
@[simps]
def preimageIso {X Y : C} (e : F.obj X ≅ F.obj Y) : X ≅ Y where
  hom := hF.preimage e.hom
  inv := hF.preimage e.inv
                                     /-
                                       C : Type u₁
                                       inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                       D : Type u₂
                                       inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                       E : Type u_1
                                       inst✝ : CategoryTheory.Category.{?u.11155, u_1} E
                                       X✝¹ Y✝¹ : C
                                       F : CategoryTheory.Functor C D
                                       X✝ Y✝ Z : C
                                       hF : F.FullyFaithful
                                       X Y : C
                                       e : CategoryTheory.Iso (F.obj X) (F.obj Y)
                                       ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (hF.preimage e.hom) (hF.preima …
                                     -/
  hom_inv_id := hF.map_injective (by simp)
                                     /-
                                       🎉 no goals
                                     -/
                                     /-
                                       C : Type u₁
                                       inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                       D : Type u₂
                                       inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                       E : Type u_1
                                       inst✝ : CategoryTheory.Category.{?u.11155, u_1} E
                                       X✝¹ Y✝¹ : C
                                       F : CategoryTheory.Functor C D
                                       X✝ Y✝ Z : C
                                       hF : F.FullyFaithful
                                       X Y : C
                                       e : CategoryTheory.Iso (F.obj X) (F.obj Y)
                                       ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (hF.preimage e.inv) (hF.preima …
                                     -/
  inv_hom_id := hF.map_injective (by simp)
                                     /-
                                       🎉 no goals
                                     -/


lemma isIso_of_isIso_map {X Y : C} (f : X ⟶ Y) [IsIso (F.map f)] :
    IsIso f := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    hF : F.FullyFaithful
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso (F.map f)
    ⊢ CategoryTheory.IsIso f
  -/
  simpa using (hF.preimageIso (asIso (F.map f))).isIso_hom
  /-
    🎉 no goals
  -/


/-- The equivalence `(X ≅ Y) ≃ (F.obj X ≅ F.obj Y)` given by `h : F.FullyFaithful`. -/
@[simps]
def isoEquiv {X Y : C} : (X ≅ Y) ≃ (F.obj X ≅ F.obj Y) where
  toFun := F.mapIso
  invFun := hF.preimageIso
                 /-
                   C : Type u₁
                   inst✝² : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                   E : Type u_1
                   inst✝ : CategoryTheory.Category.{?u.12564, u_1} E
                   X✝¹ Y✝¹ : C
                   F : CategoryTheory.Functor C D
                   X✝ Y✝ Z : C
                   hF : F.FullyFaithful
                   X Y : C
                   ⊢ Function.LeftInverse hF.preimageIso F.mapIso
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    D : Type u₂
                    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                    E : Type u_1
                    inst✝ : CategoryTheory.Category.{?u.12564, u_1} E
                    X✝¹ Y✝¹ : C
                    F : CategoryTheory.Functor C D
                    X✝ Y✝ Z : C
                    hF : F.FullyFaithful
                    X Y : C
                    ⊢ Function.RightInverse hF.preimageIso F.mapIso
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


/-- Fully faithful functors are stable by composition. -/
@[simps]
def comp {G : D ⥤ E} (hG : G.FullyFaithful) : (F ⋙ G).FullyFaithful where
  preimage f := hF.preimage (hG.preimage f)


/-- If `F ⋙ G` is fully faithful and `G` is faithful, then `F` is fully faithful. -/
def ofCompFaithful {G : D ⥤ E} [G.Faithful] (hFG : (F ⋙ G).FullyFaithful) :
    F.FullyFaithful where
  preimage f := hFG.preimage (G.map f)
  map_preimage f := G.map_injective (hFG.map_preimage (G.map f))
  preimage_map f := hFG.preimage_map f


/-- If the image of a morphism under a fully faithful functor in an isomorphism,
then the original morphisms is also an isomorphism.
-/
theorem isIso_of_fully_faithful (f : X ⟶ Y) [IsIso (F.map f)] : IsIso f :=
                                                     /-
                                                       C : Type u₁
                                                       inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                       D : Type u₂
                                                       inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                                                       F : CategoryTheory.Functor C D
                                                       inst✝² : F.Full
                                                       inst✝¹ : F.Faithful
                                                       X Y : C
                                                       f : Quiver.Hom X Y
                                                       inst✝ : CategoryTheory.IsIso (F.map f)
                                                       ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp f (F.preimage (CategoryTheory. …
                                                     -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  ⟨⟨F.preimage (inv (F.map f)), ⟨F.map_injective (by simp), F.map_injective (by simp)⟩⟩⟩
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/



instance Full.id : Full (𝟭 C) where map_surjective := Function.surjective_id


instance Faithful.id : Functor.Faithful (𝟭 C) := { }


instance Faithful.comp [F.Faithful] [G.Faithful] :
    (F ⋙ G).Faithful where map_injective p := F.map_injective (G.map_injective p)


theorem Faithful.of_comp [(F ⋙ G).Faithful] : F.Faithful :=
  -- Porting note: (F ⋙ G).map_injective.of_comp has the incorrect type
  { map_injective := fun {_ _} => Function.Injective.of_comp (F ⋙ G).map_injective }


instance (priority := 100) [Quiver.IsThin C] : F.Faithful where


/-- If `F` is full, and naturally isomorphic to some `F'`, then `F'` is also full. -/
lemma Full.of_iso [Full F] (α : F ≅ F') : Full F' where
  map_surjective {X Y} f :=
                                                        /-
                                                          C : Type u₁
                                                          inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                          D : Type u₂
                                                          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                          F F' : CategoryTheory.Functor C D
                                                          inst✝ : F.Full
                                                          α : CategoryTheory.Iso F F'
                                                          X Y : C
                                                          f : Quiver.Hom (F'.obj X) (F'.obj Y)
                                                          ⊢ Eq (F'.map (F.preimage (CategoryTheory.CategoryStruct.comp (α.app X).hom (Ca …
                                                        -/
    ⟨F.preimage ((α.app X).hom ≫ f ≫ (α.app Y).inv), by simp [← NatIso.naturality_1 α]⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem Faithful.of_iso [F.Faithful] (α : F ≅ F') : F'.Faithful :=
  { map_injective := fun h =>
                          /-
                            C : Type u₁
                            inst✝² : CategoryTheory.Category.{v₁, u₁} C
                            D : Type u₂
                            inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                            F F' : CategoryTheory.Functor C D
                            inst✝ : F.Faithful
                            α : CategoryTheory.Iso F F'
                            X✝ Y✝ : C
                            a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
                            h : Eq (F'.map a₁✝) (F'.map a₂✝)
                            ⊢ Eq (F.map a₁✝) (F.map a₂✝)
                          -/
      F.map_injective (by rw [← NatIso.naturality_1 α.symm, h, NatIso.naturality_1 α.symm]) }
                          /-
                            🎉 no goals
                          -/


theorem Faithful.of_comp_iso {H : C ⥤ E} [H.Faithful] (h : F ⋙ G ≅ H) : F.Faithful :=
  @Faithful.of_comp _ _ _ _ _ _ F G (Faithful.of_iso h.symm)


alias _root_.CategoryTheory.Iso.faithful_of_comp := Faithful.of_comp_iso

-- We could prove this from `Faithful.of_comp_iso` using `eq_to_iso`,
-- but that would introduce a cyclic import.

theorem Faithful.of_comp_eq {H : C ⥤ E} [ℋ : H.Faithful] (h : F ⋙ G = H) : F.Faithful :=
  @Faithful.of_comp _ _ _ _ _ _ F G (h.symm ▸ ℋ)


alias _root_.Eq.faithful_of_comp := Faithful.of_comp_eq


/-- “Divide” a functor by a faithful functor. -/
protected def Faithful.div (F : C ⥤ E) (G : D ⥤ E) [G.Faithful] (obj : C → D)
    (h_obj : ∀ X, G.obj (obj X) = F.obj X) (map : ∀ {X Y}, (X ⟶ Y) → (obj X ⟶ obj Y))
    (h_map : ∀ {X Y} {f : X ⟶ Y}, HEq (G.map (map f)) (F.map f)) : C ⥤ D :=
  { obj, map := @map,
    map_id := by
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
        F✝ F' : CategoryTheory.Functor C D
        G✝ : CategoryTheory.Functor D E
        F : CategoryTheory.Functor C E
        G : CategoryTheory.Functor D E
        inst✝ : G.Faithful
        obj : C → D
        h_obj : ∀ (X : C), Eq (G.obj (obj X)) (F.obj X)
        map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
        h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq (G.map (map f)) (F.map f)
        ⊢ ∀ (X : C), Eq ({ obj := obj, map := map }.map (CategoryTheory.CategoryStruct …
      -/
      intros X
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
        F✝ F' : CategoryTheory.Functor C D
        G✝ : CategoryTheory.Functor D E
        F : CategoryTheory.Functor C E
        G : CategoryTheory.Functor D E
        inst✝ : G.Faithful
        obj : C → D
        h_obj : ∀ (X : C), Eq (G.obj (obj X)) (F.obj X)
        map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
        h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq (G.map (map f)) (F.map f)
        X : C
        ⊢ Eq ({ obj := obj, map := map }.map (CategoryTheory.CategoryStruct.id X)) (Ca …
      -/
      apply G.map_injective
      /-
        case a
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
        F✝ F' : CategoryTheory.Functor C D
        G✝ : CategoryTheory.Functor D E
        F : CategoryTheory.Functor C E
        G : CategoryTheory.Functor D E
        inst✝ : G.Faithful
        obj : C → D
        h_obj : ∀ (X : C), Eq (G.obj (obj X)) (F.obj X)
        map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
        h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq (G.map (map f)) (F.map f)
        X : C
        ⊢ Eq (G.map ({ obj := obj, map := map }.map (CategoryTheory.CategoryStruct.id  …
      -/
      apply eq_of_heq
      /-
        case a.h
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
        F✝ F' : CategoryTheory.Functor C D
        G✝ : CategoryTheory.Functor D E
        F : CategoryTheory.Functor C E
        G : CategoryTheory.Functor D E
        inst✝ : G.Faithful
        obj : C → D
        h_obj : ∀ (X : C), Eq (G.obj (obj X)) (F.obj X)
        map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
        h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq (G.map (map f)) (F.map f)
        X : C
        ⊢ HEq (G.map ({ obj := obj, map := map }.map (CategoryTheory.CategoryStruct.id …
      -/
      trans F.map (𝟙 X)
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          E : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
          F✝ F' : CategoryTheory.Functor C D
          G✝ : CategoryTheory.Functor D E
          F : CategoryTheory.Functor C E
          G : CategoryTheory.Functor D E
          inst✝ : G.Faithful
          obj : C → D
          h_obj : ∀ (X : C), Eq (G.obj (obj X)) (F.obj X)
          map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
          h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq (G.map (map f)) (F.map f)
          X : C
          ⊢ HEq (G.map ({ obj := obj, map := map }.map (CategoryTheory.CategoryStruct.id …
        -/
      · exact h_map
        /-
          🎉 no goals
        -/
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          E : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
          F✝ F' : CategoryTheory.Functor C D
          G✝ : CategoryTheory.Functor D E
          F : CategoryTheory.Functor C E
          G : CategoryTheory.Functor D E
          inst✝ : G.Faithful
          obj : C → D
          h_obj : ∀ (X : C), Eq (G.obj (obj X)) (F.obj X)
          map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
          h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq (G.map (map f)) (F.map f)
          X : C
          ⊢ HEq (F.map (CategoryTheory.CategoryStruct.id X)) (G.map (CategoryTheory.Cate …
        -/
      · rw [F.map_id, G.map_id, h_obj X]
        /-
          🎉 no goals
        -/
    map_comp := by
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
        F✝ F' : CategoryTheory.Functor C D
        G✝ : CategoryTheory.Functor D E
        F : CategoryTheory.Functor C E
        G : CategoryTheory.Functor D E
        inst✝ : G.Faithful
        obj : C → D
        h_obj : ∀ (X : C), Eq (G.obj (obj X)) (F.obj X)
        map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
        h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq (G.map (map f)) (F.map f)
        ⊢ ∀ {X Y Z : C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ obj := obj, m …
      -/
      intros X Y Z f g
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
        F✝ F' : CategoryTheory.Functor C D
        G✝ : CategoryTheory.Functor D E
        F : CategoryTheory.Functor C E
        G : CategoryTheory.Functor D E
        inst✝ : G.Faithful
        obj : C → D
        h_obj : ∀ (X : C), Eq (G.obj (obj X)) (F.obj X)
        map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
        h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq (G.map (map f)) (F.map f)
        X Y Z : C
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        ⊢ Eq ({ obj := obj, map := map }.map (CategoryTheory.CategoryStruct.comp f g)) …
      -/
      refine G.map_injective <| eq_of_heq <| h_map.trans ?_
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
        F✝ F' : CategoryTheory.Functor C D
        G✝ : CategoryTheory.Functor D E
        F : CategoryTheory.Functor C E
        G : CategoryTheory.Functor D E
        inst✝ : G.Faithful
        obj : C → D
        h_obj : ∀ (X : C), Eq (G.obj (obj X)) (F.obj X)
        map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
        h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq (G.map (map f)) (F.map f)
        X Y Z : C
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        ⊢ HEq (F.map (CategoryTheory.CategoryStruct.comp f g)) (G.map (CategoryTheory. …
      -/
      simp only [Functor.map_comp]
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
        F✝ F' : CategoryTheory.Functor C D
        G✝ : CategoryTheory.Functor D E
        F : CategoryTheory.Functor C E
        G : CategoryTheory.Functor D E
        inst✝ : G.Faithful
        obj : C → D
        h_obj : ∀ (X : C), Eq (G.obj (obj X)) (F.obj X)
        map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
        h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq (G.map (map f)) (F.map f)
        X Y Z : C
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        ⊢ HEq (CategoryTheory.CategoryStruct.comp (F.map f) (F.map g)) (CategoryTheory …
      -/
      convert HEq.refl (F.map f ≫ F.map g)
      /-
        case h.e'_3.h.e'_3
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
        F✝ F' : CategoryTheory.Functor C D
        G✝ : CategoryTheory.Functor D E
        F : CategoryTheory.Functor C E
        G : CategoryTheory.Functor D E
        inst✝ : G.Faithful
        obj : C → D
        h_obj : ∀ (X : C), Eq (G.obj (obj X)) (F.obj X)
        map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
        h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq (G.map (map f)) (F.map f)
        X Y Z : C
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        ⊢ Eq (G.obj (obj X)) (F.obj X)
      -/
      all_goals { first | apply h_obj | apply h_map } }
      /-
        🎉 no goals
      -/

-- This follows immediately from `Functor.hext` (`Functor.hext h_obj @h_map`),
-- but importing `CategoryTheory.EqToHom` causes an import loop:
-- CategoryTheory.EqToHom → CategoryTheory.Opposites →
-- CategoryTheory.Equivalence → CategoryTheory.FullyFaithful

theorem Faithful.div_comp (F : C ⥤ E) [F.Faithful] (G : D ⥤ E) [G.Faithful] (obj : C → D)
    (h_obj : ∀ X, G.obj (obj X) = F.obj X) (map : ∀ {X Y}, (X ⟶ Y) → (obj X ⟶ obj Y))
    (h_map : ∀ {X Y} {f : X ⟶ Y}, HEq (G.map (map f)) (F.map f)) :
    Faithful.div F G obj @h_obj @map @h_map ⋙ G = F := by
  -- Porting note: Have to unfold the structure twice because the first one recovers only the
  -- prefunctor `F_pre`
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C E
    inst✝¹ : F.Faithful
    G : CategoryTheory.Functor D E
    inst✝ : G.Faithful
    obj : C → D
    h_obj : ∀ (X : C), Eq (G.obj (obj X)) (F.obj X)
    map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
    h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq (G.map (map f)) (F.map f)
    ⊢ Eq ((CategoryTheory.Functor.Faithful.div F G obj h_obj map h_map).comp G) F
  -/
  cases' F with F_pre _ _; cases' G with G_pre _ _
  /-
    case mk.mk
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    obj : C → D
    map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
    F_pre : Prefunctor C E
    map_id✝¹ : ∀ (X : C), Eq (F_pre.map (CategoryTheory.CategoryStruct.id X)) (Cat …
    map_comp✝¹ : ∀ {X Y Z : C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq (F_pr …
    inst✝¹ : { toPrefunctor := F_pre, map_id := map_id✝¹, map_comp := map_comp✝¹ } …
    G_pre : Prefunctor D E
    map_id✝ : ∀ (X : D), Eq (G_pre.map (CategoryTheory.CategoryStruct.id X)) (Cate …
    map_comp✝ : ∀ {X Y Z : D} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq (G_pre …
    inst✝ : { toPrefunctor := G_pre, map_id := map_id✝, map_comp := map_comp✝ }.Fa …
    h_obj : ∀ (X : C), Eq ({ toPrefunctor := G_pre, map_id := map_id✝, map_comp := …
    h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq ({ toPrefunctor := G_pre, map_id …
    ⊢ Eq ((CategoryTheory.Functor.Faithful.div { toPrefunctor := F_pre, map_id :=  …
  -/
  cases' F_pre with F_obj _; cases' G_pre with G_obj _
  /-
    case mk.mk.mk.mk
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    obj : C → D
    map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
    F_obj : C → E
    map✝¹ : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (F_obj X) (F_obj Y)
    map_id✝¹ : ∀ (X : C), Eq ({ obj := F_obj, map := map✝¹ }.map (CategoryTheory.C …
    map_comp✝¹ : ∀ {X Y Z : C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ ob …
    inst✝¹ : { obj := F_obj, map := map✝¹, map_id := map_id✝¹, map_comp := map_com …
    G_obj : D → E
    map✝ : {X Y : D} → Quiver.Hom X Y → Quiver.Hom (G_obj X) (G_obj Y)
    map_id✝ : ∀ (X : D), Eq ({ obj := G_obj, map := map✝ }.map (CategoryTheory.Cat …
    map_comp✝ : ∀ {X Y Z : D} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ obj …
    inst✝ : { obj := G_obj, map := map✝, map_id := map_id✝, map_comp := map_comp✝  …
    h_obj : ∀ (X : C), Eq ({ obj := G_obj, map := map✝, map_id := map_id✝, map_com …
    h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq ({ obj := G_obj, map := map✝, ma …
    ⊢ Eq ((CategoryTheory.Functor.Faithful.div { obj := F_obj, map := map✝¹, map_i …
  -/
  unfold Faithful.div Functor.comp
  -- Porting note: unable to find the lean4 analogue to `unfold_projs`, works without it
  /-
    case mk.mk.mk.mk
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    obj : C → D
    map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
    F_obj : C → E
    map✝¹ : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (F_obj X) (F_obj Y)
    map_id✝¹ : ∀ (X : C), Eq ({ obj := F_obj, map := map✝¹ }.map (CategoryTheory.C …
    map_comp✝¹ : ∀ {X Y Z : C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ ob …
    inst✝¹ : { obj := F_obj, map := map✝¹, map_id := map_id✝¹, map_comp := map_com …
    G_obj : D → E
    map✝ : {X Y : D} → Quiver.Hom X Y → Quiver.Hom (G_obj X) (G_obj Y)
    map_id✝ : ∀ (X : D), Eq ({ obj := G_obj, map := map✝ }.map (CategoryTheory.Cat …
    map_comp✝ : ∀ {X Y Z : D} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ obj …
    inst✝ : { obj := G_obj, map := map✝, map_id := map_id✝, map_comp := map_comp✝  …
    h_obj : ∀ (X : C), Eq ({ obj := G_obj, map := map✝, map_id := map_id✝, map_com …
    h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq ({ obj := G_obj, map := map✝, ma …
    ⊢ Eq { obj := fun X => { obj := G_obj, map := map✝, map_id := map_id✝, map_com …
  -/
  have : F_obj = G_obj ∘ obj := (funext h_obj).symm
  /-
    case mk.mk.mk.mk
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    obj : C → D
    map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
    F_obj : C → E
    map✝¹ : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (F_obj X) (F_obj Y)
    map_id✝¹ : ∀ (X : C), Eq ({ obj := F_obj, map := map✝¹ }.map (CategoryTheory.C …
    map_comp✝¹ : ∀ {X Y Z : C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ ob …
    inst✝¹ : { obj := F_obj, map := map✝¹, map_id := map_id✝¹, map_comp := map_com …
    G_obj : D → E
    map✝ : {X Y : D} → Quiver.Hom X Y → Quiver.Hom (G_obj X) (G_obj Y)
    map_id✝ : ∀ (X : D), Eq ({ obj := G_obj, map := map✝ }.map (CategoryTheory.Cat …
    map_comp✝ : ∀ {X Y Z : D} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ obj …
    inst✝ : { obj := G_obj, map := map✝, map_id := map_id✝, map_comp := map_comp✝  …
    h_obj : ∀ (X : C), Eq ({ obj := G_obj, map := map✝, map_id := map_id✝, map_com …
    h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq ({ obj := G_obj, map := map✝, ma …
    this : Eq F_obj (Function.comp G_obj obj)
    ⊢ Eq { obj := fun X => { obj := G_obj, map := map✝, map_id := map_id✝, map_com …
  -/
  subst this
  /-
    case mk.mk.mk.mk
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    obj : C → D
    map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
    G_obj : D → E
    map✝¹ : {X Y : D} → Quiver.Hom X Y → Quiver.Hom (G_obj X) (G_obj Y)
    map_id✝¹ : ∀ (X : D), Eq ({ obj := G_obj, map := map✝¹ }.map (CategoryTheory.C …
    map_comp✝¹ : ∀ {X Y Z : D} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ ob …
    inst✝¹ : { obj := G_obj, map := map✝¹, map_id := map_id✝¹, map_comp := map_com …
    map✝ : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (Function.comp G_obj obj X) (Fu …
    map_id✝ : ∀ (X : C), Eq ({ obj := Function.comp G_obj obj, map := map✝ }.map ( …
    map_comp✝ : ∀ {X Y Z : C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ obj …
    inst✝ : { obj := Function.comp G_obj obj, map := map✝, map_id := map_id✝, map_ …
    h_obj : ∀ (X : C), Eq ({ obj := G_obj, map := map✝¹, map_id := map_id✝¹, map_c …
    h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq ({ obj := G_obj, map := map✝¹, m …
    ⊢ Eq { obj := fun X => { obj := G_obj, map := map✝¹, map_id := map_id✝¹, map_c …
  -/
  congr
  /-
    case mk.mk.mk.mk.e_toPrefunctor.e_map
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    obj : C → D
    map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
    G_obj : D → E
    map✝¹ : {X Y : D} → Quiver.Hom X Y → Quiver.Hom (G_obj X) (G_obj Y)
    map_id✝¹ : ∀ (X : D), Eq ({ obj := G_obj, map := map✝¹ }.map (CategoryTheory.C …
    map_comp✝¹ : ∀ {X Y Z : D} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ ob …
    inst✝¹ : { obj := G_obj, map := map✝¹, map_id := map_id✝¹, map_comp := map_com …
    map✝ : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (Function.comp G_obj obj X) (Fu …
    map_id✝ : ∀ (X : C), Eq ({ obj := Function.comp G_obj obj, map := map✝ }.map ( …
    map_comp✝ : ∀ {X Y Z : C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ obj …
    inst✝ : { obj := Function.comp G_obj obj, map := map✝, map_id := map_id✝, map_ …
    h_obj : ∀ (X : C), Eq ({ obj := G_obj, map := map✝¹, map_id := map_id✝¹, map_c …
    h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq ({ obj := G_obj, map := map✝¹, m …
    ⊢ Eq (fun {X Y} f => { obj := G_obj, map := map✝¹, map_id := map_id✝¹, map_com …
  -/
  simp only [Function.comp_apply, heq_eq_eq] at h_map
  /-
    case mk.mk.mk.mk.e_toPrefunctor.e_map
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    obj : C → D
    map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
    G_obj : D → E
    map✝¹ : {X Y : D} → Quiver.Hom X Y → Quiver.Hom (G_obj X) (G_obj Y)
    map_id✝¹ : ∀ (X : D), Eq ({ obj := G_obj, map := map✝¹ }.map (CategoryTheory.C …
    map_comp✝¹ : ∀ {X Y Z : D} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ ob …
    inst✝¹ : { obj := G_obj, map := map✝¹, map_id := map_id✝¹, map_comp := map_com …
    map✝ : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (Function.comp G_obj obj X) (Fu …
    map_id✝ : ∀ (X : C), Eq ({ obj := Function.comp G_obj obj, map := map✝ }.map ( …
    map_comp✝ : ∀ {X Y Z : C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ obj …
    inst✝ : { obj := Function.comp G_obj obj, map := map✝, map_id := map_id✝, map_ …
    h_obj : ∀ (X : C), Eq ({ obj := G_obj, map := map✝¹, map_id := map_id✝¹, map_c …
    h_map✝ : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq ({ obj := G_obj, map := map✝¹,  …
    h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, Eq (map✝¹ (map f)) (map✝ f)
    ⊢ Eq (fun {X Y} f => { obj := G_obj, map := map✝¹, map_id := map_id✝¹, map_com …
  -/
  ext
  /-
    case mk.mk.mk.mk.e_toPrefunctor.e_map.h.h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    obj : C → D
    map : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
    G_obj : D → E
    map✝¹ : {X Y : D} → Quiver.Hom X Y → Quiver.Hom (G_obj X) (G_obj Y)
    map_id✝¹ : ∀ (X : D), Eq ({ obj := G_obj, map := map✝¹ }.map (CategoryTheory.C …
    map_comp✝¹ : ∀ {X Y Z : D} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ ob …
    inst✝¹ : { obj := G_obj, map := map✝¹, map_id := map_id✝¹, map_comp := map_com …
    map✝ : {X Y : C} → Quiver.Hom X Y → Quiver.Hom (Function.comp G_obj obj X) (Fu …
    map_id✝ : ∀ (X : C), Eq ({ obj := Function.comp G_obj obj, map := map✝ }.map ( …
    map_comp✝ : ∀ {X Y Z : C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ obj …
    inst✝ : { obj := Function.comp G_obj obj, map := map✝, map_id := map_id✝, map_ …
    h_obj : ∀ (X : C), Eq ({ obj := G_obj, map := map✝¹, map_id := map_id✝¹, map_c …
    h_map✝ : ∀ {X Y : C} {f : Quiver.Hom X Y}, HEq ({ obj := G_obj, map := map✝¹,  …
    h_map : ∀ {X Y : C} {f : Quiver.Hom X Y}, Eq (map✝¹ (map f)) (map✝ f)
    x✝² x✝¹ : C
    x✝ : Quiver.Hom x✝² x✝¹
    ⊢ Eq ({ obj := G_obj, map := map✝¹, map_id := map_id✝¹, map_comp := map_comp✝¹ …
  -/
  exact h_map
  /-
    🎉 no goals
  -/


theorem Faithful.div_faithful (F : C ⥤ E) [F.Faithful] (G : D ⥤ E) [G.Faithful] (obj : C → D)
    (h_obj : ∀ X, G.obj (obj X) = F.obj X) (map : ∀ {X Y}, (X ⟶ Y) → (obj X ⟶ obj Y))
    (h_map : ∀ {X Y} {f : X ⟶ Y}, HEq (G.map (map f)) (F.map f)) :
    Functor.Faithful (Faithful.div F G obj @h_obj @map @h_map) :=
  (Faithful.div_comp F G _ h_obj _ @h_map).faithful_of_comp


instance Full.comp [Full F] [Full G] : Full (F ⋙ G) where
                                                     /-
                                                       C : Type u₁
                                                       inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                       D : Type u₂
                                                       inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                                                       E : Type u₃
                                                       inst✝² : CategoryTheory.Category.{v₃, u₃} E
                                                       F F' : CategoryTheory.Functor C D
                                                       G : CategoryTheory.Functor D E
                                                       inst✝¹ : F.Full
                                                       inst✝ : G.Full
                                                       X✝ Y✝ : C
                                                       f : Quiver.Hom ((F.comp G).obj X✝) ((F.comp G).obj Y✝)
                                                       ⊢ Eq ((F.comp G).map (F.preimage (G.preimage f))) f
                                                     -/
  map_surjective f := ⟨F.preimage (G.preimage f), by simp⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- If `F ⋙ G` is full and `G` is faithful, then `F` is full. -/
lemma Full.of_comp_faithful [Full <| F ⋙ G] [G.Faithful] : Full F where
  map_surjective f := ⟨(F ⋙ G).preimage (G.map f), G.map_injective ((F ⋙ G).map_preimage _)⟩


/-- If `F ⋙ G` is full and `G` is faithful, then `F` is full. -/
lemma Full.of_comp_faithful_iso {F : C ⥤ D} {G : D ⥤ E} {H : C ⥤ E} [Full H] [G.Faithful]
    (h : F ⋙ G ≅ H) : Full F := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    H : CategoryTheory.Functor C E
    inst✝¹ : H.Full
    inst✝ : G.Faithful
    h : CategoryTheory.Iso (F.comp G) H
    ⊢ F.Full
  -/
  have := Full.of_iso h.symm
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    H : CategoryTheory.Functor C E
    inst✝¹ : H.Full
    inst✝ : G.Faithful
    h : CategoryTheory.Iso (F.comp G) H
    this : (F.comp G).Full
    ⊢ F.Full
  -/
  exact Full.of_comp_faithful F G
  /-
    🎉 no goals
  -/


/-- Given a natural isomorphism between `F ⋙ H` and `G ⋙ H` for a fully faithful functor `H`, we
can 'cancel' it to give a natural iso between `F` and `G`.
-/
noncomputable def fullyFaithfulCancelRight {F G : C ⥤ D} (H : D ⥤ E) [Full H] [H.Faithful]
    (comp_iso : F ⋙ H ≅ G ⋙ H) : F ≅ G :=
  NatIso.ofComponents (fun X => H.preimageIso (comp_iso.app X)) fun f =>
                        /-
                          C : Type u₁
                          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                          D : Type u₂
                          inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                          E : Type u₃
                          inst✝² : CategoryTheory.Category.{v₃, u₃} E
                          F✝ F' : CategoryTheory.Functor C D
                          G✝ : CategoryTheory.Functor D E
                          F G : CategoryTheory.Functor C D
                          H : CategoryTheory.Functor D E
                          inst✝¹ : H.Full
                          inst✝ : H.Faithful
                          comp_iso : CategoryTheory.Iso (F.comp H) (G.comp H)
                          X✝ Y✝ : C
                          f : Quiver.Hom X✝ Y✝
                          ⊢ Eq (H.map (CategoryTheory.CategoryStruct.comp (F.map f) ((fun X => H.preimag …
                        -/
    H.map_injective (by simpa using comp_iso.hom.naturality f)
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem fullyFaithfulCancelRight_hom_app {F G : C ⥤ D} {H : D ⥤ E} [Full H] [H.Faithful]
    (comp_iso : F ⋙ H ≅ G ⋙ H) (X : C) :
    (fullyFaithfulCancelRight H comp_iso).hom.app X = H.preimage (comp_iso.hom.app X) :=
  rfl


@[simp]
theorem fullyFaithfulCancelRight_inv_app {F G : C ⥤ D} {H : D ⥤ E} [Full H] [H.Faithful]
    (comp_iso : F ⋙ H ≅ G ⋙ H) (X : C) :
    (fullyFaithfulCancelRight H comp_iso).inv.app X = H.preimage (comp_iso.inv.app X) :=
  rfl


@[deprecated (since := "2024-04-06")] alias Full := Functor.Full

@[deprecated (since := "2024-04-06")] alias Faithful := Functor.Faithful

@[deprecated (since := "2024-04-06")] alias preimage_id := Functor.preimage_id

@[deprecated (since := "2024-04-06")] alias preimage_comp := Functor.preimage_comp

@[deprecated (since := "2024-04-06")] alias preimage_map := Functor.preimage_map

@[deprecated (since := "2024-04-06")] alias Faithful.of_comp := Functor.Faithful.of_comp

@[deprecated (since := "2024-04-06")] alias Full.ofIso := Functor.Full.of_iso

@[deprecated (since := "2024-04-06")] alias Faithful.of_iso := Functor.Faithful.of_iso

@[deprecated (since := "2024-04-06")] alias Faithful.of_comp_iso := Functor.Faithful.of_comp_iso

@[deprecated (since := "2024-04-06")] alias Faithful.of_comp_eq := Functor.Faithful.of_comp_eq

@[deprecated (since := "2024-04-06")] alias Faithful.div := Functor.Faithful.div

@[deprecated (since := "2024-04-06")] alias Faithful.div_comp := Functor.Faithful.div_comp

@[deprecated (since := "2024-04-06")] alias Faithful.div_faithful := Functor.Faithful.div_faithful

@[deprecated (since := "2024-04-06")] alias Full.ofCompFaithful := Functor.Full.of_comp_faithful


@[deprecated (since := "2024-04-06")]
alias Full.ofCompFaithfulIso := Functor.Full.of_comp_faithful_iso


@[deprecated (since := "2024-04-06")]
alias fullyFaithfulCancelRight := Functor.fullyFaithfulCancelRight


@[deprecated (since := "2024-04-06")]
alias fullyFaithfulCancelRight_hom_app := Functor.fullyFaithfulCancelRight_hom_app


@[deprecated (since := "2024-04-06")]
alias fullyFaithfulCancelRight_inv_app := Functor.fullyFaithfulCancelRight_inv_app


@[deprecated (since := "2024-04-26")] alias Functor.image_preimage := Functor.map_preimage


