/-- Constructor for pseudofunctors from a locally discrete bicategory. In that
case, we do not need to provide the `map₂` field of pseudofunctors. -/
@[simps obj map mapId mapComp]
def pseudofunctorOfIsLocallyDiscrete
    {B C : Type*} [Bicategory B] [IsLocallyDiscrete B] [Bicategory C]
    (obj : B → C)
    (map : ∀ {b b' : B}, (b ⟶ b') → (obj b ⟶ obj b'))
    (mapId : ∀ (b : B), map (𝟙 b) ≅ 𝟙 _)
    (mapComp : ∀ {b₀ b₁ b₂ : B} (f : b₀ ⟶ b₁) (g : b₁ ⟶ b₂), map (f ≫ g) ≅ map f ≫ map g)
    (map₂_associator : ∀ {b₀ b₁ b₂ b₃ : B} (f : b₀ ⟶ b₁) (g : b₁ ⟶ b₂) (h : b₂ ⟶ b₃),
      (mapComp (f ≫ g) h).hom ≫
        (mapComp f g).hom ▷ map h ≫ (α_ (map f) (map g) (map h)).hom ≫
                                                                            /-
                                                                              B : Type u_1
                                                                              C : Type u_2
                                                                              inst✝² : CategoryTheory.Bicategory B
                                                                              inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
                                                                              inst✝ : CategoryTheory.Bicategory C
                                                                              obj : B → C
                                                                              map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
                                                                              mapId : (b : B) → CategoryTheory.Iso (map (CategoryTheory.CategoryStruct.id b) …
                                                                              mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → C …
                                                                              b₀ b₁ b₂ b₃ : B
                                                                              f : Quiver.Hom b₀ b₁
                                                                              g : Quiver.Hom b₁ b₂
                                                                              h : Quiver.Hom b₂ b₃
                                                                              ⊢ Eq (map (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
                                                                            -/
          map f ◁ (mapComp g h).inv ≫ (mapComp f (g ≫ h)).inv = eqToHom (by simp) := by aesop_cat)
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
    (map₂_left_unitor : ∀ {b₀ b₁ : B} (f : b₀ ⟶ b₁),
                                                                                       /-
                                                                                         B : Type u_1
                                                                                         C : Type u_2
                                                                                         inst✝² : CategoryTheory.Bicategory B
                                                                                         inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
                                                                                         inst✝ : CategoryTheory.Bicategory C
                                                                                         obj : B → C
                                                                                         map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
                                                                                         mapId : (b : B) → CategoryTheory.Iso (map (CategoryTheory.CategoryStruct.id b) …
                                                                                         mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → C …
                                                                                         map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B} (f : Quiver.Hom b₀ b₁) (g : Q …
                                                                                         b₀ b₁ : B
                                                                                         f : Quiver.Hom b₀ b₁
                                                                                         ⊢ Eq (map (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.i …
                                                                                       -/
      (mapComp (𝟙 b₀) f).hom ≫ (mapId b₀).hom ▷ map f ≫ (λ_ (map f)).hom = eqToHom (by simp) := by
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
        aesop_cat)
    (map₂_right_unitor : ∀ {b₀ b₁ : B} (f : b₀ ⟶ b₁),
                                                                                       /-
                                                                                         B : Type u_1
                                                                                         C : Type u_2
                                                                                         inst✝² : CategoryTheory.Bicategory B
                                                                                         inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
                                                                                         inst✝ : CategoryTheory.Bicategory C
                                                                                         obj : B → C
                                                                                         map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
                                                                                         mapId : (b : B) → CategoryTheory.Iso (map (CategoryTheory.CategoryStruct.id b) …
                                                                                         mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → C …
                                                                                         map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B} (f : Quiver.Hom b₀ b₁) (g : Q …
                                                                                         map₂_left_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Catego …
                                                                                         b₀ b₁ : B
                                                                                         f : Quiver.Hom b₀ b₁
                                                                                         ⊢ Eq (map (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct …
                                                                                       -/
      (mapComp f (𝟙 b₁)).hom ≫ map f ◁ (mapId b₁).hom ≫ (ρ_ (map f)).hom = eqToHom (by simp) := by
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
        aesop_cat) :
    Pseudofunctor B C where
  obj := obj
  map := map
  map₂ φ := eqToHom (by
    /-
      B : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Bicategory B
      inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
      inst✝ : CategoryTheory.Bicategory C
      obj : B → C
      map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
      mapId : (b : B) → CategoryTheory.Iso (map (CategoryTheory.CategoryStruct.id b) …
      mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → C …
      map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B} (f : Quiver.Hom b₀ b₁) (g : Q …
      map₂_left_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Catego …
      map₂_right_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Categ …
      a✝ b✝ : B
      f✝ g✝ : Quiver.Hom a✝ b✝
      φ : Quiver.Hom f✝ g✝
      ⊢ Eq ({ obj := obj, map := fun {X Y} => map }.map f✝) ({ obj := obj, map := fu …
    -/
    obtain rfl := obj_ext_of_isDiscrete φ
    /-
      B : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Bicategory B
      inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
      inst✝ : CategoryTheory.Bicategory C
      obj : B → C
      map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
      mapId : (b : B) → CategoryTheory.Iso (map (CategoryTheory.CategoryStruct.id b) …
      mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → C …
      map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B} (f : Quiver.Hom b₀ b₁) (g : Q …
      map₂_left_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Catego …
      map₂_right_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Categ …
      a✝ b✝ : B
      f✝ : Quiver.Hom a✝ b✝
      φ : Quiver.Hom f✝ f✝
      ⊢ Eq ({ obj := obj, map := fun {X Y} => map }.map f✝) ({ obj := obj, map := fu …
    -/
    dsimp)
    /-
      🎉 no goals
    -/
  mapId := mapId
  mapComp := mapComp
  map₂_whisker_left _ _ _ η := by
    /-
      B : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Bicategory B
      inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
      inst✝ : CategoryTheory.Bicategory C
      obj : B → C
      map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
      mapId : (b : B) → CategoryTheory.Iso (map (CategoryTheory.CategoryStruct.id b) …
      mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → C …
      map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B} (f : Quiver.Hom b₀ b₁) (g : Q …
      map₂_left_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Catego …
      map₂_right_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Categ …
      a✝ b✝ c✝ : B
      x✝² : Quiver.Hom a✝ b✝
      x✝¹ x✝ : Quiver.Hom b✝ c✝
      η : Quiver.Hom x✝¹ x✝
      ⊢ Eq ({ obj := obj, map := fun {X Y} => map, map₂ := fun {a b} {f g} φ => Cate …
    -/
    obtain rfl := obj_ext_of_isDiscrete η
    /-
      B : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Bicategory B
      inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
      inst✝ : CategoryTheory.Bicategory C
      obj : B → C
      map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
      mapId : (b : B) → CategoryTheory.Iso (map (CategoryTheory.CategoryStruct.id b) …
      mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → C …
      map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B} (f : Quiver.Hom b₀ b₁) (g : Q …
      map₂_left_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Catego …
      map₂_right_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Categ …
      a✝ b✝ c✝ : B
      x✝¹ : Quiver.Hom a✝ b✝
      x✝ : Quiver.Hom b✝ c✝
      η : Quiver.Hom x✝ x✝
      ⊢ Eq ({ obj := obj, map := fun {X Y} => map, map₂ := fun {a b} {f g} φ => Cate …
    -/
    simp
    /-
      🎉 no goals
    -/
  map₂_whisker_right η _ := by
    /-
      B : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Bicategory B
      inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
      inst✝ : CategoryTheory.Bicategory C
      obj : B → C
      map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
      mapId : (b : B) → CategoryTheory.Iso (map (CategoryTheory.CategoryStruct.id b) …
      mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → C …
      map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B} (f : Quiver.Hom b₀ b₁) (g : Q …
      map₂_left_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Catego …
      map₂_right_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Categ …
      a✝ b✝ c✝ : B
      f✝ g✝ : Quiver.Hom a✝ b✝
      η : Quiver.Hom f✝ g✝
      x✝ : Quiver.Hom b✝ c✝
      ⊢ Eq ({ obj := obj, map := fun {X Y} => map, map₂ := fun {a b} {f g} φ => Cate …
    -/
    obtain rfl := obj_ext_of_isDiscrete η
    /-
      B : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Bicategory B
      inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
      inst✝ : CategoryTheory.Bicategory C
      obj : B → C
      map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
      mapId : (b : B) → CategoryTheory.Iso (map (CategoryTheory.CategoryStruct.id b) …
      mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → C …
      map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B} (f : Quiver.Hom b₀ b₁) (g : Q …
      map₂_left_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Catego …
      map₂_right_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Categ …
      a✝ b✝ c✝ : B
      f✝ : Quiver.Hom a✝ b✝
      x✝ : Quiver.Hom b✝ c✝
      η : Quiver.Hom f✝ f✝
      ⊢ Eq ({ obj := obj, map := fun {X Y} => map, map₂ := fun {a b} {f g} φ => Cate …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Constructor for oplax functors from a locally discrete bicategory. In that
case, we do not need to provide the `map₂` field of oplax functors. -/
@[simps obj map mapId mapComp]
def oplaxFunctorOfIsLocallyDiscrete
    {B C : Type*} [Bicategory B] [IsLocallyDiscrete B] [Bicategory C]
    (obj : B → C)
    (map : ∀ {b b' : B}, (b ⟶ b') → (obj b ⟶ obj b'))
    (mapId : ∀ (b : B), map (𝟙 b) ⟶ 𝟙 _)
    (mapComp : ∀ {b₀ b₁ b₂ : B} (f : b₀ ⟶ b₁) (g : b₁ ⟶ b₂), map (f ≫ g) ⟶ map f ≫ map g)
    (map₂_associator : ∀ {b₀ b₁ b₂ b₃ : B} (f : b₀ ⟶ b₁) (g : b₁ ⟶ b₂) (h : b₂ ⟶ b₃),
                  /-
                    B : Type u_1
                    C : Type u_2
                    inst✝² : CategoryTheory.Bicategory B
                    inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
                    inst✝ : CategoryTheory.Bicategory C
                    obj : B → C
                    map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
                    mapId : (b : B) → Quiver.Hom (map (CategoryTheory.CategoryStruct.id b)) (Categ …
                    mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → Q …
                    b₀ b₁ b₂ b₃ : B
                    f : Quiver.Hom b₀ b₁
                    g : Quiver.Hom b₁ b₂
                    h : Quiver.Hom b₂ b₃
                    ⊢ Eq (map (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
                  -/
      eqToHom (by simp) ≫ mapComp f (g ≫ h) ≫ map f ◁ mapComp g h =
                  /-
                    🎉 no goals
                  -/
        mapComp (f ≫ g) h ≫ mapComp f g ▷ map h ≫ (α_ (map f) (map g) (map h)).hom := by
          aesop_cat)
    (map₂_left_unitor : ∀ {b₀ b₁ : B} (f : b₀ ⟶ b₁),
                                                                           /-
                                                                             B : Type u_1
                                                                             C : Type u_2
                                                                             inst✝² : CategoryTheory.Bicategory B
                                                                             inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
                                                                             inst✝ : CategoryTheory.Bicategory C
                                                                             obj : B → C
                                                                             map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
                                                                             mapId : (b : B) → Quiver.Hom (map (CategoryTheory.CategoryStruct.id b)) (Categ …
                                                                             mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → Q …
                                                                             map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B} (f : Quiver.Hom b₀ b₁) (g : Q …
                                                                             b₀ b₁ : B
                                                                             f : Quiver.Hom b₀ b₁
                                                                             ⊢ Eq (map (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.i …
                                                                           -/
      mapComp (𝟙 b₀) f ≫ mapId b₀ ▷ map f ≫ (λ_ (map f)).hom = eqToHom (by simp) := by
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
        aesop_cat)
    (map₂_right_unitor : ∀ {b₀ b₁ : B} (f : b₀ ⟶ b₁),
                                                                           /-
                                                                             B : Type u_1
                                                                             C : Type u_2
                                                                             inst✝² : CategoryTheory.Bicategory B
                                                                             inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
                                                                             inst✝ : CategoryTheory.Bicategory C
                                                                             obj : B → C
                                                                             map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
                                                                             mapId : (b : B) → Quiver.Hom (map (CategoryTheory.CategoryStruct.id b)) (Categ …
                                                                             mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → Q …
                                                                             map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B} (f : Quiver.Hom b₀ b₁) (g : Q …
                                                                             map₂_left_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Catego …
                                                                             b₀ b₁ : B
                                                                             f : Quiver.Hom b₀ b₁
                                                                             ⊢ Eq (map (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct …
                                                                           -/
      mapComp f (𝟙 b₁) ≫ map f ◁ mapId b₁ ≫ (ρ_ (map f)).hom = eqToHom (by simp) := by
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
        aesop_cat) :
    OplaxFunctor B C where
  obj := obj
  map := map
  map₂ φ := eqToHom (by
    /-
      B : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Bicategory B
      inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
      inst✝ : CategoryTheory.Bicategory C
      obj : B → C
      map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
      mapId : (b : B) → Quiver.Hom (map (CategoryTheory.CategoryStruct.id b)) (Categ …
      mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → Q …
      map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B} (f : Quiver.Hom b₀ b₁) (g : Q …
      map₂_left_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Catego …
      map₂_right_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Categ …
      a✝ b✝ : B
      f✝ g✝ : Quiver.Hom a✝ b✝
      φ : Quiver.Hom f✝ g✝
      ⊢ Eq ({ obj := obj, map := fun {X Y} => map }.map f✝) ({ obj := obj, map := fu …
    -/
    obtain rfl := obj_ext_of_isDiscrete φ
    /-
      B : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Bicategory B
      inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
      inst✝ : CategoryTheory.Bicategory C
      obj : B → C
      map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
      mapId : (b : B) → Quiver.Hom (map (CategoryTheory.CategoryStruct.id b)) (Categ …
      mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → Q …
      map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B} (f : Quiver.Hom b₀ b₁) (g : Q …
      map₂_left_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Catego …
      map₂_right_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Categ …
      a✝ b✝ : B
      f✝ : Quiver.Hom a✝ b✝
      φ : Quiver.Hom f✝ f✝
      ⊢ Eq ({ obj := obj, map := fun {X Y} => map }.map f✝) ({ obj := obj, map := fu …
    -/
    dsimp)
    /-
      🎉 no goals
    -/
  mapId := mapId
  mapComp := mapComp
  mapComp_naturality_left η := by
    /-
      B : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Bicategory B
      inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
      inst✝ : CategoryTheory.Bicategory C
      obj : B → C
      map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
      mapId : (b : B) → Quiver.Hom (map (CategoryTheory.CategoryStruct.id b)) (Categ …
      mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → Q …
      map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B} (f : Quiver.Hom b₀ b₁) (g : Q …
      map₂_left_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Catego …
      map₂_right_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Categ …
      a✝ b✝ c✝ : B
      f✝ f'✝ : Quiver.Hom a✝ b✝
      η : Quiver.Hom f✝ f'✝
      ⊢ ∀ (g : Quiver.Hom b✝ c✝), Eq (CategoryTheory.CategoryStruct.comp ({ obj := o …
    -/
    obtain rfl := obj_ext_of_isDiscrete η
    /-
      B : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Bicategory B
      inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
      inst✝ : CategoryTheory.Bicategory C
      obj : B → C
      map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
      mapId : (b : B) → Quiver.Hom (map (CategoryTheory.CategoryStruct.id b)) (Categ …
      mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → Q …
      map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B} (f : Quiver.Hom b₀ b₁) (g : Q …
      map₂_left_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Catego …
      map₂_right_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Categ …
      a✝ b✝ c✝ : B
      f✝ : Quiver.Hom a✝ b✝
      η : Quiver.Hom f✝ f✝
      ⊢ ∀ (g : Quiver.Hom b✝ c✝), Eq (CategoryTheory.CategoryStruct.comp ({ obj := o …
    -/
    simp
    /-
      🎉 no goals
    -/
  mapComp_naturality_right _ _ _ η := by
    /-
      B : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Bicategory B
      inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
      inst✝ : CategoryTheory.Bicategory C
      obj : B → C
      map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
      mapId : (b : B) → Quiver.Hom (map (CategoryTheory.CategoryStruct.id b)) (Categ …
      mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → Q …
      map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B} (f : Quiver.Hom b₀ b₁) (g : Q …
      map₂_left_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Catego …
      map₂_right_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Categ …
      a✝ b✝ c✝ : B
      x✝² : Quiver.Hom a✝ b✝
      x✝¹ x✝ : Quiver.Hom b✝ c✝
      η : Quiver.Hom x✝¹ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ({ obj := obj, map := fun {X Y} => ma …
    -/
    obtain rfl := obj_ext_of_isDiscrete η
    /-
      B : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Bicategory B
      inst✝¹ : CategoryTheory.Bicategory.IsLocallyDiscrete B
      inst✝ : CategoryTheory.Bicategory C
      obj : B → C
      map : {b b' : B} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
      mapId : (b : B) → Quiver.Hom (map (CategoryTheory.CategoryStruct.id b)) (Categ …
      mapComp : {b₀ b₁ b₂ : B} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) → Q …
      map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B} (f : Quiver.Hom b₀ b₁) (g : Q …
      map₂_left_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Catego …
      map₂_right_unitor : autoParam (∀ {b₀ b₁ : B} (f : Quiver.Hom b₀ b₁), Eq (Categ …
      a✝ b✝ c✝ : B
      x✝¹ : Quiver.Hom a✝ b✝
      x✝ : Quiver.Hom b✝ c✝
      η : Quiver.Hom x✝ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ({ obj := obj, map := fun {X Y} => ma …
    -/
    simp
    /-
      🎉 no goals
    -/


/--
If `B` is a strict bicategory and `I` is a (1-)category, any functor (of 1-categories) `I ⥤ B` can
be promoted to a pseudofunctor from `LocallyDiscrete I` to `B`.
-/
@[simps! obj map mapId mapComp]
def Functor.toPseudoFunctor : Pseudofunctor (LocallyDiscrete I) B :=
  /-
    I : Type u_1
    B : Type u_2
    inst✝² : CategoryTheory.Category.{?u.82508, u_1} I
    inst✝¹ : CategoryTheory.Bicategory B
    inst✝ : CategoryTheory.Bicategory.Strict B
    F : CategoryTheory.Functor I B
    ⊢ ∀ {b₀ b₁ b₂ b₃ : CategoryTheory.LocallyDiscrete I} (f : Quiver.Hom b₀ b₁) (g …
  -/
  /-
    🎉 no goals
  -/
                           /-
                             I : Type u_1
                             B : Type u_2
                             inst✝² : CategoryTheory.Category.{?u.82508, u_1} I
                             inst✝¹ : CategoryTheory.Bicategory B
                             inst✝ : CategoryTheory.Bicategory.Strict B
                             F : CategoryTheory.Functor I B
                             x✝ : CategoryTheory.LocallyDiscrete I
                             X : I
                             ⊢ Eq ((fun {b b'} x => CategoryTheory.Functor.toPseudoFunctor.match_2 (fun x = …
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
  pseudofunctorOfIsLocallyDiscrete
  /-
    🎉 no goals
  -/
    (fun ⟨X⟩ ↦ F.obj X) (fun ⟨f⟩ ↦ F.map f)
    (fun ⟨X⟩ ↦ eqToIso (by simp)) (fun f g ↦ eqToIso (by simp))


/--
If `B` is a strict bicategory and `I` is a (1-)category, any functor (of 1-categories) `I ⥤ B` can
be promoted to an oplax functor from `LocallyDiscrete I` to `B`.
-/
@[simps! obj map mapId mapComp]
def Functor.toOplaxFunctor : OplaxFunctor (LocallyDiscrete I) B :=
  /-
    I : Type u_1
    B : Type u_2
    inst✝² : CategoryTheory.Category.{?u.116800, u_1} I
    inst✝¹ : CategoryTheory.Bicategory B
    inst✝ : CategoryTheory.Bicategory.Strict B
    F : CategoryTheory.Functor I B
    ⊢ ∀ {b₀ b₁ b₂ b₃ : CategoryTheory.LocallyDiscrete I} (f : Quiver.Hom b₀ b₁) (g …
  -/
  /-
    🎉 no goals
  -/
                           /-
                             I : Type u_1
                             B : Type u_2
                             inst✝² : CategoryTheory.Category.{?u.116800, u_1} I
                             inst✝¹ : CategoryTheory.Bicategory B
                             inst✝ : CategoryTheory.Bicategory.Strict B
                             F : CategoryTheory.Functor I B
                             x✝ : CategoryTheory.LocallyDiscrete I
                             X : I
                             ⊢ Eq ((fun {b b'} x => CategoryTheory.Functor.toPseudoFunctor.match_2 (fun x = …
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
  oplaxFunctorOfIsLocallyDiscrete
  /-
    🎉 no goals
  -/
    (fun ⟨X⟩ ↦ F.obj X) (fun ⟨f⟩ ↦ F.map f)
    (fun ⟨X⟩ ↦ eqToHom (by simp)) (fun f g ↦ eqToHom (by simp))


/-- Constructor for pseudofunctors from a locally discrete bicategory. In that
case, we do not need to provide the `map₂` field of pseudofunctors. -/
@[simps! obj map mapId mapComp]
def mkPseudofunctor {B₀ C : Type*} [Category B₀] [Bicategory C]
    (obj : B₀ → C)
    (map : ∀ {b b' : B₀}, (b ⟶ b') → (obj b ⟶ obj b'))
    (mapId : ∀ (b : B₀), map (𝟙 b) ≅ 𝟙 _)
    (mapComp : ∀ {b₀ b₁ b₂ : B₀} (f : b₀ ⟶ b₁) (g : b₁ ⟶ b₂), map (f ≫ g) ≅ map f ≫ map g)
    (map₂_associator : ∀ {b₀ b₁ b₂ b₃ : B₀} (f : b₀ ⟶ b₁) (g : b₁ ⟶ b₂) (h : b₂ ⟶ b₃),
      (mapComp (f ≫ g) h).hom ≫
        (mapComp f g).hom ▷ map h ≫ (α_ (map f) (map g) (map h)).hom ≫
                                                                            /-
                                                                              B₀ : Type u_1
                                                                              C : Type u_2
                                                                              inst✝¹ : CategoryTheory.Category.{?u.145943, u_1} B₀
                                                                              inst✝ : CategoryTheory.Bicategory C
                                                                              obj : B₀ → C
                                                                              map : {b b' : B₀} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
                                                                              mapId : (b : B₀) → CategoryTheory.Iso (map (CategoryTheory.CategoryStruct.id b …
                                                                              mapComp : {b₀ b₁ b₂ : B₀} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) →  …
                                                                              b₀ b₁ b₂ b₃ : B₀
                                                                              f : Quiver.Hom b₀ b₁
                                                                              g : Quiver.Hom b₁ b₂
                                                                              h : Quiver.Hom b₂ b₃
                                                                              ⊢ Eq (map (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
                                                                            -/
          map f ◁ (mapComp g h).inv ≫ (mapComp f (g ≫ h)).inv = eqToHom (by simp) := by aesop_cat)
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
    (map₂_left_unitor : ∀ {b₀ b₁ : B₀} (f : b₀ ⟶ b₁),
                                                                                       /-
                                                                                         B₀ : Type u_1
                                                                                         C : Type u_2
                                                                                         inst✝¹ : CategoryTheory.Category.{?u.145943, u_1} B₀
                                                                                         inst✝ : CategoryTheory.Bicategory C
                                                                                         obj : B₀ → C
                                                                                         map : {b b' : B₀} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
                                                                                         mapId : (b : B₀) → CategoryTheory.Iso (map (CategoryTheory.CategoryStruct.id b …
                                                                                         mapComp : {b₀ b₁ b₂ : B₀} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) →  …
                                                                                         map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B₀} (f : Quiver.Hom b₀ b₁) (g :  …
                                                                                         b₀ b₁ : B₀
                                                                                         f : Quiver.Hom b₀ b₁
                                                                                         ⊢ Eq (map (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.i …
                                                                                       -/
      (mapComp (𝟙 b₀) f).hom ≫ (mapId b₀).hom ▷ map f ≫ (λ_ (map f)).hom = eqToHom (by simp) := by
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
        aesop_cat)
    (map₂_right_unitor : ∀ {b₀ b₁ : B₀} (f : b₀ ⟶ b₁),
                                                                                       /-
                                                                                         B₀ : Type u_1
                                                                                         C : Type u_2
                                                                                         inst✝¹ : CategoryTheory.Category.{?u.145943, u_1} B₀
                                                                                         inst✝ : CategoryTheory.Bicategory C
                                                                                         obj : B₀ → C
                                                                                         map : {b b' : B₀} → Quiver.Hom b b' → Quiver.Hom (obj b) (obj b')
                                                                                         mapId : (b : B₀) → CategoryTheory.Iso (map (CategoryTheory.CategoryStruct.id b …
                                                                                         mapComp : {b₀ b₁ b₂ : B₀} → (f : Quiver.Hom b₀ b₁) → (g : Quiver.Hom b₁ b₂) →  …
                                                                                         map₂_associator : autoParam (∀ {b₀ b₁ b₂ b₃ : B₀} (f : Quiver.Hom b₀ b₁) (g :  …
                                                                                         map₂_left_unitor : autoParam (∀ {b₀ b₁ : B₀} (f : Quiver.Hom b₀ b₁), Eq (Categ …
                                                                                         b₀ b₁ : B₀
                                                                                         f : Quiver.Hom b₀ b₁
                                                                                         ⊢ Eq (map (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct …
                                                                                       -/
      (mapComp f (𝟙 b₁)).hom ≫ map f ◁ (mapId b₁).hom ≫ (ρ_ (map f)).hom = eqToHom (by simp) := by
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
        aesop_cat) :
    Pseudofunctor (LocallyDiscrete B₀) C :=
  pseudofunctorOfIsLocallyDiscrete (fun b ↦ obj b.as) (fun f ↦ map f.as)
    (fun _ ↦ mapId _) (fun _ _ ↦ mapComp _ _) (fun _ _ _ ↦ map₂_associator _ _ _)
    (fun _ ↦ map₂_left_unitor _) (fun _ ↦ map₂_right_unitor _)


