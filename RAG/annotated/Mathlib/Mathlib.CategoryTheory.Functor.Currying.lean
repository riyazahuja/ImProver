/-- The uncurrying functor, taking a functor `C ⥤ (D ⥤ E)` and producing a functor `(C × D) ⥤ E`.
-/
@[simps]
def uncurry : (C ⥤ D ⥤ E) ⥤ C × D ⥤ E where
  obj F :=
    { obj := fun X => (F.obj X.1).obj X.2
      map := fun {X} {Y} f => (F.map f.1).app X.2 ≫ (F.obj Y.1).map f.2
      map_comp := fun f g => by
        simp only [prod_comp_fst, prod_comp_snd, Functor.map_comp, NatTrans.comp_app,
          Category.assoc]
        /-
          B : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
          C : Type u₂
          inst✝³ : CategoryTheory.Category.{v₂, u₂} C
          D : Type u₃
          inst✝² : CategoryTheory.Category.{v₃, u₃} D
          E : Type u₄
          inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
          H : Type u₅
          inst✝ : CategoryTheory.Category.{v₅, u₅} H
          F : CategoryTheory.Functor C (CategoryTheory.Functor D E)
          X✝ Y✝ Z✝ : Prod C D
          f : Quiver.Hom X✝ Y✝
          g : Quiver.Hom Y✝ Z✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map f.1).app X✝.2) (CategoryTheor …
        -/
        slice_lhs 2 3 => rw [← NatTrans.naturality]
        /-
          B : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
          C : Type u₂
          inst✝³ : CategoryTheory.Category.{v₂, u₂} C
          D : Type u₃
          inst✝² : CategoryTheory.Category.{v₃, u₃} D
          E : Type u₄
          inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
          H : Type u₅
          inst✝ : CategoryTheory.Category.{v₅, u₅} H
          F : CategoryTheory.Functor C (CategoryTheory.Functor D E)
          X✝ Y✝ Z✝ : Prod C D
          f : Quiver.Hom X✝ Y✝
          g : Quiver.Hom Y✝ Z✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map f.1).app X✝.2) (CategoryTheor …
        -/
        rw [Category.assoc] }
        /-
          🎉 no goals
        -/
  map T :=
    { app := fun X => (T.app X.1).app X.2
      naturality := fun X Y f => by
        simp only [prod_comp_fst, prod_comp_snd, Category.comp_id, Category.assoc, Functor.map_id,
          Functor.map_comp, NatTrans.id_app, NatTrans.comp_app]
        /-
          B : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
          C : Type u₂
          inst✝³ : CategoryTheory.Category.{v₂, u₂} C
          D : Type u₃
          inst✝² : CategoryTheory.Category.{v₃, u₃} D
          E : Type u₄
          inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
          H : Type u₅
          inst✝ : CategoryTheory.Category.{v₅, u₅} H
          X✝ Y✝ : CategoryTheory.Functor C (CategoryTheory.Functor D E)
          T : Quiver.Hom X✝ Y✝
          X Y : Prod C D
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((X✝.map f.1).app X.2) (CategoryTheor …
        -/
        slice_lhs 2 3 => rw [NatTrans.naturality]
        /-
          B : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
          C : Type u₂
          inst✝³ : CategoryTheory.Category.{v₂, u₂} C
          D : Type u₃
          inst✝² : CategoryTheory.Category.{v₃, u₃} D
          E : Type u₄
          inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
          H : Type u₅
          inst✝ : CategoryTheory.Category.{v₅, u₅} H
          X✝ Y✝ : CategoryTheory.Functor C (CategoryTheory.Functor D E)
          T : Quiver.Hom X✝ Y✝
          X Y : Prod C D
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((X✝.map f.1).app X.2) (CategoryTheor …
        -/
        slice_lhs 1 2 => rw [← NatTrans.comp_app, NatTrans.naturality, NatTrans.comp_app]
        /-
          B : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
          C : Type u₂
          inst✝³ : CategoryTheory.Category.{v₂, u₂} C
          D : Type u₃
          inst✝² : CategoryTheory.Category.{v₃, u₃} D
          E : Type u₄
          inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
          H : Type u₅
          inst✝ : CategoryTheory.Category.{v₅, u₅} H
          X✝ Y✝ : CategoryTheory.Functor C (CategoryTheory.Functor D E)
          T : Quiver.Hom X✝ Y✝
          X Y : Prod C D
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        rw [Category.assoc] }
        /-
          🎉 no goals
        -/


/-- The object level part of the currying functor. (See `curry` for the functorial version.)
-/
def curryObj (F : C × D ⥤ E) : C ⥤ D ⥤ E where
  obj X :=
    { obj := fun Y => F.obj (X, Y)
      map := fun g => F.map (𝟙 X, g)
                            /-
                              B : Type u₁
                              inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
                              C : Type u₂
                              inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                              D : Type u₃
                              inst✝² : CategoryTheory.Category.{v₃, u₃} D
                              E : Type u₄
                              inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
                              H : Type u₅
                              inst✝ : CategoryTheory.Category.{v₅, u₅} H
                              F : CategoryTheory.Functor (Prod C D) E
                              X : C
                              Y : D
                              ⊢ Eq ({ obj := fun Y => F.obj { fst := X, snd := Y }, map := fun {X_1 Y} g =>  …
                            -/
      map_id := fun Y => by simp only [F.map_id]; rw [← prod_id]; exact F.map_id ⟨X,Y⟩
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
                                /-
                                  B : Type u₁
                                  inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
                                  C : Type u₂
                                  inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                                  D : Type u₃
                                  inst✝² : CategoryTheory.Category.{v₃, u₃} D
                                  E : Type u₄
                                  inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
                                  H : Type u₅
                                  inst✝ : CategoryTheory.Category.{v₅, u₅} H
                                  F : CategoryTheory.Functor (Prod C D) E
                                  X : C
                                  X✝ Y✝ Z✝ : D
                                  f : Quiver.Hom X✝ Y✝
                                  g : Quiver.Hom Y✝ Z✝
                                  ⊢ Eq ({ obj := fun Y => F.obj { fst := X, snd := Y }, map := fun {X_1 Y} g =>  …
                                -/
      map_comp := fun f g => by simp [← F.map_comp]}
                                /-
                                  🎉 no goals
                                -/
  map f :=
    { app := fun Y => F.map (f, 𝟙 Y)
                                         /-
                                           B : Type u₁
                                           inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
                                           C : Type u₂
                                           inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                                           D : Type u₃
                                           inst✝² : CategoryTheory.Category.{v₃, u₃} D
                                           E : Type u₄
                                           inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
                                           H : Type u₅
                                           inst✝ : CategoryTheory.Category.{v₅, u₅} H
                                           F : CategoryTheory.Functor (Prod C D) E
                                           X✝ Y✝ : C
                                           f : Quiver.Hom X✝ Y✝
                                           Y Y' : D
                                           g : Quiver.Hom Y Y'
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun X => { obj := fun Y => F.obj { …
                                         -/
      naturality := fun {Y} {Y'} g => by simp [← F.map_comp] }
                                         /-
                                           🎉 no goals
                                         -/
                        /-
                          B : Type u₁
                          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
                          C : Type u₂
                          inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                          D : Type u₃
                          inst✝² : CategoryTheory.Category.{v₃, u₃} D
                          E : Type u₄
                          inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
                          H : Type u₅
                          inst✝ : CategoryTheory.Category.{v₅, u₅} H
                          F : CategoryTheory.Functor (Prod C D) E
                          X : C
                          ⊢ Eq ({ obj := fun X => { obj := fun Y => F.obj { fst := X, snd := Y }, map := …
                        -/
  map_id := fun X => by ext Y; exact F.map_id _
                               /-
                                 🎉 no goals
                               -/
                            /-
                              B : Type u₁
                              inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
                              C : Type u₂
                              inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                              D : Type u₃
                              inst✝² : CategoryTheory.Category.{v₃, u₃} D
                              E : Type u₄
                              inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
                              H : Type u₅
                              inst✝ : CategoryTheory.Category.{v₅, u₅} H
                              F : CategoryTheory.Functor (Prod C D) E
                              X✝ Y✝ Z✝ : C
                              f : Quiver.Hom X✝ Y✝
                              g : Quiver.Hom Y✝ Z✝
                              ⊢ Eq ({ obj := fun X => { obj := fun Y => F.obj { fst := X, snd := Y }, map := …
                            -/
  map_comp := fun f g => by ext Y; dsimp; simp [← F.map_comp]
                                          /-
                                            🎉 no goals
                                          -/


/-- The currying functor, taking a functor `(C × D) ⥤ E` and producing a functor `C ⥤ (D ⥤ E)`.
-/
@[simps! obj_obj_obj obj_obj_map obj_map_app map_app_app]
def curry : (C × D ⥤ E) ⥤ C ⥤ D ⥤ E where
  obj F := curryObj F
  map T :=
    { app := fun X =>
        { app := fun Y => T.app (X, Y)
          naturality := fun Y Y' g => by
            /-
              B : Type u₁
              inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
              C : Type u₂
              inst✝³ : CategoryTheory.Category.{v₂, u₂} C
              D : Type u₃
              inst✝² : CategoryTheory.Category.{v₃, u₃} D
              E : Type u₄
              inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
              H : Type u₅
              inst✝ : CategoryTheory.Category.{v₅, u₅} H
              X✝ Y✝ : CategoryTheory.Functor (Prod C D) E
              T : Quiver.Hom X✝ Y✝
              X : C
              Y Y' : D
              g : Quiver.Hom Y Y'
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((fun F => CategoryTheory.curryObj  …
            -/
            dsimp [curryObj]
            /-
              B : Type u₁
              inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
              C : Type u₂
              inst✝³ : CategoryTheory.Category.{v₂, u₂} C
              D : Type u₃
              inst✝² : CategoryTheory.Category.{v₃, u₃} D
              E : Type u₄
              inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
              H : Type u₅
              inst✝ : CategoryTheory.Category.{v₅, u₅} H
              X✝ Y✝ : CategoryTheory.Functor (Prod C D) E
              T : Quiver.Hom X✝ Y✝
              X : C
              Y Y' : D
              g : Quiver.Hom Y Y'
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (X✝.map { fst := CategoryTheory.Categ …
            -/
            rw [NatTrans.naturality] }
            /-
              🎉 no goals
            -/
      naturality := fun X X' f => by
        /-
          B : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
          C : Type u₂
          inst✝³ : CategoryTheory.Category.{v₂, u₂} C
          D : Type u₃
          inst✝² : CategoryTheory.Category.{v₃, u₃} D
          E : Type u₄
          inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
          H : Type u₅
          inst✝ : CategoryTheory.Category.{v₅, u₅} H
          X✝ Y✝ : CategoryTheory.Functor (Prod C D) E
          T : Quiver.Hom X✝ Y✝
          X X' : C
          f : Quiver.Hom X X'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun F => CategoryTheory.curryObj F …
        -/
        ext; dsimp [curryObj]
        /-
          case w.h
          B : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
          C : Type u₂
          inst✝³ : CategoryTheory.Category.{v₂, u₂} C
          D : Type u₃
          inst✝² : CategoryTheory.Category.{v₃, u₃} D
          E : Type u₄
          inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
          H : Type u₅
          inst✝ : CategoryTheory.Category.{v₅, u₅} H
          X✝ Y✝ : CategoryTheory.Functor (Prod C D) E
          T : Quiver.Hom X✝ Y✝
          X X' : C
          f : Quiver.Hom X X'
          x✝ : D
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X✝.map { fst := f, snd := CategoryTh …
        -/
        rw [NatTrans.naturality] }
        /-
          🎉 no goals
        -/

-- create projection simp lemmas even though this isn't a `{ .. }`.

/-- The equivalence of functor categories given by currying/uncurrying.
-/
@[simps!]
def currying : C ⥤ D ⥤ E ≌ C × D ⥤ E where
  functor := uncurry
  inverse := curry
                                          /-
                                            B : Type u₁
                                            inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
                                            C : Type u₂
                                            inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                                            D : Type u₃
                                            inst✝² : CategoryTheory.Category.{v₃, u₃} D
                                            E : Type u₄
                                            inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
                                            H : Type u₅
                                            inst✝ : CategoryTheory.Category.{v₅, u₅} H
                                            x✝ : CategoryTheory.Functor C (CategoryTheory.Functor D E)
                                            ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
                                          -/
             /-
               B : Type u₁
               inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
               C : Type u₂
               inst✝³ : CategoryTheory.Category.{v₂, u₂} C
               D : Type u₃
               inst✝² : CategoryTheory.Category.{v₃, u₃} D
               E : Type u₄
               inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
               H : Type u₅
               inst✝ : CategoryTheory.Category.{v₅, u₅} H
               x✝¹ : CategoryTheory.Functor C (CategoryTheory.Functor D E)
               x✝ : C
               ⊢ ∀ {X Y : D} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
             -/
                                          /-
                                            🎉 no goals
                                          -/
             /-
               🎉 no goals
             -/
  unitIso := NatIso.ofComponents (fun _ ↦ NatIso.ofComponents
             /-
               🎉 no goals
             -/
    (fun _ ↦ NatIso.ofComponents (fun _ ↦ Iso.refl _)))
               /-
                 B : Type u₁
                 inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
                 C : Type u₂
                 inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                 D : Type u₃
                 inst✝² : CategoryTheory.Category.{v₃, u₃} D
                 E : Type u₄
                 inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
                 H : Type u₅
                 inst✝ : CategoryTheory.Category.{v₅, u₅} H
                 ⊢ ∀ {X Y : CategoryTheory.Functor (Prod C D) E} (f : Quiver.Hom X Y), Eq (Cate …
               -/
  counitIso := NatIso.ofComponents
      /-
        B : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
        C : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} C
        D : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} D
        E : Type u₄
        inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
        H : Type u₅
        inst✝ : CategoryTheory.Category.{v₅, u₅} H
        F : CategoryTheory.Functor (Prod C D) E
        ⊢ ∀ {X Y : Prod C D} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.c …
      -/
               /-
                 🎉 no goals
               -/
      /-
        case mk.mk.mk
        B : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
        C : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} C
        D : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} D
        E : Type u₄
        inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
        H : Type u₅
        inst✝ : CategoryTheory.Category.{v₅, u₅} H
        F : CategoryTheory.Functor (Prod C D) E
        X₁ : C
        X₂ : D
        Y₁ : C
        Y₂ : D
        f₁ : Quiver.Hom { fst := X₁, snd := X₂ }.1 { fst := Y₁, snd := Y₂ }.1
        f₂ : Quiver.Hom { fst := X₁, snd := X₂ }.2 { fst := Y₁, snd := Y₂ }.2
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.curry.comp Category …
      -/
    (fun F ↦ NatIso.ofComponents (fun _ ↦ Iso.refl _) (by
      /-
        case mk.mk.mk
        B : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
        C : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} C
        D : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} D
        E : Type u₄
        inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
        H : Type u₅
        inst✝ : CategoryTheory.Category.{v₅, u₅} H
        F : CategoryTheory.Functor (Prod C D) E
        X₁ : C
        X₂ : D
        Y₁ : C
        Y₂ : D
        f₁ : Quiver.Hom X₁ Y₁
        f₂ : Quiver.Hom X₂ Y₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      rintro ⟨X₁, X₂⟩ ⟨Y₁, Y₂⟩ ⟨f₁, f₂⟩
      /-
        🎉 no goals
      -/
      dsimp at f₁ f₂ ⊢
      simp only [← F.map_comp, prod_comp, Category.comp_id, Category.id_comp]))


/-- The functor `uncurry : (C ⥤ D ⥤ E) ⥤ C × D ⥤ E` is fully faithful. -/
def fullyFaithfulUncurry : (uncurry : (C ⥤ D ⥤ E) ⥤ C × D ⥤ E).FullyFaithful :=
  currying.fullyFaithfulFunctor


instance : (uncurry : (C ⥤ D ⥤ E) ⥤ C × D ⥤ E).Full :=
  fullyFaithfulUncurry.full


instance : (uncurry : (C ⥤ D ⥤ E) ⥤ C × D ⥤ E).Faithful :=
  fullyFaithfulUncurry.faithful


/-- Given functors `F₁ : C ⥤ D`, `F₂ : C' ⥤ D'` and `G : D × D' ⥤ E`, this is the isomorphism
between `curry.obj ((F₁.prod F₂).comp G)` and
`F₁ ⋙ curry.obj G ⋙ (whiskeringLeft C' D' E).obj F₂` in the category `C ⥤ C' ⥤ E`. -/
@[simps!]
def curryObjProdComp {C' D' : Type*} [Category C'] [Category D']
    (F₁ : C ⥤ D) (F₂ : C' ⥤ D') (G : D × D' ⥤ E) :
    curry.obj ((F₁.prod F₂).comp G) ≅
      F₁ ⋙ curry.obj G ⋙ (whiskeringLeft C' D' E).obj F₂ :=
                                /-
                                  B : Type u₁
                                  inst✝⁶ : CategoryTheory.Category.{v₁, u₁} B
                                  C : Type u₂
                                  inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
                                  D : Type u₃
                                  inst✝⁴ : CategoryTheory.Category.{v₃, u₃} D
                                  E : Type u₄
                                  inst✝³ : CategoryTheory.Category.{v₄, u₄} E
                                  H : Type u₅
                                  inst✝² : CategoryTheory.Category.{v₅, u₅} H
                                  C' : Type u_1
                                  D' : Type u_2
                                  inst✝¹ : CategoryTheory.Category.{?u.48407, u_1} C'
                                  inst✝ : CategoryTheory.Category.{?u.48411, u_2} D'
                                  F₁ : CategoryTheory.Functor C D
                                  F₂ : CategoryTheory.Functor C' D'
                                  G : CategoryTheory.Functor (Prod D D') E
                                  X₁ : C
                                  ⊢ ∀ {X Y : C'} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (( …
                                -/
                                /-
                                  🎉 no goals
                                -/
  NatIso.ofComponents (fun X₁ ↦ NatIso.ofComponents (fun X₂ ↦ Iso.refl _))
  /-
    🎉 no goals
  -/


/-- `F.flip` is isomorphic to uncurrying `F`, swapping the variables, and currying. -/
@[simps!]
def flipIsoCurrySwapUncurry (F : C ⥤ D ⥤ E) : F.flip ≅ curry.obj (Prod.swap _ _ ⋙ uncurry.obj F) :=
                               /-
                                 B : Type u₁
                                 inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
                                 C : Type u₂
                                 inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                                 D : Type u₃
                                 inst✝² : CategoryTheory.Category.{v₃, u₃} D
                                 E : Type u₄
                                 inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
                                 H : Type u₅
                                 inst✝ : CategoryTheory.Category.{v₅, u₅} H
                                 F : CategoryTheory.Functor C (CategoryTheory.Functor D E)
                                 d : D
                                 ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((F …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun d => NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


/-- The uncurrying of `F.flip` is isomorphic to
swapping the factors followed by the uncurrying of `F`. -/
@[simps!]
def uncurryObjFlip (F : C ⥤ D ⥤ E) : uncurry.obj F.flip ≅ Prod.swap _ _ ⋙ uncurry.obj F :=
  /-
    B : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
    C : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} C
    D : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} D
    E : Type u₄
    inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
    H : Type u₅
    inst✝ : CategoryTheory.Category.{v₅, u₅} H
    F : CategoryTheory.Functor C (CategoryTheory.Functor D E)
    ⊢ ∀ {X Y : Prod D C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.c …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


/-- A version of `CategoryTheory.whiskeringRight` for bifunctors, obtained by uncurrying,
applying `whiskeringRight` and currying back
-/
@[simps!]
def whiskeringRight₂ : (C ⥤ D ⥤ E) ⥤ (B ⥤ C) ⥤ (B ⥤ D) ⥤ B ⥤ E :=
  uncurry ⋙
    whiskeringRight _ _ _ ⋙ (whiskeringLeft _ _ _).obj (prodFunctorToFunctorProd _ _ _) ⋙ curry


lemma uncurry_obj_curry_obj (F : B × C ⥤ D) : uncurry.obj (curry.obj F) = F :=
                  /-
                    B : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} B
                    C : Type u₂
                    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C
                    D : Type u₃
                    inst✝ : CategoryTheory.Category.{v₃, u₃} D
                    F : CategoryTheory.Functor (Prod B C) D
                    ⊢ ∀ (X : Prod B C), Eq ((CategoryTheory.uncurry.obj (CategoryTheory.curry.obj  …
                  -/
  Functor.ext (by simp) (fun ⟨x₁, x₂⟩ ⟨y₁, y₂⟩ ⟨f₁, f₂⟩ => by
                  /-
                    🎉 no goals
                  -/
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} B
      C : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} C
      D : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} D
      F : CategoryTheory.Functor (Prod B C) D
      x✝² x✝¹ : Prod B C
      x₁ : B
      x₂ : C
      y₁ : B
      y₂ : C
      x✝ : Quiver.Hom { fst := x₁, snd := x₂ } { fst := y₁, snd := y₂ }
      f₁ : Quiver.Hom { fst := x₁, snd := x₂ }.1 { fst := y₁, snd := y₂ }.1
      f₂ : Quiver.Hom { fst := x₁, snd := x₂ }.2 { fst := y₁, snd := y₂ }.2
      ⊢ Eq ((CategoryTheory.uncurry.obj (CategoryTheory.curry.obj F)).map { fst := f …
    -/
    dsimp
    /-
      B : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} B
      C : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} C
      D : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} D
      F : CategoryTheory.Functor (Prod B C) D
      x✝² x✝¹ : Prod B C
      x₁ : B
      x₂ : C
      y₁ : B
      y₂ : C
      x✝ : Quiver.Hom { fst := x₁, snd := x₂ } { fst := y₁, snd := y₂ }
      f₁ : Quiver.Hom { fst := x₁, snd := x₂ }.1 { fst := y₁, snd := y₂ }.1
      f₂ : Quiver.Hom { fst := x₁, snd := x₂ }.2 { fst := y₁, snd := y₂ }.2
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map { fst := f₁, snd := CategoryTh …
    -/
    simp only [← F.map_comp, Category.id_comp, Category.comp_id, prod_comp])
    /-
      🎉 no goals
    -/


lemma curry_obj_injective {F₁ F₂ : C × D ⥤ E} (h : curry.obj F₁ = curry.obj F₂) :
    F₁ = F₂ := by
  /-
    C : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} C
    D : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} D
    E : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} E
    F₁ F₂ : CategoryTheory.Functor (Prod C D) E
    h : Eq (CategoryTheory.curry.obj F₁) (CategoryTheory.curry.obj F₂)
    ⊢ Eq F₁ F₂
  -/
  rw [← uncurry_obj_curry_obj F₁, ← uncurry_obj_curry_obj F₂, h]
  /-
    🎉 no goals
  -/


lemma curry_obj_uncurry_obj (F : B ⥤ C ⥤ D) : curry.obj (uncurry.obj F) = F :=
                                        /-
                                          B : Type u₁
                                          inst✝² : CategoryTheory.Category.{v₁, u₁} B
                                          C : Type u₂
                                          inst✝¹ : CategoryTheory.Category.{v₂, u₂} C
                                          D : Type u₃
                                          inst✝ : CategoryTheory.Category.{v₃, u₃} D
                                          F : CategoryTheory.Functor B (CategoryTheory.Functor C D)
                                          x✝ : B
                                          ⊢ ∀ (X : C), Eq (((CategoryTheory.curry.obj (CategoryTheory.uncurry.obj F)).ob …
                                        -/
                                        /-
                                          🎉 no goals
                                        -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  Functor.ext (fun _ => Functor.ext (by simp) (by simp)) (by aesop_cat)
                                                             /-
                                                               🎉 no goals
                                                             -/


lemma uncurry_obj_injective {F₁ F₂ : B ⥤ C ⥤ D} (h : uncurry.obj F₁ = uncurry.obj F₂) :
    F₁ = F₂ := by
  /-
    B : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} B
    C : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C
    D : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} D
    F₁ F₂ : CategoryTheory.Functor B (CategoryTheory.Functor C D)
    h : Eq (CategoryTheory.uncurry.obj F₁) (CategoryTheory.uncurry.obj F₂)
    ⊢ Eq F₁ F₂
  -/
  rw [← curry_obj_uncurry_obj F₁, ← curry_obj_uncurry_obj F₂, h]
  /-
    🎉 no goals
  -/


lemma flip_flip (F : B ⥤ C ⥤ D) : F.flip.flip = F := rfl


lemma flip_injective {F₁ F₂ : B ⥤ C ⥤ D} (h : F₁.flip = F₂.flip) :
    F₁ = F₂ := by
  /-
    B : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} B
    C : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C
    D : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} D
    F₁ F₂ : CategoryTheory.Functor B (CategoryTheory.Functor C D)
    h : Eq F₁.flip F₂.flip
    ⊢ Eq F₁ F₂
  -/
  rw [← flip_flip F₁, ← flip_flip F₂, h]
  /-
    🎉 no goals
  -/


lemma uncurry_obj_curry_obj_flip_flip (F₁ : B ⥤ C) (F₂ : D ⥤ E) (G : C × E ⥤ H) :
    uncurry.obj (F₂ ⋙ (F₁ ⋙ curry.obj G).flip).flip = (F₁.prod F₂) ⋙ G :=
                  /-
                    B : Type u₁
                    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
                    C : Type u₂
                    inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                    D : Type u₃
                    inst✝² : CategoryTheory.Category.{v₃, u₃} D
                    E : Type u₄
                    inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
                    H : Type u₅
                    inst✝ : CategoryTheory.Category.{v₅, u₅} H
                    F₁ : CategoryTheory.Functor B C
                    F₂ : CategoryTheory.Functor D E
                    G : CategoryTheory.Functor (Prod C E) H
                    ⊢ ∀ (X : Prod B D), Eq ((CategoryTheory.uncurry.obj (F₂.comp (F₁.comp (Categor …
                  -/
  Functor.ext (by simp) (fun ⟨x₁, x₂⟩ ⟨y₁, y₂⟩ ⟨f₁, f₂⟩ => by
                  /-
                    🎉 no goals
                  -/
    /-
      B : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
      C : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} C
      D : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} D
      E : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
      H : Type u₅
      inst✝ : CategoryTheory.Category.{v₅, u₅} H
      F₁ : CategoryTheory.Functor B C
      F₂ : CategoryTheory.Functor D E
      G : CategoryTheory.Functor (Prod C E) H
      x✝² x✝¹ : Prod B D
      x₁ : B
      x₂ : D
      y₁ : B
      y₂ : D
      x✝ : Quiver.Hom { fst := x₁, snd := x₂ } { fst := y₁, snd := y₂ }
      f₁ : Quiver.Hom { fst := x₁, snd := x₂ }.1 { fst := y₁, snd := y₂ }.1
      f₂ : Quiver.Hom { fst := x₁, snd := x₂ }.2 { fst := y₁, snd := y₂ }.2
      ⊢ Eq ((CategoryTheory.uncurry.obj (F₂.comp (F₁.comp (CategoryTheory.curry.obj  …
    -/
    dsimp
    /-
      B : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
      C : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} C
      D : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} D
      E : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
      H : Type u₅
      inst✝ : CategoryTheory.Category.{v₅, u₅} H
      F₁ : CategoryTheory.Functor B C
      F₂ : CategoryTheory.Functor D E
      G : CategoryTheory.Functor (Prod C E) H
      x✝² x✝¹ : Prod B D
      x₁ : B
      x₂ : D
      y₁ : B
      y₂ : D
      x✝ : Quiver.Hom { fst := x₁, snd := x₂ } { fst := y₁, snd := y₂ }
      f₁ : Quiver.Hom { fst := x₁, snd := x₂ }.1 { fst := y₁, snd := y₂ }.1
      f₂ : Quiver.Hom { fst := x₁, snd := x₂ }.2 { fst := y₁, snd := y₂ }.2
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map { fst := F₁.map f₁, snd := Cat …
    -/
    simp only [Category.id_comp, Category.comp_id, ← G.map_comp, prod_comp])
    /-
      🎉 no goals
    -/


lemma uncurry_obj_curry_obj_flip_flip' (F₁ : B ⥤ C) (F₂ : D ⥤ E) (G : C × E ⥤ H) :
    uncurry.obj (F₁ ⋙ (F₂ ⋙ (curry.obj G).flip).flip) = (F₁.prod F₂) ⋙ G :=
                  /-
                    B : Type u₁
                    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
                    C : Type u₂
                    inst✝³ : CategoryTheory.Category.{v₂, u₂} C
                    D : Type u₃
                    inst✝² : CategoryTheory.Category.{v₃, u₃} D
                    E : Type u₄
                    inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
                    H : Type u₅
                    inst✝ : CategoryTheory.Category.{v₅, u₅} H
                    F₁ : CategoryTheory.Functor B C
                    F₂ : CategoryTheory.Functor D E
                    G : CategoryTheory.Functor (Prod C E) H
                    ⊢ ∀ (X : Prod B D), Eq ((CategoryTheory.uncurry.obj (F₁.comp (F₂.comp (Categor …
                  -/
  Functor.ext (by simp) (fun ⟨x₁, x₂⟩ ⟨y₁, y₂⟩ ⟨f₁, f₂⟩ => by
                  /-
                    🎉 no goals
                  -/
    /-
      B : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
      C : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} C
      D : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} D
      E : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
      H : Type u₅
      inst✝ : CategoryTheory.Category.{v₅, u₅} H
      F₁ : CategoryTheory.Functor B C
      F₂ : CategoryTheory.Functor D E
      G : CategoryTheory.Functor (Prod C E) H
      x✝² x✝¹ : Prod B D
      x₁ : B
      x₂ : D
      y₁ : B
      y₂ : D
      x✝ : Quiver.Hom { fst := x₁, snd := x₂ } { fst := y₁, snd := y₂ }
      f₁ : Quiver.Hom { fst := x₁, snd := x₂ }.1 { fst := y₁, snd := y₂ }.1
      f₂ : Quiver.Hom { fst := x₁, snd := x₂ }.2 { fst := y₁, snd := y₂ }.2
      ⊢ Eq ((CategoryTheory.uncurry.obj (F₁.comp (F₂.comp (CategoryTheory.curry.obj  …
    -/
    dsimp
    /-
      B : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} B
      C : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} C
      D : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} D
      E : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} E
      H : Type u₅
      inst✝ : CategoryTheory.Category.{v₅, u₅} H
      F₁ : CategoryTheory.Functor B C
      F₂ : CategoryTheory.Functor D E
      G : CategoryTheory.Functor (Prod C E) H
      x✝² x✝¹ : Prod B D
      x₁ : B
      x₂ : D
      y₁ : B
      y₂ : D
      x✝ : Quiver.Hom { fst := x₁, snd := x₂ } { fst := y₁, snd := y₂ }
      f₁ : Quiver.Hom { fst := x₁, snd := x₂ }.1 { fst := y₁, snd := y₂ }.1
      f₂ : Quiver.Hom { fst := x₁, snd := x₂ }.2 { fst := y₁, snd := y₂ }.2
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map { fst := F₁.map f₁, snd := Cat …
    -/
    simp only [Category.id_comp, Category.comp_id, ← G.map_comp, prod_comp])
    /-
      🎉 no goals
    -/


