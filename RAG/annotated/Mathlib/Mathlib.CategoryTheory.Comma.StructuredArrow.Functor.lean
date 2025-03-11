/-- The structured arrow category `StructuredArrow d T` depends on the chosen domain `d : D` in a
functorial way, inducing a functor `Dᵒᵖ ⥤ Cat`. -/
@[simps]
def functor (T : C ⥤ D) : Dᵒᵖ ⥤ Cat where
  obj d := .of <| StructuredArrow d.unop T
  map f := map f.unop
                                               /-
                                                 C : Type u₁
                                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                 D : Type u₂
                                                 inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                 T : CategoryTheory.Functor C D
                                                 d : Opposite D
                                                 x✝ : ↑({ obj := fun d => CategoryTheory.Cat.of (CategoryTheory.StructuredArrow …
                                                 left✝ : CategoryTheory.Discrete PUnit.{1}
                                                 right✝ : C
                                                 hom✝ : Quiver.Hom ((CategoryTheory.Functor.fromPUnit (Opposite.unop d)).obj le …
                                                 ⊢ Eq (({ obj := fun d => CategoryTheory.Cat.of (CategoryTheory.StructuredArrow …
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
  map_id d := Functor.ext (fun ⟨_, _, _⟩ => by simp [CostructuredArrow.map, Comma.mapRight])
              /-
                🎉 no goals
              -/
                                           /-
                                             C : Type u₁
                                             inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                             D : Type u₂
                                             inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                             T : CategoryTheory.Functor C D
                                             X✝ Y✝ Z✝ : Opposite D
                                             f : Quiver.Hom X✝ Y✝
                                             g : Quiver.Hom Y✝ Z✝
                                             x✝ : ↑({ obj := fun d => CategoryTheory.Cat.of (CategoryTheory.StructuredArrow …
                                             ⊢ Eq (({ obj := fun d => CategoryTheory.Cat.of (CategoryTheory.StructuredArrow …
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  map_comp f g := Functor.ext (fun _ => by simp [CostructuredArrow.map, Comma.mapRight])
                  /-
                    🎉 no goals
                  -/


/-- The costructured arrow category `CostructuredArrow T d` depends on the chosen codomain `d : D`
in a functorial way, inducing a functor `D ⥤ Cat`. -/
@[simps]
def functor (T : C ⥤ D) : D ⥤ Cat where
  obj d := .of <| CostructuredArrow T d
  map f := CostructuredArrow.map f
                                               /-
                                                 C : Type u₁
                                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                 D : Type u₂
                                                 inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                 T : CategoryTheory.Functor C D
                                                 d : D
                                                 x✝ : ↑({ obj := fun d => CategoryTheory.Cat.of (CategoryTheory.CostructuredArr …
                                                 left✝ : C
                                                 right✝ : CategoryTheory.Discrete PUnit.{1}
                                                 hom✝ : Quiver.Hom (T.obj left✝) ((CategoryTheory.Functor.fromPUnit d).obj righ …
                                                 ⊢ Eq (({ obj := fun d => CategoryTheory.Cat.of (CategoryTheory.CostructuredArr …
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
  map_id d := Functor.ext (fun ⟨_, _, _⟩ => by simp [CostructuredArrow.map, Comma.mapRight])
              /-
                🎉 no goals
              -/
                                           /-
                                             C : Type u₁
                                             inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                             D : Type u₂
                                             inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                             T : CategoryTheory.Functor C D
                                             X✝ Y✝ Z✝ : D
                                             f : Quiver.Hom X✝ Y✝
                                             g : Quiver.Hom Y✝ Z✝
                                             x✝ : ↑({ obj := fun d => CategoryTheory.Cat.of (CategoryTheory.CostructuredArr …
                                             ⊢ Eq (({ obj := fun d => CategoryTheory.Cat.of (CategoryTheory.CostructuredArr …
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  map_comp f g := Functor.ext (fun _ => by simp [CostructuredArrow.map, Comma.mapRight])
                  /-
                    🎉 no goals
                  -/


/-- The functor used to establish the equivalence `grothendieckPrecompFunctorEquivalence` between
the Grothendieck construction on `CostructuredArrow.functor` and the comma category. -/
@[simps]
def grothendieckPrecompFunctorToComma : Grothendieck (R ⋙ functor L) ⥤ Comma L R where
  obj P := ⟨P.fiber.left, P.base, P.fiber.hom⟩
                                     /-
                                       C : Type u₁
                                       inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                       D : Type u₂
                                       inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                       E : Type u₃
                                       inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                       L : CategoryTheory.Functor C D
                                       R : CategoryTheory.Functor E D
                                       X✝ Y✝ : CategoryTheory.Grothendieck (R.comp (CategoryTheory.CostructuredArrow. …
                                       f : Quiver.Hom X✝ Y✝
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map f.fiber.left) ((fun P => { lef …
                                     -/
  map f := ⟨f.fiber.left, f.base, by simp⟩
                                     /-
                                       🎉 no goals
                                     -/


/-- Fibers of `grothendieckPrecompFunctorToComma L R`, composed with `Comma.fst L R`, are isomorphic
to the projection `proj L (R.obj X)`. -/
@[simps!]
def ιCompGrothendieckPrecompFunctorToCommaCompFst (X : E) :
    Grothendieck.ι (R ⋙ functor L) X ⋙ grothendieckPrecompFunctorToComma L R ⋙ Comma.fst _ _ ≅
    proj L (R.obj X) :=
                                                         /-
                                                           C : Type u₁
                                                           inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                           D : Type u₂
                                                           inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                           E : Type u₃
                                                           inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                                           L : CategoryTheory.Functor C D
                                                           R : CategoryTheory.Functor E D
                                                           X : E
                                                           X✝ Y✝ : ↑((R.comp (CategoryTheory.CostructuredArrow.functor L)).obj X)
                                                           x✝ : Quiver.Hom X✝ Y✝
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Grothendieck.ι (R.c …
                                                         -/
  NatIso.ofComponents (fun X => Iso.refl _) (fun _ => by simp)
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- The inverse functor used to establish the equivalence `grothendieckPrecompFunctorEquivalence`
between the Grothendieck construction on `CostructuredArrow.functor` and the comma category. -/
@[simps]
def commaToGrothendieckPrecompFunctor : Comma L R ⥤ Grothendieck (R ⋙ functor L) where
  obj X := ⟨X.right, mk X.hom⟩
                     /-
                       C : Type u₁
                       inst✝² : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                       E : Type u₃
                       inst✝ : CategoryTheory.Category.{v₃, u₃} E
                       L : CategoryTheory.Functor C D
                       R : CategoryTheory.Functor E D
                       X✝ Y✝ : CategoryTheory.Comma L R
                       f : Quiver.Hom X✝ Y✝
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map f.left) ((fun X => { base := X …
                     -/
  map f := ⟨f.right, homMk f.left⟩
                     /-
                       🎉 no goals
                     -/
                                           /-
                                             C : Type u₁
                                             inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                             D : Type u₂
                                             inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                             E : Type u₃
                                             inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                             L : CategoryTheory.Functor C D
                                             R : CategoryTheory.Functor E D
                                             X : CategoryTheory.Comma L R
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) ({ obj :=  …
                                           -/
  map_id X := Grothendieck.ext _ _ rfl (by simp)
                                           /-
                                             🎉 no goals
                                           -/
                                               /-
                                                 C : Type u₁
                                                 inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                 D : Type u₂
                                                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                 E : Type u₃
                                                 inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                                 L : CategoryTheory.Functor C D
                                                 R : CategoryTheory.Functor E D
                                                 X✝ Y✝ Z✝ : CategoryTheory.Comma L R
                                                 f : Quiver.Hom X✝ Y✝
                                                 g : Quiver.Hom Y✝ Z✝
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) ({ obj :=  …
                                               -/
  map_comp f g := Grothendieck.ext _ _ rfl (by simp)
                                               /-
                                                 🎉 no goals
                                               -/


/-- For `L : C ⥤ D`, taking the Grothendieck construction of `CostructuredArrow.functor L`
precomposed with another functor `R : E ⥤ D` results in a category which is equivalent to
the comma category `Comma L R`. -/
@[simps]
def grothendieckPrecompFunctorEquivalence : Grothendieck (R ⋙ functor L) ≌ Comma L R where
  functor := grothendieckPrecompFunctorToComma _ _
  inverse := commaToGrothendieckPrecompFunctor _ _
             /-
               C : Type u₁
               inst✝² : CategoryTheory.Category.{v₁, u₁} C
               D : Type u₂
               inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
               E : Type u₃
               inst✝ : CategoryTheory.Category.{v₃, u₃} E
               L : CategoryTheory.Functor C D
               R : CategoryTheory.Functor E D
               ⊢ ∀ {X Y : CategoryTheory.Grothendieck (R.comp (CategoryTheory.CostructuredArr …
             -/
  unitIso := NatIso.ofComponents (fun _ => Iso.refl _)
             /-
               🎉 no goals
             -/
               /-
                 C : Type u₁
                 inst✝² : CategoryTheory.Category.{v₁, u₁} C
                 D : Type u₂
                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                 E : Type u₃
                 inst✝ : CategoryTheory.Category.{v₃, u₃} E
                 L : CategoryTheory.Functor C D
                 R : CategoryTheory.Functor E D
                 ⊢ ∀ {X Y : CategoryTheory.Comma L R} (f : Quiver.Hom X Y), Eq (CategoryTheory. …
               -/
  counitIso := NatIso.ofComponents (fun _ => Iso.refl _)
               /-
                 🎉 no goals
               -/


/-- The functor projecting out the domain of arrows from the Grothendieck construction on
costructured arrows. -/
@[simps!]
def grothendieckProj : Grothendieck (functor L) ⥤ C :=
  grothendieckPrecompFunctorToComma L (𝟭 _) ⋙ Comma.fst _ _


/-- Fibers of `grothendieckProj L` are isomorphic to the projection `proj L X`. -/
@[simps!]
def ιCompGrothendieckProj (X : D) :
    Grothendieck.ι (functor L) X ⋙ grothendieckProj L ≅ proj L X :=
  ιCompGrothendieckPrecompFunctorToCommaCompFst L (𝟭 _) X


/-- Functors between costructured arrow categories induced by morphisms in the base category
composed with fibers of `grothendieckProj L` are isomorphic to the projection `proj L X`. -/
@[simps!]
def mapCompιCompGrothendieckProj {X Y : D} (f : X ⟶ Y) :
    CostructuredArrow.map f ⋙ Grothendieck.ι (functor L) Y ⋙ grothendieckProj L ≅ proj L X :=
  isoWhiskerLeft (CostructuredArrow.map f) (ιCompGrothendieckPrecompFunctorToCommaCompFst L (𝟭 _) Y)


