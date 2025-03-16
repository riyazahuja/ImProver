/-- `Fiber p S` is the type of elements of `𝒳` mapping to `S` via `p`.  -/
def Fiber (p : 𝒳 ⥤ 𝒮) (S : 𝒮) := { a : 𝒳 // p.obj a = S }


/-- `Fiber p S` has the structure of a category with morphisms being those lying over `𝟙 S`. -/
instance fiberCategory : Category (Fiber p S) where
  Hom a b := {φ : a.1 ⟶ b.1 // IsHomLift p (𝟙 S) φ}
  id a := ⟨𝟙 a.1, IsHomLift.id a.2⟩
                                 /-
                                   𝒮 : Type u₁
                                   𝒳 : Type u₂
                                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} 𝒮
                                   inst✝ : CategoryTheory.Category.{v₂, u₂} 𝒳
                                   p : CategoryTheory.Functor 𝒳 𝒮
                                   S : 𝒮
                                   X✝ Y✝ Z✝ : p.Fiber S
                                   φ : Quiver.Hom X✝ Y✝
                                   ψ : Quiver.Hom Y✝ Z✝
                                   ⊢ p.IsHomLift (CategoryTheory.CategoryStruct.id S) (CategoryTheory.CategoryStr …
                                 -/
  comp φ ψ := ⟨φ.val ≫ ψ.val, by have := φ.2; have := ψ.2; infer_instance⟩
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- The functor including `Fiber p S` into `𝒳`. -/
def fiberInclusion : Fiber p S ⥤ 𝒳 where
  obj a := a.1
  map φ := φ.1


instance {a b : Fiber p S} (φ : a ⟶ b) : IsHomLift p (𝟙 S) (fiberInclusion.map φ) := φ.2


@[ext]
lemma hom_ext {a b : Fiber p S} {φ ψ : a ⟶ b}
    (h : fiberInclusion.map φ = fiberInclusion.map ψ) : φ = ψ :=
  Subtype.ext h


instance : (fiberInclusion : Fiber p S ⥤ _).Faithful where


/-- For fixed `S : 𝒮` this is the natural isomorphism between `fiberInclusion ⋙ p` and the constant
function valued at `S`. -/
@[simps!]
def fiberInclusionCompIsoConst : fiberInclusion ⋙ p ≅ (const (Fiber p S)).obj S :=
  NatIso.ofComponents (fun X ↦ eqToIso X.2)
                /-
                  𝒮 : Type u₁
                  𝒳 : Type u₂
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} 𝒮
                  inst✝ : CategoryTheory.Category.{v₂, u₂} 𝒳
                  p : CategoryTheory.Functor 𝒳 𝒮
                  S : 𝒮
                  X✝ Y✝ : p.Fiber S
                  φ : Quiver.Hom X✝ Y✝
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.Fiber.fiberI …
                -/
    (fun φ ↦ by simp [IsHomLift.fac' p (𝟙 S) (fiberInclusion.map φ)])
                /-
                  🎉 no goals
                -/


lemma fiberInclusion_comp_eq_const : fiberInclusion ⋙ p = (const (Fiber p S)).obj S :=
  Functor.ext (fun x ↦ x.2) (fun _ _ φ ↦ IsHomLift.fac' p (𝟙 S) (fiberInclusion.map φ))


/-- The object of the fiber over `S` corresponding to a `a : 𝒳` such that `p(a) = S`. -/
def mk {p : 𝒳 ⥤ 𝒮} {S : 𝒮} {a : 𝒳} (ha : p.obj a = S) : Fiber p S := ⟨a, ha⟩


@[simp]
lemma fiberInclusion_mk {p : 𝒳 ⥤ 𝒮} {S : 𝒮} {a : 𝒳} (ha : p.obj a = S) :
    fiberInclusion.obj (mk ha) = a :=
  rfl


/-- The morphism in the fiber over `S` corresponding to a morphism in `𝒳` lifting `𝟙 S`. -/
def homMk (p : 𝒳 ⥤ 𝒮) (S : 𝒮) {a b : 𝒳} (φ : a ⟶ b) [IsHomLift p (𝟙 S) φ] :
    mk (domain_eq p (𝟙 S) φ) ⟶ mk (codomain_eq p (𝟙 S) φ) :=
  ⟨φ, inferInstance⟩


@[simp]
lemma fiberInclusion_homMk (p : 𝒳 ⥤ 𝒮) (S : 𝒮) {a b : 𝒳} (φ : a ⟶ b) [IsHomLift p (𝟙 S) φ] :
    fiberInclusion.map (homMk p S φ) = φ :=
  rfl


@[simp]
lemma homMk_id (p : 𝒳 ⥤ 𝒮) (S : 𝒮) (a : 𝒳) [IsHomLift p (𝟙 S) (𝟙 a)] :
    homMk p S (𝟙 a) = 𝟙 (mk (domain_eq p (𝟙 S) (𝟙 a))) :=
  rfl


@[simp]
lemma homMk_comp {a b c : 𝒳} (φ : a ⟶ b) (ψ : b ⟶ c) [IsHomLift p (𝟙 S) φ]
    [IsHomLift p (𝟙 S) ψ] : homMk p S φ ≫ homMk p S ψ = homMk p S (φ ≫ ψ) :=
  rfl


/-- Given a functor `F : C ⥤ 𝒳` such that `F ⋙ p` is constant at some `S : 𝒮`, then
we get an induced functor `C ⥤ Fiber p S` that `F` factors through. -/
@[simps]
def inducedFunctor : C ⥤ Fiber p S where
                        /-
                          𝒮 : Type u₁
                          𝒳 : Type u₂
                          inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
                          inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
                          p✝ : CategoryTheory.Functor 𝒳 𝒮
                          S✝ : 𝒮
                          p : CategoryTheory.Functor 𝒳 𝒮
                          S : 𝒮
                          C : Type u₃
                          inst✝ : CategoryTheory.Category.{v₃, u₃} C
                          F : CategoryTheory.Functor C 𝒳
                          hF : Eq (F.comp p) ((CategoryTheory.Functor.const C).obj S)
                          x : C
                          ⊢ Eq (p.obj (F.obj x)) S
                        -/
  obj x := ⟨F.obj x, by simp only [← comp_obj, hF, const_obj_obj]⟩
                        /-
                          🎉 no goals
                        -/
                                               /-
                                                 𝒮 : Type u₁
                                                 𝒳 : Type u₂
                                                 inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
                                                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
                                                 p✝ : CategoryTheory.Functor 𝒳 𝒮
                                                 S✝ : 𝒮
                                                 p : CategoryTheory.Functor 𝒳 𝒮
                                                 S : 𝒮
                                                 C : Type u₃
                                                 inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                                 F : CategoryTheory.Functor C 𝒳
                                                 hF : Eq (F.comp p) ((CategoryTheory.Functor.const C).obj S)
                                                 X✝ Y✝ : C
                                                 φ : Quiver.Hom X✝ Y✝
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (p.map (F.map φ)) (CategoryTheory.eqT …
                                               -/
  map φ := ⟨F.map φ, of_commsq _ _ _ _ _ <| by simpa using (eqToIso hF).hom.naturality φ⟩
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
lemma inducedFunctor_map {X Y : C} (f : X ⟶ Y) :
    fiberInclusion.map ((inducedFunctor hF).map f) = F.map f := rfl


/-- Given a functor `F : C ⥤ 𝒳` such that `F ⋙ p` is constant at some `S : 𝒮`, then
we get a natural isomorphism between `inducedFunctor _ ⋙ fiberInclusion` and `F`. -/
@[simps!]
def inducedFunctorCompIsoSelf : (inducedFunctor hF) ⋙ fiberInclusion ≅ F := Iso.refl _


lemma inducedFunctor_comp : (inducedFunctor hF) ⋙ fiberInclusion = F := rfl


