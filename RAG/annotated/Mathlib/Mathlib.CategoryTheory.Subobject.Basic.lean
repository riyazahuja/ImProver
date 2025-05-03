/-- The category of subobjects of `X : C`, defined as isomorphism classes of monomorphisms into `X`.
-/
def Subobject (X : C) :=
  ThinSkeleton (MonoOver X)


instance (X : C) : PartialOrder (Subobject X) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X✝ Y Z : C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    X : C
    ⊢ PartialOrder (CategoryTheory.Subobject X)
  -/
  dsimp only [Subobject]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X✝ Y Z : C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    X : C
    ⊢ PartialOrder (CategoryTheory.ThinSkeleton (CategoryTheory.MonoOver X))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Convenience constructor for a subobject. -/
def mk {X A : C} (f : A ⟶ X) [Mono f] : Subobject X :=
  (toThinSkeleton _).obj (MonoOver.mk' f)


attribute [local ext] CategoryTheory.Comma


protected theorem ind {X : C} (p : Subobject X → Prop)
    (h : ∀ ⦃A : C⦄ (f : A ⟶ X) [Mono f], p (Subobject.mk f)) (P : Subobject X) : p P := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    p : CategoryTheory.Subobject X → Prop
    h : ∀ ⦃A : C⦄ (f : Quiver.Hom A X) [inst : CategoryTheory.Mono f], p (Category …
    P : CategoryTheory.Subobject X
    ⊢ p P
  -/
  apply Quotient.inductionOn'
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    p : CategoryTheory.Subobject X → Prop
    h : ∀ ⦃A : C⦄ (f : Quiver.Hom A X) [inst : CategoryTheory.Mono f], p (Category …
    P : CategoryTheory.Subobject X
    ⊢ ∀ (a : CategoryTheory.MonoOver X), p (Quotient.mk'' a)
  -/
  intro a
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    p : CategoryTheory.Subobject X → Prop
    h : ∀ ⦃A : C⦄ (f : Quiver.Hom A X) [inst : CategoryTheory.Mono f], p (Category …
    P : CategoryTheory.Subobject X
    a : CategoryTheory.MonoOver X
    ⊢ p (Quotient.mk'' a)
  -/
  exact h a.arrow
  /-
    🎉 no goals
  -/


protected theorem ind₂ {X : C} (p : Subobject X → Subobject X → Prop)
    (h : ∀ ⦃A B : C⦄ (f : A ⟶ X) (g : B ⟶ X) [Mono f] [Mono g],
      p (Subobject.mk f) (Subobject.mk g))
    (P Q : Subobject X) : p P Q := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    p : CategoryTheory.Subobject X → CategoryTheory.Subobject X → Prop
    h : ∀ ⦃A B : C⦄ (f : Quiver.Hom A X) (g : Quiver.Hom B X) [inst : CategoryTheo …
    P Q : CategoryTheory.Subobject X
    ⊢ p P Q
  -/
  apply Quotient.inductionOn₂'
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    p : CategoryTheory.Subobject X → CategoryTheory.Subobject X → Prop
    h : ∀ ⦃A B : C⦄ (f : Quiver.Hom A X) (g : Quiver.Hom B X) [inst : CategoryTheo …
    P Q : CategoryTheory.Subobject X
    ⊢ ∀ (a₁ a₂ : CategoryTheory.MonoOver X), p (Quotient.mk'' a₁) (Quotient.mk'' a₂)
  -/
  intro a b
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    p : CategoryTheory.Subobject X → CategoryTheory.Subobject X → Prop
    h : ∀ ⦃A B : C⦄ (f : Quiver.Hom A X) (g : Quiver.Hom B X) [inst : CategoryTheo …
    P Q : CategoryTheory.Subobject X
    a b : CategoryTheory.MonoOver X
    ⊢ p (Quotient.mk'' a) (Quotient.mk'' b)
  -/
  exact h a.arrow b.arrow
  /-
    🎉 no goals
  -/


/-- Declare a function on subobjects of `X` by specifying a function on monomorphisms with
    codomain `X`. -/
protected def lift {α : Sort*} {X : C} (F : ∀ ⦃A : C⦄ (f : A ⟶ X) [Mono f], α)
    (h :
      ∀ ⦃A B : C⦄ (f : A ⟶ X) (g : B ⟶ X) [Mono f] [Mono g] (i : A ≅ B),
        i.hom ≫ g = f → F f = F g) :
    Subobject X → α := fun P =>
  Quotient.liftOn' P (fun m => F m.arrow) fun m n ⟨i⟩ =>
    h m.arrow n.arrow ((MonoOver.forget X ⋙ Over.forget X).mapIso i) (Over.w i.hom)


@[simp]
protected theorem lift_mk {α : Sort*} {X : C} (F : ∀ ⦃A : C⦄ (f : A ⟶ X) [Mono f], α) {h A}
    (f : A ⟶ X) [Mono f] : Subobject.lift F h (Subobject.mk f) = F f :=
  rfl


/-- The category of subobjects is equivalent to the `MonoOver` category. It is more convenient to
use the former due to the partial order instance, but oftentimes it is easier to define structures
on the latter. -/
noncomputable def equivMonoOver (X : C) : Subobject X ≌ MonoOver X :=
  ThinSkeleton.equivalence _


/-- Use choice to pick a representative `MonoOver X` for each `Subobject X`.
-/
noncomputable def representative {X : C} : Subobject X ⥤ MonoOver X :=
  (equivMonoOver X).functor


/-- Starting with `A : MonoOver X`, we can take its equivalence class in `Subobject X`
then pick an arbitrary representative using `representative.obj`.
This is isomorphic (in `MonoOver X`) to the original `A`.
-/
noncomputable def representativeIso {X : C} (A : MonoOver X) :
    representative.obj ((toThinSkeleton _).obj A) ≅ A :=
  (equivMonoOver X).counitIso.app A


/-- Use choice to pick a representative underlying object in `C` for any `Subobject X`.

Prefer to use the coercion `P : C` rather than explicitly writing `underlying.obj P`.
-/
noncomputable def underlying {X : C} : Subobject X ⥤ C :=
  representative ⋙ MonoOver.forget _ ⋙ Over.forget _


instance : CoeOut (Subobject X) C where coe Y := underlying.obj Y

-- Porting note: removed as it has become a syntactic tautology
-- @[simp]
-- theorem underlying_as_coe {X : C} (P : Subobject X) : underlying.obj P = P :=
--   rfl


/-- If we construct a `Subobject Y` from an explicit `f : X ⟶ Y` with `[Mono f]`,
then pick an arbitrary choice of underlying object `(Subobject.mk f : C)` back in `C`,
it is isomorphic (in `C`) to the original `X`.
-/
noncomputable def underlyingIso {X Y : C} (f : X ⟶ Y) [Mono f] : (Subobject.mk f : C) ≅ X :=
  (MonoOver.forget _ ⋙ Over.forget _).mapIso (representativeIso (MonoOver.mk' f))


/-- The morphism in `C` from the arbitrarily chosen underlying object to the ambient object.
-/
noncomputable def arrow {X : C} (Y : Subobject X) : (Y : C) ⟶ X :=
  (representative.obj Y).obj.hom


instance arrow_mono {X : C} (Y : Subobject X) : Mono Y.arrow :=
  (representative.obj Y).property


@[simp]
theorem arrow_congr {A : C} (X Y : Subobject A) (h : X = Y) :
    eqToHom (congr_arg (fun X : Subobject A => (X : C)) h) ≫ Y.arrow = X.arrow := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    A : C
    X Y : CategoryTheory.Subobject A
    h : Eq X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) Y.arrow) X …
  -/
  induction h
  /-
    case refl
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    A : C
    X Y : CategoryTheory.Subobject A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) X.arrow) X …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem representative_coe (Y : Subobject X) : (representative.obj Y : C) = (Y : C) :=
  rfl


@[simp]
theorem representative_arrow (Y : Subobject X) : (representative.obj Y).arrow = Y.arrow :=
  rfl


@[reassoc (attr := simp)]
theorem underlying_arrow {X : C} {Y Z : Subobject X} (f : Y ⟶ Z) :
    underlying.map f ≫ arrow Z = arrow Y :=
  Over.w (representative.map f)


@[reassoc (attr := simp), elementwise (attr := simp)]
theorem underlyingIso_arrow {X Y : C} (f : X ⟶ Y) [Mono f] :
    (underlyingIso f).inv ≫ (Subobject.mk f).arrow = f :=
  Over.w _


@[reassoc (attr := simp)]
theorem underlyingIso_hom_comp_eq_mk {X Y : C} (f : X ⟶ Y) [Mono f] :
    (underlyingIso f).hom ≫ f = (mk f).arrow :=
  (Iso.eq_inv_comp _).1 (underlyingIso_arrow f).symm


/-- Two morphisms into a subobject are equal exactly if
the morphisms into the ambient object are equal -/
@[ext]
theorem eq_of_comp_arrow_eq {X Y : C} {P : Subobject Y} {f g : X ⟶ P}
    (h : f ≫ P.arrow = g ≫ P.arrow) : f = g :=
  (cancel_mono P.arrow).mp h


theorem mk_le_mk_of_comm {B A₁ A₂ : C} {f₁ : A₁ ⟶ B} {f₂ : A₂ ⟶ B} [Mono f₁] [Mono f₂] (g : A₁ ⟶ A₂)
    (w : g ≫ f₂ = f₁) : mk f₁ ≤ mk f₂ :=
  ⟨MonoOver.homMk _ w⟩


@[simp]
theorem mk_arrow (P : Subobject X) : mk P.arrow = P :=
  Quotient.inductionOn' P fun Q => by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      P : CategoryTheory.Subobject X
      Q : CategoryTheory.MonoOver X
      ⊢ Eq (CategoryTheory.Subobject.mk (CategoryTheory.Subobject.arrow (Quotient.mk …
    -/
    obtain ⟨e⟩ := @Quotient.mk_out' _ (isIsomorphicSetoid _) Q
    /-
      case intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      P : CategoryTheory.Subobject X
      Q : CategoryTheory.MonoOver X
      e : CategoryTheory.Iso (Quotient.mk'' Q).out Q
      ⊢ Eq (CategoryTheory.Subobject.mk (CategoryTheory.Subobject.arrow (Quotient.mk …
    -/
    exact Quotient.sound' ⟨MonoOver.isoMk (Iso.refl _) ≪≫ e⟩
    /-
      🎉 no goals
    -/


theorem le_of_comm {B : C} {X Y : Subobject B} (f : (X : C) ⟶ (Y : C)) (w : f ≫ Y.arrow = X.arrow) :
    X ≤ Y := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    B : C
    X Y : CategoryTheory.Subobject B
    f : Quiver.Hom (CategoryTheory.Subobject.underlying.obj X) (CategoryTheory.Sub …
    w : Eq (CategoryTheory.CategoryStruct.comp f Y.arrow) X.arrow
    ⊢ LE.le X Y
  -/
                                   /-
                                     🎉 no goals
                                   -/
  convert mk_le_mk_of_comm _ w <;> simp
                                   /-
                                     🎉 no goals
                                   -/


theorem le_mk_of_comm {B A : C} {X : Subobject B} {f : A ⟶ B} [Mono f] (g : (X : C) ⟶ A)
    (w : g ≫ f = X.arrow) : X ≤ mk f :=
                                               /-
                                                 C : Type u₁
                                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                 B A : C
                                                 X : CategoryTheory.Subobject B
                                                 f : Quiver.Hom A B
                                                 inst✝ : CategoryTheory.Mono f
                                                 g : Quiver.Hom (CategoryTheory.Subobject.underlying.obj X) A
                                                 w : Eq (CategoryTheory.CategoryStruct.comp g f) X.arrow
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
                                               -/
  le_of_comm (g ≫ (underlyingIso f).inv) <| by simp [w]
                                               /-
                                                 🎉 no goals
                                               -/


theorem mk_le_of_comm {B A : C} {X : Subobject B} {f : A ⟶ B} [Mono f] (g : A ⟶ (X : C))
    (w : g ≫ X.arrow = f) : mk f ≤ X :=
                                               /-
                                                 C : Type u₁
                                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                 B A : C
                                                 X : CategoryTheory.Subobject B
                                                 f : Quiver.Hom A B
                                                 inst✝ : CategoryTheory.Mono f
                                                 g : Quiver.Hom A (CategoryTheory.Subobject.underlying.obj X)
                                                 w : Eq (CategoryTheory.CategoryStruct.comp g X.arrow) f
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                               -/
  le_of_comm ((underlyingIso f).hom ≫ g) <| by simp [w]
                                               /-
                                                 🎉 no goals
                                               -/


/-- To show that two subobjects are equal, it suffices to exhibit an isomorphism commuting with
    the arrows. -/
@[ext (iff := false)]
theorem eq_of_comm {B : C} {X Y : Subobject B} (f : (X : C) ≅ (Y : C))
    (w : f.hom ≫ Y.arrow = X.arrow) : X = Y :=
  le_antisymm (le_of_comm f.hom w) <| le_of_comm f.inv <| f.inv_comp_eq.2 w.symm


/-- To show that two subobjects are equal, it suffices to exhibit an isomorphism commuting with
    the arrows. -/
theorem eq_mk_of_comm {B A : C} {X : Subobject B} (f : A ⟶ B) [Mono f] (i : (X : C) ≅ A)
    (w : i.hom ≫ f = X.arrow) : X = mk f :=
                                                    /-
                                                      C : Type u₁
                                                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                      B A : C
                                                      X : CategoryTheory.Subobject B
                                                      f : Quiver.Hom A B
                                                      inst✝ : CategoryTheory.Mono f
                                                      i : CategoryTheory.Iso (CategoryTheory.Subobject.underlying.obj X) A
                                                      w : Eq (CategoryTheory.CategoryStruct.comp i.hom f) X.arrow
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (i.trans (CategoryTheory.Subobject.un …
                                                    -/
  eq_of_comm (i.trans (underlyingIso f).symm) <| by simp [w]
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- To show that two subobjects are equal, it suffices to exhibit an isomorphism commuting with
    the arrows. -/
theorem mk_eq_of_comm {B A : C} {X : Subobject B} (f : A ⟶ B) [Mono f] (i : A ≅ (X : C))
    (w : i.hom ≫ X.arrow = f) : mk f = X :=
                                          /-
                                            C : Type u₁
                                            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                            B A : C
                                            X : CategoryTheory.Subobject B
                                            f : Quiver.Hom A B
                                            inst✝ : CategoryTheory.Mono f
                                            i : CategoryTheory.Iso A (CategoryTheory.Subobject.underlying.obj X)
                                            w : Eq (CategoryTheory.CategoryStruct.comp i.hom X.arrow) f
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp i.symm.hom f) X.arrow
                                          -/
  Eq.symm <| eq_mk_of_comm _ i.symm <| by rw [Iso.symm_hom, Iso.inv_comp_eq, w]
                                          /-
                                            🎉 no goals
                                          -/


/-- To show that two subobjects are equal, it suffices to exhibit an isomorphism commuting with
    the arrows. -/
theorem mk_eq_mk_of_comm {B A₁ A₂ : C} (f : A₁ ⟶ B) (g : A₂ ⟶ B) [Mono f] [Mono g] (i : A₁ ≅ A₂)
    (w : i.hom ≫ g = f) : mk f = mk g :=
                                                    /-
                                                      C : Type u₁
                                                      inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                      B A₁ A₂ : C
                                                      f : Quiver.Hom A₁ B
                                                      g : Quiver.Hom A₂ B
                                                      inst✝¹ : CategoryTheory.Mono f
                                                      inst✝ : CategoryTheory.Mono g
                                                      i : CategoryTheory.Iso A₁ A₂
                                                      w : Eq (CategoryTheory.CategoryStruct.comp i.hom g) f
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Subobject.underlying …
                                                    -/
  eq_mk_of_comm _ ((underlyingIso f).trans i) <| by simp [w]
                                                    /-
                                                      🎉 no goals
                                                    -/

-- We make `X` and `Y` explicit arguments here so that when `ofLE` appears in goal statements
-- it is possible to see its source and target
-- (`h` will just display as `_`, because it is in `Prop`).

/-- An inequality of subobjects is witnessed by some morphism between the corresponding objects. -/
def ofLE {B : C} (X Y : Subobject B) (h : X ≤ Y) : (X : C) ⟶ (Y : C) :=
  underlying.map <| h.hom


@[reassoc (attr := simp)]
theorem ofLE_arrow {B : C} {X Y : Subobject B} (h : X ≤ Y) : ofLE X Y h ≫ Y.arrow = X.arrow :=
  underlying_arrow _


instance {B : C} (X Y : Subobject B) (h : X ≤ Y) : Mono (ofLE X Y h) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X✝ Y✝ Z : C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    B : C
    X Y : CategoryTheory.Subobject B
    h : LE.le X Y
    ⊢ CategoryTheory.Mono (X.ofLE Y h)
  -/
  fconstructor
  /-
    case right_cancellation
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X✝ Y✝ Z : C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    B : C
    X Y : CategoryTheory.Subobject B
    h : LE.le X Y
    ⊢ ∀ {Z : C} (g h_1 : Quiver.Hom Z (CategoryTheory.Subobject.underlying.obj X)) …
  -/
  intro Z f g w
  /-
    case right_cancellation
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X✝ Y✝ Z✝ : C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    B : C
    X Y : CategoryTheory.Subobject B
    h : LE.le X Y
    Z : C
    f g : Quiver.Hom Z (CategoryTheory.Subobject.underlying.obj X)
    w : Eq (CategoryTheory.CategoryStruct.comp f (X.ofLE Y h)) (CategoryTheory.Cat …
    ⊢ Eq f g
  -/
  replace w := w =≫ Y.arrow
  /-
    case right_cancellation
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X✝ Y✝ Z✝ : C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    B : C
    X Y : CategoryTheory.Subobject B
    h : LE.le X Y
    Z : C
    f g : Quiver.Hom Z (CategoryTheory.Subobject.underlying.obj X)
    w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
    ⊢ Eq f g
  -/
  ext
  /-
    case right_cancellation.h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X✝ Y✝ Z✝ : C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    B : C
    X Y : CategoryTheory.Subobject B
    h : LE.le X Y
    Z : C
    f g : Quiver.Hom Z (CategoryTheory.Subobject.underlying.obj X)
    w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f X.arrow) (CategoryTheory.CategorySt …
  -/
  simpa using w
  /-
    🎉 no goals
  -/


theorem ofLE_mk_le_mk_of_comm {B A₁ A₂ : C} {f₁ : A₁ ⟶ B} {f₂ : A₂ ⟶ B} [Mono f₁] [Mono f₂]
    (g : A₁ ⟶ A₂) (w : g ≫ f₂ = f₁) :
    ofLE _ _ (mk_le_mk_of_comm g w) = (underlyingIso _).hom ≫ g ≫ (underlyingIso _).inv := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    B A₁ A₂ : C
    f₁ : Quiver.Hom A₁ B
    f₂ : Quiver.Hom A₂ B
    inst✝¹ : CategoryTheory.Mono f₁
    inst✝ : CategoryTheory.Mono f₂
    g : Quiver.Hom A₁ A₂
    w : Eq (CategoryTheory.CategoryStruct.comp g f₂) f₁
    ⊢ Eq ((CategoryTheory.Subobject.mk f₁).ofLE (CategoryTheory.Subobject.mk f₂) ⋯ …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    B A₁ A₂ : C
    f₁ : Quiver.Hom A₁ B
    f₂ : Quiver.Hom A₂ B
    inst✝¹ : CategoryTheory.Mono f₁
    inst✝ : CategoryTheory.Mono f₂
    g : Quiver.Hom A₁ A₂
    w : Eq (CategoryTheory.CategoryStruct.comp g f₂) f₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Subobject.mk f₁).ofL …
  -/
  simp [w]
  /-
    🎉 no goals
  -/


/-- An inequality of subobjects is witnessed by some morphism between the corresponding objects. -/
def ofLEMk {B A : C} (X : Subobject B) (f : A ⟶ B) [Mono f] (h : X ≤ mk f) : (X : C) ⟶ A :=
  ofLE X (mk f) h ≫ (underlyingIso f).hom


instance {B A : C} (X : Subobject B) (f : A ⟶ B) [Mono f] (h : X ≤ mk f) :
    Mono (ofLEMk X f h) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X✝ Y Z : C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    B A : C
    X : CategoryTheory.Subobject B
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.Mono f
    h : LE.le X (CategoryTheory.Subobject.mk f)
    ⊢ CategoryTheory.Mono (X.ofLEMk f h)
  -/
  dsimp only [ofLEMk]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X✝ Y Z : C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    B A : C
    X : CategoryTheory.Subobject B
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.Mono f
    h : LE.le X (CategoryTheory.Subobject.mk f)
    ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp (X.ofLE (CategoryThe …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp]
theorem ofLEMk_comp {B A : C} {X : Subobject B} {f : A ⟶ B} [Mono f] (h : X ≤ mk f) :
                                     /-
                                       C : Type u₁
                                       inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                       B A : C
                                       X : CategoryTheory.Subobject B
                                       f : Quiver.Hom A B
                                       inst✝ : CategoryTheory.Mono f
                                       h : LE.le X (CategoryTheory.Subobject.mk f)
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.ofLEMk f h) f) X.arrow
                                     -/
    ofLEMk X f h ≫ f = X.arrow := by simp [ofLEMk]
                                     /-
                                       🎉 no goals
                                     -/


/-- An inequality of subobjects is witnessed by some morphism between the corresponding objects. -/
def ofMkLE {B A : C} (f : A ⟶ B) [Mono f] (X : Subobject B) (h : mk f ≤ X) : A ⟶ (X : C) :=
  (underlyingIso f).inv ≫ ofLE (mk f) X h


instance {B A : C} (f : A ⟶ B) [Mono f] (X : Subobject B) (h : mk f ≤ X) :
    Mono (ofMkLE f X h) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X✝ Y Z : C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    B A : C
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.Mono f
    X : CategoryTheory.Subobject B
    h : LE.le (CategoryTheory.Subobject.mk f) X
    ⊢ CategoryTheory.Mono (CategoryTheory.Subobject.ofMkLE f X h)
  -/
  dsimp only [ofMkLE]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X✝ Y Z : C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    B A : C
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.Mono f
    X : CategoryTheory.Subobject B
    h : LE.le (CategoryTheory.Subobject.mk f) X
    ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subo …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp]
theorem ofMkLE_arrow {B A : C} {f : A ⟶ B} [Mono f] {X : Subobject B} (h : mk f ≤ X) :
                                     /-
                                       C : Type u₁
                                       inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                       B A : C
                                       f : Quiver.Hom A B
                                       inst✝ : CategoryTheory.Mono f
                                       X : CategoryTheory.Subobject B
                                       h : LE.le (CategoryTheory.Subobject.mk f) X
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.ofMkLE f X  …
                                     -/
    ofMkLE f X h ≫ X.arrow = f := by simp [ofMkLE]
                                     /-
                                       🎉 no goals
                                     -/


/-- An inequality of subobjects is witnessed by some morphism between the corresponding objects. -/
def ofMkLEMk {B A₁ A₂ : C} (f : A₁ ⟶ B) (g : A₂ ⟶ B) [Mono f] [Mono g] (h : mk f ≤ mk g) :
    A₁ ⟶ A₂ :=
  (underlyingIso f).inv ≫ ofLE (mk f) (mk g) h ≫ (underlyingIso g).hom


instance {B A₁ A₂ : C} (f : A₁ ⟶ B) (g : A₂ ⟶ B) [Mono f] [Mono g] (h : mk f ≤ mk g) :
    Mono (ofMkLEMk f g h) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    B A₁ A₂ : C
    f : Quiver.Hom A₁ B
    g : Quiver.Hom A₂ B
    inst✝¹ : CategoryTheory.Mono f
    inst✝ : CategoryTheory.Mono g
    h : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
    ⊢ CategoryTheory.Mono (CategoryTheory.Subobject.ofMkLEMk f g h)
  -/
  dsimp only [ofMkLEMk]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    B A₁ A₂ : C
    f : Quiver.Hom A₁ B
    g : Quiver.Hom A₂ B
    inst✝¹ : CategoryTheory.Mono f
    inst✝ : CategoryTheory.Mono g
    h : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
    ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subo …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp]
theorem ofMkLEMk_comp {B A₁ A₂ : C} {f : A₁ ⟶ B} {g : A₂ ⟶ B} [Mono f] [Mono g] (h : mk f ≤ mk g) :
                                 /-
                                   C : Type u₁
                                   inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                   B A₁ A₂ : C
                                   f : Quiver.Hom A₁ B
                                   g : Quiver.Hom A₂ B
                                   inst✝¹ : CategoryTheory.Mono f
                                   inst✝ : CategoryTheory.Mono g
                                   h : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.ofMkLEMk f  …
                                 -/
    ofMkLEMk f g h ≫ g = f := by simp [ofMkLEMk]
                                 /-
                                   🎉 no goals
                                 -/


@[reassoc (attr := simp)]
theorem ofLE_comp_ofLE {B : C} (X Y Z : Subobject B) (h₁ : X ≤ Y) (h₂ : Y ≤ Z) :
    ofLE X Y h₁ ≫ ofLE Y Z h₂ = ofLE X Z (h₁.trans h₂) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    B : C
    X Y Z : CategoryTheory.Subobject B
    h₁ : LE.le X Y
    h₂ : LE.le Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.ofLE Y h₁) (Y.ofLE Z h₂)) (X.ofLE  …
  -/
  simp only [ofLE, ← Functor.map_comp underlying]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    B : C
    X Y Z : CategoryTheory.Subobject B
    h₁ : LE.le X Y
    h₂ : LE.le Y Z
    ⊢ Eq (CategoryTheory.Subobject.underlying.map (CategoryTheory.CategoryStruct.c …
  -/
  congr 1
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem ofLE_comp_ofLEMk {B A : C} (X Y : Subobject B) (f : A ⟶ B) [Mono f] (h₁ : X ≤ Y)
    (h₂ : Y ≤ mk f) : ofLE X Y h₁ ≫ ofLEMk Y f h₂ = ofLEMk X f (h₁.trans h₂) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    B A : C
    X Y : CategoryTheory.Subobject B
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.Mono f
    h₁ : LE.le X Y
    h₂ : LE.le Y (CategoryTheory.Subobject.mk f)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.ofLE Y h₁) (Y.ofLEMk f h₂)) (X.ofL …
  -/
  simp only [ofMkLE, ofLEMk, ofLE, ← Functor.map_comp_assoc underlying]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    B A : C
    X Y : CategoryTheory.Subobject B
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.Mono f
    h₁ : LE.le X Y
    h₂ : LE.le Y (CategoryTheory.Subobject.mk f)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.underlying. …
  -/
  congr 1
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem ofLEMk_comp_ofMkLE {B A : C} (X : Subobject B) (f : A ⟶ B) [Mono f] (Y : Subobject B)
    (h₁ : X ≤ mk f) (h₂ : mk f ≤ Y) : ofLEMk X f h₁ ≫ ofMkLE f Y h₂ = ofLE X Y (h₁.trans h₂) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    B A : C
    X : CategoryTheory.Subobject B
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.Mono f
    Y : CategoryTheory.Subobject B
    h₁ : LE.le X (CategoryTheory.Subobject.mk f)
    h₂ : LE.le (CategoryTheory.Subobject.mk f) Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.ofLEMk f h₁) (CategoryTheory.Subob …
  -/
  simp only [ofMkLE, ofLEMk, ofLE, ← Functor.map_comp underlying, assoc, Iso.hom_inv_id_assoc]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    B A : C
    X : CategoryTheory.Subobject B
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.Mono f
    Y : CategoryTheory.Subobject B
    h₁ : LE.le X (CategoryTheory.Subobject.mk f)
    h₂ : LE.le (CategoryTheory.Subobject.mk f) Y
    ⊢ Eq (CategoryTheory.Subobject.underlying.map (CategoryTheory.CategoryStruct.c …
  -/
  congr 1
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem ofLEMk_comp_ofMkLEMk {B A₁ A₂ : C} (X : Subobject B) (f : A₁ ⟶ B) [Mono f] (g : A₂ ⟶ B)
    [Mono g] (h₁ : X ≤ mk f) (h₂ : mk f ≤ mk g) :
    ofLEMk X f h₁ ≫ ofMkLEMk f g h₂ = ofLEMk X g (h₁.trans h₂) := by
  simp only [ofMkLE, ofLEMk, ofLE, ofMkLEMk, ← Functor.map_comp_assoc underlying,
    assoc, Iso.hom_inv_id_assoc]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    B A₁ A₂ : C
    X : CategoryTheory.Subobject B
    f : Quiver.Hom A₁ B
    inst✝¹ : CategoryTheory.Mono f
    g : Quiver.Hom A₂ B
    inst✝ : CategoryTheory.Mono g
    h₁ : LE.le X (CategoryTheory.Subobject.mk f)
    h₂ : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.underlying. …
  -/
  congr 1
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem ofMkLE_comp_ofLE {B A₁ : C} (f : A₁ ⟶ B) [Mono f] (X Y : Subobject B) (h₁ : mk f ≤ X)
    (h₂ : X ≤ Y) : ofMkLE f X h₁ ≫ ofLE X Y h₂ = ofMkLE f Y (h₁.trans h₂) := by
  simp only [ofMkLE, ofLEMk, ofLE, ofMkLEMk, ← Functor.map_comp underlying,
    assoc]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    B A₁ : C
    f : Quiver.Hom A₁ B
    inst✝ : CategoryTheory.Mono f
    X Y : CategoryTheory.Subobject B
    h₁ : LE.le (CategoryTheory.Subobject.mk f) X
    h₂ : LE.le X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.underlyingI …
  -/
  congr 1
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem ofMkLE_comp_ofLEMk {B A₁ A₂ : C} (f : A₁ ⟶ B) [Mono f] (X : Subobject B) (g : A₂ ⟶ B)
    [Mono g] (h₁ : mk f ≤ X) (h₂ : X ≤ mk g) :
    ofMkLE f X h₁ ≫ ofLEMk X g h₂ = ofMkLEMk f g (h₁.trans h₂) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    B A₁ A₂ : C
    f : Quiver.Hom A₁ B
    inst✝¹ : CategoryTheory.Mono f
    X : CategoryTheory.Subobject B
    g : Quiver.Hom A₂ B
    inst✝ : CategoryTheory.Mono g
    h₁ : LE.le (CategoryTheory.Subobject.mk f) X
    h₂ : LE.le X (CategoryTheory.Subobject.mk g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.ofMkLE f X  …
  -/
  simp only [ofMkLE, ofLEMk, ofLE, ofMkLEMk, ← Functor.map_comp_assoc underlying, assoc]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    B A₁ A₂ : C
    f : Quiver.Hom A₁ B
    inst✝¹ : CategoryTheory.Mono f
    X : CategoryTheory.Subobject B
    g : Quiver.Hom A₂ B
    inst✝ : CategoryTheory.Mono g
    h₁ : LE.le (CategoryTheory.Subobject.mk f) X
    h₂ : LE.le X (CategoryTheory.Subobject.mk g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.underlyingI …
  -/
  congr 1
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem ofMkLEMk_comp_ofMkLE {B A₁ A₂ : C} (f : A₁ ⟶ B) [Mono f] (g : A₂ ⟶ B) [Mono g]
    (X : Subobject B) (h₁ : mk f ≤ mk g) (h₂ : mk g ≤ X) :
    ofMkLEMk f g h₁ ≫ ofMkLE g X h₂ = ofMkLE f X (h₁.trans h₂) := by
  simp only [ofMkLE, ofLEMk, ofLE, ofMkLEMk, ← Functor.map_comp underlying,
    assoc, Iso.hom_inv_id_assoc]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    B A₁ A₂ : C
    f : Quiver.Hom A₁ B
    inst✝¹ : CategoryTheory.Mono f
    g : Quiver.Hom A₂ B
    inst✝ : CategoryTheory.Mono g
    X : CategoryTheory.Subobject B
    h₁ : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
    h₂ : LE.le (CategoryTheory.Subobject.mk g) X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.underlyingI …
  -/
  congr 1
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem ofMkLEMk_comp_ofMkLEMk {B A₁ A₂ A₃ : C} (f : A₁ ⟶ B) [Mono f] (g : A₂ ⟶ B) [Mono g]
    (h : A₃ ⟶ B) [Mono h] (h₁ : mk f ≤ mk g) (h₂ : mk g ≤ mk h) :
    ofMkLEMk f g h₁ ≫ ofMkLEMk g h h₂ = ofMkLEMk f h (h₁.trans h₂) := by
  simp only [ofMkLE, ofLEMk, ofLE, ofMkLEMk, ← Functor.map_comp_assoc underlying, assoc,
    Iso.hom_inv_id_assoc]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    B A₁ A₂ A₃ : C
    f : Quiver.Hom A₁ B
    inst✝² : CategoryTheory.Mono f
    g : Quiver.Hom A₂ B
    inst✝¹ : CategoryTheory.Mono g
    h : Quiver.Hom A₃ B
    inst✝ : CategoryTheory.Mono h
    h₁ : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
    h₂ : LE.le (CategoryTheory.Subobject.mk g) (CategoryTheory.Subobject.mk h)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.underlyingI …
  -/
  congr 1
  /-
    🎉 no goals
  -/


@[simp]
theorem ofLE_refl {B : C} (X : Subobject B) : ofLE X X le_rfl = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    B : C
    X : CategoryTheory.Subobject B
    ⊢ Eq (X.ofLE X ⋯) (CategoryTheory.CategoryStruct.id (CategoryTheory.Subobject. …
  -/
  apply (cancel_mono X.arrow).mp
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    B : C
    X : CategoryTheory.Subobject B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.ofLE X ⋯) X.arrow) (CategoryTheory …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem ofMkLEMk_refl {B A₁ : C} (f : A₁ ⟶ B) [Mono f] : ofMkLEMk f f le_rfl = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    B A₁ : C
    f : Quiver.Hom A₁ B
    inst✝ : CategoryTheory.Mono f
    ⊢ Eq (CategoryTheory.Subobject.ofMkLEMk f f ⋯) (CategoryTheory.CategoryStruct. …
  -/
  apply (cancel_mono f).mp
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    B A₁ : C
    f : Quiver.Hom A₁ B
    inst✝ : CategoryTheory.Mono f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.ofMkLEMk f  …
  -/
  simp
  /-
    🎉 no goals
  -/

-- As with `ofLE`, we have `X` and `Y` as explicit arguments for readability.

/-- An equality of subobjects gives an isomorphism of the corresponding objects.
(One could use `underlying.mapIso (eqToIso h))` here, but this is more readable.) -/
@[simps]
def isoOfEq {B : C} (X Y : Subobject B) (h : X = Y) : (X : C) ≅ (Y : C) where
  hom := ofLE _ _ h.le
  inv := ofLE _ _ h.ge


/-- An equality of subobjects gives an isomorphism of the corresponding objects. -/
@[simps]
def isoOfEqMk {B A : C} (X : Subobject B) (f : A ⟶ B) [Mono f] (h : X = mk f) : (X : C) ≅ A where
  hom := ofLEMk X f h.le
  inv := ofMkLE f X h.ge


/-- An equality of subobjects gives an isomorphism of the corresponding objects. -/
@[simps]
def isoOfMkEq {B A : C} (f : A ⟶ B) [Mono f] (X : Subobject B) (h : mk f = X) : A ≅ (X : C) where
  hom := ofMkLE f X h.le
  inv := ofLEMk X f h.ge


/-- An equality of subobjects gives an isomorphism of the corresponding objects. -/
@[simps]
def isoOfMkEqMk {B A₁ A₂ : C} (f : A₁ ⟶ B) (g : A₂ ⟶ B) [Mono f] [Mono g] (h : mk f = mk g) :
    A₁ ≅ A₂ where
  hom := ofMkLEMk f g h.le
  inv := ofMkLEMk g f h.ge


/-- Any functor `MonoOver X ⥤ MonoOver Y` descends to a functor
`Subobject X ⥤ Subobject Y`, because `MonoOver Y` is thin. -/
def lower {Y : D} (F : MonoOver X ⥤ MonoOver Y) : Subobject X ⥤ Subobject Y :=
  ThinSkeleton.map F


/-- Isomorphic functors become equal when lowered to `Subobject`.
(It's not as evil as usual to talk about equality between functors
because the categories are thin and skeletal.) -/
theorem lower_iso (F₁ F₂ : MonoOver X ⥤ MonoOver Y) (h : F₁ ≅ F₂) : lower F₁ = lower F₂ :=
  ThinSkeleton.map_iso_eq h


/-- A ternary version of `Subobject.lower`. -/
def lower₂ (F : MonoOver X ⥤ MonoOver Y ⥤ MonoOver Z) : Subobject X ⥤ Subobject Y ⥤ Subobject Z :=
  ThinSkeleton.map₂ F


@[simp]
theorem lower_comm (F : MonoOver Y ⥤ MonoOver X) :
    toThinSkeleton _ ⋙ lower F = F ⋙ toThinSkeleton _ :=
  rfl


/-- An adjunction between `MonoOver A` and `MonoOver B` gives an adjunction
between `Subobject A` and `Subobject B`. -/
def lowerAdjunction {A : C} {B : D} {L : MonoOver A ⥤ MonoOver B} {R : MonoOver B ⥤ MonoOver A}
    (h : L ⊣ R) : lower L ⊣ lower R :=
  ThinSkeleton.lowerAdjunction _ _ h


/-- An equivalence between `MonoOver A` and `MonoOver B` gives an equivalence
between `Subobject A` and `Subobject B`. -/
@[simps]
def lowerEquivalence {A : C} {B : D} (e : MonoOver A ≌ MonoOver B) : Subobject A ≌ Subobject B where
  functor := lower e.functor
  inverse := lower e.inverse
  unitIso := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      A : C
      B : D
      e : CategoryTheory.Equivalence (CategoryTheory.MonoOver A) (CategoryTheory.Mon …
      ⊢ CategoryTheory.Iso (CategoryTheory.Functor.id (CategoryTheory.Subobject A))  …
    -/
    apply eqToIso
    /-
      case p
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      A : C
      B : D
      e : CategoryTheory.Equivalence (CategoryTheory.MonoOver A) (CategoryTheory.Mon …
      ⊢ Eq (CategoryTheory.Functor.id (CategoryTheory.Subobject A)) ((CategoryTheory …
    -/
    convert ThinSkeleton.map_iso_eq e.unitIso
      /-
        case h.e'_2
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        A : C
        B : D
        e : CategoryTheory.Equivalence (CategoryTheory.MonoOver A) (CategoryTheory.Mon …
        ⊢ Eq (CategoryTheory.Functor.id (CategoryTheory.Subobject A)) (CategoryTheory. …
      -/
    · exact ThinSkeleton.map_id_eq.symm
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        A : C
        B : D
        e : CategoryTheory.Equivalence (CategoryTheory.MonoOver A) (CategoryTheory.Mon …
        ⊢ Eq ((CategoryTheory.Subobject.lower e.functor).comp (CategoryTheory.Subobjec …
      -/
    · exact (ThinSkeleton.map_comp_eq _ _).symm
      /-
        🎉 no goals
      -/
  counitIso := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      A : C
      B : D
      e : CategoryTheory.Equivalence (CategoryTheory.MonoOver A) (CategoryTheory.Mon …
      ⊢ CategoryTheory.Iso ((CategoryTheory.Subobject.lower e.inverse).comp (Categor …
    -/
    apply eqToIso
    /-
      case p
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      A : C
      B : D
      e : CategoryTheory.Equivalence (CategoryTheory.MonoOver A) (CategoryTheory.Mon …
      ⊢ Eq ((CategoryTheory.Subobject.lower e.inverse).comp (CategoryTheory.Subobjec …
    -/
    convert ThinSkeleton.map_iso_eq e.counitIso
      /-
        case h.e'_2
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        A : C
        B : D
        e : CategoryTheory.Equivalence (CategoryTheory.MonoOver A) (CategoryTheory.Mon …
        ⊢ Eq ((CategoryTheory.Subobject.lower e.inverse).comp (CategoryTheory.Subobjec …
      -/
    · exact (ThinSkeleton.map_comp_eq _ _).symm
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        A : C
        B : D
        e : CategoryTheory.Equivalence (CategoryTheory.MonoOver A) (CategoryTheory.Mon …
        ⊢ Eq (CategoryTheory.Functor.id (CategoryTheory.Subobject B)) (CategoryTheory. …
      -/
    · exact ThinSkeleton.map_id_eq.symm
      /-
        🎉 no goals
      -/


/-- When `C` has pullbacks, a morphism `f : X ⟶ Y` induces a functor `Subobject Y ⥤ Subobject X`,
by pulling back a monomorphism along `f`. -/
def pullback (f : X ⟶ Y) : Subobject Y ⥤ Subobject X :=
  lower (MonoOver.pullback f)


theorem pullback_id (x : Subobject X) : (pullback (𝟙 X)).obj x = x := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    x : CategoryTheory.Subobject X
    ⊢ Eq ((CategoryTheory.Subobject.pullback (CategoryTheory.CategoryStruct.id X)) …
  -/
  induction' x using Quotient.inductionOn' with f
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    f : CategoryTheory.MonoOver X
    ⊢ Eq ((CategoryTheory.Subobject.pullback (CategoryTheory.CategoryStruct.id X)) …
  -/
  exact Quotient.sound ⟨MonoOver.pullbackId.app f⟩
  /-
    🎉 no goals
  -/


theorem pullback_comp (f : X ⟶ Y) (g : Y ⟶ Z) (x : Subobject Z) :
    (pullback (f ≫ g)).obj x = (pullback f).obj ((pullback g).obj x) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    x : CategoryTheory.Subobject Z
    ⊢ Eq ((CategoryTheory.Subobject.pullback (CategoryTheory.CategoryStruct.comp f …
  -/
  induction' x using Quotient.inductionOn' with t
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    t : CategoryTheory.MonoOver Z
    ⊢ Eq ((CategoryTheory.Subobject.pullback (CategoryTheory.CategoryStruct.comp f …
  -/
  exact Quotient.sound ⟨(MonoOver.pullbackComp _ _).app t⟩
  /-
    🎉 no goals
  -/


instance (f : X ⟶ Y) : (pullback f).Faithful where


/-- We can map subobjects of `X` to subobjects of `Y`
by post-composition with a monomorphism `f : X ⟶ Y`.
-/
def map (f : X ⟶ Y) [Mono f] : Subobject X ⥤ Subobject Y :=
  lower (MonoOver.map f)


theorem map_id (x : Subobject X) : (map (𝟙 X)).obj x = x := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    x : CategoryTheory.Subobject X
    ⊢ Eq ((CategoryTheory.Subobject.map (CategoryTheory.CategoryStruct.id X)).obj  …
  -/
  induction' x using Quotient.inductionOn' with f
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    f : CategoryTheory.MonoOver X
    ⊢ Eq ((CategoryTheory.Subobject.map (CategoryTheory.CategoryStruct.id X)).obj  …
  -/
  exact Quotient.sound ⟨(MonoOver.mapId _).app f⟩
  /-
    🎉 no goals
  -/


theorem map_comp (f : X ⟶ Y) (g : Y ⟶ Z) [Mono f] [Mono g] (x : Subobject X) :
    (map (f ≫ g)).obj x = (map g).obj ((map f).obj x) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.Mono f
    inst✝ : CategoryTheory.Mono g
    x : CategoryTheory.Subobject X
    ⊢ Eq ((CategoryTheory.Subobject.map (CategoryTheory.CategoryStruct.comp f g)). …
  -/
  induction' x using Quotient.inductionOn' with t
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.Mono f
    inst✝ : CategoryTheory.Mono g
    t : CategoryTheory.MonoOver X
    ⊢ Eq ((CategoryTheory.Subobject.map (CategoryTheory.CategoryStruct.comp f g)). …
  -/
  exact Quotient.sound ⟨(MonoOver.mapComp _ _).app t⟩
  /-
    🎉 no goals
  -/


/-- Isomorphic objects have equivalent subobject lattices. -/
def mapIso {A B : C} (e : A ≅ B) : Subobject A ≌ Subobject B :=
  lowerEquivalence (MonoOver.mapIso e)

-- Porting note: the note below doesn't seem true anymore
-- @[simps] here generates a lemma `map_iso_to_order_iso_to_equiv_symm_apply`
-- whose left hand side is not in simp normal form.

/-- In fact, there's a type level bijection between the subobjects of isomorphic objects,
which preserves the order. -/
def mapIsoToOrderIso (e : X ≅ Y) : Subobject X ≃o Subobject Y where
  toFun := (map e.hom).obj
  invFun := (map e.inv).obj
                   /-
                     C : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                     X Y Z : C
                     D : Type u₂
                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                     e : CategoryTheory.Iso X Y
                     g : CategoryTheory.Subobject X
                     ⊢ Eq ((CategoryTheory.Subobject.map e.inv).obj ((CategoryTheory.Subobject.map  …
                   -/
  left_inv g := by simp_rw [← map_comp, e.hom_inv_id, map_id]
                   /-
                     🎉 no goals
                   -/
                    /-
                      C : Type u₁
                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                      X Y Z : C
                      D : Type u₂
                      inst✝ : CategoryTheory.Category.{v₂, u₂} D
                      e : CategoryTheory.Iso X Y
                      g : CategoryTheory.Subobject Y
                      ⊢ Eq ((CategoryTheory.Subobject.map e.hom).obj ((CategoryTheory.Subobject.map  …
                    -/
  right_inv g := by simp_rw [← map_comp, e.inv_hom_id, map_id]
                    /-
                      🎉 no goals
                    -/
  map_rel_iff' {A B} := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      e : CategoryTheory.Iso X Y
      A B : CategoryTheory.Subobject X
      ⊢ Iff (LE.le ({ toFun := (CategoryTheory.Subobject.map e.hom).obj, invFun := ( …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      e : CategoryTheory.Iso X Y
      A B : CategoryTheory.Subobject X
      ⊢ Iff (LE.le ((CategoryTheory.Subobject.map e.hom).obj A) ((CategoryTheory.Sub …
    -/
    constructor
      /-
        case mp
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        e : CategoryTheory.Iso X Y
        A B : CategoryTheory.Subobject X
        ⊢ LE.le ((CategoryTheory.Subobject.map e.hom).obj A) ((CategoryTheory.Subobjec …
      -/
    · intro h
      /-
        case mp
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        e : CategoryTheory.Iso X Y
        A B : CategoryTheory.Subobject X
        h : LE.le ((CategoryTheory.Subobject.map e.hom).obj A) ((CategoryTheory.Subobj …
        ⊢ LE.le A B
      -/
      apply_fun (map e.inv).obj at h
        /-
          case mp
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X Y Z : C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          e : CategoryTheory.Iso X Y
          A B : CategoryTheory.Subobject X
          h : LE.le ((CategoryTheory.Subobject.map e.inv).obj ((CategoryTheory.Subobject …
          ⊢ LE.le A B
        -/
      · simpa only [← map_comp, e.hom_inv_id, map_id] using h
        /-
          🎉 no goals
        -/
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X Y Z : C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          e : CategoryTheory.Iso X Y
          A B : CategoryTheory.Subobject X
          h : LE.le ((CategoryTheory.Subobject.map e.hom).obj A) ((CategoryTheory.Subobj …
          ⊢ Monotone (CategoryTheory.Subobject.map e.inv).obj
        -/
      · apply Functor.monotone
        /-
          🎉 no goals
        -/
      /-
        case mpr
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        e : CategoryTheory.Iso X Y
        A B : CategoryTheory.Subobject X
        ⊢ LE.le A B → LE.le ((CategoryTheory.Subobject.map e.hom).obj A) ((CategoryThe …
      -/
    · intro h
      /-
        case mpr
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        e : CategoryTheory.Iso X Y
        A B : CategoryTheory.Subobject X
        h : LE.le A B
        ⊢ LE.le ((CategoryTheory.Subobject.map e.hom).obj A) ((CategoryTheory.Subobjec …
      -/
      apply_fun (map e.hom).obj at h
        /-
          case mpr
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X Y Z : C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          e : CategoryTheory.Iso X Y
          A B : CategoryTheory.Subobject X
          h : LE.le ((CategoryTheory.Subobject.map e.hom).obj A) ((CategoryTheory.Subobj …
          ⊢ LE.le ((CategoryTheory.Subobject.map e.hom).obj A) ((CategoryTheory.Subobjec …
        -/
      · exact h
        /-
          🎉 no goals
        -/
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X Y Z : C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          e : CategoryTheory.Iso X Y
          A B : CategoryTheory.Subobject X
          h : LE.le A B
          ⊢ Monotone (CategoryTheory.Subobject.map e.hom).obj
        -/
      · apply Functor.monotone
        /-
          🎉 no goals
        -/


@[simp]
theorem mapIsoToOrderIso_apply (e : X ≅ Y) (P : Subobject X) :
    mapIsoToOrderIso e P = (map e.hom).obj P :=
  rfl


@[simp]
theorem mapIsoToOrderIso_symm_apply (e : X ≅ Y) (Q : Subobject Y) :
    (mapIsoToOrderIso e).symm Q = (map e.inv).obj Q :=
  rfl


/-- `map f : Subobject X ⥤ Subobject Y` is
the left adjoint of `pullback f : Subobject Y ⥤ Subobject X`. -/
def mapPullbackAdj [HasPullbacks C] (f : X ⟶ Y) [Mono f] : map f ⊣ pullback f :=
  lowerAdjunction (MonoOver.mapPullbackAdj f)


@[simp]
theorem pullback_map_self [HasPullbacks C] (f : X ⟶ Y) [Mono f] (g : Subobject X) :
    (pullback f).obj ((map f).obj g) = g := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    g : CategoryTheory.Subobject X
    ⊢ Eq ((CategoryTheory.Subobject.pullback f).obj ((CategoryTheory.Subobject.map …
  -/
  revert g
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    ⊢ ∀ (g : CategoryTheory.Subobject X), Eq ((CategoryTheory.Subobject.pullback f …
  -/
  exact Quotient.ind (fun g' => Quotient.sound ⟨(MonoOver.pullbackMapSelf f).app _⟩)
  /-
    🎉 no goals
  -/


theorem map_pullback [HasPullbacks C] {X Y Z W : C} {f : X ⟶ Y} {g : X ⟶ Z} {h : Y ⟶ W} {k : Z ⟶ W}
    [Mono h] [Mono g] (comm : f ≫ h = g ≫ k) (t : IsLimit (PullbackCone.mk f g comm))
    (p : Subobject Y) : (map g).obj ((pullback f).obj p) = (pullback k).obj ((map h).obj p) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Limits.HasPullbacks C
    X Y Z W : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    h : Quiver.Hom Y W
    k : Quiver.Hom Z W
    inst✝¹ : CategoryTheory.Mono h
    inst✝ : CategoryTheory.Mono g
    comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
    t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
    p : CategoryTheory.Subobject Y
    ⊢ Eq ((CategoryTheory.Subobject.map g).obj ((CategoryTheory.Subobject.pullback …
  -/
  revert p
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Limits.HasPullbacks C
    X Y Z W : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    h : Quiver.Hom Y W
    k : Quiver.Hom Z W
    inst✝¹ : CategoryTheory.Mono h
    inst✝ : CategoryTheory.Mono g
    comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
    t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
    ⊢ ∀ (p : CategoryTheory.Subobject Y), Eq ((CategoryTheory.Subobject.map g).obj …
  -/
  apply Quotient.ind'
  /-
    case h
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Limits.HasPullbacks C
    X Y Z W : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    h : Quiver.Hom Y W
    k : Quiver.Hom Z W
    inst✝¹ : CategoryTheory.Mono h
    inst✝ : CategoryTheory.Mono g
    comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
    t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
    ⊢ ∀ (a : CategoryTheory.MonoOver Y), Eq ((CategoryTheory.Subobject.map g).obj  …
  -/
  intro a
  /-
    case h
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Limits.HasPullbacks C
    X Y Z W : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    h : Quiver.Hom Y W
    k : Quiver.Hom Z W
    inst✝¹ : CategoryTheory.Mono h
    inst✝ : CategoryTheory.Mono g
    comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
    t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
    a : CategoryTheory.MonoOver Y
    ⊢ Eq ((CategoryTheory.Subobject.map g).obj ((CategoryTheory.Subobject.pullback …
  -/
  apply Quotient.sound
  /-
    case h.a
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Limits.HasPullbacks C
    X Y Z W : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    h : Quiver.Hom Y W
    k : Quiver.Hom Z W
    inst✝¹ : CategoryTheory.Mono h
    inst✝ : CategoryTheory.Mono g
    comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
    t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
    a : CategoryTheory.MonoOver Y
    ⊢ HasEquiv.Equiv ((CategoryTheory.MonoOver.map g).obj ((CategoryTheory.MonoOve …
  -/
  apply ThinSkeleton.equiv_of_both_ways
    /-
      case h.a.f
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Limits.HasPullbacks C
      X Y Z W : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      h : Quiver.Hom Y W
      k : Quiver.Hom Z W
      inst✝¹ : CategoryTheory.Mono h
      inst✝ : CategoryTheory.Mono g
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      a : CategoryTheory.MonoOver Y
      ⊢ Quiver.Hom ((CategoryTheory.MonoOver.map g).obj ((CategoryTheory.MonoOver.pu …
    -/
  · refine MonoOver.homMk (pullback.lift (pullback.fst _ _) _ ?_) (pullback.lift_snd _ _ _)
    /-
      case h.a.f
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Limits.HasPullbacks C
      X Y Z W : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      h : Quiver.Hom Y W
      k : Quiver.Hom Z W
      inst✝¹ : CategoryTheory.Mono h
      inst✝ : CategoryTheory.Mono g
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      a : CategoryTheory.MonoOver Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
    -/
    change _ ≫ a.arrow ≫ h = (pullback.snd _ _ ≫ g) ≫ _
    /-
      case h.a.f
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Limits.HasPullbacks C
      X Y Z W : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      h : Quiver.Hom Y W
      k : Quiver.Hom Z W
      inst✝¹ : CategoryTheory.Mono h
      inst✝ : CategoryTheory.Mono g
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      a : CategoryTheory.MonoOver Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
    -/
    rw [assoc, ← comm, pullback.condition_assoc]
    /-
      🎉 no goals
    -/
  · refine MonoOver.homMk (pullback.lift (pullback.fst _ _)
      (PullbackCone.IsLimit.lift t (pullback.fst _ _ ≫ a.arrow) (pullback.snd _ _) _)
      (PullbackCone.IsLimit.lift_fst _ _ _ ?_).symm) ?_
      /-
        case h.a.g.refine_1
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        inst✝² : CategoryTheory.Limits.HasPullbacks C
        X Y Z W : C
        f : Quiver.Hom X Y
        g : Quiver.Hom X Z
        h : Quiver.Hom Y W
        k : Quiver.Hom Z W
        inst✝¹ : CategoryTheory.Mono h
        inst✝ : CategoryTheory.Mono g
        comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
        t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
        a : CategoryTheory.MonoOver Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · rw [← pullback.condition, assoc]
      /-
        case h.a.g.refine_1
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        inst✝² : CategoryTheory.Limits.HasPullbacks C
        X Y Z W : C
        f : Quiver.Hom X Y
        g : Quiver.Hom X Z
        h : Quiver.Hom Y W
        k : Quiver.Hom Z W
        inst✝¹ : CategoryTheory.Mono h
        inst✝ : CategoryTheory.Mono g
        comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
        t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
        a : CategoryTheory.MonoOver Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case h.a.g.refine_2
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        inst✝² : CategoryTheory.Limits.HasPullbacks C
        X Y Z W : C
        f : Quiver.Hom X Y
        g : Quiver.Hom X Z
        h : Quiver.Hom Y W
        k : Quiver.Hom Z W
        inst✝¹ : CategoryTheory.Mono h
        inst✝ : CategoryTheory.Mono g
        comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
        t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
        a : CategoryTheory.MonoOver Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
      -/
    · dsimp
      /-
        case h.a.g.refine_2
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        inst✝² : CategoryTheory.Limits.HasPullbacks C
        X Y Z W : C
        f : Quiver.Hom X Y
        g : Quiver.Hom X Z
        h : Quiver.Hom Y W
        k : Quiver.Hom Z W
        inst✝¹ : CategoryTheory.Mono h
        inst✝ : CategoryTheory.Mono g
        comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
        t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
        a : CategoryTheory.MonoOver Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
      -/
      rw [pullback.lift_snd_assoc]
      /-
        case h.a.g.refine_2
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        inst✝² : CategoryTheory.Limits.HasPullbacks C
        X Y Z W : C
        f : Quiver.Hom X Y
        g : Quiver.Hom X Z
        h : Quiver.Hom Y W
        k : Quiver.Hom Z W
        inst✝¹ : CategoryTheory.Mono h
        inst✝ : CategoryTheory.Mono g
        comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
        t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
        a : CategoryTheory.MonoOver Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PullbackCone.I …
      -/
      apply PullbackCone.IsLimit.lift_snd
      /-
        🎉 no goals
      -/


/-- The functor from subobjects of `X` to subobjects of `Y` given by
sending the subobject `S` to its "image" under `f`, usually denoted $\exists_f$.
For instance, when `C` is the category of types,
viewing `Subobject X` as `Set X` this is just `Set.image f`.

This functor is left adjoint to the `pullback f` functor (shown in `existsPullbackAdj`)
provided both are defined, and generalises the `map f` functor, again provided it is defined.
-/
def «exists» (f : X ⟶ Y) : Subobject X ⥤ Subobject Y :=
  lower (MonoOver.exists f)


/-- When `f : X ⟶ Y` is a monomorphism, `exists f` agrees with `map f`.
-/
theorem exists_iso_map (f : X ⟶ Y) [Mono f] : «exists» f = map f :=
  lower_iso _ _ (MonoOver.existsIsoMap f)


/-- `exists f : Subobject X ⥤ Subobject Y` is
left adjoint to `pullback f : Subobject Y ⥤ Subobject X`.
-/
def existsPullbackAdj (f : X ⟶ Y) [HasPullbacks C] : «exists» f ⊣ pullback f :=
  lowerAdjunction (MonoOver.existsPullbackAdj f)


