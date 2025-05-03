/-- When `f : X ⟶ Y` and `P : MonoOver Y`,
`P.Factors f` expresses that there exists a factorisation of `f` through `P`.
Given `h : P.Factors f`, you can recover the morphism as `P.factorThru f h`.
-/
def Factors {X Y : C} (P : MonoOver Y) (f : X ⟶ Y) : Prop :=
  ∃ g : X ⟶ (P : C), g ≫ P.arrow = f


theorem factors_congr {X : C} {f g : MonoOver X} {Y : C} (h : Y ⟶ X) (e : f ≅ g) :
    f.Factors h ↔ g.Factors h :=
                                                                /-
                                                                  C : Type u₁
                                                                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                                  X : C
                                                                  f g : CategoryTheory.MonoOver X
                                                                  Y : C
                                                                  h : Quiver.Hom Y X
                                                                  e : CategoryTheory.Iso f g
                                                                  x✝ : f.Factors h
                                                                  u : Quiver.Hom Y f.obj.left
                                                                  hu : Eq (CategoryTheory.CategoryStruct.comp u f.arrow) h
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp u …
                                                                -/
  ⟨fun ⟨u, hu⟩ => ⟨u ≫ ((MonoOver.forget _).map e.hom).left, by simp [hu]⟩, fun ⟨u, hu⟩ =>
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                  /-
                                                    C : Type u₁
                                                    inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                    X : C
                                                    f g : CategoryTheory.MonoOver X
                                                    Y : C
                                                    h : Quiver.Hom Y X
                                                    e : CategoryTheory.Iso f g
                                                    x✝ : g.Factors h
                                                    u : Quiver.Hom Y g.obj.left
                                                    hu : Eq (CategoryTheory.CategoryStruct.comp u g.arrow) h
                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp u …
                                                  -/
    ⟨u ≫ ((MonoOver.forget _).map e.inv).left, by simp [hu]⟩⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- `P.factorThru f h` provides a factorisation of `f : X ⟶ Y` through some `P : MonoOver Y`,
given the evidence `h : P.Factors f` that such a factorisation exists. -/
def factorThru {X Y : C} (P : MonoOver Y) (f : X ⟶ Y) (h : Factors P f) : X ⟶ (P : C) :=
  Classical.choose h


/-- When `f : X ⟶ Y` and `P : Subobject Y`,
`P.Factors f` expresses that there exists a factorisation of `f` through `P`.
Given `h : P.Factors f`, you can recover the morphism as `P.factorThru f h`.
-/
def Factors {X Y : C} (P : Subobject Y) (f : X ⟶ Y) : Prop :=
  Quotient.liftOn' P (fun P => P.Factors f)
    (by
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X✝ Y✝ Z : C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        X Y : C
        P : CategoryTheory.Subobject Y
        f : Quiver.Hom X Y
        ⊢ ∀ (a b : CategoryTheory.MonoOver Y), (CategoryTheory.isIsomorphicSetoid (Cat …
      -/
      rintro P Q ⟨h⟩
      /-
        case intro
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X✝ Y✝ Z : C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        X Y : C
        P✝ : CategoryTheory.Subobject Y
        f : Quiver.Hom X Y
        P Q : CategoryTheory.MonoOver Y
        h : CategoryTheory.Iso P Q
        ⊢ Eq ((fun P => P.Factors f) P) ((fun P => P.Factors f) Q)
      -/
      apply propext
      /-
        case intro.a
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X✝ Y✝ Z : C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        X Y : C
        P✝ : CategoryTheory.Subobject Y
        f : Quiver.Hom X Y
        P Q : CategoryTheory.MonoOver Y
        h : CategoryTheory.Iso P Q
        ⊢ Iff ((fun P => P.Factors f) P) ((fun P => P.Factors f) Q)
      -/
      constructor
        /-
          case intro.a.mp
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X✝ Y✝ Z : C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          X Y : C
          P✝ : CategoryTheory.Subobject Y
          f : Quiver.Hom X Y
          P Q : CategoryTheory.MonoOver Y
          h : CategoryTheory.Iso P Q
          ⊢ (fun P => P.Factors f) P → (fun P => P.Factors f) Q
        -/
      · rintro ⟨i, w⟩
        /-
          case intro.a.mp.intro
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X✝ Y✝ Z : C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          X Y : C
          P✝ : CategoryTheory.Subobject Y
          f : Quiver.Hom X Y
          P Q : CategoryTheory.MonoOver Y
          h : CategoryTheory.Iso P Q
          i : Quiver.Hom X P.obj.left
          w : Eq (CategoryTheory.CategoryStruct.comp i P.arrow) f
          ⊢ Q.Factors f
        -/
        exact ⟨i ≫ h.hom.left, by erw [Category.assoc, Over.w h.hom, w]⟩
        /-
          🎉 no goals
        -/
        /-
          case intro.a.mpr
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X✝ Y✝ Z : C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          X Y : C
          P✝ : CategoryTheory.Subobject Y
          f : Quiver.Hom X Y
          P Q : CategoryTheory.MonoOver Y
          h : CategoryTheory.Iso P Q
          ⊢ (fun P => P.Factors f) Q → (fun P => P.Factors f) P
        -/
      · rintro ⟨i, w⟩
        /-
          case intro.a.mpr.intro
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X✝ Y✝ Z : C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          X Y : C
          P✝ : CategoryTheory.Subobject Y
          f : Quiver.Hom X Y
          P Q : CategoryTheory.MonoOver Y
          h : CategoryTheory.Iso P Q
          i : Quiver.Hom X Q.obj.left
          w : Eq (CategoryTheory.CategoryStruct.comp i Q.arrow) f
          ⊢ P.Factors f
        -/
        exact ⟨i ≫ h.inv.left, by erw [Category.assoc, Over.w h.inv, w]⟩)
        /-
          🎉 no goals
        -/


@[simp]
theorem mk_factors_iff {X Y Z : C} (f : Y ⟶ X) [Mono f] (g : Z ⟶ X) :
    (Subobject.mk f).Factors g ↔ (MonoOver.mk' f).Factors g :=
  Iff.rfl


theorem mk_factors_self (f : X ⟶ Y) [Mono f] : (mk f).Factors f :=
           /-
             C : Type u₁
             inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
             X Y : C
             f : Quiver.Hom X Y
             inst✝ : CategoryTheory.Mono f
             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
           -/
  ⟨𝟙 _, by simp⟩
           /-
             🎉 no goals
           -/


theorem factors_iff {X Y : C} (P : Subobject Y) (f : X ⟶ Y) :
    P.Factors f ↔ (representative.obj P).Factors f :=
  Quot.inductionOn P fun _ => MonoOver.factors_congr _ (representativeIso _).symm


theorem factors_self {X : C} (P : Subobject X) : P.Factors P.arrow :=
                                       /-
                                         C : Type u₁
                                         inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                         X : C
                                         P : CategoryTheory.Subobject X
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Ca …
                                       -/
  (factors_iff _ _).mpr ⟨𝟙 (P : C), by simp⟩
                                       /-
                                         🎉 no goals
                                       -/


theorem factors_comp_arrow {X Y : C} {P : Subobject Y} (f : X ⟶ P) : P.Factors (f ≫ P.arrow) :=
  (factors_iff _ _).mpr ⟨f, rfl⟩


theorem factors_of_factors_right {X Y Z : C} {P : Subobject Z} (f : X ⟶ Y) {g : Y ⟶ Z}
    (h : P.Factors g) : P.Factors (f ≫ g) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    P : CategoryTheory.Subobject Z
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : P.Factors g
    ⊢ P.Factors (CategoryTheory.CategoryStruct.comp f g)
  -/
  induction' P using Quotient.ind' with P
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    P : CategoryTheory.MonoOver Z
    h : CategoryTheory.Subobject.Factors (Quotient.mk'' P) g
    ⊢ CategoryTheory.Subobject.Factors (Quotient.mk'' P) (CategoryTheory.CategoryS …
  -/
  obtain ⟨g, rfl⟩ := h
  /-
    case h.intro
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    f : Quiver.Hom X Y
    P : CategoryTheory.MonoOver Z
    g : Quiver.Hom Y P.obj.left
    ⊢ CategoryTheory.Subobject.Factors (Quotient.mk'' P) (CategoryTheory.CategoryS …
  -/
  exact ⟨f ≫ g, by simp⟩
  /-
    🎉 no goals
  -/


theorem factors_zero [HasZeroMorphisms C] {X Y : C} {P : Subobject Y} : P.Factors (0 : X ⟶ Y) :=
                               /-
                                 C : Type u₁
                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                 X Y : C
                                 P : CategoryTheory.Subobject Y
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 (CategoryTheory.Subobject.represent …
                               -/
  (factors_iff _ _).mpr ⟨0, by simp⟩
                               /-
                                 🎉 no goals
                               -/


theorem factors_of_le {Y Z : C} {P Q : Subobject Y} (f : Z ⟶ Y) (h : P ≤ Q) :
    P.Factors f → Q.Factors f := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    Y Z : C
    P Q : CategoryTheory.Subobject Y
    f : Quiver.Hom Z Y
    h : LE.le P Q
    ⊢ P.Factors f → Q.Factors f
  -/
  simp only [factors_iff]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    Y Z : C
    P Q : CategoryTheory.Subobject Y
    f : Quiver.Hom Z Y
    h : LE.le P Q
    ⊢ (CategoryTheory.Subobject.representative.obj P).Factors f → (CategoryTheory. …
  -/
  exact fun ⟨u, hu⟩ => ⟨u ≫ ofLE _ _ h, by simp [← hu]⟩
  /-
    🎉 no goals
  -/


/-- `P.factorThru f h` provides a factorisation of `f : X ⟶ Y` through some `P : Subobject Y`,
given the evidence `h : P.Factors f` that such a factorisation exists. -/
def factorThru {X Y : C} (P : Subobject Y) (f : X ⟶ Y) (h : Factors P f) : X ⟶ P :=
  Classical.choose ((factors_iff _ _).mp h)


@[reassoc (attr := simp)]
theorem factorThru_arrow {X Y : C} (P : Subobject Y) (f : X ⟶ Y) (h : Factors P f) :
    P.factorThru f h ≫ P.arrow = f :=
  Classical.choose_spec ((factors_iff _ _).mp h)


@[simp]
theorem factorThru_self {X : C} (P : Subobject X) (h) : P.factorThru P.arrow h = 𝟙 (P : C) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    P : CategoryTheory.Subobject X
    h : P.Factors P.arrow
    ⊢ Eq (P.factorThru P.arrow h) (CategoryTheory.CategoryStruct.id (CategoryTheor …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    P : CategoryTheory.Subobject X
    h : P.Factors P.arrow
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.factorThru P.arrow h) P.arrow) (Ca …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem factorThru_mk_self (f : X ⟶ Y) [Mono f] :
    (mk f).factorThru f (mk_factors_self f) = (underlyingIso f).inv := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    ⊢ Eq ((CategoryTheory.Subobject.mk f).factorThru f ⋯) (CategoryTheory.Subobjec …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Subobject.mk f).fact …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem factorThru_comp_arrow {X Y : C} {P : Subobject Y} (f : X ⟶ P) (h) :
    P.factorThru (f ≫ P.arrow) h = f := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    P : CategoryTheory.Subobject Y
    f : Quiver.Hom X (CategoryTheory.Subobject.underlying.obj P)
    h : P.Factors (CategoryTheory.CategoryStruct.comp f P.arrow)
    ⊢ Eq (P.factorThru (CategoryTheory.CategoryStruct.comp f P.arrow) h) f
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    P : CategoryTheory.Subobject Y
    f : Quiver.Hom X (CategoryTheory.Subobject.underlying.obj P)
    h : P.Factors (CategoryTheory.CategoryStruct.comp f P.arrow)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.factorThru (CategoryTheory.Categor …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem factorThru_eq_zero [HasZeroMorphisms C] {X Y : C} {P : Subobject Y} {f : X ⟶ Y}
    {h : Factors P f} : P.factorThru f h = 0 ↔ f = 0 := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    P : CategoryTheory.Subobject Y
    f : Quiver.Hom X Y
    h : P.Factors f
    ⊢ Iff (Eq (P.factorThru f h) 0) (Eq f 0)
  -/
  fconstructor
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      P : CategoryTheory.Subobject Y
      f : Quiver.Hom X Y
      h : P.Factors f
      ⊢ Eq (P.factorThru f h) 0 → Eq f 0
    -/
  · intro w
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      P : CategoryTheory.Subobject Y
      f : Quiver.Hom X Y
      h : P.Factors f
      w : Eq (P.factorThru f h) 0
      ⊢ Eq f 0
    -/
    replace w := w =≫ P.arrow
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      P : CategoryTheory.Subobject Y
      f : Quiver.Hom X Y
      h : P.Factors f
      w : Eq (CategoryTheory.CategoryStruct.comp (P.factorThru f h) P.arrow) (Catego …
      ⊢ Eq f 0
    -/
    simpa using w
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      P : CategoryTheory.Subobject Y
      f : Quiver.Hom X Y
      h : P.Factors f
      ⊢ Eq f 0 → Eq (P.factorThru f h) 0
    -/
  · rintro rfl
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      P : CategoryTheory.Subobject Y
      h : P.Factors 0
      ⊢ Eq (P.factorThru 0 h) 0
    -/
    ext
    /-
      case mpr.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      P : CategoryTheory.Subobject Y
      h : P.Factors 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.factorThru 0 h) P.arrow) (Category …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem factorThru_right {X Y Z : C} {P : Subobject Z} (f : X ⟶ Y) (g : Y ⟶ Z) (h : P.Factors g) :
    f ≫ P.factorThru g h = P.factorThru (f ≫ g) (factors_of_factors_right f h) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    P : CategoryTheory.Subobject Z
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : P.Factors g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (P.factorThru g h)) (P.factorThru ( …
  -/
  apply (cancel_mono P.arrow).mp
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    P : CategoryTheory.Subobject Z
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : P.Factors g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem factorThru_zero [HasZeroMorphisms C] {X Y : C} {P : Subobject Y}
                                                             /-
                                                               C : Type u₁
                                                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                               inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                               X Y : C
                                                               P : CategoryTheory.Subobject Y
                                                               h : P.Factors 0
                                                               ⊢ Eq (P.factorThru 0 h) 0
                                                             -/
    (h : P.Factors (0 : X ⟶ Y)) : P.factorThru 0 h = 0 := by simp
                                                             /-
                                                               🎉 no goals
                                                             -/

-- `h` is an explicit argument here so we can use
-- `rw factorThru_ofLE h`, obtaining a subgoal `P.Factors f`.
-- (While the reverse direction looks plausible as a simp lemma, it seems to be unproductive.)

theorem factorThru_ofLE {Y Z : C} {P Q : Subobject Y} {f : Z ⟶ Y} (h : P ≤ Q) (w : P.Factors f) :
    Q.factorThru f (factors_of_le f h w) = P.factorThru f w ≫ ofLE P Q h := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    Y Z : C
    P Q : CategoryTheory.Subobject Y
    f : Quiver.Hom Z Y
    h : LE.le P Q
    w : P.Factors f
    ⊢ Eq (Q.factorThru f ⋯) (CategoryTheory.CategoryStruct.comp (P.factorThru f w) …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    Y Z : C
    P Q : CategoryTheory.Subobject Y
    f : Quiver.Hom Z Y
    h : LE.le P Q
    w : P.Factors f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Q.factorThru f ⋯) Q.arrow) (Category …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem factors_add {X Y : C} {P : Subobject Y} (f g : X ⟶ Y) (wf : P.Factors f)
    (wg : P.Factors g) : P.Factors (f + g) :=
                                                                   /-
                                                                     C : Type u₁
                                                                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                     inst✝ : CategoryTheory.Preadditive C
                                                                     X Y : C
                                                                     P : CategoryTheory.Subobject Y
                                                                     f g : Quiver.Hom X Y
                                                                     wf : P.Factors f
                                                                     wg : P.Factors g
                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd (P.factorThru f wf) (P.fac …
                                                                   -/
  (factors_iff _ _).mpr ⟨P.factorThru f wf + P.factorThru g wg, by simp⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/

-- This can't be a `simp` lemma as `wf` and `wg` may not exist.
-- However you can `rw` by it to assert that `f` and `g` factor through `P` separately.

theorem factorThru_add {X Y : C} {P : Subobject Y} (f g : X ⟶ Y) (w : P.Factors (f + g))
    (wf : P.Factors f) (wg : P.Factors g) :
    P.factorThru (f + g) w = P.factorThru f wf + P.factorThru g wg := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    P : CategoryTheory.Subobject Y
    f g : Quiver.Hom X Y
    w : P.Factors (HAdd.hAdd f g)
    wf : P.Factors f
    wg : P.Factors g
    ⊢ Eq (P.factorThru (HAdd.hAdd f g) w) (HAdd.hAdd (P.factorThru f wf) (P.factor …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    P : CategoryTheory.Subobject Y
    f g : Quiver.Hom X Y
    w : P.Factors (HAdd.hAdd f g)
    wf : P.Factors f
    wg : P.Factors g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.factorThru (HAdd.hAdd f g) w) P.ar …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem factors_left_of_factors_add {X Y : C} {P : Subobject Y} (f g : X ⟶ Y)
    (w : P.Factors (f + g)) (wg : P.Factors g) : P.Factors f :=
                                                                        /-
                                                                          C : Type u₁
                                                                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                          inst✝ : CategoryTheory.Preadditive C
                                                                          X Y : C
                                                                          P : CategoryTheory.Subobject Y
                                                                          f g : Quiver.Hom X Y
                                                                          w : P.Factors (HAdd.hAdd f g)
                                                                          wg : P.Factors g
                                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (P.factorThru (HAdd.hAdd f …
                                                                        -/
  (factors_iff _ _).mpr ⟨P.factorThru (f + g) w - P.factorThru g wg, by simp⟩
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
theorem factorThru_add_sub_factorThru_right {X Y : C} {P : Subobject Y} (f g : X ⟶ Y)
    (w : P.Factors (f + g)) (wg : P.Factors g) :
    P.factorThru (f + g) w - P.factorThru g wg =
      P.factorThru f (factors_left_of_factors_add f g w wg) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    P : CategoryTheory.Subobject Y
    f g : Quiver.Hom X Y
    w : P.Factors (HAdd.hAdd f g)
    wg : P.Factors g
    ⊢ Eq (HSub.hSub (P.factorThru (HAdd.hAdd f g) w) (P.factorThru g wg)) (P.facto …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    P : CategoryTheory.Subobject Y
    f g : Quiver.Hom X Y
    w : P.Factors (HAdd.hAdd f g)
    wg : P.Factors g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (P.factorThru (HAdd.hAdd f …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem factors_right_of_factors_add {X Y : C} {P : Subobject Y} (f g : X ⟶ Y)
    (w : P.Factors (f + g)) (wf : P.Factors f) : P.Factors g :=
                                                                        /-
                                                                          C : Type u₁
                                                                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                          inst✝ : CategoryTheory.Preadditive C
                                                                          X Y : C
                                                                          P : CategoryTheory.Subobject Y
                                                                          f g : Quiver.Hom X Y
                                                                          w : P.Factors (HAdd.hAdd f g)
                                                                          wf : P.Factors f
                                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (P.factorThru (HAdd.hAdd f …
                                                                        -/
  (factors_iff _ _).mpr ⟨P.factorThru (f + g) w - P.factorThru f wf, by simp⟩
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
theorem factorThru_add_sub_factorThru_left {X Y : C} {P : Subobject Y} (f g : X ⟶ Y)
    (w : P.Factors (f + g)) (wf : P.Factors f) :
    P.factorThru (f + g) w - P.factorThru f wf =
      P.factorThru g (factors_right_of_factors_add f g w wf) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    P : CategoryTheory.Subobject Y
    f g : Quiver.Hom X Y
    w : P.Factors (HAdd.hAdd f g)
    wf : P.Factors f
    ⊢ Eq (HSub.hSub (P.factorThru (HAdd.hAdd f g) w) (P.factorThru f wf)) (P.facto …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    P : CategoryTheory.Subobject Y
    f g : Quiver.Hom X Y
    w : P.Factors (HAdd.hAdd f g)
    wf : P.Factors f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (P.factorThru (HAdd.hAdd f …
  -/
  simp
  /-
    🎉 no goals
  -/


