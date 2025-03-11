/--
An object `P` is called *projective* if every morphism out of `P` factors through every epimorphism.
-/
class Projective (P : C) : Prop where
  factors : ∀ {E X : C} (f : P ⟶ X) (e : E ⟶ X) [Epi e], ∃ f', f' ≫ e = f


lemma Limits.IsZero.projective {X : C} (h : IsZero X) : Projective X where
  factors _ _ _ := ⟨h.to_ _, h.eq_of_src _ _⟩


/-- A projective presentation of an object `X` consists of an epimorphism `f : P ⟶ X`
from some projective object `P`.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): was @[nolint has_nonempty_instance]
structure ProjectivePresentation (X : C) where
  p : C
  [projective : Projective p]
  f : p ⟶ X
  [epi : Epi f]


/-- A category "has enough projectives" if for every object `X` there is a projective object `P` and
    an epimorphism `P ↠ X`. -/
class EnoughProjectives : Prop where
  presentation : ∀ X : C, Nonempty (ProjectivePresentation X)


/--
An arbitrarily chosen factorisation of a morphism out of a projective object through an epimorphism.
-/
def factorThru {P X E : C} [Projective P] (f : P ⟶ X) (e : E ⟶ X) [Epi e] : P ⟶ E :=
  (Projective.factors f e).choose


@[reassoc (attr := simp)]
theorem factorThru_comp {P X E : C} [Projective P] (f : P ⟶ X) (e : E ⟶ X) [Epi e] :
    factorThru f e ≫ e = f :=
  (Projective.factors f e).choose_spec


instance zero_projective [HasZeroObject C] : Projective (0 : C) :=
  (isZero_zero C).projective


theorem of_iso {P Q : C} (i : P ≅ Q) (_ : Projective P) : Projective Q where
  factors f e _ :=
    let ⟨f', hf'⟩ := Projective.factors (i.hom ≫ f) e
                    /-
                      C : Type u
                      inst✝ : CategoryTheory.Category.{v, u} C
                      P Q : C
                      i : CategoryTheory.Iso P Q
                      x✝¹ : CategoryTheory.Projective P
                      E✝ X✝ : C
                      f : Quiver.Hom Q X✝
                      e : Quiver.Hom E✝ X✝
                      x✝ : CategoryTheory.Epi e
                      f' : Quiver.Hom P E✝
                      hf' : Eq (CategoryTheory.CategoryStruct.comp f' e) (CategoryTheory.CategoryStr …
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp i …
                    -/
    ⟨i.inv ≫ f', by simp [hf']⟩
                    /-
                      🎉 no goals
                    -/


theorem iso_iff {P Q : C} (i : P ≅ Q) : Projective P ↔ Projective Q :=
  ⟨of_iso i, of_iso i.symm⟩


/-- The axiom of choice says that every type is a projective object in `Type`. -/
instance (X : Type u) : Projective X where
  factors f e _ :=
    have he : Function.Surjective e := surjective_of_epi e
    ⟨fun x => (he (f x)).choose, funext fun x ↦ (he (f x)).choose_spec⟩


instance Type.enoughProjectives : EnoughProjectives (Type u) where
  presentation X := ⟨⟨X, 𝟙 X⟩⟩


instance {P Q : C} [HasBinaryCoproduct P Q] [Projective P] [Projective Q] : Projective (P ⨿ Q) where
  factors f e epi := ⟨coprod.desc (factorThru (coprod.inl ≫ f) e) (factorThru (coprod.inr ≫ f) e),
       /-
         C : Type u
         inst✝³ : CategoryTheory.Category.{v, u} C
         P Q : C
         inst✝² : CategoryTheory.Limits.HasBinaryCoproduct P Q
         inst✝¹ : CategoryTheory.Projective P
         inst✝ : CategoryTheory.Projective Q
         E✝ X✝ : C
         f : Quiver.Hom (CategoryTheory.Limits.coprod P Q) X✝
         e : Quiver.Hom E✝ X✝
         epi : CategoryTheory.Epi e
         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.desc (C …
       -/
    by aesop_cat⟩
       /-
         🎉 no goals
       -/


instance {β : Type v} (g : β → C) [HasCoproduct g] [∀ b, Projective (g b)] : Projective (∐ g) where
                                                                             /-
                                                                               C : Type u
                                                                               inst✝² : CategoryTheory.Category.{v, u} C
                                                                               β : Type v
                                                                               g : β → C
                                                                               inst✝¹ : CategoryTheory.Limits.HasCoproduct g
                                                                               inst✝ : ∀ (b : β), CategoryTheory.Projective (g b)
                                                                               E✝ X✝ : C
                                                                               f : Quiver.Hom (CategoryTheory.Limits.sigmaObj g) X✝
                                                                               e : Quiver.Hom E✝ X✝
                                                                               epi : CategoryTheory.Epi e
                                                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.desc fun …
                                                                             -/
  factors f e epi := ⟨Sigma.desc fun b => factorThru (Sigma.ι g b ≫ f) e, by aesop_cat⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


instance {P Q : C} [HasZeroMorphisms C] [HasBinaryBiproduct P Q] [Projective P] [Projective Q] :
    Projective (P ⊞ Q) where
  factors f e epi := ⟨biprod.desc (factorThru (biprod.inl ≫ f) e) (factorThru (biprod.inr ≫ f) e),
       /-
         C : Type u
         inst✝⁴ : CategoryTheory.Category.{v, u} C
         P Q : C
         inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
         inst✝² : CategoryTheory.Limits.HasBinaryBiproduct P Q
         inst✝¹ : CategoryTheory.Projective P
         inst✝ : CategoryTheory.Projective Q
         E✝ X✝ : C
         f : Quiver.Hom (CategoryTheory.Limits.biprod P Q) X✝
         e : Quiver.Hom E✝ X✝
         epi : CategoryTheory.Epi e
         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.desc (C …
       -/
    by aesop_cat⟩
       /-
         🎉 no goals
       -/


instance {β : Type v} (g : β → C) [HasZeroMorphisms C] [HasBiproduct g] [∀ b, Projective (g b)] :
    Projective (⨁ g) where
                                                                                     /-
                                                                                       C : Type u
                                                                                       inst✝³ : CategoryTheory.Category.{v, u} C
                                                                                       β : Type v
                                                                                       g : β → C
                                                                                       inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                       inst✝¹ : CategoryTheory.Limits.HasBiproduct g
                                                                                       inst✝ : ∀ (b : β), CategoryTheory.Projective (g b)
                                                                                       E✝ X✝ : C
                                                                                       f : Quiver.Hom (CategoryTheory.Limits.biproduct g) X✝
                                                                                       e : Quiver.Hom E✝ X✝
                                                                                       epi : CategoryTheory.Epi e
                                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.desc …
                                                                                     -/
  factors f e epi := ⟨biproduct.desc fun b => factorThru (biproduct.ι g b ≫ f) e, by aesop_cat⟩
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem projective_iff_preservesEpimorphisms_coyoneda_obj (P : C) :
    Projective P ↔ (coyoneda.obj (op P)).PreservesEpimorphisms :=
  ⟨fun hP =>
    ⟨fun f _ =>
      (epi_iff_surjective _).2 fun g =>
        have : Projective (unop (op P)) := hP
        ⟨factorThru g f, factorThru_comp _ _⟩⟩,
    fun _ =>
    ⟨fun f e _ =>
      (epi_iff_surjective _).1 (inferInstance : Epi ((coyoneda.obj (op P)).map e)) f⟩⟩


/-- `Projective.over X` provides an arbitrarily chosen projective object equipped with
an epimorphism `Projective.π : Projective.over X ⟶ X`.
-/
def over (X : C) : C :=
  (EnoughProjectives.presentation X).some.p


instance projective_over (X : C) : Projective (over X) :=
  (EnoughProjectives.presentation X).some.projective


/-- The epimorphism `projective.π : projective.over X ⟶ X`
from the arbitrarily chosen projective object over `X`.
-/
def π (X : C) : over X ⟶ X :=
  (EnoughProjectives.presentation X).some.f


instance π_epi (X : C) : Epi (π X) :=
  (EnoughProjectives.presentation X).some.epi


/-- When `C` has enough projectives, the object `Projective.syzygies f` is
an arbitrarily chosen projective object over `kernel f`.
-/
def syzygies : C := over (kernel f)


instance : Projective (syzygies f) := inferInstanceAs (Projective (over _))


/-- When `C` has enough projectives,
`Projective.d f : Projective.syzygies f ⟶ X` is the composition
`π (kernel f) ≫ kernel.ι f`.

(When `C` is abelian, we have `exact (projective.d f) f`.)
-/
abbrev d : syzygies f ⟶ X :=
  π (kernel f) ≫ kernel.ι f


theorem map_projective (adj : F ⊣ G) [G.PreservesEpimorphisms] (P : C) (hP : Projective P) :
    Projective (F.obj P) where
  factors f g _ := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : G.PreservesEpimorphisms
      P : C
      hP : CategoryTheory.Projective P
      E✝ X✝ : D
      f : Quiver.Hom (F.obj P) X✝
      g : Quiver.Hom E✝ X✝
      x✝ : CategoryTheory.Epi g
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' g) f
    -/
    rcases hP.factors (adj.unit.app P ≫ G.map f) (G.map g) with ⟨f', hf'⟩
    /-
      case intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : G.PreservesEpimorphisms
      P : C
      hP : CategoryTheory.Projective P
      E✝ X✝ : D
      f : Quiver.Hom (F.obj P) X✝
      g : Quiver.Hom E✝ X✝
      x✝ : CategoryTheory.Epi g
      f' : Quiver.Hom P (G.obj E✝)
      hf' : Eq (CategoryTheory.CategoryStruct.comp f' (G.map g)) (CategoryTheory.Cat …
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' g) f
    -/
    use F.map f' ≫ adj.counit.app _
    /-
      case h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : G.PreservesEpimorphisms
      P : C
      hP : CategoryTheory.Projective P
      E✝ X✝ : D
      f : Quiver.Hom (F.obj P) X✝
      g : Quiver.Hom E✝ X✝
      x✝ : CategoryTheory.Epi g
      f' : Quiver.Hom P (G.obj E✝)
      hf' : Eq (CategoryTheory.CategoryStruct.comp f' (G.map g)) (CategoryTheory.Cat …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [Category.assoc, ← Adjunction.counit_naturality, ← Category.assoc, ← F.map_comp, hf']
    /-
      case h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : G.PreservesEpimorphisms
      P : C
      hP : CategoryTheory.Projective P
      E✝ X✝ : D
      f : Quiver.Hom (F.obj P) X✝
      g : Quiver.Hom E✝ X✝
      x✝ : CategoryTheory.Epi g
      f' : Quiver.Hom P (G.obj E✝)
      hf' : Eq (CategoryTheory.CategoryStruct.comp f' (G.map g)) (CategoryTheory.Cat …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem projective_of_map_projective (adj : F ⊣ G) [F.Full] [F.Faithful] (P : C)
    (hP : Projective (F.obj P)) : Projective P where
  factors f g _ := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      P : C
      hP : CategoryTheory.Projective (F.obj P)
      E✝ X✝ : C
      f : Quiver.Hom P X✝
      g : Quiver.Hom E✝ X✝
      x✝ : CategoryTheory.Epi g
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' g) f
    -/
    haveI := Adjunction.leftAdjoint_preservesColimits.{0, 0} adj
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      P : C
      hP : CategoryTheory.Projective (F.obj P)
      E✝ X✝ : C
      f : Quiver.Hom P X✝
      g : Quiver.Hom E✝ X✝
      x✝ : CategoryTheory.Epi g
      this : CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v, v', u, u'} F
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' g) f
    -/
    rcases (@hP).1 (F.map f) (F.map g) with ⟨f', hf'⟩
    /-
      case intro
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      P : C
      hP : CategoryTheory.Projective (F.obj P)
      E✝ X✝ : C
      f : Quiver.Hom P X✝
      g : Quiver.Hom E✝ X✝
      x✝ : CategoryTheory.Epi g
      this : CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v, v', u, u'} F
      f' : Quiver.Hom (F.obj P) (F.obj E✝)
      hf' : Eq (CategoryTheory.CategoryStruct.comp f' (F.map g)) (F.map f)
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' g) f
    -/
    use adj.unit.app _ ≫ G.map f' ≫ (inv <| adj.unit.app _)
    /-
      case h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      P : C
      hP : CategoryTheory.Projective (F.obj P)
      E✝ X✝ : C
      f : Quiver.Hom P X✝
      g : Quiver.Hom E✝ X✝
      x✝ : CategoryTheory.Epi g
      this : CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v, v', u, u'} F
      f' : Quiver.Hom (F.obj P) (F.obj E✝)
      hf' : Eq (CategoryTheory.CategoryStruct.comp f' (F.map g)) (F.map f)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    exact F.map_injective (by simpa)
    /-
      🎉 no goals
    -/


/-- Given an adjunction `F ⊣ G` such that `G` preserves epis, `F` maps a projective presentation of
`X` to a projective presentation of `F(X)`. -/
def mapProjectivePresentation (adj : F ⊣ G) [G.PreservesEpimorphisms] (X : C)
    (Y : ProjectivePresentation X) : ProjectivePresentation (F.obj X) where
  p := F.obj Y.p
  projective := adj.map_projective _ Y.projective
  f := F.map Y.f
  epi := have := Adjunction.leftAdjoint_preservesColimits.{0, 0} adj; inferInstance


theorem map_projective_iff (P : C) : Projective (F.functor.obj P) ↔ Projective P :=
  ⟨F.toAdjunction.projective_of_map_projective P, F.toAdjunction.map_projective P⟩


/-- Given an equivalence of categories `F`, a projective presentation of `F(X)` induces a
projective presentation of `X.` -/
def projectivePresentationOfMapProjectivePresentation (X : C)
    (Y : ProjectivePresentation (F.functor.obj X)) : ProjectivePresentation X where
  p := F.inverse.obj Y.p
  projective := Adjunction.map_projective F.symm.toAdjunction Y.p Y.projective
  f := F.inverse.map Y.f ≫ F.unitInv.app _
  epi := epi_comp _ _


theorem enoughProjectives_iff (F : C ≌ D) : EnoughProjectives C ↔ EnoughProjectives D := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Equivalence C D
    ⊢ Iff (CategoryTheory.EnoughProjectives C) (CategoryTheory.EnoughProjectives D)
  -/
  constructor
  /-
    case mp
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Equivalence C D
    ⊢ CategoryTheory.EnoughProjectives C → CategoryTheory.EnoughProjectives D
  -/
  all_goals intro H; constructor; intro X; constructor
  · exact F.symm.projectivePresentationOfMapProjectivePresentation _
      (Nonempty.some (H.presentation (F.inverse.obj X)))
  · exact F.projectivePresentationOfMapProjectivePresentation X
      (Nonempty.some (H.presentation (F.functor.obj X)))


