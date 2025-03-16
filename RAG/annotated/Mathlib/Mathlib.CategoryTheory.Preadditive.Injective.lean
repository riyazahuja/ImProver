/--
An object `J` is injective iff every morphism into `J` can be obtained by extending a monomorphism.
-/
class Injective (J : C) : Prop where
  factors : ∀ {X Y : C} (g : X ⟶ J) (f : X ⟶ Y) [Mono f], ∃ h : Y ⟶ J, f ≫ h = g


lemma Limits.IsZero.injective {X : C} (h : IsZero X) : Injective X where
  factors _ _ _ := ⟨h.from_ _, h.eq_of_tgt _ _⟩


/-- An injective presentation of an object `X` consists of a monomorphism `f : X ⟶ J`
to some injective object `J`.
-/
structure InjectivePresentation (X : C) where
  J : C
  injective : Injective J := by infer_instance
  f : X ⟶ J
  mono : Mono f := by infer_instance


/-- A category "has enough injectives" if every object has an injective presentation,
i.e. if for every object `X` there is an injective object `J` and a monomorphism `X ↪ J`. -/
class EnoughInjectives : Prop where
  presentation : ∀ X : C, Nonempty (InjectivePresentation X)


/--
Let `J` be injective and `g` a morphism into `J`, then `g` can be factored through any monomorphism.
-/
def factorThru {J X Y : C} [Injective J] (g : X ⟶ J) (f : X ⟶ Y) [Mono f] : Y ⟶ J :=
  (Injective.factors g f).choose


@[simp]
theorem comp_factorThru {J X Y : C} [Injective J] (g : X ⟶ J) (f : X ⟶ Y) [Mono f] :
    f ≫ factorThru g f = g :=
  (Injective.factors g f).choose_spec


instance zero_injective [HasZeroObject C] : Injective (0 : C) :=
  (isZero_zero C).injective


theorem of_iso {P Q : C} (i : P ≅ Q) (hP : Injective P) : Injective Q :=
  {
    factors := fun g f mono => by
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        P Q : C
        i : CategoryTheory.Iso P Q
        hP : CategoryTheory.Injective P
        X✝ Y✝ : C
        g : Quiver.Hom X✝ Q
        f : Quiver.Hom X✝ Y✝
        mono : CategoryTheory.Mono f
        ⊢ Exists fun h => Eq (CategoryTheory.CategoryStruct.comp f h) g
      -/
      obtain ⟨h, h_eq⟩ := @Injective.factors C _ P _ _ _ (g ≫ i.inv) f mono
      /-
        case intro
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        P Q : C
        i : CategoryTheory.Iso P Q
        hP : CategoryTheory.Injective P
        X✝ Y✝ : C
        g : Quiver.Hom X✝ Q
        f : Quiver.Hom X✝ Y✝
        mono : CategoryTheory.Mono f
        h : Quiver.Hom Y✝ P
        h_eq : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
        ⊢ Exists fun h => Eq (CategoryTheory.CategoryStruct.comp f h) g
      -/
      refine ⟨h ≫ i.hom, ?_⟩
      /-
        case intro
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        P Q : C
        i : CategoryTheory.Iso P Q
        hP : CategoryTheory.Injective P
        X✝ Y✝ : C
        g : Quiver.Hom X✝ Q
        f : Quiver.Hom X✝ Y✝
        mono : CategoryTheory.Mono f
        h : Quiver.Hom Y✝ P
        h_eq : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
      -/
      rw [← Category.assoc, h_eq, Category.assoc, Iso.inv_hom_id, Category.comp_id] }
      /-
        🎉 no goals
      -/


theorem iso_iff {P Q : C} (i : P ≅ Q) : Injective P ↔ Injective Q :=
  ⟨of_iso i, of_iso i.symm⟩


/-- The axiom of choice says that every nonempty type is an injective object in `Type`. -/
instance (X : Type u₁) [Nonempty X] : Injective X where
  factors g f mono :=
    ⟨fun z => by
      classical
      exact
          if h : z ∈ Set.range f then g (Classical.choose h) else Nonempty.some inferInstance, by
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X : Type u₁
        inst✝ : Nonempty X
        X✝ Y✝ : Type u₁
        g : Quiver.Hom X✝ X
        f : Quiver.Hom X✝ Y✝
        mono : CategoryTheory.Mono f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f fun z => dite (Membership.mem (Set. …
      -/
      ext y
      classical
      change dite (f y ∈ Set.range f) (fun h => g (Classical.choose h)) _ = _
      split_ifs <;> rename_i h
      · rw [mono_iff_injective] at mono
        rw [mono (Classical.choose_spec h)]
      · exact False.elim (h ⟨y, rfl⟩)⟩


instance Type.enoughInjectives : EnoughInjectives (Type u₁) where
  presentation X :=
    Nonempty.intro
      { J := WithBot X
        injective := inferInstance
        f := Option.some
        mono := by
          /-
            C : Type u₁
            inst✝ : CategoryTheory.Category.{v₁, u₁} C
            X : Type u₁
            ⊢ CategoryTheory.Mono Option.some
          -/
          rw [mono_iff_injective]
          /-
            C : Type u₁
            inst✝ : CategoryTheory.Category.{v₁, u₁} C
            X : Type u₁
            ⊢ Function.Injective Option.some
          -/
          exact Option.some_injective X }
          /-
            🎉 no goals
          -/


instance {P Q : C} [HasBinaryProduct P Q] [Injective P] [Injective Q] : Injective (P ⨯ Q) where
  factors g f mono := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      P Q : C
      inst✝² : CategoryTheory.Limits.HasBinaryProduct P Q
      inst✝¹ : CategoryTheory.Injective P
      inst✝ : CategoryTheory.Injective Q
      X✝ Y✝ : C
      g : Quiver.Hom X✝ (CategoryTheory.Limits.prod P Q)
      f : Quiver.Hom X✝ Y✝
      mono : CategoryTheory.Mono f
      ⊢ Exists fun h => Eq (CategoryTheory.CategoryStruct.comp f h) g
    -/
    use Limits.prod.lift (factorThru (g ≫ Limits.prod.fst) f) (factorThru (g ≫ Limits.prod.snd) f)
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      P Q : C
      inst✝² : CategoryTheory.Limits.HasBinaryProduct P Q
      inst✝¹ : CategoryTheory.Injective P
      inst✝ : CategoryTheory.Injective Q
      X✝ Y✝ : C
      g : Quiver.Hom X✝ (CategoryTheory.Limits.prod P Q)
      f : Quiver.Hom X✝ Y✝
      mono : CategoryTheory.Mono f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.prod.lift (C …
    -/
    simp only [prod.comp_lift, comp_factorThru]
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      P Q : C
      inst✝² : CategoryTheory.Limits.HasBinaryProduct P Q
      inst✝¹ : CategoryTheory.Injective P
      inst✝ : CategoryTheory.Injective Q
      X✝ Y✝ : C
      g : Quiver.Hom X✝ (CategoryTheory.Limits.prod P Q)
      f : Quiver.Hom X✝ Y✝
      mono : CategoryTheory.Mono f
      ⊢ Eq (CategoryTheory.Limits.prod.lift (CategoryTheory.CategoryStruct.comp g Ca …
    -/
    ext
      /-
        case h.h₁
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        P Q : C
        inst✝² : CategoryTheory.Limits.HasBinaryProduct P Q
        inst✝¹ : CategoryTheory.Injective P
        inst✝ : CategoryTheory.Injective Q
        X✝ Y✝ : C
        g : Quiver.Hom X✝ (CategoryTheory.Limits.prod P Q)
        f : Quiver.Hom X✝ Y✝
        mono : CategoryTheory.Mono f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift (Cat …
      -/
    · simp only [prod.lift_fst]
      /-
        🎉 no goals
      -/
      /-
        case h.h₂
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        P Q : C
        inst✝² : CategoryTheory.Limits.HasBinaryProduct P Q
        inst✝¹ : CategoryTheory.Injective P
        inst✝ : CategoryTheory.Injective Q
        X✝ Y✝ : C
        g : Quiver.Hom X✝ (CategoryTheory.Limits.prod P Q)
        f : Quiver.Hom X✝ Y✝
        mono : CategoryTheory.Mono f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift (Cat …
      -/
    · simp only [prod.lift_snd]
      /-
        🎉 no goals
      -/


instance {β : Type v} (c : β → C) [HasProduct c] [∀ b, Injective (c b)] : Injective (∏ᶜ c) where
  factors g f mono := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      β : Type v
      c : β → C
      inst✝¹ : CategoryTheory.Limits.HasProduct c
      inst✝ : ∀ (b : β), CategoryTheory.Injective (c b)
      X✝ Y✝ : C
      g : Quiver.Hom X✝ (CategoryTheory.Limits.piObj c)
      f : Quiver.Hom X✝ Y✝
      mono : CategoryTheory.Mono f
      ⊢ Exists fun h => Eq (CategoryTheory.CategoryStruct.comp f h) g
    -/
    refine ⟨Pi.lift fun b => factorThru (g ≫ Pi.π c _) f, ?_⟩
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      β : Type v
      c : β → C
      inst✝¹ : CategoryTheory.Limits.HasProduct c
      inst✝ : ∀ (b : β), CategoryTheory.Injective (c b)
      X✝ Y✝ : C
      g : Quiver.Hom X✝ (CategoryTheory.Limits.piObj c)
      f : Quiver.Hom X✝ Y✝
      mono : CategoryTheory.Mono f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Pi.lift fun  …
    -/
    ext b
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      β : Type v
      c : β → C
      inst✝¹ : CategoryTheory.Limits.HasProduct c
      inst✝ : ∀ (b : β), CategoryTheory.Injective (c b)
      X✝ Y✝ : C
      g : Quiver.Hom X✝ (CategoryTheory.Limits.piObj c)
      f : Quiver.Hom X✝ Y✝
      mono : CategoryTheory.Mono f
      b : β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    simp only [Category.assoc, limit.lift_π, Fan.mk_π_app, comp_factorThru]
    /-
      🎉 no goals
    -/


instance {P Q : C} [HasZeroMorphisms C] [HasBinaryBiproduct P Q] [Injective P] [Injective Q] :
    Injective (P ⊞ Q) where
  factors g f mono := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      P Q : C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasBinaryBiproduct P Q
      inst✝¹ : CategoryTheory.Injective P
      inst✝ : CategoryTheory.Injective Q
      X✝ Y✝ : C
      g : Quiver.Hom X✝ (CategoryTheory.Limits.biprod P Q)
      f : Quiver.Hom X✝ Y✝
      mono : CategoryTheory.Mono f
      ⊢ Exists fun h => Eq (CategoryTheory.CategoryStruct.comp f h) g
    -/
    refine ⟨biprod.lift (factorThru (g ≫ biprod.fst) f) (factorThru (g ≫ biprod.snd) f), ?_⟩
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      P Q : C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasBinaryBiproduct P Q
      inst✝¹ : CategoryTheory.Injective P
      inst✝ : CategoryTheory.Injective Q
      X✝ Y✝ : C
      g : Quiver.Hom X✝ (CategoryTheory.Limits.biprod P Q)
      f : Quiver.Hom X✝ Y✝
      mono : CategoryTheory.Mono f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.biprod.lift  …
    -/
    ext
      /-
        case h₀
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        P Q : C
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasBinaryBiproduct P Q
        inst✝¹ : CategoryTheory.Injective P
        inst✝ : CategoryTheory.Injective Q
        X✝ Y✝ : C
        g : Quiver.Hom X✝ (CategoryTheory.Limits.biprod P Q)
        f : Quiver.Hom X✝ Y✝
        mono : CategoryTheory.Mono f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
      -/
    · simp only [Category.assoc, biprod.lift_fst, comp_factorThru]
      /-
        🎉 no goals
      -/
      /-
        case h₁
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        P Q : C
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasBinaryBiproduct P Q
        inst✝¹ : CategoryTheory.Injective P
        inst✝ : CategoryTheory.Injective Q
        X✝ Y✝ : C
        g : Quiver.Hom X✝ (CategoryTheory.Limits.biprod P Q)
        f : Quiver.Hom X✝ Y✝
        mono : CategoryTheory.Mono f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
      -/
    · simp only [Category.assoc, biprod.lift_snd, comp_factorThru]
      /-
        🎉 no goals
      -/


instance {β : Type v} (c : β → C) [HasZeroMorphisms C] [HasBiproduct c] [∀ b, Injective (c b)] :
    Injective (⨁ c) where
  factors g f mono := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      β : Type v
      c : β → C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasBiproduct c
      inst✝ : ∀ (b : β), CategoryTheory.Injective (c b)
      X✝ Y✝ : C
      g : Quiver.Hom X✝ (CategoryTheory.Limits.biproduct c)
      f : Quiver.Hom X✝ Y✝
      mono : CategoryTheory.Mono f
      ⊢ Exists fun h => Eq (CategoryTheory.CategoryStruct.comp f h) g
    -/
    refine ⟨biproduct.lift fun b => factorThru (g ≫ biproduct.π _ _) f, ?_⟩
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      β : Type v
      c : β → C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasBiproduct c
      inst✝ : ∀ (b : β), CategoryTheory.Injective (c b)
      X✝ Y✝ : C
      g : Quiver.Hom X✝ (CategoryTheory.Limits.biproduct c)
      f : Quiver.Hom X✝ Y✝
      mono : CategoryTheory.Mono f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.biproduct.li …
    -/
    ext
    /-
      case w
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      β : Type v
      c : β → C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasBiproduct c
      inst✝ : ∀ (b : β), CategoryTheory.Injective (c b)
      X✝ Y✝ : C
      g : Quiver.Hom X✝ (CategoryTheory.Limits.biproduct c)
      f : Quiver.Hom X✝ Y✝
      mono : CategoryTheory.Mono f
      j✝ : β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    simp only [Category.assoc, biproduct.lift_π, comp_factorThru]
    /-
      🎉 no goals
    -/


instance {P : Cᵒᵖ} [Projective P] : Injective no_index (unop P) where
  factors g f mono :=
                                                                                    /-
                                                                                      C : Type u₁
                                                                                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                      P : Opposite C
                                                                                      inst✝ : CategoryTheory.Projective P
                                                                                      X✝ Y✝ : C
                                                                                      g : Quiver.Hom X✝ (Opposite.unop P)
                                                                                      f : Quiver.Hom X✝ Y✝
                                                                                      mono : CategoryTheory.Mono f
                                                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Projective.factorTh …
                                                                                    -/
    ⟨(@Projective.factorThru Cᵒᵖ _ P _ _ _ g.op f.op _).unop, Quiver.Hom.op_inj (by simp)⟩
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


instance {J : Cᵒᵖ} [Injective J] : Projective no_index (unop J) where
  factors f e he :=
                                                                         /-
                                                                           C : Type u₁
                                                                           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                           J : Opposite C
                                                                           inst✝ : CategoryTheory.Injective J
                                                                           E✝ X✝ : C
                                                                           f : Quiver.Hom (Opposite.unop J) X✝
                                                                           e : Quiver.Hom E✝ X✝
                                                                           he : CategoryTheory.Epi e
                                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Injective.factorThru  …
                                                                         -/
    ⟨(@factorThru Cᵒᵖ _ J _ _ _ f.op e.op _).unop, Quiver.Hom.op_inj (by simp)⟩
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


instance {J : C} [Injective J] : Projective (op J) where
  factors f e epi :=
                                                                           /-
                                                                             C : Type u₁
                                                                             inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                             J : C
                                                                             inst✝ : CategoryTheory.Injective J
                                                                             E✝ X✝ : Opposite C
                                                                             f : Quiver.Hom { unop := J } X✝
                                                                             e : Quiver.Hom E✝ X✝
                                                                             epi : CategoryTheory.Epi e
                                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Injective.factorThru  …
                                                                           -/
    ⟨(@factorThru C _ J _ _ _ f.unop e.unop _).op, Quiver.Hom.unop_inj (by simp)⟩
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


instance {P : C} [Projective P] : Injective (op P) where
  factors g f mono :=
                                                                                      /-
                                                                                        C : Type u₁
                                                                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                        P : C
                                                                                        inst✝ : CategoryTheory.Projective P
                                                                                        X✝ Y✝ : Opposite C
                                                                                        g : Quiver.Hom X✝ { unop := P }
                                                                                        f : Quiver.Hom X✝ Y✝
                                                                                        mono : CategoryTheory.Mono f
                                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Projective.factorTh …
                                                                                      -/
    ⟨(@Projective.factorThru C _ P _ _ _ g.unop f.unop _).op, Quiver.Hom.unop_inj (by simp)⟩
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


theorem injective_iff_projective_op {J : C} : Injective J ↔ Projective (op J) :=
  ⟨fun _ => inferInstance, fun _ => show Injective (unop (op J)) from inferInstance⟩


theorem projective_iff_injective_op {P : C} : Projective P ↔ Injective (op P) :=
  ⟨fun _ => inferInstance, fun _ => show Projective (unop (op P)) from inferInstance⟩


theorem injective_iff_preservesEpimorphisms_yoneda_obj (J : C) :
    Injective J ↔ (yoneda.obj J).PreservesEpimorphisms := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    J : C
    ⊢ Iff (CategoryTheory.Injective J) (CategoryTheory.yoneda.obj J).PreservesEpim …
  -/
  rw [injective_iff_projective_op, Projective.projective_iff_preservesEpimorphisms_coyoneda_obj]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    J : C
    ⊢ Iff (CategoryTheory.coyoneda.obj { unop := { unop := J } }).PreservesEpimorp …
  -/
  exact Functor.preservesEpimorphisms.iso_iff (Coyoneda.objOpOp _)
  /-
    🎉 no goals
  -/


theorem injective_of_adjoint (adj : L ⊣ R) (J : D) [Injective J] : Injective <| R.obj J :=
  ⟨fun {A} {_} g f im =>
    ⟨adj.homEquiv _ _ (factorThru ((adj.homEquiv A J).symm g) (L.map f)),
      (adj.homEquiv _ _).symm.injective
            /-
              C : Type u₁
              inst✝³ : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝² : CategoryTheory.Category.{v₂, u₂} D
              L : CategoryTheory.Functor C D
              R : CategoryTheory.Functor D C
              inst✝¹ : L.PreservesMonomorphisms
              adj : CategoryTheory.Adjunction L R
              J : D
              inst✝ : CategoryTheory.Injective J
              A x✝ : C
              g : Quiver.Hom A (R.obj J)
              f : Quiver.Hom A x✝
              im : CategoryTheory.Mono f
              ⊢ Eq ((adj.homEquiv A J).symm (CategoryTheory.CategoryStruct.comp f ((adj.homE …
            -/
        (by simp [Adjunction.homEquiv_unit, Adjunction.homEquiv_counit])⟩⟩
            /-
              🎉 no goals
            -/


/-- `Injective.under X` provides an arbitrarily chosen injective object equipped with
a monomorphism `Injective.ι : X ⟶ Injective.under X`.
-/
def under (X : C) : C :=
  (EnoughInjectives.presentation X).some.J


instance injective_under (X : C) : Injective (under X) :=
  (EnoughInjectives.presentation X).some.injective


/-- The monomorphism `Injective.ι : X ⟶ Injective.under X`
from the arbitrarily chosen injective object under `X`.
-/
def ι (X : C) : X ⟶ under X :=
  (EnoughInjectives.presentation X).some.f


instance ι_mono (X : C) : Mono (ι X) :=
  (EnoughInjectives.presentation X).some.mono


/-- When `C` has enough injectives, the object `Injective.syzygies f` is
an arbitrarily chosen injective object under `cokernel f`.
-/
def syzygies : C :=
  under (cokernel f) -- Porting note: no deriving Injective


instance : Injective <| syzygies f := injective_under (cokernel f)


/-- When `C` has enough injective,
`Injective.d f : Y ⟶ syzygies f` is the composition
`cokernel.π f ≫ ι (cokernel f)`.

(When `C` is abelian, we have `exact f (injective.d f)`.)
-/
abbrev d : Y ⟶ syzygies f :=
  cokernel.π f ≫ ι (cokernel f)


instance [EnoughInjectives C] : EnoughProjectives Cᵒᵖ :=
  ⟨fun X => ⟨{ p := _, f := (Injective.ι (unop X)).op}⟩⟩


instance [EnoughProjectives C] : EnoughInjectives Cᵒᵖ :=
  ⟨fun X => ⟨⟨_, inferInstance, (Projective.π (unop X)).op, inferInstance⟩⟩⟩


theorem enoughProjectives_of_enoughInjectives_op [EnoughInjectives Cᵒᵖ] : EnoughProjectives C :=
  ⟨fun X => ⟨{ p := _, f := (Injective.ι (op X)).unop} ⟩⟩


theorem enoughInjectives_of_enoughProjectives_op [EnoughProjectives Cᵒᵖ] : EnoughInjectives C :=
  ⟨fun X => ⟨⟨_, inferInstance, (Projective.π (op X)).unop, inferInstance⟩⟩⟩


theorem map_injective (adj : F ⊣ G) [F.PreservesMonomorphisms] (I : D) (hI : Injective I) :
    Injective (G.obj I) :=
  ⟨fun {X} {Y} f g => by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.PreservesMonomorphisms
      I : D
      hI : CategoryTheory.Injective I
      X Y : C
      f : Quiver.Hom X (G.obj I)
      g : Quiver.Hom X Y
      ⊢ ∀ [inst : CategoryTheory.Mono g], Exists fun h => Eq (CategoryTheory.Categor …
    -/
    intro
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝¹ : F.PreservesMonomorphisms
      I : D
      hI : CategoryTheory.Injective I
      X Y : C
      f : Quiver.Hom X (G.obj I)
      g : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono g
      ⊢ Exists fun h => Eq (CategoryTheory.CategoryStruct.comp g h) f
    -/
    rcases hI.factors (F.map f ≫ adj.counit.app _) (F.map g) with ⟨w,h⟩
    /-
      case intro
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝¹ : F.PreservesMonomorphisms
      I : D
      hI : CategoryTheory.Injective I
      X Y : C
      f : Quiver.Hom X (G.obj I)
      g : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono g
      w : Quiver.Hom (F.obj Y) I
      h : Eq (CategoryTheory.CategoryStruct.comp (F.map g) w) (CategoryTheory.Catego …
      ⊢ Exists fun h => Eq (CategoryTheory.CategoryStruct.comp g h) f
    -/
    use adj.unit.app Y ≫ G.map w
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝¹ : F.PreservesMonomorphisms
      I : D
      hI : CategoryTheory.Injective I
      X Y : C
      f : Quiver.Hom X (G.obj I)
      g : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono g
      w : Quiver.Hom (F.obj Y) I
      h : Eq (CategoryTheory.CategoryStruct.comp (F.map g) w) (CategoryTheory.Catego …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.CategoryStruct.comp …
    -/
    rw [← unit_naturality_assoc, ← G.map_comp, h]
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝¹ : F.PreservesMonomorphisms
      I : D
      hI : CategoryTheory.Injective I
      X Y : C
      f : Quiver.Hom X (G.obj I)
      g : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono g
      w : Quiver.Hom (F.obj Y) I
      h : Eq (CategoryTheory.CategoryStruct.comp (F.map g) w) (CategoryTheory.Catego …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app X) (G.map (CategoryTheo …
    -/
    simp⟩
    /-
      🎉 no goals
    -/


theorem injective_of_map_injective (adj : F ⊣ G) [G.Full] [G.Faithful] (I : D)
    (hI : Injective (G.obj I)) : Injective I :=
  ⟨fun {X} {Y} f g => by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝¹ : G.Full
      inst✝ : G.Faithful
      I : D
      hI : CategoryTheory.Injective (G.obj I)
      X Y : D
      f : Quiver.Hom X I
      g : Quiver.Hom X Y
      ⊢ ∀ [inst : CategoryTheory.Mono g], Exists fun h => Eq (CategoryTheory.Categor …
    -/
    intro
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝² : G.Full
      inst✝¹ : G.Faithful
      I : D
      hI : CategoryTheory.Injective (G.obj I)
      X Y : D
      f : Quiver.Hom X I
      g : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono g
      ⊢ Exists fun h => Eq (CategoryTheory.CategoryStruct.comp g h) f
    -/
    haveI : PreservesLimitsOfSize.{0, 0} G := adj.rightAdjoint_preservesLimits
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝² : G.Full
      inst✝¹ : G.Faithful
      I : D
      hI : CategoryTheory.Injective (G.obj I)
      X Y : D
      f : Quiver.Hom X I
      g : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono g
      this : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_2, v₁, u_1, u₁} G
      ⊢ Exists fun h => Eq (CategoryTheory.CategoryStruct.comp g h) f
    -/
    rcases hI.factors (G.map f) (G.map g) with ⟨w,h⟩
    /-
      case intro
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝² : G.Full
      inst✝¹ : G.Faithful
      I : D
      hI : CategoryTheory.Injective (G.obj I)
      X Y : D
      f : Quiver.Hom X I
      g : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono g
      this : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_2, v₁, u_1, u₁} G
      w : Quiver.Hom (G.obj Y) (G.obj I)
      h : Eq (CategoryTheory.CategoryStruct.comp (G.map g) w) (G.map f)
      ⊢ Exists fun h => Eq (CategoryTheory.CategoryStruct.comp g h) f
    -/
    use inv (adj.counit.app _) ≫ F.map w ≫ adj.counit.app _
    /-
      case h
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝² : G.Full
      inst✝¹ : G.Faithful
      I : D
      hI : CategoryTheory.Injective (G.obj I)
      X Y : D
      f : Quiver.Hom X I
      g : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono g
      this : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_2, v₁, u_1, u₁} G
      w : Quiver.Hom (G.obj Y) (G.obj I)
      h : Eq (CategoryTheory.CategoryStruct.comp (G.map g) w) (G.map f)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.CategoryStruct.comp …
    -/
    exact G.map_injective (by simpa)⟩
    /-
      🎉 no goals
    -/


/-- Given an adjunction `F ⊣ G` such that `F` preserves monos, `G` maps an injective presentation
of `X` to an injective presentation of `G(X)`. -/
def mapInjectivePresentation (adj : F ⊣ G) [F.PreservesMonomorphisms] (X : D)
    (I : InjectivePresentation X) : InjectivePresentation (G.obj X) where
  J := G.obj I.J
  injective := adj.map_injective _ I.injective
  f := G.map I.f
  mono := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.39754, u_1} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.PreservesMonomorphisms
      X : D
      I : CategoryTheory.InjectivePresentation X
      ⊢ CategoryTheory.Mono (G.map I.f)
    -/
    haveI : PreservesLimitsOfSize.{0, 0} G := adj.rightAdjoint_preservesLimits; infer_instance
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


/-- Given an adjunction `F ⊣ G` such that `F` preserves monomorphisms and is faithful,
  then any injective presentation of `F(X)` can be pulled back to an injective presentation of `X`.
  This is similar to `mapInjectivePresentation`. -/
def injectivePresentationOfMap (adj : F ⊣ G)
    [F.PreservesMonomorphisms] [F.ReflectsMonomorphisms] (X : C)
    (I : InjectivePresentation <| F.obj X) :
    InjectivePresentation X where
  J := G.obj I.J
  injective := Injective.injective_of_adjoint adj _
  f := adj.homEquiv _ _ I.f


/--
[Lemma 3.8](https://ncatlab.org/nlab/show/injective+object#preservation_of_injective_objects)
-/
lemma EnoughInjectives.of_adjunction {C : Type u₁} {D : Type u₂}
    [Category.{v₁} C] [Category.{v₂} D]
    {L : C ⥤ D} {R : D ⥤ C} (adj : L ⊣ R) [L.PreservesMonomorphisms] [L.ReflectsMonomorphisms]
    [EnoughInjectives D] : EnoughInjectives C where
  presentation _ :=
    ⟨adj.injectivePresentationOfMap _ (EnoughInjectives.presentation _).some⟩


/-- An equivalence of categories transfers enough injectives. -/
lemma EnoughInjectives.of_equivalence {C : Type u₁} {D : Type u₂}
    [Category.{v₁} C] [Category.{v₂} D]
    (e : C ⥤ D) [e.IsEquivalence] [EnoughInjectives D] : EnoughInjectives C :=
  EnoughInjectives.of_adjunction (adj := e.asEquivalence.toAdjunction)


theorem map_injective_iff (P : C) : Injective (F.functor.obj P) ↔ Injective P :=
  ⟨F.symm.toAdjunction.injective_of_map_injective P, F.symm.toAdjunction.map_injective P⟩


/-- Given an equivalence of categories `F`, an injective presentation of `F(X)` induces an
injective presentation of `X.` -/
def injectivePresentationOfMapInjectivePresentation (X : C)
    (I : InjectivePresentation (F.functor.obj X)) : InjectivePresentation X :=
  F.toAdjunction.injectivePresentationOfMap _ I


theorem enoughInjectives_iff (F : C ≌ D) : EnoughInjectives C ↔ EnoughInjectives D :=
  ⟨fun h => h.of_adjunction F.symm.toAdjunction, fun h => h.of_adjunction F.toAdjunction⟩


