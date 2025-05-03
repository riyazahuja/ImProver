/-- Given a family of morphisms `f i : A i ⟶ B i` and a morphism `πX : X ⟶ S`,
this type parametrizes the commutative squares with a morphism `f i` on the left
and `πX` in the right. -/
structure FunctorObjIndex where
  /-- an element in the index type -/
  i : I
  /-- the top morphism in the square -/
  t : A i ⟶ X
  /-- the bottom morphism in the square -/
  b : B i ⟶ S
  w : t ≫ πX = f i ≫ b


attribute [reassoc (attr := simp)] FunctorObjIndex.w


/-- The family of objects `A x.i` parametrized by `x : FunctorObjIndex f πX`. -/
abbrev functorObjSrcFamily (x : FunctorObjIndex f πX) : C := A x.i


/-- The family of objects `B x.i` parametrized by `x : FunctorObjIndex f πX`. -/
abbrev functorObjTgtFamily (x : FunctorObjIndex f πX) : C := B x.i


/-- The family of the morphisms `f x.i : A x.i ⟶ B x.i`
parametrized by `x : FunctorObjIndex f πX`. -/
abbrev functorObjLeftFamily (x : FunctorObjIndex f πX) :
    functorObjSrcFamily f πX x ⟶ functorObjTgtFamily f πX x := f x.i


/-- The top morphism in the pushout square in the definition of `pushoutObj f πX`. -/
noncomputable abbrev functorObjTop : ∐ functorObjSrcFamily f πX ⟶ X :=
  Limits.Sigma.desc (fun x => x.t)


/-- The left morphism in the pushout square in the definition of `pushoutObj f πX`. -/
noncomputable abbrev functorObjLeft :
    ∐ functorObjSrcFamily f πX ⟶ ∐ functorObjTgtFamily f πX :=
  Limits.Sigma.map (functorObjLeftFamily f πX)


/-- The functor `SmallObject.functor f S : Over S ⥤ Over S` that is part of
the small object argument for a family of morphisms `f`, on an object given
as a morphism `πX : X ⟶ S`. -/
noncomputable abbrev functorObj : C :=
  pushout (functorObjTop f πX) (functorObjLeft f πX)


/-- The canonical morphism `X ⟶ functorObj f πX`. -/
noncomputable abbrev ιFunctorObj : X ⟶ functorObj f πX := pushout.inl _ _


/-- The canonical morphism `∐ (functorObjTgtFamily f πX) ⟶ functorObj f πX`. -/
noncomputable abbrev ρFunctorObj : ∐ functorObjTgtFamily f πX ⟶ functorObj f πX := pushout.inr _ _


@[reassoc]
lemma functorObj_comm :
    functorObjTop f πX ≫ ιFunctorObj f πX = functorObjLeft f πX ≫ ρFunctorObj f πX :=
  pushout.condition


@[reassoc]
lemma FunctorObjIndex.comm (x : FunctorObjIndex f πX) :
    f x.i ≫ Sigma.ι (functorObjTgtFamily f πX) x ≫ ρFunctorObj f πX = x.t ≫ ιFunctorObj f πX := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type w
    A B : I → C
    f : (i : I) → Quiver.Hom (A i) (B i)
    S X : C
    πX : Quiver.Hom X S
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
    inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorOb …
    x : CategoryTheory.SmallObject.FunctorObjIndex f πX
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f x.i) (CategoryTheory.CategoryStruc …
  -/
  simpa using (Sigma.ι (functorObjSrcFamily f πX) x ≫= functorObj_comm f πX).symm
  /-
    🎉 no goals
  -/


/-- The canonical projection on the base object. -/
noncomputable abbrev π'FunctorObj : ∐ functorObjTgtFamily f πX ⟶ S := Sigma.desc (fun x => x.b)


/-- The canonical projection on the base object. -/
noncomputable def πFunctorObj : functorObj f πX ⟶ S :=
                                          /-
                                            C : Type u
                                            inst✝³ : CategoryTheory.Category.{v, u} C
                                            I : Type w
                                            A B : I → C
                                            f : (i : I) → Quiver.Hom (A i) (B i)
                                            S X Y : C
                                            πX : Quiver.Hom X S
                                            πY : Quiver.Hom Y S
                                            φ : Quiver.Hom X Y
                                            inst✝² : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
                                            inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
                                            inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorOb …
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.SmallObject.functorOb …
                                          -/
  pushout.desc πX (π'FunctorObj f πX) (by ext; simp [π'FunctorObj])
                                               /-
                                                 🎉 no goals
                                               -/


@[reassoc (attr := simp)]
lemma ρFunctorObj_π : ρFunctorObj f πX ≫ πFunctorObj f πX = π'FunctorObj f πX := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type w
    A B : I → C
    f : (i : I) → Quiver.Hom (A i) (B i)
    S X : C
    πX : Quiver.Hom X S
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
    inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorOb …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.SmallObject.ρFunctorO …
  -/
  simp [πFunctorObj]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ιFunctorObj_πFunctorObj : ιFunctorObj f πX ≫ πFunctorObj f πX = πX := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type w
    A B : I → C
    f : (i : I) → Quiver.Hom (A i) (B i)
    S X : C
    πX : Quiver.Hom X S
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
    inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorOb …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.SmallObject.ιFunctorO …
  -/
  simp [ιFunctorObj, πFunctorObj]
  /-
    🎉 no goals
  -/


/-- The canonical morphism `∐ (functorObjSrcFamily f πX) ⟶ ∐ (functorObjSrcFamily f πY)`
induced by a morphism in `φ : X ⟶ Y` such that `φ ≫ πX = πY`. -/
noncomputable def functorMapSrc (hφ : φ ≫ πY = πX) :
    ∐ (functorObjSrcFamily f πX) ⟶ ∐ functorObjSrcFamily f πY :=
                                                                /-
                                                                  C : Type u
                                                                  inst✝³ : CategoryTheory.Category.{v, u} C
                                                                  I : Type w
                                                                  A B : I → C
                                                                  f : (i : I) → Quiver.Hom (A i) (B i)
                                                                  S X Y : C
                                                                  πX : Quiver.Hom X S
                                                                  πY : Quiver.Hom Y S
                                                                  φ : Quiver.Hom X Y
                                                                  inst✝² : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
                                                                  inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
                                                                  inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorOb …
                                                                  hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
                                                                  x : CategoryTheory.SmallObject.FunctorObjIndex f πX
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp x …
                                                                -/
  Sigma.map' (fun x => FunctorObjIndex.mk x.i (x.t ≫ φ) x.b (by simp [hφ])) (fun _ => 𝟙 _)
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[reassoc]
lemma ι_functorMapSrc (i : I) (t : A i ⟶ X) (b : B i ⟶ S) (w : t ≫ πX = f i ≫ b)
    (t' : A i ⟶ Y) (fac : t ≫ φ = t') :
    Sigma.ι _ (FunctorObjIndex.mk i t b w) ≫ functorMapSrc f πX πY φ hφ =
      Sigma.ι (functorObjSrcFamily f πY)
                                       /-
                                         C : Type u
                                         inst✝² : CategoryTheory.Category.{v, u} C
                                         I : Type w
                                         A B : I → C
                                         f : (i : I) → Quiver.Hom (A i) (B i)
                                         S X Y : C
                                         πX : Quiver.Hom X S
                                         πY : Quiver.Hom Y S
                                         φ : Quiver.Hom X Y
                                         inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
                                         inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Cat …
                                         hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
                                         i : I
                                         t : Quiver.Hom (A i) X
                                         b : Quiver.Hom (B i) S
                                         w : Eq (CategoryTheory.CategoryStruct.comp t πX) (CategoryTheory.CategoryStruc …
                                         t' : Quiver.Hom (A i) Y
                                         fac : Eq (CategoryTheory.CategoryStruct.comp t φ) t'
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp t' πY) (CategoryTheory.CategoryStruct …
                                       -/
        (FunctorObjIndex.mk i t' b (by rw [← w, ← fac, assoc, hφ])) := by
                                       /-
                                         🎉 no goals
                                       -/
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type w
    A B : I → C
    f : (i : I) → Quiver.Hom (A i) (B i)
    S X Y : C
    πX : Quiver.Hom X S
    πY : Quiver.Hom Y S
    φ : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Cat …
    hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
    i : I
    t : Quiver.Hom (A i) X
    b : Quiver.Hom (B i) S
    w : Eq (CategoryTheory.CategoryStruct.comp t πX) (CategoryTheory.CategoryStruc …
    t' : Quiver.Hom (A i) Y
    fac : Eq (CategoryTheory.CategoryStruct.comp t φ) t'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (Categ …
  -/
  subst fac
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type w
    A B : I → C
    f : (i : I) → Quiver.Hom (A i) (B i)
    S X Y : C
    πX : Quiver.Hom X S
    πY : Quiver.Hom Y S
    φ : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Cat …
    hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
    i : I
    t : Quiver.Hom (A i) X
    b : Quiver.Hom (B i) S
    w : Eq (CategoryTheory.CategoryStruct.comp t πX) (CategoryTheory.CategoryStruc …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (Categ …
  -/
  simp [functorMapSrc]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma functorMapSrc_functorObjTop :
    functorMapSrc f πX πY φ hφ ≫ functorObjTop f πY = functorObjTop f πX ≫ φ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type w
    A B : I → C
    f : (i : I) → Quiver.Hom (A i) (B i)
    S X Y : C
    πX : Quiver.Hom X S
    πY : Quiver.Hom Y S
    φ : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Cat …
    hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.SmallObject.functorMa …
  -/
  ext ⟨i, t, b, w⟩
  /-
    case h.mk
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type w
    A B : I → C
    f : (i : I) → Quiver.Hom (A i) (B i)
    S X Y : C
    πX : Quiver.Hom X S
    πY : Quiver.Hom Y S
    φ : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Cat …
    hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
    i : I
    t : Quiver.Hom (A i) X
    b : Quiver.Hom (B i) S
    w : Eq (CategoryTheory.CategoryStruct.comp t πX) (CategoryTheory.CategoryStruc …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (Categ …
  -/
  simp [ι_functorMapSrc_assoc f πX πY φ hφ i t b w _ rfl]
  /-
    🎉 no goals
  -/


/-- The canonical morphism `∐ functorObjTgtFamily f πX ⟶ ∐ functorObjTgtFamily f πY`
induced by a morphism in `φ : X ⟶ Y` such that `φ ≫ πX = πY`. -/
noncomputable def functorMapTgt (hφ : φ ≫ πY = πX) :
    ∐ functorObjTgtFamily f πX ⟶ ∐ functorObjTgtFamily f πY :=
                                                                /-
                                                                  C : Type u
                                                                  inst✝² : CategoryTheory.Category.{v, u} C
                                                                  I : Type w
                                                                  A B : I → C
                                                                  f : (i : I) → Quiver.Hom (A i) (B i)
                                                                  S X Y : C
                                                                  πX : Quiver.Hom X S
                                                                  πY : Quiver.Hom Y S
                                                                  φ : Quiver.Hom X Y
                                                                  inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
                                                                  inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Cat …
                                                                  hφ✝ hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
                                                                  x : CategoryTheory.SmallObject.FunctorObjIndex f πX
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp x …
                                                                -/
  Sigma.map' (fun x => FunctorObjIndex.mk x.i (x.t ≫ φ) x.b (by simp [hφ])) (fun _ => 𝟙 _)
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[reassoc]
lemma ι_functorMapTgt (i : I) (t : A i ⟶ X) (b : B i ⟶ S) (w : t ≫ πX = f i ≫ b)
    (t' : A i ⟶ Y) (fac : t ≫ φ = t') :
    Sigma.ι _ (FunctorObjIndex.mk i t b w) ≫ functorMapTgt f πX πY φ hφ =
      Sigma.ι (functorObjTgtFamily f πY)
                                       /-
                                         C : Type u
                                         inst✝² : CategoryTheory.Category.{v, u} C
                                         I : Type w
                                         A B : I → C
                                         f : (i : I) → Quiver.Hom (A i) (B i)
                                         S X Y : C
                                         πX : Quiver.Hom X S
                                         πY : Quiver.Hom Y S
                                         φ : Quiver.Hom X Y
                                         inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
                                         inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Cat …
                                         hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
                                         i : I
                                         t : Quiver.Hom (A i) X
                                         b : Quiver.Hom (B i) S
                                         w : Eq (CategoryTheory.CategoryStruct.comp t πX) (CategoryTheory.CategoryStruc …
                                         t' : Quiver.Hom (A i) Y
                                         fac : Eq (CategoryTheory.CategoryStruct.comp t φ) t'
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp t' πY) (CategoryTheory.CategoryStruct …
                                       -/
        (FunctorObjIndex.mk i t' b (by rw [← w, ← fac, assoc, hφ])) := by
                                       /-
                                         🎉 no goals
                                       -/
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type w
    A B : I → C
    f : (i : I) → Quiver.Hom (A i) (B i)
    S X Y : C
    πX : Quiver.Hom X S
    πY : Quiver.Hom Y S
    φ : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Cat …
    hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
    i : I
    t : Quiver.Hom (A i) X
    b : Quiver.Hom (B i) S
    w : Eq (CategoryTheory.CategoryStruct.comp t πX) (CategoryTheory.CategoryStruc …
    t' : Quiver.Hom (A i) Y
    fac : Eq (CategoryTheory.CategoryStruct.comp t φ) t'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (Categ …
  -/
  subst fac
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type w
    A B : I → C
    f : (i : I) → Quiver.Hom (A i) (B i)
    S X Y : C
    πX : Quiver.Hom X S
    πY : Quiver.Hom Y S
    φ : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Cat …
    hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
    i : I
    t : Quiver.Hom (A i) X
    b : Quiver.Hom (B i) S
    w : Eq (CategoryTheory.CategoryStruct.comp t πX) (CategoryTheory.CategoryStruc …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (Categ …
  -/
  simp [functorMapTgt]
  /-
    🎉 no goals
  -/


lemma functorMap_comm :
    functorObjLeft f πX ≫ functorMapTgt f πX πY φ hφ =
      functorMapSrc f πX πY φ hφ ≫ functorObjLeft f πY := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type w
    A B : I → C
    f : (i : I) → Quiver.Hom (A i) (B i)
    S X Y : C
    πX : Quiver.Hom X S
    πY : Quiver.Hom Y S
    φ : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Cat …
    hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.SmallObject.functorOb …
  -/
  ext ⟨i, t, b, w⟩
  simp only [ι_colimMap_assoc, Discrete.natTrans_app, ι_colimMap,
    ι_functorMapTgt f πX πY φ hφ i t b w _ rfl,
    ι_functorMapSrc_assoc f πX πY φ hφ i t b w _ rfl]


/-- The functor `SmallObject.functor f S : Over S ⥤ Over S` that is part of
the small object argument for a family of morphisms `f`, on morphisms. -/
noncomputable def functorMap : functorObj f πX ⟶ functorObj f πY :=
                                                                                      /-
                                                                                        C : Type u
                                                                                        inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                                                        I : Type w
                                                                                        A B : I → C
                                                                                        f : (i : I) → Quiver.Hom (A i) (B i)
                                                                                        S X Y : C
                                                                                        πX : Quiver.Hom X S
                                                                                        πY : Quiver.Hom Y S
                                                                                        φ : Quiver.Hom X Y
                                                                                        inst✝³ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
                                                                                        inst✝² : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
                                                                                        hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
                                                                                        inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorO …
                                                                                        inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorOb …
                                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.SmallObject.functorOb …
                                                                                      -/
  pushout.map _ _ _ _ φ (functorMapTgt f πX πY φ hφ) (functorMapSrc f πX πY φ hφ) (by simp)
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
    (functorMap_comm f πX πY φ hφ)


@[reassoc (attr := simp)]
lemma functorMap_π : functorMap f πX πY φ hφ ≫ πFunctorObj f πY = πFunctorObj f πX := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    I : Type w
    A B : I → C
    f : (i : I) → Quiver.Hom (A i) (B i)
    S X Y : C
    πX : Quiver.Hom X S
    πY : Quiver.Hom Y S
    φ : Quiver.Hom X Y
    inst✝³ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
    hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
    inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorO …
    inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorOb …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.SmallObject.functorMa …
  -/
  ext ⟨i, t, b, w⟩
    /-
      case h₀
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      I : Type w
      A B : I → C
      f : (i : I) → Quiver.Hom (A i) (B i)
      S X Y : C
      πX : Quiver.Hom X S
      πY : Quiver.Hom Y S
      φ : Quiver.Hom X Y
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
      hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
      inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorO …
      inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorOb …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl (C …
    -/
  · simp [functorMap, hφ]
    /-
      🎉 no goals
    -/
    /-
      case h₁.h.mk
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      I : Type w
      A B : I → C
      f : (i : I) → Quiver.Hom (A i) (B i)
      S X Y : C
      πX : Quiver.Hom X S
      πY : Quiver.Hom Y S
      φ : Quiver.Hom X Y
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
      hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
      inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorO …
      inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorOb …
      i : I
      t : Quiver.Hom (A i) X
      b : Quiver.Hom (B i) S
      w : Eq (CategoryTheory.CategoryStruct.comp t πX) (CategoryTheory.CategoryStruc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (Categ …
    -/
  · simp [functorMap, ι_functorMapTgt_assoc f πX πY φ hφ i t b w _ rfl]
    /-
      🎉 no goals
    -/


variable (X) in
@[simp]
                                                   /-
                                                     C : Type u
                                                     inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                     I : Type w
                                                     A B : I → C
                                                     f : (i : I) → Quiver.Hom (A i) (B i)
                                                     S X Y : C
                                                     πX : Quiver.Hom X S
                                                     πY : Quiver.Hom Y S
                                                     φ : Quiver.Hom X Y
                                                     inst✝³ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
                                                     inst✝² : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
                                                     hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
                                                     inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorO …
                                                     inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorOb …
                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
                                                   -/
lemma functorMap_id : functorMap f πX πX (𝟙 X) (by simp) = 𝟙 _ := by
                                                   /-
                                                     🎉 no goals
                                                   -/
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    I : Type w
    A B : I → C
    f : (i : I) → Quiver.Hom (A i) (B i)
    S X : C
    πX : Quiver.Hom X S
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
    inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorOb …
    ⊢ Eq (CategoryTheory.SmallObject.functorMap f πX πX (CategoryTheory.CategorySt …
  -/
  ext ⟨i, t, b, w⟩
    /-
      case h₀
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      I : Type w
      A B : I → C
      f : (i : I) → Quiver.Hom (A i) (B i)
      S X : C
      πX : Quiver.Hom X S
      inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
      inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorOb …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl (C …
    -/
  · simp [functorMap]
    /-
      🎉 no goals
    -/
    /-
      case h₁.h.mk
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      I : Type w
      A B : I → C
      f : (i : I) → Quiver.Hom (A i) (B i)
      S X : C
      πX : Quiver.Hom X S
      inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
      inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorOb …
      i : I
      t : Quiver.Hom (A i) X
      b : Quiver.Hom (B i) S
      w : Eq (CategoryTheory.CategoryStruct.comp t πX) (CategoryTheory.CategoryStruc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (Categ …
    -/
  · simp [functorMap, ι_functorMapTgt_assoc f πX πX (𝟙 X) (by simp) i t b w t (by simp)]
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma ιFunctorObj_naturality :
    ιFunctorObj f πX ≫ functorMap f πX πY φ hφ = φ ≫ ιFunctorObj f πY := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    I : Type w
    A B : I → C
    f : (i : I) → Quiver.Hom (A i) (B i)
    S X Y : C
    πX : Quiver.Hom X S
    πY : Quiver.Hom Y S
    φ : Quiver.Hom X Y
    inst✝³ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
    hφ : Eq (CategoryTheory.CategoryStruct.comp φ πY) πX
    inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorO …
    inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorOb …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.SmallObject.ιFunctorO …
  -/
  simp [ιFunctorObj, functorMap]
  /-
    🎉 no goals
  -/


lemma ιFunctorObj_extension {i : I} (t : A i ⟶ X) (b : B i ⟶ S)
    (sq : CommSq t (f i) πX b) :
    ∃ (l : B i ⟶ functorObj f πX), f i ≫ l = t ≫ ιFunctorObj f πX ∧
      l ≫ πFunctorObj f πX = b :=
  ⟨Sigma.ι (functorObjTgtFamily f πX) (FunctorObjIndex.mk i t b sq.w) ≫
                                                            /-
                                                              C : Type u
                                                              inst✝² : CategoryTheory.Category.{v, u} C
                                                              I : Type w
                                                              A B : I → C
                                                              f : (i : I) → Quiver.Hom (A i) (B i)
                                                              S X : C
                                                              πX : Quiver.Hom X S
                                                              inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Ca …
                                                              inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.SmallObject.functorOb …
                                                              i : I
                                                              t : Quiver.Hom (A i) X
                                                              b : Quiver.Hom (B i) S
                                                              sq : CategoryTheory.CommSq t (f i) πX b
                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                                            -/
    ρFunctorObj f πX, (FunctorObjIndex.mk i t b _).comm, by simp⟩
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- The functor `Over S ⥤ Over S` that is constructed in order to apply the small
object argument to a family of morphisms `f i : A i ⟶ B i`, see the introduction
of the file `Mathlib.CategoryTheory.SmallObject.Construction` -/
@[simps! obj map]
noncomputable def functor : Over S ⥤ Over S where
  obj π := Over.mk (πFunctorObj f π.hom)
                   /-
                     C : Type u
                     inst✝² : CategoryTheory.Category.{v, u} C
                     I : Type w
                     A B : I → C
                     f : (i : I) → Quiver.Hom (A i) (B i)
                     S : C
                     inst✝¹ : CategoryTheory.Limits.HasPushouts C
                     inst✝ : ∀ {X : C} (πX : Quiver.Hom X S), CategoryTheory.Limits.HasColimitsOfSh …
                     π₁ π₂ : CategoryTheory.Over S
                     φ : Quiver.Hom π₁ π₂
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.SmallObject.functorMa …
                   -/
  map {π₁ π₂} φ := Over.homMk (functorMap f π₁.hom π₂.hom φ.left (Over.w φ))
                   /-
                     🎉 no goals
                   -/
                 /-
                   C : Type u
                   inst✝² : CategoryTheory.Category.{v, u} C
                   I : Type w
                   A B : I → C
                   f : (i : I) → Quiver.Hom (A i) (B i)
                   S : C
                   inst✝¹ : CategoryTheory.Limits.HasPushouts C
                   inst✝ : ∀ {X : C} (πX : Quiver.Hom X S), CategoryTheory.Limits.HasColimitsOfSh …
                   x✝ : CategoryTheory.Over S
                   ⊢ Eq ({ obj := fun π => CategoryTheory.Over.mk (CategoryTheory.SmallObject.πFu …
                 -/
  map_id _ := by ext; dsimp; simp
                             /-
                               🎉 no goals
                             -/
  map_comp {π₁ π₂ π₃} φ φ' := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      I : Type w
      A B : I → C
      f : (i : I) → Quiver.Hom (A i) (B i)
      S : C
      inst✝¹ : CategoryTheory.Limits.HasPushouts C
      inst✝ : ∀ {X : C} (πX : Quiver.Hom X S), CategoryTheory.Limits.HasColimitsOfSh …
      π₁ π₂ π₃ : CategoryTheory.Over S
      φ : Quiver.Hom π₁ π₂
      φ' : Quiver.Hom π₂ π₃
      ⊢ Eq ({ obj := fun π => CategoryTheory.Over.mk (CategoryTheory.SmallObject.πFu …
    -/
    ext1
    /-
      case h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      I : Type w
      A B : I → C
      f : (i : I) → Quiver.Hom (A i) (B i)
      S : C
      inst✝¹ : CategoryTheory.Limits.HasPushouts C
      inst✝ : ∀ {X : C} (πX : Quiver.Hom X S), CategoryTheory.Limits.HasColimitsOfSh …
      π₁ π₂ π₃ : CategoryTheory.Over S
      φ : Quiver.Hom π₁ π₂
      φ' : Quiver.Hom π₂ π₃
      ⊢ Eq ({ obj := fun π => CategoryTheory.Over.mk (CategoryTheory.SmallObject.πFu …
    -/
    dsimp
    /-
      case h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      I : Type w
      A B : I → C
      f : (i : I) → Quiver.Hom (A i) (B i)
      S : C
      inst✝¹ : CategoryTheory.Limits.HasPushouts C
      inst✝ : ∀ {X : C} (πX : Quiver.Hom X S), CategoryTheory.Limits.HasColimitsOfSh …
      π₁ π₂ π₃ : CategoryTheory.Over S
      φ : Quiver.Hom π₁ π₂
      φ' : Quiver.Hom π₂ π₃
      ⊢ Eq (CategoryTheory.SmallObject.functorMap f π₁.hom π₃.hom (CategoryTheory.Ca …
    -/
    ext ⟨i, t, b, w⟩
      /-
        case h.h₀
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        I : Type w
        A B : I → C
        f : (i : I) → Quiver.Hom (A i) (B i)
        S : C
        inst✝¹ : CategoryTheory.Limits.HasPushouts C
        inst✝ : ∀ {X : C} (πX : Quiver.Hom X S), CategoryTheory.Limits.HasColimitsOfSh …
        π₁ π₂ π₃ : CategoryTheory.Over S
        φ : Quiver.Hom π₁ π₂
        φ' : Quiver.Hom π₂ π₃
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl (C …
      -/
    · simp
      /-
        🎉 no goals
      -/
    · simp [functorMap, ι_functorMapTgt_assoc f π₁.hom π₂.hom φ.left (Over.w φ) i t b w _ rfl,
        ι_functorMapTgt_assoc f π₁.hom π₃.hom (φ.left ≫ φ'.left) (Over.w (φ ≫ φ')) i t b w _ rfl,
        ι_functorMapTgt_assoc f π₂.hom π₃.hom (φ'.left) (Over.w φ') i (t ≫ φ.left) b
          (by simp [w]) (t ≫ φ.left ≫ φ'.left) (by simp)]


/-- The canonical natural transformation `𝟭 (Over S) ⟶ functor f S`. -/
@[simps! app]
noncomputable def ε : 𝟭 (Over S) ⟶ functor f S where
           /-
             C : Type u
             inst✝² : CategoryTheory.Category.{v, u} C
             I : Type w
             A B : I → C
             f : (i : I) → Quiver.Hom (A i) (B i)
             S : C
             inst✝¹ : CategoryTheory.Limits.HasPushouts C
             inst✝ : ∀ {X : C} (πX : Quiver.Hom X S), CategoryTheory.Limits.HasColimitsOfSh …
             w : CategoryTheory.Over S
             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.SmallObject.ιFunctorO …
           -/
  app w := Over.homMk (ιFunctorObj f w.hom)
           /-
             🎉 no goals
           -/


