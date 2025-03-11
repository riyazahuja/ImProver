/-- Given `F : C ⥤ D ⥤ D`, `X : C`, `e : F.obj X ≅ 𝟭 D` and `Y : GradedObject J D`,
this is the isomorphism `((mapBifunctor F I J).obj ((single₀ I).obj X)).obj Y a ≅ Y a.2`
when `a : I × J` is such that `a.1 = 0`. -/
@[simps!]
noncomputable def mapBifunctorObjSingle₀ObjIso (a : I × J) (ha : a.1 = 0) :
    ((mapBifunctor F I J).obj ((single₀ I).obj X)).obj Y a ≅ Y a.2 :=
  (F.mapIso (singleObjApplyIsoOfEq _ X _ ha)).app _ ≪≫ e.app (Y a.2)


/-- Given `F : C ⥤ D ⥤ D`, `X : C` and `Y : GradedObject J D`,
`((mapBifunctor F I J).obj ((single₀ I).obj X)).obj Y a` is an initial object
when `a : I × J` is such that `a.1 ≠ 0`. -/
noncomputable def mapBifunctorObjSingle₀ObjIsInitial (a : I × J) (ha : a.1 ≠ 0) :
    IsInitial (((mapBifunctor F I J).obj ((single₀ I).obj X)).obj Y a) :=
  IsInitial.isInitialObj (F.flip.obj (Y a.2)) _ (isInitialSingleObjApply _ _ _ ha)


/-- Given `F : C ⥤ D ⥤ D`, `X : C`, `e : F.obj X ≅ 𝟭 D`, `Y : GradedObject J D` and
`p : I × J → J` such that `p ⟨0, j⟩ = j` for all `j`,
this is the (colimit) cofan which shall be used to construct the isomorphism
`mapBifunctorMapObj F p ((single₀ I).obj X) Y ≅ Y`, see `mapBifunctorLeftUnitor`. -/
noncomputable def mapBifunctorLeftUnitorCofan (hp : ∀ (j : J), p ⟨0, j⟩ = j) (Y) (j : J) :
    (((mapBifunctor F I J).obj ((single₀ I).obj X)).obj Y).CofanMapObjFun p j :=
  CofanMapObjFun.mk _ _ _ (Y j) (fun a ha =>
    if ha : a.1 = 0 then
                                                                    /-
                                                                      C : Type u_1
                                                                      D : Type u_2
                                                                      I : Type u_3
                                                                      J : Type u_4
                                                                      inst✝⁵ : CategoryTheory.Category.{?u.7531, u_1} C
                                                                      inst✝⁴ : CategoryTheory.Category.{?u.7535, u_2} D
                                                                      inst✝³ : Zero I
                                                                      inst✝² : DecidableEq I
                                                                      inst✝¹ : CategoryTheory.Limits.HasInitial C
                                                                      F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
                                                                      X : C
                                                                      e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
                                                                      inst✝ : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
                                                                      p : Prod I J → J
                                                                      hp✝ : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
                                                                      Y✝ Y' : CategoryTheory.GradedObject J D
                                                                      φ : Quiver.Hom Y✝ Y'
                                                                      hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
                                                                      Y : CategoryTheory.GradedObject J D
                                                                      j : J
                                                                      a : Prod I J
                                                                      ha✝ : Eq (p a) j
                                                                      ha : Eq a.1 0
                                                                      ⊢ Eq (Y a.2) (Y j)
                                                                    -/
      (mapBifunctorObjSingle₀ObjIso F X e Y a ha).hom ≫ eqToHom (by aesop)
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    else
      (mapBifunctorObjSingle₀ObjIsInitial F X Y a ha).to _)


@[simp, reassoc]
lemma mapBifunctorLeftUnitorCofan_inj (j : J) :
    (mapBifunctorLeftUnitorCofan F X e p hp Y j).inj ⟨⟨0, j⟩, hp j⟩ =
      (F.map (singleObjApplyIso (0 : I) X).hom).app (Y j) ≫ e.hom.app (Y j) := by
  /-
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁵ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    inst✝³ : Zero I
    inst✝² : DecidableEq I
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
    X : C
    e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
    inst✝ : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
    p : Prod I J → J
    hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
    Y : CategoryTheory.GradedObject J D
    j : J
    ⊢ Eq (CategoryTheory.Limits.Cofan.inj (CategoryTheory.GradedObject.mapBifuncto …
  -/
  simp [mapBifunctorLeftUnitorCofan]
  /-
    🎉 no goals
  -/


/-- The cofan `mapBifunctorLeftUnitorCofan F X e p hp Y j` is a colimit. -/
noncomputable def mapBifunctorLeftUnitorCofanIsColimit (j : J) :
    IsColimit (mapBifunctorLeftUnitorCofan F X e p hp Y j) :=
  mkCofanColimit _
    (fun s => e.inv.app (Y j) ≫
      (F.map (singleObjApplyIso (0 : I) X).inv).app (Y j) ≫ s.inj ⟨⟨0, j⟩, hp j⟩)
    (fun s => by
      /-
        C : Type u_1
        D : Type u_2
        I : Type u_3
        J : Type u_4
        inst✝⁵ : CategoryTheory.Category.{?u.17002, u_1} C
        inst✝⁴ : CategoryTheory.Category.{?u.17006, u_2} D
        inst✝³ : Zero I
        inst✝² : DecidableEq I
        inst✝¹ : CategoryTheory.Limits.HasInitial C
        F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
        X : C
        e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
        inst✝ : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
        p : Prod I J → J
        hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
        Y Y' : CategoryTheory.GradedObject J D
        φ : Quiver.Hom Y Y'
        j : J
        s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
        ⊢ ∀ (j_1 : ↑(Set.preimage p (Singleton.singleton j))), Eq (CategoryTheory.Cate …
      -/
      rintro ⟨⟨i, j'⟩, h⟩
      /-
        case mk.mk
        C : Type u_1
        D : Type u_2
        I : Type u_3
        J : Type u_4
        inst✝⁵ : CategoryTheory.Category.{?u.17002, u_1} C
        inst✝⁴ : CategoryTheory.Category.{?u.17006, u_2} D
        inst✝³ : Zero I
        inst✝² : DecidableEq I
        inst✝¹ : CategoryTheory.Limits.HasInitial C
        F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
        X : C
        e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
        inst✝ : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
        p : Prod I J → J
        hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
        Y Y' : CategoryTheory.GradedObject J D
        φ : Quiver.Hom Y Y'
        j : J
        s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
        i : I
        j' : J
        h : Membership.mem (Set.preimage p (Singleton.singleton j)) { fst := i, snd := …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (Cat …
      -/
      by_cases hi : i = 0
        /-
          case pos
          C : Type u_1
          D : Type u_2
          I : Type u_3
          J : Type u_4
          inst✝⁵ : CategoryTheory.Category.{?u.17002, u_1} C
          inst✝⁴ : CategoryTheory.Category.{?u.17006, u_2} D
          inst✝³ : Zero I
          inst✝² : DecidableEq I
          inst✝¹ : CategoryTheory.Limits.HasInitial C
          F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
          X : C
          e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
          inst✝ : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
          p : Prod I J → J
          hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
          Y Y' : CategoryTheory.GradedObject J D
          φ : Quiver.Hom Y Y'
          j : J
          s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
          i : I
          j' : J
          h : Membership.mem (Set.preimage p (Singleton.singleton j)) { fst := i, snd := …
          hi : Eq i 0
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (Cat …
        -/
      · subst hi
        /-
          case pos
          C : Type u_1
          D : Type u_2
          I : Type u_3
          J : Type u_4
          inst✝⁵ : CategoryTheory.Category.{?u.17002, u_1} C
          inst✝⁴ : CategoryTheory.Category.{?u.17006, u_2} D
          inst✝³ : Zero I
          inst✝² : DecidableEq I
          inst✝¹ : CategoryTheory.Limits.HasInitial C
          F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
          X : C
          e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
          inst✝ : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
          p : Prod I J → J
          hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
          Y Y' : CategoryTheory.GradedObject J D
          φ : Quiver.Hom Y Y'
          j : J
          s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
          j' : J
          h : Membership.mem (Set.preimage p (Singleton.singleton j)) { fst := 0, snd := …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (Cat …
        -/
        simp only [Set.mem_preimage, hp, Set.mem_singleton_iff] at h
        /-
          case pos
          C : Type u_1
          D : Type u_2
          I : Type u_3
          J : Type u_4
          inst✝⁵ : CategoryTheory.Category.{?u.17002, u_1} C
          inst✝⁴ : CategoryTheory.Category.{?u.17006, u_2} D
          inst✝³ : Zero I
          inst✝² : DecidableEq I
          inst✝¹ : CategoryTheory.Limits.HasInitial C
          F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
          X : C
          e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
          inst✝ : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
          p : Prod I J → J
          hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
          Y Y' : CategoryTheory.GradedObject J D
          φ : Quiver.Hom Y Y'
          j : J
          s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
          j' : J
          h✝ : Membership.mem (Set.preimage p (Singleton.singleton j)) { fst := 0, snd : …
          h : Eq j' j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (Cat …
        -/
        subst h
        /-
          case pos
          C : Type u_1
          D : Type u_2
          I : Type u_3
          J : Type u_4
          inst✝⁵ : CategoryTheory.Category.{?u.17002, u_1} C
          inst✝⁴ : CategoryTheory.Category.{?u.17006, u_2} D
          inst✝³ : Zero I
          inst✝² : DecidableEq I
          inst✝¹ : CategoryTheory.Limits.HasInitial C
          F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
          X : C
          e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
          inst✝ : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
          p : Prod I J → J
          hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
          Y Y' : CategoryTheory.GradedObject J D
          φ : Quiver.Hom Y Y'
          j' : J
          s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
          h : Membership.mem (Set.preimage p (Singleton.singleton j')) { fst := 0, snd : …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (Cat …
        -/
        simp
        /-
          🎉 no goals
        -/
        /-
          case neg
          C : Type u_1
          D : Type u_2
          I : Type u_3
          J : Type u_4
          inst✝⁵ : CategoryTheory.Category.{?u.17002, u_1} C
          inst✝⁴ : CategoryTheory.Category.{?u.17006, u_2} D
          inst✝³ : Zero I
          inst✝² : DecidableEq I
          inst✝¹ : CategoryTheory.Limits.HasInitial C
          F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
          X : C
          e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
          inst✝ : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
          p : Prod I J → J
          hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
          Y Y' : CategoryTheory.GradedObject J D
          φ : Quiver.Hom Y Y'
          j : J
          s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
          i : I
          j' : J
          h : Membership.mem (Set.preimage p (Singleton.singleton j)) { fst := i, snd := …
          hi : Not (Eq i 0)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (Cat …
        -/
      · apply IsInitial.hom_ext
        /-
          case neg.t
          C : Type u_1
          D : Type u_2
          I : Type u_3
          J : Type u_4
          inst✝⁵ : CategoryTheory.Category.{?u.17002, u_1} C
          inst✝⁴ : CategoryTheory.Category.{?u.17006, u_2} D
          inst✝³ : Zero I
          inst✝² : DecidableEq I
          inst✝¹ : CategoryTheory.Limits.HasInitial C
          F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
          X : C
          e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
          inst✝ : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
          p : Prod I J → J
          hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
          Y Y' : CategoryTheory.GradedObject J D
          φ : Quiver.Hom Y Y'
          j : J
          s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
          i : I
          j' : J
          h : Membership.mem (Set.preimage p (Singleton.singleton j)) { fst := i, snd := …
          hi : Not (Eq i 0)
          ⊢ CategoryTheory.Limits.IsInitial ((((CategoryTheory.GradedObject.mapBifunctor …
        -/
        exact mapBifunctorObjSingle₀ObjIsInitial _ _ _ _ hi)
        /-
          🎉 no goals
        -/
                      /-
                        C : Type u_1
                        D : Type u_2
                        I : Type u_3
                        J : Type u_4
                        inst✝⁵ : CategoryTheory.Category.{?u.17002, u_1} C
                        inst✝⁴ : CategoryTheory.Category.{?u.17006, u_2} D
                        inst✝³ : Zero I
                        inst✝² : DecidableEq I
                        inst✝¹ : CategoryTheory.Limits.HasInitial C
                        F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
                        X : C
                        e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
                        inst✝ : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
                        p : Prod I J → J
                        hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
                        Y Y' : CategoryTheory.GradedObject J D
                        φ : Quiver.Hom Y Y'
                        j : J
                        s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
                        m : Quiver.Hom (CategoryTheory.GradedObject.mapBifunctorLeftUnitorCofan F X e  …
                        hm : ∀ (j_1 : ↑(Set.preimage p (Singleton.singleton j))), Eq (CategoryTheory.C …
                        ⊢ Eq m ((fun s => CategoryTheory.CategoryStruct.comp (e.inv.app (Y j)) (Catego …
                      -/
    (fun s m hm => by simp [← hm ⟨⟨0, j⟩, hp j⟩])
                      /-
                        🎉 no goals
                      -/


include e hp in
lemma mapBifunctorLeftUnitor_hasMap :
    HasMap (((mapBifunctor F I J).obj ((single₀ I).obj X)).obj Y) p :=
  CofanMapObjFun.hasMap _ _ _ (mapBifunctorLeftUnitorCofanIsColimit F X e p hp Y)


/-- Given `F : C ⥤ D ⥤ D`, `X : C`, `e : F.obj X ≅ 𝟭 D`, `Y : GradedObject J D` and
`p : I × J → J` such that `p ⟨0, j⟩ = j` for all `j`,
this is the left unitor isomorphism `mapBifunctorMapObj F p ((single₀ I).obj X) Y ≅ Y`. -/
noncomputable def mapBifunctorLeftUnitor : mapBifunctorMapObj F p ((single₀ I).obj X) Y ≅ Y :=
  isoMk _ _ (fun j => (CofanMapObjFun.iso
    (mapBifunctorLeftUnitorCofanIsColimit F X e p hp Y j)).symm)


@[reassoc (attr := simp)]
lemma ι_mapBifunctorLeftUnitor_hom_apply (j : J) :
    ιMapBifunctorMapObj F p ((single₀ I).obj X) Y 0 j j (hp j) ≫
      (mapBifunctorLeftUnitor F X e p hp Y).hom j =
      (F.map (singleObjApplyIso (0 : I) X).hom).app _ ≫ e.hom.app (Y j) := by
  /-
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁶ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁴ : Zero I
    inst✝³ : DecidableEq I
    inst✝² : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
    X : C
    e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
    inst✝¹ : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    p : Prod I J → J
    hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
    Y : CategoryTheory.GradedObject J D
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj ((CategoryTheor …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapBifu …
  -/
  dsimp [mapBifunctorLeftUnitor]
  /-
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁶ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁴ : Zero I
    inst✝³ : DecidableEq I
    inst✝² : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
    X : C
    e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
    inst✝¹ : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    p : Prod I J → J
    hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
    Y : CategoryTheory.GradedObject J D
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj ((CategoryTheor …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapBifu …
  -/
  erw [CofanMapObjFun.ιMapObj_iso_inv]
  /-
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁶ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁴ : Zero I
    inst✝³ : DecidableEq I
    inst✝² : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
    X : C
    e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
    inst✝¹ : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    p : Prod I J → J
    hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
    Y : CategoryTheory.GradedObject J D
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj ((CategoryTheor …
    j : J
    ⊢ Eq (CategoryTheory.Limits.Cofan.inj (CategoryTheory.GradedObject.mapBifuncto …
  -/
  rw [mapBifunctorLeftUnitorCofan_inj]
  /-
    🎉 no goals
  -/


lemma mapBifunctorLeftUnitor_inv_apply (j : J) :
    (mapBifunctorLeftUnitor F X e p hp Y).inv j =
      e.inv.app (Y j) ≫ (F.map (singleObjApplyIso (0 : I) X).inv).app (Y j) ≫
      ιMapBifunctorMapObj F p ((single₀ I).obj X) Y 0 j j (hp j) := rfl


@[reassoc]
lemma mapBifunctorLeftUnitor_inv_naturality :
    φ ≫ (mapBifunctorLeftUnitor F X e p hp Y').inv =
      (mapBifunctorLeftUnitor F X e p hp Y).inv ≫ mapBifunctorMapMap F p (𝟙 _) φ := by
  /-
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁵ : Zero I
    inst✝⁴ : DecidableEq I
    inst✝³ : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
    X : C
    e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
    inst✝² : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    p : Prod I J → J
    hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
    Y Y' : CategoryTheory.GradedObject J D
    φ : Quiver.Hom Y Y'
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj ((CategoryTheo …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj ((CategoryTheor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.GradedObject.mapBif …
  -/
  ext j
  /-
    case h
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁵ : Zero I
    inst✝⁴ : DecidableEq I
    inst✝³ : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
    X : C
    e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
    inst✝² : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    p : Prod I J → J
    hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
    Y Y' : CategoryTheory.GradedObject J D
    φ : Quiver.Hom Y Y'
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj ((CategoryTheo …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj ((CategoryTheor …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.GradedObject.mapBif …
  -/
  dsimp
  rw [mapBifunctorLeftUnitor_inv_apply, mapBifunctorLeftUnitor_inv_apply, assoc, assoc,
    ι_mapBifunctorMapMap]
  /-
    case h
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁵ : Zero I
    inst✝⁴ : DecidableEq I
    inst✝³ : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
    X : C
    e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
    inst✝² : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    p : Prod I J → J
    hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
    Y Y' : CategoryTheory.GradedObject J D
    φ : Quiver.Hom Y Y'
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj ((CategoryTheo …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj ((CategoryTheor …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ j) (CategoryTheory.CategoryStruct. …
  -/
  dsimp
  rw [Functor.map_id, NatTrans.id_app, id_comp, ← NatTrans.naturality_assoc,
    ← NatTrans.naturality_assoc]
  /-
    case h
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁵ : Zero I
    inst✝⁴ : DecidableEq I
    inst✝³ : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor C (CategoryTheory.Functor D D)
    X : C
    e : CategoryTheory.Iso (F.obj X) (CategoryTheory.Functor.id D)
    inst✝² : ∀ (Y : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    p : Prod I J → J
    hp : ∀ (j : J), Eq (p { fst := 0, snd := j }) j
    Y Y' : CategoryTheory.GradedObject J D
    φ : Quiver.Hom Y Y'
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj ((CategoryTheo …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj ((CategoryTheor …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ j) (CategoryTheory.CategoryStruct. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma mapBifunctorLeftUnitor_naturality :
    mapBifunctorMapMap F p (𝟙 _) φ ≫ (mapBifunctorLeftUnitor F X e p hp Y').hom =
      (mapBifunctorLeftUnitor F X e p hp Y).hom ≫ φ := by
  rw [← cancel_mono (mapBifunctorLeftUnitor F X e p hp Y').inv, assoc, assoc, Iso.hom_inv_id,
    comp_id, mapBifunctorLeftUnitor_inv_naturality, Iso.hom_inv_id_assoc]


/-- Given `F : D ⥤ C ⥤ D`, `Y : C`, `e : F.flip.obj X ≅ 𝟭 D` and `X : GradedObject J D`,
this is the isomorphism `((mapBifunctor F J I).obj X).obj ((single₀ I).obj Y) a ≅ Y a.2`
when `a : J × I` is such that `a.2 = 0`. -/
@[simps!]
noncomputable def mapBifunctorObjObjSingle₀Iso (a : J × I) (ha : a.2 = 0) :
    ((mapBifunctor F J I).obj X).obj ((single₀ I).obj Y) a ≅ X a.1 :=
  Functor.mapIso _ (singleObjApplyIsoOfEq _ Y _ ha) ≪≫ e.app (X a.1)


/-- Given `F : D ⥤ C ⥤ D`, `Y : C` and `X : GradedObject J D`,
`((mapBifunctor F J I).obj X).obj ((single₀ I).obj X) a` is an initial when `a : J × I`
is such that `a.2 ≠ 0`. -/
noncomputable def mapBifunctorObjObjSingle₀IsInitial (a : J × I) (ha : a.2 ≠ 0) :
    IsInitial (((mapBifunctor F J I).obj X).obj ((single₀ I).obj Y) a) :=
  IsInitial.isInitialObj (F.obj (X a.1)) _ (isInitialSingleObjApply _ _ _ ha)


/-- Given `F : D ⥤ C ⥤ D`, `Y : C`, `e : F.flip.obj Y ≅ 𝟭 D`, `X : GradedObject J D` and
`p : J × I → J` such that `p ⟨j, 0⟩ = j` for all `j`,
this is the (colimit) cofan which shall be used to construct the isomorphism
`mapBifunctorMapObj F p X ((single₀ I).obj Y) ≅ X`, see `mapBifunctorRightUnitor`. -/
noncomputable def mapBifunctorRightUnitorCofan (hp : ∀ (j : J), p ⟨j, 0⟩ = j) (X) (j : J) :
    (((mapBifunctor F J I).obj X).obj ((single₀ I).obj Y)).CofanMapObjFun p j :=
  CofanMapObjFun.mk _ _ _ (X j) (fun a ha =>
    if ha : a.2 = 0 then
                                                                    /-
                                                                      C : Type u_1
                                                                      D : Type u_2
                                                                      I : Type u_3
                                                                      J : Type u_4
                                                                      inst✝⁵ : CategoryTheory.Category.{?u.79401, u_1} C
                                                                      inst✝⁴ : CategoryTheory.Category.{?u.79405, u_2} D
                                                                      inst✝³ : Zero I
                                                                      inst✝² : DecidableEq I
                                                                      inst✝¹ : CategoryTheory.Limits.HasInitial C
                                                                      F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
                                                                      Y : C
                                                                      e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
                                                                      inst✝ : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
                                                                      p : Prod J I → J
                                                                      hp✝ : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
                                                                      X✝ X' : CategoryTheory.GradedObject J D
                                                                      φ : Quiver.Hom X✝ X'
                                                                      hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
                                                                      X : CategoryTheory.GradedObject J D
                                                                      j : J
                                                                      a : Prod J I
                                                                      ha✝ : Eq (p a) j
                                                                      ha : Eq a.2 0
                                                                      ⊢ Eq (X a.1) (X j)
                                                                    -/
      (mapBifunctorObjObjSingle₀Iso F Y e X a ha).hom ≫ eqToHom (by aesop)
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    else
      (mapBifunctorObjObjSingle₀IsInitial F Y X a ha).to _)


@[simp, reassoc]
lemma mapBifunctorRightUnitorCofan_inj (j : J) :
    (mapBifunctorRightUnitorCofan F Y e p hp X j).inj ⟨⟨j, 0⟩, hp j⟩ =
      (F.obj (X j)).map (singleObjApplyIso (0 : I) Y).hom ≫ e.hom.app (X j) := by
  /-
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁵ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    inst✝³ : Zero I
    inst✝² : DecidableEq I
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
    Y : C
    e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
    inst✝ : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
    p : Prod J I → J
    hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
    X : CategoryTheory.GradedObject J D
    j : J
    ⊢ Eq (CategoryTheory.Limits.Cofan.inj (CategoryTheory.GradedObject.mapBifuncto …
  -/
  simp [mapBifunctorRightUnitorCofan]
  /-
    🎉 no goals
  -/


/-- The cofan `mapBifunctorRightUnitorCofan F Y e p hp X j` is a colimit. -/
noncomputable def mapBifunctorRightUnitorCofanIsColimit (j : J) :
    IsColimit (mapBifunctorRightUnitorCofan F Y e p hp X j) :=
  mkCofanColimit _
    (fun s => e.inv.app (X j) ≫
      (F.obj (X j)).map (singleObjApplyIso (0 : I) Y).inv ≫ s.inj ⟨⟨j, 0⟩, hp j⟩)
    (fun s => by
      /-
        C : Type u_1
        D : Type u_2
        I : Type u_3
        J : Type u_4
        inst✝⁵ : CategoryTheory.Category.{?u.90268, u_1} C
        inst✝⁴ : CategoryTheory.Category.{?u.90272, u_2} D
        inst✝³ : Zero I
        inst✝² : DecidableEq I
        inst✝¹ : CategoryTheory.Limits.HasInitial C
        F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
        Y : C
        e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
        inst✝ : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
        p : Prod J I → J
        hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
        X X' : CategoryTheory.GradedObject J D
        φ : Quiver.Hom X X'
        j : J
        s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
        ⊢ ∀ (j_1 : ↑(Set.preimage p (Singleton.singleton j))), Eq (CategoryTheory.Cate …
      -/
      rintro ⟨⟨j', i⟩, h⟩
      /-
        case mk.mk
        C : Type u_1
        D : Type u_2
        I : Type u_3
        J : Type u_4
        inst✝⁵ : CategoryTheory.Category.{?u.90268, u_1} C
        inst✝⁴ : CategoryTheory.Category.{?u.90272, u_2} D
        inst✝³ : Zero I
        inst✝² : DecidableEq I
        inst✝¹ : CategoryTheory.Limits.HasInitial C
        F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
        Y : C
        e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
        inst✝ : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
        p : Prod J I → J
        hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
        X X' : CategoryTheory.GradedObject J D
        φ : Quiver.Hom X X'
        j : J
        s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
        j' : J
        i : I
        h : Membership.mem (Set.preimage p (Singleton.singleton j)) { fst := j', snd : …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (Cat …
      -/
      by_cases hi : i = 0
        /-
          case pos
          C : Type u_1
          D : Type u_2
          I : Type u_3
          J : Type u_4
          inst✝⁵ : CategoryTheory.Category.{?u.90268, u_1} C
          inst✝⁴ : CategoryTheory.Category.{?u.90272, u_2} D
          inst✝³ : Zero I
          inst✝² : DecidableEq I
          inst✝¹ : CategoryTheory.Limits.HasInitial C
          F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
          Y : C
          e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
          inst✝ : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
          p : Prod J I → J
          hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
          X X' : CategoryTheory.GradedObject J D
          φ : Quiver.Hom X X'
          j : J
          s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
          j' : J
          i : I
          h : Membership.mem (Set.preimage p (Singleton.singleton j)) { fst := j', snd : …
          hi : Eq i 0
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (Cat …
        -/
      · subst hi
        /-
          case pos
          C : Type u_1
          D : Type u_2
          I : Type u_3
          J : Type u_4
          inst✝⁵ : CategoryTheory.Category.{?u.90268, u_1} C
          inst✝⁴ : CategoryTheory.Category.{?u.90272, u_2} D
          inst✝³ : Zero I
          inst✝² : DecidableEq I
          inst✝¹ : CategoryTheory.Limits.HasInitial C
          F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
          Y : C
          e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
          inst✝ : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
          p : Prod J I → J
          hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
          X X' : CategoryTheory.GradedObject J D
          φ : Quiver.Hom X X'
          j : J
          s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
          j' : J
          h : Membership.mem (Set.preimage p (Singleton.singleton j)) { fst := j', snd : …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (Cat …
        -/
        simp only [Set.mem_preimage, hp, Set.mem_singleton_iff] at h
        /-
          case pos
          C : Type u_1
          D : Type u_2
          I : Type u_3
          J : Type u_4
          inst✝⁵ : CategoryTheory.Category.{?u.90268, u_1} C
          inst✝⁴ : CategoryTheory.Category.{?u.90272, u_2} D
          inst✝³ : Zero I
          inst✝² : DecidableEq I
          inst✝¹ : CategoryTheory.Limits.HasInitial C
          F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
          Y : C
          e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
          inst✝ : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
          p : Prod J I → J
          hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
          X X' : CategoryTheory.GradedObject J D
          φ : Quiver.Hom X X'
          j : J
          s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
          j' : J
          h✝ : Membership.mem (Set.preimage p (Singleton.singleton j)) { fst := j', snd  …
          h : Eq j' j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (Cat …
        -/
        subst h
        /-
          case pos
          C : Type u_1
          D : Type u_2
          I : Type u_3
          J : Type u_4
          inst✝⁵ : CategoryTheory.Category.{?u.90268, u_1} C
          inst✝⁴ : CategoryTheory.Category.{?u.90272, u_2} D
          inst✝³ : Zero I
          inst✝² : DecidableEq I
          inst✝¹ : CategoryTheory.Limits.HasInitial C
          F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
          Y : C
          e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
          inst✝ : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
          p : Prod J I → J
          hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
          X X' : CategoryTheory.GradedObject J D
          φ : Quiver.Hom X X'
          j' : J
          s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
          h : Membership.mem (Set.preimage p (Singleton.singleton j')) { fst := j', snd  …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (Cat …
        -/
        dsimp
        rw [mapBifunctorRightUnitorCofan_inj, assoc, Iso.hom_inv_id_app_assoc,
          ← Functor.map_comp_assoc, Iso.hom_inv_id, Functor.map_id, id_comp]
        /-
          case neg
          C : Type u_1
          D : Type u_2
          I : Type u_3
          J : Type u_4
          inst✝⁵ : CategoryTheory.Category.{?u.90268, u_1} C
          inst✝⁴ : CategoryTheory.Category.{?u.90272, u_2} D
          inst✝³ : Zero I
          inst✝² : DecidableEq I
          inst✝¹ : CategoryTheory.Limits.HasInitial C
          F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
          Y : C
          e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
          inst✝ : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
          p : Prod J I → J
          hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
          X X' : CategoryTheory.GradedObject J D
          φ : Quiver.Hom X X'
          j : J
          s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
          j' : J
          i : I
          h : Membership.mem (Set.preimage p (Singleton.singleton j)) { fst := j', snd : …
          hi : Not (Eq i 0)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (Cat …
        -/
      · apply IsInitial.hom_ext
        /-
          case neg.t
          C : Type u_1
          D : Type u_2
          I : Type u_3
          J : Type u_4
          inst✝⁵ : CategoryTheory.Category.{?u.90268, u_1} C
          inst✝⁴ : CategoryTheory.Category.{?u.90272, u_2} D
          inst✝³ : Zero I
          inst✝² : DecidableEq I
          inst✝¹ : CategoryTheory.Limits.HasInitial C
          F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
          Y : C
          e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
          inst✝ : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
          p : Prod J I → J
          hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
          X X' : CategoryTheory.GradedObject J D
          φ : Quiver.Hom X X'
          j : J
          s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
          j' : J
          i : I
          h : Membership.mem (Set.preimage p (Singleton.singleton j)) { fst := j', snd : …
          hi : Not (Eq i 0)
          ⊢ CategoryTheory.Limits.IsInitial ((((CategoryTheory.GradedObject.mapBifunctor …
        -/
        exact mapBifunctorObjObjSingle₀IsInitial _ _ _ _ hi)
        /-
          🎉 no goals
        -/
    (fun s m hm => by
      /-
        C : Type u_1
        D : Type u_2
        I : Type u_3
        J : Type u_4
        inst✝⁵ : CategoryTheory.Category.{?u.90268, u_1} C
        inst✝⁴ : CategoryTheory.Category.{?u.90272, u_2} D
        inst✝³ : Zero I
        inst✝² : DecidableEq I
        inst✝¹ : CategoryTheory.Limits.HasInitial C
        F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
        Y : C
        e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
        inst✝ : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Func …
        p : Prod J I → J
        hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
        X X' : CategoryTheory.GradedObject J D
        φ : Quiver.Hom X X'
        j : J
        s : CategoryTheory.Limits.Cofan ((((CategoryTheory.GradedObject.mapBifunctor F …
        m : Quiver.Hom (CategoryTheory.GradedObject.mapBifunctorRightUnitorCofan F Y e …
        hm : ∀ (j_1 : ↑(Set.preimage p (Singleton.singleton j))), Eq (CategoryTheory.C …
        ⊢ Eq m ((fun s => CategoryTheory.CategoryStruct.comp (e.inv.app (X j)) (Catego …
      -/
      dsimp
      rw [← hm ⟨⟨j, 0⟩, hp j⟩, mapBifunctorRightUnitorCofan_inj, assoc, ← Functor.map_comp_assoc,
        Iso.inv_hom_id, Functor.map_id, id_comp, Iso.inv_hom_id_app_assoc])


include e hp in
lemma mapBifunctorRightUnitor_hasMap :
    HasMap (((mapBifunctor F J I).obj X).obj ((single₀ I).obj Y)) p :=
  CofanMapObjFun.hasMap _ _ _ (mapBifunctorRightUnitorCofanIsColimit F Y e p hp X)


/-- Given `F : D ⥤ C ⥤ D`, `Y : C`, `e : F.flip.obj Y ≅ 𝟭 D`, `X : GradedObject J D` and
`p : J × I → J` such that `p ⟨j, 0⟩ = j` for all `j`,
this is the right unitor isomorphism `mapBifunctorMapObj F p X ((single₀ I).obj Y) ≅ X`. -/
noncomputable def mapBifunctorRightUnitor : mapBifunctorMapObj F p X ((single₀ I).obj Y) ≅ X :=
  isoMk _ _ (fun j => (CofanMapObjFun.iso
    (mapBifunctorRightUnitorCofanIsColimit F Y e p hp X j)).symm)


@[reassoc (attr := simp)]
lemma ι_mapBifunctorRightUnitor_hom_apply (j : J) :
    ιMapBifunctorMapObj F p X ((single₀ I).obj Y) j 0 j (hp j) ≫
        (mapBifunctorRightUnitor F Y e p hp X).hom j =
      (F.obj (X j)).map (singleObjApplyIso (0 : I) Y).hom ≫ e.hom.app (X j) := by
  /-
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁶ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁴ : Zero I
    inst✝³ : DecidableEq I
    inst✝² : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
    Y : C
    e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
    inst✝¹ : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    p : Prod J I → J
    hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
    X : CategoryTheory.GradedObject J D
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F J I).obj X).obj ((Catego …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapBifu …
  -/
  dsimp [mapBifunctorRightUnitor]
  /-
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁶ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁴ : Zero I
    inst✝³ : DecidableEq I
    inst✝² : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
    Y : C
    e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
    inst✝¹ : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    p : Prod J I → J
    hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
    X : CategoryTheory.GradedObject J D
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F J I).obj X).obj ((Catego …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapBifu …
  -/
  erw [CofanMapObjFun.ιMapObj_iso_inv]
  /-
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁶ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁴ : Zero I
    inst✝³ : DecidableEq I
    inst✝² : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
    Y : C
    e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
    inst✝¹ : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    p : Prod J I → J
    hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
    X : CategoryTheory.GradedObject J D
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F J I).obj X).obj ((Catego …
    j : J
    ⊢ Eq (CategoryTheory.Limits.Cofan.inj (CategoryTheory.GradedObject.mapBifuncto …
  -/
  rw [mapBifunctorRightUnitorCofan_inj]
  /-
    🎉 no goals
  -/


lemma mapBifunctorRightUnitor_inv_apply (j : J) :
    (mapBifunctorRightUnitor F Y e p hp X).inv j =
      e.inv.app (X j) ≫ (F.obj (X j)).map (singleObjApplyIso (0 : I) Y).inv ≫
        ιMapBifunctorMapObj F p X ((single₀ I).obj Y) j 0 j (hp j) := rfl


@[reassoc]
lemma mapBifunctorRightUnitor_inv_naturality :
    φ ≫ (mapBifunctorRightUnitor F Y e p hp X').inv =
      (mapBifunctorRightUnitor F Y e p hp X).inv ≫ mapBifunctorMapMap F p φ (𝟙 _) := by
  /-
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁵ : Zero I
    inst✝⁴ : DecidableEq I
    inst✝³ : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
    Y : C
    e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
    inst✝² : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    p : Prod J I → J
    hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
    X X' : CategoryTheory.GradedObject J D
    φ : Quiver.Hom X X'
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F J I).obj X).obj ((Categ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F J I).obj X').obj ((Categ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.GradedObject.mapBif …
  -/
  ext j
  /-
    case h
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁵ : Zero I
    inst✝⁴ : DecidableEq I
    inst✝³ : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
    Y : C
    e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
    inst✝² : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    p : Prod J I → J
    hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
    X X' : CategoryTheory.GradedObject J D
    φ : Quiver.Hom X X'
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F J I).obj X).obj ((Categ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F J I).obj X').obj ((Categ …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.GradedObject.mapBif …
  -/
  dsimp
  rw [mapBifunctorRightUnitor_inv_apply, mapBifunctorRightUnitor_inv_apply, assoc, assoc,
    ι_mapBifunctorMapMap]
  /-
    case h
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁵ : Zero I
    inst✝⁴ : DecidableEq I
    inst✝³ : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
    Y : C
    e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
    inst✝² : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    p : Prod J I → J
    hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
    X X' : CategoryTheory.GradedObject J D
    φ : Quiver.Hom X X'
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F J I).obj X).obj ((Categ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F J I).obj X').obj ((Categ …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ j) (CategoryTheory.CategoryStruct. …
  -/
  dsimp
  /-
    case h
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁵ : Zero I
    inst✝⁴ : DecidableEq I
    inst✝³ : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
    Y : C
    e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
    inst✝² : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    p : Prod J I → J
    hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
    X X' : CategoryTheory.GradedObject J D
    φ : Quiver.Hom X X'
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F J I).obj X).obj ((Categ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F J I).obj X').obj ((Categ …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ j) (CategoryTheory.CategoryStruct. …
  -/
  rw [Functor.map_id, id_comp, NatTrans.naturality_assoc]
  /-
    case h
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁵ : Zero I
    inst✝⁴ : DecidableEq I
    inst✝³ : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
    Y : C
    e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
    inst✝² : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    p : Prod J I → J
    hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
    X X' : CategoryTheory.GradedObject J D
    φ : Quiver.Hom X X'
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F J I).obj X).obj ((Categ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F J I).obj X').obj ((Categ …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ j) (CategoryTheory.CategoryStruct. …
  -/
  erw [← NatTrans.naturality_assoc e.inv]
  /-
    case h
    C : Type u_1
    D : Type u_2
    I : Type u_3
    J : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} D
    inst✝⁵ : Zero I
    inst✝⁴ : DecidableEq I
    inst✝³ : CategoryTheory.Limits.HasInitial C
    F : CategoryTheory.Functor D (CategoryTheory.Functor C D)
    Y : C
    e : CategoryTheory.Iso (F.flip.obj Y) (CategoryTheory.Functor.id D)
    inst✝² : ∀ (X : D), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    p : Prod J I → J
    hp : ∀ (j : J), Eq (p { fst := j, snd := 0 }) j
    X X' : CategoryTheory.GradedObject J D
    φ : Quiver.Hom X X'
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F J I).obj X).obj ((Categ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F J I).obj X').obj ((Categ …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ j) (CategoryTheory.CategoryStruct. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma mapBifunctorRightUnitor_naturality :
    mapBifunctorMapMap F p φ (𝟙 _) ≫ (mapBifunctorRightUnitor F Y e p hp X').hom =
      (mapBifunctorRightUnitor F Y e p hp X).hom ≫ φ := by
  rw [← cancel_mono (mapBifunctorRightUnitor F Y e p hp X').inv, assoc, assoc, Iso.hom_inv_id,
    comp_id, mapBifunctorRightUnitor_inv_naturality, Iso.hom_inv_id_assoc]


/-- Given two maps `r : I₁ × I₂ × I₃ → J` and `π : I₁ × I₃ → J`, this structure is the
input in the formulation of the triangle equality `mapBifunctor_triangle` which
relates the left and right unitor and the associator for `GradedObject.mapBifunctor`. -/
structure TriangleIndexData (r : I₁ × I₂ × I₃ → J) (π : I₁ × I₃ → J) where
  /-- a map `I₁ × I₂ → I₁` -/
  p₁₂ : I₁ × I₂ → I₁
  hp₁₂ (i : I₁ × I₂ × I₃) : π ⟨p₁₂ ⟨i.1, i.2.1⟩, i.2.2⟩ = r i
  /-- a map `I₂ × I₃ → I₃` -/
  p₂₃ : I₂ × I₃ → I₃
  hp₂₃ (i : I₁ × I₂ × I₃) : π ⟨i.1, p₂₃ i.2⟩ = r i
  h₁ (i₁ : I₁) : p₁₂ (i₁, 0) = i₁
  h₃ (i₃ : I₃) : p₂₃ (0, i₃) = i₃


lemma r_zero (i₁ : I₁) (i₃ : I₃) : r ⟨i₁, 0, i₃⟩ = π ⟨i₁, i₃⟩ := by
  /-
    I₁ : Type u_1
    I₂ : Type u_2
    I₃ : Type u_3
    J : Type u_4
    inst✝ : Zero I₂
    r : Prod I₁ (Prod I₂ I₃) → J
    π : Prod I₁ I₃ → J
    τ : CategoryTheory.GradedObject.TriangleIndexData r π
    i₁ : I₁
    i₃ : I₃
    ⊢ Eq (r { fst := i₁, snd := { fst := 0, snd := i₃ } }) (π { fst := i₁, snd :=  …
  -/
  rw [← τ.hp₂₃, τ.h₃ i₃]
  /-
    🎉 no goals
  -/


/-- The `BifunctorComp₁₂IndexData r` attached to a `TriangleIndexData r π`. -/
@[reducible]
def ρ₁₂ : BifunctorComp₁₂IndexData r where
  I₁₂ := I₁
  p := τ.p₁₂
  q := π
  hpq := τ.hp₁₂


/-- The `BifunctorComp₂₃IndexData r` attached to a `TriangleIndexData r π`. -/
@[reducible]
def ρ₂₃ : BifunctorComp₂₃IndexData r where
  I₂₃ := I₃
  p := τ.p₂₃
  q := π
  hpq := τ.hp₂₃


lemma mapBifunctor_triangle
    (triangle : ∀ (X₁ : C₁) (X₃ : C₃), ((associator.hom.app X₁).app X₂).app X₃ ≫
    (G.obj X₁).map (e₂.hom.app X₃) = (G.map (e₁.hom.app X₁)).app X₃) :
    (mapBifunctorAssociator associator τ.ρ₁₂ τ.ρ₂₃ X₁ ((single₀ I₂).obj X₂) X₃).hom ≫
    mapBifunctorMapMap G π (𝟙 X₁) (mapBifunctorLeftUnitor F₂ X₂ e₂ τ.p₂₃ τ.h₃ X₃).hom =
      mapBifunctorMapMap G π (mapBifunctorRightUnitor F₁ X₂ e₁ τ.p₁₂ τ.h₁ X₁).hom (𝟙 X₃) := by
  rw [← cancel_epi ((mapBifunctorMapMap G π
    (mapBifunctorRightUnitor F₁ X₂ e₁ τ.p₁₂ τ.h₁ X₁).inv (𝟙 X₃)))]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D : Type u_4
    I₁ : Type u_5
    I₂ : Type u_6
    I₃ : Type u_7
    J : Type u_8
    inst✝¹⁵ : CategoryTheory.Category.{u_12, u_1} C₁
    inst✝¹⁴ : CategoryTheory.Category.{u_11, u_2} C₂
    inst✝¹³ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝¹² : CategoryTheory.Category.{u_9, u_4} D
    inst✝¹¹ : Zero I₂
    inst✝¹⁰ : DecidableEq I₂
    inst✝⁹ : CategoryTheory.Limits.HasInitial C₂
    F₁ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁)
    F₂ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₃)
    G : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₃ D)
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁ G) (Categor …
    X₂ : C₂
    e₁ : CategoryTheory.Iso (F₁.flip.obj X₂) (CategoryTheory.Functor.id C₁)
    e₂ : CategoryTheory.Iso (F₂.obj X₂) (CategoryTheory.Functor.id C₃)
    inst✝⁸ : ∀ (X₁ : C₁), CategoryTheory.Limits.PreservesColimit (CategoryTheory.F …
    inst✝⁷ : ∀ (X₃ : C₃), CategoryTheory.Limits.PreservesColimit (CategoryTheory.F …
    r : Prod I₁ (Prod I₂ I₃) → J
    π : Prod I₁ I₃ → J
    τ : CategoryTheory.GradedObject.TriangleIndexData r π
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝⁶ : (((CategoryTheory.GradedObject.mapBifunctor F₁ I₁ I₂).obj X₁).obj ((C …
    inst✝⁵ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj (CategoryThe …
    inst✝⁴ : (((CategoryTheory.GradedObject.mapBifunctor F₂ I₂ I₃).obj ((CategoryT …
    inst✝³ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj X₁).obj (Cat …
    inst✝² : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁ G τ.ρ₁₂ X₁ ((Ca …
    inst✝¹ : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj G F₂ τ.ρ₂₃ X₁ ((Ca …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj X₁).obj X₃).H …
    triangle : ∀ (X₁ : C₁) (X₃ : C₃), Eq (CategoryTheory.CategoryStruct.comp (((as …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.mapBifun …
  -/
  ext j i₁ i₃ hj
  simp only [categoryOfGradedObjects_comp, ι_mapBifunctorMapMap_assoc,
    mapBifunctorRightUnitor_inv_apply, Functor.id_obj, Functor.flip_obj_obj, Functor.map_comp,
    NatTrans.comp_app, categoryOfGradedObjects_id, Functor.map_id, id_comp, assoc,
    ι_mapBifunctorMapMap]
  /-
    case h.h
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D : Type u_4
    I₁ : Type u_5
    I₂ : Type u_6
    I₃ : Type u_7
    J : Type u_8
    inst✝¹⁵ : CategoryTheory.Category.{u_12, u_1} C₁
    inst✝¹⁴ : CategoryTheory.Category.{u_11, u_2} C₂
    inst✝¹³ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝¹² : CategoryTheory.Category.{u_9, u_4} D
    inst✝¹¹ : Zero I₂
    inst✝¹⁰ : DecidableEq I₂
    inst✝⁹ : CategoryTheory.Limits.HasInitial C₂
    F₁ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁)
    F₂ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₃)
    G : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₃ D)
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁ G) (Categor …
    X₂ : C₂
    e₁ : CategoryTheory.Iso (F₁.flip.obj X₂) (CategoryTheory.Functor.id C₁)
    e₂ : CategoryTheory.Iso (F₂.obj X₂) (CategoryTheory.Functor.id C₃)
    inst✝⁸ : ∀ (X₁ : C₁), CategoryTheory.Limits.PreservesColimit (CategoryTheory.F …
    inst✝⁷ : ∀ (X₃ : C₃), CategoryTheory.Limits.PreservesColimit (CategoryTheory.F …
    r : Prod I₁ (Prod I₂ I₃) → J
    π : Prod I₁ I₃ → J
    τ : CategoryTheory.GradedObject.TriangleIndexData r π
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝⁶ : (((CategoryTheory.GradedObject.mapBifunctor F₁ I₁ I₂).obj X₁).obj ((C …
    inst✝⁵ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj (CategoryThe …
    inst✝⁴ : (((CategoryTheory.GradedObject.mapBifunctor F₂ I₂ I₃).obj ((CategoryT …
    inst✝³ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj X₁).obj (Cat …
    inst✝² : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁ G τ.ρ₁₂ X₁ ((Ca …
    inst✝¹ : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj G F₂ τ.ρ₂₃ X₁ ((Ca …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj X₁).obj X₃).H …
    triangle : ∀ (X₁ : C₁) (X₃ : C₃), Eq (CategoryTheory.CategoryStruct.comp (((as …
    j : J
    i₁ : I₁
    i₃ : I₃
    hj : Eq (π { fst := i₁, snd := i₃ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.map (e₁.inv.app (X₁ i₁))).app (X₃ …
  -/
  congr 2
  rw [← ιMapBifunctor₁₂BifunctorMapObj_eq_assoc F₁ G τ.ρ₁₂ _ _ _ i₁ 0 i₃ j
    (by rw [τ.r_zero, hj]) i₁ (by simp), ι_mapBifunctorAssociator_hom_assoc,
    ιMapBifunctorBifunctor₂₃MapObj_eq_assoc G F₂ τ.ρ₂₃ _ _ _ i₁ 0 i₃ j
    (by rw [τ.r_zero, hj]) i₃ (by simp), ι_mapBifunctorMapMap]
  /-
    case h.h.e_a.e_a
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D : Type u_4
    I₁ : Type u_5
    I₂ : Type u_6
    I₃ : Type u_7
    J : Type u_8
    inst✝¹⁵ : CategoryTheory.Category.{u_12, u_1} C₁
    inst✝¹⁴ : CategoryTheory.Category.{u_11, u_2} C₂
    inst✝¹³ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝¹² : CategoryTheory.Category.{u_9, u_4} D
    inst✝¹¹ : Zero I₂
    inst✝¹⁰ : DecidableEq I₂
    inst✝⁹ : CategoryTheory.Limits.HasInitial C₂
    F₁ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁)
    F₂ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₃)
    G : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₃ D)
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁ G) (Categor …
    X₂ : C₂
    e₁ : CategoryTheory.Iso (F₁.flip.obj X₂) (CategoryTheory.Functor.id C₁)
    e₂ : CategoryTheory.Iso (F₂.obj X₂) (CategoryTheory.Functor.id C₃)
    inst✝⁸ : ∀ (X₁ : C₁), CategoryTheory.Limits.PreservesColimit (CategoryTheory.F …
    inst✝⁷ : ∀ (X₃ : C₃), CategoryTheory.Limits.PreservesColimit (CategoryTheory.F …
    r : Prod I₁ (Prod I₂ I₃) → J
    π : Prod I₁ I₃ → J
    τ : CategoryTheory.GradedObject.TriangleIndexData r π
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝⁶ : (((CategoryTheory.GradedObject.mapBifunctor F₁ I₁ I₂).obj X₁).obj ((C …
    inst✝⁵ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj (CategoryThe …
    inst✝⁴ : (((CategoryTheory.GradedObject.mapBifunctor F₂ I₂ I₃).obj ((CategoryT …
    inst✝³ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj X₁).obj (Cat …
    inst✝² : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁ G τ.ρ₁₂ X₁ ((Ca …
    inst✝¹ : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj G F₂ τ.ρ₂₃ X₁ ((Ca …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj X₁).obj X₃).H …
    triangle : ∀ (X₁ : C₁) (X₃ : C₃), Eq (CategoryTheory.CategoryStruct.comp (((as …
    j : J
    i₁ : I₁
    i₃ : I₃
    hj : Eq (π { fst := i₁, snd := i₃ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((associator.hom.app (X₁ i₁)).app (( …
  -/
  dsimp
  rw [Functor.map_id, NatTrans.id_app, id_comp,
    ← Functor.map_comp_assoc, ← NatTrans.comp_app_assoc, ← Functor.map_comp,
    ι_mapBifunctorLeftUnitor_hom_apply F₂ X₂ e₂ τ.p₂₃ τ.h₃ X₃ i₃,
    ι_mapBifunctorRightUnitor_hom_apply F₁ X₂ e₁ τ.p₁₂ τ.h₁ X₁ i₁]
  /-
    case h.h.e_a.e_a
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D : Type u_4
    I₁ : Type u_5
    I₂ : Type u_6
    I₃ : Type u_7
    J : Type u_8
    inst✝¹⁵ : CategoryTheory.Category.{u_12, u_1} C₁
    inst✝¹⁴ : CategoryTheory.Category.{u_11, u_2} C₂
    inst✝¹³ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝¹² : CategoryTheory.Category.{u_9, u_4} D
    inst✝¹¹ : Zero I₂
    inst✝¹⁰ : DecidableEq I₂
    inst✝⁹ : CategoryTheory.Limits.HasInitial C₂
    F₁ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁)
    F₂ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₃)
    G : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₃ D)
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁ G) (Categor …
    X₂ : C₂
    e₁ : CategoryTheory.Iso (F₁.flip.obj X₂) (CategoryTheory.Functor.id C₁)
    e₂ : CategoryTheory.Iso (F₂.obj X₂) (CategoryTheory.Functor.id C₃)
    inst✝⁸ : ∀ (X₁ : C₁), CategoryTheory.Limits.PreservesColimit (CategoryTheory.F …
    inst✝⁷ : ∀ (X₃ : C₃), CategoryTheory.Limits.PreservesColimit (CategoryTheory.F …
    r : Prod I₁ (Prod I₂ I₃) → J
    π : Prod I₁ I₃ → J
    τ : CategoryTheory.GradedObject.TriangleIndexData r π
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝⁶ : (((CategoryTheory.GradedObject.mapBifunctor F₁ I₁ I₂).obj X₁).obj ((C …
    inst✝⁵ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj (CategoryThe …
    inst✝⁴ : (((CategoryTheory.GradedObject.mapBifunctor F₂ I₂ I₃).obj ((CategoryT …
    inst✝³ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj X₁).obj (Cat …
    inst✝² : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁ G τ.ρ₁₂ X₁ ((Ca …
    inst✝¹ : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj G F₂ τ.ρ₂₃ X₁ ((Ca …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj X₁).obj X₃).H …
    triangle : ∀ (X₁ : C₁) (X₃ : C₃), Eq (CategoryTheory.CategoryStruct.comp (((as …
    j : J
    i₁ : I₁
    i₃ : I₃
    hj : Eq (π { fst := i₁, snd := i₃ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((associator.hom.app (X₁ i₁)).app (( …
  -/
  dsimp
  /-
    case h.h.e_a.e_a
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D : Type u_4
    I₁ : Type u_5
    I₂ : Type u_6
    I₃ : Type u_7
    J : Type u_8
    inst✝¹⁵ : CategoryTheory.Category.{u_12, u_1} C₁
    inst✝¹⁴ : CategoryTheory.Category.{u_11, u_2} C₂
    inst✝¹³ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝¹² : CategoryTheory.Category.{u_9, u_4} D
    inst✝¹¹ : Zero I₂
    inst✝¹⁰ : DecidableEq I₂
    inst✝⁹ : CategoryTheory.Limits.HasInitial C₂
    F₁ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁)
    F₂ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₃)
    G : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₃ D)
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁ G) (Categor …
    X₂ : C₂
    e₁ : CategoryTheory.Iso (F₁.flip.obj X₂) (CategoryTheory.Functor.id C₁)
    e₂ : CategoryTheory.Iso (F₂.obj X₂) (CategoryTheory.Functor.id C₃)
    inst✝⁸ : ∀ (X₁ : C₁), CategoryTheory.Limits.PreservesColimit (CategoryTheory.F …
    inst✝⁷ : ∀ (X₃ : C₃), CategoryTheory.Limits.PreservesColimit (CategoryTheory.F …
    r : Prod I₁ (Prod I₂ I₃) → J
    π : Prod I₁ I₃ → J
    τ : CategoryTheory.GradedObject.TriangleIndexData r π
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝⁶ : (((CategoryTheory.GradedObject.mapBifunctor F₁ I₁ I₂).obj X₁).obj ((C …
    inst✝⁵ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj (CategoryThe …
    inst✝⁴ : (((CategoryTheory.GradedObject.mapBifunctor F₂ I₂ I₃).obj ((CategoryT …
    inst✝³ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj X₁).obj (Cat …
    inst✝² : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁ G τ.ρ₁₂ X₁ ((Ca …
    inst✝¹ : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj G F₂ τ.ρ₂₃ X₁ ((Ca …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj X₁).obj X₃).H …
    triangle : ∀ (X₁ : C₁) (X₃ : C₃), Eq (CategoryTheory.CategoryStruct.comp (((as …
    j : J
    i₁ : I₁
    i₃ : I₃
    hj : Eq (π { fst := i₁, snd := i₃ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((associator.hom.app (X₁ i₁)).app (( …
  -/
  simp only [Functor.map_comp, NatTrans.comp_app, ← triangle (X₁ i₁) (X₃ i₃), ← assoc]
  /-
    case h.h.e_a.e_a
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D : Type u_4
    I₁ : Type u_5
    I₂ : Type u_6
    I₃ : Type u_7
    J : Type u_8
    inst✝¹⁵ : CategoryTheory.Category.{u_12, u_1} C₁
    inst✝¹⁴ : CategoryTheory.Category.{u_11, u_2} C₂
    inst✝¹³ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝¹² : CategoryTheory.Category.{u_9, u_4} D
    inst✝¹¹ : Zero I₂
    inst✝¹⁰ : DecidableEq I₂
    inst✝⁹ : CategoryTheory.Limits.HasInitial C₂
    F₁ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁)
    F₂ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₃)
    G : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₃ D)
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁ G) (Categor …
    X₂ : C₂
    e₁ : CategoryTheory.Iso (F₁.flip.obj X₂) (CategoryTheory.Functor.id C₁)
    e₂ : CategoryTheory.Iso (F₂.obj X₂) (CategoryTheory.Functor.id C₃)
    inst✝⁸ : ∀ (X₁ : C₁), CategoryTheory.Limits.PreservesColimit (CategoryTheory.F …
    inst✝⁷ : ∀ (X₃ : C₃), CategoryTheory.Limits.PreservesColimit (CategoryTheory.F …
    r : Prod I₁ (Prod I₂ I₃) → J
    π : Prod I₁ I₃ → J
    τ : CategoryTheory.GradedObject.TriangleIndexData r π
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝⁶ : (((CategoryTheory.GradedObject.mapBifunctor F₁ I₁ I₂).obj X₁).obj ((C …
    inst✝⁵ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj (CategoryThe …
    inst✝⁴ : (((CategoryTheory.GradedObject.mapBifunctor F₂ I₂ I₃).obj ((CategoryT …
    inst✝³ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj X₁).obj (Cat …
    inst✝² : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁ G τ.ρ₁₂ X₁ ((Ca …
    inst✝¹ : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj G F₂ τ.ρ₂₃ X₁ ((Ca …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj X₁).obj X₃).H …
    triangle : ∀ (X₁ : C₁) (X₃ : C₃), Eq (CategoryTheory.CategoryStruct.comp (((as …
    j : J
    i₁ : I₁
    i₃ : I₃
    hj : Eq (π { fst := i₁, snd := i₃ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  congr 2
  /-
    case h.h.e_a.e_a.e_a.e_a
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D : Type u_4
    I₁ : Type u_5
    I₂ : Type u_6
    I₃ : Type u_7
    J : Type u_8
    inst✝¹⁵ : CategoryTheory.Category.{u_12, u_1} C₁
    inst✝¹⁴ : CategoryTheory.Category.{u_11, u_2} C₂
    inst✝¹³ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝¹² : CategoryTheory.Category.{u_9, u_4} D
    inst✝¹¹ : Zero I₂
    inst✝¹⁰ : DecidableEq I₂
    inst✝⁹ : CategoryTheory.Limits.HasInitial C₂
    F₁ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁)
    F₂ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₃)
    G : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₃ D)
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁ G) (Categor …
    X₂ : C₂
    e₁ : CategoryTheory.Iso (F₁.flip.obj X₂) (CategoryTheory.Functor.id C₁)
    e₂ : CategoryTheory.Iso (F₂.obj X₂) (CategoryTheory.Functor.id C₃)
    inst✝⁸ : ∀ (X₁ : C₁), CategoryTheory.Limits.PreservesColimit (CategoryTheory.F …
    inst✝⁷ : ∀ (X₃ : C₃), CategoryTheory.Limits.PreservesColimit (CategoryTheory.F …
    r : Prod I₁ (Prod I₂ I₃) → J
    π : Prod I₁ I₃ → J
    τ : CategoryTheory.GradedObject.TriangleIndexData r π
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝⁶ : (((CategoryTheory.GradedObject.mapBifunctor F₁ I₁ I₂).obj X₁).obj ((C …
    inst✝⁵ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj (CategoryThe …
    inst✝⁴ : (((CategoryTheory.GradedObject.mapBifunctor F₂ I₂ I₃).obj ((CategoryT …
    inst✝³ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj X₁).obj (Cat …
    inst✝² : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁ G τ.ρ₁₂ X₁ ((Ca …
    inst✝¹ : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj G F₂ τ.ρ₂₃ X₁ ((Ca …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj X₁).obj X₃).H …
    triangle : ∀ (X₁ : C₁) (X₃ : C₃), Eq (CategoryTheory.CategoryStruct.comp (((as …
    j : J
    i₁ : I₁
    i₃ : I₃
    hj : Eq (π { fst := i₁, snd := i₃ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((associator.hom.app (X₁ i₁)).app (( …
  -/
  symm
  /-
    case h.h.e_a.e_a.e_a.e_a
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D : Type u_4
    I₁ : Type u_5
    I₂ : Type u_6
    I₃ : Type u_7
    J : Type u_8
    inst✝¹⁵ : CategoryTheory.Category.{u_12, u_1} C₁
    inst✝¹⁴ : CategoryTheory.Category.{u_11, u_2} C₂
    inst✝¹³ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝¹² : CategoryTheory.Category.{u_9, u_4} D
    inst✝¹¹ : Zero I₂
    inst✝¹⁰ : DecidableEq I₂
    inst✝⁹ : CategoryTheory.Limits.HasInitial C₂
    F₁ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁)
    F₂ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₃)
    G : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₃ D)
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁ G) (Categor …
    X₂ : C₂
    e₁ : CategoryTheory.Iso (F₁.flip.obj X₂) (CategoryTheory.Functor.id C₁)
    e₂ : CategoryTheory.Iso (F₂.obj X₂) (CategoryTheory.Functor.id C₃)
    inst✝⁸ : ∀ (X₁ : C₁), CategoryTheory.Limits.PreservesColimit (CategoryTheory.F …
    inst✝⁷ : ∀ (X₃ : C₃), CategoryTheory.Limits.PreservesColimit (CategoryTheory.F …
    r : Prod I₁ (Prod I₂ I₃) → J
    π : Prod I₁ I₃ → J
    τ : CategoryTheory.GradedObject.TriangleIndexData r π
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝⁶ : (((CategoryTheory.GradedObject.mapBifunctor F₁ I₁ I₂).obj X₁).obj ((C …
    inst✝⁵ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj (CategoryThe …
    inst✝⁴ : (((CategoryTheory.GradedObject.mapBifunctor F₂ I₂ I₃).obj ((CategoryT …
    inst✝³ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj X₁).obj (Cat …
    inst✝² : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁ G τ.ρ₁₂ X₁ ((Ca …
    inst✝¹ : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj G F₂ τ.ρ₂₃ X₁ ((Ca …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor G I₁ I₃).obj X₁).obj X₃).H …
    triangle : ∀ (X₁ : C₁) (X₃ : C₃), Eq (CategoryTheory.CategoryStruct.comp (((as …
    j : J
    i₁ : I₁
    i₃ : I₃
    hj : Eq (π { fst := i₁, snd := i₃ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.map ((F₁.obj (X₁ i₁)).map (Catego …
  -/
  apply NatTrans.naturality_app (associator.hom.app (X₁ i₁))
  /-
    🎉 no goals
  -/


