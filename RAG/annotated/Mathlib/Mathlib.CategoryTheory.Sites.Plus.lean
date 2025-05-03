/-- The diagram whose colimit defines the values of `plus`. -/
@[simps]
def diagram (X : C) : (J.Cover X)ᵒᵖ ⥤ D where
  obj S := multiequalizer (S.unop.index P)
  map {S _} f :=
    Multiequalizer.lift _ _ (fun I => Multiequalizer.ι (S.unop.index P) (I.map f.unop))
      (fun I => Multiequalizer.condition (S.unop.index P) (Cover.Relation.mk' (I.r.map f.unop)))


/-- A helper definition used to define the morphisms for `plus`. -/
@[simps]
def diagramPullback {X Y : C} (f : X ⟶ Y) : J.diagram P Y ⟶ (J.pullback f).op ⋙ J.diagram P X where
  app S :=
    Multiequalizer.lift _ _ (fun I => Multiequalizer.ι (S.unop.index P) I.base) fun I =>
      Multiequalizer.condition (S.unop.index P) (Cover.Relation.mk' I.r.base)
                                                                /-
                                                                  C : Type u
                                                                  inst✝² : CategoryTheory.Category.{v, u} C
                                                                  J : CategoryTheory.GrothendieckTopology C
                                                                  D : Type w
                                                                  inst✝¹ : CategoryTheory.Category.{max v u, w} D
                                                                  inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
                                                                  P : CategoryTheory.Functor (Opposite C) D
                                                                  X Y : C
                                                                  f✝ : Quiver.Hom X Y
                                                                  S T : Opposite (J.Cover Y)
                                                                  f : Quiver.Hom S T
                                                                  I : ((Opposite.unop ((J.pullback f✝).op.obj T)).index P).L
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                                                -/
  naturality S T f := Multiequalizer.hom_ext _ _ _ (fun I => by dsimp; simp; rfl)
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- A natural transformation `P ⟶ Q` induces a natural transformation
between diagrams whose colimits define the values of `plus`. -/
@[simps]
def diagramNatTrans {P Q : Cᵒᵖ ⥤ D} (η : P ⟶ Q) (X : C) : J.diagram P X ⟶ J.diagram Q X where
  app W :=
    Multiequalizer.lift _ _ (fun _ => Multiequalizer.ι _ _ ≫ η.app _) (fun i => by
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝¹ : CategoryTheory.Category.{max v u, w} D
        inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
        P✝ P Q : CategoryTheory.Functor (Opposite C) D
        η : Quiver.Hom P Q
        X : C
        W : Opposite (J.Cover X)
        i : ((Opposite.unop W).index Q).R
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x => CategoryTheory.CategoryStr …
      -/
      dsimp only
      erw [Category.assoc, Category.assoc, ← η.naturality, ← η.naturality,
        Multiequalizer.condition_assoc]
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        D : Type w
        inst✝¹ : CategoryTheory.Category.{max v u, w} D
        inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
        P✝ P Q : CategoryTheory.Functor (Opposite C) D
        η : Quiver.Hom P Q
        X : C
        W : Opposite (J.Cover X)
        i : ((Opposite.unop W).index Q).R
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
      -/
      rfl)
      /-
        🎉 no goals
      -/


@[simp]
theorem diagramNatTrans_id (X : C) (P : Cᵒᵖ ⥤ D) :
    J.diagramNatTrans (𝟙 P) X = 𝟙 (J.diagram P X) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝¹ : CategoryTheory.Category.{max v u, w} D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    ⊢ Eq (J.diagramNatTrans (CategoryTheory.CategoryStruct.id P) X) (CategoryTheor …
  -/
  ext : 2
  /-
    case w.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝¹ : CategoryTheory.Category.{max v u, w} D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    x✝ : Opposite (J.Cover X)
    ⊢ Eq ((J.diagramNatTrans (CategoryTheory.CategoryStruct.id P) X).app x✝) ((Cat …
  -/
  refine Multiequalizer.hom_ext _ _ _ (fun i => ?_)
  /-
    case w.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝¹ : CategoryTheory.Category.{max v u, w} D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    x✝ : Opposite (J.Cover X)
    i : ((Opposite.unop x✝).index P).L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((J.diagramNatTrans (CategoryTheory.C …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem diagramNatTrans_zero [Preadditive D] (X : C) (P Q : Cᵒᵖ ⥤ D) :
    J.diagramNatTrans (0 : P ⟶ Q) X = 0 := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : CategoryTheory.Preadditive D
    X : C
    P Q : CategoryTheory.Functor (Opposite C) D
    ⊢ Eq (J.diagramNatTrans 0 X) 0
  -/
  ext : 2
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : CategoryTheory.Preadditive D
    X : C
    P Q : CategoryTheory.Functor (Opposite C) D
    x✝ : Opposite (J.Cover X)
    ⊢ Eq ((J.diagramNatTrans 0 X).app x✝) (CategoryTheory.NatTrans.app 0 x✝)
  -/
  refine Multiequalizer.hom_ext _ _ _ (fun i => ?_)
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : CategoryTheory.Preadditive D
    X : C
    P Q : CategoryTheory.Functor (Opposite C) D
    x✝ : Opposite (J.Cover X)
    i : ((Opposite.unop x✝).index Q).L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((J.diagramNatTrans 0 X).app x✝) (Cat …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem diagramNatTrans_comp {P Q R : Cᵒᵖ ⥤ D} (η : P ⟶ Q) (γ : Q ⟶ R) (X : C) :
    J.diagramNatTrans (η ≫ γ) X = J.diagramNatTrans η X ≫ J.diagramNatTrans γ X := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝¹ : CategoryTheory.Category.{max v u, w} D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    P Q R : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    γ : Quiver.Hom Q R
    X : C
    ⊢ Eq (J.diagramNatTrans (CategoryTheory.CategoryStruct.comp η γ) X) (CategoryT …
  -/
  ext : 2
  /-
    case w.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝¹ : CategoryTheory.Category.{max v u, w} D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    P Q R : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    γ : Quiver.Hom Q R
    X : C
    x✝ : Opposite (J.Cover X)
    ⊢ Eq ((J.diagramNatTrans (CategoryTheory.CategoryStruct.comp η γ) X).app x✝) ( …
  -/
  refine Multiequalizer.hom_ext _ _ _ (fun i => ?_)
  /-
    case w.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝¹ : CategoryTheory.Category.{max v u, w} D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    P Q R : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    γ : Quiver.Hom Q R
    X : C
    x✝ : Opposite (J.Cover X)
    i : ((Opposite.unop x✝).index R).L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((J.diagramNatTrans (CategoryTheory.C …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- `J.diagram P`, as a functor in `P`. -/
@[simps]
def diagramFunctor (X : C) : (Cᵒᵖ ⥤ D) ⥤ (J.Cover X)ᵒᵖ ⥤ D where
  obj P := J.diagram P X
  map η := J.diagramNatTrans η X


/-- The plus construction, associating a presheaf to any presheaf.
See `plusFunctor` below for a functorial version. -/
def plusObj : Cᵒᵖ ⥤ D where
  obj X := colimit (J.diagram P X.unop)
  map f := colimMap (J.diagramPullback P f.unop) ≫ colimit.pre _ _
  map_id := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      ⊢ ∀ (X : Opposite C), Eq ({ obj := fun X => CategoryTheory.Limits.colimit (J.d …
    -/
    intro X
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X : Opposite C
      ⊢ Eq ({ obj := fun X => CategoryTheory.Limits.colimit (J.diagram P (Opposite.u …
    -/
    refine colimit.hom_ext (fun S => ?_)
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X : Opposite C
      S : Opposite (J.Cover (Opposite.unop X))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
    -/
    dsimp
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X : Opposite C
      S : Opposite (J.Cover (Opposite.unop X))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
    -/
    simp only [diagramPullback_app, colimit.ι_pre, ι_colimMap_assoc, Category.comp_id]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X : Opposite C
      S : Opposite (J.Cover (Opposite.unop X))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
    -/
    let e := S.unop.pullbackId
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X : Opposite C
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
    -/
    dsimp only [Functor.op, pullback_obj]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X : Opposite C
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
    -/
    rw [← colimit.w _ e.inv.op, ← Category.assoc]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X : Opposite C
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    convert Category.id_comp (colimit.ι (diagram J P (unop X)) S)
    /-
      case h.e'_2.h.h.e'_6
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X : Opposite C
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      e_1✝ : Eq (Quiver.Hom (CategoryTheory.Limits.multiequalizer ((Opposite.unop S) …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
    -/
    refine Multiequalizer.hom_ext _ _ _ (fun I => ?_)
    /-
      case h.e'_2.h.h.e'_6
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X : Opposite C
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      e_1✝ : Eq (Quiver.Hom (CategoryTheory.Limits.multiequalizer ((Opposite.unop S) …
      I : ((Opposite.unop { unop := Opposite.unop S }).index P).L
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    dsimp
    /-
      case h.e'_2.h.h.e'_6
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X : Opposite C
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      e_1✝ : Eq (Quiver.Hom (CategoryTheory.Limits.multiequalizer ((Opposite.unop S) …
      I : ((Opposite.unop { unop := Opposite.unop S }).index P).L
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [Multiequalizer.lift_ι, Category.id_comp, Category.assoc]
    /-
      case h.e'_2.h.h.e'_6
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X : Opposite C
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      e_1✝ : Eq (Quiver.Hom (CategoryTheory.Limits.multiequalizer ((Opposite.unop S) …
      I : ((Opposite.unop { unop := Opposite.unop S }).index P).L
      ⊢ Eq (CategoryTheory.Limits.Multiequalizer.ι ((Opposite.unop S).index P) (Cate …
    -/
    dsimp [Cover.Arrow.map, Cover.Arrow.base]
    /-
      case h.e'_2.h.h.e'_6
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X : Opposite C
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      e_1✝ : Eq (Quiver.Hom (CategoryTheory.Limits.multiequalizer ((Opposite.unop S) …
      I : ((Opposite.unop { unop := Opposite.unop S }).index P).L
      ⊢ Eq (CategoryTheory.Limits.Multiequalizer.ι ((Opposite.unop S).index P) { Y : …
    -/
    cases I
    /-
      case h.e'_2.h.h.e'_6.mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X : Opposite C
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      e_1✝ : Eq (Quiver.Hom (CategoryTheory.Limits.multiequalizer ((Opposite.unop S) …
      Y✝ : C
      f✝ : Quiver.Hom Y✝ (Opposite.unop X)
      hf✝ : (↑(Opposite.unop { unop := Opposite.unop S })).arrows f✝
      ⊢ Eq (CategoryTheory.Limits.Multiequalizer.ι ((Opposite.unop S).index P) { Y : …
    -/
    congr
    /-
      case h.e'_2.h.h.e'_6.mk.h.e_5.h.e_f
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X : Opposite C
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      e_1✝ : Eq (Quiver.Hom (CategoryTheory.Limits.multiequalizer ((Opposite.unop S) …
      Y✝ : C
      f✝ : Quiver.Hom Y✝ (Opposite.unop X)
      hf✝ : (↑(Opposite.unop { unop := Opposite.unop S })).arrows f✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { Y := Y✝, f := f✝, hf := hf✝ }.f (Ca …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_comp := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      ⊢ ∀ {X Y Z : Opposite C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ obj  …
    -/
    intro X Y Z f g
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun X => CategoryTheory.Limits.colimit (J.diagram P (Opposite.u …
    -/
    refine colimit.hom_ext (fun S => ?_)
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      S : Opposite (J.Cover (Opposite.unop X))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
    -/
    dsimp
    simp only [diagramPullback_app, colimit.ι_pre_assoc, colimit.ι_pre, ι_colimMap_assoc,
      Category.assoc]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      S : Opposite (J.Cover (Opposite.unop X))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
    -/
    let e := S.unop.pullbackComp g.unop f.unop
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
    -/
    dsimp only [Functor.op, pullback_obj]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
    -/
    rw [← colimit.w _ e.inv.op, ← Category.assoc, ← Category.assoc]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    congr 1
    /-
      case e_a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
    -/
    refine Multiequalizer.hom_ext _ _ _ (fun I => ?_)
    /-
      case e_a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      I : ((Opposite.unop { unop := ((Opposite.unop S).pullback f.unop).pullback g.u …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    dsimp
    /-
      case e_a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      I : ((Opposite.unop { unop := ((Opposite.unop S).pullback f.unop).pullback g.u …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [Multiequalizer.lift_ι, Category.assoc]
    /-
      case e_a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      I : ((Opposite.unop { unop := ((Opposite.unop S).pullback f.unop).pullback g.u …
      ⊢ Eq (CategoryTheory.Limits.Multiequalizer.ι ((Opposite.unop S).index P) (Cate …
    -/
    cases I
    /-
      case e_a.mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      Y✝ : C
      f✝ : Quiver.Hom Y✝ (Opposite.unop Z)
      hf✝ : (↑(Opposite.unop { unop := ((Opposite.unop S).pullback f.unop).pullback  …
      ⊢ Eq (CategoryTheory.Limits.Multiequalizer.ι ((Opposite.unop S).index P) ({ Y  …
    -/
    dsimp only [Cover.Arrow.base, Cover.Arrow.map]
    /-
      case e_a.mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      Y✝ : C
      f✝ : Quiver.Hom Y✝ (Opposite.unop Z)
      hf✝ : (↑(Opposite.unop { unop := ((Opposite.unop S).pullback f.unop).pullback  …
      ⊢ Eq (CategoryTheory.Limits.Multiequalizer.ι ((Opposite.unop S).index P) { Y : …
    -/
    congr 2
    /-
      case e_a.mk.h.e_5.h.e_f
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y Z : Opposite C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      S : Opposite (J.Cover (Opposite.unop X))
      e : CategoryTheory.Iso ((Opposite.unop S).pullback (CategoryTheory.CategoryStr …
      Y✝ : C
      f✝ : Quiver.Hom Y✝ (Opposite.unop Z)
      hf✝ : (↑(Opposite.unop { unop := ((Opposite.unop S).pullback f.unop).pullback  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f✝ (CategoryTheory.CategoryStruct.com …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- An auxiliary definition used in `plus` below. -/
def plusMap {P Q : Cᵒᵖ ⥤ D} (η : P ⟶ Q) : J.plusObj P ⟶ J.plusObj Q where
  app X := colimMap (J.diagramNatTrans η X.unop)
  naturality := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P✝ : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      P Q : CategoryTheory.Functor (Opposite C) D
      η : Quiver.Hom P Q
      ⊢ ∀ ⦃X Y : Opposite C⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
    -/
    intro X Y f
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P✝ : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      P Q : CategoryTheory.Functor (Opposite C) D
      η : Quiver.Hom P Q
      X Y : Opposite C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((J.plusObj P).map f) ((fun X => Cate …
    -/
    dsimp [plusObj]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P✝ : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      P Q : CategoryTheory.Functor (Opposite C) D
      η : Quiver.Hom P Q
      X Y : Opposite C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    ext
    simp only [diagramPullback_app, ι_colimMap, colimit.ι_pre_assoc, colimit.ι_pre,
      ι_colimMap_assoc, Category.assoc]
    /-
      case w
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P✝ : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      P Q : CategoryTheory.Functor (Opposite C) D
      η : Quiver.Hom P Q
      X Y : Opposite C
      f : Quiver.Hom X Y
      j✝ : Opposite (J.Cover (Opposite.unop X))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
    -/
    simp_rw [← Category.assoc]
    /-
      case w
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P✝ : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      P Q : CategoryTheory.Functor (Opposite C) D
      η : Quiver.Hom P Q
      X Y : Opposite C
      f : Quiver.Hom X Y
      j✝ : Opposite (J.Cover (Opposite.unop X))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    congr 1
    /-
      case w.e_a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P✝ : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      P Q : CategoryTheory.Functor (Opposite C) D
      η : Quiver.Hom P Q
      X Y : Opposite C
      f : Quiver.Hom X Y
      j✝ : Opposite (J.Cover (Opposite.unop X))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
    -/
    exact Multiequalizer.hom_ext _ _ _ (fun I => by dsimp; simp)
    /-
      🎉 no goals
    -/


@[simp]
theorem plusMap_id (P : Cᵒᵖ ⥤ D) : J.plusMap (𝟙 P) = 𝟙 _ := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P : CategoryTheory.Functor (Opposite C) D
    ⊢ Eq (J.plusMap (CategoryTheory.CategoryStruct.id P)) (CategoryTheory.Category …
  -/
  ext : 2
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P : CategoryTheory.Functor (Opposite C) D
    x✝ : Opposite C
    ⊢ Eq ((J.plusMap (CategoryTheory.CategoryStruct.id P)).app x✝) ((CategoryTheor …
  -/
  dsimp only [plusMap, plusObj]
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P : CategoryTheory.Functor (Opposite C) D
    x✝ : Opposite C
    ⊢ Eq (CategoryTheory.Limits.colimMap (J.diagramNatTrans (CategoryTheory.Catego …
  -/
  rw [J.diagramNatTrans_id, NatTrans.id_app]
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P : CategoryTheory.Functor (Opposite C) D
    x✝ : Opposite C
    ⊢ Eq (CategoryTheory.Limits.colimMap (CategoryTheory.CategoryStruct.id (J.diag …
  -/
  ext
  /-
    case w.h.w
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P : CategoryTheory.Functor (Opposite C) D
    x✝ : Opposite C
    j✝ : Opposite (J.Cover (Opposite.unop x✝))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
  -/
  dsimp
  /-
    case w.h.w
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P : CategoryTheory.Functor (Opposite C) D
    x✝ : Opposite C
    j✝ : Opposite (J.Cover (Opposite.unop x✝))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem plusMap_zero [Preadditive D] (P Q : Cᵒᵖ ⥤ D) : J.plusMap (0 : P ⟶ Q) = 0 := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝³ : CategoryTheory.Category.{max v u, w} D
    inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : CategoryTheory.Preadditive D
    P Q : CategoryTheory.Functor (Opposite C) D
    ⊢ Eq (J.plusMap 0) 0
  -/
  ext : 2
  /-
    case w.h
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝³ : CategoryTheory.Category.{max v u, w} D
    inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : CategoryTheory.Preadditive D
    P Q : CategoryTheory.Functor (Opposite C) D
    x✝ : Opposite C
    ⊢ Eq ((J.plusMap 0).app x✝) (CategoryTheory.NatTrans.app 0 x✝)
  -/
  refine colimit.hom_ext (fun S => ?_)
  /-
    case w.h
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝³ : CategoryTheory.Category.{max v u, w} D
    inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : CategoryTheory.Preadditive D
    P Q : CategoryTheory.Functor (Opposite C) D
    x✝ : Opposite C
    S : Opposite (J.Cover (Opposite.unop x✝))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
  -/
  erw [comp_zero, colimit.ι_map, J.diagramNatTrans_zero, zero_comp]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem plusMap_comp {P Q R : Cᵒᵖ ⥤ D} (η : P ⟶ Q) (γ : Q ⟶ R) :
    J.plusMap (η ≫ γ) = J.plusMap η ≫ J.plusMap γ := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q R : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    γ : Quiver.Hom Q R
    ⊢ Eq (J.plusMap (CategoryTheory.CategoryStruct.comp η γ)) (CategoryTheory.Cate …
  -/
  ext : 2
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q R : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    γ : Quiver.Hom Q R
    x✝ : Opposite C
    ⊢ Eq ((J.plusMap (CategoryTheory.CategoryStruct.comp η γ)).app x✝) ((CategoryT …
  -/
  refine colimit.hom_ext (fun S => ?_)
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q R : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    γ : Quiver.Hom Q R
    x✝ : Opposite C
    S : Opposite (J.Cover (Opposite.unop x✝))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
  -/
  simp [plusMap, J.diagramNatTrans_comp]
  /-
    🎉 no goals
  -/


/-- The plus construction, a functor sending `P` to `J.plusObj P`. -/
@[simps]
def plusFunctor : (Cᵒᵖ ⥤ D) ⥤ Cᵒᵖ ⥤ D where
  obj P := J.plusObj P
  map η := J.plusMap η


/-- The canonical map from `P` to `J.plusObj P`.
See `toPlusNatTrans` for a functorial version. -/
def toPlus : P ⟶ J.plusObj P where
  app X := Cover.toMultiequalizer (⊤ : J.Cover X.unop) P ≫ colimit.ι (J.diagram P X.unop) (op ⊤)
  naturality := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      ⊢ ∀ ⦃X Y : Opposite C⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
    -/
    intro X Y f
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y : Opposite C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map f) ((fun X => CategoryTheory.C …
    -/
    dsimp [plusObj]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y : Opposite C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map f) (CategoryTheory.CategoryStr …
    -/
    delta Cover.toMultiequalizer
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y : Opposite C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map f) (CategoryTheory.CategoryStr …
    -/
    simp only [diagramPullback_app, colimit.ι_pre, ι_colimMap_assoc, Category.assoc]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y : Opposite C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map f) (CategoryTheory.CategoryStr …
    -/
    dsimp only [Functor.op, unop_op]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y : Opposite C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map f) (CategoryTheory.CategoryStr …
    -/
    let e : (J.pullback f.unop).obj ⊤ ⟶ ⊤ := homOfLE (OrderTop.le_top _)
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y : Opposite C
      f : Quiver.Hom X Y
      e : Quiver.Hom ((J.pullback f.unop).obj Top.top) Top.top := CategoryTheory.hom …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map f) (CategoryTheory.CategoryStr …
    -/
    rw [← colimit.w _ e.op, ← Category.assoc, ← Category.assoc, ← Category.assoc]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y : Opposite C
      f : Quiver.Hom X Y
      e : Quiver.Hom ((J.pullback f.unop).obj Top.top) Top.top := CategoryTheory.hom …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    congr 1
    /-
      case e_a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y : Opposite C
      f : Quiver.Hom X Y
      e : Quiver.Hom ((J.pullback f.unop).obj Top.top) Top.top := CategoryTheory.hom …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    refine Multiequalizer.hom_ext _ _ _ (fun I => ?_)
    /-
      case e_a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y : Opposite C
      f : Quiver.Hom X Y
      e : Quiver.Hom ((J.pullback f.unop).obj Top.top) Top.top := CategoryTheory.hom …
      I : ((Opposite.unop { unop := (J.pullback f.unop).obj Top.top }).index P).L
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [Multiequalizer.lift_ι, Category.assoc]
    /-
      case e_a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y : Opposite C
      f : Quiver.Hom X Y
      e : Quiver.Hom ((J.pullback f.unop).obj Top.top) Top.top := CategoryTheory.hom …
      I : ((Opposite.unop { unop := (J.pullback f.unop).obj Top.top }).index P).L
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map f) (CategoryTheory.CategoryStr …
    -/
    dsimp [Cover.Arrow.base]
    /-
      case e_a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝² : CategoryTheory.Category.{max v u, w} D
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
      X Y : Opposite C
      f : Quiver.Hom X Y
      e : Quiver.Hom ((J.pullback f.unop).obj Top.top) Top.top := CategoryTheory.hom …
      I : ((Opposite.unop { unop := (J.pullback f.unop).obj Top.top }).index P).L
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map f) (CategoryTheory.CategoryStr …
    -/
    simp
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
theorem toPlus_naturality {P Q : Cᵒᵖ ⥤ D} (η : P ⟶ Q) :
    η ≫ J.toPlus Q = J.toPlus _ ≫ J.plusMap η := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp η (J.toPlus Q)) (CategoryTheory.Categ …
  -/
  ext
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    x✝ : Opposite C
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp η (J.toPlus Q)).app x✝) ((CategoryTh …
  -/
  dsimp [toPlus, plusMap]
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    x✝ : Opposite C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (η.app x✝) (CategoryTheory.CategorySt …
  -/
  delta Cover.toMultiequalizer
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    x✝ : Opposite C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (η.app x✝) (CategoryTheory.CategorySt …
  -/
  simp only [ι_colimMap, Category.assoc]
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    x✝ : Opposite C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (η.app x✝) (CategoryTheory.CategorySt …
  -/
  simp_rw [← Category.assoc]
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    x✝ : Opposite C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  congr 1
  /-
    case w.h.e_a
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    x✝ : Opposite C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (η.app x✝) (CategoryTheory.Limits.Mul …
  -/
  exact Multiequalizer.hom_ext _ _ _ (fun I => by dsimp; simp)
  /-
    🎉 no goals
  -/


/-- The natural transformation from the identity functor to `plus`. -/
@[simps]
def toPlusNatTrans : 𝟭 (Cᵒᵖ ⥤ D) ⟶ J.plusFunctor D where
  app P := J.toPlus P


/-- `(P ⟶ P⁺)⁺ = P⁺ ⟶ P⁺⁺` -/
@[simp]
theorem plusMap_toPlus : J.plusMap (J.toPlus P) = J.toPlus (J.plusObj P) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    ⊢ Eq (J.plusMap (J.toPlus P)) (J.toPlus (J.plusObj P))
  -/
  ext X : 2
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    X : Opposite C
    ⊢ Eq ((J.plusMap (J.toPlus P)).app X) ((J.toPlus (J.plusObj P)).app X)
  -/
  refine colimit.hom_ext (fun S => ?_)
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    X : Opposite C
    S : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
  -/
  dsimp only [plusMap, toPlus]
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    X : Opposite C
    S : Opposite (J.Cover (Opposite.unop X))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
  -/
  let e : S.unop ⟶ ⊤ := homOfLE (OrderTop.le_top _)
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    X : Opposite C
    S : Opposite (J.Cover (Opposite.unop X))
    e : Quiver.Hom (Opposite.unop S) Top.top := CategoryTheory.homOfLE ⋯
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
  -/
  rw [ι_colimMap, ← colimit.w _ e.op, ← Category.assoc, ← Category.assoc]
  /-
    case w.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    X : Opposite C
    S : Opposite (J.Cover (Opposite.unop X))
    e : Quiver.Hom (Opposite.unop S) Top.top := CategoryTheory.homOfLE ⋯
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((J.diagramNatTrans { app := fun X => …
  -/
  congr 1
  /-
    case w.h.e_a
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    X : Opposite C
    S : Opposite (J.Cover (Opposite.unop X))
    e : Quiver.Hom (Opposite.unop S) Top.top := CategoryTheory.homOfLE ⋯
    ⊢ Eq ((J.diagramNatTrans { app := fun X => CategoryTheory.CategoryStruct.comp  …
  -/
  refine Multiequalizer.hom_ext _ _ _ (fun I => ?_)
  /-
    case w.h.e_a
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    X : Opposite C
    S : Opposite (J.Cover (Opposite.unop X))
    e : Quiver.Hom (Opposite.unop S) Top.top := CategoryTheory.homOfLE ⋯
    I : ((Opposite.unop S).index (J.plusObj P)).L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((J.diagramNatTrans { app := fun X => …
  -/
  erw [Multiequalizer.lift_ι]
  simp only [unop_op, op_unop, diagram_map, Category.assoc, limit.lift_π,
    Multifork.ofι_π_app]
  /-
    case w.h.e_a
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    X : Opposite C
    S : Opposite (J.Cover (Opposite.unop X))
    e : Quiver.Hom (Opposite.unop S) Top.top := CategoryTheory.homOfLE ⋯
    I : ((Opposite.unop S).index (J.plusObj P)).L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalizer …
  -/
  let ee : (J.pullback (I.map e).f).obj S.unop ⟶ ⊤ := homOfLE (OrderTop.le_top _)
  erw [← colimit.w _ ee.op, ι_colimMap_assoc, colimit.ι_pre, diagramPullback_app,
    ← Category.assoc, ← Category.assoc]
  /-
    case w.h.e_a
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    X : Opposite C
    S : Opposite (J.Cover (Opposite.unop X))
    e : Quiver.Hom (Opposite.unop S) Top.top := CategoryTheory.homOfLE ⋯
    I : ((Opposite.unop S).index (J.plusObj P)).L
    ee : Quiver.Hom ((J.pullback (CategoryTheory.GrothendieckTopology.Cover.Arrow. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  congr 1
  /-
    case w.h.e_a.e_a
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    X : Opposite C
    S : Opposite (J.Cover (Opposite.unop X))
    e : Quiver.Hom (Opposite.unop S) Top.top := CategoryTheory.homOfLE ⋯
    I : ((Opposite.unop S).index (J.plusObj P)).L
    ee : Quiver.Hom ((J.pullback (CategoryTheory.GrothendieckTopology.Cover.Arrow. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  refine Multiequalizer.hom_ext _ _ _ (fun II => ?_)
  convert Multiequalizer.condition (S.unop.index P)
    (Cover.Relation.mk I II.base { g₁ := II.f, g₂ := 𝟙 _ }) using 1
  /-
    case h.e'_2.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    X : Opposite C
    S : Opposite (J.Cover (Opposite.unop X))
    e : Quiver.Hom (Opposite.unop S) Top.top := CategoryTheory.homOfLE ⋯
    I : ((Opposite.unop S).index (J.plusObj P)).L
    ee : Quiver.Hom ((J.pullback (CategoryTheory.GrothendieckTopology.Cover.Arrow. …
    II : ((Opposite.unop { unop := (J.pullback (CategoryTheory.GrothendieckTopolog …
    e_1✝ : Eq (Quiver.Hom ((J.diagram P (Opposite.unop X)).obj S) (((Opposite.unop …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  all_goals dsimp; simp
  /-
    🎉 no goals
  -/


theorem isIso_toPlus_of_isSheaf (hP : Presheaf.IsSheaf J P) : IsIso (J.toPlus P) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    hP : CategoryTheory.Presheaf.IsSheaf J P
    ⊢ CategoryTheory.IsIso (J.toPlus P)
  -/
  rw [Presheaf.isSheaf_iff_multiequalizer] at hP
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    hP : ∀ (X : C) (S : J.Cover X), CategoryTheory.IsIso (S.toMultiequalizer P)
    ⊢ CategoryTheory.IsIso (J.toPlus P)
  -/
  suffices ∀ X, IsIso ((J.toPlus P).app X) from NatIso.isIso_of_isIso_app _
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    hP : ∀ (X : C) (S : J.Cover X), CategoryTheory.IsIso (S.toMultiequalizer P)
    ⊢ ∀ (X : Opposite C), CategoryTheory.IsIso ((J.toPlus P).app X)
  -/
  intro X
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    hP : ∀ (X : C) (S : J.Cover X), CategoryTheory.IsIso (S.toMultiequalizer P)
    X : Opposite C
    ⊢ CategoryTheory.IsIso ((J.toPlus P).app X)
  -/
  refine IsIso.comp_isIso' inferInstance ?_
  suffices ∀ (S T : (J.Cover X.unop)ᵒᵖ) (f : S ⟶ T), IsIso ((J.diagram P X.unop).map f) from
    isIso_ι_of_isInitial (initialOpOfTerminal isTerminalTop) _
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    hP : ∀ (X : C) (S : J.Cover X), CategoryTheory.IsIso (S.toMultiequalizer P)
    X : Opposite C
    ⊢ ∀ (S T : Opposite (J.Cover (Opposite.unop X))) (f : Quiver.Hom S T), Categor …
  -/
  intro S T e
  have : S.unop.toMultiequalizer P ≫ (J.diagram P X.unop).map e = T.unop.toMultiequalizer P :=
    Multiequalizer.hom_ext _ _ _ (fun II => by dsimp; simp)
  have :
    (J.diagram P X.unop).map e = inv (S.unop.toMultiequalizer P) ≫ T.unop.toMultiequalizer P := by
    simp [← this]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    hP : ∀ (X : C) (S : J.Cover X), CategoryTheory.IsIso (S.toMultiequalizer P)
    X : Opposite C
    S T : Opposite (J.Cover (Opposite.unop X))
    e : Quiver.Hom S T
    this✝ : Eq (CategoryTheory.CategoryStruct.comp ((Opposite.unop S).toMultiequal …
    this : Eq ((J.diagram P (Opposite.unop X)).map e) (CategoryTheory.CategoryStru …
    ⊢ CategoryTheory.IsIso ((J.diagram P (Opposite.unop X)).map e)
  -/
  rw [this]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    hP : ∀ (X : C) (S : J.Cover X), CategoryTheory.IsIso (S.toMultiequalizer P)
    X : Opposite C
    S T : Opposite (J.Cover (Opposite.unop X))
    e : Quiver.Hom S T
    this✝ : Eq (CategoryTheory.CategoryStruct.comp ((Opposite.unop S).toMultiequal …
    this : Eq ((J.diagram P (Opposite.unop X)).map e) (CategoryTheory.CategoryStru …
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The natural isomorphism between `P` and `P⁺` when `P` is a sheaf. -/
def isoToPlus (hP : Presheaf.IsSheaf J P) : P ≅ J.plusObj P :=
  letI := isIso_toPlus_of_isSheaf J P hP
  asIso (J.toPlus P)


@[simp]
theorem isoToPlus_hom (hP : Presheaf.IsSheaf J P) : (J.isoToPlus P hP).hom = J.toPlus P :=
  rfl


/-- Lift a morphism `P ⟶ Q` to `P⁺ ⟶ Q` when `Q` is a sheaf. -/
def plusLift {P Q : Cᵒᵖ ⥤ D} (η : P ⟶ Q) (hQ : Presheaf.IsSheaf J Q) : J.plusObj P ⟶ Q :=
  J.plusMap η ≫ (J.isoToPlus Q hQ).inv


@[reassoc (attr := simp)]
theorem toPlus_plusLift {P Q : Cᵒᵖ ⥤ D} (η : P ⟶ Q) (hQ : Presheaf.IsSheaf J Q) :
    J.toPlus P ≫ J.plusLift η hQ = η := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    hQ : CategoryTheory.Presheaf.IsSheaf J Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toPlus P) (J.plusLift η hQ)) η
  -/
  dsimp [plusLift]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    hQ : CategoryTheory.Presheaf.IsSheaf J Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toPlus P) (CategoryTheory.Category …
  -/
  rw [← Category.assoc]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    hQ : CategoryTheory.Presheaf.IsSheaf J Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [Iso.comp_inv_eq]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    hQ : CategoryTheory.Presheaf.IsSheaf J Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toPlus P) (J.plusMap η)) (Category …
  -/
  dsimp only [isoToPlus, asIso]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    hQ : CategoryTheory.Presheaf.IsSheaf J Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toPlus P) (J.plusMap η)) (Category …
  -/
  rw [toPlus_naturality]
  /-
    🎉 no goals
  -/


theorem plusLift_unique {P Q : Cᵒᵖ ⥤ D} (η : P ⟶ Q) (hQ : Presheaf.IsSheaf J Q)
    (γ : J.plusObj P ⟶ Q) (hγ : J.toPlus P ≫ γ = η) : γ = J.plusLift η hQ := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    hQ : CategoryTheory.Presheaf.IsSheaf J Q
    γ : Quiver.Hom (J.plusObj P) Q
    hγ : Eq (CategoryTheory.CategoryStruct.comp (J.toPlus P) γ) η
    ⊢ Eq γ (J.plusLift η hQ)
  -/
  dsimp only [plusLift]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    hQ : CategoryTheory.Presheaf.IsSheaf J Q
    γ : Quiver.Hom (J.plusObj P) Q
    hγ : Eq (CategoryTheory.CategoryStruct.comp (J.toPlus P) γ) η
    ⊢ Eq γ (CategoryTheory.CategoryStruct.comp (J.plusMap η) (J.isoToPlus Q hQ).inv)
  -/
  rw [Iso.eq_comp_inv, ← hγ, plusMap_comp]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    hQ : CategoryTheory.Presheaf.IsSheaf J Q
    γ : Quiver.Hom (J.plusObj P) Q
    hγ : Eq (CategoryTheory.CategoryStruct.comp (J.toPlus P) γ) η
    ⊢ Eq (CategoryTheory.CategoryStruct.comp γ (J.isoToPlus Q hQ).hom) (CategoryTh …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem plus_hom_ext {P Q : Cᵒᵖ ⥤ D} (η γ : J.plusObj P ⟶ Q) (hQ : Presheaf.IsSheaf J Q)
    (h : J.toPlus P ≫ η = J.toPlus P ≫ γ) : η = γ := by
  have : γ = J.plusLift (J.toPlus P ≫ γ) hQ := by
    apply plusLift_unique
    rfl
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η γ : Quiver.Hom (J.plusObj P) Q
    hQ : CategoryTheory.Presheaf.IsSheaf J Q
    h : Eq (CategoryTheory.CategoryStruct.comp (J.toPlus P) η) (CategoryTheory.Cat …
    this : Eq γ (J.plusLift (CategoryTheory.CategoryStruct.comp (J.toPlus P) γ) hQ)
    ⊢ Eq η γ
  -/
  rw [this]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η γ : Quiver.Hom (J.plusObj P) Q
    hQ : CategoryTheory.Presheaf.IsSheaf J Q
    h : Eq (CategoryTheory.CategoryStruct.comp (J.toPlus P) η) (CategoryTheory.Cat …
    this : Eq γ (J.plusLift (CategoryTheory.CategoryStruct.comp (J.toPlus P) γ) hQ)
    ⊢ Eq η (J.plusLift (CategoryTheory.CategoryStruct.comp (J.toPlus P) γ) hQ)
  -/
  apply plusLift_unique
  /-
    case hγ
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η γ : Quiver.Hom (J.plusObj P) Q
    hQ : CategoryTheory.Presheaf.IsSheaf J Q
    h : Eq (CategoryTheory.CategoryStruct.comp (J.toPlus P) η) (CategoryTheory.Cat …
    this : Eq γ (J.plusLift (CategoryTheory.CategoryStruct.comp (J.toPlus P) γ) hQ)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toPlus P) η) (CategoryTheory.Categ …
  -/
  exact h
  /-
    🎉 no goals
  -/


@[simp]
theorem isoToPlus_inv (hP : Presheaf.IsSheaf J P) :
    (J.isoToPlus P hP).inv = J.plusLift (𝟙 _) hP := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    hP : CategoryTheory.Presheaf.IsSheaf J P
    ⊢ Eq (J.isoToPlus P hP).inv (J.plusLift (CategoryTheory.CategoryStruct.id P) hP)
  -/
  apply J.plusLift_unique
  /-
    case hγ
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    hP : CategoryTheory.Presheaf.IsSheaf J P
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toPlus P) (J.isoToPlus P hP).inv)  …
  -/
  rw [Iso.comp_inv_eq, Category.id_comp]
  /-
    case hγ
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    P : CategoryTheory.Functor (Opposite C) D
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    hP : CategoryTheory.Presheaf.IsSheaf J P
    ⊢ Eq (J.toPlus P) (J.isoToPlus P hP).hom
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem plusMap_plusLift {P Q R : Cᵒᵖ ⥤ D} (η : P ⟶ Q) (γ : Q ⟶ R) (hR : Presheaf.IsSheaf J R) :
    J.plusMap η ≫ J.plusLift γ hR = J.plusLift (η ≫ γ) hR := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q R : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    γ : Quiver.Hom Q R
    hR : CategoryTheory.Presheaf.IsSheaf J R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.plusMap η) (J.plusLift γ hR)) (J.p …
  -/
  apply J.plusLift_unique
  /-
    case hγ
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q R : CategoryTheory.Functor (Opposite C) D
    η : Quiver.Hom P Q
    γ : Quiver.Hom Q R
    hR : CategoryTheory.Presheaf.IsSheaf J R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toPlus P) (CategoryTheory.Category …
  -/
  rw [← Category.assoc, ← J.toPlus_naturality, Category.assoc, J.toPlus_plusLift]
  /-
    🎉 no goals
  -/


instance plusFunctor_preservesZeroMorphisms [Preadditive D] :
    (plusFunctor J D).PreservesZeroMorphisms where
  map_zero F G := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝³ : CategoryTheory.Category.{max v u, w} D
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝ : CategoryTheory.Preadditive D
      F G : CategoryTheory.Functor (Opposite C) D
      ⊢ Eq ((J.plusFunctor D).map 0) 0
    -/
    ext
    /-
      case w.h
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝³ : CategoryTheory.Category.{max v u, w} D
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝ : CategoryTheory.Preadditive D
      F G : CategoryTheory.Functor (Opposite C) D
      x✝ : Opposite C
      ⊢ Eq (((J.plusFunctor D).map 0).app x✝) (CategoryTheory.NatTrans.app 0 x✝)
    -/
    dsimp
    /-
      case w.h
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝³ : CategoryTheory.Category.{max v u, w} D
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      P : CategoryTheory.Functor (Opposite C) D
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝ : CategoryTheory.Preadditive D
      F G : CategoryTheory.Functor (Opposite C) D
      x✝ : Opposite C
      ⊢ Eq ((J.plusMap 0).app x✝) 0
    -/
    rw [J.plusMap_zero, NatTrans.app_zero]
    /-
      🎉 no goals
    -/


