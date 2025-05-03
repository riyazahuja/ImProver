/-- A concrete version of the multiequalizer, to be used below. -/
def Meq {X : C} (P : Cᵒᵖ ⥤ D) (S : J.Cover X) :=
  { x : ∀ I : S.Arrow, P.obj (op I.Y) //
    ∀ I : S.Relation, P.map I.r.g₁.op (x I.fst) = P.map I.r.g₂.op (x I.snd) }


instance {X} (P : Cᵒᵖ ⥤ D) (S : J.Cover X) :
    CoeFun (Meq P S) fun _ => ∀ I : S.Arrow, P.obj (op I.Y) :=
  ⟨fun x => x.1⟩


lemma congr_apply {X} {P : Cᵒᵖ ⥤ D} {S : J.Cover X} (x : Meq P S) {Y}
    {f g : Y ⟶ X} (h : f = g) (hf : S f) :
                               /-
                                 C : Type u
                                 inst✝² : CategoryTheory.Category.{v, u} C
                                 J : CategoryTheory.GrothendieckTopology C
                                 D : Type w
                                 inst✝¹ : CategoryTheory.Category.{max v u, w} D
                                 inst✝ : CategoryTheory.ConcreteCategory D
                                 X : C
                                 P : CategoryTheory.Functor (Opposite C) D
                                 S : J.Cover X
                                 x : CategoryTheory.Meq P S
                                 Y : C
                                 f g : Quiver.Hom Y X
                                 h : Eq f g
                                 hf : (↑S).arrows f
                                 ⊢ (↑S).arrows g
                               -/
    x ⟨_, _, hf⟩ = x ⟨_, g, by simpa only [← h] using hf⟩ := by
                               /-
                                 🎉 no goals
                               -/
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝¹ : CategoryTheory.Category.{max v u, w} D
    inst✝ : CategoryTheory.ConcreteCategory D
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    Y : C
    f g : Quiver.Hom Y X
    h : Eq f g
    hf : (↑S).arrows f
    ⊢ Eq (↑x { Y := Y, f := f, hf := hf }) (↑x { Y := Y, f := g, hf := ⋯ })
  -/
  subst h
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝¹ : CategoryTheory.Category.{max v u, w} D
    inst✝ : CategoryTheory.ConcreteCategory D
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    Y : C
    f : Quiver.Hom Y X
    hf : (↑S).arrows f
    ⊢ Eq (↑x { Y := Y, f := f, hf := hf }) (↑x { Y := Y, f := f, hf := ⋯ })
  -/
  rfl
  /-
    🎉 no goals
  -/


@[ext]
theorem ext {X} {P : Cᵒᵖ ⥤ D} {S : J.Cover X} (x y : Meq P S) (h : ∀ I : S.Arrow, x I = y I) :
    x = y :=
  Subtype.ext <| funext <| h


theorem condition {X} {P : Cᵒᵖ ⥤ D} {S : J.Cover X} (x : Meq P S) (I : S.Relation) :
    P.map I.r.g₁.op (x ((S.index P).fstTo I)) = P.map I.r.g₂.op (x ((S.index P).sndTo I)) :=
  x.2 _


/-- Refine a term of `Meq P T` with respect to a refinement `S ⟶ T` of covers. -/
def refine {X : C} {P : Cᵒᵖ ⥤ D} {S T : J.Cover X} (x : Meq P T) (e : S ⟶ T) : Meq P S :=
  ⟨fun I => x ⟨I.Y, I.f, (leOfHom e) _ I.hf⟩, fun I =>
    x.condition (GrothendieckTopology.Cover.Relation.mk' (I.r.map e))⟩


@[simp]
theorem refine_apply {X : C} {P : Cᵒᵖ ⥤ D} {S T : J.Cover X} (x : Meq P T) (e : S ⟶ T)
    (I : S.Arrow) : x.refine e I = x ⟨I.Y, I.f, (leOfHom e) _ I.hf⟩ :=
  rfl


/-- Pull back a term of `Meq P S` with respect to a morphism `f : Y ⟶ X` in `C`. -/
def pullback {Y X : C} {P : Cᵒᵖ ⥤ D} {S : J.Cover X} (x : Meq P S) (f : Y ⟶ X) :
    Meq P ((J.pullback f).obj S) :=
  ⟨fun I => x ⟨_, I.f ≫ f, I.hf⟩, fun I =>
    x.condition (GrothendieckTopology.Cover.Relation.mk' I.r.base)⟩


@[simp]
theorem pullback_apply {Y X : C} {P : Cᵒᵖ ⥤ D} {S : J.Cover X} (x : Meq P S) (f : Y ⟶ X)
    (I : ((J.pullback f).obj S).Arrow) : x.pullback f I = x ⟨_, I.f ≫ f, I.hf⟩ :=
  rfl


@[simp]
theorem pullback_refine {Y X : C} {P : Cᵒᵖ ⥤ D} {S T : J.Cover X} (h : S ⟶ T) (f : Y ⟶ X)
    (x : Meq P T) : (x.pullback f).refine ((J.pullback f).map h) = (refine x h).pullback _ :=
  rfl


/-- Make a term of `Meq P S`. -/
def mk {X : C} {P : Cᵒᵖ ⥤ D} (S : J.Cover X) (x : P.obj (op X)) : Meq P S :=
  ⟨fun I => P.map I.f.op x, fun I => by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝¹ : CategoryTheory.Category.{max v u, w} D
      inst✝ : CategoryTheory.ConcreteCategory D
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S : J.Cover X
      x : (CategoryTheory.forget D).obj (P.obj { unop := X })
      I : S.Relation
      ⊢ Eq ((P.map I.r.g₁.op) ((fun I => (P.map I.f.op) x) I.fst)) ((P.map I.r.g₂.op …
    -/
    dsimp
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝¹ : CategoryTheory.Category.{max v u, w} D
      inst✝ : CategoryTheory.ConcreteCategory D
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S : J.Cover X
      x : (CategoryTheory.forget D).obj (P.obj { unop := X })
      I : S.Relation
      ⊢ Eq ((P.map I.r.g₁.op) ((P.map I.fst.f.op) x)) ((P.map I.r.g₂.op) ((P.map I.s …
    -/
    simp only [← comp_apply, ← P.map_comp, ← op_comp, I.r.w]⟩
    /-
      🎉 no goals
    -/


theorem mk_apply {X : C} {P : Cᵒᵖ ⥤ D} (S : J.Cover X) (x : P.obj (op X)) (I : S.Arrow) :
    mk S x I = P.map I.f.op x :=
  rfl


/-- The equivalence between the type associated to `multiequalizer (S.index P)` and `Meq P S`. -/
noncomputable def equiv {X : C} (P : Cᵒᵖ ⥤ D) (S : J.Cover X) [HasMultiequalizer (S.index P)] :
    (multiequalizer (S.index P) : D) ≃ Meq P S :=
  Limits.Concrete.multiequalizerEquiv _


@[simp]
theorem equiv_apply {X : C} {P : Cᵒᵖ ⥤ D} {S : J.Cover X} [HasMultiequalizer (S.index P)]
    (x : (multiequalizer (S.index P) : D)) (I : S.Arrow) :
    equiv P S x I = Multiequalizer.ι (S.index P) I x :=
  rfl


theorem equiv_symm_eq_apply {X : C} {P : Cᵒᵖ ⥤ D} {S : J.Cover X} [HasMultiequalizer (S.index P)]
    (x : Meq P S) (I : S.Arrow) :
    Multiequalizer.ι (S.index P) I ((Meq.equiv P S).symm x) = x I := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝³ : CategoryTheory.Category.{max v u, w} D
    inst✝² : CategoryTheory.ConcreteCategory D
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    inst✝ : CategoryTheory.Limits.HasMultiequalizer (S.index P)
    x : CategoryTheory.Meq P S
    I : S.Arrow
    ⊢ Eq ((CategoryTheory.Limits.Multiequalizer.ι (S.index P) I) ((CategoryTheory. …
  -/
  rw [← equiv_apply]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝³ : CategoryTheory.Category.{max v u, w} D
    inst✝² : CategoryTheory.ConcreteCategory D
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    inst✝ : CategoryTheory.Limits.HasMultiequalizer (S.index P)
    x : CategoryTheory.Meq P S
    I : S.Arrow
    ⊢ Eq (↑((CategoryTheory.Meq.equiv P S) ((CategoryTheory.Meq.equiv P S).symm x) …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Make a term of `(J.plusObj P).obj (op X)` from `x : Meq P S`. -/
def mk {X : C} {P : Cᵒᵖ ⥤ D} {S : J.Cover X} (x : Meq P S) : (J.plusObj P).obj (op X) :=
  colimit.ι (J.diagram P X) (op S) ((Meq.equiv P S).symm x)


theorem res_mk_eq_mk_pullback {Y X : C} {P : Cᵒᵖ ⥤ D} {S : J.Cover X} (x : Meq P S) (f : Y ⟶ X) :
    (J.plusObj P).map f.op (mk x) = mk (x.pullback f) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    Y X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    f : Quiver.Hom Y X
    ⊢ Eq (((J.plusObj P).map f.op) (CategoryTheory.GrothendieckTopology.Plus.mk x) …
  -/
  dsimp [mk, plusObj]
  rw [← comp_apply (x := (Meq.equiv P S).symm x), ι_colimMap_assoc, colimit.ι_pre,
    comp_apply (x := (Meq.equiv P S).symm x)]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    Y X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    f : Quiver.Hom Y X
    ⊢ Eq ((CategoryTheory.Limits.colimit.ι (J.diagram P Y) ((J.pullback f).op.obj  …
  -/
  apply congr_arg
  /-
    case h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    Y X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    f : Quiver.Hom Y X
    ⊢ Eq (((J.diagramPullback P f).app { unop := S }) ((CategoryTheory.Meq.equiv P …
  -/
  apply (Meq.equiv P _).injective
  /-
    case h.a
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    Y X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    f : Quiver.Hom Y X
    ⊢ Eq ((CategoryTheory.Meq.equiv P (Opposite.unop ((J.pullback f).op.obj { unop …
  -/
  erw [Equiv.apply_symm_apply]
  /-
    case h.a
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    Y X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    f : Quiver.Hom Y X
    ⊢ Eq ((CategoryTheory.Meq.equiv P (Opposite.unop ((J.pullback f).op.obj { unop …
  -/
  ext i
  simp only [Functor.op_obj, unop_op, pullback_obj, diagram_obj, Functor.comp_obj,
    diagramPullback_app, Meq.equiv_apply, Meq.pullback_apply]
  /-
    case h.a.h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    Y X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    f : Quiver.Hom Y X
    i : (Opposite.unop ((J.pullback f).op.obj { unop := S })).Arrow
    ⊢ Eq ((CategoryTheory.Limits.Multiequalizer.ι ((S.pullback f).index P) i) ((Ca …
  -/
  rw [← comp_apply, Multiequalizer.lift_ι]
  /-
    case h.a.h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    Y X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    f : Quiver.Hom Y X
    i : (Opposite.unop ((J.pullback f).op.obj { unop := S })).Arrow
    ⊢ Eq ((CategoryTheory.Limits.Multiequalizer.ι (S.index P) i.base) ((CategoryTh …
  -/
  erw [Meq.equiv_symm_eq_apply]
  /-
    case h.a.h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    Y X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    f : Quiver.Hom Y X
    i : (Opposite.unop ((J.pullback f).op.obj { unop := S })).Arrow
    ⊢ Eq (↑x i.base) (↑x { Y := i.Y, f := CategoryTheory.CategoryStruct.comp i.f f …
  -/
  cases i; rfl
           /-
             🎉 no goals
           -/


theorem toPlus_mk {X : C} {P : Cᵒᵖ ⥤ D} (S : J.Cover X) (x : P.obj (op X)) :
    (J.toPlus P).app _ x = mk (Meq.mk S x) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    ⊢ Eq (((J.toPlus P).app { unop := X }) x) (CategoryTheory.GrothendieckTopology …
  -/
  dsimp [mk, toPlus]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (Top.top.toMultiequalizer P) (Catego …
  -/
  let e : S ⟶ ⊤ := homOfLE (OrderTop.le_top _)
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    e : Quiver.Hom S Top.top := CategoryTheory.homOfLE ⋯
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (Top.top.toMultiequalizer P) (Catego …
  -/
  rw [← colimit.w _ e.op]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    e : Quiver.Hom S Top.top := CategoryTheory.homOfLE ⋯
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (Top.top.toMultiequalizer P) (Catego …
  -/
  delta Cover.toMultiequalizer
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    e : Quiver.Hom S Top.top := CategoryTheory.homOfLE ⋯
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalize …
  -/
  rw [comp_apply]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    e : Quiver.Hom S Top.top := CategoryTheory.homOfLE ⋯
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((J.diagram P X).map e.op) (Category …
  -/
  erw [comp_apply]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    e : Quiver.Hom S Top.top := CategoryTheory.homOfLE ⋯
    ⊢ Eq ((CategoryTheory.Limits.colimit.ι (J.diagram P X) { unop := S }) (((J.dia …
  -/
  apply congr_arg
  /-
    case h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    e : Quiver.Hom S Top.top := CategoryTheory.homOfLE ⋯
    ⊢ Eq (((J.diagram P X).map e.op) ((CategoryTheory.Limits.Multiequalizer.lift ( …
  -/
  dsimp [diagram]
  /-
    case h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    e : Quiver.Hom S Top.top := CategoryTheory.homOfLE ⋯
    ⊢ Eq ((CategoryTheory.Limits.Multiequalizer.lift (S.index P) (CategoryTheory.L …
  -/
  apply Concrete.multiequalizer_ext
  /-
    case h.h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    e : Quiver.Hom S Top.top := CategoryTheory.homOfLE ⋯
    ⊢ ∀ (t : (S.index P).L), Eq ((CategoryTheory.Limits.Multiequalizer.ι (S.index  …
  -/
  intro i
  simp only [← comp_apply, Category.assoc, Multiequalizer.lift_ι, Category.comp_id,
    Meq.equiv_symm_eq_apply]
  /-
    case h.h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    e : Quiver.Hom S Top.top := CategoryTheory.homOfLE ⋯
    i : (S.index P).L
    ⊢ Eq ((P.map (CategoryTheory.GrothendieckTopology.Cover.Arrow.map i e).f.op) x …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem toPlus_apply {X : C} {P : Cᵒᵖ ⥤ D} (S : J.Cover X) (x : Meq P S) (I : S.Arrow) :
    (J.toPlus P).app _ (x I) = (J.plusObj P).map I.f.op (mk x) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    ⊢ Eq (((J.toPlus P).app { unop := I.Y }) (↑x I)) (((J.plusObj P).map I.f.op) ( …
  -/
  dsimp only [toPlus, plusObj]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (Top.top.toMultiequalizer P) (Catego …
  -/
  delta Cover.toMultiequalizer
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalize …
  -/
  dsimp [mk]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalize …
  -/
  erw [← comp_apply]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalize …
  -/
  rw [ι_colimMap_assoc, colimit.ι_pre, comp_apply, comp_apply]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    ⊢ Eq ((CategoryTheory.Limits.colimit.ι (J.diagram P I.Y) { unop := Top.top })  …
  -/
  dsimp only [Functor.op]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    ⊢ Eq ((CategoryTheory.Limits.colimit.ι (J.diagram P I.Y) { unop := Top.top })  …
  -/
  let e : (J.pullback I.f).obj (unop (op S)) ⟶ ⊤ := homOfLE (OrderTop.le_top _)
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    e : Quiver.Hom ((J.pullback I.f).obj (Opposite.unop { unop := S })) Top.top := …
    ⊢ Eq ((CategoryTheory.Limits.colimit.ι (J.diagram P I.Y) { unop := Top.top })  …
  -/
  rw [← colimit.w _ e.op]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    e : Quiver.Hom ((J.pullback I.f).obj (Opposite.unop { unop := S })) Top.top := …
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((J.diagram P I.Y).map e.op) (Catego …
  -/
  erw [comp_apply]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    e : Quiver.Hom ((J.pullback I.f).obj (Opposite.unop { unop := S })) Top.top := …
    ⊢ Eq ((CategoryTheory.Limits.colimit.ι (J.diagram P I.Y) { unop := (J.pullback …
  -/
  apply congr_arg
  /-
    case h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    e : Quiver.Hom ((J.pullback I.f).obj (Opposite.unop { unop := S })) Top.top := …
    ⊢ Eq (((J.diagram P I.Y).map e.op) ((CategoryTheory.Limits.Multiequalizer.lift …
  -/
  apply Concrete.multiequalizer_ext
  /-
    case h.h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    e : Quiver.Hom ((J.pullback I.f).obj (Opposite.unop { unop := S })) Top.top := …
    ⊢ ∀ (t : ((Opposite.unop { unop := (J.pullback I.f).obj (Opposite.unop { unop  …
  -/
  intro i
  /-
    case h.h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    e : Quiver.Hom ((J.pullback I.f).obj (Opposite.unop { unop := S })) Top.top := …
    i : ((Opposite.unop { unop := (J.pullback I.f).obj (Opposite.unop { unop := S  …
    ⊢ Eq ((CategoryTheory.Limits.Multiequalizer.ι ((Opposite.unop { unop := (J.pul …
  -/
  dsimp
  /-
    case h.h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    e : Quiver.Hom ((J.pullback I.f).obj (Opposite.unop { unop := S })) Top.top := …
    i : ((Opposite.unop { unop := (J.pullback I.f).obj (Opposite.unop { unop := S  …
    ⊢ Eq ((CategoryTheory.Limits.Multiequalizer.ι ((S.pullback I.f).index P) i) (( …
  -/
  erw [← comp_apply, ← comp_apply, ← comp_apply]
  /-
    case h.h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    e : Quiver.Hom ((J.pullback I.f).obj (Opposite.unop { unop := S })) Top.top := …
    i : ((Opposite.unop { unop := (J.pullback I.f).obj (Opposite.unop { unop := S  …
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalize …
  -/
  rw [Multiequalizer.lift_ι, Multiequalizer.lift_ι, Multiequalizer.lift_ι]
  /-
    case h.h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    e : Quiver.Hom ((J.pullback I.f).obj (Opposite.unop { unop := S })) Top.top := …
    i : ((Opposite.unop { unop := (J.pullback I.f).obj (Opposite.unop { unop := S  …
    ⊢ Eq ((P.map (CategoryTheory.GrothendieckTopology.Cover.Arrow.map i e).f.op) ( …
  -/
  erw [Meq.equiv_symm_eq_apply]
  /-
    case h.h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x : CategoryTheory.Meq P S
    I : S.Arrow
    e : Quiver.Hom ((J.pullback I.f).obj (Opposite.unop { unop := S })) Top.top := …
    i : ((Opposite.unop { unop := (J.pullback I.f).obj (Opposite.unop { unop := S  …
    ⊢ Eq ((P.map (CategoryTheory.GrothendieckTopology.Cover.Arrow.map i e).f.op) ( …
  -/
  simpa using (x.condition (Cover.Relation.mk' (I.precompRelation i.f))).symm
  /-
    🎉 no goals
  -/


theorem toPlus_eq_mk {X : C} {P : Cᵒᵖ ⥤ D} (x : P.obj (op X)) :
    (J.toPlus P).app _ x = mk (Meq.mk ⊤ x) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    ⊢ Eq (((J.toPlus P).app { unop := X }) x) (CategoryTheory.GrothendieckTopology …
  -/
  dsimp [mk, toPlus]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (Top.top.toMultiequalizer P) (Catego …
  -/
  delta Cover.toMultiequalizer
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multiequalize …
  -/
  simp only [comp_apply]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    ⊢ Eq ((CategoryTheory.Limits.colimit.ι (J.diagram P X) { unop := Top.top }) (( …
  -/
  apply congr_arg
  /-
    case h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    ⊢ Eq ((CategoryTheory.Limits.Multiequalizer.lift (Top.top.index P) (P.obj { un …
  -/
  apply (Meq.equiv P ⊤).injective
  /-
    case h.a
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    ⊢ Eq ((CategoryTheory.Meq.equiv P Top.top) ((CategoryTheory.Limits.Multiequali …
  -/
  ext i
  /-
    case h.a.h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    i : Top.top.Arrow
    ⊢ Eq (↑((CategoryTheory.Meq.equiv P Top.top) ((CategoryTheory.Limits.Multiequa …
  -/
  rw [Meq.equiv_apply, Equiv.apply_symm_apply, ← comp_apply, Multiequalizer.lift_ι]
  /-
    case h.a.h
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁴ : CategoryTheory.Category.{max v u, w} D
    inst✝³ : CategoryTheory.ConcreteCategory D
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X), …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    x : (CategoryTheory.forget D).obj (P.obj { unop := X })
    i : Top.top.Arrow
    ⊢ Eq ((P.map i.f.op) x) (↑(CategoryTheory.Meq.mk Top.top x) i)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem exists_rep {X : C} {P : Cᵒᵖ ⥤ D} (x : (J.plusObj P).obj (op X)) :
    ∃ (S : J.Cover X) (y : Meq P S), x = mk y := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    x : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
    ⊢ Exists fun S => Exists fun y => Eq x (CategoryTheory.GrothendieckTopology.Pl …
  -/
  obtain ⟨S, y, h⟩ := Concrete.colimit_exists_rep (J.diagram P X) x
  /-
    case intro.intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    x : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
    S : Opposite (J.Cover X)
    y : (CategoryTheory.forget D).obj ((J.diagram P X).obj S)
    h : Eq ((CategoryTheory.Limits.colimit.ι (J.diagram P X) S) y) x
    ⊢ Exists fun S => Exists fun y => Eq x (CategoryTheory.GrothendieckTopology.Pl …
  -/
  use S.unop, Meq.equiv _ _ y
  /-
    case h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    x : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
    S : Opposite (J.Cover X)
    y : (CategoryTheory.forget D).obj ((J.diagram P X).obj S)
    h : Eq ((CategoryTheory.Limits.colimit.ι (J.diagram P X) S) y) x
    ⊢ Eq x (CategoryTheory.GrothendieckTopology.Plus.mk ((CategoryTheory.Meq.equiv …
  -/
  rw [← h]
  /-
    case h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    x : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
    S : Opposite (J.Cover X)
    y : (CategoryTheory.forget D).obj ((J.diagram P X).obj S)
    h : Eq ((CategoryTheory.Limits.colimit.ι (J.diagram P X) S) y) x
    ⊢ Eq ((CategoryTheory.Limits.colimit.ι (J.diagram P X) S) y) (CategoryTheory.G …
  -/
  dsimp [mk]
  /-
    case h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    x : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
    S : Opposite (J.Cover X)
    y : (CategoryTheory.forget D).obj ((J.diagram P X).obj S)
    h : Eq ((CategoryTheory.Limits.colimit.ι (J.diagram P X) S) y) x
    ⊢ Eq ((CategoryTheory.Limits.colimit.ι (J.diagram P X) S) y) ((CategoryTheory. …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem eq_mk_iff_exists {X : C} {P : Cᵒᵖ ⥤ D} {S T : J.Cover X} (x : Meq P S) (y : Meq P T) :
    mk x = mk y ↔ ∃ (W : J.Cover X) (h1 : W ⟶ S) (h2 : W ⟶ T), x.refine h1 = y.refine h2 := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S T : J.Cover X
    x : CategoryTheory.Meq P S
    y : CategoryTheory.Meq P T
    ⊢ Iff (Eq (CategoryTheory.GrothendieckTopology.Plus.mk x) (CategoryTheory.Grot …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S T : J.Cover X
      x : CategoryTheory.Meq P S
      y : CategoryTheory.Meq P T
      ⊢ Eq (CategoryTheory.GrothendieckTopology.Plus.mk x) (CategoryTheory.Grothendi …
    -/
  · intro h
    /-
      case mp
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S T : J.Cover X
      x : CategoryTheory.Meq P S
      y : CategoryTheory.Meq P T
      h : Eq (CategoryTheory.GrothendieckTopology.Plus.mk x) (CategoryTheory.Grothen …
      ⊢ Exists fun W => Exists fun h1 => Exists fun h2 => Eq (x.refine h1) (y.refine …
    -/
    obtain ⟨W, h1, h2, hh⟩ := Concrete.colimit_exists_of_rep_eq.{u} _ _ _ h
    /-
      case mp.intro.intro.intro
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S T : J.Cover X
      x : CategoryTheory.Meq P S
      y : CategoryTheory.Meq P T
      h : Eq (CategoryTheory.GrothendieckTopology.Plus.mk x) (CategoryTheory.Grothen …
      W : Opposite (J.Cover X)
      h1 : Quiver.Hom { unop := S } W
      h2 : Quiver.Hom { unop := T } W
      hh : Eq (((J.diagram P X).map h1) ((CategoryTheory.Meq.equiv P S).symm x)) ((( …
      ⊢ Exists fun W => Exists fun h1 => Exists fun h2 => Eq (x.refine h1) (y.refine …
    -/
    use W.unop, h1.unop, h2.unop
    /-
      case h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S T : J.Cover X
      x : CategoryTheory.Meq P S
      y : CategoryTheory.Meq P T
      h : Eq (CategoryTheory.GrothendieckTopology.Plus.mk x) (CategoryTheory.Grothen …
      W : Opposite (J.Cover X)
      h1 : Quiver.Hom { unop := S } W
      h2 : Quiver.Hom { unop := T } W
      hh : Eq (((J.diagram P X).map h1) ((CategoryTheory.Meq.equiv P S).symm x)) ((( …
      ⊢ Eq (x.refine h1.unop) (y.refine h2.unop)
    -/
    ext I
    /-
      case h.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S T : J.Cover X
      x : CategoryTheory.Meq P S
      y : CategoryTheory.Meq P T
      h : Eq (CategoryTheory.GrothendieckTopology.Plus.mk x) (CategoryTheory.Grothen …
      W : Opposite (J.Cover X)
      h1 : Quiver.Hom { unop := S } W
      h2 : Quiver.Hom { unop := T } W
      hh : Eq (((J.diagram P X).map h1) ((CategoryTheory.Meq.equiv P S).symm x)) ((( …
      I : (Opposite.unop W).Arrow
      ⊢ Eq (↑(x.refine h1.unop) I) (↑(y.refine h2.unop) I)
    -/
    apply_fun Multiequalizer.ι (W.unop.index P) I at hh
    /-
      case h.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S T : J.Cover X
      x : CategoryTheory.Meq P S
      y : CategoryTheory.Meq P T
      h : Eq (CategoryTheory.GrothendieckTopology.Plus.mk x) (CategoryTheory.Grothen …
      W : Opposite (J.Cover X)
      h1 : Quiver.Hom { unop := S } W
      h2 : Quiver.Hom { unop := T } W
      I : (Opposite.unop W).Arrow
      hh : Eq ((CategoryTheory.Limits.Multiequalizer.ι ((Opposite.unop W).index P) I …
      ⊢ Eq (↑(x.refine h1.unop) I) (↑(y.refine h2.unop) I)
    -/
    convert hh
    all_goals
      dsimp [diagram]
      erw [← comp_apply, Multiequalizer.lift_ι, Meq.equiv_symm_eq_apply]
      cases I; rfl
    /-
      case mpr
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S T : J.Cover X
      x : CategoryTheory.Meq P S
      y : CategoryTheory.Meq P T
      ⊢ (Exists fun W => Exists fun h1 => Exists fun h2 => Eq (x.refine h1) (y.refin …
    -/
  · rintro ⟨S, h1, h2, e⟩
    /-
      case mpr.intro.intro.intro
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S✝ T : J.Cover X
      x : CategoryTheory.Meq P S✝
      y : CategoryTheory.Meq P T
      S : J.Cover X
      h1 : Quiver.Hom S S✝
      h2 : Quiver.Hom S T
      e : Eq (x.refine h1) (y.refine h2)
      ⊢ Eq (CategoryTheory.GrothendieckTopology.Plus.mk x) (CategoryTheory.Grothendi …
    -/
    apply Concrete.colimit_rep_eq_of_exists
    /-
      case mpr.intro.intro.intro.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S✝ T : J.Cover X
      x : CategoryTheory.Meq P S✝
      y : CategoryTheory.Meq P T
      S : J.Cover X
      h1 : Quiver.Hom S S✝
      h2 : Quiver.Hom S T
      e : Eq (x.refine h1) (y.refine h2)
      ⊢ Exists fun k => Exists fun f => Exists fun g => Eq (((J.diagram P X).map f)  …
    -/
    use op S, h1.op, h2.op
    /-
      case h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S✝ T : J.Cover X
      x : CategoryTheory.Meq P S✝
      y : CategoryTheory.Meq P T
      S : J.Cover X
      h1 : Quiver.Hom S S✝
      h2 : Quiver.Hom S T
      e : Eq (x.refine h1) (y.refine h2)
      ⊢ Eq (((J.diagram P X).map h1.op) ((CategoryTheory.Meq.equiv P S✝).symm x)) (( …
    -/
    apply Concrete.multiequalizer_ext
    /-
      case h.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S✝ T : J.Cover X
      x : CategoryTheory.Meq P S✝
      y : CategoryTheory.Meq P T
      S : J.Cover X
      h1 : Quiver.Hom S S✝
      h2 : Quiver.Hom S T
      e : Eq (x.refine h1) (y.refine h2)
      ⊢ ∀ (t : ((Opposite.unop { unop := S }).index P).L), Eq ((CategoryTheory.Limit …
    -/
    intro i
    /-
      case h.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S✝ T : J.Cover X
      x : CategoryTheory.Meq P S✝
      y : CategoryTheory.Meq P T
      S : J.Cover X
      h1 : Quiver.Hom S S✝
      h2 : Quiver.Hom S T
      e : Eq (x.refine h1) (y.refine h2)
      i : ((Opposite.unop { unop := S }).index P).L
      ⊢ Eq ((CategoryTheory.Limits.Multiequalizer.ι ((Opposite.unop { unop := S }).i …
    -/
    apply_fun fun ee => ee i at e
    /-
      case h.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S✝ T : J.Cover X
      x : CategoryTheory.Meq P S✝
      y : CategoryTheory.Meq P T
      S : J.Cover X
      h1 : Quiver.Hom S S✝
      h2 : Quiver.Hom S T
      i : ((Opposite.unop { unop := S }).index P).L
      e : Eq (↑(x.refine h1) i) (↑(y.refine h2) i)
      ⊢ Eq ((CategoryTheory.Limits.Multiequalizer.ι ((Opposite.unop { unop := S }).i …
    -/
    convert e
    all_goals
      dsimp
      erw [← comp_apply, Multiequalizer.lift_ι]
      erw [Meq.equiv_symm_eq_apply]
      cases i; rfl


/-- `P⁺` is always separated. -/
theorem sep {X : C} (P : Cᵒᵖ ⥤ D) (S : J.Cover X) (x y : (J.plusObj P).obj (op X))
    (h : ∀ I : S.Arrow, (J.plusObj P).map I.f.op x = (J.plusObj P).map I.f.op y) : x = y := by
  -- First, we choose representatives for x and y.
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    x y : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
    h : ∀ (I : S.Arrow), Eq (((J.plusObj P).map I.f.op) x) (((J.plusObj P).map I.f …
    ⊢ Eq x y
  -/
  obtain ⟨Sx, x, rfl⟩ := exists_rep x
  /-
    case intro.intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S : J.Cover X
    y : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
    Sx : J.Cover X
    x : CategoryTheory.Meq P Sx
    h : ∀ (I : S.Arrow), Eq (((J.plusObj P).map I.f.op) (CategoryTheory.Grothendie …
    ⊢ Eq (CategoryTheory.GrothendieckTopology.Plus.mk x) y
  -/
  obtain ⟨Sy, y, rfl⟩ := exists_rep y
  /-
    case intro.intro.intro.intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S Sx : J.Cover X
    x : CategoryTheory.Meq P Sx
    Sy : J.Cover X
    y : CategoryTheory.Meq P Sy
    h : ∀ (I : S.Arrow), Eq (((J.plusObj P).map I.f.op) (CategoryTheory.Grothendie …
    ⊢ Eq (CategoryTheory.GrothendieckTopology.Plus.mk x) (CategoryTheory.Grothendi …
  -/
  simp only [res_mk_eq_mk_pullback] at h
  -- Next, using our assumption,
  -- choose covers over which the pullbacks of these representatives become equal.
  /-
    case intro.intro.intro.intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S Sx : J.Cover X
    x : CategoryTheory.Meq P Sx
    Sy : J.Cover X
    y : CategoryTheory.Meq P Sy
    h : ∀ (I : S.Arrow), Eq (CategoryTheory.GrothendieckTopology.Plus.mk (x.pullba …
    ⊢ Eq (CategoryTheory.GrothendieckTopology.Plus.mk x) (CategoryTheory.Grothendi …
  -/
  choose W h1 h2 hh using fun I : S.Arrow => (eq_mk_iff_exists _ _).mp (h I)
  -- To prove equality, it suffices to prove that there exists a cover over which
  -- the representatives become equal.
  /-
    case intro.intro.intro.intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S Sx : J.Cover X
    x : CategoryTheory.Meq P Sx
    Sy : J.Cover X
    y : CategoryTheory.Meq P Sy
    h : ∀ (I : S.Arrow), Eq (CategoryTheory.GrothendieckTopology.Plus.mk (x.pullba …
    W : (I : S.Arrow) → J.Cover I.Y
    h1 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sx)
    h2 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sy)
    hh : ∀ (I : S.Arrow), Eq ((x.pullback I.f).refine (h1 I)) ((y.pullback I.f).re …
    ⊢ Eq (CategoryTheory.GrothendieckTopology.Plus.mk x) (CategoryTheory.Grothendi …
  -/
  rw [eq_mk_iff_exists]
  -- Construct the cover over which the representatives become equal by combining the various
  -- covers chosen above.
  /-
    case intro.intro.intro.intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S Sx : J.Cover X
    x : CategoryTheory.Meq P Sx
    Sy : J.Cover X
    y : CategoryTheory.Meq P Sy
    h : ∀ (I : S.Arrow), Eq (CategoryTheory.GrothendieckTopology.Plus.mk (x.pullba …
    W : (I : S.Arrow) → J.Cover I.Y
    h1 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sx)
    h2 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sy)
    hh : ∀ (I : S.Arrow), Eq ((x.pullback I.f).refine (h1 I)) ((y.pullback I.f).re …
    ⊢ Exists fun W => Exists fun h1 => Exists fun h2 => Eq (x.refine h1) (y.refine …
  -/
  let B : J.Cover X := S.bind W
  /-
    case intro.intro.intro.intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S Sx : J.Cover X
    x : CategoryTheory.Meq P Sx
    Sy : J.Cover X
    y : CategoryTheory.Meq P Sy
    h : ∀ (I : S.Arrow), Eq (CategoryTheory.GrothendieckTopology.Plus.mk (x.pullba …
    W : (I : S.Arrow) → J.Cover I.Y
    h1 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sx)
    h2 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sy)
    hh : ∀ (I : S.Arrow), Eq ((x.pullback I.f).refine (h1 I)) ((y.pullback I.f).re …
    B : J.Cover X := S.bind W
    ⊢ Exists fun W => Exists fun h1 => Exists fun h2 => Eq (x.refine h1) (y.refine …
  -/
  use B
  -- Prove that this cover refines the two covers over which our representatives are defined
  -- and use these proofs.
  let ex : B ⟶ Sx :=
    homOfLE
      (by
        rintro Y f ⟨Z, e1, e2, he2, he1, hee⟩
        rw [← hee]
        apply leOfHom (h1 ⟨_, _, he2⟩)
        exact he1)
  let ey : B ⟶ Sy :=
    homOfLE
      (by
        rintro Y f ⟨Z, e1, e2, he2, he1, hee⟩
        rw [← hee]
        apply leOfHom (h2 ⟨_, _, he2⟩)
        exact he1)
  /-
    case h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S Sx : J.Cover X
    x : CategoryTheory.Meq P Sx
    Sy : J.Cover X
    y : CategoryTheory.Meq P Sy
    h : ∀ (I : S.Arrow), Eq (CategoryTheory.GrothendieckTopology.Plus.mk (x.pullba …
    W : (I : S.Arrow) → J.Cover I.Y
    h1 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sx)
    h2 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sy)
    hh : ∀ (I : S.Arrow), Eq ((x.pullback I.f).refine (h1 I)) ((y.pullback I.f).re …
    B : J.Cover X := S.bind W
    ex : Quiver.Hom B Sx := CategoryTheory.homOfLE ⋯
    ey : Quiver.Hom B Sy := CategoryTheory.homOfLE ⋯
    ⊢ Exists fun h1 => Exists fun h2 => Eq (x.refine h1) (y.refine h2)
  -/
  use ex, ey
  -- Now prove that indeed the representatives become equal over `B`.
  -- This will follow by using the fact that our representatives become
  -- equal over the chosen covers.
  /-
    case h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S Sx : J.Cover X
    x : CategoryTheory.Meq P Sx
    Sy : J.Cover X
    y : CategoryTheory.Meq P Sy
    h : ∀ (I : S.Arrow), Eq (CategoryTheory.GrothendieckTopology.Plus.mk (x.pullba …
    W : (I : S.Arrow) → J.Cover I.Y
    h1 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sx)
    h2 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sy)
    hh : ∀ (I : S.Arrow), Eq ((x.pullback I.f).refine (h1 I)) ((y.pullback I.f).re …
    B : J.Cover X := S.bind W
    ex : Quiver.Hom B Sx := CategoryTheory.homOfLE ⋯
    ey : Quiver.Hom B Sy := CategoryTheory.homOfLE ⋯
    ⊢ Eq (x.refine ex) (y.refine ey)
  -/
  ext1 I
  /-
    case h.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S Sx : J.Cover X
    x : CategoryTheory.Meq P Sx
    Sy : J.Cover X
    y : CategoryTheory.Meq P Sy
    h : ∀ (I : S.Arrow), Eq (CategoryTheory.GrothendieckTopology.Plus.mk (x.pullba …
    W : (I : S.Arrow) → J.Cover I.Y
    h1 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sx)
    h2 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sy)
    hh : ∀ (I : S.Arrow), Eq ((x.pullback I.f).refine (h1 I)) ((y.pullback I.f).re …
    B : J.Cover X := S.bind W
    ex : Quiver.Hom B Sx := CategoryTheory.homOfLE ⋯
    ey : Quiver.Hom B Sy := CategoryTheory.homOfLE ⋯
    I : B.Arrow
    ⊢ Eq (↑(x.refine ex) I) (↑(y.refine ey) I)
  -/
  let IS : S.Arrow := I.fromMiddle
  /-
    case h.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S Sx : J.Cover X
    x : CategoryTheory.Meq P Sx
    Sy : J.Cover X
    y : CategoryTheory.Meq P Sy
    h : ∀ (I : S.Arrow), Eq (CategoryTheory.GrothendieckTopology.Plus.mk (x.pullba …
    W : (I : S.Arrow) → J.Cover I.Y
    h1 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sx)
    h2 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sy)
    hh : ∀ (I : S.Arrow), Eq ((x.pullback I.f).refine (h1 I)) ((y.pullback I.f).re …
    B : J.Cover X := S.bind W
    ex : Quiver.Hom B Sx := CategoryTheory.homOfLE ⋯
    ey : Quiver.Hom B Sy := CategoryTheory.homOfLE ⋯
    I : B.Arrow
    IS : S.Arrow := I.fromMiddle
    ⊢ Eq (↑(x.refine ex) I) (↑(y.refine ey) I)
  -/
  specialize hh IS
  /-
    case h.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S Sx : J.Cover X
    x : CategoryTheory.Meq P Sx
    Sy : J.Cover X
    y : CategoryTheory.Meq P Sy
    h : ∀ (I : S.Arrow), Eq (CategoryTheory.GrothendieckTopology.Plus.mk (x.pullba …
    W : (I : S.Arrow) → J.Cover I.Y
    h1 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sx)
    h2 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sy)
    B : J.Cover X := S.bind W
    ex : Quiver.Hom B Sx := CategoryTheory.homOfLE ⋯
    ey : Quiver.Hom B Sy := CategoryTheory.homOfLE ⋯
    I : B.Arrow
    IS : S.Arrow := I.fromMiddle
    hh : Eq ((x.pullback IS.f).refine (h1 IS)) ((y.pullback IS.f).refine (h2 IS))
    ⊢ Eq (↑(x.refine ex) I) (↑(y.refine ey) I)
  -/
  let IW : (W IS).Arrow := I.toMiddle
  /-
    case h.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S Sx : J.Cover X
    x : CategoryTheory.Meq P Sx
    Sy : J.Cover X
    y : CategoryTheory.Meq P Sy
    h : ∀ (I : S.Arrow), Eq (CategoryTheory.GrothendieckTopology.Plus.mk (x.pullba …
    W : (I : S.Arrow) → J.Cover I.Y
    h1 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sx)
    h2 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sy)
    B : J.Cover X := S.bind W
    ex : Quiver.Hom B Sx := CategoryTheory.homOfLE ⋯
    ey : Quiver.Hom B Sy := CategoryTheory.homOfLE ⋯
    I : B.Arrow
    IS : S.Arrow := I.fromMiddle
    hh : Eq ((x.pullback IS.f).refine (h1 IS)) ((y.pullback IS.f).refine (h2 IS))
    IW : (W IS).Arrow := I.toMiddle
    ⊢ Eq (↑(x.refine ex) I) (↑(y.refine ey) I)
  -/
  apply_fun fun e => e IW at hh
  /-
    case h.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    X : C
    P : CategoryTheory.Functor (Opposite C) D
    S Sx : J.Cover X
    x : CategoryTheory.Meq P Sx
    Sy : J.Cover X
    y : CategoryTheory.Meq P Sy
    h : ∀ (I : S.Arrow), Eq (CategoryTheory.GrothendieckTopology.Plus.mk (x.pullba …
    W : (I : S.Arrow) → J.Cover I.Y
    h1 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sx)
    h2 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sy)
    B : J.Cover X := S.bind W
    ex : Quiver.Hom B Sx := CategoryTheory.homOfLE ⋯
    ey : Quiver.Hom B Sy := CategoryTheory.homOfLE ⋯
    I : B.Arrow
    IS : S.Arrow := I.fromMiddle
    IW : (W IS).Arrow := I.toMiddle
    hh : Eq (↑((x.pullback IS.f).refine (h1 IS)) IW) (↑((y.pullback IS.f).refine ( …
    ⊢ Eq (↑(x.refine ex) I) (↑(y.refine ey) I)
  -/
  convert hh using 1
    /-
      case h.e'_2.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S Sx : J.Cover X
      x : CategoryTheory.Meq P Sx
      Sy : J.Cover X
      y : CategoryTheory.Meq P Sy
      h : ∀ (I : S.Arrow), Eq (CategoryTheory.GrothendieckTopology.Plus.mk (x.pullba …
      W : (I : S.Arrow) → J.Cover I.Y
      h1 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sx)
      h2 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sy)
      B : J.Cover X := S.bind W
      ex : Quiver.Hom B Sx := CategoryTheory.homOfLE ⋯
      ey : Quiver.Hom B Sy := CategoryTheory.homOfLE ⋯
      I : B.Arrow
      IS : S.Arrow := I.fromMiddle
      IW : (W IS).Arrow := I.toMiddle
      hh : Eq (↑((x.pullback IS.f).refine (h1 IS)) IW) (↑((y.pullback IS.f).refine ( …
      e_1✝ : Eq ((CategoryTheory.forget D).obj (P.obj { unop := I.Y })) ((CategoryTh …
      ⊢ Eq (↑(x.refine ex) I) (↑((x.pullback IS.f).refine (h1 IS)) IW)
    -/
  · exact x.congr_apply I.middle_spec.symm _
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      X : C
      P : CategoryTheory.Functor (Opposite C) D
      S Sx : J.Cover X
      x : CategoryTheory.Meq P Sx
      Sy : J.Cover X
      y : CategoryTheory.Meq P Sy
      h : ∀ (I : S.Arrow), Eq (CategoryTheory.GrothendieckTopology.Plus.mk (x.pullba …
      W : (I : S.Arrow) → J.Cover I.Y
      h1 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sx)
      h2 : (I : S.Arrow) → Quiver.Hom (W I) ((J.pullback I.f).obj Sy)
      B : J.Cover X := S.bind W
      ex : Quiver.Hom B Sx := CategoryTheory.homOfLE ⋯
      ey : Quiver.Hom B Sy := CategoryTheory.homOfLE ⋯
      I : B.Arrow
      IS : S.Arrow := I.fromMiddle
      IW : (W IS).Arrow := I.toMiddle
      hh : Eq (↑((x.pullback IS.f).refine (h1 IS)) IW) (↑((y.pullback IS.f).refine ( …
      e_1✝ : Eq ((CategoryTheory.forget D).obj (P.obj { unop := I.Y })) ((CategoryTh …
      ⊢ Eq (↑(y.refine ey) I) (↑((y.pullback IS.f).refine (h2 IS)) IW)
    -/
  · exact y.congr_apply I.middle_spec.symm _
    /-
      🎉 no goals
    -/


theorem inj_of_sep (P : Cᵒᵖ ⥤ D)
    (hsep :
      ∀ (X : C) (S : J.Cover X) (x y : P.obj (op X)),
        (∀ I : S.Arrow, P.map I.f.op x = P.map I.f.op y) → x = y)
    (X : C) : Function.Injective ((J.toPlus P).app (op X)) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    ⊢ Function.Injective ⇑((J.toPlus P).app { unop := X })
  -/
  intro x y h
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    x y : (CategoryTheory.forget D).obj (P.obj { unop := X })
    h : Eq (((J.toPlus P).app { unop := X }) x) (((J.toPlus P).app { unop := X }) y)
    ⊢ Eq x y
  -/
  simp only [toPlus_eq_mk] at h
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    x y : (CategoryTheory.forget D).obj (P.obj { unop := X })
    h : Eq (CategoryTheory.GrothendieckTopology.Plus.mk (CategoryTheory.Meq.mk Top …
    ⊢ Eq x y
  -/
  rw [eq_mk_iff_exists] at h
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    x y : (CategoryTheory.forget D).obj (P.obj { unop := X })
    h : Exists fun W => Exists fun h1 => Exists fun h2 => Eq ((CategoryTheory.Meq. …
    ⊢ Eq x y
  -/
  obtain ⟨W, h1, h2, hh⟩ := h
  /-
    case intro.intro.intro
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    x y : (CategoryTheory.forget D).obj (P.obj { unop := X })
    W : J.Cover X
    h1 h2 : Quiver.Hom W Top.top
    hh : Eq ((CategoryTheory.Meq.mk Top.top x).refine h1) ((CategoryTheory.Meq.mk  …
    ⊢ Eq x y
  -/
  apply hsep X W
  /-
    case intro.intro.intro.a
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    x y : (CategoryTheory.forget D).obj (P.obj { unop := X })
    W : J.Cover X
    h1 h2 : Quiver.Hom W Top.top
    hh : Eq ((CategoryTheory.Meq.mk Top.top x).refine h1) ((CategoryTheory.Meq.mk  …
    ⊢ ∀ (I : W.Arrow), Eq ((P.map I.f.op) x) ((P.map I.f.op) y)
  -/
  intro I
  /-
    case intro.intro.intro.a
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    x y : (CategoryTheory.forget D).obj (P.obj { unop := X })
    W : J.Cover X
    h1 h2 : Quiver.Hom W Top.top
    hh : Eq ((CategoryTheory.Meq.mk Top.top x).refine h1) ((CategoryTheory.Meq.mk  …
    I : W.Arrow
    ⊢ Eq ((P.map I.f.op) x) ((P.map I.f.op) y)
  -/
  apply_fun fun e => e I at hh
  /-
    case intro.intro.intro.a
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    x y : (CategoryTheory.forget D).obj (P.obj { unop := X })
    W : J.Cover X
    h1 h2 : Quiver.Hom W Top.top
    I : W.Arrow
    hh : Eq (↑((CategoryTheory.Meq.mk Top.top x).refine h1) I) (↑((CategoryTheory. …
    ⊢ Eq ((P.map I.f.op) x) ((P.map I.f.op) y)
  -/
  exact hh
  /-
    🎉 no goals
  -/


/-- An auxiliary definition to be used in the proof of `exists_of_sep` below.
  Given a compatible family of local sections for `P⁺`, and representatives of said sections,
  construct a compatible family of local sections of `P` over the combination of the covers
  associated to the representatives.
  The separatedness condition is used to prove compatibility among these local sections of `P`. -/
def meqOfSep (P : Cᵒᵖ ⥤ D)
    (hsep :
      ∀ (X : C) (S : J.Cover X) (x y : P.obj (op X)),
        (∀ I : S.Arrow, P.map I.f.op x = P.map I.f.op y) → x = y)
    (X : C) (S : J.Cover X) (s : Meq (J.plusObj P) S) (T : ∀ I : S.Arrow, J.Cover I.Y)
    (t : ∀ I : S.Arrow, Meq P (T I)) (ht : ∀ I : S.Arrow, s I = mk (t I)) : Meq P (S.bind T) where
  val I := t I.fromMiddle I.toMiddle
  property := by
    /-
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      s : CategoryTheory.Meq (J.plusObj P) S
      T : (I : S.Arrow) → J.Cover I.Y
      t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
      ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
      ⊢ ∀ (I : (S.bind T).Relation), Eq ((P.map I.r.g₁.op) ((fun I => ↑(t I.fromMidd …
    -/
    intro II
    /-
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      s : CategoryTheory.Meq (J.plusObj P) S
      T : (I : S.Arrow) → J.Cover I.Y
      t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
      ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
      II : (S.bind T).Relation
      ⊢ Eq ((P.map II.r.g₁.op) ((fun I => ↑(t I.fromMiddle) I.toMiddle) II.fst)) ((P …
    -/
    apply inj_of_sep P hsep
    rw [← comp_apply, ← comp_apply, (J.toPlus P).naturality, (J.toPlus P).naturality, comp_apply,
      comp_apply]
    erw [toPlus_apply (T II.fst.fromMiddle) (t II.fst.fromMiddle) II.fst.toMiddle,
      toPlus_apply (T II.snd.fromMiddle) (t II.snd.fromMiddle) II.snd.toMiddle, ← ht, ← ht, ←
      comp_apply, ← comp_apply, ← (J.plusObj P).map_comp, ← (J.plusObj P).map_comp]
    /-
      case a
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁵ : CategoryTheory.Category.{max v u, w} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      s : CategoryTheory.Meq (J.plusObj P) S
      T : (I : S.Arrow) → J.Cover I.Y
      t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
      ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
      II : (S.bind T).Relation
      ⊢ Eq (((J.plusObj P).map (CategoryTheory.CategoryStruct.comp II.fst.toMiddle.f …
    -/
    rw [← op_comp, ← op_comp]
    exact s.condition
      (Cover.Relation.mk { hf := II.fst.from_middle_condition }
        { hf := II.snd.from_middle_condition }
        { g₁ := II.r.g₁ ≫ II.fst.toMiddleHom
          g₂ := II.r.g₂ ≫ II.snd.toMiddleHom
          w := by simpa only [Category.assoc, Cover.Arrow.middle_spec] using II.r.w })


theorem exists_of_sep (P : Cᵒᵖ ⥤ D)
    (hsep :
      ∀ (X : C) (S : J.Cover X) (x y : P.obj (op X)),
        (∀ I : S.Arrow, P.map I.f.op x = P.map I.f.op y) → x = y)
    (X : C) (S : J.Cover X) (s : Meq (J.plusObj P) S) :
    ∃ t : (J.plusObj P).obj (op X), Meq.mk S t = s := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    ⊢ Exists fun t => Eq (CategoryTheory.Meq.mk S t) s
  -/
  have inj : ∀ X : C, Function.Injective ((J.toPlus P).app (op X)) := inj_of_sep _ hsep
  -- Choose representatives for the given local sections.
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    ⊢ Exists fun t => Eq (CategoryTheory.Meq.mk S t) s
  -/
  choose T t ht using fun I => exists_rep (s I)
  -- Construct a large cover over which we will define a representative that will
  -- provide the gluing of the given local sections.
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    ⊢ Exists fun t => Eq (CategoryTheory.Meq.mk S t) s
  -/
  let B : J.Cover X := S.bind T
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    ⊢ Exists fun t => Eq (CategoryTheory.Meq.mk S t) s
  -/
  choose Z e1 e2 he2 _ _ using fun I : B.Arrow => I.hf
  -- Construct a compatible system of local sections over this large cover, using the chosen
  -- representatives of our local sections.
  -- The compatibility here follows from the separatedness assumption.
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    ⊢ Exists fun t => Eq (CategoryTheory.Meq.mk S t) s
  -/
  let w : Meq P B := meqOfSep P hsep X S s T t ht
  -- The associated gluing will be the candidate section.
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    ⊢ Exists fun t => Eq (CategoryTheory.Meq.mk S t) s
  -/
  use mk w
  /-
    case h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    ⊢ Eq (CategoryTheory.Meq.mk S (CategoryTheory.GrothendieckTopology.Plus.mk w)) s
  -/
  ext I
  /-
    case h.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    I : S.Arrow
    ⊢ Eq (↑(CategoryTheory.Meq.mk S (CategoryTheory.GrothendieckTopology.Plus.mk w …
  -/
  dsimp [Meq.mk]
  /-
    case h.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    I : S.Arrow
    ⊢ Eq (((J.plusObj P).map I.f.op) (CategoryTheory.GrothendieckTopology.Plus.mk  …
  -/
  rw [ht, res_mk_eq_mk_pullback]
  -- Use the separatedness of `P⁺` to prove that this is indeed a gluing of our
  -- original local sections.
  /-
    case h.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    I : S.Arrow
    ⊢ Eq (CategoryTheory.GrothendieckTopology.Plus.mk (w.pullback I.f)) (CategoryT …
  -/
  apply sep P (T I)
  /-
    case h.h.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    I : S.Arrow
    ⊢ ∀ (I_1 : (T I).Arrow), Eq (((J.plusObj P).map I_1.f.op) (CategoryTheory.Grot …
  -/
  intro II
  /-
    case h.h.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    I : S.Arrow
    II : (T I).Arrow
    ⊢ Eq (((J.plusObj P).map II.f.op) (CategoryTheory.GrothendieckTopology.Plus.mk …
  -/
  simp only [res_mk_eq_mk_pullback, eq_mk_iff_exists]
  -- It suffices to prove equality for representatives over a
  -- convenient sufficiently large cover...
  /-
    case h.h.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    I : S.Arrow
    II : (T I).Arrow
    ⊢ Exists fun W => Exists fun h1 => Exists fun h2 => Eq (((w.pullback I.f).pull …
  -/
  use (J.pullback II.f).obj (T I)
  let e0 : (J.pullback II.f).obj (T I) ⟶ (J.pullback II.f).obj ((J.pullback I.f).obj B) :=
    homOfLE
      (by
        intro Y f hf
        apply Sieve.le_pullback_bind _ _ _ I.hf
        · cases I
          exact hf)
  /-
    case h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    I : S.Arrow
    II : (T I).Arrow
    e0 : Quiver.Hom ((J.pullback II.f).obj (T I)) ((J.pullback II.f).obj ((J.pullb …
    ⊢ Exists fun h1 => Exists fun h2 => Eq (((w.pullback I.f).pullback II.f).refin …
  -/
  use e0, 𝟙 _
  /-
    case h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    I : S.Arrow
    II : (T I).Arrow
    e0 : Quiver.Hom ((J.pullback II.f).obj (T I)) ((J.pullback II.f).obj ((J.pullb …
    ⊢ Eq (((w.pullback I.f).pullback II.f).refine e0) (((t I).pullback II.f).refin …
  -/
  ext IV
  let IA : B.Arrow := ⟨_, (IV.f ≫ II.f) ≫ I.f,
    ⟨I.Y, _, _, I.hf, Sieve.downward_closed _ II.hf _, rfl⟩⟩
  /-
    case h.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    I : S.Arrow
    II : (T I).Arrow
    e0 : Quiver.Hom ((J.pullback II.f).obj (T I)) ((J.pullback II.f).obj ((J.pullb …
    IV : ((J.pullback II.f).obj (T I)).Arrow
    IA : B.Arrow := { Y := IV.Y, f := CategoryTheory.CategoryStruct.comp (Category …
    ⊢ Eq (↑(((w.pullback I.f).pullback II.f).refine e0) IV) (↑(((t I).pullback II. …
  -/
  let IB : S.Arrow := IA.fromMiddle
  /-
    case h.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    I : S.Arrow
    II : (T I).Arrow
    e0 : Quiver.Hom ((J.pullback II.f).obj (T I)) ((J.pullback II.f).obj ((J.pullb …
    IV : ((J.pullback II.f).obj (T I)).Arrow
    IA : B.Arrow := { Y := IV.Y, f := CategoryTheory.CategoryStruct.comp (Category …
    IB : S.Arrow := IA.fromMiddle
    ⊢ Eq (↑(((w.pullback I.f).pullback II.f).refine e0) IV) (↑(((t I).pullback II. …
  -/
  let IC : (T IB).Arrow := IA.toMiddle
  /-
    case h.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    I : S.Arrow
    II : (T I).Arrow
    e0 : Quiver.Hom ((J.pullback II.f).obj (T I)) ((J.pullback II.f).obj ((J.pullb …
    IV : ((J.pullback II.f).obj (T I)).Arrow
    IA : B.Arrow := { Y := IV.Y, f := CategoryTheory.CategoryStruct.comp (Category …
    IB : S.Arrow := IA.fromMiddle
    IC : (T IB).Arrow := IA.toMiddle
    ⊢ Eq (↑(((w.pullback I.f).pullback II.f).refine e0) IV) (↑(((t I).pullback II. …
  -/
  let ID : (T I).Arrow := ⟨IV.Y, IV.f ≫ II.f, Sieve.downward_closed (T I).1 II.hf IV.f⟩
  /-
    case h.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    I : S.Arrow
    II : (T I).Arrow
    e0 : Quiver.Hom ((J.pullback II.f).obj (T I)) ((J.pullback II.f).obj ((J.pullb …
    IV : ((J.pullback II.f).obj (T I)).Arrow
    IA : B.Arrow := { Y := IV.Y, f := CategoryTheory.CategoryStruct.comp (Category …
    IB : S.Arrow := IA.fromMiddle
    IC : (T IB).Arrow := IA.toMiddle
    ID : (T I).Arrow := { Y := IV.Y, f := CategoryTheory.CategoryStruct.comp IV.f  …
    ⊢ Eq (↑(((w.pullback I.f).pullback II.f).refine e0) IV) (↑(((t I).pullback II. …
  -/
  change t IB IC = t I ID
  /-
    case h.h
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    I : S.Arrow
    II : (T I).Arrow
    e0 : Quiver.Hom ((J.pullback II.f).obj (T I)) ((J.pullback II.f).obj ((J.pullb …
    IV : ((J.pullback II.f).obj (T I)).Arrow
    IA : B.Arrow := { Y := IV.Y, f := CategoryTheory.CategoryStruct.comp (Category …
    IB : S.Arrow := IA.fromMiddle
    IC : (T IB).Arrow := IA.toMiddle
    ID : (T I).Arrow := { Y := IV.Y, f := CategoryTheory.CategoryStruct.comp IV.f  …
    ⊢ Eq (↑(t IB) IC) (↑(t I) ID)
  -/
  apply inj IV.Y
  /-
    case h.h.a
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    I : S.Arrow
    II : (T I).Arrow
    e0 : Quiver.Hom ((J.pullback II.f).obj (T I)) ((J.pullback II.f).obj ((J.pullb …
    IV : ((J.pullback II.f).obj (T I)).Arrow
    IA : B.Arrow := { Y := IV.Y, f := CategoryTheory.CategoryStruct.comp (Category …
    IB : S.Arrow := IA.fromMiddle
    IC : (T IB).Arrow := IA.toMiddle
    ID : (T I).Arrow := { Y := IV.Y, f := CategoryTheory.CategoryStruct.comp IV.f  …
    ⊢ Eq (((J.toPlus P).app { unop := IV.Y }) (↑(t IB) IC)) (((J.toPlus P).app { u …
  -/
  erw [toPlus_apply (T I) (t I) ID, toPlus_apply (T IB) (t IB) IC, ← ht, ← ht]
  -- Conclude by constructing the relation showing equality...
  /-
    case h.h.a
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    I : S.Arrow
    II : (T I).Arrow
    e0 : Quiver.Hom ((J.pullback II.f).obj (T I)) ((J.pullback II.f).obj ((J.pullb …
    IV : ((J.pullback II.f).obj (T I)).Arrow
    IA : B.Arrow := { Y := IV.Y, f := CategoryTheory.CategoryStruct.comp (Category …
    IB : S.Arrow := IA.fromMiddle
    IC : (T IB).Arrow := IA.toMiddle
    ID : (T I).Arrow := { Y := IV.Y, f := CategoryTheory.CategoryStruct.comp IV.f  …
    ⊢ Eq (((J.plusObj P).map IC.f.op) (↑s IB)) (((J.plusObj P).map ID.f.op) (↑s I))
  -/
  let IR : S.Relation := Cover.Relation.mk { hf := IB.hf } { hf := I.hf } { w := IA.middle_spec }
  /-
    case h.h.a
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{max v u, w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite (J …
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    s : CategoryTheory.Meq (J.plusObj P) S
    inj : ∀ (X : C), Function.Injective ⇑((J.toPlus P).app { unop := X })
    T : (I : S.Arrow) → J.Cover I.Y
    t : (I : S.Arrow) → CategoryTheory.Meq P (T I)
    ht : ∀ (I : S.Arrow), Eq (↑s I) (CategoryTheory.GrothendieckTopology.Plus.mk ( …
    B : J.Cover X := S.bind T
    Z : B.Arrow → C
    e1 : (I : B.Arrow) → Quiver.Hom I.Y (Z I)
    e2 : (I : B.Arrow) → Quiver.Hom (Z I) X
    he2 : ∀ (I : B.Arrow), (↑S).arrows (e2 I)
    h✝ : ∀ (I : B.Arrow), (fun x x_1 h => ((fun Y f hf => ↑(T { Y := Y, f := f, hf …
    a✝ : ∀ (I : B.Arrow), Eq (CategoryTheory.CategoryStruct.comp (e1 I) (e2 I)) I.f
    w : CategoryTheory.Meq P B := CategoryTheory.GrothendieckTopology.Plus.meqOfSe …
    I : S.Arrow
    II : (T I).Arrow
    e0 : Quiver.Hom ((J.pullback II.f).obj (T I)) ((J.pullback II.f).obj ((J.pullb …
    IV : ((J.pullback II.f).obj (T I)).Arrow
    IA : B.Arrow := { Y := IV.Y, f := CategoryTheory.CategoryStruct.comp (Category …
    IB : S.Arrow := IA.fromMiddle
    IC : (T IB).Arrow := IA.toMiddle
    ID : (T I).Arrow := { Y := IV.Y, f := CategoryTheory.CategoryStruct.comp IV.f  …
    IR : S.Relation := { fst := { Y := IB.Y, f := IB.f, hf := ⋯ }, snd := { Y := I …
    ⊢ Eq (((J.plusObj P).map IC.f.op) (↑s IB)) (((J.plusObj P).map ID.f.op) (↑s I))
  -/
  exact s.condition IR
  /-
    🎉 no goals
  -/


/-- If `P` is separated, then `P⁺` is a sheaf. -/
theorem isSheaf_of_sep (P : Cᵒᵖ ⥤ D)
    (hsep :
      ∀ (X : C) (S : J.Cover X) (x y : P.obj (op X)),
        (∀ I : S.Arrow, P.map I.f.op x = P.map I.f.op y) → x = y) :
    Presheaf.IsSheaf J (J.plusObj P) := by
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁶ : CategoryTheory.Category.{max v u, w} D
    inst✝⁵ : CategoryTheory.ConcreteCategory D
    inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    ⊢ CategoryTheory.Presheaf.IsSheaf J (J.plusObj P)
  -/
  rw [Presheaf.isSheaf_iff_multiequalizer]
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁶ : CategoryTheory.Category.{max v u, w} D
    inst✝⁵ : CategoryTheory.ConcreteCategory D
    inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    ⊢ ∀ (X : C) (S : J.Cover X), CategoryTheory.IsIso (S.toMultiequalizer (J.plusO …
  -/
  intro X S
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁶ : CategoryTheory.Category.{max v u, w} D
    inst✝⁵ : CategoryTheory.ConcreteCategory D
    inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    ⊢ CategoryTheory.IsIso (S.toMultiequalizer (J.plusObj P))
  -/
  apply @isIso_of_reflects_iso _ _ _ _ _ _ _ (forget D) ?_
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁶ : CategoryTheory.Category.{max v u, w} D
    inst✝⁵ : CategoryTheory.ConcreteCategory D
    inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    ⊢ CategoryTheory.IsIso ((CategoryTheory.forget D).map (S.toMultiequalizer (J.p …
  -/
  rw [isIso_iff_bijective]
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁶ : CategoryTheory.Category.{max v u, w} D
    inst✝⁵ : CategoryTheory.ConcreteCategory D
    inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
    P : CategoryTheory.Functor (Opposite C) D
    hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
    X : C
    S : J.Cover X
    ⊢ Function.Bijective ((CategoryTheory.forget D).map (S.toMultiequalizer (J.plu …
  -/
  constructor
    /-
      case left
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁶ : CategoryTheory.Category.{max v u, w} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      ⊢ Function.Injective ((CategoryTheory.forget D).map (S.toMultiequalizer (J.plu …
    -/
  · intro x y h
    /-
      case left
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁶ : CategoryTheory.Category.{max v u, w} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      x y : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
      h : Eq ((CategoryTheory.forget D).map (S.toMultiequalizer (J.plusObj P)) x) (( …
      ⊢ Eq x y
    -/
    apply sep P S _ _
    /-
      case left
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁶ : CategoryTheory.Category.{max v u, w} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      x y : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
      h : Eq ((CategoryTheory.forget D).map (S.toMultiequalizer (J.plusObj P)) x) (( …
      ⊢ ∀ (I : S.Arrow), Eq (((J.plusObj P).map I.f.op) x) (((J.plusObj P).map I.f.o …
    -/
    intro I
    /-
      case left
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁶ : CategoryTheory.Category.{max v u, w} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      x y : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
      h : Eq ((CategoryTheory.forget D).map (S.toMultiequalizer (J.plusObj P)) x) (( …
      I : S.Arrow
      ⊢ Eq (((J.plusObj P).map I.f.op) x) (((J.plusObj P).map I.f.op) y)
    -/
    apply_fun Meq.equiv _ _ at h
    /-
      case left
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁶ : CategoryTheory.Category.{max v u, w} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      x y : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
      I : S.Arrow
      h : Eq ((CategoryTheory.Meq.equiv (J.plusObj P) S) ((CategoryTheory.forget D). …
      ⊢ Eq (((J.plusObj P).map I.f.op) x) (((J.plusObj P).map I.f.op) y)
    -/
    apply_fun fun e => e I at h
    /-
      case left
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁶ : CategoryTheory.Category.{max v u, w} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      x y : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
      I : S.Arrow
      h : Eq (↑((CategoryTheory.Meq.equiv (J.plusObj P) S) ((CategoryTheory.forget D …
      ⊢ Eq (((J.plusObj P).map I.f.op) x) (((J.plusObj P).map I.f.op) y)
    -/
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
    convert h <;> erw [Meq.equiv_apply, ← comp_apply, Multiequalizer.lift_ι] <;> rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
    /-
      case right
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁶ : CategoryTheory.Category.{max v u, w} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      ⊢ Function.Surjective ((CategoryTheory.forget D).map (S.toMultiequalizer (J.pl …
    -/
  · rintro (x : (multiequalizer (S.index _) : D))
    /-
      case right
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁶ : CategoryTheory.Category.{max v u, w} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      x : (CategoryTheory.forget D).obj (CategoryTheory.Limits.multiequalizer (S.ind …
      ⊢ Exists fun a => Eq ((CategoryTheory.forget D).map (S.toMultiequalizer (J.plu …
    -/
    obtain ⟨t, ht⟩ := exists_of_sep P hsep X S (Meq.equiv _ _ x)
    /-
      case right.intro
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁶ : CategoryTheory.Category.{max v u, w} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      x : (CategoryTheory.forget D).obj (CategoryTheory.Limits.multiequalizer (S.ind …
      t : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
      ht : Eq (CategoryTheory.Meq.mk S t) ((CategoryTheory.Meq.equiv (J.plusObj P) S …
      ⊢ Exists fun a => Eq ((CategoryTheory.forget D).map (S.toMultiequalizer (J.plu …
    -/
    use t
    /-
      case h
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁶ : CategoryTheory.Category.{max v u, w} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      x : (CategoryTheory.forget D).obj (CategoryTheory.Limits.multiequalizer (S.ind …
      t : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
      ht : Eq (CategoryTheory.Meq.mk S t) ((CategoryTheory.Meq.equiv (J.plusObj P) S …
      ⊢ Eq ((CategoryTheory.forget D).map (S.toMultiequalizer (J.plusObj P)) t) x
    -/
    apply (Meq.equiv _ _).injective
    /-
      case h.a
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁶ : CategoryTheory.Category.{max v u, w} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      x : (CategoryTheory.forget D).obj (CategoryTheory.Limits.multiequalizer (S.ind …
      t : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
      ht : Eq (CategoryTheory.Meq.mk S t) ((CategoryTheory.Meq.equiv (J.plusObj P) S …
      ⊢ Eq ((CategoryTheory.Meq.equiv (J.plusObj P) S) ((CategoryTheory.forget D).ma …
    -/
    rw [← ht]
    /-
      case h.a
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁶ : CategoryTheory.Category.{max v u, w} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      x : (CategoryTheory.forget D).obj (CategoryTheory.Limits.multiequalizer (S.ind …
      t : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
      ht : Eq (CategoryTheory.Meq.mk S t) ((CategoryTheory.Meq.equiv (J.plusObj P) S …
      ⊢ Eq ((CategoryTheory.Meq.equiv (J.plusObj P) S) ((CategoryTheory.forget D).ma …
    -/
    ext i
    /-
      case h.a.h
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁶ : CategoryTheory.Category.{max v u, w} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      x : (CategoryTheory.forget D).obj (CategoryTheory.Limits.multiequalizer (S.ind …
      t : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
      ht : Eq (CategoryTheory.Meq.mk S t) ((CategoryTheory.Meq.equiv (J.plusObj P) S …
      i : S.Arrow
      ⊢ Eq (↑((CategoryTheory.Meq.equiv (J.plusObj P) S) ((CategoryTheory.forget D). …
    -/
    dsimp
    /-
      case h.a.h
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁶ : CategoryTheory.Category.{max v u, w} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      x : (CategoryTheory.forget D).obj (CategoryTheory.Limits.multiequalizer (S.ind …
      t : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
      ht : Eq (CategoryTheory.Meq.mk S t) ((CategoryTheory.Meq.equiv (J.plusObj P) S …
      i : S.Arrow
      ⊢ Eq ((CategoryTheory.Limits.Multiequalizer.ι (S.index (J.plusObj P)) i) ((Cat …
    -/
    erw [← comp_apply]
    /-
      case h.a.h
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁶ : CategoryTheory.Category.{max v u, w} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      x : (CategoryTheory.forget D).obj (CategoryTheory.Limits.multiequalizer (S.ind …
      t : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
      ht : Eq (CategoryTheory.Meq.mk S t) ((CategoryTheory.Meq.equiv (J.plusObj P) S …
      i : S.Arrow
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (S.toMultiequalizer (J.plusObj P)) ( …
    -/
    rw [Multiequalizer.lift_ι]
    /-
      case h.a.h
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁶ : CategoryTheory.Category.{max v u, w} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
      P : CategoryTheory.Functor (Opposite C) D
      hsep : ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj (P.obj { …
      X : C
      S : J.Cover X
      x : (CategoryTheory.forget D).obj (CategoryTheory.Limits.multiequalizer (S.ind …
      t : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
      ht : Eq (CategoryTheory.Meq.mk S t) ((CategoryTheory.Meq.equiv (J.plusObj P) S …
      i : S.Arrow
      ⊢ Eq (((J.plusObj P).map i.f.op) t) (↑(CategoryTheory.Meq.mk S t) i)
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- `P⁺⁺` is always a sheaf. -/
theorem isSheaf_plus_plus (P : Cᵒᵖ ⥤ D) : Presheaf.IsSheaf J (J.plusObj (J.plusObj P)) := by
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁶ : CategoryTheory.Category.{max v u, w} D
    inst✝⁵ : CategoryTheory.ConcreteCategory D
    inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
    P : CategoryTheory.Functor (Opposite C) D
    ⊢ CategoryTheory.Presheaf.IsSheaf J (J.plusObj (J.plusObj P))
  -/
  apply isSheaf_of_sep
  /-
    case hsep
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁶ : CategoryTheory.Category.{max v u, w} D
    inst✝⁵ : CategoryTheory.ConcreteCategory D
    inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
    P : CategoryTheory.Functor (Opposite C) D
    ⊢ ∀ (X : C) (S : J.Cover X) (x y : (CategoryTheory.forget D).obj ((J.plusObj P …
  -/
  intro X S x y
  /-
    case hsep
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝⁶ : CategoryTheory.Category.{max v u, w} D
    inst✝⁵ : CategoryTheory.ConcreteCategory D
    inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
    inst✝² : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
    P : CategoryTheory.Functor (Opposite C) D
    X : C
    S : J.Cover X
    x y : (CategoryTheory.forget D).obj ((J.plusObj P).obj { unop := X })
    ⊢ (∀ (I : S.Arrow), Eq (((J.plusObj P).map I.f.op) x) (((J.plusObj P).map I.f. …
  -/
  apply sep
  /-
    🎉 no goals
  -/


/-- The sheafification of a presheaf `P`.
*NOTE:* Additional hypotheses are needed to obtain a proof that this is a sheaf! -/
noncomputable def sheafify (P : Cᵒᵖ ⥤ D) : Cᵒᵖ ⥤ D :=
  J.plusObj (J.plusObj P)


/-- The canonical map from `P` to its sheafification. -/
noncomputable def toSheafify (P : Cᵒᵖ ⥤ D) : P ⟶ J.sheafify P :=
  J.toPlus P ≫ J.plusMap (J.toPlus P)


/-- The canonical map on sheafifications induced by a morphism. -/
noncomputable def sheafifyMap {P Q : Cᵒᵖ ⥤ D} (η : P ⟶ Q) : J.sheafify P ⟶ J.sheafify Q :=
  J.plusMap <| J.plusMap η


@[simp]
theorem sheafifyMap_id (P : Cᵒᵖ ⥤ D) : J.sheafifyMap (𝟙 P) = 𝟙 (J.sheafify P) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P : CategoryTheory.Functor (Opposite C) D
    ⊢ Eq (J.sheafifyMap (CategoryTheory.CategoryStruct.id P)) (CategoryTheory.Cate …
  -/
  dsimp [sheafifyMap, sheafify]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P : CategoryTheory.Functor (Opposite C) D
    ⊢ Eq (J.plusMap (J.plusMap (CategoryTheory.CategoryStruct.id P))) (CategoryThe …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem sheafifyMap_comp {P Q R : Cᵒᵖ ⥤ D} (η : P ⟶ Q) (γ : Q ⟶ R) :
    J.sheafifyMap (η ≫ γ) = J.sheafifyMap η ≫ J.sheafifyMap γ := by
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
    ⊢ Eq (J.sheafifyMap (CategoryTheory.CategoryStruct.comp η γ)) (CategoryTheory. …
  -/
  dsimp [sheafifyMap, sheafify]
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
    ⊢ Eq (J.plusMap (J.plusMap (CategoryTheory.CategoryStruct.comp η γ))) (Categor …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem toSheafify_naturality {P Q : Cᵒᵖ ⥤ D} (η : P ⟶ Q) :
    η ≫ J.toSheafify _ = J.toSheafify _ ≫ J.sheafifyMap η := by
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
    ⊢ Eq (CategoryTheory.CategoryStruct.comp η (J.toSheafify Q)) (CategoryTheory.C …
  -/
  dsimp [sheafifyMap, sheafify, toSheafify]
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
    ⊢ Eq (CategoryTheory.CategoryStruct.comp η (CategoryTheory.CategoryStruct.comp …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The sheafification of a presheaf `P`, as a functor.
*NOTE:* Additional hypotheses are needed to obtain a proof that this is a sheaf! -/
noncomputable def sheafification : (Cᵒᵖ ⥤ D) ⥤ Cᵒᵖ ⥤ D :=
  J.plusFunctor D ⋙ J.plusFunctor D


@[simp]
theorem sheafification_obj (P : Cᵒᵖ ⥤ D) : (J.sheafification D).obj P = J.sheafify P :=
  rfl


@[simp]
theorem sheafification_map {P Q : Cᵒᵖ ⥤ D} (η : P ⟶ Q) :
    (J.sheafification D).map η = J.sheafifyMap η :=
  rfl


/-- The canonical map from `P` to its sheafification, as a natural transformation.
*Note:* We only show this is a sheaf under additional hypotheses on `D`. -/
noncomputable def toSheafification : 𝟭 _ ⟶ sheafification J D :=
  J.toPlusNatTrans D ≫ whiskerRight (J.toPlusNatTrans D) (J.plusFunctor D)


@[simp]
theorem toSheafification_app (P : Cᵒᵖ ⥤ D) :
    (J.toSheafification D).app P = J.toSheafify P :=
  rfl


theorem isIso_toSheafify {P : Cᵒᵖ ⥤ D} (hP : Presheaf.IsSheaf J P) : IsIso (J.toSheafify P) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P : CategoryTheory.Functor (Opposite C) D
    hP : CategoryTheory.Presheaf.IsSheaf J P
    ⊢ CategoryTheory.IsIso (J.toSheafify P)
  -/
  dsimp [toSheafify]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P : CategoryTheory.Functor (Opposite C) D
    hP : CategoryTheory.Presheaf.IsSheaf J P
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (J.toPlus P) (J.plu …
  -/
  haveI := isIso_toPlus_of_isSheaf J P hP
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P : CategoryTheory.Functor (Opposite C) D
    hP : CategoryTheory.Presheaf.IsSheaf J P
    this : CategoryTheory.IsIso (J.toPlus P)
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (J.toPlus P) (J.plu …
  -/
  change (IsIso (toPlus J P ≫ (J.plusFunctor D).map (toPlus J P)))
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P : CategoryTheory.Functor (Opposite C) D
    hP : CategoryTheory.Presheaf.IsSheaf J P
    this : CategoryTheory.IsIso (J.toPlus P)
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (J.toPlus P) ((J.pl …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If `P` is a sheaf, then `P` is isomorphic to `J.sheafify P`. -/
noncomputable def isoSheafify {P : Cᵒᵖ ⥤ D} (hP : Presheaf.IsSheaf J P) : P ≅ J.sheafify P :=
  letI := isIso_toSheafify J hP
  asIso (J.toSheafify P)


@[simp]
theorem isoSheafify_hom {P : Cᵒᵖ ⥤ D} (hP : Presheaf.IsSheaf J P) :
    (J.isoSheafify hP).hom = J.toSheafify P :=
  rfl


/-- Given a sheaf `Q` and a morphism `P ⟶ Q`, construct a morphism from `J.sheafify P` to `Q`. -/
noncomputable def sheafifyLift {P Q : Cᵒᵖ ⥤ D} (η : P ⟶ Q) (hQ : Presheaf.IsSheaf J Q) :
    J.sheafify P ⟶ Q :=
  J.plusLift (J.plusLift η hQ) hQ


@[reassoc (attr := simp)]
theorem toSheafify_sheafifyLift {P Q : Cᵒᵖ ⥤ D} (η : P ⟶ Q) (hQ : Presheaf.IsSheaf J Q) :
    J.toSheafify P ≫ sheafifyLift J η hQ = η := by
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
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toSheafify P) (J.sheafifyLift η hQ …
  -/
  dsimp only [sheafifyLift, toSheafify]
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
  simp
  /-
    🎉 no goals
  -/


theorem sheafifyLift_unique {P Q : Cᵒᵖ ⥤ D} (η : P ⟶ Q) (hQ : Presheaf.IsSheaf J Q)
    (γ : J.sheafify P ⟶ Q) : J.toSheafify P ≫ γ = η → γ = sheafifyLift J η hQ := by
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
    γ : Quiver.Hom (J.sheafify P) Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toSheafify P) γ) η → Eq γ (J.sheaf …
  -/
  intro h
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
    γ : Quiver.Hom (J.sheafify P) Q
    h : Eq (CategoryTheory.CategoryStruct.comp (J.toSheafify P) γ) η
    ⊢ Eq γ (J.sheafifyLift η hQ)
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
    η : Quiver.Hom P Q
    hQ : CategoryTheory.Presheaf.IsSheaf J Q
    γ : Quiver.Hom (J.sheafify P) Q
    h : Eq (CategoryTheory.CategoryStruct.comp (J.toSheafify P) γ) η
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toPlus (J.plusObj P)) γ) (J.plusLi …
  -/
  apply plusLift_unique
  /-
    case hγ.hγ
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
    γ : Quiver.Hom (J.sheafify P) Q
    h : Eq (CategoryTheory.CategoryStruct.comp (J.toSheafify P) γ) η
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toPlus P) (CategoryTheory.Category …
  -/
  rw [← Category.assoc, ← plusMap_toPlus]
  /-
    case hγ.hγ
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
    γ : Quiver.Hom (J.sheafify P) Q
    h : Eq (CategoryTheory.CategoryStruct.comp (J.toSheafify P) γ) η
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  exact h
  /-
    🎉 no goals
  -/


@[simp]
theorem isoSheafify_inv {P : Cᵒᵖ ⥤ D} (hP : Presheaf.IsSheaf J P) :
    (J.isoSheafify hP).inv = J.sheafifyLift (𝟙 _) hP := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P : CategoryTheory.Functor (Opposite C) D
    hP : CategoryTheory.Presheaf.IsSheaf J P
    ⊢ Eq (J.isoSheafify hP).inv (J.sheafifyLift (CategoryTheory.CategoryStruct.id  …
  -/
  apply J.sheafifyLift_unique
  /-
    case a
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P : CategoryTheory.Functor (Opposite C) D
    hP : CategoryTheory.Presheaf.IsSheaf J P
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toSheafify P) (J.isoSheafify hP).i …
  -/
  simp [Iso.comp_inv_eq]
  /-
    🎉 no goals
  -/


theorem sheafify_hom_ext {P Q : Cᵒᵖ ⥤ D} (η γ : J.sheafify P ⟶ Q) (hQ : Presheaf.IsSheaf J Q)
    (h : J.toSheafify P ≫ η = J.toSheafify P ≫ γ) : η = γ := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η γ : Quiver.Hom (J.sheafify P) Q
    hQ : CategoryTheory.Presheaf.IsSheaf J Q
    h : Eq (CategoryTheory.CategoryStruct.comp (J.toSheafify P) η) (CategoryTheory …
    ⊢ Eq η γ
  -/
  apply J.plus_hom_ext _ _ hQ
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η γ : Quiver.Hom (J.sheafify P) Q
    hQ : CategoryTheory.Presheaf.IsSheaf J Q
    h : Eq (CategoryTheory.CategoryStruct.comp (J.toSheafify P) η) (CategoryTheory …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toPlus (J.plusObj P)) η) (Category …
  -/
  apply J.plus_hom_ext _ _ hQ
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η γ : Quiver.Hom (J.sheafify P) Q
    hQ : CategoryTheory.Presheaf.IsSheaf J Q
    h : Eq (CategoryTheory.CategoryStruct.comp (J.toSheafify P) η) (CategoryTheory …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toPlus P) (CategoryTheory.Category …
  -/
  rw [← Category.assoc, ← Category.assoc, ← plusMap_toPlus]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type w
    inst✝² : CategoryTheory.Category.{max v u, w} D
    inst✝¹ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cover …
    P Q : CategoryTheory.Functor (Opposite C) D
    η γ : Quiver.Hom (J.sheafify P) Q
    hQ : CategoryTheory.Presheaf.IsSheaf J Q
    h : Eq (CategoryTheory.CategoryStruct.comp (J.toSheafify P) η) (CategoryTheory …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  exact h
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem sheafifyMap_sheafifyLift {P Q R : Cᵒᵖ ⥤ D} (η : P ⟶ Q) (γ : Q ⟶ R)
    (hR : Presheaf.IsSheaf J R) :
    J.sheafifyMap η ≫ J.sheafifyLift γ hR = J.sheafifyLift (η ≫ γ) hR := by
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
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.sheafifyMap η) (J.sheafifyLift γ h …
  -/
  apply J.sheafifyLift_unique
  /-
    case a
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
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toSheafify P) (CategoryTheory.Cate …
  -/
  rw [← Category.assoc, ← J.toSheafify_naturality, Category.assoc, toSheafify_sheafifyLift]
  /-
    🎉 no goals
  -/


theorem GrothendieckTopology.sheafify_isSheaf (P : Cᵒᵖ ⥤ D) : Presheaf.IsSheaf J (J.sheafify P) :=
  GrothendieckTopology.Plus.isSheaf_plus_plus _ _


/-- The sheafification functor, as a functor taking values in `Sheaf`. -/
@[simps]
noncomputable def plusPlusSheaf : (Cᵒᵖ ⥤ D) ⥤ Sheaf J D where
  obj P := ⟨J.sheafify P, J.sheafify_isSheaf P⟩
  map η := ⟨J.sheafifyMap η⟩
  map_id _ := Sheaf.Hom.ext <| J.sheafifyMap_id _
  map_comp _ _ := Sheaf.Hom.ext <| J.sheafifyMap_comp _ _


instance plusPlusSheaf_preservesZeroMorphisms [Preadditive D] :
    (plusPlusSheaf J D).PreservesZeroMorphisms where
  map_zero F G := by
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁷ : CategoryTheory.Category.{max v u, w} D
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝⁴ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝¹ : (CategoryTheory.forget D).ReflectsIsomorphisms
      inst✝ : CategoryTheory.Preadditive D
      F G : CategoryTheory.Functor (Opposite C) D
      ⊢ Eq ((CategoryTheory.plusPlusSheaf J D).map 0) 0
    -/
    ext : 3
    /-
      case h.w.h
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁷ : CategoryTheory.Category.{max v u, w} D
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝⁴ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝¹ : (CategoryTheory.forget D).ReflectsIsomorphisms
      inst✝ : CategoryTheory.Preadditive D
      F G : CategoryTheory.Functor (Opposite C) D
      x✝ : Opposite C
      ⊢ Eq (((CategoryTheory.plusPlusSheaf J D).map 0).val.app x✝) ((CategoryTheory. …
    -/
    refine colimit.hom_ext (fun j => ?_)
    /-
      case h.w.h
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type w
      inst✝⁷ : CategoryTheory.Category.{max v u, w} D
      inst✝⁶ : CategoryTheory.ConcreteCategory D
      inst✝⁵ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
      inst✝⁴ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
      inst✝³ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
      inst✝² : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
      inst✝¹ : (CategoryTheory.forget D).ReflectsIsomorphisms
      inst✝ : CategoryTheory.Preadditive D
      F G : CategoryTheory.Functor (Opposite C) D
      x✝ : Opposite C
      j : Opposite (J.Cover (Opposite.unop x✝))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (J.d …
    -/
    erw [colimit.ι_map, comp_zero, J.plusMap_zero, J.diagramNatTrans_zero, zero_comp]
    /-
      🎉 no goals
    -/


/-- The sheafification functor is left adjoint to the forgetful functor. -/
@[simps! unit_app counit_app_val]
noncomputable def plusPlusAdjunction : plusPlusSheaf J D ⊣ sheafToPresheaf J D :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun P Q =>
        { toFun := fun e => J.toSheafify P ≫ e.val
          invFun := fun e => ⟨J.sheafifyLift e Q.2⟩
          left_inv := fun _ => Sheaf.Hom.ext <| (J.sheafifyLift_unique _ _ _ rfl).symm
          right_inv := fun _ => J.toSheafify_sheafifyLift _ _ }
      homEquiv_naturality_left_symm := by
        /-
          C : Type u
          inst✝⁷ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝⁶ : CategoryTheory.Category.{max v u, w} D
          inst✝⁵ : CategoryTheory.ConcreteCategory D
          inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
          inst✝³ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
          inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
          inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
          ⊢ ∀ {X' X : CategoryTheory.Functor (Opposite C) D} {Y : CategoryTheory.Sheaf J …
        -/
        intro P Q R η γ; ext1; dsimp; symm
        /-
          case h
          C : Type u
          inst✝⁷ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝⁶ : CategoryTheory.Category.{max v u, w} D
          inst✝⁵ : CategoryTheory.ConcreteCategory D
          inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
          inst✝³ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
          inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
          inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
          P Q : CategoryTheory.Functor (Opposite C) D
          R : CategoryTheory.Sheaf J D
          η : Quiver.Hom P Q
          γ : Quiver.Hom Q ((CategoryTheory.sheafToPresheaf J D).obj R)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.sheafifyMap η) (J.sheafifyLift γ ⋯ …
        -/
        apply J.sheafifyMap_sheafifyLift
        /-
          🎉 no goals
        -/
      homEquiv_naturality_right := fun η γ => by
        /-
          C : Type u
          inst✝⁷ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝⁶ : CategoryTheory.Category.{max v u, w} D
          inst✝⁵ : CategoryTheory.ConcreteCategory D
          inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
          inst✝³ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
          inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
          inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
          X✝ : CategoryTheory.Functor (Opposite C) D
          Y✝ Y'✝ : CategoryTheory.Sheaf J D
          η : Quiver.Hom ((CategoryTheory.plusPlusSheaf J D).obj X✝) Y✝
          γ : Quiver.Hom Y✝ Y'✝
          ⊢ Eq (((fun P Q => { toFun := fun e => CategoryTheory.CategoryStruct.comp (J.t …
        -/
        dsimp
        /-
          C : Type u
          inst✝⁷ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type w
          inst✝⁶ : CategoryTheory.Category.{max v u, w} D
          inst✝⁵ : CategoryTheory.ConcreteCategory D
          inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
          inst✝³ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
          inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
          inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
          X✝ : CategoryTheory.Functor (Opposite C) D
          Y✝ Y'✝ : CategoryTheory.Sheaf J D
          η : Quiver.Hom ((CategoryTheory.plusPlusSheaf J D).obj X✝) Y✝
          γ : Quiver.Hom Y✝ Y'✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.toSheafify X✝) (CategoryTheory.Cat …
        -/
        rw [Category.assoc] }
        /-
          🎉 no goals
        -/


instance sheafToPresheaf_isRightAdjoint : (sheafToPresheaf J D).IsRightAdjoint  :=
  (plusPlusAdjunction J D).isRightAdjoint


instance presheaf_mono_of_mono {F G : Sheaf J D} (f : F ⟶ G) [Mono f] : Mono f.1 :=
  (sheafToPresheaf J D).map_mono _


theorem Sheaf.Hom.mono_iff_presheaf_mono {F G : Sheaf J D} (f : F ⟶ G) : Mono f ↔ Mono f.1 :=
               /-
                 C : Type u
                 inst✝⁷ : CategoryTheory.Category.{v, u} C
                 J : CategoryTheory.GrothendieckTopology C
                 D : Type w
                 inst✝⁶ : CategoryTheory.Category.{max v u, w} D
                 inst✝⁵ : CategoryTheory.ConcreteCategory D
                 inst✝⁴ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
                 inst✝³ : ∀ (P : CategoryTheory.Functor (Opposite C) D) (X : C) (S : J.Cover X) …
                 inst✝² : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cove …
                 inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
                 inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
                 F G : CategoryTheory.Sheaf J D
                 f : Quiver.Hom F G
                 m : CategoryTheory.Mono f
                 ⊢ CategoryTheory.Mono f.val
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun m => by infer_instance, fun m => by exact Sheaf.Hom.mono_of_presheaf_mono J D f⟩
                                           /-
                                             🎉 no goals
                                           -/


